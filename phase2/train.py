"""Phase 3: Training script for the conditioned model.

Usage:
  python -m phase2.train                          # full training
  python -m phase2.train --smoke-test              # quick CPU test
  python -m phase2.train --eval-only --checkpoint best.pkl  # eval only

Trains the conditioned model on all training variants, periodically
evaluating on both training variants (to check learning) and held-out
test variants (to check compositional generalisation).
"""

import argparse
import functools
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import clrs
import jax
import jax.numpy as jnp
import numpy as np

from phase0.variant_registry import TRAIN_VARIANTS, TEST_VARIANTS, BASE_VARIANTS
from phase1.data_pipeline import ConditionedDataPipeline, EvalPipeline
from phase1.spec import RELAXATION_SPEC
from phase2.model import ConditionedModel

try:
    import wandb
except ImportError:
    wandb = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data.
    n: int = 16                     # graph size
    k: int = 5                      # conditioning examples (eval and default train)
    batch_size: int = 32            # query batch size
    p: float = 0.3                  # ER edge probability
    weight_range: tuple = (0.1, 1.0)
    randomize_k: bool = True        # vary k during training to prevent z memorisation
    k_range: tuple = (1, 2, 3, 5, 8)  # k values to sample from when randomize_k=True

    # Model.
    hidden_dim: int = 128
    z_dim: int = 4                # 4 dims for 4 compositional axes; forces factored encoding
    cond_hidden_dim: int = 64
    cond_nb_layers: int = 2
    processor_type: str = 'triplet_gmpnn'
    use_ln: bool = True
    nb_triplet_fts: int = 8
    nb_heads: int = 1
    nb_msg_passing_steps: int = 1
    use_lstm: bool = False
    encode_hints: bool = True
    decode_hints: bool = True
    encoder_init: str = 'xavier_on_scalars'
    dropout_prob: float = 0.0
    hint_teacher_forcing: float = 0.5
    hint_repred_mode: str = 'soft'

    # Training.
    train_steps: int = 10000
    learning_rate: float = 0.001
    grad_clip_max_norm: float = 1.0
    seed: int = 42

    # Per-example conditioning: each query gets its own k conditioning examples.
    per_example_conditioning: bool = True

    # Episodic training: hold out variants to incentivise compositional z.
    episodic: bool = False          # disabled: z bottleneck + per-example should suffice
    episode_length: int = 500        # steps per episode
    episode_train_frac: float = 0.8  # fraction for training on active variants
    episode_holdout: int = 3         # variants held out per episode

    # Evaluation.
    eval_every: int = 250
    eval_samples: int = 128
    eval_batch_size: int = 32
    eval_k: int = 5

    # Logging / checkpointing.
    checkpoint_dir: str = 'checkpoints/conditioned'
    wandb_enabled: bool = True
    wandb_project: str = 'compositional_algorithmic_generalisation'
    wandb_entity: str = 'shanai'
    name: str = 'conditioned_v1'


SMOKE_CONFIG = TrainConfig(
    n=8, k=3, batch_size=4, hidden_dim=32, z_dim=4,
    cond_hidden_dim=16, cond_nb_layers=2,
    nb_triplet_fts=4, train_steps=100,
    eval_every=50, eval_samples=8, eval_batch_size=4,
    randomize_k=False, per_example_conditioning=True,
    episodic=False,
    wandb_enabled=False, name='smoke_test',
    checkpoint_dir='checkpoints/smoke',
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_variant(model, eval_pipeline, variant_name, rng_key, k):
    """Evaluate model on a single variant. Returns metrics dict."""
    n_correct_pi = 0
    n_total_pi = 0
    d_mse_sum = 0.0
    d_mse_count = 0

    for batch in eval_pipeline.eval_batches(variant_name):
        rng_key, sub_key = jax.random.split(rng_key)
        output_preds, _ = model.predict(
            sub_key, batch.query.features, batch.conditioning.features)

        # Predecessor accuracy: argmax of logits vs ground truth.
        pi_pred = output_preds['pi']  # (B, n, n) logits
        pi_pred_idx = np.argmax(pi_pred, axis=-1)  # (B, n)
        pi_true = batch.query.outputs[0].data  # (B, n)

        # Exclude source node (always self-pointing, trivially correct).
        source_mask = np.array(
            [dp.data for dp in batch.query.features.inputs
             if dp.name == 's'][0])  # (B, n) one-hot
        non_source = 1.0 - source_mask  # (B, n)

        correct = (pi_pred_idx == pi_true.astype(int)) * non_source
        n_correct_pi += correct.sum()
        n_total_pi += non_source.sum()

        # Distance MSE from last hint step.
        d_hints = [h for h in batch.query.features.hints if h.name == 'd'][0]
        d_true_final = d_hints.data[-1]  # (B, n) — last hint step
        # Only compute MSE on nodes with reasonable d values (not INF_PROXY).
        finite_mask = (np.abs(d_true_final) < 1e3) * non_source
        if finite_mask.sum() > 0:
            # We don't have direct d output prediction — d is a hint.
            # Skip d_mse for now; it requires hint predictions which
            # we don't return in eval mode.
            pass

    pred_acc = float(n_correct_pi / max(n_total_pi, 1))
    return {'pred_accuracy': pred_acc}


def run_evaluation(model, config, rng_key, variant_names, label, step=None):
    """Evaluate on a set of variants and log results."""
    eval_pipe = EvalPipeline(
        variant_names=variant_names, n=config.n, k=config.eval_k,
        num_eval_samples=config.eval_samples,
        eval_batch_size=config.eval_batch_size)

    results = {}
    for name in variant_names:
        rng_key, sub_key = jax.random.split(rng_key)
        metrics = evaluate_variant(model, eval_pipe, name, sub_key, config.eval_k)
        results[name] = metrics

        if config.wandb_enabled and wandb is not None and step is not None:
            for k, v in metrics.items():
                wandb.log({f'{label}/{name}/{k}': v, 'step': step}, step=step)

    # Summary.
    accs = [m['pred_accuracy'] for m in results.values()]
    mean_acc = np.mean(accs) if accs else 0.0
    if config.wandb_enabled and wandb is not None and step is not None:
        wandb.log({f'{label}/mean_pred_accuracy': mean_acc, 'step': step},
                  step=step)

    return results, mean_acc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: TrainConfig):
    print(f'=== Training: {config.name} ===')
    print(f'  Variants: {len(TRAIN_VARIANTS)} train, {len(TEST_VARIANTS)} test')
    print(f'  Graph: n={config.n}, k={config.k}, batch={config.batch_size}')
    print(f'  Model: hidden={config.hidden_dim}, z={config.z_dim}, '
          f'processor={config.processor_type}')
    print(f'  Steps: {config.train_steps}, lr={config.learning_rate}')

    # Wandb.
    if config.wandb_enabled and wandb is not None:
        wandb.init(project=config.wandb_project, entity=config.wandb_entity,
                   name=config.name, config=vars(config))

    # Data pipeline.
    pipeline = ConditionedDataPipeline(
        variant_names=TRAIN_VARIANTS, n=config.n, k=config.k,
        batch_size=config.batch_size, p=config.p,
        weight_range=config.weight_range, seed=config.seed,
        randomize_k=config.randomize_k, k_range=config.k_range,
        per_example_conditioning=config.per_example_conditioning)

    # First batch for model init.
    dummy_batch = pipeline.next()

    # Model.
    processor_factory = clrs.get_processor_factory(
        config.processor_type, use_ln=config.use_ln,
        nb_triplet_fts=config.nb_triplet_fts, nb_heads=config.nb_heads)

    model = ConditionedModel(
        spec=RELAXATION_SPEC, dummy_trajectory=dummy_batch.query,
        processor_factory=processor_factory,
        hidden_dim=config.hidden_dim, z_dim=config.z_dim,
        cond_hidden_dim=config.cond_hidden_dim,
        cond_nb_layers=config.cond_nb_layers,
        encode_hints=config.encode_hints, decode_hints=config.decode_hints,
        encoder_init=config.encoder_init, use_lstm=config.use_lstm,
        learning_rate=config.learning_rate,
        grad_clip_max_norm=config.grad_clip_max_norm,
        dropout_prob=config.dropout_prob,
        hint_teacher_forcing=config.hint_teacher_forcing,
        hint_repred_mode=config.hint_repred_mode,
        nb_msg_passing_steps=config.nb_msg_passing_steps,
        checkpoint_path=config.checkpoint_dir,
    )

    print('Initializing model...')
    model.init(dummy_batch.query.features,
               dummy_batch.conditioning.features, seed=config.seed)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(model.params))
    print(f'Parameters: {n_params:,}')

    # Training loop.
    rng = jax.random.PRNGKey(config.seed)
    ep_rng = np.random.RandomState(config.seed + 1)
    best_train_acc = -1.0
    t0 = time.time()

    # Episodic state.
    active_variants = list(TRAIN_VARIANTS)
    held_out_variants = []

    for step in range(1, config.train_steps + 1):
        # Episodic variant management: periodically hold out variants.
        if config.episodic:
            step_in_episode = (step - 1) % config.episode_length
            if step_in_episode == 0:
                # Start new episode: choose held-out variants.
                held_out_variants = list(ep_rng.choice(
                    TRAIN_VARIANTS, config.episode_holdout, replace=False))
                active_variants = [v for v in TRAIN_VARIANTS
                                   if v not in held_out_variants]
                print(f'  [Episode] Held out: {held_out_variants}')

            train_phase_steps = int(
                config.episode_length * config.episode_train_frac)
            if step_in_episode < train_phase_steps:
                # Train phase: sample from active variants only.
                batch = pipeline.next(allowed_variants=active_variants)
            else:
                # Meta-eval phase: sample from held-out variants.
                # Gradient still flows — this pushes the model toward
                # z representations that generalise to unseen variants.
                batch = pipeline.next(allowed_variants=held_out_variants)
        else:
            batch = pipeline.next()

        rng, step_key = jax.random.split(rng)

        # During meta-eval: repred=True forces the model to rely on z alone
        # (no teacher forcing). This is true meta-learning: the model must
        # execute held-out variants from conditioning, not from ground-truth
        # hints. Gradient flows through the conditioning encoder.
        is_meta = (config.episodic and
                   (step - 1) % config.episode_length >= int(
                       config.episode_length * config.episode_train_frac))
        loss = model.feedback(step_key, batch.query,
                              batch.conditioning.features,
                              repred=is_meta)
        loss_val = float(loss)

        if config.wandb_enabled and wandb is not None:
            wandb.log({
                'train/loss': loss_val,
                'train/is_meta_eval': float(is_meta),
                'step': step,
            }, step=step)

        if step % max(1, config.eval_every // 5) == 0:
            elapsed = time.time() - t0
            phase = 'meta' if is_meta else 'train'
            print(f'  step {step}/{config.train_steps}: '
                  f'loss={loss_val:.2f}, variant={batch.variant_name}, '
                  f'phase={phase}, {elapsed:.0f}s')

        # Periodic evaluation.
        if step % config.eval_every == 0 or step == config.train_steps:
            rng, eval_key = jax.random.split(rng)

            # Evaluate on training variants.
            print(f'\n  --- Eval (step {step}) ---')
            train_results, train_mean = run_evaluation(
                model, config, eval_key, TRAIN_VARIANTS,
                'eval_train', step=step)
            print(f'  Train variants mean pred_acc: {train_mean:.3f}')
            for name, m in sorted(train_results.items()):
                print(f'    {name:30s}: {m["pred_accuracy"]:.3f}')

            # Evaluate on test variants.
            rng, eval_key = jax.random.split(rng)
            test_results, test_mean = run_evaluation(
                model, config, eval_key, TEST_VARIANTS,
                'eval_test', step=step)
            print(f'  Test variants mean pred_acc:  {test_mean:.3f}')
            for name, m in sorted(test_results.items()):
                print(f'    {name:30s}: {m["pred_accuracy"]:.3f}')

            # Checkpoint best.
            if train_mean > best_train_acc:
                best_train_acc = train_mean
                model.save_model('best.pkl')
                print(f'  New best model saved (train_acc={best_train_acc:.3f})')

            print()

    # Final eval on best checkpoint.
    print('=== Final evaluation (best checkpoint) ===')
    model.restore_model('best.pkl')
    rng, eval_key = jax.random.split(rng)
    train_results, train_mean = run_evaluation(
        model, config, eval_key, TRAIN_VARIANTS, 'final_train')
    test_results, test_mean = run_evaluation(
        model, config, eval_key, TEST_VARIANTS, 'final_test')

    print(f'\nFinal train mean pred_acc: {train_mean:.3f}')
    print(f'Final test mean pred_acc:  {test_mean:.3f}')

    for name in sorted(test_results.keys()):
        print(f'  TEST {name:30s}: {test_results[name]["pred_accuracy"]:.3f}')

    if config.wandb_enabled and wandb is not None:
        wandb.finish()

    print('\nDone!')
    return train_mean, test_mean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train conditioned model for compositional generalisation.')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Quick CPU smoke test with tiny model.')
    parser.add_argument('--eval-only', action='store_true',
                        help='Run evaluation only (requires --checkpoint).')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint file to restore.')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging.')
    args = parser.parse_args()

    if args.smoke_test:
        config = SMOKE_CONFIG
    else:
        config = TrainConfig()

    if args.no_wandb:
        config.wandb_enabled = False

    train(config)


if __name__ == '__main__':
    main()
