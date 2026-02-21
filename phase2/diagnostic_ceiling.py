"""Diagnostic 3: Parameter vector ceiling.

Trains a SEPARATE model where z is a learnable per-variant embedding
(no conditioning encoder). Each training variant gets its own learned
vector. At test time, test variants also get their own learned vectors,
trained alongside the processor.

This measures: if the processor knows EXACTLY which variant to execute
(via a learned task ID), how well can it do? The gap between this ceiling
and the conditioned model tells us whether the bottleneck is the
conditioning encoder or the processor.

Usage:
  python -m phase2.diagnostic_ceiling                    # full training
  python -m phase2.diagnostic_ceiling --smoke-test       # quick CPU test
"""

import argparse
import os
import time

import clrs
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from phase0.variant_registry import (
    BASE_VARIANTS, TRAIN_VARIANTS, TEST_VARIANTS,
)
from phase1.data_pipeline import ConditionedDataPipeline, EvalPipeline
from phase1.spec import RELAXATION_SPEC
from phase2.model import ConditionedModel
from phase2.train import TrainConfig, SMOKE_CONFIG, evaluate_variant

try:
    import wandb
except ImportError:
    wandb = None


def run_ceiling_diagnostic(config):
    """Train a model with learned per-variant embeddings instead of conditioning."""
    print('=== Diagnostic 3: Parameter Vector Ceiling ===')
    print('  Training a model with learned variant embeddings (no encoder).\n')

    all_train = list(TRAIN_VARIANTS)
    all_test = list(TEST_VARIANTS)
    all_variants = all_train + all_test
    variant_to_idx = {name: i for i, name in enumerate(all_variants)}
    n_variants = len(all_variants)

    if config.wandb_enabled and wandb is not None:
        wandb.init(project=config.wandb_project, entity=config.wandb_entity,
                   name=config.name + '_ceiling', config=vars(config))

    # Data pipeline (we still generate conditioning for shape compatibility,
    # but the model won't use it).
    pipeline = ConditionedDataPipeline(
        variant_names=all_train, n=config.n, k=config.k,
        batch_size=config.batch_size, seed=config.seed)
    dummy_batch = pipeline.next()

    # Build model.
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
        checkpoint_path=config.checkpoint_dir + '_ceiling',
    )

    # Initialise with conditioning (to create all parameters including encoder).
    model.init(dummy_batch.query.features,
               dummy_batch.conditioning.features, seed=config.seed)

    # Create learnable variant embeddings as separate parameters.
    # These are optimised alongside the model.
    rng = jax.random.PRNGKey(config.seed + 100)
    variant_embeddings = jax.random.normal(
        rng, (n_variants, config.z_dim)) * 0.1

    # Separate optimizer for embeddings.
    emb_opt = optax.adam(config.learning_rate)
    emb_opt_state = emb_opt.init(variant_embeddings)

    # Training: use variant embeddings as z_override.
    @jax.jit
    def train_step(params, emb, rng_key, query_feedback, variant_idx):
        """One training step with learned variant embedding."""
        z = emb[variant_idx]  # (z_dim,)

        def loss_fn(params, emb_single):
            output_preds, hint_preds = model.net_fn.apply(
                params, rng_key,
                [query_feedback.features],
                False,      # repred
                0,          # algorithm_index
                True,       # return_hints
                False,      # return_all_outputs
                None,       # cond_features
                emb_single, # z_override
            )

            from phase2.model import _masked_d_hint_loss
            from clrs._src import losses
            nb_nodes = query_feedback.features.inputs[0].data.shape[1]
            total_loss = 0.0
            for truth in query_feedback.outputs:
                total_loss += losses.output_loss(
                    truth=truth, pred=output_preds[truth.name],
                    nb_nodes=nb_nodes)
            if hint_preds is not None:
                lengths = query_feedback.features.lengths
                for truth in query_feedback.features.hints:
                    if truth.name == 'd':
                        total_loss += _masked_d_hint_loss(
                            truth, [hp['d'] for hp in hint_preds], lengths)
                    else:
                        total_loss += losses.hint_loss(
                            truth=truth,
                            preds=[hp[truth.name] for hp in hint_preds],
                            lengths=lengths, nb_nodes=nb_nodes)
            return total_loss

        # Gradient w.r.t. both model params and this variant's embedding.
        loss, (grads_params, grads_emb) = jax.value_and_grad(
            loss_fn, argnums=(0, 1))(params, z)
        return loss, grads_params, grads_emb

    print(f'Training for {config.train_steps} steps...')
    rng = jax.random.PRNGKey(config.seed)
    t0 = time.time()

    for step in range(1, config.train_steps + 1):
        batch = pipeline.next()
        variant_idx = variant_to_idx[batch.variant_name]
        rng, key = jax.random.split(rng)

        loss, grads_params, grads_emb = train_step(
            model.params, variant_embeddings, key,
            batch.query, variant_idx)

        # Update model parameters.
        updates, model.opt_state = model.opt.update(grads_params, model.opt_state)
        model.params = optax.apply_updates(model.params, updates)

        # Update this variant's embedding.
        emb_updates, emb_opt_state = emb_opt.update(
            # Expand grad to full embedding matrix (zero for other variants).
            jnp.zeros_like(variant_embeddings).at[variant_idx].set(grads_emb),
            emb_opt_state)
        variant_embeddings = optax.apply_updates(variant_embeddings, emb_updates)

        if step % max(1, config.eval_every // 5) == 0:
            elapsed = time.time() - t0
            print(f'  step {step}: loss={float(loss):.2f}, '
                  f'variant={batch.variant_name}, {elapsed:.0f}s')

        # Evaluate.
        if step % config.eval_every == 0 or step == config.train_steps:
            print(f'\n  --- Ceiling Eval (step {step}) ---')
            rng, eval_key = jax.random.split(rng)

            eval_pipe = EvalPipeline(
                variant_names=all_variants, n=config.n, k=config.k,
                num_eval_samples=config.eval_samples,
                eval_batch_size=config.eval_batch_size)

            for name in all_train + all_test:
                idx = variant_to_idx[name]
                z = variant_embeddings[idx]
                eval_key, sub_key = jax.random.split(eval_key)

                n_correct = 0
                n_total = 0
                for eb in eval_pipe.eval_batches(name):
                    eval_key, sub_key2 = jax.random.split(eval_key)
                    out, _ = model.predict_with_z(
                        sub_key2, eb.query.features, z)
                    pi_pred = np.argmax(out['pi'], axis=-1)
                    pi_true = eb.query.outputs[0].data.astype(int)
                    source_mask = np.array(
                        [dp.data for dp in eb.query.features.inputs
                         if dp.name == 's'][0])
                    non_source = 1.0 - source_mask
                    n_correct += ((pi_pred == pi_true) * non_source).sum()
                    n_total += non_source.sum()

                acc = float(n_correct / max(n_total, 1))
                split = 'TEST' if name in all_test else 'train'
                print(f'    [{split:5s}] {name:<30s}: {acc:.3f}')

                if config.wandb_enabled and wandb is not None:
                    wandb.log({
                        f'ceiling/{split}/{name}/pred_accuracy': acc,
                        'step': step}, step=step)

            print()

    if config.wandb_enabled and wandb is not None:
        wandb.finish()
    print('Done!')


def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic 3: Parameter vector ceiling')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    config = SMOKE_CONFIG if args.smoke_test else TrainConfig()
    if args.no_wandb:
        config.wandb_enabled = False

    run_ceiling_diagnostic(config)


if __name__ == '__main__':
    main()
