"""Batch experiment runner for overnight cluster execution.

Runs all training experiments sequentially, then eval-only diagnostics.
Each experiment gets a unique wandb run name and checkpoint directory.

Usage:
    python -m phase2.run_experiments
"""

import os
import time
from dataclasses import replace
from typing import List, Tuple

import clrs
import jax
import jax.numpy as jnp
import numpy as np

from phase0.variant_registry import (
    BASE_VARIANTS, TRAIN_VARIANTS, TEST_VARIANTS,
)
from phase1.data_pipeline import ConditionedDataPipeline, EvalPipeline
from phase1.spec import RELAXATION_SPEC
from phase1.sampler import RelaxationSampler
from phase2.model import ConditionedModel
from phase2.train import TrainConfig, train, run_evaluation, evaluate_variant


# ---------------------------------------------------------------------------
# Variant lists
# ---------------------------------------------------------------------------

ALL_GOOD = [
    'add_min_<', 'add_min_<_one_minus', 'add_min_<_reciprocal', 'add_min_<_square',
    'max_min_<', 'max_min_<_one_minus', 'max_min_<_reciprocal', 'max_min_<_square',
    'min_max_>', 'min_max_>_one_minus', 'min_max_>_reciprocal', 'min_max_>_square',
    'multiply_max_>', 'multiply_max_>_one_minus', 'multiply_max_>_square',
    'multiply_min_<_reciprocal',
]

ALL_MARGINAL = [
    'max_min_>', 'max_min_>_one_minus', 'max_min_>_reciprocal', 'max_min_>_square',
    'min_max_<', 'min_max_<_one_minus', 'min_max_<_reciprocal', 'min_max_<_square',
]

ALL_VIABLE = ALL_GOOD + ALL_MARGINAL

ORIGINAL_15_TRAIN = [
    'add_min_<', 'add_min_<_square',
    'max_min_<', 'max_min_<_reciprocal',
    'min_max_>', 'min_max_>_square',
    'multiply_max_>', 'multiply_max_>_square',
    'multiply_min_<_reciprocal',
    'max_min_>', 'max_min_>_reciprocal', 'max_min_>_square',
    'min_max_<', 'min_max_<_reciprocal', 'min_max_<_square',
]

ORIGINAL_3_TEST = [
    'add_min_<_reciprocal',
    'max_min_<_square',
    'min_max_>_reciprocal',
]


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

BASE = TrainConfig(
    hidden_dim=256,
    z_dim=32,
    d_node=16,
    d_edge=16,
    d_graph=8,
    cond_hidden_dim=128,
    cond_nb_layers=2,
    train_steps=5000,
    learning_rate=0.0005,
    warmup_steps=500,
    eval_every=500,
    eval_samples=256,
    wandb_enabled=True,
    wandb_project='compositional_algorithmic_generalisation',
    wandb_entity='shanai',
)

SHORT = replace(BASE, train_steps=3000, eval_every=500)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_loo_split(holdout: str) -> Tuple[List[str], List[str]]:
    """Train on all 24 viable minus holdout, test on holdout."""
    train_v = [v for v in ALL_VIABLE if v != holdout]
    return train_v, [holdout]


def make_family_holdout_split(family_prefix: str) -> Tuple[List[str], List[str]]:
    """Hold out all GOOD variants matching a (combine, agg, cmp) family."""
    holdout = set()
    for v in ALL_GOOD:
        # Identity variant: exact match.
        if v == family_prefix:
            holdout.add(v)
        # Variants with weight transform suffix.
        for wt in ['_reciprocal', '_square', '_one_minus']:
            if v == family_prefix + wt:
                holdout.add(v)
    train_v = [v for v in ALL_VIABLE if v not in holdout]
    return train_v, sorted(holdout)


def run_training_experiment(name: str, config: TrainConfig,
                            train_variants: List[str],
                            test_variants: List[str]):
    """Run a single training experiment with custom variant lists."""
    import phase0.variant_registry as registry
    old_train = registry.TRAIN_VARIANTS
    old_test = registry.TEST_VARIANTS

    try:
        registry.TRAIN_VARIANTS = list(train_variants)
        registry.TEST_VARIANTS = list(test_variants)

        print(f'\n  Config: n={config.n}, k={config.k}, batch={config.batch_size}, '
              f'hidden={config.hidden_dim}, z={config.z_dim}, '
              f'd_node={config.d_node}, d_edge={config.d_edge}, '
              f'd_graph={config.d_graph}, processor={config.processor_type}')
        print(f'  Steps: {config.train_steps}, lr={config.learning_rate}')

        return train(config, quiet=True)
    finally:
        registry.TRAIN_VARIANTS = old_train
        registry.TEST_VARIANTS = old_test


# ---------------------------------------------------------------------------
# Eval-only diagnostics
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(checkpoint_dir: str, config: TrainConfig):
    """Build a model and restore from checkpoint."""
    dummy_pipe = ConditionedDataPipeline(
        variant_names=list(TRAIN_VARIANTS), n=config.n, k=5,
        batch_size=config.eval_batch_size, seed=999)
    dummy_batch = dummy_pipe.next()

    processor_factory = clrs.get_processor_factory(
        config.processor_type, use_ln=config.use_ln,
        nb_triplet_fts=config.nb_triplet_fts, nb_heads=config.nb_heads)

    model = ConditionedModel(
        spec=RELAXATION_SPEC, dummy_trajectory=dummy_batch.query,
        processor_factory=processor_factory,
        hidden_dim=config.hidden_dim, z_dim=config.z_dim,
        d_node=config.d_node, d_edge=config.d_edge, d_graph=config.d_graph,
        cond_hidden_dim=config.cond_hidden_dim,
        cond_nb_layers=config.cond_nb_layers,
        encode_hints=config.encode_hints, decode_hints=config.decode_hints,
        encoder_init=config.encoder_init, use_lstm=config.use_lstm,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        train_steps=config.train_steps,
        grad_clip_max_norm=config.grad_clip_max_norm,
        hint_teacher_forcing=config.hint_teacher_forcing,
        hint_repred_mode=config.hint_repred_mode,
        nb_msg_passing_steps=config.nb_msg_passing_steps,
        checkpoint_path=checkpoint_dir,
    )
    model.init(dummy_batch.query.features,
               dummy_batch.conditioning.features, seed=config.seed)
    model.restore_model('best.pkl')
    return model


def run_k_sweep(checkpoint_dir: str, config: TrainConfig,
                k_values: List[int]):
    """Evaluate with varying numbers of conditioning examples."""
    print('\n' + '=' * 60)
    print('DIAGNOSTIC: k-sweep')
    print('=' * 60)

    model = build_model_from_checkpoint(checkpoint_dir, config)
    print(f'  Restored from {checkpoint_dir}/best.pkl')

    rng = jax.random.PRNGKey(42)

    for k in k_values:
        print(f'\n--- k={k} ---')
        eval_k = max(k, 1)  # k=0 still needs k=1 for valid shapes
        eval_pipe = EvalPipeline(
            variant_names=ORIGINAL_3_TEST, n=config.n, k=eval_k,
            num_eval_samples=config.eval_samples,
            eval_batch_size=config.eval_batch_size, seed=123)

        accs = []
        for name in ORIGINAL_3_TEST:
            if k == 0:
                # For k=0: run with k=1 conditioning but from a random
                # mismatched variant. This gives an uninformative z.
                wrong_sampler = RelaxationSampler(
                    variant=BASE_VARIANTS['max_min_>'],  # arbitrary
                    n=config.n, seed=789, randomize_pos=False)
                wrong_cond = wrong_sampler.next(1)

                query_sampler = RelaxationSampler(
                    variant=BASE_VARIANTS[name], n=config.n,
                    seed=123, randomize_pos=False)

                n_correct = 0
                n_total = 0
                n_batches = config.eval_samples // config.eval_batch_size
                for _ in range(n_batches):
                    query = query_sampler.next(config.eval_batch_size)
                    rng, sub_key = jax.random.split(rng)
                    output_preds, _ = model.predict(
                        sub_key, query.features, wrong_cond.features)
                    pi_pred = np.argmax(output_preds['pi'], axis=-1)
                    pi_true = query.outputs[0].data
                    source_mask = np.array(
                        [dp.data for dp in query.features.inputs
                         if dp.name == 's'][0])
                    non_source = 1.0 - source_mask
                    correct = (pi_pred == pi_true.astype(int)) * non_source
                    n_correct += correct.sum()
                    n_total += non_source.sum()
                acc = float(n_correct / max(n_total, 1))
            else:
                rng, sub_key = jax.random.split(rng)
                metrics = evaluate_variant(
                    model, eval_pipe, name, sub_key, eval_k)
                acc = metrics['pred_accuracy']

            accs.append(acc)
            print(f'  {name:35s}: {acc:.3f}')

        print(f'  {"MEAN":35s}: {np.mean(accs):.3f}')


def run_wrong_conditioning(checkpoint_dir: str, config: TrainConfig):
    """Evaluate each test variant conditioned on a wrong variant's traces."""
    print('\n' + '=' * 60)
    print('DIAGNOSTIC: Wrong conditioning')
    print('=' * 60)

    model = build_model_from_checkpoint(checkpoint_dir, config)
    print(f'  Restored from {checkpoint_dir}/best.pkl')

    # Each test variant gets conditioned on a completely different family.
    wrong_pairs = [
        ('add_min_<_reciprocal', 'min_max_>'),
        ('add_min_<_reciprocal', 'multiply_max_>_square'),
        ('max_min_<_square', 'multiply_max_>'),
        ('max_min_<_square', 'add_min_<_one_minus'),
        ('min_max_>_reciprocal', 'add_min_<'),
        ('min_max_>_reciprocal', 'max_min_<_reciprocal'),
    ]

    rng = jax.random.PRNGKey(42)

    # First: correct conditioning (reference).
    print(f'\n  --- Correct conditioning (reference) ---')
    eval_pipe = EvalPipeline(
        variant_names=ORIGINAL_3_TEST, n=config.n, k=config.eval_k,
        num_eval_samples=config.eval_samples,
        eval_batch_size=config.eval_batch_size, seed=123)
    for name in ORIGINAL_3_TEST:
        rng, sub_key = jax.random.split(rng)
        metrics = evaluate_variant(model, eval_pipe, name, sub_key, config.eval_k)
        print(f'    {name:35s}: {metrics["pred_accuracy"]:.3f}')

    # Then: wrong conditioning.
    print(f'\n  --- Wrong conditioning ---')
    for test_name, wrong_name in wrong_pairs:
        query_sampler = RelaxationSampler(
            variant=BASE_VARIANTS[test_name], n=config.n,
            seed=123, randomize_pos=False)
        wrong_sampler = RelaxationSampler(
            variant=BASE_VARIANTS[wrong_name], n=config.n,
            seed=456, randomize_pos=False)
        wrong_cond = wrong_sampler.next(config.eval_k)

        n_correct = 0
        n_total = 0
        n_batches = config.eval_samples // config.eval_batch_size
        for _ in range(n_batches):
            query = query_sampler.next(config.eval_batch_size)
            rng, sub_key = jax.random.split(rng)
            output_preds, _ = model.predict(
                sub_key, query.features, wrong_cond.features)
            pi_pred = np.argmax(output_preds['pi'], axis=-1)
            pi_true = query.outputs[0].data
            source_mask = np.array(
                [dp.data for dp in query.features.inputs
                 if dp.name == 's'][0])
            non_source = 1.0 - source_mask
            correct = (pi_pred == pi_true.astype(int)) * non_source
            n_correct += correct.sum()
            n_total += non_source.sum()

        acc = float(n_correct / max(n_total, 1))
        print(f'    Test: {test_name:30s} | Cond: {wrong_name:25s} | Acc: {acc:.3f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    t0 = time.time()

    # ==================================================================
    # EXP 1: Main reference (21 train, 3 test, 5K steps)
    # ==================================================================
    print('\n' + '#' * 70)
    print('# EXP 1: main_reference')
    print('#' * 70)
    run_training_experiment(
        'EXP1: main_reference',
        replace(BASE,
                name='EXP1_main_reference',
                checkpoint_dir='checkpoints/exp1_main'),
        list(TRAIN_VARIANTS),
        ORIGINAL_3_TEST,
    )

    # ==================================================================
    # EXP 2: Ablation — no one_minus (15 train, 3 test)
    # ==================================================================
    print('\n' + '#' * 70)
    print('# EXP 2: ablation_no_one_minus')
    print('#' * 70)
    run_training_experiment(
        'EXP2: no_one_minus',
        replace(BASE,
                name='EXP2_ablation_no_one_minus',
                checkpoint_dir='checkpoints/exp2_no_one_minus'),
        ORIGINAL_15_TRAIN,
        ORIGINAL_3_TEST,
    )

    # ==================================================================
    # EXP 3: Ablation — no edge z (d_edge=0)
    # ==================================================================
    print('\n' + '#' * 70)
    print('# EXP 3: ablation_no_edge_z')
    print('#' * 70)
    run_training_experiment(
        'EXP3: no_edge_z',
        replace(BASE,
                name='EXP3_ablation_no_edge_z',
                checkpoint_dir='checkpoints/exp3_no_edge_z',
                d_edge=0),
        list(TRAIN_VARIANTS),
        ORIGINAL_3_TEST,
    )

    # ==================================================================
    # EXP 4–8: Leave-one-out (train on 23, test on 1)
    # ==================================================================
    loo_variants = [
        'add_min_<_reciprocal',
        'max_min_<_square',
        'min_max_>_reciprocal',
        'multiply_max_>_one_minus',
        'multiply_min_<_reciprocal',
    ]

    for i, holdout in enumerate(loo_variants, start=4):
        safe = holdout.replace('<', 'lt').replace('>', 'gt')
        print('\n' + '#' * 70)
        print(f'# EXP {i}: loo_{holdout}')
        print('#' * 70)

        loo_train, loo_test = make_loo_split(holdout)
        run_training_experiment(
            f'EXP{i}: LOO {holdout}',
            replace(SHORT,
                    name=f'EXP{i}_loo_{safe}',
                    checkpoint_dir=f'checkpoints/exp{i}_loo_{safe}'),
            loo_train,
            loo_test,
        )

    # ==================================================================
    # EXP 9–10: Family hold-out (never seen this combine operator)
    # ==================================================================
    families = [
        ('add_min_<', 'add'),
        ('min_max_>', 'min_max_gt'),
    ]

    for i, (prefix, label) in enumerate(families, start=9):
        print('\n' + '#' * 70)
        print(f'# EXP {i}: family_holdout_{label}')
        print('#' * 70)

        fam_train, fam_test = make_family_holdout_split(prefix)
        run_training_experiment(
            f'EXP{i}: family_holdout_{label}',
            replace(SHORT,
                    name=f'EXP{i}_family_{label}',
                    checkpoint_dir=f'checkpoints/exp{i}_family_{label}'),
            fam_train,
            fam_test,
        )

    # ==================================================================
    # EXP 11: k-sweep (eval only on EXP1 checkpoint)
    # ==================================================================
    run_k_sweep('checkpoints/exp1_main', BASE, k_values=[0, 1, 2, 5, 8])

    # ==================================================================
    # EXP 12: Wrong conditioning (eval only on EXP1 checkpoint)
    # ==================================================================
    run_wrong_conditioning('checkpoints/exp1_main', BASE)

    # ==================================================================
    elapsed = time.time() - t0
    print('\n' + '=' * 70)
    print(f'ALL EXPERIMENTS COMPLETED in {elapsed/3600:.1f} hours')
    print('=' * 70)


if __name__ == '__main__':
    run_all()
