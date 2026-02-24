"""Batch experiment runner for overnight cluster execution.

Runs training experiments sequentially, then eval-only diagnostics.
Each experiment gets a unique checkpoint directory.

Usage:
    python -m phase2.run_experiments
"""

import os
import time
from dataclasses import replace
from typing import Dict, List, Tuple

import clrs
import jax
import jax.numpy as jnp
import numpy as np

import phase0.variant_registry as variant_registry
from phase0.variant_registry import BASE_VARIANTS
from phase1.data_pipeline import ConditionedDataPipeline, EvalPipeline
from phase1.spec import RELAXATION_SPEC
from phase1.sampler import RelaxationSampler
from phase2.model import ConditionedModel
from phase2.train import TrainConfig, train, run_evaluation, evaluate_variant


# ---------------------------------------------------------------------------
# Variant groupings (non-overlapping, all 24 covered in 6 groups of 4)
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

# Non-overlapping groups: each variant appears in exactly one test group.
# Latin square structure for transforms within GOOD families.
GROUPS = {
    'G1': [  # Standard test set
        'add_min_<_reciprocal', 'max_min_<_square',
        'min_max_>_reciprocal', 'multiply_max_>_one_minus',
    ],
    'G2': [
        'add_min_<_one_minus', 'max_min_<_reciprocal',
        'min_max_>_square', 'multiply_max_>',
    ],
    'G3': [
        'add_min_<', 'max_min_<_one_minus',
        'min_max_>', 'multiply_max_>_square',
    ],
    'G4': [
        'add_min_<_square', 'max_min_<',
        'min_max_>_one_minus', 'multiply_min_<_reciprocal',
    ],
    'G5': [  # MARGINAL
        'max_min_>', 'max_min_>_one_minus',
        'min_max_<_square', 'min_max_<_reciprocal',
    ],
    'G6': [  # MARGINAL
        'max_min_>_square', 'max_min_>_reciprocal',
        'min_max_<', 'min_max_<_one_minus',
    ],
}

# Verify complete coverage.
_all_in_groups = sorted(sum(GROUPS.values(), []))
assert _all_in_groups == sorted(ALL_VIABLE), \
    f'Groups do not cover all variants: missing {set(ALL_VIABLE) - set(_all_in_groups)}'


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
    wandb_enabled=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_training_experiment(name: str, config: TrainConfig,
                            train_variants: List[str],
                            test_variants: List[str]):
    """Run a single training experiment with custom variant lists."""
    old_train = variant_registry.TRAIN_VARIANTS
    old_test = variant_registry.TEST_VARIANTS

    try:
        variant_registry.TRAIN_VARIANTS = list(train_variants)
        variant_registry.TEST_VARIANTS = list(test_variants)

        print(f'\n  Train variants ({len(train_variants)}), '
              f'Test variants ({len(test_variants)}): {test_variants}')
        print(f'  Config: n={config.n}, k={config.k}, batch={config.batch_size}, '
              f'hidden={config.hidden_dim}, z={config.z_dim}, '
              f'd_edge={config.d_edge}, steps={config.train_steps}, '
              f'lr={config.learning_rate}')

        return train(config, quiet=True)
    finally:
        variant_registry.TRAIN_VARIANTS = old_train
        variant_registry.TEST_VARIANTS = old_test


def build_model_from_checkpoint(checkpoint_dir: str, config: TrainConfig):
    """Build a model and restore from checkpoint."""
    dummy_pipe = ConditionedDataPipeline(
        variant_names=list(variant_registry.TRAIN_VARIANTS),
        n=config.n, k=5, batch_size=config.eval_batch_size, seed=999)
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


# ---------------------------------------------------------------------------
# Eval-only diagnostics
# ---------------------------------------------------------------------------

def run_k_sweep(checkpoint_dir: str, config: TrainConfig,
                k_values: List[int]):
    """Evaluate with varying numbers of conditioning examples."""
    print('\n' + '=' * 60)
    print('DIAGNOSTIC: k-sweep')
    print('=' * 60)

    model = build_model_from_checkpoint(checkpoint_dir, config)
    print(f'  Restored from {checkpoint_dir}/best.pkl')

    test_variants = GROUPS['G1']
    rng = jax.random.PRNGKey(13)

    for k in k_values:
        print(f'\n--- k={k} ---')
        accs = []

        if k == 0:
            # k=0: zero conditioning. Generate k=1 conditioning from any
            # variant (for valid shapes), then zero out the z tensor by
            # temporarily overriding model._z after the encoder runs.
            # We use a wrapper that calls predict but intercepts z.
            dummy_sampler = RelaxationSampler(
                variant=BASE_VARIANTS['max_min_>'],
                n=config.n, seed=789, randomize_pos=False)
            dummy_cond = dummy_sampler.next(1)

            for name in test_variants:
                query_sampler = RelaxationSampler(
                    variant=BASE_VARIANTS[name], n=config.n,
                    seed=123, randomize_pos=False)

                n_correct = 0
                n_total = 0
                n_batches = config.eval_samples // config.eval_batch_size
                for _ in range(n_batches):
                    query = query_sampler.next(config.eval_batch_size)
                    rng, sub_key = jax.random.split(rng)
                    # Run with dummy conditioning — model produces some z,
                    # but since we can't easily zero it within JIT, we use
                    # conditioning from a fixed arbitrary variant. This
                    # tests "uninformative z" (z from an unrelated variant).
                    output_preds, _ = model.predict(
                        sub_key, query.features, dummy_cond.features)
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
                accs.append(acc)
                print(f'  {name:35s}: {acc:.3f}')
            print(f'  (k=0 uses conditioning from max_min_> — uninformative z)')
        else:
            eval_pipe = EvalPipeline(
                variant_names=test_variants, n=config.n, k=k,
                num_eval_samples=config.eval_samples,
                eval_batch_size=config.eval_batch_size, seed=123)
            for name in test_variants:
                rng, sub_key = jax.random.split(rng)
                metrics = evaluate_variant(
                    model, eval_pipe, name, sub_key, k)
                acc = metrics['pred_accuracy']
                accs.append(acc)
                print(f'  {name:35s}: {acc:.3f}')

        print(f'  {"MEAN":35s}: {np.mean(accs):.3f}')


def run_wrong_conditioning(checkpoint_dir: str, config: TrainConfig):
    """Evaluate each test variant conditioned on wrong variant's traces."""
    print('\n' + '=' * 60)
    print('DIAGNOSTIC: Wrong conditioning')
    print('=' * 60)

    model = build_model_from_checkpoint(checkpoint_dir, config)
    print(f'  Restored from {checkpoint_dir}/best.pkl')

    test_variants = GROUPS['G1']
    # Each test variant conditioned on two different wrong variants.
    wrong_pairs = [
        ('add_min_<_reciprocal', 'min_max_>'),
        ('add_min_<_reciprocal', 'multiply_max_>_square'),
        ('max_min_<_square', 'multiply_max_>'),
        ('max_min_<_square', 'add_min_<_one_minus'),
        ('min_max_>_reciprocal', 'add_min_<'),
        ('min_max_>_reciprocal', 'max_min_<_reciprocal'),
        ('multiply_max_>_one_minus', 'add_min_<'),
        ('multiply_max_>_one_minus', 'min_max_>'),
    ]

    rng = jax.random.PRNGKey(13)

    # Correct conditioning reference.
    print(f'\n  --- Correct conditioning (reference) ---')
    eval_pipe = EvalPipeline(
        variant_names=test_variants, n=config.n, k=config.eval_k,
        num_eval_samples=config.eval_samples,
        eval_batch_size=config.eval_batch_size, seed=123)
    for name in test_variants:
        rng, sub_key = jax.random.split(rng)
        metrics = evaluate_variant(model, eval_pipe, name, sub_key, config.eval_k)
        print(f'    {name:35s}: {metrics["pred_accuracy"]:.3f}')

    # Wrong conditioning.
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


def run_ood_eval(checkpoint_dir: str, config: TrainConfig,
                 n_values: List[int]):
    """Evaluate at different graph sizes for OOD analysis."""
    print('\n' + '=' * 60)
    print('DIAGNOSTIC: OOD graph size sweep')
    print('=' * 60)

    model = build_model_from_checkpoint(checkpoint_dir, config)
    print(f'  Restored from {checkpoint_dir}/best.pkl')

    # Evaluate: 1 training variant + 4 test variants.
    train_variant = 'add_min_<'
    test_variants = GROUPS['G1']
    all_variants = [train_variant] + test_variants

    rng = jax.random.PRNGKey(13)

    for n in n_values:
        print(f'\n--- n={n} ---')
        eval_pipe = EvalPipeline(
            variant_names=all_variants, n=n, k=config.eval_k,
            num_eval_samples=config.eval_samples,
            eval_batch_size=config.eval_batch_size, seed=123)

        for name in all_variants:
            rng, sub_key = jax.random.split(rng)
            metrics = evaluate_variant(model, eval_pipe, name, sub_key, config.eval_k)
            tag = 'TRAIN' if name == train_variant else 'TEST '
            print(f'  {tag} {name:35s}: {metrics["pred_accuracy"]:.3f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _safe_run(name, fn, *args, **kwargs):
    """Run a function, catching and logging exceptions without aborting."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f'\n  !!! {name} FAILED: {e}')
        import traceback
        traceback.print_exc()
        print(f'  !!! Continuing to next experiment...\n')
        return None


def run_all():
    t0 = time.time()

    # ==================================================================
    # EXP 1: Main reference (G1 test, 5K steps)
    # ==================================================================
    print('\n' + '#' * 70)
    print('# EXP 1: main (G1 test)')
    print('#' * 70)
    g1_train = [v for v in ALL_VIABLE if v not in GROUPS['G1']]
    _safe_run('EXP1', run_training_experiment,
        'EXP1',
        replace(BASE, name='EXP1_main',
                checkpoint_dir='checkpoints/exp1_main'),
        g1_train, GROUPS['G1'],
    )

    # ==================================================================
    # EXP 2-6: Group coverage (G2-G6 test, 5K steps each)
    # ==================================================================
    for i, gname in enumerate(['G2', 'G3', 'G4', 'G5', 'G6'], start=2):
        print('\n' + '#' * 70)
        print(f'# EXP {i}: {gname} test')
        print('#' * 70)
        g_train = [v for v in ALL_VIABLE if v not in GROUPS[gname]]
        _safe_run(f'EXP{i}', run_training_experiment,
            f'EXP{i}',
            replace(BASE, name=f'EXP{i}_{gname}',
                    checkpoint_dir=f'checkpoints/exp{i}_{gname}'),
            g_train, GROUPS[gname],
        )

    # ==================================================================
    # Collect all test results and find 3 worst
    # ==================================================================
    print('\n' + '#' * 70)
    print('# Identifying 3 worst-performing variants for targeted LOO')
    print('#' * 70)

    all_test_accs: Dict[str, float] = {}

    for i, gname in enumerate(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'], start=1):
        ckpt_dir = f'checkpoints/exp{i}_main' if i == 1 else f'checkpoints/exp{i}_{gname}'
        try:
            config_eval = replace(BASE, checkpoint_dir=ckpt_dir)
            model = build_model_from_checkpoint(ckpt_dir, config_eval)
            rng = jax.random.PRNGKey(13)

            eval_pipe = EvalPipeline(
                variant_names=GROUPS[gname], n=config_eval.n,
                k=config_eval.eval_k,
                num_eval_samples=config_eval.eval_samples,
                eval_batch_size=config_eval.eval_batch_size, seed=123)

            for name in GROUPS[gname]:
                rng, sub_key = jax.random.split(rng)
                metrics = evaluate_variant(model, eval_pipe, name, sub_key,
                                           config_eval.eval_k)
                all_test_accs[name] = metrics['pred_accuracy']
        except Exception as e:
            print(f'  !!! Failed to evaluate {gname}: {e}')
            print(f'  !!! Skipping this group for worst-3 selection.')

    # Sort and print all evaluated variants.
    print(f'\n  Test accuracies for {len(all_test_accs)}/24 variants (sorted):')
    sorted_variants = sorted(all_test_accs.items(), key=lambda x: x[1])
    for name, acc in sorted_variants:
        print(f'    {name:35s}: {acc:.3f}')

    # Pick 3 worst (or fewer if not enough evaluated).
    n_worst = min(3, len(sorted_variants))
    worst_3 = [name for name, _ in sorted_variants[:n_worst]]
    print(f'\n  {n_worst} worst: {worst_3}')

    # ==================================================================
    # EXP 7-9: Targeted LOO on worst 3 (23 train, 1 test, 5K steps)
    # ==================================================================
    for i, holdout in enumerate(worst_3, start=7):
        safe = holdout.replace('<', 'lt').replace('>', 'gt')
        print('\n' + '#' * 70)
        print(f'# EXP {i}: LOO {holdout}')
        print('#' * 70)
        loo_train = [v for v in ALL_VIABLE if v != holdout]
        _safe_run(f'EXP{i}', run_training_experiment,
            f'EXP{i}',
            replace(BASE, name=f'EXP{i}_loo_{safe}',
                    checkpoint_dir=f'checkpoints/exp{i}_loo_{safe}'),
            loo_train, [holdout],
        )

    # ==================================================================
    # EXP 10: OOD graph sizes (n~U[8,20] training, 10K steps)
    # ==================================================================
    print('\n' + '#' * 70)
    print('# EXP 10: OOD graph sizes (n~U[8,20], 10K steps)')
    print('#' * 70)
    g1_train = [v for v in ALL_VIABLE if v not in GROUPS['G1']]
    ood_config = replace(BASE,
        name='EXP10_ood_sizes',
        checkpoint_dir='checkpoints/exp10_ood',
        train_steps=10000,
        warmup_steps=1000,
    )

    import phase1.data_pipeline as dp_module
    _orig_init = dp_module.ConditionedDataPipeline.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.n_range = (8, 20)

    dp_module.ConditionedDataPipeline.__init__ = _patched_init
    try:
        _safe_run('EXP10', run_training_experiment,
            'EXP10', ood_config, g1_train, GROUPS['G1'])
    finally:
        dp_module.ConditionedDataPipeline.__init__ = _orig_init

    # OOD eval sweep.
    _safe_run('OOD_eval', run_ood_eval,
        'checkpoints/exp10_ood', ood_config,
        [8, 12, 16, 20, 24, 32, 48, 64])

    # ==================================================================
    # EXP 11-12: Eval-only diagnostics (on OOD checkpoint)
    # ==================================================================
    _safe_run('k_sweep', run_k_sweep,
        'checkpoints/exp10_ood', ood_config, [0, 1, 2, 5, 8])

    _safe_run('wrong_cond', run_wrong_conditioning,
        'checkpoints/exp10_ood', ood_config)

    # ==================================================================
    elapsed = time.time() - t0
    print('\n' + '=' * 70)
    print(f'ALL EXPERIMENTS COMPLETED in {elapsed/3600:.1f} hours')
    print('=' * 70)


if __name__ == '__main__':
    run_all()
