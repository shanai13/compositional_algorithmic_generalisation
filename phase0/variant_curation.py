"""Task 0.3: Variant curation.

For each variant, run on random graphs of varying sizes. Check:
1. Convergence within n-1 steps
2. Output values in reasonable range (no explosion/collapse)
3. Non-trivial outputs (not all identical, not all at init value)
4. Source dependence (output changes when source changes)

Filter degenerate combinations and report findings.
"""

from typing import Dict, List

import numpy as np

from phase0.parameterized_relaxation import (
    generate_er_graph,
    parameterized_relaxation,
)
from phase0.variant_registry import BASE_VARIANTS, VariantParams


def curate_variant(
    variant: VariantParams,
    sizes: List[int] = (8, 12, 16, 24, 32),
    graphs_per_size: int = 10,
    p: float = 0.3,
    weight_range: tuple = (0.1, 1.0),
    seed: int = 42,
) -> Dict:
    """Run curation checks on a single variant.

    Returns dict of curation metrics.
    """
    rng = np.random.RandomState(seed)

    results = {
        'convergence_rates': [],       # Fraction converging within n-1 steps
        'avg_convergence_step': [],     # Average step at convergence
        'update_counts': [],            # Number of nodes updated (pred changed)
        'unique_d_counts': [],          # Number of unique distance values
        'd_ranges': [],                 # (min, max) of finite distances
        'source_dependent': [],         # Whether output changes with source
        'all_same_d': [],               # Whether all nodes get same distance
        'sizes': [],
    }

    for n in sizes:
        for g in range(graphs_per_size):
            adj, weights = generate_er_graph(
                n, p=p, weight_range=weight_range, rng=rng,
                ensure_connected=True,
            )
            source = rng.randint(n)

            trace = parameterized_relaxation(
                adj, weights, source,
                combine=variant.combine,
                aggregate=variant.aggregate,
                compare=variant.compare,
                init_source=variant.init_source,
                init_other=variant.init_other,
                weight_transform=variant.weight_transform,
            )

            # Convergence.
            converged = trace.converged_at >= 0
            results['convergence_rates'].append(converged)
            results['avg_convergence_step'].append(
                trace.converged_at if converged else trace.num_steps)

            # Update count.
            n_updated = int(np.sum(
                trace.pred_final != np.arange(n)))
            results['update_counts'].append(n_updated / n)

            # Distance diversity.
            d_finite = trace.d_final[np.isfinite(trace.d_final)]
            n_unique = len(set(np.round(d_finite, 8))) if len(d_finite) > 0 else 0
            results['unique_d_counts'].append(n_unique / n if n > 0 else 0)
            results['all_same_d'].append(n_unique <= 1)

            if len(d_finite) > 0:
                results['d_ranges'].append(
                    (float(d_finite.min()), float(d_finite.max())))
            else:
                results['d_ranges'].append((float('inf'), float('-inf')))

            # Source dependence (test with different source).
            source2 = (source + 1) % n
            trace2 = parameterized_relaxation(
                adj, weights, source2,
                combine=variant.combine,
                aggregate=variant.aggregate,
                compare=variant.compare,
                init_source=variant.init_source,
                init_other=variant.init_other,
                weight_transform=variant.weight_transform,
            )
            differs = not (
                np.allclose(trace.d_final, trace2.d_final, equal_nan=True)
                and np.array_equal(trace.pred_final, trace2.pred_final)
            )
            results['source_dependent'].append(differs)
            results['sizes'].append(n)

    return results


def classify_variant(name: str, results: Dict) -> Dict:
    """Classify a variant based on curation results."""
    conv_rate = np.mean(results['convergence_rates'])
    avg_update_frac = np.mean(results['update_counts'])
    avg_unique_frac = np.mean(results['unique_d_counts'])
    src_dep_rate = np.mean(results['source_dependent'])
    all_same_rate = np.mean(results['all_same_d'])

    issues = []
    if avg_update_frac < 0.1:
        issues.append('NO_UPDATES')
    if avg_unique_frac < 0.15:
        issues.append('LOW_DIVERSITY')
    if src_dep_rate < 0.1:
        issues.append('SOURCE_INDEPENDENT')
    if all_same_rate > 0.9:
        issues.append('UNIFORM_OUTPUT')

    # Check for value explosion.
    finite_ranges = [(lo, hi) for lo, hi in results['d_ranges']
                     if np.isfinite(lo) and np.isfinite(hi)]
    if finite_ranges:
        max_range = max(hi - lo for lo, hi in finite_ranges)
        if max_range > 1e6:
            issues.append('VALUE_EXPLOSION')

    viable = len(issues) == 0
    quality = 'GOOD' if viable else 'DEGENERATE'
    if not viable and 'SOURCE_INDEPENDENT' in issues and len(issues) == 1:
        quality = 'MARGINAL'  # Source-independent but otherwise OK

    return {
        'name': name,
        'convergence_rate': conv_rate,
        'avg_update_fraction': avg_update_frac,
        'avg_unique_fraction': avg_unique_frac,
        'source_dependent_rate': src_dep_rate,
        'all_same_rate': all_same_rate,
        'issues': issues,
        'quality': quality,
        'viable': viable,
    }


def run_curation(
    sizes: List[int] = (8, 12, 16, 24, 32),
    graphs_per_size: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Run curation for all base variants.

    Returns dict mapping variant name -> classification results.
    """
    if verbose:
        print(f'Running variant curation: sizes={sizes}, '
              f'{graphs_per_size} graphs/size')

    classifications = {}
    for name in sorted(BASE_VARIANTS.keys()):
        variant = BASE_VARIANTS[name]
        if verbose:
            print(f'  Curating {name}...')
        results = curate_variant(
            variant, sizes=sizes, graphs_per_size=graphs_per_size, seed=seed,
        )
        classifications[name] = classify_variant(name, results)

    return classifications


def print_curation_report(classifications: Dict[str, Dict]):
    """Print curation report."""
    print('\n' + '=' * 80)
    print('PHASE 0.3: VARIANT CURATION REPORT')
    print('=' * 80)

    print(f'\n{"Variant":<16s} {"Quality":<12s} {"Conv%":>6s} {"Upd%":>6s} '
          f'{"UniqD%":>7s} {"SrcDep%":>8s} {"Issues"}')
    print('-' * 80)

    good, marginal, degenerate = [], [], []
    for name in sorted(classifications.keys()):
        c = classifications[name]
        issues_str = ', '.join(c['issues']) if c['issues'] else '-'
        print(f'{name:<16s} {c["quality"]:<12s} '
              f'{c["convergence_rate"]*100:5.1f}% '
              f'{c["avg_update_fraction"]*100:5.1f}% '
              f'{c["avg_unique_fraction"]*100:6.1f}% '
              f'{c["source_dependent_rate"]*100:7.1f}% '
              f' {issues_str}')

        if c['quality'] == 'GOOD':
            good.append(name)
        elif c['quality'] == 'MARGINAL':
            marginal.append(name)
        else:
            degenerate.append(name)

    print(f'\n--- Summary ---')
    print(f'GOOD ({len(good)}):       {", ".join(good) if good else "none"}')
    print(f'MARGINAL ({len(marginal)}):   '
          f'{", ".join(marginal) if marginal else "none"}')
    print(f'DEGENERATE ({len(degenerate)}): '
          f'{", ".join(degenerate) if degenerate else "none"}')

    if len(good) < 8:
        print(f'\nWARNING: Only {len(good)} viable variants. Need to expand '
              f'variant set (add multiply operator, init variations, or '
              f'directed graphs).')

    return good, marginal, degenerate


if __name__ == '__main__':
    classifications = run_curation(verbose=True)
    print_curation_report(classifications)
