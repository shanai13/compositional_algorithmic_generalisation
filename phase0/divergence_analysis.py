"""Task 0.2: Output divergence analysis.

Generate random ER graphs, run all variants, compute pairwise disagreement
on distances and predecessors. This is a GO/NO-GO gate: if variants agree
>80% of the time, the graph distribution or variant set needs redesign.
"""

import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from phase0.parameterized_relaxation import (
    AlgorithmTrace,
    generate_er_graph,
    parameterized_relaxation,
)
from phase0.variant_registry import BASE_VARIANTS, VariantParams


def run_variant(adj, weights, source, variant: VariantParams) -> AlgorithmTrace:
    """Run a single variant on a graph."""
    return parameterized_relaxation(
        adj, weights, source,
        combine=variant.combine,
        aggregate=variant.aggregate,
        compare=variant.compare,
        init_source=variant.init_source,
        init_other=variant.init_other,
        weight_transform=variant.weight_transform,
    )


def predecessor_disagreement(pred_a: np.ndarray, pred_b: np.ndarray,
                             source: int) -> float:
    """Fraction of non-source nodes where predecessors differ.

    Excludes the source node (always self-pointing) and nodes where both
    variants have self-loops (unreachable by both).
    """
    n = len(pred_a)
    mask = np.ones(n, dtype=bool)
    mask[source] = False
    # Exclude nodes unreachable by both (both self-loops).
    both_self = (pred_a == np.arange(n)) & (pred_b == np.arange(n))
    mask &= ~both_self
    if mask.sum() == 0:
        return 0.0
    return np.mean(pred_a[mask] != pred_b[mask])


def distance_rank_disagreement(d_a: np.ndarray, d_b: np.ndarray,
                               source: int) -> float:
    """Fraction of node pairs with different distance rank ordering.

    Uses only finite-valued nodes. Computes Kendall-tau-style disagreement:
    for each pair (i, j), check if d_a[i] < d_a[j] has different sign from
    d_b[i] < d_b[j].
    """
    # Nodes finite in both variants.
    finite = np.isfinite(d_a) & np.isfinite(d_b)
    finite[source] = False
    idx = np.where(finite)[0]
    if len(idx) < 2:
        return 0.0

    n_pairs = 0
    n_disagree = 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            u, v = idx[i], idx[j]
            sign_a = np.sign(d_a[u] - d_a[v])
            sign_b = np.sign(d_b[u] - d_b[v])
            n_pairs += 1
            if sign_a != sign_b:
                n_disagree += 1
    return n_disagree / n_pairs if n_pairs > 0 else 0.0


def run_divergence_analysis(
    num_graphs: int = 1000,
    n: int = 16,
    p: float = 0.3,
    weight_range: tuple = (0.1, 1.0),
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Run full divergence analysis across all base variants.

    Returns dict with:
        'pred_disagreement': (V, V) matrix of avg predecessor disagreement
        'rank_disagreement': (V, V) matrix of avg rank disagreement
        'variant_names': list of variant names (same order as matrices)
        'variant_stats': per-variant stats (convergence, update counts, etc.)
    """
    rng = np.random.RandomState(seed)
    variant_names = sorted(BASE_VARIANTS.keys())
    V = len(variant_names)

    pred_disagree_sum = np.zeros((V, V))
    rank_disagree_sum = np.zeros((V, V))
    pred_disagree_count = np.zeros((V, V))
    rank_disagree_count = np.zeros((V, V))

    # Per-variant stats.
    stats = {name: {
        'converged_counts': [],
        'n_updated': [],
        'n_unique_d': [],
        'source_dependent': [],
    } for name in variant_names}

    if verbose:
        print(f'Running divergence analysis: {num_graphs} graphs, n={n}, '
              f'p={p}, weights={weight_range}')

    for g in range(num_graphs):
        if verbose and (g + 1) % 100 == 0:
            print(f'  Graph {g + 1}/{num_graphs}')

        adj, weights = generate_er_graph(n, p=p, weight_range=weight_range,
                                         rng=rng, ensure_connected=True)
        source = rng.randint(n)

        # Run all variants.
        traces = {}
        for name in variant_names:
            v = BASE_VARIANTS[name]
            traces[name] = run_variant(adj, weights, source, v)

        # Collect per-variant stats.
        for name in variant_names:
            t = traces[name]
            n_updated = int(np.sum(t.pred_final != np.arange(n)))
            d_finite = t.d_final[np.isfinite(t.d_final)]
            n_unique = len(set(np.round(d_finite, 8))) if len(d_finite) > 0 else 0
            stats[name]['converged_counts'].append(t.converged_at)
            stats[name]['n_updated'].append(n_updated)
            stats[name]['n_unique_d'].append(n_unique)

        # Check source dependence on a few graphs.
        if g < 50:
            source2 = (source + 1) % n
            for name in variant_names:
                v = BASE_VARIANTS[name]
                t2 = run_variant(adj, weights, source2, v)
                differs = not (np.allclose(traces[name].d_final, t2.d_final,
                                           equal_nan=True)
                               and np.array_equal(traces[name].pred_final,
                                                  t2.pred_final))
                stats[name]['source_dependent'].append(differs)

        # Pairwise disagreement.
        for i in range(V):
            for j in range(i + 1, V):
                ti = traces[variant_names[i]]
                tj = traces[variant_names[j]]

                # Skip pairs where both produced no updates.
                if (np.array_equal(ti.pred_final, np.arange(n)) and
                        np.array_equal(tj.pred_final, np.arange(n))):
                    continue

                pd = predecessor_disagreement(ti.pred_final, tj.pred_final,
                                              source)
                pred_disagree_sum[i, j] += pd
                pred_disagree_sum[j, i] += pd
                pred_disagree_count[i, j] += 1
                pred_disagree_count[j, i] += 1

                rd = distance_rank_disagreement(ti.d_final, tj.d_final,
                                                source)
                rank_disagree_sum[i, j] += rd
                rank_disagree_sum[j, i] += rd
                rank_disagree_count[i, j] += 1
                rank_disagree_count[j, i] += 1

    # Average.
    with np.errstate(divide='ignore', invalid='ignore'):
        pred_matrix = np.where(pred_disagree_count > 0,
                               pred_disagree_sum / pred_disagree_count, 0.0)
        rank_matrix = np.where(rank_disagree_count > 0,
                               rank_disagree_sum / rank_disagree_count, 0.0)

    # Summarise stats.
    summary = {}
    for name in variant_names:
        s = stats[name]
        n_updates = np.array(s['n_updated'])
        src_dep = s['source_dependent']
        summary[name] = {
            'avg_converged_at': np.mean(s['converged_counts']),
            'avg_n_updated': np.mean(n_updates),
            'frac_no_updates': np.mean(n_updates == 0),
            'avg_unique_d': np.mean(s['n_unique_d']),
            'source_dependent_frac': np.mean(src_dep) if src_dep else None,
        }

    return {
        'pred_disagreement': pred_matrix,
        'rank_disagreement': rank_matrix,
        'variant_names': variant_names,
        'variant_stats': summary,
        'num_graphs': num_graphs,
    }


def print_report(results: Dict):
    """Print a human-readable report of divergence analysis results."""
    names = results['variant_names']
    pred = results['pred_disagreement']
    rank = results['rank_disagreement']
    stats = results['variant_stats']
    V = len(names)

    print('\n' + '=' * 80)
    print('PHASE 0.2: OUTPUT DIVERGENCE ANALYSIS')
    print(f'({results["num_graphs"]} graphs)')
    print('=' * 80)

    # Per-variant summary.
    print('\n--- Per-Variant Summary ---\n')
    print(f'{"Variant":<16s} {"AvgConv":>8s} {"AvgUpd":>8s} {"NoUpd%":>8s} '
          f'{"UniqD":>8s} {"SrcDep%":>8s}')
    print('-' * 64)

    viable_variants = []
    for name in names:
        s = stats[name]
        src = f'{s["source_dependent_frac"]*100:.0f}' if s["source_dependent_frac"] is not None else '?'
        print(f'{name:<16s} {s["avg_converged_at"]:8.1f} '
              f'{s["avg_n_updated"]:8.1f} '
              f'{s["frac_no_updates"]*100:7.1f}% '
              f'{s["avg_unique_d"]:8.1f} '
              f'{src:>7s}%')

        # Classify viability.
        if s['frac_no_updates'] < 0.1 and s['avg_n_updated'] > 1:
            viable_variants.append(name)

    print(f'\nViable variants (>90% graphs have updates): {len(viable_variants)}')
    for name in viable_variants:
        src = stats[name]['source_dependent_frac']
        tag = 'source-dependent' if src and src > 0.5 else 'source-INDEPENDENT'
        print(f'  {name}: {tag}')

    # Pairwise disagreement (only viable variants).
    if len(viable_variants) > 1:
        print('\n--- Predecessor Disagreement (viable pairs) ---\n')
        vidx = [names.index(n) for n in viable_variants]
        header = f'{"":16s}' + ''.join(f'{n[:10]:>11s}' for n in viable_variants)
        print(header)
        for i, ni in enumerate(viable_variants):
            row = f'{ni:<16s}'
            for j, nj in enumerate(viable_variants):
                val = pred[vidx[i], vidx[j]]
                row += f'{val*100:10.1f}%'
            print(row)

        print('\n--- Distance Rank Disagreement (viable pairs) ---\n')
        print(header)
        for i, ni in enumerate(viable_variants):
            row = f'{ni:<16s}'
            for j, nj in enumerate(viable_variants):
                val = rank[vidx[i], vidx[j]]
                row += f'{val*100:10.1f}%'
            print(row)

        # Summary statistics.
        upper_pred = [pred[vidx[i], vidx[j]]
                      for i in range(len(vidx))
                      for j in range(i + 1, len(vidx))]
        upper_rank = [rank[vidx[i], vidx[j]]
                      for i in range(len(vidx))
                      for j in range(i + 1, len(vidx))]

        if upper_pred:
            print(f'\nPredecessor disagreement: '
                  f'mean={np.mean(upper_pred)*100:.1f}%, '
                  f'min={np.min(upper_pred)*100:.1f}%, '
                  f'max={np.max(upper_pred)*100:.1f}%')
        if upper_rank:
            print(f'Rank disagreement:        '
                  f'mean={np.mean(upper_rank)*100:.1f}%, '
                  f'min={np.min(upper_rank)*100:.1f}%, '
                  f'max={np.max(upper_rank)*100:.1f}%')

        # GO/NO-GO assessment.
        if upper_pred:
            mean_agree = 1 - np.mean(upper_pred)
            print(f'\n--- GO/NO-GO GATE ---')
            if mean_agree > 0.8:
                print(f'FAIL: Mean predecessor agreement = {mean_agree*100:.1f}% '
                      f'(>80%). Need to redesign graph distribution or variant set.')
            else:
                print(f'PASS: Mean predecessor agreement = {mean_agree*100:.1f}% '
                      f'(<80%). Sufficient divergence.')


if __name__ == '__main__':
    results = run_divergence_analysis(
        num_graphs=1000, n=16, p=0.3, seed=42, verbose=True,
    )
    print_report(results)
