"""Run all Phase 0 feasibility checks.

Usage:
    python -m phase0.run_phase0 [--quick]

The --quick flag runs with reduced graph counts for fast iteration.
"""

import argparse
import json
import os
import sys

import numpy as np

from phase0.divergence_analysis import run_divergence_analysis, print_report
from phase0.variant_curation import run_curation, print_curation_report
from phase0.variant_registry import BASE_VARIANTS


def main():
    parser = argparse.ArgumentParser(description='Phase 0 feasibility checks')
    parser.add_argument('--quick', action='store_true',
                        help='Reduced counts for fast smoke test')
    parser.add_argument('--output-dir', type=str, default='phase0_results',
                        help='Directory to save results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.quick:
        div_graphs = 50
        cur_sizes = [8, 16]
        cur_graphs = 5
    else:
        div_graphs = 1000
        cur_sizes = [8, 12, 16, 24, 32]
        cur_graphs = 10

    # ----------------------------------------------------------------
    # Task 0.3: Variant curation (run first — identifies degenerate variants)
    # ----------------------------------------------------------------
    print('\n' + '#' * 80)
    print('# TASK 0.3: VARIANT CURATION')
    print('#' * 80)

    classifications = run_curation(
        sizes=cur_sizes, graphs_per_size=cur_graphs, verbose=True,
    )
    good, marginal, degenerate = print_curation_report(classifications)

    # Save curation results.
    curation_path = os.path.join(args.output_dir, 'curation_results.json')
    with open(curation_path, 'w') as f:
        json.dump(classifications, f, indent=2)
    print(f'\nCuration results saved to {curation_path}')

    # ----------------------------------------------------------------
    # Task 0.2: Output divergence analysis
    # ----------------------------------------------------------------
    print('\n' + '#' * 80)
    print('# TASK 0.2: OUTPUT DIVERGENCE ANALYSIS')
    print('#' * 80)

    results = run_divergence_analysis(
        num_graphs=div_graphs, n=16, p=0.3, seed=42, verbose=True,
    )
    print_report(results)

    # Save matrices.
    div_path = os.path.join(args.output_dir, 'divergence_results.npz')
    np.savez(
        div_path,
        pred_disagreement=results['pred_disagreement'],
        rank_disagreement=results['rank_disagreement'],
        variant_names=results['variant_names'],
    )
    print(f'\nDivergence matrices saved to {div_path}')

    # ----------------------------------------------------------------
    # Recommendations
    # ----------------------------------------------------------------
    print('\n' + '#' * 80)
    print('# RECOMMENDATIONS')
    print('#' * 80)

    print(f'\nViable variants for the project: {good}')
    if len(good) < 8:
        print('\nRecommended expansions to reach 12+ viable variants:')
        print('  1. Add multiply operator (weights in [0,1]):')
        print('     - (multiply, min, <) with init_source=1, init_other=inf')
        print('     - (multiply, max, >) with init_source=1, init_other=-inf')
        print('  2. Add init_source variations for add combine:')
        print('     - (add, min, <) with init_source=1 (shifts distances by 1)')
        print('  3. Consider directed graphs (doubles variant count)')
        print('  4. Consider different weight ranges')

    if marginal:
        print(f'\nMarginal variants (source-independent): {marginal}')
        print('  These compute valid graph properties but ignore the source node.')
        print('  Include if needed for variant count, but note the limitation.')


if __name__ == '__main__':
    main()
