"""Registry of parameterised algorithm variants.

Each variant is a tuple (combine, aggregate, compare, weight_transform,
init_source, init_other). Init values are chosen to be consistent with
the operators:
  - init_source: identity element of combine (add->0, min->inf, max->-inf)
  - init_other: worst-case for compare (<->inf, >->-inf)

The weight_transform is a 4th compositional axis: the same (combine, agg, cmp)
operators applied with different weight transforms produce genuinely different
predecessor trees.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class VariantParams:
    """Parameters defining a single algorithm variant."""
    combine: str
    aggregate: str
    compare: str
    init_source: str
    init_other: str
    weight_transform: str = 'identity'

    @property
    def short_name(self) -> str:
        if self.weight_transform == 'identity':
            return f"{self.combine}_{self.aggregate}_{self.compare}"
        return f"{self.combine}_{self.aggregate}_{self.compare}_{self.weight_transform}"

    @property
    def display_name(self) -> str:
        wt = f', wt={self.weight_transform}' if self.weight_transform != 'identity' else ''
        return f"combine={self.combine}, agg={self.aggregate}, cmp={self.compare}{wt}"

    def as_dict(self) -> Dict[str, str]:
        return dict(combine=self.combine, aggregate=self.aggregate,
                    compare=self.compare, init_source=self.init_source,
                    init_other=self.init_other,
                    weight_transform=self.weight_transform)


# ---------------------------------------------------------------------------
# Init value defaults: principled choices based on operator semantics
# ---------------------------------------------------------------------------

COMBINE_INIT_SOURCE = {
    'add': '0',        # d + w = 0 + w = w (additive identity)
    'min': 'inf',      # min(d, w) = min(inf, w) = w (min identity)
    'max': '-inf',     # max(d, w) = max(-inf, w) = w (max identity)
    'multiply': '1',   # d * w = 1 * w = w (multiplicative identity)
}

COMPARE_INIT_OTHER = {
    '<': 'inf',        # Any real value is < inf (minimisation)
    '>': '-inf',       # Any real value is > -inf (maximisation)
}

WEIGHT_TRANSFORM_OPTIONS = ['identity', 'reciprocal', 'square', 'one_minus']


def make_variant(combine: str, aggregate: str, compare: str,
                 weight_transform: str = 'identity',
                 init_source: Optional[str] = None,
                 init_other: Optional[str] = None) -> VariantParams:
    """Create a variant with default init values if not specified."""
    if init_source is None:
        init_source = COMBINE_INIT_SOURCE[combine]
    if init_other is None:
        init_other = COMPARE_INIT_OTHER[compare]
    return VariantParams(combine, aggregate, compare, init_source, init_other,
                         weight_transform)


# ---------------------------------------------------------------------------
# All variants: {add, min, max, multiply} x {min, max} x {<, >}
#               x {identity, reciprocal, square, one_minus}
# ---------------------------------------------------------------------------

BASE_VARIANTS: Dict[str, VariantParams] = {}

for comb in ['add', 'min', 'max', 'multiply']:
    for agg in ['min', 'max']:
        for cmp in ['<', '>']:
            for wt in WEIGHT_TRANSFORM_OPTIONS:
                v = make_variant(comb, agg, cmp, wt)
                BASE_VARIANTS[v.short_name] = v

# Named aliases for well-known algorithms (identity transform only).
KNOWN_ALGORITHMS = {
    'bellman_ford_shortest': 'add_min_<',
    'longest_path': 'add_max_>',
    'widest_path': 'min_max_>',
    'narrowest_path': 'max_min_<',
    'most_reliable': 'multiply_max_>',
    'least_reliable': 'multiply_min_<',
}


def get_variant(name: str) -> VariantParams:
    """Look up a variant by short name or known algorithm name."""
    if name in KNOWN_ALGORITHMS:
        name = KNOWN_ALGORITHMS[name]
    return BASE_VARIANTS[name]


def get_all_variants() -> Dict[str, VariantParams]:
    """Return all variants."""
    return dict(BASE_VARIANTS)


# ---------------------------------------------------------------------------
# Viable variants (after Phase 0 curation with weight transforms)
# ---------------------------------------------------------------------------
#
# Curation on 64 candidates ({add,min,max,multiply} x {min,max} x {<,>}
# x {identity,reciprocal,square,one_minus}):
#   22 GOOD, 8 MARGINAL, 34 DEGENERATE from automated curation.
#
# Post-curation filtering:
#   Dropped add_max_> (all 4 transforms) — longest path via iterative
#     relaxation does NOT converge on cyclic graphs. Values grow each
#     iteration. converged_at=-1 in curation.
#   Dropped multiply_max_>_reciprocal — value explosion (products of 1/w
#     grow exponentially).
#   Dropped multiply_min_< (identity) — convergence detection issue
#     (products shrink below float64 exact-equality threshold). Recoverable
#     with approximate convergence, but low priority.
#   Dropped multiply_min_<_one_minus — borderline: low diversity (17.6%
#     unique d), 90% source-dependent (not 100%). Skip for cleanliness.
#   Dropped multiply_min_<_square — reclassified DEGENERATE in 64-variant
#     curation (unique_fraction=12.6%, LOW_DIVERSITY).
#
# Final: 16 GOOD + 8 MARGINAL = 24 viable.

VIABLE_VARIANTS: List[str] = [
    # GOOD (16)
    'add_min_<', 'add_min_<_one_minus', 'add_min_<_reciprocal', 'add_min_<_square',
    'max_min_<', 'max_min_<_one_minus', 'max_min_<_reciprocal', 'max_min_<_square',
    'min_max_>', 'min_max_>_one_minus', 'min_max_>_reciprocal', 'min_max_>_square',
    'multiply_max_>', 'multiply_max_>_one_minus', 'multiply_max_>_square',
    'multiply_min_<_reciprocal',
    # MARGINAL (8 — source-independent)
    'max_min_>', 'max_min_>_one_minus', 'max_min_>_reciprocal', 'max_min_>_square',
    'min_max_<', 'min_max_<_one_minus', 'min_max_<_reciprocal', 'min_max_<_square',
]

# ---------------------------------------------------------------------------
# Train / test split (4-axis compositional)
# ---------------------------------------------------------------------------
#
# 20 train + 4 test. Training includes 12 GOOD + 8 MARGINAL.
# Test variants cover all 4 combine operators, both directions, and
# both order types (preserving + reversing).
#
# Test variants — novel 4-tuples (each primitive seen individually in training):
#   add_min_<_reciprocal:         add + min/< + reciprocal (reversing)
#   max_min_<_square:             max + min/< + square (preserving)
#   min_max_>_reciprocal:         min + max/> + reciprocal (reversing, hard)
#   multiply_max_>_one_minus:     multiply + max/> + one_minus (reversing)
#
# Primitive coverage in training (20 variants):
#   combine:          add(3) max(7) min(6) multiply(3) — all 4 covered
#   aggregate:        min(10) max(10) — both covered
#   compare:          <(10) >(10) — both covered
#   weight_transform: identity(5) reciprocal(3) square(3) one_minus(5)
#
# Order-reversing coverage in GOOD training:
#   reciprocal + (min, <): max_min_<_reciprocal, multiply_min_<_reciprocal
#   one_minus  + (min, <): add_min_<_one_minus, max_min_<_one_minus
#   one_minus  + (max, >): min_max_>_one_minus
#   reciprocal + (max, >): NONE — held-out test combination

TRAIN_VARIANTS: List[str] = [
    # GOOD (12 — source-dependent)
    'add_min_<',                  # shortest path (identity)
    'add_min_<_square',           # shortest path (squared weights)
    'add_min_<_one_minus',        # shortest path (complement weights)
    'max_min_<',                  # minimax path (identity)
    'max_min_<_reciprocal',       # minimax path (reciprocal)
    'max_min_<_one_minus',        # minimax path (complement)
    'min_max_>',                  # widest path (identity)
    'min_max_>_square',           # widest path (squared)
    'min_max_>_one_minus',        # widest path (complement) — order-reversing + (max,>)
    'multiply_max_>',             # most-reliable path (identity)
    'multiply_max_>_square',      # most-reliable path (squared)
    'multiply_min_<_reciprocal',  # least-reliable path (reciprocal)
    # MARGINAL (8 — source-independent, adds primitive diversity)
    'max_min_>',                  # global min-maxweight (identity)
    'max_min_>_reciprocal',       # global min-maxweight (reciprocal)
    'max_min_>_square',           # global min-maxweight (squared)
    'max_min_>_one_minus',        # global min-maxweight (complement)
    'min_max_<',                  # global max-bottleneck (identity)
    'min_max_<_reciprocal',       # global max-bottleneck (reciprocal)
    'min_max_<_square',           # global max-bottleneck (squared)
    'min_max_<_one_minus',        # global max-bottleneck (complement)
]

TEST_VARIANTS: List[str] = [
    'add_min_<_reciprocal',       # add + min/< + reciprocal (reversing)
    'max_min_<_square',           # max + min/< + square (preserving)
    'min_max_>_reciprocal',       # min + max/> + reciprocal (reversing, hard)
    'multiply_max_>_one_minus',   # multiply + max/> + one_minus (reversing)
]


def finalize_split(viable: List[str], train: List[str], test: List[str]):
    """Override the default split after re-curation."""
    global VIABLE_VARIANTS, TRAIN_VARIANTS, TEST_VARIANTS
    VIABLE_VARIANTS = list(viable)
    TRAIN_VARIANTS = list(train)
    TEST_VARIANTS = list(test)
