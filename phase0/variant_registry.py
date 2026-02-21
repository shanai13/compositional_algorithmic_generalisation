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

WEIGHT_TRANSFORM_OPTIONS = ['identity', 'reciprocal', 'square']


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
#               x {identity, reciprocal, square}
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
# Curation on 48 candidates ({add,min,max,multiply} x {min,max} x {<,>}
# x {identity,reciprocal,square}):
#   17 GOOD, 6 MARGINAL, 25 DEGENERATE from initial curation.
#
# Additional filtering (post-curation):
#   Dropped multiply_min_< (identity, square) — float32 underflow
#   Dropped multiply_max_>_reciprocal — value explosion
#   Dropped add_max_> (identity, reciprocal, square) — longest path via
#     iterative relaxation does NOT converge on cyclic graphs. Values grow
#     each iteration (cycles allow revisiting edges). First training run
#     confirmed: add_max_>_reciprocal reached only 20.2% accuracy while
#     other reciprocal variants hit 90%+, polluting the reciprocal signal
#     for test-time composition. converged_at=-1 in curation.
#
# Final: 12 GOOD + 6 MARGINAL = 18 viable.

VIABLE_VARIANTS: List[str] = [
    # GOOD (12)
    'add_min_<', 'add_min_<_reciprocal', 'add_min_<_square',
    'max_min_<', 'max_min_<_reciprocal', 'max_min_<_square',
    'min_max_>', 'min_max_>_reciprocal', 'min_max_>_square',
    'multiply_max_>', 'multiply_max_>_square',
    'multiply_min_<_reciprocal',
    # MARGINAL (6 — source-independent)
    'max_min_>', 'max_min_>_reciprocal', 'max_min_>_square',
    'min_max_<', 'min_max_<_reciprocal', 'min_max_<_square',
]

# ---------------------------------------------------------------------------
# Train / test split (4-axis compositional)
# ---------------------------------------------------------------------------
#
# 15 train + 3 test. Training includes 9 GOOD (source-dependent) + 6 MARGINAL
# (source-independent) variants. Test is 3 GOOD variants unchanged from v2.
#
# Why include MARGINAL variants? Two reasons:
# 1. Reciprocal diversity: the previous split only paired reciprocal with
#    (agg=min, cmp=<). MARGINAL variants max_min_>_reciprocal and
#    min_max_<_reciprocal add reciprocal+(min,>) and reciprocal+(max,<),
#    so the model sees reciprocal across all agg/cmp directions.
# 2. Source-independence is a learnable property: the model must detect from
#    conditioning traces that some variants ignore the source node.
#
# Test variants — novel 4-tuples (UNCHANGED from v2):
#   add_min_<_reciprocal:    add ✓ min ✓ < ✓ reciprocal ✓
#   max_min_<_square:        max ✓ min ✓ < ✓ square ✓
#   min_max_>_reciprocal:    min ✓ max ✓ > ✓ reciprocal ✓
#
# Primitive coverage in training:
#   combine:          add(2) max(5) min(5) multiply(3) — all 4 covered
#   aggregate:        min(8) max(7) — both covered
#   compare:          <(8) >(7) — both covered
#   weight_transform: identity(7) reciprocal(4) square(4) — all 3 covered
#
# Reciprocal coverage in training:
#   max_min_<_reciprocal   → reciprocal + (min, <)
#   multiply_min_<_reciprocal → reciprocal + (min, <)
#   max_min_>_reciprocal   → reciprocal + (min, >)  [NEW — MARGINAL]
#   min_max_<_reciprocal   → reciprocal + (max, <)  [NEW — MARGINAL]
#   → reciprocal now appears with all 4 (agg, cmp) combinations in training

TRAIN_VARIANTS: List[str] = [
    # GOOD (9 — source-dependent)
    'add_min_<',                # shortest path (identity)
    'add_min_<_square',         # shortest path (squared weights)
    'max_min_<',                # minimax path (identity)
    'max_min_<_reciprocal',     # minimax path (reciprocal)
    'min_max_>',                # widest path (identity)
    'min_max_>_square',         # widest path (squared)
    'multiply_max_>',           # most-reliable path (identity)
    'multiply_max_>_square',    # most-reliable path (squared)
    'multiply_min_<_reciprocal',  # least-reliable path (reciprocal)
    # MARGINAL (6 — source-independent, adds primitive diversity)
    'max_min_>',                # global min-maxweight (identity)
    'max_min_>_reciprocal',     # global min-maxweight (reciprocal) — reciprocal + >
    'max_min_>_square',         # global min-maxweight (squared)
    'min_max_<',                # global max-bottleneck (identity)
    'min_max_<_reciprocal',     # global max-bottleneck (reciprocal) — reciprocal + max
    'min_max_<_square',         # global max-bottleneck (squared)
]

TEST_VARIANTS: List[str] = [
    'add_min_<_reciprocal',     # novel: add+min+<+reciprocal
    'max_min_<_square',         # novel: max+min+<+square
    'min_max_>_reciprocal',     # novel: min+max+>+reciprocal
]


def finalize_split(viable: List[str], train: List[str], test: List[str]):
    """Override the default split after re-curation."""
    global VIABLE_VARIANTS, TRAIN_VARIANTS, TEST_VARIANTS
    VIABLE_VARIANTS = list(viable)
    TRAIN_VARIANTS = list(train)
    TEST_VARIANTS = list(test)
