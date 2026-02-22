"""Parameterised iterative relaxation algorithm template.

All algorithm variants share one computational template:

    d[source] = init_source;  d[other] = init_other;  pred[i] = i
    Repeat K iterations (synchronous updates):
        For each node v:
            messages = [combine(d[u], transform(w[u,v])) for u in neighbors(v)]
            best_msg = aggregate(messages)
            best_neighbor = argaggregate(messages)
            if compare(best_msg, d[v]):
                d[v] = best_msg
                pred[v] = best_neighbor
    Output: d (values), pred (predecessors)

Different parameter choices (combine, aggregate, compare, weight_transform,
init_source, init_other) recover different algorithms. The weight_transform
is applied internally — the trace stores original (untransformed) weights
as inputs, so the model must infer the transform from conditioning traces.
"""

from typing import NamedTuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Operator definitions
# ---------------------------------------------------------------------------

def _combine_add(d_u: np.ndarray, w_uv: np.ndarray) -> np.ndarray:
    return d_u + w_uv


def _combine_min(d_u: np.ndarray, w_uv: np.ndarray) -> np.ndarray:
    return np.minimum(d_u, w_uv)


def _combine_max(d_u: np.ndarray, w_uv: np.ndarray) -> np.ndarray:
    return np.maximum(d_u, w_uv)


def _combine_multiply(d_u: np.ndarray, w_uv: np.ndarray) -> np.ndarray:
    return d_u * w_uv


COMBINE_OPS = {
    'add': _combine_add,
    'min': _combine_min,
    'max': _combine_max,
    'multiply': _combine_multiply,
}

AGGREGATE_OPS = {
    'min': (np.min, np.argmin),
    'max': (np.max, np.argmax),
}

COMPARE_OPS = {
    '<': lambda new, old: new < old,
    '>': lambda new, old: new > old,
}

INIT_VALUES = {
    '0': 0.0,
    '1': 1.0,
    'inf': np.inf,
    '-inf': -np.inf,
}

# Weight transforms: applied to edge weights before the combine step.
# The graph input always stores the original (untransformed) weights.
WEIGHT_TRANSFORMS = {
    'identity': lambda w: w,
    'reciprocal': lambda w: 1.0 / np.maximum(w, 1e-8),  # 1/w
    'square': lambda w: w * w,                            # w^2
    'one_minus': lambda w: 1.0 - w,                       # 1-w
}


# ---------------------------------------------------------------------------
# Trace dataclass
# ---------------------------------------------------------------------------

class AlgorithmTrace(NamedTuple):
    """Full execution trace of a parameterised relaxation algorithm."""
    # Inputs
    adj: np.ndarray         # (n, n) binary adjacency matrix
    weights: np.ndarray     # (n, n) edge weights
    source: int             # source node index

    # Hints at each step (length K+1, including initial state)
    d_hints: np.ndarray     # (K+1, n) distance/value at each step
    pred_hints: np.ndarray  # (K+1, n) predecessor pointers at each step

    # Outputs (final state)
    d_final: np.ndarray     # (n,) final distance/value
    pred_final: np.ndarray  # (n,) final predecessor pointers

    # Metadata
    num_steps: int          # K (number of relaxation iterations)
    converged_at: int       # step at which values stopped changing (-1 if didn't)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def parameterized_relaxation(
    adj: np.ndarray,
    weights: np.ndarray,
    source: int,
    combine: str,
    aggregate: str,
    compare: str,
    init_source: str,
    init_other: str,
    weight_transform: str = 'identity',
    max_steps: Optional[int] = None,
) -> AlgorithmTrace:
    """Run parameterised iterative relaxation on a weighted graph.

    Uses synchronous (parallel) updates: all nodes read from the previous
    iteration's values, not the current iteration's partially-updated values.

    The weight_transform is applied internally to edge weights before the
    combine step. The trace stores the ORIGINAL (untransformed) weights as
    inputs, so the model must infer the transform from conditioning traces.

    Args:
        adj: (n, n) binary adjacency matrix (1 = edge exists).
        weights: (n, n) edge weights (meaningful only where adj=1).
        source: source node index.
        combine: 'add', 'min', 'max', or 'multiply'.
        aggregate: 'min' or 'max'.
        compare: '<' or '>'.
        init_source: '0', '1', 'inf', or '-inf'.
        init_other: '0', '1', 'inf', or '-inf'.
        weight_transform: 'identity', 'reciprocal', or 'square'.
        max_steps: max iterations (default: n-1, per CLRS convention).

    Returns:
        AlgorithmTrace with full execution trace (original weights in inputs).
    """
    n = adj.shape[0]
    if max_steps is None:
        max_steps = n - 1

    combine_fn = COMBINE_OPS[combine]
    agg_fn, agg_argfn = AGGREGATE_OPS[aggregate]
    compare_fn = COMPARE_OPS[compare]
    init_s = INIT_VALUES[init_source]
    init_o = INIT_VALUES[init_other]
    w_transform = WEIGHT_TRANSFORMS[weight_transform]

    # Apply weight transform (used internally, not stored in trace).
    w_effective = w_transform(weights)

    # Precompute neighbor lists for efficiency.
    neighbors = [np.where(adj[:, v] > 0)[0] for v in range(n)]

    # Initialise.
    d = np.full(n, init_o, dtype=np.float64)
    d[source] = init_s
    pred = np.arange(n, dtype=np.int32)

    d_history = [d.copy()]
    pred_history = [pred.copy()]

    converged_at = -1

    for step in range(max_steps):
        d_new = d.copy()
        pred_new = pred.copy()

        for v in range(n):
            nbrs = neighbors[v]
            if len(nbrs) == 0:
                continue

            # Compute messages from all neighbors (vectorised).
            # Uses transformed weights internally.
            messages = combine_fn(d[nbrs], w_effective[nbrs, v])

            # Handle inf/nan: if all messages are inf/-inf, skip update.
            finite_mask = np.isfinite(messages)
            if not np.any(finite_mask):
                continue

            # Aggregate over neighbor messages.
            agg_msg = agg_fn(messages)
            best_idx = agg_argfn(messages)
            best_neighbor = nbrs[best_idx]

            # Compare and conditionally update.
            if np.isfinite(agg_msg) and compare_fn(agg_msg, d[v]):
                d_new[v] = agg_msg
                pred_new[v] = best_neighbor

        # Check convergence.
        if np.array_equal(d_new, d) and np.array_equal(pred_new, pred):
            if converged_at == -1:
                converged_at = step
            d_history.append(d_new.copy())
            pred_history.append(pred_new.copy())
            # Pad remaining steps with converged state.
            for _ in range(step + 1, max_steps):
                d_history.append(d_new.copy())
                pred_history.append(pred_new.copy())
            break

        d = d_new
        pred = pred_new
        d_history.append(d.copy())
        pred_history.append(pred.copy())

    return AlgorithmTrace(
        adj=adj,
        weights=weights,
        source=source,
        d_hints=np.array(d_history),
        pred_hints=np.array(pred_history),
        d_final=d.copy(),
        pred_final=pred.copy(),
        num_steps=max_steps,
        converged_at=converged_at,
    )


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_er_graph(
    n: int,
    p: float = 0.3,
    weight_range: tuple = (0.1, 1.0),
    rng: Optional[np.random.RandomState] = None,
    ensure_connected: bool = True,
    max_attempts: int = 100,
) -> tuple:
    """Generate an Erdos-Renyi random graph with random positive weights.

    Args:
        n: number of nodes.
        p: edge probability.
        weight_range: (low, high) for uniform edge weights.
        rng: random state (created if None).
        ensure_connected: if True, regenerate until graph is connected.
        max_attempts: max regeneration attempts for connectivity.

    Returns:
        (adj, weights) where adj is (n,n) binary and weights is (n,n) float.
    """
    if rng is None:
        rng = np.random.RandomState()

    for attempt in range(max_attempts):
        # Upper triangular random edges, then symmetrise (undirected).
        mask = rng.random((n, n)) < p
        mask = np.triu(mask, k=1)
        adj = (mask | mask.T).astype(np.float64)

        if ensure_connected and not _is_connected(adj, n):
            continue

        # Random symmetric weights.
        raw = rng.uniform(weight_range[0], weight_range[1], size=(n, n))
        raw = np.triu(raw, k=1)
        weights = (raw + raw.T) * adj
        return adj, weights

    # Fallback: force connectivity by adding a spanning path.
    order = rng.permutation(n)
    for i in range(n - 1):
        adj[order[i], order[i + 1]] = 1.0
        adj[order[i + 1], order[i]] = 1.0
    raw = rng.uniform(weight_range[0], weight_range[1], size=(n, n))
    raw = np.triu(raw, k=1)
    weights = (raw + raw.T) * adj
    return adj, weights


def _is_connected(adj: np.ndarray, n: int) -> bool:
    """Check graph connectivity via BFS."""
    visited = set()
    queue = [0]
    visited.add(0)
    while queue:
        u = queue.pop(0)
        for v in range(n):
            if adj[u, v] > 0 and v not in visited:
                visited.add(v)
                queue.append(v)
    return len(visited) == n
