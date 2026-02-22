"""Sampler producing CLRS-format Feedback from parameterised relaxation.

Converts AlgorithmTrace objects into the standard CLRS data format
(Features with DataPoint lists) so the CLRS model infrastructure can
be used directly.
"""

from typing import Iterator, List, Optional, Tuple

import numpy as np

from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs

from phase0.parameterized_relaxation import (
    AlgorithmTrace,
    generate_er_graph,
    parameterized_relaxation,
)
from phase0.variant_registry import BASE_VARIANTS, VariantParams
from phase1.spec import INF_PROXY, RELAXATION_SPEC

Stage = specs.Stage
Location = specs.Location
Type = specs.Type

DataPoint = probing.DataPoint
Feedback = samplers.Feedback
Features = samplers.Features


def _clip_inf(arr: np.ndarray) -> np.ndarray:
    """Replace inf/-inf with finite proxy values.

    Unreachable nodes are clipped to ±INF_PROXY. Values are kept at
    natural scale (not normalized) so the d hint MSE loss has full
    gradient magnitude. Normalization for Xavier-init Linear layers
    is handled locally where needed (e.g., inside the conditioning
    encoder).
    """
    out = arr.copy()
    out[out == np.inf] = INF_PROXY
    out[out == -np.inf] = -INF_PROXY
    return out


def trace_to_feedback(
    trace: AlgorithmTrace,
    n: int,
    randomize_pos: bool = True,
    rng: Optional[np.random.RandomState] = None,
) -> Feedback:
    """Convert a single AlgorithmTrace to CLRS Feedback (unbatched).

    Returns Feedback where all arrays have a leading batch dimension of 1.

    Args:
        trace: AlgorithmTrace from parameterized_relaxation.
        n: number of nodes.
        randomize_pos: whether to randomize positional encoding.
        rng: random state for positional encoding.

    Returns:
        Feedback with batch_size=1.
    """
    if rng is None:
        rng = np.random.RandomState()

    # --- Inputs (batch at axis 0) ---

    # Positional encoding: random permutation of linspace or random uniform.
    if randomize_pos:
        pos = rng.uniform(size=(1, n)).astype(np.float32)
    else:
        pos = np.linspace(0, 1, n, dtype=np.float32).reshape(1, n)

    # Source: one-hot (MASK_ONE).
    s = np.zeros((1, n), dtype=np.float32)
    s[0, trace.source] = 1.0

    # Edge weights (SCALAR).
    A = trace.weights.reshape(1, n, n).astype(np.float32)

    # Adjacency (MASK).
    adj = trace.adj.reshape(1, n, n).astype(np.float32)

    inputs = [
        DataPoint(name='pos', location=Location.NODE, type_=Type.SCALAR,
                  data=pos),
        DataPoint(name='s', location=Location.NODE, type_=Type.MASK_ONE,
                  data=s),
        DataPoint(name='A', location=Location.EDGE, type_=Type.SCALAR,
                  data=A),
        DataPoint(name='adj', location=Location.EDGE, type_=Type.MASK,
                  data=adj),
    ]

    # --- Hints (time at axis 0, batch at axis 1) ---

    # d_hints: (T, n) -> (T, 1, n), clipping inf.
    T = trace.d_hints.shape[0]
    d_data = _clip_inf(trace.d_hints).reshape(T, 1, n).astype(np.float32)

    # pi_h: (T, n) -> (T, 1, n), integer predecessor pointers.
    pi_h_data = trace.pred_hints.reshape(T, 1, n).astype(np.float32)

    hints = [
        DataPoint(name='d', location=Location.NODE, type_=Type.SCALAR,
                  data=d_data),
        DataPoint(name='pi_h', location=Location.NODE, type_=Type.POINTER,
                  data=pi_h_data),
    ]

    # Lengths: actual number of hint steps (includes initial state).
    lengths = np.array([T], dtype=np.float32)

    # --- Output (batch at axis 0) ---

    pi_data = trace.pred_final.reshape(1, n).astype(np.float32)

    outputs = [
        DataPoint(name='pi', location=Location.NODE, type_=Type.POINTER,
                  data=pi_data),
    ]

    return Feedback(
        features=Features(inputs=inputs, hints=hints, lengths=lengths),
        outputs=outputs,
    )


def _stack_feedbacks(feedbacks: List[Feedback]) -> Feedback:
    """Stack multiple single-sample Feedbacks into a batched Feedback.

    Each input Feedback should have batch_size=1.
    """
    B = len(feedbacks)
    if B == 0:
        raise ValueError('Cannot stack empty list of feedbacks')

    # Stack inputs: concatenate along batch axis (0 for inputs).
    stacked_inputs = []
    for i, dp in enumerate(feedbacks[0].features.inputs):
        data = np.concatenate([fb.features.inputs[i].data for fb in feedbacks],
                              axis=0)
        stacked_inputs.append(DataPoint(
            name=dp.name, location=dp.location, type_=dp.type_, data=data))

    # Stack hints: need to pad to max time length, batch axis is 1.
    max_T = max(fb.features.hints[0].data.shape[0] for fb in feedbacks)
    stacked_hints = []
    for i, dp in enumerate(feedbacks[0].features.hints):
        padded = []
        for fb in feedbacks:
            hint_data = fb.features.hints[i].data  # (T_j, 1, ...)
            T_j = hint_data.shape[0]
            if T_j < max_T:
                # Pad with last value repeated.
                pad_shape = list(hint_data.shape)
                pad_shape[0] = max_T - T_j
                last_val = np.broadcast_to(
                    hint_data[-1:], pad_shape)
                hint_data = np.concatenate([hint_data, last_val], axis=0)
            padded.append(hint_data)
        data = np.concatenate(padded, axis=1)  # (max_T, B, ...)
        stacked_hints.append(DataPoint(
            name=dp.name, location=dp.location, type_=dp.type_, data=data))

    # Stack lengths.
    lengths = np.concatenate(
        [fb.features.lengths for fb in feedbacks], axis=0)

    # Stack outputs.
    stacked_outputs = []
    for i, dp in enumerate(feedbacks[0].outputs):
        data = np.concatenate([fb.outputs[i].data for fb in feedbacks],
                              axis=0)
        stacked_outputs.append(DataPoint(
            name=dp.name, location=dp.location, type_=dp.type_, data=data))

    return Feedback(
        features=Features(
            inputs=stacked_inputs, hints=stacked_hints, lengths=lengths),
        outputs=stacked_outputs,
    )


class RelaxationSampler:
    """Generates batched CLRS-format Feedback for a single variant.

    Each call to next() generates batch_size random graphs, runs the
    variant on each, and returns a batched Feedback.
    """

    def __init__(
        self,
        variant: VariantParams,
        n: int = 16,
        p: float = 0.3,
        weight_range: tuple = (0.1, 1.0),
        seed: int = 42,
        randomize_pos: bool = True,
    ):
        self.variant = variant
        self.n = n
        self.p = p
        self.weight_range = weight_range
        self.rng = np.random.RandomState(seed)
        self.randomize_pos = randomize_pos

    def _generate_one(self) -> Feedback:
        """Generate a single trace and convert to Feedback."""
        adj, weights = generate_er_graph(
            self.n, p=self.p, weight_range=self.weight_range,
            rng=self.rng, ensure_connected=True,
        )
        source = self.rng.randint(self.n)
        trace = parameterized_relaxation(
            adj, weights, source,
            combine=self.variant.combine,
            aggregate=self.variant.aggregate,
            compare=self.variant.compare,
            init_source=self.variant.init_source,
            init_other=self.variant.init_other,
            weight_transform=self.variant.weight_transform,
        )
        return trace_to_feedback(
            trace, self.n, randomize_pos=self.randomize_pos, rng=self.rng)

    def next(self, batch_size: int) -> Feedback:
        """Generate a batch of traces."""
        feedbacks = [self._generate_one() for _ in range(batch_size)]
        return _stack_feedbacks(feedbacks)

    def __iter__(self) -> Iterator[Feedback]:
        """Infinite iterator (call with batch_size via next())."""
        return self

    def __next__(self) -> Feedback:
        raise TypeError(
            'Use sampler.next(batch_size) instead of next(sampler)')
