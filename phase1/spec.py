"""Probe specification for parameterised relaxation variants.

Follows the CLRS convention: inputs are single snapshots (batch at axis 0),
hints are time series (time at axis 0, batch at axis 1), outputs are single
snapshots (batch at axis 0).

This spec is deliberately compatible with the baselines' CLRS format
(same input fields as BFS/Bellman-Ford/Dijkstra) so the CLRS model
architecture can be reused with minimal changes.
"""

from clrs._src import specs

Stage = specs.Stage
Location = specs.Location
Type = specs.Type


RELAXATION_SPEC = {
    # --- Inputs (identical to CLRS shortest-path algorithms) ---
    'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
    's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
    'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
    'adj': (Stage.INPUT, Location.EDGE, Type.MASK),

    # --- Hints (step-by-step trace) ---
    'd': (Stage.HINT, Location.NODE, Type.SCALAR),
    'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),

    # --- Output ---
    'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
}


# Inf proxy: finite value replacing inf/-inf in neural network inputs.
# Must be larger than any real distance (max ~160 for add_max_>_reciprocal
# at n=16) but small enough to not dominate the MSE loss.
# Previously 1e4, which caused training instability (MSE on 1e4 ≈ 1e8).
INF_PROXY = 200.0
