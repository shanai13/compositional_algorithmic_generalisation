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


# Inf proxy: large finite value replacing inf/-inf in neural network inputs.
# Chosen to be much larger than any real distance (max ~n*max_weight ≈ 16*1.0)
# but not so large as to cause numerical issues in float32.
INF_PROXY = 1e4
