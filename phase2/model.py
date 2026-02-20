"""Conditioned model: conditioning encoder + z-augmented CLRS processor.

Architecture overview:
    Conditioning examples (k traces of the same variant on different graphs)
      -> ConditioningEncoder (small GNN, mean pool) -> task embedding z

    Query graph
      -> Standard CLRS encoder -> (node_fts, edge_fts, graph_fts, adj_mat)
      -> graph_fts += Linear(z)   [z injection]
      -> CLRS processor (message passing)
      -> CLRS decoder -> (distances, predecessors)

The only architectural change from standard CLRS is z injection into graph_fts.
Everything else (encoder, processor, decoder, loss, hint supervision) is reused.
"""

from typing import Dict, List, Optional, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from clrs._src import decoders
from clrs._src import encoders
from clrs._src import losses
from clrs._src import nets
from clrs._src import probing
from clrs._src import processors
from clrs._src import samplers
from clrs._src import specs

from phase1.spec import INF_PROXY

_Array = chex.Array
_DataPoint = probing.DataPoint
_Features = samplers.Features
_Feedback = samplers.Feedback
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Type = specs.Type


# ---------------------------------------------------------------------------
# Conditioning Encoder
# ---------------------------------------------------------------------------

class ConditioningEncoder(hk.Module):
    """Encodes k conditioning traces into a single task embedding z.

    For each conditioning example:
      1. Extract node features: source indicator, initial/final/delta distances
      2. Extract edge features: weights, adjacency, predecessor tree edges
      3. Run a small GNN (2-3 message passing layers)
      4. Mean pool over nodes -> per-example vector e_j

    Mean pool across k examples -> z.
    """

    def __init__(self, hidden_dim: int = 64, z_dim: int = 128,
                 nb_layers: int = 2, name: str = 'cond_encoder'):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.nb_layers = nb_layers

    def __call__(self, cond_features: _Features) -> _Array:
        """Encode conditioning examples into task embedding z.

        Args:
            cond_features: Features with batch_size=k (k conditioning examples).
                inputs: [pos(k,n), s(k,n), A(k,n,n), adj(k,n,n)]
                hints: [d(T,k,n), pi_h(T,k,n)]

        Returns:
            z: (z_dim,) task embedding vector.
        """
        # Extract raw data from DataPoints.
        inp_map = {dp.name: dp.data for dp in cond_features.inputs}
        hint_map = {dp.name: dp.data for dp in cond_features.hints}

        s = inp_map['s']          # (k, n) source indicator
        A = inp_map['A']          # (k, n, n) edge weights
        adj = inp_map['adj']      # (k, n, n) adjacency

        d_hints = hint_map['d']   # (T, k, n) distance values over time
        pi_hints = hint_map['pi_h']  # (T, k, n) predecessor pointers over time

        k = s.shape[0]
        n = s.shape[1]

        # Build node features: [s, d_init, d_final, d_delta] -> (k, n, 4)
        d_init = d_hints[0]       # (k, n)
        d_final = d_hints[-1]     # (k, n)
        d_delta = d_final - d_init  # (k, n)
        node_fts = jnp.stack([s, d_init, d_final, d_delta], axis=-1)

        # Build predecessor edge indicator from final predecessors.
        # pi_final[b, v] = predecessor of node v in example b.
        # pred_edge[b, j, v] = 1 if j is the predecessor of v.
        pi_final = pi_hints[-1].astype(jnp.int32)  # (k, n)
        pred_onehot = jax.nn.one_hot(pi_final, n)  # (k, n, n) where [b,v,:] is one-hot at pi[b,v]
        pred_edge = jnp.transpose(pred_onehot, (0, 2, 1))  # (k, n, n) where [b,j,v]=1 if j=pred(v)

        # Edge features: [weights, adjacency, predecessor] -> (k, n, n, 3)
        edge_fts = jnp.stack([A, adj, pred_edge], axis=-1)

        # Project to hidden dim.
        node_fts = hk.Linear(self.hidden_dim, name='node_in')(node_fts)
        edge_fts = hk.Linear(self.hidden_dim, name='edge_in')(edge_fts)

        # Adjacency with self-loops for message passing.
        adj_self = adj + jnp.eye(n)

        # Small GNN: message passing layers.
        for layer in range(self.nb_layers):
            # Messages from sender u to receiver v: h[u] + e[u,v].
            msgs = jnp.expand_dims(node_fts, 2) + edge_fts  # (k, n_u, n_v, h)

            # Mask by adjacency (with self-loops).
            mask = jnp.expand_dims(adj_self, -1)  # (k, n, n, 1)
            msgs = msgs * mask

            # Mean aggregation over senders (axis=1) for each receiver v.
            in_degree = jnp.maximum(
                jnp.sum(adj_self, axis=1), 1.0)  # (k, n)
            msg_agg = jnp.sum(msgs, axis=1) / in_degree[..., None]  # (k, n, h)

            # Update: concatenate old features + aggregated messages, project.
            node_fts = jax.nn.relu(hk.Linear(
                self.hidden_dim, name=f'mp_{layer}')(
                jnp.concatenate([node_fts, msg_agg], axis=-1)))

        # Mean pool over nodes -> per-example embedding.
        e_j = jnp.mean(node_fts, axis=1)  # (k, hidden_dim)

        # Project to z_dim.
        e_j = hk.Linear(self.z_dim, name='to_z')(e_j)  # (k, z_dim)

        # Mean pool over k conditioning examples -> task embedding.
        z = jnp.mean(e_j, axis=0)  # (z_dim,)

        return z


# ---------------------------------------------------------------------------
# Conditioned Net (subclass of CLRS Net)
# ---------------------------------------------------------------------------

class ConditionedNet(nets.Net):
    """CLRS Net with task embedding z injected into graph features.

    Overrides _one_step_pred to add z to graph_fts between encoding
    and processing. All other CLRS machinery (hint loop, teacher forcing,
    scanning) is inherited unchanged.
    """

    def __init__(self, *args, z_dim: int = 128, cond_hidden_dim: int = 64,
                 cond_nb_layers: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_dim = z_dim
        self.cond_hidden_dim = cond_hidden_dim
        self.cond_nb_layers = cond_nb_layers
        self._z = None  # Set before forward pass.

    def __call__(self, features_list, repred, algorithm_index,
                 return_hints, return_all_outputs, cond_features=None):
        """Forward pass with optional conditioning.

        Args:
            features_list: Query features (same as parent Net).
            repred: True at eval (use own predictions), False at train.
            algorithm_index: Which algorithm (0 for us, -1 for init).
            return_hints: Whether to return hint predictions.
            return_all_outputs: Whether to return all timestep outputs.
            cond_features: Optional Features for conditioning (batch_size=k).
                If provided, computes z and injects into processing.

        Returns:
            (output_preds, hint_preds) same as parent.
        """
        if cond_features is not None:
            encoder = ConditioningEncoder(
                hidden_dim=self.cond_hidden_dim,
                z_dim=self.z_dim,
                nb_layers=self.cond_nb_layers,
            )
            self._z = encoder(cond_features)
        else:
            self._z = None

        return super().__call__(
            features_list, repred, algorithm_index,
            return_hints, return_all_outputs)

    def _one_step_pred(self, inputs, hints, hidden, batch_size, nb_nodes,
                       lstm_state, spec, encs, decs, repred):
        """One step of encode -> process -> decode, with z injection."""

        # ENCODE (identical to parent).
        node_fts = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))
        edge_fts = jnp.zeros((batch_size, nb_nodes, nb_nodes, self.hidden_dim))
        graph_fts = jnp.zeros((batch_size, self.hidden_dim))
        adj_mat = jnp.repeat(
            jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)

        trajectories = [inputs]
        if self.encode_hints:
            trajectories.append(hints)

        for trajectory in trajectories:
            for dp in trajectory:
                try:
                    dp = encoders.preprocess(dp, nb_nodes)
                    assert dp.type_ != _Type.SOFT_POINTER
                    adj_mat = encoders.accum_adj_mat(dp, adj_mat)
                    encoder = encs[dp.name]
                    edge_fts = encoders.accum_edge_fts(encoder, dp, edge_fts)
                    node_fts = encoders.accum_node_fts(encoder, dp, node_fts)
                    graph_fts = encoders.accum_graph_fts(encoder, dp, graph_fts)
                except Exception as e:
                    raise Exception(f'Failed to process {dp}') from e

        # Z INJECTION: add task embedding to graph features.
        if self._z is not None:
            z = self._z
            # Broadcast z (z_dim,) -> (batch_size, z_dim).
            if z.ndim == 1:
                z = jnp.broadcast_to(z[None, :], (batch_size, z.shape[0]))
            # Project z to hidden_dim and add to graph_fts.
            z_proj = hk.Linear(self.hidden_dim, name='z_to_graph')(z)
            graph_fts = graph_fts + z_proj

        # PROCESS (identical to parent).
        nxt_hidden = hidden
        for _ in range(self.nb_msg_passing_steps):
            nxt_hidden, nxt_edge = self.processor(
                node_fts, edge_fts, graph_fts, adj_mat, nxt_hidden,
                batch_size=batch_size, nb_nodes=nb_nodes)

        if not repred:
            nxt_hidden = hk.dropout(
                hk.next_rng_key(), self._dropout_prob, nxt_hidden)

        if self.use_lstm:
            nxt_hidden, nxt_lstm_state = jax.vmap(self.lstm)(
                nxt_hidden, lstm_state)
        else:
            nxt_lstm_state = None

        h_t = jnp.concatenate([node_fts, hidden, nxt_hidden], axis=-1)
        if nxt_edge is not None:
            e_t = jnp.concatenate([edge_fts, nxt_edge], axis=-1)
        else:
            e_t = edge_fts

        # DECODE (identical to parent).
        hint_preds, output_preds = decoders.decode_fts(
            decoders=decs, spec=spec, h_t=h_t, adj_mat=adj_mat,
            edge_fts=e_t, graph_fts=graph_fts,
            inf_bias=self.processor.inf_bias,
            inf_bias_edge=self.processor.inf_bias_edge,
            repred=repred)

        return nxt_hidden, output_preds, hint_preds, nxt_lstm_state


# ---------------------------------------------------------------------------
# Conditioned Model (training wrapper)
# ---------------------------------------------------------------------------

def _masked_d_hint_loss(truth, preds, lengths):
    """MSE loss on d hints, masking unreachable nodes.

    Standard CLRS hint_loss applies MSE to ALL nodes including those at
    ±INF_PROXY (unreachable). This overwhelms the loss with useless signal.
    This function masks those nodes so the model only learns from meaningful
    distance values.

    Args:
        truth: DataPoint with shape (T+1, B, n) for d hints.
        preds: List of T predictions, each (B, n).
        lengths: (B,) actual number of active timesteps.

    Returns:
        Scalar loss.
    """
    truth_data = truth.data[1:]           # (T, B, n) — skip initial state
    pred_stack = jnp.stack(preds)         # (T, B, n)

    # Per-element MSE.
    loss = (pred_stack - truth_data) ** 2  # (T, B, n)

    # Mask: 1 for reachable nodes (|d| below threshold), 0 for unreachable.
    threshold = INF_PROXY * 0.95
    reachable = (jnp.abs(truth_data) < threshold).astype(jnp.float32)

    # Timestep activity mask (from CLRS lengths convention).
    T = truth_data.shape[0]
    time_idx = jnp.arange(T)[:, None]     # (T, 1)
    time_mask = (lengths[None, :] > time_idx + 1).astype(jnp.float32)
    time_mask = jnp.expand_dims(time_mask, -1)  # (T, B, 1)

    # Combined mask.
    mask = reachable * time_mask

    return jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1e-8)


class ConditionedModel:
    """Complete conditioned model with training, prediction, and checkpointing.

    Similar to clrs.models.BaselineModel but uses ConditionedNet with
    task embedding z from conditioning examples.
    """

    def __init__(
        self,
        spec: _Spec,
        dummy_trajectory: _Feedback,
        processor_factory: processors.ProcessorFactory,
        hidden_dim: int = 128,
        z_dim: int = 128,
        cond_hidden_dim: int = 64,
        cond_nb_layers: int = 2,
        encode_hints: bool = True,
        decode_hints: bool = True,
        encoder_init: str = 'xavier_on_scalars',
        use_lstm: bool = False,
        learning_rate: float = 0.001,
        grad_clip_max_norm: float = 1.0,
        dropout_prob: float = 0.0,
        hint_teacher_forcing: float = 0.0,
        hint_repred_mode: str = 'soft',
        nb_msg_passing_steps: int = 1,
        checkpoint_path: str = 'checkpoints/conditioned',
    ):
        self._spec = [spec]  # List of specs (one for our single "algorithm").
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.encode_hints = encode_hints
        self.decode_hints = decode_hints

        # Compute nb_dims from dummy trajectory (needed for decoder construction).
        self._nb_dims = []
        dummy_spec = {k: v for k, v in spec.items()}
        nb_dims_dict = {}
        for inp in dummy_trajectory.features.inputs:
            nb_dims_dict[inp.name] = inp.data.shape[-1] if inp.data.ndim > 2 else 0
        for hint in dummy_trajectory.features.hints:
            nb_dims_dict[hint.name] = hint.data.shape[-1] if hint.data.ndim > 3 else 0
        for out in dummy_trajectory.outputs:
            nb_dims_dict[out.name] = out.data.shape[-1] if out.data.ndim > 2 else 0
        self._nb_dims = [nb_dims_dict]

        # Build Haiku function.
        def _use_net(features_list, repred, algorithm_index,
                     return_hints, return_all_outputs, cond_features=None):
            net = ConditionedNet(
                spec=self._spec,
                hidden_dim=hidden_dim,
                encode_hints=encode_hints,
                decode_hints=decode_hints,
                processor_factory=processor_factory,
                use_lstm=use_lstm,
                encoder_init=encoder_init,
                dropout_prob=dropout_prob,
                hint_teacher_forcing=hint_teacher_forcing,
                hint_repred_mode=hint_repred_mode,
                nb_dims=self._nb_dims,
                nb_msg_passing_steps=nb_msg_passing_steps,
                z_dim=z_dim,
                cond_hidden_dim=cond_hidden_dim,
                cond_nb_layers=cond_nb_layers,
            )
            return net(features_list, repred, algorithm_index,
                       return_hints, return_all_outputs,
                       cond_features=cond_features)

        self.net_fn = hk.transform(_use_net)

        # Optimizer.
        self.opt = optax.chain(
            optax.clip_by_global_norm(grad_clip_max_norm),
            optax.adam(learning_rate),
        )

        self.params = None
        self.opt_state = None
        self.checkpoint_path = checkpoint_path

        # JIT-compiled functions.
        self._jitted_grad = jax.jit(self._compute_grad)
        self._jitted_predict = jax.jit(self._predict)

    def init(self, query_features: _Features, cond_features: _Features,
             seed: int = 42):
        """Initialize model parameters.

        Args:
            query_features: Dummy query features for shape inference.
            cond_features: Dummy conditioning features for shape inference.
            seed: Random seed.
        """
        rng = jax.random.PRNGKey(seed)
        self.params = self.net_fn.init(
            rng,
            [query_features],  # features_list
            True,              # repred
            -1,                # algorithm_index (-1 = init all)
            False,             # return_hints
            False,             # return_all_outputs
            cond_features,     # cond_features
        )
        self.opt_state = self.opt.init(self.params)

    def _compute_loss(self, params, rng_key, query_feedback: _Feedback,
                      cond_features: _Features) -> _Array:
        """Compute training loss."""
        output_preds, hint_preds = self.net_fn.apply(
            params, rng_key,
            [query_feedback.features],  # features_list
            False,                      # repred (training mode)
            0,                          # algorithm_index
            True,                       # return_hints
            False,                      # return_all_outputs
            cond_features,              # cond_features
        )

        nb_nodes = query_feedback.features.inputs[0].data.shape[1]
        total_loss = 0.0

        # Output loss.
        for truth in query_feedback.outputs:
            loss = losses.output_loss(
                truth=truth,
                pred=output_preds[truth.name],
                nb_nodes=nb_nodes,
            )
            total_loss += loss

        # Hint loss.
        if self.decode_hints and hint_preds is not None:
            lengths = query_feedback.features.lengths
            for truth in query_feedback.features.hints:
                if truth.name == 'd':
                    # Custom masked loss for distance hints: ignore
                    # unreachable nodes (those at ±INF_PROXY).
                    loss = _masked_d_hint_loss(
                        truth, [hp['d'] for hp in hint_preds], lengths)
                else:
                    loss = losses.hint_loss(
                        truth=truth,
                        preds=[hp[truth.name] for hp in hint_preds],
                        lengths=lengths,
                        nb_nodes=nb_nodes,
                    )
                total_loss += loss

        return total_loss

    def _compute_grad(self, params, rng_key, query_feedback, cond_features):
        """Compute loss and gradients."""
        loss, grads = jax.value_and_grad(self._compute_loss)(
            params, rng_key, query_feedback, cond_features)
        return loss, grads

    def feedback(self, rng_key: _Array, query_feedback: _Feedback,
                 cond_features: _Features) -> float:
        """One training step: compute loss, gradients, update params.

        Args:
            rng_key: JAX random key.
            query_feedback: Query Feedback (what to predict).
            cond_features: Conditioning Features (what to read).

        Returns:
            Training loss (scalar).
        """
        loss, grads = self._jitted_grad(
            self.params, rng_key, query_feedback, cond_features)

        updates, self.opt_state = self.opt.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss

    def _predict(self, params, rng_key, query_features, cond_features):
        """Run prediction (no gradient)."""
        output_preds, hint_preds = self.net_fn.apply(
            params, rng_key,
            [query_features],
            True,    # repred (inference mode)
            0,       # algorithm_index
            False,   # return_hints
            False,   # return_all_outputs
            cond_features,
        )
        return output_preds, hint_preds

    def predict(self, rng_key: _Array, query_features: _Features,
                cond_features: _Features):
        """Run inference.

        Args:
            rng_key: JAX random key.
            query_features: Query features to predict on.
            cond_features: Conditioning features (k example traces).

        Returns:
            (output_preds, hint_preds) dictionaries.
        """
        return self._jitted_predict(
            self.params, rng_key, query_features, cond_features)

    def save_model(self, filename: str):
        """Save model parameters."""
        import pickle
        import os
        os.makedirs(self.checkpoint_path, exist_ok=True)
        path = os.path.join(self.checkpoint_path, filename)
        with open(path, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'opt_state': self.opt_state,
            }, f)

    def restore_model(self, filename: str):
        """Restore model parameters."""
        import pickle
        import os
        path = os.path.join(self.checkpoint_path, filename)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.params = data['params']
        self.opt_state = data['opt_state']
