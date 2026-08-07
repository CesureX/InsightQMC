import math
from typing import Callable, List, Sequence, Union

import numpy as np
from jax import numpy as jnp

from flax import nnx

from ..layers import get_layer
from ..layers.utils import adam_transition


class MultKAN(nnx.Module):
    """
    KAN model with multiplication nodes inspired by pykan's MultKAN.

    Width format:
    - Plain int: number of additive nodes, e.g. [2, 8, 1]
    - Pair [n_sum, n_mult]: additive and multiplicative nodes for a layer,
      e.g. [2, [5, 3], 1]
    """

    def __init__(
        self,
        width: Sequence[Union[int, Sequence[int]]],
        layer_type: str = "base",
        required_parameters: Union[None, dict] = None,
        mult_arity: Union[int, Sequence[Sequence[int]]] = 2,
        affine_trainable: bool = False,
        seed: int = 42,
    ):
        del affine_trainable

        object.__setattr__(self, "layer_type", layer_type.lower())
        object.__setattr__(self, "required_parameters", dict(required_parameters or {}))
        object.__setattr__(self, "seed", int(seed))

        LayerClass = get_layer(self.layer_type)

        if required_parameters is None:
            raise ValueError(
                "required_parameters must be provided as a dictionary for the selected layer_type."
            )

        self.width = [
            [int(item), 0] if isinstance(item, int) else [int(item[0]), int(item[1])]
            for item in width
        ]
        self.depth = len(self.width) - 1

        if isinstance(mult_arity, int):
            self.mult_homo = True
        else:
            self.mult_homo = False
        self.mult_arity = mult_arity

        self.layers = nnx.List(
            [
                LayerClass(
                    n_in=self.width_in[i],
                    n_out=self.width_out[i + 1],
                    **required_parameters,
                    seed=seed + i,
                )
                for i in range(self.depth)
            ]
        )

        self.node_bias = nnx.List(
            [nnx.Param(jnp.zeros((self.width_in[i + 1],))) for i in range(self.depth)]
        )
        self.node_scale = nnx.List(
            [nnx.Param(jnp.ones((self.width_in[i + 1],))) for i in range(self.depth)]
        )
        self.subnode_bias = nnx.List(
            [nnx.Param(jnp.zeros((self.width_out[i + 1],))) for i in range(self.depth)]
        )
        self.subnode_scale = nnx.List(
            [nnx.Param(jnp.ones((self.width_out[i + 1],))) for i in range(self.depth)]
        )
        object.__setattr__(self, "_act_cache", None)
        object.__setattr__(self, "_attribute_cache", None)
        object.__setattr__(self, "_symbolic_fits", None)

    def _arity_list_for_width(self, width_idx: int) -> List[int]:
        dim_mult = self.width[width_idx][1]
        if dim_mult == 0:
            return []
        if self.mult_homo:
            return [int(self.mult_arity)] * dim_mult
        if width_idx >= len(self.mult_arity):
            raise ValueError(
                f"Missing multiplication arities for width index {width_idx}."
            )
        arities = [int(v) for v in self.mult_arity[width_idx]]
        if len(arities) != dim_mult:
            raise ValueError(
                f"Expected {dim_mult} multiplication arities at width index {width_idx}, got {len(arities)}."
            )
        return arities

    def _prepare_input(self, x):
        input_id = getattr(self, "input_id", None)
        if input_id is None:
            return x
        input_id = np.asarray(input_id, dtype=np.int32)
        if x.shape[1] == input_id.shape[0]:
            return x
        if input_id.size == 0:
            return x[:, :0]
        if int(np.max(input_id)) >= x.shape[1]:
            raise ValueError(
                f"Input has {x.shape[1]} columns, but this pruned MKAN expects "
                f"columns up to index {int(np.max(input_id))}."
            )
        return x[:, input_id]

    def _node_to_subnode_ids(self, width_idx: int, node_id: int) -> List[int]:
        n_sum = self.width[width_idx][0]
        if node_id < n_sum:
            return [node_id]
        arities = self._arity_list_for_width(width_idx)
        mult_idx = node_id - n_sum
        offset = n_sum + sum(arities[:mult_idx])
        return list(range(offset, offset + arities[mult_idx]))

    def _nodes_to_subnode_ids(self, width_idx: int, node_ids: Sequence[int]) -> List[int]:
        subnode_ids: List[int] = []
        for node_id in node_ids:
            subnode_ids.extend(self._node_to_subnode_ids(width_idx, int(node_id)))
        return subnode_ids

    def _current_layer_parameters(self) -> dict:
        """Return constructor parameters needed to rebuild a pruned layer."""

        first_layer = self.layers[0]
        params = dict(self.required_parameters)
        if hasattr(first_layer, "k"):
            params["k"] = int(first_layer.k)
        if hasattr(first_layer, "grid"):
            params["G"] = int(first_layer.grid.G)
            params["grid_range"] = tuple(first_layer.grid.grid_range)
            params["grid_e"] = float(first_layer.grid.grid_e)
        if hasattr(first_layer, "D"):
            params["D"] = int(first_layer.D)
        if hasattr(first_layer, "flavor"):
            params["flavor"] = first_layer.flavor
        params["residual"] = getattr(first_layer, "residual", None)
        params["external_weights"] = (
            getattr(first_layer, "c_spl", None) is not None
            or getattr(first_layer, "c_ext", None) is not None
        )
        params["add_bias"] = getattr(first_layer, "bias", None) is not None
        return params

    @property
    def width_in(self) -> List[int]:
        return [layer[0] + layer[1] for layer in self.width]

    @property
    def width_out(self) -> List[int]:
        width_out = []
        for idx, layer in enumerate(self.width):
            n_sum, _ = layer
            width_out.append(n_sum + sum(self._arity_list_for_width(idx)))
        return width_out

    def _apply_multiplication(self, x, layer_idx: int):
        width_idx = layer_idx + 1
        dim_sum = self.width[width_idx][0]
        arities = self._arity_list_for_width(width_idx)

        if not arities:
            return x[:, :dim_sum]

        x_sum = x[:, :dim_sum]
        offset = dim_sum
        mult_terms = []
        for arity in arities:
            mult_terms.append(jnp.prod(x[:, offset : offset + arity], axis=1, keepdims=True))
            offset += arity
        x_mult = jnp.concatenate(mult_terms, axis=1)
        return jnp.concatenate([x_sum, x_mult], axis=1)

    def _node_score_to_subnode_score(self, node_score, width_idx: int):
        width = self.width[width_idx]
        n_sum = width[0]
        arities = self._arity_list_for_width(width_idx)
        pieces = [node_score[:, :n_sum]]

        for mult_idx, arity in enumerate(arities):
            score = node_score[:, n_sum + mult_idx : n_sum + mult_idx + 1]
            pieces.append(jnp.repeat(score, arity, axis=1))

        if len(pieces) == 1:
            return pieces[0]
        return jnp.concatenate(pieces, axis=1)

    def update_grids(self, x, G_new):
        x = self._prepare_input(x)
        for idx, layer in enumerate(self.layers):
            layer.update_grid(x, G_new)
            x = layer(x)
            x = self.subnode_scale[idx][...][None, :] * x + self.subnode_bias[idx][...][None, :]
            x = self._apply_multiplication(x, idx)
            x = self.node_scale[idx][...][None, :] * x + self.node_bias[idx][...][None, :]

    def extend_grids(self, x, G_new, optimizer=None):
        """
        Extend/refine all spline grids and transfer edge functions to the new
        bases with the same least-squares projection used by each layer.
        """

        self.update_grids(x, G_new)

        if optimizer is not None:
            _, model_state = nnx.split(self)
            adam_transition(optimizer.opt_state, model_state)

    def refine_grids(self, x, G_new, optimizer=None):
        """Alias for :meth:`extend_grids` using KAN refinement terminology."""

        self.extend_grids(x, G_new, optimizer=optimizer)

    def get_act(self, x):
        """
        Run an interpretability forward pass and cache per-edge activations.

        This method is intended for analysis outside the training step. It
        leaves ``__call__`` untouched and returns the intermediate quantities
        needed for plotting and attribution.
        """

        raw_x = x
        x = self._prepare_input(x)

        cache = {
            "raw_input": raw_x,
            "acts": [x],
            "preacts": [],
            "postacts": [],
            "acts_scale": [],
            "edge_actscale": [],
            "subnode_actscale": [],
            "acts_premult": [],
        }

        for idx, layer in enumerate(self.layers):
            if not hasattr(layer, "edge_activations"):
                raise NotImplementedError(
                    f"{type(layer).__name__} does not expose edge_activations(). "
                    "This MKAN cannot run attribution-based pruning."
                )

            y, info = layer.edge_activations(x)
            preacts = info["preacts"]
            postacts = info["postacts"]

            input_range = jnp.std(preacts, axis=0) + 0.1
            output_range = jnp.std(postacts, axis=0)
            cache["preacts"].append(preacts)
            cache["postacts"].append(postacts)
            cache["edge_actscale"].append(output_range)
            cache["acts_scale"].append(output_range / input_range[None, :])

            y = self.subnode_scale[idx][...][None, :] * y + self.subnode_bias[idx][...][None, :]
            cache["subnode_actscale"].append(jnp.std(y, axis=0))
            cache["acts_premult"].append(y)

            x = self._apply_multiplication(y, idx)
            x = self.node_scale[idx][...][None, :] * x + self.node_bias[idx][...][None, :]
            cache["acts"].append(x)

        object.__setattr__(self, "_act_cache", cache)
        return cache

    def attribute(self, cache=None, l=None, i=None, out_score=None, plot=False):
        """
        Compute pykan-style backward attribution scores from cached activations.

        Returns a dictionary with per-layer ``node_scores``, ``edge_scores``,
        ``subnode_scores`` and the input ``feature_score``.
        """

        if cache is None:
            cache = self._act_cache
        if cache is None:
            raise ValueError("Call get_act(x) first or pass a cache to attribute().")

        l_query = l
        l_end = self.depth if l is None else int(l)
        if l_end < 0 or l_end > self.depth:
            raise ValueError(f"l must be between 0 and {self.depth}, got {l}.")

        out_dim = self.width_in[l_end]
        if out_score is None:
            node_score = jnp.eye(out_dim)
        else:
            node_score = jnp.diag(jnp.asarray(out_score))

        node_scores_all = [node_score]
        edge_scores_all = []
        subnode_scores_all = []

        for layer_idx in range(l_end - 1, -1, -1):
            subnode_score = self._node_score_to_subnode_score(node_score, layer_idx + 1)
            edge_actscale = cache["edge_actscale"][layer_idx]
            subnode_actscale = cache["subnode_actscale"][layer_idx]

            edge_score = (
                edge_actscale[None, :, :]
                * subnode_score[:, :, None]
                / (subnode_actscale[None, :, None] + 1e-4)
            )

            node_score = jnp.sum(edge_score, axis=1)
            subnode_scores_all.append(subnode_score)
            edge_scores_all.append(edge_score)
            node_scores_all.append(node_score)

        node_scores_all = list(reversed(node_scores_all))
        edge_scores_all = list(reversed(edge_scores_all))
        subnode_scores_all = list(reversed(subnode_scores_all))

        result = {
            "node_scores_all": node_scores_all,
            "edge_scores_all": edge_scores_all,
            "subnode_scores_all": subnode_scores_all,
            "node_scores": [jnp.mean(score, axis=0) for score in node_scores_all],
            "edge_scores": [jnp.mean(score, axis=0) for score in edge_scores_all],
            "subnode_scores": [jnp.mean(score, axis=0) for score in subnode_scores_all],
        }
        result["feature_score"] = result["node_scores"][0]
        object.__setattr__(self, "_attribute_cache", result)

        if l_query is not None:
            queried = result["node_scores_all"][0]
            if i is None:
                return queried
            if plot:
                import matplotlib.pyplot as plt

                values = queried[int(i)]
                plt.figure(figsize=(max(3, values.shape[0]), 3))
                plt.bar(range(values.shape[0]), values)
                plt.xticks(range(values.shape[0]))
            return queried[int(i)]

        return result

    @property
    def feature_score(self):
        """Input attribution scores from the most recent get_act/attribute run."""

        if self._attribute_cache is None:
            if self._act_cache is None:
                return None
            self.attribute(self._act_cache)
        return self._attribute_cache["feature_score"]

    def feature_interaction(self, l, neuron_th=1e-2, feature_th=1e-2, cache=None):
        """
        Count active input-feature groups used by neurons in layer ``l``.

        This mirrors pykan's ``feature_interaction`` helper: for each active
        neuron, features whose score exceeds ``feature_th`` times that
        neuron's maximum input score are grouped together.
        """

        if cache is None:
            cache = self._act_cache
        if cache is None:
            raise ValueError("Call get_act(x) first or pass a cache to feature_interaction().")

        interactions = {}
        for neuron_idx in range(self.width_in[int(l)]):
            score = np.asarray(self.attribute(cache, l=l, i=neuron_idx, plot=False))
            if score.size == 0 or np.max(score) <= neuron_th:
                continue
            features = tuple(np.where(score > np.max(score) * feature_th)[0].tolist())
            interactions[features] = interactions.get(features, 0) + 1
        return interactions

    def get_fun(self, l, i, j, cache=None, plot=True):
        """
        Return sorted ``(x, y)`` samples for edge ``(l, i, j)``.
        """

        if cache is None:
            cache = self._act_cache
        if cache is None:
            raise ValueError("Call get_act(x) first or pass a cache to get_fun().")

        inputs = np.asarray(cache["acts"][int(l)][:, int(i)])
        outputs = np.asarray(cache["postacts"][int(l)][:, int(j), int(i)])
        order = np.argsort(inputs)
        inputs = inputs[order]
        outputs = outputs[order]

        if plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(3, 3))
            plt.plot(inputs, outputs, marker="o")

        return inputs, outputs

    def get_range(self, l, i, j, cache=None, verbose=True):
        """
        Return the input/output min and max for edge ``(l, i, j)``.
        """

        if cache is None:
            cache = self._act_cache
        if cache is None:
            raise ValueError("Call get_act(x) first or pass a cache to get_range().")

        x = np.asarray(cache["preacts"][int(l)][:, int(i)])
        y = np.asarray(cache["postacts"][int(l)][:, int(j), int(i)])
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        if verbose:
            print(f"x range: [{x_min:.2f}, {x_max:.2f}]")
            print(f"y range: [{y_min:.2f}, {y_max:.2f}]")
        return x_min, x_max, y_min, y_max

    def _set_edge_zero(self, layer_idx: int, in_idx: int, out_idx: int):
        layer = self.layers[int(layer_idx)]
        in_idx = int(in_idx)
        out_idx = int(out_idx)

        if hasattr(layer, "set_edge_mask"):
            layer.set_edge_mask(in_idx, out_idx, 0.0)
            return

        if self.layer_type == "base":
            flat_idx = out_idx * layer.n_in + in_idx
            layer.c_basis = nnx.Param(layer.c_basis[...].at[flat_idx, :].set(0.0))
        else:
            layer.c_basis = nnx.Param(layer.c_basis[...].at[out_idx, in_idx, :].set(0.0))

        if getattr(layer, "c_spl", None) is not None:
            layer.c_spl = nnx.Param(layer.c_spl[...].at[out_idx, in_idx].set(0.0))
        if getattr(layer, "residual", None) is not None and hasattr(layer, "c_res"):
            layer.c_res = nnx.Param(layer.c_res[...].at[out_idx, in_idx].set(0.0))

    def remove_edge(self, l, i, j):
        """Set edge ``(l, i, j)`` to zero in-place."""

        self._set_edge_zero(l, i, j)
        object.__setattr__(self, "_act_cache", None)
        object.__setattr__(self, "_attribute_cache", None)
        object.__setattr__(self, "_symbolic_fits", None)
        return self

    def prune_edge(self, threshold=3e-2, cache=None):
        """
        Zero edges whose backward attribution score is below ``threshold``.
        """

        if cache is None:
            cache = self._act_cache
        if cache is None:
            raise ValueError("Call get_act(x) first or pass a cache to prune_edge().")

        attr = self.attribute(cache)
        for layer_idx, scores in enumerate(attr["edge_scores"]):
            scores_np = np.asarray(scores)
            for out_idx in range(scores_np.shape[0]):
                for in_idx in range(scores_np.shape[1]):
                    if scores_np[out_idx, in_idx] <= threshold:
                        self._set_edge_zero(layer_idx, in_idx, out_idx)

        object.__setattr__(self, "_act_cache", None)
        object.__setattr__(self, "_attribute_cache", None)
        object.__setattr__(self, "_symbolic_fits", None)
        return self

    def remove_node(self, l, i, mode="all"):
        """
        Set incoming and/or outgoing edges of node ``(l, i)`` to zero in-place.
        """

        l = int(l)
        i = int(i)
        if mode not in ("all", "up", "down"):
            raise ValueError("mode must be 'all', 'up', or 'down'.")

        if l > 0 and mode in ("all", "up"):
            for out_idx in self._node_to_subnode_ids(l, i):
                for in_idx in range(self.width_in[l - 1]):
                    self._set_edge_zero(l - 1, in_idx, out_idx)

        if l < self.depth and mode in ("all", "down"):
            for out_idx in range(self.width_out[l + 1]):
                self._set_edge_zero(l, i, out_idx)

        object.__setattr__(self, "_act_cache", None)
        object.__setattr__(self, "_attribute_cache", None)
        object.__setattr__(self, "_symbolic_fits", None)
        return self

    def _copy_layer_subset(self, dst_layer, src_layer, in_ids, out_ids):
        in_ids = jnp.asarray(in_ids, dtype=jnp.int32)
        out_ids = jnp.asarray(out_ids, dtype=jnp.int32)

        dst_layer.n_in = int(in_ids.shape[0])
        dst_layer.n_out = int(out_ids.shape[0])
        if hasattr(src_layer, "grid") and hasattr(dst_layer, "grid"):
            dst_layer.grid.G = int(src_layer.grid.G)
            dst_layer.grid.grid_range = tuple(src_layer.grid.grid_range)
            dst_layer.grid.grid_e = float(src_layer.grid.grid_e)

        if self.layer_type == "base":
            old_basis = src_layer.c_basis[...].reshape(src_layer.n_out, src_layer.n_in, -1)
            new_basis = jnp.take(jnp.take(old_basis, out_ids, axis=0), in_ids, axis=1)
            dst_layer.c_basis = nnx.Param(new_basis.reshape(out_ids.shape[0] * in_ids.shape[0], -1))

            old_grid = src_layer.grid.item.reshape(src_layer.n_out, src_layer.n_in, -1)
            new_grid = jnp.take(jnp.take(old_grid, out_ids, axis=0), in_ids, axis=1)
            dst_layer.grid.item = new_grid.reshape(out_ids.shape[0] * in_ids.shape[0], -1)
            dst_layer.grid.n_in = int(in_ids.shape[0])
            dst_layer.grid.n_out = int(out_ids.shape[0])
        else:
            new_basis = jnp.take(jnp.take(src_layer.c_basis[...], out_ids, axis=0), in_ids, axis=1)
            dst_layer.c_basis = nnx.Param(new_basis)
            if hasattr(src_layer, "grid") and hasattr(dst_layer, "grid"):
                dst_layer.grid.item = src_layer.grid.item[in_ids]
                dst_layer.grid.n_nodes = int(in_ids.shape[0])

        if getattr(src_layer, "c_spl", None) is not None:
            dst_layer.c_spl = nnx.Param(jnp.take(jnp.take(src_layer.c_spl[...], out_ids, axis=0), in_ids, axis=1))
        elif hasattr(dst_layer, "c_spl"):
            dst_layer.c_spl = None

        if getattr(src_layer, "c_ext", None) is not None:
            dst_layer.c_ext = nnx.Param(
                jnp.take(jnp.take(src_layer.c_ext[...], out_ids, axis=0), in_ids, axis=1)
            )
        elif hasattr(dst_layer, "c_ext"):
            dst_layer.c_ext = None

        if getattr(src_layer, "residual", None) is not None and hasattr(src_layer, "c_res"):
            dst_layer.c_res = nnx.Param(jnp.take(jnp.take(src_layer.c_res[...], out_ids, axis=0), in_ids, axis=1))

        if getattr(src_layer, "bias", None) is not None:
            dst_layer.bias = nnx.Param(src_layer.bias[...][out_ids])
        elif hasattr(dst_layer, "bias"):
            dst_layer.bias = None

        if hasattr(src_layer, "edge_mask") and hasattr(dst_layer, "edge_mask"):
            dst_layer.edge_mask = nnx.Variable(
                jnp.take(jnp.take(src_layer.edge_mask[...], out_ids, axis=0), in_ids, axis=1)
            )

    def _require_structural_pruning_support(self):
        supported = {"base", "spline", "chebyshev"}
        if self.layer_type not in supported:
            raise NotImplementedError(
                "Structural pruning currently supports only base, spline, and "
                f"chebyshev layers; got {self.layer_type!r}."
            )

    def _prune_with_active_nodes(self, active_nodes: Sequence[Sequence[int]]):
        self._require_structural_pruning_support()

        active_nodes = [list(map(int, ids)) for ids in active_nodes]
        if len(active_nodes) != self.depth + 1:
            raise ValueError(f"Expected {self.depth + 1} active-node lists.")

        new_width = []
        new_mult_arity: list[list[int]] = []

        for width_idx, node_ids in enumerate(active_nodes):
            n_sum = self.width[width_idx][0]
            sum_count = sum(node_id < n_sum for node_id in node_ids)
            mult_ids = [node_id - n_sum for node_id in node_ids if node_id >= n_sum]
            mult_count = len(mult_ids)
            if mult_count == 0:
                new_width.append(sum_count)
            else:
                new_width.append([sum_count, mult_count])

            arities = self._arity_list_for_width(width_idx)
            new_mult_arity.append([arities[mult_id] for mult_id in mult_ids])

        mult_arity = self.mult_arity if self.mult_homo else new_mult_arity
        model = MultKAN(
            width=new_width,
            layer_type=self.layer_type,
            required_parameters=self._current_layer_parameters(),
            mult_arity=mult_arity,
            seed=self.seed,
        )

        old_input_id = np.asarray(getattr(self, "input_id", np.arange(self.width_in[0])))
        object.__setattr__(model, "input_id", old_input_id[np.asarray(active_nodes[0], dtype=int)])

        for layer_idx in range(self.depth):
            in_ids = active_nodes[layer_idx]
            out_ids = self._nodes_to_subnode_ids(layer_idx + 1, active_nodes[layer_idx + 1])
            self._copy_layer_subset(model.layers[layer_idx], self.layers[layer_idx], in_ids, out_ids)

            node_ids = jnp.asarray(active_nodes[layer_idx + 1], dtype=jnp.int32)
            subnode_ids = jnp.asarray(out_ids, dtype=jnp.int32)
            model.node_bias[layer_idx] = nnx.Param(self.node_bias[layer_idx][...][node_ids])
            model.node_scale[layer_idx] = nnx.Param(self.node_scale[layer_idx][...][node_ids])
            model.subnode_bias[layer_idx] = nnx.Param(self.subnode_bias[layer_idx][...][subnode_ids])
            model.subnode_scale[layer_idx] = nnx.Param(self.subnode_scale[layer_idx][...][subnode_ids])

        return model

    def prune_node(self, threshold=1e-2, mode="auto", active_neurons_id=None, cache=None):
        """
        Return a smaller MKAN with hidden nodes pruned by attribution.
        """

        self._require_structural_pruning_support()

        if cache is None:
            cache = self._act_cache
        if cache is None:
            raise ValueError("Call get_act(x) first or pass a cache to prune_node().")

        if active_neurons_id is not None:
            mode = "manual"
        if mode not in ("auto", "manual"):
            raise ValueError("mode must be 'auto' or 'manual'.")

        attr = self.attribute(cache)
        active_nodes = [list(range(self.width_in[0]))]

        for width_idx in range(1, self.depth):
            if mode == "manual":
                ids = list(map(int, active_neurons_id[width_idx - 1]))
            else:
                scores = np.asarray(attr["node_scores"][width_idx])
                ids = np.where(scores > threshold)[0].astype(int).tolist()
                if not ids and scores.size:
                    ids = [int(np.argmax(scores))]
            active_nodes.append(ids)

        active_nodes.append(list(range(self.width_in[-1])))
        return self._prune_with_active_nodes(active_nodes)

    def prune_input(self, threshold=1e-2, active_inputs=None, cache=None):
        """
        Return a smaller MKAN that keeps only selected input features.

        The returned model keeps ``input_id`` so it can still be called with the
        original full feature matrix.
        """

        self._require_structural_pruning_support()

        if active_inputs is None:
            if cache is None:
                cache = self._act_cache
            if cache is None:
                raise ValueError("Call get_act(x) first or pass a cache to prune_input().")
            attr = self.attribute(cache)
            scores = np.asarray(attr["feature_score"])
            active_inputs = np.where(scores > threshold)[0].astype(int).tolist()
            if not active_inputs and scores.size:
                active_inputs = [int(np.argmax(scores))]
            print("keep:", [idx in active_inputs for idx in range(self.width_in[0])])

        active_nodes = [list(map(int, active_inputs))]
        for width_idx in range(1, self.depth + 1):
            active_nodes.append(list(range(self.width_in[width_idx])))
        return self._prune_with_active_nodes(active_nodes)

    def prune(self, node_th=1e-2, edge_th=3e-2, cache=None):
        """
        Return a smaller node-pruned MKAN and zero low-attribution edges.
        """

        if cache is None:
            cache = self._act_cache
        if cache is None:
            raise ValueError("Call get_act(x) first or pass a cache to prune().")

        model = self.prune_node(threshold=node_th, cache=cache)
        x = cache.get("raw_input", cache["acts"][0])
        new_cache = model.get_act(x)
        model.prune_edge(threshold=edge_th, cache=new_cache)
        model.get_act(x)
        return model

    def _ensure_symbolic_fits(self):
        if self._symbolic_fits is None:
            fits = []
            for layer_idx in range(self.depth):
                fits.append(
                    [
                        [None for _ in range(self.width_in[layer_idx])]
                        for _ in range(self.width_out[layer_idx + 1])
                    ]
                )
            object.__setattr__(self, "_symbolic_fits", fits)

    @staticmethod
    def _symbolic_library():
        import sympy

        with np.errstate(all="ignore"):
            lib = {
                "x": (lambda x: x, lambda x: x, 1),
                "x^2": (lambda x: x**2, lambda x: x**2, 2),
                "x^3": (lambda x: x**3, lambda x: x**3, 3),
                "x^4": (lambda x: x**4, lambda x: x**4, 3),
                "x^5": (lambda x: x**5, lambda x: x**5, 3),
                "1/x": (lambda x: 1 / x, lambda x: 1 / x, 2),
                "1/x^2": (lambda x: 1 / x**2, lambda x: 1 / x**2, 2),
                "1/x^3": (lambda x: 1 / x**3, lambda x: 1 / x**3, 3),
                "sqrt": (lambda x: np.sqrt(x), lambda x: sympy.sqrt(x), 2),
                "x^0.5": (lambda x: np.sqrt(x), lambda x: sympy.sqrt(x), 2),
                "x^1.5": (lambda x: np.sqrt(x) ** 3, lambda x: sympy.sqrt(x) ** 3, 4),
                "1/sqrt(x)": (lambda x: 1 / np.sqrt(x), lambda x: 1 / sympy.sqrt(x), 2),
                "exp": (lambda x: np.exp(x), lambda x: sympy.exp(x), 2),
                "log": (lambda x: np.log(x), lambda x: sympy.log(x), 2),
                "abs": (lambda x: np.abs(x), lambda x: sympy.Abs(x), 3),
                "sin": (lambda x: np.sin(x), lambda x: sympy.sin(x), 2),
                "cos": (lambda x: np.cos(x), lambda x: sympy.cos(x), 2),
                "tan": (lambda x: np.tan(x), lambda x: sympy.tan(x), 3),
                "tanh": (lambda x: np.tanh(x), lambda x: sympy.tanh(x), 3),
                "sgn": (lambda x: np.sign(x), lambda x: sympy.sign(x), 3),
                "arcsin": (lambda x: np.arcsin(x), lambda x: sympy.asin(x), 4),
                "arccos": (lambda x: np.arccos(x), lambda x: sympy.acos(x), 4),
                "arctan": (lambda x: np.arctan(x), lambda x: sympy.atan(x), 4),
                "arctanh": (lambda x: np.arctanh(x), lambda x: sympy.atanh(x), 4),
                "0": (lambda x: x * 0.0, lambda x: x * 0, 0),
                "gaussian": (lambda x: np.exp(-(x**2)), lambda x: sympy.exp(-(x**2)), 3),
            }
        return lib

    @staticmethod
    def _nan_to_num(values):
        return np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)

    @classmethod
    def _fit_symbolic_params(
        cls,
        x,
        y,
        fun: Callable[[np.ndarray], np.ndarray],
        a_range=(-10, 10),
        b_range=(-10, 10),
        grid_number=101,
        iteration=3,
    ):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.size == 0 or y.size == 0:
            return np.array([1.0, 0.0, 0.0, 0.0]), -math.inf

        a_range = [float(a_range[0]), float(a_range[1])]
        b_range = [float(b_range[0]), float(b_range[1])]
        best_a = 1.0
        best_b = 0.0
        best_r2 = -math.inf

        for _ in range(iteration):
            a_values = np.linspace(a_range[0], a_range[1], grid_number)
            b_values = np.linspace(b_range[0], b_range[1], grid_number)
            a_grid, b_grid = np.meshgrid(a_values, b_values, indexing="ij")
            with np.errstate(all="ignore"):
                post_fun = fun(a_grid[None, :, :] * x[:, None, None] + b_grid[None, :, :])
            post_fun = cls._nan_to_num(post_fun)

            x_mean = np.mean(post_fun, axis=0, keepdims=True)
            y_mean = np.mean(y)
            numerator = np.sum((post_fun - x_mean) * (y[:, None, None] - y_mean), axis=0) ** 2
            denominator = (
                np.sum((post_fun - x_mean) ** 2, axis=0)
                * np.sum((y - y_mean) ** 2)
                + 1e-12
            )
            r2 = cls._nan_to_num(numerator / denominator)
            best_flat = int(np.argmax(r2))
            a_id, b_id = np.unravel_index(best_flat, r2.shape)
            best_a = float(a_values[a_id])
            best_b = float(b_values[b_id])
            best_r2 = float(r2[a_id, b_id])

            a_low = max(0, a_id - 1)
            a_high = min(grid_number - 1, a_id + 1)
            b_low = max(0, b_id - 1)
            b_high = min(grid_number - 1, b_id + 1)
            a_range = [float(a_values[a_low]), float(a_values[a_high])]
            b_range = [float(b_values[b_low]), float(b_values[b_high])]

        with np.errstate(all="ignore"):
            post_fun = cls._nan_to_num(fun(best_a * x + best_b))

        if np.std(post_fun) < 1e-12:
            c_best = 0.0
            d_best = float(np.mean(y))
        else:
            design = np.stack([post_fun, np.ones_like(post_fun)], axis=1)
            c_best, d_best = np.linalg.lstsq(design, y, rcond=None)[0]

        return np.array([best_a, best_b, float(c_best), float(d_best)]), best_r2

    def _edge_xy(self, l, i, j, cache=None):
        if cache is None:
            cache = self._act_cache
        if cache is None:
            raise ValueError("Call get_act(x) first or pass a cache.")
        return (
            np.asarray(cache["acts"][int(l)][:, int(i)]),
            np.asarray(cache["postacts"][int(l)][:, int(j), int(i)]),
        )

    def fix_symbolic(
        self,
        l,
        i,
        j,
        fun_name,
        fit_params_bool=True,
        a_range=(-10, 10),
        b_range=(-10, 10),
        verbose=True,
        cache=None,
    ):
        """
        Store a symbolic approximation for edge ``(l, i, j)``.

        Unlike pykan's Torch implementation, this does not change the JAX
        forward pass; it records an analysis-only fit consumed by
        ``symbolic_formula``.
        """

        lib = self._symbolic_library()
        if fun_name not in lib:
            raise ValueError(f"Unknown symbolic function {fun_name!r}.")

        if fun_name == "0":
            params = np.array([1.0, 0.0, 0.0, 0.0])
            r2 = -math.inf
        elif fit_params_bool:
            x, y = self._edge_xy(l, i, j, cache=cache)
            params, r2 = self._fit_symbolic_params(
                x, y, lib[fun_name][0], a_range=a_range, b_range=b_range
            )
            if verbose:
                print(f"r2 is {r2}")
        else:
            params = np.array([1.0, 0.0, 1.0, 0.0])
            r2 = math.nan

        self._ensure_symbolic_fits()
        self._symbolic_fits[int(l)][int(j)][int(i)] = {
            "function": fun_name,
            "params": params.tolist(),
            "r2": float(r2) if np.isfinite(r2) else r2,
            "complexity": lib[fun_name][2],
        }
        return r2

    def unfix_symbolic(self, l, i, j):
        self._ensure_symbolic_fits()
        self._symbolic_fits[int(l)][int(j)][int(i)] = None

    def unfix_symbolic_all(self):
        object.__setattr__(self, "_symbolic_fits", None)

    def suggest_symbolic(
        self,
        l,
        i,
        j,
        a_range=(-10, 10),
        b_range=(-10, 10),
        lib=None,
        topk=5,
        verbose=True,
        r2_loss_fun=None,
        c_loss_fun=None,
        weight_simple=0.8,
        cache=None,
    ):
        """
        Fit candidate symbolic functions to one edge and return the best one.
        """

        symbolic_lib = self._symbolic_library()
        if lib is not None:
            symbolic_lib = {name: symbolic_lib[name] for name in lib}

        if r2_loss_fun is None:
            r2_loss_fun = lambda value: np.log2(np.maximum(1e-12, 1 + 1e-5 - value))
        if c_loss_fun is None:
            c_loss_fun = lambda value: value

        x, y = self._edge_xy(l, i, j, cache=cache)
        rows = []
        for name, (fun, _sym_fun, complexity) in symbolic_lib.items():
            if name == "0":
                params = np.array([1.0, 0.0, 0.0, 0.0])
                r2 = 0.0
            else:
                params, r2 = self._fit_symbolic_params(
                    x, y, fun, a_range=a_range, b_range=b_range
                )
            rows.append(
                {
                    "function": name,
                    "params": params,
                    "fitting r2": float(r2),
                    "complexity": float(complexity),
                }
            )

        for row in rows:
            row["r2 loss"] = float(r2_loss_fun(row["fitting r2"]))
            row["complexity loss"] = float(c_loss_fun(row["complexity"]))
            row["total loss"] = (
                weight_simple * row["complexity loss"]
                + (1 - weight_simple) * row["r2 loss"]
            )

        rows = sorted(rows, key=lambda row: row["total loss"])
        top_rows = rows[: min(int(topk), len(rows))]

        if verbose:
            try:
                import pandas as pd

                print(pd.DataFrame([{k: v for k, v in row.items() if k != "params"} for row in top_rows]))
            except Exception:
                for row in top_rows:
                    print({k: v for k, v in row.items() if k != "params"})

        best = rows[0]
        return (
            best["function"],
            symbolic_lib[best["function"]][0],
            best["fitting r2"],
            best["complexity"],
        )

    def auto_symbolic(
        self,
        a_range=(-10, 10),
        b_range=(-10, 10),
        lib=None,
        verbose=1,
        weight_simple=0.8,
        r2_threshold=0.0,
        cache=None,
    ):
        """
        Fit symbolic approximations for every edge.

        The fitted functions are analysis metadata; they are used by
        ``symbolic_formula`` and do not replace the learned JAX spline layers.
        """

        if cache is None:
            cache = self._act_cache
        if cache is None:
            raise ValueError("Call get_act(x) first or pass a cache to auto_symbolic().")

        self._ensure_symbolic_fits()
        symbolic_lib = self._symbolic_library()
        if lib is not None:
            symbolic_lib = {name: symbolic_lib[name] for name in lib}

        for layer_idx in range(self.depth):
            edge_scale = np.asarray(cache["edge_actscale"][layer_idx])
            for in_idx in range(self.width_in[layer_idx]):
                for out_idx in range(self.width_out[layer_idx + 1]):
                    if edge_scale[out_idx, in_idx] < 1e-12:
                        self.fix_symbolic(
                            layer_idx,
                            in_idx,
                            out_idx,
                            "0",
                            verbose=False,
                            cache=cache,
                        )
                        if verbose >= 1:
                            print(f"fixing ({layer_idx},{in_idx},{out_idx}) with 0")
                        continue

                    name, _fun, r2, complexity = self.suggest_symbolic(
                        layer_idx,
                        in_idx,
                        out_idx,
                        a_range=a_range,
                        b_range=b_range,
                        lib=list(symbolic_lib.keys()),
                        verbose=False,
                        weight_simple=weight_simple,
                        cache=cache,
                    )
                    if r2 >= r2_threshold:
                        self.fix_symbolic(
                            layer_idx,
                            in_idx,
                            out_idx,
                            name,
                            a_range=a_range,
                            b_range=b_range,
                            verbose=False,
                            cache=cache,
                        )
                        if verbose >= 1:
                            print(
                                f"fixing ({layer_idx},{in_idx},{out_idx}) "
                                f"with {name}, r2={r2}, c={complexity}"
                            )
                    else:
                        self.fix_symbolic(
                            layer_idx,
                            in_idx,
                            out_idx,
                            "0",
                            verbose=False,
                            cache=cache,
                        )
                        if verbose >= 1:
                            print(
                                f"omitting ({layer_idx},{in_idx},{out_idx}); "
                                f"best {name} had r2={r2} < {r2_threshold}"
                            )

    def symbolic_formula(
        self,
        var=None,
        normalizer=None,
        output_normalizer=None,
        simplify=False,
    ):
        """
        Compose the fitted symbolic edge functions into output formulas.

        Run ``auto_symbolic`` or ``fix_symbolic`` first.
        """

        import sympy

        if self._symbolic_fits is None:
            raise ValueError("Run auto_symbolic() or fix_symbolic() before symbolic_formula().")

        symbolic_lib = self._symbolic_library()

        input_id = getattr(self, "input_id", None)
        if var is None:
            if input_id is None:
                x = [sympy.Symbol(f"x_{idx + 1}") for idx in range(self.width_in[0])]
            else:
                x = [sympy.Symbol(f"x_{int(idx) + 1}") for idx in np.asarray(input_id)]
        elif isinstance(var[0], sympy.Expr):
            x = list(var)
        else:
            x = [sympy.symbols(str(item)) for item in var]

        if input_id is not None and len(x) > self.width_in[0]:
            x = [x[int(idx)] for idx in np.asarray(input_id)]

        x0 = list(x)
        if normalizer is not None:
            mean, std = normalizer
            if input_id is not None and len(mean) > len(x):
                mean = np.asarray(mean)[np.asarray(input_id)]
                std = np.asarray(std)[np.asarray(input_id)]
            x = [(x[idx] - mean[idx]) / std[idx] for idx in range(len(x))]

        symbolic_acts = [x]
        symbolic_acts_premult = []

        for layer_idx in range(self.depth):
            y = []
            for out_idx in range(self.width_out[layer_idx + 1]):
                expr = sympy.Integer(0)
                for in_idx in range(self.width_in[layer_idx]):
                    fit = self._symbolic_fits[layer_idx][out_idx][in_idx]
                    if fit is None:
                        raise ValueError(
                            f"Missing symbolic fit for edge ({layer_idx},{in_idx},{out_idx})."
                        )
                    name = fit["function"]
                    a, b, c, d = fit["params"]
                    sym_fun = symbolic_lib[name][1]
                    expr += c * sym_fun(a * x[in_idx] + b) + d

                layer = self.layers[layer_idx]
                if layer.bias is not None:
                    expr += float(np.asarray(layer.bias[...][out_idx]))

                expr = (
                    float(np.asarray(self.subnode_scale[layer_idx][...][out_idx])) * expr
                    + float(np.asarray(self.subnode_bias[layer_idx][...][out_idx]))
                )
                y.append(sympy.simplify(expr) if simplify else expr)

            symbolic_acts_premult.append(y)

            n_sum = self.width[layer_idx + 1][0]
            arities = self._arity_list_for_width(layer_idx + 1)
            x_next = y[:n_sum]
            offset = n_sum
            for arity in arities:
                term = sympy.Integer(1)
                for sub_idx in range(offset, offset + arity):
                    term *= y[sub_idx]
                x_next.append(sympy.simplify(term) if simplify else term)
                offset += arity

            for node_idx in range(len(x_next)):
                x_next[node_idx] = (
                    float(np.asarray(self.node_scale[layer_idx][...][node_idx])) * x_next[node_idx]
                    + float(np.asarray(self.node_bias[layer_idx][...][node_idx]))
                )
                if simplify:
                    x_next[node_idx] = sympy.simplify(x_next[node_idx])

            x = x_next
            symbolic_acts.append(x)

        if output_normalizer is not None:
            means, stds = output_normalizer
            symbolic_acts[-1] = [
                symbolic_acts[-1][idx] * stds[idx] + means[idx]
                for idx in range(len(symbolic_acts[-1]))
            ]
            if simplify:
                symbolic_acts[-1] = [sympy.simplify(expr) for expr in symbolic_acts[-1]]

        object.__setattr__(self, "symbolic_acts", symbolic_acts)
        object.__setattr__(self, "symbolic_acts_premult", symbolic_acts_premult)
        return symbolic_acts[-1], x0

    def plot(
        self,
        cache=None,
        folder="./figures",
        beta=3,
        metric="backward",
        scale=0.5,
        sample=False,
        in_vars=None,
        out_vars=None,
        title=None,
    ):
        """
        Plot learned edge functions for each MKAN layer.

        A PNG named ``layer_<idx>_edges.png`` is written for each layer. The
        returned value is the list of matplotlib figures.
        """

        del out_vars

        if cache is None:
            cache = self._act_cache
        if cache is None:
            raise ValueError("Call get_act(x) first or pass a cache to plot().")

        import os
        import numpy as np
        import matplotlib.pyplot as plt

        if metric == "backward":
            attr = self.attribute(cache)
            scores = attr["edge_scores"]
        elif metric == "forward_n":
            scores = cache["acts_scale"]
        elif metric == "forward_u":
            scores = cache["edge_actscale"]
        else:
            raise ValueError("metric must be 'backward', 'forward_n', or 'forward_u'.")

        os.makedirs(folder, exist_ok=True)
        figures = []

        def input_label(layer_idx: int, in_idx: int) -> str:
            if layer_idx == 0 and in_vars is not None and in_idx < len(in_vars):
                return str(in_vars[in_idx])
            if layer_idx == 0:
                return f"x{in_idx}"
            return f"h{in_idx}"

        for layer_idx in range(self.depth):
            n_in = self.width_in[layer_idx]
            n_out = self.width_out[layer_idx + 1]
            fig_w = max(2.0, n_in * 2.0 * scale)
            fig_h = max(2.0, n_out * 1.8 * scale)
            fig, axes = plt.subplots(n_out, n_in, figsize=(fig_w, fig_h), squeeze=False)

            for out_idx in range(n_out):
                for in_idx in range(n_in):
                    ax = axes[out_idx][in_idx]
                    x_edge = np.asarray(cache["acts"][layer_idx][:, in_idx])
                    y_edge = np.asarray(cache["postacts"][layer_idx][:, out_idx, in_idx])
                    order = np.argsort(x_edge)
                    alpha = float(np.tanh(beta * np.asarray(scores[layer_idx][out_idx, in_idx])))
                    alpha = max(0.05, min(1.0, alpha))

                    ax.plot(x_edge[order], y_edge[order], color="black", alpha=alpha, linewidth=1.5)
                    if sample:
                        ax.scatter(x_edge, y_edge, color="black", alpha=alpha, s=8)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if out_idx == n_out - 1:
                        ax.set_xlabel(input_label(layer_idx, in_idx), fontsize=8)
                    if in_idx == 0:
                        ax.set_ylabel(f"y{out_idx}", fontsize=8)

            if title is not None:
                fig.suptitle(f"{title} layer {layer_idx}")
            else:
                fig.suptitle(f"MKAN layer {layer_idx}")
            fig.tight_layout()
            fig.savefig(f"{folder}/layer_{layer_idx}_edges.png", bbox_inches="tight", dpi=200)
            figures.append(fig)

        return figures

    def __call__(self, x):
        x = self._prepare_input(x)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x = self.subnode_scale[idx][...][None, :] * x + self.subnode_bias[idx][...][None, :]
            x = self._apply_multiplication(x, idx)
            x = self.node_scale[idx][...][None, :] * x + self.node_bias[idx][...][None, :]
        return x

    def collect_layer_inputs(self, x):
        """Collect each layer's raw and effective basis inputs.

        The layer transition below intentionally mirrors :meth:`__call__` so
        collecting diagnostics does not alter either the model state or the
        values presented to later layers.  ``basis_input`` records the value
        after a basis-specific input transform: FastKAN's LayerNorm and the
        tanh domain mapping used by Chebyshev and Legendre layers.
        """

        records = []
        x = self._prepare_input(x)

        for idx, layer in enumerate(self.layers):
            record = {
                "layer": idx,
                "raw_input": x,
            }

            if hasattr(layer, "normalize"):
                record["basis_input"] = layer.normalize(x)
            elif self.layer_type in ("chebyshev", "legendre"):
                record["basis_input"] = jnp.tanh(x)
            else:
                record["basis_input"] = x

            records.append(record)

            x = layer(x)
            x = self.subnode_scale[idx][...][None, :] * x + self.subnode_bias[idx][...][None, :]
            x = self._apply_multiplication(x, idx)
            x = self.node_scale[idx][...][None, :] * x + self.node_bias[idx][...][None, :]

        return records
