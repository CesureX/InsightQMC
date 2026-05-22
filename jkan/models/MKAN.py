from jax import numpy as jnp

from flax import nnx

from ..layers import get_layer
from ..layers.utils import adam_transition

from typing import List, Sequence, Union


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

        LayerClass = get_layer(layer_type.lower())

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

        cache = {
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
                    "Currently the MKAN interpretability helpers support base "
                    "and spline layers."
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
                        label = in_vars[in_idx] if in_vars is not None and in_idx < len(in_vars) else f"x{in_idx}"
                        ax.set_xlabel(label, fontsize=8)
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
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x = self.subnode_scale[idx][...][None, :] * x + self.subnode_bias[idx][...][None, :]
            x = self._apply_multiplication(x, idx)
            x = self.node_scale[idx][...][None, :] * x + self.node_bias[idx][...][None, :]
        return x
