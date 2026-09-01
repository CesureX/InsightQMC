"""QMC-specific wavefunction components for trainer-side assembly."""

from __future__ import annotations

import functools
from typing import Any, Iterable, MutableMapping, Optional, Sequence, Union

import chex
import jax.numpy as jnp
from flax import nnx

from jkan.layers import get_layer
from jkan.layers.Dense import DenseLayer
from jkan.models.KAN import KAN
from jkan.models.MKAN import MultKAN
from jkan.models.utils import get_activation


Array = jnp.ndarray
ParamTree = Union[Array, Iterable["ParamTree"], MutableMapping[Any, "ParamTree"]]


@chex.dataclass
class KANetsData:
    positions: Any
    spins: Any
    atoms: Any
    charges: Any


class DenseOrbitalHead(nnx.Module):
    """Dense/MLP map from MKAN nodes to orbital channels."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: Sequence[int] = (),
        activation: str = "silu",
        add_bias: bool = True,
        rwf: Optional[dict[str, float]] = None,
        seed: int = 42,
    ):
        dims = [int(input_dim), *[int(dim) for dim in hidden_dims], int(output_dim)]
        if any(dim <= 0 for dim in dims):
            raise ValueError("Orbital head dimensions must be positive.")

        activation_fn = get_activation(activation)
        dense_kwargs = {"add_bias": bool(add_bias)}
        if rwf is not None:
            dense_kwargs["RWF"] = dict(rwf)

        self.width = tuple(dims)
        self.activation_name = str(activation)
        self.layers = nnx.List(
            [
                DenseLayer(
                    n_in=dims[idx],
                    n_out=dims[idx + 1],
                    activation=activation_fn if idx < len(dims) - 2 else None,
                    seed=int(seed) + idx,
                    **dense_kwargs,
                )
                for idx in range(len(dims) - 1)
            ]
        )

    def __call__(self, nodes: Array) -> Array:
        x = nodes
        for layer in self.layers:
            x = layer(x)
        return x


class KANOrbitalHead(nnx.Module):
    """KAN map from MKAN nodes to orbital channels."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: Sequence[int] = (),
        layer_type: str = "base",
        required_parameters: Optional[dict] = None,
        seed: int = 42,
    ):
        dims = [int(input_dim), *[int(dim) for dim in hidden_dims], int(output_dim)]
        if any(dim <= 0 for dim in dims):
            raise ValueError("Orbital head dimensions must be positive.")

        self.width = tuple(dims)
        self.layer_type = str(layer_type).lower()
        self.model = KAN(
            layer_dims=dims,
            layer_type=self.layer_type,
            required_parameters=dict(required_parameters or {}),
            seed=int(seed),
        )

    def __call__(self, nodes: Array) -> Array:
        return self.model(nodes)


class MKANOrbitalHead(nnx.Module):
    """MultKAN map from MKAN nodes to orbital channels."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: Sequence[int] = (),
        width: Optional[Sequence[Union[int, Sequence[int]]]] = None,
        layer_type: str = "base",
        required_parameters: Optional[dict] = None,
        mult_arity: Union[int, Sequence[Sequence[int]]] = 2,
        seed: int = 42,
    ):
        if width is None:
            head_width = [int(input_dim), *[int(dim) for dim in hidden_dims], int(output_dim)]
        else:
            head_width = list(width)
            if len(head_width) < 2:
                raise ValueError("MKAN orbital head width must include input and output entries.")
            head_width[0] = int(input_dim)
            head_width[-1] = int(output_dim)

        self.width = tuple(head_width)
        self.layer_type = str(layer_type).lower()
        self.model = MultKAN(
            width=head_width,
            layer_type=self.layer_type,
            required_parameters=dict(required_parameters or {}),
            mult_arity=mult_arity,
            seed=int(seed),
        )

    def __call__(self, nodes: Array) -> Array:
        return self.model(nodes)


class FermiNetStreamKAN(nnx.Module):
    """KAN stream with FermiNet-style electron feature merging before each layer."""

    def __init__(
        self,
        width: Sequence[int],
        *,
        electrons: Sequence[int],
        layer_type: str = "base",
        required_parameters: Optional[dict] = None,
        project_context: bool = True,
        projection_activation: str = "silu",
        projection_rwf: Optional[dict[str, float]] = None,
        seed: int = 42,
    ):
        dims = [int(dim) for dim in width]
        if len(dims) < 2:
            raise ValueError("FermiNetStreamKAN width must include input and output dimensions.")
        if any(dim <= 0 for dim in dims):
            raise ValueError("FermiNetStreamKAN dimensions must be positive.")
        if len(tuple(electrons)) == 0 or sum(int(v) for v in electrons) <= 0:
            raise ValueError("FermiNetStreamKAN requires at least one electron.")

        self.width = tuple(dims)
        self.electrons = tuple(int(v) for v in electrons)
        self.layer_type = str(layer_type).lower()
        self.required_parameters = dict(required_parameters or {})
        self.project_context = bool(project_context)
        self.projection_activation_name = str(projection_activation)
        self.seed = int(seed)

        LayerClass = get_layer(self.layer_type)
        projection_activation_fn = get_activation(self.projection_activation_name)
        projection_kwargs = {"add_bias": True}
        if projection_rwf is not None:
            projection_kwargs["RWF"] = dict(projection_rwf)
        self.context_projectors = nnx.List(
            [
                DenseLayer(
                    n_in=4 * dims[idx],
                    n_out=dims[idx],
                    activation=projection_activation_fn,
                    seed=self.seed + 1009 + idx,
                    **projection_kwargs,
                )
                for idx in range(len(dims) - 1)
            ]
            if self.project_context
            else []
        )
        self.layers = nnx.List(
            [
                LayerClass(
                    n_in=dims[idx] if self.project_context else 4 * dims[idx],
                    n_out=dims[idx + 1],
                    **self.required_parameters,
                    seed=self.seed + idx,
                )
                for idx in range(len(dims) - 1)
            ]
        )

    def _spin_slices(self) -> list[tuple[int, int]]:
        start = 0
        slices = []
        for count in self.electrons:
            stop = start + int(count)
            if stop > start:
                slices.append((start, stop))
            start = stop
        return slices

    def _merge_electron_streams(self, x: Array) -> Array:
        global_mean = jnp.mean(x, axis=0, keepdims=True)
        spin_slices = self._spin_slices()
        contexts = []
        for index, (start, stop) in enumerate(spin_slices):
            local = x[start:stop]
            same_mean = jnp.mean(local, axis=0, keepdims=True)
            opposite_parts = [
                x[other_start:other_stop]
                for other_index, (other_start, other_stop) in enumerate(spin_slices)
                if other_index != index
            ]
            if opposite_parts:
                opposite = jnp.concatenate(opposite_parts, axis=0)
                opposite_mean = jnp.mean(opposite, axis=0, keepdims=True)
            else:
                opposite_mean = jnp.zeros_like(same_mean)
            row_count = stop - start
            contexts.append(
                jnp.concatenate(
                    [
                        local,
                        jnp.broadcast_to(global_mean, (row_count, x.shape[-1])),
                        jnp.broadcast_to(same_mean, (row_count, x.shape[-1])),
                        jnp.broadcast_to(opposite_mean, (row_count, x.shape[-1])),
                    ],
                    axis=-1,
                )
            )
        return jnp.concatenate(contexts, axis=0)

    def _layer_input(self, x: Array, layer_idx: int) -> Array:
        context = self._merge_electron_streams(x)
        if self.project_context:
            return self.context_projectors[layer_idx](context)
        return context

    def update_grids(self, x: Array, G_new: int):
        for idx, layer in enumerate(self.layers):
            layer_input = self._layer_input(x, idx)
            if hasattr(layer, "update_grid"):
                layer.update_grid(layer_input, G_new)
            x = layer(layer_input)

    def extend_grids(self, x: Array, G_new: int, optimizer=None):
        del optimizer
        self.update_grids(x, G_new)

    def refine_grids(self, x: Array, G_new: int, optimizer=None):
        self.extend_grids(x, G_new, optimizer=optimizer)

    def __call__(self, x: Array) -> Array:
        for idx, layer in enumerate(self.layers):
            x = layer(self._layer_input(x, idx))
        return x


def construct_input_features(
    pos: Array,
    atoms: Array,
    ndim: int = 3,
) -> tuple[Array, Array, Array, Array]:
    """Construct electron-atom/electron-electron vectors and distances."""

    if atoms.shape[1] != ndim:
        raise ValueError(f"Expected atom coordinates with ndim={ndim}, got {atoms.shape}.")
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    n = ee.shape[0]
    r_ee = jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n))
    return ae, ee, r_ae, r_ee[..., None]


def construct_orbital_features(
    pos: Array,
    atoms: Array,
    ndim: int = 3,
    feature_mode: str = "one_body",
    spins: Optional[Array] = None,
    charges: Optional[Array] = None,
) -> Array:
    """Construct per-electron orbital features.

    ``one_body`` keeps the original features [r_ae, ae].
    ``ee_aggregate`` appends permutation-equivariant electron-electron summaries:
    sum_j 1/(1+r_ij) and sum_j (r_i-r_j)/(1+r_ij).
    ``physics_exp8`` keeps the same 8-dimensional one-atom layout as
    ``ee_aggregate`` but replaces [r, x, y, z] with
    [exp(-r), x exp(-r), y exp(-r), z exp(-r)].
    ``physics_exp_spin_ee9`` uses the same exponential one-body features,
    replaces the mixed ee density with same-/opposite-spin exp(-r_ij)
    densities, and keeps one total exp-weighted ee vector.
    ``p_orbital_coulomb_ee`` uses per-atom features
    [x, y, z, r, x exp(-Zr), y exp(-Zr), z exp(-Zr), Z/(1+Zr)]
    and appends the four stable electron-electron aggregate features.
    ``interaction_ee_spin`` uses the same charge-scaled one-body features and
    appends same-/opposite-spin electron-electron aggregates with stable
    Coulomb and exponential radial kernels.
    ``ee_aggregate_angles`` augments ``ee_aggregate`` with stable angular
    features [cos(theta), cos(phi), sin(phi)] for each electron-nucleus vector.
    """

    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=ndim)
    return orbital_features_from_components(
        ae,
        ee,
        r_ae,
        r_ee,
        feature_mode=feature_mode,
        spins=spins,
        charges=charges,
    )


def orbital_features_from_components(
    ae: Array,
    ee: Array,
    r_ae: Array,
    r_ee: Array,
    feature_mode: str = "one_body",
    spins: Optional[Array] = None,
    charges: Optional[Array] = None,
) -> Array:
    """Construct orbital features from precomputed geometric components."""

    mode = str(feature_mode).lower()
    one_body = jnp.concatenate((r_ae, ae), axis=2).reshape(ae.shape[0], -1)
    exp_decay = jnp.exp(-r_ae)
    exp_one_body = jnp.concatenate((exp_decay, ae * exp_decay), axis=2).reshape(
        ae.shape[0], -1
    )
    if mode in ("one_body", "base", "original"):
        return one_body
    if mode in ("exp_one_body", "physics_exp_one_body"):
        return exp_one_body
    if mode not in (
        "ee_aggregate",
        "ee_agg",
        "equivariant_ee",
        "physics_exp8",
        "physics_exp_replace",
        "exp_ee_aggregate",
        "physics_exp_spin_ee9",
        "exp_spin_ee9",
        "p_orbital_coulomb_ee",
        "p_orbital_coulomb_ee12",
        "cartesian_exp_coulomb_ee",
        "interaction_ee_spin",
        "rich_interaction_ee",
        "p_orbital_spin_ee",
        "spin_coulomb_exp_ee",
        "ee_aggregate_angles",
        "ee_angles",
        "angular_ee_aggregate",
    ):
        raise ValueError(f"Unsupported orbital feature_mode={feature_mode!r}.")

    n = ee.shape[0]
    offdiag = (1.0 - jnp.eye(n, dtype=ae.dtype))[..., None]
    inv_weight = offdiag / (1.0 + r_ee)
    ee_density = jnp.sum(inv_weight, axis=1)
    ee_vector = jnp.sum(-ee * inv_weight, axis=1)

    if mode in (
        "interaction_ee_spin",
        "rich_interaction_ee",
        "p_orbital_spin_ee",
        "spin_coulomb_exp_ee",
    ):
        if charges is None:
            raise ValueError(f"feature_mode={feature_mode!r} requires atom charges.")
        if spins is None:
            raise ValueError(f"feature_mode={feature_mode!r} requires spin labels.")
        z = jnp.reshape(jnp.asarray(charges, dtype=ae.dtype), (1, -1, 1))
        z_exp_decay = jnp.exp(-z * r_ae)
        stable_coulomb = z / (1.0 + z * r_ae)
        atom_features = jnp.concatenate(
            (ae, r_ae, ae * z_exp_decay, stable_coulomb),
            axis=2,
        ).reshape(ae.shape[0], -1)

        spin_labels = jnp.reshape(spins, (-1,))[:n]
        same_mask = (
            spin_labels[:, None] == spin_labels[None, :]
        ).astype(ae.dtype)[..., None] * offdiag
        opposite_mask = (
            spin_labels[:, None] != spin_labels[None, :]
        ).astype(ae.dtype)[..., None] * offdiag
        exp_kernel = jnp.exp(-r_ee)

        def spin_aggregate(mask, kernel):
            weight = mask * kernel
            return jnp.sum(weight, axis=1), jnp.sum(-ee * weight, axis=1)

        same_inv_density, same_inv_vector = spin_aggregate(
            same_mask, 1.0 / (1.0 + r_ee)
        )
        opposite_inv_density, opposite_inv_vector = spin_aggregate(
            opposite_mask, 1.0 / (1.0 + r_ee)
        )
        same_exp_density, same_exp_vector = spin_aggregate(same_mask, exp_kernel)
        opposite_exp_density, opposite_exp_vector = spin_aggregate(
            opposite_mask, exp_kernel
        )
        return jnp.concatenate(
            (
                atom_features,
                same_inv_density,
                same_inv_vector,
                opposite_inv_density,
                opposite_inv_vector,
                same_exp_density,
                same_exp_vector,
                opposite_exp_density,
                opposite_exp_vector,
            ),
            axis=1,
        )

    if mode in (
        "p_orbital_coulomb_ee",
        "p_orbital_coulomb_ee12",
        "cartesian_exp_coulomb_ee",
    ):
        if charges is None:
            raise ValueError(f"feature_mode={feature_mode!r} requires atom charges.")
        z = jnp.reshape(jnp.asarray(charges, dtype=ae.dtype), (1, -1, 1))
        z_exp_decay = jnp.exp(-z * r_ae)
        stable_coulomb = z / (1.0 + z * r_ae)
        p_orbital_features = jnp.concatenate(
            (ae, r_ae, ae * z_exp_decay, stable_coulomb),
            axis=2,
        ).reshape(ae.shape[0], -1)
        return jnp.concatenate((p_orbital_features, ee_density, ee_vector), axis=1)

    if mode in ("ee_aggregate_angles", "ee_angles", "angular_ee_aggregate"):
        eps = jnp.asarray(1.0e-8, dtype=ae.dtype)
        safe_r = jnp.sqrt(jnp.sum(ae * ae, axis=2, keepdims=True) + eps * eps)
        rho = jnp.sqrt(
            ae[..., 0:1] * ae[..., 0:1]
            + ae[..., 1:2] * ae[..., 1:2]
            + eps * eps
        )
        cos_theta = ae[..., 2:3] / safe_r
        cos_phi = ae[..., 0:1] / rho
        sin_phi = ae[..., 1:2] / rho
        angular_features = jnp.concatenate(
            (cos_theta, cos_phi, sin_phi),
            axis=2,
        ).reshape(ae.shape[0], -1)
        return jnp.concatenate(
            (one_body, angular_features, ee_density, ee_vector),
            axis=1,
        )

    if mode in ("physics_exp_spin_ee9", "exp_spin_ee9"):
        if spins is None:
            raise ValueError(f"feature_mode={feature_mode!r} requires spin labels.")
        spin_labels = jnp.reshape(spins, (-1,))[:n]
        same_mask = (
            spin_labels[:, None] == spin_labels[None, :]
        ).astype(ae.dtype)[..., None] * offdiag
        opposite_mask = (
            spin_labels[:, None] != spin_labels[None, :]
        ).astype(ae.dtype)[..., None] * offdiag
        ee_weight = jnp.exp(-r_ee)
        same_density = jnp.sum(same_mask * ee_weight, axis=1)
        opposite_density = jnp.sum(opposite_mask * ee_weight, axis=1)
        ee_vector = jnp.sum(-ee * offdiag * ee_weight, axis=1)
        return jnp.concatenate(
            (exp_one_body, same_density, opposite_density, ee_vector),
            axis=1,
        )

    atom_features = (
        exp_one_body
        if mode in ("physics_exp8", "physics_exp_replace", "exp_ee_aggregate")
        else one_body
    )
    return jnp.concatenate((atom_features, ee_density, ee_vector), axis=1)


def active_spin_channels(nspins: Sequence[int]) -> list[int]:
    return [int(spin) for spin in nspins if int(spin) > 0]


def slogdet(x: Array) -> tuple[Array, Array]:
    """Compute determinant phase/sign and log magnitude."""

    if x.shape[-1] == 1:
        value = x[..., 0, 0]
        if value.dtype in (jnp.complex64, jnp.complex128):
            sign = value / jnp.abs(value)
        else:
            sign = jnp.sign(value)
        logdet = jnp.log(jnp.abs(value))
    else:
        sign, logdet = jnp.linalg.slogdet(x)
    return sign, logdet


def logdet_matmul(xs: Sequence[Array], w: Optional[Array] = None) -> tuple[Array, Array]:
    """Combine determinant blocks in log-domain."""

    det1d = functools.reduce(
        lambda a, b: a * b,
        [x.reshape(-1) for x in xs if x.shape[-1] == 1],
        1,
    )
    phase_in, logdet = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]),
        [slogdet(x) for x in xs if x.shape[-1] > 1],
        (1, 0),
    )

    maxlogdet = jnp.max(logdet)
    det = phase_in * det1d * jnp.exp(logdet - maxlogdet)
    if w is None:
        result = jnp.sum(det)
    else:
        result = jnp.matmul(det, w)[0]

    if result.dtype in (jnp.complex64, jnp.complex128):
        phase_out = result / jnp.abs(result)
    else:
        phase_out = jnp.sign(result)
    log_out = jnp.log(jnp.abs(result)) + maxlogdet
    return phase_out, log_out
