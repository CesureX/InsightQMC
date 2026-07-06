"""QMC-specific wavefunction components for trainer-side assembly."""

from __future__ import annotations

import functools
from typing import Any, Iterable, MutableMapping, Optional, Sequence, Union

import chex
import jax.numpy as jnp
from flax import nnx

from jkan.models.KAN import KAN
from jkan.models.MKAN import MultKAN
from jkan.layers.Dense import DenseLayer
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
        dense_kwargs = {
            "add_bias": bool(add_bias),
        }
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
) -> Array:
    """Construct per-electron orbital features.

    ``one_body`` keeps the original features [r_ae, ae].
    ``ee_aggregate`` appends permutation-equivariant electron-electron summaries:
    sum_j 1/(1+r_ij) and sum_j (r_i-r_j)/(1+r_ij).
    """

    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=ndim)
    return orbital_features_from_components(ae, ee, r_ae, r_ee, feature_mode=feature_mode)


def orbital_features_from_components(
    ae: Array,
    ee: Array,
    r_ae: Array,
    r_ee: Array,
    feature_mode: str = "one_body",
) -> Array:
    """Construct orbital features from precomputed geometric components."""

    one_body = jnp.concatenate((r_ae, ae), axis=2).reshape(ae.shape[0], -1)
    mode = str(feature_mode).lower()
    if mode in ("one_body", "base", "original"):
        return one_body
    if mode not in ("ee_aggregate", "ee_agg", "equivariant_ee"):
        raise ValueError(f"Unsupported orbital feature_mode={feature_mode!r}.")

    n = ee.shape[0]
    offdiag = (1.0 - jnp.eye(n, dtype=ae.dtype))[..., None]
    inv_weight = offdiag / (1.0 + r_ee)
    ee_density = jnp.sum(inv_weight, axis=1)
    ee_vector = jnp.sum(-ee * inv_weight, axis=1)
    return jnp.concatenate((one_body, ee_density, ee_vector), axis=1)


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
