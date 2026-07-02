"""Envelope functions for QMC wavefunctions."""

from typing import Sequence

import jax.numpy as jnp

from networks import Array


LEGENDRE_ANISOTROPIC_TYPES = frozenset(
    ("legendre_anisotropic", "anisotropic_legendre", "ferminet_legendre")
)


def is_legendre_anisotropic(envelope_type: str) -> bool:
    """Return whether an envelope type names the anisotropic Legendre envelope."""

    return str(envelope_type).lower() in LEGENDRE_ANISOTROPIC_TYPES


def init_isotropic_envelope(natom: int, output_dims: Sequence[int]) -> list[dict[str, Array]]:
    """Initialize FermiNet-style isotropic exponential envelope parameters."""

    return [
        {
            "pi": jnp.ones((natom, int(output_dim))),
            "sigma": jnp.ones((natom, int(output_dim))),
        }
        for output_dim in output_dims
    ]


def apply_isotropic_envelope(*, r_ae: Array, pi: Array, sigma: Array) -> Array:
    """Evaluate sum_a pi_a exp(-sigma_a r_ae) for one spin channel."""

    return jnp.sum(jnp.exp(-r_ae * sigma) * pi, axis=1)


def init_chebyshev_envelope(
    natom: int,
    output_dims: Sequence[int],
    degree: int = 5,
) -> list[dict[str, Array]]:
    """Initialize a Chebyshev-modulated isotropic envelope.

    The degree-0 coefficient is initialized to one and higher-order coefficients
    to zero, so this starts equivalent to the isotropic envelope:
    sum_a pi_a exp(-sigma_a r_ae).
    """

    return [
        {
            "sigma": jnp.ones((natom, int(output_dim))),
            "c_basis": jnp.concatenate(
                [
                    jnp.ones((natom, 1, int(output_dim))),
                    jnp.zeros((natom, degree, int(output_dim))),
                ],
                axis=1,
            ),
        }
        for output_dim in output_dims
    ]


def _chebyshev_basis(x: Array, degree: int) -> Array:
    """Compute Chebyshev basis T_0 through T_degree on tanh-scaled inputs."""

    x = jnp.tanh(x)
    values = [jnp.ones_like(x)]
    if degree >= 1:
        values.append(x)
    for order in range(2, degree + 1):
        values.append(2.0 * x * values[-1] - values[-2])
    return jnp.stack(values, axis=-1)


def apply_chebyshev_envelope(
    *,
    r_ae: Array,
    sigma: Array,
    c_basis: Array,
) -> Array:
    """Evaluate a Chebyshev-modulated isotropic envelope.

    Args:
      r_ae: Electron-atom distances with shape (n_electrons_spin, natom, 1).
      sigma: Isotropic decay rates with shape (natom, output_dim).
      c_basis: Chebyshev coefficients with shape (natom, degree + 1, output_dim).
    """

    degree = c_basis.shape[1] - 1
    basis = _chebyshev_basis(r_ae[..., 0], degree)
    cheb_modulation = jnp.einsum("nad,ado->nao", basis, c_basis)
    isotropic_decay = jnp.exp(-r_ae * sigma)
    return jnp.sum(isotropic_decay * cheb_modulation, axis=1)


def init_legendre_envelope(
    natom: int,
    output_dims: Sequence[int],
    degree: int = 5,
) -> list[dict[str, Array]]:
    """Initialize a Legendre-modulated isotropic envelope.

    The degree-0 coefficient is initialized to one and higher-order coefficients
    to zero, so this starts equivalent to the isotropic envelope:
    sum_a exp(-sigma_a r_ae).
    """

    return [
        {
            "sigma": jnp.ones((natom, int(output_dim))),
            "p_basis": jnp.concatenate(
                [
                    jnp.ones((natom, 1, int(output_dim))),
                    jnp.zeros((natom, degree, int(output_dim))),
                ],
                axis=1,
            ),
        }
        for output_dim in output_dims
    ]


def _legendre_basis(x: Array, degree: int) -> Array:
    """Compute Legendre basis P_0 through P_degree on tanh-scaled inputs."""

    x = jnp.tanh(x)
    values = [jnp.ones_like(x)]
    if degree >= 1:
        values.append(x)
    for order in range(2, degree + 1):
        values.append(((2 * order - 1) * x * values[-1] - (order - 1) * values[-2]) / order)
    return jnp.stack(values, axis=-1)


def apply_legendre_envelope(
    *,
    r_ae: Array,
    sigma: Array,
    p_basis: Array,
) -> Array:
    """Evaluate a Legendre-modulated isotropic envelope.

    Args:
      r_ae: Electron-atom distances with shape (n_electrons_spin, natom, 1).
      sigma: Isotropic decay rates with shape (natom, output_dim).
      p_basis: Legendre coefficients with shape (natom, degree + 1, output_dim).
    """

    degree = p_basis.shape[1] - 1
    basis = _legendre_basis(r_ae[..., 0], degree)
    legendre_modulation = jnp.einsum("nad,ado->nao", basis, p_basis)
    isotropic_decay = jnp.exp(-r_ae * sigma)
    return jnp.sum(isotropic_decay * legendre_modulation, axis=1)


def init_legendre_anisotropic_envelope(
    natom: int,
    output_dims: Sequence[int],
    degree: int = 5,
    ndim: int = 3,
) -> list[dict[str, Array]]:
    """Initialize FermiNet-style anisotropic Legendre envelope parameters.

    ``sigma`` stores one full ``ndim x ndim`` decay matrix for every
    atom/output channel. It is initialized to the identity so the envelope
    starts as the isotropic Legendre envelope with unit decay.
    """

    eye = jnp.eye(ndim)
    return [
        {
            "sigma": jnp.broadcast_to(
                eye[None, None, :, :],
                (natom, int(output_dim), ndim, ndim),
            ),
            "p_basis": jnp.concatenate(
                [
                    jnp.ones((natom, 1, int(output_dim))),
                    jnp.zeros((natom, degree, int(output_dim))),
                ],
                axis=1,
            ),
        }
        for output_dim in output_dims
    ]


def apply_legendre_anisotropic_envelope(
    *,
    ae: Array,
    sigma: Array,
    p_basis: Array,
) -> Array:
    """Evaluate an anisotropic Legendre-modulated exponential envelope.

    Args:
      ae: Electron-atom displacement vectors with shape
        (n_electrons_spin, natom, ndim).
      sigma: Anisotropic decay matrices with shape
        (natom, output_dim, ndim, ndim).
      p_basis: Legendre coefficients with shape (natom, degree + 1, output_dim).
    """

    transformed = jnp.einsum("nai,aoij->naoj", ae, sigma)
    anisotropic_radius = jnp.linalg.norm(transformed, axis=-1)
    degree = p_basis.shape[1] - 1
    basis = _legendre_basis(anisotropic_radius, degree)
    legendre_modulation = jnp.einsum("naod,ado->nao", basis, p_basis)
    anisotropic_decay = jnp.exp(-anisotropic_radius)
    return jnp.sum(anisotropic_decay * legendre_modulation, axis=1)
