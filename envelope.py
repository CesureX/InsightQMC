"""Envelope functions for QMC wavefunctions."""

import math
from typing import Sequence

import jax.numpy as jnp

from networks import Array


LEGENDRE_ANISOTROPIC_TYPES = frozenset(
    ("legendre_anisotropic", "anisotropic_legendre", "ferminet_legendre")
)
ANGULAR_MOMENTUM_TYPES = frozenset(
    ("angular_momentum", "angular_legendre", "spherical_harmonic", "spherical_harmonics")
)
LEGENDRE_ANGULAR_TYPES = frozenset(
    (
        "legendre_angular",
        "radial_legendre_angular",
        "legendre_angular_momentum",
        "legendre_spherical_harmonic",
        "legendre_spherical_harmonics",
    )
)
COMPLEX_ANGULAR_MOMENTUM_TYPES = frozenset(
    (
        "complex_angular_momentum",
        "complex_angular_legendre",
        "complex_spherical_harmonic",
        "complex_spherical_harmonics",
    )
)
FERMINET_ANGULAR_TYPES = frozenset(
    (
        "ferminet_angular",
        "ferminet_angular_momentum",
        "ferminet_spherical_harmonic",
        "ferminet_spherical_harmonics",
        "anisotropic_angular",
        "anisotropic_angular_momentum",
    )
)


def is_legendre_anisotropic(envelope_type: str) -> bool:
    """Return whether an envelope type names the anisotropic Legendre envelope."""

    return str(envelope_type).lower() in LEGENDRE_ANISOTROPIC_TYPES


def is_angular_momentum(envelope_type: str) -> bool:
    """Return whether an envelope type names the angular-momentum envelope."""

    return str(envelope_type).lower() in ANGULAR_MOMENTUM_TYPES


def is_legendre_angular(envelope_type: str) -> bool:
    """Return whether an envelope type names the Legendre-angular envelope."""

    return str(envelope_type).lower() in LEGENDRE_ANGULAR_TYPES


def is_complex_angular_momentum(envelope_type: str) -> bool:
    """Return whether an envelope type names the complex angular envelope."""

    return str(envelope_type).lower() in COMPLEX_ANGULAR_MOMENTUM_TYPES


def is_ferminet_angular(envelope_type: str) -> bool:
    """Return whether an envelope type names the FermiNet-style angular envelope."""

    return str(envelope_type).lower() in FERMINET_ANGULAR_TYPES


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


def angular_coordinates(ae: Array, eps: float = 1.0e-12) -> tuple[Array, Array]:
    """Compute spherical angles from electron-atom displacement vectors.

    Args:
      ae: Electron-atom displacement vectors with shape (..., 3).

    Returns:
      theta and phi arrays with shape ae.shape[:-1].
    """

    r = jnp.linalg.norm(ae, axis=-1)
    safe_r = jnp.maximum(r, eps)
    cos_theta = jnp.clip(ae[..., 2] / safe_r, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    phi = jnp.arctan2(ae[..., 1], ae[..., 0])
    return theta, phi


def _associated_legendre(l: int, m: int, x: Array) -> Array:
    """Compute associated Legendre function P_l^m(x) for m >= 0."""

    if m < 0 or m > l:
        raise ValueError(f"Require 0 <= m <= l, got l={l}, m={m}.")

    p_mm = jnp.ones_like(x)
    if m > 0:
        sqrt_term = jnp.sqrt(jnp.maximum(0.0, 1.0 - x * x))
        factor = 1.0
        for _ in range(1, m + 1):
            p_mm = -p_mm * factor * sqrt_term
            factor += 2.0

    if l == m:
        return p_mm

    p_m1_m = (2 * m + 1) * x * p_mm
    if l == m + 1:
        return p_m1_m

    p_lm_minus_2 = p_mm
    p_lm_minus_1 = p_m1_m
    for ell in range(m + 2, l + 1):
        p_lm = ((2 * ell - 1) * x * p_lm_minus_1 - (ell + m - 1) * p_lm_minus_2) / (ell - m)
        p_lm_minus_2 = p_lm_minus_1
        p_lm_minus_1 = p_lm
    return p_lm_minus_1


def _spherical_harmonic_norm(l: int, m: int) -> float:
    """Normalization constant for complex spherical harmonics."""

    return math.sqrt(
        ((2 * l + 1) / (4.0 * math.pi))
        * (math.factorial(l - m) / math.factorial(l + m))
    )


def _real_spherical_harmonic_basis(theta: Array, phi: Array, degree: int) -> Array:
    """Compute real spherical harmonic basis up to angular momentum degree."""

    x = jnp.cos(theta)
    values = []
    for l in range(degree + 1):
        for m_signed in range(-l, l + 1):
            m = abs(m_signed)
            p_lm = _associated_legendre(l, m, x)
            norm = _spherical_harmonic_norm(l, m)
            if m_signed < 0:
                values.append(math.sqrt(2.0) * norm * p_lm * jnp.sin(m * phi))
            elif m_signed == 0:
                values.append(norm * p_lm)
            else:
                values.append(math.sqrt(2.0) * norm * p_lm * jnp.cos(m * phi))
    return jnp.stack(values, axis=-1)


def _associated_legendre_trig_components_from_cartesian(
    ae: Array,
    degree: int,
    eps: float = 1.0e-12,
) -> dict[tuple[int, int], tuple[Array, Array]]:
    """Return P_l^m(z/r) cos(m phi) and P_l^m(z/r) sin(m phi).

    The calculation uses Cartesian direction cosines and recurrence relations.
    This is algebraically equivalent to the theta/phi implementation but avoids
    differentiating through atan2 and arccos.
    """

    r = jnp.maximum(jnp.linalg.norm(ae, axis=-1), eps)
    x = ae[..., 0] / r
    y = ae[..., 1] / r
    z = ae[..., 2] / r
    zero = jnp.zeros_like(x)
    one = jnp.ones_like(x)

    components: dict[tuple[int, int], tuple[Array, Array]] = {(0, 0): (one, zero)}
    if degree <= 0:
        return components

    components[(1, 0)] = (z, zero)
    for l in range(2, degree + 1):
        p_lm_minus_2, _ = components[(l - 2, 0)]
        p_lm_minus_1, _ = components[(l - 1, 0)]
        p_lm = ((2 * l - 1) * z * p_lm_minus_1 - (l - 1) * p_lm_minus_2) / l
        components[(l, 0)] = (p_lm, zero)

    real_power = one
    imag_power = zero
    double_factorial = 1.0
    for m in range(1, degree + 1):
        real_power, imag_power = (
            real_power * x - imag_power * y,
            real_power * y + imag_power * x,
        )
        double_factorial *= 2 * m - 1
        scale = -double_factorial if m % 2 else double_factorial
        p_mm_cos = scale * real_power
        p_mm_sin = scale * imag_power
        components[(m, m)] = (p_mm_cos, p_mm_sin)

        if m < degree:
            factor = 2 * m + 1
            p_m1_m_cos = factor * z * p_mm_cos
            p_m1_m_sin = factor * z * p_mm_sin
            components[(m + 1, m)] = (p_m1_m_cos, p_m1_m_sin)

        for l in range(m + 2, degree + 1):
            p_lm_minus_2_cos, p_lm_minus_2_sin = components[(l - 2, m)]
            p_lm_minus_1_cos, p_lm_minus_1_sin = components[(l - 1, m)]
            denom = l - m
            p_lm_cos = (
                (2 * l - 1) * z * p_lm_minus_1_cos
                - (l + m - 1) * p_lm_minus_2_cos
            ) / denom
            p_lm_sin = (
                (2 * l - 1) * z * p_lm_minus_1_sin
                - (l + m - 1) * p_lm_minus_2_sin
            ) / denom
            components[(l, m)] = (p_lm_cos, p_lm_sin)

    return components


def _real_spherical_harmonic_basis_from_cartesian(
    ae: Array,
    degree: int,
    eps: float = 1.0e-12,
) -> Array:
    """Compute real spherical harmonics from Cartesian directions.

    This produces the same basis/order as ``_real_spherical_harmonic_basis``:
    for every l, channels are ordered by m=-l..l.  It avoids theta/phi and
    therefore avoids the polar-axis derivative singularity from atan2/arccos.
    """

    components = _associated_legendre_trig_components_from_cartesian(ae, degree, eps)
    values = []
    for l in range(degree + 1):
        for m_signed in range(-l, l + 1):
            m = abs(m_signed)
            p_lm_cos, p_lm_sin = components[(l, m)]
            norm = _spherical_harmonic_norm(l, m)
            if m_signed < 0:
                values.append(math.sqrt(2.0) * norm * p_lm_sin)
            elif m_signed == 0:
                values.append(norm * p_lm_cos)
            else:
                values.append(math.sqrt(2.0) * norm * p_lm_cos)
    return jnp.stack(values, axis=-1)


def _complex_spherical_harmonic_basis(theta: Array, phi: Array, degree: int) -> Array:
    """Compute complex spherical harmonic basis up to angular momentum degree."""

    x = jnp.cos(theta)
    values = []
    for l in range(degree + 1):
        for m_signed in range(-l, l + 1):
            m = abs(m_signed)
            p_lm = _associated_legendre(l, m, x)
            norm = _spherical_harmonic_norm(l, m)
            phase = jnp.cos(m * phi) + 1.0j * jnp.sin(m * phi)
            y_pos = norm * p_lm * phase
            if m_signed < 0:
                values.append(((-1.0) ** m) * jnp.conj(y_pos))
            else:
                values.append(y_pos)
    return jnp.stack(values, axis=-1)


def _complex_spherical_harmonic_basis_from_cartesian(
    ae: Array,
    degree: int,
    eps: float = 1.0e-12,
) -> Array:
    """Compute complex spherical harmonics from Cartesian directions."""

    components = _associated_legendre_trig_components_from_cartesian(ae, degree, eps)
    values = []
    for l in range(degree + 1):
        for m_signed in range(-l, l + 1):
            m = abs(m_signed)
            p_lm_cos, p_lm_sin = components[(l, m)]
            p_lm_phase = p_lm_cos + 1.0j * p_lm_sin
            y_pos = _spherical_harmonic_norm(l, m) * p_lm_phase
            if m_signed < 0:
                values.append(((-1.0) ** m) * jnp.conj(y_pos))
            else:
                values.append(y_pos)
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


def init_angular_momentum_envelope(
    natom: int,
    output_dims: Sequence[int],
    degree: int = 3,
) -> list[dict[str, Array]]:
    """Initialize an envelope using learnable real spherical-harmonic sums.

    The angular basis contains all real spherical harmonics Y_lm(theta, phi)
    with 0 <= l <= degree. Coefficients are initialized so the l=0 term equals
    one, making the initial envelope sum_A exp(-sigma_A r_iA).
    """

    basis_count = (degree + 1) ** 2
    y00_inverse = math.sqrt(4.0 * math.pi)
    return [
        {
            "sigma": jnp.ones((natom, int(output_dim))),
            "angular_coeff": jnp.concatenate(
                [
                    y00_inverse * jnp.ones((natom, 1, int(output_dim))),
                    jnp.zeros((natom, basis_count - 1, int(output_dim))),
                ],
                axis=1,
            ),
        }
        for output_dim in output_dims
    ]


def apply_angular_momentum_envelope(
    *,
    r_ae: Array,
    sigma: Array,
    angular_coeff: Array,
    ae: Array | None = None,
    theta: Array | None = None,
    phi: Array | None = None,
) -> Array:
    """Evaluate an exponential envelope modulated by angular-momentum functions.

    Args:
      r_ae: Electron-atom distances with shape (n_electrons_spin, natom, 1).
      ae: Electron-atom displacement vectors with shape
        (n_electrons_spin, natom, 3). Preferred for l <= 1 because it avoids
        spherical-coordinate derivative singularities.
      theta: Optional polar angles with shape (n_electrons_spin, natom).
      phi: Optional azimuthal angles with shape (n_electrons_spin, natom).
      sigma: Isotropic decay rates with shape (natom, output_dim).
      angular_coeff: Learnable coefficients with shape
        (natom, (degree + 1) ** 2, output_dim).
    """

    basis_count = int(angular_coeff.shape[1])
    degree = int(round(math.sqrt(basis_count))) - 1
    if ae is not None:
        angular_basis = _real_spherical_harmonic_basis_from_cartesian(ae, degree)
    elif theta is not None and phi is not None:
        angular_basis = _real_spherical_harmonic_basis(theta, phi, degree)
    else:
        raise ValueError("Need ae or theta/phi to evaluate angular envelope.")
    angular_modulation = jnp.einsum("nab,abo->nao", angular_basis, angular_coeff)
    isotropic_decay = jnp.exp(-r_ae * sigma)
    return jnp.sum(isotropic_decay * angular_modulation, axis=1)


def init_legendre_angular_envelope(
    natom: int,
    output_dims: Sequence[int],
    degree: int = 3,
) -> list[dict[str, Array]]:
    """Initialize radial Legendre times real angular-momentum envelope.

    This represents exp(-sigma r) times a learnable radial Legendre sum and a
    learnable real spherical-harmonic sum. Both sums initialize to one.
    """

    angular_basis_count = (degree + 1) ** 2
    y00_inverse = math.sqrt(4.0 * math.pi)
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
            "angular_coeff": jnp.concatenate(
                [
                    y00_inverse * jnp.ones((natom, 1, int(output_dim))),
                    jnp.zeros((natom, angular_basis_count - 1, int(output_dim))),
                ],
                axis=1,
            ),
        }
        for output_dim in output_dims
    ]


def apply_legendre_angular_envelope(
    *,
    r_ae: Array,
    sigma: Array,
    p_basis: Array,
    angular_coeff: Array,
    ae: Array | None = None,
    theta: Array | None = None,
    phi: Array | None = None,
) -> Array:
    """Evaluate exp(-sigma r) times radial Legendre and real angular sums."""

    radial_degree = p_basis.shape[1] - 1
    radial_basis = _legendre_basis(r_ae[..., 0], radial_degree)
    radial_modulation = jnp.einsum("nad,ado->nao", radial_basis, p_basis)

    angular_basis_count = int(angular_coeff.shape[1])
    angular_degree = int(round(math.sqrt(angular_basis_count))) - 1
    if ae is not None:
        angular_basis = _real_spherical_harmonic_basis_from_cartesian(ae, angular_degree)
    elif theta is not None and phi is not None:
        angular_basis = _real_spherical_harmonic_basis(theta, phi, angular_degree)
    else:
        raise ValueError("Need ae or theta/phi to evaluate Legendre-angular envelope.")
    angular_modulation = jnp.einsum("nab,abo->nao", angular_basis, angular_coeff)

    isotropic_decay = jnp.exp(-r_ae * sigma)
    return jnp.sum(isotropic_decay * radial_modulation * angular_modulation, axis=1)


def init_complex_angular_momentum_envelope(
    natom: int,
    output_dims: Sequence[int],
    degree: int = 3,
) -> list[dict[str, Array]]:
    """Initialize an envelope using complex spherical-harmonic sums.

    Complex coefficients are stored as separate real and imaginary arrays so
    optimizers see real-valued trainable parameters.
    """

    basis_count = (degree + 1) ** 2
    y00_inverse = math.sqrt(4.0 * math.pi)
    return [
        {
            "sigma": jnp.ones((natom, int(output_dim))),
            "angular_coeff_real": jnp.concatenate(
                [
                    y00_inverse * jnp.ones((natom, 1, int(output_dim))),
                    jnp.zeros((natom, basis_count - 1, int(output_dim))),
                ],
                axis=1,
            ),
            "angular_coeff_imag": jnp.zeros((natom, basis_count, int(output_dim))),
        }
        for output_dim in output_dims
    ]


def apply_complex_angular_momentum_envelope(
    *,
    r_ae: Array,
    sigma: Array,
    angular_coeff_real: Array,
    angular_coeff_imag: Array,
    ae: Array | None = None,
    theta: Array | None = None,
    phi: Array | None = None,
) -> Array:
    """Evaluate an exponential envelope modulated by complex spherical harmonics."""

    basis_count = int(angular_coeff_real.shape[1])
    degree = int(round(math.sqrt(basis_count))) - 1
    if ae is not None:
        angular_basis = _complex_spherical_harmonic_basis_from_cartesian(ae, degree)
    elif theta is not None and phi is not None:
        angular_basis = _complex_spherical_harmonic_basis(theta, phi, degree)
    else:
        raise ValueError("Need ae or theta/phi to evaluate complex angular envelope.")
    angular_coeff = angular_coeff_real + 1.0j * angular_coeff_imag
    angular_modulation = jnp.einsum("nab,abo->nao", angular_basis, angular_coeff)
    isotropic_decay = jnp.exp(-r_ae * sigma)
    return jnp.sum(isotropic_decay * angular_modulation, axis=1)


def init_ferminet_angular_envelope(
    natom: int,
    output_dims: Sequence[int],
    degree: int = 3,
    ndim: int = 3,
) -> list[dict[str, Array]]:
    """Initialize FermiNet-style anisotropic angular envelope parameters.

    This represents a learnable FermiNet ``pi`` weight times an anisotropic
    exponential decay matrix and a real spherical-harmonic angular sum:
    pi * exp(-|Sigma (r - R)|) * sum_lm c_lm Y_lm(theta, phi).
    """

    angular_basis_count = (degree + 1) ** 2
    y00_inverse = math.sqrt(4.0 * math.pi)
    eye = jnp.eye(ndim)
    return [
        {
            "pi": jnp.ones((natom, int(output_dim))),
            "sigma": jnp.broadcast_to(
                eye[None, None, :, :],
                (natom, int(output_dim), ndim, ndim),
            ),
            "angular_coeff": jnp.concatenate(
                [
                    y00_inverse * jnp.ones((natom, 1, int(output_dim))),
                    jnp.zeros((natom, angular_basis_count - 1, int(output_dim))),
                ],
                axis=1,
            ),
        }
        for output_dim in output_dims
    ]


def apply_ferminet_angular_envelope(
    *,
    ae: Array,
    pi: Array,
    sigma: Array,
    angular_coeff: Array,
    theta: Array | None = None,
    phi: Array | None = None,
) -> Array:
    """Evaluate pi * exp(-|Sigma ae|) times a real angular-momentum sum."""

    transformed = jnp.einsum("nai,aoij->naoj", ae, sigma)
    anisotropic_radius = jnp.linalg.norm(transformed, axis=-1)

    angular_basis_count = int(angular_coeff.shape[1])
    angular_degree = int(round(math.sqrt(angular_basis_count))) - 1
    angular_basis = _real_spherical_harmonic_basis_from_cartesian(ae, angular_degree)
    angular_modulation = jnp.einsum("nab,abo->nao", angular_basis, angular_coeff)

    anisotropic_decay = jnp.exp(-anisotropic_radius)
    return jnp.sum(pi * anisotropic_decay * angular_modulation, axis=1)
