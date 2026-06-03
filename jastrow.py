"""Jastrow factors for QMC wavefunctions."""

from typing import Callable, Mapping, Sequence

import jax.numpy as jnp

from networks import Array


def spin_pair_indices(nspins: Sequence[int]) -> tuple[Array, Array]:
    """Return same-spin and opposite-spin electron-pair indices."""

    n_alpha, n_beta = (int(nspins[0]), int(nspins[1]))
    alpha = list(range(n_alpha))
    beta = list(range(n_alpha, n_alpha + n_beta))

    same_spin = []
    for group in (alpha, beta):
        for i, electron_i in enumerate(group):
            for electron_j in group[i + 1 :]:
                same_spin.append((electron_i, electron_j))

    opposite_spin = [(electron_i, electron_j) for electron_i in alpha for electron_j in beta]

    same_spin = same_spin or [(0, 0)]
    opposite_spin = opposite_spin or [(0, 0)]
    return jnp.asarray(same_spin, dtype=jnp.int32), jnp.asarray(opposite_spin, dtype=jnp.int32)


def spin_pair_indices_or_empty(nspins: Sequence[int]) -> tuple[Array, Array]:
    """Return same-spin and opposite-spin pair indices without sentinel pairs."""

    n_alpha, n_beta = (int(nspins[0]), int(nspins[1]))
    alpha = list(range(n_alpha))
    beta = list(range(n_alpha, n_alpha + n_beta))

    same_spin = []
    for group in (alpha, beta):
        for i, electron_i in enumerate(group):
            for electron_j in group[i + 1 :]:
                same_spin.append((electron_i, electron_j))

    opposite_spin = [(electron_i, electron_j) for electron_i in alpha for electron_j in beta]
    return (
        jnp.asarray(same_spin, dtype=jnp.int32).reshape((-1, 2)),
        jnp.asarray(opposite_spin, dtype=jnp.int32).reshape((-1, 2)),
    )


def init_pade_ee_jastrow() -> dict[str, Array]:
    """Initialize electron-electron Pade Jastrow variational parameters."""

    return {
        "ee_par": jnp.ones((1,)),
        "ee_anti": jnp.ones((1,)),
    }


def init_ferminet_ee_jastrow() -> dict[str, Array]:
    """Initialize FermiNet simple electron-electron Jastrow parameters."""

    return {
        "ee_par": jnp.ones((1,)),
        "ee_anti": jnp.ones((1,)),
    }


def init_ferminet_plus_ee_jastrow() -> dict[str, Array]:
    """Initialize FermiNet cusp Jastrow with extra radial correction terms."""

    return {
        "ee_par": jnp.ones((1,)),
        "ee_anti": jnp.ones((1,)),
        "ee_par_coeff": jnp.zeros((4,)),
        "ee_anti_coeff": jnp.zeros((4,)),
    }


def init_ferminet_three_body_jastrow() -> dict[str, Array]:
    """Initialize pair-cusp Jastrow plus electron-electron-nucleus terms."""

    params = init_ferminet_plus_ee_jastrow()
    params["een_par_coeff"] = jnp.zeros((4,))
    params["een_anti_coeff"] = jnp.zeros((4,))
    return params


def _pair_distances(r_ee: Array, pair_indices: Array) -> Array:
    return r_ee[pair_indices[:, 0], pair_indices[:, 1], 0]


def _pair_electron_nucleus_features(r_ae: Array, pair_indices: Array) -> tuple[Array, Array]:
    proximity = 1.0 / (1.0 + r_ae[..., 0])
    electron_feature = jnp.sum(proximity, axis=-1)
    return electron_feature[pair_indices[:, 0]], electron_feature[pair_indices[:, 1]]


def _pade_cusp(r: Array, cusp: float, alpha: Array) -> Array:
    return cusp * r / (1.0 + alpha * r)


def _ferminet_cusp(r: Array, cusp: float, alpha: Array) -> Array:
    return -(cusp * alpha**2) / (alpha + r)


def _ferminet_plus_cusp(
    r: Array,
    cusp: float,
    alpha: Array,
    coeff: Array,
) -> Array:
    base = _ferminet_cusp(r, cusp, alpha)
    x = r / (1.0 + r)
    powers = x[..., None] ** jnp.arange(2, coeff.shape[0] + 2)
    return base + jnp.sum(powers * coeff, axis=-1)


def _three_body_correction(
    r_ee_pair: Array,
    r_iA_feature: Array,
    r_jA_feature: Array,
    coeff: Array,
) -> Array:
    x = r_ee_pair / (1.0 + r_ee_pair)
    y_sum = r_iA_feature + r_jA_feature
    y_prod = r_iA_feature * r_jA_feature
    y_diff2 = (r_iA_feature - r_jA_feature) ** 2
    features = jnp.stack(
        (
            x**2 * y_sum,
            x**2 * y_prod,
            x**2 * y_diff2,
            x**3 * y_sum,
        ),
        axis=-1,
    )
    return jnp.sum(features * coeff, axis=-1)


def _sum_pair_cusps(
    r_ee: Array,
    params: Mapping[str, Array],
    same_spin_pairs: Array,
    opposite_spin_pairs: Array,
    cusp_fn: Callable[[Array, float, Array], Array],
) -> Array:
    total = jnp.asarray(0.0)
    if same_spin_pairs.shape[0] > 0:
        r_parallel = _pair_distances(r_ee, same_spin_pairs)
        total = total + jnp.sum(cusp_fn(r_parallel, 0.25, params["ee_par"]))
    if opposite_spin_pairs.shape[0] > 0:
        r_antiparallel = _pair_distances(r_ee, opposite_spin_pairs)
        total = total + jnp.sum(cusp_fn(r_antiparallel, 0.5, params["ee_anti"]))
    return total


def apply_pade_ee_jastrow(
    r_ee: Array,
    params: Mapping[str, Array],
    same_spin_pairs: Array,
    opposite_spin_pairs: Array,
) -> Array:
    """Evaluate log J_ee for Pade electron-electron cusp factors."""

    return _sum_pair_cusps(
        r_ee,
        params,
        same_spin_pairs,
        opposite_spin_pairs,
        _pade_cusp,
    )


def apply_ferminet_ee_jastrow(
    r_ee: Array,
    params: Mapping[str, Array],
    same_spin_pairs: Array,
    opposite_spin_pairs: Array,
) -> Array:
    """Evaluate the FermiNet simple electron-electron Jastrow factor."""

    return _sum_pair_cusps(
        r_ee,
        params,
        same_spin_pairs,
        opposite_spin_pairs,
        _ferminet_cusp,
    )

def apply_ferminet_plus_ee_jastrow(
    r_ee: Array,
    params: Mapping[str, Array],
    same_spin_pairs: Array,
    opposite_spin_pairs: Array,
) -> Array:
    """Evaluate the FermiNet cusp Jastrow plus trainable radial corrections."""

    total = jnp.asarray(0.0)
    if same_spin_pairs.shape[0] > 0:
        r_parallel = _pair_distances(r_ee, same_spin_pairs)
        total = total + jnp.sum(
            _ferminet_plus_cusp(
                r_parallel,
                0.25,
                params["ee_par"],
                params["ee_par_coeff"],
            )
        )
    if opposite_spin_pairs.shape[0] > 0:
        r_antiparallel = _pair_distances(r_ee, opposite_spin_pairs)
        total = total + jnp.sum(
            _ferminet_plus_cusp(
                r_antiparallel,
                0.5,
                params["ee_anti"],
                params["ee_anti_coeff"],
            )
        )
    return total


def apply_ferminet_three_body_jastrow(
    r_ee: Array,
    r_ae: Array,
    params: Mapping[str, Array],
    same_spin_pairs: Array,
    opposite_spin_pairs: Array,
) -> Array:
    """Evaluate pair Jastrow plus bounded electron-electron-nucleus terms."""

    total = apply_ferminet_plus_ee_jastrow(
        r_ee,
        params,
        same_spin_pairs,
        opposite_spin_pairs,
    )
    if same_spin_pairs.shape[0] > 0:
        r_parallel = _pair_distances(r_ee, same_spin_pairs)
        r_iA, r_jA = _pair_electron_nucleus_features(r_ae, same_spin_pairs)
        total = total + jnp.sum(
            _three_body_correction(
                r_parallel,
                r_iA,
                r_jA,
                params["een_par_coeff"],
            )
        )
    if opposite_spin_pairs.shape[0] > 0:
        r_antiparallel = _pair_distances(r_ee, opposite_spin_pairs)
        r_iA, r_jA = _pair_electron_nucleus_features(r_ae, opposite_spin_pairs)
        total = total + jnp.sum(
            _three_body_correction(
                r_antiparallel,
                r_iA,
                r_jA,
                params["een_anti_coeff"],
            )
        )
    return total
