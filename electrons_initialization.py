"""in this module, we need initialize the coordinates of electrons as the number of the walkers that we identify in the input parameters."""
import jax
import jax.numpy as jnp
from typing import Sequence, Mapping, Tuple
from tools.utils import system
from absl import logging


def _balanced_spin_assignment(
    atomic_spin_configs: Sequence[Tuple[int, int]],
    target_electrons: Tuple[int, int],
) -> Sequence[Tuple[int, int]]:
    """Distribute requested spin totals while preserving per-atom electron counts."""
    counts = [int(alpha) + int(beta) for alpha, beta in atomic_spin_configs]
    target_alpha, target_beta = (int(target_electrons[0]), int(target_electrons[1]))
    if target_alpha + target_beta != sum(counts):
        return list(atomic_spin_configs)

    configs = []
    alpha_remaining = target_alpha
    remaining_counts = sum(counts)
    for atom_index, electron_count in enumerate(counts):
        remaining_counts -= electron_count
        min_alpha = max(0, alpha_remaining - remaining_counts)
        max_alpha = min(electron_count, alpha_remaining)
        if min_alpha > max_alpha:
            return list(atomic_spin_configs)

        default_alpha = int(atomic_spin_configs[atom_index][0])
        preferred = electron_count / 2.0
        alpha = min(
            range(min_alpha, max_alpha + 1),
            key=lambda value: (abs(value - preferred), abs(value - default_alpha)),
        )
        configs.append((alpha, electron_count - alpha))
        alpha_remaining -= alpha

    if alpha_remaining != 0 or tuple(sum(x) for x in zip(*configs)) != (target_alpha, target_beta):
        return list(atomic_spin_configs)
    return configs


def _find_spin_assignment(
    atomic_spin_configs: Sequence[Tuple[int, int]],
    target_electrons: Tuple[int, int],
) -> Sequence[Tuple[int, int]]:
    """Choose per-atom spin orientations that match the requested spin totals."""
    configs = tuple((int(a), int(b)) for a, b in atomic_spin_configs)
    target = tuple(int(x) for x in target_electrons)

    def search(index, alpha_total, beta_total, chosen):
        if alpha_total > target[0] or beta_total > target[1]:
            return None
        if index == len(configs):
            return chosen if (alpha_total, beta_total) == target else None
        alpha, beta = configs[index]
        for candidate in ((alpha, beta), (beta, alpha)):
            result = search(
                index + 1,
                alpha_total + candidate[0],
                beta_total + candidate[1],
                chosen + (candidate,),
            )
            if result is not None:
                return result
        return None

    result = search(0, 0, 0, ())
    if result is not None:
        return list(result)
    return _balanced_spin_assignment(configs, target)



def _assign_spin_configuration(nalpha: int, nbeta: int, batch_size: int = 1) -> jnp.ndarray:
    spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
    return jnp.tile(spins[None], reps=(batch_size, 1))

def init_electrons(  # pylint: disable=dangerous-default-value
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    batch_size: int,
    init_width: float,
    core_electrons: Mapping[str, int] = {},
    max_iter: int = 10000,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    niter = 0
    electrons = tuple(int(x) for x in electrons)
    total_electrons = sum(atom.charge - core_electrons.get(atom.symbol, 0)
                          for atom in molecule)
    if total_electrons != sum(electrons):
        if len(molecule) == 1:
            atomic_spin_configs = [electrons]
        else:
            raise NotImplementedError('No initialization policy yet '
                                      'exists for charged molecules.')
    else:
        atomic_spin_configs = [
            (atom.element.nalpha - core_electrons.get(atom.symbol, 0) // 2,
             atom.element.nbeta - core_electrons.get(atom.symbol, 0) // 2)
            for atom in molecule
        ]
        assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
        atomic_spin_configs = _find_spin_assignment(atomic_spin_configs, electrons)

    if tuple(sum(x) for x in zip(*atomic_spin_configs)) == electrons:
        # Assign each electron to an atom initially.
        electron_positions = []
        for i in range(2):
            for j in range(len(molecule)):
                atom_position = jnp.asarray(molecule[j].coords)
                electron_positions.append(
                    jnp.tile(atom_position, atomic_spin_configs[j][i]))
        electron_positions = jnp.concatenate(electron_positions)
    else:
        logging.warning(
            'Failed to find a valid initial electron configuration after %i'
            ' iterations. Initializing all electrons from a Gaussian distribution'
            ' centred on the origin. This might require increasing the number of'
            ' iterations used for pretraining and MCMC burn-in. Consider'
            ' implementing a custom initialisation.',
            niter,
        )
        electron_positions = jnp.zeros(shape=(3 * sum(electrons),))

    # Create a batch of configurations with a Gaussian distribution about each atom.
    key, subkey = jax.random.split(key)
    electron_positions += (
            jax.random.normal(subkey, shape=(batch_size, electron_positions.size))
            * init_width
    )

    electron_spins = _assign_spin_configuration(
        electrons[0], electrons[1], batch_size
    )

    return electron_positions, electron_spins
