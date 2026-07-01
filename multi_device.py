"""Helpers for single-host multi-device JAX training."""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import constants
import networks


DATA_IN_AXES = networks.KANetsData(
    positions=0,
    spins=None,
    atoms=None,
    charges=None,
)


def canonical_key(key: Any) -> jax.Array:
    """Returns a single PRNG key from old scalar or sharded checkpoint keys."""
    key = jnp.asarray(key)
    if key.shape == (2,):
        return key
    if key.shape[-1] != 2:
        raise ValueError(f'Expected a JAX PRNG key with trailing shape 2; got {key.shape}.')
    return jnp.reshape(key, (-1, 2))[0]


def split_step_key(key: Any, num_devices: int, use_pmap: bool):
    """Splits a host key into the next host key and step key(s)."""
    next_key, step_key = jax.random.split(canonical_key(key))
    if use_pmap:
        return next_key, _put_on_device_axis(jax.random.split(step_key, num_devices), num_devices)
    return next_key, step_key


def _device_axis_sharding(num_devices: int):
    devices = tuple(jax.local_devices()[:num_devices])
    mesh = Mesh(devices, (constants.PMAP_AXIS_NAME,))
    return NamedSharding(mesh, PartitionSpec(constants.PMAP_AXIS_NAME))


def _put_on_device_axis(value: Any, num_devices: int):
    return jax.device_put(value, _device_axis_sharding(num_devices))


def shard_data(data: networks.KANetsData, num_devices: int) -> networks.KANetsData:
    """Splits walker positions across local devices, keeping metadata shared."""
    positions = jnp.asarray(data.positions)
    batch_size = int(positions.shape[0])
    if batch_size % num_devices != 0:
        raise ValueError(
            f'batch_size={batch_size} must be divisible by num_devices={num_devices} '
            'when multi_device is enabled.'
        )
    per_device = batch_size // num_devices
    positions = jnp.reshape(positions, (num_devices, per_device, *positions.shape[1:]))
    positions = _put_on_device_axis(positions, num_devices)
    return networks.KANetsData(
        positions=positions,
        spins=data.spins,
        atoms=data.atoms,
        charges=data.charges,
    )


def unshard_data(data: networks.KANetsData) -> networks.KANetsData:
    """Merges sharded walker positions back into a host-shaped batch."""
    positions = jnp.asarray(data.positions)
    if positions.ndim >= 3:
        positions = jnp.reshape(positions, (-1, *positions.shape[2:]))
    return networks.KANetsData(
        positions=positions,
        spins=data.spins,
        atoms=data.atoms,
        charges=data.charges,
    )


def replicate(tree: Any, devices: Sequence[jax.Device]):
    if tree is None:
        return None
    devices = tuple(devices)
    num_devices = len(devices)
    mesh = Mesh(devices, (constants.PMAP_AXIS_NAME,))
    sharding = NamedSharding(mesh, PartitionSpec(constants.PMAP_AXIS_NAME))

    def replicate_leaf(leaf):
        leaf = jnp.asarray(leaf)
        return jax.device_put(
            jnp.broadcast_to(leaf, (num_devices, *leaf.shape)),
            sharding,
        )

    return jax.tree_util.tree_map(replicate_leaf, tree)


def unreplicate(tree: Any):
    if tree is None:
        return None
    return jax.tree_util.tree_map(lambda x: x[0], tree)
