"""ReLU-KAN layer with compact, optionally trainable support intervals."""

import jax.numpy as jnp
from jax import lax
from flax import nnx


class ReLUKANLayer(nnx.Module):
    """JAX/NNX implementation of the ReLU-KAN basis."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        G: int = 5,
        k: int = 3,
        train_ab: bool = True,
        add_bias: bool = True,
        weight_init_scale: float = 0.1,
        seed: int = 42,
    ):
        if G <= 0 or k < 0:
            raise ValueError("ReLU-KAN requires G > 0 and k >= 0.")
        self.n_in, self.n_out = int(n_in), int(n_out)
        self.G, self.k, self.D = int(G), int(k), int(G + k)
        self.residual = None
        self.c_spl = None
        low = jnp.arange(-self.k, self.G, dtype=jnp.float32) / self.G
        high = low + (self.k + 1) / self.G
        low = jnp.tile(low[None, :], (self.n_in, 1))
        high = jnp.tile(high[None, :], (self.n_in, 1))
        if train_ab:
            self.phase_low = nnx.Param(low)
            self.phase_high = nnx.Param(high)
        else:
            self.phase_low = nnx.Variable(low)
            self.phase_high = nnx.Variable(high)

        rngs = nnx.Rngs(seed)
        self.c_basis = nnx.Param(
            nnx.initializers.normal(stddev=weight_init_scale)(
                rngs.params(), (self.n_out, self.n_in, self.D), jnp.float32
            )
        )
        self.bias = nnx.Param(jnp.zeros((self.n_out,))) if add_bias else None

    def basis(self, x):
        low = self.phase_low[...]
        high = self.phase_high[...]
        width = jnp.maximum(high - low, 1.0e-6)
        left = jax_relu(x[..., None] - low)
        right = jax_relu(high - x[..., None])
        return jnp.square(left * right) * (16.0 / jnp.power(width, 4))

    def __call__(self, x):
        batch = x.shape[0]
        basis = self.basis(x).reshape(batch, -1)
        basis_weights = self.c_basis[...].reshape(self.n_out, -1)
        y = jnp.matmul(
            basis, basis_weights.T, precision=lax.Precision.HIGHEST
        )
        if self.bias is not None:
            y += self.bias[...]
        return y

    def edge_activations(self, x):
        return jnp.einsum("bid,oid->boi", self.basis(x), self.c_basis[...])


def jax_relu(x):
    return jnp.maximum(x, 0.0)
