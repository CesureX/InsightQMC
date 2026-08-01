"""FastKAN layer using fixed Gaussian RBFs and input layer normalization."""

import jax.numpy as jnp
from jax import lax
from flax import nnx


class FastKANLayer(nnx.Module):
    """JAX/NNX implementation of the FastKAN layer."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        D: int = 8,
        grid_range=(-2.0, 2.0),
        use_layernorm: bool = True,
        use_base_update: bool = True,
        base_activation=nnx.silu,
        spline_weight_init_scale: float = 0.1,
        layernorm_eps: float = 1.0e-5,
        add_bias: bool = True,
        seed: int = 42,
    ):
        if D < 2:
            raise ValueError("FastKAN requires D >= 2.")
        self.n_in, self.n_out, self.D = int(n_in), int(n_out), int(D)
        self.grid_range = tuple(float(v) for v in grid_range)
        self.use_layernorm = bool(use_layernorm)
        self.use_base_update = bool(use_base_update)
        self.base_activation = base_activation
        self.residual = base_activation if self.use_base_update else None
        self.c_spl = None
        self.layernorm_eps = float(layernorm_eps)
        self.grid = nnx.Variable(
            jnp.linspace(self.grid_range[0], self.grid_range[1], self.D)
        )
        self.denominator = (self.grid_range[1] - self.grid_range[0]) / (self.D - 1)

        rngs = nnx.Rngs(seed)
        self.c_basis = nnx.Param(
            nnx.initializers.truncated_normal(stddev=spline_weight_init_scale)(
                rngs.params(), (self.n_out, self.n_in, self.D), jnp.float32
            )
        )
        if self.use_layernorm:
            self.ln_scale = nnx.Param(jnp.ones((self.n_in,)))
            self.ln_bias = nnx.Param(jnp.zeros((self.n_in,)))
        if self.use_base_update:
            self.c_res = nnx.Param(
                nnx.initializers.lecun_normal()(
                    rngs.params(), (self.n_out, self.n_in), jnp.float32
                )
            )
            self.base_bias = nnx.Param(jnp.zeros((self.n_out,)))
        self.bias = nnx.Param(jnp.zeros((self.n_out,))) if add_bias else None

    def normalize(self, x):
        if not self.use_layernorm:
            return x
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        normalized = (x - mean) * jax_lax_rsqrt(var + self.layernorm_eps)
        return normalized * self.ln_scale[...] + self.ln_bias[...]

    def basis(self, x):
        x = self.normalize(x)
        return jnp.exp(
            -jnp.square((x[..., None] - self.grid[...]) / self.denominator)
        )

    def __call__(self, x):
        batch = x.shape[0]
        basis = self.basis(x).reshape(batch, -1)
        basis_weights = self.c_basis[...].reshape(self.n_out, -1)
        y = jnp.matmul(
            basis,
            basis_weights.T,
            precision=lax.Precision.HIGHEST,
        )
        if self.use_base_update:
            y += jnp.matmul(
                self.base_activation(x),
                self.c_res[...].T,
                precision=lax.Precision.HIGHEST,
            )
            y += self.base_bias[...]
        if self.bias is not None:
            y += self.bias[...]
        return y

    def edge_activations(self, x):
        basis = self.basis(x)
        edges = jnp.einsum("bid,oid->boi", basis, self.c_basis[...])
        if self.use_base_update:
            edges += self.base_activation(x)[:, None, :] * self.c_res[...][None, :, :]
            edges += self.base_bias[...][None, :, None] / self.n_in
        return edges


def jax_lax_rsqrt(x):
    """Keep the normalization implementation backend-friendly."""
    return jnp.reciprocal(jnp.sqrt(x))
