"""Wavelet KAN layer."""

import jax.numpy as jnp
from flax import nnx


class WavKANLayer(nnx.Module):
    """Per-edge wavelet activations with trainable scale and translation."""

    _SUPPORTED = ("mexican_hat", "morlet", "dog", "meyer", "shannon")

    def __init__(
        self,
        n_in: int,
        n_out: int,
        wavelet_type: str = "mexican_hat",
        scale_eps: float = 1.0e-6,
        add_bias: bool = True,
        weight_init_scale: float = 1.0,
        seed: int = 42,
    ):
        wavelet_type = str(wavelet_type).lower()
        if wavelet_type not in self._SUPPORTED:
            raise ValueError(
                f"Unsupported wavelet_type={wavelet_type!r}; expected one of {self._SUPPORTED}."
            )
        self.n_in, self.n_out = int(n_in), int(n_out)
        self.wavelet_type = wavelet_type
        self.scale_eps = float(scale_eps)
        self.residual = None
        self.c_spl = None
        self.scale = nnx.Param(jnp.ones((self.n_out, self.n_in)))
        self.translation = nnx.Param(jnp.zeros((self.n_out, self.n_in)))
        rngs = nnx.Rngs(seed)
        self.c_basis = nnx.Param(
            nnx.initializers.variance_scaling(
                scale=weight_init_scale, mode="fan_in", distribution="uniform"
            )(rngs.params(), (self.n_out, self.n_in, 1), jnp.float32)
        )
        self.bias = nnx.Param(jnp.zeros((self.n_out,))) if add_bias else None

    def _wavelet(self, z):
        if self.wavelet_type == "mexican_hat":
            constant = 2.0 / (jnp.sqrt(3.0) * jnp.power(jnp.pi, 0.25))
            return constant * (jnp.square(z) - 1.0) * jnp.exp(-0.5 * jnp.square(z))
        if self.wavelet_type == "morlet":
            return jnp.exp(-0.5 * jnp.square(z)) * jnp.cos(5.0 * z)
        if self.wavelet_type == "dog":
            return -z * jnp.exp(-0.5 * jnp.square(z))
        if self.wavelet_type == "meyer":
            v = jnp.abs(z)
            t = jnp.clip(2.0 * v - 1.0, 0.0, 1.0)
            nu = t**4 * (35.0 - 84.0 * t + 70.0 * t**2 - 20.0 * t**3)
            aux = jnp.where(v <= 0.5, 1.0, jnp.where(v >= 1.0, 0.0, jnp.cos(0.5 * jnp.pi * nu)))
            return jnp.sin(jnp.pi * v) * aux
        # The source repository uses a Hamming-windowed Shannon wavelet.
        window = 0.54 - 0.46 * jnp.cos(
            2.0 * jnp.pi * jnp.arange(self.n_in) / max(self.n_in - 1, 1)
        )
        return jnp.sinc(z / jnp.pi) * window[None, None, :]

    def basis(self, x):
        scale = self.scale[...]
        safe_scale = jnp.where(
            jnp.abs(scale) < self.scale_eps,
            jnp.where(scale < 0, -self.scale_eps, self.scale_eps),
            scale,
        )
        z = (x[:, None, :] - self.translation[...][None, :, :]) / safe_scale[None, :, :]
        return self._wavelet(z)

    def __call__(self, x):
        y = jnp.sum(self.edge_activations(x), axis=-1)
        if self.bias is not None:
            y += self.bias[...]
        return y

    def edge_activations(self, x):
        return self.basis(x) * self.c_basis[...][..., 0][None, :, :]
