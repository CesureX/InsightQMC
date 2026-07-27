"""Smoke and regression tests for the imported KAN variants."""

import unittest

import jax.numpy as jnp

from . import get_layer


class VariantLayerTest(unittest.TestCase):

    def setUp(self):
        self.x = jnp.asarray(
            [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=jnp.float32
        )

    def _check_layer(self, name, **kwargs):
        layer = get_layer(name)(n_in=3, n_out=4, seed=7, **kwargs)
        output = layer(self.x)
        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(bool(jnp.all(jnp.isfinite(output))))
        edges = layer.edge_activations(self.x)
        self.assertEqual(edges.shape, (2, 4, 3))
        self.assertTrue(
            bool(jnp.allclose(output - layer.bias[...], jnp.sum(edges, axis=-1)))
        )
    def test_fastkan(self):
        self._check_layer("fastkan", D=8)

    def test_relukan(self):
        self._check_layer("relukan", G=5, k=3)

    def test_wavkan_wavelets(self):
        for wavelet_type in ("mexican_hat", "morlet", "dog", "meyer", "shannon"):
            with self.subTest(wavelet_type=wavelet_type):
                self._check_layer("wavkan", wavelet_type=wavelet_type)


if __name__ == "__main__":
    unittest.main()
