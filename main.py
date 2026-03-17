"""Project entry point for training."""
import jax

from train import config, train
from tools.utils import system


def train_config():
    """Build the default training config for the single-stream workflow."""
    cfg = config.default()

    cfg.system.electrons = (3, 3)
    cfg.system.nelectrons = 6
    cfg.system.molecule = [system.Atom('C', (0, 0, 0))]

    cfg.batch_size = 10
    cfg.layer_dims = [4, 20, 20, 20, 20]
    cfg.g = [10, 10, 10, 10]
    cfg.k = [3, 3, 3, 3]
    cfg.grid_range = [[0, 2], [0, 2], [0, 2], [0, 2]]

    cfg.envelope.g_envelope = 10
    cfg.envelope.k_envelope = 5
    cfg.envelope.grid_range_envelope = [0, 2]

    cfg.iterations = 100
    cfg.preiterations = 100
    cfg.chebyshev = True
    cfg.spline = False
    cfg.add_bias = True
    cfg.external_weights = True
    cfg.add_residual = True
    cfg.envelope_chebyshev = False
    cfg.envelope_spline = False
    cfg.envelope_simple = True

    return cfg


def main():
    jax.config.update("jax_traceback_filtering", "off")
    cfg = train_config()
    train.train(cfg)

if __name__ == "__main__":
    main()
