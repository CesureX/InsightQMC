import ml_collections
from ml_collections import config_dict

def default() -> ml_collections.ConfigDict:

    cfg = ml_collections.ConfigDict({
        'batch_size': 128,
        'pos': [[0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6],
                [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6]],
        'charges': [6.],
        'spins': [1, 1, 1, -1, -1, -1],
        'atoms': [[0.0, 0.0, 0.0]],
        'layer_dims': [4, 4, 4, 6],
        'g': [3, 3, 3,],
        'k': [3, 3, 3,],
        'grid_range':[[-10, 10], [-10, 10], [-10, 10]],
        'iterations': 1000,
        'preiterations': 1000,
        'seed': 42,
        'seed_electrons_coords': 22,
        'init_width': 0.1,
        'core_electrons': {},
        'hf_basis': 'ccpvdz',
        'hf_restricted': False,
        'hf_states': 0,
        'hf_excitation_type': 'ordered',
        'scf_fraction': 1.0,
        'nfeatures': 4,
        'mcmc_batch_per_device': 2,
        'mcmc_steps': 10,
        'mcmc_blocks': 1,
        'mcmc_width': 0.1,
        'clip_local_energy': 5.0,
        'use_scan': False,
        'complex_output': True,
        'laplacian_method': 'default',
        't_init': 0,
        'debug': False,
        'learning_rate': 0.05,
        'learning_rate_decay': 10000.0,
        'chebyshev': True,
        'spline': False,
        'envelope_chebyshev': False,
        'envelope_spline': False,
        'envelope_simple': True,
        'add_residual' : False,
        'add_bias': True,
        'external_weights': True,
        'system':{
            'molecule': config_dict.placeholder(list),
            'electrons': (3, 3),
            'nelectrons': 6,

        },
        'envelope':{
            'g_envelope': 10,
            'k_envelope': 3,
            'grid_range_envelope': [-10, 10],
        }
    })
    return cfg