import ml_collections
from tools.utils import system


def default() -> ml_collections.ConfigDict:

    cfg = ml_collections.ConfigDict({
        'batch_size': 2048, #4096
        'layer_dims': [4, 16, 16],
        'g': [10, 10, 10, 10],
        'k': [3, 3, 3, 3], #7
        'grid_range': [[0, 2], [0, 2], [0, 2], [0, 2]],
        'iterations': 100000,
        'preiterations': 5000,
        'run_pretrain': True,
        'seed': 42,
        'seed_electrons_coords': 22,
        'init_width': 0.1,
        'core_electrons': {},
        'pretrain_method': 'hf', #'pretrain_method': 'dft',
        'pretrain_basis': 'ccpvdz',
        'pretrain_restricted': False,
        'hf_states': 0,
        'hf_excitation_type': 'ordered',
        'dft_xc': 'pbe,pbe',
        'dft_grid_level': 3,
        'scf_fraction': 0.0,
        'nfeatures': 4,
        'mcmc_steps': 30,
        'mcmc_width': 0.02, #0.02 for C,
        'pretrain_mcmc_steps': 1,
        'pretrain_mcmc_width': 0.02,
        'clip_local_energy': 5.0,
        'use_scan': False,
        'complex_output': True, #True is recommended for better performance
        'full_det': False,  # True: det(NxN); False: det(alpha) * det(beta)
        'laplacian_method': 'default',
        't_init': 0,
        'debug': False,
        'learning_rate': 0.00004,
        'learning_rate_decay': 20000.0,
        'envelope_on': True,
        'envelope_type': 'chebyshev', #(isotropic), (chebyshev) is recommended for better performance
        'envelope_degree': 7,
        'add_bias': True,
        'external_weights': True,
        'mkan': {
            # Orbital MKAN receives one electron feature row at a time:
            # [r_ae, ae] for every atom, so input_dim defaults to nfeatures.
            # The final output is 2 * nelectrons real channels when
            'layer_type': 'base',       # 原始 KAN B-spline
            # 'layer_type': 'spline',   # efficient KAN spline
            #'layer_type': 'chebyshev',
            # 'layer_type': 'legendre'
            # 'layer_type': 'rbf'
            # 'layer_type': 'sine'
            # 'layer_type': 'fourier'
            'input_dim': None,
            'output_dim': None,
            'width': None,
            'mult_arity': 2,
            'required_parameters': None,
            'pretrain_phase_weight': 1.0e-2,
        },
        'grid_extension': {
            'enabled': False,
            'steps': [500],
            'g_values': [20],
            'sample_size': None,
        },
        'system': {
            'molecule': [system.Atom('C', (0, 0, 0))],
            'electrons': (4,2),
        },
        'jastrow': {
            'ee': True,
            'type': 'ferminet_plus', #pade, ferminet, ferminet_plus, ferminet_three_body
        },
        'output': {
            'root_dir': 'outputs/C_LZW6022246',
            'checkpoint_every': 50,
            'metrics_every': 5,
            'resume': True,
            'enable_tensorboard': True,
        },
    })
    return cfg
