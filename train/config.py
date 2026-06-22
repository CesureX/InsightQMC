import ml_collections
from tools.utils import system


def default() -> ml_collections.ConfigDict:

    cfg = ml_collections.ConfigDict({
        'batch_size': 2048,
        'layer_dims': [8, 16, 24],
        'g': [10],
        'k': [3], #7
        #'grid_range': [[0, 2], [0, 2], [0, 2], [0, 2]],
        'grid_range': [-2, 2],
        'iterations': 50000,
        'preiterations': 4000,
        'run_pretrain': True,
        'seed': 42,
        'seed_electrons_coords': 22,
        'init_width': 0.1,
        'core_electrons': {},
        'pretrain_method': 'hf', #'pretrain_method': 'dft',
        'pretrain_basis': 'ccpvtz',
        'pretrain_restricted': False,
        'hf_states': 0,
        'hf_excitation_type': 'ordered',
        'dft_xc': 'pbe,pbe',
        'dft_grid_level': 3,
        'scf_fraction': 0.0,
        'nfeatures': 8,
        'orbital_features': 'ee_aggregate',
        'mcmc_steps': 50,
        'mcmc_width': 0.05,
        'pretrain_mcmc_steps': 1,
        'pretrain_mcmc_width': 0.02,
        'clip_local_energy': 5.0,
        'use_scan': False,
        'complex_output': True, #True is recommended for better performance
        'full_det': False,  # True: det(NxN); False: det(alpha) * det(beta)
        'ndeterminants': 4,
        'determinant_weights': True,
        'laplacian_method': 'default',
        't_init': 0,
        'debug': False,
        'learning_rate': 0.00002,
        'learning_rate_decay': 50000.0,
        'gradient_clip_norm': 1.0,
        'reset_optimizer_on_resume': True,
        'resize_resumed_noise': 0.0,
        'envelope_on': True,
        'envelope_type': 'chebyshev', #(isotropic), (chebyshev) is recommended for better performance
        'envelope_degree': 7,
        'add_bias': True,
        'external_weights': True,
        'mkan': {
            # Orbital MKAN receives one electron feature row at a time:
            # [r_ae, ae, ee_density, ee_vec] for Li when orbital_features='ee_aggregate'.
            # The final output is 2 * ndeterminants * nelectrons real channels when
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
            'prune_mask_checkpoint': None,
        },
        'grid_extension': {
            'enabled': False,
            'steps': [],
            'g_values': [],
            'sample_size': 4096,
        },
        'system': {
            'molecule': [system.Atom('Li', (0, 0, 0))],
            'electrons': (2,1),
        },
        'jastrow': {
            'ee': True,
            'en': False,
            'en_radial_order': 4,
            'radial_order': 4,
            'type': 'ferminet_plus', #pade, ferminet, ferminet_plus, ferminet_three_body
        },
        'output': {
            'root_dir': 'outputs/Li622_eeagg_ndet4_wdet_g10',
            'checkpoint_every': 50,
            'metrics_every': 5,
            'resume': False,
            'enable_tensorboard': True,
        },
    })
    return cfg
