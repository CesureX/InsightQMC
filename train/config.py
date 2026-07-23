from datetime import datetime

import ml_collections
from tools.utils import system


def default() -> ml_collections.ConfigDict:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    cfg = ml_collections.ConfigDict({
        # 'batch_size': 32768,
        'batch_size': 8192,
        'layer_dims': [8, 8, 8, 6],
        'g': [10],
        'k': [3], #7
        #'grid_range': [[0, 2], [0, 2], [0, 2], [0, 2]],
        'grid_range': [-2, 2],
        'iterations': 2000,
        'preiterations': 1000,
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
        'pretrain_step_jit': True,
        'clip_local_energy': 5.0,
        'use_scan': False,
        'complex_output': True, #True is recommended for better performance
        'full_det': False,  # True: det(NxN); False: det(alpha) * det(beta)
        'ndeterminants': 1,
        'determinant_weights': True,
        'laplacian_method': 'default',
        't_init': 0,
        'debug': False,
        # Optimizer used for the VMC energy minimization. Pretraining always
        # uses Adam. Supported values: 'adam', 'adamw', 'rgn', 'kfac'.
        'optimizer': 'kfac',
        'learning_rate': 0.00002,
        'learning_rate_decay': 50000.0,
        'gradient_clip_norm': 1.0,
        'adamw': {
            # Decoupled weight decay applied to all trainable parameters.
            'weight_decay': 1.0e-4,
        },
        'rgn': {
            # Compile MCMC and the RGN/CG solve as two separate executables.
            'split_compilation': True,
            # P = H_RGN + (S + eta I) / epsilon.
            'epsilon': 0.01,
            'eta': 1.0e-3,
            'cg_maxiter': 20,
            'cg_tol': 1.0e-4,
            # Optional conservative scaling of the solved RGN direction.
            'step_scale': 1.0,
            'max_update_norm': 0.1,
        },
        'kfac': {
            'damping': 1.0e-3,
            'min_damping': 1.0e-4,
            'norm_constraint': 1.0e-3,
            'cov_ema_decay': 0.95,
            'invert_every': 1,
            'l2_reg': 0.0,
            # InsightQMC's KAN layers do not provide custom KFAC tags.
            'register_only_generic': True,
        },
        'multi_device': True,
        # 0 means use all local JAX devices. batch_size is the global batch and
        # must be divisible by the number of devices used.
        'num_devices': 0,
        # 1 keeps HF/DFT pretraining on a single visible device; 0 follows
        # num_devices so pretraining uses the same devices as VMC training.
        'pretrain_num_devices': 1,
        'reset_optimizer_on_resume': False,
        'resize_resumed_noise': 0.0,
        'envelope_on': True,
        'envelope_type': 'ferminet_angular', # isotropic, chebyshev, legendre, legendre_anisotropic, angular_momentum, legendre_angular, complex_angular_momentum, ferminet_angular
        'envelope_degree': 7,
        'add_bias': True,
        'external_weights': True,
        'mkan': {
            # Orbital MKAN receives one electron feature row at a time:
            # [r_ae, ae, ee_density, ee_vec] for Li when orbital_features='ee_aggregate'.
            # The final output is 2 * ndeterminants * nelectrons real channels when
            # complex_output=True.
            'layer_type': 'base',       # 原始 KAN B-spline
            # 'layer_type': 'spline',   # efficient KAN spline
            #'layer_type': 'chebyshev',
            # 'layer_type': 'legendre'
            # 'layer_type': 'rbf'
            # 'layer_type': 'sine'
            # 'layer_type': 'fourier'
            # 'layer_type': 'fastkan'
            # 'layer_type': 'relukan'
            # 'layer_type': 'wavkan'
            'input_dim': None,
            'output_dim': None,
            # Set to [n_sum, n_mult] pairs to open MKAN multiplication nodes.
            'width': None,
            'mult_arity': 2,
            'required_parameters': None,
            'orbital_head': {
                # When enabled, MKAN first returns node features and this head maps
                # those nodes to final orbital channels.
                'enabled': True,
                # type='mlp' with hidden_dims=[] is the shared linear map H M0 + b.
                'type': 'mlp',
                # shared_rows applies one shared head to every electron row of H.
                'input_mode': 'shared_rows',
                # [] makes the MLP head a single linear layer.
                'hidden_dims': [],
                # MKAN-only width. Use e.g. [None, [8, 4], None] for 4 multiplication nodes.
                'width': None,
                'mult_arity': 2,
                'layer_type': 'base',
                'required_parameters': None,
                'activation': 'silu',
                'bias': True,
                'rwf': {'mean': 1.0, 'std': 0.1},
            },
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
            'root_dir': f'outputs/Li/{timestamp}',
            'checkpoint_every': 50,
            'metrics_every': 5,
            'resume': False,
            'enable_tensorboard': True,
        },
    })
    return cfg
