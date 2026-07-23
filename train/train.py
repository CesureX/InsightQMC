import pickle
from pathlib import Path
from typing import Any, Callable, Optional

import flax
from flax import nnx
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from tqdm.auto import trange

import constants
import hamiltonian
import electrons_initialization
import envelope
import jastrow
import multi_device
import networks
from jkan.models import MultKAN
import loss as qmc_loss_functions
import vmcmc
from opt import (
    make_kfac_training_step,
    make_opt_update_step,
    make_rgn_update_step,
    make_split_training_step,
    make_training_step,
)
from train.pretrain_runner import PretrainRunner
from train.training_io import RunManager


def _to_scalar(x):
    return float(jnp.asarray(x).reshape(-1)[0])


def _first_int(values, default: int) -> int:
    if values is None:
        return default
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return default
    return int(arr[0])


def _first_grid_range(values, default=(-1.0, 1.0)) -> tuple[float, float]:
    if values is None:
        return tuple(default)
    arr = np.asarray(values)
    if arr.ndim == 1 and arr.size >= 2:
        return (float(arr[0]), float(arr[1]))
    if arr.ndim >= 2 and arr.shape[-1] >= 2:
        return (float(arr.reshape(-1, arr.shape[-1])[0, 0]), float(arr.reshape(-1, arr.shape[-1])[0, 1]))
    return tuple(default)


def _array_partitions(sizes):
    return list(np.cumsum(tuple(int(size) for size in sizes)))[:-1]


def _mkan_width_output_dim(width) -> int:
    last = list(width)[-1]
    if isinstance(last, (list, tuple)):
        return int(last[0]) + int(last[1])
    return int(last)


def _merge_static_state_with_template(template_state, checkpoint_state):
    if checkpoint_state is None:
        return template_state
    try:
        merged = dict(template_state.flat_state())
        checkpoint_flat = dict(checkpoint_state.flat_state())
    except AttributeError:
        return checkpoint_state
    for path, value in checkpoint_flat.items():
        if path in merged:
            merged[path] = value
    return nnx.State.from_flat_path(merged)


def _construct_input_features(pos: jnp.ndarray, atoms: jnp.ndarray, ndim: int = 3):
    return networks.construct_input_features(pos, atoms, ndim=ndim)


@flax.struct.dataclass
class RuntimeState:
    data: networks.KANetsData
    key: Any
    mcmc_width: jnp.ndarray
    pmoves: np.ndarray


class VMCTrainer:
    """Flax-style trainer that keeps setup and loop logic modular."""

    def __init__(self, cfg: ml_collections.ConfigDict):
        self.cfg = cfg
        self._mkan_graphdef = None
        self._mkan_static_state = None
        self._orbital_head_graphdef = None
        self._orbital_head_static_state = None
        self.run_manager = RunManager(cfg.output)
        self.run_manager.save_config(cfg)
        self._read_config()
        self.pretrain_runner = PretrainRunner(
            run_manager=self.run_manager,
            build_checkpoint_state=self._build_checkpoint_state,
            enabled=self.run_pretrain,
            preiterations=self.preiterations,
            method=self.pretrain_method,
            pyscf_mol=self.pyscf_mol,
            molecule=self.molecule,
            electrons=self.electrons,
            restricted=self.pretrain_restricted,
            basis=self.pretrain_basis,
            core_electrons=self.core_electrons,
            hf_states=self.hf_states,
            hf_excitation_type=self.hf_excitation_type,
            dft_xc=self.dft_xc,
            dft_grid_level=self.dft_grid_level,
            scf_fraction=self.scf_fraction,
            batch_size=self.batch_size,
            pretrain_mcmc_steps=self.pretrain_mcmc_steps,
            pretrain_mcmc_width=self.pretrain_mcmc_width,
            step_jit=self.pretrain_step_jit,
            full_det=self.full_det,
            debug=self.debug,
            scalar_pretrain=False,
            phase_weight=self.mkan_pretrain_phase_weight,
            use_pmap=self.pretrain_use_pmap,
            devices=self.pretrain_devices,
            num_devices=self.pretrain_num_devices,
        )

    def _read_config(self) -> None:
        cfg = self.cfg
        self.molecule = cfg.system.molecule
        self.electrons = tuple(cfg.system.electrons)
        self.nelectrons = sum(self.electrons)
        self.natoms = len(self.molecule)

        self.batch_size = int(cfg.batch_size)
        nfeatures = int(cfg.nfeatures)
        self.atoms = jnp.array([atom.coords for atom in self.molecule])
        self.charges = jnp.array([atom.charge for atom in self.molecule])

        spins = [1] * self.electrons[0] + [-1] * self.electrons[1]
        self.spins = jnp.array([spins])

        self.g = jnp.array(cfg.g)
        self.k = jnp.array(cfg.k)
        self.layer_dims = jnp.array(cfg.layer_dims)
        self.grid_range = cfg.grid_range
        self.orbital_feature_mode = str(cfg.get('orbital_features', 'one_body')).lower()

        self.seed = int(cfg.seed)
        self.seed_electrons_coords = int(cfg.seed_electrons_coords)
        self.init_width = float(cfg.init_width)
        self.core_electrons = cfg.core_electrons

        self.pretrain_method = str(cfg.get('pretrain_method', 'hf')).lower()
        self.pretrain_basis = cfg.get('pretrain_basis', 'ccpvdz')
        self.pretrain_restricted = bool(cfg.get('pretrain_restricted', False))
        self.hf_states = int(cfg.get('hf_states', 0))
        self.hf_excitation_type = cfg.get('hf_excitation_type', 'ordered')
        self.dft_xc = cfg.get('dft_xc', 'pbe,pbe')
        self.dft_grid_level = cfg.get('dft_grid_level', 3)
        self.pyscf_mol = cfg.system.get('pyscf_mol')

        self.mcmc_steps = int(cfg.mcmc_steps)
        self.mcmc_width = float(cfg.mcmc_width)
        self.pretrain_mcmc_steps = int(cfg.get('pretrain_mcmc_steps', 1))
        self.pretrain_mcmc_width = float(cfg.get('pretrain_mcmc_width', 0.02))
        self.pretrain_step_jit = bool(cfg.get('pretrain_step_jit', True))

        self.clip_local_energy = float(cfg.clip_local_energy)
        self.use_scan = bool(cfg.use_scan)
        self.complex_output = bool(cfg.complex_output)
        self.full_det = bool(cfg.get('full_det', True))
        self.ndeterminants = int(cfg.get('ndeterminants', 1))
        if self.ndeterminants <= 0:
            raise ValueError('ndeterminants must be positive.')
        self.use_determinant_weights = bool(cfg.get('determinant_weights', self.ndeterminants > 1))
        self.laplacian_method = cfg.laplacian_method
        self.scf_fraction = float(cfg.scf_fraction)
        self.t_init = int(cfg.t_init)
        self.debug = bool(cfg.debug)

        self.learning_rate = float(cfg.learning_rate)
        self.learning_rate_decay = float(cfg.learning_rate_decay)
        self.gradient_clip_norm = float(cfg.get('gradient_clip_norm', 0.0))
        self.optimizer_name = str(cfg.get('optimizer', 'adam')).lower()
        if self.optimizer_name not in ('adam', 'adamw', 'rgn', 'kfac'):
            raise ValueError(
                f"Unsupported optimizer {self.optimizer_name!r}; "
                "expected 'adam', 'adamw', 'rgn', or 'kfac'."
            )
        adamw_cfg = cfg.get('adamw', {})
        self.adamw_weight_decay = float(
            adamw_cfg.get('weight_decay', 1.0e-4))
        if self.adamw_weight_decay < 0.0:
            raise ValueError('adamw.weight_decay must be non-negative.')
        rgn_cfg = cfg.get('rgn', {})
        self.rgn_epsilon = float(rgn_cfg.get('epsilon', 0.01))
        self.rgn_split_compilation = bool(
            rgn_cfg.get('split_compilation', True))
        self.rgn_eta = float(rgn_cfg.get('eta', 1.0e-3))
        self.rgn_cg_maxiter = int(rgn_cfg.get('cg_maxiter', 20))
        self.rgn_cg_tol = float(rgn_cfg.get('cg_tol', 1.0e-4))
        self.rgn_step_scale = float(rgn_cfg.get('step_scale', 1.0))
        self.rgn_max_update_norm = float(rgn_cfg.get('max_update_norm', 0.1))
        kfac_cfg = cfg.get('kfac', {})
        self.kfac_damping = float(kfac_cfg.get('damping', 1.0e-3))
        self.kfac_min_damping = float(kfac_cfg.get('min_damping', 1.0e-4))
        self.kfac_norm_constraint = float(kfac_cfg.get('norm_constraint', 1.0e-3))
        self.kfac_cov_ema_decay = float(kfac_cfg.get('cov_ema_decay', 0.95))
        self.kfac_invert_every = int(kfac_cfg.get('invert_every', 1))
        self.kfac_l2_reg = float(kfac_cfg.get('l2_reg', 0.0))
        self.kfac_register_only_generic = bool(
            kfac_cfg.get('register_only_generic', True))
        self.multi_device = bool(cfg.get('multi_device', True))
        self.requested_num_devices = int(cfg.get('num_devices', 0))
        if self.requested_num_devices < 0:
            raise ValueError('num_devices must be non-negative.')
        local_devices = tuple(jax.local_devices())
        if not local_devices:
            raise RuntimeError('JAX did not report any local devices.')
        if not self.multi_device:
            self.num_devices = 1
        elif self.requested_num_devices == 0:
            self.num_devices = len(local_devices)
        else:
            if self.requested_num_devices > len(local_devices):
                raise ValueError(
                    f'num_devices={self.requested_num_devices} was requested, '
                    f'but only {len(local_devices)} local JAX devices are available.'
                )
            self.num_devices = self.requested_num_devices
        if self.optimizer_name == 'kfac' and not self.multi_device:
            raise ValueError(
                "InsightQMC's KFAC path requires multi_device=True. It also "
                'works with a single visible JAX device through pmap.')
        if (
            self.optimizer_name == 'kfac'
            and self.requested_num_devices not in (0, len(local_devices))
        ):
            raise ValueError(
                'KFAC currently requires all visible local JAX devices; set '
                'num_devices=0 or restrict CUDA_VISIBLE_DEVICES before launch.')
        self.devices = local_devices[: self.num_devices]
        self.use_pmap = self.multi_device and (
            self.num_devices > 1 or self.optimizer_name == 'kfac')
        if self.use_pmap and self.batch_size % self.num_devices != 0:
            raise ValueError(
                f'batch_size={self.batch_size} must be divisible by '
                f'num_devices={self.num_devices} when multi_device is enabled.'
            )
        self.device_batch_size = self.batch_size // (self.num_devices if self.use_pmap else 1)

        self.requested_pretrain_num_devices = int(cfg.get('pretrain_num_devices', 0))
        if self.requested_pretrain_num_devices < 0:
            raise ValueError('pretrain_num_devices must be non-negative.')
        if self.requested_pretrain_num_devices == 0:
            self.pretrain_num_devices = self.num_devices
        else:
            if self.requested_pretrain_num_devices > len(local_devices):
                raise ValueError(
                    f'pretrain_num_devices={self.requested_pretrain_num_devices} was requested, '
                    f'but only {len(local_devices)} local JAX devices are available.'
                )
            self.pretrain_num_devices = self.requested_pretrain_num_devices
        self.pretrain_devices = local_devices[: self.pretrain_num_devices]
        self.pretrain_use_pmap = self.pretrain_num_devices > 1
        if self.pretrain_use_pmap and self.batch_size % self.pretrain_num_devices != 0:
            raise ValueError(
                f'batch_size={self.batch_size} must be divisible by '
                f'pretrain_num_devices={self.pretrain_num_devices} when pretraining uses pmap.'
            )
        self.reset_optimizer_on_resume = bool(cfg.get('reset_optimizer_on_resume', False))
        self.resize_resumed_noise = float(cfg.get('resize_resumed_noise', 0.0))
        self.preiterations = int(cfg.preiterations)
        self.run_pretrain = bool(cfg.run_pretrain)
        self.iterations = int(cfg.iterations)

        self.add_bias = bool(cfg.add_bias)
        self.external_weights = bool(cfg.external_weights)
        self.envelope_on = bool(cfg.envelope_on)
        self.envelope_type = str(cfg.get('envelope_type', 'isotropic')).lower()
        self.envelope_degree = int(cfg.get('envelope_degree', 5))
        if envelope.is_complex_angular_momentum(self.envelope_type) and not self.complex_output:
            raise ValueError('complex_angular_momentum envelope requires complex_output=True.')
        jastrow_cfg = cfg.get('jastrow', {})
        self.jastrow_ee = bool(jastrow_cfg.get('ee', True))
        self.jastrow_en = bool(jastrow_cfg.get('en', False))
        self.jastrow_en_order = int(jastrow_cfg.get('en_radial_order', 4))
        self.jastrow_type = str(jastrow_cfg.get('type', 'pade')).lower()
        self.jastrow_radial_order = int(jastrow_cfg.get('radial_order', 4))

        mkan_cfg = cfg.get('mkan', {})
        self.mkan_layer_type = str(mkan_cfg.get('layer_type', 'spline')).lower()
        self.mkan_mult_arity = mkan_cfg.get('mult_arity', 2)
        self.mkan_width = mkan_cfg.get('width', None)
        self.mkan_required_parameters = mkan_cfg.get('required_parameters', None)
        self.mkan_pretrain_phase_weight = float(mkan_cfg.get('pretrain_phase_weight', 1.0e-2))
        self.mkan_prune_mask_checkpoint = mkan_cfg.get('prune_mask_checkpoint', None)
        mkan_input_dim = mkan_cfg.get('input_dim', None)
        mkan_output_dim = mkan_cfg.get('output_dim', None)
        orbital_head_cfg = mkan_cfg.get('orbital_head', cfg.get('orbital_head', {}))
        self.orbital_head_enabled = bool(orbital_head_cfg.get('enabled', False))
        self.orbital_head_type = str(orbital_head_cfg.get('type', 'dense')).lower()
        self.orbital_head_bias = bool(orbital_head_cfg.get('bias', True))
        self.orbital_head_input_mode = str(
            orbital_head_cfg.get('input_mode', 'shared_rows')
        ).lower()
        if self.orbital_head_input_mode not in (
            'shared_rows',
            'per_electron',
            'all_electrons',
            'global',
            'flatten',
        ):
            raise ValueError(
                "orbital_head.input_mode must be 'shared_rows', 'per_electron', "
                "'all_electrons', 'global', or 'flatten'."
            )
        self.orbital_head_hidden_dims = tuple(
            int(dim) for dim in orbital_head_cfg.get('hidden_dims', ())
        )
        self.orbital_head_width = orbital_head_cfg.get('width', None)
        self.orbital_head_activation = str(orbital_head_cfg.get('activation', 'silu')).lower()
        self.orbital_head_rwf = orbital_head_cfg.get('rwf', None)
        self.orbital_head_layer_type = str(
            orbital_head_cfg.get('layer_type', self.mkan_layer_type)
        ).lower()
        self.orbital_head_required_parameters = orbital_head_cfg.get('required_parameters', None)
        self.orbital_head_mult_arity = orbital_head_cfg.get('mult_arity', self.mkan_mult_arity)
        self.mkan_input_dim = int(nfeatures if mkan_input_dim is None else mkan_input_dim)
        self.orbital_output_dim = (
            (2 * self.ndeterminants * self.nelectrons)
            if self.complex_output
            else (self.ndeterminants * self.nelectrons)
        )
        if self.orbital_head_enabled:
            if mkan_output_dim is None:
                if self.mkan_width is not None:
                    self.mkan_output_dim = _mkan_width_output_dim(self.mkan_width)
                else:
                    layer_dims = np.asarray(cfg.layer_dims).reshape(-1)
                    self.mkan_output_dim = int(
                        layer_dims[-1] if layer_dims.size else self.orbital_output_dim
                    )
            else:
                self.mkan_output_dim = int(mkan_output_dim)
            if self.mkan_output_dim <= 0:
                raise ValueError('mkan.output_dim must be positive when orbital_head is enabled.')
        else:
            self.mkan_output_dim = int(
                self.orbital_output_dim if mkan_output_dim is None else mkan_output_dim
            )
        if not self.orbital_head_enabled and self.mkan_output_dim < self.orbital_output_dim:
            raise ValueError(
                f'mkan.output_dim must be at least {self.orbital_output_dim} for '
                'orbital MKAN wavefunctions.'
            )
        self.orbital_head_uses_all_electrons = self.orbital_head_input_mode == 'all_electrons'
        self.orbital_head_uses_flattened_electrons = self.orbital_head_input_mode in (
            'global',
            'flatten',
        )
        self.orbital_head_input_dim = self.mkan_output_dim
        self.orbital_head_output_dim = self.orbital_output_dim
        if self.orbital_head_enabled and self.orbital_head_uses_all_electrons:
            self.orbital_head_input_dim = 4 * self.mkan_output_dim
        elif self.orbital_head_enabled and self.orbital_head_uses_flattened_electrons:
            self.orbital_head_input_dim = self.nelectrons * self.mkan_output_dim
            self.orbital_head_output_dim = self.nelectrons * self.orbital_output_dim
        self.adapt_frequency = int(cfg.get('mcmc_adapt_frequency', 20))
        self.pmove_min = float(cfg.get('mcmc_pmove_min', 0.50))
        self.pmove_max = float(cfg.get('mcmc_pmove_max', 0.60))
        self.width_scale = float(cfg.get('mcmc_width_scale', 1.05))

        grid_extension_cfg = cfg.get('grid_extension', {})
        self.grid_extension_enabled = bool(grid_extension_cfg.get('enabled', False))
        self.grid_extension_steps = tuple(int(v) for v in grid_extension_cfg.get('steps', ()))
        self.grid_extension_g_values = tuple(int(v) for v in grid_extension_cfg.get('g_values', ()))
        sample_size = grid_extension_cfg.get('sample_size', None)
        self.grid_extension_sample_size = None if sample_size is None else int(sample_size)
        if self.grid_extension_enabled:
            if self.mkan_layer_type not in ('base', 'spline'):
                raise ValueError(
                    'grid_extension currently supports only base/spline MKAN layers. '
                    f'Got layer_type={self.mkan_layer_type!r}.'
                )
            if len(self.grid_extension_steps) != len(self.grid_extension_g_values):
                raise ValueError('grid_extension.steps and grid_extension.g_values must have the same length.')
            if any(step <= 0 for step in self.grid_extension_steps):
                raise ValueError('grid_extension.steps must contain positive training step numbers.')
            if any(g <= 0 for g in self.grid_extension_g_values):
                raise ValueError('grid_extension.g_values must contain positive grid sizes.')

    def _build_checkpoint_state(
        self,
        *,
        stage: str,
        step: int,
        params,
        data: networks.KANetsData,
        key,
        pretrain_opt_state=None,
        train_opt_state=None,
    ):
        return {
            'stage': stage,
            'step': int(step),
            'params': params,
            'data': data,
            'key': key,
            'pretrain_opt_state': pretrain_opt_state,
            'train_opt_state': train_opt_state,
            'train_optimizer': self.optimizer_name,
            'mkan_static_state': self._mkan_static_state,
            'orbital_head_static_state': self._orbital_head_static_state,
        }

    def _build_networks(self):
        model_template = self._make_mkan_template()
        self._mkan_graphdef, initial_params, self._mkan_static_state = nnx.split(
            model_template, nnx.Param, ...
        )
        initial_head_params = None
        if self.orbital_head_enabled:
            head_template = self._make_orbital_head_template()
            (
                self._orbital_head_graphdef,
                initial_head_params,
                self._orbital_head_static_state,
            ) = nnx.split(head_template, nnx.Param, ...)
        if self.mkan_prune_mask_checkpoint:
            mask_checkpoint_path = Path(str(self.mkan_prune_mask_checkpoint)).expanduser()
            with mask_checkpoint_path.open('rb') as handle:
                mask_checkpoint = pickle.load(handle)
            mask_static_state = mask_checkpoint.get('mkan_static_state')
            if mask_static_state is None:
                raise ValueError(
                    f'No mkan_static_state found in prune mask checkpoint: {mask_checkpoint_path}'
                )
            self._mkan_static_state = _merge_static_state_with_template(
                self._mkan_static_state,
                mask_static_state,
            )
        active_spin_channels = networks.active_spin_channels(self.electrons)
        envelope_output_dims = [
            self.ndeterminants * (self.nelectrons if self.full_det else spin)
            for spin in active_spin_channels
        ]
        envelope_params = None
        if self.envelope_on:
            if self.envelope_type == 'isotropic':
                envelope_params = envelope.init_isotropic_envelope(
                    self.natoms, envelope_output_dims
                )
            elif self.envelope_type == 'chebyshev':
                envelope_params = envelope.init_chebyshev_envelope(
                    self.natoms,
                    envelope_output_dims,
                    degree=self.envelope_degree,
                )
            elif self.envelope_type == 'legendre':
                envelope_params = envelope.init_legendre_envelope(
                    self.natoms,
                    envelope_output_dims,
                    degree=self.envelope_degree,
                )
            elif envelope.is_legendre_anisotropic(self.envelope_type):
                envelope_params = envelope.init_legendre_anisotropic_envelope(
                    self.natoms,
                    envelope_output_dims,
                    degree=self.envelope_degree,
                    ndim=3,
                )
            elif envelope.is_angular_momentum(self.envelope_type):
                envelope_params = envelope.init_angular_momentum_envelope(
                    self.natoms,
                    envelope_output_dims,
                    degree=self.envelope_degree,
                )
            elif envelope.is_legendre_angular(self.envelope_type):
                envelope_params = envelope.init_legendre_angular_envelope(
                    self.natoms,
                    envelope_output_dims,
                    degree=self.envelope_degree,
                )
            elif envelope.is_complex_angular_momentum(self.envelope_type):
                envelope_params = envelope.init_complex_angular_momentum_envelope(
                    self.natoms,
                    envelope_output_dims,
                    degree=self.envelope_degree,
                )
            elif envelope.is_ferminet_angular(self.envelope_type):
                envelope_params = envelope.init_ferminet_angular_envelope(
                    self.natoms,
                    envelope_output_dims,
                    degree=self.envelope_degree,
                    ndim=3,
                )
            else:
                raise ValueError(f'Unsupported envelope_type={self.envelope_type!r}.')
        if self.jastrow_type == 'pade':
            same_spin_pairs, opposite_spin_pairs = jastrow.spin_pair_indices(self.electrons)
            init_jastrow = jastrow.init_pade_ee_jastrow
            apply_jastrow = jastrow.apply_pade_ee_jastrow
        elif self.jastrow_type == 'ferminet':
            same_spin_pairs, opposite_spin_pairs = jastrow.spin_pair_indices_or_empty(self.electrons)
            init_jastrow = jastrow.init_ferminet_ee_jastrow
            apply_jastrow = jastrow.apply_ferminet_ee_jastrow
        elif self.jastrow_type == 'ferminet_plus':
            same_spin_pairs, opposite_spin_pairs = jastrow.spin_pair_indices_or_empty(self.electrons)
            init_jastrow = lambda: jastrow.init_ferminet_plus_ee_jastrow(
                radial_order=self.jastrow_radial_order
            )
            apply_jastrow = jastrow.apply_ferminet_plus_ee_jastrow
        elif self.jastrow_type == 'ferminet_three_body':
            same_spin_pairs, opposite_spin_pairs = jastrow.spin_pair_indices_or_empty(self.electrons)
            init_jastrow = lambda: jastrow.init_ferminet_three_body_jastrow(
                radial_order=self.jastrow_radial_order
            )
            apply_jastrow = jastrow.apply_ferminet_three_body_jastrow
        else:
            raise ValueError(f'Unsupported jastrow.type={self.jastrow_type!r}.')
        jastrow_uses_r_ae = self.jastrow_type == 'ferminet_three_body'
        jastrow_params = init_jastrow() if self.jastrow_ee else None
        jastrow_en_params = (
            jastrow.init_one_body_en_jastrow(self.natoms, self.jastrow_en_order)
            if self.jastrow_en
            else None
        )

        def kan_init(key):
            del key
            params = {'mkan': initial_params}
            if self.orbital_head_enabled:
                params['orbital_head'] = initial_head_params
            if self.use_determinant_weights and self.ndeterminants > 1:
                params['det_weights'] = jnp.ones((self.ndeterminants, 1)) / self.ndeterminants
            if envelope_params is not None:
                params['envelope'] = envelope_params
            if jastrow_params is not None:
                params['jastrow_ee'] = jastrow_params
            if jastrow_en_params is not None:
                params['jastrow_en'] = jastrow_en_params
            return params

        def apply_mkan(params, features):
            model_params = params['mkan'] if isinstance(params, dict) and 'mkan' in params else params
            model = nnx.merge(self._mkan_graphdef, model_params, self._mkan_static_state)
            if features.shape[-1] != self.mkan_input_dim:
                raise ValueError(
                    f'MKAN input dimension mismatch: got {features.shape[-1]}, '
                    f'expected {self.mkan_input_dim}. Set cfg.mkan.input_dim '
                    'or cfg.mkan.width if you want a different feature size.'
            )
            return model(features)

        def all_electron_head_inputs(node_values):
            global_mean = jnp.mean(node_values, axis=0, keepdims=True)
            contexts = []
            spin_slices = []
            start = 0
            for spin in self.electrons:
                stop = start + int(spin)
                if stop > start:
                    spin_slices.append((start, stop))
                start = stop

            for index, (start, stop) in enumerate(spin_slices):
                local_nodes = node_values[start:stop]
                same_mean = jnp.mean(local_nodes, axis=0, keepdims=True)
                opposite_parts = [
                    node_values[other_start:other_stop]
                    for other_index, (other_start, other_stop) in enumerate(spin_slices)
                    if other_index != index
                ]
                if opposite_parts:
                    opposite_nodes = jnp.concatenate(opposite_parts, axis=0)
                    opposite_mean = jnp.mean(opposite_nodes, axis=0, keepdims=True)
                else:
                    opposite_mean = jnp.zeros_like(same_mean)
                row_count = stop - start
                contexts.append(
                    jnp.concatenate(
                        [
                            local_nodes,
                            jnp.broadcast_to(global_mean, (row_count, self.mkan_output_dim)),
                            jnp.broadcast_to(same_mean, (row_count, self.mkan_output_dim)),
                            jnp.broadcast_to(opposite_mean, (row_count, self.mkan_output_dim)),
                        ],
                        axis=-1,
                    )
                )
            return jnp.concatenate(contexts, axis=0)

        def apply_orbital_head(params, node_values):
            if not self.orbital_head_enabled:
                return node_values
            if not (isinstance(params, dict) and 'orbital_head' in params):
                raise ValueError('Missing orbital_head parameters.')
            head = nnx.merge(
                self._orbital_head_graphdef,
                params['orbital_head'],
                self._orbital_head_static_state,
            )
            if self.orbital_head_uses_all_electrons:
                return head(all_electron_head_inputs(node_values))
            if self.orbital_head_uses_flattened_electrons:
                flat_nodes = jnp.reshape(node_values, (1, self.orbital_head_input_dim))
                flat_values = head(flat_nodes)
                return jnp.reshape(flat_values, (self.nelectrons, self.orbital_output_dim))
            return head(node_values)

        def orbitals_apply(params, pos, spins, atoms, charges):
            del spins, charges
            ae, ee, r_ae, r_ee = _construct_input_features(pos, atoms, ndim=3)
            h_one = networks.orbital_features_from_components(
                ae,
                ee,
                r_ae,
                r_ee,
                feature_mode=self.orbital_feature_mode,
            )
            mkan_nodes = apply_mkan(params, h_one)
            orbital_values = apply_orbital_head(params, mkan_nodes)
            if self.complex_output:
                real_channel_count = 2 * self.ndeterminants * self.nelectrons
                orbital_values = (
                    orbital_values[..., 0:real_channel_count:2]
                    + 1.0j * orbital_values[..., 1:real_channel_count:2]
                )
            else:
                orbital_values = orbital_values[..., : self.ndeterminants * self.nelectrons]

            spin_partitions = _array_partitions(self.electrons)
            orbital_row_channels = jnp.split(orbital_values, spin_partitions, axis=0)
            active_spin_channels = [spin for spin in self.electrons if spin > 0]
            if self.full_det:
                orbital_channels = [
                    channel
                    for channel, spin in zip(orbital_row_channels, self.electrons)
                    if spin > 0
                ]
            else:
                starts = np.cumsum((0, *[self.ndeterminants * int(spin) for spin in self.electrons[:-1]]))
                orbital_channels = [
                    channel[:, start : start + self.ndeterminants * spin]
                    for channel, spin, start in zip(
                        orbital_row_channels,
                        self.electrons,
                        starts,
                    )
                    if spin > 0
                ]
            if self.envelope_on:
                if not (isinstance(params, dict) and 'envelope' in params):
                    raise ValueError('Missing envelope parameters for simple envelope.')
                r_ae_channels = jnp.split(r_ae, spin_partitions, axis=0)
                r_ae_channels = [
                    channel for channel, spin in zip(r_ae_channels, self.electrons) if spin > 0
                ]
                ae_channels = jnp.split(ae, spin_partitions, axis=0)
                ae_channels = [
                    channel for channel, spin in zip(ae_channels, self.electrons) if spin > 0
                ]
                theta, phi = envelope.angular_coordinates(ae)
                theta_channels = jnp.split(theta, spin_partitions, axis=0)
                theta_channels = [
                    channel for channel, spin in zip(theta_channels, self.electrons) if spin > 0
                ]
                phi_channels = jnp.split(phi, spin_partitions, axis=0)
                phi_channels = [
                    channel for channel, spin in zip(phi_channels, self.electrons) if spin > 0
                ]
                if self.envelope_type == 'isotropic':
                    apply_envelope = envelope.apply_isotropic_envelope
                elif self.envelope_type == 'chebyshev':
                    apply_envelope = envelope.apply_chebyshev_envelope
                elif self.envelope_type == 'legendre':
                    apply_envelope = envelope.apply_legendre_envelope
                elif envelope.is_legendre_anisotropic(self.envelope_type):
                    apply_envelope = envelope.apply_legendre_anisotropic_envelope
                elif envelope.is_angular_momentum(self.envelope_type):
                    apply_envelope = envelope.apply_angular_momentum_envelope
                elif envelope.is_legendre_angular(self.envelope_type):
                    apply_envelope = envelope.apply_legendre_angular_envelope
                elif envelope.is_complex_angular_momentum(self.envelope_type):
                    apply_envelope = envelope.apply_complex_angular_momentum_envelope
                elif envelope.is_ferminet_angular(self.envelope_type):
                    apply_envelope = envelope.apply_ferminet_angular_envelope
                else:
                    raise ValueError(f'Unsupported envelope_type={self.envelope_type!r}.')
                if envelope.is_legendre_anisotropic(self.envelope_type):
                    orbital_channels = [
                        channel * apply_envelope(ae=ae_channel, **envelope_param)
                        for channel, ae_channel, envelope_param in zip(
                            orbital_channels, ae_channels, params['envelope']
                        )
                    ]
                elif envelope.is_angular_momentum(self.envelope_type):
                    orbital_channels = [
                        channel
                        * apply_envelope(
                            r_ae=r_ae_channel,
                            theta=theta_channel,
                            phi=phi_channel,
                            **envelope_param,
                        )
                        for channel, r_ae_channel, theta_channel, phi_channel, envelope_param in zip(
                            orbital_channels,
                            r_ae_channels,
                            theta_channels,
                            phi_channels,
                            params['envelope'],
                        )
                    ]
                elif envelope.is_legendre_angular(self.envelope_type):
                    orbital_channels = [
                        channel
                        * apply_envelope(
                            r_ae=r_ae_channel,
                            theta=theta_channel,
                            phi=phi_channel,
                            **envelope_param,
                        )
                        for channel, r_ae_channel, theta_channel, phi_channel, envelope_param in zip(
                            orbital_channels,
                            r_ae_channels,
                            theta_channels,
                            phi_channels,
                            params['envelope'],
                        )
                    ]
                elif envelope.is_complex_angular_momentum(self.envelope_type):
                    orbital_channels = [
                        channel
                        * apply_envelope(
                            r_ae=r_ae_channel,
                            theta=theta_channel,
                            phi=phi_channel,
                            **envelope_param,
                        )
                        for channel, r_ae_channel, theta_channel, phi_channel, envelope_param in zip(
                            orbital_channels,
                            r_ae_channels,
                            theta_channels,
                            phi_channels,
                            params['envelope'],
                        )
                    ]
                elif envelope.is_ferminet_angular(self.envelope_type):
                    orbital_channels = [
                        channel
                        * apply_envelope(
                            ae=ae_channel,
                            theta=theta_channel,
                            phi=phi_channel,
                            **envelope_param,
                        )
                        for channel, ae_channel, theta_channel, phi_channel, envelope_param in zip(
                            orbital_channels,
                            ae_channels,
                            theta_channels,
                            phi_channels,
                            params['envelope'],
                        )
                    ]
                else:
                    orbital_channels = [
                        channel * apply_envelope(r_ae=r_ae_channel, **envelope_param)
                        for channel, r_ae_channel, envelope_param in zip(
                            orbital_channels, r_ae_channels, params['envelope']
                        )
                    ]
            shapes = [
                (spin, -1, self.nelectrons if self.full_det else spin)
                for spin in active_spin_channels
            ]
            orbital_channels = [
                jnp.reshape(channel, shape)
                for channel, shape in zip(orbital_channels, shapes)
            ]
            orbital_channels = [
                jnp.transpose(channel, (1, 0, 2))
                for channel in orbital_channels
            ]
            if self.full_det:
                return [jnp.concatenate(orbital_channels, axis=1)]
            return orbital_channels

        def signed_network(params, pos, spins, atoms, charges):
            determinant = orbitals_apply(params, pos, spins, atoms, charges)
            det_weights = None
            if self.use_determinant_weights and isinstance(params, dict):
                det_weights = params.get('det_weights', None)
            phase, logmag = networks.logdet_matmul(determinant, det_weights)
            if self.jastrow_ee or self.jastrow_en:
                _, _, r_ae, r_ee = _construct_input_features(pos, atoms, ndim=3)
                if self.jastrow_ee:
                    if not (isinstance(params, dict) and 'jastrow_ee' in params):
                        raise ValueError('Missing Jastrow parameters for electron-electron Jastrow.')
                    if jastrow_uses_r_ae:
                        logmag = logmag + apply_jastrow(
                            r_ee,
                            r_ae,
                            params['jastrow_ee'],
                            same_spin_pairs,
                            opposite_spin_pairs,
                        )
                    else:
                        logmag = logmag + apply_jastrow(
                            r_ee,
                            params['jastrow_ee'],
                            same_spin_pairs,
                            opposite_spin_pairs,
                        )
                if self.jastrow_en:
                    if not (isinstance(params, dict) and 'jastrow_en' in params):
                        raise ValueError('Missing Jastrow parameters for electron-nucleus Jastrow.')
                    logmag = logmag + jastrow.apply_one_body_en_jastrow(
                        r_ae,
                        params['jastrow_en'],
                    )
            return phase, logmag

        def logabs_network(params, pos, spins, atoms, charges):
            return signed_network(params, pos, spins, atoms, charges)[1]

        def log_network(params, pos, spins, atoms, charges):
            phase, mag = signed_network(params, pos, spins, atoms, charges)
            if self.complex_output:
                return mag + jnp.log(phase)
            return mag

        batch_network = jax.vmap(logabs_network, in_axes=(None, 0, None, None, None), out_axes=0)
        batch_log_network = jax.vmap(log_network, in_axes=(None, 0, None, None, None), out_axes=0)
        orbitals_vmap = jax.vmap(orbitals_apply, in_axes=(None, 0, None, None, None), out_axes=0)

        def extend_mkan_grid(params, data, g_new: int):
            if not (isinstance(params, dict) and 'mkan' in params):
                raise ValueError('Grid extension requires trainer params with an mkan entry.')
            samples = self._grid_extension_samples(data)
            model = nnx.merge(self._mkan_graphdef, params['mkan'], self._mkan_static_state)
            model.extend_grids(samples, int(g_new))
            self._mkan_graphdef, mkan_params, self._mkan_static_state = nnx.split(
                model, nnx.Param, ...
            )
            new_params = dict(params)
            new_params['mkan'] = mkan_params
            return new_params

        return (
            kan_init,
            signed_network,
            logabs_network,
            log_network,
            batch_network,
            batch_log_network,
            orbitals_vmap,
            extend_mkan_grid,
        )

    def _grid_extension_samples(self, data: networks.KANetsData) -> jnp.ndarray:
        positions = jnp.reshape(data.positions, (-1, self.nelectrons * 3))

        def single_position_features(pos):
            return networks.construct_orbital_features(
                pos,
                data.atoms,
                ndim=3,
                feature_mode=self.orbital_feature_mode,
            )

        samples = jax.vmap(single_position_features)(positions)
        samples = jnp.reshape(samples, (-1, self.mkan_input_dim))
        if self.grid_extension_sample_size is not None:
            samples = samples[:self.grid_extension_sample_size]
        return samples

    def _make_mkan_template(self):
        if self.mkan_width is None:
            hidden_dims = [int(v) for v in np.asarray(self.layer_dims).reshape(-1)[1:-1]]
            width = [self.mkan_input_dim, *hidden_dims, self.mkan_output_dim]
        else:
            width = list(self.mkan_width)
            width[0] = self.mkan_input_dim
            width[-1] = self.mkan_output_dim

        required_parameters = self._mkan_required_parameters()
        return MultKAN(
            width=width,
            layer_type=self.mkan_layer_type,
            required_parameters=required_parameters,
            mult_arity=self.mkan_mult_arity,
            seed=self.seed,
        )

    def _make_orbital_head_template(self):
        if self.orbital_head_type in ('dense', 'mlp'):
            return networks.DenseOrbitalHead(
                input_dim=self.orbital_head_input_dim,
                output_dim=self.orbital_head_output_dim,
                hidden_dims=self.orbital_head_hidden_dims,
                activation=self.orbital_head_activation,
                add_bias=self.orbital_head_bias,
                rwf=self.orbital_head_rwf,
                seed=self.seed + 7919,
            )
        if self.orbital_head_type == 'kan':
            return networks.KANOrbitalHead(
                input_dim=self.orbital_head_input_dim,
                output_dim=self.orbital_head_output_dim,
                hidden_dims=self.orbital_head_hidden_dims,
                layer_type=self.orbital_head_layer_type,
                required_parameters=self._orbital_head_required_parameters(),
                seed=self.seed + 7919,
            )
        if self.orbital_head_type in ('mkan', 'multkan'):
            return networks.MKANOrbitalHead(
                input_dim=self.orbital_head_input_dim,
                output_dim=self.orbital_head_output_dim,
                hidden_dims=self.orbital_head_hidden_dims,
                width=self.orbital_head_width,
                layer_type=self.orbital_head_layer_type,
                required_parameters=self._orbital_head_required_parameters(),
                mult_arity=self.orbital_head_mult_arity,
                seed=self.seed + 7919,
            )
        raise ValueError(
            f"Unsupported orbital_head.type={self.orbital_head_type!r}. "
            "Expected 'mlp', 'dense', 'kan', or 'mkan'."
        )

    def _kan_required_parameters_for_layer(
        self,
        layer_type: str,
        required_parameters,
        *,
        add_bias: Optional[bool] = None,
        external_weights: Optional[bool] = None,
    ):
        if required_parameters is not None:
            return dict(required_parameters)

        use_bias = self.add_bias if add_bias is None else bool(add_bias)
        use_external_weights = (
            self.external_weights if external_weights is None else bool(external_weights)
        )

        if layer_type in ('chebyshev', 'legendre'):
            return {
                'D': _first_int(self.k, 3),
                'flavor': 'exact' if layer_type == 'chebyshev' else None,
                'external_weights': use_external_weights,
                'add_bias': use_bias,
            }
        if layer_type in ('base', 'spline'):
            return {
                'k': _first_int(self.k, 3),
                'G': _first_int(self.g, 5),
                'grid_range': _first_grid_range(self.grid_range),
                'external_weights': use_external_weights,
                'add_bias': use_bias,
            }
        if layer_type == 'rbf':
            return {
                'D': _first_int(self.k, 5),
                'grid_range': _first_grid_range(self.grid_range, default=(-2.0, 2.0)),
                'external_weights': use_external_weights,
                'add_bias': use_bias,
            }
        if layer_type == 'fastkan':
            return {
                'D': _first_int(self.g, 8),
                'grid_range': _first_grid_range(self.grid_range, default=(-2.0, 2.0)),
                'add_bias': use_bias,
            }
        if layer_type == 'relukan':
            return {
                'G': _first_int(self.g, 5),
                'k': _first_int(self.k, 3),
                'add_bias': use_bias,
            }
        if layer_type == 'wavkan':
            return {'wavelet_type': 'mexican_hat', 'add_bias': use_bias}
        if layer_type == 'sine':
            return {
                'D': _first_int(self.k, 5),
                'external_weights': use_external_weights,
                'add_bias': use_bias,
            }
        if layer_type == 'fourier':
            return {
                'D': _first_int(self.k, 5),
                'add_bias': use_bias,
            }
        raise ValueError(f'Unsupported KAN layer_type: {layer_type}')

    def _orbital_head_required_parameters(self):
        return self._kan_required_parameters_for_layer(
            self.orbital_head_layer_type,
            self.orbital_head_required_parameters,
            add_bias=self.orbital_head_bias,
        )

    def _mkan_required_parameters(self):
        return self._kan_required_parameters_for_layer(
            self.mkan_layer_type,
            self.mkan_required_parameters,
        )

    def _initialize_params_and_data(self, kan_init):
        resume_state = self.run_manager.load_last_checkpoint()

        key = jax.random.PRNGKey(self.seed)
        key, subkey = jax.random.split(key)
        params = kan_init(subkey)
        target_params = params
        sharded_key = key

        pretrain_start_step = 0
        train_start_step = self.t_init
        pretrain_opt_state = None
        train_opt_state = None
        data = None
        params_changed_on_resume = False

        if resume_state is not None:
            params = resume_state['params']
            params, params_changed_on_resume = self._prepare_resumed_params(
                params,
                target_params=target_params,
            )
            data = resume_state['data']
            data = self._resize_resumed_data(data)
            sharded_key = resume_state['key']
            if resume_state.get('mkan_static_state') is not None:
                self._mkan_static_state = _merge_static_state_with_template(
                    self._mkan_static_state,
                    resume_state['mkan_static_state'],
                )
            if resume_state.get('orbital_head_static_state') is not None:
                self._orbital_head_static_state = _merge_static_state_with_template(
                    self._orbital_head_static_state,
                    resume_state['orbital_head_static_state'],
                )
            stage = resume_state.get('stage')
            if stage == 'pretrain':
                pretrain_start_step = int(resume_state.get('step', 0))
                pretrain_opt_state = resume_state.get('pretrain_opt_state')
                if self.reset_optimizer_on_resume or params_changed_on_resume:
                    pretrain_opt_state = None
            elif stage == 'train':
                train_start_step = int(resume_state.get('step', self.t_init))
                train_opt_state = resume_state.get('train_opt_state')
                checkpoint_optimizer = str(
                    resume_state.get('train_optimizer', 'adam')).lower()
                if (
                    self.reset_optimizer_on_resume
                    or params_changed_on_resume
                    or checkpoint_optimizer != self.optimizer_name
                ):
                    train_opt_state = None

        if data is None:
            key_electrons_coords = jax.random.PRNGKey(self.seed_electrons_coords)
            key_electrons_coords, subkey_electrons_coords = jax.random.split(key_electrons_coords)
            pos, _ = electrons_initialization.init_electrons(
                subkey_electrons_coords,
                self.molecule,
                self.electrons,
                batch_size=self.batch_size,
                init_width=self.init_width,
                core_electrons=self.core_electrons,
            )
            data = networks.KANetsData(positions=pos, spins=self.spins, atoms=self.atoms, charges=self.charges)

        sharded_key = multi_device.canonical_key(sharded_key)
        return params, data, sharded_key, pretrain_start_step, train_start_step, pretrain_opt_state, train_opt_state

    def _prepare_resumed_params(self, params, target_params=None):
        if not isinstance(params, dict):
            return params, False

        changed = False
        new_params = dict(params)
        if target_params is not None and isinstance(target_params, dict):
            new_params, did_prepare = self._prepare_resumed_mkan(new_params, target_params)
            changed = changed or did_prepare
            if self.orbital_head_enabled:
                new_params, did_prepare = self._prepare_resumed_orbital_head(
                    new_params,
                    target_params,
                )
                changed = changed or did_prepare

        if self.use_determinant_weights and self.ndeterminants > 1:
            new_params, did_prepare = self._prepare_resumed_det_weights(new_params)
            changed = changed or did_prepare

        if self.envelope_on:
            new_params, did_prepare = self._prepare_resumed_envelope(new_params)
            changed = changed or did_prepare

        if self.jastrow_type in ('ferminet_plus', 'ferminet_three_body'):
            if 'jastrow_ee' in new_params:
                jastrow_params = dict(new_params['jastrow_ee'])
                jastrow_changed = False
                for name in ('ee_par_coeff', 'ee_anti_coeff'):
                    if name not in jastrow_params:
                        continue
                    resized, did_resize = self._resize_1d_param(
                        jastrow_params[name],
                        self.jastrow_radial_order,
                    )
                    jastrow_params[name] = resized
                    jastrow_changed = jastrow_changed or did_resize

                if jastrow_changed:
                    new_params['jastrow_ee'] = jastrow_params
                    changed = True

        if self.jastrow_en:
            target = jastrow.init_one_body_en_jastrow(self.natoms, self.jastrow_en_order)
            if 'jastrow_en' not in new_params:
                new_params['jastrow_en'] = target
                changed = True
            else:
                en_params = dict(new_params['jastrow_en'])
                if 'en_coeff' not in en_params:
                    en_params['en_coeff'] = target['en_coeff']
                    changed = True
                else:
                    resized, did_resize = self._resize_2d_param(
                        en_params['en_coeff'],
                        target['en_coeff'].shape,
                    )
                    en_params['en_coeff'] = resized
                    changed = changed or did_resize
                new_params['jastrow_en'] = en_params
        return new_params, changed

    def _prepare_resumed_mkan(self, params, target_params):
        if 'mkan' not in params or 'mkan' not in target_params:
            return params, False
        try:
            source_output_dim = self._mkan_state_output_dim(params['mkan'])
            target_output_dim = self._mkan_state_output_dim(target_params['mkan'])
        except (KeyError, TypeError, ValueError):
            return params, False
        if source_output_dim == target_output_dim:
            return params, False

        new_params = dict(params)
        target_mkan = target_params['mkan']
        if self.orbital_head_enabled:
            self._copy_state_overlap(params['mkan'], target_mkan)
            new_params['mkan'] = target_mkan
            return new_params, True

        self._copy_mkan_state_overlap(
            params['mkan'],
            target_mkan,
            source_output_dim,
            target_output_dim,
        )
        new_params['mkan'] = target_mkan
        return new_params, True

    def _prepare_resumed_orbital_head(self, params, target_params):
        if 'orbital_head' not in target_params:
            return params, False
        if 'orbital_head' not in params:
            new_params = dict(params)
            new_params['orbital_head'] = target_params['orbital_head']
            return new_params, True

        if self._state_shape_signature(params['orbital_head']) == self._state_shape_signature(
            target_params['orbital_head']
        ):
            return params, False

        target_head = target_params['orbital_head']
        self._copy_state_overlap(params['orbital_head'], target_head)
        new_params = dict(params)
        new_params['orbital_head'] = target_head
        return new_params, True

    @classmethod
    def _state_shape_signature(cls, state):
        try:
            flat = dict(state.flat_state())
        except AttributeError:
            if hasattr(state, 'keys'):
                items = []
                for key in sorted(state.keys()):
                    items.extend(
                        (f'{key}/{path}', shape)
                        for path, shape in cls._state_shape_signature(state[key])
                    )
                return tuple(items)
            value = getattr(state, 'value', state)
            shape = tuple(value.shape) if hasattr(value, 'shape') else ()
            return (('', shape),)

        signature = []
        for path, value in sorted(flat.items(), key=lambda item: str(item[0])):
            param_value = getattr(value, 'value', value)
            shape = tuple(param_value.shape) if hasattr(param_value, 'shape') else ()
            signature.append((str(path), shape))
        return tuple(signature)

    @staticmethod
    def _mkan_state_output_dim(state):
        layers = state['layers']
        final_key = sorted(layers.keys())[-1]
        return int(layers[final_key]['bias'].value.shape[0])

    def _copy_mkan_state_overlap(self, source, target, source_output_dim, target_output_dim):
        layers = target['layers']
        final_key = sorted(layers.keys())[-1]
        for key in source['layers'].keys():
            if key in target['layers'] and key != final_key:
                self._copy_state_overlap(source['layers'][key], target['layers'][key])

        output_map = self._output_real_channel_map(source_output_dim, target_output_dim)
        self._remap_final_mkan_layer(
            source['layers'][final_key],
            target['layers'][final_key],
            output_map,
            source_output_dim,
            target_output_dim,
        )
        for name in ('node_bias', 'node_scale', 'subnode_bias', 'subnode_scale'):
            if name not in source or name not in target:
                continue
            for key in source[name].keys():
                if key not in target[name]:
                    continue
                if key == final_key:
                    self._remap_output_vector(source[name][key], target[name][key], output_map)
                else:
                    self._copy_state_overlap(source[name][key], target[name][key])

    def _output_real_channel_map(self, source_output_dim: int, target_output_dim: int):
        if not self.complex_output:
            source_ndet = source_output_dim // self.nelectrons
            target_ndet = target_output_dim // self.nelectrons
            factor = 1
        else:
            source_ndet = source_output_dim // (2 * self.nelectrons)
            target_ndet = target_output_dim // (2 * self.nelectrons)
            factor = 2
        ndet_common = min(source_ndet, target_ndet)
        pairs = []
        source_spin_offset = source_ndet * self.electrons[0]
        target_spin_offset = target_ndet * self.electrons[0]
        for det in range(ndet_common):
            for orbital in range(self.electrons[0]):
                source_channel = det * self.electrons[0] + orbital
                target_channel = det * self.electrons[0] + orbital
                for part in range(factor):
                    pairs.append((factor * source_channel + part, factor * target_channel + part))
            for orbital in range(self.electrons[1]):
                source_channel = source_spin_offset + det * self.electrons[1] + orbital
                target_channel = target_spin_offset + det * self.electrons[1] + orbital
                for part in range(factor):
                    pairs.append((factor * source_channel + part, factor * target_channel + part))
        return pairs

    def _remap_final_mkan_layer(
        self,
        source_layer,
        target_layer,
        output_map,
        source_output_dim,
        target_output_dim,
    ):
        for name in ('bias', 'c_res', 'c_spl'):
            if name in source_layer and name in target_layer:
                self._remap_output_vector(source_layer[name], target_layer[name], output_map)
        if 'c_basis' not in source_layer or 'c_basis' not in target_layer:
            return
        source_value = source_layer['c_basis'].value
        target_value = target_layer['c_basis'].value
        source_basis = source_value.reshape(source_output_dim, -1, source_value.shape[-1])
        target_basis = target_value.reshape(target_output_dim, -1, target_value.shape[-1])
        hidden = min(int(source_basis.shape[1]), int(target_basis.shape[1]))
        basis = min(int(source_basis.shape[2]), int(target_basis.shape[2]))
        for source_index, target_index in output_map:
            target_basis = target_basis.at[target_index, :hidden, :basis].set(
                source_basis[source_index, :hidden, :basis]
            )
        target_layer['c_basis'].value = target_basis.reshape(target_value.shape)

    def _remap_output_vector(self, source_param, target_param, output_map):
        source_value = source_param.value
        target_value = target_param.value
        if source_value.ndim == 0 or target_value.ndim == 0:
            return
        updated = jnp.asarray(target_value)
        trailing = tuple(
            slice(0, min(int(source_value.shape[axis]), int(target_value.shape[axis])))
            for axis in range(1, source_value.ndim)
        )
        for source_index, target_index in output_map:
            updated = updated.at[(target_index, *trailing)].set(
                source_value[(source_index, *trailing)]
            )
        target_param.value = updated

    def _prepare_resumed_det_weights(self, params):
        target_shape = (self.ndeterminants, 1)
        if 'det_weights' not in params:
            new_params = dict(params)
            new_params['det_weights'] = jnp.ones(target_shape) / self.ndeterminants
            return new_params, True

        value = jnp.asarray(params['det_weights'])
        if tuple(value.shape) == target_shape:
            return params, False
        resized = jnp.zeros(target_shape, dtype=value.dtype)
        rows = min(int(value.shape[0]), target_shape[0])
        cols = min(int(value.shape[1]), target_shape[1])
        resized = resized.at[:rows, :cols].set(value[:rows, :cols])
        new_params = dict(params)
        new_params['det_weights'] = resized
        return new_params, True

    def _target_envelope_params(self):
        active_spin_channels = networks.active_spin_channels(self.electrons)
        output_dims = [
            self.ndeterminants * (self.nelectrons if self.full_det else spin)
            for spin in active_spin_channels
        ]
        if self.envelope_type == 'isotropic':
            return envelope.init_isotropic_envelope(self.natoms, output_dims)
        if self.envelope_type == 'chebyshev':
            return envelope.init_chebyshev_envelope(
                self.natoms,
                output_dims,
                degree=self.envelope_degree,
            )
        if self.envelope_type == 'legendre':
            return envelope.init_legendre_envelope(
                self.natoms,
                output_dims,
                degree=self.envelope_degree,
            )
        if envelope.is_legendre_anisotropic(self.envelope_type):
            return envelope.init_legendre_anisotropic_envelope(
                self.natoms,
                output_dims,
                degree=self.envelope_degree,
                ndim=3,
            )
        if envelope.is_angular_momentum(self.envelope_type):
            return envelope.init_angular_momentum_envelope(
                self.natoms,
                output_dims,
                degree=self.envelope_degree,
            )
        if envelope.is_legendre_angular(self.envelope_type):
            return envelope.init_legendre_angular_envelope(
                self.natoms,
                output_dims,
                degree=self.envelope_degree,
            )
        if envelope.is_complex_angular_momentum(self.envelope_type):
            return envelope.init_complex_angular_momentum_envelope(
                self.natoms,
                output_dims,
                degree=self.envelope_degree,
            )
        if envelope.is_ferminet_angular(self.envelope_type):
            return envelope.init_ferminet_angular_envelope(
                self.natoms,
                output_dims,
                degree=self.envelope_degree,
                ndim=3,
            )
        return None

    def _prepare_resumed_envelope(self, params):
        target = self._target_envelope_params()
        if target is None:
            return params, False
        if 'envelope' not in params:
            new_params = dict(params)
            new_params['envelope'] = target
            return new_params, True

        changed = False
        current = list(params['envelope'])
        resized_envelope = []
        for index, target_channel in enumerate(target):
            if index >= len(current):
                resized_envelope.append(target_channel)
                changed = True
                continue
            current_channel = dict(current[index])
            resized_channel = {}
            for name, target_value in target_channel.items():
                if name not in current_channel:
                    resized_channel[name] = target_value
                    changed = True
                    continue
                resized, did_resize = self._resize_param_like(current_channel[name], target_value)
                resized_channel[name] = resized
                changed = changed or did_resize
            resized_envelope.append(resized_channel)

        if len(current) != len(target):
            changed = True
        if not changed:
            return params, False
        new_params = dict(params)
        new_params['envelope'] = resized_envelope
        return new_params, True

    @staticmethod
    def _resize_1d_param(value, target_size: int):
        value = jnp.asarray(value)
        current_size = int(value.shape[0])
        target_size = int(target_size)
        if current_size == target_size:
            return value, False
        if current_size > target_size:
            return value[:target_size], True
        padding = jnp.zeros((target_size - current_size,), dtype=value.dtype)
        return jnp.concatenate((value, padding), axis=0), True

    @staticmethod
    def _resize_2d_param(value, target_shape):
        value = jnp.asarray(value)
        target_shape = tuple(int(v) for v in target_shape)
        if tuple(value.shape) == target_shape:
            return value, False
        resized = jnp.zeros(target_shape, dtype=value.dtype)
        rows = min(int(value.shape[0]), target_shape[0])
        cols = min(int(value.shape[1]), target_shape[1])
        resized = resized.at[:rows, :cols].set(value[:rows, :cols])
        return resized, True

    @staticmethod
    def _resize_param_like(value, template):
        value = jnp.asarray(value)
        template = jnp.asarray(template)
        if tuple(value.shape) == tuple(template.shape):
            return value, False
        if value.ndim != template.ndim:
            return template, True
        resized = jnp.asarray(template, dtype=value.dtype)
        common = tuple(slice(0, min(int(a), int(b))) for a, b in zip(value.shape, template.shape))
        resized = resized.at[common].set(value[common])
        return resized, True

    def _copy_state_overlap(self, source, target):
        if hasattr(source, 'value') and hasattr(target, 'value'):
            source_value = source.value
            target_value = target.value
            if hasattr(source_value, 'shape') and hasattr(target_value, 'shape'):
                resized, _ = self._resize_param_like(source_value, target_value)
                target.value = resized
            return
        if not (hasattr(source, 'keys') and hasattr(target, 'keys')):
            return
        for key in source.keys():
            if key in target:
                self._copy_state_overlap(source[key], target[key])

    def _resize_resumed_data(self, data: networks.KANetsData) -> networks.KANetsData:
        positions = data.positions
        current_batch = int(positions.shape[0])
        if current_batch == self.batch_size:
            return data
        if current_batch > self.batch_size:
            positions = positions[: self.batch_size]
        else:
            repeats = int(np.ceil(self.batch_size / current_batch))
            positions = jnp.tile(positions, (repeats, 1))[: self.batch_size]
            if self.resize_resumed_noise > 0.0:
                key = jax.random.PRNGKey(
                    self.seed_electrons_coords + 9973 * self.batch_size + current_batch
                )
                noise = (
                    jax.random.normal(key, positions.shape, dtype=positions.dtype)
                    * self.resize_resumed_noise
                )
                keep_original = jnp.arange(self.batch_size) < current_batch
                noise = noise * (~keep_original)[:, None]
                positions = positions + noise
        return networks.KANetsData(
            positions=positions,
            spins=data.spins,
            atoms=data.atoms,
            charges=data.charges,
        )

    def _prepare_train_state_for_devices(self, state: train_state.TrainState) -> train_state.TrainState:
        if not self.use_pmap:
            return state
        return state.replace(
            params=multi_device.replicate(state.params, self.devices),
            opt_state=multi_device.replicate(state.opt_state, self.devices),
        )

    def _prepare_data_for_devices(self, data: networks.KANetsData) -> networks.KANetsData:
        if not self.use_pmap:
            return data
        return multi_device.shard_data(data, self.num_devices)

    def _host_params(self, params):
        if not self.use_pmap:
            return params
        return multi_device.unreplicate(params)

    def _host_opt_state(self, opt_state):
        if not self.use_pmap:
            return opt_state
        return multi_device.unreplicate(opt_state)

    def _host_data(self, data: networks.KANetsData) -> networks.KANetsData:
        if not self.use_pmap:
            return data
        return multi_device.unshard_data(data)

    def _build_optimizer(self):
        def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
            return self.learning_rate / (1.0 + t_ / self.learning_rate_decay)

        transforms = []
        if self.gradient_clip_norm > 0.0:
            transforms.append(optax.clip_by_global_norm(self.gradient_clip_norm))
        transforms.append(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-6))
        if self.optimizer_name == 'adamw':
            transforms.append(
                optax.add_decayed_weights(self.adamw_weight_decay))
        transforms.extend((
            optax.scale_by_schedule(learning_rate_schedule),
            optax.scale(-1.0),
        ))
        return optax.chain(*transforms)

    def _build_train_step(self, signed_network: Callable, logabs_network: Callable, log_network: Callable):
        loss_network = log_network if self.complex_output else logabs_network
        local_energy = hamiltonian.local_energy(
            f=signed_network,
            nspins=self.electrons,
            charges=self.charges,
            use_scan=self.use_scan,
            complex_output=self.complex_output,
            laplacian_method=self.laplacian_method,
        )
        evaluate_loss = qmc_loss_functions.make_loss(
            loss_network,
            local_energy,
            clip_local_energy=self.clip_local_energy,
            clip_from_median=True,
            center_at_clipped_energy=True,
            complex_output=self.complex_output,
        )

        batch_signed_network = jax.vmap(
            signed_network, in_axes=(None, 0, None, None, None), out_axes=(0, 0)
        )
        monte_carlo = vmcmc.make_vmcmc_step(
            f=batch_signed_network,
            ndim=3,
            nelectrons=self.nelectrons,
            steps=self.mcmc_steps,
            jit=not self.use_pmap,
        )
        if self.optimizer_name in ('adam', 'adamw'):
            optimizer = self._build_optimizer()
            optimizer_step = make_opt_update_step(evaluate_loss, optimizer)
            step_fn = make_training_step(
                mcmc_step=monte_carlo,
                optimizer_step=optimizer_step,
                reset_if_nan=True,
                jit=not self.use_pmap,
            )
        elif self.optimizer_name == 'rgn':
            optimizer = optax.identity()
            optimizer_step = make_rgn_update_step(
                evaluate_loss,
                loss_network,
                local_energy,
                epsilon=self.rgn_epsilon,
                eta=self.rgn_eta,
                cg_maxiter=self.rgn_cg_maxiter,
                cg_tol=self.rgn_cg_tol,
                step_scale=self.rgn_step_scale,
                max_update_norm=self.rgn_max_update_norm,
                reset_if_nan=self.rgn_split_compilation,
            )
            if self.rgn_split_compilation:
                if self.use_pmap:
                    compiled_monte_carlo = constants.pmap(
                        monte_carlo,
                        in_axes=(0, multi_device.DATA_IN_AXES, 0, None),
                        out_axes=(multi_device.DATA_IN_AXES, 0),
                        devices=self.devices,
                    )
                    compiled_optimizer_step = constants.pmap(
                        optimizer_step,
                        in_axes=(0, multi_device.DATA_IN_AXES, 0, 0),
                        out_axes=(0, 0, 0, 0),
                        devices=self.devices,
                    )
                else:
                    # make_vmcmc_step already jits the single-device sampler.
                    compiled_monte_carlo = monte_carlo
                    compiled_optimizer_step = jax.jit(optimizer_step)
                step_fn = make_split_training_step(
                    compiled_monte_carlo, compiled_optimizer_step)
            else:
                step_fn = make_training_step(
                    mcmc_step=monte_carlo,
                    optimizer_step=optimizer_step,
                    reset_if_nan=True,
                    jit=not self.use_pmap,
                )
        else:
            if constants.kfac_jax is None:
                raise ImportError(
                    'KFAC was selected but kfac_jax could not be imported. '
                    'Install a version compatible with the active JAX version.')

            def evaluate_kfac_loss(params, key, positions):
                """KFAC loss with only walker positions on its pmap batch axis."""
                data = networks.KANetsData(
                    positions=positions,
                    spins=self.spins,
                    atoms=self.atoms,
                    charges=self.charges,
                )
                return evaluate_loss(params, key, data)

            def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
                return self.learning_rate / (1.0 + t_ / self.learning_rate_decay)

            val_and_grad = jax.value_and_grad(
                evaluate_kfac_loss, argnums=0, has_aux=True)
            optimizer = constants.kfac_jax.Optimizer(
                val_and_grad,
                l2_reg=self.kfac_l2_reg,
                norm_constraint=self.kfac_norm_constraint,
                value_func_has_aux=True,
                value_func_has_rng=True,
                learning_rate_schedule=learning_rate_schedule,
                curvature_ema=self.kfac_cov_ema_decay,
                inverse_update_period=self.kfac_invert_every,
                min_damping=self.kfac_min_damping,
                num_burnin_steps=0,
                register_only_generic=self.kfac_register_only_generic,
                estimation_mode='fisher_exact',
                multi_device=True,
                pmap_axis_name=constants.PMAP_AXIS_NAME,
            )
            pmapped_monte_carlo = constants.pmap(
                monte_carlo,
                in_axes=(0, multi_device.DATA_IN_AXES, 0, None),
                out_axes=(multi_device.DATA_IN_AXES, 0),
                devices=self.devices,
            )
            step_fn = make_kfac_training_step(
                pmapped_monte_carlo,
                optimizer,
                damping=self.kfac_damping,
            )
        if (
            self.use_pmap
            and self.optimizer_name != 'kfac'
            and not (
                self.optimizer_name == 'rgn' and self.rgn_split_compilation)
        ):
            step_fn = constants.pmap(
                step_fn,
                in_axes=(multi_device.DATA_IN_AXES, 0, 0, 0, None),
                out_axes=(multi_device.DATA_IN_AXES, 0, 0, 0, 0, 0),
                devices=self.devices,
            )
        return optimizer, step_fn

    def _build_train_state(self, params, optimizer, train_opt_state):
        state_optimizer = optax.identity() if self.optimizer_name == 'kfac' else optimizer
        state = train_state.TrainState.create(
            apply_fn=lambda *_args, **_kwargs: None,
            params=params,
            tx=state_optimizer,
        )
        if train_opt_state is not None:
            state = state.replace(opt_state=train_opt_state)
        return state

    def _initialize_kfac_state(self, state, optimizer, data, key):
        # kfac_jax requires the initialization RNG to be identical on every
        # device (training-step RNGs, in contrast, must be different).
        _, init_key = jax.random.split(multi_device.canonical_key(key))
        init_keys = multi_device.replicate(init_key, self.devices)
        # kfac_jax pmaps every batch leaf.  Only positions carry a leading
        # device axis; molecular metadata is closed over by evaluate_kfac_loss.
        opt_state = optimizer.init(state.params, init_keys, data.positions)
        return state.replace(opt_state=opt_state)

    def _run_train_loop(
        self,
        *,
        train_start_step: int,
        runtime: RuntimeState,
        state: train_state.TrainState,
        step_fn,
        signed_network: Callable,
        logabs_network: Callable,
        log_network: Callable,
        extend_mkan_grid: Callable,
    ):
        initial_state = self._build_checkpoint_state(
            stage='train',
            step=train_start_step,
            params=self._host_params(state.params),
            data=self._host_data(runtime.data),
            key=runtime.key,
            train_opt_state=self._host_opt_state(state.opt_state),
        )
        self.run_manager.checkpoints.save_last(initial_state)

        if self.debug:
            jax.debug.print('sharded_key:{}', runtime.key)

        iterator: Any = trange(train_start_step, self.iterations, desc='Training', dynamic_ncols=True)
        grid_extension_targets = (
            dict(zip(self.grid_extension_steps, self.grid_extension_g_values))
            if self.grid_extension_enabled
            else {}
        )
        for t in iterator:
            key, subkeys = multi_device.split_step_key(
                runtime.key,
                self.num_devices,
                self.use_pmap,
            )
            data, params, opt_state, loss, aux_data, pmove = step_fn(
                runtime.data,
                state.params,
                state.opt_state,
                subkeys,
                runtime.mcmc_width,
            )
            state = state.replace(step=state.step + 1, params=params, opt_state=opt_state)

            pmove_mean = jnp.mean(pmove)
            t_since_update = t % self.adapt_frequency
            runtime.pmoves[t_since_update] = _to_scalar(pmove_mean)
            valid_pmoves = runtime.pmoves[~np.isnan(runtime.pmoves)]
            if (
                t > 0
                and t_since_update == 0
                and valid_pmoves.size == self.adapt_frequency
            ):
                mean_pmove = float(np.mean(valid_pmoves))
                if mean_pmove > self.pmove_max:
                    runtime = runtime.replace(mcmc_width=runtime.mcmc_width * self.width_scale)
                elif mean_pmove < self.pmove_min:
                    runtime = runtime.replace(mcmc_width=runtime.mcmc_width / self.width_scale)

            pmove_window_mean = float(np.mean(valid_pmoves)) if valid_pmoves.size else _to_scalar(pmove_mean)
            step_id = t + 1
            loss_value = float(jnp.mean(jnp.real(loss)))
            variance_value = float(jnp.mean(jnp.real(aux_data.variance)))
            iterator.set_postfix(iter=step_id, loss=f'{loss_value:.6f}')

            if self.run_manager.should_log(step_id, self.iterations):
                self.run_manager.log_scalars(
                    'train',
                    step_id,
                    {
                        'loss': loss_value,
                        'variance': variance_value,
                        'pmove': _to_scalar(pmove_mean),
                        'pmove_window': pmove_window_mean,
                        'mcmc_width': _to_scalar(runtime.mcmc_width),
                    },
                )

            if step_id in grid_extension_targets:
                g_new = grid_extension_targets[step_id]
                params = extend_mkan_grid(self._host_params(state.params), self._host_data(data), g_new)
                optimizer, step_fn = self._build_train_step(signed_network, logabs_network, log_network)
                step = state.step
                state = self._build_train_state(params, optimizer, train_opt_state=None)
                state = self._prepare_train_state_for_devices(state)
                if self.optimizer_name == 'kfac':
                    state = self._initialize_kfac_state(state, optimizer, data, key)
                state = state.replace(step=step)
                iterator.write(f'Extended MKAN grid to G={g_new} at train step {step_id}.')
                self.run_manager.log_scalars('train', step_id, {'grid_G': float(g_new)})

            checkpoint_state = self._build_checkpoint_state(
                stage='train',
                step=step_id,
                params=self._host_params(state.params),
                data=self._host_data(data),
                key=key,
                train_opt_state=self._host_opt_state(state.opt_state),
            )
            if self.run_manager.should_checkpoint(step_id, self.iterations):
                self.run_manager.checkpoints.save_step('train', step_id, checkpoint_state)

            runtime = runtime.replace(data=data, key=key)

    def run(self) -> None:
        try:
            (
                kan_init,
                signed_network,
                logabs_network,
                log_network,
                batch_network,
                batch_log_network,
                orbitals_vmap,
                extend_mkan_grid,
            ) = self._build_networks()
            params, data, sharded_key, pretrain_start_step, train_start_step, pretrain_opt_state, train_opt_state = (
                self._initialize_params_and_data(kan_init)
            )

            params, data, sharded_key, pretrain_opt_state, train_start_step = self.pretrain_runner.run(
                params=params,
                data=data,
                sharded_key=sharded_key,
                pretrain_start_step=pretrain_start_step,
                train_start_step=train_start_step,
                train_opt_state=train_opt_state,
                pretrain_opt_state=pretrain_opt_state,
                batch_network=batch_network,
                batch_log_network=batch_log_network,
                orbitals_vmap=orbitals_vmap,
                t_init=self.t_init,
            )
            del pretrain_opt_state

            optimizer, step_fn = self._build_train_step(signed_network, logabs_network, log_network)
            state = self._build_train_state(params, optimizer, train_opt_state)
            state = self._prepare_train_state_for_devices(state)

            runtime = RuntimeState(
                data=self._prepare_data_for_devices(data),
                key=sharded_key,
                mcmc_width=jnp.asarray(self.mcmc_width),
                pmoves=np.full((self.adapt_frequency,), np.nan, dtype=np.float32),
            )
            if self.optimizer_name == 'kfac' and train_opt_state is None:
                state = self._initialize_kfac_state(
                    state, optimizer, runtime.data, runtime.key)
            self._run_train_loop(
                train_start_step=train_start_step,
                runtime=runtime,
                state=state,
                step_fn=step_fn,
                signed_network=signed_network,
                logabs_network=logabs_network,
                log_network=log_network,
                extend_mkan_grid=extend_mkan_grid,
            )
        finally:
            self.run_manager.close()


def train(cfg: ml_collections.ConfigDict):
    """Main training loop entry."""
    trainer = VMCTrainer(cfg)
    # breakpoint()
    trainer.run()
