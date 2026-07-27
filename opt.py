import optax
import jax.numpy as jnp
import networks
from typing import Any, Union, Tuple, Optional
import chex
import jax
import constants
import functools
from typing_extensions import Protocol
import loss as qmc_loss_functions

try:
  import kfac_jax
except Exception:
  kfac_jax = None


OptimizerState = Union[optax.OptState, Any]
OptUpdateResults = Tuple[networks.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]

class OptUpdate(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      data: networks.KANetsData,
      opt_state: optax.OptState,
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters accordingly.

    Args:
      params: network parameters.
      data: electron positions, spins and atomic positions.
      opt_state: optimizer internal state.
      key: RNG state.

    Returns:
      Tuple of (params, opt_state, loss, aux_data), where params and opt_state
      are the updated parameters and optimizer state, loss is the evaluated loss
      and aux_data auxiliary data (see AuxiliaryLossData docstring).
    """


StepResults = Tuple[
    networks.KANetsData,
    networks.ParamTree,
    Optional[optax.OptState],
    jnp.ndarray,
    qmc_loss_functions.AuxiliaryLossData,
    jnp.ndarray,
]


class Step(Protocol):

  def __call__(
      self,
      data: networks.KANetsData,
      params: networks.ParamTree,
      state: OptimizerState,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """Performs one set of MCMC moves and an optimization step.

    Args:
      data: batch of MCMC configurations, spins and atomic positions.
      params: network parameters.
      state: optimizer internal state.
      key: JAX RNG state.
      mcmc_width: width of MCMC move proposal. See vmcmc.make_vmcmc_step.

    Returns:
      Tuple of (data, params, state, loss, aux_data, pmove).
        data: Updated MCMC configurations drawn from the network given the
          *input* network parameters.
        params: updated network parameters after the gradient update.
        state: updated optimization state.
        loss: energy of system based on input network parameters averaged over
          the entire set of MCMC configurations.
        aux_data: AuxiliaryLossData object also returned from evaluating the
          loss of the system.
        pmove: probability that a proposed MCMC move was accepted.
    """


def null_update(
    params: networks.ParamTree,
    data: networks.KANetsData,
    opt_state: Optional[optax.OptState],
    key: chex.PRNGKey,
) -> OptUpdateResults:
  """Performs an identity operation with an OptUpdate interface."""
  del data, key
  return params, opt_state, jnp.zeros(1), None


def make_opt_update_step(evaluate_loss: qmc_loss_functions.LossFn,
                         optimizer: optax.GradientTransformation) -> OptUpdate:
  """Returns an OptUpdate function for performing a parameter update."""

  # Differentiate wrt parameters (argument 0)
  loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

  def opt_update(
      params: networks.ParamTree,
      data: networks.KANetsData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters using optax."""
    (loss, aux_data), grad = loss_and_grad(params, key, data)
    # Single-device: identity. Multi-device (inside pmap): cross-device mean.
    grad = constants.pmean(grad)
    #jax.debug.print("grad:{}", grad.type)
    #jax.debug.print("params:{}", params.type)
    #jax.debug.print("opt_state:{}", opt_state.type)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux_data

  return opt_update


def make_rgn_update_step(
    evaluate_loss: qmc_loss_functions.LossFn,
    network,
    local_energy,
    *,
    epsilon: float,
    eta: float,
    cg_maxiter: int,
    cg_tol: float,
    step_scale: float = 1.0,
    max_update_norm: float = 0.0,
    reset_if_nan: bool = True,
) -> OptUpdate:
  """Build a matrix-free Rayleigh--Gauss--Newton VMC update.

  The linear system is ``[H_RGN + (S + eta I) / epsilon] delta = -g``.
  ``S`` is the centered log-wavefunction Gram matrix and ``H_RGN`` is the
  Rayleigh curvature on the wavefunction tangent space. Neither matrix is
  materialized; JVP/VJP products and conjugate gradients are used instead.

  This implementation assumes real trainable parameters, while allowing the
  wavefunction and local energies to be complex.
  """
  if epsilon <= 0.0:
    raise ValueError('rgn.epsilon must be positive.')
  if eta <= 0.0:
    raise ValueError('rgn.eta must be positive (CG requires regularization).')
  if cg_maxiter <= 0:
    raise ValueError('rgn.cg_maxiter must be positive.')

  loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
  batch_network = jax.vmap(
      network, in_axes=(None, 0, None, None, None), out_axes=0)
  batch_local_energy = jax.vmap(
      local_energy,
      in_axes=(None, 0, networks.KANetsData(
          positions=0, spins=None, atoms=None, charges=None)),
      out_axes=(0, 0),
  )

  def tree_add(*trees):
    return jax.tree.map(lambda *xs: sum(xs), *trees)

  def tree_scale(scale, tree):
    return jax.tree.map(lambda x: scale * x, tree)

  def tree_real(tree):
    return jax.tree.map(lambda x: jnp.real(x).astype(x.dtype), tree)

  def tree_squared_norm(tree):
    leaves = jax.tree.leaves(tree)
    return sum(jnp.real(jnp.vdot(x, x)) for x in leaves)

  def opt_update(params, data, opt_state, key):
    (loss, aux_data), grad = loss_and_grad(params, key, data)
    grad = tree_real(constants.pmean(grad))
    keys = jax.random.split(key, num=data.positions.shape[0])

    def log_values(p):
      return batch_network(
          p, data.positions, data.spins, data.atoms, data.charges)

    def energy_values(p):
      return batch_local_energy(p, keys, data)[0]

    log_primal, log_linear = jax.linearize(log_values, params)
    energy_primal, energy_linear = jax.linearize(energy_values, params)
    del log_primal

    def global_mean(x):
      return constants.pmean(jnp.mean(x))

    energy_mean = global_mean(energy_primal)

    def centered_log_tangent(v):
      tangent = log_linear(v)
      return tangent - global_mean(tangent)

    # A scalar VJP is used so complex wavefunctions and real parameters follow
    # JAX's well-defined real differential convention.
    def tangent_pullback(cotangent):
      cotangent = jax.lax.stop_gradient(cotangent)

      def pairing(p):
        values = log_values(p)
        values = values - global_mean(values)
        return jnp.real(global_mean(jnp.conj(values) * cotangent))

      return jax.grad(pairing)(params)

    inv_epsilon = 1.0 / epsilon

    def matvec(v):
      qv = centered_log_tangent(v)
      dlocal_v = energy_linear(v)
      # H(d psi) / psi - E (d psi) / psi = dE_local +
      # (E_local - E) dlog(psi).
      rayleigh_residual = (
          dlocal_v + (energy_primal - energy_mean) * qv)
      h_v = tangent_pullback(rayleigh_residual)
      s_v = tangent_pullback(qv)
      result = tree_add(
          h_v,
          tree_scale(inv_epsilon, s_v),
          tree_scale(inv_epsilon * eta, v),
      )
      return tree_real(constants.pmean(result))

    rhs = tree_scale(-1.0, grad)
    direction, _ = jax.scipy.sparse.linalg.cg(
        matvec, rhs, tol=cg_tol, maxiter=cg_maxiter)
    direction = tree_scale(step_scale, direction)
    if max_update_norm > 0.0:
      norm = jnp.sqrt(tree_squared_norm(direction))
      scale = jnp.minimum(1.0, max_update_norm / (norm + 1.0e-16))
      direction = tree_scale(scale, direction)
    new_params = optax.apply_updates(params, direction)
    if reset_if_nan:
      invalid = ~jnp.isfinite(loss)
      new_params = jax.tree.map(
          lambda new, old: jnp.where(invalid, old, new), new_params, params)
    return new_params, opt_state, loss, aux_data

  return opt_update


def make_split_training_step(mcmc_step, optimizer_step) -> Step:
  """Compose separately compiled MCMC and optimizer executables.

  Unlike ``make_training_step``, this wrapper is intentionally not jitted. Its
  two arguments should already be jitted or pmapped. This keeps the sampler and
  the (potentially very large) RGN/CG graph in separate XLA executables.
  """
  def step(data, params, state, key, mcmc_width):
    if key.ndim == 1:
      mcmc_key, loss_key = jax.random.split(key, num=2)
    else:
      split_keys = jax.vmap(lambda k: jax.random.split(k, num=2))(key)
      mcmc_key, loss_key = split_keys[:, 0], split_keys[:, 1]
    data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)
    new_params, new_state, loss, aux_data = optimizer_step(
        params, data, state, loss_key)
    return data, new_params, new_state, loss, aux_data, pmove

  return step


def make_kfac_training_step(
    mcmc_step,
    optimizer,
    *,
    damping: float,
) -> Step:
  """Build a full MCMC + KFAC update for an already pmapped setup."""
  if kfac_jax is None:
    raise ImportError(
        'KFAC was selected but kfac_jax could not be imported. Install a '
        'kfac-jax version compatible with the active JAX version.')

  shared_momentum = kfac_jax.utils.replicate_all_local_devices(
      jnp.zeros([]), axis_name=constants.PMAP_AXIS_NAME)
  shared_damping = kfac_jax.utils.replicate_all_local_devices(
      jnp.asarray(damping), axis_name=constants.PMAP_AXIS_NAME)

  def step(data, params, state, key, mcmc_width):
    mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
    data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)
    new_params, new_state, stats = optimizer.step(
        params=params,
        state=state,
        rng=loss_keys,
        # kfac_jax internally pmaps every array leaf in ``batch`` over its
        # leading device axis.  KANetsData deliberately keeps spins, atoms,
        # and charges shared (without that axis), so passing the whole object
        # makes metadata such as a single atom at shape (1, 3) look sharded
        # across all GPUs.  The KFAC value function closes over that immutable
        # metadata and accepts only the genuinely sharded walker positions.
        batch=data.positions,
        momentum=shared_momentum,
        damping=shared_damping,
    )
    # kfac_jax donates params/state buffers to its compiled step.  The old
    # arrays are therefore invalid after the call and cannot be used for the
    # rollback pattern used by the Optax/RGN paths.
    return data, new_params, new_state, stats['loss'], stats['aux'], pmove

  return step


def make_loss_step(evaluate_loss: qmc_loss_functions.LossFn) -> OptUpdate:
  """Returns an OptUpdate function for evaluating the loss."""

  def loss_eval(
      params: networks.ParamTree,
      data: networks.KANetsData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates just the loss and gradients with an OptUpdate interface."""
    loss, aux_data = evaluate_loss(params, key, data)
    return params, opt_state, loss, aux_data

  return loss_eval


def make_training_step(
    mcmc_step,
    optimizer_step: OptUpdate,
    reset_if_nan: bool = False,
    jit: bool = True,
) -> Step:
  """Factory to create training step for non-KFAC optimizers."""
  #@functools.partial(jax.vmap, donate_argnums=(0, 1, 2)) we dont have the parallel strategy for it. So comment out this line.
  def step(
      data: networks.KANetsData,
      params: networks.ParamTree,
      state: Optional[optax.OptState],
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration (except for KFAC): MCMC steps + optimization."""
    # MCMC loop
    mcmc_key, loss_key = jax.random.split(key, num=2)
    data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)
    #data, pmove = mcmc_step(params, data, mcmc_key)

    # Optimization step
    new_params, new_state, loss, aux_data = optimizer_step(params,
                                                           data,
                                                           state,
                                                           loss_key)
    if reset_if_nan:
      new_params = jax.lax.cond(jnp.isnan(loss),
                                lambda: params,
                                lambda: new_params)
      new_state = jax.lax.cond(jnp.isnan(loss),
                               lambda: state,
                               lambda: new_state)
    return data, new_params, new_state, loss, aux_data, pmove

  if jit:
    return jax.jit(step)
  return step
