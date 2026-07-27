QMC based on KAN
[![中关村学院 GitHub 组织](https://img.shields.io/badge/Linked%20to-bjzgcai%20Org-blue?logo=github)](https://github.com/bjzgcai)

## VMC optimizer

The production VMC stage supports Adam, AdamW, matrix-free
Rayleigh--Gauss--Newton, and KFAC. Hartree--Fock/DFT pretraining continues to
use Adam in every case.

Select AdamW with decoupled weight decay using:

```python
cfg.optimizer = 'adamw'
cfg.adamw.weight_decay = 1.0e-4
```

AdamW otherwise shares Adam's gradient clipping, moment coefficients, and
learning-rate schedule. Weight decay is applied to all trainable parameters.

Select RGN in the training config:

```python
cfg.optimizer = 'rgn'
cfg.rgn.split_compilation = True
cfg.rgn.epsilon = 0.01
cfg.rgn.eta = 1.0e-3
cfg.rgn.cg_maxiter = 20
cfg.rgn.cg_tol = 1.0e-4
cfg.rgn.step_scale = 1.0
cfg.rgn.max_update_norm = 0.1
```

RGN solves the matrix-free system
`[H_RGN + (S + eta I) / epsilon] delta = -gradient` with conjugate gradients.
It is substantially more expensive per iteration than Adam. Start with a
small `cg_maxiter` and a conservative `max_update_norm`; increase `epsilon`
only after training is stable. Set `cfg.optimizer = 'adam'` to retain the
original training path.

With `split_compilation=True` (the default), MCMC and the RGN/CG update are
compiled as separate XLA executables. Set it to `False` to recover the original
single combined training-step compilation.

Select KFAC with:

```python
cfg.optimizer = 'kfac'
cfg.multi_device = True
cfg.num_devices = 0
cfg.kfac.damping = 1.0e-3
cfg.kfac.norm_constraint = 1.0e-3
cfg.kfac.cov_ema_decay = 0.95
cfg.kfac.invert_every = 1
```

The KFAC path uses `kfac_jax.Optimizer` with exact-Fisher estimation and generic
layer registration. It currently uses all visible local JAX devices. To use a
subset, restrict `CUDA_VISIBLE_DEVICES` before starting the process. A compatible
`kfac-jax` installation is required.
