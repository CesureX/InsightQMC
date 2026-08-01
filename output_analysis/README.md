# QMC_LZW output analysis

Tools for reading `outputs/<run>/tensorboard/events...` or falling back to
`outputs/<run>/logs/metrics.csv`, exporting scalar CSV files, plotting training
curves, and summarizing the final stable segment.

## Usage

From the `QMC_LZW` directory:

```bash
conda run -p /vepfs-mlp2/c20250516/250504030/env/qmc python output_analysis/analyze_run.py \
  --run outputs/Li07281258_with_M_70k_testadam
```

Outputs are written to `<run>/analysis/` by default:

- `csv/*.csv`: one CSV per TensorBoard scalar tag
- `scalars_combined.csv`: selected scalar tags joined by step
- `training_curves.png`: loss, variance, pmove, and mcmc width
- `loss_tail.png`: tail-region loss with mean and standard-error band
- `summary.json`: statistics for the selected tail window

By default, summaries and `loss_tail.png` use the last 10% of scalar points. Use `--tail N` to override with the last N points, or `--out some/path` to choose a different output directory.
