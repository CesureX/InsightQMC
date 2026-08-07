"""Post-training MKAN input-range diagnostics and plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


_STAT_NAMES = ("min", "p01", "median", "p99", "max", "mean", "std")


def _as_feature_matrix(values, *, name: str) -> np.ndarray:
    """Convert inputs to ``(rows, features)`` while preserving the feature axis."""

    values = np.asarray(values)
    if values.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions; got {values.shape}.")
    values = values.reshape(-1, values.shape[-1])
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and feature.")
    return values


def _feature_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "min": np.nanmin(values, axis=0),
        "p01": np.nanquantile(values, 0.01, axis=0),
        "median": np.nanquantile(values, 0.50, axis=0),
        "p99": np.nanquantile(values, 0.99, axis=0),
        "max": np.nanmax(values, axis=0),
        "mean": np.nanmean(values, axis=0),
        "std": np.nanstd(values, axis=0),
    }


def _layer_bounds(layer, layer_type: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return nominal lower/upper bounds with shape ``(n_edges_or_1, n_in)``."""

    layer_type = str(layer_type).lower()

    if layer_type == "base":
        grid = np.asarray(layer.grid.item).reshape(layer.n_out, layer.n_in, -1)
        return grid[:, :, layer.k], grid[:, :, -(layer.k + 1)]

    if layer_type == "spline":
        grid = np.asarray(layer.grid.item)
        return grid[None, :, layer.k], grid[None, :, -(layer.k + 1)]

    if layer_type == "rbf":
        grid = np.asarray(layer.grid.item)
        return np.min(grid, axis=-1)[None, :], np.max(grid, axis=-1)[None, :]

    if layer_type == "fastkan":
        grid = np.asarray(layer.grid[...]).reshape(-1)
        lower = np.full((1, layer.n_in), np.min(grid), dtype=grid.dtype)
        upper = np.full((1, layer.n_in), np.max(grid), dtype=grid.dtype)
        return lower, upper

    if layer_type in ("chebyshev", "legendre"):
        return -np.ones((1, layer.n_in)), np.ones((1, layer.n_in))

    if layer_type == "relukan":
        lower = np.min(np.asarray(layer.phase_low[...]), axis=-1)
        upper = np.max(np.asarray(layer.phase_high[...]), axis=-1)
        return lower[None, :], upper[None, :]

    # Fourier, sine, and learned wavelet bases have no fixed nominal interval.
    return None


def _outside_fraction(
    values: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray | None:
    if bounds is None:
        return None
    lower, upper = bounds
    if lower.shape != upper.shape or lower.shape[-1] != values.shape[-1]:
        raise ValueError(
            "Layer boundary shape does not match its basis input: "
            f"lower={lower.shape}, upper={upper.shape}, input={values.shape}."
        )
    outside = (
        (values[:, None, :] < lower[None, :, :])
        | (values[:, None, :] > upper[None, :, :])
    )
    return np.mean(outside, axis=(0, 1))


def _gaussian_inactive_fraction(
    layer,
    raw_input: np.ndarray,
    threshold: float,
) -> np.ndarray:
    # Pass the raw input: FastKAN.basis performs LayerNorm internally.
    basis = np.asarray(layer.basis(raw_input))
    if basis.ndim != 3 or basis.shape[:2] != raw_input.shape:
        raise ValueError(
            "Gaussian basis output must have shape (rows, features, basis_functions); "
            f"got {basis.shape}."
        )
    return np.mean(np.max(basis, axis=-1) < threshold, axis=0)


def _optional_float(value):
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _write_csv(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    with (output_dir / "feature_ranges.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_layer(
    *,
    output_dir: Path,
    layer_idx: int,
    layer_type: str,
    raw_input: np.ndarray,
    basis_input: np.ndarray,
    raw_stats: dict[str, np.ndarray],
    basis_stats: dict[str, np.ndarray],
    bounds: tuple[np.ndarray, np.ndarray] | None,
    outside_fraction: np.ndarray | None,
    inactive_fraction: np.ndarray | None,
    saturation_fraction: np.ndarray | None,
    histogram_bins: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    raw_flat = raw_input[np.isfinite(raw_input)]
    basis_flat = basis_input[np.isfinite(basis_input)]
    if raw_flat.size:
        axes[0].hist(
            raw_flat,
            bins=histogram_bins,
            density=True,
            alpha=0.55,
            label="raw input",
        )
    if basis_flat.size and not np.array_equal(raw_input, basis_input):
        axes[0].hist(
            basis_flat,
            bins=histogram_bins,
            density=True,
            alpha=0.55,
            label="basis input",
        )
    axes[0].set_title(f"Layer {layer_idx}: input distribution")
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("density")
    axes[0].grid(alpha=0.25)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False)

    feature_ids = np.arange(basis_input.shape[1])
    axes[1].fill_between(
        feature_ids,
        basis_stats["p01"],
        basis_stats["p99"],
        alpha=0.25,
        label="p01-p99",
    )
    axes[1].plot(feature_ids, basis_stats["median"], label="median", linewidth=1.5)
    if bounds is not None:
        lower, upper = bounds
        axes[1].plot(feature_ids, np.mean(lower, axis=0), "--", label="lower bound")
        axes[1].plot(feature_ids, np.mean(upper, axis=0), "--", label="upper bound")
    axes[1].set_title("Effective basis-input range by feature")
    axes[1].set_xlabel("input feature")
    axes[1].set_ylabel("value")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    has_metric = False
    if outside_fraction is not None:
        axes[2].plot(feature_ids, outside_fraction, label="outside nominal range")
        has_metric = True
    if inactive_fraction is not None:
        axes[2].plot(feature_ids, inactive_fraction, label="Gaussian basis inactive")
        has_metric = True
    if saturation_fraction is not None:
        axes[2].plot(feature_ids, saturation_fraction, label="|basis input| > 0.99")
        has_metric = True
    if has_metric:
        axes[2].set_ylim(-0.02, 1.02)
        axes[2].legend(frameon=False)
    else:
        axes[2].text(
            0.5,
            0.5,
            "This basis has no fixed nominal input boundary.",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
    axes[2].set_title("Boundary diagnostics")
    axes[2].set_xlabel("input feature")
    axes[2].set_ylabel("fraction")
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        f"MKAN {layer_type} layer {layer_idx} ({raw_input.shape[0]} sampled rows)"
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"layer_{layer_idx:02d}_ranges.png", dpi=180)
    plt.close(fig)


def analyze_activation_ranges(
    *,
    model,
    samples,
    output_dir: str | Path,
    histogram_bins: int = 100,
    inactive_threshold: float = 1.0e-3,
) -> dict:
    """Collect, summarize, and plot MKAN inputs after training."""

    if histogram_bins <= 0:
        raise ValueError("histogram_bins must be positive.")
    if inactive_threshold <= 0.0:
        raise ValueError("inactive_threshold must be positive.")
    if not hasattr(model, "collect_layer_inputs"):
        raise TypeError("model must provide collect_layer_inputs(samples).")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = model.collect_layer_inputs(samples)
    rows: list[dict] = []
    layer_summaries = []
    sampled_rows = None

    for record in records:
        layer_idx = int(record["layer"])
        layer = model.layers[layer_idx]
        raw_input = _as_feature_matrix(
            record["raw_input"], name=f"layer {layer_idx} raw_input"
        )
        basis_input = _as_feature_matrix(
            record["basis_input"], name=f"layer {layer_idx} basis_input"
        )
        if raw_input.shape != basis_input.shape:
            raise ValueError(
                f"Layer {layer_idx} raw and basis inputs have different shapes: "
                f"{raw_input.shape} and {basis_input.shape}."
            )
        if sampled_rows is None:
            sampled_rows = int(raw_input.shape[0])

        raw_stats = _feature_stats(raw_input)
        basis_stats = _feature_stats(basis_input)
        bounds = _layer_bounds(layer, model.layer_type)
        outside = _outside_fraction(basis_input, bounds)
        inactive = None
        if model.layer_type in ("rbf", "fastkan"):
            inactive = _gaussian_inactive_fraction(
                layer, raw_input, inactive_threshold
            )
        saturation = None
        if model.layer_type in ("chebyshev", "legendre"):
            saturation = np.mean(np.abs(basis_input) > 0.99, axis=0)

        lower = upper = None
        if bounds is not None:
            lower, upper = (np.mean(value, axis=0) for value in bounds)

        for feature_idx in range(raw_input.shape[1]):
            row = {
                "layer": layer_idx,
                "feature": feature_idx,
            }
            for name in _STAT_NAMES:
                row[f"raw_{name}"] = float(raw_stats[name][feature_idx])
                row[f"basis_{name}"] = float(basis_stats[name][feature_idx])
            row["boundary_lower"] = "" if lower is None else float(lower[feature_idx])
            row["boundary_upper"] = "" if upper is None else float(upper[feature_idx])
            row["outside_fraction"] = "" if outside is None else float(outside[feature_idx])
            row["inactive_fraction"] = "" if inactive is None else float(inactive[feature_idx])
            row["saturation_fraction"] = "" if saturation is None else float(saturation[feature_idx])
            rows.append(row)

        layer_summaries.append(
            {
                "layer": layer_idx,
                "n_features": int(raw_input.shape[1]),
                "outside_fraction_mean": _optional_float(
                    None if outside is None else np.mean(outside)
                ),
                "outside_fraction_max": _optional_float(
                    None if outside is None else np.max(outside)
                ),
                "inactive_fraction_mean": _optional_float(
                    None if inactive is None else np.mean(inactive)
                ),
                "saturation_fraction_mean": _optional_float(
                    None if saturation is None else np.mean(saturation)
                ),
            }
        )

        _plot_layer(
            output_dir=output_dir,
            layer_idx=layer_idx,
            layer_type=model.layer_type,
            raw_input=raw_input,
            basis_input=basis_input,
            raw_stats=raw_stats,
            basis_stats=basis_stats,
            bounds=bounds,
            outside_fraction=outside,
            inactive_fraction=inactive,
            saturation_fraction=saturation,
            histogram_bins=histogram_bins,
        )

    if sampled_rows is None:
        raise ValueError("collect_layer_inputs returned no layer records.")

    summary = {
        "layer_type": model.layer_type,
        "sampled_rows": sampled_rows,
        "inactive_threshold": float(inactive_threshold),
        "layers": layer_summaries,
    }
    _write_csv(rows, output_dir)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary
