#!/usr/bin/env python3
"""Discover a compact wavefunction form from a trained one-electron run.

This script intentionally does not fit to the known hydrogen analytic answer.
It evaluates the trained network, checks whether the amplitude is effectively
radial, and then performs sparse regression on log-amplitude using a generic
library of coordinate functions.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_wavefunction import _evaluate_wavefunction, _phase_align
from run_inference import _build_network, _load_checkpoint, _load_config


EPS = 1.0e-12


def _fibonacci_sphere(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("--directions must be positive.")
    idx = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (idx + 0.5) / float(n)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = np.pi * (3.0 - np.sqrt(5.0)) * idx
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))


def _feature_library(r: np.ndarray) -> list[tuple[str, np.ndarray]]:
    r = np.asarray(r, dtype=np.float64)
    return [
        ("r", r),
        ("r^2", r**2),
        ("r^3", r**3),
        ("r^4", r**4),
        ("sqrt(r)", np.sqrt(np.maximum(r, 0.0))),
        ("log(1+r)", np.log1p(r)),
        ("1/(1+r)", 1.0 / (1.0 + r)),
        ("r/(1+r)", r / (1.0 + r)),
        ("tanh(r)", np.tanh(r)),
    ]


def _fit_linear_terms(
    y: np.ndarray,
    terms: list[tuple[str, np.ndarray]],
) -> dict[str, Any]:
    names = [name for name, _values in terms]
    cols = [values for _name, values in terms]
    x = np.column_stack([np.ones_like(y), *cols])
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coeffs
    residual = y - pred
    rss = float(np.sum(residual**2))
    n = int(y.size)
    k = int(x.shape[1])
    return {
        "terms": names,
        "coefficients": [float(v) for v in coeffs],
        "rms_log_error": float(np.sqrt(np.mean(residual**2))),
        "max_abs_log_error": float(np.max(np.abs(residual))),
        "rss": rss,
        "aic": float(n * np.log(rss / max(n, 1) + EPS) + 2 * k),
        "bic": float(n * np.log(rss / max(n, 1) + EPS) + k * np.log(max(n, 2))),
        "prediction": pred,
        "residual": residual,
    }


def _all_sparse_models(
    r: np.ndarray,
    y: np.ndarray,
    max_terms: int,
) -> list[dict[str, Any]]:
    library = _feature_library(r)
    models: list[dict[str, Any]] = []
    upper = min(max_terms, len(library))
    for size in range(1, upper + 1):
        for combo in itertools.combinations(library, size):
            models.append(_fit_linear_terms(y, list(combo)))
    models.sort(key=lambda item: item["bic"])
    return models


def _expr_from_model(model: dict[str, Any], response_name: str = "log(|Psi|/|Psi(0)|)") -> str:
    coeffs = model["coefficients"]
    terms = model["terms"]
    pieces = [f"{coeffs[0]:+.10g}"]
    for coef, term in zip(coeffs[1:], terms):
        pieces.append(f"{coef:+.10g}*{term}")
    return f"{response_name} ~= " + " ".join(pieces)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items() if key not in {"prediction", "residual"}}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _write_report(
    path: Path,
    payload: dict[str, Any],
    primary_model: dict[str, Any],
    best_bic_model: dict[str, Any],
) -> None:
    primary_coeffs = primary_model["coefficients"]
    primary_terms = primary_model["terms"]
    lines = [
        "# Wavefunction Form Discovery",
        "",
        "This report is generated from the trained checkpoint only. It does not fit to the known hydrogen analytic wavefunction, and `exp(-r)` is not included in the candidate library.",
        "",
        "## Radiality Check",
        "",
        f"- angular directions per radius: `{payload['settings']['directions']}`",
        f"- mean angular coefficient of variation of `|Psi|`: `{payload['radiality']['mean_cv']:.6e}`",
        f"- max angular coefficient of variation of `|Psi|`: `{payload['radiality']['max_cv']:.6e}`",
        "",
        "Small angular variation means the trained wavefunction amplitude is well described as a function of `r` alone.",
        "",
        "## Candidate Library",
        "",
        "`r`, `r^2`, `r^3`, `r^4`, `sqrt(r)`, `log(1+r)`, `1/(1+r)`, `r/(1+r)`, `tanh(r)`",
        "",
        "## Discovered Compact Law",
        "",
        f"`{_expr_from_model(primary_model)}`",
        "",
    ]
    if len(primary_terms) == 1 and primary_terms[0] == "r":
        c0, c1 = primary_coeffs
        lines.extend(
            [
                "Because the simplest selected coordinate is `r`, exponentiating the discovered log-amplitude relation gives:",
                "",
                f"`|Psi(r)| / |Psi(0)| ~= exp({c0:.10g}) * exp({c1:.10g} r)`",
                "",
                f"Equivalently, the discovered decay constant is `alpha = {-c1:.10g}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Fit Quality",
            "",
            f"- compact model terms: `{', '.join(primary_model['terms'])}`",
            f"- compact model RMS log error: `{primary_model['rms_log_error']:.6e}`",
            f"- compact model max log error: `{primary_model['max_abs_log_error']:.6e}`",
            f"- best BIC model terms: `{', '.join(best_bic_model['terms'])}`",
            f"- best BIC model RMS log error: `{best_bic_model['rms_log_error']:.6e}`",
            "",
            "The BIC model is shown as a possible small correction. The compact law is the first-principles readable form: it comes from the trained wavefunction data and a generic sparse library, not from a hydrogen-answer template.",
            "",
            "## Best BIC Sparse Model",
            "",
            f"`{_expr_from_model(best_bic_model)}`",
            "",
            "## Outputs",
            "",
            f"- data: `{path.with_suffix('.json').name}`",
            f"- plot: `{path.with_suffix('.png').name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def generate_wavefunction_discovery(
    *,
    run_dir: str | Path,
    checkpoint: str | Path | None = None,
    out_dir: str | Path | None = None,
    r_max: float = 6.0,
    radial_points: int = 600,
    directions: int = 96,
    angular_radii: list[float] | tuple[float, ...] | np.ndarray | None = None,
    max_terms: int = 3,
    top_models: int = 12,
) -> dict[str, Path]:
    run_dir = Path(run_dir).expanduser()
    checkpoint_path = (
        Path(checkpoint).expanduser()
        if checkpoint
        else run_dir / "checkpoints" / "last.pkl"
    )
    out_dir = (
        Path(out_dir).expanduser()
        if out_dir
        else run_dir / "mkan_interpretation" / "wavefunction_plots"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(run_dir / "config.json")
    nelectrons = int(sum(cfg.system.electrons))
    if nelectrons != 1:
        raise ValueError(
            "discover_wavefunction_form.py currently handles one-electron wavefunctions. "
            f"This run has {nelectrons} electrons."
        )

    checkpoint_data = _load_checkpoint(checkpoint_path)
    params = checkpoint_data["params"]
    signed_network, _orbitals_apply, atoms, charges, spins, _electrons = _build_network(
        cfg,
        checkpoint_data,
    )

    rs = np.linspace(0.0, float(r_max), int(radial_points))
    radial_pos = np.column_stack([rs, np.zeros_like(rs), np.zeros_like(rs)])
    psi_radial = _evaluate_wavefunction(
        signed_network,
        params,
        spins,
        atoms,
        charges,
        radial_pos,
    ).astype(np.complex128, copy=False)
    psi_radial, _phase = _phase_align(psi_radial)

    amp = np.abs(psi_radial)
    amp0 = float(amp[0]) if amp.size and amp[0] > 0.0 else float(np.max(amp))
    amp_norm = amp / max(amp0, EPS)
    valid = amp_norm > 1.0e-8
    fit_r = rs[valid]
    log_amp = np.log(np.maximum(amp_norm[valid], EPS))

    sparse_models = _all_sparse_models(fit_r, log_amp, int(max_terms))
    one_term_models = [model for model in sparse_models if len(model["terms"]) == 1]
    primary_model = min(one_term_models, key=lambda item: item["rms_log_error"])
    best_bic_model = sparse_models[0]

    angular_radii_arr = (
        np.asarray(angular_radii, dtype=np.float64)
        if angular_radii is not None
        else np.linspace(0.25, min(float(r_max), 5.0), 20)
    )
    direction_vectors = _fibonacci_sphere(int(directions))
    angular_positions = np.concatenate(
        [radius * direction_vectors for radius in angular_radii_arr],
        axis=0,
    )
    angular_psi = _evaluate_wavefunction(
        signed_network,
        params,
        spins,
        atoms,
        charges,
        angular_positions,
    )
    angular_amp = np.abs(
        np.asarray(angular_psi).reshape(len(angular_radii_arr), len(direction_vectors))
    )
    angular_mean = np.mean(angular_amp, axis=1)
    angular_std = np.std(angular_amp, axis=1)
    angular_cv = angular_std / np.maximum(angular_mean, EPS)

    primary_fit = _fit_linear_terms(
        log_amp,
        [(name, values) for name, values in _feature_library(fit_r) if name in primary_model["terms"]],
    )
    bic_fit = _fit_linear_terms(
        log_amp,
        [(name, values) for name, values in _feature_library(fit_r) if name in best_bic_model["terms"]],
    )

    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "settings": {
            "r_max": float(r_max),
            "radial_points": int(radial_points),
            "directions": int(directions),
            "max_terms": int(max_terms),
            "candidate_library": [name for name, _values in _feature_library(fit_r)],
            "known_analytic_answer_used": False,
        },
        "radiality": {
            "radii": angular_radii_arr.tolist(),
            "cv_by_radius": angular_cv.tolist(),
            "mean_cv": float(np.mean(angular_cv)),
            "max_cv": float(np.max(angular_cv)),
        },
        "primary_compact_model": _jsonable(primary_model),
        "best_bic_model": _jsonable(best_bic_model),
        "top_bic_models": _jsonable(sparse_models[: int(top_models)]),
    }

    json_path = out_dir / "wavefunction_discovery.json"
    md_path = out_dir / "wavefunction_discovery.md"
    png_path = out_dir / "wavefunction_discovery.png"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2))
    _write_report(md_path, payload, primary_model, best_bic_model)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), constrained_layout=True)
    axes[0].plot(fit_r, log_amp, label="trained log amplitude", linewidth=1.8)
    axes[0].plot(
        fit_r,
        primary_fit["prediction"],
        "--",
        label="compact discovered law",
        linewidth=1.6,
    )
    axes[0].plot(
        fit_r,
        bic_fit["prediction"],
        ":",
        label="best BIC sparse correction",
        linewidth=1.6,
    )
    axes[0].set_xlabel("r / bohr")
    axes[0].set_ylabel("log(|Psi| / |Psi(0)|)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(fit_r, primary_fit["residual"], label="compact residual", linewidth=1.4)
    axes[1].plot(fit_r, bic_fit["residual"], label="best BIC residual", linewidth=1.2)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("r / bohr")
    axes[1].set_ylabel("log residual")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    axes[2].plot(angular_radii_arr, angular_cv, marker="o", linewidth=1.4)
    axes[2].set_xlabel("r / bohr")
    axes[2].set_ylabel("angular CV of |Psi|")
    axes[2].grid(alpha=0.25)
    axes[2].set_title("Radiality check from sampled directions")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    return {
        "discovery_md": md_path,
        "discovery_json": json_path,
        "discovery_plot": png_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Training output directory.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to <run-dir>/checkpoints/last.pkl.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <run-dir>/mkan_interpretation/wavefunction_plots.",
    )
    parser.add_argument("--r-max", type=float, default=6.0)
    parser.add_argument("--radial-points", type=int, default=600)
    parser.add_argument("--directions", type=int, default=96)
    parser.add_argument(
        "--angular-radii",
        type=float,
        nargs="*",
        default=None,
        help="Radii used for angular-variation checks.",
    )
    parser.add_argument("--max-terms", type=int, default=3)
    parser.add_argument("--top-models", type=int, default=12)
    args = parser.parse_args()

    outputs = generate_wavefunction_discovery(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        r_max=args.r_max,
        radial_points=args.radial_points,
        directions=args.directions,
        angular_radii=args.angular_radii,
        max_terms=args.max_terms,
        top_models=args.top_models,
    )
    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
