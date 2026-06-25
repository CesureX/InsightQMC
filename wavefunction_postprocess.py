#!/usr/bin/env python3
"""Post-process one-electron wavefunction plot data into compact reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPS = 1.0e-12


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _load_radial_data(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(data_path)
    r = np.asarray(data["r"], dtype=np.float64)
    psi = np.asarray(data["psi_radial"])
    amp = np.abs(psi).astype(np.float64)
    amp0 = float(amp[0]) if amp.size and amp[0] > 0.0 else float(np.max(amp))
    amp_norm = amp / max(amp0, EPS)
    valid = amp_norm > 1.0e-8
    return r[valid], amp_norm[valid], psi[valid]


def _poly_expr(coeffs: list[float], lhs: str) -> str:
    pieces = []
    for power, coef in enumerate(coeffs):
        if power == 0:
            pieces.append(f"{coef:.10g}")
        elif power == 1:
            pieces.append(f"{coef:+.10g} r")
        else:
            pieces.append(f"{coef:+.10g} r^{power}")
    return f"{lhs} ~= " + " ".join(pieces)


def _fit_polynomial(r: np.ndarray, y: np.ndarray, degree: int) -> dict[str, Any]:
    coeffs_desc = np.polyfit(r, y, int(degree))
    prediction = np.polyval(coeffs_desc, r)
    residual = y - prediction
    return {
        "degree": int(degree),
        "coefficients_descending": [float(v) for v in coeffs_desc],
        "coefficients_ascending": [float(v) for v in coeffs_desc[::-1]],
        "rms_error": float(np.sqrt(np.mean(residual**2))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "prediction": prediction,
        "residual": residual,
    }


def write_radial_exponential_fit(
    data_path: str | Path,
    out_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Fit a compact exponential surrogate to trained radial wavefunction data."""
    data_path = Path(data_path).expanduser()
    out_dir = Path(out_dir).expanduser() if out_dir else data_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    r, amp_norm, _psi = _load_radial_data(data_path)
    log_amp = np.log(np.maximum(amp_norm, EPS))
    design = np.column_stack([np.ones_like(r), r])
    coeffs, *_ = np.linalg.lstsq(design, log_amp, rcond=None)
    c0, c1 = [float(v) for v in coeffs]
    pred_log = design @ coeffs
    pred_amp = np.exp(pred_log)
    residual_log = log_amp - pred_log
    residual_amp = amp_norm - pred_amp
    rss = float(np.sum(residual_log**2))
    tss = float(np.sum((log_amp - np.mean(log_amp)) ** 2))
    payload = {
        "source_data": str(data_path),
        "known_analytic_answer_used": False,
        "normalization": "|Psi(r)| / |Psi(0)|",
        "fit_interval": {"r_min": float(np.min(r)), "r_max": float(np.max(r))},
        "model": "|Psi(r)| / |Psi(0)| ~= exp(c0 + c1*r)",
        "c0": c0,
        "c1": c1,
        "alpha": -c1,
        "rms_log_error": float(np.sqrt(np.mean(residual_log**2))),
        "max_abs_log_error": float(np.max(np.abs(residual_log))),
        "rms_amplitude_error": float(np.sqrt(np.mean(residual_amp**2))),
        "max_abs_amplitude_error": float(np.max(np.abs(residual_amp))),
        "r2_log_amplitude": float(1.0 - rss / max(tss, EPS)),
    }

    json_path = out_dir / "wavefunction_radial_fit.json"
    md_path = out_dir / "wavefunction_radial_fit.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2))
    md_path.write_text(
        "\n".join(
            [
                "# Radial Exponential Surrogate",
                "",
                "This report fits a compact exponential surrogate to the trained radial wavefunction data only. It does not use a known analytic answer as the fitting target.",
                "",
                "## Fitted Form",
                "",
                "`|Psi(r)| / |Psi(0)| ~= exp(c0 + c1*r)`",
                "",
                f"`|Psi(r)| / |Psi(0)| ~= exp({c0:.10g} {c1:+.10g} r)`",
                "",
                f"Equivalently, `alpha = {-c1:.10g}` in `exp(-alpha*r)`.",
                "",
                "## Fit Numbers",
                "",
                f"- `R^2` on log amplitude = `{payload['r2_log_amplitude']:.12g}`",
                f"- RMS log error = `{payload['rms_log_error']:.6e}`",
                f"- max log error = `{payload['max_abs_log_error']:.6e}`",
                f"- RMS amplitude error = `{payload['rms_amplitude_error']:.6e}`",
                f"- max amplitude error = `{payload['max_abs_amplitude_error']:.6e}`",
                "",
            ]
        )
    )
    return json_path, md_path


def write_polynomial_fit(
    data_path: str | Path,
    out_dir: str | Path | None = None,
    amplitude_degrees: tuple[int, ...] = (6, 8),
    log_degrees: tuple[int, ...] = (4, 6),
) -> tuple[Path, Path, Path]:
    """Fit powers of r to trained radial data and write Markdown/JSON/PNG."""
    data_path = Path(data_path).expanduser()
    out_dir = Path(out_dir).expanduser() if out_dir else data_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    r, amp_norm, _psi = _load_radial_data(data_path)
    log_amp = np.log(np.maximum(amp_norm, EPS))
    amp_fits = {int(degree): _fit_polynomial(r, amp_norm, degree) for degree in amplitude_degrees}
    log_fits = {int(degree): _fit_polynomial(r, log_amp, degree) for degree in log_degrees}

    payload = {
        "source_data": str(data_path),
        "known_analytic_answer_used": False,
        "fit_interval": {"r_min": float(np.min(r)), "r_max": float(np.max(r))},
        "normalization": "|Psi(r)| / |Psi(0)|",
        "amplitude_polynomials": {
            str(degree): {key: val for key, val in fit.items() if key not in {"prediction", "residual"}}
            for degree, fit in amp_fits.items()
        },
        "log_amplitude_polynomials": {
            str(degree): {key: val for key, val in fit.items() if key not in {"prediction", "residual"}}
            for degree, fit in log_fits.items()
        },
    }

    json_path = out_dir / "wavefunction_polynomial_fit.json"
    md_path = out_dir / "wavefunction_polynomial_fit.md"
    png_path = out_dir / "wavefunction_polynomial_fit.png"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2))

    md_lines = [
        "# Wavefunction Polynomial Fit",
        "",
        "This file uses only powers of `r` to approximate the trained radial wavefunction data. It does not use a known analytic answer as a target or candidate.",
        "",
        f"Fit interval: `r in [{np.min(r):.6g}, {np.max(r):.6g}]`.",
        "",
        "## Polynomial For Amplitude",
        "",
    ]
    for degree in sorted(amp_fits):
        fit = amp_fits[degree]
        md_lines.extend(
            [
                f"Degree {degree}:",
                "",
                f"`{_poly_expr(fit['coefficients_ascending'], '|Psi(r)|/|Psi(0)|')}`",
                "",
                f"- RMS error: `{fit['rms_error']:.6e}`",
                f"- max absolute error: `{fit['max_abs_error']:.6e}`",
                "",
            ]
        )
    md_lines.extend(
        [
            "## Polynomial For Log-Amplitude",
            "",
            "This is usually more stable for a decaying wavefunction. Exponentiating this polynomial gives the amplitude.",
            "",
        ]
    )
    for degree in sorted(log_fits):
        fit = log_fits[degree]
        md_lines.extend(
            [
                f"Degree {degree}:",
                "",
                f"`{_poly_expr(fit['coefficients_ascending'], 'log(|Psi(r)|/|Psi(0)|)')}`",
                "",
                f"- RMS log error: `{fit['rms_error']:.6e}`",
                f"- max absolute log error: `{fit['max_abs_error']:.6e}`",
                "",
            ]
        )
    md_lines.extend(
        [
            "## Note",
            "",
            "A finite polynomial is an approximation on this interval, not a global bound-state wavefunction form.",
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines))

    plot_degrees = sorted(amp_fits)[-2:]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), constrained_layout=True)
    axes[0].plot(r, amp_norm, label="trained |Psi|/|Psi(0)|", linewidth=1.8)
    for degree in plot_degrees:
        axes[0].plot(
            r,
            amp_fits[degree]["prediction"],
            "--" if degree == plot_degrees[0] else ":",
            label=f"degree {degree} polynomial",
            linewidth=1.5,
        )
    axes[0].set_xlabel("r / bohr")
    axes[0].set_ylabel("normalized amplitude")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    for degree in plot_degrees:
        axes[1].plot(
            r,
            amp_fits[degree]["residual"],
            label=f"degree {degree} residual",
            linewidth=1.4,
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("r / bohr")
    axes[1].set_ylabel("amplitude residual")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return json_path, md_path, png_path


def write_radial_postprocess_reports(
    data_path: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Path]:
    radial_json, radial_md = write_radial_exponential_fit(data_path, out_dir)
    poly_json, poly_md, poly_png = write_polynomial_fit(data_path, out_dir)
    return {
        "radial_fit_json": radial_json,
        "radial_fit_md": radial_md,
        "polynomial_fit_json": poly_json,
        "polynomial_fit_md": poly_md,
        "polynomial_fit_png": poly_png,
    }
