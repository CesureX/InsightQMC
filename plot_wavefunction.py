#!/usr/bin/env python3
"""Plot the final one-electron wavefunction from an InsightQMC checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_inference import _build_network, _load_checkpoint, _load_config
from wavefunction_postprocess import write_radial_postprocess_reports


def _evaluate_wavefunction(signed_network, params, spins, atoms, charges, positions):
    batch_signed = jax.jit(
        jax.vmap(
            signed_network,
            in_axes=(None, 0, None, None, None),
            out_axes=(0, 0),
        )
    )
    signs, logabs = batch_signed(
        params,
        jnp.asarray(positions, dtype=jnp.float32),
        spins,
        atoms,
        charges,
    )
    return np.asarray(signs) * np.exp(np.asarray(logabs))


def _phase_align(psi):
    ref_idx = int(np.argmax(np.abs(psi)))
    if np.abs(psi[ref_idx]) == 0.0:
        return psi, 1.0 + 0.0j
    factor = np.exp(-1j * np.angle(psi[ref_idx]))
    return psi * factor, factor


def generate_wavefunction_plots(
    *,
    run_dir: str | Path,
    checkpoint: str | Path | None = None,
    out_dir: str | Path | None = None,
    r_max: float = 6.0,
    radial_points: int = 500,
    plane_extent: float = 4.0,
    grid_size: int = 181,
    phase_align: bool = True,
    reference_hydrogen_1s: bool = False,
    write_reports: bool = True,
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
            "plot_wavefunction.py currently plots one-electron wavefunctions. "
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
    if phase_align:
        psi_radial, phase_factor = _phase_align(psi_radial)
    else:
        phase_factor = 1.0 + 0.0j

    norm = float(np.max(np.abs(psi_radial))) or 1.0
    psi_radial_norm = psi_radial / norm

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), constrained_layout=True)
    axes[0].plot(rs, np.real(psi_radial_norm), label="Re Psi", linewidth=1.8)
    axes[0].plot(rs, np.imag(psi_radial_norm), label="Im Psi", linewidth=1.4)
    axes[0].plot(rs, np.abs(psi_radial_norm), label="|Psi|", linewidth=1.8)
    axes[0].set_xlabel("r / bohr, along x axis")
    axes[0].set_ylabel("normalized wavefunction")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].set_title("Final wavefunction on radial line")

    axes[1].plot(rs, np.abs(psi_radial_norm), label="|Psi| normalized", linewidth=1.8)
    if reference_hydrogen_1s:
        hydrogen_1s = np.exp(-rs)
        hydrogen_1s = hydrogen_1s / np.max(hydrogen_1s)
        axes[1].plot(rs, hydrogen_1s, "--", label="exp(-r), normalized", linewidth=1.6)
    axes[1].set_xlabel("r / bohr")
    axes[1].set_ylabel("normalized amplitude")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1.0e-4, 1.2)
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False)
    axes[1].set_title("Radial amplitude on log scale")
    radial_path = out_dir / "wavefunction_radial.png"
    fig.savefig(radial_path, dpi=180)
    plt.close(fig)

    extent = float(plane_extent)
    n_grid = int(grid_size)
    xs = np.linspace(-extent, extent, n_grid)
    ys = np.linspace(-extent, extent, n_grid)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    plane_pos = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    psi_plane = _evaluate_wavefunction(
        signed_network,
        params,
        spins,
        atoms,
        charges,
        plane_pos,
    ).reshape(n_grid, n_grid)
    psi_plane = psi_plane.astype(np.complex128, copy=False) * phase_factor
    plane_norm = float(np.max(np.abs(psi_plane))) or 1.0
    psi_plane_norm = psi_plane / plane_norm

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), constrained_layout=True)
    plots = [
        (np.abs(psi_plane_norm), "|Psi|, normalized", "magma", 0.0, 1.0),
        (np.real(psi_plane_norm), "Re Psi, normalized", "coolwarm", -1.0, 1.0),
        (np.imag(psi_plane_norm), "Im Psi, normalized", "coolwarm", -1.0, 1.0),
        (np.angle(psi_plane_norm), "phase(Psi)", "twilight", -np.pi, np.pi),
    ]
    for ax, (values, title, cmap, vmin, vmax) in zip(axes.flat, plots):
        image = ax.imshow(
            values,
            origin="lower",
            extent=[-extent, extent, -extent, extent],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("x / bohr")
        ax.set_ylabel("y / bohr")
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, shrink=0.84)
    fig.suptitle("Final wavefunction, z=0 slice", fontsize=13)
    slice_path = out_dir / "wavefunction_slice_z0.png"
    fig.savefig(slice_path, dpi=180)
    plt.close(fig)

    data_path = out_dir / "wavefunction_plot_data.npz"
    save_payload = {
        "r": rs,
        "psi_radial": psi_radial,
        "psi_radial_norm": psi_radial_norm,
        "x": xs,
        "y": ys,
        "psi_plane": psi_plane,
        "psi_plane_norm": psi_plane_norm,
    }
    if reference_hydrogen_1s:
        save_payload["hydrogen_1s"] = hydrogen_1s
    np.savez_compressed(data_path, **save_payload)

    outputs = {
        "radial_plot": radial_path,
        "slice_plot": slice_path,
        "plot_data": data_path,
    }
    if write_reports:
        outputs.update(write_radial_postprocess_reports(data_path, out_dir))
    return outputs


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
    parser.add_argument("--r-max", type=float, default=6.0, help="Radial plot max radius.")
    parser.add_argument("--radial-points", type=int, default=500)
    parser.add_argument("--plane-extent", type=float, default=4.0)
    parser.add_argument("--grid-size", type=int, default=181)
    parser.add_argument(
        "--no-phase-align",
        action="store_true",
        help="Do not remove the arbitrary global complex phase before plotting.",
    )
    parser.add_argument(
        "--reference-hydrogen-1s",
        action="store_true",
        help="Overlay exp(-r) as an optional H 1s reference curve.",
    )
    parser.add_argument(
        "--skip-radial-reports",
        action="store_true",
        help="Only write plot images/data; skip compact fit reports.",
    )
    args = parser.parse_args()

    outputs = generate_wavefunction_plots(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        r_max=args.r_max,
        radial_points=args.radial_points,
        plane_extent=args.plane_extent,
        grid_size=args.grid_size,
        phase_align=not args.no_phase_align,
        reference_hydrogen_1s=args.reference_hydrogen_1s,
        write_reports=not args.skip_radial_reports,
    )
    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
