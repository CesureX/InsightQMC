"""Prune MKAN edges or hidden nodes and write reusable artifacts.

Structural node pruning is intentionally limited to base, spline, and
Chebyshev layers because their parameter-subset transfer is fully supported.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx

from interpret_mkan import (
    _build_mkan,
    _feature_names,
    _jsonable,
    _load_checkpoint,
    _load_config,
    _load_positions,
    _make_features,
    _mkan_width_and_params,
)


def _threshold_slug(value: float) -> str:
    return f"{value:.3g}".replace("-", "m").replace("+", "").replace(".", "p")


def _default_out_checkpoint(run_dir: Path, edge_threshold: float) -> Path:
    slug = _threshold_slug(edge_threshold)
    return run_dir / "checkpoints" / f"edge_pruned_th_{slug}.pkl"


def _default_structural_run_dir(run_dir: Path, node_threshold: float) -> Path:
    slug = _threshold_slug(node_threshold)
    return run_dir.parent / f"{run_dir.name}_structural_node_th_{slug}"


def _model_width_to_config(model) -> list[Any]:
    width = []
    for n_sum, n_mult in model.width:
        n_sum = int(n_sum)
        n_mult = int(n_mult)
        width.append(n_sum if n_mult == 0 else [n_sum, n_mult])
    return width


def _plain_layer_dims(width: list[Any]) -> list[int]:
    dims = []
    for item in width:
        if isinstance(item, (list, tuple)):
            dims.append(int(item[0]) + int(item[1]))
        else:
            dims.append(int(item))
    return dims


def _model_grid_state(model) -> dict[str, Any] | None:
    """Serialize grid metadata with the pruned layer dimensions."""

    layer_states = []
    has_grid = False
    for layer in model.layers:
        grid = getattr(layer, "grid", None)
        if grid is None:
            layer_states.append(None)
            continue
        has_grid = True
        layer_states.append(
            {
                "G": int(grid.G),
                "grid_range": tuple(float(value) for value in grid.grid_range),
                "grid_e": float(grid.grid_e),
                "item": jnp.asarray(grid.item),
            }
        )
    return {"layers": layer_states} if has_grid else None


def _config_jsonable(value: Any) -> Any:
    if isinstance(value, (jnp.ndarray, np.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "symbol") and hasattr(value, "coords"):
        return {
            "symbol": value.symbol,
            "coords": list(value.coords),
            "charge": float(value.charge),
            "atomic_number": int(value.atomic_number),
            "units": getattr(value, "units", "bohr"),
        }
    if hasattr(value, "items"):
        return {str(key): _config_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_config_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_structural_config(
    path: Path,
    cfg,
    *,
    out_run_dir: Path,
    pruned_width: list[Any],
    pruned_mult_arity: Any,
    resume: bool,
    iterations: int | None,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass --overwrite or choose another config path."
        )
    config = _config_jsonable(cfg)
    config["layer_dims"] = _plain_layer_dims(pruned_width)
    config.setdefault("mkan", {})
    config["mkan"]["width"] = pruned_width
    config["mkan"]["mult_arity"] = _config_jsonable(pruned_mult_arity)
    config["mkan"]["prune_mask_checkpoint"] = None
    config.setdefault("output", {})
    config["output"]["root_dir"] = str(out_run_dir)
    config["output"]["resume"] = bool(resume)
    if iterations is not None:
        config["iterations"] = int(iterations)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(config), indent=2, sort_keys=True))


def _edge_score_stats(edge_scores, edge_threshold: float) -> list[dict[str, Any]]:
    stats = []
    for layer_idx, scores in enumerate(edge_scores):
        arr = np.asarray(scores)
        prune_mask = arr <= edge_threshold
        stats.append(
            {
                "layer": layer_idx,
                "shape": tuple(arr.shape),
                "threshold": float(edge_threshold),
                "total_edges": int(arr.size),
                "pruned_edges": int(np.sum(prune_mask)),
                "kept_edges": int(arr.size - np.sum(prune_mask)),
                "min": float(np.min(arr)) if arr.size else None,
                "max": float(np.max(arr)) if arr.size else None,
                "mean": float(np.mean(arr)) if arr.size else None,
                "median": float(np.median(arr)) if arr.size else None,
            }
        )
    return stats


def _node_score_stats(node_scores, node_threshold: float) -> list[dict[str, Any]]:
    stats = []
    for width_idx, scores in enumerate(node_scores):
        arr = np.asarray(scores)
        keep_mask = arr > node_threshold
        if arr.size and not np.any(keep_mask):
            # Match _active_hidden_nodes_from_scores: never remove an entire
            # hidden layer when every score is below the threshold.
            keep_mask[int(np.argmax(arr))] = True
        prune_mask = ~keep_mask
        stats.append(
            {
                "width_index": width_idx,
                "threshold": float(node_threshold),
                "total_nodes": int(arr.size),
                "pruned_nodes": int(np.sum(prune_mask)),
                "kept_nodes": int(arr.size - np.sum(prune_mask)),
                "min": float(np.min(arr)) if arr.size else None,
                "max": float(np.max(arr)) if arr.size else None,
                "mean": float(np.mean(arr)) if arr.size else None,
                "median": float(np.median(arr)) if arr.size else None,
            }
        )
    return stats


def _parse_keep_fractions(value: str | None) -> list[float] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return None
    fractions = []
    for part in parts:
        if part.endswith("%"):
            fraction = float(part[:-1]) / 100.0
        else:
            fraction = float(part)
            if fraction > 1.0:
                fraction /= 100.0
        if not (0.0 < fraction <= 1.0):
            raise ValueError(
                f"Keep fractions must be in (0, 1] or percentages, got {part!r}."
            )
        fractions.append(fraction)
    return fractions


def _active_hidden_nodes_from_keep_fractions(
    node_scores,
    keep_fractions: list[float],
) -> tuple[list[list[int]], list[dict[str, Any]], list[float | None]]:
    if len(keep_fractions) == 1:
        keep_fractions = keep_fractions * len(node_scores)
    if len(keep_fractions) != len(node_scores):
        raise ValueError(
            "--node-keep-fractions must provide one value, or one value per "
            f"hidden layer ({len(node_scores)} here)."
        )

    active = []
    stats = []
    cutoffs: list[float | None] = []
    for width_idx, (scores, fraction) in enumerate(zip(node_scores, keep_fractions)):
        arr = np.asarray(scores)
        if arr.size == 0:
            active.append([])
            cutoffs.append(None)
            stats.append(
                {
                    "width_index": width_idx,
                    "keep_fraction": float(fraction),
                    "total_nodes": 0,
                    "pruned_nodes": 0,
                    "kept_nodes": 0,
                    "cutoff_score": None,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "median": None,
                }
            )
            continue

        keep_count = max(1, min(arr.size, int(np.ceil(arr.size * fraction))))
        order = np.argsort(-arr)
        kept = np.sort(order[:keep_count]).astype(int).tolist()
        cutoff = float(arr[order[keep_count - 1]])
        active.append(kept)
        cutoffs.append(cutoff)
        stats.append(
            {
                "width_index": width_idx,
                "keep_fraction": float(fraction),
                "total_nodes": int(arr.size),
                "pruned_nodes": int(arr.size - keep_count),
                "kept_nodes": int(keep_count),
                "cutoff_score": cutoff,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
            }
        )
    return active, stats, cutoffs


def _active_hidden_nodes_from_scores(node_scores, node_threshold: float) -> list[list[int]]:
    active = []
    for scores in node_scores:
        arr = np.asarray(scores)
        ids = np.where(arr > node_threshold)[0].astype(int).tolist()
        if not ids and arr.size:
            ids = [int(np.argmax(arr))]
        active.append(ids)
    return active


def _width_count(item: Any) -> int:
    if isinstance(item, (list, tuple)):
        return int(item[0]) + int(item[1])
    return int(item)


def _plot_pruned_structure(
    path: Path,
    *,
    source_width: list[Any],
    active_hidden_nodes: list[list[int]],
    hidden_node_scores,
    node_threshold: float | None,
    layer_thresholds: list[float | None] | None = None,
    title: str,
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib unavailable, skipping structure plot: {exc}")
        return None

    source_counts = [_width_count(item) for item in source_width]
    active_by_layer: list[list[int]] = [list(range(source_counts[0]))]
    for layer_idx, count in enumerate(source_counts[1:-1]):
        if layer_idx < len(active_hidden_nodes):
            active_by_layer.append(active_hidden_nodes[layer_idx])
        else:
            active_by_layer.append(list(range(count)))
    active_by_layer.append(list(range(source_counts[-1])))
    pruned_counts = [len(ids) for ids in active_by_layer]

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0])
    ax_net = fig.add_subplot(grid[0, 0])
    ax_score = fig.add_subplot(grid[1, 0])

    x_positions = np.arange(len(source_counts))
    for layer_idx, (x_pos, count, active_ids) in enumerate(
        zip(x_positions, source_counts, active_by_layer)
    ):
        y = np.linspace(0.0, 1.0, count) if count > 1 else np.array([0.5])
        active_set = set(map(int, active_ids))
        kept = np.array([idx in active_set for idx in range(count)])
        size = max(8.0, min(28.0, 900.0 / max(count, 1)))
        ax_net.scatter(
            np.full(np.sum(kept), x_pos),
            y[kept],
            s=size,
            color="#2563eb",
            alpha=0.9,
            zorder=3,
        )
        if np.any(~kept):
            ax_net.scatter(
                np.full(np.sum(~kept), x_pos),
                y[~kept],
                s=size * 1.4,
                color="#dc2626",
                marker="x",
                linewidths=0.9,
                alpha=0.9,
                zorder=4,
            )
        ax_net.text(
            x_pos,
            -0.11,
            f"L{layer_idx}\n{count}->{len(active_ids)}",
            ha="center",
            va="top",
            fontsize=10,
        )

    for left, right in zip(x_positions[:-1], x_positions[1:]):
        ax_net.plot([left, right], [0.5, 0.5], color="#94a3b8", linewidth=6, alpha=0.18)

    ax_net.set_xlim(-0.5, len(source_counts) - 0.5)
    ax_net.set_ylim(-0.2, 1.08)
    ax_net.set_xticks([])
    ax_net.set_yticks([])
    ax_net.set_title(
        f"{title}: {source_counts} -> {pruned_counts}",
        fontsize=13,
        pad=10,
    )
    for spine in ax_net.spines.values():
        spine.set_visible(False)

    colors = ["#2563eb", "#16a34a", "#f97316", "#9333ea"]
    for layer_idx, scores in enumerate(hidden_node_scores):
        arr = np.asarray(scores)
        if arr.size == 0:
            continue
        x = layer_idx + np.linspace(-0.33, 0.33, arr.size)
        active_set = set(active_hidden_nodes[layer_idx])
        kept = np.array([idx in active_set for idx in range(arr.size)])
        ax_score.scatter(
            x[kept],
            arr[kept],
            s=18,
            color=colors[layer_idx % len(colors)],
            alpha=0.85,
            label=f"hidden {layer_idx + 1} kept",
        )
        if np.any(~kept):
            ax_score.scatter(
                x[~kept],
                arr[~kept],
                s=24,
                color="#dc2626",
                marker="x",
                linewidths=0.9,
                alpha=0.9,
                label=f"hidden {layer_idx + 1} pruned",
            )
    if layer_thresholds is not None:
        for layer_idx, threshold in enumerate(layer_thresholds):
            if threshold is None:
                continue
            ax_score.hlines(
                threshold,
                layer_idx - 0.4,
                layer_idx + 0.4,
                color="#111827",
                linestyle="--",
                linewidth=1.2,
                label="layer cutoff" if layer_idx == 0 else None,
            )
    elif node_threshold is not None:
        ax_score.axhline(
            node_threshold,
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            label=f"threshold {node_threshold:g}",
        )
    ax_score.set_xticks(np.arange(len(hidden_node_scores)))
    ax_score.set_xticklabels([f"hidden {idx + 1}" for idx in range(len(hidden_node_scores))])
    ax_score.set_ylabel("node score")
    ax_score.grid(True, alpha=0.25)
    handles, labels = ax_score.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_score.legend(unique.values(), unique.keys(), fontsize=8, ncol=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _mask_stats(model) -> list[dict[str, Any]]:
    stats = []
    for layer_idx, layer in enumerate(model.layers):
        if not hasattr(layer, "edge_mask"):
            continue
        mask = np.asarray(layer.edge_mask[...])
        stats.append(
            {
                "layer": layer_idx,
                "shape": tuple(mask.shape),
                "zeros": int(np.sum(mask == 0)),
                "ones": int(np.sum(mask == 1)),
                "min": float(np.min(mask)) if mask.size else None,
                "max": float(np.max(mask)) if mask.size else None,
            }
        )
    return stats


def _save_checkpoint(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune low-attribution MKAN edges or hidden nodes in an InsightQMC checkpoint."
    )
    parser.add_argument(
        "--run-dir",
        default="outputs/carbon_spinblock_test_001",
        help="Training output directory containing config.json and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Source checkpoint. Defaults to <run-dir>/checkpoints/last.pkl.",
    )
    parser.add_argument(
        "--positions-file",
        default=None,
        help="Optional JSON or .npy positions file. Defaults to checkpoint positions.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2048,
        help="Maximum number of per-electron feature rows used for attribution. Use 0 for all.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=3e-2,
        help="Prune edges with backward attribution score <= this value.",
    )
    parser.add_argument(
        "--node-threshold",
        type=float,
        default=1e-2,
        help=(
            "For structural pruning, keep hidden nodes with attribution score > "
            "this value. Ignored when --node-keep-fractions is set."
        ),
    )
    parser.add_argument(
        "--node-keep-fractions",
        default=None,
        help=(
            "Comma-separated per-hidden-layer keep fractions for node pruning, "
            "for example 0.7,0.5,0.3. Percent values like 70,50,30 or 70%% "
            "are also accepted. Overrides --node-threshold."
        ),
    )
    parser.add_argument(
        "--prune-mode",
        choices=("edge", "node", "node-edge"),
        default="edge",
        help=(
            "edge: only zero low-score edges with masks; node: build a smaller "
            "network by removing low-score hidden nodes; node-edge: structurally "
            "prune nodes, then mask low-score edges in the smaller network."
        ),
    )
    parser.add_argument(
        "--out-run-dir",
        default=None,
        help=(
            "Output run directory. Defaults to a sibling structural-prune directory "
            "for node/node-edge mode. If set, the checkpoint defaults to "
            "<out-run-dir>/checkpoints/last.pkl and the report to "
            "<out-run-dir>/pruning_report.json."
        ),
    )
    parser.add_argument(
        "--out-checkpoint",
        default=None,
        help=(
            "Path for the pruned checkpoint. Defaults to "
            "<run-dir>/checkpoints/edge_pruned_th_<threshold>.pkl."
        ),
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path for pruning_report.json. Defaults to <out-checkpoint>.json.",
    )
    parser.add_argument(
        "--drop-opt-state",
        action="store_true",
        help="Drop pretrain/train optimizer state in the pruned checkpoint.",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help=(
            "Write a config.json next to the pruned checkpoint with mkan.width set "
            "to the new structural width."
        ),
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help=(
            "Path for pruned_structure.png. Defaults to <out-run-dir>/"
            "pruned_structure.png for structural pruning, or next to the report."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the pruning structure plot.",
    )
    parser.add_argument(
        "--config-resume",
        action="store_true",
        help=(
            "When writing config.json, set output.resume=True so training starts "
            "from the pruned checkpoint. By default the written config starts a "
            "fresh run with the new architecture."
        ),
    )
    parser.add_argument(
        "--config-iterations",
        type=int,
        default=None,
        help="Optional iterations value to write into the generated config.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --out-checkpoint/--report if they already exist.",
    )
    args = parser.parse_args()

    keep_fractions = _parse_keep_fractions(args.node_keep_fractions)
    if keep_fractions is not None and args.prune_mode == "edge":
        raise ValueError("--node-keep-fractions requires --prune-mode node or node-edge.")

    run_dir = Path(args.run_dir).expanduser()
    checkpoint_path = (
        Path(args.checkpoint).expanduser()
        if args.checkpoint
        else run_dir / "checkpoints" / "last.pkl"
    )
    out_run_dir = (
        Path(args.out_run_dir).expanduser()
        if args.out_run_dir
        else (
            _default_structural_run_dir(run_dir, args.node_threshold)
            if args.prune_mode in ("node", "node-edge")
            else None
        )
    )
    if args.out_checkpoint:
        out_checkpoint = Path(args.out_checkpoint).expanduser()
    elif out_run_dir is not None:
        out_checkpoint = out_run_dir / "checkpoints" / "last.pkl"
    else:
        out_checkpoint = _default_out_checkpoint(run_dir, args.edge_threshold)
    report_path = (
        Path(args.report).expanduser()
        if args.report
        else (
            out_run_dir / "pruning_report.json"
            if out_run_dir is not None
            else out_checkpoint.with_suffix(out_checkpoint.suffix + ".json")
        )
    )

    if out_checkpoint.exists() and not args.overwrite:
        raise FileExistsError(
            f"{out_checkpoint} already exists. Pass --overwrite or choose another --out-checkpoint."
        )
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{report_path} already exists. Pass --overwrite or choose another --report."
        )
    plot_path = None
    if not args.no_plot:
        plot_path = (
            Path(args.plot_path).expanduser()
            if args.plot_path
            else (
                out_run_dir / "pruned_structure.png"
                if out_run_dir is not None
                else report_path.with_name("pruned_structure.png")
            )
        )
        if plot_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{plot_path} already exists. Pass --overwrite or choose another --plot-path."
            )

    cfg = _load_config(run_dir / "config.json")
    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_data = checkpoint["data"]
    spec = _mkan_width_and_params(cfg)
    if spec["layer_type"] not in {"base", "spline", "chebyshev"}:
        raise NotImplementedError(
            "Attribution-based pruning currently supports only base, spline, and "
            f"chebyshev layers; got {spec['layer_type']!r}."
        )
    model = _build_mkan(cfg, checkpoint)

    positions = _load_positions(
        Path(args.positions_file).expanduser() if args.positions_file else None,
        checkpoint_data,
    )
    sample_size = None if args.sample_size == 0 else args.sample_size
    features = _make_features(
        positions,
        checkpoint_data.atoms,
        checkpoint_data.charges,
        spec["electrons"],
        spec["input_dim"],
        sample_size,
        spec["orbital_feature_mode"],
        spins=checkpoint_data.spins,
    )

    cache = model.get_act(features)
    attribution = model.attribute(cache)
    before_mask_stats = _mask_stats(model)
    hidden_node_scores = attribution["node_scores"][1:-1]
    layer_score_cutoffs = None
    if keep_fractions is None:
        node_selection_mode = "threshold"
        active_hidden_nodes = _active_hidden_nodes_from_scores(
            hidden_node_scores,
            args.node_threshold,
        )
        node_stats_before = _node_score_stats(
            hidden_node_scores,
            args.node_threshold,
        )
    else:
        node_selection_mode = "keep_fraction"
        active_hidden_nodes, node_stats_before, layer_score_cutoffs = (
            _active_hidden_nodes_from_keep_fractions(
                hidden_node_scores,
                keep_fractions,
            )
        )
    edge_stats_before = _edge_score_stats(
        attribution["edge_scores"],
        args.edge_threshold,
    )

    if args.prune_mode == "edge":
        model.prune_edge(threshold=args.edge_threshold, cache=cache)
        edge_stats_after_node = None
    elif args.prune_mode == "node":
        if keep_fractions is None:
            model = model.prune_node(threshold=args.node_threshold, cache=cache)
        else:
            model = model.prune_node(
                active_neurons_id=active_hidden_nodes,
                cache=cache,
            )
        edge_stats_after_node = None
    else:
        if keep_fractions is None:
            model = model.prune_node(threshold=args.node_threshold, cache=cache)
        else:
            model = model.prune_node(
                active_neurons_id=active_hidden_nodes,
                cache=cache,
            )
        node_cache = model.get_act(features)
        node_attribution = model.attribute(node_cache)
        edge_stats_after_node = _edge_score_stats(
            node_attribution["edge_scores"],
            args.edge_threshold,
        )
        model.prune_edge(threshold=args.edge_threshold, cache=node_cache)
    after_mask_stats = _mask_stats(model)
    pruned_width = _model_width_to_config(model)
    pruned_mult_arity = _config_jsonable(model.mult_arity)

    _, mkan_params, mkan_static_state = nnx.split(model, nnx.Param, ...)
    new_checkpoint = dict(checkpoint)
    new_params = dict(checkpoint["params"])
    new_params["mkan"] = mkan_params
    new_checkpoint["params"] = new_params
    new_checkpoint["mkan_static_state"] = mkan_static_state
    new_checkpoint["mkan_grid_state"] = _model_grid_state(model)
    new_checkpoint["mkan_pruned_width"] = pruned_width
    new_checkpoint["mkan_pruned_mult_arity"] = pruned_mult_arity
    drop_opt_state = bool(args.drop_opt_state or args.prune_mode in ("node", "node-edge"))
    if drop_opt_state:
        new_checkpoint["pretrain_opt_state"] = None
        new_checkpoint["train_opt_state"] = None

    _save_checkpoint(out_checkpoint, new_checkpoint, overwrite=args.overwrite)

    config_path = None
    if args.write_config or out_run_dir is not None:
        config_run_dir = out_run_dir if out_run_dir is not None else out_checkpoint.parent.parent
        config_path = config_run_dir / "config.json"
        _write_structural_config(
            config_path,
            cfg,
            out_run_dir=config_run_dir,
            pruned_width=pruned_width,
            pruned_mult_arity=pruned_mult_arity,
            resume=bool(args.config_resume),
            iterations=args.config_iterations,
            overwrite=args.overwrite,
        )

    written_plot = None
    if plot_path is not None:
        written_plot = _plot_pruned_structure(
            plot_path,
            source_width=spec["width"],
            active_hidden_nodes=(
                active_hidden_nodes
                if args.prune_mode in ("node", "node-edge")
                else [
                    list(range(_width_count(item)))
                    for item in spec["width"][1:-1]
                ]
            ),
            hidden_node_scores=hidden_node_scores,
            node_threshold=None if keep_fractions is not None else args.node_threshold,
            layer_thresholds=layer_score_cutoffs,
            title=f"{run_dir.name} {args.prune_mode} prune",
        )

    feature_score = np.asarray(attribution["feature_score"])
    feature_names = _feature_names(
        spec["natoms"],
        spec["input_dim"],
        spec["orbital_feature_mode"],
    )
    ranking = np.argsort(-feature_score)

    report = {
        "source_checkpoint": str(checkpoint_path),
        "pruned_checkpoint": str(out_checkpoint),
        "source_stage": checkpoint.get("stage"),
        "source_step": checkpoint.get("step"),
        "prune_mode": args.prune_mode,
        "node_selection_mode": node_selection_mode,
        "layer_type": spec["layer_type"],
        "source_width": spec["width"],
        "pruned_width": pruned_width,
        "pruned_mult_arity": pruned_mult_arity,
        "features_shape": tuple(features.shape),
        "sample_size": None if args.sample_size == 0 else int(args.sample_size),
        "node_threshold": float(args.node_threshold),
        "node_keep_fractions": keep_fractions,
        "edge_threshold": float(args.edge_threshold),
        "drop_opt_state": drop_opt_state,
        "written_config": None if config_path is None else str(config_path),
        "written_plot": written_plot,
        "config_resume": bool(args.config_resume),
        "edge_score_stats": edge_stats_before,
        "edge_score_stats_after_node_prune": edge_stats_after_node,
        "node_score_stats": node_stats_before,
        "mask_stats_before": before_mask_stats,
        "mask_stats_after": after_mask_stats,
        "feature_ranking": [
            {"feature": feature_names[idx], "score": float(feature_score[idx])}
            for idx in ranking
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_jsonable(report), indent=2))

    total_pruned = sum(item["pruned_edges"] for item in edge_stats_before)
    total_edges = sum(item["total_edges"] for item in edge_stats_before)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Features shape: {tuple(features.shape)}")
    print(f"Prune mode: {args.prune_mode}")
    print(f"Source width: {spec['width']}")
    print(f"Pruned width: {pruned_width}")
    if args.prune_mode == "edge":
        print(f"Pruned edges: {total_pruned}/{total_edges} with threshold <= {args.edge_threshold:g}")
    else:
        total_node_pruned = sum(item["pruned_nodes"] for item in node_stats_before)
        total_nodes = sum(item["total_nodes"] for item in node_stats_before)
        if keep_fractions is None:
            print(
                f"Structurally pruned hidden nodes: {total_node_pruned}/{total_nodes} "
                f"with threshold <= {args.node_threshold:g}"
            )
        else:
            print(
                f"Structurally pruned hidden nodes: {total_node_pruned}/{total_nodes} "
                f"with keep fractions {','.join(f'{x:g}' for x in keep_fractions)}"
            )
        if edge_stats_after_node is not None:
            total_pruned_after_node = sum(item["pruned_edges"] for item in edge_stats_after_node)
            total_edges_after_node = sum(item["total_edges"] for item in edge_stats_after_node)
            print(
                f"Masked edges after node pruning: {total_pruned_after_node}/{total_edges_after_node} "
                f"with threshold <= {args.edge_threshold:g}"
            )
    for item in after_mask_stats:
        print(
            f"  layer {item['layer']}: mask zeros={item['zeros']}, "
            f"ones={item['ones']}, shape={item['shape']}"
        )
    print(f"Wrote pruned checkpoint: {out_checkpoint}")
    print(f"Wrote pruning report: {report_path}")
    if config_path is not None:
        print(f"Wrote structural config: {config_path}")
    if written_plot is not None:
        print(f"Wrote pruning structure plot: {written_plot}")


if __name__ == "__main__":
    main()
