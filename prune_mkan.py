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
        description="Prune low-attribution MKAN edges in an InsightQMC checkpoint."
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
        "--overwrite",
        action="store_true",
        help="Overwrite --out-checkpoint/--report if they already exist.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    checkpoint_path = (
        Path(args.checkpoint).expanduser()
        if args.checkpoint
        else run_dir / "checkpoints" / "last.pkl"
    )
    out_checkpoint = (
        Path(args.out_checkpoint).expanduser()
        if args.out_checkpoint
        else _default_out_checkpoint(run_dir, args.edge_threshold)
    )
    report_path = (
        Path(args.report).expanduser()
        if args.report
        else out_checkpoint.with_suffix(out_checkpoint.suffix + ".json")
    )

    if out_checkpoint.exists() and not args.overwrite:
        raise FileExistsError(
            f"{out_checkpoint} already exists. Pass --overwrite or choose another --out-checkpoint."
        )
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{report_path} already exists. Pass --overwrite or choose another --report."
        )

    cfg = _load_config(run_dir / "config.json")
    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_data = checkpoint["data"]
    spec = _mkan_width_and_params(cfg)
    model = _build_mkan(cfg, checkpoint)

    positions = _load_positions(
        Path(args.positions_file).expanduser() if args.positions_file else None,
        checkpoint_data,
    )
    sample_size = None if args.sample_size == 0 else args.sample_size
    features = _make_features(
        positions,
        checkpoint_data.atoms,
        spec["nelectrons"],
        spec["input_dim"],
        sample_size,
    )

    cache = model.get_act(features)
    attribution = model.attribute(cache)
    before_mask_stats = _mask_stats(model)
    edge_stats_before = _edge_score_stats(
        attribution["edge_scores"],
        args.edge_threshold,
    )

    model.prune_edge(threshold=args.edge_threshold, cache=cache)
    after_mask_stats = _mask_stats(model)

    _, mkan_params, mkan_static_state = nnx.split(model, nnx.Param, ...)
    new_checkpoint = dict(checkpoint)
    new_params = dict(checkpoint["params"])
    new_params["mkan"] = mkan_params
    new_checkpoint["params"] = new_params
    new_checkpoint["mkan_static_state"] = mkan_static_state
    if args.drop_opt_state:
        new_checkpoint["pretrain_opt_state"] = None
        new_checkpoint["train_opt_state"] = None

    _save_checkpoint(out_checkpoint, new_checkpoint, overwrite=args.overwrite)

    feature_score = np.asarray(attribution["feature_score"])
    feature_names = _feature_names(spec["natoms"], spec["input_dim"])
    ranking = np.argsort(-feature_score)

    report = {
        "source_checkpoint": str(checkpoint_path),
        "pruned_checkpoint": str(out_checkpoint),
        "source_stage": checkpoint.get("stage"),
        "source_step": checkpoint.get("step"),
        "layer_type": spec["layer_type"],
        "width": spec["width"],
        "features_shape": tuple(features.shape),
        "sample_size": None if args.sample_size == 0 else int(args.sample_size),
        "edge_threshold": float(args.edge_threshold),
        "drop_opt_state": bool(args.drop_opt_state),
        "edge_score_stats": edge_stats_before,
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
    print(f"Pruned edges: {total_pruned}/{total_edges} with threshold <= {args.edge_threshold:g}")
    for item in after_mask_stats:
        print(
            f"  layer {item['layer']}: mask zeros={item['zeros']}, "
            f"ones={item['ones']}, shape={item['shape']}"
        )
    print(f"Wrote pruned checkpoint: {out_checkpoint}")
    print(f"Wrote pruning report: {report_path}")


if __name__ == "__main__":
    main()
