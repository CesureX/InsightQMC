import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from flax import nnx

import networks
from jkan.models import MultKAN
from tools.utils import system


def _load_config(config_path: Path) -> ml_collections.ConfigDict:
    raw_cfg = json.loads(config_path.read_text())
    molecule = [
        system.Atom(
            atom["symbol"],
            atom["coords"],
            charge=atom["charge"],
            atomic_number=atom["atomic_number"],
            units=atom.get("units", "bohr"),
        )
        for atom in raw_cfg["system"]["molecule"]
    ]
    raw_cfg["system"]["molecule"] = molecule
    return ml_collections.ConfigDict(raw_cfg)


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    with checkpoint_path.open("rb") as handle:
        return pickle.load(handle)


def _load_positions(path: Path | None, checkpoint_data) -> jnp.ndarray:
    if path is None:
        return checkpoint_data.positions
    if path.suffix == ".npy":
        return jnp.array(np.load(path))
    return jnp.array(json.loads(path.read_text()))


def _first_int(values, default: int) -> int:
    if values is None:
        return default
    arr = np.asarray(values).reshape(-1)
    return int(arr[0]) if arr.size else default


def _first_grid_range(values, default=(-1.0, 1.0)) -> tuple[float, float]:
    if values is None:
        return tuple(default)
    arr = np.asarray(values)
    if arr.ndim == 1 and arr.size >= 2:
        return (float(arr[0]), float(arr[1]))
    if arr.ndim >= 2 and arr.shape[-1] >= 2:
        flat = arr.reshape(-1, arr.shape[-1])
        return (float(flat[0, 0]), float(flat[0, 1]))
    return tuple(default)


def _mkan_width_and_params(cfg: ml_collections.ConfigDict):
    molecule = cfg.system.molecule
    electrons = tuple(cfg.system.electrons)
    nelectrons = sum(electrons)
    nfeatures = int(cfg.nfeatures)

    mkan_cfg = cfg.get("mkan", {})
    layer_type = str(mkan_cfg.get("layer_type", "spline")).lower()
    mkan_input_dim = int(nfeatures if mkan_cfg.get("input_dim", None) is None else mkan_cfg.input_dim)
    output_default = (2 * nelectrons) if bool(cfg.complex_output) else nelectrons
    mkan_output_dim = int(output_default if mkan_cfg.get("output_dim", None) is None else mkan_cfg.output_dim)

    if mkan_cfg.get("width", None) is None:
        hidden_dims = [int(v) for v in np.asarray(cfg.layer_dims).reshape(-1)[1:-1]]
        width = [mkan_input_dim, *hidden_dims, mkan_output_dim]
    else:
        width = list(mkan_cfg.width)
        width[0] = mkan_input_dim
        width[-1] = mkan_output_dim

    required_parameters = mkan_cfg.get("required_parameters", None)
    if required_parameters is None:
        if layer_type in ("base", "spline"):
            required_parameters = {
                "k": _first_int(cfg.k, 3),
                "G": _first_int(cfg.g, 5),
                "grid_range": _first_grid_range(cfg.grid_range),
                "external_weights": bool(cfg.external_weights),
                "add_bias": bool(cfg.add_bias),
            }
        else:
            raise ValueError(
                "The current MKAN interpretation helpers support only "
                f"layer_type='base' or 'spline', got {layer_type!r}."
            )
    else:
        required_parameters = dict(required_parameters)

    return {
        "width": width,
        "layer_type": layer_type,
        "required_parameters": required_parameters,
        "mult_arity": mkan_cfg.get("mult_arity", 2),
        "seed": int(cfg.seed),
        "nelectrons": nelectrons,
        "natoms": len(molecule),
        "input_dim": mkan_input_dim,
    }


def _build_mkan(cfg: ml_collections.ConfigDict, checkpoint: dict[str, Any]) -> MultKAN:
    spec = _mkan_width_and_params(cfg)
    model_template = MultKAN(
        width=spec["width"],
        layer_type=spec["layer_type"],
        required_parameters=spec["required_parameters"],
        mult_arity=spec["mult_arity"],
        seed=spec["seed"],
    )
    graphdef, _, static_state = nnx.split(model_template, nnx.Param, ...)
    static_state = checkpoint.get("mkan_static_state") or static_state
    params = checkpoint["params"]
    mkan_params = params["mkan"] if isinstance(params, dict) and "mkan" in params else params
    return nnx.merge(graphdef, mkan_params, static_state)


def _feature_names(natoms: int, input_dim: int) -> list[str]:
    names = []
    for atom_idx in range(natoms):
        names.extend(
            [
                f"r_ae[{atom_idx}]",
                f"ae_x[{atom_idx}]",
                f"ae_y[{atom_idx}]",
                f"ae_z[{atom_idx}]",
            ]
        )
    if len(names) < input_dim:
        names.extend([f"x{i}" for i in range(len(names), input_dim)])
    return names[:input_dim]


def _make_features(positions, atoms, nelectrons: int, input_dim: int, sample_size: int | None):
    positions = jnp.reshape(positions, (-1, nelectrons * 3))

    def single_position_features(pos):
        ae, _, r_ae, _ = networks.construct_input_features(pos, atoms, ndim=3)
        return jnp.concatenate((r_ae, ae), axis=2).reshape(nelectrons, -1)

    features = jax.vmap(single_position_features)(positions)
    features = jnp.reshape(features, (-1, input_dim))
    if sample_size is not None:
        features = features[:sample_size]
    return features


def _jsonable(value: Any) -> Any:
    if isinstance(value, (jnp.ndarray, np.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpret a saved InsightQMC MKAN checkpoint.")
    parser.add_argument(
        "--run-dir",
        default="outputs/carbon_spinblock_test_001",
        help="Training output directory containing config.json and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path. Defaults to <run-dir>/checkpoints/last.pkl.",
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
        help="Maximum number of per-electron feature rows to use. Use 0 for all.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for feature_score.json and plots. Defaults to <run-dir>/mkan_interpretation.",
    )
    parser.add_argument(
        "--metric",
        default="backward",
        choices=("backward", "forward_n", "forward_u"),
        help="Edge strength metric used for plot opacity.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser() if args.checkpoint else run_dir / "checkpoints" / "last.pkl"
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else run_dir / "mkan_interpretation"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(run_dir / "config.json")
    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_data = checkpoint["data"]
    spec = _mkan_width_and_params(cfg)
    model = _build_mkan(cfg, checkpoint)

    positions = _load_positions(Path(args.positions_file).expanduser() if args.positions_file else None, checkpoint_data)
    atoms = checkpoint_data.atoms
    sample_size = None if args.sample_size == 0 else args.sample_size
    features = _make_features(positions, atoms, spec["nelectrons"], spec["input_dim"], sample_size)

    cache = model.get_act(features)
    attribution = model.attribute(cache)
    feature_score = np.asarray(attribution["feature_score"])
    feature_names = _feature_names(spec["natoms"], spec["input_dim"])
    ranking = np.argsort(-feature_score)

    summary = {
        "checkpoint": str(checkpoint_path),
        "stage": checkpoint.get("stage"),
        "step": checkpoint.get("step"),
        "layer_type": spec["layer_type"],
        "width": spec["width"],
        "features_shape": tuple(features.shape),
        "feature_score": {
            name: float(feature_score[idx]) for idx, name in enumerate(feature_names)
        },
        "feature_ranking": [
            {"feature": feature_names[idx], "score": float(feature_score[idx])}
            for idx in ranking
        ],
    }
    (out_dir / "feature_score.json").write_text(json.dumps(_jsonable(summary), indent=2))

    import matplotlib

    matplotlib.use("Agg")
    model.plot(cache, folder=str(out_dir / "edge_plots"), metric=args.metric, in_vars=feature_names)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Features shape: {features.shape}")
    print(f"Wrote feature scores: {out_dir / 'feature_score.json'}")
    print(f"Wrote edge plots: {out_dir / 'edge_plots'}")
    print("Top features:")
    for idx in ranking[: min(10, len(ranking))]:
        print(f"  {feature_names[idx]}: {feature_score[idx]:.6g}")


if __name__ == "__main__":
    main()
