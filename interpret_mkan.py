#  python interpret_mkan.py --run-dir outputs/H6180813

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

import envelope
import networks
from jkan.models import MultKAN
from tools.utils import system

'''
python interpret_mkan.py  --run-dir /vepfs-mlp2/c20250516/250504030/jing/InsightQMC/outputs/C_LZW6022246
'''

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


def _mkan_width_output_dim(width) -> int:
    last = list(width)[-1]
    if isinstance(last, (list, tuple)):
        return int(last[0]) + int(last[1])
    return int(last)


def _merge_static_state_with_template(template_state, checkpoint_state):
    if checkpoint_state is None:
        return template_state
    try:
        merged = dict(template_state.flat_state())
        checkpoint_flat = dict(checkpoint_state.flat_state())
    except AttributeError:
        return checkpoint_state
    for path, value in checkpoint_flat.items():
        if path in merged:
            merged[path] = value
    return nnx.State.from_flat_path(merged)


def _mkan_width_and_params(cfg: ml_collections.ConfigDict):
    molecule = cfg.system.molecule
    electrons = tuple(cfg.system.electrons)
    nelectrons = sum(electrons)
    ndeterminants = int(cfg.get("ndeterminants", 1))
    nfeatures = int(cfg.nfeatures)

    mkan_cfg = cfg.get("mkan", {})
    layer_type = str(mkan_cfg.get("layer_type", "spline")).lower()
    orbital_feature_mode = str(cfg.get("orbital_features", "one_body")).lower()
    mkan_input_dim = int(nfeatures if mkan_cfg.get("input_dim", None) is None else mkan_cfg.input_dim)
    orbital_output_dim = (
        (2 * ndeterminants * nelectrons)
        if bool(cfg.complex_output)
        else (ndeterminants * nelectrons)
    )
    orbital_head_cfg = mkan_cfg.get("orbital_head", cfg.get("orbital_head", {}))
    orbital_head_enabled = bool(orbital_head_cfg.get("enabled", False))
    orbital_head_type = str(orbital_head_cfg.get("type", "dense")).lower()
    if orbital_head_enabled:
        if mkan_cfg.get("output_dim", None) is None:
            if mkan_cfg.get("width", None) is not None:
                mkan_output_dim = _mkan_width_output_dim(mkan_cfg.width)
            else:
                layer_dims = np.asarray(cfg.layer_dims).reshape(-1)
                mkan_output_dim = int(
                    layer_dims[-1] if layer_dims.size else orbital_output_dim
                )
        else:
            mkan_output_dim = int(mkan_cfg.output_dim)
    else:
        mkan_output_dim = int(
            orbital_output_dim if mkan_cfg.get("output_dim", None) is None else mkan_cfg.output_dim
        )

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
        elif layer_type in ("chebyshev", "legendre"):
            required_parameters = {
                "D": _first_int(cfg.k, 3),
                "flavor": "exact" if layer_type == "chebyshev" else None,
                "external_weights": bool(cfg.external_weights),
                "add_bias": bool(cfg.add_bias),
            }
        elif layer_type == "rbf":
            required_parameters = {
                "D": _first_int(cfg.k, 5),
                "grid_range": _first_grid_range(cfg.grid_range, default=(-2.0, 2.0)),
                "external_weights": bool(cfg.external_weights),
                "add_bias": bool(cfg.add_bias),
            }
        elif layer_type == "fastkan":
            required_parameters = {
                "D": _first_int(cfg.g, 8),
                "grid_range": _first_grid_range(cfg.grid_range, default=(-2.0, 2.0)),
                "add_bias": bool(cfg.add_bias),
            }
        elif layer_type == "relukan":
            required_parameters = {
                "G": _first_int(cfg.g, 5),
                "k": _first_int(cfg.k, 3),
                "add_bias": bool(cfg.add_bias),
            }
        elif layer_type == "wavkan":
            required_parameters = {
                "wavelet_type": "mexican_hat",
                "add_bias": bool(cfg.add_bias),
            }
        elif layer_type == "sine":
            required_parameters = {
                "D": _first_int(cfg.k, 5),
                "external_weights": bool(cfg.external_weights),
                "add_bias": bool(cfg.add_bias),
            }
        elif layer_type == "fourier":
            required_parameters = {
                "D": _first_int(cfg.k, 5),
                "add_bias": bool(cfg.add_bias),
            }
        else:
            raise ValueError(f"Unsupported MKAN layer_type={layer_type!r}.")
    else:
        required_parameters = dict(required_parameters)

    return {
        "width": width,
        "layer_type": layer_type,
        "required_parameters": required_parameters,
        "mult_arity": mkan_cfg.get("mult_arity", 2),
        "seed": int(cfg.seed),
        "nelectrons": nelectrons,
        "electrons": electrons,
        "natoms": len(molecule),
        "input_dim": mkan_input_dim,
        "output_dim": mkan_output_dim,
        "orbital_output_dim": orbital_output_dim,
        "orbital_head_enabled": orbital_head_enabled,
        "orbital_head_type": orbital_head_type,
        "orbital_feature_mode": orbital_feature_mode,
        "ndeterminants": ndeterminants,
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
    static_state = _merge_static_state_with_template(
        static_state,
        checkpoint.get("mkan_static_state"),
    )
    params = checkpoint["params"]
    mkan_params = params["mkan"] if isinstance(params, dict) and "mkan" in params else params
    return nnx.merge(graphdef, mkan_params, static_state)


def _feature_names(natoms: int, input_dim: int, feature_mode: str = "one_body") -> list[str]:
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
    if str(feature_mode).lower() in ("ee_aggregate", "ee_agg", "equivariant_ee"):
        names.extend(["ee_density", "ee_vec_x", "ee_vec_y", "ee_vec_z"])
    if len(names) < input_dim:
        names.extend([f"x{i}" for i in range(len(names), input_dim)])
    return names[:input_dim]


def _make_features(
    positions,
    atoms,
    electrons,
    input_dim: int,
    sample_size: int | None,
    feature_mode: str = "one_body",
):
    nelectrons = sum(electrons)
    positions = jnp.reshape(positions, (-1, nelectrons * 3))

    def single_position_features(pos):
        return networks.construct_orbital_features(
            pos,
            atoms,
            ndim=3,
            feature_mode=feature_mode,
        )

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


def _callable_name(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "__name__", type(value).__name__)


def _format_float(value: Any, precision: int = 8) -> str:
    return f"{float(np.asarray(value)):.{precision}g}"


def _format_hard_float(value: Any, precision: int) -> str:
    return _format_float(value, precision=max(1, int(precision)))


def _hard_residual_text(residual_name: str | None, arg: str) -> str:
    if residual_name is None:
        return "0"
    name = residual_name.lower()
    if name in ("silu", "swish"):
        return f"(({arg})/(1+exp(-({arg}))))"
    if name in ("sigmoid", "logistic"):
        return f"(1/(1+exp(-({arg}))))"
    if name == "relu":
        return f"max(0, {arg})"
    if name == "gelu":
        return f"0.5*({arg})*(1+tanh(sqrt(2/pi)*(({arg})+0.044715*({arg})^3)))"
    if name == "identity":
        return f"({arg})"
    return f"{residual_name}({arg})"


def _chebyshev_expr(order: int, arg: str) -> str:
    if order <= 0:
        return "1"
    if order == 1:
        return f"({arg})"
    prev2 = "1"
    prev1 = f"({arg})"
    for _ in range(2, order + 1):
        current = f"(2*({arg})*({prev1}) - ({prev2}))"
        prev2, prev1 = prev1, current
    return prev1


def _legendre_expr(order: int, arg: str) -> str:
    if order <= 0:
        return "1"
    if order == 1:
        return f"({arg})"
    prev2 = "1"
    prev1 = f"({arg})"
    for n in range(2, order + 1):
        current = f"(((2*{n}-1)*({arg})*({prev1}) - ({n - 1})*({prev2}))/{n})"
        prev2, prev1 = prev1, current
    return prev1


def _real_spherical_harmonic_symbol(index: int) -> str:
    """Return the real spherical-harmonic label for packed index."""

    cursor = 0
    for l in range(128):
        width = 2 * l + 1
        if index < cursor + width:
            m = -l + (index - cursor)
            return f"Yreal_{l}_{m}"
        cursor += width
    return f"Yreal_index_{index}"


def _bspline_basis_expr(
    basis_idx: int,
    degree: int,
    arg: str,
    grid: np.ndarray,
    precision: int,
) -> str:
    knots = np.asarray(grid, dtype=np.float64).reshape(-1)
    memo: dict[tuple[int, int], str] = {}

    def rec(j: int, k: int) -> str:
        key = (j, k)
        if key in memo:
            return memo[key]
        if j < 0 or j + k + 1 >= knots.size:
            memo[key] = "0"
            return memo[key]
        if k == 0:
            left = _format_hard_float(knots[j], precision)
            right = _format_hard_float(knots[j + 1], precision)
            memo[key] = f"I({left} <= ({arg}) < {right})"
            return memo[key]

        terms = []
        left_den = knots[j + k] - knots[j]
        if abs(float(left_den)) > 1.0e-14:
            left = _format_hard_float(knots[j], precision)
            den = _format_hard_float(left_den, precision)
            terms.append(f"((({arg}) - ({left}))/({den}))*({rec(j, k - 1)})")

        right_den = knots[j + k + 1] - knots[j + 1]
        if abs(float(right_den)) > 1.0e-14:
            right = _format_hard_float(knots[j + k + 1], precision)
            den = _format_hard_float(right_den, precision)
            terms.append(f"(({right} - ({arg}))/({den}))*({rec(j + 1, k - 1)})")

        memo[key] = _sum_terms(terms)
        return memo[key]

    return rec(int(basis_idx), int(degree))


def _maybe_param_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value[...])
    except TypeError:
        return np.asarray(value)


def _edge_mask_value(layer, out_idx: int, in_idx: int) -> float:
    if not hasattr(layer, "edge_mask"):
        return 1.0
    mask = np.asarray(layer.edge_mask[...])
    return float(mask[int(out_idx), int(in_idx)])


def _edge_coefficients(layer, layer_type: str, out_idx: int, in_idx: int) -> np.ndarray:
    c_basis = np.asarray(layer.c_basis[...])
    mask = _edge_mask_value(layer, out_idx, in_idx)
    if layer_type == "base":
        flat_idx = out_idx * int(layer.n_in) + in_idx
        coeffs = c_basis[flat_idx]
        if layer.c_spl is not None:
            coeffs = coeffs * float(np.asarray(layer.c_spl[...])[out_idx, in_idx])
        return coeffs * mask
    if layer_type == "spline":
        coeffs = c_basis[out_idx, in_idx]
        if layer.c_spl is not None:
            coeffs = coeffs * float(np.asarray(layer.c_spl[...])[out_idx, in_idx])
        return coeffs * mask
    if layer_type in ("chebyshev", "legendre", "rbf", "sine"):
        coeffs = c_basis[out_idx, in_idx]
        c_ext = _maybe_param_array(getattr(layer, "c_ext", None))
        if c_ext is not None:
            coeffs = coeffs * float(c_ext[out_idx, in_idx])
        return coeffs * mask
    raise ValueError(f"Unsupported expression export for layer_type={layer_type!r}.")


def _edge_grid(layer, layer_type: str, out_idx: int, in_idx: int) -> np.ndarray:
    grid = np.asarray(layer.grid.item)
    if layer_type == "base":
        flat_idx = out_idx * int(layer.n_in) + in_idx
        return grid[flat_idx]
    if layer_type == "spline":
        return grid[in_idx]
    if layer_type == "rbf":
        return grid[in_idx]
    raise ValueError(f"Unsupported expression export for layer_type={layer_type!r}.")


def _residual_weight(layer, out_idx: int, in_idx: int) -> float | None:
    if not hasattr(layer, "c_res"):
        return None
    return float(np.asarray(layer.c_res[...])[out_idx, in_idx]) * _edge_mask_value(
        layer, out_idx, in_idx
    )


def _layer_degree(layer) -> int | None:
    if hasattr(layer, "k"):
        return int(layer.k)
    if hasattr(layer, "D"):
        return int(layer.D)
    return None


def _basis_label(layer_type: str) -> str:
    return {
        "base": "B-spline",
        "spline": "B-spline",
        "chebyshev": "Chebyshev polynomial",
        "legendre": "Legendre polynomial",
        "rbf": "Gaussian radial basis",
        "sine": "normalized sine",
        "fourier": "Fourier sine/cosine",
    }.get(layer_type, layer_type)


def _edge_expression_text(
    layer,
    layer_type: str,
    layer_idx: int,
    out_idx: int,
    in_idx: int,
    residual_weight: float | None,
) -> str:
    prefix = f"phi_L{layer_idx}_y{out_idx}_x{in_idx}(x) = "
    degree = _layer_degree(layer)
    residual_name = _callable_name(getattr(layer, "residual", None))
    residual = ""
    if residual_weight is not None and abs(residual_weight) > 0.0:
        residual = f" + ({residual_weight:.8g}) * {residual_name}(x)"
    if layer_type in ("base", "spline"):
        return prefix + f"sum_j coefficients[j] * B_j,{degree}(x; grid)" + residual
    if layer_type == "chebyshev":
        offset = 1 if layer.bias is not None else 0
        return prefix + f"sum_j coefficients[j] * T_(j+{offset})(tanh(x))" + residual
    if layer_type == "legendre":
        offset = 1 if layer.bias is not None else 0
        return prefix + f"sum_j coefficients[j] * P_(j+{offset})(tanh(x))" + residual
    if layer_type == "rbf":
        std = getattr(layer, "kernel", {}).get("std", 1.0)
        return prefix + f"sum_j coefficients[j] * exp(-0.5*((x-grid[j])/{std})^2)" + residual
    if layer_type == "sine":
        return prefix + "sum_j coefficients[j] * normalized_sin_j(x)" + residual
    if layer_type == "fourier":
        return prefix + "sum_j cos_coefficients[j]*cos((j+1)x) + sin_coefficients[j]*sin((j+1)x)"
    return prefix + f"{_basis_label(layer_type)} edge function"


def _export_expressions(
    model: MultKAN,
    layer_type: str,
    feature_names: list[str],
    out_dir: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    spec: dict[str, Any],
    md_coeff_limit: int,
) -> tuple[Path, Path]:
    layers = []
    md_lines = [
        "# MKAN Edge Expressions",
        "",
        f"checkpoint: `{checkpoint_path}`",
        f"stage: `{checkpoint.get('stage')}`",
        f"step: `{checkpoint.get('step')}`",
        f"layer_type: `{layer_type}`",
        f"width: `{spec['width']}`",
        "",
        "Each edge function is represented in basis-expansion form:",
        "",
        "`phi_{layer,out,in}(x) = sum_j c_j * B_{j,k}(x; grid) + c_res * residual(x)`",
        "",
        "The JSON file contains the full grids and coefficients. This Markdown file keeps coefficients compact for reading.",
        "",
    ]

    for layer_idx, layer in enumerate(model.layers):
        degree = _layer_degree(layer)
        layer_grid = None
        if hasattr(layer, "grid"):
            layer_grid = np.asarray(layer.grid.item)
        layer_info = {
            "layer": layer_idx,
            "layer_type": layer_type,
            "n_in": int(layer.n_in),
            "n_out": int(layer.n_out),
            "degree_or_order": degree,
            "G": (
                int(getattr(layer.grid, "G", layer_grid.shape[-1] - 2 * int(layer.k) - 1))
                if layer_type in ("base", "spline") and layer_grid is not None
                else None
            ),
            "basis": _basis_label(layer_type),
            "formula": (
                "phi(layer,out,in,x) is the per-edge activation before layer bias; "
                "layer bias, subnode affine transforms, multiplication nodes, and "
                "node affine transforms are composed in wavefunction_formula.json."
            ),
            "residual": _callable_name(getattr(layer, "residual", None)),
            "bias": None if layer.bias is None else np.asarray(layer.bias[...]).tolist(),
            "subnode_scale": np.asarray(model.subnode_scale[layer_idx][...]).tolist(),
            "subnode_bias": np.asarray(model.subnode_bias[layer_idx][...]).tolist(),
            "node_scale": np.asarray(model.node_scale[layer_idx][...]).tolist(),
            "node_bias": np.asarray(model.node_bias[layer_idx][...]).tolist(),
            "edge_mask": (
                None
                if not hasattr(layer, "edge_mask")
                else np.asarray(layer.edge_mask[...]).tolist()
            ),
            "edges": [],
        }
        md_lines.extend([f"## Layer {layer_idx}", ""])

        for out_idx in range(int(layer.n_out)):
            for in_idx in range(int(layer.n_in)):
                input_name = (
                    feature_names[in_idx]
                    if layer_idx == 0 and in_idx < len(feature_names)
                    else f"layer{layer_idx}_x{in_idx}"
                )
                output_name = f"layer{layer_idx}_y{out_idx}"
                residual_weight = _residual_weight(layer, out_idx, in_idx)
                edge = {
                    "out_idx": out_idx,
                    "in_idx": in_idx,
                    "input_name": input_name,
                    "output_name": output_name,
                    "mask": _edge_mask_value(layer, out_idx, in_idx),
                    "residual_weight": residual_weight,
                    "expression": _edge_expression_text(
                        layer,
                        layer_type,
                        layer_idx,
                        out_idx,
                        in_idx,
                        residual_weight,
                    ),
                }

                if layer_type == "fourier":
                    edge["cos_coefficients"] = np.asarray(
                        layer.c_cos[...][out_idx, in_idx]
                    ).tolist()
                    edge["sin_coefficients"] = np.asarray(
                        layer.c_sin[...][out_idx, in_idx]
                    ).tolist()
                else:
                    coeffs = _edge_coefficients(layer, layer_type, out_idx, in_idx)
                    edge["coefficients"] = coeffs.tolist()
                    if layer_type in ("base", "spline", "rbf"):
                        edge["grid"] = _edge_grid(layer, layer_type, out_idx, in_idx).tolist()
                    if layer_type == "sine":
                        edge["omega"] = np.asarray(layer.omega[...]).tolist()
                        edge["phase"] = np.asarray(layer.phase[...]).tolist()
                    if layer_type == "rbf":
                        edge["kernel"] = dict(getattr(layer, "kernel", {"type": "gaussian"}))

                layer_info["edges"].append(edge)

                md_lines.extend([f"### y{out_idx} <- {input_name}", ""])
                md_lines.append(f"`{edge['expression']}`")
                md_lines.extend(["", f"- mask: `{edge['mask']:.6g}`"])
                if "grid" in edge:
                    grid = np.asarray(edge["grid"])
                    md_lines.append(
                        f"- grid: `{np.array2string(grid, precision=6, separator=', ')}`"
                    )
                if "coefficients" in edge:
                    coeffs = np.asarray(edge["coefficients"])
                    shown_coeffs = [f"{float(v):.6g}" for v in coeffs[:md_coeff_limit]]
                    if coeffs.size > md_coeff_limit:
                        shown_coeffs.append("...")
                    md_lines.append(f"- coefficients: `[{', '.join(shown_coeffs)}]`")
                if "cos_coefficients" in edge:
                    cos = np.asarray(edge["cos_coefficients"])
                    sin = np.asarray(edge["sin_coefficients"])
                    shown_cos = [f"{float(v):.6g}" for v in cos[:md_coeff_limit]]
                    shown_sin = [f"{float(v):.6g}" for v in sin[:md_coeff_limit]]
                    if cos.size > md_coeff_limit:
                        shown_cos.append("...")
                    if sin.size > md_coeff_limit:
                        shown_sin.append("...")
                    md_lines.append(f"- cos coefficients: `[{', '.join(shown_cos)}]`")
                    md_lines.append(f"- sin coefficients: `[{', '.join(shown_sin)}]`")
                md_lines.append("")

        layers.append(layer_info)

    payload = {
        "checkpoint": str(checkpoint_path),
        "stage": checkpoint.get("stage"),
        "step": checkpoint.get("step"),
        "layer_type": layer_type,
        "width": spec["width"],
        "expression_format": "basis_expansion",
        "layers": layers,
    }
    json_path = out_dir / "expressions.json"
    md_path = out_dir / "expressions.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2))
    md_path.write_text("\n".join(md_lines))
    return json_path, md_path


def _active_feature_names(model: MultKAN, feature_names: list[str]) -> list[str]:
    input_id = getattr(model, "input_id", None)
    if input_id is None:
        return feature_names[: model.width_in[0]]
    ids = np.asarray(input_id, dtype=np.int32).reshape(-1)
    return [feature_names[int(idx)] for idx in ids]


def _active_spin_blocks(electrons: tuple[int, ...]) -> list[dict[str, Any]]:
    labels = ["alpha", "beta"]
    blocks = []
    start = 0
    for spin_idx, count in enumerate(electrons):
        count = int(count)
        indices = list(range(start, start + count))
        if count > 0:
            label = labels[spin_idx] if spin_idx < len(labels) else f"spin{spin_idx}"
            blocks.append(
                {
                    "spin_index": spin_idx,
                    "label": label,
                    "size": count,
                    "global_electrons": indices,
                }
            )
        start += count
    return blocks


def _spin_pair_lists(electrons: tuple[int, ...]) -> tuple[list[list[int]], list[list[int]]]:
    n_alpha = int(electrons[0]) if len(electrons) > 0 else 0
    n_beta = int(electrons[1]) if len(electrons) > 1 else 0
    alpha = list(range(n_alpha))
    beta = list(range(n_alpha, n_alpha + n_beta))

    same_spin = []
    for group in (alpha, beta):
        for idx, electron_i in enumerate(group):
            for electron_j in group[idx + 1 :]:
                same_spin.append([electron_i, electron_j])
    opposite_spin = [[electron_i, electron_j] for electron_i in alpha for electron_j in beta]
    return same_spin, opposite_spin


def _checkpoint_params(checkpoint: dict[str, Any]) -> dict[str, Any]:
    params = checkpoint.get("params", {})
    return params if isinstance(params, dict) else {}


def _affine_formula(scale: Any, value: str, bias: Any) -> str:
    scale_f = _format_float(scale)
    bias_f = _format_float(bias)
    return f"({scale_f})*({value}) + ({bias_f})"


def _replace_e_symbol(expr: str, electron_label: str | int) -> str:
    return expr.replace("(e)", f"({electron_label})")


def _sum_terms(terms: list[str]) -> str:
    if not terms:
        return "0"
    return " + ".join(terms)


def _product_terms(terms: list[str]) -> str:
    if not terms:
        return "1"
    return " * ".join(f"({term})" for term in terms)


def _det_formula(entries: list[list[str]]) -> str:
    if not entries:
        return "1"
    if len(entries) == 1 and len(entries[0]) == 1:
        return entries[0][0]
    rows = ["[" + ", ".join(row) + "]" for row in entries]
    return "det([" + ", ".join(rows) + "])"


def _coord_expr(coord: float) -> str:
    value = float(coord)
    if abs(value) < 1.0e-12:
        return "0"
    return _format_float(value)


def _difference_expr(var: str, coord: float) -> str:
    value = float(coord)
    if abs(value) < 1.0e-12:
        return var
    if value > 0.0:
        return f"({var}-{_format_float(value)})"
    return f"({var}+{_format_float(abs(value))})"


def _feature_formula(feature_name: str, electron_idx: int, atoms: list[dict[str, Any]]) -> str:
    if feature_name.startswith("r_ae[") and feature_name.endswith("]"):
        atom_idx = int(feature_name[5:-1])
        atom = atoms[atom_idx]
        dx = _difference_expr(f"x{electron_idx}", atom["coords"][0])
        dy = _difference_expr(f"y{electron_idx}", atom["coords"][1])
        dz = _difference_expr(f"z{electron_idx}", atom["coords"][2])
        return f"sqrt(({dx})^2 + ({dy})^2 + ({dz})^2)"
    if feature_name.startswith("ae_") and "[" in feature_name and feature_name.endswith("]"):
        axis = feature_name[3]
        atom_idx = int(feature_name.split("[", 1)[1][:-1])
        atom = atoms[atom_idx]
        axis_to_coord = {"x": 0, "y": 1, "z": 2}
        return _difference_expr(f"{axis}{electron_idx}", atom["coords"][axis_to_coord[axis]])
    return f"{feature_name.replace('[', '_').replace(']', '')}_{electron_idx}"


def _basis_term_text(
    *,
    layer,
    layer_type: str,
    basis_idx: int,
    arg: str,
    grid: np.ndarray | None,
    hard_basis: bool = False,
    precision: int = 8,
) -> str:
    if layer_type in ("base", "spline"):
        if hard_basis and grid is not None:
            return _bspline_basis_expr(
                basis_idx=basis_idx,
                degree=int(layer.k),
                arg=arg,
                grid=grid,
                precision=precision,
            )
        grid_text = "None" if grid is None else np.array2string(
            grid, precision=8, separator=", "
        )
        return f"B_{basis_idx},{int(layer.k)}({arg}; grid={grid_text})"
    if layer_type == "chebyshev":
        offset = 1 if layer.bias is not None else 0
        order = basis_idx + offset
        tanh_arg = f"tanh({arg})"
        return _chebyshev_expr(order, tanh_arg) if hard_basis else f"T_{order}({tanh_arg})"
    if layer_type == "legendre":
        offset = 1 if layer.bias is not None else 0
        order = basis_idx + offset
        tanh_arg = f"tanh({arg})"
        return _legendre_expr(order, tanh_arg) if hard_basis else f"P_{order}({tanh_arg})"
    if layer_type == "rbf":
        center = 0.0 if grid is None else float(grid[basis_idx])
        std = getattr(layer, "kernel", {}).get("std", 1.0)
        return (
            f"exp(-0.5*((({arg})-({_format_float(center, precision)}))"
            f"/{_format_float(std, precision)})^2)"
        )
    if layer_type == "sine":
        omega = float(np.asarray(layer.omega[...])[basis_idx])
        phase = float(np.asarray(layer.phase[...])[basis_idx])
        mu = np.exp(-0.5 * omega**2) * np.sin(phase)
        std = np.sqrt(0.5 * (1.0 - np.exp(-2.0 * omega**2) * np.cos(2.0 * phase)) - mu**2)
        return (
            f"(sin(({_format_float(omega, precision)})*({arg})+({_format_float(phase, precision)}))"
            f"-({_format_float(mu, precision)}))/({_format_float(std, precision)}+1e-8)"
        )
    raise ValueError(f"Unsupported basis term for layer_type={layer_type!r}.")


def _edge_basis_formula(
    layer,
    layer_type: str,
    out_idx: int,
    in_idx: int,
    arg: str,
    *,
    hard_basis: bool = False,
    hard_residual: bool = False,
    precision: int = 8,
) -> str:
    if layer_type == "fourier":
        terms = []
        cos_coeffs = np.asarray(layer.c_cos[...][out_idx, in_idx])
        sin_coeffs = np.asarray(layer.c_sin[...][out_idx, in_idx])
        for basis_idx, value in enumerate(cos_coeffs):
            if abs(float(value)) > 0.0:
                terms.append(
                    f"({_format_float(value, precision)})*cos({basis_idx + 1}*({arg}))"
                )
        for basis_idx, value in enumerate(sin_coeffs):
            if abs(float(value)) > 0.0:
                terms.append(
                    f"({_format_float(value, precision)})*sin({basis_idx + 1}*({arg}))"
                )
        bias = None if layer.bias is None else np.asarray(layer.bias[...])[out_idx]
        if bias is not None and abs(float(bias)) > 0.0:
            terms.append(f"({_format_float(bias, precision)})")
        return _sum_terms(terms)

    coeffs = _edge_coefficients(layer, layer_type, out_idx, in_idx)
    grid = None
    if layer_type in ("base", "spline", "rbf"):
        grid = _edge_grid(layer, layer_type, out_idx, in_idx)
    terms = [
        f"({_format_float(value, precision)})*{_basis_term_text(layer=layer, layer_type=layer_type, basis_idx=basis_idx, arg=arg, grid=grid, hard_basis=hard_basis, precision=precision)}"
        for basis_idx, value in enumerate(coeffs)
        if abs(float(value)) > 0.0
    ]

    residual_weight = _residual_weight(layer, out_idx, in_idx)
    if residual_weight is not None and abs(float(residual_weight)) > 0.0:
        residual_name = _callable_name(getattr(layer, "residual", None))
        residual_expr = (
            _hard_residual_text(residual_name, arg)
            if hard_residual
            else f"{residual_name}({arg})"
        )
        terms.append(f"({_format_float(residual_weight, precision)})*{residual_expr}")
    return _sum_terms(terms)


def _inline_edge_basis_outputs(
    *,
    model: MultKAN,
    layer_type: str,
    feature_names: list[str],
    atoms: list[dict[str, Any]],
    nelectrons: int,
    hard_basis: bool = False,
    hard_residual: bool = False,
    precision: int = 8,
) -> list[list[str]]:
    active_features = _active_feature_names(model, feature_names)
    all_outputs = []
    for electron_idx in range(nelectrons):
        current = [
            _feature_formula(feature, electron_idx, atoms)
            for feature in active_features
        ]

        for layer_idx, layer in enumerate(model.layers):
            layer_bias = (
                np.zeros((int(layer.n_out),), dtype=np.float64)
                if layer.bias is None
                else np.asarray(layer.bias[...])
            )
            subnode_scale = np.asarray(model.subnode_scale[layer_idx][...])
            subnode_bias = np.asarray(model.subnode_bias[layer_idx][...])
            node_scale = np.asarray(model.node_scale[layer_idx][...])
            node_bias = np.asarray(model.node_bias[layer_idx][...])

            subnodes = []
            for out_idx in range(int(layer.n_out)):
                edge_terms = [
                    _edge_basis_formula(
                        layer,
                        layer_type,
                        out_idx,
                        in_idx,
                        current[in_idx],
                        hard_basis=hard_basis,
                        hard_residual=hard_residual,
                        precision=precision,
                    )
                    for in_idx in range(int(layer.n_in))
                ]
                u_formula = _sum_terms(edge_terms)
                if abs(float(layer_bias[out_idx])) > 0.0:
                    u_formula = f"{u_formula} + ({_format_float(layer_bias[out_idx])})"
                subnodes.append(
                    _affine_formula(subnode_scale[out_idx], u_formula, subnode_bias[out_idx])
                )

            n_sum = int(model.width[layer_idx + 1][0])
            arities = model._arity_list_for_width(layer_idx + 1)
            next_nodes = []
            for node_idx in range(n_sum):
                next_nodes.append(
                    _affine_formula(node_scale[node_idx], subnodes[node_idx], node_bias[node_idx])
                )

            offset = n_sum
            for mult_idx, arity in enumerate(arities):
                node_idx = n_sum + mult_idx
                raw = _product_terms(subnodes[offset : offset + arity])
                next_nodes.append(_affine_formula(node_scale[node_idx], raw, node_bias[node_idx]))
                offset += arity
            current = next_nodes
        all_outputs.append(current)
    return all_outputs


def _export_hard_edge_expressions(
    *,
    model: MultKAN,
    layer_type: str,
    feature_names: list[str],
    out_dir: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    spec: dict[str, Any],
    precision: int,
) -> tuple[Path, Path, dict[str, Any]]:
    layers = []
    md_lines = [
        "# Hard-Inlined MKAN Edge Expressions",
        "",
        f"checkpoint: `{checkpoint_path}`",
        f"stage: `{checkpoint.get('stage')}`",
        f"step: `{checkpoint.get('step')}`",
        f"layer_type: `{layer_type}`",
        f"width: `{spec['width']}`",
        f"numeric precision: `{precision}` significant digits",
        "",
        "These are the trained edge functions with basis symbols expanded.",
        "`I(a <= x < b)` is an interval indicator: it is 1 inside that interval and 0 outside.",
        "The values are rounded for readability, so this file is a finite-decimal symbolic approximation of the checkpoint.",
        "",
    ]

    for layer_idx, layer in enumerate(model.layers):
        degree = _layer_degree(layer)
        layer_info = {
            "layer": layer_idx,
            "layer_type": layer_type,
            "n_in": int(layer.n_in),
            "n_out": int(layer.n_out),
            "degree_or_order": degree,
            "basis": _basis_label(layer_type),
            "residual": _callable_name(getattr(layer, "residual", None)),
            "edges": [],
        }
        md_lines.extend([f"## Layer {layer_idx}", ""])
        for out_idx in range(int(layer.n_out)):
            for in_idx in range(int(layer.n_in)):
                input_name = (
                    feature_names[in_idx]
                    if layer_idx == 0 and in_idx < len(feature_names)
                    else f"a_{layer_idx}_{in_idx}(e)"
                )
                symbol = f"phi_L{layer_idx}_y{out_idx}_x{in_idx}"
                formula_body = _edge_basis_formula(
                    layer,
                    layer_type,
                    out_idx,
                    in_idx,
                    "x",
                    hard_basis=True,
                    hard_residual=True,
                    precision=precision,
                )
                edge = {
                    "symbol": symbol,
                    "out_idx": out_idx,
                    "in_idx": in_idx,
                    "input_name": input_name,
                    "mask": _edge_mask_value(layer, out_idx, in_idx),
                    "formula": f"{symbol}(x) = {formula_body}",
                    "rounded_precision": int(precision),
                }
                if layer_type != "fourier":
                    edge["coefficients"] = [
                        float(_format_hard_float(value, precision))
                        for value in _edge_coefficients(layer, layer_type, out_idx, in_idx)
                    ]
                    if layer_type in ("base", "spline", "rbf"):
                        edge["grid"] = [
                            float(_format_hard_float(value, precision))
                            for value in _edge_grid(layer, layer_type, out_idx, in_idx)
                        ]
                layer_info["edges"].append(edge)
                md_lines.extend(
                    [
                        f"### {symbol}",
                        "",
                        f"- input: `{input_name}`",
                        f"- mask: `{edge['mask']:.6g}`",
                        "",
                        "```text",
                        edge["formula"],
                        "```",
                        "",
                    ]
                )
        layers.append(layer_info)

    payload = {
        "checkpoint": str(checkpoint_path),
        "stage": checkpoint.get("stage"),
        "step": checkpoint.get("step"),
        "layer_type": layer_type,
        "width": spec["width"],
        "rounded_precision": int(precision),
        "indicator_convention": "I(a <= x < b) equals 1 if x is in [a,b), else 0.",
        "expression_format": "hard_inlined_edge_basis",
        "layers": layers,
    }
    json_path = out_dir / "edge_piecewise_expressions.json"
    md_path = out_dir / "edge_piecewise_expressions.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2))
    md_path.write_text("\n".join(md_lines))
    return json_path, md_path, payload


def _mkan_composition_payload(
    model: MultKAN,
    feature_names: list[str],
) -> dict[str, Any]:
    active_features = _active_feature_names(model, feature_names)
    input_symbols = [f"a_0_{idx}(e)" for idx in range(len(active_features))]
    current = list(input_symbols)
    current_expanded = list(input_symbols)

    layers = []
    for layer_idx, layer in enumerate(model.layers):
        layer_bias = (
            np.zeros((int(layer.n_out),), dtype=np.float64)
            if layer.bias is None
            else np.asarray(layer.bias[...])
        )
        subnode_scale = np.asarray(model.subnode_scale[layer_idx][...])
        subnode_bias = np.asarray(model.subnode_bias[layer_idx][...])
        node_scale = np.asarray(model.node_scale[layer_idx][...])
        node_bias = np.asarray(model.node_bias[layer_idx][...])

        pre_subnodes = []
        subnodes = []
        subnode_expanded = []
        for out_idx in range(int(layer.n_out)):
            edge_terms = [
                f"phi_L{layer_idx}_y{out_idx}_x{in_idx}({current[in_idx]})"
                for in_idx in range(int(layer.n_in))
            ]
            edge_terms_expanded = [
                f"phi_L{layer_idx}_y{out_idx}_x{in_idx}({current_expanded[in_idx]})"
                for in_idx in range(int(layer.n_in))
            ]
            u_symbol = f"u_{layer_idx}_{out_idx}(e)"
            u_formula = _sum_terms(edge_terms)
            u_expanded = _sum_terms(edge_terms_expanded)
            if abs(float(layer_bias[out_idx])) > 0.0:
                bias_term = f"({_format_float(layer_bias[out_idx])})"
                u_formula = f"{u_formula} + {bias_term}"
                u_expanded = f"{u_expanded} + {bias_term}"
            pre_subnodes.append(
                {
                    "symbol": u_symbol,
                    "formula": u_formula,
                    "expanded_formula": u_expanded,
                    "edge_terms": edge_terms,
                    "layer_bias": float(layer_bias[out_idx]),
                }
            )

            v_symbol = f"v_{layer_idx}_{out_idx}(e)"
            v_formula = _affine_formula(
                subnode_scale[out_idx],
                u_symbol,
                subnode_bias[out_idx],
            )
            v_expanded = _affine_formula(
                subnode_scale[out_idx],
                u_expanded,
                subnode_bias[out_idx],
            )
            subnodes.append(
                {
                    "symbol": v_symbol,
                    "formula": v_formula,
                    "expanded_formula": v_expanded,
                    "source": u_symbol,
                    "scale": float(subnode_scale[out_idx]),
                    "bias": float(subnode_bias[out_idx]),
                }
            )
            subnode_expanded.append(v_expanded)

        n_sum = int(model.width[layer_idx + 1][0])
        arities = model._arity_list_for_width(layer_idx + 1)
        next_nodes = []
        next_current = []
        next_expanded = []

        for node_idx in range(n_sum):
            source = f"v_{layer_idx}_{node_idx}(e)"
            symbol = f"a_{layer_idx + 1}_{node_idx}(e)"
            formula = _affine_formula(node_scale[node_idx], source, node_bias[node_idx])
            expanded_formula = _affine_formula(
                node_scale[node_idx],
                subnode_expanded[node_idx],
                node_bias[node_idx],
            )
            next_nodes.append(
                {
                    "symbol": symbol,
                    "formula": formula,
                    "expanded_formula": expanded_formula,
                    "source": source,
                    "kind": "additive",
                    "scale": float(node_scale[node_idx]),
                    "bias": float(node_bias[node_idx]),
                }
            )
            next_current.append(symbol)
            next_expanded.append(expanded_formula)

        offset = n_sum
        for mult_idx, arity in enumerate(arities):
            sources = [f"v_{layer_idx}_{idx}(e)" for idx in range(offset, offset + arity)]
            raw = _product_terms(sources)
            raw_expanded = _product_terms(
                [subnode_expanded[idx] for idx in range(offset, offset + arity)]
            )
            node_idx = n_sum + mult_idx
            symbol = f"a_{layer_idx + 1}_{node_idx}(e)"
            formula = _affine_formula(node_scale[node_idx], raw, node_bias[node_idx])
            expanded_formula = _affine_formula(
                node_scale[node_idx],
                raw_expanded,
                node_bias[node_idx],
            )
            next_nodes.append(
                {
                    "symbol": symbol,
                    "formula": formula,
                    "expanded_formula": expanded_formula,
                    "source": raw,
                    "sources": sources,
                    "kind": "multiplicative",
                    "arity": int(arity),
                    "scale": float(node_scale[node_idx]),
                    "bias": float(node_bias[node_idx]),
                }
            )
            next_current.append(symbol)
            next_expanded.append(expanded_formula)
            offset += arity

        layers.append(
            {
                "layer": layer_idx,
                "pre_subnodes": pre_subnodes,
                "subnodes": subnodes,
                "nodes": next_nodes,
            }
        )
        current = next_current
        current_expanded = next_expanded

    outputs = [
        {
            "index": idx,
            "symbol": f"F_{idx}(e)",
            "definition": current[idx],
            "expanded_formula": current_expanded[idx],
        }
        for idx in range(len(current))
    ]
    return {
        "input_symbols": [
            {"symbol": symbol, "feature": feature}
            for symbol, feature in zip(input_symbols, active_features)
        ],
        "input_id": (
            None
            if getattr(model, "input_id", None) is None
            else np.asarray(getattr(model, "input_id")).astype(int).tolist()
        ),
        "layers": layers,
        "outputs": outputs,
    }


def _envelope_payload(
    cfg: ml_collections.ConfigDict,
    checkpoint: dict[str, Any],
    electrons: tuple[int, ...],
    ndeterminants: int,
    nelectrons: int,
) -> dict[str, Any]:
    enabled = bool(cfg.get("envelope_on", False))
    active_blocks = _active_spin_blocks(electrons)
    if not enabled:
        return {
            "enabled": False,
            "formula": "E_b(i,c) = 1",
            "definitions": [],
            "parameters": None,
        }

    params = _checkpoint_params(checkpoint).get("envelope", None)
    envelope_type = str(cfg.get("envelope_type", "isotropic")).lower()
    output_dims = [
        int(ndeterminants * (nelectrons if bool(cfg.get("full_det", True)) else block["size"]))
        for block in active_blocks
    ]
    if envelope_type == "chebyshev":
        formula = (
            "E_b(i,c) = sum_A exp(-sigma[b,A,c] * r_iA) * "
            "sum_d c_basis[b,A,d,c] * T_d(tanh(r_iA))"
        )
    elif envelope_type == "legendre":
        formula = (
            "E_b(i,c) = sum_A exp(-sigma[b,A,c] * r_iA) * "
            "sum_d p_basis[b,A,d,c] * P_d(tanh(r_iA))"
        )
    elif envelope.is_legendre_anisotropic(envelope_type):
        formula = (
            "E_b(i,c) = sum_A exp(-rho[b,A,c]) * "
            "sum_d p_basis[b,A,d,c] * P_d(tanh(rho[b,A,c])), "
            "rho[b,A,c] = ||Sigma[b,A,c] * (r_i - R_A)||"
        )
    elif envelope.is_angular_momentum(envelope_type):
        formula = (
            "E_b(i,c) = sum_A exp(-sigma[b,A,c] * r_iA) * "
            "sum_{l=0..L} sum_{m=-l..l} angular_coeff[b,A,l,m,c] * "
            "Yreal_lm(theta_iA, phi_iA)"
        )
    elif envelope.is_legendre_angular(envelope_type):
        formula = (
            "E_b(i,c) = sum_A exp(-sigma[b,A,c] * r_iA) * "
            "sum_d p_basis[b,A,d,c] * P_d(tanh(r_iA)) * "
            "sum_{l=0..L} sum_{m=-l..l} angular_coeff[b,A,l,m,c] * "
            "Yreal_lm(theta_iA, phi_iA)"
        )
    elif envelope.is_complex_angular_momentum(envelope_type):
        formula = (
            "E_b(i,c) = sum_A exp(-sigma[b,A,c] * r_iA) * "
            "sum_{l=0..L} sum_{m=-l..l} "
            "(angular_coeff_real[b,A,l,m,c] + i*angular_coeff_imag[b,A,l,m,c]) * "
            "Y_lm(theta_iA, phi_iA)"
        )
    elif envelope_type == "isotropic":
        formula = "E_b(i,c) = sum_A pi[b,A,c] * exp(-sigma[b,A,c] * r_iA)"
    else:
        formula = f"E_b(i,c) = {envelope_type} envelope used by training code"

    definitions = []
    if params is not None:
        for block_idx, block in enumerate(active_blocks):
            if block_idx >= len(params):
                continue
            block_params = params[block_idx]
            label = block["label"]
            output_dim = output_dims[block_idx]
            if envelope_type == "chebyshev":
                sigma = np.asarray(block_params["sigma"])
                c_basis = np.asarray(block_params["c_basis"])
                for channel_idx in range(output_dim):
                    atom_terms = []
                    for atom_idx in range(sigma.shape[0]):
                        r_symbol = f"r_{label}[a,A{atom_idx}]"
                        cheb_terms = [
                            f"({_format_float(c_basis[atom_idx, degree, channel_idx])})"
                            f"*({_chebyshev_expr(degree, f'tanh({r_symbol})')})"
                            for degree in range(c_basis.shape[1])
                            if abs(float(c_basis[atom_idx, degree, channel_idx])) > 0.0
                        ]
                        if not cheb_terms:
                            cheb_terms = ["0"]
                        atom_terms.append(
                            f"exp(-({_format_float(sigma[atom_idx, channel_idx])})"
                            f"*{r_symbol})*({_sum_terms(cheb_terms)})"
                        )
                    definitions.append(
                        {
                            "symbol": f"E_{label}(a,{channel_idx})",
                            "formula": _sum_terms(atom_terms),
                        }
                    )
            elif envelope_type == "legendre":
                sigma = np.asarray(block_params["sigma"])
                p_basis = np.asarray(block_params["p_basis"])
                for channel_idx in range(output_dim):
                    atom_terms = []
                    for atom_idx in range(sigma.shape[0]):
                        r_symbol = f"r_{label}[a,A{atom_idx}]"
                        legendre_terms = [
                            f"({_format_float(p_basis[atom_idx, degree, channel_idx])})"
                            f"*({_legendre_expr(degree, f'tanh({r_symbol})')})"
                            for degree in range(p_basis.shape[1])
                            if abs(float(p_basis[atom_idx, degree, channel_idx])) > 0.0
                        ]
                        if not legendre_terms:
                            legendre_terms = ["0"]
                        atom_terms.append(
                            f"exp(-({_format_float(sigma[atom_idx, channel_idx])})"
                            f"*{r_symbol})*({_sum_terms(legendre_terms)})"
                        )
                    definitions.append(
                        {
                            "symbol": f"E_{label}(a,{channel_idx})",
                            "formula": _sum_terms(atom_terms),
                        }
                    )
            elif envelope.is_legendre_anisotropic(envelope_type):
                sigma = np.asarray(block_params["sigma"])
                p_basis = np.asarray(block_params["p_basis"])
                for channel_idx in range(output_dim):
                    atom_terms = []
                    for atom_idx in range(sigma.shape[0]):
                        matrix_rows = [
                            "["
                            + ", ".join(
                                _format_float(sigma[atom_idx, channel_idx, row, col])
                                for col in range(sigma.shape[3])
                            )
                            + "]"
                            for row in range(sigma.shape[2])
                        ]
                        matrix_expr = "[" + ", ".join(matrix_rows) + "]"
                        ae_symbol = f"ae_{label}[a,A{atom_idx},:]"
                        rho_symbol = f"rho_{label}(a,A{atom_idx},{channel_idx})"
                        definitions.append(
                            {
                                "symbol": rho_symbol,
                                "formula": f"norm(({matrix_expr}) @ {ae_symbol})",
                            }
                        )
                        legendre_terms = [
                            f"({_format_float(p_basis[atom_idx, degree, channel_idx])})"
                            f"*({_legendre_expr(degree, f'tanh({rho_symbol})')})"
                            for degree in range(p_basis.shape[1])
                            if abs(float(p_basis[atom_idx, degree, channel_idx])) > 0.0
                        ]
                        if not legendre_terms:
                            legendre_terms = ["0"]
                        atom_terms.append(
                            f"exp(-{rho_symbol})*({_sum_terms(legendre_terms)})"
                        )
                    definitions.append(
                        {
                            "symbol": f"E_{label}(a,{channel_idx})",
                            "formula": _sum_terms(atom_terms),
                        }
                    )
            elif envelope.is_angular_momentum(envelope_type):
                sigma = np.asarray(block_params["sigma"])
                angular_coeff = np.asarray(block_params["angular_coeff"])
                for channel_idx in range(output_dim):
                    atom_terms = []
                    for atom_idx in range(sigma.shape[0]):
                        r_symbol = f"r_{label}[a,A{atom_idx}]"
                        theta_symbol = f"theta_{label}[a,A{atom_idx}]"
                        phi_symbol = f"phi_{label}[a,A{atom_idx}]"
                        angular_terms = [
                            f"({_format_float(angular_coeff[atom_idx, basis_idx, channel_idx])})"
                            f"*{_real_spherical_harmonic_symbol(basis_idx)}"
                            f"({theta_symbol},{phi_symbol})"
                            for basis_idx in range(angular_coeff.shape[1])
                            if abs(float(angular_coeff[atom_idx, basis_idx, channel_idx])) > 0.0
                        ]
                        if not angular_terms:
                            angular_terms = ["0"]
                        atom_terms.append(
                            f"exp(-({_format_float(sigma[atom_idx, channel_idx])})"
                            f"*{r_symbol})*({_sum_terms(angular_terms)})"
                        )
                    definitions.append(
                        {
                            "symbol": f"E_{label}(a,{channel_idx})",
                            "formula": _sum_terms(atom_terms),
                        }
                    )
            elif envelope.is_legendre_angular(envelope_type):
                sigma = np.asarray(block_params["sigma"])
                p_basis = np.asarray(block_params["p_basis"])
                angular_coeff = np.asarray(block_params["angular_coeff"])
                for channel_idx in range(output_dim):
                    atom_terms = []
                    for atom_idx in range(sigma.shape[0]):
                        r_symbol = f"r_{label}[a,A{atom_idx}]"
                        theta_symbol = f"theta_{label}[a,A{atom_idx}]"
                        phi_symbol = f"phi_{label}[a,A{atom_idx}]"
                        legendre_terms = [
                            f"({_format_float(p_basis[atom_idx, degree, channel_idx])})"
                            f"*({_legendre_expr(degree, f'tanh({r_symbol})')})"
                            for degree in range(p_basis.shape[1])
                            if abs(float(p_basis[atom_idx, degree, channel_idx])) > 0.0
                        ]
                        if not legendre_terms:
                            legendre_terms = ["0"]
                        angular_terms = [
                            f"({_format_float(angular_coeff[atom_idx, basis_idx, channel_idx])})"
                            f"*{_real_spherical_harmonic_symbol(basis_idx)}"
                            f"({theta_symbol},{phi_symbol})"
                            for basis_idx in range(angular_coeff.shape[1])
                            if abs(float(angular_coeff[atom_idx, basis_idx, channel_idx])) > 0.0
                        ]
                        if not angular_terms:
                            angular_terms = ["0"]
                        atom_terms.append(
                            f"exp(-({_format_float(sigma[atom_idx, channel_idx])})"
                            f"*{r_symbol})"
                            f"*({_sum_terms(legendre_terms)})"
                            f"*({_sum_terms(angular_terms)})"
                        )
                    definitions.append(
                        {
                            "symbol": f"E_{label}(a,{channel_idx})",
                            "formula": _sum_terms(atom_terms),
                        }
                    )
            elif envelope.is_complex_angular_momentum(envelope_type):
                sigma = np.asarray(block_params["sigma"])
                angular_coeff_real = np.asarray(block_params["angular_coeff_real"])
                angular_coeff_imag = np.asarray(block_params["angular_coeff_imag"])
                for channel_idx in range(output_dim):
                    atom_terms = []
                    for atom_idx in range(sigma.shape[0]):
                        r_symbol = f"r_{label}[a,A{atom_idx}]"
                        theta_symbol = f"theta_{label}[a,A{atom_idx}]"
                        phi_symbol = f"phi_{label}[a,A{atom_idx}]"
                        angular_terms = []
                        for basis_idx in range(angular_coeff_real.shape[1]):
                            real_value = float(angular_coeff_real[atom_idx, basis_idx, channel_idx])
                            imag_value = float(angular_coeff_imag[atom_idx, basis_idx, channel_idx])
                            if abs(real_value) <= 0.0 and abs(imag_value) <= 0.0:
                                continue
                            coeff = (
                                f"({_format_float(real_value)}"
                                f"+i*{_format_float(imag_value)})"
                            )
                            angular_terms.append(
                                f"{coeff}*Ycomplex_{_real_spherical_harmonic_symbol(basis_idx)[6:]}"
                                f"({theta_symbol},{phi_symbol})"
                            )
                        if not angular_terms:
                            angular_terms = ["0"]
                        atom_terms.append(
                            f"exp(-({_format_float(sigma[atom_idx, channel_idx])})"
                            f"*{r_symbol})*({_sum_terms(angular_terms)})"
                        )
                    definitions.append(
                        {
                            "symbol": f"E_{label}(a,{channel_idx})",
                            "formula": _sum_terms(atom_terms),
                        }
                    )
            elif envelope_type == "isotropic":
                pi = np.asarray(block_params["pi"])
                sigma = np.asarray(block_params["sigma"])
                for channel_idx in range(output_dim):
                    atom_terms = []
                    for atom_idx in range(sigma.shape[0]):
                        r_symbol = f"r_{label}[a,A{atom_idx}]"
                        atom_terms.append(
                            f"({_format_float(pi[atom_idx, channel_idx])})"
                            f"*exp(-({_format_float(sigma[atom_idx, channel_idx])})*{r_symbol})"
                        )
                    definitions.append(
                        {
                            "symbol": f"E_{label}(a,{channel_idx})",
                            "formula": _sum_terms(atom_terms),
                        }
                    )
    return {
        "enabled": True,
        "type": envelope_type,
        "active_spin_blocks": active_blocks,
        "output_dims": output_dims,
        "formula": formula,
        "definitions": definitions,
        "parameters": _jsonable(params),
    }


def _jastrow_payload(
    cfg: ml_collections.ConfigDict,
    checkpoint: dict[str, Any],
    electrons: tuple[int, ...],
) -> dict[str, Any]:
    jastrow_cfg = cfg.get("jastrow", {})
    ee_enabled = bool(jastrow_cfg.get("ee", True))
    en_enabled = bool(jastrow_cfg.get("en", False))
    jastrow_type = str(jastrow_cfg.get("type", "pade")).lower()
    same_spin_pairs, opposite_spin_pairs = _spin_pair_lists(electrons)
    params = _checkpoint_params(checkpoint)

    if jastrow_type == "pade":
        ee_kernel = "u_cusp(r;c,alpha) = c*r/(1+alpha*r)"
    elif jastrow_type == "ferminet":
        ee_kernel = "u_cusp(r;c,alpha) = -(c*alpha^2)/(alpha+r)"
    elif jastrow_type in ("ferminet_plus", "ferminet_three_body"):
        ee_kernel = (
            "u_cusp_plus(r;c,alpha,a) = -(c*alpha^2)/(alpha+r) + "
            "sum_n a[n]*(r/(1+r))^(n+2)"
        )
    else:
        ee_kernel = f"{jastrow_type} electron-electron Jastrow"

    ee_terms = []
    jastrow_ee_params = params.get("jastrow_ee", None)
    if ee_enabled and jastrow_ee_params is not None:
        jastrow_ee_params = {
            key: np.asarray(value) for key, value in dict(jastrow_ee_params).items()
        }

        def pair_term(pair: list[int], same_spin: bool) -> str:
            i, j = pair
            r = f"r_{i}{j}"
            if same_spin:
                cusp = 0.25
                alpha_name = "ee_par"
                coeff_name = "ee_par_coeff"
            else:
                cusp = 0.5
                alpha_name = "ee_anti"
                coeff_name = "ee_anti_coeff"
            alpha = _format_float(jastrow_ee_params[alpha_name][0])
            if jastrow_type == "pade":
                return f"({cusp})*{r}/(1+({alpha})*{r})"
            if jastrow_type == "ferminet":
                return f"-(({cusp})*({alpha})^2)/(({alpha})+{r})"
            if jastrow_type in ("ferminet_plus", "ferminet_three_body"):
                coeff = jastrow_ee_params.get(coeff_name, np.asarray([]))
                radial_terms = [
                    f"({_format_float(value)})*({r}/(1+{r}))^{order + 2}"
                    for order, value in enumerate(coeff)
                    if abs(float(value)) > 0.0
                ]
                base = f"-(({cusp})*({alpha})^2)/(({alpha})+{r})"
                return _sum_terms([base, *radial_terms])
            return f"u_{'same' if same_spin else 'opposite'}({r})"

        ee_terms.extend(pair_term(pair, same_spin=True) for pair in same_spin_pairs)
        ee_terms.extend(pair_term(pair, same_spin=False) for pair in opposite_spin_pairs)

        if jastrow_type == "ferminet_three_body":
            for pair_group, coeff_name in (
                (same_spin_pairs, "een_par_coeff"),
                (opposite_spin_pairs, "een_anti_coeff"),
            ):
                coeff = jastrow_ee_params.get(coeff_name, np.asarray([]))
                for i, j in pair_group:
                    r = f"r_{i}{j}"
                    rho_i = f"rho_{i}"
                    rho_j = f"rho_{j}"
                    features = [
                        f"({r}/(1+{r}))^2*({rho_i}+{rho_j})",
                        f"({r}/(1+{r}))^2*{rho_i}*{rho_j}",
                        f"({r}/(1+{r}))^2*({rho_i}-{rho_j})^2",
                        f"({r}/(1+{r}))^3*({rho_i}+{rho_j})",
                    ]
                    for value, feature in zip(coeff, features):
                        if abs(float(value)) > 0.0:
                            ee_terms.append(f"({_format_float(value)})*{feature}")

    ee_formula = "J_ee = 0" if not ee_terms else f"J_ee = {_sum_terms(ee_terms)}"
    if ee_enabled and not ee_terms and (same_spin_pairs or opposite_spin_pairs):
        ee_formula = (
            "J_ee = sum_(i,j in same_spin_pairs) u_same(r_ij) + "
            "sum_(i,j in opposite_spin_pairs) u_opposite(r_ij)"
        )

    en_terms = []
    jastrow_en_params = params.get("jastrow_en", None)
    if en_enabled and jastrow_en_params is not None:
        coeff = np.asarray(jastrow_en_params["en_coeff"])
        for electron_idx in range(sum(electrons)):
            for atom_idx in range(coeff.shape[0]):
                r = f"r_{electron_idx}A{atom_idx}"
                for order in range(coeff.shape[1]):
                    value = float(coeff[atom_idx, order])
                    if abs(value) > 0.0:
                        en_terms.append(
                            f"({_format_float(value)})*({r}/(1+{r}))^{order + 1}"
                        )

    en_formula = "J_en = 0" if not en_terms else f"J_en = {_sum_terms(en_terms)}"
    expanded_formula = _sum_terms(ee_terms + en_terms)

    return {
        "ee_enabled": ee_enabled,
        "en_enabled": en_enabled,
        "type": jastrow_type,
        "same_spin_pairs": same_spin_pairs,
        "opposite_spin_pairs": opposite_spin_pairs,
        "r_ij": "r_ij = ||r_i - r_j||",
        "ee_kernel": ee_kernel,
        "ee_formula": ee_formula,
        "en_formula": en_formula,
        "total_formula": "J(R) = J_ee(R) + J_en(R)",
        "expanded_formula": expanded_formula,
        "parameters": {
            "jastrow_ee": _jsonable(params.get("jastrow_ee", None)),
            "jastrow_en": _jsonable(params.get("jastrow_en", None)),
        },
    }


def _determinant_payload(
    cfg: ml_collections.ConfigDict,
    checkpoint: dict[str, Any],
    electrons: tuple[int, ...],
    ndeterminants: int,
    nelectrons: int,
) -> dict[str, Any]:
    complex_output = bool(cfg.get("complex_output", False))
    full_det = bool(cfg.get("full_det", True))
    use_weights = bool(cfg.get("determinant_weights", ndeterminants > 1))
    params = _checkpoint_params(checkpoint)
    blocks = _active_spin_blocks(electrons)

    orbital_channels = (
        "Q_iq = F_(2q)(i) + 1j*F_(2q+1)(i)"
        if complex_output
        else "Q_iq = F_q(i)"
    )
    if full_det:
        matrix_formula = (
            "M_d[i,j] = Q_i,(d*N+j) * E_spin(i, d*N+j), "
            "for i,j=0..N-1"
        )
        block_formulas = [
            {
                "kind": "full_det",
                "formula": matrix_formula,
            }
        ]
        determinant_formula = "D_d = det(M_d)"
    else:
        starts = np.cumsum(
            (0, *[ndeterminants * int(spin) for spin in electrons[:-1]])
        ).astype(int)
        block_formulas = []
        for block in blocks:
            spin_idx = int(block["spin_index"])
            spin_size = int(block["size"])
            start = int(starts[spin_idx])
            label = block["label"]
            block_formulas.append(
                {
                    "kind": "spin_block",
                    "label": label,
                    "spin_index": spin_idx,
                    "column_start": start,
                    "formula": (
                        f"M_{label},d[a,b] = Q_i,(start_{label}+d*n_{label}+b) "
                        f"* E_{label}(a, d*n_{label}+b), "
                        f"where i is global electron index in {label}"
                    ),
                    "size": spin_size,
                }
            )
        determinant_formula = "D_d = product_over_active_spin_blocks det(M_block,d)"

    if ndeterminants > 1:
        weighted = "sum_d w_d * D_d" if use_weights else "sum_d D_d"
    else:
        weighted = "D_0"
    psi_formula = f"Psi(R) = exp(J(R)) * ({weighted})"

    return {
        "complex_output": complex_output,
        "full_det": full_det,
        "ndeterminants": ndeterminants,
        "nelectrons": nelectrons,
        "active_spin_blocks": blocks,
        "orbital_channel_formula": orbital_channels,
        "matrix_formulas": block_formulas,
        "determinant_formula": determinant_formula,
        "determinant_weights_enabled": use_weights,
        "determinant_weights": _jsonable(params.get("det_weights", None)),
        "wavefunction_formula": psi_formula,
    }


def _mkan_output_formula(
    mkan: dict[str, Any],
    output_idx: int,
    electron_idx: int,
    *,
    expanded: bool,
    edge_basis_outputs: list[list[str]] | None = None,
) -> str:
    if edge_basis_outputs is not None:
        return edge_basis_outputs[electron_idx][output_idx]
    item = mkan["outputs"][output_idx]
    key = "expanded_formula" if expanded else "symbol"
    return _replace_e_symbol(item[key], electron_idx)


def _orbital_formula(
    mkan: dict[str, Any],
    channel_idx: int,
    electron_idx: int,
    complex_output: bool,
    *,
    expanded: bool,
    edge_basis_outputs: list[list[str]] | None = None,
) -> str:
    if complex_output:
        real_part = _mkan_output_formula(
            mkan,
            2 * channel_idx,
            electron_idx,
            expanded=expanded,
            edge_basis_outputs=edge_basis_outputs,
        )
        imag_part = _mkan_output_formula(
            mkan,
            2 * channel_idx + 1,
            electron_idx,
            expanded=expanded,
            edge_basis_outputs=edge_basis_outputs,
        )
        return f"(({real_part}) + 1j*({imag_part}))"
    return _mkan_output_formula(
        mkan,
        channel_idx,
        electron_idx,
        expanded=expanded,
        edge_basis_outputs=edge_basis_outputs,
    )


def _envelope_ref(
    envelope_info: dict[str, Any],
    block_label: str,
    row_idx: int,
    channel_idx: int,
    *,
    inline: bool = False,
    global_electron: int | None = None,
    atoms: list[dict[str, Any]] | None = None,
) -> str:
    if not envelope_info.get("enabled", False):
        return "1"
    if inline:
        symbol = f"E_{block_label}(a,{channel_idx})"
        for item in envelope_info.get("definitions", []):
            if item.get("symbol") != symbol:
                continue
            formula = item["formula"]
            if global_electron is not None and atoms is not None:
                for atom_idx in range(len(atoms)):
                    formula = formula.replace(
                        f"r_{block_label}[a,A{atom_idx}]",
                        _feature_formula(f"r_ae[{atom_idx}]", global_electron, atoms),
                    )
            return formula
    return f"E_{block_label}({row_idx},{channel_idx})"


def _build_direct_wavefunction_formula(
    *,
    cfg: ml_collections.ConfigDict,
    electrons: tuple[int, ...],
    ndeterminants: int,
    nelectrons: int,
    mkan: dict[str, Any],
    envelope_info: dict[str, Any],
    jastrow_info: dict[str, Any],
    determinant_info: dict[str, Any],
    expanded_mkan: bool,
    edge_basis_outputs: list[list[str]] | None = None,
    inline_envelope: bool = False,
    atoms: list[dict[str, Any]] | None = None,
) -> str:
    complex_output = bool(cfg.get("complex_output", False))
    full_det = bool(cfg.get("full_det", True))
    use_weights = bool(determinant_info["determinant_weights_enabled"])
    det_weights = determinant_info.get("determinant_weights", None)
    blocks = _active_spin_blocks(electrons)

    determinant_terms = []
    if full_det:
        row_specs = []
        for block in blocks:
            for local_row, global_electron in enumerate(block["global_electrons"]):
                row_specs.append((block, local_row, global_electron))
        for det_idx in range(ndeterminants):
            matrix = []
            for block, local_row, global_electron in row_specs:
                row = []
                for col_idx in range(nelectrons):
                    channel_idx = det_idx * nelectrons + col_idx
                    orbital = _orbital_formula(
                        mkan,
                        channel_idx,
                        global_electron,
                        complex_output,
                        expanded=expanded_mkan,
                        edge_basis_outputs=edge_basis_outputs,
                    )
                    env = _envelope_ref(
                        envelope_info,
                        block["label"],
                        local_row,
                        channel_idx,
                        inline=inline_envelope,
                        global_electron=global_electron,
                        atoms=atoms,
                    )
                    row.append(f"({orbital})*({env})")
                matrix.append(row)
            determinant_terms.append(_det_formula(matrix))
    else:
        starts = np.cumsum(
            (0, *[ndeterminants * int(spin) for spin in electrons[:-1]])
        ).astype(int)
        for det_idx in range(ndeterminants):
            block_dets = []
            for block in blocks:
                spin_idx = int(block["spin_index"])
                spin_size = int(block["size"])
                start = int(starts[spin_idx])
                matrix = []
                for local_row, global_electron in enumerate(block["global_electrons"]):
                    row = []
                    for col_idx in range(spin_size):
                        local_channel = det_idx * spin_size + col_idx
                        global_channel = start + local_channel
                        orbital = _orbital_formula(
                            mkan,
                            global_channel,
                            global_electron,
                            complex_output,
                            expanded=expanded_mkan,
                            edge_basis_outputs=edge_basis_outputs,
                        )
                        env = _envelope_ref(
                            envelope_info,
                            block["label"],
                            local_row,
                            local_channel,
                            inline=inline_envelope,
                            global_electron=global_electron,
                            atoms=atoms,
                        )
                        row.append(f"({orbital})*({env})")
                    matrix.append(row)
                block_dets.append(_det_formula(matrix))
            determinant_terms.append(_product_terms(block_dets))

    weighted_terms = []
    for det_idx, det_expr in enumerate(determinant_terms):
        if ndeterminants > 1 and use_weights:
            if det_weights is None:
                weight = f"w_{det_idx}"
            else:
                weight_arr = np.asarray(det_weights).reshape(-1)
                weight = _format_float(weight_arr[det_idx])
            weighted_terms.append(f"({weight})*({det_expr})")
        else:
            weighted_terms.append(det_expr)
    determinant_expr = _sum_terms(weighted_terms)

    jastrow_expr = jastrow_info.get("expanded_formula", "J(R)")
    if jastrow_expr == "0":
        return determinant_expr
    return f"exp({jastrow_expr})*({determinant_expr})"


def _export_hard_wavefunction_formula(
    *,
    out_dir: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    atoms: list[dict[str, Any]],
    electrons: tuple[int, ...],
    coordinates: dict[str, str],
    feature_formula: str,
    mkan: dict[str, Any],
    envelope_info: dict[str, Any],
    jastrow_info: dict[str, Any],
    determinant_info: dict[str, Any],
    hard_edge_payload: dict[str, Any],
    direct_formula_expanded_mkan: str,
    precision: int,
) -> tuple[Path, Path]:
    payload = {
        "checkpoint": str(checkpoint_path),
        "stage": checkpoint.get("stage"),
        "step": checkpoint.get("step"),
        "hard_edge_rounded_precision": int(precision),
        "exactness": (
            "Hard symbolic network formula with finite-decimal checkpoint "
            "parameters. MKAN edge basis symbols are expanded in "
            "edge_piecewise_expressions.json; determinant, envelope, and Jastrow "
            "follow the training computation graph."
        ),
        "indicator_convention": hard_edge_payload.get("indicator_convention"),
        "coordinates": coordinates,
        "system": {
            "atoms": atoms,
            "electrons": electrons,
            "nelectrons": int(sum(electrons)),
        },
        "input_features": {
            "formula": feature_formula,
            "symbols": mkan["input_symbols"],
        },
        "hard_edge_functions_file": "edge_piecewise_expressions.json",
        "hard_edge_functions": hard_edge_payload,
        "mkan_recursion": mkan,
        "envelope": envelope_info,
        "jastrow": jastrow_info,
        "determinant": determinant_info,
        "final_formula_using_hard_edges": direct_formula_expanded_mkan,
    }

    md_lines = [
        "# Hard-Inlined Wavefunction Formula",
        "",
        f"checkpoint: `{checkpoint_path}`",
        f"stage: `{checkpoint.get('stage')}`",
        f"step: `{checkpoint.get('step')}`",
        f"hard edge numeric precision: `{precision}` significant digits",
        "",
        "This is the symbolic computation chain obtained from the trained network itself.",
        "It does not assume a target analytic solution. The only simplification is finite-decimal formatting.",
        "",
        "Convention: `I(a <= x < b)` is 1 on the interval `[a,b)` and 0 outside.",
        "",
        "## Inputs",
        "",
        f"- atoms: `{atoms}`",
        f"- electrons: `{electrons}`",
        f"- `{coordinates['electron_atom_distance']}`",
        f"- feature rule: `{feature_formula}`",
        "",
    ]
    for item in mkan["input_symbols"]:
        md_lines.append(f"- `{item['symbol']} = {item['feature']}`")

    edge_count = sum(len(layer["edges"]) for layer in hard_edge_payload["layers"])
    md_lines.extend(
        [
            "",
            "## Hard Edge Functions",
            "",
            f"The `{edge_count}` edge functions are hard-expanded in `edge_piecewise_expressions.md/json`.",
            "They contain no `B_j,k(...)` placeholders; B-splines are written with knot values and interval indicators.",
            "",
            "## MKAN Recursion",
            "",
        ]
    )
    for layer in mkan["layers"]:
        md_lines.extend([f"### Layer {layer['layer']}", ""])
        for item in layer["pre_subnodes"]:
            md_lines.append(f"- `{item['symbol']} = {item['formula']}`")
        for item in layer["subnodes"]:
            md_lines.append(f"- `{item['symbol']} = {item['formula']}`")
        for item in layer["nodes"]:
            md_lines.append(f"- `{item['symbol']} = {item['formula']}`")
        md_lines.append("")

    md_lines.extend(["MKAN outputs:", ""])
    for item in mkan["outputs"]:
        md_lines.append(f"- `{item['symbol']} = {item['definition']}`")

    md_lines.extend(
        [
            "",
            "## Envelope And Jastrow",
            "",
            f"- envelope enabled: `{envelope_info['enabled']}`",
            f"- envelope formula: `{envelope_info['formula']}`",
            f"- Jastrow expanded: `{jastrow_info.get('expanded_formula', 'J(R)')}`",
            "",
        ]
    )
    if envelope_info.get("definitions"):
        md_lines.append("Envelope entries:")
        for item in envelope_info["definitions"]:
            md_lines.append(f"- `{item['symbol']} = {item['formula']}`")
        md_lines.append("")
    md_lines.extend(
        [
            "## Determinant Assembly",
            "",
            f"- `{determinant_info['orbital_channel_formula']}`",
            f"- `{determinant_info['determinant_formula']}`",
            "",
            "## Final Wavefunction",
            "",
            "Use the hard edge definitions from `edge_piecewise_expressions.md/json`, then apply the MKAN recursion above:",
            "",
            "```text",
            f"Psi(R) = {direct_formula_expanded_mkan}",
            "```",
            "",
        ]
    )

    json_path = out_dir / "wavefunction_formula_hard_inlined.json"
    md_path = out_dir / "wavefunction_formula_hard_inlined.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2))
    md_path.write_text("\n".join(md_lines))
    return json_path, md_path


def _export_wavefunction_formula(
    *,
    model: MultKAN,
    cfg: ml_collections.ConfigDict,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    spec: dict[str, Any],
    feature_names: list[str],
    out_dir: Path,
    hard_precision: int = 6,
) -> tuple[Path, ...]:
    electrons = tuple(int(v) for v in cfg.system.electrons)
    nelectrons = int(spec["nelectrons"])
    ndeterminants = int(spec["ndeterminants"])
    atoms = [
        {
            "index": atom_idx,
            "symbol": atom.symbol,
            "coords": list(atom.coords),
            "charge": float(atom.charge),
        }
        for atom_idx, atom in enumerate(cfg.system.molecule)
    ]
    coordinates = {
        "electron_coordinates": "r_i = (x_i, y_i, z_i)",
        "atom_coordinates": "R_A = (X_A, Y_A, Z_A)",
        "electron_atom_distance": "r_iA = ||r_i - R_A||",
        "electron_atom_vector": "ae_iA = r_i - R_A",
    }
    feature_formula = (
        "h_i = concat_A [r_iA, ae_iA_x, ae_iA_y, ae_iA_z], "
        "then optionally select input_id after node pruning"
    )

    mkan = _mkan_composition_payload(model, feature_names)
    envelope_info = _envelope_payload(cfg, checkpoint, electrons, ndeterminants, nelectrons)
    jastrow_info = _jastrow_payload(cfg, checkpoint, electrons)
    determinant_info = _determinant_payload(
        cfg,
        checkpoint,
        electrons,
        ndeterminants,
        nelectrons,
    )
    edge_basis_outputs = _inline_edge_basis_outputs(
        model=model,
        layer_type=spec["layer_type"],
        feature_names=feature_names,
        atoms=atoms,
        nelectrons=nelectrons,
    )
    direct_formula = _build_direct_wavefunction_formula(
        cfg=cfg,
        electrons=electrons,
        ndeterminants=ndeterminants,
        nelectrons=nelectrons,
        mkan=mkan,
        envelope_info=envelope_info,
        jastrow_info=jastrow_info,
        determinant_info=determinant_info,
        expanded_mkan=False,
    )
    direct_formula_expanded_mkan = _build_direct_wavefunction_formula(
        cfg=cfg,
        electrons=electrons,
        ndeterminants=ndeterminants,
        nelectrons=nelectrons,
        mkan=mkan,
        envelope_info=envelope_info,
        jastrow_info=jastrow_info,
        determinant_info=determinant_info,
        expanded_mkan=True,
    )
    direct_formula_expanded_mkan_inline_envelope = _build_direct_wavefunction_formula(
        cfg=cfg,
        electrons=electrons,
        ndeterminants=ndeterminants,
        nelectrons=nelectrons,
        mkan=mkan,
        envelope_info=envelope_info,
        jastrow_info=jastrow_info,
        determinant_info=determinant_info,
        expanded_mkan=True,
        inline_envelope=True,
        atoms=atoms,
    )
    direct_formula_fully_inlined = _build_direct_wavefunction_formula(
        cfg=cfg,
        electrons=electrons,
        ndeterminants=ndeterminants,
        nelectrons=nelectrons,
        mkan=mkan,
        envelope_info=envelope_info,
        jastrow_info=jastrow_info,
        determinant_info=determinant_info,
        expanded_mkan=True,
        edge_basis_outputs=edge_basis_outputs,
        inline_envelope=True,
        atoms=atoms,
    )
    hard_edges_json, hard_edges_md, hard_edge_payload = _export_hard_edge_expressions(
        model=model,
        layer_type=spec["layer_type"],
        feature_names=feature_names,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        spec=spec,
        precision=hard_precision,
    )
    hard_wavefunction_json, hard_wavefunction_md = _export_hard_wavefunction_formula(
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        atoms=atoms,
        electrons=electrons,
        coordinates=coordinates,
        feature_formula=feature_formula,
        mkan=mkan,
        envelope_info=envelope_info,
        jastrow_info=jastrow_info,
        determinant_info=determinant_info,
        hard_edge_payload=hard_edge_payload,
        direct_formula_expanded_mkan=direct_formula_expanded_mkan_inline_envelope,
        precision=hard_precision,
    )

    payload = {
        "checkpoint": str(checkpoint_path),
        "stage": checkpoint.get("stage"),
        "step": checkpoint.get("step"),
        "system": {
            "atoms": atoms,
            "electrons": electrons,
            "nelectrons": nelectrons,
        },
        "exactness": (
            "Architecture-level exact formula. MKAN edge functions phi_L*_y*_x* "
            "are defined in expressions.json; determinant, envelope, and Jastrow "
            "composition follows the training network."
        ),
        "coordinates": coordinates,
        "input_features": {
            "nfeatures": int(spec["input_dim"]),
            "feature_names": feature_names,
            "formula": feature_formula,
        },
        "mkan_composition": mkan,
        "envelope": envelope_info,
        "jastrow": jastrow_info,
        "determinant": determinant_info,
        "final_formula": direct_formula,
        "final_formula_expanded_mkan": direct_formula_expanded_mkan,
        "final_formula_expanded_mkan_inline_envelope": direct_formula_expanded_mkan_inline_envelope,
        "mkan_edge_basis_outputs": edge_basis_outputs,
        "final_formula_fully_inlined": direct_formula_fully_inlined,
        "hard_inlined_outputs": {
            "edge_piecewise_expressions_json": str(hard_edges_json),
            "edge_piecewise_expressions_md": str(hard_edges_md),
            "wavefunction_formula_hard_inlined_json": str(hard_wavefunction_json),
            "wavefunction_formula_hard_inlined_md": str(hard_wavefunction_md),
            "rounded_precision": int(hard_precision),
        },
    }

    md_lines = [
        "# Full Wavefunction Formula",
        "",
        f"checkpoint: `{checkpoint_path}`",
        f"stage: `{checkpoint.get('stage')}`",
        f"step: `{checkpoint.get('step')}`",
        "",
        "This file composes the full QMC wavefunction from the trained MKAN edge functions.",
        "The edge functions `phi_L*_y*_x*` are defined in `expressions.json` and `expressions.md`.",
        "",
        "## Coordinates And Features",
        "",
        f"- atoms: `{atoms}`",
        f"- electrons: `{electrons}`",
        f"- `{coordinates['electron_atom_distance']}`",
        f"- `{feature_formula}`",
        "",
        "## MKAN Composition",
        "",
        "Input nodes:",
    ]
    for item in mkan["input_symbols"]:
        md_lines.append(f"- `{item['symbol']} = {item['feature']}`")
    md_lines.append("")

    for layer in mkan["layers"]:
        md_lines.extend([f"### Layer {layer['layer']}", ""])
        for item in layer["pre_subnodes"]:
            md_lines.append(f"- `{item['symbol']} = {item['formula']}`")
        for item in layer["subnodes"]:
            md_lines.append(f"- `{item['symbol']} = {item['formula']}`")
        for item in layer["nodes"]:
            md_lines.append(f"- `{item['symbol']} = {item['formula']}`")
        md_lines.append("")

    md_lines.extend(
        [
            "MKAN output channels:",
            "",
        ]
    )
    for item in mkan["outputs"]:
        md_lines.append(f"- `{item['symbol']} = {item['definition']}`")
    md_lines.extend(["", "Expanded MKAN output channels:", ""])
    for item in mkan["outputs"]:
        md_lines.append(f"- `{item['symbol']} = {item['expanded_formula']}`")
    md_lines.extend(["", "Fully inlined MKAN output channels for each electron:", ""])
    for electron_idx, outputs in enumerate(edge_basis_outputs):
        md_lines.append(f"Electron {electron_idx}:")
        for output_idx, formula in enumerate(outputs):
            md_lines.append(f"- `F_{output_idx}({electron_idx}) = {formula}`")
    md_lines.extend(
        [
            "",
            "## Orbital Channels",
            "",
            f"- `{determinant_info['orbital_channel_formula']}`",
            "",
            "## Envelope",
            "",
            f"- enabled: `{envelope_info['enabled']}`",
            f"- formula: `{envelope_info['formula']}`",
            "",
        ]
    )
    if envelope_info.get("definitions"):
        md_lines.append("Envelope entries:")
        for item in envelope_info["definitions"]:
            md_lines.append(f"- `{item['symbol']} = {item['formula']}`")
        md_lines.append("")
    md_lines.extend(
        [
            "## Jastrow",
            "",
            f"- `{jastrow_info['ee_formula']}`",
            f"- `{jastrow_info['en_formula']}`",
            f"- `{jastrow_info['total_formula']}`",
            f"- expanded: `{jastrow_info.get('expanded_formula', 'J(R)')}`",
            "",
            "## Determinants",
            "",
        ]
    )
    for item in determinant_info["matrix_formulas"]:
        md_lines.append(f"- `{item['formula']}`")
    md_lines.extend(
        [
            f"- `{determinant_info['determinant_formula']}`",
            "",
            "## Final Wavefunction",
            "",
            "Compact expression with MKAN output symbols:",
            "",
            f"`Psi(R) = {direct_formula}`",
            "",
            "Expression with MKAN outputs substituted through the network recursion:",
            "",
            f"`Psi(R) = {direct_formula_expanded_mkan}`",
            "",
            "Expression with edge functions substituted into their basis expansions:",
            "",
            "```text",
            f"Psi(R) = {direct_formula_fully_inlined}",
            "```",
            "",
            "Hard symbolic edge expansion files:",
            "",
            f"- `{hard_edges_md}`",
            f"- `{hard_wavefunction_md}`",
            "",
        ]
    )

    json_path = out_dir / "wavefunction_formula.json"
    md_path = out_dir / "wavefunction_formula.md"
    simplified_path = out_dir / "wavefunction_formula_simplified.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2))
    md_path.write_text("\n".join(md_lines))
    simplified_lines = [
        "# Simplified Wavefunction Formula",
        "",
        "This is a compact, factored view of the same trained checkpoint. The full exact network expansion is in `wavefunction_formula.md` and `wavefunction_formula.json`.",
        "",
        f"checkpoint: `{checkpoint_path}`",
        f"stage: `{checkpoint.get('stage')}`",
        f"step: `{checkpoint.get('step')}`",
        "",
        "## Inputs",
        "",
        f"- electrons: `{electrons}`",
        f"- atoms: `{atoms}`",
        f"- feature rule: `{feature_formula}`",
        "",
    ]
    if nelectrons == 1 and int(spec["input_dim"]) == 4 and len(atoms) == 1:
        simplified_lines.extend(
            [
                "For a one-electron, one-atom run with four features, this corresponds to:",
                "",
                "`h = [r, x, y, z]`, where `r = sqrt(x^2 + y^2 + z^2)` after shifting by the atom coordinate.",
                "",
            ]
        )
    simplified_lines.extend(["Input symbols:", ""])
    for item in mkan["input_symbols"]:
        simplified_lines.append(f"- `{item['symbol']} = {item['feature']}`")
    simplified_lines.extend(["", "## MKAN Recursion", ""])
    for layer in mkan["layers"]:
        simplified_lines.extend([f"Layer {layer['layer']}:", ""])
        for item in layer["nodes"]:
            simplified_lines.append(f"- `{item['symbol']} = {item['formula']}`")
        simplified_lines.append("")
    simplified_lines.extend(["MKAN outputs:", ""])
    for item in mkan["outputs"]:
        simplified_lines.append(f"- `{item['symbol']} = {item['definition']}`")
    simplified_lines.extend(
        [
            "",
            "## Envelope And Jastrow",
            "",
            f"- envelope enabled: `{envelope_info['enabled']}`",
            f"- envelope formula: `{envelope_info['formula']}`",
            f"- Jastrow total: `{jastrow_info['total_formula']}`",
            f"- Jastrow expanded: `{jastrow_info.get('expanded_formula', 'J(R)')}`",
            "",
            "## Determinant Assembly",
            "",
            f"- `{determinant_info['orbital_channel_formula']}`",
            f"- `{determinant_info['determinant_formula']}`",
            "",
            "## Final Wavefunction",
            "",
            "Compact:",
            "",
            f"`Psi(R) = {direct_formula}`",
            "",
            "With MKAN recursion substituted:",
            "",
            "```text",
            f"Psi(R) = {direct_formula_expanded_mkan}",
            "```",
            "",
            "For the fully inlined edge-basis expression, use `final_formula_fully_inlined` in `wavefunction_formula.json` or the final section of `wavefunction_formula.md`.",
            f"For the hard-expanded finite-decimal edge expressions, use `{hard_wavefunction_md.name}` and `{hard_edges_md.name}`.",
            "",
        ]
    )
    simplified_path.write_text("\n".join(simplified_lines))
    return (
        json_path,
        md_path,
        simplified_path,
        hard_wavefunction_json,
        hard_wavefunction_md,
        hard_edges_json,
        hard_edges_md,
    )


def _export_edge_samples(cache: dict[str, Any], out_dir: Path) -> Path:
    arrays = {}
    for layer_idx, (acts, postacts) in enumerate(zip(cache["acts"][:-1], cache["postacts"])):
        arrays[f"layer_{layer_idx}_x"] = np.asarray(acts)
        arrays[f"layer_{layer_idx}_postacts"] = np.asarray(postacts)
    samples_path = out_dir / "edge_samples.npz"
    np.savez_compressed(samples_path, **arrays)
    return samples_path


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
    parser.add_argument(
        "--md-coeff-limit",
        type=int,
        default=12,
        help="Maximum number of coefficients to show per edge in expressions.md. Full coefficients are always in expressions.json.",
    )
    parser.add_argument(
        "--hard-formula-precision",
        type=int,
        default=6,
        help="Significant digits used in finite-decimal hard-expanded symbolic formulas.",
    )
    parser.add_argument(
        "--no-wavefunction-analysis",
        action="store_true",
        help="Skip one-electron wavefunction plots and compact expression discovery.",
    )
    parser.add_argument(
        "--wavefunction-r-max",
        type=float,
        default=6.0,
        help="Maximum radius for one-electron wavefunction plots/discovery.",
    )
    parser.add_argument(
        "--wavefunction-radial-points",
        type=int,
        default=600,
        help="Number of radial points for one-electron wavefunction plots/discovery.",
    )
    parser.add_argument(
        "--wavefunction-plane-extent",
        type=float,
        default=4.0,
        help="Half-width of the z=0 one-electron wavefunction slice.",
    )
    parser.add_argument(
        "--wavefunction-grid-size",
        type=int,
        default=181,
        help="Grid size for the z=0 one-electron wavefunction slice.",
    )
    parser.add_argument(
        "--wavefunction-discovery-directions",
        type=int,
        default=96,
        help="Number of angular directions used for one-electron radiality discovery.",
    )
    parser.add_argument(
        "--wavefunction-discovery-max-terms",
        type=int,
        default=3,
        help="Maximum number of sparse terms used in the compact discovery report.",
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
    features = _make_features(
        positions,
        atoms,
        tuple(cfg.system.electrons),
        spec["input_dim"],
        sample_size,
        spec["orbital_feature_mode"],
    )

    cache = model.get_act(features)
    attribution = model.attribute(cache)
    feature_score = np.asarray(attribution["feature_score"])
    feature_names = _feature_names(
        spec["natoms"],
        spec["input_dim"],
        spec["orbital_feature_mode"],
    )
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
    expressions_json, expressions_md = _export_expressions(
        model=model,
        layer_type=spec["layer_type"],
        feature_names=feature_names,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        spec=spec,
        md_coeff_limit=max(0, args.md_coeff_limit),
    )
    wavefunction_paths = _export_wavefunction_formula(
        model=model,
        cfg=cfg,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        spec=spec,
        feature_names=feature_names,
        out_dir=out_dir,
        hard_precision=max(1, args.hard_formula_precision),
    )
    (
        wavefunction_json,
        wavefunction_md,
        wavefunction_simplified,
        hard_wavefunction_json,
        hard_wavefunction_md,
        hard_edges_json,
        hard_edges_md,
    ) = wavefunction_paths
    edge_samples = _export_edge_samples(cache, out_dir)

    import matplotlib

    matplotlib.use("Agg")
    model.plot(cache, folder=str(out_dir / "edge_plots"), metric=args.metric, in_vars=feature_names)

    wavefunction_analysis_outputs = {}
    if not args.no_wavefunction_analysis:
        nelectrons = int(sum(cfg.system.electrons))
        if nelectrons == 1:
            try:
                from discover_wavefunction_form import generate_wavefunction_discovery
                from plot_wavefunction import generate_wavefunction_plots

                wavefunction_plot_dir = out_dir / "wavefunction_plots"
                wavefunction_analysis_outputs.update(
                    generate_wavefunction_plots(
                        run_dir=run_dir,
                        checkpoint=checkpoint_path,
                        out_dir=wavefunction_plot_dir,
                        r_max=args.wavefunction_r_max,
                        radial_points=args.wavefunction_radial_points,
                        plane_extent=args.wavefunction_plane_extent,
                        grid_size=args.wavefunction_grid_size,
                        reference_hydrogen_1s=False,
                        write_reports=True,
                    )
                )
                wavefunction_analysis_outputs.update(
                    generate_wavefunction_discovery(
                        run_dir=run_dir,
                        checkpoint=checkpoint_path,
                        out_dir=wavefunction_plot_dir,
                        r_max=args.wavefunction_r_max,
                        radial_points=args.wavefunction_radial_points,
                        directions=args.wavefunction_discovery_directions,
                        max_terms=args.wavefunction_discovery_max_terms,
                    )
                )
            except Exception as exc:
                print(f"WARNING: skipped one-electron wavefunction analysis: {exc}")
        else:
            print(
                "Skipped one-electron wavefunction plots/discovery: "
                f"this run has {nelectrons} electrons."
            )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Features shape: {features.shape}")
    print(f"Wrote feature scores: {out_dir / 'feature_score.json'}")
    print(f"Wrote expressions JSON: {expressions_json}")
    print(f"Wrote expressions Markdown: {expressions_md}")
    print(f"Wrote wavefunction formula JSON: {wavefunction_json}")
    print(f"Wrote wavefunction formula Markdown: {wavefunction_md}")
    print(f"Wrote simplified wavefunction formula: {wavefunction_simplified}")
    print(f"Wrote hard-inlined wavefunction JSON: {hard_wavefunction_json}")
    print(f"Wrote hard-inlined wavefunction Markdown: {hard_wavefunction_md}")
    print(f"Wrote hard edge expressions JSON: {hard_edges_json}")
    print(f"Wrote hard edge expressions Markdown: {hard_edges_md}")
    print(f"Wrote edge samples: {edge_samples}")
    print(f"Wrote edge plots: {out_dir / 'edge_plots'}")
    for label, path in wavefunction_analysis_outputs.items():
        print(f"Wrote {label}: {path}")
    print("Top features:")
    for idx in ranking[: min(10, len(ranking))]:
        print(f"  {feature_names[idx]}: {feature_score[idx]:.6g}")


if __name__ == "__main__":
    main()
