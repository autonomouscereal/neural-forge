#!/usr/bin/env python3
"""Convert a measured curve-form H3 coreset to one native charted-block container.

No training, decoder, parent runtime or checkpoint installation occurs here.
Frozen attention/norm/adaLN tensors and the source curve/RoPE coefficients are
retained; only the already-folded MLP is replaced. Actual captured modulation
must reproduce before anything is written. H3 source coordinates are required
at runtime, separately from the trunk's native THWR packet coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from chimera_omni.charted import H3_FP32_PARAMETERS, ChartedBlockSpec
from chimera_omni.importing.safetensors_io import read_safetensors_file
from chimera_omni.modeling.charted import H3CurveModulation
from chimera_omni.importing.h3_modulation import normalize_h3_coefficients


SCHEMA = "chimera_omni.composite.charted_block.v1"
SOURCE_CODE_SHA256 = "882c280b05aa60cd17f231e5e2389b921b52880fb3b4e23574c0aba5f2bd5024"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tensors(path: Path, names=None):
    views = read_safetensors_file(path, names)
    if names is not None and set(views) != set(names):
        raise ValueError(f"{path}: missing required tensors {sorted(set(names) - set(views))}")
    return {name: view.torch() for name, view in views.items()}, {name: view.sha256() for name, view in views.items()}


def relative_error(actual, reference):
    return float((actual.double() - reference.double()).norm() / reference.double().norm().clamp_min(1e-30))


def captured_time_reference(paths: list[Path], width: int, curve_dim: int):
    sigmas, mods, curves, provenance = [], [], [], []
    names = ("sigma_video", "sigma_audio", "temb", "adaln")
    for path in paths:
        tensors, digests = read_tensors(path, names)
        sv, sa = tensors["sigma_video"].float(), tensors["sigma_audio"].float()
        if sv.ndim != 1 or sa.shape != sv.shape or not sv.numel():
            raise ValueError("capture sigmas must be non-empty [forwards] vectors")
        # The plain AV source builds unique_t = sorted({1-sigma_v, 1-sigma_a}).
        pairs = torch.stack((sv, sa), dim=1).sort(dim=1, descending=True).values
        if bool((pairs[:, 0] == pairs[:, 1]).any()):
            raise ValueError("this converter requires the captured two-time plain AV layout")
        forwards = len(sv)
        if tensors["temb"].shape != (forwards, 2, curve_dim):
            raise ValueError("capture temb does not match the source two-time curve layout")
        if tensors["adaln"].shape != (forwards, 6, 6, width):
            raise ValueError("capture adaLN is not [forwards,6,2*3,width]")
        sigmas.append(pairs.reshape(-1))
        curves.append(tensors["temb"].float().reshape(-1, curve_dim))
        mods.append(tensors["adaln"].float().permute(0, 2, 1, 3).reshape(-1, 3, 6, width))
        provenance.append({"path": str(path.resolve()), "selected_tensor_sha256": digests})
    sigma, modulation, curve = torch.cat(sigmas), torch.cat(mods), torch.cat(curves)
    if not all(bool(torch.isfinite(t).all()) for t in (sigma, modulation, curve)):
        raise ValueError("non-finite time reference")
    if bool(((sigma < 0) | (sigma > 1)).any()):
        raise ValueError("capture sigma outside [0,1]")
    # Repeated forwards at one sigma add no information; verify before deduplicating.
    unique, inverse = torch.unique(sigma, sorted=True, return_inverse=True)
    keep = torch.stack([torch.nonzero(inverse == i).flatten()[0] for i in range(len(unique))])
    if not torch.equal(modulation, modulation[keep][inverse]) or not torch.equal(curve, curve[keep][inverse]):
        raise ValueError("repeated capture sigma has different modulation/curve; do not collapse its evidence")
    return {"sigma": unique, "modulation": modulation[keep], "curve": curve[keep]}, provenance


def convert(*, block_state: Path, folded_mlp: Path, source_weights: Path, coreset_receipt: Path,
            captures: list[Path], out: Path, block_index: int = 25, source_depth: int = 50,
            heads: int = 56, head_dim: int = 128, charts: Path | None = None,
            time_rel_tol: float = 1e-5):
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be empty; historical containers are never overwritten")
    inputs = [block_state, folded_mlp, source_weights, coreset_receipt, *captures]
    if charts is not None:
        inputs.append(charts)
    if any(out.resolve() == p.resolve() or out.resolve() in p.resolve().parents for p in inputs):
        raise ValueError("output overlaps an input")
    if not captures or not (0 < time_rel_tol <= 1e-5):
        raise ValueError("captured references and a tolerance no looser than 1e-5 are required")
    if source_depth <= 1 or not 0 <= block_index < source_depth:
        raise ValueError("invalid source block/depth")
    receipt = json.loads(coreset_receipt.read_text(encoding="utf-8"))
    if int(receipt.get("block", -1)) != block_index:
        raise ValueError("coreset receipt names a different block")
    expected_mlp = receipt.get("child", {}).get("file_sha256")
    expected_block = receipt.get("capture", {}).get("block_weights_sha256")
    if not expected_mlp or file_sha(folded_mlp) != expected_mlp:
        raise ValueError("folded MLP file does not match the coreset receipt")
    if not expected_block or file_sha(block_state) != expected_block:
        raise ValueError("frozen block file does not match the coreset receipt")

    prefix = f"blocks.{block_index}."
    frozen, frozen_digests = read_tensors(block_state)
    if not all(name.startswith(prefix) for name in frozen):
        raise ValueError("block-state file contains tensors outside the requested block")
    frozen = {name[len(prefix):]: value for name, value in frozen.items()}
    mlp, mlp_digests = read_tensors(folded_mlp, (prefix + "mlp.fc1.weight", prefix + "mlp.fc2.weight"))
    globals_, global_digests = read_tensors(source_weights, ("adaln_t_table", "rope.inv_freq"))
    mapping = {"norm1.weight": "attn_norm.weight", "norm2.weight": "mlp_norm.weight",
               "attn.out_proj.weight": "attn.o_proj.weight", "attn.q_norm.weight": "attn.q_norm.weight",
               "attn.k_norm.weight": "attn.k_norm.weight", "adaln_proj.linear.weight": "h3_time.proj.weight",
               "adaln_proj.linear.bias": "h3_time.proj.bias"}
    allowed = set(mapping) | {"attn.qkv_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"}
    if set(frozen) != allowed:
        raise ValueError(f"unexpected source block tensor set: {sorted(set(frozen) ^ allowed)}")
    tensors = {target: frozen[source] for source, target in mapping.items()}
    width = int(frozen["norm1.weight"].numel())
    qkv = frozen["attn.qkv_proj.weight"]
    if qkv.shape != (3 * heads * head_dim, width):
        raise ValueError("fused source QKV geometry does not match declared MHA")
    for name, tensor in zip(("q", "k", "v"), qkv.chunk(3, dim=0)):
        tensors[f"attn.{name}_proj.weight"] = tensor
    fc1, fc2 = mlp[prefix + "mlp.fc1.weight"], mlp[prefix + "mlp.fc2.weight"]
    if fc1.ndim != 2 or fc1.shape[0] % 2 or fc1.shape[1] != width:
        raise ValueError("folded H3 fc1 must have fused gate/up rows")
    keep = fc1.shape[0] // 2
    if fc2.shape != (width, keep):
        raise ValueError("folded H3 fc2 has wrong geometry")
    tensors["mlp.gate_proj.weight"], tensors["mlp.up_proj.weight"] = fc1.chunk(2, dim=0)
    tensors["mlp.down_proj.weight"] = fc2
    tensors["h3_time.curve_table"] = globals_["adaln_t_table"]
    tensors["attn.rope_inv_freq"] = globals_["rope.inv_freq"]
    table = globals_["adaln_t_table"]
    if table.ndim != 2 or globals_["rope.inv_freq"].ndim != 1:
        raise ValueError("source curve/RoPE coefficients have invalid ranks")
    spec = ChartedBlockSpec(layer_index=0, source="h3", component="h3.dit.block.mlp", type_ids=(2, 3),
        width=width, heads=heads, kv_heads=heads, head_dim=head_dim, mlp_width=keep,
        modulation="h3_curve6", norm_eps=1e-5, h3_curve_grid=table.shape[0], h3_curve_dim=table.shape[1],
        h3_rope_freqs=globals_["rope.inv_freq"].numel(), rung="measured-coreset",
        source_depth_fraction=block_index / (source_depth - 1),
        metadata={"source_position_schema": "h3_thw_v1", "context": "text read-only plus AV; no context tower",
                  "sigma_contract": "actual per-type sigma; source shifts video12/audio3", "source_code_sha256": SOURCE_CODE_SHA256})
    expected = {k: v for k, v in spec.parameter_shapes("", 1).items() if not k.startswith("chart_")}
    if {k: tuple(v.shape) for k, v in tensors.items()} != expected:
        raise ValueError("converted tensor geometry does not match the H3 native spec")
    # Native curve-form constructor requests FP32 projections. The current
    # pruned source stores weights/biases in F16; widen their exact represented
    # values, retaining raw block pins and an explicit before/after value receipt.
    tensors, coefficient_widening = normalize_h3_coefficients(
        tensors, {name: expected[name] for name in H3_FP32_PARAMETERS},
        {"h3_time.proj.weight", "h3_time.proj.bias"})
    if any(tensors[name].dtype != torch.float32 for name in H3_FP32_PARAMETERS):
        raise ValueError("H3 curve, time projection and RoPE source coefficients must be stored in FP32")
    if not all(bool(torch.isfinite(t).all()) for t in tensors.values()):
        raise ValueError("converted source tensors contain non-finite values")
    parent_mlp = sum(frozen[k].numel() for k in ("mlp.fc1.weight", "mlp.fc2.weight"))
    child_mlp = sum(t.numel() for k, t in tensors.items() if k.startswith("mlp."))
    if child_mlp * 4 > parent_mlp:
        raise ValueError("stored coreset does not meet the 4x MLP component requirement")
    counts = receipt.get("header_counts", {})
    if counts.get("parent_mlp_parameters") != parent_mlp or counts.get("child_mlp_parameters") != child_mlp:
        raise ValueError("MLP header counts disagree with the coreset receipt")

    reference, reference_sources = captured_time_reference(captures, width, table.shape[1])
    time_path = H3CurveModulation(spec)
    time_path.load_state_dict({name[len("h3_time."):]: value.float() for name, value in tensors.items()
                              if name.startswith("h3_time.")}, strict=True)
    with torch.no_grad():
        curve_error = relative_error(time_path.curve(reference["sigma"]), reference["curve"])
        modulation_error = relative_error(time_path(reference["sigma"]), reference["modulation"])
    if max(curve_error, modulation_error) > time_rel_tol:
        raise ValueError(f"source time path does not reproduce capture: curve={curve_error:g}, modulation={modulation_error:g}")
    chart_state = None
    if charts is not None:
        chart_state, _ = read_tensors(charts)
        required = {"chart_in.weight", "chart_in.bias", "chart_out.weight"}
        if set(chart_state) != required or chart_state["chart_in.weight"].ndim != 2:
            raise ValueError("charts must contain exactly the native residual-chart keys")
        hidden = chart_state["chart_in.weight"].shape[1]
        shapes = {k: v for k, v in spec.parameter_shapes("", hidden).items() if k.startswith("chart_")}
        if {k: tuple(v.shape) for k, v in chart_state.items()} != shapes:
            raise ValueError("chart shapes disagree with source/trunk widths")
        if not all(bool(torch.isfinite(t).all()) for t in chart_state.values()):
            raise ValueError("non-finite charts")
    body = {"schema": SCHEMA, "spec": {k: v for k, v in spec.to_manifest().items() if k != "layer_index"},
        "provenance": {"converter": "c16_convert_h3.v1", "converter_sha256": file_sha(Path(__file__)),
            "coreset_receipt": str(coreset_receipt.resolve()), "coreset_receipt_sha256": file_sha(coreset_receipt),
            "block_state_sha256": expected_block, "folded_mlp_sha256": expected_mlp,
            "source_weights": str(source_weights.resolve()), "source_globals_tensor_sha256": global_digests,
            "source_globals_scope": "selected tensors only; full source-file identity is not reverified here",
            "frozen_tensor_sha256": frozen_digests, "folded_tensor_sha256": mlp_digests,
            "h3_native_modulation_fp32_policy": coefficient_widening,
            "capture_references": reference_sources},
        "accounting": {"parent_mlp_parameters": parent_mlp, "child_mlp_parameters": child_mlp,
            "parent_block_parameters": sum(t.numel() for t in frozen.values()),
            "child_block_parameters": sum(t.numel() for k, t in tensors.items() if k not in
                                          {"h3_time.curve_table", "attn.rope_inv_freq"}),
            "retained_shared_coefficients": table.numel() + globals_["rope.inv_freq"].numel()},
        "holdout": receipt.get("measured", {}).get("holdout_scores_replace", {}),
        "capture_deviations": receipt.get("capture_deviations", []),
        "time_reference": {"file": "time_reference.safetensors", "sigma": reference["sigma"].tolist(),
            "modality_order": ["video", "text", "audio"], "conversion_rel_l2": modulation_error,
            "curve_rel_l2": curve_error, "tolerance": time_rel_tol, "verified": True},
        "runtime_requirements": {"source_positions": "explicit h3_thw_v1 coordinates for text and AV rows",
            "coordinate_inference": "forbidden: native thwr_v1 values are not H3 source coordinates",
            "charts": "provided" if chart_state is not None else "paired host/source tap rows still required",
            "capability_claim": "none: this converts one measured coreset, not an all-source child or media proof"},
        "tensor_sha256": {}}
    out.mkdir(parents=True, exist_ok=True)
    outputs = {"charted_block.safetensors": tensors, "time_reference.safetensors": reference}
    if chart_state is not None:
        outputs["charts.safetensors"] = chart_state
    for name, state in outputs.items():
        save_file({k: v.contiguous() for k, v in state.items()}, str(out / name))
        body["tensor_sha256"][name] = {k: v.sha256() for k, v in read_safetensors_file(out / name).items()}
    (out / "charted_block.json").write_bytes((json.dumps(body, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("block-state", "folded-mlp", "source-weights", "coreset-receipt", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--block-index", type=int, default=25)
    parser.add_argument("--source-depth", type=int, default=50)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--charts", type=Path)
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    main()
