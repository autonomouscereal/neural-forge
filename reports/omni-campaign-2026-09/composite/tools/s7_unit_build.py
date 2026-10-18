#!/usr/bin/env python3
"""s7_unit_build.py -- convert existing per-source media artefacts into c16_install_media units.

Lane E4 (S7 media-install readiness), Claude, 2026-10-09. Every unit is written under
`--units-root` (default `$OC/composite/s7/units`) as a `chimera_omni.composite.charted_block.v1`
container (COMPOSITE_S6_INPUT_FORMATS.md section 3) that `c16_install_media.py` >= 1.2.0 loads.
Source artefacts are READ ONLY; nothing under Codex's runtime or Lexar output dirs is changed.

  ace   ACE-Step DiT block 12 4x coreset ($OC/media/out/c16_L12_v4) through c16's own
        convert_ace_run (both timestep paths, captured temb reference). Its ridge chart was
        fitted on a 128-wide toy host, so it is kept as charts_host128.reference.safetensors
        and the unit ships WITHOUT charts.safetensors (a 5,120-wide chart must be fitted).
  h3    Codex's MiniMax-H3 block-25 h3_curve6 charted block. Both files are symlinked under
        their canonical names (charted_block.safetensors, time_reference.safetensors); the
        h3_curve6 gate (rebuilt time_path vs the captured [n,3,6,width] modulation, 1e-5) runs
        here AND in c16_install_media >= 1.2.0, which also keeps the H3 FP32 coefficients wide.
        (1.1.0 of this tool shipped the reference as h3_time_reference.safetensors.)
  qimg  Qwen-Image MMDiT block 30 image stream from Codex's ownerpolicy-v4 candidate
        (img_mlp keep 3071 of 12288). FP8_E4M3 tensors are widened to BF16 (exact: every
        e4m3 value is a bf16 value); the text stream (add_* projections, txt_mlp, txt_mod)
        is dropped like ACE's cross-attention. time_embed = time_text_embed.timestep_embedder
        linear_1/linear_2, time_proj = img_mod.1, scale_shift_table = 0. Time reference:
        `--time-reference` = the FP32 parent time path (m3_qimg_time_reference_fp32.py, run in
        the ComfyUI image with the parent's own code) at the capture sigmas; without it, the
        capture's BF16 img_modulation (rounding floor ~2e-3, recorded, not forced).
  hy3d  Hunyuan3D-2.1 DiT block 10 (dense, no skip input) from the parent F16 weights plus
        Codex's dense child MLP (fc1/fc2 keep 2047 of 8192). Self-attention Q/K/V are
        re-packed from the source's cat(q,k,v)->view(heads,3*head_dim) head split into the
        standard per-head layout; zero q/k/v biases are written because the spec's attn_bias
        is one flag (out_proj carries the real bias). The parent's time token (row 0 of the
        self-attention sequence, x = cat([t_embedder(t), x_embedder(z)])) is carried as an owned
        type-7 runtime row (runtime_requirements.hy3d_time_token_row) with an FP32
        t_embedder reference file; cross-attention (attn2, norm2) is dropped (structural:
        the charted kind has no conditioning input; owner decision recorded in the unit).
  qvl   Qwen3-VL-32B ViT block 13 from Codex's qvl32-1075 child; the parent MLP width is read
        from `--parent`'s header (not assumed); type ids default [6] (ocr) so it never runs on
        QIMG's image rows. C11 rows are built by m3_c11_qvl_rows.py.

1.2.0: lane M3 (Claude, 2026-10-10) -- the four changes above (h3 canonical reference name,
qimg --time-reference, hy3d time-token row, qvl --parent/--type-ids). ace is unchanged.
1.3.0: PREP-B (Claude, 2026-10-10) -- hy3d: the time-token contract now names the tools that implement
the pairing (s7_pairs_parent >= 1.1.0 keeps row 0, s7_host_rows >= 1.2.0 emits/validates the trunk time
row, s7_chart_fit >= 1.1.0 refuses rows without it), and `--hy3d-block0-capture` records how the FP32
t_embedder reference reproduces the parent's captured block-0 time token (diagnostic, never a gate).
Block tensors are unchanged from 1.2.0.

Every unit gets `s7_unit` in charted_block.json (status, 4x component check, deviations).
Run with PYTHONPATH=<chimera tree>/src (tree_ef1ffb9: the only server tree with h3_curve6).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

TOOL_NAME = "s7_unit_build"
TOOL_VERSION = "1.3.0"
OC = Path("/home/cereal/omni-campaign")
HERE = Path(__file__).resolve().parent


def _c16():
    spec = importlib.util.spec_from_file_location("c16_install_media", HERE / "c16_install_media.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["c16_install_media"] = module
    spec.loader.exec_module(module)
    return module


C16 = _c16()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, body: dict[str, Any]) -> str:
    text = json.dumps(body, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def table(path: Path) -> dict[str, str]:
    return {name: C16.view_hash(view) for name, view in sorted(C16.read_single(path).items())}


def component_check(accounting: dict[str, Any]) -> dict[str, Any]:
    parent = int(accounting["parent_mlp_parameters"])
    child = int(accounting["child_mlp_parameters"])
    ratio = child / parent
    block = None
    if accounting.get("parent_block_parameters") and accounting.get("child_block_parameters"):
        block = int(accounting["child_block_parameters"]) / int(accounting["parent_block_parameters"])
    return {
        "rule": "c16 components(): child MLP / parent MLP <= 0.25 registers as --component",
        "mlp_ratio": ratio,
        "mlp_compression_x": parent / child,
        "meets_4x_component": ratio <= 0.25,
        "whole_block_ratio_recorded_only": block,
    }


def time_check(spec_fields: dict[str, Any], tensors: dict[str, Any], sigma: Any, modulation: Any) -> dict[str, Any]:
    from chimera_omni.charted import ChartedBlockSpec

    torch = C16._torch()
    fields = dict(spec_fields)
    fields["type_ids"] = tuple(int(v) for v in fields["type_ids"])
    spec = ChartedBlockSpec(layer_index=0, **fields)
    spec.validate()
    width = int(spec.width)
    dummy = {"chart_in.weight": torch.zeros(width, 1), "chart_out.weight": torch.zeros(1, width)}
    if spec.chart_in_bias:
        dummy["chart_in.bias"] = torch.zeros(width)
    block = C16.build_source_block(spec, 1, {k: v.to(torch.float32) for k, v in tensors.items()}, dummy)
    with torch.no_grad():
        got = block.time_path(sigma.to(torch.float32)).to(torch.float32)
    expected = modulation.to(torch.float32)
    if tuple(got.shape) != tuple(expected.shape):
        raise C16.Refusal(f"time path shape {tuple(got.shape)} != reference {tuple(expected.shape)}")
    rel = float((got - expected).norm() / expected.norm().clamp_min(1e-30))
    return {"rel_l2": rel, "max_abs": float((got - expected).abs().max()), "reference_norm": float(expected.norm()),
            "shape": list(expected.shape), "finite": bool(torch.isfinite(got).all())}


def check_canonical(spec_fields: dict[str, Any], tensors_path: Path) -> dict[str, Any]:
    from chimera_omni.charted import ChartedBlockSpec

    fields = dict(spec_fields)
    fields["type_ids"] = tuple(int(v) for v in fields["type_ids"])
    spec = ChartedBlockSpec(layer_index=0, **fields)
    spec.validate()
    expected = {k: v for k, v in spec.parameter_shapes("", 5120).items() if not k.startswith("chart_")}
    views = C16.read_single(tensors_path)
    got = {k: tuple(v.shape) for k, v in views.items()}
    if got != {k: tuple(v) for k, v in expected.items()}:
        raise C16.Refusal(f"{tensors_path}: tensor set/shapes differ from the spec: missing {sorted(set(expected) - set(got))} "
                          f"unexpected {sorted(set(got) - set(expected))}")
    chart_params_at_5120 = 2 * int(spec.width) * 5120 + (int(spec.width) if spec.chart_in_bias else 0)
    return {
        "canonical_tensor_set": True,
        "block_parameters_without_charts": spec.parameter_count(5120) - chart_params_at_5120,
        "chart_parameters_at_hidden_5120": chart_params_at_5120,
        "installed_parameters_at_hidden_5120": spec.parameter_count(5120),
        "spec_mlp_parameter_count": spec.mlp_parameter_count(),
    }


# --------------------------------------------------------------------------
# ACE
# --------------------------------------------------------------------------


def build_ace(run_dir: Path, ace_weights: Path, out: Path) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()):
        raise C16.Refusal(f"{out} exists and is not empty")
    body = C16.convert_ace_run(run_dir, ace_weights, out)
    charts = out / "charts.safetensors"
    kept = out / "charts_host128.reference.safetensors"
    charts.rename(kept)
    shapes = {k: list(v.shape) for k, v in C16.read_single(kept).items()}
    body["tensor_sha256"].pop("charts.safetensors", None)
    body["provenance"]["charts_host128_reference"] = {
        "file": kept.name, "sha256": sha256_file(kept), "shapes": shapes,
        "why_not_installed": "chart_in.weight is [2048, 128]: fitted on the c16_transplant_run toy host (hidden 128), "
                             "not the 5,120-wide S3 trunk; c16 would refuse it on shape at S7",
    }
    canon = check_canonical(body["spec"], out / "charted_block.safetensors")
    body["s7_unit"] = {
        "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "source": "ACE",
        "built_by": "c16_install_media.convert_ace_run (unchanged) + chart set aside",
        "charts_status": "MISSING at hidden 5120 (needs paired host/parent rows: --tap-rows <dir>/<unit>/paired_rows.safetensors)",
        "component_4x": component_check(body["accounting"]),
        "canonical": canon,
        "time_reference_conversion_rel_l2": body["time_reference"]["conversion_rel_l2"],
        "deviations": ["cross_attn.* dropped (trunk is the context)", "timestep_r = 0 as captured"],
    }
    write_json(out / "charted_block.json", body)
    return body["s7_unit"]


# --------------------------------------------------------------------------
# H3
# --------------------------------------------------------------------------


def build_h3(src: Path, out: Path) -> dict[str, Any]:
    torch = C16._torch()
    body = json.loads((src / "charted_block.json").read_text(encoding="utf-8"))
    if body.get("schema") != C16.BLOCK_SCHEMA:
        raise C16.Refusal(f"{src}: schema {body.get('schema')}")
    tensors_src = src / "charted_block.safetensors"
    ref_src = src / "time_reference.safetensors"
    declared = body.get("tensor_sha256", {})
    views = C16.read_single(tensors_src)
    check_block = C16.verify_declared_sha256(views, declared.get("charted_block.safetensors"), where=str(tensors_src))
    ref_views = C16.read_single(ref_src)
    check_ref = C16.verify_declared_sha256(ref_views, declared.get("time_reference.safetensors"), where=str(ref_src))
    tensors = {k: v.torch(dtype=torch.float32) for k, v in views.items()}
    sigma = ref_views["sigma"].torch(dtype=torch.float32)
    modulation = ref_views["modulation"].torch(dtype=torch.float32)
    gate = time_check(body["spec"], tensors, sigma, modulation)
    gate["tolerance"] = 1.0e-5
    gate["verified"] = gate["rel_l2"] <= 1.0e-5 and gate["finite"]
    if not gate["verified"]:
        raise C16.Refusal(f"H3 time path does not reproduce the captured modulation: {gate}")
    fp32_dtypes = {name: str(views[name].dtype) for name in C16.h3_fp32_parameters() if name in views}
    del tensors
    out.mkdir(parents=True, exist_ok=False)
    (out / "charted_block.safetensors").symlink_to(tensors_src)
    # c16 >= 1.2.0 gates h3_curve6 references itself, under the canonical file name
    (out / C16.TIME_REFERENCE_FILENAME).symlink_to(ref_src)
    new = json.loads(json.dumps(body))
    new["tensor_sha256"] = {"charted_block.safetensors": declared["charted_block.safetensors"],
                            C16.TIME_REFERENCE_FILENAME: declared["time_reference.safetensors"]}
    new["provenance"]["codex_container"] = {
        "dir": str(src), "charted_block.json_sha256": sha256_file(src / "charted_block.json"),
        "charted_block.safetensors_sha256": sha256_file(tensors_src),
        "time_reference.safetensors_sha256": sha256_file(ref_src),
        "declared_tensor_sha256_check": {"charted_block": check_block, "time_reference": check_ref},
    }
    canon = check_canonical(new["spec"], tensors_src)
    new["s7_unit"] = {
        "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "source": "H3",
        "built_by": "symlinks to Codex's h3-native-curve6-sep07 container (both files, canonical names) + json rewrite",
        "charts_status": "MISSING (Codex json: 'paired host/source tap rows still required')",
        "component_4x": component_check(new["accounting"]),
        "canonical": canon,
        "h3_curve6_time_gate": gate,
        "h3_fp32_parameters_source_dtypes": fp32_dtypes,
        "requires_installer": "c16_install_media >= 1.2.0 (gates the h3_curve6 time_reference.safetensors; keeps "
                              "H3_FP32_PARAMETERS in FP32 under --charted-dtype match)",
        "requires_tree": "chimera-omni with MODULATION_KINDS incl. h3_curve6 (tree_ef1ffb9 / 11adb32; NOT c16's default tree_2ec15a8)",
        "runtime_requirements": body.get("runtime_requirements"),
        "deviations": [d.get("name") for d in body.get("capture_deviations", [])],
    }
    new["time_reference"] = dict(body.get("time_reference", {}), file=C16.TIME_REFERENCE_FILENAME)
    write_json(out / "charted_block.json", new)
    return new["s7_unit"]


# --------------------------------------------------------------------------
# QIMG
# --------------------------------------------------------------------------


def build_qimg(run_dir: Path, captures: list[Path], out: Path, block: int = 30,
               time_reference: Path | None = None) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch = C16._torch()
    folded = run_dir / "fold" / "folded_source_block_and_time.safetensors"
    selection = run_dir / "fold" / "selection.safetensors"
    receipt = json.loads((run_dir / "fold" / "fold_receipt.json").read_text(encoding="utf-8"))
    p = f"transformer_blocks.{block}."
    rename = {
        "attn.q_proj.weight": p + "attn.to_q.weight", "attn.q_proj.bias": p + "attn.to_q.bias",
        "attn.k_proj.weight": p + "attn.to_k.weight", "attn.k_proj.bias": p + "attn.to_k.bias",
        "attn.v_proj.weight": p + "attn.to_v.weight", "attn.v_proj.bias": p + "attn.to_v.bias",
        "attn.o_proj.weight": p + "attn.to_out.0.weight", "attn.o_proj.bias": p + "attn.to_out.0.bias",
        "attn.q_norm.weight": p + "attn.norm_q.weight", "attn.k_norm.weight": p + "attn.norm_k.weight",
        "mlp.fc1.weight": p + "img_mlp.net.0.proj.weight", "mlp.fc1.bias": p + "img_mlp.net.0.proj.bias",
        "mlp.fc2.weight": p + "img_mlp.net.2.weight", "mlp.fc2.bias": p + "img_mlp.net.2.bias",
        "time_embed.linear_1.weight": "time_text_embed.timestep_embedder.linear_1.weight",
        "time_embed.linear_1.bias": "time_text_embed.timestep_embedder.linear_1.bias",
        "time_embed.linear_2.weight": "time_text_embed.timestep_embedder.linear_2.weight",
        "time_embed.linear_2.bias": "time_text_embed.timestep_embedder.linear_2.bias",
        "time_embed.time_proj.weight": p + "img_mod.1.weight", "time_embed.time_proj.bias": p + "img_mod.1.bias",
    }
    tensors: dict[str, Any] = {}
    source_dtypes: dict[str, str] = {}
    with safe_open(str(folded), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for name, key in rename.items():
            if key not in keys:
                raise C16.Refusal(f"{folded}: {key} missing")
            value = handle.get_tensor(key)
            source_dtypes[name] = str(value.dtype)
            widened = value.to(torch.bfloat16)
            if not bool(torch.equal(widened.to(torch.float32), value.to(torch.float32))):
                raise C16.Refusal(f"{key}: widening {value.dtype} -> bfloat16 is not exact")
            tensors[name] = widened.contiguous()
        dropped = sorted(k for k in keys if k not in set(rename.values()))
    width = int(tensors["attn.q_proj.weight"].shape[1])
    keep = int(tensors["mlp.fc1.weight"].shape[0])
    tensors["scale_shift_table"] = torch.zeros(6, width, dtype=torch.bfloat16)
    heads = int(tensors["attn.q_proj.weight"].shape[0]) // int(tensors["attn.q_norm.weight"].shape[0])
    head_dim = int(tensors["attn.q_norm.weight"].shape[0])
    spec_fields = {
        "source": "qimg", "component": f"qimg.mmdit.block{block}.img_mlp", "type_ids": [1], "width": width,
        "heads": heads, "kv_heads": heads, "head_dim": head_dim, "mlp_width": keep, "mlp_kind": "gelu_tanh",
        "modulation": "adaln6", "norm_kind": "layer", "norm_affine": False, "norm_eps": 1.0e-6, "qk_norm": True,
        "attn_bias": True, "rope_theta": 0.0, "sliding_window": 0,
        "time_freq_dim": int(tensors["time_embed.linear_1.weight"].shape[1]),
        "time_hidden": int(tensors["time_embed.linear_1.weight"].shape[0]), "time_factor": 1000.0,
        "time_embed_r": False, "delta_scale": 1.0, "chart_in_bias": True, "rung": "ownerpolicy-v4-candidate",
        "source_depth_fraction": block / 59.0,
        "metadata": {"layer_type": "mmdit_img_stream", "text_stream": "dropped (trunk is the context)",
                     "rope": "3-D MM-RoPE dropped (rope_theta 0)", "source_dtype": "F8_E4M3/BF16 widened exactly to BF16",
                     "converted_from": str(run_dir)},
    }
    # time reference from the captures' img_modulation (shift,scale,gate x2 == ACE chunk order)
    sigmas, mods, files = [], [], []
    for cap in captures:
        cv = C16.read_single(cap)
        sigmas.append(cv["sigma"].torch(dtype=torch.float32).reshape(-1)[0])
        mods.append(cv["img_modulation"].torch(dtype=torch.float32).reshape(6, width))
        files.append({"file": str(cap), "sha256": sha256_file(cap)})
    sigma = torch.stack(sigmas)
    modulation = torch.stack(mods)
    bf16_gate = time_check(spec_fields, tensors, sigma, modulation)
    bf16_gate["note"] = ("captured img_modulation is BF16; the rebuilt FP32 path is compared against the bf16-rounded capture, "
                         "so ~1e-3 is the rounding floor, not a path mismatch (a wrong sinusoid/chunk order gives O(1))")
    reference_source = {"source": "capture img_modulation (BF16) at capture sigma"}
    if time_reference is not None:
        # FP32 reference from the PARENT's own code (m3_qimg_time_reference_fp32.py, ComfyUI
        # QwenTimestepProjEmbeddings + img_mod in FP32 from the parent's exactly-widened tensors)
        rv = C16.read_single(time_reference)
        ref_sigma = rv["sigma"].torch(dtype=torch.float32).reshape(-1)
        if not torch.equal(ref_sigma, sigma):
            raise C16.Refusal(f"{time_reference}: sigma {ref_sigma.tolist()} != capture sigma {sigma.tolist()}")
        modulation = rv["modulation"].torch(dtype=torch.float32)
        receipt_path = time_reference.parent / "time_reference_receipt.json"
        reference_source = {
            "source": "FP32 parent time path (ComfyUI QwenTimestepProjEmbeddings + transformer_blocks.%d.img_mod) at the capture sigma" % block,
            "file": str(time_reference), "sha256": sha256_file(time_reference),
            "receipt": str(receipt_path) if receipt_path.is_file() else None,
            "receipt_sha256": sha256_file(receipt_path) if receipt_path.is_file() else None,
            "bf16_capture_gate_diagnostic": bf16_gate,
        }
        gate = time_check(spec_fields, tensors, sigma, modulation)
        gate["note"] = "unit time path vs the FP32 parent reference (the BF16 capture comparison is kept as a diagnostic)"
    else:
        gate = bf16_gate
    gate["c16_default_tolerance"] = 1.0e-5
    gate["passes_c16_default"] = gate["rel_l2"] <= 1.0e-5
    out.mkdir(parents=True, exist_ok=False)
    save_file({k: v for k, v in tensors.items()}, str(out / "charted_block.safetensors"))
    reference_table = C16.write_time_reference(out, sigma, modulation)
    canon = check_canonical(spec_fields, out / "charted_block.safetensors")
    parent_mlp = 2 * width * (4 * width) + 4 * width + width
    child_mlp = 2 * width * keep + keep + width
    accounting = {
        "parent_mlp_parameters": parent_mlp, "child_mlp_parameters": child_mlp,
        "parent_block_parameters": canon["block_parameters_without_charts"] - child_mlp + parent_mlp,
        "child_block_parameters": canon["block_parameters_without_charts"],
        "scope": "image stream only; Codex's dual-stream candidate counts 236,784,126 / 350,061,056 incl. the text stream",
    }
    body = {
        "schema": C16.BLOCK_SCHEMA, "spec": spec_fields,
        "provenance": {
            "source_candidate": str(folded), "source_candidate_sha256": sha256_file(folded),
            "selection": str(selection), "selection_sha256": sha256_file(selection),
            "fold_receipt": str(run_dir / "fold" / "fold_receipt.json"),
            "fold_receipt_sha256": sha256_file(run_dir / "fold" / "fold_receipt.json"),
            "codex_receipt_capability_claim": receipt.get("capability_claim"),
            "codex_receipt_accepted": receipt.get("accepted"),
            "source_dtypes": source_dtypes, "dropped_tensors": dropped, "captures": files,
        },
        "accounting": accounting,
        "holdout": {"note": "Codex's fold_receipt streams/diagnostics; not re-derived here"},
        "time_reference": {"file": C16.TIME_REFERENCE_FILENAME, **reference_source,
                           "sigma": [float(v) for v in sigma], "conversion_rel_l2": gate["rel_l2"]},
        "tensor_sha256": {"charted_block.safetensors": table(out / "charted_block.safetensors"),
                          C16.TIME_REFERENCE_FILENAME: reference_table},
        "insert_after": None,
    }
    body["s7_unit"] = {
        "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "source": "QIMG",
        "built_by": "image-stream extraction from Codex's ownerpolicy-v4 candidate (read only)",
        "charts_status": "MISSING at hidden 5120",
        "component_4x": component_check(accounting), "canonical": canon, "time_gate": gate,
        "deviations": ["text stream (joint-attention partner) dropped", "3-D RoPE dropped"] + (
            ["time reference is a BF16 capture: installer needs --time-reference-rel-tol >= measured rel_l2"]
            if time_reference is None else []),
    }
    write_json(out / "charted_block.json", body)
    return body["s7_unit"]


# --------------------------------------------------------------------------
# HY3D
# --------------------------------------------------------------------------


HY3D_TIME_TOKEN_FILENAME = "hy3d_time_token_reference.safetensors"
HY3D_TIME_TOKEN_SIGMAS = tuple(round(0.05 * i, 2) for i in range(21))


def hy3d_time_token(parent: Path, sigma: Any) -> dict[str, Any]:
    """The parent's block-0 time token t_embedder(sigma) in FP32 (exactly-widened F16 weights).

    Source math (Hunyuan3D-2.1 hunyuandit.py, as pinned by chimera-omni
    codex/work/codex/native-qimg-hy3d-integration docs/HY3D_NATIVE_COMPONENT.md and
    modeling/hy3d.py `time_token`): phase = sigma * exp(-ln(1e4) * k / half) with sigma in
    [0, 1] (no *1000), emb = cat(sin, cos), t = mlp.2(GELU_exact(mlp.0(emb))); the model then
    runs `x = cat([t, x_embedder(latents)], dim=1)`, so the time token is ROW 0 of every block's
    self-attention sequence and evolves with the latent rows block to block.
    """

    import math

    from safetensors import safe_open

    torch = C16._torch()
    with safe_open(str(parent), framework="pt", device="cpu") as handle:
        w0 = handle.get_tensor("model.t_embedder.mlp.0.weight")
        b0 = handle.get_tensor("model.t_embedder.mlp.0.bias")
        w2 = handle.get_tensor("model.t_embedder.mlp.2.weight")
        b2 = handle.get_tensor("model.t_embedder.mlp.2.bias")
    dims = int(w0.shape[1])
    half = dims // 2
    sigma = sigma.to(torch.float32).reshape(-1)
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
    phase = sigma[:, None] * torch.exp(exponent)[None, :]
    emb = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
    hidden = torch.nn.functional.gelu(emb @ w0.to(torch.float32).T + b0.to(torch.float32), approximate="none")
    token = hidden @ w2.to(torch.float32).T + b2.to(torch.float32)
    return {"sigma": sigma, "time_token": token.contiguous(),
            "weights": {"model.t_embedder.mlp.0.weight": list(w0.shape), "model.t_embedder.mlp.2.weight": list(w2.shape),
                        "dtype": str(w0.dtype)}}


def hy3d_reference_vs_capture(parent: Path, capture: Path) -> dict[str, Any]:
    """Diagnostic: the FP32 t_embedder math vs the parent's captured block-0 time token (F16 run)."""

    from safetensors import safe_open

    torch = C16._torch()
    if not capture.is_file():
        return {"checked": False, "reason": f"{capture} missing"}
    with safe_open(str(capture), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        captured = handle.get_tensor("time_token").to(torch.float32).reshape(-1, handle.get_slice("time_token").get_shape()[-1])
        sigma_key = "sigma" if "sigma" in keys else ("timestep" if "timestep" in keys else None)
        if sigma_key is None:
            return {"checked": False, "reason": f"{capture}: no sigma/timestep tensor"}
        sigma = handle.get_tensor(sigma_key).to(torch.float32).reshape(-1)
    ours = hy3d_time_token(parent, sigma)["time_token"]
    rel = [float((ours[i] - captured[i]).norm() / captured[i].norm().clamp_min(1e-30)) for i in range(int(captured.shape[0]))]
    return {"checked": True, "capture": str(capture), "capture_sha256": sha256_file(capture), "sigma_key": sigma_key,
            "sigma": [float(v) for v in sigma], "rel_l2_fp32_reference_vs_captured_f16_time_token": rel,
            "note": "diagnostic only: the capture ran the parent in F16; a wrong sinusoid scale (x1000) or cos/sin order gives O(1)"}


def build_hy3d(parent: Path, dense_child: Path, out: Path, block: int = 10, depth: int = 21,
               block0_capture: Path | None = None) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch = C16._torch()
    p = f"model.blocks.{block}."
    with safe_open(str(parent), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if any(k.startswith(p + "skip_") for k in keys):
            raise C16.Refusal(f"block {block} has a skip input; a charted block cannot carry it")
        if any(k.startswith(p + "moe.") for k in keys):
            raise C16.Refusal(f"block {block} is an MoE block")
        get = lambda k: handle.get_tensor(p + k)
        wq, wk, wv = get("attn1.to_q.weight"), get("attn1.to_k.weight"), get("attn1.to_v.weight")
        width = int(wq.shape[1])
        head_dim = int(get("attn1.q_norm.weight").shape[0])
        heads = width // head_dim
        # source: qkv = cat(q,k,v,-1).view(B,N,heads,3*head_dim).split(head_dim) -> per-head (q|k|v) slices
        packed = torch.cat([wq, wk, wv], dim=0).view(heads, 3, head_dim, width)
        tensors = {
            "attn_norm.weight": get("norm1.weight"), "attn_norm.bias": get("norm1.bias"),
            "mlp_norm.weight": get("norm3.weight"), "mlp_norm.bias": get("norm3.bias"),
            "attn.q_proj.weight": packed[:, 0].reshape(heads * head_dim, width).contiguous(),
            "attn.k_proj.weight": packed[:, 1].reshape(heads * head_dim, width).contiguous(),
            "attn.v_proj.weight": packed[:, 2].reshape(heads * head_dim, width).contiguous(),
            "attn.q_proj.bias": torch.zeros(heads * head_dim, dtype=wq.dtype),
            "attn.k_proj.bias": torch.zeros(heads * head_dim, dtype=wq.dtype),
            "attn.v_proj.bias": torch.zeros(heads * head_dim, dtype=wq.dtype),
            "attn.o_proj.weight": get("attn1.out_proj.weight"), "attn.o_proj.bias": get("attn1.out_proj.bias"),
            "attn.q_norm.weight": get("attn1.q_norm.weight"), "attn.k_norm.weight": get("attn1.k_norm.weight"),
        }
        parent_mlp = sum(int(handle.get_slice(p + k).get_shape()[0]) * (int(handle.get_slice(p + k).get_shape()[1]) if len(handle.get_slice(p + k).get_shape()) > 1 else 1)
                         for k in ("mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"))
        dropped = sorted(k[len(p):] for k in keys if k.startswith(p) and k[len(p):].split(".")[0] in ("attn2", "norm2"))
    with safe_open(str(dense_child), framework="pt", device="cpu") as handle:
        for k in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
            tensors["mlp." + k] = handle.get_tensor(k)
    keep = int(tensors["mlp.fc1.weight"].shape[0])
    spec_fields = {
        "source": "hy3d", "component": f"hy3d.dit.block{block}.mlp", "type_ids": [7], "width": width, "heads": heads,
        "kv_heads": heads, "head_dim": head_dim, "mlp_width": keep, "mlp_kind": "gelu", "modulation": "none",
        "norm_kind": "layer", "norm_affine": True, "norm_eps": 1.0e-6, "qk_norm": True, "attn_bias": True,
        "rope_theta": 0.0, "sliding_window": 0, "time_freq_dim": 256, "time_hidden": 0, "time_factor": 1000.0,
        "time_embed_r": False, "delta_scale": 1.0, "chart_in_bias": True, "rung": "hy3d-native-dense-child",
        "source_depth_fraction": block / (depth - 1),
        "metadata": {"layer_type": "hunyuandit_plain_dense",
                     "time": "time token = ROW 0 of the volume segment, an owned type-7 row (runtime_requirements.hy3d_time_token_row); "
                             "the block attends to it exactly as the parent self-attention does (modulation none: HY3D has no adaLN)",
                     "cross_attention": "attn2 (DINO image context) dropped: the charted kind has no conditioning input (structural; owner decision)", "qkv": "re-packed from source per-head (q|k|v) split",
                     "qkv_bias": "zero rows written (source q/k/v have no bias; out_proj does)",
                     "mlp_gelu": "exact (non-approximate) per HY3D_NATIVE_COMPONENT.md", "converted_from": str(dense_child)},
    }
    out.mkdir(parents=True, exist_ok=False)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "charted_block.safetensors"))
    canon = check_canonical(spec_fields, out / "charted_block.safetensors")
    child_mlp = canon["spec_mlp_parameter_count"]
    accounting = {
        "parent_mlp_parameters": parent_mlp, "child_mlp_parameters": child_mlp,
        "parent_block_parameters": canon["block_parameters_without_charts"] - child_mlp + parent_mlp,
        "child_block_parameters": canon["block_parameters_without_charts"],
        "scope": "self-attention + MLP; zero q/k/v bias rows (3 x width) are counted",
    }
    token = hy3d_time_token(parent, C16._torch().tensor(HY3D_TIME_TOKEN_SIGMAS))
    save_file({"sigma": token["sigma"], "time_token": token["time_token"]}, str(out / HY3D_TIME_TOKEN_FILENAME))
    time_token_row = {
        "what": "the parent time token: HunYuanDiTPlain prepends t_embedder(t) as sequence row 0 "
                "(x = cat([t, x_embedder(latents)], dim=1)); every block self-attends over it and updates it",
        "layout": "each volume segment = [1 time-token row | latent rows], all type_id 7, contiguous, time row FIRST",
        "time_row_feature": "all zeros over native_feature_dim (no latent content)",
        "time_row_noise_level": "the segment sigma in [0, 1] (the same value as its latent rows): the bridge noise_proj "
                                "is then the only content-bearing input of that row",
        "time_row_position": "thwr_v1 [-1, -1, -1, -1] (reserved: no latent row uses a negative coordinate)",
        "charted_block_effect": "an owned type-7 row like the latent rows: chart_in reads it, the block attends to it, "
                                "and the block delta is written back to it (as the parent updates row 0)",
        "pairing": "R3 pairs the time row trunk state with the parent block-10 INPUT row 0 (the evolved time token), "
                   "not with t_embedder(t); the block-0 token is shipped only as a reference",
        "pairing_implemented_by": {
            "s7_pairs_parent": ">= 1.1.0: hy3d keeps block_input row 0 as the forward's first row (time_row=True, zero features, "
                               "feature_width 0, position [-1,-1,-1,-1], the segment sigma)",
            "s7_host_rows": ">= 1.2.0: runs the trunk prefix over [prompt | time row | latent rows], validates this layout "
                            "structurally and refuses pairs without time rows for this unit; writes time_row into paired_rows",
            "s7_chart_fit": ">= 1.1.0: refuses paired rows without time rows for this unit; time rows are rows of both charts "
                            "(fit-design weight recorded in charts.time_rows)",
        },
        "reference_file": HY3D_TIME_TOKEN_FILENAME,
        "reference": "FP32 t_embedder(sigma) at sigma 0.00..1.00 step 0.05 = the block-0 time row; the block-10 row has evolved",
        "reference_tensor_sha256": table(out / HY3D_TIME_TOKEN_FILENAME),
        "reference_weights": token["weights"],
        "reference_vs_block0_capture": (hy3d_reference_vs_capture(parent, block0_capture) if block0_capture is not None
                                        else {"checked": False, "reason": "no --hy3d-block0-capture"}),
        "installer_check": "none in c16 1.2.0 (it loads modulation-none containers without a time gate); the row is a runtime "
                           "layout contract: the S7 driver must emit it at inference exactly as s7_host_rows emits it",
    }
    body = {
        "schema": C16.BLOCK_SCHEMA, "spec": spec_fields,
        "provenance": {"source_weights": str(parent), "source_prefix": p, "dense_child": str(dense_child),
                       "dense_child_sha256": sha256_file(dense_child),
                       "dense_child_fold_receipt": str(dense_child.parent / "fold_receipt.json"),
                       "dense_child_fold_receipt_sha256": sha256_file(dense_child.parent / "fold_receipt.json"),
                       "dropped_tensors": dropped,
                       "source_math_reference": "chimera-omni codex/validated docs/HY3D_NATIVE_COMPONENT.md"},
        "accounting": accounting,
        "holdout": {"note": "Codex fold_receipt candidate_diagnostics (block_output relL2 etc.); not re-derived here"},
        "tensor_sha256": {"charted_block.safetensors": table(out / "charted_block.safetensors")},
        "runtime_requirements": {"hy3d_time_token_row": time_token_row},
        "insert_after": None,
    }
    body["s7_unit"] = {
        "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "source": "HY3D",
        "built_by": "parent F16 block tensors + Codex dense child MLP (read only)",
        "charts_status": "MISSING at hidden 5120",
        "component_4x": component_check(accounting), "canonical": canon,
        "time_token": "RESTORED as a runtime row: runtime_requirements.hy3d_time_token_row (the block sees it in self-attention); "
                      "charts must be fitted on paired rows that carry it (s7_chart_fit >= 1.1.0 refuses otherwise)",
        "structural_limitation_owner_decision": {
            "what": "cross-attention attn2 to the DINO image conditioner (1370 x 1024 context rows) is dropped",
            "why": "the charted block kind (chimera-omni modeling/charted.py) carries no conditioning input: a block reads only "
                   "the trunk rows it owns (plus H3 text rows through the same chart); attn2.to_k/to_v expect 1024-wide "
                   "DINO features, not trunk rows",
            "options": ["accept the drop (conditioning reaches the volume rows only through the trunk context)",
                        "extend the charted kind with a conditioning chart (new model code in chimera-omni + c16; context rows "
                        "would come from trunk text/image rows through a second [1024 x 5120] chart)"],
            "decision_needed_from": "Cyril",
        },
        "deviations": ["cross-attention to DINO context dropped (structural, see structural_limitation_owner_decision)",
                       "zero q/k/v bias rows"],
    }
    write_json(out / "charted_block.json", body)
    return body["s7_unit"]


# --------------------------------------------------------------------------
# QVL32 (ViT block)
# --------------------------------------------------------------------------


def parent_vision_mlp(parent: Path, block: int) -> dict[str, Any]:
    """The parent's TRUE vision MLP width, read from its safetensors header (no assumption)."""

    from safetensors import safe_open

    with safe_open(str(parent), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        prefixes = sorted({k[: k.index("blocks.")] for k in keys if ".blocks." in k and "visual" in k and k.endswith("mlp.linear_fc1.weight")})
        if len(prefixes) != 1:
            raise C16.Refusal(f"{parent}: expected one visual tower prefix, found {prefixes}")
        prefix = prefixes[0]
        widths = {}
        for k in keys:
            if k.startswith(prefix + "blocks.") and k.endswith("mlp.linear_fc1.weight"):
                widths[int(k[len(prefix + "blocks."):].split(".")[0])] = [int(v) for v in handle.get_slice(k).get_shape()]
        fc2 = [int(v) for v in handle.get_slice(f"{prefix}blocks.{block}.mlp.linear_fc2.weight").get_shape()]
    shapes = {tuple(v) for v in widths.values()}
    if len(shapes) != 1:
        raise C16.Refusal(f"{parent}: vision MLP widths differ across blocks: {sorted(shapes)}")
    intermediate, hidden = widths[block]
    if fc2 != [hidden, intermediate]:
        raise C16.Refusal(f"{parent}: block {block} fc2 {fc2} != [{hidden}, {intermediate}]")
    return {"parent": str(parent), "prefix": prefix, "depth": len(widths), "intermediate_size": intermediate,
            "hidden_size": hidden, "fc1_shape_all_blocks": list(next(iter(shapes))), "fc2_shape_block": fc2,
            "read_from": "safetensors header (linear_fc1/linear_fc2 shapes of every visual block)"}


def build_qvl(child_dir: Path, out: Path, block: int, parent: Path | None = None,
              type_ids: tuple[int, ...] = (6,)) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch = C16._torch()
    if parent is None:
        raise C16.Refusal("qvl: --parent is required (the parent's MLP width is read from its header, not assumed)")
    parent_mlp_info = parent_vision_mlp(parent, block)
    parent_intermediate = int(parent_mlp_info["intermediate_size"])
    component = child_dir / "native_component.safetensors"
    config = json.loads((child_dir / "native_config.json").read_text(encoding="utf-8"))
    width = int(config["hidden_size"])
    heads = int(config["num_heads"])
    head_dim = width // heads
    depth = int(config["depth"])
    p = f"blocks.{block}."
    with safe_open(str(component), framework="pt", device="cpu") as handle:
        meta = handle.metadata()
        get = lambda k: handle.get_tensor(p + k)
        qkv_w, qkv_b = get("attn.qkv.weight"), get("attn.qkv.bias")
        # Qwen3-VL vision: qkv(x).reshape(seq, 3, heads, head_dim) -> standard q | k | v row blocks
        tensors = {
            "attn_norm.weight": get("norm1.weight"), "attn_norm.bias": get("norm1.bias"),
            "mlp_norm.weight": get("norm2.weight"), "mlp_norm.bias": get("norm2.bias"),
            "attn.q_proj.weight": qkv_w[:width].contiguous(), "attn.k_proj.weight": qkv_w[width:2 * width].contiguous(),
            "attn.v_proj.weight": qkv_w[2 * width:].contiguous(),
            "attn.q_proj.bias": qkv_b[:width].contiguous(), "attn.k_proj.bias": qkv_b[width:2 * width].contiguous(),
            "attn.v_proj.bias": qkv_b[2 * width:].contiguous(),
            "attn.o_proj.weight": get("attn.proj.weight"), "attn.o_proj.bias": get("attn.proj.bias"),
            "mlp.fc1.weight": get("mlp.linear_fc1.weight"), "mlp.fc1.bias": get("mlp.linear_fc1.bias"),
            "mlp.fc2.weight": get("mlp.linear_fc2.weight"), "mlp.fc2.bias": get("mlp.linear_fc2.bias"),
        }
        block_keys = sorted(k for k in handle.keys() if k.startswith(p))
    keep = int(tensors["mlp.fc1.weight"].shape[0])
    spec_fields = {
        "source": "qvl32", "component": f"qvl32.vit.block{block}.mlp", "type_ids": [int(t) for t in type_ids], "width": width,
        "heads": heads, "kv_heads": heads, "head_dim": head_dim, "mlp_width": keep, "mlp_kind": "gelu_tanh",
        "modulation": "none", "norm_kind": "layer", "norm_affine": True, "norm_eps": 1.0e-6, "qk_norm": False,
        "attn_bias": True, "rope_theta": 0.0, "sliding_window": 0, "time_freq_dim": 256, "time_hidden": 0,
        "time_factor": 1000.0, "time_embed_r": False, "delta_scale": 1.0, "chart_in_bias": True,
        "rung": "qvl32-1075-diagnostic-fit", "source_depth_fraction": block / (depth - 1),
        "metadata": {"layer_type": "qwen3vl_vision_block", "rope": "2-D vision RoPE dropped (rope_theta 0)",
                     "act": "gelu tanh: ComfyUI 0.37 comfy/text_encoders/qwen35.py Qwen35VisionMLP F.gelu(approximate='tanh'), "
                            "the class Qwen3VLVisionModel runs; Qwen3.8-27B-U config vision hidden_act gelu_pytorch_tanh",
                     "type_ids": "ocr (6) only: image (1) is QIMG's C15/charted type; [1, 6] made both blocks run on image rows "
                                 "and c16 refuses type 1 in C15 and C11 together",
                     "converted_from": str(child_dir)},
    }
    out.mkdir(parents=True, exist_ok=False)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "charted_block.safetensors"))
    canon = check_canonical(spec_fields, out / "charted_block.safetensors")
    parent_mlp = 2 * width * parent_intermediate + parent_intermediate + width
    child_mlp = canon["spec_mlp_parameter_count"]
    accounting = {
        "parent_mlp_parameters": parent_mlp, "child_mlp_parameters": child_mlp,
        "parent_block_parameters": canon["block_parameters_without_charts"] - child_mlp + parent_mlp,
        "child_block_parameters": canon["block_parameters_without_charts"],
        "parent_intermediate": parent_intermediate,
        "parent_intermediate_source": parent_mlp_info,
    }
    receipt = child_dir / "fold_receipt.json"
    body = {
        "schema": C16.BLOCK_SCHEMA, "spec": spec_fields,
        "provenance": {"source_component": str(component), "source_component_sha256": sha256_file(component),
                       "source_component_metadata": meta, "fold_receipt": str(receipt),
                       "fold_receipt_sha256": sha256_file(receipt), "native_config": config, "block_keys": block_keys},
        "accounting": accounting,
        "holdout": {"note": "Codex receipt split 'diagnostic_fit' (24 observation rows per block); no held split"},
        "tensor_sha256": {"charted_block.safetensors": table(out / "charted_block.safetensors")},
        "insert_after": None,
    }
    body["s7_unit"] = {
        "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "source": "QVL32",
        "built_by": "one ViT block from Codex's qvl32-1075 diagnostic FFN child (read only)",
        "charts_status": "MISSING at hidden 5120",
        "component_4x": component_check(accounting), "canonical": canon,
        "deviations": ["2-D vision RoPE dropped", "learned interpolated pos_embed not carried (trunk positions instead)",
                       "DeepStack mergers (blocks 8/16/24 -> LM) not carried by a single charted block"],
        "c11_rows": "built separately: composite/s7/c11/<arm>/companions.json (m3_c11_qvl_rows.py), type 6 ocr",
    }
    write_json(out / "charted_block.json", body)
    return body["s7_unit"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", choices=("ace", "h3", "qimg", "hy3d", "qvl"))
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ace-weights", default="/home/cereal/comfyui/models/diffusion_models/acestep_v1.5_turbo.safetensors")
    parser.add_argument("--captures", nargs="*", default=[])
    parser.add_argument("--dense-child")
    parser.add_argument("--block", type=int)
    parser.add_argument("--time-reference", help="qimg: FP32 time_reference.safetensors (m3_qimg_time_reference_fp32.py)")
    parser.add_argument("--parent", help="qvl: the parent checkpoint whose visual.* header gives the true MLP width; "
                                         "hy3d: (optional) parent for the time-token reference, defaults to --src")
    parser.add_argument("--type-ids", type=int, nargs="+", help="qvl: routed type ids (default [6] = ocr)")
    parser.add_argument("--hy3d-block0-capture", help="hy3d: the parent's block-0 capture (time_token + sigma) for the reference diagnostic")
    args = parser.parse_args(argv)
    C16.tree_identity(Path(os.environ["CHIMERA_OMNI_SRC"]))  # the tree this process imports must be the named one
    try:
        if args.which == "ace":
            report = build_ace(Path(args.src), Path(args.ace_weights), Path(args.out))
        elif args.which == "h3":
            report = build_h3(Path(args.src), Path(args.out))
        elif args.which == "qimg":
            report = build_qimg(Path(args.src), [Path(c) for c in args.captures], Path(args.out), block=args.block or 30,
                                time_reference=Path(args.time_reference) if args.time_reference else None)
        elif args.which == "qvl":
            report = build_qvl(Path(args.src), Path(args.out), block=args.block if args.block is not None else 13,
                               parent=Path(args.parent) if args.parent else None,
                               type_ids=tuple(args.type_ids) if args.type_ids else (6,))
        else:
            report = build_hy3d(Path(args.src), Path(args.dense_child), Path(args.out), block=args.block if args.block is not None else 10,
                                block0_capture=Path(args.hy3d_block0_capture) if args.hy3d_block0_capture else None)
    except C16.Refusal as error:
        print(json.dumps({"refused": str(error)}))
        return 3
    print(json.dumps({"unit": args.out, **report}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
