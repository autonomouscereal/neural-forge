#!/usr/bin/env python3
"""m3_qimg_time_reference_fp32.py -- FP32 time reference for the QIMG block-30 charted unit.

Lane M3 (Claude, 2026-10-10). The QIMG unit's time reference was the BF16 `img_modulation`
the Sep-07 broad capture recorded, so c16's time-path gate measured BF16 rounding
(rel_l2 2.04e-3), not the path. This tool re-runs the PARENT's own time path in FP32 with
the PARENT's own code (ComfyUI `comfy.ldm.qwen_image.model.QwenTimestepProjEmbeddings` +
the block's `img_mod = Sequential(SiLU, Linear)`), driven by the capture's exact sigma
values, from the parent checkpoint's tensors widened exactly to FP32 (F8_E4M3 -> F32 is
exact). It does NOT import chimera_omni: the reference is independent of the charted
block's `time_path` it will gate.

Run inside the ComfyUI image (the only place the parent code lives), CPU only:

    docker run --rm --network none --cpus 8 -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 \
      -e OPENBLAS_NUM_THREADS=8 -v <tools>:/tools:ro -v <models>:/parent:ro \
      -v <capture>:/capture:ro -v <out>:/out --entrypoint nice <image> -n 19 \
      /opt/environments/python/comfyui/bin/python /tools/m3_qimg_time_reference_fp32.py \
      --parent /parent/qwen_image_fp8_e4m3fn.safetensors --captures /capture/...sigma00.safetensors ... \
      --block 30 --out /out

Writes <out>/time_reference.safetensors (sigma [n] F32, modulation [n, 6, 3072] F32,
chunk order shift/scale/gate x (msa, mlp) = ComfyUI's img_mod chunk(2) then chunk(3)) and
<out>/time_reference_receipt.json (parent tensor sha256s, ComfyUI version, and the
diagnostic rel_l2 of the BF16 capture against this FP32 reference -- a diagnostic, not a gate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

TOOL_NAME = "m3_qimg_time_reference_fp32"
TOOL_VERSION = "1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Ops:
    """Plain torch layers for ComfyUI's `operations=` hook (no casting, no lazy init)."""

    import torch.nn as _nn

    Linear = _nn.Linear
    Embedding = _nn.Embedding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--captures", nargs="+", required=True)
    parser.add_argument("--block", type=int, default=30)
    parser.add_argument("--comfy-root", default="/opt/ComfyUI")
    parser.add_argument("--hash-parent-file", action="store_true", help="also sha256 the whole parent file (20 GB read)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_grad_enabled(False)
    sys.path.insert(0, args.comfy_root)
    import comfy.cli_args  # parsed with [] (comfy.options.args_parsing is False)

    comfy.cli_args.args.cpu = True  # model_management must not probe CUDA: this run is CPU FP32
    from comfy.ldm.qwen_image.model import QwenTimestepProjEmbeddings  # the parent's own time code

    version = "unknown"
    version_file = Path(args.comfy_root) / "comfyui_version.py"
    if version_file.is_file():
        for line in version_file.read_text().splitlines():
            if line.startswith("__version__"):
                version = line.split("=", 1)[1].strip().strip('"')
    model_file = Path(args.comfy_root) / "comfy" / "ldm" / "qwen_image" / "model.py"

    parent = Path(args.parent)
    p = f"transformer_blocks.{args.block}."
    names = {
        "timestep_embedder.linear_1.weight": "time_text_embed.timestep_embedder.linear_1.weight",
        "timestep_embedder.linear_1.bias": "time_text_embed.timestep_embedder.linear_1.bias",
        "timestep_embedder.linear_2.weight": "time_text_embed.timestep_embedder.linear_2.weight",
        "timestep_embedder.linear_2.bias": "time_text_embed.timestep_embedder.linear_2.bias",
        "img_mod.1.weight": p + "img_mod.1.weight",
        "img_mod.1.bias": p + "img_mod.1.bias",
    }
    source: dict[str, dict] = {}
    wide: dict[str, torch.Tensor] = {}
    with safe_open(str(parent), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        scale_keys = sorted(k for k in keys if "scale_weight" in k or k.endswith(".scale") or "input_scale" in k)
        if scale_keys:
            raise SystemExit(f"parent carries fp8 scale tensors {scale_keys[:4]}; an exact widening would ignore them")
        for local, key in names.items():
            value = handle.get_tensor(key)
            widened = value.to(torch.float32)
            # exactness: F8_E4M3 / BF16 / F16 -> F32 is exact; prove it by the round trip
            back = widened.to(value.dtype)
            same = (torch.equal(back.view(torch.uint8), value.view(torch.uint8)) if value.element_size() == 1
                    else torch.equal(back, value))
            if not same:
                raise SystemExit(f"{key}: widening {value.dtype} -> float32 does not round-trip")
            wide[local] = widened
            source[key] = {"dtype": str(value.dtype), "shape": list(value.shape),
                           "sha256_raw_bytes": hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
    width = int(wide["img_mod.1.weight"].shape[1])

    embed = QwenTimestepProjEmbeddings(embedding_dim=width, pooled_projection_dim=width, dtype=torch.float32,
                                       device="cpu", operations=_Ops)
    missing = embed.load_state_dict({k: v for k, v in wide.items() if k.startswith("timestep_embedder.")}, strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        raise SystemExit(f"time embed state mismatch: {missing}")
    img_mod = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(width, 6 * width))
    img_mod.load_state_dict({"1.weight": wide["img_mod.1.weight"], "1.bias": wide["img_mod.1.bias"]}, strict=True)
    embed.eval().float()
    img_mod.eval().float()

    sigmas, captured, files = [], [], []
    for cap in args.captures:
        with safe_open(cap, framework="pt", device="cpu") as handle:
            sigma = handle.get_tensor("sigma").to(torch.float32).reshape(-1)
            sigmas.append(sigma[0])
            captured.append(handle.get_tensor("img_modulation").to(torch.float32).reshape(6, width))
        files.append({"file": cap, "sha256": sha256_file(Path(cap))})
    sigma = torch.stack(sigmas)
    # the parent passes `timestep` = the capture's sigma; Timesteps(scale=1000) scales inside
    hidden_states = torch.zeros(len(sigma), 1, width, dtype=torch.float32)
    temb = embed(sigma, hidden_states)
    params = img_mod(temb)  # [n, 6*width]; chunk(2) -> (mod1, mod2), each chunk(3) -> shift, scale, gate
    modulation = params.view(len(sigma), 6, width).contiguous()
    if not bool(torch.isfinite(modulation).all()):
        raise SystemExit("non-finite FP32 modulation")
    stacked = torch.stack(captured)
    rel_vs_bf16 = float((stacked - modulation).norm() / modulation.norm())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = out / "time_reference.safetensors"
    save_file({"sigma": sigma.contiguous(), "modulation": modulation}, str(ref))
    receipt = {
        "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "tool_sha256": sha256_file(Path(__file__).resolve()),
        "argv": sys.argv,
        "parent_file": str(parent), "parent_file_bytes": parent.stat().st_size,
        "parent_file_sha256": sha256_file(parent) if args.hash_parent_file else "not computed (pass --hash-parent-file)",
        "parent_tensors": source,
        "parent_code": {"comfyui_version": version, "module": "comfy.ldm.qwen_image.model.QwenTimestepProjEmbeddings + img_mod Sequential(SiLU, Linear)",
                        "model_py_sha256": sha256_file(model_file) if model_file.is_file() else None},
        "compute": {"dtype": "float32", "device": "cpu", "torch": torch.__version__,
                    "threads": torch.get_num_threads(), "widening": "parent F8_E4M3 -> F32 (exact, round-trip checked)"},
        "convention": {"timestep": "the capture's sigma, passed as ComfyUI passes it; Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)",
                       "chunk_order": "img_mod output viewed [6, width] = (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)"},
        "captures": files,
        "sigma": [float(v) for v in sigma],
        "reference": {"file": ref.name, "sha256": sha256_file(ref), "modulation_norm": float(modulation.norm())},
        "diagnostic_bf16_capture_vs_fp32_reference": {"rel_l2": rel_vs_bf16, "max_abs": float((stacked - modulation).abs().max()),
                                                       "note": "BF16 rounding of the captured img_modulation; recorded, not a gate"},
    }
    (out / "time_reference_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(ref), "sigma": receipt["sigma"], "rel_l2_bf16_capture_vs_fp32": rel_vs_bf16,
                      "modulation_norm": receipt["reference"]["modulation_norm"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
