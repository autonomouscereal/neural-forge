#!/usr/bin/env python3
"""m3_c11_qvl_capture.py -- parent vision-tower captures for the C11 patch-embed/merger rows.

Lane M3 (Claude, 2026-10-10). Runs the PARENT's own vision tower code (ComfyUI 0.37
`comfy.text_encoders.qwen35.Qwen35VisionModel`, the class Qwen3-VL's vision model extends)
on real images, in FP32, with the parent's own preprocessing
(`comfy.text_encoders.qwen_vl.process_qwen2vl_images`, patch 16, mean/std 0.5 as Qwen3-VL
uses), and records per image:

  pixels        [N, 1536]  F32  the raw tubelets in merge order (= one native ocr row each)
  patch_embed   [N, 1152]  F32  patch_embed(pixels), BEFORE the interpolated pos_embed
  block13_in    [N, 1152]  F32  input of visual.blocks[13] (R2 parent rows for the ViT unit)
  merger_in     [N, 1152]  F32  input of visual.merger (output of the last block)
  merged        [N/4, 5120] F32 visual.merger output (the LM-space image tokens)
  grid_thw      [3]        I64

Two parents use the same code: Qwen3-VL-32B (`--prefix visual.`, the DeepStack mergers are
not run: they feed LM layers, not the merger) and Qwen3.8-27B-Uncensored's own
`model.visual.*` (`--prefix model.visual.`, qwen3_5_vision, no DeepStack). Weights are
widened BF16 -> F32 exactly; no training, no quantization.

    docker run --rm --network none --gpus device=<i> -v ...:ro ... <comfy image> \
      /opt/environments/python/comfyui/bin/python /tools/m3_c11_qvl_capture.py \
      --parent /parent/<file or HF dir> --prefix visual. --images /run/images.txt --out /out/<arm>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

TOOL_NAME = "m3_c11_qvl_capture"
TOOL_VERSION = "1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_visual_state(parent: Path, prefix: str):
    """{name without prefix: tensor} for every `prefix*` tensor of a single file or an HF index dir."""

    from safetensors import safe_open

    files: list[Path]
    if parent.is_dir():
        index = json.loads((parent / "model.safetensors.index.json").read_text())
        files = sorted({parent / f for k, f in index["weight_map"].items() if k.startswith(prefix)})
    else:
        files = [parent]
    state = {}
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state[key[len(prefix):]] = handle.get_tensor(key)
    return state, [str(f) for f in files]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--prefix", required=True, choices=("visual.", "model.visual."))
    parser.add_argument("--images", required=True, help="text file, one image path per line")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--block", type=int, default=13)
    parser.add_argument("--comfy-root", default="/opt/ComfyUI")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import torch
    import torch.nn as nn
    from PIL import Image
    from safetensors.torch import save_file

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, args.comfy_root)
    from comfy.text_encoders.qwen35 import Qwen35VisionModel
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images

    class Ops:  # plain torch layers for ComfyUI's ops= hook
        Linear = nn.Linear
        Conv3d = nn.Conv3d
        LayerNorm = nn.LayerNorm
        Embedding = nn.Embedding

    parent = Path(args.parent)
    raw_state, files = load_visual_state(parent, args.prefix)
    state = {k: v for k, v in raw_state.items() if not k.startswith("deepstack_merger_list.")}
    dropped = sorted(k for k in raw_state if k.startswith("deepstack_merger_list."))
    depth = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
    intermediate = int(state["blocks.0.mlp.linear_fc1.weight"].shape[0])
    config = {"hidden_size": int(state["blocks.0.norm1.weight"].shape[0]), "intermediate_size": intermediate,
              "depth": depth, "num_heads": 16, "patch_size": 16, "temporal_patch_size": 2, "in_channels": 3,
              "spatial_merge_size": 2, "num_position_embeddings": int(state["pos_embed.weight"].shape[0]),
              "out_hidden_size": int(state["merger.linear_fc2.weight"].shape[0])}
    model = Qwen35VisionModel(config, device="cpu", dtype=torch.float32, ops=Ops)
    missing = model.load_state_dict({k: v.to(torch.float32) for k, v in state.items()}, strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        raise SystemExit(f"vision state mismatch: missing {missing.missing_keys[:5]} unexpected {missing.unexpected_keys[:5]}")
    model = model.to(args.device).eval()

    captured: dict[str, torch.Tensor] = {}
    hooks = [
        model.patch_embed.register_forward_hook(lambda m, i, o: captured.__setitem__("patch_embed", o.detach())),
        model.blocks[args.block].register_forward_pre_hook(lambda m, i: captured.__setitem__("block_in", i[0].detach())),
        model.merger.register_forward_pre_hook(lambda m, i: captured.__setitem__("merger_in", i[0].detach())),
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [line.strip() for line in Path(args.images).read_text().splitlines() if line.strip()]
    records = []
    started = time.time()
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB").resize((args.size, args.size), Image.BICUBIC)
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(args.size, args.size, 3)
        tensor = tensor.to(torch.float32).div(255.0).unsqueeze(0)
        pixels, grid = process_qwen2vl_images(tensor, patch_size=16, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
        pixels = pixels.to(args.device, dtype=torch.float32)
        merged = model(pixels, grid.to(args.device))
        body = {
            "pixels": pixels.float().cpu().contiguous(),
            "patch_embed": captured["patch_embed"].float().cpu().contiguous(),
            "block13_in": captured["block_in"].float().cpu().contiguous(),
            "merger_in": captured["merger_in"].float().cpu().contiguous(),
            "merged": merged.float().cpu().contiguous(),
            "grid_thw": grid.reshape(-1).cpu().to(torch.int64).contiguous(),
        }
        for name, value in body.items():
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise SystemExit(f"{path}: non-finite {name}")
        name = f"img{index:04d}.safetensors"
        save_file(body, str(out / name))
        records.append({"index": index, "image": path, "image_sha256": sha256_file(Path(path)), "file": name,
                        "rows": int(body["pixels"].shape[0]), "merged_rows": int(body["merged"].shape[0]),
                        "grid_thw": body["grid_thw"].tolist()})
    for hook in hooks:
        hook.remove()
    receipt = {
        "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "tool_sha256": sha256_file(Path(__file__).resolve()),
        "argv": sys.argv, "parent": str(parent), "parent_files": files, "prefix": args.prefix,
        "config": config, "deepstack_tensors_not_run": len(dropped),
        "parent_code": {"comfy_root": args.comfy_root, "classes": "comfy.text_encoders.qwen35.Qwen35VisionModel + qwen_vl.process_qwen2vl_images",
                        "qwen35_py_sha256": sha256_file(Path(args.comfy_root) / "comfy/text_encoders/qwen35.py"),
                        "qwen_vl_py_sha256": sha256_file(Path(args.comfy_root) / "comfy/text_encoders/qwen_vl.py")},
        "compute": {"dtype": "float32 (BF16 weights widened exactly)", "device": args.device, "tf32": False,
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None},
        "preprocess": {"resize": f"PIL bicubic {args.size}x{args.size}", "patch_size": 16, "mean_std": [0.5, 0.5],
                       "temporal": "frame duplicated to temporal_patch_size 2 (process_qwen2vl_images)"},
        "images": records, "wall_seconds": time.time() - started,
    }
    (out / "capture_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "images": len(records), "rows": sum(r["rows"] for r in records),
                      "config": config, "wall_seconds": receipt["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
