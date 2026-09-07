"""Bounded K3 CUDA checks; invoke only inside a root-owned allocated container.

This does not allocate GPUs or launch containers. It never changes parent files.
Synthetic exact-fold outputs live in one fresh scratch directory. --parent adds
one real L1 expert's packed matrices, not a parent forward or benchmark rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def compare_dequant(dequant, packed, scale, device: str) -> dict:
    cpu = dequant(packed.cpu(), scale.cpu(), dtype=torch.bfloat16)
    actual = dequant(packed.to(device), scale.to(device), dtype=torch.bfloat16).cpu()
    if not torch.isfinite(cpu).all() or not torch.isfinite(actual).all():
        raise RuntimeError("dequant canary produced nonfinite values")
    if not torch.equal(cpu, actual):
        raise RuntimeError("CPU/CUDA packed-expert dequant differs")
    return {"shape": list(cpu.shape), "dtype": str(cpu.dtype), "byte_value_equal": True,
            "max_abs_difference": float((cpu.float() - actual.float()).abs().max())}


def synthetic_dequant_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    # Every packed nibble value and a bounded span of E8M0 exponents; no RNG.
    packed = (torch.arange(8 * 32).reshape(8, 32) % 256).to(torch.uint8)
    scale = torch.arange(120, 136).reshape(8, 2).to(torch.uint8)
    return packed, scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nf", required=True, type=Path)
    ap.add_argument("--chimera", required=True, type=Path)
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/k3-gpu-canary"))
    ap.add_argument("--parent", type=Path)
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        ap.exit(2, "canary refused: exactly one allocated visible CUDA device is required\n")
    if args.scratch.exists() and (not args.scratch.is_dir() or next(args.scratch.iterdir(), None) is not None):
        ap.exit(2, "canary refused: scratch must be absent or empty\n")
    for source in (args.nf, args.chimera, args.parent):
        if source is not None and (args.scratch.resolve().is_relative_to(source.resolve())
                                   or source.resolve().is_relative_to(args.scratch.resolve())):
            ap.exit(2, "canary refused: scratch must not overlap source mounts\n")
    tools = args.nf / "reports/omni-campaign-2026-09/composite/tools"
    sys.path.insert(0, str(tools))
    from preflight_k3_tracked import SOURCE_HASHES, verify_file
    for rel, expected in SOURCE_HASHES.items():
        verify_file(args.nf / rel, expected)
    import test_k3_dense_pass as fixtures

    device = "cuda:0"
    dequant = fixtures.k3_common.dequant_mxfp4
    packed, scale = synthetic_dequant_inputs()
    result = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
              "synthetic_dequant": compare_dequant(dequant, packed, scale, device)}
    x = (torch.arange(64).reshape(4, 16).float() / 16 - 2).to(device, torch.bfloat16)
    identity = torch.eye(16, device=device, dtype=torch.bfloat16)
    y = torch.nn.functional.linear(x, identity)
    if not torch.equal(x, y):
        raise RuntimeError("BF16 CUDA identity-linear canary differs")
    result["bf16_identity_linear"] = {"equal": True, "shape": list(y.shape)}
    if args.parent is not None:
        index = fixtures.k3_common.K3Index(str(args.parent))
        keys = [key for key in index.layer_keys(1) if ".experts.0." in key]
        packed_keys = [key for key in keys if key.endswith("weight_packed")]
        if len(keys) != 6 or len(packed_keys) != 3:
            raise RuntimeError("real L1 expert0 does not have the expected three packed/scale pairs")
        tensors, timing = index.read_keys(keys, progress_tag="canary-L01-E000")
        result["real_l1_expert0"] = {
            "matrices": {key: compare_dequant(dequant, tensors[key],
                                              tensors[key.replace("weight_packed", "weight_scale")], device)
                         for key in packed_keys},
            "read": timing, "scope": "one expert dequant only; no model forward or intelligence claim"}
        del tensors, index
    args.scratch.mkdir(parents=True, exist_ok=True)
    uniform = fixtures.uniform_fixture.__wrapped__(args.scratch)
    corpus = fixtures.corpus_files.__wrapped__(args.scratch, uniform)
    cfg = uniform["index"].text_config
    width = int(cfg["num_experts"]) * int(cfg["moe_intermediate_size"])
    argv = fixtures.base_argv(uniform["dir"], corpus, args.scratch / "fold", args.scratch / "receipts",
                              rung=width, device=device, experts_device=device, compute_dtype="fp32",
                              acceptance="gate", chimera_src=args.chimera / "src")
    receipt = fixtures.run_pass(argv)
    if receipt["layers_folded"] != [1, 2]:
        raise RuntimeError("exact-fold canary did not complete both synthetic MoE layers")
    layers = {}
    for layer in ("1", "2"):
        entry = receipt["layers"][layer]
        if (entry["width_installed"] != width or entry["acceptance"]["verdict"] != "accepted"
                or entry["fit_relL2_heldout"] >= 1e-4
                or entry["stream_divergence"]["held"]["relL2"] >= 1e-4):
            raise RuntimeError(f"L{layer} fails the existing full-width FP32 exact-fold contract")
        layers[layer] = {"width": width, "accepted": True,
                         "fit_relL2_heldout": entry["fit_relL2_heldout"],
                         "prefix_relL2_heldout": entry["stream_divergence"]["held"]["relL2"]}
    torch.cuda.synchronize()
    result.update({"status": "passed", "fp32_uniform_exact_fold": layers,
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "limitations": "runtime mechanics only; not real K3 fold acceptance, whole-model fit, or intelligence"})
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
