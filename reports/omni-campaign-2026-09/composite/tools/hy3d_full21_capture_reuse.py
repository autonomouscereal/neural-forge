#!/usr/bin/env python3
"""Capture one admitted FULL21 block using exact sealed firststage native inputs.

No mesh/image/posterior resampling, parent mesh generation or branch substitute.
Every chosen-block capture still runs the complete unchanged source DiT.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
import c16_capture_hy3d as capture
from c16_hy3d_contract import CAPTURE_SCHEMA, MODEL_PATH, SOURCE_ROOT, inspect_source, stamp, validate_capture
import hy3d_full21_native as native21
from hy3d_full21_contract import block_contract, SOURCE_PRESERVATION


def convert(firststage, index, parent, source_root, output, device):
    block_contract(index)
    source = inspect_source(parent, source_root)
    parent_stamp = stamp(Path(parent))
    # Authentic firststage baseline and joined source surface ownership are
    # validated before reusing any realized latent or context tensors.
    firststage_tensors, rows, tracked, first_body = native21.load_capture(firststage, 0, source)
    del firststage_tensors
    for name in ("hy3d_full21_capture_reuse.py", "hy3d_full21_native.py", "hy3d_full21_contract.py", "hy3d_full21_linalg.py",
                 "c16_capture_hy3d.py", "c16_replay_hy3d.py", "c16_hy3d_contract.py"):
        path = Path(__file__).with_name(name)
        tracked[path] = capture.digest_file(path)
    firststage = Path(firststage).resolve(strict=True)
    output = Path(output).resolve()
    capture.require(not output.exists(), "fresh bounded block capture output required")
    os.environ.update(USE_SAGEATTN="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HY3DGEN_DEBUG="0")
    import yaml
    config = yaml.safe_load(capture.bounded_bytes(Path(source_root) / "configs/dit_config_2_1.yaml"))
    captures, receipts, samples, replay_diagnostics = [], [], [], []
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": capture.SCHEMA, "source_identity": source["source_identity"], "producer": capture.PRODUCER,
              "input_origin": "exact sealed firststage latent/context/surface/posterior; no resampling",
              "firststage_capture_sha256": first_body["file_sha256"], "block_index": index,
              "source_preservation": SOURCE_PRESERVATION, "folded_child_proven": False,
              "mesh_generation": False, "capability_evaluation": False}
    try:
        with capture.native_modules(source_root, include_dit=True) as native, torch.inference_mode():
            with torch.device("meta"):
                model = native["dit"].HunYuanDiTPlain(**config["model"]["params"])
            model = native21.strict_load(model, parent, "model.", device)
            capture.sealed_json(output / "source.json", {**source, "extra_source_sha256": native["extra_source_sha256"]})
            with safe_open(str(firststage / "block-00" / first_body["file"]), framework="pt", device="cpu") as combined:
                for i, row in enumerate(rows):
                    with safe_open(str(firststage / row["file"]), framework="pt", device="cpu") as handle:
                        payload = {name: handle.get_tensor(name) for name in handle.keys()}
                    latent = payload["dit_latent_input"].to(device)
                    sigma = combined.get_tensor("sigma")[i:i + 1].to(device)
                    context = combined.get_tensor("context")[i:i + 1].to(device)
                    captured, final = capture.capture_dit(model, latent, sigma, {"main": context}, index)
                    capture.require(final.shape == payload["dit_output"].shape and final.dtype == payload["dit_output"].dtype,
                                    "native complete-output shape/dtype changed")
                    replay_diagnostics.append({"object_sha256": row["object_sha256"], "split": row["split"],
                          "byte_exact": native21.bit_exact(final, payload["dit_output"]),
                          "scores": native21.scores(final.reshape(-1, final.shape[-1]), payload["dit_output"].reshape(-1, final.shape[-1])),
                          "numeric_veto": False})
                    payload["dit_output"] = final
                    for key in ("self_attention_write", "cross_attention_write"):
                        payload[key] = captured.pop(key)
                    captured.update(surface=payload["surface"], query_source_index=payload["query_source_index"],
                                    latent_xyz=payload["anchor_xyz"])
                    captures.append(captured)
                    sample = {key: row[key] for key in ("object_sha256", "conditioning_sha256", "seed", "split", "geometry_identity")}
                    samples.append(sample)
                    file = capture.sealed_tensors(output / f"sample-{i:04d}.safetensors", payload)
                    receipt = {**sample, **file, "requested_sigma": row["requested_sigma"]}
                    capture.sealed_json(output / f"sample-{i:04d}.json", receipt)
                    receipts.append(receipt)
                    del payload, latent, sigma, context, final
            block_dir = output / f"block-{index:02d}"
            block_dir.mkdir()
            combined = {name: torch.cat([row[name] for row in captures]) for name in captures[0]}
            capture.require(sum(v.numel() * v.element_size() for v in combined.values()) < 2 << 30,
                            "capture exceeds bounded two-GiB contract")
            file = capture.sealed_tensors(block_dir / "capture.safetensors", combined)
            capture.sealed_json(block_dir / "capture.json", {"schema": CAPTURE_SCHEMA, "kind": "dit_block", **file,
                  "source_identity": source["source_identity"], "samples": samples, "producer": capture.PRODUCER,
                  "block_index": index, "attention": "self_then_image_cross", "time_path": "sigma_prepended_token",
                  "position_path": "sampled_anchors_no_dit_rope"})
            checked = validate_capture(block_dir, source)
        for path, digest in tracked.items():
            capture.require(capture.digest_file(path) == digest, "firststage evidence changed during recapture")
        capture.require(parent_stamp == stamp(Path(parent)) and
                        inspect_source(parent, source_root)["source_identity"] == source["source_identity"], "source changed during capture")
        capture.verify_native_dependencies(source_root)
        report.update(samples=receipts, dit_capture=checked, input_file_sha256={str(p): d for p, d in tracked.items()},
                      posterior="sealed_original_mode", field_kind="native_binary_field_logits_not_metric_sdf",
                      seed_scope="reused_exact_realized_firststage_inputs", runtime_versions=capture.runtime_versions(),
                      complete_native_dit_firststage_replay_byte_exact=all(r["byte_exact"] for r in replay_diagnostics),
                      complete_native_dit_firststage_replay_diagnostics=replay_diagnostics,
                      numeric_policy="chimera_omni.owner_finite_diagnostics_only.2026-10-06.v1",
                      finite_replay_mismatch_veto=False)
        capture.sealed_json(output / "capture-run.json", report)
        return report
    except Exception as exc:
        capture.sealed_json(output / "incomplete.json", {**report, "status": "incomplete", "exception_type": type(exc).__name__, "error": str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firststage", required=True)
    parser.add_argument("--block", type=int, choices=range(1, 21), required=True)
    parser.add_argument("--parent-checkpoint", default=MODEL_PATH)
    parser.add_argument("--source-root", default=SOURCE_ROOT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    parser.add_argument("--cpu-threads", type=int, choices=range(1, 33), default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    result = convert(args.firststage, args.block, args.parent_checkpoint, args.source_root, args.output, args.device)
    print(json.dumps({"status": "captured_native_block", "block_index": result["block_index"], "output": args.output, "folded_child_proven": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
