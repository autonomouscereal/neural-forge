#!/usr/bin/env python3
"""Create an immutable Qwen path-fold run and launch it without SSH FD inheritance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    from experiments.launch_detached import launch_detached
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.launch_detached import launch_detached


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument(
        "--hidden-basis-mode",
        choices=["activation-coordinate", "activation-grouped", "activation-oblique", "activation-token-pca"],
        required=True,
    )
    parser.add_argument(
        "--hidden-basis-preprocess",
        choices=["centered-rowrms", "uncentered-rowrms", "uncentered"],
        default="uncentered-rowrms",
    )
    parser.add_argument("--hidden-basis-ridge", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-basis-ridge-mode",
        choices=["absolute", "trace-scaled"],
        default="trace-scaled",
    )
    parser.add_argument("--activation-fold-assignment", default="balanced-similarity")
    parser.add_argument("--activation-fit-linears", action="store_true")
    parser.add_argument(
        "--activation-fit-target",
        choices=["all", "attention", "attention-output", "mlp", "mlp-output", "input-projections"],
        default="input-projections",
    )
    parser.add_argument("--activation-fit-max-rows", type=int, default=4096)
    parser.add_argument("--activation-fit-alpha", type=float, default=0.5)
    parser.add_argument("--activation-fit-ridge", type=float, default=1.0)
    parser.add_argument("--activation-fit-max-delta-rms-ratio", type=float, default=2.0)
    parser.add_argument("--transport-support", default="importance")
    parser.add_argument("--fold-stat-repair", choices=["none", "rms"], default="none")
    parser.add_argument(
        "--donor-layer-fold-mode",
        choices=["representative", "average", "fold"],
        default="representative",
    )
    parser.add_argument(
        "--donor-layer-group-strategy",
        choices=[
            "depth",
            "activation-energy",
            "activation-geometry",
            "activation-ot",
            "activation-cka",
            "activation-cka-weighted",
        ],
        default="depth",
    )
    parser.add_argument(
        "--donor-layer-output-scale",
        choices=["group-count", "effective-count", "sqrt-effective-count", "unit"],
        default="unit",
    )
    parser.add_argument("--untie-output-head", action="store_true")
    parser.add_argument("--source-script", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = args.target.resolve(strict=True)
    run_root = args.run_root.resolve(strict=True)
    if not (target / "config.json").is_file() or not (target / "model.safetensors").is_file():
        raise ValueError("target must contain config.json and model.safetensors.")
    repo = Path(__file__).resolve().parents[1]
    source = args.source_script or repo / "experiments" / "run_qwen4b_blank_qwen146m_path_conjugate_20260709.sh"
    source = source.resolve(strict=True)
    output = run_root / args.tag
    output.mkdir(exist_ok=False)
    (output / "reports").mkdir()
    snapshot = output / "run_snapshot.sh"
    snapshot.write_bytes(source.read_bytes())
    os.chmod(snapshot, 0o500)
    environment = {
        "REPO": str(repo),
        "RUN_ROOT": str(run_root),
        "TARGET_MODEL": str(target),
        "TAG": args.tag,
        "GPU_UUID": args.gpu_uuid,
        "HIDDEN_BASIS_MODE": args.hidden_basis_mode,
        "HIDDEN_BASIS_PREPROCESS": args.hidden_basis_preprocess,
        "HIDDEN_BASIS_RIDGE": str(args.hidden_basis_ridge),
        "HIDDEN_BASIS_RIDGE_MODE": args.hidden_basis_ridge_mode,
        "ACTIVATION_FOLD_ASSIGNMENT": args.activation_fold_assignment,
        "ACTIVATION_FIT_LINEARS": "1" if args.activation_fit_linears else "0",
        "ACTIVATION_FIT_TARGET": args.activation_fit_target,
        "ACTIVATION_FIT_MAX_ROWS": str(args.activation_fit_max_rows),
        "ACTIVATION_FIT_ALPHA": str(args.activation_fit_alpha),
        "ACTIVATION_FIT_RIDGE": str(args.activation_fit_ridge),
        "ACTIVATION_FIT_MAX_DELTA_RMS_RATIO": str(args.activation_fit_max_delta_rms_ratio),
        "TRANSPORT_SUPPORT": args.transport_support,
        "DONOR_LAYER_GROUP_STRATEGY": args.donor_layer_group_strategy,
        "DONOR_LAYER_FOLD_MODE": args.donor_layer_fold_mode,
        "DONOR_LAYER_OUTPUT_SCALE": args.donor_layer_output_scale,
        "FOLD_STAT_REPAIR": args.fold_stat_repair,
        "UNTIE_OUTPUT_HEAD": "1" if args.untie_output_head else "0",
    }
    identity = launch_detached(
        ["bash", str(snapshot)],
        log_path=output / "run.log",
        pid_path=output / "pid",
        identity_path=output / "process_identity.json",
        cwd=repo,
        environment=environment,
    )
    manifest = {
        "schema": "neural-forge-path-run-v1",
        "target": str(target),
        "target_config_sha256": _sha256(target / "config.json"),
        "snapshot": str(snapshot),
        "snapshot_sha256": _sha256(snapshot),
        "environment": environment,
        "process": identity,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
