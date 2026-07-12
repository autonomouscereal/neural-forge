#!/usr/bin/env python3
"""Snapshot and detach a zero-gradient functional-fold plus full-eval run."""

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
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--source-model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--functional-target", choices=["module-output", "module-all"], default="module-output")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-tokens-per-layer", type=int, default=4096)
    parser.add_argument("--max-anchors", type=int, default=20_000)
    parser.add_argument("--max-calibration-texts", type=int, default=1024)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--fit-gate-validation-fraction", type=float, default=0.25)
    parser.add_argument("--fit-gate-min-validation-rows", type=int, default=128)
    parser.add_argument("--fit-gate-min-relative-improvement", type=float, default=0.02)
    parser.add_argument("--fit-gate-max-cosine-drop", type=float, default=0.0)
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--wait-poll-seconds", type=int, default=60)
    parser.add_argument("--source-script", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.wait_poll_seconds <= 0:
        raise ValueError("wait-poll-seconds must be positive.")
    if any(pid <= 0 for pid in args.wait_pid):
        raise ValueError("wait-pid values must be positive.")
    candidate = args.candidate.resolve(strict=True)
    run_root = args.run_root.resolve(strict=True)
    required = [
        candidate / "config.json",
        candidate / "model.safetensors",
        candidate / "neural_forge_hidden_basis.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"candidate is missing required fold artifacts: {missing}")

    repo = Path(__file__).resolve().parents[1]
    source = args.source_script or repo / "experiments" / "run_hf_functional_fold_eval.sh"
    source = source.resolve(strict=True)
    calibration_file = (args.calibration_file or run_root / "calibration_mix_plus_recovery_drills_20260702.jsonl").resolve(
        strict=True
    )
    output = run_root / args.tag
    output.mkdir(exist_ok=False)
    (output / "reports").mkdir()
    snapshot = output / "run_snapshot.sh"
    snapshot.write_bytes(source.read_bytes())
    os.chmod(snapshot, 0o500)

    environment = {
        "REPO": str(repo),
        "RUN_ROOT": str(run_root),
        "TAG": args.tag,
        "SOURCE_MODEL": args.source_model,
        "CANDIDATE_MODEL": str(candidate),
        "GPU_UUID": args.gpu_uuid,
        "FUNCTIONAL_TARGET": args.functional_target,
        "ALPHA": str(args.alpha),
        "RIDGE": str(args.ridge),
        "MAX_LENGTH": str(args.max_length),
        "MAX_TOKENS_PER_LAYER": str(args.max_tokens_per_layer),
        "MAX_ANCHORS": str(args.max_anchors),
        "MAX_CALIBRATION_TEXTS": str(args.max_calibration_texts),
        "CALIBRATION_FILE": str(calibration_file),
        "FIT_GATE_VALIDATION_FRACTION": str(args.fit_gate_validation_fraction),
        "FIT_GATE_MIN_VALIDATION_ROWS": str(args.fit_gate_min_validation_rows),
        "FIT_GATE_MIN_RELATIVE_IMPROVEMENT": str(args.fit_gate_min_relative_improvement),
        "FIT_GATE_MAX_COSINE_DROP": str(args.fit_gate_max_cosine_drop),
        "WAIT_PIDS": " ".join(str(pid) for pid in args.wait_pid),
        "WAIT_POLL_SECONDS": str(args.wait_poll_seconds),
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
        "schema": "neural-forge-functional-fold-run-v1",
        "candidate": str(candidate),
        "candidate_config_sha256": _sha256(candidate / "config.json"),
        "candidate_fold_basis_sha256": _sha256(candidate / "neural_forge_hidden_basis.safetensors"),
        "calibration_file": str(calibration_file),
        "calibration_sha256": _sha256(calibration_file),
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
