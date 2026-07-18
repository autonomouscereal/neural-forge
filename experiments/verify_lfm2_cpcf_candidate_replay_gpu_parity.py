#!/usr/bin/env python3
"""Verify real two-GPU CPCF candidate replay against the CPU reference.

This is a model-free representative fixture for the optional LFM2 CPCF replay
backend.  It performs no fitting, backward pass, optimizer step, pruning, or
quantization.  Run it in a fresh process, for example::

    CUDA_VISIBLE_DEVICES=<first-rtx3090-uuid>,<second-rtx3090-uuid> \
    python experiments/verify_lfm2_cpcf_candidate_replay_gpu_parity.py \
      --devices 0 1 \
      --output /tmp/lfm2-cpcf-candidate-replay-parity.json

The requested output is created only after the deterministic scheduler,
representative CPU/CUDA tensor parity, complete candidate frontier,
reauthentication, and finalized metric parity all pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


SCHEMA = "neural-forge-lfm2-cpcf-candidate-gpu-parity-v1"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG


class CandidateGpuParityError(RuntimeError):
    """The representative GPU replay contract failed closed."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CandidateGpuParityError(
            "Parity receipt must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _atomic_write_new(path: Path, payload: bytes) -> None:
    destination = path.expanduser().resolve(strict=False)
    if not destination.parent.is_dir():
        raise CandidateGpuParityError(
            "Output parent directory must already exist."
        )
    if destination.exists():
        raise CandidateGpuParityError("Output path already exists.")
    temporary = destination.with_name(
        f".{destination.name}.stage-{os.getpid()}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise CandidateGpuParityError(
            "Output or staging path already exists."
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except OSError:
            pass


def _configure_torch() -> Any:
    import torch

    if torch.cuda.is_initialized():
        raise CandidateGpuParityError(
            "CUDA initialized before deterministic flags were pinned."
        )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    return torch


def _fixture(torch: Any, *, target_dtype: str):
    from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
        Lfm2CpcfShortConvV4SolverPolicy,
        Lfm2CpcfV4ProjectionCase,
    )

    generator = torch.Generator().manual_seed(20260718)
    dtype = getattr(torch, target_dtype)
    hidden = 16
    states = {}
    base = {
        "in_proj_weight": torch.randn(
            (3 * hidden, hidden),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype),
        "in_proj_bias": torch.randn(
            (3 * hidden,),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype),
        "fir_kernel": torch.randn(
            (hidden, 4),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype),
        "fir_bias": torch.randn(
            (hidden,),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype),
        "writer_weight": torch.randn(
            (hidden, hidden),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype),
        "writer_bias": torch.randn(
            (hidden,),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype),
    }
    for candidate_index, offset in enumerate((0.0, 0.005, 0.01, 0.02)):
        states[f"candidate-{candidate_index:02d}"] = {
            role: (
                None
                if value is None
                else (value.to(torch.float32) + offset)
                .to(dtype)
                .contiguous()
            )
            for role, value in base.items()
        }
    cases = []
    for case_index, sequence in enumerate((32, 40, 48)):
        suffix_start = case_index
        child = torch.randn(
            (4, sequence, hidden),
            generator=generator,
            dtype=torch.float64,
        )
        desired_context = torch.randn(
            child.shape,
            generator=generator,
            dtype=torch.float64,
        )
        desired_write = torch.randn(
            (4, sequence - suffix_start, hidden),
            generator=generator,
            dtype=torch.float64,
        )
        mask = torch.ones((4, sequence), dtype=torch.bool)
        mask[-1, -2:] = False
        cases.append(
            Lfm2CpcfV4ProjectionCase(
                case_id=f"selection-{case_index:02d}",
                split="selection",
                child_input_history=child,
                desired_context_history=desired_context,
                desired_write_suffix=desired_write,
                suffix_start=suffix_start,
                valid_prefix_mask=mask,
                valid_suffix_mask=mask[:, suffix_start:],
            )
        )
    policy = Lfm2CpcfShortConvV4SolverPolicy(
        ridge_alphas=(1e-8, 1e-4),
        variable_projection_passes=2,
        target_dtype=target_dtype,
    )
    return states, tuple(cases), policy


def _direct_report(state: Any, cases: Any, policy: Any) -> Any:
    from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
        evaluate_candidate_state_target_dtype,
    )

    return evaluate_candidate_state_target_dtype(
        cases=cases,
        in_proj_weight=state["in_proj_weight"],
        in_proj_bias=state["in_proj_bias"],
        fir_kernel=state["fir_kernel"],
        fir_bias=state["fir_bias"],
        writer_weight=state["writer_weight"],
        writer_bias=state["writer_bias"],
        required_split="selection",
        policy=policy,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    torch = _configure_torch()
    from neural_forge.dynamic_replay_scheduler import (
        DeterministicReplayScheduler,
        ReplayReplicaReceipt,
        ReplaySchedulerPolicy,
        capture_cuda_replay_device_receipt,
        fingerprint_installed_replay_state,
    )
    from neural_forge.hf_lfm2_cpcf_candidate_replay import (
        AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator,
        evaluate_lfm2_cpcf_candidate_states_with_backend,
        replay_lfm2_cpcf_candidate_work_payload,
    )

    devices = tuple(args.devices)
    if (
        len(devices) != 2
        or len(set(devices)) != 2
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= torch.cuda.device_count()
            for index in devices
        )
    ):
        raise CandidateGpuParityError(
            "--devices must name two distinct available CUDA indices."
        )
    states, cases, solver_policy = _fixture(
        torch,
        target_dtype=args.target_dtype,
    )
    direct = {
        candidate_id: _direct_report(state, cases, solver_policy)
        for candidate_id, state in sorted(states.items())
    }

    contract_cpu = {
        "candidate_replay_contract": torch.arange(
            64,
            dtype=torch.float32,
        )
    }
    contract_gpu = {
        index: {
            name: value.to(device=f"cuda:{index}")
            for name, value in contract_cpu.items()
        }
        for index in devices
    }
    cpu_fingerprint = fingerprint_installed_replay_state(contract_cpu)
    device_receipts = {
        index: capture_cuda_replay_device_receipt(index)
        for index in devices
    }
    gpu_fingerprints = {
        index: fingerprint_installed_replay_state(contract_gpu[index])
        for index in devices
    }
    if any(
        fingerprint != cpu_fingerprint
        for fingerprint in gpu_fingerprints.values()
    ):
        raise CandidateGpuParityError(
            "CPU and CUDA replay contract fingerprints differ."
        )
    replicas = tuple(
        ReplayReplicaReceipt.create(
            device=device_receipts[index],
            installed_state=gpu_fingerprints[index],
        )
        for index in devices
    )
    scheduler_policy = ReplaySchedulerPolicy(
        expected_device_name=device_receipts[devices[0]].device_name,
        maximum_pinned_bytes_per_slot=args.maximum_pinned_mib << 20,
        parity_absolute_tolerance=args.parity_absolute_tolerance,
        parity_relative_tolerance=args.parity_relative_tolerance,
        parity_maximum_relative_rmse=args.parity_maximum_relative_rmse,
        parity_minimum_cosine_similarity=(
            args.parity_minimum_cosine_similarity
        ),
    )
    scheduler = DeterministicReplayScheduler(
        replicas=replicas,
        policy=scheduler_policy,
        gpu_worker=lambda _item, _replica, payload: (
            replay_lfm2_cpcf_candidate_work_payload(payload)
        ),
        cpu_parity_worker=lambda _item, payload: (
            replay_lfm2_cpcf_candidate_work_payload(payload)
        ),
        device_receipt_probe=capture_cuda_replay_device_receipt,
        installed_state_probe=lambda replica: (
            fingerprint_installed_replay_state(
                contract_gpu[replica.device.device_index]
            )
        ),
        cpu_installed_state_probe=lambda: (
            fingerprint_installed_replay_state(contract_cpu)
        ),
    )
    scheduled, backend, items = (
        evaluate_lfm2_cpcf_candidate_states_with_backend(
            states,
            cases=cases,
            required_split="selection",
            policy=solver_policy,
            evaluator=(
                AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator(
                    scheduler
                )
            ),
        )
    )
    records = backend["execution_report"]["scheduled_records"]
    if (
        backend.get("representative_cpu_cuda_parity_passed") is not True
        or backend["execution_report"].get(
            "canonical_result_order_restored"
        )
        is not True
        or backend["execution_report"].get(
            "backend_receipts_reauthenticated"
        )
        is not True
        or backend["execution_report"].get(
            "installed_state_reauthenticated"
        )
        is not True
        or {record["device_index"] for record in records} != set(devices)
    ):
        raise CandidateGpuParityError(
            "Scheduled replay receipt does not prove both authenticated GPUs."
        )
    metric_names = (
        "mean_case_relative_rmse",
        "maximum_case_relative_rmse",
        "mean_case_cosine_similarity",
        "minimum_case_cosine_similarity",
        "mean_case_context_relative_rmse",
        "maximum_case_context_relative_rmse",
        "mean_case_context_cosine_similarity",
        "minimum_case_context_cosine_similarity",
    )
    metric_deltas = {
        candidate_id: {
            name: abs(
                float(scheduled[candidate_id][name])
                - float(direct[candidate_id][name])
            )
            for name in metric_names
        }
        for candidate_id in sorted(states)
    }
    maximum_metric_delta = max(
        delta
        for candidate in metric_deltas.values()
        for delta in candidate.values()
    )
    if maximum_metric_delta > args.maximum_metric_delta:
        raise CandidateGpuParityError(
            "Finalized CPU/CUDA candidate metrics exceed the fixed tolerance."
        )
    body = {
        "schema": SCHEMA,
        "status": "passed",
        "target_dtype": args.target_dtype,
        "candidate_count": len(states),
        "case_count": len(cases),
        "devices": [
            device_receipts[index].as_dict() for index in devices
        ],
        "solver_policy_sha256": solver_policy.authentication_sha256,
        "candidate_work_items": [item.as_dict() for item in items],
        "backend_report": dict(backend),
        "direct_cpu_report_sha256": {
            candidate_id: direct[candidate_id]["report_sha256"]
            for candidate_id in sorted(direct)
        },
        "scheduled_cuda_report_sha256": {
            candidate_id: scheduled[candidate_id]["report_sha256"]
            for candidate_id in sorted(scheduled)
        },
        "candidate_state_sha256_equal": all(
            direct[candidate_id]["candidate_state_sha256"]
            == scheduled[candidate_id]["candidate_state_sha256"]
            for candidate_id in states
        ),
        "metric_deltas": metric_deltas,
        "maximum_metric_delta": maximum_metric_delta,
        "maximum_allowed_metric_delta": args.maximum_metric_delta,
        "closed_form_solve_math_changed": False,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                root
                / "src"
                / "neural_forge"
                / "hf_lfm2_cpcf_candidate_replay.py",
                root
                / "src"
                / "neural_forge"
                / "dynamic_replay_scheduler.py",
            )
        },
    }
    return {**body, "receipt_sha256": _sha(body)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs=2, type=int, default=(0, 1))
    parser.add_argument(
        "--target-dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--maximum-pinned-mib", type=int, default=256)
    parser.add_argument(
        "--parity-absolute-tolerance",
        type=float,
        default=0.125,
    )
    parser.add_argument(
        "--parity-relative-tolerance",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--parity-maximum-relative-rmse",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--parity-minimum-cosine-similarity",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--maximum-metric-delta",
        type=float,
        default=0.05,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.maximum_pinned_mib <= 0 or args.maximum_metric_delta < 0.0:
        raise CandidateGpuParityError(
            "Pinned-memory size must be positive and metric tolerance nonnegative."
        )
    receipt = run(args)
    _atomic_write_new(
        args.output,
        _canonical_json(receipt) + b"\n",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "maximum_metric_delta": receipt["maximum_metric_delta"],
                "output": str(args.output.expanduser().resolve(strict=False)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
