#!/usr/bin/env python3
"""Run a sealed, model-free two-RTX-3090 replay-scheduler smoke test.

Launch this in a fresh process with exactly two RTX 3090s made visible, for
example::

    CUDA_VISIBLE_DEVICES=GPU-<first-3090-uuid>,GPU-<second-3090-uuid> python \
      experiments/smoke_dynamic_replay_scheduler.py \
      --output /absolute/path/replay-scheduler-smoke.json

The workload is fixed synthetic matrix multiplication.  It loads no model,
tokenizer, cache, or hub resource and performs no fitting or training.  The
only published artifact is a canonical success receipt.  Any failed hardware,
determinism, state-identity, parity, scheduling, or receipt check leaves the
requested output path absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "neural-forge-dynamic-replay-scheduler-smoke-v1"
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 3090"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
ABSOLUTE_TOLERANCE = 3.125e-2
RELATIVE_TOLERANCE = 2.0e-2
MAXIMUM_RELATIVE_RMSE = 2.0e-2
MINIMUM_COSINE_SIMILARITY = 9.99e-1
MAXIMUM_PINNED_BYTES_PER_SLOT = 1 << 20
WORK_ITEM_COUNT = 8

# This assignment must precede importing PyTorch or any local module that can
# import it.  run_smoke also rejects a process whose CUDA runtime was already
# initialized, so setting this too late can never produce a success receipt.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG


class SchedulerSmokeError(RuntimeError):
    """The smoke contract failed before success-receipt publication."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode finite, stable, ASCII JSON."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SchedulerSmokeError(
            "The smoke receipt must be finite canonical JSON."
        ) from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_new(path: Path, payload: bytes) -> None:
    """Publish one complete new file without replacing an existing path."""

    destination = path.expanduser().resolve(strict=False)
    if not destination.parent.is_dir():
        raise SchedulerSmokeError(
            "The explicit output parent directory must already exist."
        )
    if destination.exists():
        raise SchedulerSmokeError("The explicit output path already exists.")
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
        # Hard-link publication is create-if-absent and exposes only the
        # already-complete inode.  It therefore cannot overwrite a raced path.
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise SchedulerSmokeError(
            "The explicit output or staging path already exists."
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except OSError:
            # Cleanup cannot invalidate an already-atomically-published
            # success receipt or turn it into a reported failure.
            pass


def _validate_device_documents(
    devices: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if len(devices) != 2:
        raise SchedulerSmokeError(
            "Exactly two logical CUDA devices must be visible."
        )
    ordered = tuple(
        sorted((dict(device) for device in devices), key=lambda row: row["device_index"])
    )
    if tuple(row.get("device_index") for row in ordered) != (0, 1):
        raise SchedulerSmokeError(
            "The two visible CUDA devices must have logical indices 0 and 1."
        )
    if any(row.get("device_name") != EXPECTED_DEVICE_NAME for row in ordered):
        raise SchedulerSmokeError(
            f"Both logical CUDA devices must be {EXPECTED_DEVICE_NAME!r}."
        )
    uuids = tuple(row.get("device_uuid") for row in ordered)
    if any(not isinstance(value, str) or not value for value in uuids):
        raise SchedulerSmokeError("Each CUDA device must disclose a UUID.")
    if len(set(uuids)) != 2:
        raise SchedulerSmokeError("CUDA device UUIDs must be distinct.")
    pci_identities = []
    for row in ordered:
        identity = tuple(
            row.get(name)
            for name in ("pci_domain_id", "pci_bus_id", "pci_device_id")
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in identity
        ):
            raise SchedulerSmokeError(
                "Each CUDA device must disclose a complete numeric PCI identity."
            )
        pci_identities.append(identity)
    if len(set(pci_identities)) != 2:
        raise SchedulerSmokeError("CUDA PCI identities must be distinct.")
    return ordered


def _validate_execution_report(report: Mapping[str, Any]) -> None:
    parity = report.get("parity_reports")
    if not isinstance(parity, list) or len(parity) != 2:
        raise SchedulerSmokeError(
            "The scheduler did not emit two device parity receipts."
        )
    if {row.get("device_index") for row in parity} != {0, 1}:
        raise SchedulerSmokeError(
            "Parity receipts do not cover both logical CUDA devices."
        )
    for row in parity:
        leaves = row.get("tensor_reports")
        if row.get("passed") is not True or not isinstance(leaves, list):
            raise SchedulerSmokeError("A CUDA parity receipt did not pass.")
        dtypes = {leaf.get("dtype") for leaf in leaves}
        if not {"bfloat16", "float32"}.issubset(dtypes):
            raise SchedulerSmokeError(
                "Each CUDA parity receipt must cover BF16 and FP32 matmul."
            )
        if any(leaf.get("passed") is not True for leaf in leaves):
            raise SchedulerSmokeError("A parity tensor leaf did not pass.")
    records = report.get("scheduled_records")
    if not isinstance(records, list) or len(records) != WORK_ITEM_COUNT:
        raise SchedulerSmokeError(
            "The scheduler did not execute the complete fixed workload."
        )
    if [row.get("ordinal") for row in records] != list(range(WORK_ITEM_COUNT)):
        raise SchedulerSmokeError(
            "Scheduled records are not in canonical ordinal order."
        )
    if {row.get("device_index") for row in records} != {0, 1}:
        raise SchedulerSmokeError(
            "Canonical workload sharding did not exercise both devices."
        )
    if (
        report.get("parity_passed_before_scheduled_work") is not True
        or report.get("canonical_result_order_restored") is not True
        or report.get("backend_receipts_reauthenticated") is not True
        or report.get("installed_state_reauthenticated") is not True
        or report.get("training_steps") != 0
        or report.get("backward_calls") != 0
        or report.get("optimizer_steps") != 0
        or report.get("folding_math_changed") is not False
    ):
        raise SchedulerSmokeError(
            "The scheduler execution receipt violates the smoke contract."
        )


def _sealed_success_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(body)
    if document.get("schema") != SCHEMA or document.get("status") != "passed":
        raise SchedulerSmokeError("Only a successful smoke body can be sealed.")
    return {**document, "receipt_sha256": canonical_sha256(document)}


def _configure_torch() -> Any:
    import torch

    if torch.cuda.is_initialized():
        raise SchedulerSmokeError(
            "CUDA was initialized before the deterministic smoke contract."
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise SchedulerSmokeError("CUBLAS_WORKSPACE_CONFIG changed unexpectedly.")
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


def _cpu_installed_state(torch: Any) -> dict[str, Any]:
    values = (
        (torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16) % 8 + 1)
        / 64.0
    )
    return {
        "matmul.weight_bfloat16": values.to(dtype=torch.bfloat16).contiguous(),
        "matmul.weight_float32": values.contiguous(),
    }


def _payload(torch: Any, ordinal: int) -> dict[str, Any]:
    values = (
        (
            torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
            + 1
            + ordinal
        )
        % 16
        + 1
    ) / 32.0
    return {
        "input_bfloat16": values.to(dtype=torch.bfloat16).contiguous(),
        "input_float32": values.contiguous(),
        "synthetic_ordinal": ordinal,
    }


def run_smoke(output_path: Path) -> dict[str, Any]:
    destination = output_path.expanduser().resolve(strict=False)
    if destination.exists():
        raise SchedulerSmokeError("The explicit output path already exists.")

    torch = _configure_torch()
    from neural_forge import dynamic_replay_scheduler as replay

    if int(torch.cuda.device_count()) != 2:
        raise SchedulerSmokeError(
            "Exactly two logical CUDA devices must be visible."
        )
    device_receipts = tuple(
        replay.capture_cuda_replay_device_receipt(index) for index in (0, 1)
    )
    device_documents = _validate_device_documents(
        tuple(receipt.as_dict() for receipt in device_receipts)
    )

    cpu_state = _cpu_installed_state(torch)
    cpu_fingerprint = replay.fingerprint_installed_replay_state(cpu_state)
    gpu_states: dict[int, dict[str, Any]] = {}
    replicas = []
    for device_receipt in device_receipts:
        index = device_receipt.device_index
        with torch.cuda.device(index), torch.inference_mode():
            installed = {
                name: value.to(
                    device=torch.device("cuda", index),
                    non_blocking=False,
                    copy=True,
                )
                for name, value in cpu_state.items()
            }
        fingerprint = replay.fingerprint_installed_replay_state(installed)
        if fingerprint != cpu_fingerprint:
            raise SchedulerSmokeError(
                f"cuda:{index} did not receive byte-identical installed state."
            )
        gpu_states[index] = installed
        replicas.append(
            replay.ReplayReplicaReceipt.create(
                device=device_receipt,
                installed_state=fingerprint,
            )
        )

    policy = replay.ReplaySchedulerPolicy(
        expected_device_count=2,
        expected_device_name=EXPECTED_DEVICE_NAME,
        maximum_pinned_bytes_per_slot=MAXIMUM_PINNED_BYTES_PER_SLOT,
        parity_absolute_tolerance=ABSOLUTE_TOLERANCE,
        parity_relative_tolerance=RELATIVE_TOLERANCE,
        parity_maximum_relative_rmse=MAXIMUM_RELATIVE_RMSE,
        parity_minimum_cosine_similarity=MINIMUM_COSINE_SIMILARITY,
        require_deterministic_debug_mode=True,
    )

    def _matmul(
        payload: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        work_id: str,
    ) -> dict[str, Any]:
        return {
            "bfloat16_matmul": torch.matmul(
                payload["input_bfloat16"],
                state["matmul.weight_bfloat16"],
            ),
            "float32_matmul": torch.matmul(
                payload["input_float32"],
                state["matmul.weight_float32"],
            ),
            "operation": "synthetic-installed-state-matmul",
            "work_id": work_id,
        }

    def gpu_worker(item: Any, replica: Any, payload: Any) -> Any:
        index = replica.device.device_index
        with torch.cuda.device(index), torch.inference_mode():
            return _matmul(payload, gpu_states[index], work_id=item.work_id)

    def cpu_worker(item: Any, payload: Any) -> Any:
        with torch.inference_mode():
            return _matmul(payload, cpu_state, work_id=item.work_id)

    scheduler = replay.DeterministicReplayScheduler(
        replicas=tuple(replicas),
        policy=policy,
        gpu_worker=gpu_worker,
        cpu_parity_worker=cpu_worker,
        device_receipt_probe=replay.capture_cuda_replay_device_receipt,
        installed_state_probe=lambda replica: (
            replay.fingerprint_installed_replay_state(
                gpu_states[replica.device.device_index]
            )
        ),
        cpu_installed_state_probe=lambda: (
            replay.fingerprint_installed_replay_state(cpu_state)
        ),
    )
    shard_keys = (
        "trajectory-a",
        "candidate-b",
        "trajectory-c",
        "trajectory-a",
        "candidate-b",
        "trajectory-c",
        "candidate-d",
        "candidate-d",
    )
    items = tuple(
        replay.ReplayWorkItem.create(
            ordinal=ordinal,
            work_id=f"synthetic-{ordinal}",
            shard_key=shard_keys[ordinal],
            payload=_payload(torch, ordinal),
        )
        for ordinal in range(WORK_ITEM_COUNT)
    )
    parity_item = replay.ReplayWorkItem.create(
        ordinal=0,
        work_id="cpu-cuda-matmul-parity",
        shard_key="parity",
        payload=_payload(torch, 31),
    )
    # Reversed caller order proves that publication order comes from canonical
    # ordinals rather than input ordering or thread completion timing.
    execution = scheduler.execute(tuple(reversed(items)), parity_item=parity_item)
    execution_report = execution.as_dict()
    _validate_execution_report(execution_report)

    scheduler_source = Path(replay.__file__).resolve()
    receipt = _sealed_success_receipt(
        {
            "schema": SCHEMA,
            "status": "passed",
            "scope": {
                "local_only": True,
                "models_loaded": 0,
                "tokenizers_loaded": 0,
                "hub_requests": 0,
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
            },
            "determinism": {
                "configured_before_cuda_initialization": True,
                "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
                "synthetic_input_construction": "fixed-arange-no-rng",
                "policy": policy.as_dict(),
            },
            "numeric_parity_tolerances": {
                "applies_to": ["bfloat16_matmul", "float32_matmul"],
                "absolute_tolerance": ABSOLUTE_TOLERANCE,
                "relative_tolerance": RELATIVE_TOLERANCE,
                "maximum_relative_rmse": MAXIMUM_RELATIVE_RMSE,
                "minimum_cosine_similarity": MINIMUM_COSINE_SIMILARITY,
            },
            "hardware": {
                "logical_device_count": 2,
                "expected_device_name": EXPECTED_DEVICE_NAME,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "devices": list(device_documents),
            },
            "installed_state": {
                "construction": "independent-device-copies-of-fixed-cpu-state",
                "tensor_names": sorted(cpu_state),
                "cpu_reference": cpu_fingerprint.as_dict(),
                "replica_states_byte_identical": True,
                "replicas": [replica.as_dict() for replica in replicas],
            },
            "workload": {
                "operation": "installed-state-bfloat16-and-float32-matmul",
                "scheduled_item_count": WORK_ITEM_COUNT,
                "caller_order_reversed": True,
                "shard_keys_by_ordinal": list(shard_keys),
                "actual_pinned_host_double_buffer": True,
                "maximum_pinned_bytes_per_slot": (
                    MAXIMUM_PINNED_BYTES_PER_SLOT
                ),
            },
            "execution": execution_report,
            "software": {
                "runner_sha256": _source_sha256(Path(__file__).resolve()),
                "scheduler_sha256": _source_sha256(scheduler_source),
            },
        }
    )
    _atomic_write_new(destination, canonical_json_bytes(receipt) + b"\n")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Explicit new path for the canonical success receipt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = run_smoke(args.output)
    except Exception as exc:
        print(f"scheduler smoke failed closed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve(strict=False)),
                "receipt_sha256": receipt["receipt_sha256"],
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
