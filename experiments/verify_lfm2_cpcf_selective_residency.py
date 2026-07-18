#!/usr/bin/env python3
"""Verify bounded selective CPCF evidence residency on a real local cache.

This verifier never loads a parent model and never runs either parent or
target inference. It authenticates an existing committed CPCF trace cache,
builds the tensor-free selective catalog, walks every source layer through the
current-plus-next residency window, releases each payload in canonical order,
and records both logical residency and Linux process RSS.

The output is a diagnostic infrastructure receipt. It does not claim that the
current model compiler uses the selective provider; that requires a distinct
compiler/resume fingerprint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = "neural-forge-lfm2-cpcf-selective-residency-verification-v1"


class SelectiveResidencyVerificationError(RuntimeError):
    """The cache, residency walk, resource receipt, or output failed."""


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
        raise SelectiveResidencyVerificationError(
            "Verification receipt must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_new(path: Path, payload: bytes) -> None:
    destination = path.expanduser().resolve(strict=False)
    if not destination.parent.is_dir():
        raise SelectiveResidencyVerificationError(
            "Output parent directory must already exist."
        )
    if destination.exists():
        raise SelectiveResidencyVerificationError(
            "Output path already exists."
        )
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
        raise SelectiveResidencyVerificationError(
            "Output or staging path already exists."
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _linux_process_memory() -> dict[str, int]:
    status = Path("/proc/self/status")
    if not status.is_file():
        raise SelectiveResidencyVerificationError(
            "Physical RSS verification requires Linux /proc/self/status."
        )
    values: dict[str, int] = {}
    for line in status.read_text(encoding="ascii").splitlines():
        if not line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
            continue
        name, raw = line.split(":", 1)
        fields = raw.split()
        if len(fields) != 2 or fields[1] != "kB":
            raise SelectiveResidencyVerificationError(
                f"Unexpected {name} unit in /proc/self/status."
            )
        values[name] = int(fields[0]) * 1024
    if set(values) != {"VmRSS", "VmHWM", "VmSize"}:
        raise SelectiveResidencyVerificationError(
            "Linux process memory fields are incomplete."
        )
    return {
        "rss_bytes": values["VmRSS"],
        "peak_rss_bytes": values["VmHWM"],
        "virtual_bytes": values["VmSize"],
    }


def _maximum_pair_bytes(references: tuple[Any, ...]) -> int:
    return max(
        reference.logical_bytes
        + (
            0
            if index + 1 == len(references)
            else references[index + 1].logical_bytes
        )
        for index, reference in enumerate(references)
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from neural_forge.dynamic_evidence_residency import (
        DynamicEvidenceResidencyWindow,
    )
    from neural_forge.hf_lfm2_circuit_trace_cache import (
        open_lfm2_circuit_trace_cache,
    )
    from neural_forge.hf_lfm2_cpcf_selective_evidence import (
        build_lfm2_cpcf_selective_evidence_provider,
    )

    cache_path = args.trace_cache_directory.expanduser().resolve(strict=True)
    if cache_path.is_symlink() or not cache_path.is_dir():
        raise SelectiveResidencyVerificationError(
            "Trace cache must be a real local directory."
        )
    started = time.monotonic()
    samples = [
        {
            "stage": "before_cache_open",
            "elapsed_seconds": 0.0,
            **_linux_process_memory(),
        }
    ]
    cache = open_lfm2_circuit_trace_cache(cache_path, verify=False)
    samples.append(
        {
            "stage": "after_cache_open",
            "elapsed_seconds": time.monotonic() - started,
            **_linux_process_memory(),
        }
    )
    provider = build_lfm2_cpcf_selective_evidence_provider(cache)
    try:
        catalog = provider.dynamic_catalog
        references = catalog.references
        required_pair_bytes = _maximum_pair_bytes(references)
        maximum_resident_bytes = (
            required_pair_bytes
            if args.maximum_logical_resident_gib == 0.0
            else int(args.maximum_logical_resident_gib * 1024**3)
        )
        if maximum_resident_bytes < required_pair_bytes:
            raise SelectiveResidencyVerificationError(
                "Configured logical residency limit is below the largest "
                "current-plus-next layer pair."
            )
        samples.append(
            {
                "stage": "after_catalog_build",
                "elapsed_seconds": time.monotonic() - started,
                **_linux_process_memory(),
            }
        )
        layer_receipts = []
        with DynamicEvidenceResidencyWindow(
            catalog,
            provider,
            maximum_resident_bytes=maximum_resident_bytes,
        ) as window:
            resident = window.start()
            while resident is not None:
                prefetched = window.wait_for_prefetch()
                authentication = window.verify_current()
                layer_receipts.append(
                    {
                        "layer_index": resident.layer_index,
                        "reference_sha256": (
                            resident.reference.reference_sha256
                        ),
                        "evidence_sha256": (
                            resident.reference.evidence_sha256
                        ),
                        "authentication_sha256": (
                            authentication.authentication_sha256
                        ),
                        "logical_bytes": resident.logical_bytes,
                        "prefetched_layer_index": prefetched,
                    }
                )
                samples.append(
                    {
                        "stage": f"layer_{resident.layer_index:03d}_resident",
                        "elapsed_seconds": time.monotonic() - started,
                        "logical_resident_bytes": window.resident_bytes,
                        "occupied_layer_indices": list(
                            window.occupied_layer_indices
                        ),
                        **_linux_process_memory(),
                    }
                )
                resident = window.advance(
                    completed_layer_index=resident.layer_index
                )
            residency_snapshot = dict(window.snapshot())
        gc.collect()
        samples.append(
            {
                "stage": "after_window_release",
                "elapsed_seconds": time.monotonic() - started,
                **_linux_process_memory(),
            }
        )
        final_norm = provider.load_final_norm_payload()
        try:
            final_norm_sha256 = provider.authenticate_final_norm_payload(
                final_norm
            )
            final_norm_logical_bytes = final_norm.logical_bytes
            samples.append(
                {
                    "stage": "final_norm_resident",
                    "elapsed_seconds": time.monotonic() - started,
                    "logical_resident_bytes": final_norm.logical_bytes,
                    **_linux_process_memory(),
                }
            )
        finally:
            provider.release_final_norm_payload(final_norm)
        provider_evidence = dict(provider.selective_session_evidence)
        catalog_receipt = provider.catalog.as_dict()
        selective_catalog_sha256 = provider.catalog.catalog_sha256
        selective_session_sha256 = (
            provider.catalog.selective_session_sha256
        )
        expected_final_norm_sha256 = (
            provider.catalog.final_norm_evidence_sha256
        )
        source_sha256 = provider.source_sha256
        provider_sha256 = provider.provider_sha256
        phase_count = provider.catalog.phase_count
    finally:
        provider.close()
    gc.collect()
    samples.append(
        {
            "stage": "after_provider_close",
            "elapsed_seconds": time.monotonic() - started,
            **_linux_process_memory(),
        }
    )
    expected_indices = list(range(catalog.layer_count))
    maximum_observed_rss = max(
        sample["rss_bytes"] for sample in samples
    )
    maximum_observed_peak_rss = max(
        sample["peak_rss_bytes"] for sample in samples
    )
    physical_limit_bytes = int(args.maximum_process_rss_gib * 1024**3)
    physical_gate_enforced = physical_limit_bytes > 0
    physical_gate_passed = (
        not physical_gate_enforced
        or maximum_observed_peak_rss <= physical_limit_bytes
    )
    if (
        [row["layer_index"] for row in layer_receipts] != expected_indices
        or residency_snapshot["requested_layer_indices"] != expected_indices
        or residency_snapshot["promoted_layer_indices"] != expected_indices
        or residency_snapshot["released_layer_indices"] != expected_indices
        or residency_snapshot["peak_occupied_layers"] > 2
        or residency_snapshot["peak_resident_bytes"]
        > maximum_resident_bytes
        or final_norm_sha256 != expected_final_norm_sha256
        or provider_evidence["authentication_contract"][
            "trace_file_sha256_passes_per_session"
        ]
        != 1
        or provider_evidence["authentication_contract"][
            "unselected_tensor_payloads_loaded"
        ]
        is not False
    ):
        raise SelectiveResidencyVerificationError(
            "Selective residency verification invariants failed."
        )
    body = {
        "schema": SCHEMA,
        "status": "passed" if physical_gate_passed else "rejected",
        "trace_cache_directory": str(cache_path),
        "cache_key": cache.cache_key,
        "batch_count": cache.batch_count,
        "layer_count": catalog.layer_count,
        "phase_count": phase_count,
        "catalog_sha256": catalog.catalog_sha256,
        "selective_catalog_sha256": selective_catalog_sha256,
        "source_sha256": source_sha256,
        "provider_sha256": provider_sha256,
        "selective_session_sha256": selective_session_sha256,
        "total_catalog_logical_bytes": catalog.total_logical_bytes,
        "largest_current_plus_next_logical_bytes": required_pair_bytes,
        "maximum_logical_resident_bytes": maximum_resident_bytes,
        "residency_snapshot": residency_snapshot,
        "layer_receipts": layer_receipts,
        "final_norm_evidence_sha256": final_norm_sha256,
        "final_norm_logical_bytes": final_norm_logical_bytes,
        "physical_rss_gate": {
            "enforced": physical_gate_enforced,
            "maximum_allowed_bytes": (
                None if not physical_gate_enforced else physical_limit_bytes
            ),
            "maximum_observed_current_rss_bytes": maximum_observed_rss,
            "maximum_observed_process_high_water_bytes": (
                maximum_observed_peak_rss
            ),
            "gate_uses_process_high_water": True,
            "passed": physical_gate_passed,
        },
        "process_memory_samples": samples,
        "selective_session_evidence": provider_evidence,
        "catalog_receipt": catalog_receipt,
        "elapsed_seconds": time.monotonic() - started,
        "parent_model_loaded": False,
        "parent_forward_calls": 0,
        "target_forward_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "compiler_integration_claimed": False,
        "source_file_sha256": {
            path.name: _file_sha256(path)
            for path in (
                Path(__file__).resolve(),
                root
                / "src"
                / "neural_forge"
                / "dynamic_evidence_residency.py",
                root
                / "src"
                / "neural_forge"
                / "hf_lfm2_circuit_trace_cache.py",
                root
                / "src"
                / "neural_forge"
                / "hf_lfm2_cpcf_selective_evidence.py",
            )
        },
    }
    return {**body, "receipt_sha256": _sha(body)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-cache-directory",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maximum-logical-resident-gib",
        type=float,
        default=0.0,
        help=(
            "Explicit logical current-plus-next limit. Zero uses the exact "
            "largest catalog pair."
        ),
    )
    parser.add_argument(
        "--maximum-process-rss-gib",
        type=float,
        default=0.0,
        help=(
            "Optional physical RSS gate. Zero records RSS without enforcing "
            "a threshold."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.maximum_logical_resident_gib < 0.0
        or args.maximum_process_rss_gib < 0.0
    ):
        raise SelectiveResidencyVerificationError(
            "Residency limits must be nonnegative."
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
                "maximum_observed_current_rss_bytes": receipt[
                    "physical_rss_gate"
                ]["maximum_observed_current_rss_bytes"],
                "output": str(
                    args.output.expanduser().resolve(strict=False)
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
