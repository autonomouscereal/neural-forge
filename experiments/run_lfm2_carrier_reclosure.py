"""Compile a complete LFM2 semantic bridge after carrier replants.

This runner is intentionally an offline analytic compiler.  Every consumed
file is a direct-local artifact whose SHA-256 is supplied by the request.  It
does not import Transformers, instantiate a parent model, capture new traces,
train, prune, quantize, or publish a runtime sidecar.

The expensive operation is split by channel and donor-row chunk.  Each C_in
and C_out chunk has a different content-addressed phase identity, binds its own
cached-trace metric, and independently performs residual-OMP support selection
and a float64 ridge solve against the *post-replant* target table.  Therefore a
positive-incidence carrier cannot be published until every donor in the
allocator's authenticated reverse-incidence closure has a replacement route
in both channels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import torch
from safetensors.torch import load_file, save_file

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_counted_child_state import ContinuousChildStateAllocation
from neural_forge.dynamic_fold_phase_cache import (
    DynamicFoldPhaseCache,
    FoldPhaseEventLog,
    FoldPhaseKey,
    VerifiedFoldPhaseCacheEntry,
)
from neural_forge.hf_lfm2_semantic_carrier_allocation import (
    Lfm2SemanticCarrierDemandEvidence,
    build_lfm2_semantic_carrier_demand_evidence,
    build_lfm2_semantic_carrier_inventory,
    compile_lfm2_semantic_carrier_allocation,
    materialize_lfm2_semantic_carrier_allocation,
    select_lfm2_semantic_carriers,
)
from neural_forge.tokenizer_control_contract import (
    compile_local_tokenizer_control_contract,
)


REQUEST_SCHEMA = "neural-forge-lfm2-carrier-reclosure-request-v1"
CARRIER_EVIDENCE_INPUT_SCHEMA = (
    "neural-forge-lfm2-carrier-fold-fit-evidence-input-v1"
)
SOLVER_EVIDENCE_SCHEMA = "neural-forge-lfm2-carrier-solver-evidence-v1"
STANDALONE_BRIDGE_SCHEMA = "neural-forge-lfm2-carrier-reclosed-bridge-v1"
STRUCTURED_LOG_SCHEMA = "neural-forge-lfm2-carrier-reclosure-log-v1"
SOLVER_CHUNK_SCHEMA = "neural-forge-lfm2-carrier-reclosure-chunk-v1"
SOLVER_MATH_ABI = {
    "algorithm": "independent-metric-residual-omp-post-replant-ridge-v1",
    "support_selection": "normalized-absolute-metric-residual-correlation",
    "coefficient_solve": "cpu-float64-ridge-solve-ex-with-lstsq-fallback",
    "published_index_dtype": "int64",
    "published_coefficient_dtype": "float32",
    "independent_channels": True,
    "all_required_reverse_incidence_rows_reclosed": True,
}
SOLVER_MATH_ABI_SHA256 = canonical_sha256(SOLVER_MATH_ABI)
CANONICAL_CHECKPOINT_ROWS_PER_CHUNK = 256
CHECKPOINT_PARTITION_ABI = {
    "schema": "neural-forge-lfm2-carrier-checkpoint-partition-v1",
    "algorithm": "sorted-required-donor-ordinal-contiguous-chunks",
    "canonical_rows_per_chunk": CANONICAL_CHECKPOINT_ROWS_PER_CHUNK,
    "requested_rows_per_chunk_affects_math_or_cache_identity": False,
    "checkpoint_completion_hashes_affect_semantic_artifact_identity": False,
}
CHECKPOINT_PARTITION_ABI_SHA256 = canonical_sha256(CHECKPOINT_PARTITION_ABI)
_SHA256_LENGTH = 64
CARRIER_MATERIALIZATION_TENSOR_HASH_DOMAIN = (
    b"neural-forge-lfm2-semantic-carrier-tensor-v1\0"
)

BASE_TENSOR_KEYS = (
    "target_table",
    "target_to_donor",
    "input_offsets",
    "input_target_ids",
    "input_coefficients",
    "output_offsets",
    "output_target_ids",
    "output_coefficients",
)
PUBLISHED_TENSOR_KEYS = BASE_TENSOR_KEYS
EVIDENCE_TENSOR_KEYS = (
    "donor_occurrence_counts",
    "donor_winner_counts",
    "donor_reconstruction_errors",
    "donor_eviction_costs",
)


class CarrierReclosureRunnerError(RuntimeError):
    """The authenticated offline carrier compile cannot safely continue."""


class StructuredLogger:
    """Flush one compact JSON record immediately for every observable event."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = sys.stdout if stream is None else stream

    def __call__(self, stage: str, **details: Any) -> None:
        record = {
            "schema": STRUCTURED_LOG_SCHEMA,
            "stage": stage,
            **details,
        }
        print(
            json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=self.stream,
            flush=True,
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise CarrierReclosureRunnerError(
            f"{label} must be a lowercase hexadecimal SHA-256 digest."
        )
    return str(value)


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CarrierReclosureRunnerError(
                    f"{label} contains duplicate JSON key {key!r}."
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except CarrierReclosureRunnerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CarrierReclosureRunnerError(
            f"{label} is not strict UTF-8 JSON."
        ) from error


def _ordinary_file(path: str | Path, label: str) -> Path:
    value = Path(path)
    try:
        resolved = value.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise CarrierReclosureRunnerError(f"{label} does not exist.") from error
    if value.is_symlink() or not resolved.is_file():
        raise CarrierReclosureRunnerError(f"{label} must be an ordinary file.")
    return resolved


def _ordinary_directory(path: str | Path, label: str) -> Path:
    value = Path(path)
    try:
        resolved = value.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise CarrierReclosureRunnerError(f"{label} does not exist.") from error
    if value.is_symlink() or not resolved.is_dir():
        raise CarrierReclosureRunnerError(f"{label} must be an ordinary directory.")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _stable_authenticated_file(
    specification: Mapping[str, Any],
    label: str,
) -> tuple[Path, str, tuple[int, int, int, int]]:
    if not isinstance(specification, Mapping):
        raise CarrierReclosureRunnerError(f"{label} specification must be an object.")
    path = _ordinary_file(specification.get("path", ""), label)
    expected = _sha256(specification.get("sha256"), f"{label} SHA-256")
    before = path.stat()
    actual = _file_sha256(path)
    after = path.stat()
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    )
    if before_identity != after_identity:
        raise CarrierReclosureRunnerError(f"{label} changed while being hashed.")
    if actual != expected:
        raise CarrierReclosureRunnerError(f"{label} SHA-256 mismatch.")
    return path, actual, after_identity


def _load_authenticated_json(
    specification: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, Any], str]:
    path, digest, identity = _stable_authenticated_file(specification, label)
    payload = path.read_bytes()
    after = path.stat()
    if identity != (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    ):
        raise CarrierReclosureRunnerError(f"{label} changed while being read.")
    value = _strict_json_bytes(payload, label)
    if not isinstance(value, dict):
        raise CarrierReclosureRunnerError(f"{label} root must be an object.")
    return value, digest


def _load_authenticated_tensors(
    specification: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, torch.Tensor], str]:
    path, digest, identity = _stable_authenticated_file(specification, label)
    try:
        tensors = load_file(str(path), device="cpu")
    except Exception as error:
        raise CarrierReclosureRunnerError(
            f"{label} is not a readable local safetensors artifact."
        ) from error
    after = path.stat()
    if identity != (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    ):
        raise CarrierReclosureRunnerError(f"{label} changed while being loaded.")
    return tensors, digest


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _materialization_tensor_sha256(value: torch.Tensor) -> str:
    """Hash a tensor under the carrier materialization receipt's v1 domain."""

    tensor = value.detach()
    digest = hashlib.sha256()
    digest.update(CARRIER_MATERIALIZATION_TENSOR_HASH_DOMAIN)
    digest.update(str(tensor.dtype).removeprefix("torch.").encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        tensor.to(device="cpu").contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    )
    return digest.hexdigest()


def _sealed_json(value: Mapping[str, Any], schema: str, label: str) -> dict[str, Any]:
    result = dict(value)
    supplied = result.pop("artifact_sha256", None)
    if result.get("schema") != schema or supplied != canonical_sha256(result):
        raise CarrierReclosureRunnerError(f"{label} self-seal does not verify.")
    return {**result, "artifact_sha256": supplied}


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CarrierReclosureRunnerError(f"{label} must be a positive integer.")
    return int(value)


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CarrierReclosureRunnerError(
            f"{label} must be a non-negative integer."
        )
    return int(value)


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CarrierReclosureRunnerError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise CarrierReclosureRunnerError(
            f"{label} must be finite and non-negative."
        )
    return result


def _solver_math_contract(solver: Mapping[str, Any]) -> dict[str, Any]:
    support_size = _positive_int(solver.get("support_size"), "support_size")
    ridge = _finite_nonnegative(solver.get("ridge"), "ridge")
    selection_device = solver.get("selection_device", "cpu")
    if not isinstance(selection_device, str) or not selection_device:
        raise CarrierReclosureRunnerError("selection_device must be a device string.")
    requested_selection_dtype = solver.get("selection_dtype", "float32")
    if requested_selection_dtype not in {"float16", "bfloat16", "float32", "float64"}:
        raise CarrierReclosureRunnerError("Unsupported selection_dtype.")
    effective_selection_dtype = (
        "float32"
        if requested_selection_dtype in {"float16", "bfloat16"}
        else requested_selection_dtype
    )
    if solver.get("solve_device", "cpu") != "cpu":
        raise CarrierReclosureRunnerError(
            "The stable reclosure coefficient solve must use CPU."
        )
    if solver.get("solve_dtype", "float64") != "float64":
        raise CarrierReclosureRunnerError(
            "The stable reclosure coefficient solve must use float64."
        )
    return {
        "solver_math_abi_sha256": SOLVER_MATH_ABI_SHA256,
        "support_size": support_size,
        "ridge": ridge,
        "requested_selection_dtype": requested_selection_dtype,
        "effective_selection_dtype": effective_selection_dtype,
        "selection_device": selection_device,
        "solve_dtype": "float64",
        "solve_device": "cpu",
        "publication_index_dtype": "int64",
        "publication_coefficient_dtype": "float32",
        "target_table_dtype_policy": "preserve-exactly",
        "low_precision_selection_promoted_to_float32": True,
        "channels_solved_independently": True,
    }


def _solver_execution_contract(solver: Mapping[str, Any]) -> dict[str, Any]:
    """Return orchestration controls that cannot invalidate folding math."""

    requested_rows = _positive_int(
        solver.get("rows_per_chunk"), "rows_per_chunk"
    )
    return {
        "checkpoint_partition_abi_sha256": CHECKPOINT_PARTITION_ABI_SHA256,
        "canonical_checkpoint_rows_per_chunk": (
            CANONICAL_CHECKPOINT_ROWS_PER_CHUNK
        ),
        "requested_orchestration_rows_per_chunk": requested_rows,
        "requested_rows_per_chunk_affects_math_or_cache_identity": False,
        "checkpoint_completion_hashes_affect_semantic_artifact_identity": False,
    }


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
    }[name]


def _validate_metric(
    metric: torch.Tensor,
    *,
    width: int,
    label: str,
) -> torch.Tensor:
    value = metric.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if tuple(value.shape) != (width, width) or not bool(torch.isfinite(value).all()):
        raise CarrierReclosureRunnerError(
            f"{label} metric must be a finite square matrix matching table width."
        )
    asymmetry = float((value - value.transpose(0, 1)).abs().max().item())
    scale = max(1.0, float(value.abs().max().item()))
    if asymmetry > 1e-8 * scale:
        raise CarrierReclosureRunnerError(f"{label} metric is not symmetric.")
    value = (value + value.transpose(0, 1)) * 0.5
    minimum_eigenvalue = float(torch.linalg.eigvalsh(value).min().item())
    if minimum_eigenvalue < -1e-8 * scale:
        raise CarrierReclosureRunnerError(
            f"{label} metric is not positive semidefinite."
        )
    return value


def _load_base_bridge(
    request: Mapping[str, Any], logger: StructuredLogger
) -> tuple[dict[str, torch.Tensor], str]:
    tensors, file_sha = _load_authenticated_tensors(
        request["base_bridge"], "base bridge"
    )
    if set(tensors) != set(BASE_TENSOR_KEYS):
        raise CarrierReclosureRunnerError(
            "Base bridge must contain exactly the complete dual-route tensor set."
        )
    for name, tensor in tensors.items():
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            tensors[name] = tensor.to(device="cpu").contiguous()
    logger(
        "artifact-authenticated",
        artifact="base-bridge",
        file_sha256=file_sha,
        target_table_dtype=str(tensors["target_table"].dtype),
        input_coefficient_dtype=str(tensors["input_coefficients"].dtype),
        output_coefficient_dtype=str(tensors["output_coefficients"].dtype),
    )
    return tensors, file_sha


def _load_parent_table(
    request: Mapping[str, Any], logger: StructuredLogger
) -> tuple[torch.Tensor, str]:
    tensors, file_sha = _load_authenticated_tensors(
        request["parent_table"], "parent table"
    )
    if set(tensors) != {"parent_table"}:
        raise CarrierReclosureRunnerError(
            "Parent-table artifact must contain exactly parent_table."
        )
    table = tensors["parent_table"].to(device="cpu").contiguous()
    if table.ndim != 2 or not table.is_floating_point() or not bool(
        torch.isfinite(table).all()
    ):
        raise CarrierReclosureRunnerError("Parent table must be finite floating point.")
    logger(
        "artifact-authenticated",
        artifact="parent-table",
        file_sha256=file_sha,
        dtype=str(table.dtype),
        rows=int(table.shape[0]),
        width=int(table.shape[1]),
    )
    return table, file_sha


def _load_carrier_evidence(
    request: Mapping[str, Any],
    *,
    numeric_contract_sha256: str,
    logger: StructuredLogger,
) -> tuple[Lfm2SemanticCarrierDemandEvidence, str]:
    specification = request["carrier_evidence"]
    tensors, tensor_file_sha = _load_authenticated_tensors(
        specification["tensors"], "carrier evidence tensors"
    )
    if set(tensors) != set(EVIDENCE_TENSOR_KEYS):
        raise CarrierReclosureRunnerError(
            "Carrier evidence tensor artifact has an incomplete tensor set."
        )
    raw_receipt, receipt_file_sha = _load_authenticated_json(
        specification["receipt"], "carrier evidence receipt"
    )
    receipt = _sealed_json(
        raw_receipt,
        CARRIER_EVIDENCE_INPUT_SCHEMA,
        "carrier evidence receipt",
    )
    if (
        receipt.get("tensor_file_sha256") != tensor_file_sha
        or receipt.get("numeric_contract_sha256") != numeric_contract_sha256
        or receipt.get("partition_role") != "fold-fit"
        or receipt.get("selection_allowed") is not True
        or receipt.get("review_or_evaluation_evidence_allowed") is not False
        or receipt.get("parent_forward_calls") != 0
        or receipt.get("parent_trace_recaptures") != 0
    ):
        raise CarrierReclosureRunnerError(
            "Carrier evidence is not sealed fold-fit-only offline evidence."
        )
    for name in EVIDENCE_TENSOR_KEYS:
        if receipt.get(f"{name}_sha256") != _tensor_sha256(tensors[name]):
            raise CarrierReclosureRunnerError(
                f"Carrier evidence tensor binding failed for {name}."
            )
    evidence = build_lfm2_semantic_carrier_demand_evidence(
        tensors["donor_occurrence_counts"],
        tensors["donor_winner_counts"],
        tensors["donor_reconstruction_errors"],
        tensors["donor_eviction_costs"],
        source_artifact_sha256=receipt["source_artifact_sha256"],
        source_corpus_sha256=receipt["source_corpus_sha256"],
        parent_identity_sha256=receipt["parent_identity_sha256"],
        fit_partition_manifest_sha256=receipt["fit_partition_manifest_sha256"],
        producer_implementation_sha256=receipt["producer_implementation_sha256"],
        numeric_contract_sha256=numeric_contract_sha256,
        base_semantic_binding_sha256=receipt["base_semantic_binding_sha256"],
        partition_role="fold-fit",
        selection_allowed=True,
    )
    expected_artifact = receipt.get("compiled_evidence_artifact_sha256")
    if expected_artifact is not None and evidence.artifact_sha256 != expected_artifact:
        raise CarrierReclosureRunnerError(
            "Compiled carrier evidence differs from its authenticated receipt."
        )
    logger(
        "evidence-authenticated",
        evidence="carrier-demand",
        tensor_file_sha256=tensor_file_sha,
        receipt_file_sha256=receipt_file_sha,
        artifact_sha256=evidence.artifact_sha256,
        partition_role="fold-fit",
        parent_forward_calls=0,
        parent_trace_recaptures=0,
    )
    return evidence, receipt["artifact_sha256"]


def _load_solver_metric(
    specification: Mapping[str, Any],
    *,
    channel: str,
    width: int,
    numeric_contract_sha256: str,
    logger: StructuredLogger,
) -> tuple[torch.Tensor, dict[str, Any]]:
    tensors, tensor_file_sha = _load_authenticated_tensors(
        specification["tensors"], f"{channel} solver evidence tensors"
    )
    if set(tensors) != {"metric"}:
        raise CarrierReclosureRunnerError(
            f"{channel} solver evidence must contain exactly metric."
        )
    raw_receipt, receipt_file_sha = _load_authenticated_json(
        specification["receipt"], f"{channel} solver evidence receipt"
    )
    receipt = _sealed_json(
        raw_receipt, SOLVER_EVIDENCE_SCHEMA, f"{channel} solver evidence receipt"
    )
    if (
        receipt.get("channel") != channel
        or receipt.get("tensor_file_sha256") != tensor_file_sha
        or receipt.get("metric_sha256") != _tensor_sha256(tensors["metric"])
        or receipt.get("numeric_contract_sha256") != numeric_contract_sha256
        or receipt.get("partition_role") != "fold-fit"
        or receipt.get("cached_trace_evidence") is not True
        or receipt.get("review_or_evaluation_evidence_allowed") is not False
        or receipt.get("parent_forward_calls") != 0
        or receipt.get("parent_trace_recaptures") != 0
    ):
        raise CarrierReclosureRunnerError(
            f"{channel} solver evidence is not authenticated cached fold-fit trace evidence."
        )
    metric = _validate_metric(tensors["metric"], width=width, label=channel)
    logger(
        "evidence-authenticated",
        evidence=f"{channel}-solver-metric",
        tensor_file_sha256=tensor_file_sha,
        receipt_file_sha256=receipt_file_sha,
        artifact_sha256=receipt["artifact_sha256"],
        metric_dtype=str(tensors["metric"].dtype),
        effective_solve_metric_dtype="torch.float64",
        cached_trace_evidence=True,
        parent_forward_calls=0,
        parent_trace_recaptures=0,
    )
    return metric, receipt


def _solve_chunk(
    *,
    channel: str,
    donor_ids: torch.Tensor,
    source_table: torch.Tensor,
    post_replant_target_table: torch.Tensor,
    metric: torch.Tensor,
    numeric_contract: Mapping[str, Any],
    progress: Callable[[Mapping[str, Any]], None],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Independently choose supports and solve one complete donor chunk."""

    support_size = int(numeric_contract["support_size"])
    target_count = int(post_replant_target_table.shape[0])
    if support_size > target_count:
        raise CarrierReclosureRunnerError(
            "support_size cannot exceed post-replant target row count."
        )
    device = torch.device(str(numeric_contract["selection_device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CarrierReclosureRunnerError(
            "CUDA selection was requested but CUDA is unavailable."
        )
    selection_dtype = _torch_dtype(str(numeric_contract["effective_selection_dtype"]))
    target_selection = post_replant_target_table.to(
        device=device, dtype=selection_dtype
    )
    source_selection = source_table.index_select(0, donor_ids).to(
        device=device, dtype=selection_dtype
    )
    metric_selection = metric.to(device=device, dtype=selection_dtype)
    target_metric = target_selection @ metric_selection
    target_norm = (target_metric * target_selection).sum(dim=1).clamp_min(0.0).sqrt()
    usable = target_norm > torch.finfo(selection_dtype).eps
    if int(usable.sum().item()) < support_size:
        raise CarrierReclosureRunnerError(
            f"{channel} target dictionary has too few nonzero metric-norm atoms."
        )
    row_count = int(donor_ids.numel())
    supports = torch.empty((row_count, support_size), dtype=torch.int64, device=device)
    selected_mask = torch.zeros((row_count, target_count), dtype=torch.bool, device=device)
    residual = source_selection.clone()
    ridge = float(numeric_contract["ridge"])
    maximum_selection_residual = 0.0
    for step in range(support_size):
        correlations = (residual @ metric_selection) @ target_selection.transpose(0, 1)
        correlations = correlations.abs() / target_norm.clamp_min(
            torch.finfo(selection_dtype).eps
        ).unsqueeze(0)
        correlations.masked_fill_(~usable.unsqueeze(0), -torch.inf)
        correlations.masked_fill_(selected_mask, -torch.inf)
        chosen = correlations.argmax(dim=1)
        if bool(selected_mask.gather(1, chosen.unsqueeze(1)).any()):
            raise CarrierReclosureRunnerError(
                f"{channel} residual OMP selected a duplicate atom."
            )
        supports[:, step] = chosen
        selected_mask.scatter_(1, chosen.unsqueeze(1), True)
        active = supports[:, : step + 1]
        basis = target_selection[active]
        basis_metric = basis @ metric_selection
        gram = torch.bmm(basis_metric, basis.transpose(1, 2))
        gram.diagonal(dim1=-2, dim2=-1).add_(ridge)
        rhs = torch.bmm(
            basis_metric, source_selection.unsqueeze(2)
        )
        solved = torch.linalg.lstsq(gram, rhs).solution
        if not bool(torch.isfinite(solved).all()):
            raise CarrierReclosureRunnerError(
                f"{channel} selection solve produced non-finite coefficients."
            )
        residual = source_selection - torch.bmm(
            solved.transpose(1, 2), basis
        ).squeeze(1)
        maximum_selection_residual = max(
            maximum_selection_residual,
            float(torch.linalg.vector_norm(residual, dim=1).max().item()),
        )
        progress(
            {
                "stage": "omp-support-step-complete",
                "channel": channel,
                "support_depth": step + 1,
                "support_size": support_size,
                "row_count": row_count,
                "maximum_selection_residual": maximum_selection_residual,
            }
        )

    support_cpu = supports.to(device="cpu", dtype=torch.int64).contiguous()
    target_solve = post_replant_target_table.to(device="cpu", dtype=torch.float64)
    source_solve = source_table.index_select(0, donor_ids).to(
        device="cpu", dtype=torch.float64
    )
    metric_solve = metric.to(device="cpu", dtype=torch.float64)
    basis = target_solve[support_cpu]
    basis_metric = basis @ metric_solve
    gram = torch.bmm(basis_metric, basis.transpose(1, 2))
    gram.diagonal(dim1=-2, dim2=-1).add_(ridge)
    rhs = torch.bmm(basis_metric, source_solve.unsqueeze(2))
    solved, info = torch.linalg.solve_ex(gram, rhs, check_errors=False)
    failed = info.ne(0) | ~torch.isfinite(solved).all(dim=(1, 2))
    fallback_count = int(failed.sum().item())
    if fallback_count:
        solved[failed] = torch.linalg.lstsq(gram[failed], rhs[failed]).solution
    if not bool(torch.isfinite(solved).all()):
        raise CarrierReclosureRunnerError(
            f"{channel} float64 coefficient solve remained non-finite."
        )
    solve_residual = torch.bmm(gram, solved) - rhs
    coefficient = solved.squeeze(2).to(dtype=torch.float32).contiguous()
    support_cpu, support_order = torch.sort(support_cpu, dim=1)
    coefficient = coefficient.gather(1, support_order).contiguous()
    offsets = torch.arange(
        0,
        (row_count + 1) * support_size,
        support_size,
        dtype=torch.int64,
    )
    tensors = {
        "donor_ids": donor_ids.to(device="cpu", dtype=torch.int64).contiguous(),
        "offsets": offsets,
        "target_ids": support_cpu.reshape(-1).contiguous(),
        "coefficients": coefficient.reshape(-1).contiguous(),
    }
    report = {
        "schema": SOLVER_CHUNK_SCHEMA,
        "channel": channel,
        "row_count": row_count,
        "support_size": support_size,
        "ridge": ridge,
        "selection_device": str(device),
        "requested_selection_dtype": numeric_contract["requested_selection_dtype"],
        "effective_selection_dtype": numeric_contract["effective_selection_dtype"],
        "solve_device": "cpu",
        "solve_dtype": "float64",
        "published_coefficient_dtype": "float32",
        "donor_ids_sha256": _tensor_sha256(tensors["donor_ids"]),
        "offsets_sha256": _tensor_sha256(tensors["offsets"]),
        "target_ids_sha256": _tensor_sha256(tensors["target_ids"]),
        "coefficients_sha256": _tensor_sha256(tensors["coefficients"]),
        "solve_fallback_count": fallback_count,
        "maximum_selection_residual": maximum_selection_residual,
        "maximum_normal_equation_residual": float(
            torch.linalg.vector_norm(solve_residual, dim=(1, 2)).max().item()
        ),
        "coefficient_abs_maximum": float(coefficient.abs().max().item()),
        "routes_complete_for_chunk": True,
        "solved_against_post_replant_target_table": True,
        "solver_math_abi_sha256": SOLVER_MATH_ABI_SHA256,
    }
    return tensors, report


def _chunk_phase_key(
    *,
    channel: str,
    donor_ids: torch.Tensor,
    source_table_sha256: str,
    post_table_sha256: str,
    metric_receipt_sha256: str,
    numeric_contract: Mapping[str, Any],
) -> FoldPhaseKey:
    return FoldPhaseKey(
        phase_name=f"lfm2-carrier-reclosure-{channel}-chunk",
        dependency_sha256s={
            "donor_ids": _tensor_sha256(donor_ids),
            "metric_evidence": metric_receipt_sha256,
            "post_replant_target_table": post_table_sha256,
            "source_table": source_table_sha256,
        },
        numeric_contract={**dict(numeric_contract), "channel": channel},
        abi={
            "solver_math": SOLVER_MATH_ABI,
            "checkpoint_partition": CHECKPOINT_PARTITION_ABI,
            "channel": channel,
        },
    )


def _load_verified_chunk(
    entry: VerifiedFoldPhaseCacheEntry,
    *,
    channel: str,
    expected_donor_ids: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = entry.payload_path / "routes.safetensors"
    if not path.is_file() or path.is_symlink():
        raise CarrierReclosureRunnerError("Cached solver chunk lacks routes.safetensors.")
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {"donor_ids", "offsets", "target_ids", "coefficients"}:
        raise CarrierReclosureRunnerError("Cached solver chunk tensor set is invalid.")
    receipt = dict(entry.result)
    if receipt.get("schema") != SOLVER_CHUNK_SCHEMA or receipt.get("channel") != channel:
        raise CarrierReclosureRunnerError("Cached solver chunk receipt is invalid.")
    if not torch.equal(tensors["donor_ids"], expected_donor_ids):
        raise CarrierReclosureRunnerError("Cached solver chunk donor IDs are stale.")
    for name, tensor in tensors.items():
        if receipt.get(f"{name}_sha256") != _tensor_sha256(tensor):
            raise CarrierReclosureRunnerError(
                f"Cached solver chunk tensor binding failed for {name}."
            )
    if (
        tensors["donor_ids"].dtype is not torch.int64
        or tensors["offsets"].dtype is not torch.int64
        or tensors["target_ids"].dtype is not torch.int64
        or tensors["coefficients"].dtype is not torch.float32
        or receipt.get("routes_complete_for_chunk") is not True
        or receipt.get("solved_against_post_replant_target_table") is not True
    ):
        raise CarrierReclosureRunnerError("Cached solver chunk ABI is invalid.")
    return tensors, receipt


def _cache_or_solve_chunk(
    *,
    cache: DynamicFoldPhaseCache,
    events: FoldPhaseEventLog,
    logger: StructuredLogger,
    phase_key: FoldPhaseKey,
    channel: str,
    donor_ids: torch.Tensor,
    source_table: torch.Tensor,
    post_replant_target_table: torch.Tensor,
    metric: torch.Tensor,
    numeric_contract: Mapping[str, Any],
    chunk_index: int,
    chunk_count: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str, bool]:
    logger(
        "phase-cache-lookup",
        phase=phase_key.phase_name,
        phase_key_sha256=phase_key.phase_key_sha256,
        channel=channel,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        row_count=int(donor_ids.numel()),
    )
    hit = cache.lookup(phase_key, events=events)
    if hit is not None:
        tensors, receipt = _load_verified_chunk(
            hit, channel=channel, expected_donor_ids=donor_ids
        )
        logger(
            "phase-cache-hit",
            phase=phase_key.phase_name,
            phase_key_sha256=phase_key.phase_key_sha256,
            completion_sha256=hit.completion_sha256,
            channel=channel,
            chunk_index=chunk_index,
        )
        return tensors, receipt, hit.completion_sha256, True
    logger(
        "phase-cache-miss",
        phase=phase_key.phase_name,
        phase_key_sha256=phase_key.phase_key_sha256,
        channel=channel,
        chunk_index=chunk_index,
    )
    started = time.monotonic()
    with cache.begin(phase_key, events=events) as transaction:
        logger(
            "solver-chunk-start",
            channel=channel,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            row_count=int(donor_ids.numel()),
            requested_selection_dtype=numeric_contract["requested_selection_dtype"],
            effective_selection_dtype=numeric_contract["effective_selection_dtype"],
            solve_dtype="float64",
            target_table_storage_dtype=str(post_replant_target_table.dtype),
        )

        def progress(details: Mapping[str, Any]) -> None:
            transaction.progress(dict(details))
            console_details = dict(details)
            solver_stage = console_details.pop("stage", None)
            logger(
                "solver-chunk-progress",
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                solver_stage=solver_stage,
                **console_details,
            )

        tensors, receipt = _solve_chunk(
            channel=channel,
            donor_ids=donor_ids,
            source_table=source_table,
            post_replant_target_table=post_replant_target_table,
            metric=metric,
            numeric_contract=numeric_contract,
            progress=progress,
        )
        save_file(tensors, str(transaction.payload_path / "routes.safetensors"))
        entry = transaction.complete(result=receipt)
    tensors, receipt = _load_verified_chunk(
        entry, channel=channel, expected_donor_ids=donor_ids
    )
    logger(
        "solver-chunk-complete",
        channel=channel,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        row_count=int(donor_ids.numel()),
        elapsed_seconds=time.monotonic() - started,
        phase_key_sha256=phase_key.phase_key_sha256,
        completion_sha256=entry.completion_sha256,
        cache_hit=False,
        solve_fallback_count=receipt["solve_fallback_count"],
        coefficient_abs_maximum=receipt["coefficient_abs_maximum"],
    )
    return tensors, receipt, entry.completion_sha256, False


def _compile_channel_routes(
    *,
    channel: str,
    required_donor_ids: torch.Tensor,
    source_table: torch.Tensor,
    post_replant_target_table: torch.Tensor,
    metric: torch.Tensor,
    metric_receipt: Mapping[str, Any],
    numeric_contract: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
    cache: DynamicFoldPhaseCache,
    events: FoldPhaseEventLog,
    logger: StructuredLogger,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, Any]]:
    rows_per_chunk = CANONICAL_CHECKPOINT_ROWS_PER_CHUNK
    row_count = int(required_donor_ids.numel())
    chunk_count = (row_count + rows_per_chunk - 1) // rows_per_chunk
    source_table_sha = _tensor_sha256(source_table)
    post_table_sha = _tensor_sha256(post_replant_target_table)
    chunks: list[dict[str, torch.Tensor]] = []
    completions: list[str] = []
    cache_hits = 0
    for chunk_index, start in enumerate(range(0, row_count, rows_per_chunk)):
        stop = min(start + rows_per_chunk, row_count)
        donor_chunk = required_donor_ids[start:stop].contiguous()
        key = _chunk_phase_key(
            channel=channel,
            donor_ids=donor_chunk,
            source_table_sha256=source_table_sha,
            post_table_sha256=post_table_sha,
            metric_receipt_sha256=metric_receipt["artifact_sha256"],
            numeric_contract=numeric_contract,
        )
        tensors, _, completion, hit = _cache_or_solve_chunk(
            cache=cache,
            events=events,
            logger=logger,
            phase_key=key,
            channel=channel,
            donor_ids=donor_chunk,
            source_table=source_table,
            post_replant_target_table=post_replant_target_table,
            metric=metric,
            numeric_contract=numeric_contract,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
        )
        chunks.append(tensors)
        completions.append(completion)
        cache_hits += int(hit)
    offsets = [0]
    target_ids: list[torch.Tensor] = []
    coefficients: list[torch.Tensor] = []
    donor_sequence: list[torch.Tensor] = []
    for chunk in chunks:
        donor_sequence.append(chunk["donor_ids"])
        target_ids.append(chunk["target_ids"])
        coefficients.append(chunk["coefficients"])
        lengths = chunk["offsets"][1:] - chunk["offsets"][:-1]
        for length in lengths.tolist():
            offsets.append(offsets[-1] + int(length))
    if not torch.equal(torch.cat(donor_sequence), required_donor_ids):
        raise CarrierReclosureRunnerError(
            f"{channel} solver chunks do not exactly cover required donor IDs."
        )
    route = (
        torch.tensor(offsets, dtype=torch.int64),
        torch.cat(target_ids).to(dtype=torch.int64).contiguous(),
        torch.cat(coefficients).to(dtype=torch.float32).contiguous(),
    )
    aggregate = {
        "channel": channel,
        "metric_evidence_artifact_sha256": metric_receipt["artifact_sha256"],
        "source_table_sha256": source_table_sha,
        "post_replant_target_table_sha256": post_table_sha,
        "numeric_contract_sha256": canonical_sha256(numeric_contract),
        "required_donor_ids_sha256": _tensor_sha256(required_donor_ids),
        "offsets_sha256": _tensor_sha256(route[0]),
        "target_ids_sha256": _tensor_sha256(route[1]),
        "coefficients_sha256": _tensor_sha256(route[2]),
        "all_required_rows_present_once": True,
        "solved_independently": True,
        "solver_math_abi_sha256": SOLVER_MATH_ABI_SHA256,
    }
    aggregate["aggregate_sha256"] = canonical_sha256(aggregate)
    logger(
        "channel-reclosure-complete",
        **aggregate,
        checkpoint_partition_abi_sha256=CHECKPOINT_PARTITION_ABI_SHA256,
        canonical_checkpoint_rows_per_chunk=rows_per_chunk,
        requested_orchestration_rows_per_chunk=execution_contract[
            "requested_orchestration_rows_per_chunk"
        ],
        chunk_completion_sha256s=completions,
        chunk_count=chunk_count,
        chunk_cache_hit_count=cache_hits,
        chunk_cache_miss_count=chunk_count - cache_hits,
        coefficient_count=int(route[2].numel()),
    )
    return route, aggregate


def _publish_output_directory(
    source: Path,
    destination: Path,
    *,
    expected_artifact_sha256: str,
) -> bool:
    destination = destination.resolve()
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise CarrierReclosureRunnerError(
                "Output destination exists but is not an ordinary directory."
            )
        receipt_path = destination / "bridge-receipt.json"
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise CarrierReclosureRunnerError(
                "Existing output directory is not a complete bridge artifact."
            )
        receipt = _strict_json_bytes(receipt_path.read_bytes(), "existing bridge receipt")
        tensor_path = destination / "bridge.safetensors"
        if (
            not isinstance(receipt, dict)
            or _sealed_json(
                receipt, STANDALONE_BRIDGE_SCHEMA, "existing bridge receipt"
            )["artifact_sha256"]
            != expected_artifact_sha256
            or not tensor_path.is_file()
            or tensor_path.is_symlink()
            or _file_sha256(tensor_path)
            != receipt.get("bridge_tensor_file_sha256")
        ):
            raise CarrierReclosureRunnerError(
                "Existing output directory contains a different bridge artifact."
            )
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        os.rename(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return False


def _cache_or_publish_bridge(
    *,
    cache: DynamicFoldPhaseCache,
    events: FoldPhaseEventLog,
    logger: StructuredLogger,
    materialization: Any,
    allocation: Any,
    selection: Any,
    tokenizer_contract: Any,
    base_bridge_file_sha256: str,
    parent_table_file_sha256: str,
    input_aggregate: Mapping[str, Any],
    output_aggregate: Mapping[str, Any],
    numeric_contract: Mapping[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    phase_key = FoldPhaseKey(
        phase_name="lfm2-carrier-reclosed-standalone-bridge",
        dependency_sha256s={
            "base_bridge_file": base_bridge_file_sha256,
            "carrier_allocation": allocation.artifact_sha256,
            "carrier_materialization": materialization.artifact_sha256,
            "input_reclosure": input_aggregate["aggregate_sha256"],
            "output_reclosure": output_aggregate["aggregate_sha256"],
            "parent_table_file": parent_table_file_sha256,
            "tokenizer_control_contract": tokenizer_contract.receipt_sha256,
        },
        numeric_contract=dict(numeric_contract),
        abi={
            "schema": STANDALONE_BRIDGE_SCHEMA,
            "tensor_keys": list(PUBLISHED_TENSOR_KEYS),
            "requires_parent_runtime": False,
            "requires_sidecar": False,
            "target_table_counted_once": True,
        },
    )
    logger(
        "phase-cache-lookup",
        phase=phase_key.phase_name,
        phase_key_sha256=phase_key.phase_key_sha256,
    )
    entry = cache.lookup(phase_key, events=events)
    cache_hit = entry is not None
    if entry is None:
        logger(
            "phase-cache-miss",
            phase=phase_key.phase_name,
            phase_key_sha256=phase_key.phase_key_sha256,
        )
        with cache.begin(phase_key, events=events) as transaction:
            tensors = {
                "target_table": materialization.target_table,
                "target_to_donor": materialization.target_to_donor,
                "input_offsets": materialization.input_offsets,
                "input_target_ids": materialization.input_target_ids,
                "input_coefficients": materialization.input_coefficients,
                "output_offsets": materialization.output_offsets,
                "output_target_ids": materialization.output_target_ids,
                "output_coefficients": materialization.output_coefficients,
            }
            tensor_path = transaction.payload_path / "bridge.safetensors"
            save_file(
                {name: tensor.to(device="cpu").contiguous() for name, tensor in tensors.items()},
                str(tensor_path),
            )
            tensor_file_sha = _file_sha256(tensor_path)
            operation_audit = {
                "parent_forward_calls": 0,
                "parent_trace_recaptures": 0,
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "parameter_pruning_calls": 0,
                "parameter_quantization_calls": 0,
                "parameter_sparsification_calls": 0,
                "parameter_sharing_calls": 0,
                "runtime_sidecar_parameters": 0,
            }
            body = {
                "schema": STANDALONE_BRIDGE_SCHEMA,
                "status": "complete",
                "bridge_tensor_file": "bridge.safetensors",
                "bridge_tensor_file_sha256": tensor_file_sha,
                "materialization_artifact_sha256": materialization.artifact_sha256,
                "allocation_artifact_sha256": allocation.artifact_sha256,
                "selection_artifact_sha256": selection.artifact_sha256,
                "tokenizer_control_contract_sha256": tokenizer_contract.receipt_sha256,
                "tokenizer_control_source_inventory_sha256": tokenizer_contract.receipt[
                    "source_inventory_sha256"
                ],
                "protected_donor_ids": list(tokenizer_contract.protected_id_values),
                "reclaimed_reserved_donor_ids": list(
                    tokenizer_contract.reclaimed_id_values
                ),
                "base_bridge_file_sha256": base_bridge_file_sha256,
                "parent_table_file_sha256": parent_table_file_sha256,
                "input_reclosure": dict(input_aggregate),
                "output_reclosure": dict(output_aggregate),
                "numeric_contract": dict(numeric_contract),
                "numeric_contract_sha256": canonical_sha256(numeric_contract),
                "target_table_dtype": str(materialization.target_table.dtype),
                "target_table_dtype_preserved": True,
                "input_and_output_routes_are_independent": True,
                "complete_reverse_incidence_closure_materialized": True,
                "runtime_requires_parent": False,
                "runtime_requires_external_sidecar": False,
                "all_continuous_bridge_state_is_in_bridge_tensor_file": True,
                "integer_csr_routes_are_structural_metadata": True,
                "continuous_semantic_coefficient_count": int(
                    materialization.input_coefficients.numel()
                    + materialization.output_coefficients.numel()
                ),
                "physical_target_table_parameter_count": int(
                    materialization.target_table.numel()
                ),
                "operation_audit": operation_audit,
                "tokenizer_control_receipt": tokenizer_contract.as_dict(),
                "selection_receipt": selection.as_dict(),
                "allocation_receipt": allocation.as_dict(),
                "materialization_receipt": materialization.as_dict(),
            }
            receipt = {**body, "artifact_sha256": canonical_sha256(body)}
            receipt_path = transaction.payload_path / "bridge-receipt.json"
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            entry = transaction.complete(
                result={
                    "artifact_sha256": receipt["artifact_sha256"],
                    "bridge_tensor_file_sha256": tensor_file_sha,
                    "bridge_receipt_file_sha256": _file_sha256(receipt_path),
                }
            )
    assert entry is not None
    if cache_hit:
        logger(
            "phase-cache-hit",
            phase=phase_key.phase_name,
            phase_key_sha256=phase_key.phase_key_sha256,
            completion_sha256=entry.completion_sha256,
        )
    receipt_path = entry.payload_path / "bridge-receipt.json"
    tensor_path = entry.payload_path / "bridge.safetensors"
    receipt = _strict_json_bytes(receipt_path.read_bytes(), "cached bridge receipt")
    if not isinstance(receipt, dict):
        raise CarrierReclosureRunnerError("Cached bridge receipt is invalid.")
    verified = _sealed_json(receipt, STANDALONE_BRIDGE_SCHEMA, "cached bridge receipt")
    if (
        _file_sha256(tensor_path) != verified["bridge_tensor_file_sha256"]
        or entry.result.get("artifact_sha256") != verified["artifact_sha256"]
    ):
        raise CarrierReclosureRunnerError("Cached standalone bridge binding failed.")
    output_reused = _publish_output_directory(
        entry.payload_path,
        output_directory,
        expected_artifact_sha256=verified["artifact_sha256"],
    )
    logger(
        "standalone-bridge-published",
        artifact_sha256=verified["artifact_sha256"],
        phase_key_sha256=phase_key.phase_key_sha256,
        completion_sha256=entry.completion_sha256,
        cache_hit=cache_hit,
        output_reused=output_reused,
        output_directory=str(output_directory.resolve()),
        target_table_dtype=verified["target_table_dtype"],
        continuous_semantic_coefficient_count=verified[
            "continuous_semantic_coefficient_count"
        ],
    )
    return verified


def carrier_bridge_allocations_from_publication(
    directory: str | os.PathLike[str],
    *,
    expected_artifact_sha256: str | None = None,
) -> tuple[ContinuousChildStateAllocation, ContinuousChildStateAllocation]:
    """Fresh-reload a standalone carrier bridge into two counted routes.

    The target table belongs to the dense child and is counted by that model's
    package observer.  This observer independently counts the two continuous
    semantic route tensors stored alongside it; structural CSR indices do not
    receive parameter credit.
    """

    root = _ordinary_directory(directory, "carrier bridge publication")
    receipt_path = _ordinary_file(
        root / "bridge-receipt.json", "carrier bridge receipt"
    )
    before = receipt_path.stat()
    raw_receipt = _strict_json_bytes(
        receipt_path.read_bytes(), "carrier bridge receipt"
    )
    after = receipt_path.stat()
    if (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    ) != (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    ):
        raise CarrierReclosureRunnerError(
            "Carrier bridge receipt changed while being read."
        )
    if not isinstance(raw_receipt, Mapping):
        raise CarrierReclosureRunnerError("Carrier bridge receipt must be an object.")
    receipt = _sealed_json(
        raw_receipt, STANDALONE_BRIDGE_SCHEMA, "carrier bridge receipt"
    )
    artifact_sha256 = _sha256(
        receipt.get("artifact_sha256"), "carrier bridge artifact SHA-256"
    )
    if expected_artifact_sha256 is not None and artifact_sha256 != _sha256(
        expected_artifact_sha256, "expected carrier bridge artifact SHA-256"
    ):
        raise CarrierReclosureRunnerError(
            "Carrier bridge publication belongs to a different artifact."
        )
    if (
        receipt.get("status") != "complete"
        or receipt.get("runtime_requires_parent") is not False
        or receipt.get("runtime_requires_external_sidecar") is not False
        or receipt.get("all_continuous_bridge_state_is_in_bridge_tensor_file")
        is not True
        or receipt.get("integer_csr_routes_are_structural_metadata") is not True
    ):
        raise CarrierReclosureRunnerError(
            "Carrier bridge publication is not a complete standalone child artifact."
        )
    operation_audit = receipt.get("operation_audit")
    if not isinstance(operation_audit, Mapping) or any(
        isinstance(value, bool) or not isinstance(value, int) or value != 0
        for value in operation_audit.values()
    ):
        raise CarrierReclosureRunnerError(
            "Carrier bridge publication does not have a zero-operation fold audit."
        )
    tensor_file = receipt.get("bridge_tensor_file")
    if tensor_file != "bridge.safetensors":
        raise CarrierReclosureRunnerError(
            "Carrier bridge receipt names an unsupported tensor payload."
        )
    tensors, observed_file_sha256 = _load_authenticated_tensors(
        {
            "path": str(root / tensor_file),
            "sha256": _sha256(
                receipt.get("bridge_tensor_file_sha256"),
                "carrier bridge tensor file SHA-256",
            ),
        },
        "carrier bridge tensors",
    )
    if observed_file_sha256 != receipt["bridge_tensor_file_sha256"]:
        raise CarrierReclosureRunnerError(
            "Carrier bridge tensor file binding changed during fresh reload."
        )
    if set(tensors) != set(PUBLISHED_TENSOR_KEYS):
        raise CarrierReclosureRunnerError(
            "Carrier bridge tensor payload has an unexpected key set."
        )
    input_coefficients = tensors["input_coefficients"]
    output_coefficients = tensors["output_coefficients"]
    if (
        input_coefficients.dtype is not torch.float32
        or output_coefficients.dtype is not torch.float32
        or input_coefficients.ndim != 1
        or output_coefficients.ndim != 1
    ):
        raise CarrierReclosureRunnerError(
            "Carrier bridge routes are not independent one-dimensional FP32 tensors."
        )
    materialization = receipt.get("materialization_receipt")
    if not isinstance(materialization, Mapping):
        raise CarrierReclosureRunnerError(
            "Carrier bridge materialization receipt is missing."
        )
    input_sha256 = _materialization_tensor_sha256(input_coefficients)
    output_sha256 = _materialization_tensor_sha256(output_coefficients)
    input_count = int(input_coefficients.numel())
    output_count = int(output_coefficients.numel())
    checks = {
        "input_coefficients_sha256": input_sha256
        == materialization.get("input_coefficients_sha256"),
        "output_coefficients_sha256": output_sha256
        == materialization.get("output_coefficients_sha256"),
        "continuous_semantic_coefficient_count": input_count + output_count
        == receipt.get("continuous_semantic_coefficient_count"),
        "continuous_fold_coefficient_count": input_count + output_count
        == materialization.get("continuous_fold_coefficient_count"),
        "physical_target_table_parameter_count": int(tensors["target_table"].numel())
        == receipt.get("physical_target_table_parameter_count"),
        "target_table_dtype": str(tensors["target_table"].dtype)
        == receipt.get("target_table_dtype"),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise CarrierReclosureRunnerError(
            "Carrier bridge counted-state metadata does not match its loaded tensors: "
            + ", ".join(failed_checks)
            + "."
        )
    return (
        ContinuousChildStateAllocation(
            state_id="semantic-carrier-route.c_in",
            parameter_elements=input_count,
            tensor_sha256=input_sha256,
            artifact_sha256=artifact_sha256,
            dtype="float32",
        ),
        ContinuousChildStateAllocation(
            state_id="semantic-carrier-route.c_out",
            parameter_elements=output_count,
            tensor_sha256=output_sha256,
            artifact_sha256=artifact_sha256,
            dtype="float32",
        ),
    )


def run_carrier_reclosure(
    request: Mapping[str, Any],
    *,
    log_stream: TextIO | None = None,
) -> dict[str, Any]:
    """Run one authenticated, resumable, offline carrier reclosure compile."""

    if not isinstance(request, Mapping) or request.get("schema") != REQUEST_SCHEMA:
        raise CarrierReclosureRunnerError("Unsupported carrier reclosure request.")
    logger = StructuredLogger(log_stream)
    solver = request.get("solver")
    if not isinstance(solver, Mapping):
        raise CarrierReclosureRunnerError("solver must be an object.")
    numeric_contract = _solver_math_contract(solver)
    execution_contract = _solver_execution_contract(solver)
    numeric_sha = canonical_sha256(numeric_contract)
    run_id = request.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CarrierReclosureRunnerError("run_id must be a nonempty identifier.")
    path_values: dict[str, str] = {}
    for field in ("phase_cache_directory", "output_directory", "event_log"):
        raw_path = request.get(field)
        if not isinstance(raw_path, str) or not raw_path:
            raise CarrierReclosureRunnerError(f"{field} must be a nonempty local path.")
        path_values[field] = raw_path
    cache_root = Path(path_values["phase_cache_directory"]).resolve()
    output_directory = Path(path_values["output_directory"])
    event_path = Path(path_values["event_log"]).resolve()
    cache = DynamicFoldPhaseCache(cache_root)
    events = FoldPhaseEventLog(event_path, run_id=run_id)
    logger(
        "run-start",
        run_id=run_id,
        solver_math_abi_sha256=SOLVER_MATH_ABI_SHA256,
        numeric_contract_sha256=numeric_sha,
        dtype_contract=numeric_contract,
        execution_contract=execution_contract,
        parent_forward_calls=0,
        parent_trace_recaptures=0,
        training_steps=0,
    )
    base, base_file_sha = _load_base_bridge(request, logger)
    parent_table, parent_file_sha = _load_parent_table(request, logger)
    target_table = base["target_table"]
    donor_count = int(base["input_offsets"].numel()) - 1
    if (
        target_table.ndim != 2
        or not target_table.is_floating_point()
        or not bool(torch.isfinite(target_table).all())
        or tuple(parent_table.shape[1:]) != tuple(target_table.shape[1:])
        or int(parent_table.shape[0]) != donor_count
    ):
        raise CarrierReclosureRunnerError(
            "Parent and target table widths or dtypes are incompatible."
        )
    tokenizer_spec = request.get("tokenizer_control")
    if not isinstance(tokenizer_spec, Mapping):
        raise CarrierReclosureRunnerError("tokenizer_control must be an object.")
    tokenizer_root = _ordinary_directory(
        tokenizer_spec.get("artifact_directory", ""), "tokenizer artifact directory"
    )
    expected_source_hashes = tokenizer_spec.get("expected_file_sha256")
    if not isinstance(expected_source_hashes, Mapping):
        raise CarrierReclosureRunnerError(
            "tokenizer expected_file_sha256 must be an object."
        )
    manifests = tokenizer_spec.get("manifest_files", [])
    if isinstance(manifests, (str, bytes)) or not isinstance(manifests, Sequence):
        raise CarrierReclosureRunnerError("manifest_files must be a list.")
    tokenizer_contract = compile_local_tokenizer_control_contract(
        tokenizer_root,
        expected_file_sha256=dict(expected_source_hashes),
        manifest_files=list(manifests),
        reclaim_policy_file=tokenizer_spec.get("reclaim_policy_file"),
        expected_reclaim_policy_sha256=tokenizer_spec.get(
            "expected_reclaim_policy_sha256"
        ),
    )
    logger(
        "tokenizer-control-contract-compiled",
        artifact_sha256=tokenizer_contract.receipt_sha256,
        protected_donor_count=len(tokenizer_contract.protected_id_values),
        mandatory_active_donor_count=len(tokenizer_contract.mandatory_active_id_values),
        reserved_placeholder_donor_count=len(
            tokenizer_contract.reserved_placeholder_id_values
        ),
        reclaimed_reserved_donor_count=len(tokenizer_contract.reclaimed_id_values),
        policy_mode=tokenizer_contract.receipt["reclaim_policy"]["mode"],
        hugging_face_loader_used=False,
        network_access_performed=False,
    )
    inventory = build_lfm2_semantic_carrier_inventory(
        input_offsets=base["input_offsets"],
        input_target_ids=base["input_target_ids"],
        input_coefficients=base["input_coefficients"],
        output_offsets=base["output_offsets"],
        output_target_ids=base["output_target_ids"],
        output_coefficients=base["output_coefficients"],
        target_to_donor=base["target_to_donor"],
    )
    evidence, evidence_receipt_sha = _load_carrier_evidence(
        request,
        numeric_contract_sha256=numeric_sha,
        logger=logger,
    )
    selection_config = request.get("selection")
    if not isinstance(selection_config, Mapping):
        raise CarrierReclosureRunnerError("selection must be an object.")
    allow_positive_incidence = selection_config.get(
        "allow_positive_incidence", False
    )
    if not isinstance(allow_positive_incidence, bool):
        raise CarrierReclosureRunnerError(
            "allow_positive_incidence must be a boolean."
        )
    selection = select_lfm2_semantic_carriers(
        inventory,
        evidence,
        promotion_count=_positive_int(
            selection_config.get("promotion_count"), "promotion_count"
        ),
        protected_donor_ids=tokenizer_contract.protected_id_values,
        protection_artifact_sha256=tokenizer_contract.receipt_sha256,
        winner_weight=_nonnegative_int(
            selection_config.get("winner_weight", 8), "winner_weight"
        ),
        allow_positive_incidence=allow_positive_incidence,
    )
    logger(
        "carrier-selection-complete",
        artifact_sha256=selection.artifact_sha256,
        inventory_artifact_sha256=inventory.artifact_sha256,
        evidence_receipt_sha256=evidence_receipt_sha,
        allocation_count=int(selection.carrier_target_ids.numel()),
        required_rerouted_donor_count=int(
            selection.required_rerouted_donor_ids.numel()
        ),
        positive_incidence_target_rows_allowed=selection.receipt[
            "positive_incidence_target_rows_allowed"
        ],
    )
    post_table = target_table.clone()
    replacement_rows = parent_table.index_select(
        0, selection.promoted_donor_ids
    ).to(dtype=target_table.dtype)
    if not bool(torch.isfinite(replacement_rows).all()):
        raise CarrierReclosureRunnerError(
            "Promoted rows are non-finite after target dtype conversion."
        )
    post_table.index_copy_(0, selection.carrier_target_ids, replacement_rows)
    input_metric, input_metric_receipt = _load_solver_metric(
        request["input_solver_evidence"],
        channel="input",
        width=int(target_table.shape[1]),
        numeric_contract_sha256=numeric_sha,
        logger=logger,
    )
    output_metric, output_metric_receipt = _load_solver_metric(
        request["output_solver_evidence"],
        channel="output",
        width=int(target_table.shape[1]),
        numeric_contract_sha256=numeric_sha,
        logger=logger,
    )
    required = selection.required_rerouted_donor_ids
    input_route, input_aggregate = _compile_channel_routes(
        channel="input",
        required_donor_ids=required,
        source_table=parent_table,
        post_replant_target_table=post_table,
        metric=input_metric,
        metric_receipt=input_metric_receipt,
        numeric_contract=numeric_contract,
        execution_contract=execution_contract,
        cache=cache,
        events=events,
        logger=logger,
    )
    output_route, output_aggregate = _compile_channel_routes(
        channel="output",
        required_donor_ids=required,
        source_table=parent_table,
        post_replant_target_table=post_table,
        metric=output_metric,
        metric_receipt=output_metric_receipt,
        numeric_contract=numeric_contract,
        execution_contract=execution_contract,
        cache=cache,
        events=events,
        logger=logger,
    )
    allocation = compile_lfm2_semantic_carrier_allocation(
        selection,
        parent_table,
        target_table,
        input_evicted_offsets=input_route[0],
        input_evicted_target_ids=input_route[1],
        input_evicted_coefficients=input_route[2],
        output_evicted_offsets=output_route[0],
        output_evicted_target_ids=output_route[1],
        output_evicted_coefficients=output_route[2],
        input_replacement_artifact_sha256=input_aggregate["aggregate_sha256"],
        output_replacement_artifact_sha256=output_aggregate["aggregate_sha256"],
        input_solver_math_abi_sha256=SOLVER_MATH_ABI_SHA256,
        output_solver_math_abi_sha256=SOLVER_MATH_ABI_SHA256,
        numeric_contract_sha256=numeric_sha,
    )
    materialization = materialize_lfm2_semantic_carrier_allocation(
        allocation,
        target_table,
        target_to_donor=base["target_to_donor"],
        input_offsets=base["input_offsets"],
        input_target_ids=base["input_target_ids"],
        input_coefficients=base["input_coefficients"],
        output_offsets=base["output_offsets"],
        output_target_ids=base["output_target_ids"],
        output_coefficients=base["output_coefficients"],
    )
    logger(
        "carrier-materialization-complete",
        allocation_artifact_sha256=allocation.artifact_sha256,
        materialization_artifact_sha256=materialization.artifact_sha256,
        target_table_dtype=str(materialization.target_table.dtype),
        required_rerouted_donor_count=int(required.numel()),
        input_coefficient_count=int(materialization.input_coefficients.numel()),
        output_coefficient_count=int(materialization.output_coefficients.numel()),
    )
    result = _cache_or_publish_bridge(
        cache=cache,
        events=events,
        logger=logger,
        materialization=materialization,
        allocation=allocation,
        selection=selection,
        tokenizer_contract=tokenizer_contract,
        base_bridge_file_sha256=base_file_sha,
        parent_table_file_sha256=parent_file_sha,
        input_aggregate=input_aggregate,
        output_aggregate=output_aggregate,
        numeric_contract=numeric_contract,
        output_directory=output_directory,
    )
    logger(
        "run-complete",
        run_id=run_id,
        artifact_sha256=result["artifact_sha256"],
        output_directory=str(output_directory.resolve()),
        parent_forward_calls=0,
        parent_trace_recaptures=0,
        training_steps=0,
    )
    return result


def _load_authenticated_request(path: Path, expected_sha256: str) -> dict[str, Any]:
    specification = {"path": str(path), "sha256": expected_sha256}
    request, _ = _load_authenticated_json(specification, "carrier reclosure request")
    return request


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile a fully reclosed offline LFM2 carrier bridge."
    )
    parser.add_argument("--request-json", required=True, type=Path)
    parser.add_argument("--expected-request-sha256", required=True)
    arguments = parser.parse_args(argv)
    request = _load_authenticated_request(
        arguments.request_json,
        _sha256(arguments.expected_request_sha256, "request SHA-256"),
    )
    run_carrier_reclosure(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
