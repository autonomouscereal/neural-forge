"""Run a cached-trace-only functional ``C_out`` refit for local LFM2 state.

Every model-dependent input is an authenticated local artifact: a sparse
semantic codebook, the parent's tied table, the blank child's tied table, and
an already-sealed circuit trace cache.  The runner never constructs or calls a
model.  It derives all geometry and support widths from CLI expectations plus
the authenticated artifacts, builds a deterministic fit/holdout metric, calls
the k-agnostic closed-form refit core, and caches the result by its math key.
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
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCache,
)
from neural_forge.hf_lfm2_functional_output_refit import (  # noqa: E402
    Lfm2FunctionalOutputRefitResult,
    authenticate_lfm2_functional_output_refit_result,
    build_lfm2_functional_output_sparse_design,
    lfm2_functional_output_refit_request,
    refit_lfm2_functional_output_coefficients,
)
from neural_forge.hf_lfm2_sparse_semantic_codebook import (  # noqa: E402
    Lfm2SparseSemanticCodebook,
    load_lfm2_sparse_semantic_codebook,
)

from experiments.run_lfm2_semantic_step_bridge_control import (  # noqa: E402
    _checkpoint_artifact_inventory,
)
from experiments.run_lfm2_sparse_semantic_codebook import (  # noqa: E402
    _core_sparse_tensor_sha256,
    inspect_parent_tied_table_source,
    load_parent_tied_rows_from_local_safetensors,
)


RUN_SCHEMA = "neural-forge-lfm2-functional-output-refit-run-v1"
PROGRESS_SCHEMA = "neural-forge-lfm2-functional-output-refit-run-progress-v1"
TRACE_ROWS_SCHEMA = "neural-forge-lfm2-functional-output-cached-rows-v1"
METRIC_SCHEMA = "neural-forge-lfm2-functional-output-metric-v1"
SCORE_SCHEMA = "neural-forge-lfm2-functional-output-cached-score-v1"
PUBLICATION_SCHEMA = "neural-forge-lfm2-functional-output-publication-v1"
COEFFICIENT_FILENAME = "output-coefficients.safetensors"
RESULT_FILENAME = "result.json"
PUBLICATION_FILENAME = "publication.json"
OUTPUT_TENSOR_KEY = "output_semantic_code_coefficients"

_SHA256_ZERO = "0" * 64
_SUPPORTED_COMPUTE_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
}
_OPERATION_AUDIT = {
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "model_objects_constructed": 0,
    "model_forward_calls": 0,
    "model_generate_calls": 0,
    "parent_forward_calls": 0,
    "parent_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "global_map_solve_calls": 0,
    "network_access": False,
}


class Lfm2FunctionalOutputRefitRunnerError(RuntimeError):
    """CLI inputs, authenticated artifacts, or cached publications are invalid."""


ProgressSink = Callable[[Mapping[str, Any]], None]


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise Lfm2FunctionalOutputRefitRunnerError(
                "Receipts cannot contain non-finite values."
            )
        return value
    raise Lfm2FunctionalOutputRefitRunnerError(
        f"Receipt contains unsupported {type(value).__name__}."
    )


def _emit_json(details: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            _plain_json(details),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _emit(progress: ProgressSink, event: str, stage: str, **details: Any) -> None:
    progress(
        {
            "schema": PROGRESS_SCHEMA,
            "event": event,
            "stage": stage,
            **_plain_json(details),
        }
    )


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2FunctionalOutputRefitRunnerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2FunctionalOutputRefitRunnerError(f"{name} must be positive.")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2FunctionalOutputRefitRunnerError(
            f"{name} must be nonnegative."
        )
    return int(value)


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2FunctionalOutputRefitRunnerError(
            f"{name} must be finite and nonnegative."
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2FunctionalOutputRefitRunnerError(
            f"{name} must be finite and nonnegative."
        )
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result <= 0.0:
        raise Lfm2FunctionalOutputRefitRunnerError(f"{name} must be positive.")
    return result


def _compute_dtype(value: str, name: str) -> torch.dtype:
    dtype = _SUPPORTED_COMPUTE_DTYPES.get(str(value).lower())
    if dtype is None:
        raise Lfm2FunctionalOutputRefitRunnerError(
            f"{name} must be explicit FP32 or FP64."
        )
    return dtype


def _compute_device(value: str, name: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise Lfm2FunctionalOutputRefitRunnerError(f"{name} is invalid.") from exc
    if device.type not in {"cpu", "cuda"}:
        raise Lfm2FunctionalOutputRefitRunnerError(
            f"{name} must be a CPU or CUDA device."
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise Lfm2FunctionalOutputRefitRunnerError(
                f"{name} requests CUDA, but CUDA is unavailable."
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise Lfm2FunctionalOutputRefitRunnerError(
                f"{name} requests an unavailable CUDA device."
            )
        return torch.device("cuda", index)
    return torch.device("cpu")


def _input_directory(value: Path, name: str) -> Path:
    lexical = value.expanduser().absolute()
    if lexical.is_symlink():
        raise Lfm2FunctionalOutputRefitRunnerError(f"{name} cannot be a link.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2FunctionalOutputRefitRunnerError(f"{name} is missing.") from exc
    if resolved != lexical or not resolved.is_dir():
        raise Lfm2FunctionalOutputRefitRunnerError(
            f"{name} must be one canonical real directory."
        )
    return resolved


def _output_directory(value: Path) -> Path:
    lexical = value.expanduser().absolute()
    if lexical.exists() or lexical.is_symlink():
        if lexical.is_symlink() or not lexical.is_dir():
            raise Lfm2FunctionalOutputRefitRunnerError(
                "output directory must be one real directory."
            )
        return lexical.resolve(strict=True)
    lexical.mkdir(parents=True)
    return lexical.resolve(strict=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256(b"neural-forge-functional-output-runner-tensor-v1\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Lfm2FunctionalOutputRefitRunnerError(f"{name} is missing or unsafe.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2FunctionalOutputRefitRunnerError(f"{name} is invalid JSON.") from exc
    if not isinstance(value, dict):
        raise Lfm2FunctionalOutputRefitRunnerError(f"{name} must be a JSON object.")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            _plain_json(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_remove_stage(stage: Path, root: Path) -> None:
    try:
        resolved_stage = stage.resolve()
        resolved_root = root.resolve()
    except OSError:
        return
    if (
        resolved_stage.parent == resolved_root
        and stage.name.startswith(".functional-output-refit-stage-")
        and stage.is_dir()
        and not stage.is_symlink()
    ):
        shutil.rmtree(stage)


@dataclass(frozen=True, slots=True)
class CachedFunctionalRows:
    hidden: torch.Tensor
    winners: torch.Tensor
    batch_ids: torch.Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        hidden = self.hidden.detach().to(device="cpu").contiguous()
        winners = self.winners.detach().to(device="cpu", dtype=torch.int64).contiguous()
        batches = self.batch_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
        if (
            hidden.ndim != 2
            or not torch.is_floating_point(hidden)
            or hidden.requires_grad
            or not bool(torch.isfinite(hidden).all())
            or winners.ndim != 1
            or batches.ndim != 1
            or int(hidden.shape[0]) != int(winners.numel())
            or int(hidden.shape[0]) != int(batches.numel())
        ):
            raise Lfm2FunctionalOutputRefitRunnerError(
                "Cached functional rows are invalid or unaligned."
            )
        receipt = _plain_json(self.receipt)
        supplied = receipt.pop("receipt_sha256", None)
        if (
            supplied != canonical_sha256(receipt)
            or receipt.get("schema") != TRACE_ROWS_SCHEMA
            or receipt.get("row_count") != int(hidden.shape[0])
            or receipt.get("hidden_size") != int(hidden.shape[1])
            or receipt.get("hidden_sha256") != _tensor_sha256(hidden)
            or receipt.get("winner_sha256") != _tensor_sha256(winners)
            or receipt.get("batch_id_sha256") != _tensor_sha256(batches)
        ):
            raise Lfm2FunctionalOutputRefitRunnerError(
                "Cached functional row receipt binding is invalid."
            )
        object.__setattr__(self, "hidden", hidden.clone())
        object.__setattr__(self, "winners", winners.clone())
        object.__setattr__(self, "batch_ids", batches.clone())
        object.__setattr__(self, "receipt", {**receipt, "receipt_sha256": supplied})


def _head_logit_pairs(names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    inventory = frozenset(names)
    suffix = ".output.lm_head_input"
    result = []
    for hidden_name in sorted(name for name in inventory if name.endswith(suffix)):
        phase = hidden_name[: -len(suffix)]
        legacy = f"{phase}.logits"
        full = f"{phase}.full_prefix_logits"
        if legacy in inventory:
            result.append((hidden_name, legacy))
        elif full in inventory:
            result.append((hidden_name, full))
    return tuple(result)


def load_cached_functional_rows(
    *,
    trace_cache_directory: Path,
    expected_cache_key: str,
    expected_parent_fingerprint: str,
    expected_batch_count: int,
    expected_row_count: int,
    donor_row_count: int,
    hidden_size: int,
    progress: ProgressSink,
) -> CachedFunctionalRows:
    """Select and authenticate only terminal hidden/logit tensors from cache."""

    cache = Lfm2CircuitTraceCache.open(trace_cache_directory, verify=False)
    expected_key = _digest(expected_cache_key, "expected trace cache key")
    expected_parent = _digest(
        expected_parent_fingerprint, "expected trace parent fingerprint"
    )
    batches_expected = _positive_int(expected_batch_count, "expected_batch_count")
    rows_expected = _positive_int(expected_row_count, "expected_row_count")
    donor_count = _positive_int(donor_row_count, "donor_row_count")
    width = _positive_int(hidden_size, "hidden_size")
    if (
        cache.cache_key != expected_key
        or cache.parent_fingerprint != expected_parent
        or cache.batch_count != batches_expected
    ):
        raise Lfm2FunctionalOutputRefitRunnerError(
            "Trace cache metadata differs from the expected sealed cache."
        )
    _emit(
        progress,
        "trace-session-start",
        "trace-cache",
        cache_key=cache.cache_key,
        parent_fingerprint=cache.parent_fingerprint,
        batch_count=cache.batch_count,
    )
    hidden_rows: list[torch.Tensor] = []
    winner_rows: list[torch.Tensor] = []
    batch_rows: list[torch.Tensor] = []
    selected_pair_count = 0
    selected_logical_bytes = 0
    with cache.selective_authenticated_session() as session:
        for reference in session.batch_references:
            pairs = _head_logit_pairs(reference.ordered_tensor_names)
            if not pairs:
                raise Lfm2FunctionalOutputRefitRunnerError(
                    f"Trace batch {reference.index} has no terminal hidden/logit pairs."
                )
            selected_names = tuple(sorted({name for pair in pairs for name in pair}))
            payload = session.load_selected_tensors(reference.index, selected_names)
            batch_row_count = 0
            try:
                selected_logical_bytes += int(payload.logical_bytes)
                for hidden_name, logit_name in pairs:
                    hidden = payload.tensors[hidden_name]
                    logits = payload.tensors[logit_name]
                    if int(hidden.shape[-1]) != width or int(logits.shape[-1]) < donor_count:
                        raise Lfm2FunctionalOutputRefitRunnerError(
                            "Cached terminal tensor geometry differs from expectations."
                        )
                    flat_hidden = hidden.reshape(-1, width).to(dtype=torch.float32)
                    flat_logits = logits[..., :donor_count].reshape(-1, donor_count)
                    if int(flat_hidden.shape[0]) != int(flat_logits.shape[0]):
                        raise Lfm2FunctionalOutputRefitRunnerError(
                            "Cached terminal hidden/logit rows are unaligned."
                        )
                    hidden_rows.append(flat_hidden)
                    winner_rows.append(flat_logits.argmax(dim=-1).to(torch.int64))
                    batch_rows.append(
                        torch.full(
                            (int(flat_hidden.shape[0]),),
                            int(reference.index),
                            dtype=torch.int64,
                        )
                    )
                    batch_row_count += int(flat_hidden.shape[0])
                    selected_pair_count += 1
            finally:
                session.release_selected_tensors(payload)
            _emit(
                progress,
                "trace-batch-complete",
                "trace-cache",
                batch_index=int(reference.index),
                selected_pair_count=len(pairs),
                selected_row_count=batch_row_count,
                selected_logical_bytes=int(payload.logical_bytes),
            )
        session.verify()
        session_receipt = session.as_dict()
    hidden = torch.cat(hidden_rows, dim=0)
    winners = torch.cat(winner_rows, dim=0)
    batch_ids = torch.cat(batch_rows, dim=0)
    if int(hidden.shape[0]) != rows_expected:
        raise Lfm2FunctionalOutputRefitRunnerError(
            "Selected cached row count differs from expected_row_count."
        )
    if bool((winners < 0).any()) or bool((winners >= donor_count).any()):
        raise Lfm2FunctionalOutputRefitRunnerError(
            "Cached parent winner leaves the defined donor vocabulary."
        )
    body = {
        "schema": TRACE_ROWS_SCHEMA,
        "trace_cache_key": cache.cache_key,
        "trace_parent_fingerprint": cache.parent_fingerprint,
        "trace_batch_count": cache.batch_count,
        "selective_session_sha256": session_receipt["session_sha256"],
        "selected_pair_count": selected_pair_count,
        "selected_logical_bytes": selected_logical_bytes,
        "row_count": int(hidden.shape[0]),
        "hidden_size": int(hidden.shape[1]),
        "hidden_storage_dtype": str(hidden.dtype).removeprefix("torch."),
        "hidden_sha256": _tensor_sha256(hidden),
        "winner_sha256": _tensor_sha256(winners),
        "batch_id_sha256": _tensor_sha256(batch_ids),
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    rows = CachedFunctionalRows(
        hidden,
        winners,
        batch_ids,
        {**body, "receipt_sha256": canonical_sha256(body)},
    )
    _emit(
        progress,
        "trace-session-complete",
        "trace-cache",
        row_count=int(hidden.shape[0]),
        selected_pair_count=selected_pair_count,
        selective_session_sha256=session_receipt["session_sha256"],
    )
    return rows


def build_cached_functional_metric(
    rows: CachedFunctionalRows,
    *,
    holdout_modulus: int,
    holdout_remainder: int,
    metric_device: str,
    metric_dtype: str,
    progress: ProgressSink,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    modulus = _positive_int(holdout_modulus, "holdout_modulus")
    remainder = _nonnegative_int(holdout_remainder, "holdout_remainder")
    if remainder >= modulus:
        raise Lfm2FunctionalOutputRefitRunnerError(
            "holdout_remainder must be smaller than holdout_modulus."
        )
    device = _compute_device(metric_device, "metric_device")
    dtype = _compute_dtype(metric_dtype, "metric_dtype")
    holdout = rows.batch_ids.remainder(modulus).eq(remainder)
    fit = ~holdout
    if not bool(fit.any()) or not bool(holdout.any()):
        raise Lfm2FunctionalOutputRefitRunnerError(
            "Deterministic partition must contain fit and holdout rows."
        )
    _emit(
        progress,
        "metric-start",
        "functional-metric",
        fit_row_count=int(fit.sum().item()),
        holdout_row_count=int(holdout.sum().item()),
        metric_device=str(device),
        metric_dtype=str(dtype).removeprefix("torch."),
    )
    hidden = rows.hidden[fit].to(device=device, dtype=dtype)
    normalized = torch.nn.functional.normalize(hidden, dim=1) * math.sqrt(
        int(hidden.shape[1])
    )
    metric = normalized.transpose(0, 1) @ normalized / int(normalized.shape[0])
    metric = ((metric + metric.transpose(0, 1)) * 0.5).to(device="cpu")
    if not bool(torch.isfinite(metric).all()):
        raise Lfm2FunctionalOutputRefitRunnerError(
            "Cached functional metric contains non-finite values."
        )
    body = {
        "schema": METRIC_SCHEMA,
        "trace_rows_receipt_sha256": rows.receipt["receipt_sha256"],
        "partition": {
            "method": "batch-id-modulus",
            "holdout_modulus": modulus,
            "holdout_remainder": remainder,
            "fit_row_count": int(fit.sum().item()),
            "holdout_row_count": int(holdout.sum().item()),
        },
        "hidden_size": int(metric.shape[0]),
        "metric_storage_device": "cpu",
        "metric_dtype": str(metric.dtype).removeprefix("torch."),
        "metric_sha256": _tensor_sha256(metric),
        "metric_trace": float(torch.trace(metric).item()),
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    _emit(
        progress,
        "metric-complete",
        "functional-metric",
        metric_sha256=receipt["metric_sha256"],
        metric_trace=receipt["metric_trace"],
        metric_dtype=receipt["metric_dtype"],
    )
    return metric, holdout, receipt


def score_cached_functional_output(
    *,
    codebook: Lfm2SparseSemanticCodebook,
    output_coefficients: torch.Tensor,
    target_table: torch.Tensor,
    rows: CachedFunctionalRows,
    holdout: torch.Tensor,
    score_device: str,
    score_dtype: str,
    rows_per_chunk: int,
    progress: ProgressSink,
) -> dict[str, Any]:
    device = _compute_device(score_device, "score_device")
    dtype = _compute_dtype(score_dtype, "score_dtype")
    chunk_size = _positive_int(rows_per_chunk, "score_rows_per_chunk")
    if tuple(holdout.shape) != (int(rows.hidden.shape[0]),) or holdout.dtype != torch.bool:
        raise Lfm2FunctionalOutputRefitRunnerError("holdout mask is invalid.")
    design = build_lfm2_functional_output_sparse_design(
        codebook, output_coefficients, device=device
    )
    table = target_table.detach().to(device=device, dtype=dtype)
    predictions: list[torch.Tensor] = []
    _emit(
        progress,
        "score-start",
        "cached-score",
        score_device=str(device),
        score_dtype=str(dtype).removeprefix("torch."),
        row_count=int(rows.hidden.shape[0]),
        rows_per_chunk=chunk_size,
    )
    with torch.no_grad():
        for chunk_index, start in enumerate(
            range(0, int(rows.hidden.shape[0]), chunk_size)
        ):
            stop = min(start + chunk_size, int(rows.hidden.shape[0]))
            hidden = rows.hidden[start:stop].to(device=device, dtype=dtype)
            target_scores = hidden @ table.transpose(0, 1)
            logits = torch.sparse.mm(
                design, target_scores.transpose(0, 1)
            ).transpose(0, 1)
            predictions.append(logits.argmax(dim=-1).to(device="cpu"))
            _emit(
                progress,
                "score-chunk-complete",
                "cached-score",
                chunk_index=chunk_index,
                row_start=start,
                row_stop=stop,
                total_rows=int(rows.hidden.shape[0]),
                score_device=str(device),
                score_dtype=str(dtype).removeprefix("torch."),
            )
    predicted = torch.cat(predictions)
    matched = predicted.eq(rows.winners)
    direct_lookup = torch.zeros(
        int(codebook.receipt["donor_row_count"]), dtype=torch.bool
    )
    direct_lookup[codebook.target_to_donor] = True
    parent_direct = direct_lookup.index_select(0, rows.winners)

    def counts(mask: torch.Tensor) -> dict[str, Any]:
        total = int(mask.sum().item())
        matches = int((matched & mask).sum().item())
        return {
            "row_count": total,
            "top1_match_count": matches,
            "top1_match_rate": None if total == 0 else matches / total,
        }

    mismatches = [
        {
            "row_index": index,
            "batch_id": int(rows.batch_ids[index]),
            "parent_winner": int(rows.winners[index]),
            "predicted_winner": int(predicted[index]),
            "parent_winner_route": (
                "direct" if bool(parent_direct[index]) else "multi"
            ),
            "partition": "holdout" if bool(holdout[index]) else "fit",
        }
        for index in torch.nonzero(~matched, as_tuple=False).reshape(-1).tolist()
    ]
    body = {
        "schema": SCORE_SCHEMA,
        "metrics_are_advisory_not_an_intelligence_gate": True,
        "all": counts(torch.ones_like(matched, dtype=torch.bool)),
        "fit": counts(~holdout),
        "holdout": counts(holdout),
        "parent_direct": counts(parent_direct),
        "parent_multi": counts(~parent_direct),
        "prediction_sha256": _tensor_sha256(predicted),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "score_device": str(device),
        "score_dtype": str(dtype).removeprefix("torch."),
        "score_rows_per_chunk": chunk_size,
        "dense_global_donor_by_target_map_materializations": 0,
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    result = {**body, "score_sha256": canonical_sha256(body)}
    _emit(
        progress,
        "score-complete",
        "cached-score",
        all=result["all"],
        fit=result["fit"],
        holdout=result["holdout"],
        parent_direct=result["parent_direct"],
        parent_multi=result["parent_multi"],
        score_sha256=result["score_sha256"],
    )
    return result


def _load_cached_result(
    cache_directory: Path,
    *,
    expected_cache_key: str,
) -> Lfm2FunctionalOutputRefitResult:
    directory = _input_directory(cache_directory, "functional output cache entry")
    if {path.name for path in directory.iterdir()} != {
        COEFFICIENT_FILENAME,
        RESULT_FILENAME,
        PUBLICATION_FILENAME,
    }:
        raise Lfm2FunctionalOutputRefitRunnerError(
            "Functional output cache entry has unexpected files."
        )
    coefficient_path = directory / COEFFICIENT_FILENAME
    result_path = directory / RESULT_FILENAME
    publication_path = directory / PUBLICATION_FILENAME
    if any(
        path.is_symlink() or not path.is_file()
        for path in (coefficient_path, result_path, publication_path)
    ):
        raise Lfm2FunctionalOutputRefitRunnerError(
            "Functional output cache files are missing or symlinked."
        )
    publication = _read_json(publication_path, "functional output publication")
    supplied = publication.pop("publication_sha256", None)
    if (
        supplied != canonical_sha256(publication)
        or publication.get("schema") != PUBLICATION_SCHEMA
        or publication.get("cache_key") != expected_cache_key
        or publication.get("coefficient_file_sha256")
        != _file_sha256(coefficient_path)
        or publication.get("result_file_sha256") != _file_sha256(result_path)
    ):
        raise Lfm2FunctionalOutputRefitRunnerError(
            "Functional output cache publication authentication failed."
        )
    with safe_open(
        str(coefficient_path), framework="pt", device="cpu"
    ) as handle:
        if tuple(handle.keys()) != (OUTPUT_TENSOR_KEY,):
            raise Lfm2FunctionalOutputRefitRunnerError(
                "Functional output coefficient tensor inventory is invalid."
            )
        coefficients = handle.get_tensor(OUTPUT_TENSOR_KEY)
    receipt = _read_json(result_path, "functional output result receipt")
    result = authenticate_lfm2_functional_output_refit_result(
        output_semantic_code_coefficients=coefficients,
        receipt=receipt,
        expected_cache_key=expected_cache_key,
    )
    if publication.get("result_sha256") != result.result_sha256:
        raise Lfm2FunctionalOutputRefitRunnerError(
            "Functional output publication names another sealed result."
        )
    return result


def _publish_cached_result(
    cache_root: Path,
    result: Lfm2FunctionalOutputRefitResult,
) -> tuple[Lfm2FunctionalOutputRefitResult, Path, str]:
    destination = cache_root / result.cache_key
    if destination.exists():
        return (
            _load_cached_result(destination, expected_cache_key=result.cache_key),
            destination,
            "authenticated-cache-race-hit",
        )
    stage = Path(
        tempfile.mkdtemp(prefix=".functional-output-refit-stage-", dir=cache_root)
    )
    try:
        coefficient_path = stage / COEFFICIENT_FILENAME
        result_path = stage / RESULT_FILENAME
        publication_path = stage / PUBLICATION_FILENAME
        save_file(
            {OUTPUT_TENSOR_KEY: result.output_semantic_code_coefficients},
            coefficient_path,
        )
        _atomic_write_json(result_path, result.as_dict())
        body = {
            "schema": PUBLICATION_SCHEMA,
            "cache_key": result.cache_key,
            "result_sha256": result.result_sha256,
            "coefficient_file": COEFFICIENT_FILENAME,
            "coefficient_file_sha256": _file_sha256(coefficient_path),
            "result_file": RESULT_FILENAME,
            "result_file_sha256": _file_sha256(result_path),
            "atomic_directory_publication": True,
        }
        _atomic_write_json(
            publication_path,
            {**body, "publication_sha256": canonical_sha256(body)},
        )
        try:
            os.replace(stage, destination)
        except OSError:
            if not destination.is_dir() or destination.is_symlink():
                raise
        loaded = _load_cached_result(destination, expected_cache_key=result.cache_key)
        source = (
            "computed-and-atomically-published"
            if loaded.result_sha256 == result.result_sha256
            else "authenticated-cache-race-hit"
        )
        return loaded, destination, source
    finally:
        if stage.exists():
            _safe_remove_stage(stage, cache_root)


def _cache_or_compute_result(
    *,
    output_root: Path,
    codebook: Lfm2SparseSemanticCodebook,
    donor_table: torch.Tensor,
    target_table: torch.Tensor,
    metric: torch.Tensor,
    functional_weight: float,
    solve_ridge: float,
    solve_device: str,
    solve_dtype: str,
    rows_per_chunk: int,
    progress: ProgressSink,
) -> tuple[Lfm2FunctionalOutputRefitResult, Path, str]:
    request = lfm2_functional_output_refit_request(
        codebook=codebook,
        donor_table=donor_table,
        target_table=target_table,
        functional_metric=metric,
        functional_weight=functional_weight,
        solve_ridge=solve_ridge,
        solve_device=solve_device,
        solve_dtype=solve_dtype,
        rows_per_chunk=rows_per_chunk,
    )
    cache_key = str(request["cache_key"])
    cache_root = output_root / "math-cache"
    cache_root.mkdir(exist_ok=True)
    if cache_root.is_symlink():
        raise Lfm2FunctionalOutputRefitRunnerError("math cache root cannot be a link.")
    cache_directory = cache_root / cache_key
    if cache_directory.exists():
        _emit(
            progress,
            "math-cache-hit",
            "math-cache",
            cache_key=cache_key,
            cache_directory=str(cache_directory),
            execution_contract=request["execution_contract"],
        )
        result = _load_cached_result(cache_directory, expected_cache_key=cache_key)
        return result, cache_directory, "authenticated-cache-hit"
    _emit(
        progress,
        "math-cache-miss",
        "math-cache",
        cache_key=cache_key,
        cache_directory=str(cache_directory),
        execution_contract=request["execution_contract"],
        dtype_contract=request["dtype_contract"],
    )
    result = refit_lfm2_functional_output_coefficients(
        codebook=codebook,
        donor_table=donor_table,
        target_table=target_table,
        functional_metric=metric,
        functional_weight=functional_weight,
        solve_ridge=solve_ridge,
        solve_device=solve_device,
        solve_dtype=solve_dtype,
        rows_per_chunk=rows_per_chunk,
        progress=progress,
    )
    loaded, directory, source = _publish_cached_result(cache_root, result)
    _emit(
        progress,
        "math-cache-published",
        "math-cache",
        cache_key=cache_key,
        cache_directory=str(directory),
        result_sha256=loaded.result_sha256,
        source=source,
    )
    return loaded, directory, source


def run(args: argparse.Namespace, *, progress: ProgressSink = _emit_json) -> dict[str, Any]:
    torch.set_grad_enabled(False)
    parent_directory = _input_directory(
        args.parent_model_directory, "parent model directory"
    )
    checkpoint_directory = _input_directory(
        args.checkpoint_directory, "child checkpoint directory"
    )
    codebook_directory = _input_directory(
        args.codebook_directory, "sparse codebook directory"
    )
    trace_directory = _input_directory(
        args.trace_cache_directory, "trace cache directory"
    )
    output_root = _output_directory(args.output_directory)
    parent_identity = _digest(
        args.expected_parent_identity_sha256, "expected parent identity"
    )
    expected_inspection = _digest(
        args.expected_parent_source_inspection_sha256,
        "expected parent source inspection",
    )
    expected_checkpoint_inventory = _digest(
        args.expected_checkpoint_artifact_inventory_sha256,
        "expected checkpoint artifact inventory",
    )
    expected_codebook_sha = _digest(
        args.expected_codebook_artifact_sha256, "expected codebook artifact"
    )
    donor_count = _positive_int(
        args.expected_codebook_donor_row_count, "expected codebook donor rows"
    )
    target_count = _positive_int(
        args.expected_codebook_target_row_count, "expected codebook target rows"
    )
    hidden_size = _positive_int(args.expected_hidden_size, "expected hidden size")
    functional_weight = _finite_nonnegative(
        args.functional_weight, "functional_weight"
    )
    solve_ridge = _positive_float(args.solve_ridge, "solve_ridge")
    rows_per_chunk = _positive_int(args.rows_per_chunk, "rows_per_chunk")
    _emit(
        progress,
        "preflight-start",
        "preflight",
        parent_model_directory=str(parent_directory),
        checkpoint_directory=str(checkpoint_directory),
        codebook_directory=str(codebook_directory),
        trace_cache_directory=str(trace_directory),
        output_directory=str(output_root),
        expected_geometry={
            "donor_row_count": donor_count,
            "target_row_count": target_count,
            "hidden_size": hidden_size,
        },
    )

    codebook = load_lfm2_sparse_semantic_codebook(
        codebook_directory, expected_artifact_sha256=expected_codebook_sha
    )
    codebook_receipt = codebook.as_dict()
    if (
        codebook_receipt.get("donor_row_count") != donor_count
        or codebook_receipt.get("target_row_count") != target_count
        or codebook_receipt.get("hidden_size") != hidden_size
    ):
        raise Lfm2FunctionalOutputRefitRunnerError(
            "Authenticated codebook geometry differs from CLI expectations."
        )
    parent_inspection = inspect_parent_tied_table_source(
        parent_directory, tensor_key=args.parent_tensor_key
    )
    if parent_inspection.get("inspection_sha256") != expected_inspection:
        raise Lfm2FunctionalOutputRefitRunnerError(
            "Parent tied-table source inspection differs from expectation."
        )
    checkpoint_inventory = _checkpoint_artifact_inventory(checkpoint_directory)
    checkpoint_inventory_sha = canonical_sha256(checkpoint_inventory)
    if checkpoint_inventory_sha != expected_checkpoint_inventory:
        raise Lfm2FunctionalOutputRefitRunnerError(
            "Child checkpoint artifact inventory differs from expectation."
        )
    donor_loaded = load_parent_tied_rows_from_local_safetensors(
        parent_directory,
        expected_parent_identity_sha256=parent_identity,
        donor_row_count=donor_count,
        hidden_size=hidden_size,
        publication_dtype=torch.float32,
        tensor_key=args.parent_tensor_key,
    )
    target_loaded = load_parent_tied_rows_from_local_safetensors(
        checkpoint_directory,
        expected_parent_identity_sha256=checkpoint_inventory_sha,
        donor_row_count=target_count,
        hidden_size=hidden_size,
        publication_dtype=torch.float32,
        tensor_key=args.target_tensor_key,
    )
    if (
        _core_sparse_tensor_sha256(donor_loaded.table)
        != codebook_receipt.get("donor_table_sha256")
    ):
        raise Lfm2FunctionalOutputRefitRunnerError(
            "Selective parent table differs from the table sealed by the codebook."
        )
    _emit(
        progress,
        "local-tables-authenticated",
        "preflight",
        parent_source_inspection_sha256=expected_inspection,
        checkpoint_artifact_inventory_sha256=checkpoint_inventory_sha,
        donor_table_receipt=donor_loaded.receipt,
        target_table_receipt=target_loaded.receipt,
        codebook_artifact_sha256=codebook.artifact_sha256,
        support_width_histogram={
            str(int(width)): int(count)
            for width, count in zip(
                *torch.unique(
                    codebook.semantic_code_offsets[1:]
                    - codebook.semantic_code_offsets[:-1],
                    return_counts=True,
                ),
                strict=True,
            )
        },
    )
    rows = load_cached_functional_rows(
        trace_cache_directory=trace_directory,
        expected_cache_key=args.expected_trace_cache_key,
        expected_parent_fingerprint=args.expected_trace_parent_fingerprint,
        expected_batch_count=args.expected_trace_batch_count,
        expected_row_count=args.expected_trace_row_count,
        donor_row_count=donor_count,
        hidden_size=hidden_size,
        progress=progress,
    )
    metric, holdout, metric_receipt = build_cached_functional_metric(
        rows,
        holdout_modulus=args.holdout_modulus,
        holdout_remainder=args.holdout_remainder,
        metric_device=args.metric_device,
        metric_dtype=args.metric_dtype,
        progress=progress,
    )
    result, cache_directory, cache_source = _cache_or_compute_result(
        output_root=output_root,
        codebook=codebook,
        donor_table=donor_loaded.table,
        target_table=target_loaded.table,
        metric=metric,
        functional_weight=functional_weight,
        solve_ridge=solve_ridge,
        solve_device=args.solve_device,
        solve_dtype=args.solve_dtype,
        rows_per_chunk=rows_per_chunk,
        progress=progress,
    )
    score = score_cached_functional_output(
        codebook=codebook,
        output_coefficients=result.output_semantic_code_coefficients,
        target_table=target_loaded.table,
        rows=rows,
        holdout=holdout,
        score_device=args.score_device,
        score_dtype=args.score_dtype,
        rows_per_chunk=args.score_rows_per_chunk,
        progress=progress,
    )
    body = {
        "schema": RUN_SCHEMA,
        "status": "complete-cached-trace-analytic-output-refit",
        "math_cache_key": result.cache_key,
        "math_cache_source": cache_source,
        "math_cache_directory": str(cache_directory),
        "result_sha256": result.result_sha256,
        "output_semantic_code_coefficients_sha256": result.receipt[
            "output_semantic_code_coefficients_sha256"
        ],
        "input_codebook_artifact_sha256": codebook.artifact_sha256,
        "expected_geometry": {
            "donor_row_count": donor_count,
            "target_row_count": target_count,
            "hidden_size": hidden_size,
        },
        "parent_identity_sha256": parent_identity,
        "parent_source_inspection_sha256": expected_inspection,
        "checkpoint_artifact_inventory_sha256": checkpoint_inventory_sha,
        "trace_rows_receipt": rows.receipt,
        "functional_metric_receipt": metric_receipt,
        "cached_score": score,
        "execution_contract": {
            "metric_device": args.metric_device,
            "metric_dtype": args.metric_dtype,
            "solve_device": args.solve_device,
            "solve_dtype": args.solve_dtype,
            "rows_per_chunk": rows_per_chunk,
            "score_device": args.score_device,
            "score_dtype": args.score_dtype,
            "score_rows_per_chunk": _positive_int(
                args.score_rows_per_chunk, "score_rows_per_chunk"
            ),
        },
        "manual_whole_response_review_remains_authoritative": True,
        "cached_numeric_score_is_advisory": True,
        "input_output_channels_separate": True,
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    run_receipt = {**body, "run_sha256": canonical_sha256(body)}
    run_path = output_root / f"run-{result.cache_key}.json"
    _atomic_write_json(run_path, run_receipt)
    _emit(
        progress,
        "run-complete",
        "complete",
        math_cache_key=result.cache_key,
        math_cache_source=cache_source,
        result_sha256=result.result_sha256,
        run_file=str(run_path),
        run_file_sha256=_file_sha256(run_path),
        cached_score=score,
    )
    return run_receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model-directory", type=Path, required=True)
    parser.add_argument("--expected-parent-identity-sha256", required=True)
    parser.add_argument(
        "--expected-parent-source-inspection-sha256", required=True
    )
    parser.add_argument("--parent-tensor-key", required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-artifact-inventory-sha256", required=True
    )
    parser.add_argument("--target-tensor-key", required=True)
    parser.add_argument("--codebook-directory", type=Path, required=True)
    parser.add_argument("--expected-codebook-artifact-sha256", required=True)
    parser.add_argument(
        "--expected-codebook-donor-row-count", type=int, required=True
    )
    parser.add_argument(
        "--expected-codebook-target-row-count", type=int, required=True
    )
    parser.add_argument("--expected-hidden-size", type=int, required=True)
    parser.add_argument("--trace-cache-directory", type=Path, required=True)
    parser.add_argument("--expected-trace-cache-key", required=True)
    parser.add_argument("--expected-trace-parent-fingerprint", required=True)
    parser.add_argument("--expected-trace-batch-count", type=int, required=True)
    parser.add_argument("--expected-trace-row-count", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--holdout-modulus", type=int, required=True)
    parser.add_argument("--holdout-remainder", type=int, required=True)
    parser.add_argument("--functional-weight", type=float, required=True)
    parser.add_argument("--solve-ridge", type=float, required=True)
    parser.add_argument("--metric-device", required=True)
    parser.add_argument("--metric-dtype", required=True)
    parser.add_argument("--solve-device", required=True)
    parser.add_argument("--solve-dtype", required=True)
    parser.add_argument("--rows-per-chunk", type=int, required=True)
    parser.add_argument("--score-device", required=True)
    parser.add_argument("--score-dtype", required=True)
    parser.add_argument("--score-rows-per-chunk", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except BaseException as exc:
        _emit_json(
            {
                "schema": PROGRESS_SCHEMA,
                "event": "run-failed",
                "stage": "failure",
                "error_type": type(exc).__qualname__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
