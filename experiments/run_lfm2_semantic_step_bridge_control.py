"""Run the same-body LFM2 hard-anchor control on the donor semantic clock.

This runner consumes an already-folded local child checkpoint.  It compiles
the global raw-token span bridge once, binds the checkpoint's existing tied
table without fitting or copying it, and then decodes each review prompt as an
independently resumable phase.  Donor tokens are the only transformer clock;
target spans are routing metadata and the complete donor response is decoded
before one whole-string target retokenization.

There is deliberately no parent model load, parent forward, ``generate()``,
training, optimizer, gradient, least-squares solve, parameter copy, pruning,
quantization, network access, or automatic intelligence gate in this program.
The cached parent envelope is attached only for side-by-side manual review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_phase_cache import (  # noqa: E402
    DynamicFoldPhaseCache,
    FoldPhaseEventLog,
    FoldPhaseKey,
    VerifiedFoldPhaseCacheEntry,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    Lfm2DirectLocalCheckpointIO,
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_semantic_step_model_runtime import (  # noqa: E402
    run_lfm2_semantic_step_model_review,
)
from neural_forge.hf_lfm2_semantic_step_span_adapter import (  # noqa: E402
    compile_lfm2_semantic_step_span_evidence_from_local_tokenizers,
    lfm2_semantic_step_span_adapter_cache_contract,
)
from neural_forge.hf_lfm2_semantic_step_token_fold import (  # noqa: E402
    bind_lfm2_semantic_step_hard_anchor_runtime,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)
from neural_forge.lfm2_decoded_review import (  # noqa: E402
    load_review_prompts,
)


RUN_SCHEMA = "neural-forge-lfm2-semantic-step-bridge-control-run-v1"
DOSSIER_SCHEMA = "neural-forge-lfm2-semantic-step-manual-review-dossier-v1"
SPAN_PHASE_NAME = "lfm2-semantic-step-global-raw-span"
PROMPT_PHASE_NAME = "lfm2-semantic-step-whole-response"
PROMPT_PHASE_ABI = "neural-forge-lfm2-semantic-step-prompt-cache-v1"
FAILURE_SCHEMA = "neural-forge-lfm2-semantic-step-bridge-control-failure-v1"
FAILURE_POINTER_SCHEMA = (
    "neural-forge-lfm2-semantic-step-bridge-control-failure-pointer-v1"
)
CONSOLE_PROGRESS_SCHEMA = (
    "neural-forge-lfm2-semantic-step-console-progress-v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DTYPE_ALIASES = {
    "auto": "auto",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
    "fp64": "float64",
    "float64": "float64",
    "double": "float64",
}
_TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class Lfm2SemanticStepBridgeControlError(RuntimeError):
    """The cached same-body semantic control could not be authenticated."""


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise Lfm2SemanticStepBridgeControlError(
        f"Evidence contains non-JSON value {type(value).__name__}."
    )


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise Lfm2SemanticStepBridgeControlError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2SemanticStepBridgeControlError(f"{name} must be positive.")
    return int(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _local_directory(path: Path, name: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise Lfm2SemanticStepBridgeControlError(f"{name} cannot be a link.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2SemanticStepBridgeControlError(f"{name} is missing.") from exc
    if resolved != lexical or not resolved.is_dir():
        raise Lfm2SemanticStepBridgeControlError(
            f"{name} must be one canonical real directory."
        )
    return resolved


def _local_file(path: Path, name: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise Lfm2SemanticStepBridgeControlError(f"{name} cannot be a link.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2SemanticStepBridgeControlError(f"{name} is missing.") from exc
    if resolved != lexical or not resolved.is_file():
        raise Lfm2SemanticStepBridgeControlError(
            f"{name} must be one canonical real file."
        )
    return resolved


def _output_directory(path: Path, name: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.exists() or lexical.is_symlink():
        if lexical.is_symlink() or not lexical.is_dir():
            raise Lfm2SemanticStepBridgeControlError(
                f"{name} must be one real directory."
            )
        return lexical.resolve(strict=True)
    lexical.mkdir(parents=True)
    return lexical.resolve(strict=True)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Lfm2SemanticStepBridgeControlError(
            f"{name} is missing or is not a real file."
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SemanticStepBridgeControlError(
            f"{name} is not readable JSON."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2SemanticStepBridgeControlError(f"{name} must be a JSON object.")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            _plain_json(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
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


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish one immutable JSON record without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            _plain_json(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise Lfm2SemanticStepBridgeControlError(
                    "Content-addressed failure path collision."
                ) from None
    finally:
        if temporary.exists():
            temporary.unlink()


def _record_failure(run_root: Path, exc: BaseException) -> tuple[Path, dict[str, Any]]:
    """Append one traceback-bearing failure and update a small latest pointer."""

    failures_root = _output_directory(
        run_root / "failures",
        "failures directory",
    )
    recorded_at_utc = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    failure_body = {
        "schema": FAILURE_SCHEMA,
        "status": "failed",
        "recorded_at_utc": recorded_at_utc,
        "recorded_at_unix_ns": time.time_ns(),
        "error_type": type(exc).__qualname__,
        "error": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
        "operation_audit": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parent_model_load_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "network_access": False,
        },
    }
    failure = {
        **failure_body,
        "result_sha256": canonical_sha256(failure_body),
    }
    failure_path = failures_root / f"{failure['result_sha256']}.json"
    _write_new_json(failure_path, failure)
    pointer_body = {
        "schema": FAILURE_POINTER_SCHEMA,
        "failure_result_sha256": failure["result_sha256"],
        "failure_relative_path": failure_path.relative_to(run_root).as_posix(),
        "recorded_at_utc": recorded_at_utc,
    }
    pointer = {
        **pointer_body,
        "pointer_sha256": canonical_sha256(pointer_body),
    }
    _atomic_write_json(run_root / "latest-failure.json", pointer)
    return failure_path, failure


def _verified_sealed_json(
    path: Path,
    *,
    seal_field: str,
    expected_sha256: str | None,
    name: str,
) -> dict[str, Any]:
    value = _read_json(path, name)
    body = dict(value)
    supplied = body.pop(seal_field, None)
    if (
        not isinstance(supplied, str)
        or supplied != canonical_sha256(body)
        or (
            expected_sha256 is not None
            and supplied != _digest(expected_sha256, f"expected {name} seal")
        )
    ):
        raise Lfm2SemanticStepBridgeControlError(
            f"{name} self-hash or expected seal does not verify."
        )
    return value


def _unique_parameter_count(model: Any) -> int:
    parameters = tuple(model.parameters())
    if not parameters:
        raise Lfm2SemanticStepBridgeControlError("Child has no parameters.")
    return sum(int(parameter.numel()) for parameter in parameters)


def _current_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError, RuntimeError):
        return None


def runtime_telemetry(
    *,
    started_at: float,
    device: str,
    requested_dtype: str,
    resolved_dtype: str,
) -> dict[str, Any]:
    """Return compact per-step process/GPU telemetry without synchronizing."""

    details: dict[str, Any] = {
        "elapsed_seconds": max(0.0, time.monotonic() - started_at),
        "rss_bytes": _current_rss_bytes(),
        "device": device,
        "requested_dtype": requested_dtype,
        "resolved_dtype": resolved_dtype,
    }
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device() if torch_device.index is None else torch_device.index
        details.update(
            {
                "cuda_device_index": int(index),
                "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(index)),
                "cuda_max_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(index)
                ),
            }
        )
    else:
        details.update(
            {
                "cuda_device_index": None,
                "cuda_memory_allocated_bytes": 0,
                "cuda_memory_reserved_bytes": 0,
                "cuda_max_memory_allocated_bytes": 0,
            }
        )
    return details


class ConsoleProgressReporter:
    """Emit bounded canonical JSON while durable logs retain every step."""

    def __init__(self, *, semantic_step_every: int = 16, stream: Any = None) -> None:
        self.semantic_step_every = _positive_int(
            semantic_step_every, "console semantic-step frequency"
        )
        self.stream = stream
        self._pending_semantic: tuple[dict[str, Any], bool] | None = None

    def _write(
        self,
        *,
        phase: str,
        details: Mapping[str, Any],
        reason: str,
    ) -> None:
        body = {
            "schema": CONSOLE_PROGRESS_SCHEMA,
            "phase": phase,
            "reason": reason,
            "details": _plain_json(details),
        }
        line = json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        kwargs: dict[str, Any] = {"flush": True}
        if self.stream is not None:
            kwargs["file"] = self.stream
        print(line, **kwargs)

    def span(self, details: Mapping[str, Any]) -> None:
        """Print every bounded span-compiler progress record."""

        self._write(phase="span", details=details, reason="span-progress")

    def runtime(self, details: Mapping[str, Any]) -> None:
        """Print lifecycle records and a throttled semantic-step sample."""

        value = _plain_json(details)
        progress = value.get("runtime_progress")
        if not isinstance(progress, Mapping):
            return
        stage = progress.get("stage")
        if stage == "semantic-step-complete":
            step_index = progress.get("step_index")
            if isinstance(step_index, bool) or not isinstance(step_index, int):
                return
            eos = progress.get("eos_selected") is True
            periodic = (step_index + 1) % self.semantic_step_every == 0
            emitted = eos or periodic
            self._pending_semantic = (value, emitted)
            if emitted:
                reason = "eos-semantic-step" if eos else "periodic-semantic-step"
                self._write(phase="runtime", details=value, reason=reason)
            return
        if stage == "prompt-complete":
            pending = self._pending_semantic
            if pending is not None and not pending[1]:
                self._write(
                    phase="runtime",
                    details=pending[0],
                    reason="final-semantic-step",
                )
            self._pending_semantic = None
        if stage in {
            "runtime-start",
            "prompt-start",
            "prompt-prefill-complete",
            "prompt-complete",
            "runtime-complete",
        }:
            self._write(
                phase="runtime",
                details=value,
                reason=f"lifecycle-{stage}",
            )


def resolve_runtime_dtype(
    requested: str,
    device: str,
) -> tuple[torch.dtype, dict[str, Any]]:
    """Resolve aliases and one backend-safe automatic/fallback dtype policy."""

    normalized = _DTYPE_ALIASES.get(str(requested).strip().lower())
    if normalized is None:
        raise Lfm2SemanticStepBridgeControlError(
            f"Unsupported runtime dtype {requested!r}."
        )
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise Lfm2SemanticStepBridgeControlError(
            f"CUDA device {device!r} was requested but CUDA is unavailable."
        )
    if torch_device.type == "cuda" and torch_device.index is not None:
        if torch_device.index < 0 or torch_device.index >= torch.cuda.device_count():
            raise Lfm2SemanticStepBridgeControlError(
                f"CUDA device {device!r} is outside the visible inventory."
            )

    reason = "explicit-supported-request"
    resolved = normalized
    if normalized == "auto":
        if torch_device.type == "cuda":
            with torch.cuda.device(torch_device):
                resolved = (
                    "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
                )
            reason = "automatic-cuda-capability-selection"
        else:
            resolved = "float32"
            reason = "automatic-cpu-safe-selection"
    elif torch_device.type == "cpu" and normalized == "float16":
        resolved = "float32"
        reason = "cpu-float16-runtime-fallback-to-float32"
    elif torch_device.type == "cuda" and normalized == "bfloat16":
        with torch.cuda.device(torch_device):
            supported = bool(torch.cuda.is_bf16_supported())
        if not supported:
            resolved = "float16"
            reason = "cuda-bfloat16-unsupported-fallback-to-float16"

    receipt = {
        "requested_dtype": normalized,
        "resolved_dtype": resolved,
        "device": str(torch_device),
        "resolution_reason": reason,
        "silent_dtype_coercion": False,
        "fallback_recorded": normalized not in {"auto", resolved},
    }
    return _TORCH_DTYPES[resolved], receipt


def resolve_checkpoint_provenance_dtype(requested: str) -> torch.dtype:
    """Resolve the exact dtype sealed by legacy checkpoint I/O policy."""

    normalized = _DTYPE_ALIASES.get(str(requested).strip().lower())
    if normalized is None or normalized == "auto":
        raise Lfm2SemanticStepBridgeControlError(
            "Checkpoint provenance dtype must be one explicit floating dtype."
        )
    return _TORCH_DTYPES[normalized]


def span_phase_key_from_contract(contract: Mapping[str, Any]) -> FoldPhaseKey:
    """Turn the adapter's authenticated cache contract into a phase key."""

    value = _plain_json(contract)
    supplied = value.pop("contract_sha256", None)
    if supplied != canonical_sha256(value):
        raise Lfm2SemanticStepBridgeControlError(
            "Span adapter cache contract self-hash does not verify."
        )
    dependencies = value.get("dependencies")
    numeric = value.get("numeric_contract")
    abi = value.get("abi")
    if not all(isinstance(item, Mapping) and item for item in (dependencies, numeric, abi)):
        raise Lfm2SemanticStepBridgeControlError(
            "Span adapter cache contract is incomplete."
        )
    return FoldPhaseKey(
        phase_name=SPAN_PHASE_NAME,
        dependency_sha256s=dependencies,
        numeric_contract=numeric,
        abi=abi,
    )


def build_prompt_phase_key(
    *,
    child_state_sha256: str,
    routing_contract_sha256: str,
    bridge_artifact_sha256: str,
    donor_tokenizer_semantics_sha256: str,
    target_tokenizer_semantics_sha256: str,
    prompt: Mapping[str, Any],
    device: str,
    dtype_receipt: Mapping[str, Any],
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
    maximum_embedding_rows_per_chunk: int,
    maximum_donor_rows_per_score_chunk: int,
) -> FoldPhaseKey:
    """Build one deterministic cache identity for one decoded prompt."""

    prompt_value = _plain_json(prompt)
    numeric_contract = {
        "selection": "greedy-defined-donor-argmax",
        "device": str(torch.device(device)),
        "dtype_resolution": _plain_json(dtype_receipt),
        "maximum_prompt_tokens": _positive_int(
            maximum_prompt_tokens, "maximum_prompt_tokens"
        ),
        "maximum_new_tokens": _positive_int(
            maximum_new_tokens, "maximum_new_tokens"
        ),
        "maximum_embedding_rows_per_chunk": _positive_int(
            maximum_embedding_rows_per_chunk,
            "maximum_embedding_rows_per_chunk",
        ),
        "maximum_donor_rows_per_score_chunk": _positive_int(
            maximum_donor_rows_per_score_chunk,
            "maximum_donor_rows_per_score_chunk",
        ),
    }
    abi = {
        "schema": PROMPT_PHASE_ABI,
        "body_clock": "one-step-per-donor-semantic-token",
        "target_spans": "external-routing-metadata-only",
        "decode_boundary": "whole-donor-sequence-then-whole-target-retokenize",
        "undefined_donor_capacity_rows_scored": 0,
        "automatic_intelligence_gate": False,
    }
    return FoldPhaseKey(
        phase_name=PROMPT_PHASE_NAME,
        dependency_sha256s={
            "bridge_artifact_sha256": _digest(
                bridge_artifact_sha256, "bridge_artifact_sha256"
            ),
            "child_state_sha256": _digest(
                child_state_sha256, "child_state_sha256"
            ),
            "donor_tokenizer_semantics_sha256": _digest(
                donor_tokenizer_semantics_sha256,
                "donor_tokenizer_semantics_sha256",
            ),
            "prompt_sha256": canonical_sha256(prompt_value),
            "routing_contract_sha256": _digest(
                routing_contract_sha256, "routing_contract_sha256"
            ),
            "target_tokenizer_semantics_sha256": _digest(
                target_tokenizer_semantics_sha256,
                "target_tokenizer_semantics_sha256",
            ),
        },
        numeric_contract=numeric_contract,
        abi=abi,
    )


@dataclass(frozen=True, slots=True)
class CachedSpanPhase:
    span_evidence: Mapping[str, Any]
    adapter_receipt: Mapping[str, Any]
    phase_key: FoldPhaseKey
    completion_sha256: str
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class CachedPromptPhase:
    prompt_id: str
    response: Mapping[str, Any]
    runtime_receipt: Mapping[str, Any]
    phase_key: FoldPhaseKey
    completion_sha256: str
    cache_hit: bool


def _load_span_entry(
    entry: VerifiedFoldPhaseCacheEntry,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = _verified_sealed_json(
        entry.payload_path / "span-evidence.json",
        seal_field="evidence_sha256",
        expected_sha256=str(entry.result.get("span_evidence_sha256")),
        name="cached span evidence",
    )
    receipt = _verified_sealed_json(
        entry.payload_path / "span-adapter-receipt.json",
        seal_field="receipt_sha256",
        expected_sha256=str(entry.result.get("adapter_receipt_sha256")),
        name="cached span adapter receipt",
    )
    if receipt.get("span_evidence_sha256") != evidence.get("evidence_sha256"):
        raise Lfm2SemanticStepBridgeControlError(
            "Cached span receipt does not bind the cached evidence."
        )
    return evidence, receipt


def cache_or_compute_span_phase(
    *,
    cache: DynamicFoldPhaseCache,
    events: FoldPhaseEventLog,
    phase_key: FoldPhaseKey,
    compile_call: Callable[[Callable[[Mapping[str, Any]], None]], Any],
    telemetry_call: Callable[[], Mapping[str, Any]] | None = None,
    console_progress_call: Callable[[Mapping[str, Any]], None] | None = None,
) -> CachedSpanPhase:
    """Load or atomically publish global span evidence using injected work."""

    hit = cache.lookup(phase_key, events=events)
    if hit is not None:
        evidence, receipt = _load_span_entry(hit)
        return CachedSpanPhase(
            span_evidence=evidence,
            adapter_receipt=receipt,
            phase_key=phase_key,
            completion_sha256=hit.completion_sha256,
            cache_hit=True,
        )

    with cache.begin(phase_key, events=events) as transaction:
        def progress(record: Mapping[str, Any]) -> None:
            details = {"compiler_progress": _plain_json(record)}
            if telemetry_call is not None:
                details["telemetry"] = _plain_json(telemetry_call())
            transaction.progress(details)
            if console_progress_call is not None:
                console_progress_call(details)

        compiled = compile_call(progress)
        evidence_raw = getattr(compiled, "span_evidence", None)
        receipt_raw = getattr(compiled, "receipt", None)
        if not isinstance(evidence_raw, Mapping) or not isinstance(
            receipt_raw, Mapping
        ):
            if isinstance(compiled, Mapping):
                evidence_raw = compiled.get("span_evidence")
                receipt_raw = compiled.get("receipt")
        if not isinstance(evidence_raw, Mapping) or not isinstance(
            receipt_raw, Mapping
        ):
            raise Lfm2SemanticStepBridgeControlError(
                "Span compiler returned no evidence/receipt mappings."
            )
        evidence = _plain_json(evidence_raw)
        receipt = _plain_json(receipt_raw)
        evidence_sha = _digest(
            evidence.get("evidence_sha256"), "span evidence SHA-256"
        )
        receipt_sha = _digest(
            receipt.get("receipt_sha256"), "span adapter receipt SHA-256"
        )
        if receipt.get("span_evidence_sha256") != evidence_sha:
            raise Lfm2SemanticStepBridgeControlError(
                "Compiled span receipt does not bind its evidence."
            )
        _atomic_write_json(transaction.payload_path / "span-evidence.json", evidence)
        _atomic_write_json(
            transaction.payload_path / "span-adapter-receipt.json", receipt
        )
        entry = transaction.complete(
            result={
                "span_evidence_sha256": evidence_sha,
                "adapter_receipt_sha256": receipt_sha,
                "donor_row_count": evidence.get("donor_row_count"),
                "target_row_count": evidence.get("target_row_count"),
                "training_steps": 0,
                "parent_forward_calls": 0,
            }
        )
    verified_evidence, verified_receipt = _load_span_entry(entry)
    return CachedSpanPhase(
        span_evidence=verified_evidence,
        adapter_receipt=verified_receipt,
        phase_key=phase_key,
        completion_sha256=entry.completion_sha256,
        cache_hit=False,
    )


def _load_prompt_entry(
    entry: VerifiedFoldPhaseCacheEntry,
    *,
    expected_prompt_id: str,
) -> dict[str, Any]:
    runtime = _verified_sealed_json(
        entry.payload_path / "runtime.json",
        seal_field="runtime_sha256",
        expected_sha256=str(entry.result.get("runtime_sha256")),
        name="cached semantic-step runtime",
    )
    responses = runtime.get("responses")
    if (
        runtime.get("prompt_count") != 1
        or not isinstance(responses, list)
        or len(responses) != 1
        or responses[0].get("id") != expected_prompt_id
    ):
        raise Lfm2SemanticStepBridgeControlError(
            "Cached prompt phase does not contain exactly its expected response."
        )
    return runtime


def cache_or_compute_prompt_phase(
    *,
    cache: DynamicFoldPhaseCache,
    events: FoldPhaseEventLog,
    phase_key: FoldPhaseKey,
    prompt_id: str,
    review_call: Callable[[Callable[[Mapping[str, Any]], None]], Mapping[str, Any]],
    telemetry_call: Callable[[], Mapping[str, Any]] | None = None,
    console_progress_call: Callable[[Mapping[str, Any]], None] | None = None,
) -> CachedPromptPhase:
    """Load or atomically publish one fully decoded review response."""

    hit = cache.lookup(phase_key, events=events)
    if hit is not None:
        runtime = _load_prompt_entry(hit, expected_prompt_id=prompt_id)
        return CachedPromptPhase(
            prompt_id=prompt_id,
            response=runtime["responses"][0],
            runtime_receipt=runtime,
            phase_key=phase_key,
            completion_sha256=hit.completion_sha256,
            cache_hit=True,
        )

    with cache.begin(phase_key, events=events) as transaction:
        def progress(record: Mapping[str, Any]) -> None:
            details = {"runtime_progress": _plain_json(record)}
            if telemetry_call is not None:
                details["telemetry"] = _plain_json(telemetry_call())
            transaction.progress(details)
            if console_progress_call is not None:
                console_progress_call(details)

        runtime = _plain_json(review_call(progress))
        runtime_sha = _digest(
            runtime.get("runtime_sha256"), "semantic-step runtime SHA-256"
        )
        runtime_body = dict(runtime)
        runtime_body.pop("runtime_sha256")
        if runtime_sha != canonical_sha256(runtime_body):
            raise Lfm2SemanticStepBridgeControlError(
                "Semantic-step runtime result self-hash does not verify."
            )
        responses = runtime.get("responses")
        if (
            runtime.get("prompt_count") != 1
            or not isinstance(responses, list)
            or len(responses) != 1
            or responses[0].get("id") != prompt_id
        ):
            raise Lfm2SemanticStepBridgeControlError(
                "Semantic-step runtime did not return exactly the requested prompt."
            )
        _atomic_write_json(transaction.payload_path / "runtime.json", runtime)
        entry = transaction.complete(
            result={
                "runtime_sha256": runtime_sha,
                "prompt_id": prompt_id,
                "donor_response_token_count": len(
                    responses[0].get("raw_donor_response_token_ids", [])
                ),
                "training_steps": 0,
                "parent_forward_calls": 0,
            }
        )
    verified = _load_prompt_entry(entry, expected_prompt_id=prompt_id)
    return CachedPromptPhase(
        prompt_id=prompt_id,
        response=verified["responses"][0],
        runtime_receipt=verified,
        phase_key=phase_key,
        completion_sha256=entry.completion_sha256,
        cache_hit=False,
    )


def load_bound_parent_envelope(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_envelope_sha256: str,
    expected_prompt_set_sha256: str,
    expected_prompt_ids: Sequence[str],
    expected_parent_identity_sha256: str,
    expected_parent_unique_parameters: int,
    expected_maximum_new_tokens: int,
) -> dict[str, Any]:
    """Authenticate the cached parent text without loading the parent model."""

    source = _local_file(path, "parent envelope")
    if _file_sha256(source) != _digest(
        expected_file_sha256, "expected parent envelope file SHA-256"
    ):
        raise Lfm2SemanticStepBridgeControlError("Parent envelope bytes changed.")
    envelope = _verified_sealed_json(
        source,
        seal_field="envelope_sha256",
        expected_sha256=expected_envelope_sha256,
        name="parent envelope",
    )
    responses = envelope.get("responses")
    if (
        envelope.get("schema")
        != "neural-forge-lfm2-whole-response-parent-envelope-v1"
        or envelope.get("status") != "complete-one-time-parent-response-cache"
        or envelope.get("prompt_set_sha256") != expected_prompt_set_sha256
        or envelope.get("prompt_count") != len(expected_prompt_ids)
        or not isinstance(responses, list)
        or [str(row.get("id")) for row in responses] != list(expected_prompt_ids)
        or envelope.get("response_bank_sha256") != canonical_sha256(responses)
        or envelope.get("parent_identity_sha256")
        != _digest(
            expected_parent_identity_sha256,
            "expected_parent_identity_sha256",
        )
        or envelope.get("parent_unique_parameters")
        != _positive_int(
            expected_parent_unique_parameters,
            "expected_parent_unique_parameters",
        )
        or envelope.get("maximum_new_tokens")
        != _positive_int(expected_maximum_new_tokens, "maximum_new_tokens")
    ):
        raise Lfm2SemanticStepBridgeControlError(
            "Parent envelope differs from the bound prompts, parent, or decode limit."
        )
    return envelope


def assemble_manual_review_dossier(
    *,
    prompts: Sequence[Mapping[str, Any]],
    prompt_phases: Sequence[CachedPromptPhase],
    parent_envelope: Mapping[str, Any],
    child_state_sha256: str,
    child_unique_parameters: int,
    parent_unique_parameters: int,
    span_phase: CachedSpanPhase,
    routing_contract_sha256: str,
    maximum_new_tokens: int,
) -> dict[str, Any]:
    """Build the no-gate side-by-side artifact from already-computed phases."""

    prompt_rows = [_plain_json(row) for row in prompts]
    if len(prompt_rows) != len(prompt_phases):
        raise Lfm2SemanticStepBridgeControlError(
            "Prompt phase count differs from prompt inventory."
        )
    parent_rows = parent_envelope.get("responses")
    if not isinstance(parent_rows, list) or len(parent_rows) != len(prompt_rows):
        raise Lfm2SemanticStepBridgeControlError(
            "Parent envelope response inventory is incomplete."
        )
    comparisons = []
    for prompt, phase, parent in zip(
        prompt_rows, prompt_phases, parent_rows, strict=True
    ):
        prompt_id = str(prompt["id"])
        if phase.prompt_id != prompt_id or str(parent.get("id")) != prompt_id:
            raise Lfm2SemanticStepBridgeControlError(
                "Prompt, child, and parent response order differs."
            )
        comparisons.append(
            {
                "id": prompt_id,
                "category": prompt.get("category"),
                "prompt": prompt,
                "parent_cached_response": _plain_json(parent),
                "child_semantic_step_response": _plain_json(phase.response),
                "child_runtime_sha256": phase.runtime_receipt["runtime_sha256"],
                "prompt_phase_key_sha256": phase.phase_key.phase_key_sha256,
                "prompt_phase_completion_sha256": phase.completion_sha256,
                "manual_review_instruction": (
                    "Judge coherence, correctness, instruction following, and useful "
                    "intelligence from the two complete decoded responses; wording "
                    "and exact-token differences are not failures by themselves."
                ),
            }
        )
    parent_count = _positive_int(parent_unique_parameters, "parent_unique_parameters")
    child_count = _positive_int(child_unique_parameters, "child_unique_parameters")
    if child_count >= parent_count:
        raise Lfm2SemanticStepBridgeControlError(
            "The reviewed child is not parameter-compacted relative to the parent."
        )
    body = {
        "schema": DOSSIER_SCHEMA,
        "status": "complete-awaiting-manual-whole-response-intelligence-review",
        "comparisons": comparisons,
        "prompt_count": len(comparisons),
        "prompt_set_sha256": canonical_sha256(prompt_rows),
        "parent_envelope_sha256": parent_envelope["envelope_sha256"],
        "parent_unique_parameters": parent_count,
        "child_unique_parameters": child_count,
        "realized_parameter_compaction_ratio": parent_count / child_count,
        "child_state_sha256": _digest(
            child_state_sha256, "child_state_sha256"
        ),
        "span_evidence_sha256": span_phase.span_evidence["evidence_sha256"],
        "span_phase_key_sha256": span_phase.phase_key.phase_key_sha256,
        "span_phase_completion_sha256": span_phase.completion_sha256,
        "routing_contract_sha256": _digest(
            routing_contract_sha256, "routing_contract_sha256"
        ),
        "maximum_new_tokens": _positive_int(
            maximum_new_tokens, "maximum_new_tokens"
        ),
        "authoritative_child_text_source": "whole-donor-sequence-decode",
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "numeric_or_binary_behavior_gate_used": False,
        "exact_token_match_gate_used": False,
        "automatic_acceptance_or_rejection_performed": False,
    }
    return {**body, "dossier_sha256": canonical_sha256(body)}


def _sum_runtime_operation_audits(
    prompt_phases: Sequence[CachedPromptPhase],
) -> dict[str, Any]:
    integer_totals: dict[str, int] = {}
    booleans: dict[str, bool] = {}
    for phase in prompt_phases:
        audit = phase.runtime_receipt.get("operation_audit")
        if not isinstance(audit, Mapping):
            raise Lfm2SemanticStepBridgeControlError(
                "Prompt runtime operation audit is missing."
            )
        for name, value in audit.items():
            if isinstance(value, bool):
                booleans[str(name)] = booleans.get(str(name), False) or value
            elif isinstance(value, int):
                integer_totals[str(name)] = integer_totals.get(str(name), 0) + value
    return {**dict(sorted(integer_totals.items())), **dict(sorted(booleans.items()))}


def _event_log(
    events_directory: Path,
    *,
    label: str,
    phase_key: FoldPhaseKey,
) -> FoldPhaseEventLog:
    suffix = phase_key.phase_key_sha256[:20]
    return FoldPhaseEventLog(
        events_directory / f"{label}-{suffix}.jsonl",
        run_id=f"{label}-{suffix}",
    )


def run(
    *,
    run_directory: Path,
    phase_cache_directory: Path | None,
    checkpoint_directory: Path,
    donor_tokenizer_directory: Path,
    target_tokenizer_directory: Path,
    bridge_directory: Path,
    expected_bridge_artifact_sha256: str,
    expected_child_state_sha256: str | None,
    expected_child_unique_parameters: int,
    expected_parent_identity_sha256: str,
    expected_parent_unique_parameters: int,
    prompts_json: Path,
    expected_prompts_file_sha256: str,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    device: str,
    dtype: str,
    positional_design: str,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
    maximum_span_rows_per_chunk: int,
    maximum_embedding_rows_per_chunk: int,
    maximum_donor_rows_per_score_chunk: int,
    console_progress_every: int = 16,
    checkpoint_provenance_device: str = "cpu",
    checkpoint_provenance_dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Execute or resume the local semantic-clock control."""

    run_root = _output_directory(run_directory, "run_directory")
    cache_root = _output_directory(
        run_root / "phase-cache"
        if phase_cache_directory is None
        else phase_cache_directory,
        "phase_cache_directory",
    )
    events_root = _output_directory(run_root / "events", "events directory")
    checkpoint = _local_directory(checkpoint_directory, "child checkpoint")
    donor_path = _local_directory(
        donor_tokenizer_directory, "donor tokenizer directory"
    )
    target_path = _local_directory(
        target_tokenizer_directory, "target tokenizer directory"
    )
    bridge_path = _local_directory(bridge_directory, "bridge directory")
    prompt_path = _local_file(prompts_json, "review prompts")
    if _file_sha256(prompt_path) != _digest(
        expected_prompts_file_sha256, "expected review prompt file SHA-256"
    ):
        raise Lfm2SemanticStepBridgeControlError("Review prompt bytes changed.")

    prompt_rows = tuple(dict(row) for row in load_review_prompts(prompt_path))
    prompt_set_sha256 = canonical_sha256([_plain_json(row) for row in prompt_rows])
    envelope = load_bound_parent_envelope(
        parent_envelope,
        expected_file_sha256=expected_parent_envelope_file_sha256,
        expected_envelope_sha256=expected_parent_envelope_sha256,
        expected_prompt_set_sha256=prompt_set_sha256,
        expected_prompt_ids=[str(row["id"]) for row in prompt_rows],
        expected_parent_identity_sha256=expected_parent_identity_sha256,
        expected_parent_unique_parameters=expected_parent_unique_parameters,
        expected_maximum_new_tokens=maximum_new_tokens,
    )

    requested_dtype = _DTYPE_ALIASES.get(str(dtype).strip().lower())
    model_dtype, dtype_receipt = resolve_runtime_dtype(dtype, device)
    resolved_dtype = str(dtype_receipt["resolved_dtype"])
    provenance_dtype = resolve_checkpoint_provenance_dtype(
        checkpoint_provenance_dtype
    )
    donor_tokenizer = load_lfm2_direct_local_tokenizer(donor_path)
    target_tokenizer = load_lfm2_direct_local_tokenizer(target_path)
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_path,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=_digest(
            expected_bridge_artifact_sha256,
            "expected_bridge_artifact_sha256",
        ),
    )
    if (
        fingerprint_hf_tokenizer_semantics(donor_tokenizer)
        != bridge.donor_tokenizer_semantics_sha256
        or fingerprint_hf_tokenizer_semantics(target_tokenizer)
        != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2SemanticStepBridgeControlError(
            "Live local tokenizer semantics differ from the bridge."
        )

    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=torch.device(checkpoint_provenance_device),
        dtype=provenance_dtype,
    )
    child, portable_load = checkpoint_io.load_authenticated_portable(
        checkpoint,
        runtime_device=torch.device(device),
        runtime_dtype=model_dtype,
    )
    portable_load_receipt = _plain_json(portable_load.as_dict())
    child.requires_grad_(False)
    child.eval()
    child_parameters = _unique_parameter_count(child)
    expected_child_parameters = _positive_int(
        expected_child_unique_parameters, "expected_child_unique_parameters"
    )
    if child_parameters != expected_child_parameters:
        raise Lfm2SemanticStepBridgeControlError(
            "Loaded child parameter count differs from its expected inventory."
        )
    checkpoint_state_sha256 = str(
        portable_load_receipt["staged_state_sha256"]
    )
    child_state_sha256 = str(portable_load_receipt["runtime_state_sha256"])
    if (
        expected_child_state_sha256 is not None
        and checkpoint_state_sha256
        != _digest(expected_child_state_sha256, "expected_child_state_sha256")
    ):
        raise Lfm2SemanticStepBridgeControlError(
            "Loaded child state differs from its expected digest."
        )

    cache = DynamicFoldPhaseCache(cache_root)
    console = ConsoleProgressReporter(
        semantic_step_every=console_progress_every,
    )
    span_contract = lfm2_semantic_step_span_adapter_cache_contract(bridge)
    span_key = span_phase_key_from_contract(span_contract)
    span_events = _event_log(
        events_root,
        label="span",
        phase_key=span_key,
    )
    span_started = time.monotonic()
    span_phase = cache_or_compute_span_phase(
        cache=cache,
        events=span_events,
        phase_key=span_key,
        compile_call=lambda progress: (
            compile_lfm2_semantic_step_span_evidence_from_local_tokenizers(
                bridge,
                donor_tokenizer,
                target_tokenizer,
                maximum_donor_rows_per_chunk=_positive_int(
                    maximum_span_rows_per_chunk,
                    "maximum_span_rows_per_chunk",
                ),
                expected_bridge_artifact_sha256=bridge.artifact_sha256,
                expected_bridge_profile_sha256=bridge.profile_fingerprint,
                expected_donor_tokenizer_semantics_sha256=(
                    bridge.donor_tokenizer_semantics_sha256
                ),
                expected_target_tokenizer_semantics_sha256=(
                    bridge.target_tokenizer_semantics_sha256
                ),
                progress_callback=progress,
            )
        ),
        telemetry_call=lambda: runtime_telemetry(
            started_at=span_started,
            device="cpu",
            requested_dtype="integer-exact",
            resolved_dtype="integer-exact",
        ),
        console_progress_call=console.span,
    )

    live_table = child.get_input_embeddings().weight
    routing = bind_lfm2_semantic_step_hard_anchor_runtime(
        span_phase.span_evidence,
        live_table,
        positional_design=positional_design,
    )
    routing_receipt = _plain_json(routing.as_dict())
    routing_contract_sha256 = canonical_sha256(routing_receipt)

    prompt_phases: list[CachedPromptPhase] = []
    for prompt in prompt_rows:
        prompt_key = build_prompt_phase_key(
            child_state_sha256=child_state_sha256,
            routing_contract_sha256=routing_contract_sha256,
            bridge_artifact_sha256=bridge.artifact_sha256,
            donor_tokenizer_semantics_sha256=(
                bridge.donor_tokenizer_semantics_sha256
            ),
            target_tokenizer_semantics_sha256=(
                bridge.target_tokenizer_semantics_sha256
            ),
            prompt=prompt,
            device=device,
            dtype_receipt=dtype_receipt,
            maximum_prompt_tokens=maximum_prompt_tokens,
            maximum_new_tokens=maximum_new_tokens,
            maximum_embedding_rows_per_chunk=maximum_embedding_rows_per_chunk,
            maximum_donor_rows_per_score_chunk=(
                maximum_donor_rows_per_score_chunk
            ),
        )
        prompt_id = str(prompt["id"])
        prompt_label = "prompt-" + hashlib.sha256(
            prompt_id.encode("utf-8")
        ).hexdigest()[:12]
        prompt_events = _event_log(
            events_root,
            label=prompt_label,
            phase_key=prompt_key,
        )
        prompt_started = time.monotonic()
        phase = cache_or_compute_prompt_phase(
            cache=cache,
            events=prompt_events,
            phase_key=prompt_key,
            prompt_id=prompt_id,
            review_call=lambda progress, row=prompt: (
                run_lfm2_semantic_step_model_review(
                    child,
                    routing,
                    donor_tokenizer,
                    target_tokenizer,
                    [row],
                    maximum_prompt_tokens=_positive_int(
                        maximum_prompt_tokens, "maximum_prompt_tokens"
                    ),
                    maximum_new_tokens=_positive_int(
                        maximum_new_tokens, "maximum_new_tokens"
                    ),
                    maximum_embedding_rows_per_chunk=_positive_int(
                        maximum_embedding_rows_per_chunk,
                        "maximum_embedding_rows_per_chunk",
                    ),
                    maximum_donor_rows_per_score_chunk=_positive_int(
                        maximum_donor_rows_per_score_chunk,
                        "maximum_donor_rows_per_score_chunk",
                    ),
                    allow_legacy_per_operation_authentication=False,
                    progress_callback=progress,
                )
            ),
            telemetry_call=lambda started=prompt_started: runtime_telemetry(
                started_at=started,
                device=device,
                requested_dtype=str(requested_dtype),
                resolved_dtype=resolved_dtype,
            ),
            console_progress_call=console.runtime,
        )
        prompt_phases.append(phase)

    final_child_state_sha256 = fingerprint_lfm2_direct_local_state(child)
    if final_child_state_sha256 != child_state_sha256:
        raise Lfm2SemanticStepBridgeControlError(
            "Child state changed while producing the review dossier."
        )
    if any(parameter.grad is not None for parameter in child.parameters()):
        raise Lfm2SemanticStepBridgeControlError(
            "Child accumulated gradients during fold-only review."
        )

    dossier = assemble_manual_review_dossier(
        prompts=prompt_rows,
        prompt_phases=prompt_phases,
        parent_envelope=envelope,
        child_state_sha256=child_state_sha256,
        child_unique_parameters=child_parameters,
        parent_unique_parameters=expected_parent_unique_parameters,
        span_phase=span_phase,
        routing_contract_sha256=routing_contract_sha256,
        maximum_new_tokens=maximum_new_tokens,
    )
    dossier_path = run_root / "whole-response-dossier.json"
    _atomic_write_json(dossier_path, dossier)

    artifact_audit = _sum_runtime_operation_audits(prompt_phases)
    prompt_hits = sum(int(phase.cache_hit) for phase in prompt_phases)
    missed_phases = [phase for phase in prompt_phases if not phase.cache_hit]
    invocation_audit = {
        "span_phase_cache_hits": int(span_phase.cache_hit),
        "span_phase_computations": int(not span_phase.cache_hit),
        "prompt_phase_cache_hits": prompt_hits,
        "prompt_phase_computations": len(prompt_phases) - prompt_hits,
        "child_body_forward_calls": sum(
            int(
                phase.runtime_receipt["operation_audit"].get(
                    "child_body_forward_calls", 0
                )
            )
            for phase in missed_phases
        ),
        "parent_model_load_calls": 0,
        "parent_forward_calls": 0,
        "parent_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "child_generate_calls": 0,
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "parameter_fold_or_copy_calls": 0,
        "least_squares_or_ridge_solve_calls": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
        "network_access": False,
    }
    parent_count = _positive_int(
        expected_parent_unique_parameters, "expected_parent_unique_parameters"
    )
    body = {
        "schema": RUN_SCHEMA,
        "status": "complete-same-body-hard-anchor-donor-semantic-clock-review",
        "run_directory": str(run_root),
        "phase_cache_directory": str(cache_root),
        "checkpoint_directory": str(checkpoint),
        "direct_local_checkpoint_io_fingerprint": (
            checkpoint_io.resource_fingerprint
        ),
        "direct_local_checkpoint_provenance_device": str(
            torch.device(checkpoint_provenance_device)
        ),
        "direct_local_checkpoint_provenance_dtype": str(provenance_dtype),
        "direct_local_portable_load_receipt": portable_load_receipt,
        "checkpoint_state_sha256": checkpoint_state_sha256,
        "runtime_initial_child_state_sha256": child_state_sha256,
        "final_runtime_child_state_sha256": final_child_state_sha256,
        "runtime_child_state_unchanged": True,
        "parent_identity_sha256": _digest(
            expected_parent_identity_sha256, "expected_parent_identity_sha256"
        ),
        "parent_unique_parameters": parent_count,
        "child_unique_parameters": child_parameters,
        "realized_parameter_compaction_ratio": parent_count / child_parameters,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "span_evidence_sha256": span_phase.span_evidence["evidence_sha256"],
        "span_phase_key_sha256": span_phase.phase_key.phase_key_sha256,
        "span_phase_completion_sha256": span_phase.completion_sha256,
        "routing_contract_sha256": routing_contract_sha256,
        "routing_receipt": routing_receipt,
        "prompt_phase_keys": [
            {
                "id": phase.prompt_id,
                "phase_key_sha256": phase.phase_key.phase_key_sha256,
                "completion_sha256": phase.completion_sha256,
                "runtime_sha256": phase.runtime_receipt["runtime_sha256"],
            }
            for phase in prompt_phases
        ],
        "prompt_count": len(prompt_phases),
        "prompt_set_sha256": prompt_set_sha256,
        "prompts_file_sha256": _file_sha256(prompt_path),
        "parent_envelope_sha256": envelope["envelope_sha256"],
        "parent_envelope_file_sha256": _file_sha256(
            _local_file(parent_envelope, "parent envelope")
        ),
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "positional_design": positional_design,
        "numeric_execution": dtype_receipt,
        "console_progress_every_semantic_steps": _positive_int(
            console_progress_every, "console_progress_every"
        ),
        "console_progress_frequency_affects_phase_cache_identity": False,
        "whole_response_dossier": str(dossier_path),
        "whole_response_dossier_sha256": dossier["dossier_sha256"],
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "numeric_or_binary_behavior_gate_used": False,
        "exact_token_match_gate_used": False,
        "parent_model_or_trace_required_at_review_runtime": False,
        "artifact_operation_audit": artifact_audit,
        "this_invocation_operation_audit": invocation_audit,
    }
    result = {**body, "result_sha256": canonical_sha256(body)}
    _atomic_write_json(run_root / "run-receipt.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--phase-cache-directory", type=Path)
    parser.add_argument("--checkpoint-directory", required=True, type=Path)
    parser.add_argument("--donor-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--target-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--expected-child-state-sha256")
    parser.add_argument(
        "--expected-child-unique-parameters", required=True, type=int
    )
    parser.add_argument("--expected-parent-identity-sha256", required=True)
    parser.add_argument(
        "--expected-parent-unique-parameters", required=True, type=int
    )
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--parent-envelope", required=True, type=Path)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument(
        "--expected-parent-envelope-file-sha256", required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--checkpoint-provenance-device", default="cpu")
    parser.add_argument(
        "--checkpoint-provenance-dtype",
        default="bfloat16",
        help=(
            "Exact dtype in the legacy v1 save-time I/O fingerprint; this is "
            "authenticated before the separate runtime dtype is applied."
        ),
    )
    parser.add_argument("--positional-design", default="uniform")
    parser.add_argument("--maximum-prompt-tokens", type=int, default=1024)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument("--maximum-span-rows-per-chunk", type=int, default=1024)
    parser.add_argument(
        "--maximum-embedding-rows-per-chunk", type=int, default=1024
    )
    parser.add_argument(
        "--maximum-donor-rows-per-score-chunk", type=int, default=4096
    )
    parser.add_argument("--console-progress-every", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(
            run_directory=args.run_directory,
            phase_cache_directory=args.phase_cache_directory,
            checkpoint_directory=args.checkpoint_directory,
            donor_tokenizer_directory=args.donor_tokenizer_directory,
            target_tokenizer_directory=args.target_tokenizer_directory,
            bridge_directory=args.bridge_directory,
            expected_bridge_artifact_sha256=(
                args.expected_bridge_artifact_sha256
            ),
            expected_child_state_sha256=args.expected_child_state_sha256,
            expected_child_unique_parameters=(
                args.expected_child_unique_parameters
            ),
            expected_parent_identity_sha256=(
                args.expected_parent_identity_sha256
            ),
            expected_parent_unique_parameters=(
                args.expected_parent_unique_parameters
            ),
            prompts_json=args.prompts_json,
            expected_prompts_file_sha256=args.expected_prompts_file_sha256,
            parent_envelope=args.parent_envelope,
            expected_parent_envelope_sha256=(
                args.expected_parent_envelope_sha256
            ),
            expected_parent_envelope_file_sha256=(
                args.expected_parent_envelope_file_sha256
            ),
            device=args.device,
            dtype=args.dtype,
            checkpoint_provenance_device=(
                args.checkpoint_provenance_device
            ),
            checkpoint_provenance_dtype=args.checkpoint_provenance_dtype,
            positional_design=args.positional_design,
            maximum_prompt_tokens=args.maximum_prompt_tokens,
            maximum_new_tokens=args.maximum_new_tokens,
            maximum_span_rows_per_chunk=args.maximum_span_rows_per_chunk,
            maximum_embedding_rows_per_chunk=(
                args.maximum_embedding_rows_per_chunk
            ),
            maximum_donor_rows_per_score_chunk=(
                args.maximum_donor_rows_per_score_chunk
            ),
            console_progress_every=args.console_progress_every,
        )
    except Exception as exc:
        run_root = args.run_directory.expanduser().absolute()
        if run_root.is_dir() and not run_root.is_symlink():
            try:
                _record_failure(run_root, exc)
            except Exception as recording_exc:
                print("failure-recording-error:", file=sys.stderr)
                traceback.print_exception(
                    type(recording_exc),
                    recording_exc,
                    recording_exc.__traceback__,
                    file=sys.stderr,
                )
        print("run-error:", file=sys.stderr)
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_directory": result["run_directory"],
                "child_unique_parameters": result["child_unique_parameters"],
                "realized_parameter_compaction_ratio": result[
                    "realized_parameter_compaction_ratio"
                ],
                "whole_response_dossier": result["whole_response_dossier"],
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
