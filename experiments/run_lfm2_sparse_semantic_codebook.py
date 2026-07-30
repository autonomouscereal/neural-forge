"""Compile and review the LFM2 dual-channel sparse semantic token bridge.

The emission channel ``S`` is the already-authenticated exact raw-token span
bridge.  The semantic channel ``C`` is a sparse, continuous code over the
existing target table and is used for both donor embeddings and donor logits.
The parent is never instantiated or executed: this runner reads only the tied
embedding rows from local safetensors and consumes an optional sealed trace
cache for an advisory hidden-state diagnostic.

Expensive work is split into content-addressed phases.  Source-table and
top-cosine candidate phases deliberately exclude ``k`` and ridge.  Residual
OMP inventories bind their completed depth, while every cumulative row-prefix
checkpoint excludes the final requested ``k`` so larger supports resume the
same math.  Coefficient phases bind their exact solve controls.  Each whole-
response prompt remains an independent resumable phase, and manual whole-
response review is authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import fingerprint_model_state  # noqa: E402
from neural_forge.dynamic_fold_phase_cache import (  # noqa: E402
    DynamicFoldPhaseCache,
    FoldPhaseEventLog,
    FoldPhaseKey,
    VerifiedFoldPhaseCacheEntry,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCache,
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
    bind_lfm2_sparse_semantic_code_runtime,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)
from neural_forge.lfm2_decoded_review import load_review_prompts  # noqa: E402

from experiments.run_lfm2_semantic_step_bridge_control import (  # noqa: E402
    CachedPromptPhase,
    ConsoleProgressReporter,
    _checkpoint_artifact_inventory,
    _event_log,
    _record_failure,
    _sum_runtime_operation_audits,
    _unique_parameter_count,
    assemble_manual_review_dossier,
    build_prompt_phase_key,
    cache_or_compute_prompt_phase,
    cache_or_compute_span_phase,
    load_bound_parent_envelope,
    resolve_checkpoint_provenance_dtype,
    resolve_runtime_dtype,
    runtime_telemetry,
    span_phase_key_from_contract,
)


RUN_SCHEMA = "neural-forge-lfm2-sparse-semantic-codebook-run-v1"
PARENT_TABLE_LOAD_SCHEMA = "neural-forge-lfm2-selective-parent-tied-table-load-v1"
TENSOR_PHASE_RECEIPT_SCHEMA = "neural-forge-sparse-semantic-tensor-phase-v1"
HIDDEN_DIAGNOSTIC_SCHEMA = "neural-forge-lfm2-sparse-semantic-hidden-rank-diagnostic-v2"
PARENT_TABLE_PHASE = "lfm2-sparse-semantic-parent-table"
NEIGHBOUR_PHASE = "lfm2-sparse-semantic-neighbour-chunk"
RESIDUAL_OMP_PREFIX_PHASE = "lfm2-residual-omp-semantic-support-prefix"
COEFFICIENT_PHASE = "lfm2-sparse-semantic-coefficient-chunk"
DIAGNOSTIC_PHASE = "lfm2-sparse-semantic-hidden-diagnostic"
CANDIDATE_INVENTORY_PHASE = "lfm2-sparse-semantic-candidate-inventory"
RESIDUAL_OMP_CANDIDATE_INVENTORY_PHASE = (
    "lfm2-residual-omp-semantic-candidate-inventory"
)
CODEBOOK_PHASE = "lfm2-sparse-semantic-codebook"
NORM_RECLOSURE_PHASE = "lfm2-sparse-semantic-norm-reclosure"
PARENT_TIED_TENSOR_KEY = "model.embed_tokens.weight"
_SHA256_ZERO = "0" * 64
_TOP_COSINE_SUPPORT_SELECTION = "top-cosine"
_RESIDUAL_OMP_SUPPORT_SELECTION = "residual-omp"
_SUPPORT_SELECTION_CHOICES = (
    _TOP_COSINE_SUPPORT_SELECTION,
    _RESIDUAL_OMP_SUPPORT_SELECTION,
)
_TOP_COSINE_CANDIDATE_REQUEST_SCHEMA = (
    "neural-forge-lfm2-sparse-semantic-candidate-request-v1"
)
_RESIDUAL_OMP_CANDIDATE_REQUEST_SCHEMA = (
    "neural-forge-lfm2-residual-omp-semantic-candidate-request-v1"
)
_RESIDUAL_OMP_CANDIDATE_CHUNK_SCHEMA = (
    "neural-forge-lfm2-residual-omp-semantic-candidate-chunk-v1"
)
COMPILE_CONSOLE_PROGRESS_SCHEMA = (
    "neural-forge-lfm2-sparse-semantic-compile-progress-v1"
)
PHASE_MATH_ABI_SCHEMA = "neural-forge-lfm2-sparse-semantic-phase-math-abi-v1"
# Frozen from the authenticated A4 residual-OMP implementation receipt.  It is
# compatibility input only, never recomputed from current source bytes.
_LEGACY_CORE_MATH_CHECKPOINT_COMPATIBILITY_SHA256 = (
    "e3cef33787d86368c07de226a92264ec3b248b73a8087618cfc5388289600d80"
)


def sparse_semantic_phase_math_abi_receipts() -> dict[str, dict[str, Any]]:
    """Return intentional math identities that ignore orchestration-only edits.

    These descriptors are the cache ABI.  A folding-math change must update the
    corresponding version or descriptor.  Source-file hashes remain useful
    observational provenance, but are deliberately excluded from expensive
    candidate, coefficient, and reclosure cache identities.
    """

    descriptors = {
        "top_cosine_candidate_selection": {
            "version": 1,
            "algorithm": "normalized-full-anchor-top-cosine",
            "normalization": "safe-row-l2",
            "ranking": "descending-cosine-then-target-id",
            "direct_anchor_policy": "exact-one-hot",
        },
        "residual_omp_support_selection": {
            "version": 1,
            "algorithm": "full-anchor-residual-omp-double-mgs",
            "normalization": "safe-row-l2",
            "dictionary_search": "full-target-table-each-new-depth",
            "orthogonalization": "deterministic-double-modified-gram-schmidt",
            "dependent_atom_policy": "retry-next-ranked-target-id",
            "prefix_payload": "integer-support-only",
            "direct_anchor_policy": "exact-one-hot",
            "checkpoint_compatibility_sha256": (
                _LEGACY_CORE_MATH_CHECKPOINT_COMPATIBILITY_SHA256
            ),
        },
        "coefficient_ridge_refit": {
            "version": 1,
            "algorithm": "active-k-local-two-stage-ridge-refit",
            "candidate_order": "authenticated-candidate-prefix",
            "direct_anchor_policy": "exact-one-hot",
            "publication_dtype": "float32",
        },
        "coefficient_norm_reclosure": {
            "version": 1,
            "algorithm": "coefficient-only-norm-ratio-interpolation",
            "support_mutation": False,
            "direct_anchor_policy": "bit-exact-preservation",
            "publication_dtype": "float32",
        },
    }
    receipts: dict[str, dict[str, Any]] = {}
    for phase, descriptor in descriptors.items():
        body = {
            "schema": PHASE_MATH_ABI_SCHEMA,
            "phase": phase,
            "descriptor": descriptor,
        }
        receipts[phase] = {**body, "math_abi_sha256": canonical_sha256(body)}
    return receipts


class CompileProgressReporter:
    """Emit every compile/cache event as one flush-immediate JSON line."""

    def __init__(self, *, stream: Any = None) -> None:
        self.stream = stream

    def __call__(self, details: Mapping[str, Any]) -> None:
        body = {
            "schema": COMPILE_CONSOLE_PROGRESS_SCHEMA,
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


def _console_phase_event(
    callback: Callable[[Mapping[str, Any]], None] | None,
    *,
    phase_key: FoldPhaseKey,
    stage: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    if callback is None:
        return
    callback(
        {
            "phase": phase_key.phase_name,
            "phase_key_sha256": phase_key.phase_key_sha256,
            "stage": stage,
            "details": {} if details is None else _plain_json(details),
        }
    )


class Lfm2SparseSemanticRunnerError(RuntimeError):
    """The sparse semantic production run violated its sealed contract."""


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise Lfm2SparseSemanticRunnerError(
        f"Evidence contains non-JSON value {type(value).__name__}."
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2SparseSemanticRunnerError(f"{name} must be positive.")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2SparseSemanticRunnerError(f"{name} must be nonnegative.")
    return int(value)


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2SparseSemanticRunnerError(
            f"{name} must be finite and nonnegative."
        ) from exc
    if not result >= 0.0 or not torch.isfinite(torch.tensor(result)).item():
        raise Lfm2SparseSemanticRunnerError(f"{name} must be finite and nonnegative.")
    return result


def _finite_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2SparseSemanticRunnerError(f"{name} must be finite.") from exc
    if not torch.isfinite(torch.tensor(result)).item():
        raise Lfm2SparseSemanticRunnerError(f"{name} must be finite.")
    return result


def _finite_positive(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result <= 0.0:
        raise Lfm2SparseSemanticRunnerError(f"{name} must be positive.")
    return result


def _support_selection(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _SUPPORT_SELECTION_CHOICES:
        choices = ", ".join(_SUPPORT_SELECTION_CHOICES)
        raise Lfm2SparseSemanticRunnerError(
            f"support_selection must be one of: {choices}."
        )
    return normalized


def _amplification_ceiling(value: Any, name: str) -> float:
    result = _finite_positive(value, name)
    if result < 1.0:
        raise Lfm2SparseSemanticRunnerError(f"{name} must be at least one.")
    return result


def _optional_cosine(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2SparseSemanticRunnerError(
            f"{name} must be a finite cosine or None."
        ) from exc
    if not torch.isfinite(torch.tensor(result)).item() or result < -1.0 or result > 1.0:
        raise Lfm2SparseSemanticRunnerError(
            f"{name} must be between negative one and one."
        )
    return result


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SparseSemanticRunnerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256(b"neural-forge-sparse-semantic-tensor-v1\0")
    digest.update(str(tensor.dtype).removeprefix("torch.").encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _core_sparse_tensor_sha256(value: torch.Tensor) -> str:
    """Match the sparse-codebook core's stable table/tensor hash domain."""

    tensor = value.detach()
    digest = hashlib.sha256(b"neural-forge-lfm2-sparse-semantic-tensor-v1\0")
    digest.update(str(tensor.dtype).removeprefix("torch.").encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    if tensor.ndim == 0:
        blocks = (tensor.reshape(1),)
    else:
        blocks = (
            tensor[start : min(start + 4096, int(tensor.shape[0]))]
            for start in range(0, int(tensor.shape[0]), 4096)
        )
    for block in blocks:
        digest.update(
            block.to(device="cpu")
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def _local_directory(path: Path, name: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise Lfm2SparseSemanticRunnerError(f"{name} cannot be a link.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2SparseSemanticRunnerError(f"{name} is missing.") from exc
    if resolved != lexical or not resolved.is_dir():
        raise Lfm2SparseSemanticRunnerError(
            f"{name} must be one canonical real directory."
        )
    return resolved


def _output_directory(path: Path, name: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.exists() or lexical.is_symlink():
        if lexical.is_symlink() or not lexical.is_dir():
            raise Lfm2SparseSemanticRunnerError(f"{name} must be one real directory.")
        return lexical.resolve(strict=True)
    lexical.mkdir(parents=True)
    return lexical.resolve(strict=True)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Lfm2SparseSemanticRunnerError(f"{name} is missing or unsafe.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SparseSemanticRunnerError(f"{name} is unreadable JSON.") from exc
    if not isinstance(value, dict):
        raise Lfm2SparseSemanticRunnerError(f"{name} must be a JSON object.")
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
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
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


def _local_safetensor_shards(directory: Path) -> tuple[Path, ...]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file() and not index_path.is_symlink():
        index = _read_json(index_path, "parent safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise Lfm2SparseSemanticRunnerError(
                "Parent safetensors index has no weight map."
            )
        raw_names = sorted(set(weight_map.values()))
        if any(
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            for name in raw_names
        ):
            raise Lfm2SparseSemanticRunnerError(
                "Parent safetensors index contains an unsafe shard path."
            )
        shards = tuple(directory / name for name in raw_names)
    else:
        single = directory / "model.safetensors"
        shards = (
            (single,)
            if single.is_file() and not single.is_symlink()
            else tuple(sorted(directory.glob("model-*.safetensors")))
        )
    if not shards or any(
        path.is_symlink() or not path.is_file() or path.parent != directory
        for path in shards
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Parent direct-local safetensors are incomplete or unsafe."
        )
    return shards


@dataclass(frozen=True, slots=True)
class SelectiveParentTiedTable:
    table: torch.Tensor
    receipt: Mapping[str, Any]


def inspect_parent_tied_table_source(
    model_directory: Path,
    *,
    tensor_key: str = PARENT_TIED_TENSOR_KEY,
) -> dict[str, Any]:
    """Hash the config and the one indexed tied-table shard without loading it."""

    directory = _local_directory(model_directory, "parent model directory")
    config_path = directory / "config.json"
    config = _read_json(config_path, "parent LFM2 config")
    if (
        config.get("model_type") != "lfm2"
        or config.get("tie_word_embeddings") is not True
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Parent source inspection requires a tied local LFM2 checkpoint."
        )
    shards = _local_safetensor_shards(directory)
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = _read_json(index_path, "parent safetensors index")["weight_map"]
        shard_name = weight_map.get(tensor_key)
        if not isinstance(shard_name, str):
            raise Lfm2SparseSemanticRunnerError(
                f"Parent safetensors index lacks {tensor_key!r}."
            )
        selected = directory / shard_name
        if selected not in shards:
            raise Lfm2SparseSemanticRunnerError(
                "Indexed parent tied-table shard is outside the safe shard inventory."
            )
    elif len(shards) == 1:
        selected = shards[0]
    else:
        locations = []
        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                if tensor_key in handle.keys():
                    locations.append(shard)
        if len(locations) != 1:
            raise Lfm2SparseSemanticRunnerError(
                "Unindexed parent checkpoint has no unambiguous tied-table shard."
            )
        selected = locations[0]
    body = {
        "schema": "neural-forge-lfm2-parent-tied-table-source-inspection-v1",
        "source_directory": str(directory),
        "tensor_key": tensor_key,
        "config_file_sha256": _file_sha256(config_path),
        "selected_shard_name": selected.name,
        "selected_shard_sha256": _file_sha256(selected),
        "safetensor_payloads_read": 0,
        "parent_model_objects_constructed": 0,
    }
    return {**body, "inspection_sha256": canonical_sha256(body)}


def load_parent_tied_rows_from_local_safetensors(
    model_directory: Path,
    *,
    expected_parent_identity_sha256: str,
    donor_row_count: int,
    hidden_size: int,
    publication_dtype: torch.dtype = torch.float32,
    tensor_key: str = PARENT_TIED_TENSOR_KEY,
) -> SelectiveParentTiedTable:
    """Read only defined tied embedding rows; never construct the parent.

    For a sharded checkpoint the index selects exactly one shard.  Safetensors'
    slice API then reads only ``[:donor_row_count, :]``; undefined capacity rows
    and every unrelated tensor remain untouched.
    """

    directory = _local_directory(model_directory, "parent model directory")
    config_path = directory / "config.json"
    config = _read_json(config_path, "parent LFM2 config")
    rows = _positive_int(donor_row_count, "donor_row_count")
    width = _positive_int(hidden_size, "hidden_size")
    if (
        config.get("model_type") != "lfm2"
        or config.get("tie_word_embeddings") is not True
        or int(config.get("hidden_size", 0)) != width
        or int(config.get("vocab_size", 0)) < rows
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Parent config is not the expected tied LFM2 table geometry."
        )
    if publication_dtype not in {torch.float32, torch.float64}:
        raise Lfm2SparseSemanticRunnerError(
            "Parent table publication dtype must be float32 or float64."
        )
    shards = _local_safetensor_shards(directory)
    index_path = directory / "model.safetensors.index.json"
    selected_shards: tuple[Path, ...]
    if index_path.is_file():
        index = _read_json(index_path, "parent safetensors index")
        weight_map = index["weight_map"]
        shard_name = weight_map.get(tensor_key)
        if not isinstance(shard_name, str):
            raise Lfm2SparseSemanticRunnerError(
                f"Parent safetensors index lacks {tensor_key!r}."
            )
        selected_shards = (directory / shard_name,)
    else:
        selected_shards = shards

    source_slice = None
    source_shard = None
    source_shape: tuple[int, ...] | None = None
    source_dtype = None
    matching_shards = 0
    for shard in selected_shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if tensor_key not in handle.keys():
                continue
            matching_shards += 1
            if source_slice is not None:
                raise Lfm2SparseSemanticRunnerError(
                    "Parent tied table appears in more than one shard."
                )
            view = handle.get_slice(tensor_key)
            source_shape = tuple(int(size) for size in view.get_shape())
            if (
                len(source_shape) != 2
                or source_shape[0] < rows
                or source_shape[1] != width
            ):
                raise Lfm2SparseSemanticRunnerError(
                    "Parent tied table shape differs from requested defined rows."
                )
            selected = view[:rows, :]
            source_dtype = selected.dtype
            source_slice = (
                selected.detach()
                .to(device="cpu", dtype=publication_dtype)
                .contiguous()
                .clone()
            )
            source_shard = shard
    if source_slice is None or source_shard is None or matching_shards != 1:
        raise Lfm2SparseSemanticRunnerError(
            f"Parent checkpoint lacks one unambiguous tied table {tensor_key!r}."
        )
    if not bool(torch.isfinite(source_slice).all()):
        raise Lfm2SparseSemanticRunnerError(
            "Parent tied rows contain nonfinite values."
        )
    body = {
        "schema": PARENT_TABLE_LOAD_SCHEMA,
        "expected_parent_identity_sha256": _digest(
            expected_parent_identity_sha256, "expected parent identity"
        ),
        "source_directory": str(directory),
        "config_file_sha256": _file_sha256(config_path),
        "source_shard_name": source_shard.name,
        "source_shard_sha256": _file_sha256(source_shard),
        "tensor_key": tensor_key,
        "source_capacity_shape": list(source_shape or ()),
        "source_storage_dtype": str(source_dtype).removeprefix("torch."),
        "defined_rows_loaded": rows,
        "undefined_capacity_rows_loaded": 0,
        "unrelated_parent_tensors_loaded": 0,
        "published_shape": [rows, width],
        "publication_dtype": str(publication_dtype).removeprefix("torch."),
        "published_table_sha256": _tensor_sha256(source_slice),
        "parent_model_objects_constructed": 0,
        "parent_forward_calls": 0,
        "parent_generate_calls": 0,
        "network_access": False,
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    return SelectiveParentTiedTable(table=source_slice, receipt=receipt)


def cache_or_load_parent_tied_table(
    *,
    cache: DynamicFoldPhaseCache,
    events: FoldPhaseEventLog,
    phase_key: FoldPhaseKey,
    model_directory: Path,
    expected_parent_identity_sha256: str,
    donor_row_count: int,
    hidden_size: int,
    console_progress_call: Callable[[Mapping[str, Any]], None] | None = None,
) -> CachedTensorPhase:
    """Load defined parent rows once and persist the exact FP32 table phase."""

    return cache_or_compute_tensor_phase(
        cache=cache,
        events=events,
        phase_key=phase_key,
        compute_call=lambda progress: _compute_parent_table_payload(
            progress,
            model_directory=model_directory,
            expected_parent_identity_sha256=expected_parent_identity_sha256,
            donor_row_count=donor_row_count,
            hidden_size=hidden_size,
        ),
        console_progress_call=console_progress_call,
    )


def _compute_parent_table_payload(
    progress: Callable[[Mapping[str, Any]], None],
    *,
    model_directory: Path,
    expected_parent_identity_sha256: str,
    donor_row_count: int,
    hidden_size: int,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
    progress(
        {
            "stage": "selective-parent-table-load-start",
            "defined_rows": donor_row_count,
            "undefined_capacity_rows_requested": 0,
        }
    )
    loaded = load_parent_tied_rows_from_local_safetensors(
        model_directory,
        expected_parent_identity_sha256=expected_parent_identity_sha256,
        donor_row_count=donor_row_count,
        hidden_size=hidden_size,
        publication_dtype=torch.float32,
    )
    progress(
        {
            "stage": "selective-parent-table-load-complete",
            "defined_rows": donor_row_count,
            "table_sha256": loaded.receipt["published_table_sha256"],
        }
    )
    return {"donor_table": loaded.table}, {"source_receipt": dict(loaded.receipt)}


def build_parent_table_phase_key(
    *,
    expected_parent_identity_sha256: str,
    config_file_sha256: str,
    selected_shard_sha256: str,
    donor_row_count: int,
    hidden_size: int,
    publication_dtype: str,
) -> FoldPhaseKey:
    """Build a table identity that is intentionally independent of k/ridge."""

    return FoldPhaseKey(
        phase_name=PARENT_TABLE_PHASE,
        dependency_sha256s={
            "config_file_sha256": _digest(config_file_sha256, "config file SHA-256"),
            "parent_identity_sha256": _digest(
                expected_parent_identity_sha256, "parent identity"
            ),
            "selected_shard_sha256": _digest(
                selected_shard_sha256, "selected shard SHA-256"
            ),
        },
        numeric_contract={
            "donor_row_count": _positive_int(donor_row_count, "donor_row_count"),
            "hidden_size": _positive_int(hidden_size, "hidden_size"),
            "publication_dtype": str(publication_dtype),
        },
        abi={
            "schema": PARENT_TABLE_LOAD_SCHEMA,
            "selective_safetensors_slice": True,
            "parent_model_load_calls": 0,
            "k_or_ridge_in_cache_identity": False,
        },
    )


def build_neighbour_chunk_phase_key(
    *,
    parent_table_sha256: str,
    target_table_sha256: str,
    row_start: int,
    row_stop: int,
    neighbour_pool_size: int,
    solve_dtype: str,
    algorithm_sha256: str,
) -> FoldPhaseKey:
    """Identity for reusable neighbour candidates; no k or ridge appears."""

    start = _nonnegative_int(row_start, "row_start")
    stop = _positive_int(row_stop, "row_stop")
    if stop <= start:
        raise Lfm2SparseSemanticRunnerError("row_stop must exceed row_start.")
    return FoldPhaseKey(
        phase_name=NEIGHBOUR_PHASE,
        dependency_sha256s={
            "algorithm_sha256": _digest(algorithm_sha256, "algorithm SHA-256"),
            "parent_table_sha256": _digest(parent_table_sha256, "parent table SHA-256"),
            "target_table_sha256": _digest(target_table_sha256, "target table SHA-256"),
        },
        numeric_contract={
            "row_start": start,
            "row_stop": stop,
            "neighbour_pool_size": _positive_int(
                neighbour_pool_size, "neighbour_pool_size"
            ),
            "solve_dtype": str(solve_dtype),
            "deterministic_tie_break": "ascending-target-id",
        },
        abi={
            "schema": "neural-forge-lfm2-sparse-semantic-neighbours-v1",
            "reusable_across_k": True,
            "reusable_across_ridge": True,
            "k_or_ridge_in_cache_identity": False,
        },
    )


def build_coefficient_chunk_phase_key(
    *,
    parent_table_sha256: str,
    target_table_sha256: str,
    neighbour_completion_sha256: str,
    row_start: int,
    row_stop: int,
    k: int,
    ridge: float,
    solve_dtype: str,
    algorithm_sha256: str,
) -> FoldPhaseKey:
    """Identity for one C solve; unlike neighbours this binds k and ridge."""

    start = _nonnegative_int(row_start, "row_start")
    stop = _positive_int(row_stop, "row_stop")
    if stop <= start:
        raise Lfm2SparseSemanticRunnerError("row_stop must exceed row_start.")
    return FoldPhaseKey(
        phase_name=COEFFICIENT_PHASE,
        dependency_sha256s={
            "algorithm_sha256": _digest(algorithm_sha256, "algorithm SHA-256"),
            "neighbour_completion_sha256": _digest(
                neighbour_completion_sha256, "neighbour completion SHA-256"
            ),
            "parent_table_sha256": _digest(parent_table_sha256, "parent table SHA-256"),
            "target_table_sha256": _digest(target_table_sha256, "target table SHA-256"),
        },
        numeric_contract={
            "row_start": start,
            "row_stop": stop,
            "k": _positive_int(k, "k"),
            "ridge": _finite_nonnegative(ridge, "ridge"),
            "solve_dtype": str(solve_dtype),
            "coefficient_publication_dtype": "float32",
        },
        abi={
            "schema": "neural-forge-lfm2-sparse-semantic-coefficients-v1",
            "continuous_coefficients_counted_as_effective_parameters": True,
        },
    )


def build_candidate_inventory_phase_key(
    *,
    parent_table_sha256: str,
    target_table_sha256: str,
    target_to_donor_sha256: str,
    maximum_candidates: int,
    donor_rows_per_chunk: int,
    selection_placement_receipt_sha256: str,
    algorithm_sha256: str,
) -> FoldPhaseKey:
    """Bind the complete k-independent candidate inventory."""

    return FoldPhaseKey(
        phase_name=CANDIDATE_INVENTORY_PHASE,
        dependency_sha256s={
            "algorithm_sha256": _digest(algorithm_sha256, "algorithm SHA-256"),
            "parent_table_sha256": _digest(parent_table_sha256, "parent table SHA-256"),
            "selection_placement_receipt_sha256": _digest(
                selection_placement_receipt_sha256,
                "selection placement receipt SHA-256",
            ),
            "target_table_sha256": _digest(target_table_sha256, "target table SHA-256"),
            "target_to_donor_sha256": _digest(
                target_to_donor_sha256, "target-to-donor SHA-256"
            ),
        },
        numeric_contract={
            "maximum_candidates": _positive_int(
                maximum_candidates, "maximum_candidates"
            ),
            "donor_rows_per_chunk": _positive_int(
                donor_rows_per_chunk, "donor_rows_per_chunk"
            ),
        },
        abi={
            "schema": "neural-forge-lfm2-sparse-semantic-candidate-inventory-v1",
            "active_k_in_cache_identity": False,
            "ridge_in_cache_identity": False,
            "reusable_across_k_and_ridge": True,
        },
    )


def build_residual_omp_candidate_inventory_phase_key(
    *,
    parent_table_sha256: str,
    target_table_sha256: str,
    target_to_donor_sha256: str,
    active_k: int,
    orthogonality_tolerance: float,
    numeric_backend_receipt_sha256: str,
    donor_rows_per_chunk: int,
    selection_placement_receipt_sha256: str,
    algorithm_sha256: str,
) -> FoldPhaseKey:
    """Bind one completed OMP inventory while row prefixes remain k-extensible."""

    return FoldPhaseKey(
        phase_name=RESIDUAL_OMP_CANDIDATE_INVENTORY_PHASE,
        dependency_sha256s={
            "algorithm_sha256": _digest(algorithm_sha256, "algorithm SHA-256"),
            "parent_table_sha256": _digest(parent_table_sha256, "parent table SHA-256"),
            "numeric_backend_receipt_sha256": _digest(
                numeric_backend_receipt_sha256,
                "OMP numeric backend receipt SHA-256",
            ),
            "selection_placement_receipt_sha256": _digest(
                selection_placement_receipt_sha256,
                "selection placement receipt SHA-256",
            ),
            "target_table_sha256": _digest(target_table_sha256, "target table SHA-256"),
            "target_to_donor_sha256": _digest(
                target_to_donor_sha256, "target-to-donor SHA-256"
            ),
        },
        numeric_contract={
            "active_k": _positive_int(active_k, "active_k"),
            "orthogonality_tolerance": _finite_positive(
                orthogonality_tolerance, "OMP orthogonality tolerance"
            ),
            "donor_rows_per_chunk": _positive_int(
                donor_rows_per_chunk, "donor_rows_per_chunk"
            ),
        },
        abi={
            "schema": (
                "neural-forge-lfm2-residual-omp-semantic-candidate-inventory-phase-v1"
            ),
            "support_selection": _RESIDUAL_OMP_SUPPORT_SELECTION,
            "active_k_in_completed_inventory_identity": True,
            "active_k_excluded_from_prefix_base_identity": True,
            "support_prefixes_reusable_across_active_k": True,
            "ridge_in_cache_identity": False,
        },
    )


def build_codebook_phase_key(
    *,
    parent_table_sha256: str,
    target_table_sha256: str,
    target_to_donor_sha256: str,
    candidate_content_sha256: str,
    active_k: int,
    candidate_ridge: float,
    refit_ridge: float,
    donor_rows_per_chunk: int,
    solve_placement_receipt_sha256: str,
    algorithm_sha256: str,
) -> FoldPhaseKey:
    return FoldPhaseKey(
        phase_name=CODEBOOK_PHASE,
        dependency_sha256s={
            "algorithm_sha256": _digest(algorithm_sha256, "algorithm SHA-256"),
            "candidate_content_sha256": _digest(
                candidate_content_sha256, "candidate content SHA-256"
            ),
            "parent_table_sha256": _digest(parent_table_sha256, "parent table SHA-256"),
            "solve_placement_receipt_sha256": _digest(
                solve_placement_receipt_sha256,
                "solve placement receipt SHA-256",
            ),
            "target_table_sha256": _digest(target_table_sha256, "target table SHA-256"),
            "target_to_donor_sha256": _digest(
                target_to_donor_sha256, "target-to-donor SHA-256"
            ),
        },
        numeric_contract={
            "active_k": _positive_int(active_k, "active_k"),
            "candidate_ridge": _finite_nonnegative(candidate_ridge, "candidate_ridge"),
            "refit_ridge": _finite_nonnegative(refit_ridge, "refit_ridge"),
            "donor_rows_per_chunk": _positive_int(
                donor_rows_per_chunk, "donor_rows_per_chunk"
            ),
            "coefficient_publication_dtype": "float32",
        },
        abi={
            "schema": "neural-forge-lfm2-sparse-semantic-codebook-v1",
            "continuous_coefficients_counted_as_effective_parameters": True,
            "direct_anchor_rows_exact": True,
        },
    )


def sparse_codebook_tensor_content(codebook: Any) -> dict[str, Any]:
    """Hash only semantic tensors, excluding audit-dependent artifact metadata."""

    names = (
        "semantic_code_offsets",
        "semantic_code_target_ids",
        "semantic_code_coefficients",
        "target_to_donor",
    )
    tensors = {name: getattr(codebook, name, None) for name in names}
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise Lfm2SparseSemanticRunnerError(
            "Sparse codebook lacks its four authenticated tensors."
        )
    coefficients = tensors["semantic_code_coefficients"]
    if (
        coefficients.dtype != torch.float32
        or coefficients.ndim != 1
        or coefficients.numel() <= 0
        or not bool(torch.isfinite(coefficients).all())
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Sparse codebook coefficients must be finite nonempty FP32."
        )
    body = {
        "schema": "neural-forge-lfm2-sparse-codebook-tensor-content-v1",
        **{f"{name}_sha256": _tensor_sha256(value) for name, value in tensors.items()},
        "continuous_coefficient_count": int(coefficients.numel()),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def authenticate_base_codebook_override(
    codebook: Any,
    *,
    source_directory: Path,
    expected_artifact_sha256: str,
    donor_table: torch.Tensor,
    target_table: torch.Tensor,
    target_to_donor: torch.Tensor,
    maximum_candidates: int,
    active_k: int,
    candidate_ridge: float,
    refit_ridge: float,
) -> dict[str, Any]:
    """Bind an existing persisted base codebook to this exact run request."""

    receipt = _plain_json(codebook.as_dict())
    artifact_sha = _digest(
        getattr(codebook, "artifact_sha256", None),
        "base codebook artifact SHA-256",
    )
    if artifact_sha != _digest(
        expected_artifact_sha256, "expected base codebook artifact SHA-256"
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Base codebook artifact differs from the explicit expected seal."
        )
    if receipt.get("norm_reclosure") is not None:
        raise Lfm2SparseSemanticRunnerError(
            "Base codebook override must be the unreclosed coefficient artifact."
        )
    expected_hashes = {
        "donor_table_sha256": _core_sparse_tensor_sha256(donor_table),
        "anchor_table_sha256": _core_sparse_tensor_sha256(target_table),
        "target_to_donor_sha256": _core_sparse_tensor_sha256(target_to_donor),
    }
    if any(receipt.get(name) != value for name, value in expected_hashes.items()):
        raise Lfm2SparseSemanticRunnerError(
            "Base codebook override is bound to different donor, target, or mapping tensors."
        )
    if (
        receipt.get("maximum_candidates")
        != _positive_int(maximum_candidates, "maximum_candidates")
        or receipt.get("active_k") != _positive_int(active_k, "active_k")
        or float(receipt.get("candidate_ridge", -1.0))
        != _finite_nonnegative(candidate_ridge, "candidate_ridge")
        or float(receipt.get("refit_ridge", -1.0))
        != _finite_nonnegative(refit_ridge, "refit_ridge")
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Base codebook override differs from requested k, pool, or ridge controls."
        )
    tensor_content = sparse_codebook_tensor_content(codebook)
    body = {
        "schema": "neural-forge-lfm2-base-codebook-override-authentication-v1",
        "source_directory": str(source_directory),
        "artifact_sha256": artifact_sha,
        "tensor_content": tensor_content,
        "candidate_content_sha256": _digest(
            receipt.get("candidate_content_sha256"),
            "base candidate content SHA-256",
        ),
        "donor_table_sha256": expected_hashes["donor_table_sha256"],
        "anchor_table_sha256": expected_hashes["anchor_table_sha256"],
        "target_to_donor_sha256": expected_hashes["target_to_donor_sha256"],
        "maximum_candidates": int(receipt["maximum_candidates"]),
        "active_k": int(receipt["active_k"]),
        "candidate_ridge": float(receipt["candidate_ridge"]),
        "refit_ridge": float(receipt["refit_ridge"]),
        "continuous_coefficient_count": int(
            codebook.semantic_code_coefficients.numel()
        ),
        "candidate_discovery_calls": 0,
        "coefficient_solve_calls": 0,
        "loaded_via_authenticated_persistence": True,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def build_norm_reclosure_phase_key(
    *,
    base_tensor_content: Mapping[str, Any],
    base_codebook_artifact_sha256: str,
    donor_table_sha256: str,
    target_table_sha256: str,
    target_to_donor_sha256: str,
    gamma: float,
    max_scale: float,
    minimum_cosine_to_amplify: float | None,
    donor_rows_per_chunk: int,
    compute_device: str,
    compute_dtype: str,
    algorithm_sha256: str,
) -> FoldPhaseKey:
    content = _plain_json(base_tensor_content)
    expected_content_sha = content.get("content_sha256")
    content_body = dict(content)
    content_body.pop("content_sha256", None)
    if expected_content_sha != canonical_sha256(content_body):
        raise Lfm2SparseSemanticRunnerError(
            "Base sparse-codebook tensor content does not authenticate."
        )
    required_tensor_hashes = (
        "semantic_code_offsets_sha256",
        "semantic_code_target_ids_sha256",
        "semantic_code_coefficients_sha256",
        "target_to_donor_sha256",
    )
    dependencies = {
        "algorithm_sha256": _digest(algorithm_sha256, "algorithm SHA-256"),
        "base_codebook_artifact_sha256": _digest(
            base_codebook_artifact_sha256,
            "base codebook artifact SHA-256",
        ),
        "base_tensor_content_sha256": _digest(
            expected_content_sha, "base tensor content SHA-256"
        ),
        "donor_table_sha256": _digest(donor_table_sha256, "donor table SHA-256"),
        "target_table_sha256": _digest(target_table_sha256, "target table SHA-256"),
        "target_to_donor_sha256": _digest(
            target_to_donor_sha256, "target-to-donor SHA-256"
        ),
    }
    dependencies.update(
        {
            f"base_{name}": _digest(content.get(name), name)
            for name in required_tensor_hashes
        }
    )
    return FoldPhaseKey(
        phase_name=NORM_RECLOSURE_PHASE,
        dependency_sha256s=dependencies,
        numeric_contract={
            "gamma": _finite_positive(gamma, "norm reclosure gamma"),
            "max_scale": _amplification_ceiling(
                max_scale, "norm reclosure maximum scale"
            ),
            "minimum_cosine_to_amplify": _optional_cosine(
                minimum_cosine_to_amplify,
                "minimum cosine to amplify",
            ),
            "donor_rows_per_chunk": _positive_int(
                donor_rows_per_chunk, "donor_rows_per_chunk"
            ),
            "compute_device": str(compute_device),
            "compute_dtype": str(compute_dtype),
            "publication_dtype": "float32",
        },
        abi={
            "schema": "neural-forge-lfm2-coefficient-only-norm-reclosure-v1",
            "base_C_offsets_and_ids_unchanged": True,
            "base_C_coefficient_count_unchanged": True,
            "external_continuous_scale_vector_added": False,
            "direct_one_hot_rows_unchanged": True,
            "training_steps": 0,
        },
    )


def authenticated_reclosure_row_diagnostics(
    norm_reclosure_receipt: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Authenticate the core's joinable per-donor norm-reclosure sidecar."""

    receipt = _plain_json(norm_reclosure_receipt)
    raw_rows = receipt.get("row_reclosure_diagnostics")
    if not isinstance(raw_rows, list) or any(
        not isinstance(row, Mapping) for row in raw_rows
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Norm reclosure receipt lacks joinable row diagnostics."
        )
    rows = tuple(dict(_plain_json(row)) for row in raw_rows)
    expected_sha = _digest(
        receipt.get("row_reclosure_diagnostics_sha256"),
        "row reclosure diagnostics SHA-256",
    )
    if canonical_sha256(list(rows)) != expected_sha:
        raise Lfm2SparseSemanticRunnerError(
            "Norm reclosure row diagnostics do not match their seal."
        )
    return rows, expected_sha


@dataclass(frozen=True, slots=True)
class CachedTensorPhase:
    tensors: Mapping[str, torch.Tensor]
    receipt: Mapping[str, Any]
    phase_key: FoldPhaseKey
    completion_sha256: str
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class CachedJsonPhase:
    document: Mapping[str, Any]
    phase_key: FoldPhaseKey
    completion_sha256: str
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class CachedArtifactPhase:
    artifact: Any
    artifact_sha256: str
    phase_key: FoldPhaseKey
    completion_sha256: str
    cache_hit: bool


def cache_or_compute_artifact_phase(
    *,
    cache: DynamicFoldPhaseCache,
    events: FoldPhaseEventLog,
    phase_key: FoldPhaseKey,
    compute_call: Callable[[Callable[[Mapping[str, Any]], None]], Any],
    save_call: Callable[[Any, Path], None],
    load_call: Callable[[Path], Any],
    artifact_sha256_call: Callable[[Any], str],
    console_progress_call: Callable[[Mapping[str, Any]], None] | None = None,
    compatible_phase_keys: Sequence[FoldPhaseKey] = (),
) -> CachedArtifactPhase:
    """Cache one core artifact directory while reloading every publication."""

    _console_phase_event(
        console_progress_call,
        phase_key=phase_key,
        stage="cache-lookup-start",
    )
    hit = cache.lookup(phase_key, events=events)
    if hit is not None:
        artifact = load_call(hit.payload_path / "artifact")
        artifact_sha = _digest(
            artifact_sha256_call(artifact), "cached artifact SHA-256"
        )
        if artifact_sha != hit.result.get("artifact_sha256"):
            raise Lfm2SparseSemanticRunnerError(
                "Cached core artifact differs from its phase completion."
            )
        _console_phase_event(
            console_progress_call,
            phase_key=phase_key,
            stage="cache-hit",
            details={
                "artifact_sha256": artifact_sha,
                "completion_sha256": hit.completion_sha256,
            },
        )
        return CachedArtifactPhase(
            artifact=artifact,
            artifact_sha256=artifact_sha,
            phase_key=phase_key,
            completion_sha256=hit.completion_sha256,
            cache_hit=True,
        )
    for compatible_key in compatible_phase_keys:
        if compatible_key.phase_key_sha256 == phase_key.phase_key_sha256:
            continue
        compatible_hit = cache.lookup(compatible_key)
        if compatible_hit is None:
            continue
        artifact = load_call(compatible_hit.payload_path / "artifact")
        artifact_sha = _digest(
            artifact_sha256_call(artifact), "compatible cached artifact SHA-256"
        )
        if artifact_sha != compatible_hit.result.get("artifact_sha256"):
            raise Lfm2SparseSemanticRunnerError(
                "Compatible cached core artifact differs from its phase completion."
            )
        with cache.begin(phase_key, events=events) as transaction:
            destination = transaction.payload_path / "artifact"
            save_call(artifact, destination)
            reloaded = load_call(destination)
            if (
                _digest(
                    artifact_sha256_call(reloaded),
                    "promoted compatible artifact SHA-256",
                )
                != artifact_sha
            ):
                raise Lfm2SparseSemanticRunnerError(
                    "Compatible artifact changed while promoting to the math ABI."
                )
            entry = transaction.complete(
                result={
                    "artifact_sha256": artifact_sha,
                    "compatible_cache_source_phase_key_sha256": (
                        compatible_key.phase_key_sha256
                    ),
                    "compatible_cache_source_completion_sha256": (
                        compatible_hit.completion_sha256
                    ),
                    "training_steps": 0,
                    "parent_forward_calls": 0,
                }
            )
        _console_phase_event(
            console_progress_call,
            phase_key=phase_key,
            stage="compatible-cache-hit-promoted",
            details={
                "artifact_sha256": artifact_sha,
                "compatible_phase_key_sha256": compatible_key.phase_key_sha256,
                "compatible_completion_sha256": compatible_hit.completion_sha256,
                "completion_sha256": entry.completion_sha256,
            },
        )
        return CachedArtifactPhase(
            artifact=load_call(entry.payload_path / "artifact"),
            artifact_sha256=artifact_sha,
            phase_key=phase_key,
            completion_sha256=entry.completion_sha256,
            cache_hit=True,
        )
    _console_phase_event(
        console_progress_call,
        phase_key=phase_key,
        stage="cache-miss",
    )
    with cache.begin(phase_key, events=events) as transaction:

        def progress(record: Mapping[str, Any]) -> None:
            details = {"compiler_progress": _plain_json(record)}
            transaction.progress(details)
            _console_phase_event(
                console_progress_call,
                phase_key=phase_key,
                stage="compute-progress",
                details=details,
            )

        _console_phase_event(
            console_progress_call,
            phase_key=phase_key,
            stage="compute-start",
        )
        computed = compute_call(progress)
        artifact_sha = _digest(
            artifact_sha256_call(computed), "computed artifact SHA-256"
        )
        destination = transaction.payload_path / "artifact"
        save_call(computed, destination)
        reloaded = load_call(destination)
        reloaded_sha = _digest(
            artifact_sha256_call(reloaded), "reloaded artifact SHA-256"
        )
        if reloaded_sha != artifact_sha:
            raise Lfm2SparseSemanticRunnerError(
                "Core artifact changed across immediate save/load."
            )
        entry = transaction.complete(
            result={
                "artifact_sha256": artifact_sha,
                "training_steps": 0,
                "parent_forward_calls": 0,
            }
        )
    _console_phase_event(
        console_progress_call,
        phase_key=phase_key,
        stage="publication-complete",
        details={
            "artifact_sha256": artifact_sha,
            "completion_sha256": entry.completion_sha256,
        },
    )
    verified = load_call(entry.payload_path / "artifact")
    return CachedArtifactPhase(
        artifact=verified,
        artifact_sha256=artifact_sha,
        phase_key=phase_key,
        completion_sha256=entry.completion_sha256,
        cache_hit=False,
    )


def _load_json_phase_entry(
    entry: VerifiedFoldPhaseCacheEntry,
) -> dict[str, Any]:
    document = _read_json(entry.payload_path / "result.json", "cached JSON phase")
    body = dict(document)
    supplied = body.pop("result_sha256", None)
    if supplied != canonical_sha256(body) or supplied != entry.result.get(
        "result_sha256"
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Cached JSON phase self-hash or completion binding failed."
        )
    return document


def cache_or_compute_json_phase(
    *,
    cache: DynamicFoldPhaseCache,
    events: FoldPhaseEventLog,
    phase_key: FoldPhaseKey,
    compute_call: Callable[[Callable[[Mapping[str, Any]], None]], Mapping[str, Any]],
    console_progress_call: Callable[[Mapping[str, Any]], None] | None = None,
) -> CachedJsonPhase:
    _console_phase_event(
        console_progress_call,
        phase_key=phase_key,
        stage="cache-lookup-start",
    )
    hit = cache.lookup(phase_key, events=events)
    if hit is not None:
        _console_phase_event(
            console_progress_call,
            phase_key=phase_key,
            stage="cache-hit",
            details={"completion_sha256": hit.completion_sha256},
        )
        return CachedJsonPhase(
            document=_load_json_phase_entry(hit),
            phase_key=phase_key,
            completion_sha256=hit.completion_sha256,
            cache_hit=True,
        )
    _console_phase_event(
        console_progress_call,
        phase_key=phase_key,
        stage="cache-miss",
    )
    with cache.begin(phase_key, events=events) as transaction:

        def progress(record: Mapping[str, Any]) -> None:
            details = {"compiler_progress": _plain_json(record)}
            transaction.progress(details)
            _console_phase_event(
                console_progress_call,
                phase_key=phase_key,
                stage="compute-progress",
                details=details,
            )

        _console_phase_event(
            console_progress_call,
            phase_key=phase_key,
            stage="compute-start",
        )
        raw = _plain_json(compute_call(progress))
        body = dict(raw)
        body.pop("result_sha256", None)
        document = {**body, "result_sha256": canonical_sha256(body)}
        _atomic_write_json(transaction.payload_path / "result.json", document)
        entry = transaction.complete(
            result={
                "result_sha256": document["result_sha256"],
                "training_steps": 0,
                "parent_forward_calls": 0,
            }
        )
    _console_phase_event(
        console_progress_call,
        phase_key=phase_key,
        stage="publication-complete",
        details={
            "result_sha256": document["result_sha256"],
            "completion_sha256": entry.completion_sha256,
        },
    )
    return CachedJsonPhase(
        document=_load_json_phase_entry(entry),
        phase_key=phase_key,
        completion_sha256=entry.completion_sha256,
        cache_hit=False,
    )


def _load_tensor_phase_entry(
    entry: VerifiedFoldPhaseCacheEntry,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensor_path = entry.payload_path / "tensors.safetensors"
    receipt_path = entry.payload_path / "receipt.json"
    if tensor_path.is_symlink() or not tensor_path.is_file():
        raise Lfm2SparseSemanticRunnerError("Cached tensor payload is missing.")
    receipt = _read_json(receipt_path, "cached tensor phase receipt")
    body = dict(receipt)
    supplied = body.pop("receipt_sha256", None)
    if supplied != canonical_sha256(body):
        raise Lfm2SparseSemanticRunnerError(
            "Cached tensor phase receipt self-hash does not verify."
        )
    if supplied != entry.result.get("receipt_sha256"):
        raise Lfm2SparseSemanticRunnerError(
            "Cached tensor phase completion does not bind its receipt."
        )
    tensors = load_file(str(tensor_path), device="cpu")
    inventory = receipt.get("tensor_inventory")
    observed = {
        name: {
            "shape": [int(size) for size in tensor.shape],
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "sha256": _tensor_sha256(tensor),
        }
        for name, tensor in sorted(tensors.items())
    }
    if observed != inventory:
        raise Lfm2SparseSemanticRunnerError(
            "Cached tensor payload differs from its sealed inventory."
        )
    return tensors, receipt


def cache_or_compute_tensor_phase(
    *,
    cache: DynamicFoldPhaseCache,
    events: FoldPhaseEventLog,
    phase_key: FoldPhaseKey,
    compute_call: Callable[
        [Callable[[Mapping[str, Any]], None]],
        tuple[Mapping[str, torch.Tensor], Mapping[str, Any]],
    ],
    console_progress_call: Callable[[Mapping[str, Any]], None] | None = None,
) -> CachedTensorPhase:
    """Atomically load or publish one tensor phase with per-chunk progress."""

    _console_phase_event(
        console_progress_call,
        phase_key=phase_key,
        stage="cache-lookup-start",
    )
    hit = cache.lookup(phase_key, events=events)
    if hit is not None:
        tensors, receipt = _load_tensor_phase_entry(hit)
        _console_phase_event(
            console_progress_call,
            phase_key=phase_key,
            stage="cache-hit",
            details={
                "completion_sha256": hit.completion_sha256,
                "receipt_sha256": receipt["receipt_sha256"],
            },
        )
        return CachedTensorPhase(
            tensors=tensors,
            receipt=receipt,
            phase_key=phase_key,
            completion_sha256=hit.completion_sha256,
            cache_hit=True,
        )
    _console_phase_event(
        console_progress_call,
        phase_key=phase_key,
        stage="cache-miss",
    )
    with cache.begin(phase_key, events=events) as transaction:

        def progress(record: Mapping[str, Any]) -> None:
            details = {"compiler_progress": _plain_json(record)}
            transaction.progress(details)
            _console_phase_event(
                console_progress_call,
                phase_key=phase_key,
                stage="compute-progress",
                details=details,
            )

        _console_phase_event(
            console_progress_call,
            phase_key=phase_key,
            stage="compute-start",
        )
        raw_tensors, raw_metadata = compute_call(progress)
        tensors = {
            str(name): value.detach().to(device="cpu").contiguous().clone()
            for name, value in raw_tensors.items()
        }
        if not tensors or any(
            not name
            or not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type != "cpu"
            or value.is_complex()
            or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
            for name, value in tensors.items()
        ):
            raise Lfm2SparseSemanticRunnerError(
                "Tensor phase returned invalid or nonfinite payloads."
            )
        inventory = {
            name: {
                "shape": [int(size) for size in tensor.shape],
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "sha256": _tensor_sha256(tensor),
            }
            for name, tensor in sorted(tensors.items())
        }
        receipt_body = {
            "schema": TENSOR_PHASE_RECEIPT_SCHEMA,
            "phase_key_sha256": phase_key.phase_key_sha256,
            "tensor_inventory": inventory,
            "metadata": _plain_json(raw_metadata),
        }
        receipt = {
            **receipt_body,
            "receipt_sha256": canonical_sha256(receipt_body),
        }
        save_file(tensors, str(transaction.payload_path / "tensors.safetensors"))
        _atomic_write_json(transaction.payload_path / "receipt.json", receipt)
        entry = transaction.complete(
            result={
                "receipt_sha256": receipt["receipt_sha256"],
                "tensor_inventory_sha256": canonical_sha256(inventory),
                "training_steps": 0,
                "parent_forward_calls": 0,
            }
        )
    _console_phase_event(
        console_progress_call,
        phase_key=phase_key,
        stage="publication-complete",
        details={
            "completion_sha256": entry.completion_sha256,
            "receipt_sha256": receipt["receipt_sha256"],
        },
    )
    verified_tensors, verified_receipt = _load_tensor_phase_entry(entry)
    return CachedTensorPhase(
        tensors=verified_tensors,
        receipt=verified_receipt,
        phase_key=phase_key,
        completion_sha256=entry.completion_sha256,
        cache_hit=False,
    )


def _verified_core_request(request: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = _plain_json(request)
    body = dict(value)
    supplied = body.pop("request_sha256", None)
    if supplied != canonical_sha256(body):
        raise Lfm2SparseSemanticRunnerError(
            f"{name} request self-hash does not verify."
        )
    return value


def _candidate_request_phase_key(request: Mapping[str, Any]) -> FoldPhaseKey:
    value = _verified_core_request(request, "candidate chunk")
    schema = value.get("schema")
    if schema == _TOP_COSINE_CANDIDATE_REQUEST_SCHEMA:
        # Keep the established top-cosine phase identity byte-for-byte stable.
        return FoldPhaseKey(
            phase_name=NEIGHBOUR_PHASE,
            dependency_sha256s={
                "anchor_table_sha256": _digest(
                    value.get("anchor_table_sha256"), "anchor table SHA-256"
                ),
                "candidate_phase_cache_key": _digest(
                    value.get("phase_cache_key"), "candidate phase cache key"
                ),
                "donor_table_sha256": _digest(
                    value.get("donor_table_sha256"), "donor table SHA-256"
                ),
                "target_to_donor_sha256": _digest(
                    value.get("target_to_donor_sha256"), "target mapping SHA-256"
                ),
            },
            numeric_contract={
                "donor_row_start": _nonnegative_int(
                    value.get("donor_row_start"), "donor_row_start"
                ),
                "donor_row_stop": _positive_int(
                    value.get("donor_row_stop"), "donor_row_stop"
                ),
                "maximum_candidates": _positive_int(
                    value.get("maximum_candidates"), "maximum_candidates"
                ),
                "selection_device": str(value.get("selection_device")),
                "selection_dtype": str(value.get("selection_dtype")),
            },
            abi={
                "schema": str(schema),
                "core_request_sha256": value["request_sha256"],
                "active_k_or_ridge_in_identity": False,
                "reusable_across_k_and_ridge": True,
            },
        )
    if schema == _RESIDUAL_OMP_CANDIDATE_REQUEST_SCHEMA:
        start = _nonnegative_int(value.get("donor_row_start"), "donor_row_start")
        stop = _positive_int(value.get("donor_row_stop"), "donor_row_stop")
        if stop <= start:
            raise Lfm2SparseSemanticRunnerError(
                "Residual OMP donor_row_stop must exceed donor_row_start."
            )
        return FoldPhaseKey(
            phase_name=RESIDUAL_OMP_PREFIX_PHASE,
            dependency_sha256s={
                "anchor_table_sha256": _digest(
                    value.get("anchor_table_sha256"), "anchor table SHA-256"
                ),
                "base_phase_cache_key": _digest(
                    value.get("base_phase_cache_key"), "OMP base phase cache key"
                ),
                "base_request_sha256": _digest(
                    value.get("base_request_sha256"), "OMP base request SHA-256"
                ),
                "donor_table_sha256": _digest(
                    value.get("donor_table_sha256"), "donor table SHA-256"
                ),
                "implementation_sha256": _digest(
                    value.get("implementation_sha256"),
                    "OMP implementation SHA-256",
                ),
                "numeric_backend_receipt_sha256": _digest(
                    value.get("numeric_backend_receipt_sha256"),
                    "OMP numeric backend receipt SHA-256",
                ),
                "target_to_donor_sha256": _digest(
                    value.get("target_to_donor_sha256"), "target mapping SHA-256"
                ),
            },
            numeric_contract={
                "donor_row_start": start,
                "donor_row_stop": stop,
                "support_depth": _positive_int(
                    value.get("support_depth"), "OMP support_depth"
                ),
                "orthogonality_tolerance": _finite_positive(
                    value.get("orthogonality_tolerance"),
                    "OMP orthogonality tolerance",
                ),
                "selection_device": str(value.get("selection_device")),
                "selection_dtype": str(value.get("selection_dtype")),
            },
            abi={
                "schema": str(schema),
                "algorithm": str(value.get("algorithm")),
                "core_request_sha256": value["request_sha256"],
                "final_active_k_in_prefix_identity": False,
                "support_prefix_reusable_across_active_k": True,
                "ridge_in_identity": False,
            },
        )
    raise Lfm2SparseSemanticRunnerError(
        f"Unsupported candidate request schema: {schema!r}."
    )


def _coefficient_request_phase_key(request: Mapping[str, Any]) -> FoldPhaseKey:
    value = _verified_core_request(request, "coefficient chunk")
    return FoldPhaseKey(
        phase_name=COEFFICIENT_PHASE,
        dependency_sha256s={
            "anchor_table_sha256": _digest(
                value.get("anchor_table_sha256"), "anchor table SHA-256"
            ),
            "candidate_content_sha256": _digest(
                value.get("candidate_content_sha256"),
                "candidate content SHA-256",
            ),
            "coefficient_phase_cache_key": _digest(
                value.get("phase_cache_key"), "coefficient phase cache key"
            ),
            "donor_table_sha256": _digest(
                value.get("donor_table_sha256"), "donor table SHA-256"
            ),
            "target_to_donor_sha256": _digest(
                value.get("target_to_donor_sha256"), "target mapping SHA-256"
            ),
        },
        numeric_contract={
            "donor_row_start": _nonnegative_int(
                value.get("donor_row_start"), "donor_row_start"
            ),
            "donor_row_stop": _positive_int(
                value.get("donor_row_stop"), "donor_row_stop"
            ),
            "active_k": _positive_int(value.get("active_k"), "active_k"),
            "candidate_ridge": _finite_nonnegative(
                value.get("candidate_ridge"), "candidate_ridge"
            ),
            "refit_ridge": _finite_nonnegative(value.get("refit_ridge"), "refit_ridge"),
            "solve_device": str(value.get("solve_device")),
            "solve_dtype": str(value.get("solve_dtype")),
            "coefficient_publication_dtype": "float32",
        },
        abi={
            "schema": str(value.get("schema")),
            "core_request_sha256": value["request_sha256"],
            "continuous_coefficients_counted_as_child_state": True,
        },
    )


class CoreChunkPhaseAdapter:
    """Translate core checkpoint hooks into immutable DynamicFoldPhaseCache entries."""

    def __init__(
        self,
        *,
        cache: DynamicFoldPhaseCache,
        events_directory: Path,
        candidate_checkpoint_type: type[Any],
        coefficient_checkpoint_type: type[Any],
        support_math_abi_sha256: str,
        coefficient_math_abi_sha256: str,
        console_progress_call: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.cache = cache
        self.events_directory = events_directory
        self.candidate_checkpoint_type = candidate_checkpoint_type
        self.coefficient_checkpoint_type = coefficient_checkpoint_type
        self.support_math_abi_sha256 = _digest(
            support_math_abi_sha256, "support math ABI SHA-256"
        )
        self.coefficient_math_abi_sha256 = _digest(
            coefficient_math_abi_sha256, "coefficient math ABI SHA-256"
        )
        self.console_progress_call = console_progress_call
        self._candidate_pending: dict[str, tuple[FoldPhaseKey, FoldPhaseEventLog]] = {}
        self._coefficient_pending: dict[
            str, tuple[FoldPhaseKey, FoldPhaseEventLog]
        ] = {}
        self.audit = {
            "candidate_chunk_cache_hits": 0,
            "candidate_chunk_legacy_compatible_cache_hits": 0,
            "candidate_chunk_cache_misses": 0,
            "candidate_chunk_publications": 0,
            "candidate_support_prefix_cache_hits_by_depth": {},
            "candidate_support_prefix_cache_misses_by_depth": {},
            "candidate_support_prefix_publications_by_depth": {},
            "coefficient_chunk_cache_hits": 0,
            "coefficient_chunk_legacy_compatible_cache_hits": 0,
            "coefficient_chunk_cache_misses": 0,
            "coefficient_chunk_publications": 0,
        }

    def _report_chunk_cache(
        self,
        *,
        kind: str,
        stage: str,
        request: Mapping[str, Any],
        phase_key: FoldPhaseKey,
    ) -> None:
        if self.console_progress_call is None:
            return
        self.console_progress_call(
            {
                "phase": phase_key.phase_name,
                "phase_key_sha256": phase_key.phase_key_sha256,
                "stage": stage,
                "details": {
                    "chunk_kind": kind,
                    "donor_row_start": request.get("donor_row_start"),
                    "donor_row_stop": request.get("donor_row_stop"),
                    "support_depth": request.get("support_depth"),
                    "request_sha256": request.get("request_sha256"),
                },
            }
        )

    def _record_support_depth(self, field: str, request: Mapping[str, Any]) -> None:
        if request.get("schema") not in {
            _RESIDUAL_OMP_CANDIDATE_REQUEST_SCHEMA,
            _RESIDUAL_OMP_CANDIDATE_CHUNK_SCHEMA,
        }:
            return
        depth = str(_positive_int(request.get("support_depth"), "OMP support_depth"))
        counts = self.audit[field]
        counts[depth] = int(counts.get(depth, 0)) + 1

    def _events(
        self,
        *,
        prefix: str,
        request: Mapping[str, Any],
        phase_key: FoldPhaseKey,
    ) -> FoldPhaseEventLog:
        start = _nonnegative_int(request.get("donor_row_start"), "donor_row_start")
        stop = _positive_int(request.get("donor_row_stop"), "donor_row_stop")
        support_depth = request.get("support_depth")
        suffix = (
            ""
            if support_depth is None
            else f"-depth-{_positive_int(support_depth, 'support_depth'):04d}"
        )
        return _event_log(
            self.events_directory,
            label=f"{prefix}-{start:06d}-{stop:06d}{suffix}",
            phase_key=phase_key,
        )

    def load_candidate(self, request: Mapping[str, Any]) -> Any | None:
        value = _verified_core_request(request, "candidate chunk")
        legacy_key = _candidate_request_phase_key(value)
        key = bind_phase_implementation(
            legacy_key,
            dependency_name="support_math_abi_sha256",
            implementation_sha256=self.support_math_abi_sha256,
        )
        events = self._events(prefix="candidate", request=value, phase_key=key)
        hit = self.cache.lookup(key, events=events)
        hit_key = key
        legacy_compatible_hit = False
        if hit is None:
            hit = self.cache.lookup(legacy_key, events=events)
            if hit is not None:
                hit_key = legacy_key
                legacy_compatible_hit = True
        if hit is None:
            self.audit["candidate_chunk_cache_misses"] += 1
            self._record_support_depth(
                "candidate_support_prefix_cache_misses_by_depth", value
            )
            self._candidate_pending[value["request_sha256"]] = (key, events)
            self._report_chunk_cache(
                kind="candidate",
                stage="chunk-cache-miss",
                request=value,
                phase_key=key,
            )
            return None
        tensors, receipt = _load_tensor_phase_entry(hit)
        metadata = receipt.get("metadata", {})
        checkpoint_receipt = metadata.get("checkpoint_receipt")
        if not isinstance(checkpoint_receipt, Mapping):
            raise Lfm2SparseSemanticRunnerError(
                "Cached candidate chunk has no core checkpoint receipt."
            )
        self.audit["candidate_chunk_cache_hits"] += 1
        self.audit["candidate_chunk_legacy_compatible_cache_hits"] += int(
            legacy_compatible_hit
        )
        self._record_support_depth(
            "candidate_support_prefix_cache_hits_by_depth", value
        )
        self._report_chunk_cache(
            kind="candidate",
            stage=(
                "chunk-legacy-compatible-cache-hit"
                if legacy_compatible_hit
                else "chunk-cache-hit"
            ),
            request=value,
            phase_key=hit_key,
        )
        return self.candidate_checkpoint_type(
            tensors["candidate_offsets"],
            tensors["candidate_target_ids"],
            checkpoint_receipt,
        )

    def publish_candidate(self, checkpoint: Any) -> None:
        if not isinstance(checkpoint, self.candidate_checkpoint_type):
            raise Lfm2SparseSemanticRunnerError(
                "Core published an invalid candidate checkpoint."
            )
        receipt = _plain_json(checkpoint.as_dict())
        request_sha = _digest(
            receipt.get("request_sha256"), "candidate request SHA-256"
        )
        pending = self._candidate_pending.pop(request_sha, None)
        if pending is None:
            raise Lfm2SparseSemanticRunnerError(
                "Candidate publisher has no preceding exact cache miss."
            )
        key, events = pending
        cache_or_compute_tensor_phase(
            cache=self.cache,
            events=events,
            phase_key=key,
            compute_call=lambda _progress: (
                {
                    "candidate_offsets": checkpoint.candidate_offsets,
                    "candidate_target_ids": checkpoint.candidate_target_ids,
                },
                {"checkpoint_receipt": receipt},
            ),
            console_progress_call=self.console_progress_call,
        )
        self.audit["candidate_chunk_publications"] += 1
        self._record_support_depth(
            "candidate_support_prefix_publications_by_depth", receipt
        )
        self._report_chunk_cache(
            kind="candidate",
            stage="chunk-publication-complete",
            request=receipt,
            phase_key=key,
        )

    def load_coefficient(self, request: Mapping[str, Any]) -> Any | None:
        value = _verified_core_request(request, "coefficient chunk")
        legacy_key = _coefficient_request_phase_key(value)
        key = bind_phase_implementation(
            legacy_key,
            dependency_name="coefficient_math_abi_sha256",
            implementation_sha256=self.coefficient_math_abi_sha256,
        )
        events = self._events(prefix="coefficient", request=value, phase_key=key)
        hit = self.cache.lookup(key, events=events)
        hit_key = key
        legacy_compatible_hit = False
        if hit is None:
            hit = self.cache.lookup(legacy_key, events=events)
            if hit is not None:
                hit_key = legacy_key
                legacy_compatible_hit = True
        if hit is None:
            self.audit["coefficient_chunk_cache_misses"] += 1
            self._coefficient_pending[value["request_sha256"]] = (key, events)
            self._report_chunk_cache(
                kind="coefficient",
                stage="chunk-cache-miss",
                request=value,
                phase_key=key,
            )
            return None
        tensors, receipt = _load_tensor_phase_entry(hit)
        metadata = receipt.get("metadata", {})
        checkpoint_receipt = metadata.get("checkpoint_receipt")
        if not isinstance(checkpoint_receipt, Mapping):
            raise Lfm2SparseSemanticRunnerError(
                "Cached coefficient chunk has no core checkpoint receipt."
            )
        self.audit["coefficient_chunk_cache_hits"] += 1
        self.audit["coefficient_chunk_legacy_compatible_cache_hits"] += int(
            legacy_compatible_hit
        )
        self._report_chunk_cache(
            kind="coefficient",
            stage=(
                "chunk-legacy-compatible-cache-hit"
                if legacy_compatible_hit
                else "chunk-cache-hit"
            ),
            request=value,
            phase_key=hit_key,
        )
        return self.coefficient_checkpoint_type(
            tensors["semantic_code_offsets"],
            tensors["semantic_code_target_ids"],
            tensors["semantic_code_coefficients"],
            checkpoint_receipt,
        )

    def publish_coefficient(self, checkpoint: Any) -> None:
        if not isinstance(checkpoint, self.coefficient_checkpoint_type):
            raise Lfm2SparseSemanticRunnerError(
                "Core published an invalid coefficient checkpoint."
            )
        receipt = _plain_json(checkpoint.as_dict())
        request_sha = _digest(
            receipt.get("request_sha256"), "coefficient request SHA-256"
        )
        pending = self._coefficient_pending.pop(request_sha, None)
        if pending is None:
            raise Lfm2SparseSemanticRunnerError(
                "Coefficient publisher has no preceding exact cache miss."
            )
        key, events = pending
        cache_or_compute_tensor_phase(
            cache=self.cache,
            events=events,
            phase_key=key,
            compute_call=lambda _progress: (
                {
                    "semantic_code_offsets": checkpoint.semantic_code_offsets,
                    "semantic_code_target_ids": checkpoint.semantic_code_target_ids,
                    "semantic_code_coefficients": (
                        checkpoint.semantic_code_coefficients
                    ),
                },
                {"checkpoint_receipt": receipt},
            ),
            console_progress_call=self.console_progress_call,
        )
        self.audit["coefficient_chunk_publications"] += 1
        self._report_chunk_cache(
            kind="coefficient",
            stage="chunk-publication-complete",
            request=receipt,
            phase_key=key,
        )

    def assert_drained(self) -> None:
        if self._candidate_pending or self._coefficient_pending:
            raise Lfm2SparseSemanticRunnerError(
                "Core returned without publishing every cache-missed chunk."
            )


def effective_child_parameter_accounting(
    *,
    physical_child_parameters: int,
    semantic_code_coefficients: torch.Tensor,
    parent_parameters: int,
) -> dict[str, Any]:
    """Count every continuous C scalar, even though routing is unregistered."""

    physical = _positive_int(physical_child_parameters, "physical child parameters")
    parent = _positive_int(parent_parameters, "parent parameters")
    coefficients = semantic_code_coefficients
    if (
        not isinstance(coefficients, torch.Tensor)
        or coefficients.dtype != torch.float32
        or coefficients.ndim != 1
        or coefficients.numel() <= 0
        or not bool(torch.isfinite(coefficients).all())
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Semantic C coefficients must be one nonempty finite FP32 vector."
        )
    continuous = int(coefficients.numel())
    effective = physical + continuous
    if effective >= parent:
        raise Lfm2SparseSemanticRunnerError(
            "Sparse semantic bridge is not parameter-compacted after counting C."
        )
    return {
        "physical_child_parameter_count": physical,
        "continuous_semantic_code_coefficient_count": continuous,
        "integer_routing_metadata_counted_as_parameters": 0,
        "effective_child_parameter_count": effective,
        "parent_parameter_count": parent,
        "effective_parameter_compaction_ratio": parent / effective,
        "all_continuous_C_coefficients_counted": True,
    }


def resolve_compile_placement(
    *,
    role: str,
    requested_device: str,
    requested_dtype: str,
) -> tuple[torch.device, torch.dtype, dict[str, Any]]:
    """Resolve candidate/solve placement without silent coercion."""

    device = torch.device(requested_device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise Lfm2SparseSemanticRunnerError(
                f"{role} requested CUDA but CUDA is unavailable."
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise Lfm2SparseSemanticRunnerError(
                f"{role} requested a CUDA device outside the visible inventory."
            )
        device = torch.device("cuda", index)
    elif device.type != "cpu":
        raise Lfm2SparseSemanticRunnerError(
            f"{role} supports only CPU or CUDA placement."
        )
    automatic_dtype = (
        torch.float32
        if role == "candidate-selection" or device.type == "cuda"
        else torch.float64
    )
    aliases = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp64": torch.float64,
        "float64": torch.float64,
        "auto": automatic_dtype,
    }
    normalized = str(requested_dtype).strip().lower()
    if normalized not in aliases:
        raise Lfm2SparseSemanticRunnerError(
            f"{role} dtype must be auto, float32, or float64."
        )
    dtype = aliases[normalized]
    body = {
        "schema": "neural-forge-sparse-semantic-compile-placement-v1",
        "role": role,
        "requested_device": requested_device,
        "resolved_device": str(device),
        "requested_dtype": normalized,
        "resolved_dtype": str(dtype).removeprefix("torch."),
        "automatic_resolution": normalized == "auto",
        "fallback_performed": False,
        "silent_dtype_coercion": False,
        "coefficient_publication_dtype": (
            "float32" if role == "coefficient-solve" else None
        ),
    }
    return device, dtype, {**body, "receipt_sha256": canonical_sha256(body)}


def resolve_runtime_accumulator_dtype(forward_dtype: torch.dtype) -> torch.dtype:
    """Mirror the sparse runtime's explicit low-precision accumulation policy."""

    if forward_dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    if forward_dtype in {torch.float32, torch.float64}:
        return forward_dtype
    raise Lfm2SparseSemanticRunnerError(
        "Forward dtype must be FP16, BF16, FP32, or FP64."
    )


def relocate_child_for_runtime(
    child: Any,
    *,
    runtime_device: torch.device,
    forward_dtype: torch.dtype,
    expected_state_sha256: str,
) -> dict[str, Any]:
    """Relocate the CPU-staged child only after analytic compilation finishes."""

    expected_sha = _digest(expected_state_sha256, "expected child state SHA-256")
    before_values = tuple(child.state_dict().values())
    if not before_values or {value.device.type for value in before_values} != {"cpu"}:
        raise Lfm2SparseSemanticRunnerError(
            "Child must remain wholly CPU staged until runtime relocation."
        )
    before_floating_dtypes = {
        value.dtype for value in before_values if value.is_floating_point()
    }
    if before_floating_dtypes != {forward_dtype}:
        raise Lfm2SparseSemanticRunnerError(
            "CPU-staged child floating state differs from the forward dtype."
        )
    before_sha = fingerprint_lfm2_direct_local_state(child)
    if before_sha != expected_sha:
        raise Lfm2SparseSemanticRunnerError(
            "CPU-staged child changed during analytic compilation."
        )

    with torch.no_grad():
        child.to(device=runtime_device, dtype=forward_dtype)
    child.requires_grad_(False)
    child.eval()
    after_values = tuple(child.state_dict().values())
    observed_devices = {value.device for value in after_values}
    observed_floating_dtypes = {
        value.dtype for value in after_values if value.is_floating_point()
    }
    if len(observed_devices) != 1 or observed_floating_dtypes != {forward_dtype}:
        raise Lfm2SparseSemanticRunnerError(
            "Runtime child state does not match its requested device/dtype policy."
        )
    observed_device = next(iter(observed_devices))
    if observed_device.type != runtime_device.type or (
        runtime_device.index is not None
        and observed_device.index != runtime_device.index
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Runtime child state is not on the requested device."
        )
    after_sha = fingerprint_lfm2_direct_local_state(child)
    if after_sha != before_sha:
        raise Lfm2SparseSemanticRunnerError(
            "Device-only runtime relocation changed child state bytes."
        )
    body = {
        "schema": "neural-forge-lfm2-deferred-runtime-relocation-v1",
        "compile_staging_device": "cpu",
        "runtime_device": str(observed_device),
        "forward_dtype": str(forward_dtype).removeprefix("torch."),
        "state_sha256_before_relocation": before_sha,
        "state_sha256_after_relocation": after_sha,
        "device_only_relocation": True,
        "analytic_compilation_completed_before_runtime_relocation": True,
        "training_steps": 0,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def build_numeric_dtype_contract(
    *,
    parent_source_receipt: Mapping[str, Any],
    portable_load_receipt: Mapping[str, Any],
    runtime_dtype_receipt: Mapping[str, Any],
    selection_receipt: Mapping[str, Any],
    solve_receipt: Mapping[str, Any],
    forward_dtype: torch.dtype,
    norm_reclosure_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Publish each independent storage/compute dtype without conflation."""

    accumulator = resolve_runtime_accumulator_dtype(forward_dtype)
    body = {
        "schema": "neural-forge-lfm2-sparse-semantic-numeric-dtype-contract-v1",
        "parent_source_storage_dtype": str(
            parent_source_receipt.get("source_storage_dtype")
        ),
        "parent_table_publication_dtype": str(
            parent_source_receipt.get("publication_dtype")
        ),
        "child_checkpoint_provenance_dtype": str(
            portable_load_receipt.get("provenance_dtype")
        ).removeprefix("torch."),
        "child_compile_staging_dtype": str(
            portable_load_receipt.get("runtime_dtype")
        ).removeprefix("torch."),
        "forward_dtype": str(forward_dtype).removeprefix("torch."),
        "runtime_dtype_resolution": _plain_json(runtime_dtype_receipt),
        "routing_accumulator_dtype": str(accumulator).removeprefix("torch."),
        "low_precision_forward_accumulates_in_float32": (
            forward_dtype not in {torch.float16, torch.bfloat16}
            or accumulator == torch.float32
        ),
        "candidate_selection_dtype": str(selection_receipt.get("resolved_dtype")),
        "coefficient_solve_dtype": str(solve_receipt.get("resolved_dtype")),
        "coefficient_publication_dtype": "float32",
        "norm_reclosure_dtype": (
            None
            if norm_reclosure_receipt is None
            else str(norm_reclosure_receipt.get("resolved_dtype"))
        ),
        "dtype_roles_are_independent": True,
        "silent_dtype_coercion": False,
    }
    return {**body, "contract_sha256": canonical_sha256(body)}


def build_hidden_diagnostic_phase_key(
    *,
    trace_cache_key: str,
    trace_manifest_sha256: str,
    routing_contract_sha256: str,
    target_table_sha256: str,
    diagnostic_implementation_sha256: str,
    reclosure_row_diagnostics_sha256: str,
    reclosure_provenance_enabled: bool,
    reclosure_max_scale: float | None,
    maximum_hidden_rows_per_score_chunk: int,
) -> FoldPhaseKey:
    if not isinstance(reclosure_provenance_enabled, bool):
        raise Lfm2SparseSemanticRunnerError(
            "Reclosure provenance enabled flag must be boolean."
        )
    normalized_max_scale = (
        _amplification_ceiling(reclosure_max_scale, "reclosure maximum scale")
        if reclosure_provenance_enabled
        else None
    )
    if not reclosure_provenance_enabled and reclosure_max_scale is not None:
        raise Lfm2SparseSemanticRunnerError(
            "Disabled reclosure provenance cannot bind a maximum scale."
        )
    return FoldPhaseKey(
        phase_name=DIAGNOSTIC_PHASE,
        dependency_sha256s={
            "diagnostic_implementation_sha256": _digest(
                diagnostic_implementation_sha256,
                "diagnostic implementation SHA-256",
            ),
            "routing_contract_sha256": _digest(
                routing_contract_sha256, "routing contract SHA-256"
            ),
            "reclosure_row_diagnostics_sha256": _digest(
                reclosure_row_diagnostics_sha256,
                "reclosure row diagnostics SHA-256",
            ),
            "target_table_sha256": _digest(target_table_sha256, "target table SHA-256"),
            "trace_cache_key": _digest(trace_cache_key, "trace cache key"),
            "trace_manifest_sha256": _digest(
                trace_manifest_sha256, "trace manifest SHA-256"
            ),
        },
        numeric_contract={
            "maximum_hidden_rows_per_score_chunk": _positive_int(
                maximum_hidden_rows_per_score_chunk,
                "maximum_hidden_rows_per_score_chunk",
            ),
            "winner_domain": "contiguous-defined-donor-rows-only",
            "reclosure_provenance_enabled": reclosure_provenance_enabled,
            "reclosure_max_scale": normalized_max_scale,
        },
        abi={
            "schema": HIDDEN_DIAGNOSTIC_SCHEMA,
            "selected_cache_tensor_suffixes": [
                ".output.lm_head_input",
                ".logits",
                ".full_prefix_logits",
            ],
            "advisory_not_gate": True,
            "reclosure_provenance_joined_by_donor_id": True,
        },
    )


def bind_phase_implementation(
    phase_key: FoldPhaseKey,
    *,
    dependency_name: str,
    implementation_sha256: str,
) -> FoldPhaseKey:
    """Add an implementation identity to an existing reusable phase key."""

    value = phase_key.as_dict()
    dependencies = dict(value["dependency_sha256s"])
    if dependency_name in dependencies:
        raise Lfm2SparseSemanticRunnerError(
            f"Phase key already binds {dependency_name!r}."
        )
    dependencies[dependency_name] = _digest(
        implementation_sha256, f"{dependency_name} SHA-256"
    )
    return FoldPhaseKey(
        phase_name=value["phase_name"],
        dependency_sha256s=dependencies,
        numeric_contract=value["numeric_contract"],
        abi=value["abi"],
    )


def _head_logit_pairs(names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    inventory = frozenset(names)
    suffix = ".output.lm_head_input"
    pairs: list[tuple[str, str]] = []
    for hidden_name in sorted(name for name in inventory if name.endswith(suffix)):
        phase = hidden_name[: -len(suffix)]
        candidates = (f"{phase}.logits", f"{phase}.full_prefix_logits")
        present = tuple(name for name in candidates if name in inventory)
        if not present:
            continue
        # Canonical logits are preferred where both recurrent and canonical
        # full-prefix copies exist.
        logit_name = (
            f"{phase}.full_prefix_logits"
            if f"{phase}.full_prefix_logits" in present
            else present[0]
        )
        pairs.append((hidden_name, logit_name))
    return tuple(pairs)


def evaluate_cached_hidden_top1_recovery(
    *,
    trace_cache_directory: Path,
    expected_parent_identity_sha256: str,
    routing: Any,
    live_target_table: torch.Tensor,
    protected_target_to_donor: Sequence[int] | torch.Tensor,
    row_reclosure_diagnostics: Sequence[Mapping[str, Any]] | None = None,
    row_reclosure_max_scale: float | None = None,
    maximum_hidden_rows_per_score_chunk: int = 8,
    expected_prior_direct_winners: int = 293,
    expected_prior_multi_code_winners: int = 72,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Measure C on sealed parent hidden rows without executing the parent."""

    cache = Lfm2CircuitTraceCache.open(trace_cache_directory, verify=False)
    expected_parent = _digest(
        expected_parent_identity_sha256, "expected parent identity"
    )
    if cache.parent_fingerprint != expected_parent:
        raise Lfm2SparseSemanticRunnerError(
            "Hidden diagnostic cache belongs to another parent."
        )
    chunk_size = _positive_int(
        maximum_hidden_rows_per_score_chunk,
        "maximum_hidden_rows_per_score_chunk",
    )
    required_methods = (
        "score_target_tokens",
        "compute_all_donor_logits_from_target_scores",
        "authenticated_live_table_session",
    )
    if any(not callable(getattr(routing, name, None)) for name in required_methods):
        raise Lfm2SparseSemanticRunnerError(
            "Sparse semantic routing lacks diagnostic operations."
        )
    donor_count = _positive_int(getattr(routing, "donor_row_count", None), "donor rows")
    if donor_count < 2:
        raise Lfm2SparseSemanticRunnerError(
            "Hidden rank diagnostics require at least two donor rows."
        )
    target_count = _positive_int(
        getattr(routing, "target_row_count", None), "target rows"
    )
    hidden_size = _positive_int(getattr(routing, "hidden_size", None), "hidden size")
    if tuple(live_target_table.shape) != (target_count, hidden_size):
        raise Lfm2SparseSemanticRunnerError(
            "Live target table differs from diagnostic routing dimensions."
        )
    protected = torch.as_tensor(
        protected_target_to_donor, dtype=torch.int64, device="cpu"
    ).reshape(-1)
    if (
        int(protected.numel()) != target_count
        or bool((protected < 0).any())
        or bool((protected >= donor_count).any())
        or int(torch.unique(protected).numel()) != target_count
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Hidden diagnostic requires the authenticated injective C anchor map."
        )
    direct_donor_ids = frozenset(int(value) for value in protected.tolist())
    reclosure_provenance_supplied = row_reclosure_diagnostics is not None
    reclosure_rows = (
        tuple()
        if row_reclosure_diagnostics is None
        else tuple(dict(_plain_json(row)) for row in row_reclosure_diagnostics)
    )
    if reclosure_provenance_supplied:
        if row_reclosure_max_scale is None:
            raise Lfm2SparseSemanticRunnerError(
                "Row reclosure provenance requires its authenticated maximum scale."
            )
        provenance_max_scale = _amplification_ceiling(
            row_reclosure_max_scale, "row reclosure maximum scale"
        )
    else:
        if row_reclosure_max_scale is not None:
            raise Lfm2SparseSemanticRunnerError(
                "A reclosure maximum scale requires row provenance."
            )
        provenance_max_scale = None
    reclosure_by_donor: dict[int, dict[str, Any]] = {}
    required_reclosure_fields = (
        "semantic_code_edge_count",
        "before_norm_ratio",
        "after_norm_ratio",
        "applied_scale",
        "before_cosine",
        "after_cosine",
        "before_relative_l2_error",
        "after_relative_l2_error",
        "amplified",
        "scale_cap_applied",
        "unresolved_low_cosine",
    )
    for row in reclosure_rows:
        donor_id = row.get("donor_id")
        if (
            isinstance(donor_id, bool)
            or not isinstance(donor_id, int)
            or donor_id < 0
            or donor_id >= donor_count
            or donor_id in direct_donor_ids
            or donor_id in reclosure_by_donor
            or any(field not in row for field in required_reclosure_fields)
        ):
            raise Lfm2SparseSemanticRunnerError(
                "Row reclosure provenance has invalid or duplicate donor identity."
            )
        _positive_int(
            row["semantic_code_edge_count"], "reclosure semantic code edge count"
        )
        for field in (
            "before_norm_ratio",
            "after_norm_ratio",
            "before_relative_l2_error",
            "after_relative_l2_error",
        ):
            _finite_nonnegative(row[field], f"reclosure {field}")
        applied_scale = _finite_positive(
            row["applied_scale"], "reclosure applied scale"
        )
        assert provenance_max_scale is not None
        if applied_scale > provenance_max_scale * (1.0 + 1e-6):
            raise Lfm2SparseSemanticRunnerError(
                "Reclosure applied scale exceeds its authenticated maximum."
            )
        for field in ("before_cosine", "after_cosine"):
            cosine = _finite_scalar(row[field], f"reclosure {field}")
            if cosine < -1.0 - 1e-5 or cosine > 1.0 + 1e-5:
                raise Lfm2SparseSemanticRunnerError(
                    "Reclosure row cosine exceeds numerical tolerance."
                )
        for field in ("amplified", "scale_cap_applied", "unresolved_low_cosine"):
            if not isinstance(row[field], bool):
                raise Lfm2SparseSemanticRunnerError(
                    f"Reclosure {field} provenance must be boolean."
                )
        reclosure_by_donor[donor_id] = row
    if reclosure_provenance_supplied and set(reclosure_by_donor) != (
        set(range(donor_count)) - set(direct_donor_ids)
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Row reclosure provenance does not cover every non-direct donor exactly once."
        )

    def cosine_band(value: float) -> str:
        value = min(1.0, max(-1.0, value))
        if value < 0.0:
            return "negative"
        if value < 0.25:
            return "zero-to-0.25"
        if value < 0.5:
            return "0.25-to-0.5"
        if value < 0.75:
            return "0.5-to-0.75"
        return "0.75-to-1.0"

    transition_groups: dict[str, dict[str, dict[str, Any]]] = {
        "by_scale_cap_applied": {},
        "by_unresolved_low_cosine": {},
        "by_before_cosine_band": {},
    }

    def record_transition(
        group: str,
        bucket: str,
        *,
        matched: bool,
        predicted_is_direct: bool,
        parent_margin: float,
    ) -> None:
        counters = transition_groups[group].setdefault(
            bucket,
            {
                "row_count": 0,
                "top1_match_count": 0,
                "top1_mismatch_count": 0,
                "predicted_direct_anchor_count": 0,
                "predicted_multi_code_count": 0,
                "parent_winner_margin_sum": 0.0,
            },
        )
        counters["row_count"] += 1
        counters["top1_match_count"] += int(matched)
        counters["top1_mismatch_count"] += int(not matched)
        counters["predicted_direct_anchor_count"] += int(predicted_is_direct)
        counters["predicted_multi_code_count"] += int(not predicted_is_direct)
        counters["parent_winner_margin_sum"] += parent_margin

    total = 0
    matches = 0
    top5_matches = 0
    top10_matches = 0
    reciprocal_rank_sum = 0.0
    direct_total = 0
    direct_matches = 0
    direct_top5_matches = 0
    direct_top10_matches = 0
    direct_reciprocal_rank_sum = 0.0
    multi_total = 0
    multi_matches = 0
    multi_top5_matches = 0
    multi_top10_matches = 0
    multi_reciprocal_rank_sum = 0.0
    predicted_direct_total = 0
    predicted_multi_total = 0
    parent_winner_margin_sum = 0.0
    parent_winner_positive_margin_count = 0
    parent_winner_zero_margin_count = 0
    parent_winner_negative_margin_count = 0
    winner_type_confusion = {
        "parent_direct_predicted_direct": 0,
        "parent_direct_predicted_multi": 0,
        "parent_multi_predicted_direct": 0,
        "parent_multi_predicted_multi": 0,
    }
    selected_tensor_bytes = 0
    batch_reports: list[dict[str, Any]] = []
    row_rank_diagnostics: list[dict[str, Any]] = []
    with cache.selective_authenticated_session() as trace_session:
        with routing.authenticated_live_table_session(
            live_target_table
        ) as route_session:
            for reference in trace_session.batch_references:
                pairs = _head_logit_pairs(reference.ordered_tensor_names)
                if not pairs:
                    continue
                selected_names = tuple(
                    sorted({name for pair in pairs for name in pair})
                )
                payload = trace_session.load_selected_tensors(
                    reference.index, selected_names
                )
                try:
                    selected_tensor_bytes += int(payload.logical_bytes)
                    batch_total = 0
                    batch_matches = 0
                    batch_top5_matches = 0
                    batch_top10_matches = 0
                    batch_reciprocal_rank_sum = 0.0
                    for hidden_name, logit_name in pairs:
                        hidden = payload.tensors[hidden_name]
                        logits = payload.tensors[logit_name]
                        if (
                            hidden.ndim < 2
                            or logits.ndim != hidden.ndim
                            or tuple(hidden.shape[:-1]) != tuple(logits.shape[:-1])
                            or int(hidden.shape[-1]) != hidden_size
                            or int(logits.shape[-1]) < donor_count
                        ):
                            raise Lfm2SparseSemanticRunnerError(
                                "Cached hidden/logit diagnostic rows are unaligned."
                            )
                        flat_hidden = hidden.reshape(-1, hidden_size)
                        parent_winners = (
                            logits[..., :donor_count]
                            .reshape(-1, donor_count)
                            .argmax(dim=-1)
                        )
                        for start in range(0, int(flat_hidden.shape[0]), chunk_size):
                            stop = min(start + chunk_size, int(flat_hidden.shape[0]))
                            hidden_chunk = flat_hidden[start:stop].to(
                                device=live_target_table.device,
                                dtype=live_target_table.dtype,
                            )
                            target_scores = routing.score_target_tokens(
                                route_session, hidden_chunk
                            )
                            donor_logits = (
                                routing.compute_all_donor_logits_from_target_scores(
                                    route_session,
                                    target_scores,
                                    maximum_donor_rows_per_chunk=4096,
                                )
                            )
                            predicted = donor_logits.argmax(dim=-1).to(device="cpu")
                            reference_winners = parent_winners[start:stop].to(
                                device="cpu"
                            )
                            reference_on_device = reference_winners.to(
                                device=donor_logits.device
                            )
                            true_scores = donor_logits.gather(
                                1, reference_on_device.unsqueeze(1)
                            ).squeeze(1)
                            donor_ids = torch.arange(
                                donor_count,
                                device=donor_logits.device,
                                dtype=torch.int64,
                            ).unsqueeze(0)
                            ranks = (
                                (donor_logits > true_scores.unsqueeze(1)).sum(dim=1)
                                + (
                                    donor_logits.eq(true_scores.unsqueeze(1))
                                    & donor_ids.lt(reference_on_device.unsqueeze(1))
                                ).sum(dim=1)
                                + 1
                            ).to(device="cpu", dtype=torch.int64)
                            predicted_scores = donor_logits.gather(
                                1,
                                predicted.to(device=donor_logits.device).unsqueeze(1),
                            ).squeeze(1)
                            logits_without_parent = donor_logits.clone()
                            logits_without_parent.scatter_(
                                1,
                                reference_on_device.unsqueeze(1),
                                -torch.inf,
                            )
                            best_other_scores = logits_without_parent.max(dim=1).values
                            best_direct_scores = (
                                donor_logits.index_select(
                                    1, protected.to(device=donor_logits.device)
                                )
                                .max(dim=1)
                                .values
                            )
                            equal = predicted.eq(reference_winners)
                            for row, winner in enumerate(reference_winners.tolist()):
                                is_direct = int(winner) in direct_donor_ids
                                predicted_id = int(predicted[row].item())
                                predicted_is_direct = predicted_id in direct_donor_ids
                                parent_reclosure = reclosure_by_donor.get(int(winner))
                                predicted_reclosure = reclosure_by_donor.get(
                                    predicted_id
                                )
                                rank = int(ranks[row].item())
                                top5 = rank <= 5
                                top10 = rank <= 10
                                reciprocal_rank = 1.0 / rank
                                matched = bool(equal[row].item())
                                parent_score = float(true_scores[row].item())
                                predicted_score = float(predicted_scores[row].item())
                                best_other_score = float(best_other_scores[row].item())
                                parent_margin = parent_score - best_other_score
                                best_direct_score = float(
                                    best_direct_scores[row].item()
                                )
                                multi_margin = (
                                    None
                                    if is_direct
                                    else parent_score - best_direct_score
                                )
                                row_rank_diagnostics.append(
                                    {
                                        "global_row_index": total,
                                        "batch_index": int(reference.index),
                                        "hidden_tensor": hidden_name,
                                        "logit_tensor": logit_name,
                                        "row_index_within_tensor": start + row,
                                        "parent_winner_donor_id": int(winner),
                                        "parent_winner_kind": (
                                            "direct" if is_direct else "multi-code"
                                        ),
                                        "predicted_winner_donor_id": predicted_id,
                                        "predicted_winner_kind": (
                                            "direct"
                                            if predicted_is_direct
                                            else "multi-code"
                                        ),
                                        "parent_winner_reclosure_provenance": (
                                            None
                                            if parent_reclosure is None
                                            else dict(parent_reclosure)
                                        ),
                                        "predicted_winner_reclosure_provenance": (
                                            None
                                            if predicted_reclosure is None
                                            else dict(predicted_reclosure)
                                        ),
                                        "parent_winner_child_rank": rank,
                                        "parent_winner_in_child_top5": top5,
                                        "parent_winner_in_child_top10": top10,
                                        "parent_winner_child_score": parent_score,
                                        "predicted_winner_child_score": predicted_score,
                                        "best_other_child_score": best_other_score,
                                        "parent_winner_margin_over_best_other_child_score": (
                                            parent_margin
                                        ),
                                        "best_protected_direct_child_score": (
                                            best_direct_score
                                        ),
                                        "parent_multi_margin_over_best_protected_direct": (
                                            multi_margin
                                        ),
                                    }
                                )
                                total += 1
                                matches += int(matched)
                                top5_matches += int(top5)
                                top10_matches += int(top10)
                                reciprocal_rank_sum += reciprocal_rank
                                parent_winner_margin_sum += parent_margin
                                parent_winner_positive_margin_count += int(
                                    parent_margin > 0.0
                                )
                                parent_winner_zero_margin_count += int(
                                    parent_margin == 0.0
                                )
                                parent_winner_negative_margin_count += int(
                                    parent_margin < 0.0
                                )
                                if parent_reclosure is not None:
                                    record_transition(
                                        "by_scale_cap_applied",
                                        (
                                            "capped"
                                            if parent_reclosure["scale_cap_applied"]
                                            else "not-capped"
                                        ),
                                        matched=matched,
                                        predicted_is_direct=predicted_is_direct,
                                        parent_margin=parent_margin,
                                    )
                                    record_transition(
                                        "by_unresolved_low_cosine",
                                        (
                                            "unresolved-low-cosine"
                                            if parent_reclosure["unresolved_low_cosine"]
                                            else "resolved-or-no-threshold"
                                        ),
                                        matched=matched,
                                        predicted_is_direct=predicted_is_direct,
                                        parent_margin=parent_margin,
                                    )
                                    record_transition(
                                        "by_before_cosine_band",
                                        cosine_band(
                                            float(parent_reclosure["before_cosine"])
                                        ),
                                        matched=matched,
                                        predicted_is_direct=predicted_is_direct,
                                        parent_margin=parent_margin,
                                    )
                                batch_total += 1
                                batch_matches += int(matched)
                                batch_top5_matches += int(top5)
                                batch_top10_matches += int(top10)
                                batch_reciprocal_rank_sum += reciprocal_rank
                                if predicted_is_direct:
                                    predicted_direct_total += 1
                                else:
                                    predicted_multi_total += 1
                                if is_direct:
                                    direct_total += 1
                                    direct_matches += int(matched)
                                    direct_top5_matches += int(top5)
                                    direct_top10_matches += int(top10)
                                    direct_reciprocal_rank_sum += reciprocal_rank
                                    confusion_key = (
                                        "parent_direct_predicted_direct"
                                        if predicted_is_direct
                                        else "parent_direct_predicted_multi"
                                    )
                                else:
                                    multi_total += 1
                                    multi_matches += int(matched)
                                    multi_top5_matches += int(top5)
                                    multi_top10_matches += int(top10)
                                    multi_reciprocal_rank_sum += reciprocal_rank
                                    confusion_key = (
                                        "parent_multi_predicted_direct"
                                        if predicted_is_direct
                                        else "parent_multi_predicted_multi"
                                    )
                                winner_type_confusion[confusion_key] += 1
                            if progress_callback is not None:
                                progress_callback(
                                    {
                                        "stage": "hidden-score-chunk-complete",
                                        "batch_index": reference.index,
                                        "hidden_tensor": hidden_name,
                                        "row_start": start,
                                        "row_stop": stop,
                                        "cumulative_rows": total,
                                        "cumulative_top1_matches": matches,
                                    }
                                )
                    batch_reports.append(
                        {
                            "batch_index": reference.index,
                            "tensor_pairs": [list(pair) for pair in pairs],
                            "row_count": batch_total,
                            "top1_match_count": batch_matches,
                            "top5_match_count": batch_top5_matches,
                            "top10_match_count": batch_top10_matches,
                            "mean_reciprocal_rank": (
                                batch_reciprocal_rank_sum / batch_total
                            ),
                        }
                    )
                finally:
                    trace_session.release_selected_tensors(payload)
        trace_session.verify()
        session_receipt = trace_session.as_dict()
    if (
        total <= 0
        or direct_total + multi_total != total
        or predicted_direct_total + predicted_multi_total != total
        or len(row_rank_diagnostics) != total
        or sum(winner_type_confusion.values()) != total
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Sealed cache exposed no aligned hidden diagnostic rows."
        )
    finalized_transition_groups: dict[str, dict[str, dict[str, Any]]] = {}
    for group, buckets in transition_groups.items():
        finalized_transition_groups[group] = {}
        for bucket, counters in sorted(buckets.items()):
            row_count = _positive_int(counters["row_count"], "transition row count")
            parent_margin_sum = float(counters.pop("parent_winner_margin_sum"))
            finalized_transition_groups[group][bucket] = {
                **counters,
                "top1_match_rate": counters["top1_match_count"] / row_count,
                "mean_parent_winner_margin_over_best_other_child_score": (
                    parent_margin_sum / row_count
                ),
            }
    body = {
        "schema": HIDDEN_DIAGNOSTIC_SCHEMA,
        "status": "complete-advisory-cached-hidden-diagnostic",
        "trace_cache_key": cache.cache_key,
        "trace_parent_identity_sha256": cache.parent_fingerprint,
        "trace_selective_session": session_receipt,
        "selected_tensor_logical_bytes": selected_tensor_bytes,
        "unselected_tensor_payloads_loaded": False,
        "cached_hidden_row_count": total,
        "cached_hidden_top1_match_count": matches,
        "cached_hidden_top1_match_rate": matches / total,
        "cached_hidden_top5_match_count": top5_matches,
        "cached_hidden_top5_match_rate": top5_matches / total,
        "cached_hidden_top10_match_count": top10_matches,
        "cached_hidden_top10_match_rate": top10_matches / total,
        "cached_hidden_mean_reciprocal_rank": reciprocal_rank_sum / total,
        "mean_parent_winner_margin_over_best_other_child_score": (
            parent_winner_margin_sum / total
        ),
        "positive_parent_winner_margin_count": parent_winner_positive_margin_count,
        "zero_parent_winner_margin_count": parent_winner_zero_margin_count,
        "negative_parent_winner_margin_count": parent_winner_negative_margin_count,
        "parent_top1_direct_anchor_count": direct_total,
        "top1_match_given_parent_direct_anchor_count": direct_matches,
        "top1_match_given_parent_direct_anchor_rate": (
            None if direct_total == 0 else direct_matches / direct_total
        ),
        "top5_match_given_parent_direct_anchor_count": direct_top5_matches,
        "top5_match_given_parent_direct_anchor_rate": (
            None if direct_total == 0 else direct_top5_matches / direct_total
        ),
        "top10_match_given_parent_direct_anchor_count": direct_top10_matches,
        "top10_match_given_parent_direct_anchor_rate": (
            None if direct_total == 0 else direct_top10_matches / direct_total
        ),
        "mean_reciprocal_rank_given_parent_direct_anchor": (
            None if direct_total == 0 else direct_reciprocal_rank_sum / direct_total
        ),
        "parent_top1_multi_code_count": multi_total,
        "top1_match_given_parent_multi_code_count": multi_matches,
        "top1_match_given_parent_multi_code_rate": (
            None if multi_total == 0 else multi_matches / multi_total
        ),
        "top5_match_given_parent_multi_code_count": multi_top5_matches,
        "top5_match_given_parent_multi_code_rate": (
            None if multi_total == 0 else multi_top5_matches / multi_total
        ),
        "top10_match_given_parent_multi_code_count": multi_top10_matches,
        "top10_match_given_parent_multi_code_rate": (
            None if multi_total == 0 else multi_top10_matches / multi_total
        ),
        "mean_reciprocal_rank_given_parent_multi_code": (
            None if multi_total == 0 else multi_reciprocal_rank_sum / multi_total
        ),
        "predicted_winner_direct_anchor_count": predicted_direct_total,
        "predicted_winner_multi_code_count": predicted_multi_total,
        "parent_winner_by_predicted_winner_kind_confusion": winner_type_confusion,
        "reclosure_row_provenance_enabled": reclosure_provenance_supplied,
        "reclosure_row_provenance_max_scale": provenance_max_scale,
        "reclosure_row_provenance_count": len(reclosure_rows),
        "reclosure_row_provenance_sha256": canonical_sha256(list(reclosure_rows)),
        "reclosure_transition_aggregates": finalized_transition_groups,
        "reclosure_transition_population": (
            "hidden-rows-whose-parent-winner-is-a-non-direct-donor"
        ),
        "before_cosine_band_boundaries": [0.0, 0.25, 0.5, 0.75, 1.0],
        "row_rank_diagnostics": row_rank_diagnostics,
        "row_rank_diagnostics_sha256": canonical_sha256(row_rank_diagnostics),
        "prior_hard_anchor_direct_winner_count": _nonnegative_int(
            expected_prior_direct_winners, "prior direct winners"
        ),
        "prior_hard_anchor_multi_code_winner_count": _nonnegative_int(
            expected_prior_multi_code_winners, "prior multi-code winners"
        ),
        "multi_code_winner_matches_gained_over_hard_anchor": multi_matches,
        "direct_vs_multi_classification_source": (
            "C-protected-target-to-donor-anchor-map"
        ),
        "S_emission_span_length_used_for_classification": False,
        "batch_reports": batch_reports,
        "metrics_are_advisory_not_an_intelligence_gate": True,
        "automatic_acceptance_or_rejection_performed": False,
        "operation_audit": {
            "parent_model_load_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "network_access": False,
        },
    }
    return {**body, "diagnostic_sha256": canonical_sha256(body)}


def assemble_sparse_semantic_manual_dossier(
    *,
    prompts: Sequence[Mapping[str, Any]],
    prompt_phases: Sequence[CachedPromptPhase],
    parent_envelope: Mapping[str, Any],
    child_state_sha256: str,
    physical_child_parameters: int,
    effective_accounting: Mapping[str, Any],
    parent_parameters: int,
    span_phase: Any,
    routing_contract_sha256: str,
    codebook_artifact_sha256: str,
    hidden_diagnostic: Mapping[str, Any] | None,
    maximum_new_tokens: int,
) -> dict[str, Any]:
    """Build a target-text-first dossier while retaining donor diagnostics."""

    effective = _positive_int(
        effective_accounting.get("effective_child_parameter_count"),
        "effective child parameters",
    )
    baseline = assemble_manual_review_dossier(
        prompts=prompts,
        prompt_phases=prompt_phases,
        parent_envelope=parent_envelope,
        child_state_sha256=child_state_sha256,
        child_unique_parameters=effective,
        parent_unique_parameters=parent_parameters,
        span_phase=span_phase,
        routing_contract_sha256=routing_contract_sha256,
        maximum_new_tokens=maximum_new_tokens,
    )
    comparisons = baseline.get("comparisons")
    if not isinstance(comparisons, list):
        raise Lfm2SparseSemanticRunnerError(
            "Baseline manual dossier has no comparisons."
        )
    for comparison in comparisons:
        response = comparison.get("child_semantic_step_response")
        if not isinstance(response, Mapping):
            raise Lfm2SparseSemanticRunnerError(
                "Manual dossier child response is missing."
            )
        if (
            not isinstance(response.get("final_target_response_text"), str)
            or not isinstance(response.get("donor_response_text"), str)
            or response.get("response") != response.get("final_target_response_text")
        ):
            raise Lfm2SparseSemanticRunnerError(
                "Manual dossier must expose authoritative target text plus donor text."
            )
        comparison["authoritative_child_target_text"] = response[
            "final_target_response_text"
        ]
        comparison["diagnostic_child_donor_text"] = response["donor_response_text"]
    body = dict(baseline)
    body.pop("dossier_sha256", None)
    body.update(
        {
            "schema": "neural-forge-lfm2-sparse-semantic-manual-dossier-v1",
            "status": "complete-awaiting-manual-target-text-intelligence-review",
            "physical_child_parameter_count": _positive_int(
                physical_child_parameters, "physical child parameters"
            ),
            "effective_parameter_accounting": _plain_json(effective_accounting),
            "child_unique_parameters": effective,
            "realized_parameter_compaction_ratio": parent_parameters / effective,
            "sparse_semantic_codebook_artifact_sha256": _digest(
                codebook_artifact_sha256, "codebook artifact SHA-256"
            ),
            "cached_hidden_diagnostic": (
                None if hidden_diagnostic is None else _plain_json(hidden_diagnostic)
            ),
            "authoritative_child_text_source": (
                "whole-target-sequence-retokenization-decode"
            ),
            "donor_decode_is_retained_as_diagnostic": True,
            "manual_whole_response_intelligence_review_is_authoritative": True,
            "cached_hidden_metrics_are_advisory": True,
            "numeric_or_binary_behavior_gate_used": False,
            "automatic_acceptance_or_rejection_performed": False,
        }
    )
    return {**body, "dossier_sha256": canonical_sha256(body)}


def run(
    *,
    run_directory: Path,
    phase_cache_directory: Path,
    checkpoint_directory: Path,
    base_codebook_directory: Path | None,
    expected_base_codebook_artifact_sha256: str | None,
    parent_model_directory: Path,
    donor_tokenizer_directory: Path,
    target_tokenizer_directory: Path,
    bridge_directory: Path,
    expected_bridge_artifact_sha256: str,
    expected_checkpoint_artifact_inventory_sha256: str | None,
    expected_child_state_sha256: str | None,
    expected_child_unique_parameters: int,
    expected_parent_identity_sha256: str,
    expected_parent_unique_parameters: int,
    prompts_json: Path,
    expected_prompts_file_sha256: str,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    complete_causal_cache_directory: Path | None,
    device: str,
    dtype: str,
    checkpoint_provenance_device: str,
    checkpoint_provenance_dtype: str,
    selection_device: str | None,
    selection_dtype: str,
    solve_device: str | None,
    solve_dtype: str,
    norm_reclosure_gamma: float | None,
    norm_reclosure_max_scale: float | None,
    norm_reclosure_minimum_cosine_to_amplify: float | None,
    norm_reclosure_device: str | None,
    norm_reclosure_dtype: str | None,
    maximum_candidates: int,
    active_k: int,
    candidate_ridge: float,
    refit_ridge: float,
    donor_rows_per_chunk: int,
    maximum_span_rows_per_chunk: int,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
    maximum_embedding_rows_per_chunk: int,
    maximum_donor_rows_per_score_chunk: int,
    maximum_hidden_rows_per_score_chunk: int,
    expected_prior_direct_winners: int,
    expected_prior_multi_code_winners: int,
    console_progress_every: int,
    support_selection: str = _TOP_COSINE_SUPPORT_SELECTION,
    omp_orthogonality_tolerance: float | None = None,
) -> dict[str, Any]:
    """Execute or resume the A1 sparse semantic bridge review end to end."""

    from neural_forge.hf_lfm2_sparse_semantic_codebook import (
        Lfm2SparseSemanticCandidateChunkCheckpoint,
        Lfm2SparseSemanticCoefficientChunkCheckpoint,
        build_lfm2_residual_omp_numeric_backend_receipt,
        compile_lfm2_sparse_semantic_codebook,
        discover_lfm2_residual_omp_semantic_candidates,
        discover_lfm2_sparse_semantic_candidates,
        load_lfm2_sparse_semantic_candidate_inventory,
        load_lfm2_sparse_semantic_codebook,
        reclose_lfm2_sparse_semantic_codebook_norms,
        save_lfm2_sparse_semantic_candidate_inventory,
        save_lfm2_sparse_semantic_codebook,
    )
    import neural_forge.hf_lfm2_sparse_semantic_codebook as sparse_core

    sparse_core_source_sha = _file_sha256(
        Path(sparse_core.__file__).resolve(strict=True)
    )
    math_abi_receipts = sparse_semantic_phase_math_abi_receipts()
    support_math_abi_name = (
        "residual_omp_support_selection"
        if _support_selection(support_selection) == _RESIDUAL_OMP_SUPPORT_SELECTION
        else "top_cosine_candidate_selection"
    )
    support_math_abi_sha = _digest(
        math_abi_receipts[support_math_abi_name]["math_abi_sha256"],
        "candidate support math ABI SHA-256",
    )
    support_checkpoint_compatibility_sha = (
        _digest(
            math_abi_receipts[support_math_abi_name]["descriptor"].get(
                "checkpoint_compatibility_sha256"
            ),
            "support checkpoint compatibility SHA-256",
        )
        if support_math_abi_name == "residual_omp_support_selection"
        else support_math_abi_sha
    )
    coefficient_math_abi_sha = _digest(
        math_abi_receipts["coefficient_ridge_refit"]["math_abi_sha256"],
        "coefficient solve math ABI SHA-256",
    )
    reclosure_math_abi_sha = _digest(
        math_abi_receipts["coefficient_norm_reclosure"]["math_abi_sha256"],
        "norm reclosure math ABI SHA-256",
    )

    run_started = time.monotonic()
    run_root = _output_directory(run_directory, "run directory")
    cache_root = _output_directory(phase_cache_directory, "phase cache directory")
    events_root = _output_directory(run_root / "events", "events directory")
    compile_console = CompileProgressReporter()
    console = ConsoleProgressReporter(
        semantic_step_every=_positive_int(
            console_progress_every, "console progress interval"
        )
    )
    checkpoint = _local_directory(checkpoint_directory, "child checkpoint")
    support_mode = _support_selection(support_selection)
    if (
        support_mode == _TOP_COSINE_SUPPORT_SELECTION
        and omp_orthogonality_tolerance is not None
    ):
        raise Lfm2SparseSemanticRunnerError(
            "omp_orthogonality_tolerance is valid only for residual-omp support."
        )
    if (
        support_mode == _RESIDUAL_OMP_SUPPORT_SELECTION
        and base_codebook_directory is not None
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Residual OMP support cannot reuse a base codebook compiled with an "
            "unsealed support-selection provenance."
        )
    if base_codebook_directory is None:
        if expected_base_codebook_artifact_sha256 is not None:
            raise Lfm2SparseSemanticRunnerError(
                "Expected base codebook seal requires a base codebook directory."
            )
        base_codebook_path = None
        expected_base_codebook_sha = None
    else:
        if expected_base_codebook_artifact_sha256 is None:
            raise Lfm2SparseSemanticRunnerError(
                "Base codebook override requires an explicit expected artifact seal."
            )
        base_codebook_path = _local_directory(
            base_codebook_directory, "base codebook directory"
        )
        expected_base_codebook_sha = _digest(
            expected_base_codebook_artifact_sha256,
            "expected base codebook artifact SHA-256",
        )
    parent_path = _local_directory(parent_model_directory, "parent model directory")
    donor_path = _local_directory(
        donor_tokenizer_directory, "donor tokenizer directory"
    )
    target_path = _local_directory(
        target_tokenizer_directory, "target tokenizer directory"
    )
    bridge_path = _local_directory(bridge_directory, "bridge directory")
    prompt_path = prompts_json.expanduser().absolute().resolve(strict=True)
    if prompt_path.is_symlink() or not prompt_path.is_file():
        raise Lfm2SparseSemanticRunnerError("Review prompt file is missing or unsafe.")
    if _file_sha256(prompt_path) != _digest(
        expected_prompts_file_sha256, "expected prompts file SHA-256"
    ):
        raise Lfm2SparseSemanticRunnerError("Review prompt bytes changed.")
    parent_identity = _digest(
        expected_parent_identity_sha256, "expected parent identity"
    )
    parent_parameters = _positive_int(
        expected_parent_unique_parameters, "expected parent unique parameters"
    )
    retained_k = _positive_int(active_k, "active_k")
    candidate_count = _positive_int(maximum_candidates, "maximum_candidates")
    if candidate_count < retained_k:
        raise Lfm2SparseSemanticRunnerError(
            "maximum_candidates must be at least active_k."
        )
    if (
        support_mode == _RESIDUAL_OMP_SUPPORT_SELECTION
        and candidate_count != retained_k
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Residual OMP requires maximum_candidates == active_k so the compiled "
            "codebook consumes the exact residual-selected support."
        )
    row_chunk = _positive_int(donor_rows_per_chunk, "donor_rows_per_chunk")

    prompt_rows = tuple(dict(row) for row in load_review_prompts(prompt_path))
    if len(prompt_rows) != 16:
        raise Lfm2SparseSemanticRunnerError(
            "Sparse semantic review requires the sealed 16-prompt review set."
        )
    prompt_set_sha256 = canonical_sha256([_plain_json(row) for row in prompt_rows])
    envelope = load_bound_parent_envelope(
        parent_envelope,
        expected_file_sha256=expected_parent_envelope_file_sha256,
        expected_envelope_sha256=expected_parent_envelope_sha256,
        expected_prompt_set_sha256=prompt_set_sha256,
        expected_prompt_ids=[str(row["id"]) for row in prompt_rows],
        expected_parent_identity_sha256=parent_identity,
        expected_parent_unique_parameters=parent_parameters,
        expected_maximum_new_tokens=maximum_new_tokens,
    )

    model_dtype, runtime_dtype_receipt = resolve_runtime_dtype(dtype, device)
    provenance_dtype = resolve_checkpoint_provenance_dtype(checkpoint_provenance_dtype)
    selection_device_value = device if selection_device is None else selection_device
    select_device, select_dtype, selection_receipt = resolve_compile_placement(
        role="candidate-selection",
        requested_device=selection_device_value,
        requested_dtype=selection_dtype,
    )
    if support_mode == _RESIDUAL_OMP_SUPPORT_SELECTION:
        resolved_omp_tolerance = (
            float(torch.finfo(select_dtype).eps * 64.0)
            if omp_orthogonality_tolerance is None
            else _finite_positive(
                omp_orthogonality_tolerance, "OMP orthogonality tolerance"
            )
        )
        omp_numeric_backend_receipt = _plain_json(
            build_lfm2_residual_omp_numeric_backend_receipt(
                select_device,
                select_dtype,
            )
        )
        omp_numeric_backend_receipt_sha = _digest(
            omp_numeric_backend_receipt.get("numeric_backend_receipt_sha256"),
            "OMP numeric backend receipt SHA-256",
        )
    else:
        resolved_omp_tolerance = None
        omp_numeric_backend_receipt = None
        omp_numeric_backend_receipt_sha = None
    solve_device_value = device if solve_device is None else solve_device
    coefficient_device, coefficient_dtype, solve_receipt = resolve_compile_placement(
        role="coefficient-solve",
        requested_device=solve_device_value,
        requested_dtype=solve_dtype,
    )
    norm_reclosure_enabled = norm_reclosure_gamma is not None
    reclosure_device: torch.device | None = None
    reclosure_dtype: torch.dtype | None = None
    reclosure_placement_receipt: Mapping[str, Any] | None = None
    if not norm_reclosure_enabled:
        if any(
            value is not None
            for value in (
                norm_reclosure_max_scale,
                norm_reclosure_minimum_cosine_to_amplify,
                norm_reclosure_device,
                norm_reclosure_dtype,
            )
        ):
            raise Lfm2SparseSemanticRunnerError(
                "Norm reclosure controls require --norm-reclosure-gamma."
            )
        normalized_reclosure_gamma = None
        normalized_reclosure_max_scale = None
        normalized_reclosure_minimum_cosine = None
    else:
        if norm_reclosure_max_scale is None:
            raise Lfm2SparseSemanticRunnerError(
                "Enabled norm reclosure requires an explicit maximum scale."
            )
        normalized_reclosure_gamma = _finite_positive(
            norm_reclosure_gamma, "norm reclosure gamma"
        )
        normalized_reclosure_max_scale = _amplification_ceiling(
            norm_reclosure_max_scale, "norm reclosure maximum scale"
        )
        normalized_reclosure_minimum_cosine = _optional_cosine(
            norm_reclosure_minimum_cosine_to_amplify,
            "minimum cosine to amplify",
        )
        reclosure_device_value = (
            str(coefficient_device)
            if norm_reclosure_device is None
            else norm_reclosure_device
        )
        reclosure_dtype_value = (
            "auto" if norm_reclosure_dtype is None else norm_reclosure_dtype
        )
        (
            reclosure_device,
            reclosure_dtype,
            reclosure_placement_receipt,
        ) = resolve_compile_placement(
            role="norm-reclosure",
            requested_device=reclosure_device_value,
            requested_dtype=reclosure_dtype_value,
        )
    reclosure_request_body = {
        "schema": "neural-forge-lfm2-norm-reclosure-run-request-v1",
        "enabled": norm_reclosure_enabled,
        "gamma": normalized_reclosure_gamma,
        "max_scale": normalized_reclosure_max_scale,
        "minimum_cosine_to_amplify": normalized_reclosure_minimum_cosine,
        "placement_receipt": (
            None
            if reclosure_placement_receipt is None
            else _plain_json(reclosure_placement_receipt)
        ),
        "external_continuous_scale_vector_requested": False,
        "coefficient_only_reclosure_requested": norm_reclosure_enabled,
    }
    reclosure_request_receipt = {
        **reclosure_request_body,
        "receipt_sha256": canonical_sha256(reclosure_request_body),
    }

    donor_tokenizer = load_lfm2_direct_local_tokenizer(donor_path)
    target_tokenizer = load_lfm2_direct_local_tokenizer(target_path)
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_path,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=_digest(
            expected_bridge_artifact_sha256,
            "expected bridge artifact SHA-256",
        ),
    )
    if (
        fingerprint_hf_tokenizer_semantics(donor_tokenizer)
        != bridge.donor_tokenizer_semantics_sha256
        or fingerprint_hf_tokenizer_semantics(target_tokenizer)
        != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Live local tokenizer semantics differ from the strict bridge."
        )

    checkpoint_inventory = _checkpoint_artifact_inventory(checkpoint)
    checkpoint_inventory_sha = canonical_sha256(checkpoint_inventory)
    expected_checkpoint_inventory = (
        None
        if expected_checkpoint_artifact_inventory_sha256 is None
        else _digest(
            expected_checkpoint_artifact_inventory_sha256,
            "expected checkpoint artifact inventory SHA-256",
        )
    )
    if (
        expected_checkpoint_inventory is not None
        and checkpoint_inventory_sha != expected_checkpoint_inventory
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Child checkpoint file inventory differs from its producing receipt."
        )
    source_inspection = inspect_parent_tied_table_source(parent_path)
    launch_body = {
        "schema": "neural-forge-lfm2-sparse-semantic-run-launch-v1",
        "status": "preflight-complete-before-child-load",
        "run_directory": str(run_root),
        "phase_cache_directory": str(cache_root),
        "checkpoint_directory": str(checkpoint),
        "checkpoint_artifact_inventory_sha256": checkpoint_inventory_sha,
        "base_codebook_override_directory": (
            None if base_codebook_path is None else str(base_codebook_path)
        ),
        "expected_base_codebook_artifact_sha256": expected_base_codebook_sha,
        "parent_model_directory": str(parent_path),
        "parent_identity_sha256": parent_identity,
        "parent_tied_table_source_inspection": source_inspection,
        "donor_tokenizer_directory": str(donor_path),
        "target_tokenizer_directory": str(target_path),
        "bridge_directory": str(bridge_path),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "prompt_file": str(prompt_path),
        "prompt_file_sha256": expected_prompts_file_sha256,
        "prompt_count": len(prompt_rows),
        "prompt_set_sha256": prompt_set_sha256,
        "runtime_dtype_receipt": runtime_dtype_receipt,
        "candidate_selection_placement_receipt": selection_receipt,
        "coefficient_solve_placement_receipt": solve_receipt,
        "norm_reclosure_request_receipt": reclosure_request_receipt,
        "support_selection": support_mode,
        "omp_orthogonality_tolerance": resolved_omp_tolerance,
        "omp_numeric_backend_receipt": omp_numeric_backend_receipt,
        "omp_numeric_backend_receipt_sha256": omp_numeric_backend_receipt_sha,
        "candidate_support_math_abi_sha256": support_math_abi_sha,
        "candidate_support_checkpoint_compatibility_sha256": (
            support_checkpoint_compatibility_sha
        ),
        "coefficient_solve_math_abi_sha256": coefficient_math_abi_sha,
        "norm_reclosure_math_abi_sha256": reclosure_math_abi_sha,
        "phase_math_abi_receipts": math_abi_receipts,
        "sparse_core_source_sha256_observational_only": sparse_core_source_sha,
        "source_hashes_used_as_expensive_math_cache_identity": False,
        "support_prefix_cache_reusable_across_active_k": (
            support_mode == _RESIDUAL_OMP_SUPPORT_SELECTION
        ),
        "maximum_candidates": candidate_count,
        "active_k": retained_k,
        "candidate_ridge": _finite_nonnegative(candidate_ridge, "candidate_ridge"),
        "refit_ridge": _finite_nonnegative(refit_ridge, "refit_ridge"),
        "donor_rows_per_chunk": row_chunk,
        "local_inputs_only": True,
        "network_access": False,
        "parent_model_execution_planned": False,
        "training_planned": False,
        "pruning_planned": False,
        "quantization_planned": False,
    }
    launch_receipt = {
        **launch_body,
        "receipt_sha256": canonical_sha256(launch_body),
    }
    _atomic_write_json(run_root / "run-launch-receipt.json", launch_receipt)
    compile_console(
        {
            "phase": "run",
            "stage": "preflight-complete",
            "details": {
                "run_launch_receipt_sha256": launch_receipt["receipt_sha256"],
                "child_compile_staging_device": "cpu",
                "requested_runtime_device": device,
                "forward_dtype": str(model_dtype).removeprefix("torch."),
            },
        }
    )
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=torch.device(checkpoint_provenance_device),
        dtype=provenance_dtype,
    )
    compile_console(
        {
            "phase": "child-checkpoint-load",
            "stage": "cpu-staging-start",
            "details": {
                "checkpoint_directory": str(checkpoint),
                "staging_device": "cpu",
                "staging_dtype": str(model_dtype).removeprefix("torch."),
            },
        }
    )
    child, portable_load = checkpoint_io.load_authenticated_portable(
        checkpoint,
        runtime_device=torch.device("cpu"),
        runtime_dtype=model_dtype,
    )
    portable_load_receipt = _plain_json(portable_load.as_dict())
    _atomic_write_json(
        run_root / "checkpoint-portable-load-receipt.json",
        portable_load_receipt,
    )
    compile_console(
        {
            "phase": "child-checkpoint-load",
            "stage": "cpu-staging-complete",
            "details": {
                "portable_load_receipt_sha256": portable_load_receipt["receipt_sha256"],
                "staged_state_sha256": portable_load_receipt["runtime_state_sha256"],
            },
        }
    )
    child.requires_grad_(False)
    child.eval()
    physical_child_parameters = _unique_parameter_count(child)
    if physical_child_parameters != _positive_int(
        expected_child_unique_parameters, "expected child unique parameters"
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Loaded child parameter count differs from its expected inventory."
        )
    initial_child_state_sha = fingerprint_lfm2_direct_local_state(child)
    legacy_state_sha = fingerprint_model_state(child)
    expected_legacy_state = (
        None
        if expected_child_state_sha256 is None
        else _digest(expected_child_state_sha256, "expected child state SHA-256")
    )
    checkpoint_authentication_body = {
        "schema": "neural-forge-lfm2-sparse-semantic-checkpoint-authentication-v1",
        "portable_load_receipt_sha256": portable_load_receipt["receipt_sha256"],
        "checkpoint_artifact_inventory_sha256": checkpoint_inventory_sha,
        "expected_checkpoint_artifact_inventory_sha256": expected_checkpoint_inventory,
        "checkpoint_artifact_inventory_matches": (
            expected_checkpoint_inventory is None
            or checkpoint_inventory_sha == expected_checkpoint_inventory
        ),
        "runtime_direct_local_state_sha256": initial_child_state_sha,
        "legacy_fold_state_sha256": legacy_state_sha,
        "legacy_expected_child_state_sha256": expected_legacy_state,
        "legacy_fold_state_matches_producing_receipt": (
            expected_legacy_state is None or legacy_state_sha == expected_legacy_state
        ),
        "legacy_fold_state_is_diagnostic_not_an_identity_gate": True,
    }
    checkpoint_authentication = {
        **checkpoint_authentication_body,
        "receipt_sha256": canonical_sha256(checkpoint_authentication_body),
    }
    _atomic_write_json(
        run_root / "checkpoint-authentication-receipt.json",
        checkpoint_authentication,
    )

    cache = DynamicFoldPhaseCache(cache_root)
    span_contract = lfm2_semantic_step_span_adapter_cache_contract(bridge)
    span_key = span_phase_key_from_contract(span_contract)
    span_started = time.monotonic()
    _console_phase_event(
        compile_console,
        phase_key=span_key,
        stage="phase-start",
    )
    span_phase = cache_or_compute_span_phase(
        cache=cache,
        events=_event_log(events_root, label="span", phase_key=span_key),
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
    _console_phase_event(
        compile_console,
        phase_key=span_key,
        stage="phase-complete",
        details={
            "cache_hit": span_phase.cache_hit,
            "completion_sha256": span_phase.completion_sha256,
        },
    )
    donor_row_count = _positive_int(
        span_phase.span_evidence.get("donor_row_count"), "span donor row count"
    )
    target_row_count = _positive_int(
        span_phase.span_evidence.get("target_row_count"), "span target row count"
    )
    live_target_table = child.get_input_embeddings().weight
    if (
        not isinstance(live_target_table, torch.Tensor)
        or live_target_table.ndim != 2
        or int(live_target_table.shape[0]) != target_row_count
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Live child target table differs from exact span dimensions."
        )
    hidden_size = int(live_target_table.shape[1])
    target_to_donor = torch.tensor(
        tuple(bridge.transport_profile.target_to_donor_row_indices),
        dtype=torch.int64,
    )
    if int(target_to_donor.numel()) != target_row_count:
        raise Lfm2SparseSemanticRunnerError(
            "Strict bridge target mapping differs from exact span target rows."
        )

    parent_table_key = build_parent_table_phase_key(
        expected_parent_identity_sha256=parent_identity,
        config_file_sha256=source_inspection["config_file_sha256"],
        selected_shard_sha256=source_inspection["selected_shard_sha256"],
        donor_row_count=donor_row_count,
        hidden_size=hidden_size,
        publication_dtype="float32",
    )
    parent_table_phase = cache_or_load_parent_tied_table(
        cache=cache,
        events=_event_log(
            events_root, label="parent-table", phase_key=parent_table_key
        ),
        phase_key=parent_table_key,
        model_directory=parent_path,
        expected_parent_identity_sha256=parent_identity,
        donor_row_count=donor_row_count,
        hidden_size=hidden_size,
        console_progress_call=compile_console,
    )
    donor_table = parent_table_phase.tensors["donor_table"]
    parent_table_sha = _tensor_sha256(donor_table)
    target_table_cpu = live_target_table.detach().to(device="cpu").contiguous().clone()
    target_table_sha = _tensor_sha256(target_table_cpu)
    target_to_donor_sha = _tensor_sha256(target_to_donor)
    parent_table_metadata = parent_table_phase.receipt.get("metadata")
    parent_source_receipt = (
        None
        if not isinstance(parent_table_metadata, Mapping)
        else parent_table_metadata.get("source_receipt")
    )
    if not isinstance(parent_source_receipt, Mapping):
        raise Lfm2SparseSemanticRunnerError(
            "Parent table phase lacks its authenticated source/storage dtype receipt."
        )
    numeric_dtype_contract = build_numeric_dtype_contract(
        parent_source_receipt=parent_source_receipt,
        portable_load_receipt=portable_load_receipt,
        runtime_dtype_receipt=runtime_dtype_receipt,
        selection_receipt=selection_receipt,
        solve_receipt=solve_receipt,
        forward_dtype=model_dtype,
        norm_reclosure_receipt=reclosure_placement_receipt,
    )
    _atomic_write_json(
        run_root / "numeric-dtype-contract.json",
        numeric_dtype_contract,
    )
    compile_console(
        {
            "phase": "numeric-dtype-contract",
            "stage": "contract-published",
            "details": numeric_dtype_contract,
        }
    )

    import neural_forge.hf_lfm2_semantic_step_model_runtime as model_runtime_core
    import neural_forge.hf_lfm2_semantic_step_token_fold as token_fold_core

    token_fold_source_sha = _file_sha256(
        Path(token_fold_core.__file__).resolve(strict=True)
    )
    model_runtime_source_sha = _file_sha256(
        Path(model_runtime_core.__file__).resolve(strict=True)
    )
    runtime_implementation_sha = canonical_sha256(
        {
            "token_fold_source_sha256": token_fold_source_sha,
            "model_runtime_source_sha256": model_runtime_source_sha,
        }
    )
    diagnostic_implementation_sha = canonical_sha256(
        {
            "runner_source_sha256": _file_sha256(Path(__file__).resolve(strict=True)),
            "token_fold_source_sha256": token_fold_source_sha,
        }
    )
    candidate_phase: CachedArtifactPhase | None = None
    base_codebook_phase: CachedArtifactPhase | None = None
    base_override_authentication: Mapping[str, Any] | None = None
    chunk_cache_audit = {
        "candidate_chunk_cache_hits": 0,
        "candidate_chunk_legacy_compatible_cache_hits": 0,
        "candidate_chunk_cache_misses": 0,
        "candidate_chunk_publications": 0,
        "candidate_support_prefix_cache_hits_by_depth": {},
        "candidate_support_prefix_cache_misses_by_depth": {},
        "candidate_support_prefix_publications_by_depth": {},
        "coefficient_chunk_cache_hits": 0,
        "coefficient_chunk_legacy_compatible_cache_hits": 0,
        "coefficient_chunk_cache_misses": 0,
        "coefficient_chunk_publications": 0,
    }
    if base_codebook_path is not None:
        assert expected_base_codebook_sha is not None
        base_codebook = load_lfm2_sparse_semantic_codebook(
            base_codebook_path,
            expected_artifact_sha256=expected_base_codebook_sha,
        )
        base_override_authentication = authenticate_base_codebook_override(
            base_codebook,
            source_directory=base_codebook_path,
            expected_artifact_sha256=expected_base_codebook_sha,
            donor_table=donor_table,
            target_table=target_table_cpu,
            target_to_donor=target_to_donor,
            maximum_candidates=maximum_candidates,
            active_k=active_k,
            candidate_ridge=candidate_ridge,
            refit_ridge=refit_ridge,
        )
        _atomic_write_json(
            run_root / "base-codebook-override-authentication.json",
            base_override_authentication,
        )
    else:
        chunk_adapter = CoreChunkPhaseAdapter(
            cache=cache,
            events_directory=events_root,
            candidate_checkpoint_type=Lfm2SparseSemanticCandidateChunkCheckpoint,
            coefficient_checkpoint_type=Lfm2SparseSemanticCoefficientChunkCheckpoint,
            support_math_abi_sha256=support_math_abi_sha,
            coefficient_math_abi_sha256=coefficient_math_abi_sha,
            console_progress_call=compile_console,
        )
        if support_mode == _RESIDUAL_OMP_SUPPORT_SELECTION:
            assert resolved_omp_tolerance is not None
            assert omp_numeric_backend_receipt_sha is not None
            candidate_key = build_residual_omp_candidate_inventory_phase_key(
                parent_table_sha256=parent_table_sha,
                target_table_sha256=target_table_sha,
                target_to_donor_sha256=target_to_donor_sha,
                active_k=retained_k,
                orthogonality_tolerance=resolved_omp_tolerance,
                numeric_backend_receipt_sha256=(omp_numeric_backend_receipt_sha),
                donor_rows_per_chunk=row_chunk,
                selection_placement_receipt_sha256=(
                    selection_receipt["receipt_sha256"]
                ),
                algorithm_sha256=support_math_abi_sha,
            )
            legacy_candidate_key = build_residual_omp_candidate_inventory_phase_key(
                parent_table_sha256=parent_table_sha,
                target_table_sha256=target_table_sha,
                target_to_donor_sha256=target_to_donor_sha,
                active_k=retained_k,
                orthogonality_tolerance=resolved_omp_tolerance,
                numeric_backend_receipt_sha256=(omp_numeric_backend_receipt_sha),
                donor_rows_per_chunk=row_chunk,
                selection_placement_receipt_sha256=(
                    selection_receipt["receipt_sha256"]
                ),
                algorithm_sha256=(_LEGACY_CORE_MATH_CHECKPOINT_COMPATIBILITY_SHA256),
            )
        else:
            candidate_key = build_candidate_inventory_phase_key(
                parent_table_sha256=parent_table_sha,
                target_table_sha256=target_table_sha,
                target_to_donor_sha256=target_to_donor_sha,
                maximum_candidates=maximum_candidates,
                donor_rows_per_chunk=donor_rows_per_chunk,
                selection_placement_receipt_sha256=(
                    selection_receipt["receipt_sha256"]
                ),
                algorithm_sha256=support_math_abi_sha,
            )
            legacy_candidate_key = build_candidate_inventory_phase_key(
                parent_table_sha256=parent_table_sha,
                target_table_sha256=target_table_sha,
                target_to_donor_sha256=target_to_donor_sha,
                maximum_candidates=maximum_candidates,
                donor_rows_per_chunk=donor_rows_per_chunk,
                selection_placement_receipt_sha256=(
                    selection_receipt["receipt_sha256"]
                ),
                algorithm_sha256=(_LEGACY_CORE_MATH_CHECKPOINT_COMPATIBILITY_SHA256),
            )
        candidate_started = time.monotonic()

        def report_candidate_progress(
            progress: Callable[[Mapping[str, Any]], None],
            record: Mapping[str, Any],
        ) -> None:
            progress(
                {
                    "support_selection": support_mode,
                    "omp_orthogonality_tolerance": resolved_omp_tolerance,
                    "core_progress": _plain_json(record),
                    "telemetry": runtime_telemetry(
                        started_at=candidate_started,
                        device=str(select_device),
                        requested_dtype=selection_dtype,
                        resolved_dtype=str(select_dtype).removeprefix("torch."),
                    ),
                }
            )

        def discover_candidates(
            progress: Callable[[Mapping[str, Any]], None],
        ) -> Any:
            common = {
                "donor_rows_per_chunk": row_chunk,
                "selection_device": select_device,
                "selection_dtype": select_dtype,
                "progress_callback": lambda record: report_candidate_progress(
                    progress, record
                ),
                "candidate_chunk_loader": chunk_adapter.load_candidate,
                "candidate_chunk_publisher": chunk_adapter.publish_candidate,
            }
            if support_mode == _RESIDUAL_OMP_SUPPORT_SELECTION:
                assert resolved_omp_tolerance is not None
                return discover_lfm2_residual_omp_semantic_candidates(
                    donor_table,
                    target_table_cpu,
                    target_to_donor,
                    active_k=retained_k,
                    orthogonality_tolerance=resolved_omp_tolerance,
                    implementation_sha256=support_checkpoint_compatibility_sha,
                    **common,
                )
            return discover_lfm2_sparse_semantic_candidates(
                donor_table,
                target_table_cpu,
                target_to_donor,
                maximum_candidates=candidate_count,
                **common,
            )

        candidate_phase = cache_or_compute_artifact_phase(
            cache=cache,
            events=_event_log(
                events_root, label="candidate-inventory", phase_key=candidate_key
            ),
            phase_key=candidate_key,
            compute_call=discover_candidates,
            save_call=lambda artifact, destination: (
                save_lfm2_sparse_semantic_candidate_inventory(artifact, destination)
            ),
            load_call=lambda destination: load_lfm2_sparse_semantic_candidate_inventory(
                destination
            ),
            artifact_sha256_call=lambda artifact: artifact.artifact_sha256,
            console_progress_call=compile_console,
            compatible_phase_keys=(legacy_candidate_key,),
        )

        codebook_key = build_codebook_phase_key(
            parent_table_sha256=parent_table_sha,
            target_table_sha256=target_table_sha,
            target_to_donor_sha256=target_to_donor_sha,
            candidate_content_sha256=candidate_phase.artifact.content_sha256,
            active_k=active_k,
            candidate_ridge=candidate_ridge,
            refit_ridge=refit_ridge,
            donor_rows_per_chunk=donor_rows_per_chunk,
            solve_placement_receipt_sha256=solve_receipt["receipt_sha256"],
            algorithm_sha256=coefficient_math_abi_sha,
        )
        legacy_codebook_key = build_codebook_phase_key(
            parent_table_sha256=parent_table_sha,
            target_table_sha256=target_table_sha,
            target_to_donor_sha256=target_to_donor_sha,
            candidate_content_sha256=candidate_phase.artifact.content_sha256,
            active_k=active_k,
            candidate_ridge=candidate_ridge,
            refit_ridge=refit_ridge,
            donor_rows_per_chunk=donor_rows_per_chunk,
            solve_placement_receipt_sha256=solve_receipt["receipt_sha256"],
            algorithm_sha256=(_LEGACY_CORE_MATH_CHECKPOINT_COMPATIBILITY_SHA256),
        )
        coefficient_started = time.monotonic()
        base_codebook_phase = cache_or_compute_artifact_phase(
            cache=cache,
            events=_event_log(events_root, label="codebook", phase_key=codebook_key),
            phase_key=codebook_key,
            compute_call=lambda progress: compile_lfm2_sparse_semantic_codebook(
                donor_table,
                target_table_cpu,
                target_to_donor,
                candidate_inventory=candidate_phase.artifact,
                maximum_candidates=_positive_int(
                    maximum_candidates, "maximum_candidates"
                ),
                active_k=_positive_int(active_k, "active_k"),
                candidate_ridge=_finite_nonnegative(candidate_ridge, "candidate_ridge"),
                refit_ridge=_finite_nonnegative(refit_ridge, "refit_ridge"),
                donor_rows_per_chunk=_positive_int(
                    donor_rows_per_chunk, "donor_rows_per_chunk"
                ),
                selection_device=select_device,
                selection_dtype=select_dtype,
                solve_device=coefficient_device,
                solve_dtype=coefficient_dtype,
                progress_callback=lambda record: progress(
                    {
                        "core_progress": _plain_json(record),
                        "telemetry": runtime_telemetry(
                            started_at=coefficient_started,
                            device=str(coefficient_device),
                            requested_dtype=solve_dtype,
                            resolved_dtype=str(coefficient_dtype).removeprefix(
                                "torch."
                            ),
                        ),
                    }
                ),
                coefficient_chunk_loader=chunk_adapter.load_coefficient,
                coefficient_chunk_publisher=chunk_adapter.publish_coefficient,
            ),
            save_call=lambda artifact, destination: save_lfm2_sparse_semantic_codebook(
                artifact, destination
            ),
            load_call=lambda destination: load_lfm2_sparse_semantic_codebook(
                destination
            ),
            artifact_sha256_call=lambda artifact: artifact.artifact_sha256,
            console_progress_call=compile_console,
            compatible_phase_keys=(legacy_codebook_key,),
        )
        chunk_adapter.assert_drained()
        chunk_cache_audit = dict(chunk_adapter.audit)
        base_codebook = base_codebook_phase.artifact

    candidate_inventory_receipt = (
        None
        if candidate_phase is None
        else _plain_json(candidate_phase.artifact.as_dict())
    )
    artifact_support_implementation_sha = (
        None
        if candidate_inventory_receipt is None
        else candidate_inventory_receipt.get("implementation_sha256")
    )
    artifact_numeric_backend_receipt = (
        None
        if candidate_inventory_receipt is None
        else candidate_inventory_receipt.get("numeric_backend_receipt")
    )
    artifact_numeric_backend_receipt_sha = (
        None
        if candidate_inventory_receipt is None
        else candidate_inventory_receipt.get("numeric_backend_receipt_sha256")
    )
    if (
        support_mode == _RESIDUAL_OMP_SUPPORT_SELECTION
        and artifact_support_implementation_sha != support_checkpoint_compatibility_sha
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Residual OMP candidate inventory implementation seal differs from "
            "the resolved support-selection math ABI."
        )
    if support_mode == _RESIDUAL_OMP_SUPPORT_SELECTION and (
        artifact_numeric_backend_receipt_sha != omp_numeric_backend_receipt_sha
        or artifact_numeric_backend_receipt != omp_numeric_backend_receipt
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Residual OMP candidate inventory numeric backend seal differs from "
            "the resolved selection backend."
        )
    candidate_support_cache_audit = {
        "support_selection": support_mode,
        "resolved_omp_orthogonality_tolerance": resolved_omp_tolerance,
        "math_abi_sha256": support_math_abi_sha,
        "checkpoint_compatibility_sha256": support_checkpoint_compatibility_sha,
        "implementation_sha256": support_checkpoint_compatibility_sha,
        "source_sha256_observational_only": sparse_core_source_sha,
        "numeric_backend_receipt": omp_numeric_backend_receipt,
        "numeric_backend_receipt_sha256": omp_numeric_backend_receipt_sha,
        "artifact_numeric_backend_receipt": artifact_numeric_backend_receipt,
        "artifact_numeric_backend_receipt_sha256": (
            artifact_numeric_backend_receipt_sha
        ),
        "artifact_numeric_backend_matches_resolved_selection_backend": (
            None
            if candidate_inventory_receipt is None
            or support_mode != _RESIDUAL_OMP_SUPPORT_SELECTION
            else artifact_numeric_backend_receipt_sha == omp_numeric_backend_receipt_sha
            and artifact_numeric_backend_receipt == omp_numeric_backend_receipt
        ),
        "artifact_implementation_sha256": artifact_support_implementation_sha,
        "artifact_implementation_matches_checkpoint_compatibility": (
            None
            if candidate_inventory_receipt is None
            or support_mode != _RESIDUAL_OMP_SUPPORT_SELECTION
            else artifact_support_implementation_sha
            == support_checkpoint_compatibility_sha
        ),
        "candidate_inventory_outer_phase_cache_hit": (
            None if candidate_phase is None else candidate_phase.cache_hit
        ),
        "current_run_chunk_cache": _plain_json(chunk_cache_audit),
        "artifact_support_prefix_reusable_across_active_k": (
            None
            if candidate_inventory_receipt is None
            else candidate_inventory_receipt.get(
                "support_prefix_cache_reusable_across_active_k"
            )
        ),
        "artifact_checkpoint_fully_reused_chunk_count": (
            None
            if candidate_inventory_receipt is None
            else candidate_inventory_receipt.get("checkpoint_fully_reused_chunk_count")
        ),
        "artifact_checkpoint_prefix_extended_chunk_count": (
            None
            if candidate_inventory_receipt is None
            else candidate_inventory_receipt.get(
                "checkpoint_prefix_extended_chunk_count"
            )
        ),
        "artifact_checkpoint_loader_invocation_count": (
            None
            if candidate_inventory_receipt is None
            else candidate_inventory_receipt.get("checkpoint_loader_invocation_count")
        ),
        "artifact_checkpoint_publisher_invocation_count": (
            None
            if candidate_inventory_receipt is None
            else candidate_inventory_receipt.get(
                "checkpoint_publisher_invocation_count"
            )
        ),
        "artifact_analytic_operation_audit": (
            None
            if candidate_inventory_receipt is None
            else candidate_inventory_receipt.get("analytic_operation_audit")
        ),
        "base_codebook_override_used": base_override_authentication is not None,
    }

    base_tensor_content = sparse_codebook_tensor_content(base_codebook)
    norm_reclosure_phase: CachedArtifactPhase | None = None
    reclosure_receipt: Mapping[str, Any] | None = None
    reclosure_row_diagnostics: tuple[dict[str, Any], ...] = tuple()
    reclosure_row_diagnostics_sha = canonical_sha256([])
    reclosure_row_diagnostics_path: Path | None = None
    if norm_reclosure_enabled:
        assert normalized_reclosure_gamma is not None
        assert normalized_reclosure_max_scale is not None
        assert reclosure_device is not None
        assert reclosure_dtype is not None
        norm_reclosure_key = build_norm_reclosure_phase_key(
            base_tensor_content=base_tensor_content,
            base_codebook_artifact_sha256=base_codebook.artifact_sha256,
            donor_table_sha256=parent_table_sha,
            target_table_sha256=target_table_sha,
            target_to_donor_sha256=target_to_donor_sha,
            gamma=normalized_reclosure_gamma,
            max_scale=normalized_reclosure_max_scale,
            minimum_cosine_to_amplify=normalized_reclosure_minimum_cosine,
            donor_rows_per_chunk=donor_rows_per_chunk,
            compute_device=str(reclosure_device),
            compute_dtype=str(reclosure_dtype).removeprefix("torch."),
            algorithm_sha256=reclosure_math_abi_sha,
        )
        legacy_norm_reclosure_key = build_norm_reclosure_phase_key(
            base_tensor_content=base_tensor_content,
            base_codebook_artifact_sha256=base_codebook.artifact_sha256,
            donor_table_sha256=parent_table_sha,
            target_table_sha256=target_table_sha,
            target_to_donor_sha256=target_to_donor_sha,
            gamma=normalized_reclosure_gamma,
            max_scale=normalized_reclosure_max_scale,
            minimum_cosine_to_amplify=normalized_reclosure_minimum_cosine,
            donor_rows_per_chunk=donor_rows_per_chunk,
            compute_device=str(reclosure_device),
            compute_dtype=str(reclosure_dtype).removeprefix("torch."),
            algorithm_sha256=(_LEGACY_CORE_MATH_CHECKPOINT_COMPATIBILITY_SHA256),
        )
        reclosure_started = time.monotonic()
        norm_reclosure_phase = cache_or_compute_artifact_phase(
            cache=cache,
            events=_event_log(
                events_root,
                label="norm-reclosure",
                phase_key=norm_reclosure_key,
            ),
            phase_key=norm_reclosure_key,
            compute_call=lambda progress: reclose_lfm2_sparse_semantic_codebook_norms(
                base_codebook,
                donor_table,
                target_table_cpu,
                gamma=normalized_reclosure_gamma,
                max_scale=normalized_reclosure_max_scale,
                minimum_cosine_to_amplify=(normalized_reclosure_minimum_cosine),
                donor_rows_per_chunk=row_chunk,
                compute_device=reclosure_device,
                compute_dtype=reclosure_dtype,
                progress_callback=lambda record: progress(
                    {
                        "core_progress": _plain_json(record),
                        "telemetry": runtime_telemetry(
                            started_at=reclosure_started,
                            device=str(reclosure_device),
                            requested_dtype=(
                                "auto"
                                if norm_reclosure_dtype is None
                                else norm_reclosure_dtype
                            ),
                            resolved_dtype=str(reclosure_dtype).removeprefix("torch."),
                        ),
                    }
                ),
            ),
            save_call=lambda artifact, destination: save_lfm2_sparse_semantic_codebook(
                artifact, destination
            ),
            load_call=lambda destination: load_lfm2_sparse_semantic_codebook(
                destination
            ),
            artifact_sha256_call=lambda artifact: artifact.artifact_sha256,
            console_progress_call=compile_console,
            compatible_phase_keys=(legacy_norm_reclosure_key,),
        )
        codebook = norm_reclosure_phase.artifact
        reclosure_receipt = codebook.as_dict().get("norm_reclosure")
        if (
            not torch.equal(
                codebook.semantic_code_offsets,
                base_codebook.semantic_code_offsets,
            )
            or not torch.equal(
                codebook.semantic_code_target_ids,
                base_codebook.semantic_code_target_ids,
            )
            or not torch.equal(codebook.target_to_donor, base_codebook.target_to_donor)
            or int(codebook.semantic_code_coefficients.numel())
            != int(base_codebook.semantic_code_coefficients.numel())
            or not isinstance(reclosure_receipt, Mapping)
            or reclosure_receipt.get("continuous_coefficient_count_added") != 0
            or reclosure_receipt.get("effective_child_parameter_count_added") != 0
            or reclosure_receipt.get("protected_direct_rows_preserved_bit_exact")
            is not True
        ):
            raise Lfm2SparseSemanticRunnerError(
                "Norm reclosure changed sparse structure, count, or direct anchors."
            )
        (
            reclosure_row_diagnostics,
            reclosure_row_diagnostics_sha,
        ) = authenticated_reclosure_row_diagnostics(reclosure_receipt)
        reclosure_sidecar_body = {
            "schema": "neural-forge-lfm2-norm-reclosure-row-sidecar-v1",
            "base_codebook_artifact_sha256": base_codebook.artifact_sha256,
            "reclosed_codebook_artifact_sha256": codebook.artifact_sha256,
            "row_reclosure_diagnostics": list(reclosure_row_diagnostics),
            "row_reclosure_diagnostics_sha256": reclosure_row_diagnostics_sha,
            "row_count": len(reclosure_row_diagnostics),
        }
        reclosure_sidecar = {
            **reclosure_sidecar_body,
            "sidecar_sha256": canonical_sha256(reclosure_sidecar_body),
        }
        reclosure_row_diagnostics_path = (
            run_root / "norm-reclosure-row-diagnostics.json"
        )
        _atomic_write_json(reclosure_row_diagnostics_path, reclosure_sidecar)
    else:
        codebook = base_codebook
    compile_console(
        {
            "phase": "analytic-compilation",
            "stage": "complete-before-child-runtime-relocation",
            "details": {
                "child_device": "cpu",
                "child_state_sha256": fingerprint_lfm2_direct_local_state(child),
                "selected_codebook_artifact_sha256": codebook.artifact_sha256,
            },
        }
    )
    runtime_relocation_receipt = relocate_child_for_runtime(
        child,
        runtime_device=torch.device(device),
        forward_dtype=model_dtype,
        expected_state_sha256=initial_child_state_sha,
    )
    _atomic_write_json(
        run_root / "child-runtime-relocation-receipt.json",
        runtime_relocation_receipt,
    )
    compile_console(
        {
            "phase": "child-runtime-relocation",
            "stage": "complete",
            "details": runtime_relocation_receipt,
        }
    )
    live_target_table = child.get_input_embeddings().weight
    if (
        not isinstance(live_target_table, torch.Tensor)
        or _tensor_sha256(live_target_table) != target_table_sha
        or live_target_table.dtype != model_dtype
        or live_target_table.device.type != torch.device(device).type
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Relocated live target table differs from the compiled CPU table."
        )
    routing = bind_lfm2_sparse_semantic_code_runtime(
        span_phase.span_evidence,
        live_target_table,
        semantic_code_offsets=codebook.semantic_code_offsets,
        semantic_code_target_ids=codebook.semantic_code_target_ids,
        semantic_code_coefficients=codebook.semantic_code_coefficients,
        target_to_donor=codebook.target_to_donor,
    )
    routing_receipt = _plain_json(routing.as_dict())
    routing_contract_sha = canonical_sha256(routing_receipt)
    accounting = effective_child_parameter_accounting(
        physical_child_parameters=physical_child_parameters,
        semantic_code_coefficients=codebook.semantic_code_coefficients,
        parent_parameters=parent_parameters,
    )
    if (
        routing_receipt.get("continuous_semantic_code_coefficient_count")
        != accounting["continuous_semantic_code_coefficient_count"]
    ):
        raise Lfm2SparseSemanticRunnerError(
            "Runtime and accounting disagree on continuous C coefficients."
        )

    diagnostic_phase: CachedJsonPhase | None = None
    if complete_causal_cache_directory is not None:
        trace_cache = Lfm2CircuitTraceCache.open(
            complete_causal_cache_directory, verify=False
        )
        trace_manifest = trace_cache.manifest
        diagnostic_key = build_hidden_diagnostic_phase_key(
            trace_cache_key=trace_cache.cache_key,
            trace_manifest_sha256=_digest(
                trace_manifest.get("manifest_sha256"),
                "trace manifest SHA-256",
            ),
            routing_contract_sha256=routing_contract_sha,
            target_table_sha256=target_table_sha,
            diagnostic_implementation_sha256=diagnostic_implementation_sha,
            reclosure_row_diagnostics_sha256=reclosure_row_diagnostics_sha,
            reclosure_provenance_enabled=norm_reclosure_enabled,
            reclosure_max_scale=normalized_reclosure_max_scale,
            maximum_hidden_rows_per_score_chunk=(maximum_hidden_rows_per_score_chunk),
        )
        diagnostic_phase = cache_or_compute_json_phase(
            cache=cache,
            events=_event_log(
                events_root, label="hidden-diagnostic", phase_key=diagnostic_key
            ),
            phase_key=diagnostic_key,
            compute_call=lambda progress: evaluate_cached_hidden_top1_recovery(
                trace_cache_directory=complete_causal_cache_directory,
                expected_parent_identity_sha256=parent_identity,
                routing=routing,
                live_target_table=live_target_table,
                protected_target_to_donor=codebook.target_to_donor,
                row_reclosure_diagnostics=(
                    reclosure_row_diagnostics if norm_reclosure_enabled else None
                ),
                row_reclosure_max_scale=normalized_reclosure_max_scale,
                maximum_hidden_rows_per_score_chunk=(
                    maximum_hidden_rows_per_score_chunk
                ),
                expected_prior_direct_winners=expected_prior_direct_winners,
                expected_prior_multi_code_winners=(expected_prior_multi_code_winners),
                progress_callback=progress,
            ),
            console_progress_call=compile_console,
        )
        _atomic_write_json(
            run_root / "cached-hidden-diagnostic.json",
            diagnostic_phase.document,
        )

    prompt_phases: list[CachedPromptPhase] = []
    for prompt in prompt_rows:
        prompt_key = bind_phase_implementation(
            build_prompt_phase_key(
                child_state_sha256=initial_child_state_sha,
                routing_contract_sha256=routing_contract_sha,
                bridge_artifact_sha256=bridge.artifact_sha256,
                donor_tokenizer_semantics_sha256=(
                    bridge.donor_tokenizer_semantics_sha256
                ),
                target_tokenizer_semantics_sha256=(
                    bridge.target_tokenizer_semantics_sha256
                ),
                prompt=prompt,
                device=device,
                dtype_receipt=runtime_dtype_receipt,
                maximum_prompt_tokens=maximum_prompt_tokens,
                maximum_new_tokens=maximum_new_tokens,
                maximum_embedding_rows_per_chunk=maximum_embedding_rows_per_chunk,
                maximum_donor_rows_per_score_chunk=maximum_donor_rows_per_score_chunk,
            ),
            dependency_name="runtime_implementation_sha256",
            implementation_sha256=runtime_implementation_sha,
        )
        prompt_id = str(prompt["id"])
        prompt_label = (
            "prompt-" + hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()[:12]
        )
        prompt_started = time.monotonic()
        _console_phase_event(
            compile_console,
            phase_key=prompt_key,
            stage="phase-start",
            details={"prompt_id": prompt_id},
        )
        phase = cache_or_compute_prompt_phase(
            cache=cache,
            events=_event_log(events_root, label=prompt_label, phase_key=prompt_key),
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
                requested_dtype=dtype,
                resolved_dtype=str(runtime_dtype_receipt["resolved_dtype"]),
            ),
            console_progress_call=console.runtime,
        )
        _console_phase_event(
            compile_console,
            phase_key=prompt_key,
            stage="phase-complete",
            details={
                "prompt_id": prompt_id,
                "cache_hit": phase.cache_hit,
                "completion_sha256": phase.completion_sha256,
            },
        )
        if (
            phase.runtime_receipt.get("effective_child_parameter_count")
            != accounting["effective_child_parameter_count"]
        ):
            raise Lfm2SparseSemanticRunnerError(
                "Prompt runtime did not count C in effective child parameters."
            )
        prompt_phases.append(phase)

    final_child_state_sha = fingerprint_lfm2_direct_local_state(child)
    if final_child_state_sha != initial_child_state_sha:
        raise Lfm2SparseSemanticRunnerError(
            "Physical child state changed during analytic compile/review."
        )
    if any(parameter.grad is not None for parameter in child.parameters()):
        raise Lfm2SparseSemanticRunnerError(
            "Child accumulated gradients during fold-only review."
        )
    dossier = assemble_sparse_semantic_manual_dossier(
        prompts=prompt_rows,
        prompt_phases=prompt_phases,
        parent_envelope=envelope,
        child_state_sha256=initial_child_state_sha,
        physical_child_parameters=physical_child_parameters,
        effective_accounting=accounting,
        parent_parameters=parent_parameters,
        span_phase=span_phase,
        routing_contract_sha256=routing_contract_sha,
        codebook_artifact_sha256=codebook.artifact_sha256,
        hidden_diagnostic=(
            None if diagnostic_phase is None else diagnostic_phase.document
        ),
        maximum_new_tokens=maximum_new_tokens,
    )
    dossier_path = run_root / "whole-response-dossier.json"
    _atomic_write_json(dossier_path, dossier)

    selected_codebook_receipt = _plain_json(codebook.as_dict())
    reclosure_receipt_summary = (
        None if reclosure_receipt is None else dict(_plain_json(reclosure_receipt))
    )
    if reclosure_receipt_summary is not None:
        reclosure_receipt_summary.pop("row_reclosure_diagnostics", None)
        reclosure_receipt_summary.update(
            {
                "row_reclosure_diagnostics_externalized": True,
                "row_reclosure_diagnostics_path": str(reclosure_row_diagnostics_path),
                "row_reclosure_diagnostics_sha256": (reclosure_row_diagnostics_sha),
                "row_reclosure_diagnostics_count": len(reclosure_row_diagnostics),
            }
        )
    if candidate_phase is None:
        if base_override_authentication is None:
            raise Lfm2SparseSemanticRunnerError(
                "Missing candidate provenance for the active base codebook."
            )
        candidate_content_sha = _digest(
            base_override_authentication.get("candidate_content_sha256"),
            "base candidate content SHA-256",
        )
    else:
        candidate_content_sha = _digest(
            candidate_phase.artifact.content_sha256,
            "candidate inventory content SHA-256",
        )
    selected_codebook_phase = (
        norm_reclosure_phase
        if norm_reclosure_phase is not None
        else base_codebook_phase
    )
    selected_phase_directory = (
        None
        if selected_codebook_phase is None
        else str(
            cache.entry_path(selected_codebook_phase.phase_key) / "payload" / "artifact"
        )
    )
    selected_artifact_directory = (
        selected_phase_directory
        if selected_phase_directory is not None
        else str(base_codebook_path)
    )
    base_phase_directory = (
        str(base_codebook_path)
        if base_codebook_phase is None
        else str(
            cache.entry_path(base_codebook_phase.phase_key) / "payload" / "artifact"
        )
    )
    base_reference_body = {
        "schema": "neural-forge-lfm2-sparse-semantic-base-codebook-reference-v1",
        "support_selection": support_mode,
        "omp_orthogonality_tolerance": resolved_omp_tolerance,
        "source_kind": (
            "explicit-authenticated-artifact"
            if base_codebook_phase is None
            else "phase-cache-compile"
        ),
        "artifact_sha256": _digest(
            base_codebook.artifact_sha256, "base codebook artifact SHA-256"
        ),
        "tensor_content": base_tensor_content,
        "candidate_content_sha256": candidate_content_sha,
        "phase_key_sha256": (
            None
            if base_codebook_phase is None
            else base_codebook_phase.phase_key.phase_key_sha256
        ),
        "phase_completion_sha256": (
            None
            if base_codebook_phase is None
            else base_codebook_phase.completion_sha256
        ),
        "phase_cache_hit": (
            None if base_codebook_phase is None else base_codebook_phase.cache_hit
        ),
        "artifact_directory": base_phase_directory,
        "override_authentication_receipt_sha256": (
            None
            if base_override_authentication is None
            else base_override_authentication["receipt_sha256"]
        ),
    }
    base_reference = {
        **base_reference_body,
        "reference_sha256": canonical_sha256(base_reference_body),
    }
    _atomic_write_json(run_root / "base-codebook-reference.json", base_reference)

    codebook_reference_body = {
        "schema": "neural-forge-lfm2-sparse-semantic-codebook-reference-v1",
        "support_selection": support_mode,
        "omp_orthogonality_tolerance": resolved_omp_tolerance,
        "artifact_sha256": _digest(
            codebook.artifact_sha256, "selected codebook artifact SHA-256"
        ),
        "source_kind": (
            "norm-reclosure-phase"
            if norm_reclosure_phase is not None
            else base_reference["source_kind"]
        ),
        "phase_key_sha256": (
            None
            if selected_codebook_phase is None
            else selected_codebook_phase.phase_key.phase_key_sha256
        ),
        "phase_completion_sha256": (
            None
            if selected_codebook_phase is None
            else selected_codebook_phase.completion_sha256
        ),
        "phase_cache_hit": (
            None
            if selected_codebook_phase is None
            else selected_codebook_phase.cache_hit
        ),
        "phase_cache_artifact_directory": selected_phase_directory,
        "artifact_directory": selected_artifact_directory,
        "base_codebook_reference_sha256": base_reference["reference_sha256"],
        "norm_reclosure_enabled": norm_reclosure_enabled,
        "norm_reclosure_sha256": selected_codebook_receipt.get("norm_reclosure_sha256"),
        "row_reclosure_diagnostics_path": (
            None
            if reclosure_row_diagnostics_path is None
            else str(reclosure_row_diagnostics_path)
        ),
        "row_reclosure_diagnostics_sha256": (
            reclosure_row_diagnostics_sha if norm_reclosure_enabled else None
        ),
        "row_reclosure_diagnostics_count": (
            len(reclosure_row_diagnostics) if norm_reclosure_enabled else 0
        ),
        "continuous_coefficient_count": accounting[
            "continuous_semantic_code_coefficient_count"
        ],
    }
    codebook_reference = {
        **codebook_reference_body,
        "reference_sha256": canonical_sha256(codebook_reference_body),
    }
    _atomic_write_json(run_root / "codebook-reference.json", codebook_reference)

    prompt_hits = sum(int(phase.cache_hit) for phase in prompt_phases)
    artifact_audit = _sum_runtime_operation_audits(prompt_phases)
    body = {
        "schema": RUN_SCHEMA,
        "status": (
            f"complete-k{_positive_int(active_k, 'active_k')}"
            + (
                "-residual-omp"
                if support_mode == _RESIDUAL_OMP_SUPPORT_SELECTION
                else ""
            )
            + ("-norm-reclosed" if norm_reclosure_enabled else "")
            + "-dual-channel-sparse-semantic-manual-review"
        ),
        "run_directory": str(run_root),
        "phase_cache_directory": str(cache_root),
        "run_launch_receipt_sha256": launch_receipt["receipt_sha256"],
        "elapsed_seconds": max(0.0, time.monotonic() - run_started),
        "parent_identity_sha256": parent_identity,
        "parent_model_directory": str(parent_path),
        "parent_unique_parameters": parent_parameters,
        "parent_tied_table_source_inspection": source_inspection,
        "parent_table_phase_key_sha256": parent_table_key.phase_key_sha256,
        "parent_table_phase_completion_sha256": (parent_table_phase.completion_sha256),
        "parent_table_phase_cache_hit": parent_table_phase.cache_hit,
        "parent_table_sha256": parent_table_sha,
        "parent_defined_rows_loaded": donor_row_count,
        "parent_undefined_capacity_rows_loaded": 0,
        "parent_unrelated_tensors_loaded": 0,
        "checkpoint_directory": str(checkpoint),
        "checkpoint_authentication": checkpoint_authentication,
        "checkpoint_artifact_inventory_sha256": checkpoint_inventory_sha,
        "child_cpu_compile_staging_receipt": portable_load_receipt,
        "child_runtime_relocation_receipt": runtime_relocation_receipt,
        "child_remained_on_cpu_through_analytic_compilation": True,
        "runtime_initial_child_state_sha256": initial_child_state_sha,
        "runtime_final_child_state_sha256": final_child_state_sha,
        "runtime_child_state_unchanged": True,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "span_evidence_sha256": span_phase.span_evidence["evidence_sha256"],
        "span_phase_key_sha256": span_phase.phase_key.phase_key_sha256,
        "span_phase_completion_sha256": span_phase.completion_sha256,
        "span_phase_cache_hit": span_phase.cache_hit,
        "candidate_inventory_artifact_sha256": (
            None if candidate_phase is None else candidate_phase.artifact_sha256
        ),
        "candidate_inventory_content_sha256": candidate_content_sha,
        "candidate_inventory_phase_key_sha256": (
            None
            if candidate_phase is None
            else candidate_phase.phase_key.phase_key_sha256
        ),
        "candidate_inventory_completion_sha256": (
            None if candidate_phase is None else candidate_phase.completion_sha256
        ),
        "candidate_inventory_cache_hit": (
            None if candidate_phase is None else candidate_phase.cache_hit
        ),
        "support_selection": support_mode,
        "omp_orthogonality_tolerance": resolved_omp_tolerance,
        "omp_numeric_backend_receipt_sha256": omp_numeric_backend_receipt_sha,
        "candidate_support_math_abi_sha256": support_math_abi_sha,
        "candidate_support_checkpoint_compatibility_sha256": (
            support_checkpoint_compatibility_sha
        ),
        "coefficient_solve_math_abi_sha256": coefficient_math_abi_sha,
        "norm_reclosure_math_abi_sha256": reclosure_math_abi_sha,
        "phase_math_abi_receipts": math_abi_receipts,
        "sparse_core_source_sha256_observational_only": sparse_core_source_sha,
        "candidate_support_cache_audit": candidate_support_cache_audit,
        "candidate_and_coefficient_compilation_skipped_due_authenticated_base_override": (
            base_override_authentication is not None
        ),
        "base_codebook_artifact_sha256": base_codebook.artifact_sha256,
        "base_codebook_tensor_content": base_tensor_content,
        "base_codebook_source_kind": base_reference["source_kind"],
        "base_codebook_phase_key_sha256": (
            None
            if base_codebook_phase is None
            else base_codebook_phase.phase_key.phase_key_sha256
        ),
        "base_codebook_phase_completion_sha256": (
            None
            if base_codebook_phase is None
            else base_codebook_phase.completion_sha256
        ),
        "base_codebook_phase_cache_hit": (
            None if base_codebook_phase is None else base_codebook_phase.cache_hit
        ),
        "base_codebook_override_authentication": base_override_authentication,
        "base_codebook_reference_sha256": base_reference["reference_sha256"],
        "sparse_semantic_codebook_artifact_sha256": codebook.artifact_sha256,
        "codebook_phase_kind": codebook_reference["source_kind"],
        "codebook_phase_key_sha256": codebook_reference["phase_key_sha256"],
        "codebook_phase_completion_sha256": codebook_reference[
            "phase_completion_sha256"
        ],
        "codebook_phase_cache_hit": codebook_reference["phase_cache_hit"],
        "codebook_reference_sha256": codebook_reference["reference_sha256"],
        "norm_reclosure": {
            "enabled": norm_reclosure_enabled,
            "request_receipt": reclosure_request_receipt,
            "phase_key_sha256": (
                None
                if norm_reclosure_phase is None
                else norm_reclosure_phase.phase_key.phase_key_sha256
            ),
            "phase_completion_sha256": (
                None
                if norm_reclosure_phase is None
                else norm_reclosure_phase.completion_sha256
            ),
            "phase_cache_hit": (
                None if norm_reclosure_phase is None else norm_reclosure_phase.cache_hit
            ),
            "base_codebook_artifact_sha256": base_codebook.artifact_sha256,
            "result_codebook_artifact_sha256": codebook.artifact_sha256,
            "base_tensor_content_sha256": base_tensor_content["content_sha256"],
            "norm_reclosure_sha256": selected_codebook_receipt.get(
                "norm_reclosure_sha256"
            ),
            "core_receipt": reclosure_receipt_summary,
            "row_reclosure_diagnostics_path": (
                None
                if reclosure_row_diagnostics_path is None
                else str(reclosure_row_diagnostics_path)
            ),
            "row_reclosure_diagnostics_sha256": (
                reclosure_row_diagnostics_sha if norm_reclosure_enabled else None
            ),
            "row_reclosure_diagnostics_count": (
                len(reclosure_row_diagnostics) if norm_reclosure_enabled else 0
            ),
            "continuous_coefficient_count_before": int(
                base_codebook.semantic_code_coefficients.numel()
            ),
            "continuous_coefficient_count_after": int(
                codebook.semantic_code_coefficients.numel()
            ),
            "continuous_coefficient_count_unchanged": (
                int(base_codebook.semantic_code_coefficients.numel())
                == int(codebook.semantic_code_coefficients.numel())
            ),
            "effective_child_parameter_count_added": 0,
        },
        "chunk_cache_audit": chunk_cache_audit,
        "active_k": _positive_int(active_k, "active_k"),
        "maximum_candidates": _positive_int(maximum_candidates, "maximum_candidates"),
        "candidate_ridge": _finite_nonnegative(candidate_ridge, "candidate_ridge"),
        "refit_ridge": _finite_nonnegative(refit_ridge, "refit_ridge"),
        "semantic_channel": "C-sparse-continuous-code-for-embeddings-and-logits",
        "emission_channel": "S-exact-raw-token-span-for-text-only",
        "emission_spans_used_as_semantic_coefficients": False,
        "routing_contract_sha256": routing_contract_sha,
        "routing_receipt": routing_receipt,
        "effective_parameter_accounting": accounting,
        "numeric_execution": {
            "dtype_contract": numeric_dtype_contract,
            "runtime": runtime_dtype_receipt,
            "runtime_relocation": runtime_relocation_receipt,
            "candidate_selection": selection_receipt,
            "coefficient_solve": solve_receipt,
            "norm_reclosure": reclosure_placement_receipt,
            "norm_reclosure_request_receipt": reclosure_request_receipt,
            "semantic_coefficient_publication_dtype": "float32",
            "silent_dtype_coercion": False,
        },
        "cached_hidden_diagnostic": (
            None
            if diagnostic_phase is None
            else {
                "phase_key_sha256": diagnostic_phase.phase_key.phase_key_sha256,
                "completion_sha256": diagnostic_phase.completion_sha256,
                "cache_hit": diagnostic_phase.cache_hit,
                "diagnostic_sha256": diagnostic_phase.document.get("diagnostic_sha256"),
                "cached_hidden_row_count": diagnostic_phase.document.get(
                    "cached_hidden_row_count"
                ),
                "cached_hidden_top1_match_count": diagnostic_phase.document.get(
                    "cached_hidden_top1_match_count"
                ),
                "cached_hidden_top5_match_count": diagnostic_phase.document.get(
                    "cached_hidden_top5_match_count"
                ),
                "cached_hidden_top10_match_count": diagnostic_phase.document.get(
                    "cached_hidden_top10_match_count"
                ),
                "cached_hidden_mean_reciprocal_rank": diagnostic_phase.document.get(
                    "cached_hidden_mean_reciprocal_rank"
                ),
                "mean_parent_winner_margin_over_best_other_child_score": (
                    diagnostic_phase.document.get(
                        "mean_parent_winner_margin_over_best_other_child_score"
                    )
                ),
                "positive_parent_winner_margin_count": (
                    diagnostic_phase.document.get("positive_parent_winner_margin_count")
                ),
                "zero_parent_winner_margin_count": diagnostic_phase.document.get(
                    "zero_parent_winner_margin_count"
                ),
                "negative_parent_winner_margin_count": (
                    diagnostic_phase.document.get("negative_parent_winner_margin_count")
                ),
                "top1_match_given_parent_multi_code_count": (
                    diagnostic_phase.document.get(
                        "top1_match_given_parent_multi_code_count"
                    )
                ),
                "parent_winner_by_predicted_winner_kind_confusion": (
                    diagnostic_phase.document.get(
                        "parent_winner_by_predicted_winner_kind_confusion"
                    )
                ),
                "row_rank_diagnostics_sha256": diagnostic_phase.document.get(
                    "row_rank_diagnostics_sha256"
                ),
                "reclosure_row_provenance_sha256": diagnostic_phase.document.get(
                    "reclosure_row_provenance_sha256"
                ),
                "reclosure_transition_aggregates": diagnostic_phase.document.get(
                    "reclosure_transition_aggregates"
                ),
                "metrics_are_advisory_not_an_intelligence_gate": True,
            }
        ),
        "prompt_count": len(prompt_phases),
        "prompt_set_sha256": prompt_set_sha256,
        "prompt_phase_cache_hits": prompt_hits,
        "prompt_phase_computations": len(prompt_phases) - prompt_hits,
        "prompt_phase_keys": [
            {
                "id": phase.prompt_id,
                "phase_key_sha256": phase.phase_key.phase_key_sha256,
                "completion_sha256": phase.completion_sha256,
                "runtime_sha256": phase.runtime_receipt["runtime_sha256"],
            }
            for phase in prompt_phases
        ],
        "whole_response_dossier": str(dossier_path),
        "whole_response_dossier_sha256": dossier["dossier_sha256"],
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "numeric_or_binary_behavior_gate_used": False,
        "exact_token_match_gate_used": False,
        "automatic_acceptance_or_rejection_performed": False,
        "artifact_operation_audit": artifact_audit,
        "operation_audit": {
            "parent_model_objects_constructed": 0,
            "parent_model_load_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
            "network_access": False,
        },
    }
    result = {**body, "result_sha256": canonical_sha256(body)}
    _atomic_write_json(run_root / "run-receipt.json", result)
    _atomic_write_json(
        run_root / "preflight-and-phase-summary.json",
        {
            "schema": "neural-forge-lfm2-sparse-semantic-preflight-summary-v1",
            "status": "complete",
            "run_launch_receipt_sha256": launch_receipt["receipt_sha256"],
            "checkpoint_authentication_receipt_sha256": (
                checkpoint_authentication["receipt_sha256"]
            ),
            "numeric_dtype_contract": numeric_dtype_contract,
            "child_runtime_relocation_receipt": runtime_relocation_receipt,
            "child_remained_on_cpu_through_analytic_compilation": True,
            "runtime_dtype_receipt": runtime_dtype_receipt,
            "selection_placement_receipt": selection_receipt,
            "solve_placement_receipt": solve_receipt,
            "support_selection": support_mode,
            "omp_orthogonality_tolerance": resolved_omp_tolerance,
            "omp_numeric_backend_receipt_sha256": omp_numeric_backend_receipt_sha,
            "candidate_support_math_abi_sha256": support_math_abi_sha,
            "candidate_support_checkpoint_compatibility_sha256": (
                support_checkpoint_compatibility_sha
            ),
            "coefficient_solve_math_abi_sha256": coefficient_math_abi_sha,
            "norm_reclosure_math_abi_sha256": reclosure_math_abi_sha,
            "phase_math_abi_receipts": math_abi_receipts,
            "sparse_core_source_sha256_observational_only": sparse_core_source_sha,
            "candidate_support_cache_audit": candidate_support_cache_audit,
            "norm_reclosure_request_receipt": reclosure_request_receipt,
            "norm_reclosure_placement_receipt": reclosure_placement_receipt,
            "base_codebook_override_authentication_receipt_sha256": (
                None
                if base_override_authentication is None
                else base_override_authentication["receipt_sha256"]
            ),
            "base_codebook_reference_sha256": base_reference["reference_sha256"],
            "norm_reclosure_phase_key_sha256": (
                None
                if norm_reclosure_phase is None
                else norm_reclosure_phase.phase_key.phase_key_sha256
            ),
            "norm_reclosure_phase_completion_sha256": (
                None
                if norm_reclosure_phase is None
                else norm_reclosure_phase.completion_sha256
            ),
            "norm_reclosure_phase_cache_hit": (
                None if norm_reclosure_phase is None else norm_reclosure_phase.cache_hit
            ),
            "source_inspection_sha256": source_inspection["inspection_sha256"],
            "result_sha256": result["result_sha256"],
        },
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--phase-cache-directory", required=True, type=Path)
    parser.add_argument("--checkpoint-directory", required=True, type=Path)
    parser.add_argument(
        "--base-codebook-directory",
        type=Path,
        help=(
            "Authenticated unreclosed codebook to reuse without repeating candidate "
            "discovery or coefficient solving."
        ),
    )
    parser.add_argument(
        "--expected-base-codebook-artifact-sha256",
        help="Required artifact seal when --base-codebook-directory is supplied.",
    )
    parser.add_argument("--parent-model-directory", required=True, type=Path)
    parser.add_argument("--donor-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--target-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument("--complete-causal-cache-directory", type=Path)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--expected-checkpoint-artifact-inventory-sha256")
    parser.add_argument("--expected-child-state-sha256")
    parser.add_argument("--expected-parent-identity-sha256", required=True)
    parser.add_argument("--expected-parent-unique-parameters", required=True, type=int)
    parser.add_argument("--expected-child-unique-parameters", required=True, type=int)
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--parent-envelope", required=True, type=Path)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-parent-envelope-file-sha256", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--candidate-ridge", type=float, default=1e-5)
    parser.add_argument("--refit-ridge", type=float, default=1e-5)
    parser.add_argument(
        "--support-selection",
        choices=_SUPPORT_SELECTION_CHOICES,
        default=_TOP_COSINE_SUPPORT_SELECTION,
        help=(
            "Candidate support algorithm. residual-omp searches the full target "
            "dictionary against each updated residual and checkpoints every prefix."
        ),
    )
    parser.add_argument(
        "--omp-orthogonality-tolerance",
        type=float,
        help=(
            "Positive double-MGS rank tolerance for residual-omp; defaults to 64 "
            "times epsilon of the resolved selection dtype."
        ),
    )
    parser.add_argument("--neighbour-pool-size", type=int, default=32)
    parser.add_argument("--row-chunk-size", type=int, default=256)
    parser.add_argument("--maximum-hidden-rows-per-score-chunk", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--checkpoint-provenance-device", default="cpu")
    parser.add_argument("--checkpoint-provenance-dtype", default="bfloat16")
    parser.add_argument("--selection-device")
    parser.add_argument("--selection-dtype", default="float32")
    parser.add_argument(
        "--solve-device",
        default="cpu",
        help="Coefficient solve device; defaults to CPU for stable FP64 solves.",
    )
    parser.add_argument("--solve-dtype", default="float64")
    parser.add_argument(
        "--norm-reclosure-gamma",
        type=float,
        help="Enable coefficient-only norm reclosure with this interpolation.",
    )
    parser.add_argument(
        "--norm-reclosure-max-scale",
        type=float,
        help="Required explicit scale ceiling when norm reclosure is enabled.",
    )
    parser.add_argument(
        "--norm-reclosure-minimum-cosine-to-amplify",
        type=float,
        help="Optionally leave rows below this source cosine unamplified.",
    )
    parser.add_argument("--norm-reclosure-device")
    parser.add_argument("--norm-reclosure-dtype")
    parser.add_argument("--maximum-span-rows-per-chunk", type=int, default=1024)
    parser.add_argument("--maximum-prompt-tokens", type=int, default=1024)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument("--maximum-embedding-rows-per-chunk", type=int, default=1024)
    parser.add_argument("--maximum-donor-rows-per-score-chunk", type=int, default=4096)
    parser.add_argument("--expected-prior-direct-winners", type=int, default=293)
    parser.add_argument("--expected-prior-multi-code-winners", type=int, default=72)
    parser.add_argument("--console-progress-every", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _positive_int(args.k, "k")
        if args.neighbour_pool_size < args.k:
            raise Lfm2SparseSemanticRunnerError(
                "neighbour_pool_size must be at least k."
            )
        if (
            args.support_selection == _RESIDUAL_OMP_SUPPORT_SELECTION
            and args.neighbour_pool_size != args.k
        ):
            raise Lfm2SparseSemanticRunnerError(
                "residual-omp requires neighbour_pool_size == k."
            )
        result = run(
            run_directory=args.run_directory,
            phase_cache_directory=args.phase_cache_directory,
            checkpoint_directory=args.checkpoint_directory,
            base_codebook_directory=args.base_codebook_directory,
            expected_base_codebook_artifact_sha256=(
                args.expected_base_codebook_artifact_sha256
            ),
            parent_model_directory=args.parent_model_directory,
            donor_tokenizer_directory=args.donor_tokenizer_directory,
            target_tokenizer_directory=args.target_tokenizer_directory,
            bridge_directory=args.bridge_directory,
            expected_bridge_artifact_sha256=(args.expected_bridge_artifact_sha256),
            expected_checkpoint_artifact_inventory_sha256=(
                args.expected_checkpoint_artifact_inventory_sha256
            ),
            expected_child_state_sha256=args.expected_child_state_sha256,
            expected_child_unique_parameters=args.expected_child_unique_parameters,
            expected_parent_identity_sha256=args.expected_parent_identity_sha256,
            expected_parent_unique_parameters=args.expected_parent_unique_parameters,
            prompts_json=args.prompts_json,
            expected_prompts_file_sha256=args.expected_prompts_file_sha256,
            parent_envelope=args.parent_envelope,
            expected_parent_envelope_sha256=(args.expected_parent_envelope_sha256),
            expected_parent_envelope_file_sha256=(
                args.expected_parent_envelope_file_sha256
            ),
            complete_causal_cache_directory=(args.complete_causal_cache_directory),
            device=args.device,
            dtype=args.dtype,
            checkpoint_provenance_device=args.checkpoint_provenance_device,
            checkpoint_provenance_dtype=args.checkpoint_provenance_dtype,
            selection_device=args.selection_device,
            selection_dtype=args.selection_dtype,
            solve_device=args.solve_device,
            solve_dtype=args.solve_dtype,
            norm_reclosure_gamma=args.norm_reclosure_gamma,
            norm_reclosure_max_scale=args.norm_reclosure_max_scale,
            norm_reclosure_minimum_cosine_to_amplify=(
                args.norm_reclosure_minimum_cosine_to_amplify
            ),
            norm_reclosure_device=args.norm_reclosure_device,
            norm_reclosure_dtype=args.norm_reclosure_dtype,
            maximum_candidates=args.neighbour_pool_size,
            active_k=args.k,
            candidate_ridge=args.candidate_ridge,
            refit_ridge=args.refit_ridge,
            donor_rows_per_chunk=args.row_chunk_size,
            maximum_span_rows_per_chunk=args.maximum_span_rows_per_chunk,
            maximum_prompt_tokens=args.maximum_prompt_tokens,
            maximum_new_tokens=args.maximum_new_tokens,
            maximum_embedding_rows_per_chunk=(args.maximum_embedding_rows_per_chunk),
            maximum_donor_rows_per_score_chunk=(
                args.maximum_donor_rows_per_score_chunk
            ),
            maximum_hidden_rows_per_score_chunk=(
                args.maximum_hidden_rows_per_score_chunk
            ),
            expected_prior_direct_winners=args.expected_prior_direct_winners,
            expected_prior_multi_code_winners=(args.expected_prior_multi_code_winners),
            console_progress_every=args.console_progress_every,
            support_selection=args.support_selection,
            omp_orthogonality_tolerance=args.omp_orthogonality_tolerance,
        )
    except Exception as exc:
        run_root = args.run_directory.expanduser().absolute()
        if run_root.exists() and run_root.is_dir() and not run_root.is_symlink():
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
                "support_selection": result["support_selection"],
                "omp_orthogonality_tolerance": result["omp_orthogonality_tolerance"],
                "result_sha256": result["result_sha256"],
                "codebook_artifact_sha256": result[
                    "sparse_semantic_codebook_artifact_sha256"
                ],
                "effective_parameter_compaction_ratio": result[
                    "effective_parameter_accounting"
                ]["effective_parameter_compaction_ratio"],
                "whole_response_dossier": result["whole_response_dossier"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
