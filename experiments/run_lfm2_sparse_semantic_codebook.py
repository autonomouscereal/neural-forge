"""Compile and review the LFM2 dual-channel sparse semantic token bridge.

The emission channel ``S`` is the already-authenticated exact raw-token span
bridge.  The semantic channel ``C`` is a sparse, continuous code over the
existing target table and is used for both donor embeddings and donor logits.
The parent is never instantiated or executed: this runner reads only the tied
embedding rows from local safetensors and consumes an optional sealed trace
cache for an advisory hidden-state diagnostic.

Expensive work is split into content-addressed phases.  Source-table and
neighbour-candidate phases deliberately exclude ``k`` and ridge so later
coefficient sweeps reuse them.  Coefficient phases include both values.  Each
whole-response prompt remains an independent resumable phase, and manual
whole-response review is authoritative.
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
HIDDEN_DIAGNOSTIC_SCHEMA = "neural-forge-lfm2-sparse-semantic-hidden-top1-diagnostic-v1"
PARENT_TABLE_PHASE = "lfm2-sparse-semantic-parent-table"
NEIGHBOUR_PHASE = "lfm2-sparse-semantic-neighbour-chunk"
COEFFICIENT_PHASE = "lfm2-sparse-semantic-coefficient-chunk"
DIAGNOSTIC_PHASE = "lfm2-sparse-semantic-hidden-diagnostic"
CANDIDATE_INVENTORY_PHASE = "lfm2-sparse-semantic-candidate-inventory"
CODEBOOK_PHASE = "lfm2-sparse-semantic-codebook"
PARENT_TIED_TENSOR_KEY = "model.embed_tokens.weight"
_SHA256_ZERO = "0" * 64


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
) -> CachedArtifactPhase:
    """Cache one core artifact directory while reloading every publication."""

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
        return CachedArtifactPhase(
            artifact=artifact,
            artifact_sha256=artifact_sha,
            phase_key=phase_key,
            completion_sha256=hit.completion_sha256,
            cache_hit=True,
        )
    with cache.begin(phase_key, events=events) as transaction:

        def progress(record: Mapping[str, Any]) -> None:
            transaction.progress({"compiler_progress": _plain_json(record)})

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
) -> CachedJsonPhase:
    hit = cache.lookup(phase_key, events=events)
    if hit is not None:
        return CachedJsonPhase(
            document=_load_json_phase_entry(hit),
            phase_key=phase_key,
            completion_sha256=hit.completion_sha256,
            cache_hit=True,
        )
    with cache.begin(phase_key, events=events) as transaction:

        def progress(record: Mapping[str, Any]) -> None:
            transaction.progress({"compiler_progress": _plain_json(record)})

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
) -> CachedTensorPhase:
    """Atomically load or publish one tensor phase with per-chunk progress."""

    hit = cache.lookup(phase_key, events=events)
    if hit is not None:
        tensors, receipt = _load_tensor_phase_entry(hit)
        return CachedTensorPhase(
            tensors=tensors,
            receipt=receipt,
            phase_key=phase_key,
            completion_sha256=hit.completion_sha256,
            cache_hit=True,
        )
    with cache.begin(phase_key, events=events) as transaction:

        def progress(record: Mapping[str, Any]) -> None:
            transaction.progress({"compiler_progress": _plain_json(record)})

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
            "schema": str(value.get("schema")),
            "core_request_sha256": value["request_sha256"],
            "active_k_or_ridge_in_identity": False,
            "reusable_across_k_and_ridge": True,
        },
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
    ) -> None:
        self.cache = cache
        self.events_directory = events_directory
        self.candidate_checkpoint_type = candidate_checkpoint_type
        self.coefficient_checkpoint_type = coefficient_checkpoint_type
        self._candidate_pending: dict[str, tuple[FoldPhaseKey, FoldPhaseEventLog]] = {}
        self._coefficient_pending: dict[
            str, tuple[FoldPhaseKey, FoldPhaseEventLog]
        ] = {}
        self.audit = {
            "candidate_chunk_cache_hits": 0,
            "candidate_chunk_cache_misses": 0,
            "candidate_chunk_publications": 0,
            "coefficient_chunk_cache_hits": 0,
            "coefficient_chunk_cache_misses": 0,
            "coefficient_chunk_publications": 0,
        }

    def _events(
        self,
        *,
        prefix: str,
        request: Mapping[str, Any],
        phase_key: FoldPhaseKey,
    ) -> FoldPhaseEventLog:
        start = _nonnegative_int(request.get("donor_row_start"), "donor_row_start")
        stop = _positive_int(request.get("donor_row_stop"), "donor_row_stop")
        return _event_log(
            self.events_directory,
            label=f"{prefix}-{start:06d}-{stop:06d}",
            phase_key=phase_key,
        )

    def load_candidate(self, request: Mapping[str, Any]) -> Any | None:
        value = _verified_core_request(request, "candidate chunk")
        key = _candidate_request_phase_key(value)
        events = self._events(prefix="candidate", request=value, phase_key=key)
        hit = self.cache.lookup(key, events=events)
        if hit is None:
            self.audit["candidate_chunk_cache_misses"] += 1
            self._candidate_pending[value["request_sha256"]] = (key, events)
            return None
        tensors, receipt = _load_tensor_phase_entry(hit)
        metadata = receipt.get("metadata", {})
        checkpoint_receipt = metadata.get("checkpoint_receipt")
        if not isinstance(checkpoint_receipt, Mapping):
            raise Lfm2SparseSemanticRunnerError(
                "Cached candidate chunk has no core checkpoint receipt."
            )
        self.audit["candidate_chunk_cache_hits"] += 1
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
        )
        self.audit["candidate_chunk_publications"] += 1

    def load_coefficient(self, request: Mapping[str, Any]) -> Any | None:
        value = _verified_core_request(request, "coefficient chunk")
        key = _coefficient_request_phase_key(value)
        events = self._events(prefix="coefficient", request=value, phase_key=key)
        hit = self.cache.lookup(key, events=events)
        if hit is None:
            self.audit["coefficient_chunk_cache_misses"] += 1
            self._coefficient_pending[value["request_sha256"]] = (key, events)
            return None
        tensors, receipt = _load_tensor_phase_entry(hit)
        metadata = receipt.get("metadata", {})
        checkpoint_receipt = metadata.get("checkpoint_receipt")
        if not isinstance(checkpoint_receipt, Mapping):
            raise Lfm2SparseSemanticRunnerError(
                "Cached coefficient chunk has no core checkpoint receipt."
            )
        self.audit["coefficient_chunk_cache_hits"] += 1
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
        )
        self.audit["coefficient_chunk_publications"] += 1

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


def build_hidden_diagnostic_phase_key(
    *,
    trace_cache_key: str,
    trace_manifest_sha256: str,
    routing_contract_sha256: str,
    target_table_sha256: str,
    diagnostic_implementation_sha256: str,
    maximum_hidden_rows_per_score_chunk: int,
) -> FoldPhaseKey:
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
        },
        abi={
            "schema": HIDDEN_DIAGNOSTIC_SCHEMA,
            "selected_cache_tensor_suffixes": [
                ".output.lm_head_input",
                ".logits",
                ".full_prefix_logits",
            ],
            "advisory_not_gate": True,
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

    total = 0
    matches = 0
    direct_total = 0
    direct_matches = 0
    multi_total = 0
    multi_matches = 0
    selected_tensor_bytes = 0
    batch_reports: list[dict[str, Any]] = []
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
                            equal = predicted.eq(reference_winners)
                            for row, winner in enumerate(reference_winners.tolist()):
                                is_direct = int(winner) in direct_donor_ids
                                matched = bool(equal[row].item())
                                total += 1
                                matches += int(matched)
                                batch_total += 1
                                batch_matches += int(matched)
                                if is_direct:
                                    direct_total += 1
                                    direct_matches += int(matched)
                                else:
                                    multi_total += 1
                                    multi_matches += int(matched)
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
                        }
                    )
                finally:
                    trace_session.release_selected_tensors(payload)
        trace_session.verify()
        session_receipt = trace_session.as_dict()
    if total <= 0 or direct_total + multi_total != total:
        raise Lfm2SparseSemanticRunnerError(
            "Sealed cache exposed no aligned hidden diagnostic rows."
        )
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
        "parent_top1_direct_anchor_count": direct_total,
        "top1_match_given_parent_direct_anchor_count": direct_matches,
        "top1_match_given_parent_direct_anchor_rate": (
            None if direct_total == 0 else direct_matches / direct_total
        ),
        "parent_top1_multi_code_count": multi_total,
        "top1_match_given_parent_multi_code_count": multi_matches,
        "top1_match_given_parent_multi_code_rate": (
            None if multi_total == 0 else multi_matches / multi_total
        ),
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
) -> dict[str, Any]:
    """Execute or resume the A1 sparse semantic bridge review end to end."""

    from neural_forge.hf_lfm2_sparse_semantic_codebook import (
        Lfm2SparseSemanticCandidateChunkCheckpoint,
        Lfm2SparseSemanticCoefficientChunkCheckpoint,
        compile_lfm2_sparse_semantic_codebook,
        discover_lfm2_sparse_semantic_candidates,
        load_lfm2_sparse_semantic_candidate_inventory,
        load_lfm2_sparse_semantic_codebook,
        save_lfm2_sparse_semantic_candidate_inventory,
        save_lfm2_sparse_semantic_codebook,
    )

    run_started = time.monotonic()
    run_root = _output_directory(run_directory, "run directory")
    cache_root = _output_directory(phase_cache_directory, "phase cache directory")
    events_root = _output_directory(run_root / "events", "events directory")
    checkpoint = _local_directory(checkpoint_directory, "child checkpoint")
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
    solve_device_value = device if solve_device is None else solve_device
    coefficient_device, coefficient_dtype, solve_receipt = resolve_compile_placement(
        role="coefficient-solve",
        requested_device=solve_device_value,
        requested_dtype=solve_dtype,
    )

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
    _atomic_write_json(
        run_root / "checkpoint-portable-load-receipt.json",
        portable_load_receipt,
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
    console = ConsoleProgressReporter(
        semantic_step_every=_positive_int(
            console_progress_every, "console progress interval"
        )
    )
    span_contract = lfm2_semantic_step_span_adapter_cache_contract(bridge)
    span_key = span_phase_key_from_contract(span_contract)
    span_started = time.monotonic()
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
    )
    donor_table = parent_table_phase.tensors["donor_table"]
    parent_table_sha = _tensor_sha256(donor_table)
    target_table_cpu = live_target_table.detach().to(device="cpu").contiguous().clone()
    target_table_sha = _tensor_sha256(target_table_cpu)
    target_to_donor_sha = _tensor_sha256(target_to_donor)

    import neural_forge.hf_lfm2_sparse_semantic_codebook as sparse_core
    import neural_forge.hf_lfm2_semantic_step_model_runtime as model_runtime_core
    import neural_forge.hf_lfm2_semantic_step_token_fold as token_fold_core

    core_source_sha = _file_sha256(Path(sparse_core.__file__).resolve(strict=True))
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
    chunk_adapter = CoreChunkPhaseAdapter(
        cache=cache,
        events_directory=events_root,
        candidate_checkpoint_type=Lfm2SparseSemanticCandidateChunkCheckpoint,
        coefficient_checkpoint_type=Lfm2SparseSemanticCoefficientChunkCheckpoint,
    )
    candidate_key = build_candidate_inventory_phase_key(
        parent_table_sha256=parent_table_sha,
        target_table_sha256=target_table_sha,
        target_to_donor_sha256=target_to_donor_sha,
        maximum_candidates=maximum_candidates,
        donor_rows_per_chunk=donor_rows_per_chunk,
        selection_placement_receipt_sha256=selection_receipt["receipt_sha256"],
        algorithm_sha256=core_source_sha,
    )
    candidate_started = time.monotonic()
    candidate_phase = cache_or_compute_artifact_phase(
        cache=cache,
        events=_event_log(
            events_root, label="candidate-inventory", phase_key=candidate_key
        ),
        phase_key=candidate_key,
        compute_call=lambda progress: discover_lfm2_sparse_semantic_candidates(
            donor_table,
            target_table_cpu,
            target_to_donor,
            maximum_candidates=_positive_int(maximum_candidates, "maximum_candidates"),
            donor_rows_per_chunk=_positive_int(
                donor_rows_per_chunk, "donor_rows_per_chunk"
            ),
            selection_device=select_device,
            selection_dtype=select_dtype,
            progress_callback=lambda record: progress(
                {
                    "core_progress": _plain_json(record),
                    "telemetry": runtime_telemetry(
                        started_at=candidate_started,
                        device=str(select_device),
                        requested_dtype=selection_dtype,
                        resolved_dtype=str(select_dtype).removeprefix("torch."),
                    ),
                }
            ),
            candidate_chunk_loader=chunk_adapter.load_candidate,
            candidate_chunk_publisher=chunk_adapter.publish_candidate,
        ),
        save_call=lambda artifact, destination: (
            save_lfm2_sparse_semantic_candidate_inventory(artifact, destination)
        ),
        load_call=lambda destination: load_lfm2_sparse_semantic_candidate_inventory(
            destination
        ),
        artifact_sha256_call=lambda artifact: artifact.artifact_sha256,
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
        algorithm_sha256=core_source_sha,
    )
    coefficient_started = time.monotonic()
    codebook_phase = cache_or_compute_artifact_phase(
        cache=cache,
        events=_event_log(events_root, label="codebook", phase_key=codebook_key),
        phase_key=codebook_key,
        compute_call=lambda progress: compile_lfm2_sparse_semantic_codebook(
            donor_table,
            target_table_cpu,
            target_to_donor,
            candidate_inventory=candidate_phase.artifact,
            maximum_candidates=_positive_int(maximum_candidates, "maximum_candidates"),
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
                        resolved_dtype=str(coefficient_dtype).removeprefix("torch."),
                    ),
                }
            ),
            coefficient_chunk_loader=chunk_adapter.load_coefficient,
            coefficient_chunk_publisher=chunk_adapter.publish_coefficient,
        ),
        save_call=lambda artifact, destination: save_lfm2_sparse_semantic_codebook(
            artifact, destination
        ),
        load_call=lambda destination: load_lfm2_sparse_semantic_codebook(destination),
        artifact_sha256_call=lambda artifact: artifact.artifact_sha256,
    )
    chunk_adapter.assert_drained()
    codebook = codebook_phase.artifact
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
                maximum_hidden_rows_per_score_chunk=(
                    maximum_hidden_rows_per_score_chunk
                ),
                expected_prior_direct_winners=expected_prior_direct_winners,
                expected_prior_multi_code_winners=(expected_prior_multi_code_winners),
                progress_callback=progress,
            ),
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
        codebook_artifact_sha256=codebook_phase.artifact_sha256,
        hidden_diagnostic=(
            None if diagnostic_phase is None else diagnostic_phase.document
        ),
        maximum_new_tokens=maximum_new_tokens,
    )
    dossier_path = run_root / "whole-response-dossier.json"
    _atomic_write_json(dossier_path, dossier)

    codebook_reference_body = {
        "schema": "neural-forge-lfm2-sparse-semantic-codebook-reference-v1",
        "artifact_sha256": codebook_phase.artifact_sha256,
        "phase_key_sha256": codebook_phase.phase_key.phase_key_sha256,
        "phase_completion_sha256": codebook_phase.completion_sha256,
        "phase_cache_artifact_directory": str(
            cache.entry_path(codebook_phase.phase_key) / "payload" / "artifact"
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
            "-dual-channel-sparse-semantic-manual-review"
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
        "runtime_initial_child_state_sha256": initial_child_state_sha,
        "runtime_final_child_state_sha256": final_child_state_sha,
        "runtime_child_state_unchanged": True,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "span_evidence_sha256": span_phase.span_evidence["evidence_sha256"],
        "span_phase_key_sha256": span_phase.phase_key.phase_key_sha256,
        "span_phase_completion_sha256": span_phase.completion_sha256,
        "span_phase_cache_hit": span_phase.cache_hit,
        "candidate_inventory_artifact_sha256": candidate_phase.artifact_sha256,
        "candidate_inventory_content_sha256": (candidate_phase.artifact.content_sha256),
        "candidate_inventory_phase_key_sha256": (
            candidate_phase.phase_key.phase_key_sha256
        ),
        "candidate_inventory_completion_sha256": (candidate_phase.completion_sha256),
        "candidate_inventory_cache_hit": candidate_phase.cache_hit,
        "sparse_semantic_codebook_artifact_sha256": (codebook_phase.artifact_sha256),
        "codebook_phase_key_sha256": codebook_phase.phase_key.phase_key_sha256,
        "codebook_phase_completion_sha256": codebook_phase.completion_sha256,
        "codebook_phase_cache_hit": codebook_phase.cache_hit,
        "chunk_cache_audit": dict(chunk_adapter.audit),
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
            "runtime": runtime_dtype_receipt,
            "candidate_selection": selection_receipt,
            "coefficient_solve": solve_receipt,
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
                "top1_match_given_parent_multi_code_count": (
                    diagnostic_phase.document.get(
                        "top1_match_given_parent_multi_code_count"
                    )
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
            "runtime_dtype_receipt": runtime_dtype_receipt,
            "selection_placement_receipt": selection_receipt,
            "solve_placement_receipt": solve_receipt,
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
        help="Coefficient solve device; defaults to the runtime device.",
    )
    parser.add_argument("--solve-dtype", default="auto")
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
        result = run(
            run_directory=args.run_directory,
            phase_cache_directory=args.phase_cache_directory,
            checkpoint_directory=args.checkpoint_directory,
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
