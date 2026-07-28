#!/usr/bin/env python3
"""Publish the authenticated layer-0 partial ShortConv folding frontier.

This producer deliberately stops at folded tensors.  It loads the direct-local
parent only to authenticate and read its parameters, opens the existing trace
cache once with one selective authenticated session, and compiles four
parent-shell candidates without executing a model:

* an output-action projector up to rank 128, capped by authenticated FIT rank,
  absorbed into the parent output writer;
* an intact-channel, 256-channel Q-DEIM ShortConv;
* that Q-DEIM fold with a FIT-only cached local-action writer; and
* the local writer factored through the same observed output-action subspace.

FIT batches 0--7 are the only construction evidence.  Batches 8--10 are
held-out diagnostics and cannot change a tensor.  Batches 11--15 are never
selected from the cache.  There is no parent forward, generation, trace
recapture, optimizer, backpropagation, pruning, quantization, sparsification,
Hub loading, or network access.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from experiments import run_lfm2_progressive_component_ablation as b038  # noqa: E402
from experiments.run_lfm2_dynamic_catalog import (  # noqa: E402
    fingerprint_local_tokenizer,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_model,
)
from neural_forge.hf_lfm2_output_action_subspace import (  # noqa: E402
    Lfm2OutputActionRankDeficitError,
    compile_lfm2_output_action_subspace,
    seal_lfm2_output_action_evidence,
    seal_lfm2_output_action_rows,
)
from neural_forge.hf_lfm2_output_action_writer_fold import (  # noqa: E402
    compile_lfm2_output_action_writer_fold,
)
from neural_forge.hf_lfm2_parent_shell_writer_catchup import (  # noqa: E402
    compile_lfm2_parent_shell_short_conv_local_writer,
    lfm2_parent_shell_local_writer_evidence_sha256,
    lfm2_parent_shell_writer_state_sha256,
)
from neural_forge.hf_lfm2_partial_short_conv_evidence import (  # noqa: E402
    Lfm2PartialShortConvCachedPhase,
    compile_lfm2_partial_short_conv_parent_shell_evidence,
    lfm2_partial_short_conv_cached_phase_evidence_sha256,
    lfm2_partial_short_conv_qdeim_candidate_sha256,
)
from neural_forge.hf_lfm2_progressive_component_ablation import (  # noqa: E402
    _unique_parameter_elements,
)
from neural_forge.hf_lfm2_short_conv_channel_qdeim import (  # noqa: E402
    authenticate_lfm2_short_conv_channel_qdeim,
    compile_lfm2_short_conv_channel_qdeim,
    lfm2_short_conv_channel_qdeim_source_state_sha256,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


PLAN_SCHEMA = "neural-forge-lfm2-partial-short-conv-frontier-plan-v1"
PRODUCER_SCHEMA = "neural-forge-lfm2-partial-short-conv-frontier-producer-v1"
ARCHIVE_SCHEMA = "neural-forge-lfm2-partial-short-conv-frontier-archive-v1"
PRODUCER_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-partial-short-conv-frontier-producer-receipt-v1"
)
PLAN_FILENAME = "partial-short-conv-frontier-plan.json"
ARCHIVE_FILENAME = "partial-short-conv-frontier.safetensors"
PRODUCER_RECEIPT_FILENAME = "partial-short-conv-frontier-producer-receipt.json"

FIT_BATCH_INDICES = tuple(range(0, 8))
HELDOUT_BATCH_INDICES = tuple(range(8, 11))
UNTOUCHED_BATCH_INDICES = tuple(range(11, 16))
EXPECTED_CACHE_BATCH_COUNT = 16
DEFAULT_TARGET_INTERNAL_CHANNELS = 256
DEFAULT_OUTPUT_ACTION_RANK = 128
DEFAULT_RIDGE = 1.0e-6

_PHASE_PATTERN = re.compile(
    r"(?P<phase>prefill|decode\.\d{3})\.layer\.000\.residual\.pre\Z"
)
_PHASE_SUFFIXES = (
    "layer.000.residual.post_operator",
    "layer.000.residual.pre",
    "layer.000.short_conv.b_history",
    "layer.000.short_conv.c_history",
    "layer.000.short_conv.context_from_fir",
    "layer.000.short_conv.o",
    "layer.000.short_conv.pre_bx_state",
    "layer.000.short_conv.x_history",
)
_LOCAL_WRITER_ROLE = "compact-short-conv-local-output"


class Lfm2PartialShortConvFrontierProducerError(RuntimeError):
    """An authenticated input, fold, or publication failed closed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required runtime dependency
        raise RuntimeError("Partial ShortConv production requires PyTorch.") from exc
    return torch


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2PartialShortConvFrontierProducerError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise Lfm2PartialShortConvFrontierProducerError(
            f"Refusing to replace or misplace publication file {path}."
        )
    path.write_text(
        json.dumps(_plain(value), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
        newline="\n",
    )


def _ordinary_directory(value: Path, *, label: str) -> Path:
    path = value.expanduser().absolute()
    if not path.is_dir() or path.is_symlink():
        raise Lfm2PartialShortConvFrontierProducerError(
            f"{label} must be one ordinary local directory: {path}."
        )
    return path


def _new_output(value: Path) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2PartialShortConvFrontierProducerError(
            "Output directory must be local."
        )
    path = value.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Lfm2PartialShortConvFrontierProducerError(
            "Output directory must not already exist."
        )
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise Lfm2PartialShortConvFrontierProducerError(
            "Output parent must be one ordinary directory."
        )
    return path


def _forbid_output_inside_inputs(output: Path, inputs: Sequence[Path]) -> None:
    target = output.resolve(strict=False)
    for source_value in inputs:
        source = source_value.resolve(strict=True)
        if target == source or source in target.parents:
            raise Lfm2PartialShortConvFrontierProducerError(
                "Output must be lexically separate from authenticated inputs."
            )


def _tensor_record(value: Any) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": [int(size) for size in value.shape],
        "sha256": lfm2_authenticated_tensor_sha256(value),
    }


def _phase_sort_key(phase: str) -> tuple[int, int]:
    if phase == "prefill":
        return (0, -1)
    return (1, int(phase.rsplit(".", 1)[1]))


def _ordered_phases(ordered_tensor_names: Sequence[str]) -> tuple[str, ...]:
    phases = {
        match.group("phase")
        for name in ordered_tensor_names
        if (match := _PHASE_PATTERN.fullmatch(name)) is not None
    }
    if not phases or "prefill" not in phases:
        raise Lfm2PartialShortConvFrontierProducerError(
            "A selected cache batch lacks canonical layer-0 prefill evidence."
        )
    ordered = tuple(sorted(phases, key=_phase_sort_key))
    for expected_step, phase in enumerate(ordered[1:]):
        if phase != f"decode.{expected_step:03d}":
            raise Lfm2PartialShortConvFrontierProducerError(
                "Decode evidence phases are not contiguous from decode.000."
            )
    return ordered


def _phase_tensor_names(phases: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted(f"{phase}.{suffix}" for phase in phases for suffix in _PHASE_SUFFIXES)
    )


def _phase_row_ids(
    *,
    batch_index: int,
    phase: str,
    row_count: int,
) -> tuple[str, ...]:
    if row_count <= 0:
        raise Lfm2PartialShortConvFrontierProducerError(
            "Cached phase row count must be positive."
        )
    return tuple(
        f"batch.{batch_index:06d}/{phase}/row.{row:04d}" for row in range(row_count)
    )


def _phase_receipt(
    *,
    cache_key: str,
    session_sha256: str,
    batch_index: int,
    phase: str,
    input_sha256: str,
    record_sha256: str,
    selection_sha256: str,
    row_ids: Sequence[str],
    tensors: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-partial-short-conv-cache-phase-binding-v1",
            "cache_key": cache_key,
            "selective_session_sha256": session_sha256,
            "batch_index": batch_index,
            "phase": phase,
            "input_sha256": input_sha256,
            "record_sha256": record_sha256,
            "selection_sha256": selection_sha256,
            "row_ids_sha256": canonical_sha256(list(row_ids)),
            "tensors": {
                name: _tensor_record(value) for name, value in sorted(tensors.items())
            },
        }
    )


def _trajectory_receipt(
    *,
    cache_key: str,
    session_sha256: str,
    batch_index: int,
    input_sha256: str,
    record_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-partial-short-conv-trajectory-binding-v1",
            "cache_key": cache_key,
            "selective_session_sha256": session_sha256,
            "batch_index": batch_index,
            "input_sha256": input_sha256,
            "record_sha256": record_sha256,
        }
    )


def _selected_cache_phases(
    *,
    cache: Any,
    session: Any,
    batch_indices: Sequence[int],
) -> tuple[Lfm2PartialShortConvCachedPhase, ...]:
    torch = _torch()
    result: list[Lfm2PartialShortConvCachedPhase] = []
    references = session.batch_references
    for batch_index in batch_indices:
        reference = references[batch_index]
        phases = _ordered_phases(reference.ordered_tensor_names)
        selected_names = _phase_tensor_names(phases)
        payload = session.load_selected_tensors(batch_index, selected_names)
        try:
            copied = {
                name: payload.tensors[name].detach().cpu().contiguous().clone()
                for name in selected_names
            }
            row_counts: dict[str, int] = {}
            for phase in phases:
                pre = copied[f"{phase}.layer.000.residual.pre"]
                if pre.ndim != 3 or any(int(size) <= 0 for size in pre.shape):
                    raise Lfm2PartialShortConvFrontierProducerError(
                        f"Batch {batch_index} phase {phase} residual shape is invalid."
                    )
                row_counts[phase] = int(pre.shape[0]) * int(pre.shape[1])
            trajectory_rows = sum(row_counts.values())
            if trajectory_rows <= 0:
                raise Lfm2PartialShortConvFrontierProducerError(
                    "One cached trajectory has no fold rows."
                )
            trajectory_receipt = _trajectory_receipt(
                cache_key=cache.cache_key,
                session_sha256=session.evidence_sha256,
                batch_index=batch_index,
                input_sha256=reference.input_sha256,
                record_sha256=reference.record_sha256,
            )
            for phase in phases:
                prefix = f"{phase}.layer.000."
                pre = copied[prefix + "residual.pre"]
                post = copied[prefix + "residual.post_operator"]
                b_history = copied[prefix + "short_conv.b_history"]
                c_history = copied[prefix + "short_conv.c_history"]
                x_history = copied[prefix + "short_conv.x_history"]
                context = copied[prefix + "short_conv.context_from_fir"]
                parent_short_conv_o = copied[prefix + "short_conv.o"]
                pre_state = copied[prefix + "short_conv.pre_bx_state"]
                rows = row_counts[phase]
                row_ids = _phase_row_ids(
                    batch_index=batch_index,
                    phase=phase,
                    row_count=rows,
                )
                phase_tensors = {
                    name.removeprefix(f"{phase}."): value
                    for name, value in copied.items()
                    if name.startswith(f"{phase}.")
                }
                phase_receipt = _phase_receipt(
                    cache_key=cache.cache_key,
                    session_sha256=session.evidence_sha256,
                    batch_index=batch_index,
                    phase=phase,
                    input_sha256=reference.input_sha256,
                    record_sha256=reference.record_sha256,
                    selection_sha256=payload.selection_sha256,
                    row_ids=row_ids,
                    tensors=phase_tensors,
                )
                weights = torch.full(
                    (int(pre.shape[0]), int(pre.shape[1])),
                    1.0 / float(trajectory_rows),
                    dtype=torch.float64,
                )
                result.append(
                    Lfm2PartialShortConvCachedPhase(
                        case_id=f"batch.{batch_index:06d}/{phase}",
                        phase=phase,
                        parent_pre_operator_residual=pre,
                        parent_post_operator_residual=post,
                        parent_short_conv_o=parent_short_conv_o,
                        parent_b_history=b_history,
                        parent_c_history=c_history,
                        parent_x_history=x_history,
                        parent_context_from_fir=context,
                        parent_pre_bx_state=pre_state,
                        row_sample_weights=weights,
                        row_ids=row_ids,
                        parent_trajectory_receipt_sha256=trajectory_receipt,
                        cached_phase_receipt_sha256=phase_receipt,
                        aligned_row_identity_sha256=canonical_sha256(list(row_ids)),
                    )
                )
        finally:
            session.release_selected_tensors(payload)
    return tuple(result)


def _output_action_rows(
    phases: Sequence[Lfm2PartialShortConvCachedPhase],
) -> tuple[tuple[str, ...], Any, Any]:
    torch = _torch()
    row_ids: list[str] = []
    action_rows: list[Any] = []
    weights: list[Any] = []
    for phase in phases:
        batch_prefix, phase_name = phase.case_id.split("/", 1)
        batch_index = int(batch_prefix.rsplit(".", 1)[1])
        rows = int(phase.parent_pre_operator_residual.shape[0]) * int(
            phase.parent_pre_operator_residual.shape[1]
        )
        expected_row_ids = _phase_row_ids(
            batch_index=batch_index,
            phase=phase_name,
            row_count=rows,
        )
        if tuple(phase.row_ids) != expected_row_ids:
            raise Lfm2PartialShortConvFrontierProducerError(
                "Cached phase explicit row identities changed."
            )
        row_ids.extend(phase.row_ids)
        hidden = int(phase.parent_short_conv_o.shape[-1])
        sampled_tokens = int(phase.parent_pre_operator_residual.shape[1])
        action_rows.append(
            phase.parent_short_conv_o[:, -sampled_tokens:, :]
            .to(torch.float32)
            .reshape(rows, hidden)
            .contiguous()
        )
        weights.append(phase.row_sample_weights.reshape(rows).to(torch.float64))
    if not row_ids:
        raise Lfm2PartialShortConvFrontierProducerError(
            "Output-action evidence cannot be empty."
        )
    return (
        tuple(row_ids),
        torch.cat(action_rows).contiguous(),
        torch.cat(weights).contiguous(),
    )


def _candidate_row(
    *,
    candidate_id: str,
    runtime_kind: str,
    external_hidden_size: int,
    internal_channels: int,
    kernel_size: int,
    source_component_parameter_elements: int,
    state: Mapping[str, Any],
    archive: dict[str, Any],
    output_action_rank: int | None = None,
    lineage: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if not candidate_id or not state:
        raise Lfm2PartialShortConvFrontierProducerError(
            "Candidate identifier and state must be nonempty."
        )
    state_keys: dict[str, str] = {}
    state_sha256: dict[str, str] = {}
    candidate_elements = 0
    for state_name, source in sorted(state.items()):
        tensor = (
            source.detach()
            .to(device="cpu", dtype=_torch().float32)
            .contiguous()
            .clone()
        )
        archive_name = f"candidate.{candidate_id}.{state_name}"
        if archive_name in archive:
            raise Lfm2PartialShortConvFrontierProducerError(
                f"Duplicate archive tensor {archive_name!r}."
            )
        archive[archive_name] = tensor
        state_keys[state_name] = archive_name
        state_sha256[state_name] = lfm2_authenticated_tensor_sha256(tensor)
        candidate_elements += int(tensor.numel())
    body: dict[str, Any] = {
        "candidate_id": candidate_id,
        "runtime_kind": runtime_kind,
        "layer_index": 0,
        "module_path": "model.layers.0.conv",
        "external_hidden_size": external_hidden_size,
        "internal_channels": internal_channels,
        "kernel_size": kernel_size,
        "source_component_parameter_elements": source_component_parameter_elements,
        "candidate_parameter_elements": candidate_elements,
        "component_compression_ratio": (
            source_component_parameter_elements / candidate_elements
        ),
        "state_tensor_keys": state_keys,
        "state_tensor_sha256": state_sha256,
        "dense_independent_fp32_parameters": True,
        "parameter_sharing_used": False,
        "parent_sidecar_required": False,
        "training_steps": 0,
        "optimizer_steps": 0,
        "parent_forward_calls": 0,
        "lineage": {} if lineage is None else copy.deepcopy(dict(lineage)),
    }
    if output_action_rank is not None:
        body["output_action_rank"] = output_action_rank
    return MappingProxyType({**body, "candidate_sha256": canonical_sha256(body)})


def _physical_component_counts(
    *,
    external_hidden_size: int,
    source_internal_channels: int,
    target_internal_channels: int,
    kernel_size: int,
    output_action_rank: int,
) -> Mapping[str, int]:
    hidden = int(external_hidden_size)
    source = int(source_internal_channels)
    target = int(target_internal_channels)
    kernel = int(kernel_size)
    rank = int(output_action_rank)
    if min(hidden, source, target, kernel, rank) <= 0:
        raise Lfm2PartialShortConvFrontierProducerError(
            "Physical component geometry must be strictly positive."
        )
    return MappingProxyType(
        {
            "source": source * (4 * hidden + kernel),
            "qdeim": target * (4 * hidden + kernel),
            "factored": (
                target * (3 * hidden + kernel) + rank * target + hidden * rank
            ),
        }
    )


def _compile_output_action_subspace_with_rank_cap(
    evidence: Any,
    *,
    requested_rank: int,
) -> tuple[Any, Mapping[str, Any]]:
    """Use every authenticated FIT action direction up to the requested cap."""

    if (
        isinstance(requested_rank, bool)
        or not isinstance(requested_rank, int)
        or requested_rank <= 0
    ):
        raise Lfm2PartialShortConvFrontierProducerError(
            "Requested output-action rank cap must be a positive integer."
        )
    deficit: Mapping[str, Any] | None = None
    try:
        compiled = compile_lfm2_output_action_subspace(
            evidence,
            target_rank=requested_rank,
        )
    except Lfm2OutputActionRankDeficitError as exc:
        observed_rank = int(exc.report.get("observed_rank", 0))
        if observed_rank <= 0 or observed_rank >= requested_rank:
            raise Lfm2PartialShortConvFrontierProducerError(
                "Output-action rank-deficit report is not a usable positive cap."
            ) from exc
        deficit = exc.report
        compiled = compile_lfm2_output_action_subspace(
            evidence,
            target_rank=observed_rank,
        )
    effective_rank = int(compiled.rank)
    body = {
        "schema": "neural-forge-lfm2-output-action-dynamic-rank-cap-v1",
        "requested_maximum_rank": requested_rank,
        "effective_authenticated_rank": effective_rank,
        "requested_rank_supported": deficit is None,
        "observed_rank_cap_applied": deficit is not None,
        "rank_deficit_report": None if deficit is None else _plain(deficit),
        "unsupported_directions_invented": 0,
        "intelligence_acceptance_or_rejection_performed": False,
    }
    return compiled, MappingProxyType(
        {**body, "rank_policy_sha256": canonical_sha256(body)}
    )


def _parent_cache_binding(
    *,
    parent_path: Path,
    parent_inventory_sha256: str,
    expected_cache_parent_identity_sha256: str,
) -> Mapping[str, Any]:
    torch = _torch()
    model = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    try:
        identity = fingerprint_lfm2_parent(model).identity_sha256
        if identity != expected_cache_parent_identity_sha256:
            raise Lfm2PartialShortConvFrontierProducerError(
                "Fresh BF16 parent identity differs from the trace cache."
            )
        body = {
            "schema": "neural-forge-lfm2-fused-body-parent-cache-binding-v1",
            "status": "exact-local-parent-cache-source-authenticated",
            "parent_checkpoint_inventory_sha256": parent_inventory_sha256,
            "parent_runtime_materialization_dtype": "bfloat16",
            "parent_state_sha256": fingerprint_lfm2_direct_local_state(model),
            "parent_identity_sha256": identity,
            "parent_unique_parameters": _unique_parameter_elements(model),
            "tokenizer_sha256": fingerprint_local_tokenizer(parent_path),
            "direct_local_checkpoint_source_authenticated": True,
            "hub_loader_permitted": False,
            "network_access_permitted": False,
        }
        return MappingProxyType({**body, "binding_sha256": canonical_sha256(body)})
    finally:
        del model


def _publish(
    *,
    output: Path,
    archive_tensors: Mapping[str, Any],
    plan_body: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - required runtime dependency
        raise RuntimeError(
            "Partial ShortConv production requires safetensors."
        ) from exc
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        archive_path = stage / ARCHIVE_FILENAME
        save_file(
            {
                name: tensor.detach().cpu().contiguous().clone()
                for name, tensor in sorted(archive_tensors.items())
            },
            str(archive_path),
            metadata={
                "format": "pt",
                "neural_forge_role": "partial-short-conv-fold-frontier",
            },
        )
        records = {
            name: _tensor_record(tensor)
            for name, tensor in sorted(archive_tensors.items())
        }
        completed_plan_body = {
            **copy.deepcopy(dict(plan_body)),
            "tensor_archive": {
                "schema": ARCHIVE_SCHEMA,
                "filename": ARCHIVE_FILENAME,
                "file_sha256": _file_sha256(archive_path),
                "tensor_count": len(records),
                "stored_tensor_elements": sum(
                    int(tensor.numel()) for tensor in archive_tensors.values()
                ),
            },
            # The runner intentionally consumes this top-level sealed namespace.
            "tensor_records": records,
        }
        plan = {
            **completed_plan_body,
            "plan_sha256": canonical_sha256(completed_plan_body),
        }
        plan_path = stage / PLAN_FILENAME
        _write_json(plan_path, plan)
        receipt_body = {
            "schema": PRODUCER_RECEIPT_SCHEMA,
            "status": "partial-short-conv-fold-frontier-publication-complete",
            "plan_filename": PLAN_FILENAME,
            "plan_sha256": plan["plan_sha256"],
            "plan_file_sha256": _file_sha256(plan_path),
            "archive_filename": ARCHIVE_FILENAME,
            "archive_file_sha256": _file_sha256(archive_path),
            "candidate_count": len(plan["candidates"]),
            "fit_batch_indices": list(FIT_BATCH_INDICES),
            "heldout_batch_indices": list(HELDOUT_BATCH_INDICES),
            "untouched_batch_indices": list(UNTOUCHED_BATCH_INDICES),
            "direct_local_checkpoint_loading_used": True,
            "selective_authenticated_cache_session_count": 1,
            "parent_model_forward_calls": 0,
            "model_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "network_access": False,
        }
        receipt = {
            **receipt_body,
            "receipt_sha256": canonical_sha256(receipt_body),
        }
        _write_json(stage / PRODUCER_RECEIPT_FILENAME, receipt)
        stage.replace(output)
        return MappingProxyType(receipt)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def prepare(
    *,
    output_directory: Path,
    parent_checkpoint: Path,
    expected_parent_checkpoint_inventory_sha256: str,
    trace_cache: Path,
    expected_trace_cache_key: str,
    expected_trace_cache_manifest_sha256: str,
    expected_trace_cache_completion_sha256: str,
    expected_trace_cache_parent_identity_sha256: str,
    target_internal_channels: int = DEFAULT_TARGET_INTERNAL_CHANNELS,
    output_action_rank: int = DEFAULT_OUTPUT_ACTION_RANK,
    ridge: float = DEFAULT_RIDGE,
) -> Mapping[str, Any]:
    """Compile and atomically publish all four folded component candidates."""

    torch = _torch()
    output = _new_output(output_directory)
    parent_path = _ordinary_directory(parent_checkpoint, label="parent checkpoint")
    cache_path = _ordinary_directory(trace_cache, label="trace cache")
    _forbid_output_inside_inputs(output, (parent_path, cache_path))
    expected_inventory = _digest(
        expected_parent_checkpoint_inventory_sha256,
        label="expected parent checkpoint inventory",
    )
    expected_cache = _digest(expected_trace_cache_key, label="expected trace cache")
    expected_manifest = _digest(
        expected_trace_cache_manifest_sha256,
        label="expected trace cache manifest",
    )
    expected_completion = _digest(
        expected_trace_cache_completion_sha256,
        label="expected trace cache completion",
    )
    expected_cache_parent = _digest(
        expected_trace_cache_parent_identity_sha256,
        label="expected trace cache parent identity",
    )
    if (
        isinstance(target_internal_channels, bool)
        or target_internal_channels != DEFAULT_TARGET_INTERNAL_CHANNELS
        or isinstance(output_action_rank, bool)
        or not isinstance(output_action_rank, int)
        or output_action_rank <= 0
        or output_action_rank > DEFAULT_OUTPUT_ACTION_RANK
    ):
        raise Lfm2PartialShortConvFrontierProducerError(
            "This frontier uses m=256 and an output-action rank cap in [1,128]."
        )
    requested_output_action_rank = output_action_rank

    inventory_sha256 = canonical_sha256(b038._checkpoint_inventory(parent_path))
    if inventory_sha256 != expected_inventory:
        raise Lfm2PartialShortConvFrontierProducerError(
            "Parent checkpoint inventory differs from its external pin."
        )
    cache = open_lfm2_circuit_trace_cache(cache_path, verify=False)
    manifest = cache.manifest
    if (
        cache.cache_key != expected_cache
        or manifest.get("manifest_sha256") != expected_manifest
        or cache.parent_fingerprint != expected_cache_parent
        or cache.batch_count != EXPECTED_CACHE_BATCH_COUNT
    ):
        raise Lfm2PartialShortConvFrontierProducerError(
            "Trace cache key, manifest, parent, or canonical 16-batch shape changed."
        )

    parent_model = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    try:
        parent_state_sha256 = fingerprint_lfm2_direct_local_state(parent_model)
        parent_parameter_elements = _unique_parameter_elements(parent_model)
        module = parent_model.get_submodule("model.layers.0.conv")
        if any(
            bias is not None
            for bias in (module.in_proj.bias, module.conv.bias, module.out_proj.bias)
        ):
            raise Lfm2PartialShortConvFrontierProducerError(
                "Canonical layer-0 ShortConv must remain bias-free."
            )
        source_in = module.in_proj.weight.detach().cpu().float().contiguous().clone()
        source_conv = module.conv.weight.detach().cpu().float().contiguous().clone()
        source_out = module.out_proj.weight.detach().cpu().float().contiguous().clone()
        source_component_elements = sum(
            int(value.numel()) for value in (source_in, source_conv, source_out)
        )
        external_hidden = int(source_out.shape[0])
        source_internal = int(source_conv.shape[0])
        kernel_size = int(source_conv.shape[-1])
        if (
            external_hidden != source_internal
            or source_internal != 1024
            or kernel_size <= 0
        ):
            raise Lfm2PartialShortConvFrontierProducerError(
                "Canonical LFM2 layer-0 ShortConv geometry changed."
            )
        expected_source_elements = source_internal * (4 * external_hidden + kernel_size)
        if source_component_elements != expected_source_elements:
            raise Lfm2PartialShortConvFrontierProducerError(
                "Source ShortConv parameters differ from dense physical accounting."
            )

        source_state_sha256 = lfm2_short_conv_channel_qdeim_source_state_sha256(
            source_in,
            source_conv,
            source_out,
        )
        qdeim = compile_lfm2_short_conv_channel_qdeim(
            source_in_proj_weight=source_in,
            source_conv_weight=source_conv,
            source_out_proj_weight=source_out,
            expected_source_state_sha256=source_state_sha256,
            target_internal_channels=target_internal_channels,
        )
        authenticate_lfm2_short_conv_channel_qdeim(qdeim)
        qdeim_state = dict(qdeim.state_dict())

        with cache.selective_authenticated_session() as session:
            fit_phases = _selected_cache_phases(
                cache=cache,
                session=session,
                batch_indices=FIT_BATCH_INDICES,
            )
            heldout_phases = _selected_cache_phases(
                cache=cache,
                session=session,
                batch_indices=HELDOUT_BATCH_INDICES,
            )
            session.verify()
            selective_session = session.as_dict()
        if selective_session.get("completion_sha256") != expected_completion:
            raise Lfm2PartialShortConvFrontierProducerError(
                "Trace cache completion differs from its external pin."
            )

        fit_ids, fit_action, fit_weights = _output_action_rows(fit_phases)
        held_ids, held_action, held_weights = _output_action_rows(heldout_phases)
        fit_actions = seal_lfm2_output_action_rows(
            role="fit",
            row_ids=fit_ids,
            branch_action_rows=fit_action,
            case_weights=fit_weights,
        )
        heldout_actions = seal_lfm2_output_action_rows(
            role="heldout",
            row_ids=held_ids,
            branch_action_rows=held_action,
            case_weights=held_weights,
        )
        action_evidence = seal_lfm2_output_action_evidence(
            parent_state_sha256=parent_state_sha256,
            cache_sha256=cache.cache_key,
            component_id="model.layers.0.conv.output-action",
            fit=fit_actions,
            heldout=heldout_actions,
        )
        output_subspace, output_rank_policy = (
            _compile_output_action_subspace_with_rank_cap(
                action_evidence,
                requested_rank=requested_output_action_rank,
            )
        )
        output_action_rank = int(output_subspace.rank)
        physical_counts = _physical_component_counts(
            external_hidden_size=external_hidden,
            source_internal_channels=source_internal,
            target_internal_channels=target_internal_channels,
            kernel_size=kernel_size,
            output_action_rank=output_action_rank,
        )
        output_materialization = output_subspace.materialize(dtype=torch.float32)
        output_materialized_state = dict(output_materialization.state_dict())
        projector_sha256 = output_subspace.report["projector"]["sha256"]
        projected_parent_writer = compile_lfm2_output_action_writer_fold(
            output_subspace,
            source_out,
            expected_parent_state_sha256=parent_state_sha256,
            expected_cache_sha256=cache.cache_key,
            expected_subspace_result_sha256=output_subspace.result_sha256,
            expected_basis_receipt_sha256=output_subspace.basis_receipt_sha256,
            expected_projector_sha256=projector_sha256,
            expected_parent_out_proj_sha256=lfm2_authenticated_tensor_sha256(
                source_out
            ),
        )

        selected_channels = qdeim.selected_source_channels
        candidate_kernel = qdeim.conv_weight
        qdeim_candidate_sha256 = lfm2_partial_short_conv_qdeim_candidate_sha256(
            selected_source_channels=selected_channels,
            candidate_fir_kernel=candidate_kernel,
            source_internal_channels=source_internal,
        )
        fit_phase_sha256 = lfm2_partial_short_conv_cached_phase_evidence_sha256(
            fit_phases,
            partition="fit",
        )
        heldout_phase_sha256 = lfm2_partial_short_conv_cached_phase_evidence_sha256(
            heldout_phases,
            partition="heldout",
        )
        fit_writer_evidence = compile_lfm2_partial_short_conv_parent_shell_evidence(
            cached_phases=fit_phases,
            partition="fit",
            expected_cached_phase_evidence_sha256=fit_phase_sha256,
            selected_source_channels=selected_channels,
            candidate_fir_kernel=candidate_kernel,
            expected_qdeim_candidate_sha256=qdeim_candidate_sha256,
        )
        heldout_writer_evidence = compile_lfm2_partial_short_conv_parent_shell_evidence(
            cached_phases=heldout_phases,
            partition="heldout",
            expected_cached_phase_evidence_sha256=heldout_phase_sha256,
            selected_source_channels=selected_channels,
            candidate_fir_kernel=candidate_kernel,
            expected_qdeim_candidate_sha256=qdeim_candidate_sha256,
        )
        writer_id = "model.layers.0.conv.out_proj"
        structural_writer_sha256 = lfm2_parent_shell_writer_state_sha256(
            writer_id=writer_id,
            writer_role=_LOCAL_WRITER_ROLE,
            weight=qdeim.out_proj_weight,
        )
        fit_writer_sha256 = lfm2_parent_shell_local_writer_evidence_sha256(
            fit_writer_evidence.evidence_cases,
            partition="fit",
        )
        heldout_writer_sha256 = lfm2_parent_shell_local_writer_evidence_sha256(
            heldout_writer_evidence.evidence_cases,
            partition="heldout",
        )
        local_writer = compile_lfm2_parent_shell_short_conv_local_writer(
            writer_id=writer_id,
            structural_out_weight=qdeim.out_proj_weight,
            expected_structural_writer_state_sha256=structural_writer_sha256,
            fit_cases=fit_writer_evidence.evidence_cases,
            expected_fit_evidence_sha256=fit_writer_sha256,
            heldout_cases=heldout_writer_evidence.evidence_cases,
            expected_heldout_evidence_sha256=heldout_writer_sha256,
            ridge=ridge,
            fit_unpenalized_bias=False,
        )
        local_writer_state = dict(local_writer.state_dict())
        local_weight = local_writer_state["weight"]
        output_encoder = (
            output_materialized_state["encoder.weight"]
            .to(torch.float64)
            .matmul(local_weight.to(torch.float64))
            .to(torch.float32)
            .contiguous()
            .clone()
        )
        output_decoder = (
            output_materialized_state["decoder.weight"]
            .detach()
            .cpu()
            .float()
            .contiguous()
            .clone()
        )
        if tuple(output_encoder.shape) != (
            output_action_rank,
            target_internal_channels,
        ) or tuple(output_decoder.shape) != (external_hidden, output_action_rank):
            raise Lfm2PartialShortConvFrontierProducerError(
                "Factored local output writer geometry changed."
            )

        archive: dict[str, Any] = {}
        candidates = [
            _candidate_row(
                candidate_id=(
                    f"layer-000-projected-parent-output-rank{output_action_rank}"
                ),
                runtime_kind="projected-parent-output",
                external_hidden_size=external_hidden,
                internal_channels=source_internal,
                kernel_size=kernel_size,
                source_component_parameter_elements=source_component_elements,
                state=projected_parent_writer.state_dict(),
                archive=archive,
                output_action_rank=output_action_rank,
                lineage={
                    "output_action_subspace_result_sha256": (
                        output_subspace.result_sha256
                    ),
                    "output_action_writer_fold_result_sha256": (
                        projected_parent_writer.result_sha256
                    ),
                    "parameter_reduction_claimed": False,
                    "diagnostic_only": True,
                },
            ),
            _candidate_row(
                candidate_id="layer-000-qdeim-structural-m256",
                runtime_kind="qdeim-structural",
                external_hidden_size=external_hidden,
                internal_channels=target_internal_channels,
                kernel_size=kernel_size,
                source_component_parameter_elements=source_component_elements,
                state=qdeim_state,
                archive=archive,
                lineage={
                    "qdeim_result_sha256": qdeim.report["result_sha256"],
                    "selected_source_channels": list(selected_channels),
                    "intact_channel_selection": True,
                },
            ),
            _candidate_row(
                candidate_id="layer-000-qdeim-local-writer-m256",
                runtime_kind="qdeim-local-writer",
                external_hidden_size=external_hidden,
                internal_channels=target_internal_channels,
                kernel_size=kernel_size,
                source_component_parameter_elements=source_component_elements,
                state={
                    "conv.weight": qdeim.conv_weight,
                    "in_proj.weight": qdeim.in_proj_weight,
                    "out_proj.weight": local_weight,
                },
                archive=archive,
                lineage={
                    "qdeim_result_sha256": qdeim.report["result_sha256"],
                    "fit_evidence_result_sha256": fit_writer_evidence.report[
                        "result_sha256"
                    ],
                    "heldout_evidence_result_sha256": (
                        heldout_writer_evidence.report["result_sha256"]
                    ),
                    "local_writer_result_sha256": local_writer.report["result_sha256"],
                    "heldout_used_for_construction": False,
                },
            ),
            _candidate_row(
                candidate_id=(
                    f"layer-000-qdeim-local-writer-output-rank{output_action_rank}"
                ),
                runtime_kind="qdeim-local-writer-output-factor",
                external_hidden_size=external_hidden,
                internal_channels=target_internal_channels,
                kernel_size=kernel_size,
                source_component_parameter_elements=source_component_elements,
                state={
                    "conv.weight": qdeim.conv_weight,
                    "in_proj.weight": qdeim.in_proj_weight,
                    "out_decoder.weight": output_decoder,
                    "out_encoder.weight": output_encoder,
                },
                archive=archive,
                output_action_rank=output_action_rank,
                lineage={
                    "qdeim_result_sha256": qdeim.report["result_sha256"],
                    "local_writer_result_sha256": local_writer.report["result_sha256"],
                    "output_action_subspace_result_sha256": (
                        output_subspace.result_sha256
                    ),
                    "output_action_materialization_sha256": (
                        output_materialization.materialization_sha256
                    ),
                    "factorization": "decoder_Q_times_encoder_QT_W_local",
                },
            ),
        ]
        if int(candidates[0]["candidate_parameter_elements"]) != (
            source_component_elements
        ):
            # Only out_proj is archived for candidate 0, but the runner retains
            # the other two parent tensors in situ.  Count the installed module,
            # not merely the replacement archive payload.
            first = dict(candidates[0])
            first["candidate_parameter_elements"] = source_component_elements
            first["component_compression_ratio"] = 1.0
            first_without_seal = dict(first)
            first_without_seal.pop("candidate_sha256", None)
            first["candidate_sha256"] = canonical_sha256(first_without_seal)
            candidates[0] = MappingProxyType(first)
        if (
            int(candidates[0]["candidate_parameter_elements"])
            != physical_counts["source"]
            or int(candidates[1]["candidate_parameter_elements"])
            != physical_counts["qdeim"]
            or int(candidates[2]["candidate_parameter_elements"])
            != physical_counts["qdeim"]
            or int(candidates[3]["candidate_parameter_elements"])
            != physical_counts["factored"]
        ):
            raise Lfm2PartialShortConvFrontierProducerError(
                "Candidate tensors differ from full installed-component accounting."
            )

        parent_binding = _parent_cache_binding(
            parent_path=parent_path,
            parent_inventory_sha256=inventory_sha256,
            expected_cache_parent_identity_sha256=expected_cache_parent,
        )
        plan_body = {
            "schema": PLAN_SCHEMA,
            "status": "partial-short-conv-fold-frontier-ready-for-manual-review",
            "producer_schema": PRODUCER_SCHEMA,
            "parent_checkpoint_inventory_sha256": inventory_sha256,
            "parent_fp32_state_sha256": parent_state_sha256,
            "parent_parameter_elements": parent_parameter_elements,
            "parent_cache_binding": _plain(parent_binding),
            "trace_cache_authentication": {
                "cache_key": cache.cache_key,
                "manifest_sha256": manifest["manifest_sha256"],
                "completion_sha256": selective_session["completion_sha256"],
                "selective_session_sha256": selective_session["session_sha256"],
                "batch_count": cache.batch_count,
                "fit_batch_indices": list(FIT_BATCH_INDICES),
                "heldout_batch_indices": list(HELDOUT_BATCH_INDICES),
                "untouched_batch_indices": list(UNTOUCHED_BATCH_INDICES),
                "selected_payload_batch_indices": list(
                    FIT_BATCH_INDICES + HELDOUT_BATCH_INDICES
                ),
                "selection_and_validation_payloads_loaded": False,
                "cache_open_verify_argument": False,
                "selective_authenticated_session_count": 1,
            },
            "evidence_partition": {
                "fit_only_constructs_all_folded_tensors": True,
                "heldout_diagnostic_only": True,
                "heldout_can_select_or_reject": False,
                "selection_validation_untouched": True,
                "equal_trajectory_mass": True,
                "per_row_weight": "1/total_rows_in_batch_trajectory",
                "stable_row_id": "batch.NNNNNN/phase/row.NNNN",
                "fit_row_count": len(fit_ids),
                "heldout_row_count": len(held_ids),
            },
            "physical_component_counts": dict(physical_counts),
            "output_action_rank_policy": dict(output_rank_policy),
            "output_action_subspace": output_subspace.as_dict(),
            "output_action_materialization": output_materialization.as_dict(),
            "qdeim": qdeim.as_dict(),
            "partial_short_conv_fit_evidence": fit_writer_evidence.as_dict(),
            "partial_short_conv_heldout_evidence": (heldout_writer_evidence.as_dict()),
            "local_writer": local_writer.as_dict(),
            "candidates": [_plain(candidate) for candidate in candidates],
            "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
            "candidate_count": len(candidates),
            "evaluation_policy": {
                "whole_response_manual_intelligence_review_required": True,
                "exact_token_match_required": False,
                "numeric_metrics_have_decision_authority": False,
                "automatic_acceptance_or_rejection_performed": False,
                "maximum_prompt_tokens": 1024,
                "maximum_new_tokens": 256,
            },
            "operation_audit": {
                "parent_model_forward_calls": 0,
                "candidate_model_forward_calls": 0,
                "model_generate_calls": 0,
                "parent_trace_recaptures": 0,
                "new_parent_trace_rows": 0,
                "training_steps": 0,
                "optimizer_constructions": 0,
                "optimizer_steps": 0,
                "autograd_grad_calls": 0,
                "backward_calls": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "parameters_sparsified": 0,
                "parameter_sharing_used": False,
                "parent_sidecars_required": 0,
                "hub_loader_used": False,
                "network_access": False,
            },
            "interpretation": {
                "diagnostic_parent_shell_only": True,
                "standalone_child_compression_claimed": False,
                "does_not_rewrite_main_dynamic_compiler": True,
                "partial_component_folding_hypothesis_isolated": True,
                "projected_parent_output_is_no_reduction_diagnostic": True,
                "qdeim_structural_is_exact_four_x_component_compaction": True,
                "local_writer_uses_cached_fit_only": True,
                "factored_writer_is_physical_output_action_compaction": True,
            },
        }
        return _publish(
            output=output,
            archive_tensors=archive,
            plan_body=plan_body,
        )
    finally:
        del parent_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-checkpoint-inventory-sha256", required=True)
    parser.add_argument("--trace-cache", type=Path, required=True)
    parser.add_argument("--expected-trace-cache-key", required=True)
    parser.add_argument("--expected-trace-cache-manifest-sha256", required=True)
    parser.add_argument("--expected-trace-cache-completion-sha256", required=True)
    parser.add_argument("--expected-trace-cache-parent-identity-sha256", required=True)
    parser.add_argument(
        "--target-internal-channels",
        type=int,
        default=DEFAULT_TARGET_INTERNAL_CHANNELS,
    )
    parser.add_argument(
        "--output-action-rank",
        type=int,
        default=DEFAULT_OUTPUT_ACTION_RANK,
    )
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = prepare(**vars(arguments))
    print(json.dumps(_plain(receipt), allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
