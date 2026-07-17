"""Exact-50% protected-weight-error ablation for standalone LFM2 checkpoints.

This module is deliberately not a causal allocator.  It exists so the exact
large-target format, hard anchor policy, and independent behavioral gate can
be exercised while the cached-causal frontier is still being produced.

Eligible FFN W1/W3/W2 and attention V/O groups are ordered by incremental
normalized reconstruction SSE times the existing role multiplier.  The
causal multiplier is fixed to exactly one.  Embedding/output, normalization,
Q/K, convolution, bias, and unclassified groups remain at the base width;
normalization tensors, depthwise-convolution kernels, and local special/chat
token rows additionally reload at the exact source dtype.

No frontier, cached activation evidence, parent forward, generation, gradient,
optimizer, backward call, or training operation is used during allocation.
The artifact uses the existing audited packed-value payload, but binds a
distinct manifest record and policy seal so it cannot be mistaken for cached
causal evidence.
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional HF dependency
    raise RuntimeError(
        "PyTorch is required for the LFM2 protected-weight-error allocator."
    ) from exc

from . import hf_lfm2_causal_value_allocator as anchors
from . import hf_lfm2_value_fold_control as packed
from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent


ALLOCATION_KEY = "protected_weight_error_allocation"
TENSOR_METADATA_KEY = "protected_weight_error_allocation"
ALLOCATION_SCHEMA = "neural-forge-lfm2-protected-weight-error-allocation-v1"
SOURCE_BINDING_SCHEMA = "neural-forge-lfm2-protected-weight-error-source-binding-v1"
POLICY_ID = "lfm2_protected_weight_error_role_aware_8b7b_v1"
TARGET_REDUCTION = 0.50


class Lfm2ProtectedWeightErrorAllocatorError(
    packed.Lfm2ValueFoldControlError,
):
    """The protected-weight-error policy or source binding is invalid."""


class Lfm2ProtectedWeightErrorBudgetError(
    packed.Lfm2ValueFoldControlBudgetError,
    Lfm2ProtectedWeightErrorAllocatorError,
):
    """Protected roles leave too few eligible groups for the exact target."""


class Lfm2ProtectedWeightErrorFormatError(
    packed.Lfm2ValueFoldControlFormatError,
    Lfm2ProtectedWeightErrorAllocatorError,
):
    """A checkpoint lacks a valid protected-weight-error allocation seal."""


def _digest(value: Any, name: str) -> str:
    try:
        return anchors._digest(value, name)
    except anchors.Lfm2CausalValueAllocatorError as exc:
        raise Lfm2ProtectedWeightErrorAllocatorError(str(exc)) from exc


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        return anchors._finite_nonnegative(value, name)
    except anchors.Lfm2CausalValueAllocatorError as exc:
        raise Lfm2ProtectedWeightErrorAllocatorError(str(exc)) from exc


def _source_binding(parent_identity_sha256: str) -> dict[str, Any]:
    body = {
        "schema": SOURCE_BINDING_SCHEMA,
        "parent_identity_sha256": _digest(
            parent_identity_sha256,
            "parent_identity_sha256",
        ),
        "cached_causal_frontier_used": False,
        "causal_evidence_used": False,
        "parent_trace_cache_used": False,
        "independent_holdout_used_for_allocation": False,
    }
    return {**body, "source_binding_sha256": canonical_sha256(body)}


def _allocation_order(
    analyses: Sequence[Any],
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], Mapping[str, Any]]:
    """Order eligible groups by protected weight error, with causal factor 1."""

    offsets = []
    roles: dict[str, Any] = {}
    eligible: list[tuple[float, str, int, int]] = []
    cursor = 0
    for analysis in analyses:
        start = cursor
        cursor += int(analysis.group_count)
        offsets.append((start, cursor))
        role = anchors._tensor_role(analysis.canonical_name)
        roles[analysis.canonical_name] = role
        if not role.eligible_for_lower_precision:
            continue
        role_multiplier = float(role.risk_multiplier)
        for group_index, distortion in enumerate(analysis.distortion_penalty):
            incremental = _finite_nonnegative(
                float(distortion),
                f"{analysis.canonical_name}.distortion_penalty",
            )
            risk = (incremental + 1e-30) * role_multiplier
            eligible.append(
                (
                    risk,
                    analysis.canonical_name,
                    group_index,
                    start + group_index,
                )
            )
    eligible.sort(key=lambda value: (value[0], value[1], value[2]))
    return (
        np.asarray([value[3] for value in eligible], dtype=np.int64),
        tuple(offsets),
        roles,
    )


def _allocation_record(
    *,
    parent_identity_sha256: str,
    analyses: Sequence[Any],
    masks: Sequence[np.ndarray],
    roles: Mapping[str, Any],
    eos_token_ids: Sequence[int],
    protected_token_ids: Sequence[int],
    exact_overlays: Mapping[str, Any],
    counterfactual_lower_group_count: int | None,
    bytes_saved_per_lower_group: int,
) -> dict[str, Any]:
    tensors = {}
    protected_groups = 0
    eligible_groups = 0
    lower_groups = 0
    by_role: dict[str, dict[str, int]] = {}
    for analysis, mask in zip(analyses, masks, strict=True):
        role = roles[analysis.canonical_name]
        low = int(np.count_nonzero(mask))
        if not role.eligible_for_lower_precision and low:
            raise AssertionError("A protected tensor received lower precision.")
        group_count = int(analysis.group_count)
        protected_groups += group_count if not role.eligible_for_lower_precision else 0
        eligible_groups += group_count if role.eligible_for_lower_precision else 0
        lower_groups += low
        role_counts = by_role.setdefault(
            role.role,
            {"tensor_count": 0, "group_count": 0, "lower_group_count": 0},
        )
        role_counts["tensor_count"] += 1
        role_counts["group_count"] += group_count
        role_counts["lower_group_count"] += low
        tensors[analysis.canonical_name] = {
            "role": role.role,
            "layer_index": role.layer_index,
            "eligible_for_lower_precision": role.eligible_for_lower_precision,
            "role_risk_multiplier": role.risk_multiplier,
            "fixed_causal_multiplier": (
                1.0 if role.eligible_for_lower_precision else None
            ),
            "normalized_layer_sensitivity": None,
            "group_count": group_count,
            "base_8bit_group_count": group_count - low,
            "lower_7bit_group_count": low,
            "exact_source_dtype_overlay": (analysis.canonical_name in exact_overlays),
        }
    overlay_raw_bytes = sum(
        int(value["exact_raw_bytes"]) for value in exact_overlays.values()
    )
    overlay_base64_characters = sum(
        len(segment["raw_base64"])
        for value in exact_overlays.values()
        for segment in value["segments"]
    )
    counterfactual = (
        None
        if counterfactual_lower_group_count is None
        else int(counterfactual_lower_group_count)
    )
    additional_lower = None if counterfactual is None else lower_groups - counterfactual
    body = {
        "schema": ALLOCATION_SCHEMA,
        "policy_id": POLICY_ID,
        "ablation_label": "protected_weight_error",
        "source_binding": _source_binding(parent_identity_sha256),
        "eos_token_ids": list(eos_token_ids),
        "protected_token_ids": list(protected_token_ids),
        "protected_token_sources": {
            "local_bos_eos_pad_and_chat_control_ids": list(protected_token_ids),
            "cached_calibration_selection_ids": [],
            "sealed_independent_holdout_ids_used": [],
            "validation_or_holdout_blind": True,
        },
        "policy": {
            "base_bit_width": 8,
            "lower_bit_width": 7,
            "fixed_causal_multiplier": 1.0,
            "causal_sensitivity_multiplier_used": False,
            "causal_evidence_used": False,
            "group_risk": (
                "incremental_normalized_reconstruction_sse * "
                "role_risk_multiplier * fixed_causal_multiplier"
            ),
            "role_risk_multipliers": dict(anchors._ROLE_RISK_MULTIPLIERS),
            "hard_protected_roles": sorted(anchors._PROTECTED_ROLES),
            "embedding_and_tied_lm_head_base_width_protected": True,
            "special_and_chat_rows_exact_source_dtype": True,
            "tied_rows_protect_input_identity_and_output_competitor_logits": True,
            "all_norms_exact_source_dtype": True,
            "all_qk_norms_exact_source_dtype": True,
            "all_depthwise_conv_kernels_exact_source_dtype": True,
            "all_attention_qk_base_width_protected": True,
            "all_convolution_state_base_width_protected": True,
            "lower_precision_spend_order_is_global": True,
            "exact_token_match_objective_used": False,
        },
        "counts": {
            "protected_group_count": protected_groups,
            "eligible_group_count": eligible_groups,
            "lower_group_count": lower_groups,
            "base_group_count": protected_groups + eligible_groups - lower_groups,
        },
        "by_role": dict(sorted(by_role.items())),
        "tensors": tensors,
        "exact_overlays": dict(exact_overlays),
        "exact_byte_exchange": {
            "exact_source_value_bytes": overlay_raw_bytes,
            "base64_payload_characters_inside_manifest": (overlay_base64_characters),
            "canonical_overlay_record_bytes": len(
                canonical_json_bytes(dict(exact_overlays))
            ),
            "lower_group_count_without_exact_overlays": counterfactual,
            "additional_lower_groups_paid_for_exact_overlays": additional_lower,
            "bytes_saved_per_additional_lower_group": bytes_saved_per_lower_group,
            "compensating_payload_bytes_saved": (
                None
                if additional_lower is None
                else additional_lower * bytes_saved_per_lower_group
            ),
            "all_exchange_bytes_counted_in_checkpoint_ceiling": True,
        },
        "execution_audit": {
            "frontier_reads": 0,
            "parent_cache_build_calls": 0,
            "parent_forward_calls": 0,
            "parent_generation_calls": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "training_used": False,
        },
    }
    return {**body, "allocation_sha256": canonical_sha256(body)}


def _manifest_document(
    *,
    model: nn.Module,
    target: packed.Lfm2ValueFoldTarget,
    aliases: Mapping[str, str],
    entries: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    dense_tensor_bytes: int,
    shared_nonweight_bytes: int,
    payload_bytes: int,
    compressed_checkpoint_bytes: int,
    manifest_bytes: int,
    lower_group_count: int,
    total_group_count: int,
    tokenizer_fingerprint: str,
    allocation: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    baseline, _raw = packed._manifest_document(
        model=model,
        target=target,
        aliases=aliases,
        entries=entries,
        artifacts=artifacts,
        dense_tensor_bytes=dense_tensor_bytes,
        shared_nonweight_bytes=shared_nonweight_bytes,
        payload_bytes=payload_bytes,
        compressed_checkpoint_bytes=compressed_checkpoint_bytes,
        manifest_bytes=manifest_bytes,
        lower_group_count=lower_group_count,
        total_group_count=total_group_count,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    core = dict(baseline)
    core.pop("manifest_sha256", None)
    core["control_label"] = (
        "PROTECTED-WEIGHT-ERROR ABLATION; EXACT 50% SERIALIZED-BYTE TARGET; "
        "NOT CACHED-CAUSAL EVIDENCE; NOT STRUCTURAL PARAMETER COMPACTION"
    )
    core["method"] = {
        **core["method"],
        "parent_traces_used": False,
        "cached_parent_traces_used": False,
        "new_parent_traces_generated": False,
        "causal_allocation_used": False,
        "protected_weight_error_ablation_used": True,
    }
    core["quantization"] = {
        **core["quantization"],
        "distortion_allocator": (
            "protected roles excluded; eligible groups ordered by incremental "
            "normalized reconstruction SSE times role risk; causal factor 1"
        ),
        "allocator_policy_id": POLICY_ID,
    }
    core[ALLOCATION_KEY] = dict(allocation)
    document = {**core, "manifest_sha256": canonical_sha256(core)}
    return document, canonical_json_bytes(document) + b"\n"


def _budget_layout(
    *,
    model: nn.Module,
    target: packed.Lfm2ValueFoldTarget,
    analyses: Sequence[Any],
    aliases: Mapping[str, str],
    masks: Sequence[np.ndarray],
    roles: Mapping[str, Any],
    parent_identity_sha256: str,
    eos_token_ids: Sequence[int],
    protected_token_ids: Sequence[int],
    exact_overlays: Mapping[str, Any],
    counterfactual_lower_group_count: int | None,
    config_artifact: Mapping[str, Any],
    generation_artifact: Mapping[str, Any],
    tokenizer_artifacts: Sequence[Mapping[str, Any]],
    tokenizer_fingerprint: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any], int]:
    entries, payload_bytes = packed._tensor_entry_layout(analyses, masks, target)
    allocation = _allocation_record(
        parent_identity_sha256=parent_identity_sha256,
        analyses=analyses,
        masks=masks,
        roles=roles,
        eos_token_ids=eos_token_ids,
        protected_token_ids=protected_token_ids,
        exact_overlays=exact_overlays,
        counterfactual_lower_group_count=counterfactual_lower_group_count,
        bytes_saved_per_lower_group=target.bytes_saved_per_lower_group,
    )
    for name, metadata in allocation["tensors"].items():
        entries[name] = {**entries[name], TENSOR_METADATA_KEY: metadata}
    shared_nonweight_bytes = (
        int(config_artifact["size_bytes"])
        + int(generation_artifact["size_bytes"])
        + sum(int(value["size_bytes"]) for value in tokenizer_artifacts)
    )
    dense_bytes = sum(value.source_nbytes for value in analyses)
    lower_count = sum(int(np.count_nonzero(value)) for value in masks)
    total_groups = sum(value.group_count for value in analyses)
    artifacts = packed._placeholder_artifacts(
        config_artifact,
        generation_artifact,
        tokenizer_artifacts,
        payload_bytes,
    )
    other_bytes = shared_nonweight_bytes + payload_bytes

    def factory(total: int, manifest_bytes: int):
        return _manifest_document(
            model=model,
            target=target,
            aliases=aliases,
            entries=entries,
            artifacts=artifacts,
            dense_tensor_bytes=dense_bytes,
            shared_nonweight_bytes=shared_nonweight_bytes,
            payload_bytes=payload_bytes,
            compressed_checkpoint_bytes=total,
            manifest_bytes=manifest_bytes,
            lower_group_count=lower_count,
            total_group_count=total_groups,
            tokenizer_fingerprint=tokenizer_fingerprint,
            allocation=allocation,
        )

    document, raw = packed._stable_manifest(
        other_bytes=other_bytes,
        factory=factory,
    )
    return document, raw, entries, payload_bytes


def _solve_budget(
    *,
    model: nn.Module,
    target: packed.Lfm2ValueFoldTarget,
    analyses: Sequence[Any],
    aliases: Mapping[str, str],
    order: np.ndarray,
    offsets: Sequence[tuple[int, int]],
    roles: Mapping[str, Any],
    parent_identity_sha256: str,
    eos_token_ids: Sequence[int],
    protected_token_ids: Sequence[int],
    exact_overlays: Mapping[str, Any],
    counterfactual_lower_group_count: int | None,
    config_artifact: Mapping[str, Any],
    generation_artifact: Mapping[str, Any],
    tokenizer_artifacts: Sequence[Mapping[str, Any]],
    tokenizer_fingerprint: str,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    lower_count = 0
    while True:
        masks = packed._selection_masks(analyses, offsets, order, lower_count)
        document, _raw, _entries, _payload_bytes = _budget_layout(
            model=model,
            target=target,
            analyses=analyses,
            aliases=aliases,
            masks=masks,
            roles=roles,
            parent_identity_sha256=parent_identity_sha256,
            eos_token_ids=eos_token_ids,
            protected_token_ids=protected_token_ids,
            exact_overlays=exact_overlays,
            counterfactual_lower_group_count=counterfactual_lower_group_count,
            config_artifact=config_artifact,
            generation_artifact=generation_artifact,
            tokenizer_artifacts=tokenizer_artifacts,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        total = int(document["storage"]["compressed_checkpoint_bytes"])
        ceiling = int(document["target"]["target_checkpoint_bytes"])
        if total <= ceiling:
            break
        deficit = total - ceiling
        lower_count += max(
            1,
            math.ceil(deficit / target.bytes_saved_per_lower_group),
        )
        if lower_count > len(order):
            maximum_savings = len(order) * target.bytes_saved_per_lower_group
            raise Lfm2ProtectedWeightErrorBudgetError(
                "The exact 50% checkpoint needs more lower-bit groups than "
                "the protected-weight-error policy permits. Protected "
                "embedding/output, norm, convolution, and attention Q/K "
                f"groups remain immutable; eligible groups can save at most "
                f"{maximum_savings} bytes."
            )
    if lower_count:
        previous_masks = packed._selection_masks(
            analyses,
            offsets,
            order,
            lower_count - 1,
        )
        previous, _raw, _entries, _payload_bytes = _budget_layout(
            model=model,
            target=target,
            analyses=analyses,
            aliases=aliases,
            masks=previous_masks,
            roles=roles,
            parent_identity_sha256=parent_identity_sha256,
            eos_token_ids=eos_token_ids,
            protected_token_ids=protected_token_ids,
            exact_overlays=exact_overlays,
            counterfactual_lower_group_count=counterfactual_lower_group_count,
            config_artifact=config_artifact,
            generation_artifact=generation_artifact,
            tokenizer_artifacts=tokenizer_artifacts,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        if (
            previous["storage"]["compressed_checkpoint_bytes"]
            <= previous["target"]["target_checkpoint_bytes"]
        ):
            raise AssertionError(
                "The protected-weight-error allocator did not choose the "
                "minimal lower-group count."
            )
    return masks, document


def export_lfm2_protected_weight_error_fold(
    model: nn.Module,
    destination: str | os.PathLike[str],
    *,
    tokenizer_directory: str | os.PathLike[str],
    eos_token_ids: Sequence[int],
    group_size: int = 256,
    analysis_chunk_groups: int = 4096,
) -> dict[str, Any]:
    """Export one exact-50% protected-weight-error ablation checkpoint."""

    target = packed.Lfm2ValueFoldTarget.from_reduction(
        TARGET_REDUCTION,
        group_size=group_size,
        analysis_chunk_groups=analysis_chunk_groups,
    )
    parent_identity = fingerprint_lfm2_parent(model).identity_sha256
    eos_ids = tuple(sorted({int(value) for value in eos_token_ids}))
    if not eos_ids or any(
        value < 0 or value >= int(model.config.vocab_size) for value in eos_ids
    ):
        raise Lfm2ProtectedWeightErrorAllocatorError(
            "eos_token_ids must be nonempty and inside the parent vocabulary."
        )
    output = Path(destination).expanduser().absolute()
    if output.exists():
        raise Lfm2ProtectedWeightErrorAllocatorError(
            f"Refusing to replace existing destination {output}."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_source = packed._safe_local_directory(
        tokenizer_directory,
        "tokenizer_directory",
    )
    try:
        protected_token_ids = anchors._local_special_token_ids(tokenizer_source)
    except anchors.Lfm2CausalValueAllocatorError as exc:
        raise Lfm2ProtectedWeightErrorAllocatorError(str(exc)) from exc
    if not set(eos_ids).issubset(protected_token_ids):
        raise Lfm2ProtectedWeightErrorAllocatorError(
            "Declared EOS IDs differ from the locally resolved special tokens."
        )
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        config, generation = packed._config_documents(model)
        packed._write_canonical_json(temporary / packed.CONFIG_FILENAME, config)
        packed._write_canonical_json(
            temporary / packed.GENERATION_CONFIG_FILENAME,
            generation,
        )
        tokenizer_artifacts = packed._copy_tokenizer_artifacts(
            tokenizer_source,
            temporary,
        )
        tokenizer_fingerprint = packed._tokenizer_fingerprint(tokenizer_artifacts)
        config_artifact = packed._artifact(temporary / packed.CONFIG_FILENAME)
        generation_artifact = packed._artifact(
            temporary / packed.GENERATION_CONFIG_FILENAME
        )
        analyses, aliases = packed._analyze_model(model, target)
        order, offsets, roles = _allocation_order(analyses)
        exact_overlays = anchors._exact_overlay_payloads(
            analyses,
            roles,
            protected_token_ids=protected_token_ids,
            hidden_size=int(model.config.hidden_size),
            group_size=target.group_size,
        )
        counterfactual_masks, counterfactual = _solve_budget(
            model=model,
            target=target,
            analyses=analyses,
            aliases=aliases,
            order=order,
            offsets=offsets,
            roles=roles,
            parent_identity_sha256=parent_identity,
            eos_token_ids=eos_ids,
            protected_token_ids=protected_token_ids,
            exact_overlays={},
            counterfactual_lower_group_count=None,
            config_artifact=config_artifact,
            generation_artifact=generation_artifact,
            tokenizer_artifacts=tokenizer_artifacts,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        counterfactual_lower_count = sum(
            int(np.count_nonzero(value)) for value in counterfactual_masks
        )
        masks, planned = _solve_budget(
            model=model,
            target=target,
            analyses=analyses,
            aliases=aliases,
            order=order,
            offsets=offsets,
            roles=roles,
            parent_identity_sha256=parent_identity,
            eos_token_ids=eos_ids,
            protected_token_ids=protected_token_ids,
            exact_overlays=exact_overlays,
            counterfactual_lower_group_count=counterfactual_lower_count,
            config_artifact=config_artifact,
            generation_artifact=generation_artifact,
            tokenizer_artifacts=tokenizer_artifacts,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        segment_digests, _error = packed._write_quantized_payload(
            temporary / packed.PAYLOAD_FILENAME,
            analyses,
            masks,
            target,
            temporary,
        )
        entries, payload_bytes = packed._tensor_entry_layout(
            analyses,
            masks,
            target,
            segment_digests=segment_digests,
        )
        allocation = _allocation_record(
            parent_identity_sha256=parent_identity,
            analyses=analyses,
            masks=masks,
            roles=roles,
            eos_token_ids=eos_ids,
            protected_token_ids=protected_token_ids,
            exact_overlays=exact_overlays,
            counterfactual_lower_group_count=counterfactual_lower_count,
            bytes_saved_per_lower_group=target.bytes_saved_per_lower_group,
        )
        for name, metadata in allocation["tensors"].items():
            entries[name] = {**entries[name], TENSOR_METADATA_KEY: metadata}
        payload_artifact = packed._artifact(temporary / packed.PAYLOAD_FILENAME)
        if payload_artifact["size_bytes"] != payload_bytes:
            raise AssertionError("Physical payload size differs from the plan.")
        shared_nonweight_bytes = (
            config_artifact["size_bytes"]
            + generation_artifact["size_bytes"]
            + sum(value["size_bytes"] for value in tokenizer_artifacts)
        )
        dense_bytes = sum(value.source_nbytes for value in analyses)
        lower_count = sum(int(np.count_nonzero(value)) for value in masks)
        total_groups = sum(value.group_count for value in analyses)
        artifacts = {
            "config": config_artifact,
            "generation_config": generation_artifact,
            "payload": payload_artifact,
            "tokenizer": list(tokenizer_artifacts),
        }
        other_bytes = shared_nonweight_bytes + payload_bytes

        def factory(total: int, manifest_bytes: int):
            return _manifest_document(
                model=model,
                target=target,
                aliases=aliases,
                entries=entries,
                artifacts=artifacts,
                dense_tensor_bytes=dense_bytes,
                shared_nonweight_bytes=shared_nonweight_bytes,
                payload_bytes=payload_bytes,
                compressed_checkpoint_bytes=total,
                manifest_bytes=manifest_bytes,
                lower_group_count=lower_count,
                total_group_count=total_groups,
                tokenizer_fingerprint=tokenizer_fingerprint,
                allocation=allocation,
            )

        manifest, manifest_raw = packed._stable_manifest(
            other_bytes=other_bytes,
            factory=factory,
        )
        if (
            manifest["storage"]["compressed_checkpoint_bytes"]
            > manifest["target"]["target_checkpoint_bytes"]
        ):
            raise Lfm2ProtectedWeightErrorBudgetError(
                "Final hashes changed the manifest enough to exceed 50%."
            )
        if (
            planned[ALLOCATION_KEY]["counts"]["lower_group_count"]
            != allocation["counts"]["lower_group_count"]
        ):
            raise AssertionError("Final protected-weight-error allocation changed.")
        if (
            counterfactual["target"]["requested_reduction"] != TARGET_REDUCTION
            or allocation["exact_byte_exchange"][
                "additional_lower_groups_paid_for_exact_overlays"
            ]
            != lower_count - counterfactual_lower_count
            or lower_count < counterfactual_lower_count
        ):
            raise AssertionError("Exact protected-byte exchange changed.")
        packed._write_new(
            temporary / packed.MANIFEST_FILENAME,
            manifest_raw,
        )
        observed_total = sum(
            path.stat().st_size for path in temporary.iterdir() if path.is_file()
        )
        if observed_total != manifest["storage"]["compressed_checkpoint_bytes"]:
            raise AssertionError(
                "On-disk checkpoint byte count differs from the manifest."
            )
        temporary.replace(output)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _validated_manifest(
    checkpoint_directory: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any]]:
    root, manifest, _paths = packed._manifest_and_artifacts(checkpoint_directory)
    allocation = manifest.get(ALLOCATION_KEY)
    if not isinstance(allocation, Mapping):
        raise Lfm2ProtectedWeightErrorFormatError(
            "Packed checkpoint has no protected-weight-error allocation record."
        )
    if "causal_value_allocation" in manifest:
        raise Lfm2ProtectedWeightErrorFormatError(
            "Protected-weight-error artifact cannot also claim causal allocation."
        )
    body = dict(allocation)
    allocation_sha = body.pop("allocation_sha256", None)
    source_binding = body.get("source_binding")
    if (
        body.get("schema") != ALLOCATION_SCHEMA
        or body.get("policy_id") != POLICY_ID
        or body.get("ablation_label") != "protected_weight_error"
        or _digest(allocation_sha, "allocation_sha256") != canonical_sha256(body)
        or not isinstance(source_binding, Mapping)
    ):
        raise Lfm2ProtectedWeightErrorFormatError(
            "Protected-weight-error allocation seal or policy is invalid."
        )
    source_body = dict(source_binding)
    source_sha = source_body.pop("source_binding_sha256", None)
    if (
        source_body.get("schema") != SOURCE_BINDING_SCHEMA
        or _digest(source_sha, "source_binding_sha256") != canonical_sha256(source_body)
        or _digest(
            source_body.get("parent_identity_sha256"),
            "parent_identity_sha256",
        )
        != source_body["parent_identity_sha256"]
        or source_body.get("cached_causal_frontier_used") is not False
        or source_body.get("causal_evidence_used") is not False
        or source_body.get("parent_trace_cache_used") is not False
        or source_body.get("independent_holdout_used_for_allocation") is not False
    ):
        raise Lfm2ProtectedWeightErrorFormatError(
            "Protected-weight-error source binding is invalid."
        )
    method = manifest.get("method")
    quantization = manifest.get("quantization")
    policy = body.get("policy")
    if (
        not isinstance(method, Mapping)
        or method.get("parent_traces_used") is not False
        or method.get("cached_parent_traces_used") is not False
        or method.get("causal_allocation_used") is not False
        or method.get("protected_weight_error_ablation_used") is not True
        or not isinstance(quantization, Mapping)
        or quantization.get("allocator_policy_id") != POLICY_ID
        or not isinstance(policy, Mapping)
        or policy.get("fixed_causal_multiplier") != 1.0
        or policy.get("causal_sensitivity_multiplier_used") is not False
        or policy.get("causal_evidence_used") is not False
    ):
        raise Lfm2ProtectedWeightErrorFormatError(
            "Manifest relabels the protected-weight-error ablation."
        )
    counts = body.get("counts")
    tensors = body.get("tensors")
    exact_overlays = body.get("exact_overlays")
    entries = manifest.get("tensors")
    if (
        not isinstance(counts, Mapping)
        or not isinstance(tensors, Mapping)
        or not isinstance(exact_overlays, Mapping)
        or not isinstance(entries, Mapping)
        or set(tensors) != set(entries)
        or counts.get("lower_group_count") != quantization.get("lower_group_count")
    ):
        raise Lfm2ProtectedWeightErrorFormatError(
            "Protected-weight-error tensor allocation differs from the payload."
        )
    observed_lower = 0
    observed_protected = 0
    for name, metadata in tensors.items():
        entry = entries[name]
        if entry.get(TENSOR_METADATA_KEY) != metadata:
            raise Lfm2ProtectedWeightErrorFormatError(
                f"Tensor {name!r} allocation metadata is inconsistent."
            )
        low = metadata.get("lower_7bit_group_count")
        eligible = metadata.get("eligible_for_lower_precision")
        if (
            isinstance(low, bool)
            or not isinstance(low, int)
            or low < 0
            or low != entry.get("lower_group_count")
            or eligible not in {True, False}
            or (eligible is False and low != 0)
            or metadata.get("normalized_layer_sensitivity") is not None
            or metadata.get("fixed_causal_multiplier") != (1.0 if eligible else None)
        ):
            raise Lfm2ProtectedWeightErrorFormatError(
                f"Tensor {name!r} violates the protected-role ablation policy."
            )
        observed_lower += low
        if eligible is False:
            observed_protected += int(entry["group_count"])
    if (
        observed_lower != counts.get("lower_group_count")
        or observed_protected != counts.get("protected_group_count")
        or tensors["model.embed_tokens.weight"]["lower_7bit_group_count"] != 0
    ):
        raise Lfm2ProtectedWeightErrorFormatError(
            "Protected/lower group totals are inconsistent."
        )
    protected_ids = body.get("protected_token_ids")
    eos_ids = body.get("eos_token_ids")
    token_sources = body.get("protected_token_sources")
    if (
        not isinstance(protected_ids, list)
        or not protected_ids
        or len(set(protected_ids)) != len(protected_ids)
        or protected_ids != sorted(protected_ids)
        or not isinstance(eos_ids, list)
        or not set(eos_ids).issubset(protected_ids)
        or not isinstance(token_sources, Mapping)
        or token_sources.get("sealed_independent_holdout_ids_used") != []
        or token_sources.get("validation_or_holdout_blind") is not True
    ):
        raise Lfm2ProtectedWeightErrorFormatError(
            "Protected local token rows are invalid."
        )
    overlay_raw_bytes = 0
    overlay_base64_characters = 0
    for name, metadata in exact_overlays.items():
        if name not in entries or not isinstance(metadata, Mapping):
            raise Lfm2ProtectedWeightErrorFormatError(
                "Exact source-dtype overlay names differ from packed tensors."
            )
        entry = entries[name]
        if (
            metadata.get("shape") != entry.get("shape")
            or metadata.get("source_dtype") != entry.get("source_dtype")
            or metadata.get("source_numel") != entry.get("source_numel")
            or not isinstance(metadata.get("segments"), list)
            or not metadata["segments"]
        ):
            raise Lfm2ProtectedWeightErrorFormatError(
                f"Exact overlay {name!r} metadata is invalid."
            )
        previous_end = 0
        observed_values = 0
        observed_bytes = 0
        for segment in metadata["segments"]:
            if not isinstance(segment, Mapping):
                raise Lfm2ProtectedWeightErrorFormatError(
                    f"Exact overlay {name!r} has a malformed segment."
                )
            start = segment.get("flat_start")
            count = segment.get("value_count")
            encoded = segment.get("raw_base64")
            raw_size = segment.get("raw_size_bytes")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or start < previous_end
                or count <= 0
                or start + count > int(entry["source_numel"])
                or not isinstance(encoded, str)
                or isinstance(raw_size, bool)
                or not isinstance(raw_size, int)
                or raw_size != count * 2
            ):
                raise Lfm2ProtectedWeightErrorFormatError(
                    f"Exact overlay {name!r} segment range is invalid."
                )
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as exc:
                raise Lfm2ProtectedWeightErrorFormatError(
                    f"Exact overlay {name!r} base64 is invalid."
                ) from exc
            if len(raw) != raw_size or hashlib.sha256(raw).hexdigest() != segment.get(
                "raw_sha256"
            ):
                raise Lfm2ProtectedWeightErrorFormatError(
                    f"Exact overlay {name!r} source bytes failed authentication."
                )
            previous_end = start + count
            observed_values += count
            observed_bytes += raw_size
            overlay_base64_characters += len(encoded)
        if observed_values != metadata.get(
            "exact_value_count"
        ) or observed_bytes != metadata.get("exact_raw_bytes"):
            raise Lfm2ProtectedWeightErrorFormatError(
                f"Exact overlay {name!r} totals are inconsistent."
            )
        overlay_raw_bytes += observed_bytes
    exchange = body.get("exact_byte_exchange")
    if (
        not isinstance(exchange, Mapping)
        or exchange.get("exact_source_value_bytes") != overlay_raw_bytes
        or exchange.get("base64_payload_characters_inside_manifest")
        != overlay_base64_characters
        or exchange.get("all_exchange_bytes_counted_in_checkpoint_ceiling") is not True
    ):
        raise Lfm2ProtectedWeightErrorFormatError(
            "Exact protected-byte exchange accounting is inconsistent."
        )
    return root, manifest


def _install_exact_overlays(
    model: nn.Module,
    manifest: Mapping[str, Any],
) -> int:
    allocation = manifest[ALLOCATION_KEY]
    overlay_metadata = allocation["exact_overlays"]
    names = sorted(manifest["tensors"])
    registry = model.neural_forge_packed_registry
    if len(names) != len(registry):
        raise Lfm2ProtectedWeightErrorFormatError(
            "Packed registry differs from the ablation tensor inventory."
        )
    stores = dict(zip(names, registry, strict=True))
    overlays: dict[int, anchors.ExactOverlayPackedTensor] = {}
    overlay_modules = []
    target_device = registry[0].scales.device
    for name, metadata in overlay_metadata.items():
        base = stores[name]
        runtime_segments = []
        for segment in metadata["segments"]:
            raw = base64.b64decode(segment["raw_base64"], validate=True)
            values = (
                torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                .clone()
                .view(base.source_dtype)
                .to(target_device)
            )
            if int(values.numel()) != int(segment["value_count"]):
                raise Lfm2ProtectedWeightErrorFormatError(
                    "Exact overlay runtime value count changed after decoding."
                )
            runtime_segments.append((int(segment["flat_start"]), values))
        overlay = anchors.ExactOverlayPackedTensor(base, runtime_segments)
        overlays[id(base)] = overlay
        overlay_modules.append(overlay)
    for module in model.modules():
        for attribute in ("_packed_weight", "_packed_bias"):
            current = getattr(module, attribute, None)
            replacement = overlays.get(id(current))
            if replacement is not None:
                object.__setattr__(module, attribute, replacement)
    model.add_module(
        "neural_forge_exact_overlay_registry",
        nn.ModuleList(overlay_modules),
    )
    expected_tied = overlays.get(id(stores["model.embed_tokens.weight"]))
    if (
        expected_tied is None
        or model.model.embed_tokens._packed_weight is not expected_tied
        or model.lm_head._packed_weight is not expected_tied
    ):
        raise Lfm2ProtectedWeightErrorFormatError(
            "Exact protected token rows did not remain tied on fresh reload."
        )
    observed = sum(value.exact_nbytes for value in overlay_modules)
    expected = allocation["exact_byte_exchange"]["exact_source_value_bytes"]
    if observed != expected:
        raise Lfm2ProtectedWeightErrorFormatError(
            "Exact overlay runtime bytes differ from checkpoint accounting."
        )
    return observed


def load_lfm2_protected_weight_error_fold(
    checkpoint_directory: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    max_checkpoint_bytes: int | None = None,
    max_packed_runtime_bytes: int | None = None,
) -> nn.Module:
    """Freshly reload the ablation checkpoint through the strict base loader."""

    _root, manifest = _validated_manifest(checkpoint_directory)
    model = packed.load_lfm2_value_fold_control(
        checkpoint_directory,
        device=device,
        max_checkpoint_bytes=max_checkpoint_bytes,
        max_packed_runtime_bytes=max_packed_runtime_bytes,
    )
    exact_runtime_bytes = _install_exact_overlays(model, manifest)
    total_runtime_bytes = (
        model.neural_forge_load_report["packed_runtime_bytes"] + exact_runtime_bytes
    )
    if (
        max_packed_runtime_bytes is not None
        and total_runtime_bytes > max_packed_runtime_bytes
    ):
        raise Lfm2ProtectedWeightErrorBudgetError(
            "Packed plus exact-overlay runtime bytes exceed max_packed_runtime_bytes."
        )
    allocation = manifest[ALLOCATION_KEY]
    model.neural_forge_load_report = {
        **model.neural_forge_load_report,
        "protected_weight_error_allocator_policy_id": POLICY_ID,
        "protected_weight_error_allocation_sha256": allocation["allocation_sha256"],
        "causal_evidence_used": False,
        "exact_source_dtype_overlay_bytes": exact_runtime_bytes,
        "total_packed_plus_exact_runtime_bytes": total_runtime_bytes,
    }
    return model


def load_lfm2_protected_weight_error_tokenizer(
    checkpoint_directory: str | os.PathLike[str],
) -> Any:
    _validated_manifest(checkpoint_directory)
    return packed.load_lfm2_value_fold_tokenizer(checkpoint_directory)


def inspect_lfm2_protected_weight_error_fold(
    checkpoint_directory: str | os.PathLike[str],
) -> dict[str, Any]:
    root, manifest = _validated_manifest(checkpoint_directory)
    baseline = packed.inspect_lfm2_value_fold_control(root)
    allocation = manifest[ALLOCATION_KEY]
    exchange = allocation["exact_byte_exchange"]
    return {
        **baseline,
        "allocator_policy_id": POLICY_ID,
        "allocation_sha256": allocation["allocation_sha256"],
        "parent_identity_sha256": allocation["source_binding"][
            "parent_identity_sha256"
        ],
        "ablation_label": "protected_weight_error",
        "causal_frontier_used": False,
        "causal_evidence_used": False,
        "fixed_causal_multiplier": 1.0,
        "protected_group_count": allocation["counts"]["protected_group_count"],
        "eligible_group_count": allocation["counts"]["eligible_group_count"],
        "lower_group_count": allocation["counts"]["lower_group_count"],
        "exact_source_dtype_overlay_bytes": exchange["exact_source_value_bytes"],
        "exact_overlay_manifest_base64_characters": exchange[
            "base64_payload_characters_inside_manifest"
        ],
        "additional_lower_groups_paid_for_exact_overlays": exchange[
            "additional_lower_groups_paid_for_exact_overlays"
        ],
        "compensating_payload_bytes_saved": exchange[
            "compensating_payload_bytes_saved"
        ],
        "embedding_lower_group_count": allocation["tensors"][
            "model.embed_tokens.weight"
        ]["lower_7bit_group_count"],
        "all_norm_qk_conv_embedding_groups_base_width": all(
            value["lower_7bit_group_count"] == 0
            for value in allocation["tensors"].values()
            if value["role"] in anchors._PROTECTED_ROLES
        ),
        "parent_forward_calls": allocation["execution_audit"]["parent_forward_calls"],
        "optimizer_steps": allocation["execution_audit"]["optimizer_steps"],
        "backward_calls": allocation["execution_audit"]["backward_calls"],
    }


__all__ = [
    "ALLOCATION_KEY",
    "ALLOCATION_SCHEMA",
    "Lfm2ProtectedWeightErrorAllocatorError",
    "Lfm2ProtectedWeightErrorBudgetError",
    "Lfm2ProtectedWeightErrorFormatError",
    "POLICY_ID",
    "SOURCE_BINDING_SCHEMA",
    "TARGET_REDUCTION",
    "export_lfm2_protected_weight_error_fold",
    "inspect_lfm2_protected_weight_error_fold",
    "load_lfm2_protected_weight_error_fold",
    "load_lfm2_protected_weight_error_tokenizer",
]
