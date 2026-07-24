"""Automatic no-training LFM2 token-interface folding frontier.

The strict tokenizer bridge proves that every target row has one donor anchor,
but that anchor is not the whole tokenization story.  A retained target token
can correspond to different donor tokens (or donor-token sequences) at lexical
boundaries.  This module deterministically recovers that context envelope from
the two sealed local tokenizers and folds it into dense target input/output
operators.  It also supports a compiler-owned untied output blank so input
composition and next-token decoding can use different analytic maps.

No parent forward, training, autograd, optimizer, pruning, quantization,
sparsity, network access, parent sidecar, or runtime parent sharing is used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.hf_lfm2_bridge_only_control import (
    _all_named_tensors,
    _body_records,
    _config_topology,
    _storage_identity,
    _tensor_sha256,
    _unique_parameter_count,
)
from neural_forge.hf_tokenizer_bridge import StrictTokenizerBridgeArtifact


LFM2_INTERFACE_FRONTIER_SCHEMA = "neural-forge-lfm2-interface-frontier-v1"
LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA = (
    "neural-forge-lfm2-interface-context-sequence-plan-v1"
)
LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA = (
    "neural-forge-lfm2-interface-inverse-output-plan-v2"
)
LFM2_CACHED_LOGIT_RESIDUAL_HEAD_SCHEMA = (
    "neural-forge-lfm2-cached-logit-residual-head-v1"
)
LFM2_UNTIED_INTERFACE_BODY_REBALANCE_SCHEMA = (
    "neural-forge-lfm2-untied-interface-body-rebalance-v1"
)
DEFAULT_INTERFACE_CONTEXTS = ("", " ", "A", "The ", "\n", "0")


class Lfm2InterfaceFrontierError(RuntimeError):
    """An automatic interface candidate could not be authenticated."""


@dataclass(frozen=True, slots=True)
class Lfm2InterfaceRecipe:
    """One deterministic dense interface transport recipe."""

    candidate_id: str
    input_strict_anchor_ppm: int
    input_preserve_anchor_norm: bool
    output_mode: str
    output_strict_anchor_ppm: int = 0
    output_preserve_anchor_norm: bool = True
    output_bucket_weighting: str = "uniform"

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise Lfm2InterfaceFrontierError("candidate_id must be nonempty text.")
        object.__setattr__(self, "candidate_id", self.candidate_id.strip())
        for name in ("input_strict_anchor_ppm", "output_strict_anchor_ppm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 1_000_000:
                raise Lfm2InterfaceFrontierError(
                    f"{name} must be an integer from zero through one million."
                )
        for name in (
            "input_preserve_anchor_norm",
            "output_preserve_anchor_norm",
        ):
            if not isinstance(getattr(self, name), bool):
                raise Lfm2InterfaceFrontierError(f"{name} must be boolean.")
        if self.output_mode not in {
            "tied-to-input",
            "strict-row",
            "context-envelope",
            "inverse-bucket-centroid",
            "prefix-transition-centroid",
            "cached-logit-residual",
        }:
            raise Lfm2InterfaceFrontierError("Unsupported output_mode.")
        if self.output_bucket_weighting not in {"uniform", "row-norm"}:
            raise Lfm2InterfaceFrontierError(
                "output_bucket_weighting must be uniform or row-norm."
            )

    @property
    def untied_output(self) -> bool:
        return self.output_mode != "tied-to-input"

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "input_transport": "context-envelope-sequence-mean",
            "input_strict_anchor_ppm": self.input_strict_anchor_ppm,
            "input_preserve_anchor_norm": self.input_preserve_anchor_norm,
            "output_mode": self.output_mode,
            "output_strict_anchor_ppm": self.output_strict_anchor_ppm,
            "output_preserve_anchor_norm": self.output_preserve_anchor_norm,
            "output_bucket_weighting": self.output_bucket_weighting,
            "untied_output": self.untied_output,
        }


DEFAULT_INTERFACE_RECIPES = (
    Lfm2InterfaceRecipe(
        candidate_id="context-mean-norm-tied",
        input_strict_anchor_ppm=0,
        input_preserve_anchor_norm=True,
        output_mode="tied-to-input",
    ),
    Lfm2InterfaceRecipe(
        candidate_id="context-anchor500-norm-tied",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="tied-to-input",
    ),
    Lfm2InterfaceRecipe(
        candidate_id="context-anchor500-input-strict-output-untied",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="strict-row",
        output_strict_anchor_ppm=1_000_000,
        output_preserve_anchor_norm=True,
    ),
    Lfm2InterfaceRecipe(
        candidate_id="context-mean-input-anchor500-output-untied",
        input_strict_anchor_ppm=0,
        input_preserve_anchor_norm=True,
        output_mode="context-envelope",
        output_strict_anchor_ppm=500_000,
        output_preserve_anchor_norm=True,
    ),
    Lfm2InterfaceRecipe(
        candidate_id="context-anchor500-input-inverse-bucket-output-untied",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="inverse-bucket-centroid",
        output_strict_anchor_ppm=250_000,
        output_preserve_anchor_norm=True,
        output_bucket_weighting="row-norm",
    ),
)


@dataclass(frozen=True, slots=True)
class Lfm2InterfaceFrontierResult:
    """One installed standalone candidate and its sealed structural receipt."""

    model: Any
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.model is None:
            raise TypeError("model must be a materialized LFM2 child.")
        body = dict(self.receipt)
        supplied = body.pop("result_sha256", None)
        if supplied != canonical_sha256(body):
            raise Lfm2InterfaceFrontierError(
                "Interface-frontier receipt seal is invalid."
            )
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


@dataclass(frozen=True, slots=True)
class Lfm2CachedLogitResidualHead:
    """One dense output table compiled analytically from sealed cache rows."""

    table: Any
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        if (
            not isinstance(self.table, torch.Tensor)
            or self.table.ndim != 2
            or not torch.is_floating_point(self.table)
            or torch.is_complex(self.table)
            or self.table.layout != torch.strided
            or not bool(torch.isfinite(self.table).all())
        ):
            raise Lfm2InterfaceFrontierError(
                "Cached-logit residual head must be one finite dense matrix."
            )
        body = dict(self.receipt)
        supplied = body.pop("result_sha256", None)
        if (
            supplied != canonical_sha256(body)
            or body.get("schema") != LFM2_CACHED_LOGIT_RESIDUAL_HEAD_SCHEMA
            or body.get("output_table_sha256") != _tensor_sha256(self.table)
        ):
            raise Lfm2InterfaceFrontierError(
                "Cached-logit residual-head receipt seal is invalid."
            )
        object.__setattr__(self, "table", self.table.detach().cpu().contiguous())
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime boundary
        raise RuntimeError("LFM2 interface folding requires PyTorch.") from exc
    return torch


def _body_topology(topology: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(topology)
    result.pop("vocabulary_size", None)
    result.pop("tie_word_embeddings", None)
    return result


def _verify_base_blank(receipt: Mapping[str, Any]) -> str:
    body = dict(receipt)
    supplied = body.pop("report_sha256", None)
    if (
        supplied != canonical_sha256(body)
        or receipt.get("compiler_created_blank") is not True
        or receipt.get("all_parameters_zero") is not True
        or receipt.get("dense_materialized_parameters") is not True
        or receipt.get("training_used") is not False
        or receipt.get("pruning_used") is not False
        or receipt.get("quantization_used") is not False
        or receipt.get("weights_installed") is not False
    ):
        raise Lfm2InterfaceFrontierError(
            "Target is not an authenticated compiler-created dense zero blank."
        )
    assert isinstance(supplied, str)
    return supplied


def _interface_storage_ids(model: Any) -> set[tuple[int, int, tuple[int, ...]]]:
    input_weight = getattr(model.get_input_embeddings(), "weight", None)
    output_weight = getattr(model.get_output_embeddings(), "weight", None)
    if input_weight is None or output_weight is None:
        raise Lfm2InterfaceFrontierError(
            "LFM2 model lacks materialized input/output weights."
        )
    return {_storage_identity(input_weight), _storage_identity(output_weight)}


def _target_token_surface(tokenizer: Any, token_id: int) -> str:
    try:
        value = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:  # pragma: no cover - legacy tokenizer boundary
        value = tokenizer.decode([token_id])
    if not isinstance(value, str):
        raise Lfm2InterfaceFrontierError("Tokenizer decode must return text.")
    return value


def _tokenize_with_offsets(tokenizer: Any, value: str) -> tuple[tuple[int, int, int], ...]:
    try:
        encoded = tokenizer(
            value,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2InterfaceFrontierError(
            "Donor tokenizer must expose exact local offset tokenization."
        ) from exc
    if not isinstance(encoded, Mapping):
        raise Lfm2InterfaceFrontierError("Tokenizer output must be a mapping.")
    raw_ids = encoded.get("input_ids")
    raw_offsets = encoded.get("offset_mapping")
    if (
        not isinstance(raw_ids, Sequence)
        or isinstance(raw_ids, (str, bytes))
        or not isinstance(raw_offsets, Sequence)
        or isinstance(raw_offsets, (str, bytes))
        or len(raw_ids) != len(raw_offsets)
    ):
        raise Lfm2InterfaceFrontierError(
            "Tokenizer IDs and offsets must be aligned sequences."
        )
    rows = []
    for raw_id, raw_offset in zip(raw_ids, raw_offsets, strict=True):
        if (
            isinstance(raw_id, bool)
            or not isinstance(raw_id, int)
            or not isinstance(raw_offset, Sequence)
            or len(raw_offset) != 2
        ):
            raise Lfm2InterfaceFrontierError("Tokenizer offset row is invalid.")
        start, end = raw_offset
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise Lfm2InterfaceFrontierError("Tokenizer offsets are invalid.")
        rows.append((int(raw_id), start, end))
    return tuple(rows)


def compile_lfm2_interface_sequence_plan(
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    *,
    contexts: Sequence[str] = DEFAULT_INTERFACE_CONTEXTS,
) -> dict[str, Any]:
    """Compile all target rows into authenticated donor context sequences."""

    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    bridge.verify_live_tokenizers(donor_tokenizer, target_tokenizer)
    normalized_contexts = tuple(contexts)
    if (
        not normalized_contexts
        or any(not isinstance(value, str) for value in normalized_contexts)
        or len(set(normalized_contexts)) != len(normalized_contexts)
    ):
        raise Lfm2InterfaceFrontierError(
            "contexts must be a nonempty unique text sequence."
        )
    profile = bridge.transport_profile
    special_ids = {
        int(value) for value in getattr(target_tokenizer, "all_special_ids", ())
    }
    records: list[dict[str, Any]] = []
    sequence_length_histogram: dict[int, int] = {}
    variant_count_histogram: dict[int, int] = {}
    rows_with_context_variants = 0
    for target_id, raw_anchor in enumerate(
        profile.target_to_donor_row_indices
    ):
        anchor = int(raw_anchor)
        variants: set[tuple[int, ...]] = {(anchor,)}
        surface = _target_token_surface(target_tokenizer, target_id)
        if target_id not in special_ids and surface:
            for prefix in normalized_contexts:
                selected = tuple(
                    token_id
                    for token_id, _start, end in _tokenize_with_offsets(
                        donor_tokenizer,
                        prefix + surface,
                    )
                    if end > len(prefix)
                )
                if selected:
                    variants.add(selected)
        ordered = tuple(sorted(variants))
        if any(
            not sequence
            or min(sequence) < 0
            or max(sequence) >= profile.donor_row_capacity
            for sequence in ordered
        ):
            raise Lfm2InterfaceFrontierError(
                "Context sequence addresses an invalid donor row."
            )
        if ordered != ((anchor,),):
            rows_with_context_variants += 1
        variant_count_histogram[len(ordered)] = (
            variant_count_histogram.get(len(ordered), 0) + 1
        )
        for sequence in ordered:
            sequence_length_histogram[len(sequence)] = (
                sequence_length_histogram.get(len(sequence), 0) + 1
            )
        records.append(
            {
                "target_id": target_id,
                "strict_donor_id": anchor,
                "donor_sequences": [list(value) for value in ordered],
            }
        )
    body = {
        "schema": LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "contexts": list(normalized_contexts),
        "target_row_count": profile.target_row_capacity,
        "rows_with_context_variants": rows_with_context_variants,
        "variant_count_histogram": {
            str(key): value for key, value in sorted(variant_count_histogram.items())
        },
        "sequence_length_histogram": {
            str(key): value for key, value in sorted(sequence_length_histogram.items())
        },
        "records": records,
        "parent_forward_calls": 0,
        "training_steps": 0,
        "network_access": False,
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def _encode_without_specials(tokenizer: Any, value: str) -> tuple[int, ...]:
    encoder = getattr(tokenizer, "encode", None)
    if not callable(encoder):
        raise Lfm2InterfaceFrontierError("Tokenizer must expose encode().")
    try:
        raw = encoder(value, add_special_tokens=False)
    except TypeError:  # pragma: no cover - legacy tokenizer boundary
        raw = encoder(value)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise Lfm2InterfaceFrontierError("Tokenizer encode must return IDs.")
    values = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise Lfm2InterfaceFrontierError(
                "Tokenizer encode returned an invalid token ID."
            )
        values.append(int(item))
    return tuple(values)


def compile_lfm2_inverse_output_plan(
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    *,
    fallback_target_id: int | None = None,
) -> dict[str, Any]:
    """Assign every donor output row to its first target-token decomposition."""

    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    bridge.verify_live_tokenizers(donor_tokenizer, target_tokenizer)
    profile = bridge.transport_profile
    reverse_anchor = {
        int(donor_id): target_id
        for target_id, donor_id in enumerate(
            profile.target_to_donor_row_indices
        )
    }
    special_assignments: dict[int, int] = {}
    for role in ("bos", "eos", "pad", "unk"):
        donor_id = getattr(donor_tokenizer, f"{role}_token_id", None)
        target_id = getattr(target_tokenizer, f"{role}_token_id", None)
        if donor_id is None or target_id is None:
            continue
        if (
            isinstance(donor_id, bool)
            or not isinstance(donor_id, int)
            or isinstance(target_id, bool)
            or not isinstance(target_id, int)
            or not 0 <= donor_id < profile.donor_row_capacity
            or not 0 <= target_id < profile.target_row_capacity
        ):
            raise Lfm2InterfaceFrontierError(
                "Tokenizer special-token roles are outside bridge capacities."
            )
        special_assignments[int(donor_id)] = int(target_id)
    target_tokenizer_rows = len(target_tokenizer)
    donor_tokenizer_rows = len(donor_tokenizer)
    if (
        isinstance(target_tokenizer_rows, bool)
        or not isinstance(target_tokenizer_rows, int)
        or target_tokenizer_rows <= 0
        or isinstance(donor_tokenizer_rows, bool)
        or not isinstance(donor_tokenizer_rows, int)
        or donor_tokenizer_rows <= 0
        or target_tokenizer_rows > profile.target_row_capacity
        or donor_tokenizer_rows > profile.donor_row_capacity
    ):
        raise Lfm2InterfaceFrontierError(
            "Tokenizer row inventories exceed their authenticated bridge capacities."
        )
    target_unk = getattr(target_tokenizer, "unk_token_id", None)
    if target_unk is not None and (
        isinstance(target_unk, bool)
        or not isinstance(target_unk, int)
        or not 0 <= target_unk < target_tokenizer_rows
    ):
        raise Lfm2InterfaceFrontierError(
            "Target unknown-token row is outside the live tokenizer."
        )
    if fallback_target_id is None:
        effective_fallback_target_id = target_unk
        fallback_policy = "target-unknown-token"
    else:
        if (
            isinstance(fallback_target_id, bool)
            or not isinstance(fallback_target_id, int)
            or not 0 <= fallback_target_id < target_tokenizer_rows
        ):
            raise Lfm2InterfaceFrontierError(
                "fallback_target_id must address one defined target-tokenizer row."
            )
        effective_fallback_target_id = int(fallback_target_id)
        fallback_policy = "explicit-authenticated-target-row"
    if effective_fallback_target_id is None:
        raise Lfm2InterfaceFrontierError(
            "Inverse output compilation requires an explicit fallback target row "
            "when the target tokenizer has no unknown token."
        )
    effective_fallback_target_id = int(effective_fallback_target_id)
    fallback_target_surface = _target_token_surface(
        target_tokenizer,
        effective_fallback_target_id,
    )

    buckets: list[list[int]] = [
        [] for _target_id in range(profile.target_row_capacity)
    ]
    empty_decomposition_count = 0
    valid_empty_decomposition_count = 0
    invalid_donor_capacity_row_count = 0
    reverse_anchor_empty_assignment_count = 0
    fallback_assignment_count = 0
    special_assignment_count = 0
    encoded_assignment_count = 0
    donor_target_decompositions: list[list[int]] = []
    prefix_transition_buckets: list[list[int]] = [
        [] for _target_id in range(profile.target_row_capacity)
    ]
    for donor_id in range(profile.donor_row_capacity):
        if donor_id >= donor_tokenizer_rows:
            target_id = effective_fallback_target_id
            decomposition = (target_id,)
            empty_decomposition_count += 1
            invalid_donor_capacity_row_count += 1
            fallback_assignment_count += 1
        elif donor_id in special_assignments:
            target_id = special_assignments[donor_id]
            decomposition = (target_id,)
            special_assignment_count += 1
        else:
            surface = _target_token_surface(donor_tokenizer, donor_id)
            encoded = _encode_without_specials(target_tokenizer, surface)
            if encoded:
                target_id = encoded[0]
                decomposition = encoded
                encoded_assignment_count += 1
            else:
                valid_empty_decomposition_count += 1
                if donor_id in reverse_anchor:
                    target_id = reverse_anchor[donor_id]
                    reverse_anchor_empty_assignment_count += 1
                else:
                    target_id = effective_fallback_target_id
                    fallback_assignment_count += 1
                decomposition = (target_id,)
                empty_decomposition_count += 1
        if not 0 <= target_id < profile.target_row_capacity:
            raise Lfm2InterfaceFrontierError(
                "Inverse output assignment left the target vocabulary."
            )
        buckets[target_id].append(donor_id)
        donor_target_decompositions.append(list(decomposition))
        for transition_target_id in decomposition:
            if not 0 <= transition_target_id < profile.target_row_capacity:
                raise Lfm2InterfaceFrontierError(
                    "Donor decomposition left the target vocabulary."
                )
            prefix_transition_buckets[transition_target_id].append(donor_id)
    flattened = [donor_id for bucket in buckets for donor_id in bucket]
    if sorted(flattened) != list(range(profile.donor_row_capacity)):
        raise Lfm2InterfaceFrontierError(
            "Inverse output plan does not cover every donor row exactly once."
        )
    empty_target_rows = [
        target_id for target_id, bucket in enumerate(buckets) if not bucket
    ]
    counts = [len(bucket) for bucket in buckets]
    decomposition_lengths = [
        len(decomposition) for decomposition in donor_target_decompositions
    ]
    body = {
        "schema": LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "assignment": "decoded-donor-surface-to-first-target-subtoken",
        "donor_row_count": profile.donor_row_capacity,
        "donor_tokenizer_defined_row_count": donor_tokenizer_rows,
        "target_tokenizer_defined_row_count": target_tokenizer_rows,
        "target_row_count": profile.target_row_capacity,
        "assigned_donor_row_count": len(flattened),
        "donor_rows_assigned_exactly_once": True,
        "special_assignment_count": special_assignment_count,
        "encoded_assignment_count": encoded_assignment_count,
        "empty_decomposition_count": empty_decomposition_count,
        "valid_empty_decomposition_count": valid_empty_decomposition_count,
        "invalid_donor_capacity_row_count": invalid_donor_capacity_row_count,
        "reverse_anchor_empty_assignment_count": (
            reverse_anchor_empty_assignment_count
        ),
        "fallback_assignment_count": fallback_assignment_count,
        "fallback_policy": fallback_policy,
        "fallback_target_id": effective_fallback_target_id,
        "fallback_target_surface": fallback_target_surface,
        "target_unknown_token_id": target_unk,
        "empty_target_row_count": len(empty_target_rows),
        "empty_target_rows": empty_target_rows,
        "minimum_bucket_size": min(counts),
        "maximum_bucket_size": max(counts),
        "buckets": buckets,
        "donor_target_decompositions": donor_target_decompositions,
        "prefix_transition_buckets": prefix_transition_buckets,
        "multi_subtoken_donor_count": sum(
            length > 1 for length in decomposition_lengths
        ),
        "maximum_decomposition_length": max(decomposition_lengths),
        "exact_autoregressive_tokenizer_bridge": False,
        "first_subtoken_mass_is_partial_bridge_only": True,
        "cached_child_features_after_inserted_target_subtokens_available": False,
        "exact_prefix_conditionals_compilable_from_this_plan": False,
        "prefix_transition_buckets_are_structural_proxies": True,
        "parent_forward_calls": 0,
        "training_steps": 0,
        "network_access": False,
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def _verify_precompiled_plan(
    plan: Mapping[str, Any],
    *,
    schema: str,
    bridge: StrictTokenizerBridgeArtifact,
) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise Lfm2InterfaceFrontierError("Precompiled plan must be a mapping.")
    value = dict(plan)
    body = dict(value)
    supplied = body.pop("plan_sha256", None)
    if (
        supplied != canonical_sha256(body)
        or value.get("schema") != schema
        or value.get("bridge_artifact_sha256") != bridge.artifact_sha256
        or value.get("bridge_profile_sha256") != bridge.profile_fingerprint
        or value.get("donor_tokenizer_semantics_sha256")
        != bridge.donor_tokenizer_semantics_sha256
        or value.get("target_tokenizer_semantics_sha256")
        != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2InterfaceFrontierError(
            "Precompiled interface plan differs from the live bridge."
        )
    return value


def _compile_context_table(
    source_rows: Any,
    plan: Mapping[str, Any],
    *,
    strict_anchor_ppm: int,
    preserve_anchor_norm: bool,
) -> Any:
    torch = _torch()
    if (
        not isinstance(source_rows, torch.Tensor)
        or source_rows.ndim != 2
        or not torch.is_floating_point(source_rows)
        or torch.is_complex(source_rows)
        or source_rows.layout != torch.strided
    ):
        raise Lfm2InterfaceFrontierError("Source interface must be a dense matrix.")
    records = plan.get("records")
    if not isinstance(records, list) or not records:
        raise Lfm2InterfaceFrontierError("Sequence plan has no target rows.")
    alpha = strict_anchor_ppm / 1_000_000.0
    source = source_rows.detach().to(device="cpu", dtype=torch.float64)
    compiled = []
    with torch.no_grad():
        for expected_target_id, record in enumerate(records):
            if (
                not isinstance(record, Mapping)
                or record.get("target_id") != expected_target_id
                or not isinstance(record.get("strict_donor_id"), int)
                or not isinstance(record.get("donor_sequences"), list)
            ):
                raise Lfm2InterfaceFrontierError(
                    "Sequence plan row ordering is invalid."
                )
            strict_id = int(record["strict_donor_id"])
            sequence_vectors = []
            for raw_sequence in record["donor_sequences"]:
                if not isinstance(raw_sequence, list) or not raw_sequence:
                    raise Lfm2InterfaceFrontierError(
                        "Sequence plan contains an empty donor sequence."
                    )
                indices = torch.tensor(raw_sequence, dtype=torch.long)
                sequence_vectors.append(source.index_select(0, indices).mean(dim=0))
            envelope = torch.stack(sequence_vectors, dim=0).mean(dim=0)
            anchor = source[strict_id]
            value = alpha * anchor + (1.0 - alpha) * envelope
            if preserve_anchor_norm:
                anchor_norm = torch.linalg.vector_norm(anchor)
                value_norm = torch.linalg.vector_norm(value)
                if float(value_norm) > 0.0:
                    value = value * (anchor_norm / value_norm)
                elif float(anchor_norm) > 0.0:
                    value = anchor
            if not bool(torch.isfinite(value).all()):
                raise Lfm2InterfaceFrontierError(
                    "Compiled interface row contains NaN or infinity."
                )
            compiled.append(value)
    result = torch.stack(compiled, dim=0).to(dtype=source_rows.dtype).contiguous()
    return result


def compile_lfm2_inverse_bucket_output_table(
    source_rows: Any,
    inverse_plan: Mapping[str, Any],
    strict_mapping: Sequence[int],
    *,
    strict_anchor_ppm: int,
    preserve_anchor_norm: bool,
    weighting: str,
) -> Any:
    torch = _torch()
    if (
        not isinstance(source_rows, torch.Tensor)
        or source_rows.ndim != 2
        or not torch.is_floating_point(source_rows)
        or torch.is_complex(source_rows)
        or source_rows.layout != torch.strided
    ):
        raise Lfm2InterfaceFrontierError("Source output head must be dense.")
    buckets = inverse_plan.get("buckets")
    if (
        not isinstance(buckets, list)
        or len(buckets) != len(strict_mapping)
        or weighting not in {"uniform", "row-norm"}
    ):
        raise Lfm2InterfaceFrontierError(
            "Inverse output buckets or weighting are invalid."
        )
    alpha = strict_anchor_ppm / 1_000_000.0
    source = source_rows.detach().to(device="cpu", dtype=torch.float64)
    compiled = []
    with torch.no_grad():
        for target_id, raw_bucket in enumerate(buckets):
            if not isinstance(raw_bucket, list) or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < int(source.shape[0])
                for value in raw_bucket
            ):
                raise Lfm2InterfaceFrontierError(
                    "Inverse output bucket contains an invalid donor row."
                )
            anchor = source[int(strict_mapping[target_id])]
            if raw_bucket:
                rows = source.index_select(
                    0,
                    torch.tensor(raw_bucket, dtype=torch.long),
                )
                if weighting == "row-norm":
                    weights = torch.linalg.vector_norm(rows, dim=1)
                    denominator = weights.sum()
                    centroid = (
                        rows.mean(dim=0)
                        if float(denominator) <= 0.0
                        else (rows * weights[:, None]).sum(dim=0) / denominator
                    )
                else:
                    centroid = rows.mean(dim=0)
            else:
                centroid = anchor
            value = alpha * anchor + (1.0 - alpha) * centroid
            if preserve_anchor_norm:
                anchor_norm = torch.linalg.vector_norm(anchor)
                value_norm = torch.linalg.vector_norm(value)
                if float(value_norm) > 0.0:
                    value = value * (anchor_norm / value_norm)
                elif float(anchor_norm) > 0.0:
                    value = anchor
            if not bool(torch.isfinite(value).all()):
                raise Lfm2InterfaceFrontierError(
                    "Inverse output row contains NaN or infinity."
                )
            compiled.append(value)
    return torch.stack(compiled, dim=0).to(dtype=source_rows.dtype).contiguous()


def compile_lfm2_prefix_transition_output_table(
    source_rows: Any,
    inverse_plan: Mapping[str, Any],
    strict_mapping: Sequence[int],
    *,
    strict_anchor_ppm: int,
    preserve_anchor_norm: bool,
) -> Any:
    """Compile a later-subtoken-aware structural proxy over decompositions.

    A donor row contributes total weight one across every target subtoken in
    its decomposition.  This retains later pieces that first-subtoken buckets
    discard, but is deliberately not claimed as an exact prefix-conditional
    autoregressive bridge because it has no per-prefix child feature row.
    """

    torch = _torch()
    decompositions = inverse_plan.get("donor_target_decompositions")
    if (
        not isinstance(source_rows, torch.Tensor)
        or source_rows.ndim != 2
        or not torch.is_floating_point(source_rows)
        or torch.is_complex(source_rows)
        or source_rows.layout != torch.strided
        or not isinstance(decompositions, list)
        or len(decompositions) != int(source_rows.shape[0])
        or len(strict_mapping) <= 0
    ):
        raise Lfm2InterfaceFrontierError(
            "Prefix-transition source rows or decomposition plan are invalid."
        )
    if (
        isinstance(strict_anchor_ppm, bool)
        or not isinstance(strict_anchor_ppm, int)
        or not 0 <= strict_anchor_ppm <= 1_000_000
    ):
        raise Lfm2InterfaceFrontierError(
            "strict_anchor_ppm must be between zero and one million."
        )
    target_count = len(strict_mapping)
    weighted_rows: list[list[tuple[int, float]]] = [
        [] for _target_id in range(target_count)
    ]
    for donor_id, decomposition in enumerate(decompositions):
        if not isinstance(decomposition, list) or not decomposition or any(
            isinstance(target_id, bool)
            or not isinstance(target_id, int)
            or not 0 <= target_id < target_count
            for target_id in decomposition
        ):
            raise Lfm2InterfaceFrontierError(
                "Prefix-transition decomposition contains an invalid target row."
            )
        weight = 1.0 / len(decomposition)
        for target_id in decomposition:
            weighted_rows[target_id].append((donor_id, weight))
    source = source_rows.detach().to(device="cpu", dtype=torch.float64)
    alpha = strict_anchor_ppm / 1_000_000.0
    compiled = []
    with torch.no_grad():
        for target_id, entries in enumerate(weighted_rows):
            anchor = source[int(strict_mapping[target_id])]
            if entries:
                indices = torch.tensor(
                    [donor_id for donor_id, _weight in entries],
                    dtype=torch.long,
                )
                weights = torch.tensor(
                    [weight for _donor_id, weight in entries],
                    dtype=torch.float64,
                )
                rows = source.index_select(0, indices)
                proxy = (rows * weights[:, None]).sum(dim=0) / weights.sum()
            else:
                proxy = anchor
            value = alpha * anchor + (1.0 - alpha) * proxy
            if preserve_anchor_norm:
                anchor_norm = torch.linalg.vector_norm(anchor)
                value_norm = torch.linalg.vector_norm(value)
                if float(value_norm) > 0.0:
                    value = value * (anchor_norm / value_norm)
                elif float(anchor_norm) > 0.0:
                    value = anchor
            if not bool(torch.isfinite(value).all()):
                raise Lfm2InterfaceFrontierError(
                    "Prefix-transition structural row contains NaN or infinity."
                )
            compiled.append(value)
    return torch.stack(compiled, dim=0).to(dtype=source_rows.dtype).contiguous()


def _relative_centered_error(actual: Any, target: Any) -> float:
    torch = _torch()
    residual = actual - target
    denominator = torch.linalg.vector_norm(target).clamp_min(1e-12)
    return float((torch.linalg.vector_norm(residual) / denominator).detach().cpu())


def _aggregate_donor_logit_mass(
    donor_logits: Any,
    donor_to_target: Any,
    *,
    target_row_count: int,
    maximum_rows_per_chunk: int,
) -> Any:
    """Compute exact logsumexp mass for a total donor-to-target assignment."""

    torch = _torch()
    row_count, donor_row_count = (
        int(donor_logits.shape[0]),
        int(donor_logits.shape[1]),
    )
    if tuple(donor_to_target.shape) != (donor_row_count,):
        raise Lfm2InterfaceFrontierError(
            "Donor-to-target assignment does not cover the logits vocabulary."
        )
    chunks = []
    for start in range(0, row_count, maximum_rows_per_chunk):
        values = donor_logits[start : start + maximum_rows_per_chunk]
        rows = int(values.shape[0])
        indices = donor_to_target.view(1, -1).expand(rows, -1)
        maxima = torch.full(
            (rows, target_row_count),
            -torch.inf,
            dtype=values.dtype,
            device=values.device,
        )
        maxima.scatter_reduce_(
            1,
            indices,
            values,
            reduce="amax",
            include_self=True,
        )
        gathered_maxima = maxima.gather(1, indices)
        contributions = torch.exp(values - gathered_maxima)
        sums = torch.zeros_like(maxima)
        sums.scatter_add_(1, indices, contributions)
        chunks.append(maxima + torch.log(sums))
    return torch.cat(chunks, dim=0)


def compile_lfm2_cached_logit_residual_head(
    structural_output_table: Any,
    hidden_rows: Any,
    donor_logits: Any,
    inverse_output_plan: Mapping[str, Any],
    *,
    source_evidence_sha256: str,
    ridge: float,
    residual_scale_ppm: int,
    protected_target_ids: Sequence[int] = (),
    row_norm_calibration_ppm: int = 0,
    solve_device: Any = "cpu",
    maximum_rows_per_aggregation_chunk: int = 32,
) -> Lfm2CachedLogitResidualHead:
    """Compile a dense target head by one closed-form cache-only solve.

    Every donor token contributes to exactly one first-target-subtoken bucket.
    The sealed parent logits are reduced by exact log-sum-exp mass, then a
    dual ridge residual around the structural output table is solved without
    autograd.  This is analytic parameter transport, not iterative fitting.
    """

    torch = _torch()
    if (
        not isinstance(source_evidence_sha256, str)
        or len(source_evidence_sha256) != 64
        or source_evidence_sha256 != source_evidence_sha256.lower()
        or any(value not in "0123456789abcdef" for value in source_evidence_sha256)
    ):
        raise Lfm2InterfaceFrontierError(
            "source_evidence_sha256 must be a lowercase SHA-256 digest."
        )
    for value, label in (
        (structural_output_table, "structural output table"),
        (hidden_rows, "hidden rows"),
        (donor_logits, "donor logits"),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or not torch.is_floating_point(value)
            or torch.is_complex(value)
            or value.layout != torch.strided
            or not bool(torch.isfinite(value).all())
        ):
            raise Lfm2InterfaceFrontierError(
                f"Cached {label} must be one finite dense matrix."
            )
    if (
        int(hidden_rows.shape[0]) != int(donor_logits.shape[0])
        or int(hidden_rows.shape[0]) <= 0
        or int(hidden_rows.shape[1]) != int(structural_output_table.shape[1])
    ):
        raise Lfm2InterfaceFrontierError(
            "Cached terminal hidden/logit rows do not align with the head."
        )
    if (
        isinstance(ridge, bool)
        or not isinstance(ridge, (int, float))
        or not float(ridge) > 0.0
        or not float(ridge) < float("inf")
    ):
        raise Lfm2InterfaceFrontierError("ridge must be finite and positive.")
    for value, label in (
        (residual_scale_ppm, "residual_scale_ppm"),
        (row_norm_calibration_ppm, "row_norm_calibration_ppm"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 1_000_000
        ):
            raise Lfm2InterfaceFrontierError(
                f"{label} must be an integer from zero through one million."
            )
    if (
        isinstance(maximum_rows_per_aggregation_chunk, bool)
        or not isinstance(maximum_rows_per_aggregation_chunk, int)
        or maximum_rows_per_aggregation_chunk <= 0
    ):
        raise Lfm2InterfaceFrontierError(
            "maximum_rows_per_aggregation_chunk must be positive."
        )
    plan_body = dict(inverse_output_plan)
    supplied_plan_sha256 = plan_body.pop("plan_sha256", None)
    buckets = inverse_output_plan.get("buckets")
    target_row_count = int(structural_output_table.shape[0])
    donor_row_count = int(donor_logits.shape[1])
    if (
        supplied_plan_sha256 != canonical_sha256(plan_body)
        or inverse_output_plan.get("schema")
        != LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA
        or inverse_output_plan.get("donor_row_count") != donor_row_count
        or inverse_output_plan.get("target_row_count") != target_row_count
        or inverse_output_plan.get("donor_rows_assigned_exactly_once") is not True
        or not isinstance(buckets, list)
        or len(buckets) != target_row_count
    ):
        raise Lfm2InterfaceFrontierError(
            "Inverse output plan is unsealed or incompatible with cached logits."
        )
    donor_to_target = torch.full((donor_row_count,), -1, dtype=torch.long)
    seen: list[int] = []
    empty_target_ids: list[int] = []
    for target_id, bucket in enumerate(buckets):
        if not isinstance(bucket, list) or any(
            isinstance(donor_id, bool)
            or not isinstance(donor_id, int)
            or not 0 <= donor_id < donor_row_count
            for donor_id in bucket
        ):
            raise Lfm2InterfaceFrontierError(
                "Inverse output plan contains an invalid donor bucket."
            )
        if not bucket:
            empty_target_ids.append(target_id)
            continue
        seen.extend(bucket)
        donor_to_target[torch.tensor(bucket, dtype=torch.long)] = target_id
    if sorted(seen) != list(range(donor_row_count)) or bool(
        (donor_to_target < 0).any()
    ):
        raise Lfm2InterfaceFrontierError(
            "Inverse output plan does not assign every donor row exactly once."
        )
    protected = tuple(sorted(set(protected_target_ids)))
    if any(
        isinstance(target_id, bool)
        or not isinstance(target_id, int)
        or not 0 <= target_id < target_row_count
        for target_id in protected
    ):
        raise Lfm2InterfaceFrontierError(
            "protected_target_ids contain a row outside the target head."
        )

    requested_device = torch.device(solve_device)
    if requested_device.type not in {"cpu", "cuda"} or (
        requested_device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise Lfm2InterfaceFrontierError(
            "solve_device must be an available CPU or CUDA device."
        )
    structural_sha256 = _tensor_sha256(structural_output_table)
    hidden_sha256 = _tensor_sha256(hidden_rows)
    donor_logits_sha256 = _tensor_sha256(donor_logits)
    scale = residual_scale_ppm / 1_000_000.0
    norm_scale = row_norm_calibration_ppm / 1_000_000.0
    with torch.no_grad():
        hidden = hidden_rows.detach().to(
            device=requested_device,
            dtype=torch.float32,
        )
        logits = donor_logits.detach().to(
            device=requested_device,
            dtype=torch.float32,
        )
        structural = structural_output_table.detach().to(
            device=requested_device,
            dtype=torch.float32,
        )
        assignment = donor_to_target.to(device=requested_device)
        target_logits = _aggregate_donor_logit_mass(
            logits,
            assignment,
            target_row_count=target_row_count,
            maximum_rows_per_chunk=maximum_rows_per_aggregation_chunk,
        )
        structural_prediction = hidden.matmul(structural.t())
        if empty_target_ids:
            empty_index = torch.tensor(
                empty_target_ids,
                dtype=torch.long,
                device=requested_device,
            )
            target_logits.index_copy_(
                1,
                empty_index,
                structural_prediction.index_select(1, empty_index),
            )
        if not bool(torch.isfinite(target_logits).all()):
            raise Lfm2InterfaceFrontierError(
                "Aggregated target logit mass contains NaN or infinity."
            )
        centered_target = target_logits - target_logits.mean(dim=1, keepdim=True)
        residual = target_logits - structural_prediction
        residual = residual - residual.mean(dim=1, keepdim=True)
        gram = hidden.matmul(hidden.t())
        gram = gram + float(ridge) * torch.eye(
            int(gram.shape[0]),
            dtype=gram.dtype,
            device=gram.device,
        )
        dual = torch.linalg.solve(gram, hidden)
        delta = residual.t().matmul(dual)
        compiled = structural + scale * delta
        if protected:
            protected_index = torch.tensor(
                protected,
                dtype=torch.long,
                device=requested_device,
            )
            compiled.index_copy_(
                0,
                protected_index,
                structural.index_select(0, protected_index),
            )
        if norm_scale > 0.0:
            structural_norm = torch.linalg.vector_norm(structural, dim=1)
            compiled_norm = torch.linalg.vector_norm(compiled, dim=1)
            calibrated_norm = (
                (1.0 - norm_scale) * compiled_norm
                + norm_scale * structural_norm
            )
            safe_scale = torch.where(
                compiled_norm > 0.0,
                calibrated_norm / compiled_norm.clamp_min(1e-12),
                torch.ones_like(compiled_norm),
            )
            compiled = compiled * safe_scale[:, None]
            zero_rows = (compiled_norm <= 0.0) & (structural_norm > 0.0)
            if bool(zero_rows.any()):
                compiled[zero_rows] = structural[zero_rows]
        if protected:
            compiled.index_copy_(
                0,
                protected_index,
                structural.index_select(0, protected_index),
            )
        final_prediction = hidden.matmul(compiled.t())
        centered_structural_prediction = (
            structural_prediction
            - structural_prediction.mean(dim=1, keepdim=True)
        )
        centered_final_prediction = (
            final_prediction - final_prediction.mean(dim=1, keepdim=True)
        )
        before_error = _relative_centered_error(
            centered_structural_prediction,
            centered_target,
        )
        after_error = _relative_centered_error(
            centered_final_prediction,
            centered_target,
        )
        result_table = compiled.to(
            device="cpu",
            dtype=structural_output_table.dtype,
        ).contiguous()
    if not bool(torch.isfinite(result_table).all()):
        raise Lfm2InterfaceFrontierError(
            "Closed-form residual solve produced NaN or infinity."
        )
    output_sha256 = _tensor_sha256(result_table)
    body = {
        "schema": LFM2_CACHED_LOGIT_RESIDUAL_HEAD_SCHEMA,
        "status": "dense-cache-only-dual-ridge-residual-head-compiled",
        "source_evidence_sha256": source_evidence_sha256,
        "inverse_output_plan_sha256": supplied_plan_sha256,
        "assignment": "full-donor-logit-mass-to-first-target-subtoken",
        "aggregation": "exact-bucket-logsumexp",
        "structural_output_table_sha256": structural_sha256,
        "hidden_rows_sha256": hidden_sha256,
        "donor_logits_sha256": donor_logits_sha256,
        "output_table_sha256": output_sha256,
        "evidence_row_count": int(hidden_rows.shape[0]),
        "hidden_size": int(hidden_rows.shape[1]),
        "donor_row_count": donor_row_count,
        "target_row_count": target_row_count,
        "empty_target_row_count": len(empty_target_ids),
        "empty_target_rows_use_structural_prediction": True,
        "ridge": float(ridge),
        "residual_scale_ppm": residual_scale_ppm,
        "row_norm_calibration_ppm": row_norm_calibration_ppm,
        "protected_target_ids": list(protected),
        "relative_centered_error_before": before_error,
        "relative_centered_error_after": after_error,
        "diagnostic_metrics_are_not_acceptance_gates": True,
        "solve_device": str(requested_device),
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "network_access": False,
        },
    }
    return Lfm2CachedLogitResidualHead(
        table=result_table,
        receipt={**body, "result_sha256": canonical_sha256(body)},
    )


def _untie_zero_blank(model: Any) -> dict[str, Any]:
    torch = _torch()
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if input_weight is not output_weight or not bool(
        getattr(model.config, "tie_word_embeddings", False)
    ):
        raise Lfm2InterfaceFrontierError(
            "Untied output conversion requires a native tied zero blank."
        )
    if int(torch.count_nonzero(input_weight.detach()).item()) != 0:
        raise Lfm2InterfaceFrontierError("Blank input/output table is not zero.")
    before = fingerprint_model_state(model)
    vocabulary, hidden = (int(value) for value in input_weight.shape)
    head = torch.nn.Linear(
        hidden,
        vocabulary,
        bias=False,
        device=input_weight.device,
        dtype=input_weight.dtype,
    )
    with torch.no_grad():
        head.weight.zero_()
    model.config.tie_word_embeddings = False
    setter = getattr(model, "set_output_embeddings", None)
    if not callable(setter):
        raise Lfm2InterfaceFrontierError(
            "LFM2 model cannot install an independent output operator."
        )
    setter(head)
    installed_output = model.get_output_embeddings().weight
    if (
        installed_output is input_weight
        or _storage_identity(installed_output) == _storage_identity(input_weight)
        or int(torch.count_nonzero(installed_output.detach()).item()) != 0
        or bool(getattr(model.config, "tie_word_embeddings", True))
    ):
        raise Lfm2InterfaceFrontierError(
            "Compiler-owned untied output blank was not materialized."
        )
    return {
        "operation": "compiler-owned-zero-output-untie",
        "base_blank_state_sha256": before,
        "untied_blank_state_sha256": fingerprint_model_state(model),
        "added_dense_parameters": vocabulary * hidden,
        "input_output_storage_distinct": True,
        "all_added_parameters_zero": True,
    }


def fold_lfm2_interface_candidate(
    parent_model: Any,
    blank_target_model: Any,
    *,
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    blank_target_receipt: Mapping[str, Any],
    recipe: Lfm2InterfaceRecipe,
    maximum_target_unique_parameters: int,
    contexts: Sequence[str] = DEFAULT_INTERFACE_CONTEXTS,
    cached_logit_residual_head: Lfm2CachedLogitResidualHead | None = None,
    precompiled_sequence_plan: Mapping[str, Any] | None = None,
    precompiled_inverse_output_plan: Mapping[str, Any] | None = None,
    inverse_output_fallback_target_id: int | None = None,
) -> Lfm2InterfaceFrontierResult:
    """Install one exact-body compositional/asymmetric interface candidate."""

    torch = _torch()
    if parent_model is blank_target_model:
        raise Lfm2InterfaceFrontierError(
            "Parent and blank child must be distinct model objects."
        )
    if not isinstance(recipe, Lfm2InterfaceRecipe):
        raise TypeError("recipe must be Lfm2InterfaceRecipe.")
    if (recipe.output_mode == "cached-logit-residual") != isinstance(
        cached_logit_residual_head,
        Lfm2CachedLogitResidualHead,
    ):
        raise Lfm2InterfaceFrontierError(
            "Cached-logit output mode requires exactly one sealed residual head."
        )
    if (
        isinstance(maximum_target_unique_parameters, bool)
        or not isinstance(maximum_target_unique_parameters, int)
        or maximum_target_unique_parameters <= 0
    ):
        raise Lfm2InterfaceFrontierError(
            "maximum_target_unique_parameters must be positive."
        )
    base_blank_sha = _verify_base_blank(blank_target_receipt)
    bridge.verify_live_tokenizers(donor_tokenizer, target_tokenizer)
    parent_topology = _config_topology(parent_model.config)
    initial_target_topology = _config_topology(blank_target_model.config)
    if (
        parent_topology["model_type"] != "lfm2"
        or initial_target_topology["model_type"] != "lfm2"
        or _body_topology(parent_topology) != _body_topology(initial_target_topology)
        or parent_topology["tie_word_embeddings"] is not True
        or initial_target_topology["tie_word_embeddings"] is not True
    ):
        raise Lfm2InterfaceFrontierError(
            "Interface frontier requires an exact-body native-tied source blank."
        )
    if any(
        value.layout != torch.strided
        or int(torch.count_nonzero(value.detach()).item()) != 0
        for value in blank_target_model.parameters()
    ):
        raise Lfm2InterfaceFrontierError(
            "Compiler-authenticated blank contains nonzero or non-dense parameters."
        )

    base_blank_state = fingerprint_model_state(blank_target_model)
    untie_receipt = None
    if recipe.untied_output:
        untie_receipt = _untie_zero_blank(blank_target_model)
    target_topology = _config_topology(blank_target_model.config)
    if target_topology["tie_word_embeddings"] == recipe.untied_output:
        raise Lfm2InterfaceFrontierError(
            "Candidate output tie differs from its requested recipe."
        )

    profile = bridge.transport_profile
    parent_input = parent_model.get_input_embeddings().weight
    parent_output = parent_model.get_output_embeddings().weight
    target_input = blank_target_model.get_input_embeddings().weight
    target_output = blank_target_model.get_output_embeddings().weight
    if (
        int(parent_input.shape[0]) != profile.donor_row_capacity
        or int(parent_output.shape[0]) != profile.donor_row_capacity
        or int(target_input.shape[0]) != profile.target_row_capacity
        or int(target_output.shape[0]) != profile.target_row_capacity
        or int(parent_input.shape[1]) != int(target_input.shape[1])
        or int(parent_output.shape[1]) != int(target_output.shape[1])
    ):
        raise Lfm2InterfaceFrontierError(
            "Bridge capacities do not address the concrete interface matrices."
        )

    parent_parameters = _unique_parameter_count(parent_model)
    target_parameters = _unique_parameter_count(blank_target_model)
    hidden = int(target_input.shape[1])
    body_parameters = parent_parameters - profile.donor_row_capacity * hidden
    expected_base = body_parameters + profile.target_row_capacity * hidden
    expected_target = expected_base + (
        profile.target_row_capacity * hidden if recipe.untied_output else 0
    )
    if (
        blank_target_receipt.get("parent_unique_parameters") != parent_parameters
        or blank_target_receipt.get("target_unique_parameters") != expected_base
        or target_parameters != expected_target
        or target_parameters > maximum_target_unique_parameters
        or target_parameters >= parent_parameters
    ):
        raise Lfm2InterfaceFrontierError(
            "Candidate does not satisfy its authenticated target-size envelope "
            f"(receipt_parent={blank_target_receipt.get('parent_unique_parameters')}, "
            f"parent={parent_parameters}, "
            f"receipt_base={blank_target_receipt.get('target_unique_parameters')}, "
            f"expected_base={expected_base}, observed={target_parameters}, "
            f"expected={expected_target}, maximum={maximum_target_unique_parameters})."
        )

    parent_interface_ids = _interface_storage_ids(parent_model)
    target_interface_ids = _interface_storage_ids(blank_target_model)
    parent_named = dict(_all_named_tensors(parent_model))
    target_named = dict(_all_named_tensors(blank_target_model))
    parent_body_names = {
        name
        for name, value in parent_named.items()
        if _storage_identity(value) not in parent_interface_ids
    }
    target_body_names = {
        name
        for name, value in target_named.items()
        if _storage_identity(value) not in target_interface_ids
    }
    if parent_body_names != target_body_names:
        raise Lfm2InterfaceFrontierError(
            "Parent and target non-interface inventories differ."
        )
    for name in sorted(parent_body_names):
        if tuple(parent_named[name].shape) != tuple(target_named[name].shape):
            raise Lfm2InterfaceFrontierError(
                f"Body tensor shape differs for {name!r}."
            )

    plan = (
        compile_lfm2_interface_sequence_plan(
            bridge,
            donor_tokenizer,
            target_tokenizer,
            contexts=contexts,
        )
        if precompiled_sequence_plan is None
        else _verify_precompiled_plan(
            precompiled_sequence_plan,
            schema=LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
            bridge=bridge,
        )
    )
    input_table = _compile_context_table(
        parent_input,
        plan,
        strict_anchor_ppm=recipe.input_strict_anchor_ppm,
        preserve_anchor_norm=recipe.input_preserve_anchor_norm,
    ).to(device=target_input.device, dtype=target_input.dtype)
    mapping = tuple(int(value) for value in profile.target_to_donor_row_indices)
    inverse_plan = None
    if recipe.output_mode == "tied-to-input":
        output_table = input_table
    elif recipe.output_mode == "strict-row":
        output_table = parent_output.detach().index_select(
            0,
            torch.tensor(mapping, dtype=torch.long, device=parent_output.device),
        ).to(device=target_output.device, dtype=target_output.dtype)
    elif recipe.output_mode == "context-envelope":
        output_table = _compile_context_table(
            parent_output,
            plan,
            strict_anchor_ppm=recipe.output_strict_anchor_ppm,
            preserve_anchor_norm=recipe.output_preserve_anchor_norm,
        ).to(device=target_output.device, dtype=target_output.dtype)
    elif recipe.output_mode == "inverse-bucket-centroid":
        inverse_plan = (
            compile_lfm2_inverse_output_plan(
                bridge,
                donor_tokenizer,
                target_tokenizer,
                fallback_target_id=inverse_output_fallback_target_id,
            )
            if precompiled_inverse_output_plan is None
            else _verify_precompiled_plan(
                precompiled_inverse_output_plan,
                schema=LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA,
                bridge=bridge,
            )
        )
        output_table = compile_lfm2_inverse_bucket_output_table(
            parent_output,
            inverse_plan,
            mapping,
            strict_anchor_ppm=recipe.output_strict_anchor_ppm,
            preserve_anchor_norm=recipe.output_preserve_anchor_norm,
            weighting=recipe.output_bucket_weighting,
        ).to(device=target_output.device, dtype=target_output.dtype)
    elif recipe.output_mode == "prefix-transition-centroid":
        inverse_plan = (
            compile_lfm2_inverse_output_plan(
                bridge,
                donor_tokenizer,
                target_tokenizer,
                fallback_target_id=inverse_output_fallback_target_id,
            )
            if precompiled_inverse_output_plan is None
            else _verify_precompiled_plan(
                precompiled_inverse_output_plan,
                schema=LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA,
                bridge=bridge,
            )
        )
        output_table = compile_lfm2_prefix_transition_output_table(
            parent_output,
            inverse_plan,
            mapping,
            strict_anchor_ppm=recipe.output_strict_anchor_ppm,
            preserve_anchor_norm=recipe.output_preserve_anchor_norm,
        ).to(device=target_output.device, dtype=target_output.dtype)
    else:
        assert cached_logit_residual_head is not None
        inverse_plan = (
            compile_lfm2_inverse_output_plan(
                bridge,
                donor_tokenizer,
                target_tokenizer,
                fallback_target_id=inverse_output_fallback_target_id,
            )
            if precompiled_inverse_output_plan is None
            else _verify_precompiled_plan(
                precompiled_inverse_output_plan,
                schema=LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA,
                bridge=bridge,
            )
        )
        structural_output_table = compile_lfm2_inverse_bucket_output_table(
            parent_output,
            inverse_plan,
            mapping,
            strict_anchor_ppm=recipe.output_strict_anchor_ppm,
            preserve_anchor_norm=recipe.output_preserve_anchor_norm,
            weighting=recipe.output_bucket_weighting,
        )
        residual_receipt = cached_logit_residual_head.as_dict()
        if (
            residual_receipt.get("inverse_output_plan_sha256")
            != inverse_plan["plan_sha256"]
            or residual_receipt.get("structural_output_table_sha256")
            != _tensor_sha256(structural_output_table)
            or tuple(cached_logit_residual_head.table.shape)
            != tuple(target_output.shape)
        ):
            raise Lfm2InterfaceFrontierError(
                "Cached-logit residual head differs from this bridge or source."
            )
        output_table = cached_logit_residual_head.table.to(
            device=target_output.device,
            dtype=target_output.dtype,
        )
    if (
        inverse_plan is not None
        and inverse_output_fallback_target_id is not None
        and inverse_plan.get("fallback_target_id")
        != inverse_output_fallback_target_id
    ):
        raise Lfm2InterfaceFrontierError(
            "Precompiled inverse-output fallback differs from the request."
        )
    if tuple(input_table.shape) != tuple(target_input.shape) or tuple(
        output_table.shape
    ) != tuple(target_output.shape):
        raise Lfm2InterfaceFrontierError(
            "Compiled interface table shape differs from the target."
        )

    parent_state_before = fingerprint_model_state(parent_model)
    input_sha256 = _tensor_sha256(input_table)
    output_sha256 = _tensor_sha256(output_table)
    with torch.no_grad():
        for name in sorted(target_body_names):
            target_named[name].copy_(
                parent_named[name].detach().to(
                    device=target_named[name].device,
                    dtype=target_named[name].dtype,
                )
            )
        target_input.copy_(input_table)
        if recipe.untied_output:
            target_output.copy_(output_table)
    parent_state_after = fingerprint_model_state(parent_model)
    installed_state = fingerprint_model_state(blank_target_model)
    source_body = _body_records(
        parent_model,
        interface_storage_ids=parent_interface_ids,
    )
    installed_body = _body_records(
        blank_target_model,
        interface_storage_ids=target_interface_ids,
    )
    if (
        source_body != installed_body
        or parent_state_after != parent_state_before
        or installed_state == base_blank_state
        or _tensor_sha256(target_input) != input_sha256
        or _tensor_sha256(target_output) != output_sha256
        or (target_input is target_output) == recipe.untied_output
        or _unique_parameter_count(blank_target_model) != target_parameters
    ):
        raise Lfm2InterfaceFrontierError(
            "Installed interface candidate failed structural replay."
        )

    recipe_document = recipe.as_dict()
    plan_summary = {
        key: value
        for key, value in plan.items()
        if key != "records"
    }
    inverse_plan_summary = (
        None
        if inverse_plan is None
        else {
            key: value
            for key, value in inverse_plan.items()
            if key
            not in {
                "buckets",
                "donor_target_decompositions",
                "prefix_transition_buckets",
            }
        }
    )
    body = {
        "schema": LFM2_INTERFACE_FRONTIER_SCHEMA,
        "status": "standalone-exact-body-interface-candidate-installed",
        "candidate_id": recipe.candidate_id,
        "recipe": recipe_document,
        "recipe_sha256": canonical_sha256(recipe_document),
        "diagnostic_role": (
            "automatic-context-compositional-and-asymmetric-interface-frontier"
        ),
        "interface_isolation_control": True,
        "final_four_x_topology": False,
        "reason_not_final_four_x_topology": (
            "exact parent body is retained to isolate the token interface"
        ),
        "parent_topology": parent_topology,
        "target_topology": target_topology,
        "exact_parent_body_depth_width_preserved": True,
        "body_transport": "exact-name-and-shape-identity-transport",
        "body_tensor_records": list(installed_body),
        "body_tensor_inventory_sha256": canonical_sha256(list(installed_body)),
        "body_tensor_count": len(installed_body),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "sequence_plan": plan_summary,
        "sequence_plan_sha256": plan["plan_sha256"],
        "inverse_output_plan": inverse_plan_summary,
        "inverse_output_plan_sha256": (
            None if inverse_plan is None else inverse_plan["plan_sha256"]
        ),
        "cached_logit_residual_head": (
            None
            if cached_logit_residual_head is None
            else cached_logit_residual_head.as_dict()
        ),
        "exact_autoregressive_tokenizer_bridge_claimed": False,
        "prefix_conditional_child_feature_rows_used": False,
        "input_table_sha256": input_sha256,
        "output_table_sha256": output_sha256,
        "native_input_output_tying_preserved": not recipe.untied_output,
        "compiler_owned_untied_blank": untie_receipt,
        "parent_unique_parameters": parent_parameters,
        "base_tied_target_unique_parameters": expected_base,
        "target_unique_parameters": target_parameters,
        "maximum_target_unique_parameters": maximum_target_unique_parameters,
        "target_size_envelope_satisfied": True,
        "realized_parameter_compaction_ratio": parent_parameters / target_parameters,
        "base_blank_report_sha256": base_blank_sha,
        "base_blank_model_state_sha256": base_blank_state,
        "installed_model_state_sha256": installed_state,
        "parent_model_state_before_sha256": parent_state_before,
        "parent_model_state_after_sha256": parent_state_after,
        "parent_state_preserved": True,
        "standalone_save_reload_pending": True,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "child_forward_calls": 0,
            "child_generate_calls": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "external_adapter_parameters": 0,
            "parent_sidecar_parameters": 0,
            "runtime_shared_parent_parameters": 0,
            "network_access": False,
        },
    }
    return Lfm2InterfaceFrontierResult(
        model=blank_target_model,
        receipt={**body, "result_sha256": canonical_sha256(body)},
    )


def target_unique_parameters_for_recipe(
    *,
    base_tied_target_unique_parameters: int,
    target_vocabulary_size: int,
    hidden_size: int,
    recipe: Lfm2InterfaceRecipe,
) -> int:
    """Return the exact standalone dense count before materialization."""

    for value, name in (
        (base_tied_target_unique_parameters, "base_tied_target_unique_parameters"),
        (target_vocabulary_size, "target_vocabulary_size"),
        (hidden_size, "hidden_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Lfm2InterfaceFrontierError(f"{name} must be positive.")
    if not isinstance(recipe, Lfm2InterfaceRecipe):
        raise TypeError("recipe must be Lfm2InterfaceRecipe.")
    return base_tied_target_unique_parameters + (
        target_vocabulary_size * hidden_size if recipe.untied_output else 0
    )


def base_tied_target_budget_for_recipe(
    *,
    maximum_total_target_unique_parameters: int,
    target_vocabulary_size: int,
    hidden_size: int,
    recipe: Lfm2InterfaceRecipe,
) -> int:
    """Reserve interface parameters before a dynamic body planner runs."""

    for value, name in (
        (
            maximum_total_target_unique_parameters,
            "maximum_total_target_unique_parameters",
        ),
        (target_vocabulary_size, "target_vocabulary_size"),
        (hidden_size, "hidden_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Lfm2InterfaceFrontierError(f"{name} must be positive.")
    if not isinstance(recipe, Lfm2InterfaceRecipe):
        raise TypeError("recipe must be Lfm2InterfaceRecipe.")
    reserved = (
        target_vocabulary_size * hidden_size if recipe.untied_output else 0
    )
    available = maximum_total_target_unique_parameters - reserved
    if available <= 0:
        raise Lfm2InterfaceFrontierError(
            "Target budget cannot materialize the requested dense interface."
        )
    return available


def plan_lfm2_dense_untied_target_rebalance(
    *,
    parent_unique_parameters: int,
    base_tied_target_unique_parameters: int,
    target_vocabulary_size: int,
    hidden_size: int,
    layer_count: int,
    current_intermediate_size: int,
    target_compaction_denominator: int,
    minimum_intermediate_size: int = 1,
) -> dict[str, Any]:
    """Reserve an untied head and reduce FFN width to a total-size ceiling.

    LFM2 has three dense hidden/intermediate matrices per block.  This hook
    leaves residual width, attention geometry, vocabulary, and depth fixed,
    and computes the smallest integer FFN-width reduction that makes the
    independently materialized output head fit the requested parameter ratio.
    """

    values = (
        (parent_unique_parameters, "parent_unique_parameters"),
        (
            base_tied_target_unique_parameters,
            "base_tied_target_unique_parameters",
        ),
        (target_vocabulary_size, "target_vocabulary_size"),
        (hidden_size, "hidden_size"),
        (layer_count, "layer_count"),
        (current_intermediate_size, "current_intermediate_size"),
        (target_compaction_denominator, "target_compaction_denominator"),
        (minimum_intermediate_size, "minimum_intermediate_size"),
    )
    for value, name in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Lfm2InterfaceFrontierError(f"{name} must be positive.")
    if minimum_intermediate_size > current_intermediate_size:
        raise Lfm2InterfaceFrontierError(
            "minimum_intermediate_size exceeds current_intermediate_size."
        )
    target_ceiling = parent_unique_parameters // target_compaction_denominator
    added_interface_parameters = target_vocabulary_size * hidden_size
    unrebalanced_parameters = (
        base_tied_target_unique_parameters + added_interface_parameters
    )
    overage = max(0, unrebalanced_parameters - target_ceiling)
    parameters_per_intermediate_step = 3 * hidden_size * layer_count
    reduction = (
        0
        if overage == 0
        else (
            overage + parameters_per_intermediate_step - 1
        )
        // parameters_per_intermediate_step
    )
    target_intermediate_size = current_intermediate_size - reduction
    if target_intermediate_size < minimum_intermediate_size:
        raise Lfm2InterfaceFrontierError(
            "FFN width cannot absorb the dense untied interface within budget."
        )
    removed_body_parameters = reduction * parameters_per_intermediate_step
    target_unique_parameters = unrebalanced_parameters - removed_body_parameters
    if target_unique_parameters > target_ceiling:
        raise Lfm2InterfaceFrontierError(
            "Computed FFN rebalance still exceeds its parameter ceiling."
        )
    body = {
        "schema": LFM2_UNTIED_INTERFACE_BODY_REBALANCE_SCHEMA,
        "parent_unique_parameters": parent_unique_parameters,
        "target_compaction_denominator": target_compaction_denominator,
        "target_unique_parameter_ceiling": target_ceiling,
        "base_tied_target_unique_parameters": base_tied_target_unique_parameters,
        "added_dense_untied_output_parameters": added_interface_parameters,
        "unrebalanced_target_unique_parameters": unrebalanced_parameters,
        "hidden_size_preserved": hidden_size,
        "target_vocabulary_size_preserved": target_vocabulary_size,
        "layer_count_preserved": layer_count,
        "current_intermediate_size": current_intermediate_size,
        "target_intermediate_size": target_intermediate_size,
        "intermediate_size_reduction": reduction,
        "parameters_removed_per_intermediate_step": (
            parameters_per_intermediate_step
        ),
        "removed_body_parameters": removed_body_parameters,
        "target_unique_parameters": target_unique_parameters,
        "parameter_ceiling_spare": target_ceiling - target_unique_parameters,
        "realized_parameter_compaction_ratio": (
            parent_unique_parameters / target_unique_parameters
        ),
        "residual_width_preserved": True,
        "attention_geometry_preserved": True,
        "vocabulary_preserved": True,
        "depth_preserved": True,
        "automatic_target_budget_accounting": True,
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


__all__ = [
    "DEFAULT_INTERFACE_CONTEXTS",
    "DEFAULT_INTERFACE_RECIPES",
    "LFM2_CACHED_LOGIT_RESIDUAL_HEAD_SCHEMA",
    "LFM2_INTERFACE_FRONTIER_SCHEMA",
    "LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA",
    "LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA",
    "LFM2_UNTIED_INTERFACE_BODY_REBALANCE_SCHEMA",
    "Lfm2InterfaceFrontierError",
    "Lfm2InterfaceFrontierResult",
    "Lfm2InterfaceRecipe",
    "base_tied_target_budget_for_recipe",
    "Lfm2CachedLogitResidualHead",
    "compile_lfm2_cached_logit_residual_head",
    "compile_lfm2_interface_sequence_plan",
    "compile_lfm2_inverse_bucket_output_table",
    "compile_lfm2_inverse_output_plan",
    "compile_lfm2_prefix_transition_output_table",
    "fold_lfm2_interface_candidate",
    "plan_lfm2_dense_untied_target_rebalance",
    "target_unique_parameters_for_recipe",
]
