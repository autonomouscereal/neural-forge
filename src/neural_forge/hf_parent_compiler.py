"""High-level multi-parent compiler for an exact-size blank HF child.

This module composes the existing architecture-neutral trace kernel, canonical
execution graph, and supported decoder bootstrap.  Parent models are not
invoked: callers provide already-captured canonical traces (from any modality
or provider satisfying the row contract), or full-token byte-span traces for
the tokenizer-alignment helper path.  A separate supported decoder is used
only as a structural carrier for the current blank-model bootstrap machinery.

The result is deliberately modest about what was established.  Compilation is
zero-gradient and uses no recovery loop, while capability and arbitrary
whole-model equivalence remain unproven until an external evaluation proves
them.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any, Literal

from neural_forge.hf_arch_layout import DecoderLayout, build_decoder_layout
from neural_forge.hf_blank_bootstrap import (
    BlankBootstrapResult,
    bootstrap_blank_decoder_models,
)
from neural_forge.hf_byte_span_trace import (
    ByteSpanTraceError,
    ByteSpanTraceInput,
    align_byte_span_traces,
    canonical_parent_trace_from_byte_alignment,
)
from neural_forge.hf_canonical_graph import (
    CanonicalChildSpec,
    CanonicalExecutionGraph,
    CanonicalGraphError,
    CanonicalParentLogits,
    build_canonical_execution_graph,
)
from neural_forge.hf_canonical_kernel import (
    CanonicalDepthCover,
    CanonicalKernelError,
    CanonicalParentTrace,
    CanonicalRowIdentity,
    build_proportional_depth_cover,
)
from neural_forge.hf_factorized_vocab import exact_parameter_report
from neural_forge.hf_transport import TargetSnapshot


_SCHEMA = "neural-forge-hf-parent-compiler-v1"
_RESERVED_BOOTSTRAP_ARGUMENTS = frozenset(
    {
        "validation_inputs",
        "target_calibration_inputs",
        "target_validation_inputs",
        "canonical_calibration_boundaries",
        "canonical_validation_boundaries",
        "canonical_calibration_target_log_probs",
        "canonical_validation_target_log_probs",
        "canonical_execution_graph_fingerprint",
    }
)


class HFParentCompilerError(ValueError):
    """Raised when the complete parent-to-child contract cannot be proven."""


@dataclass(frozen=True, slots=True)
class ParentContribution:
    """Immutable evidence for one simultaneous parent contribution."""

    parent_name: str
    normalized_weight: float
    calibration_row_count: int
    validation_row_count: int
    source_layer_count: int
    calibration_alignment_fingerprint: str
    validation_alignment_fingerprint: str
    maximum_calibration_map_relative_rmse: float
    mean_calibration_map_cosine_similarity: float
    calibration_mean_kl_to_probability_mixture: float
    validation_mean_kl_to_probability_mixture: float
    contributes_to_every_boundary: bool = True
    contributes_to_probability_mixture: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HFParentCompilerReport:
    """Immutable truth record for one compiler invocation."""

    schema: str
    parent_input_kind: Literal["canonical", "byte-span"]
    parent_contributions: tuple[ParentContribution, ...]
    execution_graph_fingerprint: str
    child_spec_fingerprint: str
    target_structure_fingerprint: str
    target_parameter_budget: int
    target_unique_parameter_count: int
    target_logical_parameter_count: int
    target_unique_learned_weight_bytes: int
    target_logical_learned_weight_bytes: int
    target_unique_buffer_bytes: int
    target_logical_buffer_bytes: int
    target_unique_parameter_and_buffer_bytes: int
    byte_accounting_scope: str
    serialized_artifact_bytes_proven: bool
    target_budget_exact: bool
    bootstrap_accepted: bool
    structural_carrier_state_sha256_before: str
    structural_carrier_state_sha256_after: str
    structural_carrier_unchanged: bool
    parent_payload_sha256_before: str
    parent_payload_sha256_after: str
    parent_payloads_unchanged: bool
    target_state_sha256_before: str
    target_state_sha256_after: str
    parent_models_invoked: bool
    recovery_performed: bool
    recovery_steps: int
    zero_recovery: bool
    gradients_used: bool
    capability_proven: bool
    capability_status: Literal["unproven"]
    whole_model_exactness_proven: bool
    exactness_class: Literal["finite-trace-approximate"]
    exactness_scope: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["parent_contributions"] = [
            contribution.as_dict() for contribution in self.parent_contributions
        ]
        return value

    # Match report-bearing lower-level objects without exposing mutable state.
    report = as_dict


@dataclass(frozen=True, slots=True)
class HFParentCompilerResult:
    """Frozen container for the in-place child and its compilation evidence."""

    child_model: Any
    execution_graph: CanonicalExecutionGraph
    bootstrap_result: BlankBootstrapResult
    report: HFParentCompilerReport

    @property
    def target_model(self) -> Any:
        """Descriptive alias for :attr:`child_model`."""

        return self.child_model


def compile_hf_parents_to_blank_child(
    calibration_parents: Sequence[CanonicalParentTrace] | Sequence[ByteSpanTraceInput],
    validation_parents: Sequence[CanonicalParentTrace] | Sequence[ByteSpanTraceInput],
    parent_logits: Sequence[CanonicalParentLogits]
    | Mapping[str, CanonicalParentLogits],
    *,
    structural_carrier_model: Any,
    blank_target_model: Any,
    carrier_calibration_inputs: Mapping[str, Any] | Any,
    carrier_validation_inputs: Mapping[str, Any] | Any,
    child_calibration_inputs: Mapping[str, Any] | Any,
    child_validation_inputs: Mapping[str, Any] | Any,
    child_spec: CanonicalChildSpec,
    parent_weights: Mapping[str, float] | None = None,
    parent_depth_covers: Mapping[str, CanonicalDepthCover] | None = None,
    exact_parameter_budget: int | None = None,
    ridge: float = 1e-6,
    center_rows: bool = True,
    bootstrap_kwargs: Mapping[str, Any] | None = None,
) -> HFParentCompilerResult:
    """Compile simultaneous parent traces into ``blank_target_model`` in place.

    ``CanonicalParentLogits`` values are always in child-vocabulary columns.
    Byte-span inputs adapt hidden trajectories only; heterogeneous parent
    vocabularies therefore still require explicit canonical logits and exact
    token contracts.  Calibration and validation prompt identities must be
    genuinely disjoint, not merely labeled with different split names.

    ``structural_carrier_model`` supplies supported attention/MLP operator
    structure to :func:`bootstrap_blank_decoder_models`; it is not treated as
    an authoritative behavioral parent.  The carrier and all parent payloads
    are fingerprinted before and after compilation and any mutation fails
    closed.
    """

    _require_torch()
    if not isinstance(child_spec, CanonicalChildSpec):
        raise HFParentCompilerError("child_spec must be a CanonicalChildSpec.")
    if structural_carrier_model is blank_target_model:
        raise HFParentCompilerError(
            "The structural carrier and blank target must be different models."
        )
    for name, value in (
        ("carrier_calibration_inputs", carrier_calibration_inputs),
        ("carrier_validation_inputs", carrier_validation_inputs),
        ("child_calibration_inputs", child_calibration_inputs),
        ("child_validation_inputs", child_validation_inputs),
    ):
        if value is None:
            raise HFParentCompilerError(f"{name} is required.")

    try:
        carrier_layout = build_decoder_layout(structural_carrier_model)
        target_layout = build_decoder_layout(blank_target_model)
    except (TypeError, ValueError) as exc:
        raise HFParentCompilerError(str(exc)) from exc
    _reject_cross_model_storage_aliases(
        structural_carrier_model,
        blank_target_model,
    )
    target_counts = _validate_target_contract(
        blank_target_model,
        target_layout,
        child_spec,
        exact_parameter_budget,
    )
    target_bytes = _model_storage_byte_report(blank_target_model)
    target_structure_before = fingerprint_hf_child_structure(
        blank_target_model,
        layout=target_layout,
    )
    _validate_carrier_capacity(carrier_layout, target_layout)

    try:
        calibration, validation, input_kind = _prepare_parent_traces(
            calibration_parents,
            validation_parents,
            child_spec=child_spec,
            parent_depth_covers=parent_depth_covers,
        )
        calibration, validation = _apply_parent_weights(
            calibration,
            validation,
            parent_weights,
        )
        ordered_logits = _ordered_parent_logits(parent_logits)
        _require_disjoint_splits(calibration, validation)
    except (ByteSpanTraceError, CanonicalKernelError, CanonicalGraphError) as exc:
        raise HFParentCompilerError(str(exc)) from exc

    source_payload_before = _source_parent_payload_fingerprint(
        calibration_parents,
        validation_parents,
        ordered_logits,
    )
    canonical_payload_before = _canonical_parent_payload_fingerprint(
        calibration,
        validation,
        ordered_logits,
    )
    try:
        execution_graph = build_canonical_execution_graph(
            calibration,
            validation,
            ordered_logits,
            child_spec=child_spec,
            ridge=ridge,
            center_rows=center_rows,
        )
    except (CanonicalKernelError, CanonicalGraphError) as exc:
        raise HFParentCompilerError(str(exc)) from exc

    carrier_before = _model_state_sha256(structural_carrier_model)
    target_before = _model_state_sha256(blank_target_model)
    target_snapshot = TargetSnapshot.capture({"model": blank_target_model})
    canonical_bootstrap_inputs = execution_graph.bootstrap_inputs()
    options = _bootstrap_options(bootstrap_kwargs)
    options.update(canonical_bootstrap_inputs)

    try:
        bootstrap_result = bootstrap_blank_decoder_models(
            structural_carrier_model,
            blank_target_model,
            carrier_calibration_inputs,
            validation_inputs=carrier_validation_inputs,
            target_calibration_inputs=child_calibration_inputs,
            target_validation_inputs=child_validation_inputs,
            **options,
        )

        carrier_after = _model_state_sha256(structural_carrier_model)
        source_payload_after = _source_parent_payload_fingerprint(
            calibration_parents,
            validation_parents,
            ordered_logits,
        )
        canonical_payload_after = _canonical_parent_payload_fingerprint(
            calibration,
            validation,
            ordered_logits,
        )
        if carrier_after != carrier_before:
            raise HFParentCompilerError(
                "Structural carrier state changed during blank-child compilation."
            )
        if (
            source_payload_after != source_payload_before
            or canonical_payload_after != canonical_payload_before
        ):
            raise HFParentCompilerError(
                "A canonical or source parent trace/logit payload changed during "
                "compilation."
            )

        target_layout_after = build_decoder_layout(blank_target_model)
        after_counts = _validate_target_contract(
            blank_target_model,
            target_layout_after,
            child_spec,
            exact_parameter_budget,
        )
        after_bytes = _model_storage_byte_report(blank_target_model)
        target_structure_after = fingerprint_hf_child_structure(
            blank_target_model,
            layout=target_layout_after,
        )
        if (
            after_counts != target_counts
            or after_bytes != target_bytes
            or target_structure_after != target_structure_before
        ):
            raise HFParentCompilerError(
                "Blank bootstrap changed the target parameter/buffer topology or "
                "accounting."
            )
        transaction = getattr(bootstrap_result, "transaction", None)
        if transaction is None or not isinstance(
            getattr(transaction, "accepted", None), bool
        ):
            raise HFParentCompilerError(
                "Blank bootstrap returned no boolean transaction acceptance result."
            )
        accepted = bool(transaction.accepted)
        if not accepted:
            target_snapshot.assert_exact()
        target_after = _model_state_sha256(blank_target_model)
        contributions = _parent_contributions(
            execution_graph,
            calibration,
            validation,
        )
        report = HFParentCompilerReport(
            schema=_SCHEMA,
            parent_input_kind=input_kind,
            parent_contributions=contributions,
            execution_graph_fingerprint=execution_graph.fingerprint,
            child_spec_fingerprint=child_spec.fingerprint,
            target_structure_fingerprint=target_structure_after,
            target_parameter_budget=child_spec.parameter_budget,
            target_unique_parameter_count=target_counts[0],
            target_logical_parameter_count=target_counts[1],
            target_unique_learned_weight_bytes=target_bytes["unique_parameter_bytes"],
            target_logical_learned_weight_bytes=target_bytes["logical_parameter_bytes"],
            target_unique_buffer_bytes=target_bytes["unique_buffer_bytes"],
            target_logical_buffer_bytes=target_bytes["logical_buffer_bytes"],
            target_unique_parameter_and_buffer_bytes=target_bytes[
                "unique_parameter_and_buffer_bytes"
            ],
            byte_accounting_scope=(
                "materialized parameter and registered-buffer tensor payload bytes; "
                "excludes serializer metadata, config/tokenizer files, padding, and "
                "compression"
            ),
            serialized_artifact_bytes_proven=False,
            target_budget_exact=True,
            bootstrap_accepted=accepted,
            structural_carrier_state_sha256_before=carrier_before,
            structural_carrier_state_sha256_after=carrier_after,
            structural_carrier_unchanged=True,
            parent_payload_sha256_before=canonical_payload_before,
            parent_payload_sha256_after=canonical_payload_after,
            parent_payloads_unchanged=True,
            target_state_sha256_before=target_before,
            target_state_sha256_after=target_after,
            parent_models_invoked=False,
            recovery_performed=False,
            recovery_steps=0,
            zero_recovery=True,
            gradients_used=False,
            capability_proven=False,
            capability_status="unproven",
            whole_model_exactness_proven=False,
            exactness_class="finite-trace-approximate",
            exactness_scope=(
                "fingerprinted calibration/validation canonical rows and child-space "
                "probability mixtures; no arbitrary whole-model or downstream "
                "capability equivalence"
            ),
        )
        result = HFParentCompilerResult(
            child_model=blank_target_model,
            execution_graph=execution_graph,
            bootstrap_result=bootstrap_result,
            report=report,
        )
    except BaseException as exc:
        _restore_target_after_failure(target_snapshot, exc)
        _prove_unchanged_after_failure(
            structural_carrier_model,
            carrier_before,
            calibration_parents,
            validation_parents,
            ordered_logits,
            source_payload_before,
            calibration,
            validation,
            canonical_payload_before,
            exc,
        )
        raise
    return result


def fingerprint_hf_child_structure(
    model: Any,
    *,
    layout: DecoderLayout | Any | None = None,
) -> str:
    """Fingerprint supported decoder structure, shapes, and storage aliases."""

    resolved = build_decoder_layout(model) if layout is None else layout
    aliases: dict[tuple[Any, ...], int] = {}
    parameters = []
    for name, parameter in sorted(
        model.named_parameters(recurse=True, remove_duplicate=False)
    ):
        key = _parameter_storage_key(parameter)
        alias = aliases.setdefault(key, len(aliases))
        parameters.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "stride": list(parameter.stride()),
                "alias_group": alias,
            }
        )
    payload = {
        "schema": "neural-forge-supported-child-structure-v1",
        "layout": _stable_layout_value(resolved),
        "parameters": parameters,
    }
    return _json_sha256(payload)


def _prepare_parent_traces(
    calibration_values: Sequence[CanonicalParentTrace] | Sequence[ByteSpanTraceInput],
    validation_values: Sequence[CanonicalParentTrace] | Sequence[ByteSpanTraceInput],
    *,
    child_spec: CanonicalChildSpec,
    parent_depth_covers: Mapping[str, CanonicalDepthCover] | None,
) -> tuple[
    tuple[CanonicalParentTrace, ...],
    tuple[CanonicalParentTrace, ...],
    Literal["canonical", "byte-span"],
]:
    calibration = _nonempty_sequence(calibration_values, "calibration_parents")
    validation = _nonempty_sequence(validation_values, "validation_parents")
    canonical = all(isinstance(value, CanonicalParentTrace) for value in calibration)
    canonical_validation = all(
        isinstance(value, CanonicalParentTrace) for value in validation
    )
    byte = all(isinstance(value, ByteSpanTraceInput) for value in calibration)
    byte_validation = all(isinstance(value, ByteSpanTraceInput) for value in validation)
    if canonical and canonical_validation:
        if parent_depth_covers is not None:
            raise HFParentCompilerError(
                "parent_depth_covers applies only to ByteSpanTraceInput values."
            )
        return tuple(calibration), tuple(validation), "canonical"
    if not (byte and byte_validation):
        raise HFParentCompilerError(
            "Calibration and validation parents must both be homogeneous "
            "CanonicalParentTrace or ByteSpanTraceInput inventories."
        )

    calibration_alignment = align_byte_span_traces(
        tuple(calibration[0].raw_prompts),
        calibration,
        split_name="calibration",
    )
    validation_alignment = align_byte_span_traces(
        tuple(validation[0].raw_prompts),
        validation,
        split_name="validation",
    )
    calibration_names = {parent.parent_name for parent in calibration_alignment.parents}
    validation_names = {parent.parent_name for parent in validation_alignment.parents}
    if calibration_names != validation_names:
        raise HFParentCompilerError(
            "Calibration and validation byte-span parent names differ."
        )
    covers = _byte_depth_covers(
        calibration_alignment,
        validation_alignment,
        child_spec.layer_count,
        parent_depth_covers,
    )
    calibration_traces = tuple(
        canonical_parent_trace_from_byte_alignment(
            calibration_alignment,
            name,
            depth_cover=covers[name],
        )
        for name in sorted(calibration_names)
    )
    validation_traces = tuple(
        canonical_parent_trace_from_byte_alignment(
            validation_alignment,
            name,
            depth_cover=covers[name],
        )
        for name in sorted(validation_names)
    )
    return calibration_traces, validation_traces, "byte-span"


def _byte_depth_covers(
    calibration: Any,
    validation: Any,
    target_layer_count: int,
    supplied: Mapping[str, CanonicalDepthCover] | None,
) -> dict[str, CanonicalDepthCover]:
    names = {parent.parent_name for parent in calibration.parents}
    if supplied is not None and not isinstance(supplied, Mapping):
        raise HFParentCompilerError("parent_depth_covers must be a mapping.")
    if supplied is not None and set(supplied) != names:
        raise HFParentCompilerError(
            "parent_depth_covers must contain exactly the byte-span parent names."
        )
    result: dict[str, CanonicalDepthCover] = {}
    for name in sorted(names):
        calibration_parent = calibration.parent(name)
        validation_parent = validation.parent(name)
        calibration_layers = len(calibration_parent.aligned_hidden_states) - 1
        validation_layers = len(validation_parent.aligned_hidden_states) - 1
        if calibration_layers != validation_layers or calibration_layers <= 0:
            raise HFParentCompilerError(
                f"Parent {name!r} has inconsistent or empty hidden boundaries."
            )
        cover = (
            build_proportional_depth_cover(calibration_layers, target_layer_count)
            if supplied is None
            else supplied[name]
        )
        if not isinstance(cover, CanonicalDepthCover):
            raise HFParentCompilerError(
                f"parent_depth_covers[{name!r}] must be a CanonicalDepthCover."
            )
        if (
            cover.source_layer_count != calibration_layers
            or cover.target_layer_count != target_layer_count
        ):
            raise HFParentCompilerError(
                f"Parent {name!r} depth cover does not match its trace or child depth."
            )
        result[name] = cover
    return result


def _apply_parent_weights(
    calibration: tuple[CanonicalParentTrace, ...],
    validation: tuple[CanonicalParentTrace, ...],
    supplied: Mapping[str, float] | None,
) -> tuple[tuple[CanonicalParentTrace, ...], tuple[CanonicalParentTrace, ...]]:
    calibration_by_name = _unique_traces_by_name(calibration, "calibration")
    validation_by_name = _unique_traces_by_name(validation, "validation")
    if set(calibration_by_name) != set(validation_by_name):
        raise HFParentCompilerError(
            "Calibration and validation parent names must match exactly."
        )
    if supplied is None:
        weights = {
            name: _positive_weight(parent.weight, f"{name}.weight")
            for name, parent in calibration_by_name.items()
        }
        for name, parent in validation_by_name.items():
            if _positive_weight(parent.weight, f"{name}.weight") != weights[name]:
                raise HFParentCompilerError(
                    "Calibration and validation parent weights must match exactly."
                )
    else:
        if not isinstance(supplied, Mapping) or set(supplied) != set(
            calibration_by_name
        ):
            raise HFParentCompilerError(
                "parent_weights must map exactly every canonical parent name."
            )
        weights = {
            name: _positive_weight(value, f"parent_weights[{name!r}]")
            for name, value in supplied.items()
        }
    return (
        tuple(replace(parent, weight=weights[parent.name]) for parent in calibration),
        tuple(replace(parent, weight=weights[parent.name]) for parent in validation),
    )


def _ordered_parent_logits(
    values: Sequence[CanonicalParentLogits] | Mapping[str, CanonicalParentLogits],
) -> tuple[CanonicalParentLogits, ...]:
    if isinstance(values, Mapping):
        supplied = tuple(values.values())
        for name, value in values.items():
            if (
                not isinstance(value, CanonicalParentLogits)
                or value.parent_name != name
            ):
                raise HFParentCompilerError(
                    "A parent_logits mapping key does not match its parent_name."
                )
    else:
        supplied = _nonempty_sequence(values, "parent_logits")
    if not supplied or any(
        not isinstance(value, CanonicalParentLogits) for value in supplied
    ):
        raise HFParentCompilerError(
            "parent_logits must contain CanonicalParentLogits values."
        )
    names = [value.parent_name for value in supplied]
    if len(set(names)) != len(names):
        raise HFParentCompilerError("Canonical parent logit names must be unique.")
    return tuple(sorted(supplied, key=lambda value: value.parent_name))


def _require_disjoint_splits(
    calibration: tuple[CanonicalParentTrace, ...],
    validation: tuple[CanonicalParentTrace, ...],
) -> None:
    calibration_identity = _shared_identity(calibration, "calibration")
    validation_identity = _shared_identity(validation, "validation")
    if calibration_identity.split_name == validation_identity.split_name:
        raise HFParentCompilerError(
            "Calibration and validation row identities must use different splits."
        )
    row_overlap = set(calibration_identity.row_ids).intersection(
        validation_identity.row_ids
    )
    prompt_overlap = set(calibration_identity.raw_prompt_sha256).intersection(
        validation_identity.raw_prompt_sha256
    )
    if row_overlap or prompt_overlap:
        raise HFParentCompilerError(
            "Calibration and validation row identities are stale or overlap."
        )
    if calibration_identity.corpus_sha256 == validation_identity.corpus_sha256:
        raise HFParentCompilerError(
            "Calibration and validation row identities reuse the same corpus."
        )


def _shared_identity(
    parents: tuple[CanonicalParentTrace, ...],
    label: str,
) -> CanonicalRowIdentity:
    if not parents:
        raise HFParentCompilerError(f"{label} parents cannot be empty.")
    identities = [parent.row_identity for parent in parents]
    if any(not isinstance(value, CanonicalRowIdentity) for value in identities):
        raise HFParentCompilerError(f"{label} parents contain an invalid row identity.")
    fingerprints = {value.alignment_fingerprint for value in identities}
    if len(fingerprints) != 1:
        raise HFParentCompilerError(
            f"{label} parent rows are stale, reordered, or misaligned."
        )
    return identities[0]


def _validate_target_contract(
    model: Any,
    layout: DecoderLayout | Any,
    child_spec: CanonicalChildSpec,
    exact_parameter_budget: int | None,
) -> tuple[int, int]:
    expected_budget = child_spec.parameter_budget
    if exact_parameter_budget is not None:
        if (
            isinstance(exact_parameter_budget, bool)
            or not isinstance(exact_parameter_budget, int)
            or exact_parameter_budget <= 0
        ):
            raise HFParentCompilerError(
                "exact_parameter_budget must be a positive integer."
            )
        if exact_parameter_budget != expected_budget:
            raise HFParentCompilerError(
                "exact_parameter_budget does not equal child_spec.parameter_budget."
            )
    actual_family = _normalized_family(layout.family)
    specified_family = _normalized_family(child_spec.family)
    if specified_family != actual_family:
        raise HFParentCompilerError(
            f"Target family {actual_family!r} does not match child spec "
            f"{child_spec.family!r}."
        )
    checks = (
        ("layer count", int(layout.layer_count), child_spec.layer_count),
        ("hidden size", int(layout.hidden_size), child_spec.hidden_size),
        (
            "vocabulary size",
            int(layout.embeddings.vocab_size),
            child_spec.vocabulary_size,
        ),
    )
    for label, actual, expected in checks:
        if actual != expected:
            raise HFParentCompilerError(
                f"Target {label} is {actual}, but child spec requires {expected}."
            )
    actual_position = _normalized_position_policy(layout.position.kind)
    expected_position = _normalized_position_policy(child_spec.position_policy)
    if actual_position != expected_position:
        raise HFParentCompilerError(
            "Target position layout does not match child_spec.position_policy."
        )
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if (
        embedding is None
        or head is None
        or not hasattr(embedding, "weight")
        or not hasattr(head, "weight")
    ):
        raise HFParentCompilerError(
            "Target must expose materialized input-embedding and output-head weights."
        )
    storage_tied = _parameter_storage_key(embedding.weight) == _parameter_storage_key(
        head.weight
    )
    if bool(layout.tie_word_embeddings) != storage_tied:
        raise HFParentCompilerError(
            "Target layout tying metadata disagrees with actual embedding/head storage."
        )
    if storage_tied != child_spec.tie_embeddings:
        raise HFParentCompilerError(
            "Target embedding/head tied layout does not match child spec."
        )
    try:
        counts = exact_parameter_report(model)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HFParentCompilerError(
            "Target exact unique-parameter accounting failed."
        ) from exc
    unique = int(counts["unique_parameter_count"])
    logical = int(counts["logical_parameter_count"])
    if unique != expected_budget:
        raise HFParentCompilerError(
            f"Target unique parameter count is {unique}, but exact budget is "
            f"{expected_budget}."
        )
    return unique, logical


def _validate_carrier_capacity(carrier: Any, target: Any) -> None:
    if int(target.layer_count) > int(carrier.layer_count):
        raise HFParentCompilerError(
            "Structural carrier has fewer layers than the blank target."
        )
    if int(target.hidden_size) > int(carrier.hidden_size):
        raise HFParentCompilerError(
            "Structural carrier hidden size is smaller than the blank target."
        )
    if int(target.intermediate_size) > int(carrier.intermediate_size):
        raise HFParentCompilerError(
            "Structural carrier MLP width is smaller than the blank target."
        )


def _reject_cross_model_storage_aliases(carrier: Any, target: Any) -> None:
    carrier_storage = _model_storage_roots(carrier)
    target_storage = _model_storage_roots(target)
    if carrier_storage.intersection(target_storage):
        raise HFParentCompilerError(
            "Structural carrier and blank target share parameter storage."
        )


def _bootstrap_options(values: Mapping[str, Any] | None) -> dict[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise HFParentCompilerError("bootstrap_kwargs must be a mapping.")
    overlap = _RESERVED_BOOTSTRAP_ARGUMENTS.intersection(values)
    if overlap:
        shown = ", ".join(sorted(overlap))
        raise HFParentCompilerError(
            f"bootstrap_kwargs cannot override compiler-owned arguments: {shown}."
        )
    return dict(values)


def _parent_contributions(
    graph: CanonicalExecutionGraph,
    calibration: tuple[CanonicalParentTrace, ...],
    validation: tuple[CanonicalParentTrace, ...],
) -> tuple[ParentContribution, ...]:
    plan = graph.boundary_splits.calibration_plan
    calibration_by_name = {value.name: value for value in calibration}
    validation_by_name = {value.name: value for value in validation}
    diagnostics = {
        str(value["parent_name"]): value for value in graph.parent_logit_diagnostics
    }
    result = []
    for index, name in enumerate(plan.parent_names):
        maps = [
            next(item for item in layer.parent_maps if item.parent_name == name)
            for layer in plan.layers
        ]
        diagnostic = diagnostics[name]
        result.append(
            ParentContribution(
                parent_name=name,
                normalized_weight=plan.normalized_parent_weights[index],
                calibration_row_count=len(
                    calibration_by_name[name].row_identity.row_ids
                ),
                validation_row_count=len(validation_by_name[name].row_identity.row_ids),
                source_layer_count=(
                    calibration_by_name[name].depth_cover.source_layer_count
                ),
                calibration_alignment_fingerprint=(
                    calibration_by_name[name].row_identity.alignment_fingerprint
                ),
                validation_alignment_fingerprint=(
                    validation_by_name[name].row_identity.alignment_fingerprint
                ),
                maximum_calibration_map_relative_rmse=max(
                    item.relative_rmse for item in maps
                ),
                mean_calibration_map_cosine_similarity=(
                    sum(item.mean_cosine_similarity for item in maps) / len(maps)
                ),
                calibration_mean_kl_to_probability_mixture=float(
                    diagnostic["calibration_mean_kl_to_mixture"]
                ),
                validation_mean_kl_to_probability_mixture=float(
                    diagnostic["validation_mean_kl_to_mixture"]
                ),
            )
        )
    return tuple(result)


def _prove_unchanged_after_failure(
    carrier: Any,
    carrier_before: str,
    source_calibration: Sequence[Any],
    source_validation: Sequence[Any],
    logits: tuple[CanonicalParentLogits, ...],
    source_before: str,
    canonical_calibration: tuple[CanonicalParentTrace, ...],
    canonical_validation: tuple[CanonicalParentTrace, ...],
    canonical_before: str,
    original: Exception,
) -> None:
    carrier_changed = _model_state_sha256(carrier) != carrier_before
    source_changed = (
        _source_parent_payload_fingerprint(
            source_calibration,
            source_validation,
            logits,
        )
        != source_before
    )
    canonical_changed = (
        _canonical_parent_payload_fingerprint(
            canonical_calibration,
            canonical_validation,
            logits,
        )
        != canonical_before
    )
    if carrier_changed or source_changed or canonical_changed:
        raise HFParentCompilerError(
            "Compilation failed after mutating the carrier or a parent payload."
        ) from original


def _restore_target_after_failure(
    snapshot: TargetSnapshot, original: BaseException
) -> None:
    try:
        snapshot.restore()
        snapshot.assert_exact()
    except BaseException as rollback_error:
        raise HFParentCompilerError(
            "Compilation failed and exact blank-target rollback also failed."
        ) from rollback_error


def _canonical_parent_payload_fingerprint(
    calibration: tuple[CanonicalParentTrace, ...],
    validation: tuple[CanonicalParentTrace, ...],
    logits: tuple[CanonicalParentLogits, ...],
) -> str:
    return _json_sha256(
        {
            "calibration": sorted(
                (_canonical_trace_record(value) for value in calibration),
                key=lambda value: value["name"],
            ),
            "validation": sorted(
                (_canonical_trace_record(value) for value in validation),
                key=lambda value: value["name"],
            ),
            "logits": sorted(
                (_canonical_logits_record(value) for value in logits),
                key=lambda value: value["name"],
            ),
        }
    )


def _source_parent_payload_fingerprint(
    calibration: Sequence[Any],
    validation: Sequence[Any],
    logits: tuple[CanonicalParentLogits, ...],
) -> str:
    return _json_sha256(
        {
            "calibration": sorted(
                (_source_trace_record(value) for value in calibration),
                key=lambda value: value["name"],
            ),
            "validation": sorted(
                (_source_trace_record(value) for value in validation),
                key=lambda value: value["name"],
            ),
            "logits": sorted(
                (_canonical_logits_record(value) for value in logits),
                key=lambda value: value["name"],
            ),
        }
    )


def _canonical_trace_record(value: CanonicalParentTrace) -> dict[str, Any]:
    return {
        "name": value.name,
        "row_identity": value.row_identity.report(),
        "depth_cover": value.depth_cover.report(),
        "weight": float(value.weight),
        "states": [_tensor_sha256(item) for item in value.layer_states],
    }


def _source_trace_record(value: Any) -> dict[str, Any]:
    if isinstance(value, CanonicalParentTrace):
        return _canonical_trace_record(value)
    if not isinstance(value, ByteSpanTraceInput):
        raise HFParentCompilerError("Unexpected parent trace payload type.")
    return {
        "name": value.parent_name,
        "raw_prompts": list(value.raw_prompts),
        "input_ids": _tensor_sha256(value.input_ids),
        "offset_mapping": _tensor_sha256(value.offset_mapping),
        "attention_mask": _tensor_sha256(value.attention_mask),
        "hidden_states": [
            _tensor_sha256(item) for item in _tensor_sequence(value.hidden_states)
        ],
        "logits": None if value.logits is None else _tensor_sha256(value.logits),
        "offset_unit": value.offset_unit,
        "tokenizer_fingerprint": value.tokenizer_fingerprint,
        "cache_fingerprint": value.cache_fingerprint,
    }


def _canonical_logits_record(value: CanonicalParentLogits) -> dict[str, Any]:
    return {
        "name": value.parent_name,
        "calibration": _tensor_sha256(value.calibration_logits),
        "validation": _tensor_sha256(value.validation_logits),
        "token_contract": value.token_contract.report(),
    }


def _model_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_sha256(value).encode("ascii"))
    aliases: dict[tuple[Any, ...], int] = {}
    for name, parameter in sorted(
        model.named_parameters(recurse=True, remove_duplicate=False)
    ):
        key = _parameter_storage_key(parameter)
        alias = aliases.setdefault(key, len(aliases))
        digest.update(name.encode("utf-8"))
        digest.update(str(alias).encode("ascii"))
        digest.update(str(bool(parameter.requires_grad)).encode("ascii"))
    for name, module in sorted(model.named_modules()):
        digest.update(name.encode("utf-8"))
        digest.update(str(bool(module.training)).encode("ascii"))
    return digest.hexdigest()


def _model_storage_byte_report(model: Any) -> dict[str, int]:
    parameter_unique: dict[tuple[Any, ...], int] = {}
    buffer_unique: dict[tuple[Any, ...], int] = {}
    combined_unique: dict[tuple[Any, ...], int] = {}
    logical_parameter_bytes = 0
    logical_buffer_bytes = 0
    for _, value in model.named_parameters(recurse=True, remove_duplicate=False):
        byte_count = int(value.numel()) * int(value.element_size())
        logical_parameter_bytes += byte_count
        key = _parameter_storage_key(value)
        parameter_unique[key] = byte_count
        combined_unique[key] = byte_count
    for _, value in model.named_buffers(recurse=True, remove_duplicate=False):
        byte_count = int(value.numel()) * int(value.element_size())
        logical_buffer_bytes += byte_count
        key = _parameter_storage_key(value)
        buffer_unique[key] = byte_count
        existing = combined_unique.get(key)
        if existing is not None and existing != byte_count:
            raise HFParentCompilerError(
                "Aliased parameter/buffer storage has inconsistent byte counts."
            )
        combined_unique[key] = byte_count
    return {
        "unique_parameter_bytes": sum(parameter_unique.values()),
        "logical_parameter_bytes": logical_parameter_bytes,
        "unique_buffer_bytes": sum(buffer_unique.values()),
        "logical_buffer_bytes": logical_buffer_bytes,
        "unique_parameter_and_buffer_bytes": sum(combined_unique.values()),
    }


def _tensor_sha256(value: Any) -> str:
    torch = _require_torch()
    if not isinstance(value, torch.Tensor):
        raise HFParentCompilerError("Parent/model payloads must contain tensors.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise HFParentCompilerError(
            "Parent/model payload tensors must be materialized and dense strided."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    digest.update(str(bool(value.requires_grad)).encode("ascii"))
    return digest.hexdigest()


def _parameter_storage_key(value: Any) -> tuple[Any, ...]:
    torch = _require_torch()
    if not isinstance(value, torch.Tensor):
        raise HFParentCompilerError("Expected a tensor-backed model parameter.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise HFParentCompilerError(
            "Exact parameter accounting requires materialized dense tensors."
        )
    return (
        value.device.type,
        value.device.index,
        str(value.dtype),
        int(value.untyped_storage().data_ptr()),
        int(value.storage_offset()),
        tuple(value.shape),
        tuple(value.stride()),
    )


def _model_storage_roots(model: Any) -> set[tuple[Any, ...]]:
    result = set()
    for _, parameter in model.named_parameters(recurse=True, remove_duplicate=False):
        key = _parameter_storage_key(parameter)
        result.add(key[:4])
    return result


def _stable_layout_value(value: Any) -> Any:
    if is_dataclass(value):
        return _stable_layout_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _stable_layout_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable_layout_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _stable_layout_value(vars(value))
    return type(value).__name__


def _json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HFParentCompilerError("Compiler evidence is not canonical JSON.") from exc
    return hashlib.sha256(encoded).hexdigest()


def _unique_traces_by_name(
    values: tuple[CanonicalParentTrace, ...],
    label: str,
) -> dict[str, CanonicalParentTrace]:
    result: dict[str, CanonicalParentTrace] = {}
    for value in values:
        if not isinstance(value.name, str) or not value.name.strip():
            raise HFParentCompilerError(f"{label} parent names must be nonempty.")
        if value.name in result:
            raise HFParentCompilerError(f"{label} parent names must be unique.")
        result[value.name] = value
    return result


def _nonempty_sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HFParentCompilerError(f"{name} must be a sequence.")
    result = tuple(value)
    if not result:
        raise HFParentCompilerError(f"{name} cannot be empty.")
    return result


def _tensor_sequence(value: Any) -> tuple[Any, ...]:
    torch = _require_torch()
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    raise HFParentCompilerError("hidden_states must be a tensor or tensor sequence.")


def _positive_weight(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise HFParentCompilerError(f"{name} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFParentCompilerError(f"{name} must be finite and positive.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise HFParentCompilerError(f"{name} must be finite and positive.")
    return result


def _normalized_position_policy(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HFParentCompilerError("Position policy must be a nonempty string.")
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"rope", "rotary", "rotary_embedding"}:
        return "rope"
    if normalized in {"learned", "absolute", "learned_absolute"}:
        return "learned_absolute"
    return normalized


def _normalized_family(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HFParentCompilerError("Model family must be a nonempty string.")
    normalized = value.strip().lower().replace("-", "_")
    if normalized.startswith("blank_"):
        normalized = normalized[6:]
    if normalized == "gemma3_text":
        return "gemma3"
    return normalized


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "PyTorch is required for neural_forge.hf_parent_compiler."
        ) from exc
    return torch


# Descriptive aliases for callers migrating from lower-level graph terminology.
compile_hf_parent_traces_to_blank_child = compile_hf_parents_to_blank_child
compile_multi_parent_blank_child = compile_hf_parents_to_blank_child


__all__ = [
    "HFParentCompilerError",
    "HFParentCompilerReport",
    "HFParentCompilerResult",
    "ParentContribution",
    "compile_hf_parent_traces_to_blank_child",
    "compile_hf_parents_to_blank_child",
    "compile_multi_parent_blank_child",
    "fingerprint_hf_child_structure",
]
