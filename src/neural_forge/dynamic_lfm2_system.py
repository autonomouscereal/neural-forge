"""Persistent LFM2 target-to-folding-stack catalog integration."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .dynamic_behavior_gate import (
    DynamicBehaviorGateError,
    expose_behavior_gate_binding,
    resolve_behavior_gate_binding,
)
from .dynamic_circuit_ir import canonical_sha256
from .dynamic_fold_compiler import DynamicFoldSystemStack
from .dynamic_fold_executor import (
    FoldDecision,
    FoldEvaluator,
    FoldOperatorRegistry,
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from .dynamic_fold_registry import (
    ClosureLevel,
    ExperimentLedger,
    LedgerRecord,
    RecordStatus,
)
from .dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from .dynamic_lfm2_adapter import (
    DynamicLfm2AdapterError,
    HYBRID_MODEL_METHOD,
    LAYER_METHOD,
    PRODUCER_FIRST_MODEL_METHOD,
    Lfm2DenseBudgetedSelectedTarget,
    Lfm2DynamicCompilerBridge,
    Lfm2EvidenceBuilder,
    Lfm2HybridSelectedTarget,
    Lfm2ModelEvidenceBuilder,
    Lfm2ProducerFirstEvidenceBuilder,
    Lfm2SelectedTarget,
    Lfm2SelectedTargetLike,
    Lfm2TargetSynthesisAdapter,
    build_lfm2_dynamic_compiler_bridge,
    encode_lfm2_layer_types,
    encode_lfm2_source_spans,
    validate_lfm2_dense_budgeted_child,
    validate_lfm2_hybrid_child,
    validate_lfm2_selected_child,
)
from .dynamic_parameter_frontier import LFM2_HETEROGENEOUS_LAYOUT_KIND
from .dynamic_target_synthesis import (
    DynamicTargetSynthesisError,
    DynamicTargetSynthesisResult,
    verify_compiler_authenticated_target,
)
from .hf_lfm2_closed_loop_fold import Lfm2ClosedLoopFoldConfig
from .hf_lfm2_balanced_basis_phase_cache import (
    Lfm2BalancedBasisPhaseCacheContext,
)
from .hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2StrictTokenizerBridgeTraceAdapter,
    Lfm2StrictTokenizerBridgeTraceError,
)
from .hf_lfm2_streaming_vocabulary_margin_adapter import (
    Lfm2TargetTokenizerVocabularyMarginBinding,
)
from .hf_tokenizer_bridge import StrictTokenizerBridgeArtifact

if TYPE_CHECKING:
    from .hf_lfm2_dense_budget_model_fold import (
        Lfm2DenseCacheVerificationLease,
    )


LFM2_STACK_ADAPTER_ID = "neural_forge.lfm2.closed_loop_catalog"
_DENSE_BUDGET_MODEL_METHOD = "lfm2_dense_budget_residual_attention_model"


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicLfm2AdapterError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicLfm2AdapterError(f"{name} must be a nonempty string.")
    return value.strip()


@dataclass(frozen=True, slots=True)
class Lfm2DynamicStackResources:
    """Reusable cache, evaluation, and local artifact callbacks."""

    parent_cache_fingerprint: str
    parent_cache_payload: Any
    evidence_builder: Lfm2EvidenceBuilder
    final_evaluator: FoldEvaluator
    model_saver: Callable[[Any, Path], None]
    model_loader: Callable[[Path], Any]
    fresh_reload_evaluator: Callable[[Any], Any]
    saver_id: str
    loader_id: str
    ledger: ExperimentLedger
    fold_config: Lfm2ClosedLoopFoldConfig | None = None
    model_evidence_builder: Lfm2ModelEvidenceBuilder | None = None
    model_fold_config: Any | None = None
    producer_first_evidence_builder: Lfm2ProducerFirstEvidenceBuilder | None = None
    producer_first_fold_config: Any | None = None
    parent_cache_integrity_check: Callable[[Any], str] | None = None
    snapshot_config: DynamicFoldSnapshotConfig | None = None
    dense_fold_config: Any | None = None
    residual_basis_phase_cache_context: (
        Lfm2BalancedBasisPhaseCacheContext | None
    ) = None
    tokenizer_bridge_artifact: StrictTokenizerBridgeArtifact | None = None
    tokenizer_bridge_trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter | None = None
    target_tokenizer_vocabulary_margin_binding: (
        Lfm2TargetTokenizerVocabularyMarginBinding | None
    ) = None
    autoregressive_hankel_observability: Any | None = None
    dense_fold_progress_sink: (
        Callable[[str, Mapping[str, Any]], None] | None
    ) = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_cache_fingerprint",
            _digest(
                self.parent_cache_fingerprint,
                "parent_cache_fingerprint",
            ),
        )
        for name in (
            "evidence_builder",
            "model_saver",
            "model_loader",
            "fresh_reload_evaluator",
        ):
            if not callable(getattr(self, name)):
                raise DynamicLfm2AdapterError(
                    f"LFM2 stack resource {name} must be callable."
                )
        if not callable(getattr(self.final_evaluator, "evaluate", None)):
            raise DynamicLfm2AdapterError(
                "LFM2 stack final_evaluator must implement evaluate()."
            )
        try:
            final_binding = resolve_behavior_gate_binding(self.final_evaluator)
        except DynamicBehaviorGateError:
            final_binding = None
        try:
            reload_binding = resolve_behavior_gate_binding(self.fresh_reload_evaluator)
        except DynamicBehaviorGateError:
            reload_binding = None
        if (final_binding is None) != (reload_binding is None):
            raise DynamicLfm2AdapterError(
                "LFM2 in-memory and fresh-reload evaluators must both expose "
                "the behavior manifest or neither expose it."
            )
        if (
            final_binding is not None
            and reload_binding is not None
            and final_binding.manifest_fingerprint
            != reload_binding.manifest_fingerprint
        ):
            raise DynamicLfm2AdapterError(
                "LFM2 in-memory and fresh-reload behavior manifests differ."
            )
        object.__setattr__(
            self,
            "saver_id",
            _text(self.saver_id, "saver_id"),
        )
        object.__setattr__(
            self,
            "loader_id",
            _text(self.loader_id, "loader_id"),
        )
        if not isinstance(self.ledger, ExperimentLedger):
            raise DynamicLfm2AdapterError("LFM2 stack ledger must be ExperimentLedger.")
        if self.fold_config is not None and not isinstance(
            self.fold_config,
            Lfm2ClosedLoopFoldConfig,
        ):
            raise DynamicLfm2AdapterError(
                "fold_config must be Lfm2ClosedLoopFoldConfig or None."
            )
        if self.model_evidence_builder is not None and not callable(
            self.model_evidence_builder
        ):
            raise DynamicLfm2AdapterError(
                "model_evidence_builder must be callable or None."
            )
        if self.model_fold_config is not None:
            try:
                from .hf_lfm2_model_fold import Lfm2ModelFoldConfig
            except ImportError as exc:  # pragma: no cover - optional at import time
                raise DynamicLfm2AdapterError(
                    "The LFM2 model-fold primitive is unavailable."
                ) from exc
            if not isinstance(self.model_fold_config, Lfm2ModelFoldConfig):
                raise DynamicLfm2AdapterError(
                    "model_fold_config must be Lfm2ModelFoldConfig or None."
                )
        if self.producer_first_evidence_builder is not None and not callable(
            self.producer_first_evidence_builder
        ):
            raise DynamicLfm2AdapterError(
                "producer_first_evidence_builder must be callable or None."
            )
        if self.producer_first_fold_config is not None:
            try:
                from .hf_lfm2_producer_first_model_fold import (
                    Lfm2ProducerFirstFoldConfig,
                )
            except ImportError as exc:  # pragma: no cover - optional at import time
                raise DynamicLfm2AdapterError(
                    "The producer-first LFM2 primitive is unavailable."
                ) from exc
            if not isinstance(
                self.producer_first_fold_config,
                Lfm2ProducerFirstFoldConfig,
            ):
                raise DynamicLfm2AdapterError(
                    "producer_first_fold_config has the wrong type."
                )
        if (self.producer_first_evidence_builder is None) != (
            self.producer_first_fold_config is None
        ):
            raise DynamicLfm2AdapterError(
                "Producer-first evidence and fold config must be supplied together."
            )
        if (
            self.producer_first_evidence_builder is not None
            and self.model_evidence_builder is not None
        ):
            raise DynamicLfm2AdapterError(
                "Producer-first full-depth and hybrid depth resources are "
                "mutually exclusive."
            )
        if self.parent_cache_integrity_check is not None and not callable(
            self.parent_cache_integrity_check
        ):
            raise DynamicLfm2AdapterError(
                "parent_cache_integrity_check must be callable or None."
            )
        if self.snapshot_config is not None and not isinstance(
            self.snapshot_config,
            DynamicFoldSnapshotConfig,
        ):
            raise DynamicLfm2AdapterError(
                "snapshot_config must be DynamicFoldSnapshotConfig or None."
            )
        if self.dense_fold_config is not None:
            try:
                from .hf_lfm2_dense_budget_model_fold import (
                    Lfm2DenseBudgetFoldConfig,
                )
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise DynamicLfm2AdapterError(
                    "The dense-budget LFM2 fold primitive is unavailable."
                ) from exc
            if not isinstance(
                self.dense_fold_config,
                Lfm2DenseBudgetFoldConfig,
            ):
                raise DynamicLfm2AdapterError(
                    "dense_fold_config has the wrong type."
                )
        if (
            self.residual_basis_phase_cache_context is not None
            and not isinstance(
                self.residual_basis_phase_cache_context,
                Lfm2BalancedBasisPhaseCacheContext,
            )
        ):
            raise DynamicLfm2AdapterError(
                "residual_basis_phase_cache_context has the wrong type."
            )
        if self.tokenizer_bridge_artifact is not None:
            if not isinstance(
                self.tokenizer_bridge_artifact,
                StrictTokenizerBridgeArtifact,
            ):
                raise DynamicLfm2AdapterError(
                    "tokenizer_bridge_artifact must be a "
                    "StrictTokenizerBridgeArtifact."
                )
            try:
                self.tokenizer_bridge_artifact.verify_files()
            except ValueError as exc:
                raise DynamicLfm2AdapterError(
                    "tokenizer_bridge_artifact failed verification."
                ) from exc
        if self.tokenizer_bridge_trace_adapter is not None:
            if not isinstance(
                self.tokenizer_bridge_trace_adapter,
                Lfm2StrictTokenizerBridgeTraceAdapter,
            ):
                raise DynamicLfm2AdapterError(
                    "tokenizer_bridge_trace_adapter has the wrong type."
                )
            if self.tokenizer_bridge_artifact is None:
                raise DynamicLfm2AdapterError(
                    "Tokenizer bridge trace evidence requires its bridge artifact."
                )
            try:
                self.tokenizer_bridge_trace_adapter.verify()
            except Lfm2StrictTokenizerBridgeTraceError as exc:
                raise DynamicLfm2AdapterError(
                    "tokenizer_bridge_trace_adapter failed verification."
                ) from exc
            if (
                self.tokenizer_bridge_trace_adapter.bridge_artifact_sha256
                != self.tokenizer_bridge_artifact.artifact_sha256
                or self.tokenizer_bridge_trace_adapter.transport_profile_fingerprint
                != self.tokenizer_bridge_artifact.profile_fingerprint
            ):
                raise DynamicLfm2AdapterError(
                    "Tokenizer bridge trace evidence differs from its artifact."
                )
        if self.target_tokenizer_vocabulary_margin_binding is not None:
            binding = self.target_tokenizer_vocabulary_margin_binding
            if not isinstance(
                binding,
                Lfm2TargetTokenizerVocabularyMarginBinding,
            ):
                raise DynamicLfm2AdapterError(
                    "target_tokenizer_vocabulary_margin_binding has the wrong "
                    "type."
                )
            if self.tokenizer_bridge_trace_adapter is None:
                raise DynamicLfm2AdapterError(
                    "Target vocabulary-margin binding requires the stack's "
                    "strict tokenizer trace adapter."
                )
            try:
                binding.verify()
            except ValueError as exc:
                raise DynamicLfm2AdapterError(
                    "Target vocabulary-margin binding failed verification."
                ) from exc
            if (
                binding.trace_adapter is not self.tokenizer_bridge_trace_adapter
                or binding.trace_adapter.evidence_sha256
                != self.tokenizer_bridge_trace_adapter.evidence_sha256
            ):
                raise DynamicLfm2AdapterError(
                    "Target vocabulary-margin binding differs from the stack's "
                    "strict tokenizer trace adapter."
                )
        if self.autoregressive_hankel_observability is not None:
            try:
                from .dynamic_balanced_circuit_basis import (
                    PrecomputedDownstreamObservability,
                )
                from .hf_lfm2_autoregressive_hankel_observability import (
                    Lfm2AutoregressiveHankelObservabilityResult,
                )
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise DynamicLfm2AdapterError(
                    "The autoregressive Hankel fold primitive is unavailable."
                ) from exc
            value = self.autoregressive_hankel_observability
            if not isinstance(
                value,
                (
                    Lfm2AutoregressiveHankelObservabilityResult,
                    PrecomputedDownstreamObservability,
                ),
            ):
                raise DynamicLfm2AdapterError(
                    "autoregressive_hankel_observability has the wrong type."
                )
            evidence = (
                value.evidence
                if isinstance(value, Lfm2AutoregressiveHankelObservabilityResult)
                else value
            )
            if not evidence.verify():
                raise DynamicLfm2AdapterError(
                    "autoregressive_hankel_observability failed authentication."
                )
            if self.dense_fold_config is None or getattr(
                self.dense_fold_config,
                "residual_basis_mode",
                None,
            ) != "circuit_balanced":
                raise DynamicLfm2AdapterError(
                    "Autoregressive Hankel evidence requires one dense circuit-"
                    "balanced fold configuration."
                )
        if self.dense_fold_progress_sink is not None and not callable(
            self.dense_fold_progress_sink
        ):
            raise DynamicLfm2AdapterError(
                "dense_fold_progress_sink must be callable or None."
            )
        bridged_margin_enabled = bool(
            self.tokenizer_bridge_artifact is not None
            and getattr(
                self.dense_fold_config,
                "vocabulary_margin_observability",
                False,
            )
        )
        if bridged_margin_enabled and (
            not bool(
                getattr(
                    self.dense_fold_config,
                    "streaming_vocabulary_margin_v2",
                    False,
                )
            )
            or self.target_tokenizer_vocabulary_margin_binding is None
        ):
            raise DynamicLfm2AdapterError(
                "Bridged vocabulary-margin observability requires streaming v2 "
                "and its fully authenticated target binding."
            )
        if (
            not bridged_margin_enabled
            and self.target_tokenizer_vocabulary_margin_binding is not None
        ):
            raise DynamicLfm2AdapterError(
                "Target vocabulary-margin binding is unused by this stack."
            )

    def build_bridge(
        self,
        selected_target: Lfm2SelectedTargetLike,
        *,
        catalog_evaluator: _Lfm2CatalogFinalEvaluator | None = None,
        blank_state_fingerprint: str | None = None,
    ) -> Lfm2DynamicCompilerBridge:
        selected_evaluator = catalog_evaluator or _Lfm2CatalogFinalEvaluator(
            selected_target,
            self.final_evaluator,
            self.ledger,
        )
        if selected_evaluator.selected_target != selected_target:
            raise DynamicLfm2AdapterError(
                "Catalog evaluator is bound to a different LFM2 target."
            )
        return build_lfm2_dynamic_compiler_bridge(
            parent_cache_fingerprint=self.parent_cache_fingerprint,
            parent_cache_payload=self.parent_cache_payload,
            evidence_builder=self.evidence_builder,
            selected_target=selected_target,
            final_evaluator=selected_evaluator,
            fold_config=self.fold_config,
            model_evidence_builder=self.model_evidence_builder,
            model_fold_config=self.model_fold_config,
            producer_first_evidence_builder=(
                self.producer_first_evidence_builder
            ),
            producer_first_fold_config=self.producer_first_fold_config,
            blank_state_fingerprint=blank_state_fingerprint,
            parent_cache_integrity_check=(self.parent_cache_integrity_check),
        )


class _Lfm2CatalogFinalEvaluator:
    """Close rejection immediately and acceptance only after fresh publication."""

    def __init__(
        self,
        selected_target: Lfm2SelectedTargetLike,
        evaluator: FoldEvaluator,
        ledger: ExperimentLedger,
    ) -> None:
        self.selected_target = selected_target
        self.evaluator = evaluator
        self.ledger = ledger
        self._pending_acceptance: FoldDecision | None = None
        self._recorded = False
        try:
            expose_behavior_gate_binding(
                self,
                resolve_behavior_gate_binding(evaluator),
            )
        except DynamicBehaviorGateError:
            pass

    def _append_record(
        self,
        *,
        decision: FoldDecision,
        status: RecordStatus,
        reason_class: str,
        extra_metrics: Mapping[str, Any] | None = None,
    ) -> None:
        if self._recorded:
            raise DynamicLfm2AdapterError(
                "Selected LFM2 target already has a catalog closure record."
            )
        if not self.ledger.decision(self.selected_target.signature).allowed:
            raise DynamicLfm2AdapterError(
                "Selected LFM2 target was closed before its catalog record."
            )
        metrics = dict(decision.metrics)
        metrics.update(
            {
                "exact_target_unique_parameters": (
                    self.selected_target.exact_target_unique_parameters
                ),
                "final_behavior_gate_present": True,
            }
        )
        metrics.update(dict(extra_metrics or {}))
        record = LedgerRecord(
            signature=self.selected_target.signature,
            closure_level=ClosureLevel.EXACT,
            status=status,
            reason_class=reason_class,
            reason=decision.reason or status.value,
            metrics=tuple(metrics.items()),
            previous_record_sha256=(
                None
                if not self.ledger.records
                else self.ledger.records[-1].record_sha256
            ),
        )
        self.ledger.append(record)
        self._recorded = True

    def evaluate(self, context: Any) -> FoldDecision:
        execution = context.execution_context
        child_state_before = fingerprint_model_state(execution.child_model)
        child_runtime_before = fingerprint_model_runtime_state(execution.child_model)
        parent_states_before = dict(execution.parent_state_fingerprints)
        parent_runtime_before = {
            name: fingerprint_model_runtime_state(model)
            for name, model in execution.parent_models.items()
        }
        value = self.evaluator.evaluate(context)
        if (
            fingerprint_model_state(execution.child_model) != child_state_before
            or fingerprint_model_runtime_state(execution.child_model)
            != child_runtime_before
            or dict(execution.parent_state_fingerprints) != parent_states_before
            or {
                name: fingerprint_model_runtime_state(model)
                for name, model in execution.parent_models.items()
            }
            != parent_runtime_before
        ):
            raise DynamicLfm2AdapterError(
                "LFM2 final evaluator mutated a parent or child model."
            )
        if isinstance(value, FoldDecision):
            decision = value
        elif isinstance(value, bool):
            decision = FoldDecision(value, "")
        else:
            raise DynamicLfm2AdapterError(
                "LFM2 final evaluator must return FoldDecision or bool."
            )
        if decision.accepted:
            self._pending_acceptance = decision
        else:
            self._append_record(
                decision=decision,
                status=RecordStatus.REJECTED,
                reason_class="whole_output_behavior_reject",
            )
        return decision

    def record_publication(self, result: Any) -> None:
        """Record success only after the standalone child passed fresh reload."""

        pending = self._pending_acceptance
        if pending is None or not pending.accepted:
            raise DynamicLfm2AdapterError(
                "Cannot accept an LFM2 catalog lane without a pending "
                "in-memory behavior acceptance."
            )
        target = getattr(result, "target", None)
        published = getattr(result, "published", None)
        artifact = getattr(published, "artifact", None)
        if (
            not bool(getattr(result, "complete", False))
            or target is None
            or target.adapter_payload != self.selected_target
            or target.resource_profile.unique_parameters
            != self.selected_target.exact_target_unique_parameters
            or artifact is None
            or artifact.exact_unique_parameters
            != self.selected_target.exact_target_unique_parameters
            or not artifact.fresh_reload_decision.accepted
            or artifact.certificate_sha256 != canonical_sha256(artifact.body())
        ):
            raise DynamicLfm2AdapterError(
                "Published LFM2 result does not authenticate the selected "
                "target and fresh-reload behavior."
            )
        certificate_path = Path(artifact.path) / "certificate.json"
        try:
            certificate_payload = json.loads(
                certificate_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise DynamicLfm2AdapterError(
                "Published LFM2 artifact certificate is missing or invalid."
            ) from exc
        if certificate_payload != artifact.as_dict():
            raise DynamicLfm2AdapterError(
                "Published LFM2 on-disk certificate differs from live evidence."
            )
        self._append_record(
            decision=pending,
            status=RecordStatus.ACCEPTED,
            reason_class="whole_output_behavior_and_fresh_reload_accept",
            extra_metrics={
                "fresh_reload_behavior_accepted": True,
                "artifact_certificate_sha256": artifact.certificate_sha256,
            },
        )


class Lfm2DynamicFoldStackAdapter:
    """Catalog adapter that authenticates an LFM target before bridge creation."""

    adapter_id = LFM2_STACK_ADAPTER_ID
    version = 1
    priority = 100
    target_families = ("lfm2",)
    target_modalities = ("text",)
    method_ids = (
        LAYER_METHOD,
        HYBRID_MODEL_METHOD,
        PRODUCER_FIRST_MODEL_METHOD,
        _DENSE_BUDGET_MODEL_METHOD,
    )

    def __init__(self, resources: Lfm2DynamicStackResources) -> None:
        if not isinstance(resources, Lfm2DynamicStackResources):
            raise DynamicLfm2AdapterError(
                "resources must be Lfm2DynamicStackResources."
            )
        self.resources = resources
        self._dense_cache_verification_lease: Lfm2DenseCacheVerificationLease | None = (
            None
        )

    def _dense_parent_cache_lease(
        self,
        trace_cache: Any,
    ) -> Lfm2DenseCacheVerificationLease | None:
        identity_check = self.resources.parent_cache_integrity_check
        if identity_check is None:
            return None
        try:
            from .hf_lfm2_dense_budget_model_fold import (
                Lfm2DenseCacheVerificationLease,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DynamicLfm2AdapterError(
                "The dense-budget LFM2 cache verification lease is unavailable."
            ) from exc
        current = self._dense_cache_verification_lease
        if current is not None:
            if (
                current.trace_cache is not trace_cache
                or current.cache_key != self.resources.parent_cache_fingerprint
            ):
                raise DynamicLfm2AdapterError(
                    "Dense cache verification lease is bound to a different "
                    "cache handle or fingerprint."
                )
            return current

        payload = self.resources.parent_cache_payload

        def recheck_identity() -> str:
            return identity_check(payload)

        lease = Lfm2DenseCacheVerificationLease(
            trace_cache,
            self.resources.parent_cache_fingerprint,
            identity_check=recheck_identity,
        )
        self._dense_cache_verification_lease = lease
        return lease

    def _selected_target(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> Lfm2SelectedTargetLike | Lfm2DenseBudgetedSelectedTarget:
        if not isinstance(target, DynamicTargetSynthesisResult):
            raise DynamicLfm2AdapterError(
                "LFM2 stack selection requires DynamicTargetSynthesisResult."
            )
        if (
            target.adapter_id != Lfm2TargetSynthesisAdapter.adapter_id
            or target.adapter_version != Lfm2TargetSynthesisAdapter.version
        ):
            raise DynamicLfm2AdapterError(
                "Synthesized target did not come from the LFM2 target adapter."
            )
        selected = target.adapter_payload
        if not isinstance(
            selected,
            (
                Lfm2SelectedTarget,
                Lfm2HybridSelectedTarget,
                Lfm2DenseBudgetedSelectedTarget,
            ),
        ):
            raise DynamicLfm2AdapterError(
                "LFM2 synthesized target lacks its selected-target payload."
            )
        if isinstance(selected, Lfm2SelectedTarget):
            selected.validate()
        return selected

    def _dense_evidence_set(self) -> Any | None:
        """The ordered ``(base, *on-policy extensions)`` set, if the payload has one.

        A payload without the attribute, or one whose set carries no extension
        member, resolves to ``None`` and every dense binding below is exactly
        the single sealed handle this stack has always used.
        """

        evidence_set = getattr(
            self.resources.parent_cache_payload,
            "evidence_set",
            None,
        )
        if evidence_set is None or not getattr(
            evidence_set,
            "has_extension_evidence",
            False,
        ):
            return None
        return evidence_set

    def _dense_parent_trace_cache(self, *, verify: bool) -> Any:
        try:
            from .hf_lfm2_circuit_trace_cache import Lfm2CircuitTraceCache
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DynamicLfm2AdapterError(
                "The dense-budget LFM2 trace-cache primitive is unavailable."
            ) from exc
        payload = self.resources.parent_cache_payload
        if isinstance(payload, Lfm2CircuitTraceCache):
            trace_cache = payload
        else:
            trace_cache = getattr(payload, "cache", None)
            if not isinstance(trace_cache, Lfm2CircuitTraceCache):
                raise DynamicLfm2AdapterError(
                    "Dense LFM2 target requires a typed sealed parent trace "
                    "cache or a verified production payload containing one."
                )
        evidence_set = self._dense_evidence_set()
        if evidence_set is not None:
            # The stack evidence fingerprint is the merged evidence identity;
            # the handle itself is still the sealed base cache, and the dense
            # writers resolve the merged read-only view from the evidence set.
            if evidence_set.base_cache is not trace_cache:
                raise DynamicLfm2AdapterError(
                    "The sealed evidence set is bound to a different base "
                    "parent trace cache than the payload."
                )
            trace_cache = evidence_set.merged_cache()
        payload_verify = getattr(payload, "verify", None)
        if (
            not callable(payload_verify)
            or getattr(payload, "cache_key", None)
            != self.resources.parent_cache_fingerprint
            or trace_cache.cache_key
            != self.resources.parent_cache_fingerprint
        ):
            raise DynamicLfm2AdapterError(
                "Dense LFM2 target requires the exact typed sealed parent "
                "trace cache bound to the stack evidence fingerprint."
            )
        if verify:
            verification_lease = self._dense_parent_cache_lease(trace_cache)
            if verification_lease is None:
                payload_verify()
                trace_cache.verify()
            else:
                verification_lease.verify(trace_cache)
        return trace_cache

    def _dense_tokenizer_bridge(
        self,
        selected: Lfm2DenseBudgetedSelectedTarget,
        *,
        verify: bool,
    ) -> StrictTokenizerBridgeArtifact | None:
        artifact = self.resources.tokenizer_bridge_artifact
        if not selected.tokenizer_bridge_required:
            if artifact is not None:
                raise DynamicLfm2AdapterError(
                    "Unbridged dense target received an unused tokenizer "
                    "bridge artifact."
                )
            return None
        if not isinstance(artifact, StrictTokenizerBridgeArtifact):
            raise DynamicLfm2AdapterError(
                "Bridged dense target requires its explicit strict tokenizer "
                "bridge artifact in stack resources."
            )
        if verify:
            try:
                artifact.verify_files()
            except ValueError as exc:
                raise DynamicLfm2AdapterError(
                    "Dense tokenizer bridge artifact failed stack verification."
                ) from exc
        if (
            artifact.artifact_sha256
            != selected.tokenizer_bridge_artifact_sha256
            or artifact.profile_fingerprint
            != selected.tokenizer_transport_profile_sha256
            or artifact.donor_tokenizer_semantics_sha256
            != selected.parent_tokenizer_semantics_sha256
            or artifact.target_tokenizer_semantics_sha256
            != selected.child_tokenizer_semantics_sha256
            or artifact.transport_profile.target_row_capacity
            != selected.vocabulary_size
        ):
            raise DynamicLfm2AdapterError(
                "Dense tokenizer bridge artifact differs from selected target "
                "metadata."
            )
        return artifact

    def _dense_tokenizer_bridge_trace(
        self,
        selected: Lfm2DenseBudgetedSelectedTarget,
        *,
        verify: bool,
    ) -> Lfm2StrictTokenizerBridgeTraceAdapter | None:
        trace = self.resources.tokenizer_bridge_trace_adapter
        if not selected.tokenizer_bridge_required:
            if trace is not None:
                raise DynamicLfm2AdapterError(
                    "Unbridged dense target received unused tokenizer bridge "
                    "trace evidence."
                )
            return None
        run_closed_loop = bool(
            getattr(self.resources.dense_fold_config, "run_closed_loop", True)
        )
        target_vocabulary_margin = bool(
            getattr(
                self.resources.dense_fold_config,
                "vocabulary_margin_observability",
                False,
            )
        )
        if trace is None:
            if run_closed_loop or target_vocabulary_margin:
                raise DynamicLfm2AdapterError(
                    "Bridged dense closed-loop or target vocabulary-margin "
                    "folding requires its sealed target-token trace adapter."
                )
            return None
        if not isinstance(trace, Lfm2StrictTokenizerBridgeTraceAdapter):
            raise DynamicLfm2AdapterError(
                "Dense tokenizer bridge trace resource has the wrong type."
            )
        if verify:
            try:
                trace.verify()
            except Lfm2StrictTokenizerBridgeTraceError as exc:
                raise DynamicLfm2AdapterError(
                    "Dense tokenizer bridge trace resource failed verification."
                ) from exc
        artifact = self._dense_tokenizer_bridge(selected, verify=verify)
        cache = self._dense_parent_trace_cache(verify=verify)
        ordered_cache_inputs = tuple(
            str(value)
            for value in cache.manifest["corpus"]["ordered_batch_input_sha256"]
        )
        if (
            artifact is None
            or trace.bridge_artifact_sha256 != artifact.artifact_sha256
            or trace.transport_profile_fingerprint != artifact.profile_fingerprint
            or trace.target_row_capacity != selected.vocabulary_size
            or trace.record_count != len(cache)
            or any(
                record.ordinal != index
                or record.donor_cache_input_sha256 != input_sha256
                for index, (record, input_sha256) in enumerate(
                    zip(trace.records, ordered_cache_inputs, strict=True)
                )
            )
        ):
            raise DynamicLfm2AdapterError(
                "Dense tokenizer bridge trace differs from selected target or "
                "sealed parent cache."
            )
        return trace

    def _validate_target(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> Lfm2SelectedTargetLike | Lfm2DenseBudgetedSelectedTarget:
        selected = self._selected_target(target)
        request = target.request
        profile = target.resource_profile
        if (
            profile.family != "lfm2"
            or profile.modalities != ("text",)
            or target.graph.family != "lfm2"
            or target.graph.model_name != request.child_name
        ):
            raise DynamicLfm2AdapterError(
                "Synthesized target has the wrong LFM2 family/modality identity."
            )
        if (
            request.exact_unique_parameters != selected.exact_target_unique_parameters
            or profile.unique_parameters != selected.exact_target_unique_parameters
            or target.graph.unique_parameter_count
            != selected.exact_target_unique_parameters
        ):
            raise DynamicLfm2AdapterError(
                "Synthesized target budget differs from its selected LFM2 stage."
            )
        if (
            request.tokenizer_fingerprint != selected.tokenizer_fingerprint
            or target.graph.tokenizer_fingerprint != selected.tokenizer_fingerprint
        ):
            raise DynamicLfm2AdapterError(
                "Synthesized target tokenizer differs from its selected stage."
            )
        parent_graphs = dict(target.parent_graph_fingerprints)
        if isinstance(selected, Lfm2DenseBudgetedSelectedTarget):
            expected_parent_graphs = {
                selected.source_parent: selected.source_graph_fingerprint
            }
        else:
            expected_parent_graphs = {
                selected.parent_name: (
                    selected.signature.parent_graph_fingerprints[0]
                )
            }
        if parent_graphs != expected_parent_graphs:
            raise DynamicLfm2AdapterError(
                "Synthesized target parent graph bindings differ from "
                "the single selected LFM2 source."
            )
        if isinstance(selected, Lfm2DenseBudgetedSelectedTarget):
            self._dense_parent_trace_cache(verify=True)
            self._dense_tokenizer_bridge(selected, verify=True)
            self._dense_tokenizer_bridge_trace(selected, verify=True)
        else:
            if (
                selected.signature.evidence_fingerprint
                != self.resources.parent_cache_fingerprint
            ):
                raise DynamicLfm2AdapterError(
                    "Selected LFM2 target is not bound to this cached evidence."
                )
            if not self.resources.ledger.decision(selected.signature).allowed:
                raise DynamicLfm2AdapterError(
                    "Selected LFM2 target is already closed in the catalog ledger."
                )
        if isinstance(selected, Lfm2SelectedTarget):
            validate_lfm2_selected_child(
                selected,
                target.model,
                target.graph,
            )
            producer_first = (
                target.request.fold_method_id == PRODUCER_FIRST_MODEL_METHOD
            )
            if producer_first:
                if (
                    dict(target.graph.metadata).get("fold_method_id")
                    != PRODUCER_FIRST_MODEL_METHOD
                    or self.resources.producer_first_evidence_builder is None
                    or self.resources.producer_first_fold_config is None
                ):
                    raise DynamicLfm2AdapterError(
                        "Producer-first LFM2 target lacks its authenticated "
                        "full-depth method marker or runtime resources."
                    )
            elif dict(target.graph.metadata).get("fold_method_id") == (
                PRODUCER_FIRST_MODEL_METHOD
            ):
                raise DynamicLfm2AdapterError(
                    "An unrequested LFM2 target carries a producer-first marker."
                )
        elif isinstance(selected, Lfm2HybridSelectedTarget):
            validate_lfm2_hybrid_child(
                selected,
                target.model,
                target.graph,
            )
            if target.request.fold_method_id != HYBRID_MODEL_METHOD:
                raise DynamicLfm2AdapterError(
                    "Hybrid LFM2 target did not explicitly request its "
                    "model-level fold method."
                )
            if self.resources.model_evidence_builder is None:
                raise DynamicLfm2AdapterError(
                    "Hybrid LFM2 target requires model_evidence_builder."
                )
        else:
            validate_lfm2_dense_budgeted_child(
                selected,
                target.model,
                target.graph,
            )
            source_depth = sum(
                len(span) for span in selected.ordered_source_spans
            )
            if selected.target_depth != source_depth or (
                selected.ordered_source_spans
                != tuple((index,) for index in range(source_depth))
            ):
                raise DynamicLfm2AdapterError(
                    "Ordered source-span composition is required for this "
                    "depth-changing dense target; the current executable LFM2 "
                    "lane supports width/GQA/FFN compaction at equal depth and "
                    "fails closed for 4x/10x depth reduction."
                )
            if target.request.fold_method_id is not None:
                raise DynamicLfm2AdapterError(
                    "Dense budget target selection owns its executable method "
                    "dynamically and must not carry a caller-forced legacy marker."
                )
        config = getattr(target.model, "config", None)
        to_dict = getattr(config, "to_dict", None)
        if not callable(to_dict):
            raise DynamicLfm2AdapterError(
                "Synthesized LFM2 target lacks a serializable local config."
            )
        config_payload = to_dict()
        if not isinstance(config_payload, Mapping):
            raise DynamicLfm2AdapterError(
                "Synthesized LFM2 config serialization is not a mapping."
            )
        expected_adapter_topology_fingerprint = canonical_sha256(
            {
                "adapter": Lfm2TargetSynthesisAdapter.adapter_id,
                "selected_target": selected.as_dict(),
                "materialized_config": canonical_sha256(dict(config_payload)),
            }
        )
        report = dict(target.adapter_report)
        blank_state = fingerprint_model_state(target.model)
        if isinstance(selected, Lfm2SelectedTarget):
            expected_report = {
                "source_parent": selected.parent_name,
                **(
                    {"fold_method_id": PRODUCER_FIRST_MODEL_METHOD}
                    if target.request.fold_method_id
                    == PRODUCER_FIRST_MODEL_METHOD
                    else {}
                ),
                "selected_stage_index": selected.stage.stage_index,
                "selected_target_plan": selected.plan.fingerprint,
                "blank_state_fingerprint": blank_state,
                "local_parent_class_instantiation": True,
                "hub_loader_used": False,
                "adapter_reported_topology_fingerprint": (
                    expected_adapter_topology_fingerprint
                ),
            }
        elif isinstance(selected, Lfm2HybridSelectedTarget):
            expected_report = {
                "source_parent": selected.parent_name,
                "fold_method_id": HYBRID_MODEL_METHOD,
                "target_depth": selected.target_depth,
                "target_layer_types": encode_lfm2_layer_types(
                    selected.target_layer_types
                ),
                "ordered_source_spans": encode_lfm2_source_spans(
                    selected.ordered_source_spans
                ),
                "blank_state_fingerprint": blank_state,
                "local_parent_class_instantiation": True,
                "hub_loader_used": False,
                "adapter_reported_topology_fingerprint": (
                    expected_adapter_topology_fingerprint
                ),
            }
        else:
            expected_report = {
                "source_parent": selected.source_parent,
                "dense_budget_catalog": selected.catalog_id,
                "component_allocation_policy": (selected.component_allocation_policy),
                "component_allocation_receipt_sha256": (
                    selected.component_allocation_receipt_sha256
                ),
                "component_allocation_topology_sha256": (
                    selected.component_allocation_topology_sha256
                ),
                "component_allocation_verified_by_target_adapter": True,
                "target_depth": selected.target_depth,
                "target_layer_types": encode_lfm2_layer_types(
                    selected.layer_types
                ),
                "ordered_source_spans": encode_lfm2_source_spans(
                    selected.ordered_source_spans
                ),
                "target_hidden_size": selected.hidden_size,
                "target_intermediate_size": selected.intermediate_size,
                "target_query_heads": selected.query_heads,
                "target_key_value_heads": selected.key_value_heads,
                "target_head_dim": selected.head_dim,
                "target_logical_parameters": (
                    selected.exact_target_logical_parameters
                ),
                "target_unique_parameters": (
                    selected.exact_target_unique_parameters
                ),
                "native_token_interface_tied": (
                    selected.tie_word_embeddings
                ),
                **(
                    {
                        "physical_layout_kind": LFM2_HETEROGENEOUS_LAYOUT_KIND,
                        "physical_layout_descriptor_sha256": (
                            selected.heterogeneous_layout_descriptor.descriptor_sha256
                        ),
                        "physical_layout_selection_receipt_sha256": (
                            selected.heterogeneous_layout_selection_receipt.receipt_sha256
                        ),
                        "heterogeneous_per_layer_widths": True,
                        "fp32_physical_parameters": True,
                    }
                    if selected.heterogeneous_layout
                    else {}
                ),
                **(
                    {
                        "tokenizer_bridge_required": True,
                        "parent_tokenizer_semantics_sha256": (
                            selected.parent_tokenizer_semantics_sha256
                        ),
                        "child_tokenizer_semantics_sha256": (
                            selected.child_tokenizer_semantics_sha256
                        ),
                        "tokenizer_bridge_artifact_sha256": (
                            selected.tokenizer_bridge_artifact_sha256
                        ),
                        "tokenizer_transport_profile_sha256": (
                            selected.tokenizer_transport_profile_sha256
                        ),
                    }
                    if selected.tokenizer_bridge_required
                    else {}
                ),
                "blank_state_fingerprint": blank_state,
                "local_parent_class_instantiation": True,
                "hub_loader_used": False,
                "dense_parameters": True,
                "quantization_used": False,
                "pruning_used": False,
                "training_used": False,
                "adapter_reported_topology_fingerprint": (
                    expected_adapter_topology_fingerprint
                ),
            }
        if report != expected_report:
            raise DynamicLfm2AdapterError(
                "Synthesized LFM2 adapter report differs from live target state."
            )
        try:
            verify_compiler_authenticated_target(target)
        except DynamicTargetSynthesisError as exc:
            raise DynamicLfm2AdapterError(
                "Synthesized LFM2 topology fingerprint is inconsistent."
            ) from exc
        return selected

    def supports(self, target: DynamicTargetSynthesisResult) -> bool:
        try:
            selected = self._selected_target(target)
        except (DynamicLfm2AdapterError, TypeError, ValueError):
            return False
        if isinstance(selected, Lfm2DenseBudgetedSelectedTarget):
            try:
                self._dense_parent_trace_cache(verify=False)
                self._dense_tokenizer_bridge(selected, verify=False)
                self._dense_tokenizer_bridge_trace(selected, verify=False)
            except DynamicLfm2AdapterError:
                return False
            return bool(
                target.resource_profile.family == "lfm2"
                and target.resource_profile.modalities == ("text",)
                and dict(target.parent_graph_fingerprints)
                == {
                    selected.source_parent: (
                        selected.source_graph_fingerprint
                    )
                }
            )
        return bool(
            target.resource_profile.family == "lfm2"
            and target.resource_profile.modalities == ("text",)
            and dict(target.parent_graph_fingerprints)
            == {selected.parent_name: (selected.signature.parent_graph_fingerprints[0])}
            and selected.signature.evidence_fingerprint
            == self.resources.parent_cache_fingerprint
        )

    def build(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> DynamicFoldSystemStack:
        selected = self._validate_target(target)
        if isinstance(selected, Lfm2DenseBudgetedSelectedTarget):
            try:
                from .hf_lfm2_dense_budget_model_fold import (
                    Lfm2DenseBudgetAdmissionEvaluator,
                    Lfm2DenseBudgetEvidenceProvider,
                    Lfm2DenseBudgetFoldConfig,
                    Lfm2DenseBudgetFoldOnlyInitializer,
                    Lfm2DenseBudgetModelOperator,
                )
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise DynamicLfm2AdapterError(
                    "The dense-budget LFM2 fold primitive is unavailable."
                ) from exc
            dense_config = self.resources.dense_fold_config
            if dense_config is None:
                dense_config = Lfm2DenseBudgetFoldConfig()
            cache = self._dense_parent_trace_cache(verify=False)
            verification_lease = self._dense_parent_cache_lease(cache)
            evidence_set = self._dense_evidence_set()
            selective_session = getattr(
                self.resources.parent_cache_payload,
                "selective_session",
                None,
            )
            if evidence_set is not None:
                # The merged session is the one the evidence set already holds
                # open across every member; it opens nothing new.
                selective_session = evidence_set.merged_session()
            provider = Lfm2DenseBudgetEvidenceProvider(
                cache,
                self.resources.parent_cache_fingerprint,
                verification_lease=verification_lease,
                evidence_set=evidence_set,
            )
            operator = Lfm2DenseBudgetModelOperator(
                selected,
                cache,
                config=dense_config,
                verification_lease=verification_lease,
                selective_session=selective_session,
                evidence_set=evidence_set,
                tokenizer_bridge_artifact=(
                    self._dense_tokenizer_bridge(
                        selected,
                        verify=True,
                    )
                ),
                tokenizer_bridge_trace_adapter=(
                    self._dense_tokenizer_bridge_trace(
                        selected,
                        verify=True,
                    )
                ),
                target_tokenizer_vocabulary_margin_binding=(
                    self.resources.target_tokenizer_vocabulary_margin_binding
                ),
                residual_basis_phase_cache_context=(
                    self.resources.residual_basis_phase_cache_context
                ),
                autoregressive_hankel_observability=(
                    self.resources.autoregressive_hankel_observability
                ),
                progress_sink=self.resources.dense_fold_progress_sink,
            )
            try:
                behavior_binding = resolve_behavior_gate_binding(
                    self.resources.final_evaluator
                )
            except DynamicBehaviorGateError:
                behavior_binding = None
            return DynamicFoldSystemStack(
                operators=FoldOperatorRegistry((operator,)),
                evidence_provider=provider,
                evaluator=Lfm2DenseBudgetAdmissionEvaluator(),
                final_evaluator=self.resources.final_evaluator,
                model_saver=self.resources.model_saver,
                model_loader=self.resources.model_loader,
                fresh_reload_evaluator=(
                    self.resources.fresh_reload_evaluator
                ),
                saver_id=self.resources.saver_id,
                loader_id=self.resources.loader_id,
                behavior_gate_id=(
                    "neural_forge.lfm2.cached_whole_output_behavior"
                    if behavior_binding is None
                    else behavior_binding.gate_id
                ),
                behavior_gate_version=(
                    1
                    if behavior_binding is None
                    else behavior_binding.gate_version
                ),
                behavior_baseline_fingerprint=(
                    getattr(
                        self.resources.final_evaluator,
                        "baseline_fingerprint",
                        self.resources.parent_cache_fingerprint,
                    )
                    if behavior_binding is None
                    else behavior_binding.baseline_fingerprint
                ),
                initializer=Lfm2DenseBudgetFoldOnlyInitializer(
                    self.resources.parent_cache_fingerprint,
                    selected,
                ),
                snapshot_config=self.resources.snapshot_config,
                evidence_fingerprint=(
                    self.resources.parent_cache_fingerprint
                ),
                publication_recorder=None,
            )
        catalog_evaluator = _Lfm2CatalogFinalEvaluator(
            selected,
            self.resources.final_evaluator,
            self.resources.ledger,
        )
        bridge = self.resources.build_bridge(
            selected,
            catalog_evaluator=catalog_evaluator,
            blank_state_fingerprint=(
                target.blank_state_fingerprint
                if (
                    isinstance(selected, Lfm2HybridSelectedTarget)
                    or target.request.fold_method_id
                    == PRODUCER_FIRST_MODEL_METHOD
                )
                else None
            ),
        )
        if (
            bridge.selected_target != selected
            or bridge.evidence_fingerprint != self.resources.parent_cache_fingerprint
        ):
            raise DynamicLfm2AdapterError(
                "Constructed LFM2 bridge differs from catalog resources."
            )
        try:
            behavior_binding = resolve_behavior_gate_binding(
                self.resources.final_evaluator
            )
        except DynamicBehaviorGateError:
            behavior_binding = None
        return DynamicFoldSystemStack(
            operators=bridge.operators,
            evidence_provider=bridge.evidence_provider,
            evaluator=bridge.evaluator,
            final_evaluator=bridge.final_evaluator,
            model_saver=self.resources.model_saver,
            model_loader=self.resources.model_loader,
            fresh_reload_evaluator=(self.resources.fresh_reload_evaluator),
            saver_id=self.resources.saver_id,
            loader_id=self.resources.loader_id,
            behavior_gate_id=(
                "neural_forge.lfm2.cached_whole_output_behavior"
                if behavior_binding is None
                else behavior_binding.gate_id
            ),
            behavior_gate_version=(
                1 if behavior_binding is None else behavior_binding.gate_version
            ),
            behavior_baseline_fingerprint=(
                getattr(
                    self.resources.final_evaluator,
                    "baseline_fingerprint",
                    self.resources.parent_cache_fingerprint,
                )
                if behavior_binding is None
                else behavior_binding.baseline_fingerprint
            ),
            initializer=bridge.initializer,
            snapshot_config=self.resources.snapshot_config,
            evidence_fingerprint=self.resources.parent_cache_fingerprint,
            publication_recorder=catalog_evaluator.record_publication,
        )


__all__ = [
    "LFM2_STACK_ADAPTER_ID",
    "Lfm2DynamicFoldStackAdapter",
    "Lfm2DynamicStackResources",
]
