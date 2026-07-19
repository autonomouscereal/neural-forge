from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import (
    DynamicFoldExecutor,
    ExactCopyOperator,
    FoldDecision,
    FoldEvidence,
    FoldExecutionContext,
    FoldOperatorRegistry,
    StaticEvidenceProvider,
    _FoldOnlyTorchGuard,
    fingerprint_fold_evidence_payload,
    fingerprint_model_state,
    fingerprint_model_state_entries,
)
from neural_forge.dynamic_fold_planner import DynamicFoldPlanner
from neural_forge.dynamic_model_inspector import inspect_model_circuits


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@dataclass(frozen=True)
class _EvidenceRecord:
    name: str
    rows: object


class _VerifiedCache:
    def __init__(self, values):
        self.values = list(values)
        self.cache_key = canonical_sha256({"values": self.values})

    def verify(self):
        if canonical_sha256({"values": self.values}) != self.cache_key:
            raise ValueError("cache seal changed")


@dataclass
class _LyingSealedDataclassCache:
    values: list[int]
    cache_key: str

    def verify(self):
        return None


def test_evidence_payload_fingerprint_is_canonical_and_content_sensitive() -> None:
    first = {
        "record": _EvidenceRecord("layer", torch.tensor([[1.0, 2.0]])),
        "scalars": (None, True, 4, 2.5, "trace", b"cache"),
    }
    second = {
        "scalars": (None, True, 4, 2.5, "trace", b"cache"),
        "record": _EvidenceRecord("layer", torch.tensor([[1.0, 2.0]])),
    }

    first_fingerprint = fingerprint_fold_evidence_payload(first)

    assert fingerprint_fold_evidence_payload(second) == first_fingerprint
    first["record"].rows[0, 0] = 3.0
    assert fingerprint_fold_evidence_payload(first) != first_fingerprint


def test_explicit_verified_cache_seal_is_supported_and_reverified() -> None:
    cache = _VerifiedCache([1, 2, 3])
    evidence = FoldEvidence("1" * 64, cache, channels=("rows",))

    assert evidence.payload_fingerprint == fingerprint_fold_evidence_payload(cache)
    assert len(fingerprint_fold_evidence_payload(torch.tensor(7))) == 64
    assert "explicit-cache-verifier-trust" in evidence.payload_integrity_mode
    cache.values.append(4)
    with pytest.raises(Exception, match="cache verification failed"):
        fingerprint_fold_evidence_payload(cache)


def test_dataclass_content_precedes_a_self_asserted_cache_seal() -> None:
    cache = _LyingSealedDataclassCache([1, 2, 3], "a" * 64)
    evidence = FoldEvidence("1" * 64, cache, channels=("rows",))

    assert evidence.payload_integrity_mode == "canonical-content"
    before = evidence.payload_fingerprint
    cache.values.append(4)
    assert fingerprint_fold_evidence_payload(cache) != before


def test_fold_evidence_binding_binds_root_channels_and_payload() -> None:
    payload = {"rows": [1, 2, 3]}
    base = FoldEvidence(
        "2" * 64,
        payload,
        channels=("residual", "attention"),
        root_fingerprint="3" * 64,
    )
    changed_channel = FoldEvidence(
        "2" * 64,
        payload,
        channels=("residual",),
        root_fingerprint="3" * 64,
    )
    changed_root = FoldEvidence(
        "2" * 64,
        payload,
        channels=("residual", "attention"),
        root_fingerprint="4" * 64,
    )

    assert base.payload_fingerprint == changed_channel.payload_fingerprint
    assert base.binding_fingerprint != changed_channel.binding_fingerprint
    assert base.binding_fingerprint != changed_root.binding_fingerprint


def test_state_fingerprints_support_scalar_registry_buffers() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    model.register_buffer(
        "_registry_schema_version",
        torch.tensor(1, dtype=torch.int64),
    )

    before = fingerprint_model_state(model)
    entries = fingerprint_model_state_entries(model)

    assert set(entries) == {
        "_registry_schema_version",
        "weight",
    }
    model._registry_schema_version.fill_(2)
    assert fingerprint_model_state(model) != before


def _llama(seed: int):
    torch.manual_seed(seed)
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=True,
        )
    )


class _Accept:
    def evaluate(self, context):
        return FoldDecision.accept(
            "exact copy is algebraically valid",
            {"candidate_bound": context.candidate_state_fingerprint is not None},
        )


class _RejectFirst:
    def __init__(self):
        self.calls = 0

    def evaluate(self, _context):
        self.calls += 1
        return FoldDecision.reject("intentional test rejection")


class _AcceptThenReject:
    def __init__(self):
        self.calls = 0

    def evaluate(self, _context):
        self.calls += 1
        if self.calls == 1:
            return FoldDecision.accept("first transaction accepted")
        return FoldDecision.reject("second transaction rejected")


class _NoOpTransaction:
    method_id = "copy_exact"
    active = True

    def __init__(self, state):
        self.candidate_state_fingerprint = state

    def commit(self):
        self.active = False
        return {"accepted": True}

    def rollback(self):
        self.active = False
        return {"accepted": False, "rollback_exact": True}


class _NoOpOperator:
    method_id = "copy_exact"

    def prepare(self, _action, context, _evidence):
        return _NoOpTransaction(context.child_state_fingerprint)


class _MaliciousCommitTransaction:
    method_id = "copy_exact"

    def __init__(self, inner):
        self.inner = inner
        self.candidate_state_fingerprint = inner.candidate_state_fingerprint

    @property
    def active(self):
        return self.inner.active

    def commit(self):
        value = torch.tensor(2.0, requires_grad=True)
        value.backward()
        return self.inner.commit()

    def rollback(self):
        return self.inner.rollback()


class _MaliciousRollbackTransaction:
    method_id = "copy_exact"

    def __init__(self, inner):
        self.inner = inner
        self.candidate_state_fingerprint = inner.candidate_state_fingerprint

    @property
    def active(self):
        return self.inner.active

    def commit(self):
        return self.inner.commit()

    def rollback(self):
        value = torch.tensor(2.0, requires_grad=True)
        value.backward()
        return self.inner.rollback()


class _ReturnedTransactionOperator:
    method_id = "copy_exact"

    def __init__(self, wrapper):
        self.inner = ExactCopyOperator()
        self.wrapper = wrapper

    def prepare(self, action, context, evidence):
        return self.wrapper(self.inner.prepare(action, context, evidence))


class _MutatingAccept:
    def evaluate(self, context):
        with torch.no_grad():
            next(context.execution_context.child_model.parameters()).add_(1.0)
        return FoldDecision.accept("invalid mutating evaluator")


class _ParentMutatingEvidence(StaticEvidenceProvider):
    def prepare(self, action, context):
        with torch.no_grad():
            next(context.parent_models["parent"].parameters()).add_(1.0)
        return super().prepare(action, context)


class _ParentForwardingEvidence(StaticEvidenceProvider):
    def prepare(self, action, context):
        parent = context.parent_models["parent"]
        device = next(parent.parameters()).device
        parent(torch.tensor([[1, 2]], device=device), use_cache=False)
        return super().prepare(action, context)


class _ForwardingAccept:
    def evaluate(self, context):
        child = context.execution_context.child_model
        device = next(child.parameters()).device
        child(torch.tensor([[1, 2]], device=device), use_cache=False)
        return FoldDecision.accept("observed whole-output model execution")


class _ChildMutatingRefresh(StaticEvidenceProvider):
    def refresh(self, action, context, transaction_report):
        super().refresh(action, context, transaction_report)
        with torch.no_grad():
            next(context.child_model.parameters()).add_(0.01)


class _ChildMutatingRestore(StaticEvidenceProvider):
    def restore(self, context, state_fingerprint, action_ids):
        super().restore(context, state_fingerprint, action_ids)
        with torch.no_grad():
            next(context.child_model.parameters()).add_(0.5)


class _GradientAfterPrepare:
    method_id = "copy_exact"

    def __init__(self):
        self.inner = ExactCopyOperator()

    def prepare(self, action, context, evidence):
        transaction = self.inner.prepare(action, context, evidence)
        parameter = next(context.child_model.parameters())
        parameter.grad = torch.zeros_like(parameter)
        return transaction


class _EvidenceMutatingOperator:
    method_id = "copy_exact"

    def __init__(self):
        self.inner = ExactCopyOperator()

    def prepare(self, action, context, evidence):
        evidence.payload["rows"].add_(1.0)
        return self.inner.prepare(action, context, evidence)


class _EvidenceMutatingEvaluator:
    def evaluate(self, context):
        context.evidence.payload["rows"].add_(1.0)
        return FoldDecision.accept("invalid evidence mutation")


class _BackwardAttemptOperator:
    method_id = "copy_exact"

    def prepare(self, _action, _context, _evidence):
        value = torch.tensor(2.0, requires_grad=True)
        (value * value).backward()
        raise AssertionError("backward guard did not fire")


class _EnableGradAttemptOperator:
    method_id = "copy_exact"

    def prepare(self, _action, _context, _evidence):
        with torch.enable_grad():
            raise AssertionError("enable-grad guard did not fire")


class _MutatingInitializer:
    def initialize(self, _plan, context):
        with torch.no_grad():
            next(context.child_model.parameters()).add_(0.25)
        return {
            "child_state_fingerprint": context.child_state_fingerprint,
            "complete_target_coverage": False,
        }


class _MalformedEvidence(StaticEvidenceProvider):
    def prepare(self, _action, _context):
        return object()


class _NonPersistentBufferMutatingRefresh(StaticEvidenceProvider):
    def refresh(self, action, context, transaction_report):
        super().refresh(action, context, transaction_report)
        child_buffer = next(context.child_model.named_buffers())[1]
        parent_buffer = next(context.parent_models["parent"].named_buffers())[1]
        with torch.no_grad():
            child_buffer.add_(0.125)
            parent_buffer.add_(0.25)


class _HookMutatingEvidence(StaticEvidenceProvider):
    def prepare(self, action, context):
        context.child_model.lm_head.register_forward_hook(
            lambda _module, _inputs, output: torch.zeros_like(output)
        )
        return super().prepare(action, context)


class _HookMutatingAccept:
    def evaluate(self, context):
        context.execution_context.child_model.lm_head.register_forward_hook(
            lambda _module, _inputs, output: torch.zeros_like(output)
        )
        return FoldDecision.accept("invalid hook-mutating evaluator")


class _MissingChannelEvidence(StaticEvidenceProvider):
    def prepare(self, _action, _context):
        return FoldEvidence(self.fingerprint, self.payload, channels=())


class _PreboundBackwardAliasAccept:
    def __init__(self, child):
        self.backward = torch.autograd.backward
        self.loss = next(child.parameters()).sum()

    def evaluate(self, _context):
        self.backward(self.loss, retain_graph=True)
        return FoldDecision.accept("pre-bound alias executed")


def _fixture():
    parent = _llama(1)
    child = _llama(2)
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child_graph = inspect_model_circuits(
        child,
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )
    state = fingerprint_model_state(child)
    evidence = canonical_sha256({"parent_cache": "immutable"})
    plan = DynamicFoldPlanner().plan(
        (parent_graph,),
        child_graph,
        child_state_fingerprint=state,
        evidence_fingerprint=evidence,
        parent_state_fingerprints={"parent": fingerprint_model_state(parent)},
    )
    context = FoldExecutionContext(
        parent_models={"parent": parent},
        child_model=child,
        parent_graphs={"parent": parent_graph},
        child_graph=child_graph,
    )
    return parent, child, plan, context, evidence


def test_executor_copies_complete_model_without_gradients_or_manual_surgery() -> None:
    parent, child, plan, context, evidence = _fixture()
    provider = StaticEvidenceProvider(evidence)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=provider,
        evaluator=_Accept(),
        final_evaluator=_Accept(),
    )

    run = executor.execute(plan, context)

    assert run.complete
    assert len(run.accepted_action_ids) == len(plan.actions)
    assert provider.refresh_count == len(plan.actions)
    assert run.completed_target_parameter_paths == tuple(
        sorted(context.child_graph.parameter_paths)
    )
    assert fingerprint_model_state(parent) == fingerprint_model_state(child)
    assert all(parameter.grad is None for parameter in child.parameters())
    assert run.as_dict()["optimizer_steps"] == 0
    assert run.as_dict()["backward_calls"] == 0
    assert run.whole_model_snapshot_backend == "memory"


def test_attempt_and_run_bind_evidence_payload_derivation_provenance() -> None:
    _parent, _child, plan, context, evidence = _fixture()
    payload = {"rows": torch.tensor([[1.0, 2.0]])}
    run = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence, payload),
        evaluator=_Accept(),
        final_evaluator=_Accept(),
    ).execute(plan, context)

    attempt = run.attempts[0]
    manifest = dict(attempt.evidence_derivation_manifest)
    assert attempt.evidence_payload_fingerprint == (
        fingerprint_fold_evidence_payload(payload)
    )
    assert attempt.evidence_payload_integrity_reverified is True
    assert attempt.evidence_channels == plan.actions[0].evidence_requirements
    assert manifest["action_sha256"] == plan.actions[0].action_sha256
    assert manifest["child_state_fingerprint"] == (attempt.incumbent_state_fingerprint)
    assert manifest["evidence_root_fingerprint"] == plan.evidence_fingerprint
    assert manifest["evidence_payload_fingerprint"] == (
        attempt.evidence_payload_fingerprint
    )
    assert attempt.evidence_derivation_fingerprint == canonical_sha256(manifest)
    assert run.final_evidence_payload_fingerprint == (
        fingerprint_fold_evidence_payload(payload)
    )
    assert run.final_evidence_payload_integrity_reverified is True
    assert run.final_evidence_derivation_manifest is not None
    assert attempt.operator_prepare_implementation_fingerprint != "0" * 64
    assert attempt.operator_prepare_static_zero_training_audit_passed is True
    assert attempt.transaction_implementation_fingerprint != "0" * 64
    assert attempt.transaction_commit_implementation_fingerprint != "0" * 64
    assert attempt.transaction_rollback_implementation_fingerprint != "0" * 64
    assert attempt.transaction_static_zero_training_audit_passed is True
    assert run.executed_fold_implementation_fingerprint != "0" * 64
    assert run.executed_fold_static_zero_training_audit_passed is True
    assert (
        run.as_dict()[
            "generic_evidence_root_authenticity_requires_registered_provider_trust"
        ]
        is True
    )


def test_instance_monkeypatch_cannot_replace_registered_prepare() -> None:
    _parent, child, plan, context, evidence = _fixture()
    child_before = fingerprint_model_state(child)
    operator = ExactCopyOperator()
    registry = FoldOperatorRegistry((operator,))

    def malicious_prepare(_self, _action, _context, _evidence):
        value = torch.tensor(2.0, requires_grad=True)
        value.backward()

    operator.prepare = malicious_prepare.__get__(operator, type(operator))
    executor = DynamicFoldExecutor(
        operators=registry,
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
    )

    with pytest.raises(Exception, match="instance-level prepare override"):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == child_before


@pytest.mark.parametrize(
    ("wrapper", "method_name"),
    (
        (_MaliciousCommitTransaction, "commit"),
        (_MaliciousRollbackTransaction, "rollback"),
    ),
)
def test_returned_transaction_methods_must_pass_static_audit(
    wrapper,
    method_name,
) -> None:
    _parent, child, plan, context, evidence = _fixture()
    child_before = fingerprint_model_state(child)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((_ReturnedTransactionOperator(wrapper),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
    )

    with pytest.raises(
        Exception,
        match=rf"transaction {method_name} failed the static zero-training audit",
    ):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == child_before


def test_operator_cannot_mutate_bound_evidence_payload() -> None:
    _parent, child, plan, context, evidence = _fixture()
    child_before = fingerprint_model_state(child)
    provider = StaticEvidenceProvider(
        evidence,
        {"rows": torch.tensor([[1.0, 2.0]])},
    )
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((_EvidenceMutatingOperator(),)),
        evidence_provider=provider,
        evaluator=_Accept(),
    )

    with pytest.raises(Exception, match="mutated the bound evidence payload"):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == child_before


def test_evaluator_cannot_mutate_bound_evidence_payload() -> None:
    _parent, child, plan, context, evidence = _fixture()
    child_before = fingerprint_model_state(child)
    provider = StaticEvidenceProvider(
        evidence,
        {"rows": torch.tensor([[1.0, 2.0]])},
    )
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=provider,
        evaluator=_EvidenceMutatingEvaluator(),
    )

    with pytest.raises(Exception, match="mutated the bound evidence payload"):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == child_before


def test_executor_observes_parent_and_final_child_forward_calls() -> None:
    _parent, _child, plan, context, evidence = _fixture()
    run = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=_ParentForwardingEvidence(evidence),
        evaluator=_Accept(),
        final_evaluator=_ForwardingAccept(),
    ).execute(plan, context)

    assert run.complete
    assert run.observed_parent_module_forward_calls > 0
    assert run.observed_parent_root_forward_calls > 0
    assert run.observed_final_child_module_forward_calls > 0
    assert run.observed_final_child_root_forward_calls > 0


def test_rejected_transaction_rolls_back_exactly_and_does_not_refresh() -> None:
    _parent, child, plan, context, evidence = _fixture()
    before = fingerprint_model_state(child)
    parameter_ids = tuple(id(value) for value in child.parameters())
    provider = StaticEvidenceProvider(evidence)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=provider,
        evaluator=_RejectFirst(),
    )

    run = executor.execute(plan, context)

    assert run.status == "rejected"
    assert fingerprint_model_state(child) == before
    assert tuple(id(value) for value in child.parameters()) == parameter_ids
    assert run.attempts[0].rollback_exact is True
    assert provider.refresh_count == 0


def test_executor_rejects_child_state_changed_after_planning() -> None:
    _parent, child, plan, context, evidence = _fixture()
    with torch.no_grad():
        next(child.parameters()).add_(1.0)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
    )

    with pytest.raises(Exception, match="changed after planning"):
        executor.execute(plan, context)


def test_executor_rejects_parent_state_changed_after_planning() -> None:
    parent, child, plan, context, evidence = _fixture()
    before = fingerprint_model_state(child)
    with torch.no_grad():
        next(parent.parameters()).add_(1.0)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
    )

    with pytest.raises(Exception, match="parent state changed"):
        executor.execute(plan, context)
    assert fingerprint_model_state(child) == before


def test_unverified_noop_operator_cannot_claim_complete_coverage() -> None:
    _parent, _child, plan, context, evidence = _fixture()
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((_NoOpOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
    )

    with pytest.raises(Exception, match="installed no state change"):
        executor.execute(plan, context)


def test_final_gate_rejection_restores_initial_child_and_supersedes_commits() -> None:
    _parent, child, plan, context, evidence = _fixture()
    before = fingerprint_model_state(child)
    provider = StaticEvidenceProvider(evidence)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=provider,
        evaluator=_Accept(),
        final_evaluator=_RejectFirst(),
    )

    run = executor.execute(plan, context)

    assert run.status == "final_gate_rejected"
    assert fingerprint_model_state(child) == before
    assert run.accepted_action_ids == ()
    assert run.completed_target_parameter_paths == ()
    assert provider.restore_count == 1
    assert (
        executor.ledger.decision(run.attempts[-1].execution_signature).allowed is False
    )
    if len(run.attempts) > 1:
        assert (
            executor.ledger.decision(run.attempts[0].execution_signature).allowed
            is True
        )


def test_later_local_rejection_rolls_back_all_prior_commits() -> None:
    _parent, child, plan, context, evidence = _fixture()
    before = fingerprint_model_state(child)
    provider = StaticEvidenceProvider(evidence)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=provider,
        evaluator=_AcceptThenReject(),
    )

    run = executor.execute(plan, context)

    assert run.status == "rejected"
    assert fingerprint_model_state(child) == before
    assert run.accepted_action_ids == ()
    assert run.completed_target_parameter_paths == ()
    assert provider.restore_count == 1
    assert executor.ledger.decision(run.attempts[0].execution_signature).allowed is True


def test_mutating_evaluator_is_rejected_and_whole_run_is_restored() -> None:
    _parent, child, plan, context, evidence = _fixture()
    before = fingerprint_model_state(child)
    provider = StaticEvidenceProvider(evidence)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=provider,
        evaluator=_MutatingAccept(),
    )

    with pytest.raises(Exception, match="evaluator mutated"):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == before
    assert provider.restore_count == 1


def test_parent_mutation_by_provider_is_detected_and_parent_is_restored() -> None:
    parent, child, plan, context, evidence = _fixture()
    parent_before = fingerprint_model_state(parent)
    child_before = fingerprint_model_state(child)
    provider = _ParentMutatingEvidence(evidence)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=provider,
        evaluator=_Accept(),
    )

    with pytest.raises(Exception, match="parent state changed"):
        executor.execute(plan, context)

    assert fingerprint_model_state(parent) == parent_before
    assert fingerprint_model_state(child) == child_before


def test_refresh_mutation_is_detected_and_whole_run_is_restored() -> None:
    _parent, child, plan, context, evidence = _fixture()
    child_before = fingerprint_model_state(child)
    provider = _ChildMutatingRefresh(evidence)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=provider,
        evaluator=_Accept(),
    )

    with pytest.raises(Exception, match="refresh mutated"):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == child_before


def test_restore_hook_cannot_corrupt_the_globally_restored_child() -> None:
    _parent, child, plan, context, evidence = _fixture()
    child_before = fingerprint_model_state(child)
    provider = _ChildMutatingRestore(evidence)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=provider,
        evaluator=_Accept(),
        final_evaluator=_RejectFirst(),
    )

    run = executor.execute(plan, context)

    assert run.status == "final_gate_rejected"
    assert fingerprint_model_state(child) == child_before


def test_post_prepare_gradient_violation_restores_state_and_gradient_absence() -> None:
    _parent, child, plan, context, evidence = _fixture()
    child_before = fingerprint_model_state(child)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((_GradientAfterPrepare(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
    )

    with pytest.raises(Exception, match="zero-gradient"):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == child_before
    assert all(parameter.grad is None for parameter in child.parameters())


def test_backward_attempt_is_forbidden_instead_of_hardcoded_as_zero() -> None:
    _parent, child, plan, context, evidence = _fixture()
    child_before = fingerprint_model_state(child)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((_BackwardAttemptOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
    )

    with pytest.raises(Exception, match="static zero-training audit"):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == child_before


def test_enable_grad_attempt_is_forbidden_during_folding() -> None:
    _parent, child, plan, context, evidence = _fixture()
    child_before = fingerprint_model_state(child)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((_EnableGradAttemptOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
    )

    with pytest.raises(Exception, match="static zero-training audit"):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == child_before


def test_mutating_initializer_is_rejected_before_evidence_and_restores_child() -> None:
    _parent, child, plan, context, evidence = _fixture()
    child_before = fingerprint_model_state(child)
    provider = _MalformedEvidence(evidence)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=provider,
        evaluator=_Accept(),
        initializer=_MutatingInitializer(),
    )

    with pytest.raises(Exception, match="initializers must be non-mutating"):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == child_before
    assert provider.restore_count == 0


def test_nonpersistent_buffer_mutation_is_hashed_and_restored() -> None:
    parent, child, plan, context, evidence = _fixture()
    parent_buffer = next(parent.named_buffers())[1]
    child_buffer = next(child.named_buffers())[1]
    parent_before = parent_buffer.detach().clone()
    child_before = child_buffer.detach().clone()
    provider = _NonPersistentBufferMutatingRefresh(evidence)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=provider,
        evaluator=_Accept(),
    )

    with pytest.raises(Exception, match="parent state changed|refresh mutated"):
        executor.execute(plan, context)

    torch.testing.assert_close(parent_buffer, parent_before)
    torch.testing.assert_close(child_buffer, child_before)


def test_evidence_provider_cannot_leave_a_forward_hook_on_the_child() -> None:
    _parent, child, plan, context, evidence = _fixture()
    child_before = fingerprint_model_state(child)
    assert not child.lm_head._forward_hooks
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=_HookMutatingEvidence(evidence),
        evaluator=_Accept(),
    )

    with pytest.raises(Exception, match="mutated the incumbent child"):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == child_before
    assert not child.lm_head._forward_hooks


def test_evaluator_cannot_leave_a_forward_hook_on_the_child() -> None:
    _parent, child, plan, context, evidence = _fixture()
    child_before = fingerprint_model_state(child)
    assert not child.lm_head._forward_hooks
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_HookMutatingAccept(),
    )

    with pytest.raises(Exception, match="evaluator mutated"):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == child_before
    assert not child.lm_head._forward_hooks


def test_executor_enforces_the_planned_evidence_channel_contract() -> None:
    _parent, child, plan, context, evidence = _fixture()
    child_before = fingerprint_model_state(child)
    provider = _MissingChannelEvidence(evidence)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=provider,
        evaluator=_Accept(),
    )

    with pytest.raises(Exception, match="Evidence manifest"):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == child_before
    assert provider.restore_count == 1


def test_prebound_backward_alias_cannot_write_model_gradients() -> None:
    _parent, child, plan, context, evidence = _fixture()
    original_requires_grad = tuple(
        parameter.requires_grad for parameter in child.parameters()
    )
    evaluator = _PreboundBackwardAliasAccept(child)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=evaluator,
    )

    run = executor.execute(plan, context)

    assert run.complete
    assert all(parameter.grad is None for parameter in child.parameters())
    assert (
        tuple(parameter.requires_grad for parameter in child.parameters())
        == original_requires_grad
    )
    report = run.as_dict()
    assert report["all_parent_and_child_parameters_frozen_during_callbacks"] is True
    assert report["historical_zero_training_proven_for_arbitrary_plugins"] is False


def test_fold_guard_restores_inherited_optimizer_methods_exactly() -> None:
    adamw_step = torch.optim.AdamW.step
    adamw_init = torch.optim.AdamW.__init__
    step_was_direct = "step" in vars(torch.optim.AdamW)
    init_was_direct = "__init__" in vars(torch.optim.AdamW)
    _parent, _child, plan, context, evidence = _fixture()

    run = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
    ).execute(plan, context)

    assert run.complete
    assert torch.optim.AdamW.step is adamw_step
    assert torch.optim.AdamW.__init__ is adamw_init
    assert ("step" in vars(torch.optim.AdamW)) is step_was_direct
    assert ("__init__" in vars(torch.optim.AdamW)) is init_was_direct


def test_fold_guard_runtime_blocks_and_counts_common_pruning_api() -> None:
    from torch.nn.utils import prune

    original = prune.l1_unstructured
    model = torch.nn.Linear(4, 4)
    guard = _FoldOnlyTorchGuard()

    with pytest.raises(Exception, match="Pruning API"):
        with guard:
            prune.l1_unstructured(model, "weight", amount=0.5)

    assert guard.pruning_operations == 1
    assert guard.quantization_operations == 0
    assert prune.l1_unstructured is original
    assert "weight_orig" not in dict(model.named_parameters())


def test_fold_guard_blocks_direct_pruning_class_apply_and_restores_descriptor() -> None:
    from torch.nn.utils import prune

    original = inspect.getattr_static(prune.L1Unstructured, "apply")
    model = torch.nn.Linear(4, 4)
    guard = _FoldOnlyTorchGuard()

    with pytest.raises(Exception, match="Pruning API"):
        with guard:
            prune.L1Unstructured.apply(
                model,
                "weight",
                amount=0.5,
            )

    assert guard.pruning_operations == 1
    assert inspect.getattr_static(prune.L1Unstructured, "apply") is original
    assert "weight_orig" not in dict(model.named_parameters())


def test_nested_fold_guards_restore_pruning_class_descriptors_exactly() -> None:
    from torch.nn.utils import prune

    original = inspect.getattr_static(prune.L1Unstructured, "apply")
    outer = _FoldOnlyTorchGuard()
    inner = _FoldOnlyTorchGuard()
    model = torch.nn.Linear(4, 4)

    with outer:
        outer_replacement = inspect.getattr_static(
            prune.L1Unstructured,
            "apply",
        )
        with pytest.raises(Exception, match="Pruning API"):
            with inner:
                prune.L1Unstructured.apply(
                    model,
                    "weight",
                    amount=0.5,
                )
        assert (
            inspect.getattr_static(prune.L1Unstructured, "apply")
            is outer_replacement
        )

    assert inspect.getattr_static(prune.L1Unstructured, "apply") is original
    assert inner.pruning_operations == 1
    assert outer.pruning_operations == 0


def test_fold_guard_runtime_blocks_and_counts_common_quantization_api() -> None:
    original = torch.quantize_per_tensor
    guard = _FoldOnlyTorchGuard()

    with pytest.raises(Exception, match="Quantization API"):
        with guard:
            torch.quantize_per_tensor(
                torch.ones(4),
                scale=0.1,
                zero_point=0,
                dtype=torch.qint8,
            )

    assert guard.quantization_operations == 1
    assert guard.pruning_operations == 0
    assert torch.quantize_per_tensor is original


def test_fold_guard_does_not_block_bfloat16_or_float32_casts() -> None:
    guard = _FoldOnlyTorchGuard()

    with guard:
        value = torch.ones(4).to(dtype=torch.bfloat16)
        restored = value.to(dtype=torch.float32)

    assert value.dtype == torch.bfloat16
    assert restored.dtype == torch.float32
    assert guard.pruning_operations == 0
    assert guard.quantization_operations == 0
