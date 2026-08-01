from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge import dynamic_causal_operator_promotion as promotion  # noqa: E402
from neural_forge.dynamic_causal_operator_promotion import (  # noqa: E402
    CausalGQAAttentionBranchOperator,
    CausalGatedMLPBranchOperator,
    DynamicCausalOperatorPromotionError,
    IdentityVocabularyProjector,
    StrictTokenInterfaceBridgeOperator,
)
from neural_forge.hf_arch_layout import build_decoder_layout  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    DynamicFoldExecutor,
    ExactCopyOperator,
    FoldDecision,
    FoldEvidence,
    FoldExecutionContext,
    FoldOperatorRegistry,
    StaticEvidenceProvider,
    fingerprint_model_state,
)
from neural_forge.dynamic_fold_planner import (  # noqa: E402
    CAUSAL_ATTENTION_BRANCH_METHOD,
    CAUSAL_GATED_MLP_BRANCH_METHOD,
    CAUSAL_LAYER_BRANCH_METHODS,
    DynamicFoldPlanner,
    DynamicFoldPlanningError,
)
from neural_forge.dynamic_fold_registry import (  # noqa: E402
    MethodStatus,
    default_method_registry,
)
from neural_forge.dynamic_model_inspector import inspect_model_circuits  # noqa: E402
from neural_forge.hf_causal_circuit_evidence import (  # noqa: E402
    capture_hf_causal_circuit_evidence,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
    seal_strict_tokenizer_bridge_artifact,
)


TOKENIZER = canonical_sha256({"tokenizer": "shared-llama-qwen3"})


def _llama(seed: int, **changes: Any):
    torch.manual_seed(seed)
    configuration = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 64,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
    }
    configuration.update(changes)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(**configuration)
    )
    model.eval()
    return model


def _qwen3(seed: int, **changes: Any):
    torch.manual_seed(seed)
    configuration = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 64,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
    }
    configuration.update(changes)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**configuration))
    model.eval()
    return model


def _input_ids():
    return torch.arange(1, 49, dtype=torch.long).reshape(2, 24) % 32


def _capture(parent, parent_graph):
    return capture_hf_causal_circuit_evidence(
        parent,
        {"input_ids": _input_ids()},
        parent_graph_fingerprint=parent_graph.fingerprint,
        metadata={"test": "family-neutral-causal-branch-promotion"},
    )


class _AcceptStructuralCandidate:
    """Behavior remains the outer authority; structure is admitted here."""

    def evaluate(self, context: Any) -> FoldDecision:
        return FoldDecision.accept("Structurally complete cached branch install.")


def _plan(
    parent_graph,
    child_graph,
    *,
    evidence_fingerprint,
    child_state,
    enabled,
    parent_state=None,
):
    return DynamicFoldPlanner(enabled_experimental_method_ids=enabled).plan(
        (parent_graph,),
        child_graph,
        child_state_fingerprint=child_state,
        evidence_fingerprint=evidence_fingerprint,
        parent_state_fingerprints={
            "parent": (
                canonical_sha256({"parent_state": "fixed"})
                if parent_state is None
                else parent_state
            )
        },
    )


def _context(parent, parent_graph, child, child_graph) -> FoldExecutionContext:
    return FoldExecutionContext(
        parent_models={"parent": parent},
        child_model=child,
        parent_graphs={"parent": parent_graph},
        child_graph=child_graph,
    )


def test_registry_promotes_both_causal_kernels_as_experimental_methods() -> None:
    registry = default_method_registry()
    attention = registry.get(CAUSAL_ATTENTION_BRANCH_METHOD)
    gated_mlp = registry.get(CAUSAL_GATED_MLP_BRANCH_METHOD)

    for method in (attention, gated_mlp):
        assert method.status == MethodStatus.EXPERIMENTAL.value
        assert method.selectable
        assert method.zero_gradient
        assert method.standalone
        assert registry.require_selectable(method.method_id) is method
    assert attention.source_kinds == attention.target_kinds == ("attention",)
    assert gated_mlp.source_kinds == gated_mlp.target_kinds == ("gated_mlp",)
    assert set(attention.coupled_roles) == {
        "gqa_group",
        "input_norm_q_k_v_o_residual_write",
        "q_k_v_o",
    }
    assert set(gated_mlp.coupled_roles) == {
        "gate_up_post_swiglu_down",
        "input_norm_gate_up_post_swiglu_down_residual_write",
    }
    # Every declared channel must name something the sealed one-forward
    # capture actually produces.
    layer_fields = {
        "block_input",
        "attention_norm_output",
        "query_projection_output",
        "key_projection_output",
        "value_projection_output",
        "attention_delta",
        "post_attention_residual",
        "mlp_norm_output",
        "mlp_gate_output",
        "mlp_up_output",
        "mlp_delta",
    }
    bundle_fields = {
        "attention_mask",
        "position_ids",
        "sealed_one_forward_parent_circuit",
        "fit_validation_token_split",
    }
    for method in (attention, gated_mlp):
        assert set(method.evidence_channels) <= layer_fields | bundle_fields
        assert "sealed_one_forward_parent_circuit" in method.evidence_channels


def test_planner_only_fills_the_cross_family_layer_gap_when_enabled() -> None:
    parent_graph = inspect_model_circuits(
        _llama(3),
        model_name="parent",
        tokenizer_fingerprint=TOKENIZER,
    )
    child_graph = inspect_model_circuits(
        _qwen3(5),
        model_name="child",
        tokenizer_fingerprint=TOKENIZER,
    )
    arguments = {
        "evidence_fingerprint": canonical_sha256({"cache": "parent-once"}),
        "child_state": canonical_sha256({"child": "blank"}),
    }

    closed = _plan(parent_graph, child_graph, enabled=(), **arguments)
    assert closed.ready is False
    assert {gap.required_capability for gap in closed.gaps} == {
        "cross_family_closed_loop_layer_operator"
    }
    assert not any(
        action.method_id in CAUSAL_LAYER_BRANCH_METHODS for action in closed.actions
    )

    opened = _plan(
        parent_graph,
        child_graph,
        enabled=CAUSAL_LAYER_BRANCH_METHODS,
        **arguments,
    )
    assert opened.ready
    assert not opened.gaps
    branch_methods = [
        action.method_id
        for action in opened.actions
        if action.method_id in CAUSAL_LAYER_BRANCH_METHODS
    ]
    assert branch_methods == [
        CAUSAL_ATTENTION_BRANCH_METHOD,
        CAUSAL_GATED_MLP_BRANCH_METHOD,
    ] * 2
    attention_action = next(
        action
        for action in opened.actions
        if action.method_id == CAUSAL_ATTENTION_BRANCH_METHOD
    )
    configuration = dict(attention_action.configuration)
    assert configuration["source_family"] == "llama"
    assert configuration["target_family"] == "qwen3"
    assert configuration["source_layer_index"] == 0
    assert set(attention_action.target_node_ids) == {
        "layer.000.attention_norm",
        "layer.000.attention",
        "layer.000.attention_residual",
    }
    assert {
        path for action in opened.actions for path in action.target_parameter_paths
    } == set(child_graph.parameter_paths)


def test_only_experimental_methods_can_be_explicitly_enabled() -> None:
    with pytest.raises(DynamicFoldPlanningError, match="experimental"):
        DynamicFoldPlanner(enabled_experimental_method_ids=("copy_exact",))


def test_causal_branch_operators_execute_one_fold_only_cross_check() -> None:
    parent = _llama(3)
    child = _llama(7, intermediate_size=16)
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=TOKENIZER,
    )
    child_graph = inspect_model_circuits(
        child,
        model_name="child",
        tokenizer_fingerprint=TOKENIZER,
    )
    evidence = _capture(parent, parent_graph)
    blank = fingerprint_model_state(child)
    plan = _plan(
        parent_graph,
        child_graph,
        evidence_fingerprint=evidence.cache_key,
        child_state=blank,
        enabled=CAUSAL_LAYER_BRANCH_METHODS,
        parent_state=fingerprint_model_state(parent),
    )
    assert plan.ready

    run = DynamicFoldExecutor(
        operators=FoldOperatorRegistry(
            (
                ExactCopyOperator(),
                CausalGQAAttentionBranchOperator(),
                CausalGatedMLPBranchOperator(),
            )
        ),
        evidence_provider=StaticEvidenceProvider(evidence.cache_key, evidence),
        evaluator=_AcceptStructuralCandidate(),
    ).execute(plan, _context(parent, parent_graph, child, child_graph))

    assert run.status == "complete"
    assert all(attempt.accepted for attempt in run.attempts)
    assert run.observed_parent_root_forward_calls == 0
    assert run.observed_parent_module_forward_calls == 0
    for name in (
        "optimizer_constructions",
        "optimizer_steps",
        "backward_calls",
        "autograd_grad_calls",
        "pruning_operations",
        "quantization_operations",
    ):
        assert getattr(run, name) == 0
    assert run.executed_fold_static_zero_training_audit_passed is True
    attention_attempts = [
        attempt
        for attempt in run.attempts
        if attempt.method_id == CAUSAL_ATTENTION_BRANCH_METHOD
    ]
    mlp_attempts = [
        attempt
        for attempt in run.attempts
        if attempt.method_id == CAUSAL_GATED_MLP_BRANCH_METHOD
    ]
    assert len(attention_attempts) == len(mlp_attempts) == 2
    for attempt in attention_attempts:
        report = attempt.transaction_report
        assert report["source_family"] == report["target_family"] == "llama"
        assert report["cached_parent_evidence_sha256"] == evidence.cache_key
        assert report["parent_forward_calls"] == 0
        # The target attention topology equals the source topology here, so the
        # coupled kernel should reproduce the held-out attention write closely.
        assert report["validation_attention_delta_relative_rmse"] < 0.05
    for attempt in mlp_attempts:
        report = attempt.transaction_report
        assert report["validation_mlp_delta_relative_rmse"] < 1.0
        assert report["parent_forward_calls"] == 0
    assert fingerprint_model_state(child) != blank
    with torch.no_grad():
        logits = child(input_ids=_input_ids(), use_cache=False).logits
    assert logits.shape == (2, 24, 32)
    assert bool(torch.isfinite(logits).all())


def test_rejected_causal_branch_rolls_back_to_the_exact_incumbent() -> None:
    parent = _llama(43)
    child = _llama(47, intermediate_size=16)
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=TOKENIZER,
    )
    child_graph = inspect_model_circuits(
        child,
        model_name="child",
        tokenizer_fingerprint=TOKENIZER,
    )
    evidence = _capture(parent, parent_graph)
    blank = fingerprint_model_state(child)
    plan = _plan(
        parent_graph,
        child_graph,
        evidence_fingerprint=evidence.cache_key,
        child_state=blank,
        enabled=CAUSAL_LAYER_BRANCH_METHODS,
        parent_state=fingerprint_model_state(parent),
    )

    class _RejectAttentionBranches:
        def evaluate(self, context: Any) -> FoldDecision:
            if context.action.method_id == CAUSAL_ATTENTION_BRANCH_METHOD:
                return FoldDecision.reject("Rejected for the rollback contract.")
            return FoldDecision.accept("Accepted.")

    run = DynamicFoldExecutor(
        operators=FoldOperatorRegistry(
            (
                ExactCopyOperator(),
                CausalGQAAttentionBranchOperator(),
                CausalGatedMLPBranchOperator(),
            )
        ),
        evidence_provider=StaticEvidenceProvider(evidence.cache_key, evidence),
        evaluator=_RejectAttentionBranches(),
    ).execute(plan, _context(parent, parent_graph, child, child_graph))

    rejected = [attempt for attempt in run.attempts if not attempt.accepted]
    assert len(rejected) == 1
    assert rejected[0].method_id == CAUSAL_ATTENTION_BRANCH_METHOD
    assert rejected[0].rollback_exact is True
    assert (
        rejected[0].transaction_report["final_state_fingerprint"]
        == rejected[0].incumbent_state_fingerprint
    )


def test_attention_branch_installs_a_per_head_query_key_normalized_target() -> None:
    parent = _llama(11)
    child = _qwen3(13)
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=TOKENIZER,
    )
    child_graph = inspect_model_circuits(
        child,
        model_name="child",
        tokenizer_fingerprint=TOKENIZER,
    )
    evidence = _capture(parent, parent_graph)
    before = fingerprint_model_state(child)
    plan = _plan(
        parent_graph,
        child_graph,
        evidence_fingerprint=evidence.cache_key,
        child_state=before,
        enabled=CAUSAL_LAYER_BRANCH_METHODS,
    )
    action = next(
        value
        for value in plan.actions
        if value.method_id == CAUSAL_ATTENTION_BRANCH_METHOD
    )

    transaction = CausalGQAAttentionBranchOperator().prepare(
        action,
        _context(parent, parent_graph, child, child_graph),
        FoldEvidence(
            evidence.cache_key,
            evidence,
            channels=action.evidence_requirements,
        ),
    )
    report = transaction.report

    # The two head-norm gains are planned target parameters of the attention
    # node, so the branch owns them or it owns nothing.
    assert report["installed_parameter_paths"] == sorted(
        action.target_parameter_paths
    )
    assert "model.layers.0.self_attn.q_norm.weight" in (
        report["installed_parameter_paths"]
    )
    assert "model.layers.0.self_attn.k_norm.weight" in (
        report["installed_parameter_paths"]
    )
    assert report["target_query_key_norm"] == "rms_norm"
    assert report["source_family"] == "llama"
    assert report["target_family"] == "qwen3"
    assert report["parent_forward_calls"] == 0
    assert report["validation_attention_delta_relative_rmse"] < 0.05
    # The head norm destroys each token's per-head magnitude, so the gain fit
    # is deliberately far from exact; it must still be a real finite fit.
    assert 0.0 < report["query_norm_gain_relative_rmse"] < 1.0
    assert 0.0 < report["key_norm_gain_relative_rmse"] < 1.0
    for path in (
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
    ):
        gains = child.get_parameter(path)
        assert tuple(gains.shape) == (4,)
        assert bool(torch.isfinite(gains).all())
        assert bool((gains != 0.0).all())

    assert transaction.rollback()["rollback_exact"] is True
    assert fingerprint_model_state(child) == before


def test_attention_branch_fails_closed_on_unsupported_head_norm_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _llama(11)
    child = _qwen3(13)
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=TOKENIZER,
    )
    child_graph = inspect_model_circuits(
        child,
        model_name="child",
        tokenizer_fingerprint=TOKENIZER,
    )
    evidence = _capture(parent, parent_graph)
    before = fingerprint_model_state(child)
    plan = _plan(
        parent_graph,
        child_graph,
        evidence_fingerprint=evidence.cache_key,
        child_state=before,
        enabled=CAUSAL_LAYER_BRANCH_METHODS,
    )
    action = next(
        value
        for value in plan.actions
        if value.method_id == CAUSAL_ATTENTION_BRANCH_METHOD
    )
    fold_evidence = FoldEvidence(
        evidence.cache_key,
        evidence,
        channels=action.evidence_requirements,
    )
    honest = build_decoder_layout(child)

    def _patched(**changes: Any):
        """Rebuild the child layout with one doctored attention head norm."""

        def _build(model: Any):
            assert model is child
            blocks = tuple(
                replace(
                    block,
                    attention=replace(block.attention, **changes),
                )
                for block in honest.blocks
            )
            return replace(honest, blocks=blocks)

        return _build

    # gemma3 stores per-head gains as ``1 + w``; a direct-scale fit would be
    # silently mis-scaled by one.
    unit_offset = replace(
        honest.blocks[0].attention.query_norm,
        weight_kind="unit_offset",
    )
    monkeypatch.setattr(
        promotion,
        "build_decoder_layout",
        _patched(query_norm=unit_offset),
    )
    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="direct-scale per-head RMS",
    ):
        CausalGQAAttentionBranchOperator().prepare(
            action,
            _context(parent, parent_graph, child, child_graph),
            fold_evidence,
        )

    # A target that normalizes only one of Q/K has no coupled pre-RoPE
    # coordinate system at all.
    monkeypatch.setattr(
        promotion,
        "build_decoder_layout",
        _patched(key_norm=None),
    )
    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="both queries and keys",
    ):
        CausalGQAAttentionBranchOperator().prepare(
            action,
            _context(parent, parent_graph, child, child_graph),
            fold_evidence,
        )

    monkeypatch.setattr(
        promotion,
        "build_decoder_layout",
        _patched(
            key_norm=replace(honest.blocks[0].attention.key_norm, epsilon=1e-5)
        ),
    )
    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="shared target Q/K normalization epsilon",
    ):
        CausalGQAAttentionBranchOperator().prepare(
            action,
            _context(parent, parent_graph, child, child_graph),
            fold_evidence,
        )

    assert fingerprint_model_state(child) == before


def test_causal_branches_fail_closed_on_a_residual_width_crossing() -> None:
    parent = _llama(17)
    child = _llama(19, intermediate_size=16)
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=TOKENIZER,
    )
    child_graph = inspect_model_circuits(
        child,
        model_name="child",
        tokenizer_fingerprint=TOKENIZER,
    )
    evidence = _capture(parent, parent_graph)
    plan = _plan(
        parent_graph,
        child_graph,
        evidence_fingerprint=evidence.cache_key,
        child_state=fingerprint_model_state(child),
        enabled=CAUSAL_LAYER_BRANCH_METHODS,
    )
    narrow = _llama(23, hidden_size=8, intermediate_size=16)
    narrow_graph = inspect_model_circuits(
        narrow,
        model_name="child",
        tokenizer_fingerprint=TOKENIZER,
    )
    context = _context(parent, parent_graph, narrow, narrow_graph)
    before = fingerprint_model_state(narrow)

    for method_id, operator in (
        (CAUSAL_ATTENTION_BRANCH_METHOD, CausalGQAAttentionBranchOperator()),
        (CAUSAL_GATED_MLP_BRANCH_METHOD, CausalGatedMLPBranchOperator()),
    ):
        action = next(
            value for value in plan.actions if value.method_id == method_id
        )
        with pytest.raises(
            DynamicCausalOperatorPromotionError,
            match="shared residual coordinate system",
        ):
            operator.prepare(
                action,
                context,
                FoldEvidence(
                    evidence.cache_key,
                    evidence,
                    channels=action.evidence_requirements,
                ),
            )

    assert fingerprint_model_state(narrow) == before


class _StubTokenizer:
    """Minimal deterministic row-addressable tokenizer for bridge sealing."""

    def __init__(self, surfaces: list[str]) -> None:
        self.surfaces = list(surfaces)
        self.all_special_ids: list[int] = []
        self.bos_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None
        self.unk_token_id = None

    def __len__(self) -> int:
        return len(self.surfaces)

    def get_vocab(self) -> dict[str, int]:
        return {
            f"raw-{token_id}-{surface}": token_id
            for token_id, surface in enumerate(self.surfaces)
        }

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"raw-{token_id}-{self.surfaces[token_id]}"

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.surfaces[token_id] for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        exact = [
            token_id
            for token_id, surface in enumerate(self.surfaces)
            if surface == text
        ]
        if len(exact) != 1:
            raise KeyError(text)
        return exact


def _sealed_bridge(tmp_path: Path):
    donor = _StubTokenizer([f"s{index}" for index in range(8)])
    target = _StubTokenizer(["s5", "s1", "s7", "s0"])
    root = tmp_path / "target-tokenizer"
    root.mkdir(parents=True)
    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "test"}}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"clean_up_tokenization_spaces": False}),
        encoding="utf-8",
    )
    artifact = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        root,
        donor_row_capacity=8,
        target_row_capacity=4,
    )
    return donor, target, artifact


def test_token_interface_bridge_operator_installs_strict_donor_rows(
    tmp_path: Path,
) -> None:
    donor, target, artifact = _sealed_bridge(tmp_path)
    parent = _llama(29, vocab_size=8)
    child = _llama(31, vocab_size=4)
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=fingerprint_hf_tokenizer_semantics(donor),
    )
    child_graph = inspect_model_circuits(
        child,
        model_name="child",
        tokenizer_fingerprint=fingerprint_hf_tokenizer_semantics(target),
    )
    evidence_fingerprint = canonical_sha256({"cache": "bridge-only"})
    plan = _plan(
        parent_graph,
        child_graph,
        evidence_fingerprint=evidence_fingerprint,
        child_state=fingerprint_model_state(child),
        enabled=(),
        parent_state=fingerprint_model_state(parent),
    )
    assert plan.ready
    bridge_actions = [
        action
        for action in plan.actions
        if action.method_id == "token_interface_bridge"
    ]
    assert len(bridge_actions) == 1

    run = DynamicFoldExecutor(
        operators=FoldOperatorRegistry(
            (
                ExactCopyOperator(),
                StrictTokenInterfaceBridgeOperator(bridge_directory=tmp_path / "target-tokenizer"),
            )
        ),
        evidence_provider=StaticEvidenceProvider(evidence_fingerprint),
        evaluator=_AcceptStructuralCandidate(),
    ).execute(plan, _context(parent, parent_graph, child, child_graph))

    assert run.status == "complete"
    bridge_attempt = next(
        attempt
        for attempt in run.attempts
        if attempt.method_id == "token_interface_bridge"
    )
    report = bridge_attempt.transaction_report
    assert report["bridge_artifact_sha256"] == artifact.artifact_sha256
    assert report["target_tied_storage"] is False
    assert report["donor_row_capacity"] == 8
    assert report["target_row_capacity"] == 4
    rows = artifact.transport_profile.target_to_donor_row_indices
    assert rows == (5, 1, 7, 0)
    with torch.no_grad():
        assert torch.equal(
            child.get_parameter("model.embed_tokens.weight"),
            parent.get_parameter("model.embed_tokens.weight")[list(rows)],
        )
        assert torch.equal(
            child.get_parameter("lm_head.weight"),
            parent.get_parameter("lm_head.weight")[list(rows)],
        )


def test_token_interface_bridge_operator_installs_one_tied_storage_group(
    tmp_path: Path,
) -> None:
    donor, target, artifact = _sealed_bridge(tmp_path)
    parent = _llama(53, vocab_size=8, tie_word_embeddings=True)
    child = _llama(59, vocab_size=4, tie_word_embeddings=True)
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=fingerprint_hf_tokenizer_semantics(donor),
    )
    child_graph = inspect_model_circuits(
        child,
        model_name="child",
        tokenizer_fingerprint=fingerprint_hf_tokenizer_semantics(target),
    )
    plan = _plan(
        parent_graph,
        child_graph,
        evidence_fingerprint=canonical_sha256({"cache": "tied-bridge"}),
        child_state=fingerprint_model_state(child),
        enabled=(),
    )
    action = next(
        value
        for value in plan.actions
        if value.method_id == "token_interface_bridge"
    )
    transaction = StrictTokenInterfaceBridgeOperator(artifact=artifact).prepare(
        action,
        _context(parent, parent_graph, child, child_graph),
        FoldEvidence(
            canonical_sha256({"cache": "tied-bridge"}),
            None,
            channels=action.evidence_requirements,
        ),
    )

    assert transaction.report["target_tied_storage"] is True
    rows = artifact.transport_profile.target_to_donor_row_indices
    with torch.no_grad():
        assert torch.equal(
            child.get_parameter("lm_head.weight"),
            parent.get_parameter("model.embed_tokens.weight")[list(rows)],
        )
    report = transaction.commit()
    assert report["accepted"] is True
    assert report["rollback_exact"] is None


def test_token_interface_bridge_operator_fails_closed_on_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    donor, target, _artifact = _sealed_bridge(tmp_path)
    parent = _llama(37, vocab_size=8)
    child = _llama(41, vocab_size=4)
    honest_parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=fingerprint_hf_tokenizer_semantics(donor),
    )
    honest_child_graph = inspect_model_circuits(
        child,
        model_name="child",
        tokenizer_fingerprint=fingerprint_hf_tokenizer_semantics(target),
    )
    plan = _plan(
        honest_parent_graph,
        honest_child_graph,
        evidence_fingerprint=canonical_sha256({"cache": "bridge-only"}),
        child_state=fingerprint_model_state(child),
        enabled=(),
    )
    action = next(
        value
        for value in plan.actions
        if value.method_id == "token_interface_bridge"
    )
    operator = StrictTokenInterfaceBridgeOperator(
        bridge_directory=tmp_path / "target-tokenizer"
    )
    fold_evidence = FoldEvidence(
        canonical_sha256({"cache": "bridge-only"}),
        None,
        channels=action.evidence_requirements,
    )
    before = fingerprint_model_state(child)

    foreign_parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=canonical_sha256({"tokenizer": "not-the-donor"}),
    )
    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="donor tokenizer differs",
    ):
        operator.prepare(
            action,
            _context(parent, foreign_parent_graph, child, honest_child_graph),
            fold_evidence,
        )

    foreign_child_graph = inspect_model_circuits(
        child,
        model_name="child",
        tokenizer_fingerprint=canonical_sha256({"tokenizer": "not-the-target"}),
    )
    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="target tokenizer differs",
    ):
        operator.prepare(
            action,
            _context(parent, honest_parent_graph, child, foreign_child_graph),
            fold_evidence,
        )

    assert fingerprint_model_state(child) == before


def test_identity_vocabulary_projector_is_a_width_sealed_row_identity() -> None:
    projector = IdentityVocabularyProjector(source_width=6)
    rows = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    assert projector.source_width == projector.target_width == 6
    assert len(projector.map_sha256) == 64
    torch.testing.assert_close(
        projector.fold_consumer_weight(rows),
        rows.to(dtype=torch.float64),
    )
    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="matching row widths",
    ):
        projector.fold_consumer_weight(rows[:, :4])
