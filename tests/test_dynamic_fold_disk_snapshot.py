from __future__ import annotations

from pathlib import Path

import pytest

from neural_forge.dynamic_behavior_gate import (
    BehaviorGateBinding,
    expose_behavior_gate_binding,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_compiler import (
    DynamicFoldSystemStack,
    compile_loaded_models,
)
from neural_forge.dynamic_fold_executor import (
    DynamicFoldExecutor,
    ExactCopyOperator,
    FoldDecision,
    FoldExecutionContext,
    FoldOperatorRegistry,
    StaticEvidenceProvider,
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from neural_forge.dynamic_fold_planner import DynamicFoldPlanner
from neural_forge.dynamic_fold_snapshot import (
    DynamicFoldSnapshotConfig,
    DynamicFoldSnapshotError,
    capture_model_state,
    snapshot_directory_inventory,
)
from neural_forge.dynamic_model_inspector import inspect_model_circuits


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _model(seed: int):
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
    def evaluate(self, _context):
        return FoldDecision.accept("accepted")


def _bound_test_gates(gate_id: str, baseline: str):
    final = _Accept()

    def fresh(_model):
        return True

    body = {
        "schema": "test-behavior-gate-manifest-v1",
        "gate_id": gate_id,
        "gate_version": 1,
        "baseline_fingerprint": baseline,
    }
    fingerprint = canonical_sha256(body)
    binding = BehaviorGateBinding(
        gate_id=gate_id,
        gate_version=1,
        baseline_fingerprint=baseline,
        manifest_fingerprint=fingerprint,
        manifest={**body, "manifest_fingerprint": fingerprint},
    )
    expose_behavior_gate_binding(final, binding)
    expose_behavior_gate_binding(fresh, binding)
    return final, fresh


class _Reject:
    def evaluate(self, _context):
        return FoldDecision.reject("rejected")


class _Explode:
    def evaluate(self, _context):
        raise RuntimeError("intentional evaluator failure")


class _MutatingFinal:
    def evaluate(self, context):
        child = context.execution_context.child_model
        child.eval()
        child.lm_head.register_forward_hook(
            lambda _module, _inputs, output: torch.zeros_like(output)
        )
        return FoldDecision.reject("invalid runtime mutation")


class _InventoryEvidence(StaticEvidenceProvider):
    def __init__(self, fingerprint: str, root: Path):
        super().__init__(fingerprint)
        self.root = root
        self.maximum_active_snapshots = 0

    def prepare(self, action, context):
        inventory = snapshot_directory_inventory(self.root)
        self.maximum_active_snapshots = max(
            self.maximum_active_snapshots,
            len(inventory),
        )
        return super().prepare(action, context)


def _fixture():
    parent = _model(1)
    child = _model(2)
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
    evidence = canonical_sha256({"cache": "immutable"})
    plan = DynamicFoldPlanner().plan(
        (parent_graph,),
        child_graph,
        child_state_fingerprint=fingerprint_model_state(child),
        evidence_fingerprint=evidence,
        parent_state_fingerprints={"parent": fingerprint_model_state(parent)},
    )
    context = FoldExecutionContext(
        parent_models={"parent": parent},
        child_model=child,
        parent_graphs={"parent": parent_graph},
        child_graph=child_graph,
    )
    return parent, child, plan, context, evidence, tokenizer


def test_compiler_disk_snapshot_accepts_and_cleans_all_files(tmp_path) -> None:
    parent, child, _plan, _context, evidence, tokenizer = _fixture()
    provider = _InventoryEvidence(evidence, tmp_path)
    parameter_ids = tuple(id(value) for value in child.parameters())
    buffer_ids = tuple(id(value) for value in child.buffers())
    runtime_before = fingerprint_model_runtime_state(child)

    result = compile_loaded_models(
        parent_models={"parent": parent},
        child_model=child,
        child_name="child",
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=provider,
        evaluator=_Accept(),
        final_evaluator=_Accept(),
        evidence_fingerprint=evidence,
        parent_tokenizer_fingerprints={"parent": tokenizer},
        child_tokenizer_fingerprint=tokenizer,
        snapshot_config=DynamicFoldSnapshotConfig.disk(
            tmp_path,
            chunk_bytes=32,
        ),
    )

    assert result.complete
    assert result.run.whole_model_snapshot_backend == "disk"
    assert result.run.as_dict()["whole_model_snapshot_cleanup_completed"] is True
    assert provider.maximum_active_snapshots == 2
    assert snapshot_directory_inventory(tmp_path) == {}
    assert tuple(id(value) for value in child.parameters()) == parameter_ids
    assert tuple(id(value) for value in child.buffers()) == buffer_ids
    assert fingerprint_model_runtime_state(child) == runtime_before
    assert fingerprint_model_state(child) == fingerprint_model_state(parent)


def test_rejected_disk_run_restores_exact_state_and_cleans(tmp_path) -> None:
    _parent, child, plan, context, evidence, _tokenizer = _fixture()
    child.train()
    child.model.layers[0].self_attn.eval()
    hook = child.lm_head.register_forward_hook(lambda _module, _inputs, output: output)
    state_before = fingerprint_model_state(child)
    runtime_before = fingerprint_model_runtime_state(child)
    parameter_ids = tuple(id(value) for value in child.parameters())
    buffer_ids = tuple(id(value) for value in child.buffers())
    hook_mapping_id = id(child.lm_head._forward_hooks)
    provider = _InventoryEvidence(evidence, tmp_path)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=provider,
        evaluator=_Accept(),
        final_evaluator=_Reject(),
        snapshot_config=DynamicFoldSnapshotConfig.disk(
            tmp_path,
            chunk_bytes=17,
        ),
    )

    run = executor.execute(plan, context)

    assert run.status == "final_gate_rejected"
    assert fingerprint_model_state(child) == state_before
    assert fingerprint_model_runtime_state(child) == runtime_before
    assert tuple(id(value) for value in child.parameters()) == parameter_ids
    assert tuple(id(value) for value in child.buffers()) == buffer_ids
    assert id(child.lm_head._forward_hooks) == hook_mapping_id
    assert hook.id in child.lm_head._forward_hooks
    assert provider.restore_count == 1
    assert snapshot_directory_inventory(tmp_path) == {}
    hook.remove()


def test_disk_snapshot_cleans_after_exception_and_restores_hooks_modes(
    tmp_path,
) -> None:
    _parent, child, plan, context, evidence, _tokenizer = _fixture()
    child.train()
    state_before = fingerprint_model_state(child)
    runtime_before = fingerprint_model_runtime_state(child)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
        final_evaluator=_MutatingFinal(),
        snapshot_config=DynamicFoldSnapshotConfig.disk(tmp_path),
    )

    with pytest.raises(Exception, match="Final fold evaluator mutated"):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == state_before
    assert fingerprint_model_runtime_state(child) == runtime_before
    assert snapshot_directory_inventory(tmp_path) == {}


def test_disk_snapshot_cleans_when_transaction_evaluator_raises(tmp_path) -> None:
    _parent, child, plan, context, evidence, _tokenizer = _fixture()
    state_before = fingerprint_model_state(child)
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Explode(),
        snapshot_config=DynamicFoldSnapshotConfig.disk(tmp_path),
    )

    with pytest.raises(RuntimeError, match="intentional evaluator failure"):
        executor.execute(plan, context)

    assert fingerprint_model_state(child) == state_before
    assert snapshot_directory_inventory(tmp_path) == {}


def test_invalid_snapshot_roots_fail_closed(tmp_path) -> None:
    invalid_file = tmp_path / "not-a-directory"
    invalid_file.write_text("x", encoding="utf-8")
    _parent, _child, plan, context, evidence, _tokenizer = _fixture()
    executor = DynamicFoldExecutor(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
        snapshot_config=DynamicFoldSnapshotConfig.disk(invalid_file),
    )

    with pytest.raises(DynamicFoldSnapshotError, match="real directory"):
        executor.execute(plan, context)

    with pytest.raises(DynamicFoldSnapshotError, match="does not exist"):
        DynamicFoldExecutor(
            operators=FoldOperatorRegistry((ExactCopyOperator(),)),
            evidence_provider=StaticEvidenceProvider(evidence),
            evaluator=_Accept(),
            snapshot_config=DynamicFoldSnapshotConfig.disk(tmp_path / "missing"),
        ).execute(plan, context)

    with pytest.raises(DynamicFoldSnapshotError, match="do not accept"):
        DynamicFoldSnapshotConfig(
            backend="memory",
            root_directory=tmp_path,
        )


def test_system_stack_accepts_explicit_disk_snapshot_config(tmp_path) -> None:
    config = DynamicFoldSnapshotConfig.disk(tmp_path)
    final_gate, fresh_gate = _bound_test_gates(
        "test.disk.whole_output",
        "e" * 64,
    )
    stack = DynamicFoldSystemStack(
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=StaticEvidenceProvider(canonical_sha256({"cache": "x"})),
        evaluator=_Accept(),
        final_evaluator=final_gate,
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: object(),
        fresh_reload_evaluator=fresh_gate,
        saver_id="test-saver",
        loader_id="test-loader",
        behavior_gate_id="test.disk.whole_output",
        behavior_gate_version=1,
        behavior_baseline_fingerprint="e" * 64,
        snapshot_config=config,
    )

    assert stack.snapshot_config is config


def test_disk_tensor_chunks_restore_bfloat_bool_and_noncontiguous(
    tmp_path,
) -> None:
    values = {
        "bf16": torch.arange(9, dtype=torch.bfloat16).reshape(3, 3),
        "bool": torch.tensor([[True, False], [False, True]]),
        "noncontiguous": torch.arange(12, dtype=torch.int64).reshape(3, 4).t(),
    }
    expected = {name: value.clone() for name, value in values.items()}
    snapshot = capture_model_state(
        tuple(values.items()),
        DynamicFoldSnapshotConfig.disk(tmp_path, chunk_bytes=3),
    )
    for value in values.values():
        value.zero_()

    snapshot.restore(values)

    for name, value in values.items():
        assert torch.equal(value, expected[name]), name
    snapshot.cleanup()
    assert snapshot_directory_inventory(tmp_path) == {}
