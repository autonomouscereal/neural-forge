from __future__ import annotations

import hashlib

import pytest

from neural_forge.dynamic_behavior_gate import (
    BehaviorGateBinding,
    expose_behavior_gate_binding,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import FoldDecision, fingerprint_model_state
from neural_forge.dynamic_qwen_adapter import (
    GLOBAL_CODEBOOK_METHOD,
    Qwen3CodebookStackFactory,
    Qwen3CodebookTargetSynthesisAdapter,
    build_qwen3_codebook_tokenizer_contract,
    build_qwen3_global_codebook_fold_cache,
    build_qwen3_global_codebook_fold_cache_from_checkpoint,
)
from neural_forge.dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tuple(tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[token_id]


def _parent(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=12,
            hidden_size=8,
            intermediate_size=6,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    ).eval()
    with torch.no_grad():
        source = model.model.layers[0].self_attn
        target = model.model.layers[1].self_attn
        for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(target, role).weight.copy_(getattr(source, role).weight)
    return model


def _tokenizers() -> tuple[_Tokenizer, _Tokenizer]:
    parent = _Tokenizer([f"token-{index}" for index in range(12)])
    target = _Tokenizer(
        [
            "token-0",
            "token-1",
            "token-2",
            "token-5",
            "token-7",
            "token-9",
            "token-11",
        ]
    )
    return parent, target


class _CachedWholeOutputGate:
    def __init__(self, input_ids: torch.Tensor, expected_logits: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.expected_logits = expected_logits
        self.call_count = 0

    def evaluate(self, context) -> FoldDecision:
        self.call_count += 1
        with torch.no_grad():
            actual = context.execution_context.child_model(
                self.input_ids,
                use_cache=False,
            ).logits
        accepted = bool(
            torch.allclose(
                actual,
                self.expected_logits,
                rtol=1e-5,
                atol=1e-6,
            )
        )
        return FoldDecision(
            accepted,
            "cached parent whole-output logits",
            (("cached_parent_logits_match", accepted),),
        )


def _fresh_reload_gate(
    input_ids: torch.Tensor,
    expected_logits: torch.Tensor,
):
    def evaluate(model) -> FoldDecision:
        with torch.no_grad():
            actual = model(input_ids, use_cache=False).logits
        accepted = bool(
            torch.allclose(
                actual,
                expected_logits,
                rtol=1e-5,
                atol=1e-6,
            )
        )
        return FoldDecision(accepted, "fresh cached parent logits")

    return evaluate


def _catalog(
    *,
    parent_tokenizer: _Tokenizer,
    target_tokenizer: _Tokenizer,
    cache,
    final_evaluator,
    fresh_reload_evaluator,
    tokenizer_contract,
) -> AutomaticDynamicFoldingSystem:
    manifest_body = {
        "schema": "test-behavior-gate-manifest-v1",
        "gate_id": "test.qwen3_codebook.whole_output",
        "gate_version": 1,
        "baseline_fingerprint": cache.fingerprint,
    }
    manifest_fingerprint = canonical_sha256(manifest_body)
    behavior_binding = BehaviorGateBinding(
        gate_id="test.qwen3_codebook.whole_output",
        gate_version=1,
        baseline_fingerprint=cache.fingerprint,
        manifest_fingerprint=manifest_fingerprint,
        manifest={
            **manifest_body,
            "manifest_fingerprint": manifest_fingerprint,
        },
    )
    expose_behavior_gate_binding(final_evaluator, behavior_binding)
    expose_behavior_gate_binding(fresh_reload_evaluator, behavior_binding)
    target_adapter = Qwen3CodebookTargetSynthesisAdapter(
        parent_tokenizer=parent_tokenizer,
        target_tokenizer=target_tokenizer,
        attention_prototypes_per_group=1,
        neuron_prototypes_per_neuron=2,
        fold_cache=cache,
    )
    stack_adapter = Qwen3CodebookStackFactory(
        cache=cache,
        final_evaluator=final_evaluator,
        tokenizer_contract=tokenizer_contract,
        fresh_reload_evaluator=fresh_reload_evaluator,
    )
    return AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((target_adapter,)),
        stack_adapters=DynamicFoldStackAdapterRegistry((stack_adapter,)),
    )


def test_automatic_qwen_global_codebook_fold_reuses_checkpoint_and_erases_seed(
    tmp_path,
) -> None:
    parent = _parent(9001)
    parent_tokenizer, target_tokenizer = _tokenizers()
    parent_state = fingerprint_model_state(parent)
    parent_tokenizer_fingerprint = canonical_sha256({"tokenizer": "tiny-qwen-parent"})
    target_tokenizer_fingerprint = canonical_sha256({"tokenizer": "tiny-qwen-target"})
    fresh_cache = build_qwen3_global_codebook_fold_cache(
        source_parent="parent",
        parent_model=parent,
        parent_tokenizer=parent_tokenizer,
        target_tokenizer=target_tokenizer,
        attention_prototypes_per_group=1,
        neuron_prototypes_per_neuron=2,
    )
    parent_rows = dict(fresh_cache.topology_buffers)["parent_token_rows"].to(
        dtype=torch.long
    )
    child_input_ids = torch.tensor([[1, 3, 4, 6]], dtype=torch.long)
    with torch.no_grad():
        expected_logits = parent(
            parent_rows[child_input_ids],
            use_cache=False,
        ).logits.index_select(-1, parent_rows)
    tokenizer_contract = build_qwen3_codebook_tokenizer_contract(
        target_tokenizer=target_tokenizer,
        cache=fresh_cache,
        tokenizer_name_or_path="unit-test-qwen-target",
    )
    first_gate = _CachedWholeOutputGate(child_input_ids, expected_logits)
    first_system = _catalog(
        parent_tokenizer=parent_tokenizer,
        target_tokenizer=target_tokenizer,
        cache=fresh_cache,
        final_evaluator=first_gate,
        fresh_reload_evaluator=_fresh_reload_gate(
            child_input_ids,
            expected_logits,
        ),
        tokenizer_contract=tokenizer_contract,
    )

    first = first_system.run(
        parent_models={"parent": parent},
        target_request=DynamicTargetRequest(
            child_name="automatic-qwen-codebook-child",
            target_family="auto",
            target_modalities=("text",),
            exact_unique_parameters=fresh_cache.expected_unique_parameters,
            tokenizer_fingerprint=target_tokenizer_fingerprint,
            blank_seed=101,
        ),
        evidence_fingerprint=fresh_cache.fingerprint,
        destination=tmp_path / "first",
        parent_tokenizer_fingerprints={
            "parent": parent_tokenizer_fingerprint,
        },
    )

    assert first.complete
    assert first_gate.call_count == 1
    assert len(first.published.compilation.plan.actions) == 1
    action = first.published.compilation.plan.actions[0]
    assert action.method_id == GLOBAL_CODEBOOK_METHOD
    assert set(action.target_node_ids) == {
        node.node_id for node in first.target.graph.nodes
    }
    assert set(action.target_parameter_paths) == set(first.target.graph.parameter_paths)
    attempt = first.published.compilation.run.attempts[0]
    assert attempt.transaction_report["seed_erasure_by_construction"] is True
    assert attempt.transaction_report["cached_parent_forward_count"] == 0
    assert first.published.compilation.run.optimizer_steps == 0
    assert first.published.compilation.run.backward_calls == 0
    assert first.published.compilation.run.autograd_grad_calls == 0
    assert first.published.artifact.fresh_reload_decision.accepted
    assert {path.name for path in (tmp_path / "first" / "child").iterdir()} == {
        "config.json",
        "manifest.json",
        "model.safetensors",
        "tokenizer_contract.json",
    }
    assert fingerprint_model_state(parent) == parent_state

    manifest_path = tmp_path / "first" / "child" / "manifest.json"
    reused_cache = build_qwen3_global_codebook_fold_cache_from_checkpoint(
        checkpoint=tmp_path / "first" / "child",
        source_parent="parent",
        live_parent_model=parent,
        attention_prototypes_per_group=1,
        neuron_prototypes_per_neuron=2,
        tokenizer_contract=tokenizer_contract,
        expected_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    assert reused_cache.parameter_fingerprint == (fresh_cache.parameter_fingerprint)
    assert reused_cache.topology_fingerprint == (fresh_cache.topology_fingerprint)
    assert reused_cache.fingerprint == fresh_cache.fingerprint

    second_gate = _CachedWholeOutputGate(child_input_ids, expected_logits)
    second_system = _catalog(
        parent_tokenizer=parent_tokenizer,
        target_tokenizer=target_tokenizer,
        cache=reused_cache,
        final_evaluator=second_gate,
        fresh_reload_evaluator=_fresh_reload_gate(
            child_input_ids,
            expected_logits,
        ),
        tokenizer_contract=tokenizer_contract,
    )
    second = second_system.run(
        parent_models={"parent": parent},
        target_request=DynamicTargetRequest(
            child_name="automatic-qwen-codebook-child",
            target_family="auto",
            target_modalities=("text",),
            exact_unique_parameters=reused_cache.expected_unique_parameters,
            tokenizer_fingerprint=target_tokenizer_fingerprint,
            blank_seed=202,
        ),
        evidence_fingerprint=reused_cache.fingerprint,
        destination=tmp_path / "second",
        parent_tokenizer_fingerprints={
            "parent": parent_tokenizer_fingerprint,
        },
    )

    assert second.complete
    assert second_gate.call_count == 1
    assert first.target.adapter_report["blank_state_fingerprint"] == (
        first.target.blank_state_fingerprint
    )
    assert second.target.adapter_report["blank_state_fingerprint"] == (
        second.target.blank_state_fingerprint
    )
    assert first.target.blank_state_fingerprint == second.target.blank_state_fingerprint
    assert first.target.blank_provenance is not None
    assert second.target.blank_provenance is not None
    assert first.target.blank_provenance.compiler_created_blank is True
    assert second.target.blank_provenance.compiler_created_blank is True
    assert first.published.artifact.child_state_fingerprint == (
        second.published.artifact.child_state_fingerprint
    )
    assert fingerprint_model_state(parent) == parent_state
