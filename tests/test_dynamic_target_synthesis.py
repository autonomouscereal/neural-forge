from __future__ import annotations

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.dynamic_model_inspector import (
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_target_synthesis import (
    AdapterMaterialization,
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesisError,
    DynamicTargetSynthesizer,
    StructuralBufferPolicy,
    serialized_inference_state_inventory,
    validate_registered_inference_state,
    validate_structural_runtime_buffers,
)


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


class _LocalLlamaTargetAdapter:
    adapter_id = "test.local_llama"
    version = 1
    priority = 10
    target_families = ("llama",)
    target_modalities = ("text",)
    zero_runtime_buffers = True

    def supports(self, request, parent_graphs):
        return (
            request.target_family in {"auto", "llama"}
            and request.target_modalities == ("text",)
            and any(graph.family == "llama" for graph in parent_graphs.values())
        )

    def materialize(
        self,
        request,
        parent_models,
        _parent_graphs,
        inspectors,
    ):
        parent = next(iter(parent_models.values()))
        torch.manual_seed(request.blank_seed)
        child = type(parent)(parent.config)
        if self.zero_runtime_buffers:
            with torch.no_grad():
                for buffer in child.buffers():
                    buffer.zero_()
        child.eval()
        graph = inspectors.inspect(
            child,
            model_name=request.child_name,
            tokenizer_fingerprint=request.tokenizer_fingerprint,
        )
        return AdapterMaterialization(
            model=child,
            graph=graph,
            topology_fingerprint=canonical_sha256(
                {
                    "family": graph.family,
                    "config": graph.config_fingerprint,
                }
            ),
            report={
                "local_parent_class_instantiation": True,
                "hub_loader_used": False,
            },
        )


class _StockLlamaTargetAdapter(_LocalLlamaTargetAdapter):
    adapter_id = "test.stock_transformers_llama"
    zero_runtime_buffers = False


class _PretrainedLlamaTargetAdapter(_LocalLlamaTargetAdapter):
    adapter_id = "test.pretrained_llama"

    def materialize(
        self,
        request,
        parent_models,
        _parent_graphs,
        inspectors,
    ):
        parent = next(iter(parent_models.values()))
        child = type(parent)(parent.config)
        child.load_state_dict(parent.state_dict())
        with torch.no_grad():
            for buffer in child.buffers():
                buffer.zero_()
        child.register_buffer(
            "compiler_runtime_marker",
            torch.zeros(1, dtype=torch.float32),
        )
        child.eval()
        graph = inspectors.inspect(
            child,
            model_name=request.child_name,
            tokenizer_fingerprint=request.tokenizer_fingerprint,
        )
        nonzero = sum(
            int(torch.count_nonzero(parameter.detach()).item())
            for parameter in child.parameters()
        )
        assert nonzero > 0
        return AdapterMaterialization(
            model=child,
            graph=graph,
            topology_fingerprint=canonical_sha256(
                {
                    "family": graph.family,
                    "config": graph.config_fingerprint,
                }
            ),
            report={
                "pretrained_nonzero_weights_returned": True,
                "pre_blank_nonzero_elements": nonzero,
            },
        )


class _AliasingLlamaTargetAdapter(_LocalLlamaTargetAdapter):
    def __init__(self, alias_kind: str) -> None:
        self.alias_kind = alias_kind
        self.adapter_id = f"test.aliasing_llama.{alias_kind}"

    def materialize(
        self,
        request,
        parent_models,
        _parent_graphs,
        inspectors,
    ):
        parent = next(iter(parent_models.values()))
        child = parent if self.alias_kind == "model" else type(parent)(parent.config)
        if self.alias_kind == "module":
            child.model.layers[0] = parent.model.layers[0]
        elif self.alias_kind == "parameter_object":
            child.model.layers[0].self_attn.q_proj.weight = parent.model.layers[
                0
            ].self_attn.q_proj.weight
        elif self.alias_kind == "parameter_storage":
            parent_weight = parent.model.layers[0].self_attn.q_proj.weight
            child.model.layers[0].self_attn.q_proj.weight = torch.nn.Parameter(
                parent_weight.detach(),
                requires_grad=parent_weight.requires_grad,
            )
        elif self.alias_kind == "buffer":
            child.register_buffer(
                "compiler_alias_buffer",
                parent.compiler_alias_buffer,
            )
        elif self.alias_kind == "hidden_parent_storage":
            parent_weight = parent.model.layers[0].self_attn.q_proj.weight
            child.hidden_payload = {
                "nested": [parent_weight.detach()],
            }
        child.eval()
        graph = inspectors.inspect(
            child,
            model_name=request.child_name,
            tokenizer_fingerprint=request.tokenizer_fingerprint,
        )
        return AdapterMaterialization(
            model=child,
            graph=graph,
            topology_fingerprint=canonical_sha256(
                {
                    "family": graph.family,
                    "config": graph.config_fingerprint,
                }
            ),
            report={"intentional_alias_kind": self.alias_kind},
        )


class _MaliciousBufferPolicyAdapter(_LocalLlamaTargetAdapter):
    adapter_id = "test.malicious_buffer_policy"

    def materialize(
        self,
        request,
        parent_models,
        parent_graphs,
        inspectors,
    ):
        materialized = super().materialize(
            request,
            parent_models,
            parent_graphs,
            inspectors,
        )
        materialized.model.register_buffer(
            "learned_weight",
            torch.tensor([[1.0, -2.0], [3.0, 4.0]]),
        )
        return materialized

    def structural_buffer_policy(self, _request, _materialized):
        return StructuralBufferPolicy(
            policy_id="malicious.self-certified-learned-weight",
            version=1,
            adapter_id=self.adapter_id,
            adapter_version=self.version,
            entries=(("learned_weight", "claimed_structural_geometry"),),
            source="claimed-compiler-registry",
            compiler_reviewed=True,
        )


def test_synthesizer_selects_adapter_and_materializes_exact_blank() -> None:
    parent = _model(1)
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    request = DynamicTargetRequest(
        child_name="child",
        target_family="auto",
        target_modalities=("text",),
        exact_unique_parameters=parent_graph.unique_parameter_count,
        maximum_parameter_bytes=parent_graph.unique_parameter_bytes,
        tokenizer_fingerprint=tokenizer,
        blank_seed=123,
    )
    synthesizer = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((_LocalLlamaTargetAdapter(),)),
    )

    result = synthesizer.synthesize(
        request=request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    assert result.adapter_id == "test.local_llama"
    assert result.graph.unique_parameter_count == request.exact_unique_parameters
    assert result.resource_profile.modalities == ("text",)
    assert result.as_dict()["independently_reinspected"] is True
    assert all(
        int(torch.count_nonzero(parameter.detach()).item()) == 0
        for parameter in result.model.parameters()
    )
    assert result.blank_provenance is not None
    assert result.blank_provenance.compiler_created_blank is True
    assert result.blank_provenance.parent_child_alias_free is True
    assert result.blank_provenance.zeroed_under_no_grad is True
    assert result.blank_provenance.all_parameters_zero is True
    assert result.structural_buffer_policy is not None
    assert result.structural_buffer_audit is not None
    assert result.blank_provenance.structural_buffer_policy_verified is True
    assert (
        result.blank_provenance.structural_buffer_policy_digest
        == result.structural_buffer_policy.digest
    )
    assert (
        result.blank_provenance.structural_buffer_audit_digest
        == result.structural_buffer_audit.digest
    )
    assert result.blank_provenance_digest == canonical_sha256(
        result.blank_provenance.as_dict()
    )
    serialized = result.as_dict()
    assert (
        serialized["structural_buffer_policy_digest"]
        == result.structural_buffer_policy.digest
    )
    assert (
        serialized["structural_buffer_audit_digest"]
        == result.structural_buffer_audit.digest
    )


def test_stock_transformers_llama_rotary_buffers_are_config_authenticated() -> None:
    parent = _model(10)
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    result = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((_StockLlamaTargetAdapter(),)),
    ).synthesize(
        request=DynamicTargetRequest(
            child_name="stock-llama-child",
            target_family="llama",
            target_modalities=("text",),
            exact_unique_parameters=parent_graph.unique_parameter_count,
            tokenizer_fingerprint=tokenizer,
        ),
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    assert result.structural_buffer_policy is not None
    assert result.structural_buffer_policy.compiler_reviewed is True
    assert result.structural_buffer_policy.source.startswith(
        "compiler_concrete_model_registry:"
    )
    assert result.structural_buffer_audit is not None
    assert result.structural_buffer_audit.nonzero_buffer_count == 2
    assert set(result.structural_buffer_audit.nonzero_buffer_paths) == {
        "model.rotary_emb.inv_freq",
        "model.rotary_emb.original_inv_freq",
    }


def test_synthesizer_reblanks_pretrained_weights_and_preserves_buffers() -> None:
    parent = _model(11)
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    parent_before = fingerprint_model_state(parent)
    request = DynamicTargetRequest(
        child_name="child",
        target_family="llama",
        target_modalities=("text",),
        exact_unique_parameters=parent_graph.unique_parameter_count,
        maximum_parameter_bytes=parent_graph.unique_parameter_bytes,
        tokenizer_fingerprint=tokenizer,
    )
    result = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((_PretrainedLlamaTargetAdapter(),)),
    ).synthesize(
        request=request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    assert result.adapter_report["pretrained_nonzero_weights_returned"] is True
    assert int(result.adapter_report["pre_blank_nonzero_elements"]) > 0
    assert all(
        int(torch.count_nonzero(parameter.detach()).item()) == 0
        for parameter in result.model.parameters()
    )
    assert torch.equal(
        result.model.compiler_runtime_marker,
        torch.zeros(1, dtype=torch.float32),
    )
    assert fingerprint_model_state(parent) == parent_before
    assert result.blank_state_fingerprint == fingerprint_model_state(result.model)
    assert result.blank_provenance is not None
    assert result.blank_provenance.runtime_buffers_preserved is True
    assert (
        result.blank_provenance.buffer_state_fingerprint_before
        == result.blank_provenance.buffer_state_fingerprint_after
    )
    assert (
        result.blank_provenance.unique_parameter_element_count
        == parent_graph.unique_parameter_count
    )
    serialized = result.as_dict()
    assert serialized["compiler_created_blank_verified"] is True
    assert serialized["adapter_blank_state_trusted"] is False
    assert serialized["blank_provenance_digest"] == result.blank_provenance_digest


def test_state_inventory_counts_nonpersistent_runtime_buffer_payloads() -> None:
    class RuntimeState(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
            self.register_buffer(
                "persistent_cache",
                torch.ones(3, dtype=torch.float32),
            )
            self.register_buffer(
                "ephemeral_cache",
                torch.ones(5, dtype=torch.float32),
                persistent=False,
            )
            self.metadata = {}
            self.metadata["cycle"] = self.metadata

    model = RuntimeState()
    inventory = serialized_inference_state_inventory(model)

    assert inventory["serialized_state_elements"] == 5
    assert inventory["serialized_state_bytes"] == 20
    assert inventory["persistent_buffer_elements"] == 3
    assert inventory["persistent_buffer_bytes"] == 12
    assert inventory["runtime_state_elements"] == 10
    assert inventory["runtime_state_bytes"] == 40
    assert inventory["runtime_buffer_elements"] == 8
    assert inventory["runtime_buffer_bytes"] == 32
    assert inventory["nonpersistent_buffer_elements"] == 5
    assert inventory["nonpersistent_buffer_bytes"] == 20
    assert inventory["runtime_named_buffer_count"] == 2
    assert inventory["nonpersistent_named_buffer_count"] == 1
    assert inventory["runtime_unique_buffer_view_count"] == 2
    assert inventory["runtime_tensor_state_fingerprint"] != "0" * 64
    assert "ephemeral_cache" not in model.state_dict()


def test_reachable_state_audit_rejects_large_plain_tensor_used_by_forward() -> None:
    class HiddenTensorForward(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.hidden_payload = {
                "weights": [torch.ones(1_048_576, dtype=torch.float32)]
            }

        def forward(self, values):
            return values * self.scale + self.hidden_payload["weights"][0][0]

    model = HiddenTensorForward()
    assert torch.equal(model(torch.ones(1)), torch.full((1,), 2.0))

    with pytest.raises(
        DynamicTargetSynthesisError,
        match=r"Unregistered Tensor.*hidden_payload",
    ):
        validate_registered_inference_state(model)
    with pytest.raises(DynamicTargetSynthesisError, match="Unregistered Tensor"):
        serialized_inference_state_inventory(model)


def test_reachable_state_audit_rejects_hidden_module_in_container() -> None:
    class HiddenModuleForward(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.hidden_modules = [torch.nn.Linear(1, 1, bias=False)]

        def forward(self, values):
            return self.hidden_modules[0](values) * self.scale

    with pytest.raises(
        DynamicTargetSynthesisError,
        match=r"Unregistered nn.Module.*hidden_modules\[0\]",
    ):
        validate_registered_inference_state(HiddenModuleForward())


def test_state_inventory_rejects_distinct_registered_views_of_one_storage() -> None:
    class SlicedBuffers(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            backing = torch.arange(8, dtype=torch.float32)
            self.register_buffer("left", backing[:6])
            self.register_buffer("right", backing[2:])

    with pytest.raises(
        DynamicTargetSynthesisError,
        match=r"Distinct .* tensor views share one backing storage",
    ):
        serialized_inference_state_inventory(SlicedBuffers())


def test_nonzero_learned_buffer_requires_explicit_structural_policy() -> None:
    class BufferBackedLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.register_buffer(
                "learned_weight",
                torch.tensor([[1.5, -2.0], [0.25, 3.0]]),
            )

        def forward(self, values):
            return values @ self.learned_weight

    model = BufferBackedLinear()
    expected = torch.tensor([[-1.0, 1.0]]) @ model.learned_weight
    assert torch.equal(model(torch.tensor([[-1.0, 1.0]])), expected)

    with pytest.raises(
        DynamicTargetSynthesisError,
        match=r"Nonzero runtime buffers lack an explicit structural policy.*learned_weight",
    ):
        validate_structural_runtime_buffers(model)


def test_adapter_cannot_self_certify_a_learned_buffer_as_structural() -> None:
    parent = _model(13)
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )

    with pytest.raises(
        DynamicTargetSynthesisError,
        match=r"Adapter-declared structural buffer policies are untrusted",
    ):
        DynamicTargetSynthesizer(
            inspectors=default_inspector_registry(),
            adapters=DynamicTargetAdapterRegistry((_MaliciousBufferPolicyAdapter(),)),
        ).synthesize(
            request=DynamicTargetRequest(
                child_name="malicious-child",
                target_family="llama",
                target_modalities=("text",),
                exact_unique_parameters=parent_graph.unique_parameter_count,
                tokenizer_fingerprint=tokenizer,
            ),
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )


@pytest.mark.parametrize(
    "alias_kind",
    (
        "model",
        "module",
        "parameter_object",
        "parameter_storage",
        "buffer",
        "hidden_parent_storage",
    ),
)
def test_synthesizer_rejects_every_parent_child_alias_before_zeroing(
    alias_kind: str,
) -> None:
    parent = _model(12)
    parent.register_buffer(
        "compiler_alias_buffer",
        torch.tensor([9.25], dtype=torch.float32),
    )
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    parent_before = fingerprint_model_state(parent)
    synthesizer = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry(
            (_AliasingLlamaTargetAdapter(alias_kind),)
        ),
    )

    with pytest.raises(
        DynamicTargetSynthesisError,
        match=r"aliases (?:a )?parent",
    ):
        synthesizer.synthesize(
            request=DynamicTargetRequest(
                child_name="child",
                target_family="llama",
                target_modalities=("text",),
                exact_unique_parameters=parent_graph.unique_parameter_count,
                tokenizer_fingerprint=tokenizer,
            ),
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )

    assert fingerprint_model_state(parent) == parent_before


def test_synthesizer_rejects_a_materialized_budget_mismatch() -> None:
    parent = _model(2)
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    request = DynamicTargetRequest(
        child_name="child",
        target_family="llama",
        target_modalities=("text",),
        exact_unique_parameters=parent_graph.unique_parameter_count - 1,
        tokenizer_fingerprint=tokenizer,
    )
    synthesizer = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((_LocalLlamaTargetAdapter(),)),
    )

    with pytest.raises(
        DynamicTargetSynthesisError,
        match="exact unique-parameter budget",
    ):
        synthesizer.synthesize(
            request=request,
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )
