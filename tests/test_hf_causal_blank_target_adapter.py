from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.dynamic_model_inspector import (  # noqa: E402
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_parameter_frontier import (  # noqa: E402
    CAP_ATTENTION_TOPOLOGY_PROJECTION,
    CAP_CAUSAL_DECODER_FAMILY_TRANSLATION,
    CAP_DENSE_BLANK_MATERIALIZATION,
    CAP_FFN_WIDTH_PROJECTION,
    CAP_RESIDUAL_WIDTH_PROJECTION,
    CAP_TOKENIZER_BRIDGE,
    COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
    DENSE_TOPOLOGY_CATALOG_CONFIG_KEY,
    HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY,
    HF_LLAMA_PARAMETER_ALIAS_POLICY_CONFIG_KEY,
    HF_LLAMA_SOURCE_FAMILY_CONFIG_KEY,
    HF_LLAMA_SOURCE_LAYER_COUNT_CONFIG_KEY,
    HF_LLAMA_TARGET_SCHEMA_CONFIG_KEY,
    HfLlamaDenseTargetTopologyCatalog,
    HfLlamaDenseTopology,
)
from neural_forge.dynamic_target_synthesis import (  # noqa: E402
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesisError,
    DynamicTargetSynthesizer,
)
from neural_forge.hf_causal_blank_target import (  # noqa: E402
    HFCausalBlankTargetError,
    materialize_hf_causal_blank_target,
)
from neural_forge.hf_causal_blank_target_adapter import (  # noqa: E402
    HFCausalBlankTargetAdapterError,
    HFCausalBlankTargetSynthesisAdapter,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY,
    PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY,
    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
    TOKENIZER_BRIDGE_CONFIG_KEYS,
    TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY,
    seal_strict_tokenizer_bridge_artifact,
)


LLAMA_EXACT_PARAMETERS = 1_016
GPT2_EXACT_PARAMETERS = 1_188


def _llama_configuration(**overrides):
    values = {
        "model_type": "llama",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "max_position_embeddings": 16,
        "use_cache": True,
    }
    values.update(overrides)
    return values


def _gpt2_alias_configuration():
    return {
        "architecture": "GPT2LMHeadModel",
        "vocab_size": 32,
        "n_embd": 8,
        "n_inner": 12,
        "n_layer": 1,
        "n_head": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "n_positions": 16,
        "tie_word_embeddings": False,
        "attention_bias": True,
        "mlp_bias": True,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }


def _forbid_pretrained_loaders(monkeypatch) -> None:
    def rejected(*_args, **_kwargs):
        raise AssertionError("A pretrained/network loader was called")

    monkeypatch.setattr(
        transformers.PreTrainedModel,
        "from_pretrained",
        rejected,
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        rejected,
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        rejected,
    )


@pytest.mark.parametrize(
    ("family", "configuration", "expected", "model_class"),
    [
        (
            "llama",
            _llama_configuration(),
            LLAMA_EXACT_PARAMETERS,
            "LlamaForCausalLM",
        ),
        (
            "auto",
            _gpt2_alias_configuration(),
            GPT2_EXACT_PARAMETERS,
            "GPT2LMHeadModel",
        ),
    ],
)
def test_public_materializer_builds_local_exact_zero_dense_untied_models(
    monkeypatch,
    family,
    configuration,
    expected,
    model_class,
) -> None:
    _forbid_pretrained_loaders(monkeypatch)

    result = materialize_hf_causal_blank_target(
        target_family=family,
        configuration=configuration,
        exact_unique_parameters=expected,
        seed=43,
    )

    parameters = tuple(
        result.model.named_parameters(recurse=True, remove_duplicate=False)
    )
    storages = {
        (
            parameter.device.type,
            parameter.device.index,
            parameter.untyped_storage().data_ptr(),
            parameter.untyped_storage().nbytes(),
        )
        for _name, parameter in parameters
    }
    assert type(result.model).__name__ == model_class
    assert result.logical_parameter_count == expected
    assert result.unique_parameter_count == expected
    assert result.logical_parameter_tensors == result.unique_parameter_tensors
    assert result.unique_parameter_storages == len(parameters) == len(storages)
    assert all(parameter.grad is None for _name, parameter in parameters)
    assert all(
        int(torch.count_nonzero(parameter.detach()).item()) == 0
        for _name, parameter in parameters
    )
    assert (
        result.model.get_input_embeddings().weight
        is not result.model.get_output_embeddings().weight
    )
    assert result.model.config.tie_word_embeddings is False
    assert not hasattr(result.model, "parent_model")
    if result.family == "llama":
        assert result.model.config.use_cache is True
        assert result.model.config.rms_norm_eps == pytest.approx(1e-6)
        assert result.model.config.rope_parameters["rope_theta"] == pytest.approx(
            10_000.0
        )
    else:
        assert result.model.config.attn_pdrop == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("configuration", "exact_parameters", "message"),
    [
        (
            _llama_configuration(tie_word_embeddings=True),
            LLAMA_EXACT_PARAMETERS,
            "tie_word_embeddings=false",
        ),
        (
            _llama_configuration(n_embd=16),
            LLAMA_EXACT_PARAMETERS,
            "Conflicting 'hidden_size' and alias 'n_embd'",
        ),
        (
            _llama_configuration(mystery_width=8),
            LLAMA_EXACT_PARAMETERS,
            "Unknown HF causal blank configuration keys",
        ),
        (
            _llama_configuration(),
            LLAMA_EXACT_PARAMETERS - 1,
            "misses the exact unique-parameter budget",
        ),
        (
            _llama_configuration(use_sliding_window=False),
            LLAMA_EXACT_PARAMETERS,
            "not applicable to llama: use_sliding_window",
        ),
    ],
)
def test_public_materializer_rejects_ties_ambiguity_unknowns_and_budget_miss(
    configuration,
    exact_parameters,
    message,
) -> None:
    with pytest.raises(HFCausalBlankTargetError, match=message):
        materialize_hf_causal_blank_target(
            target_family="llama",
            configuration=configuration,
            exact_unique_parameters=exact_parameters,
        )


def test_public_materializer_rejects_colliding_count_wrong_topology(
    monkeypatch,
) -> None:
    runtime_transformers = importlib.import_module("transformers")
    original_config = runtime_transformers.LlamaConfig

    def colliding_config(**kwargs):
        kwargs.update(
            intermediate_size=3,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=2,
        )
        return original_config(**kwargs)

    monkeypatch.setattr(runtime_transformers, "LlamaConfig", colliding_config)
    with pytest.raises(
        HFCausalBlankTargetError,
        match="did not preserve required intermediate_size",
    ):
        materialize_hf_causal_blank_target(
            target_family="llama",
            configuration=_llama_configuration(),
            exact_unique_parameters=LLAMA_EXACT_PARAMETERS,
        )


@pytest.mark.parametrize("key", ["rope_theta", "rms_norm_eps"])
def test_gpt2_rejects_inapplicable_rotary_and_rms_configuration(key) -> None:
    configuration = {**_gpt2_alias_configuration(), key: 1e-6}
    with pytest.raises(
        HFCausalBlankTargetError,
        match=rf"not applicable to gpt2: {key}",
    ):
        materialize_hf_causal_blank_target(
            target_family="gpt2",
            configuration=configuration,
            exact_unique_parameters=GPT2_EXACT_PARAMETERS,
        )


def _qwen3_parent(*, vocab_size: int = 32):
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=16,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
            attention_bias=False,
        )
    )


class _BridgeTokenizer:
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
            f"raw-{index}-{surface}": index
            for index, surface in enumerate(self.surfaces)
        }

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"raw-{token_id}-{self.surfaces[token_id]}"

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.surfaces[token_id] for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [self.surfaces.index(text)]


def _bridge_artifact(tmp_path: Path):
    donor = _BridgeTokenizer([f"<{index}>" for index in range(32)])
    target = _BridgeTokenizer(donor.surfaces[:16])
    root = tmp_path / "target-tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        root,
        donor_row_capacity=32,
        target_row_capacity=16,
    )


def test_adapter_synthesizes_llama_from_different_parent_family_without_loader(
    monkeypatch,
) -> None:
    if not hasattr(transformers, "Qwen3ForCausalLM"):
        pytest.skip("Transformers build has no Qwen3 support")
    parent = _qwen3_parent()
    parent_parameters = tuple(parent.parameters())
    parent_storage = {
        parameter.untyped_storage().data_ptr() for parameter in parent_parameters
    }
    _forbid_pretrained_loaders(monkeypatch)
    configuration = _llama_configuration()
    request = DynamicTargetRequest(
        child_name="cross-family-llama-child",
        target_family="llama",
        target_modalities=("text",),
        exact_unique_parameters=LLAMA_EXACT_PARAMETERS,
        adapter_id="neural_forge.hf.causal_dense_blank",
        blank_seed=47,
        configuration=tuple(configuration.items()),
    )

    result = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry(
            (HFCausalBlankTargetSynthesisAdapter(),)
        ),
    ).synthesize(
        request=request,
        parent_models={"qwen-parent": parent},
    )

    report = dict(result.adapter_report)
    child_parameters = tuple(
        result.model.named_parameters(recurse=True, remove_duplicate=False)
    )
    assert result.graph.family == "llama"
    assert result.graph.logical_parameter_count == LLAMA_EXACT_PARAMETERS
    assert result.graph.unique_parameter_count == LLAMA_EXACT_PARAMETERS
    assert result.blank_provenance is not None
    assert result.blank_provenance.compiler_created_blank is True
    assert result.blank_provenance.parent_child_alias_free is True
    assert result.blank_provenance.all_parameters_zero is True
    assert result.topology_provenance_verified is True
    assert report["parent_family_independent"] is True
    assert report["pretrained_load_calls"] == 0
    assert report["network_loader_used"] is False
    assert report["parent_weights_read"] is False
    assert report["parameter_sharing_used"] is False
    assert report["training_used"] is False
    assert report["pruning_used"] is False
    assert report["quantization_used"] is False
    assert report["sparsity_used"] is False
    assert report["sidecar_used"] is False
    assert report["compiler_metadata_key_count"] == 0
    assert all(
        parameter.untyped_storage().data_ptr() not in parent_storage
        for _name, parameter in child_parameters
    )
    assert all(
        int(torch.count_nonzero(parameter.detach()).item()) == 0
        for _name, parameter in child_parameters
    )


def _exact_llama_catalog_candidate(parent_graph):
    catalog = HfLlamaDenseTargetTopologyCatalog(
        minimum_hidden_size=8,
        hidden_size_multiple=8,
        minimum_intermediate_size=4,
        intermediate_size_multiple=4,
        minimum_head_dim=4,
        head_dim_multiple=2,
        compiler_capabilities=(
            CAP_DENSE_BLANK_MATERIALIZATION,
            CAP_CAUSAL_DECODER_FAMILY_TRANSLATION,
            CAP_RESIDUAL_WIDTH_PROJECTION,
            CAP_FFN_WIDTH_PROJECTION,
            CAP_ATTENTION_TOPOLOGY_PROJECTION,
        ),
    )
    return catalog.enumerate_frontier(
        parent_graph,
        requested_unique_parameters=LLAMA_EXACT_PARAMETERS,
        target_vocabulary_size=32,
        match_policy="exact",
        maximum_candidates=8,
    )


def _changed_vocab_catalog_candidate(artifact, parent_graph):
    topology = HfLlamaDenseTopology(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    catalog = HfLlamaDenseTargetTopologyCatalog(
        minimum_hidden_size=8,
        hidden_size_multiple=8,
        minimum_intermediate_size=4,
        intermediate_size_multiple=4,
        minimum_head_dim=4,
        head_dim_multiple=2,
        compiler_capabilities=(
            CAP_DENSE_BLANK_MATERIALIZATION,
            CAP_CAUSAL_DECODER_FAMILY_TRANSLATION,
            CAP_RESIDUAL_WIDTH_PROJECTION,
            CAP_FFN_WIDTH_PROJECTION,
            CAP_ATTENTION_TOPOLOGY_PROJECTION,
        ),
        tokenizer_bridge_artifact=artifact,
    )
    frontier = catalog.enumerate_frontier(
        parent_graph,
        requested_unique_parameters=topology.unique_parameters,
        target_vocabulary_size=topology.vocab_size,
        match_policy="exact",
        maximum_candidates=8,
    )
    assert len(frontier.candidates) == 1
    candidate = frontier.candidates[0]
    assert candidate.tokenizer_bridge_required is True
    assert CAP_TOKENIZER_BRIDGE in candidate.declared_compiler_capabilities
    assert candidate.compiler_executable is True
    return artifact, candidate


def test_changed_vocab_catalog_candidate_materializes_through_synthesizer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = _bridge_artifact(tmp_path)
    parent = _qwen3_parent(vocab_size=32)
    parent_graph = inspect_model_circuits(
        parent,
        model_name="qwen-parent",
        tokenizer_fingerprint=artifact.donor_tokenizer_semantics_sha256,
    )
    artifact, candidate = _changed_vocab_catalog_candidate(
        artifact,
        parent_graph,
    )
    assert candidate.tokenizer_bridge_required is True
    assert CAP_TOKENIZER_BRIDGE in candidate.declared_compiler_capabilities
    _forbid_pretrained_loaders(monkeypatch)
    request = DynamicTargetRequest(
        child_name="changed-vocab-llama-child",
        target_family=candidate.family,
        target_modalities=candidate.modalities,
        exact_unique_parameters=candidate.unique_parameters,
        tokenizer_fingerprint=artifact.target_tokenizer_semantics_sha256,
        adapter_id="neural_forge.hf.causal_dense_blank",
        configuration=candidate.configuration,
        component_allocation_receipt=candidate.component_allocation_receipt,
    )

    result = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry(
            (HFCausalBlankTargetSynthesisAdapter(),)
        ),
    ).synthesize(
        request=request,
        parent_models={"qwen-parent": parent},
        parent_tokenizer_fingerprints={
            "qwen-parent": artifact.donor_tokenizer_semantics_sha256
        },
    )

    report = dict(result.adapter_report)
    assert result.graph.family == "llama"
    assert result.graph.unique_parameter_count == candidate.unique_parameters
    assert result.graph.tokenizer_fingerprint == (
        artifact.target_tokenizer_semantics_sha256
    )
    assert report["compiler_metadata_key_count"] == 13
    assert report["tokenizer_bridge_required"] is True
    assert report["tokenizer_bridge_authenticated"] is True
    assert report["tokenizer_bridge_artifact_sha256"] == artifact.artifact_sha256
    assert report["tokenizer_bridge_profile_sha256"] == artifact.profile_fingerprint
    assert report["parent_tokenizer_semantics_sha256"] == (
        artifact.donor_tokenizer_semantics_sha256
    )
    assert report["child_tokenizer_semantics_sha256"] == (
        artifact.target_tokenizer_semantics_sha256
    )


def test_changed_vocab_bridge_provenance_rejects_tampering(
    tmp_path: Path,
) -> None:
    artifact = _bridge_artifact(tmp_path)
    parent = _qwen3_parent(vocab_size=32)
    parent_graph = inspect_model_circuits(
        parent,
        model_name="qwen-parent",
        tokenizer_fingerprint=artifact.donor_tokenizer_semantics_sha256,
    )
    artifact, candidate = _changed_vocab_catalog_candidate(
        artifact,
        parent_graph,
    )
    receipt = candidate.component_allocation_receipt
    assert receipt is not None
    base = dict(candidate.configuration)
    cases: list[tuple[str, dict[str, object], str, str]] = []

    incomplete = dict(base)
    incomplete.pop(TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY)
    cases.append(("incomplete", incomplete, "incomplete", artifact.target_tokenizer_semantics_sha256))
    absent = {
        key: value for key, value in base.items() if key not in TOKENIZER_BRIDGE_CONFIG_KEYS
    }
    cases.append(("absent", absent, "changed-vocabulary provenance", artifact.target_tokenizer_semantics_sha256))
    wrong_artifact = dict(base)
    wrong_artifact[TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY] = "a" * 64
    cases.append(("artifact", wrong_artifact, "not bound to the receipt topology", artifact.target_tokenizer_semantics_sha256))
    wrong_profile = dict(base)
    wrong_profile[TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY] = "b" * 64
    cases.append(("profile", wrong_profile, "not bound to the receipt topology", artifact.target_tokenizer_semantics_sha256))
    wrong_parent = dict(base)
    wrong_parent[PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY] = "c" * 64
    cases.append(("parent", wrong_parent, "parent semantics", artifact.target_tokenizer_semantics_sha256))
    wrong_child = dict(base)
    wrong_child[CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY] = "d" * 64
    cases.append(("child", wrong_child, "child semantics", artifact.target_tokenizer_semantics_sha256))
    unknown = dict(base)
    unknown["_neural_forge_tokenizer_bridge_artfact_sha256"] = "e" * 64
    cases.append(("unknown", unknown, "Unknown HF causal compiler-provenance key", artifact.target_tokenizer_semantics_sha256))
    cases.append(("request-child", base, "child semantics", "f" * 64))

    adapter = HFCausalBlankTargetSynthesisAdapter()
    for label, configuration, message, child_semantics in cases:
        request = SimpleNamespace(
            child_name=f"invalid-bridge-{label}",
            target_family="llama",
            target_modalities=("text",),
            exact_unique_parameters=candidate.unique_parameters,
            blank_seed=0,
            tokenizer_fingerprint=child_semantics,
            configuration=tuple(configuration.items()),
            component_allocation_receipt=receipt,
            fingerprint="0" * 64,
        )
        assert adapter.supports(request, {"qwen-parent": parent_graph}) is False
        with pytest.raises(HFCausalBlankTargetAdapterError, match=message):
            adapter.materialize(
                request,
                {"qwen-parent": parent},
                {"qwen-parent": parent_graph},
                default_inspector_registry(),
            )


def test_catalog_candidate_receipt_materializes_through_unchanged_synthesizer(
    monkeypatch,
) -> None:
    if not hasattr(transformers, "Qwen3ForCausalLM"):
        pytest.skip("Transformers build has no Qwen3 support")
    parent = _qwen3_parent()
    parent_graph = inspect_model_circuits(parent, model_name="qwen-parent")
    frontier = _exact_llama_catalog_candidate(parent_graph)
    assert len(frontier.candidates) == 1
    candidate = frontier.candidates[0]
    assert candidate.component_allocation_receipt is not None
    assert candidate.compiler_executable is True
    assert (
        dict(candidate.configuration)[HF_LLAMA_SOURCE_LAYER_COUNT_CONFIG_KEY]
        == len(parent_graph.layer_indices)
        == 1
    )
    _forbid_pretrained_loaders(monkeypatch)
    request = DynamicTargetRequest(
        child_name="catalog-cross-family-llama-child",
        target_family=candidate.family,
        target_modalities=candidate.modalities,
        exact_unique_parameters=candidate.unique_parameters,
        adapter_id="neural_forge.hf.causal_dense_blank",
        configuration=candidate.configuration,
        component_allocation_receipt=candidate.component_allocation_receipt,
    )

    result = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry(
            (HFCausalBlankTargetSynthesisAdapter(),)
        ),
    ).synthesize(
        request=request,
        parent_models={"qwen-parent": parent},
    )

    report = dict(result.adapter_report)
    assert result.graph.family == candidate.family == "llama"
    assert result.graph.unique_parameter_count == candidate.unique_parameters
    assert result.graph.logical_parameter_count == candidate.logical_parameters
    assert report["component_allocation_verified_by_target_adapter"] is True
    assert (
        report["component_allocation_receipt_sha256"]
        == candidate.component_allocation_receipt.fingerprint
    )
    assert (
        report["component_allocation_topology_sha256"]
        == candidate.component_allocation_receipt.topology_configuration_sha256
    )


def test_adapter_rejects_unbound_or_malformed_compiler_provenance() -> None:
    parent = _qwen3_parent()
    parent_graph = inspect_model_circuits(parent, model_name="qwen-parent")
    candidate = _exact_llama_catalog_candidate(parent_graph).candidates[0]
    receipt = candidate.component_allocation_receipt
    assert receipt is not None
    base = dict(candidate.configuration)
    cases: list[tuple[str, dict[str, object], str]] = []

    unknown = dict(base)
    unknown["_neural_forge_hf_llama_target_schemma"] = "typo"
    cases.append(("unknown", unknown, "Unknown HF causal compiler-provenance key"))
    wrong_catalog = dict(base)
    wrong_catalog[DENSE_TOPOLOGY_CATALOG_CONFIG_KEY] = "other-catalog"
    cases.append(("catalog", wrong_catalog, "catalog/schema/parameter-alias"))
    wrong_schema = dict(base)
    wrong_schema[HF_LLAMA_TARGET_SCHEMA_CONFIG_KEY] = "other-schema"
    cases.append(("schema", wrong_schema, "catalog/schema/parameter-alias"))
    tied = dict(base)
    tied[HF_LLAMA_PARAMETER_ALIAS_POLICY_CONFIG_KEY] = "tied"
    cases.append(("tied", tied, "catalog/schema/parameter-alias"))
    wrong_spans = dict(base)
    wrong_spans[HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY] = "0;1"
    cases.append(("spans", wrong_spans, "source spans/layer count disagree"))
    wrong_layer_count = dict(base)
    wrong_layer_count[HF_LLAMA_SOURCE_LAYER_COUNT_CONFIG_KEY] = 2
    cases.append(
        (
            "source-layer-count",
            wrong_layer_count,
            "source spans/layer count disagree",
        )
    )
    wrong_family = dict(base)
    wrong_family[HF_LLAMA_SOURCE_FAMILY_CONFIG_KEY] = "llama"
    cases.append(("source-family", wrong_family, "source-family provenance"))
    wrong_receipt = dict(base)
    wrong_receipt[COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY] = "0" * 64
    cases.append(("receipt", wrong_receipt, "receipt seals"))
    missing_schema = dict(base)
    missing_schema.pop(HF_LLAMA_TARGET_SCHEMA_CONFIG_KEY)
    cases.append(("partial", missing_schema, "complete catalog and receipt"))

    adapter = HFCausalBlankTargetSynthesisAdapter()
    for label, configuration, message in cases:
        request = SimpleNamespace(
            child_name=f"invalid-{label}",
            target_family="llama",
            target_modalities=("text",),
            exact_unique_parameters=candidate.unique_parameters,
            blank_seed=0,
            tokenizer_fingerprint=None,
            configuration=tuple(configuration.items()),
            component_allocation_receipt=receipt,
            fingerprint="0" * 64,
        )
        assert adapter.supports(request, {"qwen-parent": parent_graph}) is False
        with pytest.raises(HFCausalBlankTargetAdapterError, match=message):
            adapter.materialize(
                request,
                {"qwen-parent": parent},
                {"qwen-parent": parent_graph},
                default_inspector_registry(),
            )


def test_explicit_adapter_rejects_unknown_noncompiler_configuration() -> None:
    request = DynamicTargetRequest(
        child_name="invalid-child",
        target_family="llama",
        target_modalities=("text",),
        exact_unique_parameters=LLAMA_EXACT_PARAMETERS,
        adapter_id="neural_forge.hf.causal_dense_blank",
        configuration=tuple(
            _llama_configuration(unknown_shape_key=7).items()
        ),
    )
    synthesizer = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry(
            (HFCausalBlankTargetSynthesisAdapter(),)
        ),
    )

    with pytest.raises(
        DynamicTargetSynthesisError,
        match="does not support the requested family/modality/source combination",
    ):
        synthesizer.synthesize(
            request=request,
            parent_models={"qwen-parent": _qwen3_parent()},
        )
