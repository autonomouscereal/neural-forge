from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from neural_forge.dynamic_circuit_ir import (
    CircuitKind,
    CircuitNode,
    CircuitPort,
    ModalityInterface,
    ModelCircuitGraph,
    ParameterSpec,
    PortKind,
    canonical_sha256,
)
from neural_forge.dynamic_parameter_frontier import (
    CAP_CAUSAL_DECODER_FAMILY_TRANSLATION,
    CAP_DENSE_BLANK_MATERIALIZATION,
    CAP_TOKENIZER_BRIDGE,
    COMPONENT_ALLOCATION_POLICY,
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    DynamicParameterFrontierError,
    HF_LLAMA_CATALOG_ID,
    HF_LLAMA_PARAMETER_ALIAS_POLICY_CONFIG_KEY,
    HF_LLAMA_PARAMETER_ALIAS_POLICY_NONE,
    HF_LLAMA_SOURCE_LAYER_COUNT_CONFIG_KEY,
    HF_LLAMA_TARGET_SCHEMA,
    HF_LLAMA_TARGET_SCHEMA_CONFIG_KEY,
    HfLlamaDenseTargetTopologyCatalog,
    HfLlamaDenseTopology,
    component_allocation_topology_fingerprint,
    synthesize_dense_parameter_frontier,
)
from neural_forge.hf_tokenizer_bridge import (
    CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY,
    PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY,
    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
    TOKENIZER_BRIDGE_CONFIG_KEYS,
    TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY,
    seal_strict_tokenizer_bridge_artifact,
)


VOCAB_SIZE = 49_152
SOURCE_HIDDEN_SIZE = 896
SOURCE_INTERMEDIATE_SIZE = 3_584
SOURCE_LAYER_COUNT = 36
SOURCE_QUERY_HEADS = 14
SOURCE_KEY_VALUE_HEADS = 2
HEAD_DIM = 64
EXACT_TARGET_PARAMETERS = 147_292_096


def _parameter(
    path: str,
    role: str,
    shape: tuple[int, ...],
    *,
    storage_group: str | None = None,
) -> ParameterSpec:
    return ParameterSpec(
        path=path,
        role=role,
        shape=shape,
        dtype="torch.float32",
        storage_group=storage_group or path,
        trainable=True,
    )


def _dense_qwen_source_graph(
    *,
    vocab_size: int = VOCAB_SIZE,
    hidden_size: int = SOURCE_HIDDEN_SIZE,
    intermediate_size: int = SOURCE_INTERMEDIATE_SIZE,
    layer_count: int = SOURCE_LAYER_COUNT,
    query_heads: int = SOURCE_QUERY_HEADS,
    key_value_heads: int = SOURCE_KEY_VALUE_HEADS,
    head_dim: int = HEAD_DIM,
    tokenizer_fingerprint: str | None = None,
) -> ModelCircuitGraph:
    """Weight-free IR shaped like one inspected, untied dense Qwen decoder."""

    nodes: list[CircuitNode] = []

    def add(
        node_id: str,
        kind: CircuitKind,
        *,
        layer_index: int | None,
        parameters: tuple[ParameterSpec, ...] = (),
        attributes: tuple[tuple[str, object], ...] = (),
        inputs: tuple[CircuitPort, ...] = (),
        outputs: tuple[CircuitPort, ...] = (),
    ) -> None:
        nodes.append(
            CircuitNode(
                node_id=node_id,
                kind=kind,
                order=len(nodes),
                path=node_id,
                layer_index=layer_index,
                inputs=inputs,
                outputs=outputs,
                parameters=parameters,
                attributes=attributes,
            )
        )

    add(
        "input.token_embedding",
        CircuitKind.TOKEN_EMBEDDING,
        layer_index=None,
        parameters=(
            _parameter(
                "model.embed_tokens.weight",
                "embedding.weight",
                (vocab_size, hidden_size),
            ),
        ),
        attributes=(
            ("vocabulary_size", vocab_size),
            ("hidden_size", hidden_size),
        ),
        inputs=(CircuitPort("token_ids", PortKind.TOKEN_IDS),),
        outputs=(
            CircuitPort(
                "residual",
                PortKind.RESIDUAL,
                ("batch", "sequence", hidden_size),
                modality="text",
            ),
        ),
    )
    add(
        "input.position",
        CircuitKind.POSITION_ENCODING,
        layer_index=None,
        attributes=(
            ("kind", "rope"),
            ("rope_theta", 1_000_000.0),
            ("max_positions", 32_768),
        ),
    )
    query_width = query_heads * head_dim
    key_value_width = key_value_heads * head_dim
    for index in range(layer_count):
        prefix = f"layer.{index:03d}"
        add(
            f"{prefix}.attention_norm",
            CircuitKind.NORM,
            layer_index=index,
            parameters=(
                _parameter(
                    f"{prefix}.attention_norm.weight",
                    "attention_norm.weight",
                    (hidden_size,),
                ),
            ),
            attributes=(("epsilon", 1e-6),),
        )
        add(
            f"{prefix}.attention",
            CircuitKind.ATTENTION,
            layer_index=index,
            parameters=(
                _parameter(
                    f"{prefix}.attention.q_proj.weight",
                    "attention.q_proj.weight",
                    (query_width, hidden_size),
                ),
                _parameter(
                    f"{prefix}.attention.k_proj.weight",
                    "attention.k_proj.weight",
                    (key_value_width, hidden_size),
                ),
                _parameter(
                    f"{prefix}.attention.v_proj.weight",
                    "attention.v_proj.weight",
                    (key_value_width, hidden_size),
                ),
                _parameter(
                    f"{prefix}.attention.o_proj.weight",
                    "attention.o_proj.weight",
                    (hidden_size, query_width),
                ),
                _parameter(
                    f"{prefix}.attention.q_norm.weight",
                    "attention.q_norm.weight",
                    (head_dim,),
                ),
                _parameter(
                    f"{prefix}.attention.k_norm.weight",
                    "attention.k_norm.weight",
                    (head_dim,),
                ),
            ),
            attributes=(
                ("projection_style", "split_qkv"),
                ("query_heads", query_heads),
                ("key_value_heads", key_value_heads),
                ("head_dim", head_dim),
            ),
        )
        add(
            f"{prefix}.mlp_norm",
            CircuitKind.NORM,
            layer_index=index,
            parameters=(
                _parameter(
                    f"{prefix}.mlp_norm.weight",
                    "mlp_norm.weight",
                    (hidden_size,),
                ),
            ),
            attributes=(("epsilon", 1e-6),),
        )
        add(
            f"{prefix}.mlp",
            CircuitKind.GATED_MLP,
            layer_index=index,
            parameters=(
                _parameter(
                    f"{prefix}.mlp.gate_proj.weight",
                    "mlp.gate_proj.weight",
                    (intermediate_size, hidden_size),
                ),
                _parameter(
                    f"{prefix}.mlp.up_proj.weight",
                    "mlp.up_proj.weight",
                    (intermediate_size, hidden_size),
                ),
                _parameter(
                    f"{prefix}.mlp.down_proj.weight",
                    "mlp.down_proj.weight",
                    (hidden_size, intermediate_size),
                ),
            ),
            attributes=(
                ("intermediate_size", intermediate_size),
                ("activation", "silu"),
            ),
        )
    add(
        "output.final_norm",
        CircuitKind.NORM,
        layer_index=None,
        parameters=(
            _parameter(
                "model.norm.weight",
                "final_norm.weight",
                (hidden_size,),
            ),
        ),
        attributes=(("epsilon", 1e-6),),
    )
    add(
        "output.lm_head",
        CircuitKind.OUTPUT_HEAD,
        layer_index=None,
        parameters=(
            _parameter(
                "lm_head.weight",
                "output_head.weight",
                (vocab_size, hidden_size),
            ),
        ),
        attributes=(
            ("vocabulary_size", vocab_size),
            ("tied_to_embedding", False),
        ),
        inputs=(
            CircuitPort(
                "residual",
                PortKind.NORMALIZED_RESIDUAL,
                ("batch", "sequence", hidden_size),
            ),
        ),
        outputs=(
            CircuitPort(
                "logits",
                PortKind.LOGITS,
                ("batch", "sequence", vocab_size),
                modality="text",
            ),
        ),
    )
    logical = sum(node.logical_parameter_count for node in nodes)
    return ModelCircuitGraph(
        model_name="synthetic-dense-qwen",
        family="qwen3",
        model_class="test.SyntheticDenseQwen",
        nodes=tuple(nodes),
        edges=(),
        interfaces=(
            ModalityInterface(
                interface_id="text.tokens.input",
                modality="text",
                direction="input",
                node_id="input.token_embedding",
                port_name="token_ids",
                representation="token_ids",
            ),
            ModalityInterface(
                interface_id="text.logits.output",
                modality="text",
                direction="output",
                node_id="output.lm_head",
                port_name="logits",
                representation="token_logits",
            ),
        ),
        logical_parameter_count=logical,
        unique_parameter_count=logical,
        unique_parameter_bytes=logical * 4,
        config_fingerprint=canonical_sha256({"fixture": "dense-qwen"}),
        tokenizer_fingerprint=tokenizer_fingerprint,
        metadata=(
            ("inspector", "hf-causal-decoder"),
            ("hidden_size", hidden_size),
            ("layer_count", layer_count),
            ("tie_word_embeddings", False),
        ),
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
    donor = _BridgeTokenizer([f"<{index}>" for index in range(16)])
    target = _BridgeTokenizer(donor.surfaces[:8])
    root = tmp_path / "target-tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        root,
        donor_row_capacity=16,
        target_row_capacity=8,
    )


def test_dense_llama_exact_unique_parameter_accounting() -> None:
    topology = HfLlamaDenseTopology(
        vocab_size=49_152,
        hidden_size=448,
        intermediate_size=1_792,
        num_hidden_layers=36,
        num_attention_heads=7,
        num_key_value_heads=1,
        head_dim=64,
    )

    assert topology.token_interface_parameters == 44_040_192
    assert topology.attention_parameters_per_layer == 458_752
    assert topology.mlp_parameters_per_layer == 2_408_448
    assert topology.normalization_parameters == 32_704
    assert topology.unique_parameters == EXACT_TARGET_PARAMETERS
    assert topology.logical_parameters == EXACT_TARGET_PARAMETERS


@pytest.mark.parametrize(
    "override,match",
    (
        ({"tie_word_embeddings": True}, "untied"),
        ({"parameter_alias_policy": "shared"}, "shared or aliased"),
        ({"attention_bias": True}, "bias-free"),
    ),
)
def test_dense_llama_topology_rejects_tied_shared_or_biased_targets(
    override: dict[str, object],
    match: str,
) -> None:
    values: dict[str, object] = {
        "vocab_size": 49_152,
        "hidden_size": 448,
        "intermediate_size": 1_792,
        "num_hidden_layers": 36,
        "num_attention_heads": 7,
        "num_key_value_heads": 1,
        "head_dim": 64,
    }
    values.update(override)

    with pytest.raises(DynamicParameterFrontierError, match=match):
        HfLlamaDenseTopology(**values)  # type: ignore[arg-type]


def test_catalog_configuration_rejects_tied_or_shared_target_requests() -> None:
    with pytest.raises(DynamicParameterFrontierError, match="tied target"):
        HfLlamaDenseTargetTopologyCatalog(tie_word_embeddings=True)
    with pytest.raises(DynamicParameterFrontierError, match="shared or aliased"):
        HfLlamaDenseTargetTopologyCatalog(parameter_alias_policy="shared")


def test_cross_family_catalog_dynamically_finds_exact_untied_llama_target() -> None:
    source = _dense_qwen_source_graph()
    frontier = synthesize_dense_parameter_frontier(
        source,
        DenseParameterBudgetRequest(
            target_family="llama",
            target_modalities=("text",),
            target_unique_parameters=EXACT_TARGET_PARAMETERS,
            match_policy="exact",
            maximum_candidates=8,
        ),
    )

    assert frontier.catalog_id == HF_LLAMA_CATALOG_ID
    assert frontier.candidates
    candidate = frontier.candidates[0]
    configuration = dict(candidate.configuration)
    assert candidate.unique_parameters == EXACT_TARGET_PARAMETERS
    assert candidate.logical_parameters == EXACT_TARGET_PARAMETERS
    assert candidate.native_tied_parameter_count == 0
    assert candidate.native_ties == ()
    assert candidate.non_native_weight_sharing is False
    assert configuration["vocab_size"] == 49_152
    assert configuration["hidden_size"] == 448
    assert configuration["intermediate_size"] == 1_792
    assert configuration["num_hidden_layers"] == 36
    assert configuration["num_attention_heads"] == 7
    assert configuration["num_key_value_heads"] == 1
    assert configuration["head_dim"] == 64
    assert configuration["tie_word_embeddings"] is False
    assert configuration[HF_LLAMA_TARGET_SCHEMA_CONFIG_KEY] == (
        HF_LLAMA_TARGET_SCHEMA
    )
    assert configuration[HF_LLAMA_PARAMETER_ALIAS_POLICY_CONFIG_KEY] == (
        HF_LLAMA_PARAMETER_ALIAS_POLICY_NONE
    )
    receipt = candidate.component_allocation_receipt
    assert receipt is not None
    assert receipt.policy == COMPONENT_ALLOCATION_POLICY
    assert receipt.target_unique_parameters == EXACT_TARGET_PARAMETERS
    assert receipt.target_logical_parameters == EXACT_TARGET_PARAMETERS
    assert receipt.source_graph_fingerprint == source.fingerprint
    assert receipt.allocation_selection.exact_unique_parameters == (
        EXACT_TARGET_PARAMETERS
    )
    assert frontier.diagnostics["parameter_sharing_used"] is False
    assert frontier.diagnostics["quantization_used"] is False
    assert frontier.diagnostics["pruning_used"] is False

    tampered = dict(candidate.configuration)
    tampered["tie_word_embeddings"] = True
    with pytest.raises(
        DynamicParameterFrontierError,
        match="untied embedding",
    ):
        replace(candidate, configuration=tuple(tampered.items()))


def _tiny_depth_compressed_candidate():
    source = _dense_qwen_source_graph(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        layer_count=4,
        query_heads=4,
        key_value_heads=2,
        head_dim=8,
    )
    topology = HfLlamaDenseTopology(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    frontier = synthesize_dense_parameter_frontier(
        source,
        DenseParameterBudgetRequest(
            target_family="llama",
            target_modalities=("text",),
            target_unique_parameters=topology.unique_parameters,
            match_policy="exact",
            maximum_candidates=8,
        ),
    )
    return source, frontier.candidates[0]


def test_candidate_spans_and_layer_counts_are_bound_to_receipt_topology() -> None:
    source, candidate = _tiny_depth_compressed_candidate()
    assert candidate.ordered_source_spans == ((0, 1), (2, 3))
    configuration = dict(candidate.configuration)
    assert configuration[HF_LLAMA_SOURCE_LAYER_COUNT_CONFIG_KEY] == 4
    receipt = candidate.component_allocation_receipt
    assert receipt is not None
    assert receipt.source_graph_fingerprint == source.fingerprint

    with pytest.raises(
        DynamicParameterFrontierError,
        match="typed/configured spans",
    ):
        replace(
            candidate,
            ordered_source_spans=((0,), (1, 2, 3)),
        )

    tampered_count = dict(candidate.configuration)
    tampered_count[HF_LLAMA_SOURCE_LAYER_COUNT_CONFIG_KEY] = 5
    assert component_allocation_topology_fingerprint(tampered_count) != (
        receipt.topology_configuration_sha256
    )
    with pytest.raises(
        DynamicParameterFrontierError,
        match="typed/configured spans",
    ):
        replace(candidate, configuration=tuple(tampered_count.items()))

    tampered_depth = dict(candidate.configuration)
    tampered_depth["num_hidden_layers"] = 3
    with pytest.raises(
        DynamicParameterFrontierError,
        match="typed/configured spans",
    ):
        replace(candidate, configuration=tuple(tampered_depth.items()))


@pytest.mark.parametrize(
    ("node_id", "role", "wrong_shape"),
    (
        ("layer.000.attention", "attention.q_proj.weight", (16, 64)),
        ("layer.000.attention", "attention.k_proj.weight", (8, 64)),
        ("layer.000.attention", "attention.v_proj.weight", (8, 64)),
        ("layer.000.attention", "attention.o_proj.weight", (16, 64)),
        ("layer.000.mlp", "mlp.gate_proj.weight", (32, 64)),
        ("layer.000.mlp", "mlp.up_proj.weight", (32, 64)),
        ("layer.000.mlp", "mlp.down_proj.weight", (64, 32)),
    ),
)
def test_catalog_rejects_malformed_source_operator_shapes(
    node_id: str,
    role: str,
    wrong_shape: tuple[int, ...],
) -> None:
    source = _dense_qwen_source_graph(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        layer_count=2,
        query_heads=4,
        key_value_heads=2,
        head_dim=8,
    )
    nodes = list(source.nodes)
    node_index = next(
        index for index, node in enumerate(nodes) if node.node_id == node_id
    )
    parameters = list(nodes[node_index].parameters)
    parameter_index = next(
        index
        for index, parameter in enumerate(parameters)
        if parameter.role == role
    )
    assert parameters[parameter_index].numel == math.prod(wrong_shape)
    parameters[parameter_index] = replace(
        parameters[parameter_index],
        shape=wrong_shape,
    )
    nodes[node_index] = replace(
        nodes[node_index],
        parameters=tuple(parameters),
    )
    malformed = replace(source, nodes=tuple(nodes))
    topology = HfLlamaDenseTopology(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )

    with pytest.raises(
        DynamicParameterFrontierError,
        match=rf"{role} with shape",
    ):
        HfLlamaDenseTargetTopologyCatalog().enumerate_frontier(
            malformed,
            requested_unique_parameters=topology.unique_parameters,
            target_vocabulary_size=16,
            match_policy="exact",
            maximum_candidates=8,
        )


def test_changed_vocab_cannot_claim_bare_tokenizer_bridge_capability() -> None:
    with pytest.raises(
        DynamicParameterFrontierError,
        match="explicit verified strict tokenizer bridge artifact",
    ):
        HfLlamaDenseTargetTopologyCatalog(
            compiler_capabilities=(CAP_TOKENIZER_BRIDGE,)
        )

    source = _dense_qwen_source_graph(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        layer_count=2,
        query_heads=4,
        key_value_heads=2,
        head_dim=8,
    )
    topology = HfLlamaDenseTopology(
        vocab_size=8,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    catalog = HfLlamaDenseTargetTopologyCatalog(
        compiler_capabilities=(
            CAP_CAUSAL_DECODER_FAMILY_TRANSLATION,
            CAP_DENSE_BLANK_MATERIALIZATION,
        )
    )
    frontier = synthesize_dense_parameter_frontier(
        source,
        DenseParameterBudgetRequest(
            target_family="llama",
            target_modalities=("text",),
            target_unique_parameters=topology.unique_parameters,
            target_vocabulary_size=8,
            match_policy="exact",
            maximum_candidates=8,
        ),
        catalogs=DenseTargetTopologyCatalogRegistry((catalog,)),
    )
    candidate = frontier.candidates[0]
    assert candidate.tokenizer_bridge_required is True
    assert CAP_TOKENIZER_BRIDGE in candidate.missing_compiler_capabilities
    assert candidate.compiler_executable is False
    assert not TOKENIZER_BRIDGE_CONFIG_KEYS.intersection(
        dict(candidate.configuration)
    )
    assert frontier.diagnostics["tokenizer_mapping_implemented"] is False


def test_verified_tokenizer_bridge_authenticates_changed_vocab_execution(
    tmp_path: Path,
) -> None:
    artifact = _bridge_artifact(tmp_path)
    source = _dense_qwen_source_graph(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        layer_count=2,
        query_heads=4,
        key_value_heads=2,
        head_dim=8,
        tokenizer_fingerprint=(
            artifact.donor_tokenizer_semantics_sha256
        ),
    )
    topology = HfLlamaDenseTopology(
        vocab_size=8,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    catalog = HfLlamaDenseTargetTopologyCatalog(
        compiler_capabilities=(
            CAP_CAUSAL_DECODER_FAMILY_TRANSLATION,
            CAP_DENSE_BLANK_MATERIALIZATION,
        ),
        tokenizer_bridge_artifact=artifact,
    )
    frontier = synthesize_dense_parameter_frontier(
        source,
        DenseParameterBudgetRequest(
            target_family="llama",
            target_modalities=("text",),
            target_unique_parameters=topology.unique_parameters,
            target_vocabulary_size=8,
            match_policy="exact",
            maximum_candidates=8,
        ),
        catalogs=DenseTargetTopologyCatalogRegistry((catalog,)),
    )
    candidate = frontier.candidates[0]
    configuration = dict(candidate.configuration)
    assert candidate.compiler_executable is True
    assert candidate.missing_compiler_capabilities == ()
    assert CAP_TOKENIZER_BRIDGE in candidate.declared_compiler_capabilities
    assert configuration[TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY] == (
        artifact.artifact_sha256
    )
    assert configuration[TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY] == (
        artifact.profile_fingerprint
    )
    assert configuration[PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY] == (
        artifact.donor_tokenizer_semantics_sha256
    )
    assert configuration[CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY] == (
        artifact.target_tokenizer_semantics_sha256
    )
    assert frontier.diagnostics["tokenizer_mapping_implemented"] is True


def test_changed_vocab_bridge_rejects_wrong_semantics_or_file_tampering(
    tmp_path: Path,
) -> None:
    artifact = _bridge_artifact(tmp_path)
    topology = HfLlamaDenseTopology(
        vocab_size=8,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    request = DenseParameterBudgetRequest(
        target_family="llama",
        target_modalities=("text",),
        target_unique_parameters=topology.unique_parameters,
        target_vocabulary_size=8,
        match_policy="exact",
        maximum_candidates=8,
    )
    catalog = HfLlamaDenseTargetTopologyCatalog(
        tokenizer_bridge_artifact=artifact,
    )
    registry = DenseTargetTopologyCatalogRegistry((catalog,))
    with pytest.raises(
        DynamicParameterFrontierError,
        match="does not match the inspected parent semantics",
    ):
        synthesize_dense_parameter_frontier(
            _dense_qwen_source_graph(
                vocab_size=16,
                hidden_size=32,
                intermediate_size=64,
                layer_count=2,
                query_heads=4,
                key_value_heads=2,
                head_dim=8,
            ),
            request,
            catalogs=registry,
        )

    graph = _dense_qwen_source_graph(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        layer_count=2,
        query_heads=4,
        key_value_heads=2,
        head_dim=8,
        tokenizer_fingerprint=(
            artifact.donor_tokenizer_semantics_sha256
        ),
    )
    (artifact.root_dir / "tokenizer.json").write_text(
        '{"tampered":true}',
        encoding="utf-8",
    )
    with pytest.raises(
        DynamicParameterFrontierError,
        match="changed after catalog registration",
    ):
        synthesize_dense_parameter_frontier(
            graph,
            request,
            catalogs=registry,
        )
