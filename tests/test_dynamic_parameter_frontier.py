from __future__ import annotations

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
    CAP_ATTENTION_TOPOLOGY_PROJECTION,
    CAP_DENSE_BLANK_MATERIALIZATION,
    CAP_RESIDUAL_WIDTH_PROJECTION,
    CAP_TOKENIZER_BRIDGE,
    LFM2_CURRENT_COMPILER_CAPABILITIES,
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    DenseTargetTopologyCandidate,
    DynamicParameterFrontierCapabilityError,
    DynamicParameterFrontierError,
    DynamicParameterFrontierUnattainable,
    LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY,
    LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED,
    Lfm2DenseTargetTopologyCatalog,
    select_best_executable_dense_candidate,
    select_lfm2_untied_rebalance_candidate,
    synthesize_dense_compression_ratio_frontier,
    synthesize_dense_parameter_frontier,
)
from neural_forge.hf_tokenizer_bridge import (
    CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY,
    PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY,
    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
    TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY,
    seal_strict_tokenizer_bridge_artifact,
)
from neural_forge.dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetSynthesisError,
    plan_budgeted_dynamic_target,
)


def _parameter(
    path: str,
    count: int,
    *,
    storage_group: str | None = None,
) -> ParameterSpec:
    return ParameterSpec(
        path=path,
        role=path.replace(".", "_"),
        shape=(count,),
        dtype="torch.float32",
        storage_group=storage_group or path,
        trainable=True,
    )


def _lfm2_graph(
    *,
    vocab: int = 256,
    hidden: int = 64,
    intermediate: int = 128,
    query_heads: int = 4,
    key_value_heads: int = 2,
    head_dim: int = 16,
    fir_length: int = 3,
    layer_types: tuple[str, ...] = (
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
    ),
) -> ModelCircuitGraph:
    """Build an inspected-IR fixture without model weights or Transformers."""

    nodes: list[CircuitNode] = []
    order = 0

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
        nonlocal order
        nodes.append(
            CircuitNode(
                node_id=node_id,
                kind=kind,
                order=order,
                path=node_id,
                layer_index=layer_index,
                inputs=inputs,
                outputs=outputs,
                parameters=parameters,
                attributes=attributes,
            )
        )
        order += 1

    token_matrix = vocab * hidden
    add(
        "input.token_embedding",
        CircuitKind.TOKEN_EMBEDDING,
        layer_index=None,
        parameters=(
            _parameter(
                "model.embed_tokens.weight",
                token_matrix,
                storage_group="token.native_tie",
            ),
        ),
        attributes=(
            ("vocabulary_size", vocab),
            ("hidden_size", hidden),
        ),
        inputs=(
            CircuitPort(
                "token_ids",
                PortKind.TOKEN_IDS,
                ("batch", "sequence"),
            ),
        ),
        outputs=(
            CircuitPort(
                "residual",
                PortKind.RESIDUAL,
                ("batch", "sequence", hidden),
                modality="text",
            ),
        ),
    )
    add(
        "input.position",
        CircuitKind.POSITION_ENCODING,
        layer_index=None,
    )
    for index, layer_type in enumerate(layer_types):
        add(
            f"layer.{index:03d}.operator_norm",
            CircuitKind.NORM,
            layer_index=index,
            parameters=(
                _parameter(f"layers.{index}.operator_norm.weight", hidden),
            ),
        )
        if layer_type == "conv":
            operator_count = 4 * hidden * hidden + fir_length * hidden
            add(
                f"layer.{index:03d}.operator",
                CircuitKind.SHORT_CONV,
                layer_index=index,
                parameters=(
                    _parameter(f"layers.{index}.conv.parameters", operator_count),
                ),
                attributes=(
                    ("operator_type", "short_conv"),
                    ("fir_length", fir_length),
                ),
            )
        else:
            operator_count = (
                2
                * hidden
                * head_dim
                * (query_heads + key_value_heads)
                + 2 * head_dim
            )
            add(
                f"layer.{index:03d}.operator",
                CircuitKind.ATTENTION,
                layer_index=index,
                parameters=(
                    _parameter(
                        f"layers.{index}.attention.parameters",
                        operator_count,
                    ),
                ),
                attributes=(
                    ("operator_type", "full_attention"),
                    ("query_heads", query_heads),
                    ("key_value_heads", key_value_heads),
                    ("head_dim", head_dim),
                ),
            )
        add(
            f"layer.{index:03d}.mlp_norm",
            CircuitKind.NORM,
            layer_index=index,
            parameters=(
                _parameter(f"layers.{index}.mlp_norm.weight", hidden),
            ),
        )
        add(
            f"layer.{index:03d}.mlp",
            CircuitKind.GATED_MLP,
            layer_index=index,
            parameters=(
                _parameter(
                    f"layers.{index}.mlp.parameters",
                    3 * hidden * intermediate,
                ),
            ),
            attributes=(
                ("intermediate_size", intermediate),
                ("activation", "silu"),
            ),
        )
    add(
        "output.final_norm",
        CircuitKind.NORM,
        layer_index=None,
        parameters=(_parameter("model.embedding_norm.weight", hidden),),
    )
    add(
        "output.lm_head",
        CircuitKind.OUTPUT_HEAD,
        layer_index=None,
        parameters=(
            _parameter(
                "lm_head.weight",
                token_matrix,
                storage_group="token.native_tie",
            ),
        ),
        attributes=(
            ("vocabulary_size", vocab),
            ("tied_to_embedding", True),
        ),
        inputs=(
            CircuitPort(
                "residual",
                PortKind.NORMALIZED_RESIDUAL,
                ("batch", "sequence", hidden),
            ),
        ),
        outputs=(
            CircuitPort(
                "logits",
                PortKind.LOGITS,
                ("batch", "sequence", vocab),
                modality="text",
            ),
        ),
    )
    logical = sum(node.logical_parameter_count for node in nodes)
    unique = logical - token_matrix
    return ModelCircuitGraph(
        model_name="synthetic-lfm2",
        family="lfm2",
        model_class="test.SyntheticLfm2",
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
        unique_parameter_count=unique,
        unique_parameter_bytes=unique * 4,
        config_fingerprint=canonical_sha256({"fixture": "lfm2"}),
        metadata=(
            ("inspector", "lfm2-hybrid-decoder"),
            ("hidden_size", hidden),
            ("layer_count", len(layer_types)),
            ("hybrid_attention_short_conv", True),
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
    donor = _BridgeTokenizer([f"<{index}>" for index in range(256)])
    target = _BridgeTokenizer(donor.surfaces[:128])
    root = tmp_path / "target-tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        root,
        donor_row_capacity=256,
        target_row_capacity=128,
    )


def test_lfm2_ratio_frontier_reaches_two_four_and_ten_x_dense() -> None:
    graph = _lfm2_graph()

    results = synthesize_dense_compression_ratio_frontier(
        graph,
        target_family="lfm2",
        target_modalities=("text",),
        compression_ratios=(2, 4, 10),
        maximum_candidates_per_ratio=3,
    )

    assert len(results) == 3
    assert [result.requested_unique_parameters for result in results] == [
        graph.unique_parameter_count // ratio for ratio in (2, 4, 10)
    ]
    for ratio, result in zip((2, 4, 10), results, strict=True):
        assert 1 <= len(result.candidates) <= 3
        winner = result.candidates[0]
        assert winner.unique_parameters <= result.requested_unique_parameters
        assert winner.compression_ratio >= ratio
        assert winner.logical_parameters - winner.unique_parameters == (
            winner.native_tied_parameter_count
        )
        assert winner.native_tied_parameter_count == (
            int(dict(winner.configuration)["vocab_size"])
            * int(dict(winner.configuration)["hidden_size"])
        )
        assert winner.native_ties == (
            (
                "input.token_embedding.weight",
                "output.lm_head.weight",
            ),
        )
        report = winner.as_dict()
        assert report["dense_parameters"] is True
        assert report["quantized"] is False
        assert report["pruned"] is False
        assert report["packed"] is False
        assert report["factorized"] is False
        assert report["non_native_weight_sharing"] is False
        assert report["model_weights_created_or_loaded"] is False
    assert results[-1].candidates[0].unique_parameters <= (
        graph.unique_parameter_count // 10
    )


@pytest.mark.parametrize("ratio", (2, 4, 10))
def test_lfm2_ratio_search_returns_diverse_audited_supported_pareto_points(
    ratio: int,
) -> None:
    result = synthesize_dense_parameter_frontier(
        _lfm2_graph(),
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=ratio,
            maximum_candidates=8,
        ),
    )

    assert len(result.candidates) >= 3
    assert result.diagnostics["selection_is_not_exact_count_only"] is True
    assert result.diagnostics["returned_candidates_pairwise_nondominated"] is True
    assert result.diagnostics["supported_pareto_candidates_before_limit"] >= len(
        result.candidates
    )
    profiles = [candidate.retention_profile for candidate in result.candidates]
    assert len({tuple(sorted(profile.items())) for profile in profiles}) >= 3
    assert all(candidate.pareto_dominance_verified for candidate in result.candidates)
    assert all(candidate.pareto_support_profiles for candidate in result.candidates)
    assert all(
        set(profile)
        == {
            "budget_utilization",
            "residual_width_retention",
            "depth_retention",
            "attention_role_retention",
            "query_channel_retention",
            "head_dim_retention",
            "gqa_ratio_retention",
            "ffn_width_retention",
            "vocabulary_budget_headroom",
            "compiler_executable",
        }
        for profile in profiles
    )
    for left_index, left in enumerate(profiles):
        for right_index, right in enumerate(profiles):
            if left_index == right_index:
                continue
            assert not (
                all(left[axis] >= right[axis] for axis in left)
                and any(left[axis] > right[axis] for axis in left)
            )


def test_capability_selector_can_choose_nonzero_pareto_rank() -> None:
    result = synthesize_dense_parameter_frontier(
        _lfm2_graph(),
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=2,
            maximum_candidates=8,
        ),
    )
    assert result.candidates[0].required_compiler_capabilities != (
        CAP_ATTENTION_TOPOLOGY_PROJECTION,
        CAP_DENSE_BLANK_MATERIALIZATION,
        CAP_RESIDUAL_WIDTH_PROJECTION,
    )

    selection = select_best_executable_dense_candidate(
        result,
        available_compiler_capabilities=(
            CAP_ATTENTION_TOPOLOGY_PROJECTION,
            CAP_DENSE_BLANK_MATERIALIZATION,
            CAP_RESIDUAL_WIDTH_PROJECTION,
        ),
    )

    assert selection.selected_candidate_rank != 0
    selected = result.candidates[selection.selected_candidate_rank]
    assert selection.selected_candidate_fingerprint == selected.fingerprint
    assert set(selected.required_compiler_capabilities).issubset(
        selection.available_compiler_capabilities
    )
    assert selection.selection_policy == (
        "architecture_discovered_component_risk_executable_v1"
    )


def test_one_slot_at_or_below_frontier_preserves_executable_candidate() -> None:
    result = synthesize_dense_parameter_frontier(
        _lfm2_graph(),
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=4,
            maximum_candidates=1,
        ),
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].compiler_executable is True
    selection = select_best_executable_dense_candidate(result)
    assert selection.selected_candidate_rank == 0


def test_two_four_ten_x_selection_surfaces_target_and_feasibility_reason() -> None:
    graph = _lfm2_graph()
    reports: dict[int, dict[str, object]] = {}
    for ratio in (2, 4, 10):
        frontier = synthesize_dense_parameter_frontier(
            graph,
            DenseParameterBudgetRequest(
                target_family="lfm2",
                target_modalities=("text",),
                compression_ratio=ratio,
                maximum_candidates=8,
            ),
        )
        try:
            selection = select_best_executable_dense_candidate(frontier)
        except DynamicParameterFrontierCapabilityError as error:
            missing = {
                capability
                for rejected in error.diagnostics["rejected_candidates"]
                for capability in rejected["missing_compiler_capabilities"]
            }
            reports[ratio] = {
                "feasible": False,
                "reason": error.diagnostics["reason"],
                "missing": missing,
            }
        else:
            candidate = frontier.candidates[
                selection.selected_candidate_rank
            ]
            reports[ratio] = {
                "feasible": True,
                "reason": selection.as_dict()["selection_reason"],
                "candidate_id": candidate.candidate_id,
                "target_unique_parameters": candidate.unique_parameters,
                "configuration": dict(candidate.configuration),
            }

    assert reports[2]["feasible"] is True
    assert reports[4]["feasible"] is True
    for ratio in (2, 4):
        assert reports[ratio]["candidate_id"]
        assert reports[ratio]["target_unique_parameters"] <= (
            graph.unique_parameter_count // ratio
        )
        assert reports[ratio]["reason"] == (
            "choose_the_lowest_discovered_component_risk_among_"
            "materializable_global_topologies"
        )
        assert reports[ratio]["configuration"]["num_hidden_layers"] == len(
            graph.layer_indices
        )
    assert reports[10] == {
        "feasible": False,
        "reason": "no_executable_pareto_candidate",
        "missing": {
            "hybrid_operator_composition",
            "ordered_depth_composition",
        },
    }


def test_capability_selector_fails_closed_without_an_executable_pareto_point() -> None:
    result = synthesize_dense_parameter_frontier(
        _lfm2_graph(),
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=10,
            maximum_candidates=8,
        ),
    )
    assert not any(candidate.compiler_executable for candidate in result.candidates)

    with pytest.raises(
        DynamicParameterFrontierCapabilityError,
        match="No supported-Pareto dense target is executable",
    ) as error:
        select_best_executable_dense_candidate(result)

    assert error.value.diagnostics["reason"] == "no_executable_pareto_candidate"
    assert error.value.diagnostics["rejected_candidates"]


def test_real_lfm2_230m_geometry_selects_exact_supported_frontiers() -> None:
    graph = _lfm2_graph(
        vocab=65_536,
        hidden=1_024,
        intermediate=2_560,
        query_heads=16,
        key_value_heads=8,
        head_dim=64,
        layer_types=(
            "conv",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
        ),
    )
    assert graph.unique_parameter_count == 229_693_184

    expected = {
        2: {
            "selected_rank": 2,
            "unique_parameters": 114_838_464,
            "hidden_size": 576,
            "intermediate_size": 2527,
            "num_hidden_layers": 14,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 64,
        },
        4: {
            "selected_rank": 2,
            "unique_parameters": 57_410_328,
            "hidden_size": 312,
            "intermediate_size": 2472,
            "num_hidden_layers": 14,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
        },
        10: {
            "selected_rank": 2,
            "unique_parameters": 22_960_640,
            "hidden_size": 256,
            "intermediate_size": 269,
            "num_hidden_layers": 14,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
        },
        100: {
            "selected_rank": 2,
            "unique_parameters": 2_296_352,
            "hidden_size": 32,
            "intermediate_size": 104,
            "num_hidden_layers": 14,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 32,
        },
    }
    for ratio, snapshot in expected.items():
        result = synthesize_dense_parameter_frontier(
            graph,
            DenseParameterBudgetRequest(
                target_family="lfm2",
                target_modalities=("text",),
                compression_ratio=ratio,
                match_policy="at_or_below",
                maximum_candidates=64,
            ),
        )
        selection = select_best_executable_dense_candidate(result)
        candidate = result.candidates[
            selection.selected_candidate_rank
        ]
        configuration = dict(candidate.configuration)

        assert selection.selected_candidate_rank == snapshot["selected_rank"]
        assert candidate.unique_parameters == snapshot["unique_parameters"]
        assert candidate.compression_ratio >= ratio
        assert candidate.compiler_executable is True
        assert candidate.missing_compiler_capabilities == ()
        assert candidate.ordered_source_spans == tuple(
            (index,) for index in range(14)
        )
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ):
            assert configuration[name] == snapshot[name]
        assert result.requested_unique_parameters == (
            graph.unique_parameter_count // ratio
        )
        assert result.diagnostics["tokenizer_bridge_required"] is False
        assert result.diagnostics["intelligence_retention_proven"] is False
        assert result.as_dict()["weights_or_downloads_used"] is False

        if ratio == 100:
            assert result.diagnostics["vocabulary_budget_pressure"] == (
                "critical"
            )
            assert candidate.vocabulary_unique_parameters / (
                graph.unique_parameter_count // ratio
            ) >= 0.85
        else:
            assert result.diagnostics["vocabulary_budget_pressure"] == (
                "bounded"
            )


def test_default_and_explicit_preserve_catalog_outputs_are_identical() -> None:
    graph = _lfm2_graph()
    request = DenseParameterBudgetRequest(
        target_family="lfm2",
        target_modalities=("text",),
        compression_ratio=4,
        maximum_candidates=8,
    )

    default = synthesize_dense_parameter_frontier(
        graph,
        request,
        catalogs=DenseTargetTopologyCatalogRegistry(
            (Lfm2DenseTargetTopologyCatalog(),)
        ),
    )
    explicit = synthesize_dense_parameter_frontier(
        graph,
        request,
        catalogs=DenseTargetTopologyCatalogRegistry(
            (
                Lfm2DenseTargetTopologyCatalog(
                    target_token_interface_mode="preserve"
                ),
            )
        ),
    )

    assert explicit == default
    assert explicit.fingerprint == default.fingerprint
    assert [candidate.candidate_id for candidate in explicit.candidates] == [
        candidate.candidate_id for candidate in default.candidates
    ]
    assert all(
        LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY
        not in dict(candidate.configuration)
        for candidate in default.candidates
    )


def test_untied_catalog_derives_exact4x_full_depth_interface_rebalance(
    tmp_path: Path,
) -> None:
    # Reuse the sealed 65,536 -> 4,096 bridge fixture already used to prove
    # the production catalog.  This test imports it lazily to avoid a module
    # collection cycle between the two focused suites.
    from tests.test_dynamic_lfm2_cpcf_v5_frontier import (
        _sealed_bridge,
    )

    bridge = _sealed_bridge(tmp_path / "strict-4096-row-bridge")
    graph = replace(
        _lfm2_graph(
            vocab=65_536,
            hidden=1_024,
            intermediate=2_560,
            query_heads=16,
            key_value_heads=8,
            head_dim=64,
            layer_types=(
                "conv",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
            ),
        ),
        tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
    )
    assert graph.unique_parameter_count == 229_693_184
    result = synthesize_dense_parameter_frontier(
        graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=4,
            target_vocabulary_size=4_096,
            match_policy="at_or_below",
            maximum_candidates=8,
        ),
        catalogs=DenseTargetTopologyCatalogRegistry(
            (
                Lfm2DenseTargetTopologyCatalog(
                    tokenizer_bridge_artifact=bridge,
                    target_token_interface_mode="untied",
                ),
            )
        ),
    )

    candidate = next(
        candidate
        for candidate in result.candidates
        if "untied_preserve_frontier_rebalance"
        in candidate.pareto_support_profiles
    )
    selection = select_lfm2_untied_rebalance_candidate(result)
    assert result.candidates[selection.selected_candidate_rank] is candidate
    assert selection.selected_candidate_fingerprint == candidate.fingerprint
    assert selection.selection_policy == (
        "lfm2_untied_preserve_frontier_rebalance_v1"
    )
    assert selection.as_dict()["selection_reason"] == (
        "select_the_authenticated_tied_balanced_topology_then_"
        "reserve_a_dense_untied_output_inside_the_same_budget"
    )
    configuration = dict(candidate.configuration)
    automatic_selection = select_best_executable_dense_candidate(result)
    assert (
        result.candidates[automatic_selection.selected_candidate_rank]
        is candidate
    )
    starved_mlp_candidate = next(
        value
        for value in result.candidates
        if (
            dict(value.configuration)["hidden_size"],
            dict(value.configuration)["intermediate_size"],
        )
        == (1_024, 29)
    )
    assert candidate.component_allocation_receipt is not None
    assert starved_mlp_candidate.component_allocation_receipt is not None
    assert (
        candidate.component_allocation_receipt.total_predicted_risk
        < starved_mlp_candidate.component_allocation_receipt.total_predicted_risk
    )
    assert result.requested_unique_parameters == 57_423_296
    assert candidate.unique_parameters == 57_422_440
    assert candidate.logical_parameters == 57_422_440
    assert candidate.compression_ratio == pytest.approx(
        4.000059628256828
    )
    assert candidate.native_tied_parameter_count == 0
    assert candidate.native_ties == ()
    assert candidate.vocabulary_unique_parameters == 2 * 4_096 * 776
    assert candidate.compiler_executable is True
    assert {
        name: configuration[name]
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "tie_word_embeddings",
            LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY,
        )
    } == {
        "vocab_size": 4_096,
        "hidden_size": 776,
        "intermediate_size": 700,
        "num_hidden_layers": 14,
        "num_attention_heads": 10,
        "num_key_value_heads": 5,
        "head_dim": 64,
        "tie_word_embeddings": False,
        LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY: (
            LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
        ),
    }
    assert result.diagnostics[
        "untied_rebalance_source_unique_parameters"
    ] == 57_405_368
    assert result.diagnostics[
        "untied_rebalance_target_unique_parameters"
    ] == 57_422_440
    assert result.diagnostics[
        "untied_rebalance_parameter_ceiling_spare"
    ] == 856
    assert result.diagnostics[
        "untied_rebalance_changed_only_ffn_width"
    ] is True
    assert result.diagnostics[
        "untied_rebalance_full_search_nondominated"
    ] is True


def test_catalog_rejects_unknown_target_token_interface_mode() -> None:
    with pytest.raises(
        DynamicParameterFrontierError,
        match="must be preserve or untied",
    ):
        Lfm2DenseTargetTopologyCatalog(
            target_token_interface_mode="retie",
        )


def test_exact_candidate_round_trip_and_unrepresentable_exact_diagnostics() -> None:
    graph = _lfm2_graph()
    nearest = synthesize_dense_parameter_frontier(
        graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            target_unique_parameters=graph.unique_parameter_count // 7,
            match_policy="nearest",
            maximum_candidates=2,
        ),
    )
    representable = nearest.candidates[0].unique_parameters

    exact = synthesize_dense_parameter_frontier(
        graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            target_unique_parameters=representable,
            match_policy="exact",
            maximum_candidates=2,
        ),
    )

    assert exact.candidates[0].exact_budget_match is True
    assert exact.candidates[0].unique_parameters == representable
    target_request = exact.candidates[0].to_target_request(
        child_name="blank-lfm2",
        blank_seed=9,
    )
    assert target_request.exact_unique_parameters == representable
    assert target_request.target_family == "lfm2"
    assert dict(target_request.configuration)["model_type"] == "lfm2"

    with pytest.raises(
        DynamicParameterFrontierUnattainable,
        match="No legal dense target topology",
    ) as error:
        synthesize_dense_parameter_frontier(
            graph,
            DenseParameterBudgetRequest(
                target_family="lfm2",
                target_modalities=("text",),
                target_unique_parameters=1,
                match_policy="exact",
            ),
        )
    assert error.value.diagnostics["nearest_representable_above"] > 1
    assert error.value.diagnostics["exact_budget_representable"] is False


def test_huge_ratio_nearest_exposes_vocab_floor_and_ceiling_policy_rejects() -> None:
    graph = _lfm2_graph()
    nearest = synthesize_dense_parameter_frontier(
        graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=100,
            match_policy="nearest",
        ),
    )

    winner = nearest.candidates[0]
    assert winner.unique_parameters > nearest.requested_unique_parameters
    assert winner.unique_parameters == nearest.diagnostics[
        "minimum_representable_unique_parameters"
    ]
    assert nearest.diagnostics["vocabulary_budget_pressure"] == "exceeds_budget"
    assert nearest.diagnostics[
        "tokenizer_bridge_required_for_requested_budget"
    ] is True
    assert nearest.diagnostics["tokenizer_mapping_implemented"] is False

    with pytest.raises(DynamicParameterFrontierUnattainable):
        synthesize_dense_parameter_frontier(
            graph,
            DenseParameterBudgetRequest(
                target_family="lfm2",
                target_modalities=("text",),
                compression_ratio=100,
                match_policy="at_or_below",
            ),
        )
    with pytest.raises(
        DynamicParameterFrontierUnattainable,
        match="below one learned parameter",
    ):
        synthesize_dense_parameter_frontier(
            graph,
            DenseParameterBudgetRequest(
                target_family="lfm2",
                target_modalities=("text",),
                compression_ratio=1_000_000,
                match_policy="nearest",
            ),
        )


def test_optional_target_vocabulary_marks_unimplemented_tokenizer_bridge() -> None:
    graph = _lfm2_graph()
    result = synthesize_dense_parameter_frontier(
        graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=10,
            target_vocabulary_size=128,
            match_policy="at_or_below",
        ),
    )

    winner = result.candidates[0]
    assert winner.source_vocabulary_size == 256
    assert winner.target_vocabulary_size == 128
    assert winner.tokenizer_bridge_required is True
    assert dict(winner.configuration)["vocab_size"] == 128
    assert result.diagnostics["tokenizer_bridge_required"] is True
    assert result.diagnostics["tokenizer_mapping_implemented"] is False


def test_tokenizer_bridge_capability_requires_verified_artifact(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        DynamicParameterFrontierError,
        match="requires an explicit verified",
    ):
        Lfm2DenseTargetTopologyCatalog(
            compiler_capabilities=(
                *LFM2_CURRENT_COMPILER_CAPABILITIES,
                CAP_TOKENIZER_BRIDGE,
            )
        )

    artifact = _bridge_artifact(tmp_path)
    graph = replace(
        _lfm2_graph(),
        tokenizer_fingerprint=(
            artifact.donor_tokenizer_semantics_sha256
        ),
    )
    catalog = Lfm2DenseTargetTopologyCatalog(
        tokenizer_bridge_artifact=artifact,
    )
    registry = DenseTargetTopologyCatalogRegistry((catalog,))
    plan = plan_budgeted_dynamic_target(
        graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=4,
            target_vocabulary_size=128,
        ),
        child_name="strict-bridged-blank",
        catalogs=registry,
    )
    candidate = plan.frontier.candidates[
        plan.target_certificate.selected_candidate_rank
    ]
    configuration = dict(candidate.configuration)

    assert CAP_TOKENIZER_BRIDGE in candidate.declared_compiler_capabilities
    assert CAP_TOKENIZER_BRIDGE not in candidate.missing_compiler_capabilities
    assert candidate.compiler_executable
    assert plan.target_request.tokenizer_fingerprint == (
        artifact.target_tokenizer_semantics_sha256
    )
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
    assert plan.frontier.diagnostics["tokenizer_mapping_implemented"] is True


def test_tokenizer_bridge_planning_rejects_wrong_semantics_and_tampering(
    tmp_path: Path,
) -> None:
    artifact = _bridge_artifact(tmp_path)
    request = DenseParameterBudgetRequest(
        target_family="lfm2",
        target_modalities=("text",),
        compression_ratio=10,
        target_vocabulary_size=128,
    )
    registry = DenseTargetTopologyCatalogRegistry(
        (
            Lfm2DenseTargetTopologyCatalog(
                tokenizer_bridge_artifact=artifact,
            ),
        )
    )
    with pytest.raises(
        DynamicParameterFrontierError,
        match="does not match the inspected parent semantics",
    ):
        synthesize_dense_parameter_frontier(
            _lfm2_graph(),
            request,
            catalogs=registry,
        )

    graph = replace(
        _lfm2_graph(),
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


def test_candidates_reject_parameter_count_cheats_and_wrong_modalities() -> None:
    graph = _lfm2_graph()
    candidate = synthesize_dense_parameter_frontier(
        graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=4,
        ),
    ).candidates[0]

    with pytest.raises(
        DynamicParameterFrontierError,
        match="parameter-count cheats",
    ):
        replace(candidate, quantized=True)
    with pytest.raises(
        DynamicParameterFrontierError,
        match="parameter-count cheats",
    ):
        replace(candidate, non_native_weight_sharing=True)
    with pytest.raises(
        DynamicParameterFrontierError,
        match="No registered dense topology catalog",
    ):
        synthesize_dense_parameter_frontier(
            graph,
            DenseParameterBudgetRequest(
                target_family="lfm2",
                target_modalities=("image", "text"),
                compression_ratio=2,
            ),
        )


def test_frontier_is_deterministic_from_inspected_ir_only() -> None:
    graph = _lfm2_graph()
    request = DenseParameterBudgetRequest(
        target_family="lfm2",
        target_modalities=("text",),
        compression_ratio=10,
        maximum_candidates=5,
    )

    first = synthesize_dense_parameter_frontier(graph, request)
    second = synthesize_dense_parameter_frontier(graph, request)

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert [candidate.fingerprint for candidate in first.candidates] == [
        candidate.fingerprint for candidate in second.candidates
    ]
    assert first.as_dict()["weights_or_downloads_used"] is False


def test_candidate_accounting_rejects_fake_native_tie() -> None:
    with pytest.raises(
        DynamicParameterFrontierError,
        match="Logical/unique accounting",
    ):
        DenseTargetTopologyCandidate(
            candidate_id="bad",
            catalog_id="test",
            family="test",
            modalities=("text",),
            requested_unique_parameters=10,
            source_unique_parameters=20,
            unique_parameters=10,
            logical_parameters=15,
            native_tied_parameter_count=4,
            retention_score_ppm=0,
            configuration=(),
            ordered_source_spans=((0,),),
            native_ties=(("embedding", "head"),),
        )


@pytest.mark.parametrize("ratio", (4, 10))
def test_budget_bridge_selects_rank_zero_and_certifies_existing_target_request(
    ratio: int,
) -> None:
    graph = _lfm2_graph()

    plan = plan_budgeted_dynamic_target(
        graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=ratio,
            maximum_candidates=3,
        ),
        child_name=f"lfm2-{ratio}x-blank",
        selected_candidate_rank=0,
        blank_seed=ratio,
    )

    candidate = plan.frontier.candidates[0]
    certificate = plan.target_certificate
    assert certificate.selected_candidate_rank == 0
    assert certificate.selected_candidate_fingerprint == candidate.fingerprint
    assert certificate.frontier_fingerprint == plan.frontier.fingerprint
    assert plan.target_request.exact_unique_parameters == candidate.unique_parameters
    assert plan.target_request.configuration == candidate.configuration
    assert certificate.planned_graph.unique_parameter_count == (
        candidate.unique_parameters
    )
    assert certificate.planned_graph.logical_parameter_count == (
        candidate.logical_parameters
    )
    assert (
        certificate.planned_graph.logical_parameter_count
        - certificate.planned_graph.unique_parameter_count
        == candidate.native_tied_parameter_count
    )
    report = plan.as_dict()
    assert report["selected_rank_provenance_preserved"] is True
    assert report["weights_or_downloads_used"] is False
    assert report["target_certificate"]["ready_for_existing_target_synthesizer"] is (
        True
    )
    assert report["target_certificate"]["planned_graph"][
        "model_or_weights_materialized"
    ] is False


def test_budget_bridge_preserves_explicit_nonzero_candidate_rank() -> None:
    graph = _lfm2_graph()
    plan = plan_budgeted_dynamic_target(
        graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=4,
            maximum_candidates=3,
        ),
        child_name="rank-one-blank",
        selected_candidate_rank=1,
    )

    assert plan.target_certificate.selected_candidate_rank == 1
    assert plan.target_certificate.selected_candidate_fingerprint == (
        plan.frontier.candidates[1].fingerprint
    )
    assert plan.target_request.exact_unique_parameters == (
        plan.frontier.candidates[1].unique_parameters
    )


def test_budget_bridge_fails_closed_on_over_budget_and_tokenizer_bridge() -> None:
    graph = _lfm2_graph()

    with pytest.raises(
        DynamicTargetSynthesisError,
        match="over-budget nearest",
    ):
        plan_budgeted_dynamic_target(
            graph,
            DenseParameterBudgetRequest(
                target_family="lfm2",
                target_modalities=("text",),
                compression_ratio=100,
                match_policy="nearest",
            ),
            child_name="infeasible-blank",
        )

    with pytest.raises(
        DynamicTargetSynthesisError,
        match="tokenizer bridge",
    ):
        plan_budgeted_dynamic_target(
            graph,
            DenseParameterBudgetRequest(
                target_family="lfm2",
                target_modalities=("text",),
                compression_ratio=10,
                target_vocabulary_size=128,
            ),
            child_name="unbridged-tokenizer-blank",
        )


def test_automatic_system_uses_its_registered_topology_catalog() -> None:
    graph = _lfm2_graph()
    selected_catalog = Lfm2DenseTargetTopologyCatalog(
        catalog_id="test.catalog.selected-lfm2",
        priority=999,
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry(),
        stack_adapters=DynamicFoldStackAdapterRegistry(),
        topology_catalogs=DenseTargetTopologyCatalogRegistry(
            (selected_catalog,)
        ),
    )

    plan = system.plan_budgeted_target(
        parent_graph=graph,
        budget_request=DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=4,
        ),
        child_name="catalog-selected-blank",
    )

    assert plan.frontier.catalog_id == selected_catalog.catalog_id
    assert plan.target_certificate.planned_graph.configuration
    assert dict(plan.target_request.configuration)[
        "_neural_forge_dense_topology_catalog"
    ] == selected_catalog.catalog_id
