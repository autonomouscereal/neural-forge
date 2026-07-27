from __future__ import annotations

from dataclasses import replace

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (
    Lfm2EndpointQkResidualFoldPolicy,
)
from neural_forge.hf_lfm2_semantic_attention_refold_adapter import (
    _qk_policy_for_body_recipe,
)
from neural_forge import hf_lfm2_semantic_layer_stack_refold_adapter as stack_adapter
from neural_forge.hf_lfm2_semantic_body_recipe import (
    LFM2_SEMANTIC_BODY_RECIPE_IDS,
    LFM2_SEMANTIC_BODY_RECIPE_LEGACY_IDS,
    LFM2_SEMANTIC_BODY_RECIPE_V4_IDS,
    LFM2_SEMANTIC_BODY_RECIPE_V5_IDS,
    LFM2_SEMANTIC_BODY_RECIPE_V7_IDS,
    LFM2_SEMANTIC_BODY_RECIPE_V9_IDS,
    LFM2_SEMANTIC_BODY_RECIPE_V10_IDS,
    LFM2_SEMANTIC_NONBLOCKING_BEHAVIOR_FLOW_RECIPE_IDS,
    LFM2_SEMANTIC_NONBLOCKING_BEHAVIOR_FLOW_SCHEMA,
    Lfm2SemanticBodyRecipe,
    Lfm2SemanticBodyRecipeError,
    is_supported_lfm2_semantic_component_equivalence_inventory,
    is_supported_lfm2_semantic_body_recipe_inventory,
    lfm2_semantic_body_recipe_ids_for_component_lane,
    resolve_lfm2_semantic_body_recipe,
    resolve_lfm2_semantic_nonblocking_behavior_flow,
)
from neural_forge.hf_lfm2_producer_aware_semantic_integration import (
    Lfm2ProducerAwareCorrectionScalePolicy,
)


def test_exact_eleven_body_recipes_are_sealed_and_non_cartesian() -> None:
    assert LFM2_SEMANTIC_BODY_RECIPE_IDS == (
        "fit_guided",
        "full_strength_structural",
        "residual_writer",
        "source_producer_residual_writer",
        "structural_sequence_residual_mlp_writer",
        "nonlinear_qdeim_structural",
        "nonlinear_qdeim_fit_corrected",
        "nonlinear_qdeim_coupled_attention_structural",
        "nonlinear_qdeim_coupled_attention_fit_corrected",
        "nonlinear_qdeim_coupled_attention_short_conv_residual_writer",
        "nonlinear_qdeim_coupled_attention_source_local_residual_fit_corrected",
    )
    recipes = tuple(
        resolve_lfm2_semantic_body_recipe(recipe_id)
        for recipe_id in LFM2_SEMANTIC_BODY_RECIPE_IDS
    )

    assert len({recipe.recipe_sha256 for recipe in recipes}) == 11
    assert all(recipe.verify() for recipe in recipes)
    assert all(recipe.as_dict()["cartesian_product"] is False for recipe in recipes)
    assert all(
        recipe.as_dict()["numeric_thresholds_can_gate_or_delete_candidates"] is False
        for recipe in recipes
    )
    assert all(
        recipe.as_dict()["heldout_metrics_role"] == "diagnostic-only"
        for recipe in recipes
    )
    hybrid = resolve_lfm2_semantic_body_recipe("source_producer_residual_writer")
    assert dict(hybrid.component_lanes) == {
        "qk": "maximum_trust_radius_full_alpha_source_derived_residual",
        "value_output": "fixed_writer_then_joint_residual_priority",
        "short_conv": "full_source_structural_priority",
        "mlp": "uncapped_w2_residual_writer_priority",
    }
    assert hybrid.fit_metrics_order_candidates is False
    structural_sequence_writer = resolve_lfm2_semantic_body_recipe(
        "structural_sequence_residual_mlp_writer"
    )
    assert dict(structural_sequence_writer.component_lanes) == {
        "qk": "maximum_trust_radius_full_alpha_source_derived_residual",
        "value_output": "joint_then_fixed_structural_priority",
        "short_conv": "full_source_structural_priority",
        "mlp": "uncapped_w2_residual_writer_priority",
    }
    assert structural_sequence_writer.fit_metrics_order_candidates is False
    qdeim_structural = resolve_lfm2_semantic_body_recipe("nonlinear_qdeim_structural")
    qdeim_corrected = resolve_lfm2_semantic_body_recipe("nonlinear_qdeim_fit_corrected")
    assert dict(qdeim_structural.component_lanes) == {
        "qk": "maximum_trust_radius_full_alpha_source_derived_residual",
        "value_output": "joint_then_fixed_structural_priority",
        "short_conv": "full_source_structural_priority",
        "mlp": "nonlinear_qdeim_structural_priority",
    }
    assert dict(qdeim_corrected.component_lanes) == {
        **dict(qdeim_structural.component_lanes),
        "mlp": "nonlinear_qdeim_fit_corrected_priority",
    }
    coupled_structural = resolve_lfm2_semantic_body_recipe(
        "nonlinear_qdeim_coupled_attention_structural"
    )
    coupled_corrected = resolve_lfm2_semantic_body_recipe(
        "nonlinear_qdeim_coupled_attention_fit_corrected"
    )
    assert dict(coupled_structural.component_lanes) == {
        "qk": "intact_gqa_group_qdeim_coupled_qkvo",
        "value_output": "intact_gqa_group_qdeim_coupled_qkvo",
        "short_conv": "full_source_structural_priority",
        "mlp": "nonlinear_qdeim_structural_priority",
    }
    assert dict(coupled_corrected.component_lanes) == {
        **dict(coupled_structural.component_lanes),
        "mlp": "nonlinear_qdeim_fit_corrected_priority",
    }
    short_conv_writer = resolve_lfm2_semantic_body_recipe(
        "nonlinear_qdeim_coupled_attention_short_conv_residual_writer"
    )
    assert dict(short_conv_writer.component_lanes) == {
        **dict(coupled_structural.component_lanes),
        "short_conv": "alpha_one_residual_writer_priority",
    }
    assert (
        short_conv_writer.component_lanes["qk"]
        == (coupled_structural.component_lanes["qk"])
    )
    assert (
        short_conv_writer.component_lanes["value_output"]
        == (coupled_structural.component_lanes["value_output"])
    )
    assert (
        short_conv_writer.component_lanes["mlp"]
        == (coupled_structural.component_lanes["mlp"])
    )
    source_local = resolve_lfm2_semantic_body_recipe(
        "nonlinear_qdeim_coupled_attention_source_local_residual_fit_corrected"
    )
    assert dict(source_local.component_lanes) == {
        **dict(coupled_corrected.component_lanes),
        "attention_output_residual": "bounded_source_local_attention_o",
        "short_conv_output_residual": "bounded_source_local_short_conv_writer",
        "global_residual": "source_local_action_boundaries",
    }


def test_coupled_tangent_body_short_conv_writer_composes_with_closure_scale_zero() -> (
    None
):
    recipe = resolve_lfm2_semantic_body_recipe(
        "nonlinear_qdeim_coupled_attention_short_conv_residual_writer"
    )
    scale = Lfm2ProducerAwareCorrectionScalePolicy(
        materialization_frontier=(0.0,),
        materialization_scale=0.0,
    )

    assert recipe.verify() is True
    assert recipe.component_lanes == {
        "qk": "intact_gqa_group_qdeim_coupled_qkvo",
        "value_output": "intact_gqa_group_qdeim_coupled_qkvo",
        "short_conv": "alpha_one_residual_writer_priority",
        "mlp": "nonlinear_qdeim_structural_priority",
    }
    assert scale.verify_authentication() is True
    assert scale.materialization_frontier == (0.0,)
    assert scale.materialization_scale == 0.0


def test_recipe_rejects_unknown_or_resealed_identity() -> None:
    with pytest.raises(Lfm2SemanticBodyRecipeError):
        resolve_lfm2_semantic_body_recipe("unknown_recipe")
    recipe = Lfm2SemanticBodyRecipe("fit_guided")
    with pytest.raises(Lfm2SemanticBodyRecipeError):
        replace(recipe, recipe_sha256="0" * 64)


def test_nonblocking_behavior_flow_is_separate_sealed_and_coupled_only() -> None:
    expected_recipe_hashes = {
        "fit_guided": "fedc74e4d1bbe28a77c00f3c78a12ba58910899134d214d324b03343c0c6056a",
        "full_strength_structural": "eed15b90833fcae8b01493b02c23716999f2e4fb1cebca2af7d198e01298e7ff",
        "residual_writer": "131a3934b912fdf564cb8a7145323fada1aeec1355f71b7a9d74f46e4baa1f23",
        "source_producer_residual_writer": "5767c45aeb2b022c50fd084d4c7e17d9451b33b6a188c4fba4e9c705095cc4da",
        "structural_sequence_residual_mlp_writer": "11298ed377b2ad775b5a1a29cae3dc38292adf995af88dc197d601d4c2f7b784",
        "nonlinear_qdeim_structural": "62a7b081dbeb739494a7d75f4e968f442724ea2add8f1f01c3107036bb2cac5d",
        "nonlinear_qdeim_fit_corrected": "6897bb34f76223e46919be696031e0afeb703c92c6ab7836f6fba23c330356ac",
        "nonlinear_qdeim_coupled_attention_structural": "2725779d70161592345868bad2270ca5dbeaecbb1416a3c129b32fe3449235dc",
        "nonlinear_qdeim_coupled_attention_fit_corrected": "0c37f95ad809c83a1a34df61afcf080a64e228bbe19a7465d300d36a541334ce",
        "nonlinear_qdeim_coupled_attention_short_conv_residual_writer": "f5b2219783f61ec33a235cad46e3ec66530cd51696868f51671addcfbd966311",
        "nonlinear_qdeim_coupled_attention_source_local_residual_fit_corrected": "a10962bd077a6412ab3becbb22ffac760fbb57fc56b6d2c6d1dd7b3cfd6b29d9",
    }
    assert {
        recipe_id: resolve_lfm2_semantic_body_recipe(recipe_id).recipe_sha256
        for recipe_id in LFM2_SEMANTIC_BODY_RECIPE_IDS
    } == expected_recipe_hashes
    assert LFM2_SEMANTIC_NONBLOCKING_BEHAVIOR_FLOW_RECIPE_IDS == (
        "nonlinear_qdeim_coupled_attention_structural",
        "nonlinear_qdeim_coupled_attention_fit_corrected",
        "nonlinear_qdeim_coupled_attention_short_conv_residual_writer",
        "nonlinear_qdeim_coupled_attention_source_local_residual_fit_corrected",
    )
    assert resolve_lfm2_semantic_nonblocking_behavior_flow("fit_guided") is None

    for recipe_id in LFM2_SEMANTIC_NONBLOCKING_BEHAVIOR_FLOW_RECIPE_IDS:
        recipe = resolve_lfm2_semantic_body_recipe(recipe_id)
        policy = resolve_lfm2_semantic_nonblocking_behavior_flow(recipe)
        assert policy is not None
        assert policy.verify() is True
        assert policy.body_recipe_sha256 == recipe.recipe_sha256
        assert policy.as_dict() == {
            "schema": LFM2_SEMANTIC_NONBLOCKING_BEHAVIOR_FLOW_SCHEMA,
            "body_recipe_id": recipe.recipe_id,
            "body_recipe_sha256": recipe.recipe_sha256,
            "structural_authentication_is_only_materialization_veto": True,
            "numeric_metrics_may_prioritize_investigation": True,
            "numeric_accept_veto_publish_delete_stop_authority": False,
            "manual_annotations_may_guide_next_hypothesis": True,
            "manual_or_binary_accept_veto_publish_delete_stop_authority": False,
            "decoded_complete_behavior_is_primary_intelligence_evidence": True,
            "retention_target_is_objective_not_cutoff": True,
            "all_structurally_safe_candidates_must_be_materialized": True,
            "all_materialized_candidates_must_receive_decoded_review": True,
            "manual_verdict_required_for_continuation": False,
            "quality_evidence_can_stop_iteration": False,
            "policy_sha256": policy.policy_sha256,
        }
        with pytest.raises(
            Lfm2SemanticBodyRecipeError,
            match="behavior-flow authentication",
        ):
            replace(policy, policy_sha256="0" * 64)


def test_receipt_inventory_accepts_historical_and_current_recipe_sets() -> None:
    assert is_supported_lfm2_semantic_body_recipe_inventory(
        list(LFM2_SEMANTIC_BODY_RECIPE_LEGACY_IDS),
        3,
        materialized_recipe_id="residual_writer",
    )
    assert is_supported_lfm2_semantic_body_recipe_inventory(
        list(LFM2_SEMANTIC_BODY_RECIPE_V4_IDS),
        4,
        materialized_recipe_id="source_producer_residual_writer",
    )
    assert is_supported_lfm2_semantic_body_recipe_inventory(
        list(LFM2_SEMANTIC_BODY_RECIPE_V5_IDS),
        5,
        materialized_recipe_id="structural_sequence_residual_mlp_writer",
    )
    assert is_supported_lfm2_semantic_body_recipe_inventory(
        list(LFM2_SEMANTIC_BODY_RECIPE_V7_IDS),
        7,
        materialized_recipe_id="nonlinear_qdeim_fit_corrected",
    )
    assert is_supported_lfm2_semantic_body_recipe_inventory(
        list(LFM2_SEMANTIC_BODY_RECIPE_V9_IDS),
        9,
        materialized_recipe_id="nonlinear_qdeim_coupled_attention_fit_corrected",
    )
    assert is_supported_lfm2_semantic_body_recipe_inventory(
        list(LFM2_SEMANTIC_BODY_RECIPE_V10_IDS),
        10,
        materialized_recipe_id=(
            "nonlinear_qdeim_coupled_attention_short_conv_residual_writer"
        ),
    )
    assert is_supported_lfm2_semantic_body_recipe_inventory(
        list(LFM2_SEMANTIC_BODY_RECIPE_IDS),
        11,
        materialized_recipe_id=(
            "nonlinear_qdeim_coupled_attention_source_local_residual_fit_corrected"
        ),
    )
    assert not is_supported_lfm2_semantic_body_recipe_inventory(
        ["fit_guided", "residual_writer"],
        2,
        materialized_recipe_id="fit_guided",
    )
    assert not is_supported_lfm2_semantic_body_recipe_inventory(
        list(LFM2_SEMANTIC_BODY_RECIPE_LEGACY_IDS),
        4,
        materialized_recipe_id="fit_guided",
    )
    assert not is_supported_lfm2_semantic_body_recipe_inventory(
        list(LFM2_SEMANTIC_BODY_RECIPE_LEGACY_IDS),
        3,
        materialized_recipe_id="source_producer_residual_writer",
    )
    assert not is_supported_lfm2_semantic_body_recipe_inventory(
        list(LFM2_SEMANTIC_BODY_RECIPE_V4_IDS),
        4,
        materialized_recipe_id="structural_sequence_residual_mlp_writer",
    )
    assert not is_supported_lfm2_semantic_body_recipe_inventory(
        list(LFM2_SEMANTIC_BODY_RECIPE_V5_IDS),
        5,
        materialized_recipe_id="nonlinear_qdeim_structural",
    )
    assert not is_supported_lfm2_semantic_body_recipe_inventory(
        list(LFM2_SEMANTIC_BODY_RECIPE_V7_IDS),
        7,
        materialized_recipe_id="nonlinear_qdeim_coupled_attention_structural",
    )


def test_component_equivalence_accepts_sealed_legacy_and_current_claims_only() -> None:
    lane = "fixed_writer_then_joint_residual_priority"
    assert is_supported_lfm2_semantic_component_equivalence_inventory(
        "value_output",
        lane,
        ["residual_writer"],
        False,
        materialized_recipe_id="residual_writer",
    )
    assert is_supported_lfm2_semantic_component_equivalence_inventory(
        "value_output",
        lane,
        ["residual_writer", "source_producer_residual_writer"],
        True,
        materialized_recipe_id="residual_writer",
    )
    assert not is_supported_lfm2_semantic_component_equivalence_inventory(
        "value_output",
        lane,
        ["residual_writer"],
        False,
        materialized_recipe_id="source_producer_residual_writer",
    )
    assert not is_supported_lfm2_semantic_component_equivalence_inventory(
        "value_output",
        lane,
        ["source_producer_residual_writer", "residual_writer"],
        True,
        materialized_recipe_id="source_producer_residual_writer",
    )
    assert not is_supported_lfm2_semantic_component_equivalence_inventory(
        "value_output",
        lane,
        ["residual_writer", "source_producer_residual_writer"],
        False,
        materialized_recipe_id="residual_writer",
    )

    structural_lane = "joint_then_fixed_structural_priority"
    assert is_supported_lfm2_semantic_component_equivalence_inventory(
        "value_output",
        structural_lane,
        ["full_strength_structural"],
        False,
        materialized_recipe_id="full_strength_structural",
    )
    assert is_supported_lfm2_semantic_component_equivalence_inventory(
        "value_output",
        structural_lane,
        [
            "full_strength_structural",
            "structural_sequence_residual_mlp_writer",
        ],
        True,
        materialized_recipe_id="structural_sequence_residual_mlp_writer",
    )
    assert is_supported_lfm2_semantic_component_equivalence_inventory(
        "value_output",
        structural_lane,
        [
            "full_strength_structural",
            "structural_sequence_residual_mlp_writer",
            "nonlinear_qdeim_structural",
            "nonlinear_qdeim_fit_corrected",
        ],
        True,
        materialized_recipe_id="nonlinear_qdeim_structural",
    )

    coupled_lane = "intact_gqa_group_qdeim_coupled_qkvo"
    coupled_recipe_ids = [
        "nonlinear_qdeim_coupled_attention_structural",
        "nonlinear_qdeim_coupled_attention_fit_corrected",
        "nonlinear_qdeim_coupled_attention_short_conv_residual_writer",
        "nonlinear_qdeim_coupled_attention_source_local_residual_fit_corrected",
    ]
    assert is_supported_lfm2_semantic_component_equivalence_inventory(
        "qk",
        coupled_lane,
        coupled_recipe_ids,
        True,
        materialized_recipe_id="nonlinear_qdeim_coupled_attention_structural",
    )
    assert is_supported_lfm2_semantic_component_equivalence_inventory(
        "value_output",
        coupled_lane,
        coupled_recipe_ids,
        True,
        materialized_recipe_id="nonlinear_qdeim_coupled_attention_fit_corrected",
    )


def test_qk_structural_recipes_use_strongest_authenticated_lane_without_metrics() -> (
    None
):
    base = Lfm2EndpointQkResidualFoldPolicy()
    fit = _qk_policy_for_body_recipe(
        base,
        Lfm2SemanticBodyRecipe("fit_guided"),
    )
    full = _qk_policy_for_body_recipe(
        base,
        Lfm2SemanticBodyRecipe("full_strength_structural"),
    )
    writer = _qk_policy_for_body_recipe(
        base,
        Lfm2SemanticBodyRecipe("residual_writer"),
    )
    producer_writer = _qk_policy_for_body_recipe(
        base,
        Lfm2SemanticBodyRecipe("source_producer_residual_writer"),
    )
    structural_sequence_writer = _qk_policy_for_body_recipe(
        base,
        Lfm2SemanticBodyRecipe("structural_sequence_residual_mlp_writer"),
    )
    qdeim_structural = _qk_policy_for_body_recipe(
        base,
        Lfm2SemanticBodyRecipe("nonlinear_qdeim_structural"),
    )
    qdeim_corrected = _qk_policy_for_body_recipe(
        base,
        Lfm2SemanticBodyRecipe("nonlinear_qdeim_fit_corrected"),
    )

    assert fit is base
    assert fit.alpha_frontier == base.alpha_frontier
    assert (
        full.alpha_frontier
        == writer.alpha_frontier
        == producer_writer.alpha_frontier
        == structural_sequence_writer.alpha_frontier
        == qdeim_structural.alpha_frontier
        == qdeim_corrected.alpha_frontier
    )
    assert producer_writer.alpha_frontier == (0.0, 1.0)
    assert (
        full.trust_radius
        == writer.trust_radius
        == producer_writer.trust_radius
        == structural_sequence_writer.trust_radius
        == qdeim_structural.trust_radius
        == qdeim_corrected.trust_radius
        == 2.0
    )
    assert (
        full.authentication_sha256
        == writer.authentication_sha256
        == producer_writer.authentication_sha256
        == structural_sequence_writer.authentication_sha256
        == qdeim_structural.authentication_sha256
        == qdeim_corrected.authentication_sha256
    )
    assert full.verify_authentication() is True
    assert writer.verify_authentication() is True
    assert producer_writer.verify_authentication() is True
    assert structural_sequence_writer.verify_authentication() is True
    assert qdeim_structural.verify_authentication() is True
    assert qdeim_corrected.verify_authentication() is True


@pytest.mark.parametrize("recipe_id", LFM2_SEMANTIC_BODY_RECIPE_IDS)
def test_layer_stack_requires_exact_recipe_binding_for_every_lane(
    recipe_id: str,
) -> None:
    recipe = resolve_lfm2_semantic_body_recipe(recipe_id)
    for component, lane in recipe.component_lanes.items():
        receipt = {
            "body_recipe": recipe.as_dict(),
            "body_recipe_sha256": recipe.recipe_sha256,
            "body_recipe_id": recipe.recipe_id,
            "recipe_component_lane": lane,
            "fit_metrics_used_for_candidate_ordering": (
                recipe.fit_metrics_order_candidates
            ),
        }

        stack_adapter._require_body_recipe_binding(
            receipt,
            recipe,
            component=component,
            label="test component",
        )
        assert (
            stack_adapter._fit_ordering_flag_matches_bound_recipe(
                receipt,
                component=component,
            )
            is True
        )
        for foreign_id in LFM2_SEMANTIC_BODY_RECIPE_IDS:
            if foreign_id == recipe_id:
                continue
            with pytest.raises(
                stack_adapter.Lfm2SemanticLayerStackRefoldError,
                match="did not bind",
            ):
                stack_adapter._require_body_recipe_binding(
                    receipt,
                    resolve_lfm2_semantic_body_recipe(foreign_id),
                    component=component,
                    label="test component",
                )
        with pytest.raises(
            stack_adapter.Lfm2SemanticLayerStackRefoldError,
            match="did not bind",
        ):
            stack_adapter._require_body_recipe_binding(
                {**receipt, "recipe_component_lane": "forged-lane"},
                recipe,
                component=component,
                label="test component",
            )
        with pytest.raises(
            stack_adapter.Lfm2SemanticLayerStackRefoldError,
            match="did not bind",
        ):
            stack_adapter._require_body_recipe_binding(
                {},
                recipe,
                component=component,
                label="test component",
            )


@pytest.mark.parametrize("recipe_id", LFM2_SEMANTIC_BODY_RECIPE_IDS)
def test_attention_nested_recipe_lane_and_candidate_policy_are_bound(
    recipe_id: str,
) -> None:
    recipe = resolve_lfm2_semantic_body_recipe(recipe_id)
    if recipe.component_lanes["qk"] == "intact_gqa_group_qdeim_coupled_qkvo":
        source_local_output = (
            recipe.component_lanes.get("attention_output_residual")
            == "bounded_source_local_attention_o"
        )
        coupled_receipt = {
            "body_recipe": recipe.as_dict(),
            "body_recipe_sha256": recipe.recipe_sha256,
            "body_recipe_id": recipe.recipe_id,
            "recipe_component_lane": recipe.component_lanes["qk"],
            "recipe_value_output_lane": recipe.component_lanes["value_output"],
            "coupled_qkvo_single_candidate": True,
            "split_qk_or_vo_fallback_used": False,
        }
        if source_local_output:
            coupled_receipt.update(
                {
                    "source_local_attention_output": True,
                    "source_local_attention_output_lane": (
                        "bounded_source_local_attention_o"
                    ),
                    "structural_coupled_qkvo_candidate_installed": True,
                    "source_local_o_only_second_stage": True,
                }
            )
        stack_adapter._require_attention_body_recipe_binding(
            coupled_receipt,
            recipe,
            label="test coupled attention",
        )
        forged_fields = [
            ("recipe_component_lane", "forged-qk-lane"),
            ("recipe_value_output_lane", "forged-value-output-lane"),
            ("split_qk_or_vo_fallback_used", True),
        ]
        if source_local_output:
            forged_fields.extend(
                [
                    ("source_local_attention_output", False),
                    ("source_local_attention_output_lane", "forged-output-lane"),
                    ("structural_coupled_qkvo_candidate_installed", False),
                    ("source_local_o_only_second_stage", False),
                ]
            )
        else:
            forged_fields.append(("coupled_qkvo_single_candidate", False))
        for field, forged_value in forged_fields:
            with pytest.raises(stack_adapter.Lfm2SemanticLayerStackRefoldError):
                stack_adapter._require_attention_body_recipe_binding(
                    {**coupled_receipt, field: forged_value},
                    recipe,
                    label="test coupled attention",
                )
        return

    published = (
        "joint_value_output"
        if recipe.component_lanes["value_output"]
        == "joint_then_fixed_structural_priority"
        else "fixed_v_output_only"
    )
    reports = {
        "incumbent": {"eligible": False},
        "fixed_v_output_only": {"eligible": True},
        "joint_value_output": {"eligible": True},
    }

    def sealed_gate(
        *,
        candidate: str = published,
        lane: str | None = None,
    ) -> dict[str, object]:
        equivalent_recipe_ids = list(
            lfm2_semantic_body_recipe_ids_for_component_lane(
                "value_output",
                recipe.component_lanes["value_output"],
            )
        )
        body: dict[str, object] = {
            "body_recipe": recipe.as_dict(),
            "body_recipe_sha256": recipe.recipe_sha256,
            "body_recipe_id": recipe.recipe_id,
            "recipe_component_lane": (
                recipe.component_lanes["value_output"] if lane is None else lane
            ),
            "candidate_reports": reports,
            "selected_candidate": candidate,
            "published_candidate": candidate,
            "structurally_eligible_candidate_count": 2,
            "fit_metrics_used_for_candidate_ordering": (
                recipe.fit_metrics_order_candidates
            ),
            "component_selection_policy_equivalent_recipe_ids": equivalent_recipe_ids,
            "component_selection_policy_equivalent_to_other_recipe": (
                len(equivalent_recipe_ids) > 1
            ),
        }
        return {**body, "gate_sha256": canonical_sha256(body)}

    def receipt(gate: dict[str, object]) -> dict[str, object]:
        return {
            "body_recipe": recipe.as_dict(),
            "body_recipe_sha256": recipe.recipe_sha256,
            "body_recipe_id": recipe.recipe_id,
            "recipe_component_lane": recipe.component_lanes["qk"],
            "analytic_refold": {
                "body_recipe": recipe.as_dict(),
                "body_recipe_sha256": recipe.recipe_sha256,
                "body_recipe_id": recipe.recipe_id,
                "recipe_qk_lane": recipe.component_lanes["qk"],
                "recipe_value_output_lane": recipe.component_lanes["value_output"],
                "value_output_candidate_gate": gate,
            },
        }

    valid = receipt(sealed_gate())
    stack_adapter._require_attention_body_recipe_binding(
        valid,
        recipe,
        label="test attention",
    )

    with pytest.raises(
        stack_adapter.Lfm2SemanticLayerStackRefoldError,
        match="recipe and lane",
    ):
        stack_adapter._require_attention_body_recipe_binding(
            {**valid, "recipe_component_lane": "forged-qk-lane"},
            recipe,
            label="test attention",
        )
    with pytest.raises(
        stack_adapter.Lfm2SemanticLayerStackRefoldError,
        match="recipe lane",
    ):
        forged_analytic = dict(valid)
        forged_analytic["analytic_refold"] = {
            **valid["analytic_refold"],
            "recipe_value_output_lane": "forged-value-output-lane",
        }
        stack_adapter._require_attention_body_recipe_binding(
            forged_analytic,
            recipe,
            label="test attention",
        )
    with pytest.raises(
        stack_adapter.Lfm2SemanticLayerStackRefoldError,
        match="recipe and lane",
    ):
        stack_adapter._require_attention_body_recipe_binding(
            receipt(sealed_gate(lane="forged-gate-lane")),
            recipe,
            label="test attention",
        )
    if recipe_id != "fit_guided":
        wrong = (
            "fixed_v_output_only"
            if published == "joint_value_output"
            else "joint_value_output"
        )
        with pytest.raises(
            stack_adapter.Lfm2SemanticLayerStackRefoldError,
            match="not bound to the declared recipe policy",
        ):
            stack_adapter._require_attention_body_recipe_binding(
                receipt(sealed_gate(candidate=wrong)),
                recipe,
                label="test attention",
            )
