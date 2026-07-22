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
    Lfm2SemanticBodyRecipe,
    Lfm2SemanticBodyRecipeError,
    is_supported_lfm2_semantic_component_equivalence_inventory,
    is_supported_lfm2_semantic_body_recipe_inventory,
    lfm2_semantic_body_recipe_ids_for_component_lane,
    resolve_lfm2_semantic_body_recipe,
)


def test_exact_four_body_recipes_are_sealed_and_non_cartesian() -> None:
    assert LFM2_SEMANTIC_BODY_RECIPE_IDS == (
        "fit_guided",
        "full_strength_structural",
        "residual_writer",
        "source_producer_residual_writer",
    )
    recipes = tuple(
        resolve_lfm2_semantic_body_recipe(recipe_id)
        for recipe_id in LFM2_SEMANTIC_BODY_RECIPE_IDS
    )

    assert len({recipe.recipe_sha256 for recipe in recipes}) == 4
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


def test_recipe_rejects_unknown_or_resealed_identity() -> None:
    with pytest.raises(Lfm2SemanticBodyRecipeError):
        resolve_lfm2_semantic_body_recipe("unknown_recipe")
    recipe = Lfm2SemanticBodyRecipe("fit_guided")
    with pytest.raises(Lfm2SemanticBodyRecipeError):
        replace(recipe, recipe_sha256="0" * 64)


def test_receipt_inventory_accepts_legacy_three_and_current_four_only() -> None:
    assert is_supported_lfm2_semantic_body_recipe_inventory(
        list(LFM2_SEMANTIC_BODY_RECIPE_LEGACY_IDS),
        3,
        materialized_recipe_id="residual_writer",
    )
    assert is_supported_lfm2_semantic_body_recipe_inventory(
        list(LFM2_SEMANTIC_BODY_RECIPE_IDS),
        4,
        materialized_recipe_id="source_producer_residual_writer",
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

    assert fit is base
    assert fit.alpha_frontier == base.alpha_frontier
    assert (
        full.alpha_frontier == writer.alpha_frontier == producer_writer.alpha_frontier
    )
    assert producer_writer.alpha_frontier == (0.0, 1.0)
    assert (
        full.trust_radius == writer.trust_radius == producer_writer.trust_radius == 2.0
    )
    assert (
        full.authentication_sha256
        == writer.authentication_sha256
        == producer_writer.authentication_sha256
    )
    assert full.verify_authentication() is True
    assert writer.verify_authentication() is True
    assert producer_writer.verify_authentication() is True


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
    published = (
        "joint_value_output"
        if recipe_id == "full_strength_structural"
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
