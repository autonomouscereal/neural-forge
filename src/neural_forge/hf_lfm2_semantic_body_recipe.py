"""Authenticated, non-Cartesian body-recipe lanes for semantic LFM2 folding.

The body compiler deliberately exposes five complete recipes rather than a
Cartesian product of every component proposal.  A recipe is selection policy,
not a quality gate: component candidates are eligible only by structural
safety and effect, FIT values may order ``fit_guided``, and held-out values are
always diagnostics for later whole-response review.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.dynamic_circuit_ir import canonical_sha256


LFM2_SEMANTIC_BODY_RECIPE_SCHEMA = "neural-forge-lfm2-semantic-body-recipe-v1"
Lfm2SemanticBodyRecipeId = Literal[
    "fit_guided",
    "full_strength_structural",
    "residual_writer",
    "source_producer_residual_writer",
    "structural_sequence_residual_mlp_writer",
]
LFM2_SEMANTIC_BODY_RECIPE_LEGACY_IDS: tuple[Lfm2SemanticBodyRecipeId, ...] = (
    "fit_guided",
    "full_strength_structural",
    "residual_writer",
)
LFM2_SEMANTIC_BODY_RECIPE_V4_IDS: tuple[Lfm2SemanticBodyRecipeId, ...] = (
    *LFM2_SEMANTIC_BODY_RECIPE_LEGACY_IDS,
    "source_producer_residual_writer",
)
LFM2_SEMANTIC_BODY_RECIPE_IDS: tuple[Lfm2SemanticBodyRecipeId, ...] = (
    *LFM2_SEMANTIC_BODY_RECIPE_V4_IDS,
    "structural_sequence_residual_mlp_writer",
)
LFM2_SEMANTIC_BODY_RECIPE_ACCEPTED_INVENTORIES = (
    LFM2_SEMANTIC_BODY_RECIPE_LEGACY_IDS,
    LFM2_SEMANTIC_BODY_RECIPE_V4_IDS,
    LFM2_SEMANTIC_BODY_RECIPE_IDS,
)


class Lfm2SemanticBodyRecipeError(ValueError):
    """A body recipe is unknown or no longer authentic."""


_RECIPE_COMPONENT_LANES = MappingProxyType(
    {
        "fit_guided": MappingProxyType(
            {
                "qk": "fit_ordered_complete_alpha_frontier",
                "value_output": "fit_ordered_structural_frontier",
                "short_conv": "fit_ordered_structural_frontier",
                "mlp": "fit_ordered_structural_frontier",
            }
        ),
        "full_strength_structural": MappingProxyType(
            {
                "qk": ("maximum_trust_radius_full_alpha_source_derived_residual"),
                "value_output": "joint_then_fixed_structural_priority",
                "short_conv": "full_source_structural_priority",
                "mlp": "full_source_w1_w3_w2_structural_priority",
            }
        ),
        "residual_writer": MappingProxyType(
            {
                "qk": ("maximum_trust_radius_full_alpha_source_derived_residual"),
                "value_output": "fixed_writer_then_joint_residual_priority",
                "short_conv": "alpha_one_residual_writer_priority",
                "mlp": "uncapped_w2_residual_writer_priority",
            }
        ),
        "source_producer_residual_writer": MappingProxyType(
            {
                "qk": ("maximum_trust_radius_full_alpha_source_derived_residual"),
                "value_output": "fixed_writer_then_joint_residual_priority",
                "short_conv": "full_source_structural_priority",
                "mlp": "uncapped_w2_residual_writer_priority",
            }
        ),
        "structural_sequence_residual_mlp_writer": MappingProxyType(
            {
                "qk": ("maximum_trust_radius_full_alpha_source_derived_residual"),
                "value_output": "joint_then_fixed_structural_priority",
                "short_conv": "full_source_structural_priority",
                "mlp": "uncapped_w2_residual_writer_priority",
            }
        ),
    }
)


@dataclass(frozen=True, slots=True)
class Lfm2SemanticBodyRecipe:
    """One sealed complete-body lane shared by every component compiler."""

    recipe_id: Lfm2SemanticBodyRecipeId
    recipe_sha256: str = ""

    def __post_init__(self) -> None:
        if self.recipe_id not in LFM2_SEMANTIC_BODY_RECIPE_IDS:
            raise Lfm2SemanticBodyRecipeError(
                "Semantic body recipe must be exactly one of: "
                + ", ".join(LFM2_SEMANTIC_BODY_RECIPE_IDS)
                + "."
            )
        expected = canonical_sha256(self.authentication_body())
        supplied = str(self.recipe_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2SemanticBodyRecipeError(
                "Semantic body recipe authentication is invalid."
            )
        object.__setattr__(self, "recipe_sha256", expected)

    @property
    def component_lanes(self) -> MappingProxyType[str, str]:
        return _RECIPE_COMPONENT_LANES[self.recipe_id]

    @property
    def fit_metrics_order_candidates(self) -> bool:
        return self.recipe_id == "fit_guided"

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_SEMANTIC_BODY_RECIPE_SCHEMA,
            "recipe_id": self.recipe_id,
            "component_lanes": dict(_RECIPE_COMPONENT_LANES[self.recipe_id]),
            "complete_body_lane": True,
            "cartesian_product": False,
            "candidate_eligibility_authority": (
                "finite-dense-exact-topology-and-nonincumbent-effect-only"
            ),
            "fit_metrics_role": (
                "deterministic-ordering-among-structurally-safe-alternatives"
                if self.fit_metrics_order_candidates
                else "diagnostic-only"
            ),
            "heldout_metrics_role": "diagnostic-only",
            "numeric_thresholds_can_gate_or_delete_candidates": False,
            "numeric_metrics_have_final_intelligence_authority": False,
            "incumbent_fallback_rule": (
                "only-when-every-analytic-alternative-is-invalid-or-identical"
            ),
            "whole_response_manual_intelligence_review_is_authoritative": True,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "recipe_sha256": self.recipe_sha256,
        }

    def verify(self) -> bool:
        return self.recipe_sha256 == canonical_sha256(self.authentication_body())


def resolve_lfm2_semantic_body_recipe(
    value: str | Lfm2SemanticBodyRecipe | None = None,
) -> Lfm2SemanticBodyRecipe:
    """Normalize a public recipe ID while preserving authenticated instances."""

    if value is None:
        return Lfm2SemanticBodyRecipe("fit_guided")
    if isinstance(value, Lfm2SemanticBodyRecipe):
        if not value.verify():
            raise Lfm2SemanticBodyRecipeError(
                "Semantic body recipe changed after authentication."
            )
        return value
    if not isinstance(value, str):
        raise TypeError("body_recipe must be a recipe ID or authenticated recipe.")
    return Lfm2SemanticBodyRecipe(value)  # type: ignore[arg-type]


def is_supported_lfm2_semantic_body_recipe_inventory(
    candidate_ids: object,
    candidate_count: object | None = None,
    *,
    materialized_recipe_id: object | None = None,
) -> bool:
    """Accept sealed legacy three/four receipts while writers emit five."""

    if (
        not isinstance(candidate_ids, Sequence)
        or isinstance(candidate_ids, (str, bytes, bytearray))
        or any(not isinstance(value, str) for value in candidate_ids)
    ):
        return False
    normalized = tuple(candidate_ids)
    if normalized not in LFM2_SEMANTIC_BODY_RECIPE_ACCEPTED_INVENTORIES:
        return False
    if candidate_count is not None and (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count != len(normalized)
    ):
        return False
    return materialized_recipe_id is None or materialized_recipe_id in normalized


def lfm2_semantic_body_recipe_ids_for_component_lane(
    component: str,
    lane: str,
) -> tuple[Lfm2SemanticBodyRecipeId, ...]:
    """Return every complete recipe that authenticates the same component lane."""

    if component not in {"qk", "value_output", "short_conv", "mlp"}:
        raise Lfm2SemanticBodyRecipeError(
            f"Unknown semantic body recipe component {component!r}."
        )
    return tuple(
        recipe_id
        for recipe_id in LFM2_SEMANTIC_BODY_RECIPE_IDS
        if _RECIPE_COMPONENT_LANES[recipe_id][component] == lane
    )


def is_supported_lfm2_semantic_component_equivalence_inventory(
    component: object,
    lane: object,
    candidate_ids: object,
    candidate_equivalent_to_other_recipe: object,
    *,
    materialized_recipe_id: object,
) -> bool:
    """Accept legacy/current lane equivalence without admitting a foreign recipe."""

    if (
        not isinstance(component, str)
        or component not in {"qk", "value_output", "short_conv", "mlp"}
        or not isinstance(lane, str)
        or not isinstance(candidate_ids, Sequence)
        or isinstance(candidate_ids, (str, bytes, bytearray))
        or any(not isinstance(value, str) for value in candidate_ids)
        or type(candidate_equivalent_to_other_recipe) is not bool
        or not isinstance(materialized_recipe_id, str)
    ):
        return False
    normalized = tuple(candidate_ids)
    accepted = tuple(
        tuple(
            recipe_id
            for recipe_id in inventory
            if _RECIPE_COMPONENT_LANES[recipe_id][component] == lane
        )
        for inventory in LFM2_SEMANTIC_BODY_RECIPE_ACCEPTED_INVENTORIES
    )
    return bool(
        normalized in accepted
        and materialized_recipe_id in normalized
        and candidate_equivalent_to_other_recipe is (len(normalized) > 1)
    )


__all__ = [
    "LFM2_SEMANTIC_BODY_RECIPE_IDS",
    "LFM2_SEMANTIC_BODY_RECIPE_LEGACY_IDS",
    "LFM2_SEMANTIC_BODY_RECIPE_V4_IDS",
    "LFM2_SEMANTIC_BODY_RECIPE_ACCEPTED_INVENTORIES",
    "LFM2_SEMANTIC_BODY_RECIPE_SCHEMA",
    "Lfm2SemanticBodyRecipe",
    "Lfm2SemanticBodyRecipeError",
    "Lfm2SemanticBodyRecipeId",
    "is_supported_lfm2_semantic_component_equivalence_inventory",
    "is_supported_lfm2_semantic_body_recipe_inventory",
    "lfm2_semantic_body_recipe_ids_for_component_lane",
    "resolve_lfm2_semantic_body_recipe",
]
