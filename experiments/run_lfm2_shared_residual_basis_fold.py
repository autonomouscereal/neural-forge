#!/usr/bin/env python3
"""Evaluate a large zero-training LFM2 fold with one shared residual basis.

This is a backend specialization of ``run_lfm2_activation_factorization_fold``.
It inherits that runner's authenticated local source-child loading, exact
source/parent cache binding, disjoint trajectory split, parent-free evaluation,
strict trajectory and whole-output gates, exact rollback, and deferred
publication policy.

Only ``w2`` is folded.  One physical residual-coordinate basis is shared across
every layer while each layer keeps a distinct post-SwiGLU coefficient factor.
The rank is uniform across depth, preventing the late-layer rank starvation
that a benefit-greedy independent allocator can produce.  Defaults run the
meaningful roughly-20% target with the roughly-15% target as fallback, starting
from the authenticated 206,726,912-parameter incumbent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    from experiments import run_lfm2_activation_factorization_fold as base
except ImportError:  # Direct execution from the experiments directory.
    import run_lfm2_activation_factorization_fold as base

from neural_forge.hf_lfm2_shared_residual_basis import (
    Lfm2SharedResidualBasisPlan,
    Lfm2SharedResidualFactorizedLinear,
    install_lfm2_shared_residual_basis_plan,
    plan_lfm2_shared_residual_basis,
)


RUN_SCHEMA = "neural-forge-lfm2-shared-residual-basis-fold-run-v1"
PUBLICATION_BLOCK_STATUS = base.PUBLICATION_BLOCK_STATUS
EXPECTED_ORIGINAL_PARENT_UNIQUE_PARAMETERS = (
    base.EXPECTED_ORIGINAL_PARENT_UNIQUE_PARAMETERS
)
STRONG_SOURCE_CHILD_UNIQUE_PARAMETERS = (
    base.STRONG_SOURCE_CHILD_UNIQUE_PARAMETERS
)
DEFAULT_TARGET_TOTAL_UNIQUE_PARAMETERS = (
    base.DEFAULT_TARGET_TOTAL_UNIQUE_PARAMETERS
)
DEFAULT_FALLBACK_TARGET_TOTAL_UNIQUE_PARAMETERS = (
    base.DEFAULT_FALLBACK_TARGET_TOTAL_UNIQUE_PARAMETERS
)
KNOWN_INDEPENDENT_15PCT_RANKS = (
    680,
    680,
    680,
    680,
    680,
    680,
    233,
    185,
    202,
    172,
    195,
    220,
    216,
    253,
)


class Lfm2SharedResidualBasisRunnerError(
    base.Lfm2ActivationFactorizationRunnerError
):
    """The shared-basis runner configuration failed closed."""


@dataclass(frozen=True, slots=True)
class SharedResidualBasisRunConfig(base.ActivationFactorizationRunConfig):
    """The existing strict local runner contract plus one ridge scalar."""

    expected_source_child_unique_parameters: int | None = (
        STRONG_SOURCE_CHILD_UNIQUE_PARAMETERS
    )
    roles: tuple[str, ...] = ("w2",)
    ridge: float = 1e-6
    basis_weighting: str = "layer_balanced"

    def __post_init__(self) -> None:
        base.ActivationFactorizationRunConfig.__post_init__(self)
        if self.roles != ("w2",):
            raise Lfm2SharedResidualBasisRunnerError(
                "The shared residual-output basis supports w2 only."
            )
        if (
            self.expected_source_child_unique_parameters
            != STRONG_SOURCE_CHILD_UNIQUE_PARAMETERS
        ):
            raise Lfm2SharedResidualBasisRunnerError(
                "This frontier runner is bound to the strong 10% incumbent "
                f"with {STRONG_SOURCE_CHILD_UNIQUE_PARAMETERS} unique Parameters."
            )
        if (
            isinstance(self.ridge, bool)
            or not isinstance(self.ridge, (int, float))
            or not math.isfinite(float(self.ridge))
            or float(self.ridge) <= 0.0
        ):
            raise Lfm2SharedResidualBasisRunnerError(
                "ridge must be a finite positive scalar."
            )
        object.__setattr__(self, "ridge", float(self.ridge))
        if self.basis_weighting not in ("layer_balanced", "raw_stack"):
            raise Lfm2SharedResidualBasisRunnerError(
                "basis_weighting must be layer_balanced or raw_stack."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            **base.ActivationFactorizationRunConfig.as_dict(self),
            "factorization": {
                "method": "shared_residual_output_basis",
                "roles": ["w2"],
                "uniform_rank_across_selected_depth": True,
                "one_physical_shared_output_basis": True,
                "distinct_private_input_factor_per_layer": True,
                "ridge": self.ridge,
                "basis_weighting": self.basis_weighting,
                "basis_fit_rows": "sealed_disjoint_fit_post_swiglu_outputs",
                "basis_validation_rows_used_for_fit": False,
                "per_layer_fit_output_scales_sealed": True,
                "late_layer_rank_starvation_permitted": False,
                "dense_w2_fallback_permitted": False,
            },
            "known_independent_15pct_comparison": {
                "ranks_by_layer": list(KNOWN_INDEPENDENT_15PCT_RANKS),
                "total_rank": sum(KNOWN_INDEPENDENT_15PCT_RANKS),
                "mean_rank": (
                    sum(KNOWN_INDEPENDENT_15PCT_RANKS)
                    / len(KNOWN_INDEPENDENT_15PCT_RANKS)
                ),
                "minimum_rank": min(KNOWN_INDEPENDENT_15PCT_RANKS),
                "maximum_rank": max(KNOWN_INDEPENDENT_15PCT_RANKS),
                "late_layers_starved": True,
                "reliable_task_retention": "0/23",
                "interpretation": (
                    "The independent benefit allocator concentrated rank in "
                    "early layers. This backend spends one common rank across "
                    "all 14 residual-write layers."
                ),
            },
        }


def _role_diagnostics(
    plan: Lfm2SharedResidualBasisPlan,
) -> list[dict[str, Any]]:
    factorization = plan.factorization
    return [
        {
            "layer_index": layer.layer_index,
            "role": "w2",
            "rank": layer.rank,
            "uniform_shared_rank": plan.rank,
            "method": layer.method,
            "fit_relative_rmse": layer.fit_relative_rmse,
            "heldout_validation_relative_rmse": (
                layer.validation_relative_rmse
            ),
            "heldout_validation_cosine": layer.validation_cosine,
            "global_worst_layer_heldout_validation_relative_rmse": (
                factorization.validation_worst_layer_relative_rmse
            ),
            "global_worst_layer_heldout_validation_cosine": (
                factorization.validation_worst_layer_cosine
            ),
            "global_mean_layer_heldout_validation_cosine": (
                factorization.validation_mean_layer_cosine
            ),
            "exact_projection_heldout_validation_relative_rmse": (
                layer.projection_validation_relative_rmse
            ),
            "dense_parameters": layer.dense_parameters,
            "private_factor_parameters": (
                layer.private_factor_parameters
            ),
            "shared_basis_parameters_accounted_once_globally": (
                factorization.shared_basis_parameters
            ),
            "shared_rank_advantage_over_independent_uniform_rank": (
                plan.rank
                - plan.independent_uniform_rank_at_same_factor_budget
            ),
            "late_layer_rank_starvation": False,
            "local_projection_reconstruction_only": True,
            "evidence_of_coherent_language_behavior": False,
        }
        for layer in factorization.layers
    ]


def _backend(config: SharedResidualBasisRunConfig) -> base.FactorizationBackend:
    def plan(
        model: Any,
        evidence: Any,
        *,
        target_unique_parameter_budget: int,
        roles: Sequence[str],
    ) -> Lfm2SharedResidualBasisPlan:
        return plan_lfm2_shared_residual_basis(
            model,
            evidence,
            target_unique_parameter_budget=target_unique_parameter_budget,
            roles=roles,
            ridge=config.ridge,
            basis_weighting=config.basis_weighting,
        )

    return base.FactorizationBackend(
        name="uniform_shared_residual_output_basis",
        plan=plan,
        install=install_lfm2_shared_residual_basis_plan,
        factorized_linear_type=Lfm2SharedResidualFactorizedLinear,
        role_diagnostics=_role_diagnostics,
        phase_prefix="shared_residual_basis",
    )


def run_experiment(
    config: SharedResidualBasisRunConfig,
) -> dict[str, Any]:
    """Run both large targets without loading or regenerating the parent."""

    if not isinstance(config, SharedResidualBasisRunConfig):
        raise TypeError("config must be SharedResidualBasisRunConfig.")
    return base.run_experiment(
        config,
        backend=_backend(config),
        run_schema=RUN_SCHEMA,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = (
        "Fold every local LFM2 w2 through one physical shared residual-output "
        "basis at the ~20% target, with an optional ~15% fallback."
    )
    parser.set_defaults(
        roles=("w2",),
        expected_source_child_unique_parameters=(
            STRONG_SOURCE_CHILD_UNIQUE_PARAMETERS
        ),
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-6,
        help=(
            "Positive relative ridge used only for a closed-form private-factor "
            "candidate; heldout rows guard it against exact projection."
        ),
    )
    parser.add_argument(
        "--basis-weighting",
        choices=("layer_balanced", "raw_stack"),
        default="layer_balanced",
        help=(
            "Balance every layer's fit-output energy by default so loud "
            "layers cannot monopolize the common residual basis."
        ),
    )
    return parser


def _config_from_namespace(
    arguments: argparse.Namespace,
) -> SharedResidualBasisRunConfig:
    values = vars(arguments).copy()
    if values["fallback_target_total_unique_parameters"] == 0:
        values["fallback_target_total_unique_parameters"] = None
    return SharedResidualBasisRunConfig(**values)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_experiment(_config_from_namespace(arguments))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == PUBLICATION_BLOCK_STATUS:
        return 0
    return 2 if report["status"] == "rejected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
