#!/usr/bin/env python3
"""Orchestrate prepared 35%/50% global-causal LFM2 proposals.

The expensive evidence producer is intentionally separate: it builds physical
isolated FFN proposals, evaluates them only against the already sealed parent
cache/output envelope, and returns ``Lfm2GlobalCausalEvidence``.  Given that
prepared evidence, this runner plans each requested large target, instantiates
real local children through the caller's factory, executes cumulative
full-prefix promotion gates, and commits or exactly rolls back the real
variable-width transaction.

This module has no model-hub loader, optimizer, training controls, or parent
generation path.  It is a programmatic runner because the sealed evidence owns
materialized activation rows and physical tensor bindings; reducing it to an
untrusted JSON-only CLI argument would break that authentication boundary.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.hf_lfm2_cached_causal_evidence import (
    replay_lfm2_cached_causal_frontier_document,
    verify_lfm2_cached_causal_frontier_document,
)
from neural_forge.hf_lfm2_canonical_frontier_replay import (
    Lfm2CanonicalFrontierReplay,
    replay_lfm2_cached_causal_frontier_from_canonical_packs,
)
from neural_forge.hf_lfm2_canonical_proposal_pack import (
    Lfm2CanonicalProposalPack,
)
from neural_forge.hf_lfm2_global_causal_allocator import (
    TARGET_FRACTIONS,
    Lfm2GlobalCausalEvidence,
    Lfm2GlobalCausalInfeasible,
    execute_lfm2_global_causal_plan,
    lfm2_global_causal_budget,
    plan_lfm2_global_causal_target,
    verify_lfm2_global_causal_evidence,
)
from neural_forge.hf_lfm2_interaction_beam_allocator import (
    plan_lfm2_global_causal_target_from_canonical_packs,
)


FRONTIER_SCHEMA = "neural-forge-lfm2-global-causal-frontier-run-v1"
DEFAULT_MAXIMUM_STAGE_FRACTION = 0.10
DEFAULT_MAX_STATES = 100_000


class Lfm2GlobalCausalRunnerError(RuntimeError):
    """The prepared frontier runner contract is invalid."""


def _fraction(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2GlobalCausalRunnerError(
            "Target fractions must be 0.35 and/or 0.50."
        ) from exc
    if result not in TARGET_FRACTIONS:
        raise Lfm2GlobalCausalRunnerError("Target fractions must be 0.35 and/or 0.50.")
    return result


def _parse_fractions(value: str) -> tuple[float, ...]:
    try:
        result = tuple(_fraction(item) for item in value.split(","))
    except Lfm2GlobalCausalRunnerError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(
            "target fractions must be unique comma-separated values"
        )
    return result


@dataclass(frozen=True, slots=True)
class PreparedGlobalCausalFrontierConfig:
    target_fractions: tuple[float, ...] = TARGET_FRACTIONS
    maximum_stage_fraction: float = DEFAULT_MAXIMUM_STAGE_FRACTION
    max_states: int = DEFAULT_MAX_STATES

    def __post_init__(self) -> None:
        fractions = tuple(_fraction(value) for value in self.target_fractions)
        if not fractions or len(set(fractions)) != len(fractions):
            raise Lfm2GlobalCausalRunnerError(
                "target_fractions must be a nonempty unique subset."
            )
        object.__setattr__(self, "target_fractions", fractions)
        if (
            isinstance(self.maximum_stage_fraction, bool)
            or not isinstance(self.maximum_stage_fraction, (int, float))
            or not math.isfinite(float(self.maximum_stage_fraction))
            or float(self.maximum_stage_fraction) <= 0.0
        ):
            raise Lfm2GlobalCausalRunnerError(
                "maximum_stage_fraction must be finite and positive."
            )
        object.__setattr__(
            self,
            "maximum_stage_fraction",
            float(self.maximum_stage_fraction),
        )
        if (
            isinstance(self.max_states, bool)
            or not isinstance(self.max_states, int)
            or self.max_states <= 0
        ):
            raise Lfm2GlobalCausalRunnerError("max_states must be a positive integer.")


@dataclass(frozen=True, slots=True)
class PreparedGlobalCausalFrontierResult:
    report: Mapping[str, Any]
    committed_children: Mapping[float, Any] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )
        object.__setattr__(
            self,
            "committed_children",
            MappingProxyType(dict(self.committed_children)),
        )


def authenticate_cached_global_causal_frontier(
    parent: Any,
    parent_cache: Any,
    parent_rows: Any,
    document: Mapping[str, Any],
    *,
    config: PreparedGlobalCausalFrontierConfig | None = None,
) -> tuple[Lfm2GlobalCausalEvidence, Any]:
    """Rehydrate tensor evidence and authenticate a producer frontier."""

    selected = PreparedGlobalCausalFrontierConfig() if config is None else config
    if not isinstance(selected, PreparedGlobalCausalFrontierConfig):
        raise Lfm2GlobalCausalRunnerError(
            "config must be PreparedGlobalCausalFrontierConfig or None."
        )
    verify_lfm2_cached_causal_frontier_document(document)
    return replay_lfm2_cached_causal_frontier_document(
        parent,
        parent_cache,
        parent_rows,
        document,
        maximum_stage_fraction=selected.maximum_stage_fraction,
        max_states=selected.max_states,
    )


def authenticate_canonical_cached_global_causal_frontier(
    parent: Any,
    parent_cache: Any,
    parent_rows: Any,
    document: Mapping[str, Any],
    canonical_packs: Mapping[str, Lfm2CanonicalProposalPack],
    finalizer_report: Mapping[str, Any],
    *,
    config: PreparedGlobalCausalFrontierConfig | None = None,
) -> Lfm2CanonicalFrontierReplay:
    """Authenticate the measured frontier without any replay reclustering."""

    selected = PreparedGlobalCausalFrontierConfig() if config is None else config
    if not isinstance(selected, PreparedGlobalCausalFrontierConfig):
        raise Lfm2GlobalCausalRunnerError(
            "config must be PreparedGlobalCausalFrontierConfig or None."
        )
    return replay_lfm2_cached_causal_frontier_from_canonical_packs(
        parent,
        parent_cache,
        parent_rows,
        document,
        canonical_packs,
        finalizer_report,
        maximum_stage_fraction=selected.maximum_stage_fraction,
        max_states=selected.max_states,
    )


def execute_prepared_global_causal_frontier(
    parent: Any,
    evidence: Lfm2GlobalCausalEvidence,
    *,
    child_factory: Callable[[float], Any],
    evaluator_factory: Callable[[float, Any], Any],
    config: PreparedGlobalCausalFrontierConfig | None = None,
    canonical_packs: Mapping[str, Lfm2CanonicalProposalPack] | None = None,
    residual_corrector_factory: (
        Callable[[float, Any, Any], Callable[..., Any] | None] | None
    ) = None,
) -> PreparedGlobalCausalFrontierResult:
    """Run real child transactions for every feasible prepared target."""

    selected = PreparedGlobalCausalFrontierConfig() if config is None else config
    if not isinstance(selected, PreparedGlobalCausalFrontierConfig):
        raise Lfm2GlobalCausalRunnerError(
            "config must be PreparedGlobalCausalFrontierConfig or None."
        )
    if not callable(child_factory) or not callable(evaluator_factory):
        raise Lfm2GlobalCausalRunnerError(
            "child_factory and evaluator_factory must be callable."
        )
    if canonical_packs is not None and not isinstance(canonical_packs, Mapping):
        raise Lfm2GlobalCausalRunnerError(
            "canonical_packs must be a mapping or None."
        )
    if residual_corrector_factory is not None and not callable(
        residual_corrector_factory
    ):
        raise Lfm2GlobalCausalRunnerError(
            "residual_corrector_factory must be callable or None."
        )
    verify_lfm2_global_causal_evidence(evidence, parent=parent)
    targets = []
    children = {}
    for fraction in selected.target_fractions:
        budget = lfm2_global_causal_budget(
            fraction,
            parent_unique_parameters=evidence.parent_unique_parameters,
            hidden_size=evidence.variable_width_evidence.hidden_size,
        )
        try:
            plan = (
                plan_lfm2_global_causal_target(
                    parent,
                    evidence,
                    target_fraction=fraction,
                    maximum_stage_fraction=(
                        selected.maximum_stage_fraction
                    ),
                    max_states=selected.max_states,
                )
                if canonical_packs is None
                else plan_lfm2_global_causal_target_from_canonical_packs(
                    parent,
                    evidence,
                    canonical_packs,
                    target_fraction=fraction,
                    maximum_stage_fraction=(
                        selected.maximum_stage_fraction
                    ),
                    max_states=selected.max_states,
                )
            )
        except Lfm2GlobalCausalInfeasible as exc:
            targets.append(
                {
                    "fraction": fraction,
                    "status": "infeasible",
                    "budget": budget.as_dict(),
                    "reason": str(exc),
                    "real_child_instantiated": False,
                    "head_or_conv_parameters_counted": False,
                }
            )
            continue
        evaluator = evaluator_factory(fraction, plan)

        def build_child(selected_fraction: float = fraction) -> Any:
            return child_factory(selected_fraction)

        def build_corrector(stage: Any, child: Any) -> Any:
            if residual_corrector_factory is None:
                return None
            return residual_corrector_factory(fraction, stage, child)

        run = execute_lfm2_global_causal_plan(
            parent,
            evidence,
            plan,
            child_factory=build_child,
            evaluator=evaluator,
            residual_corrector_factory=(
                None if residual_corrector_factory is None else build_corrector
            ),
        )
        if run.committed_child is not None:
            children[fraction] = run.committed_child
        targets.append(
            {
                "fraction": fraction,
                "status": run.status,
                "budget": budget.as_dict(),
                "plan": plan.as_dict(),
                "run": run.as_dict(),
                "run_sha256": run.fingerprint,
            }
        )
    report = {
        "schema": FRONTIER_SCHEMA,
        "status": (
            "accepted"
            if any(target["status"] == "accepted" for target in targets)
            else (
                "rejected"
                if any(target["status"] == "rejected" for target in targets)
                else "infeasible"
            )
        ),
        "evidence_sha256": evidence.evidence_sha256,
        "parent_identity_sha256": evidence.parent_identity_sha256,
        "parent_trace_cache_key": evidence.parent_trace_cache_key,
        "targets": targets,
        "target_order": list(selected.target_fractions),
        "new_parent_generation_calls": 0,
        "parent_cache_build_count": evidence.parent_cache_build_count,
        "hugging_face_hub_loader_used": False,
        "network_model_resolution_allowed": False,
        "physical_installer_used_for_feasible_targets": True,
        "canonical_payloads_used": canonical_packs is not None,
        "reclustering_used": canonical_packs is None,
        "unsupported_head_or_conv_options_counted": False,
        "checkpoint_published": False,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    return PreparedGlobalCausalFrontierResult(report, children)


def build_parser() -> argparse.ArgumentParser:
    """Expose the immutable orchestration policy for launch wrappers."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepared zero-training global-causal LFM2 frontier contract. "
            "Materialized evidence is supplied through the Python API."
        )
    )
    parser.add_argument(
        "--target-fractions",
        type=_parse_fractions,
        default=TARGET_FRACTIONS,
    )
    parser.add_argument(
        "--maximum-stage-fraction",
        type=float,
        default=DEFAULT_MAXIMUM_STAGE_FRACTION,
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=DEFAULT_MAX_STATES,
    )
    parser.add_argument(
        "--describe-contract",
        action="store_true",
        help="Print the programmatic prepared-evidence runner contract.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PreparedGlobalCausalFrontierConfig(
        target_fractions=tuple(args.target_fractions),
        maximum_stage_fraction=args.maximum_stage_fraction,
        max_states=args.max_states,
    )
    if not args.describe_contract:
        raise Lfm2GlobalCausalRunnerError(
            "This runner requires materialized sealed evidence through "
            "execute_prepared_global_causal_frontier(); use "
            "--describe-contract for launch metadata."
        )
    print(
        json.dumps(
            {
                "schema": FRONTIER_SCHEMA,
                "target_fractions": list(config.target_fractions),
                "maximum_stage_fraction": config.maximum_stage_fraction,
                "max_states": config.max_states,
                "physical_installer": ("lfm2_variable_width_medoid_transaction_v1"),
                "child_trace_mode": "full_prefix_canonical",
                "parent_generation_calls": 0,
                "training_controls_available": False,
                "cached_frontier_document_authentication_available": True,
                "canonical_pack_authentication_available": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
