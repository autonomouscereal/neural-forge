"""Strict serialization boundary for the LFM2 CPCF short-conv v4 policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_json_bytes
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
    LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
    Lfm2CpcfShortConvV4Error,
    Lfm2CpcfShortConvV4SolverPolicy,
)


_POLICY_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "ridge_alphas",
        "predictor_ridge_alpha",
        "rank_relative_tolerance",
        "participation_relative_tolerance",
        "variable_projection_passes",
        "writer_trust_radius",
        "fir_trust_radius",
        "bcx_trust_radius",
        "correction_rms_floor",
        "context_objective_weight",
        "target_dtype",
        "execution",
        "authentication_sha256",
    }
)
_EXECUTION = {
    "closed_form_only": True,
    "gradients": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


def parse_lfm2_cpcf_short_conv_v4_solver_policy(
    value: Any,
) -> Lfm2CpcfShortConvV4SolverPolicy:
    """Reconstruct one authenticated policy without accepting extra fields."""

    if not isinstance(value, Mapping):
        raise Lfm2CpcfShortConvV4Error("Serialized v4 solver policy must be a mapping.")
    document = dict(value)
    if (
        set(document) != _POLICY_FIELDS
        or document.get("schema") != LFM2_CPCF_SHORT_CONV_V4_SCHEMA
        or document.get("kind") != "authenticated_solver_policy"
        or document.get("execution") != _EXECUTION
        or not isinstance(document.get("ridge_alphas"), (list, tuple))
    ):
        raise Lfm2CpcfShortConvV4Error(
            "Serialized v4 solver policy is incomplete or noncanonical."
        )
    constructor = {
        key: item
        for key, item in document.items()
        if key not in {"schema", "kind", "execution"}
    }
    constructor["ridge_alphas"] = tuple(document["ridge_alphas"])
    try:
        policy = Lfm2CpcfShortConvV4SolverPolicy(**constructor)
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfShortConvV4Error(
            "Serialized v4 solver policy is invalid."
        ) from exc
    if canonical_json_bytes(policy.as_dict()) != canonical_json_bytes(document):
        raise Lfm2CpcfShortConvV4Error(
            "Serialized v4 solver policy does not round-trip exactly."
        )
    return policy


__all__ = ["parse_lfm2_cpcf_short_conv_v4_solver_policy"]
