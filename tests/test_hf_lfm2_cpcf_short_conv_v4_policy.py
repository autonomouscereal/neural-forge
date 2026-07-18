from __future__ import annotations

import copy

import pytest

from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
    Lfm2CpcfShortConvV4Error,
    Lfm2CpcfShortConvV4SolverPolicy,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4_policy import (
    parse_lfm2_cpcf_short_conv_v4_solver_policy,
)


def test_v4_solver_policy_round_trips_exactly() -> None:
    expected = Lfm2CpcfShortConvV4SolverPolicy(
        ridge_alphas=(1e-8, 1e-4),
        predictor_ridge_alpha=3e-6,
        variable_projection_passes=2,
        writer_trust_radius=3.0,
        fir_trust_radius=0.75,
        bcx_trust_radius=0.2,
        context_objective_weight=0.075,
        target_dtype="bfloat16",
    )

    observed = parse_lfm2_cpcf_short_conv_v4_solver_policy(expected.as_dict())

    assert observed == expected
    assert observed.verify_authentication()
    assert observed.as_dict() == expected.as_dict()


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update({"unknown": True}),
        lambda value: value.pop("execution"),
        lambda value: value["execution"].update({"training_steps": 1}),
        lambda value: value.update({"ridge_alphas": [1e-4, 1e-8]}),
        lambda value: value.update({"authentication_sha256": "0" * 64}),
    ),
)
def test_v4_solver_policy_rejects_drift(mutation) -> None:
    document = copy.deepcopy(
        Lfm2CpcfShortConvV4SolverPolicy(
            ridge_alphas=(1e-8, 1e-4),
            variable_projection_passes=2,
        ).as_dict()
    )
    mutation(document)

    with pytest.raises(Lfm2CpcfShortConvV4Error):
        parse_lfm2_cpcf_short_conv_v4_solver_policy(document)
