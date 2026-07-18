from __future__ import annotations

import copy

import pytest
import torch

from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
    Lfm2CpcfShortConvV4SolverPolicy,
    causal_depthwise_context,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4_fold import (
    Lfm2CpcfShortConvNumericalFrontierRequest,
    Lfm2CpcfShortConvV4FoldCase,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5 import (
    V5_ACCEPTED_EDGE_INTERPOLATION_FRACTIONS,
    V5_PRIOR_CONTROL_SUFFIX,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5_numerical_adapter import (
    Lfm2CpcfShortConvV5NumericalAdapterError,
    Lfm2CpcfShortConvV5NumericalFrontierAdapter,
)


def _request(
    *,
    source_fir_rows: torch.Tensor | None = None,
    reverse_cases: bool = False,
    source_hidden: int = 4,
    target_hidden: int = 2,
    include_biases: bool = True,
    zero_writer: bool = False,
) -> Lfm2CpcfShortConvNumericalFrontierRequest:
    generator = torch.Generator().manual_seed(20260718)
    source_fir_length = 3
    target_fir_length = 2
    source_in_weight = (
        torch.randn(
            (3 * source_hidden, source_hidden),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.2
    )
    source_in_bias = (
        torch.randn(
            (3 * source_hidden,),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.03
        if include_biases
        else None
    )
    if source_fir_rows is None:
        source_fir_rows = torch.tensor(
            [0.75, -0.25, 0.125],
            dtype=torch.float64,
        ).repeat(source_hidden, 1)
    assert tuple(source_fir_rows.shape) == (
        source_hidden,
        source_fir_length,
    )
    source_fir_bias = (
        torch.randn(
            (source_hidden,),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.02
        if include_biases
        else None
    )
    source_out_weight = (
        torch.zeros(
            (source_hidden, source_hidden),
            dtype=torch.float64,
        )
        if zero_writer
        else (
            torch.randn(
                (source_hidden, source_hidden),
                generator=generator,
                dtype=torch.float64,
            )
            * 0.25
        )
    )
    source_out_bias = (
        (
            torch.zeros((source_hidden,), dtype=torch.float64)
            if zero_writer
            else (
                torch.randn(
                    (source_hidden,),
                    generator=generator,
                    dtype=torch.float64,
                )
                * 0.02
            )
        )
        if include_biases
        else None
    )
    output_encoder = torch.randn(
        (source_hidden, target_hidden),
        generator=generator,
        dtype=torch.float64,
    )
    input_lift = torch.randn(
        (target_hidden, source_hidden),
        generator=generator,
        dtype=torch.float64,
    )
    cases = []
    contexts = {}
    for case_index, case_id in enumerate(("fit-a", "fit-b")):
        batch = 2
        sequence = 7
        suffix_start = 2
        child = torch.randn(
            (batch, sequence, target_hidden),
            generator=generator,
            dtype=torch.float64,
        )
        parent_input = child @ input_lift + 0.01 * case_index
        parent_b = torch.randn(
            (batch, sequence, source_hidden),
            generator=generator,
            dtype=torch.float64,
        )
        parent_c = torch.randn(
            (batch, sequence, source_hidden),
            generator=generator,
            dtype=torch.float64,
        )
        parent_x = torch.randn(
            (batch, sequence, source_hidden),
            generator=generator,
            dtype=torch.float64,
        )
        parent_context = causal_depthwise_context(
            parent_b,
            parent_c,
            parent_x,
            source_fir_rows,
            source_fir_bias,
        )
        parent_write = parent_context @ source_out_weight.T
        if source_out_bias is not None:
            parent_write = parent_write + source_out_bias
        parent_write = parent_write @ output_encoder
        actual_pre = (
            torch.randn(
                (
                    batch,
                    sequence - suffix_start,
                    target_hidden,
                ),
                generator=generator,
                dtype=torch.float64,
            )
            * 0.05
        )
        valid_prefix = torch.ones(
            (batch, sequence),
            dtype=torch.bool,
        )
        valid_prefix[1, -1] = False
        case = Lfm2CpcfShortConvV4FoldCase(
            case_id=case_id,
            split="fit",
            suffix_start=suffix_start,
            actual_child_operator_input_history=child,
            parent_operator_input_history=parent_input,
            parent_b_history=parent_b,
            parent_c_history=parent_c,
            parent_x_history=parent_x,
            parent_post_operator_residual_target_space_suffix=(
                actual_pre + parent_write[:, suffix_start:, :]
            ),
            actual_child_pre_operator_residual_suffix=actual_pre,
            valid_prefix_mask=valid_prefix,
        )
        cases.append(case)
        contexts[case_id] = parent_context
    if reverse_cases:
        cases.reverse()
    return Lfm2CpcfShortConvNumericalFrontierRequest(
        source_tensors={
            "source_in_proj_weight": source_in_weight,
            "source_in_proj_bias": source_in_bias,
            "source_fir_kernel": source_fir_rows,
            "source_fir_bias": source_fir_bias,
            "source_out_proj_weight": source_out_weight,
            "source_out_proj_bias": source_out_bias,
            "output_encoder": output_encoder,
        },
        cases=tuple(cases),
        parent_contexts=contexts,
        target_templates={
            "in_proj_weight": torch.zeros(
                (3 * target_hidden, target_hidden),
                dtype=torch.float64,
            ),
            "in_proj_bias": torch.zeros(
                (3 * target_hidden,),
                dtype=torch.float64,
            )
            if include_biases
            else None,
            "fir_kernel": torch.zeros(
                (target_hidden, target_fir_length),
                dtype=torch.float64,
            ),
            "fir_bias": torch.zeros(
                (target_hidden,),
                dtype=torch.float64,
            )
            if include_biases
            else None,
            "writer_weight": torch.zeros(
                (target_hidden, target_hidden),
                dtype=torch.float64,
            ),
            "writer_bias": torch.zeros(
                (target_hidden,),
                dtype=torch.float64,
            )
            if include_biases
            else None,
        },
        policy=Lfm2CpcfShortConvV4SolverPolicy(
            ridge_alphas=(1e-6,),
            variable_projection_passes=2,
            target_dtype="float64",
        ),
    )


def test_real_adapter_generates_complete_authenticated_dynamic_frontier() -> None:
    request = _request()
    frontier = Lfm2CpcfShortConvV5NumericalFrontierAdapter().generate(request)

    assert request.verify_authentication()
    assert frontier.verify_authentication()
    assert frontier.request_sha256 == request.request_sha256
    assert tuple(frontier.context_encoder.shape) == (4, 2)
    assert frontier.full_source_numeric_participation
    assert frontier.source_channels_selected == 4
    assert frontier.source_channels_dropped == 0
    assert all(
        candidate.verify_authentication() for candidate in frontier.raw_candidates
    )
    assert all(
        candidate.request_sha256 == request.request_sha256
        for candidate in frontier.raw_candidates
    )
    assert all(
        tuple(candidate.in_proj_weight.shape) == (6, 2)
        and tuple(candidate.fir_kernel.shape) == (2, 2)
        and tuple(candidate.writer_weight.shape) == (2, 2)
        for candidate in frontier.raw_candidates
    )
    provenance = frontier.report["adapter_provenance"]
    assert provenance["real_v5_numerical_algorithm"] is True
    assert provenance["dynamic_source_target_topology"] is True
    assert provenance["source_fir_resized_before_basis"] is True
    assert provenance["basis_actual_commutator_gate_passed"] is True
    assert provenance["basis_full_source_numeric_participation"] is True
    assert provenance["fit_descriptor_case_count"] == 2
    assert provenance["fit_projection_case_count"] == 2
    assert provenance["selection_cases_consumed"] == 0
    assert provenance["validation_cases_consumed"] == 0
    assert provenance["model_forward_calls"] == 0
    assert provenance["training"] is False
    assert provenance["quantization"] is False
    assert provenance["pruning"] is False
    assert set(provenance["condition_cap_inventory"]) == {
        "kappa-01000",
        "kappa-10000",
    }
    assert all(
        value["interpolated_candidate_count"]
        == value["accepted_edge_count"] * len(V5_ACCEPTED_EDGE_INTERPOLATION_FRACTIONS)
        for value in provenance["condition_cap_inventory"].values()
    )
    for cap in provenance["condition_cap_inventory"].values():
        assert cap["phase_attempt_count"] == 6
        assert [value["phase"] for value in cap["phase_inventory"]] == [
            "writer",
            "fir",
            "bcx",
            "writer",
            "fir",
            "bcx",
        ]
        assert all(
            value["solver_intercept_contract"]
            == (
                "physical_bias_coordinate_in_design_with_"
                "separate_intercept_absent_frozen_zero"
            )
            for value in cap["phase_inventory"]
            if value["phase"] == "fir"
        )
        assert all(
            value["solver_intercept_contract"]
            == (
                "physical_bias_coordinates_in_design_with_"
                "separate_intercept_absent_frozen_zero"
            )
            for value in cap["phase_inventory"]
            if value["phase"] == "bcx"
        )


def test_real_adapter_has_exactly_one_prior_control_and_every_edge_interior() -> None:
    frontier = Lfm2CpcfShortConvV5NumericalFrontierAdapter().generate(_request())
    candidate_ids = [candidate.candidate_id for candidate in frontier.raw_candidates]
    prior_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id.endswith(V5_PRIOR_CONTROL_SUFFIX)
    ]

    assert len(prior_ids) == 1
    assert (
        prior_ids[0]
        == frontier.report["adapter_provenance"]["prior_control_candidate_id"]
    )
    assert any(
        "v5-kappa-10000--anchor-00-transported" == value for value in candidate_ids
    )
    assert any("--interior-" in value for value in candidate_ids)
    assert len(candidate_ids) == len(set(candidate_ids))


def test_real_adapter_accepts_fir_and_bcx_context_folding_when_supported() -> None:
    frontier = Lfm2CpcfShortConvV5NumericalFrontierAdapter().generate(
        _request(zero_writer=True)
    )

    for cap in frontier.report["adapter_provenance"][
        "condition_cap_inventory"
    ].values():
        accepted_phases = {
            value["phase"] for value in cap["phase_inventory"] if value["accepted"]
        }
        assert {"writer", "fir", "bcx"} <= accepted_phases
        assert cap["accepted_path_state_count"] >= 4
        assert cap["accepted_edge_count"] == cap["accepted_path_state_count"] - 1


def test_real_adapter_derives_alternate_unbiased_topology_from_request() -> None:
    request = _request(
        source_hidden=6,
        target_hidden=3,
        include_biases=False,
    )
    frontier = Lfm2CpcfShortConvV5NumericalFrontierAdapter().generate(request)

    assert tuple(frontier.context_encoder.shape) == (6, 3)
    assert frontier.source_channels_selected == 6
    assert all(
        tuple(candidate.in_proj_weight.shape) == (9, 3)
        and candidate.in_proj_bias is None
        and tuple(candidate.fir_kernel.shape) == (3, 2)
        and candidate.fir_bias is None
        and tuple(candidate.writer_weight.shape) == (3, 3)
        and candidate.writer_bias is None
        for candidate in frontier.raw_candidates
    )
    assert (
        frontier.report["adapter_provenance"]["dynamic_source_target_topology"] is True
    )


def test_real_adapter_is_case_order_deterministic_and_request_immutable() -> None:
    first_request = _request()
    second_request = _request(reverse_cases=True)
    first_request_before = copy.deepcopy(first_request.as_dict())
    second_request_before = copy.deepcopy(second_request.as_dict())
    adapter = Lfm2CpcfShortConvV5NumericalFrontierAdapter()

    first = adapter.generate(first_request)
    second = adapter.generate(second_request)

    assert first_request.as_dict() == first_request_before
    assert second_request.as_dict() == second_request_before
    assert first_request.request_sha256 == second_request.request_sha256
    assert first.as_dict() == second.as_dict()
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            (candidate.in_proj_weight for candidate in first.raw_candidates),
            (candidate.in_proj_weight for candidate in second.raw_candidates),
        )
    )


def test_real_adapter_fails_closed_on_actual_fir_commutator_distortion() -> None:
    distorted = torch.tensor(
        [
            [0.75, -0.25, 0.125],
            [1.10, 0.40, -0.30],
            [-0.80, 0.65, 0.20],
            [0.15, -1.20, 0.90],
        ],
        dtype=torch.float64,
    )

    with pytest.raises(
        Lfm2CpcfShortConvV5NumericalAdapterError,
        match="commutator residual",
    ) as captured:
        Lfm2CpcfShortConvV5NumericalFrontierAdapter().generate(
            _request(source_fir_rows=distorted)
        )
    diagnostics = captured.value.diagnostics
    assert diagnostics is not None
    assert diagnostics["kind"] == "fir_commutator_gate_failure_evidence"
    assert diagnostics["requested_gate_passed"] is False
    assert (
        diagnostics["observed_maximum_per_lag_commutator_relative_frobenius"]
        > diagnostics["requested_maximum_per_lag_commutator_relative_frobenius"]
    )
    assert len(diagnostics["per_lag_commutator_relative_frobenius"]) == 2
    assert (
        diagnostics["audit_basis_report"]["full_source_numeric_participation"] is True
    )
