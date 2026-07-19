from __future__ import annotations

import copy
import hashlib
import json

import pytest
import torch

from neural_forge import (
    hf_lfm2_cpcf_short_conv_v5_numerical_adapter as numerical_adapter_module,
)
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
    Lfm2CpcfV5FirBasisPolicy,
    build_fir_commuting_balanced_all_channel_basis,
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
    target_dtype: str = "float64",
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
            target_dtype=target_dtype,
        ),
    )


def _bfloat16_request() -> Lfm2CpcfShortConvNumericalFrontierRequest:
    """Mirror real BF16 evidence with an FP64 boundary-chart encoder."""

    source = _request(target_dtype="bfloat16")

    def bf16(value):
        return None if value is None else value.to(dtype=torch.bfloat16)

    cases = tuple(
        Lfm2CpcfShortConvV4FoldCase(
            case_id=case.case_id,
            split=case.split,
            suffix_start=case.suffix_start,
            actual_child_operator_input_history=bf16(
                case.actual_child_operator_input_history
            ),
            parent_operator_input_history=bf16(
                case.parent_operator_input_history
            ),
            parent_b_history=bf16(case.parent_b_history),
            parent_c_history=bf16(case.parent_c_history),
            parent_x_history=bf16(case.parent_x_history),
            parent_post_operator_residual_target_space_suffix=bf16(
                case.parent_post_operator_residual_target_space_suffix
            ),
            actual_child_pre_operator_residual_suffix=bf16(
                case.actual_child_pre_operator_residual_suffix
            ),
            valid_prefix_mask=case.valid_prefix_mask,
        )
        for case in source.cases
    )
    return Lfm2CpcfShortConvNumericalFrontierRequest(
        source_tensors={
            role: (
                value.clone()
                if role == "output_encoder"
                else bf16(value)
            )
            for role, value in source.source_tensors.items()
        },
        cases=cases,
        parent_contexts={
            case_id: bf16(value)
            for case_id, value in source.parent_contexts.items()
        },
        target_templates={
            role: bf16(value)
            for role, value in source.target_templates.items()
        },
        policy=source.policy,
    )


def _sequence_sha256(values) -> str:
    payload = json.dumps(
        values,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


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


def test_audited_basis_computes_once_and_reseals_exact_requested_report(
    monkeypatch,
) -> None:
    request = _request()
    resized_fir = numerical_adapter_module._resize_fir_rows(
        request.source_tensors["source_fir_kernel"],
        int(request.report["target_fir_length"]),
    )
    descriptors = numerical_adapter_module._basis_cases(request)
    requested_policy = Lfm2CpcfV5FirBasisPolicy()
    expected = build_fir_commuting_balanced_all_channel_basis(
        descriptors,
        source_fir_kernel=resized_fir,
        target_channels=int(request.report["target_hidden_size"]),
        policy=requested_policy,
    )
    calls = []
    original_builder = (
        numerical_adapter_module.build_fir_commuting_balanced_all_channel_basis
    )

    def counted_builder(*args, **kwargs):
        result = original_builder(*args, **kwargs)
        calls.append((kwargs["policy"], result))
        return result

    monkeypatch.setattr(
        numerical_adapter_module,
        "build_fir_commuting_balanced_all_channel_basis",
        counted_builder,
    )
    actual = numerical_adapter_module._build_audited_basis(
        request,
        descriptors=descriptors,
        resized_fir=resized_fir,
        target_hidden=int(request.report["target_hidden_size"]),
        requested_policy=requested_policy,
    )

    assert len(calls) == 1
    audit_policy, audit = calls[0]
    assert audit.verify_authentication()
    assert audit_policy.maximum_per_lag_commutator_relative_frobenius == 1e300
    assert audit.report["policy_sha256"] == audit_policy.authentication_sha256
    assert actual.verify_authentication()
    assert actual.report["policy_sha256"] == requested_policy.authentication_sha256
    assert torch.equal(actual.encoder, expected.encoder)
    assert torch.equal(actual.decoder, expected.decoder)
    assert torch.equal(actual.target_fir_kernel, expected.target_fir_kernel)
    assert actual.source_to_target_group == expected.source_to_target_group
    assert dict(actual.report) == dict(expected.report)


def test_real_adapter_is_byte_deterministic_within_one_runtime() -> None:
    first = Lfm2CpcfShortConvV5NumericalFrontierAdapter().generate(_request())
    second = Lfm2CpcfShortConvV5NumericalFrontierAdapter().generate(_request())

    assert first.report["report_sha256"] == second.report["report_sha256"]
    assert (
        first.report["adapter_provenance"]["basis_report_sha256"]
        == second.report["adapter_provenance"]["basis_report_sha256"]
    )
    assert len(first.raw_candidates) == len(second.raw_candidates) == 72
    assert _sequence_sha256(
        [candidate.candidate_id for candidate in first.raw_candidates]
    ) == _sequence_sha256(
        [candidate.candidate_id for candidate in second.raw_candidates]
    )
    assert _sequence_sha256(
        [candidate.report["report_sha256"] for candidate in first.raw_candidates]
    ) == _sequence_sha256(
        [candidate.report["report_sha256"] for candidate in second.raw_candidates]
    )
    assert _sequence_sha256(
        [candidate.state_sha256 for candidate in first.raw_candidates]
    ) == _sequence_sha256(
        [candidate.state_sha256 for candidate in second.raw_candidates]
    )


def test_projection_path_reuses_one_authenticated_replay_bank_per_state(
    monkeypatch,
) -> None:
    request = _request()
    calls = []
    original = numerical_adapter_module.replay_sampled_cp_volterra

    def counted_replay(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        numerical_adapter_module,
        "replay_sampled_cp_volterra",
        counted_replay,
    )

    frontier = Lfm2CpcfShortConvV5NumericalFrontierAdapter().generate(request)

    assert frontier.verify_authentication()
    # Per condition-cap path: one replay bank for the transported prior and
    # one for each of the six attempted phase candidates.  The same bank is
    # consumed by the solver and its before/after objective instead of
    # replaying the identical state three times.
    assert len(calls) == 2 * (1 + 6) * len(request.cases)


def test_real_adapter_promotes_bfloat16_evidence_before_fp64_folding(
    monkeypatch,
) -> None:
    request = _bfloat16_request()
    captured_projection_cases = []
    original_projection_cases = numerical_adapter_module._projection_cases

    def capture_projection_cases(*args, **kwargs):
        result = original_projection_cases(*args, **kwargs)
        captured_projection_cases.extend(result)
        return result

    monkeypatch.setattr(
        numerical_adapter_module,
        "_projection_cases",
        capture_projection_cases,
    )
    frontier = Lfm2CpcfShortConvV5NumericalFrontierAdapter().generate(request)

    assert request.verify_authentication()
    assert frontier.verify_authentication()
    assert captured_projection_cases
    for case in captured_projection_cases:
        assert case.child_input_history.dtype == torch.float64
        assert case.desired_context_history.dtype == torch.float64
        assert case.desired_write_suffix.dtype == torch.float64
        assert case.valid_prefix_mask.dtype == torch.bool
        assert case.valid_suffix_mask.dtype == torch.bool
    assert all(
        tensor.dtype == torch.float64
        for candidate in frontier.raw_candidates
        for tensor in candidate.state_dict().values()
        if tensor is not None
    )
    assert all(
        candidate.verify_authentication()
        for candidate in frontier.raw_candidates
    )
    provenance = frontier.report["adapter_provenance"]
    assert provenance["fit_projection_case_count"] == len(request.cases)
    assert provenance["training"] is False
    assert provenance["quantization"] is False
    assert provenance["pruning"] is False


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


def test_adapter_dispatches_scalable_endpoint_residual_bcx(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        numerical_adapter_module,
        "_SCALABLE_BCX_MINIMUM_HIDDEN_SIZE",
        1,
    )

    frontier = Lfm2CpcfShortConvV5NumericalFrontierAdapter().generate(
        _request(zero_writer=True)
    )

    assert frontier.verify_authentication()
    provenance = frontier.report["adapter_provenance"]
    assert provenance["bcx_solver_backend_inventory"] == {
        "kappa-01000": "global_full_coordinate_matrix_free_residual",
        "kappa-10000": (
            "coupled_endpoint_residual_subspace_coordinate"
        ),
    }
    assert provenance["bcx_endpoint_residual_aware"] is True
    assert provenance["bcx_dense_transported_prior_preserved"] is True
    assert provenance["causal_horizon_phase_acceptance"] == {
        "uniform_mass": 0.5,
        "exponential_tail_mass": 0.5,
        "tail_tau": "max(4,sqrt(valid_rows))",
        "uniform_reference_metric_retained": True,
    }


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
