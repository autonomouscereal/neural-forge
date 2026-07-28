from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_parent_shell_writer_catchup import (
    Lfm2ParentShellCatchupEvidenceCase,
    Lfm2ParentShellLocalWriterEvidenceCase,
    Lfm2ParentShellWriterCatchupError,
    build_lfm2_parent_shell_catchup_replay_binding,
    compile_lfm2_parent_shell_one_downstream_catchup_writer,
    compile_lfm2_parent_shell_short_conv_local_writer,
    lfm2_parent_shell_catchup_evidence_sha256,
    lfm2_parent_shell_local_writer_evidence_sha256,
    lfm2_parent_shell_writer_state_sha256,
)


LOCAL_ROLE = "compact-short-conv-local-output"
CATCHUP_ROLE = "one-downstream-absolute-endpoint-catchup"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _local_case(
    case_id: str,
    design: torch.Tensor,
    action: torch.Tensor,
    *,
    receipt_suffix: str = "",
) -> Lfm2ParentShellLocalWriterEvidenceCase:
    parent_pre = (
        torch.arange(design.shape[0] * action.shape[1], dtype=torch.float64).reshape(
            design.shape[0], action.shape[1]
        )
        / 17.0
    )
    return Lfm2ParentShellLocalWriterEvidenceCase(
        case_id=case_id,
        compact_writer_design=design.contiguous(),
        parent_local_action=action.contiguous(),
        parent_pre_operator_residual=parent_pre,
        parent_post_operator_residual=(parent_pre + action).contiguous(),
        row_sample_weights=torch.linspace(
            0.5, 1.5, design.shape[0], dtype=torch.float64
        ),
        row_ids=tuple(f"{case_id}:row:{index:04d}" for index in range(design.shape[0])),
        parent_trajectory_receipt_sha256=_digest(f"parent-{case_id}-{receipt_suffix}"),
        compact_replay_receipt_sha256=_digest(f"compact-{case_id}-{receipt_suffix}"),
        aligned_row_identity_sha256=_digest(f"rows-{case_id}-{receipt_suffix}"),
    )


def _catchup_case(
    case_id: str,
    design: torch.Tensor,
    action: torch.Tensor,
    *,
    receipt_suffix: str = "",
) -> Lfm2ParentShellCatchupEvidenceCase:
    live_pre = (
        torch.arange(design.shape[0] * action.shape[1], dtype=torch.float64).reshape(
            design.shape[0], action.shape[1]
        )
        / 23.0
    )
    return Lfm2ParentShellCatchupEvidenceCase(
        case_id=case_id,
        downstream_writer_design=design.contiguous(),
        live_pre_downstream_residual=live_pre,
        parent_absolute_endpoint=(live_pre + action).contiguous(),
        row_sample_weights=torch.linspace(
            0.75, 1.25, design.shape[0], dtype=torch.float64
        ),
        row_ids=tuple(f"{case_id}:row:{index:04d}" for index in range(design.shape[0])),
        parent_trajectory_receipt_sha256=_digest(
            f"parent-catchup-{case_id}-{receipt_suffix}"
        ),
        post_local_replay_receipt_sha256=_digest(
            f"post-local-{case_id}-{receipt_suffix}"
        ),
        aligned_row_identity_sha256=_digest(f"catchup-rows-{case_id}-{receipt_suffix}"),
    )


def _compile_local(
    prior: torch.Tensor,
    fit: tuple[Lfm2ParentShellLocalWriterEvidenceCase, ...],
    heldout: tuple[Lfm2ParentShellLocalWriterEvidenceCase, ...] = (),
    *,
    bias: torch.Tensor | None = None,
    fit_unpenalized_bias: bool = False,
):
    writer_id = "model.layers.0.conv.out_proj"
    return compile_lfm2_parent_shell_short_conv_local_writer(
        writer_id=writer_id,
        structural_out_weight=prior,
        structural_out_bias=bias,
        expected_structural_writer_state_sha256=(
            lfm2_parent_shell_writer_state_sha256(
                writer_id=writer_id,
                writer_role=LOCAL_ROLE,
                weight=prior,
                bias=bias,
            )
        ),
        fit_cases=fit,
        expected_fit_evidence_sha256=(
            lfm2_parent_shell_local_writer_evidence_sha256(fit, partition="fit")
        ),
        heldout_cases=heldout,
        expected_heldout_evidence_sha256=(
            lfm2_parent_shell_local_writer_evidence_sha256(heldout, partition="heldout")
        ),
        ridge=1.0e-10,
        fit_unpenalized_bias=fit_unpenalized_bias,
    )


def _compile_catchup(
    prior: torch.Tensor,
    fit: tuple[Lfm2ParentShellCatchupEvidenceCase, ...],
    heldout: tuple[Lfm2ParentShellCatchupEvidenceCase, ...] = (),
):
    writer_id = "model.layers.1.feed_forward.w2"
    replay_binding = build_lfm2_parent_shell_catchup_replay_binding(
        upstream_local_writer_result_sha256=_digest("local-writer-result"),
        downstream_writer_id=writer_id,
        fit_cases=fit,
        heldout_cases=heldout,
    )
    return compile_lfm2_parent_shell_one_downstream_catchup_writer(
        downstream_writer_id=writer_id,
        structural_downstream_weight=prior,
        expected_structural_writer_state_sha256=(
            lfm2_parent_shell_writer_state_sha256(
                writer_id=writer_id,
                writer_role=CATCHUP_ROLE,
                weight=prior,
            )
        ),
        replay_binding=replay_binding,
        fit_cases=fit,
        expected_fit_evidence_sha256=(
            lfm2_parent_shell_catchup_evidence_sha256(fit, partition="fit")
        ),
        heldout_cases=heldout,
        expected_heldout_evidence_sha256=(
            lfm2_parent_shell_catchup_evidence_sha256(heldout, partition="heldout")
        ),
        ridge=1.0e-10,
    )


def test_local_writer_recovers_parent_width_action_without_touching_prior() -> None:
    design = torch.tensor(
        [
            [-2.0, 1.0],
            [-1.0, -1.0],
            [0.5, 2.0],
            [1.0, -2.0],
            [2.0, 0.25],
            [3.0, 1.5],
        ],
        dtype=torch.float64,
    )
    truth = torch.tensor([[0.75, -0.25], [-0.5, 1.25], [1.5, 0.5]], dtype=torch.float64)
    fit = (_local_case("fit-000", design, design @ truth.T),)
    heldout_design = torch.tensor(
        [[-3.0, 0.5], [0.25, -0.75], [4.0, 2.0]], dtype=torch.float64
    )
    heldout = (_local_case("heldout-000", heldout_design, heldout_design @ truth.T),)
    prior = torch.zeros((3, 2), dtype=torch.float32)
    prior_before = prior.clone()

    compiled = _compile_local(prior, fit, heldout)

    assert torch.equal(prior, prior_before)
    assert compiled.weight_compiled.dtype == torch.float32
    assert compiled.weight_compiled.is_contiguous()
    assert compiled.weight_compiled.requires_grad is False
    assert compiled.weight_compiled.grad_fn is None
    assert compiled.weight_compiled.data_ptr() != prior.data_ptr()
    assert torch.allclose(compiled.weight_compiled.double(), truth, atol=1.0e-6)
    assert tuple(compiled.weight_compiled.shape) == (3, 2)
    assert compiled.report["topology"] == {
        "output_width": 3,
        "input_width": 2,
        "weight_shape": [3, 2],
        "bias_shape": None,
        "weight_parameter_elements": 6,
        "bias_parameter_elements": 0,
        "parameter_elements": 6,
        "exact_shape_preserved": True,
        "dense_fp32": True,
        "standalone_independent_storage": True,
        "aliases_structural_prior": False,
        "parameter_sharing_used": False,
    }
    assert compiled.report["target"]["absolute_endpoint_used"] is False
    assert compiled.report["solver"]["closed_form_linear_solve_count"] == 1
    assert compiled.report["operation_audit"]["training_steps"] == 0
    assert compiled.report["operation_audit"]["model_forward_calls"] == 0
    assert compiled.report["operation_audit"]["filesystem_reads"] == 0
    assert compiled.verify() == compiled.report["result_sha256"]
    state = compiled.state_dict()
    assert state["weight"].data_ptr() != compiled.weight_compiled.data_ptr()


def test_heldout_is_authenticated_but_cannot_change_local_writer() -> None:
    design = torch.tensor(
        [[-2.0, 1.0], [-1.0, -1.0], [1.0, 2.0], [2.0, -0.5]],
        dtype=torch.float64,
    )
    truth = torch.tensor([[1.0, 0.25], [-0.5, 2.0]], dtype=torch.float64)
    fit = (_local_case("fit", design, design @ truth.T),)
    heldout_design = torch.tensor([[0.25, 3.0], [4.0, -2.0]], dtype=torch.float64)
    good = (_local_case("good-heldout", heldout_design, heldout_design @ truth.T),)
    adversarial = (
        _local_case(
            "bad-heldout",
            heldout_design + 0.125,
            torch.full((2, 2), 1000.0, dtype=torch.float64),
            receipt_suffix="different",
        ),
    )
    prior = torch.full((2, 2), 0.125, dtype=torch.float32)

    good_result = _compile_local(prior, fit, good)
    bad_result = _compile_local(prior, fit, adversarial)

    assert torch.equal(good_result.weight_compiled, bad_result.weight_compiled)
    assert torch.equal(good_result.weight_delta_fp64, bad_result.weight_delta_fp64)
    assert (
        good_result.report["authentication"]["heldout_evidence_sha256_before"]
        != bad_result.report["authentication"]["heldout_evidence_sha256_before"]
    )
    assert (
        good_result.report["evidence_policy"]["heldout_partition_used_for_construction"]
        is False
    )
    assert (
        bad_result.report["evidence_policy"]["heldout_metrics_can_reject_candidate"]
        is False
    )


def test_local_target_is_direct_cached_action_not_residual_endpoint_subtraction() -> (
    None
):
    design = torch.tensor(
        [[-2.0, 1.0], [-1.0, -1.0], [1.0, 2.0], [2.0, -0.5]],
        dtype=torch.float64,
    )
    truth = torch.tensor([[1.0, 0.25], [-0.5, 2.0]], dtype=torch.float64)
    fit = (_local_case("direct-action", design, design @ truth.T),)
    shifted = (
        replace(
            fit[0],
            parent_pre_operator_residual=(fit[0].parent_pre_operator_residual + 1000.0),
            parent_post_operator_residual=(
                fit[0].parent_post_operator_residual - 750.0
            ),
        ),
    )
    prior = torch.zeros((2, 2), dtype=torch.float32)

    baseline = _compile_local(prior, fit)
    residual_shifted = _compile_local(prior, shifted)

    assert torch.equal(baseline.weight_compiled, residual_shifted.weight_compiled)
    assert baseline.report["target"] == {
        "kind": "canonical-cached-short-conv-o-action",
        "formula": "direct cached short_conv.o rows",
        "cached_short_conv_o_has_target_authority": True,
        "residual_pre_post_difference_has_target_authority": False,
        "residual_pre_post_rows_are_diagnostic_only": True,
        "absolute_endpoint_used": False,
        "upstream_accumulated_defect_absorbed": False,
    }


def test_unpenalized_bias_requires_explicit_existing_bias_topology() -> None:
    design = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=torch.float64)
    slope = torch.tensor([[1.5], [-0.75]], dtype=torch.float64)
    intercept = torch.tensor([2.25, -1.125], dtype=torch.float64)
    fit = (_local_case("fit-bias", design, design @ slope.T + intercept),)
    prior = torch.zeros((2, 1), dtype=torch.float32)
    prior_bias = torch.zeros(2, dtype=torch.float32)

    fixed = _compile_local(prior, fit, bias=prior_bias)
    fitted = _compile_local(
        prior,
        fit,
        bias=prior_bias,
        fit_unpenalized_bias=True,
    )

    assert torch.equal(fixed.bias_compiled, prior_bias)
    assert torch.allclose(fitted.weight_compiled.double(), slope, atol=1.0e-6)
    assert torch.allclose(fitted.bias_compiled.double(), intercept, atol=1.0e-6)
    assert fitted.report["solver"]["bias_delta_penalized"] is False
    assert fitted.report["solver"]["unpenalized_bias_fit_explicitly_enabled"] is True

    writer_id = "model.layers.0.conv.out_proj"
    with pytest.raises(
        Lfm2ParentShellWriterCatchupError,
        match="without an existing bias topology",
    ):
        compile_lfm2_parent_shell_short_conv_local_writer(
            writer_id=writer_id,
            structural_out_weight=prior,
            expected_structural_writer_state_sha256=(
                lfm2_parent_shell_writer_state_sha256(
                    writer_id=writer_id,
                    writer_role=LOCAL_ROLE,
                    weight=prior,
                )
            ),
            fit_cases=fit,
            expected_fit_evidence_sha256=(
                lfm2_parent_shell_local_writer_evidence_sha256(fit, partition="fit")
            ),
            expected_heldout_evidence_sha256=(
                lfm2_parent_shell_local_writer_evidence_sha256((), partition="heldout")
            ),
            fit_unpenalized_bias=True,
        )


def test_one_downstream_catchup_recovers_absolute_endpoint_action() -> None:
    design = torch.tensor(
        [
            [-2.0, 0.5],
            [-1.0, -1.0],
            [0.25, 2.0],
            [1.0, -2.0],
            [2.0, 0.75],
            [3.0, 1.0],
        ],
        dtype=torch.float64,
    )
    truth = torch.tensor([[0.25, 1.5], [-1.0, 0.5], [2.0, -0.75]], dtype=torch.float64)
    fit = (_catchup_case("fit-catchup", design, design @ truth.T),)
    prior = torch.zeros((3, 2), dtype=torch.float32)

    result = _compile_catchup(prior, fit)

    assert torch.allclose(result.weight_compiled.double(), truth, atol=1.0e-6)
    assert result.writer_role == CATCHUP_ROLE
    assert result.report["target"]["absolute_endpoint_used"] is True
    assert result.report["target"]["accumulated_upstream_defect_may_be_absorbed"]
    assert result.report["downstream_scope"] == {
        "writer_count": 1,
        "exactly_one_downstream_writer": True,
        "other_downstream_writers_changed": 0,
        "fresh_post_local_cached_replay_required": True,
    }
    assert result.report["upstream_local_writer_result_sha256"] == _digest(
        "local-writer-result"
    )
    binding = result.report["catchup_replay_binding"]
    assert binding["binding_sha256"] == result.report["catchup_replay_binding_sha256"]
    assert binding["fit"]["rows"][0]["row_id"] == fit[0].row_ids[0]
    assert len(binding["fit"]["rows"]) == design.shape[0]
    assert binding["row_tensor_seals_complete"] is True
    assert result.report["topology"]["parameter_elements"] == 6
    assert result.verify() == result.report["result_sha256"]


def test_compilers_are_deterministic_and_each_execute_one_closed_form_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = torch.tensor(
        [[-1.0, 0.5], [0.0, -2.0], [1.0, 1.5], [2.0, -0.25]],
        dtype=torch.float64,
    )
    truth = torch.tensor([[0.5, -1.0], [1.25, 0.75]], dtype=torch.float64)
    fit = (_local_case("fit-deterministic", design, design @ truth.T),)
    prior = torch.zeros((2, 2), dtype=torch.float32)
    original = torch.linalg.solve
    calls = 0

    def counted_solve(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve", counted_solve)
    first = _compile_local(prior, fit)
    second = _compile_local(prior, fit)

    assert calls == 2
    assert torch.equal(first.weight_compiled, second.weight_compiled)
    assert first.report["result_sha256"] == second.report["result_sha256"]


def test_authentication_partition_and_cached_tensor_contracts_fail_closed() -> None:
    design = torch.tensor([[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    action = torch.tensor([[1.0, -1.0], [0.5, 0.25], [-0.5, 2.0]], dtype=torch.float64)
    fit = (_local_case("same", design, action),)
    prior = torch.zeros((2, 2), dtype=torch.float32)
    writer_id = "model.layers.0.conv.out_proj"
    prior_digest = lfm2_parent_shell_writer_state_sha256(
        writer_id=writer_id,
        writer_role=LOCAL_ROLE,
        weight=prior,
    )
    fit_digest = lfm2_parent_shell_local_writer_evidence_sha256(fit, partition="fit")
    empty_heldout_digest = lfm2_parent_shell_local_writer_evidence_sha256(
        (), partition="heldout"
    )

    with pytest.raises(
        Lfm2ParentShellWriterCatchupError, match="FIT evidence authentication failed"
    ):
        compile_lfm2_parent_shell_short_conv_local_writer(
            writer_id=writer_id,
            structural_out_weight=prior,
            expected_structural_writer_state_sha256=prior_digest,
            fit_cases=fit,
            expected_fit_evidence_sha256="0" * 64,
            expected_heldout_evidence_sha256=empty_heldout_digest,
        )

    with pytest.raises(
        Lfm2ParentShellWriterCatchupError,
        match="stable row ID sets must be disjoint",
    ):
        compile_lfm2_parent_shell_short_conv_local_writer(
            writer_id=writer_id,
            structural_out_weight=prior,
            expected_structural_writer_state_sha256=prior_digest,
            fit_cases=fit,
            expected_fit_evidence_sha256=fit_digest,
            heldout_cases=fit,
            expected_heldout_evidence_sha256=(
                lfm2_parent_shell_local_writer_evidence_sha256(fit, partition="heldout")
            ),
        )

    grad_case = replace(
        fit[0], compact_writer_design=design.clone().requires_grad_(True)
    )
    with pytest.raises(
        Lfm2ParentShellWriterCatchupError, match="exact detached finite"
    ):
        lfm2_parent_shell_local_writer_evidence_sha256((grad_case,), partition="fit")

    parameter_case = replace(
        fit[0], compact_writer_design=torch.nn.Parameter(design.clone())
    )
    with pytest.raises(
        Lfm2ParentShellWriterCatchupError, match="exact detached finite"
    ):
        lfm2_parent_shell_local_writer_evidence_sha256(
            (parameter_case,), partition="fit"
        )

    with pytest.raises(Lfm2ParentShellWriterCatchupError, match="strictly positive"):
        compile_lfm2_parent_shell_short_conv_local_writer(
            writer_id=writer_id,
            structural_out_weight=prior,
            expected_structural_writer_state_sha256=prior_digest,
            fit_cases=fit,
            expected_fit_evidence_sha256=fit_digest,
            expected_heldout_evidence_sha256=empty_heldout_digest,
            ridge=0.0,
        )


def test_result_receipt_and_tensors_detect_post_compile_mutation() -> None:
    design = torch.tensor([[-1.0], [0.0], [1.0], [2.0]], dtype=torch.float64)
    truth = torch.tensor([[2.0], [-0.5]], dtype=torch.float64)
    fit = (_local_case("fit-mutation", design, design @ truth.T),)
    result = _compile_local(torch.zeros((2, 1), dtype=torch.float32), fit)
    receipt = result.as_dict()
    seal = receipt.pop("result_sha256")

    assert canonical_sha256(receipt) == seal
    result.weight_compiled[0, 0] += 1.0
    with pytest.raises(
        Lfm2ParentShellWriterCatchupError, match="tensors changed after sealing"
    ):
        result.verify()


def test_result_accessors_verify_prior_and_fp64_delta_before_returning() -> None:
    design = torch.tensor([[-1.0], [0.0], [1.0], [2.0]], dtype=torch.float64)
    truth = torch.tensor([[2.0], [-0.5]], dtype=torch.float64)
    fit = (_local_case("verify-components", design, design @ truth.T),)

    prior_changed = _compile_local(torch.zeros((2, 1), dtype=torch.float32), fit)
    prior_changed.weight_structural_prior[0, 0] += 1.0
    with pytest.raises(
        Lfm2ParentShellWriterCatchupError, match="prior tensors changed"
    ):
        prior_changed.state_dict()

    delta_changed = _compile_local(torch.zeros((2, 1), dtype=torch.float32), fit)
    delta_changed.weight_delta_fp64[0, 0] += 1.0
    with pytest.raises(Lfm2ParentShellWriterCatchupError, match="deltas changed"):
        delta_changed.as_dict()


def test_catchup_requires_authenticated_upstream_local_result_and_fresh_receipts() -> (
    None
):
    design = torch.tensor([[-1.0], [0.0], [1.0], [2.0]], dtype=torch.float64)
    action = torch.tensor([[1.0], [0.0], [-1.0], [-2.0]], dtype=torch.float64)
    fit = (_catchup_case("fit", design, action),)
    prior = torch.zeros((1, 1), dtype=torch.float32)
    writer_id = "model.layers.1.feed_forward.w2"

    with pytest.raises(Lfm2ParentShellWriterCatchupError, match="upstream local"):
        build_lfm2_parent_shell_catchup_replay_binding(
            upstream_local_writer_result_sha256="not-a-seal",
            downstream_writer_id=writer_id,
            fit_cases=fit,
        )

    binding = build_lfm2_parent_shell_catchup_replay_binding(
        upstream_local_writer_result_sha256=_digest("local-result"),
        downstream_writer_id=writer_id,
        fit_cases=fit,
    )
    changed = (
        replace(
            fit[0],
            parent_absolute_endpoint=(fit[0].parent_absolute_endpoint + 0.25),
        ),
    )
    with pytest.raises(
        Lfm2ParentShellWriterCatchupError,
        match="does not match supplied rows/tensors",
    ):
        compile_lfm2_parent_shell_one_downstream_catchup_writer(
            downstream_writer_id=writer_id,
            structural_downstream_weight=prior,
            expected_structural_writer_state_sha256=(
                lfm2_parent_shell_writer_state_sha256(
                    writer_id=writer_id,
                    writer_role=CATCHUP_ROLE,
                    weight=prior,
                )
            ),
            replay_binding=binding,
            fit_cases=changed,
            expected_fit_evidence_sha256=(
                lfm2_parent_shell_catchup_evidence_sha256(changed, partition="fit")
            ),
            expected_heldout_evidence_sha256=(
                lfm2_parent_shell_catchup_evidence_sha256((), partition="heldout")
            ),
        )
