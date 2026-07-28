from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_parent_shell_writer_catchup import (
    compile_lfm2_parent_shell_short_conv_local_writer,
    lfm2_parent_shell_local_writer_evidence_sha256,
    lfm2_parent_shell_writer_state_sha256,
)
from neural_forge.hf_lfm2_partial_short_conv_evidence import (
    Lfm2PartialShortConvCachedPhase,
    Lfm2PartialShortConvEvidenceError,
    compile_lfm2_partial_short_conv_parent_shell_evidence,
    lfm2_partial_short_conv_cached_phase_evidence_sha256,
    lfm2_partial_short_conv_qdeim_candidate_sha256,
)


LOCAL_ROLE = "compact-short-conv-local-output"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _pre_state(
    b: torch.Tensor,
    x: torch.Tensor,
    *,
    prefix_tokens: int,
    fir_length: int,
) -> torch.Tensor:
    if prefix_tokens == 0:
        return torch.zeros((b.shape[0], b.shape[2], fir_length), dtype=b.dtype)
    state = (b[:, :prefix_tokens] * x[:, :prefix_tokens]).transpose(1, 2)
    if state.shape[-1] < fir_length:
        state = torch.nn.functional.pad(state, (fir_length - state.shape[-1], 0))
    return state[..., -fir_length:].contiguous()


def _reference_context(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    pre_state: torch.Tensor,
    selected: tuple[int, ...],
    kernel: torch.Tensor,
    *,
    sampled_tokens: int,
) -> torch.Tensor:
    index = torch.tensor(selected, dtype=torch.long)
    state = pre_state.index_select(1, index).float().clone()
    start = b.shape[1] - sampled_tokens
    result = []
    for token in range(start, b.shape[1]):
        bx = (
            b[:, token].index_select(1, index) * x[:, token].index_select(1, index)
        ).float()
        state = torch.cat((state[..., 1:], bx.unsqueeze(-1)), dim=-1)
        filtered = (state * kernel[:, 0].unsqueeze(0)).sum(dim=-1)
        result.append(c[:, token].index_select(1, index).float() * filtered)
    return torch.stack(result, dim=1).contiguous()


def _cached_phase(
    *,
    case_id: str,
    phase: str,
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    selected: tuple[int, ...],
    kernel: torch.Tensor,
    writer: torch.Tensor,
    sampled_tokens: int,
) -> tuple[Lfm2PartialShortConvCachedPhase, torch.Tensor]:
    prefix = 0 if phase == "prefill" else b.shape[1] - sampled_tokens
    state = _pre_state(
        b,
        x,
        prefix_tokens=prefix,
        fir_length=kernel.shape[-1],
    )
    context = _reference_context(
        b,
        c,
        x,
        state,
        selected,
        kernel,
        sampled_tokens=sampled_tokens,
    )
    batch = b.shape[0]
    hidden = writer.shape[0]
    parent_pre = (
        torch.arange(batch * sampled_tokens * hidden, dtype=torch.float32).reshape(
            batch, sampled_tokens, hidden
        )
        / 19.0
    )
    action = context @ writer.T
    parent_context = torch.zeros(
        (batch, sampled_tokens, b.shape[-1]), dtype=torch.float32
    )
    parent_context[..., torch.tensor(selected, dtype=torch.long)] = context
    value = Lfm2PartialShortConvCachedPhase(
        case_id=case_id,
        phase=phase,
        parent_pre_operator_residual=parent_pre,
        parent_post_operator_residual=(parent_pre + action).contiguous(),
        parent_short_conv_o=action.to(b.dtype).contiguous(),
        parent_b_history=b,
        parent_c_history=c,
        parent_x_history=x,
        parent_context_from_fir=parent_context.contiguous(),
        parent_pre_bx_state=state,
        row_sample_weights=torch.linspace(
            0.5,
            1.5,
            batch * sampled_tokens,
            dtype=torch.float64,
        ).reshape(batch, sampled_tokens),
        row_ids=tuple(
            f"{case_id}:batch:{batch_index:03d}:token:{token_index:04d}"
            for batch_index in range(batch)
            for token_index in range(sampled_tokens)
        ),
        parent_trajectory_receipt_sha256=_digest(f"parent-{case_id}"),
        cached_phase_receipt_sha256=_digest(f"phase-{case_id}"),
        aligned_row_identity_sha256=_digest(f"rows-{case_id}"),
    )
    return value, context


def _compile(
    phases: tuple[Lfm2PartialShortConvCachedPhase, ...],
    selected: tuple[int, ...],
    kernel: torch.Tensor,
    *,
    partition: str = "fit",
):
    source_channels = phases[0].parent_b_history.shape[-1]
    return compile_lfm2_partial_short_conv_parent_shell_evidence(
        cached_phases=phases,
        partition=partition,
        expected_cached_phase_evidence_sha256=(
            lfm2_partial_short_conv_cached_phase_evidence_sha256(
                phases, partition=partition
            )
        ),
        selected_source_channels=selected,
        candidate_fir_kernel=kernel,
        expected_qdeim_candidate_sha256=(
            lfm2_partial_short_conv_qdeim_candidate_sha256(
                selected_source_channels=selected,
                candidate_fir_kernel=kernel,
                source_internal_channels=source_channels,
            )
        ),
    )


def _fixture(dtype: torch.dtype = torch.float32):
    b = torch.tensor(
        [
            [
                [0.5, -1.0, 0.25, 2.0],
                [1.0, 0.5, -0.75, 0.25],
                [-0.5, 1.5, 0.5, -1.0],
                [2.0, -0.25, 1.0, 0.75],
            ],
            [
                [-0.25, 0.75, 1.0, -0.5],
                [0.5, -1.5, 0.25, 1.0],
                [1.5, 0.5, -0.5, 0.25],
                [-1.0, 2.0, 0.75, -0.25],
            ],
        ],
        dtype=dtype,
    )
    c = torch.tensor(
        [
            [
                [1.0, 0.5, -1.0, 0.25],
                [0.75, -0.5, 1.5, 1.0],
                [-0.25, 2.0, 0.5, -1.5],
                [1.25, 0.25, -0.75, 0.5],
            ],
            [
                [0.5, -0.25, 1.0, 1.5],
                [-1.0, 0.75, 0.25, -0.5],
                [1.5, 1.0, -0.25, 0.75],
                [0.25, -1.5, 0.5, 1.0],
            ],
        ],
        dtype=dtype,
    )
    x = torch.tensor(
        [
            [
                [1.0, -0.5, 0.75, 0.25],
                [-0.25, 1.5, 0.5, -1.0],
                [0.5, 0.25, -1.5, 2.0],
                [1.25, -0.75, 0.25, 0.5],
            ],
            [
                [0.75, 1.0, -0.5, 0.25],
                [1.5, -0.25, 0.75, -1.0],
                [-0.5, 2.0, 0.25, 1.0],
                [0.25, -1.0, 1.5, 0.5],
            ],
        ],
        dtype=dtype,
    )
    selected = (3, 1)
    kernel = torch.tensor(
        [[[0.25, -0.5, 1.0]], [[-0.75, 0.5, 0.25]]], dtype=torch.float32
    )
    writer = torch.tensor(
        [[0.75, -0.25], [-0.5, 1.25], [1.5, 0.5]], dtype=torch.float32
    )
    return b.contiguous(), c.contiguous(), x.contiguous(), selected, kernel, writer


def test_prefill_reconstructs_selected_qdeim_context_and_flattens_rows() -> None:
    b, c, x, selected, kernel, writer = _fixture(torch.bfloat16)
    phase, expected_context = _cached_phase(
        case_id="trajectory-000-prefill",
        phase="prefill",
        b=b,
        c=c,
        x=x,
        selected=selected,
        kernel=kernel,
        writer=writer,
        sampled_tokens=b.shape[1],
    )
    snapshots = {
        "b": b.clone(),
        "c": c.clone(),
        "x": x.clone(),
        "kernel": kernel.clone(),
    }

    compiled = _compile((phase,), selected, kernel)
    case = compiled.evidence_cases[0]

    assert torch.equal(b, snapshots["b"])
    assert torch.equal(c, snapshots["c"])
    assert torch.equal(x, snapshots["x"])
    assert torch.equal(kernel, snapshots["kernel"])
    assert case.compact_writer_design.dtype == torch.float32
    assert case.compact_writer_design.is_contiguous()
    assert torch.equal(
        case.compact_writer_design,
        expected_context.reshape(-1, len(selected)),
    )
    assert torch.equal(
        case.parent_pre_operator_residual,
        phase.parent_pre_operator_residual.reshape(-1, writer.shape[0]),
    )
    assert torch.equal(
        case.parent_local_action,
        phase.parent_short_conv_o.float().reshape(-1, writer.shape[0]),
    )
    assert torch.equal(
        case.row_sample_weights,
        phase.row_sample_weights.float().reshape(-1),
    )
    assert compiled.report["output"] == {
        "evidence_case_count": 1,
        "flattened_row_count": b.shape[0] * b.shape[1],
        "parent_hidden_width": 3,
        "compact_writer_design_width": 2,
        "parent_shell_writer_evidence_sha256": (
            lfm2_parent_shell_local_writer_evidence_sha256(
                compiled.evidence_cases, partition="fit"
            )
        ),
        "emitted_type": "Lfm2ParentShellLocalWriterEvidenceCase",
        "row_alignment": "phase action rows paired with final causal history rows",
        "flatten_order": "batch-major-then-sequence-major",
        "dense_independent_fp32_tensors": True,
        "local_action_target_source": "canonical-cached-short-conv-o",
        "residual_endpoint_subtraction_used_for_target": False,
        "stable_row_ids": list(phase.row_ids),
        "preferred_design_source": (
            "authenticated-parent-context-from-fir-selected-intact-channels"
        ),
    }
    replay = compiled.report["phase_replays"][0]["causal_replay_receipt"]
    replay_body = dict(replay)
    replay_seal = replay_body.pop("result_sha256")
    assert canonical_sha256(replay_body) == replay_seal
    assert replay["prefill_zero_state_used"] is True
    assert compiled.report["operation_audit"]["model_objects_constructed"] == 0
    assert compiled.report["operation_audit"]["parent_model_forward_calls"] == 0
    assert compiled.report["operation_audit"]["filesystem_reads"] == 0
    assert compiled.verify() == compiled.report["result_sha256"]


def test_decode_uses_authenticated_pre_bx_tail_and_only_samples_suffix() -> None:
    b, c, x, selected, kernel, writer = _fixture()
    b = torch.cat((b, (b[:, -1:] * 0.375).contiguous()), dim=1)
    c = torch.cat((c, (c[:, -1:] * -0.625).contiguous()), dim=1)
    x = torch.cat((x, (x[:, -1:] * 1.25).contiguous()), dim=1)
    phase, expected_context = _cached_phase(
        case_id="trajectory-000-decode-000",
        phase="decode.000",
        b=b,
        c=c,
        x=x,
        selected=selected,
        kernel=kernel,
        writer=writer,
        sampled_tokens=1,
    )
    # Canonical decode caches contain only the newly executed suffix while the
    # independent pre-Bx cache state carries the preceding causal history.
    phase = replace(
        phase,
        parent_b_history=phase.parent_b_history[:, -1:].contiguous(),
        parent_c_history=phase.parent_c_history[:, -1:].contiguous(),
        parent_x_history=phase.parent_x_history[:, -1:].contiguous(),
    )

    compiled = _compile((phase,), selected, kernel)
    case = compiled.evidence_cases[0]

    assert tuple(case.compact_writer_design.shape) == (b.shape[0], len(selected))
    assert torch.equal(
        case.compact_writer_design,
        expected_context.reshape(b.shape[0], len(selected)),
    )
    replay = compiled.report["phase_replays"][0]["causal_replay_receipt"]
    assert replay["decode_authenticated_pre_bx_state_used"] is True
    assert replay["sampled_history_start"] == 0
    assert replay["sampled_history_stop"] == 1
    assert replay["diagnostic_local_replay_matches_direct_context"] is True
    assert compiled.report["diagnostics"]["decode_phase_count"] == 1

    corrupted = replace(
        phase,
        parent_pre_bx_state=(phase.parent_pre_bx_state + 0.125).contiguous(),
    )
    with pytest.raises(
        Lfm2PartialShortConvEvidenceError,
        match="phase evidence authentication failed",
    ):
        compile_lfm2_partial_short_conv_parent_shell_evidence(
            cached_phases=(corrupted,),
            partition="fit",
            expected_cached_phase_evidence_sha256=(
                lfm2_partial_short_conv_cached_phase_evidence_sha256(
                    (phase,), partition="fit"
                )
            ),
            selected_source_channels=selected,
            candidate_fir_kernel=kernel,
            expected_qdeim_candidate_sha256=(
                lfm2_partial_short_conv_qdeim_candidate_sha256(
                    selected_source_channels=selected,
                    candidate_fir_kernel=kernel,
                    source_internal_channels=b.shape[-1],
                )
            ),
        )


def test_writer_ready_cases_feed_parent_shell_local_writer_without_model_replay() -> (
    None
):
    b, c, x, selected, kernel, writer = _fixture()
    phase, _ = _cached_phase(
        case_id="writer-fit-prefill",
        phase="prefill",
        b=b,
        c=c,
        x=x,
        selected=selected,
        kernel=kernel,
        writer=writer,
        sampled_tokens=b.shape[1],
    )
    phase = replace(
        phase,
        parent_pre_operator_residual=(phase.parent_pre_operator_residual + 500.0),
        parent_post_operator_residual=(phase.parent_post_operator_residual - 250.0),
    )
    partial = _compile((phase,), selected, kernel)
    prior = torch.zeros_like(writer)
    writer_id = "model.layers.0.conv.out_proj"

    fitted = compile_lfm2_parent_shell_short_conv_local_writer(
        writer_id=writer_id,
        structural_out_weight=prior,
        expected_structural_writer_state_sha256=(
            lfm2_parent_shell_writer_state_sha256(
                writer_id=writer_id,
                writer_role=LOCAL_ROLE,
                weight=prior,
            )
        ),
        fit_cases=partial.evidence_cases,
        expected_fit_evidence_sha256=partial.report["output"][
            "parent_shell_writer_evidence_sha256"
        ],
        expected_heldout_evidence_sha256=(
            lfm2_parent_shell_local_writer_evidence_sha256((), partition="heldout")
        ),
        ridge=1.0e-8,
    )

    assert torch.allclose(fitted.weight_compiled, writer, atol=2.0e-5)
    assert fitted.report["target"]["cached_short_conv_o_has_target_authority"]
    assert (
        fitted.report["target"]["residual_pre_post_difference_has_target_authority"]
        is False
    )
    assert fitted.report["solver"]["closed_form_linear_solve_count"] == 1
    assert fitted.report["operation_audit"]["model_forward_calls"] == 0


def test_fit_and_heldout_partitions_are_bound_into_distinct_output_evidence() -> None:
    b, c, x, selected, kernel, writer = _fixture()
    phase, _ = _cached_phase(
        case_id="partition-prefill",
        phase="prefill",
        b=b,
        c=c,
        x=x,
        selected=selected,
        kernel=kernel,
        writer=writer,
        sampled_tokens=b.shape[1],
    )

    fit = _compile((phase,), selected, kernel, partition="fit")
    heldout = _compile((phase,), selected, kernel, partition="heldout")

    assert fit.evidence_cases[0].compact_replay_receipt_sha256 == (
        heldout.evidence_cases[0].compact_replay_receipt_sha256
    )
    assert (
        fit.report["output"]["parent_shell_writer_evidence_sha256"]
        != heldout.report["output"]["parent_shell_writer_evidence_sha256"]
    )
    assert fit.report["partition"] == "fit"
    assert heldout.report["partition"] == "heldout"


def test_authentication_shapes_selection_and_dtype_fail_closed() -> None:
    b, c, x, selected, kernel, writer = _fixture()
    phase, _ = _cached_phase(
        case_id="auth-prefill",
        phase="prefill",
        b=b,
        c=c,
        x=x,
        selected=selected,
        kernel=kernel,
        writer=writer,
        sampled_tokens=b.shape[1],
    )
    phase_digest = lfm2_partial_short_conv_cached_phase_evidence_sha256(
        (phase,), partition="fit"
    )
    candidate_digest = lfm2_partial_short_conv_qdeim_candidate_sha256(
        selected_source_channels=selected,
        candidate_fir_kernel=kernel,
        source_internal_channels=b.shape[-1],
    )

    with pytest.raises(
        Lfm2PartialShortConvEvidenceError,
        match="phase evidence authentication failed",
    ):
        compile_lfm2_partial_short_conv_parent_shell_evidence(
            cached_phases=(phase,),
            partition="fit",
            expected_cached_phase_evidence_sha256="0" * 64,
            selected_source_channels=selected,
            candidate_fir_kernel=kernel,
            expected_qdeim_candidate_sha256=candidate_digest,
        )

    with pytest.raises(
        Lfm2PartialShortConvEvidenceError, match="nonempty, unique, and in source"
    ):
        compile_lfm2_partial_short_conv_parent_shell_evidence(
            cached_phases=(phase,),
            partition="fit",
            expected_cached_phase_evidence_sha256=phase_digest,
            selected_source_channels=(3, 3),
            candidate_fir_kernel=kernel,
            expected_qdeim_candidate_sha256=candidate_digest,
        )

    with pytest.raises(
        Lfm2PartialShortConvEvidenceError, match="candidate_fir_kernel must be"
    ):
        lfm2_partial_short_conv_qdeim_candidate_sha256(
            selected_source_channels=selected,
            candidate_fir_kernel=kernel.to(torch.bfloat16),
            source_internal_channels=b.shape[-1],
        )

    bad_weights = replace(
        phase,
        row_sample_weights=torch.ones(
            (phase.row_sample_weights.numel(),), dtype=torch.float64
        ),
    )
    with pytest.raises(Lfm2PartialShortConvEvidenceError, match="rank-2 cached tensor"):
        lfm2_partial_short_conv_cached_phase_evidence_sha256(
            (bad_weights,), partition="fit"
        )

    grad_phase = replace(
        phase,
        parent_b_history=phase.parent_b_history.clone().requires_grad_(True),
    )
    with pytest.raises(
        Lfm2PartialShortConvEvidenceError, match="exact detached finite"
    ):
        lfm2_partial_short_conv_cached_phase_evidence_sha256(
            (grad_phase,), partition="fit"
        )


def test_prefill_requires_zero_state_full_sampling_and_unique_case_ids() -> None:
    b, c, x, selected, kernel, writer = _fixture()
    phase, _ = _cached_phase(
        case_id="prefill-contract",
        phase="prefill",
        b=b,
        c=c,
        x=x,
        selected=selected,
        kernel=kernel,
        writer=writer,
        sampled_tokens=b.shape[1],
    )

    nonzero = replace(
        phase,
        parent_pre_bx_state=torch.ones_like(phase.parent_pre_bx_state),
    )
    with pytest.raises(
        Lfm2PartialShortConvEvidenceError, match="exactly zero pre-Bx state"
    ):
        lfm2_partial_short_conv_cached_phase_evidence_sha256(
            (nonzero,), partition="fit"
        )

    short_action = replace(
        phase,
        parent_pre_operator_residual=(
            phase.parent_pre_operator_residual[:, -1:].contiguous()
        ),
        parent_post_operator_residual=(
            phase.parent_post_operator_residual[:, -1:].contiguous()
        ),
        row_sample_weights=phase.row_sample_weights[:, -1:].contiguous(),
        row_ids=tuple(
            phase.row_ids[index * b.shape[1] + b.shape[1] - 1]
            for index in range(b.shape[0])
        ),
    )
    with pytest.raises(
        Lfm2PartialShortConvEvidenceError, match="sample its full history"
    ):
        lfm2_partial_short_conv_cached_phase_evidence_sha256(
            (short_action,), partition="fit"
        )

    with pytest.raises(
        Lfm2PartialShortConvEvidenceError, match="case IDs must be unique"
    ):
        lfm2_partial_short_conv_cached_phase_evidence_sha256(
            (phase, phase), partition="fit"
        )


def test_output_receipt_detects_case_tensor_mutation() -> None:
    b, c, x, selected, kernel, writer = _fixture()
    phase, _ = _cached_phase(
        case_id="mutation-prefill",
        phase="prefill",
        b=b,
        c=c,
        x=x,
        selected=selected,
        kernel=kernel,
        writer=writer,
        sampled_tokens=b.shape[1],
    )
    result = _compile((phase,), selected, kernel)
    body = result.as_dict()
    seal = body.pop("result_sha256")

    assert canonical_sha256(body) == seal
    result.evidence_cases[0].compact_writer_design[0, 0] += 1.0
    with pytest.raises(
        Lfm2PartialShortConvEvidenceError, match="changed after sealing"
    ):
        result.verify()
