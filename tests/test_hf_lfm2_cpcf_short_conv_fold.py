from __future__ import annotations

from typing import Any

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_cpcf_short_conv_fold import (  # noqa: E402
    LFM2_CPCF_SHORT_CONV_FOLD_SCHEMA,
    Lfm2CpcfShortConvFoldCase,
    Lfm2CpcfShortConvFoldError,
    Lfm2CpcfShortConvGatePolicy,
    Lfm2CpcfShortConvValidationError,
    compile_lfm2_cpcf_short_conv_fold,
)


_SOURCE_HIDDEN = 6
_TARGET_HIDDEN = 2
_FIR_LENGTH = 3


def _source_fir() -> tuple[Any, Any]:
    row = torch.tensor([0.45, -0.2, 0.65], dtype=torch.float64)
    weight = (
        row.reshape(1, 1, _FIR_LENGTH)
        .expand(_SOURCE_HIDDEN, -1, -1)
        .contiguous()
        .clone()
    )
    bias = torch.full((_SOURCE_HIDDEN,), 0.125, dtype=torch.float64)
    return weight, bias


def _source_context(
    b_history: Any,
    c_history: Any,
    x_history: Any,
    weight: Any,
    bias: Any,
) -> Any:
    batch, sequence, hidden = map(int, b_history.shape)
    gated = (b_history * x_history).transpose(1, 2)
    padded = torch.nn.functional.pad(gated, (_FIR_LENGTH - 1, 0))
    filtered = torch.nn.functional.conv1d(
        padded,
        weight,
        bias,
        groups=hidden,
    )
    assert tuple(filtered.shape) == (batch, hidden, sequence)
    return c_history * filtered.transpose(1, 2)


def _case(
    *,
    case_id: str,
    split: str,
    batch: int,
    sequence: int,
    suffix: int,
    seed: int,
    corrupt_endpoint: bool = False,
    force_suffix_to_require_warm_state: bool = False,
) -> Lfm2CpcfShortConvFoldCase:
    generator = torch.Generator().manual_seed(seed)
    target_input = torch.randn(
        (batch, sequence, _TARGET_HIDDEN),
        generator=generator,
        dtype=torch.float64,
    )
    suffix_start = sequence - suffix
    if force_suffix_to_require_warm_state:
        # B is zero throughout the suffix while C remains nonzero.  The first
        # two suffix contexts can only come from B*x values in the warm prefix.
        target_input[:, suffix_start:, 0] = 0.0
        target_input[:, suffix_start:, 1] = 1.0
    u_value = target_input[..., 0]
    v_value = target_input[..., 1]
    b_single = u_value
    c_single = 0.4 * u_value + 0.6 * v_value
    x_single = -0.2 * u_value + 0.8 * v_value
    parent_b = b_single.unsqueeze(-1).expand(-1, -1, _SOURCE_HIDDEN).clone()
    parent_c = c_single.unsqueeze(-1).expand(-1, -1, _SOURCE_HIDDEN).clone()
    parent_x = x_single.unsqueeze(-1).expand(-1, -1, _SOURCE_HIDDEN).clone()
    fir_weight, fir_bias = _source_fir()
    source_context = _source_context(
        parent_b,
        parent_c,
        parent_x,
        fir_weight,
        fir_bias,
    )
    # All source channels are behaviorally identical, so every legal
    # all-source barycenter has these same two target context coordinates.
    target_context = source_context[..., :1].expand(
        -1,
        -1,
        _TARGET_HIDDEN,
    )
    writer = torch.tensor(
        [[1.2, -0.2], [-0.3, 0.8]],
        dtype=torch.float64,
    )
    actual_pre = torch.randn(
        (batch, suffix, _TARGET_HIDDEN),
        generator=generator,
        dtype=torch.float64,
    )
    endpoint = actual_pre + target_context[:, suffix_start:, :] @ writer.T
    if corrupt_endpoint:
        endpoint = endpoint + 20.0 * torch.randn(
            endpoint.shape,
            generator=generator,
            dtype=torch.float64,
        )
    valid_mask = torch.ones((batch, sequence), dtype=torch.bool)
    return Lfm2CpcfShortConvFoldCase(
        case_id=case_id,
        split=split,
        suffix_start=suffix_start,
        actual_child_operator_input_history=target_input,
        parent_b_history=parent_b,
        parent_c_history=parent_c,
        parent_x_history=parent_x,
        parent_post_operator_residual_target_space_suffix=endpoint,
        actual_child_pre_operator_residual_suffix=actual_pre,
        valid_prefix_mask=valid_mask,
    )


def _templates(*, seed: int, requires_grad: bool = False) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)

    def template(shape: tuple[int, ...]) -> Any:
        value = torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
        )
        return value.requires_grad_(requires_grad)

    return {
        "target_in_proj_weight_template": template(
            (3 * _TARGET_HIDDEN, _TARGET_HIDDEN)
        ),
        "target_in_proj_bias_template": template((3 * _TARGET_HIDDEN,)),
        "target_conv_weight_template": template((_TARGET_HIDDEN, 1, _FIR_LENGTH)),
        "target_conv_bias_template": template((_TARGET_HIDDEN,)),
        "target_out_proj_weight_template": template((_TARGET_HIDDEN, _TARGET_HIDDEN)),
        "target_out_proj_bias_template": template((_TARGET_HIDDEN,)),
    }


def _strict_policy(*, enforce: bool = True) -> Lfm2CpcfShortConvGatePolicy:
    return Lfm2CpcfShortConvGatePolicy(
        maximum_fit_endpoint_relative_rmse=2e-5,
        maximum_validation_endpoint_relative_rmse=2e-5,
        minimum_fit_endpoint_cosine_similarity=0.99999,
        minimum_validation_endpoint_cosine_similarity=0.99999,
        enforce=enforce,
    )


def _compile(
    cases: tuple[Lfm2CpcfShortConvFoldCase, ...],
    *,
    templates: dict[str, Any] | None = None,
    policy: Lfm2CpcfShortConvGatePolicy | None = None,
) -> Any:
    weight, bias = _source_fir()
    weight.requires_grad_(True)
    return compile_lfm2_cpcf_short_conv_fold(
        source_fir_weight=weight,
        source_fir_bias=bias,
        cases=cases,
        ridge=0.0,
        gate_policy=_strict_policy() if policy is None else policy,
        **(_templates(seed=9127) if templates is None else templates),
    )


def test_full_prefix_fir_warmup_suffix_writer_and_full_source_coverage() -> None:
    fit = _case(
        case_id="fit-prefix",
        split="fit",
        batch=2,
        sequence=9,
        suffix=4,
        seed=1001,
    )
    validation = _case(
        case_id="heldout-warm-prefix",
        split="validation",
        batch=1,
        sequence=11,
        suffix=5,
        seed=1002,
        force_suffix_to_require_warm_state=True,
    )
    input_snapshots = {
        "fit_target": fit.actual_child_operator_input_history.clone(),
        "fit_b": fit.parent_b_history.clone(),
        "fit_endpoint": (fit.parent_post_operator_residual_target_space_suffix.clone()),
        "validation_target": (validation.actual_child_operator_input_history.clone()),
    }
    templates = _templates(seed=1003, requires_grad=True)
    template_snapshots = {
        name: value.detach().clone() for name, value in templates.items()
    }

    result = _compile(
        (fit, validation),
        templates=templates,
    )

    assert result.report["schema"] == LFM2_CPCF_SHORT_CONV_FOLD_SCHEMA
    assert tuple(result.in_proj_weight.shape) == (6, 2)
    assert tuple(result.in_proj_bias.shape) == (6,)
    assert tuple(result.conv_weight.shape) == (2, 1, 3)
    assert tuple(result.conv_bias.shape) == (2,)
    assert tuple(result.out_proj_weight.shape) == (2, 2)
    assert tuple(result.out_proj_bias.shape) == (2,)
    assert result.full_source_coverage is True
    assert len(result.source_to_target) == _SOURCE_HIDDEN
    assert sorted(
        source for group in result.channel_groups for source in group
    ) == list(range(_SOURCE_HIDDEN))
    assert all(group for group in result.channel_groups)
    assert result.report["provenance"]["source_channels_dropped"] == 0
    assert result.report["provenance"]["source_channels_selected"] == 0
    assert result.report["provenance"]["source_channels_pruned"] == 0

    # Only the 2x4 suffix rows fit the writer, while all 2x9 rows fit the
    # producer and 2x5 earlier tokens warm its causal FIR.
    assert result.report["writer_fit_rows"] == 8
    assert result.report["fit_prefix_rows"] == 18
    assert result.report["fit_fir_warmup_rows"] == 10
    assert result.report["output_writer_scope"].endswith("suffix_only")
    heldout_report = result.report["validation_case_reports"][0]
    assert heldout_report["prefix_sequence_length"] == 11
    assert heldout_report["suffix_sequence_length"] == 5
    assert heldout_report["fir_warmup_tokens_per_sequence"] == 6
    assert result.behavior_gate_passed is True
    assert result.validation_metrics["residual_endpoint_suffix"]["relative_rmse"] < 2e-5

    audit = result.report["audit"]
    assert audit["source_inputs_unchanged"] is True
    assert audit["target_templates_unchanged"] is True
    assert audit["evidence_unchanged"] is True
    assert audit["case_authentication_verified"] is True
    assert audit["blank_target_values_used_by_compiler"] is False
    assert result.report["execution"]["parent_model_forward_calls"] == 0
    assert result.report["execution"]["target_model_forward_calls"] == 0
    assert result.report["execution"]["backward_calls"] == 0
    assert result.report["execution"]["optimizer_steps"] == 0
    assert result.report["execution"]["parameters_pruned"] == 0
    assert result.report["execution"]["parameters_quantized"] == 0
    assert all(not value.requires_grad for value in result.state_dict().values())

    assert torch.equal(
        fit.actual_child_operator_input_history,
        input_snapshots["fit_target"],
    )
    assert torch.equal(fit.parent_b_history, input_snapshots["fit_b"])
    assert torch.equal(
        fit.parent_post_operator_residual_target_space_suffix,
        input_snapshots["fit_endpoint"],
    )
    assert torch.equal(
        validation.actual_child_operator_input_history,
        input_snapshots["validation_target"],
    )
    for name, value in templates.items():
        assert torch.equal(value.detach(), template_snapshots[name])
        assert value.grad is None


def test_blank_template_values_are_ignored_and_validation_is_held_out() -> None:
    fit = _case(
        case_id="fit",
        split="fit",
        batch=2,
        sequence=8,
        suffix=3,
        seed=2001,
    )
    validation = _case(
        case_id="validation-clean",
        split="validation",
        batch=1,
        sequence=10,
        suffix=4,
        seed=2002,
    )
    first = _compile(
        (fit, validation),
        templates=_templates(seed=2003),
    )
    second = _compile(
        (fit, validation),
        templates=_templates(seed=9999),
    )
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])

    corrupted_validation = _case(
        case_id="validation-corrupt",
        split="validation",
        batch=1,
        sequence=10,
        suffix=4,
        seed=2002,
        corrupt_endpoint=True,
    )
    permissive = Lfm2CpcfShortConvGatePolicy(
        maximum_fit_endpoint_relative_rmse=2e-5,
        maximum_validation_endpoint_relative_rmse=2e-5,
        minimum_fit_endpoint_cosine_similarity=0.99999,
        minimum_validation_endpoint_cosine_similarity=0.99999,
        enforce=False,
    )
    heldout_failure = _compile(
        (fit, corrupted_validation),
        templates=_templates(seed=7777),
        policy=permissive,
    )
    assert heldout_failure.behavior_gate_passed is False
    assert (
        heldout_failure.report["behavior_gate"]["heldout_validation_used_for_fitting"]
        is False
    )
    for name, value in first.state_dict().items():
        assert torch.equal(value, heldout_failure.state_dict()[name])

    with pytest.raises(
        Lfm2CpcfShortConvValidationError,
        match="validation_endpoint",
    ) as captured:
        _compile(
            (fit, corrupted_validation),
            templates=_templates(seed=8888),
            policy=_strict_policy(enforce=True),
        )
    assert captured.value.report["behavior_gate"]["passed"] is False


def test_case_authentication_rejects_post_seal_tensor_mutation() -> None:
    fit = _case(
        case_id="mutable-fit",
        split="fit",
        batch=1,
        sequence=7,
        suffix=3,
        seed=3001,
    )
    validation = _case(
        case_id="validation",
        split="validation",
        batch=1,
        sequence=7,
        suffix=3,
        seed=3002,
    )
    assert fit.verify_authentication() is True
    fit.parent_b_history[0, 0, 0] += 1.0
    assert fit.verify_authentication() is False
    with pytest.raises(
        Lfm2CpcfShortConvFoldError,
        match="failed authentication",
    ):
        _compile((fit, validation))


def test_residual_evidence_must_be_a_terminal_suffix() -> None:
    case = _case(
        case_id="misaligned",
        split="fit",
        batch=1,
        sequence=7,
        suffix=3,
        seed=4001,
    )
    misaligned = Lfm2CpcfShortConvFoldCase(
        case_id="misaligned-copy",
        split="fit",
        suffix_start=2,
        actual_child_operator_input_history=(case.actual_child_operator_input_history),
        parent_b_history=case.parent_b_history,
        parent_c_history=case.parent_c_history,
        parent_x_history=case.parent_x_history,
        parent_post_operator_residual_target_space_suffix=(
            case.parent_post_operator_residual_target_space_suffix
        ),
        actual_child_pre_operator_residual_suffix=(
            case.actual_child_pre_operator_residual_suffix
        ),
        valid_prefix_mask=case.valid_prefix_mask,
    )
    validation = _case(
        case_id="validation",
        split="validation",
        batch=1,
        sequence=7,
        suffix=3,
        seed=4002,
    )
    with pytest.raises(
        Lfm2CpcfShortConvFoldError,
        match="terminal residual suffix",
    ):
        _compile((misaligned, validation))
