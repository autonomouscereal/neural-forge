from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_cpcf_short_conv_fold import (  # noqa: E402
    LFM2_CPCF_SHORT_CONV_FOLD_SCHEMA,
    Lfm2CpcfShortConvFoldCase,
    Lfm2CpcfShortConvFoldError,
    Lfm2CpcfShortConvGatePolicy,
    Lfm2CpcfShortConvValidationError,
    _ShortConvComputePolicy,
    _cluster_all_source_channels,
    _cuda_running_nearest_seeds,
    _running_nearest_work_counts,
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
    compute_device: str | None = None,
) -> Any:
    weight, bias = _source_fir()
    weight.requires_grad_(True)
    return compile_lfm2_cpcf_short_conv_fold(
        source_fir_weight=weight,
        source_fir_bias=bias,
        cases=cases,
        ridge=0.0,
        gate_policy=_strict_policy() if policy is None else policy,
        **({"compute_device": compute_device} if compute_device is not None else {}),
        **(_templates(seed=9127) if templates is None else templates),
    )


def _compute_backend_state() -> dict[str, bool]:
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
    }


def test_real_four_x_geometry_eliminates_quadratic_distance_recomputation() -> None:
    legacy, running = _running_nearest_work_counts(1024, 776)

    assert legacy == 152_454_900
    assert running == 793_600
    assert legacy / running == pytest.approx(192.10546875)


def test_single_target_cpu_clustering_covers_every_source_without_distance_work() -> (
    None
):
    generator = torch.Generator().manual_seed(20_260_721)
    source_hidden = 5
    histories = (
        {
            "parent_b": torch.randn(
                (1, 4, source_hidden), generator=generator, dtype=torch.float64
            ),
            "parent_c": torch.randn(
                (1, 4, source_hidden), generator=generator, dtype=torch.float64
            ),
            "parent_x": torch.randn(
                (1, 4, source_hidden), generator=generator, dtype=torch.float64
            ),
        },
    )
    contexts = (
        torch.randn((1, 4, source_hidden), generator=generator, dtype=torch.float64),
    )
    source_kernel = torch.randn(
        (source_hidden, 3), generator=generator, dtype=torch.float64
    )

    assignment, groups, weights, _importance, _features, execution = (
        _cluster_all_source_channels(
            histories=histories,
            parent_contexts=contexts,
            source_kernel=source_kernel,
            target_hidden=1,
        )
    )

    assert assignment == (0,) * source_hidden
    assert groups == (tuple(range(source_hidden)),)
    assert sum(weights[0]) == pytest.approx(1.0)
    assert execution is None
    assert _running_nearest_work_counts(source_hidden, 1) == (0, 0)


def test_single_target_cuda_receipt_reports_zero_distance_work_without_division(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(name="mock-cuda", major=8, minor=6),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device: (_ for _ in ()).throw(
            AssertionError("single-target CUDA folding has no distance work")
        ),
    )
    standardized = torch.tensor(
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
        dtype=torch.float64,
    )
    importance = torch.tensor([1.0, 7.0, 3.0], dtype=torch.float64)
    backend_before = _compute_backend_state()

    seeds, execution = _cuda_running_nearest_seeds(
        standardized=standardized,
        importance=importance,
        target_hidden=1,
        compute=_ShortConvComputePolicy(
            requested_device="cuda:0",
            backend="cuda",
            device_index=0,
        ),
    )

    body = dict(execution)
    seal = body.pop("execution_sha256")
    assert canonical_sha256(body) == seal
    assert seeds == [1]
    assert execution["distance_vector_update_count"] == 0
    assert execution["source_distance_row_evaluation_count"] == 0
    assert execution["legacy_recomputed_distance_evaluation_count"] == 0
    assert execution["legacy_to_running_distance_evaluation_ratio"] == 1.0
    assert execution["distance_evaluation_reduction_ratio_applicable"] is False
    assert execution["distance_work_status"] == (
        "not-applicable-single-target-first-seed-only"
    )
    assert execution["cuda_distance_compute_executed"] is False
    assert _compute_backend_state() == backend_before


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


def test_explicit_cpu_compute_policy_preserves_default_state_and_report_bytes() -> None:
    cases = (
        _case(
            case_id="fit-cpu-byte-compatibility",
            split="fit",
            batch=2,
            sequence=9,
            suffix=4,
            seed=1501,
        ),
        _case(
            case_id="validation-cpu-byte-compatibility",
            split="validation",
            batch=1,
            sequence=10,
            suffix=4,
            seed=1502,
        ),
    )

    implicit = _compile(cases)
    explicit = _compile(cases, compute_device="cpu")

    assert implicit.as_dict() == explicit.as_dict()
    assert "compute_execution" not in explicit.report["execution"]
    assert all(
        torch.equal(value, explicit.state_dict()[name])
        for name, value in implicit.state_dict().items()
    )


@pytest.mark.parametrize("compute_device", ("cuda", "cuda:-1", "cpu:0"))
def test_invalid_compute_device_fails_closed(compute_device: str) -> None:
    cases = (
        _case(
            case_id="fit-invalid-device",
            split="fit",
            batch=1,
            sequence=7,
            suffix=3,
            seed=1601,
        ),
        _case(
            case_id="validation-invalid-device",
            split="validation",
            batch=1,
            sequence=7,
            suffix=3,
            seed=1602,
        ),
    )
    with pytest.raises(Lfm2CpcfShortConvFoldError, match="explicitly indexed"):
        _compile(cases, compute_device=compute_device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_indexed_cuda_running_nearest_is_repeatedly_deterministic_and_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cases = (
        _case(
            case_id="fit-cuda-determinism",
            split="fit",
            batch=2,
            sequence=9,
            suffix=4,
            seed=1701,
        ),
        _case(
            case_id="validation-cuda-determinism",
            split="validation",
            batch=1,
            sequence=10,
            suffix=4,
            seed=1702,
        ),
    )
    backend_before = _compute_backend_state()

    first = _compile(cases, compute_device="cuda:0")
    second = _compile(cases, compute_device="cuda:0")

    assert first.as_dict() == second.as_dict()
    assert first.source_to_target == second.source_to_target
    assert first.channel_groups == second.channel_groups
    assert first.channel_barycentric_weights == second.channel_barycentric_weights
    assert all(
        torch.equal(value, second.state_dict()[name])
        for name, value in first.state_dict().items()
    )
    execution = first.report["execution"]["compute_execution"]
    execution_body = dict(execution)
    execution_sha256 = execution_body.pop("execution_sha256")
    assert canonical_sha256(execution_body) == execution_sha256
    assert execution["backend"] == "cuda"
    assert execution["requested_device"] == "cuda:0"
    assert execution["compute_dtype"] == "float64"
    assert execution["distance_vector_update_count"] == _TARGET_HIDDEN - 1
    assert execution["source_distance_row_evaluation_count"] == (
        _SOURCE_HIDDEN * (_TARGET_HIDDEN - 1)
    )
    assert execution["training_steps"] == 0
    assert _compute_backend_state() == backend_before


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
