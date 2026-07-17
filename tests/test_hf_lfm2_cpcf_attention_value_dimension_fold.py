from __future__ import annotations

from dataclasses import fields

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_cpcf_attention_value_dimension_fold as MODULE  # noqa: E402
from neural_forge.hf_lfm2_cpcf_attention_dimension_fold import (  # noqa: E402
    Lfm2CpcfAttentionDimensionTopology,
    Lfm2CpcfBarycentricHeadGroup,
)
from neural_forge.hf_lfm2_cpcf_attention_value_dimension_fold import (  # noqa: E402
    Lfm2CpcfAttentionValueDimensionCase,
    Lfm2CpcfAttentionValueDimensionInstallError,
    Lfm2CpcfAttentionValueDimensionPolicy,
    Lfm2CpcfAttentionValueDimensionValidationError,
    compile_lfm2_cpcf_attention_value_dimension_fold,
    install_lfm2_cpcf_attention_value_dimension_fold,
)


TOPOLOGY = Lfm2CpcfAttentionDimensionTopology(
    source_query_heads=16,
    source_key_value_heads=8,
    target_query_heads=2,
    target_key_value_heads=1,
    source_head_dim=64,
    target_head_dim=32,
)
QUERY_GROUPS = (
    Lfm2CpcfBarycentricHeadGroup(
        tuple(range(8)),
        (1.0 / 8.0,) * 8,
    ),
    Lfm2CpcfBarycentricHeadGroup(
        tuple(range(8, 16)),
        (1.0 / 8.0,) * 8,
    ),
)
KEY_VALUE_GROUPS = (
    Lfm2CpcfBarycentricHeadGroup(
        tuple(range(8)),
        (1.0 / 8.0,) * 8,
    ),
)
FIXED_QK_SHA256 = "a" * 64


def _orthonormal_embedding(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(
        64,
        32,
        generator=generator,
        dtype=torch.float64,
    )
    return torch.linalg.qr(matrix, mode="reduced").Q


def _causal_probabilities(
    *,
    seed: int,
    batch: int,
    query_length: int,
    key_length: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(
        batch,
        2,
        query_length,
        key_length,
        generator=generator,
        dtype=torch.float64,
    )
    query_positions = torch.arange(query_length).reshape(
        1,
        1,
        query_length,
        1,
    )
    key_positions = torch.arange(key_length).reshape(
        1,
        1,
        1,
        key_length,
    )
    mask = key_positions <= (key_length - query_length + query_positions)
    return torch.softmax(
        logits.masked_fill(~mask, -torch.inf),
        dim=-1,
        dtype=torch.float64,
    )


def _parent_probabilities(target: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            target[:, 0:1].expand(-1, 8, -1, -1),
            target[:, 1:2].expand(-1, 8, -1, -1),
        ),
        dim=1,
    ).contiguous()


def _case(
    *,
    case_id: str,
    split: str,
    seed: int,
    batch: int,
    true_value_weight: torch.Tensor,
    true_output_weight: torch.Tensor,
    embedding: torch.Tensor,
) -> Lfm2CpcfAttentionValueDimensionCase:
    generator = torch.Generator().manual_seed(seed)
    query_length = 3
    key_length = 6
    child_input = torch.randn(
        batch,
        key_length,
        64,
        generator=generator,
        dtype=torch.float64,
    )
    compact_values = (
        child_input.reshape(-1, 64) @ true_value_weight.T
    ).reshape(batch, key_length, 32)
    source_values = compact_values @ embedding.T
    parent_values = source_values.unsqueeze(1).expand(
        -1,
        8,
        -1,
        -1,
    ).clone()
    target_probabilities = _causal_probabilities(
        seed=seed + 1,
        batch=batch,
        query_length=query_length,
        key_length=key_length,
    )
    parent_probabilities = _parent_probabilities(target_probabilities)
    repeated_values = parent_values.repeat_interleave(2, dim=1)
    parent_context = torch.matmul(
        parent_probabilities,
        repeated_values,
    )
    target_context = torch.matmul(
        target_probabilities,
        compact_values.unsqueeze(1).expand(-1, 2, -1, -1),
    )
    target_context_rows = target_context.permute(0, 2, 1, 3).reshape(
        batch,
        query_length,
        64,
    )
    child_pre = torch.randn(
        batch,
        query_length,
        64,
        generator=generator,
        dtype=torch.float64,
    )
    endpoint = child_pre + target_context_rows @ true_output_weight.T
    return Lfm2CpcfAttentionValueDimensionCase(
        case_id=case_id,
        split=split,
        fixed_query_key_sha256=FIXED_QK_SHA256,
        actual_child_key_value_residual_history=child_input,
        target_query_key_probabilities=target_probabilities,
        parent_value_history=parent_values,
        parent_query_key_probabilities=parent_probabilities,
        parent_context=parent_context,
        actual_child_pre_operator_residual=child_pre,
        parent_post_operator_residual_target_space=endpoint,
        valid_key_mask=torch.ones(batch, key_length, dtype=torch.bool),
        valid_query_mask=torch.ones(batch, query_length, dtype=torch.bool),
    )


def _weights() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value_generator = torch.Generator().manual_seed(9101)
    output_generator = torch.Generator().manual_seed(9102)
    value = (
        torch.randn(
            32,
            64,
            generator=value_generator,
            dtype=torch.float64,
        )
        / 8.0
    )
    output = (
        torch.randn(
            64,
            64,
            generator=output_generator,
            dtype=torch.float64,
        )
        / 8.0
    )
    return value, output, _orthonormal_embedding(9103)


def _strict_policy(
    *,
    enforce: bool = True,
) -> Lfm2CpcfAttentionValueDimensionPolicy:
    return Lfm2CpcfAttentionValueDimensionPolicy(
        minimum_fit_value_basis_energy_fraction=0.999999,
        maximum_fit_value_relative_rmse=2e-6,
        maximum_validation_value_relative_rmse=2e-6,
        maximum_fit_context_relative_rmse=2e-6,
        maximum_validation_context_relative_rmse=2e-6,
        minimum_validation_context_cosine_similarity=0.999999,
        maximum_fit_endpoint_relative_rmse=2e-6,
        maximum_validation_endpoint_relative_rmse=2e-6,
        minimum_validation_endpoint_cosine_similarity=0.999999,
        enforce=enforce,
    )


def _compile(
    fit: Lfm2CpcfAttentionValueDimensionCase,
    validation: Lfm2CpcfAttentionValueDimensionCase,
    *,
    policy: Lfm2CpcfAttentionValueDimensionPolicy | None = None,
):
    value_template = torch.randn(32, 64, dtype=torch.float64)
    output_template = torch.randn(64, 64, dtype=torch.float64)
    return compile_lfm2_cpcf_attention_value_dimension_fold(
        topology=TOPOLOGY,
        cases=(fit, validation),
        query_groups=QUERY_GROUPS,
        key_value_groups=KEY_VALUE_GROUPS,
        target_value_weight_template=value_template,
        target_output_weight_template=output_template,
        policy=policy or _strict_policy(),
        ridge=1e-10,
    )


@pytest.fixture(scope="module")
def exact_compilation():
    value, output, embedding = _weights()
    fit = _case(
        case_id="fit-real-shape",
        split="fit",
        seed=9110,
        batch=96,
        true_value_weight=value,
        true_output_weight=output,
        embedding=embedding,
    )
    validation = _case(
        case_id="validation-real-shape",
        split="validation",
        seed=9120,
        batch=48,
        true_value_weight=value,
        true_output_weight=output,
        embedding=embedding,
    )
    return _compile(fit, validation), fit, validation


def test_real_lfm2_250x_shape_d64_to_d32_value_output_fold(
    exact_compilation,
) -> None:
    compiled, _fit, validation = exact_compilation

    assert tuple(compiled.value_weight.shape) == (32, 64)
    assert tuple(compiled.output_weight.shape) == (64, 64)
    assert len(compiled.value_feature_bases) == 1
    assert tuple(compiled.value_feature_bases[0].shape) == (64, 32)
    assert compiled.validation_metrics["value"]["relative_rmse"] < 2e-6
    assert compiled.validation_metrics["context"]["relative_rmse"] < 2e-6
    assert (
        compiled.validation_metrics["residual_endpoint"]["relative_rmse"]
        < 2e-6
    )
    assert compiled.report["accepted"] is True
    assert (
        compiled.report["provenance"]["source_query_head_assignment_count"]
        == 16
    )
    assert (
        compiled.report["provenance"][
            "source_key_value_head_assignment_count"
        ]
        == 8
    )
    assert compiled.report["provenance"]["source_key_value_heads_dropped"] == 0
    assert (
        compiled.report["behavior_gate"][
            "heldout_validation_used_for_fitting"
        ]
        is False
    )
    assert compiled.report["execution"]["training_steps"] == 0
    assert compiled.report["execution"]["parameters_quantized"] == 0
    replay = compiled.replay(validation)
    assert tuple(replay.target_values.shape) == (48, 1, 6, 32)
    assert tuple(replay.actual_target_contexts.shape) == (48, 2, 3, 32)
    assert tuple(replay.post_operator_residual.shape) == (48, 3, 64)


def test_heldout_endpoint_gate_rejects_a_different_value_write() -> None:
    value, output, embedding = _weights()
    fit = _case(
        case_id="fit-heldout-rejection",
        split="fit",
        seed=9210,
        batch=96,
        true_value_weight=value,
        true_output_weight=output,
        embedding=embedding,
    )
    validation = _case(
        case_id="validation-heldout-rejection",
        split="validation",
        seed=9220,
        batch=48,
        true_value_weight=value,
        true_output_weight=-output,
        embedding=embedding,
    )

    with pytest.raises(
        Lfm2CpcfAttentionValueDimensionValidationError,
        match="behavior gates",
    ) as captured:
        _compile(fit, validation)

    report = captured.value.report
    assert report["accepted"] is False
    assert (
        report["behavior_gate"]["checks"][
            "maximum_validation_endpoint_relative_rmse"
        ]
        is False
        or report["behavior_gate"]["checks"][
            "minimum_validation_endpoint_cosine_similarity"
        ]
        is False
    )
    assert (
        report["behavior_gate"]["heldout_validation_used_for_fitting"]
        is False
    )


def test_validation_never_changes_fit_bases_or_compiled_state() -> None:
    value, output, embedding = _weights()
    fit = _case(
        case_id="fit-only-state",
        split="fit",
        seed=9310,
        batch=96,
        true_value_weight=value,
        true_output_weight=output,
        embedding=embedding,
    )
    validation_a = _case(
        case_id="validation-a",
        split="validation",
        seed=9320,
        batch=40,
        true_value_weight=value,
        true_output_weight=output,
        embedding=embedding,
    )
    validation_b = _case(
        case_id="validation-b",
        split="validation",
        seed=9330,
        batch=40,
        true_value_weight=-value,
        true_output_weight=-output,
        embedding=_orthonormal_embedding(9331),
    )
    first = _compile(
        fit,
        validation_a,
        policy=_strict_policy(enforce=False),
    )
    second = _compile(
        fit,
        validation_b,
        policy=_strict_policy(enforce=False),
    )

    assert torch.equal(first.value_weight, second.value_weight)
    assert torch.equal(first.output_weight, second.output_weight)
    assert torch.equal(
        first.value_feature_bases[0],
        second.value_feature_bases[0],
    )
    assert (
        first.report["compiled_state_sha256"]
        == second.report["compiled_state_sha256"]
    )
    assert (
        first.report["provenance"]["fit_only_value_bases"][0][
            "basis_sha256"
        ]
        == second.report["provenance"]["fit_only_value_bases"][0][
            "basis_sha256"
        ]
    )
    assert (
        first.report["validation_metrics"]["value"]["relative_rmse"]
        != second.report["validation_metrics"]["value"]["relative_rmse"]
    )


def test_compile_preserves_every_evidence_and_template_tensor() -> None:
    value, output, embedding = _weights()
    fit = _case(
        case_id="fit-preservation",
        split="fit",
        seed=9410,
        batch=72,
        true_value_weight=value,
        true_output_weight=output,
        embedding=embedding,
    )
    validation = _case(
        case_id="validation-preservation",
        split="validation",
        seed=9420,
        batch=36,
        true_value_weight=value,
        true_output_weight=output,
        embedding=embedding,
    )
    snapshots = {
        (case.split, field.name): getattr(case, field.name).clone()
        for case in (fit, validation)
        for field in fields(case)
        if isinstance(getattr(case, field.name), torch.Tensor)
    }
    value_template = torch.randn(32, 64, dtype=torch.float64)
    output_template = torch.randn(64, 64, dtype=torch.float64)
    value_snapshot = value_template.clone()
    output_snapshot = output_template.clone()

    compiled = compile_lfm2_cpcf_attention_value_dimension_fold(
        topology=TOPOLOGY,
        cases=(fit, validation),
        query_groups=QUERY_GROUPS,
        key_value_groups=KEY_VALUE_GROUPS,
        target_value_weight_template=value_template,
        target_output_weight_template=output_template,
        policy=_strict_policy(),
        ridge=1e-10,
    )

    for case in (fit, validation):
        for field in fields(case):
            observed = getattr(case, field.name)
            if isinstance(observed, torch.Tensor):
                assert torch.equal(
                    observed,
                    snapshots[(case.split, field.name)],
                )
    assert torch.equal(value_template, value_snapshot)
    assert torch.equal(output_template, output_snapshot)
    assert compiled.report["audit"]["evidence_unchanged"] is True
    assert compiled.report["audit"]["target_templates_unchanged"] is True
    detached = compiled.state_dict()
    detached["v_proj.weight"].zero_()
    assert not torch.equal(detached["v_proj.weight"], compiled.value_weight)


class _TargetAttention(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.v_proj = torch.nn.Linear(
            64,
            32,
            bias=False,
            dtype=dtype,
        )
        self.out_proj = torch.nn.Linear(
            64,
            64,
            bias=False,
            dtype=dtype,
        )


def test_compiled_state_and_installer_preserve_blank_target_bfloat16_dtype() -> None:
    value, output, embedding = _weights()
    fit = _case(
        case_id="fit-bfloat16-template",
        split="fit",
        seed=9510,
        batch=96,
        true_value_weight=value,
        true_output_weight=output,
        embedding=embedding,
    )
    validation = _case(
        case_id="validation-bfloat16-template",
        split="validation",
        seed=9520,
        batch=40,
        true_value_weight=value,
        true_output_weight=output,
        embedding=embedding,
    )
    compiled = compile_lfm2_cpcf_attention_value_dimension_fold(
        topology=TOPOLOGY,
        cases=(fit, validation),
        query_groups=QUERY_GROUPS,
        key_value_groups=KEY_VALUE_GROUPS,
        target_value_weight_template=torch.empty(
            32,
            64,
            dtype=torch.bfloat16,
        ),
        target_output_weight_template=torch.empty(
            64,
            64,
            dtype=torch.bfloat16,
        ),
        policy=_strict_policy(enforce=False),
        ridge=1e-10,
    )

    assert compiled.value_weight.dtype == torch.bfloat16
    assert compiled.output_weight.dtype == torch.bfloat16
    assert compiled.state_dict()["v_proj.weight"].dtype == torch.bfloat16
    target = _TargetAttention(dtype=torch.bfloat16)
    transaction = install_lfm2_cpcf_attention_value_dimension_fold(
        target,
        compiled,
    )
    assert transaction.install_report.accepted
    assert target.v_proj.weight.dtype == torch.bfloat16
    assert target.out_proj.weight.dtype == torch.bfloat16
    assert transaction.rollback().accepted


def test_atomic_installer_preserves_parameter_and_gradient_identity_and_rolls_back(
    exact_compilation,
) -> None:
    compiled, _fit, _validation = exact_compilation
    target = _TargetAttention()
    target.v_proj.weight.grad = torch.randn_like(target.v_proj.weight)
    target.out_proj.weight.grad = torch.randn_like(target.out_proj.weight)
    identities = (
        id(target.v_proj.weight),
        id(target.out_proj.weight),
    )
    gradient_identities = (
        id(target.v_proj.weight.grad),
        id(target.out_proj.weight.grad),
    )
    original_value = target.v_proj.weight.detach().clone()
    original_output = target.out_proj.weight.detach().clone()
    original_value_gradient = target.v_proj.weight.grad.clone()
    original_output_gradient = target.out_proj.weight.grad.clone()

    transaction = install_lfm2_cpcf_attention_value_dimension_fold(
        target,
        compiled,
    )

    assert transaction.install_report.accepted
    assert torch.equal(target.v_proj.weight, compiled.value_weight)
    assert torch.equal(target.out_proj.weight, compiled.output_weight)
    report = transaction.rollback()
    assert report.accepted
    assert report.phase == "rolled-back"
    assert identities == (
        id(target.v_proj.weight),
        id(target.out_proj.weight),
    )
    assert gradient_identities == (
        id(target.v_proj.weight.grad),
        id(target.out_proj.weight.grad),
    )
    assert torch.equal(target.v_proj.weight, original_value)
    assert torch.equal(target.out_proj.weight, original_output)
    assert torch.equal(target.v_proj.weight.grad, original_value_gradient)
    assert torch.equal(target.out_proj.weight.grad, original_output_gradient)

    committed = install_lfm2_cpcf_attention_value_dimension_fold(
        target,
        compiled,
    )
    commit_report = committed.commit()
    assert commit_report.accepted
    assert commit_report.phase == "committed"
    with pytest.raises(
        Lfm2CpcfAttentionValueDimensionInstallError,
        match="committed",
    ):
        committed.rollback()


def test_install_copy_failure_restores_both_parameters_atomically(
    exact_compilation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled, _fit, _validation = exact_compilation
    target = _TargetAttention()
    original_value = target.v_proj.weight.detach().clone()
    original_output = target.out_proj.weight.detach().clone()
    original_copy = MODULE._copy_parameter
    calls = 0

    def fail_second_copy(parameter, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic second-parameter copy failure")
        original_copy(parameter, value)

    monkeypatch.setattr(MODULE, "_copy_parameter", fail_second_copy)

    with pytest.raises(
        Lfm2CpcfAttentionValueDimensionInstallError,
        match="rolled back exactly",
    ):
        install_lfm2_cpcf_attention_value_dimension_fold(target, compiled)

    assert torch.equal(target.v_proj.weight, original_value)
    assert torch.equal(target.out_proj.weight, original_output)
