from __future__ import annotations

import ast
import inspect
from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

import neural_forge.hf_lfm2_structural_residual_attention_fold as module  # noqa: E402
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    Lfm2CachedAttentionFoldCase,
    Lfm2CachedShortConvFoldCase,
    Lfm2SparseResidualAssignment,
    Lfm2StructuralResidualAttentionInstallError,
    compile_lfm2_structural_attention_consumer_fold,
    compile_lfm2_structural_attention_producer_fold,
    compile_lfm2_structural_attention_value_output_fold,
    compile_lfm2_structural_short_conv_consumer_fold,
    compile_lfm2_structural_short_conv_producer_fold,
    install_lfm2_structural_attention_consumer_fold,
    install_lfm2_structural_attention_producer_fold,
    install_lfm2_structural_attention_score_producer_fold,
    install_lfm2_structural_attention_value_output_fold,
    install_lfm2_structural_short_conv_factor_consumer_fold,
    install_lfm2_structural_short_conv_consumer_fold,
    install_lfm2_structural_short_conv_producer_fold,
    rollback_lfm2_structural_split_install,
)


class _ToyAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden: int,
        query_heads: int,
        key_value_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(
            hidden,
            query_heads * head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.k_proj = nn.Linear(
            hidden,
            key_value_heads * head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.v_proj = nn.Linear(
            hidden,
            key_value_heads * head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            query_heads * head_dim,
            hidden,
            bias=False,
            dtype=torch.float64,
        )


class _ToyShortConv(nn.Module):
    def __init__(
        self,
        *,
        hidden: int,
        fir_length: int,
        bias: bool,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(
            hidden,
            3 * hidden,
            bias=bias,
            dtype=torch.float64,
        )
        self.conv = nn.Conv1d(
            hidden,
            hidden,
            fir_length,
            groups=hidden,
            bias=bias,
            padding=fir_length - 1,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            hidden,
            hidden,
            bias=bias,
            dtype=torch.float64,
        )


def _state(module_: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module_.named_parameters()
    }


def _identities(module_: nn.Module) -> dict[str, int]:
    return {
        name: id(parameter)
        for name, parameter in module_.named_parameters()
    }


def _gradient_identities(module_: nn.Module) -> dict[str, int | None]:
    return {
        name: None if parameter.grad is None else id(parameter.grad)
        for name, parameter in module_.named_parameters()
    }


def _gradients(module_: nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: (
            None
            if parameter.grad is None
            else parameter.grad.detach().clone()
        )
        for name, parameter in module_.named_parameters()
    }


def _seed_gradients(module_: nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    for parameter in module_.parameters():
        parameter.grad = torch.randn(
            parameter.shape,
            generator=generator,
            dtype=parameter.dtype,
            device=parameter.device,
        )


def _assert_exact_snapshot(
    target: nn.Module,
    *,
    state: dict[str, torch.Tensor],
    identities: dict[str, int],
    gradient_identities: dict[str, int | None],
    gradients: dict[str, torch.Tensor | None],
) -> None:
    assert _identities(target) == identities
    assert _gradient_identities(target) == gradient_identities
    for name, parameter in target.named_parameters():
        assert torch.equal(parameter, state[name])
        expected_gradient = gradients[name]
        if expected_gradient is None:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, expected_gradient)


def _attention_oracle() -> tuple[
    dict[str, torch.Tensor],
    Lfm2CachedAttentionFoldCase,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(101)
    source = {
        "query": torch.randn(
            8, 8, generator=generator, dtype=torch.float64
        ),
        "key": torch.randn(
            4, 8, generator=generator, dtype=torch.float64
        ),
        "value": torch.randn(
            4, 8, generator=generator, dtype=torch.float64
        ),
        "output": torch.randn(
            8, 8, generator=generator, dtype=torch.float64
        ),
    }
    target_input = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    target_q = torch.eye(4, dtype=torch.float64)
    target_k = torch.tensor(
        [[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]],
        dtype=torch.float64,
    )
    target_v = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    target_o = torch.eye(4, dtype=torch.float64)
    q_target = (target_input @ target_q.T).reshape(1, 4, 2, 2)
    q_target = q_target.permute(0, 2, 1, 3)
    k_target = (target_input @ target_k.T).reshape(1, 4, 1, 2)
    k_target = k_target.permute(0, 2, 1, 3)
    v_target = (target_input @ target_v.T).reshape(1, 4, 1, 2)
    v_target = v_target.permute(0, 2, 1, 3)
    parent_q = torch.cat([q_target[:, 0:1], q_target[:, 1:2]] * 2, dim=1)
    parent_k = k_target.repeat(1, 2, 1, 1)
    parent_v = v_target.repeat(1, 2, 1, 1)
    parent_probability = torch.softmax(
        parent_q
        @ parent_k.repeat_interleave(2, dim=1).transpose(-2, -1)
        / (2.0**0.5),
        dim=-1,
    )
    parent_context = (
        parent_probability
        @ parent_v.repeat_interleave(2, dim=1)
    )
    target_probability = torch.softmax(
        q_target
        @ k_target.repeat_interleave(2, dim=1).transpose(-2, -1)
        / (2.0**0.5),
        dim=-1,
    )
    actual_target_context = (
        target_probability @ v_target.repeat_interleave(2, dim=1)
    )
    flattened = actual_target_context.permute(0, 2, 1, 3).reshape(
        1, 4, 4
    )
    residual = torch.tensor(
        [[[0.5, -0.25, 0.75, -1.0]]], dtype=torch.float64
    ).repeat(1, 4, 1)
    endpoint = residual + flattened @ target_o.T
    return (
        source,
        Lfm2CachedAttentionFoldCase(
            target_query_input=target_input,
            target_key_value_input=target_input,
            parent_query_linear=parent_q,
            parent_key_linear=parent_k,
            parent_value=parent_v,
            parent_query=parent_q,
            parent_key=parent_k,
            parent_context=parent_context,
            parent_post_attention_residual_target_space=endpoint,
            target_pre_attention_residual=residual,
        ),
        actual_target_context,
    )


def _compile_attention_producer(
    target: _ToyAttention,
):
    source, case, actual_context = _attention_oracle()
    producer = compile_lfm2_structural_attention_producer_fold(
        source_query_weight=source["query"],
        source_key_weight=source["key"],
        source_value_weight=source["value"],
        source_output_weight=source["output"],
        target_query_weight_template=target.q_proj.weight,
        target_key_weight_template=target.k_proj.weight,
        target_value_weight_template=target.v_proj.weight,
        target_output_weight_template=target.out_proj.weight,
        cases=(case,),
        head_dim=2,
        residual_assignment_sha256="a" * 64,
        ridge=0.0,
        cluster_ridge=0.0,
    )
    return producer, case, actual_context


def _live_attention_context(
    target: _ToyAttention,
    case: Lfm2CachedAttentionFoldCase,
) -> torch.Tensor:
    batch, query_length, _hidden = case.target_query_input.shape
    key_length = case.target_key_value_input.shape[1]
    query_heads = target.q_proj.out_features // 2
    key_value_heads = target.k_proj.out_features // 2
    q = (
        case.target_query_input @ target.q_proj.weight.detach().T
    ).reshape(batch, query_length, query_heads, 2).permute(0, 2, 1, 3)
    k = (
        case.target_key_value_input @ target.k_proj.weight.detach().T
    ).reshape(
        batch,
        key_length,
        key_value_heads,
        2,
    ).permute(0, 2, 1, 3)
    value = (
        case.target_key_value_input @ target.v_proj.weight.detach().T
    ).reshape(
        batch,
        key_length,
        key_value_heads,
        2,
    ).permute(0, 2, 1, 3)
    repeats = query_heads // key_value_heads
    probability = torch.softmax(
        q @ k.repeat_interleave(repeats, dim=1).transpose(-2, -1)
        / (2.0**0.5),
        dim=-1,
    )
    return probability @ value.repeat_interleave(repeats, dim=1)


def test_attention_split_uses_actual_context_and_preserves_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(201)
    target = _ToyAttention(
        hidden=4, query_heads=2, key_value_heads=1, head_dim=2
    )
    _seed_gradients(target, 202)
    identities = _identities(target)
    gradient_identities = _gradient_identities(target)
    roles: list[str] = []
    original_fit = module._fit_linear

    def tracked_fit(*args, **kwargs):
        roles.append(kwargs["role"])
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(module, "_fit_linear", tracked_fit)
    producer, case, actual_context = _compile_attention_producer(target)
    assert "output" not in roles
    assert producer.full_source_coverage is True
    assert producer.as_dict()["consumer_compiled"] is False

    transaction = install_lfm2_structural_attention_producer_fold(
        target, producer
    )
    assert transaction.active is True
    assert transaction.producer_report.accepted is True
    assert _identities(target) == identities
    assert _gradient_identities(target) == gradient_identities

    consumer = compile_lfm2_structural_attention_consumer_fold(
        target_output_weight_template=target.out_proj.weight,
        actual_child_contexts=(actual_context,),
        cases=(case,),
        output_weight_prior=target.out_proj.weight.detach().clone(),
        ridge=0.0,
        max_prior_delta_ratio=None,
    )
    assert roles[-1] == "attention_output_actual_child_context"
    assert consumer.residual_endpoint_relative_rmse < 1e-10
    report = install_lfm2_structural_attention_consumer_fold(
        target, consumer, transaction
    )

    assert report.accepted is True
    assert transaction.committed is True
    assert transaction.active is False
    assert _identities(target) == identities
    assert _gradient_identities(target) == gradient_identities
    assert consumer.as_dict()["context_source"] == (
        "caller-supplied-actual-child-context"
    )


def test_coupled_attention_split_owns_qk_then_vo_and_never_worsens_fit() -> None:
    torch.manual_seed(251)
    target = _ToyAttention(
        hidden=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
    )
    _seed_gradients(target, 252)
    identities = _identities(target)
    gradient_identities = _gradient_identities(target)
    producer, case, _unused_context = _compile_attention_producer(target)
    value_before = target.v_proj.weight.detach().clone()
    output_before = target.out_proj.weight.detach().clone()

    transaction = install_lfm2_structural_attention_score_producer_fold(
        target,
        producer,
    )
    assert transaction.as_dict()["producer_parameter_names"] == [
        "q_proj.weight",
        "k_proj.weight",
    ]
    assert transaction.as_dict()["consumer_parameter_names"] == [
        "v_proj.weight",
        "out_proj.weight",
    ]
    assert torch.equal(target.v_proj.weight, value_before)
    assert torch.equal(target.out_proj.weight, output_before)
    fixed_query = target.q_proj.weight.detach().clone()
    fixed_key = target.k_proj.weight.detach().clone()
    actual_context = _live_attention_context(target, case)

    consumer = compile_lfm2_structural_attention_value_output_fold(
        producer=producer,
        target_value_weight_template=target.v_proj.weight,
        target_output_weight_template=target.out_proj.weight,
        actual_child_contexts=(actual_context,),
        cases=(case,),
        live_value_weight=target.v_proj.weight.detach(),
        value_weight_prior=producer.value_weight,
        output_weight_prior=target.out_proj.weight.detach(),
        ridge=0.0,
        context_ridge=1e-8,
        max_prior_delta_ratio=None,
        iterations=3,
        fit_tolerance=1e-12,
        context_equivalence_tolerance=1e-12,
    )
    assert consumer.context_reconstruction_relative_rmse < 1e-12
    assert consumer.value_context_solver == (
        "complete-fixed-score-weighted-input"
    )
    assert consumer.fit_nonincreasing
    assert all(
        current <= previous + consumer.fit_tolerance
        for previous, current in zip(
            consumer.fit_relative_rmse_history,
            consumer.fit_relative_rmse_history[1:],
        )
    )
    report = install_lfm2_structural_attention_value_output_fold(
        target,
        consumer,
        transaction,
    )

    assert report.accepted
    assert transaction.committed
    assert torch.equal(target.q_proj.weight, fixed_query)
    assert torch.equal(target.k_proj.weight, fixed_key)
    assert torch.equal(target.v_proj.weight, consumer.value_weight)
    assert torch.equal(target.out_proj.weight, consumer.output_weight)
    assert _identities(target) == identities
    assert _gradient_identities(target) == gradient_identities


def test_coupled_attention_consumer_failure_rolls_back_exact_snapshot() -> None:
    torch.manual_seed(271)
    target = _ToyAttention(
        hidden=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
    )
    _seed_gradients(target, 272)
    original_state = _state(target)
    original_ids = _identities(target)
    original_gradient_ids = _gradient_identities(target)
    original_gradients = _gradients(target)
    producer, case, _unused_context = _compile_attention_producer(target)
    transaction = install_lfm2_structural_attention_score_producer_fold(
        target,
        producer,
    )
    actual_context = _live_attention_context(target, case)
    consumer = compile_lfm2_structural_attention_value_output_fold(
        producer=producer,
        target_value_weight_template=target.v_proj.weight,
        target_output_weight_template=target.out_proj.weight,
        actual_child_contexts=(actual_context,),
        cases=(case,),
        live_value_weight=target.v_proj.weight.detach(),
        value_weight_prior=producer.value_weight,
        output_weight_prior=target.out_proj.weight.detach(),
        ridge=0.0,
        context_ridge=1e-8,
        max_prior_delta_ratio=None,
        iterations=2,
        context_equivalence_tolerance=1e-12,
    )
    invalid = replace(
        consumer,
        value_weight=torch.zeros(3, 4, dtype=torch.float64),
    )

    with pytest.raises(
        Lfm2StructuralResidualAttentionInstallError,
        match="does not match target",
    ):
        install_lfm2_structural_attention_value_output_fold(
            target,
            invalid,
            transaction,
        )

    assert transaction.rolled_back
    _assert_exact_snapshot(
        target,
        state=original_state,
        identities=original_ids,
        gradient_identities=original_gradient_ids,
        gradients=original_gradients,
    )


def test_attention_consumer_failure_rolls_back_exact_snapshot() -> None:
    torch.manual_seed(301)
    target = _ToyAttention(
        hidden=4, query_heads=2, key_value_heads=1, head_dim=2
    )
    _seed_gradients(target, 302)
    original_state = _state(target)
    original_ids = _identities(target)
    original_gradient_ids = _gradient_identities(target)
    original_gradients = _gradients(target)
    producer, case, actual_context = _compile_attention_producer(target)
    consumer = compile_lfm2_structural_attention_consumer_fold(
        target_output_weight_template=target.out_proj.weight,
        actual_child_contexts=(actual_context,),
        cases=(case,),
        ridge=0.0,
        max_prior_delta_ratio=None,
    )
    transaction = install_lfm2_structural_attention_producer_fold(
        target, producer
    )
    invalid = replace(
        consumer,
        output_weight=torch.zeros(5, 4, dtype=torch.float64),
    )

    with pytest.raises(
        Lfm2StructuralResidualAttentionInstallError,
        match="does not match target",
    ):
        install_lfm2_structural_attention_consumer_fold(
            target, invalid, transaction
        )

    assert transaction.rolled_back is True
    _assert_exact_snapshot(
        target,
        state=original_state,
        identities=original_ids,
        gradient_identities=original_gradient_ids,
        gradients=original_gradients,
    )


def _short_conv_case() -> tuple[
    Lfm2CachedShortConvFoldCase,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(401)
    batch, sequence, source_hidden, target_hidden = 1, 7, 4, 2
    target_input = torch.randn(
        batch,
        sequence,
        target_hidden,
        generator=generator,
        dtype=torch.float64,
    )
    actual_context = torch.randn(
        batch,
        sequence,
        target_hidden,
        generator=generator,
        dtype=torch.float64,
    )
    output_weight = torch.tensor(
        [[1.25, -0.5], [0.75, 1.5]], dtype=torch.float64
    )
    output_bias = torch.tensor([0.1, -0.2], dtype=torch.float64)
    residual = torch.randn(
        batch,
        sequence,
        target_hidden,
        generator=generator,
        dtype=torch.float64,
    )
    endpoint = (
        residual
        + actual_context @ output_weight.T
        + output_bias
    )
    return (
        Lfm2CachedShortConvFoldCase(
            target_input=target_input,
            parent_b=torch.randn(
                batch,
                sequence,
                source_hidden,
                generator=generator,
                dtype=torch.float64,
            ),
            parent_c=torch.randn(
                batch,
                sequence,
                source_hidden,
                generator=generator,
                dtype=torch.float64,
            ),
            parent_x=torch.randn(
                batch,
                sequence,
                source_hidden,
                generator=generator,
                dtype=torch.float64,
            ),
            parent_post_operator_residual_target_space=endpoint,
            target_pre_operator_residual=residual,
            reset_state=True,
        ),
        actual_context,
    )


def test_short_conv_split_installs_core_then_actual_context_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(501)
    source = _ToyShortConv(hidden=4, fir_length=3, bias=True)
    target = _ToyShortConv(hidden=2, fir_length=3, bias=True)
    _seed_gradients(target, 502)
    identities = _identities(target)
    gradient_identities = _gradient_identities(target)
    case, actual_context = _short_conv_case()
    assignment = Lfm2SparseResidualAssignment(
        target_width=2,
        source_to_target=(0, 1, 0, 1),
        coefficients=(1.0, 1.0, 1.0, 1.0),
    )
    roles: list[str] = []
    original_fit = module._fit_affine

    def tracked_fit(*args, **kwargs):
        roles.append(kwargs["role"])
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(module, "_fit_affine", tracked_fit)
    producer = compile_lfm2_structural_short_conv_producer_fold(
        source_in_proj_weight=source.in_proj.weight,
        source_conv_weight=source.conv.weight,
        source_out_proj_weight=source.out_proj.weight,
        source_in_proj_bias=source.in_proj.bias,
        source_conv_bias=source.conv.bias,
        source_out_proj_bias=source.out_proj.bias,
        target_in_proj_weight_template=target.in_proj.weight,
        target_conv_weight_template=target.conv.weight,
        target_out_proj_weight_template=target.out_proj.weight,
        target_in_proj_bias_template=target.in_proj.bias,
        target_conv_bias_template=target.conv.bias,
        target_out_proj_bias_template=target.out_proj.bias,
        cases=(case,),
        residual_assignment=assignment,
        ridge=1e-8,
        max_prior_delta_ratio=None,
    )
    assert roles == ["short_conv_in_proj"]
    assert producer.full_source_coverage is True
    transaction = install_lfm2_structural_short_conv_producer_fold(
        target, producer
    )
    consumer = compile_lfm2_structural_short_conv_consumer_fold(
        target_out_proj_weight_template=target.out_proj.weight,
        target_out_proj_bias_template=target.out_proj.bias,
        actual_child_contexts=(actual_context,),
        cases=(case,),
        out_proj_weight_prior=target.out_proj.weight.detach().clone(),
        out_proj_bias_prior=target.out_proj.bias.detach().clone(),
        ridge=0.0,
        max_prior_delta_ratio=None,
    )
    assert roles[-1] == "short_conv_output_actual_child_context"
    assert consumer.residual_endpoint_relative_rmse < 1e-10
    report = install_lfm2_structural_short_conv_consumer_fold(
        target, consumer, transaction
    )

    assert report.accepted is True
    assert transaction.committed is True
    assert _identities(target) == identities
    assert _gradient_identities(target) == gradient_identities
    assert consumer.as_dict()["context_source"] == (
        "caller-supplied-actual-child-context"
    )


def test_short_conv_split_commits_fixed_rank_factor_consumer() -> None:
    torch.manual_seed(551)
    source = _ToyShortConv(hidden=4, fir_length=3, bias=False)
    target = _ToyShortConv(hidden=2, fir_length=3, bias=False)
    target.out_proj = nn.Sequential(
        nn.Linear(2, 1, bias=False, dtype=torch.float64),
        nn.Linear(1, 2, bias=False, dtype=torch.float64),
    )
    _seed_gradients(target, 552)
    identities = _identities(target)
    gradient_identities = _gradient_identities(target)
    case, actual_context = _short_conv_case()
    assignment = Lfm2SparseResidualAssignment(
        target_width=2,
        source_to_target=(0, 1, 0, 1),
        coefficients=(1.0, 1.0, 1.0, 1.0),
    )
    effective_prior = (
        target.out_proj[1].weight.detach() @ target.out_proj[0].weight.detach()
    )
    producer = compile_lfm2_structural_short_conv_producer_fold(
        source_in_proj_weight=source.in_proj.weight,
        source_conv_weight=source.conv.weight,
        source_out_proj_weight=source.out_proj.weight,
        source_in_proj_bias=None,
        source_conv_bias=None,
        source_out_proj_bias=None,
        target_in_proj_weight_template=target.in_proj.weight,
        target_conv_weight_template=target.conv.weight,
        target_out_proj_weight_template=effective_prior,
        target_in_proj_bias_template=None,
        target_conv_bias_template=None,
        target_out_proj_bias_template=None,
        cases=(case,),
        residual_assignment=assignment,
        ridge=1e-8,
        max_prior_delta_ratio=None,
    )
    transaction = install_lfm2_structural_short_conv_producer_fold(
        target,
        producer,
    )
    assert transaction.as_dict()["consumer_parameter_names"] == [
        "out_proj.0.weight",
        "out_proj.1.weight",
    ]
    consumer = compile_lfm2_structural_short_conv_consumer_fold(
        target_out_proj_weight_template=effective_prior,
        actual_child_contexts=(actual_context,),
        cases=(case,),
        out_proj_weight_prior=effective_prior,
        ridge=0.0,
        max_prior_delta_ratio=None,
    )
    left, singular_values, right = torch.linalg.svd(
        consumer.out_proj_weight,
        full_matrices=False,
    )
    decoder = left[:, :1] * singular_values[:1].unsqueeze(0)
    encoder = right[:1, :]

    report = install_lfm2_structural_short_conv_factor_consumer_fold(
        target,
        encoder_weight=encoder,
        decoder_weight=decoder,
        transaction=transaction,
    )

    assert report.accepted is True
    assert transaction.committed is True
    assert _identities(target) == identities
    assert _gradient_identities(target) == gradient_identities
    torch.testing.assert_close(
        target.out_proj[1].weight @ target.out_proj[0].weight,
        decoder @ encoder,
        rtol=0,
        atol=0,
    )


def test_explicit_producer_rollback_is_exact_and_idempotent() -> None:
    torch.manual_seed(601)
    target = _ToyAttention(
        hidden=4, query_heads=2, key_value_heads=1, head_dim=2
    )
    _seed_gradients(target, 602)
    original_state = _state(target)
    original_ids = _identities(target)
    original_gradient_ids = _gradient_identities(target)
    original_gradients = _gradients(target)
    producer, _case, _actual = _compile_attention_producer(target)
    transaction = install_lfm2_structural_attention_producer_fold(
        target, producer
    )

    first = rollback_lfm2_structural_split_install(transaction)
    second = rollback_lfm2_structural_split_install(transaction)

    assert first.accepted is True
    assert second.accepted is True
    assert transaction.rolled_back is True
    _assert_exact_snapshot(
        target,
        state=original_state,
        identities=original_ids,
        gradient_identities=original_gradient_ids,
        gradients=original_gradients,
    )


def test_split_surfaces_do_not_execute_models_or_train() -> None:
    functions = (
        compile_lfm2_structural_attention_producer_fold,
        compile_lfm2_structural_attention_consumer_fold,
        compile_lfm2_structural_attention_value_output_fold,
        compile_lfm2_structural_short_conv_producer_fold,
        compile_lfm2_structural_short_conv_consumer_fold,
        install_lfm2_structural_attention_producer_fold,
        install_lfm2_structural_attention_consumer_fold,
        install_lfm2_structural_attention_score_producer_fold,
        install_lfm2_structural_attention_value_output_fold,
        install_lfm2_structural_short_conv_factor_consumer_fold,
        install_lfm2_structural_short_conv_producer_fold,
        install_lfm2_structural_short_conv_consumer_fold,
    )
    called = set()
    for value in functions:
        tree = ast.parse(inspect.getsource(value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    called.add(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    called.add(node.func.id)
    assert "forward" not in called
    assert "generate" not in called
    assert "backward" not in called
    assert "step" not in called
