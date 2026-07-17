from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

import neural_forge.hf_lfm2_structural_residual_attention_fold as fold_module  # noqa: E402
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    Lfm2CachedAttentionFoldCase,
    Lfm2CachedMlpFoldCase,
    Lfm2CachedShortConvFoldCase,
    Lfm2SparseResidualAssignment,
    Lfm2StructuralResidualAttentionFoldError,
    cached_attention_case_from_lfm2_trace,
    cached_attention_cases_from_lfm2_trace_phases,
    cached_mlp_case_from_lfm2_trace,
    cached_short_conv_case_from_lfm2_trace,
    compose_lfm2_structural_layer_fold,
    compile_lfm2_structural_attention_producer_fold,
    compile_lfm2_structural_residual_attention_fold,
    compile_lfm2_structural_residual_mlp_fold,
    compile_lfm2_structural_residual_short_conv_fold,
    install_lfm2_structural_attention_fold,
    install_lfm2_structural_layer_fold,
    install_lfm2_structural_short_conv_fold,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    open_lfm2_circuit_trace_cache,
)


class _ToyAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        output_size: int,
        query_heads: int,
        key_value_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(
            hidden_size, query_heads * head_dim, bias=False, dtype=dtype
        )
        self.k_proj = nn.Linear(
            hidden_size, key_value_heads * head_dim, bias=False, dtype=dtype
        )
        self.v_proj = nn.Linear(
            hidden_size, key_value_heads * head_dim, bias=False, dtype=dtype
        )
        self.out_proj = nn.Linear(
            query_heads * head_dim, output_size, bias=False, dtype=dtype
        )


class _ToyMlp(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype
        )
        self.w3 = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype
        )
        self.w2 = nn.Linear(
            intermediate_size, hidden_size, bias=False, dtype=dtype
        )


class _ToyShortConv(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        fir_length: int,
        bias: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(
            hidden_size, 3 * hidden_size, bias=bias, dtype=dtype
        )
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            fir_length,
            groups=hidden_size,
            bias=bias,
            padding=fir_length - 1,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(
            hidden_size, hidden_size, bias=bias, dtype=dtype
        )


def _source_weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(101)
    return {
        "query": torch.randn(8, 8, generator=generator, dtype=torch.float64),
        "key": torch.randn(4, 8, generator=generator, dtype=torch.float64),
        "value": torch.randn(4, 8, generator=generator, dtype=torch.float64),
        "output": torch.randn(8, 8, generator=generator, dtype=torch.float64),
    }


def _duplicate_head_oracle() -> tuple[
    dict[str, torch.Tensor],
    Lfm2CachedAttentionFoldCase,
]:
    # Four source Q heads (two GQA offsets in each of two KV groups) collapse
    # into two target Q heads and one target KV head.  The second source GQA
    # group is an exact functional duplicate of the first, so a full-source
    # merge has an exact smaller representation.
    source = _source_weights()
    target_input = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    target_q_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    # The key rows have zero sequence mean, so centering the score rows does
    # not change this exact rank-two attention oracle.
    target_k_weight = torch.tensor(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    target_v_weight = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    target_o_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    query_linear_target = (target_input @ target_q_weight.T).reshape(1, 4, 2, 2)
    query_linear_target = query_linear_target.permute(0, 2, 1, 3)
    key_linear_target = (target_input @ target_k_weight.T).reshape(1, 4, 1, 2)
    key_linear_target = key_linear_target.permute(0, 2, 1, 3)
    value_target = (target_input @ target_v_weight.T).reshape(1, 4, 1, 2)
    value_target = value_target.permute(0, 2, 1, 3)

    parent_query = torch.cat(
        [query_linear_target[:, 0:1], query_linear_target[:, 1:2]] * 2,
        dim=1,
    )
    parent_key = key_linear_target.repeat(1, 2, 1, 1)
    parent_value = value_target.repeat(1, 2, 1, 1)
    repeated_parent_key = parent_key.repeat_interleave(2, dim=1)
    parent_scores = (
        parent_query @ repeated_parent_key.transpose(-2, -1)
    ) / (2.0**0.5)
    parent_probability = torch.softmax(parent_scores, dim=-1)
    parent_context = parent_probability @ parent_value.repeat_interleave(2, dim=1)

    target_scores = (
        query_linear_target
        @ key_linear_target.repeat_interleave(2, dim=1).transpose(-2, -1)
    ) / (2.0**0.5)
    target_probability = torch.softmax(target_scores, dim=-1)
    target_context = target_probability @ value_target.repeat_interleave(2, dim=1)
    flat_context = target_context.permute(0, 2, 1, 3).reshape(1, 4, 4)
    branch_delta = flat_context @ target_o_weight.T
    residual = torch.tensor(
        [[[0.25, -0.5, 0.75, -1.0]]], dtype=torch.float64
    ).repeat(1, 4, 1)
    endpoint = residual + branch_delta
    case = Lfm2CachedAttentionFoldCase(
        target_query_input=target_input,
        target_key_value_input=target_input,
        parent_query_linear=parent_query,
        parent_key_linear=parent_key,
        parent_value=parent_value,
        parent_query=parent_query,
        parent_key=parent_key,
        parent_context=parent_context,
        parent_post_attention_residual_target_space=endpoint,
        target_pre_attention_residual=residual,
    )
    return source, case


def _compile(
    source: dict[str, torch.Tensor],
    case: Lfm2CachedAttentionFoldCase,
    target: _ToyAttention,
):
    return compile_lfm2_structural_residual_attention_fold(
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


def test_exact_duplicate_attention_oracle_is_full_coverage_and_structural() -> None:
    source, case = _duplicate_head_oracle()
    source_before = {name: value.clone() for name, value in source.items()}
    torch.manual_seed(201)
    target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.float64,
    )

    compiled = _compile(source, case, target)

    assert compiled.audit.all_hold is True
    assert compiled.cluster_plan.report.full_source_coverage is True
    assert sorted(
        source_head
        for group in compiled.query_groups
        for source_head in group
    ) == [0, 1, 2, 3]
    assert sorted(
        source_head
        for group in compiled.key_value_groups
        for source_head in group
    ) == [0, 1]
    assert compiled.as_dict()["full_source_assignment_fraction"] == 1.0
    assert compiled.as_dict()["source_groups_dropped"] == 0
    assert compiled.centered_score_relative_rmse < 1e-10
    assert compiled.residual_endpoint_relative_rmse < 1e-10

    expected_shapes = {
        "q_proj.weight": (4, 4),
        "k_proj.weight": (2, 4),
        "v_proj.weight": (2, 4),
        "out_proj.weight": (4, 4),
    }
    state = compiled.state_dict()
    assert {name: tuple(value.shape) for name, value in state.items()} == expected_shapes
    source_numels = {
        "q_proj.weight": source["query"].numel(),
        "k_proj.weight": source["key"].numel(),
        "v_proj.weight": source["value"].numel(),
        "out_proj.weight": source["output"].numel(),
    }
    assert all(state[name].numel() < source_numels[name] for name in state)
    assert compiled.source_logical_parameters == sum(source_numels.values())
    assert compiled.target_logical_parameters == sum(
        value.numel() for value in state.values()
    )
    assert compiled.target_logical_parameters == 48
    assert compiled.removed_logical_parameters == 144
    assert compiled.parameter_reduction_fraction == pytest.approx(0.75)
    for name, value in source.items():
        assert torch.equal(value, source_before[name])


def test_compile_erases_blank_seed_and_install_preserves_parameter_identity() -> None:
    source, case = _duplicate_head_oracle()
    torch.manual_seed(301)
    first_target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.float64,
    )
    torch.manual_seed(999)
    second_target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.float64,
    )
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first_target.parameters(), second_target.parameters())
    )

    first = _compile(source, case, first_target)
    second = _compile(source, case, second_target)

    assert first.target_template_sha256 == second.target_template_sha256
    assert first.compiled_state_sha256 == second.compiled_state_sha256
    for name in first.state_dict():
        assert torch.equal(first.state_dict()[name], second.state_dict()[name])

    parameter_ids = tuple(id(value) for value in first_target.parameters())
    parameter_shapes = tuple(tuple(value.shape) for value in first_target.parameters())
    for parameter in first_target.parameters():
        parameter.grad = torch.zeros_like(parameter)
    gradient_ids = tuple(id(value.grad) for value in first_target.parameters())
    report = install_lfm2_structural_attention_fold(first_target, first)

    assert report.accepted is True
    assert report.target_logical_parameters == first.target_logical_parameters
    assert tuple(id(value) for value in first_target.parameters()) == parameter_ids
    assert tuple(tuple(value.shape) for value in first_target.parameters()) == (
        parameter_shapes
    )
    assert tuple(id(value.grad) for value in first_target.parameters()) == gradient_ids
    installed = {
        "q_proj.weight": first_target.q_proj.weight.detach(),
        "k_proj.weight": first_target.k_proj.weight.detach(),
        "v_proj.weight": first_target.v_proj.weight.detach(),
        "out_proj.weight": first_target.out_proj.weight.detach(),
    }
    for name, value in first.state_dict().items():
        assert torch.equal(installed[name], value)


def test_bfloat16_target_is_dense_and_all_four_projections_are_smaller() -> None:
    source, case = _duplicate_head_oracle()
    target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.bfloat16,
    )

    compiled = _compile(source, case, target)

    assert compiled.audit.every_projection_structurally_smaller is True
    assert compiled.audit.target_shapes_exact is True
    for value in compiled.state_dict().values():
        assert value.dtype == torch.bfloat16
        assert value.layout == torch.strided
        assert value.is_floating_point()


def _runtime_lfm2_head_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    direction = value.float()
    direction = direction * torch.rsqrt(
        direction.square().mean(dim=-1, keepdim=True) + epsilon
    )
    return weight * direction.to(dtype=value.dtype)


def _runtime_bfloat16_rope(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    return (
        value * cosine.unsqueeze(1)
        + fold_module._rotate_half(value) * sine.unsqueeze(1)
    )


def test_bfloat16_rope_recovery_is_quantization_aware_and_qk_cross_checked() -> None:
    generator = torch.Generator().manual_seed(2026071701)
    epsilon = 1e-5
    query_linear = torch.randn(
        1, 4, 5, 4, generator=generator, dtype=torch.bfloat16
    )
    key_linear = torch.randn(
        1, 2, 5, 4, generator=generator, dtype=torch.bfloat16
    )
    query_weight = torch.tensor(
        [0.75, 1.125, 0.875, 1.25],
        dtype=torch.bfloat16,
    )
    key_weight = torch.tensor(
        [1.25, 0.875, 1.125, 0.75],
        dtype=torch.bfloat16,
    )
    angles = torch.tensor(
        [[0.0, 0.15, 0.30, 0.45, 0.60]],
        dtype=torch.float32,
    ).unsqueeze(-1)
    cosine_half = torch.cat((angles.cos(), (angles * 0.5).cos()), dim=-1)
    sine_half = torch.cat((angles.sin(), (angles * 0.5).sin()), dim=-1)
    cosine = torch.cat((cosine_half, cosine_half), dim=-1).to(torch.bfloat16)
    sine = torch.cat((sine_half, sine_half), dim=-1).to(torch.bfloat16)
    normalized_query_runtime = _runtime_lfm2_head_norm(
        query_linear,
        query_weight,
        epsilon,
    )
    normalized_key_runtime = _runtime_lfm2_head_norm(
        key_linear,
        key_weight,
        epsilon,
    )
    rotated_query = _runtime_bfloat16_rope(
        normalized_query_runtime,
        cosine,
        sine,
    )
    rotated_key = _runtime_bfloat16_rope(
        normalized_key_runtime,
        cosine,
        sine,
    )
    normalized_query = fold_module._cached_lfm2_rms_norm(
        query_linear,
        query_weight,
        epsilon,
    )
    normalized_key = fold_module._cached_lfm2_rms_norm(
        key_linear,
        key_weight,
        epsilon,
    )

    query_cosine, query_sine = fold_module._infer_rope_from_cached_rows(
        normalized_query,
        rotated_query.to(torch.float64),
        "synthetic.query",
        evidence_dtype=torch.bfloat16,
    )
    key_cosine, key_sine = fold_module._infer_rope_from_cached_rows(
        normalized_key,
        rotated_key.to(torch.float64),
        "synthetic.key",
        evidence_dtype=torch.bfloat16,
    )
    fold_module._validate_shared_qk_rope(
        query_cosine,
        query_sine,
        key_cosine,
        key_sine,
        "synthetic",
        evidence_dtype=torch.bfloat16,
    )

    shifted_cosine = torch.roll(cosine, shifts=1, dims=1)
    shifted_sine = torch.roll(sine, shifts=1, dims=1)
    shifted_key = _runtime_bfloat16_rope(
        normalized_key_runtime,
        shifted_cosine,
        shifted_sine,
    )
    wrong_key_cosine, wrong_key_sine = fold_module._infer_rope_from_cached_rows(
        normalized_key,
        shifted_key.to(torch.float64),
        "synthetic.shifted_key",
        evidence_dtype=torch.bfloat16,
    )
    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="same legal causal positions",
    ):
        fold_module._validate_shared_qk_rope(
            query_cosine,
            query_sine,
            wrong_key_cosine,
            wrong_key_sine,
            "synthetic.shifted",
            evidence_dtype=torch.bfloat16,
        )


def test_full_prefix_phase_assembler_never_concatenates_overlapping_phases() -> None:
    generator = torch.Generator().manual_seed(2026071702)
    tensors: dict[str, torch.Tensor] = {}
    for phase, length in (("prefill", 3), ("decode.000", 2)):
        prefix = f"{phase}.layer.002."
        query = torch.randn(
            1, 2, length, 2, generator=generator, dtype=torch.float64
        )
        key = torch.randn(
            1, 1, length, 2, generator=generator, dtype=torch.float64
        )
        tensors.update(
            {
                prefix + "operator.input": torch.randn(
                    1, length, 4, generator=generator, dtype=torch.float64
                ),
                prefix + "attention.q_linear": query.clone(),
                prefix + "attention.k_linear": key.clone(),
                prefix + "attention.q": query,
                prefix + "attention.k": key,
                prefix + "attention.v": key.clone(),
                prefix + "attention.context": query.clone(),
                prefix + "residual.pre": torch.randn(
                    1, length, 4, generator=generator, dtype=torch.float64
                ),
                prefix + "residual.post_operator": torch.randn(
                    1, length, 4, generator=generator, dtype=torch.float64
                ),
            }
        )

    cases = cached_attention_cases_from_lfm2_trace_phases(
        tensors,
        phases=("prefill", "decode.000"),
        layer_index=2,
        decode_trace_mode="full_prefix_canonical",
    )

    assert [case.parent_key.shape[2] for case in cases] == [3, 2]
    assert [case.target_key_value_input.shape[1] for case in cases] == [3, 2]
    assert cases[1].key_mask.tolist() == [
        [[True, False], [True, True]]
    ]
    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="missing.*key/value cache",
    ):
        cached_attention_cases_from_lfm2_trace_phases(
            tensors,
            phases=("prefill", "decode.000"),
            layer_index=2,
            decode_trace_mode="cached_verified",
        )


def test_direct_full_prefix_decode_adapter_uses_a_causal_mask() -> None:
    generator = torch.Generator().manual_seed(2026071703)
    prefix = "decode.000.layer.002."
    query = torch.randn(
        1, 2, 3, 2, generator=generator, dtype=torch.float64
    )
    key = torch.randn(
        1, 1, 3, 2, generator=generator, dtype=torch.float64
    )
    tensors = {
        prefix + "operator.input": torch.randn(
            1, 3, 4, generator=generator, dtype=torch.float64
        ),
        prefix + "attention.q_linear": query.clone(),
        prefix + "attention.k_linear": key.clone(),
        prefix + "attention.q": query,
        prefix + "attention.k": key,
        prefix + "attention.v": key.clone(),
        prefix + "attention.context": query.clone(),
        prefix + "residual.pre": torch.randn(
            1, 3, 4, generator=generator, dtype=torch.float64
        ),
        prefix + "residual.post_operator": torch.randn(
            1, 3, 4, generator=generator, dtype=torch.float64
        ),
    }

    case = cached_attention_case_from_lfm2_trace(
        tensors,
        phase="decode.000",
        layer_index=2,
        decode_trace_mode="full_prefix_canonical",
    )

    assert case.key_mask.tolist() == [
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    ]


@pytest.mark.skipif(
    not os.environ.get("NEURAL_FORGE_REAL_LFM2_CACHE")
    or not os.environ.get("NEURAL_FORGE_REAL_LFM2_MODEL"),
    reason="real sealed LFM2 cache/model paths were not supplied",
)
def test_real_bfloat16_cache_compiles_q_and_k_without_parent_forward() -> None:
    transformers = pytest.importorskip("transformers")
    cache_path = Path(os.environ["NEURAL_FORGE_REAL_LFM2_CACHE"])
    model_path = Path(os.environ["NEURAL_FORGE_REAL_LFM2_MODEL"])
    cache = open_lfm2_circuit_trace_cache(cache_path, verify=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cpu",
    )

    def forbidden_forward(*_args, **_kwargs):
        raise AssertionError("real-cache compile probe executed the parent")

    model.forward = forbidden_forward
    layer = model.model.layers[2].self_attn
    batch = cache.load_batch(0)
    case = cached_attention_case_from_lfm2_trace(
        batch.tensors,
        phase="prefill",
        layer_index=2,
        decode_trace_mode="cached_verified",
    )
    target_query = torch.empty(
        (8 * int(layer.head_dim), 1024),
        dtype=torch.bfloat16,
    )
    target_key = torch.empty(
        (4 * int(layer.head_dim), 1024),
        dtype=torch.bfloat16,
    )
    target_value = torch.empty_like(target_key)
    target_output = torch.empty(
        (1024, 8 * int(layer.head_dim)),
        dtype=torch.bfloat16,
    )
    compiled = compile_lfm2_structural_attention_producer_fold(
        source_query_weight=layer.q_proj.weight,
        source_key_weight=layer.k_proj.weight,
        source_value_weight=layer.v_proj.weight,
        source_output_weight=layer.out_proj.weight,
        target_query_weight_template=target_query,
        target_key_weight_template=target_key,
        target_value_weight_template=target_value,
        target_output_weight_template=target_output,
        cases=(case,),
        head_dim=int(layer.head_dim),
        query_norm_weight=layer.q_layernorm.weight,
        key_norm_weight=layer.k_layernorm.weight,
        query_weight_prior=layer.q_proj.weight[: target_query.shape[0]],
        key_weight_prior=layer.k_proj.weight[: target_key.shape[0]],
        value_weight_prior=layer.v_proj.weight[: target_value.shape[0]],
        output_weight_prior=layer.out_proj.weight[:, : target_output.shape[1]],
        residual_assignment_sha256="f" * 64,
        norm_epsilon=float(layer.q_layernorm.variance_epsilon),
        ridge=1e-6,
        max_prior_delta_ratio=0.0,
    )

    assert compiled.full_source_coverage is True
    assert torch.isfinite(compiled.query_weight.float()).all()
    assert torch.isfinite(compiled.key_weight.float()).all()


def test_trace_adapter_uses_residual_endpoint_not_isolated_attention_delta() -> None:
    prefix = "prefill.layer.003."
    query_linear = torch.zeros(1, 2, 2, 2, dtype=torch.float64)
    key_linear = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    tensors = {
        prefix + "operator.input": torch.zeros(1, 2, 4, dtype=torch.float64),
        prefix + "attention.q_linear": query_linear,
        prefix + "attention.k_linear": key_linear,
        prefix + "attention.v": key_linear.clone(),
        prefix + "attention.q": query_linear.clone(),
        prefix + "attention.k": key_linear.clone(),
        prefix + "attention.context": query_linear.clone(),
        prefix + "attention.o": torch.full(
            (1, 2, 4), 7.0, dtype=torch.float64
        ),
        prefix + "residual.pre": torch.full(
            (1, 2, 4), 11.0, dtype=torch.float64
        ),
        prefix + "residual.post_operator": torch.full(
            (1, 2, 4), 18.0, dtype=torch.float64
        ),
    }

    case = cached_attention_case_from_lfm2_trace(
        tensors, phase="prefill", layer_index=3
    )

    assert torch.equal(
        case.parent_post_attention_residual_target_space,
        tensors[prefix + "residual.post_operator"],
    )
    assert torch.equal(
        case.target_pre_attention_residual,
        tensors[prefix + "residual.pre"],
    )
    assert not torch.equal(
        case.parent_post_attention_residual_target_space,
        tensors[prefix + "attention.o"],
    )
    assert torch.equal(
        case.parent_post_attention_residual_target_space
        - case.target_pre_attention_residual,
        tensors[prefix + "attention.o"],
    )


def test_sealed_phase_assembler_reuses_prefill_and_decode_kv_history() -> None:
    generator = torch.Generator().manual_seed(661)
    tensors: dict[str, torch.Tensor] = {}
    key_history = []
    value_history = []
    for phase, length in (("prefill", 2), ("decode.000", 1)):
        prefix = f"{phase}.layer.001."
        query = torch.randn(
            1, 2, length, 2, generator=generator, dtype=torch.float64
        )
        key = torch.randn(
            1, 1, length, 2, generator=generator, dtype=torch.float64
        )
        value = torch.randn(
            1, 1, length, 2, generator=generator, dtype=torch.float64
        )
        key_history.append(key)
        value_history.append(value)
        tensors[prefix + "operator.input"] = torch.randn(
            1, length, 4, generator=generator, dtype=torch.float64
        )
        tensors[prefix + "attention.q_linear"] = query.clone()
        tensors[prefix + "attention.k_linear"] = key.clone()
        tensors[prefix + "attention.q"] = query
        tensors[prefix + "attention.k"] = key
        tensors[prefix + "attention.v"] = value
        tensors[prefix + "attention.context"] = torch.randn(
            1, 2, length, 2, generator=generator, dtype=torch.float64
        )
        tensors[prefix + "residual.pre"] = torch.randn(
            1, length, 4, generator=generator, dtype=torch.float64
        )
        tensors[prefix + "residual.post_operator"] = torch.randn(
            1, length, 4, generator=generator, dtype=torch.float64
        )
        if phase.startswith("decode"):
            tensors[prefix + "attention.key_cache"] = torch.cat(
                key_history, dim=2
            )
            tensors[prefix + "attention.value_cache"] = torch.cat(
                value_history, dim=2
            )

    cases = cached_attention_cases_from_lfm2_trace_phases(
        tensors,
        phases=("prefill", "decode.000"),
        layer_index=1,
    )

    assert len(cases) == 2
    assert cases[0].parent_key.shape[2] == 2
    assert cases[1].parent_key.shape[2] == 3
    assert cases[1].parent_key_linear.shape[2] == 3
    assert cases[1].target_key_value_input.shape[1] == 3
    assert cases[0].key_mask.tolist() == [[[True, False], [True, True]]]
    assert bool(cases[1].key_mask.all().item()) is True


def test_sealed_phase_assembler_keeps_real_rolling_cache_alignment() -> None:
    generator = torch.Generator().manual_seed(20260717)
    tensors: dict[str, torch.Tensor] = {}
    prefill_length = 8
    head_dim = 4
    prefill_input = torch.randn(
        1, prefill_length, 8, generator=generator, dtype=torch.float64
    )
    prefill_q = torch.randn(
        1, 2, prefill_length, head_dim, generator=generator, dtype=torch.float64
    )
    prefill_k = torch.randn(
        1, 1, prefill_length, head_dim, generator=generator, dtype=torch.float64
    )
    prefill_v = torch.randn(
        1, 1, prefill_length, head_dim, generator=generator, dtype=torch.float64
    )
    decode_input = torch.randn(
        1, 1, 8, generator=generator, dtype=torch.float64
    )
    decode_q = torch.randn(
        1, 2, 1, head_dim, generator=generator, dtype=torch.float64
    )
    decode_k = torch.randn(
        1, 1, 1, head_dim, generator=generator, dtype=torch.float64
    )
    decode_v = torch.randn(
        1, 1, 1, head_dim, generator=generator, dtype=torch.float64
    )
    for phase, operator_input, query, key, value in (
        ("prefill", prefill_input, prefill_q, prefill_k, prefill_v),
        ("decode.000", decode_input, decode_q, decode_k, decode_v),
    ):
        prefix = f"{phase}.layer.003."
        tensors.update(
            {
                prefix + "operator.input": operator_input,
                prefix + "attention.q_linear": query.clone(),
                prefix + "attention.k_linear": key.clone(),
                prefix + "attention.q": query,
                prefix + "attention.k": key,
                prefix + "attention.v": value,
                prefix + "attention.context": torch.randn(
                    1,
                    2,
                    query.shape[2],
                    head_dim,
                    generator=generator,
                    dtype=torch.float64,
                ),
                prefix + "residual.pre": torch.randn(
                    1,
                    query.shape[2],
                    8,
                    generator=generator,
                    dtype=torch.float64,
                ),
                prefix + "residual.post_operator": torch.randn(
                    1,
                    query.shape[2],
                    8,
                    generator=generator,
                    dtype=torch.float64,
                ),
            }
        )
    rolling_key = torch.cat((prefill_k, decode_k), dim=2)[:, :, -8:, :]
    rolling_value = torch.cat((prefill_v, decode_v), dim=2)[:, :, -8:, :]
    tensors["decode.000.layer.003.attention.key_cache"] = rolling_key.clone()
    tensors["decode.000.layer.003.attention.value_cache"] = (
        rolling_value.clone()
    )

    cases = cached_attention_cases_from_lfm2_trace_phases(
        tensors,
        phases=("prefill", "decode.000"),
        layer_index=3,
    )

    assert len(cases) == 2
    assert cases[1].parent_key.shape[2] == 8
    assert cases[1].parent_value.shape[2] == 8
    assert cases[1].parent_key_linear.shape[2] == 8
    assert cases[1].target_key_value_input.shape[1] == 8
    assert torch.equal(cases[1].parent_key, rolling_key)
    assert torch.equal(cases[1].parent_value, rolling_value)
    assert torch.equal(
        cases[1].target_key_value_input[:, 0],
        prefill_input[:, 1],
    )


def test_mlp_trace_adapter_flattens_cached_batch_sequence_rows() -> None:
    prefix = "prefill.layer.002."
    tensors = {
        prefix + "mlp.input": torch.zeros(1, 3, 4, dtype=torch.float64),
        prefix + "mlp.gate": torch.ones(1, 3, 6, dtype=torch.float64),
        prefix + "mlp.up": torch.full((1, 3, 6), 2.0, dtype=torch.float64),
        prefix + "mlp.post_swiglu": torch.full(
            (1, 3, 6), 3.0, dtype=torch.float64
        ),
        prefix + "residual.post_operator": torch.full(
            (1, 3, 4), 5.0, dtype=torch.float64
        ),
        prefix + "residual.post_layer": torch.full(
            (1, 3, 4), 8.0, dtype=torch.float64
        ),
    }

    case = cached_mlp_case_from_lfm2_trace(
        tensors, phase="prefill", layer_index=2
    )

    assert case.target_input.shape == (3, 4)
    assert case.parent_gate.shape == (3, 6)
    assert case.parent_post_mlp_residual_target_space.shape == (3, 4)
    assert torch.equal(
        case.parent_post_mlp_residual_target_space - case.target_pre_mlp_residual,
        torch.full((3, 4), 3.0, dtype=torch.float64),
    )


def test_rejects_source_output_projection_with_non_residual_output_width() -> None:
    source, case = _duplicate_head_oracle()
    target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.float64,
    )
    bad_output = source["output"][:7]

    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="valid GQA topology",
    ):
        compile_lfm2_structural_residual_attention_fold(
            source_query_weight=source["query"],
            source_key_weight=source["key"],
            source_value_weight=source["value"],
            source_output_weight=bad_output,
            target_query_weight_template=target.q_proj.weight,
            target_key_weight_template=target.k_proj.weight,
            target_value_weight_template=target.v_proj.weight,
            target_output_weight_template=target.out_proj.weight,
            cases=(case,),
            head_dim=2,
            residual_assignment_sha256="a" * 64,
        )


def test_underdetermined_cached_fit_requires_and_preserves_source_prior_nullspace() -> None:
    source, case = _duplicate_head_oracle()
    padded_input = torch.nn.functional.pad(case.target_query_input, (0, 2))
    case = replace(
        case,
        target_query_input=padded_input,
        target_key_value_input=padded_input,
    )
    target = _ToyAttention(
        hidden_size=6,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.float64,
    )
    common = dict(
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

    with pytest.raises(
        Lfm2StructuralResidualAttentionFoldError,
        match="source-derived target-shaped prior is required",
    ):
        compile_lfm2_structural_residual_attention_fold(**common)

    generator = torch.Generator().manual_seed(777)
    q_prior = torch.randn(4, 6, generator=generator, dtype=torch.float64)
    k_prior = torch.randn(2, 6, generator=generator, dtype=torch.float64)
    v_prior = torch.randn(2, 6, generator=generator, dtype=torch.float64)
    compiled = compile_lfm2_structural_residual_attention_fold(
        **common,
        query_weight_prior=q_prior,
        key_weight_prior=k_prior,
        value_weight_prior=v_prior,
        max_prior_delta_ratio=None,
    )

    # The cache has no signal in the last two input coordinates.  Residual
    # fitting around the source-derived prior leaves that nullspace planted,
    # instead of silently resetting it to zero.
    assert torch.equal(compiled.query_weight[:, 4:], q_prior[:, 4:])
    assert torch.equal(compiled.key_weight[:, 4:], k_prior[:, 4:])
    assert torch.equal(compiled.value_weight[:, 4:], v_prior[:, 4:])
    reports = {value.role: value for value in compiled.prior_correction_reports}
    assert reports["query"].underdetermined is True
    assert reports["query"].prior_provided is True
    assert reports["key"].underdetermined is True
    assert reports["value"].underdetermined is True
    assert compiled.source_derived_prior_sha256 is not None


def test_exact_duplicate_swiglu_oracle_uses_every_neuron_and_installs_atomically() -> None:
    generator = torch.Generator().manual_seed(880)
    source_gate_weight = torch.randn(
        4, 8, generator=generator, dtype=torch.float64
    )
    source_up_weight = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    source_down_weight = torch.randn(
        8, 4, generator=generator, dtype=torch.float64
    )
    target_input = torch.eye(4, dtype=torch.float64)
    desired_gate = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    desired_up = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    gate_two = target_input @ desired_gate.T
    up_two = target_input @ desired_up.T
    parent_gate = torch.stack(
        (gate_two[:, 0], gate_two[:, 0], gate_two[:, 1], gate_two[:, 1]),
        dim=1,
    )
    parent_up = torch.stack(
        (up_two[:, 0], up_two[:, 0], up_two[:, 1], up_two[:, 1]),
        dim=1,
    )
    parent_post = torch.nn.functional.silu(parent_gate) * parent_up
    desired_down = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]],
        dtype=torch.float64,
    )
    target_post = torch.nn.functional.silu(gate_two) * up_two
    residual = torch.tensor(
        [[0.25, -0.5, 0.75, -1.0]], dtype=torch.float64
    ).repeat(4, 1)
    endpoint = residual + target_post @ desired_down.T
    case = Lfm2CachedMlpFoldCase(
        target_input=target_input,
        parent_gate=parent_gate,
        parent_up=parent_up,
        parent_post_swiglu=parent_post,
        parent_post_mlp_residual_target_space=endpoint,
        target_pre_mlp_residual=residual,
    )
    target = _ToyMlp(
        hidden_size=4, intermediate_size=2, dtype=torch.float64
    )
    parameter_ids = tuple(id(value) for value in target.parameters())
    parameter_shapes = tuple(tuple(value.shape) for value in target.parameters())

    compiled = compile_lfm2_structural_residual_mlp_fold(
        source_gate_weight=source_gate_weight,
        source_up_weight=source_up_weight,
        source_down_weight=source_down_weight,
        target_gate_weight_template=target.w1.weight,
        target_up_weight_template=target.w3.weight,
        target_down_weight_template=target.w2.weight,
        cases=(case,),
        residual_assignment_sha256="a" * 64,
        ridge=0.0,
        cluster_ridge=0.0,
    )

    assert compiled.full_source_coverage is True
    assert compiled.as_dict()["source_neurons_dropped"] == 0
    assert sorted(
        neuron for group in compiled.neuron_groups for neuron in group
    ) == [0, 1, 2, 3]
    assert compiled.residual_endpoint_relative_rmse < 1e-10
    assert compiled.source_logical_parameters == 96
    assert compiled.target_logical_parameters == 24
    assert compiled.removed_logical_parameters == 72
    assert compiled.parameter_reduction_fraction == pytest.approx(0.75)
    assert {
        name: tuple(value.shape)
        for name, value in compiled.state_dict().items()
    } == {
        "w1.weight": (2, 4),
        "w3.weight": (2, 4),
        "w2.weight": (4, 2),
    }

    attention_source, attention_case = _duplicate_head_oracle()
    attention_target = _ToyAttention(
        hidden_size=4,
        output_size=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
        dtype=torch.float64,
    )
    attention_compiled = _compile(
        attention_source, attention_case, attention_target
    )
    target_layer = nn.Module()
    target_layer.self_attn = attention_target
    target_layer.feed_forward = target
    all_parameter_ids = tuple(id(value) for value in target_layer.parameters())
    all_parameter_shapes = tuple(
        tuple(value.shape) for value in target_layer.parameters()
    )
    layer_compiled = compose_lfm2_structural_layer_fold(
        attention_compiled, compiled
    )
    report = install_lfm2_structural_layer_fold(target_layer, layer_compiled)
    assert report.accepted is True
    assert report.target_logical_parameters == 72
    assert tuple(id(value) for value in target_layer.parameters()) == all_parameter_ids
    assert tuple(tuple(value.shape) for value in target_layer.parameters()) == (
        all_parameter_shapes
    )
    assert tuple(id(value) for value in target.parameters()) == parameter_ids
    assert tuple(tuple(value.shape) for value in target.parameters()) == (
        parameter_shapes
    )
    assert torch.equal(target.w1.weight, compiled.gate_weight)
    assert torch.equal(target.w3.weight, compiled.up_weight)
    assert torch.equal(target.w2.weight, compiled.down_weight)


def test_short_conv_fold_uses_all_channels_actual_fir_and_endpoint_prior() -> None:
    generator = torch.Generator().manual_seed(1441)
    target_input = torch.randn(
        1, 7, 2, generator=generator, dtype=torch.float64
    )
    desired_in_weight = torch.tensor(
        [
            [1.0, 0.2],
            [-0.3, 0.8],
            [0.4, -0.7],
            [0.9, 0.1],
            [-0.6, 0.5],
            [0.25, 1.1],
        ],
        dtype=torch.float64,
    )
    desired_in_bias = torch.tensor(
        [0.1, -0.2, 0.05, 0.15, -0.1, 0.2],
        dtype=torch.float64,
    )
    desired_kernel = torch.tensor(
        [[[0.25, 0.75]], [[-0.5, 0.4]]], dtype=torch.float64
    )
    desired_conv_bias = torch.tensor(
        [0.125, -0.075], dtype=torch.float64
    )
    desired_out_weight = torch.tensor(
        [[1.2, -0.4], [0.3, 0.9]], dtype=torch.float64
    )
    desired_out_bias = torch.tensor(
        [0.05, -0.1], dtype=torch.float64
    )

    projected = target_input @ desired_in_weight.T + desired_in_bias
    target_b, target_c, target_x = projected.chunk(3, dim=-1)
    target_bx = (target_b * target_x).transpose(1, 2)
    filtered = torch.nn.functional.conv1d(
        torch.nn.functional.pad(target_bx, (1, 0)),
        desired_kernel,
        bias=desired_conv_bias,
        groups=2,
    )
    target_context = target_c * filtered.transpose(1, 2)
    desired_delta = (
        target_context @ desired_out_weight.T + desired_out_bias
    )
    residual = torch.randn(
        1, 7, 2, generator=generator, dtype=torch.float64
    )
    endpoint = residual + desired_delta

    def duplicate_channels(value: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                value[..., 0],
                value[..., 0],
                value[..., 1],
                value[..., 1],
            ),
            dim=-1,
        )

    parent_b = duplicate_channels(target_b)
    parent_c = duplicate_channels(target_c)
    parent_x = duplicate_channels(target_x)
    parent_contribution = duplicate_channels(target_context)
    split = 4
    cases = (
        Lfm2CachedShortConvFoldCase(
            target_input=target_input[:, :split],
            parent_b=parent_b[:, :split],
            parent_c=parent_c[:, :split],
            parent_x=parent_x[:, :split],
            parent_post_operator_residual_target_space=endpoint[:, :split],
            target_pre_operator_residual=residual[:, :split],
            reset_state=True,
            parent_contribution=parent_contribution[:, :split],
        ),
        Lfm2CachedShortConvFoldCase(
            target_input=target_input[:, split:],
            parent_b=parent_b[:, split:],
            parent_c=parent_c[:, split:],
            parent_x=parent_x[:, split:],
            parent_post_operator_residual_target_space=endpoint[:, split:],
            target_pre_operator_residual=residual[:, split:],
            reset_state=False,
            parent_contribution=parent_contribution[:, split:],
        ),
    )
    source_values = {
        "in_proj.weight": torch.randn(
            12, 4, generator=generator, dtype=torch.float64
        ),
        "in_proj.bias": torch.randn(
            12, generator=generator, dtype=torch.float64
        ),
        "conv.weight": torch.stack(
            (
                desired_kernel[0],
                desired_kernel[0],
                desired_kernel[1],
                desired_kernel[1],
            )
        ),
        "conv.bias": torch.tensor(
            [
                desired_conv_bias[0],
                desired_conv_bias[0],
                desired_conv_bias[1],
                desired_conv_bias[1],
            ],
            dtype=torch.float64,
        ),
        "out_proj.weight": torch.randn(
            4, 4, generator=generator, dtype=torch.float64
        ),
        "out_proj.bias": torch.randn(
            4, generator=generator, dtype=torch.float64
        ),
    }
    source_before = {
        name: value.clone() for name, value in source_values.items()
    }
    target = _ToyShortConv(
        hidden_size=2, fir_length=2, bias=True, dtype=torch.float64
    )
    assignment = Lfm2SparseResidualAssignment(
        target_width=2,
        source_to_target=(0, 0, 1, 1),
        coefficients=(1.0, 1.0, 1.0, 1.0),
    )

    compiled = compile_lfm2_structural_residual_short_conv_fold(
        source_in_proj_weight=source_values["in_proj.weight"],
        source_in_proj_bias=source_values["in_proj.bias"],
        source_conv_weight=source_values["conv.weight"],
        source_conv_bias=source_values["conv.bias"],
        source_out_proj_weight=source_values["out_proj.weight"],
        source_out_proj_bias=source_values["out_proj.bias"],
        target_in_proj_weight_template=target.in_proj.weight,
        target_in_proj_bias_template=target.in_proj.bias,
        target_conv_weight_template=target.conv.weight,
        target_conv_bias_template=target.conv.bias,
        target_out_proj_weight_template=target.out_proj.weight,
        target_out_proj_bias_template=target.out_proj.bias,
        cases=cases,
        residual_assignment=assignment,
        ridge=0.0,
        max_prior_delta_ratio=None,
    )

    assert compiled.full_source_coverage is True
    assert compiled.residual_assignment_sha256 == assignment.assignment_sha256
    assert compiled.as_dict()["source_channels_dropped"] == 0
    assert sorted(
        source
        for group in compiled.channel_groups
        for source in group
    ) == [0, 1, 2, 3]
    assert compiled.residual_endpoint_relative_rmse < 1e-10
    assert compiled.source_logical_parameters == 92
    assert compiled.target_logical_parameters == 30
    assert compiled.removed_logical_parameters == 62
    assert compiled.parameter_reduction_fraction == pytest.approx(62 / 92)
    assert {
        name: tuple(value.shape)
        for name, value in compiled.state_dict().items()
    } == {
        "in_proj.weight": (6, 2),
        "conv.weight": (2, 1, 2),
        "out_proj.weight": (2, 2),
        "in_proj.bias": (6,),
        "conv.bias": (2,),
        "out_proj.bias": (2,),
    }
    for name, value in source_values.items():
        assert torch.equal(value, source_before[name])

    parameter_ids = tuple(id(value) for value in target.parameters())
    parameter_shapes = tuple(
        tuple(value.shape) for value in target.parameters()
    )
    for parameter in target.parameters():
        parameter.grad = torch.zeros_like(parameter)
    gradient_ids = tuple(id(value.grad) for value in target.parameters())
    report = install_lfm2_structural_short_conv_fold(target, compiled)
    assert report.accepted is True
    assert tuple(id(value) for value in target.parameters()) == parameter_ids
    assert tuple(tuple(value.shape) for value in target.parameters()) == (
        parameter_shapes
    )
    assert tuple(id(value.grad) for value in target.parameters()) == gradient_ids

    installed_projection = target.in_proj(target_input)
    installed_b, installed_c, installed_x = installed_projection.chunk(
        3, dim=-1
    )
    installed_filtered = torch.nn.functional.conv1d(
        torch.nn.functional.pad(
            (installed_b * installed_x).transpose(1, 2), (1, 0)
        ),
        target.conv.weight,
        bias=target.conv.bias,
        groups=2,
    )
    installed_delta = target.out_proj(
        installed_c * installed_filtered.transpose(1, 2)
    )
    torch.testing.assert_close(
        residual + installed_delta,
        endpoint,
        rtol=1e-9,
        atol=1e-10,
    )


def test_short_conv_trace_adapter_preserves_endpoint_and_state_order() -> None:
    tensors: dict[str, torch.Tensor] = {}
    for phase, sequence in (("prefill", 3), ("decode.000", 1)):
        prefix = f"{phase}.layer.004."
        tensors[prefix + "operator.input"] = torch.zeros(
            1, sequence, 2, dtype=torch.float64
        )
        for role, fill in (
            ("short_conv.b", 1.0),
            ("short_conv.c", 2.0),
            ("short_conv.x", 3.0),
            ("short_conv.context", 4.0),
        ):
            tensors[prefix + role] = torch.full(
                (1, sequence, 4), fill, dtype=torch.float64
            )
        tensors[prefix + "residual.pre"] = torch.full(
            (1, sequence, 2), 5.0, dtype=torch.float64
        )
        tensors[prefix + "residual.post_operator"] = torch.full(
            (1, sequence, 2), 9.0, dtype=torch.float64
        )

    prefill = cached_short_conv_case_from_lfm2_trace(
        tensors, phase="prefill", layer_index=4
    )
    decode = cached_short_conv_case_from_lfm2_trace(
        tensors, phase="decode.000", layer_index=4
    )

    assert prefill.reset_state is True
    assert decode.reset_state is False
    assert torch.equal(
        prefill.parent_post_operator_residual_target_space
        - prefill.target_pre_operator_residual,
        torch.full((1, 3, 2), 4.0, dtype=torch.float64),
    )
    assert torch.equal(
        prefill.parent_contribution,
        tensors["prefill.layer.004.short_conv.context"],
    )
