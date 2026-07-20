from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields, replace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_cached_qk_functional_refit import (  # noqa: E402
    Lfm2CachedQkArtifactAuthenticationError,
    Lfm2CachedQkFunctionalRefitError,
    Lfm2CachedQkFunctionalRefitPolicy,
    adapt_lfm2_cached_attention_case_for_qk_refit,
    compile_lfm2_cached_qk_functional_refit,
    partition_lfm2_cached_qk_fit_cases,
    seal_lfm2_cached_qk_fit_bank,
    seal_lfm2_cached_qk_fit_check_bank,
)
from neural_forge.hf_lfm2_cpcf_attention_dimension_fold import (  # noqa: E402
    Lfm2CpcfAttentionCaseProvenance,
    Lfm2CpcfAttentionDimensionFoldError,
    Lfm2CpcfAttentionDimensionRuntime,
    Lfm2CpcfAttentionDimensionTopology,
    Lfm2CpcfBarycentricHeadGroup,
    compile_lfm2_cpcf_attention_dimension_fold,
    invert_lfm2_rope,
    lfm2_runtime_rms_norm,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    Lfm2CachedAttentionFoldCase,
)


DTYPE = torch.bfloat16
SOURCE_Q = 16
SOURCE_KV = 8
TARGET_Q = 10
TARGET_KV = 5
HEAD_DIM = 64
HIDDEN = 8
SOURCE_SPLIT = hashlib.sha256(b"upstream-fit-receipt").hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _topology() -> Lfm2CpcfAttentionDimensionTopology:
    return Lfm2CpcfAttentionDimensionTopology(
        source_query_heads=SOURCE_Q,
        source_key_value_heads=SOURCE_KV,
        target_query_heads=TARGET_Q,
        target_key_value_heads=TARGET_KV,
        source_head_dim=HEAD_DIM,
        target_head_dim=HEAD_DIM,
    )


def _groups() -> tuple[
    tuple[Lfm2CpcfBarycentricHeadGroup, ...],
    tuple[Lfm2CpcfBarycentricHeadGroup, ...],
]:
    kv_members = ((0, 1), (2, 3), (4, 5), (6,), (7,))
    key_groups = tuple(
        Lfm2CpcfBarycentricHeadGroup(
            members,
            tuple(1.0 / len(members) for _ in members),
        )
        for members in kv_members
    )
    query_groups = []
    for members in kv_members:
        for offset in (0, 1):
            sources = tuple(2 * value + offset for value in members)
            query_groups.append(
                Lfm2CpcfBarycentricHeadGroup(
                    sources,
                    tuple(1.0 / len(sources) for _ in sources),
                )
            )
    return tuple(query_groups), key_groups


def _norm_weight() -> torch.Tensor:
    return torch.linspace(0.73, 1.27, HEAD_DIM, dtype=torch.float32).to(DTYPE)


def _runtime(*, hidden: int = HIDDEN) -> Lfm2CpcfAttentionDimensionRuntime:
    norm = _norm_weight()
    return Lfm2CpcfAttentionDimensionRuntime(
        source_query_norm_weight=norm,
        source_key_norm_weight=torch.flip(norm, dims=(0,)).contiguous(),
        target_query_norm_weight=norm,
        target_key_norm_weight=torch.flip(norm, dims=(0,)).contiguous(),
        target_query_weight_template=torch.zeros(
            (TARGET_Q * HEAD_DIM, hidden),
            dtype=DTYPE,
        ),
        target_key_weight_template=torch.zeros(
            (TARGET_KV * HEAD_DIM, hidden),
            dtype=DTYPE,
        ),
        norm_epsilon=1e-5,
    )


def _true_weights() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20001)
    query = (
        torch.randn(
            TARGET_Q * HEAD_DIM,
            HIDDEN,
            generator=generator,
            dtype=torch.float32,
        )
        / math.sqrt(HIDDEN)
    ).to(DTYPE)
    key = (
        torch.randn(
            TARGET_KV * HEAD_DIM,
            HIDDEN,
            generator=generator,
            dtype=torch.float32,
        )
        / math.sqrt(HIDDEN)
    ).to(DTYPE)
    return query, key


def _rope(
    batch: int,
    length: int,
    *,
    start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = HEAD_DIM // 2
    positions = torch.arange(
        start,
        start + length,
        dtype=torch.float32,
    ).reshape(length, 1)
    frequencies = torch.linspace(
        0.011,
        0.193,
        half,
        dtype=torch.float32,
    ).reshape(1, half)
    angles = positions * frequencies
    # A legal scaled LFM2 rotary table has a finite non-unit BF16 radius.
    scale = 0.998
    cosine_half = (angles.cos() * scale).to(DTYPE)
    sine_half = (angles.sin() * scale).to(DTYPE)
    cosine = torch.cat((cosine_half, cosine_half), dim=-1)
    sine = torch.cat((sine_half, sine_half), dim=-1)
    return (
        cosine.unsqueeze(0).expand(batch, -1, -1).clone(),
        sine.unsqueeze(0).expand(batch, -1, -1).clone(),
    )


def _apply_rope(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    half = value.shape[-1] // 2
    rotate_half = torch.cat(
        (-value[..., half:], value[..., :half]),
        dim=-1,
    )
    return value * cosine.unsqueeze(1) + rotate_half * sine.unsqueeze(1)


def _case(
    seed: int,
    *,
    batch: int = 1,
    query_length: int = 1,
    key_length: int = 4,
) -> Lfm2CachedAttentionFoldCase:
    generator = torch.Generator().manual_seed(seed)
    query_input = torch.randn(
        batch,
        query_length,
        HIDDEN,
        generator=generator,
        dtype=torch.float32,
    ).to(DTYPE)
    key_input = torch.randn(
        batch,
        key_length,
        HIDDEN,
        generator=generator,
        dtype=torch.float32,
    ).to(DTYPE)
    query_weight, key_weight = _true_weights()
    target_query_linear = (
        (query_input @ query_weight.T)
        .reshape(batch, query_length, TARGET_Q, HEAD_DIM)
        .permute(0, 2, 1, 3)
    )
    target_key_linear = (
        (key_input @ key_weight.T)
        .reshape(batch, key_length, TARGET_KV, HEAD_DIM)
        .permute(0, 2, 1, 3)
    )
    query_source_to_target = torch.tensor(
        (0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9),
        dtype=torch.long,
    )
    key_source_to_target = torch.tensor(
        (0, 0, 1, 1, 2, 2, 3, 4),
        dtype=torch.long,
    )
    query_linear = target_query_linear.index_select(
        1,
        query_source_to_target,
    ).contiguous()
    key_linear = target_key_linear.index_select(
        1,
        key_source_to_target,
    ).contiguous()
    runtime = _runtime()
    query_pre = lfm2_runtime_rms_norm(
        query_linear,
        runtime.source_query_norm_weight,
        runtime.norm_epsilon,
    )
    key_pre = lfm2_runtime_rms_norm(
        key_linear,
        runtime.source_key_norm_weight,
        runtime.norm_epsilon,
    )
    key_cos, key_sin = _rope(batch, key_length, start=0)
    query_cos, query_sin = _rope(
        batch,
        query_length,
        start=key_length - query_length,
    )
    query_post = _apply_rope(query_pre, query_cos, query_sin)
    key_post = _apply_rope(key_pre, key_cos, key_sin)
    query_positions = torch.arange(query_length).reshape(1, query_length, 1)
    key_positions = torch.arange(key_length).reshape(1, 1, key_length)
    mask = (
        (key_positions <= key_length - query_length + query_positions)
        .expand(batch, -1, -1)
        .clone()
    )
    query_sqrt = (
        0.65
        + 0.7
        * torch.rand(
            batch,
            query_length,
            generator=generator,
            dtype=torch.float32,
        )
    ).to(DTYPE)
    key_sqrt = (
        0.65
        + 0.7
        * torch.rand(
            batch,
            key_length,
            generator=generator,
            dtype=torch.float32,
        )
    ).to(DTYPE)
    return Lfm2CachedAttentionFoldCase(
        target_query_input=query_input,
        target_key_value_input=key_input,
        parent_query_linear=query_linear,
        parent_key_linear=key_linear,
        parent_value=torch.zeros(
            (batch, SOURCE_KV, key_length, HEAD_DIM),
            dtype=DTYPE,
        ),
        parent_query=query_post,
        parent_key=key_post,
        parent_context=torch.zeros(
            (batch, SOURCE_Q, query_length, HEAD_DIM),
            dtype=DTYPE,
        ),
        parent_post_attention_residual_target_space=query_input.clone(),
        target_pre_attention_residual=query_input.clone(),
        query_cosine=query_cos,
        query_sine=query_sin,
        key_cosine=key_cos,
        key_sine=key_sin,
        key_mask=mask,
        query_row_sqrt_weights=query_sqrt,
        key_value_row_sqrt_weights=key_sqrt,
    )


def _slice_case(
    case: Lfm2CachedAttentionFoldCase,
    batch_slice: slice,
) -> Lfm2CachedAttentionFoldCase:
    values = {}
    for field in fields(case):
        value = getattr(case, field.name)
        values[field.name] = value if value is None else value[batch_slice].clone()
    return Lfm2CachedAttentionFoldCase(**values)


def _banks(
    fit_cases: tuple[Lfm2CachedAttentionFoldCase, ...],
    fit_check_cases: tuple[Lfm2CachedAttentionFoldCase, ...],
    *,
    fit_label: str = "fit",
    fit_check_label: str = "fit-check",
    source_runtime_sha256: str | None = None,
):
    runtime_sha256 = source_runtime_sha256 or _runtime().source_runtime_sha256
    return (
        seal_lfm2_cached_qk_fit_bank(
            fit_cases,
            trajectory_sha256s=tuple(
                _digest(f"{fit_label}-{index}") for index in range(len(fit_cases))
            ),
            source_split_sha256=SOURCE_SPLIT,
            source_runtime_sha256=runtime_sha256,
            source_case_ordinals=tuple(range(len(fit_cases))),
        ),
        seal_lfm2_cached_qk_fit_check_bank(
            fit_check_cases,
            trajectory_sha256s=tuple(
                _digest(f"{fit_check_label}-{index}")
                for index in range(len(fit_check_cases))
            ),
            source_split_sha256=SOURCE_SPLIT,
            source_runtime_sha256=runtime_sha256,
            source_case_ordinals=tuple(
                range(len(fit_cases), len(fit_cases) + len(fit_check_cases))
            ),
        ),
    )


def _gates_off_policy() -> Lfm2CachedQkFunctionalRefitPolicy:
    return Lfm2CachedQkFunctionalRefitPolicy(
        minimum_score_spectral_energy_fraction=0.0,
        maximum_fit_centered_score_relative_rmse=None,
        maximum_fit_check_centered_score_relative_rmse=None,
        maximum_fit_check_js_divergence=None,
        maximum_fit_check_route_probability_relative_rmse=None,
        minimum_fit_check_top1_route_agreement=None,
        anchor_exact_relative_rmse=0.08,
    )


def _compile(
    *,
    fit_cases: tuple[Lfm2CachedAttentionFoldCase, ...] | None = None,
    fit_check_cases: tuple[Lfm2CachedAttentionFoldCase, ...] | None = None,
    candidate_id: str = "bf16-real-qk",
    runtime: Lfm2CpcfAttentionDimensionRuntime | None = None,
):
    fit = fit_cases or tuple(_case(21000 + index) for index in range(12))
    fit_check = fit_check_cases or tuple(
        _case(
            22000 + index,
            query_length=2 if index % 2 == 0 else 1,
            key_length=4 + index % 2,
        )
        for index in range(4)
    )
    fit_bank, fit_check_bank = _banks(fit, fit_check)
    query_groups, key_groups = _groups()
    return compile_lfm2_cached_qk_functional_refit(
        candidate_id=candidate_id,
        topology=_topology(),
        runtime=runtime or _runtime(),
        fit_bank=fit_bank,
        fit_check_bank=fit_check_bank,
        query_groups=query_groups,
        key_value_groups=key_groups,
        policy=_gates_off_policy(),
        ridge=1e-5,
    )


def _compile_primitive():
    runtime = _runtime()
    fit_bank, fit_check_bank = _banks(
        tuple(_case(21500 + index) for index in range(3)),
        (_case(21600),),
    )
    topology = _topology()
    fit_cases = tuple(
        adapt_lfm2_cached_attention_case_for_qk_refit(
            case,
            topology=topology,
            runtime=runtime,
            provenance=Lfm2CpcfAttentionCaseProvenance(
                partition="fit",
                trajectory_sha256=trajectory,
                source_split_sha256=fit_bank.source_split_sha256,
                source_runtime_sha256=fit_bank.source_runtime_sha256,
                case_ordinal=ordinal,
            ),
        )
        for case, trajectory, ordinal in zip(
            fit_bank.cases,
            fit_bank.trajectory_sha256s,
            fit_bank.source_case_ordinals,
            strict=True,
        )
    )
    fit_check_cases = tuple(
        adapt_lfm2_cached_attention_case_for_qk_refit(
            case,
            topology=topology,
            runtime=runtime,
            provenance=Lfm2CpcfAttentionCaseProvenance(
                partition="fit_check",
                trajectory_sha256=trajectory,
                source_split_sha256=fit_check_bank.source_split_sha256,
                source_runtime_sha256=fit_check_bank.source_runtime_sha256,
                case_ordinal=ordinal,
            ),
        )
        for case, trajectory, ordinal in zip(
            fit_check_bank.cases,
            fit_check_bank.trajectory_sha256s,
            fit_check_bank.source_case_ordinals,
            strict=True,
        )
    )
    query_groups, key_groups = _groups()
    return compile_lfm2_cpcf_attention_dimension_fold(
        topology=topology,
        fit_cases=fit_cases,
        fit_check_cases=fit_check_cases,
        validation_cases=(),
        query_groups=query_groups,
        key_value_groups=key_groups,
        runtime=runtime,
        policy=_gates_off_policy().to_primitive(),
        ridge=1e-5,
    )


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def test_real_bf16_qk_runtime_chain_and_exact_published_state() -> None:
    compiled = _compile()

    assert tuple(compiled.query_weight.shape) == (TARGET_Q * HEAD_DIM, HIDDEN)
    assert tuple(compiled.key_weight.shape) == (TARGET_KV * HEAD_DIM, HIDDEN)
    assert compiled.query_weight.dtype == DTYPE
    assert compiled.key_weight.dtype == DTYPE
    assert all(value.dtype == DTYPE for value in compiled.state_dict().values())
    assert compiled.report["query_weight_dtype"] == "torch.bfloat16"
    assert compiled.report["key_weight_dtype"] == "torch.bfloat16"
    assert compiled.report["operation_audit"]["published_state_cast_before_fit_check"]
    assert compiled.report["operation_audit"]["training_steps"] == 0
    assert (
        compiled.report["operation_audit"][
            "target_parameter_mutations_during_fit_check"
        ]
        == 0
    )
    assert compiled.report["fit_partition_contract"]["macrocell_selection_external"]
    assert compiled.report["primitive_solver_contract"][
        "current_mode_is_not_claimed_as_qk_collapse_solution"
    ]
    assert compiled.report["primitive_solver_contract"][
        "factor_sufficient_state_exposed"
    ]
    ladder = {
        value["stage"]: value["status"]
        for value in compiled.report["primitive_solver_error_ladder"]
    }
    assert ladder["cross_case_global_gauge_radius_synchronization"] == (
        "required_not_implemented"
    )
    assert ladder["qk_collapse_solution"] == "not_claimed_by_current_mode"
    assert compiled.report["primitive_factor_sufficient_state_count"] == 16
    assert compiled.report["macrocell_selection_case_count"] == 0
    assert compiled.report["true_validation_case_count"] == 0
    assert compiled.report["cached_resource_usage"]["aggregate_query_rows"] > 0
    assert compiled.report["primitive_capacity"]["accepted"]
    assert "selection_routing" not in compiled.report
    assert "heldout_routing" not in compiled.report
    assert compiled.verify() == compiled.artifact_sha256


def test_future_global_gauge_radius_mode_has_a_stable_fail_closed_seam() -> None:
    runtime = replace(
        _runtime(),
        solver_mode="global_gauge_radius_synchronized",
    )
    with pytest.raises(
        Lfm2CpcfAttentionDimensionFoldError,
        match="reserved.*not implemented",
    ):
        _compile(runtime=runtime)


def test_nonunit_bf16_rope_inverse_preserves_dtype_and_exact_formula() -> None:
    case = _case(23001, query_length=2, key_length=5)
    runtime = _runtime()
    normalized = lfm2_runtime_rms_norm(
        case.parent_query_linear,
        runtime.source_query_norm_weight,
        runtime.norm_epsilon,
    )
    recovered = invert_lfm2_rope(
        case.parent_query,
        case.query_cosine,
        case.query_sine,
    )
    half = HEAD_DIM // 2
    rotate_half = torch.cat(
        (-case.parent_query[..., half:], case.parent_query[..., :half]),
        dim=-1,
    )
    radius = case.query_cosine.square() + case.query_sine.square()
    expected = (
        case.parent_query * case.query_cosine.unsqueeze(1)
        - rotate_half * case.query_sine.unsqueeze(1)
    ) / radius.unsqueeze(1)

    assert recovered.dtype == DTYPE
    assert torch.equal(recovered, expected)
    assert float((radius[..., :half].to(torch.float32) - 1.0).abs().max().item()) > 0.0
    assert torch.allclose(
        recovered.to(torch.float32),
        normalized.to(torch.float32),
        rtol=0.03,
        atol=0.03,
    )


def test_fit_check_rows_cannot_change_candidate_when_gates_are_off() -> None:
    fit = tuple(_case(24000 + index) for index in range(12))
    first = _compile(
        fit_cases=fit,
        fit_check_cases=tuple(_case(24100 + index) for index in range(3)),
        candidate_id="fit-check-a",
    )
    second = _compile(
        fit_cases=fit,
        fit_check_cases=tuple(_case(24200 + index, key_length=5) for index in range(5)),
        candidate_id="fit-check-b",
    )

    assert torch.equal(first.query_weight, second.query_weight)
    assert torch.equal(first.key_weight, second.key_weight)
    assert (
        first.report["primitive_analytical_solvers"][
            "frequency_selection_uses_fit_check"
        ]
        is False
    )
    assert (
        first.report["primitive_analytical_solvers"]["candidate_fit_uses_fit_check"]
        is False
    )


def test_phase_repacking_preserves_exact_bf16_weights() -> None:
    packed = _case(25001, batch=8)
    unpacked = tuple(_slice_case(packed, slice(index, index + 1)) for index in range(8))
    fit_check = tuple(_case(25020 + index) for index in range(2))
    packed_result = _compile(
        fit_cases=(packed,),
        fit_check_cases=fit_check,
        candidate_id="packed",
    )
    unpacked_result = _compile(
        fit_cases=unpacked,
        fit_check_cases=fit_check,
        candidate_id="unpacked",
    )

    assert torch.equal(packed_result.query_weight, unpacked_result.query_weight)
    assert torch.equal(packed_result.key_weight, unpacked_result.key_weight)


def test_repacked_subset_overlap_is_rejected_by_row_provenance() -> None:
    packed = _case(26001, batch=2)
    subset = _slice_case(packed, slice(0, 1))
    trajectory = _digest("same-trajectory")
    fit_bank = seal_lfm2_cached_qk_fit_bank(
        (packed,),
        trajectory_sha256s=(trajectory,),
        source_split_sha256=SOURCE_SPLIT,
        source_runtime_sha256=_runtime().source_runtime_sha256,
        source_case_ordinals=(0,),
    )
    fit_check_bank = seal_lfm2_cached_qk_fit_check_bank(
        (subset,),
        trajectory_sha256s=(trajectory,),
        source_split_sha256=SOURCE_SPLIT,
        source_runtime_sha256=_runtime().source_runtime_sha256,
        source_case_ordinals=(1,),
    )
    query_groups, key_groups = _groups()

    with pytest.raises(
        Lfm2CachedQkFunctionalRefitError,
        match="source trajectories overlap",
    ):
        compile_lfm2_cached_qk_functional_refit(
            candidate_id="overlap",
            topology=_topology(),
            runtime=_runtime(),
            fit_bank=fit_bank,
            fit_check_bank=fit_check_bank,
            query_groups=query_groups,
            key_value_groups=key_groups,
            policy=_gates_off_policy(),
        )


def test_distinct_phases_from_one_trajectory_cannot_cross_fit_boundary() -> None:
    trajectory = _digest("shared-distinct-trajectory")
    runtime = _runtime()
    fit_bank = seal_lfm2_cached_qk_fit_bank(
        (_case(26501),),
        trajectory_sha256s=(trajectory,),
        source_split_sha256=SOURCE_SPLIT,
        source_runtime_sha256=runtime.source_runtime_sha256,
        source_case_ordinals=(0,),
    )
    fit_check_bank = seal_lfm2_cached_qk_fit_check_bank(
        (_case(26502, query_length=2, key_length=5),),
        trajectory_sha256s=(trajectory,),
        source_split_sha256=SOURCE_SPLIT,
        source_runtime_sha256=runtime.source_runtime_sha256,
        source_case_ordinals=(1,),
    )
    query_groups, key_groups = _groups()

    with pytest.raises(
        Lfm2CachedQkFunctionalRefitError,
        match="source trajectories overlap",
    ):
        compile_lfm2_cached_qk_functional_refit(
            candidate_id="shared-distinct-trajectory",
            topology=_topology(),
            runtime=runtime,
            fit_bank=fit_bank,
            fit_check_bank=fit_check_bank,
            query_groups=query_groups,
            key_value_groups=key_groups,
            policy=_gates_off_policy(),
        )


def test_source_norm_runtime_receipt_rejects_one_percent_tamper() -> None:
    runtime = _runtime()
    tampered = replace(
        runtime,
        source_query_norm_weight=(
            runtime.source_query_norm_weight.to(torch.float32) * 1.01
        ).to(DTYPE),
        source_key_norm_weight=(
            runtime.source_key_norm_weight.to(torch.float32) * 1.01
        ).to(DTYPE),
    )

    with pytest.raises(
        Lfm2CachedQkArtifactAuthenticationError,
        match="source RMSNorm runtime receipt",
    ):
        _compile(runtime=tampered)


def test_aggregate_row_ceiling_fails_before_production_cloning() -> None:
    runtime = replace(
        _runtime(),
        maximum_aggregate_query_rows=1,
    )

    with pytest.raises(
        Lfm2CachedQkFunctionalRefitError,
        match="aggregate row or byte ceiling",
    ):
        _compile(runtime=runtime)


def test_primitive_authentication_binds_typed_head_groups() -> None:
    compiled = _compile_primitive()
    original = compiled.query_groups[0]
    changed = Lfm2CpcfBarycentricHeadGroup(
        original.source_indices,
        (0.6, 0.4),
    )

    with pytest.raises(
        Lfm2CpcfAttentionDimensionFoldError,
        match="typed state",
    ):
        replace(
            compiled,
            query_groups=(changed, *compiled.query_groups[1:]),
        )


def test_unequal_qk_input_width_and_tampered_runtime_chain_are_rejected() -> None:
    fit = tuple(_case(27000 + index) for index in range(8))
    fit_check = (_case(27100),)
    unequal = replace(
        fit_check[0],
        target_key_value_input=torch.cat(
            (
                fit_check[0].target_key_value_input,
                torch.zeros(
                    (
                        fit_check[0].target_key_value_input.shape[0],
                        fit_check[0].target_key_value_input.shape[1],
                        1,
                    ),
                    dtype=DTYPE,
                ),
            ),
            dim=-1,
        ),
    )
    with pytest.raises(
        Lfm2CpcfAttentionDimensionFoldError,
        match="equal width",
    ):
        _compile(fit_cases=fit, fit_check_cases=(unequal,))

    tampered = replace(
        fit_check[0],
        parent_query=fit_check[0].parent_query.clone(),
    )
    tampered.parent_query[0, 0, 0, 0] += torch.tensor(0.5, dtype=DTYPE)
    with pytest.raises(
        Lfm2CpcfAttentionDimensionFoldError,
        match="RMSNorm.*RoPE authentication",
    ):
        _compile(fit_cases=fit, fit_check_cases=(tampered,))


def test_resealed_nonzero_training_or_fit_check_mutation_is_semantically_rejected() -> (
    None
):
    clean = _compile()
    payload = clean.as_dict()
    payload["operation_audit"]["training_steps"] = 1
    payload["operation_audit"]["target_parameter_mutations_during_fit_check"] = 1
    body = dict(payload)
    body.pop("artifact_sha256")
    payload["artifact_sha256"] = _canonical_sha256(body)

    with pytest.raises(
        Lfm2CachedQkArtifactAuthenticationError,
        match="no-training",
    ):
        replace(clean, report=payload)


def test_default_production_limits_accept_154_fit_and_52_fit_check_cases() -> None:
    # One-token cases exercise the real bounded compiler path without inflating
    # individual SVD matrices.
    fit = tuple(_case(28000 + index, key_length=2) for index in range(154))
    fit_check = tuple(_case(29000 + index, key_length=2) for index in range(52))
    compiled = _compile(
        fit_cases=fit,
        fit_check_cases=fit_check,
        candidate_id="production-limits",
    )

    assert compiled.report["fit_case_count"] == 154
    assert compiled.report["fit_check_case_count"] == 52
    assert compiled.report["limits"]["maximum_fit_cases"] >= 154
    assert compiled.report["limits"]["maximum_validation_cases"] >= 52


def test_deterministic_upstream_fit_partition_is_134_plus_20_for_h776() -> None:
    cases = tuple(_case(30000 + index, key_length=2) for index in range(154))
    trajectories = tuple(_digest(f"h776-{index}") for index in range(154))
    first_fit, first_check = partition_lfm2_cached_qk_fit_cases(
        cases,
        trajectory_sha256s=trajectories,
        source_split_sha256=SOURCE_SPLIT,
        source_runtime_sha256=_runtime().source_runtime_sha256,
        fit_check_count=20,
    )
    second_fit, second_check = partition_lfm2_cached_qk_fit_cases(
        cases,
        trajectory_sha256s=trajectories,
        source_split_sha256=SOURCE_SPLIT,
        source_runtime_sha256=_runtime().source_runtime_sha256,
        fit_check_count=20,
    )

    assert len(first_fit.cases) == 134
    assert len(first_check.cases) == 20
    assert first_fit.upstream_split == "fit"
    assert first_check.upstream_split == "fit"
    assert first_fit.evidence_sha256 == second_fit.evidence_sha256
    assert first_check.evidence_sha256 == second_check.evidence_sha256
    assert not (
        set(first_fit.source_case_ordinals) & set(first_check.source_case_ordinals)
    )


def test_public_adapter_uses_exact_fit_and_fit_check_provenance() -> None:
    runtime = _runtime()
    case = _case(31001)
    provenance = Lfm2CpcfAttentionCaseProvenance(
        partition="fit",
        trajectory_sha256=_digest("adapter-trajectory"),
        source_split_sha256=SOURCE_SPLIT,
        source_runtime_sha256=runtime.source_runtime_sha256,
        case_ordinal=17,
    )
    adapted = adapt_lfm2_cached_attention_case_for_qk_refit(
        case,
        topology=_topology(),
        runtime=runtime,
        provenance=provenance,
    )

    assert adapted.provenance.upstream_split == "fit"
    assert adapted.provenance.partition == "fit"
    assert adapted.query_rope_cos.dtype == DTYPE
    assert adapted.provenance.source_runtime_sha256 == runtime.source_runtime_sha256
    assert adapted.verify() == adapted.evidence_sha256
