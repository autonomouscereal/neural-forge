from __future__ import annotations

from dataclasses import fields

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_cpcf_attention_dimension_fold import (  # noqa: E402
    Lfm2CpcfAttentionDimensionCase,
    Lfm2CpcfAttentionDimensionPolicy,
    Lfm2CpcfAttentionDimensionTopology,
    Lfm2CpcfAttentionDimensionValidationError,
    Lfm2CpcfBarycentricHeadGroup,
    compile_lfm2_cpcf_attention_dimension_fold,
)


def _rope(
    batch: int,
    length: int,
    head_dim: int,
    *,
    start_position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    positions = torch.arange(
        start_position,
        start_position + length,
        dtype=torch.float64,
    ).reshape(length, 1)
    frequencies = torch.linspace(
        0.017,
        0.371,
        half,
        dtype=torch.float64,
    ).reshape(1, half)
    angles = positions * frequencies
    cosine = torch.cat((torch.cos(angles), torch.cos(angles)), dim=-1)
    sine = torch.cat((torch.sin(angles), torch.sin(angles)), dim=-1)
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
    rotated_half = torch.cat(
        (-value[..., half:], value[..., :half]),
        dim=-1,
    )
    return value * cosine.unsqueeze(1) + rotated_half * sine.unsqueeze(1)


def _center(
    scores: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (scores * mask).sum(dim=-1, keepdim=True) / count
    return torch.where(mask, scores - mean, torch.zeros_like(scores))


def _softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    values = torch.softmax(
        scores.masked_fill(~mask, -torch.inf),
        dim=-1,
        dtype=torch.float64,
    )
    return torch.where(mask, values, torch.zeros_like(values))


def _embed_selected_pairs(
    values: torch.Tensor,
    *,
    source_head_dim: int,
    selected_frequencies: tuple[int, ...],
) -> torch.Tensor:
    target_head_dim = values.shape[-1]
    target_pairs = target_head_dim // 2
    assert len(selected_frequencies) == target_pairs
    result = torch.zeros(
        (*values.shape[:-1], source_head_dim),
        dtype=torch.float64,
    )
    first = torch.tensor(selected_frequencies, dtype=torch.long)
    second = first + source_head_dim // 2
    result[..., first] = values[..., :target_pairs]
    result[..., second] = values[..., target_pairs:]
    return result


def _projection_weights(
    seed: int,
    count: int,
    width: int,
    head_dim: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(
            count,
            width,
            head_dim,
            generator=generator,
            dtype=torch.float64,
        )
        / width**0.5
    )


def _case(
    *,
    topology: Lfm2CpcfAttentionDimensionTopology,
    seed: int,
    query_projection_weights: torch.Tensor,
    key_projection_weights: torch.Tensor,
    source_query_to_prototype: tuple[int, ...],
    source_key_value_to_prototype: tuple[int, ...],
    selected_frequencies: tuple[int, ...],
    batch: int,
    query_length: int,
    key_length: int,
    input_width: int,
) -> Lfm2CpcfAttentionDimensionCase:
    generator = torch.Generator().manual_seed(seed)
    query_inputs = torch.randn(
        batch,
        query_length,
        input_width,
        generator=generator,
        dtype=torch.float64,
    )
    key_inputs = torch.randn(
        batch,
        key_length,
        input_width,
        generator=generator,
        dtype=torch.float64,
    )
    query_prototypes = torch.einsum(
        "bqw,hwd->bhqd",
        query_inputs,
        query_projection_weights,
    )
    key_prototypes = torch.einsum(
        "bkw,hwd->bhkd",
        key_inputs,
        key_projection_weights,
    )
    source_query = _embed_selected_pairs(
        query_prototypes[
            :,
            torch.tensor(source_query_to_prototype, dtype=torch.long),
        ],
        source_head_dim=topology.source_head_dim,
        selected_frequencies=selected_frequencies,
    )
    source_key = _embed_selected_pairs(
        key_prototypes[
            :,
            torch.tensor(
                source_key_value_to_prototype,
                dtype=torch.long,
            ),
        ],
        source_head_dim=topology.source_head_dim,
        selected_frequencies=selected_frequencies,
    )
    key_cos, key_sin = _rope(
        batch,
        key_length,
        topology.source_head_dim,
        start_position=0,
    )
    query_cos, query_sin = _rope(
        batch,
        query_length,
        topology.source_head_dim,
        start_position=key_length - query_length,
    )
    query_post = _apply_rope(source_query, query_cos, query_sin)
    key_post = _apply_rope(source_key, key_cos, key_sin)
    repeated_key = key_post.repeat_interleave(
        topology.source_queries_per_key_value,
        dim=1,
    )
    raw_scores = (
        torch.matmul(query_post, repeated_key.transpose(-2, -1))
        / topology.source_head_dim**0.5
    )
    query_positions = torch.arange(query_length).reshape(1, 1, query_length, 1)
    key_positions = torch.arange(key_length).reshape(1, 1, 1, key_length)
    causal_mask = key_positions <= (key_length - query_length + query_positions)
    causal_mask = causal_mask.expand(batch, 1, query_length, key_length).clone()
    expanded_mask = causal_mask.expand(
        batch,
        topology.source_query_heads,
        query_length,
        key_length,
    )
    centered_scores = _center(raw_scores, expanded_mask)
    probabilities = _softmax(centered_scores, expanded_mask)
    return Lfm2CpcfAttentionDimensionCase(
        target_query_residual_inputs=query_inputs,
        target_key_value_residual_inputs=key_inputs,
        parent_query_pre_rope=source_query,
        parent_key_pre_rope=source_key,
        parent_centered_scores=centered_scores,
        parent_probabilities=probabilities,
        query_rope_cos=query_cos,
        query_rope_sin=query_sin,
        key_rope_cos=key_cos,
        key_rope_sin=key_sin,
        causal_mask=causal_mask,
    )


def _strict_policy() -> Lfm2CpcfAttentionDimensionPolicy:
    return Lfm2CpcfAttentionDimensionPolicy(
        minimum_score_spectral_energy_fraction=0.999999,
        maximum_fit_centered_score_relative_rmse=2e-6,
        maximum_validation_centered_score_relative_rmse=2e-6,
        maximum_validation_js_divergence=1e-10,
        maximum_validation_route_probability_relative_rmse=2e-6,
        minimum_validation_top1_route_agreement=1.0,
    )


def test_exact_representable_d64_to_d32_recovery_and_pair_selection() -> None:
    topology = Lfm2CpcfAttentionDimensionTopology(
        source_query_heads=2,
        source_key_value_heads=1,
        target_query_heads=1,
        target_key_value_heads=1,
        source_head_dim=64,
        target_head_dim=32,
    )
    active = tuple(range(0, 32, 2))
    width = 7
    query_weight = _projection_weights(8101, 1, width, 32)
    key_weight = _projection_weights(8102, 1, width, 32)
    fit = _case(
        topology=topology,
        seed=8103,
        query_projection_weights=query_weight,
        key_projection_weights=key_weight,
        source_query_to_prototype=(0, 0),
        source_key_value_to_prototype=(0,),
        selected_frequencies=active,
        batch=48,
        query_length=1,
        key_length=8,
        input_width=width,
    )
    validation = _case(
        topology=topology,
        seed=8104,
        query_projection_weights=query_weight,
        key_projection_weights=key_weight,
        source_query_to_prototype=(0, 0),
        source_key_value_to_prototype=(0,),
        selected_frequencies=active,
        batch=32,
        query_length=1,
        key_length=8,
        input_width=width,
    )
    result = compile_lfm2_cpcf_attention_dimension_fold(
        topology=topology,
        fit_cases=(fit,),
        validation_cases=(validation,),
        query_groups=(Lfm2CpcfBarycentricHeadGroup((0, 1), (0.5, 0.5)),),
        key_value_groups=(Lfm2CpcfBarycentricHeadGroup((0,), (1.0,)),),
        policy=_strict_policy(),
        ridge=1e-10,
    )

    assert result.selected_source_rope_frequencies == active
    assert tuple(result.query_weight.shape) == (32, width)
    assert tuple(result.key_weight.shape) == (32, width)
    assert result.validation_metrics.centered_score_relative_rmse < 2e-6
    assert result.validation_metrics.top1_route_agreement_fraction == 1.0
    assert (
        result.report["capacity"]["minimum_retained_spectral_energy_fraction"]
        > 0.999999
    )
    assert result.report["operation_audit"]["training_steps"] == 0
    assert result.report["operation_audit"]["source_query_heads_dropped"] == 0


def test_gqa_replay_preserves_causal_mask_and_full_head_barycenters() -> None:
    topology = Lfm2CpcfAttentionDimensionTopology(
        source_query_heads=4,
        source_key_value_heads=2,
        target_query_heads=2,
        target_key_value_heads=1,
        source_head_dim=8,
        target_head_dim=4,
    )
    active = (1, 3)
    width = 5
    query_weight = _projection_weights(8201, 2, width, 4)
    key_weight = _projection_weights(8202, 1, width, 4)
    common = {
        "topology": topology,
        "query_projection_weights": query_weight,
        "key_projection_weights": key_weight,
        "source_query_to_prototype": (0, 0, 1, 1),
        "source_key_value_to_prototype": (0, 0),
        "selected_frequencies": active,
        "batch": 12,
        "query_length": 4,
        "key_length": 4,
        "input_width": width,
    }
    fit = _case(seed=8203, **common)
    validation = _case(seed=8204, **common)
    query_groups = (
        Lfm2CpcfBarycentricHeadGroup((0, 1), (0.4, 0.6)),
        Lfm2CpcfBarycentricHeadGroup((2, 3), (0.7, 0.3)),
    )
    key_groups = (Lfm2CpcfBarycentricHeadGroup((0, 1), (0.25, 0.75)),)
    result = compile_lfm2_cpcf_attention_dimension_fold(
        topology=topology,
        fit_cases=(fit,),
        validation_cases=(validation,),
        query_groups=query_groups,
        key_value_groups=key_groups,
        policy=_strict_policy(),
        ridge=1e-10,
    )
    replay = result.replay(validation)

    assert tuple(replay.probabilities.shape) == (12, 2, 4, 4)
    future = torch.triu(
        torch.ones(4, 4, dtype=torch.bool),
        diagonal=1,
    ).reshape(1, 1, 4, 4)
    assert torch.equal(
        replay.probabilities.masked_select(future.expand(12, 2, 4, 4)),
        torch.zeros(12 * 2 * 6, dtype=torch.float64),
    )
    assert result.report["full_source_assignment_fraction"] == 1.0
    assert result.report["query_groups"][0]["source_indices"] == [0, 1]
    assert result.report["query_groups"][1]["source_indices"] == [2, 3]
    assert result.report["key_value_groups"][0]["source_indices"] == [0, 1]
    assert result.report["operation_audit"]["all_source_key_value_heads_contribute"]


def test_heldout_route_gate_rejects_an_unrepresented_child_mapping() -> None:
    topology = Lfm2CpcfAttentionDimensionTopology(
        source_query_heads=1,
        source_key_value_heads=1,
        target_query_heads=1,
        target_key_value_heads=1,
        source_head_dim=8,
        target_head_dim=4,
    )
    active = (0, 2)
    width = 4
    fit_query_weight = _projection_weights(8301, 1, width, 4)
    key_weight = _projection_weights(8302, 1, width, 4)
    fit = _case(
        topology=topology,
        seed=8303,
        query_projection_weights=fit_query_weight,
        key_projection_weights=key_weight,
        source_query_to_prototype=(0,),
        source_key_value_to_prototype=(0,),
        selected_frequencies=active,
        batch=24,
        query_length=1,
        key_length=7,
        input_width=width,
    )
    validation = _case(
        topology=topology,
        seed=8304,
        query_projection_weights=-fit_query_weight,
        key_projection_weights=key_weight,
        source_query_to_prototype=(0,),
        source_key_value_to_prototype=(0,),
        selected_frequencies=active,
        batch=24,
        query_length=1,
        key_length=7,
        input_width=width,
    )

    with pytest.raises(
        Lfm2CpcfAttentionDimensionValidationError,
        match="held-out routing",
    ) as captured:
        compile_lfm2_cpcf_attention_dimension_fold(
            topology=topology,
            fit_cases=(fit,),
            validation_cases=(validation,),
            query_groups=(Lfm2CpcfBarycentricHeadGroup((0,), (1.0,)),),
            key_value_groups=(Lfm2CpcfBarycentricHeadGroup((0,), (1.0,)),),
            policy=_strict_policy(),
            ridge=1e-10,
        )

    report = captured.value.report
    assert report["accepted"] is False
    assert (
        report["heldout_routing"]["top1_route_agreement_fraction"] < 1.0
        or report["heldout_routing"]["centered_score_relative_rmse"] > 2e-6
    )
    assert any(value is False for value in report["checks"].values())
    assert report["operation_audit"]["source_evidence_unchanged"] is True


def test_compile_does_not_mutate_any_fit_or_validation_tensor() -> None:
    topology = Lfm2CpcfAttentionDimensionTopology(
        source_query_heads=1,
        source_key_value_heads=1,
        target_query_heads=1,
        target_key_value_heads=1,
        source_head_dim=8,
        target_head_dim=4,
    )
    width = 4
    query_weight = _projection_weights(8401, 1, width, 4)
    key_weight = _projection_weights(8402, 1, width, 4)
    common = {
        "topology": topology,
        "query_projection_weights": query_weight,
        "key_projection_weights": key_weight,
        "source_query_to_prototype": (0,),
        "source_key_value_to_prototype": (0,),
        "selected_frequencies": (1, 2),
        "batch": 16,
        "query_length": 1,
        "key_length": 6,
        "input_width": width,
    }
    fit = _case(seed=8403, **common)
    validation = _case(seed=8404, **common)
    snapshots = {
        (split, field.name): getattr(case, field.name).clone()
        for split, case in (("fit", fit), ("validation", validation))
        for field in fields(case)
        if getattr(case, field.name) is not None
    }
    result = compile_lfm2_cpcf_attention_dimension_fold(
        topology=topology,
        fit_cases=(fit,),
        validation_cases=(validation,),
        query_groups=(Lfm2CpcfBarycentricHeadGroup((0,), (1.0,)),),
        key_value_groups=(Lfm2CpcfBarycentricHeadGroup((0,), (1.0,)),),
        policy=_strict_policy(),
        ridge=1e-10,
    )

    for split, case in (("fit", fit), ("validation", validation)):
        for field in fields(case):
            value = getattr(case, field.name)
            if value is not None:
                assert torch.equal(value, snapshots[(split, field.name)])
    assert (
        result.report["source_evidence_sha256_before"]
        == result.report["source_evidence_sha256_after"]
    )
    detached_state = result.state_dict()
    detached_state["q_proj.weight"].zero_()
    assert not torch.equal(
        detached_state["q_proj.weight"],
        result.query_weight,
    )
