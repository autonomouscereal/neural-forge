from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (  # noqa: E402
    Lfm2EndpointQkResidualCase,
)
from neural_forge.hf_lfm2_semantic_span_cpcf_alignment import (  # noqa: E402
    Lfm2SemanticSpanCpcfPhaseAlignment,
)
from neural_forge.hf_semantic_attention_lattice import (  # noqa: E402
    LFM2_LEGACY_KEY_LINEAR_SUFFIX,
    LFM2_LEGACY_KEY_POST_ROPE_SUFFIX,
    LFM2_LEGACY_QUERY_LINEAR_SUFFIX,
    LFM2_LEGACY_QUERY_POST_ROPE_SUFFIX,
    LFM2_PRE_ROPE_KEY_SUFFIX,
    LFM2_PRE_ROPE_QUERY_SUFFIX,
    LFM2_VALUE_HISTORY_SUFFIX,
    LFM2_VALUE_SUFFIX,
    SemanticAttentionLatticeError,
    SemanticAttentionPreRopeHistory,
    assemble_lfm2_explicit_pre_rope_history,
    fold_semantic_attention_parent_pre_rope_priors,
    inspect_lfm2_semantic_attention_cache_capability,
    project_rephased_semantic_attention_lattice,
    reconstruct_lfm2_normalized_pre_rope_rows,
    transport_lfm2_phase_probability_preserving_attention,
    transport_probability_preserving_attention_expansion,
)


def _name(phase: str, layer: int, suffix: str) -> str:
    return f"{phase}.layer.{layer:03d}.{suffix}"


def _explicit_cache() -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    phases = ("prefill", "decode.000")
    generator = torch.Generator().manual_seed(13)
    tensors: dict[str, torch.Tensor] = {}
    value_parts = []
    for phase, tokens in (("prefill", 3), ("decode.000", 1)):
        tensors[_name(phase, 2, LFM2_PRE_ROPE_QUERY_SUFFIX)] = torch.randn(
            1,
            4,
            tokens,
            2,
            generator=generator,
        )
        tensors[_name(phase, 2, LFM2_PRE_ROPE_KEY_SUFFIX)] = torch.randn(
            1,
            2,
            tokens,
            2,
            generator=generator,
        )
        value = torch.randn(1, 2, tokens, 2, generator=generator)
        tensors[_name(phase, 2, LFM2_VALUE_SUFFIX)] = value
        value_parts.append(value)
    tensors[_name("decode.000", 2, LFM2_VALUE_HISTORY_SUFFIX)] = torch.cat(
        value_parts,
        dim=2,
    )
    return tensors, phases


def _legacy_cache(
    *,
    prompt_capture: int,
    decode_steps: int,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    phases = (
        "prefill",
        *(f"decode.{index:03d}" for index in range(decode_steps)),
    )
    tensors = {}
    for ordinal, phase in enumerate(phases):
        tokens = prompt_capture if ordinal == 0 else 1
        for suffix, heads in (
            (LFM2_LEGACY_QUERY_LINEAR_SUFFIX, 4),
            (LFM2_LEGACY_QUERY_POST_ROPE_SUFFIX, 4),
            (LFM2_LEGACY_KEY_LINEAR_SUFFIX, 2),
            (LFM2_LEGACY_KEY_POST_ROPE_SUFFIX, 2),
            (LFM2_VALUE_SUFFIX, 2),
        ):
            tensors[_name(phase, 2, suffix)] = torch.ones(1, heads, tokens, 2)
    return tensors, phases


def _history(*, donor_tokens: int = 3) -> SemanticAttentionPreRopeHistory:
    generator = torch.Generator().manual_seed(29)
    return SemanticAttentionPreRopeHistory(
        query_normalized_pre_rope=torch.randn(
            1,
            4,
            donor_tokens,
            2,
            generator=generator,
        ),
        key_normalized_pre_rope=torch.randn(
            1,
            2,
            donor_tokens,
            2,
            generator=generator,
        ),
        value=torch.randn(
            1,
            2,
            donor_tokens,
            2,
            generator=generator,
        ),
        source_capability_sha256="a" * 64,
        source_tensor_names=("q", "k", "v"),
    )


def _parent_attention(
    donor_tokens: int = 2,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    query = torch.randn(1, 2, donor_tokens, 2, generator=generator).to(dtype)
    key = torch.randn(1, 1, donor_tokens, 2, generator=generator).to(dtype)
    value = torch.randn(1, 1, donor_tokens, 2, generator=generator).to(dtype)
    return query, key, value


def _source_probabilities_and_context(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    scores = query.to(torch.float64) @ key.transpose(2, 3).to(torch.float64)
    scores = scores / math.sqrt(query.shape[-1])
    tokens = query.shape[2]
    mask = (
        torch.ones(tokens, tokens, dtype=torch.bool)
        .tril()
        .reshape(
            1,
            1,
            tokens,
            tokens,
        )
    )
    probabilities = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
    return probabilities, probabilities @ value.to(torch.float64)


def test_broad_enhanced_cache_has_exact_explicit_pre_rope_route() -> None:
    tensors, phases = _explicit_cache()
    capability = inspect_lfm2_semantic_attention_cache_capability(
        tensors,
        phase_order=phases,
        layer_index=2,
        donor_prompt_token_count=3,
    )

    assert capability.selected_route == "explicit_pre_rope"
    assert capability.exact_history_supported is True
    assert capability.missing_donor_prefix_positions == ()
    assert capability.blocking_reasons == ()
    assert (
        capability.to_evidence_dict()["rolling_key_value_cache_is_complete_history"]
        is False
    )


def test_legacy_long_horizon_truncated_prefill_fails_closed_by_row() -> None:
    tensors, phases = _legacy_cache(prompt_capture=8, decode_steps=58)
    capability = inspect_lfm2_semantic_attention_cache_capability(
        tensors,
        phase_order=phases,
        layer_index=2,
        donor_prompt_token_count=18,
    )

    assert capability.selected_route == "legacy_linear_reconstruction"
    assert capability.exact_history_supported is False
    assert capability.prefill_captured_token_count == 8
    assert capability.assembled_current_row_count == 66
    assert capability.missing_donor_prefix_positions == tuple(range(10))
    assert "truncated_prefill_evidence" in capability.blocking_reasons
    assert (
        "model.layers.2.self_attn.q_layernorm.weight"
        in capability.required_external_legacy_evidence
    )
    with pytest.raises(
        SemanticAttentionLatticeError,
        match="missing_prefix_rows",
    ):
        capability.require_exact_history()


def test_explicit_history_assembles_phase_rows_and_authenticates_full_v() -> None:
    tensors, phases = _explicit_cache()
    history = assemble_lfm2_explicit_pre_rope_history(
        tensors,
        phase_order=phases,
        layer_index=2,
        donor_prompt_token_count=3,
    )

    assert history.token_count == 4
    assert torch.equal(
        history.value,
        tensors[_name("decode.000", 2, LFM2_VALUE_HISTORY_SUFFIX)],
    )
    assert history.to_evidence_dict()["model_operations"]["model_forward_calls"] == 0

    tampered = dict(tensors)
    tampered[_name("decode.000", 2, LFM2_VALUE_HISTORY_SUFFIX)] = tampered[
        _name("decode.000", 2, LFM2_VALUE_HISTORY_SUFFIX)
    ].clone()
    tampered[_name("decode.000", 2, LFM2_VALUE_HISTORY_SUFFIX)][0, 0, 0, 0] += 1
    with pytest.raises(SemanticAttentionLatticeError, match="sealed full V history"):
        assemble_lfm2_explicit_pre_rope_history(
            tampered,
            phase_order=phases,
            layer_index=2,
            donor_prompt_token_count=3,
        )


def test_legacy_linear_qk_norm_reconstruction_matches_lfm2_finite_precision() -> None:
    query = torch.tensor(
        [[[[1.0, -2.0]], [[0.5, 4.0]]]],
        dtype=torch.bfloat16,
    )
    key = torch.tensor([[[[3.0, -1.0]]]], dtype=torch.bfloat16)
    q_weight = torch.tensor([0.75, 1.25], dtype=torch.bfloat16)
    k_weight = torch.tensor([1.5, 0.5], dtype=torch.bfloat16)
    q_norm, k_norm = reconstruct_lfm2_normalized_pre_rope_rows(
        query,
        key,
        query_norm_weight=q_weight,
        key_norm_weight=k_weight,
        norm_epsilon=1e-5,
    )

    def expected(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        direction = value.float()
        direction *= torch.rsqrt(direction.square().mean(-1, keepdim=True) + 1e-5)
        return weight * direction.to(torch.bfloat16)

    assert torch.equal(q_norm, expected(query, q_weight))
    assert torch.equal(k_norm, expected(key, k_weight))
    assert q_norm.requires_grad is False


def test_common_endpoint_projection_rephases_and_recomputes_causality() -> None:
    history = _history()
    projection = project_rephased_semantic_attention_lattice(
        history,
        semantic_alignment_sha256="b" * 64,
        target_to_donor_boundary_positions=(0, 2),
        target_position_ids=torch.tensor([[4, 7]]),
        target_rope_inv_freq=torch.tensor([0.5]),
    )

    assert projection.donor_gather_indices == (0, 2)
    assert projection.query.shape == (1, 4, 2, 2)
    assert projection.key.shape == (1, 2, 2, 2)
    assert projection.causal_mask.tolist() == [[[[True, False], [True, True]]]]
    assert torch.allclose(
        projection.probabilities.sum(dim=-1),
        torch.ones(1, 4, 2, dtype=torch.float64),
    )
    receipt = projection.to_evidence_dict()
    assert receipt["semantic_equal_endpoints_only"] is True
    assert receipt["donor_context_preservation_claimed"] is False
    assert receipt["model_operations"]["training_steps"] == 0


def test_common_endpoint_projection_rejects_inserted_target_boundary() -> None:
    with pytest.raises(
        SemanticAttentionLatticeError,
        match="without semantically equal donor endpoints",
    ):
        project_rephased_semantic_attention_lattice(
            _history(donor_tokens=2),
            semantic_alignment_sha256="c" * 64,
            target_to_donor_boundary_positions=(None, 0, 1),
            target_position_ids=torch.tensor([[0, 1, 2]]),
            target_rope_inv_freq=torch.tensor([1.0]),
        )


def test_split_token_transport_preserves_group_mass_and_context_at_endpoints() -> None:
    query, key, value = _parent_attention()
    transported = transport_probability_preserving_attention_expansion(
        semantic_alignment_sha256="d" * 64,
        target_to_donor_boundary_positions=(None, 0, None, None, 1),
        target_to_donor_span_owner_positions=(0, 0, 1, 1, 1),
        donor_span_multiplicities=(2, 3),
        parent_query_post_rope=query,
        parent_key_post_rope=key,
        parent_value=value,
        target_position_ids=torch.arange(10, 15).reshape(1, -1),
        target_rope_inv_freq=torch.tensor([0.25]),
    )
    source_probabilities, source_context = _source_probabilities_and_context(
        query,
        key,
        value,
    )

    assert transported.query_target_positions == (1, 4)
    assert transported.query_donor_positions == (0, 1)
    assert torch.allclose(
        transported.probabilities[..., :2].sum(-1),
        source_probabilities[..., 0],
        rtol=0.0,
        atol=1e-12,
    )
    assert torch.allclose(
        transported.probabilities[..., 2:].sum(-1),
        source_probabilities[..., 1],
        rtol=0.0,
        atol=1e-12,
    )
    assert torch.allclose(
        transported.context,
        source_context,
        rtol=0.0,
        atol=1e-12,
    )
    assert transported.maximum_probability_mass_error <= 1e-12
    assert transported.maximum_context_absolute_error <= 1e-12
    assert transported.inserted_query_rows_are_semantically_exact is True


def test_parent_pre_rope_priors_follow_head_fold_and_target_owner_map() -> None:
    history = _history(donor_tokens=2)
    transported = transport_probability_preserving_attention_expansion(
        semantic_alignment_sha256="4" * 64,
        target_to_donor_boundary_positions=(None, 0, None, None, 1),
        target_to_donor_span_owner_positions=(0, 0, 1, 1, 1),
        donor_span_multiplicities=(2, 3),
        parent_query_post_rope=history.query_normalized_pre_rope,
        parent_key_post_rope=history.key_normalized_pre_rope,
        parent_value=history.value,
        target_position_ids=torch.arange(5).reshape(1, -1),
        target_rope_inv_freq=torch.tensor([0.25]),
    )

    query_prior, key_prior, receipt = fold_semantic_attention_parent_pre_rope_priors(
        history,
        transported,
        query_groups=((0, 1), (2, 3)),
        query_group_weights=((0.25, 0.75), (0.6, 0.4)),
        key_value_groups=((0, 1),),
        key_value_group_weights=((0.2, 0.8),),
    )

    expected_query_zero = (
        history.query_normalized_pre_rope[:, 0] * 0.25
        + history.query_normalized_pre_rope[:, 1] * 0.75
    )
    expected_query_one = (
        history.query_normalized_pre_rope[:, 2] * 0.6
        + history.query_normalized_pre_rope[:, 3] * 0.4
    )
    owner = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    torch.testing.assert_close(query_prior[:, 0], expected_query_zero.to(torch.float64))
    torch.testing.assert_close(query_prior[:, 1], expected_query_one.to(torch.float64))
    expected_key = (
        history.key_normalized_pre_rope[:, 0] * 0.2
        + history.key_normalized_pre_rope[:, 1] * 0.8
    )
    assert key_prior.shape == (1, 1, 5, 2)
    torch.testing.assert_close(
        key_prior[:, 0],
        expected_key.index_select(1, owner).to(torch.float64),
    )
    assert receipt["parent_activation_evidence_only"] is True
    assert receipt["owner_mapped_key_prior"] is True
    assert receipt["target_query_head_count"] == 2
    assert receipt["target_key_value_head_count"] == 1
    assert receipt["key_value_groups"] == [[0, 1]]
    assert receipt["key_value_group_weights"] == [[0.2, 0.8]]
    assert receipt["query_and_kv_lineage_explicit"] is True
    assert receipt["key_prior_indexed_by_target_kv_head"] is True
    assert len(receipt["head_fold_lineage_sha256"]) == 64
    assert receipt["model_operations"]["training_steps"] == 0


@pytest.mark.parametrize(
    ("key_value_groups", "key_value_group_weights"),
    (
        (((0,),), ((1.0,),)),
        (((0, 1),), ((0.4, 0.4),)),
        (((0, 0), (1,)), ((0.5, 0.5), (1.0,))),
    ),
)
def test_parent_pre_rope_prior_rejects_unsealed_kv_lineage(
    key_value_groups: tuple[tuple[int, ...], ...],
    key_value_group_weights: tuple[tuple[float, ...], ...],
) -> None:
    history = _history(donor_tokens=2)
    transported = transport_probability_preserving_attention_expansion(
        semantic_alignment_sha256="7" * 64,
        target_to_donor_boundary_positions=(0, 1),
        target_to_donor_span_owner_positions=(0, 1),
        donor_span_multiplicities=(1, 1),
        parent_query_post_rope=history.query_normalized_pre_rope,
        parent_key_post_rope=history.key_normalized_pre_rope,
        parent_value=history.value,
        target_position_ids=torch.arange(2).reshape(1, -1),
        target_rope_inv_freq=torch.tensor([0.25]),
    )

    with pytest.raises(
        SemanticAttentionLatticeError,
        match="KV head-fold lineage is not an exhaustive weighted partition",
    ):
        fold_semantic_attention_parent_pre_rope_priors(
            history,
            transported,
            query_groups=((0, 1), (2, 3)),
            query_group_weights=((0.5, 0.5), (0.5, 0.5)),
            key_value_groups=key_value_groups,
            key_value_group_weights=key_value_group_weights,
        )


def test_full_surrogate_uses_visible_copy_count_at_inserted_queries() -> None:
    query, key, value = _parent_attention()
    transported = transport_probability_preserving_attention_expansion(
        semantic_alignment_sha256="e" * 64,
        target_to_donor_boundary_positions=(None, 0, None, None, 1),
        target_to_donor_span_owner_positions=(0, 0, 1, 1, 1),
        donor_span_multiplicities=(2, 3),
        parent_query_post_rope=query,
        parent_key_post_rope=key,
        parent_value=value,
        target_position_ids=torch.arange(5).reshape(1, -1),
        target_rope_inv_freq=torch.tensor([0.25]),
        query_mode="owner_surrogate_full_lattice",
    )
    source_probabilities, source_context = _source_probabilities_and_context(
        query,
        key,
        value,
    )
    owners = torch.tensor([0, 0, 1, 1, 1])

    assert transported.query_target_positions == (0, 1, 2, 3, 4)
    assert torch.allclose(
        transported.context,
        source_context.index_select(2, owners),
        rtol=0.0,
        atol=1e-12,
    )
    assert torch.allclose(
        transported.probabilities[:, :, 0, 0],
        source_probabilities[:, :, 0, 0],
        rtol=0.0,
        atol=1e-12,
    )
    assert transported.inserted_query_rows_are_semantically_exact is False
    assert transported.direct_endpoint_case_suffix_compatible is True


def test_probability_preserving_transport_rejects_contraction() -> None:
    query, key, value = _parent_attention(donor_tokens=3)
    with pytest.raises(SemanticAttentionLatticeError, match="contraction"):
        transport_probability_preserving_attention_expansion(
            semantic_alignment_sha256="f" * 64,
            target_to_donor_boundary_positions=(0, 2),
            target_to_donor_span_owner_positions=(0, 2),
            donor_span_multiplicities=(1, 0, 1),
            parent_query_post_rope=query,
            parent_key_post_rope=key,
            parent_value=value,
            target_position_ids=torch.tensor([[0, 1]]),
            target_rope_inv_freq=torch.tensor([1.0]),
        )


def test_expansion_evidence_is_directly_accepted_by_endpoint_qk_case() -> None:
    query, key, value = _parent_attention(donor_tokens=1, dtype=torch.bfloat16)
    transported = transport_probability_preserving_attention_expansion(
        semantic_alignment_sha256="1" * 64,
        target_to_donor_boundary_positions=(None, 0),
        target_to_donor_span_owner_positions=(0, 0),
        donor_span_multiplicities=(2,),
        parent_query_post_rope=query,
        parent_key_post_rope=key,
        parent_value=value,
        target_position_ids=torch.tensor([[0, 1]]),
        target_rope_inv_freq=torch.tensor([1.0]),
    )
    evidence = transported.endpoint_residual_evidence()
    case = Lfm2EndpointQkResidualCase(
        target_query_inputs=torch.ones(1, 1, 4, dtype=torch.bfloat16),
        target_key_value_inputs=torch.ones(1, 2, 4, dtype=torch.bfloat16),
        parent_probabilities=evidence["parent_probabilities"],
        parent_causal_mask=evidence["parent_causal_mask"],
        parent_endpoint_target=torch.ones(1, 1, 4, dtype=torch.bfloat16),
        query_rope_cos=evidence["query_rope_cos"],
        query_rope_sin=evidence["query_rope_sin"],
        key_rope_cos=evidence["key_rope_cos"],
        key_rope_sin=evidence["key_rope_sin"],
        query_row_sqrt_weights=torch.ones(1, 1),
        query_groups=((0,), (1,)),
        query_group_weights=((1.0,), (1.0,)),
        trajectory_sha256="2" * 64,
        parent_probability_provenance=evidence["parent_probability_provenance"],
    )

    assert case.parent_probabilities.shape == (1, 2, 1, 2)
    assert transported.direct_endpoint_case_suffix_compatible is True
    assert case.target_receipt["parent_probability_provenance"] == (
        "explicit_parent_probability_tensor"
    )


def test_lfm2_wrapper_binds_the_authenticated_alignment_and_positions() -> None:
    alignment = Lfm2SemanticSpanCpcfPhaseAlignment(
        phase="prefill",
        semantic_phase_sha256="3" * 64,
        donor_full_prefix_token_ids=(10, 11),
        target_full_prefix_token_ids=(20, 21, 22),
        target_position_origin=7,
        target_to_donor_boundary_positions=(0, None, 1),
        target_to_donor_span_owner_positions=(0, 1, 1),
        parent_sampled_suffix_tokens=2,
        parent_history_suffix_tokens=2,
        target_rewrite_start=0,
        target_replaced_suffix_tokens=0,
        target_replacement_token_ids=(20, 21, 22),
    )
    query, key, value = _parent_attention()
    transported = transport_lfm2_phase_probability_preserving_attention(
        alignment,
        parent_query_post_rope=query,
        parent_key_post_rope=key,
        parent_value=value,
        target_rope_inv_freq=torch.tensor([1.0]),
    )

    assert transported.semantic_alignment_sha256 == alignment.alignment_sha256
    assert transported.target_position_ids.tolist() == [[7, 8, 9]]
    assert transported.donor_span_multiplicities == (1, 2)
