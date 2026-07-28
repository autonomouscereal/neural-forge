from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_attention_fold import (  # noqa: E402
    AttentionFoldError,
    build_gqa_attention_fold_maps,
    validate_gqa_attention_fold_geometry,
)


def test_gqa_fold_preserves_heads_rotary_pairs_and_full_coverage() -> None:
    maps = build_gqa_attention_fold_maps(
        source_query_heads=8,
        source_key_value_heads=2,
        source_head_dim=8,
        target_query_heads=4,
        target_key_value_heads=1,
        target_head_dim=4,
    )

    for fold in (maps.query, maps.key, maps.value, maps.context):
        assert fold.report["full_source_coverage"] is True
        assert fold.report["gqa_routing_preserved"] is True
        assert fold.report["rotary_pairs_preserved"] is True
        assert bool((fold.coefficients > 0.0).all())

    source_half = 4
    target_half = 2
    for head in range(8):
        for pair in range(source_half):
            first = int(maps.query.assignment[head * 8 + pair])
            second = int(maps.query.assignment[head * 8 + pair + source_half])
            assert first // 4 == second // 4
            assert second % 4 - first % 4 == target_half

    query_group_sizes = torch.bincount(maps.query.assignment, minlength=16)
    assert bool((query_group_sizes == 4).all())


def test_attention_fold_rejects_expansion_and_invalid_gqa() -> None:
    with pytest.raises(AttentionFoldError, match="expansion"):
        build_gqa_attention_fold_maps(
            source_query_heads=4,
            source_key_value_heads=2,
            source_head_dim=8,
            target_query_heads=8,
            target_key_value_heads=2,
            target_head_dim=8,
        )
    with pytest.raises(AttentionFoldError, match="divide into KV groups"):
        build_gqa_attention_fold_maps(
            source_query_heads=7,
            source_key_value_heads=2,
            source_head_dim=8,
            target_query_heads=4,
            target_key_value_heads=1,
            target_head_dim=4,
        )


def test_gqa_geometry_rejects_unpopulated_target_local_query_heads() -> None:
    with pytest.raises(AttentionFoldError, match="target local-query head"):
        validate_gqa_attention_fold_geometry(
            source_query_heads=4,
            source_key_value_heads=2,
            source_head_dim=16,
            target_query_heads=3,
            target_key_value_heads=1,
            target_head_dim=16,
        )

    validate_gqa_attention_fold_geometry(
        source_query_heads=4,
        source_key_value_heads=2,
        source_head_dim=16,
        target_query_heads=2,
        target_key_value_heads=1,
        target_head_dim=16,
    )
