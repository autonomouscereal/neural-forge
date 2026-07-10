from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_canonical_kernel import (  # noqa: E402
    CanonicalDepthCover,
    CanonicalKernelError,
    CanonicalParentTrace,
    CanonicalRowIdentity,
    apply_parent_map,
    build_canonical_boundary_splits,
    build_canonical_kernel_plan,
    build_proportional_depth_cover,
    canonical_parent_trace_from_cache,
    fuse_parent_rows,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(
    rows: int,
    *,
    split: str = "calibration",
    pooling: str = "last_valid",
    tokenizer: str = "shared-tokenizer",
    cache: str = "cache",
) -> CanonicalRowIdentity:
    row_ids = tuple(f"{split}-row-{index:04d}" for index in range(rows))
    prompt_hashes = tuple(
        _digest(f"{split}-raw-prompt-{index:04d}") for index in range(rows)
    )
    return CanonicalRowIdentity(
        split_name=split,
        pooling=pooling,
        row_ids=row_ids,
        raw_prompt_sha256=prompt_hashes,
        corpus_sha256=_digest(f"{split}-corpus"),
        tokenizer_fingerprint=_digest(tokenizer),
        cache_fingerprint=_digest(cache),
    )


def _parent(
    name: str,
    states,
    *,
    target_layers: int,
    rows: int,
    split: str = "calibration",
    weight: float = 1.0,
) -> CanonicalParentTrace:
    return CanonicalParentTrace(
        name,
        states,
        row_identity=_identity(
            rows,
            split=split,
            tokenizer=f"{name}-tokenizer",
            cache=f"{name}-{split}-cache",
        ),
        depth_cover=build_proportional_depth_cover(
            len(states) - 1,
            target_layers,
        ),
        weight=weight,
    )


def _trajectory(seed: int, rows: int, latent: int, width: int, depth: int):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(rows, latent, generator=generator, dtype=torch.float64)
    projection = torch.randn(latent, width, generator=generator, dtype=torch.float64)
    states = []
    for layer in range(depth + 1):
        update = torch.sin(base * (0.15 + 0.03 * layer))
        states.append((base + layer * 0.2 * update).matmul(projection))
    return tuple(states)


def test_cross_width_and_depth_parents_compile_to_one_deterministic_gauge() -> None:
    latent_states = _trajectory(11, rows=32, latent=4, width=4, depth=6)
    generator = torch.Generator().manual_seed(12)
    parent_a_projection = torch.randn(4, 7, generator=generator, dtype=torch.float64)
    parent_b_projection = torch.randn(4, 11, generator=generator, dtype=torch.float64)
    parent_a = CanonicalParentTrace(
        "transformer-a",
        tuple(state.matmul(parent_a_projection) for state in latent_states[::2]),
        row_identity=_identity(32, tokenizer="a", cache="a-cache"),
        depth_cover=CanonicalDepthCover((0, 1, 2, 3)),
        weight=0.7,
    )
    parent_b = CanonicalParentTrace(
        "hybrid-b",
        tuple(state.matmul(parent_b_projection) for state in latent_states),
        row_identity=_identity(32, tokenizer="b", cache="b-cache"),
        depth_cover=CanonicalDepthCover((0, 2, 4, 6)),
        weight=0.3,
    )

    first = build_canonical_kernel_plan(
        (parent_a, parent_b),
        target_layer_count=3,
        target_hidden_size=6,
        ridge=1e-8,
    )
    second = build_canonical_kernel_plan(
        (parent_a, parent_b),
        target_layer_count=3,
        target_hidden_size=6,
        ridge=1e-8,
    )
    reversed_order = build_canonical_kernel_plan(
        (parent_b, parent_a),
        target_layer_count=3,
        target_hidden_size=6,
        ridge=1e-8,
    )

    assert first.fingerprint == second.fingerprint
    assert first.fingerprint == reversed_order.fingerprint
    for left, right in zip(first.boundary_rows, reversed_order.boundary_rows, strict=True):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)
    assert first.parent_names == ("hybrid-b", "transformer-a")
    assert first.normalized_parent_weights == pytest.approx((0.3, 0.7))
    assert len(first.layers) == 4
    assert [mapping.source_layer_index for mapping in first.layers[1].parent_maps] == [
        2,
        1,
    ]
    for layer in first.layers:
        assert layer.canonical_rows.shape == (32, 6)
        assert layer.effective_rank == 4
        assert layer.retained_kernel_fraction == pytest.approx(1.0)
        assert torch.isfinite(layer.canonical_rows).all()
        parents_by_name = {parent.name: parent for parent in (parent_a, parent_b)}
        for mapping in layer.parent_maps:
            source = parents_by_name[mapping.parent_name].layer_states[
                mapping.source_layer_index
            ]
            encoded = apply_parent_map(source, mapping)
            assert mapping.relative_rmse < 1e-5
            assert mapping.mean_cosine_similarity > 0.999
            torch.testing.assert_close(
                encoded,
                layer.canonical_rows,
                atol=2e-4,
                rtol=2e-4,
            )
        source_rows = tuple(
            parents_by_name[mapping.parent_name].layer_states[
                mapping.source_layer_index
            ]
            for mapping in layer.parent_maps
        )
        fused = fuse_parent_rows(
            source_rows,
            layer,
            first.normalized_parent_weights,
        )
        torch.testing.assert_close(
            fused,
            layer.canonical_rows,
            atol=2e-4,
            rtol=2e-4,
        )
    assert all(
        layer.gauge_alignment_cosine is None
        or -1.0 <= layer.gauge_alignment_cosine <= 1.0
        for layer in first.layers
    )
    report = first.report()
    assert report["architecture_neutral_parent_interface"] == "output_hidden_states"
    assert report["training_steps"] == 0
    assert report["gradients_used"] is False
    assert report["row_alignment_fingerprint"] == (
        parent_a.row_identity.alignment_fingerprint
    )
    assert all(
        item["source_layers_covered_exactly_once"] is True
        for item in report["source_depth_covers"]
    )
    assert report["algebraically_exact"] is False
    assert report["exactness_class"] in {
        "finite-trace-exact",
        "finite-trace-approximate",
    }
    assert all(
        layer["algebraically_exact"] is False for layer in report["layers"]
    )
    assert all(
        mapping["exactness_scope"] == "fingerprinted calibration rows only"
        for layer in report["layers"]
        for mapping in layer["parent_maps"]
    )


def test_parent_weight_changes_the_canonical_row_kernel() -> None:
    first_states = _trajectory(21, rows=24, latent=3, width=6, depth=2)
    second_states = _trajectory(22, rows=24, latent=3, width=8, depth=2)
    first = _parent(
        "first", first_states, target_layers=2, rows=24, weight=0.99
    )
    second = _parent(
        "second", second_states, target_layers=2, rows=24, weight=0.01
    )
    plan = build_canonical_kernel_plan(
        (first, second),
        target_layer_count=2,
        target_hidden_size=5,
    )
    reversed_plan = build_canonical_kernel_plan(
        (
            _parent("first", first_states, target_layers=2, rows=24, weight=0.01),
            _parent("second", second_states, target_layers=2, rows=24, weight=0.99),
        ),
        target_layer_count=2,
        target_hidden_size=5,
    )

    left_kernel = plan.layers[-1].canonical_rows.matmul(
        plan.layers[-1].canonical_rows.T
    )
    right_kernel = reversed_plan.layers[-1].canonical_rows.matmul(
        reversed_plan.layers[-1].canonical_rows.T
    )
    assert not torch.allclose(left_kernel, right_kernel)
    assert plan.fingerprint != reversed_plan.fingerprint


def test_rank_truncation_is_reported_as_approximate_not_algebraically_exact() -> None:
    states = _trajectory(23, rows=24, latent=6, width=8, depth=2)
    plan = build_canonical_kernel_plan(
        (_parent("wide", states, target_layers=2, rows=24),),
        target_layer_count=2,
        target_hidden_size=2,
    )

    assert any(layer.retained_kernel_fraction < 1.0 for layer in plan.layers)
    assert plan.exactness_class == "finite-trace-approximate"
    report = plan.report()
    assert report["exactness_class"] == "finite-trace-approximate"
    assert report["algebraically_exact"] is False


def test_validation_boundaries_reuse_calibration_parent_maps() -> None:
    calibration_latent = _trajectory(31, rows=32, latent=4, width=4, depth=6)
    validation_latent = _trajectory(32, rows=12, latent=4, width=4, depth=6)
    generator = torch.Generator().manual_seed(33)
    projection_a = torch.randn(4, 7, generator=generator, dtype=torch.float64)
    projection_b = torch.randn(4, 9, generator=generator, dtype=torch.float64)

    def parents(states, split):
        return (
            CanonicalParentTrace(
                "a",
                tuple(state.matmul(projection_a) for state in states[::2]),
                row_identity=_identity(
                    int(states[0].shape[0]),
                    split=split,
                    tokenizer="a-tokenizer",
                    cache=f"a-{split}-cache",
                ),
                depth_cover=CanonicalDepthCover((0, 1, 2, 3)),
                weight=0.6,
            ),
            CanonicalParentTrace(
                "b",
                tuple(state.matmul(projection_b) for state in states),
                row_identity=_identity(
                    int(states[0].shape[0]),
                    split=split,
                    tokenizer="b-tokenizer",
                    cache=f"b-{split}-cache",
                ),
                depth_cover=CanonicalDepthCover((0, 2, 4, 6)),
                weight=0.4,
            ),
        )

    splits = build_canonical_boundary_splits(
        parents(calibration_latent, "calibration"),
        parents(validation_latent, "validation"),
        target_layer_count=3,
        target_hidden_size=6,
        ridge=1e-8,
    )

    assert len(splits.calibration_rows) == len(splits.validation_rows) == 4
    assert all(rows.shape == (12, 6) for rows in splits.validation_rows)
    assert all(torch.isfinite(rows).all() for rows in splits.validation_rows)
    assert len(splits.fingerprint) == 64
    report = splits.report()
    assert report["validation_boundary_count"] == 4
    assert report["validation_row_count"] == 12
    assert report["training_steps"] == 0
    assert report["exactness_class"] == "finite-trace-approximate"
    assert report["algebraically_exact"] is False


def test_depth_cover_supports_expansion_with_explicit_identity_spans() -> None:
    cover = build_proportional_depth_cover(2, 5)

    assert cover.boundary_indices == (0, 0, 1, 1, 2, 2)
    assert cover.source_spans == ((), (0,), (), (1,), ())
    assert cover.report()["source_layers_covered_exactly_once"] is True
    assert cover.report()["empty_identity_spans"] == 3


def test_equal_row_counts_with_different_prompt_identity_fail_closed() -> None:
    states = _trajectory(41, rows=8, latent=3, width=4, depth=1)
    first = _parent("a", states, target_layers=1, rows=8)
    second = CanonicalParentTrace(
        "b",
        states,
        row_identity=_identity(
            8,
            split="different-split",
            tokenizer="b-tokenizer",
            cache="b-cache",
        ),
        depth_cover=CanonicalDepthCover((0, 1)),
    )

    with pytest.raises(CanonicalKernelError, match="same ordered raw-prompt rows"):
        build_canonical_kernel_plan(
            (first, second),
            target_layer_count=1,
            target_hidden_size=3,
        )


def test_parent_trace_cache_adapter_pools_prompt_rows_without_parent_forwards() -> None:
    batches = []
    for offset in (0.0, 100.0):
        hidden = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) + offset
        batches.append(
            SimpleNamespace(
                valid_token_mask=torch.tensor(
                    [[1, 1, 0], [1, 1, 1]],
                    dtype=torch.bool,
                ),
                hidden_states={0: hidden, 2: hidden + 10.0},
            )
        )
    cache = SimpleNamespace(iter_batches=lambda: iter(batches))

    last = canonical_parent_trace_from_cache(
        cache,
        name="cached-parent",
        row_identity=_identity(4, pooling="last_valid"),
        depth_cover=CanonicalDepthCover((0, 1)),
        pooling="last_valid",
    )
    mean = canonical_parent_trace_from_cache(
        cache,
        name="cached-parent",
        row_identity=_identity(4, pooling="mean"),
        depth_cover=CanonicalDepthCover((0, 1)),
        pooling="mean",
    )

    assert len(last.layer_states) == 2
    assert last.layer_states[0].shape == (4, 4)
    torch.testing.assert_close(
        last.layer_states[0][0], batches[0].hidden_states[0][0, 1]
    )
    torch.testing.assert_close(
        last.layer_states[0][1], batches[0].hidden_states[0][1, 2]
    )
    torch.testing.assert_close(
        mean.layer_states[0][0],
        batches[0].hidden_states[0][0, :2].mean(dim=0),
    )
    assert last.weight == 1.0


def test_canonical_kernel_fails_closed_on_bad_parent_rows() -> None:
    valid = CanonicalParentTrace(
        "valid",
        (
            torch.ones(4, 3, dtype=torch.float64),
            torch.arange(12, dtype=torch.float64).reshape(4, 3),
        ),
        row_identity=_identity(4, tokenizer="valid", cache="valid"),
        depth_cover=CanonicalDepthCover((0, 1)),
    )
    mismatched = CanonicalParentTrace(
        "mismatched",
        (
            torch.ones(5, 2, dtype=torch.float64),
            torch.ones(5, 2, dtype=torch.float64),
        ),
        row_identity=_identity(5, tokenizer="bad", cache="bad"),
        depth_cover=CanonicalDepthCover((0, 1)),
    )
    with pytest.raises(CanonicalKernelError, match="same ordered raw-prompt rows"):
        build_canonical_kernel_plan(
            (valid, mismatched),
            target_layer_count=1,
            target_hidden_size=2,
        )

    nonfinite = CanonicalParentTrace(
        "nonfinite",
        (
            torch.ones(4, 2, dtype=torch.float64),
            torch.full((4, 2), torch.nan, dtype=torch.float64),
        ),
        row_identity=_identity(4, tokenizer="nonfinite", cache="nonfinite"),
        depth_cover=CanonicalDepthCover((0, 1)),
    )
    with pytest.raises(CanonicalKernelError, match="NaN or infinity"):
        build_canonical_kernel_plan(
            (nonfinite,),
            target_layer_count=1,
            target_hidden_size=2,
        )
