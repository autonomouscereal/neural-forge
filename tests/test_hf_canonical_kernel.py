from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_canonical_kernel import (  # noqa: E402
    CanonicalKernelError,
    CanonicalParentTrace,
    apply_parent_map,
    build_canonical_boundary_splits,
    build_canonical_kernel_plan,
    canonical_parent_trace_from_cache,
    fuse_parent_rows,
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
        weight=0.7,
    )
    parent_b = CanonicalParentTrace(
        "hybrid-b",
        tuple(state.matmul(parent_b_projection) for state in latent_states),
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

    assert first.fingerprint == second.fingerprint
    assert first.parent_names == ("transformer-a", "hybrid-b")
    assert first.normalized_parent_weights == pytest.approx((0.7, 0.3))
    assert len(first.layers) == 4
    assert [mapping.source_layer_index for mapping in first.layers[1].parent_maps] == [
        1,
        2,
    ]
    for layer in first.layers:
        assert layer.canonical_rows.shape == (32, 6)
        assert layer.effective_rank == 4
        assert layer.retained_kernel_fraction == pytest.approx(1.0)
        assert torch.isfinite(layer.canonical_rows).all()
        for parent_index, mapping in enumerate(layer.parent_maps):
            source = (parent_a, parent_b)[parent_index].layer_states[
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
            parent.layer_states[mapping.source_layer_index]
            for parent, mapping in zip(
                (parent_a, parent_b),
                layer.parent_maps,
                strict=True,
            )
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
    first = CanonicalParentTrace("first", first_states, weight=0.99)
    second = CanonicalParentTrace("second", second_states, weight=0.01)
    plan = build_canonical_kernel_plan(
        (first, second),
        target_layer_count=2,
        target_hidden_size=5,
    )
    reversed_plan = build_canonical_kernel_plan(
        (
            CanonicalParentTrace("first", first_states, weight=0.01),
            CanonicalParentTrace("second", second_states, weight=0.99),
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
        (CanonicalParentTrace("wide", states),),
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

    def parents(states):
        return (
            CanonicalParentTrace(
                "a",
                tuple(state.matmul(projection_a) for state in states[::2]),
                weight=0.6,
            ),
            CanonicalParentTrace(
                "b",
                tuple(state.matmul(projection_b) for state in states),
                weight=0.4,
            ),
        )

    splits = build_canonical_boundary_splits(
        parents(calibration_latent),
        parents(validation_latent),
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
        pooling="last_valid",
    )
    mean = canonical_parent_trace_from_cache(
        cache,
        name="cached-parent",
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
    )
    mismatched = CanonicalParentTrace(
        "mismatched",
        (
            torch.ones(5, 2, dtype=torch.float64),
            torch.ones(5, 2, dtype=torch.float64),
        ),
    )
    with pytest.raises(CanonicalKernelError, match="same prompt-aligned rows"):
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
    )
    with pytest.raises(CanonicalKernelError, match="NaN or infinity"):
        build_canonical_kernel_plan(
            (nonfinite,),
            target_layer_count=1,
            target_hidden_size=2,
        )
