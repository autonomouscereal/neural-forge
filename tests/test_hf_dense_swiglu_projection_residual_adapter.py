from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from neural_forge.hf_dense_swiglu_projection_residual_adapter import (
    DenseSwiGLUProjectionResidualError,
    SparseDenseSwiGLUProjectionResidualRegistry,
)


def _registry() -> SparseDenseSwiGLUProjectionResidualRegistry:
    return SparseDenseSwiGLUProjectionResidualRegistry(
        (0, 1),
        hidden_size=4,
        intermediate_size=6,
        rank=2,
        dtype=torch.float32,
        device="cpu",
        initialization_seed=17,
    )


def _track_isfinite(
    monkeypatch: pytest.MonkeyPatch,
    registry: SparseDenseSwiGLUProjectionResidualRegistry,
) -> tuple[list[Tensor], int]:
    scanned: list[Tensor] = []
    parameter_count = len(tuple(registry.parameters()))
    original = torch.isfinite

    def tracked(value: Tensor) -> Tensor:
        scanned.append(value)
        return original(value)

    monkeypatch.setattr(torch, "isfinite", tracked)
    return scanned, parameter_count


def test_repeated_corrections_scan_one_full_registry_after_version_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    # This is the same version-changing operation used by an optimizer step.
    with torch.no_grad():
        registry.gate_left.add_(0.125)
    scanned, parameter_count = _track_isfinite(monkeypatch, registry)
    inputs = torch.randn(2, 3, 4)

    expected = F.linear(
        F.linear(inputs, registry.gate_right[0]), registry.gate_left[0]
    )
    first = registry.correction("gate", 0, inputs)
    for _ in range(12):
        repeated = registry.correction("gate", 0, inputs)
        torch.testing.assert_close(repeated, first, rtol=0.0, atol=0.0)

    assert len(scanned) == parameter_count
    torch.testing.assert_close(first, expected, rtol=0.0, atol=0.0)


def test_in_place_nonfinite_update_invalidates_cache_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    scanned, parameter_count = _track_isfinite(monkeypatch, registry)
    inputs = torch.randn(2, 4)

    # An ordinary finite update invalidates the prior initialization scan.
    with torch.no_grad():
        registry.up_right.add_(0.01)
    registry.correction("up", 0, inputs)
    assert len(scanned) == parameter_count
    registry.correction("up", 0, inputs)
    assert len(scanned) == parameter_count

    original_value = registry.up_right[0, 0, 0].detach().clone()
    with torch.no_grad():
        registry.up_right[0, 0, 0] = torch.inf
    with pytest.raises(
        DenseSwiGLUProjectionResidualError,
        match="contain NaN or infinity",
    ):
        registry.correction("up", 0, inputs)

    # A restore is also a version-changing copy and is validated before use.
    with torch.no_grad():
        registry.up_right[0, 0, 0].copy_(original_value)
    restored = registry.correction("up", 0, inputs)
    assert bool(torch.isfinite(restored).all())


def test_explicit_assertion_rescans_even_when_versions_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    scanned, parameter_count = _track_isfinite(monkeypatch, registry)

    registry.assert_finite()
    registry.assert_finite()

    assert len(scanned) == 2 * parameter_count
