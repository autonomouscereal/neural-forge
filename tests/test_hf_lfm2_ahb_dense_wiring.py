from __future__ import annotations

import pytest


pytest.importorskip("torch")

import neural_forge.hf_lfm2_balanced_circuit_basis as balanced_module  # noqa: E402
from neural_forge.hf_lfm2_dense_budget_model_fold import (  # noqa: E402
    Lfm2DenseBudgetFoldConfig,
    _build_residual_basis,
)


def test_dense_circuit_basis_forwards_optional_ahb_evidence(monkeypatch) -> None:
    sentinel = object()
    expected = object()
    captured = {}

    def fake_builder(parent_model, trace_cache, target_width, **kwargs):
        captured.update(
            {
                "parent_model": parent_model,
                "trace_cache": trace_cache,
                "target_width": target_width,
                **kwargs,
            }
        )
        return expected

    monkeypatch.setattr(
        balanced_module,
        "build_lfm2_circuit_balanced_basis",
        fake_builder,
    )
    parent_model = object()
    parent_graph = object()
    trace_cache = object()
    result = _build_residual_basis(
        parent_model,
        parent_graph,
        trace_cache,
        8,
        Lfm2DenseBudgetFoldConfig(
            residual_basis_mode="circuit_balanced",
            run_closed_loop=False,
        ),
        autoregressive_hankel_observability=sentinel,
    )

    assert result is expected
    assert captured["parent_model"] is parent_model
    assert captured["trace_cache"] is trace_cache
    assert captured["target_width"] == 8
    assert captured["circuit_graph"] is parent_graph
    assert captured["autoregressive_hankel_observability"] is sentinel
