from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest


pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack  # noqa: E402


def test_source_local_global_wrapper_selects_action_evidence_and_v2b_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = SimpleNamespace(evidence_sha256="1" * 64)
    physical = MappingProxyType({"model.layers.0.conv.out_proj.weight": object()})
    calls: dict[str, object] = {}

    def build(_accumulators, **kwargs):
        calls["source_local"] = kwargs["source_local"]
        return evidence, MappingProxyType({})

    def integrate(observed_evidence, **kwargs):
        calls["evidence"] = observed_evidence
        calls["physical"] = kwargs["live_physical_parameters"]
        calls["boundary_mode"] = kwargs["boundary_mode"]
        calls["scale_policy"] = kwargs["correction_scale_policy"]
        return SimpleNamespace(report=MappingProxyType({}))

    monkeypatch.setattr(
        stack,
        "_build_global_residual_evidence_from_accumulators",
        build,
    )
    monkeypatch.setattr(
        stack,
        "_global_live_physical_state",
        lambda _target: physical,
    )
    monkeypatch.setattr(
        stack,
        "integrate_lfm2_producer_aware_global_closure",
        integrate,
    )

    installation, returned_evidence = (
        stack._compile_install_global_residual_from_accumulators(
            MappingProxyType({}),
            target_model=object(),
            mapped_parent_tied_table=object(),
            deferred_final_w2_prior=object(),
            deferred_final_writer_parameter=object(),
            correction_scale_frontier=(0.5, 1.0),
            correction_scale=0.5,
            source_local_boundaries=True,
        )
    )

    assert installation.report == {}
    assert returned_evidence is evidence
    assert calls["source_local"] is True
    assert calls["evidence"] is evidence
    assert calls["physical"] is physical
    assert calls["boundary_mode"] == (
        stack.LFM2_PRODUCER_AWARE_SOURCE_LOCAL_BOUNDARY_MODE
    )
    scale_policy = calls["scale_policy"]
    assert scale_policy.materialization_frontier == (0.5, 1.0)
    assert scale_policy.materialization_scale == 0.5


def test_legacy_global_wrapper_does_not_emit_a_boundary_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = SimpleNamespace(evidence_sha256="2" * 64)
    calls: dict[str, object] = {}

    def build(_accumulators, **kwargs):
        calls["source_local"] = kwargs["source_local"]
        return evidence, MappingProxyType({})

    def integrate(_evidence, **kwargs):
        calls["has_boundary_mode"] = "boundary_mode" in kwargs
        return SimpleNamespace(report=MappingProxyType({}))

    monkeypatch.setattr(
        stack,
        "_build_global_residual_evidence_from_accumulators",
        build,
    )
    monkeypatch.setattr(
        stack,
        "_global_live_physical_state",
        lambda _target: MappingProxyType({}),
    )
    monkeypatch.setattr(
        stack,
        "integrate_lfm2_producer_aware_global_closure",
        integrate,
    )

    stack._compile_install_global_residual_from_accumulators(
        MappingProxyType({}),
        target_model=object(),
        mapped_parent_tied_table=object(),
        deferred_final_w2_prior=object(),
        deferred_final_writer_parameter=object(),
    )

    assert calls == {"source_local": False, "has_boundary_mode": False}
