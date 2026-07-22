from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
import neural_forge.hf_lfm2_semantic_foundation_statistics as statistics_module  # noqa: E402
import neural_forge.hf_lfm2_semantic_foundation as foundation  # noqa: E402
from neural_forge.hf_lfm2_semantic_foundation_statistics import (  # noqa: E402
    Lfm2SemanticFoundationStatisticsError,
    open_lfm2_semantic_foundation_interface_statistics,
    replay_lfm2_semantic_foundation_interface_statistics,
    write_lfm2_semantic_foundation_interface_statistics,
)
from neural_forge.hf_tied_causal_interface_anchor_fold import (  # noqa: E402
    TiedCausalAnchorSplit,
    fit_tied_causal_interface_anchor_residual,
)
from neural_forge.hf_tied_causal_interface_fold import (  # noqa: E402
    fit_tied_causal_interface_from_statistics,
    fit_tied_causal_interface_with_statistics,
    whiten_tied_causal_interface,
)


def _seal(body, field):
    return {**body, field: canonical_sha256(body)}


def _fixture():
    generator = torch.Generator().manual_seed(1701)
    vocabulary = 17
    source_width = 9
    target_width = 5
    basis = torch.randn(4, source_width, generator=generator, dtype=torch.float64)
    coefficients = torch.randn(
        vocabulary,
        4,
        generator=generator,
        dtype=torch.float64,
    )
    mean = torch.randn(source_width, generator=generator, dtype=torch.float64)
    table = mean.unsqueeze(0) + coefficients @ basis
    fit = torch.randn(31, source_width, generator=generator, dtype=torch.float64)
    validation = torch.randn(
        13,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    base, statistics = fit_tied_causal_interface_with_statistics(
        table,
        fit,
        target_width=target_width,
        validation_producer_rows=validation,
        metric_chunk_rows=7,
    )
    splits = {}
    for split_index, (name, rows) in enumerate(
        (("selection", 4), ("fit", 6), ("validation", 3))
    ):
        raw = torch.randn(
            rows,
            source_width,
            generator=generator,
            dtype=torch.float32,
        ).to(dtype=torch.bfloat16)
        splits[name] = TiedCausalAnchorSplit(
            name=name,
            row_ids=tuple(f"{name}/row.{index:04d}" for index in range(rows)),
            prefix_positions=torch.arange(
                100 * (split_index + 1),
                100 * (split_index + 1) + rows,
            ),
            source_producer_rows=raw,
            target_next_token_ids=torch.arange(rows) % vocabulary,
        )
    anchor = fit_tied_causal_interface_anchor_residual(
        base,
        table,
        selection_split=splits["selection"],
        fit_split=splits["fit"],
        validation_split=splits["validation"],
        metric_chunk_rows=7,
    )
    active = anchor.interface
    digest_values = iter(character * 64 for character in "1234567890abcdef")
    solve_body = {
        "schema": "neural-forge-lfm2-semantic-foundation-interface-v1",
        "compiler_cache_key": next(digest_values),
        "cache_parent_fingerprint": next(digest_values),
        "cache_manifest_sha256": next(digest_values),
        "bridge_artifact_sha256": next(digest_values),
        "trajectory_corpus": "/fixture/trajectory.json",
        "trajectory_corpus_sha256": next(digest_values),
        "trace_adapter_sha256": next(digest_values),
        "selective_catalog_preparation_receipt_sha256": next(digest_values),
        "selective_catalog_binding_sha256": next(digest_values),
        "selective_catalog_manifest_sha256": next(digest_values),
        "selective_catalog_artifact_sha256": next(digest_values),
        "final_norm_authentication_sha256": next(digest_values),
        "target_hidden_width": target_width,
        "target_norm_epsilon": 1e-5,
        "base_interface_report": dict(base.report),
        "base_interface_report_sha256": base.report["report_sha256"],
        "causal_anchor_report": dict(anchor.report),
        "causal_anchor_report_sha256": anchor.report["report_sha256"],
        "causal_anchor_selection_contract": (
            foundation._causal_anchor_selection_contract(anchor.report)
        ),
        "active_interface_report": dict(active.report),
        "active_interface_report_sha256": active.report["report_sha256"],
        "source_tied_table_sha256": statistics_module._foundation_tensor_sha256(table),
        "active_target_weight_sha256": (
            statistics_module._foundation_tensor_sha256(active.target_weight)
        ),
        "producer_encoder_sha256": statistics_module._foundation_tensor_sha256(
            active.producer_encoder
        ),
        "only_final_norm_payload_materialized": True,
        "legacy_layer_compiler_calls": 0,
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    solve = _seal(solve_body, "result_sha256")
    receipt = _seal(
        {
            "schema": "neural-forge-lfm2-semantic-foundation-v1",
            "interface_solve": solve,
            "source_selected_tied_table_sha256": (
                statistics_module._foundation_tensor_sha256(table)
            ),
        },
        "receipt_sha256",
    )
    return table, statistics, splits, base, anchor, receipt


def test_compact_statistics_round_trip_replays_exact_interface(monkeypatch, tmp_path):
    table, statistics, splits, base, anchor, receipt = _fixture()

    artifact = write_lfm2_semantic_foundation_interface_statistics(
        tmp_path / "statistics-store",
        base_statistics=statistics,
        anchor_splits=splits,
        foundation_receipt=receipt,
        foundation_receipt_file_sha256="f" * 64,
    )

    import safetensors
    import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache

    trace_accesses = []

    def reject_trace_access(*args, **kwargs):
        trace_accesses.append((args, kwargs))
        pytest.fail("compact statistics replay attempted to open parent traces")

    monkeypatch.setattr(trace_cache, "_load_verified_batch", reject_trace_access)
    monkeypatch.setattr(
        trace_cache,
        "open_lfm2_circuit_trace_cache",
        reject_trace_access,
    )
    monkeypatch.setattr(safetensors, "safe_open", reject_trace_access)
    reopened = open_lfm2_semantic_foundation_interface_statistics(
        artifact.path,
        expected_artifact_sha256=artifact.artifact_sha256,
    )
    replay = replay_lfm2_semantic_foundation_interface_statistics(
        reopened,
        table,
        foundation_receipt=receipt,
        foundation_receipt_file_sha256="f" * 64,
    )
    replayed_base = fit_tied_causal_interface_from_statistics(
        table,
        reopened.base_statistics,
    )

    assert torch.equal(replayed_base.target_weight, base.target_weight)
    assert torch.equal(replayed_base.producer_encoder, base.producer_encoder)
    assert dict(replayed_base.report) == dict(base.report)
    assert reopened.base_statistics.source_table_sha256 == (
        statistics.source_table_sha256
    )
    assert torch.equal(replay.interface.target_weight, anchor.interface.target_weight)
    assert torch.equal(
        replay.interface.producer_encoder,
        anchor.interface.producer_encoder,
    )
    assert dict(replay.interface.report) == dict(anchor.interface.report)
    assert dict(replay.solve_report) == receipt["interface_solve"]
    assert replay.replay_report["compiler_cache_open_calls"] == 0
    assert replay.replay_report["trace_safetensor_open_calls"] == 0
    full_self_dual = whiten_tied_causal_interface(anchor.interface)
    replayed_self_dual = whiten_tied_causal_interface(replay.interface)
    assert torch.equal(
        replayed_self_dual.producer_encoder,
        full_self_dual.producer_encoder,
    )
    assert torch.equal(replayed_self_dual.target_weight, full_self_dual.target_weight)
    assert dict(replayed_self_dual.report) == dict(full_self_dual.report)
    assert artifact.path.name == artifact.artifact_sha256
    assert artifact.manifest["payload_file_bytes"] < 32 * 1024
    anchor_policy = artifact.manifest["anchor_policy"]
    assert anchor_policy["minimum_mapped_parent_behavior_retention"] == 0.90
    assert (
        anchor_policy[
            "minimum_mapped_parent_behavior_retention_is_diagnostic_review_goal"
        ]
        is True
    )
    assert (
        anchor_policy[
            "minimum_mapped_parent_behavior_retention_used_as_candidate_eligibility_gate"
        ]
        is False
    )
    assert anchor_policy["maximum_relative_table_drift_is_diagnostic_review_goal"]
    assert (
        anchor_policy["maximum_relative_table_drift_used_as_candidate_eligibility_gate"]
        is False
    )
    assert anchor_policy["base_control_used_only_for_structural_fallback"] is True
    descriptors = {
        value["name"]: value for value in artifact.manifest["payload_tensors"]
    }
    assert descriptors["anchor.source_rows_bfloat16"]["dtype"] == "torch.bfloat16"
    assert trace_accesses == []


def test_compact_statistics_rejects_one_payload_byte_change(tmp_path):
    _table, statistics, splits, _base, _anchor, receipt = _fixture()
    artifact = write_lfm2_semantic_foundation_interface_statistics(
        tmp_path / "statistics-store",
        base_statistics=statistics,
        anchor_splits=splits,
        foundation_receipt=receipt,
        foundation_receipt_file_sha256="f" * 64,
    )
    payload = Path(artifact.path) / "statistics.safetensors"
    data = bytearray(payload.read_bytes())
    data[-1] ^= 1
    payload.write_bytes(data)

    with pytest.raises(
        Lfm2SemanticFoundationStatisticsError,
        match="payload differs",
    ):
        open_lfm2_semantic_foundation_interface_statistics(
            artifact.path,
            expected_artifact_sha256=artifact.artifact_sha256,
        )


def test_foundation_rehydration_sidecar_path_never_opens_full_cache(
    monkeypatch,
    tmp_path,
):
    table, statistics, splits, _base, anchor, receipt = _fixture()
    artifact = write_lfm2_semantic_foundation_interface_statistics(
        tmp_path / "statistics-store",
        base_statistics=statistics,
        anchor_splits=splits,
        foundation_receipt=receipt,
        foundation_receipt_file_sha256="f" * 64,
    )

    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                anchor.interface.target_weight.to(dtype=torch.bfloat16)
            )

        def get_input_embeddings(self):
            return SimpleNamespace(weight=self.weight)

        def get_output_embeddings(self):
            return SimpleNamespace(weight=self.weight)

    target = Target()
    solve = receipt["interface_solve"]
    semantic_cache = SimpleNamespace(
        cache_key="e" * 64,
        parent_fingerprint=solve["cache_parent_fingerprint"],
    )
    bridge = SimpleNamespace(artifact_sha256=solve["bridge_artifact_sha256"])
    access_attempts = []

    def reject_access(*args, **kwargs):
        access_attempts.append((args, kwargs))
        pytest.fail("sidecar rehydration attempted to open full-cache evidence")

    monkeypatch.setattr(
        foundation,
        "authenticate_lfm2_semantic_foundation_receipt",
        lambda *_args, **_kwargs: (receipt, "f" * 64),
    )
    monkeypatch.setattr(
        foundation,
        "fit_lfm2_semantic_foundation_interface",
        reject_access,
    )
    monkeypatch.setattr(
        foundation,
        "open_lfm2_cpcf_selective_evidence_provider_from_catalog",
        reject_access,
    )
    import safetensors
    import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache

    monkeypatch.setattr(trace_cache, "_load_verified_batch", reject_access)
    monkeypatch.setattr(
        trace_cache,
        "open_lfm2_circuit_trace_cache",
        reject_access,
    )
    monkeypatch.setattr(safetensors, "safe_open", reject_access)

    interface, rehydration = foundation.rehydrate_lfm2_semantic_foundation_interface(
        target,
        table,
        semantic_cache=semantic_cache,
        bridge=bridge,
        target_tokenizer=object(),
        parent_model_directory=tmp_path,
        input_metadata={},
        foundation_receipt_json=tmp_path / "never-read-foundation.json",
        expected_foundation_receipt_sha256=receipt["receipt_sha256"],
        interface_statistics_directory=artifact.path,
        expected_interface_statistics_sha256=artifact.artifact_sha256,
    )

    assert torch.equal(interface.target_weight, anchor.interface.target_weight)
    assert torch.equal(interface.producer_encoder, anchor.interface.producer_encoder)
    assert dict(interface.report) == dict(anchor.interface.report)
    assert rehydration["compiler_cache_open_calls"] == 0
    assert rehydration["trace_safetensor_open_calls"] == 0
    assert rehydration["selective_provider_open_calls"] == 0
    assert access_attempts == []


def test_authenticated_backfill_runs_legacy_oracle_once_and_replays_exactly(
    monkeypatch,
    tmp_path,
):
    table, statistics, splits, _base, anchor, receipt = _fixture()
    compiler_cache_directory = tmp_path / "compiler-cache"
    compiler_cache_directory.mkdir()
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_sha256")
    receipt_body.update(
        {
            "compiler_cache_directory": str(compiler_cache_directory.resolve()),
            "compiler_cache_key": receipt["interface_solve"]["compiler_cache_key"],
            "trajectory_corpus": "/fixture/trajectory.json",
            "catalog_preparation": {},
        }
    )
    receipt = _seal(receipt_body, "receipt_sha256")
    solve = receipt["interface_solve"]

    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                anchor.interface.target_weight.to(dtype=torch.bfloat16)
            )

        def get_input_embeddings(self):
            return SimpleNamespace(weight=self.weight)

        def get_output_embeddings(self):
            return SimpleNamespace(weight=self.weight)

    target = Target()
    semantic_cache = SimpleNamespace(
        cache_key="e" * 64,
        parent_fingerprint=solve["cache_parent_fingerprint"],
    )
    bridge = SimpleNamespace(artifact_sha256=solve["bridge_artifact_sha256"])
    compiler_cache = SimpleNamespace(
        cache_key=solve["compiler_cache_key"],
        parent_fingerprint=solve["cache_parent_fingerprint"],
    )
    calls = {"open": 0, "fit": 0}

    monkeypatch.setattr(
        foundation,
        "authenticate_lfm2_semantic_foundation_receipt",
        lambda *_args, **_kwargs: (receipt, "f" * 64),
    )
    import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache

    def open_once(*_args, **_kwargs):
        calls["open"] += 1
        return compiler_cache

    def fit_once(**_kwargs):
        calls["fit"] += 1
        return foundation.Lfm2SemanticFoundationInterfaceSolve(
            interface=anchor.interface,
            report=solve,
            base_statistics=statistics,
            anchor_splits=splits,
        )

    monkeypatch.setattr(trace_cache, "open_lfm2_circuit_trace_cache", open_once)
    monkeypatch.setattr(
        foundation,
        "fit_lfm2_semantic_foundation_interface",
        fit_once,
    )

    result = foundation.backfill_lfm2_semantic_foundation_interface_statistics(
        target,
        table,
        semantic_cache=semantic_cache,
        bridge=bridge,
        target_tokenizer=object(),
        parent_model_directory=tmp_path,
        input_metadata={},
        foundation_receipt_json=tmp_path / "authenticated-by-fixture.json",
        expected_foundation_receipt_sha256=receipt["receipt_sha256"],
        artifact_store_directory=tmp_path / "statistics-store",
    )

    assert calls == {"open": 1, "fit": 1}
    assert result.report["full_cache_oracle_open_calls"] == 1
    assert result.report["compact_replay_compiler_cache_open_calls"] == 0
    assert result.report["compact_replay_trace_safetensor_open_calls"] == 0
    assert result.report["full_cache_solve_report_reproduced_exactly"] is True
    assert result.report["self_dual_gauge_reproduced_exactly"] is True
    assert (
        result.artifact.artifact_sha256 == result.report["statistics_artifact_sha256"]
    )


def test_foundation_rehydration_rejects_tampered_sidecar_before_cache_open(
    monkeypatch,
    tmp_path,
):
    table, statistics, splits, _base, anchor, receipt = _fixture()
    artifact = write_lfm2_semantic_foundation_interface_statistics(
        tmp_path / "statistics-store",
        base_statistics=statistics,
        anchor_splits=splits,
        foundation_receipt=receipt,
        foundation_receipt_file_sha256="f" * 64,
    )
    payload = Path(artifact.path) / "statistics.safetensors"
    raw = bytearray(payload.read_bytes())
    raw[-1] ^= 1
    payload.write_bytes(raw)

    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                anchor.interface.target_weight.to(dtype=torch.bfloat16)
            )

        def get_input_embeddings(self):
            return SimpleNamespace(weight=self.weight)

        def get_output_embeddings(self):
            return SimpleNamespace(weight=self.weight)

    solve = receipt["interface_solve"]
    semantic_cache = SimpleNamespace(
        cache_key="e" * 64,
        parent_fingerprint=solve["cache_parent_fingerprint"],
    )
    bridge = SimpleNamespace(artifact_sha256=solve["bridge_artifact_sha256"])
    monkeypatch.setattr(
        foundation,
        "authenticate_lfm2_semantic_foundation_receipt",
        lambda *_args, **_kwargs: (receipt, "f" * 64),
    )
    import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache

    def reject_cache_open(*_args, **_kwargs):
        pytest.fail("tampered sidecar fallback attempted a full-cache open")

    monkeypatch.setattr(
        trace_cache,
        "open_lfm2_circuit_trace_cache",
        reject_cache_open,
    )

    with pytest.raises(
        foundation.Lfm2SemanticFoundationError,
        match="failed authentication",
    ):
        foundation.rehydrate_lfm2_semantic_foundation_interface(
            Target(),
            table,
            semantic_cache=semantic_cache,
            bridge=bridge,
            target_tokenizer=object(),
            parent_model_directory=tmp_path,
            input_metadata={},
            foundation_receipt_json=tmp_path / "authenticated-by-fixture.json",
            expected_foundation_receipt_sha256=receipt["receipt_sha256"],
            interface_statistics_directory=artifact.path,
            expected_interface_statistics_sha256=artifact.artifact_sha256,
        )
