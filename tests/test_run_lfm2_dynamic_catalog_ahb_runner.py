"""Focused production-runner contract for fold-only AHB integration."""

from __future__ import annotations

import importlib.util
import gc
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "experiments" / "run_lfm2_dynamic_catalog.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_dynamic_catalog_ahb_runner",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _required_args(root: Path) -> list[str]:
    return [
        "--parent-model",
        str(root / "parent"),
        "--calibration",
        str(root / "calibration.json"),
        "--cached-parent-outputs",
        str(root / "parent-outputs.json"),
        "--cached-parent-outputs-sha256",
        "d" * 64,
        "--cache-root",
        str(root / "run" / "cache"),
        "--snapshot-root",
        str(root / "run" / "snapshots"),
        "--publish-destination",
        str(root / "run" / "published"),
        "--ledger",
        str(root / "run" / "ledger.jsonl"),
        "--receipt",
        str(root / "run" / "receipt.json"),
    ]


def _config(root: Path, **overrides):
    return MODULE.ProductionLfm2CatalogConfig(
        parent_model=root / "parent",
        calibration=None,
        sealed_parent_cache_directory=root / "sealed-parent-cache",
        cached_parent_outputs=root / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=root / "run" / "cache",
        snapshot_root=root / "run" / "snapshots",
        publish_destination=root / "run" / "published",
        ledger_path=root / "run" / "ledger.jsonl",
        receipt_path=root / "run" / "receipt.json",
        device="cpu",
        dtype="float32",
        **overrides,
    )


def _wiring_kwargs() -> dict[str, object]:
    return {
        "parent_cache_payload": None,
        "tokenizer_fingerprint": "tokenizer",
        "evidence_builder": None,
        "final_evaluator": None,
        "model_saver": None,
        "model_loader": None,
        "fresh_reload_evaluator": None,
        "ledger": None,
        "device": "cpu",
        "dtype": MODULE.torch.float32,
    }


def test_ahb_defaults_preserve_the_existing_runner_identity(tmp_path) -> None:
    config = _config(tmp_path)

    assert config.uses_autoregressive_hankel_observability is False
    assert config.dense_autoregressive_hankel_future_lags == ()
    assert "autoregressive_hankel_cache" not in config.identity()
    assert (
        "dense_autoregressive_hankel_observability" not in config.identity()["target"]
    )
    assert MODULE._autoregressive_hankel_cache_receipt_fields(config) == {}


def test_ahb_cli_binds_sparse_lags_solver_and_stable_cache(tmp_path) -> None:
    parser = MODULE.build_parser()
    args = parser.parse_args(
        [
            *_required_args(tmp_path),
            "--compression-ratio",
            "4",
            "--autoregressive-hankel-cache-root",
            str(tmp_path / "stable" / "ahb"),
            "--autoregressive-hankel-source-cache-directory",
            str(tmp_path / "sealed-ahb-source"),
            "--dense-autoregressive-hankel-future-lags",
            "0,1,2,4,8,16,32",
            "--dense-autoregressive-hankel-state-ridge-fraction",
            "2e-8",
            "--dense-autoregressive-hankel-future-ridge-fraction",
            "3e-8",
            "--dense-autoregressive-hankel-maximum-condition-number",
            "2e9",
        ]
    )
    config = MODULE.config_from_args(args)

    assert config.uses_autoregressive_hankel_observability is True
    assert config.dense_autoregressive_hankel_future_lags == (
        0,
        1,
        2,
        4,
        8,
        16,
        32,
    )
    policy = config.identity()["target"]["dense_autoregressive_hankel_observability"]
    assert policy["future_lags"] == [0, 1, 2, 4, 8, 16, 32]
    assert policy["fold_only_closed_form"] is True
    assert policy["training_steps"] == 0
    receipt = MODULE._autoregressive_hankel_cache_receipt_fields(config)[
        "autoregressive_hankel_cache"
    ]
    assert receipt["checkpoint_granularity_batches"] == 1
    assert receipt["bounded_memory_cumulative_aggregation_required"] is True
    assert receipt["all_statistics_shards_retained_in_memory"] is False
    assert receipt["parent_trace_regeneration"] is False
    assert receipt["source_cache_is_explicit_not_active_session"] is True
    assert receipt["merged_active_session_inference_forbidden"] is True
    assert receipt["source_cache_directory"] == str(
        (tmp_path / "sealed-ahb-source").absolute()
    )


@pytest.mark.parametrize("lags", ["", "1,2", "0", "0,2,1", "0,1,1", "0,-1,2"])
def test_ahb_cli_refuses_noncausal_or_ambiguous_lag_schedules(lags) -> None:
    with pytest.raises(SystemExit):
        MODULE.build_parser().parse_args(
            [
                "--parent-model",
                "parent",
                "--cached-parent-outputs",
                "outputs.json",
                "--cached-parent-outputs-sha256",
                "d" * 64,
                "--cache-root",
                "cache",
                "--snapshot-root",
                "snapshots",
                "--publish-destination",
                "published",
                "--ledger",
                "ledger.jsonl",
                "--receipt",
                "receipt.json",
                "--dense-autoregressive-hankel-future-lags",
                lags,
            ]
        )


def test_ahb_requires_paired_cache_and_lags_and_automatic_balanced_fold(
    tmp_path,
) -> None:
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="explicit authenticated source-cache directory",
    ):
        _config(
            tmp_path,
            autoregressive_hankel_cache_root=tmp_path / "stable" / "ahb",
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="explicit authenticated source-cache directory",
    ):
        _config(
            tmp_path,
            compression_ratio=4.0,
            autoregressive_hankel_cache_root=tmp_path / "stable" / "ahb",
            dense_autoregressive_hankel_future_lags=(0, 1),
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="compression_ratio request",
    ):
        _config(
            tmp_path,
            autoregressive_hankel_cache_root=tmp_path / "stable" / "ahb",
            autoregressive_hankel_source_cache_directory=(
                tmp_path / "sealed-ahb-source"
            ),
            dense_autoregressive_hankel_future_lags=(0, 1),
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="circuit_balanced",
    ):
        _config(
            tmp_path,
            compression_ratio=4.0,
            autoregressive_hankel_cache_root=tmp_path / "stable" / "ahb",
            autoregressive_hankel_source_cache_directory=(
                tmp_path / "sealed-ahb-source"
            ),
            dense_autoregressive_hankel_future_lags=(0, 1),
            dense_residual_basis_mode="weight_complete",
        )


def test_ahb_wiring_seam_fails_closed_on_configuration_artifact_mismatch(
    tmp_path,
) -> None:
    enabled = _config(
        tmp_path,
        compression_ratio=4.0,
        autoregressive_hankel_cache_root=tmp_path / "stable" / "ahb",
        autoregressive_hankel_source_cache_directory=(tmp_path / "sealed-ahb-source"),
        dense_autoregressive_hankel_future_lags=(0, 1, 2),
    )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="fully authenticated aggregate observability artifact",
    ):
        MODULE.build_production_catalog_wiring(enabled, **_wiring_kwargs())

    disabled = _config(tmp_path / "disabled")
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="configuration did not bind",
    ):
        MODULE.build_production_catalog_wiring(
            disabled,
            autoregressive_hankel_observability=object(),
            **_wiring_kwargs(),
        )


@pytest.mark.parametrize("prune_fails", [False, True])
def test_ahb_materializer_resumes_before_build_and_holds_one_shard(
    tmp_path,
    monkeypatch,
    prune_fails,
) -> None:
    source_directory = tmp_path / "sealed-ahb-source"
    source_directory.mkdir()
    config = _config(
        tmp_path,
        compression_ratio=4.0,
        autoregressive_hankel_cache_root=tmp_path / "stable" / "ahb",
        autoregressive_hankel_source_cache_directory=source_directory,
        dense_autoregressive_hankel_future_lags=(0, 1, 2),
    )
    phase_keys = tuple(character * 64 for character in "abc")
    references = tuple(SimpleNamespace(index=index) for index in range(3))

    class FakeSession:
        evidence_sha256 = "d" * 64
        batch_references = references
        closed = False

        def verify(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

    session = FakeSession()
    source_cache = SimpleNamespace(
        cache_key="e" * 64,
        parent_fingerprint="parent",
        layer_topology_fingerprint="topology",
        implementation_fingerprint="source-implementation",
        selective_authenticated_session=lambda: session,
    )
    current_cache = SimpleNamespace(
        parent_fingerprint="parent",
        layer_topology_fingerprint="topology",
        implementation_fingerprint="implementation",
    )
    open_calls = []

    def open_source(path, *, verify):
        open_calls.append((Path(path), verify))
        return source_cache

    monkeypatch.setattr(MODULE, "open_lfm2_circuit_trace_cache", open_source)
    described = []

    def describe_phase(*, session, reference, future_lags):
        assert session is not None
        assert future_lags == (0, 1, 2)
        described.append(reference.index)
        return SimpleNamespace(
            phase_key_sha256=phase_keys[reference.index],
            boundary_names=("layer.000.residual.pre", "layer.000.residual.post_layer"),
        )

    monkeypatch.setattr(
        MODULE,
        "describe_lfm2_autoregressive_hankel_phase",
        describe_phase,
    )
    live_shards: list[weakref.ReferenceType] = []
    built_ordinals = []

    class FakeShard:
        def __init__(self, phase_key, *, cache_hit):
            self.phase_key_sha256 = phase_key
            self.cache_hit = cache_hit

    def load_phase(_root, *, session, reference, future_lags):
        del session, future_lags
        gc.collect()
        assert not any(reference() is not None for reference in live_shards)
        built_ordinals.append(reference.index)
        shard = FakeShard(
            phase_keys[reference.index],
            cache_hit=reference.index == 1,
        )
        live_shards.append(weakref.ref(shard))
        return shard

    monkeypatch.setattr(
        MODULE,
        "load_or_build_lfm2_autoregressive_hankel_statistics",
        load_phase,
    )

    evidence = SimpleNamespace(evidence_sha256="f" * 64, verify=lambda: True)
    result = SimpleNamespace(
        evidence=evidence,
        as_dict=lambda: {"report_sha256": "1" * 64},
    )
    aggregate = SimpleNamespace(
        result=result,
        receipt={"receipt_sha256": "2" * 64},
        provenance={"provenance_sha256": "3" * 64},
        path=tmp_path / "aggregate",
        aggregate_key_sha256="4" * 64,
        cache_hit=False,
    )

    class FakeAccumulator:
        plan = SimpleNamespace(plan_key_sha256="5" * 64)

        def __init__(self):
            self.processed_source_count = 1
            self.committed_phase_keys = phase_keys[:1]

        @property
        def completed(self):
            return self.processed_source_count == len(phase_keys)

        @property
        def next_expected_phase_key(self):
            if self.completed:
                return None
            return phase_keys[self.processed_source_count]

        def absorb(self, shard):
            ordinal = self.processed_source_count
            assert shard.phase_key_sha256 == phase_keys[ordinal]
            self.processed_source_count += 1
            body = {
                "phase_key_sha256": shard.phase_key_sha256,
                "durable_processed_source_count": self.processed_source_count,
                "phase_shard_path": f"shard-{ordinal}",
                "phase_shard_prunable_after_commit": True,
                "pooled_checkpoint_authenticated_before_prunable": True,
                "source_tensor_references_retained_by_accumulator": 0,
            }
            return {**body, "receipt_sha256": MODULE.canonical_sha256(body)}

        def finalize(self):
            assert self.completed
            return aggregate

    accumulator = FakeAccumulator()

    def open_accumulator(
        _root,
        *,
        planned_phase_keys,
        boundary_names,
        state_ridge_fraction,
        future_ridge_fraction,
        maximum_condition_number,
    ):
        assert planned_phase_keys == phase_keys
        assert len(boundary_names) == 2
        assert state_ridge_fraction > 0
        assert future_ridge_fraction > 0
        assert maximum_condition_number > 0
        return accumulator

    monkeypatch.setattr(
        MODULE,
        "open_lfm2_autoregressive_hankel_observability_accumulator",
        open_accumulator,
    )
    pruned = []

    def prune_phase(_root, absorption):
        pruned.append(absorption["phase_key_sha256"])
        if prune_fails:
            raise MODULE.ProductionLfm2CatalogError("cleanup deferred")
        return 100 + len(pruned)

    monkeypatch.setattr(
        MODULE,
        "_prune_absorbed_autoregressive_hankel_phase_shard",
        prune_phase,
    )
    progress_rows = []
    progress = SimpleNamespace(
        emit=lambda stage, **metadata: progress_rows.append((stage, metadata))
    )

    observed, runtime = (
        MODULE._materialize_production_autoregressive_hankel_observability(
            config,
            expected_parent_cache=current_cache,
            progress=progress,
        )
    )

    assert observed is aggregate
    assert open_calls == [(source_directory.resolve(), False)]
    assert described == [0, 1, 2]
    assert built_ordinals == [1, 2]
    assert pruned == list(phase_keys[1:])
    assert not any(reference() is not None for reference in live_shards)
    assert session.closed is True
    assert runtime["resumed_processed_source_count"] == 1
    assert runtime["phase_cache_hit_count"] == 1
    assert runtime["phase_cache_build_count"] == 1
    assert runtime["phase_shard_pruned_count"] == (0 if prune_fails else 2)
    assert runtime["phase_shard_prune_deferred_count"] == (2 if prune_fails else 0)
    assert runtime["active_or_merged_session_inferred"] is False
    assert runtime["cross_implementation_source"] is True
    assert runtime["source_trace_implementation_fingerprint"] == (
        "source-implementation"
    )
    assert runtime["active_trace_implementation_fingerprint"] == "implementation"
    assert runtime["training_steps"] == 0
    stages = [stage for stage, _metadata in progress_rows]
    assert "autoregressive_hankel_accumulator_resumed" in stages
    assert stages.count("autoregressive_hankel_phase_started") == 2
    assert stages.count("autoregressive_hankel_phase_committed") == 2
    assert stages.count("autoregressive_hankel_phase_prune_deferred") == (
        2 if prune_fails else 0
    )
    assert stages[-1] == "autoregressive_hankel_observability_ready"


def test_ahb_phase_shard_pruning_requires_sealed_absorption(tmp_path) -> None:
    statistics_root = tmp_path / "statistics"
    phase_key = "a" * 64
    shard = statistics_root / "shards" / phase_key[:2] / phase_key
    shard.mkdir(parents=True)
    (shard / "receipt.json").write_bytes(b"receipt")
    (shard / "statistics.safetensors").write_bytes(b"statistics")
    body = {
        "phase_key_sha256": phase_key,
        "phase_shard_path": str(shard.absolute()),
        "phase_shard_prunable_after_commit": True,
        "pooled_checkpoint_authenticated_before_prunable": True,
        "source_tensor_references_retained_by_accumulator": 0,
    }
    absorption = {**body, "receipt_sha256": MODULE.canonical_sha256(body)}

    released = MODULE._prune_absorbed_autoregressive_hankel_phase_shard(
        statistics_root,
        absorption,
    )

    assert released == len(b"receiptstatistics")
    assert not shard.exists()

    unsafe = dict(absorption)
    unsafe["pooled_checkpoint_authenticated_before_prunable"] = False
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="exact sealed durable absorption receipt",
    ):
        MODULE._prune_absorbed_autoregressive_hankel_phase_shard(
            statistics_root,
            unsafe,
        )
