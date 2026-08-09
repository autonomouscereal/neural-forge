from __future__ import annotations

from types import MappingProxyType

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_autoregressive_hankel_observability_cache import (  # noqa: E402
    Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError,
    describe_lfm2_autoregressive_hankel_observability_plan,
    load_or_build_lfm2_autoregressive_hankel_observability,
    open_lfm2_autoregressive_hankel_observability_accumulator,
)
from neural_forge.hf_lfm2_autoregressive_hankel_statistics import (  # noqa: E402
    accumulate_lfm2_ahb_trajectory,
)
import neural_forge.hf_lfm2_autoregressive_hankel_observability_cache as cache  # noqa: E402
import neural_forge.hf_lfm2_autoregressive_hankel_phase_cache as phase  # noqa: E402


_BOUNDARIES = (
    "layer.000.residual.pre",
    "layer.000.residual.post_layer",
)


def _digest(seed: int) -> str:
    return f"{seed:064x}"


def _statistics(source_digest: str, *, nuisance_scale: float = 1.0):
    nuisance = nuisance_scale * torch.tensor(
        [10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0]
    )
    signal = torch.tensor([0.1, 0.1, 0.2, 0.2, -0.1, -0.1, 0.3, 0.3, 0.0])
    rows = torch.stack((nuisance, signal), dim=1)
    boundaries = torch.stack(
        (rows, rows * torch.tensor([1.0, 2.0])),
        dim=0,
    )
    behavior = torch.zeros_like(rows)
    behavior[1:, 1] = signal[:-1]
    return accumulate_lfm2_ahb_trajectory(
        boundaries,
        behavior,
        future_lags=(0, 1),
        source_receipt_sha256=source_digest,
    )


def _phase_shard(root, ordinal: int):
    statistics = _statistics(
        _digest(100 + ordinal),
        nuisance_scale=1.0 + ordinal,
    )
    topology = {
        "schema": "neural-forge-lfm2-ahb-boundary-topology-v1",
        "trace_layer_topology_sha256": _digest(20),
        "layer_indices": [0],
        "boundary_names": list(_BOUNDARIES),
        "boundary_count": len(_BOUNDARIES),
        "hidden_size": 2,
    }
    future_lags = (0, 1)
    future_lags_body = {
        "schema": "neural-forge-lfm2-ahb-future-lags-v1",
        "lags": list(future_lags),
    }
    phase_body = {
        "schema": phase.LFM2_AUTOREGRESSIVE_HANKEL_PHASE_IDENTITY_SCHEMA,
        "parent_identity_sha256": _digest(1),
        "member_cache_key": _digest(30 + ordinal),
        "member_corpus_sha256": _digest(40 + ordinal),
        "member_identity_sha256": _digest(50 + ordinal),
        "member_session_sha256": _digest(60 + ordinal),
        "member_batch_index": ordinal,
        "trace_input_sha256": _digest(70 + ordinal),
        "trace_record_sha256": _digest(80 + ordinal),
        "selected_payload_sha256": _digest(90 + ordinal),
        "selected_tensor_names": ["authenticated.synthetic.endpoint"],
        "selected_tensor_count": 1,
        "trace_implementation_sha256": _digest(2),
        "future_lags": list(future_lags),
        "future_lags_sha256": canonical_sha256(future_lags_body),
        "boundary_topology": topology,
        "boundary_topology_sha256": canonical_sha256(topology),
        "numeric_abi": {"schema": "synthetic-phase-numeric-abi-v1"},
        "numeric_abi_sha256": canonical_sha256(
            {"schema": "synthetic-phase-numeric-abi-v1"}
        ),
        "statistics_source_sha256": _digest(3),
        "phase_cache_source_sha256": _digest(4),
        "checkpoint_granularity_batches": 1,
        "raw_uncentered_moments": True,
        "merged_evidence_identity_in_key": False,
        "member_shards_independently_reusable": True,
    }
    key = canonical_sha256(phase_body)
    identity = phase.Lfm2AutoregressiveHankelPhaseIdentity(
        body=MappingProxyType(phase_body),
        phase_key_sha256=key,
        session=None,
        reference=None,
        future_lags=future_lags,
        boundary_names=_BOUNDARIES,
    )
    path = root / "shards" / key[:2] / key
    path.mkdir(parents=True)
    phase._write_shard(path, identity, statistics)
    return phase._load_shard(path, identity, cache_hit=False)


def test_streaming_prefix_resumes_skips_completed_and_solves_one_pool(
    tmp_path,
    monkeypatch,
) -> None:
    first = _phase_shard(tmp_path / "phase", 0)
    second = _phase_shard(tmp_path / "phase", 1)
    planned = (first.phase_key_sha256, second.phase_key_sha256)
    accumulator = open_lfm2_autoregressive_hankel_observability_accumulator(
        tmp_path / "aggregate",
        planned_phase_keys=planned,
        boundary_names=_BOUNDARIES,
    )

    assert accumulator.processed_source_count == 0
    assert accumulator.next_expected_phase_key == first.phase_key_sha256
    absorption = accumulator.absorb(first)
    assert absorption["phase_shard_prunable_after_commit"] is True
    assert absorption["pooled_checkpoint_authenticated_before_prunable"] is True
    assert absorption["source_tensor_references_retained_by_accumulator"] == 0
    assert accumulator.processed_source_count == 1
    assert accumulator.next_expected_phase_key == second.phase_key_sha256
    assert accumulator.committed_phase_keys == (first.phase_key_sha256,)
    assert len(list((accumulator.work / "prefixes").iterdir())) == 1

    resumed = open_lfm2_autoregressive_hankel_observability_accumulator(
        tmp_path / "aggregate",
        planned_phase_keys=planned,
        boundary_names=_BOUNDARIES,
    )
    assert resumed.processed_source_count == 1
    assert resumed.next_expected_phase_key == second.phase_key_sha256
    resumed.absorb(second)
    assert resumed.next_expected_phase_key is None
    assert resumed.completed is True
    assert len(list((resumed.work / "prefixes").iterdir())) == 1
    expected_counts = (
        first.statistics.lag_pair_counts + second.statistics.lag_pair_counts
    )
    assert torch.equal(
        resumed._pooled_statistics.lag_pair_counts,
        expected_counts,
    )
    assert all("statistics" not in description for description in resumed._sources)

    calls = []
    original = cache._solver.build_lfm2_autoregressive_hankel_observability

    def one_pool(statistics, **kwargs):
        values = tuple(statistics)
        calls.append(values)
        assert len(values) == 1
        assert torch.equal(values[0].lag_pair_counts, expected_counts)
        return original(values, **kwargs)

    monkeypatch.setattr(
        cache._solver,
        "build_lfm2_autoregressive_hankel_observability",
        one_pool,
    )
    cold = resumed.finalize()
    assert len(calls) == 1
    assert cold.cache_hit is False
    assert cold.result.evidence.verify() is True
    assert cold.result.report["source_trajectory_count"] == 2
    assert cold.result.report["streaming_raw_moment_pooling"] is True
    assert cold.result.report["peak_source_shards_resident"] == 1
    assert cold.result.gram.dtype == torch.float64
    assert {path.name for path in cold.path.iterdir()} == {
        "receipt.json",
        "provenance.json",
        "report.json",
        "observability.safetensors",
    }
    assert (resumed.work / "current.json").is_file()
    assert len(list((resumed.work / "prefixes").iterdir())) == 1

    expected_gram = cold.result.gram.clone()
    monkeypatch.setattr(
        cache._solver,
        "build_lfm2_autoregressive_hankel_observability",
        lambda *_args, **_kwargs: pytest.fail("warm final hit reran solver"),
    )
    monkeypatch.setattr(
        torch.linalg,
        "eigh",
        lambda *_args, **_kwargs: pytest.fail("warm final hit reran eigensolve"),
    )
    warm = open_lfm2_autoregressive_hankel_observability_accumulator(
        tmp_path / "aggregate",
        planned_phase_keys=planned,
        boundary_names=_BOUNDARIES,
    )
    assert warm.next_expected_phase_key is None
    assert warm._pooled_statistics is None
    hit = warm.finalize()
    assert hit.cache_hit is True
    assert torch.equal(hit.result.gram, expected_gram)


def test_moment_plan_is_solver_independent_but_result_key_is_not(tmp_path) -> None:
    first = _phase_shard(tmp_path / "phase", 0)
    second = _phase_shard(tmp_path / "phase", 1)
    planned = (first.phase_key_sha256, second.phase_key_sha256)
    baseline = describe_lfm2_autoregressive_hankel_observability_plan(
        planned,
        boundary_names=_BOUNDARIES,
    )
    other_policy = describe_lfm2_autoregressive_hankel_observability_plan(
        planned,
        boundary_names=_BOUNDARIES,
        state_ridge_fraction=1e-6,
    )
    reordered = describe_lfm2_autoregressive_hankel_observability_plan(
        tuple(reversed(planned)),
        boundary_names=_BOUNDARIES,
    )

    assert baseline.plan_key_sha256 == other_policy.plan_key_sha256
    assert baseline.plan_key_sha256 != reordered.plan_key_sha256
    assert "solver_numeric_policy" not in baseline.body
    assert "solver_implementation_sha256" not in baseline.body
    assert baseline.body["peak_shard_tensor_residency"] == 1
    assert baseline.body["completed_prefixes_skippable_before_shard_load"] is True

    first_result = load_or_build_lfm2_autoregressive_hankel_observability(
        tmp_path / "aggregate",
        planned_phase_keys=planned,
        shards=iter((first, second)),
        boundary_names=_BOUNDARIES,
    )
    work = (
        tmp_path
        / "aggregate"
        / "work"
        / baseline.plan_key_sha256[:2]
        / baseline.plan_key_sha256
    )
    prefix_before = next((work / "prefixes").iterdir())
    pooled_before = (prefix_before / "pooled-statistics.safetensors").read_bytes()
    variant = load_or_build_lfm2_autoregressive_hankel_observability(
        tmp_path / "aggregate",
        planned_phase_keys=planned,
        shards=iter(()),
        boundary_names=_BOUNDARIES,
        state_ridge_fraction=1e-6,
    )
    assert first_result.aggregate_key_sha256 != variant.aggregate_key_sha256
    assert first_result.path != variant.path
    assert len(list((work / "prefixes").iterdir())) == 1
    prefix_after = next((work / "prefixes").iterdir())
    assert prefix_after == prefix_before
    assert (
        prefix_after / "pooled-statistics.safetensors"
    ).read_bytes() == pooled_before


def test_corrupt_resumable_prefix_fails_closed(tmp_path) -> None:
    first = _phase_shard(tmp_path / "phase", 0)
    second = _phase_shard(tmp_path / "phase", 1)
    planned = (first.phase_key_sha256, second.phase_key_sha256)
    accumulator = open_lfm2_autoregressive_hankel_observability_accumulator(
        tmp_path / "aggregate",
        planned_phase_keys=planned,
        boundary_names=_BOUNDARIES,
    )
    accumulator.absorb(first)
    pooled_path = next((accumulator.work / "prefixes").iterdir()) / (
        "pooled-statistics.safetensors"
    )
    payload = bytearray(pooled_path.read_bytes())
    payload[-1] ^= 0x01
    pooled_path.write_bytes(payload)

    with pytest.raises(
        Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError,
        match="bytes changed",
    ):
        open_lfm2_autoregressive_hankel_observability_accumulator(
            tmp_path / "aggregate",
            planned_phase_keys=planned,
            boundary_names=_BOUNDARIES,
        )


def test_corrupt_final_or_current_source_fails_closed_without_rebuild(
    tmp_path,
    monkeypatch,
) -> None:
    first = _phase_shard(tmp_path / "phase", 0)
    second = _phase_shard(tmp_path / "phase", 1)
    planned = (first.phase_key_sha256, second.phase_key_sha256)
    accumulator = open_lfm2_autoregressive_hankel_observability_accumulator(
        tmp_path / "aggregate",
        planned_phase_keys=planned,
        boundary_names=_BOUNDARIES,
    )
    source_path = first.path / "statistics.safetensors"
    source_bytes = bytearray(source_path.read_bytes())
    source_bytes[-1] ^= 0x01
    source_path.write_bytes(source_bytes)
    with pytest.raises(
        Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError,
        match="bytes changed",
    ):
        accumulator.absorb(first)
    assert accumulator.processed_source_count == 0

    first = _phase_shard(tmp_path / "phase-rebuilt", 0)
    second = _phase_shard(tmp_path / "phase-rebuilt", 1)
    planned = (first.phase_key_sha256, second.phase_key_sha256)
    cold = load_or_build_lfm2_autoregressive_hankel_observability(
        tmp_path / "aggregate-rebuilt",
        planned_phase_keys=planned,
        shards=iter((first, second)),
        boundary_names=_BOUNDARIES,
    )
    result_path = cold.path / "observability.safetensors"
    result_bytes = bytearray(result_path.read_bytes())
    result_bytes[-1] ^= 0x01
    result_path.write_bytes(result_bytes)
    monkeypatch.setattr(
        cache._solver,
        "build_lfm2_autoregressive_hankel_observability",
        lambda *_args, **_kwargs: pytest.fail("corrupt final was rebuilt"),
    )
    with pytest.raises(
        Lfm2AutoregressiveHankelObservabilityCacheAuthenticationError,
        match="bytes changed",
    ):
        load_or_build_lfm2_autoregressive_hankel_observability(
            tmp_path / "aggregate-rebuilt",
            planned_phase_keys=planned,
            shards=iter(()),
            boundary_names=_BOUNDARIES,
        )
