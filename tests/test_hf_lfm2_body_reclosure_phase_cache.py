from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from neural_forge.dynamic_fold_phase_cache import (  # noqa: E402
    DynamicFoldPhaseCache,
    FoldPhaseCacheIntegrityError,
    FoldPhaseKey,
)
from neural_forge.hf_lfm2_body_reclosure_phase_cache import (  # noqa: E402
    Lfm2BodyReclosureArtifact,
    Lfm2BodyReclosurePhaseAuthenticationError,
    Lfm2BodyReclosurePhaseCacheContext,
    Lfm2BodyReclosurePhaseCacheError,
    Lfm2BodyReclosurePhaseInProgressError,
    Lfm2BodyReclosurePhaseRequest,
    Lfm2ExistingBodyReclosurePhaseReuseRequest,
    build_lfm2_body_reclosure_phase_request,
    load_or_build_lfm2_body_reclosure_phase,
    lookup_lfm2_body_reclosure_phase,
    publish_lfm2_body_reclosure_phase,
)


def _digest(character: str) -> str:
    return character * 64


def _request(
    *,
    math_contract=None,
    evidence_contract=None,
    abi_contract=None,
    acceptance_policy=None,
):
    return build_lfm2_body_reclosure_phase_request(
        portable_pre_child_state_sha256=_digest("2"),
        portable_pre_child_topology_sha256=_digest("4"),
        pre_capture_sha256=_digest("6"),
        math_contract=(
            {"algorithm": "moving-gauge-v3", "ridge": 1e-10}
            if math_contract is None
            else math_contract
        ),
        evidence_contract=(
            {"fit_capture": _digest("8"), "heldout_capture": _digest("9")}
            if evidence_contract is None
            else evidence_contract
        ),
        abi_contract=(
            {"adapter": "lfm2", "implementation_sha256": _digest("a")}
            if abi_contract is None
            else abi_contract
        ),
        acceptance_policy=acceptance_policy,
    )


def _artifact(*, first_writer_offset: float = 0.0) -> Lfm2BodyReclosureArtifact:
    first_writer = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    if first_writer_offset:
        first_writer = first_writer + first_writer_offset
    return Lfm2BodyReclosureArtifact(
        tensors={
            "model.layers.000.writer": first_writer,
            "model.layers.001.writer": torch.tensor(
                [1.0, -2.0, 3.0], dtype=torch.float32
            ),
        },
        artifact_metadata={
            "math_artifact_sha256": _digest("1"),
            "portable_post_child_state_sha256": _digest("3"),
            "portable_post_child_topology_sha256": _digest("5"),
            "post_capture_sha256": _digest("7"),
            "cold_math_report_sha256": _digest("d"),
            "restore_manifest": {
                "logical_targets": [
                    "logical.first_writer",
                    "logical.final_writer",
                ],
                "restore_mode": "adapter-owned-exact",
            },
            "validation_manifest": {
                "capture_role": "post_body_reclosure",
                "record_set_sha256": _digest("e"),
            },
        },
        aliases={
            "logical.final_writer": "model.layers.001.writer",
            "logical.first_writer": "model.layers.000.writer",
        },
    )


def test_policy_is_excluded_but_policy_fields_in_identity_are_rejected() -> None:
    low = _request(
        acceptance_policy={
            "terminal_composition_regression_tolerance": 0.01,
            "manual_intelligence_review_required": True,
        }
    )
    high = _request(
        acceptance_policy={
            "terminal_composition_regression_tolerance": 0.25,
            "manual_intelligence_review_required": False,
        }
    )

    assert low.phase_key.phase_key_sha256 == high.phase_key.phase_key_sha256
    assert low.request_sha256 == high.request_sha256
    assert "acceptance_policy" not in low.as_dict()
    with pytest.raises(
        Lfm2BodyReclosurePhaseCacheError,
        match="acceptance-only policy field",
    ):
        _request(
            math_contract={
                "algorithm": "moving-gauge-v3",
                "terminal_composition_regression_tolerance": 0.25,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "final_performance_metrics_diagnostic_only",
        "dense_closed_loop_terminal_regression_tolerance",
        "dense_closed_loop_all_layer_finite_numeric_checks_diagnostic_only",
        "dense_closed_loop_intermediate_numeric_checks_diagnostic_only",
        "Dense-Closed-Loop-Terminal-Regression-Tolerance",
    ),
)
def test_current_terminal_and_diagnostic_policy_aliases_are_rejected(
    field_name: str,
) -> None:
    with pytest.raises(
        Lfm2BodyReclosurePhaseCacheError,
        match="acceptance-only policy field",
    ):
        _request(math_contract={"algorithm": "moving-gauge-v3", field_name: 0.25})


def test_math_affecting_boundary_tolerance_remains_in_identity() -> None:
    low = _request(
        math_contract={
            "algorithm": "moving-gauge-v3",
            "moving_gauge_boundary_regression_tolerance": 1e-8,
        }
    )
    high = _request(
        math_contract={
            "algorithm": "moving-gauge-v3",
            "moving_gauge_boundary_regression_tolerance": 1e-6,
        }
    )

    assert low.phase_key.phase_key_sha256 != high.phase_key.phase_key_sha256


def test_math_evidence_and_abi_mutations_each_change_the_phase_key() -> None:
    baseline = _request()
    changed_math = _request(
        math_contract={"algorithm": "moving-gauge-v3", "ridge": 2e-10}
    )
    changed_evidence = _request(
        evidence_contract={
            "fit_capture": _digest("b"),
            "heldout_capture": _digest("9"),
        }
    )
    changed_abi = _request(
        abi_contract={
            "adapter": "lfm2",
            "implementation_sha256": _digest("c"),
        }
    )

    assert (
        len(
            {
                baseline.phase_key.phase_key_sha256,
                changed_math.phase_key.phase_key_sha256,
                changed_evidence.phase_key.phase_key_sha256,
                changed_abi.phase_key.phase_key_sha256,
            }
        )
        == 4
    )


def test_direct_request_constructor_rejects_a_self_consistent_forged_key() -> None:
    request = _request()
    document = request.as_dict()
    document.pop("request_sha256")
    forged_key = FoldPhaseKey(
        phase_name=request.phase_key.phase_name,
        dependency_sha256s={"forged-input": _digest("f")},
        numeric_contract={"forged": True},
        abi={"forged": True},
    )
    document["phase_key"] = forged_key.as_dict()
    document["phase_key_sha256"] = forged_key.phase_key_sha256

    with pytest.raises(
        Lfm2BodyReclosurePhaseCacheError,
        match="differs from its phase key or seal",
    ):
        Lfm2BodyReclosurePhaseRequest(
            phase_key=forged_key,
            document=document,
        )


def test_artifact_reauthentication_detects_live_tensor_mutation() -> None:
    artifact = _artifact()
    assert artifact.reauthenticate() == artifact.artifact_sha256
    copies = artifact.authenticated_tensor_copies()
    assert (
        copies["model.layers.000.writer"]
        is not artifact.tensors["model.layers.000.writer"]
    )
    assert torch.equal(
        copies["model.layers.000.writer"],
        artifact.tensors["model.layers.000.writer"],
    )

    artifact.tensors["model.layers.000.writer"][0, 0] += 1.0
    with pytest.raises(
        Lfm2BodyReclosurePhaseAuthenticationError,
        match="changed after authentication",
    ):
        artifact.reauthenticate()
    with pytest.raises(Lfm2BodyReclosurePhaseAuthenticationError):
        artifact.authenticated_tensor_copies()


def test_publication_serializes_an_authenticated_tensor_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import safetensors.torch as safetensors_torch

    context = Lfm2BodyReclosurePhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "snapshot-cache")
    )
    request = _request()
    artifact = _artifact()
    real_save_file = safetensors_torch.save_file
    observed_snapshot_ids = {}

    def capture_snapshot(tensors, *args, **kwargs):
        observed_snapshot_ids.update(
            {name: id(tensor) for name, tensor in tensors.items()}
        )
        assert all(
            tensor is not artifact.tensors[name] for name, tensor in tensors.items()
        )
        source = artifact.tensors["model.layers.000.writer"]
        source.add_(1.0)
        try:
            return real_save_file(tensors, *args, **kwargs)
        finally:
            source.sub_(1.0)

    monkeypatch.setattr(safetensors_torch, "save_file", capture_snapshot)
    completed = load_or_build_lfm2_body_reclosure_phase(
        context,
        request,
        producer=lambda: artifact,
    )

    assert set(observed_snapshot_ids) == set(artifact.tensors)
    assert completed.artifact.artifact_sha256 == artifact.artifact_sha256


def test_split_lookup_miss_is_explicit_and_never_reserves_or_produces(
    tmp_path: Path,
) -> None:
    context = Lfm2BodyReclosurePhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "lookup-only-cache")
    )
    request = _request()

    missing = lookup_lfm2_body_reclosure_phase(context, request)

    assert missing is None
    assert context.cache.lookup(request.phase_key) is None
    assert context.cache.entry_path(request.phase_key).exists() is False


def test_split_delayed_cold_publication_reloads_an_authenticated_hit(
    tmp_path: Path,
) -> None:
    events = []
    context = Lfm2BodyReclosurePhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "delayed-publication-cache"),
        progress=lambda stage, details: events.append((stage, dict(details))),
    )
    request = _request()
    artifact = _artifact()
    artifact.reauthenticate()

    assert lookup_lfm2_body_reclosure_phase(context, request) is None
    published = publish_lfm2_body_reclosure_phase(context, request, artifact)
    reloaded = lookup_lfm2_body_reclosure_phase(context, request)

    assert published.cache_hit is False
    assert published.producer_invoked is False
    assert published.publication_race_lost is False
    assert published.artifact is not artifact
    assert published.artifact.artifact_sha256 == artifact.artifact_sha256
    assert reloaded is not None
    assert reloaded.cache_hit is True
    assert reloaded.artifact.artifact_sha256 == artifact.artifact_sha256
    assert any(stage == "body_reclosure_phase_publish_started" for stage, _ in events)
    assert any(stage == "body_reclosure_phase_published" for stage, _ in events)


@pytest.mark.parametrize("byte_identical", [True, False])
def test_split_publication_race_authenticates_byte_identity_or_fails_closed(
    tmp_path: Path,
    monkeypatch,
    byte_identical: bool,
) -> None:
    context = Lfm2BodyReclosurePhaseCacheContext(
        cache=DynamicFoldPhaseCache(
            tmp_path / f"split-race-{'same' if byte_identical else 'different'}"
        )
    )
    request = _request()
    winner_candidate = _artifact()
    local_candidate = (
        _artifact() if byte_identical else _artifact(first_writer_offset=1000.0)
    )
    original_lookup = context.cache.lookup
    winners = []

    def publish_winner_after_reported_miss(phase_key, *, events=None):
        monkeypatch.setattr(context.cache, "lookup", original_lookup)
        winners.append(
            publish_lfm2_body_reclosure_phase(
                context,
                request,
                winner_candidate,
            )
        )
        return None

    monkeypatch.setattr(
        context.cache,
        "lookup",
        publish_winner_after_reported_miss,
    )
    if byte_identical:
        resumed = publish_lfm2_body_reclosure_phase(
            context,
            request,
            local_candidate,
        )
        assert resumed.cache_hit is True
        assert resumed.producer_invoked is False
        assert resumed.publication_race_lost is True
        assert resumed.artifact.artifact_sha256 == local_candidate.artifact_sha256
    else:
        with pytest.raises(
            Lfm2BodyReclosurePhaseAuthenticationError,
            match="differs from the caller's authenticated artifact",
        ):
            publish_lfm2_body_reclosure_phase(
                context,
                request,
                local_candidate,
            )

    assert len(winners) == 1
    durable = lookup_lfm2_body_reclosure_phase(context, request)
    assert durable is not None
    assert durable.artifact.artifact_sha256 == winner_candidate.artifact_sha256


def test_split_publish_reports_active_reservation_without_waiting(
    tmp_path: Path,
) -> None:
    context = Lfm2BodyReclosurePhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "split-in-progress-cache")
    )
    request = _request()
    transaction = context.cache.begin(request.phase_key)
    try:
        with pytest.raises(Lfm2BodyReclosurePhaseInProgressError) as raised:
            publish_lfm2_body_reclosure_phase(context, request, _artifact())
    finally:
        transaction.__exit__(
            RuntimeError,
            RuntimeError("release test reservation"),
            None,
        )

    assert raised.value.phase_key_sha256 == request.phase_key.phase_key_sha256
    assert raised.value.retryable is True
    assert context.cache.lookup(request.phase_key) is None


def test_cold_publication_hot_hit_and_exact_hit_reuse_forbid_producer(
    tmp_path: Path,
) -> None:
    events = []
    context = Lfm2BodyReclosurePhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "cache"),
        progress=lambda stage, details: events.append((stage, dict(details))),
    )
    request = _request()
    calls = []

    def produce():
        calls.append(True)
        return _artifact()

    cold = load_or_build_lfm2_body_reclosure_phase(
        context,
        request,
        producer=produce,
    )
    hot = load_or_build_lfm2_body_reclosure_phase(
        context,
        request,
        producer=lambda: pytest.fail("hot hit invoked producer"),
    )
    exact_context = Lfm2BodyReclosurePhaseCacheContext(
        cache=context.cache,
        existing_phase_reuse_request=(
            Lfm2ExistingBodyReclosurePhaseReuseRequest(
                request.phase_key.phase_key_sha256,
                expected_completion_sha256=cold.completion_sha256,
                expected_artifact_sha256=cold.artifact.artifact_sha256,
                expected_math_artifact_sha256=_digest("1"),
                expected_post_child_state_sha256=_digest("3"),
                expected_post_child_topology_sha256=_digest("5"),
                expected_post_capture_sha256=_digest("7"),
                expected_cold_math_report_sha256=_digest("d"),
                expected_restore_manifest_sha256=cold.artifact.artifact_metadata[
                    "restore_manifest_sha256"
                ],
                expected_validation_manifest_sha256=(
                    cold.artifact.artifact_metadata["validation_manifest_sha256"]
                ),
            )
        ),
    )
    exact = load_or_build_lfm2_body_reclosure_phase(
        exact_context,
        request,
        producer=lambda: pytest.fail("exact hit invoked producer"),
    )

    assert calls == [True]
    assert cold.cache_hit is False and cold.producer_invoked is True
    assert hot.cache_hit is True and hot.producer_invoked is False
    assert exact.cache_hit is True and exact.producer_invoked is False
    assert hot.artifact.artifact_sha256 == cold.artifact.artifact_sha256
    assert exact.artifact.artifact_sha256 == cold.artifact.artifact_sha256
    assert dict(hot.artifact.aliases) == dict(cold.artifact.aliases)
    assert all(value == 0 for value in hot.artifact.execution_audit.values())
    assert {path.name for path in cold.entry.payload_path.iterdir()} == {
        "request.json",
        "receipt.json",
        "tensors.safetensors",
    }
    receipt = json.loads(
        (cold.entry.payload_path / "receipt.json").read_text(encoding="ascii")
    )
    metadata = receipt["artifact_metadata"]
    assert metadata["math_artifact_sha256"] == _digest("1")
    assert receipt["portable_pre_child_state_sha256"] == _digest("2")
    assert metadata["portable_post_child_state_sha256"] == _digest("3")
    assert receipt["portable_pre_child_topology_sha256"] == _digest("4")
    assert metadata["portable_post_child_topology_sha256"] == _digest("5")
    assert receipt["pre_capture_sha256"] == _digest("6")
    assert metadata["post_capture_sha256"] == _digest("7")
    assert metadata["cold_math_report_sha256"] == _digest("d")
    assert metadata["restore_manifest"]["restore_mode"] == "adapter-owned-exact"
    assert metadata["validation_manifest"]["capture_role"] == ("post_body_reclosure")
    assert [row["dtype"] for row in receipt["tensor_descriptors"]] == [
        "float64",
        "float32",
    ]
    assert [row["shape"] for row in receipt["tensor_descriptors"]] == [
        [3, 4],
        [3],
    ]
    assert receipt["tensor_aliases"] == dict(cold.artifact.aliases)
    assert receipt["tensor_file"]["pickle_allowed"] is False
    assert any(stage == "body_reclosure_phase_published" for stage, _ in events)
    assert any(stage == "body_reclosure_phase_cache_hit" for stage, _ in events)


def test_exact_hit_only_miss_is_fatal_and_never_calls_producer(
    tmp_path: Path,
) -> None:
    request = _request()
    context = Lfm2BodyReclosurePhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "missing"),
        existing_phase_reuse_request=Lfm2ExistingBodyReclosurePhaseReuseRequest(
            request.phase_key.phase_key_sha256
        ),
    )
    for action in (
        lambda: lookup_lfm2_body_reclosure_phase(context, request),
        lambda: publish_lfm2_body_reclosure_phase(context, request, _artifact()),
        lambda: load_or_build_lfm2_body_reclosure_phase(
            context,
            request,
            producer=lambda: pytest.fail("miss invoked producer"),
        ),
    ):
        with pytest.raises(
            Lfm2BodyReclosurePhaseCacheError,
            match="hit-only reuse forbids building",
        ):
            action()
    assert context.cache.lookup(request.phase_key) is None


def test_completion_between_lookup_and_reservation_resumes_exact_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = Lfm2BodyReclosurePhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "completed-race")
    )
    request = _request()
    original_lookup = context.cache.lookup
    winner_results = []

    def publish_after_reported_miss(phase_key, *, events=None):
        monkeypatch.setattr(context.cache, "lookup", original_lookup)
        winner_results.append(
            load_or_build_lfm2_body_reclosure_phase(
                context,
                request,
                producer=_artifact,
            )
        )
        return None

    monkeypatch.setattr(
        context.cache,
        "lookup",
        publish_after_reported_miss,
    )
    resumed = load_or_build_lfm2_body_reclosure_phase(
        context,
        request,
        producer=lambda: pytest.fail("completed race invoked outer producer"),
    )

    assert len(winner_results) == 1
    assert winner_results[0].cache_hit is False
    assert resumed.cache_hit is True
    assert resumed.producer_invoked is False
    assert resumed.publication_race_lost is True
    assert resumed.artifact.artifact_sha256 == (
        winner_results[0].artifact.artifact_sha256
    )


@pytest.mark.parametrize("tampered_name", ["receipt.json", "tensors.safetensors"])
def test_json_and_tensor_tamper_fail_closed(
    tmp_path: Path,
    tampered_name: str,
) -> None:
    context = Lfm2BodyReclosurePhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / tampered_name.replace(".", "-"))
    )
    request = _request()
    cold = load_or_build_lfm2_body_reclosure_phase(
        context,
        request,
        producer=_artifact,
    )
    path = cold.entry.payload_path / tampered_name
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    path.write_bytes(payload)

    with pytest.raises(FoldPhaseCacheIntegrityError):
        load_or_build_lfm2_body_reclosure_phase(
            context,
            request,
            producer=lambda: pytest.fail("corrupt entry silently rebuilt"),
        )


def test_concurrent_owner_returns_nonblocking_in_progress_and_produces_once(
    tmp_path: Path,
) -> None:
    producer_started = threading.Event()
    release_producer = threading.Event()
    count_lock = threading.Lock()
    producer_calls = 0
    in_progress_events = []

    def progress(stage, details):
        if stage == "body_reclosure_phase_in_progress":
            in_progress_events.append(dict(details))

    context = Lfm2BodyReclosurePhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "two-thread-cache"),
        progress=progress,
    )
    request = _request()

    def produce():
        nonlocal producer_calls
        with count_lock:
            producer_calls += 1
        producer_started.set()
        assert release_producer.wait(timeout=5.0)
        return _artifact()

    def resolve():
        return load_or_build_lfm2_body_reclosure_phase(
            context,
            request,
            producer=produce,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        winner_future = pool.submit(resolve)
        assert producer_started.wait(timeout=5.0)
        loser_future = pool.submit(resolve)
        with pytest.raises(Lfm2BodyReclosurePhaseInProgressError) as raised:
            loser_future.result(timeout=1.0)
        assert winner_future.done() is False
        release_producer.set()
        winner = winner_future.result(timeout=10.0)

    retry = load_or_build_lfm2_body_reclosure_phase(
        context,
        request,
        producer=lambda: pytest.fail("retry after publication invoked producer"),
    )

    assert producer_calls == 1
    assert winner.cache_hit is False
    assert winner.producer_invoked is True
    assert winner.publication_race_lost is False
    assert retry.cache_hit is True
    assert retry.producer_invoked is False
    assert retry.artifact.artifact_sha256 == winner.artifact.artifact_sha256
    disposition = raised.value.as_dict()
    assert disposition == {
        "schema": "neural-forge-lfm2-body-reclosure-in-progress-v1",
        "status": "in_progress",
        "phase_key_sha256": request.phase_key.phase_key_sha256,
        "retryable": True,
        "producer_invoked": False,
        "ownership_stolen": False,
        "owner_liveness_authenticated": False,
    }
    assert in_progress_events == [
        {
            "phase_key_sha256": request.phase_key.phase_key_sha256,
            "status": "in_progress",
            "retryable": True,
            "producer_invoked": False,
            "ownership_stolen": False,
            "owner_liveness_authenticated": False,
        }
    ]
