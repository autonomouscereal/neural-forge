from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import neural_forge.dynamic_fold_phase_cache as phase_cache_module
from neural_forge.dynamic_fold_phase_cache import (
    COMPLETE_FILENAME,
    EVENT_BEGIN,
    EVENT_COMPLETE,
    EVENT_FAIL,
    EVENT_HEARTBEAT,
    EVENT_HIT,
    EVENT_MISS,
    EVENT_PROGRESS,
    DynamicFoldPhaseCache,
    FoldPhaseCacheAlreadyCompleteError,
    FoldPhaseCacheIntegrityError,
    FoldPhaseCachePublicationError,
    FoldPhaseEventLog,
    FoldPhaseEventLogError,
    FoldPhaseKey,
    FoldPhaseKeyError,
    read_fold_phase_events,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _key(
    *,
    parent: str = "parent-a",
    atol: float = 1e-5,
    abi_version: int = 1,
) -> FoldPhaseKey:
    return FoldPhaseKey(
        phase_name="causal-attention-fold",
        dependency_sha256s={
            "blank-target": _digest("blank"),
            "parent-traces": _digest(parent),
            "planner-source": _digest("planner"),
        },
        numeric_contract={
            "accumulator_dtype": "float32",
            "atol": atol,
            "device_class": "cuda-sm86",
            "rtol": 1e-4,
            "tf32": False,
        },
        abi={
            "consumer": "lfm2-causal-body-v1",
            "phase_api": abi_version,
            "tensor_layout": "q-major-contiguous",
        },
    )


def _clock() -> str:
    return "2026-07-30T12:34:56.123456Z"


def test_phase_key_is_canonical_and_binds_all_reuse_contracts() -> None:
    first = _key()
    reordered = FoldPhaseKey(
        phase_name="causal-attention-fold",
        dependency_sha256s={
            "planner-source": _digest("planner"),
            "parent-traces": _digest("parent-a"),
            "blank-target": _digest("blank"),
        },
        numeric_contract={
            "tf32": False,
            "rtol": 1e-4,
            "device_class": "cuda-sm86",
            "atol": 1e-5,
            "accumulator_dtype": "float32",
        },
        abi={
            "tensor_layout": "q-major-contiguous",
            "phase_api": 1,
            "consumer": "lfm2-causal-body-v1",
        },
    )

    assert reordered.phase_key_sha256 == first.phase_key_sha256
    assert FoldPhaseKey.from_dict(first.as_dict()).as_dict() == first.as_dict()
    assert _key(parent="parent-b").phase_key_sha256 != first.phase_key_sha256
    assert _key(atol=2e-5).phase_key_sha256 != first.phase_key_sha256
    assert _key(abi_version=2).phase_key_sha256 != first.phase_key_sha256

    tampered = first.as_dict()
    tampered["numeric_contract"]["atol"] = 0.5
    with pytest.raises(FoldPhaseKeyError, match="does not verify"):
        FoldPhaseKey.from_dict(tampered)
    with pytest.raises(FoldPhaseKeyError, match="at least one dependency"):
        FoldPhaseKey(
            phase_name="phase",
            dependency_sha256s={},
            numeric_contract={"dtype": "float32"},
            abi={"version": 1},
        )
    with pytest.raises(FoldPhaseKeyError, match="canonical JSON"):
        FoldPhaseKey(
            phase_name="phase",
            dependency_sha256s={"parent": _digest("parent")},
            numeric_contract={"atol": float("nan")},
            abi={"version": 1},
        )


def test_cache_round_trip_and_all_structured_events(tmp_path: Path) -> None:
    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    events = FoldPhaseEventLog(
        tmp_path / "run" / "events.jsonl",
        run_id="a0011",
        clock=_clock,
    )
    key = _key()

    assert cache.lookup(key, events=events) is None
    with cache.begin(key, events=events) as transaction:
        (transaction.payload_path / "layers").mkdir()
        (transaction.payload_path / "layers" / "attention.bin").write_bytes(
            b"folded-attention"
        )
        (transaction.payload_path / "summary.json").write_text(
            '{"rows":2048}\n',
            encoding="ascii",
        )
        transaction.progress({"completed_layers": 7, "total_layers": 14})
        transaction.heartbeat({"resident_bytes": 1234})
        published = transaction.complete(
            result={"completed_layers": 14, "training_steps": 0}
        )

    assert published.path == cache.entry_path(key)
    assert published.result == {"completed_layers": 14, "training_steps": 0}
    assert published.payload_inventory["total_bytes"] > 0
    hit = cache.lookup(key, events=events)
    assert hit is not None
    assert hit.completion_sha256 == published.completion_sha256

    marker = published.path / COMPLETE_FILENAME
    marker_value = json.loads(marker.read_text(encoding="ascii"))
    canonical = json.dumps(
        marker_value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii") + b"\n"
    assert marker.read_bytes() == canonical

    records = events.read()
    assert [record.kind for record in records] == [
        EVENT_MISS,
        EVENT_BEGIN,
        EVENT_PROGRESS,
        EVENT_HEARTBEAT,
        EVENT_COMPLETE,
        EVENT_HIT,
    ]
    assert [record.sequence for record in records] == list(range(6))
    assert records[0].previous_event_sha256 == "0" * 64
    assert all(
        record.previous_event_sha256 == records[index - 1].event_sha256
        for index, record in enumerate(records[1:], start=1)
    )
    assert records[-1].phase_key_sha256 == key.phase_key_sha256


def test_payload_and_completion_tampering_fail_closed(tmp_path: Path) -> None:
    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    key = _key()
    with cache.begin(key) as transaction:
        payload = transaction.payload_path / "operator.bin"
        payload.write_bytes(b"original")
        entry = transaction.complete(result={"operator_count": 1})

    (entry.payload_path / "operator.bin").write_bytes(b"tampered")
    with pytest.raises(FoldPhaseCacheIntegrityError, match="payload differs"):
        cache.lookup(key)

    second_cache = DynamicFoldPhaseCache(tmp_path / "second-cache")
    with second_cache.begin(key) as transaction:
        (transaction.payload_path / "operator.bin").write_bytes(b"original")
        second_entry = transaction.complete()
    marker = second_entry.path / COMPLETE_FILENAME
    value = json.loads(marker.read_text(encoding="ascii"))
    value["result"] = {"invented": True}
    marker.write_bytes(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")
        + b"\n"
    )
    with pytest.raises(FoldPhaseCacheIntegrityError, match="self-hash"):
        second_cache.lookup(key)


def test_stale_entry_cannot_be_reused_under_a_new_contract(tmp_path: Path) -> None:
    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    old_key = _key(atol=1e-5)
    new_key = _key(atol=1e-3)
    with cache.begin(old_key) as transaction:
        (transaction.payload_path / "result.bin").write_bytes(b"result")
        old_entry = transaction.complete()

    stale_destination = cache.entry_path(new_key)
    stale_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(old_entry.path, stale_destination)
    with pytest.raises(FoldPhaseCacheIntegrityError, match="stale"):
        cache.lookup(new_key)


def test_completed_entry_is_immutable_and_cannot_be_republished(
    tmp_path: Path,
) -> None:
    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    key = _key()
    with cache.begin(key) as transaction:
        (transaction.payload_path / "result.bin").write_bytes(b"first")
        entry = transaction.complete()
    before = (entry.path / COMPLETE_FILENAME).read_bytes()

    with pytest.raises(FoldPhaseCacheAlreadyCompleteError):
        cache.begin(key)
    assert (entry.path / COMPLETE_FILENAME).read_bytes() == before
    assert (entry.payload_path / "result.bin").read_bytes() == b"first"


def test_failed_transaction_is_removed_and_recorded(tmp_path: Path) -> None:
    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    events = FoldPhaseEventLog(
        tmp_path / "events.jsonl",
        run_id="failure-run",
        clock=_clock,
    )
    key = _key()

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with cache.begin(key, events=events) as transaction:
            (transaction.payload_path / "partial.bin").write_bytes(b"partial")
            transaction.progress({"row": 1})
            raise RuntimeError("synthetic failure")

    assert not cache.entry_path(key).exists()
    assert not list((cache.root / key.phase_name).glob("*.staging"))
    records = events.read()
    assert [record.kind for record in records] == [
        EVENT_BEGIN,
        EVENT_PROGRESS,
        EVENT_FAIL,
    ]
    assert records[-1].details["error_type"] == "RuntimeError"

    with pytest.raises(FoldPhaseCachePublicationError, match="without complete"):
        with cache.begin(key):
            pass


def test_event_log_rejects_tamper_noncanonical_and_truncation(
    tmp_path: Path,
) -> None:
    key = _key()
    path = tmp_path / "events.jsonl"
    events = FoldPhaseEventLog(path, run_id="chain", clock=_clock)
    events.miss(key)
    events.begin(key)
    original = path.read_bytes()

    first, second = original.splitlines(keepends=True)
    mutated = json.loads(first)
    mutated["kind"] = EVENT_HIT
    path.write_bytes(
        json.dumps(mutated, separators=(",", ":"), sort_keys=True).encode("ascii")
        + b"\n"
        + second
    )
    with pytest.raises(FoldPhaseEventLogError, match="self-hash"):
        read_fold_phase_events(path, expected_run_id="chain")

    path.write_bytes(b" " + original)
    with pytest.raises(FoldPhaseEventLogError, match="unique canonical"):
        read_fold_phase_events(path, expected_run_id="chain")

    path.write_bytes(original[:-1])
    with pytest.raises(FoldPhaseEventLogError, match="partial event"):
        read_fold_phase_events(path, expected_run_id="chain")


def test_event_log_rejects_reordering_and_wrong_run(tmp_path: Path) -> None:
    key = _key()
    path = tmp_path / "events.jsonl"
    events = FoldPhaseEventLog(path, run_id="chain", clock=_clock)
    events.miss(key)
    events.begin(key)
    first, second = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(second + first)
    with pytest.raises(FoldPhaseEventLogError, match="sequence|hash chain"):
        read_fold_phase_events(path)

    path.write_bytes(first + second)
    with pytest.raises(FoldPhaseEventLogError, match="run IDs"):
        read_fold_phase_events(path, expected_run_id="different-run")


def test_event_log_steady_state_append_does_not_reparse_or_rewrite_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _key()
    path = tmp_path / "events.jsonl"
    events = FoldPhaseEventLog(path, run_id="long-run", clock=_clock)
    events.begin(key)
    first_size = path.stat().st_size

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("steady-state append touched the existing prefix")

    with monkeypatch.context() as context:
        context.setattr(phase_cache_module, "_parse_event_log", forbidden)
        context.setattr(phase_cache_module, "_read_limited", forbidden)
        context.setattr(phase_cache_module, "_atomic_replace_bytes", forbidden)
        context.setattr(phase_cache_module.os, "fsync", lambda descriptor: None)
        for index in range(4096):
            record = events.progress(key, {"token_index": index})
            assert record.sequence == index + 1

    assert path.stat().st_size > first_size
    records = events.read()
    assert len(records) == 4097
    assert records[-1].details == {"token_index": 4095}
    assert records[-1].previous_event_sha256 == records[-2].event_sha256


def test_event_log_append_detects_external_mutation_and_truncation(
    tmp_path: Path,
) -> None:
    key = _key()
    path = tmp_path / "events.jsonl"
    events = FoldPhaseEventLog(path, run_id="observed-run", clock=_clock)
    events.miss(key)
    events.begin(key)
    original = path.read_bytes()

    first, second = original.splitlines(keepends=True)
    mutated = json.loads(first)
    mutated["kind"] = EVENT_HIT
    path.write_bytes(
        json.dumps(mutated, separators=(",", ":"), sort_keys=True).encode("ascii")
        + b"\n"
        + second
    )
    with pytest.raises(FoldPhaseEventLogError, match="self-hash"):
        events.progress(key)

    path.write_bytes(original)
    path.write_bytes(first)
    with pytest.raises(FoldPhaseEventLogError, match="truncated"):
        events.progress(key)
    assert path.read_bytes() == first


def test_event_log_accepts_a_valid_external_extension(tmp_path: Path) -> None:
    key = _key()
    path = tmp_path / "events.jsonl"
    primary = FoldPhaseEventLog(path, run_id="shared-run", clock=_clock)
    secondary = FoldPhaseEventLog(path, run_id="shared-run", clock=_clock)

    assert primary.miss(key).sequence == 0
    assert secondary.begin(key).sequence == 1
    assert primary.progress(key).sequence == 2
    assert [event.sequence for event in primary.read()] == [0, 1, 2]
