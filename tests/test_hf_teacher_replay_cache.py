from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from neural_forge import hf_teacher_replay_cache as replay_cache
from neural_forge import hf_teacher_token_replay


SOURCE_TREE_SHA = "1" * 64
SOURCE_STATE_SHA = "2" * 64
SOURCE_TOKENIZER_SHA = "3" * 64
CANDIDATE_TOKENIZER_SHA = "4" * 64


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


class _CpuCuda:
    @staticmethod
    def is_available() -> bool:
        return False


def _runtime() -> dict[str, Any]:
    torch = SimpleNamespace(
        __version__="test-torch",
        cuda=_CpuCuda(),
        version=SimpleNamespace(cuda=None),
        backends=SimpleNamespace(),
    )
    transformers = SimpleNamespace(__version__="test-transformers")
    return replay_cache.runtime_identity(torch, transformers, "cpu")


def _producer_manifest(tmp_path: Path, *, suffix: str = "") -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, Path]] = []
    for role in (
        "prepare_recovery_inputs",
        "hf_teacher_token_replay",
        "hf_recovery",
        "hf_teacher_replay_cache",
    ):
        path = tmp_path / f"{role}{suffix}.py"
        path.write_text(f"# {role} {suffix}\n", encoding="utf-8")
        rows.append((role, path))
    return replay_cache.producer_byte_hash_manifest(rows)


def _request(
    tmp_path: Path,
    *,
    calibration_texts: tuple[str, ...] = ("alpha calibration", "beta calibration"),
    tree_sha: str = SOURCE_TREE_SHA,
    teacher_parameters: dict[str, Any] | None = None,
    source_tokenizer_sha: str = SOURCE_TOKENIZER_SHA,
    producer_suffix: str = "",
) -> dict[str, Any]:
    parameters = teacher_parameters or {"steps": 180, "device": "cpu"}
    return replay_cache.build_request(
        expected_parent_artifact_tree_sha256=tree_sha,
        actual_parent_artifact_tree={
            "tree_sha256": tree_sha,
            "file_count": 3,
            "total_bytes": 42,
        },
        calibration_source_file_sha256="5" * 64,
        ordered_calibration_texts=calibration_texts,
        source_tokenizer_fingerprint={"sha256": source_tokenizer_sha},
        candidate_tokenizer_fingerprint={"sha256": CANDIDATE_TOKENIZER_SHA},
        teacher_parameters=parameters,
        producer_manifest=_producer_manifest(tmp_path, suffix=producer_suffix),
        runtime=_runtime(),
    )


def _source_fingerprint(*, state_sha: str = SOURCE_STATE_SHA) -> dict[str, Any]:
    return {
        "class": "tests.FakeParent",
        "model_state_fingerprint": {"state_dict_sha256": state_sha},
        "sha256": "6" * 64,
    }


def _teacher_payload(
    request: dict[str, Any],
    *,
    replay_nonce: str = "baseline",
) -> dict[str, Any]:
    body = {
        "schema": hf_teacher_token_replay.TEACHER_TOKEN_REPLAY_SCHEMA,
        "status": "frozen",
        "calibration_texts": ["alpha calibration", "beta calibration"],
        "parameters": copy.deepcopy(request["teacher_parameters"]),
        "tokenizer_fingerprints": copy.deepcopy(request["tokenizer_fingerprints"]),
        "source_model_fingerprint": _source_fingerprint(),
        "replay_nonce": replay_nonce,
    }
    return {**body, "payload_sha256": _canonical_sha256(body)}


@pytest.fixture
def fake_teacher_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_fields = {
        "schema",
        "status",
        "calibration_texts",
        "parameters",
        "tokenizer_fingerprints",
        "source_model_fingerprint",
        "replay_nonce",
        "payload_sha256",
    }

    def verify(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise ValueError("fake teacher fields changed")
        value = copy.deepcopy(payload)
        claimed = value.pop("payload_sha256")
        if claimed != _canonical_sha256(value):
            raise ValueError("fake teacher hash changed")
        return copy.deepcopy(payload)

    monkeypatch.setattr(
        replay_cache.hf_teacher_token_replay,
        "verify_teacher_token_replay_payload",
        verify,
    )


def test_request_identity_binds_every_replay_producer_input(tmp_path: Path) -> None:
    baseline = _request(tmp_path / "baseline")
    assert replay_cache.verify_request(baseline) == baseline
    assert baseline["source_artifact_tree"]["expected_tree_sha256"] == SOURCE_TREE_SHA
    assert "reservoir" not in json.dumps(baseline, sort_keys=True).casefold()
    assert "held-out" not in json.dumps(baseline, sort_keys=True).casefold()

    changed_calibration = _request(
        tmp_path / "calibration",
        calibration_texts=("alpha calibration", "changed calibration"),
    )
    changed_tree = _request(tmp_path / "tree", tree_sha="7" * 64)
    changed_parameters = _request(
        tmp_path / "parameters",
        teacher_parameters={"steps": 181, "device": "cpu"},
    )
    changed_tokenizer = _request(
        tmp_path / "tokenizer",
        source_tokenizer_sha="8" * 64,
    )
    changed_producer = _request(tmp_path / "producer", producer_suffix="-v2")
    identities = {
        value["request_sha256"]
        for value in (
            baseline,
            changed_calibration,
            changed_tree,
            changed_parameters,
            changed_tokenizer,
            changed_producer,
        )
    }
    assert len(identities) == 6


def test_envelope_rejects_tamper_wrong_request_and_wrong_live_model(
    tmp_path: Path,
    fake_teacher_verifier: None,
) -> None:
    request = _request(tmp_path / "request")
    teacher = _teacher_payload(request)
    envelope = replay_cache.build_envelope(
        request=request,
        teacher_payload=teacher,
        live_source_model_fingerprint=_source_fingerprint(),
    )
    assert replay_cache.verify_envelope(
        envelope,
        expected_request=request,
        live_source_model_fingerprint=_source_fingerprint(),
    ) == envelope
    assert envelope["status"] == "frozen-unaccepted"
    assert envelope["contains_sealed_data"] is False
    assert envelope["sealed_data_accessed"] is False
    assert envelope["publication_authorized"] is False

    tampered = copy.deepcopy(envelope)
    tampered["teacher_token_replay"]["replay_nonce"] = "tampered"
    with pytest.raises(replay_cache.TeacherReplayCacheError, match="strict verification"):
        replay_cache.verify_envelope(tampered, expected_request=request)

    wrong_request = _request(
        tmp_path / "wrong-request",
        teacher_parameters={"steps": 181, "device": "cpu"},
    )
    with pytest.raises(replay_cache.TeacherReplayCacheError, match="different request"):
        replay_cache.verify_envelope(envelope, expected_request=wrong_request)

    with pytest.raises(replay_cache.TeacherReplayCacheError, match="Live source model"):
        replay_cache.verify_envelope(
            envelope,
            expected_request=request,
            live_source_model_fingerprint=_source_fingerprint(state_sha="9" * 64),
        )

    falsely_resealed = copy.deepcopy(envelope)
    falsely_resealed["contains_sealed_data"] = True
    falsely_resealed_body = dict(falsely_resealed)
    falsely_resealed_body.pop("envelope_sha256")
    falsely_resealed["envelope_sha256"] = _canonical_sha256(falsely_resealed_body)
    with pytest.raises(replay_cache.TeacherReplayCacheError, match="self-hash failed"):
        replay_cache.verify_envelope(falsely_resealed, expected_request=request)


def test_publish_is_immutable_and_divergent_replay_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_teacher_verifier: None,
) -> None:
    cache_dir = tmp_path / "cache"
    request = _request(tmp_path / "request")
    teacher = _teacher_payload(request)
    synced_directories: list[Path] = []
    linked_paths: list[tuple[Path, Path]] = []
    original_fsync_directory = replay_cache._fsync_directory
    original_link = replay_cache.os.link

    def observed_fsync_directory(path: Path) -> None:
        synced_directories.append(path)
        original_fsync_directory(path)

    monkeypatch.setattr(
        replay_cache,
        "_fsync_directory",
        observed_fsync_directory,
    )

    def observed_link(source: Any, destination: Any, **kwargs: Any) -> None:
        linked_paths.append((Path(source), Path(destination)))
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(replay_cache.os, "link", observed_link)
    monkeypatch.setattr(
        replay_cache.os,
        "rename",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache publication must not use replacing rename")
        ),
    )
    envelope, result = replay_cache.publish_entry(
        cache_dir=cache_dir,
        request=request,
        teacher_payload=teacher,
        live_source_model_fingerprint=_source_fingerprint(),
    )
    assert result == "published"
    assert synced_directories[0] == cache_dir.parent.resolve()
    assert any(path.name.endswith(".staging") for path in synced_directories)
    assert cache_dir.resolve() in synced_directories
    assert len(linked_paths) == 1
    assert linked_paths[0][1] == (
        cache_dir.resolve() / request["request_sha256"] / replay_cache.CACHE_FILENAME
    )
    entry = replay_cache.cache_entry_path(cache_dir, request)
    envelope_path = entry / replay_cache.CACHE_FILENAME
    before = envelope_path.read_bytes()
    assert [path.name for path in entry.iterdir()] == [replay_cache.CACHE_FILENAME]
    assert replay_cache.read_entry(cache_dir, request) == envelope

    existing, result = replay_cache.publish_entry(
        cache_dir=cache_dir,
        request=request,
        teacher_payload=teacher,
        live_source_model_fingerprint=_source_fingerprint(),
    )
    assert result == "existing-byte-identical"
    fsync_count_after_publication = len(synced_directories)
    assert existing == envelope
    assert envelope_path.read_bytes() == before

    with pytest.raises(replay_cache.TeacherReplayCacheError, match="refusing overwrite"):
        replay_cache.publish_entry(
            cache_dir=cache_dir,
            request=request,
            teacher_payload=_teacher_payload(request, replay_nonce="different"),
            live_source_model_fingerprint=_source_fingerprint(),
        )
    assert envelope_path.read_bytes() == before
    assert len(synced_directories) == fsync_count_after_publication


def test_concurrent_identical_writer_is_adopted_without_staging_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_teacher_verifier: None,
) -> None:
    cache_dir = tmp_path / "cache"
    request = _request(tmp_path / "request")
    teacher = _teacher_payload(request)
    competing = replay_cache.build_envelope(
        request=request,
        teacher_payload=teacher,
        live_source_model_fingerprint=_source_fingerprint(),
    )

    def collide(reservation: Path) -> None:
        entry = reservation.parent / request["request_sha256"]
        entry.mkdir()
        (entry / replay_cache.CACHE_FILENAME).write_text(
            json.dumps(competing, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return None

    monkeypatch.setattr(replay_cache, "_reserve_publication", collide)
    envelope, result = replay_cache.publish_entry(
        cache_dir=cache_dir,
        request=request,
        teacher_payload=teacher,
        live_source_model_fingerprint=_source_fingerprint(),
    )
    assert result == "concurrent-byte-identical"
    assert envelope == competing
    assert not list(cache_dir.glob(".*.staging"))


def test_reader_rechecks_reservation_created_after_initial_false_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_teacher_verifier: None,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    request = _request(tmp_path / "request")
    envelope = replay_cache.build_envelope(
        request=request,
        teacher_payload=_teacher_payload(request),
        live_source_model_fingerprint=_source_fingerprint(),
    )
    entry = replay_cache.cache_entry_path(cache_dir, request)
    reservation = replay_cache._reservation_path(cache_dir, request)
    calls = 0
    original_reservation_present = replay_cache._reservation_present

    def interleaved_reservation_present(path: Path) -> bool:
        nonlocal calls
        assert path == reservation
        calls += 1
        if calls == 1:
            reservation.mkdir()
            entry.mkdir()
            return False
        if calls == 2:
            return True
        if calls == 3:
            return False
        if calls == 4:
            return True
        if calls == 5:
            (entry / replay_cache.CACHE_FILENAME).write_text(
                json.dumps(envelope, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            reservation.rmdir()
            return False
        return original_reservation_present(path)

    monkeypatch.setattr(
        replay_cache,
        "_reservation_present",
        interleaved_reservation_present,
    )
    assert replay_cache.read_entry(cache_dir, request) == envelope
    assert calls == 5
    assert not os.path.lexists(reservation)


def test_existing_malformed_or_nonexclusive_entry_fails_closed(
    tmp_path: Path,
    fake_teacher_verifier: None,
) -> None:
    cache_dir = tmp_path / "cache"
    request = _request(tmp_path / "request")
    replay_cache.publish_entry(
        cache_dir=cache_dir,
        request=request,
        teacher_payload=_teacher_payload(request),
        live_source_model_fingerprint=_source_fingerprint(),
    )
    entry = replay_cache.cache_entry_path(cache_dir, request)
    envelope_path = entry / replay_cache.CACHE_FILENAME
    envelope_path.write_text('{"schema": 1, "schema": 2}\n', encoding="utf-8")
    with pytest.raises(replay_cache.TeacherReplayCacheError, match="Duplicate JSON key"):
        replay_cache.read_entry(cache_dir, request)

    envelope_path.write_text("{}\n", encoding="utf-8")
    (entry / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(replay_cache.TeacherReplayCacheError, match="exactly"):
        replay_cache.read_entry(cache_dir, request)


def test_dangling_root_entry_and_reservation_symlinks_fail_closed(
    tmp_path: Path,
    fake_teacher_verifier: None,
) -> None:
    request = _request(tmp_path / "request")
    teacher = _teacher_payload(request)
    cache_dir = tmp_path / "cache"
    cache_dir.symlink_to(tmp_path / "missing-root", target_is_directory=True)
    with pytest.raises(replay_cache.TeacherReplayCacheError, match="real directory"):
        replay_cache.read_entry(cache_dir, request)
    with pytest.raises(replay_cache.TeacherReplayCacheError, match="real directory"):
        replay_cache.publish_entry(
            cache_dir=cache_dir,
            request=request,
            teacher_payload=teacher,
            live_source_model_fingerprint=_source_fingerprint(),
        )

    cache_dir.unlink()
    cache_dir.mkdir()
    entry = replay_cache.cache_entry_path(cache_dir, request)
    entry.symlink_to(tmp_path / "missing-entry", target_is_directory=True)
    with pytest.raises(replay_cache.TeacherReplayCacheError, match="real directory"):
        replay_cache.read_entry(cache_dir, request)
    entry.unlink()

    reservation = cache_dir / (
        f".{request['request_sha256']}{replay_cache._RESERVATION_SUFFIX}"
    )
    reservation.symlink_to(tmp_path / "missing-lock", target_is_directory=True)
    with pytest.raises(replay_cache.TeacherReplayCacheError, match="real directory"):
        replay_cache.read_entry(cache_dir, request)


@pytest.mark.parametrize("race_kind", ["empty-directory", "dangling-symlink"])
def test_entry_path_race_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_teacher_verifier: None,
    race_kind: str,
) -> None:
    request = _request(tmp_path / "request")
    cache_dir = tmp_path / "cache"
    entry = replay_cache.cache_entry_path(cache_dir, request)
    original_create = replay_cache._create_entry_noreplace

    def race_then_create(path: Path) -> tuple[int, int]:
        if race_kind == "empty-directory":
            path.mkdir()
        else:
            path.symlink_to(tmp_path / "missing-race-target", target_is_directory=True)
        return original_create(path)

    monkeypatch.setattr(replay_cache, "_create_entry_noreplace", race_then_create)
    with pytest.raises(replay_cache.TeacherReplayCacheError, match="refusing overwrite"):
        replay_cache.publish_entry(
            cache_dir=cache_dir,
            request=request,
            teacher_payload=_teacher_payload(request),
            live_source_model_fingerprint=_source_fingerprint(),
        )
    assert os.path.lexists(entry)
    assert not (entry / replay_cache.CACHE_FILENAME).exists()
    if race_kind == "empty-directory":
        assert entry.is_dir() and not any(entry.iterdir())
    else:
        assert entry.is_symlink() and not entry.exists()
    with pytest.raises(replay_cache.TeacherReplayCacheError):
        replay_cache.read_entry(cache_dir, request)
    assert not list(cache_dir.glob(".*.staging"))
    assert not os.path.lexists(
        cache_dir
        / f".{request['request_sha256']}{replay_cache._RESERVATION_SUFFIX}"
    )


def test_envelope_symlink_race_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_teacher_verifier: None,
) -> None:
    request = _request(tmp_path / "request")
    cache_dir = tmp_path / "cache"
    entry = replay_cache.cache_entry_path(cache_dir, request)
    original_link = replay_cache._link_envelope_noreplace

    def race_then_link(source: Path, destination: Path) -> None:
        destination.symlink_to(tmp_path / "missing-envelope")
        original_link(source, destination)

    monkeypatch.setattr(replay_cache, "_link_envelope_noreplace", race_then_link)
    with pytest.raises(replay_cache.TeacherReplayCacheError, match="refusing overwrite"):
        replay_cache.publish_entry(
            cache_dir=cache_dir,
            request=request,
            teacher_payload=_teacher_payload(request),
            live_source_model_fingerprint=_source_fingerprint(),
        )
    envelope_path = entry / replay_cache.CACHE_FILENAME
    assert envelope_path.is_symlink()
    assert not envelope_path.exists()
    with pytest.raises(replay_cache.TeacherReplayCacheError, match="real file"):
        replay_cache.read_entry(cache_dir, request)
