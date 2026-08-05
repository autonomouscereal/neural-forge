from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import neural_forge.dynamic_child_replay_phase_cache as replay_cache_module
from neural_forge.dynamic_child_replay_phase_cache import (
    DYNAMIC_CHILD_REPLAY_PHASE_IDENTITY_SCHEMA,
    DynamicChildReplayPhaseCacheContext,
    DynamicChildReplayPhaseIdentity,
    load_or_build_dynamic_child_replay_phase,
)
from neural_forge.dynamic_fold_phase_cache import (
    DynamicFoldPhaseCache,
    FoldPhaseKey,
)


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _identity(label: str = "alpha") -> DynamicChildReplayPhaseIdentity:
    key = FoldPhaseKey(
        phase_name="test-child-replay-v1",
        dependency_sha256s={"child": _digest({"child": label})},
        numeric_contract={"dtype": "float32", "training_steps": 0},
        abi={"codec": "test-child-replay-codec-v1"},
    )
    return DynamicChildReplayPhaseIdentity(
        phase_key=key,
        codec_id="test-child-replay-codec-v1",
        request={
            "schema": DYNAMIC_CHILD_REPLAY_PHASE_IDENTITY_SCHEMA,
            "phase_key_sha256": key.phase_key_sha256,
            "codec_id": "test-child-replay-codec-v1",
            "family": "test",
        },
    )


class _Codec:
    codec_id = "test-child-replay-codec-v1"

    def authenticate(self, value, identity):
        assert identity.codec_id == self.codec_id
        return _digest(value)

    def write(self, value, payload_path, identity):
        path = payload_path / "replay.json"
        path.write_bytes(_canonical(value) + b"\n")
        return {
            "format": "canonical-json",
            "request_sha256": identity.request_sha256,
        }

    def load(self, entry, identity):
        assert identity.codec_id == self.codec_id
        path = entry.payload_path / "replay.json"
        raw = path.read_bytes()
        value = json.loads(raw)
        assert raw == _canonical(value) + b"\n"
        return value


def test_generic_child_replay_phase_cold_hot_and_failure_cleanup(
    tmp_path: Path,
) -> None:
    context = DynamicChildReplayPhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "cache")
    )
    identity = _identity()
    calls = []

    def produce():
        calls.append(True)
        return {"child": "alpha", "records": [1, 2, 3]}

    cold = load_or_build_dynamic_child_replay_phase(
        context,
        identity,
        codec=_Codec(),
        producer=produce,
    )
    hot = load_or_build_dynamic_child_replay_phase(
        context,
        identity,
        codec=_Codec(),
        producer=lambda: (_ for _ in ()).throw(
            AssertionError("cache hit invoked producer")
        ),
    )

    assert calls == [True]
    assert cold.cache_hit is False
    assert cold.producer_invoked is True
    assert hot.cache_hit is True
    assert hot.producer_invoked is False
    assert hot.replay == cold.replay
    assert hot.replay_sha256 == cold.replay_sha256

    failed_identity = _identity("failed")
    with pytest.raises(RuntimeError, match="synthetic failure"):
        load_or_build_dynamic_child_replay_phase(
            context,
            failed_identity,
            codec=_Codec(),
            producer=lambda: (_ for _ in ()).throw(
                RuntimeError("synthetic failure")
            ),
        )
    assert not context.cache.entry_path(failed_identity.phase_key).exists()


def test_generic_child_replay_publication_race_preserves_local_execution_value(
    tmp_path: Path,
) -> None:
    context = DynamicChildReplayPhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "cache")
    )
    identity = _identity("race")
    codec = _Codec()
    local = {"child": "race", "runtime_telemetry": {"forwards": 3}}

    def produce_and_lose_race():
        replay_sha256 = codec.authenticate(local, identity)
        with context.cache.begin(identity.phase_key) as transaction:
            codec_result = codec.write(
                dict(local),
                transaction.payload_path,
                identity,
            )
            transaction.complete(
                result=replay_cache_module._completion_result(
                    identity,
                    replay_sha256,
                    codec_result,
                )
            )
        return local

    raced = load_or_build_dynamic_child_replay_phase(
        context,
        identity,
        codec=codec,
        producer=produce_and_lose_race,
    )

    assert raced.cache_hit is True
    assert raced.producer_invoked is True
    assert raced.publication_race_lost is True
    assert raced.replay is local
    assert raced.replay_sha256 == codec.authenticate(local, identity)
