from __future__ import annotations

import concurrent.futures
import importlib
import shutil
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

checkpoint = importlib.import_module(
    "neural_forge.hf_lfm2_child_replay_batch_checkpoint"
)
trace_cache = importlib.import_module("neural_forge.hf_lfm2_circuit_trace_cache")
_dynamic_cache = importlib.import_module(
    "neural_forge.dynamic_child_replay_phase_cache"
)
_fold_cache = importlib.import_module("neural_forge.dynamic_fold_phase_cache")
_phase_cache = importlib.import_module("neural_forge.hf_lfm2_child_replay_phase_cache")

DynamicChildReplayPhaseCacheContext = _dynamic_cache.DynamicChildReplayPhaseCacheContext
DynamicFoldPhaseCache = _fold_cache.DynamicFoldPhaseCache
FoldPhaseEventLog = _fold_cache.FoldPhaseEventLog
FoldPhaseCacheIntegrityError = _fold_cache.FoldPhaseCacheIntegrityError
describe_lfm2_child_replay_phase = _phase_cache.describe_lfm2_child_replay_phase
load_or_capture_lfm2_child_replay_phase = (
    _phase_cache.load_or_capture_lfm2_child_replay_phase
)
FULL_PREFIX_CANONICAL_TRACE_MODE = trace_cache.FULL_PREFIX_CANONICAL_TRACE_MODE
Lfm2CircuitTraceCaptureConfig = trace_cache.Lfm2CircuitTraceCaptureConfig
capture_lfm2_child_replay = trace_cache.capture_lfm2_child_replay


def _model(*, seed: int = 801):
    config = transformers.Lfm2Config(
        vocab_size=31,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


def _batch(*, prompt=(1, 2, 3, 4), forced=(5, 6)):
    input_ids = torch.tensor([list(prompt)], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "forced_decode_ids": torch.tensor([list(forced)], dtype=torch.long),
    }


def _config() -> Lfm2CircuitTraceCaptureConfig:
    return Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=2,
        max_decode_steps=4,
        decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
    )


def _assert_replays_are_byte_identical(reference, candidate) -> None:
    assert candidate.replay_sha256 == reference.replay_sha256
    assert trace_cache._canonical_json(candidate.metadata) == (
        trace_cache._canonical_json(reference.metadata)
    )
    assert len(candidate) == len(reference)
    for index in range(len(reference)):
        expected = reference.load_batch(index)
        observed = candidate.load_batch(index)
        assert trace_cache._canonical_json(observed.record) == (
            trace_cache._canonical_json(expected.record)
        )
        assert set(observed.tensors) == set(expected.tensors)
        for name, value in expected.tensors.items():
            assert observed.tensors[name].dtype == value.dtype
            assert torch.equal(observed.tensors[name], value)


def _sealed_json_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name in {"request.json", "replay.json", "record.json"}
    }


def _request(child, batches):
    return describe_lfm2_child_replay_phase(
        child,
        batches,
        selected_layer_indices=(0, 1),
        config=_config(),
        input_device="cpu",
    )


def test_interrupted_capture_resumes_only_missing_batch_and_is_byte_identical(
    tmp_path: Path,
    monkeypatch,
) -> None:
    child = _model(seed=810)
    batches = (_batch(), _batch(prompt=(7, 8, 9), forced=(10,)))
    uninterrupted = load_or_capture_lfm2_child_replay_phase(
        DynamicChildReplayPhaseCacheContext(
            cache=DynamicFoldPhaseCache(tmp_path / "uninterrupted")
        ),
        child,
        batches,
        selected_layer_indices=(0, 1),
        config=_config(),
        input_device="cpu",
    )
    root = tmp_path / "replays"
    request = _request(child, batches)
    first_checkpoint = checkpoint.describe_lfm2_child_replay_batch_checkpoint(
        request,
        0,
    )
    second_checkpoint = checkpoint.describe_lfm2_child_replay_batch_checkpoint(
        request,
        1,
    )
    monkeypatch.setattr(trace_cache, "_SEAL_PIPELINE_ENABLED", False)

    def interrupt_after_first(stage, details):
        if (
            stage == checkpoint.LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_PUBLISHED_STAGE
            and details["batch_index"] == 0
        ):
            raise RuntimeError("synthetic interruption after batch zero")

    interrupted_context = DynamicChildReplayPhaseCacheContext(
        cache=DynamicFoldPhaseCache(root),
        progress=interrupt_after_first,
    )
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        checkpoint.load_or_capture_lfm2_child_replay_phase_resumable(
            interrupted_context,
            child,
            batches,
            selected_layer_indices=(0, 1),
            config=_config(),
            input_device="cpu",
        )
    assert not interrupted_context.cache.entry_path(request.phase_key).exists()
    assert interrupted_context.cache.entry_path(first_checkpoint.phase_key).exists()
    assert not interrupted_context.cache.entry_path(
        second_checkpoint.phase_key
    ).exists()

    replayed_indices = []
    original_replay_batch = trace_cache._replay_child_batch

    def observed_replay(*args, **kwargs):
        replayed_indices.append(args[4].index)
        return original_replay_batch(*args, **kwargs)

    monkeypatch.setattr(trace_cache, "_replay_child_batch", observed_replay)
    resumed = checkpoint.load_or_capture_lfm2_child_replay_phase_resumable(
        DynamicChildReplayPhaseCacheContext(cache=DynamicFoldPhaseCache(root)),
        child,
        batches,
        selected_layer_indices=(0, 1),
        config=_config(),
        input_device="cpu",
    )
    assert replayed_indices == [1]
    timing = resumed.replay.capture_timing["batch_checkpoint_resume"]
    assert timing["resumed_batch_indices"] == [0]
    assert timing["captured_batch_indices"] == [1]
    assert timing["current_invocation_child_forwards"] == 2
    _assert_replays_are_byte_identical(uninterrupted.replay, resumed.replay)
    assert _sealed_json_bytes(uninterrupted.entry.path) == _sealed_json_bytes(
        resumed.entry.path
    )


def test_tampered_completed_batch_fails_before_any_new_child_forward(
    tmp_path: Path,
    monkeypatch,
) -> None:
    child = _model(seed=820)
    batches = (_batch(), _batch(prompt=(11, 12, 13), forced=(14,)))
    request = _request(child, batches)
    batch_request = checkpoint.describe_lfm2_child_replay_batch_checkpoint(
        request,
        0,
    )
    root = tmp_path / "replays"
    monkeypatch.setattr(trace_cache, "_SEAL_PIPELINE_ENABLED", False)

    def interrupt(stage, details):
        if (
            stage == checkpoint.LFM2_CHILD_REPLAY_BATCH_CHECKPOINT_PUBLISHED_STAGE
            and details["batch_index"] == 0
        ):
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        checkpoint.load_or_capture_lfm2_child_replay_phase_resumable(
            DynamicChildReplayPhaseCacheContext(
                cache=DynamicFoldPhaseCache(root),
                progress=interrupt,
            ),
            child,
            batches,
            selected_layer_indices=(0, 1),
            config=_config(),
            input_device="cpu",
        )
    record_path = (
        DynamicFoldPhaseCache(root).entry_path(batch_request.phase_key)
        / "payload"
        / "record.json"
    )
    record_path.write_bytes(record_path.read_bytes() + b" ")

    def forbidden_replay(*_args, **_kwargs):
        raise AssertionError("tampered checkpoint permitted a child forward")

    monkeypatch.setattr(trace_cache, "_replay_child_batch", forbidden_replay)
    with pytest.raises(FoldPhaseCacheIntegrityError, match="payload differs"):
        checkpoint.load_or_capture_lfm2_child_replay_phase_resumable(
            DynamicChildReplayPhaseCacheContext(cache=DynamicFoldPhaseCache(root)),
            child,
            batches,
            selected_layer_indices=(0, 1),
            config=_config(),
            input_device="cpu",
        )


def test_complete_rebuild_from_all_batch_checkpoints_runs_zero_forwards(
    tmp_path: Path,
    monkeypatch,
) -> None:
    child = _model(seed=830)
    batches = (_batch(), _batch(prompt=(17, 18, 20), forced=(19,)))
    root = tmp_path / "replays"
    context = DynamicChildReplayPhaseCacheContext(cache=DynamicFoldPhaseCache(root))
    cold = checkpoint.load_or_capture_lfm2_child_replay_phase_resumable(
        context,
        child,
        batches,
        selected_layer_indices=(0, 1),
        config=_config(),
        input_device="cpu",
    )
    request = _request(child, batches)
    shutil.rmtree(context.cache.entry_path(request.phase_key))

    def forbidden_replay(*_args, **_kwargs):
        raise AssertionError("complete rebuild repeated a child forward")

    monkeypatch.setattr(trace_cache, "_replay_child_batch", forbidden_replay)
    rebuilt = checkpoint.load_or_capture_lfm2_child_replay_phase_resumable(
        DynamicChildReplayPhaseCacheContext(cache=DynamicFoldPhaseCache(root)),
        child,
        batches,
        selected_layer_indices=(0, 1),
        config=_config(),
        input_device="cpu",
    )
    assert rebuilt.producer_invoked is True
    timing = rebuilt.replay.capture_timing["batch_checkpoint_resume"]
    assert timing["resumed_batch_count"] == 2
    assert timing["captured_batch_count"] == 0
    assert timing["current_invocation_child_forwards"] == 0
    _assert_replays_are_byte_identical(cold.replay, rebuilt.replay)


def test_batch_checkpoint_key_rejects_foreign_child_and_batch_contract() -> None:
    batches = (_batch(), _batch(prompt=(7, 8, 9), forced=(10,)))
    baseline = checkpoint.describe_lfm2_child_replay_batch_checkpoint(
        _request(_model(seed=840), batches),
        0,
    )
    changed_child = checkpoint.describe_lfm2_child_replay_batch_checkpoint(
        _request(_model(seed=841), batches),
        0,
    )
    changed_batch = checkpoint.describe_lfm2_child_replay_batch_checkpoint(
        _request(
            _model(seed=840),
            (_batch(forced=(5, 7)), batches[1]),
        ),
        0,
    )
    reordered = checkpoint.describe_lfm2_child_replay_batch_checkpoint(
        _request(_model(seed=840), tuple(reversed(batches))),
        1,
    )
    assert changed_child.phase_key.phase_key_sha256 != (
        baseline.phase_key.phase_key_sha256
    )
    assert changed_batch.phase_key.phase_key_sha256 != (
        baseline.phase_key.phase_key_sha256
    )
    assert reordered.phase_key.phase_key_sha256 != (baseline.phase_key.phase_key_sha256)


def test_concurrent_identical_batch_publishers_authenticate_one_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    child = _model(seed=850)
    batches = (_batch(),)
    request = _request(child, batches)
    batch_request = checkpoint.describe_lfm2_child_replay_batch_checkpoint(
        request,
        0,
    )
    batch = capture_lfm2_child_replay(
        child,
        batches,
        selected_layer_indices=(0, 1),
        config=_config(),
        input_device="cpu",
    ).load_batch(0)
    context = DynamicChildReplayPhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "replays"),
    )
    write_started = threading.Event()
    release_writer = threading.Event()
    second_write_started = threading.Event()
    write_count = 0
    write_lock = threading.Lock()
    progress_events = []
    original_write = checkpoint.Lfm2ChildReplayBatchCheckpointCodec.write

    def blocking_write(self, value, payload_path, identity):
        nonlocal write_count
        with write_lock:
            write_count += 1
            ordinal = write_count
        if ordinal == 1:
            write_started.set()
            assert release_writer.wait(timeout=10.0)
        else:
            second_write_started.set()
        return original_write(self, value, payload_path, identity)

    monkeypatch.setattr(
        checkpoint.Lfm2ChildReplayBatchCheckpointCodec,
        "write",
        blocking_write,
    )
    context = DynamicChildReplayPhaseCacheContext(
        cache=context.cache,
        progress=lambda stage, _details: progress_events.append(stage),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            checkpoint._publish_batch_checkpoint,
            context,
            batch_request,
            batch,
        )
        assert write_started.wait(timeout=10.0)
        second = executor.submit(
            checkpoint._publish_batch_checkpoint,
            context,
            batch_request,
            batch,
        )
        assert second_write_started.wait(timeout=10.0)
        second_result = second.result(timeout=10.0)
        release_writer.set()
        resolved = (first.result(timeout=10.0), second_result)
    assert resolved[0].record == resolved[1].record
    assert context.cache.lookup(batch_request.phase_key) is not None
    assert "dynamic_child_replay_phase_publish_race_resumed" in progress_events


def test_orphan_locks_and_staging_cannot_block_batch_or_complete_publish(
    tmp_path: Path,
) -> None:
    child = _model(seed=855)
    batches = (_batch(),)
    complete_request = _request(child, batches)
    batch_request = checkpoint.describe_lfm2_child_replay_batch_checkpoint(
        complete_request,
        0,
    )
    cache = DynamicFoldPhaseCache(tmp_path / "replays")
    orphan_paths = []
    for request in (batch_request, complete_request):
        destination = cache.entry_path(request.phase_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lock = destination.parent / (f".{request.phase_key.phase_key_sha256}.lock")
        lock.write_bytes(b"pid=999999999\n")
        staging = destination.parent / (
            f".{request.phase_key.phase_key_sha256}.dead.staging"
        )
        (staging / "payload").mkdir(parents=True)
        (staging / "payload" / "incomplete.bin").write_bytes(b"orphan")
        orphan_paths.append((lock, staging))

    resolved = checkpoint.load_or_capture_lfm2_child_replay_phase_resumable(
        DynamicChildReplayPhaseCacheContext(cache=cache),
        child,
        batches,
        selected_layer_indices=(0, 1),
        config=_config(),
        input_device="cpu",
    )
    assert cache.lookup(batch_request.phase_key) is not None
    assert cache.lookup(complete_request.phase_key) is not None
    assert resolved.replay.replay_sha256 == resolved.replay_sha256
    # Unverifiable foreign artifacts are ignored, never stolen or trusted.
    for lock, staging in orphan_paths:
        assert lock.read_bytes() == b"pid=999999999\n"
        assert (staging / "payload" / "incomplete.bin").read_bytes() == b"orphan"


def test_lockless_resume_fails_closed_instead_of_dropping_event_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    child = _model(seed=857)

    def forbidden_replay(*_args, **_kwargs):
        raise AssertionError("unsupported audit mode executed a child forward")

    monkeypatch.setattr(trace_cache, "_replay_child_batch", forbidden_replay)
    context = DynamicChildReplayPhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "replays"),
        events=FoldPhaseEventLog(
            tmp_path / "events.jsonl",
            run_id="batch-checkpoint-audit-test",
        ),
    )
    with pytest.raises(
        checkpoint.Lfm2ChildReplayBatchCheckpointError,
        match="does not support FoldPhaseEventLog",
    ):
        checkpoint.load_or_capture_lfm2_child_replay_phase_resumable(
            context,
            child,
            (_batch(),),
            selected_layer_indices=(0, 1),
            config=_config(),
            input_device="cpu",
        )
    assert not (tmp_path / "events.jsonl").exists()


def test_sharded_resume_merges_sparse_original_indices_exactly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    child = _model(seed=860)
    batches = (
        _batch(),
        _batch(prompt=(7, 8, 9), forced=(10,)),
        _batch(prompt=(11, 12, 13, 14), forced=(15, 16)),
    )
    reference = capture_lfm2_child_replay(
        child,
        batches,
        selected_layer_indices=(0, 1),
        config=_config(),
        input_device="cpu",
    )
    root = tmp_path / "replays"
    context = DynamicChildReplayPhaseCacheContext(cache=DynamicFoldPhaseCache(root))
    complete_request = _request(child, batches)
    checkpoint._publish_batch_checkpoint(
        context,
        checkpoint.describe_lfm2_child_replay_batch_checkpoint(
            complete_request,
            1,
        ),
        reference.load_batch(1),
    )
    resumed = checkpoint.load_or_capture_lfm2_child_replay_phase_resumable(
        context,
        child,
        batches,
        selected_layer_indices=(0, 1),
        config=_config(),
        input_device="cpu",
        replay_devices=("cpu", "cpu"),
    )
    timing = resumed.replay.capture_timing["batch_checkpoint_resume"]
    assert timing["resumed_batch_indices"] == [1]
    assert timing["captured_batch_indices"] == [0, 2]
    sharding = resumed.replay.capture_timing["replay_sharding"]
    assert sharding["batch_assignment"] == [0, 1]
    assert sharding["batch_assignment_index_space"] == "planned_batch_indices"
    assert sharding["planned_batch_indices"] == [0, 2]
    assert sharding["original_batch_assignment"] == [
        {"batch_index": 0, "worker_index": 0, "batch_cost": 15},
        {"batch_index": 2, "worker_index": 1, "batch_cost": 15},
    ]
    _assert_replays_are_byte_identical(reference, resumed.replay)
