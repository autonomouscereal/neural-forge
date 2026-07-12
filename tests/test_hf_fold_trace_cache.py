from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from neural_forge.hf_fold_trace_cache import (  # noqa: E402
    BATCH_RECORD_FILENAME,
    COMPLETION_FILENAME,
    MANIFEST_FILENAME,
    TRACE_FILENAME,
    ParentTraceCache,
    ParentTraceCacheCorruptionError,
    ParentTraceCaptureConfig,
    ParentTraceCaptureError,
    build_parent_trace_cache,
    fingerprint_parent_model,
    open_parent_trace_cache,
)


class _TinyConfig:
    def __init__(self, *, width: int, layers: int, vocab_size: int) -> None:
        self.width = width
        self.layers = layers
        self.vocab_size = vocab_size

    def to_dict(self):
        return {
            "model_type": "tiny-hf-parent",
            "hidden_size": self.width,
            "num_hidden_layers": self.layers,
            "vocab_size": self.vocab_size,
        }


class _TinyHFParent(torch.nn.Module):
    def __init__(self, *, width: int = 6, layers: int = 2, vocab_size: int = 17):
        super().__init__()
        self.config = _TinyConfig(width=width, layers=layers, vocab_size=vocab_size)
        self.embedding = torch.nn.Embedding(vocab_size, width)
        self.dropout = torch.nn.Dropout(p=0.75)
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(width, width) for _ in range(layers)
        )
        self.lm_head = torch.nn.Linear(width, vocab_size, bias=False)
        self.forward_calls = 0
        self.forward_markers: list[int] = []
        self.observed_training: list[bool] = []
        self.observed_grad_enabled: list[bool] = []
        self.fail_on_token: int | None = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids=None,
        *,
        use_cache,
        return_dict,
        output_attentions,
        output_hidden_states,
    ):
        self.forward_calls += 1
        marker = int(input_ids[0, 0].item())
        self.forward_markers.append(marker)
        self.observed_training.append(bool(self.training))
        self.observed_grad_enabled.append(torch.is_grad_enabled())
        if self.fail_on_token == marker:
            raise RuntimeError("injected parent failure")
        assert use_cache is False
        assert return_dict is True
        assert output_attentions is False
        assert output_hidden_states is True

        hidden = self.embedding(input_ids)
        hidden = hidden * attention_mask.to(hidden.dtype).unsqueeze(-1)
        if position_ids is not None:
            hidden = hidden + position_ids.to(hidden.dtype).unsqueeze(-1) / 100.0
        hidden = self.dropout(hidden)
        hidden_states = [hidden]
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
            hidden_states.append(hidden)
        return SimpleNamespace(
            logits=self.lm_head(hidden),
            hidden_states=tuple(hidden_states),
            past_key_values=None,
        )


class _StateMutatingParent(_TinyHFParent):
    def __init__(self):
        super().__init__()
        self.register_buffer("forward_counter", torch.zeros((), dtype=torch.int64))

    def forward(self, *args, **kwargs):
        self.forward_counter.add_(1)
        return super().forward(*args, **kwargs)


class _MissingHiddenStatesParent(_TinyHFParent):
    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output.hidden_states = None
        return output


def _parent(*, seed: int = 101, **kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _TinyHFParent(**kwargs)


def _batches():
    return [
        {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "valid_token_mask": torch.tensor(
                [[True, True, False, False], [True, True, False, False]]
            ),
            "position_ids": torch.tensor([[0, 1, 2, 0], [0, 1, 0, 0]]),
        },
        {
            "input_ids": torch.tensor([[6, 7, 8]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "position_ids": torch.tensor([[3, 4, 5]]),
        },
    ]


def _single_batch(marker: int):
    return {
        "input_ids": torch.tensor([[marker, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }


def _module_modes(model):
    return {name: bool(module.training) for name, module in model.named_modules()}


def _direct_outputs(model, batches):
    modes = [(module, bool(module.training)) for module in model.modules()]
    model.eval()
    outputs = []
    try:
        for batch in batches:
            inputs = {
                name: tensor
                for name, tensor in batch.items()
                if name != "valid_token_mask"
            }
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            with torch.inference_mode():
                output = model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=True,
                )
            outputs.append(
                SimpleNamespace(
                    logits=output.logits.detach().cpu().clone(),
                    hidden_states=tuple(
                        value.detach().cpu().clone() for value in output.hidden_states
                    ),
                )
            )
    finally:
        for module, was_training in modes:
            module.training = was_training
    return outputs


def _clear_forward_telemetry(model):
    model.forward_calls = 0
    model.forward_markers.clear()
    model.observed_training.clear()
    model.observed_grad_enabled.clear()


def test_topk_cache_is_exact_one_pass_reusable_and_restores_process_state(
    tmp_path,
) -> None:
    model = _parent()
    batches = _batches()
    expected = _direct_outputs(model, batches)
    _clear_forward_telemetry(model)
    model.train()
    model.layers[0].eval()
    modes_before = _module_modes(model)
    rng_before = torch.random.get_rng_state().clone()
    algorithms_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    config = ParentTraceCaptureConfig(
        hidden_state_indices=(0, 2),
        logits_mode="topk",
        top_k=3,
        seed=90210,
    )

    cache = build_parent_trace_cache(tmp_path / "cache", model, batches, config=config)

    assert cache.is_complete is True
    assert cache.path.name == cache.cache_key
    assert len(cache.cache_key) == 64
    assert len(cache.parent_fingerprint) == 64
    assert len(cache.calibration_fingerprint) == 64
    assert len(cache) == 2
    assert model.forward_calls == 2
    assert model.forward_markers == [1, 6]
    assert model.observed_training == [False, False]
    assert model.observed_grad_enabled == [False, False]
    assert _module_modes(model) == modes_before
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.are_deterministic_algorithms_enabled() == algorithms_before
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only_before

    manifest = cache.manifest
    assert manifest["runtime"]["local_files_only"] is True
    assert manifest["runtime"]["network_access_performed"] is False
    assert manifest["storage"]["committed_batches_immutable"] is True
    assert manifest["capture"]["hidden_state_indices"] == [0, 2]
    assert len(manifest["capture"]["selection_sha256"]) == 64
    assert (cache.path / MANIFEST_FILENAME).is_file()
    assert (cache.path / COMPLETION_FILENAME).is_file()

    for index, loaded in enumerate(cache.iter_batches()):
        source = batches[index]
        assert torch.equal(loaded.input_ids, source["input_ids"])
        assert torch.equal(loaded.attention_mask, source["attention_mask"])
        expected_valid = source.get(
            "valid_token_mask", source["attention_mask"].to(torch.bool)
        )
        assert torch.equal(loaded.valid_token_mask, expected_valid)
        assert tuple(loaded.hidden_states) == (0, 2)
        assert loaded.layer_boundary_states.keys() == loaded.hidden_states.keys()
        assert torch.equal(loaded.hidden_states[0], expected[index].hidden_states[0])
        assert torch.equal(loaded.hidden_states[2], expected[index].hidden_states[2])
        values, indices = torch.topk(expected[index].logits, k=3, dim=-1)
        assert torch.equal(loaded.topk_values, values)
        assert torch.equal(loaded.topk_indices, indices)
        torch.testing.assert_close(
            loaded.logits_logsumexp,
            torch.logsumexp(expected[index].logits.float(), dim=-1),
        )
        assert loaded.logits is None
        assert set(loaded.model_inputs) == {
            "attention_mask",
            "input_ids",
            "position_ids",
        }
        batch_path = cache.path / "batches" / f"batch-{index:06d}"
        assert {item.name for item in batch_path.iterdir()} == {
            BATCH_RECORD_FILENAME,
            TRACE_FILENAME,
        }

    trace_before = (
        cache.path / "batches" / "batch-000000" / TRACE_FILENAME
    ).read_bytes()
    reused = build_parent_trace_cache(
        tmp_path / "cache",
        model,
        _batches(),
        config=config,
    )
    assert reused.path == cache.path
    assert model.forward_calls == 2
    assert (
        cache.path / "batches" / "batch-000000" / TRACE_FILENAME
    ).read_bytes() == trace_before
    assert open_parent_trace_cache(cache.path).cache_key == cache.cache_key


def test_full_logits_and_all_hidden_boundaries_are_architecture_neutral(
    tmp_path,
) -> None:
    model = _parent(seed=202, width=5, layers=3, vocab_size=11)
    batches = [_single_batch(3)]
    expected = _direct_outputs(model, batches)[0]
    _clear_forward_telemetry(model)

    cache = build_parent_trace_cache(
        tmp_path,
        model,
        batches,
        config=ParentTraceCaptureConfig(logits_mode="full"),
    )
    loaded = cache.load_batch(0)

    assert model.forward_calls == 1
    assert tuple(loaded.hidden_states) == (0, 1, 2, 3)
    for index, value in enumerate(expected.hidden_states):
        assert torch.equal(loaded.hidden_states[index], value)
    assert torch.equal(loaded.logits, expected.logits)
    assert loaded.topk_values is None
    assert loaded.topk_indices is None
    assert loaded.logits_logsumexp is None
    with pytest.raises(TypeError):
        loaded.hidden_states[0] = torch.empty(0)


def test_interrupted_capture_resumes_without_recomputing_committed_batches(
    tmp_path,
) -> None:
    model = _parent(seed=303)
    batches = [_single_batch(1), _single_batch(7), _single_batch(9)]
    model.fail_on_token = 7
    model.train()
    modes_before = _module_modes(model)
    config = ParentTraceCaptureConfig(hidden_state_indices=(1,), top_k=2)

    with pytest.raises(ParentTraceCaptureError, match="calibration batch 1"):
        build_parent_trace_cache(tmp_path, model, batches, config=config)

    assert model.forward_markers == [1, 7]
    assert _module_modes(model) == modes_before
    cache_paths = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(cache_paths) == 1
    partial = ParentTraceCache.open(
        cache_paths[0],
        verify=True,
        require_complete=False,
    )
    assert partial.is_complete is False
    assert [path.name for path in (partial.path / "batches").iterdir()] == [
        "batch-000000"
    ]
    assert partial.load_batch(0).input_ids[0, 0].item() == 1

    metadata_stage = partial.path / f".{COMPLETION_FILENAME}.stage-{'a' * 32}"
    metadata_stage.write_bytes(b"interrupted completion")
    batch_stage = partial.path / "batches" / f".batch-000001.stage-1-{'b' * 32}"
    batch_stage.mkdir()
    (batch_stage / "partial").write_bytes(b"interrupted batch")
    model.fail_on_token = None
    resumed = build_parent_trace_cache(tmp_path, model, batches, config=config)

    assert resumed.path == partial.path
    assert resumed.is_complete is True
    assert not metadata_stage.exists()
    assert not batch_stage.exists()
    assert model.forward_markers == [1, 7, 7, 9]
    assert [batch.input_ids[0, 0].item() for batch in resumed.iter_batches()] == [
        1,
        7,
        9,
    ]


def test_corruption_is_detected_and_never_recomputed_or_overwritten(tmp_path) -> None:
    model = _parent(seed=404)
    batches = [_single_batch(4)]
    config = ParentTraceCaptureConfig(hidden_state_indices=(0,), top_k=2)
    cache = build_parent_trace_cache(tmp_path, model, batches, config=config)
    assert model.forward_calls == 1
    trace_path = cache.path / "batches" / "batch-000000" / TRACE_FILENAME
    corrupted = bytearray(trace_path.read_bytes())
    corrupted[-1] ^= 0x01
    trace_path.write_bytes(corrupted)

    with pytest.raises(
        ParentTraceCacheCorruptionError,
        match="safetensors file is truncated or corrupt",
    ):
        open_parent_trace_cache(cache.path)
    with pytest.raises(ParentTraceCacheCorruptionError):
        build_parent_trace_cache(tmp_path, model, batches, config=config)

    assert model.forward_calls == 1
    assert trace_path.read_bytes() == corrupted


def test_parent_calibration_and_selection_fingerprints_separate_caches(
    tmp_path,
) -> None:
    model = _parent(seed=505)
    parent_before = fingerprint_parent_model(model)
    first = build_parent_trace_cache(
        tmp_path,
        model,
        [_single_batch(1)],
        config=ParentTraceCaptureConfig(hidden_state_indices=(0,), top_k=1),
    )
    second = build_parent_trace_cache(
        tmp_path,
        model,
        [_single_batch(2)],
        config=ParentTraceCaptureConfig(hidden_state_indices=(0,), top_k=1),
    )
    third = build_parent_trace_cache(
        tmp_path,
        model,
        [_single_batch(1)],
        config=ParentTraceCaptureConfig(hidden_state_indices=(1,), top_k=1),
    )

    assert len({first.path, second.path, third.path}) == 3
    assert first.parent_fingerprint == second.parent_fingerprint
    assert first.calibration_fingerprint != second.calibration_fingerprint
    assert (
        first.manifest["capture"]["selection_sha256"]
        != third.manifest["capture"]["selection_sha256"]
    )
    assert fingerprint_parent_model(model) == parent_before
    assert first.verify() is first

    with torch.no_grad():
        model.embedding.weight[0, 0].add_(1)
    changed_parent = build_parent_trace_cache(
        tmp_path,
        model,
        [_single_batch(1)],
        config=ParentTraceCaptureConfig(hidden_state_indices=(0,), top_k=1),
    )
    assert changed_parent.path not in {first.path, second.path, third.path}
    assert changed_parent.parent_fingerprint != first.parent_fingerprint
    assert open_parent_trace_cache(first.path).cache_key == first.cache_key


def test_parent_state_mutation_fails_before_batch_publication(tmp_path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(606)
        model = _StateMutatingParent()
    model.train()
    modes_before = _module_modes(model)

    with pytest.raises(ParentTraceCaptureError, match="state or configuration changed"):
        build_parent_trace_cache(
            tmp_path,
            model,
            [_single_batch(1)],
            config=ParentTraceCaptureConfig(hidden_state_indices=(0,), top_k=1),
        )

    assert _module_modes(model) == modes_before
    cache_path = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert not list((cache_path / "batches").iterdir())
    assert not (cache_path / COMPLETION_FILENAME).exists()


def test_invalid_inputs_and_missing_hidden_states_fail_closed_before_completion(
    tmp_path,
) -> None:
    model = _parent(seed=707)
    with pytest.raises(ParentTraceCaptureError, match="reserved forward controls"):
        build_parent_trace_cache(
            tmp_path / "reserved",
            model,
            [{**_single_batch(1), "use_cache": torch.tensor(False)}],
        )
    assert model.forward_calls == 0

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(808)
        missing = _MissingHiddenStatesParent()
    with pytest.raises(ParentTraceCaptureError, match="hidden_states"):
        build_parent_trace_cache(
            tmp_path / "missing",
            missing,
            [_single_batch(1)],
            config=ParentTraceCaptureConfig(hidden_state_indices=(0,), top_k=1),
        )
    cache_path = next((tmp_path / "missing").iterdir())
    assert not (cache_path / COMPLETION_FILENAME).exists()


def test_json_records_are_canonical_and_self_hashed(tmp_path) -> None:
    cache = build_parent_trace_cache(
        tmp_path,
        _parent(seed=909),
        [_single_batch(1)],
        config=ParentTraceCaptureConfig(hidden_state_indices=(0,), top_k=1),
    )
    paths = [
        cache.path / MANIFEST_FILENAME,
        cache.path / COMPLETION_FILENAME,
        cache.path / "batches" / "batch-000000" / BATCH_RECORD_FILENAME,
    ]
    for path in paths:
        raw = path.read_bytes()
        parsed = json.loads(raw)
        expected = (
            json.dumps(
                parsed,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        assert raw == expected
    record = json.loads(paths[-1].read_text(encoding="utf-8"))
    assert len(record["record_sha256"]) == 64
    assert all(len(item["sha256"]) == 64 for item in record["tensors"].values())
