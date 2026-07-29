from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceCaptureError,
    capture_lfm2_child_replay,
    fingerprint_lfm2_parent,
)
from test_dynamic_lfm2_heterogeneous_budgeted_materialization import (  # noqa: E402
    _heterogeneous_case,
)


def _materialized_child():
    parent, graph, _tokenizer, synthesis, _candidate, request = _heterogeneous_case()
    return synthesis.materialize(
        request,
        {"parent": parent},
        {"parent": graph},
        None,
    ).model


def _batch():
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
        "forced_decode_ids": torch.tensor([[5, 6]], dtype=torch.long),
    }


def _config() -> Lfm2CircuitTraceCaptureConfig:
    return Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=3,
        max_decode_steps=4,
        max_capture_bytes=64 * 1024 * 1024,
        decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
    )


def _admit_exact_installed_transformers_runtime(monkeypatch) -> None:
    from transformers import masking_utils
    from transformers.models.lfm2 import modeling_lfm2

    runtime_identity = (
        trace_cache._package_version("transformers"),
        trace_cache._source_file_sha256(
            Path(modeling_lfm2.__file__).resolve(),
            "test Transformers LFM2",
        ),
        trace_cache._source_file_sha256(
            Path(masking_utils.__file__).resolve(),
            "test Transformers causal masking",
        ),
    )
    monkeypatch.setattr(
        trace_cache,
        "_SINGLE_PASS_CAUSAL_RUNTIME_ALLOWLIST",
        frozenset({runtime_identity}),
    )


def _capture_pair(child, device: str):
    source = _batch()
    config = _config()
    canonical = capture_lfm2_child_replay(
        child,
        [source],
        selected_layer_indices=[0, 1, 2, 3],
        config=config,
        input_device=device,
    )
    single_pass = capture_lfm2_child_replay(
        child,
        [source],
        selected_layer_indices=[0, 1, 2, 3],
        config=config,
        input_device=device,
        replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        causal_equivalence_reference=canonical,
    )
    return canonical, single_pass


def _assert_trace_equivalent(canonical, single_pass) -> None:
    expected = canonical.load_batch(0)
    actual = single_pass.load_batch(0)
    assert actual.phase_order == expected.phase_order
    assert set(actual.tensors) == set(expected.tensors)
    for name in sorted(expected.tensors):
        left = expected.tensors[name]
        right = actual.tensors[name]
        assert left.shape == right.shape, name
        assert left.dtype == right.dtype, name
        if left.is_floating_point():
            torch.testing.assert_close(right, left, rtol=1e-5, atol=1e-6, msg=name)
        else:
            assert torch.equal(right, left), name
        if name.endswith(".logits"):
            assert torch.equal(
                torch.argmax(right, dim=-1),
                torch.argmax(left, dim=-1),
            ), name
    equivalence = single_pass.metadata["causal_equivalence"]
    assert equivalence["reference_supplied"] is True
    assert equivalence["tolerance_equivalent"] is True
    assert equivalence["compared_tensors"] == len(actual.tensors)
    assert equivalence["compared_logits"] == len(actual.phase_order)


def test_heterogeneous_cpu_single_pass_is_strictly_close_and_descriptor_bound(
    monkeypatch,
) -> None:
    _admit_exact_installed_transformers_runtime(monkeypatch)
    child = _materialized_child()
    state_before = fingerprint_lfm2_parent(child)
    canonical, single_pass = _capture_pair(child, "cpu")

    _assert_trace_equivalent(canonical, single_pass)
    assert fingerprint_lfm2_parent(child) == state_before
    contract = single_pass.load_batch(0).capture[
        "causal_equivalence_contract"
    ]
    binding = contract["heterogeneous_runtime_binding"]
    assert contract["runtime_variant"] == "sealed_heterogeneous_dynamic_lfm2"
    assert binding["descriptor_sha256"] == (
        child.config.neural_forge_dynamic_physical_layout_sha256
    )
    assert binding["floating_dtypes"] == ["float32"]
    assert binding["materializer_source_allowlisted"] is True
    assert binding["exact_runtime_class_methods_reauthenticated"] is True
    assert single_pass.metadata["replay_benchmark"] == {
        "batch_count": 1,
        "phase_count": 3,
        "actual_child_forwards": 1,
        "canonical_prefix_child_forwards": 3,
        "saved_child_forwards": 2,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_heterogeneous_cuda_single_pass_matches_canonical_trace(monkeypatch) -> None:
    _admit_exact_installed_transformers_runtime(monkeypatch)
    child = _materialized_child().to(device="cuda")
    state_before = fingerprint_lfm2_parent(child)
    canonical, single_pass = _capture_pair(child, "cuda")

    _assert_trace_equivalent(canonical, single_pass)
    assert fingerprint_lfm2_parent(child) == state_before
    contract = single_pass.load_batch(0).capture[
        "causal_equivalence_contract"
    ]
    binding = contract["heterogeneous_runtime_binding"]
    assert contract["attention_backend"] in {"eager", "sdpa"}
    assert binding["floating_dtypes"] == ["float32"]
    assert binding["execution_sha256"] == contract["execution_sha256"]


def test_heterogeneous_single_pass_rejects_unrecognized_operator_class(
    monkeypatch,
) -> None:
    _admit_exact_installed_transformers_runtime(monkeypatch)
    child = _materialized_child()
    operator = child.model.layers[1].self_attn
    authentic_class = type(operator)

    class UnrecognizedAttention(authentic_class):
        pass

    operator.__class__ = UnrecognizedAttention
    try:
        with pytest.raises(Lfm2CircuitTraceCaptureError):
            capture_lfm2_child_replay(
                child,
                [_batch()],
                selected_layer_indices=[0, 1, 2, 3],
                config=_config(),
                input_device="cpu",
                replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
            )
    finally:
        operator.__class__ = authentic_class


def test_single_pass_rejects_causal_circuit_evidence_before_forward(
    monkeypatch,
) -> None:
    child = _materialized_child()
    config = _config()
    object.__setattr__(config, "capture_causal_circuit_evidence", True)
    authentic_forward = child.forward
    forwarded: list[bool] = []

    def recording_forward(*args, **kwargs):
        forwarded.append(True)
        return authentic_forward(*args, **kwargs)

    monkeypatch.setattr(child, "forward", recording_forward)
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="does not yet support causal-circuit evidence",
    ):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[0, 1, 2, 3],
            config=config,
            input_device="cpu",
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        )
    assert forwarded == []


def test_heterogeneous_single_pass_rejects_tampered_native_subclass_method(
    monkeypatch,
) -> None:
    _admit_exact_installed_transformers_runtime(monkeypatch)
    child = _materialized_child()
    runtime_class = type(child.model.layers[1].self_attn)
    authentic_forward = runtime_class.forward
    forwarded = []

    def wrapped_forward(self, *args, **kwargs):
        forwarded.append(True)
        return authentic_forward(self, *args, **kwargs)

    monkeypatch.setattr(runtime_class, "forward", wrapped_forward)
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="physical runtime authentication failed",
    ):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[0, 1, 2, 3],
            config=_config(),
            input_device="cpu",
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        )
    assert forwarded == []


def test_heterogeneous_single_pass_rejects_unallowlisted_materializer_source(
    monkeypatch,
) -> None:
    _admit_exact_installed_transformers_runtime(monkeypatch)
    child = _materialized_child()
    authentic_source_hash = trace_cache._source_file_sha256

    def tampered_source_hash(path, name):
        if name == "Neural Forge heterogeneous materializer":
            return "0" * 64
        return authentic_source_hash(path, name)

    monkeypatch.setattr(trace_cache, "_source_file_sha256", tampered_source_hash)
    with pytest.raises(
        Lfm2CircuitTraceCaptureError,
        match="has not certified this Neural Forge materializer source revision",
    ):
        capture_lfm2_child_replay(
            child,
            [_batch()],
            selected_layer_indices=[0, 1, 2, 3],
            config=_config(),
            input_device="cpu",
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        )
