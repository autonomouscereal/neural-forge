from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache  # noqa: E402
from neural_forge.dynamic_evidence_residency import (  # noqa: E402
    DynamicEvidenceLayerReference,
    DynamicEvidenceResidencyError,
    DynamicEvidenceResidencyWindow,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    BATCH_RECORD_FILENAME,
    TRACE_FILENAME,
    Lfm2CircuitTraceCacheError,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceCorruptionError,
    build_lfm2_circuit_trace_cache,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_cpcf_installability import (  # noqa: E402
    iter_lfm2_cpcf_compiler_inputs,
)
from neural_forge.hf_lfm2_cpcf_selective_evidence import (  # noqa: E402
    Lfm2CpcfSelectiveEvidenceError,
    build_lfm2_cpcf_selective_evidence_provider,
)


def _parent() -> Any:
    config = transformers.Lfm2Config(
        vocab_size=31,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(909)
        return transformers.Lfm2ForCausalLM(config)


@pytest.fixture(scope="module")
def sealed_cache_path(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("selective-lfm2-cache")
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "forced_decode_ids": torch.tensor([[5]], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[1, 4, 3, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "forced_decode_ids": torch.tensor([[6]], dtype=torch.long),
        },
    ]
    cache = build_lfm2_circuit_trace_cache(
        root,
        _parent(),
        batches,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=1,
            capture_causal_circuit_evidence=True,
        ),
        input_device="cpu",
    )
    return cache.path


def _contains_tensor(value: Any) -> bool:
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def _assert_same_tensors(
    left: dict[str, Any] | Any,
    right: dict[str, Any] | Any,
) -> None:
    assert tuple(left) == tuple(right)
    for name in left:
        assert left[name].dtype == right[name].dtype
        assert tuple(left[name].shape) == tuple(right[name].shape)
        assert torch.equal(left[name], right[name])


def test_selective_provider_matches_every_full_phase_layer_and_final_norm(
    sealed_cache_path,
) -> None:
    full_phases = tuple(
        iter_lfm2_cpcf_compiler_inputs(
            open_lfm2_circuit_trace_cache(
                sealed_cache_path,
                verify=False,
            )
        )
    )
    provider = build_lfm2_cpcf_selective_evidence_provider(
        open_lfm2_circuit_trace_cache(
            sealed_cache_path,
            verify=False,
        )
    )
    try:
        catalog = provider.catalog
        assert catalog.layer_count == 2
        assert catalog.phase_count == len(full_phases) == 4
        assert _contains_tensor(catalog.as_dict()) is False
        assert [
            reference.existing_phase_evidence_sha256
            for reference in catalog.phase_references
        ] == [phase.evidence_sha256 for phase in full_phases]

        for dynamic_reference in provider.dynamic_catalog.references:
            payload = provider.load_layer_payload(dynamic_reference)
            try:
                receipt = provider.authenticate_layer_payload(
                    dynamic_reference,
                    payload,
                )
                assert receipt.evidence_sha256 == dynamic_reference.evidence_sha256
                assert receipt.logical_bytes == dynamic_reference.logical_bytes
                assert len(payload.phases) == len(full_phases)
                expected_bytes = 0
                for phase_index, phase_payload in enumerate(payload.phases):
                    actual = phase_payload.evidence
                    expected = full_phases[phase_index].layers[
                        dynamic_reference.layer_index
                    ]
                    assert actual.evidence_sha256 == expected.evidence_sha256
                    assert actual.operator_kind == expected.operator_kind
                    _assert_same_tensors(actual.tensors, expected.tensors)
                    expected_bytes += sum(
                        int(value.numel() * value.element_size())
                        for value in expected.tensors.values()
                    )
                assert payload.logical_bytes == expected_bytes
            finally:
                provider.release_layer_payload(
                    dynamic_reference,
                    payload,
                )
            assert payload.released is True
            with pytest.raises(
                Lfm2CpcfSelectiveEvidenceError,
                match="released",
            ):
                _ = payload.phases

        final_norm = provider.load_final_norm_payload()
        try:
            assert (
                provider.authenticate_final_norm_payload(final_norm)
                == catalog.final_norm_evidence_sha256
            )
            assert len(final_norm.phases) == len(full_phases)
            for phase_index, phase_payload in enumerate(final_norm.phases):
                _assert_same_tensors(
                    phase_payload.tensors,
                    full_phases[phase_index].final_norm_tensors,
                )
        finally:
            provider.release_final_norm_payload(final_norm)
        assert final_norm.released is True

        maximum_bytes = sum(
            reference.logical_bytes
            for reference in provider.dynamic_catalog.references
        )
        with DynamicEvidenceResidencyWindow(
            provider.dynamic_catalog,
            provider,
            maximum_resident_bytes=maximum_bytes,
        ) as window:
            current = window.start()
            assert current.layer_index == 0
            assert window.wait_for_prefetch() == 1
            current = window.advance(completed_layer_index=0)
            assert current is not None and current.layer_index == 1
            assert window.advance(completed_layer_index=1) is None
            assert window.peak_occupied_layers == 2
            assert window.peak_resident_bytes == maximum_bytes
    finally:
        provider.close()


def test_selective_session_hashes_each_trace_once_and_never_rehashes_per_layer(
    sealed_cache_path,
    monkeypatch,
) -> None:
    trace_hashes: list[Path] = []
    original = trace_cache._file_sha256

    def counted(path: Path) -> str:
        if Path(path).name == TRACE_FILENAME:
            trace_hashes.append(Path(path))
        return original(path)

    monkeypatch.setattr(trace_cache, "_file_sha256", counted)
    provider = build_lfm2_cpcf_selective_evidence_provider(
        open_lfm2_circuit_trace_cache(
            sealed_cache_path,
            verify=False,
        )
    )
    try:
        assert len(trace_hashes) == 2
        for reference in provider.dynamic_catalog.references:
            payload = provider.load_layer_payload(reference)
            provider.authenticate_layer_payload(reference, payload)
            provider.release_layer_payload(reference, payload)
        final_norm = provider.load_final_norm_payload()
        provider.authenticate_final_norm_payload(final_norm)
        provider.release_final_norm_payload(final_norm)
        provider._session.verify()
        assert len(trace_hashes) == 2
        evidence = provider.selective_session_evidence
        assert evidence["authentication_contract"][
            "trace_file_sha256_passes_per_session"
        ] == 1
        assert evidence["authentication_contract"][
            "unselected_tensor_payloads_loaded"
        ] is False
        assert "safe_open path/stat interval" in evidence[
            "authentication_contract"
        ]["toctou_boundary"]
    finally:
        provider.close()


def test_selective_payload_order_content_and_byte_bounds_fail_closed(
    sealed_cache_path,
) -> None:
    provider = build_lfm2_cpcf_selective_evidence_provider(
        open_lfm2_circuit_trace_cache(
            sealed_cache_path,
            verify=False,
        )
    )
    try:
        reference = provider.dynamic_catalog.references[0]
        payload = provider.load_layer_payload(reference)
        try:
            phases = payload.phases
            payload._phases = tuple(reversed(phases))
            with pytest.raises(
                Lfm2CpcfSelectiveEvidenceError,
                match="phase reference|canonical|differs",
            ):
                provider.authenticate_layer_payload(reference, payload)
            payload._phases = phases

            first_tensor = next(
                iter(payload.phases[0].evidence.tensors.values())
            )
            first_tensor.reshape(-1)[0].add_(1)
            with pytest.raises(
                Lfm2CpcfSelectiveEvidenceError,
                match="differs|does not match",
            ):
                provider.authenticate_layer_payload(reference, payload)
        finally:
            provider.release_layer_payload(reference, payload)

        changed_bytes = DynamicEvidenceLayerReference.create(
            source_sha256=reference.source_sha256,
            provider_sha256=reference.provider_sha256,
            layer_index=reference.layer_index,
            evidence_sha256=reference.evidence_sha256,
            logical_bytes=reference.logical_bytes + 1,
        )
        with pytest.raises(
            Lfm2CpcfSelectiveEvidenceError,
            match="differs from the sealed catalog",
        ):
            provider.load_layer_payload(changed_bytes)

        pair_bytes = sum(
            value.logical_bytes
            for value in provider.dynamic_catalog.references[:2]
        )
        window = DynamicEvidenceResidencyWindow(
            provider.dynamic_catalog,
            provider,
            maximum_resident_bytes=pair_bytes - 1,
        )
        with pytest.raises(
            DynamicEvidenceResidencyError,
            match="above maximum_resident_bytes",
        ):
            window.start()
        window.close()

        batch_reference = provider._session.batch_references[0]
        names = batch_reference.ordered_tensor_names[:2]
        with pytest.raises(
            Lfm2CircuitTraceCacheError,
            match="canonical sorted order",
        ):
            provider._session.load_selected_tensors(
                0,
                tuple(reversed(names)),
            )
    finally:
        provider.close()


def test_selective_session_detects_post_authentication_record_tamper(
    sealed_cache_path,
    tmp_path,
) -> None:
    copied = tmp_path / sealed_cache_path.name
    shutil.copytree(sealed_cache_path, copied)
    provider = build_lfm2_cpcf_selective_evidence_provider(
        open_lfm2_circuit_trace_cache(copied, verify=False)
    )
    record_path = (
        copied
        / "batches"
        / "batch-000000"
        / BATCH_RECORD_FILENAME
    )
    original = record_path.read_bytes()
    record_path.write_bytes(original + b" ")
    try:
        with pytest.raises(
            Lfm2CircuitTraceCorruptionError,
            match="record identity changed",
        ):
            provider.load_layer_payload(
                provider.dynamic_catalog.references[0]
            )
    finally:
        with pytest.raises(
            Lfm2CircuitTraceCorruptionError,
            match="identity changed",
        ):
            provider.close()
