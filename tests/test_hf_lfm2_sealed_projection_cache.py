from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_sealed_projection_cache as MODULE  # noqa: E402
from neural_forge.hf_lfm2_sealed_projection_cache import (  # noqa: E402
    Lfm2SealedProjectionCacheError,
    authenticate_lfm2_materialized_projection_cache,
    seal_lfm2_materialized_projection_entry,
    seal_lfm2_projection_cache,
    seal_lfm2_projection_cache_binding,
    seal_lfm2_projection_entry,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _tensors(
    offset: float = 0.0,
    *,
    top_k: int = 3,
    dtype: Any = torch.float32,
) -> dict[str, Any]:
    logits = torch.tensor(
        [[[4.0 + offset, 1.0 + offset, 3.0 + offset, 2.0 + offset]]],
        dtype=dtype,
    )
    ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[
        ..., :top_k
    ].contiguous()
    values = logits.gather(-1, ids).contiguous()
    margins = (values[..., :1] - values[..., 1:]).contiguous()
    return {
        "target_logits": logits,
        "topk_target_ids": ids,
        "topk_values": values,
        "winner_to_topk_margins": margins,
    }


def _entry(
    batch_index: int,
    phase: str,
    *,
    offset: float = 0.0,
    cache_key: str | None = None,
    bridge_sha: str | None = None,
    top_k: int = 3,
) -> Any:
    tensors = _tensors(offset, top_k=top_k)
    return seal_lfm2_materialized_projection_entry(
        record_id=f"record-{batch_index}",
        split="calibration" if batch_index == 0 else "holdout",
        phase=phase,
        cache_key=cache_key or _digest("cache"),
        cache_batch_index=batch_index,
        bridge_artifact_sha256=bridge_sha or _digest("bridge"),
        alignment_sha256=_digest(f"alignment-{batch_index}"),
        parent_cache_input_sha256=_digest(f"input-{batch_index}"),
        source_logits_sha256=_digest(f"source-{batch_index}-{phase}"),
        top_k=top_k,
        **tensors,
    )


def _binding(
    keys: tuple[tuple[int, str], ...],
    *,
    cache_key: str | None = None,
    selective_sha: str | None = None,
    bridge_sha: str | None = None,
    top_k: int = 3,
) -> Any:
    return seal_lfm2_projection_cache_binding(
        cache_key=cache_key or _digest("cache"),
        parent_fingerprint_sha256=_digest("parent"),
        corpus_fingerprint_sha256=_digest("corpus"),
        selective_session_evidence_sha256=(
            selective_sha or _digest("selective-session")
        ),
        bridge_artifact_sha256=bridge_sha or _digest("bridge"),
        bridge_profile_sha256=_digest("bridge-profile"),
        adapter_evidence_sha256=_digest("adapter"),
        adapter_records_sha256=_digest("adapter-records"),
        mapping_sha256=_digest("mapping"),
        top_k=top_k,
        expected_record_count=max(index for index, _phase in keys) + 1,
        projection_keys=keys,
    )


def _cache() -> Any:
    keys = (
        (0, "cached_prefill"),
        (0, "decode.000"),
        (1, "cached_prefill"),
    )
    entries = (
        _entry(0, "cached_prefill", offset=0.0),
        _entry(0, "decode.000", offset=0.25),
        _entry(1, "cached_prefill", offset=0.5),
    )
    return seal_lfm2_projection_cache(binding=_binding(keys), entries=entries)


def _cloned_payload(cache: Any) -> dict[str, Any]:
    payload = cache.to_materialized_payload()
    return {
        "manifest": json.loads(json.dumps(payload["manifest"])),
        "tensors": [
            {name: tensor.clone() for name, tensor in row.items()}
            for row in payload["tensors"]
        ],
    }


def test_materialized_roundtrip_uses_values_not_tensor_identity() -> None:
    cache = _cache()
    payload = _cloned_payload(cache)
    original = cache.entries[0].target_logits

    assert payload["tensors"][0]["target_logits"].data_ptr() != original.data_ptr()
    restored = authenticate_lfm2_materialized_projection_cache(
        payload,
        expected_cache_sha256=cache.cache_sha256,
    )

    assert restored.cache_sha256 == cache.cache_sha256
    assert restored.projection_evidence_sha256 == cache.projection_evidence_sha256
    assert restored.binding.binding_sha256 == cache.binding.binding_sha256
    assert restored.binding.projection_keys == cache.binding.projection_keys
    assert restored.to_manifest_dict() == cache.to_manifest_dict()
    assert all(
        torch.equal(left.target_logits, right.target_logits)
        for left, right in zip(restored.entries, cache.entries)
    )
    assert restored.to_manifest_dict()["model_operations"] == {
        "parent_model_calls": 0,
        "child_model_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
    }
    assert "path" not in restored.binding.to_evidence_dict()

    exposed = restored.entries[0].target_logits
    exposed.add_(100.0)
    assert not torch.equal(exposed, restored.entries[0].target_logits)


def test_each_live_tensor_is_hashed_once_per_seal_or_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = MODULE._tensor_bytes

    def counted(value: Any) -> bytes:
        calls.append(int(value.numel()))
        return original(value)

    monkeypatch.setattr(MODULE, "_tensor_bytes", counted)
    entry = _entry(0, "cached_prefill")
    assert len(calls) == 4

    binding = _binding(((0, "cached_prefill"),))
    cache = seal_lfm2_projection_cache(binding=binding, entries=(entry,))
    payload = cache.to_materialized_payload()
    cache.to_manifest_dict()
    assert len(calls) == 4

    calls.clear()
    restored = authenticate_lfm2_materialized_projection_cache(
        payload,
        expected_cache_sha256=cache.cache_sha256,
    )
    assert restored.cache_sha256 == cache.cache_sha256
    assert len(calls) == 4


def test_public_projection_wrapper_never_uses_private_tensor_identity() -> None:
    entry = _entry(0, "cached_prefill")
    evidence = entry.projection_evidence_dict()

    tensor_names = {
        "target_logits",
        "topk_target_ids",
        "topk_values",
        "winner_to_topk_margins",
    }

    class _Projection:
        def __init__(self) -> None:
            self.values: dict[str, Any] = {}
            self.getter_calls = {name: 0 for name in tensor_names}

        def __getattr__(self, name: str) -> Any:
            if name not in self.values:
                raise AttributeError(name)
            if name in tensor_names:
                self.getter_calls[name] += 1
                return self.values[name].clone()
            return self.values[name]

    projection = _Projection()
    for name in (
        "record_id",
        "split",
        "phase",
        "cache_key",
        "cache_batch_index",
        "bridge_artifact_sha256",
        "alignment_sha256",
        "parent_cache_input_sha256",
        "source_logits_sha256",
        "top_k",
        "projection_sha256",
    ):
        projection.values[name] = evidence[name]
    tensor_payload = entry.to_tensor_payload_dict()
    for name, tensor in tensor_payload.items():
        projection.values[name] = tensor.clone()

    resealed = seal_lfm2_projection_entry(projection)
    assert resealed.entry_sha256 == entry.entry_sha256
    assert resealed.projection_sha256 == entry.projection_sha256
    assert resealed.target_logits.data_ptr() != entry.target_logits.data_ptr()
    assert projection.getter_calls == {name: 1 for name in tensor_names}

    projection.values["target_logits"][0, 0, 0] += 1.0
    with pytest.raises(Lfm2SealedProjectionCacheError):
        seal_lfm2_projection_entry(projection)


def test_entry_owns_immutable_bytes_and_rejects_source_alias_mutation() -> None:
    tensors = _tensors()
    entry = seal_lfm2_materialized_projection_entry(
        record_id="record-0",
        split="calibration",
        phase="cached_prefill",
        cache_key=_digest("cache"),
        cache_batch_index=0,
        bridge_artifact_sha256=_digest("bridge"),
        alignment_sha256=_digest("alignment"),
        parent_cache_input_sha256=_digest("input"),
        source_logits_sha256=_digest("source"),
        top_k=3,
        **tensors,
    )
    expected = entry.target_logits
    tensors["target_logits"].add_(100.0)

    assert torch.equal(entry.target_logits, expected)
    assert isinstance(entry._target_logits_bytes, bytes)
    with pytest.raises(TypeError):
        entry._target_logits_bytes[0] = 1  # type: ignore[index]

    cache = seal_lfm2_projection_cache(
        binding=_binding(((0, "cached_prefill"),)),
        entries=(entry,),
    )
    assert len(cache) == 1
    assert torch.equal(cache.load_target_logits(0, "cached_prefill"), expected)
    with pytest.raises(Lfm2SealedProjectionCacheError, match="does not contain"):
        cache.load_target_logits(0, "decode.000")


@pytest.mark.parametrize(
    "tensor_name",
    [
        "target_logits",
        "topk_target_ids",
        "topk_values",
        "winner_to_topk_margins",
    ],
)
def test_tensor_tampering_fails_closed(tensor_name: str) -> None:
    cache = _cache()
    payload = _cloned_payload(cache)
    selected = payload["tensors"][0][tensor_name]
    if selected.is_floating_point():
        selected.reshape(-1)[0] += 0.125
    else:
        selected.reshape(-1)[0] = (int(selected.reshape(-1)[0]) + 1) % 4

    with pytest.raises(Lfm2SealedProjectionCacheError):
        authenticate_lfm2_materialized_projection_cache(
            payload,
            expected_cache_sha256=cache.cache_sha256,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("binding", "cache_key"), _digest("different-cache")),
        (("binding", "selective_session_evidence_sha256"), _digest("new-session")),
        (("projection_entries", 0, "projection_evidence", "record_id"), "tampered"),
        (("projection_entries", 0, "entry_sha256"), _digest("bad-entry")),
        (("projection_evidence_sha256",), _digest("bad-aggregate")),
        (("model_operations", "training_steps"), 1),
    ],
)
def test_manifest_tampering_fails_closed(
    path: tuple[Any, ...],
    value: Any,
) -> None:
    cache = _cache()
    payload = _cloned_payload(cache)
    selected: Any = payload["manifest"]
    for component in path[:-1]:
        selected = selected[component]
    selected[path[-1]] = value

    with pytest.raises(Lfm2SealedProjectionCacheError):
        authenticate_lfm2_materialized_projection_cache(
            payload,
            expected_cache_sha256=cache.cache_sha256,
        )


def test_external_cache_seal_catches_fully_resealed_manifest() -> None:
    cache = _cache()
    payload = _cloned_payload(cache)
    payload["manifest"]["binding"]["corpus_fingerprint_sha256"] = _digest(
        "attacker-corpus"
    )
    binding_body = dict(payload["manifest"]["binding"])
    binding_body.pop("binding_sha256")
    payload["manifest"]["binding"]["binding_sha256"] = MODULE._canonical_sha256(
        binding_body
    )
    cache_body = dict(payload["manifest"])
    cache_body.pop("cache_sha256")
    payload["manifest"]["cache_sha256"] = MODULE._canonical_sha256(cache_body)

    with pytest.raises(
        Lfm2SealedProjectionCacheError,
        match="expected external seal",
    ):
        authenticate_lfm2_materialized_projection_cache(
            payload,
            expected_cache_sha256=cache.cache_sha256,
        )


def test_binding_identity_changes_invalidate_reuse() -> None:
    keys = ((0, "cached_prefill"),)
    baseline = _binding(keys)
    changed_session = _binding(keys, selective_sha=_digest("new-session"))
    changed_cache = _binding(keys, cache_key=_digest("new-cache"))
    changed_bridge = _binding(keys, bridge_sha=_digest("new-bridge"))

    assert (
        len(
            {
                baseline.binding_sha256,
                changed_session.binding_sha256,
                changed_cache.binding_sha256,
                changed_bridge.binding_sha256,
            }
        )
        == 4
    )

    cache = seal_lfm2_projection_cache(
        binding=baseline,
        entries=(_entry(0, "cached_prefill"),),
    )
    payload = _cloned_payload(cache)
    payload["manifest"]["binding"] = changed_session.to_evidence_dict()
    with pytest.raises(Lfm2SealedProjectionCacheError):
        authenticate_lfm2_materialized_projection_cache(
            payload,
            expected_cache_sha256=cache.cache_sha256,
        )


def test_binding_has_explicit_no_selective_session_sentinel() -> None:
    keys = ((0, "cached_prefill"),)
    binding = seal_lfm2_projection_cache_binding(
        cache_key=_digest("cache"),
        parent_fingerprint_sha256=_digest("parent"),
        corpus_fingerprint_sha256=_digest("corpus"),
        selective_session_evidence_sha256=None,
        bridge_artifact_sha256=_digest("bridge"),
        bridge_profile_sha256=_digest("profile"),
        adapter_evidence_sha256=_digest("adapter"),
        adapter_records_sha256=_digest("records"),
        mapping_sha256=_digest("mapping"),
        top_k=3,
        expected_record_count=1,
        projection_keys=keys,
    )
    assert (
        binding.selective_session_evidence_sha256
        == MODULE.NO_SELECTIVE_SESSION_EVIDENCE_SHA256
    )
    assert binding.verify() == binding


def test_binding_accepts_exact_r12_256_by_9_projection_inventory() -> None:
    keys = tuple(
        key
        for record_index in range(256)
        for key in (
            (record_index, "cached_prefill"),
            *((record_index, f"decode.{index:03d}") for index in range(8)),
        )
    )
    binding = seal_lfm2_projection_cache_binding(
        cache_key=_digest("cache"),
        parent_fingerprint_sha256=_digest("parent"),
        corpus_fingerprint_sha256=_digest("corpus"),
        selective_session_evidence_sha256=_digest("session"),
        bridge_artifact_sha256=_digest("bridge"),
        bridge_profile_sha256=_digest("profile"),
        adapter_evidence_sha256=_digest("adapter"),
        adapter_records_sha256=_digest("records"),
        mapping_sha256=_digest("mapping"),
        top_k=8,
        expected_record_count=256,
        projection_keys=keys,
    )
    assert binding.expected_record_count == 256
    assert binding.expected_projection_count == 2_304
    assert binding.projection_keys == keys
    assert binding.verify() == binding


def test_projection_inventory_order_and_coverage_are_enforced() -> None:
    with pytest.raises(
        Lfm2SealedProjectionCacheError,
        match="does not cover every cache record",
    ):
        seal_lfm2_projection_cache_binding(
            cache_key=_digest("cache"),
            parent_fingerprint_sha256=_digest("parent"),
            corpus_fingerprint_sha256=_digest("corpus"),
            selective_session_evidence_sha256=_digest("session"),
            bridge_artifact_sha256=_digest("bridge"),
            bridge_profile_sha256=_digest("profile"),
            adapter_evidence_sha256=_digest("adapter"),
            adapter_records_sha256=_digest("records"),
            mapping_sha256=_digest("mapping"),
            top_k=3,
            expected_record_count=2,
            projection_keys=((0, "cached_prefill"),),
        )

    keys = ((0, "cached_prefill"), (0, "decode.000"))
    binding = _binding(keys)
    entries = (
        _entry(0, "cached_prefill"),
        _entry(0, "decode.000", offset=0.25),
    )
    with pytest.raises(
        Lfm2SealedProjectionCacheError,
        match="order differs",
    ):
        seal_lfm2_projection_cache(
            binding=binding,
            entries=tuple(reversed(entries)),
        )

    with pytest.raises(
        Lfm2SealedProjectionCacheError,
        match="duplicate phase key",
    ):
        _binding(((0, "cached_prefill"), (0, "cached_prefill")))

    with pytest.raises(
        Lfm2SealedProjectionCacheError,
        match="must begin with one cached_prefill",
    ):
        _binding(((0, "decode.000"),))

    with pytest.raises(
        Lfm2SealedProjectionCacheError,
        match="contiguous and canonically ordered",
    ):
        _binding(((0, "cached_prefill"), (0, "decode.001")))

    with pytest.raises(
        Lfm2SealedProjectionCacheError,
        match="canonical record order",
    ):
        _binding(((0, "cached_prefill"), (1, "cached_prefill"), (0, "decode.000")))


def test_noncanonical_binding_text_is_rejected_before_object_creation() -> None:
    evidence = _binding(((0, "cached_prefill"),)).to_evidence_dict()
    evidence["projection_inventory"][0]["phase"] = " cached_prefill "
    body = dict(evidence)
    body.pop("binding_sha256")
    evidence["binding_sha256"] = MODULE._canonical_sha256(body)
    with pytest.raises(
        Lfm2SealedProjectionCacheError,
        match="canonical NFC text",
    ):
        MODULE.authenticate_lfm2_projection_cache_binding(evidence)

    evidence = _binding(((0, "cached_prefill"),)).to_evidence_dict()
    evidence["projection_inventory"][0]["phase"] = "cache\u0301d_prefill"
    body = dict(evidence)
    body.pop("binding_sha256")
    evidence["binding_sha256"] = MODULE._canonical_sha256(body)
    with pytest.raises(
        Lfm2SealedProjectionCacheError,
        match="canonical NFC text",
    ):
        MODULE.authenticate_lfm2_projection_cache_binding(evidence)


def test_dtype_shape_nonfinite_and_topk_semantics_fail_closed() -> None:
    tensors = _tensors()
    tensors["target_logits"][0, 0, 0] = float("nan")
    with pytest.raises(Lfm2SealedProjectionCacheError, match="finite rank-3"):
        seal_lfm2_materialized_projection_entry(
            record_id="record-0",
            split="calibration",
            phase="cached_prefill",
            cache_key=_digest("cache"),
            cache_batch_index=0,
            bridge_artifact_sha256=_digest("bridge"),
            alignment_sha256=_digest("alignment"),
            parent_cache_input_sha256=_digest("input"),
            source_logits_sha256=_digest("source"),
            top_k=3,
            **tensors,
        )

    tensors = _tensors()
    tensors["topk_target_ids"] = tensors["topk_target_ids"].to(torch.int32)
    with pytest.raises(Lfm2SealedProjectionCacheError, match="torch.long"):
        seal_lfm2_materialized_projection_entry(
            record_id="record-0",
            split="calibration",
            phase="cached_prefill",
            cache_key=_digest("cache"),
            cache_batch_index=0,
            bridge_artifact_sha256=_digest("bridge"),
            alignment_sha256=_digest("alignment"),
            parent_cache_input_sha256=_digest("input"),
            source_logits_sha256=_digest("source"),
            top_k=3,
            **tensors,
        )

    tensors = _tensors()
    tensors["topk_target_ids"] = tensors["topk_target_ids"].flip(-1)
    tensors["topk_values"] = tensors["target_logits"].gather(
        -1,
        tensors["topk_target_ids"],
    )
    tensors["winner_to_topk_margins"] = (
        tensors["topk_values"][..., :1] - tensors["topk_values"][..., 1:]
    )
    with pytest.raises(
        Lfm2SealedProjectionCacheError,
        match="deterministic target-space ordering",
    ):
        seal_lfm2_materialized_projection_entry(
            record_id="record-0",
            split="calibration",
            phase="cached_prefill",
            cache_key=_digest("cache"),
            cache_batch_index=0,
            bridge_artifact_sha256=_digest("bridge"),
            alignment_sha256=_digest("alignment"),
            parent_cache_input_sha256=_digest("input"),
            source_logits_sha256=_digest("source"),
            top_k=3,
            **tensors,
        )


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_supported_float_dtypes_roundtrip_as_contiguous_cpu(dtype: Any) -> None:
    tensors = _tensors(dtype=dtype)
    entry = seal_lfm2_materialized_projection_entry(
        record_id="record-0",
        split="calibration",
        phase="cached_prefill",
        cache_key=_digest("cache"),
        cache_batch_index=0,
        bridge_artifact_sha256=_digest("bridge"),
        alignment_sha256=_digest("alignment"),
        parent_cache_input_sha256=_digest("input"),
        source_logits_sha256=_digest("source"),
        top_k=3,
        **tensors,
    )
    restored = entry.target_logits
    assert restored.dtype == dtype
    assert restored.device.type == "cpu"
    assert restored.is_contiguous()
    assert torch.equal(restored, tensors["target_logits"])


def test_noncontiguous_inputs_are_canonicalized_without_source_aliases() -> None:
    base = torch.tensor(
        [[[4.0, 99.0, 1.0, 99.0, 3.0, 99.0, 2.0, 99.0]]],
        dtype=torch.float32,
    )
    logits = base[..., ::2]
    assert not logits.is_contiguous()
    ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[..., :3]
    values = logits.gather(-1, ids)
    margins = values[..., :1] - values[..., 1:]
    entry = seal_lfm2_materialized_projection_entry(
        record_id="record-0",
        split="calibration",
        phase="cached_prefill",
        cache_key=_digest("cache"),
        cache_batch_index=0,
        bridge_artifact_sha256=_digest("bridge"),
        alignment_sha256=_digest("alignment"),
        parent_cache_input_sha256=_digest("input"),
        source_logits_sha256=_digest("source"),
        top_k=3,
        target_logits=logits,
        topk_target_ids=ids,
        topk_values=values,
        winner_to_topk_margins=margins,
    )
    expected = logits.contiguous().clone()
    base.add_(1000.0)
    assert entry.target_logits.is_contiguous()
    assert torch.equal(entry.target_logits, expected)


def test_unsupported_tensor_forms_raise_cache_errors() -> None:
    tensors = _tensors()
    tensors["target_logits"] = tensors["target_logits"].to(torch.complex64)
    with pytest.raises(Lfm2SealedProjectionCacheError):
        seal_lfm2_materialized_projection_entry(
            record_id="record-0",
            split="calibration",
            phase="cached_prefill",
            cache_key=_digest("cache"),
            cache_batch_index=0,
            bridge_artifact_sha256=_digest("bridge"),
            alignment_sha256=_digest("alignment"),
            parent_cache_input_sha256=_digest("input"),
            source_logits_sha256=_digest("source"),
            top_k=3,
            **tensors,
        )


def test_float8_is_rejected_before_unsupported_backend_operations() -> None:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.skip("Installed PyTorch has no float8 dtype")
    tensors = _tensors()
    for name in (
        "target_logits",
        "topk_values",
        "winner_to_topk_margins",
    ):
        tensors[name] = tensors[name].to(dtype)
    with pytest.raises(
        Lfm2SealedProjectionCacheError,
        match="supported floating tensor",
    ):
        seal_lfm2_materialized_projection_entry(
            record_id="record-0",
            split="calibration",
            phase="cached_prefill",
            cache_key=_digest("cache"),
            cache_batch_index=0,
            bridge_artifact_sha256=_digest("bridge"),
            alignment_sha256=_digest("alignment"),
            parent_cache_input_sha256=_digest("input"),
            source_logits_sha256=_digest("source"),
            top_k=3,
            **tensors,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_inputs_are_sealed_as_cpu_values() -> None:
    tensors = {name: value.to("cuda") for name, value in _tensors().items()}
    entry = seal_lfm2_materialized_projection_entry(
        record_id="record-0",
        split="calibration",
        phase="cached_prefill",
        cache_key=_digest("cache"),
        cache_batch_index=0,
        bridge_artifact_sha256=_digest("bridge"),
        alignment_sha256=_digest("alignment"),
        parent_cache_input_sha256=_digest("input"),
        source_logits_sha256=_digest("source"),
        top_k=3,
        **tensors,
    )
    assert entry.target_logits.device.type == "cpu"


def test_payload_field_additions_and_projection_reordering_fail_closed() -> None:
    cache = _cache()
    extra = _cloned_payload(cache)
    extra["manifest"]["unexpected"] = True
    with pytest.raises(Lfm2SealedProjectionCacheError, match="fields differ"):
        authenticate_lfm2_materialized_projection_cache(
            extra,
            expected_cache_sha256=cache.cache_sha256,
        )

    reordered = _cloned_payload(cache)
    (
        reordered["manifest"]["projection_entries"][0],
        reordered["manifest"]["projection_entries"][1],
    ) = (
        reordered["manifest"]["projection_entries"][1],
        reordered["manifest"]["projection_entries"][0],
    )
    reordered["tensors"][0], reordered["tensors"][1] = (
        reordered["tensors"][1],
        reordered["tensors"][0],
    )
    with pytest.raises(Lfm2SealedProjectionCacheError):
        authenticate_lfm2_materialized_projection_cache(
            reordered,
            expected_cache_sha256=cache.cache_sha256,
        )


def test_payload_can_survive_deepcopy_without_aliases() -> None:
    cache = _cache()
    payload = cache.to_materialized_payload()
    copied = copy.deepcopy(payload)
    assert copied["manifest"] == payload["manifest"]
    assert (
        copied["tensors"][0]["target_logits"].data_ptr()
        != payload["tensors"][0]["target_logits"].data_ptr()
    )
    restored = authenticate_lfm2_materialized_projection_cache(
        copied,
        expected_cache_sha256=cache.cache_sha256,
    )
    assert restored.cache_sha256 == cache.cache_sha256
