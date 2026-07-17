from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest
import torch

from neural_forge.hf_lfm2_causal_residual_entropy_value_fold import (
    CREVF_ANCHOR_MAGIC,
    CREVF_ARTIFACT_FIXED_BYTES,
    CREVF_ENTROPY_MAGIC,
    CREVF_RECORD_ENVELOPE_BYTES,
    CREVF_ZLIB_PROFILE,
    CrevfAuxiliarySpec,
    CrevfBudgetError,
    CrevfError,
    CrevfExactRun,
    CrevfExactValue,
    CrevfFormatError,
    CrevfReductionTarget,
    CrevfTensorSpec,
    _decode_entropy,
    _decode_uleb128,
    _encode_uleb128,
    account_crevf_artifact,
    build_crevf_exact_spec,
    build_crevf_explicit_quantized_spec,
    build_crevf_quantized_spec,
    decode_crevf_artifact,
    decode_crevf_tensor,
    encode_crevf_artifact,
    encode_crevf_auxiliary,
    encode_crevf_tensor,
    inspect_crevf_auxiliary,
    inspect_crevf_tensor,
    iter_crevf_quantized_candidates,
    select_crevf_group_topk_outliers,
)


def _quantized_blob(
    name: str,
    role: str,
    values: torch.Tensor,
    widths,
    *,
    outliers=(),
    anchors=(),
) -> bytes:
    return encode_crevf_tensor(
        build_crevf_quantized_spec(
            name=name,
            role=role,
            values=values,
            group_bit_widths=widths,
            group_size=16,
            exact_outlier_indices=outliers,
            exact_anchor_ranges=anchors,
        )
    )


def test_tensor_codec_is_canonical_entropy_coded_and_preserves_exact_values() -> None:
    values = torch.linspace(-3.0, 4.0, 32, dtype=torch.float32).to(torch.bfloat16)
    spec = build_crevf_quantized_spec(
        name="model.layers.0.feed_forward.w2.weight",
        role="ffn_w2",
        values=values,
        group_bit_widths=(4, 8),
        group_size=16,
        exact_outlier_indices=(2,),
        exact_anchor_ranges=((20, 4),),
    )
    first = encode_crevf_tensor(spec)
    second = encode_crevf_tensor(spec)
    assert first == second
    assert CREVF_ANCHOR_MAGIC in first
    assert b"base64" not in first
    metadata = inspect_crevf_tensor(first)
    assert metadata.group_bit_widths == (4, 8)
    assert metadata.exact_outlier_count == 1
    assert metadata.exact_anchor_count == 4
    runtime = decode_crevf_tensor(first)
    assert tuple(runtime.parameters()) == ()
    assert runtime.state_dict() == {}
    assert runtime.packed_runtime_bytes == metadata.runtime_packed_bytes
    assert runtime.persistent_encoded_bytes == metadata.persistent_encoded_bytes
    assert runtime.decoded_full_state_bytes == metadata.decoded_full_state_bytes
    assert runtime.decoded_code_transient_bytes == metadata.decoded_code_bytes
    assert hasattr(runtime, "entropy_segment")
    assert not hasattr(runtime, "packed_codes")
    restored = runtime.dequantize().reshape(-1)
    assert restored.dtype == torch.bfloat16
    assert bool(torch.isfinite(restored).all())
    assert restored[2].view(torch.uint16) == values[2].view(torch.uint16)
    assert torch.equal(
        restored[20:24].view(torch.uint16),
        values[20:24].view(torch.uint16),
    )


def test_explicit_solution_builder_preserves_solved_codes_scales_and_overlays() -> None:
    outlier_bits = int(
        torch.tensor(5.0, dtype=torch.bfloat16).view(torch.uint16).item()
    )
    anchor_bits = int(
        torch.tensor(-3.0, dtype=torch.bfloat16).view(torch.uint16).item()
    )
    codes = np.asarray(
        [
            [7, 7, 8, 6, 9, 5, 10, 4],
            [127, 129, 127, 127, 127, 127, 127, 127],
        ],
        dtype=np.uint8,
    )
    scales = np.asarray([0.5, 0.25], dtype=np.float16)
    spec = build_crevf_explicit_quantized_spec(
        name="model.layers.0.feed_forward.w2.weight",
        role="ffn_w2",
        shape=(10,),
        group_size=8,
        group_bit_widths=(4, 8),
        group_codes=codes,
        scales=scales,
        exact_outliers=(CrevfExactValue(1, outlier_bits),),
        exact_anchors=(CrevfExactRun.from_bits(8, (anchor_bits,)),),
    )
    assert np.array_equal(spec.group_codes, codes)
    assert np.array_equal(
        np.asarray(spec.scales).view(np.uint16),
        scales.view(np.uint16),
    )
    blob = encode_crevf_tensor(spec)
    restored = decode_crevf_tensor(blob).dequantize().float()
    expected = torch.tensor(
        [0.0, 5.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, -3.0, 0.5]
    )
    assert torch.equal(restored, expected)
    assert inspect_crevf_tensor(blob).group_bit_widths == (4, 8)


def test_explicit_solution_builder_rejects_mutated_geometry_or_numeric_policy() -> None:
    codes = np.full((2, 8), 7, dtype=np.uint8)
    codes[1, :] = 127
    scales = np.asarray([1.0, 1.0], dtype=np.float16)
    with pytest.raises(CrevfError, match="float16"):
        build_crevf_explicit_quantized_spec(
            name="w",
            role="ffn_w2",
            shape=(10,),
            group_size=8,
            group_bit_widths=(4, 8),
            group_codes=codes,
            scales=scales.astype(np.float32),
        )
    bad_padding = codes.copy()
    bad_padding[1, 7] = 128
    with pytest.raises(CrevfError, match="padded"):
        build_crevf_explicit_quantized_spec(
            name="w",
            role="ffn_w2",
            shape=(10,),
            group_size=8,
            group_bit_widths=(4, 8),
            group_codes=bad_padding,
            scales=scales,
        )
    bad_overlay = codes.copy()
    bad_overlay[0, 0] = 8
    with pytest.raises(CrevfError, match="canonical zero"):
        build_crevf_explicit_quantized_spec(
            name="w",
            role="ffn_w2",
            shape=(10,),
            group_size=8,
            group_bit_widths=(4, 8),
            group_codes=bad_overlay,
            scales=scales,
            exact_outliers=(CrevfExactValue(0, 0),),
        )
    bad_range = codes.astype(np.int16)
    bad_range[0, 0] = 15
    with pytest.raises(CrevfError, match="exceeds"):
        build_crevf_explicit_quantized_spec(
            name="w",
            role="ffn_w2",
            shape=(10,),
            group_size=8,
            group_bit_widths=(4, 8),
            group_codes=bad_range,
            scales=scales,
        )


def test_auxiliary_artifact_is_bounded_authenticated_and_parameter_free() -> None:
    values = torch.zeros(256, dtype=torch.bfloat16)
    tensor_blob = _quantized_blob(
        "model.layers.0.feed_forward.w2.weight",
        "ffn_w2",
        values,
        (8,) * 16,
    )
    payload = struct.pack("<4H", 3, 9, 17, 255)
    auxiliary_blob = encode_crevf_auxiliary(
        CrevfAuxiliarySpec(
            name="model.layers.0.feed_forward.w2.weight.writer.indices",
            kind="writer_column_indices_u16",
            payload=payload,
        )
    )
    metadata = inspect_crevf_auxiliary(auxiliary_blob)
    assert metadata.payload_bytes == len(payload)
    assert metadata.persistent_encoded_bytes == len(payload)
    assert metadata.decoded_full_state_bytes == len(payload)
    artifact = encode_crevf_artifact(
        (tensor_blob,),
        auxiliary_blobs=(auxiliary_blob,),
    )
    runtime = decode_crevf_artifact(artifact)
    assert runtime.tensor_names == ("model.layers.0.feed_forward.w2.weight",)
    assert runtime.auxiliary_names == (
        "model.layers.0.feed_forward.w2.weight.writer.indices",
    )
    assert runtime.auxiliary(runtime.auxiliary_names[0]).payload_bytes() == payload
    assert tuple(runtime.parameters()) == ()
    assert runtime.state_dict() == {}
    tensor_metadata = inspect_crevf_tensor(tensor_blob)
    assert runtime.persistent_encoded_bytes == (
        tensor_metadata.persistent_encoded_bytes + len(payload)
    )
    assert runtime.decoded_full_state_bytes == (
        tensor_metadata.decoded_full_state_bytes + len(payload)
    )
    assert len(artifact) == (
        CREVF_ARTIFACT_FIXED_BYTES
        + 2 * CREVF_RECORD_ENVELOPE_BYTES
        + len(tensor_blob)
        + len(auxiliary_blob)
    )
    with pytest.raises(CrevfBudgetError, match="decode bound"):
        decode_crevf_artifact(
            artifact,
            max_auxiliary_payload_bytes=len(payload) - 1,
        )
    corrupt = bytearray(auxiliary_blob)
    corrupt[-33] ^= 1
    with pytest.raises(CrevfFormatError, match="auxiliary SHA-256"):
        inspect_crevf_auxiliary(bytes(corrupt))
    duplicate_name = encode_crevf_auxiliary(
        CrevfAuxiliarySpec(
            name="model.layers.0.feed_forward.w2.weight",
            kind="collision",
            payload=b"x",
        )
    )
    with pytest.raises(CrevfError, match="duplicate record names"):
        encode_crevf_artifact(
            (tensor_blob,),
            auxiliary_blobs=(duplicate_name,),
        )
    auxiliary_only = decode_crevf_artifact(
        encode_crevf_artifact((), auxiliary_blobs=(auxiliary_blob,))
    )
    assert auxiliary_only.tensor_names == ()
    assert auxiliary_only.auxiliary_names == (metadata.name,)
    assert auxiliary_only.layer_decode_transient_bytes == ()
    assert tuple(auxiliary_only.parameters()) == ()


def test_per_tensor_canonical_entropy_compresses_repetitive_code_bytes() -> None:
    values = torch.zeros(2048, dtype=torch.bfloat16)
    blob = encode_crevf_tensor(
        build_crevf_quantized_spec(
            name="model.embed_tokens.weight",
            role="embedding",
            values=values,
            group_bit_widths=8,
            group_size=256,
        )
    )
    metadata = inspect_crevf_tensor(blob)
    assert metadata.decoded_code_bytes == 2048
    assert metadata.entropy_segment_bytes < metadata.decoded_code_bytes
    assert metadata.persistent_encoded_bytes < metadata.decoded_full_state_bytes
    first = decode_crevf_tensor(blob)
    second = decode_crevf_tensor(blob)
    assert torch.equal(first.dequantize(), second.dequantize())


def test_exact_outlier_is_excluded_from_base_scale_and_uses_zero_base_code() -> None:
    values = torch.tensor(
        [1000.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=torch.bfloat16,
    )
    spec = build_crevf_quantized_spec(
        name="w",
        role="ffn_w2",
        values=values,
        group_bit_widths=4,
        group_size=8,
        exact_outlier_indices=(0,),
    )
    restored = decode_crevf_tensor(encode_crevf_tensor(spec)).dequantize()
    assert restored[0].view(torch.uint16) == values[0].view(torch.uint16)
    assert torch.equal(restored[1:], values[1:])
    assert int(spec.group_codes[0, 0]) == 7

    noncanonical_codes = spec.group_codes.clone()
    noncanonical_codes[0, 0] = 8
    invalid = CrevfTensorSpec(
        name=spec.name,
        role=spec.role,
        shape=spec.shape,
        group_size=spec.group_size,
        group_bit_widths=spec.group_bit_widths,
        group_codes=noncanonical_codes,
        scales=spec.scales,
        exact_outliers=spec.exact_outliers,
    )
    with pytest.raises(CrevfError, match="canonical zero base code"):
        encode_crevf_tensor(invalid)


def test_row64_topk_peeling_and_weighted_clipping_repair_absmax_damage() -> None:
    values = torch.ones((2, 64), dtype=torch.bfloat16)
    values[0, 0] = 1000
    values[1, 63] = -500
    selected = select_crevf_group_topk_outliers(
        values,
        group_size=64,
        outliers_per_group=1,
    )
    assert selected == (0, 127)

    plain = decode_crevf_tensor(
        encode_crevf_tensor(
            build_crevf_quantized_spec(
                name="model.layers.0.feed_forward.w2.weight",
                role="ffn_w2",
                values=values,
                group_bit_widths=8,
                group_size=64,
            )
        )
    ).dequantize()
    peeled = decode_crevf_tensor(
        encode_crevf_tensor(
            build_crevf_quantized_spec(
                name="model.layers.0.feed_forward.w2.weight",
                role="ffn_w2",
                values=values,
                group_bit_widths=8,
                group_size=64,
                exact_outlier_indices=selected,
            )
        )
    ).dequantize()
    ordinary = torch.ones_like(values, dtype=torch.bool)
    ordinary.reshape(-1)[list(selected)] = False
    plain_error = torch.mean(
        (plain[ordinary].float() - values[ordinary].float()).square()
    )
    peeled_error = torch.mean(
        (peeled[ordinary].float() - values[ordinary].float()).square()
    )
    assert peeled_error < plain_error
    assert torch.equal(
        peeled.reshape(-1)[list(selected)].view(torch.uint16),
        values.reshape(-1)[list(selected)].view(torch.uint16),
    )

    clip_values = torch.ones(64, dtype=torch.bfloat16)
    clip_values[0] = 10
    importance = torch.ones(64)
    importance[0] = 0
    clipped = decode_crevf_tensor(
        encode_crevf_tensor(
            build_crevf_quantized_spec(
                name="model.layers.1.feed_forward.w2.weight",
                role="ffn_w2",
                values=clip_values,
                group_bit_widths=4,
                group_size=64,
                clipping_ratios=(1.0, 0.7),
                rounding_offsets=(0.0,),
                importance_weights=importance,
            )
        )
    ).dequantize()
    assert torch.equal(clipped[1:], clip_values[1:])


def test_candidate_iterator_charges_real_row_aligned_policy_blobs() -> None:
    values = torch.arange(128, dtype=torch.float32).reshape(2, 64).to(torch.bfloat16)
    candidates = tuple(
        iter_crevf_quantized_candidates(
            name="model.layers.0.feed_forward.w2.weight",
            role="ffn_w2",
            values=values,
            group_sizes=(64,),
            bit_widths=(7, 8),
            outliers_per_group=(0, 1),
            clipping_ratios=(1.0,),
            rounding_offsets=(0.0,),
        )
    )
    assert len(candidates) == 4
    assert len({candidate.candidate_id for candidate in candidates}) == 4
    assert all(
        candidate.tensor_physical_bytes == len(candidate.tensor_blob)
        for candidate in candidates
    )
    assert {
        inspect_crevf_tensor(candidate.tensor_blob).exact_outlier_count
        for candidate in candidates
    } == {0, 2}
    with pytest.raises(CrevfError, match="divide"):
        tuple(
            iter_crevf_quantized_candidates(
                name="w",
                role="ffn_w2",
                values=values,
                group_sizes=(32, 128),
                bit_widths=(8,),
                outliers_per_group=(0,),
                clipping_ratios=(1.0,),
                rounding_offsets=(0.0,),
            )
        )


def test_invalid_group_codes_are_rejected_before_uint8_narrowing() -> None:
    spec = CrevfTensorSpec(
        name="w",
        role="ffn_w2",
        shape=(8,),
        group_size=8,
        group_bit_widths=(8,),
        group_codes=np.asarray([[-2, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int16),
        scales=np.ones(1, dtype=np.float16),
    )
    with pytest.raises(CrevfError, match="nonnegative"):
        encode_crevf_tensor(spec)


def test_entropy_and_uleb_decoders_reject_declared_allocation_overflow() -> None:
    malformed = (
        struct.pack(
            "<4sBBBBQQ",
            CREVF_ENTROPY_MAGIC,
            *CREVF_ZLIB_PROFILE,
            1_000_000,
            2,
        )
        + b"\x03\x00"
    )
    with pytest.raises(CrevfFormatError, match="tensor geometry"):
        _decode_entropy(malformed, expected_raw_size=8)
    with pytest.raises(CrevfError, match="uint64"):
        _encode_uleb128(1 << 64)
    with pytest.raises(CrevfFormatError, match="uint64"):
        _decode_uleb128(b"\x80\x80\x80\x80\x80\x80\x80\x80\x80\x02", 0)


def test_exact_only_binary_anchor_runtime_has_no_parameters() -> None:
    values = torch.tensor(
        [1.0, -2.0, 0.5, 9.0, -0.125, 4.0, 7.0, -8.0],
        dtype=torch.bfloat16,
    )
    blob = encode_crevf_tensor(
        build_crevf_exact_spec(
            name="model.layers.0.operator_norm.weight",
            role="rms_norm",
            values=values,
        )
    )
    metadata = inspect_crevf_tensor(blob)
    assert metadata.exact_only is True
    assert metadata.minimum_bit_width == 16
    assert metadata.exact_anchor_count == values.numel()
    assert CREVF_ANCHOR_MAGIC in blob
    runtime = decode_crevf_tensor(blob)
    assert tuple(runtime.named_parameters()) == ()
    assert not hasattr(runtime, "group_bit_widths")
    assert torch.equal(
        runtime.dequantize().view(torch.uint16),
        values.view(torch.uint16),
    )


def test_artifact_is_sorted_deterministic_and_strictly_hash_authenticated() -> None:
    values = torch.arange(32, dtype=torch.float32).to(torch.bfloat16)
    z_blob = _quantized_blob("z.weight", "ffn_w2", values, (6, 6))
    a_blob = encode_crevf_tensor(
        build_crevf_exact_spec(
            name="a.weight",
            role="operator_norm",
            values=values[:8],
        )
    )
    first = encode_crevf_artifact([z_blob, a_blob])
    second = encode_crevf_artifact([a_blob, z_blob])
    assert first == second
    runtime = decode_crevf_artifact(first)
    assert runtime.tensor_names == ("a.weight", "z.weight")
    assert tuple(runtime.parameters()) == ()
    assert runtime.state_dict() == {}
    assert runtime.physical_artifact_bytes == len(first)
    assert len(first) == (
        CREVF_ARTIFACT_FIXED_BYTES
        + 2 * CREVF_RECORD_ENVELOPE_BYTES
        + len(a_blob)
        + len(z_blob)
    )

    corrupt_tensor = bytearray(z_blob)
    corrupt_tensor[len(corrupt_tensor) // 2] ^= 1
    with pytest.raises(CrevfFormatError, match="tensor SHA-256"):
        decode_crevf_tensor(bytes(corrupt_tensor))

    # Re-seal the whole artifact but leave the record seal stale.  This proves
    # record authentication is independently enforced after the outer hash.
    core = bytearray(first[:-32])
    first_blob_offset = 24 + CREVF_RECORD_ENVELOPE_BYTES
    core[first_blob_offset + 12] ^= 1
    corrupt_record = bytes(core) + hashlib.sha256(core).digest()
    with pytest.raises(CrevfFormatError, match="record SHA-256"):
        decode_crevf_artifact(corrupt_record)


def test_tensor_only_artifact_remains_byte_identical_to_legacy_v1_layout() -> None:
    values = torch.arange(32, dtype=torch.float32).to(torch.bfloat16)
    blob = _quantized_blob("w.weight", "ffn_w2", values, (8, 8))
    records = struct.pack(
        "<Q32s",
        len(blob),
        hashlib.sha256(blob).digest(),
    ) + blob
    core = struct.pack(
        "<8sHHIQ",
        b"NFCREVF1",
        1,
        0,
        1,
        len(records),
    ) + records
    legacy = core + hashlib.sha256(core).digest()
    assert encode_crevf_artifact((blob,)) == legacy
    runtime = decode_crevf_artifact(legacy)
    assert runtime.auxiliary_names == ()
    assert runtime.state_dict() == {}


@pytest.mark.parametrize(
    ("reduction", "ppm"),
    [
        (0.50, 500_000),
        (0.55, 550_000),
        (0.60, 600_000),
        (0.65, 650_000),
    ],
)
def test_exact_physical_and_runtime_accounting_for_large_targets(
    reduction: float,
    ppm: int,
) -> None:
    values = torch.arange(64, dtype=torch.float32).to(torch.bfloat16)
    artifact = encode_crevf_artifact(
        [
            _quantized_blob("w.weight", "ffn_w2", values, (5, 5, 5, 5)),
            encode_crevf_tensor(
                build_crevf_exact_spec(
                    name="norm.weight",
                    role="rms_norm",
                    values=values[:8],
                )
            ),
        ]
    )
    accounting = account_crevf_artifact(
        artifact,
        dense_reference_bytes=10_000,
        target_reduction=reduction,
        shared_checkpoint_bytes=137,
        shared_persistent_encoded_bytes=29,
        shared_decoded_full_state_bytes=31,
    )
    runtime = decode_crevf_artifact(artifact)
    assert accounting.reduction_ppm == ppm
    assert accounting.target_checkpoint_bytes == 10_000 * (1_000_000 - ppm) // 1_000_000
    assert accounting.artifact_physical_bytes == len(artifact)
    assert accounting.checkpoint_physical_bytes == len(artifact) + 137
    assert accounting.runtime_packed_bytes == runtime.packed_runtime_bytes
    assert accounting.total_runtime_bytes == runtime.packed_runtime_bytes + 29
    assert (
        accounting.total_decoded_full_state_bytes
        == runtime.decoded_full_state_bytes + 31
    )
    assert (
        accounting.maximum_layer_decode_transient_bytes
        == runtime.maximum_layer_decode_transient_bytes
    )
    assert accounting.headroom_bytes == (
        accounting.target_checkpoint_bytes - accounting.checkpoint_physical_bytes
    )


def test_target_and_accounting_reject_unsupported_or_exceeded_ceiling() -> None:
    assert (
        CrevfReductionTarget.from_reduction(0.50).checkpoint_ceiling(464_128_028)
        == 232_064_014
    )
    assert 232_064_014 - 214_133_416 == 17_930_598
    assert (
        CrevfReductionTarget.from_reduction(0.55).checkpoint_ceiling(464_128_028)
        == 208_857_612
    )
    assert (
        CrevfReductionTarget.from_reduction(0.60).checkpoint_ceiling(464_128_028)
        == 185_651_211
    )
    assert (
        CrevfReductionTarget.from_reduction(0.65).checkpoint_ceiling(464_128_028)
        == 162_444_809
    )
    with pytest.raises(CrevfFormatError):
        decode_crevf_artifact(b"not an artifact")
    with pytest.raises(ValueError, match="500000"):
        CrevfReductionTarget.from_reduction(0.49)
    values = torch.ones(16, dtype=torch.bfloat16)
    artifact = encode_crevf_artifact([_quantized_blob("w", "ffn_w2", values, 4)])
    runtime_bytes = decode_crevf_artifact(artifact).packed_runtime_bytes
    decoded_full_state_bytes = decode_crevf_artifact(artifact).decoded_full_state_bytes
    layer_transient_bytes = decode_crevf_artifact(
        artifact
    ).maximum_layer_decode_transient_bytes
    sealed = account_crevf_artifact(
        artifact,
        dense_reference_bytes=10_000,
        target_reduction=0.65,
        must_beat_physical_bytes=len(artifact) + 1,
        max_persistent_encoded_bytes=runtime_bytes,
        max_decoded_full_state_bytes=decoded_full_state_bytes,
        max_layer_transient_bytes=layer_transient_bytes,
    )
    assert sealed.effective_checkpoint_ceiling_bytes == len(artifact)
    assert sealed.effective_headroom_bytes == 0
    assert sealed.runtime_headroom_bytes == 0
    assert sealed.decoded_full_state_headroom_bytes == 0
    assert sealed.layer_transient_headroom_bytes == 0
    with pytest.raises(CrevfBudgetError, match="effective"):
        account_crevf_artifact(
            artifact,
            dense_reference_bytes=10_000,
            target_reduction=0.65,
            must_beat_physical_bytes=len(artifact),
        )
    with pytest.raises(CrevfBudgetError, match="runtime"):
        account_crevf_artifact(
            artifact,
            dense_reference_bytes=10_000,
            target_reduction=0.65,
            max_persistent_encoded_bytes=runtime_bytes - 1,
        )
    with pytest.raises(CrevfBudgetError, match="decoded full-state"):
        account_crevf_artifact(
            artifact,
            dense_reference_bytes=10_000,
            target_reduction=0.65,
            max_decoded_full_state_bytes=decoded_full_state_bytes - 1,
        )
    with pytest.raises(CrevfBudgetError, match="layer group"):
        account_crevf_artifact(
            artifact,
            dense_reference_bytes=10_000,
            target_reduction=0.65,
            max_layer_transient_bytes=layer_transient_bytes - 1,
        )
    with pytest.raises(CrevfBudgetError, match="exceeding"):
        account_crevf_artifact(
            artifact,
            dense_reference_bytes=100,
            target_reduction=0.65,
        )


def test_decode_budget_rejects_before_entropy_inflation(monkeypatch) -> None:
    values = torch.zeros(2048, dtype=torch.bfloat16)
    blob = _quantized_blob(
        "model.layers.0.feed_forward.w2.weight",
        "ffn_w2",
        values,
        (8,) * 128,
    )
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("entropy decoder must not run after preflight failure")

    monkeypatch.setattr(
        "neural_forge.hf_lfm2_causal_residual_entropy_value_fold._decode_entropy",
        fail_if_called,
    )
    with pytest.raises(CrevfBudgetError, match="pre-inflate"):
        decode_crevf_tensor(blob, max_decoded_code_bytes=2047)
    assert called is False


def test_artifact_layer_budget_stops_before_inflating_failing_tensor(
    monkeypatch,
) -> None:
    values = torch.zeros(128, dtype=torch.bfloat16)
    first = _quantized_blob(
        "model.layers.0.feed_forward.w1.weight",
        "ffn_w1",
        values,
        (8,) * 8,
    )
    second = _quantized_blob(
        "model.layers.0.feed_forward.w2.weight",
        "ffn_w2",
        values,
        (8,) * 8,
    )
    artifact = encode_crevf_artifact((first, second))
    original = _decode_entropy
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "neural_forge.hf_lfm2_causal_residual_entropy_value_fold._decode_entropy",
        counted,
    )
    with pytest.raises(CrevfBudgetError, match="layer group"):
        decode_crevf_artifact(
            artifact,
            max_layer_transient_bytes=255,
        )
    assert calls == 1
