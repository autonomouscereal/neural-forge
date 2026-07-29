from __future__ import annotations

import hashlib

import pytest


torch = pytest.importorskip("torch")


def _record_id(index: int, namespace: str = "fit") -> str:
    return hashlib.sha256(
        f"v2-margin-{namespace}-record-{index}".encode("ascii")
    ).hexdigest()


def _fixture(
    horizons=(3, 2, 4, 3),
    *,
    width: int = 6,
    vocabulary_size: int = 19,
    split=None,
    namespace: str = "fit",
):
    from neural_forge import hf_lfm2_streaming_vocabulary_margin as streaming
    from neural_forge.dynamic_balanced_circuit_basis import EvidenceSplit

    if split is None:
        split = EvidenceSplit.FIT

    generator = torch.Generator().manual_seed(20260729)
    head = torch.randn(
        vocabulary_size,
        width,
        dtype=torch.float64,
        generator=generator,
    )
    norm = (
        torch.rand(width, dtype=torch.float64, generator=generator) + 0.5
    )
    descriptors = []
    batches = []
    raw = []
    for record_index, phase_count in enumerate(horizons):
        phases = ("prefill",) + tuple(
            f"decode.{index:03d}" for index in range(phase_count - 1)
        )
        for phase_index, phase in enumerate(phases):
            row_count = 5 + record_index + phase_index
            residual = torch.randn(
                1,
                row_count,
                width,
                dtype=torch.float64,
                generator=generator,
            )
            logits = torch.randn(
                1,
                row_count,
                vocabulary_size,
                dtype=torch.float64,
                generator=generator,
            ) * 0.25
            logits[0, 0, :4] = 4.0
            descriptor = (
                streaming.Lfm2StreamingVocabularyMarginPhaseDescriptor(
                    record_index=record_index,
                    record_id=_record_id(record_index, namespace),
                    phase=phase,
                    row_count=row_count,
                    residual_tensor_sha256=(
                        streaming._authenticated_trace_tensor_sha256(residual)
                    ),
                    logits_tensor_sha256=(
                        streaming._authenticated_trace_tensor_sha256(logits)
                    ),
                )
            )
            descriptors.append(descriptor)
            batches.append(
                streaming.Lfm2StreamingVocabularyMarginPhaseBatch(
                    descriptor=descriptor,
                    final_residual=residual,
                    logits=logits,
                )
            )
            raw.append((descriptor, residual, logits))
    manifest = streaming.Lfm2StreamingVocabularyMarginManifest(
        split=split,
        phases=tuple(descriptors),
    )
    return head, norm, manifest, tuple(batches), tuple(raw)


def _legacy_margin_evidence(
    head,
    norm,
    raw,
    *,
    top_k: int,
    margin_floor_fraction: float,
):
    from neural_forge import hf_lfm2_balanced_circuit_basis as legacy
    from neural_forge.dynamic_balanced_circuit_basis import EvidenceSplit
    from neural_forge.dynamic_weight_complete_basis import (
        FeatureOrientation,
        ResidualFacingOperator,
    )

    by_record = {}
    for descriptor, residual, logits in raw:
        value = by_record.setdefault(
            descriptor.record_index,
            {
                "record_id": descriptor.record_id,
                "locations": {},
                "margins": {},
            },
        )
        value["locations"][(descriptor.phase, 1, "post_layer")] = (
            residual.reshape(-1, head.shape[1]).contiguous()
        )
        value["margins"][descriptor.phase] = (
            legacy._deterministic_top_tokens(
                logits,
                vocabulary_size=int(head.shape[0]),
                top_k=top_k,
                phase=descriptor.phase,
            )
        )
    records = tuple(
        legacy._CacheRecord(
            index=index,
            record_id=value["record_id"],
            locations=value["locations"],
            vocabulary_margins=value["margins"],
        )
        for index, value in sorted(by_record.items())
    )
    lexical = ResidualFacingOperator(
        role="lexical_output_interface",
        name="lexical.tied_embedding_head.weight",
        tensor=head,
        orientation=FeatureOrientation.INPUT,
        feature_axis=1,
    )
    final_norm = ResidualFacingOperator(
        role="normalization_scale",
        name="output.final_norm.weight",
        tensor=norm,
        orientation=FeatureOrientation.INPUT,
        diagonal=True,
    )
    return legacy._build_vocabulary_margin_direction_set(
        records,
        split=EvidenceSplit.FIT,
        lexical_head=lexical,
        final_norm=final_norm,
        layer_count=2,
        top_k=top_k,
        margin_floor_fraction=margin_floor_fraction,
    )


def test_v2_authenticated_hash_matches_trace_cache_schema() -> None:
    from neural_forge import hf_lfm2_circuit_trace_cache as trace_cache
    from neural_forge import hf_lfm2_streaming_vocabulary_margin as streaming

    generator = torch.Generator().manual_seed(73)
    for value in (
        torch.randn(2, 3, 5, dtype=torch.float64, generator=generator),
        torch.randn(4, 7, dtype=torch.float32, generator=generator).T,
    ):
        assert streaming._authenticated_trace_tensor_sha256(value) == (
            trace_cache._tensor_sha256(torch, value)
        )


def test_v2_all_row_statistics_match_legacy_semantics_and_decisions() -> None:
    from neural_forge import hf_lfm2_balanced_circuit_basis as legacy
    from neural_forge import hf_lfm2_streaming_vocabulary_margin as streaming

    head, norm, manifest, batches, raw = _fixture()
    config = streaming.Lfm2StreamingVocabularyMarginConfig(
        top_k=3,
        margin_floor_fraction=0.05,
        pair_chunk_size=7,
        maximum_phase_rows=32,
    )
    result = streaming.build_streaming_vocabulary_margin_statistics_v2(
        manifest,
        iter(batches),
        lexical_head=head,
        final_norm=norm,
        config=config,
    )
    reference = _legacy_margin_evidence(
        head,
        norm,
        raw,
        top_k=config.top_k,
        margin_floor_fraction=config.margin_floor_fraction,
    )

    for descriptor, _residual, logits in raw:
        old_table = legacy._deterministic_top_tokens(
            logits,
            vocabulary_size=int(head.shape[0]),
            top_k=config.top_k,
            phase=descriptor.phase,
        )
        new_table = streaming.deterministic_top_token_table_v2(
            logits,
            top_k=config.top_k,
        )
        assert torch.equal(new_table.token_ids, old_table.token_ids)
        assert torch.equal(new_table.logit_values, old_table.logit_values)

    assert torch.allclose(
        result.second_moment_gram,
        reference.gram,
        rtol=2e-12,
        atol=2e-13,
    )
    maximum_gram_delta = float(
        (result.second_moment_gram - reference.gram).abs().max().item()
    )
    assert maximum_gram_delta < 2e-12

    basis = torch.linalg.qr(
        torch.randn(
            head.shape[1],
            3,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(91),
        )
    ).Q
    legacy_retained = float(
        torch.trace(basis.T @ reference.gram @ basis).item()
        / torch.trace(reference.gram).item()
    )
    v2_retained = result.retained_energy_fraction(basis)
    assert v2_retained == pytest.approx(legacy_retained, abs=2e-12)
    for threshold in (
        max(0.0, legacy_retained - 1e-6),
        min(1.0, legacy_retained + 1e-6),
    ):
        assert (v2_retained + 1e-15 >= threshold) == (
            legacy_retained + 1e-15 >= threshold
        )

    report = result.as_dict()
    legacy_phases = reference.report["phases"]
    assert report["logit_row_count"] == reference.report["logit_row_count"]
    assert report["direction_count"] == reference.report["direction_count"]
    assert report["zero_direction_count_skipped"] == reference.report[
        "zero_direction_count_skipped"
    ]
    assert len(report["phases"]) == len(legacy_phases)
    for actual, expected in zip(report["phases"], legacy_phases, strict=True):
        for name in (
            "record_index",
            "record_id",
            "phase",
            "logit_row_count",
            "top_k_competitor_count",
            "direction_count",
            "zero_direction_count_skipped",
        ):
            assert actual[name] == expected[name]
        for name in (
            "hierarchical_phase_weight",
            "margin_floor",
            "minimum_cached_margin",
            "median_cached_margin",
            "maximum_cached_margin",
        ):
            assert actual[name] == pytest.approx(expected[name], abs=1e-15)
    assert report["all_authenticated_rows_consumed"] is True
    assert report["row_cap_or_subsample_applied"] is False
    assert report["integration_status"]["default_basis_builder_changed"] is False


def test_v2_is_deterministic_and_retained_tensor_size_is_row_independent() -> None:
    from neural_forge import hf_lfm2_streaming_vocabulary_margin as streaming

    config = streaming.Lfm2StreamingVocabularyMarginConfig(
        top_k=2,
        pair_chunk_size=5,
        maximum_phase_rows=32,
    )
    head, norm, manifest, batches, _raw = _fixture(horizons=(2, 2))
    first = streaming.build_streaming_vocabulary_margin_statistics_v2(
        manifest,
        iter(batches),
        lexical_head=head,
        final_norm=norm,
        config=config,
    )
    second = streaming.build_streaming_vocabulary_margin_statistics_v2(
        manifest,
        iter(batches),
        lexical_head=head,
        final_norm=norm,
        config=config,
    )
    assert torch.equal(first.second_moment_gram, second.second_moment_gram)
    assert torch.equal(first.weighted_mean, second.weighted_mean)
    assert first.as_dict() == second.as_dict()

    large_head, large_norm, large_manifest, large_batches, _raw = _fixture(
        horizons=(2,) * 16
    )
    large = streaming.build_streaming_vocabulary_margin_statistics_v2(
        large_manifest,
        iter(large_batches),
        lexical_head=large_head,
        final_norm=large_norm,
        config=config,
    )
    expected_numel = int(head.shape[1] ** 2 + head.shape[1])
    assert first.retained_tensor_numel == expected_numel
    assert large.retained_tensor_numel == expected_numel
    assert large.as_dict()["logit_row_count"] > first.as_dict()[
        "logit_row_count"
    ]
    assert large.as_dict()["retention"]["full_direction_rows_retained"] == 0
    assert large.as_dict()["retention"]["full_residual_rows_retained"] == 0
    assert large.as_dict()["retention"]["full_top_token_tables_retained"] == 0


def test_v2_fails_closed_on_row_alignment_and_phase_bound() -> None:
    from neural_forge import hf_lfm2_streaming_vocabulary_margin as streaming

    head, norm, manifest, batches, _raw = _fixture(horizons=(2, 2))
    first = batches[0]
    misaligned = streaming.Lfm2StreamingVocabularyMarginPhaseBatch(
        descriptor=first.descriptor,
        final_residual=first.final_residual,
        logits=first.logits[:, :-1],
    )
    with pytest.raises(
        streaming.Lfm2StreamingVocabularyMarginError,
        match="not aligned",
    ):
        streaming.build_streaming_vocabulary_margin_statistics_v2(
            manifest,
            iter((misaligned, *batches[1:])),
            lexical_head=head,
            final_norm=norm,
            config=streaming.Lfm2StreamingVocabularyMarginConfig(
                top_k=2,
                maximum_phase_rows=32,
            ),
        )

    with pytest.raises(
        streaming.Lfm2StreamingVocabularyMarginError,
        match="not aligned",
    ):
        streaming.build_streaming_vocabulary_margin_statistics_v2(
            manifest,
            iter(batches),
            lexical_head=head,
            final_norm=norm,
            config=streaming.Lfm2StreamingVocabularyMarginConfig(
                top_k=2,
                maximum_phase_rows=1,
            ),
        )


def test_v2_fit_manifest_is_independent_of_heldout_evidence() -> None:
    from neural_forge import hf_lfm2_streaming_vocabulary_margin as streaming
    from neural_forge.dynamic_balanced_circuit_basis import EvidenceSplit

    config = streaming.Lfm2StreamingVocabularyMarginConfig(
        top_k=2,
        pair_chunk_size=4,
        maximum_phase_rows=32,
    )
    head, norm, fit_manifest, fit_batches, _raw = _fixture(horizons=(2, 3))
    baseline = streaming.build_streaming_vocabulary_margin_statistics_v2(
        fit_manifest,
        iter(fit_batches),
        lexical_head=head,
        final_norm=norm,
        config=config,
    )
    _other_head, _other_norm, heldout_manifest, heldout_batches, _raw = _fixture(
        horizons=(4, 2, 3),
        split=EvidenceSplit.HELDOUT,
        namespace="heldout",
    )
    assert heldout_manifest.split is EvidenceSplit.HELDOUT
    assert heldout_manifest.manifest_sha256 != fit_manifest.manifest_sha256
    assert len(heldout_batches) != len(fit_batches)
    repeated = streaming.build_streaming_vocabulary_margin_statistics_v2(
        fit_manifest,
        iter(fit_batches),
        lexical_head=head,
        final_norm=norm,
        config=config,
    )
    assert repeated.as_dict() == baseline.as_dict()
    assert torch.equal(
        repeated.second_moment_gram,
        baseline.second_moment_gram,
    )
