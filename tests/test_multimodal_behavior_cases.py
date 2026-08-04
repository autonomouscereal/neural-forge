from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from neural_forge.dynamic_behavior_gate import (
    AUDIO_HASH_BASIS,
    CANONICAL_MEDIA_HASH_BASES,
    IMAGE_HASH_BASIS,
    MEDIA_REFERENCE_SCHEMA,
    MODALITIES_WITHOUT_CANONICAL_HASH_BASIS,
    POINTS3D_HASH_BASIS,
    MULTIMODAL_CANDIDATE_SCHEMA,
    MULTIMODAL_CASE_SCHEMA,
    TEXT_HASH_BASIS,
    BehaviorGateBinding,
    CandidateWholeOutput,
    DynamicBehaviorGateError,
    MediaReference,
    MultimodalCandidateOutput,
    MultimodalWholeOutputCase,
    WholeOutputCase,
    WholeOutputMetadata,
    audio_content_sha256,
    build_multimodal_behavior_gate_binding,
    identify_behavior_callable,
    image_content_sha256,
    media_content_sha256,
    multimodal_case_baseline_fingerprint,
    multimodal_case_coverage,
    multimodal_cases_from_manifest,
    text_content_sha256,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256


def _pcm16(values: tuple[int, ...]) -> bytes:
    return b"".join(
        int(value).to_bytes(2, "little", signed=True) for value in values
    )


def _audio_reference(tmp_path: Path, samples: tuple[int, ...]) -> MediaReference:
    payload = _pcm16(samples)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "clip.pcm"
    path.write_bytes(payload)
    return MediaReference(
        modality="audio",
        content_sha256=audio_content_sha256(16_000, 1, len(samples), payload),
        uri=str(path),
        byte_count=path.stat().st_size,
        codec="audio/wav-pcm16-le",
        container_sha256=hashlib.sha256(payload).hexdigest(),
        attributes={"channels": 1, "frames": len(samples), "sample_rate": 16_000},
    )


def test_hash_bases_match_the_documented_decoded_content_formulas() -> None:
    assert IMAGE_HASH_BASIS == "sha256('RGB:<width>x<height>\\n' + decoded RGB bytes)"
    assert AUDIO_HASH_BASIS == (
        "sha256('PCM16:<rate>:<channels>:<frames>\\n' + little-endian int16 samples)"
    )
    assert CANONICAL_MEDIA_HASH_BASES["audio"] == AUDIO_HASH_BASIS
    assert CANONICAL_MEDIA_HASH_BASES["image"] == IMAGE_HASH_BASIS
    assert CANONICAL_MEDIA_HASH_BASES["text"] == TEXT_HASH_BASIS

    pixels = bytes(range(12))
    expected_image = hashlib.sha256(b"RGB:2x2\n" + pixels).hexdigest()
    assert image_content_sha256(2, 2, pixels) == expected_image

    samples = _pcm16((0, 1, -1, 32_767))
    expected_audio = hashlib.sha256(b"PCM16:16000:1:4\n" + samples).hexdigest()
    assert audio_content_sha256(16_000, 1, 4, samples) == expected_audio
    assert text_content_sha256("hi") == media_content_sha256("TEXT:utf-8", b"hi")


def test_audio_hash_binds_rate_channels_and_frame_count() -> None:
    samples = _pcm16((5, -5, 5, -5))
    baseline = audio_content_sha256(16_000, 1, 4, samples)
    assert audio_content_sha256(8_000, 1, 4, samples) != baseline
    assert audio_content_sha256(16_000, 2, 2, samples) != baseline
    with pytest.raises(DynamicBehaviorGateError):
        # Frame count must agree with the byte payload or the case is unsound.
        audio_content_sha256(16_000, 1, 3, samples)


def test_media_reference_round_trips_and_defaults_its_hash_basis(
    tmp_path: Path,
) -> None:
    reference = _audio_reference(tmp_path, (0, 100, -100, 0))
    assert reference.hash_basis == AUDIO_HASH_BASIS
    payload = reference.as_dict()
    assert payload["schema"] == MEDIA_REFERENCE_SCHEMA
    assert json.loads(json.dumps(payload)) == payload
    assert MediaReference.from_dict(payload) == reference
    assert MediaReference.from_dict(payload).fingerprint == reference.fingerprint


def test_media_reference_requires_an_explicit_basis_without_a_canonical_one() -> None:
    # ``points3d`` used to be the example here.  It gained a canonical basis
    # when the native points3d lane landed, so the example moved to the one
    # canonical modality that still deliberately owns no basis.  The module
    # declares that set rather than leaving it to be counted by hand.
    assert MODALITIES_WITHOUT_CANONICAL_HASH_BASIS == frozenset({"ocr"})
    with pytest.raises(DynamicBehaviorGateError):
        MediaReference(
            modality="ocr",
            content_sha256="0" * 64,
            uri="page.bin",
            byte_count=4,
            codec="raw",
        )
    reference = MediaReference(
        modality="ocr",
        content_sha256="0" * 64,
        uri="page.bin",
        byte_count=4,
        codec="raw",
        hash_basis="sha256('OCR:<width>x<height>\\n' + rendered page bytes)",
    )
    assert reference.modality == "ocr"


def test_points3d_now_resolves_a_canonical_basis() -> None:
    reference = MediaReference(
        modality="points3d",
        content_sha256="0" * 64,
        uri="cloud.nfp",
        byte_count=6,
        codec="points3d/nfpc1-int16",
    )
    assert reference.hash_basis == POINTS3D_HASH_BASIS
    assert CANONICAL_MEDIA_HASH_BASES["points3d"] == POINTS3D_HASH_BASIS
    # Every canonical modality is now either based or explicitly declared
    # unbased; nothing is merely undecided.
    assert (
        set(CANONICAL_MEDIA_HASH_BASES) | MODALITIES_WITHOUT_CANONICAL_HASH_BASIS
    ) == {"text", "image", "audio", "video", "ocr", "points3d"}


def test_media_reference_rejects_non_canonical_modalities() -> None:
    with pytest.raises(DynamicBehaviorGateError):
        MediaReference(
            modality="speech",
            content_sha256="0" * 64,
            uri="clip.wav",
            byte_count=1,
            codec="audio/wav-pcm16-le",
        )


def test_container_verification_fails_closed_on_tamper(tmp_path: Path) -> None:
    reference = _audio_reference(tmp_path, (1, 2, 3, 4))
    assert reference.verify_container() == Path(reference.uri).resolve()

    tampered = _pcm16((1, 2, 3, 5))
    Path(reference.uri).write_bytes(tampered)
    with pytest.raises(DynamicBehaviorGateError):
        reference.verify_container()

    # Re-decoding also fails closed, which is what a lane actually checks.
    with pytest.raises(DynamicBehaviorGateError):
        reference.verify_content(
            audio_content_sha256(16_000, 1, 4, tampered),
        )
    reference.verify_content(reference.content_sha256)


def test_local_path_refuses_to_escape_a_declared_root(tmp_path: Path) -> None:
    escaping = MediaReference(
        modality="audio",
        content_sha256="0" * 64,
        uri="../outside.wav",
        byte_count=2,
        codec="audio/wav-pcm16-le",
    )
    with pytest.raises(DynamicBehaviorGateError):
        escaping.local_path(tmp_path)
    remote = replace(escaping, uri="https://example.invalid/clip.wav")
    with pytest.raises(DynamicBehaviorGateError):
        remote.local_path()


def test_multimodal_case_round_trips_through_canonical_json(tmp_path: Path) -> None:
    reference = _audio_reference(tmp_path, (0, 7, -7, 0))
    case = MultimodalWholeOutputCase(
        case_id="tone_count_0000",
        prompt="How many tone bursts?",
        parent_output="2",
        inputs=(reference,),
        expected="2",
        category="tone_count",
        parent_eos_margin=1.5,
        critical_nonempty=True,
        parent_metadata=WholeOutputMetadata(
            generated_token_count=2,
            generation_limit=8,
            maximum_repeated_ngram_share=0.0,
        ),
        split="heldout",
    )
    payload = case.as_dict()
    assert payload["schema"] == MULTIMODAL_CASE_SCHEMA
    assert payload["modalities"] == ["audio", "text"]
    assert json.loads(json.dumps(payload)) == payload
    assert MultimodalWholeOutputCase.from_dict(payload) == case
    assert case.input_modalities == ("audio",)
    assert case.parent_output_is_media is False
    assert case.parent_empty is False


def test_multimodal_case_supports_media_on_the_output_side(tmp_path: Path) -> None:
    spoken = _audio_reference(tmp_path, (3, 3, 3, 3))
    case = MultimodalWholeOutputCase(
        case_id="tts",
        prompt="Say two.",
        parent_output=spoken,
        expected=spoken,
    )
    assert case.parent_output_is_media is True
    assert case.modalities == ("audio", "text")
    assert MultimodalWholeOutputCase.from_dict(case.as_dict()) == case

    empty = replace(spoken, byte_count=0)
    assert MultimodalWholeOutputCase(
        case_id="silent",
        prompt="Say nothing.",
        parent_output=empty,
    ).parent_empty


def test_multimodal_case_requires_media_somewhere() -> None:
    with pytest.raises(DynamicBehaviorGateError):
        MultimodalWholeOutputCase(
            case_id="text-only",
            prompt="Say hello.",
            parent_output="Hello.",
        )


def test_case_payload_tamper_changes_the_fingerprint(tmp_path: Path) -> None:
    reference = _audio_reference(tmp_path, (0, 9, -9, 0))
    case = MultimodalWholeOutputCase(
        case_id="probe",
        prompt="What is this?",
        parent_output="tone",
        inputs=(reference,),
        category="sound_type",
    )
    tampered = copy.deepcopy(case.as_dict())
    tampered["inputs"][0]["content_sha256"] = "1" * 64
    rebuilt = MultimodalWholeOutputCase.from_dict(tampered)
    assert rebuilt != case
    assert rebuilt.fingerprint != case.fingerprint

    modality_tamper = copy.deepcopy(case.as_dict())
    modality_tamper["modalities"] = ["text"]
    with pytest.raises(DynamicBehaviorGateError):
        MultimodalWholeOutputCase.from_dict(modality_tamper)

    unknown_field = copy.deepcopy(case.as_dict())
    unknown_field["extra"] = True
    with pytest.raises(DynamicBehaviorGateError):
        MultimodalWholeOutputCase.from_dict(unknown_field)

    wrong_schema = copy.deepcopy(case.as_dict())
    wrong_schema["schema"] = "neural-forge-something-else"
    with pytest.raises(DynamicBehaviorGateError):
        MultimodalWholeOutputCase.from_dict(wrong_schema)


def test_multimodal_candidate_round_trips_text_and_media(tmp_path: Path) -> None:
    text_candidate = MultimodalCandidateOutput("2", eos_margin=0.5)
    assert text_candidate.as_dict()["schema"] == MULTIMODAL_CANDIDATE_SCHEMA
    assert MultimodalCandidateOutput.from_dict(text_candidate.as_dict()) == (
        text_candidate
    )
    assert text_candidate.modality == "text"
    assert MultimodalCandidateOutput("   ").empty

    media_candidate = MultimodalCandidateOutput(
        _audio_reference(tmp_path, (1, 1, 1, 1)),
        metadata=WholeOutputMetadata(generated_token_count=0, generation_limit=4),
    )
    assert media_candidate.is_media
    assert media_candidate.modality == "audio"
    assert MultimodalCandidateOutput.from_dict(media_candidate.as_dict()) == (
        media_candidate
    )


def test_binding_mirrors_the_text_gate_manifest_and_seals_cases(
    tmp_path: Path,
) -> None:
    cases = (
        MultimodalWholeOutputCase(
            case_id="calibration_0",
            prompt="How many tone bursts?",
            parent_output="1",
            inputs=(_audio_reference(tmp_path / "a", (0, 5, 0, 0)),),
            expected="1",
            category="tone_count",
            split="calibration",
        ),
        MultimodalWholeOutputCase(
            case_id="heldout_0",
            prompt="Is this silence, a tone, or noise?",
            parent_output="noise",
            inputs=(_audio_reference(tmp_path / "b", (7, -7, 7, -7)),),
            expected="noise",
            category="sound_type",
            split="heldout",
        ),
    )
    generator_identity = identify_behavior_callable(lambda model, values: {})
    binding = build_multimodal_behavior_gate_binding(
        gate_id="neural_forge.test.multimodal",
        gate_version=1,
        cases=cases,
        threshold_config={"require_all_parent_correct_retained": True},
        generator_identity=generator_identity,
        scorer_identity=None,
        coverage_policy={"require_any_parent_correct_case": True},
    )
    assert isinstance(binding, BehaviorGateBinding)
    assert binding.baseline_fingerprint == multimodal_case_baseline_fingerprint(cases)
    assert multimodal_cases_from_manifest(binding.manifest) == cases

    coverage = [
        component
        for component in binding.manifest["components"]
        if component["schema"] == multimodal_case_coverage(cases)["schema"]
    ]
    assert coverage == [multimodal_case_coverage(cases)]
    assert coverage[0]["input_modalities"] == ["audio"]
    assert coverage[0]["output_modalities"] == ["text"]

    # The seal is the text gate's seal: recomputing the body must reproduce it.
    body = {
        key: value
        for key, value in binding.manifest.items()
        if key != "manifest_fingerprint"
    }
    assert canonical_sha256(body) == binding.manifest_fingerprint

    tampered = dict(binding.manifest)
    tampered_cases = copy.deepcopy(list(tampered["cases"]))
    tampered_cases[0]["parent_output"]["text"] = "2"
    tampered["cases"] = tampered_cases
    with pytest.raises(DynamicBehaviorGateError):
        BehaviorGateBinding(
            gate_id=binding.gate_id,
            gate_version=binding.gate_version,
            baseline_fingerprint=binding.baseline_fingerprint,
            manifest_fingerprint=binding.manifest_fingerprint,
            manifest=tampered,
        )


def test_existing_text_only_case_types_are_untouched() -> None:
    case = WholeOutputCase("chat", "Say hello.", "Hello!", category="conversation")
    assert case.as_dict()["prompt"] == "Say hello."
    assert "schema" not in case.as_dict()
    assert CandidateWholeOutput("").empty
    assert not isinstance(case, MultimodalWholeOutputCase)
