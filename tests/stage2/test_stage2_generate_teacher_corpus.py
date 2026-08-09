"""The stage-2 corpus generator's screen and manifest logic, on CPU alone.

Two properties carry the whole corpus's evidence value and are therefore tested
without a model, a GPU, or a shard on disk larger than a few kilobytes:

* the frozen V5 review surface cannot reach the training corpus, and the screen
  that enforces that reproduces the digest sets already sealed in the fit-prompt
  pack rather than a set this module invented; and
* a shard is only ever "done" when a manifest that seals its own body also
  matches the files on disk, because resume trusts exactly that.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments import stage2_generate_teacher_corpus as generator  # noqa: E402


ROUTED_REVIEW = ROOT / "research" / "lfm2_semantic_whole_response_prompts_v1.json"
FUTURE_REVIEW = ROOT / "research" / "lfm2_causal_support_future_manual_review_v1.json"
FIT_PROMPTS = ROOT / "experiments" / "lfm2_on_policy_fit_prompts_v1.json"
# The local mirror of the pinned PACK calibration corpus; it is an ignored
# staging directory, so the calibration half of the screen skips when absent.
CALIBRATION_MIRROR = ROOT / "upload_staging" / "target_tokenized_calibration.json"


@pytest.fixture(scope="module")
def screen() -> generator.ReviewScreen:
    return generator.build_review_screen(
        routed_review_path=ROUTED_REVIEW,
        future_review_path=FUTURE_REVIEW,
        calibration_path=None,
        parent_outputs_path=None,
    )


# --------------------------------------------------------------------------
# the contamination screen
# --------------------------------------------------------------------------


def test_screen_reproduces_the_sealed_reserved_digest_set(screen) -> None:
    recorded = screen.provenance["reserved_review_screen"]
    assert recorded["digest_count"] == 75
    assert (
        recorded["digest_set_sha256"] == generator.RESERVED_REVIEW_DIGEST_SET_SHA256
    )
    assert recorded["sources"] == generator.RESERVED_REVIEW_SOURCE_SHA256
    # The fit-prompt pack recorded the same set from the same two documents.
    pack = json.loads(FIT_PROMPTS.read_text(encoding="utf-8"))
    fit = pack["reserved_manual_review_disjointness"]
    assert fit["reserved_digest_count"] == recorded["digest_count"]
    assert fit["reserved_digest_set_sha256"] == recorded["digest_set_sha256"]
    assert fit["reserved_prompt_rendering_count"] == recorded["rendering_count"]


def test_every_v5_prompt_is_dropped_when_spliced_into_a_sequence(screen) -> None:
    probes = generator.review_probe_texts(ROUTED_REVIEW, 16)
    assert len(probes) == 16
    for probe in probes:
        assert "reserved_review_containment" in screen.reserved_drop_reasons(probe)


def test_a_rewrapped_v5_prompt_is_still_dropped(screen) -> None:
    # A teacher sample that reflows a review prompt across different line
    # breaks would defeat an exact-substring screen; the normalized pass is
    # what catches it.
    document = json.loads(ROUTED_REVIEW.read_text(encoding="utf-8"))
    content = document["prompts"][6]["messages"][0]["content"]
    rewrapped = "\n   ".join(content.split())
    assert rewrapped not in content
    assert "reserved_review_containment" in screen.reserved_drop_reasons(
        f"preamble\n{rewrapped}\npostamble"
    )


def test_a_whole_v5_prompt_is_dropped_by_digest_as_well(screen) -> None:
    document = json.loads(ROUTED_REVIEW.read_text(encoding="utf-8"))
    content = document["prompts"][0]["messages"][0]["content"]
    reasons = screen.reserved_drop_reasons(f"  {content}  ")
    assert "reserved_review_digest" in reasons
    assert "reserved_review_containment" in reasons


def test_ordinary_teacher_text_survives_the_screen(screen) -> None:
    clean = (
        "The harbour master kept a ledger of every vessel that entered the "
        "channel after dusk, and the ledger outlived him by forty years.\n"
    ) * 20
    assert screen.reserved_drop_reasons(clean) == ()
    assert screen.drop_reasons(clean) == ()


def test_a_short_review_fragment_is_not_a_containment_needle(screen) -> None:
    # Needles below the minimum length would drop ordinary text for no
    # contamination reason; the review surface's shortest message is far above
    # it, so nothing is lost by the guard.
    assert generator.MINIMUM_CONTAINMENT_NEEDLE_CHARACTERS == 48
    assert min(len(needle) for needle in screen.reserved_needles) >= 48


def test_the_screen_refuses_a_tampered_review_pack(tmp_path) -> None:
    document = json.loads(ROUTED_REVIEW.read_text(encoding="utf-8"))
    document["prompts"] = document["prompts"][:8]
    tampered = tmp_path / "routed.json"
    tampered.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(generator.Stage2TeacherCorpusError):
        generator.build_review_screen(
            routed_review_path=tampered,
            future_review_path=FUTURE_REVIEW,
            calibration_path=None,
            parent_outputs_path=None,
        )


@pytest.mark.skipif(
    not CALIBRATION_MIRROR.is_file(),
    reason="local PACK calibration mirror is absent",
)
def test_the_calibration_half_is_generated_only_and_reproduces_552(
    tmp_path,
) -> None:
    parent_outputs = _parent_outputs_path()
    if parent_outputs is None:
        pytest.skip("the pinned canonical parent outputs pack is not local")
    built = generator.build_review_screen(
        routed_review_path=ROUTED_REVIEW,
        future_review_path=FUTURE_REVIEW,
        calibration_path=CALIBRATION_MIRROR,
        parent_outputs_path=parent_outputs,
    )
    recorded = built.provenance["calibration_corpus_screen"]
    assert recorded["digest_count"] == generator.CALIBRATION_DIGEST_COUNT
    assert recorded["digest_set_sha256"] == generator.CALIBRATION_DIGEST_SET_SHA256
    assert recorded["applied_to_prefixes"] is False
    assert recorded["applied_to_generated_sequences"] is True
    calibration_text = json.loads(
        CALIBRATION_MIRROR.read_text(encoding="utf-8")
    )[0]["target_input_texts"][0]
    assert built.drop_reasons(calibration_text) == ("calibration_corpus_digest",)
    assert built.reserved_drop_reasons(calibration_text) == ()


def _parent_outputs_path() -> Path | None:
    for candidate in (
        ROOT / "upload_staging" / "canonical_parent_outputs_bridge_bound.json",
    ):
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------
# the prefix estate
# --------------------------------------------------------------------------


def test_the_estate_screens_a_contaminated_source_and_seals_its_body(
    tmp_path, screen
) -> None:
    document = json.loads(ROUTED_REVIEW.read_text(encoding="utf-8"))
    contaminated = document["prompts"][2]["messages"][0]["content"]
    pack = {
        "prompts": [
            {"prompt": "Describe a lighthouse keeper's morning in three sentences."},
            {"prompt": contaminated},
            {"prompt": "Explain why a bridge deck expands on a hot afternoon."},
        ]
    }
    pack_path = tmp_path / "fit.json"
    pack_path.write_text(json.dumps(pack), encoding="utf-8")

    estate = generator.build_prefix_estate(
        bridge_folding_corpus=None,
        fit_prompt_pack=pack_path,
        repository_corpora=(),
        screen=screen,
    )
    assert estate.provenance["text_count"] == 2
    assert estate.provenance["screened_out_text_count"] == 1
    assert contaminated not in estate.texts

    bundle = tmp_path / "estate.json"
    generator.write_prefix_estate(estate, bundle)
    reloaded = generator.read_prefix_estate(bundle)
    assert reloaded.texts == estate.texts
    assert reloaded.estate_sha256 == estate.estate_sha256


def test_a_tampered_estate_bundle_is_refused(tmp_path, screen) -> None:
    pack = {"prompts": [{"prompt": "Explain a tide table to a new deckhand."}]}
    pack_path = tmp_path / "fit.json"
    pack_path.write_text(json.dumps(pack), encoding="utf-8")
    estate = generator.build_prefix_estate(
        bridge_folding_corpus=None,
        fit_prompt_pack=pack_path,
        repository_corpora=(),
        screen=screen,
    )
    bundle = tmp_path / "estate.json"
    generator.write_prefix_estate(estate, bundle)

    document = json.loads(bundle.read_text(encoding="utf-8"))
    document["texts"].append("smuggled text")
    bundle.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(generator.Stage2TeacherCorpusError):
        generator.read_prefix_estate(bundle)


def test_a_repository_corpus_admits_only_byte_identical_files(
    tmp_path, screen
) -> None:
    kept = tmp_path / "kept.md"
    kept.write_text("A verified corpus file with stable bytes.\n", encoding="utf-8")
    drifted = tmp_path / "drifted.md"
    drifted.write_text("This file changed after the sealed run.\n", encoding="utf-8")
    evidence = {
        "files": [
            {"path": "kept.md", "sha256": generator._file_sha256(kept), "size": 41},
            {"path": "drifted.md", "sha256": "0" * 64, "size": 39},
            {"path": "vanished.md", "sha256": "1" * 64, "size": 10},
        ]
    }
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    estate = generator.build_prefix_estate(
        bridge_folding_corpus=None,
        fit_prompt_pack=None,
        repository_corpora=(("corpus", evidence_path, tmp_path),),
        screen=screen,
    )
    source = estate.provenance["sources"][0]
    assert source["verified_file_count"] == 1
    assert source["sha256_mismatched_file_count"] == 1
    assert source["missing_file_count"] == 1
    assert estate.texts == ("A verified corpus file with stable bytes.\n",)


def test_the_prefix_sampler_never_crosses_a_document_edge() -> None:
    import random

    documents = [list(range(0, 100)), list(range(1000, 1010)), list(range(2000, 2200))]
    sampler = generator._PrefixSampler(documents, prefix_tokens=64)
    # The 10-token document cannot host a 64-token window and is skipped.
    assert sampler.skipped_document_count == 1
    assert sampler.total_windows == (100 - 63) + (200 - 63)
    rng = random.Random(0)
    for _ in range(500):
        index, offset, window = sampler.draw(rng)
        assert len(window) == 64
        assert list(window) == documents[index][offset : offset + 64]


def test_the_prefix_sampler_is_deterministic_in_its_seed() -> None:
    import random

    documents = [list(range(0, 500)), list(range(5000, 5400))]
    sampler = generator._PrefixSampler(documents, prefix_tokens=64)
    first = [sampler.draw(random.Random(7))[:2] for _ in range(1)]
    rng = random.Random(11)
    series = [sampler.draw(rng)[:2] for _ in range(20)]
    rng = random.Random(11)
    assert [sampler.draw(rng)[:2] for _ in range(20)] == series
    assert first == [sampler.draw(random.Random(7))[:2]]


# --------------------------------------------------------------------------
# curation
# --------------------------------------------------------------------------


def test_three_gram_repetition_fraction_bounds() -> None:
    assert generator.three_gram_repetition_fraction([]) == 0.0
    assert generator.three_gram_repetition_fraction([1, 2]) == 0.0
    assert generator.three_gram_repetition_fraction(list(range(1024))) == 0.0
    degenerate = generator.three_gram_repetition_fraction([5] * 1024)
    assert degenerate > 0.99
    # Exactly half the trigrams repeat: the threshold is strictly greater-than,
    # so a sequence at 0.5 is kept.
    half = [0, 1, 2, 3] * 2
    fraction = generator.three_gram_repetition_fraction(half)
    assert 0.0 < fraction < 1.0


# --------------------------------------------------------------------------
# shard manifests and resume
# --------------------------------------------------------------------------


def _write_shard(shard_root: Path, index: int, split: str, rows: int = 2) -> dict:
    import numpy

    tokens = numpy.arange(rows * 8, dtype=numpy.uint16).reshape(rows, 8) + index
    payload = tokens.tobytes(order="C")
    text = (
        "\n".join(json.dumps({"row": row, "text": f"row {row}"}) for row in range(rows))
        + "\n"
    ).encode("utf-8")
    paths = generator.shard_paths(shard_root, index)
    generator._atomic_write_bytes(paths["tokens"], payload)
    generator._atomic_write_bytes(paths["text"], text)
    import hashlib

    manifest = generator.seal_shard_manifest(
        {
            "schema": generator.SHARD_MANIFEST_SCHEMA,
            "filter_statistics": {"accepted_count": rows, "dropped_total": 1},
            "generation": {"sequence_length": 8},
            "row_count": rows,
            "run": {"stage": "stage2"},
            "sequence_length": 8,
            "shard_index": index,
            "shard_seed": index,
            "split": split,
            "text_bytes": len(text),
            "text_sha256": hashlib.sha256(text).hexdigest(),
            "token_count": rows * 8,
            "tokens_bytes": len(payload),
            "tokens_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    generator._atomic_write_bytes(
        paths["manifest"], generator._canonical_json(manifest)
    )
    return manifest


def test_a_shard_manifest_seal_covers_its_own_body(tmp_path) -> None:
    manifest = _write_shard(tmp_path, 0, "train")
    loaded = generator.load_shard_manifest(
        generator.shard_paths(tmp_path, 0)["manifest"]
    )
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]

    path = generator.shard_paths(tmp_path, 0)["manifest"]
    mutated = json.loads(path.read_text(encoding="utf-8"))
    mutated["token_count"] = 999
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(generator.Stage2TeacherCorpusError):
        generator.load_shard_manifest(path)


def test_resume_treats_only_a_sealed_consistent_shard_as_done(tmp_path) -> None:
    assert generator.shard_is_complete(tmp_path, 0) is False
    _write_shard(tmp_path, 0, "train")
    assert generator.shard_is_complete(tmp_path, 0) is True
    assert generator.shard_is_complete(tmp_path, 0, verify_digests=True) is True

    # A truncated token file is caught by the recorded byte length alone.
    tokens = generator.shard_paths(tmp_path, 0)["tokens"]
    tokens.write_bytes(tokens.read_bytes()[:-2])
    assert generator.shard_is_complete(tmp_path, 0) is False


def test_resume_catches_same_length_corruption_only_when_verifying(tmp_path) -> None:
    _write_shard(tmp_path, 0, "train")
    tokens = generator.shard_paths(tmp_path, 0)["tokens"]
    payload = bytearray(tokens.read_bytes())
    payload[0] ^= 0xFF
    tokens.write_bytes(bytes(payload))
    assert generator.shard_is_complete(tmp_path, 0) is True
    assert generator.shard_is_complete(tmp_path, 0, verify_digests=True) is False


def test_a_missing_manifest_makes_resume_regenerate_that_shard(tmp_path) -> None:
    _write_shard(tmp_path, 0, "train")
    _write_shard(tmp_path, 1, "train")
    generator.shard_paths(tmp_path, 0)["manifest"].unlink()
    assert generator.shard_is_complete(tmp_path, 0) is False
    assert generator.shard_is_complete(tmp_path, 1) is True


# --------------------------------------------------------------------------
# the corpus manifest
# --------------------------------------------------------------------------


def test_the_corpus_manifest_seals_the_whole_inventory(tmp_path) -> None:
    for index in range(3):
        _write_shard(tmp_path, index, "train" if index < 2 else "holdout")
    manifest = generator.build_corpus_manifest(
        shard_root=tmp_path,
        run_provenance={"stage": "stage2", "teacher_shard_sha256": "a" * 64},
        verify_digests=True,
    )
    assert manifest["schema"] == generator.CORPUS_MANIFEST_SCHEMA
    assert manifest["shard_count"] == 3
    assert manifest["train_token_count"] == 2 * 2 * 8
    assert manifest["holdout_token_count"] == 2 * 8
    assert manifest["total_token_count"] == 3 * 2 * 8
    assert manifest["contiguous_shard_indices"] is True
    assert manifest["aggregate_filter_statistics"]["dropped_total"] == 3
    assert manifest["verified_shard_file_digests"] is True

    body = {
        key: value
        for key, value in manifest.items()
        if key != "corpus_manifest_sha256"
    }
    assert manifest["corpus_manifest_sha256"] == generator._canonical_sha256(body)


def test_the_corpus_manifest_refuses_a_corrupted_shard(tmp_path) -> None:
    _write_shard(tmp_path, 0, "train")
    tokens = generator.shard_paths(tmp_path, 0)["tokens"]
    payload = bytearray(tokens.read_bytes())
    payload[1] ^= 0xFF
    tokens.write_bytes(bytes(payload))
    with pytest.raises(generator.Stage2TeacherCorpusError):
        generator.build_corpus_manifest(
            shard_root=tmp_path, run_provenance={}, verify_digests=True
        )


def test_the_corpus_manifest_refuses_an_empty_shard_root(tmp_path) -> None:
    with pytest.raises(generator.Stage2TeacherCorpusError):
        generator.build_corpus_manifest(shard_root=tmp_path, run_provenance={})


def test_the_corpus_manifest_refuses_an_unknown_split(tmp_path) -> None:
    _write_shard(tmp_path, 0, "validation")
    with pytest.raises(generator.Stage2TeacherCorpusError):
        generator.build_corpus_manifest(shard_root=tmp_path, run_provenance={})


# --------------------------------------------------------------------------
# the donor vocabulary contract
# --------------------------------------------------------------------------


def test_the_donor_vocabulary_contract_is_pinned() -> None:
    assert (
        generator.EXPECTED_DEFINED_TOKEN_IDS
        + generator.EXPECTED_UNDEFINED_TOKEN_IDS
        == generator.EXPECTED_PHYSICAL_VOCAB_ROWS
    )
    assert generator.EXPECTED_DEFINED_TOKEN_IDS == 64402
    assert generator.EXPECTED_UNDEFINED_TOKEN_IDS == 1134
    # Every defined id fits the uint16 shard dtype.
    assert generator.EXPECTED_PHYSICAL_VOCAB_ROWS <= 2**16
