"""The r12 on-policy fit corpus stays disjoint from every reserved corpus.

The behavioral review set is the only thing that grades a folded child, so fit
evidence that touches it would grade the child on its own training surface.
``hf_lfm2_on_policy_evidence_round`` enforces that at round time; these tests
enforce it at commit time, against the same digest rule, so a staged corpus
cannot reach the server and fail there.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from neural_forge.hf_lfm2_on_policy_evidence_round import (  # noqa: E402
    Lfm2OnPolicyEvidenceRoundDisjointnessError,
    _normalized_fit_prompts,
    _prompt_corpus_evidence,
    _verified_disjointness,
    load_manual_review_prompt_manifest,
)

CORPUS_PATH = ROOT / "experiments" / "lfm2_on_policy_fit_prompts_v1.json"
MANUAL_DIGEST_PATH = (
    ROOT / "experiments" / "lfm2_manual_review_reserved_digests_v1.json"
)
ROUTED_MANUAL_PATH = (
    ROOT / "research" / "lfm2_semantic_whole_response_prompts_v1.json"
)
FUTURE_MANUAL_PATH = (
    ROOT / "research" / "lfm2_causal_support_future_manual_review_v1.json"
)
# The local mirror of the PACK calibration corpus the r12 launcher pins. It is
# an ignored staging directory, so the calibration check skips when absent.
CALIBRATION_MIRROR = (
    ROOT / "upload_staging" / "target_tokenized_calibration.json"
)

# The behavioral dimensions BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23 names.
REQUIRED_DIMENSIONS = (
    "coherence",
    "task-correctness-arithmetic",
    "task-correctness-code",
    "reasoning",
    "instruction-following",
    "retrieval",
    "multi-turn",
    "explanation-summarization",
)


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest_set(texts) -> set[str]:
    """Exact and whitespace-stripped digests, the round's comparison rule."""

    digests: set[str] = set()
    for value in texts:
        digests.add(_text_sha256(value))
        digests.add(_text_sha256(value.strip()))
    return digests


def _reserved_renderings(path: Path) -> list[str]:
    """Every plain-text rendering of one reserved review-prompt document.

    The routed set is chat messages and the future set is flat text, so a fit
    prompt could collide with either a single message or a transcript of the
    whole exchange. Both are compared.
    """

    document = json.loads(path.read_text(encoding="utf-8"))
    texts: list[str] = []
    for row in document["prompts"]:
        if "text" in row:
            texts.append(row["text"])
            continue
        messages = row["messages"]
        texts.extend(message["content"] for message in messages)
        texts.append("\n\n".join(m["content"] for m in messages))
        texts.append("\n".join(f"{m['role']}: {m['content']}" for m in messages))
        texts.append(
            "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
        )
    return texts


@pytest.fixture(scope="module")
def corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fit_prompts(corpus) -> tuple[str, ...]:
    return tuple(row["prompt"] for row in corpus["prompts"])


def test_corpus_seal_covers_its_own_body(corpus) -> None:
    body = {key: value for key, value in corpus.items() if key != "corpus_sha256"}
    assert corpus["corpus_sha256"] == _canonical_sha256(body)


def test_corpus_is_bounded_and_covers_every_review_dimension(corpus) -> None:
    prompts = corpus["prompts"]
    assert 96 <= len(prompts) <= 128
    assert corpus["prompt_count"] == len(prompts)
    counts: dict[str, int] = {}
    for row in prompts:
        counts[row["dimension"]] = counts.get(row["dimension"], 0) + 1
    assert counts == corpus["dimension_counts"]
    assert set(counts) == set(REQUIRED_DIMENSIONS)
    assert min(counts.values()) >= 8
    assert len({row["id"] for row in prompts}) == len(prompts)


def test_every_prompt_survives_the_round_normalizer(fit_prompts) -> None:
    # Duplicate, empty, and over-budget corpora are round-time failures; the
    # round's own normalizer is the authority, so it is what runs here.
    normalized = _normalized_fit_prompts(fit_prompts, 128)
    assert normalized == fit_prompts
    for prompt in fit_prompts:
        # The child tokenizer must round-trip every seed prompt byte-exactly
        # (strict_byte_identity), and the target vocabulary is only 4,096 rows.
        assert prompt.isascii()
        assert prompt == prompt.strip()


def test_fit_prompts_are_disjoint_from_the_reserved_review_sets(
    corpus, fit_prompts
) -> None:
    renderings = _reserved_renderings(ROUTED_MANUAL_PATH) + _reserved_renderings(
        FUTURE_MANUAL_PATH
    )
    reserved = _digest_set(renderings)
    overlap = [
        row["id"]
        for row in corpus["prompts"]
        if _text_sha256(row["prompt"]) in reserved
        or _text_sha256(row["prompt"].strip()) in reserved
    ]
    assert overlap == []

    recorded = corpus["reserved_manual_review_disjointness"]
    assert recorded["verified_disjoint"] is True
    assert recorded["overlapping_prompt_count"] == 0
    assert recorded["comparison"] == "utf8-sha256-exact-and-stripped"
    assert recorded["reserved_prompt_rendering_count"] == len(renderings)
    assert recorded["reserved_digest_count"] == len(reserved)
    assert recorded["reserved_digest_set_sha256"] == _canonical_sha256(
        sorted(reserved)
    )
    for name, digest in recorded["sources"].items():
        source = ROOT / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_sealed_reserved_digest_manifest_proves_disjointness(fit_prompts) -> None:
    # This is the exact contract the round applies: the reserved manifest is
    # loaded by the library loader and every fit prompt is hashed against it.
    manifest = load_manual_review_prompt_manifest(MANUAL_DIGEST_PATH)
    assert manifest.form == "sealed-digest-manifest"
    proof = _verified_disjointness(fit_prompts, manifest)
    assert proof["verified_disjoint"] is True
    assert proof["overlapping_prompt_count"] == 0
    assert proof["comparison"] == "utf8-sha256-exact-and-stripped"


def test_reserved_digest_manifest_covers_both_reserved_documents() -> None:
    manifest = load_manual_review_prompt_manifest(MANUAL_DIGEST_PATH)
    renderings = _reserved_renderings(ROUTED_MANUAL_PATH) + _reserved_renderings(
        FUTURE_MANUAL_PATH
    )
    assert manifest.digests == frozenset(_digest_set(renderings))


def test_a_reserved_prompt_is_still_refused_by_the_manifest() -> None:
    # Guards the manifest itself: if it stopped covering the reserved corpora,
    # every disjointness assertion above would pass vacuously.
    manifest = load_manual_review_prompt_manifest(MANUAL_DIGEST_PATH)
    reserved = json.loads(ROUTED_MANUAL_PATH.read_text(encoding="utf-8"))
    smuggled = reserved["prompts"][3]["messages"][0]["content"]
    with pytest.raises(Lfm2OnPolicyEvidenceRoundDisjointnessError):
        _verified_disjointness(("alpha beta gamma", f"  {smuggled}  "), manifest)


def test_recorded_prompt_corpus_digest_matches_the_round_evidence(
    corpus, fit_prompts
) -> None:
    evidence = _prompt_corpus_evidence(fit_prompts)
    assert evidence["prompt_count"] == corpus["prompt_count"]
    assert evidence["prompt_corpus_sha256"] == (
        "834a95856f3c50f195e9ee51855b1a7bf0f05edcf80025d412f17b39f8389c26"
    )


@pytest.mark.skipif(
    not CALIBRATION_MIRROR.is_file(),
    reason="local PACK calibration mirror is absent",
)
def test_fit_prompts_are_disjoint_from_the_pinned_calibration_corpus(
    corpus, fit_prompts
) -> None:
    recorded = corpus["calibration_corpus_disjointness"]
    mirror_digest = hashlib.sha256(CALIBRATION_MIRROR.read_bytes()).hexdigest()
    assert recorded["sources"]["target_tokenized_calibration.json"] == (
        mirror_digest
    )
    texts: list[str] = []
    for row in json.loads(CALIBRATION_MIRROR.read_text(encoding="utf-8")):
        texts.extend(row.get("target_input_texts") or ())
        texts.extend(row.get("target_forced_decode_texts") or ())
    calibration = _digest_set(texts)
    overlap = [
        row["id"]
        for row in corpus["prompts"]
        if _text_sha256(row["prompt"]) in calibration
        or _text_sha256(row["prompt"].strip()) in calibration
    ]
    assert overlap == []
    assert recorded["verified_disjoint"] is True
    assert recorded["overlapping_prompt_count"] == 0


def test_staged_runner_rebuilds_the_r12_base_capture_selection() -> None:
    # The staged round runner derives its capture selection from the sealed
    # base manifest rather than from flags, because no catalog-runner flag set
    # reproduces this lane's selection. That reconstruction is load bearing.
    pytest.importorskip("torch")
    from experiments import run_r12_on_policy_round as runner

    sealed = {
        "allow_top1_disagreement_within_numeric_error_envelope": True,
        "attention_complete_kv_history": True,
        "attention_exact_causal_scores_and_probabilities": True,
        "attention_pre_rope_qk": True,
        "cached_prefill_use_cache": True,
        "capture_causal_circuit_evidence": True,
        "final_normalization_boundary": True,
        "forced_decode_use_cache": True,
        "full_prefix_verification_use_cache": False,
        "gradients_used": False,
        "inference_mode": True,
        "logit_atol": 1.0,
        "logit_rtol": 0.2,
        "max_capture_bytes": 536870912,
        "max_decode_steps": 8,
        "model_eval_mode": True,
        "prefill_use_cache": False,
        "require_top1_agreement": True,
        "sample_last_tokens": 256,
        "selection_sha256": (
            "f229e5f399858b7a512ec0d7034fe4b3d76cf4dc3bb214ec461a8a6f7c000258"
        ),
        "short_conv_exact_fir_state_and_response": True,
        "training_steps": 0,
    }
    rebuilt = runner._capture_config_from_sealed_manifest(sealed)
    assert rebuilt.as_dict()["selection_sha256"] == sealed["selection_sha256"]
    assert rebuilt.max_decode_steps == 8

    mutated = {**sealed, "max_decode_steps": 16}
    with pytest.raises(runner.R12OnPolicyRoundError):
        runner._capture_config_from_sealed_manifest(mutated)


def test_runner_receipt_thaws_frozen_manifest_rows_deeply() -> None:
    """A sealed round manifest is deep-frozen; the receipt writer must thaw
    it deeply or json.dumps dies on nested mappingproxy rows.  The first
    round with non-empty rejected_prompts/admissibility rows failed exactly
    there after a complete, sealed capture."""

    from types import MappingProxyType

    from experiments import run_r12_on_policy_round as runner

    frozen = MappingProxyType(
        {
            "rejected_prompts": (
                MappingProxyType(
                    {"fit_prompt_index": 4, "reason": "re-tokenization"}
                ),
            ),
            "excluded": (
                MappingProxyType(
                    {
                        "batch_index": 0,
                        "first_inadmissible_phase": "decode.001",
                        "max_abs_error": 1.90625,
                    }
                ),
            ),
        }
    )
    thawed = runner._deep_thaw(frozen)
    json.dumps(thawed, ensure_ascii=True, sort_keys=True)

    # The receipt body must never shallow-thaw a manifest value again: a
    # top-level dict() call leaves every nested row frozen.
    source = (ROOT / "experiments" / "run_r12_on_policy_round.py").read_text(
        encoding="utf-8"
    )
    assert "dict(manifest[" not in source
    assert "dict(admissibility)" not in source
