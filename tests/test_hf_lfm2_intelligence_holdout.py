from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments import evaluate_lfm2_intelligence_holdout as evaluator
from experiments import evaluate_lfm2_intelligence_retention as posthoc
from neural_forge import hf_lfm2_intelligence_gate as gate
from neural_forge.dynamic_circuit_ir import canonical_json_bytes
from neural_forge.hf_lfm2_intelligence_gate import (
    HOLDOUT_CASE_COUNT,
    bind_manual_ratings,
    load_lfm2_intelligence_holdout_manifest,
)
from neural_forge.hf_lfm2_intelligence_holdout import (
    Lfm2IntelligenceHoldoutError,
    candidate_outputs_from_lfm2_holdout_rows,
    load_lfm2_intelligence_holdout_candidate_outputs,
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_candidate_outputs,
    seal_lfm2_intelligence_holdout_parent_envelope,
)


PARENT_STATE = hashlib.sha256(b"holdout-parent-state").hexdigest()
PARENT_MODEL = hashlib.sha256(b"holdout-parent-model").hexdigest()
CANDIDATE_STATE = hashlib.sha256(b"holdout-candidate-state").hexdigest()
EOS_ID = 7


_CODE_BODIES = {
    "holdout_v1_python_00": (
        "def holdout_is_even(n):\n"
        "    return n % 2 == 0"
    ),
    "holdout_v1_python_01": (
        "def holdout_clamp(x, low, high):\n"
        "    return low if x < low else high if x > high else x"
    ),
    "holdout_v1_python_02": (
        "def holdout_last(items):\n"
        "    return items[-1]"
    ),
    "holdout_v1_python_03": (
        "def holdout_reverse(text):\n"
        "    return text[::-1]"
    ),
    "holdout_v1_python_04": (
        "def holdout_between(x, low, high):\n"
        "    return low <= x and x <= high"
    ),
    "holdout_v1_python_05": (
        "def holdout_choose(flag, yes, no):\n"
        "    return yes if flag else no"
    ),
    "holdout_v1_python_06": (
        "def holdout_difference(a, b):\n"
        "    return a - b"
    ),
    "holdout_v1_python_07": (
        "def holdout_join(left, right):\n"
        "    return left + right"
    ),
}


def _word_count(value: str) -> int:
    return len(gate._words(value))


def _constraint_output(spec: Any, *, candidate: bool) -> str:
    validator = spec.validator
    assert validator is not None
    if validator.exact_word is not None:
        return validator.exact_word
    terms = [
        group[-1] if candidate and len(group) > 1 else group[0]
        for group in validator.required_groups
    ]
    filler = (
        "This alternate concise response clearly explains the requested "
        "result safely"
        if candidate
        else "This concise answer clearly explains the requested result safely"
    ).split()
    output = " ".join(terms)
    index = 0
    while _word_count(output) < validator.minimum_words:
        output = f"{output} {filler[index % len(filler)]}".strip()
        index += 1
    assert (
        validator.maximum_words is None
        or _word_count(output) <= validator.maximum_words
    )
    return output


def _manual_output(spec: Any, *, candidate: bool) -> str:
    assert spec.capability is not None
    terms = [
        group[-1] if candidate and len(group) > 1 else group[0]
        for group in spec.capability.required_concept_groups
    ]
    filler = (
        "A different coherent explanation gives practical context and a useful "
        "answer"
        if candidate
        else "A clear coherent explanation gives practical context and a useful "
        "answer"
    ).split()
    output = " ".join(terms)
    index = 0
    while _word_count(output) < spec.capability.minimum_words:
        output = f"{output} {filler[index % len(filler)]}".strip()
        index += 1
    return output


def _passing_output(spec: Any, *, candidate: bool) -> str:
    validator = spec.validator
    if validator is None:
        return _manual_output(spec, candidate=candidate)
    if validator.kind == "arithmetic_exact":
        prefix = (
            "A separate calculation gives"
            if candidate
            else "The checked calculation gives"
        )
        return f"{prefix} {validator.expected_numeric}"
    if validator.kind == "python_pure_function":
        source = _CODE_BODIES[spec.case_id]
        return f"```python\n{source}\n```" if candidate else source
    return _constraint_output(spec, candidate=candidate)


def _generated_parent_rows(manifest: Any) -> list[dict[str, Any]]:
    return [
        {
            "case_id": spec.case_id,
            "input_token_count": 20 + index,
            "output": _passing_output(spec, candidate=False),
            "output_token_ids": [
                1000 + index * 4,
                1001 + index * 4,
                1002 + index * 4,
                EOS_ID,
            ],
        }
        for index, spec in enumerate(manifest.cases)
    ]


def _candidate_rows(manifest: Any) -> list[dict[str, Any]]:
    return [
        {
            "case_id": spec.case_id,
            "output": _passing_output(spec, candidate=True),
            "output_token_ids": [
                5000 + index * 4,
                5001 + index * 4,
                5002 + index * 4,
                EOS_ID,
            ],
            "generation_limit": spec.generation_limit,
            "topical_relevance": True,
        }
        for index, spec in enumerate(manifest.cases)
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tokenizer(path: Path) -> str:
    path.mkdir()
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 0, "content": "<pad>", "special": True},
                    {"id": EOS_ID, "content": "<eos>", "special": True},
                ],
                "model": {"type": "BPE", "vocab": {"hello": 10}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {"pad_token": "<pad>", "eos_token": "<eos>"},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return posthoc.fingerprint_bound_local_tokenizer(path)[0]


def _sealed_fixture(tmp_path: Path) -> dict[str, Any]:
    manifest = load_lfm2_intelligence_holdout_manifest()
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_fingerprint = _write_tokenizer(tokenizer_directory)
    parent = seal_lfm2_intelligence_holdout_parent_envelope(
        manifest,
        _generated_parent_rows(manifest),
        parent_state_fingerprint=PARENT_STATE,
        parent_model_fingerprint=PARENT_MODEL,
        tokenizer_fingerprint=tokenizer_fingerprint,
        eos_token_ids=[EOS_ID],
    )
    candidate = seal_lfm2_intelligence_holdout_candidate_outputs(
        manifest,
        _candidate_rows(manifest),
        candidate_state_fingerprint=CANDIDATE_STATE,
        tokenizer_fingerprint=tokenizer_fingerprint,
        eos_token_ids=[EOS_ID],
    )
    parent_path = tmp_path / "parent.json"
    candidate_path = tmp_path / "candidate.json"
    _write_json(parent_path, parent)
    _write_json(candidate_path, candidate)
    return {
        "manifest": manifest,
        "tokenizer_directory": tokenizer_directory,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "parent": parent,
        "parent_path": parent_path,
        "candidate": candidate,
        "candidate_path": candidate_path,
    }


def _config(
    fixture: dict[str, Any],
    tmp_path: Path,
    *,
    suffix: str,
    ratings_path: Path | None = None,
    ratings_file_sha: str | None = None,
    ratings_sha: str | None = None,
) -> evaluator.HoldoutEvaluationConfig:
    return evaluator.HoldoutEvaluationConfig(
        parent_envelope=fixture["parent_path"],
        expected_parent_envelope_file_sha256=_file_sha(
            fixture["parent_path"]
        ),
        expected_parent_envelope_sha256=fixture["parent"][
            "envelope_sha256"
        ],
        expected_manifest_sha256=fixture["manifest"].manifest_sha256,
        expected_parent_state_fingerprint=PARENT_STATE,
        expected_parent_model_fingerprint=PARENT_MODEL,
        tokenizer_directory=fixture["tokenizer_directory"],
        expected_tokenizer_fingerprint=fixture["tokenizer_fingerprint"],
        candidate_report=fixture["candidate_path"],
        expected_candidate_report_sha256=_file_sha(
            fixture["candidate_path"]
        ),
        expected_candidate_outputs_sha256=fixture["candidate"][
            "candidate_outputs_sha256"
        ],
        expected_candidate_state_fingerprint=CANDIDATE_STATE,
        report_path=tmp_path / f"report-{suffix}.json",
        manual_review_packet_path=tmp_path / f"packet-{suffix}.json",
        manual_ratings_path=ratings_path,
        expected_manual_ratings_file_sha256=ratings_file_sha,
        expected_manual_ratings_sha256=ratings_sha,
    )


def test_manifest_is_exact_disjoint_and_covers_long_horizon_behaviors() -> None:
    manifest = load_lfm2_intelligence_holdout_manifest()
    cases = manifest.cases
    counts = {
        category: sum(item.category == category for item in cases)
        for category in {item.category for item in cases}
    }

    assert len(cases) == HOLDOUT_CASE_COUNT == 64
    assert counts == {
        "arithmetic": 10,
        "python": 8,
        "factual_common_sense": 8,
        "instruction": 8,
        "reasoning": 6,
        "summary": 4,
        "safety": 4,
        "open_ended_manual": 6,
        "termination_repetition": 4,
        "longer_prefix": 6,
    }
    assert sum(item.manual_review_required for item in cases) == 6
    assert sum(item.longer_prefix for item in cases) == 6
    assert not {item.case_id for item in cases}.intersection(
        gate._RESERVED_CALIBRATION_CASE_IDS
    )
    assert not {
        gate._prompt_sha256(item.prompt) for item in cases
    }.intersection(
        gate._prompt_sha256(value)
        for value in gate._RESERVED_CALIBRATION_PROMPTS
    )


def test_semantic_validators_accept_different_wording_and_reject_unsafe_code() -> None:
    manifest = load_lfm2_intelligence_holdout_manifest()
    factual = manifest.case_map["holdout_v1_factual_00"]
    parent = _passing_output(factual, candidate=False)
    candidate = _passing_output(factual, candidate=True)
    code = manifest.case_map["holdout_v1_python_00"]

    assert parent != candidate
    assert factual.validator is not None
    assert gate._validator_result(parent, factual.validator)[0]
    assert gate._validator_result(candidate, factual.validator)[0]
    assert code.validator is not None
    assert gate._validator_result(
        _passing_output(code, candidate=True),
        code.validator,
    )[0]
    assert not gate._validator_result(
        (
            "def holdout_is_even(n):\n"
            "    return __import__('os').system('never-run')"
        ),
        code.validator,
    )[0]


def test_parent_and_candidate_seals_round_trip_and_reject_tampering(
    tmp_path: Path,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    parent = load_lfm2_intelligence_holdout_parent_envelope(
        fixture["parent"],
        expected_envelope_sha256=fixture["parent"]["envelope_sha256"],
        expected_manifest_sha256=fixture["manifest"].manifest_sha256,
        expected_parent_state_fingerprint=PARENT_STATE,
        expected_parent_model_fingerprint=PARENT_MODEL,
        expected_tokenizer_fingerprint=fixture["tokenizer_fingerprint"],
    )
    candidate = load_lfm2_intelligence_holdout_candidate_outputs(
        fixture["candidate"],
        expected_candidate_outputs_sha256=fixture["candidate"][
            "candidate_outputs_sha256"
        ],
        expected_manifest_sha256=fixture["manifest"].manifest_sha256,
        expected_candidate_state_fingerprint=CANDIDATE_STATE,
        expected_tokenizer_fingerprint=fixture["tokenizer_fingerprint"],
        eos_token_ids=[EOS_ID],
    )

    assert len(parent.rows) == len(candidate.candidates) == 64
    changed_parent = copy.deepcopy(fixture["parent"])
    changed_parent["rows"][0]["parent_output"] = "tampered"
    with pytest.raises(Lfm2IntelligenceHoldoutError, match="seal"):
        load_lfm2_intelligence_holdout_parent_envelope(
            changed_parent,
            expected_envelope_sha256=fixture["parent"][
                "envelope_sha256"
            ],
            expected_manifest_sha256=fixture["manifest"].manifest_sha256,
            expected_parent_state_fingerprint=PARENT_STATE,
            expected_parent_model_fingerprint=PARENT_MODEL,
            expected_tokenizer_fingerprint=fixture[
                "tokenizer_fingerprint"
            ],
        )
    changed_candidate = copy.deepcopy(fixture["candidate"])
    changed_candidate["rows"][0]["output_token_ids"][0] += 1
    with pytest.raises(Lfm2IntelligenceHoldoutError, match="seal"):
        load_lfm2_intelligence_holdout_candidate_outputs(
            changed_candidate,
            expected_candidate_outputs_sha256=fixture["candidate"][
                "candidate_outputs_sha256"
            ],
            expected_manifest_sha256=fixture["manifest"].manifest_sha256,
            expected_candidate_state_fingerprint=CANDIDATE_STATE,
            expected_tokenizer_fingerprint=fixture[
                "tokenizer_fingerprint"
            ],
            eos_token_ids=[EOS_ID],
        )


def test_holdout_evaluator_withholds_then_certifies_packet_bound_review(
    tmp_path: Path,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    pending_config = _config(fixture, tmp_path, suffix="pending")
    pending = evaluator.evaluate_lfm2_intelligence_holdout(pending_config)

    assert pending["promoted"] is False
    assert pending["manual_review"]["status"] == "pending"
    assert pending["intelligence_retention_claim"]["status"] == (
        "withheld_pending_manual_review"
    )
    assert pending["diagnostic_lower_bound"] is None
    assert len(pending["manual_review"]["packet_case_ids"]) == 6
    assert pending["execution_audit"] == {
        "parent_model_loaded": False,
        "candidate_model_loaded": False,
        "parent_forward_passes": 0,
        "parent_generate_invocations": 0,
        "candidate_forward_passes": 0,
        "candidate_generate_invocations": 0,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "gradients_used": False,
        "network_access": False,
        "hub_loader_used": False,
        "candidate_python_exec_used": False,
        "cached_outputs_only": True,
    }

    packet = json.loads(
        pending_config.manual_review_packet_path.read_text(encoding="utf-8")
    )
    ratings = bind_manual_ratings(
        packet,
        [
            {
                "case_id": row["case_id"],
                "correctness": 4,
                "coherence": 4,
                "instruction_following": 4,
                "capability_retention": 4,
            }
            for row in packet["cases"]
        ],
        reviewer_id="independent-reviewer",
    )
    ratings_path = tmp_path / "ratings.json"
    _write_json(ratings_path, ratings.as_dict())
    accepted_config = _config(
        fixture,
        tmp_path,
        suffix="accepted",
        ratings_path=ratings_path,
        ratings_file_sha=_file_sha(ratings_path),
        ratings_sha=ratings.ratings_sha256,
    )
    accepted = evaluator.evaluate_lfm2_intelligence_holdout(
        accepted_config
    )

    assert accepted["promoted"] is True
    assert accepted["manual_review"]["status"] == "accepted"
    assert accepted["intelligence_retention_claim"][
        "certified_lower_bound"
    ] == pytest.approx(1.0)
    assert accepted["parent_relative_task_retention"]["rate"] == pytest.approx(
        1.0
    )
    assert accepted["parent_demonstrated_capability_retention"][
        "rate"
    ] == pytest.approx(1.0)
    assert accepted["policy"]["all_required_component_floor_is_0_90"] is True
    assert accepted["semantic_evaluation"]["exact_parent_wording_used"] is False


def test_raw_candidate_report_supports_all_manifest_cases_and_requires_limits(
    tmp_path: Path,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    rows = _candidate_rows(fixture["manifest"])
    raw_path = tmp_path / "raw-candidate.json"
    _write_json(
        raw_path,
        {"phase": "reloaded_child", "outputs": rows},
    )
    config = _config(fixture, tmp_path, suffix="unused")
    raw_config = evaluator.HoldoutEvaluationConfig(
        parent_envelope=config.parent_envelope,
        expected_parent_envelope_file_sha256=(
            config.expected_parent_envelope_file_sha256
        ),
        expected_parent_envelope_sha256=(
            config.expected_parent_envelope_sha256
        ),
        expected_manifest_sha256=config.expected_manifest_sha256,
        expected_parent_state_fingerprint=(
            config.expected_parent_state_fingerprint
        ),
        expected_parent_model_fingerprint=(
            config.expected_parent_model_fingerprint
        ),
        tokenizer_directory=config.tokenizer_directory,
        expected_tokenizer_fingerprint=(
            config.expected_tokenizer_fingerprint
        ),
        candidate_report=raw_path,
        expected_candidate_report_sha256=_file_sha(raw_path),
        candidate_phase="reloaded_child",
        report_path=tmp_path / "raw-report.json",
        manual_review_packet_path=tmp_path / "raw-packet.json",
    )
    report = evaluator.evaluate_lfm2_intelligence_holdout(raw_config)

    assert report["inputs"]["case_count"] == 64
    assert report["inputs"]["source_kind"] == "nested_candidate_report"
    assert report["execution_audit"]["parent_generate_invocations"] == 0

    missing_limit = copy.deepcopy(rows)
    missing_limit[0].pop("generation_limit")
    bad_path = tmp_path / "raw-bad.json"
    _write_json(
        bad_path,
        {"phase": "reloaded_child", "outputs": missing_limit},
    )
    bad_config = evaluator.HoldoutEvaluationConfig(
        parent_envelope=config.parent_envelope,
        expected_parent_envelope_file_sha256=(
            config.expected_parent_envelope_file_sha256
        ),
        expected_parent_envelope_sha256=(
            config.expected_parent_envelope_sha256
        ),
        expected_manifest_sha256=config.expected_manifest_sha256,
        expected_parent_state_fingerprint=(
            config.expected_parent_state_fingerprint
        ),
        expected_parent_model_fingerprint=(
            config.expected_parent_model_fingerprint
        ),
        tokenizer_directory=config.tokenizer_directory,
        expected_tokenizer_fingerprint=(
            config.expected_tokenizer_fingerprint
        ),
        candidate_report=bad_path,
        expected_candidate_report_sha256=_file_sha(bad_path),
        candidate_phase="reloaded_child",
        report_path=tmp_path / "bad-report.json",
        manual_review_packet_path=tmp_path / "bad-packet.json",
    )
    with pytest.raises(
        evaluator.Lfm2HoldoutEvaluationError,
        match="generation limit",
    ):
        evaluator.evaluate_lfm2_intelligence_holdout(bad_config)


def test_candidate_typing_rejects_early_eos_and_output_overwrite(
    tmp_path: Path,
) -> None:
    fixture = _sealed_fixture(tmp_path)
    rows = _candidate_rows(fixture["manifest"])
    rows[0]["output_token_ids"] = [EOS_ID, 999]
    with pytest.raises(Lfm2IntelligenceHoldoutError, match="EOS"):
        candidate_outputs_from_lfm2_holdout_rows(
            fixture["manifest"],
            rows,
            eos_token_ids=[EOS_ID],
        )

    existing = tmp_path / "existing.json"
    existing.write_text("owned", encoding="utf-8")
    base = _config(fixture, tmp_path, suffix="other")
    with pytest.raises(
        evaluator.Lfm2HoldoutEvaluationError,
        match="replace existing",
    ):
        evaluator.HoldoutEvaluationConfig(
            parent_envelope=base.parent_envelope,
            expected_parent_envelope_file_sha256=(
                base.expected_parent_envelope_file_sha256
            ),
            expected_parent_envelope_sha256=(
                base.expected_parent_envelope_sha256
            ),
            expected_manifest_sha256=base.expected_manifest_sha256,
            expected_parent_state_fingerprint=(
                base.expected_parent_state_fingerprint
            ),
            expected_parent_model_fingerprint=(
                base.expected_parent_model_fingerprint
            ),
            tokenizer_directory=base.tokenizer_directory,
            expected_tokenizer_fingerprint=(
                base.expected_tokenizer_fingerprint
            ),
            candidate_report=base.candidate_report,
            expected_candidate_report_sha256=(
                base.expected_candidate_report_sha256
            ),
            report_path=existing,
            manual_review_packet_path=tmp_path / "new-packet.json",
        )
