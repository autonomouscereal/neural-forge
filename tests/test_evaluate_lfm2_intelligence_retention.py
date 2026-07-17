from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from experiments import evaluate_lfm2_intelligence_retention as runner
from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.hf_lfm2_intelligence_gate import bind_manual_ratings


PARENT_CACHE_KEY = hashlib.sha256(b"parent-cache").hexdigest()
PARENT_STATE = hashlib.sha256(b"parent-state").hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tokenizer(path: Path) -> str:
    path.mkdir()
    tokenizer = {
        "added_tokens": [
            {
                "id": 0,
                "content": "<|pad|>",
                "special": True,
            },
            {
                "id": 1,
                "content": "<|startoftext|>",
                "special": True,
            },
            {
                "id": 7,
                "content": "<|im_end|>",
                "special": True,
            },
        ],
        "model": {"type": "BPE", "vocab": {"hello": 100}},
    }
    (path / "tokenizer.json").write_text(
        json.dumps(tokenizer, sort_keys=True),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<|startoftext|>",
                "eos_token": "<|im_end|>",
                "pad_token": "<|pad|>",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return runner.fingerprint_bound_local_tokenizer(path)[0]


def _parent_row(
    case_id: str,
    category: str,
    prompt: str,
    parent_output: str,
    *,
    expected: str | None = None,
    token_seed: int,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "category": category,
        "prompt": prompt,
        "expected_response": expected,
        "parent_output_text": parent_output,
        "parent_output_ids": [token_seed, token_seed + 1, 7],
    }


def _suite_rows() -> list[dict[str, Any]]:
    rows = []
    for index in range(8):
        expected = str(20 + index)
        rows.append(
            _parent_row(
                f"recovery_math_{index:05d}",
                "math",
                f"Question: What is {10 + index} + 10? Answer with only the number.\nAnswer:",
                f" {expected}\nThe answer is {expected}",
                expected=expected,
                token_seed=100 + index * 10,
            )
        )
    operators = ("+", "-", "*", "+", "-", "*", "+", "-")
    names = []
    for index, operator in enumerate(operators):
        stem = {"+": "add", "-": "subtract", "*": "multiply"}[operator]
        name = f"{stem}_{index:03d}"
        names.append(name)
        rows.append(
            _parent_row(
                f"recovery_code_{index:05d}",
                "code",
                f"Write a tiny Python function named {name} that returns the {operator}.\n```python\n",
                (
                    f"def {name}():\n"
                    f"    return {index}\n"
                    "```\nThis parent response has the right topic but wrong behavior."
                ),
                expected=(
                    f"def {name}(a, b):\n"
                    f"    return a {operator} b\n"
                    "```"
                ),
                token_seed=300 + index * 10,
            )
        )
    rows.extend(
        [
            _parent_row(
                "manual_chat_00",
                "conversation",
                "Hi! Briefly introduce yourself and ask how you can help.",
                "I am an AI assistant ready to help. How can I assist you?",
                token_seed=500,
            ),
            _parent_row(
                "manual_fact_00",
                "factual",
                "Why does rain fall from clouds? Explain in two clear sentences.",
                (
                    "Cloud water condenses into droplets. Gravity makes heavy "
                    "droplets fall as rain."
                ),
                token_seed=510,
            ),
            _parent_row(
                "manual_safety_00",
                "commonsense",
                "You smell gas in a house. What should you do first? Give a concise safe answer.",
                "",
                token_seed=520,
            ),
            _parent_row(
                "manual_math_00",
                "math",
                "Calculate 27 times 14. Show the short reasoning, then give the answer.",
                "27 * 14 = 378. Answer: 378",
                token_seed=530,
            ),
            _parent_row(
                "manual_code_00",
                "code",
                (
                    "Write a small Python function is_palindrome(text) that "
                    "ignores spaces and capitalization."
                ),
                (
                    "```python\n"
                    "def is_palindrome(text):\n"
                    "    text = ''.join(c for c in text if c.isalnum()).lower()\n"
                    "    return text == text[::-1]\n"
                    "```"
                ),
                token_seed=540,
            ),
            _parent_row(
                "manual_ai_00",
                "model_surgery",
                (
                    "Explain why matching one next token is not enough to prove "
                    "a compressed language model is intelligent."
                ),
                (
                    "One token does not test whether context remains coherent "
                    "across a sequence of generated tokens."
                ),
                token_seed=550,
            ),
            _parent_row(
                "manual_compare_00",
                "factual",
                (
                    "What is the practical difference between RAM and disk "
                    "storage? Keep it brief."
                ),
                (
                    "RAM is fast temporary memory, while disk storage is "
                    "persistent long-term storage."
                ),
                token_seed=560,
            ),
            _parent_row(
                "manual_instruction_00",
                "instruction",
                (
                    "Reply with exactly one word naming the color of a clear "
                    "daytime sky."
                ),
                "",
                token_seed=570,
            ),
            _parent_row(
                "manual_reason_00",
                "reasoning",
                (
                    "All bloops are razzies. No razzies are green. Can any "
                    "bloop be green? Explain."
                ),
                "",
                token_seed=580,
            ),
            _parent_row(
                "manual_summary_00",
                "summary",
                (
                    "Summarize this in one sentence: A model can preserve local "
                    "activations yet fail after several generated tokens because "
                    "small state errors compound through attention and the "
                    "residual stream."
                ),
                "",
                token_seed=590,
            ),
            _parent_row(
                "manual_creative_00",
                "creative",
                (
                    "Write a coherent three-sentence story about a robot finding "
                    "a lost key."
                ),
                (
                    "A robot searched the city. It found a lost key. The robot "
                    "returned it."
                ),
                token_seed=600,
            ),
            _parent_row(
                "manual_fold_00",
                "model_surgery",
                (
                    "In plain language, describe residual-attention-aware "
                    "neuron folding."
                ),
                (
                    "Residual attention neuron folding compresses weights while "
                    "preserving attention and residual circuit behavior."
                ),
                token_seed=610,
            ),
        ]
    )
    assert len(rows) == 28
    return rows


def _write_envelope(
    path: Path,
    *,
    tokenizer_fingerprint: str,
) -> dict[str, Any]:
    body = {
        "schema": runner.WHOLE_OUTPUT_ENVELOPE_SCHEMA,
        "parent_cache_key": PARENT_CACHE_KEY,
        "parent_state_fingerprint": PARENT_STATE,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "generation": {
            "do_sample": False,
            "max_new_tokens": 64,
            "skip_special_tokens": True,
            "tokenizer_cleanup": "tokenizer_default",
            "use_cache": True,
        },
        "source_rows_sha256": hashlib.sha256(b"source rows").hexdigest(),
        "rows": _suite_rows(),
        "parent_outputs_generated_once": True,
        "parent_forward_passes_after_seal": 0,
    }
    document = {**body, "envelope_sha256": canonical_sha256(body)}
    path.write_bytes(canonical_json_bytes(document) + b"\n")
    return document


def _manual_outputs() -> dict[str, str]:
    return {
        "manual_chat_00": (
            "I am an AI assistant, and I can help answer questions. How can I "
            "assist you?"
        ),
        "manual_fact_00": (
            "Water condenses in a cloud into droplets. Gravity pulls the heavy "
            "droplets down as rain."
        ),
        "manual_safety_00": (
            "Leave the house immediately and call the gas company from outside."
        ),
        "manual_math_00": "27 times 14 is 378, so the answer is 378.",
        "manual_code_00": (
            "```python\n"
            "def is_palindrome(text):\n"
            "    text = ''.join(c for c in text if c.isalnum()).casefold()\n"
            "    return text == text[::-1]\n"
            "```"
        ),
        "manual_ai_00": (
            "Matching one token ignores whether context and a coherent sequence "
            "survive many next-token transitions."
        ),
        "manual_compare_00": (
            "RAM is volatile temporary working memory, whereas disk storage is "
            "persistent long-term storage."
        ),
        "manual_instruction_00": "blue",
        "manual_reason_00": (
            "No, a bloop is a razzy, and no razzy can be green."
        ),
        "manual_summary_00": (
            "Small activation errors compound through attention and the residual "
            "stream, so generation can fail later."
        ),
        "manual_creative_00": (
            "A small robot searched the station for a lost key. It found the key "
            "under a bench. The robot returned it to its owner."
        ),
        "manual_fold_00": (
            "Residual-attention-aware neuron folding compresses or merges neuron "
            "weights while preserving attention outputs and residual circuit "
            "behavior."
        ),
    }


def _candidate_rows(*, bad: bool) -> list[dict[str, Any]]:
    rows = []
    suite = _suite_rows()
    for index, parent in enumerate(suite):
        case_id = parent["id"]
        if case_id.startswith("recovery_math_"):
            output = f"The final answer is {parent['expected_response']}"
        elif case_id.startswith("recovery_code_"):
            if bad:
                output = "code code code code code code code code"
            else:
                output = parent["expected_response"]
        else:
            output = _manual_outputs()[case_id]
        if bad and case_id.startswith("recovery_code_"):
            token_ids = [900 + index] * 64
        else:
            token_ids = [1000 + index * 3, 1001 + index * 3, 7]
        if index < 18:
            topical: bool | None = True
        elif index < 20:
            topical = False
        else:
            topical = None
        rows.append(
            {
                "case_id": case_id,
                "category": parent["category"],
                "output": output,
                "output_token_ids": token_ids,
                "task_relevant": topical,
                "task_correct": None,
            }
        )
    return rows


def _write_candidate(path: Path, *, bad: bool) -> dict[str, Any]:
    event = {
        "phase": "freshly_reloaded_child",
        "decision": {
            "accepted": True,
            "reason": "The old topical gate was permissive.",
            "metrics": {},
        },
        "outputs": _candidate_rows(bad=bad),
        "parent_forward_passes": 0,
    }
    document = {
        "schema": "legacy-whole-output-report",
        "whole_output": {"event": event},
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document


def _config(
    tmp_path: Path,
    *,
    envelope: Path,
    tokenizer: Path,
    tokenizer_fingerprint: str,
    candidate: Path,
    suffix: str,
    ratings: Path | None = None,
    ratings_sha256: str | None = None,
) -> runner.EvaluationConfig:
    return runner.EvaluationConfig(
        parent_envelope=envelope,
        expected_parent_envelope_sha256=_file_sha(envelope),
        expected_parent_cache_key=PARENT_CACHE_KEY,
        expected_parent_state_fingerprint=PARENT_STATE,
        tokenizer_directory=tokenizer,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
        candidate_report=candidate,
        expected_candidate_report_sha256=_file_sha(candidate),
        candidate_phase="freshly_reloaded_child",
        report_path=tmp_path / f"report-{suffix}.json",
        manual_review_packet_path=tmp_path / f"manual-{suffix}.json",
        manual_ratings_path=ratings,
        expected_manual_ratings_sha256=ratings_sha256,
    )


def test_point_nine_topical_relevance_rejects_looping_wrong_code(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer"
    tokenizer_fingerprint = _write_tokenizer(tokenizer)
    envelope = tmp_path / "parent.envelope.json"
    candidate = tmp_path / "candidate.json"
    _write_envelope(envelope, tokenizer_fingerprint=tokenizer_fingerprint)
    _write_candidate(candidate, bad=True)

    report = runner.evaluate_posthoc(
        _config(
            tmp_path,
            envelope=envelope,
            tokenizer=tokenizer,
            tokenizer_fingerprint=tokenizer_fingerprint,
            candidate=candidate,
            suffix="bad",
        )
    )

    gate = report["core_intelligence_gate"]
    components = {item["name"]: item for item in gate["components"]}
    assert gate["topical_relevance_diagnostic"]["rate"] == pytest.approx(0.90)
    assert gate["topical_relevance_diagnostic"]["promotion_eligible"] is False
    assert components["code_syntax"]["passed"] is False
    assert components["eos_generation_limit"]["passed"] is False
    assert components["repetition_degeneration"]["passed"] is False
    assert report["absolute_task_correctness"]["rate"] < 0.90
    assert report["absolute_task_correctness"]["gate_is_parent_relative"] is True
    assert report["absolute_task_correctness"]["minimum_rate"] == pytest.approx(
        report["absolute_task_correctness"]["parent_rate"] * 0.90
    )
    assert report["promoted"] is False
    assert report["intelligence_retention_claim"]["certified_lower_bound"] is None
    # A child is not required to solve tasks the parent itself could not solve.
    # Independent code, termination, repetition, capability, and manual gates
    # still reject this looping candidate.
    assert report["combined_reliable_correctness"]["passed"] is True
    assert report["execution_audit"]["parent_forward_passes"] == 0
    assert report["execution_audit"]["candidate_forward_passes"] == 0
    assert report["execution_audit"]["optimizer_steps"] == 0


def test_parent_code_failures_are_separate_from_absolute_candidate_accuracy(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer"
    tokenizer_fingerprint = _write_tokenizer(tokenizer)
    envelope = tmp_path / "parent.envelope.json"
    candidate = tmp_path / "candidate.json"
    _write_envelope(envelope, tokenizer_fingerprint=tokenizer_fingerprint)
    _write_candidate(candidate, bad=False)
    pending_config = _config(
        tmp_path,
        envelope=envelope,
        tokenizer=tokenizer,
        tokenizer_fingerprint=tokenizer_fingerprint,
        candidate=candidate,
        suffix="pending",
    )

    runner.evaluate_posthoc(pending_config)
    packet = json.loads(
        pending_config.manual_review_packet_path.read_text(encoding="ascii")
    )
    ratings = bind_manual_ratings(
        packet,
        [
            {
                "case_id": value["case_id"],
                "correctness": 4,
                "coherence": 4,
                "instruction_following": 4,
                "capability_retention": 4,
            }
            for value in packet["cases"]
        ],
        reviewer_id="human-reviewer",
    )
    ratings_path = tmp_path / "ratings.json"
    ratings_path.write_bytes(canonical_json_bytes(ratings.as_dict()) + b"\n")
    final_config = _config(
        tmp_path,
        envelope=envelope,
        tokenizer=tokenizer,
        tokenizer_fingerprint=tokenizer_fingerprint,
        candidate=candidate,
        suffix="final",
        ratings=ratings_path,
        ratings_sha256=ratings.ratings_sha256,
    )

    final = runner.evaluate_posthoc(final_config)

    parent_retention = final["parent_demonstrated_task_retention"]
    parent_code_ids = {
        f"recovery_code_{index:05d}" for index in range(8)
    }
    assert parent_code_ids.issubset(
        set(parent_retention["parent_incorrect_case_ids"])
    )
    assert final["absolute_task_correctness"]["rate"] == pytest.approx(1.0)
    assert parent_retention["rate"] == pytest.approx(1.0)
    assert final["parent_semantic_capability_retention"]["retention"] == pytest.approx(
        1.0
    )
    assert final["combined_reliable_correctness"]["per_case"][
        "manual_code_00"
    ]["candidate_detail"]["safe_ast_interpreter"] is True
    assert final["manual_review"]["status"] == "accepted"
    assert final["promoted"] is True
    assert final["intelligence_retention_claim"]["status"] == (
        "certified_conservative_lower_bound"
    )
    assert final["intelligence_retention_claim"]["certified_lower_bound"] == (
        pytest.approx(1.0)
    )


def test_identity_tampering_and_ambiguous_candidate_events_fail_closed(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer"
    tokenizer_fingerprint = _write_tokenizer(tokenizer)
    envelope = tmp_path / "parent.envelope.json"
    candidate = tmp_path / "candidate.json"
    _write_envelope(envelope, tokenizer_fingerprint=tokenizer_fingerprint)
    candidate_document = _write_candidate(candidate, bad=False)
    config = _config(
        tmp_path,
        envelope=envelope,
        tokenizer=tokenizer,
        tokenizer_fingerprint=tokenizer_fingerprint,
        candidate=candidate,
        suffix="identity",
    )

    changed = copy.deepcopy(candidate_document)
    changed["whole_output"]["second"] = copy.deepcopy(
        changed["whole_output"]["event"]
    )
    candidate.write_text(json.dumps(changed), encoding="utf-8")
    ambiguous = replace(
        config,
        expected_candidate_report_sha256=_file_sha(candidate),
        candidate_phase="freshly_reloaded_child",
    )
    with pytest.raises(
        runner.Lfm2PosthocEvaluationError,
        match="exactly one whole-output event",
    ):
        runner.evaluate_posthoc(ambiguous)

    with pytest.raises(
        runner.Lfm2PosthocEvaluationError,
        match="fingerprint differs",
    ):
        runner.load_sealed_parent_envelope(
            envelope,
            expected_sha256="0" * 64,
            expected_parent_cache_key=PARENT_CACHE_KEY,
            expected_parent_state_fingerprint=PARENT_STATE,
            expected_tokenizer_fingerprint=tokenizer_fingerprint,
        )


def test_runner_source_has_no_model_execution_training_or_network_loader() -> None:
    source = inspect.getsource(runner)

    assert ".generate(" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "from_pretrained" not in source
    assert "AutoTokenizer" not in source
    assert "exec(" not in source
