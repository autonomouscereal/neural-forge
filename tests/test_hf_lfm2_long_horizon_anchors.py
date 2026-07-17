from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from typing import Any

import pytest

from neural_forge.hf_lfm2_long_horizon_anchors import (
    ANCHOR_POLICY_IDENTITY,
    ANCHOR_QUOTAS,
    Lfm2LongHorizonAnchorError,
    build_lfm2_long_horizon_anchor_plan,
    seal_lfm2_long_horizon_case_envelope,
    tag_lfm2_long_horizon_rows,
)


TOKEN_COUNT = 64


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _token_case(
    *,
    case_id: str,
    domain: str,
    ordinal: int,
    correct_arithmetic: bool | None,
    sparse_code_syntax: bool = False,
) -> dict[str, Any]:
    texts = ["x"] * TOKEN_COUNT
    texts[0] = "Return"
    texts[1] = " "
    ids = [10_000 + ordinal * TOKEN_COUNT + index for index in range(TOKEN_COUNT)]
    if domain == "math":
        texts[2] = "4"
        texts[3] = "2"
        if ordinal == 0:
            texts[10] = "π"
    else:
        if not sparse_code_syntax or ordinal == 8:
            texts[3:6] = ["`", "`", "`"]
            texts[6] = "+"
        if ordinal == 8:
            texts[10] = "§"
    texts[-1] = "<eos>"
    ids[-1] = 2
    row: dict[str, Any] = {
        "case_id": case_id,
        "source_split": "heldout_recovery",
        "domain": domain,
        "parent_output_ids": ids,
        "parent_output_token_texts": texts,
        "parent_output_text": "".join(texts),
        "termination_token_indices": [TOKEN_COUNT - 1],
        "termination_confirmed": True,
    }
    if domain == "math":
        row["arithmetic_answer_correct"] = correct_arithmetic
        if correct_arithmetic is True:
            row["arithmetic_answer_token_indices"] = [2, 3]
    return row


def _evidence(
    *,
    all_arithmetic_mismatches: bool = False,
    sparse_code_syntax: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases = []
    for index in range(8):
        correct: bool | None
        if all_arithmetic_mismatches:
            correct = False
        else:
            correct = True if index == 0 else False
        cases.append(
            _token_case(
                case_id=f"recovery_math_{index:02d}",
                domain="math",
                ordinal=index,
                correct_arithmetic=correct,
                sparse_code_syntax=sparse_code_syntax,
            )
        )
    for index in range(8):
        cases.append(
            _token_case(
                case_id=f"recovery_code_{index:02d}",
                domain="code",
                ordinal=index + 8,
                correct_arithmetic=None,
                sparse_code_syntax=sparse_code_syntax,
            )
        )
    cases.extend(
        [
            {
                "case_id": "manual_chat_00",
                "source_split": "heldout_manual",
                "domain": "manual",
                "parent_output_ids": [],
                "parent_output_token_texts": [],
                "parent_output_text": "",
            },
            {
                "case_id": "manual_reason_00",
                "source_split": "heldout_manual",
                "domain": "manual",
                "parent_output_ids": [900_001],
                "parent_output_token_texts": ["Manual output is never fold evidence."],
                "parent_output_text": "Manual output is never fold evidence.",
            },
        ]
    )
    envelope = seal_lfm2_long_horizon_case_envelope(
        cases,
        source_identity_sha256=_digest("sealed-parent-output-envelope"),
        tokenizer_fingerprint=_digest("local-lfm2-tokenizer"),
        eos_token_ids=[2],
    )
    descriptors: list[dict[str, Any]] = []
    for batch_index, case in enumerate(cases[:16]):
        token_ids = case["parent_output_ids"]
        descriptors.extend(
            [
                {
                    "path": (
                        f"batch.{batch_index:06d}/prefill/row.0000"
                    ),
                    "case_id": case["case_id"],
                    "phase": "prefill",
                    "row_index": 0,
                    "predicted_output_index": None,
                    "predicted_token_id": None,
                },
                {
                    "path": (
                        f"batch.{batch_index:06d}/prefill/row.0001"
                    ),
                    "case_id": case["case_id"],
                    "phase": "prefill",
                    "row_index": 1,
                    "predicted_output_index": 0,
                    "predicted_token_id": token_ids[0],
                },
            ]
        )
        for output_index in range(1, len(token_ids)):
            step = output_index - 1
            descriptors.append(
                {
                    "path": (
                        f"batch.{batch_index:06d}/decode.{step:03d}/row.0000"
                    ),
                    "case_id": case["case_id"],
                    "phase": f"decode.{step:03d}",
                    "row_index": 0,
                    "predicted_output_index": output_index,
                    "predicted_token_id": token_ids[output_index],
                }
            )
    return envelope, descriptors


def _row_map(tagging: Any) -> dict[tuple[str, int], Any]:
    return {
        (row.case_id, row.predicted_output_index): row
        for row in tagging.rows
    }


def _quota_map(plan: Any) -> dict[str, Any]:
    return {
        fulfillment.tag: fulfillment
        for fulfillment in plan.quota_fulfillment
    }


def _case_domains(envelope: Mapping[str, Any]) -> dict[str, str]:
    return {
        row["case_id"]: row["domain"]
        for row in envelope["cases"]
    }


def test_final_prefill_and_decode_alignment_tag_exact_eos_predecessor() -> None:
    envelope, descriptors = _evidence()

    tagging = tag_lfm2_long_horizon_rows(envelope, descriptors)
    rows = _row_map(tagging)

    first = rows[("recovery_math_00", 0)]
    assert first.phase == "prefill"
    assert first.path.endswith("/prefill/row.0001")
    eos = rows[("recovery_math_00", TOKEN_COUNT - 1)]
    assert eos.phase == f"decode.{TOKEN_COUNT - 2:03d}"
    assert "termination" in eos.tags


def test_multi_token_code_fence_tags_every_predicted_token_predecessor() -> None:
    envelope, descriptors = _evidence()

    rows = _row_map(tag_lfm2_long_horizon_rows(envelope, descriptors))

    for output_index in (3, 4, 5):
        row = rows[("recovery_code_00", output_index)]
        assert "code_delimiter" in row.tags
        assert row.phase == f"decode.{output_index - 1:03d}"
    assert "code_operator" in rows[("recovery_code_00", 6)].tags


def test_correct_arithmetic_span_is_tagged_and_known_mismatch_is_excluded() -> None:
    envelope, descriptors = _evidence()

    tagging = tag_lfm2_long_horizon_rows(envelope, descriptors)
    rows = _row_map(tagging)

    assert "arithmetic_answer" in rows[("recovery_math_00", 2)].tags
    assert "arithmetic_answer" in rows[("recovery_math_00", 3)].tags
    assert all(
        "arithmetic_answer" not in rows[("recovery_math_01", index)].tags
        for index in range(TOKEN_COUNT)
    )
    assert {
        (exclusion.case_id, exclusion.reason)
        for exclusion in tagging.exclusions
    } >= {
        ("recovery_math_01", "known-arithmetic-mismatch"),
    }


def test_instruction_start_and_unique_rare_symbols_are_tagged() -> None:
    envelope, descriptors = _evidence()

    rows = _row_map(tag_lfm2_long_horizon_rows(envelope, descriptors))

    instruction = rows[("recovery_math_00", 0)]
    assert "instruction" in instruction.tags
    assert instruction.phase == "prefill"
    assert "rare_symbol" in rows[("recovery_math_00", 10)].tags
    assert "rare_symbol" in rows[("recovery_code_00", 10)].tags


def test_exact_quotas_use_integer_multiplicity_and_spill_to_general() -> None:
    envelope, descriptors = _evidence(
        all_arithmetic_mismatches=True,
        sparse_code_syntax=True,
    )

    plan = build_lfm2_long_horizon_anchor_plan(envelope, descriptors)

    for split_name in ("selection", "fit", "validation"):
        split = getattr(plan, split_name)
        expected = sum(ANCHOR_QUOTAS[split_name].values())
        assert split.total_weight == expected
        assert len(split.expanded_indices) == expected
        assert sum(split.integer_weights) == expected
        assert len(split.indices) == len(split.integer_weights)
        assert all(type(weight) is int and weight > 0 for weight in split.integer_weights)
        quotas = _quota_map(split)
        arithmetic = quotas["arithmetic_answer"]
        assert arithmetic.allocated_occurrences == 0
        assert (
            arithmetic.spilled_to_general
            == ANCHOR_QUOTAS[split_name]["arithmetic_answer"]
        )
        general = quotas["general_coverage"]
        assert general.spill_received >= arithmetic.spilled_to_general
    assert plan.selection.repeated_occurrence_count > 0
    assert plan.fit.repeated_occurrence_count > 0
    assert plan.validation.repeated_occurrence_count == 0
    assert set(plan.validation.integer_weights) == {1}


def test_case_sha_split_is_domain_balanced_disjoint_and_excludes_manual() -> None:
    envelope, descriptors = _evidence()

    tagging = tag_lfm2_long_horizon_rows(envelope, descriptors)
    plan = build_lfm2_long_horizon_anchor_plan(envelope, descriptors)
    domains = _case_domains(envelope)
    case_sets = {
        split: set(getattr(plan, split).case_ids)
        for split in ("selection", "fit", "validation")
    }

    assert len(case_sets["selection"]) == 6
    assert len(case_sets["fit"]) == 8
    assert len(case_sets["validation"]) == 2
    assert not case_sets["selection"] & case_sets["fit"]
    assert not case_sets["selection"] & case_sets["validation"]
    assert not case_sets["fit"] & case_sets["validation"]
    for split, expected_per_domain in (
        ("selection", 3),
        ("fit", 4),
        ("validation", 1),
    ):
        assert sum(domains[case_id] == "math" for case_id in case_sets[split]) == (
            expected_per_domain
        )
        assert sum(domains[case_id] == "code" for case_id in case_sets[split]) == (
            expected_per_domain
        )
    assert tagging.excluded_case_ids == ("manual_chat_00", "manual_reason_00")
    assert not any(row.case_id.startswith("manual_") for row in tagging.rows)


def test_plan_is_byte_deterministic_and_binds_policy_and_paths() -> None:
    envelope, descriptors = _evidence()

    first = build_lfm2_long_horizon_anchor_plan(envelope, descriptors)
    second = build_lfm2_long_horizon_anchor_plan(
        copy.deepcopy(envelope),
        copy.deepcopy(descriptors),
        expected_envelope_sha256=envelope["envelope_sha256"],
        expected_ordered_path_sha256=first.ordered_path_sha256,
        expected_policy_identity=ANCHOR_POLICY_IDENTITY,
    )

    assert first == second
    assert first.as_dict() == second.as_dict()
    assert first.policy_identity == ANCHOR_POLICY_IDENTITY
    assert len(first.composite_evidence_identity) == 64
    assert len(first.plan_sha256) == 64


def test_identity_or_token_path_tamper_fails_closed() -> None:
    envelope, descriptors = _evidence()
    plan = build_lfm2_long_horizon_anchor_plan(envelope, descriptors)

    bad_envelope = copy.deepcopy(envelope)
    bad_envelope["cases"][0]["parent_output_ids"][0] += 1
    with pytest.raises(Lfm2LongHorizonAnchorError, match="identity"):
        build_lfm2_long_horizon_anchor_plan(bad_envelope, descriptors)

    bad_paths = copy.deepcopy(descriptors)
    mapped = next(
        row for row in bad_paths if row["predicted_output_index"] == 1
    )
    mapped["predicted_token_id"] += 1
    with pytest.raises(Lfm2LongHorizonAnchorError, match="token ID"):
        build_lfm2_long_horizon_anchor_plan(envelope, bad_paths)

    with pytest.raises(Lfm2LongHorizonAnchorError, match="expected identity"):
        build_lfm2_long_horizon_anchor_plan(
            envelope,
            descriptors,
            expected_ordered_path_sha256="0" * 64,
        )
    assert plan.ordered_path_sha256 != "0" * 64
