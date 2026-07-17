from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments import run_lfm2_long_horizon_repair as runner
from neural_forge.hf_lfm2_long_horizon_anchors import (
    build_lfm2_long_horizon_anchor_plan,
    tag_lfm2_long_horizon_rows,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recovery_row(
    *,
    domain: str,
    index: int,
    enriched: bool,
) -> dict[str, Any]:
    case_id = f"recovery_{domain}_{index:05d}"
    token_ids = [10_000 + index * 100 + offset for offset in range(64)]
    texts = ["x"] * 64
    texts[0] = "Return"
    texts[1] = " "
    if domain == "math":
        texts[2:4] = ["4", "2"]
    else:
        texts[3:7] = ["`", "`", "`", "+"]
    texts[10] = "π" if domain == "math" else "§"
    texts[-1] = "<eos>"
    token_ids[-1] = 2
    row: dict[str, Any] = {
        "id": case_id,
        "category": domain,
        "prompt": f"Return the {domain} answer for case {index}.",
        "parent_output_ids": token_ids,
        "forced_decode_ids": list(token_ids),
        "parent_output_text": "".join(texts[:-1]),
        "expected_response": "42" if domain == "math" else None,
    }
    if enriched:
        row.update(
            {
                "input_ids": [101, 200 + index, 102],
                "attention_mask": [1, 1, 1],
                "parent_output_token_texts": texts,
                "termination_token_indices": [63],
                "termination_confirmed": True,
            }
        )
        if domain == "math":
            row["arithmetic_answer_correct"] = False
    return row


def _manual_row(index: int) -> dict[str, Any]:
    return {
        "id": f"manual_case_{index:02d}",
        "category": "manual",
        "prompt": f"Manual held-out prompt {index}.",
        "parent_output_ids": [500 + index, 2],
        "forced_decode_ids": [500 + index, 2],
        "parent_output_text": f"Manual response {index}.",
        "expected_response": None,
    }


def _write_envelope(path: Path, *, enriched: bool) -> dict[str, Any]:
    rows = [
        *(
            _recovery_row(domain="math", index=index, enriched=enriched)
            for index in range(8)
        ),
        *(
            _recovery_row(domain="code", index=index + 8, enriched=enriched)
            for index in range(8)
        ),
        *(_manual_row(index) for index in range(12)),
    ]
    body = {
        "schema": runner.WHOLE_OUTPUT_ENVELOPE_SCHEMA,
        "parent_cache_key": _digest("original-parent-cache"),
        "parent_state_fingerprint": _digest("parent-state"),
        "tokenizer_fingerprint": _digest("tokenizer"),
        "generation": {
            "do_sample": False,
            "use_cache": True,
            "max_new_tokens": 64,
        },
        "source_rows_sha256": _digest("original-source-rows"),
        "rows": rows,
        "parent_outputs_generated_once": True,
        "parent_forward_passes_after_seal": 0,
        "eos_token_ids": [2],
    }
    envelope = {**body, "envelope_sha256": runner._json_sha256(body)}
    path.write_bytes(_canonical(envelope) + b"\n")
    return envelope


def _write_source_report(
    path: Path,
    *,
    rejected_layers: list[int] | None = None,
    parent_cache_key: str | None = None,
) -> dict[str, Any]:
    report = {
        "schema": "test-source-child-report",
        "rejected_layers": (
            list(runner.REJECTED_LAYER_ORDER)
            if rejected_layers is None
            else rejected_layers
        ),
        "accounting": {
            "observed_parent_unique_parameters": (
                runner.EXPECTED_PARENT_UNIQUE_PARAMETERS
            ),
            "observed_target_unique_parameters": (
                runner.EXPECTED_CHILD_UNIQUE_PARAMETERS
            ),
            "fraction_removed": runner.EXPECTED_COMPRESSION_FRACTION,
        },
        "final_checkpoint": {
            "state_sha256": _digest("source-child-state"),
            "unique_parameters": runner.EXPECTED_CHILD_UNIQUE_PARAMETERS,
        },
        "cache": {
            "cache_key": (
                _digest("original-parent-cache")
                if parent_cache_key is None
                else parent_cache_key
            )
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return report


def _config(
    tmp_path: Path,
    *,
    envelope: Path,
    source_report: Path,
    data_contract_only: bool = False,
    candidate_indices: tuple[int, ...] = runner.FIXED_CANDIDATE_INDICES,
) -> runner.RepairRunConfig:
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    tokenizer = tmp_path / "tokenizer"
    for directory in (parent, child, tokenizer):
        directory.mkdir(exist_ok=True)
    return runner.RepairRunConfig(
        parent_model=parent,
        source_child=child,
        source_child_report=source_report,
        source_child_report_sha256=_file_sha(source_report),
        cached_parent_outputs=envelope,
        cached_parent_outputs_sha256=_file_sha(envelope),
        tokenizer_directory=tokenizer,
        parent_horizon_cache_root=tmp_path / "parent-horizon-cache",
        child_horizon_cache_root=tmp_path / "child-horizon-cache",
        work_root=tmp_path / "work",
        checkpoint_destination=tmp_path / "published-child",
        report_path=tmp_path / "repair-report.json",
        device="cpu",
        data_contract_only=data_contract_only,
        candidate_indices=candidate_indices,
    )


def _ready_audit(tmp_path: Path) -> runner.SealedEnvelopeAudit:
    envelope_path = tmp_path / "envelope.json"
    _write_envelope(envelope_path, enriched=True)
    return runner.audit_sealed_horizon_envelope(
        envelope_path,
        expected_sha256=_file_sha(envelope_path),
    )


def test_existing_ids_without_tokenized_prompt_or_pieces_stop_before_models(
    tmp_path: Path,
) -> None:
    envelope_path = tmp_path / "envelope.json"
    source_report = tmp_path / "source-report.json"
    _write_envelope(envelope_path, enriched=False)
    _write_source_report(source_report)
    config = _config(
        tmp_path,
        envelope=envelope_path,
        source_report=source_report,
    )

    report = runner.execute_repair(config)

    assert report["status"] == "data_contract_blocked"
    assert report["execution"] == {
        "models_loaded": False,
        "parent_forward_calls": 0,
        "parent_response_generation_calls": 0,
        "child_forward_calls": 0,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "gradients_used": False,
        "checkpoint_published": False,
    }
    missing = report["sealed_envelope"]["missing_field_counts"]
    assert missing["input_ids"] == 16
    assert "parent_output_token_texts" not in missing
    assert "arithmetic_answer_correct" not in missing
    assert "termination_token_indices" not in missing
    assert report["sealed_envelope"][
        "existing_parent_output_ids_reused_as_forced_decode_ids"
    ]
    assert report["sealed_envelope"]["parent_response_generation_required"] is False
    assert config.report_path.is_file()
    assert not config.parent_horizon_cache_root.exists()
    assert not config.checkpoint_destination.exists()


def test_strict_existing_schema_is_enriched_by_bound_local_tokenizer(
    tmp_path: Path,
) -> None:
    envelope_path = tmp_path / "envelope.json"
    envelope = _write_envelope(envelope_path, enriched=False)
    for index, row in enumerate(envelope["rows"][:16]):
        row["input_ids"] = [101, 200 + index, 102]
        row["attention_mask"] = [1, 1, 1]
        row["split"] = "heldout"
    body = dict(envelope)
    body.pop("envelope_sha256")
    envelope["envelope_sha256"] = runner._json_sha256(body)
    envelope_path.write_bytes(_canonical(envelope) + b"\n")

    def decode(ids: Any) -> list[str]:
        pieces = ["x"] * len(ids)
        pieces[0] = "Return"
        pieces[1] = " "
        pieces[2:4] = ["4", "2"]
        pieces[4] = "\n"
        pieces[-1] = "<eos>"
        return pieces

    audit = runner.audit_sealed_horizon_envelope(
        envelope_path,
        expected_sha256=_file_sha(envelope_path),
        token_text_decoder=decode,
        tokenizer_eos_token_ids=(2,),
    )

    assert audit.ready
    assert audit.eos_token_ids == (2,)
    assert len(audit.recovery_rows) == 16
    assert all(
        row["forced_decode_ids"] == row["parent_output_ids"]
        for row in audit.recovery_rows
    )
    for row in audit.recovery_rows[:8]:
        assert row["arithmetic_answer_correct"] is True
        assert row["arithmetic_answer_token_indices"] == [2, 3]
        assert row["termination_token_indices"] == [63]
        assert row["termination_confirmed"] is True


def test_ready_contract_builds_only_sixteen_exact_forced_decode_batches(
    tmp_path: Path,
) -> None:
    audit = _ready_audit(tmp_path)

    batches = runner.build_horizon_batch_records(audit)

    assert audit.ready
    assert len(batches) == 16
    assert len(audit.manual_rows) == 12
    assert not set(audit.recovery_case_ids) & set(audit.manual_case_ids)
    for batch, row in zip(batches, audit.recovery_rows):
        assert batch["forced_decode_ids"] == row["parent_output_ids"]
        assert batch["input_ids"] == row["input_ids"]
        assert batch["attention_mask"] == row["attention_mask"]
    assert all(
        not case_id.startswith("manual_")
        for case_id in audit.recovery_case_ids
    )


def test_actual_path_binding_uses_final_prefill_and_decode_t_plus_one(
    tmp_path: Path,
) -> None:
    audit = _ready_audit(tmp_path)
    paths = []
    for batch_index in range(16):
        paths.extend(
            [
                f"batch.{batch_index:06d}/prefill/row.0000",
                f"batch.{batch_index:06d}/prefill/row.0001",
            ]
        )
        paths.extend(
            f"batch.{batch_index:06d}/decode.{step:03d}/row.0000"
            for step in range(64)
        )

    descriptors = runner.trace_path_descriptors_from_paths(paths, audit)
    first_case = audit.recovery_rows[0]
    first = [row for row in descriptors if row["case_id"] == first_case["id"]]

    assert first[0]["predicted_output_index"] is None
    assert first[1]["predicted_output_index"] == 0
    assert first[1]["predicted_token_id"] == first_case["parent_output_ids"][0]
    assert first[2]["phase"] == "decode.000"
    assert first[2]["predicted_output_index"] == 1
    assert first[-2]["phase"] == "decode.062"
    assert first[-2]["predicted_output_index"] == 63
    assert first[-1]["phase"] == "decode.063"
    assert first[-1]["predicted_output_index"] is None


def test_anchor_profiles_and_six_candidate_grid_are_exact_and_deterministic(
    tmp_path: Path,
) -> None:
    audit = _ready_audit(tmp_path)
    anchor_envelope = runner.build_anchor_case_envelope(audit)
    descriptors = runner.synthetic_trace_path_descriptors(audit)
    tagging = tag_lfm2_long_horizon_rows(anchor_envelope, descriptors)
    plan = build_lfm2_long_horizon_anchor_plan(anchor_envelope, descriptors)

    first_profiles = runner.build_evidence_profiles(plan, tagging)
    second_profiles = runner.build_evidence_profiles(plan, tagging)
    grid = runner.build_retry_grid(first_profiles)

    assert first_profiles == second_profiles
    balanced, termination = first_profiles
    assert len(balanced.selection_indices) == 256
    assert len(balanced.fit_indices) == 512
    assert len(balanced.validation_indices) == 128
    assert len(set(balanced.validation_indices)) == 128
    assert len(termination.selection_indices) == 256
    assert len(termination.fit_indices) == 512
    assert termination.validation_indices == balanced.validation_indices
    assert (
        termination.selection_termination_occurrences
        == balanced.selection_termination_occurrences + 24
    )
    assert (
        termination.fit_termination_occurrences
        == balanced.fit_termination_occurrences + 64
    )
    assert len(grid) == 6
    assert [
        (candidate.profile_name, candidate.endpoint_ridge)
        for candidate in grid
    ] == [
        ("balanced", 1e-8),
        ("balanced", 1e-6),
        ("balanced", 1e-4),
        ("termination_heavy", 1e-8),
        ("termination_heavy", 1e-6),
        ("termination_heavy", 1e-4),
    ]
    assert all(candidate.layer_order == (6, 8, 11, 13) for candidate in grid)
    assert len(tagging.excluded_case_ids) == 12
    selected = runner.filter_retry_grid(grid, (1, 4, 5))
    repeated = runner.filter_retry_grid(grid, (1, 4, 5))
    assert selected == repeated
    assert [candidate.candidate_index for candidate in selected] == [1, 4, 5]
    assert [candidate.candidate_sha256 for candidate in selected] == [
        grid[index].candidate_sha256 for index in (1, 4, 5)
    ]
    selection_report = runner.candidate_selection_report(
        grid,
        selected,
        (1, 4, 5),
    )
    assert selection_report["result_scope"] == (
        "requested_candidate_subset_only"
    )
    assert selection_report["covers_all_six_fixed_candidates"] is False
    with pytest.raises(
        runner.Lfm2LongHorizonRepairError,
        match="complete authenticated fixed grid",
    ):
        runner.filter_retry_grid(grid[1:], (1, 4, 5))


def test_real_length_fallback_has_128_unique_row_disjoint_validation_paths(
    tmp_path: Path,
) -> None:
    envelope_path = tmp_path / "real-length-envelope.json"
    envelope = _write_envelope(envelope_path, enriched=True)
    lengths = [9] * 8 + [27, 31, 28, 31, 38, 33, 58, 31]
    for row, length in zip(envelope["rows"][:16], lengths):
        row["parent_output_ids"] = row["parent_output_ids"][:length]
        row["forced_decode_ids"] = list(row["parent_output_ids"])
        row["parent_output_ids"][-1] = 2
        row["forced_decode_ids"][-1] = 2
        row["parent_output_token_texts"] = row[
            "parent_output_token_texts"
        ][:length]
        row["parent_output_token_texts"][-1] = "<eos>"
        row["parent_output_text"] = "".join(
            row["parent_output_token_texts"][:-1]
        )
        row["termination_token_indices"] = [length - 1]
    body = dict(envelope)
    body.pop("envelope_sha256")
    envelope["envelope_sha256"] = runner._json_sha256(body)
    envelope_path.write_bytes(_canonical(envelope) + b"\n")
    audit = runner.audit_sealed_horizon_envelope(
        envelope_path,
        expected_sha256=_file_sha(envelope_path),
    )
    anchor_envelope = runner.build_anchor_case_envelope(audit)
    descriptors = runner.synthetic_trace_path_descriptors(audit)
    tagging = tag_lfm2_long_horizon_rows(anchor_envelope, descriptors)

    plan = runner.build_repair_anchor_plan(
        anchor_envelope,
        descriptors,
        tagging=tagging,
    )
    profiles = runner.build_evidence_profiles(plan, tagging)

    assert isinstance(plan, runner.RowDisjointFallbackPlan)
    assert plan.policy_identity == runner.ROW_FALLBACK_POLICY_IDENTITY
    assert plan.strict_validation_unique_count < 128
    assert len(plan.validation.indices) == 128
    assert len(set(plan.validation.expanded_indices)) == 128
    split_sets = {
        "selection": set(plan.selection.indices),
        "fit": set(plan.fit.indices),
        "validation": set(plan.validation.indices),
    }
    assert not split_sets["selection"] & split_sets["fit"]
    assert not split_sets["selection"] & split_sets["validation"]
    assert not split_sets["fit"] & split_sets["validation"]
    assert all(
        len(split.case_ids) == 16
        for split in (plan.selection, plan.fit, plan.validation)
    )
    used = set().union(*split_sets.values())
    assert all(
        descriptors[index]["case_id"].startswith(
            ("recovery_math_", "recovery_code_")
        )
        for index in used
    )
    assert len(profiles[0].selection_indices) == 256
    assert len(profiles[0].fit_indices) == 512
    assert len(profiles[0].validation_indices) == 128
    serialized = plan.as_dict()
    seal = serialized.pop("plan_sha256")
    assert seal == runner._json_sha256(serialized)


def _state_digest(state: dict[int, int]) -> str:
    return runner._json_sha256(
        {str(index): value for index, value in sorted(state.items())}
    )


def test_atomic_tranche_rolls_back_all_four_layers_on_gate_rejection() -> None:
    state = {index: index for index in range(14)}
    baseline = copy.deepcopy(state)

    result = runner.run_atomic_tranche(
        (6, 8, 11, 13),
        snapshot_layer=lambda index: state[index],
        restore_layer=lambda index, snapshot: (
            state.__setitem__(index, snapshot) is None
        ),
        layer_state_sha256=lambda index: _digest(
            f"layer-{index}-state-{state[index]}"
        ),
        model_state_sha256=lambda: _state_digest(state),
        mutate_layer=lambda index: (
            state.__setitem__(index, state[index] + 100)
            or {"layer_index": index}
        ),
        promotion_gate=lambda: {
            "accepted": False,
            "reason": "manual-only gate rejected",
        },
    )

    assert state == baseline
    assert not result.accepted
    assert result.rolled_back
    assert result.rollback_exact
    assert result.baseline_model_sha256 == result.result_model_sha256
    assert [row["layer_index"] for row in result.layer_reports] == [6, 8, 11, 13]


def test_atomic_tranche_rolls_back_prior_layers_when_later_fit_raises() -> None:
    state = {index: index for index in range(14)}
    baseline = copy.deepcopy(state)

    def mutate(index: int) -> dict[str, int]:
        state[index] += 100
        if index == 11:
            raise ValueError("closed-form fit rejected")
        return {"layer_index": index}

    result = runner.run_atomic_tranche(
        (6, 8, 11, 13),
        snapshot_layer=lambda index: state[index],
        restore_layer=lambda index, snapshot: (
            state.__setitem__(index, snapshot) is None
        ),
        layer_state_sha256=lambda index: _digest(
            f"layer-{index}-state-{state[index]}"
        ),
        model_state_sha256=lambda: _state_digest(state),
        mutate_layer=mutate,
        promotion_gate=lambda: {"accepted": True},
    )

    assert state == baseline
    assert not result.accepted
    assert result.rollback_exact
    assert result.error == {
        "type": "ValueError",
        "message": "closed-form fit rejected",
    }
    assert [row["layer_index"] for row in result.layer_reports] == [6, 8]


def test_source_binding_rejects_wrong_frontier_or_cache_identity(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "source.json"
    _write_source_report(report_path, rejected_layers=[6, 8, 11])
    with pytest.raises(runner.Lfm2LongHorizonRepairError, match="frontier"):
        runner.load_source_child_binding(
            report_path,
            expected_sha256=_file_sha(report_path),
        )

    envelope_path = tmp_path / "envelope.json"
    _write_envelope(envelope_path, enriched=False)
    audit = runner.audit_sealed_horizon_envelope(
        envelope_path,
        expected_sha256=_file_sha(envelope_path),
    )
    _write_source_report(
        report_path,
        parent_cache_key=_digest("different-cache"),
    )
    source = runner.load_source_child_binding(
        report_path,
        expected_sha256=_file_sha(report_path),
    )
    config = _config(
        tmp_path,
        envelope=envelope_path,
        source_report=report_path,
    )
    with pytest.raises(runner.Lfm2LongHorizonRepairError, match="different"):
        runner.build_data_contract_report(
            config,
            envelope=audit,
            source=source,
        )


def test_source_binding_accepts_internal_report_seal_and_nested_parent_cache(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "sealed-source.json"
    report = _write_source_report(report_path)
    report["parent"] = {"cache": report.pop("cache")}
    report["report_sha256"] = runner._json_sha256(report)
    report_path.write_bytes(_canonical(report) + b"\n")

    source = runner.load_source_child_binding(
        report_path,
        expected_sha256=report["report_sha256"],
    )

    assert source.report_payload_sha256 == report["report_sha256"]
    assert source.parent_cache_key == _digest("original-parent-cache")
    assert source.source_child_state_sha256 == _digest("source-child-state")


def test_envelope_content_or_order_tamper_fails_self_seal(
    tmp_path: Path,
) -> None:
    envelope_path = tmp_path / "envelope.json"
    envelope = _write_envelope(envelope_path, enriched=False)
    envelope["rows"][0]["parent_output_ids"][0] += 1
    envelope_path.write_bytes(_canonical(envelope) + b"\n")

    with pytest.raises(runner.Lfm2LongHorizonRepairError, match="seal"):
        runner.audit_sealed_horizon_envelope(
            envelope_path,
            expected_sha256=_file_sha(envelope_path),
        )


def test_parser_exposes_only_fixed_zero_training_repair_controls() -> None:
    parser = runner.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--parent-horizon-cache-root" in option_strings
    assert "--data-contract-only" in option_strings
    assert "--candidate-indices" in option_strings
    assert "--train" not in option_strings
    assert "--learning-rate" not in option_strings
    assert runner.RIDGE_VALUES == (1e-8, 1e-6, 1e-4)
    assert runner.REJECTED_LAYER_ORDER == (6, 8, 11, 13)
    assert parser.get_default("candidate_indices") == (0, 1, 2, 3, 4, 5)


@pytest.mark.parametrize(
    "value",
    ("", "2,1", "1,1", "0,6", " 0,1", "0, 1", "00", "1,"),
)
def test_candidate_indices_parser_rejects_noncanonical_subsets(
    value: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        runner.parse_candidate_indices(value)


def test_candidate_indices_parse_and_config_request_are_sealed(
    tmp_path: Path,
) -> None:
    assert runner.parse_candidate_indices("0,2,5") == (0, 2, 5)
    envelope_path = tmp_path / "envelope.json"
    source_report = tmp_path / "source-report.json"
    _write_envelope(envelope_path, enriched=False)
    _write_source_report(source_report)
    config = _config(
        tmp_path,
        envelope=envelope_path,
        source_report=source_report,
        candidate_indices=(0, 2, 5),
    )

    request = config.as_dict()["candidate_selection"]
    seal = request.pop("request_sha256")
    assert request["requested_candidate_indices"] == [0, 2, 5]
    assert request["result_scope"] == "requested_candidate_subset_only"
    assert seal == runner._json_sha256(request)
    report = runner.execute_repair(config)
    assert report["candidate_selection"]["requested_candidate_indices"] == [
        0,
        2,
        5,
    ]
    report_body = dict(report)
    report_seal = report_body.pop("report_sha256")
    assert report_seal == runner._json_sha256(report_body)
    with pytest.raises(
        runner.Lfm2LongHorizonRepairError,
        match="sorted unique subset",
    ):
        _config(
            tmp_path,
            envelope=envelope_path,
            source_report=source_report,
            candidate_indices=(2, 1),
        )
