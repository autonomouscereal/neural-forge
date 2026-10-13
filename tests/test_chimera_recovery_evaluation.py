"""Independent wrong-answer, execution, continuation and comparison cases."""
import copy
import json

import pytest

from experiments.chimera_recovery import evaluation as e
from experiments.chimera_recovery.corpus import digest, file_sha256


def task(kind="arithmetic", **oracle):
    return {"schema": "recovery_eval_task.v1", "id": "case-1", "group_id": "held-group-1", "split": "test", "route": "text_arithmetic",
            "input": {"messages": [{"role": "user", "content": "What is 17 times 6?"}], "tools": [], "media": []},
            "oracle": {"kind": kind, **({"expected": "102"} if kind == "arithmetic" else {}), **oracle}}


def response(t, text="102", messages=None):
    return {"schema": "recovery_eval_response.v1", "id": t["id"], "input_sha256": digest(t["input"]),
            "messages": messages if messages is not None else [{"role": "assistant", "content": text}],
            "generation": {"stop_reason": "eos", "max_new_tokens": 128, "generated_tokens": 5}}


def receipt(responses, phase="baseline"):
    r = {"schema": "recovery_eval_run.v1", "phase": phase, "pins": {key: "a" * 64 for key in e.PIN_FIELDS},
         "model": {"checkpoint_sha256": "b" * 64 if phase == "baseline" else "c" * 64},
         "recording_complete": True, "full_answers_published": True, "responses_sha256": digest(responses), "training_group_ids": []}
    if phase == "after_recovery":
        r["recovery"] = {"base_checkpoint_sha256": "b" * 64, "training_receipt_sha256": "d" * 64}
    return r


def report(t, r, phase="baseline"):
    return e.evaluate([t], receipt([r], phase), [r], holdout_sha256="a" * 64)


def tool_task():
    t = task("tool")
    t["route"] = "tool_retry"
    t["input"]["tools"] = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"], "additionalProperties": False}}}]
    t["oracle"]["steps"] = [{"calls": [{"name": "lookup", "accepted_arguments": [{"key": "item"}], "outcome": {"status": "success", "json_equals": {"count": 12}}}]}]
    return t


def call(identifier, arguments='{"key":"item"}'):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": identifier, "type": "function", "function": {"name": "lookup", "arguments": arguments}}]}


def execution(identifier, content='{"count":12}', status="success"):
    result = {"role": "tool", "tool_call_id": identifier, "name": "lookup", "content": content}
    evidence = {"tool_call_id": identifier, "status": status, "source_verified": True, "receipt_sha256": "e" * 64, "result_sha256": digest(result)}
    return result, evidence


def test_wrong_arithmetic_and_alternate_correct_explanations():
    t = task()
    assert e.grade_semantics(t, response(t, "17 × 6 is 118."))["status"] == "fail"
    assert e.grade_semantics(t, response(t, "Six groups of seventeen make one hundred and two.\nAnswer: 102."))["status"] == "pass"
    assert e.grade_semantics(t, response(t, '{"answer": 102}'))["status"] == "pass"
    assert e.grade_semantics(t, response(t, "102 is one possibility. Final answer: 118"))["status"] == "fail"


def test_correct_fragment_followed_by_loop_is_not_complete_success():
    t = task()
    r = response(t, "Answer: 102\n" + "102 " * 12)
    r["generation"].update(stop_reason="length", token_ids=[1, 2, 3, 7, 7, 7, 7, 7])
    graded = report(t, r)["rows"][0]
    assert graded["semantic"]["status"] == "pass"
    assert graded["generation"]["cap_reached"] and graded["generation"]["loop_candidate"]
    assert not graded["successful_complete_response"]


def test_safe_code_fixtures_accept_distinct_algorithms_and_catch_bug():
    t = task("python_function", function="dedupe", cases=[{"args": [[4, 2, 4, 3, 2]], "expected": [4, 2, 3]}, {"args": [[]], "expected": []}, {"args": [[1, 1]], "expected": [1]}])
    first = "def dedupe(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out"
    alternate = "def dedupe(xs):\n    return list(dict.fromkeys(xs))"
    wrong = "def dedupe(xs):\n    return sorted(set(xs))"
    assert e.grade_semantics(t, response(t, first))["status"] == "pass"
    assert e.grade_semantics(t, response(t, alternate))["status"] == "pass"
    assert e.grade_semantics(t, response(t, wrong))["status"] == "fail"
    assert e.grade_semantics(t, response(t, "import os\ndef dedupe(xs):\n    os.system('should never execute')\n    return xs"))["status"] == "ungraded"


def test_python_instruction_and_allocation_limits():
    t = task("python_function", function="f", cases=[{"args": [], "expected": []}])
    assert e.grade_semantics(t, response(t, "def f():\n    a = [1]\n    a *= 100000000000\n    return a"))["status"] == "fail"
    assert e.grade_semantics(t, response(t, "def f():\n    return f()"))["status"] == "fail"


def test_invalid_arguments_unanswered_tool_and_unverified_results_are_failures():
    t = tool_task()
    bad = response(t, messages=[call("c1", '{"key":12}')])
    assert e.grade_semantics(t, bad)["status"] == "fail"
    unanswered = response(t, messages=[call("c1")])
    assert "unanswered_tool_at_end" in e.grade_semantics(t, unanswered)["errors"]
    result, evidence = execution("c1")
    r = response(t, messages=[call("c1"), result, {"role": "assistant", "content": "There are 12."}])
    assert e.grade_semantics(t, r)["status"] == "fail"
    r["tool_outcomes"] = [evidence]
    assert e.grade_semantics(t, r)["status"] == "pass"
    r["tool_outcomes"][0]["result_sha256"] = "f" * 64
    assert e.grade_semantics(t, r)["status"] == "fail"


def test_proposed_call_is_separate_from_observed_execution():
    t = tool_task()
    del t["oracle"]["steps"][0]["calls"][0]["outcome"]
    result = e.grade_semantics(t, response(t, messages=[call("c1")]))
    assert result["status"] == "pass" and result["proposed_calls_correct"]
    assert not result["execution_required"] and result["pending_tool_ids"] == ["c1"]


def test_error_retry_and_followup_require_linked_outcomes_and_actual_turn():
    t = tool_task()
    t["oracle"]["steps"].insert(0, {"calls": [{"name": "lookup", "accepted_arguments": [{"key": "item"}], "outcome": {"status": "error", "content": "timeout"}}]})
    t["oracle"].update(min_followup_user_turns=1, followup_user_messages=[{"role": "user", "content": "Try again."}])
    error, error_receipt = execution("failed", "timeout", "error")
    success, success_receipt = execution("retry")
    r = response(t, messages=[call("failed"), error, {"role": "user", "content": "Try again."}, call("retry"), success, {"role": "assistant", "content": "Done."}])
    r["tool_outcomes"] = [error_receipt, success_receipt]
    assert e.grade_semantics(t, r)["status"] == "pass"
    r["messages"][2]["content"] = "Ignore the task."
    assert "multiturn_user_followup_content_mismatch" in e.grade_semantics(t, r)["errors"]


def test_parallel_calls_require_same_batch_and_allow_order_variation():
    t = tool_task()
    t["oracle"]["steps"] = [{"parallel": True, "calls": [{"name": "lookup", "accepted_arguments": [{"key": key}]} for key in ("a", "b")]}]
    one, two = call("a", '{"key":"a"}'), call("b", '{"key":"b"}')
    same = response(t, messages=[{"role": "assistant", "content": None, "tool_calls": two["tool_calls"] + one["tool_calls"]}])
    assert e.grade_semantics(t, same)["status"] == "pass"
    assert e.grade_semantics(t, response(t, messages=[one, two]))["status"] == "fail"


def test_teacher_topk_coverage_is_not_consecutive_free_run_and_early_eos_is_not_exact8():
    t = task()
    t["teacher_reference"] = {"token_ids": list(range(32)), "context_sha256": "c" * 64, "checkpoint_sha256": "b" * 64}
    r = response(t)
    r["generation"]["token_ids"] = list(range(8)) + [99] * 24
    r["teacher_forced"] = {"context_kind": "teacher_forced", "context_sha256": "c" * 64, "teacher_checkpoint_sha256": "b" * 64,
                           "k": 2, "positions": [{"position": i, "topk_token_ids": [99, i]} for i in range(16)]}
    metrics = e.token_metrics(t, r)
    assert metrics["teacher_forced"]["position_coverage"] == .5
    assert metrics["teacher_forced"]["topk_rate_on_covered"] == 1
    assert metrics["free_run"]["lcp"] == 8
    assert metrics["free_run"]["exact_continuation"] == {"5": True, "8": True, "16": False, "32": False}
    r["generation"]["token_ids"] = [0]
    assert e.token_metrics(t, r)["free_run"]["exact_continuation"]["8"] is None
    r["teacher_forced"]["positions"].append(r["teacher_forced"]["positions"][0])
    with pytest.raises(ValueError, match="duplicate"):
        e.token_metrics(t, r)


def test_complete_recordings_and_heldout_groups_are_required():
    t, r = task(), response(task())
    run = receipt([r])
    run["training_group_ids"] = [t["group_id"]]
    with pytest.raises(ValueError, match="entered training"):
        e.evaluate([t], run, [r], holdout_sha256="a" * 64)
    run = receipt([r])
    run["responses_sha256"] = digest([])
    with pytest.raises(ValueError, match="every heldout"):
        e.evaluate([t], run, [], holdout_sha256="a" * 64)


def test_matched_comparison_requires_identical_pins_lineage_and_publishes_full_answers(tmp_path):
    t = task()
    before, after = report(t, response(t, "118")), report(t, response(t, "102"), "after_recovery")
    comparison = e.compare(before, after)
    e.publish(comparison, tmp_path / "comparison")
    published = (tmp_path / "comparison/full_answers_and_outcomes.md").read_text(encoding="utf-8")
    assert '"content": "118"' in published and '"content": "102"' in published
    assert comparison["before_counts"]["semantic"] == {"fail": 1}
    changed = copy.deepcopy(after)
    changed["run"]["pins"]["template_sha256"] = "e" * 64
    changed["report_sha256"] = digest({k: v for k, v in changed.items() if k != "report_sha256"})
    with pytest.raises(ValueError, match="template_sha256"):
        e.compare(before, changed)


def test_captured_optional_omission_accepts_real_alternatives_and_not_extra_args():
    t = tool_task()
    function = t["input"]["tools"][0]["function"]
    function["parameters"]["properties"]["mode"] = {"type": "string"}
    t["oracle"]["steps"] = [{"calls": [{"name": "lookup", "argument_choices": {"key": ["item", "entry"], "mode": ["fast", ""]}, "optional_arguments": ["mode"]}]}]
    assert e.grade_semantics(t, response(t, messages=[call("a", '{"key":"entry"}')]))["status"] == "pass"
    assert e.grade_semantics(t, response(t, messages=[call("a", '{"key":"item","mode":"fast"}')]))["status"] == "pass"
    assert e.grade_semantics(t, response(t, messages=[call("a", '{"key":"item","mode":"slow"}')]))["status"] == "fail"
    assert e.grade_semantics(t, response(t, messages=[call("a", '{"key":"item","extra":1}')]))["status"] == "fail"


def test_bfcl_pack_excludes_targets_from_runner_inputs_and_refuses_training_group(tmp_path):
    t = tool_task()
    source = {"schema": "recovery_corpus.v1", "id": "real-source", "split": "test", "route": "tool_call", "messages": t["input"]["messages"] + [call("target")],
              "tools": t["input"]["tools"], "quality": {"schema_complete": True},
              "provenance": {"source_verified": True, "recovery_group_id": "held-group", "source_ids": ["registry:42"], "bfcl_schema_target_adaptation": {
                  "original_acceptable_targets": {"expected_tool_calls": [{"lookup": {"key": ["item", "entry"]}}]}, "original_schemas": []}}}
    held, train = tmp_path / "held.jsonl", tmp_path / "train.jsonl"
    held.write_text(json.dumps(source) + "\n", encoding="utf-8")
    train.write_text(json.dumps({"split": "train", "provenance": {"recovery_group_id": "train-group"}}) + "\n", encoding="utf-8")
    pack = e.prepare_bfcl(heldout_corpus=held, training_corpus=train, expected_heldout_sha256=file_sha256(held), expected_training_sha256=file_sha256(train), output=tmp_path / "pack")
    assert pack["tasks"] == 1 and pack["task_source_groups_disjoint_from_training"]
    inputs = json.loads((tmp_path / "pack/inputs.jsonl").read_text(encoding="utf-8"))
    assert "oracle" not in inputs and all(m["role"] != "assistant" for m in inputs["input"]["messages"])
    train.write_text(json.dumps({"split": "train", "provenance": {"recovery_group_id": "held-group"}}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps training"):
        e.prepare_bfcl(heldout_corpus=held, training_corpus=train, expected_heldout_sha256=file_sha256(held), expected_training_sha256=file_sha256(train), output=tmp_path / "leaking")
