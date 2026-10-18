"""Tests for ``toolcall_selfgen.py``.

What is REAL here: the parent's pinned tokenizer.json and chat_template.jinja (every render,
every mask, every round trip goes through the template the teacher will be rescored with),
the campaign review surface and eval split (`finalize` screens with
distill_corpus_build.build_review_index), and the rescoring hand-off is loaded through its
consumer, ``qtext_teacher_cache.load_text_corpus``.  The only stand-in is the network peer:
an in-process llama-server ``/completion`` that answers by role with scripted text, so the
generator's control flow (categories, injected errors, clarify / follow-up turns, journal,
resume, cutoff) runs end to end on CPU.  The model's behaviour itself is proven by the
live smoke run on the P40s, not here.

  test_version_is_pinned
  test_sandbox_tools_are_exact                      values + error objects
  test_heldout_filter_catches_bank_families_only    every bank function caught; common train names not
  test_toolsets_may_share_names_but_not_reserved_or_repeated_ones   1.0.3
  test_validate_tool_and_arguments
  test_parse_rejects_malformed_turns
  test_native_call_round_trip_through_the_template  render -> parse -> identical calls (real template)
  test_supervised_mask_is_exactly_the_assistant_turns
  test_grounded_reads_both_number_conventions
  test_category_checks
  test_bank_is_deterministic_sealed_and_heldout
  test_bank_grader_on_constructed_responses          perfect passes, wrong value / missing parallel call fail
  test_http_4xx_rejects_the_sample_and_keeps_the_server   1.0.1
  (end to end also runs bank-run over two server URLs and bank-grade: 1.0.2)
  test_sandbox_never_raises_on_extreme_values        1.1.0: OverflowError escaped the sandbox and killed a worker
  test_claimed_actions_flags_claims_not_declines     1.1.0: zero-call content rule, patterns
  test_zero_call_check_verdicts                      1.1.0: patterns short-circuit; judge fits/claims/no verdict
  test_end_to_end_stub_run_finalize_export           schemas -> plan -> run (kill + resume) -> finalize -> export -> qtext load
      1.1.0: + an internal failure rejects one attempt and the run completes; + a 1.0.x journal
      (no verdicts, one claiming answer) resumes: re-check, reopen, re-run; finalize refuses an
      unresolved reopened row
      1.2.0: + continuation: continue-plan (frozen, refuses another plan) -> a second run under a new
      base seed, stopped by --stop-by (rc 4; a reopened row in hand is journaled `deadline`, no gap)
      -> finalize --accept-short --prior-manifest (overlapping ids refused) -> --cut-at-gap ->
      export over both runs (row ranges, unique ids, qtext load; a run named twice refused)
  test_continue_plan_reindexes_the_tail              1.2.0
  test_parse_utc_needs_a_zone                        1.2.0

Server:
  cd $OC/distill/tools && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$OC/distill/tools \\
    CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src \\
    $PY -m pytest -o addopts="" -p no:cacheprovider test_toolcall_selfgen.py -q
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import toolcall_selfgen as tc  # noqa: E402

PARENT = Path(os.environ.get("TOOLCALL_PARENT", str(tc.DEFAULT_PARENT)))
needs_parent = pytest.mark.skipif(not (PARENT / "tokenizer.json").is_file(), reason="parent tokenizer not on this machine")


@pytest.fixture(scope="module")
def renderer():
    if not (PARENT / "tokenizer.json").is_file():
        pytest.skip("parent tokenizer not on this machine")
    return tc.Renderer(PARENT)


def test_version_is_pinned():
    assert tc.TOOL_VERSION == "1.2.0" and tc.TOOL_VERSION.split(".")[0] == "1"  # tool_major unchanged: 1.x runs resume
    assert tc.CONTENT_RULES == "zero-call-content-v1" and "judge" not in tc.SAMPLING  # the run lock is unchanged by 1.1.0
    assert tc.EOG_IDS == (248_046, 248_044)
    assert abs(sum(tc.CATEGORY_WEIGHTS.values()) - 1.0) < 1e-9
    assert tc.SAMPLING["backend"]["n_predict"] == 900 and tc.BACKEND_MAX_CHARS == 4000  # 1.0.4


def test_continue_plan_reindexes_the_tail():
    plan = [{"index": n, "id": f"tc-{n:05d}", "category": "single"} for n in range(10)]
    tail = tc.continue_plan(plan, 6)
    assert [t["index"] for t in tail] == [0, 1, 2] and [t["source_index"] for t in tail] == [7, 8, 9]
    assert [t["id"] for t in tail] == ["tc-00007", "tc-00008", "tc-00009"] and list(tail[0])[0] == "index"  # key order kept
    assert plan[7]["index"] == 7 and "source_index" not in plan[7]  # the input plan is not touched
    with pytest.raises(tc.ToolcallError):
        tc.continue_plan(plan, 9)  # nothing left
    with pytest.raises(tc.ToolcallError):
        tc.continue_plan(plan[1:], 3)  # not 0..n-1


def test_parse_utc_needs_a_zone():
    assert tc.parse_utc("2026-10-15T21:00:00Z") == 1792098000.0
    with pytest.raises(tc.ToolcallError):
        tc.parse_utc("2026-10-15T21:00:00")


def test_sandbox_tools_are_exact():
    assert tc.calculate("(17 * 6) / 3")["result"] == 34
    assert tc.calculate("2 ** 10")["result"] == 1024
    assert tc.calculate("1/0")["error"]["code"] == "MATH_ERROR"
    assert tc.calculate("__import__('os')")["error"]["code"] == "INVALID_ARGUMENT"
    assert tc.convert_units(5, "km", "mi")["result"] == pytest.approx(3.106856, abs=1e-6)
    assert tc.convert_units(212, "F", "C")["result"] == 100
    assert tc.convert_units(1, "kg", "m")["error"]["code"] == "INVALID_ARGUMENT"
    assert tc.date_add("2026-10-14", 10) == {"start_date": "2026-10-14", "days": 10, "date": "2026-10-24", "weekday": "Saturday"}
    assert tc.days_between("2026-01-01", "2026-10-14")["days"] == 286
    assert tc.currency_convert(100, "USD", "EUR")["converted"] == 92.0
    assert tc.split_bill(120, 18, 4)["per_person"] == 35.4
    assert tc.percent_change(80, 100)["percent_change"] == 25.0
    assert tc.text_statistics("One two three. Four!")["words"] == 4
    assert tc.loan_payment(240000, 6.0, 360)["monthly_payment"] == pytest.approx(1438.92, abs=0.01)
    assert tc.timezone_convert("23:40", -5, 5.5) == {"time": "10:10", "day_shift": 1}
    assert tc.execute_sandbox("split_bill", {"total": 10, "tip_percent": 10, "people": 0})["error"]["code"] == "INVALID_ARGUMENT"
    assert tc.execute_sandbox("calculate", {"expr": "1"})["error"]["code"] == "INVALID_ARGUMENT"


def test_sandbox_never_raises_on_extreme_values():
    """1.1.0: the review reproduced three OverflowErrors that escaped execute_sandbox (it caught
    only TypeError) and, through process(), killed a worker thread."""
    cases = [("calculate", {"expression": "999999999999**64"}), ("calculate", {"expression": "10**60*10**60*10**60*10**60*10**60*10**60"}),
             ("date_add", {"date": "9999-12-30", "days": 5}), ("date_add", {"date": "0001-01-02", "days": -5}),
             ("currency_convert", {"amount": 10 ** 400, "from_currency": "USD", "to_currency": "EUR"}),
             ("convert_units", {"value": 1e308, "from_unit": "mi", "to_unit": "mm"}),
             ("convert_units", {"value": float("nan"), "from_unit": "km", "to_unit": "m"}),
             ("split_bill", {"total": 10 ** 400, "tip_percent": 18, "people": 2}),
             ("percent_change", {"old_value": 1e-320, "new_value": 1e308}), ("calculate", {"expression": "exp(1000)"}),
             ("calculate", {"expression": "-" * 290 + "1"}), ("calculate", {"expression": "(" * 140 + "1" + ")" * 140})]
    for name, arguments in cases:
        result = tc.execute_sandbox(name, arguments)
        assert isinstance(result, dict), (name, arguments)
        if "error" not in result:
            assert tc._finite_tree(result) and tc.grounded("x", tc.computed_values(result, arguments)) in (True, False)
    assert tc.execute_sandbox("calculate", {"expression": "999999999999**64"})["error"]["code"] == "MATH_ERROR"
    assert tc.execute_sandbox("date_add", {"date": "9999-12-30", "days": 5})["error"]["code"] == "INVALID_ARGUMENT"
    # unchanged for everything 1.0.4 already handled (journaled rows keep their exact tool text)
    assert tc.calculate("exp(1000)") == {"error": {"code": "INVALID_ARGUMENT", "message": "cannot evaluate expression: math range error"}}
    assert tc.execute_sandbox("calculate", {"expression": "1e308*10"}) == {"error": {"code": "MATH_ERROR", "message": "result is not a finite real number"}}


_TC00030 = ("Estimado Sr. Ortiz,  Gracias por su consulta. Para poder darle una respuesta precisa, he verificado el estado actual de la "
            "reserva en el sistema.  1. Estado de la reserva: La reserva para el 20 de octubre de 2026 a las 20:30 h para 4 personas en "
            "el restaurante L'Atelier de la Garance (ID: R-89302) se encuentra confirmada.")


def test_claimed_actions_flags_claims_not_declines():
    assert tc.claimed_actions(_TC00030)  # production row tc-00030, accepted by 1.0.4
    claims = ["I checked your order and it shipped yesterday.", "I've booked the table for 8pm.", "Our records show your account is active.",
              "According to the system, your plan renews on May 2.", "Ich habe Ihre Buchung im System überprüft.",
              "J'ai vérifié votre réservation.", "Ho controllato la prenotazione.", "Verifiquei o seu pedido.", "我已经为您查询了订单状态。",
              "予約を確認しました。", "He reservado la mesa para las 21:00."]
    for text in claims:
        assert tc.claimed_actions(text), text
    honest = ["I checked the available functions, and none of them can cancel a subscription.",
              "No he verificado nada todavía; puedo hacerlo si quieres.", "I have not checked your order.",
              "A no-show policy usually allows 15 minutes. If the system shows an error, call the restaurant.",
              "Unfortunately I don't have a function for that; you can do it in the app.",
              "He revisado las herramientas disponibles y ninguna permite cambiar la contraseña.",
              "Compound growth means interest is added to the principal each period."]
    for text in honest:
        assert not tc.claimed_actions(text), text


def test_zero_call_check_verdicts(renderer):
    tools = [tc.SANDBOX_TOOLS["calculate"][0]]
    asked = []

    def judge(text, end="eos"):
        def gen(ids, role):
            asked.append(role)
            return {"text": text, "end": end}
        return gen
    ok = tc.zero_call_check(judge('{"fits": false, "claims": false}'), renderer, tools, "What is a mortgage?", "A loan secured by a home.")
    assert ok["failed"] == [] and ok["rules"] == tc.CONTENT_RULES and ok["judge"] == {"fits": False, "claims": False}
    assert tc.zero_call_check(judge('{"fits": true, "claims": false}'), renderer, tools, "q", "a")["failed"] == ["judge: an offered function serves this request"]
    assert tc.zero_call_check(judge('```json\n{"fits": false, "claims": true}\n```'), renderer, tools, "q", "a")["failed"] == [
        "judge: the reply claims a lookup, an action or record facts"]
    assert tc.zero_call_check(judge("yes"), renderer, tools, "q", "a")["failed"] == ["judge gave no verdict"]
    assert tc.zero_call_check(judge('{"fits": false, "claims": false}', end="limit"), renderer, tools, "q", "a")["failed"] == ["judge gave no verdict"]
    n = len(asked)
    pat = tc.zero_call_check(judge('{"fits": false, "claims": false}'), renderer, tools, "q", _TC00030)
    assert pat["failed"] and pat["judge"] is None and len(asked) == n  # patterns short-circuit the judge
    assert all(role == "judge" for role in asked)
    assert tc.CHECK_SAMPLING["judge"]["temperature"] == 0.0


def test_heldout_filter_catches_bank_families_only():
    for name, tool in list(tc.HELDOUT_API.items()) + [(n, t[0]) for n, t in tc.HELDOUT_SANDBOX.items()]:
        assert tc.heldout_touch(tool) is not None, name
    clean = [("create_campaign", "Create a social media campaign."), ("get_route_distance", "Driving route length between two addresses."),
             ("list_packages", "List installed software packages on a VM."), ("get_account_interest_rate", "Savings account rate."),
             ("schedule_meeting", "Add a meeting to the calendar."), ("search_hotels", "Find hotels by city and dates.")]
    for name, desc in clean:
        tool = tc._fn(name, desc, {"x": {"type": "string", "description": "x"}}, ["x"])
        assert tc.heldout_touch(tool) is None, name
    for name in tc.SANDBOX_TOOLS:
        assert tc.heldout_touch(tc.SANDBOX_TOOLS[name][0]) is None, name


def test_storage_keeps_key_order(tmp_path):
    """1.0.0 pre-release regression: a sort_keys journal re-rendered different tokens (the template
    renders schemas and arguments in insertion order)."""
    tool = tc.SANDBOX_TOOLS["split_bill"][0]
    row = {"tools": [tool], "args": {"total": 1, "tip_percent": 2, "people": 3}}
    tc.write_jsonl(tmp_path / "a.jsonl", [row])
    tc.write_json_atomic(tmp_path / "a.json", row)
    for back in (tc.read_jsonl(tmp_path / "a.jsonl")[0], tc.read_json(tmp_path / "a.json")):
        assert list(back["tools"][0]) == ["type", "function"]
        assert list(back["tools"][0]["function"]["parameters"]["properties"]) == ["total", "tip_percent", "people"]
        assert list(back["args"]) == ["total", "tip_percent", "people"]


class _FakeClient:
    """generate() returns a fixed schema answer: no HTTP, only the acceptance logic is under test."""

    def __init__(self, text):
        self.text = text

    def generate(self, prompt_ids, role, seed):
        return {"text": self.text, "end": "eos", "tokens": 10, "server": "fake", "wall_s": 0.0}


class _NoRender:
    def render(self, *a, **k):
        return ""

    def encode(self, text):
        return [1]


def test_toolsets_may_share_names_but_not_reserved_or_repeated_ones():
    """1.0.3: library-wide name uniqueness refused most second toolsets of a domain."""
    def spec(name):
        return {"name": name, "description": "d.", "parameters": {"type": "object", "properties": {"x": {"type": "string", "description": "x"}}, "required": ["x"]}}
    text = json.dumps([spec("get_stock_quote"), spec("update_watchlist"), spec("get_stock_quote"), spec("calculate"), spec("track_parcel")])
    first = tc.synthesize_toolset(_FakeClient(text), _NoRender(), "stock_market", "stocks", "flavor", 0, 1)
    second = tc.synthesize_toolset(_FakeClient(text), _NoRender(), "stock_market", "stocks", "flavor", 1, 1)
    for record in (first, second):
        assert [t["function"]["name"] for t in record["tools"]] == ["get_stock_quote", "update_watchlist"]
        reasons = {d["name"]: d["problems"][0] for d in record["attempts"][-1]["dropped"]}
        assert reasons["calculate"] == "reserved name" and "held-out" in reasons["track_parcel"]


def test_validate_tool_and_arguments():
    good = {"name": "get_order", "description": "Fetch an order.", "parameters": {"type": "object", "properties": {
        "order_id": {"type": "string", "description": "id"}, "limit": {"type": "integer", "description": "n", "minimum": 1},
        "status": {"type": "string", "enum": ["open", "closed"], "description": "s"},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "t"}}, "required": ["order_id"]}}
    assert tc.validate_tool(good) == []
    assert tc.validate_tool(dict(good, name="GetOrder"))
    assert tc.validate_tool(dict(good, parameters={"type": "object", "properties": {"a": {"type": "date"}}, "required": []}))
    assert tc.validate_tool(dict(good, parameters={"type": "object", "properties": {}, "required": ["ghost"]}))
    tool = tc.normalize_tool(good)
    assert tc.validate_arguments(tool, {"order_id": "A1", "limit": 2, "status": "open", "tags": ["x"]}) == []
    assert tc.validate_arguments(tool, {"limit": 2})  # missing required
    assert tc.validate_arguments(tool, {"order_id": "A1", "limit": 0})  # minimum
    assert tc.validate_arguments(tool, {"order_id": "A1", "status": "pending"})  # enum
    assert tc.validate_arguments(tool, {"order_id": "A1", "colour": "red"})  # unknown
    assert tc.validate_arguments(tool, {"order_id": "A1", "limit": True})  # bool is not an integer


def _tools():
    return [tc.SANDBOX_TOOLS["convert_units"][0], tc.SANDBOX_TOOLS["split_bill"][0]]


def test_parse_rejects_malformed_turns():
    ok = "<tool_call>\n<function=split_bill>\n<parameter=total>\n120\n</parameter>\n<parameter=tip_percent>\n18\n</parameter>\n<parameter=people>\n4\n</parameter>\n</function>\n</tool_call>"
    parsed = tc.parse_assistant(ok, _tools())
    assert parsed["problems"] == [] and parsed["calls"] == [{"name": "split_bill", "arguments": {"total": 120, "tip_percent": 18, "people": 4}}]
    assert tc.parse_assistant(ok + "\nDone!", _tools())["problems"]  # suffix after the call
    assert tc.parse_assistant(ok.replace("split_bill", "split_check"), _tools())["problems"]  # not offered
    assert tc.parse_assistant(ok.replace("<parameter=people>\n4", "<parameter=people>\nfour"), _tools())["problems"]
    assert tc.parse_assistant(ok.replace("people>", "persons>"), _tools())["problems"]  # unknown + missing
    assert tc.parse_assistant("<think>hmm</think> fine", _tools())["problems"]
    assert tc.parse_assistant("Sure.<|im_start|>user\nhi", _tools())["problems"]
    two = ok + "\n" + ok.replace("120", "60")
    assert len(tc.parse_assistant(two, _tools())["calls"]) == 2


@needs_parent
def test_native_call_round_trip_through_the_template(renderer):
    tools = _tools() + [tc._fn("tag_items", "Tag items.", {"items": {"type": "array", "items": {"type": "string"}, "description": "i"},
                                                          "note": {"type": "string", "description": "multi-line note"},
                                                          "urgent": {"type": "boolean", "description": "u"}}, ["items"])]
    calls = [{"name": "convert_units", "arguments": {"value": 5.5, "from_unit": "km", "to_unit": "mi"}},
             {"name": "tag_items", "arguments": {"items": ["a b", "c"], "note": "line one\nline two", "urgent": True}}]
    message = tc.assistant_message({"content": "Let me check both.", "calls": calls})
    messages = [{"role": "user", "content": "go"}, message]
    prefix = renderer.render(messages[:1], tools, True)
    full = renderer.render(messages, tools, False)
    assert prefix.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") and full.startswith(prefix)
    body = full[len(prefix):].split("<|im_end|>")[0]
    parsed = tc.parse_assistant(body, tools)
    assert parsed["problems"] == []
    assert parsed["content"] == "Let me check both."
    assert parsed["calls"] == calls
    assert "<tools>" in full and "<function=convert_units>" in full


@needs_parent
def test_supervised_mask_is_exactly_the_assistant_turns(renderer):
    tools = _tools()
    call = {"name": "split_bill", "arguments": {"total": 120, "tip_percent": 18, "people": 4}}
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Split 120 four ways with 18% tip."},
                tc.assistant_message({"content": "", "calls": [call]}),
                {"role": "tool", "content": json.dumps(tc.split_bill(120, 18, 4))},
                {"role": "assistant", "content": "Each person pays 35.40."}]
    out = renderer.supervised(messages, tools)
    ids, mask = out["ids"], out["target_mask"]
    assert len(ids) == len(mask) and mask[0] == 0
    supervised = renderer.decode([i for i, m in zip(ids, mask) if m])
    expected = renderer.render(messages[:3], tools, False)[len(renderer.render(messages[:2], tools, True)):].rstrip("\n") + \
        renderer.render(messages, tools, False)[len(renderer.render(messages[:4], tools, True)):].rstrip("\n")
    assert supervised == expected
    assert "<tool_response>" not in supervised and "<think>" not in supervised
    assert supervised.count("<|im_end|>") == 2 and supervised.endswith("Each person pays 35.40.<|im_end|>")
    # The tools system block shows the call format twice; the one real call is supervised.
    assert ids.count(tc.TOOL_TOKEN_IDS["<tool_call>"]) == 3
    assert [i for i, m in zip(ids, mask) if m].count(tc.TOOL_TOKEN_IDS["<tool_call>"]) == 1


def test_grounded_reads_both_number_conventions():
    assert tc.grounded("Each person pays $35.40.", [35.4])
    assert tc.grounded("Jeder zahlt 35,40 €.", [35.4])
    assert tc.grounded("That is 1,609.34 metres.", [1609.344])
    assert tc.grounded("Es sind 1.609,34 Meter.", [1609.344])
    assert not tc.grounded("Each person pays about forty dollars.", [35.4])
    assert tc.grounded("It lands on October 24, a Saturday.", ["2026-10-24"])
    assert tc.grounded("anything", [])
    assert tc.computed_values(tc.split_bill(120, 18, 4), {"total": 120, "tip_percent": 18, "people": 4}) == [21.6, 141.6, 35.4]


def test_category_checks():
    call = {"name": "x", "arguments": {}}
    turn = lambda n, text="": {"calls": [call] * n, "content": text}  # noqa: E731
    assert not tc.category_checks("single", [[turn(1), turn(0, "done")]], {})["failed"]
    assert tc.category_checks("parallel", [[turn(1), turn(1), turn(0, "x")]], {})["failed"]
    assert not tc.category_checks("parallel", [[turn(2), turn(0, "x")]], {})["failed"]
    assert not tc.category_checks("sequential", [[turn(1), turn(1), turn(0, "x")]], {})["failed"]
    assert tc.category_checks("no_tool", [[turn(1), turn(0, "x")]], {})["failed"]
    assert not tc.category_checks("clarify", [[turn(0, "Which order?")], [turn(1), turn(0, "ok")]], {})["failed"]
    assert tc.category_checks("clarify", [[turn(0, "Sure.")], [turn(1), turn(0, "ok")]], {})["failed"]
    assert tc.category_checks("multiturn", [[turn(1), turn(0, "a")], [turn(0, "b")]], {})["failed"]


def test_bank_is_deterministic_sealed_and_heldout():
    a, b = tc.build_bank(), tc.build_bank()
    assert a == b and tc.check_seal(a, "bank_sha256")
    assert a["lanes"] == {"exec_single": 40, "exec_parallel": 20, "api_select": 40, "multi_turn": 15, "error_recovery": 15,
                          "no_tool": 20, "missing_tool": 20, "clarify": 15}
    train_names = set(tc.SANDBOX_TOOLS)
    for case in a["cases"]:
        offered = {t["function"]["name"] for t in case["tools"]}
        assert offered and not offered & train_names, case["id"]
        assert offered <= set(a["heldout_functions"])
        for call in case["expected"].get("calls", []):
            assert call["name"] in offered
            full = dict(call["arguments"], **{k: "free text" for k in call.get("ungraded_required", [])})
            assert tc.validate_arguments(next(t for t in case["tools"] if t["function"]["name"] == call["name"]), full) == []
    assert len({c["id"] for c in a["cases"]}) == len(a["cases"])


def _native(calls):
    out = []
    for call in calls:
        args = dict(call["arguments"], **{k: "The invoice PDF does not download." for k in call.get("ungraded_required", [])})
        params = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in args.items())
        out.append(f"<tool_call>\n<function={call['name']}>\n{params}</function>\n</tool_call>")
    return "\n".join(out)


def test_bank_grader_on_constructed_responses():
    bank = tc.build_bank()
    for case in bank["cases"]:
        if case["expected"].get("no_call"):
            good = "I need one more detail: which value should I use?" if case["expected"].get("asks") else "Here is a plain answer."
        else:
            good = _native(list(reversed(case["expected"]["calls"])))
        assert tc.grade_case(case, good + "<|im_end|>")["passed"], case["id"]
    single = next(c for c in bank["cases"] if c["lane"] == "exec_single")
    wrong = json.loads(json.dumps(single["expected"]["calls"]))
    key = next(iter(wrong[0]["arguments"]))
    value = wrong[0]["arguments"][key]
    wrong[0]["arguments"][key] = value + 7 if isinstance(value, (int, float)) else value + "x"
    assert not tc.grade_case(single, _native(wrong))["passed"]
    par = next(c for c in bank["cases"] if c["lane"] == "exec_parallel")
    assert not tc.grade_case(par, _native(par["expected"]["calls"][:1]))["passed"]
    assert not tc.grade_case(next(c for c in bank["cases"] if c["lane"] == "no_tool"), _native(single["expected"]["calls"]))["passed"]
    clar = next(c for c in bank["cases"] if c["lane"] == "clarify")
    assert not tc.grade_case(clar, "Sure, done.")["passed"]


# --------------------------------------------------------------------------
# End to end against a scripted llama-server
# --------------------------------------------------------------------------

_SANDBOX_ARGS = {"calculate": {"expression": "17*6"}, "convert_units": {"value": 5, "from_unit": "km", "to_unit": "mi"},
                 "date_add": {"date": "2026-10-14", "days": 10}, "days_between": {"start_date": "2026-10-01", "end_date": "2026-10-14"},
                 "currency_convert": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"},
                 "text_statistics": {"text": "One two three."}, "split_bill": {"total": 120, "tip_percent": 18, "people": 4},
                 "percent_change": {"old_value": 80, "new_value": 100}}


def _fill(props, required):
    args = {}
    for key in required:
        prop = props[key]
        kind = tc.first_type(prop)
        if prop.get("enum"):
            args[key] = prop["enum"][0]
        else:
            args[key] = {"string": f"v-{key}", "integer": 3, "number": 2.5, "boolean": True, "array": ["a"], "object": {}}[kind]
    return args


class Scripted:
    """Answers each role from the decoded prompt; counts requests (to kill a run part-way)."""

    def __init__(self, renderer):
        self.renderer = renderer
        self.counter = 0
        self.requests = 0
        self.fail_after = None
        self.lock = threading.Lock()

    def respond(self, prompt_ids):
        with self.lock:
            self.requests += 1
            if self.fail_after is not None and self.requests > self.fail_after:
                return None
            self.counter += 1
            n = self.counter
        text = self.renderer.decode(prompt_ids)
        if "senior API designer" in text:
            body = [{"name": f"get_record_{n}", "description": "Fetch one record by id.", "parameters": {"type": "object", "properties": {
                "record_id": {"type": "string", "description": "id"}, "mode": {"type": "string", "enum": ["brief", "full"], "description": "m"}}, "required": ["record_id"]}},
                    {"name": f"update_record_{n}", "description": "Update a record.", "parameters": {"type": "object", "properties": {
                        "record_id": {"type": "string", "description": "id"}, "count": {"type": "integer", "description": "c"}}, "required": ["record_id", "count"]}},
                    {"name": f"find_flight_{n}", "description": "Find flights.", "parameters": {"type": "object", "properties": {}, "required": []}}]
            return json.dumps(body)
        if "training-data filter" in text:  # 1.1.0 judge
            return '{"fits": false, "claims": false}'
        if "You simulate API backends" in text:
            return json.dumps({"ok": True, "record": {"id": "r-1", "state": "active"}})
        if "You write the next message" in text:
            if "just asked the person" in text:
                return f"VALUE use 42 please #{n}"
            if "follow-up request" in text:
                return f"FOLLOW now do one more #{n}"
            if "does not give the required detail" in text:
                return f"CLARIFY handle my request #{n}"
            if "general knowledge" in text or "none of these functions" in text:
                return f"NOTOOL tell me something #{n}"
            if "at once" in text:
                return f"PARALLEL do several things #{n}"
            if "two steps" in text:
                return f"SEQ do it in two steps #{n}"
            return f"SINGLE do one thing #{n}"
        tools = [json.loads(line) for line in re.search(r"<tools>\n(.*?)\n</tools>", text, re.S).group(1).split("\n")]
        users = [u for u in re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.S) if not u.strip().startswith("<tool_response>")]
        last_user = users[-1]
        tail = text[text.rfind(last_user):]
        calls_so_far = tail.count("<tool_call>")
        responses = re.findall(r"<tool_response>\n(.*?)\n</tool_response>", tail, re.S)
        sandbox = [t for t in tools if t["function"]["name"] in tc.SANDBOX_TOOLS]
        pick = (sandbox or tools)[0]["function"]
        args = _SANDBOX_ARGS.get(pick["name"]) or _fill(pick["parameters"]["properties"], pick["parameters"].get("required", []))

        def call(a):
            return _native([{"name": pick["name"], "arguments": a}])
        if last_user.startswith("NOTOOL"):
            if int(last_user.rsplit("#", 1)[1]) % 5 == 0:  # 1.1.0: some first answers claim a lookup -> retried
                return "I checked your account in the system: it is active."
            return "Here is a general explanation without any lookup."
        if last_user.startswith("CLARIFY"):
            return "Could you tell me which value to use?"
        if calls_so_far == 0:
            if last_user.startswith("PARALLEL"):
                return call(args) + "\n" + call(args)
            return call(args)
        if responses and '"error"' in responses[-1] and calls_so_far < 3:
            return call(args)
        if last_user.startswith("SEQ") and calls_so_far == 1:
            return call(args)
        return "Results: " + " | ".join(responses)


def test_http_4xx_rejects_the_sample_and_keeps_the_server(renderer):
    """1.0.1: an HTTPError is a URLError; a 4xx must not be mistaken for a dead server."""
    class Refuse(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(400)
            self.send_header("Content-Length", "0")
            self.end_headers()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Refuse)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = tc.Client(f"http://127.0.0.1:{server.server_address[1]}", renderer, patience_s=600.0)
        with pytest.raises(tc._Reject):
            client.generate([1, 2, 3], "assistant", 7)
    finally:
        server.shutdown()


class _Handler(BaseHTTPRequestHandler):
    script: Scripted = None

    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        blob = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        if self.path == "/props":
            self._send(200, {"model_path": None, "total_slots": 4, "build_info": "stub"})
        else:
            self._send(200, {"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = self.script.respond(body["prompt"])
        if text is None:
            self.close_connection = True
            self.connection.shutdown(2)
            return
        ids = self.script.renderer.encode(text) + [tc.EOG_IDS[0]]
        self._send(200, {"tokens": ids, "content": text, "stop_type": "eos"})


@pytest.fixture()
def stub(renderer):
    script = Scripted(renderer)
    handler = type("H", (_Handler,), {"script": script})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield script, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@needs_parent
def test_end_to_end_stub_run_finalize_export(stub, tmp_path, monkeypatch):
    script, url = stub
    monkeypatch.setattr(tc, "DOMAINS", tc.DOMAINS[:4])
    assert tc.main(["schemas", "--server", url, "--out-dir", str(tmp_path / "schemas"), "--per-domain", "2", "--workers-per-server", "2"]) == 0
    library = tc.read_jsonl(tmp_path / "schemas" / "schemas.jsonl")
    synthesized = [r for r in library if r["source"] == "parent-synthesized"]
    assert len(synthesized) == 8
    assert all(not any(t["function"]["name"].startswith("find_flight") for t in r["tools"]) for r in synthesized)  # held-out family refused
    plan = tmp_path / "plan.jsonl"
    assert tc.main(["plan", "--schemas", str(tmp_path / "schemas" / "schemas.jsonl"), "--out", str(plan), "--items", "60"]) == 0
    items = tc.read_jsonl(plan)
    assert {i["category"] for i in items} == set(tc.CATEGORY_WEIGHTS)
    bank = tmp_path / "bank.json"
    assert tc.main(["bank-build", "--out", str(bank)]) == 0
    responses = tmp_path / "bank.responses.jsonl"
    assert tc.main(["bank-run", "--bank", str(bank), "--server", url, "--server", url, "--out", str(responses)]) == 0  # 1.0.2: parallel
    assert [r["id"] for r in tc.read_jsonl(responses)] == [c["id"] for c in tc.read_json(bank)["cases"]]
    assert tc.main(["bank-grade", "--bank", str(bank), "--responses", str(responses), "--label", "stub", "--out", str(tmp_path / "g.json")]) == 0
    run_dir = tmp_path / "run"
    common = ["run", "--plan", str(plan), "--server", url, "--out-dir", str(run_dir), "--workers-per-server", "3", "--progress-every", "0"]
    monkeypatch.setattr(tc, "CLIENT_PATIENCE_S", 1.0)
    script.fail_after = script.requests + 40
    rc = tc.main(common + ["--token-target", "40000"])
    assert rc == 2  # the peer went away part-way: resumable
    journaled = tc.read_jsonl(run_dir / "journal.jsonl")
    script.fail_after = None
    real_grounded, fired = tc.grounded, []

    def flaky_grounded(answer, values):  # 1.1.0: one internal failure must reject an attempt, not kill a worker
        if not fired:
            fired.append(1)
            raise RuntimeError("injected internal failure")
        return real_grounded(answer, values)
    monkeypatch.setattr(tc, "grounded", flaky_grounded)
    assert tc.main(common + ["--token-target", "40000"]) == 0
    monkeypatch.setattr(tc, "grounded", real_grounded)
    raw = tc.read_jsonl(run_dir / "journal.jsonl")
    internal = [t for r in raw for t in r.get("rejections", []) if t["reason"].startswith("internal: RuntimeError")]
    assert len(internal) == 1 and "flaky_grounded" in internal[0]["traceback"]
    assert tc.main(common + ["--token-target", "41000"]) == 1  # changed semantics refused
    # 1.1.0: content rule.  Rows the stub's claiming replies produced were retried, never accepted.
    journal = list(tc.load_journal(run_dir / "journal.jsonl").values())
    zero = [r for r in journal if r["status"] == "accepted" and r["category"] in tc.ZERO_CALL_CATEGORIES]
    assert zero and all(r["content_check"]["rules"] == tc.CONTENT_RULES and not r["content_check"]["failed"] for r in zero)
    assert any(t["reason"].startswith("content: claims an action") for r in journal for t in r.get("rejections", []))
    # A 1.0.x journal: strip the verdicts, make one accepted answer claim a lookup; resume re-checks.
    victim = zero[0]
    legacy = []
    for r in tc.read_jsonl(run_dir / "journal.jsonl"):
        r.pop("content_check", None)
        if r["index"] == victim["index"] and r["status"] == "accepted":
            r["messages"][-1]["content"] = "I have checked the system and your order is confirmed."
        legacy.append(r)
    tc.write_jsonl(run_dir / "journal.jsonl", legacy)
    assert tc.main(["finalize", "--out-dir", str(run_dir), "--bank", str(bank)]) == 1  # no verdicts: resume first
    assert tc.main(common + ["--token-target", "40000"]) in (0, 3)
    raw = tc.read_jsonl(run_dir / "journal.jsonl")
    history = [r for r in raw if r["index"] == victim["index"]]
    reopened = [r for r in history if r["status"] == "reopened"]
    assert len(reopened) == 1 and reopened[0]["rejections"][-1]["reason"].startswith("content: claims an action")
    assert history[-1]["status"] in ("accepted", "rejected") and history.index(reopened[0]) < len(history) - 1
    if history[-1]["status"] == "accepted":
        assert history[-1]["attempt"] > victim["attempt"]
    journal = list(tc.load_journal(run_dir / "journal.jsonl").values())
    assert not tc.stale_zero_call_rows({r["index"]: r for r in journal})
    assert len({r["index"] for r in journal}) == len(journal) >= len(journaled)
    accepted = [r for r in journal if r["status"] == "accepted"]
    assert {r["category"] for r in accepted} == set(tc.CATEGORY_WEIGHTS), Counter_of(journal)
    err = next(r for r in accepted if r["category"] == "error")
    assert '"error"' in next(m["content"] for m in err["messages"] if m["role"] == "tool")
    assert tc.main(["finalize", "--out-dir", str(run_dir), "--bank", str(bank)]) == 0
    manifest = tc.read_json(run_dir / "manifest.json")
    assert tc.check_seal(manifest, "manifest_sha256") and manifest["tokens"] >= 40000 - 9000
    assert manifest["screen"]["added_sources"][1]["name"] == "toolcall_bank_v1"
    assert manifest["content_rule"]["rules"] == tc.CONTENT_RULES and manifest["internal_failure_attempts"] >= 0
    if victim["index"] <= manifest["cutoff_plan_index"]:
        assert victim["id"] in manifest["content_rule"]["reopened_rows"]
    # finalize refuses an unresolved reopened row
    import shutil
    run2 = tmp_path / "run2"
    shutil.copytree(run_dir, run2)
    tc.append_journal(run2 / "journal.jsonl", {"index": victim["index"], "id": victim["id"], "status": "reopened", "attempt": 0,
                                               "category": victim["category"], "domain": victim["domain"], "rejections": []})
    assert tc.main(["finalize", "--out-dir", str(run2), "--bank", str(bank)]) == 1
    dest = tmp_path / "corpus" / "toolcall.safetensors"
    assert tc.main(["export", "--out-dir", str(run_dir), "--dest", str(dest)]) == 0
    import qtext_teacher_cache as qt
    import torch
    loaded = qt.load_text_corpus(dest)
    sidecar = tc.read_json(dest.with_suffix(".rows.json"))
    assert loaded["rows"] == sidecar["rows"] == manifest["conversations"]
    assert loaded["mask_source"].startswith("target_mask shifted by one")
    from safetensors.torch import load_file
    raw = load_file(str(dest))
    target = raw["target_mask"].to(torch.bool)
    store = loaded["store_mask"]
    assert torch.equal(store[:, :-1], target[:, 1:]) and int(store.sum()) == int(target.sum()) == sidecar["supervised_tokens"]
    assert int(raw["input_ids"].max()) < tc.TEXT_VOCAB_ROWS
    assert sidecar["source_manifest_sha256"] == manifest["manifest_sha256"] and len(sidecar["source_manifests"]) == 1

    # 1.2.0: a continuation run.  Its own first run (cut early in an 80-item plan), then the plan tail
    # after the sealed cutoff under a new seed range, sealed and exported with it.
    plan_a = tmp_path / "planA.jsonl"
    assert tc.main(["plan", "--schemas", str(tmp_path / "schemas" / "schemas.jsonl"), "--out", str(plan_a), "--items", "80"]) == 0
    run_a = tmp_path / "runA"
    assert tc.main(["run", "--plan", str(plan_a), "--server", url, "--out-dir", str(run_a), "--workers-per-server", "3",
                    "--progress-every", "0", "--token-target", "5000"]) == 0
    assert tc.main(["finalize", "--out-dir", str(run_a), "--bank", str(bank)]) == 0
    manifest_a = tc.read_json(run_a / "manifest.json")
    cutoff = manifest_a["cutoff_plan_index"]
    assert cutoff < 40, cutoff
    plan2 = tmp_path / "plan2.jsonl"
    cont = ["continue-plan", "--plan", str(plan_a), "--after-manifest", str(run_a / "manifest.json"), "--out", str(plan2)]
    assert tc.main(cont) == 0
    tail = tc.read_jsonl(plan2)
    assert [t["source_index"] for t in tail] == list(range(cutoff + 1, 80)) and [t["index"] for t in tail] == list(range(len(tail)))
    assert {t["id"] for t in tail}.isdisjoint({r["id"] for r in tc.read_jsonl(Path(manifest_a["file"]["path"]))})
    plan2_manifest = tc.read_json(plan2.with_suffix(".manifest.json"))
    assert tc.check_seal(plan2_manifest, "manifest_sha256") and plan2_manifest["continues"]["after_run"]["manifest_sha256"] == manifest_a["manifest_sha256"]
    assert tc.main(cont) == 0  # same bytes: idempotent
    assert tc.main(["continue-plan", "--plan", str(plan), "--after-manifest", str(run_a / "manifest.json"), "--out", str(tmp_path / "x.jsonl")]) == 1
    tc.write_jsonl(plan2, tail[:-1])
    assert tc.main(cont) == 1  # frozen once written
    tc.write_jsonl(plan2, tail)
    run3 = tmp_path / "run3"
    common3 = ["run", "--plan", str(plan2), "--server", url, "--out-dir", str(run3), "--workers-per-server", "3", "--progress-every", "0",
               "--base-seed", "20261015", "--token-target", "1000000"]
    # A deadline already past: nothing is dispatched, exit 4.
    assert tc.main(common3 + ["--stop-by", "2020-01-01T00:00:00Z"]) == 4
    assert not (run3 / "journal.jsonl").exists() or not tc.read_jsonl(run3 / "journal.jsonl")
    # The deadline passes part-way (its own clock moves; the client's patience clock does not).
    real_now, start_requests = tc.deadline_now, script.requests
    monkeypatch.setattr(tc, "deadline_now", lambda: real_now() + (10 ** 9 if script.requests > start_requests + 25 else 0))
    assert tc.main(common3 + ["--stop-by", "2030-01-01T00:00:00Z"]) == 4
    journal3 = list(tc.load_journal(run3 / "journal.jsonl").values())
    assert len(journal3) >= 2 and sorted(r["index"] for r in journal3) == list(range(len(journal3))) and len(journal3) < len(tail)  # no gap
    assert tc.read_json(run3 / "run_config.json")["base_seed"] == 20261015
    assert tc.read_json(run3 / "progress.json")["stop_by"] == "2030-01-01T00:00:00Z"
    # A reopened row in hand after the deadline is journaled rejected (reason deadline), not left as a gap.
    victim3 = journal3[0]
    tc.append_journal(run3 / "journal.jsonl", {"index": victim3["index"], "id": victim3["id"], "status": "reopened", "attempt": 0,
                                               "category": victim3["category"], "domain": victim3["domain"], "rejections": []})
    assert tc.main(common3 + ["--stop-by", "2030-01-01T00:00:00Z"]) == 4
    last = tc.load_journal(run3 / "journal.jsonl")[victim3["index"]]
    assert last["status"] == "rejected" and last["rejections"][-1]["reason"].startswith("deadline") and last["rejections"][-1]["model_calls"] == 0
    monkeypatch.setattr(tc, "deadline_now", real_now)
    assert tc.main(["finalize", "--out-dir", str(run3), "--bank", str(bank)]) == 1  # short: needs --accept-short
    # a run named as its own predecessor: its ids recur, refused, nothing written
    assert tc.main(["finalize", "--out-dir", str(run_a), "--bank", str(bank), "--prior-manifest", str(run_a / "manifest.json")]) == 1
    assert tc.read_json(run_a / "manifest.json") == manifest_a
    fin3 = ["finalize", "--out-dir", str(run3), "--bank", str(bank), "--accept-short", "--prior-manifest", str(run_a / "manifest.json")]
    assert tc.main(fin3) == 0
    manifest3 = tc.read_json(run3 / "manifest.json")
    assert manifest3["short_of_target"] and manifest3["prior_manifests"][0]["manifest_sha256"] == manifest_a["manifest_sha256"]
    assert manifest3["deadline_rejected_items"] >= 1 and manifest3["cut_at_gap"] is None and manifest3["conversations"] >= 1
    assert manifest3["attempt_rejections_by_reason"].get("deadline", 0) >= 1
    # --cut-at-gap: only with --accept-short, then the prefix before the gap
    run4 = tmp_path / "run4"
    shutil.copytree(run3, run4)
    tc.write_jsonl(run4 / "journal.jsonl", [r for r in tc.read_jsonl(run4 / "journal.jsonl") if r["index"] != 1])
    assert tc.main(["finalize", "--out-dir", str(run4), "--bank", str(bank), "--accept-short"]) == 1
    assert tc.main(["finalize", "--out-dir", str(run4), "--bank", str(bank), "--cut-at-gap"]) == 1
    assert tc.main(["finalize", "--out-dir", str(run4), "--bank", str(bank), "--accept-short", "--cut-at-gap"]) == 0
    manifest4 = tc.read_json(run4 / "manifest.json")
    assert manifest4["cut_at_gap"] == 1 and manifest4["cutoff_plan_index"] == 0
    # one corpus over both runs
    one = tmp_path / "corpus" / "toolcall-a.safetensors"
    assert tc.main(["export", "--out-dir", str(run_a), "--dest", str(one)]) == 0
    side1 = tc.read_json(one.with_suffix(".rows.json"))
    both = tmp_path / "corpus" / "toolcall-both.safetensors"
    assert tc.main(["export", "--out-dir", str(run_a), "--out-dir", str(run3), "--dest", str(both)]) == 0
    side2 = tc.read_json(both.with_suffix(".rows.json"))
    n_a = manifest_a["conversations"]
    assert tc.check_seal(side2, "receipt_sha256") and side2["rows"] == n_a + manifest3["conversations"]
    assert [s["row_range"] for s in side2["source_manifests"]] == [[0, n_a], [n_a, side2["rows"]]]
    assert side2["source_manifest_sha256"] == manifest_a["manifest_sha256"] + "+" + manifest3["manifest_sha256"]
    assert side2["row_ids"][:n_a] == side1["row_ids"] and len(set(side2["row_ids"])) == side2["rows"]
    assert side2["corpus"]["sha256"] == tc.file_sha256(both)  # what post_d1_queue's corpus_ready greps for
    loaded2 = qt.load_text_corpus(both)
    assert loaded2["rows"] == side2["rows"] and int(loaded2["store_mask"].sum()) == side2["supervised_tokens"]
    lengths2 = load_file(str(both))["lengths"]
    assert side2["tokens"] == side1["tokens"] + int(lengths2[n_a:].sum()) == manifest_a["tokens"] + manifest3["tokens"]
    assert tc.main(["export", "--out-dir", str(run3), "--out-dir", str(run3), "--dest", str(tmp_path / "dup.safetensors")]) == 1


def Counter_of(journal):
    from collections import Counter
    return Counter((r["category"], r["status"]) for r in journal)
