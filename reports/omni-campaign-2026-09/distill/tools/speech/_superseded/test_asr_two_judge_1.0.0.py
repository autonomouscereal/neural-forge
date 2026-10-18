"""test_asr_two_judge.py -- asr_two_judge 1.0.0: the plan v2 2.3.3 agreement rule, the C-s3 admission rule, the
normaliser, result assembly, and the real orchestration path (subprocess per judge, A then B, card hiding, failures)
against a stand-in runtime that answers the judge CLI.  CPU only (conftest.py hides every card); no model is loaded --
the model path is proven by the CPU smoke (`asr_two_judge.py smoke`), whose output the report quotes.

    cd /home/cereal/omni-campaign/distill/tools/speech/tests
    nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -p no:cacheprovider -q test_asr_two_judge.py
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SPEECH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SPEECH))
import asr_two_judge as atj  # noqa: E402

TWENTY = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"


# --------------------------------------------------------------------------------------------------------------------
# normaliser and WER
# --------------------------------------------------------------------------------------------------------------------


def test_normaliser_folds_case_punctuation_quotes_and_hyphens():
    assert atj.normalise_words("Well-known, isn’t it?  YES.") == ["well", "known", "isn't", "it", "yes"]
    assert atj.normalise_words("") == [] and atj.normalise_words(None) == []


def test_normaliser_keeps_other_scripts_and_splits_cjk():
    assert atj.normalise_words("Grüße, Señor") == ["grüsse", "señor"]   # casefold: ß -> ss
    assert atj.normalise_words("今日は晴れ ok") == ["今", "日", "は", "晴", "れ", "ok"]


def test_numbers_are_not_normalised():
    assert atj.normalise_words("20 apples") == ["20", "apples"]
    assert atj.agreement("I have twenty apples", "I have 20 apples")["transcript_status"] == "review"


def test_wer_basic():
    assert atj.wer("a b c d", "a b x d") == 0.25
    assert atj.wer("", "") == 0.0 and atj.wer("", "x") == 1.0


# --------------------------------------------------------------------------------------------------------------------
# agreement (plan 2.3.3)
# --------------------------------------------------------------------------------------------------------------------


def test_identical_and_formatting_only_differences_are_auto():
    v = atj.agreement("The river rose overnight.", "the river rose overnight")
    assert (v["transcript_status"], v["transcript_conf"], v["agreement_wer"], v["reasons"]) == ("auto", 1.0, 0.0, [])


def test_five_percent_boundary():
    def swap(text, i):  # replace word i only ("seven" is also inside "seventeen")
        w = text.split()
        w[i] = "heaven"
        return " ".join(w)

    v = atj.agreement(TWENTY, swap(TWENTY, 6))
    assert v["agreement_wer"] == 0.05 and v["transcript_status"] == "auto" and v["transcript_conf"] == 0.95
    nineteen = " ".join(TWENTY.split()[:19])
    v = atj.agreement(nineteen, swap(nineteen, 6))
    assert v["agreement_wer"] > 0.05 and v["transcript_status"] == "review" and v["reasons"] == ["judges_disagree"]


def test_verdict_does_not_depend_on_judge_order():
    a, b = TWENTY, " ".join(TWENTY.split()[:-1])   # one deletion: 1/20 one way, 1/19 the other
    va, vb = atj.agreement(a, b), atj.agreement(b, a)
    assert va == {**vb, "words_a": va["words_a"], "words_b": va["words_b"]}
    assert va["agreement_wer"] == round(1 / 19, 4) and va["transcript_status"] == "review"   # stricter than 1/20


def test_empty_or_failed_judges_go_to_review():
    assert atj.agreement("", "")["reasons"] == ["empty_transcript"]
    assert atj.agreement("hello there", "")["transcript_status"] == "review"
    v = atj.agreement(None, "hello there")
    assert v["transcript_status"] == "review" and v["reasons"] == ["judge_a_error"] and v["transcript_conf"] == 0.0
    assert atj.agreement("hello", None)["reasons"] == ["judge_b_error"]


# --------------------------------------------------------------------------------------------------------------------
# admission (C-s3)
# --------------------------------------------------------------------------------------------------------------------


def test_admission_needs_both_judges_exact():
    ref = "Please set the oven to two hundred degrees."
    assert atj.admission(ref, "please set the oven to two hundred degrees", "Please set the oven to two hundred degrees!")["verdict"] == "admitted"
    one_off = atj.admission(ref, "please set the oven to two hundred degrees", "please set the oven to 200 degrees")
    assert one_off["verdict"] == "quarantine" and one_off["judge_a_exact"] and not one_off["judge_b_exact"]
    assert one_off["judge_b_wer"] == round(2 / 8, 4)
    assert atj.admission(ref, None, ref)["verdict"] == "quarantine"
    assert atj.admission("", "", "")["verdict"] == "quarantine"


# --------------------------------------------------------------------------------------------------------------------
# assembly and orchestration
# --------------------------------------------------------------------------------------------------------------------


def test_combine_keeps_order_and_judge_a_text():
    meta_a = {"repo": atj.JUDGES["A"]["repo"], "revision": atj.JUDGES["A"]["revision"]}
    meta_b = {"repo": atj.JUDGES["B"]["repo"], "revision": atj.JUDGES["B"]["revision"]}
    rows_a = {"/x/2.flac": {"text": "Second clip."}, "/x/1.flac": {"text": "First clip here."}}
    rows_b = {"/x/1.flac": {"text": "first clip here"}, "/x/2.flac": {"text": None, "error": "RuntimeError('bad')"}}
    out = atj.combine(["/x/1.flac", "/x/2.flac"], rows_a, rows_b, meta_a, meta_b)
    assert [r["audio"] for r in out] == ["/x/1.flac", "/x/2.flac"]
    assert out[0]["transcript"] == "First clip here." and out[0]["transcript_status"] == "auto"
    assert out[1]["transcript_status"] == "review" and out[1]["reasons"] == ["judge_b_error"]
    assert out[1]["judge_b"]["error"] == "RuntimeError('bad')"
    tool = out[0]["transcript_tool"]
    assert "Qwen/Qwen3-ASR-0.6B@5eb14417" in tool and "ibm-granite/granite-speech-4.1-2b@de575db6" in tool


FAKE_JUDGE = r'''
import json, os, sys
args = sys.argv[1:]
assert args[0] == "-I" and args[1].endswith("asr_two_judge.py") and args[2] == "judge", args
opt = {args[i]: args[i + 1] for i in range(3, len(args), 2)}
log = os.environ["FAKE_LOG"]
with open(log, "a") as h:
    h.write(json.dumps({"judge": opt["--judge"], "device": opt["--device"], "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "offline": os.environ.get("HF_HUB_OFFLINE")}) + "\n")
if os.environ.get("FAKE_FAIL") == opt["--judge"]:
    sys.exit(7)
texts = json.loads(os.environ["FAKE_TEXTS"])[opt["--judge"]]
rows = [{"id": r["id"], "audio": r["audio"], "text": texts.get(os.path.basename(r["audio"])), "seconds": 0.1}
        for r in json.load(open(opt["--manifest"]))]
json.dump({"repo": "fake-" + opt["--judge"], "revision": "f" * 40, "rows": rows}, open(opt["--out"], "w"))
'''


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    script = tmp_path / "fake_judge.py"
    script.write_text(FAKE_JUDGE)
    runtime = tmp_path / "python"
    runtime.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    runtime.chmod(runtime.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_TEXTS", json.dumps({
        "A": {"a.flac": "The cat sat on the mat.", "b.flac": "A dog barked twice at night."},
        "B": {"a.flac": "the cat sat on the mat", "b.flac": "a frog parked twice at night"}}))
    return runtime, log


@pytest.mark.skipif(os.name == "nt", reason="the stand-in runtime is a POSIX shell script")
def test_transcribe_two_judges_runs_a_then_b_hidden_on_cpu(fake_runtime):
    runtime, log = fake_runtime
    out = atj.transcribe_two_judges(["/d/a.flac", "/d/b.flac"], runtime_python=str(runtime), threads=2)
    calls = [json.loads(l) for l in log.read_text().splitlines()]
    assert [c["judge"] for c in calls] == ["A", "B"]
    assert all(c["cuda"] == "" and c["device"] == "cpu" and c["offline"] == "1" for c in calls)
    assert [r["transcript_status"] for r in out] == ["auto", "review"]
    assert out[0]["transcript"] == "The cat sat on the mat." and out[1]["judge_b"]["text"] == "a frog parked twice at night"


@pytest.mark.skipif(os.name == "nt", reason="the stand-in runtime is a POSIX shell script")
def test_transcribe_two_judges_names_the_claimed_p40(fake_runtime):
    runtime, log = fake_runtime
    atj.transcribe_two_judges(["/d/a.flac"], gpu=3, runtime_python=str(runtime), threads=2)
    calls = [json.loads(l) for l in log.read_text().splitlines()]
    assert all(c["cuda"] == "3" and c["device"] == "cuda:0" for c in calls)


@pytest.mark.parametrize("gpu", [0, 4, 5, 6, 7])
def test_non_p40_cards_are_refused_before_any_process(gpu, tmp_path):
    with pytest.raises(ValueError):
        atj.transcribe_two_judges(["/d/a.flac"], gpu=gpu, runtime_python=str(tmp_path / "never-run"))


@pytest.mark.skipif(os.name == "nt", reason="the stand-in runtime is a POSIX shell script")
def test_a_failed_judge_process_raises(fake_runtime, monkeypatch):
    runtime, log = fake_runtime
    monkeypatch.setenv("FAKE_FAIL", "B")
    with pytest.raises(RuntimeError, match="judge B failed"):
        atj.transcribe_two_judges(["/d/a.flac"], runtime_python=str(runtime), threads=2)


def test_empty_input_starts_nothing(tmp_path):
    assert atj.transcribe_two_judges([], runtime_python=str(tmp_path / "never-run")) == []


def test_module_imports_with_the_standard_library_only():
    # -S: no site-packages at all, so a third-party import at module level (torch, numpy, pydantic ...) would fail
    code = f"import sys; sys.path.insert(0, {str(SPEECH)!r}); import asr_two_judge; print(asr_two_judge.VERSION)"
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == atj.VERSION


def test_verdict_cli(capsys):
    assert atj.main(["verdict", "--a", "hello world", "--b", "Hello, world!", "--reference", "hello world"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["agreement"]["transcript_status"] == "auto" and out["admission"]["verdict"] == "admitted"


def test_pinned_judges_are_distinct_vendors_with_full_revisions():
    assert atj.JUDGES["A"]["repo"].split("/")[0] != atj.JUDGES["B"]["repo"].split("/")[0]
    assert all(len(j["revision"]) == 40 for j in atj.JUDGES.values())
    assert atj.JUDGES["B"]["repo"] == "ibm-granite/granite-speech-4.1-2b"
