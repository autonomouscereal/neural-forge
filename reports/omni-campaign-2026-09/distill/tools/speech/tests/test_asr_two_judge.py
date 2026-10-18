"""test_asr_two_judge.py -- asr_two_judge 1.1.0: the plan v2 2.3.3 agreement rule, the C-s3 admission rule, the
normaliser, result assembly, and the real orchestration path (subprocess per judge, A then B, card hiding, failures)
against a stand-in runtime that answers the judge CLI.  CPU only (conftest.py hides every card); no model is loaded --
the model path is proven by the CPU smoke (`asr_two_judge.py smoke`), whose output the report quotes.
1.1.0 adds card selection (review must-fix 2) against the AI server's real nvidia-smi listing (SERVER_LISTING): the
claimed index becomes its UUID with CUDA_DEVICE_ORDER=PCI_BUS_ID, a non-P40 index is refused before any process, and
the judge-side check refuses a wrong / extra / differently named card before loading a model.

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

# `nvidia-smi --query-gpu=index,name,uuid,pci.bus_id --format=csv,noheader` on the AI server, 2026-10-15 (nvidia-smi
# numbers by PCI bus; torch's default order puts the 3090s and the RTX 8000 first)
SERVER_LISTING = """0, NVIDIA GeForce RTX 3090, GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561, 00000000:04:00.0
1, Tesla P40, GPU-c95fb5b3-f97c-bfb2-9760-79fe107ffe6d, 00000000:05:00.0
2, Tesla P40, GPU-a92b0dfe-83f8-11ff-db0d-c65921f365b4, 00000000:08:00.0
3, Tesla P40, GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8, 00000000:09:00.0
4, NVIDIA GeForce RTX 3090, GPU-2ce6a95c-7b1f-644a-3808-d781732a0582, 00000000:84:00.0
5, Tesla P40, GPU-ce4a1da0-baa8-3d62-419c-a28f6565610f, 00000000:85:00.0
6, Quadro RTX 8000, GPU-f5a09535-fc85-a027-50f7-734b344747dd, 00000000:88:00.0
"""
GPU3_UUID = "GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8"

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
                        "order": os.environ.get("CUDA_DEVICE_ORDER"), "expect": os.environ.get("ASR2J_EXPECT_UUID"),
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
def test_transcribe_two_judges_names_the_claimed_p40_by_uuid(fake_runtime, monkeypatch):
    runtime, log = fake_runtime
    monkeypatch.setattr(atj, "query_gpus", lambda: atj.parse_gpu_listing(SERVER_LISTING))
    atj.transcribe_two_judges(["/d/a.flac"], gpu=3, runtime_python=str(runtime), threads=2)
    calls = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(calls) == 2
    assert all(c["cuda"] == GPU3_UUID and c["expect"] == GPU3_UUID and c["order"] == "PCI_BUS_ID"
               and c["device"] == "cuda:0" for c in calls)


@pytest.mark.skipif(os.name == "nt", reason="the stand-in runtime is a POSIX shell script")
def test_a_listing_whose_index_is_not_a_p40_starts_no_process(fake_runtime, monkeypatch):
    runtime, log = fake_runtime
    swapped = SERVER_LISTING.replace("3, Tesla P40", "3, NVIDIA GeForce RTX 3090")    # torch-style order at index 3
    monkeypatch.setattr(atj, "query_gpus", lambda: atj.parse_gpu_listing(swapped))
    with pytest.raises(ValueError, match="not a Tesla P40"):
        atj.transcribe_two_judges(["/d/a.flac"], gpu=3, runtime_python=str(runtime), threads=2)
    assert not log.exists()


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


# --------------------------------------------------------------------------------------------------------------------
# card selection (1.1.0, review must-fix 2)
# --------------------------------------------------------------------------------------------------------------------


def test_server_listing_resolves_each_p40_to_its_uuid():
    cards = atj.parse_gpu_listing(SERVER_LISTING)
    assert [c["index"] for c in cards] == list(range(7))
    want = {1: "GPU-c95fb5b3-f97c-bfb2-9760-79fe107ffe6d", 2: "GPU-a92b0dfe-83f8-11ff-db0d-c65921f365b4", 3: GPU3_UUID}
    for gpu, uuid in want.items():
        card = atj.card_for(gpu, cards)
        assert (card["uuid"], card["name"]) == (uuid, "Tesla P40")
        env = atj.judge_env(gpu, 4, card)
        assert env["CUDA_VISIBLE_DEVICES"] == uuid and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert env[atj.EXPECT_UUID_ENV] == uuid and env[atj.EXPECT_NAME_ENV] == "Tesla P40"


@pytest.mark.parametrize("gpu", [0, 4, 5, 6, 7])
def test_card_for_refuses_every_card_but_p40s_1_to_3(gpu):
    with pytest.raises(ValueError):
        atj.card_for(gpu, atj.parse_gpu_listing(SERVER_LISTING))      # GPU5 is a P40 too: refused by rule


def test_card_for_refuses_an_index_that_is_not_a_p40_or_is_missing():
    torch_order = SERVER_LISTING.replace("1, Tesla P40", "1, NVIDIA GeForce RTX 3090")
    with pytest.raises(ValueError, match="not a Tesla P40"):
        atj.card_for(1, atj.parse_gpu_listing(torch_order))
    with pytest.raises(ValueError, match="0 cards"):
        atj.card_for(2, [c for c in atj.parse_gpu_listing(SERVER_LISTING) if c["index"] != 2])
    no_uuid = SERVER_LISTING.replace("GPU-a92b0dfe", "MIG-a92b0dfe")
    with pytest.raises(ValueError, match="UUID"):
        atj.card_for(2, atj.parse_gpu_listing(no_uuid))


def test_judge_env_never_names_a_card_by_index():
    env = atj.judge_env(None, 4)
    assert env["CUDA_VISIBLE_DEVICES"] == "" and atj.EXPECT_UUID_ENV not in env
    with pytest.raises(ValueError):
        atj.judge_env(3, 4)                                            # no card record: refused, never "3"
    with pytest.raises(ValueError):
        atj.judge_env(3, 4, atj.card_for(2, atj.parse_gpu_listing(SERVER_LISTING)))   # record of another index


def test_query_gpus_runs_nvidia_smi_and_parses():
    seen = {}

    def runner(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=SERVER_LISTING, stderr="")

    cards = atj.query_gpus(runner)
    assert seen["cmd"][0] == "nvidia-smi" and "--query-gpu=index,name,uuid,pci.bus_id" in seen["cmd"]
    assert atj.card_for(3, cards)["pci_bus_id"] == "00000000:09:00.0"
    with pytest.raises(RuntimeError):
        atj.query_gpus(lambda cmd, **kw: subprocess.CompletedProcess(cmd, 9, stdout="", stderr="NVML error"))


class FakeTorch:
    """The few torch.cuda calls check_visible_card makes."""

    def __init__(self, names, uuid=None):
        names_ = list(names)

        class Props:
            pass

        props = Props()
        if uuid is not None:
            props.uuid = uuid

        class Cuda:
            @staticmethod
            def device_count():
                return len(names_)

            @staticmethod
            def get_device_name(i):
                return names_[i]

            @staticmethod
            def get_device_properties(i):
                return props

        self.cuda = Cuda()


def test_judge_side_check_refuses_anything_but_the_handed_p40(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU3_UUID)
    assert atj.check_visible_card(FakeTorch(["Tesla P40"], "08b615f9-8dbe-3edd-d59f-f6ebe8212ca8"), GPU3_UUID) is None
    assert atj.check_visible_card(FakeTorch(["Tesla P40"]), GPU3_UUID) is None                 # torch without .uuid
    assert "not a Tesla P40" in atj.check_visible_card(FakeTorch(["NVIDIA GeForce RTX 3090"]), GPU3_UUID)
    assert "2 CUDA devices" in atj.check_visible_card(FakeTorch(["Tesla P40", "Tesla P40"]), GPU3_UUID)
    assert "UUID" in atj.check_visible_card(FakeTorch(["Tesla P40"], "c95fb5b3-f97c-bfb2-9760-79fe107ffe6d"), GPU3_UUID)
    assert "by index" in atj.check_visible_card(FakeTorch(["Tesla P40"]), "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert "not the handed card" in atj.check_visible_card(FakeTorch(["Tesla P40"]), GPU3_UUID)


def test_judge_cli_refuses_another_cuda_index_before_importing_torch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(atj, "mem_available_gib", lambda: 200.0)
    monkeypatch.setattr(atj, "FORBIDDEN_MODULES", ())                 # this interpreter is not the judge runtime
    monkeypatch.setitem(sys.modules, "librosa", None)                  # an import would raise: the refusal comes first
    rc = atj.main(["judge", "--judge", "A", "--manifest", str(tmp_path / "m.json"), "--out", str(tmp_path / "o.json"),
                   "--device", "cuda:1"])
    assert rc == atj.EXIT_WRONG_CARD and "cuda:0" in capsys.readouterr().err
