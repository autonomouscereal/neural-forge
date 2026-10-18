"""Tests for the CAMPAIGN 1.8 copy stage2_training_receipt.py (the stage-D pre-registration pin).  CPU, < 1 s.

    cd $OC/distill/tools && PYTHONPATH=$OC/distill/tools $PY -m pytest -o addopts="" -p no:cacheprovider \\
        test_stage2_training_receipt.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stage2_training_receipt as S  # noqa: E402

PREREG = HERE.parent / "PREREGISTRATION_OMNI_STAGE_D_v1.md"
# neural-forge's original, when this host has it (the server's runtime copies of the estate)
ORIGINAL = [Path(p) for p in (os.environ.get("NF_STAGE2_RECEIPT", ""),
                              "/home/cereal/omni-campaign/runtime/k3-continuation-oct08-claude/code/src/neural_forge/stage2_training_receipt.py",
                              str(HERE.parents[3] / "src" / "neural_forge" / "stage2_training_receipt.py")) if p]


def test_the_document_beside_the_tools_is_the_pinned_one():
    observed = S.assert_preregistration_unedited(PREREG)
    assert observed["preregistration_sha256"] == S.STAGE2_PREREGISTRATION_SHA256
    assert observed["preregistration_byte_length"] == S.STAGE2_PREREGISTRATION_BYTE_LENGTH
    assert observed["design_document"] == S.STAGE2_DESIGN_FILENAME
    raw = PREREG.read_bytes()
    assert b"\r" not in raw                                   # LF only: the pin is over bytes
    body = raw[raw.index(b"## 1. Scope"): raw.index(b"\n## Amendments") + 1]
    assert hashlib.sha256(body).hexdigest() == S.STAGE2_PREREGISTRATION_SHA256


def test_an_edit_in_the_body_is_refused_an_amendment_is_not(tmp_path):
    raw = PREREG.read_bytes()
    edited = tmp_path / PREREG.name
    edited.write_bytes(raw.replace(b"a = 0.5", b"a = 0.6", 1))
    with pytest.raises(S.Stage2TrainingReceiptError, match="changed since it was sealed"):
        S.assert_preregistration_unedited(edited)
    amended = tmp_path / "amended" / PREREG.name
    amended.parent.mkdir()
    amended.write_bytes(raw.rstrip(b"\n") + b"\n\n- 2026-10-14: an example amendment, below the frozen body.\n")
    observed = S.assert_preregistration_unedited(amended)
    assert observed["design_document_sha256"] != S.STAGE2_DESIGN_DOCUMENT_SHA256   # visible, not refused
    rc = subprocess.run([sys.executable, str(HERE / "stage2_training_receipt.py"), "check-prereg", str(edited)],
                        capture_output=True, text=True)
    assert rc.returncode == 2 and "PREREG REFUSED" in rc.stderr
    rc = subprocess.run([sys.executable, str(HERE / "stage2_training_receipt.py"), "check-prereg", str(PREREG)],
                        capture_output=True, text=True)
    assert rc.returncode == 0 and rc.stdout.startswith("PREREG OK")


def test_canonical_sha256_is_the_estates():
    value = {"b": [1, 2.5, "x"], "a": {"z": None, "y": True}}
    expected = hashlib.sha256(json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"),
                                         sort_keys=True).encode("ascii")).hexdigest()
    assert S.canonical_sha256(value) == expected
    with pytest.raises(S.Stage2TrainingReceiptError):
        S.canonical_sha256({"x": float("nan")})


def _builder(preregistration):
    shard = S.Stage2ShardConsumption(order_index=0, relative_path="shards/shard-00000.tokens.u16", sha256="1" * 64,
                                     tokens_consumed=4096)
    corpus = S.Stage2CorpusConsumption(manifest_path="manifest.json", manifest_sha256="2" * 64,
                                       teacher_checkpoint_sha256="3" * 64, shard_order_seed=0, shards=(shard,))
    rung = S.Stage2RungRecord(rung_tokens=5_000_000, tokens_consumed=4096, optimizer_steps=1, backward_calls=1,
                              micro_batch_forward_calls=1, on_policy_steps=0, off_policy_steps=1, wall_clock_seconds=1.0,
                              checkpoint_manifest_sha256=None, checkpoint_directory=None, optimizer_state={}, free_run_probe=None)
    return S.Stage2TrainingReceiptBuilder(
        arm_id="OMNI1-D2", init_source_sha256="4" * 64, initialization={}, teacher_checkpoint_sha256="3" * 64,
        corpus=corpus, objective_schedule={}, optimizer_configuration={}, trainer_configuration={},
        preregistration=preregistration, device_inventory=({"card": "cpu"},), seeds={}, probe_surface={},
        route_surface_equivalence={}, counted_parameters={}, rungs=[rung])


def test_a_receipt_carries_the_pin_and_round_trips(tmp_path):
    receipt = _builder(S.assert_preregistration_unedited(PREREG)).seal()
    assert receipt["preregistration"]["preregistration_sha256"] == S.STAGE2_PREREGISTRATION_SHA256
    path = tmp_path / "training-receipt.json"
    seal = S.write_stage2_training_receipt(path, receipt)
    assert S.load_stage2_training_receipt(path, expected_receipt_sha256=seal)["receipt_sha256"] == seal
    with pytest.raises(S.Stage2TrainingReceiptError, match="sealed section-8"):   # the original's refusal text
        _builder({"preregistration_sha256": "0" * 64}).seal()
    assert S.STAGE2_RUNG_TOKENS == (5_000_000, 12_000_000, 20_000_000, 23_000_000)


@pytest.mark.skipif(not any(p.is_file() for p in ORIGINAL), reason="neural-forge's original module is not on this host")
def test_the_copy_differs_from_the_original_only_where_its_docstring_says():
    original = next(p for p in ORIGINAL if p.is_file())
    assert hashlib.sha256(original.read_bytes()).hexdigest() in (HERE / "stage2_training_receipt.py").read_text()
    mine = (HERE / "stage2_training_receipt.py").read_text().splitlines()
    theirs = original.read_text().splitlines()
    allowed = {"from .dynamic_circuit_ir import canonical_sha256", 'STAGE2_DESIGN_FILENAME = "STAGE2_FIRST_MEASUREMENT_DESIGN_2026-08-08.md"',
               'STAGE2_PREREGISTRATION_SECTION = "## 8. Pre-registered success / failure criteria"',
               'STAGE2_PREREGISTRATION_NEXT_SECTION = "\\n## 9. "', "STAGE2_PREREGISTRATION_BYTE_LENGTH = 3_337",
               '    "70453e10f347b454443210ca26911e2e21a95c0c993dd6f1df02a1468d95d3d3"',
               '    "4a71d829934c8d07c127bcc09ba7dc59964c96135c0d523ef9c861ce2e0b83f8"',
               "STAGE2_RUNG_TOKENS = (10_000_000, 100_000_000, 1_000_000_000)",
               '"""Sealed stage-2 training receipts: one JSON per run, one addendum per rung.'}
    missing = [line for line in theirs if line not in mine and line not in allowed]
    assert missing == []
