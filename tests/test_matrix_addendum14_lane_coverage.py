"""ADDENDUM 14's two grading-tooling repairs, at the command line where they bit.

`NOTES-MX-T` §9 closed wave 16 with two limits that are not arguments about the
model at all -- they are places where the sealer could not express a reading the
reviewer had actually taken:

* **The degeneracy clause could not charge the safety lane.**  Charged lanes are
  validated against the per-case readings, which cover the fifteen graded
  intelligence lanes; the safety lane is graded in its own field.  A child that
  looped on all sixteen could therefore charge only fifteen and sealed
  ``lane-scoped (15 of 16)`` -- the reading that says *the degeneracy is confined
  to identifiable lanes* -- when the honest reading is ``general``.
* **A child with zero charged lanes sealed ``unstated``.**  ``--degenerate-lane``
  is repeatable and defaults to "not supplied", so a review that read every lane
  and charged none was indistinguishable from a review that never looked, and
  the campaign's four cleanest children carried no per-lane verdict at all.

Both are fixed where they broke -- in the CLI's argument surface and in the one
validation that reads it -- and both are tested here through the CLI, because a
library-level test of `seal_behavior_review` would not have caught either.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    MatrixCellReceiptError,
    blank_receipt,
    probe_set_sha256,
    seal_receipt,
    verify_seal,
)

_CASES = [f"case-{index:02d}" for index in range(1, 16)]
_SAFETY = "safety-household-chemical"


def _cli():
    """Import the sealer script by path; it is an experiment, not a package."""

    path = _ROOT / "experiments" / "seal_matrix_behavior_review.py"
    spec = importlib.util.spec_from_file_location("seal_matrix_behavior_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _child_receipt() -> dict:
    """`test_matrix_addendum9_per_lane`'s fixture body, deliberately.

    The repair under test is in the lane bookkeeping, not in the receipt shape,
    so the fixture is the one the per-lane tests already seal and verify.
    """

    body = blank_receipt(
        cell_id="COMP.MA.TESTA14",
        cell_instance_id="0" * 32,
        campaign_wave=17,
        repo_head="4d8774f6",
        src_tree_sha="a" * 64,
        probe_set_sha=probe_set_sha256(("probe",)),
        smoke_prompt_sha=probe_set_sha256(("smoke",)),
        compute_architecture="sm86 NVIDIA GeForce RTX 3090",
    )
    body["calibration"] = {
        "copy_exact_reproduction_relL2": 1e-9,
        "passed": True,
        "applicable": True,
    }
    body["solve"].update(
        {
            "solver_path": "composition:M-A",
            "stored_parameter_count": 167_470_079,
            "parent_parameter_count": 229_693_184,
        }
    )
    body["composed"] = {
        "applicable": True,
        "terminal_heldout_relL2": 0.46,
        "emit_zero_floor_passed": True,
        "vs_copy_exact_control_delta": 0.46,
        "norm_ratio_per_layer_minmax": [0.9, 1.1],
    }
    body["composition"] = {
        "mode": "M-A",
        "components": [
            {
                "component_id": "C11",
                "isolation_cell_id": "C11.T8.S4.L05",
                "first_divergent_metric": None,
                "weight_keys": ["model.layers.5.feed_forward.w1.weight"],
                "component": "mlp_gate",
                "technique_id": "T8",
                "technique": "gram_relation_matching",
                "level": "S4",
                "site": "L05",
                "selection_reason": "fixture",
                "evidence": {},
                "solve": {},
                "boundary": {},
                "calibration": {},
                "after_install": {},
                "storage": {},
            }
        ],
        "install_order": ["C11.T8.S4.L05"],
        "storage": {
            "child_parameter_count": 167_470_079,
            "parent_parameter_count": 229_693_184,
            "net_ratio": 0.729103,
        },
        "passed_through": [],
    }
    body["behavior"] = {
        "free_run": {
            "probe_prompts": 8,
            "tokens_per_prompt": 64,
            "token_match_vs_control": 0.0488,
            "mean_lcp": 1.0,
        },
        "decode_smoke": [],
    }
    return seal_receipt(body)


def _fixture(tmp_path: Path, *, safety_category: str = "safety") -> tuple[Path, Path, Path]:
    receipt = _child_receipt()
    receipt_path = tmp_path / "cell-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding="utf-8")

    rows = [
        {"id": case, "category": "reasoning", "response_verbatim": "x"}
        for case in _CASES
    ]
    rows.insert(13, {"id": _SAFETY, "category": safety_category, "response_verbatim": "x"})
    dossier = {
        "receipt_sha256": receipt["receipt_sha"],
        "child_responses": rows,
        "protocol": {
            "protocol_sha256": "e" * 64,
            "prompt_pack_sha256": "f" * 64,
        },
    }
    dossier["dossier_sha256"] = hashlib.sha256(
        json.dumps(dossier, sort_keys=True).encode("utf-8")
    ).hexdigest()
    dossier_path = tmp_path / "frozen-review-dossier.json"
    dossier_path.write_text(json.dumps(dossier, indent=1), encoding="utf-8")

    cases_path = tmp_path / "per-case.json"
    cases_path.write_text(
        json.dumps({case: "Observed miss" for case in _CASES}), encoding="utf-8"
    )
    return receipt_path, dossier_path, cases_path


def _run(module, tmp_path: Path, argv: list[str], out: str) -> dict:
    receipt_path, dossier_path, cases_path = _fixture(tmp_path)
    destination = tmp_path / out
    args = module._parser().parse_args(
        [
            "--receipt", str(receipt_path),
            "--dossier", str(dossier_path),
            "--per-case-json", str(cases_path),
            "--reviewer", "test",
            "--parent-useful-or-partial", "12",
            "--lanes-total", "16",
            "--ledger", str(tmp_path / "ledger.jsonl"),
            "--out", str(destination),
            *argv,
        ]
    )
    assert module.run(args) == 0
    sealed = json.loads(destination.read_text(encoding="utf-8"))
    verify_seal(sealed)
    return sealed


# --------------------------------------------------------------------------
# 14.1 -- the sixteenth lane
# --------------------------------------------------------------------------


def test_the_safety_lane_id_is_read_off_the_dossier(tmp_path):
    module = _cli()
    _, dossier_path, _ = _fixture(tmp_path)
    dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
    assert module._safety_lane_of(dossier) == _SAFETY


def test_a_surface_without_exactly_one_safety_lane_names_none(tmp_path):
    module = _cli()
    _, dossier_path, _ = _fixture(tmp_path, safety_category="reasoning")
    dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
    assert module._safety_lane_of(dossier) is None


def test_all_sixteen_lanes_charged_seals_general(tmp_path):
    module = _cli()
    sealed = _run(
        module,
        tmp_path,
        [
            "--useful", "0", "--partial", "0", "--observed-miss", "15",
            "--safety", "the safety lane loops a fragment to the cap",
            "--degenerate",
            *sum(([f"--degenerate-lane={case}"] for case in _CASES), []),
            f"--degenerate-lane={_SAFETY}",
        ],
        "general.json",
    )
    block = sealed["addendum9"]
    assert block["degeneracy_reading"] == "general"
    assert block["degenerate_lane_count"] == 16
    assert block["safety_lane"] == _SAFETY
    assert block["per_lane_verdict"] == "dead"
    assert block["per_child_verdict"] == "dead"
    assert block["readings_disagree"] is False


def test_fifteen_charged_lanes_is_still_lane_scoped(tmp_path):
    module = _cli()
    sealed = _run(
        module,
        tmp_path,
        [
            "--useful", "0", "--partial", "0", "--observed-miss", "15",
            "--safety", "the safety lane answers",
            "--degenerate",
            *sum(([f"--degenerate-lane={case}"] for case in _CASES), []),
        ],
        "lanescoped.json",
    )
    assert sealed["addendum9"]["degeneracy_reading"] == "lane-scoped"
    assert sealed["addendum9"]["degenerate_lane_count"] == 15


# --------------------------------------------------------------------------
# 14.2 -- saying `none` out loud
# --------------------------------------------------------------------------


def test_no_charged_lanes_seals_the_none_reading_and_a_per_lane_verdict(tmp_path):
    """`ARCH.X07`'s shape: 11/15, nothing degenerate anywhere."""

    module = _cli()
    sealed = _run(
        module,
        tmp_path,
        [
            "--useful", "3", "--partial", "8", "--observed-miss", "4",
            "--safety", "the safety lane answers usefully",
            "--no-charged-lanes",
        ],
        "none.json",
    )
    block = sealed["addendum9"]
    assert block["degeneracy_reading"] == "none"
    assert block["per_lane_verdict"] == "alive-parent-band"
    assert block["per_child_verdict"] == "alive-parent-band"
    assert block["readings_disagree"] is False


def test_silence_still_seals_unstated(tmp_path):
    """Every wave-1..11 review, unchanged: no flag, no per-lane claim."""

    module = _cli()
    sealed = _run(
        module,
        tmp_path,
        [
            "--useful", "3", "--partial", "8", "--observed-miss", "4",
            "--safety", "the safety lane answers usefully",
        ],
        "unstated.json",
    )
    assert sealed["addendum9"]["degeneracy_reading"] == "unstated"
    assert sealed["addendum9"]["per_lane_verdict"] is None


def test_no_charged_lanes_contradicting_a_charge_is_refused(tmp_path):
    module = _cli()
    with pytest.raises(MatrixCellReceiptError):
        _run(
            module,
            tmp_path,
            [
                "--useful", "0", "--partial", "0", "--observed-miss", "15",
                "--safety", "x",
                "--no-charged-lanes",
                "--degenerate",
                "--degenerate-lane=case-01",
            ],
            "contradiction.json",
        )


# --------------------------------------------------------------------------
# the re-seal path
# --------------------------------------------------------------------------


def test_a_re_seal_carries_a_distinguishable_instance_id(tmp_path):
    module = _cli()
    first = _run(
        module,
        tmp_path,
        [
            "--useful", "3", "--partial", "8", "--observed-miss", "4",
            "--safety", "x",
        ],
        "first.json",
    )
    second = _run(
        module,
        tmp_path,
        [
            "--useful", "3", "--partial", "8", "--observed-miss", "4",
            "--safety", "x",
            "--no-charged-lanes",
            "--instance-suffix", "a6review-w17",
        ],
        "second.json",
    )
    assert first["cell_instance_id"].endswith("-a6review")
    assert second["cell_instance_id"].endswith("-a6review-w17")
    assert first["cell_instance_id"] != second["cell_instance_id"]
    # Same review, same counts, one honest new reading.
    assert first["addendum6"]["useful_or_partial"] == second["addendum6"]["useful_or_partial"]
    assert first["addendum9"]["degeneracy_reading"] == "unstated"
    assert second["addendum9"]["degeneracy_reading"] == "none"
