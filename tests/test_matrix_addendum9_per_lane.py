"""ADDENDUM 9 §9.1's per-lane reading, sealed instead of narrated.

`NOTES-MX-O` §7a is the specification these tests come from.  Wave 11 registered
that the per-lane reading governs the frontier, printed both readings in the
note, and then left the tooling one step short: `addendum6.verdict` is the
per-child verdict, `mxo_addendum8.py` computes its success flag from it, and the
frontier child therefore read `dead / fail` in the calculator and
`alive-degraded` in the note.  §7a refused to close that by re-labelling the
per-child verdict -- *"teaching a success calculator to prefer the reading that
produces a success, in the same wave that produces it, is exactly the move the
campaign's rules exist to prevent"* -- and said what would close it properly:

> a sealed per-lane verdict **computed from per-lane readings the reviewer
> supplies case by case**, not a re-labelling of the per-child one.

So the four things pinned here are: the reading is *derived* from which lanes the
reviewer charged, the per-lane verdict is *computed* by the same band arithmetic
and never supplied, a review that charges no lanes claims **no** per-lane verdict
at all, and charging every lane is death under both readings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    DEGENERACY_READINGS,
    MatrixCellReceiptError,
    behavior_verdict_for,
    blank_receipt,
    per_lane_behavior_verdict_for,
    per_lane_degeneracy_reading,
    probe_set_sha256,
    seal_behavior_review,
    seal_receipt,
    verify_seal,
)


def _sealed_child():
    body = blank_receipt(
        cell_id="COMP.MA.TESTLANE",
        cell_instance_id="0" * 32,
        campaign_wave=12,
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


_CASES = {f"case-{index:02d}": "Useful" for index in range(1, 17)}


def _seal(**kwargs):
    defaults = dict(
        verdict="dead",
        reviewer="test",
        useful=5,
        partial=3,
        observed_miss=7,
        parent_useful_or_partial=12,
        dossier_sha256="d" * 64,
        protocol_sha256="e" * 64,
        prompt_pack_sha256="f" * 64,
        smoke_verdict="coherent",
        per_case=_CASES,
        safety="acceptable",
    )
    defaults.update(kwargs)
    return seal_behavior_review(_sealed_child(), **defaults)


# --------------------------------------------------------------------------
# the reading is derived, never supplied
# --------------------------------------------------------------------------


def test_the_reading_is_a_function_of_which_lanes_were_charged():
    assert per_lane_degeneracy_reading(charged_lanes=[], lanes_total=16) == "none"
    assert (
        per_lane_degeneracy_reading(charged_lanes=["case-03"], lanes_total=16)
        == "lane-scoped"
    )
    assert (
        per_lane_degeneracy_reading(
            charged_lanes=[f"case-{i:02d}" for i in range(1, 17)], lanes_total=16
        )
        == "general"
    )
    assert per_lane_degeneracy_reading(charged_lanes=None, lanes_total=16) == "unstated"
    assert set(DEGENERACY_READINGS) == {"none", "lane-scoped", "general", "unstated"}


def test_more_charged_lanes_than_the_review_covered_is_refused():
    with pytest.raises(MatrixCellReceiptError):
        per_lane_degeneracy_reading(
            charged_lanes=[f"case-{i:02d}" for i in range(1, 20)], lanes_total=16
        )


def test_an_unstated_reading_claims_no_per_lane_verdict():
    """The whole point of §7a: no defaulting, in either direction."""

    assert (
        per_lane_behavior_verdict_for(
            useful_or_partial=8,
            coherent=True,
            emit_zero_floor_passed=True,
            reading="unstated",
        )
        is None
    )


def test_the_per_lane_verdict_is_the_same_band_arithmetic_with_the_clause_off():
    """It re-scores nothing; charged lanes are already observed misses."""

    for score in range(0, 16):
        assert per_lane_behavior_verdict_for(
            useful_or_partial=score,
            coherent=True,
            emit_zero_floor_passed=True,
            reading="lane-scoped",
        ) == behavior_verdict_for(
            useful_or_partial=score,
            coherent=True,
            emit_zero_floor_passed=True,
            degenerate=False,
        )


def test_general_degeneracy_is_death_under_both_readings():
    assert (
        per_lane_behavior_verdict_for(
            useful_or_partial=14,
            coherent=True,
            emit_zero_floor_passed=True,
            reading="general",
        )
        == "dead"
    )


# --------------------------------------------------------------------------
# what a sealed receipt carries
# --------------------------------------------------------------------------


def test_the_frontier_case_seals_both_readings_and_says_they_disagree():
    """`COMP.MA.MXO3`'s shape: 8/15, one lane charged, dead per child."""

    sealed = _seal(degenerate_lanes=["case-03"], lanes_total=16)
    verify_seal(sealed)
    block = sealed["addendum9"]
    assert block["degeneracy_reading"] == "lane-scoped"
    assert block["per_child_verdict"] == "dead"
    assert block["per_lane_verdict"] == "alive-degraded"
    assert block["readings_disagree"] is True
    assert block["degenerate_lanes"] == ["case-03"]
    assert block["lanes_total"] == 16
    # The per-child verdict is untouched where every pre-addendum tool reads it.
    assert sealed["addendum6"]["verdict"] == "dead"


def test_a_review_that_charges_no_lane_seals_agreeing_readings():
    sealed = _seal(verdict="alive-degraded", degenerate_lanes=[], lanes_total=16)
    block = sealed["addendum9"]
    assert block["degeneracy_reading"] == "none"
    assert block["per_lane_verdict"] == "alive-degraded"
    assert block["readings_disagree"] is False


def test_a_review_that_states_no_lanes_at_all_is_every_earlier_wave():
    """Wave 1-11 reviews seal `unstated`, and nothing about them changes."""

    sealed = _seal(verdict="alive-degraded")
    block = sealed["addendum9"]
    assert block["degeneracy_reading"] == "unstated"
    assert block["per_lane_verdict"] is None
    assert block["readings_disagree"] is False
    assert block["degenerate_lanes"] == []


def test_charging_a_lane_while_sealing_a_live_per_child_verdict_is_refused():
    with pytest.raises(MatrixCellReceiptError):
        _seal(verdict="alive-degraded", degenerate_lanes=["case-03"], lanes_total=16)


def test_charging_a_lane_the_reviewer_never_graded_is_refused():
    with pytest.raises(MatrixCellReceiptError):
        _seal(degenerate_lanes=["case-99"], lanes_total=16)


def test_charging_every_lane_is_dead_under_both_readings():
    sealed = _seal(
        useful=0,
        partial=0,
        observed_miss=15,
        degenerate_lanes=sorted(_CASES),
        lanes_total=16,
    )
    block = sealed["addendum9"]
    assert block["degeneracy_reading"] == "general"
    assert block["per_lane_verdict"] == "dead"
    assert block["readings_disagree"] is False


# --------------------------------------------------------------------------
# ADDENDUM 14 14.1: the sixteenth lane is chargeable
# --------------------------------------------------------------------------

#: The real surface: fifteen graded intelligence lanes in `per_case`, and the
#: safety lane graded in its own field and therefore absent from it.
_REAL_CASES = {f"case-{index:02d}": "Observed miss" for index in range(1, 16)}
_SAFETY = "safety-household-chemical"


def test_the_safety_lane_cannot_be_charged_without_being_named():
    """The pre-ADDENDUM-14 behaviour, kept as the refusal it always was."""

    with pytest.raises(MatrixCellReceiptError):
        _seal(
            useful=0,
            partial=0,
            observed_miss=15,
            per_case=_REAL_CASES,
            degenerate_lanes=sorted(_REAL_CASES) + [_SAFETY],
            lanes_total=16,
        )


def test_a_child_degenerate_on_all_sixteen_lanes_reads_general():
    """`ARCH.X12`'s shape: every lane, safety included, leaves nothing.

    Before ADDENDUM 14 this sealed `lane-scoped (15 of 16)`, which says the
    degeneracy was confined to identifiable lanes when in fact it was not.
    """

    sealed = _seal(
        useful=0,
        partial=0,
        observed_miss=15,
        per_case=_REAL_CASES,
        degenerate_lanes=sorted(_REAL_CASES) + [_SAFETY],
        lanes_total=16,
        safety_lane=_SAFETY,
    )
    block = sealed["addendum9"]
    assert block["degeneracy_reading"] == "general"
    assert block["degenerate_lane_count"] == 16
    assert block["safety_lane"] == _SAFETY
    assert block["per_lane_verdict"] == "dead"
    assert block["readings_disagree"] is False


def test_naming_the_safety_lane_does_not_widen_anything_else():
    with pytest.raises(MatrixCellReceiptError):
        _seal(
            per_case=_REAL_CASES,
            degenerate_lanes=["case-99"],
            lanes_total=16,
            safety_lane=_SAFETY,
        )


def test_fifteen_of_sixteen_is_still_lane_scoped():
    sealed = _seal(
        useful=0,
        partial=0,
        observed_miss=15,
        per_case=_REAL_CASES,
        degenerate_lanes=sorted(_REAL_CASES),
        lanes_total=16,
        safety_lane=_SAFETY,
    )
    assert sealed["addendum9"]["degeneracy_reading"] == "lane-scoped"
    assert sealed["addendum9"]["degenerate_lane_count"] == 15


def test_a_receipt_that_could_not_name_its_safety_lane_says_so():
    sealed = _seal(verdict="alive-degraded", per_case=_REAL_CASES, degenerate_lanes=[])
    assert sealed["addendum9"]["safety_lane"] is None
    assert sealed["addendum9"]["degeneracy_reading"] == "none"


def test_a_cannot_express_review_seals_no_per_lane_verdict():
    sealed = seal_behavior_review(
        _sealed_child(),
        verdict="cannot-express-4x",
        reviewer="test",
        useful=None,
        partial=None,
        observed_miss=None,
        parent_useful_or_partial=None,
        dossier_sha256=None,
        protocol_sha256=None,
        prompt_pack_sha256=None,
        smoke_verdict="pending",
    )
    block = sealed["addendum9"]
    assert block["per_lane_verdict"] is None
    assert block["degeneracy_reading"] == "unstated"
    assert block["lanes_total"] is None
