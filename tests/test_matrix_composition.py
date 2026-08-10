"""Contract tests for the composition wave: recipes, receipts, aggregation.

ADDENDUM 4 turns "do the isolation winners compose?" into a measurement, and a
measurement is only as good as the things that cannot silently go wrong in it.
Five of those are pinned here:

1. A composed child's cell id parses, and a composition receipt without its
   per-component provenance array cannot be sealed at all.
2. A recipe that folds one parent tensor twice is refused on a workstation --
   before a GPU is touched -- because the second install would erase the first
   and score the survivor against evidence the loser invalidated.  C10 owns all
   four attention projections at its site, so this collision is not
   hypothetical.
3. Selection is a function of sealed numbers.  The behaviour-first ranking picks
   the highest free-run *except* where the storage-honesty clause applies, the
   storage-honest ranking admits only folds that contract, and both carry their
   reason.
4. Aggregation is pessimistic.  Six clean solves must not average away one bad
   one -- the aggregate write cosine is the worst, the aggregate macrocell is
   the worst.
5. Net storage is computed over the tensors actually replaced.  C14's sealed
   0.5x is a bookkeeping artifact (it owns two norms and refits one), and a
   composed child that inherited that ratio would claim a saving it does not
   have.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    CellId,
    MatrixCellReceiptError,
    blank_receipt,
    campaign_pass,
    first_divergent_metric,
    ledger_line,
    probe_set_sha256,
    seal_receipt,
    verify_seal,
)
from neural_forge.matrix_composition import (  # noqa: E402
    STORAGE_TIEBREAK_FREE_RUN,
    MatrixCompositionError,
    aggregate_boundary,
    aggregate_calibration,
    aggregate_solve,
    build_recipe,
    candidate_from_receipt,
    component_weight_keys,
    eligible,
    evidence_cache_name,
    evidence_endpoints,
    grade,
    install_order_key,
    net_storage,
    resolve_conflicts,
    validate_recipe,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _solve(**overrides):
    body = {
        "solver_path": "test",
        "ridge_relative": 0.0,
        "condition_number": 10.0,
        "condition_cap_saturated": False,
        "prior_norm": 1.0,
        "correction_norm": 0.1,
        "requested_clamp_ratio": 0.0,
        "allowed_clamp_ratio": 0.0,
        "clamp_engaged": False,
        "omp": None,
    }
    body.update(overrides)
    return body


def _boundary(**overrides):
    body = {
        "applicable": True,
        "local_solve_applicable": True,
        "macrocell_endpoint": "residual.post_layer",
        "write_heldout_relL2": 0.01,
        "write_heldout_cos": 0.999,
        "post_operator_heldout_relL2": 0.02,
        "post_operator_heldout_cos": 0.999,
        "endpoint_band_le_0p4": True,
    }
    body.update(overrides)
    return body


def _entry(cell_id="C06.T5.M16.L12", **overrides):
    cell = CellId.parse(cell_id)
    entry = {
        "component_id": cell.component,
        "component": "attn_q",
        "technique_id": cell.technique,
        "technique": "sparse_omp_routes",
        "level": cell.level,
        "site": cell.site,
        "isolation_cell_id": cell.text,
        "selection_reason": "test",
        "weight_keys": list(component_weight_keys(cell)),
        "evidence": {"conditioned": False, "upstream_installed": []},
        "solve": _solve(),
        "boundary": _boundary(),
        "calibration": {
            "copy_exact_reproduction_relL2": 1e-3,
            "passed": True,
            "applicable": True,
        },
        "after_install": {
            "terminal_relL2_vs_control": 0.004,
            "free_run_token_match_vs_control": 0.98,
            "free_run_mean_lcp": 0.9,
            "probe_prompts": 8,
            "tokens_per_prompt": 64,
            "norm_ratio_per_layer_minmax": [0.99, 1.01],
        },
        "storage": {
            "stored_parameter_count": 983552,
            "installed_parameter_count": 1048576,
            "owned_parameter_count": 1048576,
            "declared_ratio": 983552 / 1048576,
        },
        "first_divergent_metric": None,
    }
    entry.update(overrides)
    return entry


def _composition_receipt(entries, *, cell_id="COMP.MA.RBEH", parent=229_693_184):
    body = blank_receipt(
        cell_id=cell_id,
        cell_instance_id="0" * 32,
        campaign_wave=4,
        repo_head="5996655b",
        src_tree_sha="a" * 64,
        probe_set_sha=probe_set_sha256(("probe",)),
        smoke_prompt_sha=probe_set_sha256(("smoke",)),
        compute_architecture="sm86 NVIDIA GeForce RTX 3090",
    )
    storage = net_storage(parent, entries)
    body["calibration"] = aggregate_calibration(entries)
    body["solve"] = aggregate_solve(
        entries,
        mode="M-A",
        child_parameter_count=storage["child_parameter_count"],
        parent_parameter_count=storage["parent_parameter_count"],
    )
    body["boundary"] = aggregate_boundary(entries)
    body["composed"] = {
        "applicable": True,
        "terminal_heldout_relL2": 0.03,
        "emit_zero_floor_passed": True,
        "vs_copy_exact_control_delta": 0.03,
        "norm_ratio_per_layer_minmax": [0.98, 1.02],
    }
    body["behavior"] = {
        "free_run": {
            "probe_prompts": 8,
            "tokens_per_prompt": 64,
            "token_match_vs_control": 0.55,
            "mean_lcp": 0.3,
        },
        "decode_smoke": [
            {"prompt_sha": "x", "output_verbatim": "hi", "terminated": "cap"}
        ],
    }
    body["composition"] = {
        "mode": "M-A",
        "install_order": [e["isolation_cell_id"] for e in entries],
        "components": entries,
        "passed_through": [],
        "storage": storage,
    }
    return body


def _candidate(**overrides):
    body = {
        "cell_id": "C06.T1.M16.L12",
        "component_id": "C06",
        "component": "attn_q",
        "technique_id": "T1",
        "technique": "dense_prior_anchored_ridge",
        "level": "M16",
        "site": "L12",
        "campaign_wave": 2,
        "status": "pass",
        "first_divergent_metric": None,
        "decode_smoke_review": "coherent",
        "free_run": 1.0,
        "vs_control_delta": 0.00148,
        "stored_parameter_count": 1966080,
        "parent_parameter_count": 1048576,
        "stored_ratio": 1.875,
        "grade": "strong",
        "receipt_sha": "b" * 64,
    }
    body.update(overrides)
    body["weight_keys"] = list(component_weight_keys(CellId.parse(body["cell_id"])))
    return body


# --------------------------------------------------------------------------
# 1. identity and the mandatory provenance array
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "mode", "variant"),
    [("COMP.MA", "MA", None), ("COMP.MB", "MB", None), ("COMP.MC.RSTO", "MC", "RSTO")],
)
def test_composition_cell_ids_round_trip(text, mode, variant):
    cell = CellId.parse(text)
    assert cell.kind == "composition"
    assert cell.text == text
    assert cell.composition_mode == mode
    assert cell.composition_variant == variant
    assert cell.layer_index is None


@pytest.mark.parametrize("text", ["COMP.MD", "COMP", "COMP.MA.lower", "COMP.MA.X.Y"])
def test_malformed_composition_ids_are_not_new_cells(text):
    with pytest.raises(MatrixCellReceiptError):
        CellId.parse(text)


def test_a_composition_receipt_without_provenance_cannot_be_sealed():
    body = _composition_receipt([_entry()])
    del body["composition"]
    with pytest.raises(MatrixCellReceiptError, match="must carry a 'composition'"):
        seal_receipt(body)


def test_only_a_composition_cell_may_carry_a_composition_block():
    body = blank_receipt(
        cell_id="C13.T5.M16.L05",
        cell_instance_id="0" * 32,
        campaign_wave=1,
        repo_head="h",
        src_tree_sha="a" * 64,
        probe_set_sha=probe_set_sha256(("p",)),
        smoke_prompt_sha=probe_set_sha256(("s",)),
        compute_architecture="sm86 x",
    )
    body["composition"] = {"mode": "M-A"}
    with pytest.raises(MatrixCellReceiptError, match="Only a COMP"):
        seal_receipt(body)


def test_empty_components_needs_a_stated_reason():
    body = _composition_receipt([_entry()])
    body["composition"]["components"] = []
    body["composition"]["install_order"] = []
    with pytest.raises(MatrixCellReceiptError, match="incomplete_reason"):
        seal_receipt(body)
    body["composition"]["incomplete_reason"] = "died at install 0"
    sealed = seal_receipt(body)
    assert verify_seal(sealed) == sealed["receipt_sha"]


def test_mode_must_agree_with_the_cell_id():
    body = _composition_receipt([_entry()], cell_id="COMP.MB")
    with pytest.raises(MatrixCellReceiptError, match="disagrees with cell id"):
        seal_receipt(body)


def test_install_order_must_match_the_components():
    body = _composition_receipt([_entry(), _entry("C07.T1.M16.L12")])
    body["composition"]["install_order"] = list(
        reversed(body["composition"]["install_order"])
    )
    with pytest.raises(MatrixCellReceiptError, match="install_order"):
        seal_receipt(body)


def test_a_composed_child_folds_each_parent_tensor_at_most_once():
    body = _composition_receipt([_entry(), _entry()])
    with pytest.raises(MatrixCellReceiptError, match="already installed"):
        seal_receipt(body)


def test_a_composition_receipt_seals_and_makes_a_ledger_row():
    sealed = seal_receipt(_composition_receipt([_entry()]))
    assert sealed["cell_kind"] == "composition"
    assert sealed["component"] == "composed_child"
    assert sealed["technique"] == "heterogeneous_composition"
    # Global drift 0.03 crosses the 0.02 gate before free-run 0.55 is reached:
    # the chain reports the FIRST crossing, and on a composed child link 5 is
    # the one ADDENDUM 4 expects to see move.
    assert sealed["outcome"]["first_divergent_metric"] == "composition"
    assert campaign_pass(sealed) is False
    row = ledger_line(sealed)
    assert row["cell_id"] == "COMP.MA.RBEH"
    assert row["component_id"] == "COMP"
    assert row["technique_id"] == "composition"
    # The net storage ratio rides in the column that already exists for it.
    assert row["stored_parameter_ratio"] == pytest.approx(
        sealed["composition"]["storage"]["net_ratio"]
    )


# --------------------------------------------------------------------------
# 2. recipes
# --------------------------------------------------------------------------


def test_component_weight_keys_resolve_without_a_model():
    assert component_weight_keys(CellId.parse("C06.T1.M16.L12")) == (
        "model.layers.12.self_attn.q_proj.weight",
    )
    assert component_weight_keys(CellId.parse("C10.T1.M16.L02")) == (
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.2.self_attn.k_proj.weight",
        "model.layers.2.self_attn.v_proj.weight",
        "model.layers.2.self_attn.out_proj.weight",
    )
    assert component_weight_keys(CellId.parse("C14.T1.NA.FINAL")) == (
        "model.embedding_norm.weight",
    )


def test_c10_reuses_c09s_evidence_cache_because_its_endpoints_moved():
    # Wave 3 moved C10's endpoints onto C09's (`matrix_technique_registry:179`)
    # and pointed every C10 cell at `L02-C09` by hand.  The stale `L02-C10`
    # directory on the run host still binds wave 1's `operator.*` endpoints and
    # would be refused by the cache binding check, so the reuse has to be
    # derived, not remembered.
    assert evidence_cache_name(CellId.parse("C10.T1.M16.L02")) == "L02-C09"
    assert evidence_cache_name(CellId.parse("C09.T3.M16.L02")) == "L02-C09"
    assert evidence_cache_name(CellId.parse("C06.T1.M16.L12")) == "L12-C06"
    assert evidence_cache_name(CellId.parse("C08.T1.M16.L02")) == "L02-C08"
    # C14 asks for exactly C03's endpoint set at the same layer.
    assert evidence_cache_name(CellId.parse("C14.T1.NA.L00")) == "L00-C03"
    assert evidence_endpoints(CellId.parse("C05.T3.M16.L05")) == (
        "operator.output",
        "residual.post_layer",
        "residual.post_operator",
        "residual.pre",
        "short_conv.context",
    )


def test_install_order_is_input_side_first_then_conv_attention_norms():
    cells = [
        CellId.parse(text)
        for text in (
            "C06.T1.M16.L12",
            "C14.T1.NA.L00",
            "C05.T3.M16.L05",
            "C03.T5.M16.L00",
            "C09.T2.M16.L02",
        )
    ]
    ordered = [cell.text for cell in sorted(cells, key=install_order_key)]
    assert ordered == [
        "C03.T5.M16.L00",
        "C14.T1.NA.L00",
        "C09.T2.M16.L02",
        "C05.T3.M16.L05",
        "C06.T1.M16.L12",
    ]


def test_grades_follow_addendum_3():
    assert grade(0.95, "coherent") == "strong"
    assert grade(0.95, "pending") == "weak"
    assert grade(0.61, "coherent") == "promising"
    assert grade(0.59, "coherent") == "weak"
    assert grade(0.05, "coherent") == "dead"
    assert grade(None, "coherent") == "dead"


def test_an_infra_cell_is_never_eligible():
    assert eligible({"status": "infra", "grade": "strong"}) is False
    assert eligible({"status": "fail", "grade": "promising"}) is True
    assert eligible({"status": "fail", "grade": "weak"}) is False


def test_storage_honesty_promotes_only_inside_the_band():
    inside = build_recipe(
        [
            _candidate(),  # T1: 1.0000 free-run, 1.875x
            _candidate(
                cell_id="C06.T5.M16.L12",
                technique_id="T5",
                free_run=1.0 - STORAGE_TIEBREAK_FREE_RUN / 2,
                stored_parameter_count=983552,
                stored_ratio=983552 / 1048576,
            ),
        ],
        variant="RBEH",
        ranking="behaviour-first",
        ledger_sha256="c" * 64,
    )
    assert inside["install_order"] == ["C06.T5.M16.L12"]
    assert "storage-honesty clause" in inside["components"][0]["selection_reason"]

    outside = build_recipe(
        [
            _candidate(),
            _candidate(
                cell_id="C06.T5.M16.L12",
                technique_id="T5",
                free_run=1.0 - 2 * STORAGE_TIEBREAK_FREE_RUN,
                stored_parameter_count=983552,
                stored_ratio=983552 / 1048576,
            ),
        ],
        variant="RBEH",
        ranking="behaviour-first",
        ledger_sha256="c" * 64,
    )
    assert outside["install_order"] == ["C06.T1.M16.L12"]


def test_storage_honest_ranking_admits_only_folds_that_contract():
    recipe = build_recipe(
        [_candidate()],
        variant="RSTO",
        ranking="storage-honest",
        ledger_sha256="c" * 64,
    )
    assert recipe["components"] == []
    assert recipe["passed_through"][0]["component_id"] == "C06"
    assert "stores >= 1.0x" in recipe["passed_through"][0]["reason"]


def test_a_component_with_no_promising_technique_is_passed_through():
    recipe = build_recipe(
        [
            _candidate(
                cell_id="C13.T3.M16.L05",
                component_id="C13",
                technique_id="T3",
                site="L05",
                free_run=0.5859,
                grade="weak",
                status="fail",
                stored_parameter_count=8_601_600,
                parent_parameter_count=2_621_440,
                stored_ratio=3.28125,
            )
        ],
        variant="RBEH",
        ranking="behaviour-first",
        ledger_sha256="c" * 64,
    )
    assert recipe["components"] == []
    entry = recipe["passed_through"][0]
    assert entry["component_id"] == "C13"
    assert "no technique reached ADDENDUM 3 `promising`" in entry["reason"]
    assert entry["best_sealed_free_run"] == pytest.approx(0.5859)


def test_conflicts_resolve_on_the_recipe_s_own_axis():
    picks = [
        {
            "cell_id": "C09.T1.M16.L02",
            "component_id": "C09",
            "weight_keys": list(
                component_weight_keys(CellId.parse("C09.T1.M16.L02"))
            ),
            "isolation": {
                "free_run": 0.8203,
                "stored_ratio": 1.875,
                "stored_parameter_count": 1966080,
                "parent_parameter_count": 1048576,
                "vs_control_delta": 0.0132,
            },
            "selection_reason": "",
        },
        {
            "cell_id": "C10.T1.M16.L02",
            "component_id": "C10",
            "weight_keys": list(
                component_weight_keys(CellId.parse("C10.T1.M16.L02"))
            ),
            "isolation": {
                "free_run": 0.7363,
                "stored_ratio": 0.9583,
                "stored_parameter_count": 3014656,
                "parent_parameter_count": 3145728,
                "vs_control_delta": 0.0305,
            },
            "selection_reason": "",
        },
    ]
    kept, dropped = resolve_conflicts(picks, ranking="behaviour-first")
    assert [e["cell_id"] for e in kept] == ["C09.T1.M16.L02"]
    assert dropped[0]["component_id"] == "C10"

    kept, dropped = resolve_conflicts(picks, ranking="storage-honest")
    assert [e["cell_id"] for e in kept] == ["C10.T1.M16.L02"]
    assert dropped[0]["component_id"] == "C09"


def test_validate_recipe_refuses_a_double_fold_and_a_broken_order():
    good = build_recipe(
        [
            _candidate(),
            _candidate(
                cell_id="C03.T5.M16.L00",
                component_id="C03",
                technique_id="T5",
                site="L00",
                free_run=0.7871,
                grade="promising",
                status="fail",
                stored_parameter_count=2949632,
                parent_parameter_count=3145728,
                stored_ratio=2949632 / 3145728,
            ),
        ],
        variant="RBEH",
        ranking="behaviour-first",
        ledger_sha256="c" * 64,
    )
    validate_recipe(good)
    assert good["install_order"] == ["C03.T5.M16.L00", "C06.T1.M16.L12"]

    broken = dict(good)
    broken["components"] = list(reversed(good["components"]))
    broken["install_order"] = [e["cell_id"] for e in broken["components"]]
    with pytest.raises(MatrixCompositionError, match="topological install order"):
        validate_recipe(broken)


def test_candidate_from_receipt_reads_storage_the_ledger_row_would_drop():
    body = blank_receipt(
        cell_id="C05.T5.M16.L05",
        cell_instance_id="0" * 32,
        campaign_wave=1,
        repo_head="h",
        src_tree_sha="a" * 64,
        probe_set_sha=probe_set_sha256(("p",)),
        smoke_prompt_sha=probe_set_sha256(("s",)),
        compute_architecture="sm86 x",
    )
    body["solve"]["stored_parameter_count"] = 983040
    body["solve"]["parent_parameter_count"] = 1048576
    body["behavior"]["free_run"]["token_match_vs_control"] = 0.791015625
    body["outcome"]["decode_smoke_review"] = "coherent"
    body["receipt_sha"] = "d" * 64
    candidate = candidate_from_receipt(body)
    assert candidate["stored_ratio"] == pytest.approx(0.9375)
    assert candidate["grade"] == "promising"
    assert eligible(candidate) is True


# --------------------------------------------------------------------------
# 3. aggregation
# --------------------------------------------------------------------------


def test_aggregate_boundary_reports_the_worst_install_not_the_mean():
    good = _entry("C06.T5.M16.L12")
    bad = _entry(
        "C05.T3.M16.L05",
        component_id="C05",
        weight_keys=list(component_weight_keys(CellId.parse("C05.T3.M16.L05"))),
        boundary=_boundary(
            write_heldout_cos=0.9,
            post_operator_heldout_relL2=0.44,
            endpoint_band_le_0p4=False,
        ),
    )
    block = aggregate_boundary([good, bad, good])
    assert block["write_heldout_cos"] == pytest.approx(0.9)
    assert block["post_operator_heldout_relL2"] == pytest.approx(0.44)
    assert block["endpoint_band_le_0p4"] is False
    assert block["worst_write_component_id"] == "C05"
    assert block["worst_macrocell_component_id"] == "C05"


def test_aggregate_boundary_declares_itself_inapplicable_with_a_reason():
    entry = _entry(
        boundary={
            "applicable": False,
            "local_solve_applicable": False,
            "macrocell_endpoint": None,
            "write_heldout_relL2": None,
            "write_heldout_cos": None,
            "post_operator_heldout_relL2": None,
            "post_operator_heldout_cos": None,
            "endpoint_band_le_0p4": None,
            "inapplicable_reason": "no nameable write",
        }
    )
    block = aggregate_boundary([entry])
    assert block["applicable"] is False
    assert block["local_solve_applicable"] is False
    assert block["inapplicable_reason"]


def test_aggregate_calibration_needs_every_applicable_gate_to_pass():
    good = _entry()
    bad = _entry(
        "C05.T3.M16.L05",
        component_id="C05",
        calibration={
            "copy_exact_reproduction_relL2": 0.9,
            "passed": False,
            "applicable": True,
        },
    )
    assert aggregate_calibration([good])["passed"] is True
    block = aggregate_calibration([good, bad])
    assert block["passed"] is False
    assert block["worst_component_id"] == "C05"

    only_inapplicable = _entry(
        "C10.T1.M16.L02",
        component_id="C10",
        calibration={
            "copy_exact_reproduction_relL2": None,
            "passed": False,
            "applicable": False,
            "inapplicable_reason": "four tensors behind a softmax",
        },
    )
    block = aggregate_calibration([only_inapplicable])
    assert block["applicable"] is False
    assert "softmax" in block["inapplicable_reason"]


def test_aggregate_solve_is_pessimistic_and_carries_the_net_counts():
    clean = _entry()
    saturated = _entry(
        "C05.T3.M16.L05",
        component_id="C05",
        solve=_solve(
            condition_number=None,
            condition_cap_saturated=True,
            clamp_engaged=True,
            requested_clamp_ratio=8.0,
            allowed_clamp_ratio=1.0,
        ),
    )
    block = aggregate_solve(
        [clean, saturated],
        mode="M-B",
        child_parameter_count=229_000_000,
        parent_parameter_count=229_693_184,
    )
    assert block["solver_path"] == "composition:M-B"
    assert block["condition_cap_saturated"] is True
    assert block["condition_cap_saturated_components"] == ["C05"]
    assert block["clamp_engaged"] is True
    assert block["requested_clamp_ratio"] == pytest.approx(8.0)
    assert block["stored_parameter_count"] == 229_000_000


def test_net_storage_ignores_the_c14_bookkeeping_artifact():
    norms = _entry(
        "C14.T1.NA.L00",
        component_id="C14",
        component="norms",
        technique_id="T1",
        weight_keys=list(component_weight_keys(CellId.parse("C14.T1.NA.L00"))),
        storage={
            # The kernel refits operator_norm only, and seals 0.5x against a
            # component that *owns* both norms.
            "stored_parameter_count": 1024,
            "installed_parameter_count": 1024,
            "owned_parameter_count": 2048,
            "declared_ratio": 0.5,
        },
    )
    storage = net_storage(229_693_184, [norms])
    assert storage["net_delta"] == 0
    assert storage["net_ratio"] == pytest.approx(1.0)
    assert storage["contracts"] is False
    assert storage["per_component"][0]["declared_ratio"] == 0.5
    assert storage["per_component"][0]["effective_ratio"] == pytest.approx(1.0)


def test_net_storage_adds_a_costly_fold_to_the_child():
    ridge = _entry(
        "C09.T1.M16.L02",
        component_id="C09",
        weight_keys=list(component_weight_keys(CellId.parse("C09.T1.M16.L02"))),
        storage={
            "stored_parameter_count": 1966080,
            "installed_parameter_count": 1048576,
            "owned_parameter_count": 1048576,
            "declared_ratio": 1.875,
        },
    )
    storage = net_storage(229_693_184, [ridge])
    assert storage["net_delta"] == 917504
    assert storage["contracts"] is False
    assert storage["net_ratio"] > 1.0


def test_the_chain_still_reads_behaviour_only_on_a_composed_child():
    # A composed child with a wrecked boundary but an intact free-run must not
    # be failed by the boundary, and one with a perfect boundary and a broken
    # free-run must fail: §5 rule 1 does not change shape because the cell got
    # bigger.
    entry = _entry(
        boundary=_boundary(post_operator_heldout_relL2=0.30),
    )
    body = _composition_receipt([entry])
    body["behavior"]["free_run"]["token_match_vs_control"] = 0.99
    body["outcome"]["decode_smoke_review"] = "coherent"
    sealed = seal_receipt(body)
    assert first_divergent_metric(sealed) == "composition"

    body = _composition_receipt([entry])
    body["composed"]["vs_copy_exact_control_delta"] = 0.001
    body["composed"]["terminal_heldout_relL2"] = 0.001
    body["behavior"]["free_run"]["token_match_vs_control"] = 0.99
    body["outcome"]["decode_smoke_review"] = "coherent"
    sealed = seal_receipt(body)
    assert sealed["outcome"]["status"] == "pass"
    assert campaign_pass(sealed) is True
