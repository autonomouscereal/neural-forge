"""Contract tests for the matrix campaign's frozen receipt schema and ledger.

The receipt is one of exactly two objects the harness lane and the technique
lane share (design §4.3), so its behaviour is a contract, not an implementation
detail.  Four things are pinned here:

1. Cell ids parse to exactly the design's axes and nothing else -- a typo in a
   component, technique, level or site is a hard failure, never a new cell.
2. The mandatory field set fails closed.  A receipt that "forgot" to serialize
   the conditioning telemetry cannot be sealed, which is the mechanism by which
   the N1/N2/N3 gaps are closed by construction rather than by discipline.
3. The seven-link chain returns the *first* crossed threshold, in the design's
   order, and a later, worse link never masks an earlier one.
4. The acceptance clause reads behavioural fields only.  A cell with a perfect
   boundary and a broken free-run must fail; a cell with a terrible boundary and
   an intact free-run must not be failed by the boundary.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    CONTROL_CELL_ID,
    CellId,
    MatrixCellReceiptError,
    append_ledger,
    blank_receipt,
    campaign_pass,
    first_divergent_metric,
    known_cell_ids,
    ledger_line,
    probe_set_sha256,
    read_ledger,
    seal_receipt,
    timeout_receipt,
    verify_seal,
    write_receipt,
)


def _renderer():
    path = _ROOT / "experiments" / "render_matrix_ledger.py"
    spec = importlib.util.spec_from_file_location("render_matrix_ledger", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _receipt(**overrides):
    body = blank_receipt(
        cell_id=overrides.pop("cell_id", "C13.T5.M16.L05"),
        cell_instance_id=overrides.pop("instance", "0" * 32),
        campaign_wave=1,
        repo_head="dee1c952",
        src_tree_sha="a" * 64,
        probe_set_sha=probe_set_sha256(("probe",)),
        smoke_prompt_sha=probe_set_sha256(("smoke",)),
        compute_architecture=overrides.pop(
            "compute_architecture", "sm86 NVIDIA GeForce RTX 3090"
        ),
    )
    body["calibration"] = {
        "copy_exact_reproduction_relL2": 1e-9,
        "passed": True,
        "applicable": True,
    }
    body["solve"].update(
        {
            "solver_path": "direct_gelsd_least_squares",
            "ridge_relative": 0.0,
            "condition_number": 12.5,
            "condition_cap_saturated": False,
            "prior_norm": 3.0,
            "correction_norm": 0.5,
            "requested_clamp_ratio": 0.16,
            "allowed_clamp_ratio": 1.0,
            "clamp_engaged": False,
        }
    )
    body["boundary"] = {
        "applicable": True,
        "local_solve_applicable": True,
        "macrocell_endpoint": "residual.post_layer",
        "write_heldout_relL2": 0.02,
        "write_heldout_cos": 0.999,
        "post_operator_heldout_relL2": 0.05,
        "post_operator_heldout_cos": 0.998,
        "endpoint_band_le_0p4": True,
    }
    body["composed"] = {
        "applicable": True,
        "terminal_heldout_relL2": 0.004,
        "emit_zero_floor_passed": True,
        "vs_copy_exact_control_delta": 0.004,
        "norm_ratio_per_layer_minmax": [0.98, 1.02],
    }
    body["behavior"] = {
        "free_run": {
            "probe_prompts": 8,
            "tokens_per_prompt": 64,
            "token_match_vs_control": 0.98,
            "mean_lcp": 0.9,
        },
        "decode_smoke": [
            {"prompt_sha": "b" * 64, "output_verbatim": "hello", "terminated": "eos"}
        ],
    }
    body["outcome"]["decode_smoke_review"] = "coherent"
    for path, value in overrides.items():
        node = body
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return body


# ---------------------------------------------------------------- cell ids


def test_cell_id_round_trips_every_designed_form():
    assert CellId.parse("C13.T5.M16.L05").text == "C13.T5.M16.L05"
    assert CellId.parse("CAL.C01.T5.X4").text == "CAL.C01.T5.X4"
    assert CellId.parse(CONTROL_CELL_ID).text == CONTROL_CELL_ID
    assert CellId.parse("C14.T1.NA.L00").text == "C14.T1.NA.L00"
    assert CellId.parse("C14.T1.NA.FINAL").layer_index is None
    assert CellId.parse("C13.T5.M16.L05").layer_index == 5


def test_cell_id_rejects_off_axis_cells():
    # A site the *schema version* does not register is not a cell.  Under v1
    # that was §1.1's chosen three sites per body component, so L07 was refused
    # rather than silently measured; ADDENDUM 7's v2 registers L07 for an MLP
    # component and still refuses a layer that does not exist and a component
    # whose block kind is not at that layer.  Both are asserted, because the
    # widening is only sound if it widened to the architecture and not to
    # everything.
    with pytest.raises(MatrixCellReceiptError):
        CellId.parse("C13.T5.M16.L07", schema_version=1)
    assert CellId.parse("C13.T5.M16.L07", schema_version=2).site == "L07"
    with pytest.raises(MatrixCellReceiptError):
        CellId.parse("C13.T5.M16.L14")  # the parent has fourteen layers, 0..13
    with pytest.raises(MatrixCellReceiptError):
        CellId.parse("C07.T5.M16.L05")  # attn_k at a short_conv layer
    with pytest.raises(MatrixCellReceiptError):
        CellId.parse("C05.T8.M16.L04")  # conv_out_proj at an attention layer
    with pytest.raises(MatrixCellReceiptError):
        CellId.parse("C99.T5.M16.L05")
    with pytest.raises(MatrixCellReceiptError):
        CellId.parse("C13.T99.M16.L05")
    with pytest.raises(MatrixCellReceiptError):
        CellId.parse("C13.T5.M32.L05")
    with pytest.raises(MatrixCellReceiptError):
        CellId.parse("C01.T5.X4.L05")  # global component with a site
    with pytest.raises(MatrixCellReceiptError):
        CellId.parse("C14.T1.M16.L00")  # norms have no level axis
    with pytest.raises(MatrixCellReceiptError):
        CellId.parse("C13.T5.NA.L05")  # only norms may be NA


def test_known_cell_ids_all_reparse():
    ids = known_cell_ids()
    assert len(ids) > 200
    for text in ids:
        assert CellId.parse(text).text == text


# ------------------------------------------------------------- fail closed


def test_missing_mandatory_field_cannot_be_sealed():
    body = _receipt()
    del body["solve"]["condition_number"]
    with pytest.raises(MatrixCellReceiptError, match="solve.condition_number"):
        seal_receipt(body)


def test_missing_clamp_ratios_break_the_chain_rather_than_defaulting():
    body = _receipt()
    body["solve"]["clamp_engaged"] = True
    body["solve"]["requested_clamp_ratio"] = None
    with pytest.raises(MatrixCellReceiptError, match="requested/allowed"):
        first_divergent_metric(body)


def test_null_free_run_is_never_skipped():
    body = _receipt()
    body["behavior"]["free_run"]["token_match_vs_control"] = None
    with pytest.raises(MatrixCellReceiptError, match="only acceptance signal"):
        first_divergent_metric(body)


def test_bad_termination_word_is_rejected():
    body = _receipt()
    body["behavior"]["decode_smoke"][0]["terminated"] = "truncated"
    with pytest.raises(MatrixCellReceiptError, match="terminated"):
        seal_receipt(body)


# ------------------------------------------------------------- the 7 links


def test_healthy_cell_has_no_divergence():
    assert first_divergent_metric(_receipt()) is None


def test_chain_order_infra_masks_everything_below_it():
    body = _receipt()
    body["calibration"]["passed"] = False
    body["boundary"]["write_heldout_cos"] = 0.1
    body["behavior"]["free_run"]["token_match_vs_control"] = 0.0
    assert first_divergent_metric(body) == "infra"


def test_chain_order_conditioning_precedes_local_solve():
    body = _receipt()
    body["solve"]["condition_cap_saturated"] = True
    body["boundary"]["write_heldout_cos"] = 0.1
    assert first_divergent_metric(body) == "solve-conditioning"


def test_clamp_only_counts_when_it_asked_for_more_than_four_times_the_budget():
    body = _receipt()
    body["solve"]["clamp_engaged"] = True
    body["solve"]["requested_clamp_ratio"] = 3.5
    body["solve"]["allowed_clamp_ratio"] = 1.0
    assert first_divergent_metric(body) is None
    body["solve"]["requested_clamp_ratio"] = 4.5
    assert first_divergent_metric(body) == "solve-conditioning"


def test_local_solve_then_macrocell_then_composition():
    body = _receipt()
    body["boundary"]["write_heldout_cos"] = 0.94
    assert first_divergent_metric(body) == "local-solve"
    body["boundary"]["write_heldout_cos"] = 0.96
    body["boundary"]["post_operator_heldout_relL2"] = 0.41
    assert first_divergent_metric(body) == "macrocell-band"
    body["boundary"]["post_operator_heldout_relL2"] = 0.39
    body["composed"]["vs_copy_exact_control_delta"] = 0.021
    assert first_divergent_metric(body) == "composition"


def test_free_run_is_the_f3_signature_after_every_numeric_link_passes():
    body = _receipt()
    body["behavior"]["free_run"]["token_match_vs_control"] = 0.89
    assert first_divergent_metric(body) == "free-run"
    sealed = seal_receipt(body)
    assert sealed["outcome"]["status"] == "fail"
    assert sealed["outcome"]["first_divergent_metric"] == "free-run"


def test_declared_inapplicable_links_are_skipped_but_absent_ones_are_not():
    body = _receipt()
    body["boundary"]["applicable"] = False
    body["boundary"]["local_solve_applicable"] = False
    body["boundary"]["write_heldout_cos"] = None
    body["boundary"]["post_operator_heldout_relL2"] = None
    assert first_divergent_metric(body) is None
    body["boundary"]["applicable"] = True
    body["boundary"]["local_solve_applicable"] = True
    with pytest.raises(MatrixCellReceiptError):
        first_divergent_metric(body)


def test_calibration_can_be_declared_inapplicable_only_with_a_reason():
    """NOTES-MX-E §4.5: four cells were void for a *harness* reason.

    C03/C04/C10/C14 have no ``held_input @ parent_weight.T`` predictor, so the
    generic gate is a category error rather than a failed measurement.  The
    receipt may say so -- and then the cell's verdict has to come from the
    links that need no per-component predictor -- but saying so costs a reason,
    because a bare ``false`` here would silently retire the chain's first link
    for every cell that copied the pattern.
    """

    body = _receipt()
    body["calibration"] = {
        "copy_exact_reproduction_relL2": None,
        "passed": False,
        "applicable": True,
    }
    assert first_divergent_metric(body) == "infra"

    body["calibration"]["applicable"] = False
    with pytest.raises(MatrixCellReceiptError, match="inapplicable_reason"):
        first_divergent_metric(body)

    body["calibration"]["inapplicable_reason"] = (
        "the RMSNorm gain is 1-D; held_input @ gain.T collapses the rows."
    )
    # Link 1 is now skipped and the verdict comes from further down the chain.
    assert first_divergent_metric(body) is None
    body["composed"]["vs_copy_exact_control_delta"] = 0.9
    assert first_divergent_metric(body) == "composition"


def test_local_solve_can_be_declared_inapplicable_without_losing_the_macrocell():
    """The boundary's two links are declared separately because they differ.

    The write-cosine link needs a per-component predictor scored against a
    named write endpoint; the macrocell link needs only a delta that composes
    into the residual.  A component can have the second and not a *calibrated*
    version of the first, and wave 1 lost both to a single flag.
    """

    body = _receipt()
    body["boundary"]["write_heldout_cos"] = 0.10  # would be a local-solve failure
    assert first_divergent_metric(body) == "local-solve"

    body["boundary"]["local_solve_applicable"] = False
    with pytest.raises(MatrixCellReceiptError, match="local_solve_inapplicable_reason"):
        first_divergent_metric(body)

    body["boundary"]["local_solve_inapplicable_reason"] = (
        "attn_per_head owns four tensors behind a softmax."
    )
    # Link 3 is skipped; link 4 is still measured and still fires.
    assert first_divergent_metric(body) is None
    body["boundary"]["post_operator_heldout_relL2"] = 0.9
    assert first_divergent_metric(body) == "macrocell-band"


def test_local_solve_cannot_be_applicable_inside_a_boundary_that_never_ran():
    body = _receipt()
    body["boundary"]["applicable"] = False
    with pytest.raises(MatrixCellReceiptError, match="boundary stage did not run"):
        first_divergent_metric(body)


def test_compute_architecture_is_mandatory_and_nonempty():
    """NOTES-MX-E §2: the one cell input the sealed identities did not cover.

    A9's fp32 OMP candidate search is architecture-non-portable, so a cell
    measured on sm_61 and a cell measured on sm_86 are not comparable -- and
    wave 1 came within one evidence-loading crash of merging five P40 rows into
    a ledger that could not have said which was which.
    """

    body = _receipt()
    assert body["code_identity"]["compute_architecture"] == "sm86 NVIDIA GeForce RTX 3090"
    assert seal_receipt(body)["code_identity"]["compute_architecture"].startswith("sm86")

    del body["code_identity"]["compute_architecture"]
    with pytest.raises(
        MatrixCellReceiptError, match="code_identity.compute_architecture"
    ):
        seal_receipt(body)

    blank = _receipt(compute_architecture="   ")
    with pytest.raises(MatrixCellReceiptError, match="compute_architecture is empty"):
        seal_receipt(blank)


def test_ledger_row_carries_the_architecture_and_the_storage_columns():
    body = _receipt()
    body["solve"]["stored_parameter_count"] = 1_966_080
    body["solve"]["parent_parameter_count"] = 1_048_576
    row = ledger_line(seal_receipt(body))
    assert row["compute_architecture"] == "sm86 NVIDIA GeForce RTX 3090"
    assert row["stored_parameter_count"] == 1_966_080
    assert row["parent_parameter_count"] == 1_048_576
    # The trap NOTES-MX-E §4.1 measured: a "contraction" that stores 1.88x the
    # dense tensor it replaced.  The ledger now has a column that shows it.
    assert row["stored_parameter_ratio"] == pytest.approx(1.875)
    assert row["calibration_applicable"] is True
    assert row["local_solve_applicable"] is True

    # A cell whose technique installed no tensor reports null, never zero.
    plain = ledger_line(seal_receipt(_receipt(instance="9" * 32)))
    assert plain["stored_parameter_count"] is None
    assert plain["stored_parameter_ratio"] is None


def test_m2_is_the_break_even_level_and_parses():
    from neural_forge.matrix_cell_receipt import LEVELS

    assert LEVELS["M2"] == 0.5
    assert CellId.parse("C05.T1.M2.L05").level == "M2"
    assert "C05.T1.M2.L05" in known_cell_ids(components=("C05",), techniques=("T1",))


# -------------------------------------------------------- acceptance clause


def test_acceptance_reads_behaviour_not_boundary():
    # A wrecked boundary with intact behaviour still passes the §5 rule 1
    # clause; boundary metrics appear nowhere in it.
    body = _receipt()
    body["boundary"]["write_heldout_cos"] = 0.10
    body["boundary"]["post_operator_heldout_relL2"] = 9.0
    body["boundary"]["endpoint_band_le_0p4"] = False
    assert campaign_pass(body) is True
    # ... and a perfect boundary with a broken free-run does not.
    good = _receipt()
    good["behavior"]["free_run"]["token_match_vs_control"] = 0.5
    assert campaign_pass(good) is False


def test_pending_review_is_not_a_pass():
    body = _receipt()
    body["outcome"]["decode_smoke_review"] = "pending"
    sealed = seal_receipt(body)
    assert sealed["outcome"]["status"] == "pending-review"
    assert campaign_pass(sealed) is False


def test_incoherent_review_is_the_behavior_link():
    body = _receipt()
    body["outcome"]["decode_smoke_review"] = "incoherent"
    sealed = seal_receipt(body)
    assert sealed["outcome"]["first_divergent_metric"] == "behavior"
    assert sealed["outcome"]["status"] == "fail"


def test_emit_zero_floor_is_necessary_when_composition_ran():
    body = _receipt()
    body["composed"]["emit_zero_floor_passed"] = False
    assert campaign_pass(body) is False
    body["composed"]["applicable"] = False
    assert campaign_pass(body) is True


# --------------------------------------------------------------- the seal


def test_seal_is_reproducible_and_tamper_evident():
    sealed = seal_receipt(_receipt())
    assert verify_seal(sealed) == sealed["receipt_sha"]
    tampered = json.loads(json.dumps(sealed))
    tampered["behavior"]["free_run"]["token_match_vs_control"] = 1.0
    with pytest.raises(MatrixCellReceiptError, match="seal mismatch"):
        verify_seal(tampered)


def test_timeout_receipt_is_still_a_complete_receipt():
    sealed = timeout_receipt(
        _receipt(), wall_time_seconds=1201.0, stage="solve", hard_cap_seconds=1200.0
    )
    verify_seal(sealed)
    assert sealed["outcome"]["status"] == "timeout"
    assert "hard cap" in sealed["outcome"]["notes"]
    assert ledger_line(sealed)["status"] == "timeout"


# --------------------------------------------------------- ledger + render


def test_ledger_is_append_only_and_rerenders(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    first = seal_receipt(_receipt(instance="1" * 32))
    second = seal_receipt(
        _receipt(
            instance="2" * 32,
            **{"behavior.free_run.token_match_vs_control": 0.2},
        )
    )
    control = _receipt(cell_id=CONTROL_CELL_ID, instance="3" * 32)
    control["boundary"]["applicable"] = False
    control["boundary"]["local_solve_applicable"] = False
    control["boundary"]["write_heldout_cos"] = None
    control["boundary"]["post_operator_heldout_relL2"] = None
    control["boundary"]["endpoint_band_le_0p4"] = None
    control["behavior"]["free_run"]["token_match_vs_control"] = 1.0
    control = seal_receipt(control)
    for body in (control, first, second):
        append_ledger(ledger, body)
    rows = read_ledger(ledger)
    assert len(rows) == 3
    assert [r["cell_id"] for r in rows][0] == CONTROL_CELL_ID

    report = _renderer().render(rows, ledger_path=ledger)
    assert "# Stage-1 matrix campaign -- ledger" in report
    assert "C13" in report
    assert "free-run" in report
    assert "Band-versus-behavior agreement" in report
    # Both instances of the re-run cell survive in the drill-down.
    assert report.count("11111111") == 1
    assert report.count("22222222") == 1


def test_ledger_bytes_are_platform_independent(tmp_path):
    """One append-only file, one byte sequence, on every host that appends.

    The campaign's ledger lives on the Linux run host and is mirrored into the
    repository; reviews are sealed from a Windows workstation.  Appending
    through a text-mode handle translated ``\\n`` to ``\\r\\n`` there, so the
    two copies of an append-only file carried identical rows and different
    bytes.  `read_ledger` splits on either, which is exactly why the divergence
    survived until the mirror was digested against the original.
    """

    ledger = tmp_path / "ledger.jsonl"
    for instance in ("1" * 32, "2" * 32):
        append_ledger(ledger, seal_receipt(_receipt(instance=instance)))
    raw = ledger.read_bytes()
    assert b"\r" not in raw
    assert raw.count(b"\n") == 2
    assert raw.endswith(b"\n")
    assert len(read_ledger(ledger)) == 2


def test_render_survives_a_traceback_shaped_note(tmp_path):
    # Receipt notes carry tracebacks and JSON tails verbatim; a raw newline or
    # pipe inside a Markdown table cell destroys the table silently, which is
    # exactly the failure mode this report exists to avoid.
    ledger = tmp_path / "ledger.jsonl"
    body = _receipt(cell_id="CAL.C01.T5.X4", instance="4" * 32)
    body["calibration"]["passed"] = False
    body["outcome"]["notes"] = (
        'CellTechniqueError: route exited 1.  Tail:\n{"a":1|2}\nTraceback...\n'
        + "x" * 3000
    )
    append_ledger(ledger, seal_receipt(body))
    report = _renderer().render(read_ledger(ledger), ledger_path=ledger)
    table_rows = [
        line
        for line in report.splitlines()
        if line.startswith("|") and "CellTechniqueError" in line
    ]
    assert table_rows, "the calibration row must still show the note"
    for line in table_rows:
        assert "\n" not in line
        # Four columns -> five *unescaped* pipes.  A raw pipe carried in from
        # the note would silently add a column and shear the table.
        assert len(re.findall(r"(?<!\\)\|", line)) == 5
        assert r"\|" in line
    # The full note survives as a blockquote, truncated with an explicit count.
    quoted = [line for line in report.splitlines() if line.startswith("> ")]
    assert any("CellTechniqueError" in line for line in quoted)
    assert "more characters in the receipt" in report


def test_ledger_rejects_foreign_rows(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"schema":"something-else"}\n', encoding="utf-8")
    with pytest.raises(MatrixCellReceiptError):
        read_ledger(ledger)


def test_write_receipt_refuses_an_unsealed_body(tmp_path):
    with pytest.raises(MatrixCellReceiptError, match="not sealed"):
        write_receipt(tmp_path / "cell-receipt.json", _receipt())
