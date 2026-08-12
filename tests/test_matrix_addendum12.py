#!/usr/bin/env python3
"""ADDENDUM 12 -- a standard-shape child can carry a sealed machine verdict.

`NOTES-MX-R` §6 is the gap: wave 14 built three children that stock
`transformers` loads and that are architecturally smaller, graded them by hand,
and appended nothing, because a narrowed child replaces no parent tensor
shape-for-shape and is not a `CellId` under receipt schema v1 or v2.  Every
assertion here is about the narrowest extension that closes it and about the
things the extension must keep refusing.

The load-bearing ones:

* v1 and v2 do not acquire the new id form (a sealed row is never re-read under
  a rule it was not sealed under, in both directions);
* an ARCH receipt without a complete architecture block does not seal;
* the sealed ratio must be the loader's own count, and must equal the ledger's
  ``stored_parameter_ratio`` -- the storage-format confusion `NOTES-MX-R` §1
  measured cannot be re-introduced in a new field;
* a non-ARCH cell may not carry the block, and an ARCH cell may not claim a
  loader it did not run;
* the ADDENDUM 6 behaviour verdict seals on an ARCH receipt exactly as it does
  on every other kind, which is the whole point of the addendum.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    ARCHITECTURE_MIN_SCHEMA_VERSION,
    CELL_KINDS,
    LATEST_SCHEMA_VERSION,
    SCHEMA_V3,
    CellId,
    MatrixCellReceiptError,
    blank_receipt,
    ledger_line,
    probe_set_sha256,
    seal_behavior_review,
    seal_receipt,
    validate_receipt,
    verify_seal,
)

PARENT_PARAMETERS = 229_693_184
CHILD_PARAMETERS = 147_117_824


def _architecture_block(**overrides):
    block = {
        "schema": "neural-forge-lfm2-matrix-addendum12-standard-shape-child-v1",
        "standard_shape": True,
        "tag": "FF640",
        "rung": None,
        "child_directory": "/mnt/ai_data/neural-forge-matrix/wave15/children/MX15",
        "child_tensors_sha256": "c" * 64,
        "build_receipt_sha256": "d" * 64,
        "config_diff": [
            {"field": "intermediate_size", "parent": 2560, "child": 640},
            {"field": "block_ff_dim", "parent": 2560, "child": 640},
        ],
        "export": {
            "loader": "transformers.Lfm2ForCausalLM.from_pretrained",
            "trust_remote_code": False,
            "loaded_class": "Lfm2ForCausalLM",
            "parameters_counted_live": CHILD_PARAMETERS,
        },
        "storage": {
            "parent_parameter_count": PARENT_PARAMETERS,
            "child_parameter_count": CHILD_PARAMETERS,
            "net_ratio": CHILD_PARAMETERS / PARENT_PARAMETERS,
        },
        "axes": [
            {
                "axis": "intermediate_size",
                "kind": "mlp",
                "sites": list(range(14)),
                "config_fields": ["intermediate_size", "block_ff_dim"],
                "selection": "pivoted-cholesky coreset on conditioned features",
                "solve": "prior-anchored ridge, lambda ladder to update_scale<=1",
                "evidence": "conditioned capture through the partial child",
                "parameters_removed": 82_575_360,
                "heldout": {"relL2": 0.51, "cos": 0.87},
            }
        ],
        "passed_through": ["head_dim", "num_hidden_layers"],
    }
    block.update(overrides)
    return block


def _receipt(cell_id: str = "ARCH.FF640", *, architecture=None, **body_overrides):
    body = blank_receipt(
        cell_id=cell_id,
        cell_instance_id="0" * 32,
        campaign_wave=15,
        repo_head="f86ebf0f",
        src_tree_sha="a" * 64,
        probe_set_sha=probe_set_sha256(("probe",)),
        smoke_prompt_sha=probe_set_sha256(("smoke",)),
        compute_architecture="sm86 NVIDIA GeForce RTX 3090",
        schema_version=ARCHITECTURE_MIN_SCHEMA_VERSION,
    )
    body["calibration"] = {
        "copy_exact_reproduction_relL2": None,
        "passed": False,
        "applicable": False,
        "inapplicable_reason": "a shape change replaces no parent tensor",
    }
    body["solve"].update(
        {
            "solver_path": "mx15_narrow.conditioned",
            "ridge_relative": 1e-6,
            "condition_number": 1.0,
            "stored_parameter_count": CHILD_PARAMETERS,
            "parent_parameter_count": PARENT_PARAMETERS,
        }
    )
    body["composed"] = {
        "applicable": True,
        "terminal_heldout_relL2": 0.42,
        "emit_zero_floor_passed": True,
        "vs_copy_exact_control_delta": 0.42,
        "norm_ratio_per_layer_minmax": [0.81, 1.11],
    }
    body["behavior"] = {
        "free_run": {
            "probe_prompts": 8,
            "tokens_per_prompt": 64,
            "token_match_vs_control": 0.11,
            "mean_lcp": 0.02,
        },
        "decode_smoke": [
            {"prompt_sha": "b" * 64, "output_verbatim": "hello", "terminated": "eos"}
        ],
    }
    if architecture is not False:
        body["architecture"] = architecture or _architecture_block()
    body.update(body_overrides)
    return body


# --------------------------------------------------------------------------
# 1. the id form, and the versions it is legal under
# --------------------------------------------------------------------------


def test_the_new_kind_and_version_exist():
    assert "architecture" in CELL_KINDS
    assert ARCHITECTURE_MIN_SCHEMA_VERSION == 3
    assert LATEST_SCHEMA_VERSION == ARCHITECTURE_MIN_SCHEMA_VERSION


def test_arch_ids_parse_with_and_without_a_rung():
    plain = CellId.parse("ARCH.FF640", schema_version=3)
    assert plain.kind == "architecture"
    assert plain.text == "ARCH.FF640"
    assert (plain.component, plain.technique, plain.level, plain.site) == (
        "ARCH",
        "architecture",
        "NA",
        "GLOBAL",
    )
    rung = CellId.parse("ARCH.FF640.HD16", schema_version=3)
    assert rung.text == "ARCH.FF640.HD16"
    assert rung.architecture_rung == "HD16"


def test_v1_and_v2_never_acquire_the_new_id_form():
    for version in (1, 2):
        with pytest.raises(MatrixCellReceiptError):
            CellId.parse("ARCH.FF640", schema_version=version)


def test_a_v2_body_cannot_declare_an_arch_cell():
    body = _receipt()
    body["schema"] = "neural-forge-lfm2-matrix-cell-receipt-v2"
    with pytest.raises(MatrixCellReceiptError):
        validate_receipt(body)


# --------------------------------------------------------------------------
# 2. the block is mandatory, complete, and exclusive
# --------------------------------------------------------------------------


def test_a_complete_arch_receipt_seals_and_reverifies():
    sealed = seal_receipt(_receipt())
    assert sealed["schema"] == SCHEMA_V3
    assert sealed["cell_kind"] == "architecture"
    assert sealed["component"] == "standard_shape_child"
    assert verify_seal(sealed) == sealed["receipt_sha"]


def test_an_arch_cell_without_the_block_is_refused():
    with pytest.raises(MatrixCellReceiptError):
        validate_receipt(_receipt(architecture=False))


def test_a_non_arch_cell_may_not_carry_the_block():
    from neural_forge.matrix_cell_receipt import SCHEMA_V2

    body = _receipt()
    body["schema"] = SCHEMA_V2
    body["cell_id"] = "C13.T8.S4.L05"
    body["cell_kind"] = "cell"
    body["component"] = "mlp_down"
    body["technique"] = "gram_relation_matching"
    body["level"] = "S4"
    body["site"] = "L05"
    with pytest.raises(MatrixCellReceiptError):
        validate_receipt(body)


@pytest.mark.parametrize(
    "mutation",
    [
        {"standard_shape": False},
        {"config_diff": []},
        {"axes": []},
        {"passed_through": None},
        {
            "export": {
                "loader": "custom module",
                "trust_remote_code": True,
                "loaded_class": "Lfm2ForCausalLM",
                "parameters_counted_live": CHILD_PARAMETERS,
            }
        },
        {
            "config_diff": [
                {"field": "intermediate_size", "parent": 2560, "child": 2560}
            ]
        },
    ],
)
def test_incomplete_or_dishonest_blocks_are_refused(mutation):
    with pytest.raises(MatrixCellReceiptError):
        validate_receipt(_receipt(architecture=_architecture_block(**mutation)))


def test_the_ratio_must_be_its_own_arithmetic():
    block = _architecture_block()
    block["storage"]["net_ratio"] = 0.25
    with pytest.raises(MatrixCellReceiptError):
        validate_receipt(_receipt(architecture=block))


def test_the_ratio_must_be_the_count_the_loader_made():
    block = _architecture_block()
    block["export"]["parameters_counted_live"] = CHILD_PARAMETERS - 1
    with pytest.raises(MatrixCellReceiptError):
        validate_receipt(_receipt(architecture=block))


def test_the_solve_counts_must_match_the_block():
    body = _receipt()
    body["solve"]["stored_parameter_count"] = 1234
    with pytest.raises(MatrixCellReceiptError):
        validate_receipt(body)


# --------------------------------------------------------------------------
# 3. the ledger, and the verdict the addendum exists to make sealable
# --------------------------------------------------------------------------


def test_the_ledger_row_carries_the_net_ratio_and_the_new_column():
    row = ledger_line(seal_receipt(_receipt()))
    assert row["standard_shape"] is True
    assert row["cell_kind"] == "architecture"
    assert row["stored_parameter_count"] == CHILD_PARAMETERS
    assert row["stored_parameter_ratio"] == pytest.approx(
        CHILD_PARAMETERS / PARENT_PARAMETERS
    )


def test_rows_sealed_before_the_addendum_read_as_unstated():
    """A row with no architecture block reads ``None``, never ``False``.

    ``None`` is "this child was never asked whether it is a smaller model";
    ``False`` would be the campaign asserting the answer for 461 rows it did not
    measure, which is exactly the confusion `NOTES-MX-R` §1 is about.
    """

    body = _receipt()
    body.pop("architecture")
    body["schema"] = "neural-forge-lfm2-matrix-cell-receipt-v2"
    body["cell_id"] = "C13.T8.S4.L05"
    body["cell_kind"] = "cell"
    body["component"] = "mlp_down"
    body["technique"] = "gram_relation_matching"
    body["level"] = "S4"
    body["site"] = "L05"
    body["boundary"] = {
        "applicable": True,
        "local_solve_applicable": True,
        "macrocell_endpoint": "residual.post_layer",
        "write_heldout_relL2": 0.18,
        "write_heldout_cos": 0.86,
        "post_operator_heldout_relL2": 0.18,
        "post_operator_heldout_cos": 0.86,
        "endpoint_band_le_0p4": True,
    }
    row = ledger_line(seal_receipt(body))
    assert row["standard_shape"] is None
    assert row["receipt_schema_version"] == 2


def test_a_behaviour_verdict_seals_on_a_standard_shape_child():
    sealed = seal_receipt(_receipt())
    reviewed = seal_behavior_review(
        sealed,
        verdict="impaired",
        reviewer="test",
        useful=1,
        partial=4,
        observed_miss=10,
        parent_useful_or_partial=12,
        dossier_sha256="e" * 64,
        protocol_sha256="f" * 64,
        prompt_pack_sha256="0" * 64,
        smoke_verdict="coherent",
    )
    assert reviewed["addendum6"]["verdict"] == "impaired"
    assert reviewed["addendum6"]["useful_or_partial"] == 5
    # The frontier column a standard-shape child exists to move.
    assert reviewed["addendum6"]["stored_parameter_ratio"] == pytest.approx(
        CHILD_PARAMETERS / PARENT_PARAMETERS
    )
    assert verify_seal(reviewed) == reviewed["receipt_sha"]
    assert ledger_line(reviewed)["standard_shape"] is True
