"""Contract tests for ADDENDUM 6: behaviour-first grading and the X4 proof bar.

ADDENDUM 6 reopens phase 1 on two rulings, and both are the kind of rule that
decays into a slogan unless something mechanical holds it:

1. **Token match gates nothing.**  A cell whose free-run is 0.55 -- the exact
   number wave 4's `COMP.MA.RSTO` sealed while scoring the parent's own 12/15 --
   must be able to reach `alive-parent-band`, and a cell at 0.99 free-run whose
   review collapses must reach `dead`.  If the verdict were even partly a
   function of the token metric these two tests would disagree with it.
2. **A 4x claim is a storage claim.**  The bound is checked against the
   receipt's own counts, and the storage models the campaign plans widths with
   are pinned against *sealed* wave-1..3 receipts, so a model that drifts from
   what a kernel really stores is a test failure rather than a silently wrong
   plan.

The third thing pinned here is the boundary between them: `cannot-express-4x`
is a statement about a technique's algebra and must not carry review counts,
because no model was ever run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    A9_REVIEW_BAND,
    LEVELS,
    PARENT_REVIEW_BAND,
    STORAGE_LEVEL_RATIOS,
    CellId,
    MatrixCellReceiptError,
    behavior_verdict_for,
    blank_receipt,
    campaign_pass,
    first_divergent_metric,
    ledger_line,
    probe_set_sha256,
    seal_behavior_review,
    seal_receipt,
    verify_seal,
)
from neural_forge.matrix_technique_registry import (  # noqa: E402
    CellTechniqueError,
    ComponentSpec,
    storage_model,
    storage_width_for_ratio,
)


def _receipt(**overrides):
    body = blank_receipt(
        cell_id=overrides.pop("cell_id", "C13.T5.S4.L05"),
        cell_instance_id=overrides.pop("instance", "0" * 32),
        campaign_wave=5,
        repo_head="9626fe33",
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
            "solver_path": "t5_body_route",
            "ridge_relative": 0.0,
            "condition_number": 12.5,
            "condition_cap_saturated": False,
            "prior_norm": 3.0,
            "correction_norm": 0.5,
            "requested_clamp_ratio": 0.16,
            "allowed_clamp_ratio": 1.0,
            "clamp_engaged": False,
            "stored_parameter_count": 655_360,
            "parent_parameter_count": 2_621_440,
            "storage_bound_ratio": 0.25,
            "storage_bound_parameters": 655_360,
            "storage_bound_satisfied": True,
            "installed_tensor_digests": ["model.layers.5.feed_forward.w2.weight=" + "c" * 64],
            "installed_tensors_sha256": "d" * 64,
        }
    )
    body["boundary"] = {
        "applicable": True,
        "local_solve_applicable": True,
        "macrocell_endpoint": "residual.post_layer",
        "write_heldout_relL2": 0.2,
        "write_heldout_cos": 0.99,
        "post_operator_heldout_relL2": 0.09,
        "post_operator_heldout_cos": 0.995,
        "endpoint_band_le_0p4": True,
    }
    # Deliberately under the composition threshold so the pre-addendum chain
    # walks all the way to free-run: the point of the fixture is that the OLD
    # chain still reads the token metric and still fails the cell on it, while
    # the new verdict reads the text and does not.
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
            # Wave 4's number, on purpose.
            "token_match_vs_control": 0.5508,
            "mean_lcp": 0.52,
        },
        "decode_smoke": [
            {"prompt_sha": "b" * 64, "output_verbatim": "hello", "terminated": "eos"}
        ],
    }
    for path, value in overrides.items():
        node = body
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return seal_receipt(body)


def _spec(component_id: str, out: int, inp: int, axis: str) -> ComponentSpec:
    return ComponentSpec(
        component_id=component_id,
        name="test",
        kind="mlp",
        site="L05",
        layer_index=5,
        weight_keys=("w",),
        contraction_axis=axis,
        parent_output_width=out,
        parent_input_width=inp,
        parent_parameter_count=out * inp,
        input_endpoint="mlp.input",
        write_endpoint="mlp.gate",
        macrocell_endpoint="residual.post_layer",
    )


# --------------------------------------------------------------------------
# 1. the levels themselves
# --------------------------------------------------------------------------


def test_storage_levels_are_bounds_not_widths():
    for level, ratio in STORAGE_LEVEL_RATIOS.items():
        assert level in LEVELS
        assert LEVELS[level] is None, (
            f"{level} must have no width fraction: it bounds what is stored, "
            "not how wide a factor is."
        )
    assert STORAGE_LEVEL_RATIOS["S4"] == 0.25
    # The marginal family is untouched, so every wave-1..4 receipt still parses.
    assert LEVELS["M16"] == 15.0 / 16.0
    assert CellId.parse("C13.T5.S4.L05").level == "S4"
    assert CellId.parse("C13.T3.M16.L05").level == "M16"


# --------------------------------------------------------------------------
# 2. the verdict is arithmetic over the review, and blind to token match
# --------------------------------------------------------------------------


def test_parent_band_verdict_at_wave4s_own_token_match():
    """0.5508 free-run + 12/15 must be able to reach the parent band.

    This is not a hypothetical pairing: it is `COMP.MA.RSTO`, sealed in wave 4
    at exactly this free-run with exactly this review score against a parent
    that scored the same 12/15 in the same process.  §5 rule 1 called it a
    fail; ADDENDUM 6 exists because the frozen review called it parent-level.
    """

    verdict = behavior_verdict_for(
        useful_or_partial=12,
        coherent=True,
        emit_zero_floor_passed=True,
        degenerate=False,
    )
    assert verdict == "alive-parent-band"
    sealed = seal_behavior_review(
        _receipt(),
        verdict=verdict,
        reviewer="test",
        useful=6,
        partial=6,
        observed_miss=3,
        parent_useful_or_partial=12,
        dossier_sha256="e" * 64,
        protocol_sha256="f" * 64,
        prompt_pack_sha256="0" * 64,
        smoke_verdict="coherent",
    )
    verify_seal(sealed)
    assert sealed["addendum6"]["verdict"] == "alive-parent-band"
    assert sealed["addendum6"]["band_delta_vs_parent"] == 0
    # The pre-addendum chain is untouched: it still reads 0.5508 and still says
    # free-run.  Both readings are sealed and they are about the same child.
    assert sealed["outcome"]["first_divergent_metric"] == "free-run"
    assert campaign_pass(sealed) is False


def test_high_token_match_with_degenerate_output_is_dead():
    """The metric cannot rescue a cell either, which is the same rule."""

    assert (
        behavior_verdict_for(
            useful_or_partial=14,
            coherent=True,
            emit_zero_floor_passed=True,
            degenerate=True,
        )
        == "dead"
    )
    assert (
        behavior_verdict_for(
            useful_or_partial=14,
            coherent=False,
            emit_zero_floor_passed=True,
            degenerate=False,
        )
        == "dead"
    )
    assert (
        behavior_verdict_for(
            useful_or_partial=14,
            coherent=True,
            emit_zero_floor_passed=False,
            degenerate=False,
        )
        == "dead"
    )


def test_the_bands_are_the_addendums_bands():
    assert PARENT_REVIEW_BAND == 12
    assert A9_REVIEW_BAND == 8
    ladder = {
        15: "alive-parent-band",
        10: "alive-parent-band",  # parent - 2
        9: "alive-degraded",
        8: "alive-degraded",  # the A9 band itself
        7: "impaired",
        0: "impaired",
    }
    for score, expected in ladder.items():
        assert (
            behavior_verdict_for(
                useful_or_partial=score,
                coherent=True,
                emit_zero_floor_passed=True,
                degenerate=False,
            )
            == expected
        ), score


def test_review_counts_must_cover_the_fifteen_intelligence_cases():
    with pytest.raises(MatrixCellReceiptError, match="15 intelligence cases"):
        seal_behavior_review(
            _receipt(),
            verdict="alive-parent-band",
            reviewer="test",
            useful=6,
            partial=6,
            observed_miss=2,  # 14, not 15
            parent_useful_or_partial=12,
            dossier_sha256="e" * 64,
            protocol_sha256="f" * 64,
            prompt_pack_sha256="0" * 64,
            smoke_verdict="coherent",
        )


def test_cannot_express_4x_seals_no_review_counts():
    """No model ran, so a score would be a reading of a review that never was."""

    with pytest.raises(MatrixCellReceiptError, match="no model was run"):
        seal_behavior_review(
            _receipt(),
            verdict="cannot-express-4x",
            reviewer="test",
            useful=6,
            partial=6,
            observed_miss=3,
            parent_useful_or_partial=12,
            dossier_sha256=None,
            protocol_sha256=None,
            prompt_pack_sha256=None,
            smoke_verdict="pending",
        )
    sealed = seal_behavior_review(
        _receipt(),
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
    verify_seal(sealed)
    assert sealed["addendum6"]["useful_or_partial"] is None


def test_a_graded_verdict_must_name_the_dossier_it_graded():
    with pytest.raises(MatrixCellReceiptError, match="name the dossier"):
        seal_behavior_review(
            _receipt(),
            verdict="alive-parent-band",
            reviewer="test",
            useful=6,
            partial=6,
            observed_miss=3,
            parent_useful_or_partial=12,
            dossier_sha256=None,
            protocol_sha256=None,
            prompt_pack_sha256=None,
            smoke_verdict="coherent",
        )


def test_ledger_carries_the_verdict_and_stays_null_on_older_rows():
    plain = ledger_line(_receipt())
    assert plain["behavior_verdict"] is None
    assert plain["behavior_useful_or_partial"] is None
    assert plain["storage_bound_ratio"] == 0.25
    assert plain["storage_bound_satisfied"] is True
    reviewed = ledger_line(
        seal_behavior_review(
            _receipt(),
            verdict="alive-degraded",
            reviewer="test",
            useful=4,
            partial=5,
            observed_miss=6,
            parent_useful_or_partial=12,
            dossier_sha256="e" * 64,
            protocol_sha256="f" * 64,
            prompt_pack_sha256="0" * 64,
            smoke_verdict="coherent",
        )
    )
    assert reviewed["behavior_verdict"] == "alive-degraded"
    assert reviewed["behavior_useful_or_partial"] == 9
    assert reviewed["behavior_parent_useful_or_partial"] == 12


# --------------------------------------------------------------------------
# 3. the storage bar
# --------------------------------------------------------------------------


def test_factor_pair_storage_model_reproduces_a_sealed_wave2_receipt():
    """`C13.T3.M16.L05` sealed 8,601,600 stored against a 2,621,440 parent.

    The model is only worth planning with if it agrees with what a kernel
    really stored, so it is checked against the number a real cell sealed --
    the same 3.28x that made ADDENDUM 6 necessary.
    """

    sealed = json.loads(
        (
            _ROOT
            / "reports"
            / "matrix-campaign-20260810"
            / "cells"
            / "C13.T3.M16.L05"
            / "cell-receipt.json"
        ).read_text(encoding="utf-8")
    )
    spec = _spec("C13", out=1024, inp=2560, axis="input")
    assert spec.parent_parameter_count == sealed["solve"]["parent_parameter_count"]
    width = sealed["solve"]["contraction_basis_rank"]
    assert storage_model("T3", "C13")(spec, width) == (
        sealed["solve"]["stored_parameter_count"]
    )
    # ...and it is 3.28x, which is the whole point.
    assert storage_model("T3", "C13")(spec, width) > 3 * spec.parent_parameter_count


def test_a_factor_pair_technique_cannot_express_a_square_component_at_a_quarter():
    """r*(in+out) <= 0.25*in*out has no solution when in == out.

    On a square component the break-even is r < w/2 (`NOTES-MX-E` §4.1), so a
    quarter of the storage needs r < w/8 -- which exists.  The refusal case is
    the *narrow* one: C07's k_proj is 512x1024, where even r=1 stores 1,536
    against a 524,288 parent... so it does exist there too.  What genuinely
    cannot be expressed is a component whose two widths are so lopsided that
    one factor alone exceeds the budget, which is what this pins.
    """

    tiny = _spec("C07", out=8, inp=1024, axis="output")  # 8,192 parameters
    # r*(1032) <= 2048 -> r <= 1, which is admissible; the model is honest here.
    assert storage_width_for_ratio(tiny, "T1", 0.25) == 1
    lopsided = _spec("C07", out=2, inp=1024, axis="output")  # 2,048 parameters
    # r*(1026) <= 512 has no positive solution.
    assert storage_width_for_ratio(lopsided, "T1", 0.25) is None


def test_t5_writer_route_can_express_a_quarter_where_a_factor_pair_cannot():
    """The whole reason the MLP question is worth re-asking at S4.

    C13's mlp_down is 1024x2560.  A basis-plus-factor technique stores
    r*(3584), so a quarter needs r <= 182 of 2560 -- a 14x width contraction.
    T5's writer stores the narrow child weight plus its kept ids, so a quarter
    needs r <= 639 of 2560, which is 3.5x more width for the same storage.
    """

    spec = _spec("C13", out=1024, inp=2560, axis="input")
    factor = storage_width_for_ratio(spec, "T1", 0.25)
    route = storage_width_for_ratio(spec, "T5", 0.25)
    assert factor == 182
    assert route == 639
    assert storage_model("T1", "C13")(spec, factor) <= 0.25 * 2_621_440
    assert storage_model("T5", "C13")(spec, route) <= 0.25 * 2_621_440
    assert storage_model("T1", "C13")(spec, factor + 1) > 0.25 * 2_621_440
    assert storage_model("T5", "C13")(spec, route + 1) > 0.25 * 2_621_440


def test_t5_reader_storage_counts_the_index_side_of_the_route():
    """A route is its coefficients AND the ids that say where they go."""

    spec = _spec("C11", out=2560, inp=1024, axis="output")
    width = 600
    dropped = 2560 - width
    expected = width * 1024 + 2 * (dropped * 8) + 2561
    assert storage_model("T5", "C11")(spec, width) == expected
    # Counting only the coefficient values would understate the fold.
    assert expected > width * 1024 + dropped * 8


def test_technique_without_a_storage_model_refuses_rather_than_guesses():
    spec = _spec("C06", out=1024, inp=1024, axis="output")
    with pytest.raises(CellTechniqueError, match="publishes no storage model"):
        storage_width_for_ratio(spec, "T7", 0.25)


def test_storage_bound_block_is_null_for_a_marginal_cell():
    """A cell that made no storage claim reports none, which is not the same
    statement as a cell that claimed a quarter and missed."""

    runner = _runner()
    marginal = runner._storage_bound_block(
        level="M16", stored=8_601_600, parent=2_621_440, level_width=2400
    )
    assert marginal["storage_bound_ratio"] is None
    assert marginal["storage_bound_satisfied"] is None
    assert marginal["storage_realized_ratio"] == pytest.approx(3.28125)
    bounded = runner._storage_bound_block(
        level="S4", stored=655_360, parent=2_621_440, level_width=639
    )
    assert bounded["storage_bound_satisfied"] is True
    missed = runner._storage_bound_block(
        level="S4", stored=655_361, parent=2_621_440, level_width=640
    )
    assert missed["storage_bound_satisfied"] is False


def _runner():
    """Load the cell runner once per session, under its own name.

    The runner registers the A9 token route into the process-wide technique
    registry at import time, so importing it a second time under a second name
    is not a fresh import -- it is a duplicate registration, and the registry
    refuses one by design.  Reusing the module the rest of the suite already
    loaded is the only correct way to reach its helpers.
    """

    import importlib.util

    existing = sys.modules.get("run_lfm2_matrix_cell")
    if existing is not None:
        return existing
    path = _ROOT / "experiments" / "run_lfm2_matrix_cell.py"
    spec = importlib.util.spec_from_file_location("run_lfm2_matrix_cell", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_storage_level_requires_an_explicit_width():
    from neural_forge.matrix_technique_registry import resolve_level_width

    spec = _spec("C13", out=1024, inp=2560, axis="input")
    with pytest.raises(CellTechniqueError, match="bounds STORAGE"):
        resolve_level_width(spec, "S4")
    assert resolve_level_width(spec, "S4", x4_width=639) == 639
