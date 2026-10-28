"""CPU tests for c23 (the fold-proxy behaviour checks) on a synthetic c21 fold child."""
from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c21_v2_fold as c21  # noqa: E402
import c23_v2_fold_checks as c23  # noqa: E402
import test_c21_v2_fold as t21  # noqa: E402


@pytest.fixture(scope="module")
def folded(tmp_path_factory):
    state = t21.random_state(seed=3, final_norm_zero=False)
    cfg = t21.target_config(hidden=16, intermediate=12, layers=4, q_heads=2, kv_heads=1, kda_heads=2)
    calib = t21.calibration(rows=8, length=20, seed=5)
    fold = c21.Fold(t21.source_of(state), cfg, calib, c21.FoldOptions(device="cpu", batch=4))
    model, receipt = fold.run()
    out = tmp_path_factory.mktemp("fold") / "child"
    c21.save_checkpoint(model, out, receipt)
    return model, receipt, out, calib


def test_structural_passes_on_a_real_fold_and_refuses_a_doctored_receipt(folded):
    model, receipt, out, _ = folded
    _, r0 = c23.structural(out, receipt, torch.device("cpu"))
    assert r0["pass"], r0
    doctored = dict(receipt, copy_methods=1)
    _, bad = c23.structural(out, doctored, torch.device("cpu"))
    assert not bad["pass"]
    unnamed = dict(receipt, parameters_by_method={"chart": 1})
    _, bad = c23.structural(out, unnamed, torch.device("cpu"))
    assert not bad["pass"]


def test_held_nll_matches_a_direct_cross_entropy_and_embed_only_zeroes_the_writes(folded):
    model, _, _, calib = folded
    got = c23.held_nll(model, calib, torch.device("cpu"), chunk=7)
    r = calib.held_rows[0]
    length = int(calib.valid[r].sum())
    ids = calib.input_ids[r:r + 1, :length]
    with torch.no_grad():
        logits = model(ids, ledger_band_logit_mask=True)["logits"][0, :-1, : t21.VOCAB]
    want = float(torch.nn.functional.cross_entropy(logits, ids[0, 1:]))
    assert abs(got["per_row"][str(r)] - want) < 1e-4
    ctl = c23.embed_only(model)
    assert all(float(b.mixer.out.weight.abs().max()) == 0.0 and float(b.ffn.down.weight.abs().max()) == 0.0
               for b in ctl.blocks)
    assert any(float(b.ffn.down.weight.abs().max()) > 0.0 for b in model.blocks)     # the child is untouched


def test_prescreen_flags_loops_and_never_grades():
    looped = c23.prescreen([1, 2, 3, 4] * 4, "abab")
    assert looped["loop_flag"] and "never a grade" in looped["what"]
    clean = c23.prescreen(list(range(16)), "a plain sentence")
    assert not clean["loop_flag"] and clean["distinct_token_ratio"] == 1.0
