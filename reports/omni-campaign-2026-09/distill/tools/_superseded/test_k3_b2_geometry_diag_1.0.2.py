"""CPU tests for k3_b2_geometry_diag (the pure parts: masks, comparisons, the verdict rule, the refusals).
The GPU arms are exercised by the run itself on GPU6; nothing here needs the K3 weights."""
import os
import sys
import tempfile
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k3_b2_geometry_diag as diag  # noqa: E402


def seg(T, lens):
    return types.SimpleNamespace(T=T, real_lengths=torch.tensor(lens))


def test_real_mask_token_major_in_segment_order():
    m = diag.real_mask([seg(4, [4, 2]), seg(3, [1])])
    assert m.tolist() == [True] * 4 + [True, True, False, False] + [True, False, False]


def test_compare_counts_bit_equal_positions_and_groups():
    ref = torch.randn(10, 6).to(torch.bfloat16)
    mine = ref.clone()
    mine[3, 0] += 1.0
    mine[8, 5] += 1.0
    mask = torch.ones(10, dtype=torch.bool)
    mask[8] = False
    c = diag.compare(mine, ref, mask, spans=[(0, 5), (5, 10)], chunk=4)
    assert c["positions"] == 9
    assert c["bit_equal_positions"] == 8
    assert c["unequal_positions_per_group"] == [1, 0]
    assert c["relL2"] > 0 and c["max_abs_diff"] >= 0.5


def test_compare_identical_is_zero_and_cross_dtype_has_no_bit_count():
    ref = torch.randn(7, 3)
    c = diag.compare(ref.clone(), ref)
    assert c["bit_equal_fraction"] == 1.0 and c["relL2"] == 0.0 and c["max_abs_diff"] == 0.0
    c2 = diag.compare(ref.to(torch.bfloat16), ref)
    assert c2["bit_equal_positions"] is None and c2["relL2"] < 1e-2


# the L2 numbers k3_b2_geometry_diag 1.0.0 measured on GPU6 (receipt sha 82532ff0...): the regression for 1.0.2
MEASURED_L2 = {
    "group_vs_single_real": {"positions": 98797, "bit_equal_positions": 98794, "bit_equal_fraction": 98794 / 98797,
                             "relL2": 2.399e-06, "unequal_positions_per_group": [0, 0, 0, 1, 2]},
    "group_vs_single_padded": {"positions": 44563, "bit_equal_positions": 44563, "bit_equal_fraction": 1.0, "relL2": 0.0},
    "group_nopad_vs_single_real": {"positions": 98797, "bit_equal_positions": 93012, "bit_equal_fraction": 93012 / 98797,
                                   "relL2": 1.955e-03, "unequal_positions_per_group": [0, 0, 0, 3124, 2661]},
    "single_nopad_vs_single_real": {"positions": 98797, "bit_equal_positions": 98797, "bit_equal_fraction": 1.0,
                                    "relL2": 0.0},
    "group_vs_fp32_real": {"relL2": 2.936e-02}, "single_vs_fp32_real": {"relL2": 2.936e-02},
}


def _result(**over):
    cmp_ = {k: dict(v) for k, v in MEASURED_L2.items()}
    for k, v in over.pop("compare", {}).items():
        cmp_.setdefault(k, {}).update(v)
    res = {"layers": {"2": {"single_matches_fold": True, "group_matches_fold": False, "compare": cmp_}}}
    res["layers"]["2"].update(over)
    return res


def _closed(groups=(0, 0, 0, 1, 2), padded=0):
    return {"ffn_rows_changed": sum(groups) + padded, "ffn_real_rows_changed": sum(groups),
            "ffn_padded_rows_changed": padded, "groups": [{"ffn_real_rows_changed": g} for g in groups]}


def test_verdict_regression_measured_numbers_are_not_padding():
    """1.0.0's rule printed padding_or_state on these numbers (single_nopad, a control, counted as a restorer)."""
    v = diag.verdict(_result())
    assert v["cause"] == "undetermined" and "no closed-loop mechanism result" in v["notes"]
    assert v["padding_reaches_real_positions"] is False and v["group_nopad_restores_reference"] is False


def test_verdict_bf16_batch_numerics_when_the_closed_loop_matches():
    v = diag.verdict(_result(), _closed())
    assert v["cause"] == "bf16_batch_numerics", v
    assert v["closed_loop_matches_group_arm"] and v["group_arm_real_rows_changed"] == 3


def test_verdict_undetermined_when_the_closed_loop_explains_other_rows():
    v = diag.verdict(_result(), _closed(groups=(0, 0, 0, 2, 2)))
    assert v["cause"] == "undetermined" and not v["closed_loop_matches_group_arm"]


def test_verdict_padding_when_removing_padding_in_groups_restores_bits():
    v = diag.verdict(_result(compare={"group_nopad_vs_single_real": {"relL2": 0.0, "bit_equal_fraction": 1.0}}), _closed())
    assert v["cause"] == "padding_or_state" and v["group_nopad_restores_reference"]


def test_verdict_padding_when_removing_padding_shrinks_error_tenfold():
    v = diag.verdict(_result(compare={"group_nopad_vs_single_real": {"relL2": 1e-7, "bit_equal_fraction": 0.99999}}),
                     _closed())
    assert v["cause"] == "padding_or_state"


def test_verdict_padding_when_padding_reaches_real_positions_in_one_call():
    v = diag.verdict(_result(compare={"single_nopad_vs_single_real": {"bit_equal_fraction": 0.999}}), _closed())
    assert v["cause"] == "padding_or_state" and v["padding_reaches_real_positions"]


@pytest.mark.parametrize("over", [
    {"compare": {"group_vs_fp32_real": {"relL2": 9e-2}}},                         # group much further from fp32
    {"compare": {"group_vs_single_real": {"relL2": 1e-3}}},                       # geometry error not << precision
    {"single_matches_fold": False},                                               # no reference reproduced
])
def test_verdict_undetermined(over):
    assert diag.verdict(_result(**over), _closed())["cause"] == "undetermined"


def test_verdict_mode_rejudges_sealed_receipts(tmp_path):
    arms = tmp_path / "arms.json"
    mech = tmp_path / "mech.json"
    import json as _json
    arms.write_text(_json.dumps({**_result(), "verdict": {"cause": "padding_or_state"}}))
    mech.write_text(_json.dumps({"closed_loop": _closed(), "layers": {"0": {"matches_fold": True},
                                                                      "1": {"matches_fold": True},
                                                                      "2": {"matches_fold": True}}}))
    before = arms.read_bytes()
    rc = diag.main(["--mode", "verdict", "--receipt", str(arms), "--mechanism", str(mech), "--out", str(tmp_path / "v")])
    assert rc == 0 and arms.read_bytes() == before
    got = _json.loads((tmp_path / "v" / "b2_geometry_verdict.json").read_text())
    assert got["verdict"]["cause"] == "bf16_batch_numerics"
    assert got["arms_receipt"]["verdict_as_written"]["cause"] == "padding_or_state"


def test_load_frozen_refuses_another_sha():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write("TOOL_VERSION = 'x'\n")
    try:
        with pytest.raises(SystemExit, match="refusing"):
            diag.load_frozen(handle.name, diag.FROZEN_TOOL_SHA256)
    finally:
        os.remove(handle.name)


def test_main_refuses_files_out_of_the_fold_order():
    with pytest.raises(SystemExit, match="fold's files in order"):
        diag.main(["--input-ids", "/x/k3_review_input_ids.safetensors", "--input-ids", "/x/k3_input_ids.safetensors",
                   "--input-ids", "/x/k3_review_native_input_ids.safetensors", "--out", "/tmp/never"])


def _fake_cp(batch_dependent: bool):
    """A stand-in engine whose FFN is per-token (or, when batch_dependent, perturbed by the call's row count M)."""
    import torch.nn.functional as F

    def ffn_half(runner, carried, ffn_in, cfg, device=None, token_chunk=None):
        bump = (1e-2 * (ffn_in.shape[0] % 7)) if batch_dependent else 0.0
        latent = ffn_in[:, :4] * 1.0
        idx = (ffn_in[:, :2].abs() * 100).long() % 3
        y = latent * 2.0 + bump
        return carried + ffn_in * 3.0 + bump, {"router_topk_idx": idx.to(torch.int16), "y_lat": y, "latent_in": latent}

    def dequant_expert(weights, expert, dtype=torch.float32):
        g = torch.Generator().manual_seed(expert)
        return {k: torch.randn(4, 4, generator=g) for k in ("w1", "w2", "w3")}

    kdp = types.SimpleNamespace(ffn_half=ffn_half, dequant_expert=dequant_expert,
                                situ=lambda a, b, beta, lb: F.silu(a) * b)
    return types.SimpleNamespace(kdp=kdp, cfg={"num_experts": 3}, args=types.SimpleNamespace(token_chunk=4),
                                 device=torch.device("cpu"))


@pytest.mark.parametrize("dependent,expect_changed", [(False, 0), (True, 1)])
def test_closed_loop_counts_only_batch_shape_changes(dependent, expect_changed):
    torch.manual_seed(0)
    n, h = 20, 6
    carried, ffn_in = torch.randn(n, h), torch.randn(n, h)
    real = torch.ones(n, dtype=torch.bool)
    real[-3:] = False
    out = diag.closed_loop(_fake_cp(dependent), None, None, carried, ffn_in, [(0, 8), (8, 20)], real)
    assert out["M_full"] == n and len(out["groups"]) == 2
    assert n <= out["expert_rows_tested"] <= 2 * n          # top-2 ids per row; a row counts once per distinct expert
    assert 0 <= out["expert_rows_changed_by_M"] <= out["expert_rows_tested"]   # CPU BLAS may or may not be M-exact
    if expect_changed:
        assert out["ffn_rows_changed"] > 0 and out["ffn_padded_rows_changed"] >= 0
        assert out["ffn_real_rows_changed"] + out["ffn_padded_rows_changed"] == out["ffn_rows_changed"]
    else:
        assert out["ffn_rows_changed"] == 0 and all(g["y_lat_rows_changed"] == 0 for g in out["groups"])
    assert torch.equal(out["final_prefix"], carried + ffn_in * 3.0 + (1e-2 * (n % 7) if dependent else 0.0))
