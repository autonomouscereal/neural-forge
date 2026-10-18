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


# ---------------------------------------------------------------- 1.1.0 --mode identity (the fold-identity gate)
import hashlib  # noqa: E402
import json as _json  # noqa: E402

BS = 4                                    # block size of the fake stack (K3: 12)


def _digest(t):
    return hashlib.sha256(t.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


def _layer_fn(layer, p, b, nb):
    """A per-token fake layer: AttnRes-like read of the pushed blocks, then an update; pushes at L % BS == 0."""
    if layer % BS == 0:
        b[:, nb].copy_(p)
        nb += 1
    mix = b[:, :nb].float().sum(dim=1) * 0.125
    p.copy_((p.float() * 0.5 + mix + 0.01 * (layer + 1)).to(p.dtype))
    return nb


def _group_fn(spans, defect_layer=None, slot_bug_layer=None):
    def step(layer, p, b, nb):
        if layer % BS == 0:
            slot = nb - 1 if (layer == slot_bug_layer and nb) else nb
            for g0, g1 in spans:
                b[g0:g1, slot].copy_(p[g0:g1])
            nb += 1
        mix = b[:, :nb].float().sum(dim=1) * 0.125
        new = (p.float() * 0.5 + mix + 0.01 * (layer + 1)).to(p.dtype)
        if layer == defect_layer:                       # a structural defect: whole rows of one group move
            g0, g1 = spans[1]
            new[g0:g1] += 0.25
        p.copy_(new)
        return nb
    return step


def _fake_stack(n=24, h=8, layers=10, seed=0):
    torch.manual_seed(seed)
    prefix = torch.randn(n, h).to(torch.bfloat16)
    blocks = torch.zeros(n, 4, h, dtype=torch.bfloat16)
    real = torch.ones(n, dtype=torch.bool)
    real[[5, 6, 17]] = False
    spans = [(0, 8), (8, 16), (16, 24)]
    p, b, nb, ref = prefix.clone(), blocks.clone(), 0, {}
    for layer in range(layers):
        nb = _layer_fn(layer, p, b, nb)
        ref[layer] = _digest(p)
    return prefix, blocks, real, spans, ref


def _identity(loop, *, ref=None, group=None, layers=10, prior=None, prefix=None, blocks=None, nb=0, after=None,
              keep_going=False):
    p0, b0, real, spans, fold = _fake_stack(layers=layers)
    p = p0.clone() if prefix is None else prefix
    b = b0.clone() if blocks is None else blocks
    return diag.identity_pass(list(range(layers)), loop, ref or fold, p, b, nb, block_size=BS,
                              group_step=group or _group_fn(spans), single_step_fn=_layer_fn, digest=_digest,
                              real=real, spans=spans, prior=prior, keep_going=keep_going, after_layer=after)


def test_judge_closed_passes_the_measured_l2_numbers():
    c = {"positions": 98797, "bit_equal_positions": 98794, "relL2": 2.399e-06, "cos_min": 0.99999994}
    j = diag.judge_closed(c, 1, 1, True, True)
    assert j["ok"] and j["real_unequal_positions"] == 3


@pytest.mark.parametrize("c,nbg,slot,earlier,needle", [
    ({"positions": 98797, "bit_equal_positions": 92929, "relL2": 4e-2, "cos_min": 0.97}, 1, True, True, "differ"),
    ({"positions": 100, "bit_equal_positions": 100, "relL2": float("nan"), "cos_min": 1.0}, 1, True, True, "relL2"),
    ({"positions": 100, "bit_equal_positions": 100, "relL2": 0.0, "cos_min": 0.99}, 1, True, True, "cos_min"),
    ({"positions": 100, "bit_equal_positions": 100, "relL2": 0.0, "cos_min": 1.0}, 2, True, True, "blocks after"),
    ({"positions": 100, "bit_equal_positions": 100, "relL2": 0.0, "cos_min": 1.0}, 1, False, True, "pushed block"),
    ({"positions": 100, "bit_equal_positions": 100, "relL2": 0.0, "cos_min": 1.0}, 1, True, False, "earlier block"),
    ({"positions": 100, "bit_equal_positions": None, "relL2": 0.0, "cos_min": 1.0}, 1, True, True, "dtype"),
])
def test_judge_closed_fails(c, nbg, slot, earlier, needle):
    j = diag.judge_closed(c, nbg, 1, slot, earlier)
    assert not j["ok"] and any(needle in r for r in j["reasons"]), j


def test_closed_loop_layers_structural_covers_mla_pushes_and_extras():
    is_kda = {i: not (i % 4 == 3 or i == 92) for i in range(93)}
    got = diag.closed_loop_layers("structural", 93, is_kda, 12, 1)
    mla = [i for i in range(93) if not is_kda[i]]
    assert set(mla) <= set(got) and {0, 12, 24, 36, 48, 60, 72, 84} <= set(got) and {1, 2, 61, 92} <= set(got)
    assert len(got) == len(set(mla) | {0, 12, 24, 36, 48, 60, 72, 84, 1, 2, 61, 92})
    assert diag.closed_loop_layers("3,7-9", 93, is_kda, 12, 1) == [3, 7, 8, 9]
    with pytest.raises(SystemExit, match="layer kinds unknown"):
        diag.closed_loop_layers("structural", 93, {0: True}, 12, 1)


def test_identity_pass_on_a_clean_production_path():
    rec, nb, failed = _identity(list(range(10)))
    v = diag.identity_verdict(rec, list(range(10)), list(range(10)))
    assert not failed and v["pass"] and v["layers_matching_fold"] == 10 and v["closed_loop_passed"] == 10
    assert nb == 3 and all(rec[str(i)]["closed_loop"]["real_unequal_positions"] == 0 for i in range(10))


def test_identity_pass_catches_a_structural_defect_and_stops():
    _p, _b, _r, spans, _ref = _fake_stack()
    rec, _nb, failed = _identity([0, 3, 4, 7], group=_group_fn(spans, defect_layer=3))
    v = diag.identity_verdict(rec, list(range(10)), [0, 3, 4, 7])
    assert failed and not v["pass"] and v["cause"] == "structural" and v["closed_loop_failed"] == [3]
    assert sorted(map(int, rec)) == [0, 1, 2, 3]                     # fail-fast
    assert rec["3"]["matches_fold"] is True                           # the fold chain itself was fine
    assert rec["3"]["closed_loop"]["real"]["unequal_positions_per_group"][1] > 0


def test_identity_pass_catches_a_wrong_block_slot():
    _p, _b, _r, spans, _ref = _fake_stack()
    rec, _nb, failed = _identity([4], group=_group_fn(spans, slot_bug_layer=4))
    cl = rec["4"]["closed_loop"]
    assert failed and not cl["ok"] and not cl["pushed_slot_bit_equal"] and not cl["earlier_slots_untouched"]


def test_identity_pass_off_the_fold_is_an_identity_failure():
    _p, _b, _r, _s, ref = _fake_stack()
    ref = dict(ref)
    ref[5] = "0" * 64
    rec, _nb, failed = _identity([], ref=ref)
    v = diag.identity_verdict(rec, list(range(10)), [])
    assert failed and v["cause"] == "identity" and v["first_layer_off_fold"] == 5 and v["layers_checked"] == 6


def test_identity_verdict_incomplete_without_every_closed_loop():
    rec, _nb, _f = _identity([])
    v = diag.identity_verdict(rec, list(range(10)), [3])
    assert not v["pass"] and v["cause"] == "incomplete"


def test_identity_checkpoint_resume_reproduces_the_uninterrupted_run(tmp_path):
    clean, _nb, _f = _identity([0, 3, 4, 7])
    state = str(tmp_path / "state")
    meta = {"tool_sha256": "t", "frozen_sha256": "f", "corpus_sha256": "c", "fold_receipt_sha256": "r",
            "layer_range": [0, 9]}
    seen = {}

    def after(layer, entry, p, b, nb, bad):
        seen[str(layer)] = entry
        if not bad and (layer + 1) % 3 == 0:
            diag.save_identity_state(state, meta, layer, p, b, nb, entry["prefix_sha256"], dict(seen), _digest)
        if layer == 6:
            raise RuntimeError("thermal stop")

    with pytest.raises(RuntimeError, match="thermal stop"):
        _identity([0, 3, 4, 7], after=after)
    p0, b0, _r, _s, _ref = _fake_stack()
    p, b = torch.zeros_like(p0), torch.zeros_like(b0)
    done, nb, prior = diag.load_identity_state(state, meta, p, b, _digest)
    assert done == 5 and nb == 2 and sorted(map(int, prior)) == list(range(6))
    rec, _nb, failed = _identity([0, 3, 4, 7], prior=prior, prefix=p, blocks=b, nb=nb)
    assert not failed and {k: v["prefix_sha256"] for k, v in rec.items()} == \
        {k: v["prefix_sha256"] for k, v in clean.items()}
    assert [n for n in os.listdir(state) if n.startswith("prefix_L")] == ["prefix_L05.safetensors"]


def test_identity_resume_refuses_another_frozen_tool_or_a_changed_file(tmp_path):
    p0, b0, _r, _s, _ref = _fake_stack()
    meta = {"tool_sha256": "t", "frozen_sha256": "f", "corpus_sha256": "c", "fold_receipt_sha256": "r",
            "layer_range": [0, 9]}
    b0[:, 0] = p0
    diag.save_identity_state(str(tmp_path), meta, 0, p0, b0, 1, _digest(p0), {}, _digest)
    with pytest.raises(SystemExit, match="frozen_sha256"):
        diag.load_identity_state(str(tmp_path), {**meta, "frozen_sha256": "g"}, p0.clone(), b0.clone(), _digest)
    st = _json.loads((tmp_path / "state.json").read_text())
    st["prefix_sha256"] = "0" * 64
    (tmp_path / "state.json").write_text(_json.dumps(st))
    with pytest.raises(SystemExit, match="does not hash"):
        diag.load_identity_state(str(tmp_path), meta, p0.clone(), b0.clone(), _digest)


def test_main_identity_mode_parses():
    a = diag.build_parser().parse_args(["--mode", "identity", "--closed-loop-layers", "all", "--resume", "--out", "x"])
    assert a.mode == "identity" and a.resume and a.checkpoint_every == 4 and not a.keep_going
