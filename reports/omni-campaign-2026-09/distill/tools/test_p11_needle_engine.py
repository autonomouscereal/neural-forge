"""Tests for p11_needle_engine.py 1.1.0 (E2, lane gpu4, Claude, 2026-10-13; 1.1.0 lane p2p11 2026-10-14). AI server, CPU fp32, no GPU:

    cd /home/cereal/omni-campaign/distill/tools
    CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9 OC_CALIBRATION_DIR=/home/cereal/omni-campaign/calibration \\
    nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" -p no:cacheprovider -q test_p11_needle_engine.py

What is pinned (each against the STOCK code, imported, never copied):
  * the vectorised short conv == CausalDepthwiseShortConv.forward_with_state (the tree's per-position loop), chained
    across chunks with the carried history;
  * a KDA block and a gated-GQA block run chunked by E2 (kernels tree / loop, chunk sizes 1..L, query sub-chunks
    down to 1 row) == the stock `block(x, token_valid_mask=...)` call of the streamed review engine;
  * on a tiny written checkpoint: NeedleEngine.parity passes (all layers within fp32 noise, top-1 identical), and
    NeedleEngine.run's scored logits == run_omni_frozen_review.StreamedChimera.run at the same positions;
  * the needle builder (real Qwen3.5-0.8B tokenizer + oc_calib.render_prompt, thinking off): every document is
    exactly its length, the scored rows predict the key digits then <|im_end|>, the needle sits at its depth and
    decodes to its sentence, the control has none, rebuilds are byte-identical, a tampered file is refused;
  * a charted block is the identity in E2; scoring, the parity gate and the CPU/fla refusals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch = pytest.importorskip("torch")

OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
TREE = Path(os.environ.get("CHIMERA_OMNI_TREE", os.environ.get("CHIMERA_OMNI_SRC", str(OC / "import/rq27b/tree_ef1ffb9"))))
if TREE.name == "src":
    TREE = TREE.parent
CALIBRATION = Path(os.environ.get("OC_CALIBRATION_DIR", str(OC / "calibration")))
REVIEW = Path(os.environ.get("OMNI_REVIEW_TOOL", str(OC / "composite/tools/run_omni_frozen_review.py")))
QWEN_0P8B = Path(os.environ.get("QWEN_0P8B_DIR", "/home/cereal/models/Qwen3.5-0.8B"))
for need, what in ((TREE / "src" / "chimera_omni", "chimera-omni tree"), (REVIEW, "review runner"), (CALIBRATION / "oc_calib.py", "calibration lane")):
    if not need.exists():
        pytest.skip(f"no {what} at {need}", allow_module_level=True)

import p11_needle_engine as e2  # noqa: E402

ROFR = e2.load_review_tool(REVIEW)
ROFR.bind_chimera_tree(TREE, None)

from chimera_omni.importing.profiles import QWEN35_FAMILY_DEFAULTS, QWEN38_27B_VERIFIED_GEOMETRY, qwen35_chimera_config  # noqa: E402
from chimera_omni.importing.safetensors_io import write_safetensors  # noqa: E402
from chimera_omni.ledger import DEFAULT_FIXED_BANDS  # noqa: E402
from chimera_omni.modeling import ChimeraOmniModel  # noqa: E402
from chimera_omni.modeling.layers import ChimeraBlock, CausalDepthwiseShortConv  # noqa: E402

BASE_VOCAB = 248320


def tiny_cfg(layers: int = 8):
    hf = {**QWEN35_FAMILY_DEFAULTS, **QWEN38_27B_VERIFIED_GEOMETRY, "vocab_size": BASE_VOCAB, "hidden_size": 64,
          "num_hidden_layers": layers, "intermediate_size": 96, "num_attention_heads": 4, "num_key_value_heads": 2,
          "head_dim": 32, "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16,
          "linear_value_head_dim": 16, "tie_word_embeddings": True, "max_position_embeddings": 4096}
    return qwen35_chimera_config(hf, residual_mode="none", ledger_bands=list(DEFAULT_FIXED_BANDS), native_feature_dim=768,
                                 max_sequence_length=4096)


def random_block(cfg, index: int, seed: int = 3):
    torch.manual_seed(seed)
    block = ChimeraBlock(cfg, index).eval()
    with torch.no_grad():
        for p in block.parameters():
            p.mul_(0.5)
    return block


def kernels(chunk, kda="loop", budget=1 << 30, min_rows=1):
    return e2.E2Kernels(chunk=chunk, kda_kernel=kda, attn_budget_bytes=budget, min_query_rows=min_rows)


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    cfg = tiny_cfg()
    torch.manual_seed(11)
    model = ChimeraOmniModel(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.5)
        model.embed.weight[BASE_VOCAB:].zero_()
    out = tmp_path_factory.mktemp("tiny") / "ckpt"
    write_safetensors(model.state_dict(), out, config=cfg.to_manifest(), ledger=cfg.ledger().to_manifest(),
                      source={"import_source": "p11-test-fixture"}, chimera_commit="test")
    return out, cfg


# ----------------------------------------------------------------------------------------------- kernels


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("channels", [24, 2048, 6144])
def test_short_conv_vectorized_equals_tree_loop_across_chunks(dtype, channels):
    """Bit-equal at the OMNI1 widths (2,048 q/k, 6,144 v); at a toy width the tree's own [1, C, 4] reduction takes a
    small-size CPU path that can round 1 ulp differently (measured 2.98e-08 at C=24 fp32), so that case gets 1 ulp."""

    torch.manual_seed(0)
    conv = CausalDepthwiseShortConv(channels, 4, bias=False)
    x = torch.randn(1, 29, channels).to(dtype)
    with torch.no_grad():
        ref, ref_state = conv.forward_with_state(x)
        hist, outs = None, []
        for s, e in e2.chunks(29, 6):
            y, hist = e2.short_conv_vectorized(conv, x[:, s:e], hist)
            outs.append(y)
    got = torch.cat(outs, dim=1)
    assert got.shape == ref.shape and torch.equal(hist, ref_state)
    diff = float((got.float() - ref.float()).abs().max())
    if channels >= 2048:
        assert torch.equal(got, ref), f"max |d| {diff:.3e}"
    else:
        assert diff <= 2 * torch.finfo(dtype).eps * max(1.0, float(ref.abs().max())), diff


@pytest.mark.parametrize("kda", ["tree", "loop"])
@pytest.mark.parametrize("chunk", [1, 5, 16, 37, 64])
def test_kda_block_chunked_equals_stock_layer(kda, chunk):
    cfg = tiny_cfg()
    block = random_block(cfg, 0)
    assert type(block.mixer).__name__ == "KimiDeltaAttention" and block.mixer.key_group == 2
    torch.manual_seed(5)
    x = torch.randn(1, 37, 64)
    with torch.no_grad():
        ref = block(x, token_valid_mask=torch.ones(1, 37, dtype=torch.bool))
        got = kernels(chunk, kda).block(block, x.clone())
    diff = float((got - ref).abs().max())
    print(f"kda {kda} chunk {chunk}: bit_equal {torch.equal(got, ref)} max|d| {diff:.3e}")
    assert diff <= 1e-5 * max(1.0, float(ref.abs().max()))


@pytest.mark.parametrize("chunk, rows", [(37, 1), (8, 3), (5, 16), (64, 64)])
def test_gqa_block_chunked_equals_stock_layer(chunk, rows):
    cfg = tiny_cfg()
    block = random_block(cfg, 3)
    assert type(block.mixer).__name__ == "GatedGQAAttention" and block.mixer.num_kv_groups == 2
    torch.manual_seed(6)
    x = torch.randn(1, 37, 64)
    k = kernels(chunk, budget=rows * block.mixer.num_heads * 37 * 4, min_rows=1)
    with torch.no_grad():
        ref = block(x, token_valid_mask=torch.ones(1, 37, dtype=torch.bool))
        got = k.block(block, x.clone())
    diff = float((got - ref).abs().max())
    print(f"gqa chunk {chunk} query rows {k.stats['query_rows_min']}: bit_equal {torch.equal(got, ref)} max|d| {diff:.3e}")
    assert k.stats["query_rows_min"] <= max(rows, 1)
    assert diff <= 1e-5 * max(1.0, float(ref.abs().max()))


def test_charted_block_is_the_identity_and_unknown_mixers_refuse():
    class Charted:
        charted = object()

    x = torch.randn(1, 4, 8)
    assert kernels(2).block(Charted(), x) is x
    cfg = tiny_cfg()
    block = random_block(cfg, 0)
    with pytest.raises(e2.Refusal):
        kernels(2).block(block, torch.randn(2, 4, 64))  # one sequence at a time


def test_query_subchunk_respects_budget_and_cap():
    assert e2.query_subchunk(24, 262144, 2048 << 20, cap=8192) == 80
    assert e2.query_subchunk(24, 4096, 2048 << 20, cap=8192) == 5456
    assert e2.query_subchunk(24, 4096, 2048 << 20, cap=100) == 100
    assert e2.query_subchunk(24, 10 ** 9, 1, cap=8192, floor=16) == 16


# ----------------------------------------------------------------------------------------------- engine


def test_engine_parity_and_run_match_the_stock_streamed_engine(tiny_ckpt):
    root, cfg = tiny_ckpt
    eng = e2.NeedleEngine(ROFR, root, device="cpu", dtype=torch.float32, mask_band=True, kernels=kernels(7, "loop", budget=4 * 4 * 50 * 3))
    torch.manual_seed(9)
    seq = [int(t) for t in torch.randint(0, 5000, (50,))]
    positions = [0, 10, 31, 48, 49]
    result = eng.parity(seq, positions, label="tiny")
    print({k: result[k] for k in ("all_layers_bit_equal", "final_residual_max_abs_diff", "logits_max_abs_diff", "top1_agreement", "top1_compared")})
    assert result["finite"] and result["logits_same_finite_mask"] and result["top1_agreement"] == result["top1_compared"] == 5
    assert result["logits_max_abs_diff"] <= 1e-4 * max(1.0, result["logits_ref_max_abs"])
    gate = e2.parity_gate(result, device="cpu", min_top1=0.95, max_rel=1e-4)
    assert gate["verdict"] == "PASS", gate
    stock = ROFR.StreamedChimera(root, device="cpu", dtype=torch.float32, ledger_band_logit_mask=True)
    ref_logits = stock.run([seq], micro_batch=1)["logits"][0][positions]
    got = eng.run([seq, seq[:30]], [positions, [0, 29]], label="tiny-run")
    d, m = e2.logits_diff(got["logits"][0], ref_logits)
    assert d <= 1e-4 * max(1.0, m), d
    short = stock.run([seq[:30]], micro_batch=1)["logits"][0][[0, 29]]
    d2, m2 = e2.logits_diff(got["logits"][1], short)
    assert d2 <= 1e-4 * max(1.0, m2), d2
    assert torch.equal(torch.isfinite(got["logits"][0]), torch.isfinite(ref_logits))  # the band mask is applied
    assert got["tokens"] == 80


def test_engine_refusals(tiny_ckpt):
    root, _ = tiny_ckpt
    base = dict(model=str(root), chunk=8, attn_budget_mib=16, min_query_rows=16, no_band_mask=False)
    with pytest.raises(e2.Refusal, match="bf16 on the CPU"):
        e2.engine_from_args(ROFR, argparse.Namespace(**base, device="cpu", dtype="bfloat16", kda_kernel="loop"))
    with pytest.raises(e2.Refusal, match="needs a CUDA device"):
        e2.engine_from_args(ROFR, argparse.Namespace(**base, device="cpu", dtype="float32", kda_kernel="fla"))
    with pytest.raises(e2.Refusal, match="--kda-kernel"):
        e2.E2Kernels(chunk=4, kda_kernel="flash", attn_budget_bytes=1)


# ----------------------------------------------------------------------------------------------- needles


@pytest.fixture(scope="module")
def tokenizer_render():
    if not (QWEN_0P8B / "tokenizer_config.json").is_file():
        pytest.skip(f"no Qwen3.5-0.8B tokenizer at {QWEN_0P8B}")
    from transformers import AutoTokenizer

    oc_calib, _ = ROFR.bind_calibration(CALIBRATION)
    tok = AutoTokenizer.from_pretrained(str(QWEN_0P8B), trust_remote_code=False)
    return tok, (lambda messages: oc_calib.render_prompt(tok, messages, thinking="off"))


@pytest.fixture(scope="module")
def filler(tmp_path_factory):
    path = tmp_path_factory.mktemp("filler") / "eval_rows.jsonl"
    rows = [{"instruction": f"Row {i}: describe the harbour at dawn.", "response": f"The harbour {i} is quiet; gulls circle the mast and the tide turns slowly."}
            for i in range(80)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_needle_documents_are_exact_and_scored_rows_predict_the_key(tokenizer_render, filler):
    tok, render = tokenizer_render
    built = e2.build_needles(tokenizer=tok, render=render, filler_path=filler, lengths=[300, 600], depths=[0.25, 0.75],
                             control_lengths=[300], seed=7)
    docs = built["docs"]
    assert [d["id"] for d in docs] == ["L300-d25", "L300-d75", "L300-d25-control", "L600-d25", "L600-d75"]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    for d in docs:
        ids = d["ids"]
        assert len(ids) == d["length"]
        p, n = d["prompt_tokens"], d["answer_tokens"]
        assert n == e2.KEY_DIGITS and tok.decode(ids[p:p + n]) == d["key"] and ids[-1] == im_end
        assert d["scored_positions"] == list(range(p - 1, p + n))
        assert "</think>" in tok.decode(ids[p - 8:p]) and tok.decode(ids[p - 40:p]).count(d["archive"]) == 1  # question then thinking-off
        if d["needle"]:
            text = tok.decode(ids[d["needle_start"]:d["needle_end"]])
            assert text.startswith(f"The secret passkey for the {d['archive']} archive is {d['key']}.")
            assert abs(d["needle_depth_actual"] - d["depth"]) < 0.2
        else:
            assert d["needle_start"] is None and d["key"] not in tok.decode(ids[:p])
    assert docs[0]["needle_start"] < docs[1]["needle_start"]
    again = e2.build_needles(tokenizer=tok, render=render, filler_path=filler, lengths=[300, 600], depths=[0.25, 0.75],
                             control_lengths=[300], seed=7)
    assert [d["ids_sha256"] for d in again["docs"]] == [d["ids_sha256"] for d in docs]
    with pytest.raises(e2.Refusal, match="no room"):
        e2.build_needles(tokenizer=tok, render=render, filler_path=filler, lengths=[40], depths=[0.5], control_lengths=[], seed=7)


def test_needle_set_roundtrip_and_tamper(tokenizer_render, filler, tmp_path):
    tok, render = tokenizer_render
    built = e2.build_needles(tokenizer=tok, render=render, filler_path=filler, lengths=[300], depths=[0.5], control_lengths=[], seed=3)
    manifest = e2.save_needles(tmp_path / "n", built, {"tokenizer_dir": str(QWEN_0P8B)})
    m2, ids = e2.load_needles(tmp_path / "n")
    assert m2["needles_sha256"] == manifest["needles_sha256"] and ids["L300-d50"] == built["docs"][0]["ids"]
    data = json.loads((tmp_path / "n" / "manifest.json").read_text())
    data["docs"][0]["ids_sha256"] = "0" * 64
    (tmp_path / "n" / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(e2.Refusal, match="does not match"):
        e2.load_needles(tmp_path / "n")


# ----------------------------------------------------------------------------------------------- scoring / gate


def _doc(key="4821379"):
    return {"id": "x", "length": 100, "depth": 0.25, "needle": True, "needle_depth_actual": 0.25, "key": key,
            "answer_ids": [10 + int(c) for c in key], "stop_id": 3}


def test_score_doc_reads_hits_ranks_and_refuses_nan():
    doc = _doc()
    logits = torch.zeros(8, 40)
    logits[:, 30:] = float("-inf")  # a masked band
    for i, a in enumerate(doc["answer_ids"]):
        logits[i, a] = 5.0
    logits[7, 3] = 5.0
    s = e2.score_doc(doc, logits)
    assert s["first_digit_hit"] and s["passkey_exact_teacher_forced"] and s["stop_after_digits"] and s["first_digit_rank"] == 1
    logits[0, 2] = 9.0
    logits[2, 1] = 9.0
    s = e2.score_doc(doc, logits)
    assert not s["first_digit_hit"] and s["first_digit_rank"] == 2 and s["digits_argmax_agree"] == 5 and not s["passkey_exact_teacher_forced"]
    logits[4, 0] = float("nan")
    with pytest.raises(e2.Refusal, match="non-finite"):
        e2.score_doc(doc, logits)
    with pytest.raises(e2.Refusal, match="scored rows"):
        e2.score_doc(doc, torch.zeros(5, 40))


def test_parity_gate_rules():
    ok = {"logits_bit_equal": False, "logits_max_abs_diff": 1e-6, "logits_ref_max_abs": 10.0, "top1_agreement": 64,
          "top1_compared": 64, "finite": True, "charted_identity_bit_equal": True, "logits_same_finite_mask": True}
    assert e2.parity_gate(ok, device="cpu", min_top1=0.95, max_rel=1e-4)["verdict"] == "PASS"
    assert e2.parity_gate({**ok, "logits_max_abs_diff": 0.1}, device="cpu", min_top1=0.95, max_rel=1e-4)["verdict"] == "FAIL"
    assert e2.parity_gate({**ok, "logits_max_abs_diff": 0.1}, device="cuda:0", min_top1=0.95, max_rel=1e-4)["verdict"] == "PASS"
    assert e2.parity_gate({**ok, "top1_agreement": 50}, device="cuda:0", min_top1=0.95, max_rel=1e-4)["verdict"] == "FAIL"
    assert e2.parity_gate({**ok, "charted_identity_bit_equal": False}, device="cuda:0", min_top1=0.95, max_rel=1e-4)["verdict"] == "FAIL"
    assert e2.parity_gate({**ok, "logits_bit_equal": True}, device="cpu", min_top1=0.95, max_rel=1e-4)["why"] == "bit-equal"


# ----------------------------------------------------------------------------------------------- 1.1.0


def _fla_available() -> bool:
    try:
        import fla.ops.gated_delta_rule.naive  # noqa: F401
    except Exception:
        return False
    return True


needs_fla = pytest.mark.skipif(not _fla_available(), reason="fla (flash-linear-attention) is not importable here")


@needs_fla
@pytest.mark.parametrize("impl", ["fla-naive-recurrent", "fla-naive-chunk"])
@pytest.mark.parametrize("chunk", [None, 1, 5, 16, 37])
def test_fla_reference_recurrences_equal_the_tree_kda_mixer(impl, chunk):
    """E2's fla argument mapping (g = log retention, beta, scale 1.0, l2-normalised q/k repeated to the value
    heads, state [B, H, K, V], state carried across chunks) checked in fp32 on the CPU through fla's own torch
    references: the KDA mixer output and the final recurrent state equal the tree's forward_with_state."""

    cfg = tiny_cfg()
    block = random_block(cfg, 0)
    mixer = block.mixer
    if mixer.decay_kind != "gdn":
        pytest.skip(f"tiny KDA decay is {mixer.decay_kind!r}; the fla path is wired for gdn")
    torch.manual_seed(6)
    x = torch.randn(1, 37, 64)
    with torch.no_grad():
        normalized = block.norm1(x)
        ref, ref_state = mixer.forward_with_state(normalized, initial_state=None, token_valid_mask=torch.ones(1, 37, dtype=torch.bool))
        if chunk is None:
            got, state = e2.kda_chunk(mixer, normalized, None, recurrence=impl)
        else:
            pieces, state = [], None
            for s, e in e2.chunks(37, chunk):
                piece, state = e2.kda_chunk(mixer, normalized[:, s:e], state, recurrence=impl)
                pieces.append(piece)
            got = torch.cat(pieces, dim=1)
    scale = max(1.0, float(ref.abs().max()))
    assert float((got - ref).abs().max()) <= 1e-5 * scale
    assert float((state.recurrent - ref_state.recurrent).abs().max()) <= 1e-5 * max(1.0, float(ref_state.recurrent.abs().max()))


@needs_fla
def test_fla_recurrence_is_called_by_keyword_so_gate_and_strength_cannot_swap(monkeypatch):
    """naive_recurrent takes (q, k, v, beta, g), the chunk forms (q, k, v, g, beta): E2 must pass both by name."""

    import fla.ops.gated_delta_rule.naive as naive

    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return kwargs["v"], kwargs["initial_state"]

    monkeypatch.setattr(naive, "naive_recurrent_gated_delta_rule", spy)
    q = torch.zeros(1, 2, 1, 4)
    g, beta = torch.full((1, 2, 1), -0.5), torch.full((1, 2, 1), 0.25)
    e2.fla_recurrence("fla-naive-recurrent", q, q, q, g, beta, torch.zeros(1, 1, 4, 4))
    assert seen["g"] is g and seen["beta"] is beta and seen["scale"] == 1.0
    with pytest.raises(e2.Refusal):
        e2.fla_recurrence("flash", q, q, q, g, beta, None)


@needs_fla
def test_kda_block_with_fla_reference_kernels_equals_stock_layer():
    cfg = tiny_cfg()
    block = random_block(cfg, 0)
    if block.mixer.decay_kind != "gdn":
        pytest.skip("not a gdn decay")
    torch.manual_seed(5)
    x = torch.randn(1, 37, 64)
    with torch.no_grad():
        ref = block(x, token_valid_mask=torch.ones(1, 37, dtype=torch.bool))
        for impl in ("fla-naive-recurrent", "fla-naive-chunk"):
            got = kernels(16, impl).block(block, x.clone())
            assert float((got - ref).abs().max()) <= 1e-5 * max(1.0, float(ref.abs().max())), impl


def test_score_doc_readouts_score_past_control_tokens():
    doc = _doc("4821379")
    control, digit_ids = [1, 2, 3], list(range(10, 20))
    logits = torch.zeros(8, 40)
    logits[:, 30:] = float("-inf")
    for i, a in enumerate(doc["answer_ids"]):
        logits[i, a] = 5.0
    logits[7, 3] = 5.0
    logits[0, 3] = 9.0   # an empty answer (control) on top of the first digit
    logits[4, 2] = 9.0   # a control token mid-key
    s = e2.score_doc(doc, logits, control_ids=control, digit_ids=digit_ids)
    r = s["readouts"]
    assert not s["first_digit_hit"] and s["digits_argmax_agree"] == 5            # 1.0.0's strict numbers, unchanged
    assert r["strict"]["partial"] == 5 and not r["strict"]["exact"]
    assert r["content"]["exact"] and r["content"]["partial"] == 7 and r["content"]["first_digit_hit"]
    assert r["digit"]["exact"] and r["digit"]["partial"] == 7
    assert s["control_argmax_positions"] == [0, 4] and s["control_mass_per_digit"][0] > 0.9 and s["control_mass_per_digit"][1] < 0.05  # 3 / (e^5 + 29)
    assert s["rows_top"]["k"] == 16 and len(s["rows_top"]["ids"]) == 8 and s["rows_top"]["ids"][0][0] == 3
    # a non-control word above the digit: content misses, digit still reads the key
    logits[0, 25] = 7.0
    s = e2.score_doc(doc, logits, control_ids=control, digit_ids=digit_ids)
    assert not s["readouts"]["content"]["first_digit_hit"] and s["readouts"]["digit"]["first_digit_hit"]
    assert s["digit_rank_first"] == 1
    # a WRONG digit is a miss in every readout
    logits[1, 10] = 20.0
    s = e2.score_doc(doc, logits, control_ids=control, digit_ids=digit_ids)
    assert all(not s["readouts"][k]["hits"][1] for k in ("strict", "content", "digit"))
    assert "content=" in e2.summary_token({**s, "id": "x"}) and "digit=" in e2.summary_token({**s, "id": "x"})


def test_rescore_doc_recovers_only_what_1_0_0_kept():
    control, digits = {100, 101, 102}, set(range(10, 20))
    answer = [11, 16, 14, 16, 12, 16, 18]  # key 1646268
    # the 2026-10-13 L262144-d25 shape: control first, rank 6, top-5 = control, word, control, word, word
    entry = {"id": "L262144-d25", "key": "1646268", "needle": True, "length": 262144, "argmax_text": "x",
             "first_digit_top5_ids": [101, 50, 100, 51, 52], "first_digit_rank": 6}
    r = e2.rescore_doc(entry, answer, encode=lambda _t: [101] + answer[1:], control=control, digits=digits)
    ro = r["readouts"]
    assert ro["strict"]["known_hits"] == 6 and ro["strict"]["exact"] is False
    assert ro["digit"]["hits"][0] is True and ro["digit"]["exact"] is True          # nothing above the answer is a digit
    assert ro["content"]["hits"][0] is False and ro["content"]["exact"] is False    # the word 50 beats it
    assert r["control_argmax_positions"] == [0] and r["wrong_digit_positions"] == []
    # rank beyond the kept top-5: the first-row digit / content readouts are unknown, never guessed
    entry2 = {**entry, "first_digit_top5_ids": [101, 100, 102, 101, 100], "first_digit_rank": 9}
    r2 = e2.rescore_doc(entry2, answer, encode=lambda _t: [101, 16, 14, 100, 12, 16, 18], control=control, digits=digits)
    assert r2["readouts"]["digit"]["hits"][0] is None and r2["readouts"]["digit"]["hits"][3] is None
    assert r2["readouts"]["digit"]["exact"] is None and r2["readouts"]["digit"]["partial_range"] == [5, 7]
    assert r2["readouts"]["content"]["hits"][0] is None
    # a wrong digit is a known miss everywhere
    r3 = e2.rescore_doc(entry, answer, encode=lambda _t: [101, 16, 13, 16, 12, 16, 18], control=control, digits=digits)
    assert r3["wrong_digit_positions"] == [2] and r3["readouts"]["digit"]["exact"] is False
    bad = e2.rescore_doc(entry, answer, encode=lambda _t: [1, 2], control=control, digits=digits)
    assert "error" in bad


def test_recurrence_parity_command_on_a_tiny_checkpoint(tiny_ckpt, tmp_path):
    root, cfg = tiny_ckpt
    seq = [int(t) for t in torch.randint(0, 5000, (64,), generator=torch.Generator().manual_seed(3))]
    built = {"docs": [{"id": "L64-d25", "ids": seq, "length": 64, "needle": True, "ids_sha256": e2.ids_sha256(seq)}]}
    e2.save_needles(tmp_path / "n", built, {"tokenizer_dir": str(QWEN_0P8B)})
    block = ChimeraBlock(cfg, 0)
    arms = "loop,fla-naive-recurrent,fla-naive-chunk" if (_fla_available() and block.mixer.decay_kind == "gdn") else "loop"
    rc = e2.main(["recurrence-parity", "--model", str(root), "--chimera-src", str(TREE), "--needles", str(tmp_path / "n"),
                  "--review-tool", str(REVIEW), "--calibration-dir", str(CALIBRATION), "--layers", "0,4", "--lengths", "20,64",
                  "--chunks", "7,16", "--arms", arms, "--out", str(tmp_path / "rp.json")])
    receipt = json.loads((tmp_path / "rp.json").read_text())
    assert rc == 0 and receipt["verdict"] == "PASS", receipt["worst"]
    assert {r["arm"] for r in receipt["rows"]} == set(arms.split(","))
    assert {r["layer"] for r in receipt["rows"]} == {0, 4} and any(r["boundaries_crossed"] > 0 for r in receipt["rows"])
    # the triton arm is refused on the CPU (exit 3), never silently swapped for a reference
    assert e2.main(["recurrence-parity", "--model", str(root), "--chimera-src", str(TREE), "--needles", str(tmp_path / "n"),
                    "--review-tool", str(REVIEW), "--calibration-dir", str(CALIBRATION), "--arms", "fla",
                    "--out", str(tmp_path / "x.json")]) == e2.EXIT_REFUSED
