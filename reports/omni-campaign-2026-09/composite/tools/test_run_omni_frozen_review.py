"""Tests for run_omni_frozen_review.py (run on the AI server; CPU fp32 on a tiny checkpoint).

    cd /home/cereal/omni-campaign/composite/tools
    CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_c5e6052 \
    OC_CALIBRATION_DIR=/home/cereal/omni-campaign/calibration \
    /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" -p no:cacheprovider -q \
        test_run_omni_frozen_review.py --basetemp /home/cereal/omni-campaign/composite/scratch_review_tests

Run-book row tests (tool table, run_omni_frozen_review.py):
  (1) on the 0.8B Chimera checkpoint the split dossier matches the HF `oc_run_frozen_bank`
      dossier lane-for-lane on bank v1 -- `test_real_0p8b_split_dossier_vs_hf_dossier` reads
      the dossier the tool wrote (OMNI_REVIEW_0P8B_DOSSIER) and the HF dossier, compares every
      case's tokens / text / termination and PRINTS the count; it skips when either is absent;
  (2) `--stream` with generation requested is refused (exit 3, nothing written);
  (3) media-out non-empty flags on procedural stimuli (music_codes band stream on the tiny model);
  (4) tap dumps reproduce the Gram inputs on one layer (hook-captured norm2 input == dump; X^T X equal).
Fixer 2026-09-06 (one test per review finding): --micro-batch threaded + defaults (1 for --input rows,
8 for bank rows) and a streamed OOM is a Refusal with `<out>.refused.json`; RoPE inv_freq stays fp32 in
both loaders under bf16 (2,048-position argmax measured); --reference-dossier is refused on a prompt
surface / thinking mismatch and per-case prompt_messages are compared; --input without
--tap-residuals --tap-out is a refusal; padded-batch taps take the validity mask and an all-zero row
refuses the dump; taps_receipt.json carries per-layer zero_gain_coordinates (the §3b trap);
GPU occupancy refusal + CUDA inventory in placement; proxy wall time named; charted block kinds refused
by name; the real 0.8B proxy (teacher-forced fp32 vs the HF dossier) has its agreement floor asserted.
Plus: split decoder == `forward_causal_with_cache` bit-for-bit on one device and across a 2-segment
split; streamed engine == resident `forward` (padded batch); cached decode == full recompute;
block_attnres refused; schema: the dossier carries every key the HF 0.8B dossier carries (top level
and per record) and `oc_assess.py` consumes it unchanged; proxy mode writes `mode`
teacher-forced-proxy with agreement numbers; vision cards run through the native bridge.

The tiny fixture is a Qwen-profile Chimera config (kda,kda,kda,gated_gqa; residual none; fixed-band
ledger with the real 248,320-id Qwen text vocabulary so the REAL Qwen3.5-0.8B tokenizer and chat
template render the prompts) with hidden 64, four layers, written by `write_safetensors`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOL = HERE / "run_omni_frozen_review.py"
CHIMERA_SRC = Path(os.environ.get("CHIMERA_OMNI_SRC", "/home/cereal/omni-campaign/import/rq27b/tree_c5e6052"))
CALIBRATION = Path(os.environ.get("OC_CALIBRATION_DIR", "/home/cereal/omni-campaign/calibration"))
QWEN_0P8B = Path(os.environ.get("QWEN_0P8B_DIR", "/home/cereal/models/Qwen3.5-0.8B"))
HF_0P8B_DOSSIER = Path(os.environ.get("HF_0P8B_DOSSIER", str(CALIBRATION / "out" / "oc-dossier-Q35-0.8B.json")))
REVIEW_RECEIPTS = HERE.parent / "receipts" / "review"
REAL_DOSSIER = os.environ.get("OMNI_REVIEW_0P8B_DOSSIER", str(REVIEW_RECEIPTS / "oc-dossier-Q35-0.8B-CHIMERA-split12x12.json"))
PROXY_DOSSIER = os.environ.get("OMNI_REVIEW_0P8B_PROXY_DOSSIER", str(REVIEW_RECEIPTS / "oc-dossier-Q35-0.8B-CHIMERA-proxy-fp32.json"))
# Measured 2026-09-07 on the 0.8B Chimera checkpoint (fp32, GPU4) teacher-forced against the HF bf16 dossier:
# see test_real_0p8b_teacher_forced_agreement_floor; these are the floors the two real-run tests assert.
REAL_SPLIT_FLOOR = {"exact_text": 3, "exact_ids": 3, "same_termination": 16}
PROXY_MIN_LANE_AGREEMENT = float(os.environ.get("OMNI_REVIEW_0P8B_PROXY_MIN_LANE", "0.90"))
PROXY_MEAN_AGREEMENT = float(os.environ.get("OMNI_REVIEW_0P8B_PROXY_MEAN", "0.95"))
PY = sys.executable

torch = pytest.importorskip("torch")

if not (CHIMERA_SRC / "src" / "chimera_omni").is_dir():
    pytest.skip(f"no chimera-omni tree at {CHIMERA_SRC}", allow_module_level=True)
if not (CALIBRATION / "oc_calib.py").is_file():
    pytest.skip(f"no calibration lane at {CALIBRATION}", allow_module_level=True)
if not (QWEN_0P8B / "tokenizer_config.json").is_file():
    pytest.skip(f"no Qwen3.5-0.8B tokenizer at {QWEN_0P8B}", allow_module_level=True)


def _load_tool():
    spec = importlib.util.spec_from_file_location("run_omni_frozen_review", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_omni_frozen_review"] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()
TREE = tool.bind_chimera_tree(CHIMERA_SRC, None)
OC_CALIB, OCR = tool.bind_calibration(CALIBRATION)

from chimera_omni.importing.profiles import QWEN35_FAMILY_DEFAULTS, QWEN38_27B_VERIFIED_GEOMETRY, qwen35_chimera_config  # noqa: E402
from chimera_omni.importing.safetensors_io import load_chimera_checkpoint, write_safetensors  # noqa: E402
from chimera_omni.ledger import DEFAULT_FIXED_BANDS  # noqa: E402
from chimera_omni.modeling import ChimeraOmniModel  # noqa: E402

BASE_VOCAB = 248320
MUSIC = 64


def tiny_hf_config() -> dict:
    return {
        **QWEN35_FAMILY_DEFAULTS,
        **QWEN38_27B_VERIFIED_GEOMETRY,
        "vocab_size": BASE_VOCAB,
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "intermediate_size": 96,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "tie_word_embeddings": True,
        "max_position_embeddings": 4096,
    }


def build_tiny(out: Path, *, residual_mode: str = "none", music: bool = True, seed: int = 7) -> tuple[Path, object]:
    bands = list(DEFAULT_FIXED_BANDS) + ([("music_codes", MUSIC)] if music else [])
    cfg = qwen35_chimera_config(tiny_hf_config(), residual_mode=residual_mode, ledger_bands=bands, native_feature_dim=768, max_sequence_length=4096)
    torch.manual_seed(seed)
    model = ChimeraOmniModel(cfg)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(0.5)
        # the band rows are zero on an import; keep that property so the mask matters
        model.embed.weight[BASE_VOCAB:].zero_()
    write_safetensors(
        model.state_dict(), out, config=cfg.to_manifest(), ledger=cfg.ledger().to_manifest(),
        source={"import_source": "test-fixture"}, chimera_commit=TREE["commit"],
    )
    return out, cfg


@pytest.fixture(scope="session")
def tiny(tmp_path_factory) -> tuple[Path, object]:
    return build_tiny(tmp_path_factory.mktemp("tiny") / "ckpt")


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(QWEN_0P8B), trust_remote_code=False)


@pytest.fixture(scope="session")
def mini_bank(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("bank") / "mini_bank.json"
    path.write_text(json.dumps({
        "schema": "test-mini-bank", "purpose": "two text cases for the runner's mechanics",
        "prompts": [
            {"id": "mini-a", "category": "conversation", "prompt": "Say hello in one word."},
            {"id": "mini-b", "category": "arithmetic", "messages": [{"role": "user", "content": "What is 2+2?"}]},
        ],
    }), encoding="utf-8")
    return path


BANKS = CALIBRATION / "banks"
PROBES = BANKS / "matrix_campaign_probe_prompts_v1.json"
SMOKE = BANKS / "matrix_campaign_smoke_prompts_v1.json"
BANK_V2 = BANKS / "omni_review_bank_v2_draft.json"


def run_cli(args: list[str], *, expect: int = 0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["CHIMERA_OMNI_SRC"] = str(CHIMERA_SRC)
    env["OC_CALIBRATION_DIR"] = str(CALIBRATION)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([PY, str(TOOL), *args], capture_output=True, text=True, env=env, timeout=1800)
    assert proc.returncode == expect, f"exit {proc.returncode} != {expect}\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
    return proc


def base_args(tiny_dir: Path, out: Path, bank: Path) -> list[str]:
    return [
        "--model", str(tiny_dir), "--tokenizer-dir", str(QWEN_0P8B), "--label", "TINY", "--stage", "parent",
        "--bank", str(bank), "--probe-prompts", str(PROBES), "--smoke-prompts", str(SMOKE),
        "--device", "cpu", "--dtype", "float32", "--out", str(out), "--ledger-band-logit-mask",
    ]


# --------------------------------------------------------------------------
# mechanics: split decoder, streamed engine, taps
# --------------------------------------------------------------------------


def prompt_ids(tokenizer, text: str) -> torch.Tensor:
    return torch.tensor([tokenizer(text, return_tensors=None)["input_ids"]], dtype=torch.long)


def test_split_decoder_matches_reference_cached_path(tiny, tokenizer):
    tiny_dir, _ = tiny
    reference, cfg, _ = load_chimera_checkpoint(tiny_dir, device="cpu", dtype=torch.float32)
    placement = tool.placement_for(cfg, [4], ["cpu"])
    model, _cfg, receipt = tool.load_chimera_split(tiny_dir, placement=placement, dtype=torch.float32, verify_tensors=True)
    assert receipt["tensor_sha256_verified"] is True
    # identical parameters after the meta+assign load, tie included
    for (name, a), (_name, b) in zip(sorted(reference.state_dict().items()), sorted(model.state_dict().items())):
        assert name == _name and torch.equal(a, b), name
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()
    decoder = tool.SplitDecoder(model, cfg, placement, ledger_band_logit_mask=True)
    ids = prompt_ids(tokenizer, "The quick brown fox jumps over the lazy dog.")
    ours = decoder.forward(ids, None)
    theirs = reference.forward_causal_with_cache(ids, ledger_band_logit_mask=True)
    assert torch.equal(ours["logits"], theirs["logits"])
    assert torch.equal(ours["hidden"], theirs["hidden"])
    # cached step == reference cached step, and both == full-prefix recompute (the preflight)
    next_id = int(ours["logits"][0, -1].argmax())
    step = decoder.forward(torch.tensor([[next_id]]), ours["cache"])
    ref_step = reference.forward_causal_with_cache(torch.tensor([[next_id]]), cache=theirs["cache"], ledger_band_logit_mask=True)
    assert torch.equal(step["logits"], ref_step["logits"])
    full = decoder.forward(torch.cat((ids, torch.tensor([[next_id]])), dim=1), None)
    assert torch.allclose(step["logits"][0, -1], full["logits"][0, -1], atol=1e-5, rtol=1e-5)
    assert int(step["logits"][0, -1].argmax()) == int(full["logits"][0, -1].argmax())
    # the band is masked
    assert torch.isinf(ours["logits"][0, -1, BASE_VOCAB:]).all()


def test_split_across_two_segments_matches_single(tiny, tokenizer):
    tiny_dir, _ = tiny
    cfg0 = load_chimera_checkpoint(tiny_dir, device="cpu", dtype=torch.float32)[1]
    single = tool.placement_for(cfg0, [4], ["cpu"])
    two = tool.placement_for(cfg0, [1, 3], ["cpu", "cpu"])
    assert two["block_device"] == ["cpu", "cpu", "cpu", "cpu"] and two["split"] == [1, 3]
    m1, cfg, _ = tool.load_chimera_split(tiny_dir, placement=single, dtype=torch.float32, verify_tensors=False)
    m2, _, _ = tool.load_chimera_split(tiny_dir, placement=two, dtype=torch.float32, verify_tensors=False)
    d1 = tool.SplitDecoder(m1, cfg, single, ledger_band_logit_mask=False)
    d2 = tool.SplitDecoder(m2, cfg, two, ledger_band_logit_mask=False)
    ids = prompt_ids(tokenizer, "Split the trunk across two devices and nothing may move.")
    assert torch.equal(d1.forward(ids, None)["logits"], d2.forward(ids, None)["logits"])
    with pytest.raises(tool.Refusal):
        tool.placement_for(cfg, [3, 3], ["cpu", "cpu"])
    with pytest.raises(tool.Refusal):
        tool.placement_for(cfg, [2, 2], ["cpu"])


def test_block_attnres_is_refused(tmp_path):
    out, _ = build_tiny(tmp_path / "attnres", residual_mode="block_attnres")
    with pytest.raises(tool.Refusal, match="residual_mode"):
        tool.load_chimera_split(out, placement={"split": [4], "devices": ["cpu"], "block_device": ["cpu"] * 4, "head_device": "cpu"}, dtype=torch.float32, verify_tensors=False)


def test_streamed_engine_matches_resident_forward(tiny, tokenizer):
    tiny_dir, _ = tiny
    reference, cfg, _ = load_chimera_checkpoint(tiny_dir, device="cpu", dtype=torch.float32)
    engine = tool.StreamedChimera(tiny_dir, device="cpu", dtype=torch.float32, ledger_band_logit_mask=True)
    a = tokenizer("A short one.", return_tensors=None)["input_ids"]
    b = tokenizer("A much longer sequence that forces right padding of the first one in the batch.", return_tensors=None)["input_ids"]
    out = engine.run([a, b], micro_batch=8)
    assert out["block_loads"] == cfg.num_layers
    for ids, logits in ((a, out["logits"][0]), (b, out["logits"][1])):
        ref = reference(torch.tensor([ids]), ledger_band_logit_mask=True)["logits"][0]
        assert torch.allclose(logits, ref, atol=1e-5, rtol=1e-5), "streamed padded batch != resident forward"
        assert torch.equal(logits.argmax(-1), ref.argmax(-1))
    assert engine.model.lm_head.weight.data_ptr() == engine.model.embed.weight.data_ptr()


def test_taps_reproduce_gram_inputs(tiny, tokenizer):
    tiny_dir, _ = tiny
    reference, cfg, _ = load_chimera_checkpoint(tiny_dir, device="cpu", dtype=torch.float32)
    placement = tool.placement_for(cfg, [4], ["cpu"])
    model, _, _ = tool.load_chimera_split(tiny_dir, placement=placement, dtype=torch.float32, verify_tensors=False)
    taps = tool.ResidualTaps(model, normed=True)
    decoder = tool.SplitDecoder(model, cfg, placement, ledger_band_logit_mask=False)
    ids = prompt_ids(tokenizer, "Gram inputs for the C17 charts come from the FFN input of every layer.")
    decoder.forward(ids, None)
    captured = {}
    handles = [block.norm2.register_forward_hook((lambda i: lambda _m, inp, _o: captured.__setitem__(i, inp[0][0].clone()))(i)) for i, block in enumerate(reference.blocks)]
    reference(ids)
    for handle in handles:
        handle.remove()
    for layer in range(cfg.num_layers):
        dumped = torch.cat(taps.rows[layer], dim=0)
        assert torch.equal(dumped, captured[layer]), f"layer {layer} tap != hook"
        assert torch.equal(dumped.T @ dumped, captured[layer].T @ captured[layer])
    receipt = tool.write_taps(taps, Path(str(tiny_dir) + "_taps"), meta={"test": True})
    assert len(receipt["layers"]) == cfg.num_layers
    from safetensors.torch import load_file

    saved = load_file(receipt["layers"][1]["path"])
    assert torch.equal(saved["prenorm"], captured[1]) and saved["normed"].shape == captured[1].shape
    taps.remove()


# --------------------------------------------------------------------------
# CLI: refusals, schema, assess, proxy, media-out, vision cards
# --------------------------------------------------------------------------


def test_stream_with_generation_is_refused(tiny, tmp_path, mini_bank):
    tiny_dir, _ = tiny
    out = tmp_path / "refused.json"
    proc = run_cli([*base_args(tiny_dir, out, mini_bank), "--stream", "--teacher-forced"], expect=3)
    assert "refuses generation" in proc.stderr and "--maximum-new-tokens" in proc.stderr
    assert not out.exists()
    proc = run_cli([*base_args(tiny_dir, out, mini_bank), "--stream", "--maximum-new-tokens", "0"], expect=3)
    assert "--teacher-forced" in proc.stderr and not out.exists()
    proc = run_cli([*base_args(tiny_dir, out, mini_bank), "--stream", "--teacher-forced", "--maximum-new-tokens", "0"], expect=3)
    assert "--reference-dossier" in proc.stderr and not out.exists()
    proc = run_cli([*base_args(tiny_dir, out, mini_bank), "--media-out", "image"], expect=3)
    assert "S6" in proc.stderr and not out.exists()


def test_free_run_dossier_schema_and_assess(tiny, tmp_path, mini_bank):
    tiny_dir, _ = tiny
    out = tmp_path / "oc-dossier-TINY.json"
    proc = run_cli([*base_args(tiny_dir, out, mini_bank), "--maximum-new-tokens", "6", "--smoke-maximum-new-tokens", "4", "--split", "4"])
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["cases"] == 2 and summary["mode"] == "free-run" and summary["decode_path_agree"] is True
    dossier = json.loads(out.read_text(encoding="utf-8"))
    assert dossier["schema"] == OCR.DOSSIER_SCHEMA
    assert dossier["dossier_sha256"] == summary["dossier_sha256"]
    if HF_0P8B_DOSSIER.is_file():
        hf = json.loads(HF_0P8B_DOSSIER.read_text(encoding="utf-8"))
        missing_top = sorted(set(hf) - set(dossier))
        assert not missing_top, missing_top
        for block in ("bank", "generation", "emit_zero_floor", "decode_smoke", "probe_prompts", "provenance", "export", "lane_digest_validity"):
            missing = sorted(set(hf[block]) - set(dossier[block]))
            assert not missing, f"{block}: {missing}"
        missing_rec = sorted(set(hf["child_responses"][0]) - set(dossier["child_responses"][0]))
        assert not missing_rec, missing_rec
        assert sorted(set(hf["decode_smoke"]["records"][0]) - set(dossier["decode_smoke"]["records"][0])) == []
    for record in dossier["child_responses"]:
        assert record["termination"] in {"complete", "censored-at-cap"}
        assert record["generated_tokens"] == len(record["generated_token_ids"]) <= 6
        assert all(t < BASE_VOCAB for t in record["generated_token_ids"]), "the mask kept the band out"
        assert record["generated_token_ids_sha256"] == OC_CALIB.canonical_sha256({"ids": record["generated_token_ids"]})
    assert dossier["generation"]["split"] == [4] and dossier["generation"]["runtime"] == "chimera-omni-native"
    assert dossier["export"]["placement"]["block_device"] == ["cpu"] * 4
    cuda = dossier["export"]["placement"]["cuda"]
    assert cuda["devices_used"] == ["cpu"] and "CUDA_VISIBLE_DEVICES" in cuda and cuda["busy_limit_mib"] == 2000
    assert dossier["generation"]["micro_batch"] is None and dossier["export"]["checkpoint"]["rope_inv_freq_dtype"] == "torch.float32"
    assert dossier["provenance"]["directory_sha256"] == OC_CALIB.directory_digest(tiny_dir)[0]
    assert dossier["emit_zero_floor"]["probe_count"] == 8 and len(dossier["decode_smoke"]["records"]) == 3
    # oc_assess consumes it unchanged
    sheet = tmp_path / "assess"
    proc = subprocess.run([PY, str(CALIBRATION / "oc_assess.py"), "--dossier", str(out), "--bank", str(mini_bank), "--out-json", str(sheet / "assessment-TINY.json"), "--out-markdown", str(sheet / "SHEET-TINY.md")], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:] + proc.stdout[-2000:]
    assert (sheet / "assessment-TINY.json").is_file()
    assessment = json.loads((sheet / "assessment-TINY.json").read_text(encoding="utf-8"))
    assert assessment.get("dossier_sha256") == dossier["dossier_sha256"]


def test_proxy_mode_scores_reference_teacher_forced(tiny, tmp_path, mini_bank):
    tiny_dir, _ = tiny
    free = tmp_path / "free.json"
    run_cli([*base_args(tiny_dir, free, mini_bank), "--maximum-new-tokens", "6", "--skip-smoke"])
    out = tmp_path / "proxy.json"
    proc = run_cli([*base_args(tiny_dir, out, mini_bank), "--stream", "--teacher-forced", "--maximum-new-tokens", "0", "--reference-dossier", str(free), "--skip-smoke"])
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["mode"] == "teacher-forced-proxy" and summary["cases"] == 2
    dossier = json.loads(out.read_text(encoding="utf-8"))
    assert dossier["mode"] == "teacher-forced-proxy" and dossier["lane_digest_validity"]["mode"] == "teacher-forced-proxy"
    assert dossier["teacher_forced_reference"]["dossier_sha256"] == json.loads(free.read_text())["dossier_sha256"]
    reference = {r["id"]: r for r in json.loads(free.read_text())["child_responses"]}
    for record in dossier["child_responses"]:
        tf = record["teacher_forced"]
        assert tf["reference_ids_source"].startswith("reference dossier token ids")
        assert tf["reference_tokens"] == reference[record["id"]]["generated_tokens"]
        # the same weights under teacher forcing predict their own greedy tokens exactly
        assert tf["argmax_agreement"] == tf["reference_tokens"], tf
        assert record["generated_token_ids"] == reference[record["id"]]["generated_token_ids"]
        assert record["termination"] == "teacher-forced-proxy"
        assert (tf["predicts_stop_after_reference"] is True) == (reference[record["id"]]["termination"] == "complete")


def test_media_out_band_stream_flags(tiny, tmp_path):
    tiny_dir, _ = tiny
    if not BANK_V2.is_file():
        pytest.skip("no bank v2 draft")
    out = tmp_path / "media.json"
    media_dir = tmp_path / "media_out"
    proc = run_cli([*base_args(tiny_dir, out, BANK_V2), "--lanes", "music-gen-two-bar-A", "--media-out", "music", "--media-tokens", "12", "--media-out-dir", str(media_dir), "--skip-smoke", "--maximum-new-tokens", "1"])
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["media_out_records"] == 1 and summary["cases"] == 0
    dossier = json.loads(out.read_text(encoding="utf-8"))
    record = dossier["media_out"]["records"][0]
    assert record["non_empty"] is True and record["tokens"] == 12 and record["band"] == "music_codes"
    stream = json.loads(Path(record["stream_path"]).read_text(encoding="utf-8"))
    assert stream["schema"] == tool.MEDIA_STREAM_SCHEMA and len(stream["token_ids"]) == 12
    assert all(stream["band_begin"] <= t < stream["band_end"] for t in stream["token_ids"])
    assert tool.file_sha256(Path(record["stream_path"])) == record["stream_sha256"]


def test_vision_cards_run_through_the_native_bridge(tiny, tmp_path):
    tiny_dir, _ = tiny
    assets = BANKS / "assets"
    if not BANK_V2.is_file() or not (assets / "vis-count-shapes-B.png").is_file():
        pytest.skip("no bank v2 vision assets")
    pytest.importorskip("PIL")
    out = tmp_path / "vision.json"
    proc = run_cli([*base_args(tiny_dir, out, BANK_V2), "--vision-cards", "--lanes", "vis-count-shapes-B", "--assets-dir", str(assets), "--maximum-new-tokens", "4", "--skip-smoke"])
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["cases"] == 1
    record = json.loads(out.read_text(encoding="utf-8"))["child_responses"][0]
    note = record["native_media"]
    assert note["asset_id"] == "vis-count-shapes-B" and note["media_tokens"] == 256 and note["patch"] == 16
    assert record["prompt_tokens"] > 256
    assert note["source_sha256"] == tool.file_sha256(assets / "vis-count-shapes-B.png")
    assert isinstance(record["prompt_messages"][0]["content"], list)


# --------------------------------------------------------------------------
# (1) the real 0.8B Chimera checkpoint against the HF runner's dossier
# --------------------------------------------------------------------------


def test_real_0p8b_split_dossier_vs_hf_dossier():
    if not REAL_DOSSIER or not Path(REAL_DOSSIER).is_file():
        pytest.skip("set OMNI_REVIEW_0P8B_DOSSIER to the dossier the tool wrote on the 0.8B Chimera checkpoint")
    if not HF_0P8B_DOSSIER.is_file():
        pytest.skip(f"no HF dossier at {HF_0P8B_DOSSIER}")
    ours = json.loads(Path(REAL_DOSSIER).read_text(encoding="utf-8"))
    hf = json.loads(HF_0P8B_DOSSIER.read_text(encoding="utf-8"))
    assert ours["bank"]["file_sha256"] == hf["bank"]["file_sha256"] == OCR.FROZEN_V1_PACK_SHA256
    assert ours["generation"]["maximum_new_tokens"] == hf["generation"]["maximum_new_tokens"]
    assert ours["bank"]["dossier_prompt_surface_sha256"] == hf["bank"]["dossier_prompt_surface_sha256"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(QWEN_0P8B), trust_remote_code=False)
    theirs = {r["id"]: r for r in hf["child_responses"]}
    exact_text = exact_ids = same_termination = 0
    lines = []
    for record in ours["child_responses"]:
        ref = theirs[record["id"]]
        their_ids = tokenizer(ref["response_verbatim"], add_special_tokens=False, return_tensors=None)["input_ids"]
        our_ids = record["generated_token_ids"]
        common = 0
        for a, b in zip(our_ids, their_ids):
            if a != b:
                break
            common += 1
        text_match = record["response_verbatim"] == ref["response_verbatim"]
        ids_match = record["generated_token_ids_sha256"] == ref["generated_token_ids_sha256"]
        exact_text += int(text_match)
        exact_ids += int(ids_match)
        same_termination += int(record["termination"] == ref["termination"])
        assert record["prompt_tokens"] == ref["prompt_tokens"], record["id"]
        lines.append(f"  {record['id']}: prompt {record['prompt_tokens']}=={ref['prompt_tokens']} tok; ours {record['generated_tokens']} ({record['termination']}) vs HF {ref['generated_tokens']} ({ref['termination']}); common token prefix {common}; text {'==' if text_match else '!='}; ids sha {'==' if ids_match else '!='}")
    n = len(ours["child_responses"])
    print("\n0.8B split dossier vs HF dossier:\n" + "\n".join(lines))
    print(f"MATCH COUNT: exact text {exact_text}/{n}, exact token-id sha {exact_ids}/{n}, same termination {same_termination}/{n}")
    assert n == 16
    # The floor is the 2026-09-07 measurement (bf16 on sm75 vs the HF runner's bf16 on sm86): exact text 3/16,
    # exact ids 3/16, termination 16/16.  The teacher-forced fp32 proxy receipt
    # (test_real_0p8b_teacher_forced_agreement_floor) is what separates that runtime noise from a wrong path.
    assert same_termination >= REAL_SPLIT_FLOOR["same_termination"]
    assert exact_text >= REAL_SPLIT_FLOOR["exact_text"] and exact_ids >= REAL_SPLIT_FLOOR["exact_ids"]


def test_real_0p8b_teacher_forced_agreement_floor():
    """The 0.8B Chimera checkpoint teacher-forced (fp32) on the HF dossier's responses: per-lane argmax agreement."""

    if not Path(PROXY_DOSSIER).is_file():
        pytest.skip("set OMNI_REVIEW_0P8B_PROXY_DOSSIER to the proxy dossier the tool wrote on the 0.8B Chimera checkpoint")
    dossier = json.loads(Path(PROXY_DOSSIER).read_text(encoding="utf-8"))
    assert dossier["mode"] == "teacher-forced-proxy" and dossier["generation"]["dtype"] == "float32"
    block = dossier["teacher_forced_reference"]
    assert block["bank_file_sha256"] == OCR.FROZEN_V1_PACK_SHA256
    assert block["surface_check"].startswith("bank.dossier_prompt_surface_sha256 ==")
    assert block["thinking_check"] is not None
    rates = []
    total = agreed = 0
    lines = []
    for record in dossier["child_responses"]:
        tf = record["teacher_forced"]
        assert tf["reference_tokens"] > 0 and tf["micro_batch"] >= 1 and "batch_wall_time_seconds" in tf
        rates.append(float(tf["argmax_agreement_rate"]))
        total += int(tf["reference_tokens"])
        agreed += int(tf["argmax_agreement"])
        lines.append(f"  {record['id']}: {tf['argmax_agreement']}/{tf['reference_tokens']} = {tf['argmax_agreement_rate']:.4f}, nll {tf['mean_nll']:.4f}, stop {tf['predicts_stop_after_reference']}")
    n = len(rates)
    print("\n0.8B teacher-forced proxy vs HF dossier:\n" + "\n".join(lines))
    print(f"AGREEMENT: min lane {min(rates):.4f}, mean lane {sum(rates) / n:.4f}, pooled {agreed}/{total} = {agreed / total:.4f} over {n} lanes")
    assert n == 16
    assert min(rates) >= PROXY_MIN_LANE_AGREEMENT
    assert sum(rates) / n >= PROXY_MEAN_AGREEMENT


# --------------------------------------------------------------------------
# fixer 2026-09-06: one test per finding
# --------------------------------------------------------------------------


def test_micro_batch_is_threaded_and_defaults(tiny, tokenizer):
    tiny_dir, _ = tiny
    engine = tool.StreamedChimera(tiny_dir, device="cpu", dtype=torch.float32, ledger_band_logit_mask=True)
    seqs = [tokenizer(t, return_tensors=None)["input_ids"] for t in ("one", "two words here", "a third and longer sequence for the padded batch")]
    one = engine.run(seqs, micro_batch=1)
    eight = engine.run(seqs, micro_batch=8)
    assert one["micro_batch"] == 1 and one["batches"] == 3
    assert eight["micro_batch"] == 8 and eight["batches"] == 1
    for a, b in zip(one["logits"], eight["logits"]):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5) and torch.equal(a.argmax(-1), b.argmax(-1))
    with pytest.raises(tool.Refusal, match="micro_batch"):
        engine.run(seqs, micro_batch=0)

    def ns(**kw):
        return argparse.Namespace(**{"micro_batch": None, "input": None, **kw})

    assert tool.resolve_micro_batch(ns()) == 8
    assert tool.resolve_micro_batch(ns(input=Path("manifest.json"))) == 1
    assert tool.resolve_micro_batch(ns(input=Path("manifest.json"), micro_batch=4)) == 4
    with pytest.raises(tool.Refusal, match="--micro-batch"):
        tool.resolve_micro_batch(ns(micro_batch=0))


def test_streamed_oom_is_a_refusal_with_receipt(tiny, tmp_path, mini_bank):
    class FakeEngine:
        torch = torch
        device = "cuda:0"
        dtype = torch.bfloat16

    def boom():
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 12.88 GiB")

    with pytest.raises(tool.Refusal, match="OUT OF MEMORY") as info:
        tool.streamed_pass(FakeEngine(), "evidence tap pass", boom, micro_batch=8, context={"rows": 24})
    receipt = info.value.receipt
    assert receipt["schema"] == tool.REFUSAL_SCHEMA and receipt["reason"] == "out-of-memory"
    assert receipt["micro_batch"] == 8 and receipt["rows"] == 24 and receipt["pass"] == "evidence tap pass"
    assert "--micro-batch 1" in str(info.value) and "--held-only" in str(info.value)
    # a plain result passes through untouched
    assert tool.streamed_pass(FakeEngine(), "x", lambda: {"ok": 1}, micro_batch=1) == {"ok": 1}
    # main() writes the receipt beside --out and exits 3 with no dossier
    out = tmp_path / "oom" / "oc-dossier-OOM.json"
    original = tool.run

    def fake_run(_args):
        raise tool.Refusal("OUT OF MEMORY in the streamed evidence tap pass (simulated)", receipt=receipt)

    tool.run = fake_run
    try:
        code = tool.main([*base_args(tiny[0], out, mini_bank)])
    finally:
        tool.run = original
    assert code == 3 and not out.exists()
    written = json.loads((out.parent / "oc-dossier-OOM.json.refused.json").read_text(encoding="utf-8"))
    assert written["reason"] == "out-of-memory" and written["message"].startswith("OUT OF MEMORY") and written["label"] == "TINY"


def test_rope_inv_freq_stays_fp32_under_bf16(tiny, tokenizer):
    tiny_dir, _ = tiny
    reference, cfg, _ = load_chimera_checkpoint(tiny_dir, device="cpu", dtype=torch.float32)
    gqa_layers = [i for i, b in enumerate(reference.blocks) if getattr(b.mixer, "inv_freq", None) is not None]
    assert gqa_layers, "the fixture has a gated_gqa block"
    placement = tool.placement_for(cfg, [4], ["cpu"])
    model, _, receipt = tool.load_chimera_split(tiny_dir, placement=placement, dtype=torch.bfloat16, verify_tensors=False)
    assert receipt["load_dtype"] == "torch.bfloat16" and receipt["rope_inv_freq_dtype"] == "torch.float32"
    inv = reference.blocks[gqa_layers[0]].mixer.inv_freq
    assert inv.dtype == torch.float32
    for i in gqa_layers:
        assert model.blocks[i].mixer.inv_freq.dtype == torch.float32
        assert torch.equal(model.blocks[i].mixer.inv_freq, reference.blocks[i].mixer.inv_freq)
        assert model.blocks[i].mixer.q_proj.weight.dtype == torch.bfloat16
    engine = tool.StreamedChimera(tiny_dir, device="cpu", dtype=torch.bfloat16, ledger_band_logit_mask=False)
    block = engine._build_block(gqa_layers[0])
    assert block.mixer.inv_freq.dtype == torch.float32 and torch.equal(block.mixer.inv_freq, inv)
    assert block.mixer.q_proj.weight.dtype == torch.bfloat16
    # the defect's size: a bf16 inv_freq scales the angle by up to (1 + 2^-9).  The fixture's table is
    # 4 frequencies (head_dim 32 x 0.25): 0.08 rad at distance 2,047; the 27B's table (theta 1e7, 64
    # rotary dims) is 2.4 rad on its second frequency -- long-range attention leaves the teacher.
    positions = torch.arange(2048, dtype=torch.float32)

    def worst_angle_error(table: torch.Tensor) -> float:
        exact = positions[:, None] * table[None, :]
        rounded = positions[:, None] * table.to(torch.bfloat16).float()[None, :]
        return float((exact - rounded).abs().max())

    worst = worst_angle_error(inv)
    assert worst > 0.05, worst
    theta = float(cfg.gqa_rope_theta)
    table_27b = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    worst_27b = worst_angle_error(table_27b)
    assert worst_27b > 2.0, worst_27b
    # a 2,048-position bf16 forward against the fp32 reference argmax: fixed inv_freq vs the old cast
    ids = torch.tensor([tokenizer("The quick brown fox jumps over the lazy dog. " * 300, return_tensors=None)["input_ids"][:2048]], dtype=torch.long)
    assert ids.shape[1] == 2048
    decoder = tool.SplitDecoder(model, cfg, placement, ledger_band_logit_mask=False)
    ref = reference(ids)["logits"][0].argmax(-1)
    fixed = decoder.forward(ids, None)["logits"][0].float().argmax(-1)
    fixed_rate = float((fixed == ref).float().mean())
    for i in gqa_layers:
        model.blocks[i].mixer.inv_freq = inv.to(torch.bfloat16)
    cast = decoder.forward(ids, None)["logits"][0].float().argmax(-1)
    cast_rate = float((cast == ref).float().mean())
    tail = slice(1024, 2048)
    fixed_tail = float((fixed[tail] == ref[tail]).float().mean())
    cast_tail = float((cast[tail] == ref[tail]).float().mean())
    print(f"ROPE bf16 2048-position argmax agreement vs fp32: fp32 inv_freq {fixed_rate:.4f} (positions 1024+: {fixed_tail:.4f}), bf16 inv_freq (old) {cast_rate:.4f} (positions 1024+: {cast_tail:.4f}); worst angle error at 2047: fixture {worst:.3f} rad, 27B table {worst_27b:.3f} rad")
    assert fixed_rate > 0.0
    assert fixed_rate >= cast_rate, (fixed_rate, cast_rate)


def test_reference_dossier_surface_and_thinking_are_checked(tiny, tmp_path, mini_bank):
    tiny_dir, _ = tiny
    free_off = tmp_path / "free_off.json"
    run_cli([*base_args(tiny_dir, free_off, mini_bank), "--maximum-new-tokens", "4", "--skip-smoke", "--thinking", "off"])
    proxy = ["--stream", "--teacher-forced", "--maximum-new-tokens", "0", "--skip-smoke"]
    out = tmp_path / "proxy.json"
    # thinking mismatch: reference off, run template-default
    proc = run_cli([*base_args(tiny_dir, out, mini_bank), *proxy, "--reference-dossier", str(free_off)], expect=3)
    assert "thinking" in proc.stderr and not out.exists()
    # the same thinking passes, and both checks are written down
    run_cli([*base_args(tiny_dir, out, mini_bank), *proxy, "--reference-dossier", str(free_off), "--thinking", "off"])
    block = json.loads(out.read_text(encoding="utf-8"))["teacher_forced_reference"]
    assert block["thinking_requested"] == "off" and block["thinking_check"].startswith("generation.thinking.requested ==")
    assert block["surface_check"].startswith("bank.dossier_prompt_surface_sha256 ==")
    assert block["dossier_prompt_surface_sha256"] == block["this_run_prompt_surface_sha256"]
    assert block["dtype"] == "float32" and block["runtime"] == "chimera-omni-native"
    # a reference on another prompt surface is refused
    tampered = json.loads(free_off.read_text(encoding="utf-8"))
    tampered["bank"]["dossier_prompt_surface_sha256"] = "0" * 64
    bad_surface = tmp_path / "bad_surface.json"
    bad_surface.write_text(json.dumps(tampered), encoding="utf-8")
    out2 = tmp_path / "proxy2.json"
    proc = run_cli([*base_args(tiny_dir, out2, mini_bank), *proxy, "--reference-dossier", str(bad_surface), "--thinking", "off"], expect=3)
    assert "prompt surface" in proc.stderr and not out2.exists()
    # a --lanes subset renders a different surface by construction: allowed, but every case's prompt must match
    tampered2 = json.loads(free_off.read_text(encoding="utf-8"))
    tampered2["child_responses"][0]["prompt_messages"] = [{"role": "user", "content": "a different prompt"}]
    bad_prompt = tmp_path / "bad_prompt.json"
    bad_prompt.write_text(json.dumps(tampered2), encoding="utf-8")
    proc = run_cli([*base_args(tiny_dir, out2, mini_bank), *proxy, "--reference-dossier", str(bad_prompt), "--thinking", "off", "--lanes", "mini-a"], expect=3)
    assert "prompt_messages" in proc.stderr and not out2.exists()
    run_cli([*base_args(tiny_dir, out2, mini_bank), *proxy, "--reference-dossier", str(free_off), "--thinking", "off", "--lanes", "mini-b"])
    subset = json.loads(out2.read_text(encoding="utf-8"))
    assert subset["teacher_forced_reference"]["surface_check"].startswith("SUBSET") and len(subset["child_responses"]) == 1
    # an HF dossier before --thinking carries no thinking block: template-default only, assumption recorded
    legacy = json.loads(free_off.read_text(encoding="utf-8"))
    legacy["generation"].pop("thinking")
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    out3 = tmp_path / "proxy3.json"
    proc = run_cli([*base_args(tiny_dir, out3, mini_bank), *proxy, "--reference-dossier", str(legacy_path), "--thinking", "off"], expect=3)
    assert "records no chat-template thinking" in proc.stderr and not out3.exists()
    run_cli([*base_args(tiny_dir, out3, mini_bank), *proxy, "--reference-dossier", str(legacy_path)])
    block3 = json.loads(out3.read_text(encoding="utf-8"))["teacher_forced_reference"]
    assert block3["thinking_requested"] is None and "template-default assumed" in block3["thinking_check"]


def test_input_without_taps_is_refused(tiny, tmp_path, mini_bank):
    tiny_dir, _ = tiny
    out = tmp_path / "taps.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    proxy = ["--stream", "--teacher-forced", "--maximum-new-tokens", "0", "--skip-smoke"]
    proc = run_cli([*base_args(tiny_dir, out, mini_bank), *proxy, "--input", str(manifest)], expect=3)
    assert "--tap-residuals --tap-out" in proc.stderr and not out.exists()
    proc = run_cli([*base_args(tiny_dir, out, mini_bank), *proxy, "--input", str(manifest), "--tap-residuals"], expect=3)
    assert "--tap-out" in proc.stderr and not out.exists()


def test_padded_batch_taps_take_the_validity_mask(tiny, tokenizer, tmp_path):
    tiny_dir, _ = tiny
    reference, cfg, _ = load_chimera_checkpoint(tiny_dir, device="cpu", dtype=torch.float32)
    engine = tool.StreamedChimera(tiny_dir, device="cpu", dtype=torch.float32, ledger_band_logit_mask=False)
    taps = tool.ResidualTaps(engine.model, normed=True, resident=False)
    assert taps.handles == [] and taps.zero_gain == {}
    tool.stream_with_taps(engine, taps)
    a = tokenizer("short", return_tensors=None)["input_ids"]
    b = tokenizer("a considerably longer sequence so the short one is right-padded in the batch", return_tensors=None)["input_ids"]
    engine.run([a, b], taps=taps, micro_batch=8)  # tap_select None -> the validity mask
    assert sorted(taps.zero_gain) == list(range(cfg.num_layers))
    captured: dict[int, list] = {}
    handles = [block.norm2.register_forward_hook((lambda i: lambda _m, inp, _o: captured.setdefault(i, []).append(inp[0][0].clone()))(i)) for i, block in enumerate(reference.blocks)]
    reference(torch.tensor([b]))  # the batch is sorted by descending length: b's rows come first
    reference(torch.tensor([a]))
    for handle in handles:
        handle.remove()
    for layer in range(cfg.num_layers):
        dumped = torch.cat(taps.rows[layer], dim=0)
        assert dumped.shape[0] == len(a) + len(b), "one row per VALID position, none for the padding"
        assert torch.allclose(dumped, torch.cat(captured[layer], dim=0), atol=1e-5, rtol=1e-5), f"layer {layer}"
    receipt = tool.write_taps(taps, tmp_path / "taps", meta={"test": True})
    assert all(entry["all_zero_rows"] == 0 and entry["rows"] == len(a) + len(b) for entry in receipt["layers"])
    # a leaked padding row refuses the whole dump
    taps.rows[0].append(torch.zeros(1, taps.rows[0][0].shape[1]))
    with pytest.raises(tool.Refusal, match="all-zero"):
        tool.write_taps(taps, tmp_path / "taps2", meta={"test": True})
    taps.remove()


def test_taps_receipt_records_zero_gain_coordinates(tiny, tokenizer, tmp_path):
    tiny_dir, _ = tiny
    cfg = load_chimera_checkpoint(tiny_dir, device="cpu", dtype=torch.float32)[1]
    placement = tool.placement_for(cfg, [4], ["cpu"])
    model, _, _ = tool.load_chimera_split(tiny_dir, placement=placement, dtype=torch.float32, verify_tensors=False)
    with torch.no_grad():  # plant the §3b trap: exact-zero norm2 gains at layer 1, channels 5 and 9
        model.blocks[1].norm2.weight[5] = 0.0
        model.blocks[1].norm2.weight[9] = 0.0
    taps = tool.ResidualTaps(model, normed=True)
    decoder = tool.SplitDecoder(model, cfg, placement, ledger_band_logit_mask=False)
    decoder.forward(prompt_ids(tokenizer, "A dead channel must be named, never read as a fit failure."), None)
    receipt = tool.write_taps(taps, tmp_path / "zg", meta={"test": True})
    assert receipt["layers"][1]["zero_gain_coordinates"] == {"count": 2, "coordinates": [5, 9], "coordinates_truncated": False, "hidden": 64}
    assert receipt["layers"][0]["zero_gain_coordinates"]["count"] == 0
    assert "3994" in receipt["zero_gain_note"] and "prenorm" in receipt["zero_gain_note"]
    from safetensors.torch import load_file

    saved = load_file(receipt["layers"][1]["path"])
    assert bool((saved["normed"][:, 5] == 0).all()) and bool((saved["normed"][:, 9] == 0).all())
    assert not bool((saved["prenorm"][:, 5] == 0).all())
    assert json.loads((tmp_path / "zg" / "taps_receipt.json").read_text(encoding="utf-8"))["layers"][1]["zero_gain_coordinates"]["count"] == 2
    taps.remove()


def test_proxy_cli_records_micro_batch_taps_and_wall(tiny, tmp_path, mini_bank):
    tiny_dir, _ = tiny
    free = tmp_path / "free.json"
    run_cli([*base_args(tiny_dir, free, mini_bank), "--maximum-new-tokens", "5", "--skip-smoke"])
    out = tmp_path / "proxy.json"
    taps_dir = tmp_path / "taps"
    run_cli([*base_args(tiny_dir, out, mini_bank), "--stream", "--teacher-forced", "--maximum-new-tokens", "0", "--reference-dossier", str(free), "--skip-smoke", "--micro-batch", "1", "--tap-residuals", "--tap-out", str(taps_dir)])
    dossier = json.loads(out.read_text(encoding="utf-8"))
    assert dossier["generation"]["micro_batch"] == 1 and dossier["export"]["placement"]["micro_batch"] == 1
    assert dossier["residual_taps"]["micro_batch"] == 1 and dossier["residual_taps"]["dtype"] == "float32"
    receipt = json.loads((taps_dir / "taps_receipt.json").read_text(encoding="utf-8"))
    assert receipt["micro_batch"] == 1 and receipt["schema"] == tool.TAP_SCHEMA
    expected_rows = sum(r["prompt_tokens"] + r["generated_tokens"] for r in dossier["child_responses"])
    for entry in receipt["layers"]:
        assert entry["rows"] == expected_rows and entry["all_zero_rows"] == 0
        assert entry["zero_gain_coordinates"]["count"] == 0 and entry["zero_gain_coordinates"]["hidden"] == 64
    for record in dossier["child_responses"]:
        tf = record["teacher_forced"]
        assert tf["micro_batch"] == 1 and tf["batch_rows"] == 2
        assert tf["batch_wall_time_seconds"] >= record["wall_time_seconds"] > 0
        assert record["wall_time_seconds_is"].startswith("mean of the streamed batch wall")


def test_gpu_occupancy_refusal_and_inventory():
    inventory = [
        {"device": "cuda:0", "name": "NVIDIA GeForce RTX 3090", "used_mib": 927},
        {"device": "cuda:1", "name": "Quadro RTX 8000", "used_mib": 5},
    ]
    tool.check_gpu_free(inventory)
    inventory[0]["used_mib"] = 2001
    with pytest.raises(tool.Refusal, match="2000 MiB") as info:
        tool.check_gpu_free(inventory)
    assert "cuda:0" in str(info.value) and "cuda:1" not in str(info.value)
    tool.check_gpu_free(inventory, limit_mib=4000)
    assert tool.check_gpu_free([{"device": "cpu", "name": None, "used_mib": None}]) is None
    assert tool.cuda_inventory(["cpu"]) == [{"device": "cpu", "name": None, "used_mib": None}]
    if torch.cuda.is_available():
        row = tool.cuda_inventory(["cuda:0"])[0]
        assert row["name"] and row["capability"].startswith("sm") and row["total_bytes"] > row["free_bytes"] >= 0


def test_charted_block_kind_is_refused(tiny):
    class Charted:
        num_layers = 3

        @staticmethod
        def block_kind(index: int) -> str:
            return ["kda", "charted", "gated_gqa"][index]

    with pytest.raises(tool.Refusal, match="charted") as info:
        tool.placement_for(Charted(), [3], ["cpu"])
    assert "claude/omni-composite" in str(info.value) and "[1]" in str(info.value)
    cfg = load_chimera_checkpoint(tiny[0], device="cpu", dtype=torch.float32)[1]
    assert tool.refuse_charted(cfg) is None
