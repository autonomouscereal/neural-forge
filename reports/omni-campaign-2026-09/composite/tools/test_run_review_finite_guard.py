"""Tests for run_review_finite_guard.py (AI server, CPU fp32 on the runner's tiny fixture).

    cd /home/cereal/omni-campaign/composite/tools
    CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9 \
    OC_CALIBRATION_DIR=/home/cereal/omni-campaign/calibration \
    CUDA_VISIBLE_DEVICES="" /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" \
        -p no:cacheprovider -q test_run_review_finite_guard.py --basetemp <scratch>

Pins: (1) the guard's hooks are read-only -- logits/hidden/cache step bit-identical with and
without the guard, on a THREE-segment split (the P40 placement shape) and vs. one segment;
(2) an injected overflow is caught and named at the first non-finite site; (3) the CLI wrapper
writes the runner's dossier with the runner's own tool_sha256 plus a per-case guard receipt
whose bank case ids match the dossier's records.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from test_run_omni_frozen_review import (  # noqa: F401  (fixtures re-exported for pytest)
    CALIBRATION,
    charted_composite,
    CHIMERA_SRC,
    PROBES,
    QWEN_0P8B,
    SMOKE,
    TOOL,
    build_tiny,
    mini_bank,
    prompt_ids,
    tiny,
    tokenizer,
    tool,
    torch,
)

HERE = Path(__file__).resolve().parent
GUARD = HERE / "run_review_finite_guard.py"
sys.path.insert(0, str(HERE))
import run_review_finite_guard as guard_mod  # noqa: E402


def _decoder(tiny_dir: Path, split: list[int]):
    from chimera_omni.importing.safetensors_io import load_chimera_checkpoint

    cfg0 = load_chimera_checkpoint(tiny_dir, device="cpu", dtype=torch.float32)[1]
    placement = tool.placement_for(cfg0, split, ["cpu"] * len(split))
    model, cfg, _ = tool.load_chimera_split(tiny_dir, placement=placement, dtype=torch.float32, verify_tensors=False)
    return model, cfg, placement


def test_guard_hooks_are_read_only_on_three_segments(tiny, tokenizer, tmp_path):
    tiny_dir, _ = tiny
    m_ref, cfg, single = _decoder(tiny_dir, [4])
    plain = tool.SplitDecoder(m_ref, cfg, single, ledger_band_logit_mask=True)
    ids = prompt_ids(tokenizer, "Three cards, one trunk, and nothing may move.")
    ref = plain.forward(ids, None)
    ref_step = plain.forward(torch.tensor([[int(ref["logits"][0, -1].argmax())]]), ref["cache"])

    m3, _, three = _decoder(tiny_dir, [1, 1, 2])
    assert three["split"] == [1, 1, 2] and len(three["devices"]) == 3
    guard = guard_mod.FiniteGuard(tmp_path / "g.json", argv=["test"], runner=tool)
    decoder = tool.SplitDecoder(m3, cfg, three, ledger_band_logit_mask=True)
    guard.attach(m3, three)
    guard.begin_segment(int(ids.shape[-1]))
    ours = decoder.forward(ids, None)
    ours_step = decoder.forward(torch.tensor([[int(ours["logits"][0, -1].argmax())]]), ours["cache"])
    guard.flush()
    assert torch.equal(ours["logits"], ref["logits"]) and torch.equal(ours["hidden"], ref["hidden"])
    assert torch.equal(ours_step["logits"], ref_step["logits"])
    body = json.loads((tmp_path / "g.json").read_text())
    seg = body["segments"][0]
    assert seg["finite"] is True and seg["residual_nonfinite_elements"] == 0 and seg["logits_nonfinite_elements"] == 0
    assert seg["residual_max_abs"] > 0 and seg["input_tokens"] == int(ids.shape[-1])
    assert len(body["site_max_abs"]) == 2 * 4 + 2
    assert body["parameter_scan"]["nonfinite_elements"] == 0
    assert body["runner_modified"] is False and body["runner_sha256"] == guard_mod.file_sha256(TOOL)
    guard.detach()


def test_guard_names_the_first_nonfinite_site(tiny, tokenizer, tmp_path):
    tiny_dir, _ = tiny
    model, cfg, placement = _decoder(tiny_dir, [2, 2])
    with torch.no_grad():
        # block 2's FFN writes inf into the residual: block 2 norm1 input is still finite,
        # block 3 norm1 input (= residual after block 2) is the first non-finite site.
        weights = [p for _n, p in model.blocks[2].ffn.named_parameters() if p.ndim == 2]
        assert weights, "the fixture's block 2 has an FFN weight"
        for weight in weights:
            weight.fill_(float("inf"))
    guard = guard_mod.FiniteGuard(tmp_path / "g.json", argv=["test"], runner=tool)
    decoder = tool.SplitDecoder(model, cfg, placement, ledger_band_logit_mask=True)
    guard.attach(model, placement)
    guard.begin_segment(3)
    decoder.forward(prompt_ids(tokenizer, "overflow"), None)
    guard.flush()
    seg = json.loads((tmp_path / "g.json").read_text())["segments"][0]
    assert seg["finite"] is False
    assert seg["first_nonfinite_site"] == "blocks.3.norm1.in", seg
    assert seg["logits_nonfinite_elements"] > 0


def test_cli_wrapper_writes_runner_dossier_and_case_receipt(tiny, tmp_path, mini_bank):
    tiny_dir, _ = tiny
    out = tmp_path / "dossier.json"
    env = dict(os.environ)
    env["CHIMERA_OMNI_SRC"] = str(CHIMERA_SRC)
    env["OC_CALIBRATION_DIR"] = str(CALIBRATION)
    env.pop("PYTHONPATH", None)
    args = [
        sys.executable, str(GUARD), "--model", str(tiny_dir), "--tokenizer-dir", str(QWEN_0P8B), "--label", "TINY",
        "--stage", "parent", "--bank", str(mini_bank), "--probe-prompts", str(PROBES), "--smoke-prompts", str(SMOKE),
        "--device", "cpu", "--dtype", "float32", "--out", str(out), "--ledger-band-logit-mask",
        "--thinking", "off", "--maximum-new-tokens", "6", "--smoke-maximum-new-tokens", "4",
    ]
    proc = subprocess.run(args, capture_output=True, text=True, env=env, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    dossier = json.loads(out.read_text())
    assert dossier["runner"]["tool_sha256"] == guard_mod.file_sha256(TOOL)
    assert dossier["generation"]["split"] == [4]  # one CPU device; the 3-segment split is pinned in-process above
    receipt = json.loads(Path(str(out) + ".finite_guard.json").read_text())
    assert receipt["exit_code"] == 0 and receipt["dossier_file_sha256"] == guard_mod.file_sha256(out)
    bank = [s for s in receipt["segments"] if s["phase"] == "bank"]
    assert [s["case_id"] for s in bank] == [r["id"] for r in dossier["child_responses"]]
    for seg, record in zip(bank, dossier["child_responses"]):
        assert seg["input_tokens"] == record["prompt_tokens"]
        # one prefill + one forward per generated token unless the case stopped on EOS first
        assert seg["forward_calls"] in (record["generated_tokens"], record["generated_tokens"] + 1)
    phases = {s["phase"] for s in receipt["segments"]}
    assert {"preflight", "emit_zero_floor", "bank", "smoke"} <= phases
    assert receipt["summary"]["segments_nonfinite"] == 0


def test_guard_args_and_move_target_parsing():
    rest, out, staged = guard_mod.split_guard_args(["--out", "x.json", "--host-staged-moves", "--guard-out", "g.json", "--split", "16,24,24"])
    assert rest == ["--out", "x.json", "--split", "16,24,24"] and out == "g.json" and staged is True
    assert guard_mod.split_guard_args(["--out", "x"])[1:] == (None, False)
    t = torch.zeros(2)
    assert guard_mod.target_device(torch, (torch.float16,), {}) is None
    assert guard_mod.target_device(torch, ("cuda:2",), {}) == torch.device("cuda", 2)
    assert guard_mod.target_device(torch, (), {"device": "cuda:1", "dtype": torch.bool}) == torch.device("cuda", 1)
    assert guard_mod.target_device(torch, (1,), {}) == torch.device("cuda", 1)
    assert guard_mod.target_device(torch, (t,), {}) == torch.device("cpu")


def test_host_staged_moves_leave_non_cuda_calls_untouched():
    stager = guard_mod.HostStagedMoves()
    original = torch.Tensor.to
    stager.install()
    try:
        x = torch.arange(6, dtype=torch.float32)
        assert torch.equal(x.to(torch.float16), original(x, torch.float16))
        assert torch.equal(x.to("cpu", dtype=torch.float64), original(x, "cpu", dtype=torch.float64))
        assert stager.moves == 0
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            receipt = guard_mod.cross_device_integrity(stager)
            assert receipt["staged_copies_exact"] is True and stager.moves > 0
    finally:
        stager.uninstall()
    assert torch.Tensor.to is original


def test_guard_skips_charted_blocks_and_stays_read_only(charted_composite, tmp_path):
    """1.2.0: a charted composite (OMNI1 shape) attaches; charted blocks get no site and are recorded."""

    _root, out, _manifest = charted_composite
    from chimera_omni.config import ChimeraConfig
    from chimera_omni.importing.safetensors_io import read_checkpoint_config

    cfg0 = ChimeraConfig.from_manifest(read_checkpoint_config(out)["config"])
    n = int(cfg0.num_layers)
    placement = tool.placement_for(cfg0, [n], ["cpu"])
    model, cfg, _ = tool.load_chimera_split(out, placement=placement, dtype=torch.float32, verify_tensors=False)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    plain = tool.SplitDecoder(model, cfg, placement, ledger_band_logit_mask=True).forward(ids, None)
    guard = guard_mod.FiniteGuard(tmp_path / "g.json", argv=["test"], runner=tool)
    decoder = tool.SplitDecoder(model, cfg, placement, ledger_band_logit_mask=True)
    guard.attach(model, placement)
    guard.begin_segment(int(ids.shape[-1]))
    guarded = decoder.forward(ids, None)
    guard.flush()
    assert torch.equal(guarded["logits"], plain["logits"])
    body = json.loads((tmp_path / "g.json").read_text())
    charted = tool.charted_indices(cfg)
    assert body["charted_blocks_not_hooked"] == charted and len(charted) == 2
    assert len(body["site_max_abs"]) == 2 * (n - len(charted)) + 2
    assert body["segments"][0]["finite"] is True
    guard.detach()
