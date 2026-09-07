"""Tests for ``receipt_q27_streamed.py`` -- the streamed R-Q exactness receipt.

Three tiers.  (1) A tiny random-init Qwen3.5 fixture (``qf_arch.tiny_text_config``
+ ``seeded_tiny_model``, both layer kinds, saved by ``save_pretrained`` and
imported into Chimera with ``load_qwen35``) proves the MECHANICS on the CPU in
seconds: the streamed HF engine and the streamed Chimera engine are
bit-identical to the resident models, the CLI receipt carries the resident
R-Q numbers, and every refusal / budget assertion fires where the run book
says.  (2) CUDA tests (skipped without a card) prove the GPU-side assertions.
(3) The real Qwen3.5-0.8B (skipped when absent) pins the engine to
``tps_probe.build_streamed_text_model`` bit-for-bit and reproduces the R-Q
columns of ``chimera-import/qwen35_0p8b_import_receipt.json``: cuda/bf16
209/210 @ 0.5 within bf16 noise on the bf16 S0 checkpoint, and cpu/fp32
210/210 @ 3.19e-05 on an fp32 import written by ``s0_import_27b.py --dtype-out
fp32`` (WITHOUT the band) into ``$RQ_SCRATCH_DIR``.

    cd $OC/composite/tools && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \\
        /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" \\
        -p no:cacheprovider test_receipt_q27_streamed.py -q
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parents[1]
TOOL = HERE / "receipt_q27_streamed.py"
S0_TOOL = HERE / "s0_import_27b.py"
CHIMERA_SRC = Path(os.environ.get("CHIMERA_OMNI_SRC", "/home/cereal/omni-campaign/import/rq27b/tree_c5e6052"))
QWEN_0P8B = Path(os.environ.get("QWEN_0P8B_DIR", "/home/cereal/models/Qwen3.5-0.8B"))
CHIMERA_0P8B_BF16 = Path(
    os.environ.get("CHIMERA_0P8B_BF16", "/home/cereal/omni-campaign/composite/s0/qwen35_0p8b_chimera_bf16_s0")
)
SCRATCH = Path(os.environ.get("RQ_SCRATCH_DIR", "/mnt/lexar/omni-campaign/receipt-q27/scratch"))
#: The R-Q 0.8B receipt's prompt pack digest (8 prompts, 210 positions) and its two columns.
RQ_0P8B_PACK_SHA256 = "7028db47e85ebfc3b65625695279411ede88ba5f82f87fdb1e1ac93c36cad951"
RQ_0P8B_FP32_MAX_ABS_LOGIT_DIFF = 3.191828727722168e-05
RQ_0P8B_BF16_MAX_ABS_LOGIT_DIFF = 0.5
RQ_0P8B_BF16_HIDDEN = 0.015625
RQ_0P8B_POSITIONS = 210
#: The R-Q pack's per-prompt lengths, mirrored by the tiny fixture's random ids.
RQ_PACK_LENGTHS = (13, 24, 26, 9, 66, 23, 14, 35)

torch = pytest.importorskip("torch")


def _load_tool():
    spec = importlib.util.spec_from_file_location("receipt_q27_streamed", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


# ------------------------------------------------------------------ helpers
def run_cli(args: list[str], *, expect: int = 0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["CHIMERA_OMNI_SRC"] = str(CHIMERA_SRC)
    env.pop("PYTHONPATH", None)  # the tool binds the tree itself; nothing may shadow it
    proc = subprocess.run(
        [sys.executable, str(TOOL), *args], capture_output=True, text=True, env=env, timeout=7200
    )
    assert proc.returncode == expect, f"rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return proc


def summary_of(proc: subprocess.CompletedProcess) -> dict:
    return json.loads(proc.stdout.strip().splitlines()[-1])


def receipt_of(path: Path) -> dict:
    return json.load(open(path, encoding="utf-8"))


def sidecar_matches(path: Path) -> bool:
    recorded = Path(str(path) + ".sha256").read_text(encoding="utf-8").split()[0]
    return recorded == hashlib.sha256(path.read_bytes()).hexdigest()


def cuda_available() -> bool:
    return bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def tree() -> dict:
    if not (CHIMERA_SRC / "src" / "chimera_omni").is_dir():
        pytest.skip(f"no chimera-omni tree at {CHIMERA_SRC}")
    return tool.bind_chimera_tree(CHIMERA_SRC)


@pytest.fixture(scope="session")
def rq(tree):
    return tool.load_rq_script(tree["tree"])


@pytest.fixture(scope="session")
def tps(tree):
    return tool.load_tps_probe()


@pytest.fixture(scope="session")
def cpu_reference(rq):
    """The R-Q cpu column's module state: torch fallbacks + the exact recurrence.

    Applied ONCE for the session before any layer is constructed on the CPU
    (transformers binds the delta-rule function at construction).  The tool's
    own cpu columns apply the same switch in their subprocess.
    """

    from transformers.models.qwen3_5 import modeling_qwen3_5

    return rq.disable_fast_paths(modeling_qwen3_5, recurrent_reference=True)


@pytest.fixture(scope="session")
def tiny(tmp_path_factory, tree, rq, cpu_reference) -> dict:
    """Tiny HF parent (both layer kinds, untied) + its exact Chimera import + a prompt pack."""

    sys.path.insert(0, str(CAMPAIGN_ROOT / "qwen-fold"))
    import qf_arch  # noqa: WPS433

    from chimera_omni.importing.profiles import qwen35_chimera_config
    from chimera_omni.importing.qwen35 import load_qwen35
    from chimera_omni.importing.safetensors_io import write_safetensors

    work = tmp_path_factory.mktemp("rq_streamed")
    hf_dir = work / "hf"
    config = qf_arch.tiny_text_config(layers_count=8)
    model = qf_arch.seeded_tiny_model(config)
    hf_dir.mkdir()
    model.save_pretrained(str(hf_dir), safe_serialization=True)
    del model

    hf_config = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    cfg = qwen35_chimera_config(hf_config)
    chimera, _receipt = load_qwen35(hf_dir, cfg, device="cpu", dtype=torch.float32, prefix="model.")
    chimera_dir = work / "chimera"
    write_safetensors(
        chimera.state_dict(),
        chimera_dir,
        config=cfg.to_manifest(),
        ledger=cfg.ledger().to_manifest(),
        source={"import_source": "qwen3_5", "model_dir": str(hf_dir)},
        chimera_commit=tree["commit_declared"] or "fixture",
    )
    del chimera

    generator = torch.Generator().manual_seed(20260906)
    packs = [
        torch.randint(0, int(config.vocab_size), (length,), generator=generator).tolist()
        for length in RQ_PACK_LENGTHS
    ]
    prompt_ids = work / "prompt_ids.json"
    prompt_ids.write_text(json.dumps(packs), encoding="utf-8")
    return {
        "work": work,
        "hf_dir": hf_dir,
        "chimera_dir": chimera_dir,
        "cfg": cfg,
        "packs": packs,
        "prompt_ids": prompt_ids,
        "base_vocab": int(cfg.ledger_base_vocab_size),
    }


def resident_rq_numbers(tiny: dict, rq, device: str = "cpu") -> dict:
    """The R-Q column loop on RESIDENT models: reference HF vs imported Chimera, fp32."""

    import transformers

    from chimera_omni.importing.safetensors_io import load_chimera_checkpoint

    reference = transformers.Qwen3_5ForCausalLM.from_pretrained(
        str(tiny["hf_dir"]), dtype=torch.float32, attn_implementation="eager"
    ).to(device).eval()
    chimera, _cfg, _receipt = load_chimera_checkpoint(tiny["chimera_dir"], device=device, dtype=torch.float32)
    chimera.eval()
    base = tiny["base_vocab"]
    out = {"hf_logits": [], "chimera_logits": [], "hf_hidden": [], "chimera_hidden": [], "prompts": []}
    worst = 0.0
    agree_total = 0
    positions_total = 0
    for ids in tiny["packs"]:
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        ref_states: list = [None] * len(reference.model.layers)
        ch_states: list = [None] * len(chimera.blocks)
        handles = rq.capture_layer_outputs(list(reference.model.layers), ref_states)
        handles += rq.capture_layer_outputs(list(chimera.blocks), ch_states)
        try:
            with torch.no_grad():
                ref_logits = reference(input_ids=input_ids).logits.float().cpu()
                ch_logits = chimera(input_ids, ledger_band_logit_mask=True)["logits"].float().cpu()
        finally:
            for handle in handles:
                handle.remove()
        out["hf_logits"].append(ref_logits)
        out["chimera_logits"].append(ch_logits)
        out["hf_hidden"].append({i: ref_states[i].float().cpu() for i in range(4)})
        out["chimera_hidden"].append({i: ch_states[i].float().cpu() for i in range(4)})
        diff = (ch_logits[..., :base] - ref_logits[..., :base]).abs()
        worst = max(worst, float(diff.max()))
        agree = int((ch_logits[..., :base].argmax(-1) == ref_logits[..., :base].argmax(-1)).sum())
        agree_total += agree
        positions_total += int(ref_logits.shape[1])
        out["prompts"].append({"max_abs_logit_diff": float(diff.max()), "agree": agree})
    out["max_abs_logit_diff"] = worst
    out["argmax_agreement"] = agree_total / positions_total
    out["argmax_positions"] = positions_total
    return out


# ---------------------------------------------------------------- unit tests
def test_check_geometry_accepts_the_tiny_import_and_refuses_each_drift(tiny):
    manifest = json.loads((tiny["chimera_dir"] / "config.json").read_text(encoding="utf-8"))["config"]
    text = tool.hf_text_config(tiny["hf_dir"])
    facts = tool.check_geometry(manifest, text)
    assert facts["num_layers"] == 8 and facts["block_kinds"] == ["kda", "kda", "kda", "gated_gqa"] * 2
    for key, value in (
        ("hidden_size", manifest["hidden_size"] + 1),
        ("num_layers", manifest["num_layers"] + 1),
        ("ledger_base_vocab_size", manifest["ledger_base_vocab_size"] - 1),
        ("residual_mode", "block_attnres"),
        ("block_pattern", ["kda", "kda", "gated_gqa", "kda"]),
    ):
        drifted = dict(manifest)
        drifted[key] = value
        with pytest.raises(tool.Refusal) as excinfo:
            tool.check_geometry(drifted, text)
        assert key in str(excinfo.value) or key == "block_pattern" and "block schedule" in str(excinfo.value)


def test_check_budgets_names_each_limit_and_decides_passed():
    ok = tool.check_budgets(
        peak_rss_bytes_value=10 * tool.GIB, sampled_rss_bytes=11 * tool.GIB, reserved_bytes=2 * tool.GIB,
        max_host_rss_gib=12.0, max_gpu_reserved_gib=3.0, cuda=True,
    )
    assert ok["passed"] and ok["host_rss"]["peak_rss_bytes"] == 11 * tool.GIB
    host_fail = tool.check_budgets(
        peak_rss_bytes_value=13 * tool.GIB, sampled_rss_bytes=0, reserved_bytes=0,
        max_host_rss_gib=12.0, max_gpu_reserved_gib=3.0, cuda=False,
    )
    assert not host_fail["passed"] and not host_fail["host_rss"]["passed"] and host_fail["gpu_reserved"]["passed"]
    gpu_fail = tool.check_budgets(
        peak_rss_bytes_value=1, sampled_rss_bytes=1, reserved_bytes=4 * tool.GIB,
        max_host_rss_gib=12.0, max_gpu_reserved_gib=3.0, cuda=True,
    )
    assert not gpu_fail["passed"] and not gpu_fail["gpu_reserved"]["passed"]
    assert abs(tool.DEFAULT_MAX_HOST_RSS_GIB - 65.19) < 0.01 and abs(tool.DEFAULT_MAX_GPU_RESERVED_GIB - 16.76) < 0.01


# ------------------------------------------------------ tiny: mechanics (CPU)
def test_tiny_streamed_engines_are_bit_identical_to_the_resident_models(tiny, rq, tps, cpu_reference):
    """Streaming changes no arithmetic: logits AND first-four hiddens are equal bit for bit."""

    resident = resident_rq_numbers(tiny, rq)
    packs = tiny["packs"]
    hf_logits, hf_hidden, hf_facts = tool.run_hf_pass(tps, tiny["hf_dir"], "cpu", torch.float32, packs, layers_reported=4)
    assert hf_facts["layer_loads"] == 8 * len(packs) and hf_facts["tensor_prefix"] == "model"
    for produced, expected, hid, hid_expected in zip(hf_logits, resident["hf_logits"], hf_hidden, resident["hf_hidden"]):
        assert torch.equal(produced, expected)
        assert sorted(hid) == [0, 1, 2, 3]
        for layer in range(4):
            assert torch.equal(hid[layer], hid_expected[layer])

    document = json.loads((tiny["chimera_dir"] / "config.json").read_text(encoding="utf-8"))
    index = json.loads((tiny["chimera_dir"] / "model.safetensors.index.json").read_text(encoding="utf-8"))
    ch_logits, ch_hidden, ch_facts = tool.run_chimera_pass(
        tiny["chimera_dir"], document, index, "cpu", torch.float32, packs, verify=True, layers_reported=4
    )
    assert ch_facts["layer_loads"] == 8 * len(packs)
    # every block load verifies each of its tensors against the index (>= one tensor per load)
    assert ch_facts["tensors_sha256_verified"] >= ch_facts["layer_loads"] > 0
    assert ch_facts["block_buffers"] == ["gated_gqa:256", "kda:256"]
    for produced, expected, hid, hid_expected in zip(ch_logits, resident["chimera_logits"], ch_hidden, resident["chimera_hidden"]):
        assert torch.equal(produced, expected)
        for layer in range(4):
            assert torch.equal(hid[layer], hid_expected[layer])
        assert not torch.isfinite(produced[..., tiny["base_vocab"]:]).any()


def test_tiny_cli_receipt_carries_the_resident_rq_numbers(tiny, rq, cpu_reference):
    resident = resident_rq_numbers(tiny, rq)
    out = tiny["work"] / "receipt_cpu_fp32.json"
    proc = run_cli(
        [
            "--chimera-dir", str(tiny["chimera_dir"]), "--hf-dir", str(tiny["hf_dir"]),
            "--prompt-ids", str(tiny["prompt_ids"]), "--device", "cpu", "--dtype", "float32",
            "--min-positions", "1", "--out", str(out),
        ]
    )
    summary = summary_of(proc)
    receipt = receipt_of(out)
    assert sidecar_matches(out) and summary["receipt_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert receipt["schema"] == "chimera_omni.receipt.import.qwen35.v1" and receipt["mode"] == "streamed-teacher-forced"
    column = receipt["columns"][0]
    assert column["device"] == "cpu" and column["dtype"] == "float32" and column["mode"] == "streamed-teacher-forced"
    assert column["fast_paths_disabled"]["multi_token_path"] == "torch_recurrent_gated_delta_rule"
    assert column["summary"]["argmax_positions"] == sum(RQ_PACK_LENGTHS) == resident["argmax_positions"]
    assert column["summary"]["max_abs_logit_diff"] == resident["max_abs_logit_diff"]
    assert column["summary"]["argmax_agreement"] == resident["argmax_agreement"]
    assert column["summary"]["band_logits_finite_total"] == 0
    for prompt, expected in zip(column["prompts"], resident["prompts"]):
        assert prompt["max_abs_logit_diff"] == expected["max_abs_logit_diff"]
        assert [row["kind"] for row in prompt["per_layer_hidden_max_abs_diff"]] == ["kda", "kda", "kda", "gated_gqa"]
    assert receipt["assertions"]["passed"] is True and receipt["assertions"]["gpu_reserved"]["applies"] is False
    assert column["peak_rss_bytes"] > 0 and receipt["inputs"]["prompt_pack_sha256"] == tool.ids_sha256(tiny["packs"])
    assert receipt["inputs"]["identity"]["chimera"]["tensor_sha256_recorded"] > 0
    assert receipt["chimera"]["tps_probe_sha256"] and receipt["chimera"]["rq_script_sha256"]
    # A second run refuses to overwrite; --overwrite replaces.
    run_cli(["--chimera-dir", str(tiny["chimera_dir"]), "--hf-dir", str(tiny["hf_dir"]), "--prompt-ids", str(tiny["prompt_ids"]),
             "--device", "cpu", "--dtype", "float32", "--min-positions", "1", "--out", str(out)], expect=3)


@pytest.mark.parametrize(
    "field,value,needle",
    [
        ("hidden_size", lambda v: v + 1, "hidden_size"),
        ("num_layers", lambda v: v + 1, "num_layers"),
        ("ledger_base_vocab_size", lambda v: v - 1, "ledger_base_vocab_size"),
        ("residual_mode", lambda v: "block_attnres", "residual_mode"),
    ],
)
def test_manifest_drift_is_refused_before_any_weight_is_read(tiny, field, value, needle):
    """Only config.json + the index are copied: a refusal that read a shard would have crashed instead."""

    fake = tiny["work"] / f"drift_{field}"
    if fake.exists():
        shutil.rmtree(fake)
    fake.mkdir()
    document = json.loads((tiny["chimera_dir"] / "config.json").read_text(encoding="utf-8"))
    document["config"][field] = value(document["config"][field])
    (fake / "config.json").write_text(json.dumps(document), encoding="utf-8")
    shutil.copy(tiny["chimera_dir"] / "model.safetensors.index.json", fake / "model.safetensors.index.json")
    out = fake / "receipt.json"
    proc = run_cli(
        ["--chimera-dir", str(fake), "--hf-dir", str(tiny["hf_dir"]), "--prompt-ids", str(tiny["prompt_ids"]),
         "--device", "cpu", "--dtype", "float32", "--min-positions", "1", "--out", str(out)],
        expect=3,
    )
    assert "REFUSED" in proc.stderr and needle in proc.stderr
    assert not out.exists()


def test_prompt_pack_below_min_positions_is_refused(tiny):
    out = tiny["work"] / "too_short.json"
    proc = run_cli(
        ["--chimera-dir", str(tiny["chimera_dir"]), "--hf-dir", str(tiny["hf_dir"]), "--prompt-ids", str(tiny["prompt_ids"]),
         "--device", "cpu", "--dtype", "float32", "--min-positions", "100000", "--out", str(out)],
        expect=3,
    )
    assert "min-positions" in proc.stderr and not out.exists()


def test_host_rss_budget_failure_writes_the_receipt_and_exits_4(tiny):
    out = tiny["work"] / "receipt_rss_budget.json"
    proc = run_cli(
        ["--chimera-dir", str(tiny["chimera_dir"]), "--hf-dir", str(tiny["hf_dir"]), "--prompt-ids", str(tiny["prompt_ids"]),
         "--device", "cpu", "--dtype", "float32", "--min-positions", "1", "--max-host-rss-gib", "0.001", "--out", str(out)],
        expect=4,
    )
    assert "BUDGET ASSERTION FAILED" in proc.stderr
    receipt = receipt_of(out)
    assert receipt["assertions"]["passed"] is False and receipt["assertions"]["host_rss"]["passed"] is False
    assert receipt["assertions"]["host_rss"]["limit_gib"] == 0.001
    assert summary_of(proc)["assertions_passed"] is False and sidecar_matches(out)


# ------------------------------------------------------------- tiny: CUDA
@pytest.mark.skipif(not cuda_available(), reason="no CUDA device visible")
def test_tiny_cuda_gpu_reserve_assertion_exits_4(tiny):
    out = tiny["work"] / "receipt_gpu_budget.json"
    proc = run_cli(
        ["--chimera-dir", str(tiny["chimera_dir"]), "--hf-dir", str(tiny["hf_dir"]), "--prompt-ids", str(tiny["prompt_ids"]),
         "--device", "cuda:0", "--dtype", "float32", "--min-positions", "1", "--max-gpu-reserved-gib", "0.00001",
         "--out", str(out)],
        expect=4,
    )
    receipt = receipt_of(out)
    assert receipt["assertions"]["gpu_reserved"]["applies"] is True
    assert receipt["assertions"]["gpu_reserved"]["passed"] is False and receipt["assertions"]["host_rss"]["passed"] is True
    assert receipt["assertions"]["gpu_reserved"]["peak_reserved_bytes"] > 0
    assert receipt["columns"][0]["fast_path_available"] in (True, False)
    assert receipt["columns"][0]["summary"]["argmax_positions"] == sum(RQ_PACK_LENGTHS)


@pytest.mark.skipif(not cuda_available(), reason="no CUDA device visible")
def test_tiny_cuda_busy_card_is_refused_before_anything_runs(tiny):
    out = tiny["work"] / "receipt_busy.json"
    proc = run_cli(
        ["--chimera-dir", str(tiny["chimera_dir"]), "--hf-dir", str(tiny["hf_dir"]), "--prompt-ids", str(tiny["prompt_ids"]),
         "--device", "cuda:0", "--dtype", "float32", "--min-positions", "1", "--max-preexisting-used-mib", "0",
         "--out", str(out)],
        expect=3,
    )
    assert "busy card" in proc.stderr and not out.exists()


# ------------------------------------------------------------ real 0.8B
real_0p8b = pytest.mark.skipif(
    not (QWEN_0P8B / "config.json").is_file() or not (CHIMERA_0P8B_BF16 / "config.json").is_file(),
    reason="Qwen3.5-0.8B or its S0 Chimera bf16 checkpoint is not on this host",
)


@real_0p8b
def test_0p8b_hf_engine_is_bit_identical_to_tps_probe_build_streamed_text_model(tree, rq, tps, cpu_reference):
    """The generalised builder IS tps_probe's engine: same logits, bit for bit, on the real 0.8B."""

    packs = rq.tokenize(QWEN_0P8B, rq.PROMPT_PACK[:2], 128)
    trunk, head, extra = tps.build_streamed_text_model(QWEN_0P8B, "cpu", torch.float32, "eager")
    probe_logits = []
    with torch.no_grad():
        for ids in packs:
            hidden = trunk(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False).last_hidden_state
            probe_logits.append(head(hidden).float())
    del trunk, head, extra
    ours, _hidden, facts = tool.run_hf_pass(tps, QWEN_0P8B, "cpu", torch.float32, packs, layers_reported=4)
    assert facts["tensor_prefix"] == "model.language_model" and facts["lm_head_tied_to_embedding"] is True
    for produced, expected in zip(ours, probe_logits):
        assert torch.equal(produced, expected)


@real_0p8b
@pytest.mark.skipif(not cuda_available(), reason="no CUDA device visible")
def test_0p8b_cuda_bf16_column_reproduces_the_rq_bf16_column_within_bf16_noise(tmp_path):
    """R-Q on GPU6 (cuda/bf16, fla chunked kernel): 209/210, max 0.5, hidden 0.015625.

    bf16 noise is hardware- and kernel-dependent, so the tolerance is +-1 position,
    a factor of two on the max-abs logit diff and four on the first-layer hiddens.
    """

    out = tmp_path / "receipt_0p8b_cuda_bf16.json"
    proc = run_cli(
        ["--chimera-dir", str(CHIMERA_0P8B_BF16), "--hf-dir", str(QWEN_0P8B), "--device", "cuda:0",
         "--dtype", "bfloat16", "--out", str(out)]
    )
    receipt = receipt_of(out)
    column = receipt["columns"][0]
    summary = column["summary"]
    print(f"0.8B cuda/bf16 streamed: {summary}")
    assert receipt["inputs"]["prompt_pack_sha256"] == RQ_0P8B_PACK_SHA256
    assert summary["argmax_positions"] == RQ_0P8B_POSITIONS
    assert summary["argmax_matching_positions"] >= RQ_0P8B_POSITIONS - 2
    assert RQ_0P8B_BF16_MAX_ABS_LOGIT_DIFF / 2 <= summary["max_abs_logit_diff"] <= RQ_0P8B_BF16_MAX_ABS_LOGIT_DIFF * 2
    assert summary["max_abs_hidden_diff_first_layers"] <= RQ_0P8B_BF16_HIDDEN * 4
    assert column["reference_multi_token_path"].startswith("fla")
    assert receipt["assertions"]["passed"] is True
    assert receipt["assertions"]["gpu_reserved"]["peak_reserved_gib"] < tool.DEFAULT_MAX_GPU_RESERVED_GIB
    assert column["engines"]["chimera"]["tensors_sha256_verified"] > 0


@pytest.fixture(scope="session")
def chimera_0p8b_fp32(tree) -> Path:
    """An fp32 Chimera import of the 0.8B WITHOUT the band, by the canonical S0 tool.

    The R-Q cpu/fp32 column held `1 + w` norm gains in fp32; the bf16 S0
    checkpoint rounds them once, so reproducing 3.19e-05 needs an fp32 write.
    """

    if not (QWEN_0P8B / "config.json").is_file():
        pytest.skip("Qwen3.5-0.8B not on this host")
    if not S0_TOOL.is_file():
        pytest.skip(f"{S0_TOOL} is absent")
    out = SCRATCH / "qwen35_0p8b_chimera_fp32_noband"
    if (out / "import_receipt.json").is_file() and (out / "model.safetensors.index.json").is_file():
        return out
    SCRATCH.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CHIMERA_OMNI_SRC"] = str(CHIMERA_SRC)
    env.pop("PYTHONPATH", None)
    args = [sys.executable, str(S0_TOOL), "--src", str(QWEN_0P8B), "--chimera-commit", tree["commit_declared"] or "c5e6052",
            "--out", str(out), "--dtype-out", "fp32"]
    if out.exists():
        args.append("--overwrite")
    proc = subprocess.run(args, capture_output=True, text=True, env=env, timeout=7200)
    assert proc.returncode == 0, f"s0 import rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return out


@real_0p8b
def test_0p8b_cpu_fp32_column_reproduces_the_rq_exactness_column(chimera_0p8b_fp32, tmp_path):
    """R-Q cpu/fp32 (torch fallbacks, recurrent delta rule): 210/210, max-abs 3.19e-05."""

    out = tmp_path / "receipt_0p8b_cpu_fp32.json"
    proc = run_cli(
        ["--chimera-dir", str(chimera_0p8b_fp32), "--hf-dir", str(QWEN_0P8B), "--device", "cpu",
         "--dtype", "float32", "--out", str(out)]
    )
    receipt = receipt_of(out)
    column = receipt["columns"][0]
    summary = column["summary"]
    print(f"0.8B cpu/fp32 streamed: {summary}")
    assert receipt["inputs"]["prompt_pack_sha256"] == RQ_0P8B_PACK_SHA256
    assert column["fast_paths_disabled"]["multi_token_path"] == "torch_recurrent_gated_delta_rule"
    assert summary["argmax_positions"] == RQ_0P8B_POSITIONS
    assert summary["argmax_matching_positions"] == RQ_0P8B_POSITIONS
    assert RQ_0P8B_FP32_MAX_ABS_LOGIT_DIFF / 2 <= summary["max_abs_logit_diff"] <= RQ_0P8B_FP32_MAX_ABS_LOGIT_DIFF * 2
    assert summary["max_abs_hidden_diff_first_layers"] <= 4e-6
    assert receipt["assertions"]["passed"] is True
