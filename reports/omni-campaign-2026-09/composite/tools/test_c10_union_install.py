"""Tests for ``c10_union_install.py`` -- CPU, fp32, synthetic Chimera + synthetic form-E K3.

The run-book row's six tests, by name:

  1 test_k3_channels_zero_equals_trunk_byte_for_byte
  2 test_k3_dense_zeros_binds_the_veto_on_every_layer
  3 test_qwen_rows_are_always_kept
  4 test_situ_to_silu_scalar_reproduces_situ_within_recorded_relL2
  5 test_size_refusal_on_a_synthetic_oversize_child
  6 test_form_e_lift_equals_form_m_lift_on_random_inputs

plus the section-3b traps this tool must survive or name:
  test_exact_zero_routed_expert_norm_gain_is_counted_not_divided
  test_non_finite_k3_tensor_is_a_named_refusal
  test_alignment_covers_every_k3_layer_exactly_once
  test_child_loads_and_runs_at_the_uniform_width

plus one test per review finding (2026-09-06):
  B1 test_charts_dir_in_c17_layout_installs_and_records_the_arm
  B2 test_c17_alignment_json_is_parsed_and_sealed
  B3 test_chart_direction_and_biases_follow_c17_convention
  B4 test_streamed_engine_equals_resident_byte_for_byte
  M5 test_k3_layers_load_lazily_and_the_check_gram_is_kept_columns_only
  M6 test_stop_and_resume_reproduce_the_single_pass_child
  m7 test_receipt_scores_are_measured_on_the_shipped_dtype
  m8 test_k3_tensors_move_to_device_once_per_install
  m9 test_k3_config_is_required_and_recorded
  m10 test_overwrite_refuses_foreign_dirs_and_input_overlap

The chart fixture is written the way c17_chart_fit writes it: ``L{k:02d}/ridge.safetensors``
(``C_in``, ``C_in_bias``, ``C_out``, ``C_out_bias``), ``L{k:02d}/layer.json``, ``alignment.json``
``{schema, summary, table}``.

Server:
    cd $OC/composite/tools && CUDA_VISIBLE_DEVICES="" \\
        /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" \\
        -p no:cacheprovider test_c10_union_install.py -q
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c10_union_install as c10  # noqa: E402
import qf_corpus  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from chimera_omni.config import ChimeraConfig  # noqa: E402
from chimera_omni.importing.safetensors_io import (  # noqa: E402
    load_chimera_checkpoint,
    read_safetensors,
    write_safetensors,
)
from chimera_omni.modeling import ChimeraOmniModel  # noqa: E402

HIDDEN = 32
TRUNK_WIDTH = 24
PARENT_WIDTH = 48
LAYERS = 4
K3_HIDDEN = 40
K3_LATENT = 16
K3_ROUTED = 12
K3_SHARED = 8
K3_DENSE = 20
K3_MOE_LAYERS = 6
VOCAB = 2048
SEQ = 64
BATCHES = 16
HELD_EVERY = 4


# ------------------------------------------------------------------ fixtures
def chimera_config(width: int) -> ChimeraConfig:
    cfg = ChimeraConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=width, num_layers=LAYERS, num_heads=4,
        num_kv_heads=4, max_sequence_length=128, block_pattern=("kda", "gated_gqa", "kda", "gated_gqa"),
        residual_mode="none", attention_head_dim=8, kda_head_dim=8, kda_value_head_dim=8, kda_decay_rank=8,
        native_feature_dim=32,
    )
    cfg.validate()
    return cfg


def write_model(model: ChimeraOmniModel, cfg: ChimeraConfig, directory: Path) -> None:
    write_safetensors(model.state_dict(), directory, config=cfg.to_manifest(), ledger=cfg.ledger().to_manifest(),
                      source={"import_source": "c10 unit test"}, chimera_commit="b" * 40)


@pytest.fixture(scope="module")
def work(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("c10")


@pytest.fixture(scope="module")
def trunks(work: Path) -> dict:
    """A random parent (width 48) and a trunk (width 24: a fixed row subset, down columns carried)."""
    torch.manual_seed(1)
    parent_cfg = chimera_config(PARENT_WIDTH)
    parent = ChimeraOmniModel(parent_cfg).float()
    parent_dir = work / "parent"
    write_model(parent, parent_cfg, parent_dir)
    trunk_cfg = chimera_config(TRUNK_WIDTH)
    trunk = ChimeraOmniModel(trunk_cfg).float()
    state = {k: v.clone() for k, v in parent.state_dict().items()}
    keep = torch.arange(0, PARENT_WIDTH, 2)[:TRUNK_WIDTH]
    probe = torch.randn(1024, HIDDEN)   # post-norm inputs have unit rms; a C2-style refit of down
    for k in range(LAYERS):
        up = state[f"blocks.{k}.ffn.up.weight"]
        down = state[f"blocks.{k}.ffn.down.weight"]
        gate_q, up_q = up[:PARENT_WIDTH][keep], up[PARENT_WIDTH:][keep]
        target = (torch.nn.functional.silu(probe @ up[:PARENT_WIDTH].T) * (probe @ up[PARENT_WIDTH:].T)) @ down.T
        design = torch.nn.functional.silu(probe @ gate_q.T) * (probe @ up_q.T)
        solved = torch.linalg.lstsq(design, target).solution.T      # [h x Q]
        state[f"blocks.{k}.ffn.up.weight"] = torch.cat([gate_q, up_q]).contiguous()
        state[f"blocks.{k}.ffn.down.weight"] = solved.contiguous()
    trunk.load_state_dict(state, strict=True)
    trunk_dir = work / "trunk"
    write_model(trunk, trunk_cfg, trunk_dir)
    evidence = work / "evidence.json"
    qf_corpus.synthetic_manifest(vocab_size=VOCAB, sequence_length=SEQ, batches=BATCHES, held_every=HELD_EVERY,
                                 out=evidence, reserved=8)
    return {"parent": parent_dir, "trunk": trunk_dir, "evidence": evidence, "keep": keep}


def make_k3_dense(root: Path, *, zeros: bool = False, seed: int = 7, zero_gain_layer: int | None = None,
                  nan_layer: int | None = None) -> dict:
    g = torch.Generator().manual_seed(seed)

    def rand(*shape, scale=1.0):
        t = torch.randn(*shape, generator=g) * scale
        return torch.zeros(*shape) if zeros else t

    raw = {}
    # read-out scales are small so the K3 term is a PERTURBATION of the parent MLP
    # (energy fraction well below 1), the regime the real install lives in.  Only the
    # read-outs matter: routed_expert_norm makes the routed write scale-invariant to
    # the latent maps, which is exactly why the first fixture read energy ~ 1.0.
    dense = {"mlp.gate_proj.weight": rand(K3_DENSE, K3_HIDDEN, scale=0.3),
             "mlp.up_proj.weight": rand(K3_DENSE, K3_HIDDEN, scale=0.3),
             "mlp.down_proj.weight": rand(K3_HIDDEN, K3_DENSE, scale=0.005)}
    (root / "L00").mkdir(parents=True, exist_ok=True)
    save_file(dense, str(root / "L00" / "mlp.safetensors"), metadata={"layer": "0", "kind": "dense_mlp_verbatim"})
    raw[0] = dense
    for l in range(1, K3_MOE_LAYERS + 1):
        gain = torch.ones(K3_LATENT) + 0.1 * torch.randn(K3_LATENT, generator=g)
        if zero_gain_layer == l:
            gain[3] = 0.0
        w1 = rand(K3_ROUTED, K3_LATENT, scale=0.5)
        if nan_layer == l:
            w1 = w1.clone()
            w1[0, 0] = float("nan")
        layer = {
            "w1_s": w1, "w3_s": rand(K3_ROUTED, K3_LATENT, scale=0.5), "w2p": rand(K3_LATENT, K3_ROUTED, scale=0.5),
            "channels": torch.stack([torch.arange(K3_ROUTED) % 3, torch.arange(K3_ROUTED)], 1).to(torch.int32),
            "block_sparse_moe.routed_expert_down_proj.weight": rand(K3_LATENT, K3_HIDDEN, scale=0.3),
            "block_sparse_moe.routed_expert_up_proj.weight": rand(K3_HIDDEN, K3_LATENT, scale=0.005),
            "block_sparse_moe.routed_expert_norm.weight": torch.zeros(K3_LATENT) if zeros else gain,
            "block_sparse_moe.shared_experts.gate_proj.weight": rand(K3_SHARED, K3_HIDDEN, scale=0.3),
            "block_sparse_moe.shared_experts.up_proj.weight": rand(K3_SHARED, K3_HIDDEN, scale=0.3),
            "block_sparse_moe.shared_experts.down_proj.weight": rand(K3_HIDDEN, K3_SHARED, scale=0.005),
        }
        (root / f"L{l:02d}").mkdir(parents=True, exist_ok=True)
        save_file(layer, str(root / f"L{l:02d}" / "ffn.safetensors"), metadata={"form": "E", "layer": str(l)})
        raw[l] = layer
    return raw


def c17_alignment_doc(moe_layers: int = K3_MOE_LAYERS, trunk_layers: int = LAYERS) -> dict:
    """What c17_chart_fit.alignment_table / alignment_summary write (dense layer 0 -> trunk 0)."""
    table = []
    for k in range(trunk_layers):
        start = 1 + (moe_layers * k) // trunk_layers
        stop = 1 + (moe_layers * (k + 1)) // trunk_layers
        moe = list(range(start, stop))
        k3 = ([0] if k == 0 else []) + moe
        table.append({"trunk": k, "k3_layers": k3, "moe_layers": moe, "moe_start": start, "moe_stop": stop})
    covered = [l for row in table for l in row["k3_layers"]]
    summary = {"k3_moe_layers": moe_layers, "trunk_layers": trunk_layers, "dense_layer0": "trunk0",
               "k3_layers_covered": len(covered),
               "covered_exactly_once": sorted(covered) == list(range(moe_layers + 1)) and len(covered) == len(set(covered)),
               "rule": "trunk k <- K3 MoE layers [1 + floor(M k / N), 1 + floor(M (k+1) / N)); dense layer 0 -> trunk 0"}
    return {"schema": c10.C17_ALIGNMENT_SCHEMA, "summary": summary, "table": table}


def write_c17_chart(layer_dir: Path, k: int, c_in: torch.Tensor, c_in_bias: torch.Tensor, c_out: torch.Tensor,
                    c_out_bias: torch.Tensor, arm: str = "ridge", rows_key: str = "normed") -> Path:
    """One trunk layer the way c17 writes it: <arm>.safetensors + layer.json (convention + file sha)."""
    layer_dir.mkdir(parents=True, exist_ok=True)
    path = layer_dir / c10.CHART_ARM_FILES[arm]
    save_file({"C_in": c_in.contiguous(), "C_in_bias": c_in_bias.contiguous(), "C_out": c_out.contiguous(),
               "C_out_bias": c_out_bias.contiguous()}, str(path))
    doc = {"trunk_layer": k, "trunk_taps": {"rows_key": rows_key},
           "files": {arm: {"path": str(path), "file_sha256": c10.sha256_file(str(path))}},
           "arms": {arm: {"convention": c10.C17_CONVENTION, "chart_dtype": "float32"}}}
    (layer_dir / "layer.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def make_charts(root: Path, seed: int = 11, bias_scale: float = 0.0) -> None:
    """c17-shaped charts dir: random, NON-orthonormal, independent C_in / C_out (+ optional biases)."""
    g = torch.Generator().manual_seed(seed)
    for k in range(LAYERS):
        c_in = torch.randn(K3_HIDDEN, HIDDEN, generator=g) / HIDDEN ** 0.5        # [d_k3 x d_trunk]
        c_out = torch.randn(HIDDEN, K3_HIDDEN, generator=g) / K3_HIDDEN ** 0.5    # [d_trunk x d_k3]
        b_in = bias_scale * torch.randn(HIDDEN, generator=g)
        b_out = bias_scale * torch.randn(K3_HIDDEN, generator=g)
        write_c17_chart(root / f"L{k:02d}", k, c_in, b_in, c_out, b_out)
    (root / "alignment.json").write_text(json.dumps(c17_alignment_doc(), indent=2) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def k3(work: Path) -> dict:
    root = work / "k3_dense"
    raw = make_k3_dense(root)
    charts = work / "charts"
    make_charts(charts)
    return {"dir": root, "raw": raw, "charts": charts}


def run(trunks: dict, k3_dir: Path, charts: Path, out: Path, k3_channels: int, extra: list[str] | None = None,
        expect: int = 0) -> dict:
    argv = ["--trunk", str(trunks["trunk"]), "--parent-trunk", str(trunks["parent"]), "--k3-dense", str(k3_dir),
            "--charts", str(charts), "--evidence", str(trunks["evidence"]), "--k3-channels", str(k3_channels),
            "--device", "cpu", "--group-size", "2", "--chunk", "64", "--gram-block", "16", "--out", str(out),
            "--overwrite", "--k3-config-from-flags"] + (extra or [])
    code = c10.main(argv)
    assert code == expect, (code, expect)
    if expect != 0:
        return {}
    return json.loads((out / "union_receipt.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def installed(work: Path, trunks: dict, k3: dict) -> dict:
    out = work / "child_k10"
    receipt = run(trunks, k3["dir"], k3["charts"], out, 10, ["--dump-captures"])
    return {"out": out, "receipt": receipt}


def child_tensor_shas(root: Path) -> dict[str, str]:
    return {name: view.sha256() for name, view in read_safetensors(root, verify=True).items()}


def child_file_bytes(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in c10.checkpoint_files(root)}


# ------------------------------------------------------------------ 1
def test_k3_channels_zero_equals_trunk_byte_for_byte(work: Path, trunks: dict, k3: dict) -> None:
    out = work / "child_k0"
    receipt = run(trunks, k3["dir"], k3["charts"], out, 0)
    trunk_files = {p.name: c10.sha256_file(str(p)) for p in c10.checkpoint_files(trunks["trunk"])}
    child_files = {p.name: c10.sha256_file(str(p)) for p in c10.checkpoint_files(out)}
    assert child_files == trunk_files
    for name in trunk_files:
        assert (out / name).read_bytes() == (trunks["trunk"] / name).read_bytes()
    assert receipt["byte_identical_to_trunk"] is True
    assert receipt["mode"].startswith("verbatim copy")
    assert receipt["veto_binding_layers"] == 0


# ------------------------------------------------------------------ 2
def test_k3_dense_zeros_binds_the_veto_on_every_layer(work: Path, trunks: dict, k3: dict) -> None:
    root = work / "k3_zeros"
    make_k3_dense(root, zeros=True)
    out = work / "child_zeros"
    receipt = run(trunks, root, k3["charts"], out, 10)
    assert len(receipt["layers"]) == LAYERS
    for layer in receipt["layers"]:
        assert layer["veto_binding"] is True, layer["layer"]
        assert layer["k3_rows_function_absent"] is True
        assert "no K3 function" in layer["veto_reason"]
        assert layer["k3_term_energy_fraction"]["held"] == 0.0
        assert layer["selection"]["rank_starved"] is True
        assert layer["installed"]["intermediate_size"] == TRUNK_WIDTH + 10
    assert receipt["veto_binding_layers"] == LAYERS
    assert receipt["veto_binding_most_layers"] is True
    assert "delete this child" in receipt["frontier_rule"]
    # the child is still a valid, uniform checkpoint: the zero rows are counted and present
    counts = c10.oc_seal.safetensors_tensor_counts(out)
    assert counts["blocks.0.ffn.up.weight"] == 2 * (TRUNK_WIDTH + 10) * HIDDEN


# ------------------------------------------------------------------ 3
def test_qwen_rows_are_always_kept(installed: dict, trunks: dict) -> None:
    receipt = installed["receipt"]
    child = read_safetensors(installed["out"], verify=True)
    trunk = read_safetensors(trunks["trunk"], verify=True)
    width = TRUNK_WIDTH + 10
    for k in range(LAYERS):
        layer = receipt["layers"][k]
        assert layer["selection"]["qwen_rows_all_kept"] is True
        assert layer["qwen_rows"] == TRUNK_WIDTH
        assert layer["installed"]["qwen_rows"] == TRUNK_WIDTH
        assert layer["installed"]["k3_rows"] == 10
        c_up = child[f"blocks.{k}.ffn.up.weight"].torch(device="cpu", dtype=torch.float32)
        t_up = trunk[f"blocks.{k}.ffn.up.weight"].torch(device="cpu", dtype=torch.float32)
        assert tuple(c_up.shape) == (2 * width, HIDDEN)
        # verbatim gate rows and verbatim up rows, in order, at index 0..Q-1 of each half
        assert torch.equal(c_up[:TRUNK_WIDTH], t_up[:TRUNK_WIDTH])
        assert torch.equal(c_up[width:width + TRUNK_WIDTH], t_up[TRUNK_WIDTH:])
        c_down = child[f"blocks.{k}.ffn.down.weight"].torch(device="cpu", dtype=torch.float32)
        assert tuple(c_down.shape) == (HIDDEN, width)
        assert bool(torch.isfinite(c_down).all())
        assert layer["update_scale"] <= 1.0
    cfg = json.loads((installed["out"] / "config.json").read_text(encoding="utf-8"))["config"]
    assert cfg["intermediate_size"] == width
    assert cfg["intermediate_sizes"] == [width] * LAYERS
    assert cfg["activation_kind"] == "silu"


# ------------------------------------------------------------------ 4
def test_situ_to_silu_scalar_reproduces_situ_within_recorded_relL2(installed: dict, k3: dict) -> None:
    receipt = installed["receipt"]
    k = 1
    layer_record = receipt["layers"][k]
    captures_dir = Path(receipt["captures_dir"])
    assert captures_dir == Path(f"{installed['out']}.captures")      # beside the child, never inside it
    assert not (installed["out"] / "captures").exists()
    captures = load_file(str(captures_dir / f"L{k:02d}.safetensors"))
    x_fit = captures["x_fit"]
    chart = c10.load_chart(str(k3["charts"]), k, K3_HIDDEN, HIDDEN, False)
    cfg = c10.K3Config()
    recorded = {row["source"]: row for row in layer_record["situ_to_silu"]}
    assert recorded, "no SiTU sources on this layer"
    for l in layer_record["k3_layers"]:
        layer = c10.load_k3_layer(str(k3["dir"]), l, cfg)
        rms_mean = next(s["y_lat_rms_mean_fit"] for s in layer_record["k3_sources"] if s["k3_layer"] == l)
        for source in c10.lift_layer(layer, chart, rms_mean):
            got = c10.situ_to_silu_scalars(source.gate, source.up, x_fit, source.beta, source.linear_beta,
                                           torch.device("cpu"), chunk=64)
            scale = got["scale"]
            pre_g = x_fit @ source.gate.T
            pre_u = x_fit @ source.up.T
            a = torch.nn.functional.silu(pre_g) * pre_u * scale
            b = c10.situ(pre_g, pre_u, source.beta, source.linear_beta)
            direct = float((a - b).norm() / b.norm())
            # the closed form IS the direct number ...
            assert abs(direct - got["relL2"]) < 1e-4 * max(1.0, direct), (source.name, direct, got["relL2"])
            # ... and the receipt recorded it
            assert abs(recorded[source.name]["relL2"] - direct) < 1e-4 * max(1.0, direct)
            # ... and the scalar is the least-squares optimum: no uniform rescale does better
            for factor in (0.9, 1.1):
                worse = float((a * factor - b).norm() / b.norm())
                assert worse >= direct - 1e-6
            assert got["dead_channels"] == 0


# ------------------------------------------------------------------ 5
def test_size_refusal_on_a_synthetic_oversize_child(work: Path, trunks: dict, k3: dict) -> None:
    out = work / "child_oversize"
    trunk_total, _ = c10.header_total(trunks["trunk"])
    projected = trunk_total + LAYERS * 3 * HIDDEN * 10
    # (a) refused from the arithmetic, before any capture, and nothing is written
    with pytest.raises(SystemExit) as refusal:
        run(trunks, k3["dir"], k3["charts"], out, 10, ["--parent-parameter-count", str(projected)])
    assert "REFUSED" in str(refusal.value) and f"{projected:,}" in str(refusal.value)
    assert not out.exists() or not any(out.iterdir())
    # (b) the post-write gate reads the headers with oc_seal and removes the child
    child = work / "child_for_gate"
    shutil.copytree(work / "child_k10", child, ignore=shutil.ignore_patterns("captures"))
    total, _ = c10.header_total(child)
    with pytest.raises(SystemExit) as refusal_b:
        c10.refuse_size(total, total, "after the write", child)
    assert "REFUSED" in str(refusal_b.value)
    assert not child.exists()
    # (c) a strictly larger parent passes
    c10.refuse_size(total, total + 1, "after the write")


# ------------------------------------------------------------------ 6
def test_form_e_lift_equals_form_m_lift_on_random_inputs(k3: dict) -> None:
    torch.manual_seed(3)
    cfg = c10.K3Config()
    layer = c10.load_k3_layer(str(k3["dir"]), 2, cfg)
    chart = c10.load_chart(str(k3["charts"]), 1, K3_HIDDEN, HIDDEN, False)
    form_m = c10.form_m_from_form_e(layer)
    gate_e, up_e = c10.lift_routed_form_e(layer, chart.trunk_to_k3)
    gate_m, up_m = c10.lift_routed_form_m(form_m, chart.trunk_to_k3)
    x = torch.randn(256, HIDDEN)
    for lhs, rhs in ((gate_e, gate_m), (up_e, up_m)):
        assert tuple(lhs.shape) == (K3_ROUTED, HIDDEN)
        assert torch.allclose(x @ lhs.T, x @ rhs.T, rtol=1e-5, atol=1e-5)
    # the whole routed channel output agrees, through the SiTU nonlinearity
    psi_e = c10.situ(x @ gate_e.T, x @ up_e.T, cfg.beta, cfg.linear_beta)
    psi_m = c10.situ(x @ gate_m.T, x @ up_m.T, cfg.beta, cfg.linear_beta)
    assert torch.allclose(psi_e, psi_m, rtol=1e-5, atol=1e-5)
    # and both agree with the form-E reference evaluated in K3 space (trunk -> K3 through T = C_out^T)
    xk = x @ chart.trunk_to_k3.T
    latent = xk @ layer.tensors["block_sparse_moe.routed_expert_down_proj.weight"].float().T
    psi_ref = c10.situ(latent @ layer.tensors["w1_s"].float().T, latent @ layer.tensors["w3_s"].float().T,
                       cfg.beta, cfg.linear_beta)
    assert torch.allclose(psi_e, psi_ref, rtol=1e-4, atol=1e-5)
    # c14's own build_form_m casts to the parent dtype; on fp32 fixtures that is the same map
    bf = {"w1_m": form_m["w1_m"].to(torch.bfloat16).float(), "w3_m": form_m["w3_m"].to(torch.bfloat16).float()}
    gate_b, up_b = c10.lift_routed_form_m(bf, chart.trunk_to_k3)
    assert float((x @ gate_b.T - x @ gate_e.T).norm() / (x @ gate_e.T).norm()) < 2e-2


# ------------------------------------------------------------------ section 3b
def test_exact_zero_routed_expert_norm_gain_is_counted_not_divided(work: Path, trunks: dict, k3: dict) -> None:
    root = work / "k3_zero_gain"
    make_k3_dense(root, zero_gain_layer=2)
    out = work / "child_zero_gain"
    receipt = run(trunks, root, k3["charts"], out, 6)
    hit = [s for layer in receipt["layers"] for s in layer["k3_sources"] if s["k3_layer"] == 2]
    assert len(hit) == 1
    assert hit[0]["zero_gains"]["zero_gain_count"] == 1
    assert hit[0]["zero_gains"]["zero_gain_coordinates_first"] == [3]
    for layer in receipt["layers"]:
        assert layer["heldout_relL2"] is not None and layer["heldout_relL2"] == layer["heldout_relL2"]  # finite
        assert layer["update_scale"] <= 1.0
    child = read_safetensors(out, verify=True)
    for name, view in child.items():
        if name.endswith("ffn.down.weight"):
            assert bool(torch.isfinite(view.torch(device="cpu", dtype=torch.float32)).all())


def test_non_finite_k3_tensor_is_a_named_refusal(work: Path, trunks: dict, k3: dict) -> None:
    root = work / "k3_nan"
    make_k3_dense(root, nan_layer=3)
    out = work / "child_nan"
    with pytest.raises(SystemExit) as refusal:
        run(trunks, root, k3["charts"], out, 6)
    assert "non-finite" in str(refusal.value) and "w1_s" in str(refusal.value)


def test_alignment_covers_every_k3_layer_exactly_once(installed: dict) -> None:
    table = installed["receipt"]["alignment"]["trunk_to_k3"]
    covered = sorted(l for ls in table.values() for l in ls)
    assert covered == list(range(K3_MOE_LAYERS + 1))
    assert 0 in table["0"]
    # 6 MoE layers over 4 trunk layers: two trunk layers get two, two get one (plus layer 0 on trunk 0)
    sizes = sorted(len(v) for v in table.values())
    assert sum(sizes) == K3_MOE_LAYERS + 1
    with pytest.raises(SystemExit):
        c10.load_alignment(None, None, [0, 1, 2], 0)


def test_child_loads_and_runs_at_the_uniform_width(installed: dict, trunks: dict) -> None:
    model, cfg, _ = load_chimera_checkpoint(installed["out"], device="cpu")
    assert cfg.resolved_intermediate_size(0) == TRUNK_WIDTH + 10
    ids = torch.randint(8, VOCAB, (1, SEQ))
    with torch.no_grad():
        out = model(ids)
    logits = out["logits"] if isinstance(out, dict) else out
    assert bool(torch.isfinite(logits).all())
    receipt = installed["receipt"]
    assert receipt["size_gate"]["passed"] is True
    assert receipt["child"]["header_parameters"] == receipt["size_gate"]["projected_child_header_parameters"]
    assert receipt["child"]["header_parameters"] < receipt["parent_parameter_count"]
    summary = [(l["layer"], l["heldout_relL2"], l["scores"]["held"]["trunk_only"]["relL2"],
                l["k3_term_energy_fraction"]["held"], l["update_scale"], l["veto_binding"], l["veto_reason"])
               for l in receipt["layers"]]
    for layer in receipt["layers"]:
        assert "k3_term_energy_fraction" in layer and 0 < layer["k3_term_energy_fraction"]["held"] < 1, summary
        assert set(layer["ridge"]) >= {"update_scale", "veto_binding", "lambda_relative", "ladder"}
        # the union FFN reproduces Y better than the trunk alone: the K3 rows do something
        assert layer["heldout_relL2"] < layer["scores"]["held"]["trunk_only"]["relL2"], summary
        assert layer["k3_rows_function_absent"] is False, summary
    # in this regime (a refit trunk + a perturbative K3 term) the veto must not bind in most layers
    assert receipt["veto_binding_most_layers"] is False, summary


# ------------------------------------------------------------------ B1: c17's chart layout and --chart-arm
def test_charts_dir_in_c17_layout_installs_and_records_the_arm(installed: dict, k3: dict, work: Path,
                                                               trunks: dict) -> None:
    receipt = installed["receipt"]
    charts = k3["charts"]
    assert receipt["charts"]["arm"] == "ridge"
    for k in range(LAYERS):
        rec = receipt["charts"]["layers"][str(k)]
        ridge = charts / f"L{k:02d}" / "ridge.safetensors"
        assert rec["path"] == str(ridge) and rec["arm"] == "ridge"
        assert rec["sha256"] == c10.sha256_file(str(ridge))
        assert rec["layer_json_sha256"] == c10.sha256_file(str(charts / f"L{k:02d}" / "layer.json"))
        assert rec["fit_trunk_rows_key"] == "normed"
        layer_rec = receipt["layers"][k]["chart"]
        assert layer_rec["sha256"] == rec["sha256"] and layer_rec["arm"] == "ridge"
    # the arm names the FILE c17 wrote; an arm that was not fitted is a named refusal
    with pytest.raises(SystemExit) as missing:
        c10.load_chart(str(charts), 0, K3_HIDDEN, HIDDEN, False, arm="procrustes")
    assert "procrustes.safetensors" in str(missing.value) and "c17 writes" in str(missing.value)
    # a control arm installs too (svd_chart.safetensors, C_out = C_in^T)
    control = work / "charts_svd"
    g = torch.Generator().manual_seed(21)
    for k in range(LAYERS):
        q, _ = torch.linalg.qr(torch.randn(K3_HIDDEN, HIDDEN, generator=g))   # orthonormal columns [d_k3 x d_trunk]
        write_c17_chart(control / f"L{k:02d}", k, q, torch.zeros(HIDDEN), q.T.contiguous(), torch.zeros(K3_HIDDEN),
                        arm="svd")
    (control / "alignment.json").write_text(json.dumps(c17_alignment_doc()), encoding="utf-8")
    rec = run(trunks, k3["dir"], control, work / "child_svd", 6, ["--chart-arm", "svd"])
    assert rec["charts"]["arm"] == "svd"
    assert all(l["chart"]["path"].endswith("svd_chart.safetensors") for l in rec["layers"])
    # layer.json that records another convention, or another file sha, is refused
    bad = work / "charts_badjson"
    shutil.copytree(charts, bad)
    doc = json.loads((bad / "L00" / "layer.json").read_text(encoding="utf-8"))
    doc["arms"]["ridge"]["convention"] = "y_hat = C_in @ x_k3"
    (bad / "L00" / "layer.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit) as conv:
        c10.load_chart(str(bad), 0, K3_HIDDEN, HIDDEN, False)
    assert "convention" in str(conv.value)
    doc["arms"]["ridge"]["convention"] = c10.C17_CONVENTION
    doc["files"]["ridge"]["file_sha256"] = "0" * 64
    (bad / "L00" / "layer.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit) as sha:
        c10.load_chart(str(bad), 0, K3_HIDDEN, HIDDEN, False)
    assert "sha256" in str(sha.value)


# ------------------------------------------------------------------ B2: c17's alignment.json
def test_c17_alignment_json_is_parsed_and_sealed(installed: dict, k3: dict, work: Path) -> None:
    receipt = installed["receipt"]
    path = k3["charts"] / "alignment.json"
    assert receipt["alignment"]["form"] == "c17 table"
    assert receipt["alignment"]["schema"] == c10.C17_ALIGNMENT_SCHEMA
    assert receipt["alignment"]["sha256"] == c10.sha256_file(str(path))
    assert receipt["charts"]["alignment_json_sha256"] == c10.sha256_file(str(path))
    assert receipt["alignment"]["summary"]["covered_exactly_once"] is True
    doc = c17_alignment_doc()
    table, record = c10.parse_alignment_document(doc, str(path))
    assert table == {row["trunk"]: row["k3_layers"] for row in doc["table"]}
    assert table == c10.default_alignment(list(range(K3_MOE_LAYERS + 1)), LAYERS)   # == c17.alignment_table
    # the seal: c17 saying covered_exactly_once = false is a refusal before any capture
    broken = dict(doc, summary=dict(doc["summary"], covered_exactly_once=False))
    broken_path = work / "alignment_broken.json"
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(SystemExit) as sealed:
        c10.parse_alignment_document(broken, str(broken_path))
    assert "covered_exactly_once" in str(sealed.value)
    # a table that silently drops a layer is caught by the coverage check
    dropped = dict(doc, table=[dict(row, k3_layers=[l for l in row["k3_layers"] if l != 4]) for row in doc["table"]],
                   summary={})
    dropped_path = work / "alignment_dropped.json"
    dropped_path.write_text(json.dumps(dropped), encoding="utf-8")
    with pytest.raises(SystemExit) as cover:
        c10.load_alignment(None, str(dropped_path), list(range(K3_MOE_LAYERS + 1)), LAYERS)
    assert "exactly once" in str(cover.value)
    # the override form still works, and an unknown document is refused rather than guessed
    override_path = work / "alignment_override.json"
    override_path.write_text(json.dumps({"trunk_to_k3": {str(k): v for k, v in table.items()}}), encoding="utf-8")
    got, rec = c10.load_alignment(None, str(override_path), list(range(K3_MOE_LAYERS + 1)), LAYERS)
    assert got == table and rec["form"] == "trunk_to_k3 override"
    unknown_path = work / "alignment_unknown.json"
    unknown_path.write_text(json.dumps({"schema": "x", "summary": {}}), encoding="utf-8")
    with pytest.raises(SystemExit) as unknown:
        c10.parse_alignment_document({"schema": "x", "summary": {}}, str(unknown_path))
    assert "table" in str(unknown.value)


# ------------------------------------------------------------------ B3: direction and biases
def test_chart_direction_and_biases_follow_c17_convention(k3: dict, work: Path, trunks: dict) -> None:
    torch.manual_seed(5)
    n = 4096
    x = torch.randn(n, HIDDEN)                                            # trunk rows
    a_map = torch.randn(HIDDEN, K3_HIDDEN) / HIDDEN ** 0.5
    a_bias = 0.3 * torch.randn(K3_HIDDEN)
    xk = x @ a_map + a_bias + 0.05 * torch.randn(n, K3_HIDDEN)             # K3 rows, noisy: bwd != pinv(fwd)

    def fit(src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        design = torch.cat([src, torch.ones(src.shape[0], 1)], dim=1).double()
        sol = torch.linalg.lstsq(design, dst.double()).solution
        return sol[:-1].float().contiguous(), sol[-1].float().contiguous()

    c_out, c_out_bias = fit(x, xk)      # trunk -> K3: x_hat = y_trunk @ C_out + C_out_bias   [d_trunk x d_k3]
    c_in, c_in_bias = fit(xk, x)        # K3 -> trunk: y_hat = x_k3 @ C_in + C_in_bias        [d_k3 x d_trunk]
    assert tuple(c_in.shape) == (K3_HIDDEN, HIDDEN) and tuple(c_out.shape) == (HIDDEN, K3_HIDDEN)
    assert not torch.allclose(c_out, c_in.T, atol=1e-3)                    # independent, non-orthonormal
    charts = work / "charts_fitted"
    write_c17_chart(charts / "L00", 0, c_in, c_in_bias, c_out, c_out_bias)
    chart = c10.load_chart(str(charts), 0, K3_HIDDEN, HIDDEN, False)
    assert torch.allclose(chart.trunk_to_k3, c_out.T) and torch.allclose(chart.k3_to_trunk, c_in.T)
    assert torch.allclose(chart.bias_to_k3, c_out_bias) and torch.allclose(chart.bias_to_trunk, c_in_bias)
    assert torch.allclose(chart.to_k3(x), x @ c_out + c_out_bias, atol=1e-5)
    assert torch.allclose(chart.to_k3(x), xk, atol=0.5)                    # it IS the fitted trunk -> K3 map
    y = torch.randn(n, K3_HIDDEN)
    assert torch.allclose(chart.to_trunk_linear(y), y @ c_in, atol=1e-5)
    # the K3 reference through the charts reproduces the teacher written in c17's raw tensors
    layer = c10.load_k3_layer(str(k3["dir"]), 2, c10.K3Config())
    dev = layer.device_tensors(torch.device("cpu"))
    teacher = layer.reference(x @ c_out + c_out_bias, dev)[0] @ c_in + c_in_bias
    tool = chart.to_trunk_linear(layer.reference(chart.to_k3(x), dev)[0]) + chart.bias_to_trunk
    assert torch.allclose(tool, teacher, rtol=1e-5, atol=1e-6)
    # ... and the transposed/swapped reading (x @ C_in^T, out @ C_out^T) is NOT the teacher
    wrong = layer.reference(x @ c_in.T, dev)[0] @ c_out.T
    assert float((wrong - (teacher - c_in_bias)).norm() / (teacher - c_in_bias).norm()) > 0.1
    # the lifted rows are the K3-space pre-activations through T (bias dropped, recorded as the shift)
    w_down = layer.tensors["block_sparse_moe.routed_expert_down_proj.weight"].float()
    w1 = layer.tensors["w1_s"].float()
    gate_e, up_e = c10.lift_routed_form_e(layer, chart.trunk_to_k3)
    assert torch.allclose(x @ gate_e.T, ((x @ c_out) @ w_down.T) @ w1.T, rtol=1e-4, atol=1e-5)
    routed, shared = c10.lift_layer(layer, chart, 1.0)
    assert torch.allclose(routed.gate_shift, (c_out_bias @ w_down.T) @ w1.T, atol=1e-5)
    assert torch.allclose(shared.up_shift, layer.tensors["block_sparse_moe.shared_experts.up_proj.weight"].float()
                          @ c_out_bias, atol=1e-5)
    assert torch.allclose(routed.prior, (c_in.T @ layer.tensors["block_sparse_moe.routed_expert_up_proj.weight"].float()
                                         @ (layer.tensors["block_sparse_moe.routed_expert_norm.weight"].float()[:, None]
                                            * layer.tensors["w2p"].float())), atol=1e-5)
    # an install through biased charts records both biases per layer and refuses above the ceiling
    biased = work / "charts_biased"
    make_charts(biased, seed=13, bias_scale=0.02)
    rec = run(trunks, k3["dir"], biased, work / "child_biased", 6, ["--chart-bias-ceiling", "0.5"])
    for l in rec["layers"]:
        bias = l["chart_bias"]
        assert bias["ceiling"] == 0.5
        assert bias["input"]["worst_shift_relL2"] > 0 and bias["output"]["energy_fraction_fit"] > 0
        assert all(row["gate_shift_relL2"] >= 0 and row["up_shift_relL2"] >= 0 for row in bias["input"]["per_source"])
        assert bias["input"]["worst_shift_relL2"] <= 0.5 and bias["output"]["energy_fraction_fit"] <= 0.5
        assert "DROPPED" in bias["input"]["treatment"] and "UNREPRESENTABLE" in bias["output"]["treatment"]
        assert l["chart"]["input_bias_norm"] > 0 and l["chart"]["output_bias_norm"] > 0
    assert rec["chart_bias_worst_input_shift_relL2"] > 0
    loud = work / "charts_loud"
    make_charts(loud, seed=17, bias_scale=5.0)
    with pytest.raises(SystemExit) as ceiling:
        run(trunks, k3["dir"], loud, work / "child_loud", 6)
    assert "chart biases are not negligible" in str(ceiling.value)
    rec2 = run(trunks, k3["dir"], loud, work / "child_loud", 6, ["--chart-bias-ceiling", "0"])
    assert rec2["layers"][0]["chart_bias"]["input"]["worst_shift_relL2"] > 0.1


# ------------------------------------------------------------------ B4: streamed == resident
def test_streamed_engine_equals_resident_byte_for_byte(installed: dict, trunks: dict, k3: dict, work: Path) -> None:
    assert installed["receipt"]["engine"] == "streamed"
    assert installed["receipt"]["trunk"]["load"]["engine"] == "streamed"
    out = work / "child_resident"
    receipt = run(trunks, k3["dir"], k3["charts"], out, 10, ["--engine", "resident"])
    assert receipt["trunk"]["load"]["engine"] == "resident"
    assert child_tensor_shas(out) == child_tensor_shas(installed["out"])
    assert child_file_bytes(out) == child_file_bytes(installed["out"])
    for a, b in zip(receipt["layers"], installed["receipt"]["layers"]):
        assert a["heldout_relL2"] == b["heldout_relL2"] and a["installed"] == b["installed"]
    # the streamed engine refuses a trunk it cannot reproduce block by block
    other = dataclasses.replace(chimera_config(TRUNK_WIDTH), residual_mode="block_attnres")
    views = read_safetensors(trunks["trunk"], verify=False)
    evidence = c10.load_evidence(str(trunks["evidence"]))
    with pytest.raises(SystemExit) as refusal:
        c10.StreamedEngine(trunks["trunk"], other, views, torch.device("cpu"), None, evidence)
    assert "residual_mode" in str(refusal.value)


# ------------------------------------------------------------------ M5: lazy K3 loading, one Gram
def test_k3_layers_load_lazily_and_the_check_gram_is_kept_columns_only(trunks: dict, k3: dict, work: Path,
                                                                        monkeypatch) -> None:
    live: list[int] = []
    high_water = [0]
    loads: list[int] = []
    real_load = c10.load_k3_layer
    real_release = c10.K3Layer.release

    def counting_load(root, index, cfg):
        layer = real_load(root, index, cfg)
        loads.append(index)
        live.append(index)
        high_water[0] = max(high_water[0], len(live))
        return layer

    def counting_release(self):
        live.remove(self.index)
        real_release(self)

    monkeypatch.setattr(c10, "load_k3_layer", counting_load)
    monkeypatch.setattr(c10.K3Layer, "release", counting_release)
    receipt = run(trunks, k3["dir"], k3["charts"], work / "child_lazy", 10)
    table = receipt["alignment"]["trunk_to_k3"]
    assert sorted(loads) == sorted(l for ls in table.values() for l in ls)     # each K3 layer loaded exactly once
    assert high_water[0] == max(len(v) for v in table.values()) == 2               # never the whole K3
    assert live == []                                                             # every layer released
    assert receipt["k3"]["loading"].startswith("lazy")
    for layer in receipt["layers"]:
        assert layer["gram"]["width"] == TRUNK_WIDTH + layer["pool"]["k3_candidates"]
        assert layer["gram"]["check_gram_width"] == TRUNK_WIDTH + layer["installed"]["k3_rows"]
        assert layer["gram"]["check_rows"] + layer["gram"]["inner_rows"] == layer["rows"]["fit"]
        assert "peak_rss_gigabytes" in layer and "peak_rss_gigabytes_after_selection" in layer
    assert set(receipt["peak_rss_gigabytes_per_layer"]) == {str(k) for k in range(LAYERS)}


# ------------------------------------------------------------------ M6: persistence + resume
def test_stop_and_resume_reproduce_the_single_pass_child(installed: dict, trunks: dict, k3: dict, work: Path) -> None:
    out = work / "child_resumed"
    partial = Path(f"{out}.partial")
    # stop inside the second group (layers 2,3) so the resume must re-capture that group
    run(trunks, k3["dir"], k3["charts"], out, 10, ["--stop-after-layer", "2"], expect=c10.EXIT_STOPPED)
    assert (partial / c10.PARTIAL_INPUTS).is_file()
    assert sorted(p.name for p in partial.glob("L*.safetensors")) == ["L00.safetensors", "L01.safetensors", "L02.safetensors"]
    assert not (out / "union_receipt.json").exists() and (out / c10.OUT_MARKER).is_file()
    saved = load_file(str(partial / "L01.safetensors"))
    assert tuple(saved["down"].shape) == (HIDDEN, TRUNK_WIDTH + 10)
    # a fresh run over a partial without --resume is refused (nothing is silently discarded)
    argv = ["--trunk", str(trunks["trunk"]), "--parent-trunk", str(trunks["parent"]), "--k3-dense", str(k3["dir"]),
            "--charts", str(k3["charts"]), "--evidence", str(trunks["evidence"]), "--k3-channels", "10",
            "--device", "cpu", "--group-size", "2", "--chunk", "64", "--gram-block", "16", "--out", str(out),
            "--k3-config-from-flags"]
    with pytest.raises(SystemExit) as stale:
        c10.main(argv)
    assert "partial run" in str(stale.value)
    # a resume with different inputs is refused, naming what differs
    with pytest.raises(SystemExit) as differ:
        c10.main([a if a != "10" else "8" for a in argv] + ["--resume"])
    assert "k3_channels" in str(differ.value)
    # the resume reloads 0..2 and solves 3; the child is the single-pass child, byte for byte
    code = c10.main(argv + ["--resume"])
    assert code == 0
    receipt = json.loads((out / "union_receipt.json").read_text(encoding="utf-8"))
    assert receipt["partial"]["resumed"] is True and receipt["partial"]["layers_reloaded"] == [0, 1, 2]
    assert [l.get("resumed_from_partial", False) for l in receipt["layers"]] == [True, True, True, False]
    assert receipt["partial"]["removed_after_write"] is True and not partial.exists()
    assert child_tensor_shas(out) == child_tensor_shas(installed["out"])
    assert child_file_bytes(out) == child_file_bytes(installed["out"])
    for a, b in zip(receipt["layers"], installed["receipt"]["layers"]):
        assert a["installed"]["up_sha256"] == b["installed"]["up_sha256"]
        assert a["installed"]["down_sha256"] == b["installed"]["down_sha256"]
        assert a["heldout_relL2"] == b["heldout_relL2"]


# ------------------------------------------------------------------ m7: scores on the shipped dtype
def test_receipt_scores_are_measured_on_the_shipped_dtype(installed: dict, trunks: dict, k3: dict, work: Path) -> None:
    def recompute(out: Path, receipt: dict, k: int) -> float:
        child = read_safetensors(out, verify=True)
        caps = load_file(str(Path(receipt["captures_dir"]) / f"L{k:02d}.safetensors"))
        up = child[f"blocks.{k}.ffn.up.weight"].torch(device="cpu", dtype=torch.float32)
        down = child[f"blocks.{k}.ffn.down.weight"].torch(device="cpu", dtype=torch.float32)
        w = up.shape[0] // 2
        x, y = caps["x_held"], caps["y_held"]
        pred = (torch.nn.functional.silu(x @ up[:w].T) * (x @ up[w:].T)) @ down.T
        return c10.write_scores(pred, y)["relL2"]

    # fp32 fixture: the receipt number is the on-disk artifact's number
    for k in range(LAYERS):
        rec = installed["receipt"]["layers"][k]
        assert rec["scores_dtype"] == "float32" and rec["cast"]["down_cast_relL2"] == 0.0
        assert abs(recompute(installed["out"], installed["receipt"], k) - rec["heldout_relL2"]) < 1e-6
    # bf16 model: the scored weights are the CAST weights, and the cast delta is recorded
    out = work / "child_bf16"
    receipt = run(trunks, k3["dir"], k3["charts"], out, 10, ["--dtype", "bfloat16", "--dump-captures"])
    child = read_safetensors(out, verify=True)
    assert child["blocks.0.ffn.down.weight"].dtype == "BF16"
    for k in range(LAYERS):
        rec = receipt["layers"][k]
        assert rec["scores_dtype"] == "bfloat16" and rec["cast"]["ffn_dtype"] == "bfloat16"
        assert rec["cast"]["down_cast_relL2"] > 0 and rec["cast"]["up_cast_relL2"] > 0
        assert abs(recompute(out, receipt, k) - rec["heldout_relL2"]) < 1e-6
        assert rec["update_scale"] != rec["update_scale_fp32_solution"]
        assert rec["update_scale_fp32_solution"] == rec["ridge"]["update_scale"]
        assert isinstance(rec["update_scale_cast_exceeds_law"], bool)


# ------------------------------------------------------------------ m8: one device copy per install
def test_k3_tensors_move_to_device_once_per_install(trunks: dict, k3: dict, work: Path, monkeypatch) -> None:
    calls: list[int] = []
    real = c10.K3Layer.device_tensors

    def counting(self, device):
        calls.append(self.index)
        return real(self, device)

    monkeypatch.setattr(c10.K3Layer, "device_tensors", counting)
    receipt = run(trunks, k3["dir"], k3["charts"], work / "child_once", 10)
    pairs = [l for layer in receipt["layers"] for l in layer["k3_layers"]]
    assert sorted(calls) == sorted(pairs)                 # once per (trunk layer, K3 layer), never per chunk
    assert len(calls) == K3_MOE_LAYERS + 1
    rows = receipt["layers"][0]["rows"]["fit"] + receipt["layers"][0]["rows"]["held"]
    assert rows // 64 > len(calls)                        # there were many more chunks than device copies


# ------------------------------------------------------------------ m9: --k3-config
def test_k3_config_is_required_and_recorded(trunks: dict, k3: dict, work: Path) -> None:
    base = ["--trunk", str(trunks["trunk"]), "--parent-trunk", str(trunks["parent"]), "--k3-dense", str(k3["dir"]),
            "--charts", str(k3["charts"]), "--evidence", str(trunks["evidence"]), "--k3-channels", "6",
            "--device", "cpu", "--group-size", "2", "--chunk", "64", "--gram-block", "16", "--overwrite"]
    with pytest.raises(SystemExit) as required:
        c10.main(base + ["--out", str(work / "child_nocfg")])
    assert "--k3-config" in str(required.value)
    config = work / "k3_config.json"
    config.write_text(json.dumps({"text_config": {"rms_norm_eps": 1e-5, "activation_situ_beta": 4.0,
                                                  "activation_situ_linear_beta": 25.0, "hidden_size": K3_HIDDEN}}),
                      encoding="utf-8")
    assert c10.main(base + ["--out", str(work / "child_cfg"), "--k3-config", str(config)]) == 0
    receipt = json.loads((work / "child_cfg" / "union_receipt.json").read_text(encoding="utf-8"))
    assert receipt["k3"]["config"]["source"] == str(config.absolute())
    assert receipt["k3"]["config"]["sha256"] == c10.sha256_file(str(config))
    assert receipt["k3"]["config"]["rms_norm_eps"] == 1e-5 and receipt["k3"]["config"]["activation_situ_beta"] == 4.0
    assert receipt["k3"]["config"]["activation_situ_linear_beta"] == 25.0


# ------------------------------------------------------------------ m10: --overwrite guard
def test_overwrite_refuses_foreign_dirs_and_input_overlap(trunks: dict, k3: dict, work: Path) -> None:
    def argv(out: Path, extra: list[str] | None = None) -> list[str]:
        return ["--trunk", str(trunks["trunk"]), "--parent-trunk", str(trunks["parent"]), "--k3-dense", str(k3["dir"]),
                "--charts", str(k3["charts"]), "--evidence", str(trunks["evidence"]), "--k3-channels", "6",
                "--device", "cpu", "--out", str(out), "--overwrite", "--k3-config-from-flags"] + (extra or [])

    # (a) --out overlapping the charts dir, the evidence file's dir, or the receipt path is refused, nothing removed
    before = sorted(p.name for p in k3["charts"].rglob("*"))
    with pytest.raises(SystemExit) as charts:
        c10.main(argv(k3["charts"]))
    assert "overlaps --charts" in str(charts.value)
    assert sorted(p.name for p in k3["charts"].rglob("*")) == before
    with pytest.raises(SystemExit) as evidence:
        c10.main(argv(trunks["evidence"].parent))
    assert "overlaps" in str(evidence.value)
    assert trunks["evidence"].is_file()
    out = work / "child_receipt_inside"
    with pytest.raises(SystemExit) as receipt:
        c10.main(argv(out, ["--receipt", str(out / "sub" / "union_receipt.json")]))
    assert "overlaps --receipt" in str(receipt.value)
    # (b) a non-empty directory this tool did not make is never rmtree'd
    foreign = work / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("not yours", encoding="utf-8")
    with pytest.raises(SystemExit) as refuse:
        c10.main(argv(foreign))
    assert "refuses to remove" in str(refuse.value)
    assert (foreign / "keep.txt").read_text(encoding="utf-8") == "not yours"
    # (c) a directory carrying this tool's receipt is rebuilt in place
    ours = work / "child_ours"
    assert c10.main(argv(ours)) == 0
    assert (ours / "union_receipt.json").is_file()
    assert c10.main(argv(ours)) == 0
