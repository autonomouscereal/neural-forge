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

plus 2026-10-10 (M4 rehearsal):
  1.1.1 test_streamed_engine_keeps_fp32_rope_tables_under_bf16
  1.2.0 test_device_pivot_matches_c14_pivot, test_selection_records_the_pivot_and_phase_seconds
  1.3.0 test_solve_device_cpu_is_recorded, test_cuda_selection_and_solve_match_the_host_child (CUDA only)
  1.4.0 test_chart_bias_ceiling_defaults_to_record_only (owner policy 2026-10-06, S7 decision D5: the
        default ceiling is 0 = record only; an explicit ceiling > 0 still refuses, asserted in B3)

plus 2026-10-11 (1.5.0, the D4 target-design experiment):
  test_parse_union_target_specs_and_refusals
  test_union_target_legacy_spec_is_bit_identical          (--union-target parent/additive:1 == no flag)
  test_union_target_trunk_additive_zero_reproduces_the_trunk
  test_union_target_scales_the_k3_term_and_follows_the_rule (additive w, rms w from the held norms)
  test_multi_solve_equals_c14_per_target                    (law_abiding_solve_gram_multi vs c14, per target)
  test_multi_solve_records_and_skips_rungs_that_do_not_factorise
  test_sweep_matches_single_target_installs_on_the_first_group
  test_sweep_layer_ranges_compose_and_resume
  test_sweep_refusals_and_disk_floor
  test_sweep_heavy_lock_is_taken_released_and_recorded
  test_cuda_sweep_matches_the_host_sweep                    (CUDA only)

plus 2026-10-11 (1.5.1, the S5 install; D4 verdict blocker 1):
  test_single_target_refit_falls_back_to_the_rung_skipping_solver
  test_single_target_refit_still_refuses_when_no_rung_is_admitted
  test_install_min_free_gib_stops_before_a_layer_and_before_the_child_write
  test_last_group_is_not_advanced

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
import math
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
def chimera_config(width: int, rope_dim: int = 0) -> ChimeraConfig:
    cfg = ChimeraConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=width, num_layers=LAYERS, num_heads=4,
        num_kv_heads=4, max_sequence_length=128, block_pattern=("kda", "gated_gqa", "kda", "gated_gqa"),
        residual_mode="none", attention_head_dim=8, kda_head_dim=8, kda_value_head_dim=8, kda_decay_rank=8,
        native_feature_dim=32, gqa_rope_dim=rope_dim, gqa_rope_theta=1.0e4,
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
    return build_trunks(work)


def build_trunks(work: Path, rope_dim: int = 0) -> dict:
    torch.manual_seed(1)
    work.mkdir(parents=True, exist_ok=True)
    parent_cfg = chimera_config(PARENT_WIDTH, rope_dim)
    parent = ChimeraOmniModel(parent_cfg).float()
    parent_dir = work / "parent"
    write_model(parent, parent_cfg, parent_dir)
    trunk_cfg = chimera_config(TRUNK_WIDTH, rope_dim)
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
    assert rec["chart_bias_policy"]["ceiling_enforced"] is True and rec["chart_bias_policy"]["ceiling"] == 0.5
    loud = work / "charts_loud"
    make_charts(loud, seed=17, bias_scale=5.0)
    # 1.4.0: an EXPLICIT ceiling > 0 is the opt-in strict mode and still refuses
    with pytest.raises(SystemExit) as ceiling:
        run(trunks, k3["dir"], loud, work / "child_loud", 6, ["--chart-bias-ceiling", "0.1"])
    assert "chart biases are not negligible" in str(ceiling.value)
    rec2 = run(trunks, k3["dir"], loud, work / "child_loud", 6, ["--chart-bias-ceiling", "0"])
    assert rec2["layers"][0]["chart_bias"]["input"]["worst_shift_relL2"] > 0.1


def test_chart_bias_ceiling_defaults_to_record_only(k3: dict, work: Path, trunks: dict) -> None:
    """1.4.0 (owner policy 2026-10-06, S7 decision D5): no flag = the bias energy is recorded, never a refusal,
    and the child equals the one an explicit ``--chart-bias-ceiling 0`` writes."""

    loud = work / "charts_loud_default"
    make_charts(loud, seed=17, bias_scale=5.0)
    rec = run(trunks, k3["dir"], loud, work / "child_loud_default", 6)          # no --chart-bias-ceiling
    assert rec["tool_version"] == c10.TOOL_VERSION == "1.5.1"
    policy = rec["chart_bias_policy"]
    assert policy["ceiling"] == 0.0 and policy["ceiling_enforced"] is False and policy["reference"] == 0.1
    assert policy["layers_above_reference"], "bias_scale 5 must put layers above the old 0.1 reference"
    assert rec["charts"]["bias_ceiling"] == 0.0
    for layer in rec["layers"]:
        bias = layer["chart_bias"]
        assert bias["ceiling"] == 0.0 and bias["ceiling_enforced"] is False and bias["policy"].startswith("diagnostic")
        above = bias["input"]["worst_shift_relL2"] > 0.1 or bias["output"]["energy_fraction_fit"] > 0.1
        assert bias["exceeds_reference_0_1"] is above
        assert (int(layer["layer"]) in policy["layers_above_reference"]) is above
    assert rec["chart_bias_worst_input_shift_relL2"] > 0.1
    explicit = run(trunks, k3["dir"], loud, work / "child_loud_explicit0", 6, ["--chart-bias-ceiling", "0"])
    assert child_tensor_shas(work / "child_loud_default") == child_tensor_shas(work / "child_loud_explicit0")
    assert explicit["chart_bias_policy"]["ceiling_enforced"] is False


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


# ------------------------------------------------------------------ 1.1.1: fp32 RoPE in the streamed engine
def test_streamed_engine_keeps_fp32_rope_tables_under_bf16(work: Path, k3: dict) -> None:
    """A bf16 streamed block must carry the constructor's fp32 ``inv_freq`` (what the loader and the
    C17 taps use), and the bf16 streamed child must equal the bf16 resident child byte for byte."""
    trunks = build_trunks(work / "rope_trunks", rope_dim=8)     # the shared fixture has gqa_rope_dim 0
    cfg = chimera_config(TRUNK_WIDTH, 8)
    views = read_safetensors(trunks["trunk"], verify=False)
    evidence = c10.load_evidence(str(trunks["evidence"]))
    engine = c10.StreamedEngine(trunks["trunk"], cfg, views, torch.device("cpu"), torch.bfloat16, evidence)
    gqa = [k for k in range(LAYERS) if cfg.block_kind(k) == "gated_gqa"]
    assert gqa, "the fixture must carry a RoPE layer"
    for k in gqa:
        block = engine.block(k, use_installed=False)
        reference = c10.ChimeraBlock(cfg, k).mixer.inv_freq
        assert reference is not None and reference.dtype == torch.float32
        assert block.mixer.inv_freq.dtype == torch.float32
        assert torch.equal(block.mixer.inv_freq, reference)
        assert not torch.equal(reference.to(torch.bfloat16).float(), reference)   # the table bf16 would round
        assert block.mixer.q_proj.weight.dtype == torch.bfloat16
    assert engine.record()["rope_tables"].startswith("fp32")
    engine.close()
    streamed = run(trunks, k3["dir"], k3["charts"], work / "child_bf16_streamed", 10, ["--dtype", "bfloat16"])
    resident = run(trunks, k3["dir"], k3["charts"], work / "child_bf16_resident", 10,
                   ["--dtype", "bfloat16", "--engine", "resident"])
    assert child_tensor_shas(work / "child_bf16_streamed") == child_tensor_shas(work / "child_bf16_resident")
    for a, b in zip(streamed["layers"], resident["layers"]):
        assert a["installed"]["up_sha256"] == b["installed"]["up_sha256"]
        assert a["installed"]["down_sha256"] == b["installed"]["down_sha256"]


# ------------------------------------------------------------------ 1.2.0: device-backed pivot selection
def test_device_pivot_matches_c14_pivot() -> None:
    """pivoted_columns_gram_on (the --selection-device path, run here on cpu) picks c14's columns."""
    g = torch.Generator().manual_seed(5)
    for n, k in ((300, 120), (257, 256), (64, 64), (40, 100)):
        a = torch.randn(4 * n, n, generator=g, dtype=torch.float64)
        gram = a.T @ a
        assert torch.equal(c10.c14.pivoted_columns_gram(gram.clone(), k),
                           c10.pivoted_columns_gram_on(gram.clone(), k, torch.device("cpu"))), (n, k)
    # exactly-dead columns stop both loops at the same step (residual 0 <= 1e-30)
    a = torch.randn(400, 100, generator=g, dtype=torch.float64)
    a[:, :10] = 0.0
    gram = a.T @ a
    ref = c10.c14.pivoted_columns_gram(gram.clone(), 95)
    got = c10.pivoted_columns_gram_on(gram.clone(), 95, torch.device("cpu"))
    assert torch.equal(ref, got) and int(got.numel()) == 90 and int(got.min()) >= 10


def test_selection_records_the_pivot_and_phase_seconds(installed: dict) -> None:
    for layer in installed["receipt"]["layers"]:
        assert layer["selection"]["pivot"] == "c14.pivoted_columns_gram (host)"
        assert layer["selection"]["pivot_seconds"] >= 0.0
        assert set(layer["phase_seconds"]) == {"targets_and_k3_reference", "chart_bias_and_situ_scalars", "pool_gram",
                                               "selection", "check_gram", "law_abiding_solve", "cast_and_scores"}
    assert installed["receipt"]["selection_device"] == "cpu"


# ------------------------------------------------------------------ 1.3.0: --solve-device
def test_solve_device_cpu_is_recorded(installed: dict) -> None:
    assert installed["receipt"]["solve_device"] == "cpu"
    for layer in installed["receipt"]["layers"]:
        assert layer["ridge"]["solve_device"] == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card (run with CUDA_VISIBLE_DEVICES=<free card>)")
def test_cuda_selection_and_solve_match_the_host_child(installed: dict, trunks: dict, k3: dict, work: Path) -> None:
    """--selection-device/--solve-device cuda: same kept channels, same lambda, weights within fp64 rounding."""
    out = work / "child_cuda_solve"
    receipt = run(trunks, k3["dir"], k3["charts"], out, 10,
                  ["--selection-device", "cuda:0", "--solve-device", "cuda:0"])
    host = installed["receipt"]["layers"]
    for a, b in zip(receipt["layers"], host):
        assert a["ridge"]["solve_device"] == "cuda:0"
        assert a["selection"]["pivot"].startswith("pivoted_columns_gram_on(cuda")
        assert a["selection"]["kept_per_source"] == b["selection"]["kept_per_source"]
        assert a["ridge"]["lambda_relative"] == b["ridge"]["lambda_relative"]
        assert abs(a["heldout_relL2"] - b["heldout_relL2"]) <= 1e-6 * max(1.0, abs(b["heldout_relL2"]))
    child = read_safetensors(out, verify=True)
    ref = read_safetensors(installed["out"], verify=True)
    for k in range(LAYERS):
        for name in ("up", "down"):
            got = child[f"blocks.{k}.ffn.{name}.weight"].torch(device="cpu", dtype=torch.float32)
            want = ref[f"blocks.{k}.ffn.{name}.weight"].torch(device="cpu", dtype=torch.float32)
            assert torch.allclose(got, want, rtol=1e-5, atol=1e-6), (k, name, float((got - want).abs().max()))


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


# ------------------------------------------------------------------ 1.5.0: target variants + the design sweep
def test_parse_union_target_specs_and_refusals() -> None:
    t = c10.parse_union_target("a01=trunk/additive:0.1")
    assert (t.name, t.base, t.rule, t.value) == ("a01", "trunk", "additive", 0.1)
    assert t.spec == "trunk/additive:0.1" and not t.is_legacy
    legacy = c10.parse_union_target("additive:1")
    assert legacy.base == "parent" and legacy.is_legacy and legacy.name == "parent_additive_1"
    assert c10.parse_union_target("convex:0.25").coefficients(3.0, 1.0)[:2] == (0.75, 0.25)
    a, w, rec = c10.parse_union_target("trunk/rms:0.3").coefficients(4.0, 2.0)
    assert a == 1.0 and abs(w - 0.6) < 1e-12 and rec["rule"] == "rms"
    assert c10.parse_union_target("trunk/rms:0.3").coefficients(4.0, 0.0)[1] == 0.0      # zero term: never 1/0
    for bad in ("trunk/convex:1.5", "trunk/additive:-1", "other/additive:1", "trunk/scaled:1", "trunk/additive",
                "a b=trunk/additive:1", "trunk/additive:nan"):
        with pytest.raises(SystemExit):
            c10.parse_union_target(bad)
    with pytest.raises(SystemExit) as dup:
        c10.parse_union_targets("x=additive:1;x=trunk/additive:0")
    assert "unique" in str(dup.value)
    assert [t.name for t in c10.parse_union_targets("p=additive:1; q=trunk/convex:0.5;")] == ["p", "q"]


def test_union_target_legacy_spec_is_bit_identical(installed: dict, trunks: dict, k3: dict, work: Path) -> None:
    out = work / "child_legacy_spec"
    receipt = run(trunks, k3["dir"], k3["charts"], out, 10, ["--union-target", "parent/additive:1"])
    assert child_tensor_shas(out) == child_tensor_shas(installed["out"])
    for a, b in zip(receipt["layers"], installed["receipt"]["layers"]):
        assert a["heldout_relL2"] == b["heldout_relL2"]
        assert a["union_target"]["legacy"] is True and a["union_target"]["w"] == 1.0
        assert b["union_target"]["spec"] == "parent/additive:1"            # the no-flag path records it too
    assert receipt["union_target"]["spec"] == "parent/additive:1"


def test_union_target_trunk_additive_zero_reproduces_the_trunk(trunks: dict, k3: dict, work: Path) -> None:
    out = work / "child_trunk_a0"
    receipt = run(trunks, k3["dir"], k3["charts"], out, 10, ["--union-target", "trunk/additive:0"])
    child = read_safetensors(out, verify=True)
    trunk = read_safetensors(trunks["trunk"], verify=True)
    for k in range(LAYERS):
        layer = receipt["layers"][k]
        assert layer["union_target"]["a"] == 1.0 and layer["union_target"]["w"] == 0.0
        assert layer["k3_term_energy_fraction"]["held"] == 0.0
        assert layer["k3_rows_function_absent"] is True and layer["veto_binding"] is True     # rows in name only
        down = child[f"blocks.{k}.ffn.down.weight"].torch(device="cpu", dtype=torch.float32)
        t_down = trunk[f"blocks.{k}.ffn.down.weight"].torch(device="cpu", dtype=torch.float32)
        assert c10.relative_l2(down[:, :TRUNK_WIDTH], t_down) < 1e-4
        assert float(down[:, TRUNK_WIDTH:].norm()) < 1e-4 * float(t_down.norm())
        assert layer["scores"]["held"]["union"]["relL2"] < 1e-4


def test_union_target_scales_the_k3_term_and_follows_the_rule(installed: dict, trunks: dict, k3: dict,
                                                              work: Path) -> None:
    """Layer 0's rows are the trunk's own forward in every run, so the dumped targets compare directly."""
    legacy = load_file(str(Path(f"{installed['out']}.captures") / "L00.safetensors"))
    out = work / "child_trunk_a05"
    receipt = run(trunks, k3["dir"], k3["charts"], out, 10, ["--union-target", "trunk/additive:0.5", "--dump-captures"])
    got = load_file(str(Path(f"{out}.captures") / "L00.safetensors"))
    assert torch.equal(got["x_fit"], legacy["x_fit"])
    assert torch.equal(got["k3_term_fit"], 0.5 * legacy["k3_term_fit"])            # exact: a power of two
    trunk_views = read_safetensors(trunks["trunk"], verify=True)
    up = trunk_views["blocks.0.ffn.up.weight"].torch(device="cpu", dtype=torch.float32)
    down = trunk_views["blocks.0.ffn.down.weight"].torch(device="cpu", dtype=torch.float32)
    base = c10.trunk_mlp_output(got["x_fit"], up[:TRUNK_WIDTH], up[TRUNK_WIDTH:], down, torch.device("cpu"), 64)
    assert torch.allclose(got["y_fit"], base + got["k3_term_fit"], rtol=1e-6, atol=1e-6)
    rec = receipt["layers"][0]["union_target"]
    assert rec["a"] == 1.0 and rec["w"] == 0.5 and rec["base"] == "trunk"
    # rms: w = gamma * ||BASE_held|| / ||TERM_held||, recorded with the norms it used; the held K3 energy
    # fraction then sits near gamma / sqrt(1 + gamma^2) (exactly that when BASE and TERM are orthogonal)
    out = work / "child_trunk_rms"
    receipt = run(trunks, k3["dir"], k3["charts"], out, 10, ["--union-target", "trunk/rms:0.3"])
    for layer in receipt["layers"]:
        r = layer["union_target"]
        assert abs(r["w"] - 0.3 * r["base_held_norm"] / r["term_held_norm"]) < 1e-9 * max(r["w"], 1.0)
        assert 0.1 < layer["k3_term_energy_fraction"]["held"] < 0.6


def test_multi_solve_equals_c14_per_target() -> None:
    g = torch.Generator().manual_seed(5)
    n, h, rows = 24, 6, 400
    a_fit = torch.randn(rows, n, generator=g, dtype=torch.float64)
    check = torch.zeros(rows, dtype=torch.bool)
    check[::5] = True
    comps = [torch.randn(rows, h, generator=g, dtype=torch.float64) for _ in range(3)]
    priors = [torch.randn(h, n, generator=g, dtype=torch.float64) * 0.3,
              torch.randn(h, n, generator=g, dtype=torch.float64) * 0.3]
    prior_index = [0, 0, 1]
    coeffs = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.1], [0.0, 0.75, 0.25], [0.0, 1.0, 0.0]]

    def grams(mask):
        a = a_fit[mask]
        y_gram = torch.tensor([[float((comps[i][mask] * comps[j][mask]).sum()) for j in range(3)] for i in range(3)],
                              dtype=torch.float64)
        return a.T @ a, torch.cat([a.T @ c[mask] for c in comps], 1), y_gram
    g_all, b_all, _ = grams(torch.ones(rows, dtype=torch.bool))
    g_ck, b_ck, y_ck = grams(check)
    multi = c10.law_abiding_solve_gram_multi(g_all - g_ck, b_all - b_ck, g_ck, b_ck, y_ck, priors, prior_index,
                                             coeffs, h, None)
    for v, cv in enumerate(coeffs):
        y = sum(cv[i] * comps[i] for i in range(3))
        prior = sum(cv[i] * priors[prior_index[i]] for i in range(3))
        a_ck = a_fit[check]
        one, ladder = c10.c14.law_abiding_solve_gram((g_all - g_ck).clone(), (a_fit[~check].T @ y[~check]).clone(),
                                                     g_ck.clone(), (a_ck.T @ y[check]).clone(),
                                                     float((y[check] ** 2).sum()), prior)
        mine, rec = multi[v]
        assert mine is not None
        assert rec["lambda_relative"] == ladder["lambda_relative"]
        assert rec["veto_binding"] == ladder["veto_binding"]
        for r1, r2 in zip(rec["ladder"], ladder["ladder"]):
            assert abs(r1["check_sse"] - r2["check_sse"]) <= 1e-8 * max(1.0, abs(r2["check_sse"]))
            assert abs(r1["update_scale"] - r2["update_scale"]) <= 1e-9 * max(1.0, r2["update_scale"])
        assert c10.relative_l2(mine.double(), one.double()) < 1e-6


def run_sweep(trunks: dict, k3: dict, out: Path, targets: str, extra: list[str] | None = None, expect: int = 0) -> int:
    argv = ["--trunk", str(trunks["trunk"]), "--parent-trunk", str(trunks["parent"]), "--k3-dense", str(k3["dir"]),
            "--charts", str(k3["charts"]), "--evidence", str(trunks["evidence"]), "--k3-channels", "10",
            "--device", "cpu", "--group-size", "2", "--chunk", "64", "--gram-block", "16", "--out", str(out),
            "--k3-config-from-flags", "--sweep-targets", targets, "--open-loop", "--min-free-gib", "0"] + (extra or [])
    code = c10.main(argv)
    assert code == expect, (code, expect)
    return code


SWEEP_TARGETS = "legacy=parent/additive:1;a03=trunk/additive:0.3;a0=trunk/additive:0;b05=trunk/convex:0.5;g01=trunk/rms:0.1"


@pytest.fixture(scope="module")
def swept(work: Path, trunks: dict, k3: dict) -> dict:
    out = work / "sweep_full"
    run_sweep(trunks, k3, out, SWEEP_TARGETS)
    return {"out": out}


def test_sweep_matches_single_target_installs_on_the_first_group(swept: dict, installed: dict, trunks: dict,
                                                                  k3: dict, work: Path) -> None:
    """Group [0, 1] is captured on the trunk's own forward in BOTH schedules, so there the open-loop
    sweep and the closed-loop single-target install see the same rows: same selection (the shared up is
    byte-identical) and the same solve up to fp64 summation order."""
    out = swept["out"]
    single_a03 = work / "child_single_a03"
    run(trunks, k3["dir"], k3["charts"], single_a03, 10, ["--union-target", "trunk/additive:0.3"])
    for name, child in (("legacy", installed["out"]), ("a03", single_a03)):
        views = read_safetensors(child, verify=True)
        for k in (0, 1):
            up = load_file(str(out / "shared" / f"L{k:02d}.up.safetensors"))["up"]
            down = load_file(str(out / "variants" / name / f"L{k:02d}.safetensors"))["down"]
            assert torch.equal(up, views[f"blocks.{k}.ffn.up.weight"].torch(device="cpu", dtype=torch.float32))
            ref = views[f"blocks.{k}.ffn.down.weight"].torch(device="cpu", dtype=torch.float32)
            assert c10.relative_l2(down, ref) < 1e-4, (name, k, c10.relative_l2(down, ref))
            rec = json.loads((out / "variants" / name / f"L{k:02d}.json").read_text(encoding="utf-8"))
            assert rec["installed"]["k3_rows"] == 10 and rec["installed"]["intermediate_size"] == TRUNK_WIDTH + 10
            assert rec["partial"]["up_tensors_sha256"] == c10.sha256_file(rec["partial"]["up_tensors"])
            assert rec["partial"]["tensors_sha256"] == c10.sha256_file(rec["partial"]["tensors"])
    single = json.loads((single_a03 / "union_receipt.json").read_text(encoding="utf-8"))["layers"][0]
    rec = json.loads((out / "variants" / "a03" / "L00.json").read_text(encoding="utf-8"))
    assert rec["ridge"]["lambda_relative"] == single["ridge"]["lambda_relative"]
    assert rec["union_target"]["w"] == single["union_target"]["w"]
    assert abs(rec["k3_term_energy_fraction"]["held"] - single["k3_term_energy_fraction"]["held"]) < 1e-5
    assert abs(rec["heldout_relL2"] - single["heldout_relL2"]) < 1e-4
    # a0 is the trunk; every variant's partial_inputs names its own target and open_loop
    for k in range(LAYERS):
        rec0 = json.loads((out / "variants" / "a0" / f"L{k:02d}.json").read_text(encoding="utf-8"))
        assert rec0["k3_rows_function_absent"] is True and rec0["heldout_relL2"] < 1e-4
        assert 0.0 <= rec0["k3_columns_write_fraction_held"] < 1e-4
    for name in ("legacy", "a03", "a0", "b05", "g01"):
        ident = json.loads((out / "variants" / name / c10.PARTIAL_INPUTS).read_text(encoding="utf-8"))["identity"]
        assert ident["open_loop"] is True and ident["union_target_name"] == name
    shared = json.loads((out / "shared" / "L03.json").read_text(encoding="utf-8"))
    assert shared["components"] == ["parent", "trunk", "term"]
    assert set(shared["variants"]) == {"legacy", "a03", "a0", "b05", "g01"}


def test_sweep_layer_ranges_compose_and_resume(swept: dict, trunks: dict, k3: dict, work: Path) -> None:
    """Ranges run separately (the second process advances 0..1 on the trunk first) give the full sweep's
    weights: the shared up exactly (same selection) and every down to fp roundoff (threaded host BLAS
    reductions are not bit-reproducible across processes)."""
    split = work / "sweep_split"
    run_sweep(trunks, k3, split, SWEEP_TARGETS, ["--trunk-layers", "2-3"])
    run_sweep(trunks, k3, split, SWEEP_TARGETS, ["--trunk-layers", "0-1"])
    full = swept["out"]
    for k in range(LAYERS):
        a = load_file(str(split / "shared" / f"L{k:02d}.up.safetensors"))["up"]
        b = load_file(str(full / "shared" / f"L{k:02d}.up.safetensors"))["up"]
        assert torch.equal(a, b), k
        for name in ("legacy", "a03", "a0", "b05", "g01"):
            a = load_file(str(split / "variants" / name / f"L{k:02d}.safetensors"))["down"]
            b = load_file(str(full / "variants" / name / f"L{k:02d}.safetensors"))["down"]
            assert c10.relative_l2(a, b) < 1e-5, (name, k, c10.relative_l2(a, b))
    # completed layers are skipped (implicit resume): nothing is rewritten
    mtimes = {p: p.stat().st_mtime_ns for p in (split / "variants").rglob("L*.safetensors")}
    run_sweep(trunks, k3, split, SWEEP_TARGETS)
    assert {p: p.stat().st_mtime_ns for p in (split / "variants").rglob("L*.safetensors")} == mtimes
    # a different target set in the same directory is refused
    with pytest.raises(SystemExit) as other:
        run_sweep(trunks, k3, split, "legacy=parent/additive:0.5;a03=trunk/additive:0.3")
    assert "differ" in str(other.value)


def test_sweep_refusals_and_disk_floor(trunks: dict, k3: dict, work: Path) -> None:
    base_argv = ["--trunk", str(trunks["trunk"]), "--parent-trunk", str(trunks["parent"]), "--k3-dense",
                 str(k3["dir"]), "--charts", str(k3["charts"]), "--evidence", str(trunks["evidence"]),
                 "--k3-channels", "10", "--device", "cpu", "--k3-config-from-flags"]
    with pytest.raises(SystemExit) as closed:
        c10.main(base_argv + ["--out", str(work / "sw_closed"), "--sweep-targets", "a=trunk/additive:0"])
    assert "--open-loop" in str(closed.value)
    with pytest.raises(SystemExit) as both:
        c10.main(base_argv + ["--out", str(work / "sw_both"), "--sweep-targets", "a=trunk/additive:0", "--open-loop",
                              "--union-target", "trunk/additive:0.1"])
    assert "exclusive" in str(both.value)
    with pytest.raises(SystemExit) as stray:
        c10.main(base_argv + ["--out", str(work / "sw_stray"), "--open-loop"])
    assert "design harness" in str(stray.value)
    floor = work / "sweep_floor"
    run_sweep(trunks, k3, floor, "a=trunk/additive:0", ["--min-free-gib", "1e9"], expect=c10.EXIT_STOPPED)
    assert not list((floor / "variants").rglob("L*.safetensors"))
    assert not list((floor / "shared").glob("L*.json"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card (run with CUDA_VISIBLE_DEVICES=<free card>)")
def test_cuda_sweep_matches_the_host_sweep(swept: dict, trunks: dict, k3: dict, work: Path) -> None:
    out = work / "sweep_cuda"
    run_sweep(trunks, k3, out, SWEEP_TARGETS, ["--device", "cuda:0", "--selection-device", "cuda:0",
                                               "--solve-device", "cuda:0"])
    for k in range(LAYERS):
        rec = json.loads((out / "variants" / "a03" / f"L{k:02d}.json").read_text(encoding="utf-8"))
        assert rec["ridge"]["solve_device"].startswith("cuda")
        a = load_file(str(out / "shared" / f"L{k:02d}.up.safetensors"))["up"]
        b = load_file(str(swept["out"] / "shared" / f"L{k:02d}.up.safetensors"))["up"]
        assert c10.relative_l2(a, b) < 1e-5, k
        for name in ("legacy", "a03", "a0", "b05", "g01"):
            a = load_file(str(out / "variants" / name / f"L{k:02d}.safetensors"))["down"]
            b = load_file(str(swept["out"] / "variants" / name / f"L{k:02d}.safetensors"))["down"]
            assert c10.relative_l2(a, b) < 1e-3, (name, k, c10.relative_l2(a, b))


def test_sweep_heavy_lock_is_taken_released_and_recorded(swept: dict, trunks: dict, k3: dict, work: Path) -> None:
    """--heavy-lock: the pool-Gram / selection phase runs under an exclusive flock (released after the
    kept slice); the weights equal the unlocked sweep's (the lock changes scheduling, not arithmetic)."""
    import fcntl

    lock = work / "heavy.lock"
    out = work / "sweep_locked"
    run_sweep(trunks, k3, out, SWEEP_TARGETS, ["--trunk-layers", "0-1", "--heavy-lock", str(lock)])
    for k in (0, 1):
        shared = json.loads((out / "shared" / f"L{k:02d}.json").read_text(encoding="utf-8"))
        assert shared["heavy_lock"]["path"] == str(lock) and shared["heavy_lock"]["waited_seconds"] >= 0.0
        a = load_file(str(out / "shared" / f"L{k:02d}.up.safetensors"))["up"]
        b = load_file(str(swept["out"] / "shared" / f"L{k:02d}.up.safetensors"))["up"]
        assert torch.equal(a, b)
    with open(lock, "a+") as handle:          # released: a non-blocking exclusive lock succeeds
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_multi_solve_records_and_skips_rungs_that_do_not_factorise() -> None:
    """An fp64 Gram made slightly indefinite (what fp32 column-block products leave behind) fails the weakest
    rungs; the sweep solver records them, chooses among the rungs that factorise, and refuses only when none does."""
    g = torch.Generator().manual_seed(9)
    n, h, rows = 16, 4, 300
    a = torch.randn(rows, n, generator=g, dtype=torch.float64)
    check = torch.zeros(rows, dtype=torch.bool)
    check[::4] = True
    y = torch.randn(rows, h, generator=g, dtype=torch.float64)
    g_in, g_ck = a[~check].T @ a[~check], a[check].T @ a[check]
    g_in[0, 0] = -0.002 * float(torch.diagonal(g_in).mean())       # indefinite below rel ~2e-3
    g_ck[0, 0] = -0.002 * float(torch.diagonal(g_ck).mean())
    prior = torch.randn(h, n, generator=g, dtype=torch.float64) * 0.1
    y_ck = torch.tensor([[float((y[check] ** 2).sum())]], dtype=torch.float64)
    (sol, rec), = c10.law_abiding_solve_gram_multi(g_in, a[~check].T @ y[~check], g_ck, a[check].T @ y[check], y_ck,
                                                   [prior], [0], [[1.0]], h, None)
    assert sol is not None and bool(torch.isfinite(sol).all())
    broken = [r["lambda_relative"] for r in rec["rungs_not_factorisable"]]
    assert 1e-6 in broken and 1.0 not in broken
    assert rec["free_solve_rung"] == min(r["lambda_relative"] for r in rec["ladder"]) > max(broken)
    assert rec["lambda_relative"] not in broken
    with pytest.raises(c10.qf_solve.GramNotPositiveDefinite) as none:
        c10.law_abiding_solve_gram_multi(-torch.eye(n, dtype=torch.float64) * 1e3, a[~check].T @ y[~check],
                                         -torch.eye(n, dtype=torch.float64), a[check].T @ y[check], y_ck,
                                         [prior], [0], [[1.0]], h, None)
    assert "no ladder rung factorises" in str(none.value)


# ------------------------------------------------------------------ 1.5.1 (2026-10-11, the S5 install)
def install_argv(trunks: dict, k3: dict, out: Path, extra: list[str] | None = None) -> list[str]:
    return ["--trunk", str(trunks["trunk"]), "--parent-trunk", str(trunks["parent"]), "--k3-dense", str(k3["dir"]),
            "--charts", str(k3["charts"]), "--evidence", str(trunks["evidence"]), "--k3-channels", "10",
            "--device", "cpu", "--group-size", "2", "--chunk", "64", "--gram-block", "16", "--out", str(out),
            "--k3-config-from-flags"] + (extra or [])


def test_single_target_refit_falls_back_to_the_rung_skipping_solver(installed: dict, trunks: dict, k3: dict,
                                                                     work: Path, monkeypatch) -> None:
    """D4 verdict blocker 1: c14's ladder raises GramNotPositiveDefinite on a rung that does not factorise.
    The install re-solves the SAME system with law_abiding_solve_gram_multi.  The fake raises only AFTER
    consuming its inputs the way c14 does (check split added back in place), so a fallback that re-used the
    mutated Gram would double-count the check rows and miss the healthy child."""
    calls = {"n": 0}

    def consuming_then_broken(g_in, b_in, g_ck, b_ck, y_sq, prior):
        calls["n"] += 1
        g_in.add_(g_ck)
        b_in.add_(b_ck)
        raise c10.qf_solve.GramNotPositiveDefinite("synthetic: leading minor not positive-definite", {})

    monkeypatch.setattr(c10.c14, "law_abiding_solve_gram", consuming_then_broken)
    out = work / "child_fallback"
    receipt = run(trunks, k3["dir"], k3["charts"], out, 10)
    assert calls["n"] == LAYERS
    child = read_safetensors(out, verify=True)
    ref = read_safetensors(installed["out"], verify=True)
    for k, (a, b) in enumerate(zip(receipt["layers"], installed["receipt"]["layers"])):
        fb = a["ridge"]["solver_fallback"]
        assert "GramNotPositiveDefinite" in fb["reason"] and fb["rungs_not_factorisable"] == []
        assert a["ridge"]["lambda_relative"] == b["ridge"]["lambda_relative"]
        assert a["ridge"]["veto_binding"] == b["ridge"]["veto_binding"]
        assert abs(a["ridge"]["update_scale"] - b["ridge"]["update_scale"]) < 1e-6
        assert "solver_fallback" not in b["ridge"]
        for name in ("up", "down"):
            got = child[f"blocks.{k}.ffn.{name}.weight"].torch(device="cpu", dtype=torch.float32)
            want = ref[f"blocks.{k}.ffn.{name}.weight"].torch(device="cpu", dtype=torch.float32)
            assert c10.relative_l2(got, want) < 1e-5, (k, name, c10.relative_l2(got, want))


def test_single_target_refit_still_refuses_when_no_rung_is_admitted(trunks: dict, k3: dict, work: Path,
                                                                      monkeypatch) -> None:
    def broken(*_a, **_k):
        raise c10.qf_solve.GramNotPositiveDefinite("synthetic", {})

    monkeypatch.setattr(c10.c14, "law_abiding_solve_gram", broken)
    monkeypatch.setattr(c10, "law_abiding_solve_gram_multi",
                        lambda *a, **k: [(None, {"refused": "no lambda on the ladder brings update_scale to 1.0 "
                                                            "(free 9.000); THE LAW refuses the install"})])
    with pytest.raises(SystemExit) as refused:
        c10.main(install_argv(trunks, k3, work / "child_fallback_refused", ["--overwrite"]))
    assert "layer 0" in str(refused.value) and "THE LAW refuses" in str(refused.value)

    def none_factorise(*_a, **_k):
        raise c10.qf_solve.GramNotPositiveDefinite("c10 union sweep refit: no ladder rung factorises (x)", {})

    monkeypatch.setattr(c10, "law_abiding_solve_gram_multi", none_factorise)
    with pytest.raises(SystemExit) as nothing:
        c10.main(install_argv(trunks, k3, work / "child_fallback_none", ["--overwrite"]))
    assert "layer 0" in str(nothing.value) and "no ladder rung factorises" in str(nothing.value)


def test_install_min_free_gib_stops_before_a_layer_and_before_the_child_write(installed: dict, trunks: dict,
                                                                               k3: dict, work: Path,
                                                                               monkeypatch) -> None:
    import collections

    # (a) a floor no disk meets: stop before layer 0, nothing persisted, exit 3
    out = work / "child_floor_a"
    code = c10.main(install_argv(trunks, k3, out, ["--overwrite", "--min-free-gib", "1e9"]))
    assert code == c10.EXIT_STOPPED
    assert not list(Path(f"{out}.partial").glob("L*.safetensors")) and not (out / "union_receipt.json").exists()

    # (b) room for every partial layer (~13 KB) but not for the child (> 260 KB of embedding alone):
    # all layers persist, the child write stops, and --resume writes it without re-solving anything
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(c10.shutil, "disk_usage", lambda _p: usage(2 ** 40, 0, 2 ** 30 + 100_000))
    out = work / "child_floor_b"
    code = c10.main(install_argv(trunks, k3, out, ["--overwrite", "--min-free-gib", "1"]))
    assert code == c10.EXIT_STOPPED
    partial = Path(f"{out}.partial")
    assert sorted(p.name for p in partial.glob("L*.safetensors")) == [f"L{k:02d}.safetensors" for k in range(LAYERS)]
    assert not (out / "union_receipt.json").exists()
    monkeypatch.undo()
    solved = {"n": 0}
    real = c10._run_layer

    def counting(*a, **k):
        solved["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(c10, "_run_layer", counting)
    code = c10.main(install_argv(trunks, k3, out, ["--resume", "--min-free-gib", "1"]))
    assert code == 0 and solved["n"] == 0
    receipt = json.loads((out / "union_receipt.json").read_text(encoding="utf-8"))
    assert receipt["partial"]["layers_reloaded"] == list(range(LAYERS))
    assert receipt["child_write_disk"]["min_free_gib"] == 1.0
    assert child_file_bytes(out) == child_file_bytes(installed["out"])


def test_last_group_is_not_advanced(trunks: dict, k3: dict, work: Path, monkeypatch) -> None:
    """1.5.1: the forward after the last group fed nothing; the install no longer runs it."""
    seen: list[list[int]] = []
    real = c10.StreamedEngine.advance

    def recording(self, group):
        seen.append(list(group))
        return real(self, group)

    monkeypatch.setattr(c10.StreamedEngine, "advance", recording)
    run(trunks, k3["dir"], k3["charts"], work / "child_advances", 10)
    assert seen == [[0, 1]]
