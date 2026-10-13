"""Actual tiny native geometry mechanics; fixtures never certify source quality."""
import copy
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.chimera_recovery import consolidation_dispatch as D, consolidation_objectives as O, train_geometry as G
from test_chimera_consolidation_runtime import module


def write(path, value):
    path.write_bytes(D.I.canonical(value) + b"\n")
    return D.I.pin(path)


def test_reviewed_membership_preserves_original_quotas_and_only_replaces_geometry_lineage(tmp_path):
    order = D.replay_lineage_order()
    assert order.index(D.EDIT_STAGE)==order.index(G.STAGE)+1
    source = write(tmp_path / "source.json", {"fixture": True, "not_a_production_seal": True})
    value = dict(schema="chimera_recovery.d4_replay_amendment.v1", selected=True, dated="2026-10-13",
        preregistration_amendment_sha256="a" * 64, frozen_original_quotas=copy.deepcopy(D.C.QUOTAS), lineage_order=order,
        rationale="all eligible authentic source batches once; no repetition to fill guided quotas",
        lanes={lane: dict(mode="all_eligible_source_batches_once", objective=D.LANES[lane],
                         source_manifests=[source], batch_ids=[lane + "-fixture"], count=1)
               for lane in D.AMENDED_LANES})
    ref = write(tmp_path / "amend.json", value)
    checked = D.reviewed_replay_plan(ref, tmp_path, "a" * 64)
    assert checked["value"]["frozen_original_quotas"]["hy3d"] == 300
    assert D.LANES["music_code"] == O.MUSIC and D.LANES["hy3d"] == O.GEOMETRY
    for mutate in (lambda v: v.update(selected=False), lambda v: v["lineage_order"].reverse(),
                   lambda v: v["lanes"]["hy3d"].update(batch_ids=["same", "same"], count=2),
                   lambda v: v["lanes"]["h3"].update(count=300),
                   lambda v: v['lanes'].pop('image_edit'),
                   lambda v: v["frozen_original_quotas"].update(hy3d=32)):
        changed = copy.deepcopy(value)
        mutate(changed)
        bad = write(tmp_path / "bad.json", changed)
        with pytest.raises(D.Error):
            D.reviewed_replay_plan(bad, tmp_path, "a" * 64)


def test_real_cli_entrypoint_defines_execute_before_invoking_main(tmp_path):
    # runpy executes the exact module as __main__. A blocked plan exercises the
    # genuine entry point; the namespace at its exit must contain the executor.
    config = tmp_path / "blocked.json"
    write(config, {"schema": D.SCHEMA, "stage": "D4"})
    source = "\n".join([
        "import runpy,sys", "seen={}",
        "def trace(frame,event,arg):",
        " if event=='call' and frame.f_code.co_name=='main' and frame.f_globals.get('__name__')=='__main__':",
        "  assert callable(frame.f_globals.get('execute')), 'CLI executes before executor definition'",
        " return trace",
        "sys.settrace(trace)",
        "sys.argv=['entry','preflight','--config',sys.argv[1]]",
        "runpy.run_module('experiments.chimera_recovery.consolidation_dispatch',run_name='__main__')"])
    result = subprocess.run([sys.executable, "-B", "-c", source, str(config)], capture_output=True, text=True)
    assert result.returncode == 2 and '"candidate_preflight_passed": false' in result.stdout and not result.stderr
    bad = G.preflight(write(tmp_path / "wrong-stage.json", {"schema": G.SCHEMA, "stage": "D3b"})["path"])
    assert not bad["candidate_preflight_passed"] and not bad["production_ready"]


def geometry_fixture(tmp_path):
    torch, M = module()
    M.TINY.FEATURE = 6144
    M.TINY_BANDS[7] = ((576, 640), (320, 384))
    M.TINY_BANDS[6] = ((640, 2176), (384, 385))
    original = M.tiny_unit_specs
    M.tiny_unit_specs = lambda after: [dataclasses.replace(spec, rope_theta=0.) if spec.source == "hy3d" else spec
                                      for spec in original(after)]
    built = M.build_tiny_child(tmp_path / "geometry-tiny", seed=212)
    T = M.trainer_module()
    def construct():
        return M.HybridD3(built["dir"], built["parent_config"], M.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch",
            mode="streamed", pin_boundaries=False, prefetch=False,
            lora=T.SitePlanConfig(rank=3, alpha=6., k3_rank=0, gain=False, trunk_ffn_width=M.TINY.TRUNK), lora_seed=213))
    hybrid = construct()
    segment = M.tiny_segment(hybrid.cfg, "hy3d", rows=5, seed=214)
    segment["types"][:] = torch.tensor([[6, 7, 7, 7, 7]])
    features = torch.zeros_like(segment["features"])
    features[:, :1, 640:2176] = .02
    features[:, 2:, 576:640] = torch.linspace(-.3, .3, 3 * 64).reshape(1, 3, 64)
    mask = torch.tensor([[0., 0., 1., 1., 1.]])
    canonical = torch.tensor([[0., .3, .3, .3, .3]])
    progress = torch.where(segment["types"] == 7, 1 - canonical, 0.)
    target = torch.zeros_like(features)
    target[:, 2:, 320:384] = .1
    batch = dict(input_ids=segment["ids"], native_media_features=features, native_media_targets=target,
        native_media_type_ids=segment["types"], native_media_indices=segment["indices"],
        native_media_positions=segment["positions"], native_media_noise_levels=progress,
        native_media_canonical_sigma=canonical, native_media_loss_weights=mask,
        native_media_loss_mask=mask.clone(), native_media_valid_mask=torch.ones(1, 5, dtype=torch.bool),
        charted_source_positions=torch.zeros(1, segment["ids"].shape[1], 3))
    def compute(prediction):
        return (-prediction[:, 2:, 320:384].float() - target[:, 2:, 320:384]).square().mean()
    return torch, M, construct, batch, compute


def test_actual_native_geometry_raw_sign_progress_and_scalar_accumulation(tmp_path):
    torch, M, construct, batch, compute = geometry_fixture(tmp_path)
    first, second = construct(), construct()
    unit = next(p.container for p in first.layout if p.spec.source == "hy3d")
    engines = [G.GeometryFlowEngine(M, h, unit, G.OPTIMIZER, {"fixture": True}, total_steps=2) for h in (first, second)]
    segment = O.geometry_segment(batch, media_placeholder=first.media_placeholder)
    assert set(segment["source_positions"]) == {"hy3d"}
    torch.testing.assert_close(segment["sigma"][0, 2:], torch.full((3,), .7))
    prediction = first.forward(segment)["emission"]
    expected = float(compute(prediction).detach())
    actual = O.geometry_backward(M, first, batch, compute, 3 * 64, 1.)
    scaled = O.geometry_backward(M, second, batch, compute, 3 * 64, .3)
    assert abs(actual - expected) < 1e-5 and abs(actual - scaled) < 1e-6
    for name, parameter in engines[0].census.items():
        assert parameter.grad is not None, name
        torch.testing.assert_close(engines[1].census[name].grad, parameter.grad * .3, rtol=2e-4, atol=2e-5)
    with pytest.raises(O.Error, match="denominator"):
        O.geometry_backward(M, first, batch, compute, 3 * 64 + 1, 1.)
    wrong = {**batch, "native_media_noise_levels": batch["native_media_canonical_sigma"]}
    with pytest.raises(O.Error, match="separate exact fields"):
        O.geometry_backward(M, first, wrong, compute, 3 * 64, 1.)


def test_actual_geometry_optimizer_authority_exact_resume_and_frozen_observation_bridge(tmp_path):
    torch, M, construct, batch, compute = geometry_fixture(tmp_path)
    unit = next(p.container for p in construct().layout if p.spec.source == "hy3d")
    def engine():
        return G.GeometryFlowEngine(M, construct(), unit, G.OPTIMIZER, {"fixture": True}, total_steps=2)
    initial = engine()
    before = {name: value.clone().detach() for name, value in initial.census.items()}
    def revoked():
        raise G.Error("fixture lease revoked")
    with pytest.raises(G.Error, match="lease revoked"):
        initial.train_step(batch, compute, gate=revoked)
    assert initial.step == initial.sampler_counter == 0 and not initial.optimizer.state
    assert all(torch.equal(value, before[name]) for name, value in initial.census.items())
    first = initial.train_step(batch, compute, gate=lambda: None)
    assert not first["skipped"] and initial.step == initial.sampler_counter == 1
    delta = next(d for d in initial.hybrid.bridge_deltas if d.name == "feature_proj.weight")
    assert delta.ranges == [(576, 640)]  # type6 observation columns are frozen
    initial.save(tmp_path / "first-step")
    resumed = engine()
    resumed.resume(tmp_path / "first-step")
    resumed.train_step(batch, compute, gate=lambda: None)
    initial.train_step(batch, compute, gate=lambda: None)
    assert resumed.step == resumed.sampler_counter == 2
    assert all(torch.equal(value, initial.census[name]) for name, value in resumed.census.items())
    assert len({id(p) for g in resumed.optimizer.param_groups for p in g["params"]}) == len(resumed.census)
    with pytest.raises(G.Error, match="remaining one-pass"):
        resumed.train_step(batch, compute, gate=lambda: None)
