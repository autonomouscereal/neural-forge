"""Genuine tiny Hybrid replay/continuity/exact resume, never production D2."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
from experiments.chimera_recovery import consolidation_runtime as R


def module():
    torch = pytest.importorskip("torch")
    path = os.environ.get("CHIMERA_D3_HYBRID", os.environ.get("CHIMERA_D3_BASELINE"))
    if not path:
        pytest.skip("genuine campaign module required for native CPU proof")
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    spec = importlib.util.spec_from_file_location("genuine_d4_union_runtime_fixture", path)
    M = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = M
    spec.loader.exec_module(M)
    return torch, M


def fixture(tmp_path):
    torch, M = module()
    built = M.build_tiny_child(tmp_path / "tiny", seed=91)
    T = M.trainer_module()
    ctx = dict(base_sha256=R.I.sha(built["dir"] / "config.json"), parent_config_sha256=R.I.sha(built["parent_config"] / "config.json"),
               hybrid_module_sha256=R.I.sha(Path(M.__file__)), fixture=True)
    class ComposedFixture:
        """Explicit synthetic D2-composed view; no authentic source claim."""
        def __init__(self, source, overlays):
            self.source, self.overlays = source, overlays
        def __getattr__(self, name):
            return getattr(self.source, name)
        def get(self, name):
            return self.overlays[name] if name in self.overlays else self.source.get(name)
        def identity(self):
            return {**self.source.identity(), "fixture_d2_composed": True}
    def prepare(*, corrupt=False):
        plan = T.SitePlanConfig(rank=3, alpha=6., k3_rank=0, gain=False, trunk_ffn_width=M.TINY.TRUNK)
        hybrid = M.HybridD3(built["dir"], built["parent_config"], M.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch",
                            mode="streamed", lora=plan, lora_seed=92, pin_boundaries=False, prefetch=False))
        original = hybrid.source
        d2_plan = T.SitePlanConfig(rank=2, alpha=3., k3_rank=2, k3_alpha=3., gain=True, trunk_ffn_width=M.TINY.TRUNK)
        d2 = T.AdapterBank(T.build_site_plan(hybrid.text_config, d2_plan), hybrid.bank.layer_types, "cpu", 93, b_init_std=.03, gain_jitter=.1)
        class D2View:
            plan, layer_types = d2.plan, d2.layer_types
            @staticmethod
            def bound_site(index, site):
                spec = next(s for s in d2.plan[d2.layer_types[index]] if s.site == site)
                parts = d2.layer_sites[index][site]
                return {**parts, "b": parts["b"] * spec.scaling}
        overlays = {}
        for index, kind in enumerate(d2.layer_types):
            base = {spec.site: original.get(f"{T.DEFAULT_TEXT_PREFIX}.layers.{index}.{spec.site}.weight") for spec in d2.plan[kind]}
            merged = R.D.merged_site_weights(base, D2View(), index)
            overlays.update({f"{T.DEFAULT_TEXT_PREFIX}.layers.{index}.{site}.weight": value for site, value in merged.items()})
        if corrupt:
            first = next(iter(overlays))
            overlays[first] = overlays[first].clone()
            overlays[first][0, 0] += .05
        hybrid.source = hybrid.engine.source = ComposedFixture(original, overlays)
        R.I.initialize_stage(hybrid, dict(mode="initialize", context=ctx), tmp_path)
        R.N.initialize_stage(hybrid, dict(mode="initialize", context=ctx), tmp_path)
        # Declare this simulated terminal native state as fixture only. Every
        # native unit exists; protect genuine frozen coefficients and code reserves.
        hybrid._recovery_native_unit_keys = {key for key in R.N.unit_parameters(hybrid)
                                            if not any(key.endswith("." + suffix) for suffix in ("h3_time.curve_table", "attn.rope_inv_freq"))}
        hybrid.add_bridge_deltas(list(hybrid.bands))
        hybrid.add_emission_deltas(list(hybrid.bands))
        for delta in hybrid.bridge_deltas + hybrid.head_deltas:
            assert not delta.delta.detach().any()
            delta.ranges.sort()
            with torch.no_grad():
                delta.delta.fill_(.001)
        values, descriptors = {}, {}
        for kind in ("bridge", "emission"):
            masters, desc = R.N.cumulative_interfaces(hybrid, kind)
            values.update({key: value.detach().clone() for key, value in masters.items()})
            descriptors.update(desc)
        hybrid.bridge_deltas, hybrid.head_deltas = [], []
        hybrid._recovery_native_interfaces, hybrid._recovery_native_descriptors = values, descriptors
        contract = R.N.music_contract(hybrid, code_rows=10)
        hybrid._recovery_native_music_contract = contract
        hybrid.band_embed_override = hybrid.band_embed.float().cpu().clone()
        hybrid._recovery_music_head_master = hybrid.band_head[contract["offset"]:].float().cpu().clone()
        hybrid.band_embed_override[contract["offset"]:contract["offset"] + 10] += .002
        hybrid._recovery_music_head_master[:10] += .003
        binding = dict(weights_semantics="uncomposed_pre_D2", d2_already_embedded=False, source_identity_sha256=R.I.digest(original.identity()))
        return hybrid, original, d2, binding
    rates = {family: .001 for family in R.D.FAMILIES}
    def construct():
        hybrid, original, d2, binding = prepare()
        return R.construct_union(M, hybrid, original, d2, binding, {}, {}, rates, fixture=True)
    return torch, M, prepare, construct


def replay(tmp_path, torch, M, hybrid):
    rows = []
    for index in range(4):
        segment = M.tiny_segment(hybrid.cfg, "ace", rows=4, seed=110 + index)
        target = torch.randn(1, 4, M.TINY.FEATURE, generator=torch.Generator().manual_seed(120 + index))
        family = ["D2.lora", "D2.k3_lora", "D2.gain", "D3.shared_lora", "D3.unit", "D3.bridge", "D3.emission"]
        if index % 2:
            segment = dict(ids=torch.tensor([[1, 2, 3, hybrid.music_band_begin + 1, hybrid.music_band_begin + 2, hybrid.music_band_begin + 3]]))
            target = torch.tensor([2, 3, 4])
            family = ["D2.lora", "D2.k3_lora", "D2.gain", "D3.shared_lora", "D3.music_embed", "D3.music_head"]
        source, truth = tmp_path / f"input{index}.pt", tmp_path / f"target{index}.pt"
        torch.save(segment, source)
        torch.save(target, truth)
        rows.append(dict(id=f"synthetic-fixture-{index}", recovery_group_id=f"fixture-group-{index}", split="train", held=False,
                         train_admitted=True, blockers=[], objective="music" if index % 2 else "flow", active_families=family,
                         supervised_units=3 if index % 2 else 4, objective_weight=1.,
                         input=R.I.pin(source), target=R.I.pin(truth)))
    catalog = tmp_path / "fixture-catalog.json"
    catalog.write_bytes(R.I.canonical(rows))
    return rows, dict(catalog=R.I.pin(catalog))


def backward(torch, M):
    def run(hybrid, row, weight):
        segment = torch.load(row["input"]["path"], weights_only=True)
        target = torch.load(row["target"]["path"], weights_only=True)
        if row["objective"] == "music":
            def loss(normed, context):
                head = hybrid._recovery_d4_music["head"].to(normed.device, normed.dtype)
                logits = torch.nn.functional.linear(normed[:, 2:5].float(), head.float())
                return torch.nn.functional.cross_entropy(logits.reshape(-1, head.shape[0]), target) * weight
        else:
            loss = lambda normed, context: (hybrid.emission(normed, context).float() - target).square().mean() * weight
        value, _, _ = M.full_depth_forward_backward(hybrid, segment, loss)
        return value / weight
    return run


def test_actual_union_native_forward_and_protected_physical_owners(tmp_path):
    torch, M, prepare, construct = fixture(tmp_path)
    hybrid, original, d2, binding = prepare()
    segment = M.tiny_segment(hybrid.cfg, "ace", rows=4, seed=105)
    before = hybrid.forward(segment)
    # A fresh source is required; even an innocent forward populates loads.
    with pytest.raises(R.Error, match="precede layer"):
        R.rebase_text(hybrid, original, d2)
    fresh, original, d2, binding = prepare()
    union = R.construct_union(M, fresh, original, d2, binding, {}, {}, {key: .001 for key in R.D.FAMILIES}, fixture=True)
    # Shared B starts zero: D2 child and once-only original+D2 runtime agree.
    after = fresh.forward(segment)
    torch.testing.assert_close(before["normed"], after["normed"], rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(before["emission"], after["emission"], rtol=3e-5, atol=3e-6)
    assert set(union["groups"]) == R.D.FAMILIES
    assert union["groups"]["D3.music_embed"]["music.embed"].shape[0] == 10
    assert union["identity"]["fixture"] and not union["identity"]["production_ready"]
    corrupted, original, d2, _ = prepare(corrupt=True)
    with pytest.raises(R.Error, match="once-only numerical composition"):
        R.rebase_text(corrupted, original, d2)
    with pytest.raises(R.Error, match="authenticated terminal"):
        R.construct_union(M, *prepare(), {}, {}, {key: .001 for key in R.D.FAMILIES})


def test_actual_native_optimizer_replay_resume_and_nonfinite_retry(tmp_path):
    torch, M, _, construct = fixture(tmp_path)
    union = construct()
    rows, sources = replay(tmp_path, torch, M, union["hybrid"])
    run = backward(torch, M)
    engine = R.UnionEngine(union, rows, sources, tmp_path, accumulation=2)
    initial = {name: value.detach().clone() for name, value in engine.census.items()}
    assert engine.window(run)["applied"] and engine.cursor == 2 and engine.step == 1
    assert all(any(not torch.equal(value, initial[name]) for name, value in named.items()) for named in union["groups"].values())
    engine.checkpoint(tmp_path / "saved")
    assert engine.window(run)["applied"] and engine.cursor == 4 and engine.step == 2
    final = {name: value.detach().clone() for name, value in engine.census.items()}
    restart = R.UnionEngine(construct(), rows, sources, tmp_path, accumulation=2)
    objects = {name: id(value) for name, value in restart.census.items()}
    restart.resume(tmp_path / "saved")
    assert all(id(value) == objects[name] for name, value in restart.census.items())
    frozen = {name: value.detach().clone() for name, value in restart.census.items()}
    rejected = restart.window(lambda *_: float("nan"))
    assert not rejected["applied"] and (restart.cursor, restart.step) == (2, 1)
    assert all(torch.equal(value, frozen[name]) for name, value in restart.census.items())
    assert restart.window(run)["applied"]
    assert all(torch.equal(value, final[name]) for name, value in restart.census.items())
    with pytest.raises(R.Error, match="already consumed"):
        restart.window(run)
    bad = R.UnionEngine(construct(), rows, sources, tmp_path, accumulation=1)
    with pytest.raises(R.Error, match="identity"):
        bad.resume(tmp_path / "saved")
    # Frozen full base must remain guarded although the compact delta uses the
    # same native name in the census.
    with torch.no_grad():
        next(iter(R.N.fixed_parameters(restart.hybrid).values())).add_(.01)
    with pytest.raises(R.Error, match="frozen native"):
        restart.checkpoint(tmp_path / "bad-base-save")
    Path(rows[0]["target"]["path"]).write_bytes(b"changed target after admission")
    with pytest.raises(R.I.Error, match="missing or changed"):
        restart.checkpoint(tmp_path / "bad-source-save")


def test_train_only_immutable_replay_is_not_a_quality_claim(tmp_path):
    source = tmp_path / "actual.bin"
    source.write_bytes(b"opaque fixture; not a real teacher cache")
    row = dict(id="fixture", recovery_group_id="fixture", split="validation", held=False, train_admitted=True, blockers=[],
               objective="fixture", supervised_units=1, objective_weight=1., active_families=["D2.lora"], input=R.I.pin(source), target=R.I.pin(source))
    with pytest.raises(R.Error, match="train-only"):
        R.replay_identity([row], dict(source=R.I.pin(source)), tmp_path)
    row["split"] = "train"
    assert R.replay_identity([row], dict(source=R.I.pin(source)), tmp_path)["exclusive_one_pass"]
    row["held"] = True
    with pytest.raises(R.Error, match="train-only"):
        R.replay_identity([row], dict(source=R.I.pin(source)), tmp_path)
