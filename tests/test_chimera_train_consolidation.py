"""Genuine tiny Hybrid runtime/merge/gradient proof; no production quality claim."""
import copy
import importlib.util
import os
from pathlib import Path
import sys

import pytest
from experiments.chimera_recovery import train_consolidation as D


def module():
    torch = pytest.importorskip("torch")
    path = os.environ.get("CHIMERA_D3_HYBRID", os.environ.get("CHIMERA_D3_BASELINE"))
    if not path:
        pytest.skip("genuine campaign module required for native CPU parity")
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    spec = importlib.util.spec_from_file_location("genuine_d4_hybrid_test", path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return torch, value


def build_joint(tmp_path):
    torch, M = module()
    built = M.build_tiny_child(tmp_path / "tiny-base", seed=71)
    T = M.trainer_module()
    def construct(mode="streamed"):
        shared_plan = T.SitePlanConfig(rank=3, alpha=6., k3_rank=0, gain=False, trunk_ffn_width=M.TINY.TRUNK)
        hybrid = M.HybridD3(built["dir"], built["parent_config"], M.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch",
                            mode=mode, lora=shared_plan, lora_seed=73, pin_boundaries=False, prefetch=False))
        original = T.SitePlanConfig(rank=2, alpha=3., k3_rank=2, k3_alpha=3., gain=True, trunk_ffn_width=M.TINY.TRUNK)
        d2 = T.AdapterBank(T.build_site_plan(hybrid.text_config, original), hybrid.bank.layer_types, "cpu", 74,
                           b_init_std=.03, gain_jitter=.15)
        with torch.no_grad():
            for name, value in hybrid.bank.tensors.items():
                if name.endswith(".b"):
                    value.fill_(.025)
        source = hybrid.source.identity()
        binding = dict(weights_semantics="uncomposed_pre_D2", d2_already_embedded=False, source_identity_sha256=D.I.digest(source))
        bank = D.JointAdapterBank(d2, hybrid.bank, source, binding, fixture=True)
        D.install_joint_bank(hybrid, bank)
        return hybrid
    return torch, M, T, built, construct


def test_joint_true_hybrid_forward_and_streamed_gradients_match_full_graph(tmp_path):
    torch, M, T, built, construct = build_joint(tmp_path)
    hybrid = construct()
    reference = M.HybridD3(built["dir"], built["parent_config"], M.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch",
                               lora=None, pin_boundaries=False, prefetch=False))
    overlays = {}
    for index, kind in enumerate(hybrid.bank.layer_types):
        bases = {spec.site: hybrid.source.get(f"{T.DEFAULT_TEXT_PREFIX}.layers.{index}.{spec.site}.weight") for spec in hybrid.bank.plan[kind]}
        composed = D.merged_site_weights(bases, hybrid.bank, index)
        overlays.update({f"{T.DEFAULT_TEXT_PREFIX}.layers.{index}.{site}.weight": value for site, value in composed.items()})
    original_get = reference.source.get
    reference.source.get = lambda name: overlays[name] if name in overlays else original_get(name)
    for kind in ("ace", "h3", "ocr", "text"):
        segment = M.tiny_segment(hybrid.cfg, kind, rows=4, seed=77)
        actual, merged = hybrid.forward(segment), reference.forward(segment)
        torch.testing.assert_close(actual["normed"], merged["normed"], rtol=3e-5, atol=3e-6)
        if "emission" in actual:
            torch.testing.assert_close(actual["emission"], merged["emission"], rtol=3e-5, atol=3e-6)
    # The production streamed primitive and a complete autograd graph traverse
    # exactly the same genuine native layers/units and both adapter banks.
    streamed, full = construct(), construct("resident")
    segment = M.tiny_segment(streamed.cfg, "ace", rows=4, seed=78)
    target = torch.randn(1, 4, M.TINY.FEATURE, generator=torch.Generator().manual_seed(79))
    loss_fn = lambda model: lambda normed, context: (model.emission(normed, context).float() - target).square().mean()
    loss, _, _ = M.full_depth_forward_backward(streamed, segment, loss_fn(streamed))
    context = full.prepare(segment)
    x = context["x"]
    x = full.apply_units(x, -1, context)
    contexts = full.contexts(x)
    for index in range(full.num_layers):
        layer = full.engine.layer(index, None)
        x = full.engine._run(layer, index, x, contexts)
        x = full.apply_units(x, index, context)
    reference_loss = loss_fn(full)(full.final_norm(x), context)
    reference_loss.backward()
    assert abs(loss - float(reference_loss.detach())) < 1e-5
    assert set(streamed.bank.tensors) == set(full.bank.tensors)
    for name, value in streamed.bank.tensors.items():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
        torch.testing.assert_close(value.grad, full.bank.tensors[name].grad, rtol=2e-4, atol=2e-6)
    # Resident binding must create a new graph on a second step, not reuse cat's
    # freed backward state; the same primitive remains valid after an update.
    full.bank.zero_grad()
    second, _, _ = M.full_depth_forward_backward(full, segment, loss_fn(full))
    assert abs(second - loss) < 1e-5
    # Duplicate application is observably different, rather than a label-only check.
    extra = copy.deepcopy(overlays)
    for index, kind in enumerate(hybrid.bank.layer_types):
        base = {spec.site: extra[f"{T.DEFAULT_TEXT_PREFIX}.layers.{index}.{spec.site}.weight"] for spec in hybrid.bank.plan[kind]}
        twice = D.merged_site_weights(base, hybrid.bank, index)
        extra.update({f"{T.DEFAULT_TEXT_PREFIX}.layers.{index}.{site}.weight": value for site, value in twice.items()})
    assert any(not torch.equal(value, extra[name]) for name, value in overlays.items())


def test_native_cumulative_activation_exact_forward_one_physical_owner(tmp_path):
    torch, M, _, _, construct = build_joint(tmp_path)
    hybrid = construct()
    # Native inheritance starts before any active delta/optimizer.
    hybrid._recovery_native_parent = "fixture inherited native source"
    hybrid._recovery_native_interfaces = {}
    hybrid._recovery_native_descriptors = {}
    hybrid._recovery_native_overlay_hashes = {}
    D.N.install_forward_adapter(hybrid)
    hybrid.add_bridge_deltas([3, 4])
    hybrid.add_emission_deltas([3, 4])
    # The original convenience helper orders by type id: tiny type3 occupies
    # feature20:24, type4 feature0:12. Cumulative descriptors rightly require
    # physical-range order. Confirm refusal, then canonicalize only these new
    # zero-initialized fixture deltas before assigning their training values.
    with pytest.raises(D.N.Error, match="overlap or repeat"):
        D.N.cumulative_interfaces(hybrid, "bridge")
    for delta in hybrid.bridge_deltas + hybrid.head_deltas:
        assert not delta.delta.detach().any()
        delta.ranges.sort()
    with torch.no_grad():
        for delta in hybrid.bridge_deltas + hybrid.head_deltas:
            delta.delta.fill_(.004)
    values, descriptors = {}, {}
    for kind in ("bridge", "emission"):
        masters, desc = D.N.cumulative_interfaces(hybrid, kind)
        values.update({key: value.detach().clone() for key, value in masters.items()})
        descriptors.update(desc)
    hybrid.bridge_deltas, hybrid.head_deltas = [], []
    hybrid._recovery_native_interfaces = values
    hybrid._recovery_native_descriptors = descriptors
    hybrid._recovery_native_overlay_hashes = {name: D.I.tensor_sha(value) for name, value in values.items()}
    segments = [M.tiny_segment(hybrid.cfg, kind, rows=4, seed=81) for kind in ("ace", "h3")]
    before = [hybrid.forward(segment) for segment in segments]
    census = D.activate_native_interfaces(M, hybrid)
    after = [hybrid.forward(segment) for segment in segments]
    for a, b in zip(before, after):
        assert torch.equal(a["normed"], b["normed"]) and torch.equal(a["emission"], b["emission"])
    for name, value in values.items():
        actual, _ = D.N.cumulative_interfaces(hybrid, name.split(".")[0])
        assert torch.equal(actual[name], value)
    assert not hybrid._recovery_native_interfaces and set(census) == {"bridge", "emission"}
    with pytest.raises(D.Error, match="no active deltas"):
        D.activate_native_interfaces(M, hybrid)


def test_real_leaf_aliases_and_composed_source_refused_fresh_optimizer_owns_banks(tmp_path):
    torch, _, _, _, construct = build_joint(tmp_path)
    hybrid = construct()
    joint = hybrid.bank
    bad = dict(joint.binding, d2_already_embedded=True)
    with pytest.raises(D.Error, match="composed D2 base"):
        D.JointAdapterBank(joint.d2, joint.shared, hybrid.source.identity(), bad, fixture=True)
    groups = {family: {} for family in ("D2.lora", "D2.k3_lora", "D2.gain", "D3.shared_lora")}
    for name, value in joint.tensors.items():
        family = "D3.shared_lora" if name.startswith("d3.") else "D2.gain" if name.endswith(".gain") else "D2.k3_lora" if ".k3_" in name else "D2.lora"
        groups[family][name] = value
    rates = {family: .001 for family in groups}
    optimizer, ownership = D.fresh_union_optimizer(groups, rates, require_complete=False)
    assert not optimizer.state and len(ownership["owners"]) == len(joint.tensors)
    assert all(group["lr"] == .0003 for group in optimizer.param_groups)
    snapshot = {name: value.detach().clone() for name, value in joint.tensors.items()}
    optimizer.zero_grad()
    loss = sum(value.square().sum() for value in joint.tensors.values())
    loss.backward()
    optimizer.step()
    assert all(not torch.equal(value.detach(), snapshot[name]) for name, value in joint.tensors.items())
    groups["D2.gain"]["second-reference"] = next(iter(groups["D2.lora"].values()))
    with pytest.raises(D.Error, match="duplicate physical"):
        D.fresh_union_optimizer(groups, rates, require_complete=False)
    with pytest.raises(D.Error, match="misses an actual"):
        D.fresh_union_optimizer({"D2.lora": groups["D2.lora"]}, {"D2.lora": .001})


def test_future_production_d2_binding_absence_is_a_refusal(tmp_path):
    # No actual D2 has been run/accepted; absence must not be filled by fixture facts.
    with pytest.raises(D.Error, match="missing or changed"):
        D.verify_d2_source(dict(path=str(tmp_path / "nonexistent-D2.json"), sha256="a" * 64), tmp_path, None, None)
