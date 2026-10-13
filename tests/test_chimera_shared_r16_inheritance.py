"""Actual full-bank byte contracts and optional real native HybridD3 CPU proof."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import types

import pytest

from experiments.chimera_recovery import stage_inheritance as I


def context():
    return dict(base_sha256="1" * 64, parent_config_sha256="2" * 64, hybrid_module_sha256="3" * 64, fixture=True)


def metadata_source(tmp_path):
    # Stdlib contract fixture, not a native checkpoint or runtime tensor seal.
    tensor = tmp_path / "shared_r16.safetensors"
    header = I.canonical({"layers.0.v_proj.a": dict(dtype="F32", shape=[2, 2], data_offsets=[0, 16])})
    tensor.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<4f", 1, 2, 3, 4))
    state, receipt = tmp_path / "synthetic-state.bin", tmp_path / "receipt.json"
    state.write_bytes(b"synthetic metadata contract; never torch loaded")
    receipt.write_bytes(I.canonical(dict(complete=True, state_sha256=I.sha(state), identity_sha256="4" * 64, step=3)))
    plan = dict(specs={}, layer_types=[])
    value = dict(schema=I.SCHEMA, complete_bank=True, all_finite=True, semantics="full_bank_replacement", context=context(),
                 exporter_sha256=I.sha(Path(I.__file__)), site_plan=plan, site_plan_sha256=I.digest(plan),
                 tensors=I.pin(tensor), tensor_census=I.tensor_header(tensor),
                 producer=dict(checkpoint_receipt=I.pin(receipt), checkpoint_state=I.pin(state), run_identity_sha256="4" * 64, step=3))
    value["seal_sha256"] = I.digest(value)
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(I.canonical(value))
    return manifest, value


def rewrite(manifest, value):
    value.pop("seal_sha256", None)
    value["seal_sha256"] = I.digest(value)
    manifest.write_bytes(I.canonical(value))


def test_stdlib_header_receipt_context_and_actual_data_file_pins(tmp_path):
    manifest, _ = metadata_source(tmp_path)
    entry = I.pin(manifest)
    assert I.verify_entry(entry, tmp_path, context())["manifest_sha256"] == entry["sha256"]
    changed = context()
    changed["base_sha256"] = "a" * 64
    with pytest.raises(I.Error, match="context"):
        I.verify_entry(entry, tmp_path, changed)
    (tmp_path / "shared_r16.safetensors").write_bytes(b"tampered raw data")
    with pytest.raises(I.Error, match="source file missing or changed"):
        I.verify_entry(entry, tmp_path, context())


@pytest.mark.parametrize("field,value,error", [
    ("semantics", "add", "replacement"), ("complete_bank", False, "complete replacement"),
    ("tensor_census", {}, "header/site census"), ("exporter_sha256", "0" * 64, "source"),
])
def test_metadata_labels_cannot_substitute_additive_or_partial_bank_or_changed_source(tmp_path, field, value, error):
    manifest, source = metadata_source(tmp_path)
    source[field] = value
    rewrite(manifest, source)
    with pytest.raises(I.Error, match=error):
        I.verify_entry(I.pin(manifest), tmp_path, context())


def test_safetensors_header_overlap_and_missing_extent_refused(tmp_path):
    path = tmp_path / "overlap.safetensors"
    header = I.canonical({"a": dict(dtype="F32", shape=[2], data_offsets=[0, 8]), "b": dict(dtype="F32", shape=[2], data_offsets=[4, 12])})
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 12)
    with pytest.raises(I.Error, match="overlap"):
        I.tensor_header(path)


def test_only_explicit_first_refit_can_initialize_and_preflight_never_loads_torch(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    config = dict(parent_config="config.json", hybrid_module=dict(sha256="3" * 64), stage="D3a", shared_r16=dict(mode="initialize", first_refit=True))
    proof = I.preflight_config(config, tmp_path, "1" * 64)
    assert proof["mode"] == "initialize" and proof["context"]["fixture"] is False
    config["stage"] = "D3b"
    with pytest.raises(I.Error, match="first D3a"):
        I.preflight_config(config, tmp_path, "1" * 64)
    config.pop("shared_r16")
    with pytest.raises(I.Error, match="explicit shared_r16"):
        I.preflight_config(config, tmp_path, "1" * 64)


def fake_hybrid(torch, seed=1):
    generator = torch.Generator().manual_seed(seed)
    tensors = {f"layers.{layer}.v_proj.{part}": torch.randn(2, 2, generator=generator).requires_grad_()
               for layer in range(2) for part in ("a", "b")}
    spec = dict(rank=2, alpha=2., k3_rank=0, gain=False, site="v_proj", kind="full", in_features=2, out_features=2)
    return types.SimpleNamespace(bank=types.SimpleNamespace(tensors=tensors, names=sorted(tensors), plan={"full": [spec]}, layer_types=["full", "full"]))


def save_engine(torch, hybrid, path, identity):
    path.mkdir()
    census = {"lora." + name: value for name, value in hybrid.bank.tensors.items() if value.requires_grad}
    engine = types.SimpleNamespace(hybrid=hybrid, census=census, identity=identity, step=1)
    torch.save(dict(identity=identity, step=1, trainables={name: value.detach().cpu() for name, value in census.items()}), path / "state.pt")
    (path / "receipt.json").write_bytes(I.canonical(dict(complete=True, state_sha256=I.sha(path / "state.pt"), identity_sha256=I.digest(identity), step=1)))
    return engine


def test_copy_preserves_frozen_leaves_owner_objects_and_never_imports_optimizer_state(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")
    donor = fake_hybrid(torch, seed=1)
    initial = I.initialize_stage(donor, dict(mode="initialize", context=context()), tmp_path)
    donor.bank.tensors["layers.0.v_proj.a"].requires_grad_(False)
    donor.bank.tensors["layers.0.v_proj.b"].requires_grad_(False)
    with torch.no_grad():
        donor.bank.tensors["layers.1.v_proj.b"].add_(7)
    engine = save_engine(torch, donor, tmp_path / "checkpoint", dict(fixture=True, shared_r16=initial))
    source = I.export_from_engine(engine, tmp_path / "checkpoint", tmp_path / "bank", context(), "fixture")
    receiver = fake_hybrid(torch, seed=42)
    receiver.bank.tensors["layers.0.v_proj.a"].requires_grad_(False)
    objects = {name: (id(value), value.requires_grad) for name, value in receiver.bank.tensors.items()}
    proof = I.import_bank(receiver, source, tmp_path, context())
    assert proof["optimizer_state_imported"] is False
    assert objects == {name: (id(value), value.requires_grad) for name, value in receiver.bank.tensors.items()}
    assert all(torch.equal(donor.bank.tensors[name], receiver.bank.tensors[name]) for name in donor.bank.tensors)
    assert len(I.verify_entry(source, tmp_path, context())["manifest"]["tensor_census"]) == 4
    assert len(engine.census) == 2  # Current trainable census omitted two inherited leaves.


def test_alias_nonfinite_rank_mismatch_and_frozen_state_mutation_refused(tmp_path):
    torch = pytest.importorskip("torch")
    hybrid = fake_hybrid(torch)
    hybrid.bank.tensors["layers.1.v_proj.b"] = hybrid.bank.tensors["layers.0.v_proj.b"]
    with pytest.raises(I.Error, match="duplicate physical"):
        I.bank_description(hybrid, context())
    hybrid = fake_hybrid(torch)
    with torch.no_grad():
        hybrid.bank.tensors["layers.0.v_proj.a"][0, 0] = float("nan")
    with pytest.raises(I.Error, match="finite"):
        I.bank_description(hybrid, context())
    hybrid = fake_hybrid(torch)
    production = {**context(), "fixture": False}
    with pytest.raises(I.Error, match="r16"):
        I.bank_description(hybrid, production)
    initial = I.initialize_stage(hybrid, dict(mode="initialize", context=context()), tmp_path)
    hybrid.bank.tensors["layers.0.v_proj.b"].requires_grad_(False)
    with torch.no_grad():
        hybrid.bank.tensors["layers.0.v_proj.b"].add_(1)
    engine = save_engine(torch, hybrid, tmp_path / "checkpoint", dict(fixture=True, shared_r16=initial))
    with pytest.raises(I.Error, match="frozen shared bank"):
        I.export_from_engine(engine, tmp_path / "checkpoint", tmp_path / "bad-bank", context(), "fixture")


def test_actual_hybrid_d3_bank_handoff_keeps_prior_lower_layer_and_optimizer_binding(tmp_path):
    torch = pytest.importorskip("torch")
    path = os.environ.get("CHIMERA_D3_HYBRID")
    if not path:
        pytest.skip("set CHIMERA_D3_HYBRID to the genuine campaign module for actual native CPU proof")
    if torch.cuda.is_initialized():
        pytest.fail("native fixture must be CPU-only")
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("actual_hybrid_shared_bank_fixture", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from experiments.chimera_recovery import train_media as T
    built = module.build_tiny_child(tmp_path / "tiny-base", seed=17)
    ctx = dict(base_sha256=I.sha(built["dir"] / "config.json"), parent_config_sha256=I.sha(built["parent_config"] / "config.json"),
               hybrid_module_sha256=I.sha(Path(path)), fixture=True)
    def construct(seed):
        plan = module.trainer_module().SitePlanConfig(rank=4, alpha=4., k3_rank=0, trunk_ffn_width=module.TINY.TRUNK, gain=False)
        return module.HybridD3(built["dir"], built["parent_config"], module.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch",
                               lora=plan, lora_seed=seed, pin_boundaries=False, prefetch=False))
    donor = construct(3)
    inheritance = I.initialize_stage(donor, dict(mode="initialize", context=ctx), tmp_path)
    # Real AdamW on the genuine bank makes lower-layer B nonzero. This is a
    # random CPU transport fixture objective, not a production recovery claim.
    optimizer = torch.optim.AdamW(list(donor.bank.tensors.values()), lr=.01, weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    loss = sum((value - .25).square().mean() for value in donor.bank.tensors.values())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    prior = save_engine(torch, donor, tmp_path / "prior-checkpoint", dict(fixture=True, shared_r16=inheritance))
    source = I.export_from_engine(prior, tmp_path / "prior-checkpoint", tmp_path / "prior-bank", ctx, "fixture")
    receiver = construct(44)
    verified = dict(mode="inherit", context=ctx, source=source)
    inherited = I.initialize_stage(receiver, verified, tmp_path)
    segment = module.tiny_segment(receiver.cfg, "ace", rows=4, seed=18)
    assert torch.equal(donor.forward(segment)["normed"], receiver.forward(segment)["normed"])
    options = dict(unit_lr=.001, emission_lr=.001, lora_lr=.001, betas=[.9, .95], eps=1e-8, weight_decay=0., clip=1., warmup_steps=0, min_lr_ratio=1.)
    engine = T.D3bEngine(module, receiver, receiver.layout[0].container, options, dict(fixture=True, shared_r16=inherited), total_steps=1)
    lower = [name for name, value in receiver.bank.tensors.items() if not value.requires_grad]
    assert lower and all("lora." + name not in engine.census for name in lower)
    assert all(torch.equal(donor.bank.tensors[name], receiver.bank.tensors[name]) for name in lower)
    owned = [id(value) for group in engine.optimizer.param_groups for value in group["params"]]
    assert len(owned) == len(set(owned))
    target = torch.randn(1, 4, module.TINY.FEATURE, generator=torch.Generator().manual_seed(19))
    objective = lambda prediction: T.R.flow_velocity_loss(prediction, target, segment["types"], torch.ones(1, 4), receiver.bands,
            param="x0", x_t=segment["features"], sigma=segment["sigma"], noisy_feature_bands={4: (6, 12)})
    result = engine.train_step(segment, objective)
    assert not result["skipped"] and engine.step == 1
    engine.save(tmp_path / "stage-checkpoint")
    successor = I.export_from_engine(engine, tmp_path / "stage-checkpoint", tmp_path / "stage-bank", ctx, "D3b", engine.placement.container)
    third = construct(999)
    I.initialize_stage(third, dict(mode="inherit", context=ctx, source=successor), tmp_path)
    assert all(torch.equal(receiver.bank.tensors[name], third.bank.tensors[name]) for name in lower)
    assert I.verify_entry(successor, tmp_path, ctx)["manifest"]["parent_manifest_sha256"] == source["sha256"]
