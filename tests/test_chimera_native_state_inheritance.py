"""State continuity refusal cases and actual sequential native engine parity."""
import copy
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys

import pytest
from experiments.chimera_recovery import native_state_inheritance as N
from experiments.chimera_recovery import stage_inheritance as I


def context():
    return dict(base_sha256="1" * 64, parent_config_sha256="2" * 64, hybrid_module_sha256="3" * 64, fixture=True)


def metadata_source(root):
    root.mkdir(exist_ok=True)
    data = root / "fixture.safetensors"
    header = I.canonical({"unit.fixture.weight": dict(dtype="F32", shape=[2, 2], data_offsets=[0, 16])})
    data.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<4f", 1, 2, 3, 4))
    state, receipt = root / "synthetic-state.bin", root / "receipt.json"
    state.write_bytes(b"metadata fixture only; no real Torch checkpoint")
    receipt.write_bytes(I.canonical(dict(complete=True, state_sha256=I.sha(state), identity_sha256="4" * 64, step=1)))
    producer = dict(checkpoint_receipt=I.pin(receipt), checkpoint_state=I.pin(state), run_identity_sha256="4" * 64, step=1)
    bank = dict(schema=I.SCHEMA, complete_bank=True, all_finite=True, semantics="full_bank_replacement", context=context(),
                exporter_sha256=I.sha(Path(I.__file__)), site_plan={}, site_plan_sha256=I.digest({}),
                tensors=I.pin(data), tensor_census=I.tensor_header(data), producer=producer)
    def write(path, value):
        value.pop("seal_sha256", None)
        value["seal_sha256"] = I.digest(value)
        path.write_bytes(I.canonical(value))
        return I.pin(path)
    shared = write(root / "bank.json", bank)
    native = dict(schema=N.SCHEMA, complete_cumulative_state=True, stage_training_complete=True, all_finite=True, context=context(), exporter_sha256=I.sha(Path(N.__file__)),
                  semantics=dict(unit="replacement", bridge="cumulative_fp32_delta", emission="cumulative_fp32_delta", music="fp32_master_replacement"),
                  shared_r16=shared, producer=producer, tensors=I.pin(data), tensor_census=I.tensor_header(data),
                  native_inventory={}, native_inventory_sha256=I.digest({}), descriptors={"unit.fixture.weight": dict(kind="unit")})
    path = root / "native.json"
    return write(path, native), shared, native, write


def test_native_bank_same_producer_and_actual_raw_state_pins_required(tmp_path):
    native, shared, _, _ = metadata_source(tmp_path)
    proof = N.verify_entry(native, tmp_path, context(), shared["sha256"])
    assert proof["manifest_sha256"] == native["sha256"]
    with pytest.raises(N.Error, match="shared-bank"):
        N.verify_entry(native, tmp_path, context(), "f" * 64)
    (tmp_path / "synthetic-state.bin").write_bytes(b"changed actual checkpoint")
    with pytest.raises(N.Error, match="missing or changed"):
        N.verify_entry(native, tmp_path, context(), shared["sha256"])


@pytest.mark.parametrize("field,change,pattern", [
    ("producer", {"step": 2}, "same checkpoint"), ("descriptors", {}, "semantics inventory"),
    ("tensor_census", {}, "actual header"), ("exporter_sha256", "0" * 64, "source"),
    ("complete_cumulative_state", False, "cumulative"),
    ("stage_training_complete", False, "cumulative"),
])
def test_cumulative_labels_cannot_substitute_a_matching_actual_inventory(tmp_path, field, change, pattern):
    native, shared, value, write = metadata_source(tmp_path)
    value[field] = {**value[field], **change} if field == "producer" else change
    native = write(Path(native["path"]), value)
    with pytest.raises(N.Error, match=pattern):
        N.verify_entry(native, tmp_path, context(), shared["sha256"])


def test_only_first_refit_initializes_native_and_bank_together(tmp_path):
    config = dict(stage="D3a", native_state=dict(mode="initialize", first_refit=True))
    shared = dict(mode="initialize", context=context())
    assert N.preflight_config(config, tmp_path, shared)["mode"] == "initialize"
    config["stage"] = "D3b"
    with pytest.raises(N.Error, match="first D3a"):
        N.preflight_config(config, tmp_path, shared)
    config["stage"] = "D3a"
    shared["mode"] = "inherit"
    with pytest.raises(N.Error, match="first D3a"):
        N.preflight_config(config, tmp_path, shared)


def test_actual_native_media_music_vision_continuity_preserves_frozen_families_and_gradients(tmp_path):
    torch = pytest.importorskip("torch")
    path = os.environ.get("CHIMERA_D3_HYBRID", os.environ.get("CHIMERA_D3_BASELINE"))
    if not path:
        pytest.skip("set genuine campaign module for actual native CPU proof")
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("actual_native_state_fixture", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_specs = module.tiny_unit_specs
    module.tiny_unit_specs = lambda depth: original_specs(depth) + [dataclasses.replace(original_specs(depth)[-1], layer_index=depth + 4, source="qvl32", component="tiny.qvl32", type_ids=(6,))]
    built = module.build_tiny_child(tmp_path / "tiny-base", seed=17)
    ctx = dict(base_sha256=I.sha(built["dir"] / "config.json"), parent_config_sha256=I.sha(built["parent_config"] / "config.json"), hybrid_module_sha256=I.sha(Path(path)), fixture=True)
    from experiments.chimera_recovery import train_media as M, train_music as C, train_vision as V
    def construct(seed):
        plan = module.trainer_module().SitePlanConfig(rank=4, alpha=4., k3_rank=0, trunk_ffn_width=module.TINY.TRUNK, gain=False)
        return module.HybridD3(built["dir"], built["parent_config"], module.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch", lora=plan, lora_seed=seed, pin_boundaries=False, prefetch=False))
    def initialize(model, source=None):
        if source:
            bank, native = source
            bank_id = I.initialize_stage(model, dict(mode="inherit", context=ctx, source=bank), tmp_path)
            native_id = N.initialize_stage(model, dict(mode="inherit", context=ctx, source=native, shared_r16_sha256=bank["sha256"]), tmp_path)
        else:
            bank_id = I.initialize_stage(model, dict(mode="initialize", context=ctx), tmp_path)
            native_id = N.initialize_stage(model, dict(mode="initialize", context=ctx), tmp_path)
        return dict(fixture=True, shared_r16=bank_id, native_state=native_id)
    def export(engine, name):
        checkpoint = tmp_path / (name + "-checkpoint")
        engine.save(checkpoint)
        bank = I.export_from_engine(engine, checkpoint, tmp_path / (name + "-bank"), ctx, name)
        native = N.export_from_engine(engine, checkpoint, tmp_path / (name + "-native"), ctx, name, bank)
        return bank, native
    def parity(left, right):
        for kind in ("ace", "h3", "ocr", "text"):
            segment = module.tiny_segment(left.cfg, kind, rows=4, seed=18)
            a, b = left.forward(segment), right.forward(segment)
            assert torch.equal(a["normed"], b["normed"]), kind
            if kind != "text":
                assert torch.equal(a["emission"], b["emission"]), kind
    # First stage uses the actual D3b optimizer and native streamed backward.
    donor = construct(3)
    segment = module.tiny_segment(donor.cfg, "ace", rows=4, seed=18)
    baseline = donor.forward(segment)["emission"].clone()
    identity = initialize(donor)
    assert torch.equal(baseline, donor.forward(segment)["emission"])
    options = dict(unit_lr=.001, emission_lr=.001, lora_lr=.001, betas=[.9, .95], eps=1e-8, weight_decay=0., clip=1., warmup_steps=0, min_lr_ratio=1.)
    media = M.D3bEngine(module, donor, donor.layout[0].container, options, identity, total_steps=1)
    target = torch.randn(1, 4, module.TINY.FEATURE, generator=torch.Generator().manual_seed(19))
    result = media.train_step(segment, lambda prediction: M.R.flow_velocity_loss(prediction, target, segment["types"], torch.ones(1, 4), donor.bands, param="x0", x_t=segment["features"], sigma=segment["sigma"], noisy_feature_bands={4: (6, 12)}))
    assert not result["skipped"] and not torch.equal(baseline, donor.forward(segment)["emission"])
    source1 = export(media, "D3b")
    receiver = construct(44)
    identity = initialize(receiver, source1)
    parity(donor, receiver)
    assert not receiver.head_deltas and not receiver.bridge_deltas
    # Actual music masters/Adam update, including six source-frozen reserves.
    music_options = dict(band_lr=.001, lora_lr=.001, betas=[.9, .95], eps=1e-8, weight_decay=0., clip=1., warmup_steps=0, min_lr_ratio=1.)
    music = C.MusicBandEngine(module, receiver, music_options, {**identity, "fixture_written_code_rows": 10}, total_steps=1)
    ids = torch.tensor([[1, 2, 3, receiver.music_band_begin + 4, receiver.music_band_begin + 5, receiver.music_band_begin + 4]])
    lp = torch.log_softmax(torch.randn(4, 10, generator=torch.Generator().manual_seed(31)), -1)
    top_lp, top_ids = lp.topk(4, -1)
    assert not music.train_step({"ids": ids}, prompt_length=3, topk_codes=top_ids, topk_logprobs_band=top_lp)["skipped"]
    source2 = export(music, "D3c")
    third = construct(91)
    identity = initialize(third, source2)
    parity(receiver, third)
    assert torch.equal(music.embed_master, third.band_embed_override[music.offset:])
    assert torch.equal(music.head_master, third._recovery_music_head_master)
    # Actual QVL engine learns bridge/type6 while retaining earlier emissions.
    vision_options = {**options, "bridge_lr": options["emission_lr"]}
    vision_options.pop("emission_lr")
    vision = V.VisionEngine(module, third, "tiny_qvl32", vision_options, identity, total_steps=1)
    owned = [id(value) for group in vision.optimizer.param_groups for value in group["params"]]
    assert len(owned) == len(set(owned))
    assert all(not name.startswith("emission.") for name in vision.census)
    seg = module.tiny_segment(third.cfg, "ocr", prompt=6, rows=5, seed=8, suffix=4)
    length = seg["ids"].shape[1]
    positions = torch.arange(length - 4, length - 1)
    lp = torch.log_softmax(torch.randn(3, third.base_vocab, generator=torch.Generator().manual_seed(12)), -1)
    qlp, qids = lp.topk(5, -1)
    targets = module.answer_kd_targets(length, positions, torch.tensor([2, 3, 4]), qids, qlp)
    features = torch.randn(1, 5, vision.placement.spec.width, generator=torch.Generator().manual_seed(13))
    assert not vision.train_step(seg, targets, features, answer_positions=3)["skipped"]
    assert all(any(value.grad is not None and value.grad.norm() > 0 for name, value in vision.census.items() if name.startswith(prefix)) for prefix in ("unit.", "bridge.", "lora."))
    source3 = export(vision, "D3d")
    fourth = construct(999)
    identity = initialize(fourth, source3)
    parity(third, fourth)
    restarted_music = C.MusicBandEngine(module, fourth, music_options, {**identity, "fixture_written_code_rows": 10}, total_steps=1)
    assert torch.equal(music.embed_master, restarted_music.embed_master)
    assert torch.equal(music.head_master, restarted_music.head_master)
    assert not restarted_music.embed_optimizer.steps.any()  # Stage optimizers are fresh.
    manifest = N.verify_entry(source3[1], tmp_path, ctx, source3[0]["sha256"])["manifest"]
    assert manifest["parent_manifest_sha256"] == source2[1]["sha256"]
    assert {"music.embed", "music.head", "bridge.feature_proj.weight", "emission.out.weight"}.issubset(manifest["descriptors"])
    # Refuse illicit sparse writes and duplicate live owners, without mutation.
    master = third._recovery_native_interfaces["emission.out.weight"].clone()
    desc = third._recovery_native_descriptors["emission.out.weight"]
    master[0, 0] = 1
    with pytest.raises(N.Error, match="outside owned"):
        N.checked_sparse(master, desc, third)
    third.bridge_deltas.append(third.bridge_deltas[0])
    with pytest.raises(N.Error, match="physical stage delta"):
        third.forward(seg)
