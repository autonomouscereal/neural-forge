"""Real tiny H3/full-depth CE, prompt-only attention, gradients and exact resume."""
import copy
import importlib.util
import os
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/train_audio.py"
spec = importlib.util.spec_from_file_location("recovery_audio_test", PATH)
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)


def setup(tmp_path):
    torch = pytest.importorskip("torch")
    source = os.environ.get("CHIMERA_D3_BASELINE")
    if not source:
        pytest.skip("set CHIMERA_D3_BASELINE for genuine native tiny fixture")
    foundation = os.environ.get("CHIMERA_OMNI_SRC")
    if foundation:
        sys.path.insert(0, str(Path(foundation) / "src"))
    torch.set_num_threads(1)
    module = A.T.load_module(Path(source), "d3_hybrid_audio_fixture")
    built = module.build_tiny_child(tmp_path / "tiny-base", seed=17)
    optimizer = {"unit_lr": .001, "bridge_lr": .001, "lora_lr": .001, "betas": [.9, .95], "eps": 1e-8, "weight_decay": 0.,
                 "clip": 1., "warmup_steps": 0, "min_lr_ratio": 1.}
    identity = {"fixture": True, "stage": "audio_ce", "base_seed": 17}
    def construct():
        plan = module.trainer_module().SitePlanConfig(rank=4, alpha=4., k3_rank=0, trunk_ffn_width=module.TINY.TRUNK, gain=False)
        model = module.HybridD3(built["dir"], built["parent_config"], module.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch",
                                lora=plan, lora_seed=3, pin_boundaries=False, prefetch=False))
        return A.AudioCEEngine(module, model, "tiny_h3", optimizer, identity, total_steps=3)
    engine = construct()
    segment = module.tiny_segment(engine.hybrid.cfg, "h3", prompt=6, rows=5, seed=8, suffix=4)
    # The original mathematical builder supplies mixed AV only. Create an
    # audio-only native fixture with the actual tiny allocated audio band.
    segment["types"].fill_(3)
    segment["features"].zero_()
    lo, hi = engine.hybrid.bands[3]["feature"]
    segment["features"][..., lo:hi] = torch.randn(1, 5, hi - lo, generator=torch.Generator().manual_seed(33))
    segment["ids"][0, -4:] = torch.tensor([4, 5, 6, 7])
    segment["sigma"].zero_()
    length = segment["ids"].shape[1]
    mask = torch.zeros_like(segment["ids"], dtype=torch.bool)
    mask[:, length - 4:] = True
    # These zero positions are exclusively this mathematical fixture. They are
    # not a production standalone-audio coordinate recipe.
    segment["source_positions"] = {"h3": torch.zeros(1, length, 3)}
    kwargs = {"prompt_length": length - 4, "completion_eos_id": int(segment["ids"][0, -1])}
    return torch, module, engine, construct, segment, mask, kwargs


def test_ce_targets_shift_once_and_exclude_prompt_media_padding(tmp_path):
    torch, _, engine, _, segment, mask, kwargs = setup(tmp_path)
    targets, count = A.assistant_ce_targets(engine.hybrid, segment, mask, **kwargs)
    prefix = kwargs["prompt_length"]
    assert count == 4 and int(targets["ce_w"].sum()) == 4
    assert targets["ce_w"][:prefix - 1].sum() == 0 and targets["ce_w"][-1] == 0
    assert torch.equal(targets["next_ids"][prefix - 1:-1], segment["ids"][0, prefix:])
    assert not targets["eos_w"].any() and "q_ids" not in targets
    bad = mask.clone()
    bad[0, 0] = True
    with pytest.raises(A.Error, match="terminal native assistant"):
        A.assistant_ce_targets(engine.hybrid, segment, bad, **kwargs)
    segment["sigma"][0, 0] = .1
    with pytest.raises(A.Error, match="clean type3"):
        A.validate_clean_observation(engine.hybrid, segment, prefix)


def test_h3_joint_attention_cannot_read_future_reference_answer(tmp_path):
    torch, _, engine, _, segment, _, kwargs = setup(tmp_path)
    model = engine.hybrid
    context = model.prepare(segment, media_grad=True)
    x = torch.randn_like(context["x"])
    changed = x.clone()
    changed[:, kwargs["prompt_length"]:] += 3.
    original = model.unit_write
    # Demonstrate the real baseline leak, then the bounded prompt-only policy.
    unsafe1, unsafe2 = original(engine.placement, x, context), original(engine.placement, changed, context)
    assert not torch.equal(unsafe1, unsafe2)
    with A.prompt_only_h3(model, engine.placement.container, kwargs["prompt_length"]):
        safe1, safe2 = model.unit_write(engine.placement, x, context), model.unit_write(engine.placement, changed, context)
        assert torch.equal(safe1, safe2)
        assert not safe1[:, kwargs["prompt_length"]:].any()
    assert "unit_write" not in model.__dict__


def test_full_depth_audio_ce_matches_independent_eager_gradient(tmp_path):
    torch, module, engine, _, segment, mask, kwargs = setup(tmp_path)
    model = engine.hybrid
    targets, count = A.assistant_ce_targets(model, segment, mask, **kwargs)
    context = model.prepare(segment, media_grad=True)
    x, contexts = context["x"], model.contexts(context["x"])
    with A.prompt_only_h3(model, engine.placement.container, kwargs["prompt_length"]):
        for index in range(model.num_layers):
            layer = copy.deepcopy(model.engine.layer(index, index + 1 if index + 1 < model.num_layers else None))
            model.bank.bind(layer, index)
            x = model.engine._run(layer, index, x, contexts)
            x = model.apply_units(x, index, context)
    normed = model.final_norm(x).reshape(-1, model.cfg.hidden_size)
    selected = targets["ce_w"] > 0
    logits = torch.nn.functional.linear(normed[selected], model.engine.head_weight[:model.base_vocab])
    eager = torch.nn.functional.cross_entropy(logits.float(), targets["next_ids"][selected], reduction="sum") / count
    eager.backward()
    expected = {name: value.grad.clone() for name, value in engine.census.items()}
    engine.optimizer.zero_grad(set_to_none=True)
    with A.prompt_only_h3(model, engine.placement.container, kwargs["prompt_length"]):
        loss, _, _ = module.full_depth_forward_backward(model, segment, A.ce_head_loss(model, targets, count))
    assert abs(loss - float(eager.detach())) < 1e-5
    for name, value in engine.census.items():
        assert torch.allclose(value.grad, expected[name], atol=2e-5, rtol=2e-4), name
    assert all(any(value.grad.norm() > 0 for name, value in engine.census.items() if name.startswith(group + ".")) for group in ("unit", "bridge", "lora"))


def test_audio_unit_bridge_lora_update_exact_resume_and_identity_refusal(tmp_path):
    torch, _, engine, construct, segment, mask, kwargs = setup(tmp_path)
    before = {name: value.clone() for name, value in engine.census.items()}
    first = engine.train_step(segment, mask, **kwargs)
    assert not first["skipped"] and first["teacher_logits_used"] is False
    assert all(any(not torch.equal(value, before[name]) for name, value in engine.census.items() if name.startswith(group + ".")) for group in ("unit", "bridge", "lora"))
    engine.save(tmp_path / "interrupted")
    expected = engine.train_step(segment, mask, **kwargs)
    resumed = construct()
    resumed.resume(tmp_path / "interrupted")
    actual = resumed.train_step(segment, mask, **kwargs)
    assert actual == expected and resumed.sampler_counter == engine.sampler_counter == 2
    assert all(torch.equal(value, resumed.census[name]) for name, value in engine.census.items())
    resumed.identity["read_policy"] = "unsafe_all_text"
    with pytest.raises(A.Error, match="exact resume refused"):
        resumed.resume(tmp_path / "interrupted")
    with pytest.raises(A.Error, match="reviewed sequential"):
        engine.export(tmp_path / "invalid-delta")
