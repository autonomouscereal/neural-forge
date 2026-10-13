"""Actual tiny native full-depth D3c gradients, transaction and exact resume."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/train_music.py"
spec = importlib.util.spec_from_file_location("recovery_music_test", PATH)
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)


def setup(tmp_path):
    torch = pytest.importorskip("torch")
    source = os.environ.get("CHIMERA_D3_BASELINE")
    if not source:
        pytest.skip("set CHIMERA_D3_BASELINE to actual campaign module; no production weights")
    torch.set_num_threads(1)
    module = C.T.load_module(Path(source), "d3_hybrid_music_fixture")
    built = module.build_tiny_child(tmp_path / "tiny-base", seed=17)
    optimizer = {"band_lr": .001, "lora_lr": .001, "betas": [.9, .95], "eps": 1e-8, "weight_decay": 0.,
                 "clip": 1., "warmup_steps": 0, "min_lr_ratio": 1.}
    identity = {"fixture": True, "stage": "D3c", "base_seed": 17}
    def construct():
        plan = module.trainer_module().SitePlanConfig(rank=4, alpha=4., k3_rank=0, trunk_ffn_width=module.TINY.TRUNK, gain=False)
        hybrid = module.HybridD3(built["dir"], built["parent_config"], module.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch",
                                lora=plan, lora_seed=3, pin_boundaries=False, prefetch=False))
        return C.MusicBandEngine(module, hybrid, optimizer, identity, total_steps=3)
    engine = construct()
    prompt = torch.tensor([[1, 2, 3]])
    previous = torch.tensor([[4, 5, 4]]) + engine.model.music_band_begin
    segment = {"ids": torch.cat([prompt, previous], dim=1)}
    teacher = torch.randn(4, engine.rows, generator=torch.Generator().manual_seed(31))
    lp = torch.log_softmax(teacher, -1)
    values, ids = lp.topk(4, -1)
    kwargs = {"prompt_length": 3, "topk_codes": ids, "topk_logprobs_band": values}
    return torch, engine, construct, segment, kwargs


def test_full_depth_music_tables_lora_update_and_exact_resume(tmp_path):
    torch, engine, construct, segment, kwargs = setup(tmp_path)
    initial_embed, initial_head = engine.embed_master.clone(), engine.head_master.clone()
    initial_lora = {name: value.clone() for name, value in engine.census.items()}
    first = engine.train_step(segment, **kwargs)
    assert not first["skipped"] and first["head_rows"] == 16 and first["embed_rows"] == 2
    assert not torch.equal(initial_embed, engine.embed_master) and not torch.equal(initial_head, engine.head_master)
    assert any(not torch.equal(initial_lora[name], value) for name, value in engine.census.items())
    assert C.R.tensor_sha256(engine.embed_full[:engine.offset]) == engine.frozen_ledger_sha
    engine.save(tmp_path / "interrupted")
    expected = engine.train_step(segment, **kwargs)
    resumed = construct()
    resumed.resume(tmp_path / "interrupted")
    actual = resumed.train_step(segment, **kwargs)
    assert actual == expected and resumed.sampler_counter == engine.sampler_counter == 2
    assert torch.equal(resumed.embed_master, engine.embed_master) and torch.equal(resumed.head_master, engine.head_master)
    for name, value in engine.census.items():
        assert torch.equal(value, resumed.census[name])
    for name in ("m", "v", "steps"):
        assert torch.equal(getattr(engine.embed_optimizer, name), getattr(resumed.embed_optimizer, name))
        assert torch.equal(getattr(engine.head_optimizer, name), getattr(resumed.head_optimizer, name))
    resumed.identity = {**resumed.identity, "base_seed": 18}
    with pytest.raises(C.Error, match="exact resume refused"):
        resumed.resume(tmp_path / "interrupted")


def test_nonfinite_lora_gradient_cannot_partially_step_host_tables(tmp_path):
    torch, engine, _, segment, kwargs = setup(tmp_path)
    before_embed, before_head = engine.embed_master.clone(), engine.head_master.clone()
    before_lora = {name: value.clone() for name, value in engine.census.items()}
    handle = next(iter(engine.census.values())).register_hook(lambda grad: torch.full_like(grad, float("nan")))
    try:
        result = engine.train_step(segment, **kwargs)
    finally:
        handle.remove()
    assert result["skipped"] and engine.step == engine.sampler_counter == 0
    assert torch.equal(before_embed, engine.embed_master) and torch.equal(before_head, engine.head_master)
    assert not engine.embed_optimizer.steps.any() and not engine.head_optimizer.steps.any()
    assert all(torch.equal(before_lora[name], value) for name, value in engine.census.items())


def test_music_optimizer_cannot_guess_defaults():
    with pytest.raises(C.Error, match="explicitly pin"):
        C.options({"band_lr": .001})
