"""D3a authentic native unit/chart gradients and missing-target refusal."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/train_refit.py"
spec = importlib.util.spec_from_file_location("recovery_refit_test", PATH)
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)


def setup(tmp_path):
    torch = pytest.importorskip("torch")
    source = os.environ.get("CHIMERA_D3_BASELINE")
    if not source:
        pytest.skip("set CHIMERA_D3_BASELINE for actual native tiny unit fixture")
    torch.set_num_threads(1)
    module = A.T.load_module(Path(source), "d3_hybrid_refit_fixture")
    built = module.build_tiny_child(tmp_path / "tiny-base", seed=17)
    optimizer = {"lr": .001, "betas": [.9, .95], "eps": 1e-8, "weight_decay": 0., "clip": 1.}
    weights = {"chart_in": 1., "block": 1., "chart_out": 1.}
    identity = {"fixture": True, "stage": "D3a", "base_seed": 17}
    def construct():
        plan = module.trainer_module().SitePlanConfig(rank=4, alpha=4., k3_rank=0, trunk_ffn_width=module.TINY.TRUNK, gain=False)
        hybrid = module.HybridD3(built["dir"], built["parent_config"], module.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch",
                                lora=plan, lora_seed=3, pin_boundaries=False, prefetch=False))
        return A.UnitRefitEngine(module, hybrid, "tiny_ace", optimizer, identity, total_steps=3, weights=weights)
    engine = construct()
    g = torch.Generator().manual_seed(22)
    host = torch.randn(4, module.TINY.HIDDEN, generator=g)
    parent_in = torch.randn(4, engine.placement.spec.width, generator=g)
    parent_out = parent_in + torch.randn(4, engine.placement.spec.width, generator=g)
    write = torch.randn(4, module.TINY.HIDDEN, generator=g)
    kwargs = {"levels": torch.full((4,), .5), "row_types": torch.full((4,), 4, dtype=torch.int64)}
    return torch, engine, construct, (host, parent_in, parent_out, write), kwargs


def test_refit_trains_chart_out_and_whole_native_unit_with_exact_resume(tmp_path):
    torch, engine, construct, tensors, kwargs = setup(tmp_path)
    initial = {name: value.clone() for name, value in engine.census.items()}
    first = engine.train_step(*tensors, **kwargs)
    assert not first["skipped"] and first["chart_out_rel_mse"] > 0
    assert any(not torch.equal(initial[name], value) for name, value in engine.census.items() if "chart_out" in name)
    source_parameters = {name: value for name, value in engine.census.items() if ".block_params." in name}
    assert source_parameters and any(not torch.equal(initial[name], value) for name, value in source_parameters.items())
    assert all(value.grad is not None for value in engine.census.values())
    assert not any(value.requires_grad for value in engine.hybrid.bank.tensors.values())
    engine.save(tmp_path / "interrupted")
    expected = engine.train_step(*tensors, **kwargs)
    resumed = construct()
    resumed.resume(tmp_path / "interrupted")
    actual = resumed.train_step(*tensors, **kwargs)
    assert actual == expected
    assert all(torch.equal(value, resumed.census[name]) for name, value in engine.census.items())


def test_missing_independent_outward_chart_target_refuses_without_parameter_mutation(tmp_path):
    torch, engine, _, tensors, kwargs = setup(tmp_path)
    before = {name: value.clone() for name, value in engine.census.items()}
    with pytest.raises(A.Error, match="old loss leaves it untrained"):
        engine.train_step(*tensors[:3], None, **kwargs)
    assert engine.step == engine.sampler_counter == 0
    assert all(torch.equal(before[name], value) for name, value in engine.census.items())
