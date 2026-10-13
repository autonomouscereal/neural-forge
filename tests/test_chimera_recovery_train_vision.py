"""D3d native QVL fixture: full objective parity and optimizer/resume proof."""
import dataclasses
import importlib.util
import os
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/train_vision.py"
spec = importlib.util.spec_from_file_location("recovery_vision_test", PATH)
V = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = V
spec.loader.exec_module(V)


def setup(tmp_path):
    torch = pytest.importorskip("torch")
    source = os.environ.get("CHIMERA_D3_BASELINE")
    if not source:
        pytest.skip("set CHIMERA_D3_BASELINE for actual native tiny fixture")
    torch.set_num_threads(1)
    module = V.T.load_module(Path(source), "d3_hybrid_vision_fixture")
    original = module.tiny_unit_specs
    def specs(insert_after):
        units = original(insert_after)
        # Add a real ChartedSourceBlock under the existing tiny builder; no mockunit.
        return units + [dataclasses.replace(units[-1], layer_index=insert_after + 4, source="qvl32", component="tiny.qvl32", type_ids=(6,))]
    module.tiny_unit_specs = specs
    built = module.build_tiny_child(tmp_path / "tiny-base", seed=17)
    optimizer = {"unit_lr": .001, "bridge_lr": .001, "lora_lr": .001, "betas": [.9, .95], "eps": 1e-8, "weight_decay": 0.,
                 "clip": 1., "warmup_steps": 0, "min_lr_ratio": 1.}
    identity = {"fixture": True, "stage": "D3d", "base_seed": 17}
    def construct():
        plan = module.trainer_module().SitePlanConfig(rank=4, alpha=4., k3_rank=0, trunk_ffn_width=module.TINY.TRUNK, gain=False)
        hybrid = module.HybridD3(built["dir"], built["parent_config"], module.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch",
                                lora=plan, lora_seed=3, pin_boundaries=False, prefetch=False))
        return V.VisionEngine(module, hybrid, "tiny_qvl32", optimizer, identity, total_steps=3)
    engine = construct()
    segment = module.tiny_segment(engine.hybrid.cfg, "ocr", prompt=6, rows=5, seed=8, suffix=4)
    length = segment["ids"].shape[1]
    positions = torch.arange(length - 4, length - 1)
    lp = torch.log_softmax(torch.randn(3, engine.hybrid.base_vocab, generator=torch.Generator().manual_seed(12)), -1)
    qlp, qids = lp.topk(5, -1)
    targets = module.answer_kd_targets(length, positions, torch.tensor([2, 3, 4]), qids, qlp)
    features = torch.randn(1, 5, engine.placement.spec.width, generator=torch.Generator().manual_seed(13))
    return torch, module, engine, construct, segment, targets, features


def test_streamed_vision_text_and_feature_gradients_match_eager_native_path(tmp_path):
    torch, module, engine, _, segment, targets, teacher = setup(tmp_path)
    model = engine.hybrid
    captures = []
    context = model.prepare(segment, media_grad=True)
    x = context["x"]
    contexts = model.contexts(x)
    with V.capture_unit_block(model.units[engine.placement.container], captures):
        for i in range(model.num_layers):
            layer = model.engine.layer(i, i + 1 if i + 1 < model.num_layers else None)
            x = model.engine._run(layer, i, x, contexts)
            x = model.apply_units(x, i, context)
    text = module.d3d_head_loss(model, targets, denominator=3.)(model.final_norm(x), context)
    eager = text + .1 * (captures[0].float() - teacher).square().mean()
    eager.backward()
    expected = {name: tensor.grad.clone() for name, tensor in engine.census.items()}
    engine.optimizer.zero_grad(set_to_none=True)
    result = V.feature_forward_backward(module, model, segment, module.d3d_head_loss(model, targets, denominator=3.), engine.placement.container, teacher)
    assert result["feature_weight"] == .1 and result["text_weight"] == 1. and result["feature_captures"] == 1
    assert abs(result["loss"] - float(eager.detach())) < 1e-5
    for name, value in engine.census.items():
        assert torch.allclose(value.grad, expected[name], atol=2e-5, rtol=2e-4), name
    assert all(any(float(value.grad.norm()) > 0 for name, value in engine.census.items() if name.startswith(group + ".")) for group in ("unit", "bridge", "lora"))
    assert "block" not in model.units[engine.placement.container].__dict__


def test_vision_qvl_bridge_lora_update_exact_resume_and_feature_alignment_refusal(tmp_path):
    torch, _, engine, construct, segment, targets, teacher = setup(tmp_path)
    first = engine.train_step(segment, targets, teacher, answer_positions=3)
    assert not first["skipped"] and first["feature_kd_weighted"] > 0
    engine.save(tmp_path / "interrupted")
    expected = engine.train_step(segment, targets, teacher, answer_positions=3)
    resumed = construct()
    resumed.resume(tmp_path / "interrupted")
    actual = resumed.train_step(segment, targets, teacher, answer_positions=3)
    assert actual == expected
    assert all(torch.equal(value, resumed.census[name]) for name, value in engine.census.items())
    with pytest.raises(V.Error, match="owned-row packing"):
        resumed.train_step(segment, targets, teacher[:, :-1], answer_positions=3)
