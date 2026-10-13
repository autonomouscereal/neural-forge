"""Actual D3d answer KD plus ViT-block13 feature KD through unit/C11/LoRA.

Feature targets must be authentic packed parent outputs aligned to this sample.
No adapter guesses a teacher feature from image pixels or token counts.
"""
from __future__ import annotations
from contextlib import contextmanager
import importlib.util
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chimera_recovery_vision_driver", HERE / "train_media.py")
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)
R, Error = T.R, T.Error


@contextmanager
def capture_unit_block(unit, outputs):
    """Observe the native block method without changing its computation/weights."""
    prior_instance_override = "block" in unit.__dict__
    original = unit.block
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        outputs.append(result)
        return result
    unit.block = observed
    try:
        yield
    finally:
        if prior_instance_override:
            unit.block = original
        else:
            delattr(unit, "block")


def feature_forward_backward(module, model, segment, text_objective, unit_name, teacher_features, *, loss_scale=1.):
    """Exact streamed backward with feature loss attached during unit recompute."""
    import torch
    if type(loss_scale) not in (int, float) or not math.isfinite(loss_scale) or loss_scale <= 0:
        raise Error("D3d loss_scale must be a finite positive whole-objective weight")
    if teacher_features.ndim != 3 or teacher_features.requires_grad or not bool(torch.isfinite(teacher_features).all()):
        raise Error("D3d requires detached finite packed teacher features [1,N,W]")
    placement = next(p for p in model.layout if p.container == unit_name)
    if placement.depth <= 0:
        raise Error("D3d selected unit must be after at least one shared trunk layer")
    context = model.prepare(segment, media_grad=True)
    x0 = context["x"]
    bounds, pre = [], {}
    h = model.run_layers(x0.detach(), 0, model.num_layers, context, bounds=bounds, pre_units=pre)
    last = h.detach().requires_grad_(True)
    with torch.enable_grad():
        text_loss = text_objective(model.final_norm(last), context)
    if not bool(torch.isfinite(text_loss)):
        raise Error("D3d text loss is not finite")
    (text_loss * loss_scale).backward()
    grad, feature_value, captures = last.grad, None, 0
    contexts = model.contexts(grad)
    for i in reversed(range(model.num_layers)):
        if i in pre:
            y = pre[i].to(model.device, copy=True).requires_grad_(True)
            feature_outputs = []
            if i == placement.depth - 1:
                with capture_unit_block(model.units[unit_name], feature_outputs), torch.enable_grad():
                    z = model.apply_units(y, i, context)
            else:
                with torch.enable_grad():
                    z = model.apply_units(y, i, context)
            if feature_outputs:
                captures += len(feature_outputs)
                actual = feature_outputs[0]
                if len(feature_outputs) != 1 or actual.shape != teacher_features.shape:
                    raise Error("D3d teacher block13 features differ from exact owned-row packing")
                # Absolute feature MSE at the frozen weight0.1; all text KD weight1.
                feature_loss = .1 * (actual.float() - teacher_features.to(actual.device).float()).square().mean()
                if not bool(torch.isfinite(feature_loss)):
                    raise Error("D3d feature loss is not finite")
                torch.autograd.backward((z, feature_loss), (grad, torch.ones_like(feature_loss) * loss_scale))
                feature_value = float(feature_loss.detach())
            else:
                z.backward(grad)
            grad = y.grad
        layer = model.engine.layer(i, i - 1 if i > 0 else None)
        x = bounds[i].to(model.device, copy=True).requires_grad_(True)
        with torch.enable_grad():
            y = model.engine._run(layer, i, x, contexts)
        y.backward(grad)
        grad = x.grad
    if captures != 1 or feature_value is None or not x0.requires_grad:
        raise Error("D3d feature objective or C11 input graph was not connected")
    x0.backward(grad)
    return {"loss": float(text_loss.detach()) + feature_value, "text_kd": float(text_loss.detach()), "feature_kd_weighted": feature_value,
            "feature_weight": .1, "text_weight": 1., "feature_captures": captures}


class VisionEngine(T.D3bEngine):
    checkpoint_schema = "chimera_recovery.d3d_checkpoint.v1"

    def __init__(self, module, hybrid, unit, optimizer, identity, *, total_steps):
        import torch
        value = dict(optimizer)
        if "bridge_lr" not in value:
            raise Error("D3d requires explicit bridge_lr (C11 interface-in rows)")
        value["emission_lr"] = value.pop("bridge_lr")
        self.module, self.hybrid, self.options, self.identity = module, hybrid, T.optimizer_config(value), identity
        self.placement = next((p for p in hybrid.layout if p.container == unit and p.type_ids == (6,)), None)
        if self.placement is None or hybrid.bridge_deltas or hybrid.head_deltas or total_steps < 1:
            raise Error("D3d needs a fresh exact QVL type6 unit and positive budget")
        if identity.get("fixture") is not True:
            for key in ("base_sha256", "teacher_cache_sha256", "amendment_sha256", "hybrid_module_sha256"):
                R.require_sha(identity.get(key), "D3d " + key)
            if identity.get("teacher_feature_block") != 13 or identity.get("heldout_groups_excluded") is not True:
                raise Error("D3d production requires admitted authentic ViT-block13 supervision")
        for native_unit in hybrid.units.values():
            native_unit.requires_grad_(False)
        hybrid.unit_parameters([unit])
        deltas = hybrid.add_bridge_deltas([6])
        for tensor in hybrid.bank.tensors.values():
            tensor.requires_grad_(True)
        self.census = {f"unit.{unit}.{name}": tensor for name, tensor in hybrid.units[unit].named_parameters() if tensor.requires_grad}
        self.census.update({f"bridge.{delta.name}": delta.delta for delta in deltas})
        self.census.update({"lora." + name: tensor for name, tensor in sorted(hybrid.bank.tensors.items()) if tensor.requires_grad})
        groups = [{"name": kind, "params": [value for name, value in self.census.items() if name.startswith(kind + ".")],
                   "lr": rate, "base_lr": rate} for kind, rate in (("unit", self.options["unit_lr"]),
                   ("bridge", self.options["emission_lr"]), ("lora", self.options["lora_lr"]))]
        if any(not group["params"] for group in groups):
            raise Error("D3d must train the QVL unit, C11 bridge and all-layer LoRA")
        self.optimizer = torch.optim.AdamW(groups, betas=tuple(self.options["betas"]), eps=self.options["eps"], weight_decay=self.options["weight_decay"])
        self.ownership = R.optimizer_ownership(self.census, self.optimizer)
        self.initial = {name: tensor.detach().cpu().clone() for name, tensor in self.census.items()}
        self.step, self.sampler_counter, self.consecutive_skips, self.skip_window, self.total_steps = 0, 0, 0, [], total_steps
        self.decode_review_passed = False

    def train_step(self, segment, answer_targets, teacher_features, *, answer_positions):
        import torch
        if self.step >= self.total_steps or answer_positions < 1:
            raise Error("D3d exhausted one-pass budget or has no assistant answer targets")
        weights, indices = answer_targets["q_w"], answer_targets["q_ids"]
        supervised = weights > 0
        if (int(supervised.sum()) != answer_positions or weights.shape != (segment["ids"].shape[1],) or
                bool((weights[supervised] != 1).any()) or bool(((indices[supervised] < 0) | (indices[supervised] >= self.hybrid.base_vocab)).any())):
            raise Error("D3d assistant target count, weights or native text vocabulary differs")
        if self.identity.get("fixture") is not True and self.hybrid.base_vocab != 248320:
            raise Error("D3d text objective must use exact rows[0,248320)")
        self.optimizer.zero_grad(set_to_none=True)
        text = self.module.d3d_head_loss(self.hybrid, answer_targets, denominator=float(answer_positions))
        parts = feature_forward_backward(self.module, self.hybrid, segment, text, self.placement.container, teacher_features)
        missing = [name for name, value in self.census.items() if value.grad is None]
        if missing:
            raise Error(f"D3d declared trainable received no gradient: {missing[:3]}")
        if not math.isfinite(parts["loss"]) or any(not bool(torch.isfinite(value.grad).all()) for value in self.census.values()):
            self.consecutive_skips += 1
            self.skip_window = (self.skip_window + [True])[-100:]
            return {"skipped": True, "step": self.step, "loss": None}
        factor = self.module.trainer_module().lr_factor(self.step, self.options["warmup_steps"], self.total_steps, self.options["min_lr_ratio"])
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        norm = torch.nn.utils.clip_grad_norm_(list(self.census.values()), self.options["clip"], error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1
        self.sampler_counter += 1
        self.consecutive_skips = 0
        self.skip_window = (self.skip_window + [False])[-100:]
        return {**parts, "skipped": False, "step": self.step, "gradient_norm": float(norm)}
