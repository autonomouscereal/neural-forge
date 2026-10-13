"""Distinct clean-audio observation -> authored assistant CE engine.

The current H3 unit accepts audio-only observations, but its joint attention
must see prompt text only, never teacher-forced answers. Authentic full-coverage
clean encoding and measured source positions remain production prerequisites.
This module creates no encoder, teacher logits, runtime allocation or GPU job.
"""
from __future__ import annotations
from contextlib import contextmanager
import importlib.util
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chimera_recovery_audio_driver", HERE / "train_media.py")
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)
R, Error = T.R, T.Error
OBJECTIVE = "authored_audio_assistant_ce"
READ_POLICY = "h3_prompt_and_clean_observation_only"


def assistant_ce_targets(model, segment, target_mask, prompt_length, completion_eos_id):
    """Own only real next assistant tokens; shift target ownership exactly once."""
    import torch
    ids = segment["ids"]
    if (ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.int64 or target_mask.shape != ids.shape or
            target_mask.dtype != torch.bool or type(prompt_length) is not int or not 0 < prompt_length < ids.shape[1] or
            type(completion_eos_id) is not int or not 0 <= completion_eos_id < model.base_vocab):
        raise Error("audio CE needs native IDs, boolean target ownership and exact generation-prompt boundary/EOS")
    owned = target_mask[0].nonzero().flatten()
    if (owned.numel() < 2 or int(owned[0]) != prompt_length or
            not torch.equal(owned, torch.arange(prompt_length, int(owned[-1]) + 1, device=owned.device)) or
            int(ids[0, owned[-1]]) != completion_eos_id or int(ids[target_mask].eq(completion_eos_id).sum()) != 1 or
            bool(((ids[target_mask] < 0) | (ids[target_mask] >= model.base_vocab)).any())):
        raise Error("audio CE supervision must be one terminal native assistant body plus completion EOS")
    next_ids = torch.full((ids.numel(),), -1, dtype=torch.int64, device=ids.device)
    weights = torch.zeros(ids.numel(), dtype=torch.float32, device=ids.device)
    selected = target_mask[0, 1:]
    next_ids[:-1] = torch.where(selected, ids[0, 1:], -1)
    weights[:-1] = selected.float()
    return {"next_ids": next_ids, "ce_w": weights, "eos_w": torch.zeros_like(weights)}, int(owned.numel())


def validate_clean_observation(model, segment, prompt_length):
    import torch
    ids, types, indices, features = (segment[key] for key in ("ids", "types", "indices", "features"))
    rows = types.numel()
    if (types.shape != (1, rows) or types.dtype != torch.int64 or rows < 1 or bool((types != 3).any()) or indices.shape != types.shape or
            indices.dtype != torch.int64 or bool(((indices < 0) | (indices >= prompt_length)).any()) or
            torch.unique(indices).numel() != rows or features.shape != (1, rows, model.cfg.native_feature_dim) or
            segment["sigma"].shape != types.shape or bool((segment["sigma"] != 0).any()) or
            segment["positions"].shape != (1, rows, 4)):
        raise Error("audio CE requires clean type3 observation rows wholly before the authentic prompt boundary")
    if bool((ids[0, indices[0]] != model.media_placeholder).any()):
        raise Error("audio CE observation indices must select genuine native media placeholder IDs")
    lo, hi = model.bands[3]["feature"]
    outside = features.clone()
    outside[..., lo:hi] = 0
    source = segment.get("source_positions", {}).get("h3")
    if (bool(outside.any()) or source is None or source.shape != (*ids.shape, 3) or
            any(not bool(torch.isfinite(value).all()) for value in (features, segment["positions"], source)) or
            any(value.requires_grad for value in (features, segment["positions"], source))):
        raise Error("audio CE needs detached genuine H3-band observations and finite measured native/source positions")


@contextmanager
def prompt_only_h3(model, unit_name, prompt_length):
    """Prevent answer -> joint audio attention -> earlier answer target leakage."""
    import torch
    prior_override, original = "unit_write" in model.__dict__, model.unit_write
    def write(placement, x, context):
        if placement.container != unit_name:
            return original(placement, x, context)
        types = context["row_types"]
        if types is None:
            raise Error("audio CE H3 context lacks actual row ownership")
        positions = context["positions"][placement.spec.source]
        prefix = torch.arange(types.shape[1], device=types.device).unsqueeze(0) < prompt_length
        visible = context["valid"] & prefix & (types.eq(0) | types.eq(3))
        result = model.units[unit_name](x.to(model.options.unit_dtype), types, context["levels"], visible, source_positions=positions)
        return result.to(x.dtype)
    model.unit_write = write
    try:
        yield
    finally:
        if prior_override:
            model.unit_write = original
        else:
            delattr(model, "unit_write")


def ce_head_loss(model, targets, positions):
    import torch
    config = model.T.ObjectiveConfig(w_q38=0., lambda_k3=0., w_ce=1., w_eos=0., w_ul=0.,
                                    text_vocab_rows=model.base_vocab, head_chunk_positions=256)
    def loss(normed, context):
        head = model.engine.head_weight[:model.base_vocab]
        parts, gradient = model.T.chunked_objective(normed.to(head.device, head.dtype), head, targets, config,
                                                    denominator=float(positions), backward=True)
        surrogate = (normed * gradient.to(normed.device, normed.dtype)).sum()
        return surrogate - surrogate.detach() + torch.tensor(float(parts["total"]), device=normed.device)
    return loss


class AudioCEEngine(T.D3bEngine):
    checkpoint_schema = "chimera_recovery.audio_ce_checkpoint.v1"

    def __init__(self, module, hybrid, unit, optimizer, identity, *, total_steps):
        import torch
        value = dict(optimizer)
        if "bridge_lr" not in value:
            raise Error("audio CE requires explicit type3 bridge_lr")
        value["emission_lr"] = value.pop("bridge_lr")
        self.module, self.hybrid, self.options = module, hybrid, T.optimizer_config(value)
        self.identity = {**identity, "objective": OBJECTIVE, "read_policy": READ_POLICY,
                         "optimizer_sha256": R.digest(optimizer), "total_steps": total_steps}
        self.placement = next((p for p in hybrid.layout if p.container == unit and p.spec.source == "h3" and 3 in p.type_ids), None)
        if self.placement is None or hybrid.head_deltas or hybrid.bridge_deltas or total_steps < 1:
            raise Error("audio CE needs a fresh genuine H3 audio unit and positive exact budget")
        if identity.get("fixture") is not True:
            for key in ("base_sha256", "clean_encoder_receipt_sha256", "batch_manifest_sha256", "amendment_sha256", "hybrid_module_sha256"):
                R.require_sha(identity.get(key), "audio CE " + key)
            if (identity.get("objective") != OBJECTIVE or identity.get("read_policy") != READ_POLICY or
                    identity.get("heldout_groups_excluded") is not True or hybrid.base_vocab != 248320 or
                    hybrid.bands[3]["feature"] != (480, 512)):
                raise Error("audio CE production requires the distinct amended clean observation/assistant CE contract")
        for native_unit in hybrid.units.values():
            native_unit.requires_grad_(False)
        hybrid.unit_parameters([unit])
        deltas = hybrid.add_bridge_deltas([3])
        for tensor in hybrid.bank.tensors.values():
            tensor.requires_grad_(True)
        self.census = {f"unit.{unit}.{name}": value for name, value in hybrid.units[unit].named_parameters() if value.requires_grad}
        self.census.update({f"bridge.{delta.name}": delta.delta for delta in deltas})
        self.census.update({"lora." + name: tensor for name, tensor in sorted(hybrid.bank.tensors.items()) if tensor.requires_grad})
        groups = [{"name": kind, "params": [value for name, value in self.census.items() if name.startswith(kind + ".")],
                   "lr": rate, "base_lr": rate} for kind, rate in (("unit", self.options["unit_lr"]), ("bridge", self.options["emission_lr"]), ("lora", self.options["lora_lr"]))]
        if any(not group["params"] for group in groups):
            raise Error("audio CE ownership requires the H3 unit, type3 input bridge and shared LoRA")
        self.optimizer = torch.optim.AdamW(groups, betas=tuple(self.options["betas"]), eps=self.options["eps"], weight_decay=self.options["weight_decay"])
        self.ownership = R.optimizer_ownership(self.census, self.optimizer)
        self.initial = {name: value.detach().cpu().clone() for name, value in self.census.items()}
        self.step, self.sampler_counter, self.consecutive_skips, self.skip_window, self.total_steps = 0, 0, 0, [], total_steps
        self.decode_review_passed = False

    def train_step(self, segment, target_mask, *, prompt_length, completion_eos_id):
        import torch
        if self.step >= self.total_steps:
            raise Error("audio CE exhausted one-pass budget")
        targets, count = assistant_ce_targets(self.hybrid, segment, target_mask, prompt_length, completion_eos_id)
        validate_clean_observation(self.hybrid, segment, prompt_length)
        self.optimizer.zero_grad(set_to_none=True)
        with prompt_only_h3(self.hybrid, self.placement.container, prompt_length):
            loss, _, _ = self.module.full_depth_forward_backward(self.hybrid, segment, ce_head_loss(self.hybrid, targets, count))
        missing = [name for name, value in self.census.items() if value.grad is None]
        if missing:
            raise Error(f"audio CE trainable received no gradient: {missing[:3]}")
        if not math.isfinite(loss) or any(not bool(torch.isfinite(value.grad).all()) for value in self.census.values()):
            self.consecutive_skips += 1
            self.skip_window = (self.skip_window + [True])[-100:]
            return {"loss": None, "skipped": True, "step": self.step}
        norms = {group["name"]: float(torch.sqrt(sum(value.grad.float().square().sum() for value in group["params"]))) for group in self.optimizer.param_groups}
        if any(value <= 0 for value in norms.values()):
            raise Error("audio CE declared optimizer group received zero gradient")
        factor = self.module.trainer_module().lr_factor(self.step, self.options["warmup_steps"], self.total_steps, self.options["min_lr_ratio"])
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        norm = torch.nn.utils.clip_grad_norm_(list(self.census.values()), self.options["clip"], error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1
        self.sampler_counter += 1
        self.consecutive_skips, self.skip_window = 0, (self.skip_window + [False])[-100:]
        return {"loss": loss, "assistant_targets": count, "skipped": False, "step": self.step, "gradient_norm": float(norm), "group_norms": norms,
                "objective": OBJECTIVE, "read_policy": READ_POLICY, "teacher_logits_used": False}

    def export(self, out):
        raise Error("audio CE checkpoint needs a reviewed sequential native composer; no generic D3b emission export")
