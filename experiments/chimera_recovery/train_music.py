"""D3c host music-band masters + all-layer LoRA, with atomic finite admission.

This is an executable engine for a pinned caller/launcher, not a synthetic rate
loop. Authentic sealed ACE-LM cache/prompt adaptation and dated optimizer pins
are prerequisites to production; CPU fixtures must declare fixture=True.
"""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path
import random
import sys
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chimera_recovery_music_driver", HERE / "train_media.py")
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)
R, Error = T.R, T.Error
SCHEMA = "chimera_recovery.d3c_checkpoint.v1"


def options(value):
    expected = {"band_lr", "lora_lr", "betas", "eps", "weight_decay", "clip", "warmup_steps", "min_lr_ratio"}
    if set(value) != expected:
        raise Error("D3c optimizer must explicitly pin band/lora rates, Adam settings, clipping and schedule")
    for key in ("band_lr", "lora_lr", "eps", "clip"):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] <= 0:
            raise Error("D3c optimizer requires finite positive rates/epsilon/clip")
    if (not isinstance(value["betas"], list) or len(value["betas"]) != 2 or any(not 0 <= beta < 1 for beta in value["betas"]) or
            value["weight_decay"] < 0 or not math.isfinite(value["weight_decay"]) or
            type(value["warmup_steps"]) is not int or value["warmup_steps"] < 0 or not 0 < value["min_lr_ratio"] <= 1):
        raise Error("invalid D3c Adam/schedule settings")
    return dict(value)


class MusicBandEngine:
    def __init__(self, module, model, optimizer, identity, *, total_steps):
        import torch
        self.module, self.model, self.options, self.identity = module, model, options(optimizer), identity
        if type(total_steps) is not int or total_steps < 1 or model.band_head is None or model.head_deltas or model.bridge_deltas:
            raise Error("D3c needs an untied head, positive step budget and a fresh stage-specific hybrid")
        self.offset = model.music_band_begin - model.base_vocab
        self.rows = model.band_embed.shape[0] - self.offset
        self.code_rows = identity.get("fixture_written_code_rows", self.rows) if identity.get("fixture") is True else 64000
        if type(self.code_rows) is not int or not 1 <= self.code_rows <= self.rows:
            raise Error("D3c written code rows must be within allocated music band")
        if identity.get("fixture") is not True:
            if self.rows != 65536 or model.music_band_begin != 252416 or model.base_vocab != 248320 or model.cfg.hidden_size != 5120:
                raise Error("D3c production must own the exact frozen65536x5120 music-band contract")
            for key in ("base_sha256", "cache_manifest_sha256", "amendment_sha256", "hybrid_module_sha256"):
                R.require_sha(identity.get(key), "D3c " + key)
            if identity.get("stage") != "D3c" or identity.get("teacher_kind") != "sealed_ACE_LM_top64_tail" or identity.get("heldout_groups_excluded") is not True:
                raise Error("D3c production identity requires sealed ACE-LM targets and source-group admission")
        for unit in model.units.values():
            unit.requires_grad_(False)
        for tensor in model.bank.tensors.values():
            tensor.requires_grad_(True)
        self.census = {"lora." + name: tensor for name, tensor in sorted(model.bank.tensors.items()) if tensor.requires_grad}
        if not self.census:
            raise Error("D3c requires all-layer LoRA trainables")
        self.lora_optimizer = torch.optim.AdamW(list(self.census.values()), lr=self.options["lora_lr"],
                                              betas=tuple(self.options["betas"]), eps=self.options["eps"], weight_decay=self.options["weight_decay"])
        lora_ownership = R.optimizer_ownership(self.census, self.lora_optimizer)
        self.embed_full = model.band_embed.detach().float().cpu().clone()
        self.embed_master = self.embed_full[self.offset:self.offset + self.rows]
        self.head_master = model.band_head[self.offset:self.offset + self.rows].detach().float().cpu().clone()
        self.embed_optimizer = module.LazyRowAdam(self.embed_master, lr=self.options["band_lr"], betas=tuple(self.options["betas"]),
                                                eps=self.options["eps"], weight_decay=self.options["weight_decay"])
        self.head_optimizer = module.LazyRowAdam(self.head_master, lr=self.options["band_lr"], betas=tuple(self.options["betas"]),
                                               eps=self.options["eps"], weight_decay=self.options["weight_decay"])
        self.frozen_ledger_sha = R.tensor_sha256(self.embed_full[:self.offset])
        self.frozen_reserved_sha = {"embed": R.tensor_sha256(self.embed_master[self.code_rows:]),
                                    "head": R.tensor_sha256(self.head_master[self.code_rows:])}
        self.ownership = {"lora": lora_ownership, "host_music_tables": {
            "embed": {"shape": list(self.embed_master.shape), "active_code_rows": self.code_rows, "optimizer": "LazyRowAdam", "row_policy": "previous sampled codes only"},
            "head": {"shape": list(self.head_master.shape), "active_code_rows": self.code_rows, "optimizer": "LazyRowAdam", "row_policy": "all written code rows per step"},
            "reserved_rows_frozen": self.rows - self.code_rows, "frozen_ledger_sha256": self.frozen_ledger_sha,
            "frozen_reserved_sha256": self.frozen_reserved_sha}}
        model.band_embed_override = self.embed_full
        self.step, self.sampler_counter, self.skips, self.total_steps = 0, 0, 0, total_steps

    def train_step(self, segment, *, prompt_length, topk_codes, topk_logprobs_band):
        import torch
        if self.step >= self.total_steps:
            raise Error("D3c exhausted declared one-pass step budget")
        ids, n = segment["ids"], topk_codes.shape[0]
        if (ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.int64 or prompt_length < 1 or n < 1 or
                topk_codes.ndim != 2 or topk_logprobs_band.shape != topk_codes.shape or
                ids.shape[1] != prompt_length + n - 1 or not bool(torch.isfinite(topk_logprobs_band).all())):
            raise Error("D3c teacher forcing requires prompt + previous codes, with one teacher row per next-code position")
        teacher_rows = self.code_rows
        if (bool(((ids[:, :prompt_length] < 0) | (ids[:, :prompt_length] >= self.model.base_vocab)).any()) or
                bool(((ids[:, prompt_length:] < self.model.music_band_begin) | (ids[:, prompt_length:] >= self.model.music_band_begin + teacher_rows)).any()) or
                bool(((topk_codes < 0) | (topk_codes >= teacher_rows)).any()) or bool((topk_logprobs_band > 1e-5).any()) or
                bool((topk_logprobs_band.exp().sum(-1) > 1.00001).any())):
            raise Error("D3c text prefix/code bands or teacher probability mass invalid")
        if any(row.unique().numel() != row.numel() for row in topk_codes):
            raise Error("D3c teacher top-k repeats a code")
        self.lora_optimizer.zero_grad(set_to_none=True)
        # Always a separate leaf even on CPU fp32, preventing master grad carry-over.
        # The allocation has65536 rows, but the actual teacher/code vocabulary
        # has64000. Unwritten reserved rows must not enter the KL normalizer.
        head = self.head_master[:self.code_rows].to(self.model.device, self.model.dtype, copy=True).requires_grad_()
        positions = torch.arange(prompt_length - 1, prompt_length - 1 + n, device=self.model.device)
        def objective(normed, context):
            return self.module.band_kl(normed[0, positions], head, topk_codes.to(normed.device), topk_logprobs_band.to(normed.device))
        try:
            loss, grad_x0, _ = self.module.full_depth_forward_backward(self.model, segment, objective, input_grad=True)
        except self.module.D3Error as error:
            if "not finite" not in str(error):
                raise
            self.skips += 1
            return {"skipped": True, "step": self.step, "loss": None}
        if head.grad is None or grad_x0 is None or any(tensor.grad is None for tensor in self.census.values()):
            raise Error("D3c declared host-head/embedding/LoRA gradient missing")
        selected = ids[0] >= self.model.music_band_begin
        embed_rows, embed_grads = self.module.unique_row_grads((ids[0][selected] - self.model.music_band_begin).cpu(), grad_x0[0][selected.to(grad_x0.device)].float().cpu())
        head_grads = head.grad.detach().float().cpu()
        gradients = [head_grads, embed_grads] + [tensor.grad for tensor in self.census.values()]
        if not math.isfinite(loss) or any(not bool(torch.isfinite(grad).all()) for grad in gradients):
            self.skips += 1
            return {"skipped": True, "step": self.step, "loss": None, "host_tables_updated": False}
        if R.tensor_sha256(self.embed_full[:self.offset]) != self.frozen_ledger_sha:
            raise Error("D3c touched frozen control/media ledger embedding rows")
        if (R.tensor_sha256(self.embed_master[self.code_rows:]) != self.frozen_reserved_sha["embed"] or
                R.tensor_sha256(self.head_master[self.code_rows:]) != self.frozen_reserved_sha["head"]):
            raise Error("D3c touched unwritten reserved music rows")
        norm = math.sqrt(sum(float(grad.float().square().sum()) for grad in gradients))
        if not math.isfinite(norm) or norm <= 0:
            raise Error("D3c joint gradient norm invalid")
        scale = min(1., self.options["clip"] / (norm + 1e-12))
        for grad in gradients:
            grad.mul_(scale)
        factor = self.module.trainer_module().lr_factor(self.step, self.options["warmup_steps"], self.total_steps, self.options["min_lr_ratio"])
        self.embed_optimizer.lr = self.head_optimizer.lr = self.options["band_lr"] * factor
        for group in self.lora_optimizer.param_groups:
            group["lr"] = self.options["lora_lr"] * factor
        # No optimizer mutates until all three gradient sets passed the same gate.
        self.head_optimizer.step(torch.arange(self.code_rows), head_grads)
        self.embed_optimizer.step(embed_rows, embed_grads)
        self.lora_optimizer.step()
        self.step += 1
        self.sampler_counter += 1
        return {"skipped": False, "step": self.step, "loss": loss, "joint_gradient_norm": norm,
                "embed_rows": int(embed_rows.numel()), "head_rows": self.code_rows, "allocated_music_rows": self.rows, "host_tables_updated": True}

    def save(self, out):
        import torch
        if out.exists():
            raise Error("D3c checkpoint output exists")
        out.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".d3c-checkpoint-", dir=out.parent))
        state = {"schema": SCHEMA, "identity": self.identity, "options": self.options, "ownership": self.ownership,
                 "rows": self.rows, "code_rows": self.code_rows, "frozen_ledger_sha256": self.frozen_ledger_sha,
                 "frozen_reserved_sha256": self.frozen_reserved_sha, "embed": self.embed_master, "head": self.head_master,
                 "embed_optimizer": self.embed_optimizer.state(), "head_optimizer": self.head_optimizer.state(),
                 "lora": {name: value.detach().cpu() for name, value in self.census.items()}, "lora_optimizer": self.lora_optimizer.state_dict(),
                 "step": self.step, "sampler_counter": self.sampler_counter, "skips": self.skips, "total_steps": self.total_steps,
                 "rng_cpu": torch.get_rng_state(), "rng_python": random.getstate(),
                 "rng_cuda": torch.cuda.get_rng_state_all() if self.model.device.type == "cuda" else []}
        torch.save(state, stage / "state.pt")
        (stage / "receipt.json").write_bytes(R.canonical({"schema": SCHEMA, "state_sha256": R.sha(stage / "state.pt"),
                                                         "identity_sha256": R.digest(self.identity), "step": self.step, "complete": True}) + b"\n")
        stage.rename(out)

    def resume(self, path):
        import torch
        receipt = T.read_json(path / "receipt.json")
        if receipt.get("complete") is not True or R.sha(path / "state.pt") != receipt.get("state_sha256"):
            raise Error("D3c checkpoint incomplete/changed")
        state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
        if (state["schema"] != SCHEMA or state["identity"] != self.identity or state["options"] != self.options or
                state["ownership"] != self.ownership or state["rows"] != self.rows or state["code_rows"] != self.code_rows or state["total_steps"] != self.total_steps or
                state["frozen_reserved_sha256"] != self.frozen_reserved_sha or
                state["frozen_ledger_sha256"] != self.frozen_ledger_sha):
            raise Error("D3c exact resume refused changed base/cache/optimizer/band inventory")
        with torch.no_grad():
            self.embed_master.copy_(state["embed"])
            self.head_master.copy_(state["head"])
            for name, tensor in self.census.items():
                tensor.copy_(state["lora"][name].to(tensor.device, tensor.dtype))
        for optimizer, name in ((self.embed_optimizer, "embed_optimizer"), (self.head_optimizer, "head_optimizer")):
            for key in ("m", "v", "steps"):
                getattr(optimizer, key).copy_(state[name][key])
        self.lora_optimizer.load_state_dict(state["lora_optimizer"])
        self.step, self.sampler_counter, self.skips = state["step"], state["sampler_counter"], state["skips"]
        torch.set_rng_state(state["rng_cpu"])
        random.setstate(state["rng_python"])
        if state["rng_cuda"]:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
