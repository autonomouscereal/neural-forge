#!/usr/bin/env python3
"""Chimera Omni stage M2: EVERY weight trained together -- text KD and media KD interleaved over one master.

WHY THIS FILE EXISTS.  Cyril, 2026-10-17/18: full distillation (no LoRA) of one model with any input and any output, from
the original parents for text (Kimi K3 primary, Qwen 3.8 27B small weight, Qwen3.8-Flash-Next) and from each modality's
teacher.  Run 1 (`streamed_full_kd_trainer.py`) trains every text weight; stage M1 (`media_full_kd_trainer.py`) trains
every weight of the media modules with the trunk frozen.  M2 trains ALL of them in one job, so the trunk learns the
media objectives too and the text objectives keep holding while it does.

ONE MASTER.  The text weights live once in host RAM (`streamed_full_kd_trainer.HostWeightStore`; initialised from the
base child, then from `--init-text` = run 1's final checkpoint / export), streamed by `OmniEngine` (the full trainer's
`FullParamEngine` + a multi-sample layer backward).  The media modules (five charted units, the native media bridge,
the flow head) are fp32 masters on the GPU (from the base child, then `--init-media` = M1's export).  One streamed
Adafactor owns every tensor.

TWO STEPS, interleaved by token share (`--media-fraction`):
  text   exactly run 1's step: `FullParamEngine.forward_backward` on a text micro-batch (K3 / same-tokenizer teacher / CE),
         with every text weight's update fused into its layer's backward.
  media  a layer-major group of native media samples (M1's binders: ACE / QIMG / H3 flow, speech out flow, speech in CE):
         forward 0..L-1 with host boundaries (units after layer d-1), head losses, then backward L-1..0 where each layer
         is recomputed for every sample, its fp32 parameter gradients summed over the group, and ONE fused update applied
         (`OmniEngine.layer_backward_multi`); the units are walked back between layer d and d-1; the input-level gradient
         trains the bridge and the text embedding rows of the group's prompts (sparse rows).  The final norm and the text
         head are trained by the text steps only (frozen inside a media step).
`selftest` proves the media step's gradients for EVERY tensor (all trunk layers, the embedding rows, units, bridge, head)
against one plain autograd graph of the same tiny Chimera child (resident layers, every parameter a leaf).

Commands
--------
  selftest   tiny Chimera child (CPU fp32): media-step gradients of every tensor == autograd; joint text + media steps run
  train      the job (resumable; checkpoint = text master + media tensors + optimizer + both samplers)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

TOOL_NAME = "omni_full_kd_trainer"
TOOL_VERSION = "1.1.0"
RECEIPT_SCHEMA = "neural-forge-omni-m2-v1"
HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path) -> Any:
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(path):
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mk = _load("media_full_kd_trainer", HERE / "media_full_kd_trainer.py")
d3 = mk.d3
fk = mk.fk
skt = fk.skt


class OmniError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def tool_identity() -> dict[str, Any]:
    return {"name": TOOL_NAME, "version": TOOL_VERSION, "path": str(Path(__file__).resolve()),
            "sha256": mk.file_sha256(Path(__file__).resolve()), "media": mk.tool_identity()}


# --------------------------------------------------------------------------
# The engine: FullParamEngine + a multi-sample fused layer backward.
# --------------------------------------------------------------------------


class OmniEngine(fk.FullParamEngine):
    def layer_backward_multi(self, layer: torch.nn.Module, index: int, boundaries: list[torch.Tensor],
                             grads: list[torch.Tensor]) -> list[torch.Tensor]:
        """Layer `index` recomputed for every sample (each with its own shape / context); the parameter gradients are
        summed in fp32 over the samples, then ONE fused update is applied (or recorded in capture mode)."""

        params = dict(layer.named_parameters())
        acc: dict[str, torch.Tensor] = {}
        handles = []
        for name, p in params.items():
            p.requires_grad_(True)
            p.grad = None

            def hook(param: torch.Tensor, name: str = name) -> None:
                held = acc.get(name)
                if held is None:
                    acc[name] = param.grad.float()
                else:
                    held.add_(param.grad)
                param.grad = None

            handles.append(p.register_post_accumulate_grad_hook(hook))
        try:
            outs = [skt.LayerwiseEngine._layer_backward(self, layer, index, b, g) for b, g in zip(boundaries, grads)]
        finally:
            for h in handles:
                h.remove()
            for p in params.values():
                p.requires_grad_(False)
                p.grad = None
        for name, p in params.items():
            full = f"{self.prefix}.{index}.{name}"
            g = acc.pop(name, None)
            if g is None:
                self.step_stats["no_grad"].append(full)
                continue
            self._apply_dense(full, p.data, g, self.store.tensors[full])
            del g
        return outs


def fresh_stats() -> dict[str, Any]:
    return {"grad_sq": 0.0, "nonfinite_tensors": [], "no_grad": [], "clipped": 0, "updated": 0}


# --------------------------------------------------------------------------
# The model: the hybrid's media modules bound to the full engine.
# --------------------------------------------------------------------------


class OmniModel:
    def __init__(self, hybrid: Any, store: Any, engine: OmniEngine, *, train_bridge: bool) -> None:
        self.hybrid = hybrid
        self.store = store
        self.engine = engine
        old = hybrid.engine
        hybrid.engine = engine
        hybrid.final_norm = engine.norm
        del old
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.depth = sorted({p.after_text_layer for p in hybrid.layout})[0] + 1
        self.media = mk.media_trainables(hybrid)  # units + flow head, fp32, requires_grad; hybrid.emission -> fp32 head
        if train_bridge:
            hybrid.bridge.float()
            for name, p in hybrid.bridge.named_parameters():
                p.requires_grad_(True)
                self.media[f"native_media_bridge.{name}"] = p
        self.train_bridge = bool(train_bridge)
        #: D3c: the written C18 code rows of the head and the embedding (media_full_kd_trainer.MusicBand), when a
        #: music_codes route is configured
        self.music = None

    def enable_music(self, codes: int, *, tokenizer: str, node_defaults: dict[str, Any]) -> None:
        self.music = mk.MusicBand(self.hybrid, codes=codes)
        mk.EXTRAS.update({"music": self.music, "tokenizer": tokenizer, "ace_node_defaults": node_defaults})

    def saved(self) -> dict[str, torch.Tensor]:
        return {**self.media, **(self.music.tensors() if self.music is not None else {})}

    def media_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.media.values()))


def build_model(args: argparse.Namespace, *, device: str, dtype: torch.dtype, optimizer: Any) -> OmniModel:
    options = d3.HybridOptions(dtype=dtype, device=device, head_device="cpu", kernels=args.kernels, lora=None,
                               plan=args.plan, pin_boundaries=False, prefetch=False)
    hybrid = d3.HybridD3(Path(args.omni1), Path(args.parent_config), options)
    store = fk.HostWeightStore(hybrid.source, dtype=dtype, pin=bool(getattr(args, "pin_weights", False)))
    if getattr(args, "init_text", None):
        store.load(Path(args.init_text))
        log(f"text master initialised from {args.init_text}")
    text_rows = min(int(getattr(args, "text_vocab_rows", skt.TEXT_VOCAB_ROWS)), int(hybrid.text_config.vocab_size))
    engine = OmniEngine(store, hybrid.text_config, optimizer=optimizer, dtype=dtype, device=device, text_vocab_rows=text_rows,
                        pin_boundaries=not getattr(args, "no_pin", False), prefetch=not getattr(args, "no_prefetch", False))
    model = OmniModel(hybrid, store, engine, train_bridge=not getattr(args, "freeze_bridge", False))
    if getattr(args, "init_media", None):
        from safetensors.torch import load_file

        values = load_file(str(Path(args.init_media)))
        model.pending_media_init = values
        loaded = 0
        with torch.no_grad():
            for name, value in values.items():
                if name in model.media:
                    model.media[name].copy_(value.to(model.media[name].device, model.media[name].dtype))
                    loaded += 1
        log(f"media modules initialised from {args.init_media}: {loaded} of {len(values)} tensors (music rows apply once the band is enabled)")
    return model


# --------------------------------------------------------------------------
# The media step (every weight).
# --------------------------------------------------------------------------


def media_step(model: OmniModel, samples: list[dict[str, Any]], *, lr: float, step_index: int, offload: str = "cpu") -> dict[str, Any]:
    hybrid, engine = model.hybrid, model.engine
    L, depth = engine.num_layers, model.depth
    engine.lr = float(lr)
    engine.step_index = int(step_index)
    engine.step_stats = fresh_stats()
    head_req, norm_req = engine.head_weight.requires_grad, engine.norm.weight.requires_grad
    engine.head_weight.requires_grad_(False)
    engine.norm.weight.requires_grad_(False)
    try:
        contexts, x0s, xs = [], [], []
        for s in samples:
            c = hybrid.prepare(s["segment"], media_grad=model.train_bridge)
            contexts.append(c)
            x0s.append(c["x"])
            xs.append(c["x"].detach())
        bounds: list[list[Any]] = [[None] * L for _ in samples]
        x_d: list[Any] = [None] * len(samples)
        with torch.no_grad():
            for i in range(L):
                layer = engine.layer(i, i + 1 if i + 1 < L else None)
                for j in range(len(samples)):
                    bounds[j][i] = xs[j].detach().to(offload, copy=True)
                    xs[j] = engine._run(layer, i, xs[j], engine._context(1, int(xs[j].shape[1])))
                if i == depth - 1:
                    for j in range(len(samples)):
                        x_d[j] = xs[j].detach()
                        with mk.scope_of(samples[j]):
                            xs[j] = hybrid.apply_units(xs[j], i, contexts[j])
        grads, losses = [], []
        for j, s in enumerate(samples):
            last = xs[j].detach().requires_grad_(True)
            with torch.enable_grad():
                normed = engine.norm(last)
                if s.get("head_loss") is not None:
                    loss = s["head_loss"](normed, contexts[j])
                else:
                    loss = s["loss_fn"](mk.emission(hybrid, normed, contexts[j]))
            if not bool(torch.isfinite(loss)):
                raise OmniError(f"media sample {s.get('name')}: loss is not finite ({float(loss)})")
            (loss * float(s["weight"])).backward()
            grads.append(last.grad)
            losses.append(float(loss.detach()))
            del last, normed, loss
        xs = None
        for i in reversed(range(L)):
            if i == depth - 1:  # the units sit between layer d-1's output and layer d's input
                for j in range(len(samples)):
                    y = x_d[j].detach().requires_grad_(True)
                    with torch.enable_grad(), mk.scope_of(samples[j]):
                        z = hybrid.apply_units(y, i, contexts[j])
                    z.backward(grads[j])
                    grads[j] = y.grad
                    del y, z
            layer = engine.layer(i, i - 1 if i > 0 else None)
            grads = engine.layer_backward_multi(layer, i, [bounds[j][i] for j in range(len(samples))], grads)
            for j in range(len(samples)):
                bounds[j][i] = None
        # the input level: the bridge (graph kept by prepare) and the text embedding rows of the prompts
        hidden = int(grads[0].shape[-1])
        ids_all, grad_all = [], []
        for j, s in enumerate(samples):
            if x0s[j].requires_grad:
                x0s[j].backward(grads[j])
            ids = s["segment"]["ids"].reshape(-1)
            text = ids < int(hybrid.base_vocab)
            ids_all.append(ids[text])
            grad_all.append(grads[j].reshape(-1, hidden)[text.to(grads[j].device)])
        ids_cat = torch.cat(ids_all).view(1, -1)
        if ids_cat.numel():
            engine._update_embedding(ids_cat, torch.cat(grad_all).view(1, -1, hidden), None)
        # the music band's code rows (D3c samples carry an input_grad callback; the band is not in the text master)
        for j, s in enumerate(samples):
            if s.get("input_grad") is not None:
                s["input_grad"](grads[j])
        if model.music is not None:
            if engine.capture is not None:
                if model.music.head.grad is not None:
                    engine._record("music.head", model.music.head.grad)
                    model.music.head.grad = None
                for r, v in model.music.row_grads.items():
                    engine._record(f"music.embed.row{r}", v)
                model.music.row_grads = {}
            else:
                mstats = {"grad_sq": 0.0, "updated": 0, "nonfinite": []}
                model.music.apply(engine.optimizer, engine.lr, engine.step_index, mstats)
                engine.step_stats["grad_sq"] += mstats["grad_sq"]
                engine.step_stats["updated"] += mstats["updated"]
                engine.step_stats["nonfinite_tensors"].extend(mstats["nonfinite"])
        # the media modules
        for name, p in model.media.items():
            g = p.grad
            if g is None:
                engine.step_stats["no_grad"].append(name)
                continue
            if engine.capture is not None:
                engine._record(name, g)
            else:
                g32 = g.float()
                if engine._finite(name, g32):
                    info = engine.optimizer.update(name, p.data, g32, engine.lr, step=engine.step_index)
                    engine.step_stats["clipped"] += int(info["scale"] < 1.0)
                    engine.step_stats["updated"] += 1
            p.grad = None
    finally:
        engine.head_weight.requires_grad_(head_req)
        engine.norm.weight.requires_grad_(norm_req)
    return {"losses": losses, "stats": engine.step_stats}


# --------------------------------------------------------------------------
# selftest: every gradient of the media step against one autograd graph.
# --------------------------------------------------------------------------


def command_selftest(args: argparse.Namespace) -> int:
    import tempfile

    torch.manual_seed(0)
    refusals: list[str] = []
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "kind": "selftest", "tool": tool_identity(), "torch": torch.__version__}
    with tempfile.TemporaryDirectory() as tmp:
        built = d3.build_tiny_child(Path(tmp) / "child")
        cfg = built["cfg"]
        opts = argparse.Namespace(omni1=str(built["dir"]), parent_config=str(built["parent_config"]), plan=None, kernels="torch",
                                  no_pin=True, no_prefetch=True, freeze_bridge=False)
        optimizer = fk.StreamedAdafactor(torch.device("cpu"), seed=3)
        model = build_model(opts, device="cpu", dtype=torch.float32, optimizer=optimizer)
        hybrid = model.hybrid
        with torch.no_grad():  # a non-trivial head and bridge so every gradient is exercised
            g = torch.Generator().manual_seed(4)
            for name, p in model.media.items():
                if name.startswith(("native_media_head.", "native_media_bridge.")):
                    p.add_(torch.randn(p.shape, generator=g) * 0.01)
        # the reference: the same child with RESIDENT layers, every parameter a leaf, one plain autograd graph per sample
        ref = d3.HybridD3(built["dir"], built["parent_config"],
                          d3.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch", lora=None, mode="resident", pin_boundaries=False))
        ref_media = mk.media_trainables(ref)
        ref.bridge.float()
        for name, p in ref.bridge.named_parameters():
            p.requires_grad_(True)
            ref_media[f"native_media_bridge.{name}"] = p
        with torch.no_grad():
            for name, p in ref_media.items():
                p.copy_(model.media[name])
        ref.final_norm = ref.engine.norm
        ref.engine.embed_weight = ref.engine.embed_weight.detach().clone().requires_grad_(True)
        for layer in ref.engine.layers:
            layer.requires_grad_(True)

        def flow_samples(h: Any, seeds: list[int]) -> list[dict[str, Any]]:
            out = []
            for k, seed in enumerate(seeds):
                kind = ("ace", "h3", "hy3d")[k % 3]
                seg = d3.tiny_flow_batch(cfg, kind, rows=5 + k, seed=seed)
                ctx = h.prepare(seg)
                mask = torch.ones(1, seg["media_rows"])
                loss_fn = d3.make_flow_loss("guided_velocity", ctx, seg, seg["raw"], mask, h.bands, target_kind="parent_raw",
                                            noisy_feature_bands=h.noisy_feature_bands(), signs=seg["signs"])
                out.append({"name": f"tiny-{kind}-{seed}", "segment": seg, "loss_fn": loss_fn})
            return out

        def all_samples(h: Any) -> list[dict[str, Any]]:
            samples = flow_samples(h, [51, 52, 53]) + [mk.tiny_speech(h, cfg, "speech_flow", 54), mk.tiny_speech(h, cfg, "speech_ce", 55)]
            for s in samples:
                s["weight"] = 1.0 / len(samples)
            return samples

        mine = all_samples(hybrid)
        model.engine.capture = {}
        out = media_step(model, mine, lr=0.0, step_index=1)
        got = dict(model.engine.capture)
        model.engine.capture = None
        # reference grads
        theirs = all_samples(ref)
        ref_losses = []
        for s in theirs:
            c = ref.prepare(s["segment"], media_grad=True)
            x = c["x"]
            ectx = ref.contexts(x)
            depth = ref.layout[0].depth
            with torch.enable_grad():
                for i in range(ref.num_layers):
                    x = ref.engine._run(ref.engine.layers[i], i, x, ectx)
                    if i == depth - 1:
                        with mk.scope_of(s):
                            x = ref.apply_units(x, i, c)
                normed = ref.final_norm(x)
                loss = s["head_loss"](normed, c) if s.get("head_loss") is not None else s["loss_fn"](mk.emission(ref, normed, c))
            (loss * float(s["weight"])).backward()
            ref_losses.append(float(loss.detach()))
        want: dict[str, torch.Tensor] = {}
        prefix = ref.engine.prefix
        for i, layer in enumerate(ref.engine.layers):
            for name, p in layer.named_parameters():
                if p.grad is not None:
                    want[f"{prefix}.{i}.{name}"] = p.grad.detach().double()
        want[model.engine.embed_name] = ref.engine.embed_weight.grad.detach().double()
        for name, p in ref_media.items():
            if p.grad is not None:
                want[name] = p.grad.detach().double()
        worst, compared, missing, zero = 0.0, 0, [], 0
        for name, b in want.items():
            a = got.get(name)
            if a is None:
                missing.append(name)
                continue
            a = a.double().reshape(b.shape)
            if float(b.norm()) == 0.0:
                zero += 1
                continue
            compared += 1
            worst = max(worst, float((a - b).norm() / b.norm()))
        extra = sorted(set(got) - set(want))
        loss_gap = max(abs(x - y) for x, y in zip(out["losses"], ref_losses))
        trunk = sum(1 for n in want if n.startswith(prefix))
        receipt["media_step_parity"] = {"reference_tensors": len(want), "trunk_tensors": trunk, "compared": compared,
                                        "zero_reference": zero, "missing": missing[:10], "extra": extra[:10], "worst_rel": worst,
                                        "loss_gap": loss_gap, "threshold": 1e-5}
        log(f"media-step parity: worst rel {worst:.3e} over {compared} tensors ({trunk} trunk), missing {len(missing)}, "
            f"extra {len(extra)}, zero-reference {zero}, loss gap {loss_gap:.2e}")
        if worst > 1e-5 or missing or loss_gap > 1e-6 or compared < trunk:
            refusals.append(f"media-step parity failed: worst {worst:.3e}, missing {missing[:4]}, extra {extra[:4]}")
        # D3c inside the M2 media step: the music code rows (head + embedding) and every trunk tensor vs autograd
        receipt["music_parity"] = m2_music_parity(model, ref)
        mp = receipt["music_parity"]
        log(f"M2 music parity: head {mp['head_rel']:.3e}, embed rows {mp['embed_rel']:.3e} over {mp['rows']} rows, "
            f"trunk worst {mp['trunk_rel']:.3e} over {mp['trunk']} tensors")
        if max(mp["head_rel"], mp["embed_rel"], mp["trunk_rel"]) > 1e-5 or not mp["rows"] or not mp["trunk"]:
            refusals.append(f"M2 music parity failed: {mp}")
        # joint training on the tiny child: media steps and text steps both reduce their losses
        objective = skt.ObjectiveConfig(w_q38=0.0, lambda_k3=0.0, w_ce=1.0, w_eos=0.0, w_ul=0.0,
                                        text_vocab_rows=int(hybrid.base_vocab), head_chunk_positions=64)
        g = torch.Generator().manual_seed(9)
        ids = torch.randint(0, int(hybrid.base_vocab), (2, 24), generator=g)
        targets = text_targets(ids)
        media_curve, text_curve = [], []
        fixed = all_samples(hybrid)
        for step in range(1, int(args.steps) + 1):
            res = media_step(model, fixed, lr=float(args.lr), step_index=2 * step)
            media_curve.append(sum(res["losses"]) / len(res["losses"]))
            model.engine.lr = float(args.lr)
            model.engine.step_index = 2 * step + 1
            parts = model.engine.forward_backward(ids, targets, objective, denominator=float(ids.numel()))
            text_curve.append(float(parts["total"]))
        receipt["joint_training"] = {"media_first": media_curve[0], "media_last": media_curve[-1],
                                     "text_first": text_curve[0], "text_last": text_curve[-1], "steps": len(media_curve)}
        log(f"joint training: media {media_curve[0]:.4g} -> {media_curve[-1]:.4g}, text {text_curve[0]:.4g} -> {text_curve[-1]:.4g}")
        if not (media_curve[-1] < media_curve[0] * float(args.decrease) and text_curve[-1] < text_curve[0]):
            refusals.append(f"joint training did not reduce both losses: {receipt['joint_training']}")
    receipt.update({"refusals": refusals, "verdict": "PASS" if not refusals else "FAIL"})
    sha = mk.write_json_atomic(Path(args.receipt), receipt)
    log(f"M2 SELFTEST {receipt['verdict']}: {args.receipt} sha256 {sha}")
    return 0 if not refusals else 3


def m2_music_parity(model: OmniModel, ref: Any) -> dict[str, Any]:
    codes = 12
    model.enable_music(codes, tokenizer="", node_defaults={})
    music = model.music
    g = torch.Generator().manual_seed(61)
    samples, specs = [], []
    for k in range(2):
        prompt = torch.randint(0, d3.TINY.VOCAB, (1, 4), generator=g)
        seq = torch.randint(0, codes, (6,), generator=g)
        ids = torch.cat([prompt, (seq + model.hybrid.music_band_begin).view(1, -1)], dim=1)
        lp_full = torch.log_softmax(torch.randn(seq.numel(), codes, generator=g) * 2.0, -1)
        top_lp, top_ids = lp_full.topk(4, dim=-1)
        lse = torch.zeros(seq.numel())
        lp_band = d3.band_teacher_logprobs(top_lp, lse, lse)
        positions = torch.arange(3, 3 + seq.numel())
        samples.append({"name": f"tiny-codes-{k}", "segment": {"ids": ids}, "weight": 0.5,
                        "head_loss": music.head_loss(positions, top_ids, lp_band), "input_grad": music.input_grad(ids)})
        specs.append((ids, positions, top_ids, lp_band))
    model.engine.capture = {}
    media_step(model, samples, lr=0.0, step_index=1)
    got = dict(model.engine.capture)
    model.engine.capture = None
    # the reference: resident layers, band table + head rows as leaves
    for layer in ref.engine.layers:
        for p in layer.parameters():
            p.grad = None
    table = music.embed.detach().clone().requires_grad_(True)
    ref.band_embed_override = table
    head_ref = music.head.detach().clone().requires_grad_(True)
    depth = ref.layout[0].depth
    for ids, positions, top_ids, lp_band in specs:
        c = ref.prepare({"ids": ids})
        x = c["x"]
        ectx = ref.contexts(x)
        with torch.enable_grad():
            for i in range(ref.num_layers):
                x = ref.engine._run(ref.engine.layers[i], i, x, ectx)
                if i == depth - 1:
                    x = ref.apply_units(x, i, c)
            loss = d3.band_kl(ref.final_norm(x)[0, positions], head_ref, top_ids, lp_band)
        (loss * 0.5).backward()
    head_rel = float((got["music.head"].double() - head_ref.grad.double()).norm() / head_ref.grad.double().norm())
    rows = sorted({int(r) - int(ref.base_vocab) for ids, *_ in specs for r in ids[0].tolist() if r >= int(ref.base_vocab)})
    embed_rel = max(float((got[f"music.embed.row{r}"].double() - table.grad[r].double()).norm() / table.grad[r].double().norm().clamp_min(1e-30))
                    for r in rows)
    trunk_rel, trunk = 0.0, 0
    for i, layer in enumerate(ref.engine.layers):
        for name, p in layer.named_parameters():
            if p.grad is None or float(p.grad.norm()) == 0.0:
                continue
            key = f"{ref.engine.prefix}.{i}.{name}"
            trunk += 1
            trunk_rel = max(trunk_rel, float((got[key].double().reshape(p.shape) - p.grad.double()).norm() / p.grad.double().norm()))
    for layer in ref.engine.layers:
        for p in layer.parameters():
            p.grad = None
    ref.band_embed_override = None
    model.music = None
    model.hybrid.band_embed_override = None
    mk.EXTRAS.pop("music", None)
    return {"head_rel": head_rel, "embed_rel": embed_rel, "rows": len(rows), "trunk_rel": trunk_rel, "trunk": trunk}


def text_targets(ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Plain next-token CE targets for a [B, L] batch (the selftest's text step)."""

    b, n = ids.shape
    next_ids = torch.full((b, n), -1, dtype=torch.int64)
    next_ids[:, :-1] = ids[:, 1:]
    ce_w = torch.zeros(b, n)
    ce_w[:, :-1] = 1.0
    return {"next_ids": next_ids.reshape(-1), "ce_w": ce_w.reshape(-1), "eos_w": torch.zeros(b * n)}  # [B*L], as build_batch



# --------------------------------------------------------------------------
# train: run 1's text loop with media groups interleaved by token share.
# --------------------------------------------------------------------------


def save_m2_checkpoint(directory: Path, step: int, *, model: OmniModel, optimizer: Any, state: dict[str, Any], keep: int) -> Path:
    from safetensors.torch import save_file

    path = fk.save_checkpoint(directory, step, store=model.store, optimizer=optimizer, state=state, keep=keep)
    media_path = path / "media.safetensors"
    saved = model.saved()
    save_file({k: v.detach().to("cpu", copy=True).contiguous() for k, v in saved.items()}, str(media_path))
    mk.write_json_atomic(path / "media.json", {"sha256": mk.file_sha256(media_path), "tensors": len(saved)})
    return path


def load_m2_media(path: Path, model: OmniModel) -> None:
    from safetensors.torch import load_file

    record = json.loads((path / "media.json").read_text(encoding="utf-8"))
    if mk.file_sha256(path / "media.safetensors") != record["sha256"]:
        raise OmniError(f"{path}/media.safetensors does not hash like media.json")
    values = load_file(str(path / "media.safetensors"))
    holders = model.saved()
    if set(values) - set(holders):
        raise OmniError(f"{path}: media tensors this run does not train: {sorted(set(values) - set(holders))[:4]}")
    with torch.no_grad():
        for name, p in holders.items():
            if name in values:
                p.copy_(values[name].to(p.device, p.dtype))
    if model.music is not None:
        model.music.check_frozen()


def command_train(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    kernels = skt.select_linear_attention_kernels(args.kernels)
    dtype = skt.resolve_dtype(args.dtype)
    device = skt.normalize_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    optimizer = fk.StreamedAdafactor(device, seed=args.seed, decay=args.beta2_decay, clip=args.clip_threshold,
                                     weight_decay=args.weight_decay)
    model = build_model(args, device=str(device), dtype=dtype, optimizer=optimizer)
    engine, hybrid, store = model.engine, model.hybrid, model.store
    text_rows = int(engine.text_vocab_rows)
    objective = skt.objective_from_args(args, text_rows)
    sources, data_identity = skt.load_data_config(Path(args.data_config))

    def check_widths(group: list[Any]) -> None:
        skt.check_source_set(group, seq_len=int(args.seq_len), max_width=int(args.max_width or 0), pad_id=int(args.pad_id), text_rows=text_rows)

    check_widths(sources)
    sampler = skt.MixtureSampler(sources, args.seed)
    media_spec = json.loads(Path(args.media_data).read_text(encoding="utf-8"))
    flows = mk.FlowSet(media_spec)
    if "music_codes" in flows.routes:
        model.enable_music(d3.ACE_CODES, tokenizer=str(args.parent_config), node_defaults=media_spec["ace_node_defaults"])
        init = getattr(model, "pending_media_init", None) or {}
        if "music.head" in init and "music.embed" in init:
            model.music.load({"music.head": init["music.head"], "music.embed": init["music.embed"]})
            log("music band initialised from --init-media")
    media_sampler = mk.GroupSampler(flows, seed=int(args.seed), held_per_route=int(args.held_per_route),
                                    group_tokens=int(args.group_tokens), max_group=int(args.max_group))
    tokens_per_step = args.micro_batch * args.seq_len
    total_steps = int(args.total_steps) if args.total_steps else max(1, int(math.ceil(float(args.token_budget) / tokens_per_step)))
    denominator = float(args.micro_batch * (args.seq_len - 1))
    share = float(args.media_fraction)
    if not 0.0 <= share < 1.0:
        raise OmniError("--media-fraction must be in [0, 1)")
    ratio = share / (1.0 - share)
    config_digest = skt.canonical_sha256({k: v for k, v in sorted(vars(args).items())
                                          if k not in ("func", "argv_list", "resume", "receipt", "allow_config_change",
                                                       "allow_identity_change", "safety_minutes")})
    data_identity_sha = skt.canonical_sha256(skt.data_content_identity(data_identity))
    out, safety_dir = Path(args.out), Path(args.safety_dir)
    step, tokens_seen, media_tokens, media_steps, history = 0, 0, 0, 0, []
    resumed_from = None
    if args.resume:
        checkpoint = fk.latest_checkpoint(safety_dir) if args.resume == "auto" else Path(args.resume)
        if checkpoint is not None:
            state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
            if state["config_digest"] != config_digest and not args.allow_config_change:
                raise OmniError(f"{checkpoint} was written by a different run config")
            was_loaded = {e["name"] for e in ((state.get("data_identity") or {}).get("sources") or [])
                          if e.get("deferred") and e["deferred"].get("loaded", True)}
            for i, s in enumerate(sources):
                if not s.loaded and s.name in was_loaded:
                    skt.load_deferred_source(sources, i, data_identity)
            check_widths(sources)
            data_identity_sha = skt.canonical_sha256(skt.data_content_identity(data_identity))
            if state.get("data_identity_sha256") != data_identity_sha and not args.allow_identity_change:
                raise OmniError(f"{checkpoint}: the text data identity changed; refusing (--allow-identity-change)")
            store.load(checkpoint / "weights")
            load_m2_media(checkpoint, model)
            optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False))
            sampler.load_state(state["sampler"])
            media_sampler.load_state(state["media_sampler"])
            step, tokens_seen = int(state["step"]), int(state["tokens_seen"])
            media_tokens, media_steps = int(state.get("media_tokens", 0)), int(state.get("media_steps", 0))
            history = list(state.get("history_tail", []))
            resumed_from = str(checkpoint)
            log(f"resumed from {checkpoint} at text step {step}, tokens {tokens_seen:,}, media tokens {media_tokens:,}")
    log(f"M2: {store.parameter_count():,} text + {model.media_parameter_count():,} media trainable; text steps {step}->{total_steps}; "
        f"media share {share}; text lr {args.lr}, media lr {args.media_lr}; routes {flows.identity()}")

    def trainer_state(halt: str | None = None) -> dict[str, Any]:
        return {"schema": fk.CHECKPOINT_SCHEMA, "tool": tool_identity(), "step": step, "tokens_seen": tokens_seen,
                "media_tokens": media_tokens, "media_steps": media_steps, "sampler": sampler.state(),
                "media_sampler": media_sampler.state(), "config_digest": config_digest, "history_tail": history[-50:],
                "data_identity": data_identity, "data_identity_sha256": data_identity_sha, "halt_reason": halt}

    last_safety = time.monotonic()
    halt, deferred_wait = None, None
    while step < total_steps:
        if (out / "STOP").exists():
            log("STOP file present")
            break
        due = [i for i, s in enumerate(sources) if not s.loaded and tokens_seen >= (s.defer_until_tokens or 0)]
        if due:
            reasons = [f"{sources[i].name}: {skt.deferred_not_ready(sources[i])}" for i in due if skt.deferred_not_ready(sources[i])]
            if reasons:
                deferred_wait = "; ".join(reasons)
                path = save_m2_checkpoint(safety_dir, step, model=model, optimizer=optimizer, state=trainer_state(), keep=args.safety_keep)
                log(f"DEFERRED at step {step}: {deferred_wait}; checkpoint {path}")
                break
            for i in due:
                skt.load_deferred_source(sources, i, data_identity)
            check_widths(sources)
            data_identity_sha = skt.canonical_sha256(skt.data_content_identity(data_identity))
        factor = skt.lr_factor(step, args.warmup_steps, total_steps, args.min_lr_ratio)
        t0 = time.perf_counter()
        if share > 0 and media_tokens < ratio * max(tokens_seen, tokens_per_step):
            group = mk.draw_group(media_sampler, hybrid, verify=not args.no_verify)
            result = media_step(model, group, lr=args.media_lr * factor, step_index=step + media_steps + 1, offload=args.offload)
            media_steps += 1
            tokens = sum(s["tokens"] for s in group)
            media_tokens += tokens
            stats = result["stats"]
            record = {"kind": "media", "media_step": media_steps, "route": group[0]["route"], "samples": len(group), "tokens": tokens,
                      "loss": sum(result["losses"]) / len(result["losses"]), "grad_norm": math.sqrt(stats["grad_sq"]),
                      "updated": stats["updated"], "no_grad": len(stats["no_grad"]), "nonfinite": len(stats["nonfinite_tensors"]),
                      "seconds": round(time.perf_counter() - t0, 2)}
        else:
            engine.lr = args.lr * factor
            engine.step_index = step + media_steps + 1
            window = json.loads(json.dumps(sampler.state()))
            picks = sampler.next_rows(args.micro_batch, tokens_seen)
            real_tokens = sum(sources[s].real_length(r) for s, r in picks)
            ids, targets = skt.build_batch(sources, picks, objective, pad_id=args.pad_id)
            parts = engine.forward_backward(ids, targets, objective, denominator=denominator)
            if engine.skipped:
                sampler.load_state(window)
                log(f"text step skipped (non-finite loss) at step {step}")
                continue
            step += 1
            tokens_seen += real_tokens if args.count_real_tokens else tokens_per_step
            stats = engine.step_stats
            record = {"kind": "text", "step": step, "tokens": tokens_seen, "loss": parts.get("total"), "k3": parts.get("k3"),
                      "q38": parts.get("q38"), "ce": parts.get("ce"), "grad_norm": math.sqrt(stats["grad_sq"]),
                      "updated": stats["updated"], "nonfinite": len(stats["nonfinite_tensors"]), "lr": engine.lr,
                      "seconds": round(time.perf_counter() - t0, 2)}
        history.append(record)
        print("STEP " + json.dumps(record, sort_keys=True), flush=True)
        if record["nonfinite"] > args.max_nonfinite_tensors:
            halt = f"{record['nonfinite']} tensors had non-finite gradients in one {record['kind']} step"
            save_m2_checkpoint(safety_dir, step, model=model, optimizer=optimizer, state=trainer_state(halt), keep=args.safety_keep)
            log(f"HALT: {halt}")
            break
        if time.monotonic() - last_safety >= args.safety_minutes * 60 or step >= total_steps:
            path = save_m2_checkpoint(safety_dir, step, model=model, optimizer=optimizer, state=trainer_state(), keep=args.safety_keep)
            log(f"checkpoint {path}")
            last_safety = time.monotonic()
    verdict = "halted" if halt else ("waiting_deferred_source" if deferred_wait else ("completed" if step >= total_steps else "segment_done"))
    exported = None
    if verdict == "completed":
        out.mkdir(parents=True, exist_ok=True)
        exported = {"text": fk.export_student(out, store, {"step": step, "tokens_seen": tokens_seen, "stage": "M2"}),
                    "media": mk.export_media(out / "media", model.saved(), {"stage": "M2", "media_steps": media_steps})}
    receipt = {"schema": RECEIPT_SCHEMA, "kind": "train", "tool": tool_identity(), "argv": sys.argv[1:], "verdict": verdict,
               "halt": halt, "deferred_wait": deferred_wait, "kernels": kernels, "step": step, "tokens_seen": tokens_seen,
               "media_steps": media_steps, "media_tokens": media_tokens, "resumed_from": resumed_from, "history": history[-200:],
               "exported": exported, "seconds": round(time.perf_counter() - started, 1)}
    sha = mk.write_json_atomic(Path(args.receipt) if args.receipt else out / "m2-receipt.json", receipt)
    log(f"M2 TRAIN {verdict.upper()} at text step {step}, media steps {media_steps}; receipt sha256 {sha[:16]}")
    return skt.EXIT_HALTED if halt else (skt.EXIT_DEFERRED if deferred_wait else 0)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--receipt", required=True)
    s.add_argument("--steps", type=int, default=25)
    s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--decrease", type=float, default=0.8)
    t = sub.add_parser("train", help="stage M2: every weight, text + media interleaved")
    fk.add_objective(t)
    t.add_argument("--omni1", required=True, help="the base Chimera child (structure, bridge, band extras)")
    t.add_argument("--parent-config", required=True)
    t.add_argument("--plan", default=None)
    t.add_argument("--init-text", default=None, help="run 1's final weights (a checkpoint weights/ dir or the exported student/)")
    t.add_argument("--init-media", default=None, help="M1's export media_modules.safetensors")
    t.add_argument("--data-config", required=True, help="the full-KD text data config")
    t.add_argument("--media-data", required=True, help="the media data config (media_full_kd_trainer FlowSet)")
    t.add_argument("--media-fraction", type=float, default=0.3, help="target share of media tokens among all tokens")
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--seq-len", type=int, required=True)
    t.add_argument("--micro-batch", type=int, default=6)
    t.add_argument("--max-width", type=int, default=0)
    t.add_argument("--pad-id", type=int, default=skt.DEFAULT_PAD_ID)
    t.add_argument("--token-budget", type=float, default=0)
    t.add_argument("--total-steps", type=int, default=0)
    t.add_argument("--count-real-tokens", action="store_true")
    t.add_argument("--lr", type=float, default=5e-6)
    t.add_argument("--media-lr", type=float, default=1e-5)
    t.add_argument("--weight-decay", type=float, default=0.0)
    t.add_argument("--warmup-steps", type=int, default=20)
    t.add_argument("--min-lr-ratio", type=float, default=0.1)
    t.add_argument("--beta2-decay", type=float, default=-0.8)
    t.add_argument("--clip-threshold", type=float, default=1.0)
    t.add_argument("--max-nonfinite-tensors", type=int, default=0)
    t.add_argument("--group-tokens", type=int, default=8000)
    t.add_argument("--max-group", type=int, default=32)
    t.add_argument("--held-per-route", type=int, default=16)
    t.add_argument("--offload", default="cpu", choices=("cpu", "cuda"))
    t.add_argument("--out", required=True)
    t.add_argument("--safety-dir", required=True)
    t.add_argument("--safety-minutes", type=float, default=60.0)
    t.add_argument("--safety-keep", type=int, default=1)
    t.add_argument("--resume", default="")
    t.add_argument("--allow-config-change", action="store_true")
    t.add_argument("--allow-identity-change", action="store_true")
    t.add_argument("--receipt", default=None)
    t.add_argument("--no-pin", action="store_true")
    t.add_argument("--no-prefetch", action="store_true")
    t.add_argument("--no-verify", action="store_true")
    t.add_argument("--freeze-bridge", action="store_true")
    t.add_argument("--pin-weights", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    log(f"{TOOL_NAME} {TOOL_VERSION} {args.command}")
    try:
        if args.command == "selftest":
            return command_selftest(args)
        if args.command == "train":
            return command_train(args)
    except (OmniError, mk.MediaError, d3.D3Error) as exc:
        log(f"REFUSED: {exc}")
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
