#!/usr/bin/env python3
"""Full-parameter streamed KD trainer for the Chimera Omni 27B student (no LoRA, no adapters).

WHY THIS FILE EXISTS.  Cyril, 2026-10-17: "[Cyril quote withheld]", and the
student is distilled from the ORIGINAL parents (Kimi K3 primary, Qwen 3.8 27B a small weight), initialised from the folds.
`streamed_kd_trainer.py` 1.3.4 (the D2 engine) trains only LoRA sites.  This file trains EVERY text weight of the
student -- embeddings, every decoder-layer tensor, the final norm and the head -- with the same streamed engine, the same
teacher-cache readers, the same data configs and the same objective, all imported from that file unchanged.  Rulings:
`cyril-full-distill-no-lora-2026-10-17`, plan `reports/omni-campaign-2026-09/REALIGNMENT_2026-10-17.md`.

THE ALGORITHM.  The 1.3.4 engine runs forward under no_grad one decoder layer at a time (layer inputs saved to host RAM)
and backward in reverse, recomputing each layer with autograd.  Here:
  * The weights live in host RAM as ONE trainable bf16 master copy (`HostWeightStore`, about 47 GB for OMNI1's text
    model), loaded once from the student source (OMNI1D1 through the P1 reader).  The engine streams layers from it.
  * In backward every parameter of the recomputed layer requires grad.  Gradients are accumulated in fp32 (a
    post-accumulate hook moves each bf16 partial -- one per row group / FFN slice -- into an fp32 buffer).
  * The layer's update is applied right there, on the GPU, before the next layer is loaded (fused update), and written
    back to the host master.  Nothing of size "all gradients" or "all optimizer moments" ever exists.
  * The head (text rows) and final norm are trained the same way after the head phase; the embedding is trained by a
    sparse row update over the ids of the step.  A tied head (head == embedding) is updated once with both gradients.
THE OPTIMIZER is Adafactor without momentum (factored second moments for 2-D tensors, full for small ones, row-wise for
the embedding rows), decay beta2_t = 1 - t^-0.8, RMS update clipping, absolute LR with the 1.3.4 warmup/cosine schedule,
bf16 weights updated in fp32 and written back with STOCHASTIC rounding (seeded per step and tensor, so a resume is exact).
Its state is the row and column statistics only (tens of MB, kept on the GPU).
WHAT IS DIFFERENT FROM 1.3.4.  One replica (world 1) and accumulation 1 in this version: the fused update needs each
layer's gradient complete when its backward ends.  A non-finite LOSS skips the whole step before any weight moves (the
window is retried, the 1.3.4 skip guard applies); a non-finite gradient of one tensor skips that tensor and is counted.
Checkpoints hold the full master weights (safetensors shards) + optimizer state + trainer state; the export is a
qwen3_5-layout text model (`student/`) readable by the 1.3.4 SafetensorsSource.  Reassembly into the Chimera layout
with the media blocks is a separate step (the P1 map), as the D2 adapter export was.

Commands
--------
  grad-parity  every full-parameter gradient of the streamed engine vs stock autograd of the same model (fp32, small model)
  probe        timing + memory of N real steps at a real geometry (synthetic weights allowed), lr = 0
  train        the job
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

TOOL_NAME = "streamed_full_kd_trainer"
#: 1.0.1 (2026-10-18): the updated weight is copied straight from the device into the host master (`master.copy_(w)`,
#: same bits) instead of through a fresh pageable CPU tensor; with --pin-weights that is one DMA.  py-spy on run 1 (step
#: 94-100): 46 % of wall time in the pageable layer upload, 27 % in this write-back, 17 % in backward compute.
TOOL_VERSION = "1.0.1"
RECEIPT_SCHEMA = "neural-forge-omni-full-kd-trainer-v1"
CHECKPOINT_SCHEMA = "neural-forge-omni-full-kd-checkpoint-v1"
STUDENT_SCHEMA = "neural-forge-omni-full-kd-student-v1"


def _load_skt():
    path = Path(os.environ.get("STREAMED_KD_TRAINER_PATH", str(Path(__file__).resolve().parent / "streamed_kd_trainer.py")))
    if "streamed_kd_trainer" in sys.modules and getattr(sys.modules["streamed_kd_trainer"], "__file__", None) == str(path):
        return sys.modules["streamed_kd_trainer"]
    spec = importlib.util.spec_from_file_location("streamed_kd_trainer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["streamed_kd_trainer"] = module
    spec.loader.exec_module(module)
    return module


skt = _load_skt()
TrainerError = skt.TrainerError
log = skt.log


# --------------------------------------------------------------------------
# The trainable master weights.
# --------------------------------------------------------------------------


class HostWeightStore:
    """Every text tensor of the student as one host-resident master copy, with the 1.3.4 source surface.

    `get` hands out the LIVE tensor (the engine copies it into its layer buffers, so forward never changes it); the
    optimizer writes updated values back into it.  Only text names are held: the media blocks of a Chimera child are not
    read by the text engine and stay untouched in the base checkpoint.
    """

    def __init__(self, base: Any, *, dtype: torch.dtype, pin: bool = False) -> None:
        self.base = base
        self.text_prefix = getattr(base, "text_prefix", skt.DEFAULT_TEXT_PREFIX)
        self.dtype = dtype
        names = sorted(base.names_under(f"{self.text_prefix}."))
        if base.has("lm_head.weight"):
            names.append("lm_head.weight")
        self.tensors: dict[str, torch.Tensor] = {}
        start = time.perf_counter()
        total = 0
        for i, name in enumerate(names):
            value = base.get(name)
            held = torch.empty(tuple(value.shape), dtype=dtype, pin_memory=bool(pin) and torch.cuda.is_available())
            held.copy_(value)
            self.tensors[name] = held
            total += held.numel() * held.element_size()
            if (i + 1) % 200 == 0:
                log(f"host weights: {i + 1}/{len(names)} tensors, {total / 1e9:.1f} GB, {time.perf_counter() - start:.0f}s")
        self.bytes_held = total
        self.load_seconds = time.perf_counter() - start
        self.bytes_read = 0
        self.loaded_from: dict[str, Any] = {"kind": "base"}
        log(f"host weights: {len(names)} tensors, {total / 1e9:.2f} GB {dtype} held in {self.load_seconds:.0f}s (pinned {bool(pin)})")

    # -- the 1.3.4 source surface --
    def has(self, name: str) -> bool:
        return name in self.tensors

    def get(self, name: str) -> torch.Tensor:
        if name not in self.tensors:
            raise TrainerError(f"{name!r} is not a held text tensor")
        tensor = self.tensors[name]
        self.bytes_read += tensor.numel() * tensor.element_size()
        return tensor

    def names_under(self, prefix: str) -> list[str]:
        return sorted(n for n in self.tensors if n.startswith(prefix))

    def config_dict(self) -> dict[str, Any]:
        return self.base.config_dict()

    def identity(self) -> dict[str, Any]:
        return {"kind": "host_weights", "base": self.base.identity(), "tensors": len(self.tensors),
                "dtype": str(self.dtype).replace("torch.", "")}

    def parameter_count(self) -> int:
        return int(sum(t.numel() for t in self.tensors.values()))

    # -- persistence --
    def save(self, directory: Path, *, shard_bytes: int = 4 << 30, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        from safetensors.torch import save_file

        directory.mkdir(parents=True, exist_ok=True)
        shards: list[list[str]] = [[]]
        size = 0
        for name in sorted(self.tensors):
            nbytes = self.tensors[name].numel() * self.tensors[name].element_size()
            if shards[-1] and size + nbytes > shard_bytes:
                shards.append([])
                size = 0
            shards[-1].append(name)
            size += nbytes
        weight_map: dict[str, str] = {}
        files = []
        for k, group in enumerate(shards):
            filename = f"model-{k + 1:05d}-of-{len(shards):05d}.safetensors"
            save_file({n: self.tensors[n].contiguous() for n in group}, str(directory / filename),
                      metadata={"format": "pt", "tool": TOOL_NAME, "tool_version": TOOL_VERSION})
            for n in group:
                weight_map[n] = filename
            files.append({"file": filename, "tensors": len(group), "bytes": (directory / filename).stat().st_size})
        index = {"metadata": {"total_size": int(sum(t.numel() * t.element_size() for t in self.tensors.values())),
                              **(metadata or {})}, "weight_map": weight_map}
        skt.write_json_atomic(directory / "model.safetensors.index.json", index)
        return {"dir": str(directory), "files": files, "tensors": len(weight_map)}

    def load(self, directory: Path) -> None:
        from safetensors import safe_open

        index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        if set(weight_map) != set(self.tensors):
            missing = sorted(set(self.tensors) - set(weight_map))[:4]
            extra = sorted(set(weight_map) - set(self.tensors))[:4]
            raise TrainerError(f"{directory}: tensor names differ from the student (missing {missing}, extra {extra})")
        by_file: dict[str, list[str]] = {}
        for name, filename in weight_map.items():
            by_file.setdefault(filename, []).append(name)
        for filename, names in sorted(by_file.items()):
            with safe_open(str(directory / filename), framework="pt", device="cpu") as handle:
                for name in names:
                    value = handle.get_tensor(name)
                    if tuple(value.shape) != tuple(self.tensors[name].shape):
                        raise TrainerError(f"{directory}/{filename} {name}: shape {tuple(value.shape)} != {tuple(self.tensors[name].shape)}")
                    self.tensors[name].copy_(value)
        self.loaded_from = {"kind": "checkpoint", "dir": str(directory)}


# --------------------------------------------------------------------------
# The optimizer: Adafactor (no momentum), blockwise, bf16 + stochastic rounding.
# --------------------------------------------------------------------------


def stochastic_round_bf16(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """fp32 -> bf16, rounding the dropped 16 bits up with probability equal to their value (unbiased in expectation)."""

    if x.dtype != torch.float32:
        raise TrainerError("stochastic_round_bf16 takes fp32")
    bits = x.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32, device=x.device, generator=generator)
    rounded = torch.bitwise_and(bits + noise, -65536)
    return rounded.view(torch.float32).to(torch.bfloat16)


class StreamedAdafactor:
    """Per-tensor Adafactor without momentum; the state is the second-moment statistics only."""

    def __init__(self, device: torch.device, *, seed: int, decay: float = -0.8, eps: float = 1e-30, clip: float = 1.0,
                 weight_decay: float = 0.0, block_elements: int = 1 << 25) -> None:
        self.device = device
        self.seed = int(seed)
        self.decay = float(decay)
        self.eps = float(eps)
        self.clip = float(clip)
        self.weight_decay = float(weight_decay)
        self.block_elements = int(block_elements)
        self.state: dict[str, dict[str, Any]] = {}

    def _beta2(self, step: int) -> float:
        return 1.0 - float(step) ** self.decay

    def _generator(self, step: int, name: str) -> torch.Generator:
        g = torch.Generator(device=self.device)
        g.manual_seed(skt.stable_seed(self.seed, "sr", int(step), name))
        return g

    def _round(self, w32: torch.Tensor, out: torch.Tensor, generator: torch.Generator) -> None:
        if out.dtype == torch.bfloat16:
            out.copy_(stochastic_round_bf16(w32, generator))
        else:
            out.copy_(w32.to(out.dtype))

    def _blocks(self, rows: int, cols: int) -> list[slice]:
        per = max(1, self.block_elements // max(1, cols))
        return [slice(a, min(a + per, rows)) for a in range(0, rows, per)]

    def update(self, name: str, weight: torch.Tensor, grad: torch.Tensor, lr: float, *, step: int) -> dict[str, float]:
        """In place: weight (on self.device) -= lr * clipped Adafactor direction of grad (fp32, same shape)."""

        if weight.shape != grad.shape:
            raise TrainerError(f"{name}: weight {tuple(weight.shape)} vs grad {tuple(grad.shape)}")
        beta2 = self._beta2(step)
        generator = self._generator(step, name)
        factored = weight.dim() == 2 and min(weight.shape) > 1
        with torch.no_grad():
            if factored:
                rows, cols = weight.shape
                st = self.state.get(name)
                if st is None:
                    st = {"kind": "factored", "vr": torch.zeros(rows, dtype=torch.float32, device=self.device),
                          "vc": torch.zeros(cols, dtype=torch.float32, device=self.device), "step": 0}
                    self.state[name] = st
                blocks = self._blocks(rows, cols)
                colsum = torch.zeros(cols, dtype=torch.float32, device=self.device)
                for b in blocks:
                    g2 = grad[b].float().square().add_(self.eps)
                    st["vr"][b].mul_(beta2).add_(g2.mean(dim=1), alpha=1.0 - beta2)
                    colsum.add_(g2.sum(dim=0))
                    del g2
                st["vc"].mul_(beta2).add_(colsum / rows, alpha=1.0 - beta2)
                r_mean = st["vr"].mean().clamp_min(self.eps)
                c_factor = st["vc"].rsqrt().unsqueeze(0)

                def direction(b: slice) -> torch.Tensor:
                    return grad[b].float() * (st["vr"][b] / r_mean).rsqrt().unsqueeze(1) * c_factor

                sumsq = torch.zeros((), dtype=torch.float64, device=self.device)
                for b in blocks:
                    sumsq += direction(b).square().sum().double()
                rms = float((sumsq / weight.numel()).sqrt())
                scale = 1.0 / max(1.0, rms / self.clip)
                for b in blocks:
                    w32 = weight[b].float()
                    if self.weight_decay:
                        w32.mul_(1.0 - lr * self.weight_decay)
                    w32.add_(direction(b), alpha=-lr * scale)
                    self._round(w32, weight[b], self._generator(step, f"{name}#{b.start}"))
                    del w32
            else:
                st = self.state.get(name)
                if st is None:
                    st = {"kind": "full", "v": torch.zeros(weight.shape, dtype=torch.float32, device=self.device), "step": 0}
                    self.state[name] = st
                g = grad.float()
                st["v"].mul_(beta2).add_(g.square().add_(self.eps), alpha=1.0 - beta2)
                u = g * st["v"].rsqrt()
                rms = float(u.square().mean().sqrt())
                scale = 1.0 / max(1.0, rms / self.clip)
                w32 = weight.float()
                if self.weight_decay:
                    w32.mul_(1.0 - lr * self.weight_decay)
                w32.add_(u, alpha=-lr * scale)
                self._round(w32, weight, generator)
            st["step"] = int(step)
        return {"rms": rms, "scale": scale}

    def update_rows(self, name: str, rows_weight: torch.Tensor, rows_grad: torch.Tensor, row_ids: torch.Tensor,
                    total_rows: int, lr: float, *, step: int) -> dict[str, float]:
        """Sparse row-wise Adafactor for the embedding: only the rows of this step move; one statistic per row."""

        beta2 = self._beta2(step)
        with torch.no_grad():
            st = self.state.get(name)
            if st is None:
                st = {"kind": "rows", "vrow": torch.zeros(int(total_rows), dtype=torch.float32, device=self.device), "step": 0}
                self.state[name] = st
            ids = row_ids.to(self.device)
            g2 = rows_grad.float().square().add_(self.eps).mean(dim=1)
            st["vrow"][ids] = st["vrow"][ids] * beta2 + g2 * (1.0 - beta2)
            u = rows_grad.float() * st["vrow"][ids].rsqrt().unsqueeze(1)
            rms = float(u.square().mean().sqrt()) if u.numel() else 0.0
            scale = 1.0 / max(1.0, rms / self.clip)
            w32 = rows_weight.float()
            if self.weight_decay:
                w32.mul_(1.0 - lr * self.weight_decay)
            w32.add_(u, alpha=-lr * scale)
            self._round(w32, rows_weight, self._generator(step, name))
            st["step"] = int(step)
        return {"rms": rms, "scale": scale}

    def state_dict(self) -> dict[str, Any]:
        out = {}
        for name, st in self.state.items():
            #: always a copy: on a CPU device .cpu() would alias the live statistics
            out[name] = {k: (v.detach().to("cpu", copy=True) if isinstance(v, torch.Tensor) else v) for k, v in st.items()}
        return {"schema": "full-kd-adafactor-v1", "seed": self.seed, "decay": self.decay, "eps": self.eps, "clip": self.clip,
                "weight_decay": self.weight_decay, "state": out}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("schema") != "full-kd-adafactor-v1":
            raise TrainerError("optimizer state schema differs")
        self.state = {}
        for name, st in payload["state"].items():
            self.state[name] = {k: (v.to(self.device, copy=True) if isinstance(v, torch.Tensor) else v) for k, v in st.items()}

    def state_bytes(self) -> int:
        return int(sum(v.numel() * v.element_size() for st in self.state.values() for v in st.values() if isinstance(v, torch.Tensor)))


# --------------------------------------------------------------------------
# The engine: the 1.3.4 streamed engine with every weight trainable.
# --------------------------------------------------------------------------


class FullParamEngine(skt.LayerwiseEngine):
    """`forward_backward` = forward, head phase (+ head/norm update), reversed fused layer updates, embedding update."""

    def __init__(self, store: HostWeightStore, config: Any, *, optimizer: StreamedAdafactor | None, dtype: torch.dtype,
                 device: str, text_vocab_rows: int, pin_boundaries: bool = True, prefetch: bool = True,
                 ffn_chunk_tokens: int = 0, recompute_tokens: int = 0) -> None:
        super().__init__(store, config, mode="streamed", dtype=dtype, layer_devices=[device], head_device=device, bank=None,
                         text_vocab_rows=text_vocab_rows, pin_boundaries=pin_boundaries, prefetch=prefetch,
                         ffn_chunk_tokens=ffn_chunk_tokens, recompute_tokens=recompute_tokens)
        self.store = store
        self.optimizer = optimizer
        self.device = skt.normalize_device(device)
        prefix = store.text_prefix
        self.embed_name = f"{prefix}.embed_tokens.weight"
        self.norm_name = f"{prefix}.norm.weight"
        self.head_name = skt.head_weight_name(store)
        self.tied = self.head_name == self.embed_name
        self.lr = 0.0
        self.step_index = 1
        #: capture mode (grad-parity): gradients are recorded (fp64, host) and no weight moves
        self.capture: dict[str, torch.Tensor] | None = None
        self.head_weight.requires_grad_(True)
        self._head_acc: torch.Tensor | None = None
        self.head_weight.register_post_accumulate_grad_hook(self._head_hook)
        self.norm.weight.requires_grad_(True)
        self.step_stats: dict[str, Any] = {}
        self.skipped = False

    # -- gradient plumbing --
    def _head_hook(self, param: torch.Tensor) -> None:
        if self._head_acc is None:
            self._head_acc = param.grad.float()
        else:
            self._head_acc.add_(param.grad)
        param.grad = None

    def _finite(self, name: str, grad: torch.Tensor) -> bool:
        #: one reduction, no full-size temporary (the head gradient is 1.27e9 elements): a NaN or inf anywhere makes the
        #: norm non-finite, so the norm is also the finiteness check
        norm = float(torch.linalg.vector_norm(grad))
        if math.isfinite(norm):
            self.step_stats["grad_sq"] += norm * norm
            return True
        self.step_stats["nonfinite_tensors"].append(name)
        return False

    def _record(self, name: str, grad: torch.Tensor) -> None:
        held = self.capture.get(name)
        value = grad.detach().to("cpu", torch.float64)
        self.capture[name] = value if held is None else held + value

    def _apply_dense(self, name: str, weight: torch.Tensor, grad: torch.Tensor, master: torch.Tensor) -> None:
        if self.capture is not None:
            self._record(name, grad)
            return
        if not self._finite(name, grad):
            return
        info = self.optimizer.update(name, weight, grad, self.lr, step=self.step_index)
        self.step_stats["clipped"] += int(info["scale"] < 1.0)
        self.step_stats["updated"] += 1
        master.copy_(weight.detach())  # device -> host master directly (1.0.1)

    def _layer_backward(self, layer: torch.nn.Module, index: int, boundary: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
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
            grad_in = super()._layer_backward(layer, index, boundary, grad)
        finally:
            for h in handles:
                h.remove()
            for p in params.values():
                p.requires_grad_(False)
                p.grad = None
        self._phase_begin(self.device, "update.layer")
        for name, p in params.items():
            full = f"{self.prefix}.{index}.{name}"
            g = acc.pop(name, None)
            if g is None:
                self.step_stats["no_grad"].append(full)
                continue
            self._apply_dense(full, p.data, g, self.store.tensors[full])
            del g
        self._phase_end("update.layer", self.device)
        return grad_in

    def forward_backward(self, ids: torch.Tensor, targets: dict[str, torch.Tensor], objective: Any, *, denominator: float,
                         loss_scale: float = 1.0) -> dict[str, float]:
        self.step_stats = {"grad_sq": 0.0, "nonfinite_tensors": [], "no_grad": [], "clipped": 0, "updated": 0}
        self.skipped = False
        batch, length = ids.shape
        hidden = self.forward_hidden(ids, store_boundaries=True)
        bounds = self._boundary_buffers(batch, length)
        self._phase_begin(self.head_device, "head")
        self._head_acc = None
        self.norm.weight.grad = None
        last = hidden.detach().requires_grad_(True)
        with torch.enable_grad():
            normed = self.norm(last)
        parts, grad_normed = skt.chunked_objective(normed, self.head_weight, targets, objective, denominator=denominator,
                                                   loss_scale=loss_scale, backward=True)
        normed.backward(grad_normed)
        grad = last.grad
        del normed, last, hidden, grad_normed
        self._phase_end("head", self.head_device)
        if not math.isfinite(float(parts.get("total", float("nan")))):
            #: nothing has moved yet: the whole step is skipped (the caller retries the window)
            self.skipped = True
            self._head_acc = None
            self.norm.weight.grad = None
            return parts
        self._phase_begin(self.head_device, "update.head")
        norm_grad = self.norm.weight.grad.float()
        self.norm.weight.grad = None
        with torch.no_grad():
            self._apply_dense(self.norm_name, self.norm.weight.data, norm_grad, self.store.tensors[self.norm_name])
        head_grad = self._head_acc
        self._head_acc = None
        if not self.tied:
            rows = self.head_weight.shape[0]
            with torch.no_grad():
                self._apply_dense(self.head_name, self.head_weight.data, head_grad, self.store.tensors[self.head_name][:rows])
            del head_grad
            head_grad = None
        self._phase_end("update.head", self.head_device)
        self._prefetch(self.num_layers - 1)
        for index in reversed(range(self.num_layers)):
            layer = self.layer(index, index - 1)
            grad = self._layer_backward(layer, index, bounds[index], grad)
        self._phase_begin(self.device, "update.embed")
        self._update_embedding(ids, grad, head_grad)
        self._phase_end("update.embed", self.device)
        del grad, head_grad
        return parts

    def _update_embedding(self, ids: torch.Tensor, grad: torch.Tensor, head_grad: torch.Tensor | None) -> None:
        hidden = grad.shape[-1]
        flat_ids = ids.reshape(-1).to(self.device)
        uniq, inverse = torch.unique(flat_ids, return_inverse=True)
        rows_grad = torch.zeros(uniq.shape[0], hidden, dtype=torch.float32, device=self.device)
        rows_grad.index_add_(0, inverse, grad.reshape(-1, hidden).float())
        master = self.store.tensors[self.embed_name]
        if self.tied:
            #: one dense update of the shared matrix with both gradients (head rows [0, text rows) + the step's embedding rows)
            text = self.head_weight.shape[0]
            dense = head_grad if head_grad is not None else torch.zeros(text, hidden, dtype=torch.float32, device=self.device)
            inside = uniq < text
            dense.index_add_(0, uniq[inside], rows_grad[inside])
            if bool((~inside).any()):
                raise TrainerError("tied head: an embedding id lies outside the text rows")
            if self.capture is not None:
                self._record(self.embed_name, dense)
                return
            with torch.no_grad():
                self._apply_dense(self.embed_name, self.head_weight.data, dense, master[:text])
            return
        if self.capture is not None:
            dense = torch.zeros(master.shape, dtype=torch.float64)
            dense.index_add_(0, uniq.cpu(), rows_grad.double().cpu())
            self._record(self.embed_name, dense)
            return
        if not self._finite(self.embed_name, rows_grad):
            return
        index_cpu = uniq.cpu()
        rows_weight = master.index_select(0, index_cpu).to(self.device)
        self.optimizer.update_rows(self.embed_name, rows_weight, rows_grad, uniq, master.shape[0], self.lr, step=self.step_index)
        master.index_copy_(0, index_cpu, rows_weight.to("cpu"))
        self.step_stats["updated"] += 1


# --------------------------------------------------------------------------
# grad-parity: the streamed full-parameter gradients against stock autograd.
# --------------------------------------------------------------------------


def command_grad_parity(args: argparse.Namespace) -> int:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    kernels = skt.select_linear_attention_kernels(args.kernels)
    kernels["sdpa_gqa"] = skt.select_sdpa_gqa(args.sdpa_gqa)
    dtype = skt.resolve_dtype(args.dtype)
    device = torch.device(args.device)
    base = skt.SafetensorsSource(Path(args.model))
    config = skt.text_config_from_source(base, dtype, args.attn_implementation)
    text_rows = min(args.text_vocab_rows, int(config.vocab_size))
    objective = skt.objective_from_args(args, text_rows)
    ids, targets = skt.synthetic_parity_batch(args.batch, args.seq_len, text_rows, args.k3_top_k, args.q38_top_k, args.seed,
                                              objective.eos_ids, objective)
    denominator = float(args.batch * (args.seq_len - 1))

    log("reference: stock full load + autograd over every text weight")
    model, text, head = skt.load_reference(Path(args.model), dtype, str(device), args.attn_implementation)
    tied = bool(getattr(model.config, "tie_word_embeddings", False)) or model.lm_head.weight.data_ptr() == text.embed_tokens.weight.data_ptr()
    text.requires_grad_(True)
    head = model.lm_head.weight.to(device)
    head.requires_grad_(True)
    rows = min(objective.text_vocab_rows, head.shape[0])
    with torch.enable_grad():
        normed = text(input_ids=ids.to(device), use_cache=False).last_hidden_state
        ref_parts, grad = skt.chunked_objective(normed, head[:rows], targets, objective, denominator=denominator)
        normed.backward(grad)
    prefix = base.text_prefix
    reference: dict[str, torch.Tensor] = {}
    for name, p in text.named_parameters():
        reference[f"{prefix}.{name}"] = (torch.zeros_like(p, dtype=torch.float64, device="cpu") if p.grad is None
                                         else p.grad.detach().to("cpu", torch.float64))
    if not tied:
        reference["lm_head.weight"] = head.grad.detach().to("cpu", torch.float64)
    del model, text, head
    if device.type == "cuda":
        torch.cuda.empty_cache()

    log("candidate: full-parameter streamed engine (capture mode)")
    store = HostWeightStore(base, dtype=dtype)
    engine = FullParamEngine(store, config, optimizer=None, dtype=dtype, device=str(device), text_vocab_rows=text_rows,
                             ffn_chunk_tokens=args.ffn_chunk_tokens, recompute_tokens=args.recompute_tokens)
    engine.capture = {}
    parts = engine.forward_backward(ids, targets, objective, denominator=denominator)
    skt.synchronise(device)
    candidate = engine.capture
    engine.close()
    missing = sorted(set(reference) - set(candidate))
    extra = sorted(set(candidate) - set(reference))
    rows_out, per_site = skt.compare_grads(reference, {n: candidate.get(n, torch.zeros_like(reference[n])) for n in reference})
    worst = max(max(v["max_rel_l2"] for v in per_site.values()), max(v["max_abs_rel"] for v in per_site.values()))
    loss_rel = abs(parts["total"] - ref_parts["total"]) / max(abs(ref_parts["total"]), 1e-30)
    worst = max(worst, loss_rel)
    print(f"loss ref {ref_parts['total']:.9f} streamed {parts['total']:.9f} rel {loss_rel:.3e}; tied head {tied}", flush=True)
    for site, entry in sorted(per_site.items()):
        print(f"  SITE {site:40s} tensors {entry['tensors']:3d} max_rel_l2 {entry['max_rel_l2']:.3e} max_abs_rel {entry['max_abs_rel']:.3e}", flush=True)
    ok = worst <= args.threshold and not missing and not extra
    receipt = {"schema": RECEIPT_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "command": "grad-parity",
               "model": str(args.model), "dtype": args.dtype, "device": str(device), "torch": torch.__version__, "kernels": kernels,
               "batch": args.batch, "seq_len": args.seq_len, "tied_head": tied, "objective": asdict(objective),
               "granularity": {"ffn_chunk_tokens": args.ffn_chunk_tokens, "recompute_tokens": args.recompute_tokens,
                               "head_chunk_positions": objective.head_chunk_positions},
               "tensors_compared": len(reference), "missing": missing, "extra": extra, "loss_ref": ref_parts, "loss_streamed": parts,
               "loss_rel_diff": loss_rel, "per_site": per_site, "per_tensor": rows_out, "worst": worst, "threshold": args.threshold,
               "pass": ok}
    if args.receipt:
        sha = skt.write_json_atomic(Path(args.receipt), receipt)
        log(f"receipt {args.receipt} sha256 {sha}")
    print(f"FULL GRAD-PARITY {'PASS' if ok else 'FAIL'}: {len(reference)} tensors, worst relative difference {worst:.3e} "
          f"(threshold {args.threshold:.1e}); missing {len(missing)} extra {len(extra)}", flush=True)
    return 0 if ok else 1


# --------------------------------------------------------------------------
# probe: real-geometry timing and memory (lr 0, so nothing moves).
# --------------------------------------------------------------------------


def build_base(args: argparse.Namespace) -> Any:
    if getattr(args, "synthetic_config", None):
        layer_types = [t.strip() for t in args.synthetic_layer_types.split(",") if t.strip()]
        return skt.SyntheticSource(Path(args.synthetic_config), intermediate_size=args.synthetic_ffn, layer_types=layer_types)
    return skt.build_source(args)


def command_probe(args: argparse.Namespace) -> int:
    dtype = skt.resolve_dtype(args.dtype)
    device = skt.normalize_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    skt.select_linear_attention_kernels(args.kernels)
    base = build_base(args)
    config = skt.text_config_from_source(base, dtype, args.attn_implementation)
    text_rows = min(args.text_vocab_rows, int(config.vocab_size))
    objective = skt.objective_from_args(args, text_rows)
    store = HostWeightStore(base, dtype=dtype)
    optimizer = StreamedAdafactor(device, seed=args.seed)
    engine = FullParamEngine(store, config, optimizer=optimizer, dtype=dtype, device=str(device), text_vocab_rows=text_rows,
                             pin_boundaries=not args.no_pin, ffn_chunk_tokens=args.ffn_chunk_tokens, recompute_tokens=args.recompute_tokens)
    engine.lr = 0.0
    records = []
    for k in range(args.steps):
        ids, targets = skt.probe_batch(args.batch, args.seq_len, text_rows, objective, args.seed + k)
        engine.step_index = k + 1
        start = time.perf_counter()
        parts = engine.forward_backward(ids, targets, objective, denominator=float(args.batch * (args.seq_len - 1)))
        skt.synchronise(device)
        seconds = time.perf_counter() - start
        rec = {"step": k + 1, "seconds": round(seconds, 3), "tok_per_s": round(args.batch * args.seq_len / seconds, 2),
               "loss": parts.get("total"), "updated_tensors": engine.step_stats["updated"],
               "nonfinite_tensors": len(engine.step_stats["nonfinite_tensors"]), "no_grad": len(engine.step_stats["no_grad"]),
               "grad_norm": math.sqrt(engine.step_stats["grad_sq"]), "layers": engine.num_layers}
        records.append(rec)
        print("PROBE " + json.dumps(rec, sort_keys=True), flush=True)
    report = engine.memory_report()
    receipt = {"schema": RECEIPT_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "command": "probe",
               "source_identity": base.identity(), "layers": engine.num_layers, "batch": args.batch, "seq_len": args.seq_len,
               "dtype": args.dtype, "device": str(device), "parameters_trainable": store.parameter_count(),
               "host_weight_bytes": store.bytes_held, "optimizer_state_bytes": optimizer.state_bytes(), "steps": records,
               "memory_engine": report, "host_rss_gb": skt.host_rss_gb(), "layer_load_seconds": round(engine.load_seconds, 3)}
    engine.close()
    print("MEMORY " + json.dumps({"peak_allocated_gib": report["peak_allocated_gib"], "phases": report["phases_peak_allocated_gib"],
                                  "optimizer_state_mib": round(optimizer.state_bytes() / 2**20, 1)}, sort_keys=True), flush=True)
    if args.receipt:
        sha = skt.write_json_atomic(Path(args.receipt), receipt)
        log(f"receipt {args.receipt} sha256 {sha}")
    return 0


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------


def save_checkpoint(directory: Path, step: int, *, store: HostWeightStore, optimizer: StreamedAdafactor, state: dict[str, Any],
                    keep: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"ckpt-{step:08d}"
    tmp = directory / f".tmp-ckpt-{step:08d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    saved = store.save(tmp / "weights", metadata={"step": str(step)})
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    skt.write_json_atomic(tmp / "trainer_state.json", state)
    skt.write_json_atomic(tmp / "complete.json", {"schema": CHECKPOINT_SCHEMA, "step": step, "weights": saved,
                                                  "optimizer_sha256": skt.file_sha256(tmp / "optimizer.pt"),
                                                  "trainer_state_sha256": skt.file_sha256(tmp / "trainer_state.json")})
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    done = sorted(p for p in directory.glob("ckpt-*") if (p / "complete.json").is_file())
    for old in done[:-max(1, int(keep))]:
        shutil.rmtree(old, ignore_errors=True)
    return final


def checkpoint_problem(candidate: Path) -> str | None:
    marker = candidate / "complete.json"
    if not marker.is_file():
        return "no complete.json"
    info = json.loads(marker.read_text(encoding="utf-8"))
    if info.get("schema") != CHECKPOINT_SCHEMA:
        return f"schema {info.get('schema')!r}"
    if skt.file_sha256(candidate / "optimizer.pt") != info.get("optimizer_sha256"):
        return "optimizer.pt changed"
    if skt.file_sha256(candidate / "trainer_state.json") != info.get("trainer_state_sha256"):
        return "trainer_state.json changed"
    for entry in info["weights"]["files"]:
        path = candidate / "weights" / entry["file"]
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            return f"weights shard {entry['file']} missing or resized"
    return None


def latest_checkpoint(directory: Path) -> Path | None:
    for candidate in sorted(directory.glob("ckpt-*"), reverse=True):
        problem = checkpoint_problem(candidate)
        if problem is None:
            return candidate
        log(f"skipping {candidate}: {problem}")
    return None


def export_student(out: Path, store: HostWeightStore, metadata: dict[str, Any]) -> dict[str, Any]:
    student = out / "student"
    saved = store.save(student, metadata={"tool": TOOL_NAME, "tool_version": TOOL_VERSION})
    skt.write_json_atomic(student / "config.json", store.config_dict())
    manifest = {"schema": STUDENT_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "layout": "qwen3_5 text model "
                "(text tensors only; media blocks are unchanged in the base checkpoint)", "weights": saved, **metadata}
    skt.write_json_atomic(student / "student_manifest.json", manifest)
    return manifest


def command_train(args: argparse.Namespace) -> int:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise TrainerError(f"{TOOL_NAME} {TOOL_VERSION} runs one replica (fused per-layer updates); launch it without torchrun")
    if int(args.accumulation) != 1:
        raise TrainerError("accumulation must be 1 (the update is fused into each layer's backward); raise --micro-batch instead")
    preregistration = skt.preregistration_record(args.preregistration, args.no_preregistration, moves_weights=args.lr > 0)
    cuda_alloc_conf = skt.default_cuda_alloc_conf()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    kernels = skt.select_linear_attention_kernels(args.kernels)
    kernels["sdpa_gqa"] = skt.select_sdpa_gqa(args.sdpa_gqa)
    dtype = skt.resolve_dtype(args.dtype)
    device = skt.normalize_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    base = skt.build_source(args)
    config = skt.text_config_from_source(base, dtype, args.attn_implementation)
    text_rows = min(args.text_vocab_rows, int(config.vocab_size))
    objective = skt.objective_from_args(args, text_rows)
    base_identity = base.identity()
    base_identity_sha = skt.canonical_sha256(base_identity)
    store = HostWeightStore(base, dtype=dtype, pin=args.pin_weights)
    optimizer = StreamedAdafactor(device, seed=args.seed, decay=args.beta2_decay, clip=args.clip_threshold,
                                  weight_decay=args.weight_decay)
    sources, data_identity = skt.load_data_config(Path(args.data_config))
    max_width = int(args.max_width) if args.max_width else int(args.seq_len)
    if max_width < args.seq_len:
        raise TrainerError(f"--max-width {max_width} is below --seq-len {args.seq_len}")

    def check_widths(group: list[Any]) -> None:
        skt.check_source_set(group, seq_len=int(args.seq_len), max_width=int(args.max_width or 0), pad_id=int(args.pad_id), text_rows=text_rows)

    check_widths(sources)
    sampler = skt.MixtureSampler(sources, args.seed)
    tokens_per_step = args.micro_batch * args.seq_len
    total_steps = int(args.total_steps) if args.total_steps else max(1, int(math.ceil(float(args.token_budget) / tokens_per_step)))
    denominator = float(args.micro_batch * (args.seq_len - 1))
    config_digest = skt.canonical_sha256({k: v for k, v in sorted(vars(args).items())
                                          if k not in ("func", "argv_list", "resume", "max_steps_this_run", "receipt", "log_every",
                                                       "allow_config_change", "allow_identity_change", "safety_minutes")})
    data_identity_sha = skt.canonical_sha256(skt.data_content_identity(data_identity))
    out = Path(args.out)
    safety_dir = Path(args.safety_dir)
    step, tokens_seen, history, segments = 0, 0, [], []
    consecutive_skips, skip_window = 0, []
    resumed_from, identity_override = None, None
    if args.resume:
        checkpoint = latest_checkpoint(safety_dir) if args.resume == "auto" else Path(args.resume)
        if checkpoint is not None and args.resume != "auto":
            problem = checkpoint_problem(checkpoint)
            if problem:
                raise TrainerError(f"--resume {checkpoint}: {problem}")
        if checkpoint is not None:
            state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
            if state["config_digest"] != config_digest and not args.allow_config_change:
                raise TrainerError(f"{checkpoint} was written by a different run config")
            was_loaded = {e["name"] for e in ((state.get("data_identity") or {}).get("sources") or [])
                          if e.get("deferred") and e["deferred"].get("loaded", True)}
            for i, s in enumerate(sources):
                if not s.loaded and s.name in was_loaded:
                    skt.load_deferred_source(sources, i, data_identity)
            check_widths(sources)
            data_identity_sha = skt.canonical_sha256(skt.data_content_identity(data_identity))
            changed = {}
            if state.get("base_identity_sha256") != base_identity_sha:
                changed["base"] = {"checkpoint": state.get("base_identity_sha256"), "now": base_identity_sha}
            if state.get("data_identity_sha256") != data_identity_sha:
                changed["data"] = {"checkpoint": state.get("data_identity_sha256"), "now": data_identity_sha}
            if changed:
                if not args.allow_identity_change:
                    raise TrainerError(f"{checkpoint}: {json.dumps(changed, sort_keys=True)} changed; refusing (--allow-identity-change)")
                identity_override = changed
            store.load(checkpoint / "weights")
            optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False))
            sampler.load_state(state["sampler"])
            step, tokens_seen = int(state["step"]), int(state["tokens_seen"])
            history = list(state.get("history_tail", []))
            guard = state.get("skip_guard") or {}
            consecutive_skips, skip_window = int(guard.get("consecutive_skips", 0)), [int(v) for v in guard.get("window", [])]
            segments = list(state.get("segments", []))
            resumed_from = str(checkpoint)
            log(f"resumed from {checkpoint} at step {step}, tokens {tokens_seen:,}")
    engine = FullParamEngine(store, config, optimizer=optimizer, dtype=dtype, device=str(device), text_vocab_rows=text_rows,
                             pin_boundaries=not args.no_pin, prefetch=not args.no_prefetch,
                             ffn_chunk_tokens=args.ffn_chunk_tokens, recompute_tokens=args.recompute_tokens)
    segments.append({"segment": len(segments), "start_step": step, "tool_version": TOOL_VERSION, "resumed_from": resumed_from,
                     "config_digest": config_digest, "base_identity_sha256": base_identity_sha,
                     "data_identity_sha256": data_identity_sha, "identity_override": identity_override, "deferred_loaded": []})
    for e in data_identity["sources"]:
        log(f"source {e['name']}: rows {e.get('rows')}, seq_len {e.get('seq_len')}, supervision {e.get('supervision')}, deferred {e.get('deferred')}")
    log(f"full-parameter KD: {store.parameter_count():,} trainable ({store.bytes_held / 1e9:.1f} GB {args.dtype} on the host); "
        f"steps {step}->{total_steps}; tokens/step {tokens_per_step:,}; lr {args.lr}; tied head {engine.tied}; "
        f"objective w_q38 {objective.w_q38} lambda_k3 {objective.lambda_k3} anchor {objective.k3_q38_anchor} w_ce {objective.w_ce}")
    stop_at = total_steps if not args.max_steps_this_run else min(total_steps, step + int(args.max_steps_this_run))
    halt_reason, deferred_wait = None, None
    receipt_steps: list[dict[str, Any]] = []
    last_safety = time.monotonic()
    started = time.perf_counter()
    attempts = 0

    def trainer_state() -> dict[str, Any]:
        return {"schema": CHECKPOINT_SCHEMA, "tool_version": TOOL_VERSION, "step": step, "tokens_seen": tokens_seen,
                "sampler": sampler.state(), "config_digest": config_digest, "history_tail": history[-50:],
                "base_identity": base_identity, "base_identity_sha256": base_identity_sha, "data_identity": data_identity,
                "data_identity_sha256": data_identity_sha, "segments": segments, "halt_reason": halt_reason,
                "skip_guard": {"consecutive_skips": consecutive_skips, "window": skip_window[-args.skip_window:] if args.skip_window > 0 else []}}

    while step < stop_at:
        due = [i for i, s in enumerate(sources) if not s.loaded and tokens_seen >= (s.defer_until_tokens or 0)]
        if due:
            reasons = [f"{sources[i].name}: {skt.deferred_not_ready(sources[i])}" for i in due if skt.deferred_not_ready(sources[i])]
            if reasons:
                deferred_wait = "; ".join(reasons)
                path = save_checkpoint(safety_dir, step, store=store, optimizer=optimizer, state=trainer_state(), keep=args.safety_keep)
                log(f"DEFERRED at step {step}, tokens {tokens_seen:,}: {deferred_wait}; checkpoint {path}")
                break
            for i in due:
                entry = skt.load_deferred_source(sources, i, data_identity)
                segments[-1]["deferred_loaded"].append({"name": entry["name"], "at_step": step, "at_tokens": tokens_seen})
            check_widths(sources)
            data_identity_sha = skt.canonical_sha256(skt.data_content_identity(data_identity))
        step_start = time.perf_counter()
        attempts += 1
        engine.lr = args.lr * skt.lr_factor(step, args.warmup_steps, total_steps, args.min_lr_ratio)
        engine.step_index = step + 1
        window_start = json.loads(json.dumps(sampler.state()))
        picks = sampler.next_rows(args.micro_batch, tokens_seen)
        rows_by_source: dict[str, int] = {}
        real_tokens = 0
        for s, r in picks:
            rows_by_source[sources[s].name] = rows_by_source.get(sources[s].name, 0) + 1
            real_tokens += sources[s].real_length(r)
        ids, targets = skt.build_batch(sources, picks, objective, pad_id=args.pad_id)
        try:
            parts = engine.forward_backward(ids, targets, objective, denominator=denominator)
        except torch.cuda.OutOfMemoryError:
            print("OOM " + json.dumps({"step": step, "phase": engine.current_phase, "engine": engine.memory_report(),
                                       "batch_width": int(ids.shape[1])}, sort_keys=True, default=str), flush=True)
            raise
        finite = not engine.skipped
        if finite:
            step += 1
            tokens_seen += real_tokens if args.count_real_tokens else tokens_per_step
        else:
            sampler.load_state(window_start)
        consecutive_skips = 0 if finite else consecutive_skips + 1
        if args.skip_window > 0:
            skip_window.append(0 if finite else 1)
            del skip_window[: -args.skip_window]
        halt_reason = skt.skip_guard_reason(consecutive_skips, skip_window, max_consecutive=args.max_consecutive_skips,
                                            max_fraction=args.max_skip_fraction, window_size=args.skip_window)
        stats = engine.step_stats
        if finite and len(stats["nonfinite_tensors"]) > args.max_nonfinite_tensors:
            halt_reason = f"{len(stats['nonfinite_tensors'])} tensors had non-finite gradients in one step (first {stats['nonfinite_tensors'][:3]})"
        seconds = time.perf_counter() - step_start
        record = {"step": step, "attempt": attempts, "tokens": tokens_seen, "loss": parts.get("total"), "q38": parts.get("q38"),
                  "k3": parts.get("k3"), "ce": parts.get("ce"), "eos": parts.get("eos"), "ul": parts.get("ul"),
                  "grad_norm": math.sqrt(stats["grad_sq"]) if finite else None, "applied": finite, "lr": engine.lr,
                  "updated_tensors": stats["updated"], "clipped_tensors": stats["clipped"],
                  "nonfinite_tensors": len(stats["nonfinite_tensors"]), "no_grad_tensors": len(stats["no_grad"]),
                  "seconds": round(seconds, 3), "tok_per_s": round(tokens_per_step / seconds, 2), "real_tokens": real_tokens,
                  "rows_by_source": rows_by_source,
                  "positions": {k: int(parts.get(k, 0)) for k in ("q38_positions", "k3_positions", "ce_positions", "eos_positions", "ul_candidates")}}
        history.append(record)
        receipt_steps.append(record)
        if step % args.log_every == 0 or step == stop_at or halt_reason or not finite:
            print("STEP " + json.dumps(record, sort_keys=True), flush=True)
        if halt_reason:
            path = save_checkpoint(safety_dir, step, store=store, optimizer=optimizer, state=trainer_state(), keep=args.safety_keep)
            log(f"HALT at step {step}: {halt_reason}; checkpoint {path}")
            break
        if not finite:
            continue
        if (args.safety_every_steps and step % args.safety_every_steps == 0) or (time.monotonic() - last_safety >= args.safety_minutes * 60) or step == stop_at:
            path = save_checkpoint(safety_dir, step, store=store, optimizer=optimizer, state=trainer_state(), keep=args.safety_keep)
            log(f"checkpoint {path}")
            last_safety = time.monotonic()
    wall = time.perf_counter() - started
    applied = [r for r in receipt_steps if r["applied"]]
    if halt_reason:
        verdict = "halted"
    elif deferred_wait:
        verdict = "waiting_deferred_source"
    elif receipt_steps and not applied:
        verdict = "no_step_applied"
    else:
        verdict = "completed" if step >= total_steps else "segment_done"
    failed = verdict in ("halted", "no_step_applied")
    exit_code = skt.EXIT_HALTED if failed else (skt.EXIT_DEFERRED if deferred_wait else 0)
    out.mkdir(parents=True, exist_ok=True)
    manifest = None
    if verdict == "completed":
        manifest = export_student(out, store, {"step": step, "tokens_seen": tokens_seen, "base_identity": base_identity,
                                               "base_identity_sha256": base_identity_sha, "data_identity_sha256": data_identity_sha,
                                               "segments": segments})
    receipt = {"schema": RECEIPT_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "command": "train",
               "argv": getattr(args, "argv_list", sys.argv), "verdict": verdict, "halt_reason": halt_reason,
               "deferred_wait": deferred_wait, "exit_code": exit_code, "dtype": args.dtype, "device": str(device),
               "kernels": kernels, "torch": torch.__version__, "config_digest": config_digest, "resumed_from": resumed_from,
               "identity_override": identity_override, "segments": segments, "preregistration": preregistration,
               "base_identity": base_identity, "base_identity_sha256": base_identity_sha, "data_identity": data_identity,
               "data_identity_sha256": data_identity_sha, "objective": asdict(objective),
               "optimizer": {"kind": "adafactor-no-momentum", "decay": args.beta2_decay, "clip": args.clip_threshold,
                             "weight_decay": args.weight_decay, "lr": args.lr, "state_bytes": optimizer.state_bytes(),
                             "stochastic_rounding": dtype == torch.bfloat16},
               "parameters_trainable": store.parameter_count(), "host_weight_bytes": store.bytes_held, "tied_head": engine.tied,
               "total_steps": total_steps, "step": step, "tokens_seen": tokens_seen, "tokens_per_step": tokens_per_step,
               "steps_this_run": len(receipt_steps), "applied_steps": len(applied), "wall_seconds": round(wall, 3),
               "tokens_per_second_this_run": round(len(applied) * tokens_per_step / wall, 3) if wall > 0 else None,
               "steps": receipt_steps, "memory_engine": engine.memory_report(), "host_rss_gb": skt.host_rss_gb(),
               "layer_load_seconds": round(engine.load_seconds, 3), "cuda_alloc_conf": cuda_alloc_conf, "student": manifest}
    receipt_path = Path(args.receipt) if args.receipt else out / "training-receipt.json"
    sha = skt.write_json_atomic(receipt_path, receipt)
    print(f"TRAIN {verdict.upper()} at step {step} ({len(applied)} applied this run), tokens {tokens_seen:,}; receipt {receipt_path} "
          f"sha256 {sha[:16]}" + (f"; student {out / 'student'}" if manifest else ""), flush=True)
    engine.close()
    return exit_code


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def add_objective(p: argparse.ArgumentParser) -> None:
    skt.add_common(p)
    #: K3 is the primary teacher (Cyril 08-04 "99% kimi k3"); Qwen 3.8 27B a small weight.  On K3 rows the K3 term gets
    #: lambda_k3 * (1 - anchor) and the Q38 term lambda_k3 * anchor; on text rows the Q38 term gets w_q38.
    p.set_defaults(w_q38=0.25, lambda_k3=1.0, k3_q38_anchor=0.2, head_chunk_mib=1024.0)


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    top.add_argument("--version", action="store_true")
    sub = top.add_subparsers(dest="command")

    gp = sub.add_parser("grad-parity", help="full-parameter streamed gradients vs stock autograd (small model, fp32)")
    add_objective(gp)
    gp.add_argument("--model", required=True)
    gp.add_argument("--device", default="cuda:0")
    gp.add_argument("--batch", type=int, default=2)
    gp.add_argument("--seq-len", type=int, default=256)
    gp.add_argument("--q38-top-k", type=int, default=64)
    gp.add_argument("--k3-top-k", type=int, default=32)
    gp.add_argument("--threshold", type=float, default=1e-4)
    gp.add_argument("--receipt")
    gp.set_defaults(func=command_grad_parity)

    pr = sub.add_parser("probe", help="timing + memory of N steps at a real geometry, lr 0")
    add_objective(pr)
    pr.add_argument("--model")
    pr.add_argument("--p1-chimera")
    pr.add_argument("--p1-parent-config")
    pr.add_argument("--p1-plan")
    pr.add_argument("--source-plugin")
    pr.add_argument("--source-plugin-args")
    pr.add_argument("--synthetic-config")
    pr.add_argument("--synthetic-ffn", type=int, default=16640)
    pr.add_argument("--synthetic-layer-types", default="linear_attention,linear_attention,linear_attention,full_attention")
    pr.add_argument("--device", default="cuda:0")
    pr.add_argument("--batch", type=int, default=8)
    pr.add_argument("--seq-len", type=int, default=4096)
    pr.add_argument("--steps", type=int, default=2)
    pr.add_argument("--no-pin", action="store_true")
    pr.add_argument("--receipt")
    pr.set_defaults(func=command_probe)

    tr = sub.add_parser("train", help="the full-parameter KD job (one replica)")
    add_objective(tr)
    tr.add_argument("--model")
    tr.add_argument("--p1-chimera")
    tr.add_argument("--p1-parent-config")
    tr.add_argument("--p1-plan")
    tr.add_argument("--source-plugin")
    tr.add_argument("--source-plugin-args")
    tr.add_argument("--device", default="cuda:0")
    tr.add_argument("--data-config", required=True)
    tr.add_argument("--seq-len", type=int, required=True)
    tr.add_argument("--micro-batch", type=int, default=8)
    tr.add_argument("--accumulation", type=int, default=1)
    tr.add_argument("--max-width", type=int, default=0)
    tr.add_argument("--pad-id", type=int, default=skt.DEFAULT_PAD_ID)
    tr.add_argument("--token-budget", type=float, default=0)
    tr.add_argument("--total-steps", type=int, default=0)
    tr.add_argument("--max-steps-this-run", type=int, default=0)
    tr.add_argument("--count-real-tokens", action="store_true")
    tr.add_argument("--lr", type=float, default=1e-5)
    tr.add_argument("--weight-decay", type=float, default=0.0)
    tr.add_argument("--warmup-steps", type=int, default=20)
    tr.add_argument("--min-lr-ratio", type=float, default=0.1)
    tr.add_argument("--beta2-decay", type=float, default=-0.8)
    tr.add_argument("--clip-threshold", type=float, default=1.0)
    tr.add_argument("--max-nonfinite-tensors", type=int, default=0, help="halt when more tensors than this had non-finite grads in one step")
    tr.add_argument("--pin-weights", action="store_true", help="page-lock the host master (faster streaming; needs the RAM locked)")
    tr.add_argument("--out", required=True)
    tr.add_argument("--safety-dir", required=True)
    tr.add_argument("--safety-minutes", type=float, default=60.0)
    tr.add_argument("--safety-every-steps", type=int, default=0)
    tr.add_argument("--safety-keep", type=int, default=2)
    tr.add_argument("--resume", default="")
    tr.add_argument("--allow-config-change", action="store_true")
    tr.add_argument("--allow-identity-change", action="store_true")
    tr.add_argument("--max-consecutive-skips", type=int, default=8)
    tr.add_argument("--max-skip-fraction", type=float, default=0.2)
    tr.add_argument("--skip-window", type=int, default=100)
    tr.add_argument("--preregistration")
    tr.add_argument("--no-preregistration", metavar="REASON")
    tr.add_argument("--receipt")
    tr.add_argument("--log-every", type=int, default=1)
    tr.add_argument("--no-pin", action="store_true")
    tr.add_argument("--no-prefetch", action="store_true")
    tr.add_argument("--deterministic", action="store_true")
    tr.set_defaults(func=command_train)
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.argv_list = [sys.argv[0] if sys.argv else TOOL_NAME, *(sys.argv[1:] if argv is None else list(argv))]
    if args.version:
        print(f"{TOOL_NAME} {TOOL_VERSION}")
        return 0
    if not getattr(args, "func", None):
        parser().print_help()
        return 2
    try:
        return int(args.func(args))
    except TrainerError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
