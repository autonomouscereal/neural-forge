#!/usr/bin/env python3
"""Training-throughput probe for the omni campaign's stage-D budgets.

WHY THIS FILE EXISTS.  Every stage-D schedule in the plan rests on a tok/s number
that nobody measured: "~180 tok/s 27B LoRA split", "~550 tok/s 9B LoRA", "~4k
tok/s 0.8B full", "~8,500 tok/s teacher-forward-inclusive".  The one time this
estate quoted a generation rate from a design estimate it was 8.7x optimistic
(NOTES-S2B).  R5 section 4 therefore makes the measurement a precondition:
"Measure BEFORE any budget is quoted."  This file is that measurement and nothing
else -- it trains no model, writes no checkpoint, and grades nothing.  It reports
tokens per second and the memory it took to get them, per cell, in a receipt.

WHAT A CELL IS.  One (model, mode, dtype, sequence length, micro-batch) point on
the hardware, measured twice: forward-only (the teacher-logit rate) and
forward+backward+optimizer-step (the student rate).  Three warmup steps are
discarded -- CUDA context, autotuned Triton kernels and the allocator's first
growth all land in step 0 -- and at least ten steps are timed after that.

WHAT IS HAND-ROLLED AND WHY.  The lab venv has no peft, no bitsandbytes and no
deepspeed (R5 section 0), so LoRA is a plain `nn.Module` wrapper and the
CPU-offloaded optimizer is a plain loop.  That is not a workaround: a measured
rate is only worth something if the thing measured is the thing that will run,
and what will run here is exactly these two hand-rolled objects.

THE LOSS PATH IS PART OF THE MEASUREMENT.  Qwen3.5's vocabulary is 248,320 rows.
At L=4,096 a materialised fp32 logit tensor is 4.07 GB before its gradient, which
is most of a 3090 and does not fit beside a split 27B at all.  So the head is run
in chunks over the sequence: each chunk's logits are built, scored, backwarded
into a detached copy of the hidden state and freed, and the trunk is backwarded
once at the end with the accumulated hidden-state gradient.  The arithmetic is
identical to the one-shot path (`--self-test` checks it against
`F.cross_entropy`) and the peak is one chunk instead of the whole sequence.
`loss_chunks` is in every receipt because it changes the memory story the R5
digest wrote from one-shot arithmetic.

WHAT IS NOT MEASURED HERE, STATED SO NOBODY INFERS IT.  The objective is
token-level cross-entropy on the input ids, not the stage-D KD objective.  The KD
term is a top-k gather plus a KL over the SAME logits, so it costs the same order
and slightly more; the CE number is therefore an upper bound on the step rate of
the real objective, not a promise.  The corpus is uniform random ids: throughput
at a fixed shape does not depend on the token values, but a real corpus brings
packing and padding, which this does not model.

Cells (`--cell`):

  08b-full            Qwen3.5-0.8B, every text parameter trainable, fp32 master
                      weights under bf16 autocast, torch AdamW.  L=1,024/4,096.
  4b-lora             Qwen3.5-4B, base frozen bf16, hand-rolled LoRA r=64 on
                      q/k/v/o, gate/up/down, and the GDN in_proj_qkv/out_proj.
  4b-full-cpuadam     Qwen3.5-4B, every text parameter trainable, bf16 weights and
                      gradients on the card, fp32 master/m/v on the host, the Adam
                      step executed on the host and the result copied back.
  27b-teacher-stream  Qwen3.8-27B-Uncensored, teacher-forced FORWARD ONLY, one
                      decoder layer resident at a time, streamed from the local
                      safetensors.  This is the stage-D teacher-logit engine rate.
  27b-lora-split      Qwen3.8-27B-Uncensored, base fp16 split over two cards by
                      `device_map="sequential"`, hand-rolled LoRA r=64, dynamic
                      loss scaling.  THE stage-D number (decision line D-R5-1).
  stream-verify       The streamed engine's logits against a stock full load of
                      the same checkpoint.  Correctness arm for 27b-teacher-stream;
                      run it on the 0.8B, where a full load is cheap.

Examples
--------
  PY=/home/cereal/model-surgery-lab/venv/bin/python
  D=/home/cereal/omni-campaign/distill
  G4=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G4 \
      $PY $D/tps_probe.py --cell 08b-full --sequence-length 1024 --micro-batch 1
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G4 \
      $PY $D/tps_probe.py --cell 08b-full --sequence-length 4096 --micro-batch auto
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<rtx8000-uuid>,$G4 \
      $PY $D/tps_probe.py --cell 27b-lora-split --max-memory 0=40GiB,1=20GiB
  $PY $D/tps_probe.py --self-test          # CPU only, no card
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "neural-forge-omni-tps-probe-v1"

#: The rungs the plan seals checkpoints at (CAMPAIGN.md 4.2, D-R5-10).  Budgets
#: are reported at these three so the receipt answers the scheduling question
#: without anyone re-doing the division.
RUNG_TOKENS = (10_000_000, 50_000_000, 200_000_000)

#: Qwen3.5 ships 248,320 physical embedding rows of which 248,044 are defined by
#: the tokenizer (PORT_SPEC section 0.3).  Random probe ids stay inside the
#: defined rows so a stray id can never be the reason a loss is odd.
DEFINED_VOCABULARY_ROWS = 248_044

#: LoRA sites.  q/k/v/o and gate/up/down are the usual set; `in_proj_qkv` and
#: `out_proj` are the Gated-DeltaNet equivalents and carry three quarters of the
#: layers in a Qwen3.5 hybrid, so leaving them out would train a quarter of the
#: trunk and call it a trunk update.
LORA_TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.out_proj",
)

TEXT_TRUNK_PREFIX = "model.language_model"

CELLS = (
    "08b-full",
    "4b-lora",
    "4b-full-cpuadam",
    "27b-teacher-stream",
    "27b-lora-split",
    "stream-verify",
)

DEFAULT_MODEL_FOR_CELL = {
    "08b-full": "/home/cereal/models/Qwen3.5-0.8B",
    "4b-lora": "/home/cereal/models/Qwen3.5-4B",
    "4b-full-cpuadam": "/home/cereal/models/Qwen3.5-4B",
    "27b-teacher-stream": "/home/cereal/models/Qwen3.8-27B-Uncensored",
    "27b-lora-split": "/home/cereal/models/Qwen3.8-27B-Uncensored",
    "stream-verify": "/home/cereal/models/Qwen3.5-0.8B",
}


class ProbeError(RuntimeError):
    """A refusal that is the probe's own, never a bare torch traceback."""


# --------------------------------------------------------------------------
# Small shared utilities.  `log` is deliberately the same one-line format the
# calibration lane prints, so two lanes' logs interleave readably.
# --------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def run_command(command: list[str], timeout: int = 60) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except Exception as exc:  # noqa: BLE001 -- absence and breakage read the same
        return 127, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def nvidia_smi_inventory() -> dict[str, Any]:
    """Whole-box GPU state, recorded and never acted on.

    The estate's rule is that no lane stops what it did not start, so this is
    provenance only: it says who else was on the box while the numbers were
    taken, which is the difference between a clean measurement and one that
    shared a card without saying so.
    """

    rc_gpu, out_gpu, err_gpu = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.used,memory.total",
            "--format=csv,noheader",
        ]
    )
    rc_app, out_app, _ = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    return {
        "query_return_code": rc_gpu,
        "gpus": [line.strip() for line in out_gpu.splitlines() if line.strip()],
        "compute_apps": [line.strip() for line in out_app.splitlines() if line.strip()],
        "compute_apps_return_code": rc_app,
        "error": err_gpu.strip()[:400] or None,
    }


def proc_status_kib(field: str) -> int | None:
    """One /proc/self/status field in KiB (`VmRSS`, `VmHWM`, ...)."""

    try:
        text = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    return None


def meminfo_kib(field: str) -> int | None:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    return None


def gigabytes(num_bytes: float | None) -> float | None:
    return None if num_bytes is None else round(float(num_bytes) / (1024.0**3), 4)


def exact_command() -> str:
    """The command that produced this receipt, environment included.

    Reproducing a GPU measurement without `CUDA_DEVICE_ORDER` and
    `CUDA_VISIBLE_DEVICES` is not reproducing it: on this host torch index 6
    without `PCI_BUS_ID` is a P40 (CAMPAIGN.md rule 4), so the pinning belongs
    in the command string, not in a reader's memory.
    """

    keep = (
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TOKENIZERS_PARALLELISM",
    )
    prefix = " ".join(
        f"{name}={shlex.quote(os.environ[name])}" for name in keep if name in os.environ
    )
    body = " ".join(
        shlex.quote(part)
        for part in [sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    )
    return f"{prefix} {body}".strip()


# --------------------------------------------------------------------------
# Device inventory and sampling.
# --------------------------------------------------------------------------


def device_records() -> list[dict[str, Any]]:
    """One record per VISIBLE cuda device: index, name, UUID, capability, size.

    The UUID is the identity that matters.  Two of this box's seven cards are
    3090s and five are not bf16-capable at all; a receipt that says "cuda:0"
    and not which physical card is not a receipt.
    """

    import torch

    if not torch.cuda.is_available():
        return []
    records = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        uuid = getattr(props, "uuid", None)
        records.append(
            {
                "torch_index": index,
                "name": props.name,
                "uuid": None if uuid is None else f"GPU-{uuid}".replace("GPU-GPU-", "GPU-"),
                "compute_architecture": f"sm{props.major}{props.minor}",
                "total_memory_gb": gigabytes(props.total_memory),
                "multi_processor_count": props.multi_processor_count,
            }
        )
    return records


class ResourceSampler:
    """Background peak sampler for device memory and host RSS.

    `torch.cuda.max_memory_allocated` only sees this process.  The peak that
    decides whether the next job fits is the DEVICE-WIDE one, so the sampler
    reads `mem_get_info` -- free/total as the driver sees it -- and keeps the
    minimum free per device beside torch's own counters.
    """

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.min_free_bytes: dict[int, int] = {}
        self.peak_host_rss_kib: int = 0
        self.samples = 0

    def _sample_once(self) -> None:
        import torch

        for index in range(torch.cuda.device_count()):
            try:
                free, _total = torch.cuda.mem_get_info(index)
            except Exception:  # noqa: BLE001 -- a sampler never breaks a run
                continue
            previous = self.min_free_bytes.get(index)
            if previous is None or free < previous:
                self.min_free_bytes[index] = int(free)
        rss = proc_status_kib("VmRSS")
        if rss is not None and rss > self.peak_host_rss_kib:
            self.peak_host_rss_kib = rss
        self.samples += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, name="tps-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._sample_once()

    def report(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "interval_seconds": self.interval,
            "device_minimum_free_gb": {
                str(index): gigabytes(value) for index, value in sorted(self.min_free_bytes.items())
            },
            "peak_host_rss_gb_sampled": gigabytes(self.peak_host_rss_kib * 1024),
        }


def torch_memory_report() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {}
    out: dict[str, Any] = {}
    for index in range(torch.cuda.device_count()):
        out[str(index)] = {
            "peak_allocated_gb": gigabytes(torch.cuda.max_memory_allocated(index)),
            "peak_reserved_gb": gigabytes(torch.cuda.max_memory_reserved(index)),
        }
    return out


def reset_memory_counters() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)


def synchronise_all() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


# --------------------------------------------------------------------------
# Hand-rolled LoRA.  No peft in the venv, and no wish for one: the wrapper has
# to survive being wrapped around a module that accelerate has already hooked
# for a multi-device dispatch, which is the only interesting thing about it.
# --------------------------------------------------------------------------


def make_lora_linear(base_class: Any) -> Any:
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        """`base(x) + B(A(x)) * (alpha / rank)` with the base frozen.

        THE ADAPTERS ARE fp32 AND THAT IS THE POINT.  They are their own fp32
        master copy: autocast casts them to the compute dtype on the way into
        the matmul and the optimizer updates the fp32 tensor, so no separate
        master buffer is needed for the only trainable tensors in a LoRA run.

        `x.to(...)` before the adapter path is not defensive noise.  Under a
        `device_map` split, accelerate's hook moves the base module's input to
        the base module's device inside `self.base(x)`; the adapter sees the
        ORIGINAL `x`, which on a layer boundary is on the previous card.
        """

        def __init__(self, base: Any, rank: int, alpha: float) -> None:
            super().__init__()
            self.base = base
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            device = base.weight.device
            self.lora_a = nn.Parameter(
                torch.empty(rank, base.in_features, dtype=torch.float32, device=device)
            )
            self.lora_b = nn.Parameter(
                torch.zeros(base.out_features, rank, dtype=torch.float32, device=device)
            )
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
            self.rank = int(rank)
            self.scaling = float(alpha) / float(rank)

        def forward(self, x: Any) -> Any:  # noqa: ANN401 -- torch.Tensor
            out = self.base(x)
            h = x.to(self.lora_a.device)
            #: The cast is EXPLICIT rather than left to autocast.  Autocast made
            #: the 27B split unmeasurable: `apply_rotary_pos_emb` promotes q and
            #: k to fp32 (the rope table is forced fp32 by the model's own
            #: `maybe_autocast(enabled=False)`), so with autocast ON the SDPA
            #: call casts them back down and a fused backend runs, and with it
            #: OFF the fp32 math backend runs -- two different saved-tensor sets
            #: for the same layer, which `torch.utils.checkpoint` rejects when
            #: the forward and the recompute disagree.  Casting here lets the
            #: LoRA cells run with no autocast at all, so both passes are the
            #: same function, and the fp32 master property is kept by the cast
            #: being differentiable back into the fp32 parameter.
            weight_a = self.lora_a.to(h.dtype)
            weight_b = self.lora_b.to(h.dtype)
            delta = torch.nn.functional.linear(
                torch.nn.functional.linear(h, weight_a), weight_b
            )
            return out + delta.to(out.dtype) * self.scaling

    return LoRALinear


def attach_lora(model: Any, rank: int, alpha: float, suffixes: tuple[str, ...]) -> dict[str, Any]:
    """Replace every matching `nn.Linear` with a LoRA wrapper.  Returns a census."""

    from torch import nn

    lora_class = make_lora_linear(nn.Linear)
    replaced: list[str] = []
    adapter_parameters = 0
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(name.endswith(suffix) for suffix in suffixes):
            targets.append(name)
    for name in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, child_name)
        wrapper = lora_class(base, rank, alpha)
        setattr(parent, child_name, wrapper)
        replaced.append(name)
        adapter_parameters += wrapper.lora_a.numel() + wrapper.lora_b.numel()
    return {
        "rank": rank,
        "alpha": alpha,
        "target_suffixes": list(suffixes),
        "sites_wrapped": len(replaced),
        "adapter_parameters": adapter_parameters,
        "sites_sample": replaced[:6],
    }


# --------------------------------------------------------------------------
# Hand-rolled host-offloaded AdamW.
# --------------------------------------------------------------------------


class HostOffloadAdamW:
    """AdamW whose master weights and moments live in host RAM.

    WHY THIS EXISTS.  A 4B student in bf16 needs 8.4 GB of weights and 8.4 GB of
    gradients on the card; fp32 master + m + v is another 50 GB, which no card
    here has and the host has four times over (R5 section 3).  So the step runs
    on 88 CPU cores and only the gradient and the updated weight cross PCIe.

    WHAT IS MEASURED SEPARATELY.  Three phases -- gradient to host, Adam on the
    host, parameter back to the device -- are timed independently, because the
    answer to "is CPU-offload full-parameter worth it at 4B" (D-R5-8) is
    entirely which of the three dominates.

    The staging buffers are pinned by default: an unpinned device-to-host copy
    of a 9 GB gradient set goes through a bounce buffer and is roughly half the
    bandwidth.  `--no-pin-host-state` exists because pinning is unswappable and
    a busy box may refuse it; the receipt says which happened.
    """

    def __init__(
        self,
        parameters: list[Any],
        *,
        learning_rate: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        pin: bool,
    ) -> None:
        import torch

        self.parameters = parameters
        self.learning_rate = float(learning_rate)
        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_count = 0
        self.pin_requested = bool(pin)
        self.pin_effective = bool(pin)
        self.master: list[Any] = []
        self.moment1: list[Any] = []
        self.moment2: list[Any] = []
        self.stage: list[Any] = []
        for parameter in parameters:
            self.master.append(parameter.detach().to("cpu", torch.float32).clone())
            self.moment1.append(torch.zeros_like(self.master[-1]))
            self.moment2.append(torch.zeros_like(self.master[-1]))
            try:
                buffer = torch.empty(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device="cpu",
                    pin_memory=self.pin_effective,
                )
            except RuntimeError:
                self.pin_effective = False
                buffer = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
            self.stage.append(buffer)
        self.phase_seconds = {"grad_to_host": [], "host_step": [], "param_to_device": []}

    def host_state_bytes(self) -> int:
        total = 0
        for group in (self.master, self.moment1, self.moment2, self.stage):
            for tensor in group:
                total += tensor.numel() * tensor.element_size()
        return total

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    def step(self) -> None:
        import torch

        self.step_count += 1
        start = time.perf_counter()
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters):
                if parameter.grad is None:
                    continue
                self.stage[index].copy_(parameter.grad, non_blocking=self.pin_effective)
        synchronise_all()
        self.phase_seconds["grad_to_host"].append(time.perf_counter() - start)

        start = time.perf_counter()
        bias1 = 1.0 - self.beta1**self.step_count
        bias2 = 1.0 - self.beta2**self.step_count
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters):
                if parameter.grad is None:
                    continue
                grad = self.stage[index].to(torch.float32)
                m = self.moment1[index]
                v = self.moment2[index]
                master = self.master[index]
                m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
                denominator = (v / bias2).sqrt_().add_(self.eps)
                if self.weight_decay:
                    master.mul_(1.0 - self.learning_rate * self.weight_decay)
                master.addcdiv_(m, denominator, value=-self.learning_rate / bias1)
                self.stage[index].copy_(master.to(self.stage[index].dtype))
        self.phase_seconds["host_step"].append(time.perf_counter() - start)

        start = time.perf_counter()
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters):
                if parameter.grad is None:
                    continue
                parameter.data.copy_(self.stage[index], non_blocking=self.pin_effective)
        synchronise_all()
        self.phase_seconds["param_to_device"].append(time.perf_counter() - start)

    def report(self) -> dict[str, Any]:
        def summarise(values: list[float]) -> dict[str, Any]:
            if not values:
                return {"count": 0}
            ordered = sorted(values)
            return {
                "count": len(values),
                "mean_seconds": round(sum(values) / len(values), 6),
                "median_seconds": round(ordered[len(ordered) // 2], 6),
                "min_seconds": round(ordered[0], 6),
                "max_seconds": round(ordered[-1], 6),
            }

        return {
            "optimizer": "hand-rolled AdamW, fp32 master/m/v in host RAM, step on CPU",
            "pin_memory_requested": self.pin_requested,
            "pin_memory_effective": self.pin_effective,
            "host_state_gb": gigabytes(self.host_state_bytes()),
            "steps": self.step_count,
            "phase_timings": {
                name: summarise(values) for name, values in self.phase_seconds.items()
            },
        }


class DynamicLossScaler:
    """Hand-rolled dynamic loss scaling for the fp16 half of a split run.

    torch ships `amp.GradScaler`, but it hides the one number this probe owes
    the campaign -- how often the scale backed off in the first hundred steps
    (receipt owed (9), R5 section 12).  So the counters are explicit and the
    policy is the ordinary one: halve on a non-finite gradient and skip the
    step, double after `growth_interval` clean steps.
    """

    def __init__(self, initial_scale: float = 2.0**16, growth_interval: int = 200) -> None:
        self.scale = float(initial_scale)
        self.growth_interval = int(growth_interval)
        self.clean_streak = 0
        self.overflow_steps = 0
        self.skipped_steps = 0
        self.growths = 0
        self.backoffs = 0

    def scale_loss(self, loss: Any) -> Any:
        return loss * self.scale

    def unscale_and_check(self, parameters: list[Any]) -> bool:
        """Divide gradients by the scale; return True when all are finite.

        The non-finite test accumulates into ONE scalar per device and reads it
        once.  A `bool(...)` per tensor would be a device synchronisation per
        tensor, and a 27B LoRA run has about 1,150 adapter tensors -- the check
        would be a measurable part of the step it is supposed to be checking.
        """

        import torch

        inverse = 1.0 / self.scale
        found: dict[Any, Any] = {}
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is None:
                    continue
                parameter.grad.mul_(inverse)
                device = parameter.grad.device
                accumulator = found.get(device)
                if accumulator is None:
                    accumulator = torch.zeros((), device=device, dtype=torch.float32)
                    found[device] = accumulator
                accumulator += (~torch.isfinite(parameter.grad).all()).to(torch.float32)
        finite = all(float(value.item()) == 0.0 for value in found.values())
        if finite:
            self.clean_streak += 1
            if self.clean_streak >= self.growth_interval:
                self.scale *= 2.0
                self.clean_streak = 0
                self.growths += 1
        else:
            self.overflow_steps += 1
            self.skipped_steps += 1
            self.clean_streak = 0
            self.scale = max(self.scale * 0.5, 1.0)
            self.backoffs += 1
        return finite

    def report(self) -> dict[str, Any]:
        return {
            "final_scale": self.scale,
            "overflow_steps": self.overflow_steps,
            "skipped_optimizer_steps": self.skipped_steps,
            "scale_growths": self.growths,
            "scale_backoffs": self.backoffs,
            "growth_interval": self.growth_interval,
        }


# --------------------------------------------------------------------------
# The chunked head.  See the module docstring for why.
# --------------------------------------------------------------------------


def sequence_chunks(length: int, chunks: int) -> list[tuple[int, int]]:
    chunks = max(1, min(int(chunks), int(length)))
    size = (length + chunks - 1) // chunks
    return [(start, min(start + size, length)) for start in range(0, length, size)]


def chunked_causal_loss(
    hidden: Any,
    lm_head: Any,
    target_ids: Any,
    chunks: int,
    *,
    loss_scale: float = 1.0,
    backward: bool = True,
) -> tuple[float, Any]:
    """Mean next-token cross-entropy, materialising one chunk of logits at a time.

    `hidden` is the trunk output WITH its graph.  The head runs on a detached
    copy so each chunk's backward can be taken and its logits freed immediately;
    the caller then backwards the trunk once with `detached.grad`.  Returns the
    unscaled loss value and the detached hidden state (whose `.grad` carries the
    accumulated head gradient).
    """

    import torch
    import torch.nn.functional as F

    detached = hidden.detach().requires_grad_(backward)
    positions = hidden.shape[1] - 1
    if positions <= 0:
        raise ProbeError("a next-token loss needs at least two positions")
    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    for start, stop in sequence_chunks(positions, chunks):
        piece = detached[:, start:stop, :]
        logits = lm_head(piece).float()
        targets = target_ids[:, start + 1 : stop + 1].to(logits.device)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        ) / float(positions * hidden.shape[0])
        if backward:
            (loss * loss_scale).backward()
        total = total + loss.detach().to(total.device)
        del logits, loss
    return float(total.item()), detached


def forward_only_head(hidden: Any, lm_head: Any, chunks: int, top_k: int) -> int:
    """Logits over every position plus a top-k gather; returns rows scored.

    A teacher forward is not "the trunk ran": stage D stores top-k logits, so
    the head and the gather are part of the price and are timed with it.
    """

    import torch

    rows = 0
    for start, stop in sequence_chunks(hidden.shape[1], chunks):
        logits = lm_head(hidden[:, start:stop, :])
        values, indices = torch.topk(logits.float(), k=min(top_k, logits.shape[-1]), dim=-1)
        rows += int(values.shape[0] * values.shape[1])
        del logits, values, indices
    return rows


# --------------------------------------------------------------------------
# Streamed 27B forward.  One decoder layer resident at a time.
# --------------------------------------------------------------------------


class ShardReader:
    """Reads named tensors out of a sharded safetensors checkpoint, on demand.

    Handles stay open across passes so the page cache -- not a re-`open` -- is
    what makes the second pass faster, and `bytes_read` counts every byte handed
    out so the receipt can report the streaming rate in MB/s beside the tok/s.
    """

    def __init__(self, model_dir: Path, device: str) -> None:
        from safetensors import safe_open

        self._safe_open = safe_open
        self.model_dir = Path(model_dir)
        self.device = device
        index_path = self.model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise ProbeError(f"no safetensors index at {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]
        self.total_size = int(index.get("metadata", {}).get("total_size", 0))
        self._handles: dict[str, Any] = {}
        self.bytes_read = 0

    def _handle(self, filename: str) -> Any:
        handle = self._handles.get(filename)
        if handle is None:
            handle = self._safe_open(
                str(self.model_dir / filename), framework="pt", device=self.device
            )
            self._handles[filename] = handle
        return handle

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> Any:
        if name not in self.weight_map:
            raise ProbeError(f"{name!r} is not in {self.model_dir}/model.safetensors.index.json")
        tensor = self._handle(self.weight_map[name]).get_tensor(name)
        self.bytes_read += tensor.numel() * tensor.element_size()
        return tensor

    def names_under(self, prefix: str) -> list[str]:
        return sorted(name for name in self.weight_map if name.startswith(prefix))


def make_streamed_layer_list(decoder_class: Any) -> Any:
    import torch
    from torch import nn

    class StreamedLayerList(nn.Module):
        """A stand-in for `Qwen3_5TextModel.layers` that owns two layer buffers.

        `Qwen3_5TextModel.forward` iterates `self.layers[: num_hidden_layers]`.
        A slice of this object is a GENERATOR that loads layer i's weights into
        a preallocated buffer of the right kind and yields it, so the stock
        prologue (mrope, causal mask, linear-attention mask), the stock decoder
        layer and the stock epilogue all run unmodified and exactly one layer's
        weights are resident at a time.  Reusing two buffers rather than
        building a module per layer keeps the allocator from fragmenting 64
        times per pass.
        """

        def __init__(self, config: Any, reader: ShardReader, prefix: str, device: str, dtype: Any) -> None:
            super().__init__()
            self.text_config = config
            self.reader = reader
            self.prefix = prefix
            self.count = int(config.num_hidden_layers)
            self.buffers_by_type = nn.ModuleDict()
            seen: dict[str, int] = {}
            for index, layer_type in enumerate(config.layer_types[: self.count]):
                seen.setdefault(str(layer_type), index)
            #: Built at the default dtype and cast, rather than under a
            #: `set_default_dtype` block: one transient fp32 layer is 1.5 GB at
            #: 27B geometry and freed immediately, which is a cheaper thing to
            #: pay than a global dtype switch that a raised exception could leak.
            for layer_type, index in seen.items():
                with torch.device(device):
                    layer = decoder_class(config, index)
                self.buffers_by_type[layer_type] = layer.to(device=device, dtype=dtype).eval()
            self.layer_load_seconds = 0.0
            self.layer_loads = 0

        def _materialise(self, index: int) -> Any:
            layer_type = str(self.text_config.layer_types[index])
            layer = self.buffers_by_type[layer_type]
            start = time.perf_counter()
            base = f"{self.prefix}.{index}."
            state = {
                name[len(base) :]: self.reader.get(name)
                for name in self.reader.names_under(base)
            }
            layer.load_state_dict(state, strict=True, assign=False)
            del state
            self.layer_load_seconds += time.perf_counter() - start
            self.layer_loads += 1
            return layer

        def __len__(self) -> int:
            return self.count

        def __getitem__(self, key: Any) -> Any:  # noqa: ANN401
            if isinstance(key, slice):
                return (self._materialise(i) for i in range(*key.indices(self.count)))
            return self._materialise(int(key))

    return StreamedLayerList


def build_streamed_text_model(
    model_dir: Path, device: str, dtype: Any, attention_implementation: str
) -> tuple[Any, Any, dict[str, Any]]:
    """Return (streamed text trunk, lm_head, facts).

    The trunk is a real `Qwen3_5TextModel` built on the meta device -- so its 64
    layers cost nothing -- with the three small resident modules materialised
    and `layers` replaced by the streaming list above.
    """

    import torch
    from torch import nn
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5RMSNorm,
        Qwen3_5TextModel,
        Qwen3_5TextRotaryEmbedding,
    )

    config = Qwen3_5Config.from_pretrained(str(model_dir))
    text_config = config.text_config
    text_config.dtype = dtype
    text_config._attn_implementation = attention_implementation

    with torch.device("meta"):
        model = Qwen3_5TextModel(text_config)

    reader = ShardReader(model_dir, device=device)
    embed = nn.Embedding(
        text_config.vocab_size, text_config.hidden_size, dtype=dtype, device=device
    )
    with torch.no_grad():
        embed.weight.copy_(reader.get(f"{TEXT_TRUNK_PREFIX}.embed_tokens.weight"))
    norm = Qwen3_5RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
    norm = norm.to(device=device, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(reader.get(f"{TEXT_TRUNK_PREFIX}.norm.weight"))
    rotary = Qwen3_5TextRotaryEmbedding(config=text_config).to(device)

    model.embed_tokens = embed
    model.norm = norm
    model.rotary_emb = rotary
    del model.layers
    streamed_class = make_streamed_layer_list(Qwen3_5DecoderLayer)
    model.layers = streamed_class(
        text_config, reader, f"{TEXT_TRUNK_PREFIX}.layers", device, dtype
    )
    model.eval()

    lm_head = nn.Linear(
        text_config.hidden_size, text_config.vocab_size, bias=False, dtype=dtype, device=device
    )
    tied = not reader.has("lm_head.weight")
    with torch.no_grad():
        if tied:
            lm_head.weight = embed.weight
        else:
            lm_head.weight.copy_(reader.get("lm_head.weight"))
    lm_head.eval()

    resident = sum(
        parameter.numel() * parameter.element_size()
        for parameter in list(model.parameters()) + list(lm_head.parameters())
    )
    facts = {
        "streaming": True,
        "layers_streamed": int(text_config.num_hidden_layers),
        "layer_buffer_types": sorted(model.layers.buffers_by_type.keys()),
        "lm_head_tied_to_embedding": tied,
        "resident_parameter_bytes_gb": gigabytes(resident),
        "checkpoint_total_size_gb": gigabytes(reader.total_size),
        "attention_implementation": attention_implementation,
    }
    return model, lm_head, {"facts": facts, "reader": reader}


# --------------------------------------------------------------------------
# Model construction for the resident (non-streamed) cells.
# --------------------------------------------------------------------------


def resolve_dtype(name: str) -> Any:
    import torch

    allowed = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in allowed:
        raise ProbeError(f"{name!r} is not one of {sorted(allowed)}")
    return allowed[name]


def parse_max_memory(text: str | None) -> dict[Any, str] | None:
    """`0=40GiB,1=20GiB` -> {0: '40GiB', 1: '20GiB'}; `cpu=...` stays a string key."""

    if not text:
        return None
    out: dict[Any, str] = {}
    for item in text.split(","):
        key, _, value = item.partition("=")
        key = key.strip()
        out[key if key == "cpu" else int(key)] = value.strip()
    return out


def load_resident_model(
    model_dir: Path,
    dtype: Any,
    device: str,
    attention_implementation: str,
    device_map: str | None,
    max_memory: dict[Any, str] | None,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import transformers

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    declared = str(config["architectures"][0])
    loader = getattr(transformers, declared, None)
    if loader is None:
        raise ProbeError(
            f"stock transformers {transformers.__version__} has no class {declared!r}; "
            "this probe never uses trust_remote_code."
        )
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "attn_implementation": attention_implementation,
        "trust_remote_code": False,
    }
    if device_map:
        kwargs["device_map"] = device_map
        if max_memory:
            kwargs["max_memory"] = max_memory
    started = time.perf_counter()
    model = loader.from_pretrained(str(model_dir), **kwargs)
    if not device_map:
        model = model.to(device)
    load_seconds = time.perf_counter() - started
    model.eval()

    placement: dict[str, int] = {}
    for _name, parameter in model.named_parameters():
        key = str(parameter.device)
        placement[key] = placement.get(key, 0) + parameter.numel()
    #: A `max_memory` budget smaller than the checkpoint does not fail at load
    #: time: accelerate leaves the modules it could not place on the META device
    #: and the run dies later, deep inside a kernel, with "Tensor.item() cannot
    #: be called on meta tensors".  That traceback names neither the cause nor
    #: the fix, so the refusal is made here where both are known.
    meta_parameters = placement.get("meta", 0)
    if meta_parameters:
        raise ProbeError(
            f"{meta_parameters:,} parameters were left on the meta device: the "
            f"--max-memory budget {max_memory} is smaller than the checkpoint "
            f"({sum(placement.values()):,} parameters at {dtype}).  Raise the "
            "budget or add a cpu/disk entry; this probe will not silently "
            "measure a partially materialised model."
        )
    #: The fp16 overflow check R5 section 12 owes (receipt 9), done with
    #: `amin`/`amax` rather than `isfinite(p).all()`.  The obvious form allocates
    #: a full bool tensor -- 1.19 GiB for a 248,320 x 5,120 embedding -- and on a
    #: split 27B that allocation is itself an OOM, so the diagnostic would be the
    #: thing that broke the run.  min and max propagate NaN and expose +-inf, so
    #: two scalars answer the same question in constant memory.
    non_finite = 0
    with torch.no_grad():
        for _name, parameter in model.named_parameters():
            if parameter.numel() == 0:
                continue
            low = parameter.amin()
            high = parameter.amax()
            if not (bool(torch.isfinite(low)) and bool(torch.isfinite(high))):
                non_finite += 1
    device_map_summary = getattr(model, "hf_device_map", None)
    if device_map:
        log(
            "placement: "
            + ", ".join(f"{k}={v/1e9:.2f}e9 params" for k, v in sorted(placement.items()))
        )
    facts = {
        "loaded_class": type(model).__name__,
        "hf_device_map_devices": sorted({str(v) for v in (device_map_summary or {}).values()})
        if device_map_summary
        else None,
        "load_seconds": round(load_seconds, 2),
        "device_map": device_map,
        "max_memory": None if max_memory is None else {str(k): v for k, v in max_memory.items()},
        "attention_implementation": attention_implementation,
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_by_device": placement,
        "non_finite_parameter_tensors_after_cast": non_finite,
    }
    return model, facts


def text_trunk_and_head(model: Any) -> tuple[Any, Any]:
    """`Qwen3_5ForConditionalGeneration` -> (text trunk, lm_head).

    The probe drives the text trunk directly rather than the conditional-
    generation wrapper: the wrapper's forward would build the whole-sequence
    logit tensor itself, which is exactly the allocation the chunked head exists
    to avoid.
    """

    inner = getattr(model, "model", None)
    trunk = getattr(inner, "language_model", None) if inner is not None else None
    if trunk is None:
        raise ProbeError(f"{type(model).__name__} has no model.language_model to drive")
    head = getattr(model, "lm_head", None)
    if head is None:
        raise ProbeError(f"{type(model).__name__} has no lm_head")
    return trunk, head


# --------------------------------------------------------------------------
# The measurement itself.
# --------------------------------------------------------------------------


def make_batch(micro_batch: int, length: int, device: str, seed: int) -> Any:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    ids = torch.randint(
        0, DEFINED_VOCABULARY_ROWS, (micro_batch, length), generator=generator, dtype=torch.long
    )
    return ids.to(device)


def summarise_steps(seconds: list[float], tokens_per_step: int) -> dict[str, Any]:
    if not seconds:
        return {"steps": 0}
    ordered = sorted(seconds)
    total = sum(seconds)
    return {
        "steps": len(seconds),
        "tokens_per_step": tokens_per_step,
        "tokens_total": tokens_per_step * len(seconds),
        "wall_seconds": round(total, 4),
        "tokens_per_second": round(tokens_per_step * len(seconds) / total, 3) if total else None,
        "seconds_per_step_mean": round(total / len(seconds), 5),
        "seconds_per_step_median": round(ordered[len(ordered) // 2], 5),
        "seconds_per_step_min": round(ordered[0], 5),
        "seconds_per_step_max": round(ordered[-1], 5),
        "seconds_per_step_all": [round(value, 5) for value in seconds],
    }


def derived_budgets(tokens_per_second: float | None, device_count: int) -> dict[str, Any]:
    """Tokens per GPU-hour and GPU-hours per rung.

    GPU-hours are wall clock times the number of devices the run occupies -- the
    same definition `stage2_training_receipt.py` seals, so a stage-D receipt and
    this probe can be compared without a unit argument.
    """

    if not tokens_per_second:
        return {"available": False}
    tokens_per_wall_hour = tokens_per_second * 3600.0
    tokens_per_gpu_hour = tokens_per_wall_hour / max(1, device_count)
    return {
        "available": True,
        "device_count": device_count,
        "tokens_per_wall_clock_hour": round(tokens_per_wall_hour, 1),
        "tokens_per_gpu_hour": round(tokens_per_gpu_hour, 1),
        "rungs": {
            str(rung): {
                "wall_clock_hours": round(rung / tokens_per_wall_hour, 3),
                "wall_clock_days": round(rung / tokens_per_wall_hour / 24.0, 3),
                "gpu_hours": round(rung / tokens_per_wall_hour * max(1, device_count), 3),
            }
            for rung in RUNG_TOKENS
        },
    }


class Runner:
    """Holds one configured cell and can execute it for N steps."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = None
        self.trunk = None
        self.lm_head = None
        self.optimizer = None
        self.host_optimizer = None
        self.scaler = None
        self.stream_reader = None
        self.trainable: list[Any] = []
        self.autocast_dtype = None
        #: WHY THE AUTOCAST WEIGHT CACHE IS OFF UNDER GRADIENT CHECKPOINTING.
        #: With the cache on, the fp16/bf16 copy of an fp32 LoRA weight is made
        #: once in the forward and reused; the cache is cleared before backward,
        #: so the checkpoint's RECOMPUTE casts again and saves two extra tensors
        #: per adapter.  `torch.utils.checkpoint` then refuses the whole step:
        #: "Recomputed values ... have different metadata", with the saved and
        #: recomputed lists offset by exactly two.  That is what killed the first
        #: three attempts at the 27B split.  Off, the two passes are identical.
        self.autocast_cache = not bool(getattr(args, "gradient_checkpointing", False))
        self.facts: dict[str, Any] = {}
        self.non_finite_losses = 0
        self.non_finite_gradient_steps = 0
        self.loss_values: list[float] = []

    # -- forward-only -----------------------------------------------------

    def forward_only_step(self, ids: Any) -> None:
        import torch

        with torch.no_grad():
            if self.autocast_dtype is None:
                hidden = self.trunk(input_ids=ids, use_cache=False).last_hidden_state
                forward_only_head(hidden, self.lm_head, self.args.loss_chunks, self.args.top_k)
            else:
                with torch.autocast(
                    "cuda", dtype=self.autocast_dtype, cache_enabled=self.autocast_cache
                ):
                    hidden = self.trunk(input_ids=ids, use_cache=False).last_hidden_state
                    forward_only_head(hidden, self.lm_head, self.args.loss_chunks, self.args.top_k)
        del hidden

    # -- train ------------------------------------------------------------

    def micro_step(self, ids: Any, loss_scale: float) -> float:
        """One forward + backward.  No optimizer touch, so it can be accumulated."""

        import torch

        if self.autocast_dtype is None:
            hidden = self.trunk(input_ids=ids, use_cache=False).last_hidden_state
            loss_value, detached = chunked_causal_loss(
                hidden, self.lm_head, ids, self.args.loss_chunks, loss_scale=loss_scale
            )
        else:
            with torch.autocast(
                "cuda", dtype=self.autocast_dtype, cache_enabled=self.autocast_cache
            ):
                hidden = self.trunk(input_ids=ids, use_cache=False).last_hidden_state
                loss_value, detached = chunked_causal_loss(
                    hidden, self.lm_head, ids, self.args.loss_chunks, loss_scale=loss_scale
                )
        hidden.backward(gradient=detached.grad)
        del hidden, detached
        return loss_value

    def train_step(self, ids: Any) -> None:
        """`accumulation` micro-steps, then ONE optimizer step.

        Accumulation is not decoration.  The stage-D batch is 16 x 4,096 tokens
        (CAMPAIGN.md 4.2), and for the host-offloaded optimizer the step cost is
        per OPTIMIZER step, not per token: measuring only at accumulation 1 would
        charge every token the full 40-second host Adam and answer D-R5-8 with a
        number no real schedule would ever pay.
        """

        import torch

        accumulation = max(1, int(self.args.accumulation))
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        if self.host_optimizer is not None:
            self.host_optimizer.zero_grad()

        scale = (self.scaler.scale if self.scaler is not None else 1.0) / accumulation
        total = 0.0
        for _ in range(accumulation):
            total += self.micro_step(ids, scale)
        loss_value = total / accumulation

        self.loss_values.append(loss_value)
        if not (loss_value == loss_value and abs(loss_value) != float("inf")):
            self.non_finite_losses += 1

        finite = True
        if self.scaler is not None:
            finite = self.scaler.unscale_and_check(self.trainable)
            if not finite:
                self.non_finite_gradient_steps += 1
        if finite and self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.trainable, self.args.grad_clip)
        if finite:
            if self.optimizer is not None:
                self.optimizer.step()
            if self.host_optimizer is not None:
                self.host_optimizer.step()

    def run_phase(self, phase: str, ids: Any, warmup: int, steps: int) -> dict[str, Any]:
        import torch

        function = self.forward_only_step if phase == "forward" else self.train_step
        for _ in range(warmup):
            function(ids)
        synchronise_all()
        reset_memory_counters()
        durations: list[float] = []
        for _ in range(steps):
            start = time.perf_counter()
            function(ids)
            synchronise_all()
            durations.append(time.perf_counter() - start)
        tokens = int(ids.shape[0] * ids.shape[1])
        if phase != "forward":
            tokens *= max(1, int(self.args.accumulation))
        report = summarise_steps(durations, tokens)
        report["accumulation"] = max(1, int(self.args.accumulation)) if phase != "forward" else 1
        report["torch_memory"] = torch_memory_report()
        return report


def build_runner(args: argparse.Namespace, model_dir: Path) -> tuple[Runner, dict[str, Any]]:
    """Wire one cell: load, freeze, wrap, and choose the optimizer."""

    import torch

    runner = Runner(args)
    dtype = resolve_dtype(args.dtype)
    facts: dict[str, Any] = {}

    if args.cell == "27b-teacher-stream":
        trunk, head, extra = build_streamed_text_model(
            model_dir, args.device, dtype, args.attn_implementation
        )
        runner.trunk = trunk
        runner.lm_head = head
        runner.autocast_dtype = None
        facts.update(extra["facts"])
        facts["stream_reader"] = {"bytes_read_setup_gb": gigabytes(extra["reader"].bytes_read)}
        runner.stream_reader = extra["reader"]
        runner.facts = facts
        return runner, facts

    device_map = args.device_map or None
    model, load_facts = load_resident_model(
        model_dir,
        dtype,
        args.device,
        args.attn_implementation,
        device_map,
        parse_max_memory(args.max_memory),
    )
    facts.update(load_facts)
    runner.model = model
    trunk, head = text_trunk_and_head(model)
    runner.trunk = trunk
    runner.lm_head = head

    vision = getattr(getattr(model, "model", None), "visual", None)
    vision_parameters = 0 if vision is None else sum(p.numel() for p in vision.parameters())
    #: A text-only throughput probe never calls the vision tower, but it still
    #: pays for its weights.  On a 27B split with ~8 GiB of headroom per card
    #: that is worth a flag -- and worth saying out loud in the receipt, because
    #: a stage-D omni student WOULD carry the tower and would pay the difference.
    if args.drop_vision and vision is not None:
        setattr(model.model, "visual", None)
        vision = None
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    facts["vision_tower_dropped"] = bool(args.drop_vision and vision_parameters)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if args.mode == "lora":
        census = attach_lora(trunk, args.lora_rank, args.lora_alpha, LORA_TARGET_SUFFIXES)
        facts["lora"] = census
        runner.trainable = [p for p in model.parameters() if p.requires_grad]
        #: No autocast: the base is already in the compute dtype and the adapters
        #: cast themselves (see LoRALinear).  Autocast would add nothing here and
        #: costs the checkpoint-recompute determinism described there.
        runner.autocast_dtype = None
        runner.optimizer = torch.optim.AdamW(
            runner.trainable,
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
        facts["optimizer"] = "torch.optim.AdamW over fp32 LoRA adapters (the adapters are their own master copy)"
    elif args.mode == "full-gpu-master":
        # fp32 master weights under bf16 autocast: the sealed LFM2 recipe.
        trunk.float()
        head.float()
        if getattr(model.config, "tie_word_embeddings", False):
            model.tie_weights()
        for parameter in trunk.parameters():
            parameter.requires_grad_(True)
        for parameter in head.parameters():
            parameter.requires_grad_(True)
        if vision is not None:
            vision.to(dtype)
            for parameter in vision.parameters():
                parameter.requires_grad_(False)
        seen: set[int] = set()
        runner.trainable = []
        for parameter in list(trunk.parameters()) + list(head.parameters()):
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                runner.trainable.append(parameter)
        runner.autocast_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else None
        runner.optimizer = torch.optim.AdamW(
            runner.trainable,
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
        facts["optimizer"] = "torch.optim.AdamW over fp32 parameters (the parameters ARE the master copy)"
        facts["weights_dtype"] = "float32 (autocast to " + args.dtype + " for compute)"
    elif args.mode == "full-cpu-adam":
        for parameter in trunk.parameters():
            parameter.requires_grad_(True)
        for parameter in head.parameters():
            parameter.requires_grad_(True)
        if vision is not None:
            for parameter in vision.parameters():
                parameter.requires_grad_(False)
        seen = set()
        runner.trainable = []
        for parameter in list(trunk.parameters()) + list(head.parameters()):
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                runner.trainable.append(parameter)
        runner.autocast_dtype = None
        log(
            f"building host optimizer state for {sum(p.numel() for p in runner.trainable):,} "
            "parameters (fp32 master + m + v + pinned staging)"
        )
        runner.host_optimizer = HostOffloadAdamW(
            runner.trainable,
            learning_rate=args.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=args.weight_decay,
            pin=args.pin_host_state,
        )
        facts["optimizer"] = "hand-rolled host-offloaded AdamW"
        facts["weights_dtype"] = args.dtype + " on device, float32 master on host"
    else:
        raise ProbeError(f"unknown mode {args.mode!r}")

    if args.dtype == "float16" and args.mode == "lora":
        runner.scaler = DynamicLossScaler(
            initial_scale=args.initial_loss_scale, growth_interval=args.loss_scale_growth_interval
        )
        facts["loss_scaling"] = "hand-rolled dynamic loss scaler (fp16)"

    if args.gradient_checkpointing:
        reentrant = bool(args.checkpoint_reentrant)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": reentrant}
        )
        if reentrant:
            #: Reentrant checkpointing drops the graph unless something in the
            #: input requires grad; with LoRA every trainable tensor is INSIDE
            #: the layer, so the inputs must be made to require it explicitly.
            model.enable_input_require_grads()
        model.train()
        facts["checkpoint_use_reentrant"] = reentrant
    facts["gradient_checkpointing"] = bool(args.gradient_checkpointing)
    facts["autocast_weight_cache_enabled"] = runner.autocast_cache

    facts["trainable_parameters"] = sum(p.numel() for p in runner.trainable)
    facts["trainable_tensors"] = len(runner.trainable)
    facts["vision_tower_parameters_frozen"] = vision_parameters
    if runner.host_optimizer is not None:
        facts["host_optimizer"] = runner.host_optimizer.report()
    runner.facts = facts
    return runner, facts


def micro_batch_search(args: argparse.Namespace, model_dir: Path, maximum: int) -> dict[str, Any]:
    """Double the micro-batch until a step OOMs; report the largest that ran.

    Each candidate is built in a FRESH runner.  Reusing one is cheaper and
    wrong: an OOM leaves the allocator and the optimizer state in a shape that
    makes the next candidate's verdict about the previous candidate.
    """

    import torch

    attempts: list[dict[str, Any]] = []
    largest = 0
    candidate = 1
    #: The search asks a memory question, and memory does not depend on how many
    #: micro-steps are accumulated before the optimizer runs, so the search runs
    #: at accumulation 1 and the timed measurement afterwards uses the real one.
    requested_accumulation = args.accumulation
    args.accumulation = 1
    while candidate <= maximum:
        log(f"micro-batch search: trying {candidate}")
        record: dict[str, Any] = {"micro_batch": candidate}
        runner = None
        try:
            runner, _facts = build_runner(args, model_dir)
            ids = make_batch(candidate, args.sequence_length, args.device, args.seed)
            reset_memory_counters()
            for _ in range(2):
                runner.train_step(ids)
            synchronise_all()
            record["fits"] = True
            record["peak_allocated_gb"] = torch_memory_report()
            largest = candidate
        except torch.OutOfMemoryError as exc:
            record["fits"] = False
            record["error"] = "torch.OutOfMemoryError: " + str(exc).splitlines()[0][:300]
        except RuntimeError as exc:
            record["fits"] = False
            record["error"] = type(exc).__name__ + ": " + str(exc).splitlines()[0][:300]
        finally:
            del runner
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        attempts.append(record)
        if not record["fits"]:
            break
        candidate *= 2
    args.accumulation = requested_accumulation
    return {
        "attempts": attempts,
        "largest_micro_batch_that_fits": largest,
        "cap": maximum,
        "searched_at_accumulation": 1,
    }


# --------------------------------------------------------------------------
# Self-test.  CPU only.
# --------------------------------------------------------------------------


def self_test() -> int:
    import torch
    import torch.nn.functional as F
    from torch import nn

    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        print(f"{status} {name}{(' :: ' + detail) if detail else ''}", flush=True)
        if not condition:
            failures.append(name)

    torch.manual_seed(20260902)

    # 1. chunked loss == one-shot loss, value and gradient.
    batch, length, hidden_size, vocab = 2, 33, 16, 61
    head = nn.Linear(hidden_size, vocab, bias=False)
    hidden = torch.randn(batch, length, hidden_size, requires_grad=True)
    ids = torch.randint(0, vocab, (batch, length))
    reference_logits = head(hidden)
    reference = F.cross_entropy(
        reference_logits[:, :-1, :].reshape(-1, vocab), ids[:, 1:].reshape(-1)
    )
    reference.backward()
    reference_grad = hidden.grad.clone()
    hidden.grad = None
    head.zero_grad()
    value, detached = chunked_causal_loss(hidden, head, ids, chunks=5)
    hidden.backward(gradient=detached.grad)
    check(
        "chunked_causal_loss value matches one-shot cross-entropy",
        abs(value - float(reference)) < 1e-5,
        f"{value:.7f} vs {float(reference):.7f}",
    )
    check(
        "chunked_causal_loss gradient matches one-shot gradient",
        torch.allclose(hidden.grad, reference_grad, atol=1e-6),
        f"max abs diff {float((hidden.grad - reference_grad).abs().max()):.3e}",
    )

    # 2. sequence_chunks covers every position exactly once.
    covered = [stop - start for start, stop in sequence_chunks(4096, 7)]
    check("sequence_chunks partitions the sequence", sum(covered) == 4096, str(covered))
    check("sequence_chunks clamps chunks to length", len(sequence_chunks(3, 99)) == 3)

    # 3. LoRA starts as the identity and its adapters are the only trainables.
    lora_class = make_lora_linear(nn.Linear)
    base = nn.Linear(8, 5, bias=False)
    wrapper = lora_class(base, rank=4, alpha=8.0)
    x = torch.randn(3, 8)
    check(
        "LoRA wrapper is the identity at initialisation (B is zero)",
        torch.allclose(wrapper(x), base(x), atol=0),
    )
    trainable = [name for name, p in wrapper.named_parameters() if p.requires_grad]
    check("LoRA freezes the base", sorted(trainable) == ["lora_a", "lora_b"], str(sorted(trainable)))
    check("LoRA adapters are fp32", wrapper.lora_a.dtype == torch.float32)

    # 4. attach_lora hits exactly the named sites.
    class Fake(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
            self.self_attn.k_proj = nn.Linear(4, 4, bias=False)
            self.mlp = nn.Module()
            self.mlp.gate_proj = nn.Linear(4, 4, bias=False)
            self.linear_attn = nn.Module()
            self.linear_attn.in_proj_qkv = nn.Linear(4, 4, bias=False)
            self.linear_attn.in_proj_z = nn.Linear(4, 4, bias=False)

    fake = Fake()
    census = attach_lora(fake, 2, 4.0, LORA_TARGET_SUFFIXES)
    check("attach_lora wrapped four sites", census["sites_wrapped"] == 4, str(census["sites_wrapped"]))
    check(
        "attach_lora left in_proj_z alone",
        type(fake.linear_attn.in_proj_z).__name__ == "Linear",
    )
    check(
        "attach_lora counted adapter parameters",
        census["adapter_parameters"] == 4 * (2 * 4 + 4 * 2),
        str(census["adapter_parameters"]),
    )

    # 5. Host-offloaded AdamW matches torch.optim.AdamW.
    reference_param = nn.Parameter(torch.randn(7, 5, dtype=torch.float32))
    ours = nn.Parameter(reference_param.detach().clone())
    reference_optimizer = torch.optim.AdamW(
        [reference_param], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    host = HostOffloadAdamW(
        [ours], learning_rate=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, pin=False
    )
    for _ in range(4):
        gradient = torch.randn(7, 5)
        reference_param.grad = gradient.clone()
        ours.grad = gradient.clone()
        reference_optimizer.step()
        host.step()
        with torch.no_grad():
            ours.copy_(host.master[0])
    difference = float((reference_param - ours).abs().max())
    check(
        "HostOffloadAdamW matches torch.optim.AdamW over four steps",
        difference < 1e-6,
        f"max abs diff {difference:.3e}",
    )

    # 6. Dynamic loss scaler backs off on a non-finite gradient and skips.
    scaler = DynamicLossScaler(initial_scale=1024.0, growth_interval=2)
    parameter = nn.Parameter(torch.zeros(3))
    parameter.grad = torch.tensor([1.0, float("inf"), 3.0])
    check("scaler reports a non-finite gradient", scaler.unscale_and_check([parameter]) is False)
    check("scaler halved the scale", scaler.scale == 512.0, str(scaler.scale))
    parameter.grad = torch.tensor([1.0, 2.0, 3.0])
    scaler.unscale_and_check([parameter])
    scaler.unscale_and_check([parameter])
    check("scaler grew after a clean streak", scaler.scale == 1024.0, str(scaler.scale))
    check("scaler counted one overflow", scaler.report()["overflow_steps"] == 1)

    # 7. Budgets divide the way the receipt claims.
    budgets = derived_budgets(1000.0, 2)
    check(
        "derived_budgets: 1,000 tok/s on two cards is 1.8M tokens per GPU-hour",
        budgets["tokens_per_gpu_hour"] == 1_800_000.0,
        str(budgets["tokens_per_gpu_hour"]),
    )
    check(
        "derived_budgets: a 10M rung at 1,000 tok/s is 2.778 wall hours",
        abs(budgets["rungs"]["10000000"]["wall_clock_hours"] - 2.778) < 1e-3,
    )
    check(
        "derived_budgets: gpu_hours = wall x devices",
        abs(
            budgets["rungs"]["10000000"]["gpu_hours"]
            - 2 * budgets["rungs"]["10000000"]["wall_clock_hours"]
        )
        < 1e-6,
    )

    # 8. Canonical bytes are stable and sorted.
    check(
        "canonical_json_bytes is key-order independent",
        canonical_sha256({"b": 1, "a": 2}) == canonical_sha256({"a": 2, "b": 1}),
    )

    # 9. max-memory parsing.
    parsed = parse_max_memory("0=40GiB,1=20GiB,cpu=64GiB")
    check(
        "parse_max_memory keys",
        parsed == {0: "40GiB", 1: "20GiB", "cpu": "64GiB"},
        str(parsed),
    )

    print(
        f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}",
        flush=True,
    )
    return 0 if not failures else 1


# --------------------------------------------------------------------------
# stream-verify: the streamed engine against a stock full load.
# --------------------------------------------------------------------------


def run_stream_verify(args: argparse.Namespace, model_dir: Path) -> tuple[dict[str, Any], Any]:
    import torch

    dtype = resolve_dtype(args.dtype)
    length = int(args.sequence_length)
    ids = make_batch(1, length, args.device, args.seed)

    log("stream-verify: building the streamed engine")
    trunk, head, extra = build_streamed_text_model(
        model_dir, args.device, dtype, args.attn_implementation
    )
    with torch.no_grad():
        streamed_hidden = trunk(input_ids=ids, use_cache=False).last_hidden_state
        streamed_logits = head(streamed_hidden).float().cpu()
    del trunk, head
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    log("stream-verify: loading the same checkpoint the stock way")
    model, load_facts = load_resident_model(
        model_dir, dtype, args.device, args.attn_implementation, None, None
    )
    stock_trunk, stock_head = text_trunk_and_head(model)
    with torch.no_grad():
        stock_hidden = stock_trunk(input_ids=ids, use_cache=False).last_hidden_state
        stock_logits = stock_head(stock_hidden).float().cpu()

    difference = (streamed_logits - stock_logits).abs()
    denominator = stock_logits.norm()
    body = {
        "model_directory": str(model_dir),
        "sequence_length": length,
        "dtype": args.dtype,
        "maximum_absolute_logit_difference": float(difference.max()),
        "relative_l2": float((streamed_logits - stock_logits).norm() / denominator)
        if float(denominator) > 0
        else None,
        "argmax_agreement": float(
            (streamed_logits.argmax(-1) == stock_logits.argmax(-1)).float().mean()
        ),
        "bit_identical": bool(torch.equal(streamed_logits, stock_logits)),
        "streamed_facts": extra["facts"],
        "stock_load_facts": load_facts,
        "note": (
            "The streamed engine reuses the stock Qwen3_5TextModel prologue, the "
            "stock decoder layer and the stock epilogue; only `layers` is replaced "
            "by a generator that loads one layer at a time.  A non-zero difference "
            "here would mean the replacement changed the arithmetic, which is the "
            "only thing this arm can be wrong about."
        ),
    }
    return body, model


# --------------------------------------------------------------------------
# Cell execution + receipt.
# --------------------------------------------------------------------------


def cell_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Per-cell defaults the caller may still override on the command line."""

    if args.model_dir is None:
        args.model_dir = DEFAULT_MODEL_FOR_CELL[args.cell]
    if args.cell == "08b-full":
        args.mode = args.mode or "full-gpu-master"
        args.dtype = args.dtype or "bfloat16"
        args.learning_rate = args.learning_rate if args.learning_rate is not None else 1.5e-5
    elif args.cell == "4b-lora":
        args.mode = args.mode or "lora"
        args.dtype = args.dtype or "bfloat16"
        args.learning_rate = args.learning_rate if args.learning_rate is not None else 1.5e-4
    elif args.cell == "4b-full-cpuadam":
        args.mode = args.mode or "full-cpu-adam"
        args.dtype = args.dtype or "bfloat16"
        args.learning_rate = args.learning_rate if args.learning_rate is not None else 1.5e-5
    elif args.cell == "27b-teacher-stream":
        args.mode = args.mode or "forward-only"
        args.dtype = args.dtype or "bfloat16"
        args.learning_rate = args.learning_rate if args.learning_rate is not None else 0.0
    elif args.cell == "27b-lora-split":
        args.mode = args.mode or "lora"
        args.dtype = args.dtype or "float16"
        args.device_map = args.device_map or "sequential"
        args.learning_rate = args.learning_rate if args.learning_rate is not None else 1.5e-4
    elif args.cell == "stream-verify":
        args.mode = args.mode or "forward-only"
        args.dtype = args.dtype or "bfloat16"
        args.learning_rate = args.learning_rate if args.learning_rate is not None else 0.0
    return args


def package_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": platform.python_version()}
    for name in ("torch", "transformers", "fla", "safetensors", "accelerate", "numpy"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001
            versions[name] = None
        else:
            versions[name] = str(getattr(module, "__version__", "unknown"))
    try:
        import torch

        versions["torch_cuda"] = torch.version.cuda
        versions["cudnn"] = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001
        pass
    return versions


def kernel_report_for(model: Any, calibration_dir: Path | None) -> dict[str, Any]:
    """The v2 per-primitive kernel path, IMPORTED from the calibration lane.

    THE FIELD IS NOT RE-IMPLEMENTED HERE ON PURPOSE.  `kernel_path` travels in
    the calibration lane's `lane_digest_validity`, and a second, drifting copy
    of that rule in the distill lane is exactly the failure the campaign has
    already paid for once (the refuted v1 inventory rule).  So the probe imports
    the function from `calibration/oc_calib.py` and seals that file's sha256 in
    the receipt: if the definition moves, the digest moves with it.
    """

    if calibration_dir is None:
        return {"kernel_path": "not-probed", "reason": "no --calibration-dir given"}
    module_path = Path(calibration_dir) / "oc_calib.py"
    if not module_path.is_file():
        raise ProbeError(
            f"{module_path} is missing; the kernel-path field is defined there and "
            "this probe will not re-implement it.  Pass --calibration-dir."
        )
    sys.path.insert(0, str(Path(calibration_dir).resolve()))
    from oc_calib import kernel_path_report  # noqa: E402

    report = dict(kernel_path_report(model))
    report["source_module"] = str(module_path)
    report["source_module_sha256"] = file_sha256(module_path)
    return report


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").is_file():
        raise ProbeError(f"{model_dir} has no config.json")

    started_at = time.time()
    wall_start = time.perf_counter()
    sampler = ResourceSampler(interval_seconds=args.sample_interval)
    inventory_before = nvidia_smi_inventory()
    cached_before_kib = meminfo_kib("Cached")

    if args.cell == "stream-verify":
        sampler.start()
        body, model_for_kernel = run_stream_verify(args, model_dir)
        sampler.stop()
        measurement: dict[str, Any] = {"stream_verify": body}
        facts: dict[str, Any] = body.get("streamed_facts", {})
        search: dict[str, Any] | None = None
    else:
        search = None
        if str(args.micro_batch) == "auto":
            if args.mode == "forward-only":
                raise ProbeError("--micro-batch auto is only defined for a training cell")
            search = micro_batch_search(args, model_dir, args.micro_batch_max)
            if not search["largest_micro_batch_that_fits"]:
                raise ProbeError("no micro-batch fits, not even 1; the search trace is the finding")
            args.micro_batch = search["largest_micro_batch_that_fits"]
        else:
            args.micro_batch = int(args.micro_batch)

        sampler.start()
        runner, facts = build_runner(args, model_dir)
        model_for_kernel = runner.model if runner.model is not None else runner.trunk
        ids = make_batch(args.micro_batch, args.sequence_length, args.device, args.seed)
        measurement = {}

        log(f"forward-only: {args.warmup} warmup + {args.steps} timed steps")
        measurement["forward_only"] = runner.run_phase("forward", ids, args.warmup, args.steps)
        log(
            "forward-only tokens/s = "
            f"{measurement['forward_only']['tokens_per_second']}"
        )

        if args.mode != "forward-only":
            log(f"train: {args.warmup} warmup + {args.steps} timed steps")
            measurement["forward_backward_step"] = runner.run_phase(
                "train", ids, args.warmup, args.steps
            )
            log(
                "train tokens/s = "
                f"{measurement['forward_backward_step']['tokens_per_second']}"
            )
            measurement["loss_values"] = [round(value, 5) for value in runner.loss_values]
            measurement["non_finite_losses"] = runner.non_finite_losses
            measurement["non_finite_gradient_steps"] = runner.non_finite_gradient_steps
            if runner.scaler is not None:
                measurement["loss_scaler"] = runner.scaler.report()
            if runner.host_optimizer is not None:
                measurement["host_optimizer"] = runner.host_optimizer.report()

        if getattr(runner, "stream_reader", None) is not None:
            reader = runner.stream_reader
            passes = args.warmup + args.steps
            if args.mode != "forward-only":
                passes *= 2
            measurement["streaming"] = {
                "bytes_read_total_gb": gigabytes(reader.bytes_read),
                "layer_loads": runner.trunk.layers.layer_loads,
                "layer_load_seconds_total": round(runner.trunk.layers.layer_load_seconds, 3),
                "passes_including_warmup": passes,
                "read_rate_mb_per_second_over_timed_wall": round(
                    (reader.bytes_read / (1024.0**2))
                    / max(1e-9, measurement["forward_only"]["wall_seconds"]),
                    1,
                ),
                "note": (
                    "bytes_read_total counts EVERY pass including warmup, while the "
                    "rate divides by the timed wall only, so the rate is an upper "
                    "bound on steady-state; the honest per-pass figure is "
                    "checkpoint_total_size_gb / seconds_per_step_median."
                ),
            }
        sampler.stop()

    kernel = kernel_report_for(model_for_kernel, args.calibration_dir)
    wall_seconds = time.perf_counter() - wall_start

    train_report = measurement.get("forward_backward_step") or {}
    forward_report = measurement.get("forward_only") or {}
    device_count = max(1, len(device_records()))

    receipt: dict[str, Any] = {
        "receipt_schema": RECEIPT_SCHEMA,
        "cell": args.cell,
        "cell_id": args.cell_id,
        "mode": args.mode,
        "model_directory": str(model_dir),
        "dtype": args.dtype,
        "sequence_length": args.sequence_length,
        "micro_batch": args.micro_batch if args.cell != "stream-verify" else None,
        "micro_batch_search": search,
        "accumulation": args.accumulation,
        "tokens_per_optimizer_step": args.accumulation * (args.micro_batch or 0) * args.sequence_length
        if args.cell != "stream-verify"
        else None,
        "loss_chunks": args.loss_chunks,
        "top_k": args.top_k,
        "objective": (
            "token-level cross-entropy on the input ids (a stand-in for the stage-D "
            "KD objective; the KD term is a top-k gather plus a KL over the same "
            "logits and costs the same order or slightly more)"
        ),
        "token_source": (
            "uniform random ids in [0, 248044) -- the tokenizer-defined rows; throughput "
            "at a fixed shape does not depend on token values, packing and padding are not modelled"
        ),
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "grad_clip": args.grad_clip,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "measurement": measurement,
        "model_facts": facts,
        "kernel_path_report": kernel,
        "devices": device_records(),
        "device_count": device_count,
        "resource_sampling": sampler.report(),
        "peak_host_rss_gb": gigabytes((proc_status_kib("VmHWM") or 0) * 1024),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_threads": __import__("torch").get_num_threads(),
            "memory_total_gb": gigabytes((meminfo_kib("MemTotal") or 0) * 1024),
            "page_cache_gb_before": gigabytes((cached_before_kib or 0) * 1024),
            "page_cache_gb_after": gigabytes((meminfo_kib("Cached") or 0) * 1024),
        },
        "versions": package_versions(),
        "nvidia_smi_before": inventory_before,
        "nvidia_smi_after": nvidia_smi_inventory(),
        "wall_seconds_total": round(wall_seconds, 2),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exact_command": exact_command(),
        "probe_sha256": file_sha256(Path(__file__).resolve()),
        "derived_budgets": {
            "forward_only": derived_budgets(forward_report.get("tokens_per_second"), device_count),
            "forward_backward_step": derived_budgets(
                train_report.get("tokens_per_second"), device_count
            ),
        },
        "grades_nothing": (
            "This receipt reports measurements only.  It sets no threshold, passes "
            "and fails nothing, and every budget in it is arithmetic on the measured "
            "rate at THIS shape -- not a promise about another shape or another objective."
        ),
    }
    return receipt


def build_cell_id(args: argparse.Namespace) -> str:
    if args.cell_id:
        return args.cell_id
    parts = [args.cell, f"L{args.sequence_length}"]
    if args.cell != "stream-verify":
        parts.append(f"mb{args.micro_batch}")
        if int(args.accumulation) != 1:
            parts.append(f"acc{args.accumulation}")
    if args.tag:
        parts.append(args.tag)
    return "-".join(str(part) for part in parts)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", choices=CELLS)
    ap.add_argument("--cell-id", default=None, help="receipt basename; derived from the cell when absent")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--mode", default=None, choices=[None, "lora", "full-gpu-master", "full-cpu-adam", "forward-only"])
    ap.add_argument("--dtype", default=None, choices=[None, "bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", default=None, help="e.g. sequential (multi-card cells only)")
    ap.add_argument("--max-memory", default=None, help="e.g. 0=40GiB,1=20GiB")
    ap.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--sequence-length", type=int, default=4096)
    ap.add_argument("--micro-batch", default="1", help="an integer, or 'auto' to search")
    ap.add_argument("--micro-batch-max", type=int, default=32)
    ap.add_argument(
        "--accumulation",
        type=int,
        default=1,
        help="micro-steps per optimizer step; the host-offloaded optimizer's cost amortises over this",
    )
    ap.add_argument("--loss-chunks", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=float, default=128.0)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--initial-loss-scale", type=float, default=2.0**16)
    ap.add_argument("--loss-scale-growth-interval", type=int, default=200)
    ap.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    ap.add_argument("--drop-vision", action="store_true", help="free the (unused) vision tower's weights")
    ap.add_argument(
        "--checkpoint-reentrant",
        action="store_true",
        help="use torch's reentrant checkpoint (no recompute metadata check)",
    )
    ap.add_argument("--pin-host-state", dest="pin_host_state", action="store_true", default=True)
    ap.add_argument("--no-pin-host-state", dest="pin_host_state", action="store_false")
    ap.add_argument("--sample-interval", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "calibration",
        help="where oc_calib.py lives; the kernel-path field is imported from it",
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "tps")
    ap.add_argument("--self-test", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.cell:
        raise SystemExit("--cell is required (or --self-test)")
    args = cell_defaults(args)
    args.cell_id = build_cell_id(args)
    log(f"cell {args.cell_id}: {args.mode} {args.dtype} on {args.device}")
    receipt = run_cell(args)
    receipt["cell_id"] = args.cell_id
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.cell_id}.json"
    text = json.dumps(receipt, indent=1, sort_keys=True)
    out_path.write_text(text, encoding="utf-8")
    log(f"wrote {out_path} ({len(text)} bytes)")
    print(
        json.dumps(
            {
                "cell_id": args.cell_id,
                "forward_only_tokens_per_second": (
                    receipt["measurement"].get("forward_only", {}).get("tokens_per_second")
                ),
                "train_tokens_per_second": (
                    receipt["measurement"].get("forward_backward_step", {}).get("tokens_per_second")
                ),
                "peak_host_rss_gb": receipt["peak_host_rss_gb"],
                "torch_peak_by_device": (
                    receipt["measurement"].get("forward_backward_step", {}).get("torch_memory")
                    or receipt["measurement"].get("forward_only", {}).get("torch_memory")
                ),
            },
            indent=1,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
