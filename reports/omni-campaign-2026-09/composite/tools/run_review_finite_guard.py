#!/usr/bin/env python3
"""run_review_finite_guard.py -- run_omni_frozen_review.py, unchanged, with an overflow/NaN record.

WHY.  The 27B-U parent band has to be decoded on Pascal P40s (sm61), which have no
bf16 path worth using; fp16 is the only 2-byte dtype that fits the 55 GB parent on
three 24 GB cards, and fp16 overflows at 65,504.  The runner checks finiteness only
in the emit-zero floor.  A residual that overflows mid-generation turns the logits
into NaN and `argmax` keeps emitting tokens, so a lane could be decoded from garbage
with nothing in the dossier saying so.  This wrapper records, per decoded sequence,
whether any residual or logit went non-finite and how close the residual came to the
fp16 ceiling.

WHAT IT DOES NOT DO.  It does not modify the runner file (the dossier's
`runner.tool_sha256` stays the runner's own sha, the same bytes that wrote the S3
child dossier), it changes no tensor and gates nothing: the hooks are read-only
`register_forward_hook` callbacks that return None, and the receipt is written beside
the dossier.  Numbers here are diagnostics (owner policy 2026-10-06); a non-finite
logit is recorded as the functional fact it is and the run continues.

HOOK SITES (forward order inside one block, then the head):
  blocks.{k}.norm1  input  = residual entering block k (block 0: the embedding);
  blocks.{k}.norm2  input  = residual after block k's attention (the FFN input, pre-norm);
  norm              input  = final residual;
  lm_head           output = logits BEFORE the ledger-band mask (the mask writes -inf on
                             purpose; those rows are not counted).

SEGMENTS.  Every `SplitDecoder.forward(ids, cache=None)` starts a new sequence (a bank
case's prefill, a probe, a preflight recompute).  Statistics accumulate on-device and
are read once per segment.  The phase (preflight / emit_zero_floor / bank / smoke) and,
for bank and smoke, the case id come from wrapping the runner's own phase functions,
which `run()` resolves from module globals at call time.

HOST-STAGED MOVES (`--host-staged-moves`, v1.1.0).  On server `ai` the P40s report peer
access (`can_device_access_peer` True, topology PIX/PHB) but every peer copy arrives
corrupted: 2026-10-09, torch 2.6 cu118, CUDA_VISIBLE_DEVICES=1,2,3, a [1,38,5120] tensor
moved 0->1, 1->2, 0->2 mismatched in EVERY element over 20 copies, fp16 and fp32 alike.
The first fp16 smoke therefore read garbage at the first block of each later card (inf at
blocks.16/40.norm1 input, zero terminal states).  With the flag, `torch.Tensor.to` routes
a CUDA->different-CUDA move through host memory (`.to("cpu")` then the original call);
every other `.to` is the original method.  The bytes that arrive are the bytes that left
(checked at start by `cross_device_integrity`, recorded in the receipt with the raw peer
result beside it).  The runner's own code is not touched.

USAGE: exactly the runner's argv, plus optional `--guard-out PATH`
(default `<--out>.finite_guard.json`) and `--host-staged-moves`.  The receipt is rewritten
atomically after every segment, so a long run can be polled.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_omni_frozen_review.py"
TOOL_NAME = "run_review_finite_guard"
TOOL_VERSION = "1.2.0"
# 1.2.0 (2026-10-13, S8-TEXT): charted blocks (OMNI1) carry no norm1/norm2 and are the identity on text in the
#   runner's split decoder, so they get no hook site; their indices are recorded as `charted_blocks_not_hooked`.
GUARD_SCHEMA = "chimera-omni-review-finite-guard-v1"
FP16_MAX = 65504.0
SCAN_CHUNK = 1 << 26


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runner() -> Any:
    if "run_omni_frozen_review" in sys.modules and Path(getattr(sys.modules["run_omni_frozen_review"], "__file__", "")).resolve() == RUNNER:
        return sys.modules["run_omni_frozen_review"]
    spec = importlib.util.spec_from_file_location("run_omni_frozen_review", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_omni_frozen_review"] = module
    spec.loader.exec_module(module)
    return module


def split_guard_args(argv: list[str]) -> tuple[list[str], str | None, bool]:
    """Strip `--guard-out X` / `--guard-out=X` / `--host-staged-moves`; the rest goes to the runner untouched."""

    rest: list[str] = []
    guard_out = None
    staged = False
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--host-staged-moves":
            staged = True
            index += 1
            continue
        if token == "--guard-out":
            if index + 1 >= len(argv):
                raise SystemExit("--guard-out needs a path")
            guard_out = argv[index + 1]
            index += 2
            continue
        if token.startswith("--guard-out="):
            guard_out = token.split("=", 1)[1]
            index += 1
            continue
        rest.append(token)
        index += 1
    return rest, guard_out, staged


def target_device(torch: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The device a `Tensor.to(...)` call names, or None when it names only a dtype."""

    candidate = kwargs.get("device")
    if candidate is None and args:
        first = args[0]
        if isinstance(first, torch.Tensor):
            return first.device
        if isinstance(first, (str, torch.device, int)) and not isinstance(first, bool):
            candidate = first
    if candidate is None:
        return None
    if isinstance(candidate, int):
        return torch.device("cuda", candidate)
    return torch.device(candidate)


class HostStagedMoves:
    """Route CUDA -> different-CUDA `Tensor.to` through host memory; count what was routed."""

    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.original = torch.Tensor.to
        self.moves = 0
        self.bytes = 0
        self.installed = False

    def _same(self, source: Any, target: Any) -> bool:
        index = target.index if target.index is not None else self.torch.cuda.current_device()
        return int(source.index) == int(index)

    def install(self) -> None:
        torch = self.torch
        original = self.original
        stager = self

        def to(self_tensor: Any, *args: Any, **kwargs: Any) -> Any:
            if self_tensor.is_cuda:
                target = target_device(torch, args, kwargs)
                if target is not None and target.type == "cuda" and not stager._same(self_tensor.device, target):
                    stager.moves += 1
                    stager.bytes += int(self_tensor.numel() * self_tensor.element_size())
                    return original(original(self_tensor, "cpu"), *args, **kwargs)
            return original(self_tensor, *args, **kwargs)

        torch.Tensor.to = to
        self.installed = True

    def uninstall(self) -> None:
        if self.installed:
            self.torch.Tensor.to = self.original
            self.installed = False


def cross_device_integrity(staged: "HostStagedMoves | None") -> dict[str, Any]:
    """Move fixed random tensors between every visible CUDA pair, raw peer and (if on) staged; count mismatches."""

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return {"checked": False, "why": "fewer than two CUDA devices visible"}
    raw_to = staged.original if staged is not None else torch.Tensor.to
    generator = torch.Generator().manual_seed(20261009)
    rows = []
    count = torch.cuda.device_count()
    for a in range(count):
        for b in range(count):
            if a == b:
                continue
            for dtype in (torch.float16, torch.float32):
                host = torch.randn((1, 38, 5120), generator=generator).to(dtype)
                source = raw_to(host, f"cuda:{a}")
                raw = raw_to(source, f"cuda:{b}")
                row = {
                    "from": a, "to": b, "dtype": str(dtype), "elements": int(host.numel()),
                    "peer_copy_mismatched": int((raw_to(raw, "cpu") != host).sum()),
                    "peer_access_reported": bool(torch.cuda.can_device_access_peer(a, b)),
                }
                if staged is not None and staged.installed:
                    moved = source.to(f"cuda:{b}")
                    row["staged_copy_mismatched"] = int((raw_to(moved, "cpu") != host).sum())
                rows.append(row)
    return {
        "checked": True,
        "pairs": rows,
        "peer_copies_corrupt": any(r["peer_copy_mismatched"] for r in rows),
        "staged_copies_exact": (all(r["staged_copy_mismatched"] == 0 for r in rows) if staged is not None and staged.installed else None),
    }


def runner_out_path(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--out" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--out="):
            return token.split("=", 1)[1]
    return None


class FiniteGuard:
    """Read-only residual/logit statistics over one SplitDecoder's model."""

    def __init__(self, path: Path, *, argv: list[str], runner: Any) -> None:
        self.path = Path(path)
        self.argv = list(argv)
        self.runner = runner
        self.torch = None
        self.sites: list[str] = []
        self.site_device: dict[str, str] = {}
        self.acc_max: dict[str, Any] = {}
        self.acc_nonfinite: dict[str, Any] = {}
        self.global_max: dict[str, float] = {}
        self.global_nonfinite: dict[str, int] = {}
        self.handles: list[Any] = []
        self.segments: list[dict[str, Any]] = []
        self.phase = "unattributed"
        self.phase_rows: list[str] | None = None
        self.phase_index = 0
        self.open_segment: dict[str, Any] | None = None
        self.parameter_scan: dict[str, Any] | None = None
        self.placement: dict[str, Any] | None = None
        self.dtype: str | None = None
        self.started = time.time()
        self.exit_code: int | None = None
        self.attached = False
        self.staged: Any = None
        self.integrity: dict[str, Any] | None = None

    # ---- attachment ---------------------------------------------------------
    def attach(self, model: Any, placement: dict[str, Any]) -> None:
        import torch

        self.torch = torch
        self.placement = {"split": placement.get("split"), "devices": placement.get("devices"), "head_device": placement.get("head_device")}
        head = str(placement["head_device"])
        self.charted_not_hooked = []
        for index, block in enumerate(model.blocks):
            device = str(placement["block_device"][index])
            if getattr(block, "charted", None) is not None:
                self.charted_not_hooked.append(index)
                continue
            self._site(f"blocks.{index}.norm1.in", block.norm1, device, use_output=False)
            self._site(f"blocks.{index}.norm2.in", block.norm2, device, use_output=False)
        self._site("norm.in", model.norm, head, use_output=False)
        self._site("lm_head.out", model.lm_head, head, use_output=True)
        first = next(iter(model.parameters()))
        self.dtype = str(first.dtype)
        self.parameter_scan = self._scan_parameters(model)
        self.attached = True
        self.write()

    def _site(self, name: str, module: Any, device: str, *, use_output: bool) -> None:
        torch = self.torch
        self.sites.append(name)
        self.site_device[name] = device
        self.acc_max[name] = torch.zeros((), dtype=torch.float32, device=device)
        self.acc_nonfinite[name] = torch.zeros((), dtype=torch.int64, device=device)
        self.global_max[name] = 0.0
        self.global_nonfinite[name] = 0

        def hook(_module: Any, inputs: tuple[Any, ...], output: Any, _name: str = name, _use_output: bool = use_output) -> None:
            tensor = output if _use_output else inputs[0]
            self._update(_name, tensor)

        self.handles.append(module.register_forward_hook(hook))

    def _update(self, name: str, tensor: Any) -> None:
        torch = self.torch
        with torch.no_grad():
            t = tensor.detach()
            finite = torch.isfinite(t)
            nonfinite = (~finite).sum()
            magnitude = t.abs().amax().float()
            magnitude = torch.nan_to_num(magnitude, nan=float("inf"))
            acc = self.acc_max[name]
            if acc.device != magnitude.device:
                magnitude = magnitude.to(acc.device)
                nonfinite = nonfinite.to(acc.device)
            torch.maximum(acc, magnitude, out=acc)
            self.acc_nonfinite[name].add_(nonfinite)

    def _scan_parameters(self, model: Any) -> dict[str, Any]:
        """Non-finite elements in the LOADED parameters (a bf16 value beyond fp16 range casts to inf)."""

        torch = self.torch
        bad: dict[str, int] = {}
        largest = 0.0
        largest_name = None
        count = 0
        seen: set[int] = set()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.data_ptr() in seen:
                    continue
                seen.add(parameter.data_ptr())
                flat = parameter.detach().reshape(-1)
                nonfinite = 0
                peak = 0.0
                for start in range(0, flat.numel(), SCAN_CHUNK):
                    piece = flat[start : start + SCAN_CHUNK]
                    nonfinite += int((~torch.isfinite(piece)).sum().item())
                    peak = max(peak, float(torch.nan_to_num(piece.abs().amax().float(), nan=float("inf")).item()))
                count += flat.numel()
                if nonfinite:
                    bad[name] = nonfinite
                if peak > largest:
                    largest, largest_name = peak, name
        return {
            "elements_scanned": int(count),
            "tensors_with_nonfinite": bad,
            "nonfinite_elements": int(sum(bad.values())),
            "max_abs_parameter": largest,
            "max_abs_parameter_tensor": largest_name,
            "note": "scanned once after load, in the run dtype; a bf16 weight beyond 65,504 would show here as inf under fp16",
        }

    # ---- segments -------------------------------------------------------------
    def begin_segment(self, input_tokens: int) -> None:
        self.flush()
        case_id = None
        if self.phase_rows is not None and self.phase_index < len(self.phase_rows):
            case_id = self.phase_rows[self.phase_index]
        self.open_segment = {
            "segment": len(self.segments),
            "phase": self.phase,
            "phase_index": self.phase_index,
            "case_id": case_id,
            "input_tokens": int(input_tokens),
            "forward_calls": 0,
            "started_unix": time.time(),
        }
        self.phase_index += 1

    def count_forward(self) -> None:
        if self.open_segment is not None:
            self.open_segment["forward_calls"] += 1

    def flush(self) -> None:
        if self.open_segment is None or self.torch is None:
            return
        torch = self.torch
        segment = self.open_segment
        self.open_segment = None
        maxima = {name: float(self.acc_max[name].item()) for name in self.sites}
        nonfinite = {name: int(self.acc_nonfinite[name].item()) for name in self.sites}
        for name in self.sites:
            self.acc_max[name].zero_()
            self.acc_nonfinite[name].zero_()
            self.global_max[name] = max(self.global_max[name], maxima[name])
            self.global_nonfinite[name] += nonfinite[name]
        residual_sites = [name for name in self.sites if name != "lm_head.out"]
        worst_site = max(residual_sites, key=lambda n: maxima[n])
        first_bad = next((name for name in self.sites if nonfinite[name] > 0), None)
        wall = time.time() - segment.pop("started_unix")
        segment.update(
            {
                "wall_seconds": round(wall, 3),
                "decode_steps": max(0, int(segment["forward_calls"]) - 1),
                "seconds_per_decode_step": round(wall / max(1, int(segment["forward_calls"]) - 1), 4) if int(segment["forward_calls"]) > 1 else None,
                "residual_max_abs": maxima[worst_site],
                "residual_max_abs_site": worst_site,
                "fp16_headroom_ratio": (FP16_MAX / maxima[worst_site]) if maxima[worst_site] > 0 else None,
                "residual_nonfinite_elements": int(sum(nonfinite[n] for n in residual_sites)),
                "logits_nonfinite_elements": int(nonfinite["lm_head.out"]),
                "logits_max_abs": maxima["lm_head.out"],
                "first_nonfinite_site": first_bad,
                "finite": first_bad is None,
            }
        )
        self.segments.append(segment)
        self.write()

    # ---- receipt --------------------------------------------------------------
    def body(self) -> dict[str, Any]:
        torch = self.torch
        peaks = []
        if torch is not None and torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                try:
                    peaks.append({"device": f"cuda:{index}", "name": torch.cuda.get_device_name(index), "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)), "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(index))})
                except RuntimeError:
                    pass
        bad_segments = [s for s in self.segments if not s["finite"]]
        residual_sites = [n for n in self.sites if n != "lm_head.out"]
        run_worst = max(residual_sites, key=lambda n: self.global_max[n]) if residual_sites else None
        bank = [s for s in self.segments if s["phase"] == "bank"]
        steps = sum(int(s["decode_steps"]) for s in bank)
        bank_wall = sum(float(s["wall_seconds"]) for s in bank)
        return {
            "schema": GUARD_SCHEMA,
            "tool": TOOL_NAME,
            "tool_version": TOOL_VERSION,
            "tool_sha256": file_sha256(Path(__file__)),
            "runner": str(RUNNER),
            "runner_sha256": file_sha256(RUNNER),
            "runner_modified": False,
            "argv": self.argv,
            "dtype": self.dtype,
            "placement": self.placement,
            "torch_version": getattr(torch, "__version__", None),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started_unix": self.started,
            "updated_unix": time.time(),
            "exit_code": self.exit_code,
            "attached": self.attached,
            "fp16_max": FP16_MAX,
            "parameter_scan": self.parameter_scan,
            "host_staged_moves": None if self.staged is None else {"installed": self.staged.installed, "moves": self.staged.moves, "bytes": self.staged.bytes},
            "cross_device_integrity": self.integrity,
            "charted_blocks_not_hooked": getattr(self, "charted_not_hooked", []),
            "summary": {
                "segments": len(self.segments),
                "segments_nonfinite": len(bad_segments),
                "first_nonfinite_segment": None if not bad_segments else {k: bad_segments[0][k] for k in ("segment", "phase", "case_id", "first_nonfinite_site")},
                "run_residual_max_abs": None if run_worst is None else self.global_max[run_worst],
                "run_residual_max_abs_site": run_worst,
                "run_fp16_headroom_ratio": (FP16_MAX / self.global_max[run_worst]) if run_worst and self.global_max[run_worst] > 0 else None,
                "bank_decode_steps": steps,
                "bank_wall_seconds": round(bank_wall, 3),
                "bank_seconds_per_decode_step": round(bank_wall / steps, 4) if steps else None,
                "definition": "finite = no non-finite element at any hooked residual site and none in the pre-mask logits, over every forward call of the segment",
            },
            "site_max_abs": {name: self.global_max[name] for name in self.sites},
            "site_nonfinite_elements": {name: self.global_nonfinite[name] for name in self.sites if self.global_nonfinite[name]},
            "segments": self.segments,
            "cuda_peaks": peaks,
            "policy_note": "diagnostic record; gates nothing; the runner and every tensor are untouched",
        }

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.body(), indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def install(runner: Any, guard: FiniteGuard) -> None:
    """Wrap SplitDecoder.__init__/forward and the three phase functions; nothing else."""

    decoder_class = runner.SplitDecoder
    original_init = decoder_class.__init__
    original_forward = decoder_class.forward

    def guarded_init(self: Any, model: Any, cfg: Any, placement: dict[str, Any], **kwargs: Any) -> None:
        original_init(self, model, cfg, placement, **kwargs)
        if not guard.attached:
            guard.attach(model, placement)

    def guarded_forward(self: Any, input_ids: Any, cache: Any = None, **kwargs: Any) -> Any:
        if cache is None:
            guard.begin_segment(int(input_ids.shape[-1]))
        guard.count_forward()
        return original_forward(self, input_ids, cache, **kwargs)

    decoder_class.__init__ = guarded_init
    decoder_class.forward = guarded_forward

    def phase(name: str, function: Any, *, rows_from: str | None = None) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            guard.flush()
            label = kwargs.get("label")
            guard.phase = ("smoke" if label == "smoke" else "bank") if name == "generate" else name
            guard.phase_index = 0
            guard.phase_rows = None
            if rows_from is not None:
                rows = args[2] if len(args) > 2 else kwargs.get(rows_from)
                guard.phase_rows = [str(row["id"]) for row in rows] if rows is not None else None
            try:
                return function(*args, **kwargs)
            finally:
                guard.flush()
                guard.phase = "unattributed"
                guard.phase_rows = None

        return wrapped

    runner.decode_path_preflight_chimera = phase("preflight", runner.decode_path_preflight_chimera)
    runner.emit_zero_floor_chimera = phase("emit_zero_floor", runner.emit_zero_floor_chimera)
    runner.generate_greedy_chimera = phase("generate", runner.generate_greedy_chimera, rows_from="rows")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    runner_argv, guard_out, staged_moves = split_guard_args(argv)
    out = runner_out_path(runner_argv)
    if guard_out is None:
        if out is None:
            raise SystemExit("run_review_finite_guard: the runner needs --out (and the guard writes beside it)")
        guard_out = out + ".finite_guard.json"
    runner = load_runner()
    guard = FiniteGuard(Path(guard_out), argv=argv, runner=runner)
    install(runner, guard)
    if staged_moves:
        guard.staged = HostStagedMoves()
        guard.staged.install()
    try:
        guard.integrity = cross_device_integrity(guard.staged)
    except RuntimeError as exc:  # recorded, never fatal
        guard.integrity = {"checked": False, "why": f"{type(exc).__name__}: {exc}"}
    guard.write()
    code = 1
    try:
        code = runner.main(runner_argv)
        return code
    finally:
        guard.flush()
        guard.exit_code = code
        if out and Path(out).is_file():
            guard_body_extra = {"dossier": out, "dossier_file_sha256": file_sha256(Path(out))}
        else:
            guard_body_extra = {"dossier": out, "dossier_file_sha256": None}
        guard.write()
        body = json.loads(Path(guard_out).read_text(encoding="utf-8"))
        body.update(guard_body_extra)
        Path(guard_out).write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
        print(json.dumps({"finite_guard": str(guard_out), "segments": body["summary"]["segments"], "segments_nonfinite": body["summary"]["segments_nonfinite"], "run_residual_max_abs": body["summary"]["run_residual_max_abs"], "run_fp16_headroom_ratio": body["summary"]["run_fp16_headroom_ratio"], "exit_code": code}, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
