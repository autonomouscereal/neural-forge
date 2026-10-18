#!/usr/bin/env python
"""S6H: MiniMax H3 DiT block I/O capture inside OUR OWN docker run of the r5 runtime image.

No ComfyUI server, no catalog path (D12), no sampler: the packed DiT is loaded with
``comfy.sd.load_diffusion_model``, made fully resident on the one visible GPU, and driven
one forward at a time on fixture latents. Forward hooks on ``blocks[i]`` record what a
transplant / coreset needs, following ``mediafold.c16_capture`` (row provenance, clip-
aligned windows, receptive_field_notes)::

    docker run --rm --name s6h_h3_capture --user $(id -u):$(id -g) \\
      --gpus '"device=GPU-f5a09535-fc85-a027-50f7-734b344747dd"' \\
      -e NVIDIA_VISIBLE_DEVICES=GPU-f5a09535-fc85-a027-50f7-734b344747dd -e CUDA_VISIBLE_DEVICES=0 \\
      -e HOME=/tmp \\
      --entrypoint /opt/environments/python/comfyui/bin/python \\
      -v /home/cereal/comfyui/models:/opt/ComfyUI/models:ro \\
      -v /home/cereal/omni-campaign:/work -v /mnt/lexar:/lexar \\
      comfyui:codex-minimax-h3-runtime-env-r5-20260808 \\
      /work/media/c16_h3_capture.py \\
        --model /opt/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors \\
        --out /lexar/omni-campaign/media/h3_blocks --blocks 25 --rung latents \\
        --video-pairs /work/media/pairs/h3v_g4_64_merged --audio-pairs /work/media/pairs/h3a_d_ctx0_merged \\
        --steps 4 --allow-deviations --receipt-copy /work/composite/media/h3_blocks \\
        --expect-device-name "Quadro RTX 8000"

``--user $(id -u):$(id -g)`` keeps every file the run writes (captures on /lexar, the
receipt copy under $OC) owned by the invoking user; a root-owned receipt cannot be
replaced in place by a later cereal-owned process (mirror sync, a rerun outside docker,
the Merge phase). ``-e HOME=/tmp`` gives the non-root process a writable home for
comfy's caches. The run REFUSES when ``--out`` already holds a ``capture_receipt.json``
unless ``--overwrite`` is passed (a rerun would otherwise append a second ``start`` to
``progress.jsonl`` and silently replace ``step*/L{i}.safetensors``), and it hashes the
source file by default (``--no-hash-source`` to skip).

Stimulus regime (see ``mediafold.c16_h3.stimulus_deviations``): ``--audio-frames -1``
(default) feeds each forward the audio span the video's model time covers
(``matched_audio_frames``); ``--audio-windows 0`` (default) cuts as many NON-overlapping
matched windows from each audio clip as fit, each a forward of its own with the PARENT
clip in the row provenance; ``--video-frames leading-run`` (default) keeps a clip's
leading run of consecutive latent frame ids so there is no time hole. Every remaining
departure from the parent's regime (a clip that does not start at latent frame 0,
gaussian text in place of the parent's Qwen encoder states, C15 procedural clips in
place of the pending S1 media stimuli) is a NAMED deviation: the run refuses unless
``--allow-deviations`` is passed, and then writes the list into the receipt. Pass
``--text-states FILE`` (safetensors ``text_states [L, 5120]``, e.g. the parent's
empty-prompt encoder output) to drive the real text path once it is on the box.

Per block and per sigma step one file ``step{s}/L{i}.safetensors`` holds, row-concatenated
over the step's forwards: ``x_in`` / ``x_out`` (the block's residual stream in and out),
``mlp_in`` (the adaLN-modulated norm2 output the MLP reads), ``mlp_out`` (the MLP's
pre-gate write), ``attn_out`` (the attention's pre-gate write), with per-row
``token_kind`` / ``token_frame`` / ``token_clip`` / ``forward_index`` and per-forward
``sigma_video`` / ``sigma_audio`` / ``temb`` / ``adaln`` (the six modulation rows) /
``mod_segments``. ``blocks/L{i}.safetensors`` carries the block's own tensors verbatim so
the coreset can run outside the image.

Two rungs, both procedural: ``random`` (seeded gaussian latents on a chosen grid; proves
the plumbing) and ``latents`` (real H3 VAE latents of procedural clips replayed from the
C15 pair corpora, one video clip + one audio clip per forward, paired one-to-one so no
clip appears in two forwards).

Compute dtype defaults to what comfy itself picks for the visible card
(``unet_manual_cast``: fp32 on sm75), so the captured rows are the parent's behaviour on
THIS GPU. Every captured tensor is checked finite before it is stored; a non-finite value
is a refusal, not a warning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import torch  # noqa: E402

from mediafold import receipts  # noqa: E402
from mediafold.c16_h3 import (  # noqa: E402
    AUDIO_CLIP_OFFSET,
    BLOCK_TENSOR_SUFFIXES,
    KIND_AUDIO,
    KIND_TEXT,
    KIND_VIDEO,
    TEXT_SOURCE_GAUSSIAN,
    TEXT_SOURCE_STATES,
    VIDEO_FRAME_POLICIES,
    H3Stimulus,
    audio_clips_from_pairs,
    audio_parent_of,
    block_tensor_names,
    block_weights_path,
    build_stimulus,
    capture_path,
    matched_audio_frames,
    mlp_reference,
    receptive_field_notes,
    shifted_sigma_grid,
    stimulus_deviations,
    uniform_sigma_grid,
    video_clips_from_pairs,
)

COMFY_ROOTS = ("/opt/ComfyUI", "/workspace/ComfyUI")
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _log(progress: Path | None, record: dict[str, Any]) -> None:
    line = json.dumps({"t": time.time(), **record}, sort_keys=False)
    print(line, flush=True)
    if progress is not None:
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def sha256_file(path: str | Path, chunk: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def cuda_memory_block(device: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    index = 0 if device == "cuda" else int(device.split(":")[1])
    free, total = torch.cuda.mem_get_info(index)
    return {
        "device": device,
        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(index)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


# --------------------------------------------------------------------------------------
# comfy loading (container only)
# --------------------------------------------------------------------------------------


def import_comfy(device: str) -> dict[str, Any]:
    for root in (os.environ.get("COMFYUI_ROOT", ""), *COMFY_ROOTS):
        if root and os.path.isdir(os.path.join(root, "comfy")) and root not in sys.path:
            sys.path.insert(0, root)
    import comfy.options  # noqa: F401
    from comfy.cli_args import args as comfy_args

    if device == "cpu":
        comfy_args.cpu = True
    import comfy.model_management as model_management
    import comfy.sd  # noqa: F401
    import comfy.utils  # noqa: F401

    torch_device = model_management.get_torch_device()
    return {
        "torch_device": str(torch_device),
        "comfy_root": next((r for r in sys.path if os.path.isdir(os.path.join(r, "comfy", "ldm", "minimax"))), None),
        "should_use_bf16": bool(model_management.should_use_bf16(torch_device)),
        "should_use_fp16": bool(model_management.should_use_fp16(torch_device, prioritize_performance=False)),
        "unet_manual_cast_for_bf16": str(model_management.unet_manual_cast(torch.bfloat16, torch_device, [torch.bfloat16, torch.float32])),
        "free_memory_bytes": int(model_management.get_free_memory(torch_device)),
        "minimum_inference_memory_bytes": int(model_management.minimum_inference_memory()),
    }


def load_h3(path: str, *, weight_dtype: torch.dtype) -> tuple[Any, Any, dict[str, Any]]:
    import comfy.model_management as model_management
    import comfy.sd

    started = time.time()
    patcher = comfy.sd.load_diffusion_model(path, model_options={"dtype": weight_dtype})
    model = patcher.model
    dit = model.diffusion_model
    config = model.model_config
    load_info = {
        "path": path,
        "bytes": int(Path(path).stat().st_size),
        "model_class": type(model).__name__,
        "diffusion_model_class": type(dit).__name__,
        "unet_config": {k: (v if isinstance(v, (int, float, str, bool, list)) else str(v)) for k, v in getattr(config, "unet_config", {}).items()},
        "weight_dtype": str(getattr(model, "get_dtype", lambda: weight_dtype)()),
        "manual_cast_dtype": str(getattr(config, "manual_cast_dtype", None)),
        "custom_operations": str(getattr(config, "custom_operations", None)),
        "model_size_bytes": int(patcher.model_size()),
        "parameters": int(sum(int(p.numel()) for p in dit.parameters())),
        "num_blocks": len(dit.blocks),
        "load_seconds": None,
    }
    model_management.load_models_gpu([patcher], memory_required=float(patcher.model_size()), force_full_load=True)
    load_info["load_seconds"] = time.time() - started
    return patcher, dit, load_info


def residency(dit: Any, device: str) -> dict[str, Any]:
    """Every parameter and buffer on the card, or the run refuses."""

    off: list[str] = []
    dtypes: dict[str, int] = {}
    total = 0
    for name, tensor in list(dit.named_parameters()) + list(dit.named_buffers()):
        total += int(tensor.numel())
        dtypes[str(tensor.dtype)] = dtypes.get(str(tensor.dtype), 0) + int(tensor.numel())
        if tensor.device.type != "cuda" and device != "cpu":
            off.append(name)
    return {"resident": not off, "off_device": off[:20], "off_device_count": len(off), "numel": total, "dtype_numel": dtypes}


# --------------------------------------------------------------------------------------
# hooks
# --------------------------------------------------------------------------------------


class H3BlockTap:
    """Forward hooks over ``dit.blocks[i]`` and its sub-writes.

    The H3 block accumulates its residual writes IN PLACE (``_mod_gate`` does
    ``x.addcmul_``), so the block's positional input is the block's output by the time a
    forward hook runs: ``x_in`` is cloned in a PRE hook.
    """

    def __init__(self, dit: Any, block_ids: list[int]) -> None:
        self.dit = dit
        self.block_ids = [int(v) for v in block_ids]
        self.records: dict[int, dict[str, Any]] = {index: {} for index in self.block_ids}
        self._handles: list[Any] = []

    def __enter__(self) -> "H3BlockTap":
        for index in self.block_ids:
            block = self.dit.blocks[index]
            self._handles.append(block.register_forward_pre_hook(self._block_pre(index), with_kwargs=True))
            self._handles.append(block.register_forward_hook(self._block_post(index)))
            self._handles.append(block.attn.register_forward_hook(self._store(index, "attn_out")))
            self._handles.append(block.mlp.register_forward_pre_hook(self._mlp_pre(index)))
            self._handles.append(block.mlp.register_forward_hook(self._store(index, "mlp_out")))
            self._handles.append(block.adaln_proj.register_forward_hook(self._adaln(index)))
        return self

    def __exit__(self, *exception: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset(self) -> None:
        for index in self.block_ids:
            self.records[index] = {}

    def _block_pre(self, index: int):
        def hook(module, args, kwargs):
            store = self.records[index]
            store["x_in"] = args[0].detach().clone()
            store["temb"] = args[1].detach().clone()
            store["mod_segments"] = [(int(a), int(b), int(row)) for a, b, row in args[2]]

        return hook

    def _block_post(self, index: int):
        def hook(module, args, output):
            self.records[index]["x_out"] = output.detach().clone()

        return hook

    def _store(self, index: int, name: str):
        def hook(module, args, output):
            self.records[index][name] = output.detach().clone()

        return hook

    def _mlp_pre(self, index: int):
        def hook(module, args):
            self.records[index]["mlp_in"] = args[0].detach().clone()

        return hook

    def _adaln(self, index: int):
        def hook(module, args, output):
            self.records[index]["adaln"] = torch.stack([o.detach().clone() for o in output], dim=0)

        return hook


def run_forward(dit: Any, stimulus: H3Stimulus, *, device: str, compute_dtype: torch.dtype) -> list[torch.Tensor]:
    video = stimulus.video.to(device=device, dtype=compute_dtype)
    audio = stimulus.audio.to(device=device, dtype=compute_dtype)
    text = stimulus.text.to(device=device, dtype=compute_dtype)
    timestep = torch.tensor([stimulus.sigma_video * 1000.0], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = dit(x=[video, audio], timestep=timestep, context=text, transformer_options={}, minimax_payload={})
    return [o.detach() for o in out]


def finite_report(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    finite = torch.isfinite(value)
    return {
        "name": name,
        "finite": bool(finite.all()),
        "non_finite": int((~finite).sum().item()),
        "max_abs": float(value[finite].abs().max().item()) if bool(finite.any()) else None,
    }


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def parse_grid(text: str) -> tuple[int, int, int]:
    parts = [int(v) for v in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("grid is T,H,W")
    return parts[0], parts[1], parts[2]


def parse_sigmas(text: str) -> list[float]:
    """``--sigmas 0.2,0.5,0.8`` -> an explicit ascending grid in (0, 1)."""

    values = [float(v) for v in text.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--sigmas needs at least one value")
    if any(not 0.0 < v < 1.0 for v in values):
        raise argparse.ArgumentTypeError("every sigma must be in (0, 1)")
    if values != sorted(values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("sigmas must be strictly ascending")
    return values


def sigma_plan(arguments: argparse.Namespace) -> tuple[list[float], str]:
    """The sigmas this run captures at and the NAME of the grid, for the receipt."""

    if getattr(arguments, "sigmas", None):
        return list(arguments.sigmas), "explicit"
    if arguments.sigma_grid == "shifted":
        return shifted_sigma_grid(arguments.steps), "shifted"
    return uniform_sigma_grid(arguments.steps), "uniform"


def load_text_states(path: str | None) -> torch.Tensor | None:
    """``--text-states`` safetensors: key ``text_states`` [L, 5120] (or [1, L, 5120])."""

    if not path:
        return None
    state = receipts.read_tensors(path)
    if "text_states" not in state:
        raise SystemExit(f"{path}: no 'text_states' tensor (keys: {sorted(state)[:8]})")
    return state["text_states"].to(torch.float32)


def build_stimuli(arguments: argparse.Namespace, sigmas: list[float]) -> tuple[list[list[H3Stimulus]], dict[str, Any]]:
    """Per step, one stimulus per clip pair (latents) or per random forward.

    ``notes["deviations"]`` names every departure from the parent's operating regime /
    the spec (``mediafold.c16_h3.stimulus_deviations``); the caller refuses on a non-empty
    list unless ``--allow-deviations`` was passed.
    """

    plan: list[list[H3Stimulus]] = []
    notes: dict[str, Any] = {"rung": arguments.rung}
    text_states = load_text_states(getattr(arguments, "text_states", None))
    text_source = TEXT_SOURCE_STATES if text_states is not None else TEXT_SOURCE_GAUSSIAN
    text_common = {"text_len": int(arguments.text_len), "text_states": text_states}
    if arguments.rung == "latents":
        from mediafold.c15_pairs import load_pair_set

        video_pairs = load_pair_set(arguments.video_pairs)
        audio_pairs = load_pair_set(arguments.audio_pairs)
        video_clips, video_notes = video_clips_from_pairs(video_pairs, frames=arguments.video_frames)
        video_ids = sorted(video_clips)
        latent_frames = int(video_clips[video_ids[0]].shape[1])
        matched = matched_audio_frames(latent_frames)
        if int(arguments.audio_frames) < 0:
            audio_frames, audio_span = matched, "matched"
        elif int(arguments.audio_frames) == 0:
            audio_frames, audio_span = 0, "whole_clip"
        else:
            audio_frames, audio_span = int(arguments.audio_frames), f"explicit:{int(arguments.audio_frames)}"
        windows = int(arguments.audio_windows) if audio_frames > 0 else 1
        audio_clips, audio_notes = audio_clips_from_pairs(audio_pairs, frames=audio_frames, windows=windows)
        audio_ids = sorted(audio_clips)
        parent_of = audio_notes["parent_of"]
        pairable = min(len(video_ids), len(audio_ids))
        offset = int(arguments.clip_offset)
        count = int(arguments.clips) if arguments.clips > 0 else pairable - offset
        if offset + count > pairable:
            raise SystemExit(f"--clip-offset {offset} + --clips {count} exceeds the {pairable} one-to-one clip pairs")
        pairs = [(video_ids[offset + k], audio_ids[offset + k]) for k in range(count)]
        audio_frames_used = int(audio_notes["frames_per_clip"][0])
        deviations = stimulus_deviations(
            frame_alignment_used=video_notes["frame_alignment"],
            audio_frames_used=audio_frames_used,
            matched_audio_frames_for_video=matched,
            text_source=text_source,
        )
        notes.update(
            {
                "video_pairs": str(arguments.video_pairs),
                "video_pairs_geometry": video_pairs.geometry,
                "video_clips": video_notes,
                "audio_pairs": str(arguments.audio_pairs),
                "audio_pairs_geometry": audio_pairs.geometry,
                "audio_clips": {k: v for k, v in audio_notes.items() if k != "parent_of"},
                "audio_frames_used": audio_frames_used,
                "matched_audio_frames_for_video": matched,
                "pairing": "one_to_one",
                "pairable_clip_pairs": pairable,
                "clip_offset": offset,
                "clip_pairs_used": count,
                "clip_pairs": pairs,
                "audio_parent_of_pairs": {str(a): audio_parent_of(a, parent_of) for _, a in pairs},
                "stimulus_arms": {
                    "video_frames": arguments.video_frames,
                    "audio_span": audio_span,
                    "audio_windows_per_parent": audio_notes["windows_per_parent"],
                    "text": "states" if text_states is not None else "gaussian",
                },
                "deviations": deviations,
                "note": (
                    "the video and audio corpora are independent procedural clips; a forward joins "
                    "video clip k with audio window k (its parent audio clip is the row provenance), each "
                    "video clip used in exactly one forward per sigma and each audio parent on one side "
                    "of an audio-parent-grouped holdout"
                ),
            }
        )
        for step, sigma in enumerate(sigmas):
            row: list[H3Stimulus] = []
            for k, (v, a) in enumerate(pairs):
                seed = int(arguments.seed) + 1000 * step + k
                row.append(
                    build_stimulus(
                        video_clips[v], audio_clips[a], sigma=sigma, seed=seed, clip_id=int(v),
                        audio_clip_id=int(a), audio_parent_clip_id=audio_parent_of(a, parent_of),
                        rung="latents", **text_common,
                    )
                )
            plan.append(row)
    else:
        t, h, w = arguments.random_grid
        audio_frames = int(arguments.audio_frames) if arguments.audio_frames > 0 else matched_audio_frames(t)
        count = int(arguments.clips) if arguments.clips > 0 else 4
        deviations = stimulus_deviations(
            frame_alignment_used=None, audio_frames_used=audio_frames,
            matched_audio_frames_for_video=matched_audio_frames(t), text_source=text_source,
        )
        deviations.insert(0, {"name": "random_rung", "what": "seeded gaussian latents, no clip content", "why_it_matters": "proves the plumbing only", "arm": "stimuli"})
        notes.update({"random_grid_thw": [t, h, w], "audio_frames": audio_frames, "forwards_per_step": count, "deviations": deviations,
                      "stimulus_arms": {"video_frames": "random", "audio_span": "matched" if audio_frames == matched_audio_frames(t) else f"explicit:{audio_frames}", "audio_windows_per_parent": [1], "text": "states" if text_states is not None else "gaussian"}})
        for step, sigma in enumerate(sigmas):
            row = []
            for k in range(count):
                seed = int(arguments.seed) + 1000 * step + k
                generator = torch.Generator().manual_seed(seed + 777)
                video = torch.randn(24, t, h, w, generator=generator)
                audio = torch.randn(32, 2, audio_frames, generator=generator)
                row.append(build_stimulus(video, audio, sigma=sigma, seed=seed, rung="random", **text_common))
            plan.append(row)
    return plan, notes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S6H MiniMax H3 DiT block I/O capture (container)")
    parser.add_argument("--model", default="/opt/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors")
    parser.add_argument("--out", required=True)
    parser.add_argument("--blocks", default="25", help="comma separated block indices (50 blocks; 25 = depth fraction 0.5)")
    parser.add_argument("--rung", default="latents", choices=("random", "latents"))
    parser.add_argument("--video-pairs", default="/work/media/pairs/h3v_g4_64_merged")
    parser.add_argument("--audio-pairs", default="/work/media/pairs/h3a_d_ctx0_merged")
    parser.add_argument("--clips", type=int, default=0, help="clip pairs per step (0 = every one-to-one pair)")
    parser.add_argument("--clip-offset", type=int, default=0)
    parser.add_argument("--audio-frames", type=int, default=-1, help="audio latent frames per forward (-1 = matched to the video's model-time span [default], 0 = whole clip [named deviation], N = explicit)")
    parser.add_argument("--audio-windows", type=int, default=0, help="non-overlapping matched windows per audio clip (0 = as many as fit [default], 1 = one)")
    parser.add_argument("--video-frames", default="leading-run", choices=VIDEO_FRAME_POLICIES, help="leading-run = keep a clip's leading consecutive latent frame ids (no time hole); all = every frame in the file")
    parser.add_argument("--allow-deviations", action="store_true", help="run even though the stimulus departs from the parent's regime / the spec; the named list goes into the receipt")
    parser.add_argument("--text-states", default=None, help="safetensors with text_states [L, 5120]: the parent's real text-encoder states (default: seeded gaussian, a named deviation)")
    parser.add_argument("--random-grid", type=parse_grid, default=(7, 12, 20), help="T,H,W latent grid for --rung random")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--sigma-grid", default="shifted", choices=("shifted", "uniform"))
    parser.add_argument("--sigmas", type=parse_sigmas, default=None, help="explicit ascending sigma list (overrides --steps/--sigma-grid)")
    parser.add_argument("--text-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expect-device-name", default=None, help="refuse unless torch's device 0 has this name")
    parser.add_argument("--weight-dtype", default="bfloat16", choices=tuple(DTYPES))
    parser.add_argument("--compute-dtype", default="auto", choices=("auto", *DTYPES), help="auto = comfy's unet_manual_cast choice for this card")
    parser.add_argument("--store-dtype", default="bfloat16", choices=tuple(DTYPES))
    parser.add_argument("--parity-rows", type=int, default=512)
    parser.add_argument("--parity-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--max-forwards", type=int, default=0, help="stop after this many forwards in total (smoke)")
    parser.add_argument("--hash-source", dest="hash_source", action="store_true", default=True, help="sha256 the 40 GB source file into the receipt (default on)")
    parser.add_argument("--no-hash-source", dest="hash_source", action="store_false")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing capture under --out (default: refuse)")
    parser.add_argument("--receipt-copy", default=None, help="directory to copy capture_receipt.json into (small receipts stay under $OC)")
    return parser


def refuse_deviations(deviations: list[dict[str, Any]], allow: bool) -> None:
    """A stimulus off the parent's regime / the spec runs only under --allow-deviations, named."""

    if deviations and not allow:
        raise SystemExit(
            "the stimulus departs from the parent's operating regime / the spec: "
            + "; ".join(f"{d['name']}: {d['what']}" for d in deviations)
            + " -- pass --allow-deviations to run and have the list written into the receipt"
        )


def refuse_existing_capture(out: Path, overwrite: bool) -> None:
    """A rerun into a finished --out is a refusal unless --overwrite (finding: idempotency)."""

    existing = out / "capture_receipt.json"
    if existing.is_file() and not overwrite:
        raise SystemExit(
            f"{existing} exists: a rerun would append a second 'start' to progress.jsonl and silently "
            "replace step*/L*.safetensors and blocks/L*.safetensors; pass --overwrite or choose another --out"
        )


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    target = Path(arguments.out)
    refuse_existing_capture(target, bool(arguments.overwrite))
    target.mkdir(parents=True, exist_ok=True)
    progress = target / "progress.jsonl"
    block_ids = [int(v) for v in arguments.blocks.split(",") if v.strip()]
    sigmas, sigma_grid_name = sigma_plan(arguments)
    _log(progress, {"event": "start", "argv": vars(arguments) | {"random_grid": list(arguments.random_grid)}, "sigmas": sigmas, "sigma_grid": sigma_grid_name, "uid": os.getuid() if hasattr(os, "getuid") else None})

    # the plan (pair corpora + torch only) and its deviation gate come BEFORE any GPU or
    # comfy work: a stimulus the operator has not signed off refuses without loading 40 GB
    plan, stimulus_notes = build_stimuli(arguments, sigmas)
    deviations = list(stimulus_notes.get("deviations") or [])
    _log(progress, {"event": "plan", "steps": len(plan), "forwards_per_step": [len(row) for row in plan], "seq_len": [s.seq_len for s in plan[0][:3]], **{k: v for k, v in stimulus_notes.items() if k not in ("clip_pairs", "audio_parent_of_pairs")}})
    refuse_deviations(deviations, bool(arguments.allow_deviations))

    if arguments.device != "cpu":
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device visible; the container must be started with the GPU6 UUID")
        properties = torch.cuda.get_device_properties(0)
        if arguments.expect_device_name and properties.name != arguments.expect_device_name:
            raise SystemExit(f"device 0 is {properties.name!r}, expected {arguments.expect_device_name!r}")
        used_before = int(torch.cuda.mem_get_info(0)[1] - torch.cuda.mem_get_info(0)[0])
        if used_before > 2000 << 20:
            raise SystemExit(f"device already has {used_before >> 20} MiB in use (> 2000 MiB): refusing to start")
    comfy_info = import_comfy(arguments.device)
    _log(progress, {"event": "comfy", **comfy_info})
    if arguments.compute_dtype == "auto":
        chosen = comfy_info["unet_manual_cast_for_bf16"]
        compute_dtype = {"torch.float32": torch.float32, "torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16, "None": torch.bfloat16}[chosen]
    else:
        compute_dtype = DTYPES[arguments.compute_dtype]
    store_dtype = DTYPES[arguments.store_dtype]

    source_sha = sha256_file(arguments.model) if arguments.hash_source else None
    patcher, dit, load_info = load_h3(arguments.model, weight_dtype=DTYPES[arguments.weight_dtype])
    dit.eval()
    if max(block_ids) >= len(dit.blocks):
        raise SystemExit(f"block ids {block_ids} exceed {len(dit.blocks)} blocks")
    where = residency(dit, arguments.device)
    if not where["resident"]:
        _log(progress, {"event": "partial_load", **where})
        dit.to(arguments.device)
        where = residency(dit, arguments.device)
        where["fallback"] = "diffusion_model.to(device) after comfy's load left tensors on the host"
        if not where["resident"]:
            raise SystemExit(f"H3 is not resident on {arguments.device}: {where['off_device_count']} tensors off device")
    vram_loaded = cuda_memory_block(arguments.device)
    _log(progress, {"event": "loaded", "load": load_info, "residency": where, "vram": vram_loaded, "compute_dtype": str(compute_dtype)})

    # block tensors verbatim, so the coreset can run outside the image
    block_receipts: dict[str, Any] = {}
    for index in block_ids:
        names = block_tensor_names(index)
        state = dict(dit.blocks[index].named_parameters())
        export = {}
        for suffix, full in names.items():
            if suffix not in state:
                raise SystemExit(f"block {index} has no parameter {suffix}; parameters are {sorted(state)[:12]}")
            export[full] = state[suffix].detach().to("cpu")
        block_receipts[str(index)] = receipts.write_tensors(export, block_weights_path(target, index))
        _log(progress, {"event": "block_weights", "block": index, "parameters": block_receipts[str(index)]["parameters"], "sha256": block_receipts[str(index)]["state_sha256"]})


    forwards_done = 0
    timings: list[dict[str, Any]] = []
    parity: dict[str, Any] = {}
    step_receipts: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    stopped_early = False
    with H3BlockTap(dit, block_ids) as tap:
        for step, row in enumerate(plan):
            accum: dict[int, dict[str, list[torch.Tensor]]] = {index: {} for index in block_ids}
            per_forward: dict[int, dict[str, list[Any]]] = {index: {"sigma_video": [], "sigma_audio": [], "temb": [], "adaln": [], "mod_segments": []} for index in block_ids}
            meta: dict[str, list[torch.Tensor]] = {"token_kind": [], "token_frame": [], "token_clip": [], "forward_index": []}
            rf_notes = None
            for k, stimulus in enumerate(row):
                if arguments.max_forwards and forwards_done >= arguments.max_forwards:
                    stopped_early = True
                    break
                tap.reset()
                torch.cuda.synchronize() if arguments.device != "cpu" else None
                started = time.time()
                outputs = run_forward(dit, stimulus, device=arguments.device, compute_dtype=compute_dtype)
                torch.cuda.synchronize() if arguments.device != "cpu" else None
                wall = time.time() - started
                reports = [finite_report("video_out", outputs[0]), finite_report("audio_out", outputs[1])]
                for index in block_ids:
                    record = tap.records[index]
                    for name in ("x_in", "x_out", "mlp_in", "mlp_out", "attn_out"):
                        if name not in record:
                            raise SystemExit(f"block {index}: hook did not record {name}")
                        if int(record[name].shape[0]) != stimulus.seq_len:
                            raise SystemExit(f"block {index}: {name} has {int(record[name].shape[0])} rows, stimulus has {stimulus.seq_len}")
                        reports.append(finite_report(f"L{index}.{name}", record[name]))
                    if str(index) not in parity:
                        rows = min(int(arguments.parity_rows), stimulus.seq_len)
                        block = dit.blocks[index]
                        with torch.no_grad():
                            reference = mlp_reference(
                                record["mlp_in"][:rows].to(torch.float32),
                                block.mlp.fc1.weight.detach().to(torch.float32),
                                block.mlp.fc2.weight.detach().to(torch.float32),
                            )
                            captured = record["mlp_out"][:rows].to(torch.float32)
                            rel = float((reference - captured).norm() / captured.norm().clamp_min(1.0e-30))
                            parity[str(index)] = {
                                "rows": rows,
                                "rel_l2": rel,
                                "max_abs_diff": float((reference - captured).abs().max()),
                                "reference": "mediafold.c16_h3.mlp_reference (fp32) vs the container's mlp.forward",
                                "tolerance": float(arguments.parity_tolerance),
                                "pass": rel <= float(arguments.parity_tolerance),
                            }
                            if rel > float(arguments.parity_tolerance):
                                raise SystemExit(f"block {index}: MLP parity rel_l2 {rel} > {arguments.parity_tolerance}; the coreset's reference MLP would not be the parent's")
                    for name in ("x_in", "x_out", "mlp_in", "mlp_out", "attn_out"):
                        accum[index].setdefault(name, []).append(record[name].to(device="cpu", dtype=store_dtype))
                    per_forward[index]["sigma_video"].append(stimulus.sigma_video)
                    per_forward[index]["sigma_audio"].append(stimulus.sigma_audio)
                    per_forward[index]["temb"].append(record["temb"].to(device="cpu", dtype=torch.float32))
                    per_forward[index]["adaln"].append(record["adaln"].to(device="cpu", dtype=torch.float32))
                    per_forward[index]["mod_segments"].append(torch.tensor(record["mod_segments"], dtype=torch.long))
                bad = [r for r in reports if not r["finite"]]
                if bad:
                    overflow.extend(bad)
                    raise SystemExit(f"non-finite values in {[r['name'] for r in bad]} at step {step} forward {k} (compute dtype {compute_dtype})")
                meta["token_kind"].append(stimulus.token_kind)
                meta["token_frame"].append(stimulus.token_frame)
                meta["token_clip"].append(stimulus.token_clip)
                meta["forward_index"].append(torch.full((stimulus.seq_len,), k, dtype=torch.long))
                if rf_notes is None:
                    rf_notes = receptive_field_notes(stimulus)
                timings.append({"step": step, "forward": k, "seq_len": stimulus.seq_len, "seconds": wall, "max_abs_x_out": max(r["max_abs"] or 0.0 for r in reports if r["name"].endswith("x_out"))})
                forwards_done += 1
                if k % 8 == 0 or k == len(row) - 1:
                    _log(progress, {"event": "forward", "step": step, "forward": k, "of": len(row), "seq_len": stimulus.seq_len, "seconds": round(wall, 3), "sigma_video": stimulus.sigma_video, "vram": cuda_memory_block(arguments.device)})
            if not meta["token_kind"]:
                break
            for index in block_ids:
                state = {name: torch.cat(values, dim=0) for name, values in accum[index].items()}
                state.update({name: torch.cat(values, dim=0) for name, values in meta.items()})
                state["sigma_video"] = torch.tensor(per_forward[index]["sigma_video"], dtype=torch.float32)
                state["sigma_audio"] = torch.tensor(per_forward[index]["sigma_audio"], dtype=torch.float32)
                state["temb"] = torch.stack(per_forward[index]["temb"], dim=0)
                state["adaln"] = torch.stack(per_forward[index]["adaln"], dim=0)
                state["mod_segments"] = torch.stack(per_forward[index]["mod_segments"], dim=0)
                path = capture_path(target, step, index)
                written = receipts.write_tensors(state, path)
                kinds = state["token_kind"]
                step_receipts.append(
                    {
                        "step": step,
                        "block": index,
                        "sigma_video": sigmas[step],
                        "sigma_audio": float(per_forward[index]["sigma_audio"][0]),
                        "forwards": len(per_forward[index]["sigma_video"]),
                        "rows": int(kinds.numel()),
                        "rows_by_kind": {"text": int((kinds == KIND_TEXT).sum()), "audio": int((kinds == KIND_AUDIO).sum()), "video": int((kinds == KIND_VIDEO).sum())},
                        "stimulus": {k: v for k, v in row[0].notes.items() if k not in ("seed",)},
                        "receptive_field": rf_notes,
                        "row_stats": {
                            name: {"rms": float(state[name].to(torch.float32).pow(2).mean().sqrt()), "max_abs": float(state[name].to(torch.float32).abs().max())}
                            for name in ("x_in", "x_out", "mlp_in", "mlp_out", "attn_out")
                        },
                        **written,
                    }
                )
                _log(progress, {"event": "capture_written", "step": step, "block": index, "path": str(path), "rows": int(kinds.numel()), "bytes": written["bytes"], "sha256": written["file_sha256"], "rms_x_out": step_receipts[-1]["row_stats"]["x_out"]["rms"]})
            if stopped_early:
                break

    body = {
        "kind": "c16_h3_capture",
        "stage": "S6H",
        "source": {"path": arguments.model, "bytes": load_info["bytes"], "sha256": source_sha},
        "load": load_info,
        "comfy": comfy_info,
        "compute_dtype": str(compute_dtype),
        "store_dtype": str(store_dtype),
        "residency": where,
        "vram_after_load": vram_loaded,
        "vram_peak": cuda_memory_block(arguments.device),
        "blocks": block_ids,
        "block_weights": block_receipts,
        "mlp_parity": parity,
        "sigmas": sigmas,
        "sigma_grid": sigma_grid_name,
        "sigma_coverage_note": "a shift-12 grid at 4 steps only samples sigma in [0.63, 0.99]; pass --sigma-grid uniform or --sigmas for lower noise (a spec-permitted 4-sigma capture either way)",
        "deviations": deviations,
        "stimulus_arms": stimulus_notes.get("stimulus_arms"),
        "stimuli": stimulus_notes,
        "captures": step_receipts,
        "forwards": forwards_done,
        "stopped_early": stopped_early,
        "timing": {
            "forwards": len(timings),
            "seconds_total": float(sum(t["seconds"] for t in timings)),
            "seconds_per_forward_mean": float(sum(t["seconds"] for t in timings) / max(1, len(timings))),
            "first": timings[:4],
        },
        "overflow": overflow,
        "runtime": receipts.runtime_block(arguments.device),
        "argv": vars(arguments) | {"random_grid": list(arguments.random_grid)},
        "source_sha256": receipts.source_sha256([str(_HERE / "c16_h3_capture.py"), str(_HERE / "mediafold" / "c16_h3.py")]),
    }
    body["writer"] = {"uid": os.getuid() if hasattr(os, "getuid") else None, "gid": os.getgid() if hasattr(os, "getgid") else None, "note": "launch with docker --user $(id -u):$(id -g) so the receipt copy is owned by the invoking user"}
    receipt_path = target / "capture_receipt.json"
    receipt_hash = receipts.write_receipt(receipt_path, body)
    if arguments.receipt_copy:
        copy_dir = Path(arguments.receipt_copy)
        copy_dir.mkdir(parents=True, exist_ok=True)
        (copy_dir / "capture_receipt.json").write_bytes(receipt_path.read_bytes())
    _log(progress, {"event": "done", "receipt": str(receipt_path), "receipt_sha256": receipt_hash, "forwards": forwards_done, "captures": len(step_receipts)})
    print(json.dumps({"blocks": block_ids, "captures": len(step_receipts), "forwards": forwards_done, "receipt_sha256": receipt_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
