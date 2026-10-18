"""C16 on the MiniMax H3 audio-video DiT: capture conventions and the 4x MLP coreset.

Pure torch. Nothing here imports ComfyUI; the container-only loading and hooking lives in
``media/c16_h3_capture.py`` and calls into this module for everything that can be tested
on tiny shapes outside the image.

What the H3 DiT block is (read out of ``comfy/ldm/minimax/model.py`` in image
``comfyui:codex-minimax-h3-runtime-env-r5-20260808``, and the safetensors header of
``minimax_h3_fl2va_pruned_bf16.safetensors``: 50 blocks, hidden 5376, 56 MHA heads of 128,
FFN 14336):

    shift/scale/gate x2 = adaln_proj(t_emb)          # [M, 8] -> 6 x [M*3, 5376]
    h = norm1(x) * (1 + scale_msa[row]) + shift_msa[row]     per packed segment
    x = x + gate_msa[row] * attn(h, rope)                    IN PLACE
    h = norm2(x) * (1 + scale_mlp[row]) + shift_mlp[row]
    x = x + gate_mlp[row] * mlp(h)                           IN PLACE
    mlp(h) = fc2(silu(gate) * up)  with  gate, up = fc1(h).chunk(2, dim=-1)

The packed sequence is ``[text | audio | video]`` for the plain fl2va/t2va layout (no
keyframes, no refs), batch size 1 only, and every token attends to every other token of
the SAME forward. So one forward is the attention receptive field, and a forward built
from ONE video clip and ONE audio clip is exactly the clip-aligned window that
``mediafold.c16_capture.receptive_field_notes`` asks for.

The coreset reuses ``mediafold.c16_transplant.coreset_block_mlp`` unchanged: the H3
block's fused ``fc1`` is split into an ``ace_dit.MLP``-shaped gate/up pair (verbatim
rows), ``fc2`` becomes ``down_proj``, and :class:`H3BlockShell` reports the whole-block
parameter count (attention, norms, adaLN counted frozen) so the per-component and
whole-block ratios are both on the receipt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ace_dit import MLP
from .linalg import law_abiding_solve, write_scores

# --------------------------------------------------------------------------------------
# geometry (defaults; every function below reads the real shapes off the tensors)
# --------------------------------------------------------------------------------------

HIDDEN = 5376
FFN = 14336
HEADS = 56
HEAD_DIM = 128
T_DIM = 8
ADALN_EXPAND = 6
ADALN_MODALITIES = 3
VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
TEXT_DIM = 5120
PATCH = (1, 2, 2)
SIGMA_SHIFT_VIDEO = 12.0
SIGMA_SHIFT_AUDIO = 3.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0

KIND_TEXT, KIND_AUDIO, KIND_VIDEO = 0, 1, 2
KIND_NAMES = {KIND_TEXT: "text", KIND_AUDIO: "audio", KIND_VIDEO: "video"}
#: modality tag per packed segment kind (model.py ``seg_tag``): video 0, text 1, audio 2
SEGMENT_TAG = {"text": 1, "video": 0, "audio": 2}
#: audio clip ids are offset so a membership row [video_clip, audio_clip] never collides
AUDIO_CLIP_OFFSET = 1_000_000
#: non-overlapping audio windows of one clip get ids ``parent * AUDIO_WINDOW_STRIDE + window``;
#: the membership row and the holdout group carry the PARENT clip so two windows of one
#: clip are never on opposite sides of the split
AUDIO_WINDOW_STRIDE = 100

BLOCK_TENSOR_SUFFIXES = (
    "attn.k_norm.weight",
    "attn.out_proj.weight",
    "attn.q_norm.weight",
    "attn.qkv_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    "norm1.weight",
    "norm2.weight",
    "adaln_proj.linear.weight",
    "adaln_proj.linear.bias",
)
MLP_SUFFIXES = ("mlp.fc1.weight", "mlp.fc2.weight")


def block_tensor_names(index: int) -> dict[str, str]:
    """suffix -> full checkpoint name for block ``index``."""

    return {suffix: f"blocks.{int(index)}.{suffix}" for suffix in BLOCK_TENSOR_SUFFIXES}


def strip_block_prefix(state: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    """Keep only block ``index``'s tensors, keyed by suffix."""

    prefix = f"blocks.{int(index)}."
    out = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
    missing = [suffix for suffix in BLOCK_TENSOR_SUFFIXES if suffix not in out]
    if missing:
        raise KeyError(f"block {index} is missing {missing}")
    return out


def parameter_count_of(state: dict[str, torch.Tensor]) -> int:
    return int(sum(int(value.numel()) for value in state.values()))


# --------------------------------------------------------------------------------------
# schedule (formulas copied from model.py so the tests can pin them)
# --------------------------------------------------------------------------------------


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_slope(sigma: float, from_shift: float, to_shift: float) -> float:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return (to_shift * (1.0 + (from_shift - 1.0) * base) ** 2) / (
        from_shift * (1.0 + (to_shift - 1.0) * base) ** 2
    )


def shifted_sigma_grid(steps: int, *, shift: float = SIGMA_SHIFT_VIDEO) -> list[float]:
    """The video sigmas a ``shift``-scheduled sampler visits at ``steps`` uniform base points.

    ``comfy.model_sampling`` maps a uniform base grid ``b`` to ``shift*b / (1 + (shift-1)*b)``
    (``MiniMaxH3.sampling_settings["shift"] == 12.0``); the mid-points of ``steps`` equal
    cells are used, the same convention as ``c16_capture`` (``(step + 0.5) / steps``).
    """

    if int(steps) < 1:
        raise ValueError("steps must be >= 1")
    return [time_shift_sigma((step + 0.5) / float(steps), 1.0, float(shift)) for step in range(int(steps))]


def uniform_sigma_grid(steps: int) -> list[float]:
    if int(steps) < 1:
        raise ValueError("steps must be >= 1")
    return [(step + 0.5) / float(steps) for step in range(int(steps))]


def video_time_span(latent_frames: int) -> float:
    """Model-time extent of ``latent_frames`` video latents (``_video_t_spans`` summed).

    The unit is one audio latent frame (40 Hz): 5 latent frames = 17 raw frames at 24 fps
    = 28.33 audio frames, which is what ``FRAME_RESCALE * (1,4,4,4,4)`` sums to.
    """

    if int(latent_frames) < 1:
        raise ValueError("latent_frames must be >= 1")
    return float(sum(FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(int(latent_frames))))


def matched_audio_frames(latent_frames: int) -> int:
    """Audio latent frames spanning the same model time as ``latent_frames`` video latents."""

    return max(1, int(round(video_time_span(latent_frames))))


# --------------------------------------------------------------------------------------
# the MLP, split into the ace_dit.MLP layout the coreset already understands
# --------------------------------------------------------------------------------------


def swiglu_reference(fc1_out: torch.Tensor) -> torch.Tensor:
    """``comfy.ops._swiglu_eager``: ``gate, up = x.chunk(2, -1); silu(gate) * up``."""

    gate, up = fc1_out.chunk(2, dim=-1)
    return F.silu(gate) * up


def split_fc1(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``fc1`` [2*ffn, hidden] -> (gate [ffn, hidden], up [ffn, hidden]) VERBATIM rows."""

    if weight.dim() != 2 or int(weight.shape[0]) % 2:
        raise ValueError(f"fc1 must be [2*ffn, hidden]; got {tuple(weight.shape)}")
    ffn = int(weight.shape[0]) // 2
    return weight[:ffn], weight[ffn:]


def fuse_fc1(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError("gate and up must have the same shape")
    return torch.cat((gate, up), dim=0)


def mlp_reference(mlp_in: torch.Tensor, fc1: torch.Tensor, fc2: torch.Tensor) -> torch.Tensor:
    """The block's MLP output from its own weights, in the dtype of ``mlp_in``."""

    return F.linear(swiglu_reference(F.linear(mlp_in, fc1.to(mlp_in.dtype))), fc2.to(mlp_in.dtype))


class H3BlockShell(nn.Module):
    """An H3 DiT block seen by the coreset: an ``ace_dit.MLP`` plus a frozen count.

    ``coreset_block_mlp`` only touches ``.mlp`` (gate_proj / up_proj / down_proj,
    ``hidden``), deep-copies the block and reads ``block_parameters()`` on parent and
    child. Everything else in the H3 block (attention, norms, adaLN) is untouched by the
    MLP coreset and enters the whole-block ratio as a constant.
    """

    def __init__(self, mlp: MLP, *, frozen_parameters: int, frozen_names: Sequence[str] = ()) -> None:
        super().__init__()
        self.mlp = mlp
        self.frozen_parameters = int(frozen_parameters)
        self.frozen_names = tuple(str(name) for name in frozen_names)

    def block_parameters(self) -> int:
        return self.frozen_parameters + self.mlp_parameters()

    def mlp_parameters(self) -> int:
        return int(sum(int(p.numel()) for p in self.mlp.parameters()))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:  # the MLP only; the rest is frozen
        return self.mlp(rows)

    def fused_state(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """The MLP back in the checkpoint's own layout (``mlp.fc1.weight`` fused)."""

        return {
            f"{prefix}mlp.fc1.weight": fuse_fc1(self.mlp.gate_proj.weight.detach(), self.mlp.up_proj.weight.detach()),
            f"{prefix}mlp.fc2.weight": self.mlp.down_proj.weight.detach(),
        }


def shell_from_block_state(state: dict[str, torch.Tensor], *, dtype: torch.dtype = torch.float32) -> H3BlockShell:
    """Build the shell from a block's tensors keyed by suffix (``strip_block_prefix``)."""

    for suffix in MLP_SUFFIXES:
        if suffix not in state:
            raise KeyError(f"block state lacks {suffix}")
    fc1 = state["mlp.fc1.weight"].detach()
    fc2 = state["mlp.fc2.weight"].detach()
    gate, up = split_fc1(fc1)
    ffn, hidden = int(gate.shape[0]), int(gate.shape[1])
    if tuple(fc2.shape) != (hidden, ffn):
        raise ValueError(f"fc2 {tuple(fc2.shape)} does not match fc1 {tuple(fc1.shape)}")
    mlp = MLP(hidden, ffn)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(gate.to(torch.float32))
        mlp.up_proj.weight.copy_(up.to(torch.float32))
        mlp.down_proj.weight.copy_(fc2.to(torch.float32))
    mlp = mlp.to(dtype)
    frozen = {name: value for name, value in state.items() if name not in MLP_SUFFIXES}
    return H3BlockShell(mlp, frozen_parameters=parameter_count_of(frozen), frozen_names=sorted(frozen))


# --------------------------------------------------------------------------------------
# clip latents out of the C15 pair corpora
# --------------------------------------------------------------------------------------


VIDEO_FRAME_POLICIES = ("all", "leading-run")


def frame_alignment(frame_ids: Sequence[int]) -> dict[str, Any]:
    """Does a clip's latent frame id list match what H3 assumes about its video input?

    ``model.py`` (``_video_t_spans`` / ``FRAME_PER_TOKEN`` (1, 4, 4, 4, 4)) treats the
    FIRST latent frame as the 1-raw-frame latent and every later one as 4 raw frames, and
    its rope time axis counts latent frames consecutively. A clip whose ids skip a frame
    carries a 4-raw-frame hole the model does not know about; one that does not start at
    latent frame 0 puts 4 raw frames of content on the 1-raw-frame slot.
    """

    ids = [int(v) for v in frame_ids]
    if not ids:
        raise ValueError("frame_ids is empty")
    gaps = [[ids[k - 1], ids[k]] for k in range(1, len(ids)) if ids[k] != ids[k - 1] + 1]
    leading = [ids[0]]
    for value in ids[1:]:
        if value != leading[-1] + 1:
            break
        leading.append(value)
    consecutive = not gaps
    starts_at_zero = ids[0] == 0
    return {
        "latent_frame_ids": ids,
        "consecutive": consecutive,
        "starts_at_zero": starts_at_zero,
        "gaps": gaps,
        "leading_run": leading,
        "aligned": consecutive and starts_at_zero,
        "model_assumption": "FRAME_PER_TOKEN (1, 4, 4, 4, 4) from latent frame 0; rope time axis counts latent frames consecutively",
    }


def video_clips_from_pairs(
    pairs: Any, *, grid: tuple[int, int] | None = None, frames: str = "all"
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Rebuild per-clip latent videos ``[C, T, H, W]`` from a measured-group pair set.

    ``mediafold.c15_video_groups.sample_grouped_video_pairs`` appends, per clip, the
    qualifying latent frames in ascending ``frame_index`` and within a frame the spatial
    cells in raster (h, w) order; the file order is therefore the only spatial index and
    it is trusted here, with the per-frame row count checked against the grid.

    ``frames="leading-run"`` keeps only the leading run of CONSECUTIVE latent frame ids
    (drops everything after the first gap), so the clip has no time hole; whether it
    starts at latent frame 0 is reported in ``notes["frame_alignment"]`` either way and
    every clip must carry the same id list.
    """

    if frames not in VIDEO_FRAME_POLICIES:
        raise ValueError(f"frames must be one of {VIDEO_FRAME_POLICIES}; got {frames!r}")
    latent = pairs.latent.detach().to(torch.float32)
    clip_index = pairs.clip_index.reshape(-1).to(torch.long)
    frame_index = pairs.frame_index.reshape(-1).to(torch.long)
    channels = int(latent.shape[1])
    clips: dict[int, torch.Tensor] = {}
    frame_ids: dict[int, list[int]] = {}
    ids_in_file: list[int] | None = None
    grid_seen: tuple[int, int] | None = None
    for clip in torch.unique(clip_index).tolist():
        rows = torch.nonzero(clip_index == clip).flatten()
        frames_of_clip = frame_index[rows]
        order = torch.unique_consecutive(frames_of_clip).tolist()
        if len(order) != int(torch.unique(frames_of_clip).numel()):
            raise ValueError(f"clip {clip}: latent frames are not contiguous in file order")
        if ids_in_file is None:
            ids_in_file = list(order)
        elif list(order) != ids_in_file:
            raise ValueError(f"clip {clip}: latent frame ids {order} differ from clip {min(clips) if clips else '?'}'s {ids_in_file}")
        if frames == "leading-run":
            order = frame_alignment(order)["leading_run"]
            rows = rows[torch.isin(frames_of_clip, torch.tensor(order, dtype=torch.long))]
            frames_of_clip = frame_index[rows]
        per_frame = [int((frames_of_clip == f).sum().item()) for f in order]
        if min(per_frame) != max(per_frame):
            raise ValueError(f"clip {clip}: uneven rows per latent frame {per_frame}")
        cells = per_frame[0]
        if grid is None:
            side = int(math.isqrt(cells))
            if side * side != cells:
                raise ValueError(
                    f"clip {clip}: {cells} cells per latent frame is not a square grid; pass grid=(h, w)"
                )
            this_grid = (side, side)
        else:
            this_grid = (int(grid[0]), int(grid[1]))
            if this_grid[0] * this_grid[1] != cells:
                raise ValueError(f"clip {clip}: grid {this_grid} does not cover {cells} cells")
        if grid_seen is None:
            grid_seen = this_grid
        elif grid_seen != this_grid:
            raise ValueError(f"clip {clip}: grid {this_grid} differs from {grid_seen}")
        block = latent[rows].reshape(len(order), this_grid[0], this_grid[1], channels)
        clips[int(clip)] = block.permute(3, 0, 1, 2).contiguous()
        frame_ids[int(clip)] = [int(f) for f in order]
    if not clips:
        raise ValueError("pair set has no clips")
    frames_per_clip = sorted({int(v.shape[1]) for v in clips.values()})
    used_ids = frame_ids[min(frame_ids)]
    notes = {
        "clips": len(clips),
        "latent_channels": channels,
        "grid_hw": list(grid_seen) if grid_seen else None,
        "latent_frames_per_clip": frames_per_clip,
        "latent_frame_ids": {str(k): v for k, v in list(frame_ids.items())[:4]},
        "latent_frame_ids_in_file": ids_in_file,
        "latent_frame_ids_used": used_ids,
        "frame_policy": frames,
        "frame_alignment": frame_alignment(used_ids),
        "frame_alignment_in_file": frame_alignment(ids_in_file or used_ids),
        "row_order_assumption": "file order within a clip = (latent frame asc, raster h, raster w)",
    }
    return clips, notes


def audio_clips_from_pairs(
    pairs: Any, *, frames: int = 0, windows: int = 1
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Rebuild per-clip audio latents ``[C, channels, T]`` from a per-channel pair set.

    ``frames`` > 0 slices each clip to that many latent frames (the matched video span);
    ``windows`` > 1 (or 0 = as many as fit) cuts NON-OVERLAPPING windows of ``frames``
    from each clip, keyed ``parent * AUDIO_WINDOW_STRIDE + window`` with the parent clip
    in ``notes["parent_of"]``; ``windows == 1`` keeps the raw clip ids.
    """

    if int(windows) != 1 and int(frames) <= 0:
        raise ValueError("audio windows need frames > 0 (the window length)")
    if int(windows) < 0:
        raise ValueError("windows must be >= 0 (0 = as many as fit)")
    latent = pairs.latent.detach().to(torch.float32)
    clip_index = pairs.clip_index.reshape(-1).to(torch.long)
    frame_index = pairs.frame_index.reshape(-1).to(torch.long)
    channel_index = pairs.channel_index.reshape(-1).to(torch.long)
    if bool((channel_index < 0).any()):
        raise ValueError("audio pair set has no per-channel provenance (stacked pairing); H3 needs [C, 2, T]")
    channels = int(channel_index.max().item()) + 1
    width = int(latent.shape[1])
    clips: dict[int, torch.Tensor] = {}
    lengths: set[int] = set()
    counts_in_file: set[int] = set()
    parent_of: dict[int, int] = {}
    windows_of: dict[int, int] = {}
    for clip in torch.unique(clip_index).tolist():
        rows = torch.nonzero(clip_index == clip).flatten()
        count = int(frame_index[rows].max().item()) + 1
        counts_in_file.add(count)
        block = torch.zeros(width, channels, count, dtype=torch.float32)
        filled = torch.zeros(channels, count, dtype=torch.bool)
        block[:, channel_index[rows], frame_index[rows]] = latent[rows].transpose(0, 1)
        filled[channel_index[rows], frame_index[rows]] = True
        if not bool(filled.all()):
            raise ValueError(f"audio clip {clip}: {int((~filled).sum())} (channel, frame) cells missing")
        if frames:
            if count < int(frames):
                raise ValueError(f"audio clip {clip} has {count} frames; --audio-frames asks {frames}")
        if int(windows) == 1:
            piece = block[:, :, : int(frames)] if frames else block
            clips[int(clip)] = piece.contiguous()
            parent_of[int(clip)] = int(clip)
            lengths.add(int(piece.shape[-1]))
            continue
        fit = count // int(frames)
        wanted = fit if int(windows) == 0 else int(windows)
        if wanted > fit:
            raise ValueError(f"audio clip {clip}: {count} frames hold {fit} non-overlapping windows of {frames}; --audio-windows asks {wanted}")
        if wanted > AUDIO_WINDOW_STRIDE - 1:
            raise ValueError(f"at most {AUDIO_WINDOW_STRIDE - 1} windows per clip fit the id stride")
        windows_of[int(clip)] = wanted
        for window in range(wanted):
            key = int(clip) * AUDIO_WINDOW_STRIDE + window
            if key >= AUDIO_CLIP_OFFSET:
                raise ValueError("audio window id collides with AUDIO_CLIP_OFFSET")
            piece = block[:, :, window * int(frames) : (window + 1) * int(frames)]
            clips[key] = piece.contiguous()
            parent_of[key] = int(clip)
            lengths.add(int(piece.shape[-1]))
    if not clips:
        raise ValueError("audio pair set has no clips")
    notes = {
        "clips": len(clips),
        "parent_clips": len(set(parent_of.values())),
        "latent_channels": width,
        "stereo_channels": channels,
        "frames_per_clip": sorted(lengths),
        "frames_in_file": sorted(counts_in_file),
        "frames_requested": int(frames),
        "windows_requested": int(windows),
        "windows_per_parent": sorted(set(windows_of.values())) if windows_of else [1],
        "window_id": f"parent * {AUDIO_WINDOW_STRIDE} + window" if windows_of else "parent clip id",
        "parent_of": {str(k): v for k, v in parent_of.items()},
    }
    return clips, notes


def audio_parent_of(window_id: int, parent_of: dict[str, int] | dict[int, int] | None) -> int:
    """The parent audio clip of a window id (identity when no window table is given)."""

    if parent_of is None:
        return int(window_id)
    return int(parent_of.get(str(window_id), parent_of.get(int(window_id), window_id)))  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# the packed layout and the stimulus
# --------------------------------------------------------------------------------------


def video_token_grid(shape: Sequence[int], patch: Sequence[int] = PATCH) -> tuple[int, int, int]:
    """(t, h, w) token grid for a ``[C, T, H, W]`` latent after comfy's pad-to-patch."""

    _, t_full, h_full, w_full = (int(v) for v in shape)
    pt, ph, pw = (int(v) for v in patch)
    return (-(-t_full // pt), -(-h_full // ph), -(-w_full // pw))


def packed_layout(text_len: int, audio_frames: int, video_shape: Sequence[int], *, patch: Sequence[int] = PATCH) -> dict[str, Any]:
    """Per-token kind / frame for the plain ``[text | audio | video]`` packed sequence.

    Audio rows are channel-major (``pack_audio``: ch0 t0..T-1, ch1 t0..T-1); video rows are
    ``patchify_video`` order (t, h, w).
    """

    t, h, w = video_token_grid(video_shape, patch)
    n_text, n_audio, n_video = int(text_len), 2 * int(audio_frames), t * h * w
    segments = [(0, n_text, "text"), (n_text, n_text + n_audio, "audio"), (n_text + n_audio, n_text + n_audio + n_video, "video")]
    kind = torch.cat(
        (
            torch.full((n_text,), KIND_TEXT, dtype=torch.int8),
            torch.full((n_audio,), KIND_AUDIO, dtype=torch.int8),
            torch.full((n_video,), KIND_VIDEO, dtype=torch.int8),
        )
    )
    frame = torch.cat(
        (
            torch.full((n_text,), -1, dtype=torch.long),
            torch.arange(int(audio_frames), dtype=torch.long).repeat(2),
            torch.arange(t, dtype=torch.long).repeat_interleave(h * w),
        )
    )
    return {
        "segments": segments,
        "seq_len": n_text + n_audio + n_video,
        "token_kind": kind,
        "token_frame": frame,
        "video_tokens_thw": (t, h, w),
    }


TEXT_SOURCE_GAUSSIAN = "seeded gaussian Qwen-width states refined by the parent's condition_proj + token_refiner"
TEXT_SOURCE_STATES = "supplied text-encoder states (--text-states) refined by the parent's condition_proj + token_refiner"


def stimulus_deviations(
    *,
    frame_alignment_used: dict[str, Any] | None,
    audio_frames_used: int,
    matched_audio_frames_for_video: int,
    text_source: str,
    s1_media_stimuli: str = "absent",
) -> list[dict[str, Any]]:
    """The NAMED deviations of a capture stimulus from the parent's operating regime / the spec.

    Each entry is ``{"name", "what", "why_it_matters", "arm"}``; an empty list means the
    stimulus is on-regime. ``c16_h3_capture.py`` refuses to run with a non-empty list
    unless its ``--allow-deviations`` flag is set, in which case the list is written into
    the receipt and the S6H stage receipt must carry it.
    """

    out: list[dict[str, Any]] = []
    if frame_alignment_used is not None and not frame_alignment_used.get("aligned", False):
        ids = frame_alignment_used.get("latent_frame_ids") or []
        out.append(
            {
                "name": "video_frame_alignment",
                "what": (
                    f"video clips are latent frames {ids} fed as frames 0..{len(ids) - 1}; consecutive="
                    f"{frame_alignment_used.get('consecutive')}, starts_at_zero={frame_alignment_used.get('starts_at_zero')}, "
                    f"gaps={frame_alignment_used.get('gaps')}"
                ),
                "why_it_matters": frame_alignment_used.get("model_assumption"),
                "arm": "video_frames",
            }
        )
    if int(audio_frames_used) != int(matched_audio_frames_for_video):
        out.append(
            {
                "name": "audio_span",
                "what": f"{int(audio_frames_used)} audio latent frames per forward beside a video whose model-time span is {int(matched_audio_frames_for_video)} audio frames",
                "why_it_matters": "H3 is trained on time-aligned audio/video; an unmatched span puts the block off the parent's operating regime",
                "arm": "audio_span",
            }
        )
    if text_source != TEXT_SOURCE_STATES:
        out.append(
            {
                "name": "text_conditioning",
                "what": f"text = {text_source}",
                "why_it_matters": (
                    "H3 is a text-to-AV model: every captured audio/video activation is conditioned through full "
                    "attention on text states the parent never sees; the parent's real text path (its Qwen encoder "
                    "states for a fixed prompt set, or the empty-prompt embedding) is not on this box"
                ),
                "arm": "text",
            }
        )
    if s1_media_stimuli != "present":
        out.append(
            {
                "name": "s1_media_stimuli",
                "what": f"S1 media stimuli {s1_media_stimuli}; the C15 pair corpora (procedural clips) stand in",
                "why_it_matters": "the spec's S6H input is 'S1 media stimuli' (STAGES S1: media stimuli remain pending)",
                "arm": "stimuli",
            }
        )
    return out


@dataclass
class H3Stimulus:
    video: torch.Tensor  # [1, 24, T, H, W] noisy at sigma_video
    audio: torch.Tensor  # [1, 32, 2, Ta] noisy at sigma_audio
    text: torch.Tensor  # [1, L, 5120] seeded gaussian Qwen states (refined by the parent)
    sigma_video: float
    sigma_audio: float
    clip_id: int  # video clip (-1 on the random rung)
    audio_clip_id: int  # audio clip (-1 on the random rung)
    token_kind: torch.Tensor  # [S] int8
    token_frame: torch.Tensor  # [S]
    token_clip: torch.Tensor  # [S]: video clip id / AUDIO_CLIP_OFFSET + audio clip id / -1 for text
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        return int(self.token_kind.numel())


def flow_noise(clean: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    """Rectified-flow interpolate ``(1 - sigma) * z + sigma * noise`` (comfy ``ModelSamplingFlow``)."""

    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
    return (1.0 - float(sigma)) * clean.to(torch.float32) + float(sigma) * noise


def build_stimulus(
    video_clean: torch.Tensor,
    audio_clean: torch.Tensor,
    *,
    sigma: float,
    seed: int,
    text_len: int = 32,
    clip_id: int = -1,
    audio_clip_id: int = -1,
    audio_parent_clip_id: int | None = None,
    text_dim: int = TEXT_DIM,
    text_states: torch.Tensor | None = None,
    shift_video: float = SIGMA_SHIFT_VIDEO,
    shift_audio: float = SIGMA_SHIFT_AUDIO,
    rung: str = "latents",
) -> H3Stimulus:
    """One forward's inputs from ONE video clip and ONE audio clip (or audio window).

    The video stream is noised at the sampler's sigma; the audio stream at the model's
    own shifted audio sigma (``time_shift_sigma(sigma, 12, 3)``), which is where the
    parent's forward places it. Without ``text_states`` the text states are seeded
    gaussians in Qwen's 5120 width (a NAMED DEVIATION: the parent's real text path is its
    Qwen encoder; see ``stimulus_deviations``); with it the given ``[L, 5120]`` states are
    used verbatim. Both go through the parent's own ``condition_proj`` + token refiner.

    ``audio_parent_clip_id`` is the clip an audio WINDOW was cut from; the per-row
    provenance (``token_clip``) carries the parent so the holdout guard and grouping see
    windows of one clip as one unit, while ``notes`` keep the window id.
    """

    if video_clean.dim() != 4:
        raise ValueError("video_clean must be [C, T, H, W]")
    if audio_clean.dim() != 3:
        raise ValueError("audio_clean must be [C, channels, T]")
    if int(video_clean.shape[2]) % PATCH[1] or int(video_clean.shape[3]) % PATCH[2]:
        raise ValueError("video H and W must be multiples of the 2x2 patch (comfy would pad; the fixture must not)")
    generator = torch.Generator().manual_seed(int(seed))
    sigma_video = float(sigma)
    sigma_audio = float(time_shift_sigma(sigma_video, float(shift_video), float(shift_audio)))
    video = flow_noise(video_clean, sigma_video, generator).unsqueeze(0)
    audio = flow_noise(audio_clean, sigma_audio, generator).unsqueeze(0)
    if text_states is None:
        text = torch.randn(1, int(text_len), int(text_dim), generator=generator, dtype=torch.float32)
        text_source = TEXT_SOURCE_GAUSSIAN
    else:
        text = text_states.detach().to(torch.float32)
        if text.dim() == 2:
            text = text.unsqueeze(0)
        if text.dim() != 3 or int(text.shape[0]) != 1 or int(text.shape[-1]) != int(text_dim):
            raise ValueError(f"text_states must be [L, {text_dim}] or [1, L, {text_dim}]; got {tuple(text.shape)}")
        text_len = int(text.shape[1])
        text_source = TEXT_SOURCE_STATES
    parent_audio = int(audio_clip_id) if audio_parent_clip_id is None else int(audio_parent_clip_id)
    layout = packed_layout(int(text_len), int(audio_clean.shape[-1]), video_clean.shape)
    kind = layout["token_kind"]
    token_clip = torch.full((layout["seq_len"],), -1, dtype=torch.long)
    if clip_id >= 0:
        token_clip[kind == KIND_VIDEO] = int(clip_id)
    if parent_audio >= 0:
        token_clip[kind == KIND_AUDIO] = AUDIO_CLIP_OFFSET + parent_audio
    notes = {
        "rung": str(rung),
        "seed": int(seed),
        "sigma_video": sigma_video,
        "sigma_audio": sigma_audio,
        "audio_velocity_slope": float(time_shift_slope(sigma_video, float(shift_video), float(shift_audio))),
        "clip_id": int(clip_id),
        "audio_clip_id": int(audio_clip_id),
        "audio_parent_clip_id": parent_audio,
        "audio_window_of_parent": int(audio_clip_id) != parent_audio,
        "text_len": int(text_len),
        "video_shape": [int(v) for v in video_clean.shape],
        "audio_shape": [int(v) for v in audio_clean.shape],
        "video_tokens_thw": list(layout["video_tokens_thw"]),
        "seq_len": int(layout["seq_len"]),
        "segments": [[a, b, k] for a, b, k in layout["segments"]],
        "text_source": text_source,
    }
    return H3Stimulus(
        video=video,
        audio=audio,
        text=text,
        sigma_video=sigma_video,
        sigma_audio=sigma_audio,
        clip_id=int(clip_id),
        audio_clip_id=parent_audio,
        token_kind=kind,
        token_frame=layout["token_frame"],
        token_clip=token_clip,
        notes=notes,
    )


def receptive_field_notes(stimulus: H3Stimulus) -> dict[str, Any]:
    """What a clip-grouped holdout can claim about rows captured from THIS forward.

    Every H3 block attends over the whole packed sequence, so a token's activation is a
    function of every row of its forward: the text rows, the audio clip and the video
    clip. A forward holds exactly one video clip and one audio clip here, so grouping the
    holdout by the (video clip, audio clip) pair -- both ids carried per row -- makes the
    fit and holdout activations independent, PROVIDED no clip id appears in two forwards
    (``one_to_one`` pairing; ``coreset_row_groups`` re-checks it through row membership).
    """

    single = stimulus.clip_id >= 0 and stimulus.audio_clip_id >= 0
    return {
        "attention_receptive_field": (
            "one packed forward [text | audio | video] (every H3 block is full attention over "
            "the packed sequence), so a token's activation is a function of every row of its forward"
        ),
        "forward_is_single_clip_pair": bool(single),
        "reading": (
            "each forward is exactly one video clip and one audio clip, so an activation carries "
            "no content from any other clip and a clip-pair-grouped holdout is independent of the fit"
            if single
            else "this forward has no clip provenance (random rung): grouping by forward removes "
            "row-level overlap but says nothing about content"
        ),
    }


# --------------------------------------------------------------------------------------
# capture rows -> coreset groups
# --------------------------------------------------------------------------------------


def coreset_row_groups(
    token_kind: torch.Tensor,
    token_clip: torch.Tensor,
    forward_index: torch.Tensor,
    *,
    rows: str = "av",
    group_by: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, str, torch.Tensor | None, dict[str, Any]]:
    """Decide how the H3 coreset holdout is grouped, over the flattened capture rows.

    Returns ``(keep_mask, groups, group_name, row_membership, notes)``.

    * text rows are dropped (``rows="av"``): they are seeded gaussians with no clip;
    * on the latents rung the group id is a clip id and the membership row is
      ``[video_clip, AUDIO_CLIP_OFFSET + audio_clip]`` for every kept row of the forward,
      so ``coreset_block_mlp``'s guard refuses a split in which a fit row's forward shares
      either clip with a held-out forward. ``group_by="video"`` groups by the video clip,
      ``"audio"`` by the (parent) audio clip, ``"auto"`` picks the id that is shared
      across forwards (audio windows of one clip) and refuses when both are shared;
    * on the random rung (no provenance) the group is the forward index and the receipt
      says so.
    """

    if group_by not in ("auto", "video", "audio"):
        raise ValueError(f"group_by must be auto|video|audio; got {group_by!r}")

    kind = token_kind.reshape(-1).to(torch.long)
    clip = token_clip.reshape(-1).to(torch.long)
    forward = forward_index.reshape(-1).to(torch.long)
    if kind.numel() != clip.numel() or kind.numel() != forward.numel():
        raise ValueError("token_kind, token_clip and forward_index must have one entry per row")
    if rows == "av":
        keep = kind != KIND_TEXT
    elif rows == "video":
        keep = kind == KIND_VIDEO
    elif rows == "audio":
        keep = kind == KIND_AUDIO
    elif rows == "all":
        keep = torch.ones_like(kind, dtype=torch.bool)
    else:
        raise ValueError(f"unknown rows selection {rows!r}")
    kept_forward = forward[keep]
    kept_kind = kind[keep]
    kept_clip = clip[keep]
    if int(keep.sum()) == 0:
        raise ValueError("no rows kept")

    has_provenance = bool((kept_clip[kept_kind != KIND_TEXT] >= 0).all()) and bool((kept_kind != KIND_TEXT).any())
    if not has_provenance:
        groups = kept_forward.clone()
        notes = {
            "grouped_by": "forward",
            "reason": "rows carry no clip provenance (random rung); the forward is the only grouping available",
            "groups": int(torch.unique(groups).numel()),
            "rows_before_drop": int(kind.numel()),
            "rows_used": int(keep.sum()),
            "rows_selection": rows,
            "membership_checked": False,
        }
        return keep, groups, "forward", None, notes

    # per forward: its video clip and audio clip (from its own rows)
    video_of: dict[int, int] = {}
    audio_of: dict[int, int] = {}
    for f in torch.unique(kept_forward).tolist():
        rows_f = kept_forward == f
        v = torch.unique(kept_clip[rows_f & (kept_kind == KIND_VIDEO)])
        a = torch.unique(kept_clip[rows_f & (kept_kind == KIND_AUDIO)])
        if v.numel() > 1 or a.numel() > 1:
            raise ValueError(f"forward {f} carries more than one video or audio clip")
        video_of[int(f)] = int(v.item()) if v.numel() else -1
        audio_of[int(f)] = int(a.item()) if a.numel() else -1
    video_vec = torch.tensor([video_of[int(f)] for f in kept_forward.tolist()], dtype=torch.long)
    audio_vec = torch.tensor([audio_of[int(f)] for f in kept_forward.tolist()], dtype=torch.long)

    # The same clip pair recurs once per sigma step (concat_captures renumbers forwards
    # globally), so "appears in several forwards" means nothing. What decides the group
    # axis is PARTNER cardinality: grouping by video is clean iff every audio clip pairs
    # with exactly one video clip (audio nested inside video groups), and vice versa.
    partners_of_video: dict[int, set[int]] = {}
    partners_of_audio: dict[int, set[int]] = {}
    for f in video_of:
        v, a = video_of[f], audio_of[f]
        if v >= 0 and a >= 0:
            partners_of_video.setdefault(v, set()).add(a)
            partners_of_audio.setdefault(a, set()).add(v)
    video_multi = sum(1 for ps in partners_of_video.values() if len(ps) > 1)  # video clips with several audio partners
    audio_multi = sum(1 for ps in partners_of_audio.values() if len(ps) > 1)  # audio clips with several video partners
    video_clean = audio_multi == 0  # every audio clip sits inside one video group
    audio_clean = video_multi == 0  # every video clip sits inside one audio group
    chosen = group_by
    if group_by == "auto":
        if not video_clean and not audio_clean:
            raise ValueError(
                f"{video_multi} video clips pair with several audio clips and {audio_multi} audio clips pair with "
                "several video clips: no single clip id partitions the material; capture with one shared axis"
            )
        chosen = "video" if video_clean else "audio"
    if chosen == "video" and not video_clean:
        raise ValueError(f"group_by='video' but {audio_multi} audio clips pair with several video clips (audio windows): group by audio")
    if chosen == "audio" and not audio_clean:
        raise ValueError(f"group_by='audio' but {video_multi} video clips pair with several audio clips: group by video")
    # a forward without rows of the chosen kind (rows="audio"/"video") groups by the other id
    if chosen == "audio":
        groups = torch.where(audio_vec >= 0, audio_vec, video_vec)
        group_id = "AUDIO_CLIP_OFFSET + parent audio clip id (video clip id where a forward has no audio rows)"
    else:
        groups = torch.where(video_vec >= 0, video_vec, audio_vec)
        group_id = "video clip id (audio clip id where a forward has no video rows)"
    membership = torch.stack((video_vec, audio_vec), dim=1)
    notes = {
        "grouped_by": "clip",
        "group_axis": chosen,
        "group_by_requested": group_by,
        "group_id": group_id,
        "video_clips_with_several_audio_partners": video_multi,
        "audio_clips_with_several_video_partners": audio_multi,
        "axis_rule": "group by video iff every audio clip pairs with one video clip; by audio iff every video clip pairs with one audio clip; the same pair recurring per sigma step is not a share",
        "groups": int(torch.unique(groups).numel()),
        "forwards": int(torch.unique(kept_forward).numel()),
        "rows_before_drop": int(kind.numel()),
        "rows_used": int(keep.sum()),
        "text_rows_dropped": int((kind == KIND_TEXT).sum()),
        "rows_selection": rows,
        "membership_checked": True,
        "membership": "[video_clip, AUDIO_CLIP_OFFSET + audio_clip] per row, re-derived from the rows of each forward",
        "guarantee": (
            "row-level clip-pair disjointness (no video clip and no audio clip has rows on both sides); "
            "with one forward = one clip pair this is also independence of the activations -- read receptive_field"
        ),
    }
    return keep, groups, "clip", membership, notes


# --------------------------------------------------------------------------------------
# MLP-level arms (the chart-level T1 arms belong to the S7 install, not here)
# --------------------------------------------------------------------------------------

MLP_ARMS = ("replace", "control", "random_select", "prior_only")


def _narrowed_mlp(parent: MLP, selection: torch.Tensor, down: torch.Tensor | None) -> MLP:
    keep = int(selection.numel())
    child = MLP(int(parent.gate_proj.weight.shape[1]), keep)
    with torch.no_grad():
        child.gate_proj.weight.copy_(parent.gate_proj.weight.detach()[selection])
        child.up_proj.weight.copy_(parent.up_proj.weight.detach()[selection])
        if down is None:
            child.down_proj.weight.copy_(parent.down_proj.weight.detach()[:, selection])
        else:
            child.down_proj.weight.copy_(down.to(child.down_proj.weight.dtype))
    return child.to(parent.gate_proj.weight.dtype)


def mlp_arm_table(
    parent: H3BlockShell,
    folded_mlp: MLP,
    rows: torch.Tensor,
    fit_index: torch.Tensor,
    hold_index: torch.Tensor,
    *,
    keep: int,
    seed: int = 0,
    arms: Sequence[str] = MLP_ARMS,
    selection: torch.Tensor | None = None,
    token_kind: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Score the folded MLP and its three controls on the SAME clip-grouped holdout.

    ``replace``        the coreset child (pivoted verbatim gate/up rows, law-abiding down refit)
    ``control``        a random-init MLP of the child's shape, output-norm matched to the
                       parent on the fit rows -- "a random MLP of this size carries nothing"
    ``random_select``  ``keep`` RANDOM channels with the same law-abiding refit -- prices the
                       pivoted selection (the analogue of the chart-level ``random_chart``)
    ``prior_only``     the pivoted channels with the parent's down_proj columns verbatim, no
                       refit -- prices the refit

    Targets are the parent MLP's exact outputs on the same rows; scores are
    ``linalg.write_scores`` (rel_l2 / cos / cos_p05 / r2). Ordering only: behaviour is the
    gate. With ``token_kind`` (one entry per row) every arm also carries
    ``fit_scores_by_kind`` / ``holdout_scores_by_kind`` (audio / video / text rows scored
    separately), because a pooled av score is dominated by whichever kind has more rows.

    ``prior_only`` scores the PIVOTED channels (``selection``, or recovered from the
    folded child's verbatim rows); ``random_select`` draws its own ``random_selection``
    and never touches the pivoted one. Both arms record ``selected_channels_sha256`` so
    the receipt says which channels were scored.
    """

    from . import receipts as receipt_module

    with torch.no_grad():
        rows32 = rows.detach().to(torch.float32)
        weight_dtype = parent.mlp.gate_proj.weight.dtype
        target = parent.mlp(rows32.to(weight_dtype)).to(torch.float64)
        hidden = parent.mlp.hidden(rows32.to(weight_dtype)).to(torch.float64)
    pivoted = selection if selection is not None else pivot_selection_of(folded_mlp, parent.mlp)
    pivoted = torch.sort(pivoted.to(torch.long)).values
    kinds = None if token_kind is None else token_kind.reshape(-1).to(torch.long)
    if kinds is not None and int(kinds.numel()) != int(rows32.shape[0]):
        raise ValueError(f"token_kind has {int(kinds.numel())} entries for {int(rows32.shape[0])} rows")
    table: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(int(seed))
    for arm in arms:
        notes: dict[str, Any] = {}
        if arm == "replace":
            child = folded_mlp
            notes = {"selected_channels_sha256": receipt_module.hash_tensor(pivoted)}
        elif arm == "control":
            child = MLP(int(parent.mlp.gate_proj.weight.shape[1]), int(keep)).to(weight_dtype)
            with torch.no_grad():
                for p in child.parameters():
                    p.copy_(torch.randn(p.shape, generator=generator) * (1.0 / math.sqrt(p.shape[1])))
                raw = child(rows32[fit_index].to(weight_dtype)).to(torch.float64)
                scale = float(target[fit_index].norm() / raw.norm().clamp_min(1.0e-30))
                child.down_proj.weight.mul_(scale)
            notes = {"norm_matched_to_parent_on_fit_rows": True, "down_scale": scale}
        elif arm == "random_select":
            perm = torch.randperm(int(parent.mlp.gate_proj.weight.shape[0]), generator=generator)
            random_selection = torch.sort(perm[: int(keep)]).values
            prior = parent.mlp.down_proj.weight.detach().to(torch.float64)[:, random_selection]
            weight, solve = law_abiding_solve(hidden[fit_index][:, random_selection], target[fit_index], prior)
            child = _narrowed_mlp(parent.mlp, random_selection, weight)
            notes = {
                "solve": {k: v for k, v in solve.items() if k != "ladder"},
                "selected_channels_sha256": receipt_module.hash_tensor(random_selection),
                "channels_shared_with_pivoted": int(torch.isin(random_selection, pivoted).sum()),
            }
        elif arm == "prior_only":
            child = _narrowed_mlp(parent.mlp, pivoted, None)
            notes = {
                "down_proj": "parent columns verbatim (no refit)",
                "selected_channels_sha256": receipt_module.hash_tensor(pivoted),
                "channels": "the pivoted selection (same channels as replace)",
            }
        else:
            raise ValueError(f"unknown arm {arm!r}")
        with torch.no_grad():
            prediction = child(rows32.to(weight_dtype)).to(torch.float64)
        entry = {
            "arm": arm,
            "fit_scores": write_scores(prediction[fit_index], target[fit_index]),
            "holdout_scores": write_scores(prediction[hold_index], target[hold_index]) if int(hold_index.numel()) else {},
            "child_mlp_parameters": int(sum(int(p.numel()) for p in child.parameters())),
            **notes,
        }
        if kinds is not None:
            entry["fit_scores_by_kind"] = scores_by_kind(prediction, target, fit_index, kinds)
            entry["holdout_scores_by_kind"] = scores_by_kind(prediction, target, hold_index, kinds)
        table.append(entry)
    return table


def scores_by_kind(
    prediction: torch.Tensor, target: torch.Tensor, index: torch.Tensor, kinds: torch.Tensor
) -> dict[str, dict[str, Any]]:
    """``write_scores`` per token kind over ``index`` (kinds absent from the index are omitted)."""

    out: dict[str, dict[str, Any]] = {}
    index = index.reshape(-1).to(torch.long)
    if int(index.numel()) == 0:
        return out
    kind_of_index = kinds[index]
    for code, name in KIND_NAMES.items():
        picked = index[kind_of_index == code]
        if int(picked.numel()):
            out[name] = write_scores(prediction[picked], target[picked])
    return out


def pivot_selection_of(child: MLP, parent: MLP) -> torch.Tensor:
    """Recover which parent channels a verbatim-row child kept (rows are exact copies)."""

    parent_rows = parent.gate_proj.weight.detach().to(torch.float32)
    child_rows = child.gate_proj.weight.detach().to(torch.float32)
    selection = []
    for row in child_rows:
        match = torch.nonzero((parent_rows == row).all(dim=1)).flatten()
        if match.numel() == 0:
            raise ValueError("child gate row is not a verbatim parent row")
        selection.append(int(match[0].item()))
    return torch.tensor(selection, dtype=torch.long)


# --------------------------------------------------------------------------------------
# capture files: composite/media/h3_blocks/step{s}/L{i}.safetensors + blocks/L{i}.safetensors
# --------------------------------------------------------------------------------------

CAPTURE_ROW_TENSORS = ("x_in", "x_out", "mlp_in", "mlp_out", "attn_out")
CAPTURE_META_TENSORS = ("token_kind", "token_frame", "token_clip", "forward_index")
CAPTURE_FORWARD_TENSORS = ("sigma_video", "sigma_audio", "temb", "adaln", "mod_segments")


def capture_path(root: Any, step: int, block: int):
    from pathlib import Path

    return Path(root) / f"step{int(step):02d}" / f"L{int(block):02d}.safetensors"


def block_weights_path(root: Any, block: int):
    from pathlib import Path

    return Path(root) / "blocks" / f"L{int(block):02d}.safetensors"


def concat_captures(states: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate several step files of one block along rows, renumbering forwards globally.

    ``forward_index`` is per file; across files it is offset by the running forward count
    so the random rung's forward grouping stays unique. ``sigma_video`` is expanded to one
    entry per ROW (``sigma_video_rows``) so a row can be filtered by sigma downstream.
    """

    if not states:
        raise ValueError("no capture states")
    out: dict[str, list[torch.Tensor]] = {name: [] for name in (*CAPTURE_ROW_TENSORS, *CAPTURE_META_TENSORS)}
    sigma_rows: list[torch.Tensor] = []
    step_rows: list[torch.Tensor] = []
    offset = 0
    for step, state in enumerate(states):
        forwards = state["forward_index"].reshape(-1).to(torch.long)
        count = int(forwards.max().item()) + 1 if forwards.numel() else 0
        for name in CAPTURE_ROW_TENSORS:
            out[name].append(state[name])
        for name in ("token_kind", "token_frame", "token_clip"):
            out[name].append(state[name].reshape(-1))
        out["forward_index"].append(forwards + offset)
        sigma_rows.append(state["sigma_video"].reshape(-1).to(torch.float32)[forwards])
        step_rows.append(torch.full((int(forwards.numel()),), step, dtype=torch.long))
        offset += count
    merged = {name: torch.cat(values, dim=0) for name, values in out.items()}
    merged["sigma_video_rows"] = torch.cat(sigma_rows)
    merged["step_rows"] = torch.cat(step_rows)
    rows = int(merged["mlp_in"].shape[0])
    for name in (*CAPTURE_ROW_TENSORS, *CAPTURE_META_TENSORS, "sigma_video_rows", "step_rows"):
        if int(merged[name].shape[0]) != rows:
            raise ValueError(f"{name} has {int(merged[name].shape[0])} rows; expected {rows}")
    return merged


def whole_block_receipt(parent: H3BlockShell, child: H3BlockShell) -> dict[str, Any]:
    """Per-component (MLP) and whole-block ratios with the kv-head-merge arm named, not built."""

    from .linalg import compression_receipt

    return {
        "component": compression_receipt(parent.mlp_parameters(), child.mlp_parameters(), "h3.dit.block.mlp"),
        "whole_block": compression_receipt(parent.block_parameters(), child.block_parameters(), "h3.dit.block"),
        "frozen_parameters": int(parent.frozen_parameters),
        "frozen_names": list(parent.frozen_names),
        "kv_head_merge_arm": (
            "NOT BUILT (rung 2): H3 attention is 56-head MHA (qkv_proj [21504, 5376], no kv sharing); "
            "the whole-block ratio above is the MLP coreset alone"
        ),
    }
