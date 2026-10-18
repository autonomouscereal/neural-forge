"""Measured-tubelet video pairing for C15 absorption.

Why this module exists
----------------------
The uniform-tubelet sampler in :mod:`mediafold.c15_pairs` assumes each latent frame owns
``temporal_stride`` consecutive raw frames. The MiniMax H3 video VAE does not work that
way, and the probe receipt says so in numbers
(``out/vae_probe/minimax_h3_video_vae_fp16_probe.json``): comfy's own frames -> latent
map is ``max(1, (frames - 5) // 17 * 5 + 2)`` (sd.py:949), i.e. 17 raw frames per 5
latent frames = 3.4, and a per-frame perturbation probe on a 22-frame clip measured the
ownership groups

    latent 0 -> [0]            latent 4 -> [13, 14, 15, 16]
    latent 1 -> [1, 2, 3]      latent 5 -> [17]
    latent 2 -> [4, 5, 6, 7]   latent 6 -> [18, 19, 20, 21]
    latent 3 -> [8 .. 12]

so group sizes are 1, 3, 4, 5 -- NOT a constant 4. Pairing raw tubelets to latent frames
1:1 at stride 4 therefore misaligns nearly every row, and a C15 map fitted on those rows
is being asked to predict a latent from frames that only partly produced it.

What this module does instead
-----------------------------
1. :func:`measure_frame_groups` MEASURES the ownership map for the loaded VAE by
   perturbing one raw frame at a time and recording which latent frame moves most. One
   encode per frame; on the H3 video VAE at 64-128 px that is well under a second per
   frame on a 3090.
2. :func:`sample_grouped_video_pairs` builds rows only from latent frames whose measured
   group has EXACTLY ``group_size`` contiguous raw frames, so every row is an exact
   (``group_size`` raw frames x patch) <-> (one latent frame x latent group) alignment.
   Latent frames with a different group size are skipped and counted in the receipt.

Feature ordering is Chimera's: the raw row is ``F.unfold`` over ``[B, g*C, H, W]``, which
is exactly what ``chimera_omni.native_media.video_to_native_patches`` produces for
``tubelet=g, patch=p`` (index ``((t*C + c)*p + ph)*p + pw``), so a fitted ``E`` composes
straight into ``NativeContinuousMediaBridge.feature_proj`` for that band.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .c15_pairs import PairSet, _clip_split
from .vae_adapters import VaeAdapter, procedural_video


def _perturbation(channels: int, size: int, amplitude: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    return amplitude * torch.randn(int(channels), int(size), int(size), generator=generator, dtype=torch.float32)


def measure_frame_groups(
    adapter: VaeAdapter,
    *,
    frames: int = 22,
    size: int = 64,
    amplitude: float = 0.25,
    seed: int = 20260902,
    clip: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Measure which latent frame each raw frame writes into.

    Returns a receipt block with ``owner_per_frame``, ``groups`` (latent index -> raw
    frame list), ``group_sizes``, ``contiguous`` and the per-frame influence rows, so a
    later reader can re-derive the pairing decision rather than trust this function.
    """

    base = procedural_video(1, frames=int(frames), size=int(size), seed=int(seed)) if clip is None else clip
    if base.dim() != 5 or int(base.shape[0]) != 1:
        raise ValueError("measure_frame_groups expects a single [1, T, C, H, W] clip")
    with torch.no_grad():
        reference = adapter.encode(base).float().cpu()
    if reference.dim() != 5:
        raise ValueError(f"expected a [B, Cz, Tz, Hz, Wz] video latent, got {tuple(reference.shape)}")
    latent_frames = int(reference.shape[2])
    noise = _perturbation(int(base.shape[2]), int(base.shape[-1]), float(amplitude), int(seed) + 1)

    owner: list[int] = []
    influence: list[list[float]] = []
    for index in range(int(base.shape[1])):
        moved = base.clone()
        moved[0, index] = moved[0, index] + noise.to(moved.dtype)
        with torch.no_grad():
            latent = adapter.encode(moved).float().cpu()
        delta = (latent - reference).abs().mean(dim=(0, 1, 3, 4))
        owner.append(int(torch.argmax(delta).item()))
        influence.append([float(value) for value in delta])

    groups: dict[int, list[int]] = {index: [] for index in range(latent_frames)}
    for frame, latent_index in enumerate(owner):
        groups[latent_index].append(frame)
    contiguous = {
        index: bool(len(rows) == 0 or rows == list(range(rows[0], rows[0] + len(rows))))
        for index, rows in groups.items()
    }
    return {
        "frames": int(base.shape[1]),
        "size": int(base.shape[-1]),
        "latent_frames": latent_frames,
        "amplitude": float(amplitude),
        "owner_per_frame": owner,
        "groups": {str(index): rows for index, rows in groups.items()},
        "group_sizes": sorted({len(rows) for rows in groups.values() if rows}),
        "contiguous": {str(index): value for index, value in contiguous.items()},
        "influence": influence,
        "uniform": bool(len({len(rows) for rows in groups.values() if rows}) == 1),
    }


def _patch_rows(block: torch.Tensor, patch: int) -> torch.Tensor:
    """``[B, g, C, H, W]`` -> ``[B, patches, g*C*patch*patch]`` in Chimera's order."""

    batch, group, channels, height, width = block.shape
    grid_h = int(height) // int(patch)
    grid_w = int(width) // int(patch)
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("frame is smaller than one patch")
    cropped = block[..., : grid_h * int(patch), : grid_w * int(patch)]
    stacked = cropped.reshape(batch, group * channels, grid_h * int(patch), grid_w * int(patch))
    return F.unfold(stacked, kernel_size=int(patch), stride=int(patch)).transpose(1, 2).contiguous()


def sample_grouped_video_pairs(
    adapter: VaeAdapter,
    clips: torch.Tensor,
    *,
    group_map: dict[str, Any],
    group_size: int = 4,
    latent_group: int = 1,
    holdout_fraction: float = 0.25,
    seed: int = 0,
    batch: int = 2,
) -> PairSet:
    """Pair each qualifying latent frame with the raw frames that MEASURABLY produced it.

    ``group_map`` is the receipt returned by :func:`measure_frame_groups` for clips of
    the same length; only latent frames whose group is contiguous and exactly
    ``group_size`` frames long are paired.
    """

    geometry = adapter.geometry
    patch = int(geometry.spatial_stride) * int(latent_group)
    if patch <= 0:
        raise ValueError("video adapters must report spatial_stride")
    if clips.dim() != 5:
        raise ValueError("clips must be [clips, frames, channels, height, width]")
    if int(clips.shape[1]) != int(group_map["frames"]):
        raise ValueError(
            f"group map was measured on {group_map['frames']}-frame clips; these are {int(clips.shape[1])}"
        )

    groups = {int(index): list(rows) for index, rows in group_map["groups"].items()}
    contiguous = {int(index): bool(value) for index, value in group_map["contiguous"].items()}
    selected = [
        index
        for index in sorted(groups)
        if len(groups[index]) == int(group_size) and contiguous.get(index, False)
    ]
    if not selected:
        raise ValueError(
            f"no latent frame owns exactly {group_size} contiguous raw frames; measured sizes "
            f"{group_map['group_sizes']}"
        )

    raw_rows: list[torch.Tensor] = []
    round_rows: list[torch.Tensor] = []
    latent_rows: list[torch.Tensor] = []
    clip_ids: list[torch.Tensor] = []
    frame_ids: list[torch.Tensor] = []

    for start in range(0, int(clips.shape[0]), int(batch)):
        block = clips[start : start + int(batch)].to(torch.float32)
        with torch.no_grad():
            latent = adapter.encode(block).float().cpu()
            reconstruction = adapter.decode(latent).float().cpu()
        if int(reconstruction.shape[1]) < int(block.shape[1]):
            raise ValueError("parent reconstruction is shorter than the input clip")
        _, latent_channels, latent_frames, height_z, width_z = latent.shape
        grid_h = int(height_z) // int(latent_group)
        grid_w = int(width_z) // int(latent_group)
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError("clip is smaller than one latent group")

        for latent_index in selected:
            if latent_index >= latent_frames:
                continue
            frames = groups[latent_index]
            raw_flat = _patch_rows(block[:, frames], patch)
            round_flat = _patch_rows(reconstruction[:, frames], patch)
            piece = latent[:, :, latent_index, : grid_h * int(latent_group), : grid_w * int(latent_group)]
            piece = piece.reshape(
                int(block.shape[0]), latent_channels, grid_h, int(latent_group), grid_w, int(latent_group)
            ).permute(0, 2, 4, 1, 3, 5).contiguous()
            latent_flat = piece.reshape(int(block.shape[0]), grid_h * grid_w, -1)
            if int(raw_flat.shape[1]) != int(latent_flat.shape[1]):
                raise ValueError(
                    f"spatial mismatch: {int(raw_flat.shape[1])} raw patches vs "
                    f"{int(latent_flat.shape[1])} latent groups"
                )
            for local in range(int(block.shape[0])):
                raw_rows.append(raw_flat[local])
                round_rows.append(round_flat[local])
                latent_rows.append(latent_flat[local])
                clip_ids.append(torch.full((int(raw_flat.shape[1]),), start + local, dtype=torch.long))
                frame_ids.append(torch.full((int(raw_flat.shape[1]),), latent_index, dtype=torch.long))

    raw = torch.cat(raw_rows, dim=0)
    roundtrip = torch.cat(round_rows, dim=0)
    latent_matrix = torch.cat(latent_rows, dim=0)
    clip_index = torch.cat(clip_ids)
    flag = _clip_split(int(clips.shape[0]), holdout_fraction, seed)
    return PairSet(
        raw=raw,
        latent=latent_matrix,
        roundtrip=roundtrip,
        clip_index=clip_index,
        frame_index=torch.cat(frame_ids),
        channel_index=torch.full((int(raw.shape[0]),), -1, dtype=torch.long),
        split=flag[clip_index],
        geometry={
            **geometry.to_json(),
            "pairing": f"measured_group{int(group_size)}_latent{int(latent_group)}",
            "latent_group": int(latent_group),
            "temporal_group_size": int(group_size),
            "selected_latent_frames": selected,
            "skipped_latent_frames": [index for index in sorted(groups) if index not in selected],
            "measured_group_sizes": group_map["group_sizes"],
            "measured_uniform": bool(group_map["uniform"]),
            "context": 0,
            "d_raw": int(raw.shape[1]),
            "d_latent": int(latent_matrix.shape[1]),
            "d_centre": int(raw.shape[1]),
            "centre_offset": 0,
            "rows": int(raw.shape[0]),
            "clips": int(clips.shape[0]),
            "chimera_patch": patch,
            "chimera_tubelet": int(group_size),
            "chimera_native_feature_dim": int(raw.shape[1]),
            "alignment": "measured per-frame ownership (perturbation probe)",
        },
    )
