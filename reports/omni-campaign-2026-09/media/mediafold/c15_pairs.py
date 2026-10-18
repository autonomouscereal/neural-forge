"""C15 pair sampler: (raw patch <-> VAE latent) rows for closed-form absorption.

The sampler is generic over :class:`~mediafold.vae_adapters.VaeAdapter`. It produces one
row per (clip, frame, channel/patch) alignment, splits held-out data BY CLIP (never by
row -- neighbouring rows of one clip are near-duplicates and would leak), and stores a
small evaluation block containing the parent VAE's own round-trip so the fitter can
score absorption against the ceiling rather than against absolute dB.

Row geometries
--------------
audio ``per_channel``  raw = one channel's ``hop`` samples          -> d_raw = hop
audio ``stacked``      raw = all channels' hop samples concatenated -> d_raw = C*hop

``per_channel`` is only WELL POSED when the parent's latent carries a channel axis
(``geometry.channel_in_latent``). When it does not -- ACE 1.5's AudioOobleckVAE emits
[B, 64, T] jointly over stereo -- the one latent vector has to be broadcast to both
channel rows, so ``E`` is asked to map two different mono frames onto an IDENTICAL
target and ``D`` to map one latent onto two different mono targets. That is an
irreducible ambiguity, not a modelling failure: no map of any width can beat the floor
it imposes, and every score measured through it is a score of the PAIRING. The sampler
therefore REFUSES multi-channel ``per_channel`` on a joint latent unless the caller
passes ``allow_ambiguous_pairing=True`` (Chimera token alignment is a legitimate reason
to want it), and when it is allowed the MEASURED floor is computed and stored in the
geometry as ``ambiguity`` so it travels beside every downstream number.
audio ``+context K``   raw = frames [t-K .. t+K] concatenated       -> d_raw *= (2K+1)
video ``patch``        raw = one tubelet patch                      -> d_raw = tub*C*p*p
video ``+group G``     latent = GxG spatial group (one DiT token)   -> d_lat = Cz*G*G

Chimera alignment (the reason these geometries and not others):

* ``waveform_to_native_segments(segment=hop, stride=hop)`` emits exactly one token per
  channel per VAE frame, so ``per_channel`` rows are 1:1 with Chimera audio tokens.
  At H3's audio VAE this means ``segment=800`` at 32 kHz, NOT the ``3072`` default.
* ``video_to_native_patches(patch=spatial_stride*G, tubelet=temporal_stride)`` emits one
  token per DiT token, so at H3's video VAE (16x spatial, 4x temporal, 2x2 DiT patch)
  the Chimera geometry is ``patch=32, tubelet=4``, ``native_feature_dim = 4*3*32*32 =
  12288`` (proposed decision line D-H3-geom).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from . import receipts
from .vae_adapters import VaeAdapter, build_adapter, procedural_audio, procedural_video


@dataclass
class PairSet:
    raw: torch.Tensor  # [N, d_raw] -- includes context columns when context > 0
    latent: torch.Tensor  # [N, d_lat]
    roundtrip: torch.Tensor  # [N, d_centre] -- the PARENT's own reconstruction, row-aligned
    clip_index: torch.Tensor
    frame_index: torch.Tensor
    channel_index: torch.Tensor
    split: torch.Tensor  # 0 = fit, 1 = holdout (split by clip)
    geometry: dict[str, Any]

    @property
    def centre_width(self) -> int:
        """Width of the row's own frame/patch inside a context-expanded raw row."""

        return int(self.geometry["d_centre"])

    def centre(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Slice the centre frame/patch out of context-expanded raw rows."""

        source = self.raw if rows is None else rows
        offset = int(self.geometry.get("centre_offset", 0))
        return source[:, offset : offset + self.centre_width]

    def fit_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.split.eq(0)
        return self.raw[mask], self.latent[mask]

    def holdout_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.split.eq(1)
        return self.raw[mask], self.latent[mask]

    def state(self) -> dict[str, torch.Tensor]:
        return {
            "raw": self.raw,
            "latent": self.latent,
            "roundtrip": self.roundtrip,
            "clip_index": self.clip_index,
            "frame_index": self.frame_index,
            "channel_index": self.channel_index,
            "split": self.split,
        }


def _clip_split(count: int, holdout_fraction: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(int(count), generator=generator)
    held = max(1, int(round(float(holdout_fraction) * count))) if holdout_fraction > 0.0 else 0
    flag = torch.zeros(int(count), dtype=torch.long)
    flag[order[:held]] = 1
    return flag


def _with_context(rows: torch.Tensor, context: int) -> torch.Tensor:
    """Concatenate +/- ``context`` neighbouring rows along the row axis (edge-replicated).

    ``rows`` is [frames, width] for one clip/channel.
    """

    if context <= 0:
        return rows
    frames = rows.shape[0]
    pieces = []
    for offset in range(-int(context), int(context) + 1):
        index = torch.arange(frames) + offset
        index = index.clamp(0, frames - 1)
        pieces.append(rows[index])
    return torch.cat(pieces, dim=-1)


def ambiguity_floor(
    centre_rows: torch.Tensor,
    clip_index: torch.Tensor,
    frame_index: torch.Tensor,
    latent: torch.Tensor,
) -> dict[str, Any]:
    """Measure the floor a many-to-one pairing imposes on ANY decoder.

    When several rows share one latent target, the best possible ``D`` predicts their
    MEAN, so the within-group spread is irreducible. Reported as the relative L2 of that
    spread and the PSNR a perfect map could reach -- the number every score of a rung
    fitted through this pairing has to be read against.
    """

    rows = centre_rows.detach().to(torch.float64)
    keys = clip_index.to(torch.long) * (int(frame_index.max().item()) + 1) + frame_index.to(torch.long)
    unique, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
    totals = torch.zeros(int(unique.numel()), rows.shape[1], dtype=torch.float64)
    totals.index_add_(0, inverse, rows)
    means = totals / counts.to(torch.float64).unsqueeze(1)
    residual = rows - means[inverse]
    denominator = rows.norm().clamp_min(1.0e-30)
    peak = rows.abs().max().clamp_min(1.0e-30)
    mse = residual.pow(2).mean().clamp_min(1.0e-30)
    unique_latents = int(torch.unique(latent.detach().to(torch.float64), dim=0).shape[0])
    return {
        "groups": int(unique.numel()),
        "rows": int(rows.shape[0]),
        "rows_per_group_max": int(counts.max().item()),
        "unique_latent_rows": unique_latents,
        "irreducible_rel_l2": float((residual.norm() / denominator).item()),
        "best_possible_psnr_db": float((20.0 * torch.log10(peak / mse.sqrt())).item()),
        "note": (
            "several raw rows share one latent target; no map can do better than their mean, "
            "so this floor must be quoted beside any score fitted through this pairing"
        ),
    }


def sample_audio_pairs(
    adapter: VaeAdapter,
    clips: torch.Tensor,
    *,
    pairing: str = "per_channel",
    context: int = 0,
    holdout_fraction: float = 0.25,
    seed: int = 0,
    batch: int = 4,
    allow_ambiguous_pairing: bool = False,
) -> PairSet:
    """Align raw waveform frames with the adapter's latent frames.

    ``clips`` is [clips, channels, samples]; the trailing partial frame is dropped so
    every row is exact.
    """

    geometry = adapter.geometry
    hop = int(geometry.hop_samples)
    if hop <= 0:
        raise ValueError("audio adapters must report hop_samples")
    channels = int(clips.shape[1])
    ambiguous = (
        pairing == "per_channel"
        and channels > 1
        and not bool(geometry.channel_in_latent)
    )
    if ambiguous and not allow_ambiguous_pairing:
        raise ValueError(
            f"{geometry.name}'s latent is JOINT over {channels} waveform channels "
            "(channel_in_latent=False), so 'per_channel' pairing would broadcast one latent "
            "to every channel row: E would see two different inputs mapped to an identical "
            "target and D one latent mapped to two different targets. Use "
            "pairing='stacked' (d_raw = channels*hop), or pass allow_ambiguous_pairing=True "
            "to keep the Chimera-token geometry and have the measured ambiguity floor "
            "recorded in geometry['ambiguity']."
        )

    raw_rows: list[torch.Tensor] = []
    round_rows: list[torch.Tensor] = []
    latent_rows: list[torch.Tensor] = []
    clip_ids: list[torch.Tensor] = []
    frame_ids: list[torch.Tensor] = []
    channel_ids: list[torch.Tensor] = []

    def chop(waveform: torch.Tensor, count: int) -> torch.Tensor:
        trimmed = waveform[..., : count * hop].to(torch.float32)
        return trimmed.reshape(trimmed.shape[0], channels, count, hop)

    for start in range(0, clips.shape[0], int(batch)):
        block = clips[start : start + int(batch)]
        latent = adapter.encode(block).to(torch.float32)
        # normalise every latent layout to [B, T, C, d_lat_per_channel]
        if latent.dim() == 4 and geometry.channel_in_latent:
            frames = latent.shape[-1]  # [B, Cz, C, T]
            latent_by_channel = latent.permute(0, 3, 2, 1)
        elif latent.dim() == 3:
            frames = latent.shape[-1]  # [B, Cz, T] shared across channels
            latent_by_channel = latent.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, channels, -1)
        else:
            raise ValueError(f"unsupported audio latent layout {tuple(latent.shape)}")

        usable = min(frames, block.shape[-1] // hop)
        if usable <= 0:
            raise ValueError("no complete latent frame in this clip block")
        reconstruction = adapter.decode(latent).to(torch.float32)
        if reconstruction.shape[-1] < usable * hop:
            usable = reconstruction.shape[-1] // hop
        chopped = chop(block, usable)
        chopped_round = chop(reconstruction, usable)
        latent_by_channel = latent_by_channel[:, :usable]

        for local in range(block.shape[0]):
            clip = start + local
            if pairing == "per_channel":
                for channel in range(channels):
                    raw_rows.append(_with_context(chopped[local, channel], int(context)))
                    round_rows.append(chopped_round[local, channel])
                    latent_rows.append(latent_by_channel[local, :, channel])
                    clip_ids.append(torch.full((usable,), clip, dtype=torch.long))
                    frame_ids.append(torch.arange(usable, dtype=torch.long))
                    channel_ids.append(torch.full((usable,), channel, dtype=torch.long))
            elif pairing == "stacked":
                stacked = chopped[local].permute(1, 0, 2).reshape(usable, channels * hop)
                stacked_round = chopped_round[local].permute(1, 0, 2).reshape(usable, channels * hop)
                raw_rows.append(_with_context(stacked, int(context)))
                round_rows.append(stacked_round)
                # On a JOINT latent, `latent_by_channel` is the same vector broadcast across
                # the channel axis, so reshaping it here would emit `channels * Cz` columns
                # that are literally the same numbers twice -- a duplicated target half,
                # which doubles the composed head's width for nothing and makes the pair set
                # unusable as a DiT stimulus. Take the latent ONCE when the parent produced
                # it once; concatenate only when the parent really has a per-channel latent.
                if geometry.channel_in_latent:
                    latent_rows.append(latent_by_channel[local].reshape(usable, -1))
                else:
                    latent_rows.append(latent_by_channel[local][:, 0])
                clip_ids.append(torch.full((usable,), clip, dtype=torch.long))
                frame_ids.append(torch.arange(usable, dtype=torch.long))
                channel_ids.append(torch.full((usable,), -1, dtype=torch.long))
            else:
                raise ValueError(f"unknown audio pairing {pairing!r}")

    raw = torch.cat(raw_rows, dim=0)
    roundtrip = torch.cat(round_rows, dim=0)
    latent_matrix = torch.cat(latent_rows, dim=0)
    clip_index = torch.cat(clip_ids)
    frame_index = torch.cat(frame_ids)
    flag = _clip_split(int(clips.shape[0]), holdout_fraction, seed)
    centre_width = hop if pairing == "per_channel" else channels * hop
    centre_offset = int(context) * centre_width
    ambiguity: dict[str, Any] | None = None
    if ambiguous:
        ambiguity = ambiguity_floor(
            raw[:, centre_offset : centre_offset + centre_width], clip_index, frame_index, latent_matrix
        )
        ambiguity["cause"] = (
            f"{geometry.name} emits a latent joint over {channels} waveform channels; "
            "per_channel pairing was explicitly allowed by the caller"
        )
    return PairSet(
        raw=raw,
        latent=latent_matrix,
        roundtrip=roundtrip,
        clip_index=clip_index,
        frame_index=frame_index,
        channel_index=torch.cat(channel_ids),
        split=flag[clip_index],
        geometry={
            **geometry.to_json(),
            "pairing": pairing,
            "ambiguous_pairing": bool(ambiguous),
            "ambiguity": ambiguity,
            "context": int(context),
            "d_raw": int(raw.shape[1]),
            "d_latent": int(latent_matrix.shape[1]),
            "d_centre": int(centre_width),
            "centre_offset": centre_offset,
            "rows": int(raw.shape[0]),
            "clips": int(clips.shape[0]),
            "chimera_segment": hop,
            "chimera_stride": hop,
            "chimera_tokens_per_frame": channels,
        },
    )


def sample_video_pairs(
    adapter: VaeAdapter,
    clips: torch.Tensor,
    *,
    latent_group: int = 1,
    holdout_fraction: float = 0.25,
    seed: int = 0,
    batch: int = 2,
    frame_head: int = 0,
    latent_head: int = 0,
    strict_alignment: bool = True,
) -> PairSet:
    """Align raw tubelet patches with latent patches (optionally GxG DiT-token groups).

    ``clips`` is [clips, frames, channels, height, width].

    ``frame_head`` / ``latent_head`` drop a leading, NON-uniform prologue before the
    uniform tail is paired. Real video VAEs are causal: MiniMax H3's frames -> latents
    map is ``max(1, (frames - 5) // 17 * 5 + 2)`` (comfy sd.py:949), i.e. a 5-frame /
    2-latent head followed by 17-frame / 5-latent groups, so pairing frame blocks to
    latent frames 1:1 across the head would misalign every row. ``strict_alignment``
    asserts that what is left after the head divides exactly.
    """

    geometry = adapter.geometry
    spatial = int(geometry.spatial_stride) * int(latent_group)
    temporal = int(geometry.temporal_stride)
    if spatial <= 0 or temporal <= 0:
        raise ValueError("video adapters must report spatial_stride and temporal_stride")
    channels = int(clips.shape[2])

    raw_rows: list[torch.Tensor] = []
    round_rows: list[torch.Tensor] = []
    latent_rows: list[torch.Tensor] = []
    clip_ids: list[torch.Tensor] = []
    frame_ids: list[torch.Tensor] = []

    def patchify(video: torch.Tensor, tubelets: int, group_h: int, group_w: int) -> torch.Tensor:
        block = video[:, : tubelets * temporal, :, : group_h * spatial, : group_w * spatial]
        block = block.reshape(
            block.shape[0], tubelets, temporal, channels, group_h, spatial, group_w, spatial
        ).permute(0, 1, 4, 6, 2, 3, 5, 7).contiguous()
        return block.reshape(block.shape[0], tubelets * group_h * group_w, -1)

    for start in range(0, clips.shape[0], int(batch)):
        block = clips[start : start + int(batch)].to(torch.float32)
        latent = adapter.encode(block).to(torch.float32)  # [B, Cz, Tz, Hz, Wz]
        if latent.dim() != 5:
            raise ValueError(f"unsupported video latent layout {tuple(latent.shape)}")
        reconstruction = adapter.decode(latent).to(torch.float32)

        # drop the causal prologue before anything is paired
        latent = latent[:, :, int(latent_head) :]
        aligned = block[:, int(frame_head) :]
        reconstruction = reconstruction[:, int(frame_head) : int(frame_head) + aligned.shape[1]]

        _, latent_channels, tz, hz, wz = latent.shape
        group_h = hz // int(latent_group)
        group_w = wz // int(latent_group)
        tubelets = min(tz, aligned.shape[1] // temporal)
        if tubelets <= 0 or group_h <= 0 or group_w <= 0:
            raise ValueError("clip is smaller than one latent group")
        if strict_alignment and (tz != aligned.shape[1] // temporal or aligned.shape[1] % temporal):
            raise ValueError(
                f"latent frames {tz} do not line up with {aligned.shape[1]} raw frames at temporal "
                f"stride {temporal} after a {frame_head}-frame / {latent_head}-latent head; pass a "
                "clip length this VAE maps uniformly, or set strict_alignment=False and record it"
            )

        raw_flat = patchify(aligned, tubelets, group_h, group_w)
        round_flat = patchify(reconstruction, tubelets, group_h, group_w)

        latent_block = latent[:, :, :tubelets, : group_h * int(latent_group), : group_w * int(latent_group)]
        latent_block = latent_block.reshape(
            aligned.shape[0], latent_channels, tubelets, group_h, int(latent_group), group_w, int(latent_group)
        ).permute(0, 2, 3, 5, 1, 4, 6).contiguous()
        latent_flat = latent_block.reshape(aligned.shape[0], tubelets * group_h * group_w, -1)

        for local in range(block.shape[0]):
            raw_rows.append(raw_flat[local])
            round_rows.append(round_flat[local])
            latent_rows.append(latent_flat[local])
            clip_ids.append(torch.full((raw_flat.shape[1],), start + local, dtype=torch.long))
            frame_ids.append(torch.arange(raw_flat.shape[1], dtype=torch.long))

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
        channel_index=torch.full((raw.shape[0],), -1, dtype=torch.long),
        split=flag[clip_index],
        geometry={
            **geometry.to_json(),
            "pairing": f"patch_group{latent_group}",
            "latent_group": int(latent_group),
            "context": 0,
            "d_raw": int(raw.shape[1]),
            "d_latent": int(latent_matrix.shape[1]),
            "d_centre": int(raw.shape[1]),
            "centre_offset": 0,
            "rows": int(raw.shape[0]),
            "clips": int(clips.shape[0]),
            "chimera_patch": spatial,
            "chimera_tubelet": temporal,
            "chimera_native_feature_dim": int(raw.shape[1]),
            "frame_head": int(frame_head),
            "latent_head": int(latent_head),
            "strict_alignment": bool(strict_alignment),
        },
    )


def write_pair_set(
    pairs: PairSet,
    output_dir: str | Path,
    *,
    evaluation: dict[str, torch.Tensor] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    pair_receipt = receipts.write_tensors(pairs.state(), target / "pairs.safetensors")
    body: dict[str, Any] = {
        "kind": "c15_pairs",
        "geometry": pairs.geometry,
        "pairs": pair_receipt,
        "split": {
            "fit_rows": int(pairs.split.eq(0).sum().item()),
            "holdout_rows": int(pairs.split.eq(1).sum().item()),
            "split_by": "clip",
        },
        "runtime": receipts.runtime_block(),
    }
    if evaluation:
        body["evaluation"] = receipts.write_tensors(evaluation, target / "evaluation.safetensors")
    if extra:
        body.update(extra)
    receipt_hash = receipts.write_receipt(target / "pairs_receipt.json", body)  # the receipt cannot contain its own hash; it is printed instead
    return body


def load_pair_set(output_dir: str | Path) -> PairSet:
    target = Path(output_dir)
    state = receipts.read_tensors(target / "pairs.safetensors")
    manifest = json.loads((target / "pairs_receipt.json").read_text(encoding="utf-8"))
    return PairSet(
        raw=state["raw"],
        latent=state["latent"],
        roundtrip=state["roundtrip"],
        clip_index=state["clip_index"],
        frame_index=state["frame_index"],
        channel_index=state["channel_index"],
        split=state["split"],
        geometry=manifest["geometry"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C15 raw<->latent pair sampler")
    parser.add_argument("--adapter", required=True, help="synthetic-audio | synthetic-video | comfy:<path>")
    parser.add_argument("--out", required=True)
    parser.add_argument("--clips", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=1.0, help="audio clip length")
    parser.add_argument("--frames", type=int, default=16, help="video clip length")
    parser.add_argument("--size", type=int, default=64, help="video frame size")
    parser.add_argument("--pairing", default="per_channel", choices=("per_channel", "stacked"))
    parser.add_argument(
        "--allow-ambiguous-pairing",
        action="store_true",
        help=(
            "keep per_channel rows on a VAE whose latent is joint over channels; the measured "
            "ambiguity floor is written into geometry['ambiguity']"
        ),
    )
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--latent-group", type=int, default=1)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--eval-clips", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args(argv)

    adapter = build_adapter(arguments.adapter, device=arguments.device)
    geometry = adapter.geometry
    if geometry.modality == "audio":
        hop = int(geometry.hop_samples)
        samples = max(hop * 8, int(round(arguments.seconds * geometry.sample_rate_hz)) // hop * hop)
        clips = procedural_audio(
            arguments.clips,
            samples=samples,
            channels=max(1, int(geometry.channels) or 2),
            sample_rate_hz=int(geometry.sample_rate_hz),
            seed=arguments.seed,
        )
        pairs = sample_audio_pairs(
            adapter,
            clips,
            pairing=arguments.pairing,
            context=arguments.context,
            holdout_fraction=arguments.holdout_fraction,
            seed=arguments.seed,
            allow_ambiguous_pairing=arguments.allow_ambiguous_pairing,
        )
    else:
        clips = procedural_video(
            arguments.clips,
            frames=arguments.frames,
            size=arguments.size,
            seed=arguments.seed,
        )
        pairs = sample_video_pairs(
            adapter,
            clips,
            latent_group=arguments.latent_group,
            holdout_fraction=arguments.holdout_fraction,
            seed=arguments.seed,
        )

    held = torch.nonzero(pairs.split.eq(1)).flatten()
    held_clips = sorted({int(pairs.clip_index[index].item()) for index in held})[: arguments.eval_clips]
    evaluation: dict[str, torch.Tensor] = {}
    if held_clips:
        block = clips[torch.tensor(held_clips, dtype=torch.long)]
        latent = adapter.encode(block)
        evaluation = {
            "eval_raw": block.to(torch.float32),
            "eval_latent": latent.to(torch.float32),
            "eval_roundtrip": adapter.decode(latent).to(torch.float32),
            "eval_clip_index": torch.tensor(held_clips, dtype=torch.long),
        }

    body = write_pair_set(pairs, arguments.out, evaluation=evaluation, extra={"argv": vars(arguments)})
    print(json.dumps({"geometry": pairs.geometry, "split": body["split"], "pairs_sha256": body["pairs"]["state_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
