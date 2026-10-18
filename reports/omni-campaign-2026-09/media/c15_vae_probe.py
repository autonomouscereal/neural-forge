"""Measure a parent VAE's REAL geometry inside the ComfyUI runtime image.

Every geometry number the C15 pair sampler relies on -- hop, latent channels, latent
layout, the frames -> latent-frames map, whether ``encode`` is deterministic -- is a
claim about someone else's code. This script turns each of them into a measurement with
a receipt, before a single pair row is written. It runs the VAE and nothing else: no
DiT, no sampler, no H3 generation (the H3 licence gate is pending).

    docker run --rm --gpus device=0 \\
      -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES=0 \\
      --entrypoint /opt/environments/python/comfyui/bin/python \\
      -v /home/cereal/comfyui/models:/opt/ComfyUI/models \\
      -v /home/cereal/omni-campaign:/work \\
      comfyui:codex-minimax-h3-runtime-env-r5-20260808 \\
      /work/media/c15_vae_probe.py --vae /opt/ComfyUI/models/vae/ace_1.5_vae.safetensors \\
        --out /work/media/out/vae_probe --device cuda

The r5 image bakes ``CUDA_VISIBLE_DEVICES=""`` and pins ``NVIDIA_VISIBLE_DEVICES`` to a
GPU UUID that is not attached, so both ``-e`` overrides are required for GPU; without
them (and without ``--gpus``) the adapter puts ComfyUI in CPU mode by itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import torch  # noqa: E402

from mediafold import receipts  # noqa: E402
from mediafold.linalg import psnr, si_sdr  # noqa: E402
from mediafold.vae_adapters import (  # noqa: E402
    COMFY_VAE_PROFILES,
    ComfyVaeAdapter,
    procedural_audio,
    procedural_video,
)


def probe_audio(adapter: ComfyVaeAdapter, *, clips: int, seconds: float, seed: int) -> dict[str, Any]:
    geometry = adapter.geometry
    hop = int(geometry.hop_samples)
    channels = max(1, int(geometry.channels) or 2)
    frames = max(8, int(round(seconds * geometry.sample_rate_hz)) // hop)
    samples = frames * hop
    raw = procedural_audio(
        clips, samples=samples, channels=channels, sample_rate_hz=int(geometry.sample_rate_hz), seed=seed
    )

    started = time.monotonic()
    latent = adapter.encode(raw)
    encode_seconds = time.monotonic() - started
    started = time.monotonic()
    reconstruction = adapter.decode(latent)
    decode_seconds = time.monotonic() - started

    again = adapter.encode(raw)
    deterministic = bool(torch.equal(latent.float().cpu(), again.float().cpu()))
    delta = float((latent.float().cpu() - again.float().cpu()).abs().max().item())

    limit = min(int(reconstruction.shape[-1]), int(raw.shape[-1]))
    left = reconstruction[..., :limit].float().cpu()
    right = raw[..., :limit].float().cpu()
    return {
        "raw_shape": list(raw.shape),
        "requested_frames": frames,
        "latent_shape": list(latent.shape),
        "latent_dtype": str(latent.dtype),
        "latent_mean": float(latent.float().mean().item()),
        "latent_std": float(latent.float().std().item()),
        "latent_absmax": float(latent.float().abs().max().item()),
        "reconstruction_shape": list(reconstruction.shape),
        "encode_deterministic": deterministic,
        "encode_repeat_max_delta": delta,
        "roundtrip_psnr_db": psnr(left, right, data_range=2.0),
        "roundtrip_si_sdr_db": si_sdr(left.reshape(left.shape[0], -1), right.reshape(right.shape[0], -1)),
        "encode_seconds": round(encode_seconds, 3),
        "decode_seconds": round(decode_seconds, 3),
        "samples_per_latent_frame_measured": (
            int(raw.shape[-1]) / float(latent.shape[-1]) if latent.shape[-1] else None
        ),
    }


def probe_video(adapter: ComfyVaeAdapter, *, size: int, frame_counts: list[int], seed: int) -> dict[str, Any]:
    table: list[dict[str, Any]] = []
    for count in frame_counts:
        clip = procedural_video(1, frames=int(count), size=int(size), seed=seed)
        try:
            started = time.monotonic()
            latent = adapter.encode(clip)
            encode_seconds = time.monotonic() - started
        except Exception as error:  # a length this VAE refuses is itself a measurement
            table.append({"frames": int(count), "error": f"{type(error).__name__}: {error}"})
            continue
        row = {
            "frames": int(count),
            "latent_shape": list(latent.shape),
            "comfy_formula_latent_frames": adapter.latent_frame_count(int(count)),
            "encode_seconds": round(encode_seconds, 3),
        }
        started = time.monotonic()
        reconstruction = adapter.decode(latent)
        row["decode_seconds"] = round(time.monotonic() - started, 3)
        row["reconstruction_shape"] = list(reconstruction.shape)
        frames_back = min(int(reconstruction.shape[1]), int(clip.shape[1]))
        row["roundtrip_psnr_db"] = psnr(
            reconstruction[:, :frames_back].float().cpu(), clip[:, :frames_back].float().cpu(), data_range=1.0
        )
        row["latent_mean"] = float(latent.float().mean().item())
        row["latent_std"] = float(latent.float().std().item())
        again = adapter.encode(clip)
        row["encode_deterministic"] = bool(torch.equal(latent.float().cpu(), again.float().cpu()))
        row["encode_repeat_max_delta"] = float((latent.float().cpu() - again.float().cpu()).abs().max().item())
        table.append(row)
    return {"size": int(size), "lengths": table}


def probe_video_temporal_support(
    adapter: ComfyVaeAdapter, *, frames: int, size: int, seed: int, amplitude: float = 0.25
) -> dict[str, Any]:
    """Which raw frames does each LATENT frame actually see?

    A video VAE's frames -> latents count (``max(1, (a - 5) // 17 * 5 + 2)`` for MiniMax
    H3) says nothing about which raw frames feed which latent, and C15 pair rows are only
    meaningful if that support is known. So it is measured directly: perturb exactly one
    raw frame, re-encode, and record how far each latent frame moved. The argmax over
    latent frames is that raw frame's owner; a clean partition means a fixed tubelet
    exists, and a ragged one means the pair sampler must be given explicit frame groups.
    """

    base = procedural_video(1, frames=int(frames), size=int(size), seed=seed)
    reference = adapter.encode(base).float().cpu()
    latent_frames = int(reference.shape[2])
    influence: list[list[float]] = []
    owners: list[int] = []
    for index in range(int(frames)):
        probe = base.clone()
        probe[:, index] = (probe[:, index] + float(amplitude)).clamp(0.0, 1.0)
        moved = adapter.encode(probe).float().cpu() - reference
        per_latent = moved[0].abs().amax(dim=(0, 2, 3))  # [Tz]
        influence.append([float(v) for v in per_latent])
        owners.append(int(torch.argmax(per_latent).item()))
    groups: dict[int, list[int]] = {}
    for frame_index, owner in enumerate(owners):
        groups.setdefault(owner, []).append(frame_index)
    sizes = sorted({len(v) for v in groups.values()})
    return {
        "frames": int(frames),
        "latent_frames": latent_frames,
        "amplitude": float(amplitude),
        "owner_per_frame": owners,
        "frame_groups": {str(k): v for k, v in sorted(groups.items())},
        "group_sizes": sizes,
        "uniform_tubelet": (len(sizes) == 1),
        "latents_touched": len(groups),
        "influence": influence,
    }


def probe_audio_temporal_support(
    adapter: ComfyVaeAdapter, *, frames: int, seed: int, amplitude: float = 0.25
) -> dict[str, Any]:
    """Which raw FRAMES does each latent frame actually see, on the audio side?

    The video side has had this measurement since the ownership map turned out not to be a
    uniform tubelet; the audio side asserted a one-frame support and a ``+/-K`` context
    window instead of measuring one. Same method: perturb exactly one ``hop``-sample frame,
    re-encode, and record how far each latent frame moved. Reported as the owner per frame,
    the mean normalised influence at each offset from the owner, and the support WIDTH --
    the number of offsets carrying more than ``support_threshold`` of the peak. A support
    of 1 justifies ``context=0``; a support of 5 is what a ``+/-2`` context row is for.
    """

    geometry = adapter.geometry
    hop = int(geometry.hop_samples)
    channels = max(1, int(geometry.channels) or 1)
    clip = procedural_audio(
        1, samples=hop * int(frames), channels=channels,
        sample_rate_hz=int(geometry.sample_rate_hz), seed=seed,
    )
    reference = adapter.encode(clip).float().cpu()
    latent_axis = -1  # every audio layout in the table ends on the time axis
    latent_frames = int(reference.shape[latent_axis])

    owners: list[int] = []
    influence: list[list[float]] = []
    for index in range(int(frames)):
        probe = clip.clone()
        probe[..., index * hop : (index + 1) * hop] += float(amplitude)
        moved = adapter.encode(probe).float().cpu() - reference
        flat = moved.abs().reshape(-1, latent_frames)
        per_latent = flat.amax(dim=0)
        owners.append(int(torch.argmax(per_latent).item()))
        peak = float(per_latent.max())
        influence.append([float(value) / peak if peak > 0 else 0.0 for value in per_latent])

    # average the influence profile in OWNER-RELATIVE coordinates
    offsets: dict[int, list[float]] = {}
    for frame_index, row in enumerate(influence):
        owner = owners[frame_index]
        for latent_index, value in enumerate(row):
            offsets.setdefault(latent_index - owner, []).append(value)
    profile = {
        str(offset): sum(values) / len(values)
        for offset, values in sorted(offsets.items())
        if abs(offset) <= 6
    }
    threshold = 0.1
    support = [offset for offset, value in profile.items() if value >= threshold]
    return {
        "frames": int(frames),
        "hop_samples": hop,
        "latent_frames": latent_frames,
        "amplitude": float(amplitude),
        "owner_per_frame": owners,
        "owner_is_monotonic": owners == sorted(owners),
        "influence_by_offset": profile,
        "support_threshold": threshold,
        "support_offsets": sorted(support, key=int),
        "support_width": len(support),
        "meaning": (
            "support_width 1 justifies context=0; a wider support is what a +/-K context row "
            "is for. Measured, not asserted."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="measure a parent VAE's real geometry")
    parser.add_argument("--vae", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clips", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--frame-counts", default="5,9,17,22")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--support-frames",
        type=int,
        default=0,
        help="video: measure which raw frames each latent frame sees, at this clip length",
    )
    parser.add_argument(
        "--audio-support-frames",
        type=int,
        default=0,
        help="audio: measure the temporal support of one latent frame over this many hop frames",
    )
    parser.add_argument("--posterior", default="mean", choices=("mean", "sample"))
    parser.add_argument("--path-mode", default="auto", choices=("auto", "direct", "wrapper"))
    arguments = parser.parse_args(argv)

    started = time.monotonic()
    adapter = ComfyVaeAdapter(arguments.vae, device=arguments.device, posterior=arguments.posterior, path_mode=arguments.path_mode)
    load_seconds = time.monotonic() - started
    geometry = adapter.geometry.to_json()
    print(json.dumps({"geometry": geometry, "load_seconds": round(load_seconds, 3)}, indent=2), flush=True)

    if adapter.modality == "audio":
        measurement = probe_audio(adapter, clips=arguments.clips, seconds=arguments.seconds, seed=arguments.seed)
        if arguments.audio_support_frames > 0:
            measurement["temporal_support"] = probe_audio_temporal_support(
                adapter, frames=arguments.audio_support_frames, seed=arguments.seed
            )
    else:
        counts = [int(v) for v in arguments.frame_counts.split(",") if v.strip()]
        measurement = probe_video(adapter, size=arguments.size, frame_counts=counts, seed=arguments.seed)
        if arguments.support_frames > 0:
            measurement["temporal_support"] = probe_video_temporal_support(
                adapter, frames=arguments.support_frames, size=arguments.size, seed=arguments.seed
            )

    name = Path(arguments.vae).name.replace(".safetensors", "").replace(".ckpt", "")
    body = {
        "kind": "c15_vae_probe",
        "vae": arguments.vae,
        "vae_file_sha256": receipts.hash_file(arguments.vae),
        "vae_bytes": Path(arguments.vae).stat().st_size,
        "expected_profile": COMFY_VAE_PROFILES.get(Path(arguments.vae).name, {}),
        "geometry": geometry,
        "load_seconds": round(load_seconds, 3),
        "posterior": arguments.posterior,
        "posterior_mean_available": adapter.posterior_mean_available(),
        "measurement": measurement,
        "runs_h3_generation": False,
        "runtime": receipts.runtime_block(arguments.device),
        "argv": vars(arguments),
    }
    target = Path(arguments.out)
    target.mkdir(parents=True, exist_ok=True)
    receipts.write_receipt(target / f"{name}_probe.json", body)
    print(json.dumps(measurement, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
