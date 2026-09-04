"""C15 pair sampling against a REAL parent VAE, inside the ComfyUI runtime image.

This is the only file in the lane that imports ComfyUI. Run it as::

    docker run --rm --gpus '"device=<bus>"' \
      -v /home/cereal/comfyui/models:/opt/ComfyUI/models \
      -v /home/cereal/omni-campaign:/work \
      comfyui:codex-minimax-h3-runtime-env-r5-20260808 \
      /opt/environments/python/comfyui/bin/python /work/media/c15_vae_pairs_container.py \
        --vae /opt/ComfyUI/models/vae/ace_1.5_vae.safetensors \
        --out /work/media/pairs/ace_1.5_vae --clips 8 --seconds 2.0

CPU-only works too (drop ``--gpus``); these VAEs are small. The minimum loader is
``comfy.sd.VAE(sd=comfy.utils.load_torch_file(path))`` -- no custom node, no pipeline,
no sampler. Only the VAE runs: the H3 DiT is never constructed and never sampled, so
this stays clear of the pending H3 licence gate.

Output is a plain ``mediafold.c15_pairs`` directory, so the fitter
(``mediafold.c15_fit``) consumes it in the lab venv with no ComfyUI in sight.

Pairing note: ``--pairing per_channel`` is only well posed when the VAE's latent carries a
channel axis. ACE 1.5's AudioOobleckVAE emits ``[B, 64, T]`` jointly over stereo, so its
band must be sampled with ``--pairing stacked`` (``d_raw = 2*hop``); the sampler refuses
``per_channel`` there unless ``--allow-ambiguous-pairing`` is passed, in which case the
measured ambiguity floor is recorded in the receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import torch  # noqa: E402

from mediafold.c15_pairs import sample_audio_pairs, sample_video_pairs, write_pair_set  # noqa: E402
from mediafold.vae_adapters import (  # noqa: E402
    COMFY_VAE_PROFILES,
    ComfyVaeAdapter,
    procedural_audio,
    procedural_video,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C15 pair sampler against a real ComfyUI VAE")
    parser.add_argument("--vae", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--pairing", default="per_channel", choices=("per_channel", "stacked"))
    parser.add_argument(
        "--allow-ambiguous-pairing",
        action="store_true",
        help=(
            "keep per_channel rows on a VAE whose latent is JOINT over waveform channels "
            "(ACE 1.5). Without it the sampler refuses, because one latent broadcast to every "
            "channel row imposes a floor no map can beat. With it the measured floor is written "
            "into geometry['ambiguity'] and travels with every downstream score."
        ),
    )
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--latent-group", type=int, default=2)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cpu", help="cpu, or cuda with the docker -e overrides")
    parser.add_argument("--path-mode", default="auto", choices=("auto", "direct", "wrapper"),
                        help="direct = first_stage_model call, wrapper = comfy VAE.encode/decode")
    parser.add_argument("--posterior", default="mean", choices=("mean", "sample"),
                        help="'mean' pins a diagonal-gaussian bottleneck to its posterior mean so pairs are reproducible")
    parser.add_argument("--frame-head", type=int, default=0, help="video: leading raw frames to drop")
    parser.add_argument("--latent-head", type=int, default=0, help="video: leading latent frames to drop")
    parser.add_argument("--loose-alignment", action="store_true",
                        help="allow a non-exact frames/latents division and record it")
    parser.add_argument("--probe-only", action="store_true", help="load the VAE, report geometry, write nothing")
    arguments = parser.parse_args(argv)

    adapter = ComfyVaeAdapter(arguments.vae, device=arguments.device, posterior=arguments.posterior, path_mode=arguments.path_mode)
    geometry = adapter.geometry
    profile = COMFY_VAE_PROFILES.get(Path(arguments.vae).name, {})
    print(json.dumps({"loaded": geometry.to_json(), "expected_profile": profile}, indent=2), flush=True)
    if arguments.probe_only:
        return 0

    if geometry.modality == "audio":
        hop = int(geometry.hop_samples)
        if hop <= 0:
            raise SystemExit("loaded VAE did not report a hop; check the sd.py branch table")
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
            batch=arguments.batch,
            allow_ambiguous_pairing=arguments.allow_ambiguous_pairing,
        )
    else:
        clips = procedural_video(arguments.clips, frames=arguments.frames, size=arguments.size, seed=arguments.seed)
        pairs = sample_video_pairs(
            adapter,
            clips,
            latent_group=arguments.latent_group,
            holdout_fraction=arguments.holdout_fraction,
            seed=arguments.seed,
            batch=arguments.batch,
            frame_head=arguments.frame_head,
            latent_head=arguments.latent_head,
            strict_alignment=not arguments.loose_alignment,
        )

    held = torch.nonzero(pairs.split.eq(1)).flatten()
    evaluation = {}
    if held.numel():
        indices = sorted({int(pairs.clip_index[i].item()) for i in held})[:2]
        block = clips[torch.tensor(indices, dtype=torch.long)]
        latent = adapter.encode(block)
        evaluation = {
            "eval_raw": block.to(torch.float32),
            "eval_latent": latent.to(torch.float32).cpu(),
            "eval_roundtrip": adapter.decode(latent).to(torch.float32).cpu(),
            "eval_clip_index": torch.tensor(indices, dtype=torch.long),
        }

    body = write_pair_set(
        pairs,
        arguments.out,
        evaluation=evaluation,
        extra={"argv": vars(arguments), "vae_profile": profile, "runs_h3_generation": False},
    )
    print(json.dumps({"geometry": pairs.geometry, "split": body["split"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
