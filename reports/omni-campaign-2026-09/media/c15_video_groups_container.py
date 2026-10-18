"""C15 video pair sampling against a real VAE, on its MEASURED tubelet.

Runs inside the ComfyUI runtime image (the VAE only -- no DiT is constructed and no H3
generation happens; the H3 licence gate is still pending)::

    docker run --rm --gpus device=0 \\
      -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES=0 \\
      --entrypoint /opt/environments/python/comfyui/bin/python \\
      -v /home/cereal/comfyui/models:/opt/ComfyUI/models \\
      -v /home/cereal/omni-campaign:/work \\
      comfyui:codex-minimax-h3-runtime-env-r5-20260808 \\
      /work/media/c15_video_groups_container.py \\
        --vae /opt/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors \\
        --out /work/media/pairs/h3_video_g4 --clips 24 --frames 22 --size 128 --device cuda

Unlike the uniform-tubelet sampler this script first MEASURES which latent frame each raw
frame writes into (per-frame perturbation probe) and then pairs only the latent frames
that own exactly ``--group-size`` contiguous raw frames, so no row is fitted across an
alignment the VAE does not actually have.
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

from mediafold.c15_pairs import write_pair_set  # noqa: E402
from mediafold.c15_video_groups import measure_frame_groups, sample_grouped_video_pairs  # noqa: E402
from mediafold.vae_adapters import COMFY_VAE_PROFILES, ComfyVaeAdapter, procedural_video  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C15 measured-tubelet video pair sampler")
    parser.add_argument("--vae", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--clips", type=int, default=24)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--latent-group", type=int, default=1)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--amplitude", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--group-map-seed",
        type=int,
        default=None,
        help=(
            "seed for the ownership PROBE clip only. The measured map is resolution- and "
            "content-dependent, so chunks that will be merged must probe the SAME clip: pin "
            "this across every chunk while --seed varies the corpus. Defaults to --seed, which "
            "re-measures per chunk (mediafold.c15_merge now refuses to merge chunks whose "
            "measured maps differ)."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--path-mode", default="auto", choices=("auto", "direct", "wrapper"))
    parser.add_argument("--posterior", default="mean", choices=("mean", "sample"))
    arguments = parser.parse_args(argv)

    adapter = ComfyVaeAdapter(
        arguments.vae,
        device=arguments.device,
        posterior=arguments.posterior,
        path_mode=arguments.path_mode,
    )
    geometry = adapter.geometry
    if geometry.modality != "video":
        raise SystemExit(f"{arguments.vae} loaded as a {geometry.modality} VAE; this script is video only")
    print(json.dumps({"loaded": geometry.to_json()}, indent=2), flush=True)

    group_map_seed = arguments.seed if arguments.group_map_seed is None else int(arguments.group_map_seed)
    group_map = measure_frame_groups(
        adapter,
        frames=arguments.frames,
        size=arguments.size,
        amplitude=arguments.amplitude,
        seed=group_map_seed,
    )
    group_map["probe_seed"] = int(group_map_seed)
    print(
        json.dumps(
            {
                "measured_groups": group_map["groups"],
                "group_sizes": group_map["group_sizes"],
                "uniform": group_map["uniform"],
            },
            indent=2,
        ),
        flush=True,
    )

    clips = procedural_video(arguments.clips, frames=arguments.frames, size=arguments.size, seed=arguments.seed + 7)
    pairs = sample_grouped_video_pairs(
        adapter,
        clips,
        group_map=group_map,
        group_size=arguments.group_size,
        latent_group=arguments.latent_group,
        holdout_fraction=arguments.holdout_fraction,
        seed=arguments.seed,
        batch=arguments.batch,
    )

    held = torch.nonzero(pairs.split.eq(1)).flatten()
    evaluation = {}
    if held.numel():
        indices = sorted({int(pairs.clip_index[i].item()) for i in held})[:2]
        block = clips[torch.tensor(indices, dtype=torch.long)]
        latent = adapter.encode(block)
        evaluation = {
            "eval_raw": block.to(torch.float32),
            "eval_latent": latent.float().cpu(),
            "eval_roundtrip": adapter.decode(latent).float().cpu(),
            "eval_clip_index": torch.tensor(indices, dtype=torch.long),
        }

    body = write_pair_set(
        pairs,
        arguments.out,
        evaluation=evaluation,
        extra={
            "argv": vars(arguments),
            "vae_profile": COMFY_VAE_PROFILES.get(Path(arguments.vae).name, {}),
            "measured_group_map": group_map,
            "runs_h3_generation": False,
        },
    )
    print(json.dumps({"geometry": pairs.geometry, "split": body["split"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
