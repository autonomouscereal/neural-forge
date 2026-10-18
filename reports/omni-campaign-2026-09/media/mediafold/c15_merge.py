"""Merge C15 pair directories sampled in chunks.

Why chunks: the parent VAEs run inside the ComfyUI image on a box other lanes are also
using. A 96-clip ACE run and a 48-clip H3-audio run were both SIGKILLed mid-flight
(``EXIT_CODE=137``, no traceback, no output) while the host sat at 178/251 GiB used. The
fix is not a smaller corpus, it is smaller PROCESSES: sample 16 clips per container and
concatenate the pair sets here, where the only cost is a memcpy.

The merge is strict about geometry -- ``d_raw``, ``d_latent``, ``d_centre``, ``pairing``,
``context`` and the parent VAE name must all agree, because a fit over rows from two
different geometries is silently wrong rather than loudly broken. Clip indices are
re-numbered so the by-clip holdout split stays a by-clip split, and every source
directory's pair hash goes into the merged receipt.

The ``pairing`` string is NOT enough on its own for the measured-tubelet video sampler.
``measured_group4_latent1`` is the same string for two chunks whose MEASURED ownership
maps differ, and :func:`mediafold.c15_video_groups.measure_frame_groups` re-measures per
chunk from one probe clip at a per-chunk seed -- the map is resolution- and content-
dependent (a 128 px probe measured group sizes {1,3,4,5}, a 64 px probe {1,4}). The
alignment keys are therefore part of ``STRICT_KEYS``, and every source chunk's map is
recorded in the merged receipt instead of only the first one's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from . import receipts
from .c15_pairs import PairSet, load_pair_set, write_pair_set

STRICT_KEYS = (
    "name",
    "modality",
    "pairing",
    "context",
    "d_raw",
    "d_latent",
    "d_centre",
    "centre_offset",
    # measured video alignment: two chunks can share a `pairing` string and still own
    # different raw frames per latent frame. Absent on audio sets (None == None).
    "temporal_group_size",
    "selected_latent_frames",
    "skipped_latent_frames",
    "measured_group_sizes",
    # a stacked/per_channel audio set fitted through an ambiguous pairing must never be
    # concatenated with an unambiguous one
    "ambiguous_pairing",
)

#: geometry keys copied per source chunk into the merged receipt so the alignment each
#: chunk was actually sampled under stays auditable after the concatenation.
ALIGNMENT_KEYS = (
    "pairing",
    "temporal_group_size",
    "latent_group",
    "selected_latent_frames",
    "skipped_latent_frames",
    "measured_group_sizes",
    "measured_uniform",
    "alignment",
    "ambiguous_pairing",
    "ambiguity",
)


def alignment_block(geometry: dict[str, Any]) -> dict[str, Any]:
    """The subset of a pair set's geometry that says how its rows were aligned."""

    return {key: geometry[key] for key in ALIGNMENT_KEYS if key in geometry}


def merge_pair_sets(sets: Sequence[PairSet]) -> PairSet:
    """Concatenate pair sets that share a geometry, re-numbering clips."""

    if not sets:
        raise ValueError("nothing to merge")
    reference = sets[0].geometry
    for other in sets[1:]:
        for key in STRICT_KEYS:
            if reference.get(key) != other.geometry.get(key):
                raise ValueError(
                    f"pair sets disagree on {key!r}: {reference.get(key)!r} vs {other.geometry.get(key)!r}"
                )

    raw: list[torch.Tensor] = []
    latent: list[torch.Tensor] = []
    roundtrip: list[torch.Tensor] = []
    clip_index: list[torch.Tensor] = []
    frame_index: list[torch.Tensor] = []
    channel_index: list[torch.Tensor] = []
    split: list[torch.Tensor] = []
    offset = 0
    clips_total = 0
    for pairs in sets:
        raw.append(pairs.raw)
        latent.append(pairs.latent)
        roundtrip.append(pairs.roundtrip)
        clip_index.append(pairs.clip_index + offset)
        frame_index.append(pairs.frame_index)
        channel_index.append(pairs.channel_index)
        split.append(pairs.split)
        local_clips = int(pairs.clip_index.max().item()) + 1 if pairs.clip_index.numel() else 0
        offset += local_clips
        clips_total += int(pairs.geometry.get("clips", local_clips))

    merged_raw = torch.cat(raw, dim=0)
    geometry = {
        **reference,
        "rows": int(merged_raw.shape[0]),
        "clips": int(offset),
        "merged_from": len(sets),
        "merged_clip_totals": [int(item.geometry.get("clips", 0)) for item in sets],
        "declared_clips_sum": clips_total,
        # every chunk's own alignment, not just set[0]'s
        "merged_alignments": [alignment_block(item.geometry) for item in sets],
    }
    return PairSet(
        raw=merged_raw,
        latent=torch.cat(latent, dim=0),
        roundtrip=torch.cat(roundtrip, dim=0),
        clip_index=torch.cat(clip_index, dim=0),
        frame_index=torch.cat(frame_index, dim=0),
        channel_index=torch.cat(channel_index, dim=0),
        split=torch.cat(split, dim=0),
        geometry=geometry,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="merge C15 pair directories")
    parser.add_argument("--pairs", action="append", required=True, help="a pair directory; repeatable")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)

    sources = [str(Path(entry).resolve()) for entry in arguments.pairs]
    sets = [load_pair_set(entry) for entry in sources]
    merged = merge_pair_sets(sets)
    body = write_pair_set(
        merged,
        arguments.out,
        extra={
            "merged_from": [
                {
                    "dir": source,
                    "rows": int(item.raw.shape[0]),
                    "clips": int(item.geometry.get("clips", 0)),
                    "receipt_sha256": receipts.hash_file(Path(source) / "pairs_receipt.json"),
                    "pairs_sha256": receipts.hash_file(Path(source) / "pairs.safetensors"),
                    "alignment": alignment_block(item.geometry),
                }
                for source, item in zip(sources, sets)
            ],
            "argv": vars(arguments),
        },
    )
    print(json.dumps({"rows": merged.geometry["rows"], "clips": merged.geometry["clips"], "split": body["split"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
