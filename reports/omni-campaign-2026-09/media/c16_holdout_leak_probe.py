#!/usr/bin/env python
"""Measure the C16 coreset holdout leak, on the exact rows a transplant run captured.

The round-2 blocker was that ``coreset_block_mlp`` was handed one group id per DiT BATCH
ROW. A batch row is a contiguous ``--length`` window of the pair corpus; an ACE clip is
100 latent rows. So one batch row spans ~2.5 clips and a holdout of whole batch rows still
shares clips with the fit -- those held-out rows are near-duplicates of rows the refit
saw, and the "holdout" score is part fit score.

This probe replays the SAME split arithmetic ``coreset_block_mlp`` performs (same seed,
same ``torch.randperm`` over the unique group ids, same 20% fraction) on the clip map the
stimulus now carries, and reports TWO different things for the old grouping and the new:

``row_level``        held-out rows whose own latent rows come from a clip that is also in
                     the fit. This is what grouping by clip fixes, and it goes to 0.
``receptive_field``  held-out rows whose ACTIVATION is a function of a fit clip. The rows
                     being split are ``mlp_in`` at block 12, i.e. the output of twelve DiT
                     layers of self-attention whose receptive field is the whole batch row
                     (``layer_types`` alternates ``full_attention``, and a
                     ``sliding_attention`` layer whose window covers the row is full too).
                     So a token in a batch row carrying clips {3,4,5} has an activation
                     that depends on all three, whatever clip it is filed under. Grouping
                     by clip does NOT fix this; only a clip-ALIGNED window does, one where
                     each batch row is a whole clip (``--row-offset`` on a clip boundary,
                     ``--length`` = the clip length).

Zero row-level leak is a real invariant and the probe exits non-zero if the clip grouping
breaks it. Zero receptive-field leak is the stronger property that makes a holdout score
an unseen-content error, and the probe reports it as ``activation_independence`` rather
than asserting it, because it is a property of the WINDOW the run chose.

No model is loaded and no weights are read: this is arithmetic on the clip index, so it
can be re-run in seconds against any transplant receipt.

    python c16_holdout_leak_probe.py --receipt out/c16_L12_v4/c16_receipt.json \
        --out out/c16_L12_v4/holdout_leak_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mediafold import receipts  # noqa: E402
from mediafold.c15_pairs import load_pair_set  # noqa: E402
from mediafold.c16_capture import token_clip_index, token_clip_membership  # noqa: E402


def grouped_holdout(groups: torch.Tensor, *, fraction: float, seed: int) -> torch.Tensor:
    """The holdout mask ``coreset_block_mlp`` builds for a given per-row group id."""

    generator = torch.Generator().manual_seed(int(seed))
    unique = torch.unique(groups)
    shuffled = unique[torch.randperm(int(unique.numel()), generator=generator)]
    held = max(1, int(round(float(fraction) * int(unique.numel())))) if fraction > 0 else 0
    return torch.isin(groups, shuffled[:held])


def leak(hold_mask: torch.Tensor, touched: torch.Tensor, reachable: torch.Tensor) -> dict[str, Any]:
    """How much of this split is not a holdout at all, at two different radii.

    ``touched``    [rows, patch]: every clip each row's own latents come from.
    ``reachable``  [rows, max_clips_per_batch_row]: every clip the row's activation can be
                   a function of, i.e. every clip in its batch row. ``-1`` pads both.

    A held-out row LEAKS AT ROW LEVEL when one of its own clips is also in the fit -- the
    refit has seen near-duplicates of it. It leaks THROUGH THE RECEPTIVE FIELD when a clip
    its activation depends on is in the fit, which is the weaker requirement and the one
    that decides whether the holdout score is an unseen-content error.
    """

    def side(mask: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        values = torch.unique(ids[mask])
        return values[values >= 0]

    fit_mask = ~hold_mask
    holdout_rows = int(hold_mask.sum().item())
    fit_rows = int(fit_mask.sum().item())
    held_clips, fit_clips = side(hold_mask, touched), side(fit_mask, touched)
    held_leaking = int(torch.isin(touched[hold_mask], fit_clips).any(dim=-1).sum().item())
    fit_leaking = int(torch.isin(touched[fit_mask], held_clips).any(dim=-1).sum().item())
    held_reach = int(torch.isin(reachable[hold_mask], fit_clips).any(dim=-1).sum().item())
    fit_reach = int(torch.isin(reachable[fit_mask], held_clips).any(dim=-1).sum().item())
    return {
        "fit_rows": fit_rows,
        "holdout_rows": holdout_rows,
        "clips_in_fit": int(fit_clips.numel()),
        "clips_in_holdout": int(held_clips.numel()),
        "clips_on_both_sides": int(torch.isin(held_clips, fit_clips).sum().item()),
        "holdout_rows_sharing_a_clip_with_fit": held_leaking,
        "holdout_leak_fraction": held_leaking / max(holdout_rows, 1),
        "fit_rows_sharing_a_clip_with_holdout": fit_leaking,
        "fit_leak_fraction": fit_leaking / max(fit_rows, 1),
        "holdout_rows_whose_receptive_field_touches_a_fit_clip": held_reach,
        "receptive_field_leak_fraction": held_reach / max(holdout_rows, 1),
        "fit_rows_whose_receptive_field_touches_a_holdout_clip": fit_reach,
        "fit_receptive_field_leak_fraction": fit_reach / max(fit_rows, 1),
        "activation_independence": held_reach == 0 and fit_reach == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C16 coreset holdout leak probe")
    parser.add_argument("--receipt", required=True, help="a c16_receipt.json to replay")
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    arguments = parser.parse_args(argv)

    receipt = json.loads(Path(arguments.receipt).read_text(encoding="utf-8"))
    argv_block = receipt["argv"]
    stimulus_notes = receipt["capture"]["stimulus"]
    if stimulus_notes.get("rung") != "latents":
        parser.error("only a --rung latents run has clip provenance to probe")

    batch = int(argv_block["batch"])
    length = int(argv_block["length"])
    start = int(stimulus_notes["row_offset"])
    seed = int(argv_block["seed"])
    patch_size = int(json.loads(Path(argv_block["config"]).read_text(encoding="utf-8"))["patch_size"])
    tokens = -(-length // patch_size)

    pairs = load_pair_set(stimulus_notes["pairs_dir"])
    clip_rows = pairs.clip_index.reshape(-1)[start : start + batch * length]
    if int(clip_rows.numel()) != batch * length:
        raise SystemExit("the pair corpus no longer covers this run's window")
    clip_rows = clip_rows.reshape(batch, length).to(torch.long)
    # The clips each token's own latents come from (padded exactly as `proj_in` pads), and
    # the clips its activation can reach through attention -- every clip in its batch row.
    touched = token_clip_membership(clip_rows, tokens=tokens, patch_size=patch_size)
    width = max(int(torch.unique(row).numel()) for row in clip_rows)
    reachable_per_row = torch.full((batch, width), -1, dtype=torch.long)
    for index, row in enumerate(clip_rows):
        present = torch.unique(row)
        reachable_per_row[index, : int(present.numel())] = present
    reachable = reachable_per_row.repeat_interleave(tokens, dim=0)

    clip_groups, straddling = token_clip_index(clip_rows, tokens=tokens, patch_size=patch_size)
    batch_row_groups = torch.arange(batch).repeat_interleave(tokens)

    old = leak(
        grouped_holdout(batch_row_groups, fraction=arguments.holdout_fraction, seed=seed),
        touched,
        reachable,
    )
    kept = ~straddling
    new_mask = torch.zeros(int(clip_groups.numel()), dtype=torch.bool)
    new_mask[kept] = grouped_holdout(
        clip_groups[kept], fraction=arguments.holdout_fraction, seed=seed
    )
    new = leak(new_mask[kept], touched[kept], reachable[kept])

    clip_aligned = width == 1
    body = {
        "kind": "c16_holdout_leak_probe",
        "receipt": {
            "path": str(Path(arguments.receipt).resolve()),
            "sha256": receipts.hash_file(Path(arguments.receipt)),
        },
        "window": {
            "batch": batch,
            "length": length,
            "row_offset": start,
            "row_offset_source": stimulus_notes.get("row_offset_source", "seeded_random"),
            "patch_size": patch_size,
            "tokens_per_batch_row": tokens,
            "rows": int(clip_groups.numel()),
            "clips_covered": int(torch.unique(clip_rows).numel()),
            "clips_per_batch_row": [int(torch.unique(row).numel()) for row in clip_rows],
            "clips_per_batch_row_max": int(width),
            "clip_aligned": bool(clip_aligned),
            "boundary_tokens": int(straddling.sum().item()),
        },
        "grouped_by_dit_batch_row": old,
        "grouped_by_clip": new,
        "reading": (
            "TWO different numbers. holdout_leak_fraction is ROW LEVEL: the share of held-out "
            "rows whose own latents come from a clip that is also in the fit. Those rows are "
            "near-duplicates of rows the refit saw, so a batch-row holdout score is part fit "
            "score; grouping by clip drives it to 0 and this probe asserts that. "
            "receptive_field_leak_fraction is the stronger question: the rows being split are "
            "mlp_in at this block, produced by DiT self-attention whose receptive field is the "
            "whole batch row, so a held-out row's activation is a function of EVERY clip in its "
            "batch row whatever clip it is filed under. Grouping by clip does not change that "
            "number at all -- only a clip-ALIGNED window does, one where each batch row is one "
            "whole clip. Read activation_independence: when it is false the clip-grouped "
            "holdout is a clean row-level split and still not an unseen-content error. "
            "Finally, two holdout SCORES from different groupings are not paired -- the splits "
            "hold out different rows, so the scores move for two reasons at once. Use "
            "holdout_decomposition.json, which scores ONE child on both splits."
        ),
        "runtime": receipts.runtime_block(),
        "argv": vars(arguments),
    }
    if new["holdout_leak_fraction"] != 0.0:
        raise SystemExit(f"clip grouping still leaks {new['holdout_leak_fraction']:.4f} of the holdout")

    target = Path(arguments.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    receipts.write_receipt(target, body)
    print(json.dumps({
        "clip_aligned": bool(clip_aligned),
        "grouped_by_dit_batch_row": old,
        "grouped_by_clip": new,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
