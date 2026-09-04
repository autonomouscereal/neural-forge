#!/usr/bin/env python
"""Split a C16 coreset holdout into the rows that leaked and the rows that did not.

Fixing the grouping moved the block-12 holdout the "wrong" way (rel_l2 0.617 -> 0.554),
which is only paradoxical if you assume the leak is the only thing that changed. It is
not: the batch-row split holds out 7 contiguous 256-row windows, the clip split holds out
19 clips scattered across the corpus, so the two holdouts are different row sets and the
scores move for two reasons at once.

This scores an ALREADY-FOLDED child on its own holdout rows, split into

    leaked  -- rows whose clip also appears in the fit (the refit saw near-duplicates)
    clean   -- rows whose clip is entirely held out

If ``leaked`` scores better than ``clean``, the leak was flattering that holdout number,
whatever the aggregate did. Nothing is refitted here; the child's weights come off disk.

    python c16_holdout_decompose.py --run out/c16_L12_v2 --grouping dit_batch_row \\
        --out out/c16_L12_v2/holdout_decomposition.json
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

from c16_holdout_leak_probe import grouped_holdout  # noqa: E402
from mediafold import receipts  # noqa: E402
from mediafold.ace_dit import MLP, AceStepDiTConfig, load_ace_dit_block  # noqa: E402
from mediafold.c15_pairs import load_pair_set  # noqa: E402
from mediafold.c16_capture import token_clip_index, token_clip_membership  # noqa: E402
from mediafold.linalg import write_scores  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C16 holdout leaked/clean decomposition")
    parser.add_argument("--run", required=True, help="a c16 transplant output directory")
    parser.add_argument(
        "--grouping",
        default="dit_batch_row",
        choices=("dit_batch_row", "clip"),
        help="the grouping whose holdout is being decomposed",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    arguments = parser.parse_args(argv)

    run = Path(arguments.run)
    receipt = json.loads((run / "c16_receipt.json").read_text(encoding="utf-8"))
    argv_block = receipt["argv"]
    notes = receipt["capture"]["stimulus"]
    if notes.get("rung") != "latents":
        parser.error("only a --rung latents run has clip provenance")

    batch = int(argv_block["batch"])
    length = int(argv_block["length"])
    block_index = int(argv_block["block"])
    keep = int(argv_block["keep"])
    seed = int(argv_block["seed"])
    config = AceStepDiTConfig.from_json(argv_block["config"])
    patch_size = int(config.patch_size)
    tokens = -(-length // patch_size)

    # rows the fold was measured on, exactly as the run captured them
    capture = receipts.read_tensors(run / f"capture_L{block_index:02d}.safetensors")
    mlp_capture = capture["mlp_in"]
    rows = mlp_capture.reshape(-1, mlp_capture.shape[-1]).to(torch.float32)

    clip_rows = (
        load_pair_set(notes["pairs_dir"]).clip_index.reshape(-1)[
            int(notes["row_offset"]) : int(notes["row_offset"]) + batch * length
        ]
        .reshape(batch, length)
        .to(torch.long)
    )
    clip_groups, straddling = token_clip_index(clip_rows, tokens=tokens, patch_size=patch_size)
    # the same padded arithmetic the runner and the leak probe use, so a --length that is
    # not a multiple of patch_size gives the clip map's own message, not a reshape error
    touched = token_clip_membership(clip_rows, tokens=tokens, patch_size=patch_size)

    if arguments.grouping == "dit_batch_row":
        groups = torch.arange(batch).repeat_interleave(tokens)
        used = torch.ones(int(clip_groups.numel()), dtype=torch.bool)
    else:
        used = ~straddling
        groups = clip_groups[used]
    hold_local = grouped_holdout(groups, fraction=arguments.holdout_fraction, seed=seed)
    hold = torch.zeros(int(clip_groups.numel()), dtype=torch.bool)
    hold[used] = hold_local

    # parent target and child output on those rows -- weights read, nothing refitted
    parent, _ = load_ace_dit_block(
        block_index, weights=argv_block["weights"], config=config, device="cpu", dtype=torch.float32
    )
    child_mlp = MLP(int(config.hidden_size), keep)
    state = receipts.read_tensors(run / f"folded_block_L{block_index:02d}_keep{keep}.safetensors")
    child_mlp.load_state_dict(
        {
            "gate_proj.weight": state["mlp.gate_proj.weight"].to(torch.float32),
            "up_proj.weight": state["mlp.up_proj.weight"].to(torch.float32),
            "down_proj.weight": state["mlp.down_proj.weight"].to(torch.float32),
        }
    )
    with torch.no_grad():
        target = parent.mlp(rows).to(torch.float64)
        output = child_mlp(rows).to(torch.float64)

    fit_clips = torch.unique(touched[used & ~hold])
    fit_clips = fit_clips[fit_clips >= 0]
    leaked = hold & torch.isin(touched, fit_clips).any(dim=-1)
    clean = hold & ~leaked
    # A row's ACTIVATION reaches every clip in its batch row, whatever clip its own latents
    # came from, because self-attention's receptive field is the batch row. "clean" above
    # is row-level clean only; this counts how many of those rows still sit in a batch row
    # that carries a fit clip.
    row_batch = torch.arange(batch).repeat_interleave(tokens)
    batch_touches_fit = torch.tensor(
        [bool(torch.isin(torch.unique(row), fit_clips).any().item()) for row in clip_rows]
    )
    reachable_leak = hold & batch_touches_fit[row_batch]

    body = {
        "kind": "c16_holdout_decomposition",
        "run": {
            "path": str(run.resolve()),
            "receipt_sha256": receipts.hash_file(run / "c16_receipt.json"),
            "child_sha256": receipts.hash_file(run / f"folded_block_L{block_index:02d}_keep{keep}.safetensors"),
        },
        "grouping": arguments.grouping,
        "reported_holdout_scores": receipt["coreset"]["holdout_scores"],
        "whole_holdout": write_scores(output[hold], target[hold]),
        "holdout_leaked_rows": write_scores(output[leaked], target[leaked]) if bool(leaked.any()) else {},
        "holdout_clean_rows": write_scores(output[clean], target[clean]) if bool(clean.any()) else {},
        "fit_rows_for_reference": write_scores(output[used & ~hold], target[used & ~hold]),
        "counts": {
            "rows_used": int(used.sum().item()),
            "holdout_rows": int(hold.sum().item()),
            "leaked_rows": int(leaked.sum().item()),
            "clean_rows": int(clean.sum().item()),
            "row_level_clean_rows_whose_receptive_field_still_touches_a_fit_clip": int(
                (clean & reachable_leak).sum().item()
            ),
            "clips_per_batch_row_max": max(int(torch.unique(row).numel()) for row in clip_rows),
        },
        "reading": (
            "leaked rows share a clip with the fit, so the refit has seen near-duplicates of "
            "them; a leaked score better than the clean score is the leak flattering the "
            "reported holdout, independent of what the aggregate number did. 'clean' here is "
            "ROW LEVEL: these rows' own latents come from held-out clips only. It does not mean "
            "their activations are independent of the fit -- the rows are mlp_in after several "
            "DiT layers whose attention receptive field is the whole batch row, so unless the "
            "window is clip-aligned (one clip per batch row) a clean row's activation is still "
            "a function of the fit clips beside it. The count "
            "row_level_clean_rows_whose_receptive_field_still_touches_a_fit_clip says how many."
        ),
        "runtime": receipts.runtime_block("cpu"),
        "argv": vars(arguments),
    }
    receipts.write_receipt(Path(arguments.out), body)
    print(json.dumps({k: body[k] for k in ("grouping", "counts", "whole_holdout",
                                           "holdout_leaked_rows", "holdout_clean_rows")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
