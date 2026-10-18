#!/usr/bin/env python
"""S6H: >= 4x channel coreset of one captured MiniMax H3 DiT block MLP, clip-grouped holdout.

Runs anywhere with torch (the server lab venv; no ComfyUI): it reads the block's verbatim
tensors and the ``step{s}/L{i}.safetensors`` captures written by ``c16_h3_capture.py``,
rebuilds the block as an ``H3BlockShell`` and calls the media lane's
``mediafold.c16_transplant.coreset_block_mlp`` unchanged -- pivoted VERBATIM gate/up rows
(LFM2 law 40/42: never SVD on a gated FFN), ``fc2`` refit by the prior-anchored ridge with
the update_scale veto, holdout split by CLIP with the row-membership guard::

    $PY media/c16_h3_coreset.py --captures /mnt/lexar/omni-campaign/media/h3_blocks --block 25 \\
        --keep 3584 --out /mnt/lexar/omni-campaign/media/h3_block_coreset \\
        --receipt-copy $OC/composite/media/h3_block_coreset

14336 -> 3584 is exactly 4.00x on the MLP component; the whole-block ratio (attention,
norms and adaLN frozen) is recorded beside it and the kv-head-merge arm is named as NOT
built. Four MLP-level arms are scored on the same held-out clips (replace / control /
random_select / prior_only), pooled AND per token kind (audio / video), because the
pooled av number is dominated by whichever kind has more rows; the chart-level T1 arms
are the S7 install's job.

Output layout: ``L{i}/mlp.safetensors`` holds ONLY the child's two MLP weights (fused
``fc1`` / ``fc2``, bf16) so its header count is the child MLP parameter count; the
pivoted channel list lives beside it in ``L{i}/selection.safetensors``. This is NOT yet
the S7 charted container (``composite/COMPOSITE_S6_INPUT_FORMATS.md`` §3): the receipt's
``s7_consumability`` block names exactly what a converter still has to decide.
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
from mediafold.c16_h3 import (  # noqa: E402
    KIND_AUDIO,
    KIND_VIDEO,
    H3BlockShell,
    block_weights_path,
    capture_path,
    concat_captures,
    coreset_row_groups,
    mlp_arm_table,
    shell_from_block_state,
    strip_block_prefix,
    whole_block_receipt,
)
from mediafold.c16_transplant import CoresetResult, coreset_block_mlp  # noqa: E402
from mediafold.linalg import write_scores  # noqa: E402

ACE_BLOCK12_REFERENCE = {"rel_l2": 0.5442424558949278, "cos": 0.8415188270864858, "source": "media/receipts/c16_L12_v4/c16_receipt.json holdout_scores (clip-aligned window)"}


def write_child(folded: CoresetResult, block: int, target: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """``mlp.safetensors`` (the two MLP weights only, bf16) + ``selection.safetensors`` beside it.

    One number per file: ``mlp.safetensors``'s header count IS the child MLP parameter
    count; the pivoted channel index is provenance, not parameters, and lives in its own
    file.
    """

    child_state = folded.block.fused_state(prefix=f"blocks.{int(block)}.")
    child_state = {name: value.to(torch.bfloat16) for name, value in child_state.items()}
    written = receipts.write_tensors(child_state, target / "mlp.safetensors")
    if int(written["parameters"]) != int(folded.block.mlp_parameters()):
        raise SystemExit("written MLP header count differs from the child's parameter count")
    selection_written = receipts.write_tensors({"selection": folded.selection.to(torch.long)}, target / "selection.safetensors")
    selection_written["note"] = "pivoted parent channel indices (provenance, not parameters); not counted in mlp.safetensors"
    return written, selection_written


def s7_consumability(block_state: dict[str, torch.Tensor], *, keep: int) -> dict[str, Any]:
    """What separates this output from the S7 charted container, named field by field.

    ``c16_install_media.py --blocks DIR`` reads ``charted_block.json`` +
    ``charted_block.safetensors`` in ``ChartedBlockSpec`` names; the only converter is
    ``--convert-ace-run``. The H3 block has three things that container cannot express
    without a spec extension, listed here with the decision each one needs.
    """

    fc1 = block_state["mlp.fc1.weight"]
    qkv = block_state["attn.qkv_proj.weight"]
    adaln = block_state["adaln_proj.linear.weight"]
    hidden = int(fc1.shape[1])
    return {
        "consumable_by_c16_install_media_blocks": False,
        "format_of_record": "composite/COMPOSITE_S6_INPUT_FORMATS.md section 3 (chimera_omni.composite.charted_block.v1)",
        "this_output": "L{i}/mlp.safetensors (blocks.{i}.mlp.fc1.weight fused [2*keep, hidden], blocks.{i}.mlp.fc2.weight [hidden, keep]) + selection.safetensors",
        "converter_available": "none for H3 (only --convert-ace-run); an H3 -> charted-container converter is NOT built",
        "mechanical_renames": {
            "mlp.fc1.weight -> mlp.gate_proj.weight / mlp.up_proj.weight": f"split_fc1 (rows [:{int(keep)}] / [{int(keep)}:]); mlp_kind swiglu",
            "mlp.fc2.weight -> mlp.down_proj.weight": "verbatim",
            "attn.qkv_proj.weight -> attn.q_proj / k_proj / v_proj": f"chunk(3, dim=0) of {list(qkv.shape)}; heads 56, kv_heads 56 (MHA), head_dim 128",
            "attn.out_proj.weight -> attn.o_proj.weight": "verbatim",
            "attn.q_norm / k_norm -> attn.q_norm / k_norm": "verbatim (qk_norm true)",
            "norm1 / norm2 -> attn_norm / mlp_norm": "verbatim; norm_kind must be checked against comfy/ldm/minimax/model.py before it is declared rms",
        },
        "spec_extensions_required": {
            "modulation": (
                f"H3 adaLN is per-MODALITY: adaln_proj.linear {list(adaln.shape)} maps an 8-dim curve embedding to "
                f"3 modalities x 6 rows x {hidden}; ChartedBlockSpec adaln6 has ONE scale_shift_table [6, P] driven by "
                "time_embed. Decision: which of the three modality row sets (video 0 / text 1 / audio 2) the trunk drives, "
                "or a per-type_id table (spec extension)"
            ),
            "time_embed": (
                "H3 has no sinusoid -> linear_1 -> linear_2 -> time_proj path; its t_emb is an 8-dim curve embedding "
                "of (sigma_video, sigma_audio). Decision: a time_embed whose output reproduces the 8-dim curve "
                "(time_freq_dim / time_hidden / time_factor have no H3 values), or a new modulation kind"
            ),
            "attention_width": f"MHA inner width 7168 (56 x 128) on hidden {hidden}: the charted attention must allow inner != width",
            "rope": "H3 packs [text | audio | video] with per-segment rope tables (rope_freqs per forward); the 1-D rope_theta field does not describe it -- record in metadata",
        },
        "price": "converter + charted_block.json writer + test: ~0.5 day once the modulation/time_embed decision is taken; the decision itself belongs to the S7 owner (Cyril), not this tool",
        "stage_receipt_note": "S7's h3_Lmid input requires the spec extension above; until it lands the H3 block is measured (this receipt) but not installable",
    }


def replicate_group_split(groups: torch.Tensor, holdout_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The exact split ``coreset_block_mlp`` draws (same generator, same order) so the arms score the same holdout."""

    generator = torch.Generator().manual_seed(int(seed))
    sequences = groups.reshape(-1).to(torch.long)
    unique = torch.unique(sequences)
    shuffled = unique[torch.randperm(int(unique.numel()), generator=generator)]
    held = max(1, int(round(holdout_fraction * int(unique.numel())))) if holdout_fraction > 0 else 0
    hold_mask = torch.isin(sequences, shuffled[:held])
    return torch.nonzero(~hold_mask).flatten(), torch.nonzero(hold_mask).flatten()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S6H H3 DiT block MLP 4x coreset")
    parser.add_argument("--captures", required=True, help="c16_h3_capture.py --out directory")
    parser.add_argument("--block", type=int, default=25)
    parser.add_argument("--steps", default="all", help="'all' or comma separated step indices")
    parser.add_argument("--keep", type=int, default=3584, help="14336 / 4 = 3584 is the 4x rung")
    parser.add_argument("--rows", default="av", choices=("av", "video", "audio", "all"))
    parser.add_argument("--group-by", default="auto", choices=("auto", "video", "audio"), help="holdout group axis: auto picks the clip id shared across forwards (audio windows -> audio)")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", default="replace,control,random_select,prior_only")
    parser.add_argument("--dtype", default="float32", choices=("float32",), help="CPU fp32 only (bf16 on these CPUs is 400x slower)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--receipt-copy", default=None)
    arguments = parser.parse_args(argv)

    started = time.time()
    root = Path(arguments.captures)
    capture_receipt = json.loads((root / "capture_receipt.json").read_text(encoding="utf-8")) if (root / "capture_receipt.json").is_file() else {}
    weights_path = block_weights_path(root, arguments.block)
    block_state = strip_block_prefix(receipts.read_tensors(weights_path), arguments.block)
    shell = shell_from_block_state(block_state, dtype=torch.float32)

    if arguments.steps == "all":
        step_paths = sorted(root.glob(f"step*/L{int(arguments.block):02d}.safetensors"))
    else:
        step_paths = [capture_path(root, int(s), arguments.block) for s in arguments.steps.split(",") if s.strip()]
    if not step_paths:
        raise SystemExit(f"no captures for block {arguments.block} under {root}")
    states = [receipts.read_tensors(p) for p in step_paths]
    merged = concat_captures(states)
    sigmas = sorted({float(v) for v in merged["sigma_video_rows"].tolist()})

    keep_mask, groups, group_name, membership, group_notes = coreset_row_groups(
        merged["token_kind"], merged["token_clip"], merged["forward_index"], rows=arguments.rows, group_by=arguments.group_by
    )
    rows = merged["mlp_in"][keep_mask].to(torch.float32)
    kinds = merged["token_kind"][keep_mask].reshape(-1).to(torch.long)
    stored_out = merged["mlp_out"][keep_mask].to(torch.float32)
    with torch.no_grad():
        recomputed = shell.mlp(rows)
    stored_parity = {
        **write_scores(stored_out.to(torch.float64), recomputed.to(torch.float64)),
        "reading": "stored (bf16) mlp_out vs the fp32 recompute from the verbatim block weights: the storage rounding, not a fit",
    }

    fit_index, hold_index = replicate_group_split(groups, arguments.holdout_fraction, arguments.seed)
    folded = coreset_block_mlp(
        shell,
        rows,
        keep=int(arguments.keep),
        holdout_fraction=float(arguments.holdout_fraction),
        seed=int(arguments.seed),
        sequence_index=groups,
        group_name=group_name,
        row_membership=membership,
    )
    if int(folded.receipt["holdout_rows"]) != int(hold_index.numel()) or int(folded.receipt["fit_rows"]) != int(fit_index.numel()):
        raise SystemExit("replicated split does not match coreset_block_mlp's split; refusing to score arms on a different holdout")
    folded.receipt["component"]["component"] = "h3.dit.block.mlp"
    folded.receipt["whole_block"]["component"] = "h3.dit.block"

    arms = [a for a in arguments.arms.split(",") if a.strip()]
    table = mlp_arm_table(shell, folded.block.mlp, rows, fit_index, hold_index, keep=int(arguments.keep), seed=int(arguments.seed), arms=arms, selection=folded.selection, token_kind=kinds)
    replace = next((t for t in table if t["arm"] == "replace"), None)
    if replace is not None and folded.receipt["holdout_scores"]:
        drift = abs(replace["holdout_scores"]["rel_l2"] - folded.receipt["holdout_scores"]["rel_l2"])
        if drift > 1.0e-6:
            raise SystemExit(f"replace arm rel_l2 differs from the coreset receipt by {drift}: the arm table is not scoring the same child on the same rows")

    ratios = whole_block_receipt(shell, folded.block)
    target = Path(arguments.out) / f"L{int(arguments.block):02d}"
    target.mkdir(parents=True, exist_ok=True)
    written, selection_written = write_child(folded, int(arguments.block), target)
    header_mlp_parameters = int(written["parameters"])
    hold_kinds = kinds[hold_index]
    rows_by_kind_holdout = {"audio": int((hold_kinds == KIND_AUDIO).sum()), "video": int((hold_kinds == KIND_VIDEO).sum())}
    fit_kinds = kinds[fit_index]
    rows_by_kind_fit = {"audio": int((fit_kinds == KIND_AUDIO).sum()), "video": int((fit_kinds == KIND_VIDEO).sum())}
    dominant = max(rows_by_kind_holdout, key=lambda k: rows_by_kind_holdout[k]) if hold_index.numel() else None
    share = (rows_by_kind_holdout[dominant] / max(1, int(hold_index.numel()))) if dominant else None

    body = {
        "kind": "c16_h3_coreset",
        "stage": "S6H",
        "block": int(arguments.block),
        "keep": int(arguments.keep),
        "intermediate": int(folded.receipt["intermediate"]),
        "component": ratios["component"],
        "whole_block": ratios["whole_block"],
        "frozen_parameters": ratios["frozen_parameters"],
        "kv_head_merge_arm": ratios["kv_head_merge_arm"],
        "header_counts": {
            "parent_mlp_parameters": int(shell.mlp_parameters()),
            "child_mlp_parameters": header_mlp_parameters,
            "parent_block_parameters": int(shell.block_parameters()),
            "child_block_parameters": int(folded.block.block_parameters()),
            "mlp_ratio_child_over_parent": header_mlp_parameters / float(shell.mlp_parameters()),
            "mlp_ratio_le_0_25": header_mlp_parameters <= 0.25 * shell.mlp_parameters(),
        },
        "child": written,
        "selection_file": selection_written,
        "s7_consumability": s7_consumability(block_state, keep=int(arguments.keep)),
        "coreset": folded.receipt,
        "arms": table,
        "measured": {
            "holdout_group": group_name,
            "holdout_group_axis": group_notes.get("group_axis"),
            "holdout_scores_replace": folded.receipt["holdout_scores"],
            "fit_scores_replace": folded.receipt["fit_scores"],
            "per_arm_holdout": {t["arm"]: t["holdout_scores"] for t in table},
            "per_arm_holdout_by_kind": {t["arm"]: t.get("holdout_scores_by_kind", {}) for t in table},
            "per_arm_fit_by_kind": {t["arm"]: t.get("fit_scores_by_kind", {}) for t in table},
            "holdout_rows_by_kind": rows_by_kind_holdout,
            "fit_rows_by_kind": rows_by_kind_fit,
            "headline_kind": (
                f"the pooled '{arguments.rows}' holdout is {100.0 * share:.0f}% {dominant} rows "
                f"({rows_by_kind_holdout}); read per_arm_holdout_by_kind for the kind the S7 path will drive"
                if dominant else "no holdout rows"
            ),
            "solver": (folded.receipt.get("solve") or {}).get("solver"),
            "ace_block12_reference": ACE_BLOCK12_REFERENCE,
            "reading": (
                "holdout_scores are on rows held out by CLIP (group axis above; the membership row [video clip, "
                "parent audio clip] is re-derived per row and no fit row touches a held-out clip); one seed, one "
                "window, error bar unswept. The arms share the holdout so replace / control / random_select / "
                "prior_only are comparable to each other; they are the MLP-level analogues of the T1 arms, not "
                "the chart-level arms of the S7 install. prior_only scores the PIVOTED channels with the parent's "
                "down_proj columns verbatim; random_select scores its own random channels."
            ),
        },
        "rows": {
            "captured": int(merged["mlp_in"].shape[0]),
            "used": int(rows.shape[0]),
            "fit": int(fit_index.numel()),
            "holdout": int(hold_index.numel()),
            "selection": arguments.rows,
            "sigmas": sigmas,
            "steps": [str(p) for p in step_paths],
        },
        "sigma_grid_consumed": {
            "sigma_grid": capture_receipt.get("sigma_grid"),
            "sigmas": capture_receipt.get("sigmas"),
            "sigmas_in_rows": sigmas,
            "coverage_note": capture_receipt.get("sigma_coverage_note"),
        },
        "capture_deviations": capture_receipt.get("deviations"),
        "capture_stimulus_arms": capture_receipt.get("stimulus_arms"),
        "grouping": group_notes,
        "receptive_field": (capture_receipt.get("captures") or [{}])[0].get("receptive_field") if capture_receipt else None,
        "stored_parity": stored_parity,
        "capture": {
            "receipt": str(root / "capture_receipt.json"),
            "block_weights": str(weights_path),
            "block_weights_sha256": receipts.hash_file(weights_path),
            "compute_dtype": capture_receipt.get("compute_dtype"),
            "mlp_parity": (capture_receipt.get("mlp_parity") or {}).get(str(arguments.block)),
            "residency": capture_receipt.get("residency"),
            "vram_after_load": capture_receipt.get("vram_after_load"),
        },
        "wall_seconds": time.time() - started,
        "runtime": receipts.runtime_block("cpu"),
        "argv": vars(arguments),
        "source_sha256": receipts.source_sha256([str(_HERE / "c16_h3_coreset.py"), str(_HERE / "mediafold" / "c16_h3.py"), str(_HERE / "mediafold" / "c16_transplant.py")]),
    }
    receipt_path = target / "coreset_receipt.json"
    receipt_hash = receipts.write_receipt(receipt_path, body)
    if arguments.receipt_copy:
        copy_dir = Path(arguments.receipt_copy) / f"L{int(arguments.block):02d}"
        copy_dir.mkdir(parents=True, exist_ok=True)
        (copy_dir / "coreset_receipt.json").write_bytes(receipt_path.read_bytes())
    print(
        json.dumps(
            {
                "block": int(arguments.block),
                "keep": int(arguments.keep),
                "mlp_compression_x": ratios["component"]["compression_x"],
                "whole_block_compression_x": ratios["whole_block"]["compression_x"],
                "holdout": folded.receipt["holdout_scores"],
                "arms": {t["arm"]: {k: t["holdout_scores"].get(k) for k in ("rel_l2", "cos")} for t in table},
                "arms_by_kind": {t["arm"]: {kind: {k: sc.get(k) for k in ("rel_l2", "cos")} for kind, sc in t.get("holdout_scores_by_kind", {}).items()} for t in table},
                "headline_kind": body["measured"]["headline_kind"],
                "rows": body["rows"],
                "receipt_sha256": receipt_hash,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
