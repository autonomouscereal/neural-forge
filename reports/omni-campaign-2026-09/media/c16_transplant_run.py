"""C16 runner: capture an ACE-Step DiT block, chart it, score the five arms, fold it 4x.

One command produces the whole T1 table for one block:

1. load block ``i`` of ``acestep_v1.5_turbo.safetensors`` with a structural receipt
   (every source tensor consumed exactly once, each one's sha256 recorded),
2. capture its I/O on a procedural stimulus (``--rung random`` or ``--rung latents``),
3. collect host residual rows -- either a real Chimera trunk tap (``--chimera-src``,
   ``blocks[k].norm1`` input, the same site ``run_chimera_omni_toy_fold`` charts) or
   scale-matched gaussian rows (``--host synthetic``, plumbing only),
4. fit ``B_in`` / ``B_out`` by every chart method that applies at this width pair,
5. build and score all five arms (replace / insert / control / identity / random_chart),
6. fold the block's SwiGLU 6144 -> ``--keep`` by verbatim channel coreset + prior-anchored
   ridge refit, receipt the per-component and whole-block compression, and re-score the
   arms with the folded block installed.

Every number this prints is a DIAGNOSTIC that orders attempts. The gate is Cyril's
blind score on decoded audio from a trunk carrying the transplant; this runner does not
decode and does not claim a win.

Row pairing honesty: with ``--host synthetic`` the host and parent rows are NOT
observations of the same content, so the chart fit measures capacity, not correspondence.
That is labelled ``pairing: index_unpaired`` in the receipt. With ``--chimera-src`` plus
``--pairs`` the parent is driven from the same procedural corpus the host trunk sees and
the receipt says ``pairing: shared_corpus``.

Depth honesty: ``--prefix-shell`` used to drive block ``i`` straight off ``proj_in``, i.e.
off the LAYER-0 residual stream, so block 12 never saw layer-12 activations. The symptom
was in the receipt -- a residual write 28x its own input, which a trained residual block
does not do in distribution. The shell now loads layers ``0..i-1`` and runs them first, so
``x_in`` is the real layer-``i`` input; the delta/input ratio is receipted as
``parent_delta_over_input`` so the next reader can check the distribution rather than
trust this note.
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
from mediafold.ace_dit import (  # noqa: E402
    ACE_DEFAULT_CONFIG_PATH,
    ACE_DEFAULT_WEIGHTS_PATH,
    AceStepDiTConfig,
    load_ace_dit,
    load_ace_dit_block,
)
from mediafold.c16_capture import (  # noqa: E402
    capture,
    coreset_row_groups,
    latent_stimulus,
    random_stimulus,
    token_clip_membership,
)
from mediafold.c16_transplant import (  # noqa: E402
    ARMS,
    arm_table,
    coreset_block_mlp,
    fit_charts,
)


def chimera_host_rows(chimera_src: str, *, layer: int, batch: int, seed: int, device: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """Tap ``blocks[layer].norm1``'s input on the proof trunk's own fixture batch."""

    root = Path(chimera_src).resolve()
    package = root / "src" if (root / "src" / "chimera_omni").is_dir() else root
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    from chimera_omni.proof import h3_hybrid_av_flow as proof

    torch_device = torch.device(device)
    model = proof.build_model(torch_device)
    model.eval()
    generator = torch.Generator().manual_seed(int(seed))
    fixture = proof.make_batch(int(batch), generator)
    wired = proof.understanding_inputs(fixture, torch_device)

    captured: list[torch.Tensor] = []

    def hook(module, args):
        captured.append(args[0].detach().float().cpu())

    handle = model.blocks[int(layer)].norm1.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(
                wired["input_ids"],
                native_media_features=fixture["features"].to(torch_device),
                native_media_type_ids=fixture["type_ids"].to(torch_device),
                native_media_positions=fixture["positions"].to(torch_device),
                native_media_indices=wired["indices"],
                native_media_valid_mask=fixture["valid_mask"].to(torch_device),
            )
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("no host rows captured; check the block index")
    rows = captured[0].reshape(-1, captured[0].shape[-1])
    return rows, {
        "source": "chimera_proof_trunk",
        "package": str(package),
        "layer": int(layer),
        "hidden": int(rows.shape[1]),
        "rows": int(rows.shape[0]),
        "site": f"blocks[{int(layer)}].norm1 input",
    }


def synthetic_host_rows(count: int, hidden: int, reference: torch.Tensor, *, seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = torch.Generator().manual_seed(int(seed))
    rows = torch.randn(int(count), int(hidden), generator=generator, dtype=torch.float64)
    scale = reference.to(torch.float64).std() / rows.std().clamp_min(1.0e-30)
    return rows * scale, {"source": "synthetic_gaussian", "hidden": int(hidden), "rows": int(count), "scale": float(scale)}


def _holdout_reading(coreset: dict[str, Any], grouping: dict[str, Any], arguments) -> str:
    """The one paragraph a reader of this receipt has to have before quoting its holdout.

    It lives in the receipt because the caveat has to travel with the number: the two
    auxiliary files (``holdout_leak_probe.json``, ``holdout_decomposition.json``) are not
    what anyone opens when they want the score.
    """

    field = grouping.get("receptive_field", {})
    aligned = bool(field.get("batch_rows_are_single_clips"))
    scores = coreset.get("holdout_scores", {})
    parts = [
        f"holdout_scores are measured on rows held out by {coreset.get('split_by')}; "
        f"fit_scores on the rest.",
        (
            "A holdout score is NOT PAIRED with one from another grouping or another window: the "
            "splits hold out different rows, so two runs' numbers move for two reasons at once "
            "(the leak that was removed, and a different holdout composition). Do not read a "
            "difference between two runs as the fold getting better or worse. "
            "holdout_decomposition.json beside this receipt re-scores THIS child on THIS holdout "
            "with its weights read off disk and nothing refitted, split into the rows that leaked "
            "and the rows that did not; that decomposition, not a difference between runs, is "
            "what says how much the leak was worth. holdout_leak_probe.json measures the leak of "
            "each grouping on this run's own window, at row level AND through the attention "
            "receptive field."
        ),
    ]
    if aligned:
        parts.append(
            "This window is clip-aligned (every DiT batch row is one whole clip), so no held-out "
            "row's activation encodes fit material through the attention receptive field: this "
            "holdout is an unseen-content error."
        )
    else:
        parts.append(
            "This window is NOT clip-aligned: batch rows carry up to "
            f"{field.get('clips_per_batch_row_max')} clips and self-attention's receptive field is "
            "the whole batch row, so a held-out row's activation is still a function of the fit "
            "clips beside it. Zero row-level clip overlap is all that is established; this is "
            "not an unseen-content error. Re-run with --row-offset on a clip boundary and "
            "--length equal to the clip length for that."
        )
    if scores:
        parts.append(
            f"The number itself: rel_l2 {scores.get('rel_l2')}, cos {scores.get('cos')} on "
            f"{scores.get('rows')} rows, one window and one seed, error bar unswept."
        )
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C16 ACE-Step block transplant scaffolding")
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights", default=ACE_DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--config", default=ACE_DEFAULT_CONFIG_PATH)
    parser.add_argument("--block", type=int, default=12)
    parser.add_argument("--depth", type=int, default=0, help="truncate the stack for capture (0 = full 24)")
    parser.add_argument("--rung", default="random", choices=("random", "latents"))
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument(
        "--row-offset",
        type=int,
        default=None,
        help=(
            "pin the pair-corpus window instead of drawing it from --seed. A DiT batch row is "
            "the attention receptive field, so a clip-grouped holdout is only independent of "
            "the fit when each batch row is one whole clip: pass an offset on a clip boundary "
            "with --length equal to the clip length. The receipt records which window was used "
            "and whether it is clip-aligned."
        ),
    )
    parser.add_argument("--host", default="synthetic", choices=("synthetic", "chimera"))
    parser.add_argument("--chimera-src", default=None)
    parser.add_argument("--host-layer", type=int, default=3)
    parser.add_argument("--host-batch", type=int, default=8)
    parser.add_argument("--keep", type=int, default=1536, help="coreset width (6144/4 = 1536 is the 4x rung)")
    parser.add_argument("--chart-methods", default="ridge,svd,random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument(
        "--prefix-shell",
        "--skip-capture",
        dest="prefix_shell",
        action="store_true",
        help=(
            "load layers 0..block-1 plus proj_in/temb and run them to produce the block's real "
            "input, instead of materialising all 24 layers. For the block's INPUT this is "
            "equivalent to the full stack (later layers cannot influence it) and much cheaper. "
            "--skip-capture is the old name for this flag and no longer means 'no forward'."
        ),
    )
    parser.add_argument(
        "--assert-shared-corpus",
        action="store_true",
        help=(
            "caller asserts the host rows and the parent rows are observations of the SAME "
            "media; without it a latents-rung run is receipted as index-aligned across two "
            "different corpora, which is what it is"
        ),
    )
    arguments = parser.parse_args(argv)

    output = Path(arguments.out)
    output.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    config = AceStepDiTConfig.from_json(arguments.config)

    block, block_receipt = load_ace_dit_block(
        arguments.block, weights=arguments.weights, config=config, device=arguments.device, dtype=dtype
    )
    block.eval()

    # ---- parent rows -----------------------------------------------------------------
    if arguments.prefix_shell:
        stimulus = (
            latent_stimulus(config, arguments.pairs, batch=arguments.batch, length=arguments.length,
                            seed=arguments.seed, row_offset=arguments.row_offset,
                            device=arguments.device, dtype=dtype)
            if arguments.rung == "latents"
            else random_stimulus(config, batch=arguments.batch, length=arguments.length,
                                 seed=arguments.seed, device=arguments.device, dtype=dtype)
        )
        # Layers 0..block-1 plus proj_in / temb: the block's REAL input, not the layer-0
        # residual stream. Layers after `block` cannot influence `block`'s input, so this
        # is the full-stack answer at a fraction of the memory.
        prefix_depth = int(arguments.block)
        shell, shell_receipt = load_ace_dit(
            weights=arguments.weights, config=config, num_layers=max(prefix_depth, 1),
            device=arguments.device, dtype=dtype,
        )
        shell.eval()
        captured_mlp_in: list[torch.Tensor] = []

        def mlp_pre_hook(module, args):
            captured_mlp_in.append(args[0].detach().float().cpu())

        handle = block.mlp.register_forward_pre_hook(mlp_pre_hook)
        try:
            with torch.no_grad():
                prepared = shell.block_inputs(
                    stimulus.hidden_states, stimulus.timestep, stimulus.timestep_r,
                    stimulus.encoder_hidden_states, stimulus.context_latents,
                )
                position = (prepared["cos"], prepared["sin"])
                x_in = prepared["hidden"]
                for depth in range(prefix_depth):
                    x_in = shell.layers[depth](
                        x_in, prepared["temb"], encoder_hidden_states=prepared["context"],
                        position_embeddings=position,
                    )
                x_out = block(x_in, prepared["temb"], encoder_hidden_states=prepared["context"],
                              position_embeddings=position)
        finally:
            handle.remove()
        if not captured_mlp_in:
            raise RuntimeError("the block's MLP was never called; cannot fit a channel coreset")
        records = {
            "x_in": x_in.float().cpu(),
            "x_out": x_out.float().cpu(),
            "temb": prepared["temb"].float().cpu(),
            "context": prepared["context"].float().cpu(),
            # the MLP's real input: modulated post-attention rows, NOT mlp_norm(x_in)
            "mlp_in": captured_mlp_in[0],
        }
        capture_receipt = {
            "mode": "prefix_shell",
            "prefix_depth": prefix_depth,
            "prefix_note": (
                f"layers 0..{prefix_depth - 1} were run before block {arguments.block}; "
                "identical to a full-stack forward for this block's input"
            ),
            "shell": shell_receipt,
            "stimulus": stimulus.notes,
        }
        del shell
    else:
        model, load_receipt = load_ace_dit(
            weights=arguments.weights, config=config, num_layers=arguments.depth or None,
            device=arguments.device, dtype=dtype,
        )
        model.eval()
        stimulus = (
            latent_stimulus(config, arguments.pairs, batch=arguments.batch, length=arguments.length,
                            seed=arguments.seed, row_offset=arguments.row_offset,
                            device=arguments.device, dtype=dtype)
            if arguments.rung == "latents"
            else random_stimulus(config, batch=arguments.batch, length=arguments.length,
                                 seed=arguments.seed, device=arguments.device, dtype=dtype)
        )
        records = capture(model, stimulus, [arguments.block])[arguments.block]
        capture_receipt = {"mode": "full_stack", "load": load_receipt, "stimulus": stimulus.notes}
        del model

    parent_in = records["x_in"].reshape(-1, records["x_in"].shape[-1]).to(torch.float64)
    parent_out = records["x_out"].reshape(-1, records["x_out"].shape[-1]).to(torch.float64)
    capture_state = receipts.write_tensors(records, output / f"capture_L{arguments.block:02d}.safetensors")

    # ---- host rows -------------------------------------------------------------------
    if arguments.host == "chimera":
        if not arguments.chimera_src:
            parser.error("--host chimera requires --chimera-src")
        host_rows, host_notes = chimera_host_rows(
            arguments.chimera_src, layer=arguments.host_layer, batch=arguments.host_batch,
            seed=arguments.seed, device=arguments.device,
        )
        host_rows = host_rows.to(torch.float64)
        # Honesty about what the chart fit actually saw. The host rows come from the proof
        # trunk's own AV fixture (32 kHz tones + moving squares); with --rung latents the
        # parent rows come from the ACE VAE's latents of the procedural audio corpus. Those
        # are two DIFFERENT corpora sampled at different rates, so the rows are aligned by
        # index, not by content, and the chart fit measures capacity rather than
        # correspondence. Only an explicit assertion by the caller (who has arranged one
        # corpus for both sides) may call it shared.
        if arguments.rung == "latents":
            pairing = "shared_corpus" if arguments.assert_shared_corpus else "index_aligned_different_corpora"
        else:
            pairing = "index_unpaired"
    else:
        host_rows, host_notes = synthetic_host_rows(parent_in.shape[0], 128, parent_in, seed=arguments.seed)
        pairing = "index_unpaired"
    count = min(host_rows.shape[0], parent_in.shape[0])
    host_rows = host_rows[:count]
    parent_in = parent_in[:count]
    parent_out = parent_out[:count]
    host_notes["paired_rows"] = int(count)
    host_notes["pairing"] = pairing

    # ---- charts ----------------------------------------------------------------------
    chart_records: dict[str, Any] = {}
    charts = {}
    for method in [value.strip() for value in arguments.chart_methods.split(",") if value.strip()]:
        if method == "procrustes" and host_rows.shape[1] != parent_in.shape[1]:
            chart_records[method] = {"skipped": "widths differ; Procrustes needs equal widths"}
            continue
        fitted = fit_charts(host_rows, parent_in, method=method, seed=arguments.seed)
        charts[method] = fitted
        chart_records[method] = {"method": method, "parameters": fitted.parameters, **fitted.notes}
        receipts.write_tensors(fitted.state(), output / f"chart_{method}.safetensors")

    primary = charts.get("ridge") or next(iter(charts.values()))

    # How hard does the parent's own delta hit the host residual stream once transported?
    # A transplant that vanishes (<<1) or explodes (>>1) is unusable whatever its arm score.
    parent_delta = parent_out - parent_in
    transported = parent_delta @ primary.chart_out.weight.transpose(0, 1)
    write_strength = {
        "parent_delta_norm": float(parent_delta.norm().item()),
        "parent_in_norm": float(parent_in.norm().item()),
        # distribution check: a trained residual block writes a delta of the same order as
        # its input. Far above 1 means the block is being driven off-manifold (that is what
        # a layer-0 stream fed to block 12 looked like: 28.5x).
        "parent_delta_over_input": float(
            (parent_delta.norm() / parent_in.norm().clamp_min(1e-30)).item()
        ),
        "transported_delta_norm": float(transported.norm().item()),
        "host_row_norm": float(host_rows.norm().item()),
        "relative_write": float((transported.norm() / host_rows.norm().clamp_min(1e-30)).item()),
        "chart": primary.method,
        "chart_forward_r2": primary.notes.get("forward_scores", {}).get("r2"),
        "chart_forward_cos": primary.notes.get("forward_scores", {}).get("cos"),
    }

    # ---- arms ------------------------------------------------------------------------
    host_batched = host_rows.reshape(1, count, -1).to(torch.float32)
    temb_rows = records["temb"][:1].to(torch.float32)
    context_rows = records["context"][:1].to(torch.float32) if "context" in records else None
    target = host_rows.reshape(1, count, -1).to(torch.float32)
    parent_arms = arm_table(
        block.to(torch.float32),
        primary,
        host_batched,
        target,
        temb=temb_rows,
        context=context_rows,
        arms=ARMS,
        seed=arguments.seed,
        rope_theta=float(config.rope_theta),
    )

    # ---- 4x coreset ------------------------------------------------------------------
    mlp_capture = records["mlp_in"]
    mlp_rows = mlp_capture.reshape(-1, mlp_capture.shape[-1])
    # The coreset holdout is grouped by CLIP, never by the DiT batch row -- see
    # `coreset_row_groups`, which owns that decision so it can be tested directly.
    keep_mask, mlp_sequences, group_name, clip_grouping = coreset_row_groups(
        stimulus, mlp_capture, patch_size=int(config.patch_size)
    )
    coreset_rows = mlp_rows[keep_mask]
    # What each kept row's latents ACTUALLY draw from, derived from the clip map rather
    # than from the group ids, so `coreset_block_mlp`'s guard has something independent to
    # check the split against.
    row_membership = (
        token_clip_membership(
            stimulus.clip_index,
            tokens=int(mlp_capture.shape[1]),
            patch_size=int(config.patch_size),
        )[keep_mask]
        if stimulus.clip_index is not None
        else None
    )
    folded = coreset_block_mlp(
        block.to(torch.float32),
        coreset_rows.to(torch.float32),
        keep=arguments.keep,
        seed=arguments.seed,
        sequence_index=mlp_sequences,
        group_name=group_name,
        row_membership=row_membership,
    )
    folded.receipt["clip_grouping"] = clip_grouping
    folded.receipt["reading"] = _holdout_reading(folded.receipt, clip_grouping, arguments)
    folded_arms = arm_table(
        folded.block,
        primary,
        host_batched,
        target,
        temb=temb_rows,
        context=context_rows,
        arms=("replace", "identity"),
        seed=arguments.seed,
        rope_theta=float(config.rope_theta),
    )
    folded_state = receipts.write_tensors(
        {name: value.detach().cpu() for name, value in folded.block.state_dict().items()},
        output / f"folded_block_L{arguments.block:02d}_keep{arguments.keep}.safetensors",
    )

    body = {
        "kind": "c16_transplant",
        "block": block_receipt,
        "capture": {**capture_receipt, "tensors": capture_state},
        "host": host_notes,
        "charts": chart_records,
        "primary_chart": primary.method,
        "arms_parent_block": parent_arms,
        "arms_folded_block": folded_arms,
        "arm_metric_meaning": (
            "each arm's rel_l2 is measured against the UNCHANGED host rows, so it is the "
            "relative size of the write the arm makes into the host residual stream "
            "(0 = the arm does nothing). It is not a fit error against a target. Every arm "
            "number is measured THROUGH the primary chart, so read it next to that chart's "
            "own forward_scores: when they are near zero the chart carries no "
            "correspondence and the arm table orders write strengths, nothing more."
        ),
        "arm_caveats": {
            "insert_equals_replace": (
                "at one transplanted block the insert and replace forward paths are the same "
                "expression; the rows are identical by construction, not two independent arms"
            ),
            "control_is_norm_matched": (
                "control keeps the parent RMSNorm gains (zeroing them makes a DEAD block) and "
                "its delta is rescaled to the parent's delta norm on the same rows"
            ),
        },
        "transplant_write_strength": write_strength,
        "coreset": folded.receipt,
        "holdout_reading": folded.receipt["reading"],
        "folded_block_tensors": folded_state,
        "diagnostic_only": (
            "arm scores order attempts; the behavioural instrument is a blind score on decoded "
            "audio from a trunk carrying the transplant"
        ),
        "runtime": receipts.runtime_block(arguments.device),
        "argv": vars(arguments),
    }
    receipt_hash = receipts.write_receipt(output / "c16_receipt.json", body)  # the receipt cannot contain its own hash; it is printed instead
    print(
        json.dumps(
            {
                "block": arguments.block,
                "layer_type": block_receipt["layer_type"],
                "block_parameters": block_receipt["block_parameters"],
                "coreset": folded.receipt["component"],
                "whole_block": folded.receipt["whole_block"],
                "coreset_holdout": folded.receipt["holdout_scores"],
                "charts": {name: value.get("roundtrip_scores", {}).get("rel_l2") for name, value in chart_records.items()},
                "arms": {row["arm"]: row["scores"]["rel_l2"] for row in parent_arms},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
