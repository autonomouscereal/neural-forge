"""C16 block I/O capture on the ACE-Step DiT.

Records, per block and per sampler step, everything a transplant needs:

``x_in`` / ``x_out``     the block's residual-stream input and output   [rows, 2048]
``temb``                 the six adaLN modulation rows                  [rows, 6, 2048]
``context``              the cross-attention condition after
                         ``condition_embedder``                         [rows_c, 2048]
``self_attn_out`` / ``cross_attn_out`` / ``mlp_out``
                         the three residual writes, so an arm can be
                         attributed to a sub-module rather than guessed
``mlp_hidden``           the gated channel activations -- the C16 4x
                         channel coreset selects columns of this

Two stimulus rungs, both procedural (no model outputs are ever used as fold data):

rung 0 ``random``   seeded gaussian latents in the parent's own smoke shape; proves the
                    plumbing bit-exactly and costs nothing.
rung 1 ``latents``  real ACE VAE latents of procedural audio, replayed from a
                    ``mediafold.c15_pairs`` directory, so the captured rows sit on the
                    manifold the parent actually sees.

No H3 is run here (licence gate pending); the ACE-Step 1.5 weights are MIT.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from . import receipts
from .ace_dit import AceStepDiTConfig, AceStepDiTLayer, AceStepDiTModel, load_ace_dit


@dataclass
class Stimulus:
    hidden_states: torch.Tensor  # [B, L, 64] noisy acoustic latents
    context_latents: torch.Tensor  # [B, L, 128] source + chunk-mask channels
    encoder_hidden_states: torch.Tensor  # [B, Lc, 2048]
    timestep: torch.Tensor  # [B]
    timestep_r: torch.Tensor  # [B]
    notes: dict[str, Any]
    # provenance of every latent ROW, [B, L], carried from the pair corpus so a
    # downstream holdout can be grouped by CLIP. ``None`` when the rows have no clip
    # provenance at all (the ``random`` rung); -1 marks a row that has none.
    clip_index: torch.Tensor | None = None


def random_stimulus(
    config: AceStepDiTConfig,
    *,
    batch: int = 2,
    length: int = 64,
    condition_length: int = 32,
    sigma: float = 0.5,
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Stimulus:
    generator = torch.Generator().manual_seed(int(seed))
    acoustic = int(config.audio_acoustic_hidden_dim)
    context_channels = int(config.in_channels) - acoustic
    hidden = torch.randn(batch, length, acoustic, generator=generator, dtype=torch.float32)
    context = torch.randn(batch, length, context_channels, generator=generator, dtype=torch.float32)
    condition = torch.randn(batch, condition_length, config.condition_dim, generator=generator, dtype=torch.float32)
    timestep = torch.full((batch,), float(sigma), dtype=torch.float32)
    return Stimulus(
        hidden_states=hidden.to(device=device, dtype=dtype),
        context_latents=context.to(device=device, dtype=dtype),
        encoder_hidden_states=condition.to(device=device, dtype=dtype),
        timestep=timestep.to(device=device, dtype=dtype),
        timestep_r=torch.zeros_like(timestep).to(device=device, dtype=dtype),
        notes={"rung": "random", "seed": int(seed), "sigma": float(sigma), "length": int(length)},
    )


def latent_stimulus(
    config: AceStepDiTConfig,
    pairs_dir: str | Path,
    *,
    batch: int = 2,
    length: int = 64,
    condition_length: int = 32,
    sigma: float = 0.5,
    seed: int = 0,
    row_offset: int | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Stimulus:
    """Drive the DiT from real VAE latents sampled by ``mediafold.c15_pairs``.

    The source channels carry the clean latent, the noisy channels carry the rectified
    -flow interpolate ``(1-sigma)*z + sigma*noise``, and the chunk-mask channels are
    zero -- the parent's own layout, filled with procedurally sourced content.

    ``row_offset`` pins the corpus window instead of drawing it from ``seed``. It exists
    for ONE reason: a batch row is the attention receptive field, so only a window whose
    batch rows are whole clips (``row_offset`` on a clip boundary, ``length`` = the clip
    length) produces activations a clip-grouped holdout can call independent. The seeded
    draw still happens either way, so the noise and condition streams are identical
    whichever window is used and two runs differ only by the window.
    """

    from .c15_pairs import load_pair_set

    pairs = load_pair_set(pairs_dir)
    acoustic = int(config.audio_acoustic_hidden_dim)
    if int(pairs.latent.shape[1]) != acoustic:
        raise ValueError(
            f"latent width {pairs.latent.shape[1]} != ACE acoustic width {acoustic}; "
            "sample pairs with the ACE 1.5 VAE (64 latent channels)"
        )
    generator = torch.Generator().manual_seed(int(seed))
    rows = pairs.latent
    take = int(batch) * int(length)
    if rows.shape[0] < take:
        raise ValueError(f"pair set has {rows.shape[0]} rows; need {take}")
    drawn = int(torch.randint(0, rows.shape[0] - take + 1, (1,), generator=generator).item())
    if row_offset is None:
        start, offset_source = drawn, "seeded_random"
    else:
        start, offset_source = int(row_offset), "explicit"
        if not 0 <= start <= rows.shape[0] - take:
            raise ValueError(
                f"row_offset {start} does not fit {take} rows in a {int(rows.shape[0])}-row corpus"
            )
    clean = rows[start : start + take].reshape(batch, length, acoustic).to(torch.float32)
    # The clip each sampled latent row came from, on the same [B, L] grid as `clean`.
    # The window is a contiguous slice of the corpus and one clip is far shorter than
    # one batch row, so a batch row spans several clips; without this the only grouping
    # available downstream is the batch row, which leaks clips across a holdout.
    clip_rows = pairs.clip_index.reshape(-1)[start : start + take].reshape(batch, length).to(torch.long)
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
    noisy = (1.0 - float(sigma)) * clean + float(sigma) * noise
    context = torch.cat((clean, torch.zeros_like(clean)), dim=-1)
    if context.shape[-1] != int(config.in_channels) - acoustic:
        raise ValueError("context channel layout does not match the parent's in_channels")
    condition = torch.randn(batch, condition_length, config.condition_dim, generator=generator, dtype=torch.float32)
    timestep = torch.full((batch,), float(sigma), dtype=torch.float32)
    return Stimulus(
        hidden_states=noisy.to(device=device, dtype=dtype),
        context_latents=context.to(device=device, dtype=dtype),
        encoder_hidden_states=condition.to(device=device, dtype=dtype),
        timestep=timestep.to(device=device, dtype=dtype),
        timestep_r=torch.zeros_like(timestep).to(device=device, dtype=dtype),
        notes={
            "rung": "latents",
            "pairs_dir": str(Path(pairs_dir).resolve()),
            "pairs_geometry": pairs.geometry,
            "seed": int(seed),
            "sigma": float(sigma),
            "row_offset": start,
            "row_offset_source": offset_source,
            "clips_covered": int(torch.unique(clip_rows).numel()),
            "clip_span": [int(clip_rows.min().item()), int(clip_rows.max().item())],
            "rows_per_clip_in_window": int(torch.bincount(clip_rows.reshape(-1) - int(clip_rows.min().item())).max().item()),
            **receptive_field_notes(clip_rows),
        },
        clip_index=clip_rows,
    )


def receptive_field_notes(clip_rows: torch.Tensor) -> dict[str, Any]:
    """What a clip-grouped holdout can and cannot claim about THIS window.

    Self-attention inside a DiT batch row is not row-local: the ``full_attention`` layers
    are unrestricted and a sliding layer whose window covers the row is full too. So every
    token's activation is a function of every latent row of its batch row. Grouping the
    holdout by clip removes row-level clip overlap; it makes the fit and holdout
    ACTIVATIONS independent only when each batch row is one whole clip.
    """

    per_row = [int(torch.unique(row).numel()) for row in clip_rows]
    single = max(per_row) == 1
    return {
        "clips_per_batch_row_min": int(min(per_row)),
        "clips_per_batch_row_max": int(max(per_row)),
        "batch_rows_are_single_clips": bool(single),
        "attention_receptive_field": (
            "one DiT batch row (the full_attention layers are unrestricted), so a token's "
            "activation is a function of every latent row of its batch row"
            + (
                ". Each batch row here is exactly one clip, so an activation carries no "
                "content from any other clip and a clip-grouped holdout is independent of "
                "the fit."
                if single
                else (
                    f". Batch rows here carry up to {int(max(per_row))} clips, so a "
                    "clip-grouped holdout removes row-level clip overlap but its activations "
                    "still encode content from the other clips in the same batch row -- the "
                    "holdout is NOT independent of the fit. Pin --row-offset to a clip "
                    "boundary and set --length to the clip length for an independent window."
                )
            )
        ),
    }


def token_clip_membership(
    clip_index: torch.Tensor,
    *,
    tokens: int,
    patch_size: int,
) -> torch.Tensor:
    """Every clip each post-patchify token's latent rows draw from, ``[B*tokens, patch]``.

    This is the membership a holdout guard has to check: the id a token is FILED under
    says nothing, the ids its rows actually touch are what leaks. ``-1`` marks a padded
    row of a trailing partial patch. Shares its arithmetic with ``token_clip_index``, so
    the two can never disagree about which rows a token owns.
    """

    return _patch_grid(clip_index, tokens=tokens, patch_size=patch_size).reshape(-1, int(patch_size))


def _patch_grid(clip_index: torch.Tensor, *, tokens: int, patch_size: int) -> torch.Tensor:
    """``[B, tokens, patch_size]`` clip ids, padded with -1 exactly where ``proj_in`` pads."""

    rows = clip_index.reshape(int(clip_index.shape[0]), -1).to(torch.long)
    batch, length = rows.shape
    patch = int(patch_size)
    if patch < 1:
        raise ValueError("patch_size must be >= 1")
    pad = (-length) % patch
    if pad:
        rows = torch.cat((rows, torch.full((batch, pad), -1, dtype=torch.long)), dim=1)
    grid = rows.reshape(batch, -1, patch)
    if int(grid.shape[1]) != int(tokens):
        raise ValueError(
            f"{length} latent rows at patch_size {patch} give {int(grid.shape[1])} tokens, "
            f"but the capture has {int(tokens)}; the clip map does not describe these rows"
        )
    return grid


def token_clip_index(
    clip_index: torch.Tensor,
    *,
    tokens: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map per-latent-row clip ids onto the DiT's post-patchify token grid.

    ``proj_in`` is a Conv1d with kernel = stride = ``patch_size``, so token j of a
    sequence owns latent rows ``[j*patch_size, (j+1)*patch_size)`` and the trailing
    partial patch is zero-padded. Returns ``(groups, mixed)``, both flattened over
    [B, tokens] in the same order the block's rows flatten:

    ``groups``  the clip of the token's FIRST latent row
    ``mixed``   the token's patch straddles a clip boundary (or eats padding), so the
                token belongs to more than one clip

    A mixed token cannot be assigned to one clip without carrying held-out content into
    the fit, so the caller drops it. There are only ~one per clip boundary.
    """

    grid = _patch_grid(clip_index, tokens=tokens, patch_size=patch_size)
    first = grid[:, :, 0]
    mixed = (grid != first.unsqueeze(-1)).any(dim=-1) | (grid < 0).any(dim=-1)
    return first.reshape(-1), mixed.reshape(-1)


def coreset_row_groups(
    stimulus: Stimulus,
    mlp_capture: torch.Tensor,
    *,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, str, dict[str, Any]]:
    """Decide how the C16 coreset holdout is grouped, for the rows of ``mlp_capture``.

    The C16 coreset holdout is grouped by CLIP, never by the DiT batch row. One batch row
    is a contiguous ``length``-row window of the pair corpus and a clip is far shorter
    than that, so a batch row spans several clips: group by batch row and most held-out
    rows still share a clip with the fit, which makes the "holdout" score part fit score.

    Returns ``(keep_mask, groups, group_name, notes)`` over the flattened [B*tokens] rows
    of ``mlp_capture``. ``keep_mask`` drops the tokens whose patch straddles a clip
    boundary: such a token carries rows from two clips and cannot be filed under one
    without putting held-out content in the fit. Only the ``random`` rung, which has no
    clip provenance at all, falls back to the batch row -- and says so.

    What this buys is exactly ROW-LEVEL clip disjointness: no clip has rows on both sides
    of the split. It is NOT statistical independence between the fit and holdout
    activations. ``mlp_capture`` is taken after several DiT layers of self-attention whose
    receptive field is the whole batch row, so a token's activation is a function of every
    clip in its batch row. The split is independent only when each batch row is one whole
    clip; ``notes["receptive_field"]`` records whether that holds for this window, and
    ``latent_stimulus(row_offset=...)`` is how a caller arranges it.
    """

    tokens = int(mlp_capture.shape[1])
    sequences = int(mlp_capture.shape[0])
    clip_index = getattr(stimulus, "clip_index", None)
    if clip_index is None:
        groups = torch.arange(sequences).repeat_interleave(tokens)
        notes = {
            "grouped_by": "dit_batch_row",
            "reason": (
                "this stimulus carries no clip provenance (the random rung), so the only "
                "grouping available is the batch row; rows of one batch row are a "
                "contiguous window and neighbouring windows may still share content"
            ),
            "groups": sequences,
            "boundary_tokens_dropped": 0,
            "rows_before_drop": sequences * tokens,
            "rows_used": sequences * tokens,
            "receptive_field": {
                "batch_rows_are_single_clips": False,
                "attention_receptive_field": (
                    "one DiT batch row; with no clip provenance nothing can be said about "
                    "what content a held-out row's activation encodes"
                ),
            },
        }
        return torch.ones(sequences * tokens, dtype=torch.bool), groups, "dit_batch_row", notes

    clip_groups, straddling = token_clip_index(clip_index, tokens=tokens, patch_size=patch_size)
    keep_mask = ~straddling
    groups = clip_groups[keep_mask]
    notes = {
        "grouped_by": "clip",
        "groups": int(torch.unique(groups).numel()),
        "boundary_tokens_dropped": int(straddling.sum().item()),
        "rows_before_drop": int(clip_groups.numel()),
        "rows_used": int(keep_mask.sum().item()),
        "note": (
            "a token whose patch straddles a clip boundary carries rows from two clips and "
            "is dropped; filing it under either clip would put held-out content in the fit"
        ),
        "guarantee": (
            "row-level clip disjointness (no clip has rows on both sides), NOT independence "
            "of the fit and holdout activations -- read receptive_field"
        ),
        "receptive_field": receptive_field_notes(clip_index),
    }
    return keep_mask, groups, "clip", notes


class BlockTap:
    """Forward hooks over a set of ``AceStepDiTLayer`` modules and their sub-writes."""

    def __init__(self, model: AceStepDiTModel, block_ids: Sequence[int]) -> None:
        self.model = model
        self.block_ids = [int(value) for value in block_ids]
        self.records: dict[int, dict[str, torch.Tensor]] = {index: {} for index in self.block_ids}
        self._handles: list[Any] = []

    def __enter__(self) -> "BlockTap":
        for index in self.block_ids:
            layer = self.model.layers[index]
            self._handles.append(layer.register_forward_hook(self._layer_hook(index), with_kwargs=True))
            self._handles.append(layer.self_attn.register_forward_hook(self._write_hook(index, "self_attn_out")))
            self._handles.append(layer.cross_attn.register_forward_hook(self._write_hook(index, "cross_attn_out")))
            self._handles.append(layer.mlp.register_forward_hook(self._write_hook(index, "mlp_out")))
            self._handles.append(layer.mlp.register_forward_pre_hook(self._mlp_input_hook(index)))
        return self

    def __exit__(self, *exception: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _layer_hook(self, index: int):
        def hook(module, args, kwargs, output):
            store = self.records[index]
            store["x_in"] = args[0].detach().float().cpu()
            store["temb"] = args[1].detach().float().cpu()
            context = kwargs.get("encoder_hidden_states")
            if context is None and len(args) > 2:
                context = args[2]
            if context is not None:
                store["context"] = context.detach().float().cpu()
            store["x_out"] = output.detach().float().cpu()

        return hook

    def _write_hook(self, index: int, name: str):
        def hook(module, args, output):
            self.records[index][name] = output.detach().float().cpu()

        return hook

    def _mlp_input_hook(self, index: int):
        def hook(module, args):
            rows = args[0].detach()
            self.records[index]["mlp_in"] = rows.float().cpu()
            with torch.no_grad():
                self.records[index]["mlp_hidden"] = module.hidden(rows).float().cpu()

        return hook


def capture(
    model: AceStepDiTModel,
    stimulus: Stimulus,
    block_ids: Sequence[int],
) -> dict[int, dict[str, torch.Tensor]]:
    with torch.no_grad(), BlockTap(model, block_ids) as tap:
        model(
            stimulus.hidden_states,
            stimulus.timestep,
            stimulus.timestep_r,
            stimulus.encoder_hidden_states,
            stimulus.context_latents,
        )
        return {index: dict(record) for index, record in tap.records.items()}


def flatten_rows(records: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """[B, L, W] -> [B*L, W]; ``temb`` [B, 6, W] is broadcast across the L axis."""

    tensor = records[name]
    if name == "temb":
        length = records["x_in"].shape[1]
        return tensor.unsqueeze(1).expand(-1, length, -1, -1).reshape(-1, tensor.shape[-2], tensor.shape[-1])
    return tensor.reshape(-1, tensor.shape[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C16 ACE-Step DiT block I/O capture")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--blocks", default="12", help="comma separated block indices")
    parser.add_argument("--depth", type=int, default=0, help="truncate the stack (0 = full 24)")
    parser.add_argument("--rung", default="random", choices=("random", "latents"))
    parser.add_argument("--pairs", default=None, help="pair dir for --rung latents")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1, help="capture at this many sigma levels")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    arguments = parser.parse_args(argv)

    from .ace_dit import ACE_DEFAULT_CONFIG_PATH, ACE_DEFAULT_WEIGHTS_PATH

    config = AceStepDiTConfig.from_json(arguments.config or ACE_DEFAULT_CONFIG_PATH)
    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    model, load_receipt = load_ace_dit(
        weights=arguments.weights or ACE_DEFAULT_WEIGHTS_PATH,
        config=config,
        num_layers=arguments.depth or None,
        device=arguments.device,
        dtype=dtype,
    )
    model.eval()
    block_ids = [int(value) for value in arguments.blocks.split(",") if value.strip()]

    target = Path(arguments.out)
    target.mkdir(parents=True, exist_ok=True)
    step_receipts: list[dict[str, Any]] = []
    for step in range(int(arguments.steps)):
        sigma = (step + 0.5) / float(arguments.steps)
        if arguments.rung == "latents":
            if not arguments.pairs:
                parser.error("--rung latents requires --pairs")
            stimulus = latent_stimulus(
                config, arguments.pairs, batch=arguments.batch, length=arguments.length,
                sigma=sigma, seed=arguments.seed + step, device=arguments.device, dtype=dtype,
            )
        else:
            stimulus = random_stimulus(
                config, batch=arguments.batch, length=arguments.length,
                sigma=sigma, seed=arguments.seed + step, device=arguments.device, dtype=dtype,
            )
        records = capture(model, stimulus, block_ids)
        for index, store in records.items():
            path = target / f"step{step:02d}" / f"L{index:02d}.safetensors"
            step_receipts.append(
                {
                    "step": step,
                    "block": index,
                    "sigma": sigma,
                    "stimulus": stimulus.notes,
                    **receipts.write_tensors(store, path),
                }
            )

    body = {
        "kind": "c16_capture",
        "load": load_receipt,
        "blocks": block_ids,
        "captures": step_receipts,
        "runtime": receipts.runtime_block(arguments.device),
        "argv": vars(arguments),
    }
    receipt_hash = receipts.write_receipt(target / "capture_receipt.json", body)  # the receipt cannot contain its own hash; it is printed instead
    print(json.dumps({"blocks": block_ids, "captures": len(step_receipts), "depth": load_receipt["depth"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
