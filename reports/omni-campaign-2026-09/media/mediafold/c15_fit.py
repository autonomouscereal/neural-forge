"""C15 map fitting and composition into Chimera's boundary projection / emission head.

Two maps per media band, both closed form:

``E : raw -> latent``  what the trunk's INPUT boundary has to learn to speak.
``D : latent -> raw``  what the trunk's OUTPUT head has to learn to speak.

Rungs (per the R6 spec, cheapest first): linear per patch -> + context -> + one
closed-form nonlinearity (a stored random-feature lift, parameters counted in full).

Scoring: latent relL2 / cos for ``E``; PSNR (and SI-SDR for audio) of ``D(E(raw))``
against the raw rows AND against the parent VAE's own round-trip. The parent's
round-trip is the CEILING; the proposed decision line for Cyril (D-d2) is

    absorbed  <=>  PSNR(D.E) >= PSNR(parent round-trip) - 3 dB   on held-out clips

Below that line the trunk keeps a companion VAE decoder for that modality (D5b) --
that is a proposal, not a gate this harness enforces.

Composition (exact linear algebra, no training)
-----------------------------------------------
Input side, ``NativeContinuousMediaBridge.feature_proj.weight`` [hidden, d_raw]::

    feature_proj  =  B_in @ P_src @ E_w
    type_embed[t] += B_in @ P_src @ E_b

where ``P_src`` is the parent DiT's patch projection (H3 ``video_patch_proj``
[5376, 96] / ``audio_patch_proj`` [5376, 32]; ACE ``proj_in`` linear part; Qwen-Image
``img_in``) and ``B_in`` is the C17 residual chart from the parent width to Chimera's
hidden width.

Output side, ``UnifiedNativeMediaEmissionHead.out.weight`` [d_raw, hidden]::

    out  =  D_w @ P_out @ B_out
    (D_b is approximated in the head's per-type affine shift; residual recorded)

Both compositions are shape-checked and receipted; nothing is trained here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from . import receipts
from .c15_pairs import PairSet, load_pair_set
from .linalg import (
    LinearMap,
    RandomFeatureLift,
    best_linear_rank_psnr,
    fit_kernel_ridge_lift,
    fit_ridge_lambda_ladder,
    fit_ridge_linear,
    psnr,
    si_sdr_detail,
    svd_chart,
    write_scores,
)

#: Proposed decision line D-d2 (for Cyril to ratify), in dB below the parent ceiling.
PROPOSED_ABSORPTION_MARGIN_DB = 3.0


@dataclass
class AbsorptionMaps:
    encoder: LinearMap
    decoder: LinearMap
    encoder_lift: RandomFeatureLift | None = None
    decoder_lift: RandomFeatureLift | None = None
    rung: str = "linear"
    metrics: dict[str, Any] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)

    def encode_rows(self, rows: torch.Tensor) -> torch.Tensor:
        source = self.encoder_lift.apply(rows) if self.encoder_lift is not None else rows.to(torch.float64)
        return self.encoder.apply(source)

    def decode_rows(self, latent: torch.Tensor) -> torch.Tensor:
        source = self.decoder_lift.apply(latent) if self.decoder_lift is not None else latent.to(torch.float64)
        return self.decoder.apply(source)

    @property
    def parameters(self) -> int:
        total = int(self.encoder.weight.numel() + self.encoder.bias.numel())
        total += int(self.decoder.weight.numel() + self.decoder.bias.numel())
        if self.encoder_lift is not None:
            total += self.encoder_lift.parameters
        if self.decoder_lift is not None:
            total += self.decoder_lift.parameters
        return total

    def state(self) -> dict[str, torch.Tensor]:
        block: dict[str, torch.Tensor] = {
            "encoder.weight": self.encoder.weight.to(torch.float32),
            "encoder.bias": self.encoder.bias.to(torch.float32),
            "decoder.weight": self.decoder.weight.to(torch.float32),
            "decoder.bias": self.decoder.bias.to(torch.float32),
        }
        if self.encoder_lift is not None:
            block["encoder_lift.projection"] = self.encoder_lift.projection.to(torch.float32)
            block["encoder_lift.offset"] = self.encoder_lift.offset.to(torch.float32)
        if self.decoder_lift is not None:
            block["decoder_lift.projection"] = self.decoder_lift.projection.to(torch.float32)
            block["decoder_lift.offset"] = self.decoder_lift.offset.to(torch.float32)
        return block


def fit_absorption(
    pairs: PairSet,
    *,
    rung: str = "linear",
    lift_features: int = 512,
    lift_kind: str = "rff",
    lift_gamma: float = 1.0,
    ridge_rel: float | None = None,
    seed: int = 0,
) -> AbsorptionMaps:
    """Fit ``E`` and ``D`` on the fit split; score on the held-out clips.

    ``rung`` is ``linear`` (ridge on the raw row as given, context columns included) or
    ``nonlinear`` (a stored random-feature lift ahead of the same ridge).
    """

    fit_mask = pairs.split.eq(0)
    hold_mask = pairs.split.eq(1)
    if int(fit_mask.sum()) == 0:
        raise ValueError("pair set has no fit rows")

    raw_fit = pairs.raw[fit_mask].to(torch.float64)
    latent_fit = pairs.latent[fit_mask].to(torch.float64)
    centre_fit = pairs.centre()[fit_mask].to(torch.float64)

    encoder_lift: RandomFeatureLift | None = None
    decoder_lift: RandomFeatureLift | None = None
    if rung == "nonlinear":
        encoder_lift, encoder = fit_kernel_ridge_lift(
            raw_fit, latent_fit, features=lift_features, kind=lift_kind, gamma=lift_gamma, seed=seed
        )
        decoder_lift, decoder = fit_kernel_ridge_lift(
            latent_fit, centre_fit, features=lift_features, kind=lift_kind, gamma=lift_gamma, seed=seed + 1
        )
    elif rung == "linear":
        if ridge_rel is None:
            encoder = fit_ridge_lambda_ladder(raw_fit, latent_fit, seed=seed)
            decoder = fit_ridge_lambda_ladder(latent_fit, centre_fit, seed=seed)
        else:
            encoder = fit_ridge_linear(raw_fit, latent_fit, ridge_rel=ridge_rel, seed=seed)
            decoder = fit_ridge_linear(latent_fit, centre_fit, ridge_rel=ridge_rel, seed=seed)
    else:
        raise ValueError(f"unknown rung {rung!r}")

    maps = AbsorptionMaps(
        encoder=encoder,
        decoder=decoder,
        encoder_lift=encoder_lift,
        decoder_lift=decoder_lift,
        rung=rung,
        geometry=dict(pairs.geometry),
    )
    maps.metrics = score_absorption(maps, pairs, split=1 if int(hold_mask.sum()) else 0)
    maps.metrics["fit_split"] = score_absorption(maps, pairs, split=0)
    return maps


def score_absorption(maps: AbsorptionMaps, pairs: PairSet, *, split: int = 1) -> dict[str, Any]:
    """Score E, D and the composed round-trip against the parent's own ceiling."""

    mask = pairs.split.eq(int(split))
    if int(mask.sum()) == 0:
        return {"rows": 0}
    raw = pairs.raw[mask].to(torch.float64)
    latent = pairs.latent[mask].to(torch.float64)
    centre = pairs.centre()[mask].to(torch.float64)
    parent = pairs.roundtrip[mask].to(torch.float64)

    latent_hat = maps.encode_rows(raw)
    reconstruction = maps.decode_rows(latent_hat)
    decoder_only = maps.decode_rows(latent)

    modality = str(pairs.geometry.get("modality", "audio"))
    data_range = 2.0 if modality == "audio" else 1.0
    ceiling_db = psnr(parent, centre, data_range=data_range)
    absorbed_db = psnr(reconstruction, centre, data_range=data_range)
    body: dict[str, Any] = {
        "rows": int(mask.sum().item()),
        "split": int(split),
        "encoder_scores": write_scores(latent_hat, latent),
        "decoder_scores": write_scores(decoder_only, centre),
        "composed_scores": write_scores(reconstruction, centre),
        "psnr_absorbed_db": absorbed_db,
        "psnr_parent_roundtrip_db": ceiling_db,
        "psnr_gap_db": absorbed_db - ceiling_db,
        "proposed_margin_db": PROPOSED_ABSORPTION_MARGIN_DB,
        "meets_proposed_absorption_line": bool(absorbed_db >= ceiling_db - PROPOSED_ABSORPTION_MARGIN_DB),
        "psnr_rows": "all rows (including silent ones)",
    }
    # A pairing that maps several raw rows onto one latent target imposes a floor no map
    # can beat. When the sampler measured one, it travels with every score here so a
    # reader never compares a fitted number against a ceiling the pairing already excluded.
    if pairs.geometry.get("ambiguous_pairing"):
        body["pairing_ambiguity"] = pairs.geometry.get("ambiguity")
        body["pairing_ambiguity_warning"] = (
            "rows were paired many-to-one onto identical latents; the fitted map cannot beat "
            "geometry['ambiguity']['best_possible_psnr_db'], so read psnr_gap_db against THAT "
            "and not against the parent round-trip"
        )
    if modality == "audio":
        absorbed_sdr = si_sdr_detail(reconstruction, centre)
        parent_sdr = si_sdr_detail(parent, centre)
        body["si_sdr_absorbed_db"] = absorbed_sdr["si_sdr_db"]
        body["si_sdr_parent_roundtrip_db"] = parent_sdr["si_sdr_db"]
        body["si_sdr_rows"] = {
            "scored": absorbed_sdr["scored_rows"],
            "silent_skipped": absorbed_sdr["silent_rows"],
            "note": (
                "SI-SDR excludes silent target rows (they have no scale to be invariant to) "
                "while PSNR above is scored over ALL rows, so the two columns of this table "
                "cover different row sets"
            ),
        }
    # The PCA bound on ANY linear map of this latent width, fitted on the fit split and
    # scored on these rows: it separates "the ridge fit is bad" from "no linear map of
    # this width can reach the parent's nonlinear round-trip".
    fit_centre = pairs.centre()[pairs.split.eq(0)].to(torch.float64)
    if int(fit_centre.shape[0]) > 1:
        bound = best_linear_rank_psnr(fit_centre, centre, int(latent.shape[1]), data_range=data_range)
        if bound["psnr_db"] > ceiling_db:
            body["absorption_line_not_informative"] = (
                f"the parent's own round-trip on this corpus ({ceiling_db:.2f} dB) is "
                f"{bound['psnr_db'] - ceiling_db:.2f} dB BELOW the best linear map of its own "
                "latent width, so meets_proposed_absorption_line only says the ceiling is low "
                "on this corpus; re-run the band on content the parent reconstructs well"
            )
        body["linear_rank_bound"] = {
            "rank": bound["rank"],
            "psnr_db": bound["psnr_db"],
            "retained_energy": bound["retained_energy"],
            "absorbed_vs_bound_db": absorbed_db - bound["psnr_db"],
            "bound_vs_parent_db": bound["psnr_db"] - ceiling_db,
            # When the parent's own round-trip on THIS corpus sits below the best linear
            # map of the parent's own latent width, "absorbed beats the ceiling" says the
            # ceiling is bad, not that absorption works. The flag rides next to
            # meets_proposed_absorption_line so the caveat cannot be lost in prose.
            "ceiling_below_linear_bound": bool(bound["psnr_db"] > ceiling_db),
            "meaning": (
                "PCA of the raw rows at the parent's latent width -- the best possible linear "
                "round-trip. A fitted map at this number is at the linear ceiling; a bound far "
                "below the parent's round-trip means the band needs a nonlinear decoder "
                "(companion VAE decoder, D5b) or a wider native feature width."
            ),
        }
    return body


# --------------------------------------------------------------------------------------
# composition into Chimera shapes
# --------------------------------------------------------------------------------------


def identity_source_projection(latent_width: int, model_width: int) -> torch.Tensor:
    """Stand-in ``P_src`` [model_width, latent_width] when the parent's patch projection
    is not being used: a zero-padded identity. Explicit, so a receipt can say so."""

    matrix = torch.zeros(int(model_width), int(latent_width), dtype=torch.float64)
    span = min(int(model_width), int(latent_width))
    matrix[:span, :span] = torch.eye(span, dtype=torch.float64)
    return matrix


def chart_from_rows(rows: torch.Tensor, hidden: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """``B_in`` [hidden, model_width] from pooled parent rows (SVD chart, C17 control)."""

    chart, notes = svd_chart(rows, hidden)
    if chart.shape[0] < int(hidden):
        pad = torch.zeros(int(hidden) - chart.shape[0], chart.shape[1], dtype=torch.float64)
        chart = torch.cat((chart, pad), dim=0)
        notes["zero_padded_rows"] = int(hidden) - int(notes["rank"])
    return chart, notes


@dataclass
class ComposedInterface:
    feature_proj_weight: torch.Tensor  # [hidden, d_raw]
    type_embed_delta: torch.Tensor  # [hidden]
    emission_out_weight: torch.Tensor  # [d_centre, hidden]
    emission_shift: torch.Tensor  # [d_centre] -- D's bias, lands in the head type affine
    notes: dict[str, Any] = field(default_factory=dict)

    def state(self) -> dict[str, torch.Tensor]:
        return {
            "bridge.feature_proj.weight": self.feature_proj_weight.to(torch.float32),
            "bridge.type_embed_delta": self.type_embed_delta.to(torch.float32),
            "head.out.weight": self.emission_out_weight.to(torch.float32),
            "head.type_affine_shift": self.emission_shift.to(torch.float32),
        }


def compose_interface(
    maps: AbsorptionMaps,
    *,
    hidden: int,
    source_projection: torch.Tensor | None = None,
    output_projection: torch.Tensor | None = None,
    chart_in: torch.Tensor | None = None,
    chart_out: torch.Tensor | None = None,
) -> ComposedInterface:
    """Compose ``E`` / ``D`` with the parent projections and the C17 charts.

    ``source_projection``  P_src  [model_width, d_latent]
    ``output_projection``  P_out  [d_latent, model_width]
    ``chart_in``           B_in   [hidden, model_width]
    ``chart_out``          B_out  [model_width, hidden]

    Any omitted piece defaults to the zero-padded identity so the composition is
    runnable end to end and the receipt says exactly which stand-ins were used.
    """

    if maps.encoder_lift is not None:
        raise ValueError(
            "compose_interface takes a LINEAR absorption map; a random-feature lift is not "
            "a linear boundary projection. Fit the 'linear' rung for composition, or install "
            "the lift as an explicit extra module and count its parameters."
        )
    encoder_weight = maps.encoder.weight.to(torch.float64)  # [d_latent, d_raw]
    encoder_bias = maps.encoder.bias.to(torch.float64)  # [d_latent]
    decoder_weight = maps.decoder.weight.to(torch.float64)  # [d_centre, d_latent]
    decoder_bias = maps.decoder.bias.to(torch.float64)  # [d_centre]
    latent_width = int(encoder_weight.shape[0])

    stand_ins: list[str] = []
    if source_projection is None:
        source_projection = identity_source_projection(latent_width, latent_width)
        stand_ins.append("source_projection=identity")
    if chart_in is None:
        chart_in = identity_source_projection(int(source_projection.shape[0]), int(hidden))
        stand_ins.append("chart_in=identity")
    if output_projection is None:
        # P_out is [latent_width, model_width]; identity_source_projection(a, b) already
        # returns [b, a], so (model_width, latent_width) is the right call and the
        # transpose that used to be here produced [model_width, latent_width] -- rejected
        # by the shape check below the moment a NON-square P_src was supplied. It only
        # ever passed because every delivered call left source_projection None (square).
        # Same defect class as the chart_out stand-in ten lines down; this branch survived
        # that fix.
        output_projection = identity_source_projection(int(source_projection.shape[0]), latent_width)
        stand_ins.append("output_projection=identity")
    if chart_out is None:
        # B_out is [model_width, hidden]; identity_source_projection(a, b) returns [b, a],
        # so the arguments are (hidden, model_width). Reversing them built a [hidden,
        # model_width] stand-in that the shape check below then rejected -- the default
        # stand-in path was unusable until the ladder driver hit it.
        chart_out = identity_source_projection(int(hidden), int(output_projection.shape[1]))
        stand_ins.append("chart_out=identity")

    source_projection = source_projection.to(torch.float64)
    output_projection = output_projection.to(torch.float64)
    chart_in = chart_in.to(torch.float64)
    chart_out = chart_out.to(torch.float64)

    if source_projection.shape[1] != latent_width:
        raise ValueError(f"P_src columns {source_projection.shape[1]} != latent width {latent_width}")
    if chart_in.shape != (int(hidden), int(source_projection.shape[0])):
        raise ValueError(f"B_in must be [hidden, model_width]; got {tuple(chart_in.shape)}")
    if output_projection.shape[0] != latent_width:
        raise ValueError(f"P_out rows {output_projection.shape[0]} != latent width {latent_width}")
    if chart_out.shape != (int(output_projection.shape[1]), int(hidden)):
        raise ValueError(f"B_out must be [model_width, hidden]; got {tuple(chart_out.shape)}")

    feature_proj = chart_in @ source_projection @ encoder_weight
    type_delta = chart_in @ source_projection @ encoder_bias
    emission_out = decoder_weight @ output_projection @ chart_out

    return ComposedInterface(
        feature_proj_weight=feature_proj,
        type_embed_delta=type_delta,
        emission_out_weight=emission_out,
        emission_shift=decoder_bias,
        notes={
            "hidden": int(hidden),
            "d_raw": int(encoder_weight.shape[1]),
            "d_latent": latent_width,
            "d_centre": int(decoder_weight.shape[0]),
            "model_width_in": int(source_projection.shape[0]),
            "model_width_out": int(output_projection.shape[1]),
            "stand_ins": stand_ins,
            "shapes": {
                "bridge.feature_proj.weight": list(feature_proj.shape),
                "bridge.type_embed_delta": list(type_delta.shape),
                "head.out.weight": list(emission_out.shape),
                "head.type_affine_shift": list(decoder_bias.shape),
            },
            "parameters": int(
                feature_proj.numel() + type_delta.numel() + emission_out.numel() + decoder_bias.numel()
            ),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C15 absorption map fitter")
    parser.add_argument("--pairs", required=True, help="directory written by mediafold.c15_pairs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rung", default="linear", choices=("linear", "nonlinear"))
    parser.add_argument("--lift-features", type=int, default=512)
    parser.add_argument("--lift-kind", default="rff", choices=("rff", "situ"))
    parser.add_argument("--hidden", type=int, default=0, help="compose into a Chimera trunk of this width (0 = skip)")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)

    pairs = load_pair_set(arguments.pairs)
    maps = fit_absorption(
        pairs,
        rung=arguments.rung,
        lift_features=arguments.lift_features,
        lift_kind=arguments.lift_kind,
        seed=arguments.seed,
    )

    target = Path(arguments.out)
    target.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "kind": "c15_absorption",
        "pairs_dir": str(Path(arguments.pairs).resolve()),
        "geometry": maps.geometry,
        "rung": maps.rung,
        "map_parameters": maps.parameters,
        "metrics": maps.metrics,
        "maps": receipts.write_tensors(maps.state(), target / "maps.safetensors"),
        "runtime": receipts.runtime_block(),
        "argv": vars(arguments),
    }
    if arguments.hidden > 0 and maps.rung == "linear":
        composed = compose_interface(maps, hidden=int(arguments.hidden))
        body["composition"] = {
            **composed.notes,
            "tensors": receipts.write_tensors(composed.state(), target / "composed_interface.safetensors"),
        }
    receipt_hash = receipts.write_receipt(target / "absorption_receipt.json", body)  # the receipt cannot contain its own hash; it is printed instead
    print(json.dumps({"rung": maps.rung, "metrics": maps.metrics, "parameters": maps.parameters}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
