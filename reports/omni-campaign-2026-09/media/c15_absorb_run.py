"""C15 absorption ladder over one or more REAL parent-VAE pair sets.

Runs in the lab venv (no ComfyUI, no model weights): it consumes pair directories written
by ``c15_vae_pairs_container.py`` / ``c15_video_groups_container.py`` and, for each rung,

1. fits ``E`` (raw -> latent) and ``D`` (latent -> raw) closed form in float64,
2. scores the composed round-trip ``D(E(raw))`` on held-out CLIPS against the parent
   VAE's own round-trip on the same rows -- the ceiling,
3. composes the linear rung into Chimera's boundary-projection / emission-head shapes at
   a given trunk width and prices the band in PARAMETERS,
4. writes one receipt per (pair set, rung) plus a combined ladder summary carrying the
   proposed decision line D-d2 (absorbed iff PSNR >= ceiling - 3 dB) as a PROPOSAL.

Nothing here is a gate. The behaviour gate is Cyril's blind score on decoded media.

    python c15_absorb_run.py \
        --pairs ace_1.5_vae=/home/cereal/omni-campaign/media/pairs/ace_1.5_vae_ctx0 \
        --pairs h3_audio=/home/cereal/omni-campaign/media/pairs/h3_audio_ctx0 \
        --rungs linear,nonlinear --hidden 5120 \
        --out /home/cereal/omni-campaign/media/out/c15_real
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
from mediafold.c15_fit import (  # noqa: E402
    PROPOSED_ABSORPTION_MARGIN_DB,
    compose_interface,
    fit_absorption,
)
from mediafold.c15_pairs import load_pair_set  # noqa: E402


def band_plan(geometry: dict[str, Any], hidden: int) -> dict[str, Any]:
    """What this band costs in Chimera, in PARAMETERS, at trunk width ``hidden``."""

    d_raw = int(geometry["d_raw"])
    d_centre = int(geometry["d_centre"])
    plan: dict[str, Any] = {
        "trunk_hidden": int(hidden),
        "native_feature_dim_required": d_centre,
        "bridge_feature_proj_parameters": int(hidden) * d_raw,
        "emission_head_out_parameters": d_centre * int(hidden),
        "modality": geometry.get("modality"),
    }
    if geometry.get("modality") == "audio":
        plan["chimera_call"] = (
            f"waveform_to_native_segments(segment={geometry.get('chimera_segment')}, "
            f"stride={geometry.get('chimera_stride')}, feature_dim={d_centre}, "
            f"sample_rate_hz={geometry.get('sample_rate_hz')})"
        )
        plan["tokens_per_second"] = geometry.get("frame_rate_hz")
        plan["tokens_per_frame"] = geometry.get("chimera_tokens_per_frame")
    else:
        plan["chimera_call"] = (
            f"video_to_native_patches(patch={geometry.get('chimera_patch')}, "
            f"tubelet={geometry.get('chimera_tubelet')}, feature_dim={d_centre})"
        )
        plan["latent_group"] = geometry.get("latent_group")
    return plan


def run_one(
    name: str,
    pairs_dir: str,
    *,
    rungs: list[str],
    hidden: int,
    lift_features: int,
    lift_kind: str,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    pairs = load_pair_set(pairs_dir)
    record: dict[str, Any] = {
        "name": name,
        "pairs_dir": str(Path(pairs_dir).resolve()),
        "geometry": pairs.geometry,
        "rows": int(pairs.raw.shape[0]),
        "fit_rows": int(pairs.split.eq(0).sum().item()),
        "holdout_rows": int(pairs.split.eq(1).sum().item()),
        "rungs": {},
    }
    for rung in rungs:
        started = time.monotonic()
        maps = fit_absorption(
            pairs,
            rung=rung,
            lift_features=lift_features,
            lift_kind=lift_kind,
            seed=seed,
        )
        elapsed = time.monotonic() - started
        target = out_dir / name / rung
        target.mkdir(parents=True, exist_ok=True)
        body: dict[str, Any] = {
            "kind": "c15_absorption",
            "name": name,
            "pairs_dir": record["pairs_dir"],
            "geometry": maps.geometry,
            "rung": rung,
            "map_parameters": maps.parameters,
            "fit_seconds": round(elapsed, 3),
            "metrics": maps.metrics,
            "maps": receipts.write_tensors(maps.state(), target / "maps.safetensors"),
            "runtime": receipts.runtime_block(),
        }
        if rung == "linear" and int(hidden) > 0:
            composed = compose_interface(maps, hidden=int(hidden))
            body["composition"] = {
                **composed.notes,
                "band_plan": band_plan(maps.geometry, int(hidden)),
                "tensors": receipts.write_tensors(
                    composed.state(), target / "composed_interface.safetensors"
                ),
            }
        receipts.write_receipt(target / "absorption_receipt.json", body)
        record["rungs"][rung] = {
            "map_parameters": maps.parameters,
            "fit_seconds": round(elapsed, 3),
            "holdout": {
                key: maps.metrics.get(key)
                for key in (
                    "rows",
                    "psnr_absorbed_db",
                    "psnr_parent_roundtrip_db",
                    "psnr_gap_db",
                    "si_sdr_absorbed_db",
                    "si_sdr_parent_roundtrip_db",
                    "si_sdr_rows",
                    "linear_rank_bound",
                    "meets_proposed_absorption_line",
                    "absorption_line_not_informative",
                    "pairing_ambiguity",
                    "pairing_ambiguity_warning",
                )
                if key in maps.metrics
            },
            "encoder_holdout_scores": maps.metrics.get("encoder_scores"),
            "decoder_holdout_scores": maps.metrics.get("decoder_scores"),
            "composed_holdout_scores": maps.metrics.get("composed_scores"),
            "receipt": str(target / "absorption_receipt.json"),
        }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C15 absorption ladder over real VAE pair sets")
    parser.add_argument("--pairs", action="append", required=True, help="name=/path/to/pair/dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rungs", default="linear")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--lift-features", type=int, default=512)
    parser.add_argument("--lift-kind", default="rff", choices=("rff", "situ"))
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)

    rungs = [value.strip() for value in arguments.rungs.split(",") if value.strip()]
    out_dir = Path(arguments.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for entry in arguments.pairs:
        if "=" not in entry:
            parser.error(f"--pairs takes name=path, got {entry!r}")
        name, path = entry.split("=", 1)
        records.append(
            run_one(
                name,
                path,
                rungs=rungs,
                hidden=arguments.hidden,
                lift_features=arguments.lift_features,
                lift_kind=arguments.lift_kind,
                seed=arguments.seed,
                out_dir=out_dir,
            )
        )

    ladder = []
    for record in records:
        for rung, block in record["rungs"].items():
            held = block["holdout"]
            bound = held.get("linear_rank_bound") or {}
            ambiguity = held.get("pairing_ambiguity") or {}
            bound_vs_parent = bound.get("bound_vs_parent_db")
            ladder.append(
                {
                    "vae": record["name"],
                    "rung": rung,
                    "d_raw": record["geometry"]["d_raw"],
                    "d_latent": record["geometry"]["d_latent"],
                    "holdout_rows": held.get("rows"),
                    "psnr_absorbed_db": held.get("psnr_absorbed_db"),
                    "psnr_parent_roundtrip_db": held.get("psnr_parent_roundtrip_db"),
                    "psnr_gap_db": held.get("psnr_gap_db"),
                    "si_sdr_absorbed_db": held.get("si_sdr_absorbed_db"),
                    "si_sdr_parent_roundtrip_db": held.get("si_sdr_parent_roundtrip_db"),
                    "encoder_cos": (block.get("encoder_holdout_scores") or {}).get("cos"),
                    "encoder_rel_l2": (block.get("encoder_holdout_scores") or {}).get("rel_l2"),
                    "linear_rank_bound_psnr_db": (held.get("linear_rank_bound") or {}).get("psnr_db"),
                    "absorbed_vs_linear_bound_db": (held.get("linear_rank_bound") or {}).get("absorbed_vs_bound_db"),
                    "map_parameters": block["map_parameters"],
                    "meets_proposed_absorption_line": held.get("meets_proposed_absorption_line"),
                    # the two caveats that must travel WITH the verdict, not in prose:
                    # (1) the parent's ceiling on this corpus is below the best linear map
                    #     of its own width, so "meets the line" says nothing about the fold;
                    # (2) the rows were paired many-to-one, so no map could have done better
                    #     than the floor recorded here.
                    "ceiling_below_linear_bound": bound.get("ceiling_below_linear_bound"),
                    "ceiling_vs_linear_bound_db": (
                        None if bound_vs_parent is None else -float(bound_vs_parent)
                    ),
                    "verdict_caveat": held.get("absorption_line_not_informative"),
                    "ambiguous_pairing": bool(record["geometry"].get("ambiguous_pairing")),
                    "ambiguity_floor_rel_l2": ambiguity.get("irreducible_rel_l2"),
                    "ambiguity_best_possible_psnr_db": ambiguity.get("best_possible_psnr_db"),
                }
            )

    summary = {
        "kind": "c15_absorption_ladder",
        "proposed_decision_line_D_d2": (
            f"absorbed <=> PSNR(D.E) >= PSNR(parent round-trip) - {PROPOSED_ABSORPTION_MARGIN_DB} dB on "
            "held-out clips. This is a PROPOSAL for Cyril, not a gate this harness enforces; "
            "below the line the trunk keeps a companion VAE decoder for that band (D5b)."
        ),
        "ladder": ladder,
        "how_to_read_a_row": (
            "meets_proposed_absorption_line is only meaningful when ceiling_below_linear_bound "
            "is false and ambiguous_pairing is false. ceiling_below_linear_bound true means the "
            "parent's own round-trip on this corpus is worse than the best linear map of its own "
            "latent width, so beating it proves nothing about the fold. ambiguous_pairing true "
            "means several raw rows were mapped onto one latent, so no map could exceed "
            "ambiguity_best_possible_psnr_db."
        ),
        "records": records,
        "source_sha256": receipts.source_sha256(
            [
                str(_HERE / "c15_absorb_run.py"),
                str(_HERE / "mediafold" / "c15_fit.py"),
                str(_HERE / "mediafold" / "c15_pairs.py"),
                str(_HERE / "mediafold" / "c15_video_groups.py"),
                str(_HERE / "mediafold" / "linalg.py"),
            ]
        ),
        "runtime": receipts.runtime_block(),
        "argv": vars(arguments),
        "runs_h3_generation": False,
    }
    receipts.write_receipt(out_dir / "c15_ladder_receipt.json", summary)
    print(json.dumps({"ladder": ladder}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
