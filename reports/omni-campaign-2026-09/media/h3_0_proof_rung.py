"""H3-0: the C15 rung on Chimera's 3.19M joint-AV proof trunk.

What it does, per arm, without editing a single line of the proof:

1. builds the proof's fixture corpus with the proof's own ``make_batch`` geometry
   (8 frames @ 8 fps, 32x32, patch 16, tubelet 2; 12288 stereo samples @ 32 kHz,
   segment 1536; ``native_feature_dim`` 1536),
2. samples C15 (raw patch <-> latent) pairs for BOTH bands through VAE adapters whose
   hop / patch / tubelet match that geometry exactly,
3. fits the shared closed-form maps ``E`` (raw -> latent) and ``D`` (latent -> raw) and
   composes them into the trunk's ``native_media_bridge.feature_proj`` and
   ``native_media_head.out``,
4. runs the proof's own before/after evaluation -- ``evaluate_understanding`` and
   ``evaluate_generation`` -- trains, re-evaluates, and reports the proof's 13 gates.

Arms
----
``raw``       the proof's own initialisation. The baseline.
``absorbed``  ``feature_proj`` and ``out`` initialised from the FITTED C15 maps.
``control``   the identical composition pipeline with ``E`` / ``D`` replaced by
              norm-matched random matrices. This is the arm that separates "the fitted
              VAE map carries something" from "this initialisation scale happens to
              help" -- without it an ``absorbed`` win means nothing.

Proposed decision line D-f1 (for Cyril, not a gate this script enforces): absorption
"carries" if ``absorbed`` passes at least as many gates as ``raw`` at equal steps and
its clips are not blind-scored worse.

Honest scope: the default adapters are the harness's deterministic synthetic VAEs, so
this rung tests the MACHINERY (pair geometry, fp64 fits, exact composition algebra,
receipts, arm ordering) end to end. Pointing ``--audio-adapter comfy:<path>`` at the
real ACE / H3 audio VAE swaps in the parent's latents with no other change -- and that
must run inside the ComfyUI image. No H3 generation happens here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mediafold import receipts  # noqa: E402
from mediafold.c15_fit import compose_interface, fit_absorption  # noqa: E402
from mediafold.c15_pairs import PairSet, sample_audio_pairs, sample_video_pairs  # noqa: E402
from mediafold.vae_adapters import SyntheticAudioVae, SyntheticVideoVae, build_adapter  # noqa: E402

ARMS = ("raw", "absorbed", "control")


def import_proof(chimera_src: str):
    """Import the proof module read-only from a chimera-omni checkout."""

    root = Path(chimera_src).resolve()
    package = root / "src" if (root / "src" / "chimera_omni").is_dir() else root
    if not (package / "chimera_omni").is_dir():
        raise FileNotFoundError(f"no chimera_omni package under {root}")
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    from chimera_omni.proof import h3_hybrid_av_flow as proof

    return proof, package


def orthonormal_chart(hidden: int, width: int, *, seed: int = 0) -> torch.Tensor:
    """[hidden, width] with orthonormal columns, so a narrow latent spreads across the
    trunk instead of living in the first ``width`` coordinates."""

    generator = torch.Generator().manual_seed(int(seed))
    matrix = torch.randn(int(hidden), int(width), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix)
    return q[:, : int(width)].contiguous()


def _norm_matched_random(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    draw = torch.randn(reference.shape, generator=generator, dtype=torch.float64)
    scale = reference.norm() / draw.norm().clamp_min(1.0e-30)
    return draw * scale


def build_fixture_pairs(
    proof,
    *,
    clips: int,
    audio_adapter_spec: str,
    video_adapter_spec: str,
    latent_width: int,
    seed: int,
) -> tuple[PairSet, dict[str, Any]]:
    """Sample C15 pairs for BOTH proof bands and pool them into one shared fit.

    Pooling is not a convenience: Chimera's boundary is ONE ``feature_proj`` shared by
    every media type (type identity rides on ``type_embed`` / ``type_feature_scale``),
    so the map that initialises it has to be fitted on the pooled rows. Latents of
    different widths are zero-padded to a common ``latent_width``.
    """

    generator = torch.Generator().manual_seed(int(seed))
    videos: list[torch.Tensor] = []
    audios: list[torch.Tensor] = []
    for _ in range(int(clips)):
        direction = int(torch.randint(0, 2, (1,), generator=generator).item())
        pitch = int(torch.randint(0, 2, (1,), generator=generator).item())
        onset = int(torch.randint(0, 2, (1,), generator=generator).item())
        video, audio = proof.make_clip(direction, pitch, onset, generator)
        videos.append(video)
        audios.append(audio)
    video_corpus = torch.stack(videos)
    audio_corpus = torch.stack(audios)

    if audio_adapter_spec == "synthetic":
        audio_adapter = SyntheticAudioVae(
            hop_samples=proof.SEGMENT,
            channels=proof.CHANNELS,
            latent_channels=min(latent_width, proof.SEGMENT),
            sample_rate_hz=proof.SAMPLE_RATE,
            name="synthetic_audio_proof",
        )
    else:
        audio_adapter = build_adapter(audio_adapter_spec)
    if video_adapter_spec == "synthetic":
        video_adapter = SyntheticVideoVae(
            patch=proof.PATCH,
            tubelet=proof.TUBELET,
            channels=3,
            latent_channels=min(latent_width, proof.FEATURE_DIM),
            name="synthetic_video_proof",
        )
    else:
        video_adapter = build_adapter(video_adapter_spec)

    # `per_channel` is REQUIRED here: the proof's `waveform_to_native_segments` emits one
    # token per channel per segment, so a row must be one channel's `SEGMENT` samples or
    # `d_centre` stops matching `proof.FEATURE_DIM`. On a VAE whose latent is joint over
    # channels that pairing is many-to-one, so it is taken deliberately and the measured
    # ambiguity floor is carried in the geometry and into the receipt.
    audio_pairs = sample_audio_pairs(
        audio_adapter,
        audio_corpus,
        pairing="per_channel",
        seed=seed,
        allow_ambiguous_pairing=True,
    )
    video_pairs = sample_video_pairs(video_adapter, video_corpus, latent_group=1, seed=seed)

    if int(audio_pairs.geometry["d_centre"]) != proof.FEATURE_DIM:
        raise RuntimeError(
            f"audio row width {audio_pairs.geometry['d_centre']} != proof FEATURE_DIM {proof.FEATURE_DIM}"
        )
    if int(video_pairs.geometry["d_centre"]) != proof.FEATURE_DIM:
        raise RuntimeError(
            f"video row width {video_pairs.geometry['d_centre']} != proof FEATURE_DIM {proof.FEATURE_DIM}"
        )

    def pad(latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[1] == latent_width:
            return latent
        if latent.shape[1] > latent_width:
            raise ValueError("latent is wider than the pooled latent width")
        return torch.nn.functional.pad(latent, (0, latent_width - latent.shape[1]))

    pooled = PairSet(
        raw=torch.cat((video_pairs.raw, audio_pairs.raw), dim=0),
        latent=torch.cat((pad(video_pairs.latent), pad(audio_pairs.latent)), dim=0),
        roundtrip=torch.cat((video_pairs.roundtrip, audio_pairs.roundtrip), dim=0),
        clip_index=torch.cat((video_pairs.clip_index, audio_pairs.clip_index + int(clips))),
        frame_index=torch.cat((video_pairs.frame_index, audio_pairs.frame_index)),
        channel_index=torch.cat((video_pairs.channel_index, audio_pairs.channel_index)),
        split=torch.cat((video_pairs.split, audio_pairs.split)),
        geometry={
            "modality": "joint_av",
            "d_raw": int(video_pairs.raw.shape[1]),
            "d_centre": int(proof.FEATURE_DIM),
            "centre_offset": 0,
            "d_latent": int(latent_width),
            "video": video_pairs.geometry,
            "audio": audio_pairs.geometry,
            "rows": int(video_pairs.raw.shape[0] + audio_pairs.raw.shape[0]),
        },
    )
    type_rows = torch.cat(
        (
            torch.full((video_pairs.raw.shape[0],), 2, dtype=torch.long),
            torch.full((audio_pairs.raw.shape[0],), 3, dtype=torch.long),
        )
    )
    return pooled, {"type_rows": type_rows, "video_geometry": video_pairs.geometry, "audio_geometry": audio_pairs.geometry}


def per_type_encoder_bias(
    pairs: PairSet,
    maps,
    type_rows: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """One ``E`` bias per media type, keeping the shared ``E`` weight.

    Chimera's ``feature_proj`` is shared by every media type, so ``E_w`` must be fitted
    on the pooled rows -- but the BIAS does not have to be, and pooling it is what made
    every type row identical. For a fixed ``E_w`` the least-squares bias restricted to
    the rows of type ``t`` is just the mean residual of those rows, which is closed form
    and averages back to the pooled bias.
    """

    weight = maps.encoder.weight.to(torch.float64)
    raw = pairs.raw.to(torch.float64)
    latent = pairs.latent.to(torch.float64)
    residual = latent - raw @ weight.transpose(0, 1)
    biases: dict[int, torch.Tensor] = {}
    for media_type in sorted({int(value) for value in type_rows.tolist()}):
        mask = type_rows.to(torch.long).eq(int(media_type))
        if not bool(mask.any()):
            continue
        biases[int(media_type)] = residual[mask].mean(dim=0).contiguous()
    return biases


def install_absorbed_interface(
    model,
    maps,
    *,
    type_rows: torch.Tensor,
    latent_width: int,
    scramble: bool,
    seed: int,
    emission_scale: str = "match_default",
    decoder_bias: str = "skip",
    type_bias: dict[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Write the composed C15 interface into the trunk's bridge and emission head.

    ``scramble=True`` is the control arm: identical shapes, identical composition, but
    ``E`` / ``D`` are norm-matched random matrices, so only the fitted CONTENT differs.
    """

    hidden = int(model.native_media_bridge.feature_proj.weight.shape[0])
    encoder_weight = maps.encoder.weight.clone()
    encoder_bias = maps.encoder.bias.clone()
    decoder_weight = maps.decoder.weight.clone()
    decoder_bias_vector = maps.decoder.bias.clone()
    if scramble:
        encoder_weight = _norm_matched_random(encoder_weight, seed + 11)
        encoder_bias = _norm_matched_random(encoder_bias, seed + 12)
        decoder_weight = _norm_matched_random(decoder_weight, seed + 13)
        decoder_bias_vector = _norm_matched_random(decoder_bias_vector, seed + 14)

    scrambled = dataclasses.replace(
        maps,
        encoder=dataclasses.replace(maps.encoder, weight=encoder_weight, bias=encoder_bias),
        decoder=dataclasses.replace(maps.decoder, weight=decoder_weight, bias=decoder_bias_vector),
    )

    chart_in = orthonormal_chart(hidden, latent_width, seed=seed)
    chart_out = chart_in.transpose(0, 1).contiguous()
    composed = compose_interface(
        scrambled,
        hidden=hidden,
        chart_in=chart_in,
        chart_out=chart_out,
    )

    dtype = model.native_media_bridge.feature_proj.weight.dtype
    device = model.native_media_bridge.feature_proj.weight.device

    # The composition is exact linear algebra, so its output SCALE is whatever E and D
    # happen to have -- and D inverts a saturating nonlinearity, so its rows are large.
    # Installed verbatim on the 3.19M proof trunk that produced a media MSE around 7e3
    # (and the norm-matched control did the same, so it was the scale, not the content).
    # `match_default` rescales each composed matrix to the RMS of the initialisation it
    # replaces, which preserves the map's DIRECTIONS and only fixes its gain.
    bridge_gain = 1.0
    head_gain = 1.0
    if emission_scale == "match_default":
        default_bridge = float(model.native_media_bridge.feature_proj.weight.detach().float().pow(2).mean().sqrt().item())
        default_head = float(model.native_media_head.out.weight.detach().float().pow(2).mean().sqrt().item())
        composed_bridge = float(composed.feature_proj_weight.pow(2).mean().sqrt().item())
        composed_head = float(composed.emission_out_weight.pow(2).mean().sqrt().item())
        bridge_gain = default_bridge / max(composed_bridge, 1.0e-30)
        head_gain = default_head / max(composed_head, 1.0e-30)
    elif emission_scale != "none":
        raise ValueError(f"unknown emission_scale {emission_scale!r}")

    with torch.no_grad():
        model.native_media_bridge.feature_proj.weight.copy_(
            (composed.feature_proj_weight * bridge_gain).to(dtype=dtype, device=device)
        )
        model.native_media_head.out.weight.copy_(
            (composed.emission_out_weight * head_gain).to(dtype=dtype, device=device)
        )
        composed.type_embed_delta = composed.type_embed_delta * bridge_gain
        composed.emission_shift = composed.emission_shift * head_gain
        # E's bias belongs on the per-type embedding rows the pairs actually covered.
        # It is a DELTA (c15_fit: `type_embed[t] += B_in @ P_src @ E_b`), so it is ADDED.
        # `copy_` here overwrote the trunk's learned type identity with one shared vector,
        # which left video (type 2) and audio (type 3) byte-identical after install -- the
        # bridge could no longer tell an audio token from a video token, in BOTH the
        # absorbed and the control arm. Per-type biases keep the rows distinct; the pooled
        # bias is only the fallback when the caller has not fitted them.
        type_deltas: dict[int, dict[str, float]] = {}
        for media_type in sorted({int(value) for value in type_rows.tolist()}):
            bias_vector = None if type_bias is None else type_bias.get(int(media_type))
            if bias_vector is None:
                delta = composed.type_embed_delta
            else:
                vector = bias_vector.to(torch.float64)
                if scramble:
                    vector = _norm_matched_random(vector, seed + 31 + int(media_type))
                delta = (
                    compose_interface(
                        dataclasses.replace(
                            scrambled,
                            encoder=dataclasses.replace(scrambled.encoder, bias=vector),
                        ),
                        hidden=hidden,
                        chart_in=chart_in,
                        chart_out=chart_out,
                    ).type_embed_delta
                    * bridge_gain
                )
            row = model.native_media_bridge.type_embed.weight[media_type]
            before_norm = float(row.detach().float().norm().item())
            row.add_(delta.to(dtype=dtype, device=device))
            type_deltas[int(media_type)] = {
                "delta_norm": float(delta.norm().item()),
                "row_norm_before": before_norm,
                "row_norm_after": float(row.detach().float().norm().item()),
                "per_type_bias": bias_vector is not None,
            }
        # D's bias is a RAW-space shift, but the head applies its per-type affine BEFORE
        # `out`, in hidden space. The nearest hidden-space shift is the least-squares
        # pre-image -- and installing it is what actually destroyed generation in the
        # first two H3-0 runs: the pre-image is large and unregularised, so the head
        # emits a huge constant (media MSE ~1e4) whatever the weight gain is. Rescaling
        # the weights did not fix it (run 2, gains 4.15 / 0.44, MSE still ~1.1e4).
        # `skip` therefore leaves the bias UNABSORBED and records how much was dropped.
        out_weight = model.native_media_head.out.weight.detach().to(torch.float64).cpu()
        solution = torch.linalg.lstsq(out_weight, composed.emission_shift.reshape(-1, 1)).solution.flatten()
        residual = float((out_weight @ solution - composed.emission_shift).norm().item())
        shift_norm = float(composed.emission_shift.norm().item())
        if decoder_bias == "preimage":
            affine = model.native_media_head.type_affine.weight
            for media_type in sorted({int(value) for value in type_rows.tolist()}):
                affine[media_type, hidden:].copy_(solution.to(dtype=affine.dtype, device=affine.device))
        elif decoder_bias != "skip":
            raise ValueError(f"unknown decoder_bias {decoder_bias!r}")

    installed_type_rows = model.native_media_bridge.type_embed.weight.detach().float()
    distinct_type_rows = int(torch.unique(installed_type_rows, dim=0).shape[0])
    return {
        **composed.notes,
        "scrambled": bool(scramble),
        "chart": "orthonormal_columns",
        "type_embed_write": "add (delta, per c15_fit composition)",
        "type_embed_deltas": type_deltas,
        "type_embed_rows": int(installed_type_rows.shape[0]),
        "type_embed_distinct_rows_after_install": distinct_type_rows,
        "emission_scale": emission_scale,
        "decoder_bias": decoder_bias,
        "decoder_bias_preimage_norm": float(solution.norm().item()),
        "bridge_gain": bridge_gain,
        "head_gain": head_gain,
        "decoder_bias_preimage_residual": residual,
        "decoder_bias_norm": shift_norm,
        "decoder_bias_relative_residual": residual / max(shift_norm, 1.0e-30),
        "installed_state_sha256": receipts.hash_state(
            {
                "bridge.feature_proj.weight": model.native_media_bridge.feature_proj.weight,
                "head.out.weight": model.native_media_head.out.weight,
                "bridge.type_embed.weight": model.native_media_bridge.type_embed.weight,
                "head.type_affine.weight": model.native_media_head.type_affine.weight,
            }
        ),
    }


def run_arm(
    proof,
    arm: str,
    *,
    maps,
    type_rows: torch.Tensor,
    latent_width: int,
    device: torch.device,
    steps: int,
    batch_size: int,
    generation_share: int,
    sample_steps: int,
    eval_count: int,
    seed: int,
    output_dir: Path,
    save_artifacts: bool,
    emission_scale: str = "match_default",
    decoder_bias: str = "skip",
    type_bias: dict[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = proof.build_model(device)
    parameters = sum(int(p.numel()) for p in model.parameters())

    interface: dict[str, Any] | None = None
    if arm in {"absorbed", "control"}:
        interface = install_absorbed_interface(
            model,
            maps,
            type_rows=type_rows,
            latent_width=latent_width,
            scramble=(arm == "control"),
            seed=seed,
            emission_scale=emission_scale,
            decoder_bias=decoder_bias,
            type_bias=type_bias,
        )

    state_before = proof.hash_state_dict(model)
    model.eval()
    before_understanding = proof.evaluate_understanding(model, device, seed=seed + 1, count=eval_count)
    before_generation, before_video, before_audio, schedule = proof.evaluate_generation(
        model, device, seed=seed + 2, count=eval_count, sample_steps=sample_steps
    )

    model.train()
    started = time.monotonic()
    training_log, training_schedule = proof.train(
        model,
        device,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        generation_share=generation_share,
    )
    train_seconds = time.monotonic() - started
    state_after = proof.hash_state_dict(model)

    model.eval()
    after_understanding = proof.evaluate_understanding(model, device, seed=seed + 1, count=eval_count)
    after_generation, after_video, after_audio, after_schedule = proof.evaluate_generation(
        model, device, seed=seed + 2, count=eval_count, sample_steps=sample_steps
    )

    gates = {
        "understanding_accuracy_ge_0_90": after_understanding >= 0.90,
        "direction_accuracy_ge_0_85": after_generation["direction_accuracy"] >= 0.85,
        "pitch_accuracy_ge_0_85": after_generation["pitch_accuracy"] >= 0.85,
        "sync_within_015_ge_0_75": after_generation["sync_within_015"] >= 0.75,
        "understanding_improved": after_understanding > before_understanding,
        "direction_improved": after_generation["direction_accuracy"] > before_generation["direction_accuracy"],
        "pitch_improved": after_generation["pitch_accuracy"] > before_generation["pitch_accuracy"],
        "sync_improved": after_generation["sync_within_015"] > before_generation["sync_within_015"],
        "video_mse_improved": after_generation["video_mse"] < before_generation["video_mse"],
        "audio_mse_improved": after_generation["audio_mse"] < before_generation["audio_mse"],
        "state_changed": state_before != state_after,
        "exact_refinement_observed": bool(after_schedule["refinement_observed"]),
        "separate_schedule_update_mask_exact": bool(training_schedule["update_mask_exact"]),
    }

    artifacts: dict[str, Any] = {}
    if save_artifacts:
        arm_dir = output_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "before": proof.save_artifacts(before_video, before_audio, output_dir=arm_dir, tag="before", count=2),
            "after": proof.save_artifacts(after_video, after_audio, output_dir=arm_dir, tag="after", count=2),
        }

    return {
        "arm": arm,
        "parameters": parameters,
        "interface": interface,
        "state_sha256": {"before": state_before, "after": state_after},
        "train_seconds": round(train_seconds, 3),
        "before": {"understanding_accuracy": before_understanding, "generation": before_generation},
        "after": {"understanding_accuracy": after_understanding, "generation": after_generation},
        "gates": gates,
        "gates_passed": int(sum(1 for value in gates.values() if value)),
        "gates_total": len(gates),
        "passed": all(gates.values()),
        "training_log": training_log,
        "sampling_schedule": {"steps": schedule["steps"], "video_shift": schedule["video_shift"], "audio_shift": schedule["audio_shift"]},
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H3-0 C15 rung on the 3.19M joint-AV proof trunk")
    parser.add_argument("--chimera-src", required=True, help="path to a chimera-omni checkout")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default="raw,absorbed,control")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generation-share", type=int, default=3)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--eval-count", type=int, default=32)
    parser.add_argument("--pair-clips", type=int, default=64)
    parser.add_argument("--latent-width", type=int, default=32)
    parser.add_argument("--audio-adapter", default="synthetic")
    parser.add_argument("--video-adapter", default="synthetic")
    parser.add_argument("--rung", default="linear", choices=("linear", "nonlinear"))
    parser.add_argument(
        "--emission-scale",
        default="match_default",
        choices=("match_default", "none"),
        help="'none' installs the composition verbatim (measured: media MSE ~7e3 on the proof trunk)",
    )
    parser.add_argument(
        "--decoder-bias",
        default="skip",
        choices=("skip", "preimage"),
        help="'preimage' installs D's bias as a hidden-space pre-image in the head affine "
             "(measured: destroys generation, media MSE ~1e4); 'skip' leaves it unabsorbed",
    )
    parser.add_argument("--seed", type=int, default=2026090110)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-artifacts", action="store_true")
    arguments = parser.parse_args(argv)

    proof, package = import_proof(arguments.chimera_src)
    if arguments.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(arguments.device)

    output_dir = Path(arguments.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs, extra = build_fixture_pairs(
        proof,
        clips=arguments.pair_clips,
        audio_adapter_spec=arguments.audio_adapter,
        video_adapter_spec=arguments.video_adapter,
        latent_width=arguments.latent_width,
        seed=arguments.seed,
    )
    maps = fit_absorption(pairs, rung=arguments.rung, seed=arguments.seed)
    type_bias = per_type_encoder_bias(pairs, maps, extra["type_rows"])
    pair_receipt = receipts.write_tensors(pairs.state(), output_dir / "fixture_pairs.safetensors")
    map_receipt = receipts.write_tensors(maps.state(), output_dir / "fixture_maps.safetensors")

    arm_results = []
    for arm in [value.strip() for value in arguments.arms.split(",") if value.strip()]:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; expected a subset of {ARMS}")
        arm_results.append(
            run_arm(
                proof,
                arm,
                maps=maps,
                type_rows=extra["type_rows"],
                latent_width=arguments.latent_width,
                device=device,
                steps=arguments.steps,
                batch_size=arguments.batch_size,
                generation_share=arguments.generation_share,
                sample_steps=arguments.sample_steps,
                eval_count=arguments.eval_count,
                seed=arguments.seed,
                output_dir=output_dir,
                save_artifacts=arguments.save_artifacts,
                emission_scale=arguments.emission_scale,
                decoder_bias=arguments.decoder_bias,
                type_bias=type_bias,
            )
        )

    summary = {
        arm["arm"]: {
            "gates_passed": arm["gates_passed"],
            "understanding_after": arm["after"]["understanding_accuracy"],
            "video_mse_after": arm["after"]["generation"]["video_mse"],
            "audio_mse_after": arm["after"]["generation"]["audio_mse"],
        }
        for arm in arm_results
    }
    by_arm = {arm["arm"]: arm["gates_passed"] for arm in arm_results}
    body = {
        "kind": "h3_0_c15_proof_rung",
        "chimera_package": str(package),
        "proof_source_sha256": receipts.source_sha256(
            [str(package / "chimera_omni" / "proof" / "h3_hybrid_av_flow.py")]
        ),
        "geometry": {
            "frames": proof.FRAMES,
            "fps": proof.FPS,
            "size": proof.SIZE,
            "patch": proof.PATCH,
            "tubelet": proof.TUBELET,
            "segment": proof.SEGMENT,
            "sample_rate_hz": proof.SAMPLE_RATE,
            "feature_dim": proof.FEATURE_DIM,
            "video_tokens": proof.VIDEO_TOKENS,
            "audio_tokens": proof.AUDIO_TOKENS,
        },
        "c15": {
            "pairs": pair_receipt,
            "maps": map_receipt,
            "pair_geometry": pairs.geometry,
            "rung": maps.rung,
            "map_parameters": maps.parameters,
            "metrics": maps.metrics,
        },
        "arms": arm_results,
        "summary": summary,
        "proposed_decision_line_D_f1": {
            "statement": "absorption carries if absorbed >= raw gates at equal steps and its clips are not blind-scored worse",
            "absorbed_ge_raw": (
                by_arm.get("absorbed", -1) >= by_arm.get("raw", 0)
                if "absorbed" in by_arm and "raw" in by_arm
                else None
            ),
            "absorbed_gt_control": (
                by_arm.get("absorbed", -1) > by_arm.get("control", -1)
                if "absorbed" in by_arm and "control" in by_arm
                else None
            ),
            "note": "a proposed decision line for Cyril; blind scoring of the decoded clips is the actual instrument",
        },
        "runtime": receipts.runtime_block(str(device)),
        "argv": vars(arguments),
    }
    receipt_hash = receipts.write_receipt(output_dir / "h3_0_report.json", body)  # the receipt cannot contain its own hash; it is printed instead
    print(json.dumps({"summary": summary, "c15_metrics": maps.metrics.get("psnr_gap_db")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
