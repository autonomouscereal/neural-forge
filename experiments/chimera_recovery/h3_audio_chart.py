"""Explicit 2026-10-13 standalone H3 audio observation chart.

The physical time lattice comes from the actual full-second encoder probe.
Rotary XYZ is a separately amended convention, never parent-layout parity.
Pure Python output lets the native packer create detached FP32 tensors.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

SCHEMA = "chimera_recovery.h3_audio_observation_chart.v1"
POLICY = "audio40hz_prompt_offset_categorical_stereo_v1"
READ_POLICY = "h3_prompt_and_clean_observation_only"
SAMPLE_RATE, SAMPLES, HOP, FRAMES = 32000, 32000, 800, 40


def _sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(label + " requires a lowercase SHA256")
    return value


def validate_encoder_proof(proof, *, require_model_space=True):
    """Check measured scope; the caller separately seals proof bytes and tensors."""
    inp, meas, dec = (proof.get(k, {}) for k in ("input_contract", "measurement", "decode"))
    if (proof.get("schema") != "chimera_recovery.h3_standalone_audio_probe.v1" or
            proof.get("probe_passed") is not True or proof.get("teacher_unchanged") is not True or
            inp.get("target_rate") != SAMPLE_RATE or inp.get("target_samples") != SAMPLES or
            inp.get("duration_seconds") != 1.0 or inp.get("no_input_padding_or_trim") is not True or
            meas.get("raw_latent_shape") != [1, 32, 2, FRAMES] or meas.get("packed_shape") != [80, 32] or
            meas.get("decoded_shape") != [1, 2, SAMPLES] or
            meas.get("measured_samples_per_latent_frame") != HOP or meas.get("nominal_hop_seconds") != HOP / SAMPLE_RATE or
            meas.get("encoder_repeat_bitwise_equal") is not True or
            meas.get("channel_major_pack_bitwise_invertible") is not True or
            any(meas.get(k) != 0 for k in ("latent_padded_frames", "latent_trimmed_frames")) or
            any(dec.get(k) != 0 for k in ("waveform_padded_samples", "waveform_trimmed_samples")) or
            dec.get("first_pts_seconds") != 0.0 or dec.get("last_end_seconds") != 1.0 or
            dec.get("target_samples") != SAMPLES or dec.get("resampler_flushed") is not True):
        raise ValueError("audio chart requires actual full-second, full-coverage stereo encoder evidence")
    intervals = meas.get("nominal_frame_interval_seconds")
    if (not isinstance(intervals, list) or len(intervals) != FRAMES or
            any(not isinstance(pair, list) or len(pair) != 2 or
                any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in pair) or
                abs(pair[0] - i * HOP / SAMPLE_RATE) > 1e-9 or
                abs(pair[1] - (i + 1) * HOP / SAMPLE_RATE) > 1e-9
                for i, pair in enumerate(intervals))):
        raise ValueError("audio chart needs measured nominal intervals covering exactly [0,1]")
    if require_model_space and proof.get("model_space_verified") is not True:
        raise ValueError("raw VAE proof does not establish the parent model-space observation")
    return True


def chart(prefix_text_count, T, full_ids_length, media_indices, text_order_indices, *,
          clean_encoder_receipt_sha256, amendment_sha256, source_pins):
    """Scatter the sealed new convention into the genuine full native sequence.

    Indices are ascending physical sequence slots. Text order includes answers;
    only its first prefix_text_count entries may enter H3 joint reads. Media is
    channel-major (left forty frames, then right forty frames), wholly in prompt.
    The caller must verify encoder bytes/assets and keep this read mask enforced.
    """
    if type(T) is not int or T != FRAMES or type(full_ids_length) is not int or not 1 <= full_ids_length <= 65536:
        raise ValueError("this measured audio chart supports exactly forty full-second frames")
    if (type(prefix_text_count) is not int or prefix_text_count < 1 or
            not isinstance(media_indices, (list, tuple)) or not isinstance(text_order_indices, (list, tuple)) or
            prefix_text_count > len(text_order_indices) or len(media_indices) != 2 * T):
        raise ValueError("audio chart requires prompt text and exactly eighty stereo media slots")
    media, text = list(media_indices), list(text_order_indices)
    for values in (media, text):
        if any(type(i) is not int or not 0 <= i < full_ids_length for i in values) or values != sorted(set(values)):
            raise ValueError("audio chart indices must be unique ascending native sequence indices")
    if sorted(media + text) != list(range(full_ids_length)):
        raise ValueError("audio chart must map every full-sequence slot exactly once")
    answer = text[prefix_text_count:]
    if answer and max(media + text[:prefix_text_count]) >= answer[0]:
        raise ValueError("audio observations and prompt text must precede every reference-answer slot")
    receipt = _sha(clean_encoder_receipt_sha256, "clean encoder receipt")
    amendment = _sha(amendment_sha256, "standalone audio chart amendment")
    required = {"stage_c", "comfy_audio_vae", "comfy_sd", "charted_attention"}
    if not isinstance(source_pins, dict) or not required.issubset(source_pins):
        raise ValueError("audio chart requires authentic encoder and charted attention source pins")
    pins = {name: _sha(value, name) for name, value in sorted(source_pins.items())}
    native, xyz = [], [[0.0, 0.0, 0.0] for _ in range(full_ids_length)]
    visible = [False] * full_ids_length
    for ordinal, index in enumerate(text):
        xyz[index] = [float(ordinal), 0.0, 0.0]
        visible[index] = ordinal < prefix_text_count
    for row, index in enumerate(media):
        channel, frame = divmod(row, T)
        native.append([(frame + 0.5) * HOP / SAMPLE_RATE, 0.0, HOP / SAMPLE_RATE, float(channel)])
        xyz[index] = [float(prefix_text_count + frame), 0.0, float(channel)]
        visible[index] = True
    return {"native_media_positions": [native], "charted_source_positions": [xyz],
            "h3_prompt_read_mask": [visible], "contract": {
                "schema": SCHEMA, "policy": POLICY, "read_policy": READ_POLICY,
                "dated": "2026-10-13", "clean_encoder_receipt_sha256": receipt,
                "amendment_sha256": amendment, "source_pins": pins,
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_coordinate_scope": "amended observation convention; categorical stereo axis",
                "time_scope": "measured nominal encoder lattice, not receptive-field localization",
                "native_parent_xyz_parity": False, "native_admission_granted": False,
                "spec_type_ids": [2, 3], "observation_type_id": 3, "clean_sigma": 0.0,
                "packing": "channel-major", "sample_rate": SAMPLE_RATE, "samples": SAMPLES,
                "latent_frames_per_channel": T, "samples_per_frame": HOP,
                "prefix_text_count": prefix_text_count, "full_ids_length": full_ids_length,
                "media_indices": media, "text_order_indices": text,
                "answer_indices_excluded_from_h3_reads": answer}}
