"""Unit tests for the media fold harness, on tiny shapes.

Standalone (no pytest dependency): ``python run_tests.py`` runs every ``test_*``
function, prints one line each, and exits non-zero on the first failure count > 0.

    python run_tests.py                       # everything that needs no big weights
    python run_tests.py --ace-weights <path>  # + the real ACE-Step block load/forward
    python run_tests.py --chimera-src <path>  # + the proof-trunk integration tests
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mediafold import receipts
from mediafold.c15_fit import compose_interface, fit_absorption
from mediafold.c15_pairs import load_pair_set, sample_audio_pairs, sample_video_pairs, write_pair_set
from mediafold.linalg import (
    compression_receipt,
    fit_ridge_linear,
    law_abiding_solve,
    orthogonal_procrustes,
    pivoted_columns,
    prior_anchored,
    psnr,
    si_sdr,
    situ_activation,
    svd_chart,
    write_scores,
)
from mediafold.vae_adapters import (
    SyntheticAudioVae,
    SyntheticVideoVae,
    procedural_audio,
    procedural_video,
)

OPTIONS: dict[str, Any] = {"ace_weights": None, "ace_config": None, "chimera_src": None}


def close(a: float, b: float, tolerance: float) -> bool:
    return abs(float(a) - float(b)) <= tolerance


# --------------------------------------------------------------------------------------
# linalg
# --------------------------------------------------------------------------------------


def test_ridge_recovers_an_exact_linear_map() -> None:
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(400, 12, generator=generator, dtype=torch.float64)
    weight = torch.randn(5, 12, generator=generator, dtype=torch.float64)
    bias = torch.randn(5, generator=generator, dtype=torch.float64)
    targets = inputs @ weight.transpose(0, 1) + bias
    fitted = fit_ridge_linear(inputs, targets, ridge_rel=1e-12, holdout_fraction=0.25)
    assert (fitted.weight - weight).abs().max() < 1e-8, (fitted.weight - weight).abs().max()
    assert (fitted.bias - bias).abs().max() < 1e-8
    assert fitted.holdout_scores["rel_l2"] < 1e-8
    assert fitted.holdout_scores["rows"] == 100


def test_ridge_without_bias_is_unbiased() -> None:
    generator = torch.Generator().manual_seed(1)
    inputs = torch.randn(200, 8, generator=generator, dtype=torch.float64)
    weight = torch.randn(3, 8, generator=generator, dtype=torch.float64)
    fitted = fit_ridge_linear(inputs, inputs @ weight.transpose(0, 1), ridge_rel=1e-12, fit_bias=False)
    assert float(fitted.bias.abs().max()) == 0.0
    assert (fitted.weight - weight).abs().max() < 1e-8


def test_write_scores_is_exact_on_a_perfect_fit() -> None:
    rows = torch.randn(64, 9, dtype=torch.float64)
    scores = write_scores(rows, rows)
    assert scores["rel_l2"] < 1e-12
    assert close(scores["cos"], 1.0, 1e-9)
    assert close(scores["cos_p05"], 1.0, 1e-9)
    assert close(scores["r2"], 1.0, 1e-9)


def test_psnr_and_si_sdr_directions() -> None:
    signal = torch.randn(4, 256, dtype=torch.float64)
    assert math.isinf(psnr(signal, signal))
    quiet = signal + 0.001 * torch.randn(signal.shape, dtype=torch.float64)
    loud = signal + 0.100 * torch.randn(signal.shape, dtype=torch.float64)
    assert psnr(quiet, signal) > psnr(loud, signal)
    assert si_sdr(quiet, signal) > si_sdr(loud, signal)
    # SI-SDR must be scale invariant
    assert close(si_sdr(2.5 * quiet, signal), si_sdr(quiet, signal), 1e-6)


def test_pivoted_columns_finds_the_independent_ones() -> None:
    generator = torch.Generator().manual_seed(3)
    base = torch.randn(300, 4, generator=generator, dtype=torch.float64)
    duplicates = base @ torch.randn(4, 6, generator=generator, dtype=torch.float64)
    features = torch.cat((base * 10.0, duplicates * 1e-6), dim=1)
    selection = pivoted_columns(features, 4)
    assert selection.numel() == 4
    assert set(int(v) for v in selection.tolist()) <= {0, 1, 2, 3}


def test_prior_anchored_returns_the_prior_at_infinite_lambda() -> None:
    generator = torch.Generator().manual_seed(4)
    design = torch.randn(120, 6, generator=generator, dtype=torch.float64)
    targets = torch.randn(120, 3, generator=generator, dtype=torch.float64)
    prior = torch.randn(3, 6, generator=generator, dtype=torch.float64)
    weight, notes = prior_anchored(design, targets, prior, ridge_rel=1e12)
    assert (weight - prior).abs().max() < 1e-4
    assert notes["update_scale"] < 1e-4


def test_law_abiding_solve_vetoes_runaway_rungs() -> None:
    generator = torch.Generator().manual_seed(5)
    design = torch.randn(200, 5, generator=generator, dtype=torch.float64)
    prior = torch.randn(4, 5, generator=generator, dtype=torch.float64)
    targets = design @ prior.transpose(0, 1)
    weight, notes = law_abiding_solve(design, targets, prior)
    assert (weight - prior).abs().max() < 1e-6
    assert all(row["update_scale"] <= 1.0 for row in notes["ladder"] if not row["vetoed"])


def test_orthogonal_procrustes_is_orthogonal() -> None:
    generator = torch.Generator().manual_seed(6)
    source = torch.randn(200, 7, generator=generator, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.randn(7, 7, generator=generator, dtype=torch.float64))
    weight, notes = orthogonal_procrustes(source, source @ rotation)
    assert notes["orthogonality_error"] < 1e-9
    assert notes["scores"]["rel_l2"] < 1e-9


def test_svd_chart_retains_all_energy_at_full_rank() -> None:
    rows = torch.randn(120, 6, dtype=torch.float64)
    chart, notes = svd_chart(rows, 6)
    assert notes["rank"] == 6
    assert close(notes["retained_energy"], 1.0, 1e-9)
    centred = rows - rows.mean(dim=0, keepdim=True)
    assert (centred @ chart.transpose(0, 1) @ chart - centred).abs().max() < 1e-9


def test_situ_at_infinite_beta_is_silu_times_up() -> None:
    gate = torch.randn(64, dtype=torch.float64)
    up = torch.randn(64, dtype=torch.float64)
    expected = torch.nn.functional.silu(gate) * up
    assert (situ_activation(gate, up, beta=float("inf"), linear_beta=float("inf")) - expected).abs().max() < 1e-12


def test_compression_receipt_enforces_the_four_x_floor() -> None:
    assert compression_receipt(4000, 1000, "x")["meets_4x_component_floor"] is True
    assert compression_receipt(4000, 1001, "x")["meets_4x_component_floor"] is False


# --------------------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------------------


def test_synthetic_audio_vae_roundtrip_is_the_row_space_projection() -> None:
    adapter = SyntheticAudioVae(hop_samples=64, channels=2, latent_channels=16, sample_rate_hz=32000)
    clips = procedural_audio(2, samples=64 * 5, channels=2, sample_rate_hz=32000, seed=0)
    latent = adapter.encode(clips)
    assert latent.shape == (2, 16, 5)
    reconstruction = adapter.decode(latent)
    assert reconstruction.shape == clips.shape
    projection = adapter.projection
    frames = clips.reshape(2, 2, 5, 64).permute(0, 2, 1, 3).reshape(2, 5, 128).to(torch.float64)
    expected = frames @ projection.transpose(0, 1) @ projection
    got = reconstruction.reshape(2, 2, 5, 64).permute(0, 2, 1, 3).reshape(2, 5, 128).to(torch.float64)
    assert (got - expected).abs().max() < 1e-6, float((got - expected).abs().max())


def test_synthetic_video_vae_shapes_and_roundtrip() -> None:
    adapter = SyntheticVideoVae(patch=8, tubelet=2, channels=3, latent_channels=12)
    clips = procedural_video(2, frames=4, size=16, seed=0)
    latent = adapter.encode(clips)
    assert latent.shape == (2, 12, 2, 2, 2)
    reconstruction = adapter.decode(latent)
    assert reconstruction.shape == clips.shape
    assert psnr(reconstruction, clips, data_range=1.0) > 0.0


# --------------------------------------------------------------------------------------
# C15
# --------------------------------------------------------------------------------------


def test_audio_pairs_geometry_and_clip_split() -> None:
    adapter = SyntheticAudioVae(hop_samples=64, channels=2, latent_channels=16, sample_rate_hz=32000)
    clips = procedural_audio(8, samples=64 * 6, channels=2, sample_rate_hz=32000, seed=1)
    pairs = sample_audio_pairs(
        adapter, clips, pairing="per_channel", holdout_fraction=0.25, seed=1,
        allow_ambiguous_pairing=True,  # this fixture's latent is joint over stereo
    )
    assert pairs.raw.shape == (8 * 2 * 6, 64)
    assert pairs.latent.shape == (8 * 2 * 6, 16)
    assert pairs.roundtrip.shape == pairs.raw.shape
    assert pairs.centre().shape == pairs.raw.shape
    # split must be constant within a clip -- otherwise held-out rows leak
    for clip in range(8):
        mask = pairs.clip_index.eq(clip)
        assert int(pairs.split[mask].unique().numel()) == 1
    assert int(pairs.split.eq(1).sum()) > 0


def test_audio_pairs_context_widens_rows_and_keeps_the_centre() -> None:
    adapter = SyntheticAudioVae(hop_samples=32, channels=1, latent_channels=8, sample_rate_hz=16000)
    clips = procedural_audio(4, samples=32 * 6, channels=1, sample_rate_hz=16000, seed=2)
    plain = sample_audio_pairs(adapter, clips, seed=2)
    contexted = sample_audio_pairs(adapter, clips, context=1, seed=2)
    assert contexted.raw.shape[1] == 3 * plain.raw.shape[1]
    assert contexted.geometry["centre_offset"] == 32
    assert (contexted.centre() - plain.raw).abs().max() < 1e-6


def test_video_pairs_group_widens_the_latent() -> None:
    adapter = SyntheticVideoVae(patch=8, tubelet=2, channels=3, latent_channels=12)
    clips = procedural_video(4, frames=4, size=32, seed=3)
    single = sample_video_pairs(adapter, clips, latent_group=1, seed=3)
    grouped = sample_video_pairs(adapter, clips, latent_group=2, seed=3)
    assert single.geometry["d_latent"] == 12
    assert grouped.geometry["d_latent"] == 12 * 4
    assert grouped.geometry["d_raw"] == 4 * single.geometry["d_raw"]
    assert grouped.raw.shape[0] * 4 == single.raw.shape[0]


def test_absorption_fit_beats_the_zero_map_and_reports_the_ceiling() -> None:
    adapter = SyntheticAudioVae(hop_samples=64, channels=2, latent_channels=32, sample_rate_hz=32000)
    clips = procedural_audio(16, samples=64 * 8, channels=2, sample_rate_hz=32000, seed=4)
    pairs = sample_audio_pairs(adapter, clips, holdout_fraction=0.25, seed=4, allow_ambiguous_pairing=True)
    maps = fit_absorption(pairs, rung="linear", seed=4)
    metrics = maps.metrics
    assert metrics["rows"] > 0
    assert metrics["encoder_scores"]["rel_l2"] < 1.0
    assert metrics["composed_scores"]["rel_l2"] < 1.0
    assert "psnr_parent_roundtrip_db" in metrics and "psnr_absorbed_db" in metrics
    assert isinstance(metrics["meets_proposed_absorption_line"], bool)
    # the absorbed map can never beat the parent's own round-trip by much on this fixture
    assert metrics["psnr_gap_db"] < 1.0


def test_nonlinear_rung_is_at_least_as_good_as_linear_on_the_fit_split() -> None:
    adapter = SyntheticAudioVae(hop_samples=32, channels=1, latent_channels=16, sample_rate_hz=16000)
    clips = procedural_audio(16, samples=32 * 8, channels=1, sample_rate_hz=16000, seed=5)
    pairs = sample_audio_pairs(adapter, clips, holdout_fraction=0.25, seed=5)
    linear = fit_absorption(pairs, rung="linear", seed=5)
    lifted = fit_absorption(pairs, rung="nonlinear", lift_features=256, seed=5)
    assert lifted.parameters > linear.parameters  # the lift is STORED and counted
    assert lifted.metrics["fit_split"]["encoder_scores"]["rel_l2"] <= linear.metrics["fit_split"]["encoder_scores"]["rel_l2"] + 1e-6


def test_compose_interface_shape_algebra() -> None:
    adapter = SyntheticAudioVae(hop_samples=32, channels=1, latent_channels=8, sample_rate_hz=16000)
    clips = procedural_audio(8, samples=32 * 6, channels=1, sample_rate_hz=16000, seed=6)
    pairs = sample_audio_pairs(adapter, clips, seed=6)
    maps = fit_absorption(pairs, seed=6)
    hidden, model_width, latent = 64, 40, 8
    source = torch.randn(model_width, latent, dtype=torch.float64)
    output = torch.randn(latent, model_width, dtype=torch.float64)
    chart_in = torch.randn(hidden, model_width, dtype=torch.float64)
    chart_out = torch.randn(model_width, hidden, dtype=torch.float64)
    composed = compose_interface(
        maps, hidden=hidden, source_projection=source, output_projection=output,
        chart_in=chart_in, chart_out=chart_out,
    )
    assert composed.feature_proj_weight.shape == (hidden, 32)
    assert composed.type_embed_delta.shape == (hidden,)
    assert composed.emission_out_weight.shape == (32, hidden)
    assert composed.emission_shift.shape == (32,)
    expected = chart_in @ source @ maps.encoder.weight
    assert (composed.feature_proj_weight - expected).abs().max() < 1e-9
    expected_out = maps.decoder.weight @ output @ chart_out
    assert (composed.emission_out_weight - expected_out).abs().max() < 1e-9
    assert composed.notes["stand_ins"] == []


def test_compose_interface_refuses_a_lifted_map() -> None:
    adapter = SyntheticAudioVae(hop_samples=32, channels=1, latent_channels=8, sample_rate_hz=16000)
    clips = procedural_audio(8, samples=32 * 6, channels=1, sample_rate_hz=16000, seed=7)
    pairs = sample_audio_pairs(adapter, clips, seed=7)
    maps = fit_absorption(pairs, rung="nonlinear", lift_features=32, seed=7)
    try:
        compose_interface(maps, hidden=32)
    except ValueError as error:
        assert "linear" in str(error)
        return
    raise AssertionError("a random-feature lift must not silently compose as a linear boundary")


def test_pair_set_round_trips_through_disk_with_receipts() -> None:
    adapter = SyntheticAudioVae(hop_samples=32, channels=2, latent_channels=8, sample_rate_hz=16000)
    clips = procedural_audio(4, samples=32 * 5, channels=2, sample_rate_hz=16000, seed=8)
    pairs = sample_audio_pairs(adapter, clips, seed=8, allow_ambiguous_pairing=True)
    with tempfile.TemporaryDirectory() as directory:
        body = write_pair_set(pairs, directory)
        restored = load_pair_set(directory)
        assert (restored.raw - pairs.raw).abs().max() == 0.0
        assert (restored.latent - pairs.latent).abs().max() == 0.0
        assert restored.geometry["d_raw"] == pairs.geometry["d_raw"]
        assert body["pairs"]["state_sha256"] == receipts.hash_state(pairs.state())
        assert body["pairs"]["parameters"] == sum(int(v.numel()) for v in pairs.state().values())


# --------------------------------------------------------------------------------------
# ACE DiT + C16
# --------------------------------------------------------------------------------------


def _tiny_ace_config():
    from mediafold.ace_dit import AceStepDiTConfig

    return AceStepDiTConfig(
        in_channels=12,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=48,
        patch_size=2,
        audio_acoustic_hidden_dim=4,
        condition_dim=32,
        sliding_window=3,
        layer_types=("sliding_attention", "full_attention"),
    )


def test_ace_block_shapes_and_zero_gate_identity() -> None:
    from mediafold.ace_dit import AceStepDiTLayer

    config = _tiny_ace_config()
    block = AceStepDiTLayer(
        config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
        config.head_dim, config.intermediate_size, config.rms_norm_eps,
        layer_type="full_attention",
    )
    x = torch.randn(2, 10, config.hidden_size)
    temb = torch.zeros(2, 6, config.hidden_size)
    context = torch.randn(2, 5, config.hidden_size)
    out = block(x, temb, encoder_hidden_states=context)
    assert out.shape == x.shape
    # scale_shift_table starts at zero, so gate_msa = gate_mlp = 0: only cross-attention
    # (ungated by construction) writes. Zero its o_proj and the block must be identity.
    with torch.no_grad():
        block.cross_attn.o_proj.weight.zero_()
    assert (block(x, temb, encoder_hidden_states=context) - x).abs().max() < 1e-6
    # and with no context at all, cross-attention is skipped entirely
    assert (block(x, temb, encoder_hidden_states=None) - x).abs().max() < 1e-6


def test_ace_sliding_window_blocks_distant_tokens() -> None:
    from mediafold.ace_dit import AceStepAttention

    attention = AceStepAttention(16, 2, 2, 8, 1e-6, sliding_window=1)
    x = torch.zeros(1, 6, 16)
    x[0, 0] = 5.0
    with torch.no_grad():
        far = attention(x)
        x2 = x.clone()
        x2[0, 0] = -5.0
        far2 = attention(x2)
    # position 5 is 5 steps from position 0 and the window is 1: it must not move
    assert (far[0, 5] - far2[0, 5]).abs().max() < 1e-5
    # position 1 is inside the window: it must move
    assert (far[0, 1] - far2[0, 1]).abs().max() > 1e-5


def test_ace_rope_is_relative() -> None:
    from mediafold.ace_dit import AceStepAttention, rope_tables

    attention = AceStepAttention(16, 2, 2, 8, 1e-6, sliding_window=None)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, 12, 16, generator=generator)
    cos, sin = rope_tables(12, 8, 1e6, x.device, x.dtype)
    with torch.no_grad():
        base = attention(x, position_embeddings=(cos, sin))
        shifted_cos, shifted_sin = rope_tables(16, 8, 1e6, x.device, x.dtype)
        shifted = attention(x, position_embeddings=(shifted_cos[4:], shifted_sin[4:]))
    assert (base - shifted).abs().max() < 1e-4, float((base - shifted).abs().max())


def test_ace_model_forward_shape_and_patch_padding() -> None:
    from mediafold.ace_dit import AceStepDiTModel

    config = _tiny_ace_config()
    model = AceStepDiTModel(config)
    acoustic = config.audio_acoustic_hidden_dim
    for length in (8, 9):  # 9 exercises the patch_size padding branch
        hidden = torch.randn(2, length, acoustic)
        context_latents = torch.randn(2, length, config.in_channels - acoustic)
        condition = torch.randn(2, 4, config.condition_dim)
        timestep = torch.full((2,), 0.5)
        out = model(hidden, timestep, torch.zeros_like(timestep), condition, context_latents)
        assert out.shape == (2, length, acoustic), (length, out.shape)


def test_c16_capture_records_every_write() -> None:
    from mediafold.ace_dit import AceStepDiTModel
    from mediafold.c16_capture import capture, flatten_rows, random_stimulus

    config = _tiny_ace_config()
    model = AceStepDiTModel(config).eval()
    stimulus = random_stimulus(config, batch=2, length=8, condition_length=4, seed=0)
    records = capture(model, stimulus, [0, 1])
    for index in (0, 1):
        store = records[index]
        for name in ("x_in", "x_out", "temb", "context", "mlp_in", "mlp_hidden", "self_attn_out", "cross_attn_out", "mlp_out"):
            assert name in store, (index, name)
        assert store["x_in"].shape == store["x_out"].shape
        assert store["mlp_hidden"].shape[-1] == config.intermediate_size
        assert flatten_rows(store, "x_in").shape == (2 * 4, config.hidden_size)
        assert flatten_rows(store, "temb").shape == (2 * 4, 6, config.hidden_size)
    # block 1 sees block 0's output
    assert (records[1]["x_in"] - records[0]["x_out"]).abs().max() < 1e-6


def test_c16_charts_and_arms() -> None:
    from mediafold.ace_dit import AceStepDiTLayer
    from mediafold.c16_transplant import ARMS, build_arm, fit_charts

    config = _tiny_ace_config()
    generator = torch.Generator().manual_seed(11)
    host = torch.randn(200, 16, generator=generator, dtype=torch.float64)
    mixing = torch.randn(config.hidden_size, 16, generator=generator, dtype=torch.float64)
    parent = host @ mixing.transpose(0, 1)

    ridge = fit_charts(host, parent, method="ridge")
    # the relationship is exactly linear, so the only residual is the ladder's own floor:
    # its smallest rung is RIDGE_REL = 1e-6, which biases the solution by ~1e-6 relative.
    assert ridge.notes["forward_scores"]["rel_l2"] < 1e-4, ridge.notes["forward_scores"]
    assert ridge.notes["roundtrip_scores"]["rel_l2"] < 1e-3, ridge.notes["roundtrip_scores"]
    random_pair = fit_charts(host, parent, method="random", seed=0)
    assert random_pair.notes["forward_scores"]["rel_l2"] > ridge.notes["forward_scores"]["rel_l2"]
    svd_pair = fit_charts(host, parent, method="svd")
    assert svd_pair.parameters > 0

    block = AceStepDiTLayer(
        config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
        config.head_dim, config.intermediate_size, config.rms_norm_eps,
    )
    rows = host.reshape(1, 200, 16).to(torch.float32)
    temb = torch.zeros(1, 6, config.hidden_size)
    for arm in ARMS:
        installed = build_arm(arm, block, ridge, seed=0)
        out = installed(rows, temb, None)
        assert out.shape == rows.shape, arm
        assert installed.frozen_parameters > 0
    # identity arm == chart round-trip, exactly
    identity = build_arm("identity", block, ridge, seed=0)
    expected = ridge.chart_out.apply(ridge.chart_in.apply(rows.to(torch.float64)))
    assert (identity(rows, temb, None).to(torch.float64) - expected).abs().max() < 1e-3
    # a zero-delta block must leave the host untouched under `replace`
    with torch.no_grad():
        block.cross_attn.o_proj.weight.zero_()
    replace = build_arm("replace", block, ridge, seed=0)
    assert (replace(rows, temb, None) - rows).abs().max() < 1e-4


def test_c16_coreset_is_exact_at_full_width_and_four_x_at_quarter_width() -> None:
    from mediafold.ace_dit import AceStepDiTLayer
    from mediafold.c16_transplant import coreset_block_mlp

    config = _tiny_ace_config()
    block = AceStepDiTLayer(
        config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
        config.head_dim, config.intermediate_size, config.rms_norm_eps,
    )
    generator = torch.Generator().manual_seed(12)
    rows = torch.randn(600, config.hidden_size, generator=generator)

    full = coreset_block_mlp(block, rows, keep=config.intermediate_size, seed=0)
    assert full.receipt["component"]["compression_x"] == 1.0
    assert full.receipt["holdout_scores"]["rel_l2"] < 1e-4, full.receipt["holdout_scores"]

    quarter = coreset_block_mlp(block, rows, keep=config.intermediate_size // 4, seed=0)
    assert quarter.receipt["component"]["meets_4x_component_floor"] is True
    assert quarter.receipt["component"]["compression_x"] >= 4.0
    assert quarter.receipt["whole_block"]["compression_x"] < quarter.receipt["component"]["compression_x"]
    assert quarter.block.mlp.gate_proj.weight.shape[0] == config.intermediate_size // 4
    # gate/up rows must be VERBATIM parent rows, never a projection of them
    selection = quarter.selection
    assert (quarter.block.mlp.gate_proj.weight - block.mlp.gate_proj.weight[selection]).abs().max() == 0.0
    assert (quarter.block.mlp.up_proj.weight - block.mlp.up_proj.weight[selection]).abs().max() == 0.0
    assert quarter.receipt["solve"]["update_scale"] <= 1.0


# --------------------------------------------------------------------------------------
# optional: real weights / real trunk
# --------------------------------------------------------------------------------------


def test_real_ace_block_loads_and_runs() -> None:
    weights = OPTIONS.get("ace_weights")
    if not weights:
        raise Skip("no --ace-weights")
    from mediafold.ace_dit import AceStepDiTConfig, load_ace_dit_block

    config = AceStepDiTConfig.from_json(OPTIONS["ace_config"])
    assert config.hidden_size == 2048 and config.num_hidden_layers == 24
    assert config.intermediate_size == 6144 and config.num_key_value_heads == 8
    block, receipt = load_ace_dit_block(12, weights=weights, config=config)
    assert receipt["consumed_tensors"] == 19, receipt["consumed_tensors"]
    assert receipt["layer_type"] == "sliding_attention"
    assert receipt["mlp_parameters"] == 3 * 6144 * 2048
    x = torch.randn(1, 24, 2048)
    temb = torch.zeros(1, 6, 2048)
    context = torch.randn(1, 8, 2048)
    with torch.no_grad():
        out = block(x, temb, encoder_hidden_states=context)
    assert out.shape == x.shape
    assert bool(torch.isfinite(out).all())


def test_real_ace_block_coreset_four_x() -> None:
    weights = OPTIONS.get("ace_weights")
    if not weights:
        raise Skip("no --ace-weights")
    from mediafold.ace_dit import AceStepDiTConfig, load_ace_dit_block
    from mediafold.c16_transplant import coreset_block_mlp

    config = AceStepDiTConfig.from_json(OPTIONS["ace_config"])
    block, _ = load_ace_dit_block(12, weights=weights, config=config)
    generator = torch.Generator().manual_seed(0)
    rows = torch.randn(4096, 2048, generator=generator) * 0.5
    result = coreset_block_mlp(block, rows, keep=1536, seed=0)
    assert result.receipt["component"]["compression_x"] == 4.0
    assert result.receipt["component"]["meets_4x_component_floor"] is True
    assert result.receipt["holdout_scores"]["rel_l2"] < 1.0
    print(f"        [real ACE coreset 6144->1536 holdout relL2 = {result.receipt['holdout_scores']['rel_l2']:.4f}]")


def test_proof_trunk_fixture_pairs_match_the_proof_geometry() -> None:
    source = OPTIONS.get("chimera_src")
    if not source:
        raise Skip("no --chimera-src")
    from h3_0_proof_rung import build_fixture_pairs, import_proof

    proof, _ = import_proof(source)
    pairs, extra = build_fixture_pairs(
        proof, clips=8, audio_adapter_spec="synthetic", video_adapter_spec="synthetic",
        latent_width=32, seed=0,
    )
    assert pairs.geometry["d_raw"] == proof.FEATURE_DIM
    assert pairs.geometry["d_latent"] == 32
    assert pairs.raw.shape[0] == extra["type_rows"].shape[0]
    assert set(int(v) for v in extra["type_rows"].unique().tolist()) == {2, 3}
    maps = fit_absorption(pairs, seed=0)
    assert maps.encoder.weight.shape == (32, proof.FEATURE_DIM)
    assert maps.decoder.weight.shape == (proof.FEATURE_DIM, 32)


def test_proof_trunk_absorbed_install_writes_the_composed_tensors() -> None:
    source = OPTIONS.get("chimera_src")
    if not source:
        raise Skip("no --chimera-src")
    from h3_0_proof_rung import build_fixture_pairs, import_proof, install_absorbed_interface

    proof, _ = import_proof(source)
    pairs, extra = build_fixture_pairs(
        proof, clips=8, audio_adapter_spec="synthetic", video_adapter_spec="synthetic",
        latent_width=32, seed=0,
    )
    maps = fit_absorption(pairs, seed=0)
    device = torch.device("cpu")
    model = proof.build_model(device)
    before = proof.hash_state_dict(model)
    notes = install_absorbed_interface(
        model, maps, type_rows=extra["type_rows"], latent_width=32, scramble=False, seed=0
    )
    after = proof.hash_state_dict(model)
    assert before != after
    assert notes["shapes"]["bridge.feature_proj.weight"] == [128, proof.FEATURE_DIM]
    assert notes["shapes"]["head.out.weight"] == [proof.FEATURE_DIM, 128]
    assert notes["decoder_bias_relative_residual"] < 1.0
    # the control arm must produce a DIFFERENT install of the same shape
    control = proof.build_model(device)
    install_absorbed_interface(
        control, maps, type_rows=extra["type_rows"], latent_width=32, scramble=True, seed=0
    )
    assert proof.hash_state_dict(control) != after
    assert control.native_media_bridge.feature_proj.weight.shape == model.native_media_bridge.feature_proj.weight.shape


# --------------------------------------------------------------------------------------
# measured-tubelet video pairing (mediafold.c15_video_groups)
# --------------------------------------------------------------------------------------


def test_measured_frame_groups_recover_a_uniform_tubelet() -> None:
    from mediafold.c15_video_groups import measure_frame_groups

    adapter = SyntheticVideoVae(patch=8, tubelet=2, latent_channels=12)
    report = measure_frame_groups(adapter, frames=8, size=16, amplitude=0.5, seed=3)
    assert report["latent_frames"] == 4
    assert report["group_sizes"] == [2]
    assert report["uniform"] is True
    assert report["owner_per_frame"] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert all(report["contiguous"].values())


def test_grouped_video_pairs_equal_the_uniform_sampler_row_for_row() -> None:
    from mediafold.c15_video_groups import measure_frame_groups, sample_grouped_video_pairs

    adapter = SyntheticVideoVae(patch=8, tubelet=2, latent_channels=12)
    clips = procedural_video(4, frames=8, size=16, seed=5)
    report = measure_frame_groups(adapter, frames=8, size=16, amplitude=0.5, seed=3)
    grouped = sample_grouped_video_pairs(
        adapter, clips, group_map=report, group_size=2, latent_group=1, holdout_fraction=0.25, seed=1
    )
    uniform = sample_video_pairs(adapter, clips, latent_group=1, holdout_fraction=0.25, seed=1)
    assert grouped.geometry["d_raw"] == uniform.geometry["d_raw"] == 2 * 3 * 8 * 8
    assert grouped.geometry["d_latent"] == uniform.geometry["d_latent"] == 12
    assert grouped.raw.shape == uniform.raw.shape
    patches = 4  # 16 / 8 grid squared
    # uniform rows are ordered (clip, tubelet, patch); grouped rows are ordered
    # (batch, latent frame, clip-in-batch, patch): compare the same coordinates explicitly.
    for clip in range(4):
        for latent_frame in range(4):
            for patch in range(patches):
                left = uniform.raw[(clip * 4 + latent_frame) * patches + patch]
                grouped_rows = torch.nonzero(
                    grouped.clip_index.eq(clip) & grouped.frame_index.eq(latent_frame)
                ).flatten()
                right = grouped.raw[grouped_rows[patch]]
                assert torch.equal(left, right)
    held_grouped = sorted({int(v) for v in grouped.clip_index[grouped.split.eq(1)].tolist()})
    held_uniform = sorted({int(v) for v in uniform.clip_index[uniform.split.eq(1)].tolist()})
    assert held_grouped == held_uniform  # the holdout is split by CLIP, identically in both samplers


def test_grouped_video_pairs_refuse_a_group_map_from_another_clip_length() -> None:
    from mediafold.c15_video_groups import measure_frame_groups, sample_grouped_video_pairs

    adapter = SyntheticVideoVae(patch=8, tubelet=2, latent_channels=12)
    report = measure_frame_groups(adapter, frames=8, size=16, amplitude=0.5, seed=3)
    clips = procedural_video(2, frames=12, size=16, seed=5)
    try:
        sample_grouped_video_pairs(adapter, clips, group_map=report, group_size=2)
    except ValueError as error:
        assert "measured on 8-frame clips" in str(error)
    else:  # pragma: no cover - the guard is the point of the test
        raise AssertionError("a mismatched group map must be refused")


def test_grouped_video_pairs_skip_the_non_conforming_latent_frames() -> None:
    from mediafold.c15_video_groups import sample_grouped_video_pairs

    adapter = SyntheticVideoVae(patch=8, tubelet=2, latent_channels=12)
    clips = procedural_video(2, frames=8, size=16, seed=5)
    # a hand-written map in the shape of the MEASURED H3 video map: sizes 1, 3, 4
    forged = {
        "frames": 8,
        "size": 16,
        "latent_frames": 4,
        "groups": {"0": [0], "1": [1, 2, 3], "2": [4, 5, 6, 7], "3": []},
        "contiguous": {"0": True, "1": True, "2": True, "3": True},
        "group_sizes": [1, 3, 4],
        "uniform": False,
    }
    pairs = sample_grouped_video_pairs(adapter, clips, group_map=forged, group_size=4, holdout_fraction=0.5)
    assert pairs.geometry["selected_latent_frames"] == [2]
    assert pairs.geometry["skipped_latent_frames"] == [0, 1, 3]
    assert pairs.geometry["d_raw"] == 4 * 3 * 8 * 8
    assert pairs.geometry["chimera_tubelet"] == 4
    assert pairs.raw.shape[0] == 2 * 4  # 2 clips x 4 spatial patches, one latent frame


def test_grouped_rows_match_chimeras_own_video_patchifier() -> None:
    source = OPTIONS.get("chimera_src")
    if not source:
        raise Skip("no --chimera-src")
    from h3_0_proof_rung import import_proof
    from mediafold.c15_video_groups import _patch_rows

    import_proof(source)  # puts the chimera_omni package on sys.path
    from chimera_omni.native_media import video_to_native_patches

    clips = procedural_video(2, frames=4, size=16, seed=11)
    packet = video_to_native_patches(clips, patch=8, tubelet=4, feature_dim=4 * 3 * 8 * 8, fps=8.0)
    rows = _patch_rows(clips[:, :4], 8)
    assert packet.features.shape == rows.shape
    assert torch.allclose(packet.features, rows, atol=0.0, rtol=0.0)


def test_audio_pair_rows_match_chimeras_own_segmenter() -> None:
    """The audio row a C15 map is fitted on must BE the Chimera token it will feed.

    Chimera emits one token per (frame, channel) with feature index = raw sample offset;
    the sampler emits one row per (channel, frame). Same content, different row order --
    this test pins the mapping so a fitted ``E`` installs into
    ``NativeContinuousMediaBridge.feature_proj`` without a silent transpose.
    """

    source = OPTIONS.get("chimera_src")
    if not source:
        raise Skip("no --chimera-src")
    from h3_0_proof_rung import import_proof

    import_proof(source)
    from chimera_omni.native_media import waveform_to_native_segments

    adapter = SyntheticAudioVae(hop_samples=800, channels=2, latent_channels=32, sample_rate_hz=32000)
    clips = procedural_audio(3, samples=800 * 6, channels=2, sample_rate_hz=32000, seed=17)
    pairs = sample_audio_pairs(
        adapter, clips, pairing="per_channel", holdout_fraction=0.34, seed=0,
        allow_ambiguous_pairing=True,
    )
    packet = waveform_to_native_segments(
        clips, kind="audio", segment=800, stride=800, feature_dim=800, sample_rate_hz=32000
    )
    frames, channels = 6, 2
    assert pairs.geometry["d_raw"] == 800
    assert packet.features.shape == (3, frames * channels, 800)
    for clip in range(3):
        for frame in range(frames):
            for channel in range(channels):
                chimera_row = packet.features[clip, frame * channels + channel]
                sampler_row = pairs.raw[clip * frames * channels + channel * frames + frame]
                assert torch.allclose(chimera_row, sampler_row, atol=0.0, rtol=0.0)


def test_merged_pair_sets_concatenate_and_renumber_clips() -> None:
    import tempfile

    from mediafold.c15_merge import merge_pair_sets

    adapter = SyntheticAudioVae(hop_samples=256, channels=2, latent_channels=8, sample_rate_hz=16000)
    first = sample_audio_pairs(
        adapter, procedural_audio(4, samples=256 * 5, channels=2, sample_rate_hz=16000, seed=1),
        holdout_fraction=0.25, seed=1, allow_ambiguous_pairing=True,
    )
    second = sample_audio_pairs(
        adapter, procedural_audio(4, samples=256 * 5, channels=2, sample_rate_hz=16000, seed=2),
        holdout_fraction=0.25, seed=2, allow_ambiguous_pairing=True,
    )
    merged = merge_pair_sets([first, second])
    assert merged.raw.shape[0] == first.raw.shape[0] + second.raw.shape[0]
    assert merged.geometry["clips"] == 8
    assert int(merged.clip_index.max().item()) == 7
    # every row still carries its own clip's split flag
    assert torch.equal(merged.split[: first.split.numel()], first.split)
    assert torch.equal(merged.split[first.split.numel() :], second.split)
    # a holdout clip is a whole clip, never half of one
    for clip in range(8):
        flags = merged.split[merged.clip_index.eq(clip)].unique()
        assert flags.numel() == 1
    with tempfile.TemporaryDirectory() as folder:
        from mediafold.c15_pairs import load_pair_set, write_pair_set

        write_pair_set(merged, folder)
        again = load_pair_set(folder)
        assert torch.equal(again.raw, merged.raw)


def test_merge_refuses_a_geometry_mismatch() -> None:
    from mediafold.c15_merge import merge_pair_sets

    narrow = SyntheticAudioVae(hop_samples=256, channels=2, latent_channels=8, sample_rate_hz=16000)
    wide = SyntheticAudioVae(hop_samples=512, channels=2, latent_channels=8, sample_rate_hz=16000)
    left = sample_audio_pairs(
        narrow, procedural_audio(2, samples=256 * 5, channels=2, sample_rate_hz=16000, seed=1),
        seed=1, allow_ambiguous_pairing=True,
    )
    right = sample_audio_pairs(
        wide, procedural_audio(2, samples=512 * 5, channels=2, sample_rate_hz=16000, seed=1),
        seed=1, allow_ambiguous_pairing=True,
    )
    try:
        merge_pair_sets([left, right])
    except ValueError as error:
        assert "d_raw" in str(error)
    else:  # pragma: no cover
        raise AssertionError("merging two geometries must be refused")


def test_compose_interface_stand_ins_are_self_consistent() -> None:
    """Every projection/chart omitted: the composition must still build.

    Regression: the default ``chart_out`` stand-in was constructed with its arguments
    reversed, so ANY call that let the harness fill in the charts died with
    ``B_out must be [model_width, hidden]``. The unit tests all passed explicit charts,
    so the first thing to hit it was a real ladder run over the ACE VAE pairs.

    This test only covers the ALL-SQUARE path (source_projection omitted), which is why it
    could not catch the identical defect one branch up in ``output_projection``; see
    ``test_compose_interface_stand_in_survives_a_non_square_source_projection``.
    """

    adapter = SyntheticAudioVae(hop_samples=128, channels=2, latent_channels=8, sample_rate_hz=16000)
    clips = procedural_audio(6, samples=128 * 8, channels=2, sample_rate_hz=16000, seed=4)
    pairs = sample_audio_pairs(adapter, clips, holdout_fraction=0.25, seed=0, allow_ambiguous_pairing=True)
    maps = fit_absorption(pairs, seed=0)
    composed = compose_interface(maps, hidden=64)
    assert composed.feature_proj_weight.shape == (64, pairs.geometry["d_raw"])
    assert composed.emission_out_weight.shape == (pairs.geometry["d_centre"], 64)
    assert composed.type_embed_delta.shape == (64,)
    assert sorted(composed.notes["stand_ins"]) == [
        "chart_in=identity", "chart_out=identity", "output_projection=identity", "source_projection=identity",
    ]
    # the stand-in path must reproduce E and D exactly on the first d_latent coordinates
    rows = pairs.raw[:16].to(torch.float64)
    latent_hat = maps.encode_rows(rows)
    hidden_rows = rows @ composed.feature_proj_weight.transpose(0, 1) + composed.type_embed_delta
    assert torch.allclose(hidden_rows[:, : latent_hat.shape[1]], latent_hat, atol=1e-9)


def test_si_sdr_ignores_silent_rows() -> None:
    from mediafold.linalg import si_sdr_detail  # noqa: F811

    generator = torch.Generator().manual_seed(0)
    voiced = torch.randn(6, 256, generator=generator)
    target = torch.cat((voiced, torch.zeros(3, 256)), dim=0)
    prediction = torch.cat((voiced * 0.99, torch.zeros(3, 256)), dim=0)
    detail = si_sdr_detail(prediction, target)
    assert detail["scored_rows"] == 6.0
    assert detail["silent_rows"] == 3.0
    assert detail["si_sdr_db"] > 40.0  # a 1% scale error is scale-invariant: near-perfect
    # the pre-fix behaviour, for the record: a silent row scored 10*log10(1e-30) = -300 dB
    # and pulled the mean down by 300 dB per silent row over the row count. On the real
    # corpus (a third of the rows silent) that turned a +13 dB parent round-trip into -72.
    voiced_only = si_sdr_detail(prediction[:6], target[:6])["si_sdr_db"]
    assert close(voiced_only, detail["si_sdr_db"], 1.0e-9)
    silent_penalty = (6.0 * voiced_only + 3.0 * -300.0) / 9.0
    assert silent_penalty < detail["si_sdr_db"]


def test_linear_rank_bound_is_an_upper_bound_on_a_fitted_linear_map() -> None:
    from mediafold.linalg import best_linear_rank_psnr, psnr

    generator = torch.Generator().manual_seed(1)
    basis = torch.linalg.qr(torch.randn(64, 8, generator=generator, dtype=torch.float64))[0]
    coefficients = torch.randn(400, 8, generator=generator, dtype=torch.float64)
    rows = coefficients @ basis.transpose(0, 1) + 0.01 * torch.randn(400, 64, generator=generator, dtype=torch.float64)
    fit, held = rows[:300], rows[300:]
    bound = best_linear_rank_psnr(fit, held, 8, data_range=2.0)
    assert bound["retained_energy"] > 0.99
    # a rank-8 ridge autoencoder fitted the ordinary way cannot beat the PCA bound
    encoder = torch.linalg.lstsq(fit, coefficients[:300]).solution
    decoder = torch.linalg.lstsq(coefficients[:300], fit).solution
    fitted = held @ encoder @ decoder
    assert psnr(fitted, held, data_range=2.0) <= bound["psnr_db"] + 1.0e-6


# --------------------------------------------------------------------------------------
# review blockers 2026-09-02: one regression test per defect
# --------------------------------------------------------------------------------------


def test_per_channel_pairing_is_refused_on_a_joint_latent() -> None:
    """BLOCKER 1. ACE 1.5's VAE emits one latent for a STEREO frame.

    Pairing per channel then broadcasts that one latent to both channel rows: E sees two
    different inputs mapped to an identical target, D one latent mapped to two different
    targets. The sampler must refuse rather than produce rows whose scores measure the
    pairing. Measured on the delivered ACE corpus: 6400 rows but only 2739 unique latents,
    an irreducible rel_l2 floor of 0.815 against a fitted 0.867.
    """

    adapter = SyntheticAudioVae(hop_samples=64, channels=2, latent_channels=16, sample_rate_hz=32000)
    assert adapter.geometry.channel_in_latent is False
    clips = procedural_audio(4, samples=64 * 5, channels=2, sample_rate_hz=32000, seed=1)
    try:
        sample_audio_pairs(adapter, clips, pairing="per_channel", seed=1)
    except ValueError as error:
        assert "joint" in str(error).lower() and "stacked" in str(error)
    else:  # pragma: no cover
        raise AssertionError("per_channel on a joint latent must be refused")
    # a single-channel VAE has no ambiguity, so it must still be allowed
    mono = SyntheticAudioVae(hop_samples=64, channels=1, latent_channels=16, sample_rate_hz=32000)
    mono_clips = procedural_audio(4, samples=64 * 5, channels=1, sample_rate_hz=32000, seed=1)
    assert sample_audio_pairs(mono, mono_clips, pairing="per_channel", seed=1).raw.shape[1] == 64


def test_stacked_pairing_has_one_latent_per_row_and_no_floor() -> None:
    """BLOCKER 1, the prescribed fix: stacked rows are one-to-one with the latents."""

    adapter = SyntheticAudioVae(hop_samples=64, channels=2, latent_channels=16, sample_rate_hz=32000)
    clips = procedural_audio(6, samples=64 * 5, channels=2, sample_rate_hz=32000, seed=2)
    pairs = sample_audio_pairs(adapter, clips, pairing="stacked", seed=2)
    assert pairs.geometry["d_raw"] == 2 * 64
    assert pairs.geometry["d_centre"] == 2 * 64
    # the parent emitted ONE latent for the stereo frame, so the row carries it once --
    # reshaping the broadcast channel axis would have emitted 32 columns twice over
    assert pairs.geometry["d_latent"] == 16
    assert pairs.geometry["ambiguous_pairing"] is False
    assert pairs.geometry["ambiguity"] is None
    # one row per (clip, frame), so the many-to-one floor is exactly zero. Counting
    # DISTINCT latents is the wrong check: the procedural corpus zero-pads, and several
    # silent frames legitimately encode to the same latent (27 of 30 rows distinct here).
    # That is a property of the corpus; the pairing is what has to be one-to-one.
    from mediafold.c15_pairs import ambiguity_floor

    floor = ambiguity_floor(pairs.centre(), pairs.clip_index, pairs.frame_index, pairs.latent)
    assert floor["groups"] == floor["rows"]
    assert floor["rows_per_group_max"] == 1
    assert floor["irreducible_rel_l2"] == 0.0


def test_stacked_pairing_concatenates_a_real_per_channel_latent() -> None:
    """The mirror case: when the parent DOES emit one latent per channel (H3's audio VAE,
    channel_in_latent=True), stacked rows must carry both of them, not one."""

    class _PerChannelVae(SyntheticAudioVae):
        def __init__(self) -> None:
            super().__init__(hop_samples=64, channels=2, latent_channels=16, sample_rate_hz=32000)
            self.geometry.channel_in_latent = True

        def encode(self, raw: torch.Tensor) -> torch.Tensor:  # [B, Cz, C, T]
            mono = [super(_PerChannelVae, self).encode(raw[:, index : index + 1].repeat(1, 2, 1))
                    for index in range(raw.shape[1])]
            return torch.stack(mono, dim=2)

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            pieces = [super(_PerChannelVae, self).decode(latent[:, :, index])[:, :1]
                      for index in range(latent.shape[2])]
            return torch.cat(pieces, dim=1)

    adapter = _PerChannelVae()
    clips = procedural_audio(4, samples=64 * 5, channels=2, sample_rate_hz=32000, seed=4)
    per_channel = sample_audio_pairs(adapter, clips, pairing="per_channel", seed=4)
    assert per_channel.geometry["d_latent"] == 16  # one channel's latent
    assert per_channel.geometry["ambiguous_pairing"] is False  # no broadcast, no floor
    stacked = sample_audio_pairs(adapter, clips, pairing="stacked", seed=4)
    assert stacked.geometry["d_latent"] == 32  # both channels' latents


def test_allowed_ambiguous_pairing_records_the_measured_floor() -> None:
    """BLOCKER 1, the escape hatch: allowed only WITH the floor beside every score."""

    adapter = SyntheticAudioVae(hop_samples=64, channels=2, latent_channels=16, sample_rate_hz=32000)
    clips = procedural_audio(6, samples=64 * 5, channels=2, sample_rate_hz=32000, seed=3)
    pairs = sample_audio_pairs(adapter, clips, pairing="per_channel", seed=3, allow_ambiguous_pairing=True)
    ambiguity = pairs.geometry["ambiguity"]
    assert pairs.geometry["ambiguous_pairing"] is True
    assert ambiguity["rows"] == pairs.raw.shape[0]
    assert ambiguity["groups"] * 2 == ambiguity["rows"]  # two channels share one latent
    assert ambiguity["rows_per_group_max"] == 2
    assert ambiguity["unique_latent_rows"] < ambiguity["rows"]
    assert 0.0 < ambiguity["irreducible_rel_l2"] < 1.0
    # no fitted map can beat the floor: check it directly with the best possible decoder
    maps = fit_absorption(pairs, rung="linear", seed=3)
    assert maps.metrics["composed_scores"]["rel_l2"] >= ambiguity["irreducible_rel_l2"] * 0.9
    assert maps.metrics["pairing_ambiguity"]["irreducible_rel_l2"] == ambiguity["irreducible_rel_l2"]
    assert "pairing_ambiguity_warning" in maps.metrics


def test_transplanted_block_runs_the_parent_with_its_rope() -> None:
    """BLOCKER 3. Block 12 is a sliding_attention layer; without its rotary phase the arm
    scores are a different function of the same weights than the parent reference."""

    from mediafold.ace_dit import AceStepDiTLayer, rope_tables
    from mediafold.c16_transplant import build_arm, fit_charts

    config = _tiny_ace_config()
    generator = torch.Generator().manual_seed(21)
    host = torch.randn(120, 16, generator=generator, dtype=torch.float64)
    mixing = torch.randn(config.hidden_size, 16, generator=generator, dtype=torch.float64)
    parent = host @ mixing.transpose(0, 1)
    charts = fit_charts(host, parent, method="ridge")
    block = AceStepDiTLayer(
        config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
        config.head_dim, config.intermediate_size, config.rms_norm_eps,
        layer_type="sliding_attention", sliding_window=config.sliding_window,
    )
    with torch.no_grad():  # ungate the block so the attention write is visible
        block.scale_shift_table.fill_(0.5)

    rows = host.reshape(1, 120, 16).to(torch.float32)
    temb = torch.zeros(1, 6, config.hidden_size)
    installed = build_arm("replace", block, charts, seed=0, rope_theta=float(config.rope_theta))
    assert installed.head_dim == config.head_dim

    projected = installed.project_in(rows)
    cos, sin = rope_tables(projected.shape[1], config.head_dim, config.rope_theta, projected.device, projected.dtype)
    with torch.no_grad():
        wanted = block(projected, temb, encoder_hidden_states=None, position_embeddings=(cos, sin)) - projected
        without = block(projected, temb, encoder_hidden_states=None, position_embeddings=None) - projected
        produced = installed.block_delta(projected, temb, None)
    assert (produced - wanted).abs().max() < 1e-6, "the arm must run the parent WITH rope"
    assert (without - wanted).abs().max() > 1e-4, "rope must actually change this block"


def test_control_arm_keeps_the_norm_gains_and_matches_the_write_strength() -> None:
    """BLOCKER 4. Zeroing every 1-D parameter zeroes the five RMSNorm weights, so the
    normalised activations are identically 0 and the block emits nothing. That is a dead
    block, not a norm-matched random one, and it attributes nothing."""

    from mediafold.ace_dit import AceStepDiTLayer
    from mediafold.c16_transplant import build_arm, fit_charts

    config = _tiny_ace_config()
    generator = torch.Generator().manual_seed(22)
    host = torch.randn(120, 16, generator=generator, dtype=torch.float64)
    mixing = torch.randn(config.hidden_size, 16, generator=generator, dtype=torch.float64)
    charts = fit_charts(host, host @ mixing.transpose(0, 1), method="ridge")
    block = AceStepDiTLayer(
        config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
        config.head_dim, config.intermediate_size, config.rms_norm_eps,
    )
    with torch.no_grad():
        block.scale_shift_table.fill_(0.5)
    rows = host.reshape(1, 120, 16).to(torch.float32)
    temb = torch.zeros(1, 6, config.hidden_size)

    control = build_arm("control", block, charts, seed=0, norm_match_rows=rows, temb=temb)
    for name, parameter in control.block.named_parameters():
        if parameter.dim() < 2:
            assert parameter.abs().max() > 0.0, f"{name} was zeroed: the control block is dead"
            assert torch.equal(parameter, dict(block.named_parameters())[name]), name
        else:
            assert not torch.equal(parameter, dict(block.named_parameters())[name]), name

    replace = build_arm("replace", block, charts, seed=0)
    projected = control.project_in(rows)
    with torch.no_grad():
        parent_norm = float(replace.block_delta(projected, temb, None).norm().item())
        control_norm = float((control.block_delta(projected, temb, None) * control.delta_scale).norm().item())
    assert parent_norm > 0.0
    assert abs(control_norm - parent_norm) / parent_norm < 1e-4, (control_norm, parent_norm)
    # and the arm now writes something, instead of the ~0.01 rel_l2 a dead block gave
    with torch.no_grad():
        written = (control(rows, temb, None) - rows).norm() / rows.norm()
    assert float(written) > 1e-3, float(written)


def test_prefix_shell_reproduces_the_real_block_input() -> None:
    """BLOCKER 5. Driving block i off ``block_inputs`` alone feeds it the LAYER-0 residual
    stream. Running layers 0..i-1 first must reproduce the full-stack capture exactly."""

    from mediafold.ace_dit import AceStepDiTModel, rope_tables
    from mediafold.c16_capture import capture, random_stimulus

    config = _tiny_ace_config()
    torch.manual_seed(23)
    model = AceStepDiTModel(config)
    model.eval()
    stimulus = random_stimulus(config, batch=2, length=16, seed=1)
    records = capture(model, stimulus, [0, 1])

    with torch.no_grad():
        prepared = model.block_inputs(
            stimulus.hidden_states, stimulus.timestep, stimulus.timestep_r,
            stimulus.encoder_hidden_states, stimulus.context_latents,
        )
        position = (prepared["cos"], prepared["sin"])
        naive = prepared["hidden"]
        prefixed = model.layers[0](
            naive, prepared["temb"], encoder_hidden_states=prepared["context"],
            position_embeddings=position,
        )
    assert (prefixed - records[1]["x_in"]).abs().max() < 1e-5, "prefix must equal the full stack"
    assert (naive - records[1]["x_in"]).abs().max() > 1e-5, "layer-0 rows are NOT layer-1 rows"
    # the distribution check the receipt now carries
    delta = records[1]["x_out"] - records[1]["x_in"]
    assert float(delta.norm() / records[1]["x_in"].norm()) < 100.0


def test_coreset_refuses_an_interpolation_threshold_row_count() -> None:
    """BLOCKER 6. 1638 fit rows for a 1536-channel coreset leaves the down_proj refit with
    1537 free parameters per output row; the guard must fire BEFORE pivoting, not after."""

    from mediafold.ace_dit import AceStepDiTLayer
    from mediafold.c16_transplant import coreset_block_mlp

    config = _tiny_ace_config()
    block = AceStepDiTLayer(
        config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
        config.head_dim, config.intermediate_size, config.rms_norm_eps,
    )
    generator = torch.Generator().manual_seed(24)
    keep = config.intermediate_size // 4  # 12
    starved = torch.randn(int(keep * 2 * 0.9), config.hidden_size, generator=generator)
    try:
        coreset_block_mlp(block, starved, keep=keep, seed=0)
    except ValueError as error:
        assert "fit rows" in str(error) and "interpolation threshold" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a coreset at the interpolation threshold must be refused")
    healthy = torch.randn(keep * 8, config.hidden_size, generator=generator)
    result = coreset_block_mlp(block, healthy, keep=keep, seed=0)
    assert result.receipt["fit_rows_per_kept_channel"] >= 2.0


def test_coreset_holdout_splits_by_sequence_when_told_how() -> None:
    """Neighbouring positions inside one sequence are correlated; a row split leaks them."""

    from mediafold.ace_dit import AceStepDiTLayer
    from mediafold.c16_transplant import coreset_block_mlp

    config = _tiny_ace_config()
    block = AceStepDiTLayer(
        config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
        config.head_dim, config.intermediate_size, config.rms_norm_eps,
    )
    generator = torch.Generator().manual_seed(25)
    sequences, length = 10, 40
    rows = torch.randn(sequences * length, config.hidden_size, generator=generator)
    index = torch.arange(sequences).repeat_interleave(length)
    result = coreset_block_mlp(
        block, rows, keep=config.intermediate_size // 4, seed=0, sequence_index=index
    )
    assert result.receipt["split_by"].startswith("sequence")
    assert result.receipt["holdout_rows"] % length == 0  # whole sequences, never half of one
    assert result.receipt["fit_rows"] + result.receipt["holdout_rows"] == sequences * length


def _clip_structured_pair_set(clips: int, rows_per_clip: int, latent_width: int, seed: int):
    """A PairSet whose rows carry a real clip structure: ``rows_per_clip`` contiguous rows
    per clip, each clip's latents drawn around its own centre so rows of one clip are near
    -duplicates of each other -- exactly the correlation a clip-blind split leaks."""

    from mediafold.c15_pairs import PairSet

    generator = torch.Generator().manual_seed(int(seed))
    clip_index = torch.arange(clips).repeat_interleave(rows_per_clip)
    centres = torch.randn(clips, latent_width, generator=generator) * 4.0
    latent = centres[clip_index] + 0.02 * torch.randn(
        clips * rows_per_clip, latent_width, generator=generator
    )
    raw = latent.clone()
    return PairSet(
        raw=raw,
        latent=latent,
        roundtrip=raw.clone(),
        clip_index=clip_index,
        frame_index=torch.arange(rows_per_clip).repeat(clips),
        channel_index=torch.zeros(clips * rows_per_clip, dtype=torch.long),
        split=torch.zeros(clips * rows_per_clip, dtype=torch.long),
        geometry={
            "name": "clip-structured-fixture",
            "modality": "audio",
            "d_raw": latent_width,
            "d_latent": latent_width,
            "d_centre": latent_width,
            "centre_offset": 0,
            "rows": clips * rows_per_clip,
            "clips": clips,
        },
    )


def _grouped_split(groups: torch.Tensor, fraction: float = 0.2, seed: int = 0) -> torch.Tensor:
    """The holdout mask ``coreset_block_mlp`` builds, reproduced here so a test can ask
    what a DIFFERENT grouping of the same rows would have held out."""

    generator = torch.Generator().manual_seed(int(seed))
    unique = torch.unique(groups)
    shuffled = unique[torch.randperm(int(unique.numel()), generator=generator)]
    held = max(1, int(round(fraction * int(unique.numel()))))
    return torch.isin(groups, shuffled[:held])


def test_coreset_holdout_groups_by_clip_not_by_batch_row() -> None:
    """ROUND-2 BLOCKER (media). ``coreset_block_mlp`` was handed one group id per DiT
    BATCH ROW. A batch row is a contiguous ``--length`` window of the pair corpus and a
    clip is only ``rows_per_clip`` rows long, so one batch row spans several clips and
    most held-out rows share a clip with the fit -- that "holdout" score is part fit
    score. The invariant is: a held-out clip has ZERO rows in the fit. This test measures
    the leak under BOTH groupings on the same stimulus, so it fails on the old one."""

    from mediafold.ace_dit import AceStepDiTLayer
    from mediafold.c15_pairs import write_pair_set
    from mediafold.c16_capture import coreset_row_groups, latent_stimulus, token_clip_index
    from mediafold.c16_transplant import coreset_block_mlp

    config = _tiny_ace_config()
    latent_width = config.audio_acoustic_hidden_dim
    # 11 rows per clip against patch_size 2: clip boundaries alternate parity, so the
    # window is guaranteed to contain patches that straddle a boundary whatever the
    # random row offset lands on.
    rows_per_clip, clips = 11, 40
    pairs = _clip_structured_pair_set(clips, rows_per_clip, latent_width, seed=31)

    with tempfile.TemporaryDirectory() as directory:
        write_pair_set(pairs, Path(directory) / "pairs")
        batch, length = 8, 32
        stimulus = latent_stimulus(
            config, Path(directory) / "pairs", batch=batch, length=length, seed=3
        )

    # 1. the corpus's own clip ids reached the stimulus, on the [B, L] latent grid
    assert stimulus.clip_index is not None, "latent_stimulus must carry clip provenance"
    assert tuple(stimulus.clip_index.shape) == (batch, length)
    start = int(stimulus.notes["row_offset"])
    expected = pairs.clip_index[start : start + batch * length].reshape(batch, length)
    assert torch.equal(stimulus.clip_index, expected)
    # a batch row really does straddle clips, or this fixture proves nothing
    spans = [int(torch.unique(row).numel()) for row in stimulus.clip_index]
    assert min(spans) > 1, f"batch rows span one clip each ({spans}); there is no leak to fix"

    # 2. map latent rows onto the post-patchify token grid the block's MLP actually sees
    tokens = length // config.patch_size
    clip_groups, straddling = token_clip_index(
        stimulus.clip_index, tokens=tokens, patch_size=config.patch_size
    )
    assert clip_groups.shape == (batch * tokens,)
    batch_row_groups = torch.arange(batch).repeat_interleave(tokens)

    def leaked_fit_rows(groups: torch.Tensor, clip_of_row: torch.Tensor) -> int:
        """Rows on the FIT side whose clip also appears on the HOLDOUT side."""

        hold = _grouped_split(groups)
        held_clips = torch.unique(clip_of_row[hold])
        return int(torch.isin(clip_of_row[~hold], held_clips).sum().item())

    old_leak = leaked_fit_rows(batch_row_groups, clip_groups)
    fit_rows_old = int((~_grouped_split(batch_row_groups)).sum().item())
    assert old_leak > 0, "the old batch-row grouping is supposed to leak; the fixture is wrong"
    print(
        f"        [batch-row grouping: {old_leak}/{fit_rows_old} fit rows "
        f"({100.0 * old_leak / fit_rows_old:.1f}%) sit on a held-out clip]"
    )

    # 3. the runner's own decision function, on this stimulus, must choose the clip
    keep_mask, run_groups, run_name, run_notes = coreset_row_groups(
        stimulus, torch.zeros(batch, tokens, config.hidden_size), patch_size=config.patch_size
    )
    assert run_name == "clip", f"the runner grouped by {run_name}, not by clip"
    assert torch.equal(keep_mask, ~straddling)
    assert torch.equal(run_groups, clip_groups[keep_mask])
    assert run_notes["boundary_tokens_dropped"] == int(straddling.sum().item())
    assert run_notes["rows_used"] == int(keep_mask.sum().item())

    # 4. the fix: group by clip, dropping the tokens whose patch straddles a boundary.
    # Scored against the clips each token's rows ACTUALLY touch, not the id it was filed
    # under, so this is not the tautology "a split by clip splits by clip".
    touched = stimulus.clip_index.reshape(batch, tokens, config.patch_size).reshape(-1, config.patch_size)
    kept = ~straddling
    assert int(straddling.sum().item()) > 0, "no straddling patch in the window; fixture is wrong"

    kept_groups = clip_groups[kept]
    hold = _grouped_split(kept_groups)
    held_clips = torch.unique(kept_groups[hold])
    fit_touching_held = int(torch.isin(touched[kept][~hold], held_clips).any(dim=-1).sum().item())
    assert fit_touching_held == 0, "a held-out clip must have zero rows in the fit"

    # keeping the straddlers and filing each under its first clip breaks exactly that
    hold_all = _grouped_split(clip_groups)
    held_all = torch.unique(clip_groups[hold_all])
    straddler_leak = int(torch.isin(touched[~hold_all], held_all).any(dim=-1).sum().item())
    assert straddler_leak > 0, "the boundary tokens are supposed to leak if kept"
    print(f"        [clip grouping: {int(straddling.sum().item())} boundary tokens dropped, "
          f"{straddler_leak} fit tokens would touch a held-out clip if kept]")

    # 5. and the solver's own receipt reports that split, on the same rows
    block = AceStepDiTLayer(
        config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
        config.head_dim, config.intermediate_size, config.rms_norm_eps,
    )
    rows = torch.randn(
        int(kept.sum().item()), config.hidden_size, generator=torch.Generator().manual_seed(32)
    )
    result = coreset_block_mlp(
        block, rows, keep=config.intermediate_size // 4, seed=0,
        sequence_index=kept_groups, group_name="clip", row_membership=touched[kept],
    )
    split = result.receipt["split_groups"]
    assert result.receipt["split_by"].startswith("clip ("), result.receipt["split_by"]
    assert split["group_name"] == "clip"
    assert split["fit_groups"] + split["holdout_groups"] == split["groups"]
    assert result.receipt["fit_rows"] + result.receipt["holdout_rows"] == int(kept.sum().item())


def test_coreset_receipt_accounts_for_every_group_on_one_side_only() -> None:
    """The receipt must account for every group exactly once, on one side only, and say
    that the membership guard ran. (Comparing the two id sets proves nothing on its own --
    splitting by id makes them disjoint by construction; that is what
    ``test_coreset_membership_guard_fires...`` is for.)"""

    from mediafold.ace_dit import AceStepDiTLayer
    from mediafold.c16_transplant import coreset_block_mlp

    config = _tiny_ace_config()
    block = AceStepDiTLayer(
        config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
        config.head_dim, config.intermediate_size, config.rms_norm_eps,
    )
    keep = config.intermediate_size // 4
    groups, length = 10, 40
    rows = torch.randn(
        groups * length, config.hidden_size, generator=torch.Generator().manual_seed(33)
    )
    index = torch.arange(groups).repeat_interleave(length)
    result = coreset_block_mlp(
        block, rows, keep=keep, seed=0, sequence_index=index, group_name="clip",
        row_membership=index.reshape(-1, 1),
    )
    split = result.receipt["split_groups"]
    assert split["membership_checked"] is True
    assert split["fit_rows_drawing_on_a_held_out_group"] == 0
    assert "NOT statistical independence" in result.receipt["split_guarantee"]
    assert split["groups"] == groups
    assert split["holdout_groups"] >= 1
    assert split["fit_groups"] + split["holdout_groups"] == split["groups"], (
        "a group counted on both sides would make these sum above the total"
    )
    assert result.receipt["holdout_rows"] == split["holdout_groups"] * length

    # the ungrouped path still works and is labelled as the leaky one it is
    row_split = coreset_block_mlp(block, rows, keep=keep, seed=0)
    assert row_split.receipt["split_by"].startswith("row (")
    assert row_split.receipt["split_groups"]["group_name"] == "row"


def test_coreset_row_groups_falls_back_to_the_batch_row_and_says_so() -> None:
    """The random rung has no clip provenance at all, so the batch row is the only
    grouping there is. That is allowed -- but the receipt has to name it, because a
    batch-row holdout on a latents run is exactly the round-2 blocker."""

    from mediafold.c16_capture import coreset_row_groups, random_stimulus

    config = _tiny_ace_config()
    stimulus = random_stimulus(config, batch=3, length=16, seed=7)
    assert stimulus.clip_index is None, "the random rung must not invent clip provenance"

    tokens = 8
    keep_mask, groups, name, notes = coreset_row_groups(
        stimulus, torch.zeros(3, tokens, config.hidden_size), patch_size=config.patch_size
    )
    assert name == "dit_batch_row"
    assert bool(keep_mask.all()) and int(keep_mask.numel()) == 3 * tokens
    assert groups.tolist() == [0] * tokens + [1] * tokens + [2] * tokens
    assert notes["grouped_by"] == "dit_batch_row"
    assert notes["boundary_tokens_dropped"] == 0
    assert "no clip provenance" in notes["reason"]


def test_token_clip_index_marks_every_straddling_patch() -> None:
    """Patchify is kernel = stride = patch_size, so a token owns patch_size consecutive
    latent rows. A token whose rows come from two clips belongs to both and must be
    reported, never silently assigned to the first one."""

    from mediafold.c16_capture import token_clip_index

    # clips of length 3 at patch_size 2: rows (0,0) (0,1) (1,1) (2,2) -- one straddles
    clip_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]])
    groups, mixed = token_clip_index(clip_index, tokens=4, patch_size=2)
    assert groups.tolist() == [0, 0, 1, 2]
    assert mixed.tolist() == [False, True, False, False]

    # an odd row count pads the last patch, which then owns no single clip
    groups, mixed = token_clip_index(torch.tensor([[5, 5, 5]]), tokens=2, patch_size=2)
    assert groups.tolist() == [5, 5]
    assert mixed.tolist() == [False, True]

    # patch_size 1 can never straddle
    groups, mixed = token_clip_index(clip_index, tokens=8, patch_size=1)
    assert not bool(mixed.any())
    assert groups.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]

    # a clip map that does not describe these rows is refused, not reshaped into place
    try:
        token_clip_index(clip_index, tokens=7, patch_size=2)
    except ValueError as error:
        assert "does not describe these rows" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a mismatched token count must be refused")


def test_coreset_membership_guard_fires_when_the_group_id_is_not_the_material() -> None:
    """ROUND-3 BLOCKER (media). The delivered guard compared ``unique(groups[fit])`` with
    ``unique(groups[hold])``, which a split BY those ids makes disjoint by construction --
    it is identically 0 for every possible input and protects nothing. The failure it was
    supposed to catch is the one the lane actually shipped: a group id that is not the
    material it is named after (the DiT batch row called a clip holdout). ``row_membership``
    is that material, derived from the clip map instead of from the ids, and the guard has
    to refuse on it."""

    from mediafold.ace_dit import AceStepDiTLayer
    from mediafold.c16_capture import token_clip_index, token_clip_membership
    from mediafold.c16_transplant import coreset_block_mlp

    config = _tiny_ace_config()
    block = AceStepDiTLayer(
        config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
        config.head_dim, config.intermediate_size, config.rms_norm_eps,
    )
    keep = config.intermediate_size // 4
    # 6 batch rows x 16 latent rows, clips 5 rows long. 5 does not divide 16, so a clip
    # straddles the boundary between two batch rows -- the shipped defect in miniature:
    # holding out a whole batch row still leaves that clip's rows in the fit.
    batch, length, rows_per_clip = 6, 16, 5
    clip_rows = (torch.arange(batch * length) // rows_per_clip).reshape(batch, length)
    tokens = length // config.patch_size
    membership = token_clip_membership(clip_rows, tokens=tokens, patch_size=config.patch_size)
    batch_row_groups = torch.arange(batch).repeat_interleave(tokens)
    rows = torch.randn(
        batch * tokens, config.hidden_size, generator=torch.Generator().manual_seed(41)
    )

    # the old code's split: ids are batch rows, material is clips -> must be refused
    try:
        coreset_block_mlp(
            block, rows, keep=keep, seed=0, sequence_index=batch_row_groups,
            group_name="dit_batch_row", row_membership=membership,
        )
    except RuntimeError as error:
        assert "held-out" in str(error), str(error)
    else:  # pragma: no cover
        raise AssertionError("a fit row drawing on a held-out clip must be refused")

    # and without the membership the SAME split sails through, which is why the argument
    # exists: the id-vs-id comparison cannot fail
    loose = coreset_block_mlp(
        block, rows, keep=keep, seed=0, sequence_index=batch_row_groups,
        group_name="dit_batch_row",
    )
    assert loose.receipt["split_groups"]["membership_checked"] is False
    assert "not evidence" in loose.receipt["split_groups"]["membership_note"]

    # naming a split "clip" without the material to check it against is refused outright
    try:
        coreset_block_mlp(
            block, rows, keep=keep, seed=0, sequence_index=batch_row_groups, group_name="clip",
        )
    except ValueError as error:
        assert "row_membership" in str(error)
    else:  # pragma: no cover
        raise AssertionError("group_name='clip' must require row_membership")

    # the honest split of the same rows passes the guard
    clip_groups, straddling = token_clip_index(
        clip_rows, tokens=tokens, patch_size=config.patch_size
    )
    kept = ~straddling
    good = coreset_block_mlp(
        block, rows[kept], keep=keep, seed=0, sequence_index=clip_groups[kept],
        group_name="clip", row_membership=membership[kept],
    )
    assert good.receipt["split_groups"]["fit_rows_drawing_on_a_held_out_group"] == 0


def test_only_a_clip_aligned_window_makes_the_holdout_independent() -> None:
    """ROUND-3 BLOCKER (media). Grouping by clip removes ROW-LEVEL clip overlap. It does
    not make the holdout independent: ``mlp_in`` is produced after DiT self-attention whose
    receptive field is the whole batch row, so a held-out token's activation is a function
    of every clip in its batch row. Only a window whose batch rows are whole clips fixes
    that, and the stimulus has to say which kind of window it is."""

    from mediafold.c15_pairs import write_pair_set
    from mediafold.c16_capture import coreset_row_groups, latent_stimulus

    config = _tiny_ace_config()
    rows_per_clip, clips = 8, 12
    pairs = _clip_structured_pair_set(
        clips, rows_per_clip, config.audio_acoustic_hidden_dim, seed=17
    )

    with tempfile.TemporaryDirectory() as directory:
        write_pair_set(pairs, Path(directory) / "pairs")

        # aligned: offset on a clip boundary, length = the clip length
        aligned = latent_stimulus(
            config, Path(directory) / "pairs", batch=clips, length=rows_per_clip,
            seed=5, row_offset=0,
        )
        # not aligned: the same corpus, a window that cuts across clips
        ragged = latent_stimulus(
            config, Path(directory) / "pairs", batch=4, length=rows_per_clip * 2 + 1,
            seed=5, row_offset=3,
        )

    assert aligned.notes["row_offset"] == 0 and aligned.notes["row_offset_source"] == "explicit"
    assert ragged.notes["row_offset"] == 3
    assert aligned.notes["clips_per_batch_row_max"] == 1
    assert aligned.notes["batch_rows_are_single_clips"] is True
    assert "independent of the fit" in aligned.notes["attention_receptive_field"]
    assert ragged.notes["clips_per_batch_row_max"] > 1
    assert ragged.notes["batch_rows_are_single_clips"] is False
    assert "NOT independent" in ragged.notes["attention_receptive_field"]

    # and the grouping decision carries the same verdict, next to the grouping it chose
    tokens = rows_per_clip // config.patch_size
    _, groups, name, notes = coreset_row_groups(
        aligned, torch.zeros(clips, tokens, config.hidden_size), patch_size=config.patch_size
    )
    assert name == "clip"
    assert notes["boundary_tokens_dropped"] == 0, "a clip-aligned window straddles nothing"
    assert int(torch.unique(groups).numel()) == clips
    assert notes["receptive_field"]["batch_rows_are_single_clips"] is True
    assert "NOT independence" in notes["guarantee"]

    ragged_tokens = -(-int(ragged.clip_index.shape[1]) // config.patch_size)
    _, _, _, ragged_notes = coreset_row_groups(
        ragged, torch.zeros(4, ragged_tokens, config.hidden_size), patch_size=config.patch_size
    )
    assert ragged_notes["receptive_field"]["batch_rows_are_single_clips"] is False
    assert ragged_notes["boundary_tokens_dropped"] > 0


def test_token_clip_membership_is_every_clip_a_token_draws_on() -> None:
    """The guard needs the clips a token's rows ACTUALLY come from, not the id it was
    filed under, and it has to pad exactly where ``proj_in`` pads."""

    from mediafold.c16_capture import token_clip_index, token_clip_membership

    clip_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]])
    membership = token_clip_membership(clip_index, tokens=4, patch_size=2)
    assert membership.tolist() == [[0, 0], [0, 1], [1, 1], [2, 2]]

    groups, mixed = token_clip_index(clip_index, tokens=4, patch_size=2)
    assert torch.equal(membership[:, 0], groups), "the filed id is the token's first row"
    assert torch.equal(
        (membership != membership[:, :1]).any(dim=-1) | (membership < 0).any(dim=-1), mixed
    ), "a token is mixed exactly when its membership is not a single clip"

    # a trailing partial patch pads with -1 rather than reshaping into a shape error, which
    # is what an odd --length used to produce here
    assert token_clip_membership(torch.tensor([[5, 5, 5]]), tokens=2, patch_size=2).tolist() == [
        [5, 5], [5, -1]
    ]


def test_transplant_receipt_carries_the_holdout_reading() -> None:
    """ROUND-3 BLOCKER (media). The caveat that a clip-grouped holdout is not paired with
    a batch-row-grouped one lived only in two auxiliary files. Nobody opens those to read
    a score: they open ``c16_receipt.json``, see the new rel_l2 next to the old one and
    record an improvement the decomposition says is not there. The caveat has to be IN the
    receipt, and it has to state which window produced the number."""

    import json

    from c16_transplant_run import _holdout_reading

    coreset = {
        "split_by": "clip (93 clips, 19 held out)",
        "holdout_scores": {"rel_l2": 0.55, "cos": 0.86, "rows": 927},
    }
    ragged = _holdout_reading(
        coreset,
        {"receptive_field": {"batch_rows_are_single_clips": False, "clips_per_batch_row_max": 4}},
        None,
    )
    assert "NOT PAIRED" in ragged
    assert "holdout_decomposition.json" in ragged
    assert "NOT clip-aligned" in ragged and "not an unseen-content error" in ragged
    assert "clip (93 clips, 19 held out)" in ragged

    aligned = _holdout_reading(
        coreset,
        {"receptive_field": {"batch_rows_are_single_clips": True, "clips_per_batch_row_max": 1}},
        None,
    )
    assert "NOT PAIRED" in aligned
    assert "unseen-content error" in aligned and "NOT clip-aligned" not in aligned

    # and every clip-grouped receipt this lane has shipped carries it
    found = 0
    for candidate in sorted(_HERE.glob("receipts/*/c16_receipt.json")) + sorted(
        _HERE.glob("out/*/c16_receipt.json")
    ):
        body = json.loads(candidate.read_text(encoding="utf-8"))
        coreset_block = body.get("coreset", {})
        if coreset_block.get("split_groups", {}).get("group_name") != "clip":
            continue
        found += 1
        assert "holdout_reading" in body, f"{candidate} has no holdout_reading"
        assert body["holdout_reading"] == coreset_block.get("reading")
        assert "NOT PAIRED" in body["holdout_reading"], candidate
    if not found:
        raise Skip("no clip-grouped c16 receipt in this checkout")


def test_compose_interface_stand_in_survives_a_non_square_source_projection() -> None:
    """BLOCKER 7. ``output_projection`` defaulted to ``identity(...).transpose(0,1)``, i.e.
    [model_width, latent_width], while the shape check wants [latent_width, model_width].
    It only passed because every delivered call left ``source_projection`` None (square).
    A real parent patch projection -- H3's ``video_patch_proj`` [5376, 96] -- is not."""

    adapter = SyntheticAudioVae(hop_samples=32, channels=1, latent_channels=8, sample_rate_hz=16000)
    clips = procedural_audio(6, samples=32 * 6, channels=1, sample_rate_hz=16000, seed=9)
    pairs = sample_audio_pairs(adapter, clips, seed=9)
    maps = fit_absorption(pairs, seed=9)
    latent = int(maps.encoder.weight.shape[0])
    model_width = 5 * latent  # the non-square case the harness never exercised
    source = torch.randn(model_width, latent, dtype=torch.float64)
    composed = compose_interface(maps, hidden=64, source_projection=source)
    assert "output_projection=identity" in composed.notes["stand_ins"]
    assert "chart_in=identity" in composed.notes["stand_ins"]
    assert "chart_out=identity" in composed.notes["stand_ins"]
    assert composed.feature_proj_weight.shape == (64, pairs.geometry["d_raw"])
    assert composed.emission_out_weight.shape == (pairs.geometry["d_centre"], 64)
    assert composed.notes["model_width_in"] == model_width
    assert composed.notes["model_width_out"] == model_width


def test_merge_refuses_a_measured_video_alignment_mismatch() -> None:
    """BLOCKER 8. Two chunks can share the pairing STRING and own different raw frames."""

    from mediafold.c15_merge import STRICT_KEYS, alignment_block, merge_pair_sets

    for key in ("selected_latent_frames", "skipped_latent_frames", "measured_group_sizes", "temporal_group_size"):
        assert key in STRICT_KEYS, key

    adapter = SyntheticVideoVae(patch=8, tubelet=2, channels=3, latent_channels=6)
    clips = procedural_video(4, frames=4, size=16, seed=1)
    left = sample_video_pairs(adapter, clips, latent_group=1, seed=1)
    right = sample_video_pairs(adapter, clips, latent_group=1, seed=2)
    # both carry the same pairing string; give them different MEASURED maps
    left.geometry = {
        **left.geometry, "pairing": "measured_group4_latent1",
        "temporal_group_size": 4, "selected_latent_frames": [1, 2, 3, 4],
        "skipped_latent_frames": [0, 5], "measured_group_sizes": [1, 4],
    }
    right.geometry = {
        **right.geometry, "pairing": "measured_group4_latent1",
        "temporal_group_size": 4, "selected_latent_frames": [2, 3, 4, 5],
        "skipped_latent_frames": [0, 1], "measured_group_sizes": [1, 3, 4, 5],
    }
    try:
        merge_pair_sets([left, right])
    except ValueError as error:
        assert "selected_latent_frames" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a measured-alignment mismatch must be refused, not concatenated")

    # matching maps merge, and EVERY chunk's map is recorded, not just set[0]'s
    right.geometry = {**left.geometry}
    merged = merge_pair_sets([left, right])
    assert len(merged.geometry["merged_alignments"]) == 2
    assert merged.geometry["merged_alignments"][0] == alignment_block(left.geometry)


def test_h3_0_installs_the_type_embed_delta_as_an_addition() -> None:
    """BLOCKER 2. ``copy_`` wrote one shared bias into every media-type row, so after
    install video (type 2) and audio (type 3) were byte-identical and the bridge could not
    tell them apart -- in the absorbed AND the control arm."""

    source = OPTIONS.get("chimera_src")
    if not source:
        raise Skip("no --chimera-src")
    import inspect

    from h3_0_proof_rung import (
        build_fixture_pairs,
        import_proof,
        install_absorbed_interface,
        per_type_encoder_bias,
    )
    from mediafold.c15_fit import fit_absorption as fit  # noqa: F811

    body = inspect.getsource(install_absorbed_interface)
    assert "type_embed.weight[media_type].copy_" not in body
    assert ".add_(" in body

    proof, _ = import_proof(source)
    pairs, extra = build_fixture_pairs(
        proof, clips=4, audio_adapter_spec="synthetic", video_adapter_spec="synthetic",
        latent_width=32, seed=0,
    )
    maps = fit(pairs, rung="linear", seed=0)
    type_bias = per_type_encoder_bias(pairs, maps, extra["type_rows"])
    assert sorted(type_bias) == [2, 3]
    assert not torch.allclose(type_bias[2], type_bias[3])

    device = torch.device("cpu")
    model = proof.build_model(device)
    before = model.native_media_bridge.type_embed.weight.detach().clone()
    notes = install_absorbed_interface(
        model, maps, type_rows=extra["type_rows"], latent_width=32, scramble=False,
        seed=0, type_bias=type_bias,
    )
    after = model.native_media_bridge.type_embed.weight.detach()
    assert not torch.allclose(after[2], after[3]), "media types must stay distinguishable"
    # it is an ADDITION: every touched row moved by exactly its own delta
    for media_type, record in notes["type_embed_deltas"].items():
        assert record["per_type_bias"] is True
        moved = (after[int(media_type)] - before[int(media_type)]).norm()
        assert abs(float(moved) - record["delta_norm"]) < 1e-3 * max(record["delta_norm"], 1.0)
    assert notes["type_embed_distinct_rows_after_install"] == int(after.shape[0])


def test_h3_0_audio_pairs_record_their_ambiguity_floor() -> None:
    """The proof's audio band MUST stay per-channel (one token per channel per segment),
    so it takes the many-to-one pairing deliberately -- with the floor on the record."""

    source = OPTIONS.get("chimera_src")
    if not source:
        raise Skip("no --chimera-src")
    from h3_0_proof_rung import build_fixture_pairs, import_proof

    proof, _ = import_proof(source)
    pairs, _ = build_fixture_pairs(
        proof, clips=3, audio_adapter_spec="synthetic", video_adapter_spec="synthetic",
        latent_width=32, seed=0,
    )
    audio = pairs.geometry["audio"]
    assert audio["d_centre"] == proof.FEATURE_DIM
    assert audio["ambiguous_pairing"] is True
    assert audio["ambiguity"]["irreducible_rel_l2"] > 0.0
    assert pairs.geometry["video"].get("ambiguous_pairing") in (None, False)


class Skip(Exception):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ace-weights", default=None)
    parser.add_argument("--ace-config", default="/home/cereal/comfyui/models/ACE-Step/acestep-v15-turbo/config.json")
    parser.add_argument("--chimera-src", default=None)
    parser.add_argument("--filter", default="")
    arguments = parser.parse_args(argv)
    OPTIONS["ace_weights"] = arguments.ace_weights
    OPTIONS["ace_config"] = arguments.ace_config
    OPTIONS["chimera_src"] = arguments.chimera_src

    tests: list[tuple[str, Callable[[], None]]] = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value) and arguments.filter in name
    ]
    passed = skipped = failed = 0
    for name, function in tests:
        try:
            function()
        except Skip as reason:
            skipped += 1
            print(f"SKIP {name} ({reason})")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"ok   {name}")
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed  (of {len(tests)})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
