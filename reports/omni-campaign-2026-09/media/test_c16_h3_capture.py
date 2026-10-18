"""pytest for the S6H H3 capture + coreset tools, on tiny shapes (no ComfyUI, no weights).

    cd $OC/media && $PY -m pytest -o addopts="" -p no:cacheprovider test_c16_h3_capture.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mediafold import receipts  # noqa: E402
from mediafold.c16_h3 import (  # noqa: E402
    AUDIO_CLIP_OFFSET,
    AUDIO_WINDOW_STRIDE,
    KIND_AUDIO,
    KIND_TEXT,
    KIND_VIDEO,
    MLP_ARMS,
    TEXT_SOURCE_GAUSSIAN,
    TEXT_SOURCE_STATES,
    audio_clips_from_pairs,
    audio_parent_of,
    build_stimulus,
    concat_captures,
    coreset_row_groups,
    frame_alignment,
    matched_audio_frames,
    mlp_arm_table,
    mlp_reference,
    packed_layout,
    receptive_field_notes,
    shell_from_block_state,
    shifted_sigma_grid,
    split_fc1,
    stimulus_deviations,
    strip_block_prefix,
    time_shift_sigma,
    time_shift_slope,
    uniform_sigma_grid,
    video_clips_from_pairs,
    video_time_span,
    whole_block_receipt,
)
from mediafold.c16_transplant import coreset_block_mlp  # noqa: E402
from mediafold.linalg import law_abiding_solve, prior_anchored  # noqa: E402

HIDDEN, FFN, HEADS, HEAD_DIM = 16, 32, 2, 4


def tiny_block_state(seed: int = 0, index: int = 3) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    inner = HEADS * HEAD_DIM
    p = f"blocks.{index}."
    return {
        f"{p}attn.qkv_proj.weight": torch.randn(3 * inner, HIDDEN, generator=g),
        f"{p}attn.out_proj.weight": torch.randn(HIDDEN, inner, generator=g),
        f"{p}attn.q_norm.weight": torch.ones(HEAD_DIM),
        f"{p}attn.k_norm.weight": torch.ones(HEAD_DIM),
        f"{p}norm1.weight": torch.ones(HIDDEN),
        f"{p}norm2.weight": torch.ones(HIDDEN),
        f"{p}mlp.fc1.weight": torch.randn(2 * FFN, HIDDEN, generator=g) / math.sqrt(HIDDEN),
        f"{p}mlp.fc2.weight": torch.randn(HIDDEN, FFN, generator=g) / math.sqrt(FFN),
        f"{p}adaln_proj.linear.weight": torch.randn(6 * HIDDEN * 3, 8, generator=g),
        f"{p}adaln_proj.linear.bias": torch.zeros(6 * HIDDEN * 3),
    }


# --------------------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------------------


def test_sigma_grids_and_shift_maps():
    grid = shifted_sigma_grid(4)
    assert len(grid) == 4 and all(0.0 < s < 1.0 for s in grid) and grid == sorted(grid)
    assert grid[0] > 0.5  # a shift-12 schedule sits near sigma 1
    assert uniform_sigma_grid(4) == [0.125, 0.375, 0.625, 0.875]
    for s in (0.1, 0.5, 0.9):
        chained = time_shift_sigma(time_shift_sigma(s, 1.0, 12.0), 12.0, 3.0)
        assert abs(chained - time_shift_sigma(s, 1.0, 3.0)) < 1e-12
        assert time_shift_slope(s, 12.0, 3.0) > 0.0
        # audio sigma is below the video sigma on the same base point (shift 3 < 12)
        assert time_shift_sigma(s, 12.0, 3.0) < s
    with pytest.raises(ValueError):
        shifted_sigma_grid(0)


def test_matched_audio_frames_follow_the_video_spans():
    assert abs(video_time_span(5) - 5.0 / 3.0 * 17) < 1e-9
    assert matched_audio_frames(5) == 28
    assert matched_audio_frames(7) == 37
    assert matched_audio_frames(1) == 2


# --------------------------------------------------------------------------------------
# the MLP split
# --------------------------------------------------------------------------------------


def test_fused_swiglu_split_matches_the_reference():
    state = strip_block_prefix(tiny_block_state(), 3)
    shell = shell_from_block_state(state)
    rows = torch.randn(9, HIDDEN)
    expected = mlp_reference(rows, state["mlp.fc1.weight"], state["mlp.fc2.weight"])
    torch.testing.assert_close(shell.mlp(rows), expected, atol=1e-6, rtol=1e-6)
    gate, up = split_fc1(state["mlp.fc1.weight"])
    torch.testing.assert_close(shell.mlp.gate_proj.weight, gate)
    torch.testing.assert_close(shell.mlp.up_proj.weight, up)
    torch.testing.assert_close(shell.fused_state()["mlp.fc1.weight"], state["mlp.fc1.weight"])
    with pytest.raises(KeyError):
        strip_block_prefix({"blocks.3.mlp.fc1.weight": torch.zeros(4, 2)}, 3)


def test_shell_counts_and_ratios():
    state = strip_block_prefix(tiny_block_state(), 3)
    parent = shell_from_block_state(state)
    assert parent.mlp_parameters() == 3 * FFN * HIDDEN
    frozen = sum(int(v.numel()) for k, v in state.items() if not k.startswith("mlp."))
    assert parent.block_parameters() == frozen + parent.mlp_parameters()
    from mediafold.ace_dit import MLP

    child = shell_from_block_state(state)
    child.mlp = MLP(HIDDEN, FFN // 4)
    ratios = whole_block_receipt(parent, child)
    assert ratios["component"]["compression_x"] == pytest.approx(4.0)
    assert ratios["component"]["meets_4x_component_floor"] is True
    assert 1.0 < ratios["whole_block"]["compression_x"] < 4.0
    assert "NOT BUILT" in ratios["kv_head_merge_arm"]


# --------------------------------------------------------------------------------------
# clip latents out of pair sets
# --------------------------------------------------------------------------------------


def _video_pair_set(clips: torch.Tensor, frame_ids: list[int]) -> SimpleNamespace:
    """[N, C, T, H, W] -> the row layout sample_grouped_video_pairs writes."""

    n, c, t, h, w = clips.shape
    latent, clip_index, frame_index = [], [], []
    for clip in range(n):
        for k, frame in enumerate(frame_ids):
            latent.append(clips[clip, :, k].permute(1, 2, 0).reshape(h * w, c))
            clip_index.append(torch.full((h * w,), clip, dtype=torch.long))
            frame_index.append(torch.full((h * w,), frame, dtype=torch.long))
    return SimpleNamespace(
        latent=torch.cat(latent), clip_index=torch.cat(clip_index), frame_index=torch.cat(frame_index),
        channel_index=torch.full((n * t * h * w,), -1, dtype=torch.long), geometry={"modality": "video"},
    )


def _audio_pair_set(clips: torch.Tensor) -> SimpleNamespace:
    """[N, C, ch, T] -> the per_channel row layout sample_audio_pairs writes."""

    n, c, ch, t = clips.shape
    latent, clip_index, frame_index, channel_index = [], [], [], []
    for clip in range(n):
        for channel in range(ch):
            latent.append(clips[clip, :, channel].transpose(0, 1))
            clip_index.append(torch.full((t,), clip, dtype=torch.long))
            frame_index.append(torch.arange(t))
            channel_index.append(torch.full((t,), channel, dtype=torch.long))
    return SimpleNamespace(
        latent=torch.cat(latent), clip_index=torch.cat(clip_index), frame_index=torch.cat(frame_index),
        channel_index=torch.cat(channel_index), geometry={"modality": "audio"},
    )


def test_video_clips_round_trip_through_the_pair_layout():
    clips = torch.randn(3, 24, 5, 4, 4)
    pairs = _video_pair_set(clips, frame_ids=[2, 3, 4, 5, 6])
    rebuilt, notes = video_clips_from_pairs(pairs)
    assert sorted(rebuilt) == [0, 1, 2]
    for clip in range(3):
        torch.testing.assert_close(rebuilt[clip], clips[clip])
    assert notes["grid_hw"] == [4, 4] and notes["latent_frames_per_clip"] == [5]
    with pytest.raises(ValueError):
        video_clips_from_pairs(_video_pair_set(torch.randn(1, 24, 2, 2, 3), [0, 1]))  # 6 cells: not square


def test_audio_clips_round_trip_and_frame_slice():
    clips = torch.randn(2, 32, 2, 10)
    rebuilt, notes = audio_clips_from_pairs(_audio_pair_set(clips))
    torch.testing.assert_close(rebuilt[1], clips[1])
    sliced, _ = audio_clips_from_pairs(_audio_pair_set(clips), frames=4)
    torch.testing.assert_close(sliced[0], clips[0, :, :, :4])
    with pytest.raises(ValueError):
        audio_clips_from_pairs(_audio_pair_set(clips), frames=11)
    stacked = _audio_pair_set(clips)
    stacked.channel_index = torch.full_like(stacked.channel_index, -1)
    with pytest.raises(ValueError):
        audio_clips_from_pairs(stacked)


# --------------------------------------------------------------------------------------
# packed layout + stimulus
# --------------------------------------------------------------------------------------


def test_packed_layout_and_stimulus_provenance():
    layout = packed_layout(5, 7, (24, 3, 4, 6))
    assert layout["video_tokens_thw"] == (3, 2, 3)
    assert layout["seq_len"] == 5 + 14 + 18
    assert [k for _, _, k in layout["segments"]] == ["text", "audio", "video"]
    kind = layout["token_kind"]
    assert int((kind == KIND_TEXT).sum()) == 5 and int((kind == KIND_AUDIO).sum()) == 14 and int((kind == KIND_VIDEO).sum()) == 18
    assert layout["token_frame"][5:19].tolist() == list(range(7)) * 2  # channel-major audio
    assert layout["token_frame"][19:].tolist() == [0] * 6 + [1] * 6 + [2] * 6

    video = torch.randn(24, 3, 4, 6)
    audio = torch.randn(32, 2, 7)
    low = build_stimulus(video, audio, sigma=1e-6, seed=1, text_len=5, clip_id=11, audio_clip_id=4)
    torch.testing.assert_close(low.video[0], video, atol=1e-4, rtol=0)
    assert low.seq_len == layout["seq_len"]
    assert low.sigma_audio == pytest.approx(time_shift_sigma(1e-6, 12.0, 3.0))
    assert low.text.shape == (1, 5, 5120)
    assert set(low.token_clip[low.token_kind == KIND_VIDEO].tolist()) == {11}
    assert set(low.token_clip[low.token_kind == KIND_AUDIO].tolist()) == {AUDIO_CLIP_OFFSET + 4}
    assert set(low.token_clip[low.token_kind == KIND_TEXT].tolist()) == {-1}
    high = build_stimulus(video, audio, sigma=0.9, seed=1, text_len=5, clip_id=11, audio_clip_id=4)
    assert not torch.allclose(high.video[0], video)
    again = build_stimulus(video, audio, sigma=0.9, seed=1, text_len=5, clip_id=11, audio_clip_id=4)
    torch.testing.assert_close(again.video, high.video)  # seeded
    assert receptive_field_notes(high)["forward_is_single_clip_pair"] is True
    assert receptive_field_notes(build_stimulus(video, audio, sigma=0.5, seed=0, text_len=5))["forward_is_single_clip_pair"] is False
    with pytest.raises(ValueError):
        build_stimulus(torch.randn(24, 2, 3, 4), audio, sigma=0.5, seed=0)  # odd H


# --------------------------------------------------------------------------------------
# grouping + the coreset guard
# --------------------------------------------------------------------------------------


def _forward_rows(forward: int, video_clip: int, audio_clip: int, text: int = 2, audio: int = 3, video: int = 4):
    kind = torch.tensor([KIND_TEXT] * text + [KIND_AUDIO] * audio + [KIND_VIDEO] * video, dtype=torch.int8)
    clip = torch.tensor([-1] * text + [AUDIO_CLIP_OFFSET + audio_clip] * audio + [video_clip] * video)
    return kind, clip, torch.full((text + audio + video,), forward, dtype=torch.long)


def test_coreset_row_groups_by_clip_pair_and_by_forward():
    parts = [_forward_rows(0, 10, 20), _forward_rows(1, 11, 21), _forward_rows(2, 12, 22)]
    kind = torch.cat([p[0] for p in parts]); clip = torch.cat([p[1] for p in parts]); fwd = torch.cat([p[2] for p in parts])
    keep, groups, name, membership, notes = coreset_row_groups(kind, clip, fwd)
    assert name == "clip" and int(keep.sum()) == 3 * 7 and notes["text_rows_dropped"] == 6
    assert sorted(torch.unique(groups).tolist()) == [10, 11, 12]
    assert membership.shape == (21, 2)
    assert membership[0].tolist() == [10, AUDIO_CLIP_OFFSET + 20]
    keep_v, groups_v, _, _, _ = coreset_row_groups(kind, clip, fwd, rows="video")
    assert int(keep_v.sum()) == 12 and groups_v.tolist() == [10] * 4 + [11] * 4 + [12] * 4
    # random rung: no provenance -> forward grouping
    keep_r, groups_r, name_r, membership_r, notes_r = coreset_row_groups(kind, torch.full_like(clip, -1), fwd)
    assert name_r == "forward" and membership_r is None and sorted(torch.unique(groups_r).tolist()) == [0, 1, 2]
    with pytest.raises(ValueError):
        coreset_row_groups(kind, clip, fwd, rows="nope")


def _rows_for(shell, forwards: list[tuple[int, int]], per_forward: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    kinds, clips, fwds, rows = [], [], [], []
    for f, (v, a) in enumerate(forwards):
        k, c, i = _forward_rows(f, v, a, text=2, audio=per_forward // 2, video=per_forward - per_forward // 2)
        kinds.append(k); clips.append(c); fwds.append(i)
        rows.append(torch.randn(int(k.numel()), HIDDEN, generator=g))
    return torch.cat(kinds), torch.cat(clips), torch.cat(fwds), torch.cat(rows)


def test_coreset_runs_on_clip_pairs_and_refuses_a_shared_audio_clip():
    shell = shell_from_block_state(strip_block_prefix(tiny_block_state(), 3))
    forwards = [(v, 100 + v) for v in range(8)]
    kind, clip, fwd, rows = _rows_for(shell, forwards, per_forward=12)
    keep, groups, name, membership, _ = coreset_row_groups(kind, clip, fwd)
    folded = coreset_block_mlp(shell, rows[keep], keep=FFN // 4, seed=0, holdout_fraction=0.25, sequence_index=groups, group_name=name, row_membership=membership)
    assert folded.receipt["split_groups"]["membership_checked"] is True
    assert folded.receipt["split_groups"]["holdout_groups"] == 2
    assert folded.receipt["component"]["compression_x"] == pytest.approx(4.0)
    assert folded.block.mlp_parameters() == 3 * (FFN // 4) * HIDDEN
    assert math.isfinite(folded.receipt["holdout_scores"]["rel_l2"])
    # two forwards sharing one audio clip: grouping by the VIDEO clip would put that audio clip
    # on both sides, so `auto` groups by the shared audio axis instead and an explicit
    # group_by="video" is refused before any split is drawn
    shared = [(v, 100 + (v % 4)) for v in range(8)]
    kind, clip, fwd, rows = _rows_for(shell, shared, per_forward=12)
    keep, groups, name, membership, notes = coreset_row_groups(kind, clip, fwd)
    assert notes["group_axis"] == "audio" and notes["audio_clips_with_several_video_partners"] == 4
    folded = coreset_block_mlp(shell, rows[keep], keep=FFN // 4, seed=0, holdout_fraction=0.25, sequence_index=groups, group_name=name, row_membership=membership)
    assert folded.receipt["split_groups"]["membership_checked"] is True
    with pytest.raises(ValueError, match="audio windows"):
        coreset_row_groups(kind, clip, fwd, group_by="video")
    # the coreset's own guard still fires when the split unit is NOT the shared clip (a
    # forward-level split separates two forwards that share an audio clip)
    fired = False
    for seed in range(8):
        try:
            coreset_block_mlp(shell, rows[keep], keep=FFN // 4, seed=seed, holdout_fraction=0.25, sequence_index=fwd[keep], group_name="clip", row_membership=membership)
        except RuntimeError as error:
            assert "held-out" in str(error)
            fired = True
            break
    assert fired, "a shared audio clip across a forward-level split was not refused"


def test_arm_table_shares_the_holdout_and_reproduces_replace():
    from c16_h3_coreset import replicate_group_split

    shell = shell_from_block_state(strip_block_prefix(tiny_block_state(), 3))
    kind, clip, fwd, rows = _rows_for(shell, [(v, 100 + v) for v in range(10)], per_forward=10)
    keep, groups, name, membership, _ = coreset_row_groups(kind, clip, fwd)
    rows = rows[keep]
    folded = coreset_block_mlp(shell, rows, keep=FFN // 4, seed=3, holdout_fraction=0.2, sequence_index=groups, group_name=name, row_membership=membership)
    fit, hold = replicate_group_split(groups, 0.2, 3)
    assert int(hold.numel()) == folded.receipt["holdout_rows"] and int(fit.numel()) == folded.receipt["fit_rows"]
    table = mlp_arm_table(shell, folded.block.mlp, rows, fit, hold, keep=FFN // 4, seed=3)
    assert [t["arm"] for t in table] == list(MLP_ARMS)
    for t in table:
        assert math.isfinite(t["holdout_scores"]["rel_l2"]) and math.isfinite(t["holdout_scores"]["cos"])
        assert t["child_mlp_parameters"] == 3 * (FFN // 4) * HIDDEN
    replace = table[0]
    assert replace["holdout_scores"]["rel_l2"] == pytest.approx(folded.receipt["holdout_scores"]["rel_l2"], abs=1e-9)
    control = next(t for t in table if t["arm"] == "control")
    assert control["norm_matched_to_parent_on_fit_rows"] is True and control["down_scale"] > 0
    # finding 1: prior_only must score the PIVOTED channels, alone or after random_select
    prior_only = next(t for t in table if t["arm"] == "prior_only")
    assert prior_only["selected_channels_sha256"] == receipts.hash_tensor(torch.sort(folded.selection).values)
    assert prior_only["selected_channels_sha256"] == replace["selected_channels_sha256"]
    alone = mlp_arm_table(shell, folded.block.mlp, rows, fit, hold, keep=FFN // 4, seed=3, arms=("prior_only",), selection=folded.selection)
    assert alone[0]["holdout_scores"]["rel_l2"] == pytest.approx(prior_only["holdout_scores"]["rel_l2"], abs=1e-12)


# --------------------------------------------------------------------------------------
# capture files
# --------------------------------------------------------------------------------------


def test_capture_files_concatenate_with_global_forward_ids(tmp_path):
    def step_state(forwards: int, sigma: float, seed: int):
        kinds, clips, fwds = [], [], []
        for f in range(forwards):
            k, c, i = _forward_rows(f, f, 50 + f)
            kinds.append(k); clips.append(c); fwds.append(i)
        kind = torch.cat(kinds)
        n = int(kind.numel())
        g = torch.Generator().manual_seed(seed)
        state = {name: torch.randn(n, HIDDEN, generator=g).to(torch.bfloat16) for name in ("x_in", "x_out", "mlp_in", "mlp_out", "attn_out")}
        state.update({"token_kind": kind, "token_frame": torch.zeros(n, dtype=torch.long), "token_clip": torch.cat(clips), "forward_index": torch.cat(fwds)})
        state["sigma_video"] = torch.full((forwards,), sigma)
        state["sigma_audio"] = torch.full((forwards,), sigma / 2)
        state["temb"] = torch.zeros(forwards, 2, 8)
        state["adaln"] = torch.zeros(forwards, 6, 6, HIDDEN)
        state["mod_segments"] = torch.zeros(forwards, 3, 3, dtype=torch.long)
        return state

    a = step_state(2, 0.9, 1)
    b = step_state(3, 0.6, 2)
    ra = receipts.write_tensors(a, tmp_path / "step00" / "L03.safetensors")
    rb = receipts.write_tensors(b, tmp_path / "step01" / "L03.safetensors")
    assert ra["file_sha256"] != rb["file_sha256"]
    merged = concat_captures([receipts.read_tensors(tmp_path / "step00" / "L03.safetensors"), receipts.read_tensors(tmp_path / "step01" / "L03.safetensors")])
    assert int(merged["mlp_in"].shape[0]) == 5 * 9
    assert sorted(torch.unique(merged["forward_index"]).tolist()) == [0, 1, 2, 3, 4]
    assert merged["sigma_video_rows"][:18].tolist() == pytest.approx([0.9] * 18)
    assert merged["sigma_video_rows"][18:].tolist() == pytest.approx([0.6] * 27)
    assert merged["step_rows"].tolist() == [0] * 18 + [1] * 27
    with pytest.raises(ValueError):
        concat_captures([])
    # the same clip pairs recur at every sigma step: that is NOT a shared clip, the auto
    # axis stays video and the membership guard holds across steps (regression: the first
    # version counted forwards instead of partners and refused every multi-sigma capture)
    keep, groups, name, membership, notes = coreset_row_groups(merged["token_kind"], merged["token_clip"], merged["forward_index"])
    assert name == "clip" and notes["group_axis"] == "video"
    assert notes["video_clips_with_several_audio_partners"] == 0 and notes["audio_clips_with_several_video_partners"] == 0
    assert sorted(torch.unique(groups).tolist()) == [0, 1, 2]
    assert sorted(torch.unique(merged["forward_index"][keep]).tolist()) == [0, 1, 2, 3, 4]
    shell = shell_from_block_state(strip_block_prefix(tiny_block_state(), 3))
    rows = torch.randn(int(keep.sum()), HIDDEN)
    folded = coreset_block_mlp(shell, rows, keep=FFN // 4, seed=0, holdout_fraction=0.34, sequence_index=groups, group_name=name, row_membership=membership, min_fit_rows_per_kept_channel=1.0)
    assert folded.receipt["split_groups"]["membership_checked"] is True and folded.receipt["split_groups"]["holdout_groups"] == 1


# --------------------------------------------------------------------------------------
# fixes from the s6h_h3_capture review (one test per finding)
# --------------------------------------------------------------------------------------


def test_prior_only_scores_the_pivoted_channels_not_random_ones():
    """Finding 1: random_select used to rebind `selection`, so prior_only scored random channels."""

    from mediafold.ace_dit import MLP
    from mediafold.c16_h3 import _narrowed_mlp

    shell = shell_from_block_state(strip_block_prefix(tiny_block_state(), 3))
    kind, clip, fwd, rows = _rows_for(shell, [(v, 100 + v) for v in range(10)], per_forward=10)
    keep, groups, name, membership, _ = coreset_row_groups(kind, clip, fwd)
    rows = rows[keep]
    folded = coreset_block_mlp(shell, rows, keep=FFN // 4, seed=3, holdout_fraction=0.2, sequence_index=groups, group_name=name, row_membership=membership)
    from c16_h3_coreset import replicate_group_split

    fit, hold = replicate_group_split(groups, 0.2, 3)
    full = mlp_arm_table(shell, folded.block.mlp, rows, fit, hold, keep=FFN // 4, seed=3, selection=folded.selection)
    alone = mlp_arm_table(shell, folded.block.mlp, rows, fit, hold, keep=FFN // 4, seed=3, arms=("prior_only",), selection=folded.selection)
    prior_full = next(t for t in full if t["arm"] == "prior_only")
    assert prior_full["holdout_scores"]["rel_l2"] == alone[0]["holdout_scores"]["rel_l2"]
    assert prior_full["fit_scores"]["rel_l2"] == alone[0]["fit_scores"]["rel_l2"]
    # the prior_only child's gate rows ARE the parent rows at the pivoted selection
    pivoted = torch.sort(folded.selection).values
    child = _narrowed_mlp(shell.mlp, pivoted, None)
    torch.testing.assert_close(child.gate_proj.weight, shell.mlp.gate_proj.weight[pivoted])
    torch.testing.assert_close(child.down_proj.weight, shell.mlp.down_proj.weight[:, pivoted])
    with torch.no_grad():
        expected = child(rows).to(torch.float64)
        target = shell.mlp(rows).to(torch.float64)
    from mediafold.linalg import write_scores

    assert write_scores(expected[hold], target[hold])["rel_l2"] == pytest.approx(prior_full["holdout_scores"]["rel_l2"], abs=1e-12)
    # and random_select's channels are its own (recorded, and not the pivoted set)
    random_arm = next(t for t in full if t["arm"] == "random_select")
    assert random_arm["selected_channels_sha256"] != prior_full["selected_channels_sha256"]
    assert 0 <= random_arm["channels_shared_with_pivoted"] < FFN // 4
    # a child that is not verbatim rows is refused when no selection is given
    with pytest.raises(ValueError):
        mlp_arm_table(shell, MLP(HIDDEN, FFN // 4), rows, fit, hold, keep=FFN // 4, arms=("prior_only",))


def test_frame_alignment_and_the_leading_run_policy():
    """Finding 2 (video): [1,2,3,4,6] has a hole and does not start at latent frame 0."""

    report = frame_alignment([1, 2, 3, 4, 6])
    assert report["consecutive"] is False and report["starts_at_zero"] is False and report["aligned"] is False
    assert report["gaps"] == [[4, 6]] and report["leading_run"] == [1, 2, 3, 4]
    assert frame_alignment([0, 1, 2, 3, 4])["aligned"] is True
    assert frame_alignment([1, 2, 3])["consecutive"] is True and frame_alignment([1, 2, 3])["aligned"] is False
    clips = torch.randn(3, 24, 5, 4, 4)
    pairs = _video_pair_set(clips, frame_ids=[1, 2, 3, 4, 6])
    whole, notes_all = video_clips_from_pairs(pairs, frames="all")
    assert notes_all["frame_alignment"]["aligned"] is False and notes_all["latent_frames_per_clip"] == [5]
    run, notes_run = video_clips_from_pairs(pairs, frames="leading-run")
    assert notes_run["latent_frames_per_clip"] == [4] and notes_run["latent_frame_ids_used"] == [1, 2, 3, 4]
    assert notes_run["frame_alignment"]["consecutive"] is True and notes_run["frame_alignment"]["starts_at_zero"] is False
    assert notes_run["latent_frame_ids_in_file"] == [1, 2, 3, 4, 6] and notes_run["frame_policy"] == "leading-run"
    torch.testing.assert_close(run[2], clips[2, :, :4])
    with pytest.raises(ValueError):
        video_clips_from_pairs(pairs, frames="nope")
    # clips with different id lists are refused (one policy, one geometry per corpus)
    mixed = _video_pair_set(clips[:1], frame_ids=[0, 1, 2, 3, 4])
    other = _video_pair_set(clips[1:2], frame_ids=[1, 2, 3, 4, 6])
    other.clip_index = other.clip_index + 1
    merged = SimpleNamespace(
        latent=torch.cat((mixed.latent, other.latent)), clip_index=torch.cat((mixed.clip_index, other.clip_index)),
        frame_index=torch.cat((mixed.frame_index, other.frame_index)), channel_index=torch.cat((mixed.channel_index, other.channel_index)), geometry={},
    )
    with pytest.raises(ValueError, match="differ"):
        video_clips_from_pairs(merged)


def test_audio_windows_carry_parent_ids_and_the_holdout_groups_by_audio():
    """Finding 2 (audio span): matched windows, parent provenance, audio-grouped split with the guard."""

    clips = torch.randn(3, 32, 2, 10)
    windows, notes = audio_clips_from_pairs(_audio_pair_set(clips), frames=4, windows=0)
    assert sorted(windows) == [0, 1, 100, 101, 200, 201] and notes["windows_per_parent"] == [2] and notes["parent_clips"] == 3
    torch.testing.assert_close(windows[101], clips[1, :, :, 4:8])
    assert audio_parent_of(101, notes["parent_of"]) == 1 and audio_parent_of(7, None) == 7
    assert notes["frames_per_clip"] == [4] and notes["frames_in_file"] == [10]
    with pytest.raises(ValueError):
        audio_clips_from_pairs(_audio_pair_set(clips), frames=4, windows=3)  # only 2 fit
    with pytest.raises(ValueError):
        audio_clips_from_pairs(_audio_pair_set(clips), frames=0, windows=2)  # windows need a length
    one, notes_one = audio_clips_from_pairs(_audio_pair_set(clips), frames=4, windows=1)
    assert sorted(one) == [0, 1, 2] and notes_one["window_id"] == "parent clip id"

    video = torch.randn(24, 3, 4, 6)
    stimulus = build_stimulus(video, windows[101], sigma=0.5, seed=0, text_len=3, clip_id=7, audio_clip_id=101, audio_parent_clip_id=1)
    assert set(stimulus.token_clip[stimulus.token_kind == KIND_AUDIO].tolist()) == {AUDIO_CLIP_OFFSET + 1}
    assert stimulus.notes["audio_clip_id"] == 101 and stimulus.notes["audio_parent_clip_id"] == 1 and stimulus.notes["audio_window_of_parent"] is True

    # 8 forwards: video clips 0..7 (unique), audio parents 0..3 (two windows each)
    shell = shell_from_block_state(strip_block_prefix(tiny_block_state(), 3))
    forwards = [(v, 100 + v // 2) for v in range(8)]
    kind, clip, fwd, rows = _rows_for(shell, forwards, per_forward=12)
    keep, groups, name, membership, group_notes = coreset_row_groups(kind, clip, fwd)
    assert group_notes["group_axis"] == "audio" and group_notes["audio_clips_with_several_video_partners"] == 4
    assert sorted(torch.unique(groups).tolist()) == [AUDIO_CLIP_OFFSET + 100 + k for k in range(4)]
    with pytest.raises(ValueError, match="audio windows"):
        coreset_row_groups(kind, clip, fwd, group_by="video")
    folded = coreset_block_mlp(shell, rows[keep], keep=FFN // 4, seed=0, holdout_fraction=0.25, sequence_index=groups, group_name=name, row_membership=membership)
    assert folded.receipt["split_groups"]["membership_checked"] is True and folded.receipt["split_groups"]["holdout_groups"] == 1
    # the same rows split by FORWARD would put two windows of one parent on opposite sides: the guard fires
    with pytest.raises(RuntimeError, match="held-out"):
        for seed in range(8):
            coreset_block_mlp(shell, rows[keep], keep=FFN // 4, seed=seed, holdout_fraction=0.25, sequence_index=fwd[keep], group_name="clip", row_membership=membership)
    with pytest.raises(ValueError):
        coreset_row_groups(kind, clip, fwd, group_by="nope")


def test_arm_table_reports_per_kind_scores():
    """Finding 3: a pooled av score hides the video rows; every arm scores audio and video separately."""

    shell = shell_from_block_state(strip_block_prefix(tiny_block_state(), 3))
    kind, clip, fwd, rows = _rows_for(shell, [(v, 100 + v) for v in range(10)], per_forward=10)
    keep, groups, name, membership, _ = coreset_row_groups(kind, clip, fwd)
    rows, kinds = rows[keep], kind[keep].to(torch.long)
    folded = coreset_block_mlp(shell, rows, keep=FFN // 4, seed=3, holdout_fraction=0.2, sequence_index=groups, group_name=name, row_membership=membership)
    from c16_h3_coreset import replicate_group_split

    fit, hold = replicate_group_split(groups, 0.2, 3)
    table = mlp_arm_table(shell, folded.block.mlp, rows, fit, hold, keep=FFN // 4, seed=3, selection=folded.selection, token_kind=kinds)
    for t in table:
        by_kind = t["holdout_scores_by_kind"]
        assert set(by_kind) == {"audio", "video"} and "text" not in by_kind
        assert by_kind["audio"]["rows"] + by_kind["video"]["rows"] == t["holdout_scores"]["rows"]
        assert by_kind["audio"]["rows"] == int((kinds[hold] == KIND_AUDIO).sum())
        assert all(math.isfinite(by_kind[k]["rel_l2"]) for k in by_kind)
        assert set(t["fit_scores_by_kind"]) == {"audio", "video"}
    with pytest.raises(ValueError):
        mlp_arm_table(shell, folded.block.mlp, rows, fit, hold, keep=FFN // 4, seed=3, selection=folded.selection, token_kind=kinds[:-1])


def test_stimulus_deviations_are_named_and_gate_the_capture(monkeypatch, tmp_path):
    """Finding 4 (+2): gaussian text, off-span audio, misaligned frames and the missing S1 stimuli are named; the run refuses without --allow-deviations."""

    import c16_h3_capture as capture

    off = stimulus_deviations(frame_alignment_used=frame_alignment([1, 2, 3, 4, 6]), audio_frames_used=80, matched_audio_frames_for_video=28, text_source=TEXT_SOURCE_GAUSSIAN)
    assert [d["name"] for d in off] == ["video_frame_alignment", "audio_span", "text_conditioning", "s1_media_stimuli"]
    assert "80 audio latent frames" in off[1]["what"] and "28" in off[1]["what"]
    on = stimulus_deviations(frame_alignment_used=frame_alignment([0, 1, 2, 3, 4]), audio_frames_used=28, matched_audio_frames_for_video=28, text_source=TEXT_SOURCE_STATES, s1_media_stimuli="present")
    assert on == []
    with pytest.raises(SystemExit, match="text_conditioning"):
        capture.refuse_deviations(off, allow=False)
    capture.refuse_deviations(off, allow=True)
    capture.refuse_deviations([], allow=False)

    # build_stimulus with supplied text states records the states source
    video = torch.randn(24, 3, 4, 6)
    audio = torch.randn(32, 2, 7)
    states = torch.randn(5, 5120)
    stimulus = build_stimulus(video, audio, sigma=0.5, seed=0, text_len=99, text_states=states)
    torch.testing.assert_close(stimulus.text[0], states)
    assert stimulus.notes["text_len"] == 5 and stimulus.notes["text_source"] == TEXT_SOURCE_STATES
    with pytest.raises(ValueError):
        build_stimulus(video, audio, sigma=0.5, seed=0, text_states=torch.randn(5, 64))

    # build_stimuli end to end on tiny corpora: matched span, windows, parents, deviations
    clips_v = torch.randn(4, 24, 5, 4, 4)
    clips_a = torch.randn(2, 32, 2, 80)
    corpora = {"V": _video_pair_set(clips_v, frame_ids=[1, 2, 3, 4, 6]), "A": _audio_pair_set(clips_a)}
    import mediafold.c15_pairs as c15_pairs

    monkeypatch.setattr(c15_pairs, "load_pair_set", lambda path: corpora[str(path)])
    arguments = capture.build_parser().parse_args(["--out", str(tmp_path), "--video-pairs", "V", "--audio-pairs", "A", "--steps", "2", "--text-len", "3"])
    plan, notes = capture.build_stimuli(arguments, [0.5, 0.9])
    matched = matched_audio_frames(4)  # leading run [1,2,3,4] -> 4 latent frames -> 22 audio frames
    assert notes["audio_frames_used"] == matched == 22 and notes["matched_audio_frames_for_video"] == matched
    assert notes["audio_clips"]["windows_per_parent"] == [80 // matched] and notes["clip_pairs_used"] == 4
    assert notes["stimulus_arms"] == {"video_frames": "leading-run", "audio_span": "matched", "audio_windows_per_parent": [3], "text": "gaussian"}
    assert [d["name"] for d in notes["deviations"]] == ["video_frame_alignment", "text_conditioning", "s1_media_stimuli"]
    assert notes["deviations"][0]["what"].startswith("video clips are latent frames [1, 2, 3, 4]")
    assert len(plan) == 2 and len(plan[0]) == 4 and plan[0][0].seq_len == 3 + 2 * matched + 4 * 2 * 2
    parents = {s.audio_clip_id for s in plan[0]}
    assert parents == {0, 1} and {s.notes["audio_clip_id"] for s in plan[0]} == {0, 1, 2, 100}
    whole = capture.build_parser().parse_args(["--out", str(tmp_path), "--video-pairs", "V", "--audio-pairs", "A", "--audio-frames", "0", "--video-frames", "all"])
    _, notes_whole = capture.build_stimuli(whole, [0.5])
    assert [d["name"] for d in notes_whole["deviations"]] == ["video_frame_alignment", "audio_span", "text_conditioning", "s1_media_stimuli"]
    assert notes_whole["audio_frames_used"] == 80 and notes_whole["stimulus_arms"]["audio_span"] == "whole_clip"


def test_child_file_counts_only_mlp_weights_and_the_s7_gap_is_named(tmp_path):
    """Finding 5: one number per file (selection out of mlp.safetensors) and the charted-container gap named."""

    from c16_h3_coreset import s7_consumability, write_child

    state = strip_block_prefix(tiny_block_state(), 3)
    shell = shell_from_block_state(state)
    kind, clip, fwd, rows = _rows_for(shell, [(v, 100 + v) for v in range(10)], per_forward=10)
    keep, groups, name, membership, _ = coreset_row_groups(kind, clip, fwd)
    folded = coreset_block_mlp(shell, rows[keep], keep=FFN // 4, seed=0, holdout_fraction=0.2, sequence_index=groups, group_name=name, row_membership=membership)
    written, selection_written = write_child(folded, 3, tmp_path)
    assert written["parameters"] == 3 * (FFN // 4) * HIDDEN == folded.block.mlp_parameters()
    assert sorted(written["tensors"]) == ["blocks.3.mlp.fc1.weight", "blocks.3.mlp.fc2.weight"]
    assert selection_written["path"].endswith("selection.safetensors") and selection_written["tensors"]["selection"]["shape"] == [FFN // 4]
    back = receipts.read_tensors(tmp_path / "selection.safetensors")["selection"]
    assert back.tolist() == folded.selection.tolist()
    gap = s7_consumability(state, keep=FFN // 4)
    assert gap["consumable_by_c16_install_media_blocks"] is False
    assert set(gap["spec_extensions_required"]) == {"modulation", "time_embed", "attention_width", "rope"}
    assert "modality" in gap["spec_extensions_required"]["modulation"] and "price" in gap


def test_launch_recipe_runs_as_the_invoking_user():
    """Finding 6: the receipt copy under $OC must not be root-owned."""

    import c16_h3_capture as capture

    assert "--user $(id -u):$(id -g)" in capture.__doc__
    assert "-e HOME=/tmp" in capture.__doc__


def test_rerun_refuses_without_overwrite_and_hash_source_defaults_on(tmp_path):
    """Finding 7: idempotency guard + the source hash on by default."""

    import c16_h3_capture as capture

    capture.refuse_existing_capture(tmp_path, overwrite=False)  # nothing there yet
    (tmp_path / "capture_receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="overwrite"):
        capture.refuse_existing_capture(tmp_path, overwrite=False)
    capture.refuse_existing_capture(tmp_path, overwrite=True)
    with pytest.raises(SystemExit, match="overwrite"):
        capture.main(["--out", str(tmp_path)])
    parsed = capture.build_parser().parse_args(["--out", "x"])
    assert parsed.hash_source is True and parsed.overwrite is False and parsed.allow_deviations is False
    assert parsed.audio_frames == -1 and parsed.audio_windows == 0 and parsed.video_frames == "leading-run"
    assert capture.build_parser().parse_args(["--out", "x", "--no-hash-source"]).hash_source is False


def test_explicit_sigmas_override_the_grid():
    """Finding 8: the 4-point shift-12 grid never goes below 0.63; an explicit / uniform grid is an arm."""

    import c16_h3_capture as capture

    assert capture.parse_sigmas("0.2,0.5,0.8") == [0.2, 0.5, 0.8]
    for bad in ("0.5,0.2", "1.0", "0", "0.3,0.3", ""):
        with pytest.raises(Exception):
            capture.parse_sigmas(bad)
    explicit = capture.build_parser().parse_args(["--out", "x", "--sigmas", "0.1,0.4"])
    assert capture.sigma_plan(explicit) == ([0.1, 0.4], "explicit")
    uniform = capture.build_parser().parse_args(["--out", "x", "--sigma-grid", "uniform", "--steps", "4"])
    assert capture.sigma_plan(uniform) == ([0.125, 0.375, 0.625, 0.875], "uniform")
    shifted = capture.build_parser().parse_args(["--out", "x", "--steps", "4"])
    sigmas, name = capture.sigma_plan(shifted)
    assert name == "shifted" and min(sigmas) > 0.6


def test_prior_anchored_names_its_solver():
    """Finding 9: a Gram defect must be named; the receipt carries cholesky vs lstsq_fallback."""

    g = torch.Generator().manual_seed(0)
    design = torch.randn(40, 6, generator=g, dtype=torch.float64)
    prior = torch.randn(3, 6, generator=g, dtype=torch.float64)
    targets = design @ prior.T + 0.01 * torch.randn(40, 3, generator=g, dtype=torch.float64)
    _, notes = prior_anchored(design, targets, prior, ridge_rel=1e-6)
    assert notes["solver"] == "cholesky" and "gram_defect" not in notes
    # rank-deficient design + no ridge: the shifted Gram is singular -> named fallback
    rank_deficient = design.clone()
    rank_deficient[:, 5] = rank_deficient[:, 4]
    weight, notes = prior_anchored(rank_deficient, targets, prior, ridge_rel=0.0)
    assert notes["solver"] == "lstsq_fallback" and "cholesky failed" in notes["gram_defect"]
    assert torch.isfinite(weight).all()
    _, solve = law_abiding_solve(design, targets, prior)
    assert solve["solver"] == "cholesky"
