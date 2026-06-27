from __future__ import annotations

import pytest

from neural_forge.cli import build_parser
from neural_forge.cli import _parse_layer_groups
from neural_forge.hf_recovery import RECOVERY_TEXTS
from neural_forge.hf_recovery import _assert_same_vocab
from neural_forge.hf_recovery import _absorb_input_map_into_linears
from neural_forge.hf_recovery import _candidate_index_after_depth_fold
from neural_forge.hf_recovery import _cross_tokenizer_recovery_loss
from neural_forge.hf_recovery import _cross_tokenizer_span_pairs
from neural_forge.hf_recovery import _fit_activation_map
from neural_forge.hf_recovery import _fit_shared_embedding_hidden_projection
from neural_forge.hf_recovery import _fold_groups_from_report
from neural_forge.hf_recovery import _mlp_projection_spec
from neural_forge.hf_recovery import _recovery_loss
from neural_forge.hf_recovery import _set_train_scope
from neural_forge.hf_recovery import load_calibration_texts
from neural_forge.hf_tools import TEXT_EVALS
from neural_forge.hf_tools import _layer_groups_for_report
from neural_forge.hf_tools import _layer_groups_tag
from neural_forge.hf_tools import _linear_cka
from neural_forge.hf_tools import _normalize_layer_groups
from neural_forge.hf_xfamily import _canonical_token
from neural_forge.hf_xfamily import _canonical_token_map
from neural_forge.hf_xfamily import _assign_donor_to_host
from neural_forge.hf_xfamily import _activation_similarity_assignment
from neural_forge.hf_xfamily import _activation_axis_weights_for_tensor
from neural_forge.hf_xfamily import _balanced_activation_assignment
from neural_forge.hf_xfamily import _channel_assignment_features
from neural_forge.hf_xfamily import _donor_layer_group_for_target
from neural_forge.hf_xfamily import _effective_layer_alpha
from neural_forge.hf_xfamily import _mlp_spec
from neural_forge.hf_xfamily import _proportional_index
from neural_forge.hf_xfamily import _resize_tensor_to_shape
from neural_forge.hf_xfamily import _scale_projection_to_target
from neural_forge.hf_xfamily import _select_compression_candidates
from neural_forge.hf_xfamily import _usable_pipeline_candidates
from neural_forge.hf_xfamily import _validate_channel_assignment
from neural_forge.hf_xfamily import _validate_layer_alpha_settings
from neural_forge.hf_xfamily import sweep_cross_family_vocab_merge
from neural_forge.hf_xfamily import sweep_cross_family_mlp_merge


class _Tokenizer:
    all_special_ids = [0]

    def __init__(self, decoded: list[str]) -> None:
        self._decoded = decoded

    def __len__(self) -> int:
        return len(self._decoded)

    def decode(self, ids: list[int], clean_up_tokenization_spaces: bool = False) -> str:
        assert clean_up_tokenization_spaces is False
        return self._decoded[ids[0]]

    def get_vocab(self) -> dict[str, int]:
        return {text: idx for idx, text in enumerate(self._decoded)}


class _Obj:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


def test_canonical_token_normalizes_common_tokenizer_markers() -> None:
    assert _canonical_token("Ġhello") == "hello"
    assert _canonical_token("\u2581world") == "world"
    assert _canonical_token("  text ") == "text"
    assert _canonical_token("") == ""
    assert _canonical_token("\n") == ""


def test_canonical_token_map_skips_special_and_empty_tokens() -> None:
    tokenizer = _Tokenizer(["<eos>", "Ġhello", "", "\u2581world", "hello"])
    mapping = _canonical_token_map(tokenizer)
    assert mapping == {"hello": 1, "world": 3}


def test_recovery_same_vocab_guard_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="matching tokenizers"):
        _assert_same_vocab(_Tokenizer(["a", "b"]), _Tokenizer(["a", "c"]))


def test_recovery_defaults_do_not_reuse_evaluation_prompts() -> None:
    assert set(RECOVERY_TEXTS).isdisjoint(TEXT_EVALS)


def test_load_calibration_texts_handles_jsonl_message_rows(tmp_path) -> None:
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        "\n".join(
            [
                '{"messages":[{"role":"user","content":"Explain folding."},{"role":"assistant","content":"Fold redundant channels before pruning."}]}',
                '{"prompt":"Solve 2+2.","target":"The answer is 4."}',
                '{"input_json":{"messages":[{"role":"user","content":"Return compact JSON."}]},"target_json":{"answer":"ok"}}',
            ]
        ),
        encoding="utf-8",
    )
    texts = load_calibration_texts(calibration, max_texts=2, min_chars=8)
    assert len(texts) == 2
    assert "user: Explain folding." in texts[0]
    assert "assistant: Fold redundant channels" in texts[0]
    assert texts[1] == "Solve 2+2. The answer is 4."


def test_cross_family_sweep_rejects_duplicate_alpha_tags(tmp_path) -> None:
    with pytest.raises(ValueError, match="Duplicate alpha output tag"):
        sweep_cross_family_vocab_merge(
            host_model="host",
            donor_model="donor",
            output_dir=tmp_path,
            alphas=[0.1, 0.10001],
        )


def test_cross_family_mlp_sweep_rejects_duplicate_alpha_tags(tmp_path) -> None:
    with pytest.raises(ValueError, match="Duplicate alpha output tag"):
        sweep_cross_family_mlp_merge(
            host_model="host",
            donor_model="donor",
            output_dir=tmp_path,
            alphas=[0.1, 0.10001],
        )


def test_channel_assignment_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="channel_assignment"):
        _validate_channel_assignment("blind-average")


def test_similarity_layer_alpha_scales_with_assignment_quality() -> None:
    alpha, scale = _effective_layer_alpha(
        0.01,
        0.5,
        layer_alpha_mode="similarity",
        min_scale=0.25,
        max_scale=1.25,
        similarity_floor=0.25,
    )
    assert scale == pytest.approx(0.25 + 1.0 * (0.25 / 0.75))
    assert alpha == pytest.approx(0.01 * scale)
    assert _effective_layer_alpha(
        0.01,
        0.1,
        layer_alpha_mode="similarity",
        min_scale=0.25,
        max_scale=1.25,
        similarity_floor=0.25,
    ) == pytest.approx((0.0025, 0.25))
    assert _effective_layer_alpha(
        0.01,
        0.5,
        layer_alpha_mode="constant",
        min_scale=0.0,
        max_scale=2.0,
        similarity_floor=0.0,
    ) == pytest.approx((0.01, 1.0))


def test_layer_alpha_settings_reject_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="layer_alpha_mode"):
        _validate_layer_alpha_settings("random", 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="min_scale"):
        _validate_layer_alpha_settings("similarity", 2.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="similarity_floor"):
        _validate_layer_alpha_settings("similarity", 0.0, 1.0, 1.0)


def test_activation_channel_assignment_uses_activation_features() -> None:
    torch = pytest.importorskip("torch")
    host_weight = torch.eye(2)
    donor_weight = torch.eye(2)
    host_activation = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    donor_activation = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    host_features, donor_features, used = _channel_assignment_features(
        torch,
        "activation-correlation",
        host_weight,
        donor_weight,
        host_activation,
        donor_activation,
    )
    assert used is True
    assert _assign_donor_to_host(torch, donor_features, host_features).tolist() == [1, 0]


def test_cli_accepts_cross_family_sweep_and_topk_resize() -> None:
    parser = build_parser()
    sweep_args = parser.parse_args(
        [
            "hf-sweep-xfamily",
            "out",
            "--host-model",
            "host",
            "--donor-model",
            "donor",
            "--alphas",
            "0.005,0.01",
        ]
    )
    resize_args = parser.parse_args(["hf-resize-ffn", "in", "out", "--ratio", "0.95", "--assignment", "topk"])
    random_args = parser.parse_args(
        [
            "hf-random-init",
            "random-out",
            "--config-model",
            "HuggingFaceTB/SmolLM2-135M",
            "--tokenizer-model",
            "Qwen/Qwen3-4B-Base",
            "--seed",
            "123",
            "--dtype",
            "float32",
        ]
    )
    donor_init_args = parser.parse_args(
        [
            "hf-donor-project-init",
            "projected-out",
            "--target-config-model",
            "HuggingFaceTB/SmolLM2-135M",
            "--target-tokenizer-model",
            "HuggingFaceTB/SmolLM2-135M",
            "--donor-model",
            "Qwen/Qwen3-4B-Base",
            "--seed",
            "456",
            "--max-anchors",
            "1000",
            "--fold-mode",
            "mean",
            "--activation-fold-assignment",
            "similarity",
            "--donor-layer-fold-mode",
            "fold",
            "--projection-scale",
            "target-rms",
            "--activation-calibration-file",
            "activation.jsonl",
            "--activation-max-calibration-texts",
            "5",
            "--activation-max-length",
            "80",
            "--activation-device",
            "cpu",
            "--dtype",
            "float32",
        ]
    )
    merge_layer_args = parser.parse_args(["hf-merge-layers", "in", "out", "--groups", "3+4,7+8"])
    rank_fold_args = parser.parse_args(
        [
            "hf-rank-layer-folds",
            "model",
            "fold_rank.json",
            "--window",
            "3",
            "--top-k",
            "5",
            "--max-length",
            "96",
            "--max-tokens-per-layer",
            "77",
            "--dtype",
            "float16",
            "--device",
            "cpu",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "8",
        ]
    )
    drop_args = parser.parse_args(["hf-drop-layers", "in", "out", "--layers", "3,7"])
    search_args = parser.parse_args(
        [
            "hf-search-compress",
            "model",
            "out",
            "--width-ratios",
            "0.9975,0.995",
            "--layer-groups",
            "13+14",
            "--target-shrink",
            "0.02",
        ]
    )
    xfamily_mlp_args = parser.parse_args(
        [
            "hf-sweep-xfamily-mlp",
            "out",
            "--host-model",
            "host",
            "--donor-model",
            "donor",
            "--alphas",
            "0.001,0.005",
            "--max-host-channels",
            "128",
            "--max-donor-channels",
            "256",
            "--channel-assignment",
            "hybrid",
            "--activation-calibration-file",
            "activation.jsonl",
            "--activation-max-calibration-texts",
            "6",
            "--activation-max-length",
            "64",
            "--layer-alpha-mode",
            "similarity",
            "--layer-alpha-min-scale",
            "0.25",
            "--layer-alpha-max-scale",
            "1.5",
            "--layer-alpha-similarity-floor",
            "0.4",
            "--recover",
            "--recover-steps",
            "3",
            "--recover-train-scope",
            "ffn",
            "--recover-calibration-file",
            "calibration.jsonl",
            "--recover-max-calibration-texts",
            "12",
        ]
    )
    xfamily_compress_args = parser.parse_args(
        [
            "hf-xfamily-compress",
            "out",
            "--host-model",
            "host",
            "--donor-model",
            "LiquidAI/LFM2.5-230M",
            "--alpha",
            "0.005",
            "--max-host-channels",
            "128",
            "--max-donor-channels",
            "256",
            "--channel-assignment",
            "activation-correlation",
            "--activation-calibration-file",
            "activation.jsonl",
            "--activation-max-calibration-texts",
            "7",
            "--activation-max-length",
            "80",
            "--layer-alpha-mode",
            "similarity",
            "--layer-alpha-min-scale",
            "0.2",
            "--layer-alpha-max-scale",
            "1.4",
            "--layer-alpha-similarity-floor",
            "0.35",
            "--width-ratios",
            "0.995",
            "--layer-groups",
            "13+14",
            "--target-shrink",
            "0.02",
            "--recover",
            "--recover-only-failed",
            "--recover-top-k",
            "2",
            "--recover-steps",
            "4",
            "--recover-calibration-file",
            "calibration.jsonl",
            "--recover-max-calibration-texts",
            "8",
        ]
    )
    xfamily_recover_args = parser.parse_args(
        [
            "hf-xfamily-recover",
            "donor",
            "candidate",
            "recovered",
            "--steps",
            "12",
            "--batch-size",
            "2",
            "--max-length",
            "64",
            "--lr",
            "0.0003",
            "--anchor-kd-weight",
            "0.7",
            "--lm-weight",
            "0.3",
            "--hidden-kd-weight",
            "0.2",
            "--span-kd-weight",
            "0.15",
            "--span-max-pairs",
            "66",
            "--hidden-ridge",
            "0.02",
            "--hidden-layer",
            "middle",
            "--hidden-max-anchors",
            "77",
            "--temperature",
            "1.5",
            "--max-anchors",
            "123",
            "--train-scope",
            "last2",
            "--dtype",
            "float16",
            "--candidate-dtype",
            "float32",
            "--device",
            "cpu",
            "--grad-clip",
            "0.4",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "6",
        ]
    )
    assert sweep_args.output_dir == "out"
    assert resize_args.assignment == "topk"
    assert random_args.output == "random-out"
    assert random_args.config_model == "HuggingFaceTB/SmolLM2-135M"
    assert random_args.tokenizer_model == "Qwen/Qwen3-4B-Base"
    assert random_args.seed == 123
    assert random_args.dtype == "float32"
    assert donor_init_args.output == "projected-out"
    assert donor_init_args.target_config_model == "HuggingFaceTB/SmolLM2-135M"
    assert donor_init_args.target_tokenizer_model == "HuggingFaceTB/SmolLM2-135M"
    assert donor_init_args.donor_model == "Qwen/Qwen3-4B-Base"
    assert donor_init_args.seed == 456
    assert donor_init_args.max_anchors == 1000
    assert donor_init_args.fold_mode == "mean"
    assert donor_init_args.activation_fold_assignment == "similarity"
    assert donor_init_args.donor_layer_fold_mode == "fold"
    assert donor_init_args.projection_scale == "target-rms"
    assert donor_init_args.activation_calibration_file == "activation.jsonl"
    assert donor_init_args.activation_max_calibration_texts == 5
    assert donor_init_args.activation_max_length == 80
    assert donor_init_args.activation_device == "cpu"
    assert rank_fold_args.model == "model"
    assert rank_fold_args.output == "fold_rank.json"
    assert rank_fold_args.window == 3
    assert rank_fold_args.top_k == 5
    assert rank_fold_args.max_length == 96
    assert rank_fold_args.max_tokens_per_layer == 77
    assert rank_fold_args.dtype == "float16"
    assert rank_fold_args.device == "cpu"
    assert rank_fold_args.calibration_file == "calibration.jsonl"
    assert rank_fold_args.max_calibration_texts == 8
    assert donor_init_args.dtype == "float32"
    assert merge_layer_args.groups == "3+4,7+8"
    assert drop_args.layers == "3,7"
    assert search_args.width_ratios == "0.9975,0.995"
    assert search_args.layer_groups == ["13+14"]
    assert search_args.target_shrink == 0.02
    assert xfamily_mlp_args.max_host_channels == 128
    assert xfamily_mlp_args.max_donor_channels == 256
    assert xfamily_mlp_args.channel_assignment == "hybrid"
    assert xfamily_mlp_args.activation_calibration_file == "activation.jsonl"
    assert xfamily_mlp_args.activation_max_calibration_texts == 6
    assert xfamily_mlp_args.activation_max_length == 64
    assert xfamily_mlp_args.layer_alpha_mode == "similarity"
    assert xfamily_mlp_args.layer_alpha_min_scale == 0.25
    assert xfamily_mlp_args.layer_alpha_max_scale == 1.5
    assert xfamily_mlp_args.layer_alpha_similarity_floor == 0.4
    assert xfamily_mlp_args.recover is True
    assert xfamily_mlp_args.recover_steps == 3
    assert xfamily_mlp_args.recover_train_scope == "ffn"
    assert xfamily_mlp_args.recover_calibration_file == "calibration.jsonl"
    assert xfamily_mlp_args.recover_max_calibration_texts == 12
    assert xfamily_compress_args.donor_model == "LiquidAI/LFM2.5-230M"
    assert xfamily_compress_args.channel_assignment == "activation-correlation"
    assert xfamily_compress_args.activation_calibration_file == "activation.jsonl"
    assert xfamily_compress_args.activation_max_calibration_texts == 7
    assert xfamily_compress_args.activation_max_length == 80
    assert xfamily_compress_args.layer_alpha_mode == "similarity"
    assert xfamily_compress_args.layer_alpha_min_scale == 0.2
    assert xfamily_compress_args.layer_alpha_max_scale == 1.4
    assert xfamily_compress_args.layer_alpha_similarity_floor == 0.35
    assert xfamily_compress_args.width_ratios == "0.995"
    assert xfamily_compress_args.layer_groups == ["13+14"]
    assert xfamily_compress_args.recover_top_k == 2
    assert xfamily_compress_args.recover_only_failed is True
    assert xfamily_compress_args.recover_calibration_file == "calibration.jsonl"
    assert xfamily_compress_args.recover_max_calibration_texts == 8
    assert xfamily_recover_args.source_model == "donor"
    assert xfamily_recover_args.candidate_model == "candidate"
    assert xfamily_recover_args.output == "recovered"
    assert xfamily_recover_args.anchor_kd_weight == 0.7
    assert xfamily_recover_args.lm_weight == 0.3
    assert xfamily_recover_args.hidden_kd_weight == 0.2
    assert xfamily_recover_args.span_kd_weight == 0.15
    assert xfamily_recover_args.span_max_pairs == 66
    assert xfamily_recover_args.hidden_ridge == 0.02
    assert xfamily_recover_args.hidden_layer == "middle"
    assert xfamily_recover_args.hidden_max_anchors == 77
    assert xfamily_recover_args.max_anchors == 123
    assert xfamily_recover_args.train_scope == "last2"
    assert xfamily_recover_args.dtype == "float16"
    assert xfamily_recover_args.candidate_dtype == "float32"
    assert xfamily_recover_args.calibration_file == "calibration.jsonl"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "hf-sweep-task",
                "out",
                "--base-model",
                "base",
                "--tuned-model",
                "tuned",
                "--alphas",
                "0.1",
                "--recover",
            ]
        )

    lm_recover_args = parser.parse_args(
        [
            "hf-lm-recover",
            "candidate",
            "out",
            "--steps",
            "5",
            "--train-scope",
            "last1",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "9",
        ]
    )
    assert lm_recover_args.candidate_model == "candidate"
    assert lm_recover_args.output == "out"
    assert lm_recover_args.steps == 5
    assert lm_recover_args.train_scope == "last1"
    assert lm_recover_args.calibration_file == "calibration.jsonl"
    assert lm_recover_args.max_calibration_texts == 9

    reconstruct_args = parser.parse_args(
        [
            "hf-reconstruct-folded",
            "source",
            "folded",
            "out",
            "--layer-merge-report",
            "merge.json",
            "--ridge",
            "0.02",
            "--reconstruction-alpha",
            "0.5",
            "--reconstruction-target",
            "survivor-mlp",
            "--reconstruction-objective",
            "cosine",
            "--cosine-steps",
            "9",
            "--cosine-lr",
            "0.03",
            "--cosine-l2",
            "0.001",
            "--max-length",
            "80",
            "--max-tokens-per-layer",
            "33",
            "--dtype",
            "float32",
            "--device",
            "cpu",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "6",
        ]
    )
    assert reconstruct_args.source_model == "source"
    assert reconstruct_args.candidate_model == "folded"
    assert reconstruct_args.output == "out"
    assert reconstruct_args.layer_merge_report == "merge.json"
    assert reconstruct_args.ridge == 0.02
    assert reconstruct_args.reconstruction_alpha == 0.5
    assert reconstruct_args.reconstruction_target == "survivor-mlp"
    assert reconstruct_args.reconstruction_objective == "cosine"
    assert reconstruct_args.cosine_steps == 9
    assert reconstruct_args.cosine_lr == 0.03
    assert reconstruct_args.cosine_l2 == 0.001
    assert reconstruct_args.max_length == 80
    assert reconstruct_args.max_tokens_per_layer == 33
    assert reconstruct_args.dtype == "float32"
    assert reconstruct_args.device == "cpu"
    assert reconstruct_args.calibration_file == "calibration.jsonl"
    assert reconstruct_args.max_calibration_texts == 6


def test_parse_layer_groups() -> None:
    assert _parse_layer_groups("3+4, 7+8+9") == [[3, 4], [7, 8, 9]]


def test_layer_group_normalization_rejects_pruning_groups() -> None:
    with pytest.raises(ValueError, match="at least two layers"):
        _normalize_layer_groups([[3]])
    with pytest.raises(ValueError, match="overlap"):
        _normalize_layer_groups([[3, 4], [4, 5]])


def test_layer_group_tags_are_stable() -> None:
    groups = _normalize_layer_groups([[14, 13], [20, 19]])
    assert _layer_groups_tag(groups) == "13-14_19-20"
    assert _layer_groups_for_report(groups) == [[13, 14], [19, 20]]


def test_fold_groups_from_report_keeps_only_fold_method_groups() -> None:
    report = {
        "operation": "merge_decoder_layers",
        "method": "fold",
        "groups": [
            {"group": [3, 4], "survivor": 3, "folded": [4]},
            {"group": [7, 8, 9], "survivor": 7, "folded": [8, 9]},
        ],
    }
    groups = _fold_groups_from_report(report)
    assert groups == [
        {"group": [3, 4], "survivor": 3, "folded": [4], "all_folded_indices": [4, 8, 9]},
        {"group": [7, 8, 9], "survivor": 7, "folded": [8, 9], "all_folded_indices": [4, 8, 9]},
    ]
    assert _fold_groups_from_report({"operation": "merge_decoder_layers", "method": "average", "groups": []}) == []
    with pytest.raises(ValueError, match="merge_decoder_layers"):
        _fold_groups_from_report({"operation": "drop_decoder_layers"})


def test_candidate_index_after_depth_fold_accounts_for_removed_layers() -> None:
    assert _candidate_index_after_depth_fold(5, [2, 4]) == 3
    assert _candidate_index_after_depth_fold(2, [2, 4]) == 2


def test_proportional_layer_index_maps_depths() -> None:
    assert [_proportional_index(idx, 4, 2) for idx in range(4)] == [0, 0, 1, 1]
    assert [_proportional_index(idx, 3, 5) for idx in range(3)] == [0, 2, 4]


def test_donor_layer_groups_partition_parent_depth() -> None:
    groups = [_donor_layer_group_for_target(idx, target_layer_count=3, donor_layer_count=8) for idx in range(3)]

    assert groups == [[0, 1], [2, 3, 4], [5, 6, 7]]
    assert [item for group in groups for item in group] == list(range(8))


def test_linear_cka_detects_similar_activation_spaces() -> None:
    torch = pytest.importorskip("torch")
    first = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    second = first * 2.0
    unrelated = torch.tensor([[0.0, -1.0], [0.0, -1.0], [0.0, 1.0], [0.0, 1.0]])

    assert _linear_cka(torch, first, second) == pytest.approx(1.0)
    assert _linear_cka(torch, first, unrelated) == pytest.approx(0.0)


def test_resize_tensor_to_shape_mean_folds_axes() -> None:
    torch = pytest.importorskip("torch")
    source = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    resized = _resize_tensor_to_shape(torch, source, (3, 2))

    assert resized.shape == (3, 2)
    assert resized.flatten().tolist() == pytest.approx(
        [
            2.5,
            4.5,
            10.5,
            12.5,
            18.5,
            20.5,
        ]
    )


def test_resize_tensor_to_shape_rms_fold_preserves_more_variance() -> None:
    torch = pytest.importorskip("torch")
    source = torch.ones((4, 2), dtype=torch.float32)
    mean_folded = _resize_tensor_to_shape(torch, source, (2, 2), fold_mode="mean")
    rms_folded = _resize_tensor_to_shape(torch, source, (2, 2), fold_mode="rms")

    assert mean_folded.tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert rms_folded.flatten().tolist() == pytest.approx([2**0.5, 2**0.5, 2**0.5, 2**0.5])


def test_scale_projection_to_target_matches_target_rms() -> None:
    torch = pytest.importorskip("torch")
    projected = torch.ones((4, 4), dtype=torch.float32) * 10.0
    target = torch.ones((4, 4), dtype=torch.float32) * 0.5

    scaled = _scale_projection_to_target(torch, projected, target, "target-rms")

    assert float(scaled.pow(2).mean().sqrt()) == pytest.approx(float(target.pow(2).mean().sqrt()))
    assert torch.all(scaled > 0)


def test_activation_weighted_fold_uses_importance_bins() -> None:
    torch = pytest.importorskip("torch")
    weights = torch.tensor([10.0, 1.0, 9.0, 1.0])
    assignment = _balanced_activation_assignment(torch, weights, 2)
    assert assignment.tolist() == [0, 0, 1, 1]

    source = torch.tensor([[0.0], [10.0], [100.0], [200.0]])
    folded = _resize_tensor_to_shape(torch, source, (2, 1), axis_weights={0: weights})

    assert folded.flatten().tolist() == pytest.approx([10.0 / 11.0, 1100.0 / 10.0])


def test_activation_axis_weights_aggregate_folded_donor_group() -> None:
    torch = pytest.importorskip("torch")
    donor_tensor = torch.ones((4, 6), dtype=torch.float32)
    profile = {
        "hidden_scores": torch.ones(5),
        "hidden_features": torch.ones((5, 2)),
        "mlp_scores": {
            2: torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            3: torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        },
        "mlp_features": {
            2: torch.ones((6, 2)),
            3: torch.ones((6, 2)) * 3.0,
        },
    }

    weights = _activation_axis_weights_for_tensor(
        torch,
        donor_tensor,
        "mlp.gate_proj.weight",
        donor_layer_idx=2,
        donor_layer_group=[2, 3],
        activation_profile=profile,
    )

    assert weights is not None
    assert weights[1]["weights"].tolist() == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert weights[1]["features"].flatten().tolist() == pytest.approx([2.0] * 12)


def test_activation_similarity_assignment_groups_activation_signatures() -> None:
    torch = pytest.importorskip("torch")
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    assignment = _activation_similarity_assignment(torch, features, target_dim=2)

    assert assignment[0].item() == assignment[1].item()
    assert assignment[2].item() == assignment[3].item()
    assert assignment[0].item() != assignment[2].item()


def test_fit_activation_map_cosine_objective_reports_directional_fit() -> None:
    torch = pytest.importorskip("torch")
    features = torch.eye(3, dtype=torch.float32)
    targets = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    report = _fit_activation_map(
        torch,
        features,
        targets,
        ridge=0.01,
        reconstruction_alpha=1.0,
        reconstruction_objective="cosine",
        cosine_steps=3,
        cosine_lr=0.01,
        cosine_l2=0.0,
    )

    assert tuple(report["map"].shape) == (3, 3)
    assert report["solver"]["objective"] == "cosine"
    assert report["solver"]["cosine_refined"] is True
    assert report["mean_cosine_after"] > report["mean_cosine_before"]


def test_recovery_loss_allows_lm_only_without_teacher() -> None:
    torch = pytest.importorskip("torch")
    student = _Obj(logits=torch.randn(1, 4, 8))
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    loss = _recovery_loss(
        torch,
        student,
        teacher=None,
        input_ids=input_ids,
        attention_mask=attention_mask,
        kd_weight=0.0,
        lm_weight=1.0,
        temperature=1.0,
    )

    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="teacher output"):
        _recovery_loss(
            torch,
            student,
            teacher=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            kd_weight=1.0,
            lm_weight=0.0,
            temperature=1.0,
        )


def test_cross_tokenizer_recovery_loss_uses_anchor_subspace() -> None:
    torch = pytest.importorskip("torch")
    student = _Obj(logits=torch.randn(1, 4, 7))
    teacher = _Obj(logits=torch.randn(1, 5, 9))
    candidate_ids = torch.tensor([[1, 2, 3, 4]])
    candidate_mask = torch.ones_like(candidate_ids)
    source_mask = torch.ones((1, 5), dtype=torch.long)

    loss = _cross_tokenizer_recovery_loss(
        torch,
        student,
        teacher,
        candidate_ids,
        candidate_mask,
        source_mask,
        candidate_anchor_idx=torch.tensor([1, 3, 5]),
        source_anchor_idx=torch.tensor([2, 4, 6]),
        anchor_kd_weight=0.7,
        lm_weight=0.3,
        hidden_kd_weight=0.0,
        span_kd_weight=0.0,
        hidden_projection=None,
        hidden_layer="last",
        span_pairs=None,
        temperature=1.5,
    )

    assert torch.isfinite(loss)


def test_cross_tokenizer_recovery_loss_uses_projected_hidden_states() -> None:
    torch = pytest.importorskip("torch")
    student = _Obj(
        logits=torch.randn(1, 4, 7),
        hidden_states=(torch.randn(1, 4, 3), torch.randn(1, 4, 3)),
    )
    teacher = _Obj(
        logits=torch.randn(1, 5, 9),
        hidden_states=(torch.randn(1, 5, 5), torch.randn(1, 5, 5)),
    )
    candidate_ids = torch.tensor([[1, 2, 3, 4]])
    candidate_mask = torch.ones_like(candidate_ids)
    source_mask = torch.ones((1, 5), dtype=torch.long)
    projection = torch.randn(5, 3)

    loss = _cross_tokenizer_recovery_loss(
        torch,
        student,
        teacher,
        candidate_ids,
        candidate_mask,
        source_mask,
        candidate_anchor_idx=torch.tensor([1, 3, 5]),
        source_anchor_idx=torch.tensor([2, 4, 6]),
        anchor_kd_weight=0.1,
        lm_weight=0.2,
        hidden_kd_weight=0.7,
        span_kd_weight=0.0,
        hidden_projection=projection,
        hidden_layer="last",
        span_pairs=None,
        temperature=1.5,
    )

    assert torch.isfinite(loss)


def test_cross_tokenizer_span_pairs_match_exact_offsets_and_tokens() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = _Tokenizer(["<eos>", "foo", "bar"])
    source_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "offset_mapping": torch.tensor([[[0, 0], [0, 3], [4, 7], [0, 0]]]),
    }
    candidate_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "offset_mapping": torch.tensor([[[0, 0], [0, 3], [4, 7], [0, 0]]]),
    }

    pairs = _cross_tokenizer_span_pairs(
        torch,
        tokenizer,
        tokenizer,
        source_batch,
        candidate_batch,
        _canonical_token,
        max_pairs=8,
    )

    assert pairs.tolist() == [[0, 0, 0], [0, 1, 1]]


def test_cross_tokenizer_recovery_loss_uses_span_pairs() -> None:
    torch = pytest.importorskip("torch")
    student = _Obj(logits=torch.randn(1, 4, 7))
    teacher = _Obj(logits=torch.randn(1, 4, 9))
    candidate_ids = torch.tensor([[1, 2, 3, 4]])
    candidate_mask = torch.ones_like(candidate_ids)
    source_mask = torch.ones((1, 4), dtype=torch.long)

    loss = _cross_tokenizer_recovery_loss(
        torch,
        student,
        teacher,
        candidate_ids,
        candidate_mask,
        source_mask,
        candidate_anchor_idx=torch.tensor([1, 3, 5]),
        source_anchor_idx=torch.tensor([2, 4, 6]),
        anchor_kd_weight=0.0,
        lm_weight=0.2,
        hidden_kd_weight=0.0,
        span_kd_weight=0.8,
        hidden_projection=None,
        hidden_layer="last",
        span_pairs=torch.tensor([[0, 0, 0], [0, 1, 1]]),
        temperature=1.5,
    )

    assert torch.isfinite(loss)


def test_shared_embedding_hidden_projection_caps_anchor_rows() -> None:
    torch = pytest.importorskip("torch")

    class _Model:
        def __init__(self, weight) -> None:
            self.embedding = torch.nn.Embedding.from_pretrained(weight, freeze=False)

        def get_input_embeddings(self):
            return self.embedding

    source_weight = torch.eye(6, 4)
    candidate_weight = torch.randn(6, 2)
    report = _fit_shared_embedding_hidden_projection(
        torch,
        _Model(source_weight),
        _Model(candidate_weight),
        source_anchor_idx=torch.arange(6),
        candidate_anchor_idx=torch.arange(6),
        ridge=0.01,
        max_anchors=3,
    )

    assert tuple(report["map"].shape) == (4, 2)
    assert report["available_anchor_count"] == 6
    assert report["anchor_count"] == 3
    assert report["relative_error"] >= 0.0


def test_absorb_input_map_into_linears_updates_weight_orientation() -> None:
    torch = pytest.importorskip("torch")
    linear = torch.nn.Linear(2, 2, bias=False)
    linear.weight.data.copy_(torch.eye(2))
    mapping = torch.tensor([[2.0, 0.0], [0.0, 3.0]])

    _absorb_input_map_into_linears(torch, [("linear", linear)], mapping)

    assert linear.weight.detach().flatten().tolist() == pytest.approx([2.0, 0.0, 0.0, 3.0])


def test_mlp_spec_detects_lfm_feed_forward_names() -> None:
    layer = _Obj(feed_forward=_Obj(w1=object(), w3=object(), w2=object()))
    spec = _mlp_spec(layer)
    assert spec is not None
    assert spec["kind"] == "gated"
    assert spec["input_names"] == ["w1", "w3"]
    assert spec["output_name"] == "w2"


def test_recovery_mlp_projection_spec_detects_lfm_feed_forward_names() -> None:
    layer = _Obj(feed_forward=_Obj(w1=object(), w3=object(), w2=object()))
    spec = _mlp_projection_spec(layer)
    assert spec is not None
    assert spec["kind"] == "gated"
    assert spec["output_name"] == "feed_forward.w2"


def test_recovery_train_scope_supports_feed_forward_layers() -> None:
    class _Param:
        def __init__(self) -> None:
            self.requires_grad = False

        def requires_grad_(self, value: bool):
            self.requires_grad = value
            return self

        def numel(self) -> int:
            return 1

    class _Module:
        def __init__(self) -> None:
            self.param = _Param()

        def parameters(self):
            return [self.param]

    class _Layer:
        def __init__(self) -> None:
            self.feed_forward = _Module()

        def parameters(self):
            return self.feed_forward.parameters()

    class _Model:
        def __init__(self) -> None:
            self.model = _Obj(layers=[_Layer(), _Layer()])
            self.output = _Module()

        def parameters(self):
            params = []
            for layer in self.model.layers:
                params.extend(layer.parameters())
            params.extend(self.output.parameters())
            return params

        def get_output_embeddings(self):
            return self.output

    model = _Model()
    trainable = _set_train_scope(model, "ffn")
    assert trainable == 2
    assert all(layer.feed_forward.param.requires_grad for layer in model.model.layers)


def test_hf_random_init_handler_passes_flags(monkeypatch) -> None:
    import neural_forge.hf_tools as hf_tools

    captured = {}

    def fake_random_init(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_tools, "random_init_causal_lm", fake_random_init)
    args = build_parser().parse_args(
        [
            "hf-random-init",
            "out",
            "--config-model",
            "config",
            "--tokenizer-model",
            "tokenizer",
            "--seed",
            "9",
            "--dtype",
            "float32",
            "--trust-remote-code",
        ]
    )
    result = args.func(args)
    assert result == {"ok": True}
    assert captured == {
        "config_model": "config",
        "tokenizer_model": "tokenizer",
        "output_dir": "out",
        "seed": 9,
        "torch_dtype": "float32",
        "trust_remote_code": True,
    }


def test_hf_donor_project_init_handler_passes_flags(monkeypatch) -> None:
    import neural_forge.cli as cli
    import neural_forge.hf_xfamily as hf_xfamily

    captured = {}

    def fake_project_init(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    def fake_load(path, max_texts):
        assert path == "activation.jsonl"
        assert max_texts == 7
        return ["activation text"]

    monkeypatch.setattr(hf_xfamily, "donor_projected_init", fake_project_init)
    monkeypatch.setattr(cli, "_load_recovery_calibration_texts", fake_load)
    args = build_parser().parse_args(
        [
            "hf-donor-project-init",
            "out",
            "--target-config-model",
            "target-config",
            "--target-tokenizer-model",
            "target-tokenizer",
            "--donor-model",
            "donor",
            "--seed",
            "11",
            "--ridge",
            "0.02",
            "--max-anchors",
            "1234",
            "--fold-mode",
            "mean",
            "--activation-fold-assignment",
            "similarity",
            "--donor-layer-fold-mode",
            "average",
            "--projection-scale",
            "target-rms",
            "--activation-calibration-file",
            "activation.jsonl",
            "--activation-max-calibration-texts",
            "7",
            "--activation-max-length",
            "64",
            "--activation-device",
            "cpu",
            "--dtype",
            "float32",
            "--trust-remote-code",
        ]
    )
    result = args.func(args)
    assert result == {"ok": True}
    assert captured == {
        "target_config_model": "target-config",
        "target_tokenizer_model": "target-tokenizer",
        "donor_model": "donor",
        "output_dir": "out",
        "seed": 11,
        "ridge": 0.02,
        "max_anchors": 1234,
        "fold_mode": "mean",
        "activation_fold_assignment": "similarity",
        "donor_layer_fold_mode": "average",
        "projection_scale": "target-rms",
        "activation_calibration_texts": ["activation text"],
        "activation_max_length": 64,
        "activation_device": "cpu",
        "torch_dtype": "float32",
        "trust_remote_code": True,
    }


def test_hf_lm_recover_handler_passes_teacherless_flags(monkeypatch) -> None:
    import neural_forge.cli as cli
    import neural_forge.hf_recovery as hf_recovery

    captured = {}

    def fake_recover(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_recovery, "recover_causal_lm", fake_recover)
    monkeypatch.setattr(cli, "_load_recovery_calibration_texts", lambda path, max_texts: ["calibration text"])
    args = build_parser().parse_args(
        [
            "hf-lm-recover",
            "candidate",
            "out",
            "--steps",
            "6",
            "--batch-size",
            "2",
            "--max-length",
            "48",
            "--lr",
            "0.0002",
            "--train-scope",
            "last2",
            "--dtype",
            "float32",
            "--device",
            "cpu",
            "--grad-clip",
            "0.5",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "4",
        ]
    )
    result = args.func(args)

    assert result == {"ok": True}
    assert captured == {
        "source_model": None,
        "candidate_model": "candidate",
        "output_dir": "out",
        "steps": 6,
        "batch_size": 2,
        "max_length": 48,
        "lr": 0.0002,
        "kd_weight": 0.0,
        "lm_weight": 1.0,
        "temperature": 1.0,
        "train_scope": "last2",
        "torch_dtype": "float32",
        "device": "cpu",
        "trust_remote_code": False,
        "calibration_texts": ["calibration text"],
        "grad_clip": 0.5,
    }


def test_hf_xfamily_recover_handler_passes_anchor_flags(monkeypatch) -> None:
    import neural_forge.cli as cli
    import neural_forge.hf_recovery as hf_recovery

    captured = {}

    def fake_recover(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_recovery, "recover_cross_tokenizer_causal_lm", fake_recover)
    monkeypatch.setattr(cli, "_load_recovery_calibration_texts", lambda path, max_texts: ["calibration text"])
    args = build_parser().parse_args(
        [
            "hf-xfamily-recover",
            "donor",
            "candidate",
            "out",
            "--steps",
            "5",
            "--batch-size",
            "2",
            "--max-length",
            "48",
            "--lr",
            "0.0002",
            "--anchor-kd-weight",
            "0.6",
            "--lm-weight",
            "0.4",
            "--hidden-kd-weight",
            "0.25",
            "--span-kd-weight",
            "0.15",
            "--span-max-pairs",
            "88",
            "--hidden-ridge",
            "0.03",
            "--hidden-layer",
            "middle",
            "--hidden-max-anchors",
            "99",
            "--temperature",
            "1.7",
            "--max-anchors",
            "321",
            "--train-scope",
            "last2",
            "--dtype",
            "float16",
            "--candidate-dtype",
            "float32",
            "--device",
            "cpu",
            "--grad-clip",
            "0.5",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "3",
        ]
    )
    result = args.func(args)

    assert result == {"ok": True}
    assert captured == {
        "source_model": "donor",
        "candidate_model": "candidate",
        "output_dir": "out",
        "steps": 5,
        "batch_size": 2,
        "max_length": 48,
        "lr": 0.0002,
        "anchor_kd_weight": 0.6,
        "lm_weight": 0.4,
        "hidden_kd_weight": 0.25,
        "span_kd_weight": 0.15,
        "span_max_pairs": 88,
        "hidden_ridge": 0.03,
        "hidden_layer": "middle",
        "hidden_max_anchors": 99,
        "temperature": 1.7,
        "max_anchors": 321,
        "train_scope": "last2",
        "torch_dtype": "float16",
        "candidate_torch_dtype": "float32",
        "device": "cpu",
        "trust_remote_code": False,
        "calibration_texts": ["calibration text"],
        "grad_clip": 0.5,
    }


def test_hf_reconstruct_folded_handler_passes_flags(monkeypatch) -> None:
    import neural_forge.cli as cli
    import neural_forge.hf_recovery as hf_recovery

    captured = {}

    def fake_reconstruct(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_recovery, "reconstruct_folded_causal_lm", fake_reconstruct)
    monkeypatch.setattr(cli, "_load_recovery_calibration_texts", lambda path, max_texts: ["calibration text"])
    args = build_parser().parse_args(
        [
            "hf-reconstruct-folded",
            "source",
            "folded",
            "out",
            "--layer-merge-report",
            "merge.json",
            "--ridge",
            "0.02",
            "--reconstruction-alpha",
            "0.4",
            "--reconstruction-target",
            "both",
            "--reconstruction-objective",
            "cosine",
            "--cosine-steps",
            "7",
            "--cosine-lr",
            "0.02",
            "--cosine-l2",
            "0.003",
            "--max-length",
            "72",
            "--max-tokens-per-layer",
            "128",
            "--dtype",
            "float32",
            "--device",
            "cpu",
            "--trust-remote-code",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "4",
        ]
    )
    result = args.func(args)

    assert result == {"ok": True}
    assert captured == {
        "source_model": "source",
        "candidate_model": "folded",
        "output_dir": "out",
        "layer_merge_report": "merge.json",
        "ridge": 0.02,
        "reconstruction_alpha": 0.4,
        "reconstruction_target": "both",
        "reconstruction_objective": "cosine",
        "cosine_steps": 7,
        "cosine_lr": 0.02,
        "cosine_l2": 0.003,
        "max_length": 72,
        "max_tokens_per_layer": 128,
        "torch_dtype": "float32",
        "device": "cpu",
        "trust_remote_code": True,
        "calibration_texts": ["calibration text"],
    }


def test_xfamily_mlp_merge_handler_passes_activation_flags(monkeypatch) -> None:
    import neural_forge.cli as cli
    import neural_forge.hf_xfamily as hf_xfamily

    captured = {}

    def fake_merge(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    def fake_load(path, max_texts):
        assert path == "activation.jsonl"
        assert max_texts == 4
        return ["activation text"]

    monkeypatch.setattr(hf_xfamily, "cross_family_mlp_merge", fake_merge)
    monkeypatch.setattr(cli, "_load_recovery_calibration_texts", fake_load)
    args = build_parser().parse_args(
        [
            "hf-xfamily-mlp-merge",
            "out",
            "--host-model",
            "host",
            "--donor-model",
            "donor",
            "--channel-assignment",
            "activation-correlation",
            "--activation-calibration-file",
            "activation.jsonl",
            "--activation-max-calibration-texts",
            "4",
            "--activation-max-length",
            "72",
            "--activation-device",
            "cpu",
            "--layer-alpha-mode",
            "similarity",
            "--layer-alpha-min-scale",
            "0.25",
            "--layer-alpha-max-scale",
            "1.25",
            "--layer-alpha-similarity-floor",
            "0.3",
        ]
    )
    result = args.func(args)
    assert result == {"ok": True}
    assert captured["channel_assignment"] == "activation-correlation"
    assert captured["activation_calibration_texts"] == ["activation text"]
    assert captured["activation_max_length"] == 72
    assert captured["activation_device"] == "cpu"
    assert captured["layer_alpha_mode"] == "similarity"
    assert captured["layer_alpha_min_scale"] == 0.25
    assert captured["layer_alpha_max_scale"] == 1.25
    assert captured["layer_alpha_similarity_floor"] == 0.3


def test_xfamily_mlp_handler_passes_recovery_flags(monkeypatch) -> None:
    import neural_forge.cli as cli
    import neural_forge.hf_xfamily as hf_xfamily

    captured = {}

    def fake_sweep(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_xfamily, "sweep_cross_family_mlp_merge", fake_sweep)
    monkeypatch.setattr(cli, "_load_recovery_calibration_texts", lambda path, max_texts: None if path is None else ["calibration text"])
    args = build_parser().parse_args(
        [
            "hf-sweep-xfamily-mlp",
            "out",
            "--host-model",
            "host",
            "--donor-model",
            "donor",
            "--alphas",
            "0.005",
            "--recover",
            "--recover-only-failed",
            "--recover-steps",
            "7",
            "--recover-train-scope",
            "last2",
            "--channel-assignment",
            "hybrid",
            "--activation-calibration-file",
            "activation.jsonl",
            "--activation-max-calibration-texts",
            "5",
            "--activation-max-length",
            "88",
            "--activation-device",
            "cpu",
            "--layer-alpha-mode",
            "similarity",
            "--layer-alpha-min-scale",
            "0.25",
            "--layer-alpha-max-scale",
            "1.25",
            "--layer-alpha-similarity-floor",
            "0.3",
            "--recover-calibration-file",
            "calibration.jsonl",
            "--recover-max-calibration-texts",
            "3",
        ]
    )
    result = args.func(args)
    assert result == {"ok": True}
    assert captured["recover_candidates"] is True
    assert captured["recover_only_failed"] is True
    assert captured["recovery_steps"] == 7
    assert captured["recovery_train_scope"] == "last2"
    assert captured["recovery_calibration_texts"] == ["calibration text"]
    assert captured["channel_assignment"] == "hybrid"
    assert captured["activation_calibration_texts"] == ["calibration text"]
    assert captured["activation_max_length"] == 88
    assert captured["activation_device"] == "cpu"
    assert captured["layer_alpha_mode"] == "similarity"
    assert captured["layer_alpha_min_scale"] == 0.25
    assert captured["layer_alpha_max_scale"] == 1.25
    assert captured["layer_alpha_similarity_floor"] == 0.3


def test_xfamily_compress_handler_passes_pipeline_flags(monkeypatch) -> None:
    import neural_forge.cli as cli
    import neural_forge.hf_xfamily as hf_xfamily

    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_xfamily, "cross_family_merge_compress_pipeline", fake_pipeline)
    monkeypatch.setattr(cli, "_load_recovery_calibration_texts", lambda path, max_texts: None if path is None else ["calibration text"])
    args = build_parser().parse_args(
        [
            "hf-xfamily-compress",
            "out",
            "--host-model",
            "host",
            "--donor-model",
            "LiquidAI/LFM2.5-230M",
            "--alpha",
            "0.005",
            "--width-ratios",
            "0.995",
            "--layer-groups",
            "13+14",
            "--depth-methods",
            "average",
            "--target-shrink",
            "0.02",
            "--recover",
            "--recover-only-failed",
            "--recover-top-k",
            "2",
            "--recover-steps",
            "7",
            "--channel-assignment",
            "activation-correlation",
            "--activation-calibration-file",
            "activation.jsonl",
            "--activation-max-calibration-texts",
            "5",
            "--activation-max-length",
            "88",
            "--activation-device",
            "cpu",
            "--layer-alpha-mode",
            "similarity",
            "--layer-alpha-min-scale",
            "0.25",
            "--layer-alpha-max-scale",
            "1.25",
            "--layer-alpha-similarity-floor",
            "0.3",
            "--recover-calibration-file",
            "calibration.jsonl",
            "--recover-max-calibration-texts",
            "3",
        ]
    )
    result = args.func(args)
    assert result == {"ok": True}
    assert captured["donor_model"] == "LiquidAI/LFM2.5-230M"
    assert captured["alpha"] == 0.005
    assert captured["width_ratios"] == [0.995]
    assert captured["layer_group_specs"] == [[[13, 14]]]
    assert captured["depth_methods"] == ["average"]
    assert captured["target_shrink"] == 0.02
    assert captured["recover_candidates"] is True
    assert captured["recover_only_failed"] is True
    assert captured["recover_top_k"] == 2
    assert captured["recovery_steps"] == 7
    assert captured["recovery_calibration_texts"] == ["calibration text"]
    assert captured["channel_assignment"] == "activation-correlation"
    assert captured["activation_calibration_texts"] == ["calibration text"]
    assert captured["activation_max_length"] == 88
    assert captured["activation_device"] == "cpu"
    assert captured["layer_alpha_mode"] == "similarity"
    assert captured["layer_alpha_min_scale"] == 0.25
    assert captured["layer_alpha_max_scale"] == 1.25
    assert captured["layer_alpha_similarity_floor"] == 0.3


def test_select_compression_candidates_prefers_target_passed() -> None:
    target = {"tag": "target", "shrink": 0.03}
    below = {"tag": "below", "shrink": 0.01}
    best = {"tag": "best", "shrink": 0.02}
    summary = {
        "target_passed_candidates": [target],
        "best_passed_below_target": below,
        "best_passed_candidate": best,
    }
    assert _select_compression_candidates(summary, 0.02) == [target]


def test_select_compression_candidates_falls_back_below_target() -> None:
    below = {"tag": "below", "shrink": 0.01}
    best = {"tag": "best", "shrink": 0.02}
    summary = {
        "target_passed_candidates": [],
        "best_passed_below_target": below,
        "best_passed_candidate": best,
    }
    assert _select_compression_candidates(summary, 0.03) == [below]


def test_usable_pipeline_candidates_keeps_only_gate_passing_outputs() -> None:
    compressed = [
        {
            "tag": "raw_pass",
            "model_dir": "raw",
            "report": "raw.json",
            "host_gate_report": "raw_gate.json",
            "host_passed": True,
            "shrink": 0.05,
            "host_metrics": {"perplexity_mean": 300.0},
        },
        {
            "tag": "raw_fail",
            "model_dir": "fail",
            "report": "fail.json",
            "host_gate_report": "fail_gate.json",
            "host_passed": False,
            "shrink": 0.06,
            "host_metrics": {"perplexity_mean": 250.0},
        },
    ]
    recovered = [
        {
            "passed": True,
            "source_candidate": compressed[1],
            "model_dir": "rec",
            "report": "rec.json",
            "gate_report": "rec_gate.json",
            "metrics": {"perplexity_mean": 240.0},
        },
        {
            "passed": False,
            "source_candidate": compressed[0],
            "model_dir": "bad_rec",
            "report": "bad_rec.json",
            "gate_report": "bad_rec_gate.json",
            "metrics": {"perplexity_mean": 200.0},
        },
    ]
    usable = _usable_pipeline_candidates(compressed, recovered)
    assert [item["tag"] for item in usable] == ["raw_fail_recovered", "raw_pass"]
