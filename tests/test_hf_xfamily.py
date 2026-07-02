from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_forge.cli import build_parser
from neural_forge.cli import _parse_layer_groups
from neural_forge.cli import _post_init_gate_decision
from neural_forge.hf_recovery import RECOVERY_TEXTS
from neural_forge.hf_recovery import _answer_contrastive_candidates
from neural_forge.hf_recovery import _answer_first_token_margin_loss
from neural_forge.hf_recovery import _answer_first_token_negative_ids
from neural_forge.hf_recovery import _alm_text_chunks
from neural_forge.hf_recovery import _append_eos_text
from neural_forge.hf_recovery import _activation_row_transport_loss
from neural_forge.hf_recovery import _attention_projection_spec
from neural_forge.hf_recovery import _assert_same_vocab
from neural_forge.hf_recovery import _absorb_input_map_into_linears
from neural_forge.hf_recovery import _behavior_checkpoint_is_better
from neural_forge.hf_recovery import _candidate_rank_kl_loss
from neural_forge.hf_recovery import _candidate_to_source_token_groups
from neural_forge.hf_recovery import _candidate_index_after_depth_fold
from neural_forge.hf_recovery import _cross_tokenizer_recovery_loss
from neural_forge.hf_recovery import _cross_tokenizer_dual_space_kd_loss
from neural_forge.hf_recovery import _cross_tokenizer_alm_chunk_loss
from neural_forge.hf_recovery import _cross_tokenizer_attention_influence_loss
from neural_forge.hf_recovery import _cross_tokenizer_focus_span_pairs
from neural_forge.hf_recovery import _cross_tokenizer_overlap_span_pairs
from neural_forge.hf_recovery import _cross_tokenizer_sequence_nll_loss
from neural_forge.hf_recovery import _cross_tokenizer_span_pairs
from neural_forge.hf_recovery import _cross_tokenizer_span_pairs_for_mode
from neural_forge.hf_recovery import _entropy_aware_kl_loss
from neural_forge.hf_recovery import _fit_activation_map
from neural_forge.hf_recovery import _fit_lm_head_from_hidden_logits
from neural_forge.hf_recovery import _fit_lm_head_mean_shift
from neural_forge.hf_recovery import _fit_shared_embedding_hidden_projection
from neural_forge.hf_recovery import _first_response_shift_mask
from neural_forge.hf_recovery import _focus_spans_to_shift_mask
from neural_forge.hf_recovery import _focus_spans_to_span_pairs
from neural_forge.hf_recovery import _focus_char_spans
from neural_forge.hf_recovery import _fold_groups_from_report
from neural_forge.hf_recovery import _feature_decorrelation_loss
from neural_forge.hf_recovery import _first_byte_head_from_output_embeddings
from neural_forge.hf_recovery import _first_byte_probs_from_logits
from neural_forge.hf_recovery import _cross_tokenizer_hidden_first_byte_loss
from neural_forge.hf_recovery import _last_response_shift_mask
from neural_forge.hf_recovery import _hidden_representation_loss
from neural_forge.hf_recovery import _infer_response_char_start
from neural_forge.hf_recovery import _join_prompt_response
from neural_forge.hf_recovery import _layer_gate_behavior_decision
from neural_forge.hf_recovery import _layer_gate_decision
from neural_forge.hf_recovery import _layer_gate_snapshot_indices
from neural_forge.hf_recovery import _linear_cka_tensor
from neural_forge.hf_recovery import _mlp_projection_spec
from neural_forge.hf_recovery import _on_policy_prompt_text
from neural_forge.hf_recovery import _output_head_tied_to_input
from neural_forge.hf_recovery import _proportional_layer_index
from neural_forge.hf_recovery import _project_candidate_probs_to_source_vocab
from neural_forge.hf_recovery import _projected_vocab_group_summary
from neural_forge.hf_recovery import _project_source_logits_to_candidate_vocab
from neural_forge.hf_recovery import _recovery_loss
from neural_forge.hf_recovery import _recent_token_unlikelihood_loss
from neural_forge.hf_recovery import _row_gram_geometry_loss
from neural_forge.hf_recovery import _select_anchor_tokens
from neural_forge.hf_recovery import _set_train_scope
from neural_forge.hf_recovery import _same_tokenizer_reference_loss
from neural_forge.hf_recovery import _same_tokenizer_hidden_loss
from neural_forge.hf_recovery import _same_tokenizer_top_k_kl
from neural_forge.hf_recovery import _apply_layer_residual_magnitude_compensation
from neural_forge.hf_recovery import _apply_residual_output_map_to_modules
from neural_forge.hf_recovery import _apply_residual_output_map_to_module
from neural_forge.hf_recovery import _block_recovery_snapshot_modules
from neural_forge.hf_recovery import _eigenspace_low_rank_delta_map
from neural_forge.hf_recovery import _low_rank_delta_map
from neural_forge.hf_recovery import _scale_layer_residual_output_modules
from neural_forge.hf_recovery import _source_group_delta_state
from neural_forge.hf_recovery import _source_group_module_output_state
from neural_forge.hf_recovery import _temporary_padding_side
from neural_forge.hf_recovery import _tokenizer_first_byte_buckets
from neural_forge.hf_recovery import _validate_blockwise_learning_rate
from neural_forge.hf_recovery import _validation_reconstruction_gate
from neural_forge.hf_recovery import load_calibration_texts
from neural_forge.hf_tools import TEXT_EVALS
from neural_forge.hf_tools import build_eval_suite
from neural_forge.hf_tools import build_recovery_calibration
from neural_forge.hf_tools import _generation_health
from neural_forge.hf_tools import _layer_groups_for_report
from neural_forge.hf_tools import _layer_groups_tag
from neural_forge.hf_tools import _linear_cka
from neural_forge.hf_tools import _normalize_generation_prompt
from neural_forge.hf_tools import _normalize_layer_groups
from neural_forge.hf_xfamily import _canonical_token
from neural_forge.hf_xfamily import _chain_stage_promotes
from neural_forge.hf_xfamily import _lookup_layer_tensor_key
from neural_forge.hf_xfamily import _lookup_layer_tensor_value
from neural_forge.hf_xfamily import _parse_chain_stages
from neural_forge.hf_xfamily import _parse_source_models_and_weights
from neural_forge.hf_xfamily import _canonical_token_map
from neural_forge.hf_xfamily import _assign_donor_to_host
from neural_forge.hf_xfamily import _activation_medoid_support_indices
from neural_forge.hf_xfamily import _activation_quantile_medoid_support_indices
from neural_forge.hf_xfamily import _activation_quantile_support_indices
from neural_forge.hf_xfamily import _activation_layer_geometry_scores
from neural_forge.hf_xfamily import _activation_similarity_assignment
from neural_forge.hf_xfamily import _activation_axis_weights_for_tensor
from neural_forge.hf_xfamily import _activation_fit_suffix_enabled
from neural_forge.hf_xfamily import _activation_channel_summary
from neural_forge.hf_xfamily import _mlp_neuron_activation_values
from neural_forge.hf_xfamily import _attention_axis_weights_for_tensor
from neural_forge.hf_xfamily import _attention_head_activation_profile
from neural_forge.hf_xfamily import _attention_projection_modules
from neural_forge.hf_xfamily import _axis_reductions_for_layer_tensor
from neural_forge.hf_xfamily import _axis_reductions_for_non_layer_tensor_key
from neural_forge.hf_xfamily import _balanced_activation_assignment
from neural_forge.hf_xfamily import _blend_projection_with_target
from neural_forge.hf_xfamily import _build_activation_hidden_basis
from neural_forge.hf_xfamily import _build_shared_token_hidden_basis
from neural_forge.hf_xfamily import _channel_assignment_features
from neural_forge.hf_xfamily import _donor_layer_group_for_target
from neural_forge.hf_xfamily import _donor_layer_groups_by_energy
from neural_forge.hf_xfamily import _donor_layer_groups_for_target_count
from neural_forge.hf_xfamily import _donor_layer_geometry_scores
from neural_forge.hf_xfamily import _representative_donor_layer_for_group
from neural_forge.hf_xfamily import _effective_activation_fit_max_rows
from neural_forge.hf_xfamily import _effective_layer_alpha
from neural_forge.hf_xfamily import _fit_linear_weight_from_activation_rows
from neural_forge.hf_xfamily import _fit_projected_linear_from_donor_activations
from neural_forge.hf_xfamily import _hidden_basis_axes_for_layer_tensor
from neural_forge.hf_xfamily import _linear_fit_rows_for_suffix
from neural_forge.hf_xfamily import _truncate_linear_weight_rank
from neural_forge.hf_xfamily import _truncate_linear_weight_delta_rank
from neural_forge.hf_xfamily import _initialize_token_weights_from_donor
from neural_forge.hf_xfamily import _initialize_non_layer_tensors_from_donor
from neural_forge.hf_xfamily import _mlp_spec
from neural_forge.hf_xfamily import _neutralized_unmatched_operator_tensor
from neural_forge.hf_xfamily import _proportional_index
from neural_forge.hf_xfamily import _project_token_weight_rows
from neural_forge.hf_xfamily import _resize_attention_tensor_to_shape
from neural_forge.hf_xfamily import _resize_tensor_axis_mean_fold
from neural_forge.hf_xfamily import _resize_tensor_to_shape
from neural_forge.hf_xfamily import _scale_projection_to_target
from neural_forge.hf_xfamily import _select_compression_candidates
from neural_forge.hf_xfamily import _sanitize_generation_config_for_save
from neural_forge.hf_xfamily import _synthesize_cross_arch_layer_tensor
from neural_forge.hf_xfamily import _summarize_projection_diagnostics
from neural_forge.hf_xfamily import _target_to_donor_token_group_specs
from neural_forge.hf_xfamily import _target_to_donor_special_token_group_specs
from neural_forge.hf_xfamily import _target_to_donor_token_groups
from neural_forge.hf_xfamily import _target_to_donor_omp_group_specs
from neural_forge.hf_xfamily import _tensor_projection_diagnostic
from neural_forge.hf_xfamily import _untie_output_head
from neural_forge.hf_xfamily import _usable_candidate_sort_key
from neural_forge.hf_xfamily import _usable_pipeline_candidates
from neural_forge.hf_xfamily import _validate_channel_assignment
from neural_forge.hf_xfamily import _validate_layer_alpha_settings
from neural_forge.hf_xfamily import _with_consumer_hidden_reduction
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

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        vocab = self.get_vocab()
        if text in vocab:
            return [vocab[text]]
        return [vocab[part] for part in text.split() if part in vocab]


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


def test_target_to_donor_token_groups_uses_tokenized_fallback() -> None:
    target_tokenizer = _Tokenizer(["<eos>", "hello", "hello world", "missing"])
    donor_tokenizer = _Tokenizer(["<eos>", "hello", "world"])
    target_map = _canonical_token_map(target_tokenizer)
    donor_map = _canonical_token_map(donor_tokenizer)

    target_ids, donor_groups, tokenized_rows = _target_to_donor_token_groups(
        target_tokenizer,
        donor_tokenizer,
        target_map,
        donor_map,
    )

    assert target_ids == [1, 2]
    assert donor_groups == [[1], [1, 2]]
    assert tokenized_rows == 1


def test_target_to_donor_token_group_specs_can_weight_tokenized_pieces() -> None:
    target_tokenizer = _Tokenizer(["<eos>", "a", "a longer"])
    donor_tokenizer = _Tokenizer(["<eos>", "a", "longer"])
    target_map = _canonical_token_map(target_tokenizer)
    donor_map = _canonical_token_map(donor_tokenizer)

    target_ids, donor_groups, donor_weights, tokenized_rows, weighted_rows = _target_to_donor_token_group_specs(
        target_tokenizer,
        donor_tokenizer,
        target_map,
        donor_map,
        weighted=True,
    )

    assert target_ids == [1, 2]
    assert donor_groups == [[1], [1, 2]]
    assert donor_weights == [[1.0], [1.0, 6.0]]
    assert tokenized_rows == 1
    assert weighted_rows == 1


def test_target_to_donor_special_token_groups_use_roles_and_surfaces() -> None:
    target_tokenizer = _Tokenizer(["<pad>", "<bos>", "<eos>", "hello", "<extra>"])
    target_tokenizer.all_special_ids = [0, 1, 2, 4]
    target_tokenizer.pad_token_id = 0
    target_tokenizer.bos_token_id = 1
    target_tokenizer.eos_token_id = 2
    target_tokenizer.additional_special_tokens_ids = [4]
    donor_tokenizer = _Tokenizer(["<s>", "</s>", "<unk>", "<pad>", "<extra>"])
    donor_tokenizer.all_special_ids = [0, 1, 2, 3, 4]
    donor_tokenizer.bos_token_id = 0
    donor_tokenizer.eos_token_id = 1
    donor_tokenizer.unk_token_id = 2
    donor_tokenizer.pad_token_id = 3
    donor_tokenizer.additional_special_tokens_ids = [4]

    target_ids, donor_groups, donor_weights = _target_to_donor_special_token_group_specs(
        target_tokenizer,
        donor_tokenizer,
        seen_target_ids={3},
    )

    assert target_ids == [1, 2, 0, 4]
    assert donor_groups == [[0], [1], [3], [4]]
    assert donor_weights == [[1.0], [1.0], [1.0], [1.0]]


def test_target_to_donor_omp_group_specs_uses_sparse_character_pieces() -> None:
    torch = pytest.importorskip("torch")
    target_tokenizer = _Tokenizer(["<eos>", "ab", "42"])
    donor_tokenizer = _Tokenizer(["<eos>", "a", "b", "4", "2"])
    target_map = _canonical_token_map(target_tokenizer)
    donor_map = _canonical_token_map(donor_tokenizer)

    target_ids, donor_groups, donor_weights, omp_rows, weighted_rows = _target_to_donor_omp_group_specs(
        torch,
        target_tokenizer,
        donor_tokenizer,
        target_map,
        donor_map,
        max_selected=3,
    )

    assert target_ids == [1, 2]
    assert donor_groups[0] == [1, 2]
    assert donor_groups[1] == [3, 4]
    assert donor_weights[0] == pytest.approx([0.5, 0.5])
    assert donor_weights[1] == pytest.approx([0.5, 0.5])
    assert omp_rows == 2
    assert weighted_rows == 2


def test_recovery_same_vocab_guard_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="matching tokenizers"):
        _assert_same_vocab(_Tokenizer(["a", "b"]), _Tokenizer(["a", "c"]))


def test_append_eos_text_adds_missing_token_once() -> None:
    assert _append_eos_text(_Obj(eos_token="<eos>"), "answer") == "answer<eos>"
    assert _append_eos_text(_Obj(eos_token="<eos>"), "answer<eos>") == "answer<eos>"
    assert _append_eos_text(_Obj(eos_token=None), "answer") == "answer"


def test_join_prompt_response_preserves_answer_and_code_surfaces() -> None:
    assert _join_prompt_response("Answer:", "156") == "Answer: 156"
    assert _join_prompt_response("```python\n", "def add(a, b):") == "```python\ndef add(a, b):"
    assert _join_prompt_response("Prompt", "Response") == "Prompt\nResponse"


def test_recovery_defaults_do_not_reuse_evaluation_prompts() -> None:
    assert set(RECOVERY_TEXTS).isdisjoint(TEXT_EVALS)


def test_generation_health_checks_expected_first_number_when_present() -> None:
    spec = _normalize_generation_prompt(
        {
            "id": "math_exact",
            "category": "math",
            "prompt": "Question: What is 6 + 4? Answer with only the number.\nAnswer:",
            "kind": "first_number",
            "answer": "10",
            "relevance_rules": [],
        },
        0,
    )
    good = dict(spec)
    good["output"] = f"{spec['prompt']} 10"
    good["diagnostics"] = {
        "distinct_1": 1.0,
        "top_token_share": 1.0,
        "max_token_run": 1,
    }
    bad = dict(spec)
    bad["output"] = f"{spec['prompt']} 30"
    bad["diagnostics"] = {
        "distinct_1": 1.0,
        "top_token_share": 1.0,
        "max_token_run": 1,
    }

    good_check = _generation_health(good, min_continuation_chars=0, min_unique_token_ratio=0.0)
    bad_check = _generation_health(bad, min_continuation_chars=0, min_unique_token_ratio=0.0)

    assert good_check["passed"] is True
    assert good_check["expected_behavior"]["passed"] is True
    assert bad_check["passed"] is False
    assert bad_check["expected_enough"] is False
    assert bad_check["expected_behavior"]["details"]["prediction"] == "30"


def test_build_eval_suite_includes_exact_generation_math_and_code(tmp_path) -> None:
    output = tmp_path / "suite.json"

    report = build_eval_suite(
        output,
        behavior_count=0,
        generation_count=14,
        choice_count=0,
        text_count=0,
        seed=7,
    )

    suite = json.loads(output.read_text(encoding="utf-8"))
    prompts = suite["suite"]["generation_prompts"]
    exact_math = [item for item in prompts if item.get("kind") == "first_number"]
    exact_code = [item for item in prompts if item.get("kind") == "python_function"]

    assert exact_math
    assert exact_math[0]["category"] == "math"
    assert exact_math[0]["answer"].lstrip("-").isdigit()
    assert exact_code
    assert exact_code[0]["category"] == "code"
    assert exact_code[0]["function_name"] == "add"
    assert report["counts"]["generation_prompts"] == 14
    assert output.exists()


def test_load_calibration_texts_handles_jsonl_message_rows(tmp_path) -> None:
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        "\n".join(
            [
                '{"messages":[{"role":"user","content":"Explain folding."},{"role":"assistant","content":"Fold redundant channels before pruning."}]}',
                '{"prompt":"Solve 2+2.","target":"The answer is 4."}',
                '{"prompt":"Name a merge risk.","response":"Tokenizer mismatch can break behavior."}',
                '{"prompt":"Answer:","response":"156"}',
                '{"input_json":{"messages":[{"role":"user","content":"Return compact JSON."}]},"target_json":{"content":"Use concise JSON.","media_refs":["ignore-me"]}}',
            ]
        ),
        encoding="utf-8",
    )
    texts = load_calibration_texts(calibration, max_texts=5, min_chars=8)
    assert len(texts) == 5
    assert "user: Explain folding." in texts[0]
    assert "assistant: Fold redundant channels" in texts[0]
    assert "\nassistant:" in texts[0]
    assert texts[1] == "Solve 2+2.\nThe answer is 4."
    assert texts[2] == "Name a merge risk.\nTokenizer mismatch can break behavior."
    assert texts[3] == "Answer: 156"
    assert texts[4] == "user: Return compact JSON.\nUse concise JSON."


def test_build_recovery_calibration_writes_prompt_response_jsonl(tmp_path) -> None:
    output = tmp_path / "recovery.jsonl"

    report = build_recovery_calibration(
        output,
        math_count=3,
        code_count=2,
        tokenizer_count=1,
        model_surgery_count=1,
        commonsense_count=1,
        seed=13,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    categories = {row["category"] for row in rows}
    math_rows = [row for row in rows if row["category"] == "math"]
    code_rows = [row for row in rows if row["category"] == "code"]
    texts = load_calibration_texts(output, max_texts=8, min_chars=8)

    assert report["operation"] == "build_recovery_calibration"
    assert report["counts"]["math"] == 3
    assert report["total_rows"] == 8
    assert len(rows) == 8
    assert categories == {"math", "code", "tokenizer", "model_surgery", "commonsense"}
    assert all(row["response"].lstrip("-").isdigit() for row in math_rows)
    assert any("def add" in row["response"] for row in code_rows)
    assert any("Answer:" in text for text in texts)
    assert any("def " in text for text in texts)


def test_build_recovery_calibration_excludes_eval_suite_prompts(tmp_path) -> None:
    output = tmp_path / "recovery.jsonl"
    suite = tmp_path / "heldout.json"
    heldout_prompt = "In one sentence, why can tokenizer mismatch hurt model stitching?\nAnswer:"
    suite.write_text(
        json.dumps(
            {
                "suite": {
                    "text_evals": [],
                    "choice_probes": [],
                    "generation_prompts": [],
                    "behavior_probes": [
                        {
                            "id": "heldout_tokenizer",
                            "category": "tokenizer",
                            "prompt": heldout_prompt,
                            "kind": "pattern_groups",
                            "pattern_groups": [],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_recovery_calibration(
        output,
        math_count=0,
        code_count=0,
        tokenizer_count=3,
        model_surgery_count=0,
        commonsense_count=0,
        seed=2,
        exclude_eval_suite=suite,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert report["excluded_eval_suite"] == str(suite)
    assert report["excluded_prompt_count"] == 1
    assert report["excluded_row_count"] == 2
    assert report["excluded_counts"] == {"tokenizer": 2}
    assert report["total_rows"] == 1
    assert all(row["prompt"] != heldout_prompt for row in rows)


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


def test_activation_fit_linears_default_collects_parent_rows() -> None:
    assert _effective_activation_fit_max_rows(True, 0, 2048) == 2048
    assert _effective_activation_fit_max_rows(True, 64, 2048) == 64
    assert _effective_activation_fit_max_rows(False, 0, 2048) == 0


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
            "--target-base-model",
            "HuggingFaceTB/SmolLM2-135M",
            "--target-tokenizer-model",
            "HuggingFaceTB/SmolLM2-135M",
            "--donor-model",
            "Qwen/Qwen3-4B-Base,openai-community/gpt2",
            "--donor-weights",
            "0.7,0.3",
            "--seed",
            "456",
            "--max-anchors",
            "1000",
            "--fold-mode",
            "mean",
            "--hidden-basis-mode",
            "shared-token-ridge",
            "--hidden-basis-scope",
            "layer",
            "--hidden-basis-max-rows",
            "777",
            "--hidden-basis-preprocess",
            "uncentered-rowrms",
            "--activation-fold-assignment",
            "similarity",
            "--transport-backend",
            "sinkhorn",
            "--transport-support",
            "quantile",
            "--donor-layer-fold-mode",
            "fold",
            "--donor-layer-group-strategy",
            "activation-energy",
            "--attention-fold-mode",
            "coupled",
            "--attention-head-dim-transport",
            "qk-vo",
            "--token-init-mode",
            "donor-omp",
            "--projection-scale",
            "geomean-rms",
            "--projection-blend-alpha",
            "0.35",
            "--consumer-hidden-reduction",
            "sqrt-sum",
            "--activation-fit-linears",
            "--activation-fit-target",
            "attention-output",
            "--activation-fit-max-rows",
            "88",
            "--activation-fit-alpha",
            "0.6",
            "--activation-fit-ridge",
            "0.2",
            "--activation-fit-rank",
            "17",
            "--activation-fit-rank-mode",
            "delta",
            "--activation-fit-method",
            "lrc",
            "--activation-fit-factor-steps",
            "9",
            "--activation-fit-factor-lr",
            "0.07",
            "--activation-fit-factor-l2",
            "0.005",
            "--activation-fit-gate",
            "validation",
            "--activation-fit-validation-fraction",
            "0.25",
            "--activation-fit-min-validation-rows",
            "12",
            "--activation-fit-min-relative-improvement",
            "0.03",
            "--activation-fit-max-cosine-drop",
            "0.04",
            "--activation-feature-mode",
            "covariance",
            "--activation-subspace-rank",
            "6",
            "--activation-domain-balance",
            "keyword",
            "--untie-output-head",
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
            "--eval-suite",
            "mlp-suite.json",
            "--max-text-evals",
            "11",
            "--max-choice-probes",
            "22",
            "--max-generation-prompts",
            "33",
            "--max-behavior-probes",
            "1000",
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
            "--eval-suite",
            "compress-suite.json",
            "--max-text-evals",
            "12",
            "--max-choice-probes",
            "23",
            "--max-generation-prompts",
            "34",
            "--max-behavior-probes",
            "1000",
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
            "--first-token-lm-weight",
            "0.44",
            "--response-end-lm-weight",
            "0.52",
            "--focused-lm-weight",
            "0.31",
            "--hidden-kd-weight",
            "0.2",
            "--span-kd-weight",
            "0.15",
            "--span-hidden-kd-weight",
            "0.12",
            "--dual-space-kd-weight",
            "0.25",
            "--attention-influence-weight",
            "0.16",
            "--attention-influence-layer",
            "middle",
            "--projected-vocab-kd-weight",
            "0.22",
            "--focused-projected-vocab-kd-weight",
            "0.18",
            "--projected-vocab-kd-mode",
            "student-to-teacher",
            "--projected-vocab-max-source-tokens",
            "6",
            "--reference-model",
            "host-reference",
            "--reference-kd-weight",
            "0.35",
            "--unlikelihood-weight",
            "0.08",
            "--unlikelihood-window",
            "24",
            "--entropy-weight",
            "0.006",
            "--token-kd-weight",
            "0.13",
            "--token-focus",
            "numbers",
            "--token-max-pairs",
            "44",
            "--token-teacher-confidence-floor",
            "0.2",
            "--alm-kd-weight",
            "0.17",
            "--alm-max-chunks",
            "33",
            "--alm-min-chars",
            "4",
            "--answer-contrastive-weight",
            "0.09",
            "--answer-contrastive-max-negatives",
            "3",
            "--answer-first-token-margin-weight",
            "0.19",
            "--answer-first-token-margin",
            "1.4",
            "--answer-first-token-max-negatives",
            "9",
            "--teacher-rank-weight",
            "0.11",
            "--teacher-rank-temperature",
            "0.7",
            "--teacher-rank-max-negatives",
            "5",
            "--teacher-generate-lm-weight",
            "0.24",
            "--teacher-generate-max-new-tokens",
            "31",
            "--stepwise-donor-kd-weight",
            "0.26",
            "--stepwise-donor-max-new-tokens",
            "11",
            "--on-policy-anchor-kd-weight",
            "0.23",
            "--on-policy-unlikelihood-weight",
            "0.37",
            "--on-policy-hidden-kd-weight",
            "0.29",
            "--on-policy-max-new-tokens",
            "19",
            "--on-policy-interval",
            "4",
            "--append-eos",
            "--entropy-aware-kd-weight",
            "0.45",
            "--entropy-aware-kd-threshold",
            "0.61",
            "--entropy-aware-kd-sharpness",
            "9.5",
            "--span-max-pairs",
            "66",
            "--hidden-ridge",
            "0.02",
            "--hidden-layer",
            "middle",
            "--hidden-kd-metric",
            "cka",
            "--hidden-max-anchors",
            "77",
            "--temperature",
            "1.5",
            "--max-anchors",
            "123",
            "--anchor-selection",
            "longest",
            "--anchor-min-chars",
            "3",
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
            "--report-interval",
            "5",
            "--save-best-loss",
            "--save-best-behavior",
            "--behavior-eval-interval",
            "4",
            "--behavior-max-new-tokens",
            "22",
            "--behavior-max-probes",
            "5",
            "--behavior-eval-suite",
            "suite.json",
            "--behavior-max-generation-prompts",
            "7",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "6",
        ]
    )
    xfamily_head_args = parser.parse_args(
        [
            "hf-xfamily-calibrate-head",
            "donor",
            "candidate",
            "head-out",
            "--ridge",
            "0.04",
            "--head-blend-alpha",
            "0.7",
            "--head-calibration-mode",
            "mean-shift",
            "--max-length",
            "80",
            "--max-calibration-rows",
            "222",
            "--projected-vocab-max-source-tokens",
            "5",
            "--dtype",
            "float16",
            "--candidate-dtype",
            "float32",
            "--device",
            "cpu",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "9",
        ]
    )
    recover_args = parser.parse_args(
        [
            "hf-recover",
            "source",
            "candidate",
            "recover-out",
            "--kd-weight",
            "0.4",
            "--lm-weight",
            "0.2",
            "--hidden-kd-weight",
            "0.8",
            "--hidden-kd-metric",
            "both",
            "--hidden-layer",
            "all",
            "--hidden-token-mode",
            "response",
            "--hidden-ridge",
            "0.02",
            "--hidden-max-anchors",
            "123",
            "--candidate-train-dtype",
            "float32",
            "--top-k-kd-weight",
            "0.6",
            "--top-k-kd-k",
            "7",
            "--first-token-top-k-kd-weight",
            "1.1",
            "--first-token-kd-weight",
            "1.5",
            "--first-token-lm-weight",
            "0.75",
            "--response-loss-mode",
            "auto",
            "--save-best-behavior",
            "--behavior-eval-suite",
            "same-suite.json",
            "--behavior-max-generation-prompts",
            "4",
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
    assert donor_init_args.target_base_model == "HuggingFaceTB/SmolLM2-135M"
    assert donor_init_args.target_tokenizer_model == "HuggingFaceTB/SmolLM2-135M"
    assert donor_init_args.donor_model == "Qwen/Qwen3-4B-Base,openai-community/gpt2"
    assert donor_init_args.donor_weights == "0.7,0.3"
    assert donor_init_args.seed == 456
    assert donor_init_args.max_anchors == 1000
    assert donor_init_args.fold_mode == "mean"
    assert donor_init_args.hidden_basis_mode == "shared-token-ridge"
    assert donor_init_args.hidden_basis_scope == "layer"
    assert donor_init_args.hidden_basis_max_rows == 777
    assert donor_init_args.hidden_basis_preprocess == "uncentered-rowrms"
    assert donor_init_args.activation_fold_assignment == "similarity"
    assert donor_init_args.transport_backend == "sinkhorn"
    assert donor_init_args.transport_support == "quantile"
    assert donor_init_args.donor_layer_fold_mode == "fold"
    assert donor_init_args.donor_layer_group_strategy == "activation-energy"
    assert donor_init_args.attention_fold_mode == "coupled"
    assert donor_init_args.attention_head_dim_transport == "qk-vo"
    assert donor_init_args.token_init_mode == "donor-omp"
    assert donor_init_args.projection_scale == "geomean-rms"
    assert donor_init_args.projection_blend_alpha == 0.35
    assert donor_init_args.consumer_hidden_reduction == "sqrt-sum"
    assert donor_init_args.activation_fit_linears is True
    assert donor_init_args.activation_fit_target == "attention-output"
    assert donor_init_args.activation_fit_max_rows == 88
    assert donor_init_args.activation_fit_alpha == pytest.approx(0.6)
    assert donor_init_args.activation_fit_ridge == pytest.approx(0.2)
    assert donor_init_args.activation_fit_rank == 17
    assert donor_init_args.activation_fit_rank_mode == "delta"
    assert donor_init_args.activation_fit_method == "lrc"
    assert donor_init_args.activation_fit_factor_steps == 9
    assert donor_init_args.activation_fit_factor_lr == pytest.approx(0.07)
    assert donor_init_args.activation_fit_factor_l2 == pytest.approx(0.005)
    assert donor_init_args.activation_fit_gate == "validation"
    assert donor_init_args.activation_fit_validation_fraction == pytest.approx(0.25)
    assert donor_init_args.activation_fit_min_validation_rows == 12
    assert donor_init_args.activation_fit_min_relative_improvement == pytest.approx(0.03)
    assert donor_init_args.activation_fit_max_cosine_drop == pytest.approx(0.04)
    assert donor_init_args.activation_feature_mode == "covariance"
    assert donor_init_args.activation_subspace_rank == 6
    assert donor_init_args.activation_domain_balance == "keyword"
    assert donor_init_args.untie_output_head is True
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
    assert xfamily_recover_args.reference_model == "host-reference"
    assert xfamily_recover_args.reference_kd_weight == 0.35
    assert xfamily_recover_args.unlikelihood_weight == 0.08
    assert xfamily_recover_args.unlikelihood_window == 24
    assert xfamily_recover_args.entropy_weight == 0.006
    assert xfamily_recover_args.token_kd_weight == 0.13
    assert xfamily_recover_args.token_focus == "numbers"
    assert xfamily_recover_args.token_max_pairs == 44
    assert xfamily_recover_args.token_teacher_confidence_floor == 0.2
    assert xfamily_recover_args.alm_kd_weight == pytest.approx(0.17)
    assert xfamily_recover_args.alm_max_chunks == 33
    assert xfamily_recover_args.alm_min_chars == 4
    assert xfamily_recover_args.answer_contrastive_weight == 0.09
    assert xfamily_recover_args.answer_contrastive_max_negatives == 3
    assert xfamily_recover_args.answer_first_token_margin_weight == pytest.approx(0.19)
    assert xfamily_recover_args.answer_first_token_margin == pytest.approx(1.4)
    assert xfamily_recover_args.answer_first_token_max_negatives == 9
    assert xfamily_recover_args.teacher_rank_weight == 0.11
    assert xfamily_recover_args.teacher_rank_temperature == 0.7
    assert xfamily_recover_args.teacher_rank_max_negatives == 5
    assert xfamily_recover_args.teacher_generate_lm_weight == pytest.approx(0.24)
    assert xfamily_recover_args.teacher_generate_max_new_tokens == 31
    assert xfamily_recover_args.stepwise_donor_kd_weight == pytest.approx(0.26)
    assert xfamily_recover_args.stepwise_donor_max_new_tokens == 11
    assert xfamily_recover_args.on_policy_anchor_kd_weight == pytest.approx(0.23)
    assert xfamily_recover_args.on_policy_unlikelihood_weight == pytest.approx(0.37)
    assert xfamily_recover_args.on_policy_hidden_kd_weight == pytest.approx(0.29)
    assert xfamily_recover_args.on_policy_max_new_tokens == 19
    assert xfamily_recover_args.on_policy_interval == 4
    assert xfamily_recover_args.append_eos is True
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
    assert xfamily_mlp_args.eval_suite == "mlp-suite.json"
    assert xfamily_mlp_args.max_text_evals == 11
    assert xfamily_mlp_args.max_choice_probes == 22
    assert xfamily_mlp_args.max_generation_prompts == 33
    assert xfamily_mlp_args.max_behavior_probes == 1000
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
    assert xfamily_compress_args.eval_suite == "compress-suite.json"
    assert xfamily_compress_args.max_text_evals == 12
    assert xfamily_compress_args.max_choice_probes == 23
    assert xfamily_compress_args.max_generation_prompts == 34
    assert xfamily_compress_args.max_behavior_probes == 1000
    assert xfamily_recover_args.source_model == "donor"
    assert xfamily_recover_args.candidate_model == "candidate"
    assert xfamily_recover_args.output == "recovered"
    assert xfamily_recover_args.anchor_kd_weight == 0.7
    assert xfamily_recover_args.lm_weight == 0.3
    assert xfamily_recover_args.first_token_lm_weight == pytest.approx(0.44)
    assert xfamily_recover_args.response_end_lm_weight == pytest.approx(0.52)
    assert xfamily_recover_args.focused_lm_weight == pytest.approx(0.31)
    assert xfamily_recover_args.hidden_kd_weight == 0.2
    assert xfamily_recover_args.span_kd_weight == 0.15
    assert xfamily_recover_args.span_hidden_kd_weight == 0.12
    assert xfamily_recover_args.dual_space_kd_weight == pytest.approx(0.25)
    assert xfamily_recover_args.attention_influence_weight == pytest.approx(0.16)
    assert xfamily_recover_args.attention_influence_layer == "middle"
    assert xfamily_recover_args.projected_vocab_kd_weight == 0.22
    assert xfamily_recover_args.focused_projected_vocab_kd_weight == pytest.approx(0.18)
    assert xfamily_recover_args.projected_vocab_kd_mode == "student-to-teacher"
    assert xfamily_recover_args.projected_vocab_max_source_tokens == 6
    assert xfamily_recover_args.unlikelihood_weight == 0.08
    assert xfamily_recover_args.unlikelihood_window == 24
    assert xfamily_recover_args.entropy_weight == 0.006
    assert xfamily_recover_args.entropy_aware_kd_weight == 0.45
    assert xfamily_recover_args.entropy_aware_kd_threshold == 0.61
    assert xfamily_recover_args.entropy_aware_kd_sharpness == 9.5
    assert xfamily_recover_args.span_max_pairs == 66
    assert xfamily_recover_args.hidden_ridge == 0.02
    assert xfamily_recover_args.hidden_layer == "middle"
    assert xfamily_recover_args.hidden_kd_metric == "cka"
    assert xfamily_recover_args.hidden_max_anchors == 77
    assert xfamily_recover_args.max_anchors == 123
    assert xfamily_recover_args.anchor_selection == "longest"
    assert xfamily_recover_args.anchor_min_chars == 3
    assert xfamily_recover_args.train_scope == "last2"
    assert xfamily_recover_args.dtype == "float16"
    assert xfamily_recover_args.candidate_dtype == "float32"
    assert xfamily_recover_args.report_interval == 5
    assert xfamily_recover_args.save_best_loss is True
    assert xfamily_recover_args.save_best_behavior is True
    assert xfamily_recover_args.behavior_eval_interval == 4
    assert xfamily_recover_args.behavior_max_new_tokens == 22
    assert xfamily_recover_args.behavior_max_probes == 5
    assert xfamily_recover_args.behavior_eval_suite == "suite.json"
    assert xfamily_recover_args.behavior_max_generation_prompts == 7
    assert xfamily_recover_args.calibration_file == "calibration.jsonl"
    assert xfamily_head_args.source_model == "donor"
    assert xfamily_head_args.candidate_model == "candidate"
    assert xfamily_head_args.output == "head-out"
    assert xfamily_head_args.ridge == 0.04
    assert xfamily_head_args.head_blend_alpha == 0.7
    assert xfamily_head_args.max_length == 80
    assert xfamily_head_args.max_calibration_rows == 222
    assert xfamily_head_args.projected_vocab_max_source_tokens == 5
    assert xfamily_head_args.dtype == "float16"
    assert xfamily_head_args.candidate_dtype == "float32"
    assert xfamily_head_args.device == "cpu"
    assert xfamily_head_args.calibration_file == "calibration.jsonl"
    assert xfamily_head_args.max_calibration_texts == 9
    assert recover_args.source_model == "source"
    assert recover_args.candidate_model == "candidate"
    assert recover_args.hidden_kd_weight == 0.8
    assert recover_args.hidden_kd_metric == "both"
    assert recover_args.hidden_layer == "all"
    assert recover_args.hidden_token_mode == "response"
    assert recover_args.hidden_ridge == 0.02
    assert recover_args.hidden_max_anchors == 123
    assert recover_args.candidate_train_dtype == "float32"
    assert recover_args.top_k_kd_weight == 0.6
    assert recover_args.top_k_kd_k == 7
    assert recover_args.first_token_top_k_kd_weight == 1.1
    assert recover_args.first_token_kd_weight == 1.5
    assert recover_args.first_token_lm_weight == 0.75
    assert recover_args.response_loss_mode == "auto"
    assert recover_args.save_best_behavior is True
    assert recover_args.behavior_eval_suite == "same-suite.json"
    assert recover_args.behavior_max_generation_prompts == 4
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
            "--unlikelihood-weight",
            "0.07",
            "--first-token-lm-weight",
            "1.25",
            "--unlikelihood-window",
            "20",
            "--entropy-weight",
            "0.004",
            "--response-loss-mode",
            "answer",
            "--append-eos",
            "--train-scope",
            "last1",
            "--behavior-eval-suite",
            "lm-suite.json",
            "--behavior-max-generation-prompts",
            "8",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "9",
        ]
    )
    assert lm_recover_args.candidate_model == "candidate"
    assert lm_recover_args.output == "out"
    assert lm_recover_args.steps == 5
    assert lm_recover_args.unlikelihood_weight == 0.07
    assert lm_recover_args.first_token_lm_weight == 1.25
    assert lm_recover_args.unlikelihood_window == 20
    assert lm_recover_args.entropy_weight == 0.004
    assert lm_recover_args.response_loss_mode == "answer"
    assert lm_recover_args.append_eos is True
    assert lm_recover_args.train_scope == "last1"
    assert lm_recover_args.behavior_eval_suite == "lm-suite.json"
    assert lm_recover_args.behavior_max_generation_prompts == 8
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

    xfamily_reconstruct_args = parser.parse_args(
        [
            "hf-xfamily-reconstruct",
            "donor",
            "candidate",
            "out",
            "--ridge",
            "0.04",
            "--reconstruction-alpha",
            "0.7",
            "--reconstruction-target",
            "block-interface",
            "--reconstruction-objective",
            "cosine",
            "--cosine-steps",
            "8",
            "--cosine-lr",
            "0.025",
            "--cosine-l2",
            "0.002",
            "--max-length",
            "88",
            "--max-tokens-per-layer",
            "55",
            "--max-anchors",
            "444",
            "--hidden-projection-scope",
            "layer-input",
            "--layer-limit",
            "3",
            "--fit-gate",
            "validation",
            "--fit-gate-validation-fraction",
            "0.2",
            "--fit-gate-min-validation-rows",
            "5",
            "--fit-gate-min-relative-improvement",
            "0.01",
            "--fit-gate-max-cosine-drop",
            "0.03",
            "--layer-gate-metric",
            "behavior",
            "--layer-gate-max-ppl-ratio",
            "1.01",
            "--layer-gate-max-texts",
            "2",
            "--layer-gate-max-behavior-drop",
            "0.17",
            "--layer-gate-behavior-max-new-tokens",
            "33",
            "--layer-gate-eval-suite",
            "reconstruct-suite.json",
            "--layer-gate-max-generation-prompts",
            "44",
            "--dtype",
            "float16",
            "--candidate-dtype",
            "float32",
            "--device",
            "cpu",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "7",
        ]
    )
    assert xfamily_reconstruct_args.source_model == "donor"
    assert xfamily_reconstruct_args.candidate_model == "candidate"
    assert xfamily_reconstruct_args.output == "out"
    assert xfamily_reconstruct_args.ridge == 0.04
    assert xfamily_reconstruct_args.reconstruction_alpha == 0.7
    assert xfamily_reconstruct_args.reconstruction_target == "block-interface"
    assert xfamily_reconstruct_args.reconstruction_objective == "cosine"
    assert xfamily_reconstruct_args.cosine_steps == 8
    assert xfamily_reconstruct_args.cosine_lr == 0.025
    assert xfamily_reconstruct_args.cosine_l2 == 0.002
    assert xfamily_reconstruct_args.max_length == 88
    assert xfamily_reconstruct_args.max_tokens_per_layer == 55
    assert xfamily_reconstruct_args.max_anchors == 444
    assert xfamily_reconstruct_args.hidden_projection_scope == "layer-input"
    assert xfamily_reconstruct_args.layer_limit == 3
    assert xfamily_reconstruct_args.fit_gate == "validation"
    assert xfamily_reconstruct_args.fit_gate_validation_fraction == pytest.approx(0.2)
    assert xfamily_reconstruct_args.fit_gate_min_validation_rows == 5
    assert xfamily_reconstruct_args.fit_gate_min_relative_improvement == pytest.approx(0.01)
    assert xfamily_reconstruct_args.fit_gate_max_cosine_drop == pytest.approx(0.03)
    assert xfamily_reconstruct_args.layer_gate_metric == "behavior"
    assert xfamily_reconstruct_args.layer_gate_max_ppl_ratio == pytest.approx(1.01)
    assert xfamily_reconstruct_args.layer_gate_max_texts == 2
    assert xfamily_reconstruct_args.layer_gate_max_behavior_drop == pytest.approx(0.17)
    assert xfamily_reconstruct_args.layer_gate_behavior_max_new_tokens == 33
    assert xfamily_reconstruct_args.layer_gate_eval_suite == "reconstruct-suite.json"
    assert xfamily_reconstruct_args.layer_gate_max_generation_prompts == 44
    assert xfamily_reconstruct_args.dtype == "float16"
    assert xfamily_reconstruct_args.candidate_dtype == "float32"
    assert xfamily_reconstruct_args.device == "cpu"
    assert xfamily_reconstruct_args.calibration_file == "calibration.jsonl"
    assert xfamily_reconstruct_args.max_calibration_texts == 7


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


def test_cross_reconstruction_proportional_layer_index_uses_bin_centers() -> None:
    assert _proportional_layer_index(0, 12, 36) == 1
    assert _proportional_layer_index(11, 12, 36) == 34
    assert _proportional_layer_index(0, 1, 36) == 18


def test_proportional_layer_index_maps_depths() -> None:
    assert [_proportional_index(idx, 4, 2) for idx in range(4)] == [0, 0, 1, 1]
    assert [_proportional_index(idx, 3, 5) for idx in range(3)] == [0, 2, 4]


def test_donor_layer_groups_partition_parent_depth() -> None:
    groups = [_donor_layer_group_for_target(idx, target_layer_count=3, donor_layer_count=8) for idx in range(3)]

    assert groups == [[0, 1], [2, 3, 4], [5, 6, 7]]
    assert [item for group in groups for item in group] == list(range(8))


def test_donor_layer_groups_can_partition_by_activation_energy() -> None:
    groups = _donor_layer_groups_by_energy([1.0, 1.0, 8.0, 1.0, 1.0, 1.0], target_layer_count=3)

    assert groups == [[0, 1], [2], [3, 4, 5]]
    assert [item for group in groups for item in group] == list(range(6))
    assert _representative_donor_layer_for_group([0, 1, 2], [0.1, 0.9, 0.2]) == 1


def test_donor_layer_energy_groups_are_depth_ratio_capped() -> None:
    scores = [0.2] * 36
    scores[6] = 22.0

    groups = _donor_layer_groups_by_energy(scores, target_layer_count=30)

    assert [item for group in groups for item in group] == list(range(36))
    assert max(len(group) for group in groups) <= 2
    assert [6] in groups


def test_donor_layer_groups_report_activation_energy_scores() -> None:
    torch = pytest.importorskip("torch")
    activation_profile = {
        "hidden_basis_input_rows": {
            0: torch.zeros(2, 3),
            1: torch.zeros(2, 3),
            2: torch.zeros(2, 3),
        },
        "hidden_basis_output_rows": {
            0: torch.ones(2, 3),
            1: torch.ones(2, 3) * 4,
            2: torch.ones(2, 3),
        },
    }

    groups, report = _donor_layer_groups_for_target_count(
        torch,
        target_layer_count=2,
        donor_layer_count=3,
        strategy="activation-energy",
        activation_profile=activation_profile,
    )

    assert [item for group in groups for item in group] == [0, 1, 2]
    assert report["strategy"] == "activation-energy"
    assert report["delta_scores"][1] > report["delta_scores"][0]


def test_linear_cka_detects_similar_activation_spaces() -> None:
    torch = pytest.importorskip("torch")
    first = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    second = first * 2.0
    unrelated = torch.tensor([[0.0, -1.0], [0.0, -1.0], [0.0, 1.0], [0.0, 1.0]])

    assert _linear_cka(torch, first, second) == pytest.approx(1.0)
    assert _linear_cka(torch, first, unrelated) == pytest.approx(0.0)


def test_hidden_representation_loss_supports_direct_cka_dimension_mismatch() -> None:
    torch = pytest.importorskip("torch")
    student = torch.tensor(
        [[-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        requires_grad=False,
    )

    cka = _linear_cka_tensor(torch, student, teacher)
    loss = _hidden_representation_loss(torch, student, teacher, hidden_projection=None, hidden_kd_metric="cka")
    loss.backward()

    assert cka.item() == pytest.approx(1.0)
    assert loss.item() == pytest.approx(0.0)
    assert student.grad is not None


def test_row_gram_geometry_loss_is_dimension_free_and_order_sensitive() -> None:
    torch = pytest.importorskip("torch")
    student = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, -2.0, 0.0]],
        requires_grad=False,
    )
    scrambled_teacher = teacher[[0, 2, 1, 3]]

    aligned = _row_gram_geometry_loss(torch, student, teacher)
    scrambled = _row_gram_geometry_loss(torch, student, scrambled_teacher)
    aligned.backward()

    assert aligned.item() == pytest.approx(0.0)
    assert scrambled.item() > aligned.item()
    assert student.grad is not None


def test_hidden_representation_loss_projected_geometry_combines_projection_and_row_geometry() -> None:
    torch = pytest.importorskip("torch")
    student = torch.randn(6, 2, requires_grad=True)
    teacher = torch.randn(6, 3)

    with pytest.raises(ValueError, match="hidden_projection"):
        _hidden_representation_loss(torch, student, teacher, hidden_projection=None, hidden_kd_metric="projected-geometry")

    projection = torch.randn(3, 2)
    loss = _hidden_representation_loss(torch, student, teacher, projection, hidden_kd_metric="projected-geometry")
    loss.backward()

    assert torch.isfinite(loss)
    assert student.grad is not None


def test_feature_decorrelation_loss_penalizes_duplicate_axes() -> None:
    torch = pytest.importorskip("torch")
    duplicate = torch.tensor(
        [[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
        requires_grad=True,
    )
    independent = torch.eye(4, dtype=torch.float32)

    duplicate_loss = _feature_decorrelation_loss(torch, duplicate)
    independent_loss = _feature_decorrelation_loss(torch, independent)
    duplicate_loss.backward()

    assert duplicate_loss.item() > independent_loss.item()
    assert duplicate.grad is not None


def test_hidden_representation_loss_rsd_is_dimension_free_and_decorrelates() -> None:
    torch = pytest.importorskip("torch")
    student = torch.tensor(
        [[-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        requires_grad=False,
    )

    loss = _hidden_representation_loss(torch, student, teacher, hidden_projection=None, hidden_kd_metric="rsd")
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert student.grad is not None


def test_hidden_representation_loss_projected_rsd_requires_projection() -> None:
    torch = pytest.importorskip("torch")
    student = torch.randn(4, 2)
    teacher = torch.randn(4, 3)

    with pytest.raises(ValueError, match="hidden_projection"):
        _hidden_representation_loss(torch, student, teacher, hidden_projection=None, hidden_kd_metric="projected-rsd")

    projection = torch.randn(3, 2)
    loss = _hidden_representation_loss(torch, student, teacher, projection, hidden_kd_metric="projected-rsd")

    assert torch.isfinite(loss)


def test_source_group_delta_state_matches_fold_and_average_modes() -> None:
    torch = pytest.importorskip("torch")
    hidden_states = [
        torch.full((1, 2, 1), 0.0),
        torch.full((1, 2, 1), 1.0),
        torch.full((1, 2, 1), 3.0),
        torch.full((1, 2, 1), 6.0),
    ]

    folded = _source_group_delta_state(torch, hidden_states, [1, 2], "fold")
    averaged = _source_group_delta_state(torch, hidden_states, [1, 2], "average")
    representative = _source_group_delta_state(torch, hidden_states, [1, 2], "representative")

    assert folded.squeeze().tolist() == [5.0, 5.0]
    assert averaged.squeeze().tolist() == [2.5, 2.5]
    assert representative.squeeze().tolist() == [2.0, 2.0]


def test_source_group_module_output_state_matches_fold_and_average_modes() -> None:
    torch = pytest.importorskip("torch")
    outputs = {
        1: torch.full((1, 2, 1), 2.0),
        2: torch.full((1, 2, 1), 5.0),
    }

    folded = _source_group_module_output_state(torch, outputs, [1, 2], "fold")
    averaged = _source_group_module_output_state(torch, outputs, [1, 2], "average")
    representative = _source_group_module_output_state(torch, outputs, [1, 2], "representative")

    assert folded.squeeze().tolist() == [7.0, 7.0]
    assert averaged.squeeze().tolist() == [3.5, 3.5]
    assert representative.squeeze().tolist() == [2.0, 2.0]


def test_activation_row_transport_loss_handles_permuted_rows() -> None:
    torch = pytest.importorskip("torch")
    teacher = torch.eye(4, dtype=torch.float32)
    student_permuted = teacher[[2, 0, 3, 1]]

    paired = _activation_row_transport_loss(
        torch,
        student_permuted,
        teacher,
        mode="paired",
        temperature=0.05,
        iterations=16,
    )
    transported = _activation_row_transport_loss(
        torch,
        student_permuted,
        teacher,
        mode="sinkhorn",
        temperature=0.05,
        iterations=16,
    )

    assert transported.item() < paired.item()
    assert transported.item() < 0.2


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


def test_activation_hidden_basis_is_orthonormal() -> None:
    torch = pytest.importorskip("torch")
    rows = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    basis, report = _build_activation_hidden_basis(torch, rows, source_dim=3, target_dim=2)

    assert basis.shape == (3, 2)
    assert basis.t().matmul(basis).flatten().tolist() == pytest.approx(torch.eye(2).flatten().tolist(), abs=1e-5)
    assert report["mode"] == "activation-pca"
    assert report["retained_energy"] > 0.99


def test_activation_hidden_basis_uncentered_keeps_common_direction() -> None:
    torch = pytest.importorskip("torch")
    rows = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [10.0, -1.0, 0.0],
            [10.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    centered, centered_report = _build_activation_hidden_basis(
        torch,
        rows,
        source_dim=3,
        target_dim=1,
        preprocess="centered-rowrms",
    )
    uncentered, uncentered_report = _build_activation_hidden_basis(
        torch,
        rows,
        source_dim=3,
        target_dim=1,
        preprocess="uncentered-rowrms",
    )

    assert centered_report["preprocess"] == "centered-rowrms"
    assert uncentered_report["preprocess"] == "uncentered-rowrms"
    assert abs(float(uncentered[0, 0])) > 0.95
    assert abs(float(centered[0, 0])) < 0.5


def test_shared_token_hidden_basis_recovers_anchor_projection() -> None:
    torch = pytest.importorskip("torch")
    donor_rows = torch.eye(3, dtype=torch.float32)
    expected = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    class _Model:
        def __init__(self, rows):
            self._embedding = torch.nn.Embedding.from_pretrained(rows, freeze=False)

        def get_input_embeddings(self):
            return self._embedding

    target = _Model(torch.cat([torch.zeros((1, 2)), expected], dim=0))
    donor = _Model(torch.cat([torch.zeros((1, 3)), donor_rows], dim=0))
    tokenizer = _Tokenizer(["<s>", "alpha", "beta", "gamma"])

    basis, report = _build_shared_token_hidden_basis(
        torch,
        target,
        donor,
        tokenizer,
        tokenizer,
        max_anchors=8,
        ridge=0.0,
    )

    assert basis.shape == (3, 2)
    assert basis.flatten().tolist() == pytest.approx(expected.flatten().tolist(), abs=1e-6)
    assert report["mode"] == "shared-token-ridge"
    assert report["anchors"] == 3
    assert report["relative_error"] == pytest.approx(0.0, abs=1e-6)
    assert report["mean_cosine"] == pytest.approx(1.0, abs=1e-6)


def test_hidden_basis_resize_projects_hidden_axes() -> None:
    torch = pytest.importorskip("torch")
    source = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
    basis = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    resized = _resize_tensor_to_shape(torch, source, (2, 2), hidden_basis=basis)

    assert resized.flatten().tolist() == pytest.approx([1.0, 2.0, 4.0, 5.0])


def test_layer_hidden_basis_axes_use_input_and_output_interfaces() -> None:
    torch = pytest.importorskip("torch")
    input_basis = torch.eye(3, dtype=torch.float32)[:, :2]
    output_basis = torch.flip(torch.eye(3, dtype=torch.float32), dims=[0])[:, :2]
    layer_basis = {"input": input_basis, "output": output_basis}

    q_axes = _hidden_basis_axes_for_layer_tensor(
        torch,
        torch.zeros((4, 3), dtype=torch.float32),
        "self_attn.q_proj.weight",
        layer_basis,
    )
    o_axes = _hidden_basis_axes_for_layer_tensor(
        torch,
        torch.zeros((3, 4), dtype=torch.float32),
        "self_attn.o_proj.weight",
        layer_basis,
    )

    assert list(q_axes) == [1]
    assert list(o_axes) == [0]
    assert torch.equal(q_axes[1], input_basis)
    assert torch.equal(o_axes[0], output_basis)


def test_outgoing_sum_fold_preserves_merged_linear_path() -> None:
    torch = pytest.importorskip("torch")
    x = torch.tensor([5.0, 7.0])
    up = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    down = torch.tensor([[3.0, 3.0], [4.0, 4.0]])

    original = down.matmul(up.matmul(x))
    folded_up = _resize_tensor_to_shape(torch, up, (1, 2))
    folded_down = _resize_tensor_to_shape(torch, down, (2, 1), axis_reductions={1: "sum"})
    folded = folded_down.matmul(folded_up.matmul(x))

    assert folded.tolist() == pytest.approx(original.tolist())


def test_consumer_hidden_axes_sum_when_neurons_are_merged() -> None:
    assert _axis_reductions_for_layer_tensor("self_attn.q_proj.weight") == {1: "sum"}
    assert _axis_reductions_for_layer_tensor("self_attn.k_proj.weight") == {1: "sum"}
    assert _axis_reductions_for_layer_tensor("mlp.gate_proj.weight") == {1: "sum"}
    assert _axis_reductions_for_layer_tensor("mlp.up_proj.weight") == {1: "sum"}
    assert _axis_reductions_for_layer_tensor("mlp.down_proj.weight") == {1: "sum"}
    assert _axis_reductions_for_non_layer_tensor_key("lm_head.weight") == {1: "sum"}
    assert _axis_reductions_for_non_layer_tensor_key("model.embed_tokens.weight") == {}


def test_token_projection_uses_sum_for_lm_head_and_mean_for_embeddings() -> None:
    torch = pytest.importorskip("torch")
    donor = torch.ones((2, 4), dtype=torch.float32)
    embedding_target = torch.zeros((2, 2), dtype=torch.float32)
    head_target = torch.zeros((2, 2), dtype=torch.float32)

    _project_token_weight_rows(
        torch,
        embedding_target,
        donor,
        target_ids=[0, 1],
        donor_id_groups=[[0], [1]],
        fold_mode="mean",
        hidden_axis_reduction="mean",
    )
    _project_token_weight_rows(
        torch,
        head_target,
        donor,
        target_ids=[0, 1],
        donor_id_groups=[[0], [1]],
        fold_mode="mean",
        hidden_axis_reduction="sum",
    )

    assert embedding_target.tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert head_target.tolist() == [[2.0, 2.0], [2.0, 2.0]]


def test_token_projection_can_weight_donor_piece_rows() -> None:
    torch = pytest.importorskip("torch")
    donor = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
    target = torch.zeros((1, 2), dtype=torch.float32)

    _project_token_weight_rows(
        torch,
        target,
        donor,
        target_ids=[0],
        donor_id_groups=[[0, 1]],
        donor_id_group_weights=[[1.0, 3.0]],
        fold_mode="mean",
    )

    assert target[0].tolist() == pytest.approx([7.5, 0.0])


def test_token_initialization_skips_tied_lm_head_second_pass() -> None:
    torch = pytest.importorskip("torch")

    class _TinyModel:
        def __init__(self, weight):
            self.embed = torch.nn.Embedding.from_pretrained(weight.clone(), freeze=False)
            self.head = self.embed

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.head

    target = _TinyModel(torch.zeros((2, 2), dtype=torch.float32))
    donor = _TinyModel(torch.ones((2, 4), dtype=torch.float32))
    tokenizer = _Tokenizer(["a", "b"])

    report = _initialize_token_weights_from_donor(
        torch,
        target,
        donor,
        tokenizer,
        tokenizer,
        max_anchors=2,
        token_init_mode="shared",
        fold_mode="mean",
        consumer_hidden_reduction="sum",
    )

    assert target.get_input_embeddings().weight.tolist() == [[0.0, 0.0], [1.0, 1.0]]
    assert report["lm_head_tied_to_embeddings"] is True
    assert report["lm_head_updated"] is False
    assert report["lm_head_rows_initialized"] == 0


def test_sqrt_sum_reduction_is_between_mean_and_sum() -> None:
    torch = pytest.importorskip("torch")
    source = torch.ones((4, 2), dtype=torch.float32)

    mean_folded = _resize_tensor_to_shape(torch, source, (2, 2), axis_reductions={0: "mean"})
    sqrt_folded = _resize_tensor_to_shape(torch, source, (2, 2), axis_reductions={0: "sqrt-sum"})
    sum_folded = _resize_tensor_to_shape(torch, source, (2, 2), axis_reductions={0: "sum"})

    assert mean_folded.flatten().tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert sqrt_folded.flatten().tolist() == pytest.approx([2**0.5, 2**0.5, 2**0.5, 2**0.5])
    assert sum_folded.flatten().tolist() == pytest.approx([2.0, 2.0, 2.0, 2.0])
    assert _with_consumer_hidden_reduction({1: "sum"}, "mean") == {1: "mean"}
    assert _with_consumer_hidden_reduction({1: "sum"}, "sqrt-sum") == {1: "sqrt-sum"}
    assert _with_consumer_hidden_reduction({1: "sum"}, "sum") == {1: "sum"}


def test_attention_tensor_resize_folds_q_heads_before_head_dim() -> None:
    torch = pytest.importorskip("torch")
    source = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    donor_config = _Obj(num_attention_heads=4, num_key_value_heads=2)
    target_config = _Obj(num_attention_heads=2, num_key_value_heads=1)

    resized = _resize_attention_tensor_to_shape(
        torch,
        source,
        (2, 1),
        "self_attn.q_proj.weight",
        donor_config,
        target_config,
    )

    assert resized is not None
    assert resized.flatten().tolist() == pytest.approx([1.5, 5.5])


def test_attention_tensor_resize_folds_o_input_heads() -> None:
    torch = pytest.importorskip("torch")
    source = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    donor_config = _Obj(num_attention_heads=4, num_key_value_heads=2)
    target_config = _Obj(num_attention_heads=2, num_key_value_heads=1)

    resized = _resize_attention_tensor_to_shape(
        torch,
        source,
        (1, 2),
        "self_attn.o_proj.weight",
        donor_config,
        target_config,
    )

    assert resized is not None
    assert resized.flatten().tolist() == pytest.approx([6.0, 22.0])


def test_attention_head_activation_profile_extracts_per_head_features() -> None:
    torch = pytest.importorskip("torch")
    activations = torch.tensor(
        [
            [1.0, -1.0, 2.0, -2.0],
            [3.0, -3.0, 4.0, -4.0],
        ],
        dtype=torch.float32,
    )

    profile = _attention_head_activation_profile(torch, activations, heads=2)

    assert profile is not None
    assert profile["scores"].tolist() == pytest.approx([2.0, 3.0])
    assert tuple(profile["features"].shape) == (2, 6)
    assert profile["head_dim_scores"].tolist() == pytest.approx([2.5, 2.5])
    assert tuple(profile["head_dim_features"].shape) == (2, 3)


def test_activation_channel_summary_can_append_subspace_signature() -> None:
    torch = pytest.importorskip("torch")
    values = torch.tensor(
        [
            [1.0, 0.0, -1.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [4.0, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    stats = _activation_channel_summary(torch, values)
    subspace = _activation_channel_summary(torch, values, mode="subspace", subspace_rank=3)
    rank_zero = _activation_channel_summary(torch, values, mode="subspace", subspace_rank=0)

    assert tuple(stats.shape) == (3, 4)
    assert tuple(subspace.shape) == (3, 7)
    assert tuple(rank_zero.shape) == (3, 4)
    assert subspace.device == values.device
    assert torch.allclose(subspace[:, :4], stats)
    assert subspace[:, 4:].abs().sum().item() > 0.0


def test_activation_channel_summary_covariance_groups_coactivation_patterns() -> None:
    torch = pytest.importorskip("torch")
    values = torch.tensor(
        [
            [1.0, 2.0, 1.0, 2.0],
            [-1.0, -2.0, 1.0, 2.0],
            [1.0, 2.0, -1.0, -2.0],
            [-1.0, -2.0, -1.0, -2.0],
            [1.0, 2.0, 1.0, 2.0],
            [-1.0, -2.0, 1.0, 2.0],
        ],
        dtype=torch.float32,
    )

    covariance = _activation_channel_summary(torch, values, mode="covariance", subspace_rank=2)
    stats = _activation_channel_summary(torch, values)
    assignment = _activation_similarity_assignment(torch, covariance, target_dim=2)

    assert tuple(covariance.shape) == (4, 6)
    assert covariance.device == values.device
    assert torch.allclose(covariance[:, :4], stats)
    assert covariance[:, 4:].abs().sum().item() > 0.0
    assert assignment[0].item() == assignment[1].item()
    assert assignment[2].item() == assignment[3].item()
    assert assignment[0].item() != assignment[2].item()


def test_attention_axis_weights_share_q_and_o_head_profiles() -> None:
    torch = pytest.importorskip("torch")
    q_scores = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    q_features = torch.eye(4, dtype=torch.float32)
    dim_scores = torch.tensor([2.0, 4.0], dtype=torch.float32)
    dim_features = torch.eye(2, dtype=torch.float32)
    profile = {"attention_head_profiles": {0: {"q": {"scores": q_scores, "features": q_features}}}}
    profile["attention_head_dim_profiles"] = {
        0: {
            "qk": {"scores": dim_scores, "features": dim_features},
            "vo": {"scores": dim_scores + 1.0, "features": dim_features.flip(0)},
        }
    }

    q_weights = _attention_axis_weights_for_tensor(
        torch,
        torch.zeros((8, 1), dtype=torch.float32),
        "self_attn.q_proj.weight",
        donor_layer_idx=0,
        donor_layer_group=[0],
        activation_profile=profile,
    )
    o_weights = _attention_axis_weights_for_tensor(
        torch,
        torch.zeros((1, 8), dtype=torch.float32),
        "self_attn.o_proj.weight",
        donor_layer_idx=0,
        donor_layer_group=[0],
        activation_profile=profile,
    )

    assert q_weights is not None
    assert o_weights is not None
    assert torch.equal(q_weights[0]["weights"], q_scores)
    assert torch.equal(o_weights[1]["features"], q_features)
    assert q_weights[0]["assignment_key"] == o_weights[1]["assignment_key"]
    assert torch.equal(q_weights[0]["head_dim_weights"], dim_scores)
    assert torch.equal(o_weights[1]["head_dim_weights"], dim_scores + 1.0)
    assert q_weights[0]["head_dim_assignment_key"] != o_weights[1]["head_dim_assignment_key"]


def test_attention_qk_and_vo_head_dim_maps_are_cached_separately() -> None:
    torch = pytest.importorskip("torch")
    head_scores = torch.tensor([1.0, 2.0], dtype=torch.float32)
    head_features = torch.eye(2, dtype=torch.float32)
    dim_scores = torch.tensor([1.0, 1.0], dtype=torch.float32)
    dim_features = torch.eye(2, dtype=torch.float32)
    profile = {
        "attention_head_profiles": {
            0: {
                "q": {"scores": head_scores, "features": head_features},
                "kv": {"scores": head_scores, "features": head_features},
            }
        },
        "attention_head_dim_profiles": {
            0: {
                "qk": {"scores": dim_scores, "features": dim_features},
                "vo": {"scores": dim_scores, "features": dim_features.flip(0)},
            }
        },
    }

    q_weights = _attention_axis_weights_for_tensor(
        torch,
        torch.zeros((4, 1), dtype=torch.float32),
        "self_attn.q_proj.weight",
        donor_layer_idx=0,
        donor_layer_group=[0],
        activation_profile=profile,
    )
    k_weights = _attention_axis_weights_for_tensor(
        torch,
        torch.zeros((4, 1), dtype=torch.float32),
        "self_attn.k_proj.weight",
        donor_layer_idx=0,
        donor_layer_group=[0],
        activation_profile=profile,
    )
    v_weights = _attention_axis_weights_for_tensor(
        torch,
        torch.zeros((4, 1), dtype=torch.float32),
        "self_attn.v_proj.weight",
        donor_layer_idx=0,
        donor_layer_group=[0],
        activation_profile=profile,
    )

    assert q_weights[0]["head_dim_assignment_key"] == k_weights[0]["head_dim_assignment_key"]
    assert q_weights[0]["head_dim_assignment_key"] != v_weights[0]["head_dim_assignment_key"]


def test_attention_tensor_activation_coupled_head_assignment_can_cross_position() -> None:
    torch = pytest.importorskip("torch")
    source = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    donor_config = _Obj(num_attention_heads=4, num_key_value_heads=2)
    target_config = _Obj(num_attention_heads=2, num_key_value_heads=1)
    attention_axis_weights = {
        0: {
            "weights": torch.ones(4, dtype=torch.float32),
            "features": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        }
    }

    resized = _resize_attention_tensor_to_shape(
        torch,
        source,
        (2, 1),
        "self_attn.q_proj.weight",
        donor_config,
        target_config,
        attention_axis_weights=attention_axis_weights,
        activation_fold_assignment="similarity",
    )

    assert resized is not None
    assert resized.flatten().tolist() == pytest.approx([2.5, 4.5])


def test_attention_head_dim_transport_uses_activation_features() -> None:
    torch = pytest.importorskip("torch")
    source = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    donor_config = _Obj(num_attention_heads=1, num_key_value_heads=1)
    target_config = _Obj(num_attention_heads=1, num_key_value_heads=1)
    attention_axis_weights = {
        0: {
            "weights": torch.ones(1, dtype=torch.float32),
            "features": torch.ones((1, 1), dtype=torch.float32),
            "head_dim_weights": torch.ones(4, dtype=torch.float32),
            "head_dim_features": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        }
    }

    resized = _resize_attention_tensor_to_shape(
        torch,
        source,
        (2, 1),
        "self_attn.q_proj.weight",
        donor_config,
        target_config,
        attention_axis_weights=attention_axis_weights,
        attention_head_dim_transport="qk-vo",
        activation_fold_assignment="similarity",
    )

    assert resized is not None
    assert resized.flatten().tolist() == pytest.approx([1.0, 2.0], abs=1e-5)


def test_fit_linear_weight_from_activation_rows_recovers_mapping() -> None:
    torch = pytest.importorskip("torch")
    inputs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    expected_weight = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float32)
    outputs = inputs.matmul(expected_weight.t())

    fitted = _fit_linear_weight_from_activation_rows(torch, inputs, outputs, ridge=0.0)

    assert fitted.flatten().tolist() == pytest.approx(expected_weight.flatten().tolist())


def test_truncate_linear_weight_rank_limits_effective_rank() -> None:
    torch = pytest.importorskip("torch")

    weight = torch.diag(torch.tensor([5.0, 4.0, 3.0, 2.0]))
    truncated, report = _truncate_linear_weight_rank(torch, weight, rank=2)

    assert report["rank_truncated"] is True
    assert report["effective_rank"] == 2
    assert report["rank_energy_ratio"] == pytest.approx((25.0 + 16.0) / (25.0 + 16.0 + 9.0 + 4.0))
    assert int(torch.linalg.matrix_rank(truncated).item()) == 2
    assert torch.diag(truncated).tolist() == pytest.approx([5.0, 4.0, 0.0, 0.0], abs=1e-5)


def test_truncate_linear_weight_delta_rank_keeps_reference_full_rank() -> None:
    torch = pytest.importorskip("torch")

    reference = torch.eye(4)
    delta = torch.diag(torch.tensor([5.0, 4.0, 3.0, 2.0]))
    fitted = reference + delta

    truncated, report = _truncate_linear_weight_delta_rank(torch, fitted, reference, rank=2)

    assert report["rank_mode"] == "delta"
    assert report["delta_rank_truncated"] is True
    assert report["effective_rank"] == 2
    assert int(torch.linalg.matrix_rank(truncated - reference).item()) == 2
    assert torch.diag(truncated).tolist() == pytest.approx([6.0, 5.0, 1.0, 1.0], abs=1e-5)


def test_synthesize_cross_arch_layer_tensor_fuses_qkv_for_gpt2() -> None:
    torch = pytest.importorskip("torch")
    q = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    k = q + 10
    v = q + 20
    layer_state = {
        "self_attn.q_proj.weight": q,
        "self_attn.k_proj.weight": k,
        "self_attn.v_proj.weight": v,
    }

    fused, report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "attn.c_attn.weight")

    expected = torch.cat([q, k, v], dim=0).t()
    torch.testing.assert_close(fused, expected)
    assert report["method"] == "fuse_qkv_transposed_for_gpt2_c_attn"


def test_synthesize_cross_arch_layer_tensor_transposes_mlp_for_gpt2() -> None:
    torch = pytest.importorskip("torch")
    gate = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    up = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    down = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
    layer_state = {
        "mlp.gate_proj.weight": gate,
        "mlp.up_proj.weight": up,
        "mlp.down_proj.weight": down,
    }

    c_fc, c_fc_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "mlp.c_fc.weight")
    c_proj, c_proj_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "mlp.c_proj.weight")
    c_fc_bias, c_fc_bias_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "mlp.c_fc.bias")

    torch.testing.assert_close(c_fc, torch.stack([gate, up], dim=0).mean(dim=0).t())
    torch.testing.assert_close(c_proj, down.t())
    torch.testing.assert_close(c_fc_bias, torch.zeros(2))
    assert c_fc_report["method"] == "average_gated_mlp_inputs_transposed_for_gpt2_c_fc"
    assert c_proj_report["method"] == "transpose_down_proj_for_gpt2_c_proj"
    assert c_fc_bias_report["method"] == "zero_mlp_input_bias_for_gpt2_c_fc"


def test_synthesize_cross_arch_layer_tensor_splits_gpt2_qkv_for_split_attention() -> None:
    torch = pytest.importorskip("torch")
    q = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    k = q + 10
    v = q + 20
    c_attn = torch.cat([q, k, v], dim=1)
    layer_state = {"attn.c_attn.weight": c_attn, "attn.c_attn.bias": torch.arange(6, dtype=torch.float32)}

    q_proj, q_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "self_attn.q_proj.weight")
    k_proj, k_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "self_attn.k_proj.weight")
    v_bias, v_bias_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "self_attn.v_proj.bias")

    torch.testing.assert_close(q_proj, q.t())
    torch.testing.assert_close(k_proj, k.t())
    torch.testing.assert_close(v_bias, torch.tensor([4.0, 5.0]))
    assert q_report["method"] == "split_gpt2_c_attn_for_q_proj"
    assert k_report["method"] == "split_gpt2_c_attn_for_k_proj"
    assert v_bias_report["method"] == "split_gpt2_c_attn_bias_for_v_proj"


def test_synthesize_cross_arch_layer_tensor_splits_lfm_conv_for_split_attention() -> None:
    torch = pytest.importorskip("torch")
    q = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    k = q + 10
    v = q + 20
    out = q + 30
    layer_state = {
        "conv.in_proj.weight": torch.cat([q, k, v], dim=0),
        "conv.out_proj.weight": out,
    }

    q_proj, q_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "self_attn.q_proj.weight")
    k_proj, k_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "self_attn.k_proj.weight")
    v_proj, v_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "self_attn.v_proj.weight")
    o_proj, o_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "self_attn.o_proj.weight")

    torch.testing.assert_close(q_proj, q)
    torch.testing.assert_close(k_proj, k)
    torch.testing.assert_close(v_proj, v)
    torch.testing.assert_close(o_proj, out)
    assert q_report["method"] == "split_lfm_conv_in_proj_for_q_proj"
    assert k_report["method"] == "split_lfm_conv_in_proj_for_k_proj"
    assert v_report["method"] == "split_lfm_conv_in_proj_for_v_proj"
    assert o_report["method"] == "lfm_conv_out_proj_for_attention_output"


def test_synthesize_cross_arch_layer_tensor_fuses_lfm_conv_for_gpt2() -> None:
    torch = pytest.importorskip("torch")
    conv_in = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    conv_out = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    layer_state = {"conv.in_proj.weight": conv_in, "conv.out_proj.weight": conv_out}

    c_attn, c_attn_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "attn.c_attn.weight")
    c_proj, c_proj_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "attn.c_proj.weight")

    torch.testing.assert_close(c_attn, conv_in.t())
    torch.testing.assert_close(c_proj, conv_out.t())
    assert c_attn_report["method"] == "transpose_lfm_conv_in_proj_for_gpt2_c_attn"
    assert c_proj_report["method"] == "transpose_lfm_conv_out_proj_for_gpt2_c_proj"


def test_synthesize_cross_arch_layer_tensor_splits_gpt2_mlp_for_gated_target() -> None:
    torch = pytest.importorskip("torch")
    c_fc = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    c_proj = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    layer_state = {"mlp.c_fc.weight": c_fc, "mlp.c_proj.weight": c_proj}

    gate, gate_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "mlp.gate_proj.weight")
    up, up_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "mlp.up_proj.weight")
    down, down_report = _synthesize_cross_arch_layer_tensor(torch, layer_state, "mlp.down_proj.weight")

    torch.testing.assert_close(gate, c_fc.t())
    torch.testing.assert_close(up, c_fc.t())
    torch.testing.assert_close(down, c_proj.t())
    assert gate_report["method"] == "transpose_gpt2_c_fc_for_gate_proj"
    assert up_report["method"] == "transpose_gpt2_c_fc_for_up_proj"
    assert down_report["method"] == "transpose_gpt2_mlp_c_proj_for_down_proj"


def test_sanitize_generation_config_for_save_repairs_negative_pad_token() -> None:
    class TinyModel:
        def __init__(self) -> None:
            self.config = type("Config", (), {"pad_token_id": -1, "eos_token_id": 2, "bos_token_id": 1})()
            self.generation_config = type("GenerationConfig", (), {"pad_token_id": -1, "eos_token_id": 2})()
            self._embedding = type("Embedding", (), {"weight": type("Weight", (), {"shape": (8, 4)})()})()

        def get_input_embeddings(self):
            return self._embedding

    tokenizer = type("Tokenizer", (), {"pad_token_id": None, "eos_token_id": 2, "bos_token_id": 1})()
    model = TinyModel()

    report = _sanitize_generation_config_for_save(model, tokenizer)

    assert report["changed"] is True
    assert model.config.pad_token_id == 2
    assert model.generation_config.pad_token_id == 2
    assert report["fallback_token_id"] == 2


def test_fit_projected_linear_from_donor_activations_reports_validation_metrics() -> None:
    torch = pytest.importorskip("torch")
    inputs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [3.0, 1.0],
            [1.0, 3.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
        ],
        dtype=torch.float32,
    )
    expected_weight = torch.eye(2, dtype=torch.float32)
    activation_profile = {
        "linear_fit_rows": {
            0: {
                "self_attn.q_proj.weight": {
                    "input": inputs,
                    "output": inputs.matmul(expected_weight.t()),
                }
            }
        }
    }
    wrong_same_rms_reference = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    result = _fit_projected_linear_from_donor_activations(
        torch,
        activation_profile,
        "self_attn.q_proj.weight",
        (2, 2),
        _Obj(num_attention_heads=1, num_key_value_heads=1),
        _Obj(num_attention_heads=1, num_key_value_heads=1),
        donor_layer_idx=0,
        donor_layer_group=[0],
        ridge=0.0,
        fold_mode="mean",
        reference_weight=wrong_same_rms_reference,
        validation_fraction=0.2,
        min_validation_rows=2,
        return_report=True,
    )

    assert result["weight"].flatten().tolist() == pytest.approx(expected_weight.flatten().tolist(), abs=1e-5)
    report = result["report"]
    assert report["accepted"] is True
    assert report["validation_rows"] == 2
    assert report["train_rows"] == 8
    assert report["validation_fit_mse"] < report["validation_reference_mse"]
    assert report["validation_relative_improvement"] > 0.0


def test_fit_projected_linear_from_donor_activations_lrc_materializes_delta() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(51)
    inputs = torch.randn(48, 4, generator=generator)
    reference = torch.eye(3, 4, dtype=torch.float32) * 0.05
    low_rank_delta = torch.tensor(
        [[0.30, -0.10, 0.20, 0.00], [0.15, -0.05, 0.10, 0.00], [-0.30, 0.10, -0.20, 0.00]],
        dtype=torch.float32,
    )
    target_weight = reference + low_rank_delta
    activation_profile = {
        "linear_fit_rows": {
            0: {
                "mlp.down_proj.weight": {
                    "input": inputs,
                    "output": inputs.matmul(target_weight.t()),
                }
            }
        }
    }

    result = _fit_projected_linear_from_donor_activations(
        torch,
        activation_profile,
        "mlp.down_proj.weight",
        tuple(reference.shape),
        _Obj(num_attention_heads=1, num_key_value_heads=1),
        _Obj(num_attention_heads=1, num_key_value_heads=1),
        donor_layer_idx=0,
        donor_layer_group=[0],
        ridge=0.0,
        rank=1,
        method="lrc",
        alpha=1.0,
        factor_steps=20,
        factor_lr=0.05,
        fold_mode="mean",
        reference_weight=reference,
        validation_fraction=0.25,
        min_validation_rows=8,
        min_relative_improvement=0.50,
        return_report=True,
    )

    report = result["report"]
    assert report["method"] == "lrc"
    assert report["accepted"] is True
    assert report["effective_lrc_rank"] == 1
    assert report["fit_gate"]["mode"] == "validation"
    assert int(torch.linalg.matrix_rank((result["weight"] - reference).to(torch.float32))) <= 1
    before = inputs.matmul(reference.t())
    after = inputs.matmul(result["weight"].to(torch.float32).t())
    target = inputs.matmul(target_weight.t())
    assert torch.mean((after - target) ** 2) < torch.mean((before - target) ** 2) * 0.1


def test_fit_projected_linear_from_donor_activations_rejects_no_validation_gain() -> None:
    torch = pytest.importorskip("torch")
    inputs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [3.0, 1.0],
            [1.0, 3.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
        ],
        dtype=torch.float32,
    )
    expected_weight = torch.eye(2, dtype=torch.float32)
    activation_profile = {
        "linear_fit_rows": {
            0: {
                "self_attn.q_proj.weight": {
                    "input": inputs,
                    "output": inputs.matmul(expected_weight.t()),
                }
            }
        }
    }

    result = _fit_projected_linear_from_donor_activations(
        torch,
        activation_profile,
        "self_attn.q_proj.weight",
        (2, 2),
        _Obj(num_attention_heads=1, num_key_value_heads=1),
        _Obj(num_attention_heads=1, num_key_value_heads=1),
        donor_layer_idx=0,
        donor_layer_group=[0],
        ridge=0.0,
        fold_mode="mean",
        reference_weight=expected_weight,
        validation_fraction=0.2,
        min_validation_rows=2,
        return_report=True,
    )

    assert result["weight"] is None
    assert result["report"]["accepted"] is False
    assert result["report"]["reason"] == "validation_mse_not_improved"


def test_fit_projected_linear_from_donor_activations_rejects_insufficient_validation_rows() -> None:
    torch = pytest.importorskip("torch")
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    activation_profile = {
        "linear_fit_rows": {
            0: {
                "self_attn.q_proj.weight": {
                    "input": inputs,
                    "output": inputs,
                }
            }
        }
    }

    result = _fit_projected_linear_from_donor_activations(
        torch,
        activation_profile,
        "self_attn.q_proj.weight",
        (2, 2),
        _Obj(num_attention_heads=1, num_key_value_heads=1),
        _Obj(num_attention_heads=1, num_key_value_heads=1),
        donor_layer_idx=0,
        donor_layer_group=[0],
        ridge=0.0,
        fold_mode="mean",
        reference_weight=torch.eye(2, dtype=torch.float32),
        validation_fraction=0.2,
        min_validation_rows=2,
        return_report=True,
    )

    assert result["weight"] is None
    assert result["report"]["accepted"] is False
    assert result["report"]["reason"] == "insufficient_validation_rows"
    assert result["report"]["validation_gate_skipped"] is True


def test_activation_fit_suffix_selector_limits_module_families() -> None:
    assert _activation_fit_suffix_enabled("self_attn.o_proj.weight", "attention-output") is True
    assert _activation_fit_suffix_enabled("operator.out_proj.weight", "attention-output") is True
    assert _activation_fit_suffix_enabled("self_attn.q_proj.weight", "attention-output") is False
    assert _activation_fit_suffix_enabled("mlp.down_proj.weight", "mlp-output") is True
    assert _activation_fit_suffix_enabled("mlp.gate_proj.weight", "mlp-output") is False
    assert _activation_fit_suffix_enabled("self_attn.k_proj.weight", "input-projections") is True
    assert _activation_fit_suffix_enabled("operator.q_proj.weight", "input-projections") is True
    assert _activation_fit_suffix_enabled("mlp.up_proj.weight", "input-projections") is True
    assert _activation_fit_suffix_enabled("mlp.down_proj.weight", "input-projections") is False


def test_non_layer_initializer_folds_final_norm_and_skips_special_tensors() -> None:
    torch = pytest.importorskip("torch")
    target_state = {
        "model.norm.weight": torch.zeros(2),
        "model.embed_tokens.weight": torch.ones((3, 2)),
        "model.layers.0.input_layernorm.weight": torch.ones(2),
    }
    donor_state = {
        "transformer.ln_f.weight": torch.tensor([1.0, 3.0, 5.0, 7.0]),
        "model.embed_tokens.weight": torch.zeros((4, 4)),
        "model.layers.0.input_layernorm.weight": torch.zeros(4),
    }

    report = _initialize_non_layer_tensors_from_donor(
        torch,
        target_state,
        donor_state,
        fold_mode="mean",
    )

    assert target_state["model.norm.weight"].tolist() == pytest.approx([2.0, 6.0])
    assert target_state["model.embed_tokens.weight"].tolist() == [[1.0, 1.0]] * 3
    assert len(report["initialized"]) == 1
    assert report["initialized"][0]["target_key"] == "model.norm.weight"
    assert report["initialized"][0]["donor_key"] == "transformer.ln_f.weight"


def test_scale_projection_to_target_matches_target_rms() -> None:
    torch = pytest.importorskip("torch")
    projected = torch.ones((4, 4), dtype=torch.float32) * 10.0
    target = torch.ones((4, 4), dtype=torch.float32) * 0.5

    scaled = _scale_projection_to_target(torch, projected, target, "target-rms")

    assert float(scaled.pow(2).mean().sqrt()) == pytest.approx(float(target.pow(2).mean().sqrt()))
    assert torch.all(scaled > 0)


def test_scale_projection_to_target_can_preserve_source_rms() -> None:
    torch = pytest.importorskip("torch")
    projected = torch.ones((2, 2), dtype=torch.float32) * 2.0
    target = torch.ones((2, 2), dtype=torch.float32) * 0.5
    source = torch.ones((4, 4), dtype=torch.float32) * 8.0

    scaled = _scale_projection_to_target(torch, projected, target, "source-rms", source_tensor=source)

    assert float(scaled.pow(2).mean().sqrt()) == pytest.approx(float(source.pow(2).mean().sqrt()))
    assert float(scaled.pow(2).mean().sqrt()) != pytest.approx(float(target.pow(2).mean().sqrt()))


def test_scale_projection_to_target_can_use_geomean_rms() -> None:
    torch = pytest.importorskip("torch")
    projected = torch.ones((2, 2), dtype=torch.float32) * 2.0
    target = torch.ones((2, 2), dtype=torch.float32) * 0.5
    source = torch.ones((4, 4), dtype=torch.float32) * 8.0

    scaled = _scale_projection_to_target(torch, projected, target, "geomean-rms", source_tensor=source)

    expected_rms = (float(target.pow(2).mean().sqrt()) * float(source.pow(2).mean().sqrt())) ** 0.5
    assert float(scaled.pow(2).mean().sqrt()) == pytest.approx(expected_rms)
    assert float(target.pow(2).mean().sqrt()) < expected_rms < float(source.pow(2).mean().sqrt())


def test_blend_projection_with_target_interpolates_values() -> None:
    torch = pytest.importorskip("torch")
    projected = torch.ones((2, 2), dtype=torch.float32) * 10.0
    target = torch.ones((2, 2), dtype=torch.float32) * 2.0

    blended = _blend_projection_with_target(torch, projected, target, 0.25)

    assert blended.tolist() == [[4.0, 4.0], [4.0, 4.0]]


def test_tensor_projection_diagnostic_reports_fold_scale() -> None:
    torch = pytest.importorskip("torch")
    source = torch.ones((4, 2), dtype=torch.float32) * 2.0
    projected = torch.ones((2, 2), dtype=torch.float32) * 6.0
    target = torch.ones((2, 2), dtype=torch.float32) * 4.0

    report = _tensor_projection_diagnostic(torch, source, projected, target)

    assert report["source"]["rms"] == pytest.approx(2.0)
    assert report["projected"]["rms"] == pytest.approx(6.0)
    assert report["target_before"]["rms"] == pytest.approx(4.0)
    assert report["projected_to_source_rms_ratio"] == pytest.approx(3.0)
    assert report["projected_to_target_rms_ratio"] == pytest.approx(1.5)
    assert report["target_update_relative_rms"] == pytest.approx(0.5)
    assert report["target_projection_cosine"] == pytest.approx(1.0)
    assert report["nonfinite_detected"] is False


def test_summarize_projection_diagnostics_keeps_extremes() -> None:
    reports = [
        {
            "projected_to_source_rms_ratio": 0.5,
            "projected_to_target_rms_ratio": 1.0,
            "target_update_relative_rms": 0.2,
            "target_projection_cosine": 0.9,
            "nonfinite_detected": False,
        },
        {
            "projected_to_source_rms_ratio": 2.0,
            "projected_to_target_rms_ratio": 4.0,
            "target_update_relative_rms": 1.2,
            "target_projection_cosine": -0.1,
            "nonfinite_detected": True,
        },
    ]

    summary = _summarize_projection_diagnostics(reports)

    assert summary["count"] == 2
    assert summary["nonfinite_detected_count"] == 1
    assert summary["projected_to_source_rms_ratio"]["mean"] == pytest.approx(1.25)
    assert summary["projected_to_target_rms_ratio"]["max"] == pytest.approx(4.0)
    assert summary["target_update_relative_rms"]["min"] == pytest.approx(0.2)
    assert summary["target_projection_cosine"]["min"] == pytest.approx(-0.1)


def test_activation_weighted_fold_uses_importance_bins() -> None:
    torch = pytest.importorskip("torch")
    weights = torch.tensor([10.0, 1.0, 9.0, 1.0])
    assignment = _balanced_activation_assignment(torch, weights, 2)
    assert assignment.tolist() == [0, 0, 1, 1]

    source = torch.tensor([[0.0], [10.0], [100.0], [200.0]])
    folded = _resize_tensor_to_shape(torch, source, (2, 1), axis_weights={0: weights})

    assert folded.flatten().tolist() == pytest.approx([10.0 / 11.0, 1100.0 / 10.0])


def test_cached_mlp_fold_reuses_assignment_and_sums_outgoing_paths() -> None:
    torch = pytest.importorskip("torch")
    cache = {}
    key = ("mlp", (0, 1))
    incoming = torch.tensor(
        [
            [1.0, 0.0],
            [10.0, 0.0],
            [100.0, 0.0],
            [1000.0, 0.0],
        ]
    )
    incoming_weights = torch.tensor([10.0, 1.0, 9.0, 1.0])
    folded_incoming = _resize_tensor_axis_mean_fold(
        torch,
        incoming,
        axis=0,
        target_dim=2,
        weights=incoming_weights,
        reduction="mean",
        assignment_cache=cache,
        assignment_cache_key=key,
    )

    outgoing = torch.tensor([[3.0, 4.0, 5.0, 6.0]])
    different_weights = torch.tensor([1.0, 9.0, 1.0, 10.0])
    folded_outgoing = _resize_tensor_axis_mean_fold(
        torch,
        outgoing,
        axis=1,
        target_dim=2,
        weights=different_weights,
        reduction="sum",
        assignment_cache=cache,
        assignment_cache_key=key,
    )

    assert folded_incoming[:, 0].tolist() == pytest.approx([20.0 / 11.0, 1900.0 / 10.0])
    assert folded_outgoing.flatten().tolist() == pytest.approx([7.0, 11.0])
    assert len(cache) == 1


def test_mlp_neuron_activation_values_use_gated_product() -> None:
    torch = pytest.importorskip("torch")

    class _TinyGatedMlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(2, 3, bias=False)
            self.up_proj = torch.nn.Linear(2, 3, bias=False)
            self.down_proj = torch.nn.Linear(3, 2, bias=False)
            self.act_fn = torch.nn.Sigmoid()

    mlp = _TinyGatedMlp()
    mlp.gate_proj.weight.data.copy_(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    )
    mlp.up_proj.weight.data.copy_(
        torch.tensor([[2.0, 0.0], [0.0, 3.0], [4.0, 5.0]], dtype=torch.float32)
    )
    spec = {
        "kind": "gated",
        "mlp": mlp,
        "inputs": [mlp.gate_proj, mlp.up_proj],
        "input_names": ["gate_proj", "up_proj"],
        "output": mlp.down_proj,
        "output_name": "down_proj",
    }
    inputs = (torch.tensor([[1.0, 2.0]], dtype=torch.float32),)

    values = _mlp_neuron_activation_values(torch, spec, inputs)

    expected = torch.sigmoid(mlp.gate_proj(inputs[0])) * mlp.up_proj(inputs[0])
    assert torch.allclose(values, expected)


def test_transport_fold_uses_balanced_sinkhorn_feature_map() -> None:
    torch = pytest.importorskip("torch")
    cache = {}
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ]
    )
    source = torch.tensor([[1.0], [3.0], [100.0], [300.0]])

    folded = _resize_tensor_axis_mean_fold(
        torch,
        source,
        axis=0,
        target_dim=2,
        fold_mode="transport",
        features=features,
        assignment_cache=cache,
        assignment_cache_key=("hidden", "transport-test"),
        transport_backend="sinkhorn",
    )
    outgoing = torch.ones((1, 4), dtype=torch.float32)
    folded_outgoing = _resize_tensor_axis_mean_fold(
        torch,
        outgoing,
        axis=1,
        target_dim=2,
        fold_mode="transport",
        features=features,
        reduction="sum",
        assignment_cache=cache,
        assignment_cache_key=("hidden", "transport-test"),
        transport_backend="sinkhorn",
    )

    assert folded.shape == (2, 1)
    assert sorted(round(float(item), 1) for item in folded.flatten()) == pytest.approx([2.0, 200.0], abs=2.0)
    assert float(folded_outgoing.sum()) == pytest.approx(4.0)
    assert len(cache) == 1
    transport = next(iter(cache.values()))
    assert transport["transport_backend"] == "sinkhorn_dense"
    assert transport["transport_indices"].shape == (4, 2)
    assert transport["transport_matrix"].shape == (4, 2)
    assert transport["transport_matrix"].sum(dim=1).tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert transport["transport_matrix"].sum(dim=0).tolist() == pytest.approx([2.0, 2.0], abs=0.05)


def test_transport_fold_can_use_importance_survivor_support() -> None:
    torch = pytest.importorskip("torch")
    cache = {}
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.02, 0.98],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([0.1, 10.0, 0.2, 8.0], dtype=torch.float32)
    source = torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float32)

    folded = _resize_tensor_axis_mean_fold(
        torch,
        source,
        axis=0,
        target_dim=2,
        fold_mode="transport",
        weights=weights,
        features=features,
        transport_support="importance",
        assignment_cache=cache,
        assignment_cache_key=("hidden", "importance-support-test"),
    )

    assert folded.shape == (2, 1)
    transport = next(iter(cache.values()))
    assert transport["transport_support"] == "importance"
    assert transport["target_support_indices"].tolist() == [1, 3]


def test_quantile_support_preserves_activation_weight_strata() -> None:
    torch = pytest.importorskip("torch")
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([0.1, 0.2, 0.3, 50.0, 60.0, 70.0], dtype=torch.float32)

    support = _activation_quantile_support_indices(torch, features, weights, target_dim=3)

    assert support.tolist() == [0, 2, 5]


def test_medoid_support_uses_real_activation_cluster_representatives() -> None:
    torch = pytest.importorskip("torch")
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.92, 0.08],
            [0.0, 1.0],
            [0.08, 0.92],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([1.0, 3.0, 1.0, 4.0], dtype=torch.float32)
    assignment = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    support = _activation_medoid_support_indices(torch, features, weights, assignment, target_dim=2)

    assert support.tolist() == [1, 3]


def test_quantile_medoid_support_uses_real_representatives_per_weight_stratum() -> None:
    torch = pytest.importorskip("torch")
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.96, 0.04],
            [0.0, 1.0],
            [0.08, 0.92],
            [0.7, 0.7],
            [0.65, 0.75],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([0.1, 0.2, 2.0, 50.0, 60.0, 70.0], dtype=torch.float32)

    support = _activation_quantile_medoid_support_indices(torch, features, weights, target_dim=3)

    assert support.tolist() == [1, 3, 5]


def test_transport_fold_can_use_quantile_support() -> None:
    torch = pytest.importorskip("torch")
    cache = {}
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([0.1, 0.2, 0.3, 50.0, 60.0, 70.0], dtype=torch.float32)
    source = torch.arange(6, dtype=torch.float32).unsqueeze(1)

    folded = _resize_tensor_axis_mean_fold(
        torch,
        source,
        axis=0,
        target_dim=3,
        fold_mode="transport",
        weights=weights,
        features=features,
        transport_support="quantile",
        assignment_cache=cache,
        assignment_cache_key=("hidden", "quantile-support-test"),
    )

    assert folded.shape == (3, 1)
    transport = next(iter(cache.values()))
    assert transport["transport_support"] == "quantile"
    assert transport["target_support_indices"].tolist() == [0, 2, 5]
    assert transport["target_masses"].sum().item() == pytest.approx(1.0)
    assert transport["target_masses"].tolist() != pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_transport_fold_can_use_medoid_support() -> None:
    torch = pytest.importorskip("torch")
    cache = {}
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.92, 0.08],
            [0.0, 1.0],
            [0.08, 0.92],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([1.0, 3.0, 1.0, 4.0], dtype=torch.float32)
    source = torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float32)

    folded = _resize_tensor_axis_mean_fold(
        torch,
        source,
        axis=0,
        target_dim=2,
        fold_mode="transport",
        weights=weights,
        features=features,
        transport_backend="sinkhorn",
        transport_support="medoid",
        assignment_cache=cache,
        assignment_cache_key=("hidden", "medoid-support-test"),
    )

    assert folded.shape == (2, 1)
    transport = next(iter(cache.values()))
    assert transport["transport_support"] == "medoid"
    assert transport["target_support_indices"].tolist() == [1, 3]
    assert transport["transport_matrix"].shape == (4, 2)
    assert transport["target_masses"].sum().item() == pytest.approx(1.0)


def test_transport_fold_can_use_quantile_medoid_support() -> None:
    torch = pytest.importorskip("torch")
    cache = {}
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.96, 0.04],
            [0.0, 1.0],
            [0.08, 0.92],
            [0.7, 0.7],
            [0.65, 0.75],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([0.1, 0.2, 2.0, 50.0, 60.0, 70.0], dtype=torch.float32)
    source = torch.arange(6, dtype=torch.float32).unsqueeze(1)

    folded = _resize_tensor_axis_mean_fold(
        torch,
        source,
        axis=0,
        target_dim=3,
        fold_mode="transport",
        weights=weights,
        features=features,
        transport_backend="sinkhorn",
        transport_support="quantile-medoid",
        assignment_cache=cache,
        assignment_cache_key=("hidden", "quantile-medoid-support-test"),
    )

    assert folded.shape == (3, 1)
    transport = next(iter(cache.values()))
    assert transport["transport_support"] == "quantile-medoid"
    assert transport["target_support_indices"].tolist() == [1, 3, 5]
    assert transport["transport_matrix"].shape == (6, 3)
    assert transport["target_masses"].sum().item() == pytest.approx(1.0)


def test_transport_cache_key_keeps_support_modes_separate() -> None:
    torch = pytest.importorskip("torch")
    cache = {}
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.96, 0.04],
            [0.0, 1.0],
            [0.08, 0.92],
            [0.7, 0.7],
            [0.65, 0.75],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([0.1, 0.2, 2.0, 50.0, 60.0, 70.0], dtype=torch.float32)
    source = torch.arange(6, dtype=torch.float32).unsqueeze(1)
    key = ("mlp", (0, 1), "shared-axis")

    _resize_tensor_axis_mean_fold(
        torch,
        source,
        axis=0,
        target_dim=3,
        fold_mode="transport",
        weights=weights,
        features=features,
        transport_support="quantile",
        assignment_cache=cache,
        assignment_cache_key=key,
    )
    _resize_tensor_axis_mean_fold(
        torch,
        source,
        axis=0,
        target_dim=3,
        fold_mode="transport",
        weights=weights,
        features=features,
        transport_support="quantile-medoid",
        assignment_cache=cache,
        assignment_cache_key=key,
    )

    assert len(cache) == 2
    supports = {item["transport_support"]: item["target_support_indices"].tolist() for item in cache.values()}
    assert supports["quantile"] == [0, 2, 5]
    assert supports["quantile-medoid"] == [1, 3, 5]


def test_activation_layer_geometry_scores_residual_novelty() -> None:
    torch = pytest.importorskip("torch")
    inputs = {
        0: torch.tensor([[1.0, 0.0], [0.9, 0.1], [1.1, -0.1]], dtype=torch.float32),
        1: torch.tensor([[1.0, 0.0], [0.9, 0.1], [1.1, -0.1]], dtype=torch.float32),
        2: torch.tensor([[0.0, 1.0], [0.1, 0.9], [-0.1, 1.1]], dtype=torch.float32),
    }
    outputs = {
        0: inputs[0] + torch.tensor([[0.01, 0.0], [0.01, 0.0], [0.01, 0.0]]),
        1: inputs[1] + torch.tensor([[0.8, 0.0], [0.8, 0.0], [0.8, 0.0]]),
        2: inputs[2] + torch.tensor([[0.0, -0.8], [0.0, -0.8], [0.0, -0.8]]),
    }

    report = _activation_layer_geometry_scores(torch, inputs, outputs, layer_count=3)

    assert report["geometry_scores"][1] > report["geometry_scores"][0]
    assert report["geometry_scores"][2] > report["delta_scores"][2]
    assert report["layer_geometry"][1]["linear_cka"] is not None
    assert report["layer_geometry"][2]["delta_novelty"] >= 0.0


def test_activation_geometry_group_strategy_reports_geometry_scores() -> None:
    torch = pytest.importorskip("torch")
    profile = {
        "hidden_basis_input_rows": {
            0: torch.zeros((4, 3)),
            1: torch.zeros((4, 3)),
            2: torch.zeros((4, 3)),
            3: torch.zeros((4, 3)),
        },
        "hidden_basis_output_rows": {
            0: torch.ones((4, 3)) * 0.01,
            1: torch.ones((4, 3)) * 0.02,
            2: torch.ones((4, 3)) * 1.5,
            3: torch.ones((4, 3)) * 0.03,
        },
    }

    groups, report = _donor_layer_groups_for_target_count(
        torch,
        target_layer_count=2,
        donor_layer_count=4,
        strategy="activation-geometry",
        activation_profile=profile,
    )
    geometry = _donor_layer_geometry_scores(torch, profile, donor_layer_count=4)

    assert len(groups) == 2
    assert report["strategy"] == "activation-geometry"
    assert report["representative_scores"] == pytest.approx(report["geometry_scores"])
    assert report["geometry_scores"] == pytest.approx(geometry["geometry_scores"])
    assert len(report["layer_geometry"]) == 4


def test_domain_balanced_mean_weights_domains_evenly() -> None:
    torch = pytest.importorskip("torch")

    from neural_forge.hf_xfamily import _activation_calibration_domain, _domain_balanced_mean

    tensors = [torch.tensor([0.0]), torch.tensor([2.0]), torch.tensor([100.0])]
    balanced = _domain_balanced_mean(torch, tensors, ["code", "code", "math"])
    unbalanced = _domain_balanced_mean(torch, tensors, None)
    assert balanced.item() == pytest.approx(50.5)
    assert unbalanced.item() == pytest.approx(34.0)
    assert _activation_calibration_domain("Write a Python function with return a + b") == "code"
    assert _activation_calibration_domain("Question: What is 6 + 4?") == "math"
    assert _activation_calibration_domain("Tokenizer vocab rows can mismatch embeddings") == "tokenizer"


def test_recovery_text_order_round_robins_domains() -> None:
    from neural_forge.hf_recovery import _order_recovery_texts, _recovery_domain_counts

    texts = [
        "def add(a, b): return a + b",
        "write a Python function",
        "Question: What is 12 + 13?",
        "Calculate 7 * 8",
        "Is this safe in rain?",
    ]

    assert _order_recovery_texts(texts, "original") == texts
    ordered = _order_recovery_texts(texts, "domain-balanced")
    assert ordered == [texts[0], texts[2], texts[4], texts[1], texts[3]]
    assert _recovery_domain_counts(ordered) == {"code": 2, "math": 2, "safety": 1}
    with pytest.raises(ValueError, match="calibration_order"):
        _order_recovery_texts(texts, "shuffle")


def test_cross_tokenizer_recovery_accepts_calibration_order_parameter() -> None:
    import inspect

    from neural_forge.hf_recovery import recover_cross_tokenizer_causal_lm

    parameters = inspect.signature(recover_cross_tokenizer_causal_lm).parameters
    assert parameters["calibration_order"].default == "original"


def test_activation_axis_weights_aggregate_folded_donor_group() -> None:
    torch = pytest.importorskip("torch")
    donor_tensor = torch.ones((6, 4), dtype=torch.float32)
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
    assert weights[0]["weights"].tolist() == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert weights[0]["features"].flatten().tolist() == pytest.approx([2.0] * 12)


def test_activation_axis_weights_route_mlp_axes_by_suffix_when_dims_overlap() -> None:
    torch = pytest.importorskip("torch")
    donor_tensor = torch.ones((6, 6), dtype=torch.float32)
    profile = {
        "hidden_scores": torch.arange(1, 7, dtype=torch.float32),
        "hidden_features": torch.ones((6, 2)),
        "mlp_scores": {0: torch.arange(11, 17, dtype=torch.float32)},
        "mlp_features": {0: torch.ones((6, 2)) * 5.0},
    }

    gate_weights = _activation_axis_weights_for_tensor(
        torch,
        donor_tensor,
        "mlp.gate_proj.weight",
        donor_layer_idx=0,
        donor_layer_group=[0],
        activation_profile=profile,
    )
    down_weights = _activation_axis_weights_for_tensor(
        torch,
        donor_tensor,
        "mlp.down_proj.weight",
        donor_layer_idx=0,
        donor_layer_group=[0],
        activation_profile=profile,
    )

    assert gate_weights is not None
    assert down_weights is not None
    assert gate_weights[0]["assignment_key"] == ("mlp", (0,))
    assert gate_weights[1]["assignment_key"] == ("hidden", "residual")
    assert down_weights[0]["assignment_key"] == ("hidden", "residual")
    assert down_weights[1]["assignment_key"] == ("mlp", (0,))


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
        loss_mask=None,
        kd_weight=0.0,
        lm_weight=1.0,
        unlikelihood_weight=0.0,
        unlikelihood_window=2,
        entropy_weight=0.0,
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
            loss_mask=None,
            kd_weight=1.0,
            lm_weight=0.0,
            unlikelihood_weight=0.0,
            unlikelihood_window=2,
            entropy_weight=0.0,
            temperature=1.0,
        )


def test_recovery_loss_handles_padded_teacher_vocab() -> None:
    torch = pytest.importorskip("torch")
    student = _Obj(logits=torch.randn(1, 4, 5))
    teacher = _Obj(logits=torch.randn(1, 4, 7))
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    loss = _recovery_loss(
        torch,
        student,
        teacher=teacher,
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=None,
        kd_weight=1.0,
        lm_weight=0.0,
        unlikelihood_weight=0.0,
        unlikelihood_window=2,
        entropy_weight=0.0,
        temperature=1.0,
    )

    assert torch.isfinite(loss)


def test_recovery_loss_response_mask_ignores_prompt_tokens() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.full((1, 4, 5), -8.0)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[False, False, False, True]])
    logits[0, 0, 1] = -20.0
    logits[0, 1, 2] = -20.0
    logits[0, 2, 3] = 20.0
    student = _Obj(logits=logits)

    masked = _recovery_loss(
        torch,
        student,
        teacher=None,
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        kd_weight=0.0,
        lm_weight=1.0,
        unlikelihood_weight=0.0,
        unlikelihood_window=2,
        entropy_weight=0.0,
        temperature=1.0,
    )
    full = _recovery_loss(
        torch,
        student,
        teacher=None,
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=None,
        kd_weight=0.0,
        lm_weight=1.0,
        unlikelihood_weight=0.0,
        unlikelihood_window=2,
        entropy_weight=0.0,
        temperature=1.0,
    )

    assert masked.item() < 1e-3
    assert full.item() > masked.item() + 5.0


def test_first_response_shift_mask_selects_first_response_label() -> None:
    torch = pytest.importorskip("torch")
    loss_mask = torch.tensor(
        [
            [False, False, False, True, True],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ]
    )
    attention_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, True, False],
            [True, True, True, True, False],
        ]
    )

    first = _first_response_shift_mask(torch, loss_mask, attention_mask)

    assert first.tolist() == [
        [False, False, True, False],
        [True, False, False, False],
        [False, False, False, False],
    ]


def test_recovery_loss_first_response_lm_focuses_boundary_token() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.full((1, 5, 6), -8.0)
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[False, False, False, True, True]])
    logits[0, 2, 3] = 20.0
    logits[0, 3, 4] = -20.0
    student = _Obj(logits=logits)

    first_only = _recovery_loss(
        torch,
        student,
        teacher=None,
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        kd_weight=0.0,
        lm_weight=0.0,
        unlikelihood_weight=0.0,
        unlikelihood_window=2,
        entropy_weight=0.0,
        temperature=1.0,
        first_token_lm_weight=1.0,
    )
    response_lm = _recovery_loss(
        torch,
        student,
        teacher=None,
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        kd_weight=0.0,
        lm_weight=1.0,
        unlikelihood_weight=0.0,
        unlikelihood_window=2,
        entropy_weight=0.0,
        temperature=1.0,
    )

    assert first_only.item() < 1e-3
    assert response_lm.item() > first_only.item() + 5.0


def test_cross_tokenizer_recovery_loss_first_response_lm_focuses_boundary_token() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.full((1, 5, 6), -8.0)
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[False, False, False, True, True]])
    logits[0, 2, 3] = 20.0
    logits[0, 3, 4] = -20.0
    student = _Obj(logits=logits)

    first_only = _cross_tokenizer_recovery_loss(
        torch,
        student=student,
        teacher=None,
        candidate_input_ids=input_ids,
        candidate_attention_mask=attention_mask,
        candidate_loss_mask=loss_mask,
        source_attention_mask=None,
        candidate_anchor_idx=torch.tensor([0]),
        source_anchor_idx=torch.tensor([0]),
        anchor_kd_weight=0.0,
        lm_weight=0.0,
        first_token_lm_weight=1.0,
        response_end_lm_weight=0.0,
        hidden_kd_weight=0.0,
        span_kd_weight=0.0,
        span_hidden_kd_weight=0.0,
        projected_vocab_kd_weight=0.0,
        focused_projected_vocab_kd_weight=0.0,
        projected_vocab_kd_mode="teacher-to-student",
        reference_kd_weight=0.0,
        unlikelihood_weight=0.0,
        unlikelihood_window=2,
        entropy_weight=0.0,
        entropy_aware_kd_weight=0.0,
        entropy_aware_kd_threshold=0.55,
        entropy_aware_kd_sharpness=8.0,
        hidden_projection=None,
        hidden_layer="last",
        hidden_kd_metric="projected",
        span_pairs=None,
        projected_vocab_groups=None,
        projected_vocab_mask=None,
        projected_vocab_valid=None,
        reference_teacher=None,
        temperature=1.0,
    )

    assert first_only.item() < 1e-3


def test_last_response_shift_mask_selects_final_response_label() -> None:
    torch = pytest.importorskip("torch")
    loss_mask = torch.tensor(
        [
            [False, False, True, True, True],
            [False, True, False, False, False],
        ]
    )
    attention_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )

    mask = _last_response_shift_mask(torch, loss_mask, attention_mask)

    assert mask.tolist() == [[False, False, False, True], [True, False, False, False]]


def test_cross_tokenizer_recovery_loss_response_end_lm_focuses_last_response_token() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.full((1, 5, 6), -8.0)
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[False, False, False, True, True]])
    logits[0, 2, 3] = -20.0
    logits[0, 3, 4] = 20.0
    student = _Obj(logits=logits)

    end_only = _cross_tokenizer_recovery_loss(
        torch,
        student=student,
        teacher=None,
        candidate_input_ids=input_ids,
        candidate_attention_mask=attention_mask,
        candidate_loss_mask=loss_mask,
        source_attention_mask=None,
        candidate_anchor_idx=torch.tensor([0]),
        source_anchor_idx=torch.tensor([0]),
        anchor_kd_weight=0.0,
        lm_weight=0.0,
        first_token_lm_weight=0.0,
        response_end_lm_weight=1.0,
        hidden_kd_weight=0.0,
        span_kd_weight=0.0,
        span_hidden_kd_weight=0.0,
        projected_vocab_kd_weight=0.0,
        focused_projected_vocab_kd_weight=0.0,
        projected_vocab_kd_mode="teacher-to-student",
        reference_kd_weight=0.0,
        unlikelihood_weight=0.0,
        unlikelihood_window=2,
        entropy_weight=0.0,
        entropy_aware_kd_weight=0.0,
        entropy_aware_kd_threshold=0.55,
        entropy_aware_kd_sharpness=8.0,
        hidden_projection=None,
        hidden_layer="last",
        hidden_kd_metric="projected",
        span_pairs=None,
        projected_vocab_groups=None,
        projected_vocab_mask=None,
        projected_vocab_valid=None,
        reference_teacher=None,
        temperature=1.0,
    )

    assert end_only.item() < 1e-3


def test_same_tokenizer_top_k_kl_uses_teacher_subset() -> None:
    torch = pytest.importorskip("torch")
    teacher = torch.full((1, 2, 6), -10.0)
    student_same = torch.full((1, 2, 6), -10.0)
    student_wrong = torch.full((1, 2, 6), -10.0)
    teacher[..., 4] = 8.0
    teacher[..., 2] = 7.0
    student_same[..., 4] = 8.0
    student_same[..., 2] = 7.0
    student_wrong[..., 4] = 2.0
    student_wrong[..., 2] = 9.0

    same = _same_tokenizer_top_k_kl(torch, student_same, teacher, kd_vocab=6, temperature=1.0, top_k=2)
    wrong = _same_tokenizer_top_k_kl(torch, student_wrong, teacher, kd_vocab=6, temperature=1.0, top_k=2)

    assert same.mean().item() < 1e-6
    assert wrong.mean().item() > same.mean().item() + 1.0


def test_recovery_loss_top_k_kd_allows_sparse_teacher_supervision() -> None:
    torch = pytest.importorskip("torch")
    student_logits = torch.zeros((1, 4, 8), dtype=torch.float32)
    teacher_logits = torch.zeros((1, 4, 8), dtype=torch.float32)
    teacher_logits[:, :, 5] = 8.0
    teacher_logits[:, :, 3] = 7.0
    student_logits[:, :, 5] = 1.0
    student_logits[:, :, 3] = 9.0
    student = _Obj(logits=student_logits)
    teacher = _Obj(logits=teacher_logits)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)

    loss = _recovery_loss(
        torch,
        student,
        teacher=teacher,
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=None,
        kd_weight=0.0,
        lm_weight=0.0,
        unlikelihood_weight=0.0,
        unlikelihood_window=2,
        entropy_weight=0.0,
        temperature=1.0,
        top_k_kd_weight=1.0,
        top_k_kd_k=2,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 1.0


def test_same_tokenizer_hidden_loss_handles_width_mismatch_with_cka() -> None:
    torch = pytest.importorskip("torch")
    student = _Obj(hidden_states=[torch.randn(1, 5, 3)])
    teacher = _Obj(hidden_states=[torch.randn(1, 5, 7)])
    attention_mask = torch.tensor([[True, True, True, True, False]])
    loss_mask = torch.tensor([[False, False, True, True, False]])

    loss = _same_tokenizer_hidden_loss(
        torch,
        student,
        teacher,
        attention_mask,
        loss_mask,
        hidden_projection=None,
        hidden_layer="last",
        hidden_kd_metric="cka",
        hidden_token_mode="response",
    )

    assert torch.isfinite(loss)


def test_same_tokenizer_hidden_loss_all_layers_aligns_proportionally() -> None:
    torch = pytest.importorskip("torch")
    student = _Obj(hidden_states=[torch.randn(1, 5, 3) for _ in range(4)])
    teacher = _Obj(hidden_states=[torch.randn(1, 5, 7) for _ in range(8)])
    attention_mask = torch.tensor([[True, True, True, True, False]])
    loss_mask = torch.tensor([[False, False, True, True, False]])

    loss = _same_tokenizer_hidden_loss(
        torch,
        student,
        teacher,
        attention_mask,
        loss_mask,
        hidden_projection=None,
        hidden_layer="all",
        hidden_kd_metric="cka",
        hidden_token_mode="tokens",
    )

    assert torch.isfinite(loss)


def test_infer_response_char_start_finds_answer_markers() -> None:
    assert _infer_response_char_start("user: Q\nassistant: A", "auto") == len("user: Q\nassistant:")
    assert _infer_response_char_start("Question\nAnswer: 42", "answer") == len("Question\nAnswer:")
    assert _infer_response_char_start("Prompt\n```python\npass\n```", "auto") == len("Prompt\n")
    assert _infer_response_char_start("Solve: 75 plus 63 = ?\n138", "auto") == len("Solve: 75 plus 63 = ?\n")
    assert _infer_response_char_start("Question: choose one\nB", "auto") == len("Question: choose one\n")
    assert _infer_response_char_start("Question\nAnswer: 42", "assistant") is None


def test_recent_token_unlikelihood_penalizes_context_repeats() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 3, 5), dtype=torch.float32)
    logits[:, :, 2] = 8.0
    input_ids = torch.tensor([[2, 3, 4, 1]])
    labels = input_ids[:, 1:]
    mask = torch.ones_like(labels, dtype=torch.bool)

    loss = _recent_token_unlikelihood_loss(torch, logits, input_ids, labels, mask, window=2)

    assert float(loss) > 1.0


def test_recent_token_unlikelihood_can_penalize_on_policy_label_repeats() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 3, 5), dtype=torch.float32)
    logits[:, :, 2] = 8.0
    input_ids = torch.tensor([[2, 2, 2, 2]])
    labels = input_ids[:, 1:]
    mask = torch.ones_like(labels, dtype=torch.bool)

    teacher_forced_loss = _recent_token_unlikelihood_loss(torch, logits, input_ids, labels, mask, window=2)
    on_policy_loss = _recent_token_unlikelihood_loss(
        torch,
        logits,
        input_ids,
        labels,
        mask,
        window=2,
        exclude_current_label=False,
    )

    assert float(teacher_forced_loss) == pytest.approx(0.0)
    assert float(on_policy_loss) > 1.0


def test_validate_blockwise_learning_rate_allows_structural_zero_step() -> None:
    _validate_blockwise_learning_rate(0.0, 0)
    _validate_blockwise_learning_rate(1e-8, 4)

    with pytest.raises(ValueError, match="positive"):
        _validate_blockwise_learning_rate(0.0, 4)
    with pytest.raises(ValueError, match="non-negative"):
        _validate_blockwise_learning_rate(-1e-8, 0)


def test_stratified_anchor_selection_prefers_wordlike_spread() -> None:
    tokens = ["!", ",", "a", "in", "cat", "merge", "neuron", "activation", "compression", "zz"]

    selected = _select_anchor_tokens(tokens, max_anchors=5, selection="stratified")

    assert selected[:3] == ["merge", "neuron", "activation"]
    assert "!" not in selected
    assert "," not in selected


def test_untie_output_head_clones_shared_embedding_weight() -> None:
    torch = pytest.importorskip("torch")

    class _Config:
        tie_word_embeddings = True

    class _Model:
        def __init__(self) -> None:
            self.config = _Config()
            self.embed = torch.nn.Embedding(3, 4)
            self.head = torch.nn.Linear(4, 3, bias=False)
            self.head.weight = self.embed.weight

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.head

        def set_output_embeddings(self, value):
            self.head = value

    model = _Model()
    old_ptr = model.get_output_embeddings().weight.data_ptr()

    changed = _untie_output_head(torch, model)

    assert changed is True
    assert model.config.tie_word_embeddings is False
    assert model.get_input_embeddings().weight.data_ptr() == old_ptr
    assert model.get_output_embeddings().weight.data_ptr() != old_ptr
    assert torch.allclose(model.get_output_embeddings().weight.detach(), model.get_input_embeddings().weight.detach())


def test_output_head_tied_to_input_detects_shared_weight() -> None:
    torch = pytest.importorskip("torch")
    shared = torch.nn.Embedding(4, 3)
    separate = torch.nn.Linear(3, 4, bias=False)

    assert _output_head_tied_to_input(shared, shared) is True
    assert _output_head_tied_to_input(separate, shared) is False


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
        None,
        source_mask,
        candidate_anchor_idx=torch.tensor([1, 3, 5]),
        source_anchor_idx=torch.tensor([2, 4, 6]),
        anchor_kd_weight=0.7,
        lm_weight=0.3,
        first_token_lm_weight=0.0,
        response_end_lm_weight=0.0,
        hidden_kd_weight=0.0,
        span_kd_weight=0.0,
        span_hidden_kd_weight=0.0,
        projected_vocab_kd_weight=0.0,
        focused_projected_vocab_kd_weight=0.0,
        projected_vocab_kd_mode="teacher-to-student",
        reference_kd_weight=0.0,
        unlikelihood_weight=0.0,
        unlikelihood_window=2,
        entropy_weight=0.0,
        entropy_aware_kd_weight=0.0,
        entropy_aware_kd_threshold=0.55,
        entropy_aware_kd_sharpness=8.0,
        hidden_projection=None,
        hidden_layer="last",
        hidden_kd_metric="projected",
        span_pairs=None,
        projected_vocab_groups=None,
        projected_vocab_mask=None,
        projected_vocab_valid=None,
        reference_teacher=None,
        temperature=1.5,
    )

    assert torch.isfinite(loss)


def test_cross_tokenizer_recovery_loss_response_mask_ignores_prompt_tokens() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.full((1, 4, 5), -8.0)
    candidate_ids = torch.tensor([[0, 1, 2, 3]])
    candidate_mask = torch.ones_like(candidate_ids)
    candidate_loss_mask = torch.tensor([[False, False, False, True]])
    source_mask = torch.ones((1, 4), dtype=torch.long)
    logits[0, 0, 1] = -20.0
    logits[0, 1, 2] = -20.0
    logits[0, 2, 3] = 20.0
    student = _Obj(logits=logits)
    teacher = _Obj(logits=torch.zeros((1, 4, 5)))

    masked = _cross_tokenizer_recovery_loss(
        torch,
        student,
        teacher,
        candidate_ids,
        candidate_mask,
        candidate_loss_mask,
        source_mask,
        candidate_anchor_idx=torch.tensor([1, 3]),
        source_anchor_idx=torch.tensor([1, 3]),
        anchor_kd_weight=0.0,
        lm_weight=1.0,
        first_token_lm_weight=0.0,
        response_end_lm_weight=0.0,
        hidden_kd_weight=0.0,
        span_kd_weight=0.0,
        span_hidden_kd_weight=0.0,
        projected_vocab_kd_weight=0.0,
        focused_projected_vocab_kd_weight=0.0,
        projected_vocab_kd_mode="teacher-to-student",
        reference_kd_weight=0.0,
        unlikelihood_weight=0.0,
        unlikelihood_window=2,
        entropy_weight=0.0,
        entropy_aware_kd_weight=0.0,
        entropy_aware_kd_threshold=0.55,
        entropy_aware_kd_sharpness=8.0,
        hidden_projection=None,
        hidden_layer="last",
        hidden_kd_metric="projected",
        span_pairs=None,
        projected_vocab_groups=None,
        projected_vocab_mask=None,
        projected_vocab_valid=None,
        reference_teacher=None,
        temperature=1.0,
    )
    full = _cross_tokenizer_recovery_loss(
        torch,
        student,
        teacher,
        candidate_ids,
        candidate_mask,
        None,
        source_mask,
        candidate_anchor_idx=torch.tensor([1, 3]),
        source_anchor_idx=torch.tensor([1, 3]),
        anchor_kd_weight=0.0,
        lm_weight=1.0,
        first_token_lm_weight=0.0,
        response_end_lm_weight=0.0,
        hidden_kd_weight=0.0,
        span_kd_weight=0.0,
        span_hidden_kd_weight=0.0,
        projected_vocab_kd_weight=0.0,
        focused_projected_vocab_kd_weight=0.0,
        projected_vocab_kd_mode="teacher-to-student",
        reference_kd_weight=0.0,
        unlikelihood_weight=0.0,
        unlikelihood_window=2,
        entropy_weight=0.0,
        entropy_aware_kd_weight=0.0,
        entropy_aware_kd_threshold=0.55,
        entropy_aware_kd_sharpness=8.0,
        hidden_projection=None,
        hidden_layer="last",
        hidden_kd_metric="projected",
        span_pairs=None,
        projected_vocab_groups=None,
        projected_vocab_mask=None,
        projected_vocab_valid=None,
        reference_teacher=None,
        temperature=1.0,
    )

    assert masked.item() < 1e-3
    assert full.item() > masked.item() + 5.0


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
        None,
        source_mask,
        candidate_anchor_idx=torch.tensor([1, 3, 5]),
        source_anchor_idx=torch.tensor([2, 4, 6]),
        anchor_kd_weight=0.1,
        lm_weight=0.2,
        first_token_lm_weight=0.0,
        response_end_lm_weight=0.0,
        hidden_kd_weight=0.7,
        span_kd_weight=0.0,
        span_hidden_kd_weight=0.0,
        projected_vocab_kd_weight=0.0,
        focused_projected_vocab_kd_weight=0.0,
        projected_vocab_kd_mode="teacher-to-student",
        reference_kd_weight=0.0,
        unlikelihood_weight=0.0,
        unlikelihood_window=2,
        entropy_weight=0.0,
        entropy_aware_kd_weight=0.0,
        entropy_aware_kd_threshold=0.55,
        entropy_aware_kd_sharpness=8.0,
        hidden_projection=projection,
        hidden_layer="last",
        hidden_kd_metric="projected",
        span_pairs=None,
        projected_vocab_groups=None,
        projected_vocab_mask=None,
        projected_vocab_valid=None,
        reference_teacher=None,
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

    activation_pairs = _cross_tokenizer_span_pairs(
        torch,
        tokenizer,
        tokenizer,
        source_batch,
        candidate_batch,
        _canonical_token,
        max_pairs=8,
        shift_positions=False,
    )

    assert activation_pairs.tolist() == [[0, 1, 1], [0, 2, 2]]


def test_layer_tensor_alias_lookup_maps_lfm_mlp_names_to_qwen_suffixes() -> None:
    index = {
        (3, "mlp.gate_proj.weight"): "model.layers.3.mlp.gate_proj.weight",
        (3, "mlp.up_proj.weight"): "model.layers.3.mlp.up_proj.weight",
        (3, "self_attn.o_proj.weight"): "model.layers.3.self_attn.o_proj.weight",
        (3, "self_attn.q_norm.weight"): "model.layers.3.self_attn.q_norm.weight",
        (3, "self_attn.k_norm.weight"): "model.layers.3.self_attn.k_norm.weight",
    }
    folded_state = {
        "mlp.gate_proj.weight": "gate",
        "mlp.down_proj.weight": "down",
        "input_layernorm.weight": "norm",
        "self_attn.o_proj.weight": "attn_out",
        "self_attn.q_norm.weight": "q_norm",
        "self_attn.k_norm.weight": "k_norm",
    }

    assert _lookup_layer_tensor_key(index, 3, "feed_forward.w1.weight") == "model.layers.3.mlp.gate_proj.weight"
    assert _lookup_layer_tensor_key(index, 3, "feed_forward.w3.weight") == "model.layers.3.mlp.up_proj.weight"
    assert _lookup_layer_tensor_key(index, 3, "operator.out_proj.weight") == "model.layers.3.self_attn.o_proj.weight"
    assert _lookup_layer_tensor_key(index, 3, "self_attn.out_proj.weight") == "model.layers.3.self_attn.o_proj.weight"
    assert _lookup_layer_tensor_key(index, 3, "self_attn.q_layernorm.weight") == "model.layers.3.self_attn.q_norm.weight"
    assert _lookup_layer_tensor_key(index, 3, "self_attn.k_layernorm.weight") == "model.layers.3.self_attn.k_norm.weight"
    assert _lookup_layer_tensor_value(folded_state, "feed_forward.w1.weight") == "gate"
    assert _lookup_layer_tensor_value(folded_state, "feed_forward.w2.weight") == "down"
    assert _lookup_layer_tensor_value(folded_state, "operator_norm.weight") == "norm"
    assert _lookup_layer_tensor_value(folded_state, "self_attn.out_proj.weight") == "attn_out"
    assert _lookup_layer_tensor_value(folded_state, "self_attn.q_layernorm.weight") == "q_norm"
    assert _lookup_layer_tensor_value(folded_state, "self_attn.k_layernorm.weight") == "k_norm"


def test_layer_tensor_alias_lookup_maps_qwen_suffixes_to_lfm_names() -> None:
    index = {
        (3, "feed_forward.w1.weight"): "model.layers.3.feed_forward.w1.weight",
        (3, "feed_forward.w3.weight"): "model.layers.3.feed_forward.w3.weight",
        (3, "feed_forward.w2.weight"): "model.layers.3.feed_forward.w2.weight",
        (3, "operator.out_proj.weight"): "model.layers.3.operator.out_proj.weight",
        (3, "operator_norm.weight"): "model.layers.3.operator_norm.weight",
        (3, "ffn_norm.weight"): "model.layers.3.ffn_norm.weight",
    }
    folded_state = {
        "feed_forward.w1.weight": "gate",
        "feed_forward.w3.weight": "up",
        "feed_forward.w2.weight": "down",
        "operator.out_proj.weight": "attn_out",
        "operator_norm.weight": "attn_norm",
        "ffn_norm.weight": "ffn_norm",
    }

    assert _lookup_layer_tensor_key(index, 3, "mlp.gate_proj.weight") == "model.layers.3.feed_forward.w1.weight"
    assert _lookup_layer_tensor_key(index, 3, "mlp.up_proj.weight") == "model.layers.3.feed_forward.w3.weight"
    assert _lookup_layer_tensor_key(index, 3, "mlp.down_proj.weight") == "model.layers.3.feed_forward.w2.weight"
    assert _lookup_layer_tensor_key(index, 3, "self_attn.o_proj.weight") == "model.layers.3.operator.out_proj.weight"
    assert _lookup_layer_tensor_key(index, 3, "input_layernorm.weight") == "model.layers.3.operator_norm.weight"
    assert _lookup_layer_tensor_key(index, 3, "post_attention_layernorm.weight") == "model.layers.3.ffn_norm.weight"
    assert _lookup_layer_tensor_value(folded_state, "mlp.gate_proj.weight") == "gate"
    assert _lookup_layer_tensor_value(folded_state, "mlp.up_proj.weight") == "up"
    assert _lookup_layer_tensor_value(folded_state, "mlp.down_proj.weight") == "down"
    assert _lookup_layer_tensor_value(folded_state, "self_attn.o_proj.weight") == "attn_out"
    assert _lookup_layer_tensor_value(folded_state, "input_layernorm.weight") == "attn_norm"
    assert _lookup_layer_tensor_value(folded_state, "post_attention_layernorm.weight") == "ffn_norm"


def test_linear_fit_rows_for_suffix_uses_cross_family_aliases() -> None:
    torch = pytest.importorskip("torch")
    rows = {
        "input": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "output": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    }
    activation_profile = {"linear_fit_rows": {3: {"feed_forward.w2.weight": rows}}}

    result = _linear_fit_rows_for_suffix(torch, activation_profile, [3], "mlp.down_proj.weight")

    assert result is not None
    torch.testing.assert_close(result["input"], rows["input"])
    torch.testing.assert_close(result["output"], rows["output"])


def test_attention_projection_modules_supports_lfm_operator_container() -> None:
    q = _Obj()
    k = _Obj()
    v = _Obj()
    out = _Obj()
    layer = _Obj(operator=_Obj(q_proj=q, k_proj=k, v_proj=v, out_proj=out))

    modules = _attention_projection_modules(layer)

    assert modules == {"q": q, "k": k, "v": v, "o": out}


def test_neutralized_unmatched_operator_tensor_zeroes_lfm_conv_output_only() -> None:
    torch = pytest.importorskip("torch")
    conv_out = torch.ones(4, 4)
    conv_in = torch.ones(12, 4)

    neutralized = _neutralized_unmatched_operator_tensor(torch, conv_out, "conv.out_proj.weight")

    assert neutralized is not None
    assert torch.count_nonzero(neutralized).item() == 0
    assert _neutralized_unmatched_operator_tensor(torch, conv_in, "conv.in_proj.weight") is None
    assert _neutralized_unmatched_operator_tensor(torch, conv_out, "feed_forward.w2.weight") is None


def test_cross_tokenizer_overlap_span_pairs_match_split_tokens_by_offsets() -> None:
    torch = pytest.importorskip("torch")
    source_batch = {
        "input_ids": torch.tensor([[0, 10, 11, 12, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "offset_mapping": torch.tensor([[[0, 0], [0, 2], [3, 4], [4, 6], [0, 0]]]),
    }
    candidate_batch = {
        "input_ids": torch.tensor([[0, 20, 21, 22, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "offset_mapping": torch.tensor([[[0, 0], [0, 1], [1, 2], [3, 6], [0, 0]]]),
    }

    pairs = _cross_tokenizer_overlap_span_pairs(
        torch,
        source_batch,
        candidate_batch,
        max_pairs=8,
        shift_positions=True,
    )

    assert pairs.tolist() == [[0, 0, 0], [0, 1, 0], [0, 2, 2]]


def test_cross_tokenizer_span_pairs_for_mode_can_union_exact_and_overlap() -> None:
    torch = pytest.importorskip("torch")
    source_tokenizer = _Tokenizer(["<eos>", "ab", "cd"])
    candidate_tokenizer = _Tokenizer(["<eos>", "ab", "c", "d"])
    source_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "offset_mapping": torch.tensor([[[0, 0], [0, 2], [3, 5], [0, 0]]]),
    }
    candidate_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "offset_mapping": torch.tensor([[[0, 0], [0, 2], [3, 4], [4, 5]]]),
    }

    exact = _cross_tokenizer_span_pairs_for_mode(
        torch,
        source_tokenizer,
        candidate_tokenizer,
        source_batch,
        candidate_batch,
        _canonical_token,
        max_pairs=8,
        shift_positions=False,
        mode="exact",
    )
    combined = _cross_tokenizer_span_pairs_for_mode(
        torch,
        source_tokenizer,
        candidate_tokenizer,
        source_batch,
        candidate_batch,
        _canonical_token,
        max_pairs=8,
        shift_positions=False,
        mode="both",
    )

    assert exact.tolist() == [[0, 1, 1]]
    assert combined.tolist() == [[0, 1, 1], [0, 2, 2], [0, 3, 2]]


def test_focus_char_spans_select_numeric_response() -> None:
    text = "Question: What is 12 times 13?\nAnswer: 156"
    start = text.index("156")

    assert _focus_char_spans(text, "answer", "numbers") == [(start, start + 3)]
    assert _focus_char_spans(text, "answer", "response") == [(text.index("156"), len(text))]
    newline_text = "Solve: 75 plus 63 = ?\n138"
    newline_start = newline_text.index("138")
    assert _focus_char_spans(newline_text, "auto", "numbers") == [(newline_start, newline_start + 3)]


def test_answer_contrastive_candidates_build_numeric_and_choice_sets() -> None:
    numeric = _answer_contrastive_candidates("Solve: 75 plus 63 = ?\n138", "auto", max_negatives=3)
    assert numeric[0] == "Solve: 75 plus 63 = ?\n138"
    assert numeric[1:] == [
        "Solve: 75 plus 63 = ?\n4725",
        "Solve: 75 plus 63 = ?\n63",
        "Solve: 75 plus 63 = ?\n213",
    ]

    choices = _answer_contrastive_candidates("Question: pick\nAnswer: B", "auto", max_negatives=2)
    assert choices == ["Question: pick\nAnswer: B", "Question: pick\nAnswer: A", "Question: pick\nAnswer: C"]


class _OffsetTokenizer:
    def __init__(self) -> None:
        self.vocab = {str(index): index for index in range(1000)}
        self.vocab.update({"A": 1001, "B": 1002, "C": 1003, "D": 1004, "?": 1005, "+": 1006})

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **_kwargs):
        import re

        ids = []
        offsets = []
        for match in re.finditer(r"-?\d+|[A-D]|\S", text):
            token = match.group(0)
            ids.append(self.vocab.get(token, 999))
            offsets.append((match.start(), match.end()))
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def test_answer_first_token_negative_ids_include_prompt_number_distractors() -> None:
    tokenizer = _OffsetTokenizer()
    text = "Question: What is 6 + 4?\nAnswer: 10"

    negative_ids = _answer_first_token_negative_ids(
        tokenizer,
        text,
        response_loss_mode="answer",
        max_negatives=6,
        gold_id=10,
        vocab_size=2000,
    )

    assert 6 in negative_ids
    assert 4 in negative_ids
    assert 10 not in negative_ids


def test_answer_first_token_margin_loss_penalizes_prompt_number_copy() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = _OffsetTokenizer()
    text = "Question: What is 6 + 4?\nAnswer: 10"
    input_ids = torch.tensor([[1, 2, 3, 10]])
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[0, 0, 0, 1]])

    bad_logits = torch.zeros((1, 3, 2000), dtype=torch.float32)
    bad_logits[0, 2, 6] = 3.0
    good_logits = torch.zeros((1, 3, 2000), dtype=torch.float32)
    good_logits[0, 2, 10] = 4.0

    bad_loss, bad_count = _answer_first_token_margin_loss(
        torch,
        bad_logits,
        input_ids,
        attention_mask,
        loss_mask,
        tokenizer,
        [text],
        response_loss_mode="answer",
        max_negatives=6,
        margin=1.0,
    )
    good_loss, good_count = _answer_first_token_margin_loss(
        torch,
        good_logits,
        input_ids,
        attention_mask,
        loss_mask,
        tokenizer,
        [text],
        response_loss_mode="answer",
        max_negatives=6,
        margin=1.0,
    )

    assert bad_count == 1
    assert good_count == 1
    assert bad_loss.item() > good_loss.item()


def test_candidate_rank_kl_rewards_matching_teacher_preference() -> None:
    torch = pytest.importorskip("torch")
    teacher_nll = torch.tensor([0.1, 4.0, 5.0])
    matching_student_nll = torch.tensor([0.2, 4.5, 5.5])
    wrong_student_nll = torch.tensor([6.0, 0.1, 5.0])

    matching = _candidate_rank_kl_loss(torch, teacher_nll, matching_student_nll, temperature=1.0)
    wrong = _candidate_rank_kl_loss(torch, teacher_nll, wrong_student_nll, temperature=1.0)

    assert matching.item() < wrong.item()


def test_cross_tokenizer_focus_span_pairs_allow_different_token_boundaries() -> None:
    torch = pytest.importorskip("torch")
    text = "Question: What is 12 times 13?\nAnswer: 156"
    answer_start = text.index("156")
    source_batch = {
        "input_ids": torch.tensor([[0, 4, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        "offset_mapping": torch.tensor([[[0, 0], [answer_start, answer_start + 3], [0, 0], [0, 0]]]),
        "texts": [text],
    }
    candidate_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "offset_mapping": torch.tensor(
            [
                [
                    [0, 0],
                    [answer_start, answer_start + 1],
                    [answer_start + 1, answer_start + 2],
                    [answer_start + 2, answer_start + 3],
                    [0, 0],
                ]
            ]
        ),
        "texts": [text],
    }

    pairs = _cross_tokenizer_focus_span_pairs(
        torch,
        source_batch,
        candidate_batch,
        response_loss_mode="answer",
        token_focus="numbers",
        max_pairs=8,
    )

    assert len(pairs) == 1
    assert pairs[0]["batch_idx"] == 0
    assert pairs[0]["source_positions"].tolist() == [0]
    assert pairs[0]["candidate_positions"].tolist() == [0, 1, 2]


def test_focus_spans_to_span_pairs_skips_mismatched_token_counts() -> None:
    torch = pytest.importorskip("torch")
    pairs = _focus_spans_to_span_pairs(
        torch,
        [
            {
                "batch_idx": 2,
                "candidate_positions": torch.tensor([10, 11, 12, 13]),
                "source_positions": torch.tensor([20, 21]),
            }
        ],
        device="cpu",
    )

    assert pairs.tolist() == []


def test_focus_spans_to_span_pairs_keeps_exact_token_alignment() -> None:
    torch = pytest.importorskip("torch")
    pairs = _focus_spans_to_span_pairs(
        torch,
        [
            {
                "batch_idx": 2,
                "candidate_positions": torch.tensor([10, 11]),
                "source_positions": torch.tensor([20, 21]),
            }
        ],
        device="cpu",
    )

    assert pairs.tolist() == [[2, 10, 20], [2, 11, 21]]


def test_focus_spans_to_shift_mask_marks_candidate_rows() -> None:
    torch = pytest.importorskip("torch")
    mask = _focus_spans_to_shift_mask(
        torch,
        [
            {
                "batch_idx": 1,
                "candidate_positions": torch.tensor([0, 2, 99]),
                "source_positions": torch.tensor([4, 5, 6]),
            }
        ],
        shape=(2, 4),
        device="cpu",
    )

    assert mask.tolist() == [[False, False, False, False], [True, False, True, False]]


def test_focus_char_spans_numbers_include_arithmetic_expression() -> None:
    text = "Question: compute.\nAnswer: 12 * 13 = 156"

    spans = _focus_char_spans(text, response_loss_mode="answer", token_focus="numbers")

    assert [text[start:end] for start, end in spans] == ["12 * 13 = 156"]


def test_cross_tokenizer_sequence_nll_loss_rewards_candidate_span_tokens() -> None:
    torch = pytest.importorskip("torch")
    candidate_ids = torch.tensor([[0, 1, 2, 3]])
    source_ids = torch.tensor([[0, 4]])
    teacher = _Obj(logits=torch.zeros((1, 2, 6)))
    teacher.logits[0, 0, 4] = 8.0
    focus = [
        {
            "batch_idx": 0,
            "source_positions": torch.tensor([0]),
            "candidate_positions": torch.tensor([0, 1, 2]),
        }
    ]
    good_logits = torch.zeros((1, 3, 5))
    bad_logits = torch.zeros((1, 3, 5))
    for pos, label in enumerate([1, 2, 3]):
        good_logits[0, pos, label] = 6.0
        bad_logits[0, pos, label] = -6.0

    good = _cross_tokenizer_sequence_nll_loss(
        torch,
        good_logits,
        teacher,
        candidate_ids,
        source_ids,
        focus,
        teacher_confidence_floor=0.0,
    )
    bad = _cross_tokenizer_sequence_nll_loss(
        torch,
        bad_logits,
        teacher,
        candidate_ids,
        source_ids,
        focus,
        teacher_confidence_floor=0.0,
    )

    assert good.item() < bad.item()


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
        None,
        source_mask,
        candidate_anchor_idx=torch.tensor([1, 3, 5]),
        source_anchor_idx=torch.tensor([2, 4, 6]),
        anchor_kd_weight=0.0,
        lm_weight=0.2,
        first_token_lm_weight=0.0,
        response_end_lm_weight=0.0,
        hidden_kd_weight=0.0,
        span_kd_weight=0.8,
        span_hidden_kd_weight=0.0,
        projected_vocab_kd_weight=0.0,
        focused_projected_vocab_kd_weight=0.0,
        projected_vocab_kd_mode="teacher-to-student",
        reference_kd_weight=0.0,
        unlikelihood_weight=0.0,
        unlikelihood_window=2,
        entropy_weight=0.0,
        entropy_aware_kd_weight=0.0,
        entropy_aware_kd_threshold=0.55,
        entropy_aware_kd_sharpness=8.0,
        hidden_projection=None,
        hidden_layer="last",
        hidden_kd_metric="projected",
        span_pairs=torch.tensor([[0, 0, 0], [0, 1, 1]]),
        projected_vocab_groups=None,
        projected_vocab_mask=None,
        projected_vocab_valid=None,
        reference_teacher=None,
        temperature=1.5,
    )

    assert torch.isfinite(loss)


def test_cross_tokenizer_dual_space_kd_loss_is_finite() -> None:
    torch = pytest.importorskip("torch")
    student_hidden = torch.randn(1, 4, 3, dtype=torch.float32)
    teacher_hidden = torch.randn(1, 4, 4, dtype=torch.float32)
    student = _Obj(
        logits=torch.randn(1, 4, 5, dtype=torch.float32),
        hidden_states=[student_hidden, student_hidden],
    )
    teacher = _Obj(
        logits=torch.randn(1, 4, 7, dtype=torch.float32),
        hidden_states=[teacher_hidden, teacher_hidden],
    )

    loss = _cross_tokenizer_dual_space_kd_loss(
        torch,
        student,
        teacher,
        torch.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 2]], dtype=torch.long),
        torch.randn(4, 3, dtype=torch.float32),
        torch.randn(3, 4, dtype=torch.float32),
        torch.randn(7, 4, dtype=torch.float32),
        torch.randn(5, 3, dtype=torch.float32),
        torch.tensor([0, 2, 4, 6], dtype=torch.long),
        torch.tensor([0, 1, 3, 4], dtype=torch.long),
        "last",
        2.0,
    )

    assert torch.isfinite(loss)
    assert float(loss.detach().cpu()) >= 0.0


def test_cross_tokenizer_attention_influence_loss_matches_aligned_attention() -> None:
    torch = pytest.importorskip("torch")
    base = torch.tensor(
        [[0.70, 0.20, 0.10], [0.10, 0.80, 0.10], [0.20, 0.20, 0.60]],
        dtype=torch.float32,
    )
    student = _Obj(
        logits=torch.zeros((1, 3, 5), dtype=torch.float32),
        attentions=(base.view(1, 1, 3, 3),),
    )
    teacher = _Obj(
        logits=torch.zeros((1, 3, 7), dtype=torch.float32),
        attentions=(base.view(1, 1, 3, 3),),
    )
    pairs = torch.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 2]], dtype=torch.long)

    matched = _cross_tokenizer_attention_influence_loss(torch, student, teacher, pairs, layer="last")
    teacher.attentions = (torch.flip(base, dims=[-1]).view(1, 1, 3, 3),)
    mismatched = _cross_tokenizer_attention_influence_loss(torch, student, teacher, pairs, layer="last")

    assert matched.item() == pytest.approx(0.0)
    assert mismatched.item() > matched.item()


def test_cross_tokenizer_alm_chunk_loss_uses_offset_chunks() -> None:
    torch = pytest.importorskip("torch")
    text = "A cat sat"
    candidate_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    source_ids = torch.tensor([[0, 4, 5, 6]], dtype=torch.long)
    candidate_offsets = torch.tensor([[[0, 0], [0, 1], [2, 5], [6, 9]]], dtype=torch.long)
    source_offsets = torch.tensor([[[0, 0], [0, 1], [2, 5], [6, 9]]], dtype=torch.long)
    shift_student = torch.randn(1, 3, 8, dtype=torch.float32)
    teacher = _Obj(logits=torch.randn(1, 4, 8, dtype=torch.float32))

    loss, chunks = _cross_tokenizer_alm_chunk_loss(
        torch,
        shift_student,
        teacher,
        candidate_ids,
        source_ids,
        {"input_ids": source_ids, "attention_mask": torch.ones_like(source_ids), "offset_mapping": source_offsets},
        {
            "input_ids": candidate_ids,
            "attention_mask": torch.ones_like(candidate_ids),
            "offset_mapping": candidate_offsets,
            "raw_texts": [text],
        },
        response_loss_mode="none",
        max_chunks=8,
        min_chars=1,
    )

    assert chunks == 3
    assert torch.isfinite(loss)


def test_alm_text_chunks_can_scope_to_answer() -> None:
    chunks = _alm_text_chunks("Question: x\nAnswer: final value", "answer", max_chunks=8, min_chars=2)

    assert chunks == [(20, 25), (26, 31)]


def test_on_policy_prompt_text_keeps_prompt_side_of_response() -> None:
    assert _on_policy_prompt_text("Question: x\nAnswer: final", "answer") == "Question: x\nAnswer:"
    assert _on_policy_prompt_text("Write code\n```python\nreturn 1", "auto") == "Write code\n"
    assert _on_policy_prompt_text("plain text", "none") == "plain text"


def test_temporary_padding_side_restores_tokenizer_state() -> None:
    class Tokenizer:
        padding_side = "right"

    tokenizer = Tokenizer()
    with _temporary_padding_side(tokenizer, "left"):
        assert tokenizer.padding_side == "left"
    assert tokenizer.padding_side == "right"


def test_project_source_logits_to_candidate_vocab_averages_source_token_groups() -> None:
    torch = pytest.importorskip("torch")
    source_logits = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    groups = torch.tensor([[1, 0], [2, 3], [0, 0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    valid = torch.tensor([True, True, False])

    projected = _project_source_logits_to_candidate_vocab(torch, source_logits, groups, mask, valid)

    assert projected.shape == (1, 3)
    assert projected[0, 0].item() == pytest.approx(3.0)
    assert projected[0, 1].item() == pytest.approx(6.0)
    assert projected[0, 2].item() == pytest.approx(-10000.0)


def test_project_source_logits_to_candidate_vocab_uses_piece_weights() -> None:
    torch = pytest.importorskip("torch")
    source_logits = torch.tensor([[0.0, 10.0, 20.0]])
    groups = torch.tensor([[1, 2]])
    mask = torch.tensor([[1.0, 1.0]])
    weights = torch.tensor([[0.25, 0.75]])
    valid = torch.tensor([True])

    projected = _project_source_logits_to_candidate_vocab(
        torch,
        source_logits,
        groups,
        mask,
        valid,
        projected_vocab_weights=weights,
    )

    assert projected[0, 0].item() == pytest.approx(17.5)


def test_candidate_to_source_token_groups_uses_sparse_surface_fallback_for_digits() -> None:
    torch = pytest.importorskip("torch")
    source_tokenizer = _Tokenizer(["<eos>", "4", "2", "hello"])
    candidate_tokenizer = _Tokenizer(["<eos>", "42", "hello"])

    groups, mask, weights, valid = _candidate_to_source_token_groups(
        torch,
        source_tokenizer,
        candidate_tokenizer,
        max_source_tokens=3,
        device="cpu",
    )

    assert valid.tolist() == [False, True, True]
    assert groups[1, :2].tolist() == [1, 2]
    assert mask[1].tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert weights[1].sum().item() == pytest.approx(1.0)
    assert groups[2, 0].item() == 3
    assert mask[2].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert weights[2].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_projected_vocab_group_summary_reports_digit_and_multi_source_rows() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = _Tokenizer(["<eos>", "42", "a+b", "hello"])
    mask = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    valid = torch.tensor([False, True, True, True])

    summary = _projected_vocab_group_summary(torch, tokenizer, mask, valid)

    assert summary["candidate_vocab_size"] == 4
    assert summary["valid_rows"] == 3
    assert summary["multi_source_rows"] == 2
    assert summary["special_rows"] == 1
    assert summary["special_valid_rows"] == 0
    assert summary["digit_rows"] == 1
    assert summary["digit_valid_rows"] == 1
    assert summary["code_rows"] == 1
    assert summary["code_valid_rows"] == 1


def test_first_byte_projection_keeps_control_bytes_and_other_bucket() -> None:
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        all_special_ids = [0]

        def __len__(self):
            return 5

        def decode(self, ids, clean_up_tokenization_spaces=False):
            table = {
                0: "<special>",
                1: "A",
                2: "\n",
                3: "é",
                4: "",
            }
            return table[int(ids[0])]

    buckets = _tokenizer_first_byte_buckets(torch, FakeTokenizer(), device="cpu")

    assert buckets.tolist() == [256, ord("A"), ord("\n"), "é".encode("utf-8")[0], 256]

    logits = torch.tensor([[0.0, 8.0, 0.0, 0.0, 0.0], [0.0, 0.0, 8.0, 0.0, 0.0]])
    projected = _first_byte_probs_from_logits(torch, logits, buckets, temperature=0.5)

    assert projected.shape == (2, 257)
    assert int(projected[0].argmax().item()) == ord("A")
    assert int(projected[1].argmax().item()) == ord("\n")
    assert projected.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0])


def test_first_byte_head_from_output_embeddings_averages_and_normalizes_buckets() -> None:
    torch = pytest.importorskip("torch")

    output_weight = torch.tensor(
        [
            [10.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
            [0.0, 1.0],
        ]
    )
    buckets = torch.tensor([ord("A"), ord("A"), ord("B"), 256])

    head = _first_byte_head_from_output_embeddings(torch, output_weight, buckets)

    assert head.shape == (257, 2)
    assert head[ord("A")].tolist() == pytest.approx([1.0, 0.0])
    assert head[ord("B")].tolist() == pytest.approx([0.0, 1.0])
    assert head[256].tolist() == pytest.approx([0.0, 1.0])


def test_hidden_first_byte_loss_prefers_hidden_state_matching_teacher_byte() -> None:
    torch = pytest.importorskip("torch")

    class FakeOutput:
        def __init__(self, hidden):
            self.hidden_states = [torch.zeros_like(hidden), hidden]
            self.logits = torch.zeros((1, 2, 3), dtype=torch.float32)

    teacher = FakeOutput(torch.zeros((1, 2, 2), dtype=torch.float32))
    teacher.logits = torch.tensor([[[0.0, 9.0, -2.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)
    source_buckets = torch.tensor([256, ord("A"), ord("B")])
    byte_probe = torch.zeros((257, 2), dtype=torch.float32)
    byte_probe[ord("A"), 0] = 1.0
    byte_probe[ord("B"), 1] = 1.0
    pairs = torch.tensor([[0, 0, 0]], dtype=torch.long)

    good_student = FakeOutput(torch.tensor([[[4.0, 0.0], [0.0, 0.0]]], dtype=torch.float32))
    bad_student = FakeOutput(torch.tensor([[[0.0, 4.0], [0.0, 0.0]]], dtype=torch.float32))

    good_loss = _cross_tokenizer_hidden_first_byte_loss(
        torch,
        good_student,
        teacher,
        pairs,
        candidate_layer_idx=0,
        byte_probe_weight=byte_probe,
        source_byte_buckets=source_buckets,
        temperature=1.0,
    )
    bad_loss = _cross_tokenizer_hidden_first_byte_loss(
        torch,
        bad_student,
        teacher,
        pairs,
        candidate_layer_idx=0,
        byte_probe_weight=byte_probe,
        source_byte_buckets=source_buckets,
        temperature=1.0,
    )

    assert good_loss.item() < bad_loss.item()


def test_entropy_aware_kl_weight_zero_matches_forward_kl() -> None:
    torch = pytest.importorskip("torch")
    student = torch.tensor([[0.2, 1.4, -0.7], [2.0, 0.1, -1.2]])
    teacher = torch.tensor([[1.0, 0.2, -0.5], [-0.4, 1.5, 0.1]])
    temperature = 1.7

    actual = _entropy_aware_kl_loss(
        torch,
        student,
        teacher,
        temperature=temperature,
        entropy_aware_kd_weight=0.0,
        entropy_aware_kd_threshold=0.5,
        entropy_aware_kd_sharpness=8.0,
    )
    student_logp = torch.nn.functional.log_softmax(student / temperature, dim=-1)
    teacher_logp = torch.nn.functional.log_softmax(teacher / temperature, dim=-1)
    teacher_p = teacher_logp.exp()
    expected = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1).mean() * (temperature**2)

    assert actual.item() == pytest.approx(expected.item())


def test_entropy_aware_kl_masks_invalid_vocab_rows() -> None:
    torch = pytest.importorskip("torch")
    teacher = torch.tensor([[2.0, -1.0, 0.0, 50.0]])
    good_student = torch.tensor([[2.0, -1.0, 0.0, -50.0]])
    bad_invalid_student = torch.tensor([[2.0, -1.0, 0.0, 50.0]])
    valid = torch.tensor([True, True, True, False])

    good = _entropy_aware_kl_loss(torch, good_student, teacher, temperature=1.0, vocab_valid=valid)
    invalid_mass = _entropy_aware_kl_loss(torch, bad_invalid_student, teacher, temperature=1.0, vocab_valid=valid)

    assert torch.isfinite(good)
    assert invalid_mass.item() == pytest.approx(good.item())


def test_project_candidate_probs_to_source_vocab_scatters_valid_groups() -> None:
    torch = pytest.importorskip("torch")
    candidate_probs = torch.tensor([[0.6, 0.3, 0.1]])
    groups = torch.tensor([[1, 2], [3, 0], [4, 0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    valid = torch.tensor([True, True, False])

    projected, source_valid = _project_candidate_probs_to_source_vocab(
        torch,
        candidate_probs,
        groups,
        mask,
        valid,
        source_vocab_size=5,
    )

    assert source_valid.tolist() == [False, True, True, True, False]
    assert projected[0, 1].item() == pytest.approx(0.3)
    assert projected[0, 2].item() == pytest.approx(0.3)
    assert projected[0, 3].item() == pytest.approx(0.3)
    assert projected[0, 4].item() == pytest.approx(0.0)


def test_project_candidate_probs_to_source_vocab_uses_piece_weights() -> None:
    torch = pytest.importorskip("torch")
    candidate_probs = torch.tensor([[0.8, 0.2]])
    groups = torch.tensor([[1, 2], [3, 0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    weights = torch.tensor([[0.25, 0.75], [1.0, 0.0]])
    valid = torch.tensor([True, True])

    projected, source_valid = _project_candidate_probs_to_source_vocab(
        torch,
        candidate_probs,
        groups,
        mask,
        valid,
        source_vocab_size=4,
        projected_vocab_weights=weights,
    )

    assert source_valid.tolist() == [False, True, True, True]
    assert projected[0, 1].item() == pytest.approx(0.2)
    assert projected[0, 2].item() == pytest.approx(0.6)
    assert projected[0, 3].item() == pytest.approx(0.2)


def test_fit_lm_head_from_hidden_logits_reduces_reconstruction_error() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    true_head = torch.tensor([[2.0, -1.0], [0.5, 1.0], [-1.0, 1.5]])
    targets = hidden.matmul(true_head.t())
    current = torch.zeros_like(true_head)

    fitted, report = _fit_lm_head_from_hidden_logits(
        torch,
        hidden,
        targets,
        current,
        ridge=1e-6,
        head_blend_alpha=1.0,
    )

    assert fitted.shape == true_head.shape
    assert report["relative_error_after"] < report["relative_error_before"]
    assert torch.allclose(fitted, true_head, atol=1e-4)


def test_fit_lm_head_mean_shift_reduces_average_logit_error() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[1.0, 0.0], [2.0, 0.0], [1.5, 0.0]])
    current = torch.zeros((2, 2), dtype=torch.float32)
    targets = torch.tensor([[3.0, -1.0], [3.0, -1.0], [3.0, -1.0]], dtype=torch.float32)

    shifted, report = _fit_lm_head_mean_shift(
        torch,
        hidden,
        targets,
        current,
        ridge=0.0,
        head_blend_alpha=1.0,
    )
    after = hidden.matmul(shifted.t())

    assert report["mode"] == "mean-shift"
    assert report["mean_abs_logit_error_after"] < report["mean_abs_logit_error_before"]
    assert torch.allclose(after.mean(dim=0), targets.mean(dim=0), atol=1e-5)


def test_same_tokenizer_reference_loss_prefers_matching_logits() -> None:
    torch = pytest.importorskip("torch")
    reference = _Obj(logits=torch.tensor([[[0.0, 2.0], [3.0, 0.0], [0.5, 1.0]]]))
    matching_student = reference.logits[:, :-1, :].clone()
    mismatched_student = torch.flip(matching_student, dims=[-1])
    mask = torch.ones((1, 2), dtype=torch.bool)

    matching = _same_tokenizer_reference_loss(torch, matching_student, reference, mask, temperature=1.0)
    mismatched = _same_tokenizer_reference_loss(torch, mismatched_student, reference, mask, temperature=1.0)

    assert float(matching) < float(mismatched)


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


def test_absorb_input_map_into_hf_conv1d_layout() -> None:
    torch = pytest.importorskip("torch")

    class Conv1D(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    Conv1D.__module__ = "transformers.pytorch_utils"
    conv = Conv1D()
    mapping = torch.tensor([[1.0, 2.0], [0.0, 1.0]])

    _absorb_input_map_into_linears(torch, [("attn.c_proj", conv)], mapping)

    expected = torch.tensor([[7.0, 10.0], [3.0, 4.0]])
    torch.testing.assert_close(conv.weight.detach(), expected)


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


def test_recovery_attention_projection_spec_detects_output_projection() -> None:
    torch = pytest.importorskip("torch")
    output = torch.nn.Linear(3, 2, bias=False)
    layer = _Obj(self_attn=_Obj(o_proj=output))

    spec = _attention_projection_spec(layer)

    assert spec is not None
    assert spec["attention"] is layer.self_attn
    assert spec["output"] is output
    assert spec["output_name"] == "self_attn.o_proj"


def test_scale_layer_residual_output_modules_scales_attention_and_mlp_outputs() -> None:
    torch = pytest.importorskip("torch")
    attention_output = torch.nn.Linear(2, 2, bias=True)
    mlp_output = torch.nn.Linear(2, 2, bias=True)
    gate = torch.nn.Linear(2, 2, bias=False)
    up = torch.nn.Linear(2, 2, bias=False)
    attention_output.weight.data.fill_(2.0)
    attention_output.bias.data.fill_(4.0)
    mlp_output.weight.data.fill_(6.0)
    mlp_output.bias.data.fill_(8.0)
    layer = _Obj(
        self_attn=_Obj(o_proj=attention_output),
        mlp=_Obj(gate_proj=gate, up_proj=up, down_proj=mlp_output),
    )

    reports = _scale_layer_residual_output_modules(torch, layer, 0.5)

    assert [item["kind"] for item in reports] == ["attention", "mlp"]
    assert attention_output.weight.detach().flatten().tolist() == pytest.approx([1.0] * 4)
    assert attention_output.bias.detach().flatten().tolist() == pytest.approx([2.0] * 2)
    assert mlp_output.weight.detach().flatten().tolist() == pytest.approx([3.0] * 4)
    assert mlp_output.bias.detach().flatten().tolist() == pytest.approx([4.0] * 2)


def test_apply_residual_output_map_to_modules_absorbs_map_into_outputs() -> None:
    torch = pytest.importorskip("torch")
    attention_output = torch.nn.Linear(2, 2, bias=True)
    mlp_output = torch.nn.Linear(2, 2, bias=True)
    gate = torch.nn.Linear(2, 2, bias=False)
    up = torch.nn.Linear(2, 2, bias=False)
    attention_output.weight.data.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    attention_output.bias.data.copy_(torch.tensor([0.5, -1.0]))
    mlp_output.weight.data.copy_(torch.tensor([[2.0, -1.0], [0.25, 0.75]]))
    mlp_output.bias.data.copy_(torch.tensor([1.5, -0.5]))
    layer = _Obj(
        self_attn=_Obj(o_proj=attention_output),
        mlp=_Obj(gate_proj=gate, up_proj=up, down_proj=mlp_output),
    )
    inputs = torch.tensor([[1.0, -2.0], [0.25, 0.5]])
    attention_before = attention_output(inputs).detach()
    mlp_before = mlp_output(inputs).detach()
    mapping = torch.tensor([[1.25, -0.5], [0.75, 0.5]])

    reports = _apply_residual_output_map_to_modules(torch, layer, mapping)

    assert [item["kind"] for item in reports] == ["attention", "mlp"]
    assert torch.allclose(attention_output(inputs).detach(), attention_before.matmul(mapping))
    assert torch.allclose(mlp_output(inputs).detach(), mlp_before.matmul(mapping))


def test_apply_residual_output_map_to_module_only_updates_selected_output() -> None:
    torch = pytest.importorskip("torch")
    attention_output = torch.nn.Linear(2, 2, bias=True)
    mlp_output = torch.nn.Linear(2, 2, bias=True)
    attention_output.weight.data.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    attention_output.bias.data.copy_(torch.tensor([0.5, -1.0]))
    mlp_output.weight.data.copy_(torch.tensor([[2.0, -1.0], [0.25, 0.75]]))
    mlp_output.bias.data.copy_(torch.tensor([1.5, -0.5]))
    inputs = torch.tensor([[1.0, -2.0], [0.25, 0.5]])
    attention_before = attention_output(inputs).detach()
    mlp_before = mlp_output(inputs).detach()
    mapping = torch.tensor([[1.25, -0.5], [0.75, 0.5]])

    report = _apply_residual_output_map_to_module(torch, "attention", "self_attn.o_proj", attention_output, mapping)

    assert report["kind"] == "attention"
    assert torch.allclose(attention_output(inputs).detach(), attention_before.matmul(mapping))
    assert torch.allclose(mlp_output(inputs).detach(), mlp_before)


def test_residual_magnitude_compensation_accepts_eora_alias() -> None:
    torch = pytest.importorskip("torch")

    report = _apply_layer_residual_magnitude_compensation(
        torch,
        object(),
        {},
        {"skipped": True, "reason": "unit-test-skip"},
        0.5,
        1.5,
        "eora-pre",
        1e-3,
        0.01,
        16,
    )

    assert report["mode"] == "pre-eora"
    assert report["reason"] == "unit-test-skip"


def test_residual_magnitude_compensation_accepts_module_eora_alias_without_alignment() -> None:
    torch = pytest.importorskip("torch")

    report = _apply_layer_residual_magnitude_compensation(
        torch,
        object(),
        {},
        {},
        0.5,
        1.5,
        "module-eora-pre",
        1e-3,
        0.01,
        16,
    )

    assert report["mode"] == "pre-module-eora"
    assert report["reason"] == "missing_module_compensation_context"
    assert "source" in report["missing"]


def test_hf_xfamily_block_recover_parser_accepts_module_residual_compensation_mode() -> None:
    args = build_parser().parse_args(
        [
            "hf-xfamily-block-recover",
            "donor",
            "candidate",
            "out",
            "--residual-magnitude-compensation",
            "module-eora-pre",
        ]
    )

    assert args.residual_magnitude_compensation == "module-eora-pre"


def test_hf_xfamily_chain_parser_accepts_module_residual_compensation_mode() -> None:
    args = build_parser().parse_args(
        [
            "hf-xfamily-chain-anchor-compress",
            "donor",
            "anchor",
            "out",
            "--block-residual-magnitude-compensation",
            "pre-module-lowrank",
        ]
    )

    assert args.block_residual_magnitude_compensation == "pre-module-lowrank"


def test_low_rank_delta_map_truncates_delta_from_identity() -> None:
    torch = pytest.importorskip("torch")
    full_map = torch.tensor([[3.0, 0.0], [0.0, 2.0]])

    lowrank, report = _low_rank_delta_map(torch, full_map, alpha=0.5, rank=1)

    assert torch.allclose(lowrank, torch.tensor([[2.0, 0.0], [0.0, 1.0]]))
    assert report["rank"] == 1
    assert report["requested_rank"] == 1
    assert report["retained_energy_ratio"] == pytest.approx(0.8)


def test_eigenspace_low_rank_delta_map_prefers_active_axis() -> None:
    torch = pytest.importorskip("torch")
    full_map = torch.tensor([[1.2, 0.0], [0.0, 4.0]], dtype=torch.float32)
    activation_rows = torch.tensor(
        [
            [10.0, 0.01],
            [9.0, -0.01],
            [11.0, 0.0],
            [10.5, 0.01],
        ],
        dtype=torch.float32,
    )

    ordinary, ordinary_report = _low_rank_delta_map(torch, full_map, alpha=1.0, rank=1)
    eora, eora_report = _eigenspace_low_rank_delta_map(torch, full_map, activation_rows, alpha=1.0, rank=1)

    assert ordinary_report["rank"] == 1
    assert eora_report["method"] == "eigenspace_low_rank_delta"
    assert abs(float(ordinary[0, 0]) - 1.2) > abs(float(eora[0, 0]) - 1.2)
    assert abs(float(eora[1, 1]) - 4.0) > abs(float(ordinary[1, 1]) - 4.0)


def test_layer_gate_decision_accepts_non_regression() -> None:
    accepted = _layer_gate_decision(previous_ppl=100.0, candidate_ppl=99.0, max_ppl_ratio=1.0)
    rejected = _layer_gate_decision(previous_ppl=100.0, candidate_ppl=103.0, max_ppl_ratio=1.02)

    assert accepted["accepted"] is True
    assert accepted["ppl_ratio"] == pytest.approx(0.99)
    assert rejected["accepted"] is False
    assert rejected["ppl_ratio"] == pytest.approx(1.03)


def test_layer_gate_behavior_decision_rejects_behavior_regression() -> None:
    previous = {
        "pass_rate": 0.5,
        "items": [{"id": "kept", "passed": True}, {"id": "failed", "passed": False}],
    }
    same = {
        "pass_rate": 0.5,
        "items": [{"id": "kept", "passed": True}, {"id": "failed", "passed": False}],
    }
    lower = {
        "pass_rate": 0.25,
        "items": [{"id": "kept", "passed": False}, {"id": "failed", "passed": False}],
    }
    traded = {
        "pass_rate": 0.5,
        "items": [{"id": "kept", "passed": False}, {"id": "new", "passed": True}],
    }
    generation_drop = {
        "pass_rate": 0.5,
        "items": [{"id": "kept", "passed": True}, {"id": "miss", "passed": False}],
        "generation_health_rate": 0.0,
        "generation_items": [{"prompt": "gen", "passed": False}],
    }
    previous["generation_health_rate"] = 1.0
    previous["generation_items"] = [{"prompt": "gen", "passed": True}]
    same["generation_health_rate"] = 1.0
    same["generation_items"] = [{"prompt": "gen", "passed": True}]

    accepted = _layer_gate_behavior_decision(previous, same, max_behavior_drop=0.0)
    tolerated = _layer_gate_behavior_decision(previous, lower, max_behavior_drop=0.25)
    traded_rejected = _layer_gate_behavior_decision(previous, traded, max_behavior_drop=0.0)
    rejected = _layer_gate_behavior_decision(previous, lower, max_behavior_drop=0.0)
    generation_rejected = _layer_gate_behavior_decision(previous, generation_drop, max_behavior_drop=0.0)

    assert accepted["accepted"] is True
    assert tolerated["accepted"] is False
    assert tolerated["lost_passed_ids"] == ["kept"]
    assert traded_rejected["accepted"] is False
    assert traded_rejected["lost_passed_ids"] == ["kept"]
    assert rejected["accepted"] is False
    assert rejected["reason"] == "behavior_regression"
    assert rejected["behavior_drop"] == pytest.approx(0.25)
    assert generation_rejected["accepted"] is False
    assert generation_rejected["lost_generation_passed_prompts"] == ["gen"]
    assert generation_rejected["generation_health_drop"] == pytest.approx(1.0)


def test_behavior_checkpoint_selection_prioritizes_behavior_over_loss() -> None:
    incumbent = {
        "correct": 1,
        "total": 6,
        "pass_rate": 1 / 6,
        "generation_health_correct": 1,
        "generation_health_rate": 1 / 3,
        "loss": 1.0,
    }
    behavior_win = {"correct": 2, "total": 6, "pass_rate": 2 / 6, "loss": 9.0}
    generation_win = {
        "correct": 1,
        "total": 6,
        "pass_rate": 1 / 6,
        "generation_health_correct": 2,
        "generation_health_rate": 2 / 3,
        "loss": 9.0,
    }
    loss_only_win = {
        "correct": 1,
        "total": 6,
        "pass_rate": 1 / 6,
        "generation_health_correct": 1,
        "generation_health_rate": 1 / 3,
        "loss": 0.5,
    }
    lower_loss_worse_generation = {
        "correct": 1,
        "total": 6,
        "pass_rate": 1 / 6,
        "generation_health_correct": 0,
        "generation_health_rate": 0.0,
        "loss": 0.1,
    }
    worse = {"correct": 0, "total": 6, "pass_rate": 0.0, "loss": 0.1}

    assert _behavior_checkpoint_is_better(behavior_win, incumbent, incumbent_loss=1.0) is True
    assert _behavior_checkpoint_is_better(generation_win, incumbent, incumbent_loss=1.0) is True
    assert _behavior_checkpoint_is_better(loss_only_win, incumbent, incumbent_loss=1.0) is True
    assert _behavior_checkpoint_is_better(lower_loss_worse_generation, incumbent, incumbent_loss=1.0) is False
    assert _behavior_checkpoint_is_better(worse, incumbent, incumbent_loss=1.0) is False


def test_behavior_checkpoint_selection_prefers_hard_categories_after_behavior_tie() -> None:
    incumbent = {
        "correct": 2,
        "total": 4,
        "pass_rate": 0.5,
        "generation_health_correct": 1,
        "generation_health_rate": 0.5,
        "loss": 0.1,
        "items": [
            {"id": "synthetic_tokenizer_0001", "passed": True},
            {"id": "synthetic_rain_0002", "passed": True},
            {"id": "synthetic_math_0000", "passed": False},
            {"id": "synthetic_python_add_0004", "passed": False},
        ],
    }
    hard_category_win = {
        "correct": 2,
        "total": 4,
        "pass_rate": 0.5,
        "generation_health_correct": 1,
        "generation_health_rate": 0.5,
        "loss": 9.0,
        "items": [
            {"id": "synthetic_tokenizer_0001", "passed": False},
            {"id": "synthetic_rain_0002", "passed": False},
            {"id": "synthetic_math_0000", "passed": True},
            {"id": "synthetic_python_add_0004", "passed": True},
        ],
    }
    hard_category_loss = dict(incumbent)
    hard_category_loss["loss"] = 0.01

    assert _behavior_checkpoint_is_better(hard_category_win, incumbent, incumbent_loss=0.1) is True
    assert _behavior_checkpoint_is_better(hard_category_loss, hard_category_win, incumbent_loss=9.0) is False


def test_usable_candidate_sort_key_prioritizes_output_before_shrink_and_ppl() -> None:
    better_output = {
        "tag": "better-output",
        "shrink": 0.10,
        "metrics": {
            "behavior_pass_rate": 0.50,
            "choice_accuracy": 0.75,
            "generation_health_rate": 1.0,
            "perplexity_mean": 900.0,
        },
    }
    worse_output_high_shrink_low_ppl = {
        "tag": "worse-output",
        "shrink": 0.90,
        "metrics": {
            "behavior_pass_rate": 0.25,
            "choice_accuracy": 0.75,
            "generation_health_rate": 1.0,
            "perplexity_mean": 100.0,
        },
    }
    same_output_higher_shrink = {
        "tag": "higher-shrink",
        "shrink": 0.80,
        "metrics": {
            "behavior_pass_rate": 0.50,
            "choice_accuracy": 0.75,
            "generation_health_rate": 1.0,
            "perplexity_mean": 1200.0,
        },
    }

    ranked = sorted([worse_output_high_shrink_low_ppl, better_output], key=_usable_candidate_sort_key)
    tied = sorted([better_output, same_output_higher_shrink], key=_usable_candidate_sort_key)

    assert ranked[0]["tag"] == "better-output"
    assert tied[0]["tag"] == "higher-shrink"


def test_layer_gate_snapshot_indices_include_consumer_layer() -> None:
    layers = [_Obj(name=str(idx)) for idx in range(3)]

    assert _layer_gate_snapshot_indices(layers, 1, "mlp") == [1]
    assert _layer_gate_snapshot_indices(layers, 1, "attention-output") == [1]
    assert _layer_gate_snapshot_indices(layers, 1, "consumer-input") == [1, 2]
    assert _layer_gate_snapshot_indices(layers, 1, "block-interface") == [1, 2]
    assert _layer_gate_snapshot_indices(layers, 1, "all") == [1, 2]
    assert _layer_gate_snapshot_indices(layers, 2, "all") == [2]


def test_block_recovery_snapshot_modules_include_consumer_for_pre_reconstruction() -> None:
    layers = [_Obj(name=str(idx)) for idx in range(3)]
    output_head = _Obj(name="head")

    no_pre = _block_recovery_snapshot_modules(layers, 1, layers[1], output_head, "none")
    interface = _block_recovery_snapshot_modules(layers, 1, layers[1], output_head, "block-interface")
    final_layer = _block_recovery_snapshot_modules(layers, 2, layers[2], output_head, "all")

    assert sorted(no_pre) == ["block", "output_head"]
    assert interface["block"] is layers[1]
    assert interface["consumer_block"] is layers[2]
    assert interface["output_head"] is output_head
    assert "consumer_block" not in final_layer


def test_validation_reconstruction_gate_rejects_bad_map() -> None:
    torch = pytest.importorskip("torch")
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    before = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    better = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    worse = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    accepted = _validation_reconstruction_gate(
        torch,
        before,
        better,
        target,
        min_relative_improvement=0.0,
        max_cosine_drop=0.0,
    )
    rejected = _validation_reconstruction_gate(
        torch,
        before,
        worse,
        target,
        min_relative_improvement=0.0,
        max_cosine_drop=0.0,
    )

    assert accepted["accepted"] is True
    assert rejected["accepted"] is False
    assert rejected["reason"] == "insufficient_relative_error_improvement"


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


def test_recovery_train_scope_supports_head_only_and_final_norm() -> None:
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
            self.mlp = _Module()

        def parameters(self):
            return self.mlp.parameters()

    class _Model:
        def __init__(self) -> None:
            self.model = _Obj(layers=[_Layer(), _Layer()], norm=_Module())
            self.output = _Module()

        def parameters(self):
            params = []
            for layer in self.model.layers:
                params.extend(layer.parameters())
            params.extend(self.model.norm.parameters())
            params.extend(self.output.parameters())
            return params

        def get_output_embeddings(self):
            return self.output

    model = _Model()
    trainable = _set_train_scope(model, "head")
    assert trainable == 1
    assert model.output.param.requires_grad is True
    assert model.model.norm.param.requires_grad is False
    assert all(not layer.mlp.param.requires_grad for layer in model.model.layers)

    model = _Model()
    trainable = _set_train_scope(model, "head_norm")
    assert trainable == 2
    assert model.output.param.requires_grad is True
    assert model.model.norm.param.requires_grad is True
    assert all(not layer.mlp.param.requires_grad for layer in model.model.layers)


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


def test_hf_tie_head_handler_passes_flags(monkeypatch) -> None:
    import neural_forge.hf_tools as hf_tools

    captured = {}

    def fake_tie_head(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_tools, "tie_output_head_checkpoint", fake_tie_head)
    args = build_parser().parse_args(
        [
            "hf-tie-head",
            "source",
            "out",
            "--dtype",
            "float16",
            "--device",
            "cpu",
            "--trust-remote-code",
        ]
    )
    result = args.func(args)

    assert result == {"ok": True}
    assert captured == {
        "model_id_or_path": "source",
        "output_dir": "out",
        "torch_dtype": "float16",
        "device": "cpu",
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
            "--target-base-model",
            "target-base",
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
            "--hidden-basis-mode",
            "activation-pca",
            "--hidden-basis-scope",
            "layer",
            "--hidden-basis-max-rows",
            "99",
            "--activation-fold-assignment",
            "similarity",
            "--transport-support",
            "quantile",
            "--donor-layer-fold-mode",
            "average",
            "--donor-layer-group-strategy",
            "activation-geometry",
            "--attention-fold-mode",
            "activation-coupled",
            "--attention-head-dim-transport",
            "qk-vo",
            "--token-init-mode",
            "donor-omp",
            "--projection-scale",
            "target-rms",
            "--projection-blend-alpha",
            "0.25",
            "--consumer-hidden-reduction",
            "sqrt-sum",
            "--activation-fit-linears",
            "--activation-fit-target",
            "mlp-output",
            "--activation-fit-max-rows",
            "44",
            "--activation-fit-alpha",
            "0.5",
            "--activation-fit-ridge",
            "0.3",
            "--activation-fit-rank",
            "12",
            "--activation-fit-rank-mode",
            "delta",
            "--activation-fit-method",
            "lrc",
            "--activation-fit-factor-steps",
            "13",
            "--activation-fit-factor-lr",
            "0.08",
            "--activation-fit-factor-l2",
            "0.006",
            "--activation-feature-mode",
            "subspace",
            "--activation-subspace-rank",
            "5",
            "--activation-domain-balance",
            "keyword",
            "--untie-output-head",
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
        "target_base_model": "target-base",
        "target_tokenizer_model": "target-tokenizer",
        "donor_model": "donor",
        "output_dir": "out",
        "seed": 11,
        "ridge": 0.02,
        "max_anchors": 1234,
        "fold_mode": "mean",
        "hidden_basis_mode": "activation-pca",
        "hidden_basis_scope": "layer",
        "hidden_basis_max_rows": 99,
        "hidden_basis_preprocess": "centered-rowrms",
        "activation_fold_assignment": "similarity",
        "transport_backend": "sparse",
        "transport_support": "quantile",
        "donor_layer_fold_mode": "average",
        "donor_layer_group_strategy": "activation-geometry",
        "attention_fold_mode": "activation-coupled",
        "attention_head_dim_transport": "qk-vo",
        "token_init_mode": "donor-omp",
        "projection_scale": "target-rms",
        "projection_blend_alpha": 0.25,
        "consumer_hidden_reduction": "sqrt-sum",
        "activation_fit_linears": True,
        "activation_fit_target": "mlp-output",
        "activation_fit_max_rows": 44,
        "activation_fit_alpha": 0.5,
        "activation_fit_ridge": 0.3,
        "activation_fit_rank": 12,
        "activation_fit_rank_mode": "delta",
        "activation_fit_method": "lrc",
        "activation_fit_factor_steps": 13,
        "activation_fit_factor_lr": 0.08,
        "activation_fit_factor_l2": 0.006,
        "activation_fit_gate": "validation",
        "activation_fit_validation_fraction": 0.20,
        "activation_fit_min_validation_rows": 8,
        "activation_fit_min_relative_improvement": 0.0,
        "activation_fit_max_cosine_drop": 0.01,
        "activation_feature_mode": "subspace",
        "activation_subspace_rank": 5,
        "activation_domain_balance": "keyword",
        "untie_output_head": True,
        "activation_calibration_texts": ["activation text"],
        "activation_max_length": 64,
        "activation_device": "cpu",
        "neutralize_unmatched_operators": False,
        "torch_dtype": "float32",
        "trust_remote_code": True,
    }


def test_hf_donor_project_init_handler_dispatches_multi_donor(monkeypatch) -> None:
    import neural_forge.cli as cli
    import neural_forge.hf_xfamily as hf_xfamily

    captured = {}

    def fake_multi_donor_init(**kwargs):
        captured.update(kwargs)
        return {"operation": "multi_donor_projected_init", "output_dir": kwargs["output_dir"]}

    def fake_single_donor_init(**kwargs):
        raise AssertionError("single-donor initializer should not run for comma-separated donors")

    monkeypatch.setattr(hf_xfamily, "multi_donor_projected_init", fake_multi_donor_init)
    monkeypatch.setattr(hf_xfamily, "donor_projected_init", fake_single_donor_init)
    monkeypatch.setattr(cli, "_load_recovery_calibration_texts", lambda path, max_texts: ["activation text"])

    args = build_parser().parse_args(
        [
            "hf-donor-project-init",
            "out",
            "--target-config-model",
            "target-config",
            "--target-base-model",
            "target-base",
            "--target-tokenizer-model",
            "target-tokenizer",
            "--donor-model",
            "donor-a,donor-b",
            "--donor-weights",
            "0.25,0.75",
            "--activation-calibration-file",
            "activation.jsonl",
            "--activation-max-calibration-texts",
            "7",
        ]
    )
    result = args.func(args)

    assert result == {"operation": "multi_donor_projected_init", "output_dir": "out"}
    assert captured["donor_models"] == ["donor-a", "donor-b"]
    assert captured["donor_weights"] == [0.25, 0.75]
    assert captured["target_config_model"] == "target-config"
    assert captured["target_base_model"] == "target-base"
    assert captured["target_tokenizer_model"] == "target-tokenizer"
    assert captured["activation_calibration_texts"] == ["activation text"]


def test_multi_donor_projected_init_uses_weighted_stage_alphas(monkeypatch, tmp_path) -> None:
    import neural_forge.hf_xfamily as hf_xfamily

    calls = []

    def fake_single_donor_init(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        return {
            "param_count": 123,
            "layer_tensors_initialized": 5,
            "projection_scale": kwargs.get("projection_scale"),
            "projection_diagnostic_summary": {"count": 5},
            "token_report": {
                "mode": "donor-tokenized-weighted",
                "anchors_used": 12,
                "embedding_rows_initialized": 34,
            },
        }

    monkeypatch.setattr(hf_xfamily, "donor_projected_init", fake_single_donor_init)

    output = tmp_path / "multi"
    summary = hf_xfamily.multi_donor_projected_init(
        target_config_model="target",
        donor_models=["donor-a", "donor/b", "donor c"],
        donor_weights=[1.0, 2.0, 1.0],
        output_dir=output,
        target_base_model=None,
        target_tokenizer_model="target-tokenizer",
        projection_blend_alpha=0.8,
        projection_scale="geomean-rms",
        seed=41,
    )

    assert [call["donor_model"] for call in calls] == ["donor-a", "donor/b", "donor c"]
    assert calls[0]["projection_blend_alpha"] == pytest.approx(0.8)
    assert calls[1]["projection_blend_alpha"] == pytest.approx(0.8 * 2.0 / 3.0)
    assert calls[2]["projection_blend_alpha"] == pytest.approx(0.8 * 1.0 / 4.0)
    assert calls[0]["target_base_model"] is None
    assert calls[1]["target_base_model"] == str(output / "multi_donor_stages" / "00_donor-a")
    assert calls[2]["target_base_model"] == str(output / "multi_donor_stages" / "01_donor_b")
    assert calls[2]["output_dir"] == output
    assert [call["seed"] for call in calls] == [41, 42, 43]

    assert summary["operation"] == "multi_donor_projected_init"
    assert summary["donor_weights"] == [1.0, 2.0, 1.0]
    assert summary["stage_reports"][2]["effective_projection_blend_alpha"] == pytest.approx(0.2)
    assert (output / "neural_forge_multi_donor_projected_init_report.json").exists()


def test_hf_donor_project_init_handler_can_run_post_init_eval(monkeypatch) -> None:
    import neural_forge.cli as cli
    import neural_forge.hf_tools as hf_tools
    import neural_forge.hf_xfamily as hf_xfamily

    captured_init = {}
    captured_eval = {}

    def fake_project_init(**kwargs):
        captured_init.update(kwargs)
        return {"operation": "donor_projected_init", "output_dir": kwargs["output_dir"]}

    def fake_eval(**kwargs):
        captured_eval.update(kwargs)
        return {
            "param_count": 123,
            "perplexity": {"mean": 42.0, "items": ["large"]},
            "choice_accuracy": {"correct": 3, "total": 4, "accuracy": 0.75, "items": ["large"]},
            "behavior_suite": {"correct": 5, "total": 10, "pass_rate": 0.5, "items": ["large"]},
            "generation_health": {"correct": 6, "total": 10, "pass_rate": 0.6},
            "generations": ["large"],
        }

    monkeypatch.setattr(hf_xfamily, "donor_projected_init", fake_project_init)
    monkeypatch.setattr(hf_tools, "evaluate_text_model", fake_eval)
    monkeypatch.setattr(cli, "_load_recovery_calibration_texts", lambda path, max_texts: None)

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
            "--post-init-eval-suite",
            "suite.json",
            "--post-init-eval-output",
            "eval.json",
            "--post-init-max-text-evals",
            "11",
            "--post-init-max-choice-probes",
            "22",
            "--post-init-max-generation-prompts",
            "33",
            "--post-init-max-behavior-probes",
            "44",
            "--post-init-max-new-tokens",
            "55",
            "--post-init-eval-device",
            "cpu",
            "--post-init-eval-dtype",
            "float32",
            "--post-init-min-behavior-pass-rate",
            "0.4",
            "--post-init-min-generation-pass-rate",
            "0.5",
            "--post-init-min-choice-accuracy",
            "0.7",
            "--post-init-max-ppl",
            "50",
            "--trust-remote-code",
        ]
    )
    result = args.func(args)

    assert captured_init["output_dir"] == "out"
    assert captured_eval == {
        "model_id_or_path": "out",
        "output_file": Path("eval.json"),
        "torch_dtype": "float32",
        "device": "cpu",
        "trust_remote_code": True,
        "max_new_tokens": 55,
        "eval_suite": "suite.json",
        "max_text_evals": 11,
        "max_choice_probes": 22,
        "max_generation_prompts": 33,
        "max_behavior_probes": 44,
    }
    assert result == {
        "operation": "donor_projected_init_with_post_init_eval",
        "init_report": {"operation": "donor_projected_init", "output_dir": "out"},
        "post_init_eval_report": "eval.json",
        "post_init_eval_summary": {
            "param_count": 123,
            "perplexity_mean": 42.0,
            "choice_correct": 3,
            "choice_total": 4,
            "choice_accuracy": 0.75,
            "behavior_correct": 5,
            "behavior_total": 10,
            "behavior_pass_rate": 0.5,
            "generation_correct": 6,
            "generation_total": 10,
            "generation_pass_rate": 0.6,
        },
        "post_init_gate": {
            "active": True,
            "passed": True,
            "checks": [
                {
                    "name": "behavior_pass_rate",
                    "metric": "behavior_pass_rate",
                    "value": 0.5,
                    "threshold": 0.4,
                    "passed": True,
                    "direction": "min",
                },
                {
                    "name": "generation_pass_rate",
                    "metric": "generation_pass_rate",
                    "value": 0.6,
                    "threshold": 0.5,
                    "passed": True,
                    "direction": "min",
                },
                {
                    "name": "choice_accuracy",
                    "metric": "choice_accuracy",
                    "value": 0.75,
                    "threshold": 0.7,
                    "passed": True,
                    "direction": "min",
                },
                {
                    "name": "perplexity_mean",
                    "metric": "perplexity_mean",
                    "value": 42.0,
                    "threshold": 50.0,
                    "passed": True,
                    "direction": "max",
                },
            ],
        },
    }


def test_post_init_gate_rejects_collapsed_initializer_metrics() -> None:
    gate = _post_init_gate_decision(
        {
            "behavior_pass_rate": 0.0,
            "generation_pass_rate": 0.0,
            "choice_accuracy": 0.17,
            "perplexity_mean": 232_000_000.0,
        },
        min_behavior_pass_rate=0.1,
        min_generation_pass_rate=0.1,
        min_choice_accuracy=0.25,
        max_ppl=100_000.0,
    )

    assert gate["active"] is True
    assert gate["passed"] is False
    assert [check["passed"] for check in gate["checks"]] == [False, False, False, False]


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
            "--first-token-lm-weight",
            "1.7",
            "--answer-contrastive-weight",
            "0.31",
            "--answer-contrastive-max-negatives",
            "6",
            "--unlikelihood-weight",
            "0.09",
            "--unlikelihood-window",
            "18",
            "--entropy-weight",
            "0.003",
            "--response-loss-mode",
            "auto",
            "--append-eos",
            "--train-scope",
            "last2",
            "--dtype",
            "float32",
            "--device",
            "cpu",
            "--grad-clip",
            "0.5",
            "--behavior-eval-suite",
            "lm-suite.json",
            "--behavior-max-generation-prompts",
            "8",
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
        "first_token_lm_weight": 1.7,
        "answer_contrastive_weight": 0.31,
        "answer_contrastive_max_negatives": 6,
        "unlikelihood_weight": 0.09,
        "unlikelihood_window": 18,
        "entropy_weight": 0.003,
        "response_loss_mode": "auto",
        "append_eos": True,
        "temperature": 1.0,
        "train_scope": "last2",
        "torch_dtype": "float32",
        "device": "cpu",
        "trust_remote_code": False,
        "calibration_texts": ["calibration text"],
        "grad_clip": 0.5,
        "report_interval": None,
        "save_best_loss": False,
        "save_best_behavior": False,
        "behavior_eval_interval": None,
        "behavior_max_new_tokens": 96,
        "behavior_max_probes": 6,
        "behavior_eval_suite": "lm-suite.json",
        "behavior_max_generation_prompts": 8,
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
            "--first-token-lm-weight",
            "0.73",
            "--response-end-lm-weight",
            "0.37",
            "--focused-lm-weight",
            "0.27",
            "--hidden-kd-weight",
            "0.25",
            "--span-kd-weight",
            "0.15",
            "--span-hidden-kd-weight",
            "0.12",
            "--dual-space-kd-weight",
            "0.19",
            "--attention-influence-weight",
            "0.17",
            "--attention-influence-layer",
            "middle",
            "--projected-vocab-kd-weight",
            "0.22",
            "--focused-projected-vocab-kd-weight",
            "0.26",
            "--projected-vocab-kd-mode",
            "both",
            "--projected-vocab-max-source-tokens",
            "6",
            "--reference-model",
            "host-reference",
            "--reference-kd-weight",
            "0.2",
            "--unlikelihood-weight",
            "0.11",
            "--unlikelihood-window",
            "22",
            "--entropy-weight",
            "0.005",
            "--response-loss-mode",
            "auto",
            "--token-kd-weight",
            "0.18",
            "--token-focus",
            "code",
            "--token-max-pairs",
            "55",
            "--token-teacher-confidence-floor",
            "0.15",
            "--alm-kd-weight",
            "0.14",
            "--alm-max-chunks",
            "22",
            "--alm-min-chars",
            "3",
            "--answer-contrastive-weight",
            "0.21",
            "--answer-contrastive-max-negatives",
            "5",
            "--answer-first-token-margin-weight",
            "0.43",
            "--answer-first-token-margin",
            "1.8",
            "--answer-first-token-max-negatives",
            "10",
            "--teacher-rank-weight",
            "0.31",
            "--teacher-rank-temperature",
            "0.8",
            "--teacher-rank-max-negatives",
            "6",
            "--teacher-generate-lm-weight",
            "0.29",
            "--teacher-generate-max-new-tokens",
            "37",
            "--stepwise-donor-kd-weight",
            "0.33",
            "--stepwise-donor-max-new-tokens",
            "13",
            "--on-policy-anchor-kd-weight",
            "0.41",
            "--on-policy-unlikelihood-weight",
            "0.52",
            "--on-policy-hidden-kd-weight",
            "0.46",
            "--on-policy-max-new-tokens",
            "17",
            "--on-policy-interval",
            "3",
            "--append-eos",
            "--entropy-aware-kd-weight",
            "0.35",
            "--entropy-aware-kd-threshold",
            "0.62",
            "--entropy-aware-kd-sharpness",
            "7.5",
            "--span-max-pairs",
            "88",
            "--hidden-ridge",
            "0.03",
            "--hidden-layer",
            "middle",
            "--hidden-kd-metric",
            "both",
            "--hidden-max-anchors",
            "99",
            "--temperature",
            "1.7",
            "--max-anchors",
            "321",
            "--anchor-selection",
            "stratified",
            "--anchor-min-chars",
            "2",
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
            "--report-interval",
            "2",
            "--save-best-loss",
            "--save-best-behavior",
            "--behavior-eval-interval",
            "4",
            "--behavior-max-new-tokens",
            "21",
            "--behavior-max-probes",
            "5",
            "--behavior-eval-suite",
            "suite.json",
            "--behavior-max-generation-prompts",
            "9",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "3",
            "--calibration-order",
            "domain-balanced",
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
        "first_token_lm_weight": 0.73,
        "response_end_lm_weight": 0.37,
        "focused_lm_weight": 0.27,
        "hidden_kd_weight": 0.25,
        "span_kd_weight": 0.15,
        "span_hidden_kd_weight": 0.12,
        "dual_space_kd_weight": 0.19,
        "attention_influence_weight": 0.17,
        "attention_influence_layer": "middle",
        "projected_vocab_kd_weight": 0.22,
        "focused_projected_vocab_kd_weight": 0.26,
        "projected_vocab_kd_mode": "both",
        "projected_vocab_max_source_tokens": 6,
        "reference_model": "host-reference",
        "reference_kd_weight": 0.2,
        "unlikelihood_weight": 0.11,
        "unlikelihood_window": 22,
        "entropy_weight": 0.005,
        "response_loss_mode": "auto",
        "token_kd_weight": 0.18,
        "token_focus": "code",
        "token_max_pairs": 55,
        "token_teacher_confidence_floor": 0.15,
        "alm_kd_weight": 0.14,
        "alm_max_chunks": 22,
        "alm_min_chars": 3,
        "answer_contrastive_weight": 0.21,
        "answer_contrastive_max_negatives": 5,
        "answer_first_token_margin_weight": 0.43,
        "answer_first_token_margin": 1.8,
        "answer_first_token_max_negatives": 10,
        "teacher_rank_weight": 0.31,
        "teacher_rank_temperature": 0.8,
        "teacher_rank_max_negatives": 6,
        "teacher_generate_lm_weight": 0.29,
        "teacher_generate_max_new_tokens": 37,
        "stepwise_donor_kd_weight": 0.33,
        "stepwise_donor_max_new_tokens": 13,
        "on_policy_anchor_kd_weight": 0.41,
        "on_policy_unlikelihood_weight": 0.52,
        "on_policy_hidden_kd_weight": 0.46,
        "on_policy_max_new_tokens": 17,
        "on_policy_interval": 3,
        "append_eos": True,
        "entropy_aware_kd_weight": 0.35,
        "entropy_aware_kd_threshold": 0.62,
        "entropy_aware_kd_sharpness": 7.5,
        "span_max_pairs": 88,
        "hidden_ridge": 0.03,
        "hidden_layer": "middle",
        "hidden_kd_metric": "both",
        "hidden_max_anchors": 99,
        "temperature": 1.7,
        "max_anchors": 321,
        "anchor_selection": "stratified",
        "anchor_min_chars": 2,
        "train_scope": "last2",
        "torch_dtype": "float16",
        "candidate_torch_dtype": "float32",
        "device": "cpu",
        "trust_remote_code": False,
        "calibration_texts": ["calibration text"],
        "calibration_order": "domain-balanced",
        "grad_clip": 0.5,
        "report_interval": 2,
        "save_best_loss": True,
        "save_best_behavior": True,
        "behavior_eval_interval": 4,
        "behavior_max_new_tokens": 21,
        "behavior_max_probes": 5,
        "behavior_eval_suite": "suite.json",
        "behavior_max_generation_prompts": 9,
    }


def test_hf_xfamily_calibrate_head_handler_passes_flags(monkeypatch) -> None:
    import neural_forge.cli as cli
    import neural_forge.hf_recovery as hf_recovery

    captured = {}

    def fake_calibrate(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_recovery, "calibrate_cross_tokenizer_lm_head", fake_calibrate)
    monkeypatch.setattr(cli, "_load_recovery_calibration_texts", lambda path, max_texts: ["calibration text"])
    args = build_parser().parse_args(
        [
            "hf-xfamily-calibrate-head",
            "donor",
            "candidate",
            "out",
            "--ridge",
            "0.04",
            "--head-blend-alpha",
            "0.7",
            "--head-calibration-mode",
            "mean-shift",
            "--max-length",
            "80",
            "--max-calibration-rows",
            "222",
            "--projected-vocab-max-source-tokens",
            "5",
            "--dtype",
            "float16",
            "--candidate-dtype",
            "float32",
            "--device",
            "cpu",
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
        "ridge": 0.04,
        "head_blend_alpha": 0.7,
        "head_calibration_mode": "mean-shift",
        "max_length": 80,
        "max_calibration_rows": 222,
        "projected_vocab_max_source_tokens": 5,
        "torch_dtype": "float16",
        "candidate_torch_dtype": "float32",
        "device": "cpu",
        "trust_remote_code": False,
        "calibration_texts": ["calibration text"],
    }


def test_hf_rank_evals_handler_passes_flags(monkeypatch) -> None:
    import neural_forge.hf_tools as hf_tools

    captured = {}

    def fake_rank(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_tools, "rank_evaluation_reports", fake_rank)
    args = build_parser().parse_args(
        [
            "hf-rank-evals",
            "rank.json",
            "first.json",
            "second.json",
            "--min-continuation-chars",
            "12",
            "--min-unique-token-ratio",
            "0.35",
        ]
    )
    result = args.func(args)

    assert result == {"ok": True}
    assert captured == {
        "reports": ["first.json", "second.json"],
        "output_file": "rank.json",
        "min_continuation_chars": 12,
        "min_unique_token_ratio": 0.35,
    }


def test_hf_rank_evals_handler_expands_report_globs(monkeypatch, tmp_path) -> None:
    import neural_forge.hf_tools as hf_tools

    first = tmp_path / "first-broad.json"
    second = tmp_path / "second-broad.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    captured = {}

    def fake_rank(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_tools, "rank_evaluation_reports", fake_rank)
    args = build_parser().parse_args(
        [
            "hf-rank-evals",
            "rank.json",
            str(tmp_path / "*-broad.json"),
        ]
    )
    result = args.func(args)

    assert result == {"ok": True}
    assert captured["reports"] == [str(first), str(second)]


def test_hf_eval_handler_passes_broad_suite_flags(monkeypatch) -> None:
    import neural_forge.hf_tools as hf_tools

    captured = {}

    def fake_eval(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_tools, "evaluate_text_model", fake_eval)
    args = build_parser().parse_args(
        [
            "hf-eval",
            "model",
            "report.json",
            "--dtype",
            "float32",
            "--device",
            "cpu",
            "--max-new-tokens",
            "33",
            "--eval-suite",
            "suite.json",
            "--max-text-evals",
            "7",
            "--max-choice-probes",
            "8",
            "--max-generation-prompts",
            "9",
            "--max-behavior-probes",
            "10",
        ]
    )
    result = args.func(args)

    assert result == {"ok": True}
    assert captured == {
        "model_id_or_path": "model",
        "output_file": "report.json",
        "torch_dtype": "float32",
        "device": "cpu",
        "trust_remote_code": False,
        "max_new_tokens": 33,
        "eval_suite": "suite.json",
        "max_text_evals": 7,
        "max_choice_probes": 8,
        "max_generation_prompts": 9,
        "max_behavior_probes": 10,
    }


def test_hf_build_eval_suite_handler_passes_counts(monkeypatch) -> None:
    import neural_forge.hf_tools as hf_tools

    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_tools, "build_eval_suite", fake_build)
    args = build_parser().parse_args(
        [
            "hf-build-eval-suite",
            "suite.json",
            "--behavior-count",
            "1000",
            "--generation-count",
            "125",
            "--choice-count",
            "250",
            "--text-count",
            "32",
            "--seed",
            "9",
        ]
    )
    result = args.func(args)

    assert result == {"ok": True}
    assert captured == {
        "output_file": "suite.json",
        "behavior_count": 1000,
        "generation_count": 125,
        "choice_count": 250,
        "text_count": 32,
        "seed": 9,
    }


def test_hf_build_recovery_calibration_handler_passes_counts(monkeypatch) -> None:
    import neural_forge.hf_tools as hf_tools

    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_tools, "build_recovery_calibration", fake_build)
    args = build_parser().parse_args(
        [
            "hf-build-recovery-calibration",
            "recovery.jsonl",
            "--math-count",
            "1200",
            "--code-count",
            "240",
            "--tokenizer-count",
            "80",
            "--model-surgery-count",
            "70",
            "--commonsense-count",
            "60",
            "--seed",
            "31",
            "--exclude-eval-suite",
            "heldout.json",
        ]
    )
    result = args.func(args)

    assert result == {"ok": True}
    assert captured == {
        "output_file": "recovery.jsonl",
        "math_count": 1200,
        "code_count": 240,
        "tokenizer_count": 80,
        "model_surgery_count": 70,
        "commonsense_count": 60,
        "seed": 31,
        "exclude_eval_suite": "heldout.json",
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


def test_hf_xfamily_reconstruct_handler_passes_flags(monkeypatch) -> None:
    import neural_forge.cli as cli
    import neural_forge.hf_recovery as hf_recovery

    captured = {}

    def fake_reconstruct(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_recovery, "reconstruct_cross_tokenizer_causal_lm", fake_reconstruct)
    monkeypatch.setattr(cli, "_load_recovery_calibration_texts", lambda path, max_texts: ["calibration text"])
    args = build_parser().parse_args(
        [
            "hf-xfamily-reconstruct",
            "donor",
            "candidate",
            "out",
            "--ridge",
            "0.03",
            "--reconstruction-alpha",
            "0.6",
            "--reconstruction-target",
            "block-interface",
            "--donor-layer-fold-mode",
            "fold",
            "--reconstruction-objective",
            "cosine",
            "--cosine-steps",
            "11",
            "--cosine-lr",
            "0.04",
            "--cosine-l2",
            "0.002",
            "--max-length",
            "96",
            "--max-tokens-per-layer",
            "144",
            "--max-anchors",
            "333",
            "--hidden-projection-scope",
            "layer-input",
            "--layer-limit",
            "4",
            "--fit-gate",
            "validation",
            "--fit-gate-validation-fraction",
            "0.2",
            "--fit-gate-min-validation-rows",
            "5",
            "--fit-gate-min-relative-improvement",
            "0.01",
            "--fit-gate-max-cosine-drop",
            "0.03",
            "--layer-gate-metric",
            "behavior",
            "--layer-gate-max-ppl-ratio",
            "1.01",
            "--layer-gate-max-texts",
            "2",
            "--layer-gate-max-behavior-drop",
            "0.17",
            "--layer-gate-behavior-max-new-tokens",
            "33",
            "--layer-gate-eval-suite",
            "reconstruct-suite.json",
            "--layer-gate-max-generation-prompts",
            "44",
            "--dtype",
            "float16",
            "--candidate-dtype",
            "float32",
            "--device",
            "cpu",
            "--trust-remote-code",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "5",
        ]
    )
    result = args.func(args)

    assert result == {"ok": True}
    assert captured == {
        "source_model": "donor",
        "candidate_model": "candidate",
        "output_dir": "out",
        "ridge": 0.03,
        "reconstruction_alpha": 0.6,
        "reconstruction_target": "block-interface",
        "donor_layer_fold_mode": "fold",
        "reconstruction_objective": "cosine",
        "cosine_steps": 11,
        "cosine_lr": 0.04,
        "cosine_l2": 0.002,
        "max_length": 96,
        "max_tokens_per_layer": 144,
        "max_anchors": 333,
        "hidden_projection_scope": "layer-input",
        "layer_limit": 4,
        "fit_gate": "validation",
        "fit_gate_validation_fraction": 0.2,
        "fit_gate_min_validation_rows": 5,
        "fit_gate_min_relative_improvement": 0.01,
        "fit_gate_max_cosine_drop": 0.03,
        "layer_gate_metric": "behavior",
        "layer_gate_max_ppl_ratio": 1.01,
        "layer_gate_max_texts": 2,
        "layer_gate_max_behavior_drop": 0.17,
        "layer_gate_behavior_max_new_tokens": 33,
        "layer_gate_eval_suite": "reconstruct-suite.json",
        "layer_gate_max_generation_prompts": 44,
        "torch_dtype": "float16",
        "candidate_torch_dtype": "float32",
        "device": "cpu",
        "trust_remote_code": True,
        "calibration_texts": ["calibration text"],
    }


def test_hf_xfamily_block_recover_handler_passes_flags(monkeypatch) -> None:
    import neural_forge.cli as cli
    import neural_forge.hf_recovery as hf_recovery

    captured = {}

    def fake_block_recover(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_recovery, "blockwise_reconstruct_cross_tokenizer_causal_lm", fake_block_recover)
    monkeypatch.setattr(cli, "_load_recovery_calibration_texts", lambda path, max_texts: ["block text"])
    args = build_parser().parse_args(
        [
            "hf-xfamily-block-recover",
            "donor",
            "candidate",
            "out",
            "--steps-per-layer",
            "7",
            "--batch-size",
            "2",
            "--max-length",
            "80",
            "--lr",
            "0.0004",
            "--lm-weight",
            "0.15",
            "--response-lm-weight",
            "0.55",
            "--first-token-lm-weight",
            "0.45",
            "--response-end-lm-weight",
            "0.25",
            "--response-loss-mode",
            "answer",
            "--hidden-ridge",
            "0.02",
            "--hidden-projection-scope",
            "layer-input",
            "--hidden-metric",
            "projected-geometry",
            "--block-target",
            "delta",
            "--residual-magnitude-compensation",
            "lowrank-pre",
            "--residual-compensation-min-scale",
            "0.25",
            "--residual-compensation-max-scale",
            "1.75",
            "--residual-compensation-linear-ridge",
            "0.003",
            "--residual-compensation-alpha",
            "0.35",
            "--residual-compensation-linear-rank",
            "11",
            "--pre-reconstruction-target",
            "block-interface",
            "--pre-reconstruction-ridge",
            "0.006",
            "--pre-reconstruction-alpha",
            "0.04",
            "--pre-reconstruction-objective",
            "cosine",
            "--pre-reconstruction-cosine-steps",
            "5",
            "--pre-reconstruction-cosine-lr",
            "0.02",
            "--pre-reconstruction-cosine-l2",
            "0.0003",
            "--pre-reconstruction-fit-gate",
            "validation",
            "--pre-reconstruction-validation-fraction",
            "0.3",
            "--pre-reconstruction-min-validation-rows",
            "9",
            "--pre-reconstruction-max-tokens-per-layer",
            "44",
            "--pre-reconstruction-min-relative-improvement",
            "0.02",
            "--pre-reconstruction-max-cosine-drop",
            "0.04",
            "--max-anchors",
            "222",
            "--span-max-pairs",
            "77",
            "--span-pair-mode",
            "both",
            "--donor-layer-fold-mode",
            "average",
            "--donor-layer-group-strategy",
            "activation-energy",
            "--layer-limit",
            "3",
            "--layer-gate-metric",
            "behavior",
            "--layer-gate-max-ppl-ratio",
            "1.05",
            "--layer-gate-max-texts",
            "2",
            "--layer-gate-max-behavior-drop",
            "0.12",
            "--layer-gate-behavior-max-new-tokens",
            "41",
            "--layer-gate-eval-suite",
            "broad_suite.json",
            "--layer-gate-max-generation-prompts",
            "17",
            "--on-policy-anchor-kd-weight",
            "0.31",
            "--on-policy-unlikelihood-weight",
            "0.27",
            "--on-policy-hidden-kd-weight",
            "0.18",
            "--on-policy-max-new-tokens",
            "13",
            "--on-policy-interval",
            "4",
            "--on-policy-response-loss-mode",
            "answer",
            "--on-policy-unlikelihood-window",
            "9",
            "--stepwise-donor-kd-weight",
            "0.22",
            "--stepwise-donor-max-new-tokens",
            "5",
            "--stepwise-donor-interval",
            "3",
            "--teacher-generate-lm-weight",
            "0.17",
            "--teacher-generate-hidden-kd-weight",
            "0.13",
            "--teacher-generate-projected-vocab-kd-weight",
            "0.15",
            "--teacher-generate-byte-kd-weight",
            "0.09",
            "--teacher-generate-max-new-tokens",
            "23",
            "--alm-kd-weight",
            "0.08",
            "--alm-max-chunks",
            "12",
            "--alm-min-chars",
            "3",
            "--projected-vocab-kd-weight",
            "0.19",
            "--focused-block-hidden-kd-weight",
            "0.16",
            "--focused-projected-vocab-kd-weight",
            "0.14",
            "--focused-lm-weight",
            "0.12",
            "--token-focus",
            "numbers",
            "--projected-vocab-kd-mode",
            "both",
            "--byte-kd-weight",
            "0.11",
            "--byte-hidden-kd-weight",
            "0.07",
            "--block-io-kd-weight",
            "0.21",
            "--block-io-ridge",
            "0.004",
            "--block-io-loss",
            "cosine",
            "--train-output-head-with-kd",
            "--untie-output-head-for-kd",
            "--allow-tied-output-head-training",
            "--module-output-kd-weight",
            "0.44",
            "--module-output-targets",
            "mlp",
            "--module-output-transport",
            "both",
            "--module-output-temperature",
            "0.6",
            "--module-output-sinkhorn-iterations",
            "9",
            "--projected-vocab-max-source-tokens",
            "6",
            "--temperature",
            "1.6",
            "--dtype",
            "float16",
            "--candidate-dtype",
            "float32",
            "--device",
            "cpu",
            "--trust-remote-code",
            "--calibration-file",
            "calibration.jsonl",
            "--max-calibration-texts",
            "6",
            "--calibration-order",
            "domain-balanced",
            "--grad-clip",
            "0.7",
            "--report-interval",
            "2",
            "--save-best-layer-loss",
            "--save-best-layer-behavior",
            "--layer-behavior-eval-interval",
            "2",
            "--layer-checkpoint-interval",
            "5",
        ]
    )
    result = args.func(args)

    assert result == {"ok": True}
    assert captured == {
        "source_model": "donor",
        "candidate_model": "candidate",
        "output_dir": "out",
        "steps_per_layer": 7,
        "batch_size": 2,
        "max_length": 80,
        "lr": 0.0004,
        "lm_weight": 0.15,
        "response_lm_weight": 0.55,
        "first_token_lm_weight": 0.45,
        "response_end_lm_weight": 0.25,
        "response_loss_mode": "answer",
        "hidden_ridge": 0.02,
        "hidden_projection_scope": "layer-input",
        "hidden_metric": "projected-geometry",
        "block_target": "delta",
        "residual_magnitude_compensation": "lowrank-pre",
        "residual_compensation_min_scale": 0.25,
        "residual_compensation_max_scale": 1.75,
        "residual_compensation_linear_ridge": 0.003,
        "residual_compensation_alpha": 0.35,
        "residual_compensation_linear_rank": 11,
        "pre_reconstruction_target": "block-interface",
        "pre_reconstruction_ridge": 0.006,
        "pre_reconstruction_alpha": 0.04,
        "pre_reconstruction_objective": "cosine",
        "pre_reconstruction_cosine_steps": 5,
        "pre_reconstruction_cosine_lr": 0.02,
        "pre_reconstruction_cosine_l2": 0.0003,
        "pre_reconstruction_fit_gate": "validation",
        "pre_reconstruction_validation_fraction": 0.3,
        "pre_reconstruction_min_validation_rows": 9,
        "pre_reconstruction_max_tokens_per_layer": 44,
        "pre_reconstruction_min_relative_improvement": 0.02,
        "pre_reconstruction_max_cosine_drop": 0.04,
        "max_anchors": 222,
        "span_max_pairs": 77,
        "span_pair_mode": "both",
        "donor_layer_fold_mode": "average",
        "donor_layer_group_strategy": "activation-energy",
        "layer_limit": 3,
        "layer_gate_metric": "behavior",
        "layer_gate_max_ppl_ratio": 1.05,
        "layer_gate_max_texts": 2,
        "layer_gate_max_behavior_drop": 0.12,
        "layer_gate_behavior_max_new_tokens": 41,
        "layer_gate_eval_suite": "broad_suite.json",
        "layer_gate_max_generation_prompts": 17,
        "on_policy_anchor_kd_weight": 0.31,
        "on_policy_unlikelihood_weight": 0.27,
        "on_policy_hidden_kd_weight": 0.18,
        "on_policy_max_new_tokens": 13,
        "on_policy_interval": 4,
        "on_policy_response_loss_mode": "answer",
        "on_policy_unlikelihood_window": 9,
        "stepwise_donor_kd_weight": 0.22,
        "stepwise_donor_max_new_tokens": 5,
        "stepwise_donor_interval": 3,
        "teacher_generate_lm_weight": 0.17,
        "teacher_generate_hidden_kd_weight": 0.13,
        "teacher_generate_projected_vocab_kd_weight": 0.15,
        "teacher_generate_byte_kd_weight": 0.09,
        "teacher_generate_max_new_tokens": 23,
        "alm_kd_weight": 0.08,
        "alm_max_chunks": 12,
        "alm_min_chars": 3,
        "projected_vocab_kd_weight": 0.19,
        "focused_block_hidden_kd_weight": 0.16,
        "focused_projected_vocab_kd_weight": 0.14,
        "focused_lm_weight": 0.12,
        "token_focus": "numbers",
        "projected_vocab_kd_mode": "both",
        "byte_kd_weight": 0.11,
        "byte_hidden_kd_weight": 0.07,
        "block_io_kd_weight": 0.21,
        "block_io_ridge": 0.004,
        "block_io_loss": "cosine",
        "train_output_head_with_kd": True,
        "untie_output_head_for_kd": True,
        "allow_tied_output_head_training": True,
        "module_output_kd_weight": 0.44,
        "module_output_targets": "mlp",
        "module_output_transport": "both",
        "module_output_temperature": 0.6,
        "module_output_sinkhorn_iterations": 9,
        "projected_vocab_max_source_tokens": 6,
        "temperature": 1.6,
        "torch_dtype": "float16",
        "candidate_torch_dtype": "float32",
        "device": "cpu",
        "trust_remote_code": True,
        "calibration_texts": ["block text"],
        "calibration_order": "domain-balanced",
        "grad_clip": 0.7,
        "report_interval": 2,
        "save_best_layer_loss": True,
        "save_best_layer_behavior": True,
        "layer_behavior_eval_interval": 2,
        "layer_checkpoint_interval": 5,
    }


def test_chain_stage_promotion_requires_gate_and_behavior_rank() -> None:
    incumbent = {
        "tag": "anchor",
        "shrink": 0.0,
        "metrics": {
            "behavior_pass_rate": 0.80,
            "choice_accuracy": 0.60,
            "generation_health_rate": 1.0,
            "perplexity_mean": 100.0,
        },
    }
    better = {
        "tag": "block",
        "shrink": 0.0,
        "metrics": {
            "behavior_pass_rate": 0.81,
            "choice_accuracy": 0.60,
            "generation_health_rate": 1.0,
            "perplexity_mean": 120.0,
        },
    }
    worse = {
        "tag": "alm",
        "shrink": 0.0,
        "metrics": {
            "behavior_pass_rate": 0.79,
            "choice_accuracy": 0.70,
            "generation_health_rate": 1.0,
            "perplexity_mean": 50.0,
        },
    }

    assert _chain_stage_promotes(incumbent, better, {"passed": True}, "gate-and-rank") is True
    assert _chain_stage_promotes(incumbent, worse, {"passed": True}, "gate-and-rank") is False
    assert _chain_stage_promotes(incumbent, better, {"passed": False}, "gate-and-rank") is False
    assert _chain_stage_promotes(incumbent, worse, {"passed": True}, "gate-only") is True


def test_parse_chain_stages_rejects_unknown_stage() -> None:
    assert _parse_chain_stages("block,alm") == ["block", "alm"]
    assert _parse_chain_stages(["init", "block"]) == ["init", "block"]

    with pytest.raises(ValueError, match="Unsupported chain stages"):
        _parse_chain_stages("block,banana")


def test_parse_source_models_and_weights_validates_multi_donor() -> None:
    assert _parse_source_models_and_weights("donor-a, donor-b", "0.25,0.75") == (
        ["donor-a", "donor-b"],
        [0.25, 0.75],
    )
    assert _parse_source_models_and_weights(["donor-a"], None) == (["donor-a"], None)

    with pytest.raises(ValueError, match="source_weights"):
        _parse_source_models_and_weights("donor-a,donor-b", "1.0")
    with pytest.raises(ValueError, match="positive finite"):
        _parse_source_models_and_weights("donor-a", "0")
    with pytest.raises(ValueError, match="source_model"):
        _parse_source_models_and_weights(",", None)


def test_chain_anchor_orchestrator_preserves_incumbent_on_failed_stage(monkeypatch, tmp_path) -> None:
    import neural_forge.hf_recovery as hf_recovery
    import neural_forge.hf_xfamily as hf_xfamily

    eval_calls = []
    block_captured = {}

    def fake_eval(model_id_or_path, output_file, **kwargs):
        eval_calls.append(str(model_id_or_path))
        report = {
            "model": str(model_id_or_path),
            "param_count": 10,
            "perplexity": {"mean": 100.0},
            "choice_accuracy": {"accuracy": 0.5},
            "generations": [],
            "behavior_suite": {
                "pass_rate": 0.80 if len(eval_calls) == 1 else 0.70,
                "correct": 8 if len(eval_calls) == 1 else 7,
                "total": 10,
                "items": [],
            },
        }
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text("{}", encoding="utf-8")
        return report

    def fake_gate(baseline_report, candidate_report, output_file=None, **kwargs):
        passed = len(eval_calls) == 1
        gate = {
            "passed": passed,
            "candidate": {
                "param_count": 10,
                "perplexity_mean": 100.0,
                "choice_accuracy": 0.5,
                "generation_health_rate": 1.0,
                "behavior_pass_rate": 0.80 if passed else 0.70,
            },
        }
        if output_file is not None:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_text("{}", encoding="utf-8")
        return gate

    def fake_block(**kwargs):
        block_captured.update(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        return {"operation": "fake_block"}

    monkeypatch.setattr(hf_xfamily, "evaluate_text_model", fake_eval)
    monkeypatch.setattr(hf_xfamily, "gate_text_report", fake_gate)
    monkeypatch.setattr(hf_recovery, "blockwise_reconstruct_cross_tokenizer_causal_lm", fake_block)

    summary = hf_xfamily.chain_anchor_staged_compress(
        "donor",
        "anchor",
        tmp_path,
        stages="block",
        promote_policy="gate-and-rank",
        block_on_policy_anchor_kd_weight=0.04,
        block_on_policy_hidden_kd_weight=0.06,
        block_stepwise_donor_kd_weight=0.05,
        block_teacher_generate_lm_weight=0.011,
        block_focused_lm_weight=0.017,
        block_token_focus="numbers",
        block_projected_vocab_kd_mode="both",
        block_train_output_head_with_kd=True,
        block_untie_output_head_for_kd=True,
        block_module_output_temperature=1.7,
        block_projected_vocab_max_source_tokens=11,
        device="cpu",
    )

    assert summary["best_model_dir"] == "anchor"
    assert summary["promoted_stage_count"] == 0
    assert summary["stage_records"][0]["promoted"] is False
    assert block_captured["on_policy_anchor_kd_weight"] == 0.04
    assert block_captured["on_policy_hidden_kd_weight"] == 0.06
    assert block_captured["stepwise_donor_kd_weight"] == 0.05
    assert block_captured["teacher_generate_lm_weight"] == 0.011
    assert block_captured["focused_lm_weight"] == 0.017
    assert block_captured["token_focus"] == "numbers"
    assert block_captured["projected_vocab_kd_mode"] == "both"
    assert block_captured["train_output_head_with_kd"] is True
    assert block_captured["untie_output_head_for_kd"] is True
    assert block_captured["module_output_temperature"] == 1.7
    assert block_captured["projected_vocab_max_source_tokens"] == 11


def test_chain_anchor_sequential_stage_input_keeps_fold_pipeline_moving(monkeypatch, tmp_path) -> None:
    import neural_forge.hf_recovery as hf_recovery
    import neural_forge.hf_xfamily as hf_xfamily

    init_captured = {}
    block_captured = {}

    def fake_eval(model_id_or_path, output_file, **kwargs):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text("{}", encoding="utf-8")
        return {
            "model": str(model_id_or_path),
            "param_count": 10,
            "perplexity": {"mean": 100.0},
            "choice_accuracy": {"accuracy": 0.5},
            "generations": [],
            "behavior_suite": {"pass_rate": 0.70, "correct": 7, "total": 10, "items": []},
        }

    def fake_gate(baseline_report, candidate_report, output_file=None, **kwargs):
        passed = (
            isinstance(baseline_report, dict)
            and isinstance(candidate_report, dict)
            and baseline_report["model"] == candidate_report["model"] == "anchor"
        )
        gate = {
            "passed": passed,
            "candidate": {
                "param_count": 10,
                "perplexity_mean": 100.0,
                "choice_accuracy": 0.5,
                "generation_health_rate": 1.0,
                "behavior_pass_rate": 0.70,
            },
        }
        if output_file is not None:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_text("{}", encoding="utf-8")
        return gate

    def fake_init(**kwargs):
        init_captured.update(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        return {"operation": "fake_init"}

    def fake_block(**kwargs):
        block_captured.update(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        return {"operation": "fake_block"}

    monkeypatch.setattr(hf_xfamily, "evaluate_text_model", fake_eval)
    monkeypatch.setattr(hf_xfamily, "gate_text_report", fake_gate)
    monkeypatch.setattr(hf_xfamily, "donor_projected_init", fake_init)
    monkeypatch.setattr(hf_recovery, "blockwise_reconstruct_cross_tokenizer_causal_lm", fake_block)

    summary = hf_xfamily.chain_anchor_staged_compress(
        "donor",
        "anchor",
        tmp_path,
        stages="init,block",
        promote_policy="gate-and-rank",
        stage_input_policy="sequential",
        device="cpu",
    )

    assert summary["stage_input_policy"] == "sequential"
    assert summary["best_tag"] == "anchor"
    assert summary["final_tag"] == "02_block"
    assert init_captured["target_base_model"] == "anchor"
    assert block_captured["candidate_model"] == str(tmp_path / "models" / "01_init")
    assert summary["stage_records"][0]["input_tag"] == "anchor"
    assert summary["stage_records"][0]["promoted"] is False
    assert summary["stage_records"][1]["input_tag"] == "01_init"
    assert summary["stage_records"][1]["input_model_dir"] == str(tmp_path / "models" / "01_init")


def test_chain_anchor_keeps_positional_stages_compatibility(monkeypatch, tmp_path) -> None:
    import neural_forge.hf_xfamily as hf_xfamily

    captured = {}

    def fake_eval(model_id_or_path, output_file, **kwargs):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text("{}", encoding="utf-8")
        return {
            "model": str(model_id_or_path),
            "param_count": 10,
            "perplexity": {"mean": 100.0},
            "choice_accuracy": {"accuracy": 0.5},
            "generations": [],
            "behavior_suite": {"pass_rate": 0.8, "correct": 8, "total": 10, "items": []},
        }

    def fake_gate(baseline_report, candidate_report, output_file=None, **kwargs):
        if output_file is not None:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_text("{}", encoding="utf-8")
        return {
            "passed": True,
            "candidate": {
                "param_count": 10,
                "perplexity_mean": 100.0,
                "choice_accuracy": 0.5,
                "generation_health_rate": 1.0,
                "behavior_pass_rate": 0.8,
            },
        }

    def fake_init(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        return {"operation": "fake_init"}

    monkeypatch.setattr(hf_xfamily, "evaluate_text_model", fake_eval)
    monkeypatch.setattr(hf_xfamily, "gate_text_report", fake_gate)
    monkeypatch.setattr(hf_xfamily, "donor_projected_init", fake_init)

    summary = hf_xfamily.chain_anchor_staged_compress("donor", "anchor", tmp_path, "init", device="cpu")

    assert summary["stages"] == ["init"]
    assert captured["donor_model"] == "donor"


def test_chain_anchor_init_stage_dispatches_multi_donor_then_primary_teacher(monkeypatch, tmp_path) -> None:
    import neural_forge.hf_recovery as hf_recovery
    import neural_forge.hf_xfamily as hf_xfamily

    init_captured = {}
    block_captured = {}

    def fake_eval(model_id_or_path, output_file, **kwargs):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text("{}", encoding="utf-8")
        return {
            "model": str(model_id_or_path),
            "param_count": 10,
            "perplexity": {"mean": 100.0},
            "choice_accuracy": {"accuracy": 0.5},
            "generations": [],
            "behavior_suite": {"pass_rate": 0.8, "correct": 8, "total": 10, "items": []},
        }

    def fake_gate(baseline_report, candidate_report, output_file=None, **kwargs):
        if output_file is not None:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_text("{}", encoding="utf-8")
        return {
            "passed": True,
            "candidate": {
                "param_count": 10,
                "perplexity_mean": 100.0,
                "choice_accuracy": 0.5,
                "generation_health_rate": 1.0,
                "behavior_pass_rate": 0.8,
            },
        }

    def fake_multi_init(**kwargs):
        init_captured.update(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        return {"operation": "fake_multi_init"}

    def fake_single_init(**kwargs):
        raise AssertionError("single-donor init should not run for comma-separated sources")

    def fake_block(**kwargs):
        block_captured.update(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        return {"operation": "fake_block"}

    monkeypatch.setattr(hf_xfamily, "evaluate_text_model", fake_eval)
    monkeypatch.setattr(hf_xfamily, "gate_text_report", fake_gate)
    monkeypatch.setattr(hf_xfamily, "multi_donor_projected_init", fake_multi_init)
    monkeypatch.setattr(hf_xfamily, "donor_projected_init", fake_single_init)
    monkeypatch.setattr(hf_recovery, "blockwise_reconstruct_cross_tokenizer_causal_lm", fake_block)

    summary = hf_xfamily.chain_anchor_staged_compress(
        "donor-a,donor-b",
        "anchor",
        tmp_path,
        source_weights="0.25,0.75",
        stages="init,block",
        promote_policy="always",
        stage_input_policy="sequential",
        device="cpu",
    )

    assert init_captured["donor_models"] == ["donor-a", "donor-b"]
    assert init_captured["donor_weights"] == [0.25, 0.75]
    assert init_captured["target_base_model"] == "anchor"
    assert block_captured["source_model"] == "donor-a"
    assert block_captured["candidate_model"] == str(tmp_path / "models" / "01_init")
    assert summary["source_model"] == "donor-a,donor-b"
    assert summary["source_models"] == ["donor-a", "donor-b"]
    assert summary["source_weights"] == [0.25, 0.75]
    assert summary["primary_source_model"] == "donor-a"
    assert summary["final_tag"] == "02_block"


def test_chain_anchor_init_stage_forwards_activation_fit_flags(monkeypatch, tmp_path) -> None:
    import neural_forge.hf_xfamily as hf_xfamily

    captured = {}

    def fake_eval(model_id_or_path, output_file, **kwargs):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text("{}", encoding="utf-8")
        return {
            "model": str(model_id_or_path),
            "param_count": 10,
            "perplexity": {"mean": 100.0},
            "choice_accuracy": {"accuracy": 0.5},
            "generations": [],
            "behavior_suite": {"pass_rate": 0.8, "correct": 8, "total": 10, "items": []},
        }

    def fake_gate(baseline_report, candidate_report, output_file=None, **kwargs):
        if output_file is not None:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_text("{}", encoding="utf-8")
        return {
            "passed": True,
            "candidate": {
                "param_count": 10,
                "perplexity_mean": 100.0,
                "choice_accuracy": 0.5,
                "generation_health_rate": 1.0,
                "behavior_pass_rate": 0.8,
            },
        }

    def fake_init(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        return {"operation": "fake_init"}

    monkeypatch.setattr(hf_xfamily, "evaluate_text_model", fake_eval)
    monkeypatch.setattr(hf_xfamily, "gate_text_report", fake_gate)
    monkeypatch.setattr(hf_xfamily, "donor_projected_init", fake_init)

    summary = hf_xfamily.chain_anchor_staged_compress(
        "donor",
        "anchor",
        tmp_path,
        stages="init",
        calibration_texts=["sample"],
        init_fold_mode="transport",
        init_hidden_basis_mode="activation-pca",
        init_hidden_basis_scope="layer",
        init_hidden_basis_max_rows=4096,
        init_hidden_basis_preprocess="uncentered-rowrms",
        init_activation_fold_assignment="similarity",
        init_transport_backend="sinkhorn",
        init_transport_support="importance",
        init_donor_layer_fold_mode="fold",
        init_donor_layer_group_strategy="activation-energy",
        init_attention_fold_mode="activation-coupled",
        init_attention_head_dim_transport="qk-vo",
        init_token_init_mode="donor-tokenized-weighted",
        init_projection_scale="target-rms",
        init_projection_blend_alpha=0.7,
        init_consumer_hidden_reduction="sqrt-sum",
        init_activation_fit_linears=True,
        init_activation_fit_target="mlp-output",
        init_activation_fit_max_rows=1024,
        init_activation_fit_alpha=0.6,
        init_activation_fit_ridge=0.03,
        init_activation_fit_rank=64,
        init_activation_fit_rank_mode="delta",
        init_activation_fit_method="lrc",
        init_activation_fit_factor_steps=11,
        init_activation_fit_factor_lr=0.04,
        init_activation_fit_factor_l2=0.002,
        init_activation_fit_gate="none",
        init_activation_fit_validation_fraction=0.15,
        init_activation_fit_min_validation_rows=4,
        init_activation_fit_min_relative_improvement=0.02,
        init_activation_fit_max_cosine_drop=0.04,
        init_activation_feature_mode="covariance",
        init_activation_subspace_rank=12,
        init_activation_domain_balance="keyword",
        init_untie_output_head=True,
        init_activation_max_length=160,
        device="cpu",
    )

    assert summary["stage_count"] == 1
    assert captured["target_config_model"] == "anchor"
    assert captured["target_base_model"] == "anchor"
    assert captured["donor_model"] == "donor"
    assert captured["activation_calibration_texts"] == ["sample"]
    assert captured["fold_mode"] == "transport"
    assert captured["hidden_basis_mode"] == "activation-pca"
    assert captured["hidden_basis_scope"] == "layer"
    assert captured["hidden_basis_max_rows"] == 4096
    assert captured["hidden_basis_preprocess"] == "uncentered-rowrms"
    assert captured["activation_fold_assignment"] == "similarity"
    assert captured["transport_backend"] == "sinkhorn"
    assert captured["transport_support"] == "importance"
    assert captured["donor_layer_group_strategy"] == "activation-energy"
    assert captured["attention_fold_mode"] == "activation-coupled"
    assert captured["attention_head_dim_transport"] == "qk-vo"
    assert captured["token_init_mode"] == "donor-tokenized-weighted"
    assert captured["projection_scale"] == "target-rms"
    assert captured["projection_blend_alpha"] == 0.7
    assert captured["consumer_hidden_reduction"] == "sqrt-sum"
    assert captured["activation_fit_linears"] is True
    assert captured["activation_fit_target"] == "mlp-output"
    assert captured["activation_fit_max_rows"] == 1024
    assert captured["activation_fit_alpha"] == 0.6
    assert captured["activation_fit_ridge"] == 0.03
    assert captured["activation_fit_rank"] == 64
    assert captured["activation_fit_rank_mode"] == "delta"
    assert captured["activation_fit_method"] == "lrc"
    assert captured["activation_fit_factor_steps"] == 11
    assert captured["activation_fit_factor_lr"] == 0.04
    assert captured["activation_fit_factor_l2"] == 0.002
    assert captured["activation_fit_gate"] == "none"
    assert captured["activation_fit_validation_fraction"] == 0.15
    assert captured["activation_fit_min_validation_rows"] == 4
    assert captured["activation_fit_min_relative_improvement"] == 0.02
    assert captured["activation_fit_max_cosine_drop"] == 0.04
    assert captured["activation_feature_mode"] == "covariance"
    assert captured["activation_subspace_rank"] == 12
    assert captured["activation_domain_balance"] == "keyword"
    assert captured["untie_output_head"] is True
    assert captured["activation_max_length"] == 160


def test_hf_xfamily_chain_anchor_handler_passes_flags(monkeypatch) -> None:
    import neural_forge.cli as cli
    import neural_forge.hf_xfamily as hf_xfamily

    captured = {}

    def fake_chain(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_xfamily, "chain_anchor_staged_compress", fake_chain)
    monkeypatch.setattr(cli, "_load_recovery_calibration_texts", lambda path, max_texts: ["chain text"])
    args = build_parser().parse_args(
        [
            "hf-xfamily-chain-anchor-compress",
            "donor-a,donor-b",
            "anchor",
            "out",
            "--donor-weights",
            "0.25,0.75",
            "--stages",
            "block,alm",
            "--eval-suite",
            "suite.json",
            "--max-behavior-probes",
            "200",
            "--max-generation-prompts",
            "80",
            "--stage-max-behavior-drop",
            "0.02",
            "--promote-policy",
            "gate-only",
            "--stage-input-policy",
            "sequential",
            "--init-fold-mode",
            "transport",
            "--init-hidden-basis-mode",
            "activation-pca",
            "--init-hidden-basis-scope",
            "layer",
            "--init-hidden-basis-max-rows",
            "4096",
            "--init-hidden-basis-preprocess",
            "uncentered-rowrms",
            "--init-activation-fold-assignment",
            "similarity",
            "--init-transport-backend",
            "sinkhorn",
            "--init-transport-support",
            "importance",
            "--init-donor-layer-group-strategy",
            "activation-geometry",
            "--init-attention-fold-mode",
            "activation-coupled",
            "--init-attention-head-dim-transport",
            "qk-vo",
            "--init-token-init-mode",
            "donor-tokenized-weighted",
            "--init-projection-scale",
            "geomean-rms",
            "--init-projection-blend-alpha",
            "0.75",
            "--init-consumer-hidden-reduction",
            "sqrt-sum",
            "--init-activation-fit-linears",
            "--init-activation-fit-target",
            "mlp-output",
            "--init-activation-fit-max-rows",
            "1024",
            "--init-activation-fit-alpha",
            "0.6",
            "--init-activation-fit-ridge",
            "0.03",
            "--init-activation-fit-rank",
            "64",
            "--init-activation-fit-rank-mode",
            "delta",
            "--init-activation-fit-method",
            "lrc",
            "--init-activation-fit-factor-steps",
            "17",
            "--init-activation-fit-factor-lr",
            "0.05",
            "--init-activation-fit-factor-l2",
            "0.003",
            "--init-activation-fit-gate",
            "none",
            "--init-activation-fit-validation-fraction",
            "0.15",
            "--init-activation-fit-min-validation-rows",
            "4",
            "--init-activation-fit-min-relative-improvement",
            "0.02",
            "--init-activation-fit-max-cosine-drop",
            "0.04",
            "--init-activation-feature-mode",
            "covariance",
            "--init-activation-subspace-rank",
            "12",
            "--init-activation-domain-balance",
            "keyword",
            "--init-untie-output-head",
            "--init-activation-max-length",
            "160",
            "--block-steps-per-layer",
            "6",
            "--block-hidden-metric",
            "projected-geometry",
            "--block-residual-magnitude-compensation",
            "lowrank-pre",
            "--block-residual-compensation-min-scale",
            "0.35",
            "--block-residual-compensation-max-scale",
            "1.65",
            "--block-residual-compensation-linear-ridge",
            "0.007",
            "--block-residual-compensation-alpha",
            "0.005",
            "--block-pre-reconstruction-target",
            "block-interface",
            "--block-pre-reconstruction-ridge",
            "0.008",
            "--block-donor-layer-fold-mode",
            "average",
            "--block-pre-reconstruction-objective",
            "cosine",
            "--block-pre-reconstruction-cosine-steps",
            "6",
            "--block-pre-reconstruction-cosine-lr",
            "0.03",
            "--block-pre-reconstruction-cosine-l2",
            "0.0007",
            "--block-pre-reconstruction-fit-gate",
            "none",
            "--block-pre-reconstruction-validation-fraction",
            "0.2",
            "--block-pre-reconstruction-min-validation-rows",
            "10",
            "--block-pre-reconstruction-max-tokens-per-layer",
            "48",
            "--block-pre-reconstruction-min-relative-improvement",
            "0.03",
            "--block-pre-reconstruction-max-cosine-drop",
            "0.05",
            "--block-donor-layer-group-strategy",
            "activation-geometry",
            "--block-on-policy-anchor-kd-weight",
            "0.04",
            "--block-on-policy-unlikelihood-weight",
            "0.005",
            "--block-on-policy-hidden-kd-weight",
            "0.06",
            "--block-on-policy-max-new-tokens",
            "19",
            "--block-on-policy-interval",
            "3",
            "--block-on-policy-response-loss-mode",
            "answer",
            "--block-on-policy-unlikelihood-window",
            "7",
            "--block-stepwise-donor-kd-weight",
            "0.05",
            "--block-stepwise-donor-max-new-tokens",
            "6",
            "--block-stepwise-donor-interval",
            "2",
            "--block-teacher-generate-lm-weight",
            "0.011",
            "--block-teacher-generate-hidden-kd-weight",
            "0.012",
            "--block-teacher-generate-projected-vocab-kd-weight",
            "0.013",
            "--block-teacher-generate-byte-kd-weight",
            "0.014",
            "--block-teacher-generate-max-new-tokens",
            "21",
            "--block-alm-max-chunks",
            "22",
            "--block-alm-min-chars",
            "3",
            "--block-focused-block-hidden-kd-weight",
            "0.015",
            "--block-focused-projected-vocab-kd-weight",
            "0.016",
            "--block-focused-lm-weight",
            "0.017",
            "--block-token-focus",
            "numbers",
            "--block-projected-vocab-kd-mode",
            "both",
            "--block-byte-hidden-kd-weight",
            "0.018",
            "--block-block-io-ridge",
            "0.019",
            "--block-block-io-loss",
            "cosine",
            "--block-train-output-head-with-kd",
            "--block-untie-output-head-for-kd",
            "--block-module-output-kd-weight",
            "0.03",
            "--block-module-output-temperature",
            "1.7",
            "--block-module-output-sinkhorn-iterations",
            "9",
            "--block-projected-vocab-max-source-tokens",
            "11",
            "--block-save-best-layer-behavior",
            "--block-calibration-order",
            "domain-balanced",
            "--alm-steps",
            "40",
            "--alm-first-token-lm-weight",
            "0.021",
            "--alm-response-end-lm-weight",
            "0.022",
            "--alm-focused-lm-weight",
            "0.023",
            "--alm-attention-influence-weight",
            "0.026",
            "--alm-attention-influence-layer",
            "middle",
            "--alm-kd-weight",
            "0.07",
            "--alm-focused-projected-vocab-kd-weight",
            "0.024",
            "--alm-token-kd-weight",
            "0.025",
            "--alm-token-focus",
            "numbers",
            "--alm-token-max-pairs",
            "66",
            "--alm-token-teacher-confidence-floor",
            "0.27",
            "--alm-answer-contrastive-weight",
            "0.028",
            "--alm-answer-contrastive-max-negatives",
            "7",
            "--alm-answer-first-token-margin-weight",
            "0.038",
            "--alm-answer-first-token-margin",
            "1.6",
            "--alm-answer-first-token-max-negatives",
            "9",
            "--alm-teacher-rank-weight",
            "0.029",
            "--alm-teacher-rank-temperature",
            "0.7",
            "--alm-teacher-rank-max-negatives",
            "8",
            "--alm-teacher-generate-lm-weight",
            "0.031",
            "--alm-teacher-generate-max-new-tokens",
            "33",
            "--alm-stepwise-donor-kd-weight",
            "0.032",
            "--alm-stepwise-donor-max-new-tokens",
            "12",
            "--alm-on-policy-anchor-kd-weight",
            "0.033",
            "--alm-on-policy-unlikelihood-weight",
            "0.034",
            "--alm-on-policy-hidden-kd-weight",
            "0.035",
            "--alm-on-policy-max-new-tokens",
            "18",
            "--alm-on-policy-interval",
            "5",
            "--alm-save-best-behavior",
            "--alm-calibration-order",
            "domain-balanced",
            "--dtype",
            "bfloat16",
            "--candidate-dtype",
            "float32",
            "--device",
            "cpu",
            "--calibration-file",
            "calib.jsonl",
            "--max-calibration-texts",
            "5",
        ]
    )
    result = args.func(args)

    assert result == {"ok": True}
    assert captured["source_model"] == "donor-a,donor-b"
    assert captured["source_weights"] == "0.25,0.75"
    assert captured["anchor_model"] == "anchor"
    assert captured["stages"] == "block,alm"
    assert captured["calibration_texts"] == ["chain text"]
    assert captured["eval_suite"] == "suite.json"
    assert captured["max_behavior_probes"] == 200
    assert captured["max_generation_prompts"] == 80
    assert captured["stage_max_behavior_drop"] == 0.02
    assert captured["promote_policy"] == "gate-only"
    assert captured["stage_input_policy"] == "sequential"
    assert captured["init_fold_mode"] == "transport"
    assert captured["init_hidden_basis_mode"] == "activation-pca"
    assert captured["init_hidden_basis_scope"] == "layer"
    assert captured["init_hidden_basis_max_rows"] == 4096
    assert captured["init_hidden_basis_preprocess"] == "uncentered-rowrms"
    assert captured["init_activation_fold_assignment"] == "similarity"
    assert captured["init_transport_backend"] == "sinkhorn"
    assert captured["init_transport_support"] == "importance"
    assert captured["init_donor_layer_group_strategy"] == "activation-geometry"
    assert captured["init_attention_fold_mode"] == "activation-coupled"
    assert captured["init_attention_head_dim_transport"] == "qk-vo"
    assert captured["init_token_init_mode"] == "donor-tokenized-weighted"
    assert captured["init_projection_scale"] == "geomean-rms"
    assert captured["init_projection_blend_alpha"] == 0.75
    assert captured["init_consumer_hidden_reduction"] == "sqrt-sum"
    assert captured["init_activation_fit_linears"] is True
    assert captured["init_activation_fit_target"] == "mlp-output"
    assert captured["init_activation_fit_max_rows"] == 1024
    assert captured["init_activation_fit_alpha"] == 0.6
    assert captured["init_activation_fit_ridge"] == 0.03
    assert captured["init_activation_fit_rank"] == 64
    assert captured["init_activation_fit_rank_mode"] == "delta"
    assert captured["init_activation_fit_method"] == "lrc"
    assert captured["init_activation_fit_factor_steps"] == 17
    assert captured["init_activation_fit_factor_lr"] == 0.05
    assert captured["init_activation_fit_factor_l2"] == 0.003
    assert captured["init_activation_fit_gate"] == "none"
    assert captured["init_activation_fit_validation_fraction"] == 0.15
    assert captured["init_activation_fit_min_validation_rows"] == 4
    assert captured["init_activation_fit_min_relative_improvement"] == 0.02
    assert captured["init_activation_fit_max_cosine_drop"] == 0.04
    assert captured["init_activation_feature_mode"] == "covariance"
    assert captured["init_activation_subspace_rank"] == 12
    assert captured["init_activation_domain_balance"] == "keyword"
    assert captured["init_untie_output_head"] is True
    assert captured["init_activation_max_length"] == 160
    assert captured["block_steps_per_layer"] == 6
    assert captured["block_hidden_metric"] == "projected-geometry"
    assert captured["block_residual_magnitude_compensation"] == "lowrank-pre"
    assert captured["block_residual_compensation_min_scale"] == 0.35
    assert captured["block_residual_compensation_max_scale"] == 1.65
    assert captured["block_residual_compensation_linear_ridge"] == 0.007
    assert captured["block_residual_compensation_alpha"] == 0.005
    assert captured["block_pre_reconstruction_target"] == "block-interface"
    assert captured["block_pre_reconstruction_ridge"] == 0.008
    assert captured["block_pre_reconstruction_objective"] == "cosine"
    assert captured["block_pre_reconstruction_cosine_steps"] == 6
    assert captured["block_pre_reconstruction_cosine_lr"] == 0.03
    assert captured["block_pre_reconstruction_cosine_l2"] == 0.0007
    assert captured["block_pre_reconstruction_fit_gate"] == "none"
    assert captured["block_pre_reconstruction_validation_fraction"] == 0.2
    assert captured["block_pre_reconstruction_min_validation_rows"] == 10
    assert captured["block_pre_reconstruction_max_tokens_per_layer"] == 48
    assert captured["block_pre_reconstruction_min_relative_improvement"] == 0.03
    assert captured["block_pre_reconstruction_max_cosine_drop"] == 0.05
    assert captured["block_donor_layer_fold_mode"] == "average"
    assert captured["block_donor_layer_group_strategy"] == "activation-geometry"
    assert captured["block_on_policy_anchor_kd_weight"] == 0.04
    assert captured["block_on_policy_unlikelihood_weight"] == 0.005
    assert captured["block_on_policy_hidden_kd_weight"] == 0.06
    assert captured["block_on_policy_max_new_tokens"] == 19
    assert captured["block_on_policy_interval"] == 3
    assert captured["block_on_policy_response_loss_mode"] == "answer"
    assert captured["block_on_policy_unlikelihood_window"] == 7
    assert captured["block_stepwise_donor_kd_weight"] == 0.05
    assert captured["block_stepwise_donor_max_new_tokens"] == 6
    assert captured["block_stepwise_donor_interval"] == 2
    assert captured["block_teacher_generate_lm_weight"] == 0.011
    assert captured["block_teacher_generate_hidden_kd_weight"] == 0.012
    assert captured["block_teacher_generate_projected_vocab_kd_weight"] == 0.013
    assert captured["block_teacher_generate_byte_kd_weight"] == 0.014
    assert captured["block_teacher_generate_max_new_tokens"] == 21
    assert captured["block_alm_max_chunks"] == 22
    assert captured["block_alm_min_chars"] == 3
    assert captured["block_focused_block_hidden_kd_weight"] == 0.015
    assert captured["block_focused_projected_vocab_kd_weight"] == 0.016
    assert captured["block_focused_lm_weight"] == 0.017
    assert captured["block_token_focus"] == "numbers"
    assert captured["block_projected_vocab_kd_mode"] == "both"
    assert captured["block_byte_hidden_kd_weight"] == 0.018
    assert captured["block_block_io_ridge"] == 0.019
    assert captured["block_block_io_loss"] == "cosine"
    assert captured["block_train_output_head_with_kd"] is True
    assert captured["block_untie_output_head_for_kd"] is True
    assert captured["block_module_output_kd_weight"] == 0.03
    assert captured["block_module_output_temperature"] == 1.7
    assert captured["block_module_output_sinkhorn_iterations"] == 9
    assert captured["block_projected_vocab_max_source_tokens"] == 11
    assert captured["block_save_best_layer_behavior"] is True
    assert captured["block_calibration_order"] == "domain-balanced"
    assert captured["alm_steps"] == 40
    assert captured["alm_first_token_lm_weight"] == 0.021
    assert captured["alm_response_end_lm_weight"] == 0.022
    assert captured["alm_focused_lm_weight"] == 0.023
    assert captured["alm_attention_influence_weight"] == 0.026
    assert captured["alm_attention_influence_layer"] == "middle"
    assert captured["alm_kd_weight"] == 0.07
    assert captured["alm_focused_projected_vocab_kd_weight"] == 0.024
    assert captured["alm_token_kd_weight"] == 0.025
    assert captured["alm_token_focus"] == "numbers"
    assert captured["alm_token_max_pairs"] == 66
    assert captured["alm_token_teacher_confidence_floor"] == 0.27
    assert captured["alm_answer_contrastive_weight"] == 0.028
    assert captured["alm_answer_contrastive_max_negatives"] == 7
    assert captured["alm_answer_first_token_margin_weight"] == 0.038
    assert captured["alm_answer_first_token_margin"] == 1.6
    assert captured["alm_answer_first_token_max_negatives"] == 9
    assert captured["alm_teacher_rank_weight"] == 0.029
    assert captured["alm_teacher_rank_temperature"] == 0.7
    assert captured["alm_teacher_rank_max_negatives"] == 8
    assert captured["alm_teacher_generate_lm_weight"] == 0.031
    assert captured["alm_teacher_generate_max_new_tokens"] == 33
    assert captured["alm_stepwise_donor_kd_weight"] == 0.032
    assert captured["alm_stepwise_donor_max_new_tokens"] == 12
    assert captured["alm_on_policy_anchor_kd_weight"] == 0.033
    assert captured["alm_on_policy_unlikelihood_weight"] == 0.034
    assert captured["alm_on_policy_hidden_kd_weight"] == 0.035
    assert captured["alm_on_policy_max_new_tokens"] == 18
    assert captured["alm_on_policy_interval"] == 5
    assert captured["alm_save_best_behavior"] is True
    assert captured["alm_calibration_order"] == "domain-balanced"
    assert captured["torch_dtype"] == "bfloat16"
    assert captured["candidate_torch_dtype"] == "float32"
    assert captured["device"] == "cpu"


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
            "--eval-suite",
            "mlp-suite.json",
            "--max-text-evals",
            "11",
            "--max-choice-probes",
            "22",
            "--max-generation-prompts",
            "33",
            "--max-behavior-probes",
            "1000",
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
    assert captured["eval_suite"] == "mlp-suite.json"
    assert captured["max_text_evals"] == 11
    assert captured["max_choice_probes"] == 22
    assert captured["max_generation_prompts"] == 33
    assert captured["max_behavior_probes"] == 1000


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
            "--eval-suite",
            "compress-suite.json",
            "--max-text-evals",
            "12",
            "--max-choice-probes",
            "23",
            "--max-generation-prompts",
            "34",
            "--max-behavior-probes",
            "1000",
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
    assert captured["eval_suite"] == "compress-suite.json"
    assert captured["max_text_evals"] == 12
    assert captured["max_choice_probes"] == 23
    assert captured["max_generation_prompts"] == 34
    assert captured["max_behavior_probes"] == 1000


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
            "host_metrics": {
                "perplexity_mean": 300.0,
                "choice_accuracy": 0.75,
                "behavior_pass_rate": 0.50,
                "generation_health_rate": 1.0,
            },
        },
        {
            "tag": "raw_low_behavior",
            "model_dir": "raw_low_behavior",
            "report": "raw_low_behavior.json",
            "host_gate_report": "raw_low_behavior_gate.json",
            "host_passed": True,
            "shrink": 0.05,
            "host_metrics": {
                "perplexity_mean": 150.0,
                "choice_accuracy": 0.75,
                "behavior_pass_rate": 0.25,
                "generation_health_rate": 1.0,
            },
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
            "source_candidate": compressed[2],
            "model_dir": "rec",
            "report": "rec.json",
            "gate_report": "rec_gate.json",
            "metrics": {
                "perplexity_mean": 240.0,
                "choice_accuracy": 0.75,
                "behavior_pass_rate": 0.50,
                "generation_health_rate": 1.0,
            },
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
    assert [item["tag"] for item in usable] == ["raw_fail_recovered", "raw_pass", "raw_low_behavior"]
