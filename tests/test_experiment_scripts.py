from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_beststack_queue_defaults_to_untied_block_output_head_kd() -> None:
    script = (REPO_ROOT / "experiments/queue_beststack_after_retest_20260706.sh").read_text(
        encoding="utf-8"
    )

    assert 'BLOCK_TRAIN_OUTPUT_HEAD_WITH_KD="${BLOCK_TRAIN_OUTPUT_HEAD_WITH_KD:-1}"' in script
    assert 'BLOCK_UNTIE_OUTPUT_HEAD_FOR_KD="${BLOCK_UNTIE_OUTPUT_HEAD_FOR_KD:-1}"' in script
    assert 'BLOCK_ALLOW_TIED_OUTPUT_HEAD_TRAINING="${BLOCK_ALLOW_TIED_OUTPUT_HEAD_TRAINING:-0}"' in script
    assert 'PRE_RECONSTRUCTION_MIN_FITTED_COSINE="${PRE_RECONSTRUCTION_MIN_FITTED_COSINE:-0.05}"' in script


def test_chain_launchers_pass_block_output_head_flags() -> None:
    for relative_path in (
        "experiments/launch_qwen4b_blank_smollm_true_lrc_chain.sh",
        "experiments/launch_lfm230m_blank_smollm_chain.sh",
    ):
        script = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "--block-train-output-head-with-kd" in script
        assert "--block-untie-output-head-for-kd" in script
        assert "--block-allow-tied-output-head-training" in script
        assert "--block-pre-reconstruction-min-fitted-cosine" in script


def test_blank_beststack_topneg_recovery_uses_folded_child_and_broad_eval() -> None:
    script = (
        REPO_ROOT / "experiments/queue_qwen4b_blank_beststack_topneg_recover_20260707.sh"
    ).read_text(encoding="utf-8")

    assert "qwen4b_blank_smollm_beststack_qrsinkhorn_module_eora_qmedoid_gpu4_20260706/models/02_block" in script
    assert 'TIE_CANDIDATE_BEFORE_RECOVERY="${TIE_CANDIDATE_BEFORE_RECOVERY:-1}"' in script
    assert "hf-tie-head" in script
    assert 'RECOVERY_INPUT="$TIED_CANDIDATE_MODEL"' in script
    assert 'FINAL_OUTPUT_HEAD_POLICY="${FINAL_OUTPUT_HEAD_POLICY:-tie}"' in script
    assert '--final-output-head-policy "$FINAL_OUTPUT_HEAD_POLICY"' in script
    assert 'data.get("final_model_dir") or data.get("best_model_dir")' in script
    assert 'EVAL_SUITE_FILE="$EVAL_SUITE_FILE"' in script
    assert "--alm-answer-first-token-margin-weight" in script
    assert "--alm-answer-first-token-top-negatives 24" in script
    assert "--alm-answer-contrastive-weight" in script
    assert "--alm-save-best-behavior" in script
    assert "eval_model_broad_and_manual.sh" in script
    assert "MAX_BEHAVIOR_PROBES=1000" in script
    assert "MAX_GENERATION_PROMPTS=1000" in script
    assert "MAX_CHOICE_PROBES=250" in script


def test_broad_manual_eval_allows_exactgen_suite_override() -> None:
    script = (REPO_ROOT / "experiments/eval_model_broad_and_manual.sh").read_text(
        encoding="utf-8"
    )

    assert 'EVAL_SUITE_FILE="${EVAL_SUITE_FILE:-$RUN_ROOT/behavior_suite.json}"' in script
    assert '--eval-suite "$EVAL_SUITE_FILE"' in script


def test_lfm_stabilizer_prefers_exact_size_chain_output() -> None:
    script = (
        REPO_ROOT / "experiments/launch_lfm230m_stabilizer_after_summary.sh"
    ).read_text(encoding="utf-8")

    assert 'r.get("final_model_dir") or r["best_model_dir"]' in script


def test_prior_anchor_topneg_queue_restores_old_best_paths() -> None:
    script = (
        REPO_ROOT / "experiments/queue_qwen4b_prior_anchor_topneg_recover_20260707.sh"
    ).read_text(encoding="utf-8")

    assert "dualalm29)" in script
    assert "tied29x_dualalm240_last4" in script
    assert "FINAL_OUTPUT_HEAD_POLICY=\"${FINAL_OUTPUT_HEAD_POLICY:-tie}\"" in script
    assert "ffn1496)" in script
    assert "alpha05_ffn1496_folded_xreconstruct_mlp_a05" in script
    assert "FINAL_OUTPUT_HEAD_POLICY=\"${FINAL_OUTPUT_HEAD_POLICY:-preserve}\"" in script
    assert "behavior_suite_exactgen_20260702.json" in script
    assert "queue_qwen4b_blank_beststack_topneg_recover_20260707.sh" in script
