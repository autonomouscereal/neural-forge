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
    assert '"$INIT_TRANSPORT_SUPPORT" == "domain-diverse-medoid"' in script
    assert 'INIT_ATTENTION_HEAD_DIM_TRANSPORT="qk-vo"' in script
    assert 'ACT_TARGET="all"' in script
    assert 'GPU_LOCK_DIR="${GPU_LOCK_DIR:-$RUN_ROOT/locks}"' in script
    assert "waiting_for_gpu_launch_lock" in script
    assert "gpu_launch_process_visible" in script
    assert 'flock 9' in script


def test_domain_diverse_beststack_wrappers_only_change_transport_support() -> None:
    lfm = (
        REPO_ROOT / "experiments/queue_qwen4b_lfm230m_multidonor_beststack_domain_diverse_20260707.sh"
    ).read_text(encoding="utf-8")
    gpt2 = (
        REPO_ROOT / "experiments/queue_qwen4b_gpt2_multidonor_beststack_domain_diverse_20260707.sh"
    ).read_text(encoding="utf-8")

    assert 'TAG="${TAG:-qwen4b_lfm230m_multidonor_beststack_seqall_domain_diverse_gpu0_20260707}"' in lfm
    assert 'SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base,LiquidAI/LFM2.5-230M}"' in lfm
    assert 'SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-0.70,0.30}"' in lfm
    assert 'TAG="${TAG:-qwen4b_gpt2_multidonor_beststack_seqall_domain_diverse_gpu4_20260707}"' in gpt2
    assert 'SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base,openai-community/gpt2}"' in gpt2
    assert 'SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-0.85,0.15}"' in gpt2
    for script in (lfm, gpt2):
        assert 'INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-qr-sinkhorn}"' in script
        assert 'TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-domain-diverse-medoid}"' in script
        assert 'RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-sequential-all}"' in script
        assert "queue_beststack_after_retest_20260706.sh" in script


def test_countsketch_beststack_wrappers_use_signed_hash_assignment() -> None:
    lfm = (
        REPO_ROOT / "experiments/queue_qwen4b_lfm230m_multidonor_beststack_countsketch_20260708.sh"
    ).read_text(encoding="utf-8")
    gpt2 = (
        REPO_ROOT / "experiments/queue_qwen4b_gpt2_multidonor_beststack_countsketch_20260708.sh"
    ).read_text(encoding="utf-8")

    assert 'TAG="${TAG:-qwen4b_lfm230m_multidonor_beststack_seqall_countsketch_gpu0_20260708}"' in lfm
    assert 'SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base,LiquidAI/LFM2.5-230M}"' in lfm
    assert 'TAG="${TAG:-qwen4b_gpt2_multidonor_beststack_seqall_countsketch_gpu4_20260708}"' in gpt2
    assert 'SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base,openai-community/gpt2}"' in gpt2
    for script in (lfm, gpt2):
        assert 'INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-qr-sinkhorn}"' in script
        assert 'TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-centroid}"' in script
        assert 'INIT_ACTIVATION_FOLD_ASSIGNMENT="${INIT_ACTIVATION_FOLD_ASSIGNMENT:-countsketch}"' in script
        assert 'RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-sequential-all}"' in script
        assert "queue_beststack_after_retest_20260706.sh" in script


def test_superpose_beststack_wrappers_use_pure_signed_superposition_fold() -> None:
    lfm = (
        REPO_ROOT / "experiments/queue_qwen4b_lfm230m_multidonor_beststack_superpose_20260708.sh"
    ).read_text(encoding="utf-8")
    gpt2 = (
        REPO_ROOT / "experiments/queue_qwen4b_gpt2_multidonor_beststack_superpose_20260708.sh"
    ).read_text(encoding="utf-8")

    assert 'TAG="${TAG:-qwen4b_lfm230m_multidonor_beststack_seqall_superpose_gpu6_20260708}"' in lfm
    assert 'TAG="${TAG:-qwen4b_gpt2_multidonor_beststack_seqall_superpose_gpu4_20260708}"' in gpt2
    for script in (lfm, gpt2):
        assert 'INIT_FOLD_MODE="${INIT_FOLD_MODE:-mean}"' in script
        assert 'INIT_HIDDEN_BASIS_MODE="${INIT_HIDDEN_BASIS_MODE:-none}"' in script
        assert 'INIT_ACTIVATION_FOLD_ASSIGNMENT="${INIT_ACTIVATION_FOLD_ASSIGNMENT:-signed-superpose}"' in script
        assert 'RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-sequential-all}"' in script
        assert "queue_beststack_after_retest_20260706.sh" in script


def test_pca_extrema_beststack_wrappers_use_signed_extreme_transport() -> None:
    lfm = (
        REPO_ROOT / "experiments/queue_qwen4b_lfm230m_multidonor_beststack_pca_extrema_20260708.sh"
    ).read_text(encoding="utf-8")
    gpt2 = (
        REPO_ROOT / "experiments/queue_qwen4b_gpt2_multidonor_beststack_pca_extrema_20260708.sh"
    ).read_text(encoding="utf-8")

    assert 'TAG="${TAG:-qwen4b_lfm230m_multidonor_beststack_seqall_pca_extrema_gpu6_20260708}"' in lfm
    assert 'TAG="${TAG:-qwen4b_gpt2_multidonor_beststack_seqall_pca_extrema_gpu4_20260708}"' in gpt2
    for script in (lfm, gpt2):
        assert 'TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-pca-extrema-medoid}"' in script
        assert 'INIT_ACTIVATION_FOLD_ASSIGNMENT="${INIT_ACTIVATION_FOLD_ASSIGNMENT:-signed-superpose}"' in script
        assert 'INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-qr-sinkhorn}"' in script
        assert 'ACT_TARGET="${ACT_TARGET:-all}"' in script
        assert "queue_beststack_after_retest_20260706.sh" in script


def test_operator_product_beststack_wrappers_combine_path_weights_and_extreme_superpose() -> None:
    lfm = (
        REPO_ROOT / "experiments/queue_qwen4b_lfm230m_multidonor_beststack_operator_product_20260708.sh"
    ).read_text(encoding="utf-8")
    gpt2 = (
        REPO_ROOT / "experiments/queue_qwen4b_gpt2_multidonor_beststack_operator_product_20260708.sh"
    ).read_text(encoding="utf-8")

    assert 'TAG="${TAG:-qwen4b_lfm230m_multidonor_beststack_seqall_operator_product_gpu6_20260708}"' in lfm
    assert 'TAG="${TAG:-qwen4b_gpt2_multidonor_beststack_seqall_operator_product_gpu4_20260708}"' in gpt2
    for script in (lfm, gpt2):
        assert 'INIT_OPERATOR_PRODUCT_AXIS_WEIGHTS="${INIT_OPERATOR_PRODUCT_AXIS_WEIGHTS:-1}"' in script
        assert 'INIT_ACTIVATION_FOLD_ASSIGNMENT="${INIT_ACTIVATION_FOLD_ASSIGNMENT:-signed-superpose}"' in script
        assert 'TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-pca-extrema-medoid}"' in script
        assert 'INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-qr-sinkhorn}"' in script
        assert "queue_beststack_after_retest_20260706.sh" in script


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
        assert '--init-fold-mode "$INIT_FOLD_MODE"' in script
        assert '--init-hidden-basis-mode "$INIT_HIDDEN_BASIS_MODE"' in script
        assert '--init-operator-product-axis-weights' in script


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
    assert 'RUN_RETENTION_COMPARE="${RUN_RETENTION_COMPARE:-1}"' in script
    assert 'FINAL_EVAL_REPORT="${FINAL_EVAL_REPORT:-$EVAL_DIR/eval_${TAG}_exactgen_20260702_1000x1000x250.json}"' in script
    assert 'FINAL_MANUAL_JSON="${FINAL_MANUAL_JSON:-$EVAL_DIR/manual_${TAG}_exactgen_20260702.json}"' in script
    assert 'REPORT="$FINAL_EVAL_REPORT"' in script
    assert 'MANUAL_JSON="$FINAL_MANUAL_JSON"' in script
    assert "queue_eval_retention_compare.sh" in script
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
    assert 'FORCE_EVAL="${FORCE_EVAL:-0}"' in script
    assert 'FORCE_MANUAL="${FORCE_MANUAL:-0}"' in script
    assert "valid_eval_report()" in script
    assert "broad_eval_cache_hit=" in script
    assert "manual_eval_cache_hit=" in script
    assert 'BROAD_JSON="${BROAD_JSON:-$OUT_DIR/eval_${TAG}_${MAX_BEHAVIOR_PROBES}x${MAX_GENERATION_PROMPTS}x${MAX_CHOICE_PROBES}.json}"' in script
    assert 'MANUAL_JSON="${MANUAL_JSON:-$OUT_DIR/manual_${TAG}.json}"' in script
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

    assert 'RUNS_ROOT="${RUNS_ROOT:-$(dirname "$RUN_ROOT")}"' in script
    assert "first_existing_dir()" in script
    assert "dualalm29)" in script
    assert "tied29x_dualalm240_last4" in script
    assert "/home/cereal/model-surgery-lab/runs/qwen4b_to_smollm135m_dualspace_alm_20260627" in script
    assert "FINAL_OUTPUT_HEAD_POLICY=\"${FINAL_OUTPUT_HEAD_POLICY:-tie}\"" in script
    assert "ffn1496)" in script
    assert "alpha05_ffn1496_folded_xreconstruct_mlp_a05" in script
    assert "/home/cereal/model-surgery-lab/runs/qwen4b_to_smollm135m_strict25_ffnfold_20260627_191250" in script
    assert "FINAL_OUTPUT_HEAD_POLICY=\"${FINAL_OUTPUT_HEAD_POLICY:-preserve}\"" in script
    assert "behavior_suite_exactgen_20260702.json" in script
    assert "queue_qwen4b_blank_beststack_topneg_recover_20260707.sh" in script


def test_prior_anchor_pair_queue_orders_dualalm_before_ffn1496() -> None:
    script = (
        REPO_ROOT / "experiments/queue_qwen4b_prior_anchor_pair_20260707.sh"
    ).read_text(encoding="utf-8")

    assert 'EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/evals}"' in script
    assert 'EVAL_DIR="$EVAL_DIR"' in script
    assert 'QUEUE_DUALALM29="${QUEUE_DUALALM29:-1}"' in script
    assert 'QUEUE_FFN1496="${QUEUE_FFN1496:-1}"' in script
    assert "launch_prior_anchor_variant \"dualalm29\"" in script
    assert 'DUAL_PID="$(cat "$RUN_ROOT/logs/${DUALALM29_TAG}.queue.pid")"' in script
    assert 'CURRENT_WAIT_PIDS="$(append_pid "$CURRENT_WAIT_PIDS" "$DUAL_PID")"' in script
    assert "launch_prior_anchor_variant \"ffn1496\"" in script
    assert "queue_qwen4b_prior_anchor_topneg_recover_20260707.sh" in script


def test_combined_anchor_beststack_matrix_keeps_old_and_cross_family_lanes() -> None:
    script = (
        REPO_ROOT / "experiments/queue_combined_anchor_beststack_matrix_20260707.sh"
    ).read_text(encoding="utf-8")

    assert 'MATRIX_EVAL_DIR="${MATRIX_EVAL_DIR:-$RUN_ROOT/evals/combined_anchor_beststack_20260707}"' in script
    assert 'DUALALM29_TAG="${DUALALM29_TAG:-qwen4b_dualalm29_anchor_topneg_exact_gpu6_20260707}"' in script
    assert 'FFN1496_TAG="${FFN1496_TAG:-qwen4b_ffn1496_strict_anchor_topneg_gpu6_20260707}"' in script
    assert 'QWEN_LFM_TAG="${QWEN_LFM_TAG:-qwen4b_lfm230m_multidonor_beststack_seqall_qrsinkhorn_gpu0_20260707}"' in script
    assert 'QWEN_GPT2_TAG="${QWEN_GPT2_TAG:-qwen4b_gpt2_multidonor_beststack_seqall_qrsinkhorn_gpu4_20260707}"' in script
    assert "queue_qwen4b_prior_anchor_pair_20260707.sh" in script
    assert "queue_qwen4b_lfm230m_multidonor_beststack_20260707.sh" in script
    assert "queue_beststack_after_retest_20260706.sh" in script
    assert "VARIANT=qwen_gpt2" in script
    assert 'BESTSTACK_EVAL_DIR="$MATRIX_EVAL_DIR"' in script
    assert 'EVAL_DIR="$MATRIX_EVAL_DIR"' in script
    assert 'WAIT_PIDS="${QWEN_LFM_WAIT_PIDS:-0}"' in script
    assert 'WAIT_PIDS="${QWEN_GPT2_WAIT_PIDS:-0}"' in script
    assert 'f"eval_{tag}_exactgen_20260702_1000x1000x250.json"' in script
    assert 'f"eval_{tag}_strict_20260706_1000x1000x250.json"' in script
    assert "retest_matrix.py\" summarize" in script
