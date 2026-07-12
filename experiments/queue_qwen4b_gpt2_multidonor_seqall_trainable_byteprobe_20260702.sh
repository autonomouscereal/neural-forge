#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TAG="${TAG:-qwen4b_gpt2_multidonor_seqall_trainable_byteprobe_full_gpu4_20260702}"
export WAIT_PIDS="${WAIT_PIDS:-3777320}"
export GPU_UUID="${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
export SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base,openai-community/gpt2}"
export SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-0.85,0.15}"
export RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-sequential-all}"
export CALIBRATION_FILE="${CALIBRATION_FILE:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/calibration_mix_plus_recovery_drills_20260702.jsonl}"
export EVAL_SUITE_FILE="${EVAL_SUITE_FILE:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/behavior_suite_exactgen_20260702.json}"
export MAX_CALIBRATION_TEXTS="${MAX_CALIBRATION_TEXTS:-1024}"
export MAX_BEHAVIOR_PROBES="${MAX_BEHAVIOR_PROBES:-1000}"
export MAX_GENERATION_PROMPTS="${MAX_GENERATION_PROMPTS:-1000}"
export MAX_CHOICE_PROBES="${MAX_CHOICE_PROBES:-250}"
export MAX_TEXT_EVALS="${MAX_TEXT_EVALS:-128}"
export STAGE_INPUT_POLICY="${STAGE_INPUT_POLICY:-incumbent}"
export BLOCK_STEPS_PER_LAYER="${BLOCK_STEPS_PER_LAYER:-5}"
export BLOCK_LR="${BLOCK_LR:-5e-7}"
export ALM_STEPS="${ALM_STEPS:-260}"
export ALM_LR="${ALM_LR:-5e-7}"

exec bash "$SCRIPT_DIR/queue_qwen4b_blank_smollm_cka_weighted_block_weighted_trainable_byteprobe_20260702.sh"
