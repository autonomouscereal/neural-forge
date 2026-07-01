#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TAG="${TAG:-qwen4b_blank_smollm_chain_mlpout_true_lrc_eora_rank16_gpu4_20260629}" \
RESIDUAL_MODE="${RESIDUAL_MODE:-eora-pre}" \
RESIDUAL_ALPHA="${RESIDUAL_ALPHA:-0.01}" \
RESIDUAL_RANK="${RESIDUAL_RANK:-16}" \
exec "$SCRIPT_DIR/launch_qwen4b_blank_smollm_true_lrc_chain.sh"
