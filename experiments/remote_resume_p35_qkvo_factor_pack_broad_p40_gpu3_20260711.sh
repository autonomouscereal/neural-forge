#!/usr/bin/env bash
set -euo pipefail

# Prepared resume wrapper only.  It reuses the completed, hash-locked live p35
# baseline from the failed-closed v2 consumer and writes a fresh JSON-only run.
# Do not launch without review.
ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}
HF_HUB=${HF_HUB:-/home/cereal/.cache/huggingface/hub}
CHECKPOINT=${CHECKPOINT:-/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child}
SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711}
RESUME_OUTPUT_ROOT=${RESUME_OUTPUT_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_qkvo_broad_resume_20260711}
RUN_ROOT=${RUN_ROOT:-$RESUME_OUTPUT_ROOT/runs}
PACK_NAME=${PACK_NAME:-p35_L19G4_to_L21G6_qkvo_selected_s16_factor_pack_fp16_p40_gpu3_v3}
PACK_ROOT=${PACK_ROOT:-$SOURCE_OUTPUT_ROOT/$PACK_NAME}
FACTOR_PACK=${FACTOR_PACK:-$PACK_ROOT/qkvo_factors.safetensors}
FACTOR_MANIFEST=${FACTOR_MANIFEST:-$PACK_ROOT/qkvo_factors.manifest.json}
BROAD_ROOT=${BROAD_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}
BROAD_SUITE=${BROAD_SUITE:-$BROAD_ROOT/behavior_suite_exactgen_20260702.json}
RAW_QWEN_REPORT=${RAW_QWEN_REPORT:-$BROAD_ROOT/evals/eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json}
BASELINE_NAME=${BASELINE_NAME:-p35_qkvo_selected_factor_pack_v3_broad_eval_fp16_p40_gpu3_v2}
BASELINE_REPORT=${BASELINE_REPORT:-$SOURCE_OUTPUT_ROOT/broad_evaluations/$BASELINE_NAME/p35_current_strict_full.json}
BASELINE_REPORT_SHA256=${BASELINE_REPORT_SHA256:-403329fea0ec53510ec88449b8512de5331a05665e5229d78ca1ffa8ff29003c}
BASELINE_CERTIFICATE=${BASELINE_CERTIFICATE:-$ROOT/experiments/p35_qkvo_broad_fresh544_resume_certificate_20260711.json}
BASELINE_CERTIFICATE_SHA256=${BASELINE_CERTIFICATE_SHA256:-365768ab766310d62afd7de141546eeb10a3576bd4bc8e79a3b212ea326e10e9}
GPU=${GPU:-GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8}
GPU_COMPUTE_CAPABILITY=${GPU_COMPUTE_CAPABILITY:-6.1}
GPU_DRIVER_VERSION=${GPU_DRIVER_VERSION:-580.159.03}
IMAGE=${IMAGE:-comfyui:codex-extended-ltx-ace-20260522}
IMAGE_ID=${IMAGE_ID:-sha256:883f0554354a0311c37a1e6d4cecb50ea6f2a48a5971d8c99d46a17bea43f2c8}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-/home/cereal/model-surgery-lab/venv/lib/python3.12/site-packages}
TRANSFORMERS_TREE_SHA256=${TRANSFORMERS_TREE_SHA256:-6fe5e3308d1708699741fbabb569f8818d1006cac78f30db5d979590cf4b158a}
HUGGINGFACE_HUB_TREE_SHA256=${HUGGINGFACE_HUB_TREE_SHA256:-61bc2dd81273ba03dfee5d5dfb3eed64b1fe34a3fb4cdd864aac3676fbaded44}
NAME=${NAME:-p35_qkvo_selected_factor_pack_v3_broad_eval_fp16_p40_gpu3_resume_v1}
CONTAINER_NAME=${CONTAINER_NAME:-neural-forge-p35-qkvo-v3-broad-resume-p40-gpu3-v1}
ARTIFACT_DIR=$RUN_ROOT/$NAME
LOG=$RESUME_OUTPUT_ROOT/$NAME.log
PID_FILE=$RESUME_OUTPUT_ROOT/$NAME.pid
STARTTICKS_FILE=$RESUME_OUTPUT_ROOT/$NAME.startticks

test -d "$ROOT"
test -d "$HF_HUB"
test -d "$CHECKPOINT"
test "$(sha256sum "$CHECKPOINT/manifest.json" | awk '{print $1}')" = \
  f62927114d5bc40fa10d239f7eb5d628ba217cd3fe2b7dcac0bdb5fbb68e9b1f
test "$(sha256sum "$CHECKPOINT/model.safetensors" | awk '{print $1}')" = \
  02fff3229a7975230650f2e227fe049008d4df38eb52cd9aecff10c066deb503
test "$(sha256sum "$CHECKPOINT/tokenizer_contract.json" | awk '{print $1}')" = \
  aabb2c58fefaba3386507654bcec6d514d84f8b70c5b4f160195abc53207afc1
test -f "$FACTOR_PACK"
test -f "$FACTOR_MANIFEST"
test "$(sha256sum "$FACTOR_PACK" | awk '{print $1}')" = \
  5807fc789516ed5548a78fb4657f369a0f1c67c45a0b9e1626aa5813dd0832a8
test "$(sha256sum "$FACTOR_MANIFEST" | awk '{print $1}')" = \
  6c6e979712e67f862e4574121e500e9c823778f6774ce81647752158ccb2886d
test -f "$BROAD_SUITE"
test -f "$RAW_QWEN_REPORT"
test "$(sha256sum "$BROAD_SUITE" | awk '{print $1}')" = \
  fa6066980faf4677b194e00951076dc477cc4243d185449aa6590ca4c8768166
test "$(sha256sum "$RAW_QWEN_REPORT" | awk '{print $1}')" = \
  3333bc54e9170dc8a65e8a53179c9a20a2ab7d1931e2da5795f6bc5ecf031395
test -f "$BASELINE_REPORT"
test "$(sha256sum "$BASELINE_REPORT" | awk '{print $1}')" = \
  "$BASELINE_REPORT_SHA256"
test -f "$BASELINE_CERTIFICATE"
test "$(sha256sum "$BASELINE_CERTIFICATE" | awk '{print $1}')" = \
  "$BASELINE_CERTIFICATE_SHA256"
test "$(sha256sum "$HF_HUB/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539/tokenizer.json" | awk '{print $1}')" = \
  c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539
test "$(sha256sum "$HF_HUB/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539/tokenizer_config.json" | awk '{print $1}')" = \
  3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5
test -f "$ROOT/experiments/run_qwen_p35_qkvo_factor_pack_broad_eval.py"
test -f "$ROOT/src/neural_forge/hf_qkvo_factor_pack.py"
test "$(sha256sum "$ROOT/experiments/run_qwen_p35_qkvo_factor_pack_broad_eval.py" | awk '{print $1}')" = \
  b3f90234a6e533c8cdd2f5afd843b6b8900508cb002c9347038146e3c8a4f16a
test "$(sha256sum "$ROOT/experiments/run_qwen_unified_surgery.py" | awk '{print $1}')" = \
  25e86fae2ed87fb80080ffd02ed1bc0597f6fc52644360dc160eb35c7e386177
test "$(sha256sum "$ROOT/experiments/run_qwen_compensated_neuron_fold.py" | awk '{print $1}')" = \
  e11cc993efa62de7abcb7e25e3a03659979bf1a8648c0fc2c5e1aaf031056c81
test "$(sha256sum "$ROOT/experiments/reproduce_pack_qwen_p35_qkvo_selected.py" | awk '{print $1}')" = \
  281d7c7367e9221d2f5b6249c3b268baab18ca579a6c4411b61d1162e2008aa3
test "$(sha256sum "$ROOT/experiments/run_qwen_p35_assignment_local_qkvo_residual.py" | awk '{print $1}')" = \
  4df8066324940f426ebb6171634350e3acebfa22307919118c42d93eff5999d7
test "$(sha256sum "$ROOT/experiments/run_qwen_p35_functional_residual_adapter.py" | awk '{print $1}')" = \
  d914e65c0bde4ccaf4bdeada17aee5208c08b92f1edc3cd08d1754bd3b1c0846
test "$(sha256sum "$ROOT/experiments/run_qwen_p35_vo_role_scale_search.py" | awk '{print $1}')" = \
  3d201b6820912981be6c0f6a1668f503269417f7b6678559be0bb5388947d89a
test "$(sha256sum "$ROOT/experiments/run_qwen_residual_attention_probe.py" | awk '{print $1}')" = \
  eb14e3fee926aa973f7ae113e6b806b637e5e211534f4f6d9564a81612e0e1e8
test "$(sha256sum "$ROOT/src/neural_forge/hf_tools.py" | awk '{print $1}')" = \
  5f3b29f5a2622f45b3d9962d6d013b43bc524866d59316340c768098db0d29f1
test "$(sha256sum "$ROOT/src/neural_forge/hf_evaluation_provenance.py" | awk '{print $1}')" = \
  c76e97457a709a3b81f2bff344f26fc51d137a20fc9a8c221df09ac85a3dcf77
test "$(sha256sum "$ROOT/src/neural_forge/hf_qkvo_factor_pack.py" | awk '{print $1}')" = \
  9e477507bdf988c7f9581fe3620d74a5061ff7b1a1ec90d65de45fbc40e4ddf6
test "$(sha256sum "$ROOT/src/neural_forge/hf_qwen_attention_surgery.py" | awk '{print $1}')" = \
  2f2b656a626e635a3dfaa21e58a86f68aeb4f01113d78b771c28846e7a471e26
test "$(sha256sum "$ROOT/src/neural_forge/hf_assignment_local_qkvo_residual_adapter.py" | awk '{print $1}')" = \
  04b6b2c1cecef6d4ff05d26692645755b0cf4025290c64613eded8182e46ae4f
test "$(sha256sum "$ROOT/src/neural_forge/hf_qwen_codebook_model.py" | awk '{print $1}')" = \
  b07a6795f3f358b17e0bc021cbdd1c38b51ba1c87f8bd2c15f4bf24866091da4
test -d "$TRANSFORMERS_SITE/transformers"
test "$(docker image inspect "$IMAGE" --format '{{.Id}}')" = "$IMAGE_ID"
test "$(find "$TRANSFORMERS_SITE/transformers" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')" = \
  "$TRANSFORMERS_TREE_SHA256"
test "$(find "$TRANSFORMERS_SITE/huggingface_hub" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')" = \
  "$HUGGINGFACE_HUB_TREE_SHA256"
mkdir -p "$RESUME_OUTPUT_ROOT"
mkdir -p "$RUN_ROOT"
case "$(readlink -f "$BASELINE_REPORT")" in
  "$(readlink -f "$RESUME_OUTPUT_ROOT")"/*)
    printf '%s\n' 'baseline report must not be reachable through resume RW root' >&2
    exit 1
    ;;
esac
test ! -e "$ARTIFACT_DIR"
test ! -e "$LOG"
test ! -e "$PID_FILE"
test ! -e "$STARTTICKS_FILE"
test -z "$(docker ps -aq --filter "name=^/${CONTAINER_NAME}$")"

GPU_ROW=$(nvidia-smi \
  --query-gpu=uuid,name,compute_cap,driver_version,memory.used,utilization.gpu \
  --format=csv,noheader,nounits | awk -F ', ' -v uuid="$GPU" '$1 == uuid')
test -n "$GPU_ROW"
test "$(printf '%s\n' "$GPU_ROW" | awk -F ', ' '{print $2}')" = "Tesla P40"
test "$(printf '%s\n' "$GPU_ROW" | awk -F ', ' '{print $3}')" = "$GPU_COMPUTE_CAPABILITY"
test "$(printf '%s\n' "$GPU_ROW" | awk -F ', ' '{print $4}')" = "$GPU_DRIVER_VERSION"
GPU_MEMORY_USED=$(printf '%s\n' "$GPU_ROW" | awk -F ', ' '{print $5}')
test "$GPU_MEMORY_USED" -lt 4096

nohup docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus "device=$GPU" \
  --entrypoint python3 \
  -e PYTHONPATH=/vendor:/workspace/src:/workspace/experiments \
  -e TOKENIZERS_PARALLELISM=false \
  -e NEURAL_FORGE_RESUME_CONTAINER_IMAGE_ID="$IMAGE_ID" \
  -e NEURAL_FORGE_RESUME_GPU_UUID="$GPU" \
  -e NEURAL_FORGE_RESUME_GPU_COMPUTE_CAPABILITY="$GPU_COMPUTE_CAPABILITY" \
  -e NEURAL_FORGE_RESUME_DRIVER_VERSION="$GPU_DRIVER_VERSION" \
  -e NEURAL_FORGE_RESUME_TRANSFORMERS_TREE_SHA256="$TRANSFORMERS_TREE_SHA256" \
  -e NEURAL_FORGE_RESUME_HUGGINGFACE_HUB_TREE_SHA256="$HUGGINGFACE_HUB_TREE_SHA256" \
  -v "$ROOT:/workspace:ro" \
  -v "$HF_HUB:/hf-hub:ro" \
  -v "$CHECKPOINT:/codebook:ro" \
  -v "$PACK_ROOT:/factor-input:ro" \
  -v "$BROAD_SUITE:/broad-suite.json:ro" \
  -v "$RAW_QWEN_REPORT:/raw-qwen-parent.json:ro" \
  -v "$BASELINE_REPORT:/fresh-p35-baseline.json:ro" \
  -v "$BASELINE_CERTIFICATE:/fresh-p35-certificate.json:ro" \
  -v "$RUN_ROOT:/output" \
  -v "$TRANSFORMERS_SITE/transformers:/vendor/transformers:ro" \
  -v "$TRANSFORMERS_SITE/transformers-5.12.1.dist-info:/vendor/transformers-5.12.1.dist-info:ro" \
  -v "$TRANSFORMERS_SITE/huggingface_hub:/vendor/huggingface_hub:ro" \
  -v "$TRANSFORMERS_SITE/huggingface_hub-1.21.0.dist-info:/vendor/huggingface_hub-1.21.0.dist-info:ro" \
  "$IMAGE_ID" \
  /workspace/experiments/run_qwen_p35_qkvo_factor_pack_broad_eval.py \
  --codebook-checkpoint /codebook \
  --tokenizer /hf-hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --factor-pack /factor-input/qkvo_factors.safetensors \
  --factor-manifest /factor-input/qkvo_factors.manifest.json \
  --broad-suite /broad-suite.json \
  --raw-qwen-parent-report /raw-qwen-parent.json \
  --resume-p35-report /fresh-p35-baseline.json \
  --resume-p35-report-sha256 "$BASELINE_REPORT_SHA256" \
  --resume-p35-certificate /fresh-p35-certificate.json \
  --resume-p35-certificate-sha256 "$BASELINE_CERTIFICATE_SHA256" \
  --fit-prompts-json /workspace/experiments/p35_p34_residual_attention_fit_prompts_20260711.json \
  --fit-prompt-manifest /workspace/experiments/p35_p34_residual_attention_fit_prompt_manifest_20260711.json \
  --calibration-selection-prompts-json /workspace/experiments/p35_p34_residual_attention_calibration_selection_prompts_20260711.json \
  --heldout-evaluation-prompts-json /workspace/experiments/p35_p34_residual_attention_heldout_evaluation_prompts_20260711.json \
  --audited-prompt-manifest /workspace/experiments/p35_p34_residual_attention_prompt_manifest_20260711.json \
  --transition-inventory /workspace/experiments/p35_p34_generation_transitions_20260711.json \
  --output-directory "/output/$NAME" \
  --device cuda:0 \
  --dtype float16 \
  --local-files-only \
  >"$LOG" 2>&1 </dev/null &

PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
sleep 1
kill -0 "$PID"
STARTTICKS=$(awk '{print $22}' "/proc/$PID/stat")
printf '%s\n' "$STARTTICKS" >"$STARTTICKS_FILE"
printf 'prepared broad-eval resume launched pid=%s startticks=%s gpu=%s artifacts=%s log=%s baseline=%s baseline_sha256=%s prelaunch_gpu=%s\n' \
  "$PID" "$STARTTICKS" "$GPU" "$ARTIFACT_DIR" "$LOG" "$BASELINE_REPORT" \
  "$BASELINE_REPORT_SHA256" "$GPU_ROW"
