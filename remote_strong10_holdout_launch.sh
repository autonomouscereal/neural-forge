#!/usr/bin/env bash
# Reusable authenticated launcher for the current strong 10% child.
set -euo pipefail

SRC=/home/cereal/model-surgery-lab/neural-forge-src
PY=/home/cereal/model-surgery-lab/venv/bin/python
MODEL=/home/cereal/model-surgery-lab/artifacts/lfm2_large_ratio_20260716/wave2-producer-first-r2/lfm2_10pct_mlp_q16_kv8_ffn2026_producer_first/output/final-child
TOKENIZER=/home/cereal/model-surgery-lab/models/LFM2.5-230M
ENVELOPE=/home/cereal/model-surgery-lab/artifacts/lfm2_intelligence_holdout_v1/parent-envelope.json
ROOT=/home/cereal/model-surgery-lab/artifacts/lfm2_intelligence_holdout_v1/candidates/strong10-v4
LOG=${ROOT}.runner.log
PIDFILE=${ROOT}.pid
GPU=GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561

[ ! -e "$ROOT" ] && [ ! -e "$LOG" ] && [ ! -e "$PIDFILE" ] || {
  echo output_collision
  exit 4
}
cd "$SRC"
sha256sum -c <<'HASHES'
5c63a7ab46c6b1671aa3e4f6e3e04af16bfccc2ad9d21a047d37dd374e46d4af  experiments/build_lfm2_intelligence_holdout_parent_cache.py
aa60272229a43db9b0b858588f29e6b4b1051aa0beff5b9a65d97b121bd141f3  experiments/evaluate_lfm2_intelligence_holdout.py
3ffbf752ce2e4a9a843975ca522ba584c5691d27ae8676406c85f4cd9dae9759  src/neural_forge/hf_lfm2_intelligence_holdout.py
cbe2acd7d11fbee54c6a7ac1302f7ce4521438f01b4b623c7eb58445906a0521  src/neural_forge/hf_lfm2_intelligence_gate.py
a8922977266e879129184ea37cb2718f34f435b6f5f8eb8f6cddecb16451a01b  /home/cereal/model-surgery-lab/artifacts/lfm2_intelligence_holdout_v1/parent-envelope.json
HASHES
FREE_MIB=$(
  nvidia-smi --id="$GPU" --query-gpu=memory.free \
    --format=csv,noheader,nounits | tr -d '[:space:]'
)
[ "$FREE_MIB" -ge 20000 ] || {
  echo insufficient_gpu_memory
  exit 6
}
nvidia-smi --id="$GPU" --query-gpu=uuid,name,memory.free --format=csv,noheader
export PYTHONPATH="$SRC/src:$SRC"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
nohup "$PY" - <<'PY' > "$LOG" 2>&1 &
import hashlib
import json
from pathlib import Path

from experiments import build_lfm2_intelligence_holdout_parent_cache as generation
from experiments import evaluate_lfm2_intelligence_holdout as evaluation
from experiments import run_lfm2_residual_attention_fold as fold_runtime
from neural_forge.hf_lfm2_intelligence_holdout import (
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_candidate_outputs,
)
from neural_forge import hf_lfm2_shared_checkpoint as shared_runtime

model_dir = Path(
    "/home/cereal/model-surgery-lab/artifacts/lfm2_large_ratio_20260716/"
    "wave2-producer-first-r2/"
    "lfm2_10pct_mlp_q16_kv8_ffn2026_producer_first/output/final-child"
)
tokenizer_dir = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")
envelope_path = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_intelligence_holdout_v1/parent-envelope.json"
)
root = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_intelligence_holdout_v1/candidates/strong10-v4"
)
if root.exists():
    raise RuntimeError(f"refusing existing root {root}")
root.mkdir(parents=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if file_sha256(envelope_path) != (
    "a8922977266e879129184ea37cb2718f34f435b6f5f8eb8f6cddecb16451a01b"
):
    raise RuntimeError("parent envelope file hash mismatch")
holdout = load_lfm2_intelligence_holdout_parent_envelope(
    json.loads(envelope_path.read_text(encoding="utf-8")),
    expected_envelope_sha256=(
        "b0a2c815b7442df5a3cb9fa113aecaf2de5f094222f7d74e57140719e486e8d2"
    ),
    expected_manifest_sha256=(
        "5a4c4031045a7cfeaa46f38fdd243c383a4e737dfef6f127a195edef1ad11690"
    ),
    expected_parent_state_fingerprint=(
        "fd580146d97c970226d4b1667305a92d7f04bfd4f01aa8afedfd7d826dcacde0"
    ),
    expected_parent_model_fingerprint=(
        "96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb"
    ),
    expected_tokenizer_fingerprint=(
        "eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46"
    ),
)
model = fold_runtime.load_local_lfm2_model(
    model_dir,
    device="cuda:0",
    dtype=fold_runtime._torch_dtype("bfloat16"),
)
model.eval()
candidate_state = fold_runtime.model_state_sha256(model)
if candidate_state != (
    "433f925c94fb8c508d7624db1c970e9e2d84c8af383fe54aa27a594159077271"
):
    raise RuntimeError("strong 10% child state mismatch")
seen_parameters = set()
unique_parameters = 0
for parameter in model.parameters():
    identity = id(parameter)
    if identity in seen_parameters:
        continue
    seen_parameters.add(identity)
    unique_parameters += int(parameter.numel())
if unique_parameters != 206726912:
    raise RuntimeError(
        f"strong 10% child parameter mismatch: {unique_parameters}"
    )
if any(parameter.grad is not None for parameter in model.parameters()):
    raise RuntimeError("strong 10% child unexpectedly has gradients")
tokenizer = fold_runtime.load_local_tokenizer(tokenizer_dir)
rows = generation.generate_lfm2_holdout_rows_once(
    model,
    tokenizer,
    holdout.manifest,
)
if len(rows) != 64:
    raise RuntimeError("candidate generation did not cover 64 cases")
if fold_runtime.model_state_sha256(model) != candidate_state:
    raise RuntimeError("candidate state changed during generation")
if any(parameter.grad is not None for parameter in model.parameters()):
    raise RuntimeError("candidate generation wrote gradients")
candidate = seal_lfm2_intelligence_holdout_candidate_outputs(
    holdout.manifest,
    rows,
    candidate_state_fingerprint=candidate_state,
    tokenizer_fingerprint=(
        "eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46"
    ),
    eos_token_ids=holdout.eos_token_ids,
)
candidate_path = root / "candidate-outputs.json"
shared_runtime._atomic_write_new_json(candidate_path, candidate)
candidate_file_sha = file_sha256(candidate_path)
report_path = root / "intelligence-report.json"
manual_path = root / "manual-review-packet.json"
report = evaluation.evaluate_lfm2_intelligence_holdout(
    evaluation.HoldoutEvaluationConfig(
        parent_envelope=envelope_path,
        expected_parent_envelope_file_sha256=(
            "a8922977266e879129184ea37cb2718f34f435b6f5f8eb8f6cddecb16451a01b"
        ),
        expected_parent_envelope_sha256=(
            "b0a2c815b7442df5a3cb9fa113aecaf2de5f094222f7d74e57140719e486e8d2"
        ),
        expected_manifest_sha256=(
            "5a4c4031045a7cfeaa46f38fdd243c383a4e737dfef6f127a195edef1ad11690"
        ),
        expected_parent_state_fingerprint=(
            "fd580146d97c970226d4b1667305a92d7f04bfd4f01aa8afedfd7d826dcacde0"
        ),
        expected_parent_model_fingerprint=(
            "96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb"
        ),
        tokenizer_directory=tokenizer_dir,
        expected_tokenizer_fingerprint=(
            "eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46"
        ),
        candidate_report=candidate_path,
        expected_candidate_report_sha256=candidate_file_sha,
        report_path=report_path,
        manual_review_packet_path=manual_path,
        expected_candidate_outputs_sha256=candidate[
            "candidate_outputs_sha256"
        ],
        expected_candidate_state_fingerprint=candidate_state,
    )
)
receipt = {
    "schema": "neural-forge-lfm2-strong10-independent-holdout-v1",
    "status": report["status"],
    "promoted": bool(report["promoted"]),
    "candidate_unique_parameters": unique_parameters,
    "candidate_state_fingerprint": candidate_state,
    "candidate_generate_invocations": len(rows),
    "parent_generate_invocations": 0,
    "parent_forward_calls": 0,
    "training_steps": 0,
    "optimizer_steps": 0,
    "backward_calls": 0,
    "candidate_outputs_file_sha256": candidate_file_sha,
    "candidate_outputs_sha256": candidate["candidate_outputs_sha256"],
    "intelligence_report_file_sha256": file_sha256(report_path),
    "intelligence_report_sha256": report["report_sha256"],
    "manual_review_packet_file_sha256": file_sha256(manual_path),
}
shared_runtime._atomic_write_new_json(root / "run-receipt.json", receipt)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
printf '%s\n' "$!" > "$PIDFILE"
printf 'started pid=%s log=%s root=%s\n' "$(cat "$PIDFILE")" "$LOG" "$ROOT"
