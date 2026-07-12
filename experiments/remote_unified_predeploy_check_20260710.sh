#!/usr/bin/env bash
set -euo pipefail

repo=/home/cereal/model-surgery-lab/neural-forge-unified-20260710
pivot=/home/cereal/model-surgery-lab/neural-forge-pivot
python=/home/cereal/model-surgery-lab/venv/bin/python
model=/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
p35=/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child
calibration=/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/calibration_mix_plus_recovery_drills_20260702.jsonl
selection=/mnt/ai_data/neural-forge/qwen3_4b_compensated_20260710/selection_suite_seed90710.json
certification=/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/behavior_suite_exactgen_20260702.json

for path in "$pivot" "$python" "$model" "$p35" "$calibration" "$selection" "$certification"; do
  test -e "$path"
  printf 'exists %s\n' "$path"
done

if [[ -e "$repo" ]]; then
  printf 'isolated_checkout_exists %s\n' "$repo"
else
  printf 'isolated_checkout_absent %s\n' "$repo"
fi

"$python" - "$selection" "$p35" <<'PY'
import json
import pathlib
import sys

import safetensors
import torch
import transformers

selection = pathlib.Path(sys.argv[1])
p35 = pathlib.Path(sys.argv[2])
suite = json.loads(selection.read_text(encoding="utf-8"))
normalized_suite = suite.get("suite", suite)
print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "safetensors": safetensors.__version__,
    "selection_keys": sorted(suite),
    "selection_counts": {
        key: len(value)
        for key, value in normalized_suite.items()
        if isinstance(value, list)
    },
}, sort_keys=True))

manifest = json.loads((p35 / "manifest.json").read_text(encoding="utf-8"))
print(json.dumps({
    "checkpoint_schema": manifest.get("schema"),
    "state_tensor_count": manifest.get("state_tensor_count"),
    "tensor_file_bytes": (p35 / manifest["tensor_file"]).stat().st_size,
    "artifact_file_bytes": sum(
        path.stat().st_size for path in p35.rglob("*") if path.is_file()
    ),
    "vocabulary_state_keys": (manifest.get("state_groups") or {}).get("vocabulary"),
    "parameter_report": manifest.get("parameter_report"),
}, sort_keys=True))
PY

nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

df -h /home/cereal/model-surgery-lab /mnt/ai_data
