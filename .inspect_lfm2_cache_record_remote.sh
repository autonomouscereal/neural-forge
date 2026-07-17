set -eu
cd /home/cereal/model-surgery-lab/neural-forge-src
export PYTHONPATH=/home/cereal/model-surgery-lab/neural-forge-src/src
/home/cereal/model-surgery-lab/venv/bin/python - <<'PY'
import json
from neural_forge.hf_lfm2_circuit_trace_cache import open_lfm2_circuit_trace_cache

path = "/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_fold_20260716/circuit-cache/parent/5e29a198be7b1716c45014ec24a16e528a5c56408ed869687d7ae8313cab7656"
cache = open_lfm2_circuit_trace_cache(path, verify=True)
rows = []
for index in range(len(cache)):
    record = cache.load_batch(index).record
    capture = record.get("capture")
    verifications = capture.get("cached_full_prefix_verifications")
    rows.append({
        "batch": index,
        "mode": capture.get("decode_trace_mode"),
        "all_cached_logits_verified": capture.get("all_cached_logits_verified"),
        "all_cached_top1_verified": capture.get("all_cached_top1_verified"),
        "verifications_type": type(verifications).__name__,
        "verifications_count": len(verifications),
        "all_mapping": all(hasattr(value, "get") for value in verifications),
        "all_numerically_close": all(value.get("numerically_close") is True for value in verifications),
        "all_top1_agreement": all(value.get("top1_agreement") is True for value in verifications),
    })
print(json.dumps(rows, default=str, indent=2, sort_keys=True))
PY
