#!/bin/bash
# k3cache lane, after the 1.2.0 review fixes (digest / fold receipt refusals, resume never re-reads finished
# shards): re-run every CPU arm on the EXISTING sealed caches with the new tool, into rerun-1.2.0/ (the original
# receipts are left untouched), plus a real-source check that the 1.2.0 up-front digest gate passes on the real
# K3 shard order against k3_lexar_digest_verify.json (hf_sha256) and that the sealed review pass's in-line
# digests equal hf_sha256.  No GPU (the 93-layer pass and D2 are numerically unchanged and need GPU3).
# Run detached:  setsid nohup bash run_k3cache_rerun_v120_2026-10-13.sh > $OC/distill/logs/k3cache/rerun_v120.log 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
SRC=$OC/distill/parity/k3cache-2026-10-12
OUT=$SRC/rerun-1.2.0
C=$OC/composite/corpus
REF=$OC/k3-run-oct09-claude/fold/review
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=$T
echo "rerun start $(date -u +%FT%TZ) tool_sha256=$(sha256sum $T/k3_teacher_cache.py | cut -c1-64)"
for arm in review-pass:parity_receipt.json d2-from-L61:parity_d2_receipt.json; do
  dir=${arm%%:*}; name=${arm##*:}
  nice -n 15 "$PY" "$T/k3_teacher_cache.py" parity --cache-dir "$SRC/$dir" \
    --reference "$REF/k3_review_input_ids_teacher_forced.safetensors" \
    --reference "$REF/k3_review_native_input_ids_teacher_forced.safetensors" --out "$OUT/$name"
  echo "PARITY_EXIT[$dir]=$?"
done
nice -n 15 "$PY" "$T/k3_teacher_cache.py" xtok --cache-dir "$SRC/review-pass" --boundary-map "$C/boundary_map.json" \
  --k3-ids "$C/k3_review_input_ids.safetensors" --qwen-ids "$C/qwen_review_input_ids.safetensors" --out "$OUT/xtok-review"
echo "XTOK_EXIT=$?"
nice -n 15 "$PY" "$T/k3_teacher_cache.py" probe --cache-dir "$SRC/review-pass" --boundary-map "$C/boundary_map.json" \
  --k3-ids "$C/k3_review_input_ids.safetensors" --qwen-ids "$C/qwen_review_input_ids.safetensors" \
  --row-meta "$C/row_meta.json" --route-key source --out "$OUT/probe_review_k3only.json"
echo "PROBE_EXIT=$?"
nice -n 15 "$PY" - "$T" "$SRC/review-pass/manifest.json" "$OUT/real_source_gate_check.json" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1])
import k3_teacher_cache as ktc
engine = ktc.Engine(ktc.DEFAULT_ENGINE_DIR)  # the pinned engine puts the k3 lane (k3_common) on sys.path
k3_common = engine.common
index = k3_common.K3Index(ktc.DEFAULT_SRC_DIR)
plan = [s for i in range(index.num_layers) for s in index.shards_for(index.layer_keys(i))]
model_level = sorted({index.weight_map[k] for k in index.weight_map if any(k.endswith(s) for s in (
    "embed_tokens.weight", "lm_head.weight", "model.norm.weight", "output_attn_res_proj.weight",
    "output_attn_res_norm.weight"))})
order = []
for s in plan + model_level:
    if s not in order:
        order.append(s)
digests = ktc.expected_digests(ktc.DEFAULT_DIGEST_RECEIPT)
problems = ktc.digest_receipt_problems(digests, order, ktc.DEFAULT_DIGEST_RECEIPT)
fold = ktc.expected_from_fold_receipt(ktc.DEFAULT_FOLD_RECEIPT)
drift = ktc.compare_stat(ktc.source_stat(index, ktc.DEFAULT_SRC_DIR), fold, what="now vs fold receipt") if fold else ["no fold receipt"]
sealed = json.load(open(sys.argv[2]))["integrity"]["inline_digest"]["results"]
vs_hf = {s: r["sha256"] == digests["files"][s]["sha256"] and int(r["bytes"]) == int(digests["files"][s]["bytes"])
         for s, r in sealed.items()}
out = {"tool_version": ktc.TOOL_VERSION, "tool_sha256": ktc.sha256_file(ktc.__file__),
       "order_shards": len(order), "up_front_digest_problems": problems,
       "fold_receipt_usable": fold is not None, "drift_vs_fold_receipt": drift,
       "sealed_review_pass_inline_results": len(sealed), "sealed_equal_hf_sha256_and_size": sum(vs_hf.values()),
       "sealed_not_equal": sorted(s for s, ok in vs_hf.items() if not ok),
       "order_minus_sealed": sorted(set(order) - set(sealed))}
json.dump(out, open(sys.argv[3], "w"), indent=1)
print("REAL_SOURCE_CHECK", json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in out.items()}))
PY
echo "REAL_CHECK_EXIT=$?"
echo "rerun end $(date -u +%FT%TZ)"
