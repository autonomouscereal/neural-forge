#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# r6 -- what the r5 pass could not answer, in the order the answers are worth.
#
# EVERY RUNG IS A SEPARATE INVOCATION ON PURPOSE.  A long-context rung that runs
# out of memory must not take the rungs below it down with it, and a rung that
# OOMs is a MEASUREMENT here, not a failure: "this parent's 200k lane does not
# decode in bf16 on one 3090 through stock transformers" is exactly what a
# calibration lane exists to find out before a child is built on top of it.  So
# every rung gets its own process, its own log and its own exit code, and the
# script never stops early.
#
# THE ORDER IS CHEAPEST-AND-MOST-INFORMATIVE FIRST, because GPU0 is shared with
# other lanes and a pass that is interrupted half way should already have paid
# for itself.  r5's case 15 took 1,760 s against ~120 s for its neighbours, which
# is what contention on this card looks like.
#
#   1. bank v1 with --thinking off -- THE SECOND SURFACE, and the one that
#      actually produces answers.  r5 measured that this parent's chat template
#      opens a think block, so thirteen of sixteen lanes spent the 1,024-token V5
#      budget on chain-of-thought and terminated censored-at-cap with the answer
#      unwritten.  The protocol says a cap hit is never a failure, and it is not
#      one here either -- but it leaves a reader nothing to read.  This run and
#      r5 are the same bank, parent, card and dtype differing in ONE template
#      variable, and together they are what Cyril needs to pick the omni review
#      surface.
#   2. the bank-v2 long-context lanes, rung by rung (8k first), which have NEVER
#      decoded.  Each regenerates its haystack and checks it against the digest
#      the bank pinned before a token is produced.
#   3. the twenty-one short bank-v2 text lanes.
#   4. the deeper long-context rungs, 32k -> 64k -> 200k, in that order, so an
#      out-of-memory at the top costs nothing below it.
#   5. bank v1 on the template default again -- the REPEAT.  The lane digest
#      either reproduces r5's or it does not, and that is the omni analogue of
#      the LFM2 parent-lane anchor being ESTABLISHED rather than asserted.  One
#      run is a number; two runs is an anchor.  It is LAST because it is the most
#      expensive arm here (an hour of chain-of-thought) and the least urgent.
#
# 1 to 4 run with thinking off, and that is a stated choice, not a default:
# with it on, a 200k prefill plus an unbounded think block cannot finish, and the
# text these lanes exist to put in front of Cyril would not be produced at all.
# The setting is in every dossier's lane_digest_validity, so no artifact here can
# be mistaken for one taken on the other surface.
#
# Usage:  ./run_r6.sh <model-dir> <label> [device]
# ---------------------------------------------------------------------------
set -u

MODEL="${1:?model dir}"
LABEL="${2:?label}"
DEV="${3:-cuda:0}"

LANE=/home/cereal/omni-campaign/calibration
PY=/home/cereal/model-surgery-lab/venv/bin/python
V1="$LANE/banks/lfm2_semantic_whole_response_prompts_v1.json"
V2="$LANE/banks/omni_review_bank_v2_draft.json"
PROBE="$LANE/banks/matrix_campaign_probe_prompts_v1.json"
SMOKE="$LANE/banks/matrix_campaign_smoke_prompts_v1.json"
OUT="$LANE/out"

cd "$LANE" || exit 1
mkdir -p "$OUT"

common=(--model-dir "$MODEL" --parent-code "$LABEL" --stage parent
        --device "$DEV" --require-compute-architecture sm86 --dtype bfloat16
        --probe-prompts "$PROBE" --smoke-prompts "$SMOKE")

echo "=== [r6] START $(date -u +%FT%TZ)  model=$MODEL label=$LABEL dev=$DEV ==="

# -- 1. the second surface: the one that produces answers --------------------
echo "=== [r6] 1/5 bank v1 with --thinking off ==="
timeout 7200 $PY oc_run_frozen_bank.py "${common[@]}"   --label "${LABEL}-NOTHINK" --bank "$V1" --maximum-new-tokens 1024   --thinking off   --out "$OUT/oc-dossier-${LABEL}-NOTHINK.json"   > "$OUT/r6-v1-nothink-${LABEL}.log" 2>&1
echo "    exit $?"

# -- the long-context rung helper -------------------------------------------
# The cap is 300 new tokens: every one of these lanes asks for a fact under an
# 80-120 word budget, and at 200k a cap-ride costs many minutes of pure waste.
run_lc () {
  local tag="$1"; shift
  local lanes="$1"; shift
  echo "=== [r6] long-context rung $tag ==="
  timeout 7200 $PY oc_run_frozen_bank.py "${common[@]}"     --label "${LABEL}-LC${tag}" --bank "$V2" --lanes "$lanes"     --skip-smoke --maximum-new-tokens 300 --thinking off     --out "$OUT/oc-dossier-lc${tag}-${LABEL}.json"     > "$OUT/r6-lc${tag}-${LABEL}.log" 2>&1
  echo "    rung $tag exit $?"
}

echo "=== [r6] 2/5 long-context 8k (never decoded before) ==="
run_lc 8k "lc-8k-notice-select-A,lc-8k-notice-select-B,lc-8k-crossref-two-facts-B"

# -- 3. the short bank-v2 text lanes ----------------------------------------
echo "=== [r6] 3/5 bank v2 short text lanes ==="
timeout 7200 $PY oc_run_frozen_bank.py "${common[@]}"   --label "${LABEL}-B2" --bank "$V2" --lanes "code-rle-encode-A,code-bugfix-pagination-B,code-review-sql-injection-B,code-trace-recursion-A,code-api-contract-B,reason-arith-two-step-A,reason-schedule-constraints-B,reason-uncertainty-churn-B,reason-unit-conversion-A,reason-causal-confound-B,reason-counterexample-B,mt-venue-revision-A,mt-dietary-correction-B,mt-instruction-persistence-B,mt-context-carryover-B,instr-format-json-strict-A,instr-word-budget-A,instr-negative-constraint-B,instr-ordered-steps-B,know-explain-tradeoff-A,know-refuse-fabrication-B"   --skip-smoke --maximum-new-tokens 1024 --thinking off   --out "$OUT/oc-dossier-b2-${LABEL}.json"   > "$OUT/r6-b2-${LABEL}.log" 2>&1
echo "    exit $?"

# -- 4. the deeper long-context rungs ---------------------------------------
echo "=== [r6] 4/5 deeper long-context rungs ==="
run_lc 32k  "lc-32k-conflicting-updates-B"
run_lc 64k  "lc-64k-notice-select-A,lc-64k-notice-select-B,lc-64k-summary-B"
run_lc 200k "lc-200k-notice-select-A,lc-200k-notice-select-B"

# -- 5. the repeat, on the template default: the anchor ---------------------
echo "=== [r6] 5/5 bank v1 REPEAT (template-default) ==="
timeout 10800 $PY oc_run_frozen_bank.py "${common[@]}"   --label "${LABEL}-REPEAT" --bank "$V1" --maximum-new-tokens 1024   --thinking template-default   --out "$OUT/oc-dossier-${LABEL}-REPEAT.json"   > "$OUT/r6-v1-repeat-${LABEL}.log" 2>&1
echo "    exit $?"

# -- 5. sheets --------------------------------------------------------------
# Every sheet is an UPPER BOUND until Cyril attaches a per-case reading; the
# assessment says so in the artifact and at the top of the markdown.
echo "=== [r6] assessments ==="
assess () {
  local dossier="$1" bank="$2" name="$3"
  [ -f "$OUT/$dossier" ] || { echo "    skip $name (no dossier)"; return; }
  timeout 900 $PY oc_assess.py --dossier "$OUT/$dossier" --bank "$bank" \
    --out-json "$OUT/assessment-${name}.json" \
    --out-markdown "$OUT/SCORING-SHEET-${name}.md" \
    > "$OUT/r6-assess-${name}.log" 2>&1
  echo "    assess $name exit $?"
}
assess "oc-dossier-${LABEL}.json"          "$V1" "${LABEL}"
assess "oc-dossier-${LABEL}-REPEAT.json"   "$V1" "${LABEL}-REPEAT"
assess "oc-dossier-${LABEL}-NOTHINK.json"  "$V1" "${LABEL}-NOTHINK"
assess "oc-dossier-b2-${LABEL}.json"       "$V2" "B2-${LABEL}"
for tag in 8k 32k 64k 200k; do
  assess "oc-dossier-lc${tag}-${LABEL}.json" "$V2" "LC${tag}-${LABEL}"
done

echo "=== [r6] DONE $(date -u +%FT%TZ) ==="
