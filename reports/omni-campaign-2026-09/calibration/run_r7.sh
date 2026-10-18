#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# r7 -- the three rungs r6 could not finish, and why they are separated out.
#
# r6 left exactly three holes, and they have two different causes:
#
#   * lc-64k and lc-200k raised torch.OutOfMemoryError.  That is a REAL RESULT
#     -- this runner does not reach 64k on one 24 GiB card in bf16 through stock
#     transformers -- but r6 recorded it as a traceback and no dossier, because
#     the ungated decode-path preflight (two prefills of the longest prompt) died
#     before any review token existed.  That is now guarded: the preflight
#     records `agree: null` instead of dying, and a case that will not fit is
#     recorded with termination `out-of-memory`, which the canonical lens boxes
#     as an ARTIFACT (it counts nothing, because the model was never asked).
#     Re-running these two rungs converts a crash into a measurement.
#   * the bank-v1 REPEAT never loaded at all: another lane's llama-server took
#     21.5 GiB of GPU0 while it was moving weights onto the card.  Nothing about
#     that rung is known yet, and until it runs, `3a13002f...` is ONE run and
#     therefore a number and not an anchor.
#
#     AND IT WILL NOT BE A REPRODUCTION OF r5.  `3a13002f...` was written before
#     `lane_digest_validity` carried a kernel element or a thinking element, so
#     the r5 side of the comparison has unmeasured variables in it and a digest
#     match against it would be luck.  The comparison below prints BOTH validity
#     tuples side by side and labels itself REPRODUCTION or NEW ANCHOR before it
#     reports the digests.  On today's box it is a NEW ANCHOR, and saying so is
#     the point.
#
# So this script REFUSES TO START on a card that is not free, rather than
# producing a third OOM that means nothing.  A dossier written while another
# process owned most of the card would not be evidence about this parent.
#
# Usage:  ./run_r7.sh <model-dir> <label> [device] [min-free-MiB]
# ---------------------------------------------------------------------------
set -u

MODEL="${1:?model dir}"
LABEL="${2:?label}"
DEV="${3:-cuda:0}"
MIN_FREE="${4:-20000}"

LANE=/home/cereal/omni-campaign/calibration
PY=/home/cereal/model-surgery-lab/venv/bin/python
V1="$LANE/banks/lfm2_semantic_whole_response_prompts_v1.json"
V2="$LANE/banks/omni_review_bank_v2_draft.json"
PROBE="$LANE/banks/matrix_campaign_probe_prompts_v1.json"
SMOKE="$LANE/banks/matrix_campaign_smoke_prompts_v1.json"
OUT="$LANE/out"

cd "$LANE" || exit 1
mkdir -p "$OUT"

index="${DEV##*:}"
free=$(nvidia-smi --id="$index" --query-gpu=memory.free --format=csv,noheader,nounits)
echo "=== [r7] $(date -u +%FT%TZ) device $DEV has ${free} MiB free (need ${MIN_FREE}) ==="
if [ "$free" -lt "$MIN_FREE" ]; then
  echo "REFUSED: $DEV is not free.  A review taken while another process owns"
  echo "the card measures the contention, not the parent.  Re-run when it clears:"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
  exit 3
fi

common=(--model-dir "$MODEL" --parent-code "$LABEL" --stage parent
        --device "$DEV" --require-compute-architecture sm86 --dtype bfloat16
        --probe-prompts "$PROBE" --smoke-prompts "$SMOKE")

run_lc () {
  local tag="$1" lanes="$2"
  echo "=== [r7] long-context rung $tag ==="
  timeout 7200 $PY oc_run_frozen_bank.py "${common[@]}" \
    --label "${LABEL}-LC${tag}" --bank "$V2" --lanes "$lanes" \
    --skip-smoke --maximum-new-tokens 300 --thinking off \
    --out "$OUT/oc-dossier-lc${tag}-${LABEL}.json" \
    > "$OUT/r7-lc${tag}-${LABEL}.log" 2>&1
  echo "    rung $tag exit $?"
}

run_lc 64k  "lc-64k-notice-select-A,lc-64k-notice-select-B,lc-64k-summary-B"
run_lc 200k "lc-200k-notice-select-A,lc-200k-notice-select-B"

echo "=== [r7] bank v1 REPEAT (template-default) -- the anchor ==="
timeout 10800 $PY oc_run_frozen_bank.py "${common[@]}" \
  --label "${LABEL}-REPEAT" --bank "$V1" --maximum-new-tokens 1024 \
  --thinking template-default \
  --out "$OUT/oc-dossier-${LABEL}-REPEAT.json" \
  > "$OUT/r7-v1-repeat-${LABEL}.log" 2>&1
echo "    exit $?"

echo "=== [r7] assessments ==="
for spec in "lc64k-${LABEL}:$V2:LC64k-${LABEL}" \
            "lc200k-${LABEL}:$V2:LC200k-${LABEL}"; do
  stem="${spec%%:*}"; rest="${spec#*:}"; bank="${rest%%:*}"; name="${rest##*:}"
  [ -f "$OUT/oc-dossier-${stem}.json" ] || { echo "    skip $name"; continue; }
  timeout 900 $PY oc_assess.py --dossier "$OUT/oc-dossier-${stem}.json" --bank "$bank" \
    --out-json "$OUT/assessment-${name}.json" \
    --out-markdown "$OUT/SCORING-SHEET-${name}.md" > "$OUT/r7-assess-${name}.log" 2>&1
  echo "    assess $name exit $?"
done
if [ -f "$OUT/oc-dossier-${LABEL}-REPEAT.json" ]; then
  timeout 900 $PY oc_assess.py --dossier "$OUT/oc-dossier-${LABEL}-REPEAT.json" --bank "$V1" \
    --out-json "$OUT/assessment-${LABEL}-REPEAT.json" \
    --out-markdown "$OUT/SCORING-SHEET-${LABEL}-REPEAT.md" \
    > "$OUT/r7-assess-repeat.log" 2>&1
  echo "    assess REPEAT exit $?"
  # THE POINT OF THE REPEAT, and the reason it is not just two digests.
  #
  # `3a13002f...` was taken before `lane_digest_validity` carried a kernel
  # element, so the r5 side of this comparison has an UNMEASURED variable in it.
  # A digest match against an unmeasured tuple is luck, not a reproduction.  So
  # the tuples are printed side by side -- which is what those fields are for --
  # and the run says which of the two things it is:
  #
  #   REPRODUCTION  both sides v2, every validity element present and equal;
  #                 a matching digest then means the PARENT held.
  #   NEW ANCHOR    anything missing or unequal on either side; the digest is
  #                 reported, and a match means only that the digests matched.
  timeout 120 $PY - "$OUT/oc-dossier-${LABEL}.json" "$OUT/oc-dossier-${LABEL}-REPEAT.json" <<'EOF'
import json, sys

KERNEL_V2 = "neural-forge-omni-kernel-path-v2"
paths = sys.argv[1:3]
a, b = (json.load(open(p)) for p in paths)
va = {k: v for k, v in a.get("lane_digest_validity", {}).items() if k != "note"}
vb = {k: v for k, v in b.get("lane_digest_validity", {}).items() if k != "note"}

print("=== lane_digest_validity, run 1 (%s) vs run 2 (%s) ===" % tuple(paths))
width = max(len(k) for k in set(va) | set(vb))
missing, differing = [], []
for key in sorted(set(va) | set(vb)):
    left = va.get(key, "<ABSENT>")
    right = vb.get(key, "<ABSENT>")
    mark = "  " if left == right else "!!"
    if key not in va or key not in vb:
        missing.append(key)
        mark = "??"
    elif left != right:
        differing.append(key)
    print("%s %-*s | %s | %s" % (mark, width, key, left, right))

schemas = (va.get("linear_attention_kernel_report_schema"),
           vb.get("linear_attention_kernel_report_schema"))
reasons = []
if missing:
    reasons.append("unmeasured on one side: " + ", ".join(missing))
if differing:
    reasons.append("unequal: " + ", ".join(differing))
if schemas != (KERNEL_V2, KERNEL_V2):
    reasons.append("kernel report schema %r/%r is not v2 on both sides" % schemas)

kind = "REPRODUCTION" if not reasons else "NEW ANCHOR"
print()
print("comparison kind:", kind)
for reason in reasons:
    print("   because:", reason)
print("lane digest run 1:", a["lane_digest"])
print("lane digest run 2:", b["lane_digest"])
same = a["lane_digest"] == b["lane_digest"]
if kind == "REPRODUCTION":
    print("ANCHOR HOLDS" if same else "*** LANE MOVED ***")
else:
    print(("DIGESTS MATCH, but this is a NEW ANCHOR, not a reproduction: the "
           "two runs are not established as one surface, so a match here is "
           "not evidence that the parent held.")
          if same else
          ("*** DIGESTS DIFFER *** and this is a NEW ANCHOR: at least one "
           "validity element is unmeasured or unequal, so the parent is NOT "
           "what this difference is attributable to.  Treat run 2 as the "
           "anchor from here."))
ra = {r["id"]: r["response_sha256"] for r in a["child_responses"]}
rb = {r["id"]: r["response_sha256"] for r in b["child_responses"]}
moved = sorted(k for k in ra if ra[k] != rb.get(k))
print("lanes moved:", moved or "none")
EOF
fi
echo "=== [r7] DONE $(date -u +%FT%TZ) ==="
