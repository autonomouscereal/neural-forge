#!/bin/bash
# Review record for OC02-F-Q38-27BU-C2K4352 from the EXISTING Oct-08 dossier and
# assessment (lane E1, 2026-10-09).  Mirrors wavea_review.sh steps 4/4a/4b:
# oc_review_upper_bound.py (assessment run WITHOUT --per-case -> every count an
# UPPER BOUND, canonical_content_miss 0 by construction), then record-review dry
# run, then (only with --confirm) record-review --confirm.  No package is built
# (wavea step 5 is not part of this lane), and nothing is decoded.
#
# PARENT COUNT.  No Q38-27BU parent dossier/assessment exists on any runner
# (find over $OC for *assessment*/*oc-dossier*/bands*.json: only the child's).
# oc_review_upper_bound.py refuses without one, so the parent count is STATED as
# 15 = the bank's full intelligence-lane count.  That is not a parent
# measurement; it is the value that keeps the verdict ceiling from ever reading
# `alive-parent-band` against a parent nobody has run.  The notes say so.
#
# Usage:  s3_review_oc02.sh            # build review JSON + record-review dry run
#         s3_review_oc02.sh --confirm  # ... then record-review --confirm
set -u
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
export OC_ROOT=$OC
ID=OC02-F-Q38-27BU-C2K4352
D1=$OC/composite/s3/review/oc-dossier-OC02-C2K4352-NATIVE-NOTHINK.json
A1=$OC/calibration/out/assessment-OC02-C2K4352-NATIVE-NOTHINK.json
REV=$OC/composite/s3/review/review-$ID.json
REVIEWER="claude-lane-e1 (oc_review_upper_bound.py over the Oct-08 native dossier)"
echo "=== review $ID  ($(date -u +%FT%TZ))"
sha256sum "$D1" "$A1" "$OC/calibration/oc_review_upper_bound.py" "$OC/ledger/oc_checkpoint.py"
NOTES="Machine UPPER BOUND, not a reading: oc_review_upper_bound.py over assessment-OC02-C2K4352-NATIVE-NOTHINK.json (file sha fb3cd4fe75c3f398..., oc_assess WITHOUT --per-case, columns.is_upper_bound true) about dossier oc-dossier-OC02-C2K4352-NATIVE-NOTHINK.json (file sha 5fe7463cc258d651..., dossier_sha256 cb4515ed106bbcea...). \
SURFACE: decoded on the NATIVE Chimera export /mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native (run_omni_frozen_review 1.1.0, tree ef1ffb9c, bf16, split 28/36 on two RTX 3090, thinking off, cap 1024, greedy), NOT on the registered HF directory; they are tied only by \$OC/composite/s3/receipt_s3_streamed_exactness_cpu_fp32.json (argmax 208/210, max logit diff 0.1224) and receipt_s3_streamed_exactness.json (bf16, 201/210, 0.5312). \
Form boxes ECHO 8 / CAP_RIDE 5 / EMPTY 2 / CONTENT 1 of 16 (13 censored at cap, 5 loops, 2 empty); column A 8 / column B 1 of 15 scored (B = A minus 7 voided copy-back lanes; the one CONTENT lane, context-retrieval, is 3 tokens). canonical_content_miss is 0 BY CONSTRUCTION, so this record cannot make the child bad or deletable. \
PARENT COUNT 15 IS NOT A MEASUREMENT: no Q38-27BU parent surface exists on any runner, the tool refuses without a parent count, and 15 (the bank's intelligence-lane total) is stated so the verdict ceiling cannot claim alive-parent-band. Replace it when the parent (PROBE.S0 on the same runner) is decoded. \
Decode smoke verdict pending (a human reads the three smoke records). No package built; no per-case reading by Cyril yet."
timeout 280 $PY $OC/calibration/oc_review_upper_bound.py \
  --assessment "$A1" --dossier "$D1" \
  --parent-useful-or-partial 15 \
  --reviewer "$REVIEWER" --notes "$NOTES" --out "$REV" || { echo "=== upper bound exited $?"; exit 1; }
sha256sum "$REV"
echo "--- record-review DRY RUN"
timeout 280 $PY $OC/ledger/oc_checkpoint.py --root $OC record-review --checkpoint-id $ID --review-json "$REV"
rc=$?; echo "=== dry-run exit $rc"
[ $rc -eq 0 ] || exit $rc
if [ "${1:-}" = "--confirm" ]; then
  echo "--- record-review --confirm"
  timeout 280 $PY $OC/ledger/oc_checkpoint.py --root $OC record-review --checkpoint-id $ID --review-json "$REV" --confirm
  echo "=== confirm exit $?"
fi
echo "=== done ($(date -u +%FT%TZ))"
