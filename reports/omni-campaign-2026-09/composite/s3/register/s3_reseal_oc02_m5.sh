#!/bin/bash
# Lane M5 (2026-10-10): re-seal the S3 review of OC02-F-Q38-27BU-C2K4352 against
# the MEASURED parent band (E2: PARENT-Q38-27BU column A 15 / B 12, both upper
# bounds) instead of the stated placeholder 15 that lane E1 used, then build the
# human-review package and re-record the review carrying the package digest
# (the wavea_review.sh 4 -> 5 -> 5b ordering: a review may not name a
# package_path without a package_sha256, and the digest only exists once the
# package is written).  CPU only, no GPU.  Deletes nothing; refuses to write
# over any existing output.
#
# Usage:
#   s3_reseal_oc02_m5.sh assess              # oc_assess (--parent-dossier) + bare review JSON + record-review DRY RUN
#   s3_reseal_oc02_m5.sh assess --confirm    # ... + record-review --confirm (record 0003)
#   s3_reseal_oc02_m5.sh package             # oc_package build + review JSON with digest + DRY RUN
#   s3_reseal_oc02_m5.sh package --confirm   # ... + record-review --confirm (record 0004)
#   s3_reseal_oc02_m5.sh dashboard           # oc_dashboard (SCORECARD.md + REVIEW_QUEUE.md), list, digest, verify
set -u
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
export OC_ROOT=$OC OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
N="nice -n 19"
ID=OC02-F-Q38-27BU-C2K4352
D1=$OC/composite/s3/review/oc-dossier-OC02-C2K4352-NATIVE-NOTHINK.json
BANK=$OC/calibration/banks/lfm2_semantic_whole_response_prompts_v1.json
PD=$OC/composite/s0/review/oc-dossier-PARENT-Q38-27BU-NATIVE-NOTHINK.json
PA=$OC/calibration/out/assessment-PARENT-Q38-27BU-NATIVE-NOTHINK.json
PAMD=$OC/calibration/out/assessment-PARENT-Q38-27BU-NATIVE-NOTHINK.md
A1=$OC/calibration/out/assessment-OC02-C2K4352-NATIVE-NOTHINK-vs-PARENT-m5.json
S1=$OC/calibration/out/SCORING-SHEET-OC02-C2K4352-NATIVE-NOTHINK-vs-PARENT-m5.md
REV1=$OC/composite/s3/review/review-$ID-m5-parent-measured.json
REV2=$OC/composite/s3/review/review-$ID-m5-parent-measured-packaged.json
NOTICE=$OC/composite/s3/review/UPPER-BOUND-NOTICE-$ID-m5.md
RX1=$OC/composite/s3/receipt_s3_streamed_exactness_cpu_fp32.json
RX2=$OC/composite/s3/receipt_s3_streamed_exactness.json
PKG=$OC/oc02/human-review/$ID
REVIEWER="claude-lane-m5 (oc_review_upper_bound.py; parent = measured E2 PARENT-Q38-27BU assessment)"
STAGE=${1:-}; CONFIRM=${2:-}
die() { echo "=== REFUSED: $1"; exit "${2:-3}"; }
now() { date -u +%FT%TZ; }
head_field() {  # print one field of the current head record
  timeout 120 $N $PY $OC/ledger/oc_checkpoint.py --root $OC show --checkpoint-id $ID 2>/dev/null \
  | $PY -c 'import json,sys
d=json.load(sys.stdin); r=d.get("review") or {}
f=sys.argv[1]
v={"rid":d.get("record_instance_id"),"reviewer":r.get("reviewer"),"pkg":r.get("package_sha256"),
   "status":d.get("status"),"dossier":r.get("dossier_sha256") or (r.get("bank1") or {}).get("dossier_sha256")}.get(f)
print(v if v is not None else "")' "$1"
}
record() {  # $1 = review json; dry run, then --confirm only when asked
  echo "--- record-review DRY RUN ($(basename "$1"))"
  timeout 280 $N $PY $OC/ledger/oc_checkpoint.py --root $OC record-review --checkpoint-id $ID --review-json "$1"
  rc=$?; echo "=== dry-run exit $rc"; [ $rc -eq 0 ] || exit $rc
  if [ "$CONFIRM" = "--confirm" ]; then
    echo "--- record-review --confirm ($(basename "$1"))"
    timeout 280 $N $PY $OC/ledger/oc_checkpoint.py --root $OC record-review --checkpoint-id $ID --review-json "$1" --confirm
    rc=$?; echo "=== confirm exit $rc"; [ $rc -eq 0 ] || exit $rc
    echo "head now $(head_field rid)  status $(head_field status)"
  fi
}
notes() {  # $1 = extra sentence
  local da db pa
  da=$(sha256sum "$D1" | cut -c1-16); db=$(sha256sum "$PD" | cut -c1-16); pa=$(sha256sum "$PA" | cut -c1-16)
  printf '%s' "Lane M5 RE-SEAL of the E1 review (record 0002, parent count 15 STATED as a placeholder). parent_useful_or_partial now comes from the MEASURED parent assessment $(basename "$PA") (file sha $pa..., column A 15 / B 12 of 15 scored, oc_assess WITHOUT --per-case, so an UPPER BOUND on the parent side too) about parent dossier $(basename "$PD") (file sha $db..., E2 run, 16,707 s, exit 0, 0 non-finite, preflight agree=True). The parent's A happens to equal the placeholder, so the verdict ceiling and status do not move; what changes is the SOURCE of the number (measured, not stated). Child: assessment $(basename "$A1") about dossier $(basename "$D1") (file sha $da...), column A 8 / B 1 of 15, form boxes ECHO 8 / CAP_RIDE 5 / EMPTY 2 / CONTENT 1 of 16, canonical_content_miss 0 BY CONSTRUCTION so this record cannot make the child bad or deletable. \
SURFACE DIFFERENCE (recorded, not a blocker -- owner policy 2026-10-06): both columns come from the SAME runner (run_omni_frozen_review 1.1.0, tool sha f42cc50f..., chimera tree ef1ffb9c, same bank v1 file 0ee6bf14..., greedy, thinking off, cap 1024, same tokenizer sha 48bfd63e...), but the PARENT was decoded in float16 on three Tesla P40 (sm61, split 16/24/24, torch 2.6.0+cu118 Pascal venv, host-staged moves because P40 P2P copies are silently corrupt; parent import commit c5e6052d) and the CHILD in bfloat16 on two RTX 3090 (sm86, split 28/36, torch 2.11.0+cu128). Lane-level differences between the columns may partly be dtype/hardware, not fold. The child dossier is from the NATIVE export, tied to the registered HF directory only by the exactness receipts (CPU fp32 argmax 208/210 max logit diff 0.1224; GPU bf16 201/210, 0.5312). \
No human has read either sheet. $1"
}

echo "=== M5 $STAGE $CONFIRM  ($(now))"
echo "head record $(head_field rid)  reviewer '$(head_field reviewer)'  status $(head_field status)"
sha256sum "$D1" "$PD" "$PA" "$OC/calibration/oc_assess.py" "$OC/calibration/oc_review_upper_bound.py" \
          "$OC/ledger/oc_checkpoint.py" "$OC/ledger/oc_package.py" "$OC/ledger/oc_dashboard.py" "$0"

case "$STAGE" in
assess)
  HEAD=$(head_field rid)
  case "$HEAD" in
    $ID-0002-review) ;;
    $ID-0003-review) [ -f "$REV1" ] && [ "$CONFIRM" = "--confirm" ] && die "0003 already sealed; run 'package'" 0 ;;
    *) die "head is $HEAD, expected $ID-0002-review (E1's placeholder review)";;
  esac
  if [ ! -f "$A1" ]; then
    echo "--- oc_assess --parent-dossier"
    timeout 280 $N $PY $OC/calibration/oc_assess.py --dossier "$D1" --parent-dossier "$PD" --bank "$BANK" \
      --out-json "$A1" --out-markdown "$S1" || die "oc_assess exited $?" 1
  else
    echo "    $A1 exists; not rewritten"
  fi
  sha256sum "$A1" "$S1" $OC/calibration/out/assessment-OC02-C2K4352-NATIVE-NOTHINK.json \
            $OC/calibration/out/assessment-OC02-C2K4352-NATIVE-NOTHINK-vs-PARENT.md
  grep -c "Parent on the same prompt" "$S1" | sed 's/^/    lanes with a parent response in the sheet: /'
  if [ "$(sha256sum < "$A1" | cut -c1-64)" = "fb3cd4fe75c3f3988aacc15565cfa9ba0b25d59f804efcaae55710946a0fc3f9" ]; then
    echo "    assessment JSON byte-identical to E1/E2 (fb3cd4fe...): --parent-dossier only adds the sheet's parent column"
  else
    echo "    NOTE: assessment JSON differs from E1/E2 fb3cd4fe..."
  fi
  if [ ! -f "$REV1" ]; then
    echo "--- oc_review_upper_bound --parent-assessment (bare, no package yet)"
    timeout 280 $N $PY $OC/calibration/oc_review_upper_bound.py --assessment "$A1" --dossier "$D1" \
      --parent-assessment "$PA" --reviewer "$REVIEWER" \
      --notes "$(notes 'Bare re-seal (no package digest yet); the next record adds package_path/package_sha256.')" \
      --out "$REV1" || die "upper bound exited $?" 1
  else
    echo "    $REV1 exists; not rewritten"
  fi
  sha256sum "$REV1"
  $PY -c 'import json,sys;b=json.load(open(sys.argv[1]))["bank1"];print("bank1.parent_useful_or_partial",b["parent_useful_or_partial"],"| is_upper_bound",b.get("parent_useful_or_partial_is_upper_bound"),"| source:",b.get("parent_useful_or_partial_source","")[:120])' "$REV1"
  [ "$HEAD" = "$ID-0002-review" ] || die "head already moved to $HEAD; not recording twice" 0
  record "$REV1"
  ;;
package)
  [ "$(head_field rid)" = "$ID-0003-review" ] || die "head is $(head_field rid), expected $ID-0003-review"
  case "$(head_field reviewer)" in claude-lane-m5*) ;; *) die "head reviewer is not lane M5";; esac
  if [ ! -f "$NOTICE" ]; then
    {
      echo "# UPPER-BOUND NOTICE -- $ID (lane M5 re-seal, $(now))"
      echo
      echo "Every 'sealed' number for this checkpoint in the ledger, SCORECARD.md and REVIEW_QUEUE.md is the"
      echo "MACHINE CEILING from oc_review_upper_bound.py over an assessment run WITHOUT --per-case. Nobody has"
      echo "read this child or its parent. 'sealed 8/15 vs parent 15/15' means at most 8 scored lanes could hold"
      echo "their case; the parent's 15 is the parent's own ceiling (column A) from the same translation, now"
      echo "MEASURED (E2, $(basename "$PD")) instead of the E1 placeholder. canonical_content_miss is 0 by"
      echo "construction, so nothing in this package can make the child bad."
      echo
      echo "## Surfaces (same runner, different dtype/hardware)"
      echo
      echo "- child: $(basename "$D1"), bfloat16, 2x RTX 3090 sm86, split 28/36, torch 2.11.0+cu128, native export of the S3 trunk"
      echo "- parent: $(basename "$PD"), float16, 3x Tesla P40 sm61, split 16/24/24, torch 2.6.0+cu118, host-staged moves"
      echo "- shared: run_omni_frozen_review 1.1.0, chimera tree ef1ffb9c, bank v1 0ee6bf14..., greedy, thinking off, cap 1024"
      echo "- the dtype/hardware difference is a recorded diagnostic, not a gate (owner policy 2026-10-06)"
      echo
      echo "## To read"
      echo
      echo "$(basename "$S1") carries the child's verbatim text with the parent's response beside each lane."
      echo "Cyril's per-case file against that sheet is the reading that replaces these ceilings."
    } > "$NOTICE" || die "cannot write $NOTICE"
  fi
  if [ -f "$PKG/MANIFEST.json" ]; then
    echo "--- $PKG already built (a previous run of this stage); verifying, never rebuilding"
  else
    [ -e "$PKG" ] && die "$PKG exists without MANIFEST.json; this lane never writes over a package"
    mkdir -p "$(dirname "$PKG")"
    echo "--- oc_package build -> $PKG"
    timeout 280 $N $PY $OC/ledger/oc_package.py --root $OC build --checkpoint-id $ID --out "$PKG" --dossier "$D1" \
      --extra "$S1" --extra "$A1" --extra "$PD" --extra "$PA" --extra "$PAMD" --extra "$NOTICE" --extra "$REV1" \
      --extra "$RX1" --extra "$RX2" || die "oc_package build exited $?" 1
  fi
  timeout 280 $N $PY $OC/ledger/oc_package.py verify --package "$PKG" || die "verify exited $?" 1
  PSHA=$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["package_sha256"])' "$PKG/MANIFEST.json")
  echo "package_sha256 $PSHA"
  if [ ! -f "$REV2" ]; then
    timeout 280 $N $PY $OC/calibration/oc_review_upper_bound.py --assessment "$A1" --dossier "$D1" \
      --parent-assessment "$PA" --reviewer "$REVIEWER" \
      --notes "$(notes "Packaged: human-review package $PKG (package_sha256 $PSHA) carries the vs-parent scoring sheet, both assessments, the parent dossier, this notice and the exactness receipts.")" \
      --package-path "$PKG" --package-sha256 "$PSHA" --out "$REV2" || die "upper bound exited $?" 1
  fi
  sha256sum "$REV2" "$NOTICE" "$PKG/MANIFEST.json"
  record "$REV2"
  ;;
dashboard)
  T=$(date -u +%Y%m%dT%H%M%SZ)
  cp -p $OC/REVIEW_QUEUE.md $OC/REVIEW_QUEUE.md.bak-m5-$T; cp -p $OC/SCORECARD.md $OC/SCORECARD.md.bak-m5-$T
  timeout 280 $N $PY $OC/ledger/oc_dashboard.py --root $OC || die "dashboard exited $?" 1
  sha256sum $OC/REVIEW_QUEUE.md $OC/SCORECARD.md
  grep -n "$ID" $OC/REVIEW_QUEUE.md $OC/SCORECARD.md
  echo "--- list"
  timeout 280 $N $PY $OC/ledger/oc_checkpoint.py --root $OC list | grep -E "$ID|^checkpoint|^ *id" | head
  echo "--- ledger digest"
  timeout 280 $N $PY $OC/ledger/oc_ledger.py digest --ledger $OC/ledger.jsonl
  echo "--- verify"
  timeout 280 $N $PY $OC/ledger/oc_checkpoint.py --root $OC verify | tail -3
  ;;
*) die "stage must be assess | package | dashboard" 2;;
esac
echo "=== done ($(now))"
