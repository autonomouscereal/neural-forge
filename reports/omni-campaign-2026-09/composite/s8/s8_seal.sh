#!/bin/bash
# S8-TEXT seal (2026-10-13, Claude, S8 lane): assessment -> upper-bound review -> record-review -> package -> dashboard
# for the OMNI1 (OC02-F-Q38-27BU-OMNI1) and S5 (OC02-F-Q38-27BU-C2K4352K3U) bank-v1 dossiers written by launch_s8_text.sh.
# Same ordering as lane M5 (s3_reseal_oc02_m5.sh): oc_assess --parent-dossier (WITHOUT --per-case -> every count an
# UPPER BOUND, canonical_content_miss 0 by construction: artifact-tolerant, nothing here can make a child bad), the
# parent count from the MEASURED E2 parent assessment, record-review DRY RUN then --confirm, then the human-review
# package and a second record carrying its digest, then oc_dashboard (SCORECARD.md + REVIEW_QUEUE.md).  CPU only.
# Deletes nothing; refuses to write over an existing output.
#
# Usage:  s8_seal.sh omni1|s5 assess    [--confirm]
#         s8_seal.sh omni1|s5 package   [--confirm]
#         s8_seal.sh any       dashboard
set -u
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
export OC_ROOT=$OC OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
N="nice -n 19"
WHICH=${1:?omni1|s5}; STAGE=${2:-}; CONFIRM=${3:-}
case "$WHICH" in
  omni1) ID=OC02-F-Q38-27BU-OMNI1; LABEL=OC02-OMNI1-NATIVE-NOTHINK; SPLIT="6/33/30 (GPU6 head + blocks 0-5; GPU4 blocks 6-38 incl. the 5 charted blocks 32-36; GPU0 blocks 39-68)"
         MODEL=/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1; RX=$OC/composite/s7/omni1/omni1_check.json ;;
  s5)    ID=OC02-F-Q38-27BU-C2K4352K3U; LABEL=OC02-C2K4352K3U-NATIVE-NOTHINK; SPLIT="6/28/30 (same trunk placement as OMNI1)"
         MODEL=$OC/composite/c10/S5-C2K4352K3U/child; RX=$OC/composite/c10/S5-C2K4352K3U/vcheck_written_child.json ;;
  any)   ID=""; LABEL="" ;;
  *) echo "usage: $0 omni1|s5 assess|package [--confirm] | $0 any dashboard"; exit 2 ;;
esac
S=$OC/composite/s8/review
D1=$S/oc-dossier-$LABEL.json
G1=$D1.finite_guard.json
BANK=$OC/calibration/banks/lfm2_semantic_whole_response_prompts_v1.json
PD=$OC/composite/s0/review/oc-dossier-PARENT-Q38-27BU-NATIVE-NOTHINK.json
PA=$OC/calibration/out/assessment-PARENT-Q38-27BU-NATIVE-NOTHINK.json
PAMD=$OC/calibration/out/assessment-PARENT-Q38-27BU-NATIVE-NOTHINK.md
A1=$OC/calibration/out/assessment-$LABEL-vs-PARENT-s8.json
S1=$OC/calibration/out/SCORING-SHEET-$LABEL-vs-PARENT-s8.md
REV1=$S/review-$ID-s8.json
REV2=$S/review-$ID-s8-packaged.json
NOTICE=$S/UPPER-BOUND-NOTICE-$ID-s8.md
PKG=$OC/oc02/human-review/$ID
REVIEWER="claude-lane-s8 (oc_review_upper_bound.py over the S8-TEXT native dossier; parent = measured E2 PARENT-Q38-27BU assessment)"
die() { echo "=== REFUSED: $1"; exit "${2:-3}"; }
now() { date -u +%FT%TZ; }
head_field() {
  timeout 120 $N $PY $OC/ledger/oc_checkpoint.py --root $OC show --checkpoint-id $ID 2>/dev/null \
  | $PY -c 'import json,sys
d=json.load(sys.stdin); r=d.get("review") or {}
v={"rid":d.get("record_instance_id"),"reviewer":r.get("reviewer"),"status":d.get("status")}.get(sys.argv[1])
print(v if v is not None else "")' "$1"
}
record() {
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
notes() {
  local da db pa g
  da=$(sha256sum "$D1" | cut -c1-16); db=$(sha256sum "$PD" | cut -c1-16); pa=$(sha256sum "$PA" | cut -c1-16)
  g=$($PY -c 'import json,sys;s=json.load(open(sys.argv[1]))["summary"];print(f"{s[\"segments\"]} segments, {s[\"segments_nonfinite\"]} non-finite")' "$G1" 2>/dev/null || echo "guard receipt unread")
  printf '%s' "Lane S8-TEXT: machine UPPER BOUND, not a reading. oc_review_upper_bound.py over $(basename "$A1") (oc_assess WITHOUT --per-case, columns.is_upper_bound true) about dossier $(basename "$D1") (file sha $da...); parent_useful_or_partial from the MEASURED parent assessment $(basename "$PA") (file sha $pa..., column A 15 / B 12) about $(basename "$PD") (file sha $db...). canonical_content_miss is 0 BY CONSTRUCTION (artifact-tolerant: loops, echo, cap-rides, missing EOS are artifacts, never content misses), so this record cannot make the checkpoint bad or deletable. \
SURFACE: decoded on $MODEL with run_omni_frozen_review 1.2.0 (charted blocks accepted on the free-run TEXT path only; identity on text, the tree's own cached-trunk rule) under run_review_finite_guard 1.2.0 ($g, host-staged moves), tree ef1ffb9c, bf16, split $SPLIT over Quadro RTX 8000 (sm75, head; in SW thermal slowdown) + 2x RTX 3090 (sm86), thinking off, --ledger-band-logit-mask (REQUIRED: OMNI1's C18 rows in lm_head win the argmax at 146/210 R-Q positions unmasked), cap 1024, greedy, bank v1 0ee6bf14..., tokenizer = the S3 dossier's (48bfd63e...). Same runner and flags as the parent (E2: fp16, 3x P40) and S3 (bf16, 2x RTX 3090) dossiers; dtype/hardware differ and are recorded as diagnostics (owner policy 2026-10-06). \
No human has read this sheet. $1"
}

echo "=== S8 $WHICH $STAGE $CONFIRM  ($(now))"
case "$STAGE" in
assess|package)
  [ -f "$D1" ] || die "no dossier $D1"
  echo "head record $(head_field rid)  reviewer '$(head_field reviewer)'  status $(head_field status)"
  sha256sum "$D1" "$G1" "$PD" "$PA" "$OC/calibration/oc_assess.py" "$OC/calibration/oc_review_upper_bound.py" \
            "$OC/ledger/oc_checkpoint.py" "$OC/ledger/oc_package.py" "$0" ;;
esac

case "$STAGE" in
assess)
  HEAD=$(head_field rid)
  case "$HEAD" in
    $ID-0001-register) ;;
    *) die "head is $HEAD, expected $ID-0001-register";;
  esac
  if [ ! -f "$A1" ]; then
    echo "--- oc_assess --parent-dossier"
    timeout 280 $N $PY $OC/calibration/oc_assess.py --dossier "$D1" --parent-dossier "$PD" --bank "$BANK" \
      --out-json "$A1" --out-markdown "$S1" || die "oc_assess exited $?" 1
  else
    echo "    $A1 exists; not rewritten"
  fi
  sha256sum "$A1" "$S1"
  $PY -c 'import json,sys;c=json.load(open(sys.argv[1]))["columns"];print("columns", json.dumps({k:c[k] for k in c if not isinstance(c[k],(list,dict))}))' "$A1"
  if [ ! -f "$REV1" ]; then
    echo "--- oc_review_upper_bound --parent-assessment (bare, no package yet)"
    timeout 280 $N $PY $OC/calibration/oc_review_upper_bound.py --assessment "$A1" --dossier "$D1" \
      --parent-assessment "$PA" --reviewer "$REVIEWER" \
      --notes "$(notes 'Bare record (no package digest yet); the next record adds package_path/package_sha256.')" \
      --out "$REV1" || die "upper bound exited $?" 1
  else
    echo "    $REV1 exists; not rewritten"
  fi
  sha256sum "$REV1"
  $PY -c 'import json,sys;b=json.load(open(sys.argv[1]))["bank1"];print("bank1", json.dumps({k:b[k] for k in ("useful","partial","observed_miss","canonical_signal","canonical_artifact_miss","canonical_content_miss","parent_useful_or_partial","sealed_verdict","lanes_total") if k in b}))' "$REV1"
  record "$REV1"
  ;;
package)
  [ "$(head_field rid)" = "$ID-0002-review" ] || die "head is $(head_field rid), expected $ID-0002-review"
  case "$(head_field reviewer)" in claude-lane-s8*) ;; *) die "head reviewer is not lane S8";; esac
  if [ ! -f "$NOTICE" ]; then
    {
      echo "# UPPER-BOUND NOTICE -- $ID (lane S8-TEXT, $(now))"
      echo
      echo "Every 'sealed' number for this checkpoint in the ledger, SCORECARD.md and REVIEW_QUEUE.md is the MACHINE"
      echo "CEILING from oc_review_upper_bound.py over an assessment run WITHOUT --per-case. Nobody has read this"
      echo "checkpoint's text. The parent's count is the parent's own ceiling (column A 15 / B 12), measured (E2)."
      echo "canonical_content_miss is 0 by construction, so nothing in this package can make the checkpoint bad."
      echo
      echo "## Surface"
      echo
      echo "- $(basename "$D1"): bf16, Quadro RTX 8000 (sm75, head) + 2x RTX 3090 (sm86), split $SPLIT, torch 2.11.0+cu128"
      echo "- run_omni_frozen_review 1.2.0 + run_review_finite_guard 1.2.0, chimera tree ef1ffb9c, bank v1 0ee6bf14..., greedy, thinking off, cap 1024"
      echo "- --ledger-band-logit-mask ON (OMNI1's C18 music band rows win the argmax unmasked: 146/210 R-Q positions)"
      echo "- parent: $(basename "$PD") (fp16, 3x P40); S3: oc-dossier-OC02-C2K4352-NATIVE-NOTHINK.json (bf16, 2x RTX 3090)"
      echo
      echo "## To read"
      echo
      echo "$(basename "$S1") carries this checkpoint's verbatim text with the parent's response beside each lane."
    } > "$NOTICE" || die "cannot write $NOTICE"
  fi
  if [ -f "$PKG/MANIFEST.json" ]; then
    echo "--- $PKG already built; verifying, never rebuilding"
  else
    [ -e "$PKG" ] && die "$PKG exists without MANIFEST.json; this lane never writes over a package"
    mkdir -p "$(dirname "$PKG")"
    echo "--- oc_package build -> $PKG"
    timeout 280 $N $PY $OC/ledger/oc_package.py --root $OC build --checkpoint-id $ID --out "$PKG" --dossier "$D1" \
      --extra "$S1" --extra "$A1" --extra "$PD" --extra "$PA" --extra "$PAMD" --extra "$NOTICE" --extra "$REV1" \
      --extra "$G1" --extra "$RX" || die "oc_package build exited $?" 1
  fi
  timeout 280 $N $PY $OC/ledger/oc_package.py verify --package "$PKG" || die "verify exited $?" 1
  PSHA=$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["package_sha256"])' "$PKG/MANIFEST.json")
  echo "package_sha256 $PSHA"
  if [ ! -f "$REV2" ]; then
    timeout 280 $N $PY $OC/calibration/oc_review_upper_bound.py --assessment "$A1" --dossier "$D1" \
      --parent-assessment "$PA" --reviewer "$REVIEWER" \
      --notes "$(notes "Packaged: human-review package $PKG (package_sha256 $PSHA) carries the vs-parent scoring sheet, the assessment, the parent dossier + assessment, this notice, the finite-guard receipt and the load/identity check.")" \
      --package-path "$PKG" --package-sha256 "$PSHA" --out "$REV2" || die "upper bound exited $?" 1
  fi
  sha256sum "$REV2" "$NOTICE" "$PKG/MANIFEST.json"
  record "$REV2"
  ;;
dashboard)
  T=$(date -u +%Y%m%dT%H%M%SZ)
  cp -p $OC/REVIEW_QUEUE.md $OC/REVIEW_QUEUE.md.bak-s8-$T; cp -p $OC/SCORECARD.md $OC/SCORECARD.md.bak-s8-$T
  timeout 280 $N $PY $OC/ledger/oc_dashboard.py --root $OC || die "dashboard exited $?" 1
  sha256sum $OC/REVIEW_QUEUE.md $OC/SCORECARD.md
  grep -n "OMNI1\|C2K4352K3U" $OC/REVIEW_QUEUE.md $OC/SCORECARD.md | cut -c1-400
  echo "--- verify"
  timeout 280 $N $PY $OC/ledger/oc_checkpoint.py --root $OC verify | tail -3
  ;;
*) die "stage must be assess | package | dashboard" 2;;
esac
echo "=== done ($(now))"
