#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# wavea_review_all.sh -- run wavea_review.sh over every unreviewed Wave A child.
#
# It walks `oc_checkpoint.py list`, keeps the `OC01-F-Q35-4B-*` FOLD checkpoints
# that carry no review record, and reviews them ONE AT A TIME on one card.
#
# WHY SERIAL, AND WHY IT WAITS RATHER THAN REFUSING.  Two reviews sharing a
# 24 GiB card is how a dossier ends up measuring contention instead of a child,
# and `run_r7.sh` already refuses that case for a single run.  A queue cannot
# refuse -- the whole point is that it survives a throughput probe holding the
# card for an hour -- so it POLLS instead, every $POLL seconds, and says each
# time how much is in use and by whom.  Nothing starts while the card is above
# --min-free-used MiB.
#
# ON FAILURE IT STOPS.  A failure on child one is almost never specific to child
# one (an OOM, a missing parent assessment, a moved bank digest), and burning
# the rest of the queue against the same fault produces a pile of half-written
# logs and no diagnosis.  `--keep-going` overrides that; the exit code is
# nonzero either way if anything failed.  Nothing is ever deleted.
#
# Usage:
#   ./wavea_review_all.sh [--gpu N] [--root DIR] [--lane DIR] [--poll SECONDS]
#                         [--min-free-used MiB] [--max-wait SECONDS]
#                         [--prefix OC01-F-Q35-4B-] [--keep-going] [--dry-run]
#                         [-- <extra args passed to wavea_review.sh>]
#
# Example:
#   ./wavea_review_all.sh --gpu 4
#   ./wavea_review_all.sh --gpu 4 --dry-run        # print the queue, run nothing
# ---------------------------------------------------------------------------
set -u

GPU=4
LANE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${OC_ROOT:-/home/cereal/omni-campaign}"
POLL=120
MIN_FREE_USED=2000
MAX_WAIT=86400
PREFIX="OC01-F-Q35-4B-"
KEEP_GOING=0
DRY_RUN=0
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)            GPU="${2:?--gpu needs an index}"; shift 2 ;;
    --root)           ROOT="${2:?--root needs a directory}"; shift 2 ;;
    --lane)           LANE="${2:?--lane needs a directory}"; shift 2 ;;
    --poll)           POLL="${2:?--poll needs seconds}"; shift 2 ;;
    --min-free-used)  MIN_FREE_USED="${2:?--min-free-used needs a number}"; shift 2 ;;
    --max-wait)       MAX_WAIT="${2:?--max-wait needs seconds}"; shift 2 ;;
    --prefix)         PREFIX="${2:?--prefix needs a string}"; shift 2 ;;
    --keep-going)     KEEP_GOING=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --)               shift; EXTRA=("$@"); break ;;
    -h|--help)        sed -n '2,33p' "$0"; exit 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PY="${PY:-/home/cereal/model-surgery-lab/venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
LOGS="$LANE/logs"
mkdir -p "$LOGS" || exit 1
LOG="$LOGS/review-all-$(date -u +%Y%m%dT%H%M%SZ).log"

now () { date -u +%Y-%m-%dT%H:%M:%SZ; }
say () { printf '[%s] %s\n' "$(now)" "$*" | tee -a "$LOG"; }
die () { printf '[%s] FAILED: %s\n' "$(now)" "$*" | tee -a "$LOG" >&2; exit "${2:-1}"; }

say "=================================================================="
say "wavea_review_all.sh  gpu=$GPU  root=$ROOT  prefix=$PREFIX"
say "poll=${POLL}s  min_free_used=${MIN_FREE_USED}MiB  max_wait=${MAX_WAIT}s"
say "log $LOG"
say "=================================================================="

# ---------------------------------------------------------------------------
# the queue, from the ledger.  `list --stage fold --json` carries `sealed` and
# `canonical` copied off the current record's bank1 block, and BOTH are null
# exactly when no review has been recorded -- so "the review is missing" is read
# off the same rows a human would read, not inferred from a filename.  A deleted
# checkpoint is dropped; its weights are gone and there is nothing to decode.
# ---------------------------------------------------------------------------
QUEUE_JSON="$LOGS/.wavea-queue.json"
timeout 300 "$PY" "$ROOT/ledger/oc_checkpoint.py" --root "$ROOT" list \
  --stage fold --json > "$QUEUE_JSON" 2>>"$LOG" \
  || die "oc_checkpoint.py list --stage fold --json refused; see $LOG" 3

QUEUE_TXT="$LOGS/.wavea-queue.txt"
QUEUE_ERR="$LOGS/.wavea-queue.err"
timeout 120 "$PY" - "$QUEUE_JSON" "$PREFIX" > "$QUEUE_TXT" 2>"$QUEUE_ERR" <<'PYQUEUE'
import json, sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prefix = sys.argv[2]
for row in rows:
    cid = str(row.get("checkpoint_id", ""))
    if not cid.startswith(prefix):
        print(f"OTHER-PREFIX {cid}", file=sys.stderr)
        continue
    if row.get("deleted"):
        print(f"DELETED      {cid}", file=sys.stderr)
        continue
    # `sealed` and `canonical` are copied off the current record's bank1 block,
    # so both are null exactly when no review has been recorded.
    if row.get("sealed") is not None or row.get("canonical") is not None:
        print(f"REVIEWED     {cid} status={row.get('status')} "
              f"sealed={row.get('sealed')} canonical={row.get('canonical')}",
              file=sys.stderr)
        continue
    print(cid)
    print(f"TO REVIEW    {cid} status={row.get('status')} "
          f"ratio={row.get('net_ratio')}", file=sys.stderr)
PYQUEUE
rc=$?
[ $rc -eq 0 ] || die "could not read the fold rows out of $QUEUE_JSON (exit $rc)" 3
sed 's/^/    /' "$QUEUE_ERR" | tee -a "$LOG"
mapfile -t QUEUE < "$QUEUE_TXT"

say "queue: ${#QUEUE[@]} unreviewed $PREFIX* fold checkpoint(s)"
for cid in ${QUEUE[@]+"${QUEUE[@]}"}; do say "    $cid"; done
if [ "${#QUEUE[@]}" -eq 0 ]; then
  say "nothing to do"
  exit 0
fi
if [ "$DRY_RUN" -eq 1 ]; then
  say "--dry-run: the queue above is what would run, in this order.  Stopping."
  exit 0
fi

# ---------------------------------------------------------------------------
# the card.  Checked BEFORE EVERY CHILD, not once: a throughput probe that
# lands between two children owns the card for the second one just as much.
# ---------------------------------------------------------------------------
[ "$GPU" != "0" ] || die "GPU0 is the build queue and is never used by this script" 3
GPUNAME=$(timeout 120 nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" 2>>"$LOG") \
  || die "nvidia-smi could not read GPU $GPU" 3
case "$GPUNAME" in
  *P40*) die "GPU $GPU is a $GPUNAME; sm61 is unsupported by this torch build" 3 ;;
esac
say "card: GPU $GPU is a $GPUNAME"

wait_for_card () {
  local waited=0 used
  while : ; do
    used=$(timeout 120 nvidia-smi --query-gpu=memory.used \
           --format=csv,noheader,nounits -i "$GPU" 2>>"$LOG") \
      || { say "    nvidia-smi failed; retrying in ${POLL}s"; used=999999; }
    if [ "$used" -le "$MIN_FREE_USED" ]; then
      say "    GPU $GPU is free (${used} MiB used, limit ${MIN_FREE_USED}) after ${waited}s"
      return 0
    fi
    if [ "$waited" -ge "$MAX_WAIT" ]; then
      say "    GPU $GPU still busy (${used} MiB) after ${waited}s; giving up"
      return 1
    fi
    say "    GPU $GPU busy: ${used} MiB used (limit ${MIN_FREE_USED}); waited ${waited}s"
    timeout 120 nvidia-smi --query-compute-apps=pid,used_memory,process_name \
      --format=csv,noheader 2>/dev/null | sed 's/^/        /' | tee -a "$LOG"
    sleep "$POLL"
    waited=$((waited + POLL))
  done
}

DONE=0
FAILED=0
FAILED_IDS=""
for cid in "${QUEUE[@]}"; do
  say "------------------------------------------------------------------"
  say "child $cid"
  if ! wait_for_card; then
    FAILED=$((FAILED + 1)); FAILED_IDS="$FAILED_IDS $cid"
    say "    NOT STARTED: the card never came free inside --max-wait"
    [ "$KEEP_GOING" -eq 1 ] || break
    continue
  fi
  say "    launching wavea_review.sh $cid --gpu $GPU"
  "$LANE/wavea_review.sh" "$cid" --gpu "$GPU" --root "$ROOT" --lane "$LANE" \
    --min-free-used "$MIN_FREE_USED" ${EXTRA[@]+"${EXTRA[@]}"}
  rc=$?
  if [ $rc -eq 0 ]; then
    DONE=$((DONE + 1))
    say "    $cid OK"
  else
    FAILED=$((FAILED + 1)); FAILED_IDS="$FAILED_IDS $cid"
    say "    $cid FAILED (exit $rc); its own log is $LOGS/review-$cid.log"
    if [ "$KEEP_GOING" -eq 0 ]; then
      say "    stopping: a fault on one child is usually a fault for all of them."
      say "    Re-run with --keep-going once you know it is child-specific."
      break
    fi
  fi
done

say "------------------------------------------------------------------"
say "reviewed $DONE, failed $FAILED${FAILED_IDS:+ ($FAILED_IDS )}"
say "queue was ${#QUEUE[@]}; log $LOG"
[ "$FAILED" -eq 0 ] || exit 1
exit 0
