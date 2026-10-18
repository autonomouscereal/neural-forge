#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# wavea_review_all.sh -- run wavea_review.sh over every unreviewed Wave A child.
#
# It walks `oc_checkpoint.py list`, keeps the `OC01-F-Q35-4B-*` FOLD checkpoints
# whose review is missing OR whose review was sealed but never packaged, and
# reviews them ONE AT A TIME on one card.  The ledger is RE-LISTED after every
# child, so a child that wavea_queue.sh registers on GPU0 while a review is in
# flight on GPU4 is picked up by the same run rather than silently left out.
#
# WHY SERIAL, AND WHY IT WAITS RATHER THAN REFUSING.  Two reviews sharing a
# 24 GiB card is how a dossier ends up measuring contention instead of a child,
# and `run_r7.sh` already refuses that case for a single run.  A queue cannot
# refuse -- the whole point is that it survives a throughput probe holding the
# card for an hour -- so it POLLS instead, every $POLL seconds, and says each
# time how much is in use and by whom.  Nothing starts while the card is above
# --min-free-used MiB.
#
# WHAT IT CHECKS BEFORE TOUCHING THE CARD.  The campaign python, the four lane
# scripts, the parent NOTHINK assessment, the pinned bank v1 digest, the bank
# v2 draft (and whether its .sha256 sidecar is present), stale ledger writer
# locks under <registry>/.locks left by a dead pid (reported, NEVER removed),
# and that no other review_all / wavea_review.sh / oc_run_frozen_bank.py is
# alive.  Each of those would otherwise cost a full GPU pass before failing.
#
# ON FAILURE IT STOPS.  A failure on child one is almost never specific to child
# one (an OOM, a missing parent assessment, a moved bank digest), and burning
# the rest of the queue against the same fault produces a pile of half-written
# logs and no diagnosis.  `--keep-going` overrides that; the exit code is
# nonzero either way if anything failed, and nonzero if the final re-list still
# shows an unreviewed child.  Nothing is ever deleted; every run's log and
# ledger snapshots carry their own UTC stamp and are never overwritten.
#
# ONLY REVIEW FLAGS ARE FORWARDED.  `-- <extra args>` may carry `--reviewer
# NAME` and `--skip-bank2` and nothing else: the mechanics flags
# (`--tiny-random`, `--max-new-tokens`, `--max-cases`) would be sealed with
# `--confirm` into the authority registry under every real checkpoint id, and
# the overrides (`--root`, `--gpu`, `--lane`, `--min-free-used`) belong to this
# script.  They are refused unless `--allow-mechanics-args` is given AND --root
# is not the live campaign root.
#
# Usage:
#   bash wavea_review_all.sh [--gpu N] [--root DIR] [--lane DIR] [--poll SECONDS]
#                            [--min-free-used MiB] [--max-wait SECONDS]
#                            [--parent-code Q35-4B] [--prefix OC01-F-Q35-4B-]
#                            [--keep-going] [--dry-run] [--allow-mechanics-args]
#                            [-- --reviewer NAME] [-- --skip-bank2]
#
# Launch DETACHED (it blocks for the whole wave -- 14 children x >= 15 min plus
# up to --max-wait of polling -- and must never ride the single SSH lease):
#   cd /home/cereal/omni-campaign/calibration && \
#     setsid nohup bash ./wavea_review_all.sh --gpu 4 < /dev/null \
#       > logs/review-all.out 2>&1 & sleep 3; echo started
# Progress is read from logs/review-all-<stamp>.log (this queue) and
# logs/review-<ID>.log (each child); the ledger rows the queue was built from
# are kept beside it as logs/review-all-<stamp>.queue-<n>.json.
#
# Plan only, touching neither the card nor the ledger:
#   bash wavea_review_all.sh --gpu 4 --dry-run
# ---------------------------------------------------------------------------
set -u

GPU=4
LANE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${OC_ROOT:-/home/cereal/omni-campaign}"
LIVE_ROOT="/home/cereal/omni-campaign"
POLL=120
MIN_FREE_USED=2000
MAX_WAIT=86400
PARENT_CODE="Q35-4B"
PREFIX=""
KEEP_GOING=0
DRY_RUN=0
ALLOW_MECH=0
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)                  GPU="${2:?--gpu needs an index}"; shift 2 ;;
    --root)                 ROOT="${2:?--root needs a directory}"; shift 2 ;;
    --lane)                 LANE="${2:?--lane needs a directory}"; shift 2 ;;
    --poll)                 POLL="${2:?--poll needs seconds}"; shift 2 ;;
    --min-free-used)        MIN_FREE_USED="${2:?--min-free-used needs a number}"; shift 2 ;;
    --max-wait)             MAX_WAIT="${2:?--max-wait needs seconds}"; shift 2 ;;
    --parent-code)          PARENT_CODE="${2:?--parent-code needs a code}"; shift 2 ;;
    --prefix)               PREFIX="${2:?--prefix needs a string}"; shift 2 ;;
    --keep-going)           KEEP_GOING=1; shift ;;
    --dry-run)              DRY_RUN=1; shift ;;
    --allow-mechanics-args) ALLOW_MECH=1; shift ;;
    --)                     shift; EXTRA=("$@"); break ;;
    -h|--help)              sed -n '2,60p' "$0"; exit 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$PREFIX" ] || PREFIX="OC01-F-$PARENT_CODE-"

# stamp + pid: two launches in the same second must not share a log or a
# queue snapshot (the test run showed three back-to-back dry runs colliding).
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOGS="$LANE/logs"
mkdir -p "$LOGS" || exit 1
LOG="$LOGS/review-all-$STAMP.log"

now () { date -u +%Y-%m-%dT%H:%M:%SZ; }
say () { printf '[%s] %s\n' "$(now)" "$*" | tee -a "$LOG"; }
die () { printf '[%s] FAILED: %s\n' "$(now)" "$*" | tee -a "$LOG" >&2; exit "${2:-1}"; }

# ---------------------------------------------------------------------------
# the forwarded flags.  Every EXTRA arg goes to EVERY child with --confirm at
# the end of it, so a mechanics flag here is a random-init or lowered-cap review
# sealed in the authority registry under a real id -- and the tiny outputs then
# sit on the exact paths the real run would SKIP on.  Refused by name.
# ---------------------------------------------------------------------------
ROOT_REAL=$(readlink -f "$ROOT" 2>/dev/null || echo "$ROOT")
LIVE_REAL=$(readlink -f "$LIVE_ROOT" 2>/dev/null || echo "$LIVE_ROOT")
ROOT_IS_LIVE=0
if [ "$ROOT" = "$LIVE_ROOT" ] || [ "$ROOT_REAL" = "$LIVE_REAL" ]; then ROOT_IS_LIVE=1; fi
for arg in ${EXTRA[@]+"${EXTRA[@]}"}; do
  case "$arg" in
    --tiny-random|--max-new-tokens|--max-new-tokens=*|--max-cases|--max-cases=*|\
    --root|--root=*|--gpu|--gpu=*|--lane|--lane=*|--min-free-used|--min-free-used=*)
      if [ "$ALLOW_MECH" -eq 1 ] && [ "$ROOT_IS_LIVE" -eq 0 ]; then
        say "WARNING: forwarding mechanics/override flag '$arg' to every child because --allow-mechanics-args was given and --root $ROOT is not the live campaign root"
      elif [ "$ALLOW_MECH" -eq 1 ]; then
        die "'$arg' after -- is refused: --allow-mechanics-args does not apply when --root is the live campaign root ($LIVE_ROOT); a random-init or lowered-cap review would be sealed with --confirm under every real checkpoint id" 2
      else
        die "'$arg' after -- is refused: only --reviewer NAME and --skip-bank2 may be forwarded to wavea_review.sh (pass --allow-mechanics-args together with a throwaway --root to override)" 2
      fi ;;
  esac
done

# ---------------------------------------------------------------------------
# the parent gate.  --prefix alone would admit Qwen3.5-0.8B/2B/9B fold children,
# whose bank1.parent_useful_or_partial wavea_review.sh would then seal from the
# hard-coded Qwen3.5-4B parent assessment.  The prefix must carry the parent
# code (R1: `OC<NN>-<STAGE>-<PARENTCODE>-<TAG>`), and the queue filter below
# additionally full-matches every id against it regardless of --prefix.
# ---------------------------------------------------------------------------
case "$PREFIX" in
  *"-$PARENT_CODE-") : ;;
  *) die "--prefix '$PREFIX' does not end in '-$PARENT_CODE-'; a prefix that does not name the parent code admits other parents' children (pass --parent-code to change the parent)" 2 ;;
esac

say "=================================================================="
say "wavea_review_all.sh  gpu=$GPU  root=$ROOT  parent=$PARENT_CODE  prefix=$PREFIX"
say "poll=${POLL}s  min_free_used=${MIN_FREE_USED}MiB  max_wait=${MAX_WAIT}s  dry_run=$DRY_RUN  keep_going=$KEEP_GOING"
say "forwarded to wavea_review.sh: ${EXTRA[*]:-(nothing)}"
say "log $LOG"
say "=================================================================="

trap 'say "SIGHUP received; the running wavea_review.sh (if any) is left to finish"' HUP

# ---------------------------------------------------------------------------
# preflight -- everything the header names as a whole-queue fault, checked
# BEFORE any card time is spent.  Every miss is a die, not a warning.
# ---------------------------------------------------------------------------
say "--- preflight ---"
PY="${PY:-/home/cereal/model-surgery-lab/venv/bin/python}"
[ -x "$PY" ] || die "campaign python $PY is missing; refusing to fall back to $(command -v python3 || echo '(no python3)')" 3
say "    python            $PY"
for f in "$LANE/wavea_review.sh" "$LANE/oc_run_frozen_bank.py" "$LANE/oc_assess.py" \
         "$LANE/oc_review_upper_bound.py" "$ROOT/ledger/oc_checkpoint.py" \
         "$ROOT/ledger/oc_package.py" "$ROOT/ledger/oc_dashboard.py"; do
  [ -r "$f" ] || die "$f is missing or unreadable" 3
done
say "    lane scripts      present (wavea_review.sh is invoked through bash; no +x needed)"

PARENT_ASSESSMENT="$LANE/out/assessment-$PARENT_CODE-NOTHINK.json"
[ -s "$PARENT_ASSESSMENT" ] || die "parent NOTHINK assessment $PARENT_ASSESSMENT is absent; run the parent first (wavea_review.sh would only find this out after a full bank pass)" 3
say "    parent assessment $PARENT_ASSESSMENT"
[ -s "$LANE/out/oc-dossier-$PARENT_CODE-NOTHINK.json" ] \
  && say "    parent dossier    $LANE/out/oc-dossier-$PARENT_CODE-NOTHINK.json" \
  || say "    NOTE: parent dossier oc-dossier-$PARENT_CODE-NOTHINK.json absent; scoring sheets get no parent column"
if [ "$PARENT_CODE" != "Q35-4B" ]; then
  say "    NOTE: wavea_review.sh hard-codes its parent assessment to Q35-4B; with --parent-code $PARENT_CODE its review records would be sealed against the wrong parent"
  die "--parent-code $PARENT_CODE is not supported by the installed wavea_review.sh (it reads assessment-Q35-4B-NOTHINK.json)" 3
fi

V1="$LANE/banks/lfm2_semantic_whole_response_prompts_v1.json"
V1_SHA=0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b
[ -s "$V1" ] || die "bank v1 $V1 is absent" 3
printf '%s  %s\n' "$V1_SHA" "$V1" | sha256sum -c --quiet 2>>"$LOG" \
  || die "bank v1 digest moved: $V1 no longer hashes to $V1_SHA (wavea_review.sh pins that value)" 3
say "    bank v1           $V1  sha256 $V1_SHA  (verified)"
V2="$LANE/banks/omni_review_bank_v2_draft.json"
[ -s "$V2" ] || die "bank v2 draft $V2 is absent" 3
if [ -s "$V2.sha256" ]; then
  V2_SHA=$(awk '{print $1; exit}' "$V2.sha256")
  if printf '%s  %s\n' "$V2_SHA" "$V2" | sha256sum -c --quiet 2>>"$LOG"; then
    say "    bank v2 draft     $V2  sidecar present, sha256 $V2_SHA  (verified)"
  else
    die "bank v2 draft moved and its sidecar did not: $V2 does not hash to $V2_SHA; oc_run_frozen_bank.py would refuse every child" 3
  fi
else
  say "    bank v2 draft     $V2  NO .sha256 sidecar -- the v2 lanes will run UNPINNED (digest_was_pinned false)"
fi

# stale writer locks left by the power cut.  A surviving lock makes every
# record-review wait 60 s and then REFUSE; the ledger never breaks one and
# neither does this script.
LOCKDIR="$ROOT/registry/.locks"
STALE=0
if [ -d "$LOCKDIR" ]; then
  for lf in "$LOCKDIR"/*.lock; do
    [ -e "$lf" ] || continue
    owner=$(timeout 60 "$PY" -c 'import json,sys
try:
    d=json.load(open(sys.argv[1],encoding="utf-8")); print(d.get("pid",""), d.get("acquired_at",""), d.get("host",""))
except Exception as e:
    print("", "", f"unreadable:{e}")' "$lf" 2>>"$LOG")
    lpid=${owner%% *}; rest=${owner#* }; lat=${rest%% *}; lhost=${rest#* }
    if [ -n "$lpid" ] && [ -d "/proc/$lpid" ]; then
      say "    ledger lock       $lf  pid $lpid ALIVE  acquired $lat  host $lhost"
    else
      say "    ledger lock       $lf  pid ${lpid:-?} DEAD   acquired ${lat:-?}  host ${lhost:-?}  -- STALE"
      STALE=$((STALE + 1))
    fi
  done
  [ "$STALE" -eq 0 ] && say "    ledger locks      none stale under $LOCKDIR"
else
  say "    ledger locks      $LOCKDIR does not exist (no lock has ever been taken)"
fi
if [ "$STALE" -gt 0 ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    say "    --dry-run: a real run would REFUSE on the $STALE stale lock(s) above"
  else
    die "$STALE stale ledger writer lock(s) under $LOCKDIR (listed above) -- remove them by hand after confirming each pid is dead; this script never deletes" 3
  fi
fi

# one instance.  Mirrors wavea_queue.sh: a mkdir lock (its own directory is
# the only path this script ever removes) plus a refusal while any review
# process is alive.  A dry run takes no lock, as the build queue does not.
LOCK="$LOGS/.wavea-review-all.lock"
if [ "$DRY_RUN" -eq 0 ]; then
  if ! mkdir "$LOCK" 2>/dev/null; then
    die "another wavea_review_all.sh holds $LOCK ($(cat "$LOCK/owner" 2>/dev/null || echo '(owner unknown)')); if that pid is dead, rmdir it by hand" 2
  fi
  echo "pid $$ started $(now) log $LOG" > "$LOCK/owner"
  trap 'rm -f "$LOCK/owner"; rmdir "$LOCK" 2>/dev/null' EXIT
  busy=$(pgrep -f 'oc_run_frozen_bank\.py|wavea_review\.sh' | grep -vx "$$" | tr '\n' ' ')
  if [ -n "$busy" ]; then
    die "a review process is already alive (pids: $busy); refusing to start a second queue" 2
  fi
  say "    instance lock     $LOCK (pid $$)"
fi

# --package-root only if the installed wavea_review.sh advertises it; the
# R7 / CAMPAIGN.md location is $ROOT/oc01/human-review/<ID>, and the older
# wavea_review.sh writes $ROOT/human-review/<ID> with no way to redirect it.
PKG_ARGS=()
if grep -q -- '--package-root' "$LANE/wavea_review.sh"; then
  PKG_ARGS=(--package-root "$ROOT/oc01/human-review")
  say "    package root      $ROOT/oc01/human-review (passed as --package-root)"
else
  say "    NOTE: this wavea_review.sh has no --package-root; packages land at $ROOT/human-review/<ID>, not the R7 path $ROOT/oc01/human-review/<ID>"
fi

# ---------------------------------------------------------------------------
# the queue, from the ledger.  Built by build_queue, and REBUILT after every
# child: `list --stage fold --json` is a single read of the registry, and the
# build lane keeps registering while a review runs.
#
# `sealed`/`canonical` are copied off the current record's bank1 block and are
# both null exactly when NO review has been recorded -- but they are set by the
# FIRST (bare) record-review, before any package exists, so a reviewed row is
# not dropped on them alone: `show` is read for its `review.package_sha256`,
# and a child whose bare review was sealed but never packaged is enqueued again
# (wavea_review.sh resumes at its package step without re-decoding).  An
# `unmeasured` child is reported separately (R8: a re-run, outputs moved aside
# by hand) and never enqueued.  A deleted checkpoint is dropped; its weights
# are gone and there is nothing to decode.
# ---------------------------------------------------------------------------
PASS=0
EXCLUDE="$LOGS/review-all-$STAMP.handled.txt"
: > "$EXCLUDE"
QUEUE=()
build_queue () {
  PASS=$((PASS + 1))
  local qjson="$LOGS/review-all-$STAMP.queue-$PASS.json"
  local qtxt="$LOGS/review-all-$STAMP.queue-$PASS.txt"
  local qerr="$LOGS/review-all-$STAMP.queue-$PASS.err"
  timeout 300 "$PY" "$ROOT/ledger/oc_checkpoint.py" --root "$ROOT" list \
    --stage fold --json > "$qjson" 2>>"$LOG" \
    || die "oc_checkpoint.py list --stage fold --json refused; see $LOG" 3
  timeout 900 "$PY" - "$qjson" "$PREFIX" "$PARENT_CODE" "$PY" "$ROOT" "$EXCLUDE" \
    > "$qtxt" 2>"$qerr" <<'PYQUEUE'
import json, re, subprocess, sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prefix, parent, py, root, exclude = sys.argv[2:7]
handled = set(Path(exclude).read_text(encoding="utf-8").split())
child_re = re.compile(r"OC\d{2}-F-" + re.escape(parent) + r"-[A-Z0-9]{1,16}")
ledger = str(Path(root) / "ledger" / "oc_checkpoint.py")

def show(cid):
    proc = subprocess.run(
        [py, ledger, "--root", root, "show", "--checkpoint-id", cid],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise SystemExit(f"show --checkpoint-id {cid} exited {proc.returncode}: {proc.stderr.strip()[:300]}")
    return json.loads(proc.stdout)

for row in rows:
    cid = str(row.get("checkpoint_id", ""))
    if not child_re.fullmatch(cid):
        print(f"WRONG-PARENT {cid}", file=sys.stderr)
        continue
    if not cid.startswith(prefix):
        print(f"OTHER-PREFIX {cid}", file=sys.stderr)
        continue
    if row.get("deleted"):
        print(f"DELETED      {cid}", file=sys.stderr)
        continue
    if cid in handled:
        print(f"HANDLED      {cid} (already run this pass)", file=sys.stderr)
        continue
    status = row.get("status")
    if row.get("sealed") is None and row.get("canonical") is None:
        print(cid)
        print(f"TO REVIEW    {cid} status={status} ratio={row.get('net_ratio')}", file=sys.stderr)
        continue
    record = show(cid)
    review = record.get("review") or {}
    if not review.get("package_sha256"):
        print(cid)
        print(f"REVIEWED-UNPACKAGED {cid} status={status} (resuming at step 5)", file=sys.stderr)
        continue
    if status == "unmeasured":
        print(f"UNMEASURED   {cid} (re-run required, outputs must be moved aside by hand)", file=sys.stderr)
        continue
    print(f"REVIEWED     {cid} status={status} sealed={row.get('sealed')} "
          f"canonical={row.get('canonical')} package_sha256={str(review.get('package_sha256'))[:16]}",
          file=sys.stderr)
PYQUEUE
  local rc=$?
  sed 's/^/    /' "$qerr" | tee -a "$LOG"
  [ $rc -eq 0 ] || die "could not read the fold rows out of $qjson (exit $rc)" 3
  mapfile -t QUEUE < "$qtxt"
  say "queue (pass $PASS, from $qjson): ${#QUEUE[@]} child(ren) to review"
  for c in ${QUEUE[@]+"${QUEUE[@]}"}; do say "    $c"; done
}

build_queue
if [ "$DRY_RUN" -eq 1 ]; then
  say "--dry-run: the queue above is what would run, in this order (re-listed after each child).  Stopping."
  exit 0
fi
if [ "${#QUEUE[@]}" -eq 0 ]; then
  say "nothing to do"
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
    used=${used//[[:space:]]/}
    case "$used" in
      ''|*[!0-9]*) say "    nvidia-smi returned '$used' for memory.used; treating the card as busy"; used=999999 ;;
    esac
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
      --format=csv,noheader -i "$GPU" 2>/dev/null | sed 's/^/        /' | tee -a "$LOG"
    sleep "$POLL"
    waited=$((waited + POLL))
  done
}

DONE=0
FAILED=0
FAILED_IDS=""
STOPPED=0
while [ "${#QUEUE[@]}" -gt 0 ]; do
  cid="${QUEUE[0]}"
  echo "$cid" >> "$EXCLUDE"
  say "------------------------------------------------------------------"
  say "child $cid"
  if ! wait_for_card; then
    FAILED=$((FAILED + 1)); FAILED_IDS="$FAILED_IDS $cid"
    say "    NOT STARTED: the card never came free inside --max-wait"
    if [ "$KEEP_GOING" -eq 0 ]; then STOPPED=1; break; fi
  else
    say "    launching bash wavea_review.sh $cid --gpu $GPU --root $ROOT --lane $LANE --min-free-used $MIN_FREE_USED ${PKG_ARGS[*]:-} ${EXTRA[*]:-}"
    bash "$LANE/wavea_review.sh" "$cid" --gpu "$GPU" --root "$ROOT" --lane "$LANE" \
      --min-free-used "$MIN_FREE_USED" ${PKG_ARGS[@]+"${PKG_ARGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
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
        STOPPED=1; break
      fi
    fi
  fi
  say "re-listing the ledger for children registered while $cid ran ..."
  build_queue
done

# a final re-list so a child registered during the last review is named, not lost.
if [ "$STOPPED" -eq 1 ]; then
  say "final re-list after stopping early:"
  build_queue
fi
LEFT=${#QUEUE[@]}
say "------------------------------------------------------------------"
say "reviewed $DONE, failed $FAILED${FAILED_IDS:+ ($FAILED_IDS )}, passes $PASS"
if [ "$LEFT" -gt 0 ]; then
  for c in "${QUEUE[@]}"; do say "    NOT REVIEWED (registered during this run or left after a stop): $c"; done
fi
say "log $LOG"
[ "$FAILED" -eq 0 ] && [ "$LEFT" -eq 0 ] || exit 1
exit 0
