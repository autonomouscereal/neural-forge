#!/usr/bin/env bash
# Wave A on the 4B, one child at a time, on GPU0.  wavea_queue.sh
#
# One GPU, one build.  The queue is serial by construction and refuses to start
# while any `qf_build.py` is alive, because two folds sharing a 3090 is how a
# wave turns into fourteen OOMs at hour six.  It builds through `bodyq.sh`, so
# every child gets the same GPU preflight and the same reproduction check, and
# after each successful build it files the child into the campaign ledger with
# `qf_register.py --confirm`.
#
# What stops the queue and what does not:
#
# * a BUILD failure stops it, immediately, with a nonzero exit.  A fold that
#   failed is a fold whose successors were solved against nothing, and the GPU
#   is better idle than producing children nobody can trace.
# * a REGISTER failure does NOT stop it.  The child and its receipt are already
#   on disk and cost an hour of 3090; the register is a bookkeeping step that can
#   be re-run against the same receipt at any time.  Wave A deliberately contains
#   rungs that fold by 2x, and `oc_checkpoint.py` refuses any component above
#   ratio 0.25 (the >=4x rule is per component), so several of these tags are
#   EXPECTED to be refused.  Each refusal is logged, counted, and appended to
#   `logs/wavea-register-refusals.txt` for Cyril; none of them costs the wave a
#   build slot.
#
# Nothing here deletes anything.  The only path it ever removes is its own lock
# directory, and an existing child is skipped rather than rebuilt.
#
#   ./wavea_queue.sh                        # run it here (blocks)
#   QF_QUEUE_DRYRUN=1 ./wavea_queue.sh      # print the plan, touch no GPU
#   setsid nohup ./wavea_queue.sh < /dev/null >> logs/wavea-queue.out 2>&1 &
#
# `say` already tees every transition into logs/wavea-queue.log, so the detached
# launch sends stdout somewhere ELSE (.out) rather than into the same file twice.

# NOT `set -e`: every failure in this file is handled by name.
set -uo pipefail

# `bodyq.sh` reads LLAMA_CPP through env.sh's `${LLAMA_CPP:-...}` default, so it
# has to be exported BEFORE env.sh is sourced or the queue silently points at
# /home/cereal/llama.cpp, which does not exist on this box.
#
# NOTE, and this is a real gap rather than a setting: the stock converter is at
# $LLAMA_CPP/llama.cpp/convert_hf_to_gguf.py, one level BELOW the path below, so
# `qf_export.convert_to_gguf` finds no script and records the exact command
# without running it ("<script> not present").  The first Wave A child,
# C2K2304, was launched against this same path, and the queue keeps it so that
# every Wave A receipt is comparable.  Point LLAMA_CPP one level deeper to make
# the conversion actually run -- and budget ~9 GB of bf16 GGUF per child for it.
export LLAMA_CPP=${LLAMA_CPP:-/home/cereal/omni-campaign/export/llamacpp}

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
set +e                                   # env.sh turns on -e; the queue handles its own

OC_ROOT=${OC_ROOT:-/home/cereal/omni-campaign}
QUEUE_LOG="$W/logs/wavea-queue.log"
REFUSALS="$W/logs/wavea-register-refusals.txt"
LOCK="$W/logs/.wavea-queue.lock"
DRYRUN=${QF_QUEUE_DRYRUN:-0}

# TAG|qf_build flags.  This array is the single source of truth for both the
# build and the `--recipe-command` the ledger row is filed under, so the recipe
# in the record cannot drift from the command that produced the bytes.
QUEUE=(
  "C2K1152|--mlp-keep 1152"
  "X03|--excise 3"
  "X15|--excise 15"
  "X31|--excise 31"
  "X00|--excise 0"
  "X08|--excise 8"
  "X16|--excise 16"
  "G5V16K8|--gdn-value-heads 16 --gdn-key-heads 8"
  "G5D32|--gdn-key-head-dim 32 --gdn-value-head-dim 32"
  "N48|--nope-keep 48"
  "K2KV|--kv-heads 2 --kv-arm kv-only"
  "K2FG|--kv-heads 2 --kv-arm full-group"
  "T9|--terminal-gain"
  "V62080|--vocab-keep 62080"
)

utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(utc)  $*" | tee -a "$QUEUE_LOG"; }

mkdir -p "$W/logs"

# ---------------------------------------------------------------- preflight
# A dry run prints the plan and touches neither the GPU nor the ledger, so it
# takes no lock and does not care what else is building.  Everything below this
# line applies to a REAL run only.
if [ "$DRYRUN" != "1" ]; then
  if ! mkdir "$LOCK" 2>/dev/null; then
    say "QUEUE   REFUSED: $LOCK is held by $(cat "$LOCK/owner" 2>/dev/null || echo '(unknown)')"
    exit 2
  fi
  echo "pid $$ started $(utc)" > "$LOCK/owner"
  # The only path this script ever removes, and it is its own.
  trap 'rm -f "$LOCK/owner"; rmdir "$LOCK" 2>/dev/null' EXIT

  if pgrep -f qf_build.py > /dev/null; then
    say "QUEUE   REFUSED: a qf_build.py is already running -- $(pgrep -f qf_build.py | tr '\n' ' ')"
    exit 2
  fi
fi

say "QUEUE   start pid $$  ${#QUEUE[@]} children  W=$W  OC_ROOT=$OC_ROOT"
say "QUEUE   LLAMA_CPP=$LLAMA_CPP  (convert_hf_to_gguf.py present: $( [ -f "$LLAMA_CPP/convert_hf_to_gguf.py" ] && echo yes || echo NO ))"
say "QUEUE   dry run: $DRYRUN"

built=0; skipped=0; registered=0; refused=0

for entry in "${QUEUE[@]}"; do
  TAG="${entry%%|*}"
  FLAGS="${entry#*|}"
  CHILD="$W/children/QF-$TAG"
  RECEIPT="$CHILD/qf-build-receipt.json"
  RECIPE="LLAMA_CPP=$LLAMA_CPP ./bodyq.sh $TAG $FLAGS"
  CHECKPOINT="OC01-F-Q35-4B-$TAG"

  if [ "$DRYRUN" = "1" ]; then
    say "PLAN    $TAG  build: ./bodyq.sh $TAG $FLAGS  |  register: $CHECKPOINT"
    continue
  fi

  if [ -f "$RECEIPT" ]; then
    say "SKIP    $TAG  receipt already exists ($RECEIPT); not rebuilding, not deleting"
    skipped=$((skipped + 1))
  else
    say "BUILD   $TAG  start  ./bodyq.sh $TAG $FLAGS"
    "$HERE/bodyq.sh" "$TAG" $FLAGS >> "$W/logs/queue-$TAG.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      say "BUILD   $TAG  FAILED rc=$rc  -- see $W/logs/queue-$TAG.log and $W/logs/build-$TAG.log"
      say "QUEUE   STOPPED at $TAG; ${#QUEUE[@]} planned, $built built, $skipped skipped, $registered registered, $refused register-refused.  Nothing deleted."
      exit 1
    fi
    if [ ! -f "$RECEIPT" ]; then
      say "BUILD   $TAG  FAILED: rc=0 but $RECEIPT was not written"
      say "QUEUE   STOPPED at $TAG.  Nothing deleted."
      exit 1
    fi
    built=$((built + 1))
    say "BUILD   $TAG  ok  $("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); s=r["storage"]; print(f"{s[\"child_parameter_count\"]:,} parameters, net {s[\"net_stored_ratio\"]:.4f} ({s[\"net_fold\"]:.2f}x), receipt {r[\"receipt_sha256\"][:16]}")' "$RECEIPT" 2>/dev/null || echo 'receipt unreadable')"
  fi

  if [ -d "$OC_ROOT/registry/$CHECKPOINT" ]; then
    say "LEDGER  $TAG  already registered as $CHECKPOINT; leaving the chain alone"
    continue
  fi

  say "LEDGER  $TAG  register --confirm as $CHECKPOINT"
  "$PY" "$W/qf_register.py" "$TAG" \
      --work "$W" --root "$OC_ROOT" \
      --recipe-command "$RECIPE" \
      --confirm >> "$W/logs/register-$TAG.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    registered=$((registered + 1))
    say "LEDGER  $TAG  registered  $(grep -m1 '^ledger sha256' "$W/logs/register-$TAG.log" || echo '(no ledger line)')"
  else
    refused=$((refused + 1))
    reason="$(grep -m1 -E '^REFUSED:|REFUSED:' "$W/logs/register-$TAG.log" | head -c 300)"
    say "LEDGER  $TAG  REGISTER REFUSED rc=$rc -- ${reason:-see $W/logs/register-$TAG.log}"
    say "LEDGER  $TAG  the child and its receipt are untouched; the queue continues"
    {
      echo "$(utc)  $TAG  rc=$rc  $CHECKPOINT"
      echo "    recipe: $RECIPE"
      echo "    log:    $W/logs/register-$TAG.log"
      echo "    reason: ${reason:-(none printed)}"
    } >> "$REFUSALS"
  fi
done

say "QUEUE   done  ${#QUEUE[@]} planned, $built built, $skipped skipped, $registered registered, $refused register-refused"
exit 0
