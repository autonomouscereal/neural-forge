#!/usr/bin/env bash
# Wave A on the 4B, one child at a time, on GPU0.  wavea_queue.sh
#
# One GPU, one build.  The queue is serial by construction and refuses to start
# (and refuses to start the NEXT build) while any `qf_build.py` is alive or
# while the build card shows more than 2000 MiB in use, because two folds
# sharing a 3090 is how a wave turns into fourteen OOMs at hour six.  It builds
# through `bodyq.sh`, so every child gets the same GPU preflight and the same
# reproduction check, and after each successful build it files the child into
# the campaign ledger with `qf_register.py --confirm`.
#
# What stops the queue and what does not:
#
# * a BUILD failure stops it, immediately, with a nonzero exit.  A fold that
#   failed is a fold whose successors were solved against nothing, and the GPU
#   is better idle than producing children nobody can trace.
# * a REGISTER outcome is classified, not assumed:
#     - exit 0                              -> registered; the ledger's own
#                                              `sealed` / `ledger sha256` /
#                                              plan lines are quoted verbatim
#                                              into the queue log.
#     - a `sealed <path>` line + nonzero    -> the row was sealed and then
#                                              qf_register's postcheck failed
#                                              (digest mismatch).  The registry
#                                              row needs repair; the queue STOPS
#                                              nonzero so nobody reviews it.
#     - exit 3 + a per-component ratio     -> the EXPECTED refusal.  Wave A
#       refusal (qf_register's "no             deliberately contains rungs that
#       granularity ..." or the ledger's       fold by 2x and CHECKPOINT_POLICY
#       "ratio N > 0.25")                      makes >=4x per component the
#                                              definition of a fold.  Logged,
#                                              counted, appended to
#                                              logs/wavea-register-refusals.txt
#                                              for Cyril; the queue continues
#                                              and exits 3 at the end so a
#                                              supervisor can see the wave is
#                                              not fully registered.
#     - anything else (1, 2, 127, a         -> a broken ledger lane, not an
#       traceback, no REFUSED line)            expected refusal.  The queue STOPS
#                                              nonzero.  The child and its receipt
#                                              are untouched.
#
# Nothing here deletes anything.  The only path it ever removes is its own lock
# directory (and only when the pid recorded in it is dead, e.g. after today's
# 17:32Z power cut), and an existing child is skipped rather than rebuilt.  A
# child directory or a stray .gguf WITHOUT a receipt -- an interrupted build --
# is refused up front, before any GPU time is spent, and left for a human to
# move aside: qf_export refuses to write into an existing --out only AFTER the
# whole fold has been solved on the 3090.
#
#   ./wavea_queue.sh                        # run it here (blocks)
#   QF_QUEUE_DRYRUN=1 ./wavea_queue.sh      # print the plan, touch no GPU
#   QUEUE_TAGS="K2KV K2FG T9 V62080" ./wavea_queue.sh   # resume the tail of a wave
#   setsid nohup ./wavea_queue.sh < /dev/null >> logs/wavea-queue.out 2>&1 &
#
# QUEUE_TAGS is a whitespace-separated SUBSET of the tags in the QUEUE array.
# It filters; it does not add, reorder or redefine.  The children still run in
# the order the array declares, an unknown tag is a refusal rather than a silent
# no-op, and every guard below (lock, live-build refusal, partial-child refusal,
# GPU-busy refusal, per-tag registration, stop-on-first-failure) applies to the
# filtered list exactly as it applied to the whole one.  It exists because a
# wave that stopped on child 11 of 14 should be resumable without editing --
# and therefore without risking an edit to -- the array that seals every
# --recipe-command.
#
# `say` already tees every transition into logs/wavea-queue.log, so the detached
# launch sends stdout somewhere ELSE (.out) rather than into the same file twice.
#
# Disk: with LLAMA_CPP pointing at the real checkout the converter runs for every
# child and writes ~9 GB of bf16 GGUF to children/QF-<TAG>.gguf; budget ~126 GB
# for the wave.  qf_export.convert_to_gguf runs the converter with a 3600 s
# timeout that it does not catch; a conversion slower than an hour would kill
# the build after the child directory exists but before the receipt is written,
# which the partial-child preflight above then reports on the next start.

# NOT `set -e`: every failure in this file is handled by name.
set -uo pipefail

# --------------------------------------------------------------- environment
# GPU pin.  env.sh's gpu_preflight sets PCI order for its own probe process
# ONLY; without these two exports the build under it enumerates in CUDA's
# default (fastest-first) order, where "cuda:0" can be the other 3090 (PCI 4,
# the review card).  Exported here so the preflight and the build see the same
# single device.  Override with QF_QUEUE_GPU=<pci index or GPU-uuid>.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${QF_QUEUE_GPU:-0}

# `bodyq.sh` reads LLAMA_CPP through env.sh's `${LLAMA_CPP:-...}` default, so it
# has to be exported BEFORE env.sh is sourced or the queue silently points at
# /home/cereal/llama.cpp, which does not exist on this box.  The real checkout
# (convert_hf_to_gguf.py and gguf-py) is the llama.cpp directory INSIDE
# export/llamacpp; the preflight below refuses to run without the converter.
export LLAMA_CPP=${LLAMA_CPP:-/home/cereal/omni-campaign/export/llamacpp/llama.cpp}

# The receipt's parent_code is written verbatim from qf_build's --parent-code
# (default "") and sealed into receipt_sha256; qf_register refuses a receipt
# whose parent_code is empty.  bodyq.sh does not add the flag, so the queue does.
PARENT_CODE=${PARENT_CODE:-Q35-4B}

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
set +e                                   # env.sh turns on -e; the queue handles its own

# qf_export.convert_to_gguf invokes bare "python", not $PY; put the venv's
# interpreter first so the converter runs under torch/transformers.
export PATH="$(dirname "$PY"):$PATH"

OC_ROOT=${OC_ROOT:-/home/cereal/omni-campaign}
LEDGER_CLI="$OC_ROOT/ledger/oc_checkpoint.py"
QUEUE_LOG="$W/logs/wavea-queue.log"
REFUSALS="$W/logs/wavea-register-refusals.txt"
LOCK="$W/logs/.wavea-queue.lock"
DRYRUN=${QF_QUEUE_DRYRUN:-0}
GPU_MAX_USED_MIB=2000

# The sealed --recipe-command names the tag, the axis flags and LLAMA_CPP; it
# does not name W / PARENT / EVIDENCE, which env.sh lets the environment
# override.  Rather than seal a recipe that rebuilds a different child, refuse
# to run with any of them overridden.
DEFAULT_W=/home/cereal/omni-campaign/qwen-fold
DEFAULT_PARENT=/home/cereal/models/Qwen3.5-4B
DEFAULT_EVIDENCE=$DEFAULT_W/evidence/omni-evidence-v1.json

# TAG|qf_build flags[|noregister].  This array is the single source of truth
# for both the build and the `--recipe-command` the ledger row is filed under,
# so the recipe in the record cannot drift from the command that produced the
# bytes.  `noregister` marks a rung the ledger cannot file: T9 (--terminal-gain)
# is a 0-parameter repair rung, so no header cut and no receipt rollup yields a
# changed piece and qf_register refuses deterministically; it is built only.
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
  "T9|--terminal-gain|noregister"
  "V62080|--vocab-keep 62080"
)

utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(utc)  $*" | tee -a "$QUEUE_LOG"; }

mkdir -p "$W/logs"

# --------------------------------------------------------- QUEUE_TAGS filter
# Applied HERE, before the preflight, so the partial-child scan, the plan a dry
# run prints and the children that actually build are all the same list.  The
# array above stays the single source of truth for flags and recipes: this only
# drops entries from it.
QUEUE_TAGS=${QUEUE_TAGS:-}
if [ -n "$QUEUE_TAGS" ]; then
  known=""
  for entry in "${QUEUE[@]}"; do known="$known ${entry%%|*}"; done
  for want in $QUEUE_TAGS; do
    case "$known " in
      *" $want "*) ;;
      *) say "QUEUE   REFUSED: QUEUE_TAGS names '$want', which is not a tag in this queue --$known"
         exit 2 ;;
    esac
  done
  selected=()
  for entry in "${QUEUE[@]}"; do
    case " $QUEUE_TAGS " in
      *" ${entry%%|*} "*) selected+=("$entry") ;;
    esac
  done
  if [ "${#selected[@]}" -eq 0 ]; then
    say "QUEUE   REFUSED: QUEUE_TAGS='$QUEUE_TAGS' selected nothing"
    exit 2
  fi
  QUEUE=("${selected[@]}")
  say "QUEUE   QUEUE_TAGS='$QUEUE_TAGS' -> ${#QUEUE[@]} children, array order preserved"
fi

# ----------------------------------------------------------------- helpers
# Anchored to the interpreter so an editor, a `less qf_build.py` or a grep on a
# log path does not read as a live build.  `timeout 86400 python ... qf_build.py`
# matches too, which is what we want.
live_builds() { pgrep -f 'python[^ ]* .*qf_build\.py' 2>/dev/null | tr '\n' ' '; }

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null | head -1 | tr -d ' '
}

# Real-run gate, run at start AND before every build: the queue runs for hours
# and the card must still be ours when the next fold starts.
refuse_if_busy() {
  local label="$1" pids used
  pids="$(live_builds)"
  if [ -n "$pids" ]; then
    say "$label  REFUSED: a qf_build.py is already running -- pids $pids"
    return 1
  fi
  used="$(gpu_used_mib)"
  case "$used" in
    ''|*[!0-9]*)
      say "$label  REFUSED: nvidia-smi gave no memory.used reading for GPU $CUDA_VISIBLE_DEVICES ('$used')"
      return 1 ;;
  esac
  if [ "$used" -gt "$GPU_MAX_USED_MIB" ]; then
    say "$label  REFUSED: GPU $CUDA_VISIBLE_DEVICES shows ${used} MiB used (limit $GPU_MAX_USED_MIB)"
    return 1
  fi
  say "$label  GPU $CUDA_VISIBLE_DEVICES shows ${used} MiB used; no live qf_build.py"
  return 0
}

# A child directory or a stray GGUF without a receipt is an interrupted build.
# The queue never removes either; it refuses and names the path.
refuse_if_partial() {
  local tag="$1" child receipt gguf
  child="$W/children/QF-$tag"; receipt="$child/qf-build-receipt.json"; gguf="$W/children/QF-$tag.gguf"
  if [ -e "$child" ] && [ ! -f "$receipt" ]; then
    say "BUILD   $tag  REFUSED: $child exists without a receipt (interrupted build?); move it aside by hand -- nothing deleted"
    return 1
  fi
  if [ -e "$gguf" ] && [ ! -f "$receipt" ]; then
    say "BUILD   $tag  REFUSED: $gguf exists without a receipt (interrupted conversion?); move it aside by hand -- nothing deleted"
    return 1
  fi
  return 0
}

# bodyq.sh's own reproduction check, re-run here when a receipt exists but no
# `[qf] reproduction ok` line was ever recorded for it (bodyq writes the receipt
# BEFORE the digest check, so a child whose check failed still has a receipt).
reproduce() {
  local child="$1" receipt="$2"
  "$PY" - "$child" "$receipt" <<'PYEOF'
import json, sys
sys.path.insert(0, "/home/cereal/omni-campaign/qwen-fold")
import qf_export

child, receipt_path = sys.argv[1], sys.argv[2]
receipt = json.loads(open(receipt_path, encoding="ascii").read())
digest, _ = qf_export.directory_digest(child)
claimed = receipt["export"]["child_tensors_sha256"]
if digest != claimed:
    raise SystemExit(f"[qf] child digest {digest} != receipt {claimed}")
print(f"[qf] reproduction ok: {digest[:16]}  "
      f"{receipt['storage']['child_parameter_count']:,} parameters  "
      f"net {receipt['storage']['net_stored_ratio']:.4f}x")
PYEOF
}

receipt_field() {           # receipt_field <receipt> <top-level key>
  "$PY" - "$1" "$2" <<'PYEOF'
import json, sys
print(json.load(open(sys.argv[1], encoding="ascii"))[sys.argv[2]])
PYEOF
}

receipt_summary() {
  "$PY" - "$1" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1], encoding="ascii")); s = r["storage"]
print(f"{s['child_parameter_count']:,} parameters, net {s['net_stored_ratio']:.4f} "
      f"({s['net_fold']:.2f}x), receipt {r['receipt_sha256'][:16]}")
PYEOF
}

# The ledger's own notion of "registered" is a record dir that CONTAINS sealed
# records (history()), not a directory that exists: the .pending marker mkdirs
# the record dir without a record.  Ask the ledger.  A deleted or refused chain
# still answers "registered", which keeps the never-re-register semantics.
already_registered() {
  "$PY" "$LEDGER_CLI" --root "$OC_ROOT" show --checkpoint-id "$1" > /dev/null 2>&1
}

# ---------------------------------------------------------------- preflight
if [ "$W" != "$DEFAULT_W" ] || [ "$PARENT" != "$DEFAULT_PARENT" ] || [ "$EVIDENCE" != "$DEFAULT_EVIDENCE" ]; then
  say "QUEUE   REFUSED: W/PARENT/EVIDENCE are overridden (W=$W PARENT=$PARENT EVIDENCE=$EVIDENCE); the sealed recipe would not rebuild these children"
  exit 2
fi

# A dry run prints the plan and touches neither the GPU nor the ledger, so it
# takes no lock and does not care what else is building.  Everything below this
# line applies to a REAL run only.
if [ "$DRYRUN" != "1" ]; then
  # The lock survives SIGKILL and power loss (the EXIT trap never ran).  A lock
  # whose pid is dead, or whose start predates this boot, is stale and is the
  # one thing this script reclaims.
  if [ -d "$LOCK" ]; then
    owner="$(cat "$LOCK/owner" 2>/dev/null || true)"
    pid="$(echo "$owner" | awk '{print $2}')"
    started="$(echo "$owner" | awk '{print $4}')"
    stale=0
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      stale=1
    elif boot_epoch="$(date -d "$(uptime -s)" +%s 2>/dev/null)" \
      && start_epoch="$(date -d "$started" +%s 2>/dev/null)" \
      && [ "$start_epoch" -lt "$boot_epoch" ]; then
      stale=1                         # pid reused after a reboot
    fi
    if [ "$stale" = "1" ]; then
      say "QUEUE   stale lock from pid ${pid:-?} started ${started:-?} (not running); reclaiming $LOCK"
      rm -f "$LOCK/owner"; rmdir "$LOCK" 2>/dev/null
    else
      say "QUEUE   REFUSED: $LOCK is held by ${owner:-(unknown)}"
      exit 2
    fi
  fi
  if ! mkdir "$LOCK" 2>/dev/null; then
    say "QUEUE   REFUSED: $LOCK appeared while starting (another queue racing?); $(cat "$LOCK/owner" 2>/dev/null || echo '(unknown owner)')"
    exit 2
  fi
  echo "pid $$ started $(utc)" > "$LOCK/owner"
  # The only path this script ever removes, and it is its own.
  trap 'rm -f "$LOCK/owner"; rmdir "$LOCK" 2>/dev/null' EXIT

  if [ ! -f "$LLAMA_CPP/convert_hf_to_gguf.py" ]; then
    say "QUEUE   REFUSED: no convert_hf_to_gguf.py under $LLAMA_CPP; every receipt would seal gguf.ran=false"
    exit 2
  fi
  if [ ! -f "$LEDGER_CLI" ]; then
    say "QUEUE   REFUSED: ledger CLI $LEDGER_CLI is missing"
    exit 2
  fi
  refuse_if_busy "QUEUE  " || exit 2
  for entry in "${QUEUE[@]}"; do
    refuse_if_partial "${entry%%|*}" || exit 2
  done
fi

say "QUEUE   start pid $$  ${#QUEUE[@]} children  W=$W  OC_ROOT=$OC_ROOT  PARENT_CODE=$PARENT_CODE"
say "QUEUE   CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  PY=$PY"
say "QUEUE   LLAMA_CPP=$LLAMA_CPP  (convert_hf_to_gguf.py present: $( [ -f "$LLAMA_CPP/convert_hf_to_gguf.py" ] && echo yes || echo NO ))"
say "QUEUE   dry run: $DRYRUN"

built=0; skipped=0; registered=0; refused=0; unregistrable=0

for entry in "${QUEUE[@]}"; do
  IFS='|' read -r TAG FLAGS MODE <<< "$entry"
  MODE=${MODE:-register}
  CHILD="$W/children/QF-$TAG"
  RECEIPT="$CHILD/qf-build-receipt.json"
  BUILD_ARGS="--parent-code $PARENT_CODE $FLAGS"
  # The sealed recipe is the literal build line, so the record rebuilds the child.
  RECIPE="LLAMA_CPP=$LLAMA_CPP ./bodyq.sh $TAG $BUILD_ARGS"
  CHECKPOINT="OC01-F-$PARENT_CODE-$TAG"

  if [ "$DRYRUN" = "1" ]; then
    state="build"
    [ -f "$RECEIPT" ] && state="skip (receipt exists)"
    [ -e "$CHILD" ] && [ ! -f "$RECEIPT" ] && state="REFUSE (child without receipt)"
    reg="register: $CHECKPOINT"
    [ "$MODE" = "noregister" ] && reg="register: none (0-parameter repair rung)"
    say "PLAN    $TAG  $state: ./bodyq.sh $TAG $BUILD_ARGS  |  $reg"
    continue
  fi

  case " $BUILD_ARGS " in
    *" --parent-code "*) ;;
    *) say "BUILD   $TAG  REFUSED: build line carries no --parent-code"; exit 2 ;;
  esac

  if [ -f "$RECEIPT" ]; then
    say "SKIP    $TAG  receipt already exists ($RECEIPT); not rebuilding, not deleting"
    if grep -q '^\[qf\] reproduction ok' "$W/logs/queue-$TAG.log" 2>/dev/null; then
      say "SKIP    $TAG  $(grep -m1 '^\[qf\] reproduction ok' "$W/logs/queue-$TAG.log")"
    else
      say "SKIP    $TAG  no reproduction record in logs/queue-$TAG.log; re-deriving the child digest"
      repro="$(reproduce "$CHILD" "$RECEIPT" 2>&1)"
      rc=$?
      if [ $rc -ne 0 ]; then
        say "SKIP    $TAG  REPRODUCTION FAILED rc=$rc: $(echo "$repro" | tail -1 | head -c 300)"
        say "QUEUE   STOPPED at $TAG; the child and its receipt are untouched -- inspect by hand.  Nothing deleted."
        exit 1
      fi
      say "SKIP    $TAG  $(echo "$repro" | tail -1)"
    fi
    skipped=$((skipped + 1))
  else
    refuse_if_busy "BUILD   $TAG" || { say "QUEUE   STOPPED at $TAG.  Nothing deleted."; exit 2; }
    refuse_if_partial "$TAG"      || { say "QUEUE   STOPPED at $TAG.  Nothing deleted."; exit 2; }
    say "BUILD   $TAG  start  ./bodyq.sh $TAG $BUILD_ARGS"
    "$HERE/bodyq.sh" "$TAG" $BUILD_ARGS >> "$W/logs/queue-$TAG.log" 2>&1
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
    summary="$(receipt_summary "$RECEIPT")" || summary='receipt unreadable'
    say "BUILD   $TAG  ok  $summary"
    say "BUILD   $TAG  $(grep -m1 '^\[qf\] reproduction ok' "$W/logs/queue-$TAG.log" || echo '(no reproduction line in queue log)')"
  fi

  # The id qf_register seals comes from receipt['parent_code'], not from this
  # script; read it back so the skip test and the sealed id cannot disagree.
  receipt_parent="$(receipt_field "$RECEIPT" parent_code 2>/dev/null)"
  if [ "$receipt_parent" != "$PARENT_CODE" ]; then
    if [ "$MODE" = "noregister" ]; then
      say "LEDGER  $TAG  receipt parent_code is '${receipt_parent}', expected $PARENT_CODE (build-only rung; noting, not stopping)"
    else
      say "LEDGER  $TAG  REFUSED: receipt parent_code is '${receipt_parent}', expected $PARENT_CODE; qf_register would file it under another id (or refuse an empty one)"
      say "QUEUE   STOPPED at $TAG; the child and its receipt are untouched.  Nothing deleted."
      exit 1
    fi
  fi

  if [ "$MODE" = "noregister" ]; then
    say "LEDGER  $TAG  not registrable (0-parameter repair rung); build only"
    unregistrable=$((unregistrable + 1))
    continue
  fi

  if already_registered "$CHECKPOINT"; then
    say "LEDGER  $TAG  already registered as $CHECKPOINT (ledger show answers); leaving the chain alone"
    continue
  fi

  RLOG="$W/logs/register-$TAG-$(utc).log"       # one fresh log per attempt
  say "LEDGER  $TAG  register --confirm as $CHECKPOINT  (log $RLOG)"
  "$PY" "$W/qf_register.py" "$TAG" \
      --work "$W" --root "$OC_ROOT" \
      --recipe-command "$RECIPE" \
      --confirm > "$RLOG" 2>&1
  rc=$?

  if [ $rc -eq 0 ]; then
    registered=$((registered + 1))
    grep -E '^(sealed |ledger sha256|  (parameters|parent parameters|net_ratio|components) |postcheck ok)' "$RLOG" \
      | while IFS= read -r line; do say "LEDGER  $TAG  $line"; done
    say "LEDGER  $TAG  registered as $CHECKPOINT"
    continue
  fi

  # Sealed, then failed: qf_register's postcheck re-hashes the child AFTER
  # oc_checkpoint printed `sealed <path>`.  That is a registry row that needs
  # repair, never an expected refusal.
  if grep -q '^sealed ' "$RLOG"; then
    say "LEDGER  $TAG  SEALED BUT POSTCHECK FAILED rc=$rc -- $(grep -m1 '^sealed ' "$RLOG")"
    say "LEDGER  $TAG  $(grep -m1 'REFUSED:' "$RLOG" | head -c 400)"
    say "QUEUE   STOPPED at $TAG; the registry row for $CHECKPOINT needs repair before anyone reviews it.  Nothing deleted."
    exit 1
  fi

  if [ $rc -eq 3 ] && grep -q -E 'no granularity of this fold satisfies|ratio [0-9.]+ > 0\.25' "$RLOG"; then
    refused=$((refused + 1))
    reason="$(grep -m1 -E 'REFUSED:' "$RLOG" | head -c 300)"
    say "LEDGER  $TAG  REGISTER REFUSED rc=$rc (per-component >=4x rule) -- ${reason:-see $RLOG}"
    say "LEDGER  $TAG  the child and its receipt are untouched; the queue continues"
    {
      echo "$(utc)  $TAG  rc=$rc  $CHECKPOINT"
      echo "    recipe: $RECIPE"
      echo "    log:    $RLOG"
      echo "    reason: ${reason:-(none printed)}"
    } >> "$REFUSALS"
    continue
  fi

  say "LEDGER  $TAG  register FAILED rc=$rc (not a ratio refusal) -- $(grep -m1 -E 'REFUSED:|Error|Traceback' "$RLOG" | head -c 300)"
  say "LEDGER  $TAG  see $RLOG; the child and its receipt are untouched"
  say "QUEUE   STOPPED at $TAG; ${#QUEUE[@]} planned, $built built, $skipped skipped, $registered registered, $refused register-refused.  Nothing deleted."
  exit 1
done

say "QUEUE   done  ${#QUEUE[@]} planned, $built built, $skipped skipped, $registered registered, $refused register-refused, $unregistrable build-only"
if [ "$DRYRUN" = "1" ]; then
  exit 0
fi
[ "$refused" -eq 0 ] && exit 0 || exit 3
