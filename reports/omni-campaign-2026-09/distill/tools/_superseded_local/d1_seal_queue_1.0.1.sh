#!/usr/bin/env bash
# $OC/distill/queues/d1_seal_queue.sh -- seal the hardened D1 UNATTENDED: verify, register the delta, compose for the
# functional check, s7, write $Q/d1.sealed and hand GPU0 / GPU4 on; then the bank v1 review on whatever three cards
# are free, record-review, and the R2 clone.  Lane d1seal, Claude, 2026-10-14 (plan STAGE_D_RECONCILED_PLAN_2026-10-14
# sections 4 and 7; prereg section 2.7; CHECKPOINT_POLICY R2).  NOT ARMED by its author.
#
#   d1_seal_queue.sh --dry-run    every step, its command, the current value of every gate; writes NOTHING; exit 0
#   d1_seal_queue.sh --gate-once  the D1-finished gate once (d1_seal_lib d1-state): exit 0 done / 10 wait / 2 down
#   d1_seal_queue.sh              armed (detached: setsid nohup bash $Q/d1_seal_queue.sh > $Q/logs/d1_seal.log 2>&1 &)
#   d1_seal_queue.sh --phase b    phase B only (the review / record / clone), once phase A has written d1.sealed
# Single instance ($Q/d1_seal.lock).  Every step leaves $SEAL/steps/<step>.done (UTC + receipt sha) and is skipped
# when re-run, so a killed queue is restarted with the same command and resumes at the first unfinished step.
#
# PHASE A -- holds $Q/gpu0.lock + $Q/gpu4.lock (the locks d1_hardened_entry held for D1's lifetime; taken when D1 exits)
#   A0 wait     claude-d1-hardened.service not active AND runtime-status completed / succeeded / exit 0 / systemd
#               SERVICE_RESULT=success / launch id = launch.json's (d1_seal_lib d1-state).  An ALERT (refusal, restart
#               budget spent) ends the queue: nothing is sealed from a D1 that did not finish.  Polls every 5 min.
#   A1 locks    flock gpu0.lock + gpu4.lock (waits; never competes)
#   A2 sync     d1_canonical_rungs 1.1.0 sync of the hardened receipt (L11..L63 into the canonical dir), verify 64 layers
#   A3 verify   d1_seal_verify 1.0.0 (terminal success, receipt, 64-rung chain, canonical, every delta byte and element,
#               base digest re-derived to the sealed clone digest)
#   A4 recipe   oc_checkpoint show <base> -> base_record.json; d1_delta_recipe 1.0.0 -> DELTA/DELTA.json (256 replace)
#   A5 register oc_checkpoint register --delta-base <base> --delta-base-sha256 <sealed>: DRY RUN, d1_seal_lib
#               register-check (256 replace ops on 192 tensors, 4,278,517,760 elements, 0 added, composed = base
#               27,535,655,753, parent 27,781,427,952, base pinned at 8edf50c4...), then --confirm; head must be -0001-register
#   A6 plan     oc_checkpoint delta-compose --deep --plan-out (re-hashes the 55 GB base and the delta)
#   A7 compose  d1_compose_eval 1.0.0 -> /mnt/lexar/omni-campaign/scratch/d1seal/OMNI1D1-composed (declared scratch; the
#               review / s7 tools read a Chimera dir, no plan; D1 rewrites 12 of 14 shards and exFAT has no links)
#   A8 s7       s7_omni1_check 1.1.0 on GPU0 (one block resident, bf16, as OMNI1's check): strict load with every pin,
#               identity vs S5 outside the delta's regions and the delta's bytes inside, finite text + media forwards,
#               R-Q 210 top-1 vs the parent and vs OMNI1 (OMNI1 == S5 on R-Q: its own receipt, 210/210, max |diff| 0.0).
#               Functional refusals stop the seal; the R-Q numbers are diagnostics.
#   A9 sealed   $Q/d1.sealed (line 1 = UTC + the phase-A receipt + sha, what p2p11_card_guard / j6_vl.sh read), the
#               phase-A receipt $SEAL/d1_seal_phaseA.json (every receipt path + sha256), a five-line Relay hand-off in
#               $Q/relay_outbox/, then gpu0.lock / gpu4.lock are RELEASED (post_d1_queue.sh takes them).
# PHASE B -- needs no 3090: the composed child is 51.29 GiB of weights (headers), no split fits 2 x 24 GB.
#   B1 cards    the first placement that is free, re-checked every 60 s: "s8" = GPU6 + GPU4 + GPU0, bf16, the lab venv,
#               split 6,33,30 exactly as S8 (gpu6/gpu4/gpu0 locks all free, cards < 2,000 MiB), else "p40" = GPU1 + 2 + 3,
#               fp16, the Pascal venv (torch 2.6 cu118: the lab venv's torch 2.11 cu128 has no sm_61 kernels), the
#               configuration the parent's E2 dossier ran (split from d1_compose_eval split: 13,30,26 at 21 GiB per card),
#               needing no live p40.claims line on 1 / 2 / 3, their gpuN.lock free, the cards < 2,000 MiB.  Claim line +
#               locks held until the reviews end.  MemAvailable >= 40 GiB.  After REVIEW_WAIT_MAX_S (72 h) without
#               cards the review is recorded as pending and B4 clones under the standing GO (--allow-unreviewed-source,
#               argued in --notes); `--phase b` later runs the review and record-review.
#   B2 nothink  bank v1, run_review_finite_guard 1.2.0 -> run_omni_frozen_review 1.2.0, --thinking off,
#               --ledger-band-logit-mask, probe + smoke packs, cap 1024, greedy, --host-staged-moves (S8's flags)
#   B3 template the same on the template-default surface (the design's second surface)
#   B4 record   oc_assess --parent-dossier (E2 parent) -> oc_review_upper_bound (parent = the MEASURED E2 assessment;
#               machine UPPER BOUND, content misses 0 by construction) -> record-review DRY RUN, --confirm; the
#               template surface is assessed alone (no parent dossier on that surface); d1_vs_s8 reading recorded
#               (prereg 5: D1 bank v1 >= OMNI1 S8 -- a reading for the D2 start, never a refusal)
#   B5 clone    oc_checkpoint clone --source OC02-D-Q38-27BU-OMNI1D1 --stage-about-to-run D --clone-index 2 -> the R2 clone
#               OMNI1D1.clone-pre-D2 on /mnt/lexar (DRY RUN, then --confirm)
#   B6 done     $Q/d1.reviewed + the phase-B receipt + a second hand-off
# Never touched: the D1 run dir, the canonical rungs it imported (only appended to by sync), GPU5, GPU6 unless its
# lock is free, any registered directory (register / clone write only new records and the new clone).
# LANE HEARTBEAT (1.0.1, card X-T2, lane infra): every armed status line heartbeats lane L-D1 in $Q/LANES.json
# (lanes.py heartbeat --source $VERSION; a receipt=<path> in the line becomes the lane's last receipt + sha256), so the
# lane stays fresh after claude-d1-hardened.service ends and the cron pulse's probe stops heartbeating it.  Never fatal
# (20 s cap), silent with no registry (the selftests' temp queue dirs).
set -u
VERSION=d1_seal_queue-1.0.1
OC=${D1SEAL_OC:-/home/cereal/omni-campaign}
Q=${D1SEAL_Q:-$OC/distill/queues}
T=$OC/distill/tools
CT=$OC/composite/tools
EF=$OC/import/rq27b/tree_ef1ffb9
PY=/home/cereal/model-surgery-lab/venv/bin/python
PYP=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
L=$OC/ledger/oc_checkpoint.py
TAG=claude-20261014
RUN=${D1SEAL_RUN:-/mnt/lexar/omni-campaign/distill/d1/run-OMNI1D1-hardened-$TAG}
CAN=${D1SEAL_CAN:-/mnt/lexar/omni-campaign/distill/d1/OMNI1D1-rungs-canonical}
DELTA=${D1SEAL_DELTA:-/mnt/lexar/omni-campaign/deltas/OC02-D-Q38-27BU-OMNI1D1-hardened-$TAG}
SEAL=${D1SEAL_SEAL:-/mnt/lexar/omni-campaign/distill/d1/seal-OMNI1D1-hardened-$TAG}
COMPOSED=/mnt/lexar/omni-campaign/scratch/d1seal/OMNI1D1-composed
UNIT=${D1SEAL_UNIT:-claude-d1-hardened}
ID=OC02-D-Q38-27BU-OMNI1D1
BASE_ID=OC02-F-Q38-27BU-OMNI1.clone-pre-D1
BASE_DIR=/mnt/lexar/omni-campaign/clones/$BASE_ID
BASE_SHA=8edf50c42090d6d556d71d6c6c71e2405569b5082adaf133f4265bad1d901264
BASE_PARAMS=27535655753
PARENT_DIR=/home/cereal/models/Qwen3.8-27B-Uncensored
PARENT_PARAMS=27781427952
D1_ELEMENTS=4278517760
CLONE_DEST=/mnt/lexar/omni-campaign/clones/$ID.clone-pre-D2
S5=$OC/composite/c10/S5-C2K4352K3U/child
PARENT_S0=/mnt/ai_data/omni-campaign/s0-probe-sep04
TAP=$OC/composite/s7/tap_rows_s5
TOK=/mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07
OMNI1_S7=$OC/composite/s7/omni1/omni1_check.json
RV=${D1SEAL_RV:-$OC/composite/d1/review}
BANK=calibration/banks/lfm2_semantic_whole_response_prompts_v1.json
PROBE=calibration/banks/matrix_campaign_probe_prompts_v1.json
SMOKE=calibration/banks/matrix_campaign_smoke_prompts_v1.json
PD=$OC/composite/s0/review/oc-dossier-PARENT-Q38-27BU-NATIVE-NOTHINK.json
PA=$OC/calibration/out/assessment-PARENT-Q38-27BU-NATIVE-NOTHINK.json
S8REV=$OC/composite/s8/review/review-OC02-F-Q38-27BU-OMNI1-s8-packaged.json
STATUS=${D1SEAL_STATUS:-$Q/d1_seal.status}
OUTBOX=$Q/relay_outbox
ALLOC=${D1SEAL_ALLOC:-$OC/composite/receipts/allocator}
NVSMI=${D1SEAL_NVSMI:-nvidia-smi}
LOCK_POLL_S=${D1SEAL_LOCK_POLL_S:-60}
POLL_S=${D1SEAL_POLL_S:-300}
LOCK_WAIT_MAX_S=${D1SEAL_LOCK_WAIT_MAX_S:-86400}
REVIEW_WAIT_MAX_S=${D1SEAL_REVIEW_WAIT_MAX_S:-259200}
LANES_PY=${LANES_PY:-$T/lanes.py}
LANES_FILE=${LANES_FILE:-$Q/LANES.json}
MODE=armed; PHASE=all
case "${1:-}" in
  --dry-run) MODE=dry ;;
  --gate-once) MODE=gate ;;
  --cards-probe) MODE=cards ;;
  --phase) PHASE=${2:?--phase a|b} ;;
  "") ;;
  *) echo "usage: $0 [--dry-run | --gate-once | --cards-probe | --phase a|b]"; exit 64 ;;
esac

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(stamp)] $*"; }
lane_hb() {  # lane_hb <status line>: heartbeat L-D1 (see LANE HEARTBEAT above); armed only, never fatal, at most 20 s
  [ "$MODE" = armed ] && [ -f "$LANES_FILE" ] && [ -f "$LANES_PY" ] || return 0
  local st rcp extra=()
  case "$1" in
    "step=A0-wait state=waiting"*) st="D1 training; d1_seal_queue armed, waiting for it to finish ($VERSION)" ;;
    state=stopped*) st="D1 seal STOPPED, see detail and $STATUS ($VERSION)" ;;
    state=end*) st="d1_seal_queue ended, phase $PHASE ($VERSION)" ;;
    *) st="D1 seal running, phase $PHASE ($VERSION)" ;;
  esac
  rcp=$(echo "$1" | grep -o 'receipt=[^ ]*' | tail -1 | cut -d= -f2-)
  [ -n "$rcp" ] && [ -f "$rcp" ] && extra=(--receipt "$rcp")
  timeout 20 python3 "$LANES_PY" --file "$LANES_FILE" heartbeat --lane L-D1 --agent claude --source "$VERSION" \
    --status "$st" --detail "${1:0:400}" ${extra[@]+"${extra[@]}"} > /dev/null 2>&1 || say "lane heartbeat L-D1 failed (ignored)"
  return 0
}
status() { say "STATUS $*"; [ "$MODE" = armed ] && { echo "$(stamp) d1_seal $*" >> "$STATUS"; lane_hb "$*"; }; return 0; }
die() { status "state=stopped step=${STEP:-?} reason=\"$1\""; exit "${2:-3}"; }
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
mem_gib() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }
used_mib() { $NVSMI --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g="$1" '$1==g{print $2}'; }
apps_on() { $NVSMI -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'; }
unit_state() { systemctl show -p ActiveState --value "$UNIT.service" 2>/dev/null; }
lock_free() { [ -e "$1" ] || return 0; ( exec 3< "$1" && flock -n 3 ) 2>/dev/null; }   # probe only (creates nothing)
done_mark() { [ -f "$SEAL/steps/$1.done" ]; }
mark() { mkdir -p "$SEAL/steps"; echo "$(stamp) ${2:-}" > "$SEAL/steps/$1.done"; }
lib() { "$PY" "$T/d1_seal_lib.py" "$@"; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID OC_ROOT=$OC OC_LEDGER_ROOT=$OC/ledger

d1_gate() { lib d1-state --run-dir "$RUN" --unit-state "$(unit_state)"; }
wait_mem() {  # wait_mem <GiB> <tag>
  local held=0
  while [ "$(mem_gib)" -lt "$1" ]; do
    [ $((held % 1800)) -eq 0 ] && status "step=$2 state=waiting reason=MemAvailable_$(mem_gib)GiB_lt_$1"
    sleep 60; held=$((held + 60))
  done
}

# ------------------------------------------------------------------------------------------------ card probe
cards_probe() {  # prints s8 | p40 | none:<why>; the tests choose_cards makes, without taking a lock or writing
  local g u why=""
  for g in 6 4 0; do lock_free "$Q/gpu$g.lock" || why="$why gpu$g.lock"; u=$(used_mib $g); [ "${u:-99999}" -lt 2000 ] || why="$why GPU$g=${u:-?}MiB"; done
  [ -z "$why" ] && { echo s8; return 0; }
  local p40why=""
  lib p40-free --claims "$Q/p40.claims" --gpus 1,2,3 > /dev/null || p40why=" p40.claims"
  for g in 1 2 3; do lock_free "$Q/gpu$g.lock" || p40why="$p40why gpu$g.lock"; u=$(used_mib $g); [ "${u:-99999}" -lt 2000 ] || p40why="$p40why GPU$g=${u:-?}MiB"
    [ "$(apps_on $g)" -eq 0 ] || p40why="$p40why GPU$g-apps"; done
  [ -z "$p40why" ] && { echo p40; return 0; }
  echo "none: s8($why ) p40($p40why )"
  return 1
}

# ------------------------------------------------------------------------------------------------ plan / dry run
plan() {
  say "$VERSION  mode=$MODE phase=$PHASE  host $(hostname)"
  say "D1 unit $UNIT.service: $(unit_state || true); $(d1_gate 2>&1 | tail -1)"
  say "run $RUN (status $(sha "$RUN/runtime-status.json" | cut -c1-16)); canonical $CAN: $(grep -c safetensors "$CAN/SHA256SUMS" 2>/dev/null) rungs"
  say "delta $DELTA: $(ls "$DELTA" 2>/dev/null | tr '\n' ' ')"
  say "locks: gpu0 $(lock_free "$Q/gpu0.lock" && echo free || echo held); gpu4 $(lock_free "$Q/gpu4.lock" && echo free || echo held); gpu6 $(lock_free "$Q/gpu6.lock" && echo free || echo held); d1_seal $(lock_free "$Q/d1_seal.lock" && echo free || echo held)"
  say "P40 claims (1,2,3): $(lib p40-free --claims "$Q/p40.claims" --gpus 1,2,3 | cut -c1-300)"
  say "review cards now: $(cards_probe)"
  say "host: MemAvailable $(mem_gib) GiB; / $(free_gb /) GB; /mnt/lexar $(free_gb /mnt/lexar) GB"
  say "cards: $($NVSMI --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ';')"
  say "tools: canonical $(grep -m1 '^TOOL_VERSION' $T/d1_canonical_rungs.py | cut -d'"' -f2) $(sha $T/d1_canonical_rungs.py | cut -c1-16); verify $(sha $T/d1_seal_verify.py | cut -c1-16); compose $(sha $T/d1_compose_eval.py | cut -c1-16); lib $(sha $T/d1_seal_lib.py | cut -c1-16); s7 $(grep -m1 '^TOOL_VERSION' $CT/s7_omni1_check.py | cut -d'"' -f2) $(sha $CT/s7_omni1_check.py | cut -c1-16); recipe $(sha $T/d1_delta_recipe.py | cut -c1-16); ledger $(sha $L | cut -c1-16)"
  local s
  for s in A2-sync A3-verify A4-recipe A5-register A6-plan A7-compose A8-s7 A9-sealed B2-nothink B3-template B4-record B5-clone B6-done; do
    say "step $s: $(done_mark $s && cat "$SEAL/steps/$s.done" || echo pending)"
  done
  say "A5 register: $L register --checkpoint-id $ID --stage distill --wave oc02 --tag OMNI1D1 --parent-code Q38-27BU --directory $DELTA --delta-base $BASE_ID --delta-base-sha256 $BASE_SHA --parent-directory $PARENT_DIR --parameter-count auto --parent-parameter-count auto --recipe $DELTA/DELTA.json --recipe-command ... [--confirm after register-check]"
  say "A8 s7: CUDA_VISIBLE_DEVICES=0 $PY $CT/s7_omni1_check.py --omni1 $COMPOSED --s5 $S5 --parent $PARENT_S0 --tap-rows $TAP --tokenizer-dir $TOK --chimera-src $EF --parent-parameter-count $PARENT_PARAMS --device cuda:0 --dtype bfloat16 --delta-plan $SEAL/$ID.plan.json --omni1-equivalence-receipt $OMNI1_S7 --out $SEAL/s7_omni1d1_check.json"
  say "B2/B3: placement s8 (GPU6,4,0 bf16 split 6,33,30, lab venv) if free, else p40 (GPU1,2,3 fp16, Pascal venv, split from d1_compose_eval split); run_review_finite_guard.py --host-staged-moves --model $COMPOSED --thinking off|template-default --ledger-band-logit-mask --maximum-new-tokens 1024 -> $RV/"
  say "B5 clone: $L clone --source $ID --stage-about-to-run D --clone-index 2 --dest $CLONE_DEST [--confirm]"
}
if [ "$MODE" = dry ]; then plan; say "DRY RUN: nothing was written"; exit 0; fi
if [ "$MODE" = gate ]; then d1_gate; exit $?; fi
if [ "$MODE" = cards ]; then cards_probe; exit $?; fi

# ------------------------------------------------------------------------------------------------ armed
mkdir -p "$SEAL/steps" "$SEAL/logs" "$OUTBOX" "$RV" "$ALLOC"
exec 9>> "$Q/d1_seal.lock"
flock -n 9 || { say "another d1_seal_queue holds $Q/d1_seal.lock"; exit 5; }
plan >> "$SEAL/logs/plan-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
status "state=armed version=$VERSION phase=$PHASE pid=$$"

run_logged() {  # run_logged <step> <log> cmd...: the command's own output to the log, rc returned
  local step=$1 log=$2; shift 2
  say "$step: $*" >> "$log"
  "$@" >> "$log" 2>&1
}

phase_a() {
  # A0 + A1 ----------------------------------------------------------------------------------------------------
  STEP=A0-wait
  local line rc held=0 last=""
  while :; do
    line=$(d1_gate 2>&1); rc=$?
    [ $rc -eq 0 ] && break
    [ $rc -eq 2 ] && die "D1 is down for good: $line" 2
    [ "$line" != "$last" ] && status "step=A0-wait state=waiting $line"
    last=$line; sleep "$POLL_S"
  done
  status "step=A0-wait state=done $line"
  STEP=A1-locks
  exec 7>> "$Q/gpu0.lock" 6>> "$Q/gpu4.lock"
  held=0
  until flock -n 7 && flock -n 6; do
    [ $held -ge "$LOCK_WAIT_MAX_S" ] && die "gpu0/gpu4 locks not obtained in ${LOCK_WAIT_MAX_S}s" 4
    [ $((held % 1800)) -eq 0 ] && status "step=A1-locks state=waiting gpu0=$(lock_free "$Q/gpu0.lock" && echo free || echo held) gpu4=$(lock_free "$Q/gpu4.lock" && echo free || echo held)"
    sleep "$LOCK_POLL_S"; held=$((held + LOCK_POLL_S))
  done
  status "step=A1-locks state=held gpu0.lock gpu4.lock (pid $$)"
  for g in 0 4; do
    printf '{"lane":"d1seal phase A","gpu":%s,"by":"claude lane d1seal","started_utc":"%s","pid":%s,"status":"%s"}\n' \
      $g "$(stamp)" $$ "$STATUS" > "$ALLOC/d1seal-gpu$g-claude.json"
  done

  # A2 sync ------------------------------------------------------------------------------------------------------
  STEP=A2-sync
  if ! done_mark A2-sync; then
    run_logged A2 "$SEAL/logs/A2-sync.log" "$PY" "$T/d1_canonical_rungs.py" sync --canonical "$CAN" \
      --receipt "$RUN/trainer/d1_receipt.json" --rungs-dir "$RUN/trainer/rungs" --label "run-OMNI1D1-hardened-$TAG" \
      || die "canonical sync refused (see $SEAL/logs/A2-sync.log)"
    "$PY" "$T/d1_canonical_rungs.py" verify --canonical "$CAN" --expect-layers 64 --json > "$SEAL/canonical_verify.json" 2>> "$SEAL/logs/A2-sync.log" \
      || die "canonical verify: not 64 layers (see $SEAL/logs/A2-sync.log)"
    mark A2-sync "SHA256SUMS $(sha "$CAN/SHA256SUMS")"
  fi
  status "step=A2-sync state=done canonical=64 SHA256SUMS=$(sha "$CAN/SHA256SUMS")"

  # A3 verify ----------------------------------------------------------------------------------------------------
  STEP=A3-verify
  if ! done_mark A3-verify; then
    run_logged A3 "$SEAL/logs/A3-verify.log" nice -n 15 "$PY" "$T/d1_seal_verify.py" --run-dir "$RUN" --canonical "$CAN" \
      --delta-dir "$DELTA" --base-dir "$BASE_DIR" --base-sealed-sha256 "$BASE_SHA" --out "$SEAL/d1_seal_verify.json" \
      || die "d1_seal_verify refused: $(grep -m1 REFUSED "$SEAL/logs/A3-verify.log" | cut -c1-300)"
    mark A3-verify "$(sha "$SEAL/d1_seal_verify.json")"
  fi
  status "step=A3-verify state=done receipt=$SEAL/d1_seal_verify.json sha=$(sha "$SEAL/d1_seal_verify.json")"

  # A4 recipe ----------------------------------------------------------------------------------------------------
  STEP=A4-recipe
  if ! done_mark A4-recipe; then
    "$PY" "$L" --root "$OC" show --checkpoint-id "$BASE_ID" > "$SEAL/base_record.json" 2>> "$SEAL/logs/A4-recipe.log" \
      || die "oc_checkpoint show $BASE_ID failed"
    if [ ! -f "$DELTA/DELTA.json" ]; then
      run_logged A4 "$SEAL/logs/A4-recipe.log" "$PY" "$T/d1_delta_recipe.py" --delta-dir "$DELTA" \
        --base-record "$SEAL/base_record.json" --base-checkpoint-id "$BASE_ID" || die "d1_delta_recipe refused"
    fi
    "$PY" - "$DELTA/DELTA.json" "$BASE_ID" "$BASE_SHA" <<'EOF' >> "$SEAL/logs/A4-recipe.log" 2>&1 || die "DELTA.json is not the expected 256-entry replace recipe on the sealed base"
import json, sys
r = json.load(open(sys.argv[1]))
ok = (r["base_checkpoint_id"] == sys.argv[2] and r["base_directory_sha256"] == sys.argv[3] and len(r["entries"]) == 256
      and {e["op"] for e in r["entries"]} == {"replace"})
print("DELTA.json", len(r["entries"]), "entries; base", r["base_checkpoint_id"], r["base_directory_sha256"][:16], "ok" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
EOF
    mark A4-recipe "$(sha "$DELTA/DELTA.json")"
  fi
  status "step=A4-recipe state=done DELTA.json=$(sha "$DELTA/DELTA.json")"

  # A5 register --------------------------------------------------------------------------------------------------
  STEP=A5-register
  if ! done_mark A5-register; then
    local head
    head=$("$PY" "$L" --root "$OC" show --checkpoint-id "$ID" 2>/dev/null | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("record_instance_id",""))' 2>/dev/null)
    if [ -z "$head" ]; then
      local rc_cmd notes
      rc_cmd="d1_trunk_recovery $(grep -m1 '^TOOL_VERSION' $T/d1_trunk_recovery.py | cut -d'"' -f2) (sha $(sha $T/d1_trunk_recovery.py | cut -c1-16)) under d1_hardened_entry / d1_hardened_launch, unit $UNIT.service, argv in $RUN/launch.json (sha $(sha "$RUN/launch.json" | cut -c1-16)); canonical prefix L00-L10 (Codex v35 / c36-r2) + L11-L63 trained by the hardened run; DELTA.json by d1_delta_recipe 1.0.0"
      notes="Stage D1 per-layer trunk recovery of OMNI1 (prereg PREREGISTRATION_OMNI_STAGE_D_v1 + A1, frozen-body sha c24d21e0...). Delta of 256 replace entries (64 layers x gate/up/down trunk slices + norm2) on $BASE_ID @${BASE_SHA:0:16}. Seal verify: $SEAL/d1_seal_verify.json (sha $(sha "$SEAL/d1_seal_verify.json" | cut -c1-16)). Registered unattended by $VERSION under the standing GO (STAGE_D_RULINGS ruling 1); functional checks only, fit numbers are diagnostics (2026-10-06)."
      ARGS=(register --checkpoint-id "$ID" --stage distill --wave oc02 --tag OMNI1D1 --parent-code Q38-27BU
            --directory "$DELTA" --delta-base "$BASE_ID" --delta-base-sha256 "$BASE_SHA" --parent-directory "$PARENT_DIR"
            --parameter-count auto --parent-parameter-count auto --recipe "$DELTA/DELTA.json" --recipe-command "$rc_cmd" --notes "$notes")
      timeout 3600 nice -n 15 "$PY" "$L" --root "$OC" "${ARGS[@]}" > "$SEAL/register_dryrun.log" 2>&1 || die "register DRY RUN refused: $(grep -m1 REFUSED "$SEAL/register_dryrun.log" | cut -c1-300)"
      lib register-check --log "$SEAL/register_dryrun.log" --checkpoint-id "$ID" --base-id "$BASE_ID" --base-sha256 "$BASE_SHA" \
        --replace-ops 256 --targets 192 --replaced $D1_ELEMENTS --parameters $BASE_PARAMS --parent-parameters $PARENT_PARAMS \
        >> "$SEAL/logs/A5-register.log" 2>&1 || die "register-check: $(tail -1 "$SEAL/logs/A5-register.log" | cut -c1-400)"
      timeout 3600 nice -n 15 "$PY" "$L" --root "$OC" "${ARGS[@]}" --confirm > "$SEAL/register_confirm.log" 2>&1 \
        || die "register --confirm failed: $(grep -m1 REFUSED "$SEAL/register_confirm.log" | cut -c1-300)"
      head=$("$PY" "$L" --root "$OC" show --checkpoint-id "$ID" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["record_instance_id"])')
      [ "$head" = "$ID-0001-register" ] || die "after register --confirm the head is '$head', not $ID-0001-register"
    fi
    "$PY" "$L" --root "$OC" show --checkpoint-id "$ID" > "$SEAL/register_record.json"
    "$PY" - "$SEAL/register_record.json" "$DELTA" "$BASE_ID" "$BASE_SHA" "$(sha "$DELTA/DELTA.json")" <<'EOF' >> "$SEAL/logs/A5-register.log" 2>&1       || die "the registered $ID is not this delta on the sealed base (see $SEAL/logs/A5-register.log)"
import json, os, sys
r = json.load(open(sys.argv[1])); c = r["checkpoint"]; d = c.get("delta") or {}
ok = (os.path.realpath(c["directory"]) == os.path.realpath(sys.argv[2]) and d.get("base_checkpoint_id") == sys.argv[3]
      and d.get("base_directory_sha256") == sys.argv[4] and d.get("recipe_file_sha256") == sys.argv[5] and c["stage"] == "distill")
print("registered record", r["record_instance_id"], "delta", c["directory"], "base", d.get("base_checkpoint_id"), "ok" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
EOF
    mark A5-register "$head $(sha "$SEAL/register_record.json")"
  fi
  status "step=A5-register state=done id=$ID record=$(cut -d' ' -f3 "$SEAL/steps/A5-register.done") ledger=$(sha $OC/ledger.jsonl | cut -c1-16)"

  # A6 plan ------------------------------------------------------------------------------------------------------
  STEP=A6-plan
  if ! done_mark A6-plan; then
    rm -f "$SEAL/$ID.plan.json"
    run_logged A6 "$SEAL/logs/A6-plan.log" nice -n 15 "$PY" "$L" --root "$OC" delta-compose --checkpoint-id "$ID" --deep \
      --plan-out "$SEAL/$ID.plan.json" || die "delta-compose --deep refused (see $SEAL/logs/A6-plan.log)"
    mark A6-plan "$(sha "$SEAL/$ID.plan.json")"
  fi
  status "step=A6-plan state=done plan=$SEAL/$ID.plan.json sha=$(sha "$SEAL/$ID.plan.json")"

  # A7 compose ---------------------------------------------------------------------------------------------------
  STEP=A7-compose
  if ! done_mark A7-compose; then
    [ "$(free_gb /mnt/lexar)" -ge 120 ] || die "/mnt/lexar has $(free_gb /mnt/lexar) GB; the composed copy needs 56 GB above a 60 GB reserve"
    if [ -e "$COMPOSED" ] && [ ! -f "$COMPOSED.compose.json" ]; then   # killed between rename and receipt: declared scratch
      status "step=A7-compose event=removing_unreceipted_scratch dir=$COMPOSED"; rm -rf -- "$COMPOSED"
    fi
    if [ -f "$COMPOSED.compose.json" ] && ! grep -q "$(sha "$SEAL/$ID.plan.json")" "$COMPOSED.compose.json"; then
      status "step=A7-compose event=removing_composed_from_another_plan dir=$COMPOSED"; rm -rf -- "$COMPOSED" "$COMPOSED.compose.json" "$COMPOSED.compose.json.sha256"
    fi
    if [ ! -e "$COMPOSED" ]; then
      rm -rf "$COMPOSED.partial"   # an earlier compose killed mid-write (declared scratch)
      mkdir -p "$(dirname "$COMPOSED")"
      run_logged A7 "$SEAL/logs/A7-compose.log" nice -n 15 "$PY" "$T/d1_compose_eval.py" compose --plan "$SEAL/$ID.plan.json" \
        --out "$COMPOSED" || die "compose refused (see $SEAL/logs/A7-compose.log)"
    fi
    mark A7-compose "$(sha "$COMPOSED.compose.json")"
  fi
  status "step=A7-compose state=done dir=$COMPOSED receipt_sha=$(sha "$COMPOSED.compose.json")"

  # A8 s7 --------------------------------------------------------------------------------------------------------
  STEP=A8-s7
  if ! done_mark A8-s7; then
    wait_mem 40 A8-s7
    local u
    u=$(used_mib 0); [ "${u:-99999}" -lt 2000 ] || die "GPU0 holds ${u:-?} MiB while this queue holds gpu0.lock"
    ( export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
             OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
      cd "$CT" && nice -n 15 /usr/bin/time -v "$PY" s7_omni1_check.py --omni1 "$COMPOSED" --s5 "$S5" --parent "$PARENT_S0" \
        --tap-rows "$TAP" --tokenizer-dir "$TOK" --chimera-src "$EF" --parent-parameter-count $PARENT_PARAMS --device cuda:0 \
        --dtype bfloat16 --delta-plan "$SEAL/$ID.plan.json" --omni1-equivalence-receipt "$OMNI1_S7" \
        --out "$SEAL/s7_omni1d1_check.json" ) > "$SEAL/logs/A8-s7.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] || die "s7_omni1_check exit $rc: $(grep -m3 'REFUSAL\|REFUSED' "$SEAL/logs/A8-s7.log" | tr '\n' ' ' | cut -c1-400)"
    mark A8-s7 "$(sha "$SEAL/s7_omni1d1_check.json")"
  fi
  status "step=A8-s7 state=done receipt=$SEAL/s7_omni1d1_check.json sha=$(sha "$SEAL/s7_omni1d1_check.json") $(grep -m1 '^text ' "$SEAL/logs/A8-s7.log" | cut -c1-300)"

  # A9 sealed ----------------------------------------------------------------------------------------------------
  STEP=A9-sealed
  if ! done_mark A9-sealed; then
    "$PY" - "$SEAL/d1_seal_phaseA.json" "$ID" "$VERSION" \
      "canonical_sha256sums=$CAN/SHA256SUMS" "verify=$SEAL/d1_seal_verify.json" "delta_recipe=$DELTA/DELTA.json" \
      "base_record=$SEAL/base_record.json" "register_dryrun=$SEAL/register_dryrun.log" "register_confirm=$SEAL/register_confirm.log" \
      "register_record=$SEAL/register_record.json" "plan=$SEAL/$ID.plan.json" "compose=$COMPOSED.compose.json" \
      "s7=$SEAL/s7_omni1d1_check.json" "delta_manifest=$DELTA/D1_delta.json" "ledger=$OC/ledger.jsonl" <<'EOF' || die "phase-A receipt not written"
import hashlib, json, os, sys, time
out, cid, ver, *pairs = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()
items = {}
for kv in pairs:
    k, p = kv.split("=", 1)
    items[k] = {"path": p, "sha256": sha(p)}
s7 = json.load(open(items["s7"]["path"]))
t = s7.get("text_forward") or {}
doc = {"schema": "omni-campaign.d1-seal-phaseA.v1", "checkpoint_id": cid, "queue": ver,
       "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "receipts": items,
       "s7_verdict": s7.get("verdict"), "s7_refusals": s7.get("refusals"),
       "rq_210": {"positions": t.get("positions"), "top1_vs_parent": t.get("rq_top1_omni1_vs_parent"),
                  "top1_vs_omni1": t.get("rq_top1_child_vs_omni1"), "omni1_vs_parent_at_s7": 110,
                  "note": "diagnostics (owner policy 2026-10-06); OMNI1 S7 receipt: 110/210 vs parent"},
       "next": "phase B: bank v1 review (both surfaces) -> record-review -> clone OMNI1D1.clone-pre-D2; GPU0 / GPU4 released to post_d1_queue.sh"}
tmp = out + ".tmp"
open(tmp, "w").write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
os.replace(tmp, out)
print("phase A receipt", out, sha(out))
EOF
    local psha
    psha=$(sha "$SEAL/d1_seal_phaseA.json")
    echo "$(stamp) D1 sealed (phase A: verified, registered $ID, composed, s7 PASS) receipt $SEAL/d1_seal_phaseA.json sha256 $psha" > "$Q/d1.sealed.tmp"
    mv "$Q/d1.sealed.tmp" "$Q/d1.sealed"
    lib handoff --out "$OUTBOX/$(date -u +%Y%m%dT%H%M%SZ)-d1seal-phaseA.md" --lane "L-D1 / d1seal (Claude): D1 seal phase A" \
      --state "OC02-D-Q38-27BU-OMNI1D1 registered and functionally checked; receipt $SEAL/d1_seal_phaseA.json sha $psha; marker $Q/d1.sealed" \
      --running "phase B of $Q/d1_seal_queue.sh (pid $$) waits for three review cards (GPU6+4+0 or P40 1+2+3, p40.claims); GPU0/GPU4 released to post_d1_queue.sh" \
      --next "read $STATUS; the review/clone finish writes $Q/d1.reviewed; nothing to launch by hand" \
      --untouchable "$DELTA, $BASE_DIR, $CAN, the registry records of $ID, $COMPOSED (scratch, in use by the review), GPU5" > /dev/null
    mark A9-sealed "$psha"
  fi
  status "step=A9-sealed state=done marker=$Q/d1.sealed receipt=$SEAL/d1_seal_phaseA.json sha=$(sha "$SEAL/d1_seal_phaseA.json") relay=$OUTBOX (post it: session.py --agent claude say --type STATUS --body-file <file>)"
  exec 7>&- 6>&-
  for g in 0 4; do
    "$PY" -c 'import json,sys,time; p=sys.argv[1]; d=json.load(open(p)); d.update(ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), state="released"); json.dump(d, open(p,"w"))' "$ALLOC/d1seal-gpu$g-claude.json" 2>/dev/null
  done
  status "step=A9-sealed event=released gpu0.lock gpu4.lock"
}

# ------------------------------------------------------------------------------------------------ phase B
PLACEMENT=""; GPUS=""; DT=""; RPY=""; SPLIT=""
choose_cards() {  # sets PLACEMENT GPUS DT RPY SPLIT and takes the locks on fds 3/4/5, or returns 1
  local g u ok
  ok=1
  for g in 6 4 0; do lock_free "$Q/gpu$g.lock" || ok=0; u=$(used_mib $g); [ "${u:-99999}" -lt 2000 ] || ok=0; done
  if [ $ok = 1 ]; then
    exec 3>> "$Q/gpu6.lock" 4>> "$Q/gpu4.lock" 5>> "$Q/gpu0.lock"
    if flock -n 3 && flock -n 4 && flock -n 5; then
      PLACEMENT=s8; GPUS=6,4,0; DT=bfloat16; RPY=$PY; SPLIT=6,33,30; return 0
    fi
    exec 3>&- 4>&- 5>&-
  fi
  lib p40-free --claims "$Q/p40.claims" --gpus 1,2,3 > /dev/null || return 1
  for g in 1 2 3; do lock_free "$Q/gpu$g.lock" || return 1; u=$(used_mib $g); [ "${u:-99999}" -lt 2000 ] || return 1
    [ "$(apps_on $g)" -eq 0 ] || return 1; done
  exec 3>> "$Q/gpu1.lock" 4>> "$Q/gpu2.lock" 5>> "$Q/gpu3.lock"
  if flock -n 3 && flock -n 4 && flock -n 5; then
    SPLIT=$("$PY" "$T/d1_compose_eval.py" split --model "$COMPOSED" --budgets-gib 21.0,21.0,21.0 --head-extra-gib 3.0 \
            --block-extra-gib 0.6 | sed -n 's/^SPLIT //p')
    [ -n "$SPLIT" ] || { exec 3>&- 4>&- 5>&-; return 1; }
    echo "$(stamp) CLAIM GPU1,GPU2,GPU3 owner=claude lane=d1seal job=d1-review pid=$$ log=$SEAL/logs (release by $VERSION)" >> "$Q/p40.claims"
    PLACEMENT=p40; GPUS=1,2,3; DT=float16; RPY=$PYP; return 0
  fi
  exec 3>&- 4>&- 5>&-
  return 1
}
release_cards() {
  [ "$PLACEMENT" = p40 ] && echo "$(stamp) RELEASE GPU1,GPU2,GPU3 owner=claude lane=d1seal job=d1-review released" >> "$Q/p40.claims"
  exec 3>&- 4>&- 5>&-
  [ -n "$PLACEMENT" ] && status "step=B event=released placement=$PLACEMENT gpus=$GPUS"
  PLACEMENT=""
}
trap 'release_cards' EXIT

review() {  # review <step> <label> <thinking>
  local step=$1 label=$2 think=$3 out
  out=$RV/oc-dossier-$label.json
  if done_mark "$step"; then return 0; fi
  [ -e "$out" ] && die "$out exists without $step.done: inspect it (a review is never written over)"
  wait_mem 40 "$step"
  status "step=$step state=started placement=$PLACEMENT gpus=$GPUS dtype=$DT split=$SPLIT thinking=$think"
  ( export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src OC_CALIBRATION_DIR=$OC/calibration CUDA_VISIBLE_DEVICES=$GPUS \
           PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    cd "$OC" && nice -n 15 /usr/bin/time -v "$RPY" "$CT/run_review_finite_guard.py" --host-staged-moves --model "$COMPOSED" \
      --tokenizer-dir "$TOK" --chimera-src "$EF" --label "$label" --parent-code Q38-27BU --stage distill --split "$SPLIT" \
      --dtype "$DT" --device cuda:0 --thinking "$think" --ledger-band-logit-mask --maximum-new-tokens 1024 \
      --bank "$BANK" --probe-prompts "$PROBE" --smoke-prompts "$SMOKE" --out "$out" ) > "$SEAL/logs/$step.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$out" ]; then
    status "step=$step state=failed rc=$rc placement=$PLACEMENT log=$SEAL/logs/$step.log $(tail -2 "$SEAL/logs/$step.log" | tr '\n' ' ' | cut -c1-300)"
    return 1
  fi
  mark "$step" "$(sha "$out") placement=$PLACEMENT gpus=$GPUS dtype=$DT split=$SPLIT"
  status "step=$step state=done dossier=$out sha=$(sha "$out") guard=$(sha "$out.finite_guard.json" | cut -c1-16)"
}

phase_b() {
  [ -s "$Q/d1.sealed" ] || die "phase B needs $Q/d1.sealed (phase A)"
  local held=0 reviewed=1
  STEP=B1-cards
  if ! { done_mark B2-nothink && done_mark B3-template; }; then
    until choose_cards; do
      if [ $held -ge "$REVIEW_WAIT_MAX_S" ]; then reviewed=0; break; fi
      [ $((held % 1800)) -eq 0 ] && status "step=B1-cards state=waiting s8(gpu6/4/0 locks + cards) p40($(lib p40-free --claims "$Q/p40.claims" --gpus 1,2,3 | cut -c1-160))"
      sleep 60; held=$((held + 60))
    done
    if [ $reviewed = 1 ]; then
      status "step=B1-cards state=held placement=$PLACEMENT gpus=$GPUS dtype=$DT split=$SPLIT python=$RPY"
      review B2-nothink OC02-OMNI1D1-NATIVE-NOTHINK off || reviewed=0
      [ $reviewed = 1 ] && { review B3-template OC02-OMNI1D1-NATIVE-TEMPLATE template-default || status "step=B3-template state=failed_nonblocking (the thinking-off surface is the review of record)"; }
      release_cards
    else
      status "step=B1-cards state=gave_up waited_s=$held: review pending; the clone proceeds under the standing GO"
    fi
  fi

  STEP=B4-record
  if [ $reviewed = 1 ] && done_mark B2-nothink && ! done_mark B4-record; then
    local D=$RV/oc-dossier-OC02-OMNI1D1-NATIVE-NOTHINK.json A=$OC/calibration/out/assessment-OC02-OMNI1D1-NATIVE-NOTHINK-vs-PARENT-d1.json
    local SH=$OC/calibration/out/SCORING-SHEET-OC02-OMNI1D1-NATIVE-NOTHINK-vs-PARENT-d1.md REV=$RV/review-$ID-d1.json
    [ -f "$A" ] || run_logged B4 "$SEAL/logs/B4-record.log" nice -n 19 "$PY" "$OC/calibration/oc_assess.py" --dossier "$D" \
      --parent-dossier "$PD" --bank "$OC/$BANK" --out-json "$A" --out-markdown "$SH" || die "oc_assess failed"
    if done_mark B3-template; then
      local DT2=$RV/oc-dossier-OC02-OMNI1D1-NATIVE-TEMPLATE.json
      run_logged B4 "$SEAL/logs/B4-record.log" nice -n 19 "$PY" "$OC/calibration/oc_assess.py" --dossier "$DT2" --bank "$OC/$BANK" \
        --out-json "$OC/calibration/out/assessment-OC02-OMNI1D1-NATIVE-TEMPLATE-d1.json" \
        --out-markdown "$OC/calibration/out/SCORING-SHEET-OC02-OMNI1D1-NATIVE-TEMPLATE-d1.md" || status "step=B4-record note=template_surface_assess_failed"
    fi
    local notes="Lane d1seal: machine UPPER BOUND, not a reading. oc_review_upper_bound over $(basename "$A") (oc_assess without --per-case; canonical_content_miss 0 by construction, so this record cannot make $ID bad or deletable) about $(basename "$D") (sha $(sha "$D" | cut -c1-16)); parent = the MEASURED E2 assessment $(basename "$PA") (sha $(sha "$PA" | cut -c1-16)). SURFACE: the composed $ID ($COMPOSED, compose receipt sha $(sha "$COMPOSED.compose.json" | cut -c1-16)), run_omni_frozen_review 1.2.0 under run_review_finite_guard 1.2.0, placement $(cut -d' ' -f4- "$SEAL/steps/B2-nothink.done"), thinking off, --ledger-band-logit-mask, cap 1024, greedy, bank v1; OMNI1 S8 ran bf16 on GPU6+4+0 and the parent E2 fp16 on 3x P40 (dtype / hardware are diagnostics, owner policy 2026-10-06). Template-default surface dossier beside it. No human has read this sheet."
    [ -f "$REV" ] || run_logged B4 "$SEAL/logs/B4-record.log" nice -n 19 "$PY" "$OC/calibration/oc_review_upper_bound.py" --assessment "$A" \
      --dossier "$D" --parent-assessment "$PA" --reviewer "claude-lane-d1seal (oc_review_upper_bound.py over the D1 native dossier; parent = measured E2 PARENT-Q38-27BU assessment)" \
      --notes "$notes" --out "$REV" || die "oc_review_upper_bound failed"
    run_logged B4 "$SEAL/logs/B4-record.log" timeout 600 "$PY" "$L" --root "$OC" record-review --checkpoint-id "$ID" --review-json "$REV" \
      || die "record-review DRY RUN refused (see $SEAL/logs/B4-record.log)"
    run_logged B4 "$SEAL/logs/B4-record.log" timeout 600 "$PY" "$L" --root "$OC" record-review --checkpoint-id "$ID" --review-json "$REV" --confirm \
      || die "record-review --confirm failed"
    "$PY" - "$REV" "$S8REV" "$SEAL/d1_vs_s8_reading.json" <<'EOF' >> "$SEAL/logs/B4-record.log" 2>&1
import json, sys, time
d1, s8 = (json.load(open(p))["bank1"] for p in sys.argv[1:3])
k = "canonical_signal"
doc = {"d1": {x: d1.get(x) for x in ("useful", "partial", "observed_miss", k, "useful_or_partial")},
       "omni1_s8": {x: s8.get(x) for x in ("useful", "partial", "observed_miss", k, "useful_or_partial")},
       "d1_at_least_s8": (d1.get(k) or 0) >= (s8.get(k) or 0),
       "rule": "prereg section 5 / STAGE_D_RULINGS promotion: D2 starts from the D1 clone when bank v1 content >= the previous stage; both are machine upper bounds until Cyril reads them; recorded, never a refusal",
       "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(doc, open(sys.argv[3], "w"), indent=1)
print("D1 vs S8", json.dumps(doc))
EOF
    mark B4-record "$(sha "$REV")"
    status "step=B4-record state=done review=$REV sha=$(sha "$REV") reading=$SEAL/d1_vs_s8_reading.json"
  fi

  STEP=B5-clone
  if ! done_mark B5-clone; then
    local extra=() why=""
    if ! done_mark B4-record; then
      extra=(--allow-unreviewed-source)
      why=" UNREVIEWED SOURCE, recorded exception under the standing GO (STAGE_D_RULINGS ruling 1): the bank v1 review had no three free cards within ${REVIEW_WAIT_MAX_S}s or failed (see $STATUS); D2 needs the R2 clone and the review is recorded later with record-review on $ID."
    fi
    if [ ! -e "$CLONE_DEST" ]; then
      run_logged B5 "$SEAL/logs/B5-clone.log" timeout 3600 "$PY" "$L" --root "$OC" clone --source "$ID" --stage-about-to-run D \
        --clone-index 2 --dest "$CLONE_DEST" "${extra[@]}" --notes "pre-D2 clone of $ID by $VERSION.$why" || die "clone DRY RUN refused"
      run_logged B5 "$SEAL/logs/B5-clone.log" timeout 7200 nice -n 15 "$PY" "$L" --root "$OC" clone --source "$ID" --stage-about-to-run D \
        --clone-index 2 --dest "$CLONE_DEST" "${extra[@]}" --notes "pre-D2 clone of $ID by $VERSION.$why" --confirm || die "clone --confirm failed"
    fi
    "$PY" "$L" --root "$OC" show --checkpoint-id "$ID.clone-pre-D2" > "$SEAL/clone_record.json" || die "clone record unreadable"
    mark B5-clone "$(sha "$SEAL/clone_record.json")${extra:+ unreviewed-source}"
  fi
  status "step=B5-clone state=done clone=$ID.clone-pre-D2 dest=$CLONE_DEST record_sha=$(sha "$SEAL/clone_record.json")"

  STEP=B6-done
  local rsha=""
  done_mark B4-record && rsha=$(cut -d' ' -f3 "$SEAL/steps/B4-record.done")
  echo "$(stamp) D1 reviewed=$(done_mark B4-record && echo yes || echo pending) clone=$ID.clone-pre-D2 review_sha=${rsha:-none} clone_record_sha=$(sha "$SEAL/clone_record.json")" > "$Q/d1.reviewed"
  lib handoff --out "$OUTBOX/$(date -u +%Y%m%dT%H%M%SZ)-d1seal-phaseB.md" --lane "L-D1 / d1seal (Claude): D1 seal phase B" \
    --state "review $(done_mark B4-record && echo recorded || echo PENDING) (sha ${rsha:-none}); clone $ID.clone-pre-D2 at $CLONE_DEST (record sha $(sha "$SEAL/clone_record.json" | cut -c1-16)); marker $Q/d1.reviewed" \
    --running "nothing from this lane (d1_seal_queue exits); post_d1_queue.sh owns GPU0/GPU4" \
    --next "$(done_mark B4-record && echo "D2 waits for K3 sealed + P2 verdict; read $SEAL/d1_vs_s8_reading.json" || echo "bash $Q/d1_seal_queue.sh --phase b when three cards are free (records the review)")" \
    --untouchable "$CLONE_DEST, $DELTA, $BASE_DIR, the registry chain of $ID" > /dev/null
  mark B6-done "$(sha "$Q/d1.reviewed")"
  status "step=B6-done state=done marker=$Q/d1.reviewed"
}

case "$PHASE" in
  all) phase_a; phase_b ;;
  a) phase_a ;;
  b) phase_b ;;
  *) die "--phase is a or b" 64 ;;
esac
status "state=end"
exit 0
