#!/usr/bin/env bash
# d2_launch.sh -- Stage D2 (end-to-end LoRA KD on the D1 child: text + tool lane + K3 rows, then the long rung) as ONE
# unattended queue.  Lane L-TRAIN (Claude, d2prep), 2026-10-14.  Plan: STAGE_D_RECONCILED_PLAN_2026-10-14 sections 3-4
# and 8; prereg D2 + amendment A2 (DRAFT until C1/C2 seal); STAGE_D_RULINGS (standing GO, G5-1 engine rule).
#
#   bash $Q/d2_launch.sh preflight   read-only.  Every gate printed OPEN / WAIT / REFUSED.  Exit 0 = D2 could start
#                                    now, 10 = waiting on a gate, 2 = refused (a pin or a file is wrong).  Also runs
#                                    `streamed_kd_trainer.py data-check` on the registry sources that exist (raw shard
#                                    verification, the bf16 mass interval, exclusions and masks against the real
#                                    caches); D2_PREFLIGHT_DATA_CHECK=0 skips it, D2_PREFLIGHT_SOURCES picks sources.
#   setsid nohup bash $Q/d2_launch.sh run > $LEX/distill/d2/logs/queue.log 2>&1 < /dev/null &
#                                    the queue.  It WAITS (never competes) for every gate, then runs D2 to the end.
#   bash $Q/d2_launch.sh register    only the OMNI1D2 registration, from a sealed $D2/d2.trained (after the run parked
#                                    for a missing recipe tool: WANT_RECIPE=<sha> bash $Q/d2_launch.sh register).
#   bash $Q/d2_launch.sh status      the last status lines.
#
# ORDER (run):  gates -> A2 finalized (automatic once C1/C2 are sealed: d2_plan a2-finalize fills the two cache shas,
#   backs the prereg up and re-checks the frozen-body pin) -> P1 plan (oc_checkpoint delta-compose of the D1 clone) ->
#   data config + data-check (row plan, total steps) -> card locks + host gate -> SHORT run (rungs 5M / 12M / 20M real
#   tokens; the K3 rows are a deferred source that joins at 20M) -> on exit 4 (K3 not sealed yet) the cards are
#   RELEASED, the queue waits for the 4 K3 passes, builds and verifies their xtok targets (d2_plan k3-seal) and resumes
#   -> LONG rung (31 x 32,768, init from the short run's adapters; only if P2's L32768 cell completed) -> seal markers
#   -> delta registration of OMNI1D2 on the D1 clone (oc_checkpoint register --delta-base, DRY RUN then --confirm),
#   which needs a pinned d2_delta_recipe.py (NOT BUILT as of 1.1.0: the queue parks there, adapters sealed and safe).
# DATA (1.1.0, BFCL-62 holdout decision, Claude lead 2026-10-14, option a): d2_masks_v1 by d2_mask_build 1.1.0 excludes
#   76 registry rows (69 BFCL-62 source rows + 7 rows of their text twins), the K3 row holding the K3 twin (pass 2,
#   row 10) and any self-gen row from a twin prompt (none rescored); the BFCL-native source is the 500-row copy
#   (d2_bfcl_native_copy 1.1.0: 506 minus 6 task twins).  All pinned by sha in d2_plan.py.
# GATES: tools pinned by sha (WANT_*); check-prereg prints the frozen pin c24d21e0...1370; A2 in force; the D2 base
#   ($D2_BASE_ID, the D1 clone) is in the ledger; a P2 verdict ($Q/p2.verdict, or derived from the P2 summaries under
#   the standing GO and written with by=auto); every start input sealed (d2_plan inventory); cards free of compute
#   and their $Q/gpu<N>.lock taken (flock, waited for); MemAvailable >= D2_MEM_GIB; root >= 60 GB, Lexar >= 120 GB.
# ENGINES (G5-1): streamed = two-replica DP on GPU0 + GPU4 (bf16, torchrun, the winning P2 cell's micro/accumulation
#   per card); split = one replica over GPU6 + GPU4 (fp16, loss scale 2^10).  Global tokens per step 65,536 either way.
# WRITES: $LEX/distill/d2/** (runs, safety checkpoints, logs, receipts), $Q/d2.status, $Q/p2.verdict (only when derived),
#   the prereg's A2 block (a2-finalize only, with a backup).  Nothing on root but status lines.
# CPU helpers run at nice 15 with CUDA hidden.  A HALT (exit 3: persistent non-finite steps) stops the queue for a
#   ruling; a refusal (exit 2) stops it; a crash is retried 3 times from the newest safety checkpoint.
set -u
VERSION="d2_launch-1.1.2"   # 1.1.2 2026-10-15 (adversarial review): the streamed engine runs under torchrun, which exits 1
                            #   for ANY worker failure, so the trainer's 4 (K3 deferred) / 3 (halted) / 2 (refused) never reached
                            #   the case below: a K3 wait was retried 3 times and then stopped the queue.  trainer_rc recovers
                            #   the trainer's own code from its log.  A k3-seal REFUSAL (rc 2) now stops the queue with a
                            #   status line instead of being retried every 30 min for ever.
                            # 1.1.1 2026-10-15: preflight partial config passes exclude_rows to self-gen (as the run does); K3 INFO parsed
                            # 1.1.0 2026-10-14: twin-aware inputs (d2_plan 1.1.0), registration sequence wired
                            # 1.0.0 2026-10-14: never deployed; _superseded/d2_launch_1.0.0.sh
MODE=${1:-preflight}
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
Q=$OC/distill/queues
LEX=/mnt/lexar/omni-campaign
D2=$LEX/distill/d2
LOGS=$D2/logs
STATUS=$Q/d2.status
PY=/home/cereal/model-surgery-lab/venv/bin/python
M27=/home/cereal/models/Qwen3.8-27B-Uncensored
PREREG=$OC/distill/PREREGISTRATION_OMNI_STAGE_D_v1.md
LEDGER=$OC/ledger/oc_checkpoint.py
D2_BASE_ID=${D2_BASE_ID:-OC02-D-Q38-27BU-OMNI1D1.clone-pre-D2}
PIN_BODY=c24d21e08815d7acd2ce14aea989d16f84f167eb24a40597966be19284cb1370
WANT_T=${WANT_T:-ac45af12995c4997f2d448dd2220729a0ec6d241c129588a69adbdb710a31716}          # streamed_kd_trainer.py 1.3.4
WANT_PLAN=${WANT_PLAN:-771adcce8165620f83988366a1fa0fb93689f3376268946519e94e21661f7b50} # d2_plan.py 1.1.0
WANT_RECIPE=${WANT_RECIPE:-}          # d2_delta_recipe.py: not built; registration parks until it exists and is pinned
D2_ID=${D2_ID:-OC02-D-Q38-27BU-OMNI1D2}
POLL_S=${POLL_S:-600}
WAIT_CAP_S=${WAIT_CAP_S:-1814400}     # 21 days
RETRIES=${RETRIES:-3}
ROOT_MIN_GB=60
LEX_MIN_GB=120
SHORT_RUN=$D2/run-short
SHORT_SAFETY=$D2/safety-short
LONG_RUN=$D2/run-long
LONG_SAFETY=$D2/safety-long
GLOBAL_TOKENS=65536

utc() { date -u +%FT%TZ; }
status() { case $MODE in run|register) echo "$(utc) $VERSION $*" >> "$STATUS";; esac; echo "[$(utc)] $*"; }
cpu() { env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OC=$OC LEX=$LEX PYTHONPATH=$T nice -n 15 "$@"; }
plan() { cpu $PY $T/d2_plan.py "$@"; }
avail_gib() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }
gpu_uuid() { nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$1" '$1==g{print $2}'; }
gpu_busy() {  # $1 index: a compute process on the card, or more than 2,000 MiB used
  local uuid used
  uuid=$(gpu_uuid "$1"); used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -dc 0-9)
  [ -z "$uuid" ] && { echo "no such GPU"; return 0; }
  if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -q "^$uuid"; then echo "compute process present"; return 0; fi
  [ "${used:-0}" -gt 2000 ] && { echo "${used} MiB used"; return 0; }
  return 1
}
lock_free() { ( flock -n 9 ) 9> "$Q/gpu$1.lock"; }
trainer_rc() {  # $1 the launcher's exit code, $2 the run's log -> the TRAINER's exit code (1.1.2)
  # torchrun (engine streamed) exits 1 whenever a worker exits non-zero; the trainer prints its outcome on its
  # own lines first: `TRAIN DEFERRED` (4), `TRAIN HALTED` (3), `REFUSED:` (2).  Anything else keeps the raw code.
  if [ "$1" -eq 0 ]; then echo 0
  elif grep -q '^TRAIN DEFERRED' "$2" 2>/dev/null; then echo 4
  elif grep -q '^TRAIN HALTED' "$2" 2>/dev/null; then echo 3
  elif grep -q '^REFUSED' "$2" 2>/dev/null; then echo 2
  else echo "$1"; fi
}

# ------------------------------------------------------------------ gates (shared by preflight and run)
GATE_OPEN=0; GATE_WAIT=0; GATE_REFUSED=0; WAITING=""
gate() {  # $1 name $2 OPEN|WAIT|REFUSED $3 detail
  printf '  %-12s %-8s %s\n' "$1" "$2" "$3"
  case $2 in OPEN) GATE_OPEN=$((GATE_OPEN+1));; WAIT) GATE_WAIT=$((GATE_WAIT+1)); WAITING="$WAITING $1";; *) GATE_REFUSED=$((GATE_REFUSED+1)); WAITING="$WAITING $1!";; esac
}

check_gates() {  # $1 = preflight|run ; sets ENGINE MB ACC CARDS LONG_OK and the gate counters
  GATE_OPEN=0; GATE_WAIT=0; GATE_REFUSED=0; WAITING=""
  echo "D2 GATES $(utc) ($VERSION, mode $1)"
  # tools
  local t p r
  t=$(sha256sum $T/streamed_kd_trainer.py | cut -c1-64); p=$(sha256sum $T/d2_plan.py | cut -c1-64)
  if [ "$t" = "$WANT_T" ] && [ "$p" = "$WANT_PLAN" ]; then gate tools OPEN "trainer ${t:0:16} ($(grep -m1 '^TOOL_VERSION' $T/streamed_kd_trainer.py | cut -d'"' -f2)), d2_plan ${p:0:16}"
  else gate tools REFUSED "trainer ${t:0:16} (want ${WANT_T:0:16}), d2_plan ${p:0:16} (want ${WANT_PLAN:0:16})"; fi
  # prereg pin + A2
  r=$(cd /tmp && cpu $PY $T/stage2_training_receipt.py check-prereg $PREREG 2>&1)
  if echo "$r" | grep -q "PREREG OK.*$PIN_BODY.*" || echo "$r" | grep -q "\"preregistration_sha256\": \"$PIN_BODY\""; then
    local a2; a2=$(plan a2-state 2>&1 | grep '^D2PLAN a2-state' | sed 's/^D2PLAN a2-state //')
    if echo "$a2" | grep -q '"in_force": true'; then gate prereg OPEN "frozen pin ${PIN_BODY:0:16}; A2 in force"
    else gate prereg WAIT "frozen pin ${PIN_BODY:0:16} OK; A2 is a DRAFT (placeholders $(echo "$a2" | grep -o '"placeholders": \[[^]]*\]' | cut -d: -f2)); finalized automatically once C1/C2 seal"; fi
  else gate prereg REFUSED "check-prereg: $(echo "$r" | tail -1 | cut -c1-200)"; fi
  # D2 base in the ledger
  local base
  base=$(cd /tmp && OC_ROOT=$OC timeout 120 $PY $LEDGER show --checkpoint-id "$D2_BASE_ID" 2>&1)
  if echo "$base" | grep -q '"directory_sha256"'; then gate d1-base OPEN "$D2_BASE_ID registered"
  else gate d1-base WAIT "$D2_BASE_ID not in the ledger yet (D1 seal + register + clone; WAVE3 ETA 10-15 20:00-22:00Z)"; fi
  # P2 verdict
  local v vline
  if [ "$1" = run ]; then v=$(plan verdict --write 2>&1); else v=$(plan verdict 2>&1); fi
  vline=$(echo "$v" | grep '^D2PLAN verdict' | sed 's/^D2PLAN verdict //')
  ENGINE=$(echo "$vline" | grep -o '"engine": "[a-z]*"' | cut -d'"' -f4)
  MB=$(echo "$vline" | grep -o '"micro_batch": "[0-9]*"' | cut -d'"' -f4)
  ACC=$(echo "$vline" | grep -o '"accumulation": "[0-9]*"' | cut -d'"' -f4)
  LONG_OK=$(echo "$vline" | grep -o '"long_rung_cell_completed": "[a-z]*"' | cut -d'"' -f4)
  if [ -n "$ENGINE" ]; then gate p2-verdict OPEN "engine $ENGINE micro $MB acc $ACC (per replica) long-cell ${LONG_OK:-?}; $(echo "$vline" | grep -o '"source": "[^"]*"')"
  else gate p2-verdict WAIT "$(echo "$vline" | grep -o '"waiting": "[^"]*"' | cut -c1-200)"; fi
  # inputs
  local inv
  inv=$(plan inventory ${INVENTORY_OUT:+--out "$INVENTORY_OUT"} 2>&1 | grep '^D2PLAN inventory' | sed 's/^D2PLAN inventory //')
  if echo "$inv" | grep -q '"waiting": \[\]'; then gate inputs OPEN "$(echo "$inv" | cut -c1-300)"
  else gate inputs WAIT "$(echo "$inv" | cut -c1-600)"; fi
  # cards and host (engine decides which cards)
  case "$ENGINE" in streamed) CARDS="0 4"; MEM_GIB=${D2_MEM_GIB:-100};; split) CARDS="6 4"; MEM_GIB=${D2_MEM_GIB:-60};; *) CARDS=""; MEM_GIB=${D2_MEM_GIB:-100};; esac
  local c busy
  for c in $CARDS; do
    busy=$(gpu_busy $c)
    if [ -n "$busy" ]; then gate "gpu$c" WAIT "$busy"
    elif ! lock_free $c; then gate "gpu$c" WAIT "$Q/gpu$c.lock held"
    else gate "gpu$c" OPEN "idle, lock free"; fi
  done
  [ -z "$CARDS" ] && gate cards WAIT "no engine yet (P2 verdict)"
  if [ "$(avail_gib)" -ge "$MEM_GIB" ]; then gate host-ram OPEN "MemAvailable $(avail_gib) GiB >= $MEM_GIB"
  else gate host-ram WAIT "MemAvailable $(avail_gib) GiB < $MEM_GIB"; fi
  if [ "$(free_gb /)" -ge $ROOT_MIN_GB ] && [ "$(free_gb /mnt/lexar)" -ge $LEX_MIN_GB ]; then gate disk OPEN "root $(free_gb /) GB, lexar $(free_gb /mnt/lexar) GB"
  else gate disk WAIT "root $(free_gb /) GB (floor $ROOT_MIN_GB), lexar $(free_gb /mnt/lexar) GB (need $LEX_MIN_GB)"; fi
  # not start gates, reported
  local k3; k3=$(plan k3-state 2>&1 | grep '^D2PLAN k3-state' | sed 's/^D2PLAN k3-state //' \
    | $PY -c 'import json,sys; d=json.load(sys.stdin); print("K3 passes sealed by the frozen tool %d/4, seal marker %s" % (d["sealed"], d["marker_exists"]))' 2>&1)
  echo "  (k3 rows)    INFO     $k3 -- deferred to the 20M rung; the short run waits there (exit 4) if not sealed"
  if [ -f "$T/d2_delta_recipe.py" ]; then echo "  (register)   INFO     d2_delta_recipe.py present $(sha256sum $T/d2_delta_recipe.py | cut -c1-16) (want ${WANT_RECIPE:0:16})"
  else echo "  (register)   INFO     d2_delta_recipe.py NOT BUILT: the queue trains, seals the adapters, then parks before registration"; fi
  echo "D2 GATES open $GATE_OPEN wait $GATE_WAIT refused $GATE_REFUSED:${WAITING:- none}"
}

# ------------------------------------------------------------------ registration (run's last step, or MODE=register)
register_d2() {  # OMNI1D2 registration from $D2/d2.trained; returns 0 (registered or parked) or 2 (refused)
  # ---- registration: OMNI1D2 = the D2 adapters as a delta on the D1 clone, the D1 seal queue's A4/A5 pattern.
  # CONTRACT for d2_delta_recipe.py (not built in 1.1.0): `d2_delta_recipe.py --trained $D2/d2.trained --base-record FILE
  #   --base-checkpoint-id ID --delta-dir DIR` writes DIR/DELTA.json (neural-forge-omni-delta-recipe-v1: lora / scale ops on
  #   the BASE's tensor names, base id + sealed directory_sha256) and the tensors it names, nothing else; exit 2 = refused.
  local REG DELTA head BASE_SHA rc_cmd notes ARGS ENGINE MB ACC DATA
  ENGINE=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["run"]["engine"])' "$D2/d2.trained")
  MB=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["run"]["micro_batch"])' "$D2/d2.trained")
  ACC=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["run"]["accumulation"])' "$D2/d2.trained")
  DATA=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["run"]["data_config"])' "$D2/d2.trained")
  if [ ! -f "$T/d2_delta_recipe.py" ] || [ -z "$WANT_RECIPE" ] || [ "$(sha256sum $T/d2_delta_recipe.py | cut -c1-64)" != "$WANT_RECIPE" ]; then
    status "PARKED before registration: d2_delta_recipe.py (adapters -> DELTA.json lora+scale ops on Chimera names) is not built / not pinned; adapters are sealed in $D2/d2.trained. Then: WANT_RECIPE=<sha> bash $Q/d2_launch.sh register (needs only $D2/d2.trained; no card)"
    return 0
  fi
  REG=$D2/register; DELTA=$D2/delta
  mkdir -p "$REG" "$DELTA"
  head=$(cd /tmp && $PY $LEDGER --root $OC show --checkpoint-id "$D2_ID" 2>/dev/null | $PY -c 'import json,sys; print(json.load(sys.stdin).get("record_instance_id",""))' 2>/dev/null)
  if [ -n "$head" ]; then status "$D2_ID already registered (head $head); nothing to do"; return 0; fi
  ( cd /tmp && $PY $LEDGER --root $OC show --checkpoint-id "$D2_BASE_ID" ) > "$REG/base_record.json" 2> "$LOGS/register.log" \
    || { status "REFUSED: oc_checkpoint show $D2_BASE_ID failed"; return 2; }
  BASE_SHA=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["directory_sha256"])' "$REG/base_record.json")
  if [ ! -f "$DELTA/DELTA.json" ]; then
    ( cd /tmp && cpu $PY $T/d2_delta_recipe.py --trained "$D2/d2.trained" --base-record "$REG/base_record.json" \
        --base-checkpoint-id "$D2_BASE_ID" --delta-dir "$DELTA" ) >> "$LOGS/register.log" 2>&1 \
      || { status "REFUSED: d2_delta_recipe (see $LOGS/register.log)"; return 2; }
  fi
  $PY - "$DELTA/DELTA.json" "$D2_BASE_ID" "$BASE_SHA" >> "$LOGS/register.log" 2>&1 <<'PYEOF' || { status "REFUSED: DELTA.json does not name the sealed D2 base or holds ops other than lora / scale"; return 2; }
import json, sys
r = json.load(open(sys.argv[1]))
ops = {e["op"] for e in r["entries"]}
ok = r["base_checkpoint_id"] == sys.argv[2] and r["base_directory_sha256"] == sys.argv[3] and r["entries"] and ops <= {"lora", "scale"}
print("DELTA.json", len(r["entries"]), "entries", sorted(ops), "base", r["base_checkpoint_id"], r["base_directory_sha256"][:16], "ok" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
PYEOF
  rc_cmd="streamed_kd_trainer $(grep -m1 '^TOOL_VERSION' $T/streamed_kd_trainer.py | cut -d'"' -f2) (sha ${WANT_T:0:16}) under $VERSION, engine $ENGINE micro $MB acc $ACC; data $DATA (sha $(sha256sum $DATA | cut -c1-16)); trained record $D2/d2.trained (sha $(sha256sum $D2/d2.trained | cut -c1-16)); DELTA.json by d2_delta_recipe (sha ${WANT_RECIPE:0:16})"
  notes="Stage D2 end-to-end LoRA KD of OMNI1D1 (prereg PREREGISTRATION_OMNI_STAGE_D_v1 + A1 + A2, frozen-body sha ${PIN_BODY:0:16}...). Delta on $D2_BASE_ID @${BASE_SHA:0:16}. BFCL-62 rows and their task twins excluded (A2 3a). Registered unattended by $VERSION under the standing GO (STAGE_D_RULINGS ruling 1); functional checks only, fit numbers are diagnostics (2026-10-06)."
  ARGS=(register --checkpoint-id "$D2_ID" --stage distill --wave oc02 --tag OMNI1D2 --parent-code Q38-27BU
        --directory "$DELTA" --delta-base "$D2_BASE_ID" --delta-base-sha256 "$BASE_SHA" --parent-directory "$M27"
        --parameter-count auto --parent-parameter-count auto --recipe "$DELTA/DELTA.json" --recipe-command "$rc_cmd" --notes "$notes")
  ( cd /tmp && timeout 3600 cpu $PY $LEDGER --root $OC "${ARGS[@]}" ) > "$REG/register_dryrun.log" 2>&1 \
    || { status "REFUSED: register DRY RUN: $(grep -m1 REFUSED "$REG/register_dryrun.log" | cut -c1-300)"; return 2; }
  status "register DRY RUN ok ($REG/register_dryrun.log sha $(sha256sum "$REG/register_dryrun.log" | cut -c1-16)); confirming"
  ( cd /tmp && timeout 3600 cpu $PY $LEDGER --root $OC "${ARGS[@]}" --confirm ) > "$REG/register_confirm.log" 2>&1 \
    || { status "REFUSED: register --confirm: $(grep -m1 REFUSED "$REG/register_confirm.log" | cut -c1-300)"; return 2; }
  head=$(cd /tmp && $PY $LEDGER --root $OC show --checkpoint-id "$D2_ID" | $PY -c 'import json,sys; print(json.load(sys.stdin)["record_instance_id"])')
  [ "$head" = "$D2_ID-0001-register" ] || { status "REFUSED: after register --confirm the head is '$head', not $D2_ID-0001-register"; return 2; }
  status "REGISTERED $D2_ID (head $head) as a delta on $D2_BASE_ID @${BASE_SHA:0:16}; next: clone-pre-D3 (deliberate ledger step)"
  return 0
}

# ------------------------------------------------------------------ preflight
if [ "$MODE" = status ]; then tail -20 "$STATUS" 2>/dev/null; exit 0; fi
if [ "$MODE" = preflight ]; then
  check_gates preflight
  if [ "${D2_PREFLIGHT_DATA_CHECK:-1}" = 1 ]; then
    # the registry sources that exist, loaded exactly as the trainer will (read-only; ~4 sources by default to keep
    # the page cache quiet beside a live D1; D2_PREFLIGHT_SOURCES=all for every one)
    tmpd=$(mktemp -d /tmp/d2-preflight.XXXXXX)
    if plan data-config --out "$tmpd/data_short.json" > "$tmpd/dc.log" 2>&1; then
      cfg=$tmpd/data_short.json; echo "  data-config  written (all inputs sealed)"
    else
      # partial: the registry part, from the inventory alone
      cpu $PY - "$tmpd/partial.json" <<'PYEOF'
import json, os, sys
sys.path.insert(0, os.environ["PYTHONPATH"])
import d2_plan as P
layout = P.Layout()
inv = P.inventory(layout)
excl = layout.masks / "exclude_rows.json"
sources = [{"name": f"registry-{r['file'].split('.')[0]}", "kind": "text", "qtext_cache": r["dir"], "budget_unit": "positions",
            "tokens": (r["row_range"][1] - r["row_range"][0]) * P.SEQ, "exclude_rows": str(excl),
            "gt_mask": str(layout.masks / "gt_mask.json"), "all_mask": str(layout.masks / "all_mask.json")}
           for r in inv["registry"] if r["sealed"]]
if inv["selfgen"]["sealed"]:
    sources.append({"name": "selfgen", "kind": "text", "qtext_cache": inv["selfgen"]["dir"], "budget_unit": "real_tokens",
                    "tokens": int(inv["selfgen"]["tokens"]), "exclude_rows": str(excl)})
json.dump({"sources": sources}, open(sys.argv[1], "w"), indent=1)
print(f"partial data config: {len(sources)} sources")
PYEOF
      cfg=$tmpd/partial.json; echo "  data-config  WAIT (inputs pending); partial config of the sealed registry + self-gen sources for the check"
    fi
    if [ "${D2_PREFLIGHT_SOURCES:-}" = all ]; then pick=""
    else pick=${D2_PREFLIGHT_SOURCES:-registry-text-00000,registry-text-00002,registry-text-00019,selfgen}; fi
    if [ -s "$cfg" ] && [ "$(avail_gib)" -ge 20 ]; then
      ( cd /tmp && cpu ionice -c3 timeout 3600 $PY $T/streamed_kd_trainer.py data-check --data-config "$cfg" --seq-len 4096 \
          --max-width 4864 --pad-id 248044 --rows-per-step 16 ${pick:+--sources $pick} --receipt "$tmpd/data_check.json" ) 2>&1 \
        | grep -E '^(DATA-CHECK|REFUSED)' | cut -c1-700
      echo "  data-check receipt (temporary) $tmpd/data_check.json"
    else
      echo "  data-check skipped (no config or MemAvailable < 20 GiB)"
    fi
  fi
  [ $GATE_REFUSED -gt 0 ] && exit 2
  [ $GATE_WAIT -gt 0 ] && exit 10
  exit 0
fi
if [ "$MODE" = register ]; then  # only the registration, from a sealed $D2/d2.trained (no card, no gates but the tools pin)
  [ -s "$D2/d2.trained" ] || { echo "no $D2/d2.trained: D2 has not finished training"; exit 10; }
  [ "$(sha256sum $T/streamed_kd_trainer.py | cut -c1-64)" = "$WANT_T" ] || { echo "trainer is not the pinned ${WANT_T:0:16}"; exit 2; }
  register_d2; exit $?
fi
[ "$MODE" = run ] || { echo "usage: d2_launch.sh preflight|run|register|status"; exit 2; }

# ================================================================== run (the queue)
mkdir -p "$D2" "$LOGS"
exec 3> "$Q/d2.lock"
flock -n 3 || { echo "another d2 queue holds $Q/d2.lock"; exit 5; }
status "queue start pid=$$ base=$D2_BASE_ID trainer=${WANT_T:0:16} plan=${WANT_PLAN:0:16}"
waited=0; next_note=0
while :; do
  out=$(check_gates run 2>&1)
  if echo "$out" | grep -q "A2 is a DRAFT"; then plan a2-finalize >> "$LOGS/a2_finalize.log" 2>&1 && out=$(check_gates run 2>&1); fi
  if echo "$out" | grep -q "refused [1-9]"; then status "REFUSED at the gates: $(echo "$out" | grep REFUSED | tr '\n' ' ' | cut -c1-600)"; echo "$out"; exit 2; fi
  if echo "$out" | grep -q "wait 0 refused 0"; then status "gates open: $(echo "$out" | tail -1)"; echo "$out"; break; fi
  if [ $waited -ge $next_note ]; then status "waiting ${waited}s: $(echo "$out" | tail -1 | cut -c1-400)"; next_note=$((next_note + 3600)); fi
  [ $waited -ge $WAIT_CAP_S ] && { status "gave up after ${waited}s at the gates"; exit 10; }
  sleep $POLL_S; waited=$((waited + POLL_S))
done
check_gates run > /dev/null 2>&1   # sets ENGINE MB ACC CARDS LONG_OK MEM_GIB
WORLD=1; [ "$ENGINE" = streamed ] && WORLD=2
[ $((MB * ACC * WORLD * 4096)) -eq $GLOBAL_TOKENS ] || ACC=$((GLOBAL_TOKENS / (MB * WORLD * 4096)))
ROWS_PER_STEP=$((MB * ACC * WORLD))

# ---- P1 plan of the D2 base (base OMNI1.clone-pre-D1 + the D1 delta)
P1PLAN=$D2/p1_plan.json
if [ ! -s "$P1PLAN" ]; then
  ( cd /tmp && OC_ROOT=$OC cpu $PY $LEDGER delta-compose --checkpoint-id "$D2_BASE_ID" --deep --plan-out "$P1PLAN" ) > "$LOGS/delta_compose.log" 2>&1 \
    || { status "REFUSED: delta-compose of $D2_BASE_ID failed (see $LOGS/delta_compose.log)"; exit 2; }
fi
ROOT=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"]["directory"])' "$P1PLAN")
status "p1 plan $P1PLAN sha $(sha256sum $P1PLAN | cut -c1-16) root $ROOT"

# ---- data config (written once; a resume refuses a changed one anyway) and the row plan
DATA=$D2/data_short.json
if [ ! -s "$DATA" ]; then plan data-config --kind short --out "$DATA" > "$LOGS/data_config.log" 2>&1 || { status "REFUSED: data-config (see $LOGS/data_config.log)"; exit 2; }; fi
MAXW=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_width"])' "$DATA")
DEFERRED_ROWS=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("deferred_rows", 0))' "$DATA")
( cd /tmp && cpu $PY $T/streamed_kd_trainer.py data-check --data-config "$DATA" --seq-len 4096 --max-width "$MAXW" --pad-id 248044 \
    --rows-per-step $ROWS_PER_STEP --receipt "$D2/data_check_short.json" ) > "$LOGS/data_check_short.log" 2>&1 \
  || { status "REFUSED: data-check (see $LOGS/data_check_short.log)"; exit 2; }
PLANNED=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["planned_rows_loaded_sources"])' "$D2/data_check_short.json")
STEPS=$(( (PLANNED + DEFERRED_ROWS + ROWS_PER_STEP - 1) / ROWS_PER_STEP ))
status "data $DATA sha $(sha256sum $DATA | cut -c1-16) max_width $MAXW rows $PLANNED + deferred K3 $DEFERRED_ROWS -> $STEPS steps of $ROWS_PER_STEP rows ($ENGINE, micro $MB acc $ACC world $WORLD)"

COMMON=(--kernels auto --seq-len 4096 --max-width "$MAXW" --pad-id 248044 --count-real-tokens --micro-batch "$MB" --accumulation "$ACC"
        --total-steps "$STEPS" --lr 1e-4 --gain-lr 1e-3 --warmup-steps 20 --min-lr-ratio 0.1 --grad-clip 1.0 --weight-decay 0
        --seed 20261012 --lora-rank 64 --k3-lora-rank 32 --trunk-ffn-width 4352 --w-q38 1.0 --lambda-k3 0.3 --k3-q38-anchor 0.5
        --w-ce 0.1 --w-eos 0.1 --w-ul 0.1 --onpolicy-divergence jsd --jsd-beta 0.5 --rungs 5000000,12000000,20000000
        --safety-minutes 30 --safety-keep 2 --resume auto --preregistration "$PREREG"
        --p1-chimera "$ROOT" --p1-parent-config "$M27" --p1-plan "$P1PLAN" --log-every 1)

take_cards() {  # waits for each card's lock (never competes), then the card idle and the host gate; sets CUDA_VISIBLE_DEVICES
  local c fd=7 uuids=""
  for c in $CARDS; do
    eval "exec $fd> $Q/gpu$c.lock"
    status "waiting for $Q/gpu$c.lock"
    flock $fd
    fd=$((fd + 1))
  done
  while :; do
    local busy=""
    for c in $CARDS; do b=$(gpu_busy $c); [ -n "$b" ] && busy="$busy gpu$c:$b"; done
    [ -z "$busy" ] && [ "$(avail_gib)" -ge "$MEM_GIB" ] && break
    status "cards held, waiting:${busy} MemAvailable $(avail_gib) GiB (need $MEM_GIB)"; sleep 300
  done
  for c in $CARDS; do uuids="$uuids,$(gpu_uuid $c)"; done
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${uuids#,} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  status "cards ${CARDS} taken (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES), MemAvailable $(avail_gib) GiB"
}
release_cards() { exec 7>&- 8>&-; unset CUDA_VISIBLE_DEVICES; status "cards ${CARDS} released"; }

train() {  # $1 tag $2 out $3 safety $4 data $5.. extra; returns the trainer's exit code
  local tag=$1 out=$2 safety=$3 data=$4; shift 4
  local log=$LOGS/train_${tag}_$(date -u +%Y%m%dT%H%M%SZ).log
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src TOKENIZERS_PARALLELISM=false
  status "train $tag start log=$log"
  if [ "$ENGINE" = streamed ]; then
    nice -n 5 $PY -m torch.distributed.run --standalone --nproc_per_node 2 $T/streamed_kd_trainer.py train --engine streamed \
      --devices cuda:0,cuda:1 --dtype bfloat16 "${COMMON[@]}" --data-config "$data" --out "$out" --safety-dir "$safety" "$@" > "$log" 2>&1
  else
    nice -n 5 $PY $T/streamed_kd_trainer.py train --engine split --devices cuda:0,cuda:1 --dtype float16 --initial-loss-scale 1024 \
      --loss-scale-growth-interval 200 "${COMMON[@]}" --data-config "$data" --out "$out" --safety-dir "$safety" "$@" > "$log" 2>&1
  fi
  local raw=$? rc
  rc=$(trainer_rc "$raw" "$log")
  status "train $tag exit=$rc (launcher rc $raw) $(grep -E '^(TRAIN|OOM|REFUSED)' "$log" | tail -1 | cut -c1-300)"
  return $rc
}

# ---- SHORT run (rungs 5M / 12M / 20M; K3 rows deferred to 20M behind $D2/k3.sealed)
take_cards
tries=0
while :; do
  train short "$SHORT_RUN" "$SHORT_SAFETY" "$DATA"; rc=$?
  case $rc in
    0) status "short run done"; break ;;
    4) release_cards
       status "K3 wait: the short run reached its K3 rows before the K3 cache was sealed"
       while :; do
         plan k3-seal >> "$LOGS/k3_seal.log" 2>&1; krc=$?
         [ $krc -eq 0 ] && break
         [ $krc -eq 10 ] || { status "REFUSED: d2_plan k3-seal rc $krc (see $LOGS/k3_seal.log); queue stopped, the short run is checkpointed in $SHORT_SAFETY (bash d2_launch.sh run resumes it)"; exit 2; }
         sleep 1800
       done
       status "K3 sealed: $(sha256sum $D2/k3.sealed | cut -c1-16)"
       take_cards ;;
    3) status "HALT: persistent non-finite steps; resume after a ruling (bash d2_launch.sh run resumes from $SHORT_SAFETY)"; exit 3 ;;
    2) status "REFUSED by the trainer (see the log); queue stopped"; exit 2 ;;
    *) tries=$((tries + 1))
       [ $tries -gt $RETRIES ] && { status "short run failed $tries times (last rc $rc); queue stopped"; exit 1; }
       status "short run rc $rc: retry $tries of $RETRIES from the newest safety checkpoint in 600 s"; sleep 600 ;;
  esac
done
SHORT_ADAPTERS=$SHORT_RUN/adapters.safetensors

# ---- LONG rung: 31 x 32,768 (micro 1 per replica), KL to Q38 at L = 32,768, from the short run's adapters
if [ "$LONG_OK" != yes ]; then
  status "PARKED: long rung not run -- P2's L32768-mb1 cell did not complete (the 32k mixer half may not fit; needs a ruling)"
  exit 6
fi
LDATA=$D2/data_long.json
[ -s "$LDATA" ] || plan data-config --kind long --out "$LDATA" > "$LOGS/data_config_long.log" 2>&1 || { status "REFUSED: long data-config"; exit 2; }
LACC=$(( 2 / WORLD )); LSTEPS=$(( (31 + WORLD * LACC - 1) / (WORLD * LACC) ))
LCOMMON=()
skip=0
for a in "${COMMON[@]}"; do   # the long rung: seq 32,768, micro 1, no width override, no rungs, its own step count
  if [ $skip = 1 ]; then skip=0; continue; fi
  case "$a" in --seq-len|--max-width|--micro-batch|--accumulation|--total-steps|--rungs) skip=1; continue;; esac
  LCOMMON+=("$a")
done
COMMON=("${LCOMMON[@]}" --seq-len 32768 --micro-batch 1 --accumulation "$LACC" --total-steps "$LSTEPS")
tries=0
while :; do
  train long "$LONG_RUN" "$LONG_SAFETY" "$LDATA" --init-adapters "$SHORT_ADAPTERS"; rc=$?
  case $rc in
    0) status "long rung done"; break ;;
    3|2) status "long rung stopped rc $rc; queue stopped"; exit $rc ;;
    *) tries=$((tries + 1)); [ $tries -gt $RETRIES ] && { status "long rung failed $tries times; queue stopped"; exit 1; }
       status "long rung rc $rc: retry $tries in 600 s"; sleep 600 ;;
  esac
done
release_cards

# ---- seal markers, then registration (needs the D2 delta recipe tool)
$PY - "$D2/d2.trained" "$SHORT_RUN" "$LONG_RUN" "$ENGINE" "$MB" "$ACC" "$DATA" <<'PYEOF'
import hashlib, json, sys, time
from pathlib import Path
out, short, long_, engine, micro, acc, data = sys.argv[1:8]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec = {"schema": "omni-d2-trained-v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "run": {"engine": engine, "micro_batch": micro, "accumulation": acc, "data_config": data, "data_config_sha256": sha(data)},
       "short": {"adapters": f"{short}/adapters.safetensors", "sha256": sha(f"{short}/adapters.safetensors"),
                 "receipt_sha256": sha(f"{short}/training-receipt.json"),
                 "rungs": {p.parent.name: sha(p) for p in sorted(Path(short, "rungs").glob("rung-*/adapters.safetensors"))}},
       "long": {"adapters": f"{long_}/adapters.safetensors", "sha256": sha(f"{long_}/adapters.safetensors"),
                "receipt_sha256": sha(f"{long_}/training-receipt.json")}}
Path(out).write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec))
PYEOF
status "D2 trained: $D2/d2.trained sha $(sha256sum $D2/d2.trained | cut -c1-16)"

register_d2; exit $?
