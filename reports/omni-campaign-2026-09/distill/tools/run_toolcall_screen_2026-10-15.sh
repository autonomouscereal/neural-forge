#!/usr/bin/env bash
# Toolcall lane, step 7 (lane toolscreen; D2_DECISIONS_2026-10-15 rule D2-X1, prereg A2 3(d)): the held-out
# screen at the L-TOOL export, without touching the live extend driver.
#
# run_toolcall_extend_2026-10-15.sh (pid $1, toolcall_extend-1.0.0) finalizes run 1 + run 2, exports both into
# C1's corpus ($TC/rescore/toolcall-q38-2026-10-14.safetensors), releases GPU1/GPU2 and ends.  It has no screen step,
# and a running bash script must not be edited in place (bash reads it as it goes).  This watcher waits for that pid
# to end, then -- if the sidecar names the corpus (the export is sealed) -- runs toolcall_heldout_screen.py screen on
# it at once: hits are dropped from the export (corpus + sidecar rewritten, unscreened copy under
# /mnt/lexar/omni-campaign/distill/toolcall/unscreened, dropped list beside the corpus), and the receipt
# rescore/heldout_screen.json names the corpus C1 will rescore.  post_d1_queue >= 1.5.0 runs the same screen before
# C1 anyway (a no-op once this receipt is valid), so this watcher only moves the screen to export time; it never
# touches a card, a lock of another lane, or the driver.  CPU, nice 10.
#
#   setsid nohup bash run_toolcall_screen_2026-10-15.sh 1912373 > $OC/distill/toolcall/logs/screen.log 2>&1 < /dev/null &
# Exit: 0 screened (or already valid); 2 the pid is not the extend driver; 3 the driver ended without a sealed export
# (C1 then waits in post_d1 as before); 4 give-up time passed with the driver alive; else the screen's own rc.
set -u
VERSION=toolcall_screen-1.0.0
DRIVER_PID=${1:?usage: run_toolcall_screen_2026-10-15.sh <pid of run_toolcall_extend_2026-10-15.sh>}
# TOOLCALL_SCREEN_* are test hooks (test_run_toolcall_screen.sh); none is set in production.
OC=/home/cereal/omni-campaign
PY=${TOOLCALL_SCREEN_PY:-/home/cereal/model-surgery-lab/venv/bin/python}
T=$OC/distill/tools
Q=${TOOLCALL_SCREEN_Q:-$OC/distill/queues}
TC=${TOOLCALL_SCREEN_TC:-$OC/distill/toolcall}
SCREEN=${TOOLCALL_SCREEN_TOOL:-$T/toolcall_heldout_screen.py}
MATCH=${TOOLCALL_SCREEN_MATCH:-run_toolcall_extend_2026-10-15.sh}
DEST=$TC/rescore/toolcall-q38-2026-10-14.safetensors
GIVE_UP=${TOOLCALL_SCREEN_GIVE_UP:-2026-10-16T12:00:00Z}
POLL_S=${TOOLCALL_SCREEN_POLL_S:-60}
STATUS=$TC/logs/extend.status
mkdir -p "$TC/logs"
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
status() {  # one line to the L-TOOL status file + a lane heartbeat (never fatal, silent without a registry)
  stamp "STATUS $*"; echo "$(date -u +%FT%TZ) $VERSION $*" >> "$STATUS"
  [ -f "$Q/LANES.json" ] || return 0
  local d="$*"
  timeout 20 python3 "$T/lanes.py" --file "$Q/LANES.json" heartbeat --lane L-TOOL --agent claude --source "$VERSION" \
    --status "tool-call held-out screen ($VERSION): $1" --detail "${d:0:400}" > /dev/null 2>&1 || stamp "lane heartbeat failed (ignored)"
}
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
corpus_ready() {  # post_d1_queue's rule: the sidecar names this file's sha256
  local side="${DEST%.safetensors}.rows.json" digest
  [ -f "$DEST" ] && [ -f "$side" ] || return 1
  digest=$(sha "$DEST"); [ -n "$digest" ] && grep -q "$digest" "$side"
}

stamp "watcher start $VERSION; screen $SCREEN $(sha "$SCREEN" | cut -c1-16) ($("$PY" "$SCREEN" --version 2>&1 | tail -1)); driver pid $DRIVER_PID"
if kill -0 "$DRIVER_PID" 2> /dev/null; then
  tr '\0' ' ' < "/proc/$DRIVER_PID/cmdline" | grep -q -- "$MATCH" || { stamp "STOP: pid $DRIVER_PID is not $MATCH"; exit 2; }
  # the status file only: while the driver runs, run 2 is L-TOOL's activity and its lane status must say so
  stamp "STATUS waiting for the extend driver pid $DRIVER_PID"
  echo "$(date -u +%FT%TZ) $VERSION waiting for the extend driver pid $DRIVER_PID; the screen runs at its exit, then C1 (post_d1 >= 1.5.0) finds a valid receipt" >> "$STATUS"
  while kill -0 "$DRIVER_PID" 2> /dev/null; do
    if [ "$(date -u +%s)" -ge "$(date -u -d "$GIVE_UP" +%s)" ]; then
      status "gave up" "driver $DRIVER_PID still alive at $GIVE_UP: nothing screened here (post_d1 screens before C1)"; exit 4
    fi
    sleep "$POLL_S"
  done
fi
stamp "driver $DRIVER_PID gone; last driver status: $(grep -v "$VERSION" "$STATUS" 2>/dev/null | tail -1 | cut -c1-200)"
if ! corpus_ready; then
  status "no sealed export" "$DEST $([ -f "$DEST" ] && echo "sha $(sha "$DEST" | cut -c1-16) not named by its sidecar" || echo absent): nothing to screen; C1 waits in post_d1 as before"
  exit 3
fi
status "screening" "export $DEST sha $(sha "$DEST" | cut -c1-16)"
CUDA_VISIBLE_DEVICES= PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  TOKENIZERS_PARALLELISM=false nice -n 10 "$PY" "$SCREEN" screen --corpus "$DEST"
rc=$?
if [ $rc -eq 0 ] && "$PY" "$SCREEN" check --corpus "$DEST"; then
  R=$TC/rescore/heldout_screen.json
  status "screened" "corpus $(sha "$DEST" | cut -c1-16) receipt $(sha "$R" | cut -c1-16) dropped=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["hits_in_export"])' "$R" 2>/dev/null) dropped_list=$TC/rescore/heldout_screen.dropped.json"
  exit 0
fi
status "screen failed" "rc=$rc on $DEST (nothing dropped unless the receipt checks; post_d1 c1 re-runs the screen and fails alone if it fails again)"
exit $rc
