#!/usr/bin/env bash
# Lane p40 side job (Claude, wave 2, 2026-10-13): the PARENT REFERENCE for the production QVL block-13 cache.
# Installed as $OC/distill/queues/p40-qvlref.sh; run detached:
#   setsid nohup bash $OC/distill/queues/p40-qvlref.sh > $OC/distill/queues/p40-qvlref.log 2>&1 < /dev/null &
#
# Why: the production corpus (qvl_vit_stagec_d7492) shares no image with the 32-image s7_pairs_parent capture and
# smart_resize keeps aspect, so the production cache's s7 arm never fires; its only per-shard check is the
# determinism replay. This job runs the SAME qvl-run (unchanged 1.0.8 -> 1.0.9) with the SAME run config (config sha
# f895de5c...: block 13, max_pixels 512*512, fp32) over the 32 s7 capture images (512x512, so smart_resize keeps
# 512x512 and the arm fires on every item at side 512, gate rel_fro <= 2e-2), on the same card kind (GPU3, P40,
# Pascal venv), into a side cache under teacher_cache/qvl32_vit_ref/. media_b_teacher_cache 1.0.9 `check` of the
# production cache finds that side cache by its config sha and refuses the cache without it.
#
# GPU3 belongs to lane p40. This job does not touch p40.sh: it waits for p40-qvl-vit-gpu3 to exit 0, for step 1's
# pid 917640 to be gone, for GPU3 < 2000 MiB, and for no p40.sh GPU3 job to be in flight (START/JOB lines balanced),
# then takes GPU3 for ~3 min while p40.sh seals the production cache (CPU; its full teacher sha alone is ~9.5 min).
# Teacher integrity is 'stamp' (size + mtime) for this short side run; the production run holds the full sha256.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools/media_b_teacher_cache.py
PY=/home/cereal/model-surgery-lab/venv/bin/python
Q=$OC/distill/queues
STATUS=$Q/p40.status
ST=$Q/p40.state
LOGS=$OC/distill/media_b/logs_p40
REC=$OC/distill/media_b/receipts_p40
TC=/mnt/lexar/omni-campaign/teacher_cache
REFC=$OC/distill/media_b/corpora/qvl_s7ref_d32.json
QV=$OC/distill/media_b/corpora/qvl_vit_stagec_d7492.json
NAME=p40-qvl-ref-gpu3
STEP1_PID=917640
export OC CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9 PYTHONPATH=$OC/import/rq27b/tree_ef1ffb9/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

st() { echo "$(date -u +%FT%TZ) QVLREF $*" >> "$STATUS"; }
host() { nice -n 15 "$PY" "$T" "$@"; }
card_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -dc 0-9; }
step1_alive() { [ -r /proc/$STEP1_PID/cmdline ] && tr '\0' ' ' < /proc/$STEP1_PID/cmdline | grep -q run_review_finite_guard; }
gpu3_inflight() {  # number of p40.sh GPU3 jobs started and not yet finished
  local s j
  s=$(grep -c '^[^ ]* START .* gpu3 ' "$STATUS"); j=$(grep -c '^[^ ]* JOB .* gpu3 exit ' "$STATUS")
  echo $((s - j))
}

st "start p40-qvlref.sh sha $(sha256sum "$0" | cut -c1-16) tool $(sha256sum "$T" | cut -c1-16) $(host --version 2>&1)"
[ "$(host --version 2>&1)" = "media_b_teacher_cache 1.0.9" ] || { st "PREFLIGHT FAIL: tool is not 1.0.9"; exit 2; }
QVD=$(host address -- qvl-run --corpus "$QV" --cache-root "$TC/qvl32_vit" | sed -n 's/^CACHE_DIR //p')
REFD=$(host address -- qvl-run --corpus "$REFC" --cache-root "$TC/qvl32_vit_ref" | sed -n 's/^CACHE_DIR //p')
[ -n "$QVD" ] && [ -n "$REFD" ] || { st "PREFLIGHT FAIL: no cache address"; exit 2; }
[ "$(basename "$QVD")" = "$(basename "$REFD")" ] || { st "PREFLIGHT FAIL: run config differs: $QVD vs $REFD"; exit 2; }
st "ADDR production $QVD"
st "ADDR reference $REFD"

said=0
while [ ! -f "$ST/p40-qvl-vit-gpu3.rc" ]; do
  [ $said = 0 ] && st "WAIT $NAME: p40-qvl-vit-gpu3 still running" && said=1
  sleep 20
done
rc=$(cat "$ST/p40-qvl-vit-gpu3.rc")
[ "$rc" = 0 ] || { st "GATE $NAME skipped: production qvl-run exit $rc"; exit 1; }
said=0
while step1_alive || [ "$(card_mib 3)" -ge 2000 ] || [ "$(gpu3_inflight)" != 0 ]; do
  [ $said = 0 ] && st "WAIT $NAME: step1 $(step1_alive && echo alive || echo gone), GPU3 $(card_mib 3) MiB, p40.sh GPU3 jobs in flight $(gpu3_inflight)" && said=1
  sleep 15
done
st "START $NAME gpu3"
host launch --card 3 --name "$NAME" --log "$LOGS/$NAME.log" --runtime pascal --threads 8 -- \
  qvl-run --corpus "$REFC" --cache-root "$TC/qvl32_vit_ref" --integrity stamp >> "$LOGS/launch.log" 2>&1
rc=$?
echo $rc > "$ST/$NAME.rc"
st "JOB $NAME exit $rc log $LOGS/$NAME.log allocator $OC/composite/receipts/allocator/$NAME.json arms $(grep -c 'qvl reference arm .*rel_fro' "$LOGS/$NAME.log")"
[ $rc = 0 ] || exit 1
host seal --cache-dir "$REFD" --corpus "$REFC" --teacher qvl32_vit_l13 --integrity stamp >> "$LOGS/seal-qvl-ref.log" 2>&1
rc=$?
st "SEAL qvl-ref exit $rc $REFD"
[ $rc = 0 ] || exit 1
host check --cache-dir "$REFD" --out "$REC/qvl-ref.check.json" >> "$LOGS/check-qvl-ref.log" 2>&1
rc=$?
st "CHECK qvl-ref exit $rc $(grep -h 'CHECK \|REFUSED' "$LOGS/check-qvl-ref.log" | tail -1 | cut -c1-400)"
[ $rc = 0 ] || exit 1

# the production cache: p40.sh seals and checks it; if its check ran before this side cache was sealed (or failed
# for any other reason), check it again against the side cache by name.
said=0
until grep -q '^[^ ]* CHECK qvl-vit exit \|^[^ ]* GATE qvl-vit \|^[^ ]* SEAL qvl-vit exit [1-9]' "$STATUS"; do
  [ $said = 0 ] && st "WAIT production QVL seal + check (p40.sh)" && said=1
  sleep 30
done
if grep -q '^[^ ]* CHECK qvl-vit exit 0 ' "$STATUS"; then
  st "DONE p40.sh CHECK qvl-vit passed with parent reference $(grep -o '"source": "[^"]*"' "$LOGS/check-qvl-vit.log" | tail -1)"
  exit 0
fi
grep -q '^[^ ]* SEAL qvl-vit exit 0 ' "$STATUS" || { st "GATE production QVL not sealed; no re-check"; exit 1; }
host check --cache-dir "$QVD" --out "$REC/qvl-vit.check.parentref.json" --parent-reference "$REFD" >> "$LOGS/check-qvl-vit-parentref.log" 2>&1
rc=$?
st "CHECK qvl-vit-parentref exit $rc $(grep -h 'CHECK \|REFUSED' "$LOGS/check-qvl-vit-parentref.log" | tail -1 | cut -c1-400)"
exit $rc
