#!/usr/bin/env bash
# run_rescope.sh - executes the 8-unit rescope allowlist (lead review 2026-10-21, after Cyril's fold-init ruling)
# with cleanup_exec 1.3.0: lexar (incl. #210 with the surviving-clone digest check), then home, then nas.
# Stops at the first non-zero worker exit. Detached: setsid nohup bash run_rescope.sh > runner.out 2>&1 < /dev/null &
set -u
cd /home/cereal/omni-campaign/lab/cleanup
R=inputs/rescope-20261021
RUN=/mnt/lexar/omni-campaign/archive/cleanup-2026-10-21/rescope-20261021
PY=/home/cereal/.venvs/chimera-intake/bin/python
CLONE=/mnt/lexar/omni-campaign/clones/OC02-D-Q38-27BU-OMNI1D1.clone-pre-D2
CLONE_SHA=03da9ab938dab873f20474e634a6dc18060325b07a62e9121abf91b26184dece
set -a; . /home/cereal/.config/omni-lab/env; set +a
mkdir -p "$RUN"
utc() { date -u +%FT%TZ; }
snap() { echo "== $1 $(utc)"; df -B1 / /mnt/lexar /mnt/ai_data; }
echo "$(utc) start pid $$ tool $(sha256sum cleanup_exec.py | cut -c1-16)"
sha256sum "$R"/*.json
snap before > "$RUN/df_before.txt"
COMMON=(--units "$R/approved_rescope.json" --protected "$R/protected_rescope.json" --generic inputs/generic_roots.json
        --overrides "$R/overrides_rescope.json" --run-dir "$RUN")
for vol in lexar home nas; do
  extra=()
  [ "$vol" = lexar ] && extra=(--pre-delete-digest "210:$CLONE:$CLONE_SHA")
  snap "$vol-before" >> "$RUN/df_steps.txt"
  nice -n 10 ionice -c3 "$PY" cleanup_exec.py execute "${COMMON[@]}" --volume "$vol" "${extra[@]}" > "$RUN/exec_$vol.out" 2>&1
  rc=$?
  snap "$vol-after" >> "$RUN/df_steps.txt"
  echo "$(utc) $vol rc=$rc deleted=$(grep -c '"step": "DELETED"' "$RUN/exec_$vol.out") blocked=$(grep -c '"step": "BLOCKED"' "$RUN/exec_$vol.out")"
  [ $rc -eq 0 ] || { echo "$(utc) STOP: $vol worker failed"; break; }
done
/usr/bin/python3 /home/cereal/claude-scratch/cleanup-rescope-20261021/clone_digest.py "$CLONE" > "$RUN/clone_digest_after.json"
echo "$(utc) clone after: $(cat "$RUN/clone_digest_after.json")"
snap after > "$RUN/df_after.txt"
echo "$(utc) DONE"
