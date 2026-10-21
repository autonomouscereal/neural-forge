#!/usr/bin/env bash
# nas_finalize.sh - runs after the NAS cleanup workers exit: verify every recorded deletion, mirror the recipe
# archive to the NAS, take the post-wave omni_lab dump, snapshot df, and rebuild the cleanup report.
#   setsid nohup bash nas_finalize.sh > $R/nas_finalize.log 2>&1 < /dev/null &
set -u
R=/mnt/lexar/omni-campaign/archive/cleanup-2026-10-21
L=/home/cereal/omni-campaign/lab
PY=/home/cereal/.venvs/chimera-intake/bin/python
say() { echo "$(date -u +%FT%TZ) $*"; }
while pgrep -f "cleanup_exec.py execute .*--volume nas" > /dev/null; do sleep 300; done
say "NAS workers exited"
cd "$L" || exit 1
set -a; . /home/cereal/.config/omni-lab/env; set +a
$PY cleanup/verify_deleted.py > "$R/verify_final.txt" 2>&1; say "verify: $(tail -1 "$R/verify_final.txt")"
{ df -B1 / /mnt/lexar /mnt/ai_data; date -u +%FT%TZ; } > "$R/df_after_waves.txt"
free_kb=$(timeout 60 df -Pk /mnt/ai_data | awk 'NR > 1 {print $4}')
if [ "${free_kb:-0}" -gt 62914560 ]; then
  mkdir -p /mnt/ai_data/backups/omni-lab-recipes
  if timeout 7200 rsync -a /mnt/lexar/omni-campaign/archive/recipes/ /mnt/ai_data/backups/omni-lab-recipes/; then
    n=$(timeout 3600 rsync -a --checksum --dry-run --itemize-changes /mnt/lexar/omni-campaign/archive/recipes/ /mnt/ai_data/backups/omni-lab-recipes/ | wc -l)
    say "recipe mirror to /mnt/ai_data/backups/omni-lab-recipes: checksum dry-run differences=$n"
  else
    say "recipe mirror FAILED (rsync rc=$?)"
  fi
else
  say "recipe mirror skipped: NAS free_kb=${free_kb:-unknown}"
fi
DB_NAME=omni_lab /home/cereal/agent-memory-backup/bin/backup_pg_db.sh run >> /home/cereal/agent-memory-backup/logs/pg-db-backup.log 2>&1
say "omni_lab dump: $(cat /mnt/lexar/backups/postgres/omni_lab/latest_backup_id.txt)"
$PY - "$R" "$L/cleanup/inputs/ws_disk.json" <<'PY'
import json, sys
R, ws = sys.argv[1], sys.argv[2]
def parse(path):
    lines = open(path).read().split("\n")
    out = {}
    for l in lines:
        f = l.split()
        if len(f) >= 6 and f[0] != "Filesystem":
            out[f[5]] = int(f[3])
    return out, [l for l in lines if l.strip()][-1]
b, bt = parse(f"{R}/df_before_waves.txt")
a, at = parse(f"{R}/df_after_waves.txt")
rows = [{"disk": f"AI server {m}", "before_free": b[m], "before_at": bt, "after_free": a[m], "after_at": at,
         "note": note} for m, note in (("/", "W2-HOME"), ("/mnt/lexar", "W3-LEXAR deletions minus recipe archive and other jobs' writes"),
                                       ("/mnt/ai_data", "W1-NAS (shared CIFS share; other writers included)"))]
json.dump(rows + json.load(open(ws)), open(f"{R}/disk_totals.json", "w"), indent=1)
PY
{
  echo "## Status: final (written by nas_finalize.sh at $(date -u +%FT%TZ), after the NAS workers exited)"
  echo ""
  echo "- All waves have ended. W0, W2-HOME, W3-LEXAR and W4-WS finished on 2026-10-21 by 19:56Z; W1-NAS ended at the time above."
  echo "- Independent verification (verify_deleted.py over every server model_deletion row): $(tail -1 "$R/verify_final.txt")"
  echo "- Recipe mirror: $(grep -h 'recipe mirror' "$R/nas_finalize.log" | tail -1 | cut -c22-)"
  echo "- Post-wave omni_lab dump: /mnt/lexar/backups/postgres/omni_lab/dumps/$(cat /mnt/lexar/backups/postgres/omni_lab/latest_backup_id.txt).dump"
} > "$R/status_final.md"
$PY cleanup/report_build.py --plan cleanup/inputs/cleanup_plan.json --logs "$R" --ws-log cleanup/inputs/ws_cleanup_log.jsonl \
   --extra cleanup/inputs/report_extra.md --disk "$R/disk_totals.json" --status "$R/status_final.md" \
   --out "$R/CLEANUP_REPORT_2026-10-21.md"
say "report: $R/CLEANUP_REPORT_2026-10-21.md"
