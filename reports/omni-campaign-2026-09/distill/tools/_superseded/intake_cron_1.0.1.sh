#!/usr/bin/env bash
# intake_cron.sh -- the unattended Chimera corpus intake pass (intake.py run --cpu-only), started by cron every 30 min.
# Written 2026-10-15 by Claude.  CPU only: CUDA_VISIBLE_DEVICES="" and nice 15 / ionice idle.  GPU transcription is
# never started from here (intake.py transcribe, P40 queue, plan 7.4); this pass only lists pending items.
#
# crontab (installed between "# BEGIN chimera-intake" / "# END chimera-intake"):
#   */30 * * * * /usr/bin/flock -n $OC/distill/queues/.intake.lock /usr/bin/timeout 1500 /bin/bash \
#       $OC/distill/tools/intake_cron.sh >> $OC/distill/queues/logs/intake.log 2>&1
# Credentials: /home/cereal/.config/chimera-intake/env (0600, CHIMERA_CORPUS_PG_PASSWORD = vault key chimera_corpus_pg).
set -euo pipefail
INTAKE_CRON_VERSION="1.0.1"   # 1.0.0 (2026-10-15): first version
                              # 1.0.1 (2026-10-15): intake.py 1.1.0; the pass budget is passed explicitly (--budget-s 1200,
                              #        under the 1500 s cron timeout) so a large drop resumes next pass instead of being killed
OC=/home/cereal/omni-campaign
ENV_FILE=/home/cereal/.config/chimera-intake/env
PY=/home/cereal/.venvs/chimera-intake/bin/python   # root fs: the exFAT Lexar cannot hold a venv (no symlinks)
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

for mount in /mnt/ai_data /mnt/lexar; do
  if ! mountpoint -q "$mount"; then echo "$(utc) intake_cron $INTAKE_CRON_VERSION: $mount not mounted; skipped"; exit 3; fi
done
if [ ! -r "$ENV_FILE" ]; then echo "$(utc) intake_cron: $ENV_FILE missing"; exit 2; fi
if [ "$(stat -c %a "$ENV_FILE")" != "600" ]; then echo "$(utc) intake_cron: $ENV_FILE must be mode 600"; exit 2; fi
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a
export CUDA_VISIBLE_DEVICES=""
echo "$(utc) intake_cron $INTAKE_CRON_VERSION start"
exec nice -n 15 ionice -c 3 "$PY" "$OC/distill/tools/intake.py" run --cpu-only --budget-s 1200
