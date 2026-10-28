set -euo pipefail

umask 077
artifact_root=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_moving_gauge_v5_20260804
v5_pidfile="$artifact_root/v5-r12-c8cfa511-rtx8000.pid"
materialize=/home/cereal/model-surgery-lab/upload_staging/materialize-v5-c8cfa511.sh
log="$artifact_root/materialize-v5-c8cfa511.log"

test -s "$v5_pidfile"
test -f "$materialize"
v5_pid=$(tr -d '[:space:]' < "$v5_pidfile")
while kill -0 "$v5_pid" 2>/dev/null; do
    sleep 30
done
bash "$materialize" >"$log" 2>&1
