set -euo pipefail

artifact_root=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_moving_gauge_v5_20260804
log="$artifact_root/v5-r12-c8cfa511-rtx8000.log"
pidfile="$artifact_root/v5-r12-c8cfa511-rtx8000.pid"
receipt="$artifact_root/candidate-packs/r12-v5-launch-receipt-v2.json"

if [ -s "$pidfile" ]; then
    pid=$(tr -d '[:space:]' < "$pidfile")
    ps -p "$pid" -o pid=,etimes=,stat=,%cpu=,%mem=,rss=,cmd= || true
    if [ -r "/proc/$pid/io" ]; then
        grep -E '^(rchar|wchar|read_bytes|write_bytes):' "/proc/$pid/io" || true
    fi
else
    printf 'pidfile=ABSENT\n'
fi
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu \
    --format=csv,noheader | grep 'GPU-f5a09535-fc85-a027-50f7-734b344747dd' || true
if [ -f "$receipt" ]; then
    sha256sum "$receipt"
else
    printf 'receipt=ABSENT\n'
fi
printf 'checkpoint_count=%s\n' "$(
    find "$artifact_root/checkpoints" -type f 2>/dev/null | wc -l
)"
find "$artifact_root/checkpoints" -type f \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 12 || true
tail -n 20 "$log" 2>/dev/null || true
