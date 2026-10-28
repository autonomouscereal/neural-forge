set -euo pipefail

artifact_root=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_moving_gauge_v5_20260804
log="$artifact_root/v5-r12-c8cfa511-rtx8000.log"
pidfile="$artifact_root/v5-r12-c8cfa511-rtx8000.pid"
receipt="$artifact_root/candidate-packs/r12-v5-launch-receipt-v2.json"

printf 'process\n'
if [ -s "$pidfile" ]; then
    pid=$(tr -d '[:space:]' < "$pidfile")
    ps -p "$pid" -o pid=,etimes=,stat=,%cpu=,%mem=,rss=,cmd= || true
    if [ -r "/proc/$pid/io" ]; then
        cat "/proc/$pid/io"
    fi
    if [ -d "/proc/$pid/fd" ]; then
        for fd in "/proc/$pid/fd"/*; do
            readlink "$fd" 2>/dev/null || true
        done | grep -E 'model-surgery-lab|neural_forge' | tail -n 20 || true
    fi
else
    printf 'pidfile=ABSENT\n'
fi
printf 'gpu\n'
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader | grep 'GPU-f5a09535-fc85-a027-50f7-734b344747dd' || true
printf 'receipt\n'
if [ -f "$receipt" ]; then
    sha256sum "$receipt"
else
    printf 'ABSENT\n'
fi
printf 'artifact_usage\n'
du -sh "$artifact_root" 2>/dev/null || true
printf 'recent_files\n'
find "$artifact_root" -maxdepth 8 -type f \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 40 || true
printf 'log_tail\n'
if [ -f "$log" ]; then
    tail -n 100 "$log"
else
    printf 'ABSENT\n'
fi
