set -euo pipefail

umask 077
root=/home/cereal/model-surgery-lab/deployments/neural-forge-f7bbf16d
artifact_root=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_moving_gauge_v5_20260804
manifest="$root/experiments/manifests/lfm2-exact4x-moving-gauge-v5-r12-launch.json"
key=/home/cereal/model-surgery-lab/keys/neural-forge-balanced-basis-phase-v1.key
receipt="$artifact_root/candidate-packs/r12-v5-launch-receipt.json"
log="$artifact_root/v5-r12-f7bbf16d-rtx8000.log"
pidfile="$artifact_root/v5-r12-f7bbf16d-rtx8000.pid"

test -f "$manifest"
test -f "$key"
if [ -e "$receipt" ]; then
    printf 'refusing-existing-receipt=%s\n' "$receipt" >&2
    exit 42
fi
for prior_pidfile in "$artifact_root"/v5-r12-*-rtx8000.pid; do
    [ -e "$prior_pidfile" ] || continue
    [ -s "$prior_pidfile" ] || continue
    prior_pid=$(tr -d '[:space:]' < "$prior_pidfile")
    if [ -n "$prior_pid" ] && kill -0 "$prior_pid" 2>/dev/null; then
        printf 'refusing-live-prior-pid=%s pidfile=%s\n' \
            "$prior_pid" "$prior_pidfile" >&2
        exit 43
    fi
done

install -d -m 700 "$artifact_root"
cd "$root"
export CUDA_VISIBLE_DEVICES=GPU-f5a09535-fc85-a027-50f7-734b344747dd
export PYTHONPATH="$root/src:$root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

setsid nohup /home/cereal/model-surgery-lab/venv/bin/python -u \
    experiments/run_lfm2_exact4x_moving_gauge_v5_r12_live.py \
    --launch-manifest-json "$manifest" \
    --authority-key-file "$key" \
    >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" > "$pidfile"
printf 'launched_pid=%s\nlog=%s\nreceipt=%s\n' "$pid" "$log" "$receipt"
