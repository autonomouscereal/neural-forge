set -euo pipefail

umask 077
root=/home/cereal/model-surgery-lab/deployments/neural-forge-c8cfa511
artifact_root=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_moving_gauge_v5_20260804
manifest="$root/experiments/manifests/lfm2-exact4x-moving-gauge-v5-r12-launch.json"
key=/home/cereal/model-surgery-lab/keys/neural-forge-balanced-basis-phase-v1.key
receipt="$artifact_root/candidate-packs/r12-v5-launch-receipt-v2.json"
index="$artifact_root/materialization-index-c8cfa511.jsonl"
temporary="$artifact_root/.materialization-index-c8cfa511.$$.tmp"

test -f "$manifest"
test -f "$key"
test -s "$receipt"
pid=$(tr -d '[:space:]' < "$artifact_root/v5-r12-c8cfa511-rtx8000.pid")
if kill -0 "$pid" 2>/dev/null; then
    printf 'v5-still-live=%s\n' "$pid" >&2
    exit 75
fi
if [ -e "$index" ]; then
    printf 'refusing-existing-index=%s\n' "$index" >&2
    exit 42
fi
trap 'rm -f "$temporary"' EXIT
cd "$root"
export PYTHONPATH="$root/src:$root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=''

for chart in E0 E0R E0A; do
    for arm in joint sequential_conditioned; do
        /home/cereal/model-surgery-lab/venv/bin/python -u -m \
            neural_forge.lfm2_exact4x_moving_gauge_v5_candidate_materializer \
            --launch-manifest-json "$manifest" \
            --authority-key-file "$key" \
            --chart-mode "$chart" \
            --target-arm "$arm" | tee -a "$temporary"
    done
done
test "$(wc -l < "$temporary")" -eq 6
ln "$temporary" "$index"
rm -f "$temporary"
trap - EXIT
printf 'materialization_index=%s\n' "$index"
