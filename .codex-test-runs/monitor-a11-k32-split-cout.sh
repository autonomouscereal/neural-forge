set -euo pipefail

run=/home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_bridge_20260730/a11-k32-split-cout-lambda0001
pid=$(cat "$run/job.pid" 2>/dev/null || printf missing)
status=$(cat "$run/job.status" 2>/dev/null || printf running)
printf 'pid=%s status=%s\n' "$pid" "$status"
if [ "$pid" != missing ] && kill -0 "$pid" 2>/dev/null; then
  ps --ppid "$pid" -o pid=,etime=,%cpu=,%mem=,stat=,cmd= 2>/dev/null || true
fi
printf 'milestones:\n'
for file in \
  functional-output-refit-authentication.json \
  hidden-diagnostic.json \
  whole-response-dossier.json \
  run-receipt.json
do
  if [ -f "$run/$file" ]; then
    printf '%s %s bytes\n' "$file" "$(stat -c %s "$run/$file")"
  fi
done
printf 'prompt-completions=%s\n' "$(find "$run" -type f -name complete.json 2>/dev/null | wc -l)"
printf 'logged-whole-response-phase-completions=%s\n' "$(grep -c '"phase":"lfm2-semantic-step-whole-response".*"stage":"phase-complete"' "$run/job.log" 2>/dev/null || true)"
printf 'latest-log:\n'
tail -n 24 "$run/job.log" 2>/dev/null || true
printf 'gpu-processes:\n'
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || true
