set -euo pipefail
archive=/home/cereal/model-surgery-lab/upload_staging/neural-forge-9100d17b.tar.gz
runtime=/home/cereal/model-surgery-lab/runtimes/neural-forge-9100d17b
expected=d4c128fe07f0a9af803108921a51fbb3db0ee015768f6dd8050a9514b2d9e54a
actual=$(sha256sum "$archive" | awk '{print $1}')
if [ "$actual" != "$expected" ]; then
  echo "archive sha mismatch: $actual" >&2
  exit 10
fi
if [ -e "$runtime" ]; then
  echo "runtime already exists: $runtime" >&2
  exit 11
fi
install -d "$runtime"
tar -xzf "$archive" -C "$runtime"
test -f "$runtime/experiments/run_lfm2_exact4x_causal_head_frontier.py"
test -f "$runtime/experiments/review_lfm2_semantic_frontier.py"
grep -q 'complete_candidate_bank_decoded_without_preselection' "$runtime/experiments/review_lfm2_semantic_frontier.py"
find "$runtime" -type d -exec chmod 0555 {} +
find "$runtime" -type f -exec chmod 0444 {} +
printf 'archive_sha256=%s\nruntime=%s\nruntime_mode=%s\nreviewer_mode=%s\n' \
  "$actual" \
  "$runtime" \
  "$(stat -c %a "$runtime")" \
  "$(stat -c %a "$runtime/experiments/review_lfm2_semantic_frontier.py")"
