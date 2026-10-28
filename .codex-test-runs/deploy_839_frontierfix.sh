set -euo pipefail
ARCHIVE=/tmp/neural-forge-839758e.tar
PATCHED_RUNNER=/tmp/run_lfm2_semantic_fold_to_review.frontierfix.py
DEST=/home/cereal/model-surgery-lab/runtime/neural-forge-839758e3b13ba5bb0eb72ad5f4e306a37fafc6a0-frontierfix-38c6d3c
test -f "$ARCHIVE"
test -f "$PATCHED_RUNNER"
test ! -e "$DEST"
mkdir -p "$DEST"
tar -xf "$ARCHIVE" -C "$DEST"
cp "$PATCHED_RUNNER" "$DEST/experiments/run_lfm2_semantic_fold_to_review.py"
test -f "$DEST/src/neural_forge/hf_lfm2_semantic_preterminal_capsule.py"
grep -F 'tuple(manifest_eligible or ()) != candidate_ids' \
  "$DEST/experiments/run_lfm2_semantic_fold_to_review.py"
printf '%s\n' "$DEST"
