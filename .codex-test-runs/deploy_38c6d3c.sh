set -euo pipefail
SHA=38c6d3c31e93b81793592ff38e44e760005cf11a
ARCHIVE=/tmp/neural-forge-38c6d3c.tar
DEST=/home/cereal/model-surgery-lab/runtime/neural-forge-$SHA
test -f "$ARCHIVE"
test ! -e "$DEST"
mkdir -p "$DEST"
tar -xf "$ARCHIVE" -C "$DEST"
test -f "$DEST/experiments/run_lfm2_semantic_fold_to_review.py"
test -f "$DEST/src/neural_forge/hf_lfm2_semantic_postfold_frontier.py"
printf '%s\n' "$DEST"
