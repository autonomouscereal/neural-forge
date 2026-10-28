set -euo pipefail
LAB=/home/cereal/model-surgery-lab
ARCHIVE=$LAB/uploads/neural-forge-runtime-9b03caf7.tar.gz
DEPLOY=$LAB/deployments/neural-forge-9b03caf7
EXPECTED=86ab768900ffab3c041dbdb428850e221f82b44dbb1f04da89c53a8ab2d3fdf3
COMMIT=9b03caf74213edfdcac68fd970e3aee15752bed4
test -f "$ARCHIVE"
test "$(sha256sum "$ARCHIVE" | cut -d' ' -f1)" = "$EXPECTED"
test ! -e "$DEPLOY"
mkdir "$DEPLOY"
tar -xzf "$ARCHIVE" -C "$DEPLOY"
printf '%s\n' "$COMMIT" > "$DEPLOY/SOURCE_COMMIT"
test "$(cat "$DEPLOY/SOURCE_COMMIT")" = "$COMMIT"
test -f "$DEPLOY/src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py"
test -f "$DEPLOY/experiments/run_lfm2_dynamic_catalog.py"
printf 'deployment-ok path=%s commit=%s\n' "$DEPLOY" "$COMMIT"
