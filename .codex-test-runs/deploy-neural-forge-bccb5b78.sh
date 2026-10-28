#!/usr/bin/env bash
set -euo pipefail

LAB=/home/cereal/model-surgery-lab
ARCHIVE=$LAB/uploads/neural-forge-runtime-bccb5b78.tar.gz
DEPLOY=$LAB/deployments/neural-forge-bccb5b78
EXPECTED=9489aea3a057f557a6c8a511fcd2445b4ef73da3d11c76075381d294622b3df4
COMMIT=bccb5b7899d8f01f449b205fdf5a316ab9e5e7b8

test -f "$ARCHIVE"
test "$(sha256sum "$ARCHIVE" | cut -d' ' -f1)" = "$EXPECTED"
test ! -e "$DEPLOY"
mkdir "$DEPLOY"
tar -xzf "$ARCHIVE" -C "$DEPLOY"
printf '%s\n' "$COMMIT" > "$DEPLOY/SOURCE_COMMIT"
test "$(cat "$DEPLOY/SOURCE_COMMIT")" = "$COMMIT"
test -f "$DEPLOY/src/neural_forge/hf_lfm2_cached_global_topology_frontier.py"
test -f "$DEPLOY/src/neural_forge/hf_lfm2_global_topology_phase_compatibility.py"
test -f "$DEPLOY/experiments/run_lfm2_dynamic_catalog.py"
printf 'deployment-ok path=%s commit=%s\n' "$DEPLOY" "$COMMIT"
