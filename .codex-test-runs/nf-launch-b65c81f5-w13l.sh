#!/usr/bin/env bash
set -euo pipefail

LAB=/home/cereal/model-surgery-lab
export SOURCE_COMMIT=b65c81f5d086830247b0a418a78d54fb12975ca6
export RUNTIME=$LAB/deployments/neural-forge-b65c81f5
export RUN=$LAB/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0026-exact4x-h816-mgv3-r12-w13l-cold
export CHILD_NAME=lfm2-integrated-carrier-exact4x-h816-mgv3-r12-w13l-cold

exec "$(dirname "$0")/nf-launch-r12-round-refold.sh" "$@"
