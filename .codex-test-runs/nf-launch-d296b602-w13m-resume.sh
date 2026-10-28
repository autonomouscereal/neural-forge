#!/usr/bin/env bash
set -euo pipefail

LAB=/home/cereal/model-surgery-lab
export SOURCE_COMMIT=d296b602eec723461d476043617641ef7a2c64c4
export RUNTIME=$LAB/deployments/neural-forge-d296b602
export RUN=$LAB/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0027-exact4x-h816-mgv3-r12-w13m-resume-d296b602
# Capsule bindings include the child name, so exact continuation intentionally
# retains W13L's name even though the process/run root advances to W13M.
export CHILD_NAME=lfm2-integrated-carrier-exact4x-h816-mgv3-r12-w13l-cold
export RESUME_FROM_CAPSULE=$LAB/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0026-exact4x-h816-mgv3-r12-w13l-cold/restart-capsules

exec "$(dirname "$0")/nf-launch-r12-round-refold.sh" "$@"
