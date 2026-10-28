#!/usr/bin/env bash
set -euo pipefail

BASE=/home/cereal/model-surgery-lab/deployments/neural-forge-b65c81f5
TRANSFER=/home/cereal/nf-w13m
BUNDLE=$TRANSFER/neural-forge-d296b602.bundle
DEST=/home/cereal/model-surgery-lab/deployments/neural-forge-d296b602
EXPECTED=d296b602eec723461d476043617641ef7a2c64c4
TRANSFER_SHA256=db9361f24f883735976ccf13523049297346a8b836c02b2cf4a179d68a6d09da
DEPLOY_REF=refs/codex-deploy/d296b602

test -e "$BASE/.git"
test ! -e "$DEST"
printf '%s  %s\n' "$TRANSFER_SHA256" "$BUNDLE" | sha256sum --check --strict
git -C "$BASE" bundle verify "$BUNDLE"
git -C "$BASE" fetch "$BUNDLE" "HEAD:$DEPLOY_REF"
test "$(git -C "$BASE" rev-parse "$DEPLOY_REF^{commit}")" = "$EXPECTED"
git -C "$BASE" worktree add --detach "$DEST" "$EXPECTED"
test "$(git -C "$DEST" rev-parse HEAD)" = "$EXPECTED"
test -z "$(git -C "$DEST" status --porcelain)"
git -C "$BASE" update-ref -d "$DEPLOY_REF" "$EXPECTED"

printf 'deployment=%s\ncommit=%s\n' "$DEST" "$EXPECTED"
