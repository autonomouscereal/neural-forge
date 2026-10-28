#!/usr/bin/env bash
set -euo pipefail

BASE=/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a
TRANSFER=/home/cereal/nf-w13l
BUNDLE=$TRANSFER/neural-forge-b65c81f5.bundle
DEST=/home/cereal/model-surgery-lab/deployments/neural-forge-b65c81f5
EXPECTED=b65c81f5d086830247b0a418a78d54fb12975ca6
TRANSFER_SHA256=57108a5f7c4a14ae180accdabfcbc8502377284992c48e7d718ddc257da233f6
DEPLOY_REF=refs/codex-deploy/b65c81f5

test -d "$BASE/.git"
test ! -e "$DEST"
printf '%s  %s\n' "$TRANSFER_SHA256" "$BUNDLE" | sha256sum --check --strict
git -C "$BASE" bundle verify "$BUNDLE"
git -C "$BASE" fetch "$BUNDLE" "refs/bundles/b65c81f5:$DEPLOY_REF"
test "$(git -C "$BASE" rev-parse "$DEPLOY_REF^{commit}")" = "$EXPECTED"
git -C "$BASE" worktree add --detach "$DEST" "$EXPECTED"
test "$(git -C "$DEST" rev-parse HEAD)" = "$EXPECTED"
test -z "$(git -C "$DEST" status --porcelain)"
git -C "$BASE" update-ref -d "$DEPLOY_REF" "$EXPECTED"

printf 'deployment=%s\ncommit=%s\n' "$DEST" "$EXPECTED"
