#!/bin/bash
# commit_scripts_docs.sh -- commit the Chimera Omni campaign's SCRIPTS and DOCUMENTATION (never weights, media, data or
# logs) to the branch claude/omni-campaign, without touching the shared checkout's HEAD, index or working tree.
# Cyril 2026-10-18: "[Cyril quote withheld]"  The checkout is shared with Codex (it sits on a codex/* branch with uncommitted work), so this
# builds the commit in a private index (GIT_INDEX_FILE) on top of the branch tip (first run: main) and moves only
# refs/heads/claude/omni-campaign.
#   bash reports/omni-campaign-2026-09/commit_scripts_docs.sh "message"        (run from the repo root)
# Kept: .py .sh .cpp .md .jinja, and .json/.txt/.csv/.sha256 under 1 MB.  Skipped: .superseded-*/.bak-*/__pycache__ dirs,
# version backups are kept (receipts name their sha256).  A secret scan refuses the commit on any hit.
set -euo pipefail
MSG=${1:?commit message}
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
SUB=reports/omni-campaign-2026-09
BRANCH=refs/heads/claude/omni-campaign
LIST=$(mktemp)
find "$SUB" -type d \( -name '.superseded-*' -o -name '.bak-*' -o -name '__pycache__' -o -name '.git' \) -prune -o -type f \
  \( -name '*.py' -o -name '*.sh' -o -name '*.cpp' -o -name '*.md' -o -name '*.jinja' \
     -o \( \( -name '*.json' -o -name '*.txt' -o -name '*.csv' -o -name '*.sha256' \) -size -1024k \) \) -print \
  | sort > "$LIST"
echo "selected $(wc -l < "$LIST") files ($(xargs -d '\n' cat < "$LIST" | wc -c) bytes)"
# secret scan: provider tokens, private keys, inline passwords
if xargs -d '\n' grep -l -E 'hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|glpat-[A-Za-z0-9_-]{15,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(password|passwd|api_key|secret)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"'<$ ]{6,}' < "$LIST" > "$LIST.hits" 2> /dev/null; then
  echo "REFUSED: possible secrets in:"; cat "$LIST.hits"; exit 2
fi
OLD=$(git rev-parse -q --verify "$BRANCH" || true)            # empty: the branch must not exist yet
PARENT=${OLD:-$(git rev-parse main)}
export GIT_INDEX_FILE=$(mktemp -u)
trap 'rm -f "$GIT_INDEX_FILE" "$LIST" "$LIST.hits"' EXIT
git read-tree "$PARENT"
git rm -r -q --cached --ignore-unmatch -- "$SUB" > /dev/null   # the subtree is rebuilt from the selection each time
xargs -d '\n' git add -f -- < "$LIST"
TREE=$(git write-tree)
if [ "$TREE" = "$(git rev-parse "$PARENT^{tree}")" ]; then echo "nothing changed"; exit 0; fi
COMMIT=$(printf '%s\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n' "$MSG" | git commit-tree "$TREE" -p "$PARENT")
git update-ref "$BRANCH" "$COMMIT" "$OLD"   # refuses if the branch moved since OLD was read
echo "committed $COMMIT on ${BRANCH#refs/heads/} (parent ${PARENT:0:8}): $(git diff --stat "$PARENT" "$COMMIT" | tail -1)"
