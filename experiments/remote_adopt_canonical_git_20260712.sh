#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cereal/model-surgery-lab/neural-forge-src"
EXPECTED_ROOT="/home/cereal/model-surgery-lab/neural-forge-src"
ORIGIN="https://github.com/autonomouscereal/neural-forge.git"
BRANCH="codex/lrc-blank-compression"
MODE="${1:---preflight}"
BUNDLE="${2:-}"

resolved="$(realpath "$ROOT")"
if [[ "$resolved" != "$EXPECTED_ROOT" ]]; then
  printf 'refusing unexpected root: %s\n' "$resolved" >&2
  exit 2
fi

if [[ ! -d "$ROOT/src" || ! -d "$ROOT/experiments" ]]; then
  printf 'refusing incomplete source root: %s\n' "$ROOT" >&2
  exit 3
fi

printf 'resolved=%s\n' "$resolved"
printf 'root_bytes='
du -sb "$ROOT" | awk '{print $1}'
printf 'source_file_count='
find "$ROOT/src" "$ROOT/experiments" "$ROOT/tests" "$ROOT/docs" \
  -type f 2>/dev/null | wc -l
printf 'old_root_processes='
ps -eo pid=,args= \
  | grep -E 'neural-forge-(pivot|unified|residual|role-scale|behavior|cached|src-981)' \
  | grep -v grep \
  | wc -l \
  || true

if [[ "$MODE" == "--preflight" ]]; then
  if [[ -d "$ROOT/.git" ]]; then
    printf 'git_present=yes\n'
  else
    printf 'git_present=no\n'
  fi
  exit 0
fi

if [[ "$MODE" != "--adopt" && "$MODE" != "--adopt-bundle" ]]; then
  printf 'usage: %s [--preflight|--adopt|--adopt-bundle BUNDLE]\n' "$0" >&2
  exit 4
fi

cd "$ROOT"

if [[ ! -d .git ]]; then
  git init
fi

if [[ "$MODE" == "--adopt-bundle" ]]; then
  if [[ -z "$BUNDLE" || ! -f "$BUNDLE" ]]; then
    printf 'refusing missing bundle: %s\n' "$BUNDLE" >&2
    exit 6
  fi
  git bundle verify "$BUNDLE"
fi

if git remote get-url origin >/dev/null 2>&1; then
  actual_origin="$(git remote get-url origin)"
  if [[ "$actual_origin" != "$ORIGIN" ]]; then
    printf 'refusing unexpected origin: %s\n' "$actual_origin" >&2
    exit 5
  fi
else
  git remote add origin "$ORIGIN"
fi

if [[ "$MODE" == "--adopt-bundle" ]]; then
  git fetch "$BUNDLE" "refs/heads/$BRANCH"
  source_commit="$(git rev-parse FETCH_HEAD)"
  git fetch "$BUNDLE" 'refs/tags/*:refs/tags/*'
else
  git fetch origin "$BRANCH"
  source_commit="$(git rev-parse FETCH_HEAD)"
  git fetch origin --tags
fi
git update-ref "refs/heads/$BRANCH" "$source_commit"
git symbolic-ref HEAD "refs/heads/$BRANCH"
git read-tree HEAD
git checkout-index --all --force
git config "branch.$BRANCH.remote" origin
git config "branch.$BRANCH.merge" "refs/heads/$BRANCH"

printf 'head='
git rev-parse HEAD
printf 'tracked_files='
git ls-files | wc -l
printf 'tracked_diff_count='
git status --short --untracked-files=no | wc -l
printf 'untracked_source_paths_begin\n'
git status --short --untracked-files=all -- src experiments tests docs \
  | head -200 \
  || true
printf 'untracked_source_paths_end\n'
