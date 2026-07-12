#!/usr/bin/env bash
set -euo pipefail

for directory in /home/cereal/model-surgery-lab/neural-forge-*; do
  test -d "$directory" || continue
  if git -C "$directory" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    commit=$(git -C "$directory" rev-parse HEAD)
    branch=$(git -C "$directory" branch --show-current)
    dirty=$(git -C "$directory" status --porcelain | wc -l)
  else
    commit=NO_GIT
    branch=-
    dirty=-
  fi
  files=$(find "$directory" -type f | wc -l)
  bytes=$(du -sb "$directory" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$directory" "$branch" "$commit" "$dirty" "$files" "$bytes"
done
