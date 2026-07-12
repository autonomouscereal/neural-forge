#!/usr/bin/env bash
set -euo pipefail

cd /home/cereal/model-surgery-lab
output=/tmp/neural-forge-source-copies-20260712.tar.gz
test ! -e "$output"

roots=(
  neural-forge-behavior-vo-tuner-20260711
  neural-forge-cached-attention-smoke-20260711
  neural-forge-pivot
  neural-forge-residual-attn-20260711
  neural-forge-residual-attn-functional-20260711
  neural-forge-role-scale-functional-20260711
  neural-forge-src
  neural-forge-src-981feb9
  neural-forge-unified-20260710
)

find "${roots[@]}" -type f \
  \( \
    -path '*/docs/*' -o \
    -path '*/experiments/*' -o \
    -path '*/research/*' -o \
    -path '*/scripts/*' -o \
    -path '*/src/*' -o \
    -path '*/tests/*' -o \
    -name '.gitattributes' -o \
    -name '.gitignore' -o \
    -name 'LICENSE' -o \
    -name 'README.md' -o \
    -name 'pyproject.toml' \
  \) \
  ! -path '*/.git/*' \
  ! -path '*/.mypy_cache/*' \
  ! -path '*/.pytest_cache/*' \
  ! -path '*/.ruff_cache/*' \
  ! -path '*/.venv/*' \
  ! -path '*/__pycache__/*' \
  ! -path '*/runs/*' \
  ! -path '*/venv/*' \
  ! -name '*.pyc' \
  ! -name '*.pyo' \
  ! -name '*.zip' \
  -print0 \
  | sort -z \
  | tar --null --files-from=- --create --gzip --file="$output"

sha256sum "$output"
du -h "$output"
