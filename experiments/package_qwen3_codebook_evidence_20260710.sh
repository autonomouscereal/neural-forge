#!/usr/bin/env bash
set -euo pipefail

base=/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710
archive="$base/qwen3_codebook_completed_evidence_20260710.tar.gz"
tags=(
  qwen3_4b_codebook_pn34_pa34_gpu6_r1_20260710
  qwen3_4b_codebook_pn35_pa35_gpu4_r1_20260710
  qwen3_4b_codebook_pn28_pa28_gpu0_r1_20260710
)

for tag in "${tags[@]}"; do
  test -f "$base/$tag/parent_candidate_comparison.json"
  test -f "$base/$tag/run_report.json"
done

rm -f "$archive"
tar -C "$base" -czf "$archive" "${tags[@]}"
sha256sum "$archive"
stat -c '%s bytes %y' "$archive"
