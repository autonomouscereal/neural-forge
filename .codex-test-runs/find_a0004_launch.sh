set -euo pipefail

rg -n -g '*.sh' -g '*.txt' -g '*.md' -g '*.json' \
  'lfm2_dynamic_heterogeneous_20260729|neural-forge-07fafdb0-d04ae8f2|a0004-aggressive-h784-exact57420144' \
  /home/cereal/model-surgery-lab 2>/dev/null || true
