set -euo pipefail
runtime=/home/cereal/model-surgery-lab/runtimes/neural-forge-9100d17b
python=/home/cereal/model-surgery-lab/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$runtime:$runtime/src"
cd "$runtime"
"$python" -m pytest -q -p no:cacheprovider \
  tests/test_review_lfm2_semantic_frontier.py \
  tests/test_run_lfm2_exact4x_causal_head_frontier.py
"$python" experiments/run_lfm2_exact4x_causal_head_frontier.py --help >/dev/null
echo causal_head_frontier_help=passed
"$python" experiments/review_lfm2_semantic_frontier.py --help >/dev/null
echo semantic_frontier_reviewer_help=passed
echo runtime_validation=passed
