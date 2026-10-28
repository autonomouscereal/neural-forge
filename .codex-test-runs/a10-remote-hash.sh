set -euo pipefail

runtime=/home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2
for relative in \
  experiments/run_lfm2_functional_output_refit.py \
  experiments/run_lfm2_sparse_semantic_codebook.py \
  src/neural_forge/hf_lfm2_functional_output_refit.py \
  src/neural_forge/hf_lfm2_semantic_step_token_fold.py
do
  path="$runtime/$relative"
  if [ -f "$path" ]; then
    sha256sum "$path"
  else
    printf 'missing  %s\n' "$path"
  fi
done

python=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
PYTHONPATH="$runtime/src" "$python" -m py_compile \
  "$runtime/experiments/run_lfm2_functional_output_refit.py" \
  "$runtime/experiments/run_lfm2_sparse_semantic_codebook.py" \
  "$runtime/src/neural_forge/hf_lfm2_functional_output_refit.py"
PYTHONPATH="$runtime/src" "$python" \
  "$runtime/experiments/run_lfm2_functional_output_refit.py" --help >/dev/null
printf 'remote_import_compile_help=ready\n'
