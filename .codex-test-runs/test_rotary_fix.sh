set -euo pipefail

ROOT=/home/cereal/model-surgery-lab/runtimes/neural-forge-07fafdb0
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
/home/cereal/model-surgery-lab/venv/bin/python -m pytest -q \
  tests/test_hf_lfm2_dynamic_blank_materializer.py::test_heterogeneous_cpu_rotary_reauthenticates_after_cuda_move \
  tests/test_hf_lfm2_dynamic_blank_materializer.py::test_materializes_fully_heterogeneous_geometry_and_native_cache
