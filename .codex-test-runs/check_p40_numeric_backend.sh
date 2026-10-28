set -eu
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=/home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2/src
/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python - <<'PY'
from neural_forge.hf_lfm2_sparse_semantic_codebook import (
    build_lfm2_residual_omp_numeric_backend_receipt,
)

receipt = build_lfm2_residual_omp_numeric_backend_receipt("cuda:0", "float32")
print(receipt["numeric_backend_receipt_sha256"])
print(receipt["selected_accelerator"])
PY
