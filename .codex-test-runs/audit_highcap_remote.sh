set -euo pipefail
echo "HOST=$(hostname) USER=$(whoami)"
for root in \
  /home/cereal/model-surgery-lab/neural-forge \
  /home/cereal/model-surgery-lab/neural-forge-src \
  /home/cereal/model-surgery-lab
do
  if [ -d "$root" ]; then
    echo "ROOT $root"
    if [ -d "$root/.git" ]; then
      git -C "$root" rev-parse --short HEAD || true
      git -C "$root" status --short | head -40 || true
    fi
  fi
done
echo "GPUS"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
echo "GPU_PROCESSES"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
echo "PYTHONS"
for py in \
  /home/cereal/model-surgery-lab/venv/bin/python \
  /home/cereal/model-surgery-lab/venv-torch26/bin/python \
  /usr/bin/python3
do
  if [ -x "$py" ]; then
    echo "$py"
    "$py" -c 'import sys; print(sys.version.split()[0]); import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())' 2>&1 || true
  fi
done
echo "BOUND_INPUTS"
for path in \
  /home/cereal/model-surgery-lab/artifacts/lfm2_bridge_only_control_20260723/89cd45b0/same-body-v4096-p40-gpu1-a0001/checkpoint \
  /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  /home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/lfm2-semantic-step-global-raw-span/3e780d07ac1034ee77756f973b483637cee846e17e6d7191d6b20c9533360167/payload/span-evidence.json \
  /home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/lfm2-sparse-semantic-codebook/5b5557b8da7aa125ada8cac305005de7b9e157933c3d4878f6be133fb5c51cc7/payload/artifact \
  /home/cereal/model-surgery-lab/artifacts/neural_forge_functional_output_refit_20260730/a10-k32-cout-lambda0001/math-cache/75cf54bc04438952a033ea494a6a269c1abd414cdb62a4b87c9bc77b73c57284/output-coefficients.safetensors \
  /home/cereal/model-surgery-lab/artifacts/neural_forge_functional_output_refit_20260730/a10-k32-cout-lambda0001/math-cache/75cf54bc04438952a033ea494a6a269c1abd414cdb62a4b87c9bc77b73c57284/result.json \
  /home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_bridge_20260730/a9-k32-residual-omp/whole-response-dossier.json \
  /home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_bridge_20260730/a11-k32-split-cout-lambda0001/whole-response-dossier.json
do
  if [ -e "$path" ]; then
    du -sh "$path"
  else
    echo "MISSING $path"
  fi
done
echo "OUTPUT_ROOT"
out=/home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_highcap_20260730
if [ -e "$out" ]; then
  find "$out" -maxdepth 3 -type f -printf '%p %s\n' | head -80
else
  echo "ABSENT $out"
fi
