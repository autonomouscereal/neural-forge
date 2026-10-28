set -euo pipefail

root=/home/cereal/model-surgery-lab/neural-forge-highcap-eb0a5460
py=/home/cereal/model-surgery-lab/venv/bin/python
cd "$root"

sha256sum -c - <<'EOF'
6b480f6eb7be5cb3c5e2697f223b87c05f376f3b5dca6c6ade1614db0cc723cf  experiments/rerun_lfm2_sparse_semantic_capped_prompts.py
83741472070652e0a65bea59052c195fb0ebef205ccf2a58cc172f48e29a1734  experiments/lfm2_a9_a11_high_cap_reuse_manifest_20260730.json
5fdde47cad42eae574402fef0830e546364d2bba6e1229d100a72ce87666eadb  experiments/launch_lfm2_a9_a11_high_cap_reuse_20260730.sh
ddd23f1ddc41ea66cc258bc9a5ff100ed558d5fe308c549609e444e4b09831f8  experiments/launch_detached.py
0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b  research/lfm2_semantic_whole_response_prompts_v1.json
b9efc943746eca62b986ec2d9b81b1638544a6561f286f88f1bb7584e0b76459  src/neural_forge/dynamic_circuit_ir.py
1b8db968dd29340b0df5213fc77aa6b0227508100b7f8dddc41240180eb2fc24  src/neural_forge/hf_lfm2_direct_local.py
ef757359d9376fd66fb1c7578ab1409c93fa8d2c64c54f07cee49bfb87fa7927  src/neural_forge/hf_lfm2_dynamic_blank_materializer.py
a17f37eeef3a2ac6024e998129fafbede9a41d6f5782492f9256ff0e18eac93b  src/neural_forge/hf_lfm2_dynamic_physical_layout.py
9a16cd7aabb9cfb5ad9f029d53b9c2f6f25e9513ed03cabc89d2248f4ed109da  src/neural_forge/hf_lfm2_functional_output_refit.py
94a299c36c655726897188e781b39a438707e92c052a6dc6ad7ba986d5f2c028  src/neural_forge/hf_lfm2_semantic_step_model_runtime.py
c0decec4cd286070181962f926fcfe815e53489e81213f1c08590b3016494e9e  src/neural_forge/hf_lfm2_semantic_step_token_fold.py
e3cef33787d86368c07de226a92264ec3b248b73a8087618cfc5388289600d80  src/neural_forge/hf_lfm2_sparse_semantic_codebook.py
5a5f05db23504b8e2f56ff38d058f616305105ea5b775c1d35ef95104dc97cfb  src/neural_forge/hf_token_align.py
865f087b7b8c213f6797087044a4163ef9f10335daa07680767b7e2bad51bd22  src/neural_forge/hf_tokenizer_bridge.py
07205ef26a0db1f341682c1428aeeeb46805d5a159cab7213b46051aa56e8ecc  src/neural_forge/hf_tokenizer_prune.py
251e5d0684f3dbeeab2aa2f70a3ef25b03ff49ee7f6048477f3d59f8ad941738  src/neural_forge/hf_transport_math.py
1b271c48fa45fac4b291c64a89503532eab36d62cd1f9d74acafd6aea3d54356  src/neural_forge/lfm2_decoded_review.py
c1f324ae5d5608ce29eb411a59193445069af5e6c2a309b9f9b10b309298e113  src/neural_forge/numeric_execution_policy.py
EOF

bash -n experiments/launch_lfm2_a9_a11_high_cap_reuse_20260730.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/src:$root"
"$py" experiments/rerun_lfm2_sparse_semantic_capped_prompts.py --help
"$py" - <<'PY'
from pathlib import Path
from experiments import rerun_lfm2_sparse_semantic_capped_prompts as runner

manifest = Path("experiments/lfm2_a9_a11_high_cap_reuse_manifest_20260730.json")
plan = runner.load_reuse_plan(manifest)
print("plan_sha256=" + plan["plan_sha256"])
for lane in plan["lanes"]:
    dossier, rows = runner.authenticate_lane_source(lane)
    print(
        "lane=" + lane["id"]
        + " source=" + dossier["dossier_sha256"]
        + " capped=" + str(len(rows))
    )
PY

output=/home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_highcap_20260730
if [ -e "$output" ]; then
  echo "OUTPUT_ALREADY_EXISTS"
  find "$output" -maxdepth 3 -type f -printf '%p %s\n' | head -50
else
  echo "OUTPUT_ABSENT=$output"
fi

echo GPU4
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed -n '5p'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits | grep 'GPU-2ce6a95c-7b1f-644a-3808-d781732a0582' || true
