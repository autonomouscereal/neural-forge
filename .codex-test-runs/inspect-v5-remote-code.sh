set -euo pipefail
cd /home/cereal/model-surgery-lab
for path in \
    experiments/run_lfm2_exact4x_moving_gauge_v5.py \
    experiments/run_lfm2_exact4x_moving_gauge_v5_live.py \
    experiments/run_lfm2_exact4x_moving_gauge_v5_r12_live.py \
    src/neural_forge/hf_lfm2_moving_gauge_boundary_phase_cache.py \
    src/neural_forge/hf_lfm2_moving_gauge_feature_phase_cache.py \
    src/neural_forge/hf_lfm2_moving_gauge_residual_reclosure.py \
    src/neural_forge/hf_lfm2_moving_gauge_causal_targets.py \
    src/neural_forge/hf_lfm2_moving_gauge_conditioned_recapture_phase_cache.py; do
    if [ -f "$path" ]; then
        sha256sum "$path"
    else
        echo "MISSING $path"
    fi
done
