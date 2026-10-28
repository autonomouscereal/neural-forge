set -euo pipefail
root=/home/cereal/model-surgery-lab/deployments/neural-forge-f7bbf16d
manifest="$root/experiments/manifests/lfm2-exact4x-moving-gauge-v5-r12-launch.json"
key=/home/cereal/model-surgery-lab/keys/neural-forge-balanced-basis-phase-v1.key
cd "$root"
export CUDA_VISIBLE_DEVICES=GPU-f5a09535-fc85-a027-50f7-734b344747dd
export PYTHONPATH="$root/src:$root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/home/cereal/model-surgery-lab/venv/bin/python - <<'PY'
from pathlib import Path

from experiments import run_lfm2_exact4x_moving_gauge_v5 as v5
from experiments import run_lfm2_exact4x_moving_gauge_v5_r12_live as launch


class PreflightComplete(RuntimeError):
    pass


def stop_before_phase_execution(request, handlers, **kwargs):
    assert tuple(handlers) == v5.PHASE_ORDER
    assert request.local_paths["published_child"].endswith("published-child/child")
    raise PreflightComplete


manifest = launch.read_r12_v5_launch_manifest(
    Path("experiments/manifests/lfm2-exact4x-moving-gauge-v5-r12-launch.json")
)
dependencies = launch.R12V5LaunchDependencies(
    run_phase_machine=stop_before_phase_execution,
)
try:
    launch.execute_r12_v5_launch(
        manifest,
        authority_key_path=Path(
            "/home/cereal/model-surgery-lab/keys/neural-forge-balanced-basis-phase-v1.key"
        ),
        dependencies=dependencies,
    )
except PreflightComplete:
    print("PRODUCTION_PREFLIGHT_COMPLETE")
else:
    raise AssertionError("Production preflight unexpectedly executed phases")
PY
