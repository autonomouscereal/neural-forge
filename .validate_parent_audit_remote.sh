set -euo pipefail
LAB=/home/cereal/model-surgery-lab
SRC="$LAB/neural-forge-src"
PYTHON="$LAB/venv/bin/python"

printf '%s  %s\n' 0afdad925899c7365fb61307af3629ea3c0836f4f1b06f0d0ee057a083b111a8 "$SRC/src/neural_forge/hf_lfm2_global_causal_allocator.py" | sha256sum -c -
printf '%s  %s\n' 9571646b41f47cd35c69eed9f1fdf657b89f46563e1b1e24619ce000f267a677 "$SRC/src/neural_forge/hf_lfm2_variable_width_fold.py" | sha256sum -c -
printf '%s  %s\n' f57a42c3b4d0630dc1e68139d40a5946a359a9912c79e5077cd56a148ad04b25 "$SRC/experiments/build_lfm2_cached_causal_evidence.py" | sha256sum -c -
printf '%s  %s\n' e85d2598a25ef3b2cde610dd9d64424c1490259fad844b6222b622c235f3c779 "$SRC/tests/test_hf_lfm2_global_causal_allocator.py" | sha256sum -c -

export PYTHONPATH="$SRC/src:$SRC"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$SRC"
"$PYTHON" -m pytest -q tests/test_hf_lfm2_global_causal_allocator.py
