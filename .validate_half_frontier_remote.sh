set -euo pipefail
LAB=/home/cereal/model-surgery-lab
SRC="$LAB/neural-forge-src"
PYTHON="$LAB/venv/bin/python"

printf '%s  %s\n' 00cd0b1323428c08f70cbd8e40c2ad6fdaa84b4ebf70c89d34756188ccfc083e "$SRC/src/neural_forge/dynamic_lfm2_targets.py" | sha256sum -c -
printf '%s  %s\n' 0c3e3f04c2800a64f96ac602f3999eba631c39915582f2259f0c268c1bfb144c "$SRC/tests/test_dynamic_lfm2_targets.py" | sha256sum -c -
export PYTHONPATH="$SRC/src:$SRC"
cd "$SRC"
"$PYTHON" -m pytest -q tests/test_dynamic_lfm2_targets.py
