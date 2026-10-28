set -euo pipefail

/home/cereal/model-surgery-lab/venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path('/home/cereal/model-surgery-lab/artifacts/lfm2_dynamic_heterogeneous_20260729')
for path in sorted(root.glob('*/a*/run.log')):
    try:
        payload = json.loads(path.read_text(encoding='utf-8').splitlines()[-1])
    except Exception as exc:
        print(path.parent, 'PARSE', type(exc).__name__, str(exc))
        continue
    chain = payload.get('failure', {}).get('exception_chain', [])
    errors = ' -> '.join(f"{row.get('type')}: {row.get('message')}" for row in chain)
    print(path.parent.relative_to(root), payload.get('status'), errors)

target = root / 'neural-forge-07fafdb0-d04ae8f2' / 'a0004-aggressive-h784-exact57420144'
for name in ('diagnostics.json', 'receipt.json'):
    path = target / name
    payload = json.loads(path.read_text(encoding='utf-8'))
    hits = []
    def walk(value, location=''):
        if isinstance(value, dict):
            for key, child in value.items():
                child_location = f'{location}.{key}' if location else key
                if 'version' in key.lower() or 'transformer' in key.lower() or 'torch' in key.lower():
                    hits.append((child_location, child))
                walk(child, child_location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f'{location}[{index}]')
    walk(payload)
    print(name, hits)
PY
