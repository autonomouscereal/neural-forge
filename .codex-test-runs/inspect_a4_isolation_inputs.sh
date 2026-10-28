set -euo pipefail

python3 - <<'PY'
import json
from pathlib import Path

paths = {
    "prompts": Path("/home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2/research/lfm2_semantic_whole_response_prompts_v1.json"),
    "envelope": Path("/home/cereal/model-surgery-lab/artifacts/lfm2_whole_response_parent_20260721/e739a88/a0002/parent-envelope.json"),
    "span": Path("/home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/lfm2-semantic-step-global-raw-span/3e780d07ac1034ee77756f973b483637cee846e17e6d7191d6b20c9533360167/payload/span-evidence.json"),
    "codebook": Path("/home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/lfm2-sparse-semantic-codebook/51b42fbcbbcd3c5b638c6a9d2c50a2596173a5e9181db0845b72d6970e15f3c0/payload/artifact/receipt.json"),
}
for label, path in paths.items():
    print(label, "exists=", path.is_file(), "bytes=", path.stat().st_size if path.is_file() else None)
    if not path.is_file():
        continue
    document = json.loads(path.read_text(encoding="utf-8"))
    print(label, "root_type=", type(document).__name__)
    if isinstance(document, dict):
        print(label, "keys=", sorted(document)[:40])
        rows = document.get("prompts") or document.get("rows") or document.get("responses") or document.get("comparisons")
    else:
        rows = document
    if isinstance(rows, list) and rows:
        print(label, "rows=", len(rows), "first_keys=", sorted(rows[0]))
        print(label, "first_id=", rows[0].get("id"))
PY
