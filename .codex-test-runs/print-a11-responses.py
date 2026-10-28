from __future__ import annotations

import json
from pathlib import Path

PATH = Path(
    "reports/sparse-semantic-a11-k32-split-cout-20260730/"
    "whole-response-dossier.json"
)

document = json.loads(PATH.read_text(encoding="utf-8"))
for index, comparison in enumerate(document["comparisons"], 1):
    print(
        f"===== {index}. {comparison['id']} "
        f"[{comparison['category']}] ====="
    )
    print("PROMPT:", comparison["prompt"])
    parent = comparison["parent_cached_response"]
    if isinstance(parent, dict):
        parent = parent.get("response_text", parent)
    print("PARENT:", parent)
    print("CHILD:", comparison["authoritative_child_target_text"])
    print()
