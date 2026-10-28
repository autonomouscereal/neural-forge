import json
import sys
from pathlib import Path


left = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
right = json.loads(Path(sys.argv[2]).read_text(encoding="ascii"))
differences = []


def compare(a, b, path="$"):
    if type(a) is not type(b):
        differences.append((path, type(a).__name__, type(b).__name__))
        return
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                differences.append((f"{path}.{key}", a.get(key), b.get(key)))
            else:
                compare(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list):
        if len(a) != len(b):
            differences.append((f"{path}.length", len(a), len(b)))
        for index, (left_item, right_item) in enumerate(zip(a, b)):
            compare(left_item, right_item, f"{path}[{index}]")
    elif a != b:
        differences.append((path, a, b))


compare(left, right)
print(f"difference_count={len(differences)}")
for row in differences[:300]:
    print(json.dumps(row, ensure_ascii=True))
