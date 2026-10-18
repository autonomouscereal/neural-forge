"""Print the lane's evidence block FROM the shipped self-test artifact.

The rule this file exists to enforce: nothing in a claim about this lane may
contain a number or a digest that is not ``grep``-able out of
``selftest-results.json``.  The previous report was written by hand from an
earlier run -- it quoted a directory digest of ``e95aa845f3e1cbf7`` where the
shipped artifact said ``6d89aa2e37c5e8de``, omitted a whole test, and inflated
every timing by 2-3x.  A hand-written transcript of a run is not a transcript.

So the transcript is generated: every line below is a leaf of the JSON, in the
JSON's own order, with the file's own SHA-256 at the top.

    python qf_evidence.py [--results selftest-results.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

LIST_PREVIEW = 6


def leaves(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten a JSON value to ``(path, rendered)`` pairs, lists previewed."""

    if isinstance(value, dict):
        rows: list[tuple[str, str]] = []
        for key, item in value.items():
            rows.extend(leaves(item, f"{prefix}.{key}" if prefix else str(key)))
        return rows
    if isinstance(value, list):
        if value and all(isinstance(item, (dict, list)) for item in value):
            rows = []
            for index, item in enumerate(value[:LIST_PREVIEW]):
                rows.extend(leaves(item, f"{prefix}[{index}]"))
            if len(value) > LIST_PREVIEW:
                rows.append((f"{prefix}[...]", f"{len(value) - LIST_PREVIEW} more entries"))
            return rows
        shown = json.dumps(value[:LIST_PREVIEW])
        if len(value) > LIST_PREVIEW:
            shown = f"{shown[:-1]}, ... {len(value) - LIST_PREVIEW} more]"
        return [(prefix, shown)]
    return [(prefix, json.dumps(value))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", default=str(Path(__file__).with_name("selftest-results.json"))
    )
    args = parser.parse_args()

    path = Path(args.results)
    raw = path.read_bytes()
    results = json.loads(raw.decode("utf-8"))

    passed = sum(1 for row in results.values() if row.get("status") == "PASS")
    seconds = sum(float(row.get("seconds", 0.0)) for row in results.values())
    print(f"artifact           {path.name}")
    print(f"artifact_sha256    {hashlib.sha256(raw).hexdigest()}")
    print(f"tests              {passed}/{len(results)} PASS")
    print(f"suite_seconds      {seconds:.2f} (sum of the recorded per-test seconds)")
    for name, row in results.items():
        print()
        print(f"{row.get('status')}  {name}  ({float(row.get('seconds', 0.0)):.2f}s)")
        for key, rendered in leaves(row.get("evidence", {})):
            print(f"    {key} = {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
