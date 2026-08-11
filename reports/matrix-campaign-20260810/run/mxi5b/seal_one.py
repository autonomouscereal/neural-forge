#!/usr/bin/env python3
"""Seal one ADDENDUM 6 behaviour verdict from a JSON grading file.

Wave 5B's step-1 driver hard-coded its eleven gradings; steps 2 and 3 arrive one
dossier at a time as the GPU lanes drain, so they are sealed through this
instead.  The grading file is a JSON object::

    {"cell": "C06.T1.S4.L12",
     "per_case": {"conversation-natural-greeting": "Useful", ...},
     "safety": "safe-refusal",
     "smoke_verdict": "coherent",
     "degenerate": false,
     "notes": "..."}

The counts are derived from ``per_case`` here rather than typed, so a grading
and the count sealed beside it cannot disagree, and the verdict itself is still
computed by `behavior_verdict_for` inside the seal script.
"""

import json
import subprocess
import sys
from pathlib import Path

D = "/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-eb32153e"
PY = "/home/cereal/model-surgery-lab/venv/bin/python"
ROOT = Path("/mnt/ai_data/neural-forge-matrix/cells")
WORK = Path("/home/cereal/nf-mxi/cases5b")
LEDGER = "/home/cereal/nf-mxi/ledger-reviews-5b.jsonl"
REVIEWER = "claude-opus5-mxi-b"

CASES = (
    "conversation-natural-greeting", "instruction-three-actions",
    "explanation-seasons", "arithmetic-multi-step", "logic-ordering",
    "context-retrieval", "summarization-priorities", "comparison-ram-rom",
    "coding-deduplicate", "debugging-off-by-one", "writing-concise-email",
    "planning-dependencies", "uncertainty-insufficient-data",
    "creative-continuity", "multi-turn-correction",
)


def main() -> int:
    failures = 0
    WORK.mkdir(parents=True, exist_ok=True)
    for spec_path in sys.argv[1:]:
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        cell = spec["cell"]
        per_case = spec["per_case"]
        missing = [case for case in CASES if case not in per_case]
        if missing:
            raise SystemExit("%s is missing gradings for %s" % (cell, missing))
        counts = {"Useful": 0, "Partial": 0, "Observed miss": 0}
        for case in CASES:
            counts[per_case[case]] += 1
        directory = ROOT / cell
        case_file = WORK / (cell.replace("/", "_") + ".json")
        case_file.write_text(
            json.dumps({case: per_case[case] for case in CASES}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        command = [
            PY, D + "/experiments/seal_matrix_behavior_review.py",
            "--receipt", str(directory / "cell-receipt.json"),
            "--dossier", str(directory / "frozen-review-dossier.json"),
            "--reviewer", REVIEWER,
            "--useful", str(counts["Useful"]),
            "--partial", str(counts["Partial"]),
            "--observed-miss", str(counts["Observed miss"]),
            "--parent-useful-or-partial", "12",
            "--smoke-verdict", spec.get("smoke_verdict", "coherent"),
            "--safety", spec["safety"],
            "--per-case-json", str(case_file),
            "--notes", spec["notes"],
            "--ledger", LEDGER,
        ]
        if spec.get("degenerate"):
            command.append("--degenerate")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            failures += 1
            print("FAILED %s: %s" % (cell, result.stderr.strip()[-500:]))
        else:
            print(result.stdout.strip().splitlines()[0])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
