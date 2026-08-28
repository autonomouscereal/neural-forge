#!/usr/bin/env python3
"""Write one child's per-case ADDENDUM 9 reading, defaulting every case to a miss.

The default is deliberate: a case a reviewer did not name is not a case that
passed.  Wave 16's `mx16_percase.py`, unchanged apart from this docstring.
"""

from __future__ import annotations

import json
import pathlib
import sys

CASES = [
    "conversation-natural-greeting", "instruction-three-actions", "explanation-seasons",
    "arithmetic-multi-step", "logic-ordering", "context-retrieval",
    "summarization-priorities", "comparison-ram-rom", "coding-deduplicate",
    "debugging-off-by-one", "writing-concise-email", "planning-dependencies",
    "uncertainty-insufficient-data", "creative-continuity", "multi-turn-correction",
]


def main() -> int:
    out = {case: "Observed miss" for case in CASES}
    for pair in sys.argv[2:]:
        case, _, verdict = pair.partition("=")
        if case not in out:
            raise SystemExit(f"unknown case {case}")
        out[case] = verdict
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(
        sys.argv[1],
        sum(1 for v in out.values() if v == "Useful"), "useful",
        sum(1 for v in out.values() if v == "Partial"), "partial",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
