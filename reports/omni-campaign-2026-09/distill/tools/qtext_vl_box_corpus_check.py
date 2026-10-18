#!/usr/bin/env python3
"""qtext_vl_box_corpus_check.py -- what qtext_teacher_cache 1.0.6 makes of the REAL corpus-lane
vision_in / video_in JSONL under --reference-boxes qwen1000 (CPU, read-only), before j6 runs.

Prints and writes a receipt with, per corpus: rows kept, rows with boxes, rows re-stated on the
0-1000 grid by convention, every quarantined row with its reason, the 1.0.6 corpus sha (the j6
cache directory name), and the re-stated reference of the j3 smoke grounding row next to the
27B teacher's own j3 answer, so a reader can see the frames now agree.

    PYTHONPATH=$OC/distill/tools $PY qtext_vl_box_corpus_check.py --receipt <json> <jsonl>...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import qtext_teacher_cache as qc

#: The j3 smoke's grounding row and the teacher's own answer for it (receipt
#: $OC/distill/tps/qtext-smoke-vl-27b-gpu0-v104-2026-10-13.json, row 1).
J3_GROUNDING_UID = "bae2dba582904c43f2d659743a950be1b3cef2705da04073114bb3be407734d5"
J3_TEACHER_ANSWER = ('[{"bbox_2d": [157, 727, 300, 795], "label": "cyan_bar_6"}, '
                     '{"bbox_2d": [207, 790, 298, 905], "label": "blue_circle_4"}, ...] (80-token cap)')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--mode", default="qwen1000", choices=list(qc.REFERENCE_BOX_MODES))
    ap.add_argument("jsonl", nargs="+")
    args = ap.parse_args()
    out = {"what": "qtext 1.0.6 reference-box handling over the real corpus-lane VL JSONL (read-only)",
           "tool": qc.tool_identity(), "mode": args.mode, "started_utc": qc.utc_now(), "corpora": {}}
    for path in args.jsonl:
        started = time.perf_counter()
        corpus = qc.load_vl_corpus(Path(path), args.mode)
        boxes = corpus["reference_boxes"]
        entry = {"path": corpus["path"], "jsonl_sha256": corpus["jsonl_sha256"], "corpus_sha256_v106": corpus["sha256"],
                 "rows_kept": corpus["rows"], "media_files": corpus["media_files"], "reference_boxes": boxes,
                 "seconds": round(time.perf_counter() - started, 1)}
        examples = {}
        for row in corpus["rows_data"]:
            answer = row["messages"][-1]["content"][0]["text"] if row["messages"][-1]["role"] == "assistant" else ""
            if row["row_id"] == J3_GROUNDING_UID:
                examples["j3_grounding_row"] = {"row_id": row["row_id"], "reference_v106": answer,
                                                "teacher_j3_answer": J3_TEACHER_ANSWER}
            elif "bbox [" in answer and "xywh_example" not in examples:
                examples["xywh_example"] = {"row_id": row["row_id"], "reference_v106": answer[:700]}
        entry["examples"] = examples
        out["corpora"][Path(path).name] = entry
        print(f"{Path(path).name}: kept {corpus['rows']} rows; with boxes {boxes['rows_with_boxes']}; "
              f"re-stated {boxes['rows_regridded']} ({boxes['boxes_regridded']} boxes, {boxes['conventions']}); "
              f"quarantined {len(boxes['rows_quarantined'])}; corpus sha {corpus['sha256']}; {entry['seconds']} s",
              flush=True)
        reasons: dict[str, int] = {}
        for q in boxes["rows_quarantined"]:
            reasons[q["reason"]] = reasons.get(q["reason"], 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  QUARANTINE {count} x {reason}")
        for name, example in examples.items():
            print(f"  {name}: {json.dumps(example)[:900]}")
    out["finished_utc"] = qc.utc_now()
    qc.write_json_atomic(Path(args.receipt), out)
    print(f"receipt {args.receipt} sha256 {qc.file_sha256(Path(args.receipt))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
