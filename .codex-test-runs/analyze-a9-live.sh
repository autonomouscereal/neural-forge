set -eu

RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_bridge_20260730/a9-k32-residual-omp
RUN="$RUN" python3 - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ["RUN"])
rows = {}
seen_depths = {}
sample_after = {}
sample_coordinates = {}
with (run / "job.log").open("r", encoding="utf-8", errors="replace") as handle:
    for raw in handle:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        cursor = event
        for key in ("details", "details", "compiler_progress", "core_progress"):
            cursor = cursor.get(key, {}) if isinstance(cursor, dict) else {}
        if not isinstance(cursor, dict):
            continue
        depth = cursor.get("support_depth")
        after = cursor.get("relative_residual_l2_after")
        start = cursor.get("donor_row_start")
        stop = cursor.get("donor_row_stop")
        if isinstance(depth, int):
            seen_depths[depth] = seen_depths.get(depth, 0) + 1
            if depth in {16, 32} and depth not in sample_after:
                sample_after[depth] = after
                sample_coordinates[depth] = {
                    "start": start,
                    "stop": stop,
                    "start_type": type(start).__name__,
                    "stop_type": type(stop).__name__,
                }
        if (
            isinstance(depth, int)
            and depth in {16, 32}
            and isinstance(after, dict)
            and isinstance(start, int)
            and isinstance(stop, int)
            and isinstance(after.get("count"), int)
            and isinstance(after.get("mean"), (int, float))
        ):
            rows[(start, stop, depth)] = {
                "count": int(after["count"]),
                "mean": float(after["mean"]),
            }

def aggregate(depth, keys):
    values = [rows[(start, stop, depth)] for start, stop in keys]
    count = sum(item["count"] for item in values)
    return {
        "chunk_count": len(values),
        "row_count": count,
        "weighted_mean_relative_residual_l2": (
            sum(item["count"] * item["mean"] for item in values) / count
            if count else None
        ),
    }

k16 = {(start, stop) for start, stop, depth in rows if depth == 16}
k32 = {(start, stop) for start, stop, depth in rows if depth == 32}
paired = sorted(k16 & k32)
a16 = aggregate(16, paired)
a32 = aggregate(32, paired)
all32 = aggregate(32, sorted(k32))
ratio = None
if a16["weighted_mean_relative_residual_l2"]:
    ratio = (
        a32["weighted_mean_relative_residual_l2"]
        / a16["weighted_mean_relative_residual_l2"]
    )
print(json.dumps({
    "schema": "neural-forge-live-k16-k32-quality-curve-v1",
    "paired_complete_chunks": len(paired),
    "covered_donor_rows": sum(stop - start for start, stop in paired),
    "k16": a16,
    "k32": a32,
    "all_available_k32": all32,
    "k32_to_k16_residual_ratio": ratio,
    "relative_residual_reduction_fraction": None if ratio is None else 1.0 - ratio,
    "seen_depths": seen_depths,
    "sample_after": sample_after,
    "sample_coordinates": sample_coordinates,
}, sort_keys=True))
PY
