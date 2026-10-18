"""Run 1's base identity NOW (live tools dir, so the reader path matches), diffed field by field against the identity
stored in the checkpoint the queue refused to resume."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/cereal/omni-campaign/distill/tools")
import streamed_kd_trainer as skt
ck = json.load(open("/mnt/lexar/omni-campaign/distill/fullkd/run1/safety/ckpt-00000163/trainer_state.json"))
src = skt.P1Source(Path("/mnt/lexar/omni-campaign/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1"),
                   Path("/home/cereal/models/Qwen3.8-27B-Uncensored"), Path("/mnt/lexar/omni-campaign/distill/d2/p1_plan.json"))
now = src.identity()
now = json.loads(json.dumps(now, default=str))
print("checkpoint sha", ck["base_identity_sha256"], "recomputed from stored", skt.canonical_sha256(ck["base_identity"]))
print("now sha", skt.canonical_sha256(now))
diffs = []
def walk(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            walk(x.get(k), y.get(k), path + "/" + k)
    elif x != y:
        diffs.append(path); print("DIFF", path, json.dumps(x)[:160], "->", json.dumps(y)[:160])
walk(ck["base_identity"], now)
print("DIFFS", diffs)
Path("/home/cereal/cleanup-20261018/baseid/live-diff.json").write_text(json.dumps({"diffs": diffs, "checkpoint": ck["base_identity"], "now": now}, indent=1))
