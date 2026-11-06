"""ingest.lanes -- $OC/distill/queues/LANES.json (omni-campaign.lanes.v1) -> lane.

Ingested before the run ingesters so runs can point at their lane.  The whole lane object is kept in
lane.detail; owner/state/lock_path are lifted into columns.
"""
from __future__ import annotations

from . import common as C
from .common import db

SOURCE = "lanes"

# lane key -> project (arms of the program, see project table)
LANE_PROJECT = {
    "L-D1": "chimera-omni", "L-TRAIN": "chimera-omni", "L-NATIVE": "chimera-omni", "L-TOOL": "chimera-omni",
    "L-K3": "k3", "L-K3ACC": "k3",
    "L-DATA": "corpus", "L-SPEECH-DATA": "speech", "L-MEDIA-DATA": "media",
    "L-FOLD": "neural-forge", "L-RESEARCH": "neural-forge",
    # 2026-10-21 P40 lane policy: experiments, and the serving cards (corpus generation + Sprite; media worker)
    "L-EXP": "neural-forge", "L-SERVE": "corpus", "L-MEDIA": "media",
}


def lanes_path() -> str:
    return C.Q + "/LANES.json"


def run(conn, batch: C.Batch, args) -> None:
    path = lanes_path()
    doc = C.load_json(path)
    if not isinstance(doc, dict) or not isinstance(doc.get("lanes"), dict):
        raise db.LabError(f"{path}: not an omni-campaign.lanes.v1 document")
    prov = batch.prov(path)
    for key, lane in doc["lanes"].items():
        batch.saw()
        if not isinstance(lane, dict):
            batch.skip(f"lane {key} is not an object")
            continue
        locks = lane.get("lock_paths") or []
        with conn.cursor() as cur:
            cur.execute("INSERT INTO lane (project, key, owner, state, lock_path, detail, src_path, src_sha256, src_host, "
                        "ingest_batch_id) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s) "
                        "ON CONFLICT (key) DO UPDATE SET project = EXCLUDED.project, owner = EXCLUDED.owner, "
                        "state = EXCLUDED.state, lock_path = EXCLUDED.lock_path, detail = EXCLUDED.detail, "
                        "src_path = EXCLUDED.src_path, src_sha256 = EXCLUDED.src_sha256, src_host = EXCLUDED.src_host, "
                        "ingest_batch_id = EXCLUDED.ingest_batch_id",
                        (LANE_PROJECT.get(key, "chimera-omni"), key, lane.get("owner"), lane.get("status"),
                         ",".join(locks) if locks else None, db.canonical_json(lane), *prov, batch.id))
        batch.wrote()
