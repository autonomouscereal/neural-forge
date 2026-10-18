#!/usr/bin/env python3
"""h3_items_ided.py -- the H3 corpus in the item schema stage_c_media_a reads, as a DERIVED file (the sealed corpus is
never edited).

`distill/corpus/media/h3/items.jsonl` (512 rows, 2026-10-12, media_a_corpus_items) carries `uid`, `video` and `audio` as
records ({"path", "probe", ...}); stage_c_media_a 1.0.2 `read_items` needs a top-level `id` and `h3-encode` reads
`item["video"]` / `item["audio"]` as PATHS.  Rule (fixed, recorded in the receipt): id = uid; video = video.path;
audio = audio.path (when the row has one); every original field kept under `source_record`; the row order is the
source order.  The output is written next to the source with a receipt naming both sha256s.
"""
import hashlib
import json
import sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/cereal/omni-campaign/distill/corpus/media/h3/items.jsonl")
out = src.with_name("items.ided.v1.jsonl")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
seen = set()
lines = []
dropped = []
for r in rows:
    if not str(r.get("prompt") or "").strip():
        dropped.append({"uid": r.get("uid"), "clip": r.get("clip"), "reason": "empty prompt"})
        continue
    ident = str(r["uid"])
    if ident in seen:
        raise SystemExit(f"duplicate uid {ident}")
    seen.add(ident)
    video = r["video"]["path"] if isinstance(r.get("video"), dict) else r.get("video")
    audio = r["audio"]["path"] if isinstance(r.get("audio"), dict) else r.get("audio")
    if not video or not Path(video).is_file():
        raise SystemExit(f"{ident}: video {video} missing")
    item = {"id": ident, "prompt": r["prompt"], "video": video, "split": r.get("split", "train"), "source_record": r}
    if audio:
        if not Path(audio).is_file():
            raise SystemExit(f"{ident}: audio {audio} missing")
        item["audio"] = audio
    lines.append(json.dumps(item, sort_keys=True))
tmp = out.with_name(out.name + ".tmp")
tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
tmp.replace(out)
receipt = {"schema": "omni-h3-items-ided-v1", "source": str(src), "source_sha256": sha(src), "out": str(out), "out_sha256": sha(out),
           "rows": len(lines), "dropped": dropped,
           "rule": "id = uid; video = video.path; audio = audio.path; original row under source_record; source order; rows with an empty prompt dropped"}
out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
print(json.dumps(receipt, indent=1))
