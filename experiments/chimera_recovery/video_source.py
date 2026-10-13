"""Add unchanged video-understanding sources to the shared global admission.

This is source eligibility only. Authentic temporal encoder and answer targets
are separate requirements; prior admitted audio/AV sources cannot be lifted.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

try:
    from . import video_cache as V
except ImportError:
    import video_cache as V

A = V.A
HELPER_SHA = "529502569ddbd46e23c1b8e1050039356d22c22c07a266a9db1ed3903052c9b0"
COMMON = ("normalized", "corpus_manifest", "eval_rows", "frozen_exclusions", "prior_admission")


def visual_answer(groundtruth):
    """Verify the authored visual projection, retaining separate mux truth."""
    keys = ("color","frame","shape","start_seconds","x","y")
    events = groundtruth.get("events", [])
    if (groundtruth.get("frame_count") != 8 or groundtruth.get("count") != len(events) or not events or
            any(set(keys)-set(e) or type(e["frame"]) is not int or not 0 <= e["frame"] < 8 or
                e["start_seconds"] != e["frame"] / 8 for e in events)):
        raise ValueError("authored visual events must identify the full actual frame/onset clock")
    return {"count":len(events),"events":[{k:e[k] for k in keys} for e in events]}


def rows(path):
    result = list(A.rows(path))
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("source IDs must occur exactly once")
    return result


def candidate(record):
    media, p = record.get("media", []), record.get("provenance", {})
    if (record.get("schema") != "recovery_corpus.v1" or record.get("route") != "video_understand" or
            record.get("split") not in {"train","validation","test"} or p.get("source_year") not in {2025,2026} or
            p.get("source_verified") is not True or p.get("split") != record["split"] or not p.get("group_id") or
            len(media) != 1 or media[0].get("type") != "video" or media[0].get("role") != "input" or
            [m.get("role") for m in record.get("messages", [])] != ["user","assistant"] or
            any(not isinstance(m.get("content"),str) or not m["content"] for m in record.get("messages", [])) or
            record.get("supervision",{}).get("policy") != "assistant_only" or
            record["supervision"].get("message_mask") != [0,1] or
            record.get("quality",{}).get("groundtruth_verified") is not True):
        raise ValueError("unchanged authored video input/answer/source split required")
    item = media[0]
    source = A.pinned(item,"actual physical video")
    truth = A.pinned({"path":p["groundtruth_path"],"sha256":p["groundtruth_sha256"]},"actual authored video truth")
    g = json.loads(truth.read_text())["groundtruth"]
    answer = json.loads(record["messages"][-1]["content"])
    stream = next((s for s in item.get("probe",{}).get("streams",[]) if s.get("codec_type") == "video"),{})
    if (source.stat().st_size != item.get("bytes") or item.get("source_verified") is not True or
            item.get("source_media_sha256") != item["sha256"] or item.get("start_seconds") != 0. or
            item.get("fps") != 8 or item.get("duration_seconds") != 1. or stream.get("avg_frame_rate") != "8/1" or
            stream.get("start_time") != "0.000000" or float(stream.get("duration",0)) != 1. or
            (stream.get("height"),stream.get("width")) != (96,96) or
            g.get("same_origin_sha256") != item["sha256"] or A.digest(visual_answer(g)) != A.digest(answer)):
        raise ValueError("full one-second eight-frame observation or authored answer seal differs")
    return {"record":record,"asset_path":str(source),"asset_sha256":item["sha256"],
            "groundtruth_sha256":p["groundtruth_sha256"],"source_scope":"full original one-second eight-frame physical video",
            "native_encoder_requirement":"actual temporal C11/ViT13 and assistant K64 source-aligned caches"}


def prepare(*, parent, parent_sha256, admission_helper, output):
    if output.exists():
        raise ValueError("fresh source-admission namespace required")
    parent = A.pinned({"path":str(parent),"sha256":parent_sha256},"prior complete shared audio/AV admission")
    old_manifest = json.loads(parent.read_text())
    if (old_manifest.get("schema") != "recovery_audio_admission.v1" or old_manifest.get("sealed") is not True or
            any(old_manifest.get(k) is not True for k in ("complete_source_traversal","frozen_exclusions_verified","connected_groups_verified")) or
            A.digest({k:v for k,v in old_manifest.items() if k != "manifest_sha256"}) != old_manifest.get("manifest_sha256")):
        raise ValueError("prior complete global source seal differs")
    helper = V.module(admission_helper,HELPER_SHA,"recovery_video_current_global_admission")
    for name,spec in old_manifest["sources"].items():
        A.pinned(spec,"prior global source "+name)
    original = rows(A.pinned(old_manifest["sources"]["source"],"prior source rows"))
    inventory_path = A.pinned(old_manifest["sources"]["source_inventory"],"full shared source inventory")
    inventory = rows(inventory_path); whole = {r["id"]:r for r in inventory}
    old = {r["id"]:r for r in rows(A.pinned(old_manifest["outputs"]["admission"],"prior admitted/excluded candidates"))}
    if set(old) != {r["id"] for r in original} or any(A.digest(r) != A.digest(whole.get(r["id"])) for r in original):
        raise ValueError("prior source rows do not match unchanged complete inventory")
    candidates = []
    removed = {"schema","id","item_id","route","split","recovery_group_id","connected_source_groups","source_group_id",
        "source_row_sha256","authored_answer_sha256","prompt_sha256","source_train_eligible","train_admitted",
        "native_train_ready","encoder_status","source_pins"}
    for record in original:
        entry = old[record["id"]]
        if A.digest(record) != entry["source_row_sha256"] or (not entry["train_admitted"] and not entry["blockers"]):
            raise ValueError("prior source identity/exclusion is ambiguous")
        A.pinned({"path":entry["asset_path"],"sha256":entry["asset_sha256"]},"prior actual source asset")
        c = {k:v for k,v in entry.items() if k not in removed}
        c.update(record=record,original_source_admission=V.pin(parent)); candidates.append(c)
    new = [candidate(r) for r in inventory if r["route"] == "video_understand"]
    if len(new) != 24 or any(c["record"]["id"] in old for c in new):
        raise ValueError("exact complete new24-video scope required")
    candidates.extend(new); output.mkdir(parents=True)
    source = output/"source_ready.jsonl"; source.write_bytes(b"".join(A.canonical(c["record"])+b"\n" for c in candidates))
    specs = {k:old_manifest["sources"][k] for k in COMMON}
    specs.update(source=V.pin(source),source_inventory=V.pin(inventory_path),prior_shared_audio_av_admission=V.pin(parent),
        prior_shared_audio_av_rows=old_manifest["outputs"]["admission"],source_admission_helper=V.pin(admission_helper))
    manifest = helper.seal_admission(specs=specs,candidates=candidates,inventory=inventory,output=output/"global_admission")
    current = {r["id"]:r for r in rows(output/"global_admission/admission.jsonl")}; changes = []
    for ident,previous in old.items():
        after = current[ident]
        if set(previous["blockers"])-set(after["blockers"]) or (not previous["train_admitted"] and after["train_admitted"]):
            raise ValueError("new traversal lifted a prior source/frozen exclusion")
        if (after["train_admitted"],after["recovery_group_id"]) != (previous["train_admitted"],previous["recovery_group_id"]):
            changes.append({"id":ident,"prior_group":previous["recovery_group_id"],"new_group":after["recovery_group_id"],
                "train_admitted":after["train_admitted"],"blockers":after["blockers"]})
    result = {"schema":"recovery_video_source_preparation.v1","sealed":True,"source_year":2026,"source":V.pin(source),
        "source_admission":V.pin(output/"global_admission/manifest.json"),"prior_source_admission":V.pin(parent),
        "full_source_inventory":len(inventory),"full_candidates":len(candidates),"prior_native_source_changes":changes,
        "video_source_ids":[c["record"]["id"] for c in new],
        "video_train_ids":[c["record"]["id"] for c in new if current[c["record"]["id"]]["train_admitted"]],
        "video_held_ids":[c["record"]["id"] for c in new if c["record"]["split"] != "train"],
        "native_train_ready":False,"training_started":False,"code_sha256":A.file_sha(Path(__file__))}
    (output/"preparation.json").write_bytes(A.canonical(result)+b"\n")
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("parent","admission-helper","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--parent-sha256",required=True)
    a=p.parse_args(argv); r=prepare(**vars(a))
    print(json.dumps({"video_sources":len(r["video_source_ids"]),"video_train":len(r["video_train_ids"]),
        "held":len(r["video_held_ids"]),"prior_changes":len(r["prior_native_source_changes"]),"native_train_ready":False}))


if __name__=="__main__":
    main()
