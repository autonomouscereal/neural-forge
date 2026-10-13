"""Seal connected media-group admission from recorded source identities, CPU only.

Admission is split/contamination eligibility, not media correctness, tensor
validation or permission to train. No target files are downloaded or executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .corpus import EvalIndex, GroupIndex, SHA_RE, canonical, digest, file_sha256, identity_keys, iter_jsonl, normalize_record, task_text


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def asset_identities(value) -> set[str]:
    """Retain both actual asset hashes and declared parent-source family hashes.

    A declared hash is an identity exclusion key, never a claim of byte checking.
    File paths link old media rows lacking hashes to verified item metadata.
    """
    result = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"sha256", "declared_sha256", "source_media_sha256", "source_clip_sha256"} and isinstance(child, str) and SHA_RE.fullmatch(child.lower()):
                result.add("asset:" + child.lower())
            elif key in {"path", "stored_path", "source_path"} and isinstance(child, str) and child:
                result.add("asset-path:" + child.replace("\\", "/"))
            elif isinstance(child, (dict, list)):
                result.update(asset_identities(child))
    elif isinstance(value, list):
        for child in value:
            result.update(asset_identities(child))
    return result


def source_keys(row: dict) -> set[str]:
    p = row.get("provenance") or {}
    keys = identity_keys(row)
    if row.get("id"):
        keys.add("uid:" + str(row["id"]))
    rid = (p.get("registry_origin") or {}).get("rid")
    if rid:
        keys.add("rid:" + str(rid))
    for field in ("group_id", "recovery_group_id"):
        if p.get(field):
            keys.add("source-group:" + str(p[field]))
    keys.update(asset_identities(row.get("media") or []))
    return keys


def item_profile(kind: str, item: dict, metadata: dict) -> dict:
    if kind == "qimg":
        origin = item.get("corpus") or {}
        rid, uid = origin.get("rid"), origin.get("uid")
        sha = item.get("sha256")
        prompt = item.get("prompt")
        conditioning = {"prompt": prompt, "negative": item.get("negative", "")}
        asset_path = item.get("image")
    elif kind == "ace":
        rid, uid = item.get("rid"), metadata.get("uid")
        sha, prompt = item.get("sha256"), item.get("caption")
        conditioning = {"caption": prompt, "lyrics": item.get("lyrics", "")}
        asset_path = item.get("path")
    elif kind == "ace_lm":
        # Authored teacher prompt digest differs from the genuine source audio.
        rid, uid = str(item.get("id", "")).split("#", 1)[0], metadata.get("uid")
        sha, prompt = item.get("source_clip_sha256"), item.get("caption")
        conditioning = {"caption": prompt, "lyrics": item.get("lyrics", ""), "seconds": item.get("seconds")}
        asset_path = (metadata.get("clip") or {}).get("path")
    elif kind in {"qvl", "qvl_ref"}:
        rid, uid = item.get("rid"), metadata.get("uid")
        sha, prompt, asset_path = item.get("sha256"), "", item.get("path")
        # The actual vision teacher consumes an image, no invented caption.
        conditioning = {"asset_sha256": sha, "route": item.get("route")}
    else:
        raise ValueError("unsupported media admission kind")
    keys = {"rid:" + str(rid)} if rid else set()
    if uid:
        keys.add("uid:" + str(uid))
    keys.update(asset_identities(metadata))
    keys.update(asset_identities({"sha256": sha, "path": asset_path}))
    # Full conditioning, not generic captions such as 'A song.', defines duplicates.
    keys.add("conditioning:" + digest(conditioning))
    return {"kind": kind, "item_id": item.get("id"), "rid": rid, "uid": uid,
            "teacher_item_sha256": item.get("sha256"),
            "asset_sha256": sha, "asset_path": asset_path,
            "prompt_sha256": text_sha(prompt) if isinstance(prompt, str) else None,
            "conditioning_sha256": digest(conditioning), "conditioning": conditioning,
            "keys": keys, "held": item.get("held") is True,
            "item_payload_sha256": digest(item), "metadata_payload_sha256": digest(metadata)}


def followup_inventory(path, record):
    """Verify actual teacher corpus/item seals; partial caches never become ready."""
    body = json.loads(path.read_text(encoding="utf-8"))
    items = body.get("items")
    if not isinstance(items, list) or len(items) != body.get("n_items"):
        raise ValueError("follow-up teacher corpus item count differs")
    corpus_sha = digest({"schema": "omni-campaign.teacher_corpus.v1", "kind": body.get("kind"), "items": items})
    if corpus_sha != body.get("corpus_sha256") or record.get("source_corpus_sha256") != file_sha256(path):
        raise ValueError("follow-up corpus content/file seal mismatch")
    if any(type(item.get("held")) is not bool for item in items):
        raise ValueError("follow-up corpus has unknown item held flags")
    manifest = record.get("manifest")
    held, cached = set(), set()
    ready = False
    if manifest:
        if (manifest.get("schema") != "omni-campaign.teacher_cache.v2" or manifest.get("finite") is not True or
                manifest.get("corpus", {}).get("file_sha256") != file_sha256(path) or
                manifest.get("corpus", {}).get("sha256") != corpus_sha):
            raise ValueError("follow-up cache is not sealed against exact source")
        for shard in manifest.get("shards", []):
            identifier, value = shard.get("item_id"), (shard.get("meta") or {}).get("held")
            if not identifier or identifier in cached or type(value) is not bool:
                raise ValueError("follow-up cache missing/duplicate item or held flag")
            cached.add(identifier)
            if value:
                held.add(identifier)
        ready = cached == {item.get("id") for item in items}
        if not ready:
            raise ValueError("sealed follow-up cache does not cover exact item inventory")
    else:
        # Only captured partial evidence is known; do not infer unseen IDs/held bits.
        first = record.get("first_sidecar") or {}
        if first.get("item_id"):
            if first.get("corpus_sha256") != corpus_sha or type((first.get("item") or {}).get("held")) is not bool:
                raise ValueError("partial follow-up cache does not match source item")
            cached.add(first["item_id"])
            if first["item"]["held"]:
                held.add(first["item_id"])
    return body, held, cached, ready


def build_admission(*, normalized: Path, corpus_manifest: Path, carve: Path,
                    eval_rows: Path, qimg_items: Path, ace_items: Path,
                    qimg_metadata: Path, ace_metadata: Path, ace_cache_manifest: Path,
                    expected_hashes: dict[str, str], output: Path,
                    followup_metadata: Path | None = None, ace_lm_items: Path | None = None,
                    qvl_items: Path | None = None, qvl_ref_items: Path | None = None,
                    vit_metadata: Path | None = None, previous_admission: Path | None = None) -> dict:
    sources = {}
    paths = {"normalized": normalized, "corpus_manifest": corpus_manifest, "carve": carve,
             "eval_rows": eval_rows, "qimg_items": qimg_items, "ace_items": ace_items,
             "qimg_metadata": qimg_metadata, "ace_metadata": ace_metadata,
             "ace_cache_manifest": ace_cache_manifest}
    extras = {"followup_metadata": followup_metadata, "ace_lm_items": ace_lm_items, "qvl_items": qvl_items,
              "qvl_ref_items": qvl_ref_items, "vit_metadata": vit_metadata, "previous_admission": previous_admission}
    if any(extras.values()) and not all(extras.values()):
        raise ValueError("follow-up admission requires every corpus/metadata/prior seal")
    paths.update({name: path for name, path in extras.items() if path})
    for name, path in paths.items():
        actual = file_sha256(path)
        expected = expected_hashes.get(name)
        if not expected or expected != actual:
            raise ValueError(f"missing or mismatched source pin: {name}")
        sources[name] = {"path": str(path.resolve()), "sha256": actual}
    manifest = json.loads(corpus_manifest.read_text(encoding="utf-8"))
    if manifest["outputs"]["normalized"]["sha256"] != sources["normalized"]["sha256"]:
        raise ValueError("normalized rows not sealed by corpus manifest")
    c = json.loads(carve.read_text(encoding="utf-8"))
    carve_seal = hashlib.sha256(json.dumps({k: v for k, v in c.items() if k != "carve_sha256"}, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()
    if carve_seal != c.get("carve_sha256") or carve_seal != manifest.get("carve_sha256"):
        raise ValueError("carve seal does not match normalized campaign")
    if c.get("rows_sha256") not in {s.get("sha256") for s in manifest.get("sources", [])}:
        raise ValueError("carve source staging not pinned")
    if sources["eval_rows"]["sha256"] not in {s.get("sha256") for s in manifest.get("eval_sources", [])}:
        raise ValueError("eval rows not bound to normalized manifest")
    ace_body = json.loads(ace_items.read_text(encoding="utf-8"))
    cache = json.loads(ace_cache_manifest.read_text(encoding="utf-8"))
    if cache.get("corpus", {}).get("file_sha256") != sources["ace_items"]["sha256"]:
        raise ValueError("ACE cache not bound to these items")
    ace_records = ace_body.get("items")
    if not isinstance(ace_records, list) or len(ace_records) != ace_body.get("n_items"):
        raise ValueError("ACE item count mismatch")
    qimg_records = list(iter_jsonl(qimg_items))
    groups = GroupIndex()
    rows = []
    by_rid = defaultdict(list)
    evaluation = EvalIndex(normalize_record(r) for r in iter_jsonl(eval_rows))
    for row in iter_jsonl(normalized):
        p = row.get("provenance") or {}
        if row.get("schema") != "recovery_corpus.v1" or not p.get("recovery_group_id") or row.get("split") not in {"train", "validation", "test"}:
            raise ValueError("normalized row lacks pinned group/split")
        rid = (p.get("registry_origin") or {}).get("rid")
        blockers = set()
        for b in (row.get("quality") or {}).get("blockers", []):
            # A removed duplicate does not turn its accepted representative
            # into holdout. Propagate exclusion identities, not dedup row flags.
            if b.startswith("eval_") or b.startswith("frozen_carve_") and b != "frozen_carve_duplicate":
                blockers.add(b)
        if row["split"] != "train":
            blockers.add("recovery_holdout_group")
            evaluation.add(row)
        if rid and c["assignment"].get(rid) not in {"train", "duplicate"}:
            blockers.add("frozen_carve_" + str(c["assignment"].get(rid, "missing")))
        ix = groups.add(source_keys(row), row["split"])
        rows.append({"ix": ix, "id": row["id"], "rid": rid, "split": row["split"],
                     "group": p["recovery_group_id"], "blockers": blockers,
                     "user_prompt_sha256": text_sha("\n".join(m["content"] for m in row.get("messages", []) if m.get("role") == "user" and isinstance(m.get("content"), str))),
                     "source_sha256": (p.get("normalized_from") or {}).get("sha256")})
        if rid:
            by_rid[rid].append(len(rows) - 1)
    metadata = {}
    metadata_paths = [("qimg", qimg_metadata), ("ace", ace_metadata)]
    if vit_metadata:
        metadata_paths.append(("qvl", vit_metadata))
    for kind, path in metadata_paths:
        index = {}
        for item in iter_jsonl(path):
            rid = item.get("rid")
            if kind == "qvl":
                if not rid:
                    raise ValueError("QVL metadata has no source RID")
                index.setdefault(rid, []).append(item)
                continue
            if not rid or rid in index:
                raise ValueError(f"{kind} metadata missing/ambiguous RID")
            index[rid] = item
        metadata[kind] = index
    held = set()
    seen_shards = set()
    for shard in cache.get("shards", []):
        item_id = shard.get("item_id")
        value = (shard.get("meta") or {}).get("held")
        if not item_id or type(value) is not bool:
            raise ValueError("ACE cache shard lacks exact item/held flag")
        seen_shards.add(item_id)
        if value:
            held.add(item_id)
    followup, prior = {}, None
    if followup_metadata:
        captured = json.loads(followup_metadata.read_text(encoding="utf-8"))
        if captured.get("exit_code") != 0:
            raise ValueError("follow-up metadata collection failed")
        captured = captured["data"]["records"]
        for kind, path in (("ace_lm", ace_lm_items), ("qvl", qvl_items), ("qvl_ref", qvl_ref_items)):
            records = [r for r in captured if Path(r.get("source_corpus_file", "")).name == path.name]
            if len(records) != 1:
                raise ValueError("follow-up metadata source join missing or ambiguous")
            body, held_items, cached_items, ready = followup_inventory(path, records[0])
            followup[kind] = {"items": body["items"], "held": held_items, "cached": cached_items,
                              "ready": ready, "metadata": records[0], "metadata_sha256": digest(records[0]), "source": body}
        prior = json.loads(previous_admission.read_text(encoding="utf-8"))
        if digest({k: v for k, v in prior.items() if k != "manifest_sha256"}) != prior.get("manifest_sha256"):
            raise ValueError("previous admission manifest seal mismatch")
        for name in ("normalized", "corpus_manifest", "carve", "eval_rows", "qimg_items", "ace_items", "qimg_metadata", "ace_metadata", "ace_cache_manifest"):
            if prior["sources"][name]["sha256"] != sources[name]["sha256"]:
                raise ValueError("follow-up changes a preserved prior source pin")
    candidates = []
    seen_items = set()
    candidate_sources = [("qimg", qimg_records), ("ace", ace_records)]
    candidate_sources.extend((kind, value["items"]) for kind, value in followup.items())
    for kind, records in candidate_sources:
        for item in records:
            rid = ((item.get("corpus") or {}).get("rid") if kind == "qimg" else
                   str(item.get("id", "")).split("#", 1)[0] if kind == "ace_lm" else item.get("rid"))
            original = metadata.get("ace" if kind == "ace_lm" else kind, {}).get(rid)
            if kind == "qvl":
                # A genuine source can contain multiple images. Exact asset
                # equality, rather than RID alone, must resolve its metadata.
                matches = {digest(value): value for value in original or []
                           if (value.get("image") or {}).get("sha256") == item.get("sha256")}
                original = next(iter(matches.values())) if len(matches) == 1 else None
            candidate = item_profile(kind, item, original or {})
            blockers = set()
            joint_id = kind + ":" + str(candidate["item_id"])
            if not candidate["item_id"] or joint_id in seen_items:
                raise ValueError("duplicate/missing teacher item ID")
            seen_items.add(joint_id)
            if not original:
                blockers.add("original_media_metadata_join_missing")
            if len(by_rid.get(rid, [])) != 1:
                blockers.add("normalized_source_join_missing_or_ambiguous")
            linked = [rows[j] for j in by_rid.get(rid, [])]
            if linked and kind in {"ace_lm", "qvl"}:
                if candidate["uid"] and candidate["uid"] != linked[0]["id"]:
                    blockers.add("source_uid_mismatch")
                candidate["uid"] = linked[0]["id"]
                candidate["source_prompt_sha256"] = linked[0]["user_prompt_sha256"]
                if kind == "qvl":
                    candidate["prompt_sha256"] = linked[0]["user_prompt_sha256"]
                    candidate["prompt_scope"] = "original source user text only; actual vision teacher consumes image without text"
            if linked and candidate["uid"] and candidate["uid"] != linked[0]["id"]:
                blockers.add("source_uid_mismatch")
            if original and kind == "qimg" and ((original.get("image") or {}).get("sha256") != candidate["asset_sha256"] or original.get("prompt") != candidate["conditioning"]["prompt"]):
                blockers.add("qimg_asset_or_prompt_mismatch")
            if original and kind == "ace" and (original.get("clip") or {}).get("sha256") != candidate["asset_sha256"]:
                blockers.add("ace_asset_mismatch")
            if original and kind == "ace_lm" and (original.get("clip") or {}).get("sha256") != candidate["asset_sha256"]:
                blockers.add("ace_lm_source_clip_mismatch")
            if kind == "ace_lm" and digest({k: item.get(k) for k in ("caption", "lyrics", "seed", "seconds")}) != item.get("sha256"):
                blockers.add("ace_lm_teacher_prompt_payload_mismatch")
            if original and kind == "qvl" and (original.get("image") or {}).get("sha256") != candidate["asset_sha256"]:
                blockers.add("qvl_source_image_mismatch")
            if not SHA_RE.fullmatch(str(candidate["asset_sha256"] or "")) or not candidate["prompt_sha256"]:
                blockers.add("asset_or_prompt_hash_missing")
            if c["assignment"].get(rid) != "train":
                blockers.add("frozen_carve_" + str(c["assignment"].get(rid, "missing")))
            if kind == "ace":
                if candidate["item_id"] not in seen_shards:
                    blockers.add("ace_cache_item_missing")
                if candidate["held"] or candidate["item_id"] in held:
                    blockers.add("ace_captured_holdout")
            if kind in followup:
                entry = followup[kind]
                candidate["cache_ready"] = entry["ready"] and kind != "qvl_ref"
                candidate["native_train_ready"] = False
                candidate["teacher_corpus_sha256"] = entry["source"]["corpus_sha256"]
                candidate["teacher_cache_metadata_sha256"] = entry["metadata_sha256"]
                candidate["teacher_cache_manifest_file_sha256"] = entry["metadata"].get("manifest_file_sha256")
                if candidate["held"] or candidate["item_id"] in entry["held"]:
                    blockers.add(kind + "_captured_holdout")
                if kind == "qvl_ref":
                    blockers.add("qvl_ref32_diagnostic_never_train")
            # Screen both complete conditioning and original source wording.
            probe = {"messages": [{"role": "user", "content": "\n".join(str(v) for v in candidate["conditioning"].values())}],
                     "media": [{"sha256": candidate["asset_sha256"]}], "provenance": {"source_ids": [rid], "source": "curated_registry"}}
            blockers.update(evaluation.hits(probe))
            candidate["ix"] = groups.add(candidate["keys"], None)
            candidate["blockers"] = blockers
            candidate["source_links"] = [{k: link[k] for k in ("id", "rid", "split", "group", "source_sha256")} for link in linked]
            candidates.append(candidate)
    root_blocks = defaultdict(set)
    root_groups = defaultdict(set)
    for row in rows:
        root = groups.find(row["ix"])
        root_blocks[root].update(row["blockers"])
        root_groups[root].add(row["group"])
    for candidate in candidates:
        root_blocks[groups.find(candidate["ix"])].update(candidate["blockers"])
    out = []
    for candidate in candidates:
        root = groups.find(candidate["ix"])
        splits = sorted(groups.explicit[root])
        blockers = set(root_blocks[root])
        if splits != ["train"]:
            blockers.add("connected_group_not_train_exclusive")
        item = {k: v for k, v in candidate.items() if k not in {"keys", "ix", "blockers", "conditioning", "held"}}
        source_train_eligible = not blockers
        # Operational partial-cache blockers do not contaminate otherwise valid
        # source groups or consume text train rows. Actual held bits do propagate.
        if candidate["kind"] in followup and not candidate.get("cache_ready"):
            blockers.add("teacher_cache_not_sealed_or_diagnostic")
        item.update(schema="recovery_media_admission.v1", recovery_group_id=digest(sorted(root_groups[root])),
                    connected_source_groups=sorted(root_groups[root]), split=splits[0] if len(splits) == 1 else "excluded",
                    train_admitted=not blockers, source_train_eligible=source_train_eligible, blockers=sorted(blockers), source_pins=sources,
                    admission_scope="source/split/contamination identity only; adapter must independently verify cache tensors and teacher provenance")
        out.append(item)
    if output.exists():
        raise ValueError("output directory must be new")
    output.mkdir(parents=True)
    paths_out = {}
    cross_route_exclusions = []
    for row in rows:
        root = groups.find(row["ix"])
        # Additional media-held/conditioning exclusions must also reach text.
        reasons = root_blocks[root]
        if reasons and row["split"] == "train":
            cross_route_exclusions.append({"id": row["id"], "rid": row["rid"], "source_group_id": row["group"],
                                           "connected_group_id": digest(sorted(root_groups[root])), "blockers": sorted(reasons)})
    for name, selected in (("admission", out), ("train_admitted", [r for r in out if r["train_admitted"]])):
        path = output / (name + ".jsonl")
        path.write_bytes(b"".join(canonical(r) + b"\n" for r in selected))
        paths_out[name] = {"path": str(path.resolve()), "sha256": file_sha256(path), "rows": len(selected)}
    exclusion_path = output / "cross_route_training_exclusions.jsonl"
    exclusion_path.write_bytes(b"".join(canonical(r) + b"\n" for r in cross_route_exclusions))
    paths_out["cross_route_training_exclusions"] = {"path": str(exclusion_path.resolve()), "sha256": file_sha256(exclusion_path), "rows": len(cross_route_exclusions)}
    report = {"schema": "recovery_media_admission.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "sources": sources, "code_sha256": file_sha256(Path(__file__)), "carve_sha256": carve_seal,
              "connected_groups_verified": True, "eval_disjoint": True,
              "eval_disjoint_scope": "train_admitted rows only; excluded rows are retained and may overlap holdout",
              "counts": {kind: dict(Counter("train_admitted" if r["train_admitted"] else "excluded" for r in out if r["kind"] == kind)) for kind, _ in candidate_sources},
              "source_eligibility_counts": {kind: sum(r["source_train_eligible"] for r in out if r["kind"] == kind) for kind, _ in candidate_sources},
              "blockers": dict(Counter(b for r in out for b in r["blockers"])), "outputs": paths_out,
              "heldout_policy": "union captured ACE held bits across item/cache, source groups, all routes and shared media/source/conditioning identities",
              "near_duplicate_policy": "complete conditioning >50% shared 13-grams with frozen eval and normalized validation/test; original source eval blockers propagate",
              "limitations": ["Split eligibility does not grade captions, generation, or cache tensors.", "Declared source-family hashes are exclusion identities, not byte-verification receipts.", "Near-duplicate screen is lexical; it is not an exhaustive semantic duplicate detector."]}
    if prior:
        report["previous_admission"] = {"source_file_sha256": sources["previous_admission"]["sha256"],
            "manifest_sha256": prior["manifest_sha256"], "outputs": prior["outputs"], "all_original_source_pins_preserved": True}
        report["followup_cache_state"] = {kind: {"cache_ready": value["ready"] and kind != "qvl_ref",
            "source_items": len(value["items"]), "captured_sidecar_count": value["metadata"].get("sidecar_count"),
            "verified_manifest_items": len(value["cached"]) if value["ready"] else 0,
            "native_train_ready": False} for kind, value in followup.items()}
        report["heldout_policy"] += "; union ACE-LM/QVL source and sealed-cache held items; reference32 diagnostic identities excluded globally; partial-cache state remains operational only"
    report["manifest_sha256"] = digest(report)
    (output / "manifest.json").write_bytes(canonical(report) + b"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("normalized", "corpus_manifest", "carve", "eval_rows", "qimg_items", "ace_items", "qimg_metadata", "ace_metadata", "ace_cache_manifest", "pins", "output"):
        parser.add_argument("--" + field.replace("_", "-"), type=Path, required=True)
    for field in ("followup_metadata", "ace_lm_items", "qvl_items", "qvl_ref_items", "vit_metadata", "previous_admission"):
        parser.add_argument("--" + field.replace("_", "-"), type=Path)
    args = vars(parser.parse_args())
    pins = json.loads(args.pop("pins").read_text(encoding="utf-8"))
    result = build_admission(**args, expected_hashes=pins)
    print(json.dumps({"counts": result["counts"], "outputs": result["outputs"], "manifest_sha256": result["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
