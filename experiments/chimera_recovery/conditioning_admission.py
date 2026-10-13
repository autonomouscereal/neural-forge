"""Repin complete authentic ACE code hints to an unchanged admitted item set.

CPU file verification only; no model loading or hint recomputation. The native
ACE converter must still verify each actual hint/code tensor before training.
Old artifacts are never modified and a new output directory is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from .corpus import SHA_RE, canonical, digest, file_sha256, iter_jsonl

IDENTITY_FIELDS = ("item_id", "recovery_group_id", "asset_sha256", "prompt_sha256",
                   "conditioning_sha256", "item_payload_sha256", "split")


def pinned(path: Path, expected: str, label: str) -> dict:
    if not isinstance(expected, str) or not SHA_RE.fullmatch(expected) or file_sha256(path) != expected:
        raise ValueError("changed or unpinned " + label)
    return json.loads(path.read_text(encoding="utf-8"))


def sibling(base: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "/" in name or "\\" in name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("unsafe cache sibling path")
    path = base / name
    if path.resolve().parent != base.resolve() or path.is_symlink():
        raise ValueError("cache sibling escapes pinned directory")
    return path


def admission(path: Path, expected: str) -> tuple[dict, dict]:
    body = pinned(path, expected, "admission manifest")
    if (body.get("schema") != "recovery_media_admission.v1" or body.get("eval_disjoint") is not True or
            body.get("connected_groups_verified") is not True or
            digest({k: v for k, v in body.items() if k != "manifest_sha256"}) != body.get("manifest_sha256")):
        raise ValueError("unverified connected admission")
    output = body["outputs"]["admission"]
    row_path = Path(str(output["path"]).replace("\\", "/"))
    if not row_path.is_file():
        row_path = sibling(path.parent, row_path.name)
    if file_sha256(row_path) != output["sha256"]:
        raise ValueError("admission rows moved")
    rows = list(iter_jsonl(row_path))
    if len(rows) != output["rows"]:
        raise ValueError("admission rows incomplete")
    result = {}
    for row in rows:
        if row.get("kind") != "ace":
            continue
        ident = row.get("item_id")
        if not ident or ident in result or row.get("schema") != "recovery_media_admission.v1" or type(row.get("train_admitted")) is not bool:
            raise ValueError("invalid ACE admission inventory")
        if row["train_admitted"] and (row.get("split") != "train" or row.get("blockers")):
            raise ValueError("admitted ACE item has holdout/blockers")
        result[ident] = row
    return body, result


def rebind_ace(*, cache: Path, items: Path, old_admission: Path, new_admission: Path,
               conditioning: Path, output: Path, source_manifest_sha256: str,
               old_admission_sha256: str, new_admission_sha256: str,
               conditioning_sha256: str, conditioner_sha256: str) -> dict:
    """Copy verified hints only when every admitted ACE identity is unchanged.

    The caller must pin the previously audited conditioner implementation.
    This does not accept expanded/subset item sets or reassigned source groups.
    """
    if output.exists():
        raise ValueError("use a new output directory")
    manifest_path = cache / "manifest.json"
    manifest = pinned(manifest_path, source_manifest_sha256, "teacher cache manifest")
    old, old_rows = admission(old_admission, old_admission_sha256)
    new, new_rows = admission(new_admission, new_admission_sha256)
    if any(new.get("sources", {}).get(k, {}).get("sha256") != v.get("sha256") for k, v in old["sources"].items()):
        raise ValueError("original source or frozen evaluation pins changed")
    if (manifest.get("schema") != "omni-campaign.teacher_cache.v2" or manifest.get("teacher") != "ace_dit_velocity" or
            manifest.get("finite") is not True or not manifest.get("sealed_at_utc") or
            manifest.get("teacher_integrity", {}).get("equal") is not True):
        raise ValueError("teacher cache is not complete finite immutable ACE")
    source_sha = file_sha256(items)
    if (old["sources"]["ace_items"]["sha256"] != source_sha or new["sources"]["ace_cache_manifest"]["sha256"] != source_manifest_sha256 or
            manifest.get("corpus", {}).get("file_sha256") != source_sha):
        raise ValueError("teacher cache source identity differs from admissions")
    corpus = json.loads(items.read_text(encoding="utf-8"))
    source_items = {item["id"]: item for item in corpus["items"]}
    if len(source_items) != len(corpus["items"]) or corpus.get("corpus_sha256") != manifest["corpus"]["sha256"]:
        raise ValueError("source corpus IDs or seal differ")
    before = {k for k, row in old_rows.items() if row["train_admitted"]}
    after = {k for k, row in new_rows.items() if row["train_admitted"]}
    if not before or before != after:
        raise ValueError("admitted ACE item set changed; recompute/review conditioning")
    if any(old_rows[k].get(f) != new_rows[k].get(f) for k in before for f in IDENTITY_FIELDS):
        raise ValueError("admitted ACE group/asset/prompt/conditioning identity changed")
    condition = pinned(conditioning, conditioning_sha256, "conditioning manifest")
    if (condition.get("schema") != "chimera_recovery.ace_conditioning.v1" or
            any(condition.get(k) is not True for k in ("sealed", "all_finite", "complete_source_traversal")) or
            condition.get("source_manifest_sha256") != source_manifest_sha256 or
            condition.get("corpus_sha256") != old_admission_sha256 or condition.get("adapter_sha256") != conditioner_sha256):
        raise ValueError("conditioning is not complete pinned original computation")
    component = condition.get("component", {})
    tensors = component.get("component_tensors")
    integrity = manifest["teacher_integrity"]
    source_pins = component.get("source_pins")
    def teacher_identity(record):
        return {v["path"]: (v["sha256"], v["bytes"]) for v in record["files"]}
    if (not isinstance(tensors, dict) or len(tensors) != 32 or any(not SHA_RE.fullmatch(str(v)) for v in tensors.values()) or
            component.get("component_sha256") != digest(tensors) or component.get("device") != "cpu" or component.get("dtype") != "float32" or
            not isinstance(source_pins, dict) or not source_pins or any(not SHA_RE.fullmatch(str(v)) for v in source_pins.values()) or
            not integrity.get("before") or any(teacher_identity(v) != teacher_identity(integrity["after"]) for v in integrity["before"]) or
            not any(v["sha256"] == component.get("teacher_sha256") for v in integrity["after"]["files"])):
        raise ValueError("conditioning component tensor identity incomplete")
    hints = {row["item_id"]: row for row in condition["items"]}
    if len(hints) != len(condition["items"]) or set(hints) != before:
        raise ValueError("conditioning hint inventory is incomplete/extra")
    seen, hint_paths = set(), {}
    for shard in manifest["shards"]:
        ident = shard["item_id"]
        if ident in seen or ident not in source_items or ident not in new_rows or shard.get("finite") is not True:
            raise ValueError("teacher item inventory incomplete/duplicate")
        seen.add(ident)
        source_path = sibling(cache, shard["name"])
        if source_path.suffix != ".safetensors" or file_sha256(source_path) != shard["sha256"]:
            raise ValueError("teacher shard bytes moved")
        sidecar_path = source_path.with_suffix(".json")
        if sidecar_path.is_symlink():
            raise ValueError("teacher sidecar escapes pinned directory")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        item = source_items[ident]
        captured_item = sidecar.get("item", {})
        if (sidecar.get("schema") != "omni-campaign.teacher_cache_shard.v2" or sidecar.get("item_id") != ident or
                sidecar.get("file_sha256") != shard["sha256"] or sidecar.get("finite") is not True or
                sidecar.get("corpus_sha256") != corpus["corpus_sha256"] or sidecar.get("config_sha256") != manifest["config_sha256"] or
                sidecar.get("teacher") != "ace_dit_velocity" or not isinstance(captured_item, dict) or
                not {"id", "sha256", "caption", "lyrics", "seed", "held"} <= set(captured_item) or
                any(item.get(k) != v for k, v in captured_item.items())):
            raise ValueError("teacher sidecar/source identity differs")
        if ident not in before:
            continue
        held = (shard.get("meta") or {}).get("held")
        if type(held) is not bool or held or item.get("held") is not False:
            raise ValueError("admitted teacher item is held or unknown")
        hint = hints[ident]
        if (hint.get("source_shard_sha256") != shard["sha256"] or hint.get("source_sidecar_sha256") != file_sha256(sidecar_path) or
                hint.get("recovery_group_id") != new_rows[ident]["recovery_group_id"] or
                type(hint.get("frames")) is not int or hint["frames"] < 1 or
                any(not SHA_RE.fullmatch(str(hint.get(k))) for k in ("hints_sha256", "codes_sha256"))):
            raise ValueError("hint source/group/tensor identity differs")
        hp = sibling(conditioning.parent, hint["path"])
        if hp in hint_paths.values() or file_sha256(hp) != hint.get("sha256"):
            raise ValueError("hint bytes changed or reused across items")
        hint_paths[ident] = hp
    if seen != set(source_items) or set(hint_paths) != before:
        raise ValueError("teacher/hint inventory not fully traversed")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ace-admission-rebind-", dir=output.parent))
    try:
        for ident, source in hint_paths.items():
            target = stage / hints[ident]["path"]
            shutil.copyfile(source, target)
            if file_sha256(target) != hints[ident]["sha256"]:
                raise ValueError("hint copy byte identity failed")
        result = dict(condition)
        result["corpus_sha256"] = new_admission_sha256
        result["admission_rebind"] = {"schema": "chimera_recovery.ace_admission_rebind.v1",
            "helper_sha256": file_sha256(Path(__file__)), "original_conditioning_manifest_sha256": conditioning_sha256,
            "original_admission_sha256": old_admission_sha256, "new_admission_sha256": new_admission_sha256,
            "source_items_sha256": source_sha, "source_manifest_sha256": source_manifest_sha256,
            "unchanged_item_identity_sha256": digest({k: {f: new_rows[k][f] for f in IDENTITY_FIELDS} for k in sorted(before)}),
            "unchanged_hint_inventory_sha256": digest(condition["items"]), "hint_bytes_preserved": True,
            "tensor_semantics": "must be independently revalidated by native prepare_ace", "model_loaded": False}
        (stage / "manifest.json").write_bytes(canonical(result) + b"\n")
        # Detect concurrent source movement before publishing. Per-file hint and
        # source shard hashes were checked above; prepare_ace checks them again.
        for path, expected in ((conditioning, conditioning_sha256), (old_admission, old_admission_sha256),
                               (new_admission, new_admission_sha256), (manifest_path, source_manifest_sha256), (items, source_sha)):
            if file_sha256(path) != expected:
                raise ValueError("source moved during immutable conditioning rebind")
        stage.rename(output)
        return result
    except BaseException:
        shutil.rmtree(stage)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "items", "old-admission", "new-admission", "conditioning", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-manifest-sha256", "old-admission-sha256", "new-admission-sha256", "conditioning-sha256", "conditioner-sha256"):
        parser.add_argument("--" + name, required=True)
    result = rebind_ace(**vars(parser.parse_args()))
    print(json.dumps({"items": len(result["items"]), "sealed": result["sealed"], "model_loaded": False,
                      "native_tensor_revalidation_required": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
