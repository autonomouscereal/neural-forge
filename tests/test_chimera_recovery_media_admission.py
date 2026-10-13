"""Connected split/identity checks using independent authored source fixtures."""
import hashlib
import json

import pytest

from experiments.chimera_recovery import corpus as c
from experiments.chimera_recovery import media_admission as m


def fixture(tmp_path, *, shared_holdout=False, ace_held=False, wrong_prompt=False):
    paths = {}

    def write(name, value, lines=False):
        p = tmp_path / (name + (".jsonl" if lines else ".json"))
        p.write_bytes(b"".join(c.canonical(v) + b"\n" for v in value) if lines else c.canonical(value))
        paths[name] = p

    def row(uid, rid, split, sha, group):
        return {"schema": c.SCHEMA, "id": uid, "split": split, "messages": [{"role": "user", "content": "A uniquely authored source task " + uid}],
                "media": [{"path": "/sources/" + uid, "sha256": sha}],
                "provenance": {"registry_origin": {"rid": rid}, "group_id": group, "recovery_group_id": group,
                               "normalized_from": {"sha256": "a" * 64}}, "quality": {"blockers": []}}
    rows = [row("picture", "a:0:1", "train", "1" * 64, "group-picture"),
            row("song", "a:0:2", "train", "2" * 64, "group-song"),
            row("song-text", "a:0:3", "train", "3" * 64, "group-song")]
    if shared_holdout:
        rows.append(row("held", "a:0:4", "validation", "1" * 64, "held-group"))
    write("normalized", rows, True)
    write("eval_rows", [{"id": "external", "instruction": "Count triangles in the purple diagram", "response": "Seven."}], True)
    carve = {"schema": "fixture", "rows_sha256": "a" * 64, "assignment": {r["provenance"]["registry_origin"]["rid"]: "train" for r in rows}}
    carve["carve_sha256"] = hashlib.sha256(json.dumps(carve, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    write("carve", carve)
    write("corpus_manifest", {"outputs": {"normalized": {"sha256": c.file_sha256(paths["normalized"])}},
                              "sources": [{"sha256": "a" * 64}], "eval_sources": [{"sha256": c.file_sha256(paths["eval_rows"])}],
                              "carve_sha256": carve["carve_sha256"]})
    qimg = {"id": "picture", "corpus": {"rid": "a:0:1", "uid": "picture"}, "image": "/sources/picture", "sha256": "1" * 64,
            "prompt": "Render a red square", "negative": "blur"}
    write("qimg_items", [qimg], True)
    write("qimg_metadata", [{"rid": "a:0:1", "uid": "picture", "image": {"sha256": "1" * 64, "path": "/sources/picture"},
                             "prompt": "Render a blue square" if wrong_prompt else qimg["prompt"]}], True)
    ace = {"id": "song#hash", "rid": "a:0:2", "path": "/sources/song", "sha256": "2" * 64,
           "caption": "A song.", "lyrics": "An independently authored bounded fixture lyric", "held": False}
    write("ace_items", {"items": [ace], "n_items": 1})
    write("ace_metadata", [{"rid": "a:0:2", "uid": "song", "clip": {"sha256": "2" * 64, "path": "/sources/song"}}], True)
    write("ace_cache_manifest", {"corpus": {"file_sha256": c.file_sha256(paths["ace_items"])},
                                 "shards": [{"item_id": ace["id"], "meta": {"held": ace_held}}]})
    return paths, {key: c.file_sha256(p) for key, p in paths.items()}


def run(tmp_path, **kw):
    paths, pins = fixture(tmp_path, **kw)
    report = m.build_admission(**paths, expected_hashes=pins, output=tmp_path / "result")
    rows = list(c.iter_jsonl(tmp_path / "result/admission.jsonl"))
    return report, {r["kind"]: r for r in rows}


def test_matching_source_uid_asset_prompt_and_source_group_is_admitted(tmp_path):
    report, rows = run(tmp_path)
    assert rows["qimg"]["train_admitted"] and rows["ace"]["train_admitted"]
    assert rows["qimg"]["conditioning_sha256"] == c.digest({"prompt": "Render a red square", "negative": "blur"})
    assert report["eval_disjoint_scope"].startswith("train_admitted")


def test_validation_shared_asset_excludes_train_image_without_reassigning_split(tmp_path):
    _, rows = run(tmp_path, shared_holdout=True)
    assert not rows["qimg"]["train_admitted"]
    assert "connected_group_not_train_exclusive" in rows["qimg"]["blockers"]
    assert rows["qimg"]["split"] == "excluded"


def test_cache_held_bit_overrides_false_source_bit_and_excludes_text_group(tmp_path):
    report, rows = run(tmp_path, ace_held=True)
    assert not rows["ace"]["train_admitted"]
    assert "ace_captured_holdout" in rows["ace"]["blockers"]
    exclusions = list(c.iter_jsonl(tmp_path / "result/cross_route_training_exclusions.jsonl"))
    assert {r["id"] for r in exclusions} == {"song", "song-text"}
    assert report["outputs"]["cross_route_training_exclusions"]["rows"] == 2


def test_real_teacher_prompt_must_match_original_source_metadata(tmp_path):
    _, rows = run(tmp_path, wrong_prompt=True)
    assert not rows["qimg"]["train_admitted"]
    assert "qimg_asset_or_prompt_mismatch" in rows["qimg"]["blockers"]


def test_mutated_source_and_unbound_frozen_eval_fail_closed(tmp_path):
    paths, pins = fixture(tmp_path)
    paths["qimg_items"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source pin: qimg_items"):
        m.build_admission(**paths, expected_hashes=pins, output=tmp_path / "result")
