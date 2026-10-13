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


def followup_fixture(tmp_path, *, qvl_held=False, lm_held=False, ref_shared=False, wrong_lm_sha=False):
    paths, pins = fixture(tmp_path)
    prior = m.build_admission(**paths, expected_hashes=pins, output=tmp_path / "prior")
    paths["previous_admission"] = tmp_path / "prior/manifest.json"
    def body(kind, items, name):
        content = {"kind": kind, "items": items, "n_items": len(items)}
        content["corpus_sha256"] = c.digest({"schema": "omni-campaign.teacher_corpus.v1", "kind": kind, "items": items})
        path = tmp_path / (name + ".json")
        path.write_bytes(c.canonical(content))
        paths[name] = path
        return content
    spec = {"caption": "A song.", "lyrics": "Genuine authored test lyrics", "seed": 11, "seconds": 60.0}
    lm = {"id": "a:0:2#2222#lm0", "sha256": "f" * 64 if wrong_lm_sha else c.digest(spec),
          **spec, "source_clip_sha256": "2" * 64, "held": lm_held}
    lm_body = body("ace_prompts", [lm], "ace_lm_items")
    qvl = {"id": "a:0:1#1111", "rid": "a:0:1", "sha256": "1" * 64, "path": "/sources/picture", "route": "image", "held": False}
    qvl_body = body("qvl_images", [qvl], "qvl_items")
    ref = {"id": "diagnostic-ref", "sha256": "1" * 64 if ref_shared else "4" * 64, "path": "/diagnostic/ref", "route": "dir", "held": False}
    ref_body = body("qvl_images", [ref], "qvl_ref_items")
    paths["vit_metadata"] = tmp_path / "vit.jsonl"
    paths["vit_metadata"].write_bytes(c.canonical({"rid": "a:0:1", "image": {"sha256": "1" * 64, "path": "/sources/picture"}}) + b"\n")
    def record(name, content, item, held, sealed):
        r = {"source_corpus_file": str(paths[name]), "source_corpus_sha256": c.file_sha256(paths[name]),
             "sidecar_count": 1, "first_sidecar": {"item_id": item["id"], "item": item, "corpus_sha256": content["corpus_sha256"]}}
        if sealed:
            r["manifest"] = {"schema": "omni-campaign.teacher_cache.v2", "finite": True,
                "corpus": {"file_sha256": c.file_sha256(paths[name]), "sha256": content["corpus_sha256"]},
                "shards": [{"item_id": item["id"], "meta": {"held": held}}]}
            r["manifest_file_sha256"] = c.digest(r["manifest"])
        return r
    metadata = {"exit_code": 0, "data": {"records": [record("ace_lm_items", lm_body, lm, lm_held, False),
        record("qvl_items", qvl_body, qvl, qvl_held, True), record("qvl_ref_items", ref_body, ref, False, True)]}}
    paths["followup_metadata"] = tmp_path / "followup.json"
    paths["followup_metadata"].write_bytes(c.canonical(metadata))
    return paths, {name: c.file_sha256(path) for name, path in paths.items()}, prior


def test_partial_lm_source_eligible_not_cache_ready_and_does_not_exclude_text(tmp_path):
    paths, pins, prior = followup_fixture(tmp_path)
    report = m.build_admission(**paths, expected_hashes=pins, output=tmp_path / "followup")
    rows = {r["kind"]: r for r in c.iter_jsonl(tmp_path / "followup/admission.jsonl")}
    assert rows["ace_lm"]["source_train_eligible"] and not rows["ace_lm"]["train_admitted"] and not rows["ace_lm"]["cache_ready"]
    assert rows["ace"]["train_admitted"] and rows["qvl"]["train_admitted"]
    assert rows["ace_lm"]["asset_sha256"] == "2" * 64
    assert rows["ace_lm"]["teacher_item_sha256"] != rows["ace_lm"]["asset_sha256"]
    assert not rows["qvl_ref"]["train_admitted"]
    assert report["outputs"]["cross_route_training_exclusions"]["rows"] == 0
    assert report["previous_admission"]["manifest_sha256"] == prior["manifest_sha256"]


def test_new_teacher_held_flags_propagate_cross_route_to_text_and_prior_admission(tmp_path):
    paths, pins, _ = followup_fixture(tmp_path, qvl_held=True, lm_held=True)
    report = m.build_admission(**paths, expected_hashes=pins, output=tmp_path / "followup")
    rows = {r["kind"]: r for r in c.iter_jsonl(tmp_path / "followup/admission.jsonl")}
    assert not any(r["train_admitted"] for r in rows.values())
    exclusions = list(c.iter_jsonl(tmp_path / "followup/cross_route_training_exclusions.jsonl"))
    assert {r["id"] for r in exclusions} == {"song", "song-text", "picture"}
    assert "qvl_captured_holdout" in rows["qimg"]["blockers"]
    assert "ace_lm_captured_holdout" in rows["ace"]["blockers"]


def test_diagnostic_reference_shared_asset_stays_excluded_across_routes(tmp_path):
    paths, pins, _ = followup_fixture(tmp_path, ref_shared=True)
    m.build_admission(**paths, expected_hashes=pins, output=tmp_path / "followup")
    rows = {r["kind"]: r for r in c.iter_jsonl(tmp_path / "followup/admission.jsonl")}
    assert not rows["qvl"]["train_admitted"] and not rows["qimg"]["train_admitted"]
    assert "qvl_ref32_diagnostic_never_train" in rows["qvl"]["blockers"]


def test_prompt_payload_sha_mismatch_and_unknown_cache_held_fail_closed(tmp_path):
    paths, pins, _ = followup_fixture(tmp_path, wrong_lm_sha=True)
    m.build_admission(**paths, expected_hashes=pins, output=tmp_path / "followup")
    rows = {r["kind"]: r for r in c.iter_jsonl(tmp_path / "followup/admission.jsonl")}
    assert "ace_lm_teacher_prompt_payload_mismatch" in rows["ace_lm"]["blockers"]
    captured = json.loads(paths["followup_metadata"].read_text())
    record = captured["data"]["records"][1]
    record["manifest"]["shards"][0]["meta"].pop("held")
    with pytest.raises(ValueError, match="held flag"):
        m.followup_inventory(paths["qvl_items"], record)


def test_multiple_images_per_source_rid_require_exact_asset_join(tmp_path):
    paths, pins, _ = followup_fixture(tmp_path)
    with paths["vit_metadata"].open("ab") as stream:
        stream.write(c.canonical({"rid": "a:0:1", "image": {"sha256": "9" * 64, "path": "/sources/other-image"}}) + b"\n")
    pins["vit_metadata"] = c.file_sha256(paths["vit_metadata"])
    m.build_admission(**paths, expected_hashes=pins, output=tmp_path / "followup")
    qvl = next(r for r in c.iter_jsonl(tmp_path / "followup/admission.jsonl") if r["kind"] == "qvl")
    assert qvl["train_admitted"] and qvl["asset_sha256"] == "1" * 64
