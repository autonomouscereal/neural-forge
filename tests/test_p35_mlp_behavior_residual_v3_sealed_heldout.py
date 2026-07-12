from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROMPT_RELATIVE_PATH = Path(
    "experiments/p35_mlp_behavior_residual_v3_sealed_heldout_prompts_20260711.json"
)
MANIFEST_RELATIVE_PATH = Path(
    "experiments/p35_mlp_behavior_residual_v3_sealed_heldout_manifest_20260711.json"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _load_json(relative_path: str | Path) -> Any:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_v3_heldout_fixture_is_sealed_and_disjoint_from_v1_and_v2() -> None:
    manifest = _load_json(MANIFEST_RELATIVE_PATH)
    prompts = _load_json(PROMPT_RELATIVE_PATH)

    assert manifest["schema"] == (
        "neural-forge-p35-mlp-behavior-residual-v3-sealed-heldout-manifest-v1"
    )
    assert manifest["sealed_before_model_access"] is True
    assert manifest["model_outputs_observed"] is False
    assert manifest["count"] == 8
    assert manifest["prompt_file"]["path"] == PROMPT_RELATIVE_PATH.as_posix()
    assert manifest["prompt_file"]["sha256"] == _sha256_bytes(
        (ROOT / PROMPT_RELATIVE_PATH).read_bytes()
    )

    entries = manifest["entries"]
    assert len(prompts) == len(entries) == manifest["count"]
    assert [entry["index"] for entry in entries] == list(range(manifest["count"]))
    assert [entry["prompt"] for entry in entries] == prompts
    assert all(
        set(entry) == {"index", "id", "prompt", "prompt_sha256"} for entry in entries
    )
    assert all(
        entry["prompt_sha256"] == _sha256_text(entry["prompt"]) for entry in entries
    )

    new_ids = {entry["id"] for entry in entries}
    new_prompts = set(prompts)
    new_prompt_hashes = {entry["prompt_sha256"] for entry in entries}
    assert len(new_ids) == len(new_prompts) == len(new_prompt_hashes) == 8

    v1_ids: set[str] = set()
    v1_prompts: set[str] = set()
    v1_prompt_hashes: set[str] = set()
    v1_sources = manifest["prior_v1_exclusion"]["sources"]
    assert [source["split"] for source in v1_sources] == [
        "fit",
        "calibration-selection",
        "heldout-evaluation",
    ]

    for source in v1_sources:
        prompt_file = source["prompt_file"]
        selection_manifest = source["selection_manifest"]
        assert prompt_file["sha256"] == _sha256_bytes(
            (ROOT / prompt_file["path"]).read_bytes()
        )
        assert selection_manifest["sha256"] == _sha256_bytes(
            (ROOT / selection_manifest["path"]).read_bytes()
        )

        source_prompts = _load_json(prompt_file["path"])
        selection = _load_json(selection_manifest["path"])
        if source["split"] == "fit":
            source_entries = selection["entries"]
        else:
            source_entries = selection["sets"][source["split"]]["entries"]

        assert source["count"] == len(source_prompts) == len(source_entries)
        assert source_prompts == [entry["prompt"] for entry in source_entries]
        assert source["ids"] == [entry["id"] for entry in source_entries]
        assert source["prompt_sha256s"] == [
            _sha256_text(prompt) for prompt in source_prompts
        ]
        assert source["prompt_sha256s"] == [
            entry["prompt_sha256"] for entry in source_entries
        ]

        v1_ids.update(source["ids"])
        v1_prompts.update(source_prompts)
        v1_prompt_hashes.update(source["prompt_sha256s"])

    v1_exclusion = manifest["prior_v1_exclusion"]
    assert v1_exclusion["prior_unique_id_count"] == len(v1_ids) == 22
    assert v1_exclusion["prior_unique_exact_prompt_count"] == len(v1_prompts) == 22

    v2_exclusion = manifest["prior_v2_exclusion"]
    v2_prompt_file = v2_exclusion["prompt_file"]
    v2_seal_file = v2_exclusion["seal_manifest"]
    assert v2_prompt_file["sha256"] == _sha256_bytes(
        (ROOT / v2_prompt_file["path"]).read_bytes()
    )
    assert v2_seal_file["sha256"] == _sha256_bytes(
        (ROOT / v2_seal_file["path"]).read_bytes()
    )

    v2_prompts_list = _load_json(v2_prompt_file["path"])
    v2_seal = _load_json(v2_seal_file["path"])
    v2_entries = v2_seal["entries"]
    assert v2_seal["sealed_before_model_access"] is True
    assert v2_seal["model_outputs_observed"] is False
    assert v2_exclusion["v2_sealed_before_model_access"] is True
    assert v2_exclusion["v2_model_outputs_observed"] is False
    assert v2_exclusion["count"] == len(v2_prompts_list) == len(v2_entries) == 8
    assert v2_prompts_list == [entry["prompt"] for entry in v2_entries]
    assert v2_exclusion["ids"] == [entry["id"] for entry in v2_entries]
    assert v2_exclusion["prompt_sha256s"] == [
        _sha256_text(prompt) for prompt in v2_prompts_list
    ]
    assert v2_exclusion["prompt_sha256s"] == [
        entry["prompt_sha256"] for entry in v2_entries
    ]

    v2_ids = set(v2_exclusion["ids"])
    v2_prompts = set(v2_prompts_list)
    v2_prompt_hashes = set(v2_exclusion["prompt_sha256s"])
    assert len(v2_ids) == len(v2_prompts) == len(v2_prompt_hashes) == 8
    assert v2_ids.isdisjoint(v1_ids)
    assert v2_prompts.isdisjoint(v1_prompts)
    assert v2_prompt_hashes.isdisjoint(v1_prompt_hashes)

    prior_ids = v1_ids | v2_ids
    prior_prompts = v1_prompts | v2_prompts
    prior_prompt_hashes = v1_prompt_hashes | v2_prompt_hashes
    prior_all = manifest["prior_all_exclusion"]
    assert prior_all["prior_unique_id_count"] == len(prior_ids) == 30
    assert prior_all["prior_unique_exact_prompt_count"] == len(prior_prompts) == 30
    assert new_ids.isdisjoint(prior_ids)
    assert new_prompts.isdisjoint(prior_prompts)
    assert new_prompt_hashes.isdisjoint(prior_prompt_hashes)

    disjointness = manifest["disjointness"]
    for field in (
        "new_fixture_id_overlap_count",
        "new_fixture_exact_prompt_overlap_count",
        "prior_v1_id_overlap_count",
        "prior_v1_exact_prompt_overlap_count",
        "prior_v2_id_overlap_count",
        "prior_v2_exact_prompt_overlap_count",
        "prior_all_id_overlap_count",
        "prior_all_exact_prompt_overlap_count",
    ):
        assert disjointness[field] == 0
    assert disjointness["all_new_ids_unique"] is True
    assert disjointness["all_new_exact_prompts_unique"] is True
    assert disjointness["zero_id_overlap_with_all_prior_splits"] is True
    assert disjointness["zero_exact_text_overlap_with_all_prior_splits"] is True

    seal = manifest["seal"]
    assert seal["created_without_model_execution"] is True
    assert seal["created_without_tokenizer_execution"] is True
    assert seal["future_model_outputs_must_not_be_added_to_this_manifest"] is True
