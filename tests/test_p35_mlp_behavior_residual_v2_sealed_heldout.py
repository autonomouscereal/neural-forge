from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROMPT_RELATIVE_PATH = Path(
    "experiments/p35_mlp_behavior_residual_v2_sealed_heldout_prompts_20260711.json"
)
MANIFEST_RELATIVE_PATH = Path(
    "experiments/p35_mlp_behavior_residual_v2_sealed_heldout_manifest_20260711.json"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _load_json(relative_path: str | Path) -> Any:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_v2_heldout_fixture_is_sealed_and_disjoint_from_all_v1_splits() -> None:
    manifest = _load_json(MANIFEST_RELATIVE_PATH)
    prompts = _load_json(PROMPT_RELATIVE_PATH)

    assert manifest["schema"] == (
        "neural-forge-p35-mlp-behavior-residual-v2-sealed-heldout-manifest-v1"
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

    prior_ids: set[str] = set()
    prior_prompts: set[str] = set()
    prior_prompt_hashes: set[str] = set()
    sources = manifest["prior_v1_exclusion"]["sources"]
    assert [source["split"] for source in sources] == [
        "fit",
        "calibration-selection",
        "heldout-evaluation",
    ]

    for source in sources:
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

        prior_ids.update(source["ids"])
        prior_prompts.update(source_prompts)
        prior_prompt_hashes.update(source["prompt_sha256s"])

    exclusion = manifest["prior_v1_exclusion"]
    assert exclusion["prior_unique_id_count"] == len(prior_ids) == 22
    assert exclusion["prior_unique_exact_prompt_count"] == len(prior_prompts) == 22
    assert new_ids.isdisjoint(prior_ids)
    assert new_prompts.isdisjoint(prior_prompts)
    assert new_prompt_hashes.isdisjoint(prior_prompt_hashes)

    disjointness = manifest["disjointness"]
    assert disjointness["new_fixture_id_overlap_count"] == 0
    assert disjointness["new_fixture_exact_prompt_overlap_count"] == 0
    assert disjointness["prior_v1_id_overlap_count"] == 0
    assert disjointness["prior_v1_exact_prompt_overlap_count"] == 0
    assert disjointness["all_new_ids_unique"] is True
    assert disjointness["all_new_exact_prompts_unique"] is True
    assert disjointness["zero_id_overlap_with_all_v1_splits"] is True
    assert disjointness["zero_exact_text_overlap_with_all_v1_splits"] is True

    seal = manifest["seal"]
    assert seal["created_without_model_execution"] is True
    assert seal["created_without_tokenizer_execution"] is True
    assert seal["future_model_outputs_must_not_be_added_to_this_manifest"] is True
