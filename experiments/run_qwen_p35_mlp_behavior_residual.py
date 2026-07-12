#!/usr/bin/env python3
"""Train one behavior-only residual SwiGLU on the preserved MLP-p35 child.

The first bounded target is logical layer 1, where the exact preserved p35 MLP
artifact redirects 2,122 of its 9,728 neuron assignments.  A width-64 parallel
SwiGLU starts at exact-zero output and only its gate/up/down tensors train from
fixed parent output actions.  Fit, calibration-selection, and heldout prompts
are disjoint.  V3 requires Pareto non-regression in calibration free-rollout
exact matches, token matches, and mean prefix agreement before one-step action
ordering, while fit-only current-student prefixes refresh after updates
8/16/24 without resetting manual Adam state.  V1/V2 inspected heldouts are
development data, so a new V3 seal is verified before model access; its model
forwards begin only after the winner is fully reproduced and locked.  Both
runtimes are restored, no checkpoint is written, and JSON is the only CLI
artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import run_qwen_p35_functional_residual_adapter as action_runner
import run_qwen_residual_attention_probe as codebook_loader
from neural_forge.hf_behavior_residual_swiglu import (
    BehaviorResidualSwiGLUAccounting,
    BehaviorResidualSwiGLUFitRefresh,
    MLPCodebookRedirectAudit,
    _assert_guard,
    _parameter_guard,
    audit_mlp_codebook_redirects,
    install_behavior_residual_swiglu,
    train_behavior_residual_swiglu_disjoint,
)
from neural_forge.hf_functional_decode_trace import FunctionalDecodeLimits
from neural_forge.hf_functional_residual_adapter import (
    FunctionalActionBatch,
    FunctionalActionEvaluation,
    evaluate_functional_action_batches,
)
from neural_forge.hf_neuron_codebook import NeuronCodebookRegistry
from neural_forge.hf_qwen_codebook_model import qwen3_codebook_parameter_report


RUN_SCHEMA = "neural-forge-qwen-p35-mlp-behavior-residual-swiglu-run-v3"
EXACT_LAYER_INDICES = (1,)
EXACT_RESIDUAL_WIDTH = 64
EXACT_HORIZON = 4
EXACT_STEPS = 32
EXACT_MICRO_BATCH_SIZE = 2
EXACT_CALIBRATION_STEPS = (0, 8, 16, 32)
EXACT_FIT_REFRESH_STEPS = (8, 16, 24)
EXACT_GATE_UP_LEARNING_RATE = 1e-3
EXACT_DOWN_LEARNING_RATE = 2e-3
SOURCE_LINEAGE_FILES = (
    "experiments/run_qwen_p35_mlp_behavior_residual.py",
    "src/neural_forge/hf_behavior_residual_swiglu.py",
    "experiments/run_qwen_p35_functional_residual_adapter.py",
    "src/neural_forge/hf_functional_decode_trace.py",
    "src/neural_forge/hf_functional_residual_adapter.py",
)
EXACT_ASSIGNMENT_SHA256 = (
    "e4b2399b55b36039634c34a3ddcbf379380e9f27dfd407847dcd0b9452a87583"
)
EXACT_PROTOTYPE_SOURCE_SHA256 = (
    "3aabaed9bb8f7e71d8c76be7cce91110324102c6608344de75f9834dc269353a"
)
EXACT_P35_MANIFEST_SHA256 = (
    "28332a650a936ba3025834d0118c708aa0632484ed89a6da66c08e9daab547a9"
)
EXACT_P35_MODEL_SHA256 = (
    "43f5e77ca079c9a8de5692bd32b02ecb57e87c4bbc95260309ec33d40e81fa91"
)
EXACT_P35_STATE_KEY_SHA256 = (
    "16c5677db66d683aa554d7c733c7ebb9a285fec5151cb1b921bb44459c56d89b"
)
EXACT_TOKENIZER_CONTRACT_SHA256 = (
    "aabb2c58fefaba3386507654bcec6d514d84f8b70c5b4f160195abc53207afc1"
)
EXACT_TOKENIZER_VOCAB_SHA256 = (
    "c96fa22dee2dd9c98c4f0c73ea423e21907bc96abda81fa5c6b2e24a670c66bf"
)
EXACT_PARENT_TOKEN_ROWS_SHA256 = (
    "abc3afb984abdb07b148a41521ccd2b412625719d562f4c78dd3d6641ede9dca"
)
EXACT_PARENT_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
EXACT_PARENT_SHARD_CONTENT_HASHES = (
    "4c807e2503d68ae373d508689d00a41f4b33f33c2536da97ab81a20caddc1241",
    "f4707585548b2fc75a6b1d732e8465c62040a8699903c32850781beeb9b27826",
    "c7b1aa8fb672de2e00423c99876926022e50b18d4f0d140670788510a27f9965",
)
REJECTED_V1_REPORT_FILE_SHA256 = (
    "ec6402bd793578487a7fe20b167096c1cbc46382921c792e7ea5a324b8c43f8e"
)
REJECTED_V1_INTERNAL_REPORT_SHA256 = (
    "b422c1b5cd87d00781011de25bf57e2b654a2236eb57cafa82f83f0f04d063d3"
)
REJECTED_V2_REPORT_FILE_SHA256 = (
    "1ef820c5e889f0678e8240adc9fd8e9c7e3cd14a09ed1d4c032625293f176443"
)
REJECTED_V2_INTERNAL_REPORT_SHA256 = (
    "5784d8e676432ae5ddccc4efe10d0b2c14b2e12609b1f6f8e24a9ad11d7bb32d"
)
REJECTED_V2_FIT_BATCH_MANIFEST_SHA256 = (
    "2cbf2dd5efe5ad4af2247637e4096b34b0ded53151208b453830d15e89dbb6af"
)
EXACT_REDIRECT_COUNTS = (
    264,
    2122,
    897,
    1264,
    165,
    133,
    105,
    112,
    95,
    177,
    126,
    134,
    129,
    112,
    188,
    185,
    196,
    187,
    191,
    270,
    247,
    284,
    214,
    206,
    147,
    128,
    111,
    136,
    137,
    106,
    74,
    67,
    92,
    149,
    239,
    339,
)

DEFAULT_FIT_PROMPTS = action_runner.DEFAULT_FIT_PROMPTS
DEFAULT_CALIBRATION_PROMPTS = action_runner.DEFAULT_CALIBRATION_PROMPTS
EXPERIMENTS = Path(__file__).resolve().parent
V1_DEVELOPMENT_HELDOUT_PROMPTS = action_runner.DEFAULT_HELDOUT_PROMPTS
V2_DEVELOPMENT_HELDOUT_PROMPTS = (
    EXPERIMENTS / "p35_mlp_behavior_residual_v2_sealed_heldout_prompts_20260711.json"
)
V2_SEALED_HELDOUT_MANIFEST = (
    EXPERIMENTS / "p35_mlp_behavior_residual_v2_sealed_heldout_manifest_20260711.json"
)
V2_SEALED_HELDOUT_FIXTURE_SHA256 = (
    "ed8e7bfe009e2ba75198e2f3c81ca5a166bf29b5ddb0743e33840f17b89dc2cb"
)
V2_SEALED_HELDOUT_MANIFEST_SHA256 = (
    "0c94a78d94f2b6cf609eab93bdb52af3f0e1d4b1d19fc9dd3d2e831aaaa19e18"
)
DEFAULT_HELDOUT_PROMPTS = (
    EXPERIMENTS / "p35_mlp_behavior_residual_v3_sealed_heldout_prompts_20260711.json"
)
DEFAULT_SEALED_HELDOUT_MANIFEST = (
    EXPERIMENTS / "p35_mlp_behavior_residual_v3_sealed_heldout_manifest_20260711.json"
)
EXACT_SEALED_HELDOUT_FIXTURE_SHA256 = (
    "edcb437d3f331e65ab4f670a3cc20352709162dc31982adc82640e51ac509f3b"
)
EXACT_SEALED_HELDOUT_MANIFEST_SHA256 = (
    "d197bab46ed02c847ebcc5a0a93c95b1dce3b7298b8871835464b8509103754b"
)
DEFAULT_FIT_MANIFEST = action_runner.DEFAULT_FIT_MANIFEST
DEFAULT_AUDITED_MANIFEST = action_runner.DEFAULT_AUDITED_MANIFEST
DEFAULT_TRANSITION_INVENTORY = action_runner.DEFAULT_TRANSITION_INVENTORY
EXPECTED_SPLIT_COUNTS = action_runner.EXPECTED_SPLIT_COUNTS


class QwenP35MLPBehaviorResidualError(RuntimeError):
    """The p35 MLP behavior-residual experiment contract failed."""


def source_file_lineage(project_root: Path | None = None) -> dict[str, Any]:
    """Content-address every Python source file that defines the V3 run."""

    root = (
        Path(__file__).resolve().parents[1]
        if project_root is None
        else Path(project_root).resolve()
    )
    files: dict[str, str] = {}
    for relative_path in SOURCE_LINEAGE_FILES:
        path = root / relative_path
        if not path.is_file():
            raise QwenP35MLPBehaviorResidualError(
                f"Missing V3 source-lineage file: {relative_path}."
            )
        files[relative_path] = codebook_loader._file_sha256(path)
    payload: dict[str, Any] = {
        "schema": "neural-forge-p35-mlp-behavior-residual-source-lineage-v1",
        "files_sha256": files,
        "content_addressed_worktree_sources": True,
    }
    payload["source_lineage_sha256"] = codebook_loader._sha256(payload)
    return payload


@dataclass(frozen=True, slots=True)
class VerifiedMLPV2PromptSplits:
    fit: tuple[str, ...]
    calibration: tuple[str, ...]
    heldout: tuple[str, ...]
    v1_development_heldout: tuple[str, ...]
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedMLPV3PromptSplits:
    fit: tuple[str, ...]
    calibration: tuple[str, ...]
    heldout: tuple[str, ...]
    v1_development_heldout: tuple[str, ...]
    v2_development_heldout: tuple[str, ...]
    provenance: Mapping[str, Any]


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_pinned_json(path: Path, *, expected_sha256: str, name: str) -> Any:
    resolved = path.resolve()
    if not resolved.is_file():
        raise QwenP35MLPBehaviorResidualError(f"{name} does not exist: {resolved}")
    observed = codebook_loader._file_sha256(resolved)
    if observed != expected_sha256:
        raise QwenP35MLPBehaviorResidualError(
            f"{name} SHA-256 changed: expected {expected_sha256}, got {observed}."
        )
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QwenP35MLPBehaviorResidualError(
            f"{name} is not valid UTF-8 JSON."
        ) from error


def load_verified_v2_prompt_splits(
    fit_path: Path,
    calibration_path: Path,
    heldout_path: Path,
    *,
    sealed_heldout_manifest_path: Path = V2_SEALED_HELDOUT_MANIFEST,
    fit_manifest_path: Path = DEFAULT_FIT_MANIFEST,
    audited_manifest_path: Path = DEFAULT_AUDITED_MANIFEST,
    transition_inventory_path: Path = DEFAULT_TRANSITION_INVENTORY,
) -> VerifiedMLPV2PromptSplits:
    """Verify V1 development rows and a fresh V2 seal before model access."""

    legacy = action_runner.load_verified_prompt_splits(
        fit_path,
        calibration_path,
        V1_DEVELOPMENT_HELDOUT_PROMPTS,
        fit_manifest_path=fit_manifest_path,
        audited_manifest_path=audited_manifest_path,
        transition_inventory_path=transition_inventory_path,
    )
    fresh_payload = _read_pinned_json(
        heldout_path,
        expected_sha256=V2_SEALED_HELDOUT_FIXTURE_SHA256,
        name="fresh v2 heldout fixture",
    )
    manifest = _read_pinned_json(
        sealed_heldout_manifest_path,
        expected_sha256=V2_SEALED_HELDOUT_MANIFEST_SHA256,
        name="fresh v2 heldout seal manifest",
    )
    if (
        not isinstance(fresh_payload, list)
        or len(fresh_payload) != EXPECTED_SPLIT_COUNTS["heldout-evaluation"]
        or any(not isinstance(value, str) or not value for value in fresh_payload)
        or len(set(fresh_payload)) != len(fresh_payload)
        or not isinstance(manifest, Mapping)
    ):
        raise QwenP35MLPBehaviorResidualError(
            "Fresh v2 heldout fixture or seal has an invalid shape."
        )
    try:
        entries = manifest["entries"]
        prompt_file = manifest["prompt_file"]
    except (KeyError, TypeError) as error:
        raise QwenP35MLPBehaviorResidualError(
            "Fresh v2 heldout seal lacks prompt entries or file binding."
        ) from error
    if (
        manifest.get("schema")
        != "neural-forge-p35-mlp-behavior-residual-v2-sealed-heldout-manifest-v1"
        or manifest.get("sealed_before_model_access") is not True
        or manifest.get("model_outputs_observed") is not False
        or manifest.get("count") != len(fresh_payload)
        or not isinstance(entries, list)
        or len(entries) != len(fresh_payload)
        or not isinstance(prompt_file, Mapping)
        or prompt_file.get("sha256") != V2_SEALED_HELDOUT_FIXTURE_SHA256
        or Path(str(prompt_file.get("path"))).name != heldout_path.resolve().name
    ):
        raise QwenP35MLPBehaviorResidualError(
            "Fresh v2 heldout seal is not a pre-model immutable eight-row seal."
        )
    fresh_ids: list[str] = []
    for index, (entry, prompt) in enumerate(zip(entries, fresh_payload, strict=True)):
        if (
            not isinstance(entry, Mapping)
            or entry.get("index") != index
            or not isinstance(entry.get("id"), str)
            or not entry["id"]
            or entry.get("prompt") != prompt
            or entry.get("prompt_sha256") != _text_sha256(prompt)
        ):
            raise QwenP35MLPBehaviorResidualError(
                "Fresh v2 heldout entry differs from its sealed ID/text/hash."
            )
        fresh_ids.append(entry["id"])
    if len(set(fresh_ids)) != len(fresh_ids):
        raise QwenP35MLPBehaviorResidualError("Fresh v2 heldout IDs are not unique.")

    legacy_manifest = _read_pinned_json(
        audited_manifest_path,
        expected_sha256=action_runner.behavior_runner.EXPECTED_FILE_SHA256[
            "audited-manifest"
        ],
        name="V1 audited prompt manifest",
    )
    try:
        legacy_audited_entries = [
            *legacy_manifest["sets"]["calibration-selection"]["entries"],
            *legacy_manifest["sets"]["heldout-evaluation"]["entries"],
        ]
        legacy_ids = {
            *legacy.provenance["fit_case_ids"],
            *(str(value["id"]) for value in legacy_audited_entries),
        }
    except (KeyError, TypeError) as error:
        raise QwenP35MLPBehaviorResidualError(
            "Verified V1 manifests cannot expose exclusion IDs."
        ) from error
    legacy_prompts = {
        *legacy.fit,
        *legacy.calibration,
        *legacy.heldout,
    }
    try:
        prior_sources = manifest["prior_v1_exclusion"]["sources"]
        sealed_prior_ids = {
            str(row_id) for source in prior_sources for row_id in source["ids"]
        }
        sealed_prior_prompt_hashes = {
            str(prompt_hash)
            for source in prior_sources
            for prompt_hash in source["prompt_sha256s"]
        }
    except (KeyError, TypeError) as error:
        raise QwenP35MLPBehaviorResidualError(
            "Fresh v2 seal cannot expose all prior V1 exclusion rows."
        ) from error
    if sealed_prior_ids != legacy_ids or sealed_prior_prompt_hashes != {
        _text_sha256(value) for value in legacy_prompts
    }:
        raise QwenP35MLPBehaviorResidualError(
            "Fresh v2 seal prior-row exclusion differs from verified V1 data."
        )

    def normalize(value: str) -> str:
        return " ".join(value.split()).casefold()

    exact_overlap = legacy_prompts & set(fresh_payload)
    normalized_overlap = {normalize(value) for value in legacy_prompts} & {
        normalize(value) for value in fresh_payload
    }
    id_overlap = legacy_ids & set(fresh_ids)
    disjointness = manifest.get("disjointness")
    if (
        exact_overlap
        or normalized_overlap
        or id_overlap
        or not isinstance(disjointness, Mapping)
        or disjointness.get("prior_v1_id_overlap_count") != 0
        or disjointness.get("prior_v1_exact_prompt_overlap_count") != 0
    ):
        raise QwenP35MLPBehaviorResidualError(
            "Fresh v2 heldout IDs/text overlap V1 fit/calibration/development data."
        )
    provenance = {
        **dict(legacy.provenance),
        "fresh_v2_heldout_fixture_sha256": V2_SEALED_HELDOUT_FIXTURE_SHA256,
        "fresh_v2_heldout_manifest_sha256": V2_SEALED_HELDOUT_MANIFEST_SHA256,
        "fresh_manifest_verified_before_model_access": True,
        "fresh_ids_and_text_disjoint_from_all_v1_splits": True,
        "v1_heldout_reclassified_as_development": True,
        "v1_development_heldout_prompt_sha256s": [
            _text_sha256(value) for value in legacy.heldout
        ],
        "fresh_v2_heldout_ids": fresh_ids,
        "fresh_v2_heldout_prompt_sha256s": [
            _text_sha256(value) for value in fresh_payload
        ],
        "fresh_v2_cross_run_overlap": {
            "id_count": len(id_overlap),
            "exact_prompt_count": len(exact_overlap),
            "normalized_prompt_count": len(normalized_overlap),
        },
    }
    provenance["prompt_text_sha256s"] = {
        **dict(legacy.provenance["prompt_text_sha256s"]),
        "heldout-evaluation": provenance["fresh_v2_heldout_prompt_sha256s"],
    }
    provenance["file_sha256"] = {
        **dict(legacy.provenance["file_sha256"]),
        "v1-development-heldout": legacy.provenance["file_sha256"][
            "heldout-evaluation"
        ],
        "heldout-evaluation": V2_SEALED_HELDOUT_FIXTURE_SHA256,
        "sealed-heldout-manifest": V2_SEALED_HELDOUT_MANIFEST_SHA256,
    }
    provenance["provenance_sha256"] = codebook_loader._sha256(
        {key: value for key, value in provenance.items() if key != "provenance_sha256"}
    )
    return VerifiedMLPV2PromptSplits(
        legacy.fit,
        legacy.calibration,
        tuple(fresh_payload),
        legacy.heldout,
        provenance,
    )


def load_verified_v3_prompt_splits(
    fit_path: Path,
    calibration_path: Path,
    heldout_path: Path,
    *,
    sealed_heldout_manifest_path: Path = DEFAULT_SEALED_HELDOUT_MANIFEST,
    fit_manifest_path: Path = DEFAULT_FIT_MANIFEST,
    audited_manifest_path: Path = DEFAULT_AUDITED_MANIFEST,
    transition_inventory_path: Path = DEFAULT_TRANSITION_INVENTORY,
) -> VerifiedMLPV3PromptSplits:
    """Verify a fresh V3 seal against every V1 and inspected V2 row."""

    prior = load_verified_v2_prompt_splits(
        fit_path,
        calibration_path,
        V2_DEVELOPMENT_HELDOUT_PROMPTS,
        sealed_heldout_manifest_path=V2_SEALED_HELDOUT_MANIFEST,
        fit_manifest_path=fit_manifest_path,
        audited_manifest_path=audited_manifest_path,
        transition_inventory_path=transition_inventory_path,
    )
    fresh_payload = _read_pinned_json(
        heldout_path,
        expected_sha256=EXACT_SEALED_HELDOUT_FIXTURE_SHA256,
        name="fresh v3 heldout fixture",
    )
    manifest = _read_pinned_json(
        sealed_heldout_manifest_path,
        expected_sha256=EXACT_SEALED_HELDOUT_MANIFEST_SHA256,
        name="fresh v3 heldout seal manifest",
    )
    if (
        not isinstance(fresh_payload, list)
        or len(fresh_payload) != EXPECTED_SPLIT_COUNTS["heldout-evaluation"]
        or any(not isinstance(value, str) or not value for value in fresh_payload)
        or len(set(fresh_payload)) != len(fresh_payload)
        or not isinstance(manifest, Mapping)
    ):
        raise QwenP35MLPBehaviorResidualError(
            "Fresh v3 heldout fixture or seal has an invalid shape."
        )
    try:
        entries = manifest["entries"]
        prompt_file = manifest["prompt_file"]
        prior_v2 = manifest["prior_v2_exclusion"]
        prior_all = manifest["prior_all_exclusion"]
    except (KeyError, TypeError) as error:
        raise QwenP35MLPBehaviorResidualError(
            "Fresh v3 seal lacks prompt or prior-exclusion bindings."
        ) from error
    if (
        manifest.get("schema")
        != "neural-forge-p35-mlp-behavior-residual-v3-sealed-heldout-manifest-v1"
        or manifest.get("sealed_before_model_access") is not True
        or manifest.get("model_outputs_observed") is not False
        or manifest.get("count") != len(fresh_payload)
        or not isinstance(entries, list)
        or len(entries) != len(fresh_payload)
        or not isinstance(prompt_file, Mapping)
        or prompt_file.get("sha256") != EXACT_SEALED_HELDOUT_FIXTURE_SHA256
        or Path(str(prompt_file.get("path"))).name != heldout_path.resolve().name
        or not isinstance(prior_v2, Mapping)
        or prior_v2.get("prompt_file", {}).get("sha256")
        != V2_SEALED_HELDOUT_FIXTURE_SHA256
        or prior_v2.get("seal_manifest", {}).get("sha256")
        != V2_SEALED_HELDOUT_MANIFEST_SHA256
        or prior_all.get("prior_unique_id_count") != 30
        or prior_all.get("prior_unique_exact_prompt_count") != 30
    ):
        raise QwenP35MLPBehaviorResidualError(
            "Fresh v3 heldout seal is not the pinned pre-model eight-row seal."
        )
    fresh_ids: list[str] = []
    for index, (entry, prompt) in enumerate(zip(entries, fresh_payload, strict=True)):
        if (
            not isinstance(entry, Mapping)
            or entry.get("index") != index
            or not isinstance(entry.get("id"), str)
            or not entry["id"]
            or entry.get("prompt") != prompt
            or entry.get("prompt_sha256") != _text_sha256(prompt)
        ):
            raise QwenP35MLPBehaviorResidualError(
                "Fresh v3 heldout entry differs from its sealed ID/text/hash."
            )
        fresh_ids.append(entry["id"])
    if len(set(fresh_ids)) != len(fresh_ids):
        raise QwenP35MLPBehaviorResidualError("Fresh v3 heldout IDs are not unique.")

    v2_manifest = _read_pinned_json(
        V2_SEALED_HELDOUT_MANIFEST,
        expected_sha256=V2_SEALED_HELDOUT_MANIFEST_SHA256,
        name="sealed v2 heldout manifest",
    )
    if manifest.get("prior_v1_exclusion") != v2_manifest.get("prior_v1_exclusion"):
        raise QwenP35MLPBehaviorResidualError(
            "Fresh v3 seal changed the verified V1 exclusion set."
        )
    v2_ids = set(prior.provenance["fresh_v2_heldout_ids"])
    if set(prior_v2.get("ids", ())) != v2_ids or set(
        prior_v2.get("prompt_sha256s", ())
    ) != {_text_sha256(value) for value in prior.heldout}:
        raise QwenP35MLPBehaviorResidualError(
            "Fresh v3 seal changed the verified V2 exclusion set."
        )
    prior_prompts = {
        *prior.fit,
        *prior.calibration,
        *prior.v1_development_heldout,
        *prior.heldout,
    }

    def normalize(value: str) -> str:
        return " ".join(value.split()).casefold()

    exact_overlap = prior_prompts & set(fresh_payload)
    normalized_overlap = {normalize(value) for value in prior_prompts} & {
        normalize(value) for value in fresh_payload
    }
    v1_ids = {
        str(row_id)
        for source in manifest["prior_v1_exclusion"]["sources"]
        for row_id in source["ids"]
    }
    id_overlap = (v1_ids | v2_ids) & set(fresh_ids)
    disjointness = manifest.get("disjointness")
    if (
        exact_overlap
        or normalized_overlap
        or id_overlap
        or not isinstance(disjointness, Mapping)
        or disjointness.get("prior_all_id_overlap_count") != 0
        or disjointness.get("prior_all_exact_prompt_overlap_count") != 0
        or disjointness.get("zero_id_overlap_with_all_prior_splits") is not True
        or disjointness.get("zero_exact_text_overlap_with_all_prior_splits") is not True
    ):
        raise QwenP35MLPBehaviorResidualError(
            "Fresh v3 heldout IDs/text overlap V1 or inspected V2 data."
        )
    provenance = {
        **dict(prior.provenance),
        "fresh_v3_heldout_fixture_sha256": EXACT_SEALED_HELDOUT_FIXTURE_SHA256,
        "fresh_v3_heldout_manifest_sha256": EXACT_SEALED_HELDOUT_MANIFEST_SHA256,
        "fresh_manifest_verified_before_model_access": True,
        "fresh_ids_and_text_disjoint_from_all_prior_splits": True,
        "v2_heldout_reclassified_as_development": True,
        "v2_development_heldout_prompt_sha256s": [
            _text_sha256(value) for value in prior.heldout
        ],
        "fresh_v3_heldout_ids": fresh_ids,
        "fresh_v3_heldout_prompt_sha256s": [
            _text_sha256(value) for value in fresh_payload
        ],
        "fresh_v3_cross_run_overlap": {
            "id_count": len(id_overlap),
            "exact_prompt_count": len(exact_overlap),
            "normalized_prompt_count": len(normalized_overlap),
        },
    }
    provenance["prompt_text_sha256s"] = {
        **dict(prior.provenance["prompt_text_sha256s"]),
        "heldout-evaluation": provenance["fresh_v3_heldout_prompt_sha256s"],
    }
    provenance["file_sha256"] = {
        **dict(prior.provenance["file_sha256"]),
        "v2-development-heldout": V2_SEALED_HELDOUT_FIXTURE_SHA256,
        "heldout-evaluation": EXACT_SEALED_HELDOUT_FIXTURE_SHA256,
        "sealed-heldout-manifest": EXACT_SEALED_HELDOUT_MANIFEST_SHA256,
    }
    provenance["provenance_sha256"] = codebook_loader._sha256(
        {key: value for key, value in provenance.items() if key != "provenance_sha256"}
    )
    return VerifiedMLPV3PromptSplits(
        prior.fit,
        prior.calibration,
        tuple(fresh_payload),
        prior.v1_development_heldout,
        prior.heldout,
        provenance,
    )


class _ParentTokenRowView(nn.Module):
    """Expose raw-parent inputs and logits in the compact child's token rows."""

    def __init__(self, parent: nn.Module, parent_token_rows: Tensor) -> None:
        super().__init__()
        if not isinstance(parent, nn.Module):
            raise QwenP35MLPBehaviorResidualError("parent must be an nn.Module.")
        rows = torch.as_tensor(parent_token_rows).detach().cpu().to(torch.int64)
        output = parent.get_output_embeddings()
        if (
            rows.ndim != 1
            or int(rows.numel()) < 2
            or int(rows.min()) < 0
            or int(rows.max()) >= int(output.weight.shape[0])
            or int(torch.unique(rows).numel()) != int(rows.numel())
        ):
            raise QwenP35MLPBehaviorResidualError(
                "Child parent_token_rows is not a valid raw-parent vocabulary map."
            )
        self.parent = parent
        self.parent_token_rows = rows
        self.config = copy.copy(parent.config)
        self.config.vocab_size = int(rows.numel())

    @property
    def child_vocabulary_size(self) -> int:
        return int(self.parent_token_rows.numel())

    def get_input_embeddings(self) -> nn.Module:
        return self.parent.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.parent.get_output_embeddings()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise QwenP35MLPBehaviorResidualError(
                "Parent token-row view requires keyword model inputs."
            )
        values = dict(kwargs)
        input_ids = values.get("input_ids")
        if not isinstance(input_ids, Tensor):
            raise QwenP35MLPBehaviorResidualError(
                "Parent token-row view requires integer input_ids."
            )
        rows = self.parent_token_rows.to(device=input_ids.device)
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= int(rows.numel()):
            raise QwenP35MLPBehaviorResidualError(
                "A compact-child token ID is outside parent_token_rows."
            )
        values["input_ids"] = rows.index_select(0, input_ids.reshape(-1)).reshape_as(
            input_ids
        )
        output = self.parent(**values)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor):
            raise QwenP35MLPBehaviorResidualError(
                "Raw parent did not return tensor logits."
            )
        output.logits = logits.index_select(
            -1, self.parent_token_rows.to(device=logits.device)
        )
        return output


def _runtime_neuron_registry(runtime: nn.Module) -> NeuronCodebookRegistry:
    codebook = getattr(runtime, "neuron_codebook", None)
    registry = getattr(codebook, "registry", None)
    if not isinstance(registry, NeuronCodebookRegistry):
        raise QwenP35MLPBehaviorResidualError(
            "Runtime lacks a live global neuron-codebook registry."
        )
    return registry


def _validate_exact_p35_audit(audit: MLPCodebookRedirectAudit) -> None:
    if (
        audit.logical_layers != 36
        or audit.intermediate_size != 9728
        or audit.prototypes_per_neuron != 35
        or audit.prototype_count != 340_480
        or audit.selected_layers != EXACT_LAYER_INDICES
        or audit.redirected_assignments_by_layer != EXACT_REDIRECT_COUNTS
        or audit.total_redirected_assignments != 9728
        or audit.selected_redirected_assignments != 2122
        or audit.assignment_sha256 != EXACT_ASSIGNMENT_SHA256
        or audit.prototype_source_indices_sha256 != EXACT_PROTOTYPE_SOURCE_SHA256
    ):
        raise QwenP35MLPBehaviorResidualError(
            "Live neuron codebook is not the preserved exact MLP-p35 artifact."
        )


def _validate_exact_lineage(lineage: Mapping[str, Any] | None) -> None:
    expected = {
        "p35_manifest_sha256": EXACT_P35_MANIFEST_SHA256,
        "p35_model_safetensors_sha256": EXACT_P35_MODEL_SHA256,
        "p35_state_key_sha256": EXACT_P35_STATE_KEY_SHA256,
        "tokenizer_contract_sha256": EXACT_TOKENIZER_CONTRACT_SHA256,
        "tokenizer_vocab_sha256": EXACT_TOKENIZER_VOCAB_SHA256,
        "parent_token_rows_sha256": EXACT_PARENT_TOKEN_ROWS_SHA256,
        "raw_parent_revision": EXACT_PARENT_REVISION,
        "raw_parent_shard_content_hashes": list(EXACT_PARENT_SHARD_CONTENT_HASHES),
        "checkpoint_loader_verified_manifest": True,
        "live_tokenizer_contract_verified": True,
        "raw_parent_content_address_verified": True,
    }
    if not isinstance(lineage, Mapping) or any(
        lineage.get(name) != value for name, value in expected.items()
    ):
        raise QwenP35MLPBehaviorResidualError(
            "Exact run lacks the required p35, raw-parent, and tokenizer lineage."
        )


def build_p35_mlp_residual_accounting(
    audit: MLPCodebookRedirectAudit,
    *,
    residual_width: int,
    hidden_size: int = 2560,
    checkpoint_element_size: int = 2,
) -> BehaviorResidualSwiGLUAccounting:
    return BehaviorResidualSwiGLUAccounting.build(
        hidden_size=hidden_size,
        residual_width=residual_width,
        site_count=len(audit.selected_layers),
        removed_prototype_count=audit.total_redirected_assignments,
        selected_redirected_assignments=audit.selected_redirected_assignments,
        checkpoint_element_size=checkpoint_element_size,
    )


def _evaluation_delta(
    baseline: FunctionalActionEvaluation,
    locked: FunctionalActionEvaluation,
) -> dict[str, Any]:
    payload = {
        "parent_action_top1_rate": (
            locked.parent_action_top1_rate - baseline.parent_action_top1_rate
        ),
        "mean_parent_action_rank": (
            locked.mean_parent_action_rank - baseline.mean_parent_action_rank
        ),
        "mean_parent_action_log_probability": (
            locked.mean_parent_action_log_probability
            - baseline.mean_parent_action_log_probability
        ),
        "total_output_loss": locked.total_loss - baseline.total_loss,
        "discrete_top1_improved": (
            locked.parent_action_top1_rate > baseline.parent_action_top1_rate
        ),
        "discrete_rank_improved": (
            locked.mean_parent_action_rank < baseline.mean_parent_action_rank
        ),
        "used_for_selection": False,
        "common_parent_and_baseline_student_prefix_domain": True,
    }
    payload["delta_sha256"] = codebook_loader._sha256(payload)
    return payload


def _rollout_selection_provenance(
    token_audit: Mapping[str, Any], split_provenance: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        calibration_hashes = list(
            token_audit["token_row_sha256s"]["calibration-selection"]
        )
        heldout_hashes = list(token_audit["token_row_sha256s"]["heldout-evaluation"])
        overlap = int(token_audit["calibration_heldout_token_overlap_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise QwenP35MLPBehaviorResidualError(
            "Token split audit cannot bind calibration rollout provenance."
        ) from error
    payload = {
        "schema": "neural-forge-calibration-rollout-selection-provenance-v1",
        "selection_split": "calibration-selection",
        "prompt_count": len(calibration_hashes),
        "prompt_token_row_sha256s": calibration_hashes,
        "prompt_token_domain_sha256": codebook_loader._sha256(calibration_hashes),
        "heldout_prompt_count": len(heldout_hashes),
        "heldout_prompt_token_domain_sha256": codebook_loader._sha256(heldout_hashes),
        "calibration_heldout_token_overlap_count": overlap,
        "fresh_v3_heldout_manifest_sha256": split_provenance.get(
            "fresh_v3_heldout_manifest_sha256"
        ),
        "fresh_v3_heldout_fixture_sha256": split_provenance.get(
            "fresh_v3_heldout_fixture_sha256"
        ),
    }
    payload["provenance_sha256"] = codebook_loader._sha256(payload)
    return payload


def _full_calibration_rollout_record(
    parent_traces: Sequence[Any],
    child_traces: Sequence[Any],
    *,
    selection_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = action_runner._greedy_rollout_comparison(parent_traces, child_traces)
    trace_hashes = []
    for parent_trace, child_trace in zip(parent_traces, child_traces, strict=True):
        if parent_trace.sequence_id != child_trace.sequence_id:
            raise QwenP35MLPBehaviorResidualError(
                "Calibration rollout trace IDs do not align."
            )
        trace_hashes.append(
            {
                "sequence_id": parent_trace.sequence_id,
                "prompt_token_ids_sha256": codebook_loader._sha256(
                    list(parent_trace.prompt_token_ids)
                ),
                "parent_selected_token_ids_sha256": codebook_loader._sha256(
                    list(parent_trace.selected_token_ids)
                ),
                "child_selected_token_ids_sha256": codebook_loader._sha256(
                    list(child_trace.selected_token_ids)
                ),
            }
        )
    record = {
        **comparison,
        "trace_hashes": trace_hashes,
        "selection_provenance": dict(selection_provenance),
    }
    record["rollout_evaluation_sha256"] = codebook_loader._sha256(record)
    return record


def _fit_prefix_domain_sha256(
    batches: Sequence[FunctionalActionBatch],
) -> str:
    hashes: set[str] = set()
    for batch in batches:
        for row_index in range(batch.batch_size):
            length = int(batch.attention_mask[row_index].sum().item())
            tokens = batch.input_ids[row_index, :length].tolist()
            hashes.add(codebook_loader._sha256([int(value) for value in tokens]))
    return codebook_loader._sha256(sorted(hashes))


def _build_on_policy_fit_phase(
    parent_fit: Sequence[Any],
    child_fit: Sequence[Any],
    parent_fit_on_student: Sequence[Any],
    fit_prompts: Sequence[Tensor],
    *,
    phase_index: int,
    refresh_after_step: int,
    micro_batch_size: int,
    base_batches: Sequence[FunctionalActionBatch] | None = None,
    batch_window_start: int = 0,
) -> BehaviorResidualSwiGLUFitRefresh:
    parent_values = tuple(parent_fit)
    child_values = tuple(child_fit)
    forced_values = tuple(parent_fit_on_student)
    prompt_values = tuple(fit_prompts)
    if (
        not parent_values
        or len(
            {
                len(parent_values),
                len(child_values),
                len(forced_values),
                len(prompt_values),
            }
        )
        != 1
    ):
        raise QwenP35MLPBehaviorResidualError(
            "Fit phase parent/current-child/forced traces do not align."
        )
    for parent_trace, child_trace, forced_trace, prompt in zip(
        parent_values,
        child_values,
        forced_values,
        prompt_values,
        strict=True,
    ):
        prompt_tokens = tuple(int(value) for value in prompt.reshape(-1).tolist())
        if (
            parent_trace.lane != "greedy"
            or child_trace.lane != "greedy"
            or forced_trace.lane != "forced"
            or tuple(parent_trace.prompt_token_ids) != prompt_tokens
            or tuple(child_trace.prompt_token_ids) != prompt_tokens
            or tuple(forced_trace.prompt_token_ids) != prompt_tokens
            or child_trace.sequence_id != forced_trace.sequence_id
            or tuple(child_trace.selected_token_ids)
            != tuple(forced_trace.selected_token_ids)
            or len(parent_trace.selected_token_ids)
            != len(child_trace.selected_token_ids)
        ):
            raise QwenP35MLPBehaviorResidualError(
                "Fit phase is not semantically bound to current-child forced targets."
            )
    parent_batches = action_runner.build_functional_action_batches(
        parent_fit,
        split="fit",
        prefix_source="parent-greedy",
        micro_batch_size=micro_batch_size,
    )
    student_batches = action_runner.build_functional_action_batches(
        parent_fit_on_student,
        split="fit",
        prefix_source="student-greedy",
        micro_batch_size=micro_batch_size,
    )
    phase_batches = action_runner._paired_batches(
        parent_batches, student_batches, split="fit"
    )
    if base_batches is None:
        batches = phase_batches
    else:
        base_values = tuple(base_batches)
        window_end = batch_window_start + len(phase_batches)
        if (
            batch_window_start < 0
            or window_end > len(base_values)
            or any(
                phase.prefix_source != base.prefix_source
                for phase, base in zip(
                    phase_batches,
                    base_values[batch_window_start:window_end],
                    strict=True,
                )
            )
        ):
            raise QwenP35MLPBehaviorResidualError(
                "Refreshed fit window does not align with the V2 batch schedule."
            )
        for phase, base in zip(
            phase_batches,
            base_values[batch_window_start:window_end],
            strict=True,
        ):
            if (
                phase.prefix_source == "parent-greedy"
                and phase.as_dict() != base.as_dict()
            ):
                raise QwenP35MLPBehaviorResidualError(
                    "A fit refresh changed a static parent-greedy anchor batch."
                )
        merged = list(base_values)
        merged[batch_window_start:window_end] = phase_batches
        batches = tuple(merged)
    prompt_hashes = list(action_runner.role_search._prompt_sha256s(fit_prompts))
    student_driver_rows = [
        {
            "sequence_id": trace.sequence_id,
            "prompt_token_ids_sha256": codebook_loader._sha256(
                list(trace.prompt_token_ids)
            ),
            "student_selected_token_ids_sha256": codebook_loader._sha256(
                list(trace.selected_token_ids)
            ),
        }
        for trace in child_fit
    ]
    parent_target_rows = {
        "parent_greedy": [trace.as_dict() for trace in parent_fit],
        "parent_on_student": [trace.as_dict() for trace in parent_fit_on_student],
    }
    provenance = {
        "schema": "neural-forge-behavior-residual-on-policy-fit-phase-v1",
        "phase_index": phase_index,
        "refresh_after_step": refresh_after_step,
        "refreshed_prompt_count": len(prompt_values),
        "refreshed_batch_window_start": batch_window_start,
        "refreshed_batch_window_count": len(phase_batches),
        "preserves_v2_batch_order_and_microbatch_composition": True,
        "fit_split": "fit",
        "fit_prompt_count": len(prompt_values),
        "fit_prompt_token_row_sha256s": prompt_hashes,
        "fit_prompt_token_domain_sha256": codebook_loader._sha256(prompt_hashes),
        "student_driver_trace_sha256": codebook_loader._sha256(student_driver_rows),
        "parent_target_trace_sha256": codebook_loader._sha256(parent_target_rows),
        "fit_prefix_domain_sha256": _fit_prefix_domain_sha256(batches),
        "batch_manifest_sha256": codebook_loader._sha256(
            [value.as_dict() for value in batches]
        ),
        "refreshed_from_current_student_rollout": True,
        "calibration_used_for_training": False,
        "calibration_model_forwards": 0,
        "heldout_model_forwards": 0,
        "parent_output_targets_only": True,
        "weight_or_activation_targets_used": False,
    }
    provenance["phase_provenance_sha256"] = codebook_loader._sha256(provenance)
    return BehaviorResidualSwiGLUFitRefresh(
        refresh_after_step=refresh_after_step,
        batches=batches,
        provenance=provenance,
    )


def _event(events: list[dict[str, Any]], name: str, **values: Any) -> None:
    events.append({"index": len(events), "event": name, **values})


def run_qwen_p35_mlp_behavior_residual(
    parent_runtime: nn.Module,
    p35_runtime: nn.Module,
    fit_prompt_input_ids: Sequence[Tensor],
    calibration_prompt_input_ids: Sequence[Tensor],
    heldout_prompt_input_ids: Sequence[Tensor],
    *,
    split_provenance: Mapping[str, Any],
    v1_development_heldout_prompt_input_ids: Sequence[Tensor] | None = None,
    v2_development_heldout_prompt_input_ids: Sequence[Tensor] | None = None,
    exact_lineage: Mapping[str, Any] | None = None,
    require_exact_p35: bool = True,
    layer_indices: Sequence[int] = EXACT_LAYER_INDICES,
    residual_width: int = EXACT_RESIDUAL_WIDTH,
    horizon: int = EXACT_HORIZON,
    steps: int = EXACT_STEPS,
    micro_batch_size: int = EXACT_MICRO_BATCH_SIZE,
    gate_up_learning_rate: float = EXACT_GATE_UP_LEARNING_RATE,
    down_learning_rate: float = EXACT_DOWN_LEARNING_RATE,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    hard_action_weight: float = 1.0,
    distribution_weight: float = 0.1,
    temperature: float = 1.0,
    max_gradient_norm: float = 1.0,
    calibration_steps: Sequence[int] = EXACT_CALIBRATION_STEPS,
    fit_refresh_steps: Sequence[int] | None = None,
    initialization_seed: int = 0,
    trace_limits: FunctionalDecodeLimits | None = None,
    artifact_sink: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train in memory, evaluate heldout after lock, and restore exact p35."""

    layer_values = tuple(int(value) for value in layer_indices)
    selection_steps = tuple(int(value) for value in calibration_steps)
    refresh_steps = tuple(
        int(value)
        for value in (
            EXACT_FIT_REFRESH_STEPS
            if fit_refresh_steps is None and require_exact_p35
            else (fit_refresh_steps or ())
        )
    )
    if require_exact_p35 and (
        layer_values != EXACT_LAYER_INDICES
        or residual_width != EXACT_RESIDUAL_WIDTH
        or horizon != EXACT_HORIZON
        or steps != EXACT_STEPS
        or micro_batch_size != EXACT_MICRO_BATCH_SIZE
        or gate_up_learning_rate != EXACT_GATE_UP_LEARNING_RATE
        or down_learning_rate != EXACT_DOWN_LEARNING_RATE
        or selection_steps != EXACT_CALIBRATION_STEPS
        or refresh_steps != EXACT_FIT_REFRESH_STEPS
        or max_gradient_norm != 1.0
    ):
        raise QwenP35MLPBehaviorResidualError(
            "Exact first run requires L1, width64, horizon4, 32 batch-2 "
            "updates, calibration 0/8/16/32, fit refresh 8/16/24, gate/up "
            "lr .001, down lr .002, and clip1."
        )
    if not isinstance(split_provenance, Mapping) or (
        require_exact_p35
        and (
            split_provenance.get("verified_exact_fixture_provenance") is not True
            or split_provenance.get("fresh_manifest_verified_before_model_access")
            is not True
            or split_provenance.get("fresh_ids_and_text_disjoint_from_all_prior_splits")
            is not True
            or split_provenance.get("fresh_v3_heldout_fixture_sha256")
            != EXACT_SEALED_HELDOUT_FIXTURE_SHA256
            or split_provenance.get("fresh_v3_heldout_manifest_sha256")
            != EXACT_SEALED_HELDOUT_MANIFEST_SHA256
        )
    ):
        raise QwenP35MLPBehaviorResidualError(
            "Exact mode requires a pre-model sealed fresh heldout provenance."
        )
    if require_exact_p35:
        _validate_exact_lineage(exact_lineage)
    fit_prompts, calibration_prompts, heldout_prompts, token_audit = (
        action_runner.behavior_runner._validate_token_splits(
            fit_prompt_input_ids,
            calibration_prompt_input_ids,
            heldout_prompt_input_ids,
            max_prompts_per_split=max(EXPECTED_SPLIT_COUNTS.values()),
            require_exact_p35=require_exact_p35,
        )
    )
    v1_development_heldout_prompts = tuple(
        v1_development_heldout_prompt_input_ids or ()
    )
    v2_development_heldout_prompts = tuple(
        v2_development_heldout_prompt_input_ids or ()
    )
    if require_exact_p35 and any(
        len(values) != EXPECTED_SPLIT_COUNTS["heldout-evaluation"]
        for values in (
            v1_development_heldout_prompts,
            v2_development_heldout_prompts,
        )
    ):
        raise QwenP35MLPBehaviorResidualError(
            "Exact v3 requires both eight-row prior development token exclusions."
        )
    if v1_development_heldout_prompts or v2_development_heldout_prompts:
        v1_development_hashes = action_runner.role_search._prompt_sha256s(
            v1_development_heldout_prompts
        )
        v2_development_hashes = action_runner.role_search._prompt_sha256s(
            v2_development_heldout_prompts
        )
        current_hashes = {
            value
            for split_hashes in token_audit["token_row_sha256s"].values()
            for value in split_hashes
        }
        if set(v1_development_hashes) & set(v2_development_hashes):
            raise QwenP35MLPBehaviorResidualError(
                "V1 and V2 development token exclusions unexpectedly overlap."
            )
        cross_run_token_overlap = current_hashes & (
            set(v1_development_hashes) | set(v2_development_hashes)
        )
        if cross_run_token_overlap:
            raise QwenP35MLPBehaviorResidualError(
                "Fresh V3 token rows overlap a prior development-heldout split."
            )
        token_audit = {
            **token_audit,
            "v1_development_heldout_token_row_sha256s": list(v1_development_hashes),
            "v2_development_heldout_token_row_sha256s": list(v2_development_hashes),
            "fresh_v3_all_current_vs_prior_heldout_token_overlap_count": len(
                cross_run_token_overlap
            ),
            "fresh_v3_token_rows_disjoint_from_all_prior_heldout": True,
        }
    limits = trace_limits or FunctionalDecodeLimits(max_horizon=horizon)
    if horizon > limits.max_horizon:
        raise QwenP35MLPBehaviorResidualError(
            "horizon exceeds the bounded functional decoder."
        )
    rollout_provenance = _rollout_selection_provenance(token_audit, split_provenance)

    registry = _runtime_neuron_registry(p35_runtime)
    audit = audit_mlp_codebook_redirects(
        registry,
        selected_layers=layer_values,
        expected_assignment_sha256=(
            EXACT_ASSIGNMENT_SHA256 if require_exact_p35 else None
        ),
        expected_prototype_source_sha256=(
            EXACT_PROTOTYPE_SOURCE_SHA256 if require_exact_p35 else None
        ),
    )
    if require_exact_p35:
        _validate_exact_p35_audit(audit)
        parameter_report = qwen3_codebook_parameter_report(p35_runtime)
        if (
            parameter_report.neuron_prototypes_per_neuron != 35
            or parameter_report.attention_prototypes_per_group != 36
            or parameter_report.unique_parameter_count != 3_947_073_536
        ):
            raise QwenP35MLPBehaviorResidualError(
                "Exact runtime parameter report is not MLP-p35/attention-p36."
            )
        if registry.prototype_gate_weight.element_size() != 2:
            raise QwenP35MLPBehaviorResidualError(
                "Exact first run requires FP16 or BF16 two-byte codebook weights."
            )
    accounting = build_p35_mlp_residual_accounting(
        audit,
        residual_width=residual_width,
        hidden_size=registry.input_width,
        checkpoint_element_size=registry.prototype_gate_weight.element_size(),
    )
    parent_guard = _parameter_guard(tuple(parent_runtime.parameters()))
    p35_guard = _parameter_guard(tuple(p35_runtime.parameters()))
    child_vocab = int(p35_runtime.get_output_embeddings().weight.shape[0])
    parent_token_rows = getattr(p35_runtime, "parent_token_rows", None)
    if not isinstance(parent_token_rows, Tensor) or (
        int(parent_token_rows.numel()) != child_vocab
    ):
        raise QwenP35MLPBehaviorResidualError(
            "P35 runtime lacks its exact raw-parent token-row bridge."
        )
    parent_view = _ParentTokenRowView(parent_runtime, parent_token_rows)
    events: list[dict[str, Any]] = []
    _event(events, "fixtures-and-p35-audited", heldout_model_forward=False)

    handle = None
    training = None
    locked_state: dict[str, Tensor] | None = None
    locked_state_sha256: str | None = None
    baseline_heldout: FunctionalActionEvaluation | None = None
    locked_heldout: FunctionalActionEvaluation | None = None
    parent_heldout_greedy = ()
    baseline_heldout_greedy = ()
    locked_heldout_greedy = ()
    fit_batches = ()
    calibration_batches = ()
    heldout_batches = ()
    selection_callback_splits: list[str] = []
    fit_refresh_callback_splits: list[str] = []
    rollback_exact = False
    try:
        _event(events, "parent-fit-calibration-targets", heldout_model_forward=False)
        parent_fit, _ = action_runner._collect_greedy_split(
            parent_view,
            fit_prompts,
            split="fit",
            horizon=horizon,
            limits=limits,
        )
        parent_calibration, _ = action_runner._collect_greedy_split(
            parent_view,
            calibration_prompts,
            split="calibration-selection",
            horizon=horizon,
            limits=limits,
        )
        _assert_guard(parent_guard)

        _event(events, "p35-fit-calibration-drivers", heldout_model_forward=False)
        child_fit, child_fit_drivers = action_runner._collect_greedy_split(
            p35_runtime,
            fit_prompts,
            split="fit",
            horizon=horizon,
            limits=limits,
        )
        _child_calibration, child_calibration_drivers = (
            action_runner._collect_greedy_split(
                p35_runtime,
                calibration_prompts,
                split="calibration-selection",
                horizon=horizon,
                limits=limits,
            )
        )
        _assert_guard(p35_guard)

        parent_fit_on_student = action_runner._collect_forced_split(
            parent_view,
            fit_prompts,
            child_fit_drivers,
            split="fit",
            horizon=horizon,
            limits=limits,
        )
        parent_calibration_on_student = action_runner._collect_forced_split(
            parent_view,
            calibration_prompts,
            child_calibration_drivers,
            split="calibration-selection",
            horizon=horizon,
            limits=limits,
        )
        initial_fit_phase = _build_on_policy_fit_phase(
            parent_fit,
            child_fit,
            parent_fit_on_student,
            fit_prompts,
            phase_index=0,
            refresh_after_step=0,
            micro_batch_size=micro_batch_size,
        )
        fit_batches = initial_fit_phase.batches
        fit_phase_count = len(refresh_steps) + 1
        if len(fit_prompts) % fit_phase_count:
            raise QwenP35MLPBehaviorResidualError(
                "Fit prompts do not divide evenly across refresh phases."
            )
        fit_prompts_per_refresh = len(fit_prompts) // fit_phase_count
        fit_batch_window = len(fit_batches) // fit_phase_count
        current_fit_schedule = fit_batches
        if require_exact_p35 and (
            len(fit_batches) != 32
            or action_runner._batch_manifest(fit_batches)["manifest_sha256"]
            != REJECTED_V2_FIT_BATCH_MANIFEST_SHA256
        ):
            raise QwenP35MLPBehaviorResidualError(
                "V3 initial fit update order differs from the pinned V2 domain."
            )
        calibration_batches = action_runner._paired_batches(
            action_runner.build_functional_action_batches(
                parent_calibration,
                split="calibration-selection",
                prefix_source="parent-greedy",
                micro_batch_size=micro_batch_size,
            ),
            action_runner.build_functional_action_batches(
                parent_calibration_on_student,
                split="calibration-selection",
                prefix_source="student-greedy",
                micro_batch_size=micro_batch_size,
            ),
            split="calibration-selection",
        )
        _assert_guard(parent_guard)

        handle = install_behavior_residual_swiglu(
            p35_runtime,
            layer_indices=layer_values,
            residual_width=residual_width,
            accounting=accounting,
            initialization_seed=initialization_seed,
        )
        _event(events, "exact-zero-residual-installed", heldout_model_forward=False)

        def evaluate_calibration_free_rollout() -> Mapping[str, Any]:
            state_before = handle.residual_state_sha256()
            selection_callback_splits.append("calibration-selection")
            child_calibration, _ = action_runner._collect_greedy_split(
                p35_runtime,
                calibration_prompts,
                split="calibration-selection",
                horizon=horizon,
                limits=limits,
            )
            handle.assert_live()
            state_after = handle.residual_state_sha256()
            if state_after != state_before:
                raise QwenP35MLPBehaviorResidualError(
                    "Calibration rollout collection mutated the residual state."
                )
            record = _full_calibration_rollout_record(
                parent_calibration,
                child_calibration,
                selection_provenance=rollout_provenance,
            )
            _event(
                events,
                "calibration-free-rollout-evaluation",
                heldout_model_forward=False,
                selection_split="calibration-selection",
                prompt_token_domain_sha256=rollout_provenance[
                    "prompt_token_domain_sha256"
                ],
                residual_state_sha256=state_after,
                rollout_evaluation_sha256=record["rollout_evaluation_sha256"],
            )
            return record

        phase_by_refresh_step = {
            refresh_step: phase_index
            for phase_index, refresh_step in enumerate(refresh_steps, start=1)
        }

        def refresh_on_policy_fit(
            refresh_after_step: int,
        ) -> BehaviorResidualSwiGLUFitRefresh:
            nonlocal current_fit_schedule
            phase_index = phase_by_refresh_step.get(refresh_after_step)
            if phase_index is None:
                raise QwenP35MLPBehaviorResidualError(
                    "Fit refresher received an undeclared phase boundary."
                )
            state_before = handle.residual_state_sha256()
            fit_refresh_callback_splits.append("fit")
            prompt_start = phase_index * fit_prompts_per_refresh
            prompt_end = prompt_start + fit_prompts_per_refresh
            phase_prompts = fit_prompts[prompt_start:prompt_end]
            phase_parent_fit = parent_fit[prompt_start:prompt_end]
            current_child_fit, current_child_drivers = (
                action_runner._collect_greedy_split(
                    p35_runtime,
                    phase_prompts,
                    split="fit",
                    horizon=horizon,
                    limits=limits,
                )
            )
            current_parent_on_student = action_runner._collect_forced_split(
                parent_view,
                phase_prompts,
                current_child_drivers,
                split="fit",
                horizon=horizon,
                limits=limits,
            )
            handle.assert_live()
            _assert_guard(parent_guard)
            state_after = handle.residual_state_sha256()
            if state_after != state_before:
                raise QwenP35MLPBehaviorResidualError(
                    "On-policy fit refresh mutated the residual state."
                )
            phase = _build_on_policy_fit_phase(
                phase_parent_fit,
                current_child_fit,
                current_parent_on_student,
                phase_prompts,
                phase_index=phase_index,
                refresh_after_step=refresh_after_step,
                micro_batch_size=micro_batch_size,
                base_batches=current_fit_schedule,
                batch_window_start=phase_index * fit_batch_window,
            )
            current_fit_schedule = phase.batches
            _event(
                events,
                "on-policy-fit-prefix-refresh",
                heldout_model_forward=False,
                training_split="fit",
                calibration_model_forward=False,
                phase_index=phase_index,
                refreshed_prompt_indices=list(range(prompt_start, prompt_end)),
                refresh_after_step=refresh_after_step,
                residual_state_sha256=state_after,
                phase_provenance_sha256=phase.provenance["phase_provenance_sha256"],
            )
            return phase

        _event(
            events,
            "calibration-free-rollout-selection-enabled",
            heldout_model_forward=False,
        )
        training = train_behavior_residual_swiglu_disjoint(
            handle,
            fit_batches,
            calibration_batches,
            steps=steps,
            gate_up_learning_rate=gate_up_learning_rate,
            down_learning_rate=down_learning_rate,
            beta1=beta1,
            beta2=beta2,
            adam_epsilon=adam_epsilon,
            max_gradient_norm=max_gradient_norm,
            hard_action_weight=hard_action_weight,
            distribution_weight=distribution_weight,
            temperature=temperature,
            calibration_steps=selection_steps,
            calibration_rollout_evaluator=evaluate_calibration_free_rollout,
            calibration_rollout_provenance=rollout_provenance,
            fit_refresh_steps=refresh_steps,
            fit_batch_refresher=(refresh_on_policy_fit if refresh_steps else None),
            initial_fit_provenance=initial_fit_phase.provenance,
        )
        expected_selection_calls = len(selection_steps) + 1
        if (
            selection_callback_splits
            != ["calibration-selection"] * expected_selection_calls
        ):
            raise QwenP35MLPBehaviorResidualError(
                "Calibration rollout callback count or split provenance changed."
            )
        if fit_refresh_callback_splits != ["fit"] * len(refresh_steps):
            raise QwenP35MLPBehaviorResidualError(
                "On-policy fit refresh count or split provenance changed."
            )
        consumed_schedule_positions = [
            int(value["fit_batch_schedule_position"]) for value in training.fit_history
        ]
        if require_exact_p35 and consumed_schedule_positions != list(range(steps)):
            raise QwenP35MLPBehaviorResidualError(
                "Exact V3 did not consume each preserved update position once."
            )
        locked_state = handle.residual_state()
        locked_state_sha256 = handle.residual_state_sha256()
        _event(events, "calibration-winner-locked", heldout_model_forward=False)

        _event(events, "parent-heldout-targets", heldout_model_forward=True)
        parent_heldout_greedy, _ = action_runner._collect_greedy_split(
            parent_view,
            heldout_prompts,
            split="heldout-evaluation",
            horizon=horizon,
            limits=limits,
        )
        handle.zero_output_correction()
        _event(events, "baseline-p35-heldout-drivers", heldout_model_forward=True)
        baseline_heldout_greedy, baseline_drivers = action_runner._collect_greedy_split(
            p35_runtime,
            heldout_prompts,
            split="heldout-evaluation",
            horizon=horizon,
            limits=limits,
        )
        parent_heldout_on_student = action_runner._collect_forced_split(
            parent_view,
            heldout_prompts,
            baseline_drivers,
            split="heldout-evaluation",
            horizon=horizon,
            limits=limits,
        )
        heldout_batches = action_runner._paired_batches(
            action_runner.build_functional_action_batches(
                parent_heldout_greedy,
                split="heldout-evaluation",
                prefix_source="parent-greedy",
                micro_batch_size=micro_batch_size,
            ),
            action_runner.build_functional_action_batches(
                parent_heldout_on_student,
                split="heldout-evaluation",
                prefix_source="student-greedy",
                micro_batch_size=micro_batch_size,
            ),
            split="heldout-evaluation",
        )
        baseline_heldout = evaluate_functional_action_batches(
            p35_runtime,
            heldout_batches,
            expected_split="heldout-evaluation",
            hard_action_weight=hard_action_weight,
            distribution_weight=distribution_weight,
            temperature=temperature,
            used_for_selection=False,
        )
        handle.load_residual_state(locked_state)
        if handle.residual_state_sha256() != locked_state_sha256:
            raise QwenP35MLPBehaviorResidualError(
                "Locked residual factors did not reinstall exactly."
            )
        _event(events, "locked-residual-heldout", heldout_model_forward=True)
        locked_heldout = evaluate_functional_action_batches(
            p35_runtime,
            heldout_batches,
            expected_split="heldout-evaluation",
            hard_action_weight=hard_action_weight,
            distribution_weight=distribution_weight,
            temperature=temperature,
            used_for_selection=False,
        )
        locked_heldout_greedy, _ = action_runner._collect_greedy_split(
            p35_runtime,
            heldout_prompts,
            split="heldout-evaluation",
            horizon=horizon,
            limits=limits,
        )
        if artifact_sink is not None:
            artifact_sink.clear()
            artifact_sink.update(
                {
                    "locked_state": {
                        name: value.clone() for name, value in locked_state.items()
                    },
                    "locked_state_sha256": locked_state_sha256,
                    "audit": audit,
                    "accounting": accounting,
                }
            )
        handle.assert_base_unchanged()
        _assert_guard(parent_guard)
    finally:
        if handle is not None and handle.is_applied:
            handle.rollback()
        try:
            _assert_guard(parent_guard)
            _assert_guard(p35_guard)
            rollback_audit = audit_mlp_codebook_redirects(
                _runtime_neuron_registry(p35_runtime),
                selected_layers=layer_values,
                expected_assignment_sha256=(
                    EXACT_ASSIGNMENT_SHA256 if require_exact_p35 else None
                ),
                expected_prototype_source_sha256=(
                    EXACT_PROTOTYPE_SOURCE_SHA256 if require_exact_p35 else None
                ),
            )
            rollback_exact = rollback_audit == audit
        except Exception:
            rollback_exact = False
    if (
        training is None
        or locked_state is None
        or locked_state_sha256 is None
        or baseline_heldout is None
        or locked_heldout is None
        or not rollback_exact
    ):
        raise QwenP35MLPBehaviorResidualError(
            "The residual run did not finish with exact p35 rollback."
        )
    first_heldout = next(value for value in events if value["heldout_model_forward"])
    lock_event = next(
        value for value in events if value["event"] == "calibration-winner-locked"
    )
    if first_heldout["index"] <= lock_event["index"]:
        raise QwenP35MLPBehaviorResidualError(
            "Heldout model access occurred before calibration lock."
        )
    selection_events = [
        value
        for value in events
        if value["event"] == "calibration-free-rollout-evaluation"
    ]
    fit_refresh_events = [
        value for value in events if value["event"] == "on-policy-fit-prefix-refresh"
    ]
    prelock_heldout_events = [
        value
        for value in events[: lock_event["index"]]
        if value["heldout_model_forward"]
    ]
    if (
        len(selection_events) != training.calibration_rollout_evaluations
        or len(fit_refresh_events) != training.fit_refresh_evaluations
        or prelock_heldout_events
    ):
        raise QwenP35MLPBehaviorResidualError(
            "Selection callback or pre-lock heldout access audit failed."
        )

    baseline_rollout = action_runner._greedy_rollout_comparison(
        parent_heldout_greedy, baseline_heldout_greedy
    )
    locked_rollout = action_runner._greedy_rollout_comparison(
        parent_heldout_greedy, locked_heldout_greedy
    )
    heldout_delta = _evaluation_delta(baseline_heldout, locked_heldout)
    calibration_improved = training.best_step != 0
    bounded_gate = bool(
        calibration_improved
        and locked_heldout.parent_action_top1_rate
        >= baseline_heldout.parent_action_top1_rate
        and locked_rollout["exact_rollout_matches"]
        >= baseline_rollout["exact_rollout_matches"]
        and locked_rollout["token_matches"] >= baseline_rollout["token_matches"]
        and locked_rollout["mean_longest_common_prefix"]
        >= baseline_rollout["mean_longest_common_prefix"]
        and rollback_exact
    )
    report: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "hypothesis": (
            "Refreshing fit-only student prefixes from the current residual "
            "rollout can repair behavior that static baseline-child prefixes "
            "miss, without matching parent weights or internal tensors."
        ),
        "p35_codebook_audit": audit.as_dict(),
        "accounting": accounting.as_dict(),
        "installation": handle.report.as_dict(),
        "fit_action_batches": action_runner._batch_manifest(fit_batches),
        "calibration_action_batches": action_runner._batch_manifest(
            calibration_batches
        ),
        "heldout_action_batches": action_runner._batch_manifest(heldout_batches),
        "token_split_audit": token_audit,
        "split_provenance": dict(split_provenance),
        "exact_lineage": None if exact_lineage is None else dict(exact_lineage),
        "parent_token_row_bridge": {
            "raw_parent_vocabulary_size": int(
                parent_runtime.get_output_embeddings().weight.shape[0]
            ),
            "child_vocabulary_size": parent_view.child_vocabulary_size,
            "parent_token_rows_sha256": codebook_loader._sha256(
                [int(value) for value in parent_token_rows.detach().cpu().tolist()]
            ),
            "inputs_mapped_child_to_parent": True,
            "logits_projected_parent_to_child": True,
            "permanent_runtime_bridge": False,
        },
        "training": training.as_dict(),
        "rejected_v1_predecessor": {
            "local_report": (
                "runs/mlp-behavior-residual-20260711/"
                "p35_mlp_L1_behavior_residual_swiglu_w64_h4_s32_fp16_"
                "p40_pair.json"
            ),
            "file_sha256": REJECTED_V1_REPORT_FILE_SHA256,
            "internal_report_sha256": REJECTED_V1_INTERNAL_REPORT_SHA256,
            "selected_step": 8,
            "baseline_heldout_action": {
                "parent_action_top1_rate": 0.609375,
                "mean_parent_action_rank": 9.046875,
            },
            "locked_heldout_action": {
                "parent_action_top1_rate": 0.625,
                "mean_parent_action_rank": 3.28125,
            },
            "baseline_heldout_rollout": {
                "exact_rollout_matches": 2,
                "token_matches": 8,
                "token_count": 32,
                "mean_longest_common_prefix": 1.0,
            },
            "locked_heldout_rollout": {
                "exact_rollout_matches": 1,
                "token_matches": 9,
                "token_count": 32,
                "mean_longest_common_prefix": 1.125,
            },
            "root_cause": (
                "action-only calibration selected higher top1/better rank while "
                "heldout exact free rollouts regressed from 2/8 to 1/8"
            ),
            "operator_training_and_lineage_reused_without_restart": True,
            "inspected_evaluation_reclassified_as_development_data": True,
        },
        "rejected_v2_predecessor": {
            "local_report": (
                "runs/mlp-behavior-residual-20260711/"
                "p35_mlp_L1_behavior_residual_swiglu_w64_h4_s32_"
                "rolloutsel_v2_fp16_p40_pair.json"
            ),
            "file_sha256": REJECTED_V2_REPORT_FILE_SHA256,
            "internal_report_sha256": REJECTED_V2_INTERNAL_REPORT_SHA256,
            "fit_batch_manifest_sha256": (REJECTED_V2_FIT_BATCH_MANIFEST_SHA256),
            "selected_step": 0,
            "calibration_initial": {
                "exact_rollout_matches": 2,
                "token_matches": 11,
                "mean_longest_common_prefix": 1.6666666666666667,
                "parent_action_top1_rate": 0.7083333333333334,
            },
            "step_8_rejected": {
                "exact_rollout_matches": 2,
                "token_matches": 9,
                "mean_longest_common_prefix": 1.5,
                "parent_action_top1_rate": 0.7291666666666666,
            },
            "fresh_v2_heldout_unchanged_at_locked_step_0": True,
            "root_cause_hypothesis": (
                "all 32 updates used parent targets on static baseline-child "
                "prefixes instead of the residual's evolving rollout domain"
            ),
            "operator_selector_lineage_and_adam_rule_reused_without_restart": True,
            "inspected_evaluation_reclassified_as_development_data": True,
        },
        "cross_run_evaluation_contract": {
            "v1_or_v2_heldout_reused_for_v3_evaluation": False,
            "v1_and_v2_heldout_used_as_method_development_evidence": True,
            "fresh_v3_heldout_fixture_sha256": split_provenance.get(
                "fresh_v3_heldout_fixture_sha256"
            ),
            "fresh_v3_heldout_manifest_sha256": split_provenance.get(
                "fresh_v3_heldout_manifest_sha256"
            ),
            "fresh_manifest_verified_before_model_access": split_provenance.get(
                "fresh_manifest_verified_before_model_access"
            ),
            "fresh_ids_and_text_disjoint_from_all_prior_splits": split_provenance.get(
                "fresh_ids_and_text_disjoint_from_all_prior_splits"
            ),
        },
        "calibration_winner_lock": {
            "best_step": training.best_step,
            "residual_state_sha256": locked_state_sha256,
            "selected_by": ("calibration-free-rollout-then-action-lexicographic"),
            "priority": [
                "require-no-regression-in-exact-rollout-matches",
                "require-no-regression-in-token-matches",
                "require-no-regression-in-mean-longest-common-prefix",
                "then-maximize-rollout-metrics-lexicographically",
                "maximize-parent-action-top1-rate",
                "minimize-mean-parent-action-rank",
            ],
            "calibration_prompts_only": (
                set(selection_callback_splits) == {"calibration-selection"}
            ),
            "heldout_inaccessible_until_final_lock": (len(prelock_heldout_events) == 0),
            "continuous_loss_can_promote": False,
        },
        "on_policy_fit_refresh_audit": {
            "method": "fit-only-current-student-prefix-dagger",
            "refresh_after_steps": list(refresh_steps),
            "callback_calls": len(fit_refresh_callback_splits),
            "callback_splits": fit_refresh_callback_splits,
            "callback_event_count": len(fit_refresh_events),
            "fit_prompt_count_per_refresh": fit_prompts_per_refresh,
            "decode_horizon_per_refresh": horizon,
            "update_mix": {
                "parent_greedy_anchor_updates": sum(
                    value["prefix_source"] == "parent-greedy"
                    for value in training.fit_history
                ),
                "student_greedy_updates_total": sum(
                    value["prefix_source"] == "student-greedy"
                    for value in training.fit_history
                ),
                "initial_exact_zero_student_updates": sum(
                    value["prefix_source"] == "student-greedy"
                    and int(value["fit_phase_index"]) == 0
                    for value in training.fit_history
                ),
                "post_refresh_current_student_updates": sum(
                    value["prefix_source"] == "student-greedy"
                    and int(value["fit_phase_index"]) > 0
                    for value in training.fit_history
                ),
                "intentional_mixed_dagger_parent_anchors": True,
            },
            "consumed_fit_batch_schedule_positions": consumed_schedule_positions,
            "each_exact_schedule_position_consumed_once": (
                consumed_schedule_positions == list(range(steps))
            ),
            "discarded_refreshed_fit_batches": 0,
            "child_greedy_forward_call_units": (
                len(refresh_steps) * fit_prompts_per_refresh * horizon
            ),
            "parent_forced_target_forward_call_units": (
                len(refresh_steps) * fit_prompts_per_refresh * horizon
            ),
            "added_forward_call_units_vs_v2": (
                2 * len(refresh_steps) * fit_prompts_per_refresh * horizon
            ),
            "forward_call_units_are_not_flops_or_wall_time": True,
            "cpu_hashing_and_batch_rebuild_overhead_not_counted": True,
            "calibration_callback_calls_for_training": sum(
                value != "fit" for value in fit_refresh_callback_splits
            ),
            "heldout_callback_calls_for_training": sum(
                value == "heldout-evaluation" for value in fit_refresh_callback_splits
            ),
            "all_refresh_events_declare_zero_calibration_forwards": all(
                value.get("calibration_model_forward") is False
                for value in fit_refresh_events
            ),
            "manual_adam_moment_reset_count": (training.manual_adam_moment_reset_count),
            "manual_adam_moments_preserved": all(
                phase["manual_adam_moment_sha256_before"]
                == phase["manual_adam_moment_sha256_after"]
                and phase["manual_adam_moments_reset"] is False
                for phase in training.fit_phase_history
            ),
            "fit_phase_history": [dict(value) for value in training.fit_phase_history],
            "v2_batch_order_and_microbatch_composition_preserved": all(
                phase["phase_provenance"].get(
                    "preserves_v2_batch_order_and_microbatch_composition"
                )
                is True
                for phase in training.fit_phase_history
            ),
            "phase_batch_cursor_starts": [
                phase["batch_cursor_start"] for phase in training.fit_phase_history
            ],
            "calibration_never_trains": True,
            "parent_output_targets_only": True,
            "weight_or_activation_targets_used": False,
        },
        "calibration_rollout_callback_audit": {
            "selection_provenance": rollout_provenance,
            "callback_calls": len(selection_callback_splits),
            "callback_splits": selection_callback_splits,
            "callback_event_count": len(selection_events),
            "heldout_split_callback_calls": sum(
                value == "heldout-evaluation" for value in selection_callback_splits
            ),
            "prelock_heldout_model_forward_event_count": len(prelock_heldout_events),
            "all_callback_state_hashes_recorded": all(
                isinstance(value.get("residual_state_sha256"), str)
                for value in selection_events
            ),
            "all_callback_rollout_hashes_recorded": all(
                isinstance(value.get("rollout_evaluation_sha256"), str)
                for value in selection_events
            ),
        },
        "calibration_selection_compute": {
            "measurement_unit": "model-forward-call-count",
            "forward_call_units_are_not_flops_or_wall_time": True,
            "training_backward_updates": training.fit_gradient_calls,
            "action_evaluation_calls": training.calibration_evaluations,
            "action_batch_forward_calls": (
                training.calibration_evaluations * len(calibration_batches)
            ),
            "free_rollout_evaluation_calls": (training.calibration_rollout_evaluations),
            "free_rollout_child_decode_forward_steps": (
                training.calibration_rollout_evaluations
                * len(calibration_prompts)
                * horizon
            ),
            "v1_reference_action_evaluation_calls": 5,
            "v2_reference_action_evaluation_calls": 5,
            "additional_action_batch_forward_calls_vs_v1": (
                max(0, training.calibration_evaluations - 5) * len(calibration_batches)
            ),
            "additional_free_rollout_child_forward_calls_vs_v1": (
                training.calibration_rollout_evaluations
                * len(calibration_prompts)
                * horizon
            ),
            "additional_child_forward_call_units_vs_v1": (
                max(0, training.calibration_evaluations - 5) * len(calibration_batches)
                + training.calibration_rollout_evaluations
                * len(calibration_prompts)
                * horizon
            ),
            "additional_calibration_forward_call_units_vs_v2": 0,
            "additional_fit_refresh_child_forward_call_units_vs_v2": (
                len(refresh_steps) * fit_prompts_per_refresh * horizon
            ),
            "additional_fit_refresh_parent_forward_call_units_vs_v2": (
                len(refresh_steps) * fit_prompts_per_refresh * horizon
            ),
            "total_additional_forward_call_units_vs_v2": (
                2 * len(refresh_steps) * fit_prompts_per_refresh * horizon
            ),
            "model_loads_added_vs_v1": 0,
            "backward_updates_added_vs_v1": 0,
            "peak_parameter_memory_expected_to_change": False,
            "wall_time_measured": False,
        },
        "heldout_evaluation": {
            "baseline_p35": baseline_heldout.as_dict(),
            "locked_residual": locked_heldout.as_dict(),
            "locked_minus_baseline": heldout_delta,
            "baseline_rollout": baseline_rollout,
            "locked_rollout": locked_rollout,
            "used_for_fitting": False,
            "used_for_selection": False,
        },
        "heldout_access_audit": {
            "events": events,
            "calibration_lock_event_index": lock_event["index"],
            "first_heldout_model_forward_event_index": first_heldout["index"],
            "heldout_model_forward_only_after_lock": (len(prelock_heldout_events) == 0),
            "evaluation_split_is_fresh_v3_seal": split_provenance.get(
                "fresh_manifest_verified_before_model_access"
            ),
        },
        "bounded_method_development_gate_passed": bounded_gate,
        "broad_behavior_certification_passed": False,
        "checkpoint_saved": False,
        "factor_pack_saved": False,
        "candidate_committed": False,
        "p35_runtime_restored_exactly": rollback_exact,
        "parent_runtime_unchanged": True,
        "factor_persistence": "in-memory-artifact-sink-only",
        "compact_factor_pack_implemented": False,
        "native_qwen_publication_bake_implemented": False,
        "dense_bake_helper_scope": (
            "mathematical local equivalence for an already dense target MLP; "
            "codebook materialization is rejected"
        ),
        "publication_requirement": (
            "reserve residual neurons inside an already-counted target "
            "intermediate width, then prove exact config/count/save/reload"
        ),
        "training_used": True,
        "excluded_training_targets": [
            "parent-weights",
            "parent-activations",
            "parent-hidden-states",
            "parent-mlp-outputs",
            "parent-residual-streams",
        ],
        "scope_warning": (
            "The 8-fit/6-calibration/8-fresh-heldout fixture is a bounded "
            "method-development gate, not broad MLP certification; both V1 "
            "and V2 inspected evaluation rows are development data only."
        ),
        "scientific_risks": [
            "nonstationary on-policy targets can make preserved Adam moments stale",
            "eight fit prompts can overfit their evolving trajectories",
            "refresh timing is coupled to the preserved deterministic batch order",
            "preserved prompt order couples refresh age to prompt category",
            "parent forced targets may be brittle on off-parent student trajectories",
            "forward-call counts do not predict FLOPs or wall time",
        ],
    }
    report["report_sha256"] = codebook_loader._sha256(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model", required=True)
    parser.add_argument("--parent-revision")
    parser.add_argument("--codebook-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--fit-prompts-json", type=Path, default=DEFAULT_FIT_PROMPTS)
    parser.add_argument(
        "--calibration-selection-prompts-json",
        type=Path,
        default=DEFAULT_CALIBRATION_PROMPTS,
    )
    parser.add_argument(
        "--heldout-evaluation-prompts-json",
        type=Path,
        default=DEFAULT_HELDOUT_PROMPTS,
    )
    parser.add_argument(
        "--sealed-heldout-manifest",
        type=Path,
        default=DEFAULT_SEALED_HELDOUT_MANIFEST,
    )
    parser.add_argument(
        "--fit-prompt-manifest", type=Path, default=DEFAULT_FIT_MANIFEST
    )
    parser.add_argument(
        "--audited-prompt-manifest", type=Path, default=DEFAULT_AUDITED_MANIFEST
    )
    parser.add_argument(
        "--transition-inventory", type=Path, default=DEFAULT_TRANSITION_INVENTORY
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-device", default="cuda:0")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--residual-width", type=int, default=EXACT_RESIDUAL_WIDTH)
    parser.add_argument("--horizon", type=int, default=EXACT_HORIZON)
    parser.add_argument("--steps", type=int, default=EXACT_STEPS)
    parser.add_argument(
        "--calibration-steps",
        type=int,
        nargs="+",
        default=EXACT_CALIBRATION_STEPS,
        help=(
            "Bounded rollout/action selection checkpoints; exact v3 preserves "
            "the v2 candidate set 0 8 16 32."
        ),
    )
    parser.add_argument(
        "--fit-refresh-steps",
        type=int,
        nargs="+",
        default=EXACT_FIT_REFRESH_STEPS,
        help="Completed-update boundaries for fit-only on-policy refresh.",
    )
    parser.add_argument("--micro-batch-size", type=int, default=EXACT_MICRO_BATCH_SIZE)
    parser.add_argument(
        "--gate-up-learning-rate",
        type=float,
        default=EXACT_GATE_UP_LEARNING_RATE,
    )
    parser.add_argument(
        "--down-learning-rate", type=float, default=EXACT_DOWN_LEARNING_RATE
    )
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--hard-action-weight", type=float, default=1.0)
    parser.add_argument("--distribution-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-vocabulary", type=int, default=1_000_000)
    parser.add_argument("--max-logit-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.set_defaults(model=None)
    return parser


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _read_json_mapping(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QwenP35MLPBehaviorResidualError(
            f"Could not read exact {name}."
        ) from error
    if not isinstance(value, dict):
        raise QwenP35MLPBehaviorResidualError(f"Exact {name} is not a mapping.")
    return value


def _verify_file_content_hashes(
    paths: Sequence[Path],
    expected: Sequence[str],
    *,
    name: str,
) -> list[str]:
    observed = [codebook_loader._file_sha256(path) for path in paths]
    if tuple(observed) != tuple(expected):
        raise QwenP35MLPBehaviorResidualError(
            f"{name} content hashes differ from the pinned lineage."
        )
    return observed


def _build_exact_lineage(
    args: argparse.Namespace,
    tokenizer: Any,
) -> dict[str, Any]:
    checkpoint = args.codebook_checkpoint.resolve()
    manifest_path = checkpoint / "manifest.json"
    contract_path = checkpoint / "tokenizer_contract.json"
    manifest = _read_json_mapping(manifest_path, "p35 manifest")
    contract = _read_json_mapping(contract_path, "tokenizer contract")
    try:
        normalized_vocab = {
            str(token): int(token_id)
            for token, token_id in tokenizer.get_vocab().items()
        }
    except Exception as error:
        raise QwenP35MLPBehaviorResidualError(
            "Live tokenizer cannot expose its complete vocabulary."
        ) from error
    observed_specials = {
        name: getattr(tokenizer, name, None)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id")
    }
    live_contract_valid = bool(
        len(tokenizer) == contract.get("vocab_size")
        and len(normalized_vocab) == contract.get("vocab_size")
        and codebook_loader._sha256(normalized_vocab) == contract.get("vocab_sha256")
        and observed_specials == contract.get("special_token_ids")
        and contract.get("parent_token_rows_sha256") == EXACT_PARENT_TOKEN_ROWS_SHA256
    )
    parent_path = Path(args.parent_model).expanduser().resolve()
    shard_names = tuple(
        f"model-{index:05d}-of-00003.safetensors" for index in range(1, 4)
    )
    shard_paths = []
    for name in shard_names:
        path = parent_path / name
        if not path.is_file():
            raise QwenP35MLPBehaviorResidualError(
                "Exact raw parent must be the pinned local snapshot directory."
            )
        shard_paths.append(path)
    # Hash bytes, not symlink names: a filename-only or mutated-blob spoof must
    # not satisfy exact raw-parent lineage.
    shard_hashes = _verify_file_content_hashes(
        shard_paths,
        EXACT_PARENT_SHARD_CONTENT_HASHES,
        name="Raw-parent shard",
    )
    lineage = {
        "p35_manifest_sha256": codebook_loader._file_sha256(manifest_path),
        "p35_model_safetensors_sha256": (
            (manifest.get("files") or {}).get("model.safetensors")
        ),
        "p35_state_key_sha256": manifest.get("state_key_sha256"),
        "tokenizer_contract_sha256": codebook_loader._file_sha256(contract_path),
        "tokenizer_vocab_sha256": contract.get("vocab_sha256"),
        "parent_token_rows_sha256": contract.get("parent_token_rows_sha256"),
        "raw_parent_revision": args.parent_revision,
        "raw_parent_shard_content_hashes": shard_hashes,
        "checkpoint_loader_verified_manifest": True,
        "live_tokenizer_contract_verified": live_contract_valid,
        "raw_parent_content_address_verified": (
            tuple(shard_hashes) == EXACT_PARENT_SHARD_CONTENT_HASHES
            and args.parent_revision == EXACT_PARENT_REVISION
        ),
    }
    _validate_exact_lineage(lineage)
    return lineage


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fixtures = load_verified_v3_prompt_splits(
        args.fit_prompts_json,
        args.calibration_selection_prompts_json,
        args.heldout_evaluation_prompts_json,
        sealed_heldout_manifest_path=args.sealed_heldout_manifest,
        fit_manifest_path=args.fit_prompt_manifest,
        audited_manifest_path=args.audited_prompt_manifest,
        transition_inventory_path=args.transition_inventory,
    )
    try:
        import transformers
    except ImportError as error:
        raise QwenP35MLPBehaviorResidualError(
            "transformers is required for the live experiment."
        ) from error
    parent = transformers.AutoModelForCausalLM.from_pretrained(
        args.parent_model,
        revision=args.parent_revision,
        dtype=_dtype(args.dtype),
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    ).to(args.parent_device)
    parent.eval()
    child, tokenizer, source = codebook_loader._load_runtime_and_tokenizer(args)
    exact_lineage = _build_exact_lineage(args, tokenizer)
    fit_prompts = action_runner._encode_prompts(
        tokenizer, fixtures.fit, max_prompt_tokens=args.max_prompt_tokens
    )
    calibration_prompts = action_runner._encode_prompts(
        tokenizer, fixtures.calibration, max_prompt_tokens=args.max_prompt_tokens
    )
    heldout_prompts = action_runner._encode_prompts(
        tokenizer, fixtures.heldout, max_prompt_tokens=args.max_prompt_tokens
    )
    v1_development_heldout_prompts = action_runner._encode_prompts(
        tokenizer,
        fixtures.v1_development_heldout,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    v2_development_heldout_prompts = action_runner._encode_prompts(
        tokenizer,
        fixtures.v2_development_heldout,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    limits = FunctionalDecodeLimits(
        max_horizon=args.horizon,
        max_prompt_tokens=args.max_prompt_tokens,
        max_vocabulary=args.max_vocabulary,
        max_logit_bytes=args.max_logit_bytes,
        max_model_devices=1,
    )
    report = run_qwen_p35_mlp_behavior_residual(
        parent,
        child,
        fit_prompts,
        calibration_prompts,
        heldout_prompts,
        split_provenance=fixtures.provenance,
        v1_development_heldout_prompt_input_ids=(v1_development_heldout_prompts),
        v2_development_heldout_prompt_input_ids=(v2_development_heldout_prompts),
        exact_lineage=exact_lineage,
        residual_width=args.residual_width,
        horizon=args.horizon,
        steps=args.steps,
        micro_batch_size=args.micro_batch_size,
        gate_up_learning_rate=args.gate_up_learning_rate,
        down_learning_rate=args.down_learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_epsilon=args.adam_epsilon,
        hard_action_weight=args.hard_action_weight,
        distribution_weight=args.distribution_weight,
        temperature=args.temperature,
        max_gradient_norm=args.max_gradient_norm,
        calibration_steps=args.calibration_steps,
        fit_refresh_steps=args.fit_refresh_steps,
        initialization_seed=args.initialization_seed,
        trace_limits=limits,
    )
    report["source"] = source
    report["source_lineage"] = source_file_lineage()
    report["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "parent_device": args.parent_device,
        "child_device": args.device,
        "dtype": args.dtype,
        "two_model_devices_required": args.parent_device != args.device,
        "output_only_training": True,
    }
    report["report_sha256"] = codebook_loader._sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    codebook_loader._atomic_write_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "report_sha256": report["report_sha256"],
                "best_step": report["calibration_winner_lock"]["best_step"],
                "heldout_top1_delta": report["heldout_evaluation"][
                    "locked_minus_baseline"
                ]["parent_action_top1_rate"],
                "runtime_restored": report["p35_runtime_restored_exactly"],
                "checkpoint_saved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
