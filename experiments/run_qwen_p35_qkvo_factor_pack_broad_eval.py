#!/usr/bin/env python3
"""Broad, staged consumer for the exact p35 QKVO factor-only pack.

The factor pack is never trained, rewritten, baked, or promoted here.  This
consumer freshly evaluates the exact p35 checkpoint under the current strict
generation contract, derives a provenance-bound prompt-decontaminated subset
containing only p35 passes, installs the exact one-site redirect and verified
factor pack, and requires zero losses on that subset.  The expensive full
suite runs only after that gate passes.  The incumbent p35 attention runtime
is restored exactly on every success and failure path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import neural_forge.hf_qwen_attention_surgery as attention_surgery
import reproduce_pack_qwen_p35_qkvo_selected as replay_runner
import run_qwen_compensated_neuron_fold as production_runner
import run_qwen_p35_assignment_local_qkvo_residual as qkvo_runner
import run_qwen_p35_functional_residual_adapter as o_runner
import run_qwen_p35_vo_role_scale_search as role_search
import run_qwen_residual_attention_probe as attention_probe
import run_qwen_unified_surgery as unified_runner
from neural_forge.hf_assignment_local_qkvo_residual_adapter import (
    InstalledAssignmentLocalQKVOResidualAdapter,
    install_assignment_local_qkvo_residual_adapter,
)
from neural_forge.hf_qkvo_factor_pack import (
    LoadedQKVOFactorPack,
    file_sha256,
    install_qkvo_factor_pack,
    load_qkvo_factor_pack,
    tensor_sha256,
)
from neural_forge.hf_qwen_attention_surgery import (
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import qwen3_codebook_parameter_report
from neural_forge.hf_tools import _generation_health, _load_eval_suite


RUN_SCHEMA = "neural-forge-qwen-p35-qkvo-factor-pack-broad-eval-v1"
DECONTAMINATION_SCHEMA = (
    "neural-forge-qwen-p35-qkvo-prompt-decontaminated-parent-pass-v1"
)
PROJECTION_SCHEMA = "neural-forge-projected-current-parent-report-v1"

EXACT_BROAD_SUITE_FILE_SHA256 = (
    "fa6066980faf4677b194e00951076dc477cc4243d185449aa6590ca4c8768166"
)
EXACT_RAW_QWEN_REPORT_FILE_SHA256 = (
    "3333bc54e9170dc8a65e8a53179c9a20a2ab7d1931e2da5795f6bc5ecf031395"
)
EXACT_MAX_NEW_TOKENS = 96
EXACT_P35_PARAMETER_COUNT = 3_995_570_176
EXACT_CANDIDATE_PARAMETER_COUNT = 3_992_386_688
EXACT_PARAMETERS_SAVED = 3_183_488
EXACT_RAW_QWEN_PARAMETER_COUNT = 4_022_468_096
EXACT_RAW_QWEN_PPL = 877.5796540017018

EXACT_FULL_COUNTS = {
    "text_evals": 128,
    "choice_probes": 250,
    "generation_prompts": 1000,
    "behavior_probes": 1000,
}
EXACT_RAW_PARENT_PASS_COUNTS = {
    "behavior": 400,
    "choice": 250,
    "generation": 388,
}
EXACT_RAW_PARENT_PASS_ID_SHA256 = {
    "behavior": "98fe6b188b78b167692fc83f97f77e93e06aa751095981a4befbca583ab563ad",
    "choice": "317bb1e8815eef3cb0a0ef6f4266c0405c31d7b3aad57b601d387dba2231abaf",
    "generation": "485790de23bb36bbde214cfeef7e47ab141ef0af93cbdec867c41b3675cf5d0e",
}
EXACT_CURRENT_P35_PASS_COUNTS = {
    "behavior": 400,
    "choice": 250,
    "generation": 546,
}
EXACT_CURRENT_P35_PASS_ID_SHA256 = {
    "behavior": "98fe6b188b78b167692fc83f97f77e93e06aa751095981a4befbca583ab563ad",
    "choice": "317bb1e8815eef3cb0a0ef6f4266c0405c31d7b3aad57b601d387dba2231abaf",
    "generation": "9335f76cd79167ee92328b320e7e432260b57346e7c5ac11fc6587ed1d5949b0",
}
EXACT_DECONTAMINATED_COUNTS = {
    "behavior": 333,
    "choice": 250,
    "generation": 388,
}
EXACT_DECONTAMINATED_PASS_ID_SHA256 = {
    "behavior": "2c528c1ed36196e00f2c422eb135eb91dcc9a27441948b5092709b12a6d47ae4",
    "choice": "317bb1e8815eef3cb0a0ef6f4266c0405c31d7b3aad57b601d387dba2231abaf",
    "generation": "6b8e537bb44cb7b7a39d4489264fdbe6b158de7a08adadf977081db262eccfae",
}
EXACT_DECONTAMINATED_SUITE_SHA256 = (
    "51f65b3f11dd402b6b27fd39657d04cfb6cb73b417490a226b2d4088a971aa00"
)
EXACT_PROMPT_OVERLAP_COUNTS = {
    "behavior": {"all": 67, "p35_pass": 67, "exact_id": 0},
    "choice": {"all": 0, "p35_pass": 0, "exact_id": 0},
    "generation": {"all": 163, "p35_pass": 158, "exact_id": 22},
}
EXACT_RAW_TO_CURRENT_P35_DELTAS = {
    "behavior": {"loss_count": 0, "gain_count": 0},
    "choice": {"loss_count": 0, "gain_count": 0},
    "generation": {"loss_count": 16, "gain_count": 174},
}

DOMAIN_COLLECTIONS = {
    "behavior": ("behavior_probes", "behavior_suite", "passed"),
    "choice": ("choice_probes", "choice_accuracy", "correct"),
    "generation": ("generation_prompts", "generation_diagnostics", "passed"),
}


class QwenP35QKVOFactorPackBroadEvalError(RuntimeError):
    """A broad-evaluation identity, gate, or rollback contract failed."""


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Evaluation provenance is not canonical JSON."
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _ordered_ids_sha256(values: Sequence[str]) -> str:
    return _canonical_sha256([str(value) for value in values])


def _exclusive_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, resolved)
        except FileExistsError as error:
            raise QwenP35QKVOFactorPackBroadEvalError(
                f"Evaluation JSON already exists: {resolved}."
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QwenP35QKVOFactorPackBroadEvalError(
            f"{label} is not valid UTF-8 JSON."
        ) from error
    if not isinstance(payload, dict):
        raise QwenP35QKVOFactorPackBroadEvalError(f"{label} must be an object.")
    return payload


def _report_items(report: Mapping[str, Any], domain: str) -> list[dict[str, Any]]:
    if domain == "generation":
        raw = report.get("generation_diagnostics")
    else:
        raw = (report.get(DOMAIN_COLLECTIONS[domain][1]) or {}).get("items")
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise QwenP35QKVOFactorPackBroadEvalError(
            f"Evaluation report lacks {domain} item rows."
        )
    return [dict(item) for item in raw]


def _report_item_map(
    report: Mapping[str, Any], domain: str
) -> dict[str, dict[str, Any]]:
    rows = _report_items(report, domain)
    result = {
        str(item["id"]): item
        for item in rows
        if isinstance(item.get("id"), str) and item["id"]
    }
    if len(result) != len(rows):
        raise QwenP35QKVOFactorPackBroadEvalError(
            f"Evaluation report contains malformed or duplicate {domain} IDs."
        )
    return result


def _pass_ids(report: Mapping[str, Any], domain: str) -> list[str]:
    key = DOMAIN_COLLECTIONS[domain][2]
    return [
        str(item["id"])
        for item in _report_items(report, domain)
        if bool(item.get(key, False))
    ]


def _assert_report_ids(
    report: Mapping[str, Any], suite: Mapping[str, Any], *, label: str
) -> None:
    for domain, (collection, _report_key, _pass_key) in DOMAIN_COLLECTIONS.items():
        expected = [str(item["id"]) for item in suite[collection]]
        observed = [str(item["id"]) for item in _report_items(report, domain)]
        if observed != expected:
            raise QwenP35QKVOFactorPackBroadEvalError(
                f"{label} {domain} ordered IDs differ from the exact suite."
            )


def _assert_pass_contract(
    report: Mapping[str, Any],
    *,
    expected_counts: Mapping[str, int],
    expected_id_sha256: Mapping[str, str],
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for domain in DOMAIN_COLLECTIONS:
        identifiers = _pass_ids(report, domain)
        digest = _ordered_ids_sha256(identifiers)
        if (
            len(identifiers) != int(expected_counts[domain])
            or digest != expected_id_sha256[domain]
        ):
            raise QwenP35QKVOFactorPackBroadEvalError(
                f"{label} {domain} pass set differs from the exact contract."
            )
        result[domain] = {"count": len(identifiers), "ordered_ids_sha256": digest}
    return result


def load_verified_broad_suite(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or file_sha256(resolved) != (
        EXACT_BROAD_SUITE_FILE_SHA256
    ):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Broad-suite file SHA-256 is not exact."
        )
    suite = _load_eval_suite(resolved)
    counts = {
        name: len(suite[name])
        for name in (
            "text_evals",
            "choice_probes",
            "generation_prompts",
            "behavior_probes",
        )
    }
    if counts != EXACT_FULL_COUNTS:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Broad-suite cardinality differs from 128/250/1000/1000."
        )
    return suite


def load_verified_raw_qwen_parent_report(
    path: Path, suite: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file() or file_sha256(resolved) != (
        EXACT_RAW_QWEN_REPORT_FILE_SHA256
    ):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Raw-Qwen strict report SHA-256 is not exact."
        )
    report = _read_json_mapping(resolved, "Raw-Qwen strict report")
    if (
        report.get("model") != "Qwen/Qwen3-4B-Base"
        or report.get("param_count") != EXACT_RAW_QWEN_PARAMETER_COUNT
        or float((report.get("perplexity") or {}).get("mean", math.nan))
        != EXACT_RAW_QWEN_PPL
    ):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Raw-Qwen report identity or PPL changed."
        )
    _assert_report_ids(report, suite, label="Raw-Qwen report")
    generations = report.get("generations")
    if not isinstance(generations, list) or len(generations) != 1000:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Raw-Qwen report lacks all saved generations."
        )
    strict = [
        _generation_health(
            dict(item),
            production_runner.GENERATION_HEALTH_CONTRACT["min_continuation_chars"],
            production_runner.GENERATION_HEALTH_CONTRACT["min_unique_token_ratio"],
        )
        for item in generations
    ]
    legacy = _report_items(report, "generation")
    if [bool(item["passed"]) for item in strict] != [
        bool(item["passed"]) for item in legacy
    ]:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Raw-Qwen legacy report differs under the current strict contract."
        )
    current = dict(report)
    current["generation_diagnostics"] = strict
    current["generation_health"] = {
        "correct": sum(bool(item["passed"]) for item in strict),
        "total": len(strict),
        "pass_rate": sum(bool(item["passed"]) for item in strict) / len(strict),
        "contract": dict(production_runner.GENERATION_HEALTH_CONTRACT),
    }
    pass_contract = _assert_pass_contract(
        current,
        expected_counts=EXACT_RAW_PARENT_PASS_COUNTS,
        expected_id_sha256=EXACT_RAW_PARENT_PASS_ID_SHA256,
        label="Raw-Qwen current-contract replay",
    )
    audit = {
        "source_file": str(resolved),
        "source_file_sha256": EXACT_RAW_QWEN_REPORT_FILE_SHA256,
        "legacy_generation_contract_was_null": (
            (report.get("generation_health") or {}).get("contract") is None
        ),
        "current_strict_replay_changed_item_count": 0,
        "pass_contract": pass_contract,
        "perplexity_mean": EXACT_RAW_QWEN_PPL,
    }
    audit["audit_sha256"] = _canonical_sha256(audit)
    return current, audit


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    try:
        encoded = tokenizer(text, add_special_tokens=True)
        raw = encoded["input_ids"]
    except Exception as error:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Tokenizer could not encode a provenance prompt."
        ) from error
    if isinstance(raw, Tensor):
        raw = raw.detach().cpu().tolist()
    if (
        isinstance(raw, Sequence)
        and len(raw) == 1
        and isinstance(raw[0], Sequence)
        and not isinstance(raw[0], (str, bytes))
    ):
        raw = raw[0]
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or not raw
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
    ):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Tokenizer returned malformed prompt input IDs."
        )
    return tuple(int(value) for value in raw)


def _prompt_token_sha256(tokenizer: Any, prompt: str) -> str:
    return _canonical_sha256({"input_ids": list(_token_ids(tokenizer, prompt))})


def load_verified_fixture_rows(
    fixtures: Any,
    *,
    fit_manifest_path: Path,
    audited_manifest_path: Path,
) -> tuple[dict[str, Any], ...]:
    fit_manifest = _read_json_mapping(fit_manifest_path, "Fit prompt manifest")
    audited = _read_json_mapping(
        audited_manifest_path, "Audited calibration/heldout prompt manifest"
    )
    try:
        partitions = (
            ("fit", tuple(fixtures.fit), fit_manifest["entries"]),
            (
                "calibration-selection",
                tuple(fixtures.calibration),
                audited["sets"]["calibration-selection"]["entries"],
            ),
            (
                "heldout-evaluation",
                tuple(fixtures.heldout),
                audited["sets"]["heldout-evaluation"]["entries"],
            ),
        )
    except (AttributeError, KeyError, TypeError) as error:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Verified fixture manifests lack their exact prompt rows."
        ) from error
    rows: list[dict[str, Any]] = []
    for split, prompts, entries in partitions:
        if not isinstance(entries, list) or [
            entry.get("prompt") for entry in entries
        ] != list(prompts):
            raise QwenP35QKVOFactorPackBroadEvalError(
                f"{split} fixture rows differ after provenance verification."
            )
        for entry in entries:
            if not isinstance(entry.get("id"), str):
                raise QwenP35QKVOFactorPackBroadEvalError(
                    f"{split} fixture contains a malformed case ID."
                )
            rows.append({**dict(entry), "split": split})
    identifiers = [str(row["id"]) for row in rows]
    prompts = [str(row["prompt"]) for row in rows]
    if len(rows) != 22 or len(set(identifiers)) != 22 or len(set(prompts)) != 22:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Factor-pack fit/calibration/heldout provenance is not 22 unique rows."
        )
    return tuple(rows)


def verify_fresh_p35_report(
    report: Mapping[str, Any], suite: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        report.get("param_count") != EXACT_P35_PARAMETER_COUNT
        or (report.get("generation_health") or {}).get("contract")
        != production_runner.GENERATION_HEALTH_CONTRACT
    ):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Fresh p35 report lacks the exact parameter or strict-generation contract."
        )
    evaluation = report.get("eval_suite") or {}
    if {
        "text_evals": evaluation.get("text_eval_count"),
        "choice_probes": evaluation.get("choice_probe_count"),
        "generation_prompts": evaluation.get("generation_prompt_count"),
        "behavior_probes": evaluation.get("behavior_probe_count"),
    } != EXACT_FULL_COUNTS:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Fresh p35 report did not score the exact full broad suite."
        )
    _assert_report_ids(report, suite, label="Fresh p35 report")
    contract = _assert_pass_contract(
        report,
        expected_counts=EXACT_CURRENT_P35_PASS_COUNTS,
        expected_id_sha256=EXACT_CURRENT_P35_PASS_ID_SHA256,
        label="Fresh current-contract p35",
    )
    perplexity = report.get("perplexity") or {}
    items = perplexity.get("items")
    mean = perplexity.get("mean")
    if (
        not isinstance(items, list)
        or len(items) != 128
        or isinstance(mean, bool)
        or not isinstance(mean, (int, float))
        or not math.isfinite(float(mean))
        or float(mean) <= 0.0
    ):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Fresh p35 report has invalid 128-row perplexity."
        )
    return {
        "model": report.get("model"),
        "param_count": report.get("param_count"),
        "pass_contract": contract,
        "perplexity_mean": float(mean),
        "strict_generation_contract": dict(
            production_runner.GENERATION_HEALTH_CONTRACT
        ),
        "evaluation_path_provenance": report.get("evaluation_path_provenance"),
    }


def derive_prompt_decontaminated_parent_pass_suite(
    tokenizer: Any,
    full_suite: Mapping[str, Any],
    p35_report: Mapping[str, Any],
    fixture_rows: Sequence[Mapping[str, Any]],
    *,
    fixture_provenance_sha256: str,
    enforce_exact_contract: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture_prompts = [str(row["prompt"]) for row in fixture_rows]
    fixture_ids = [str(row["id"]) for row in fixture_rows]
    if (
        len(fixture_prompts) != 22
        or len(set(fixture_prompts)) != 22
        or len(set(fixture_ids)) != 22
    ):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Decontamination requires the exact 22 unique fixture rows."
        )
    fixture_prompt_set = set(fixture_prompts)
    fixture_id_set = set(fixture_ids)
    fixture_token_sha256s = [
        _prompt_token_sha256(tokenizer, prompt) for prompt in fixture_prompts
    ]
    if len(set(fixture_token_sha256s)) != 22:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Fixture prompts collide after tokenization."
        )
    fixture_token_set = set(fixture_token_sha256s)

    subset: dict[str, Any] = {"text_evals": []}
    domains: dict[str, Any] = {}
    for domain, (collection, _report_key, pass_key) in DOMAIN_COLLECTIONS.items():
        rows = [dict(item) for item in full_suite[collection]]
        report_map = _report_item_map(p35_report, domain)
        expected_ids = [str(row["id"]) for row in rows]
        if set(report_map) != set(expected_ids):
            raise QwenP35QKVOFactorPackBroadEvalError(
                f"Fresh p35 {domain} rows differ from the broad suite."
            )
        selected: list[dict[str, Any]] = []
        exact_prompt_overlap_ids: list[str] = []
        token_prompt_overlap_ids: list[str] = []
        p35_pass_overlap_ids: list[str] = []
        exact_id_overlap_ids: list[str] = []
        for row in rows:
            item_id = str(row["id"])
            prompt = str(row["prompt"])
            token_sha = _prompt_token_sha256(tokenizer, prompt)
            exact_prompt_overlap = prompt in fixture_prompt_set
            token_prompt_overlap = token_sha in fixture_token_set
            if exact_prompt_overlap:
                exact_prompt_overlap_ids.append(item_id)
            if token_prompt_overlap:
                token_prompt_overlap_ids.append(item_id)
            if item_id in fixture_id_set:
                exact_id_overlap_ids.append(item_id)
            parent_passed = bool(report_map[item_id].get(pass_key, False))
            if parent_passed and token_prompt_overlap:
                p35_pass_overlap_ids.append(item_id)
            if parent_passed and not token_prompt_overlap:
                selected.append(row)
        if exact_prompt_overlap_ids != token_prompt_overlap_ids:
            raise QwenP35QKVOFactorPackBroadEvalError(
                f"{domain} exact-text and exact-token prompt overlap differ."
            )
        selected_ids = [str(row["id"]) for row in selected]
        subset[collection] = selected
        domains[domain] = {
            "full_item_count": len(rows),
            "full_p35_pass_count": sum(
                bool(report_map[item_id].get(pass_key, False))
                for item_id in expected_ids
            ),
            "fixture_exact_id_overlap_count": len(exact_id_overlap_ids),
            "fixture_exact_id_overlap_ids": exact_id_overlap_ids,
            "fixture_exact_prompt_overlap_count": len(exact_prompt_overlap_ids),
            "fixture_token_prompt_overlap_count": len(token_prompt_overlap_ids),
            "fixture_prompt_overlap_ids": token_prompt_overlap_ids,
            "fixture_prompt_p35_pass_overlap_count": len(p35_pass_overlap_ids),
            "fixture_prompt_p35_pass_overlap_ids": p35_pass_overlap_ids,
            "retained_p35_pass_count": len(selected_ids),
            "retained_p35_pass_ids_sha256": _ordered_ids_sha256(selected_ids),
        }

    text_exact_overlap = [
        index
        for index, value in enumerate(full_suite["text_evals"])
        if str(value) in fixture_prompt_set
    ]
    text_token_overlap = [
        index
        for index, value in enumerate(full_suite["text_evals"])
        if _prompt_token_sha256(tokenizer, str(value)) in fixture_token_set
    ]
    if text_exact_overlap != text_token_overlap:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "PPL exact-text and exact-token fixture overlap differ."
        )
    subset_sha256 = _canonical_sha256(subset)
    audit: dict[str, Any] = {
        "schema": DECONTAMINATION_SCHEMA,
        "method": "exclude-any-exact-tokenized-fixture-prompt-then-keep-p35-passes",
        "fixture_provenance_sha256": str(fixture_provenance_sha256),
        "fixture_row_count": len(fixture_rows),
        "fixture_ids": fixture_ids,
        "fixture_ids_sha256": _ordered_ids_sha256(fixture_ids),
        "fixture_prompt_text_sha256s": [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for prompt in fixture_prompts
        ],
        "fixture_prompt_token_sha256s": fixture_token_sha256s,
        "text_eval_overlap_indices": text_token_overlap,
        "domains": domains,
        "subset_counts": {
            domain: len(subset[DOMAIN_COLLECTIONS[domain][0]])
            for domain in DOMAIN_COLLECTIONS
        },
        "subset_suite_sha256": subset_sha256,
        "full_suite_modified": False,
        "fit_calibration_heldout_used_for_exclusion_only": True,
        "candidate_outputs_used_to_build_subset": False,
    }
    if enforce_exact_contract:
        if audit["subset_counts"] != EXACT_DECONTAMINATED_COUNTS:
            raise QwenP35QKVOFactorPackBroadEvalError(
                "Decontaminated p35-pass counts differ from 333/250/388."
            )
        if subset_sha256 != EXACT_DECONTAMINATED_SUITE_SHA256:
            raise QwenP35QKVOFactorPackBroadEvalError(
                "Decontaminated suite content SHA-256 changed."
            )
        for domain, expected in EXACT_DECONTAMINATED_PASS_ID_SHA256.items():
            if domains[domain]["retained_p35_pass_ids_sha256"] != expected:
                raise QwenP35QKVOFactorPackBroadEvalError(
                    f"Decontaminated {domain} ordered pass IDs changed."
                )
            overlap = EXACT_PROMPT_OVERLAP_COUNTS[domain]
            if (
                domains[domain]["fixture_exact_prompt_overlap_count"] != overlap["all"]
                or domains[domain]["fixture_prompt_p35_pass_overlap_count"]
                != overlap["p35_pass"]
                or domains[domain]["fixture_exact_id_overlap_count"]
                != overlap["exact_id"]
            ):
                raise QwenP35QKVOFactorPackBroadEvalError(
                    f"Decontamination overlap audit changed for {domain}."
                )
    audit["audit_sha256"] = _canonical_sha256(audit)
    return subset, audit


def project_parent_report_to_suite(
    parent: Mapping[str, Any],
    subset: Mapping[str, Any],
    *,
    source_report_sha256: str,
    decontamination_audit_sha256: str,
) -> dict[str, Any]:
    selected_ids = {
        domain: [str(row["id"]) for row in subset[collection]]
        for domain, (collection, _report_key, _pass_key) in DOMAIN_COLLECTIONS.items()
    }
    maps = {domain: _report_item_map(parent, domain) for domain in DOMAIN_COLLECTIONS}
    behavior = [maps["behavior"][value] for value in selected_ids["behavior"]]
    choices = [maps["choice"][value] for value in selected_ids["choice"]]
    diagnostics = [maps["generation"][value] for value in selected_ids["generation"]]
    generation_map = {
        str(item["id"]): dict(item)
        for item in parent.get("generations", [])
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if any(value not in generation_map for value in selected_ids["generation"]):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Fresh p35 report lacks generation payloads for its projected passes."
        )
    generations = [generation_map[value] for value in selected_ids["generation"]]
    if (
        not all(bool(item.get("passed")) for item in behavior)
        or not all(bool(item.get("correct")) for item in choices)
        or not all(bool(item.get("passed")) for item in diagnostics)
    ):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Projected decontaminated parent contains a non-pass row."
        )
    result = {
        "model": parent.get("model"),
        "param_count": parent.get("param_count"),
        "eval_suite": {
            "source": "projected-fresh-current-contract-p35",
            "text_eval_count": 0,
            "choice_probe_count": len(choices),
            "generation_prompt_count": len(generations),
            "behavior_probe_count": len(behavior),
        },
        "perplexity": {"mean": 0.0, "items": []},
        "choice_accuracy": {
            "correct": len(choices),
            "total": len(choices),
            "accuracy": 1.0,
            "items": choices,
        },
        "generations": generations,
        "generation_diagnostics": diagnostics,
        "generation_health": {
            "correct": len(diagnostics),
            "total": len(diagnostics),
            "pass_rate": 1.0,
            "contract": dict(production_runner.GENERATION_HEALTH_CONTRACT),
        },
        "behavior_suite": {
            "correct": len(behavior),
            "total": len(behavior),
            "pass_rate": 1.0,
            "items": behavior,
        },
        "projection_provenance": {
            "schema": PROJECTION_SCHEMA,
            "source_report_sha256": str(source_report_sha256),
            "decontamination_audit_sha256": str(decontamination_audit_sha256),
            "ordered_ids_sha256": {
                domain: _ordered_ids_sha256(values)
                for domain, values in selected_ids.items()
            },
            "no_model_forward_performed_for_projection": True,
        },
    }
    result["projection_provenance"]["projection_sha256"] = _canonical_sha256(
        {key: value for key, value in result.items() if key != "projection_provenance"}
    )
    return result


def _report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": report.get("model"),
        "param_count": report.get("param_count"),
        "perplexity_mean": float((report.get("perplexity") or {}).get("mean", 0.0)),
        "behavior_correct": int((report.get("behavior_suite") or {}).get("correct", 0)),
        "behavior_total": int((report.get("behavior_suite") or {}).get("total", 0)),
        "choice_correct": int((report.get("choice_accuracy") or {}).get("correct", 0)),
        "choice_total": int((report.get("choice_accuracy") or {}).get("total", 0)),
        "generation_correct": int(
            (report.get("generation_health") or {}).get("correct", 0)
        ),
        "generation_total": int(
            (report.get("generation_health") or {}).get("total", 0)
        ),
    }


def _ppl_ratio(candidate: Mapping[str, Any], parent: Mapping[str, Any]) -> float:
    numerator = float((candidate.get("perplexity") or {})["mean"])
    denominator = float((parent.get("perplexity") or {})["mean"])
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or numerator <= 0.0
        or denominator <= 0.0
    ):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Full evaluation produced invalid perplexity."
        )
    return numerator / denominator


def _zero_loss_gate(
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    suite: unified_runner.BoundEvaluationSuite,
) -> dict[str, Any]:
    return production_runner.score_cached_parent_gate(
        parent,
        candidate,
        behavior_ids=suite.behavior_ids,
        choice_ids=suite.choice_ids,
        generation_ids=suite.generation_ids,
        thresholds=production_runner.GateThresholds(),
    )


def _assert_raw_to_current_p35_comparison(verdict: Mapping[str, Any]) -> None:
    domains = verdict.get("domains")
    if not isinstance(domains, Mapping):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Raw-Qwen/current-p35 comparison has no domain rows."
        )
    for domain, expected in EXACT_RAW_TO_CURRENT_P35_DELTAS.items():
        observed = domains.get(domain)
        if not isinstance(observed, Mapping) or any(
            observed.get(name) != value for name, value in expected.items()
        ):
            raise QwenP35QKVOFactorPackBroadEvalError(
                f"Raw-Qwen/current-p35 {domain} losses or gains changed."
            )


def run_staged_candidate_evaluation(
    runtime: nn.Module,
    *,
    executor: unified_runner.SuiteExecutor,
    full_suite: unified_runner.BoundEvaluationSuite,
    decontaminated_suite: unified_runner.BoundEvaluationSuite,
    p35_full_report: Mapping[str, Any],
    p35_projected_report: Mapping[str, Any],
    raw_qwen_report: Mapping[str, Any],
    output_directory: Path,
    child_unique_parameters: int,
) -> dict[str, Any]:
    decontaminated_path = output_directory / "candidate_decontaminated.json"
    candidate_decontaminated = executor.evaluate(
        runtime,
        decontaminated_suite,
        decontaminated_path,
        label="p35-qkvo-factor-pack:decontaminated-candidate",
        child_unique_parameters=child_unique_parameters,
    )
    gate = _zero_loss_gate(
        p35_projected_report, candidate_decontaminated, decontaminated_suite
    )
    stage: dict[str, Any] = {
        "schema": "neural-forge-staged-broad-factor-pack-evaluation-v1",
        "decontaminated": {
            "candidate_report": str(decontaminated_path.resolve()),
            "candidate_report_sha256": file_sha256(decontaminated_path),
            "candidate_summary": _report_summary(candidate_decontaminated),
            "zero_loss_gate": gate,
            "used_for_acceptance": True,
        },
        "full": {
            "executed": False,
            "reason": "decontaminated-zero-loss-gate-failed",
            "used_for_acceptance": False,
        },
        "promotion_gate_passed": bool(gate["passed"]),
    }
    if not gate["passed"]:
        return stage

    full_path = output_directory / "candidate_full_broad.json"
    candidate_full = executor.evaluate(
        runtime,
        full_suite,
        full_path,
        label="p35-qkvo-factor-pack:full-broad-candidate",
        child_unique_parameters=child_unique_parameters,
    )
    p35_comparison = _zero_loss_gate(p35_full_report, candidate_full, full_suite)
    raw_candidate = _zero_loss_gate(raw_qwen_report, candidate_full, full_suite)
    raw_p35 = _zero_loss_gate(raw_qwen_report, p35_full_report, full_suite)
    stage["full"] = {
        "executed": True,
        "reason": "decontaminated-zero-loss-gate-passed",
        "candidate_report": str(full_path.resolve()),
        "candidate_report_sha256": file_sha256(full_path),
        "candidate_summary": _report_summary(candidate_full),
        "p35_parent_relative_zero_loss_comparison": p35_comparison,
        "raw_qwen_parent_pass_retention_by_p35": raw_p35,
        "raw_qwen_parent_pass_retention_by_candidate": raw_candidate,
        "perplexity_ratios": {
            "candidate_over_current_p35": _ppl_ratio(candidate_full, p35_full_report),
            "candidate_over_raw_qwen": _ppl_ratio(candidate_full, raw_qwen_report),
            "current_p35_over_raw_qwen": _ppl_ratio(p35_full_report, raw_qwen_report),
        },
        "full_p35_zero_loss_passed": bool(p35_comparison["passed"]),
        "used_for_acceptance": False,
        "not_independent_because_fixture_prompt_overlap_is_nonzero": True,
    }
    return stage


def _verify_factor_pack_candidate_contract(
    loaded: LoadedQKVOFactorPack,
    *,
    fixtures: Any,
    source_runtime: Mapping[str, Any],
    proposed_registry: nn.Module,
    site: Any,
) -> dict[str, Any]:
    manifest = loaded.manifest
    provenance = manifest.get("provenance")
    role_provenance = manifest.get("role_scale_provenance")
    if not isinstance(provenance, Mapping) or not isinstance(role_provenance, Mapping):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Factor pack lacks source or role-scale provenance."
        )
    source = provenance.get("source_runtime")
    fixture_sha = fixtures.provenance.get("provenance_sha256")
    if (
        manifest.get("selected_adapter_state_sha256")
        != replay_runner.EXPECTED_SELECTED_ADAPTER_STATE_SHA256
        or manifest.get("factor_parameter_count") != 92_160
        or manifest.get("rank") != 8
        or provenance.get("fixture_provenance_sha256") != fixture_sha
        or not isinstance(source, Mapping)
        or source.get("manifest_sha256")
        != replay_runner.EXPECTED_CODEBOOK_MANIFEST_SHA256
        or source_runtime.get("manifest_sha256")
        != replay_runner.EXPECTED_CODEBOOK_MANIFEST_SHA256
    ):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Factor pack is not bound to the exact p35 replay and fixtures."
        )
    audit = o_runner._incumbent_role_scale_audit(
        proposed_registry,
        redirect=qkvo_runner.EXACT_REDIRECT,
        require_exact_p35=True,
    )
    expected_site = {
        **site.as_dict(),
        "redirect": qkvo_runner.EXACT_REDIRECT.as_dict(),
        "canonical_assignment_axis": audit["canonical_assignment_axis"],
    }
    expected_roles = replay_runner._role_scale_provenance(
        proposed_registry, {"incumbent_role_scale_audit": audit}
    )
    if (
        manifest.get("site") != expected_site
        or manifest.get("registry_geometry")
        != replay_runner._registry_geometry(proposed_registry)
        or role_provenance != expected_roles
        or role_provenance.get("table_sha256")
        != tensor_sha256(proposed_registry.role_scales)
    ):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Factor pack redirect, geometry, or role scales differ from p35."
        )
    return {
        "manifest_sha256": manifest.get("manifest_sha256"),
        "factor_file_sha256": manifest.get("factor_file_sha256"),
        "selected_adapter_state_sha256": manifest.get("selected_adapter_state_sha256"),
        "factor_parameter_count": manifest.get("factor_parameter_count"),
        "factor_tensor_bytes": manifest.get("factor_tensor_bytes"),
        "fixture_provenance_sha256": fixture_sha,
        "site": expected_site,
        "registry_geometry": manifest.get("registry_geometry"),
        "role_scale_provenance": expected_roles,
    }


def _restore_after_baseline_failure(
    runtime: nn.Module,
    *,
    parent_installation: Any,
    parent_snapshot: Any,
    parent_identity: str,
    immutable_input_hashes: Mapping[Path, str],
    primary_error: BaseException,
) -> None:
    cleanup_error: Exception | None = None
    try:
        if not o_runner._restore_parent_exact(
            runtime,
            parent_installation,
            parent_snapshot,
            parent_identity,
        ):
            cleanup_error = QwenP35QKVOFactorPackBroadEvalError(
                "Baseline failure did not restore p35 exactly."
            )
    except Exception as error:  # pragma: no cover - hardware cleanup path
        cleanup_error = error
    try:
        immutable = all(
            path.is_file() and file_sha256(path) == digest
            for path, digest in immutable_input_hashes.items()
        )
        if not immutable and cleanup_error is None:
            cleanup_error = QwenP35QKVOFactorPackBroadEvalError(
                "Factor pack changed during failed baseline evaluation."
            )
    except Exception as error:  # pragma: no cover - filesystem failure
        if cleanup_error is None:
            cleanup_error = error
    if cleanup_error is not None:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Fresh p35 baseline evaluation failed and cleanup also failed; "
            f"primary={type(primary_error).__name__}: {primary_error}; "
            f"cleanup={type(cleanup_error).__name__}: {cleanup_error}"
        ) from cleanup_error


def consume_factor_pack_broad(
    runtime: nn.Module,
    tokenizer: Any,
    *,
    source_runtime: Mapping[str, Any],
    fixtures: Any,
    fixture_rows: Sequence[Mapping[str, Any]],
    loaded_factor_pack: LoadedQKVOFactorPack,
    factor_path: Path,
    manifest_path: Path,
    full_suite_payload: Mapping[str, Any],
    full_suite: unified_runner.BoundEvaluationSuite,
    raw_qwen_report: Mapping[str, Any],
    raw_qwen_audit: Mapping[str, Any],
    executor: unified_runner.SuiteExecutor,
    output_directory: Path,
) -> dict[str, Any]:
    factor_path = factor_path.resolve()
    manifest_path = manifest_path.resolve()
    immutable_input_hashes = {
        factor_path: file_sha256(factor_path),
        manifest_path: file_sha256(manifest_path),
    }
    if immutable_input_hashes[factor_path] != loaded_factor_pack.manifest.get(
        "factor_file_sha256"
    ) or _read_json_mapping(manifest_path, "Factor manifest") != dict(
        loaded_factor_pack.manifest
    ):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Factor pack changed after its verified load."
        )
    parent_registry = attention_probe._registry(runtime)
    parent_snapshot = qwen3_attention_runtime_snapshot(runtime)
    parent_identity = role_search._runtime_state_identity_sha256(runtime)
    parent_installation = attention_surgery._capture_installed(runtime)
    parent_parameters = qwen3_codebook_parameter_report(runtime).unique_parameter_count
    if parent_parameters != EXACT_P35_PARAMETER_COUNT:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Loaded p35 unique parameter count is not exact."
        )

    p35_path = output_directory / "p35_current_strict_full.json"
    try:
        p35_report = executor.evaluate(
            runtime,
            full_suite,
            p35_path,
            label="p35-current-parent:fresh-current-strict-full-broad",
            child_unique_parameters=parent_parameters,
        )
        p35_summary = verify_fresh_p35_report(p35_report, full_suite_payload)
        p35_report_sha256 = file_sha256(p35_path)
        if (
            qwen3_attention_runtime_snapshot(runtime) != parent_snapshot
            or role_search._runtime_state_identity_sha256(runtime) != parent_identity
        ):
            raise QwenP35QKVOFactorPackBroadEvalError(
                "Fresh p35 baseline evaluation changed the incumbent runtime."
            )
    except BaseException as error:
        _restore_after_baseline_failure(
            runtime,
            parent_installation=parent_installation,
            parent_snapshot=parent_snapshot,
            parent_identity=parent_identity,
            immutable_input_hashes=immutable_input_hashes,
            primary_error=error,
        )
        raise

    subset_payload, decontamination = derive_prompt_decontaminated_parent_pass_suite(
        tokenizer,
        full_suite_payload,
        p35_report,
        fixture_rows,
        fixture_provenance_sha256=str(fixtures.provenance["provenance_sha256"]),
    )
    subset_path = output_directory / "decontaminated_parent_pass_suite.json"
    provenance_path = output_directory / "decontamination_provenance.json"
    _exclusive_atomic_json(subset_path, subset_payload)
    _exclusive_atomic_json(provenance_path, decontamination)
    decontaminated_suite = unified_runner.bind_evaluation_suite(
        tokenizer,
        subset_payload,
        name="p35-qkvo-prompt-decontaminated-parent-pass",
        max_new_tokens=EXACT_MAX_NEW_TOKENS,
    )
    if decontaminated_suite.suite_sha256 != EXACT_DECONTAMINATED_SUITE_SHA256:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Bound decontaminated suite SHA-256 differs from its content proof."
        )
    projected = project_parent_report_to_suite(
        p35_report,
        subset_payload,
        source_report_sha256=p35_report_sha256,
        decontamination_audit_sha256=str(decontamination["audit_sha256"]),
    )
    projected_path = output_directory / "p35_decontaminated_projection.json"
    _exclusive_atomic_json(projected_path, projected)

    raw_to_p35 = _zero_loss_gate(raw_qwen_report, p35_report, full_suite)
    _assert_raw_to_current_p35_comparison(raw_to_p35)
    _bridge, _proposal, _candidates, tentative = attention_probe._tentative_codebook(
        runtime, (qkvo_runner.EXACT_REDIRECT,)
    )
    codebook, _used_projections = tentative
    proposed_registry = codebook.registry
    site = o_runner._redirected_site(parent_registry, qkvo_runner.EXACT_REDIRECT)
    factor_contract = _verify_factor_pack_candidate_contract(
        loaded_factor_pack,
        fixtures=fixtures,
        source_runtime=source_runtime,
        proposed_registry=proposed_registry,
        site=site,
    )

    adapter_handle: InstalledAssignmentLocalQKVOResidualAdapter | None = None
    staged: dict[str, Any] | None = None
    installation: dict[str, Any] | None = None
    installed_state_sha256: str | None = None
    cleanup_error: Exception | None = None
    restored = False
    pack_immutable = False
    primary_error: BaseException | None = None
    try:
        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        adapter_handle = install_assignment_local_qkvo_residual_adapter(
            runtime,
            proposed_registry,
            sites=(site,),
            rank=8,
            initialization_seed=0,
            require_qwen3_4b_dimensions=True,
            role_scale_table_new_for_redirect=True,
        )
        installation = adapter_handle.report.as_dict()
        accounting = installation["accounting"]
        if (
            installation["model_parameter_count_installed"]
            != EXACT_CANDIDATE_PARAMETER_COUNT
            or accounting["net_parameters_saved_vs_incumbent"] != EXACT_PARAMETERS_SAVED
            or installation["adapter_parameter_count"] != 92_160
        ):
            raise QwenP35QKVOFactorPackBroadEvalError(
                "Installed structural child plus factors has incorrect accounting."
            )
        installed_state_sha256 = install_qkvo_factor_pack(
            adapter_handle.adapter_registry, loaded_factor_pack
        )
        if installed_state_sha256 != (
            replay_runner.EXPECTED_SELECTED_ADAPTER_STATE_SHA256
        ):
            raise QwenP35QKVOFactorPackBroadEvalError(
                "Factor pack did not reconstruct the selected adapter state."
            )
        adapter_handle.assert_live()
        staged = run_staged_candidate_evaluation(
            runtime,
            executor=executor,
            full_suite=full_suite,
            decontaminated_suite=decontaminated_suite,
            p35_full_report=p35_report,
            p35_projected_report=projected,
            raw_qwen_report=raw_qwen_report,
            output_directory=output_directory,
            child_unique_parameters=EXACT_CANDIDATE_PARAMETER_COUNT,
        )
        adapter_handle.assert_live()
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if adapter_handle is not None and adapter_handle.is_applied:
            try:
                adapter_handle.rollback()
            except Exception as error:  # pragma: no cover - hardware cleanup path
                cleanup_error = error
        try:
            restored = o_runner._restore_parent_exact(
                runtime,
                parent_installation,
                parent_snapshot,
                parent_identity,
            )
            if not restored and cleanup_error is None:
                cleanup_error = QwenP35QKVOFactorPackBroadEvalError(
                    "Broad consumer did not restore p35 exactly."
                )
        except Exception as error:  # pragma: no cover - hardware cleanup path
            if cleanup_error is None:
                cleanup_error = error
        try:
            pack_immutable = all(
                path.is_file() and file_sha256(path) == digest
                for path, digest in immutable_input_hashes.items()
            )
            if not pack_immutable and cleanup_error is None:
                cleanup_error = QwenP35QKVOFactorPackBroadEvalError(
                    "Factor pack or manifest bytes changed during evaluation."
                )
        except Exception as error:  # pragma: no cover - filesystem failure
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None:
            if primary_error is not None:
                raise QwenP35QKVOFactorPackBroadEvalError(
                    "Broad evaluation failed and cleanup also failed; "
                    f"primary={type(primary_error).__name__}: {primary_error}; "
                    f"cleanup={type(cleanup_error).__name__}: {cleanup_error}"
                ) from cleanup_error
            raise QwenP35QKVOFactorPackBroadEvalError(
                f"Broad evaluation cleanup failed: {cleanup_error}"
            ) from cleanup_error

    if staged is None or installation is None or installed_state_sha256 is None:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Broad consumer ended without a staged result."
        )
    result: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": (
            "decontaminated-zero-loss-pass"
            if staged["promotion_gate_passed"]
            else "decontaminated-zero-loss-reject"
        ),
        "inputs": {
            "p35_runtime": dict(source_runtime),
            "factor_pack": str(factor_path),
            "factor_pack_file_sha256": immutable_input_hashes[factor_path],
            "factor_manifest": str(manifest_path),
            "factor_manifest_file_sha256": immutable_input_hashes[manifest_path],
            "broad_suite_file_sha256": EXACT_BROAD_SUITE_FILE_SHA256,
            "raw_qwen_report_file_sha256": (EXACT_RAW_QWEN_REPORT_FILE_SHA256),
        },
        "raw_qwen_parent": {
            "audit": dict(raw_qwen_audit),
            "summary": _report_summary(raw_qwen_report),
        },
        "fresh_current_p35": {
            "report": str(p35_path.resolve()),
            "report_sha256": p35_report_sha256,
            "summary": p35_summary,
            "raw_qwen_parent_pass_retention": raw_to_p35,
            "legacy_633_report_reused": False,
        },
        "decontamination": {
            "suite": str(subset_path.resolve()),
            "suite_file_sha256": file_sha256(subset_path),
            "bound_suite_sha256": decontaminated_suite.suite_sha256,
            "provenance": str(provenance_path.resolve()),
            "provenance_file_sha256": file_sha256(provenance_path),
            "audit": decontamination,
            "projected_parent_report": str(projected_path.resolve()),
            "projected_parent_report_sha256": file_sha256(projected_path),
        },
        "factor_pack_verification": factor_contract,
        "candidate_installation": {
            "installed_adapter_state_sha256": installed_state_sha256,
            "installation_report": installation,
            "p35_unique_parameters": EXACT_P35_PARAMETER_COUNT,
            "candidate_unique_parameters": EXACT_CANDIDATE_PARAMETER_COUNT,
            "unique_parameters_saved": EXACT_PARAMETERS_SAVED,
            "compression_ratio": (
                EXACT_P35_PARAMETER_COUNT / EXACT_CANDIDATE_PARAMETER_COUNT
            ),
        },
        "staged_evaluation": staged,
        "transaction": {
            "factor_pack_immutable": pack_immutable,
            "adapter_rolled_back": (
                adapter_handle is not None and not adapter_handle.is_applied
            ),
            "incumbent_p35_restored_exactly": restored,
            "candidate_installed_after_return": False,
            "checkpoint_saved": False,
            "factor_pack_written_or_modified": False,
            "optimizer_state_saved": False,
        },
        "promotion_gate_passed": bool(staged["promotion_gate_passed"]),
        "full_regression_used_for_acceptance": False,
        "one_redirect_compaction_certified": bool(
            staged["promotion_gate_passed"] and restored and pack_immutable
        ),
    }
    result["report_sha256"] = _canonical_sha256(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--factor-pack", type=Path, required=True)
    parser.add_argument("--factor-manifest", type=Path, required=True)
    parser.add_argument("--broad-suite", type=Path, required=True)
    parser.add_argument("--raw-qwen-parent-report", type=Path, required=True)
    parser.add_argument(
        "--fit-prompts-json",
        type=Path,
        default=qkvo_runner.DEFAULT_FIT_PROMPTS,
    )
    parser.add_argument(
        "--calibration-selection-prompts-json",
        type=Path,
        default=qkvo_runner.DEFAULT_CALIBRATION_PROMPTS,
    )
    parser.add_argument(
        "--heldout-evaluation-prompts-json",
        type=Path,
        default=qkvo_runner.DEFAULT_HELDOUT_PROMPTS,
    )
    parser.add_argument(
        "--fit-prompt-manifest",
        type=Path,
        default=qkvo_runner.DEFAULT_FIT_MANIFEST,
    )
    parser.add_argument(
        "--audited-prompt-manifest",
        type=Path,
        default=qkvo_runner.DEFAULT_AUDITED_MANIFEST,
    )
    parser.add_argument(
        "--transition-inventory",
        type=Path,
        default=qkvo_runner.DEFAULT_TRANSITION_INVENTORY,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.set_defaults(model=None)
    return parser


def _reserve_output_directory(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Broad-evaluation output directory must be absent and fresh."
        )
    try:
        resolved.mkdir(parents=False, exist_ok=False)
    except FileNotFoundError as error:
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Broad-evaluation output parent must already exist."
        ) from error
    return resolved


def _assert_output_disjoint_from_inputs(
    output_directory: Path, *, files: Sequence[Path], directories: Sequence[Path]
) -> None:
    output = output_directory.resolve()
    file_values = [path.resolve() for path in files]
    directory_values = [path.resolve() for path in directories]
    if output in file_values or any(
        directory in output.parents for directory in directory_values
    ):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Output directory may not be inside a checkpoint or factor-pack input."
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _assert_output_disjoint_from_inputs(
        args.output_directory,
        files=(
            args.factor_pack,
            args.factor_manifest,
            args.broad_suite,
            args.raw_qwen_parent_report,
        ),
        directories=(args.codebook_checkpoint,),
    )
    full_suite_payload = load_verified_broad_suite(args.broad_suite)
    raw_qwen_report, raw_qwen_audit = load_verified_raw_qwen_parent_report(
        args.raw_qwen_parent_report, full_suite_payload
    )
    fixtures = qkvo_runner.load_verified_prompt_splits(
        args.fit_prompts_json,
        args.calibration_selection_prompts_json,
        args.heldout_evaluation_prompts_json,
        fit_manifest_path=args.fit_prompt_manifest,
        audited_manifest_path=args.audited_prompt_manifest,
        transition_inventory_path=args.transition_inventory,
    )
    fixture_rows = load_verified_fixture_rows(
        fixtures,
        fit_manifest_path=args.fit_prompt_manifest,
        audited_manifest_path=args.audited_prompt_manifest,
    )
    loaded_factor_pack = load_qkvo_factor_pack(args.factor_pack, args.factor_manifest)
    runtime, tokenizer, source_runtime = attention_probe._load_runtime_and_tokenizer(
        args
    )
    if source_runtime.get("manifest_sha256") != (
        replay_runner.EXPECTED_CODEBOOK_MANIFEST_SHA256
    ):
        raise QwenP35QKVOFactorPackBroadEvalError(
            "Loaded checkpoint is not the exact p35 artifact used by the factor pack."
        )
    environment = replay_runner.assert_exact_execution_environment(
        runtime, device=args.device, dtype=args.dtype
    )
    full_suite = unified_runner.bind_evaluation_suite(
        tokenizer,
        args.broad_suite,
        name="p35-qkvo-full-broad-regression",
        max_new_tokens=EXACT_MAX_NEW_TOKENS,
    )
    executor = unified_runner.SuiteExecutor(
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=EXACT_MAX_NEW_TOKENS,
    )
    output_directory = _reserve_output_directory(args.output_directory)
    result_path = output_directory / "broad_evaluation_report.json"
    result = consume_factor_pack_broad(
        runtime,
        tokenizer,
        source_runtime=source_runtime,
        fixtures=fixtures,
        fixture_rows=fixture_rows,
        loaded_factor_pack=loaded_factor_pack,
        factor_path=args.factor_pack,
        manifest_path=args.factor_manifest,
        full_suite_payload=full_suite_payload,
        full_suite=full_suite,
        raw_qwen_report=raw_qwen_report,
        raw_qwen_audit=raw_qwen_audit,
        executor=executor,
        output_directory=output_directory,
    )
    result["environment"] = environment
    result["paths"] = {
        "codebook_checkpoint": str(args.codebook_checkpoint.resolve()),
        "tokenizer": str(args.tokenizer),
        "broad_suite": str(args.broad_suite.resolve()),
        "raw_qwen_parent_report": str(args.raw_qwen_parent_report.resolve()),
        "output_directory": str(output_directory),
    }
    result["report_sha256"] = _canonical_sha256(
        {key: value for key, value in result.items() if key != "report_sha256"}
    )
    _exclusive_atomic_json(result_path, result)
    print(
        json.dumps(
            {
                "output": str(result_path),
                "report_sha256": result["report_sha256"],
                "promotion_gate_passed": result["promotion_gate_passed"],
                "full_evaluation_executed": result["staged_evaluation"]["full"][
                    "executed"
                ],
                "incumbent_p35_restored_exactly": result["transaction"][
                    "incumbent_p35_restored_exactly"
                ],
                "factor_pack_immutable": result["transaction"]["factor_pack_immutable"],
                "checkpoint_saved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
