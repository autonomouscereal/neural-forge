#!/usr/bin/env python3
"""Replay and publish the authenticated Qwen3-4B compensated-v10 fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from neural_forge.dynamic_behavior_gate import (
    CachedWholeOutputBehaviorGate,
    CandidateWholeOutput,
    TaskScoreCoveragePolicy,
    WholeOutputCase,
    WholeOutputGateConfig,
    WholeOutputMetadata,
    identify_behavior_callable,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import FoldDecision
from neural_forge.dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from neural_forge.dynamic_production_catalog import (
    ProductionDynamicFoldCatalog,
    ProductionFoldBundle,
    ProductionFoldCandidateSpec,
    ProductionFoldCapability,
)
from neural_forge.dynamic_qwen_compensated_adapter import (
    Qwen3CompensatedFoldStackAdapter,
    Qwen3CompensatedReplayEvidence,
    Qwen3CompensatedTargetSynthesisAdapter,
)
from neural_forge.dynamic_search import DynamicFoldSearchObjective
from neural_forge.dynamic_target_synthesis import DynamicTargetRequest
from neural_forge.hf_qwen_direct_local import (
    file_sha256,
    load_local_qwen3_model,
    load_local_tokenizer,
    read_json_object,
    state_identity,
)


RUN_ROOT = Path(
    "/mnt/ai_data/neural-forge/qwen3_4b_compensated_20260710/"
    "qwen3_4b_multi_anchor_layer34_drop4_runtime_v2_v10_gpu6_"
    "sourcepadded_20260711"
)
PARENT_CACHE = Path(
    "/mnt/ai_data/neural-forge/qwen3_4b_compensated_20260710/"
    "parent_caches/qwen3_4b_base_certification_runtime_v2_v10_gpu6.json"
)
PARENT_MODEL = Path(
    "/home/cereal/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-4B-Base/snapshots/"
    "906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
EXPECTED_PARENT_PARAMETERS = 4_022_468_096
EXPECTED_TARGET_PARAMETERS = 4_022_437_376
EXPECTED_LAYER_INDEX = 34
GENERATION_LIMIT = 96
SUBSET_SIZE = 25
SUBSET_SEED = 20_260_716
FRAGMENT_MINIMUM_CHARACTERS = 12


class Qwen3V10RunnerError(RuntimeError):
    """Production evidence or execution failed closed."""


@dataclass(frozen=True, slots=True)
class EvidencePaths:
    transaction_report: Path
    parent_cache: Path
    candidate_eval: Path
    manual_packet: Path
    certification_verdict: Path
    manual_verdict: Path


@dataclass(frozen=True, slots=True)
class EvidenceHashes:
    transaction_report: str
    parent_cache: str
    candidate_eval: str
    manual_packet: str
    certification_verdict: str


PRODUCTION_HASHES = EvidenceHashes(
    transaction_report=(
        "2734ed6e9f5a10ee00a4d9eb6b010e71ea9f38a4cff24b83a6d86ec2b9bc3c69"
    ),
    parent_cache=("38053acfd8b7aaa8f86e008654a2de25cf6f4d7a07a08765393f4a18f893e384"),
    candidate_eval=("39d219fe3660dddaf370b66c8cfdec1a26bd19fee62761a33db88ce197ec932e"),
    manual_packet=("d9440c5d6c5397af1c217d210dd1c53f4213f25a5e7dc9cec14a229202bcdcbc"),
    certification_verdict=(
        "5c2a2b5aa35dedf06ca5e5453dc2085eef91a56e9c7c21e6b56a25f6910d55bf"
    ),
)


EXPECTED_STEPS = (
    {
        "source_width": 9728,
        "target_width": 9727,
        "dropped": 6555,
        "anchors": [2964, 426, 3337],
        "coefficients": [
            0.07458669588479849,
            0.05364216124656531,
            0.0844598145106747,
        ],
        "fold_sha256": (
            "856432077eeeabd9038972ff3192adfaeb4ddc1a401898fdf20700aa92718cfa"
        ),
        "compute_indices_sha256": (
            "acd00006b39ce072212f1ae804bf07a2078dcc608e246092b85ad2130871e159"
        ),
    },
    {
        "source_width": 9727,
        "target_width": 9726,
        "dropped": 5974,
        "anchors": [9706, 3907, 8596],
        "coefficients": [
            -0.04941608255544608,
            -0.02378432284389942,
            0.04769815110351222,
        ],
        "fold_sha256": (
            "9c39dcdce83d2f5044efbdfd1aceb18627cc5bbb95f49f2308baa4d28d3ed0e7"
        ),
        "compute_indices_sha256": (
            "c4c4d757226ec65620b03c3ffc3f67735ea11f16ca848c678a30a9de4023a4bf"
        ),
    },
    {
        "source_width": 9726,
        "target_width": 9725,
        "dropped": 7630,
        "anchors": [465, 8328, 3075],
        "coefficients": [
            -0.0166840400137718,
            0.08174723633528191,
            0.013013572467597476,
        ],
        "fold_sha256": (
            "06a67ba198625669a31a3b1eca88801292f84050ed69b2620d72857320e43390"
        ),
        "compute_indices_sha256": (
            "60a1421c00b11b2be9f3c5553d5b25f17cb04d69388e41329f64516595d8f678"
        ),
    },
    {
        "source_width": 9725,
        "target_width": 9724,
        "dropped": 8388,
        "anchors": [2688, 1394, 2940],
        "coefficients": [
            0.028131605334933512,
            -0.03492454332444081,
            0.015498113130565634,
        ],
        "fold_sha256": (
            "872028a1d5c58e31d5e007430122d0e5e44b8cef788360492b46b266303f21ed"
        ),
        "compute_indices_sha256": (
            "a498dd4f6a67710e0a3919f5d2ef4696f25708e9d479e2d36c542901abf56755"
        ),
    },
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _verify_file(path: Path, expected: str, label: str) -> str:
    source = path.expanduser().absolute()
    if not source.is_file():
        raise Qwen3V10RunnerError(f"{label} does not exist: {source}.")
    actual = file_sha256(source)
    if actual != expected:
        raise Qwen3V10RunnerError(f"{label} SHA-256 differs: {actual} != {expected}.")
    return actual


def _surface_repeated_bigram_share(text: str) -> float:
    tokens = re.findall(r"[A-Za-z0-9_]+", str(text).lower())
    bigrams = list(zip(tokens, tokens[1:]))
    if not bigrams:
        return 0.0
    return (len(bigrams) - len(set(bigrams))) / len(bigrams)


def _fragment_like(text: str) -> bool:
    return len(str(text).strip()) < FRAGMENT_MINIMUM_CHARACTERS


def _normalized_rules(
    value: Any,
    case_id: str,
) -> tuple[tuple[str, ...], ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise Qwen3V10RunnerError(f"Relevance rules are malformed for {case_id}.")
    result = []
    for group in value:
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            raise Qwen3V10RunnerError(
                f"Relevance rule group is malformed for {case_id}."
            )
        patterns = tuple(str(pattern) for pattern in group if str(pattern))
        if not patterns:
            raise Qwen3V10RunnerError(f"Relevance rule group is empty for {case_id}.")
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise Qwen3V10RunnerError(
                    f"Relevance rule is invalid for {case_id}."
                ) from exc
        result.append(patterns)
    return tuple(result)


def _report_maps(
    payload: Mapping[str, Any],
    label: str,
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    generations = payload.get("generations")
    diagnostics = payload.get("generation_diagnostics")
    if (
        not isinstance(generations, list)
        or not generations
        or not isinstance(diagnostics, list)
        or not diagnostics
    ):
        raise Qwen3V10RunnerError(
            f"{label} lacks generations and generation diagnostics."
        )
    rows: dict[str, dict[str, Any]] = {}
    ordered = []
    for raw in generations:
        if not isinstance(raw, Mapping):
            raise Qwen3V10RunnerError(f"{label} generation row is malformed.")
        case_id = str(raw.get("id", "")).strip()
        if not case_id or case_id in rows:
            raise Qwen3V10RunnerError(f"{label} generation IDs repeat.")
        rows[case_id] = dict(raw)
        ordered.append(case_id)
    diagnostic_rows: dict[str, dict[str, Any]] = {}
    diagnostic_order = []
    for raw in diagnostics:
        if not isinstance(raw, Mapping):
            raise Qwen3V10RunnerError(f"{label} generation diagnostic is malformed.")
        case_id = str(raw.get("id", "")).strip()
        if not case_id or case_id in diagnostic_rows:
            raise Qwen3V10RunnerError(f"{label} generation diagnostic IDs repeat.")
        diagnostic_rows[case_id] = dict(raw)
        diagnostic_order.append(case_id)
    if diagnostic_order != ordered:
        raise Qwen3V10RunnerError(f"{label} generation and diagnostic order differs.")
    return tuple(ordered), rows, diagnostic_rows


def _stable_rows(
    rows: Sequence[dict[str, Any]],
    *,
    stratum: tuple[str, bool],
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(
                f"{SUBSET_SEED}\0{stratum[0]}\0{int(stratum[1])}\0{row['id']}".encode()
            ).hexdigest(),
            str(row["id"]),
        ),
    )


@dataclass(frozen=True, slots=True)
class GenerationBaseline:
    cases: tuple[WholeOutputCase, ...]
    rules: Mapping[str, tuple[tuple[str, ...], ...]]
    selected_strata: tuple[tuple[str, bool], ...]

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "cases": [value.as_dict() for value in self.cases],
                "rules": {
                    key: [list(group) for group in value]
                    for key, value in sorted(self.rules.items())
                },
                "selected_strata": [
                    [category, passed] for category, passed in self.selected_strata
                ],
                "subset_seed": SUBSET_SEED,
                "generation_limit": GENERATION_LIMIT,
            }
        )

    def task_score(
        self,
        case: WholeOutputCase,
        output: str,
    ) -> bool | None:
        groups = self.rules.get(case.case_id, ())
        if not groups:
            return None
        lowered = str(output).lower()
        return all(
            any(re.search(pattern, lowered) is not None for pattern in group)
            for group in groups
        )


def _continuation(row: Mapping[str, Any], case_id: str) -> str:
    prompt = row.get("prompt")
    output = row.get("output")
    if not isinstance(prompt, str) or not isinstance(output, str):
        raise Qwen3V10RunnerError(
            f"Generation prompt/output is malformed for {case_id}."
        )
    if output.startswith(prompt):
        return output[len(prompt) :]
    explicit = row.get("continuation")
    if isinstance(explicit, str):
        return explicit
    raise Qwen3V10RunnerError(
        f"Generation output lost its prompt prefix for {case_id}."
    )


def _metadata(
    row: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    case_id: str,
) -> WholeOutputMetadata:
    raw = row.get("diagnostics")
    if not isinstance(raw, Mapping):
        raw = diagnostic.get("diagnostics")
    if not isinstance(raw, Mapping):
        raise Qwen3V10RunnerError(f"Generation metadata is missing for {case_id}.")
    token_count = raw.get("generated_token_count")
    repetition = raw.get("repeated_bigram_share")
    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or token_count < 0
        or token_count > GENERATION_LIMIT
        or isinstance(repetition, bool)
        or not isinstance(repetition, (int, float))
        or not math.isfinite(float(repetition))
        or not 0.0 <= float(repetition) <= 1.0
    ):
        raise Qwen3V10RunnerError(f"Generation metadata is invalid for {case_id}.")
    return WholeOutputMetadata(
        generated_token_count=token_count,
        generation_limit=GENERATION_LIMIT,
        maximum_repeated_ngram_share=float(repetition),
    )


def build_generation_baseline(
    parent_payload: Mapping[str, Any],
) -> GenerationBaseline:
    ordered, rows, diagnostics = _report_maps(
        parent_payload,
        "Cached parent",
    )
    enriched = []
    for case_id in ordered:
        row = dict(rows[case_id])
        diagnostic = diagnostics[case_id]
        category = str(row.get("category", "")).strip()
        prompt = row.get("prompt")
        if not category or not isinstance(prompt, str) or not prompt.strip():
            raise Qwen3V10RunnerError(
                f"Cached parent contract is malformed for {case_id}."
            )
        row["id"] = case_id
        row["diagnostic"] = diagnostic
        enriched.append(row)
    strata: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for row in enriched:
        key = (
            str(row["category"]),
            bool(row["diagnostic"].get("passed", False)),
        )
        strata.setdefault(key, []).append(row)
    if SUBSET_SIZE < len(strata):
        raise Qwen3V10RunnerError(
            "Production subset cannot cover every category/pass stratum."
        )
    ordered_strata = {
        key: _stable_rows(value, stratum=key) for key, value in strata.items()
    }
    offsets = {key: 0 for key in ordered_strata}
    selected = []
    while len(selected) < SUBSET_SIZE:
        progressed = False
        for key in sorted(ordered_strata):
            offset = offsets[key]
            values = ordered_strata[key]
            if offset >= len(values):
                continue
            selected.append(values[offset])
            offsets[key] += 1
            progressed = True
            if len(selected) == SUBSET_SIZE:
                break
        if not progressed:
            raise Qwen3V10RunnerError(
                "Could not fill the deterministic generation subset."
            )
    cases = []
    rules = {}
    selected_strata = []
    for row in selected:
        case_id = str(row["id"])
        continuation = _continuation(row, case_id)
        rules[case_id] = _normalized_rules(
            row.get("relevance_rules"),
            case_id,
        )
        cases.append(
            WholeOutputCase(
                case_id=case_id,
                prompt=str(row["prompt"]),
                parent_output=continuation,
                category=str(row["category"]),
                parent_fragment_like=_fragment_like(continuation),
                parent_metadata=_metadata(
                    row,
                    row["diagnostic"],
                    case_id,
                ),
                split="heldout",
            )
        )
        selected_strata.append(
            (
                str(row["category"]),
                bool(row["diagnostic"].get("passed", False)),
            )
        )
    return GenerationBaseline(
        cases=tuple(cases),
        rules=rules,
        selected_strata=tuple(selected_strata),
    )


def _build_gate(
    baseline: GenerationBaseline,
    generator: Callable[
        [Any, tuple[WholeOutputCase, ...]],
        Mapping[str, CandidateWholeOutput],
    ],
) -> CachedWholeOutputBehaviorGate:
    required_categories = tuple(
        sorted(
            {
                case.category
                for case in baseline.cases
                if baseline.task_score(case, case.parent_output) is True
            }
        )
    )
    generator_identity = identify_behavior_callable(
        generator,
        implementation_id=("neural_forge.qwen3.compensated.whole_output_generator"),
        version=1,
    )
    scorer_identity = identify_behavior_callable(
        baseline.task_score,
        implementation_id=("neural_forge.qwen3.compensated.regex_task_scorer"),
        version=1,
    )
    return CachedWholeOutputBehaviorGate(
        baseline.cases,
        generator=generator,
        config=WholeOutputGateConfig(
            maximum_new_empty_responses=0,
            maximum_new_fragment_responses=0,
            minimum_parent_nonempty_retention=1.0,
            minimum_task_score_retention=1.0,
            maximum_new_generation_limit_saturations=0,
            maximum_repeated_ngram_share_regressions=0,
            maximum_repeated_ngram_share_delta=0.05,
            task_score_coverage=TaskScoreCoveragePolicy.production(
                required_splits=("heldout",),
                required_categories=required_categories,
                require_split_category_cross_product=bool(required_categories),
            ),
        ),
        task_scorer=baseline.task_score,
        gate_id="neural_forge.qwen3.cached_whole_output_behavior",
        gate_version=2,
        generator_identity=generator_identity,
        task_scorer_identity=scorer_identity,
        baseline_binding={
            "generation_baseline_fingerprint": baseline.fingerprint,
            "selected_strata": [list(value) for value in baseline.selected_strata],
            "subset_seed": SUBSET_SEED,
            "generation_limit": GENERATION_LIMIT,
        },
    )


def _candidate_outputs(
    baseline: GenerationBaseline,
    candidate_payload: Mapping[str, Any],
    parent_payload: Mapping[str, Any],
) -> Mapping[str, CandidateWholeOutput]:
    parent_ids, parent_rows, _parent_diagnostics = _report_maps(
        parent_payload,
        "Cached parent",
    )
    candidate_ids, candidate_rows, candidate_diagnostics = _report_maps(
        candidate_payload,
        "Cached candidate",
    )
    if parent_ids != candidate_ids:
        raise Qwen3V10RunnerError(
            "Cached parent and candidate generation inventories differ."
        )
    for case_id in parent_ids:
        parent = parent_rows[case_id]
        candidate = candidate_rows[case_id]
        for name in ("prompt", "category", "relevance_rules"):
            if candidate.get(name) != parent.get(name):
                raise Qwen3V10RunnerError(
                    f"Cached candidate changed {name} for {case_id}."
                )
    outputs = {}
    for case in baseline.cases:
        row = candidate_rows[case.case_id]
        continuation = _continuation(row, case.case_id)
        outputs[case.case_id] = CandidateWholeOutput(
            output=continuation,
            fragment_like=_fragment_like(continuation),
            metadata=_metadata(
                row,
                candidate_diagnostics[case.case_id],
                case.case_id,
            ),
        )
    return outputs


def _validate_manual_packet(
    packet: Mapping[str, Any],
    verdict: Mapping[str, Any],
) -> str:
    body = dict(packet)
    internal = body.pop("packet_sha256", None)
    if (
        packet.get("schema") != "neural-forge-qualitative-parent-child-review-v1"
        or internal != hashlib.sha256(_canonical_bytes(body)).hexdigest()
        or packet.get("status") != "awaiting-explicit-review"
        or packet.get("selected_sample_count") != 128
        or packet.get("required_sample_count") != 128
        or not isinstance(packet.get("samples"), list)
        or len(packet["samples"]) != 128
    ):
        raise Qwen3V10RunnerError("Manual review packet integrity/coverage failed.")
    reviewer = str(verdict.get("reviewer", "")).strip()
    notes = str(verdict.get("notes", "")).strip()
    if (
        verdict.get("schema") != "neural-forge-qualitative-review-verdict-v1"
        or verdict.get("packet_sha256") != internal
        or verdict.get("reviewed_sample_count") != 128
        or str(verdict.get("verdict", "")).lower() != "pass"
        or not reviewer
        or reviewer.startswith("replace-with")
        or not notes
        or notes.startswith("replace-with")
    ):
        raise Qwen3V10RunnerError(
            "A concrete positive 128-sample manual verdict is required "
            "before the 4B model is loaded."
        )
    return str(internal)


def _validate_certification(verdict: Mapping[str, Any]) -> None:
    domains = verdict.get("domains")
    if (
        verdict.get("passed") is not True
        or verdict.get("parent_forward_passes") != 0
        or verdict.get("perplexity_used_for_acceptance") is not False
        or not isinstance(domains, Mapping)
        or set(domains) != {"behavior", "choice", "generation"}
    ):
        raise Qwen3V10RunnerError(
            "Cached certification verdict is not the expected zero-parent gate."
        )
    for name, domain in domains.items():
        if (
            not isinstance(domain, Mapping)
            or domain.get("passed") is not True
            or domain.get("loss_count") != 0
            or domain.get("retained_parent_passes") != domain.get("parent_correct")
        ):
            raise Qwen3V10RunnerError(f"Cached certification domain regressed: {name}.")


def _validate_parent_identity(
    parent_payload: Mapping[str, Any],
    *,
    expected_parent_parameters: int,
) -> Mapping[str, Any]:
    identity = parent_payload.get("identity_binding")
    if not isinstance(identity, Mapping):
        raise Qwen3V10RunnerError("Parent cache lacks identity_binding.")
    body = dict(identity)
    observed = body.pop("identity_sha256", None)
    if (
        observed != canonical_sha256(body)
        or not isinstance(identity.get("state_sha256"), str)
        or len(identity["state_sha256"]) != 64
        or not isinstance(identity.get("state_tensor_count"), int)
        or identity["state_tensor_count"] <= 0
        or parent_payload.get("param_count") != expected_parent_parameters
    ):
        raise Qwen3V10RunnerError(
            "Parent cache identity or parameter accounting differs."
        )
    return identity


@dataclass(frozen=True, slots=True)
class CachedV10Preflight:
    evidence: Qwen3CompensatedReplayEvidence
    baseline: GenerationBaseline
    decision: FoldDecision
    parent_identity: Mapping[str, Any]
    manual_packet_internal_sha256: str
    manual_verdict_sha256: str

    @property
    def accepted(self) -> bool:
        return self.decision.accepted

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-qwen3-v10-cached-preflight-v1",
            "accepted": self.accepted,
            "decision": self.decision.as_dict(),
            "evidence": self.evidence.as_dict(),
            "baseline_fingerprint": self.baseline.fingerprint,
            "baseline_case_ids": [case.case_id for case in self.baseline.cases],
            "manual_packet_internal_sha256": (self.manual_packet_internal_sha256),
            "manual_verdict_sha256": self.manual_verdict_sha256,
            "parent_identity_sha256": self.parent_identity["identity_sha256"],
            "model_loading_performed": False,
            "gpu_loading_performed": False,
            "parent_forward_passes": 0,
        }


def preflight_v10(
    paths: EvidencePaths,
    *,
    expected_hashes: EvidenceHashes = PRODUCTION_HASHES,
    expected_steps: Sequence[Mapping[str, Any]] = EXPECTED_STEPS,
    expected_parent_parameters: int = EXPECTED_PARENT_PARAMETERS,
    expected_target_parameters: int = EXPECTED_TARGET_PARAMETERS,
    expected_layer_index: int = EXPECTED_LAYER_INDEX,
) -> CachedV10Preflight:
    """Authenticate and behavior-gate every cache before any model load."""

    observed_hashes = {
        "transaction_report": _verify_file(
            paths.transaction_report,
            expected_hashes.transaction_report,
            "transaction report",
        ),
        "parent_cache": _verify_file(
            paths.parent_cache,
            expected_hashes.parent_cache,
            "parent cache",
        ),
        "candidate_eval": _verify_file(
            paths.candidate_eval,
            expected_hashes.candidate_eval,
            "candidate evaluation",
        ),
        "manual_packet": _verify_file(
            paths.manual_packet,
            expected_hashes.manual_packet,
            "manual review packet",
        ),
        "certification_verdict": _verify_file(
            paths.certification_verdict,
            expected_hashes.certification_verdict,
            "certification verdict",
        ),
    }
    if not paths.manual_verdict.is_file():
        raise Qwen3V10RunnerError(
            "Positive manual verdict is missing; 4B model load skipped: "
            f"{paths.manual_verdict}."
        )
    manual_verdict_sha256 = file_sha256(paths.manual_verdict)
    transaction_report = read_json_object(
        paths.transaction_report,
        "transaction report",
    )
    parent_payload = read_json_object(paths.parent_cache, "parent cache")
    candidate_payload = read_json_object(
        paths.candidate_eval,
        "candidate evaluation",
    )
    packet = read_json_object(paths.manual_packet, "manual review packet")
    certification = read_json_object(
        paths.certification_verdict,
        "certification verdict",
    )
    manual_verdict = read_json_object(
        paths.manual_verdict,
        "manual review verdict",
    )
    packet_internal = _validate_manual_packet(packet, manual_verdict)
    _validate_certification(certification)
    parent_identity = _validate_parent_identity(
        parent_payload,
        expected_parent_parameters=expected_parent_parameters,
    )
    if (
        candidate_payload.get("param_count") != expected_target_parameters
        or candidate_payload.get("model") != "qwen3-compensated-ragged-parent-free"
        or parent_payload.get("param_count") != expected_parent_parameters
        or _canonical_bytes(candidate_payload.get("eval_suite"))
        != _canonical_bytes(parent_payload.get("eval_suite"))
    ):
        raise Qwen3V10RunnerError("Cached parent/candidate evaluation binding differs.")
    binding = {
        "schema": "neural-forge-qwen3-v10-production-evidence-v1",
        "file_sha256": observed_hashes,
        "manual_verdict_sha256": manual_verdict_sha256,
        "manual_packet_internal_sha256": packet_internal,
        "parent_identity_sha256": parent_identity["identity_sha256"],
        "expected_parent_parameters": expected_parent_parameters,
        "expected_target_parameters": expected_target_parameters,
        "expected_layer_index": expected_layer_index,
        "subset_size": SUBSET_SIZE,
        "subset_seed": SUBSET_SEED,
        "generation_limit": GENERATION_LIMIT,
    }
    evidence_fingerprint = canonical_sha256(binding)
    evidence = Qwen3CompensatedReplayEvidence.from_transaction_report(
        transaction_report,
        transaction_report_sha256=observed_hashes["transaction_report"],
        cached_parent_sha256=observed_hashes["parent_cache"],
        cached_candidate_sha256=observed_hashes["candidate_eval"],
        manual_packet_sha256=observed_hashes["manual_packet"],
        expected_parent_state_sha256=str(parent_identity["state_sha256"]),
        expected_parent_state_tensor_count=int(parent_identity["state_tensor_count"]),
        evidence_fingerprint=evidence_fingerprint,
    )
    if (
        evidence.layer_index != expected_layer_index
        or evidence.expected_parent_parameters != expected_parent_parameters
        or evidence.expected_target_parameters != expected_target_parameters
        or [step.as_dict() for step in evidence.steps]
        != [dict(value) for value in expected_steps]
    ):
        raise Qwen3V10RunnerError(
            "Authenticated transaction sequence differs from compensated-v10."
        )
    baseline = build_generation_baseline(parent_payload)
    outputs = _candidate_outputs(
        baseline,
        candidate_payload,
        parent_payload,
    )
    gate = _build_gate(
        baseline,
        lambda _model, _cases: outputs,
    )
    decision = gate.evaluate_model(
        object(),
        verify_model_execution=False,
    )
    if not decision.accepted:
        raise Qwen3V10RunnerError(
            "Cached compensated-v10 candidate failed the production "
            f"whole-output preflight: {decision.reason}"
        )
    return CachedV10Preflight(
        evidence=evidence,
        baseline=baseline,
        decision=decision,
        parent_identity=parent_identity,
        manual_packet_internal_sha256=packet_internal,
        manual_verdict_sha256=manual_verdict_sha256,
    )


class GreedyWholeOutputGenerator:
    """Generate independent deterministic continuations without a parent call."""

    def __init__(self, tokenizer: Any, *, maximum_prompt_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.maximum_prompt_tokens = int(maximum_prompt_tokens)
        self.invocations: list[dict[str, Any]] = []

    @staticmethod
    def _device(model: nn.Module) -> torch.device:
        devices = {value.device for value in model.parameters()}
        if len(devices) != 1:
            raise Qwen3V10RunnerError(
                "Whole-output generation requires one model device."
            )
        return next(iter(devices))

    def __call__(
        self,
        model: nn.Module,
        cases: tuple[WholeOutputCase, ...],
    ) -> Mapping[str, CandidateWholeOutput]:
        device = self._device(model)
        was_training = bool(model.training)
        model.eval()
        outputs = {}
        summaries = []
        try:
            with torch.inference_mode():
                for case in cases:
                    encoded = self.tokenizer(
                        case.prompt,
                        return_tensors="pt",
                        add_special_tokens=True,
                    )
                    input_ids = encoded.get("input_ids")
                    if (
                        not isinstance(input_ids, Tensor)
                        or input_ids.ndim != 2
                        or input_ids.shape[0] != 1
                        or not 0 < input_ids.shape[1] <= self.maximum_prompt_tokens
                    ):
                        raise Qwen3V10RunnerError(
                            f"Prompt tokenization failed for {case.case_id}."
                        )
                    input_ids = input_ids.to(device=device, dtype=torch.long)
                    attention_mask = encoded.get("attention_mask")
                    kwargs: dict[str, Any] = {
                        "input_ids": input_ids,
                        "max_new_tokens": GENERATION_LIMIT,
                        "do_sample": False,
                        "use_cache": True,
                    }
                    if isinstance(attention_mask, Tensor):
                        kwargs["attention_mask"] = attention_mask.to(
                            device=device,
                            dtype=torch.long,
                        )
                    if self.tokenizer.pad_token_id is not None:
                        kwargs["pad_token_id"] = int(self.tokenizer.pad_token_id)
                    generated = model.generate(**kwargs)
                    if (
                        not isinstance(generated, Tensor)
                        or generated.ndim != 2
                        or generated.shape[0] != 1
                        or generated.shape[1] < input_ids.shape[1]
                    ):
                        raise Qwen3V10RunnerError(
                            "Generation returned an invalid whole-output tensor "
                            f"for {case.case_id}."
                        )
                    if not torch.equal(
                        generated[:, : input_ids.shape[1]],
                        input_ids,
                    ):
                        raise Qwen3V10RunnerError(
                            "Generation changed the exact prompt prefix for "
                            f"{case.case_id}."
                        )
                    continuation = generated[:, input_ids.shape[1] :]
                    token_ids = continuation[0].detach().cpu().tolist()
                    text = self.tokenizer.decode(
                        token_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    repetition = _surface_repeated_bigram_share(text)
                    outputs[case.case_id] = CandidateWholeOutput(
                        output=text,
                        fragment_like=_fragment_like(text),
                        metadata=WholeOutputMetadata(
                            generated_token_count=len(token_ids),
                            generation_limit=GENERATION_LIMIT,
                            maximum_repeated_ngram_share=repetition,
                        ),
                    )
                    summaries.append(
                        {
                            "case_id": case.case_id,
                            "generated_token_count": len(token_ids),
                            "token_ids_sha256": canonical_sha256(token_ids),
                            "output": text,
                            "empty": not bool(text.strip()),
                            "fragment_like": _fragment_like(text),
                            "repeated_bigram_share": repetition,
                        }
                    )
        finally:
            model.train(was_training)
        self.invocations.append(
            {
                "case_count": len(cases),
                "device": str(device),
                "summaries": summaries,
                "parent_forward_passes": 0,
            }
        )
        return outputs


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    vocabulary = tokenizer.get_vocab()
    rows = sorted((str(token), int(index)) for token, index in vocabulary.items())
    payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab_size": len(rows),
        "vocab_sha256": canonical_sha256(rows),
        "special_tokens_map": dict(tokenizer.special_tokens_map or {}),
        "added_vocab": dict(tokenizer.get_added_vocab() or {}),
        "padding_side": str(tokenizer.padding_side),
        "truncation_side": str(tokenizer.truncation_side),
    }
    payload["tokenizer_sha256"] = canonical_sha256(payload)
    return payload


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, str(name).removeprefix("torch."), None)
    if value not in {torch.float16, torch.bfloat16, torch.float32}:
        raise Qwen3V10RunnerError(f"Unsupported model dtype: {name}.")
    return value


def run_production(
    args: argparse.Namespace,
    *,
    model_loader: Callable[..., nn.Module] = load_local_qwen3_model,
) -> dict[str, Any]:
    """Preflight first, then load, replay, gate, publish, and reload."""

    paths = EvidencePaths(
        transaction_report=Path(args.transaction_report),
        parent_cache=Path(args.parent_cache),
        candidate_eval=Path(args.candidate_eval),
        manual_packet=Path(args.manual_packet),
        certification_verdict=Path(args.certification_verdict),
        manual_verdict=Path(args.manual_verdict),
    )
    preflight = preflight_v10(paths)
    if args.preflight_only:
        return {
            "schema": "neural-forge-qwen3-v10-dynamic-run-v1",
            "status": "preflight-accepted",
            "preflight": preflight.as_dict(),
        }
    snapshot_root = Path(args.snapshot_root).expanduser().absolute()
    if not snapshot_root.is_dir():
        raise Qwen3V10RunnerError(f"Snapshot root must already exist: {snapshot_root}.")
    parent_model_path = Path(args.parent_model)
    expected_tokenizer_path = Path(
        str(preflight.parent_identity["source_tokenizer"])
    ).expanduser()
    if (
        not expected_tokenizer_path.is_dir()
        or parent_model_path.expanduser().resolve(strict=True)
        != expected_tokenizer_path.resolve(strict=True)
        or parent_model_path.expanduser().resolve(strict=True).name
        != str(preflight.parent_identity["source_revision"])
    ):
        raise Qwen3V10RunnerError(
            "Direct tokenizer/model directory differs from the exact "
            "cache-bound local snapshot revision."
        )
    tokenizer = load_local_tokenizer(parent_model_path)
    tokenizer_identity = _tokenizer_identity(tokenizer)
    if (
        len(tokenizer) <= 0
        or tokenizer.eos_token_id is None
        or tokenizer.pad_token_id is None
        or not (parent_model_path / "tokenizer.json").is_file()
        or not (parent_model_path / "tokenizer_config.json").is_file()
    ):
        raise Qwen3V10RunnerError("Direct local tokenizer contract is incomplete.")
    if (
        tokenizer_identity["tokenizer_sha256"]
        != preflight.parent_identity["tokenizer_sha256"]
    ):
        raise Qwen3V10RunnerError(
            "Direct local tokenizer differs from the authenticated cache: "
            f"{tokenizer_identity['tokenizer_sha256']} != "
            f"{preflight.parent_identity['tokenizer_sha256']}."
        )
    parent = model_loader(
        parent_model_path,
        device=args.device,
        dtype=_dtype(args.dtype),
    )
    actual_state = state_identity(parent)
    actual_parent_parameters = sum(value.numel() for value in parent.parameters())
    actual_config_sha256 = canonical_sha256(parent.config.to_dict())
    parent_identity_comparison = {
        "state_sha256": {
            "actual": actual_state["state_sha256"],
            "expected": preflight.evidence.expected_parent_state_sha256,
        },
        "state_tensor_count": {
            "actual": actual_state["tensor_count"],
            "expected": (preflight.evidence.expected_parent_state_tensor_count),
        },
        "parameter_count": {
            "actual": actual_parent_parameters,
            "expected": EXPECTED_PARENT_PARAMETERS,
        },
        "config_sha256": {
            "actual": actual_config_sha256,
            "expected": preflight.parent_identity["config_sha256"],
        },
        "tokenizer_sha256": {
            "actual": tokenizer_identity["tokenizer_sha256"],
            "expected": preflight.parent_identity["tokenizer_sha256"],
        },
    }
    if any(
        values["actual"] != values["expected"]
        for values in parent_identity_comparison.values()
    ):
        raise Qwen3V10RunnerError(
            "Direct local Qwen3 parent differs from the authenticated cache: "
            + json.dumps(parent_identity_comparison, sort_keys=True)
        )
    generator = GreedyWholeOutputGenerator(
        tokenizer,
        maximum_prompt_tokens=args.maximum_prompt_tokens,
    )
    gate = _build_gate(preflight.baseline, generator)
    snapshot_config = DynamicFoldSnapshotConfig.disk(
        snapshot_root,
        chunk_bytes=args.snapshot_chunk_bytes,
    )
    released = {"performed": False}

    def release_parent() -> None:
        before = state_identity(parent)
        if before != actual_state:
            raise Qwen3V10RunnerError("Parent changed before fresh child reload.")
        parent.to(device="cpu")
        if state_identity(parent) != actual_state:
            raise Qwen3V10RunnerError("Parent values changed while releasing its GPU.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        released["performed"] = True

    target_adapter = Qwen3CompensatedTargetSynthesisAdapter(preflight.evidence)
    stack_adapter = Qwen3CompensatedFoldStackAdapter(
        evidence=preflight.evidence,
        behavior_gate=gate,
        tokenizer_source=parent_model_path,
        reload_device=args.reload_device,
        snapshot_config=snapshot_config,
        evidence_binding=preflight.as_dict(),
        before_reload=release_parent,
    )
    output_path = Path(args.output).expanduser().absolute()
    candidate_id = output_path.name
    target_request = DynamicTargetRequest(
        child_name="qwen3-compensated-v10",
        target_family="qwen3",
        target_modalities=("text",),
        exact_unique_parameters=EXPECTED_TARGET_PARAMETERS,
        tokenizer_fingerprint=str(preflight.parent_identity["tokenizer_sha256"]),
        adapter_id=target_adapter.adapter_id,
        blank_seed=0,
    )
    capability = ProductionFoldCapability(
        capability_id="qwen3-4b-compensated-v10-exact",
        source_families=("qwen3",),
        target_family="qwen3",
        target_modalities=("text",),
        minimum_target_unique_parameters=EXPECTED_TARGET_PARAMETERS,
        maximum_target_unique_parameters=EXPECTED_TARGET_PARAMETERS,
    )
    candidate = ProductionFoldCandidateSpec(
        candidate_id=candidate_id,
        method_id="qwen3_compensated_transaction_replay",
        evidence_fingerprint=preflight.evidence.evidence_fingerprint,
        target_adapter_id=target_adapter.adapter_id,
        priority=100,
        blank_seed=0,
        configuration=(
            ("layer_index", EXPECTED_LAYER_INDEX),
            ("accepted_step_count", 4),
            ("compute_shape", "source-padded"),
        ),
        metadata=(
            ("authenticated_cache_replay", True),
            ("training_performed", False),
        ),
    )
    bundle = ProductionFoldBundle(
        capability=capability,
        target_adapters=(target_adapter,),
        stack_adapters=(stack_adapter,),
        candidate_specs=(candidate,),
        maximum_attempts=1,
        portfolio_id="neural_forge.production.qwen3-compensated-v10",
    )
    facade = ProductionDynamicFoldCatalog(
        (bundle,),
        ledger_path=(
            Path(args.ledger_path).expanduser().absolute()
            if args.ledger_path
            else Path(args.report).expanduser().absolute().with_suffix(".ledger.jsonl")
        ),
        snapshot_config=snapshot_config,
        require_positive_compaction=True,
    ).assemble()
    search_result = facade.run(
        parent_models={"qwen3-4b-base": parent},
        objective=DynamicFoldSearchObjective(
            target=target_request,
            maximum_attempts=1,
            portfolio_id=bundle.portfolio_id,
            stop_after_first_accept=True,
        ),
        destination_root=output_path.parent,
        parent_tokenizer_fingerprints={
            "qwen3-4b-base": str(preflight.parent_identity["tokenizer_sha256"])
        },
    )
    result = search_result.selected_result
    if (
        not search_result.complete
        or search_result.selected_candidate_id != candidate_id
        or search_result.selected_destination != str(output_path.resolve(strict=True))
        or result is None
    ):
        raise Qwen3V10RunnerError(
            "Production facade did not select the exact v10 candidate/output."
        )
    certificate = result.published.artifact
    run = result.published.compilation.run
    if (
        not result.complete
        or not released["performed"]
        or certificate.artifact_file_count <= 0
        or certificate.artifact_total_bytes <= 0
        or run.whole_model_snapshot_backend != "disk"
        or any(
            value != 0
            for value in (
                certificate.compilation_optimizer_constructions,
                certificate.compilation_optimizer_steps,
                certificate.compilation_backward_calls,
                certificate.compilation_autograd_grad_calls,
                certificate.optimizer_constructions,
                certificate.optimizer_steps,
                certificate.backward_calls,
                certificate.autograd_grad_calls,
            )
        )
    ):
        raise Qwen3V10RunnerError(
            "Production facade returned an incomplete or unsafe certificate."
        )
    return {
        "schema": "neural-forge-qwen3-v10-dynamic-run-v1",
        "status": "complete",
        "preflight": preflight.as_dict(),
        "result": result.as_dict(),
        "production_catalog": {
            "facade_fingerprint": facade.fingerprint,
            "bundle": bundle.as_dict(),
            "search": search_result.as_dict(),
        },
        "generation_receipt": {
            "invocation_count": len(generator.invocations),
            "invocations": generator.invocations,
            "parent_forward_passes": 0,
        },
        "direct_local_tokenizer_identity": tokenizer_identity,
        "cache_bound_tokenizer_identity_sha256": (
            preflight.parent_identity["tokenizer_sha256"]
        ),
        "tokenizer_snapshot_revision_bound": True,
        "parent_gpu_release": released,
        "zero_optimizer_steps": 0,
        "backward_called": False,
        "training_performed": False,
    }


def _write_report(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise Qwen3V10RunnerError(f"Refusing stale report temporary: {temporary}.")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(dict(value)))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model", default=str(PARENT_MODEL))
    parser.add_argument(
        "--transaction-report",
        default=str(RUN_ROOT / "transaction_report.json"),
    )
    parser.add_argument("--parent-cache", default=str(PARENT_CACHE))
    parser.add_argument(
        "--candidate-eval",
        default=str(RUN_ROOT / "final_eval.json"),
    )
    parser.add_argument(
        "--manual-packet",
        default=str(RUN_ROOT / "manual_review_packet.json"),
    )
    parser.add_argument(
        "--certification-verdict",
        default=str(RUN_ROOT / "certification_verdict.json"),
    )
    parser.add_argument(
        "--manual-verdict",
        default=str(RUN_ROOT / "manual_review_verdict.json"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--ledger-path",
        default=None,
        help=("Durable append-only experiment ledger. Defaults beside --report."),
    )
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--reload-device", default="cuda:6")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--maximum-prompt-tokens", type=int, default=512)
    parser.add_argument(
        "--snapshot-chunk-bytes",
        type=int,
        default=64 * 1024**2,
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_production(args)
    except Exception as exc:
        report = {
            "schema": "neural-forge-qwen3-v10-dynamic-run-v1",
            "status": "skipped-before-or-during-load",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_report(Path(args.report), report)
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    _write_report(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
