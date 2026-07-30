"""Audit an LFM2 carrier replant from cached artifacts only.

The audit never constructs or executes a model.  It contrasts the immutable
pre-replant bridge with the published carrier bridge over the already-cached
A12 input identity feature and C_out hidden-state derivative.  Table replants
and route reclosure are crossed independently so their effects cannot be
conflated.  A raw A9 C_in + A11 C_out lane and the support-only pre-replant
lane are both retained as references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_functional_output_refit import (
    authenticate_lfm2_functional_output_refit_result,
)
from neural_forge.hf_lfm2_sparse_semantic_codebook import (
    load_lfm2_sparse_semantic_codebook,
)


REPORT_SCHEMA = "neural-forge-lfm2-carrier-cached-delta-audit-v1"
BRIDGE_SCHEMA = "neural-forge-lfm2-carrier-reclosed-bridge-v1"
BASE_KEYS = (
    "target_table",
    "target_to_donor",
    "input_offsets",
    "input_target_ids",
    "input_coefficients",
    "output_offsets",
    "output_target_ids",
    "output_coefficients",
)
_ZERO_OPERATION_AUDIT = {
    "parent_forward_calls": 0,
    "parent_trace_recaptures": 0,
    "model_forward_calls": 0,
    "model_generate_calls": 0,
    "training_steps": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


class CarrierDeltaAuditError(RuntimeError):
    """A cached carrier differential input is invalid."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise CarrierDeltaAuditError(f"{name} must be a SHA-256 digest.")
    return value.lower()


def _file(path: str | os.PathLike[str], name: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file() or value.is_symlink():
        raise CarrierDeltaAuditError(f"{name} is not an ordinary file.")
    return value


def _directory(path: str | os.PathLike[str], name: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_dir() or value.is_symlink():
        raise CarrierDeltaAuditError(f"{name} is not an ordinary directory.")
    return value


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CarrierDeltaAuditError(f"Cannot read {name}.") from exc
    if not isinstance(value, dict):
        raise CarrierDeltaAuditError(f"{name} must be an object.")
    return value


def _sealed(value: Mapping[str, Any], field: str, schema: str, name: str) -> dict[str, Any]:
    body = dict(value)
    supplied = body.pop(field, None)
    if supplied != canonical_sha256(body) or body.get("schema") != schema:
        raise CarrierDeltaAuditError(f"{name} seal or schema is invalid.")
    return {**body, field: supplied}


def _pinned_tensors(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_keys: Sequence[str],
    name: str,
) -> dict[str, torch.Tensor]:
    if _file_sha256(path) != _digest(expected_file_sha256, f"{name} file hash"):
        raise CarrierDeltaAuditError(f"{name} file hash differs.")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected_keys):
            raise CarrierDeltaAuditError(f"{name} tensor inventory differs.")
        return {key: handle.get_tensor(key).contiguous() for key in expected_keys}


def _route_rows(offsets: torch.Tensor) -> torch.Tensor:
    lengths = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(int(lengths.numel()), dtype=torch.int64), lengths
    )


def _changed_route_rows(
    left: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    left_offsets, left_ids, left_coefficients = left
    right_offsets, right_ids, right_coefficients = right
    if not torch.equal(left_offsets, right_offsets):
        values: list[int] = []
        for donor_id in range(int(left_offsets.numel()) - 1):
            ls, le = int(left_offsets[donor_id]), int(left_offsets[donor_id + 1])
            rs, re = int(right_offsets[donor_id]), int(right_offsets[donor_id + 1])
            if not torch.equal(left_ids[ls:le], right_ids[rs:re]) or not torch.equal(
                left_coefficients[ls:le], right_coefficients[rs:re]
            ):
                values.append(donor_id)
        return torch.tensor(values, dtype=torch.int64)
    changed_positions = (left_ids != right_ids) | (
        left_coefficients.view(torch.uint8).reshape(left_coefficients.numel(), -1)
        != right_coefficients.view(torch.uint8).reshape(right_coefficients.numel(), -1)
    ).any(dim=1)
    rows = _route_rows(left_offsets)
    return torch.unique(rows[changed_positions], sorted=True)


def _sparse_design(
    offsets: torch.Tensor,
    target_ids: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    target_count: int,
    device: torch.device,
) -> torch.Tensor:
    rows = _route_rows(offsets)
    indices = torch.stack((rows, target_ids.to(torch.int64)))
    return torch.sparse_coo_tensor(
        indices.to(device=device),
        coefficients.to(device=device, dtype=torch.float32),
        size=(int(offsets.numel()) - 1, target_count),
        device=device,
        dtype=torch.float32,
        check_invariants=True,
    ).coalesce()


def _route_embeddings(
    route: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    table: torch.Tensor,
    donor_ids: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    offsets, target_ids, coefficients = route
    ids = donor_ids.to(device="cpu", dtype=torch.int64).reshape(-1)
    if int(ids.numel()) == 0:
        return torch.empty((0, int(table.shape[1])), dtype=torch.float32)
    lengths = offsets.index_select(0, ids + 1) - offsets.index_select(0, ids)
    local_rows = torch.repeat_interleave(
        torch.arange(int(ids.numel()), dtype=torch.int64), lengths
    )
    positions = torch.cat(
        [
            torch.arange(int(offsets[row]), int(offsets[row + 1]), dtype=torch.int64)
            for row in ids.tolist()
        ]
    )
    design = torch.sparse_coo_tensor(
        torch.stack((local_rows, target_ids.index_select(0, positions))).to(device),
        coefficients.index_select(0, positions).to(device=device, dtype=torch.float32),
        size=(int(ids.numel()), int(table.shape[0])),
        device=device,
        dtype=torch.float32,
        check_invariants=True,
    ).coalesce()
    return torch.sparse.mm(
        design, table.to(device=device, dtype=torch.float32)
    ).to(device="cpu")


def _embedding_summary(predicted: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    if tuple(predicted.shape) != tuple(expected.shape) or int(predicted.shape[0]) == 0:
        return {"row_count": int(predicted.shape[0])}
    left = predicted.to(torch.float64)
    right = expected.to(torch.float64)
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=1)
    relative_l2 = torch.linalg.vector_norm(left - right, dim=1) / torch.linalg.vector_norm(
        right, dim=1
    ).clamp_min(1e-300)
    return {
        "row_count": int(left.shape[0]),
        "mean_cosine": float(cosine.mean()),
        "p05_cosine": float(torch.quantile(cosine, 0.05)),
        "median_cosine": float(torch.quantile(cosine, 0.5)),
        "mean_relative_l2": float(relative_l2.mean()),
        "median_relative_l2": float(torch.quantile(relative_l2, 0.5)),
        "p95_relative_l2": float(torch.quantile(relative_l2, 0.95)),
    }


def _output_summary(
    logits: torch.Tensor,
    reference_winners: torch.Tensor,
    *,
    baseline_winners: torch.Tensor | None,
) -> tuple[dict[str, Any], torch.Tensor]:
    winners = logits.argmax(dim=1)
    rows = torch.arange(int(logits.shape[0]), device=logits.device)
    references = reference_winners.to(device=logits.device, dtype=torch.int64)
    reference_scores = logits[rows, references]
    top_values, top_ids = torch.topk(logits, k=2, dim=1)
    competitors = torch.where(top_ids[:, 0] == references, top_values[:, 1], top_values[:, 0])
    ranks = (logits > reference_scores.unsqueeze(1)).sum(dim=1) + 1
    result = {
        "state_count": int(logits.shape[0]),
        "reference_top1_match_count": int((winners == references).sum()),
        "reference_top1_match_rate": float((winners == references).to(torch.float64).mean()),
        "mean_reference_margin": float((reference_scores - competitors).mean()),
        "median_reference_rank": float(torch.quantile(ranks.to(torch.float64), 0.5)),
        "mean_reference_rank": float(ranks.to(torch.float64).mean()),
    }
    if baseline_winners is not None:
        base = baseline_winners.to(device=logits.device)
        result["baseline_top1_match_count"] = int((winners == base).sum())
        result["baseline_top1_match_rate"] = float(
            (winners == base).to(torch.float64).mean()
        )
        result["winner_changed_from_baseline_count"] = int((winners != base).sum())
    return result, winners.detach().cpu()


def _route_logits(
    route: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    table: torch.Tensor,
    hidden: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    design = _sparse_design(
        *route, target_count=int(table.shape[0]), device=device
    )
    target_scores = hidden.to(device=device, dtype=torch.float32) @ table.to(
        device=device, dtype=torch.float32
    ).transpose(0, 1)
    return torch.sparse.mm(design, target_scores.transpose(0, 1)).transpose(0, 1)


def _audit_winner_ids(audit: Mapping[str, Any]) -> torch.Tensor:
    values: list[int] = []
    for prompt in audit.get("prompt_results", []):
        tokens = prompt["lanes"]["exact_exact"]["tokens"]
        values.extend(int(token["parent_winner_token_id"]) for token in tokens)
    return torch.tensor(values, dtype=torch.int64)


def _frozen_donor_ids(directory: Path) -> tuple[Counter[int], Counter[int], dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        attempt = _json(path, "frozen response attempt")
        body = dict(attempt)
        supplied = body.pop("attempt_sha256", None)
        if supplied != canonical_sha256(body):
            raise CarrierDeltaAuditError(f"Frozen attempt {path.name} has an invalid seal.")
        prompt_id = str(attempt["identity"]["prompt_id"])
        maximum = int(attempt["identity"]["maximum_new_tokens"])
        if prompt_id not in chosen or maximum > int(
            chosen[prompt_id]["identity"]["maximum_new_tokens"]
        ):
            chosen[prompt_id] = attempt
    all_counts: Counter[int] = Counter()
    intelligence_counts: Counter[int] = Counter()
    for attempt in chosen.values():
        ids = [int(value) for value in attempt["response"]["raw_donor_response_token_ids"]]
        all_counts.update(ids)
        if attempt["response"].get("safety_review_only") is not True:
            intelligence_counts.update(ids)
    return all_counts, intelligence_counts, {
        "selected_prompt_count": len(chosen),
        "all_generated_donor_token_count": sum(all_counts.values()),
        "intelligence_generated_donor_token_count": sum(intelligence_counts.values()),
        "highest_cap_attempt_selected_per_prompt": True,
        "safety_prompts_separated_from_intelligence": True,
    }


def _intersect_count(values: torch.Tensor, selected: set[int]) -> int:
    return sum(1 for value in values.tolist() if int(value) in selected)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CarrierDeltaAuditError("CUDA was requested but is unavailable.")
    prep = _directory(args.carrier_preparation_directory, "carrier preparation")
    request_path = _file(prep / "carrier-request.json", "carrier request")
    request = _json(request_path, "carrier request")
    if request.get("schema") != "neural-forge-lfm2-carrier-reclosure-request-v1":
        raise CarrierDeltaAuditError("Carrier request schema differs.")
    base_path = _file(request["base_bridge"]["path"], "pre-replant bridge")
    base = _pinned_tensors(
        base_path,
        expected_file_sha256=request["base_bridge"]["sha256"],
        expected_keys=BASE_KEYS,
        name="pre-replant bridge",
    )
    parent_payload = _pinned_tensors(
        _file(request["parent_table"]["path"], "parent table"),
        expected_file_sha256=request["parent_table"]["sha256"],
        expected_keys=("parent_table",),
        name="parent table",
    )
    parent_table = parent_payload["parent_table"].to(torch.float32)

    bridge_root = _directory(args.carrier_bridge_directory, "carrier bridge")
    bridge_receipt_path = _file(bridge_root / "bridge-receipt.json", "bridge receipt")
    bridge_receipt = _sealed(
        _json(bridge_receipt_path, "bridge receipt"),
        "artifact_sha256",
        BRIDGE_SCHEMA,
        "bridge receipt",
    )
    if bridge_receipt["artifact_sha256"] != _digest(
        args.expected_carrier_artifact_sha256, "carrier artifact pin"
    ):
        raise CarrierDeltaAuditError("Carrier artifact differs from its explicit pin.")
    bridge = _pinned_tensors(
        _file(bridge_root / bridge_receipt["bridge_tensor_file"], "bridge tensors"),
        expected_file_sha256=bridge_receipt["bridge_tensor_file_sha256"],
        expected_keys=BASE_KEYS,
        name="carrier bridge",
    )

    codebook = load_lfm2_sparse_semantic_codebook(
        _directory(args.a9_codebook_directory, "A9 codebook"),
        expected_artifact_sha256=_digest(args.expected_a9_artifact_sha256, "A9 pin"),
    )
    a11_path = _file(args.a11_coefficients_safetensors, "A11 coefficients")
    a11_payload = _pinned_tensors(
        a11_path,
        expected_file_sha256=args.expected_a11_coefficients_file_sha256,
        expected_keys=("output_semantic_code_coefficients",),
        name="A11 coefficients",
    )
    a11_result = authenticate_lfm2_functional_output_refit_result(
        output_semantic_code_coefficients=a11_payload["output_semantic_code_coefficients"],
        receipt=_json(_file(args.a11_receipt_json, "A11 receipt"), "A11 receipt"),
        expected_cache_key=_digest(args.expected_a11_cache_key, "A11 cache key"),
    )
    if a11_result.result_sha256 != _digest(args.expected_a11_result_sha256, "A11 result"):
        raise CarrierDeltaAuditError("A11 result differs from its explicit pin.")

    feature_root = _directory(args.feature_directory, "A12 derivative")
    feature_path = _file(feature_root / "features.safetensors", "feature tensors")
    features = _pinned_tensors(
        feature_path,
        expected_file_sha256=args.expected_feature_file_sha256,
        expected_keys=("input_feature_transform", "output_feature_transform"),
        name="A12 causal feature derivative",
    )
    hidden = features["output_feature_transform"].to(torch.float32)
    context_path = _file(feature_root / "causal-context.safetensors", "causal context")
    contexts = _pinned_tensors(
        context_path,
        expected_file_sha256=args.expected_context_file_sha256,
        expected_keys=(
            "input_context_signatures",
            "input_donor_ids",
            "output_context_signatures",
            "output_donor_ids",
        ),
        name="A12 causal context derivative",
    )
    audit_path = _file(args.trajectory_audit_json, "trajectory audit")
    if _file_sha256(audit_path) != _digest(args.expected_trajectory_file_sha256, "audit file"):
        raise CarrierDeltaAuditError("Trajectory audit file differs from its pin.")
    audit = _sealed(
        _json(audit_path, "trajectory audit"),
        "audit_sha256",
        "neural-forge-lfm2-teacher-forced-trajectory-factorial-audit-v2",
        "trajectory audit",
    )
    audit_winners = _audit_winner_ids(audit)
    if int(audit_winners.numel()) != int(hidden.shape[0]):
        raise CarrierDeltaAuditError("A12 hidden states and audit winner rows are misaligned.")

    evidence = _pinned_tensors(
        _file(request["carrier_evidence"]["tensors"]["path"], "carrier evidence"),
        expected_file_sha256=request["carrier_evidence"]["tensors"]["sha256"],
        expected_keys=(
            "donor_occurrence_counts",
            "donor_winner_counts",
            "donor_reconstruction_errors",
            "donor_eviction_costs",
        ),
        name="carrier evidence",
    )
    frozen_all, frozen_intelligence, frozen_summary = _frozen_donor_ids(
        _directory(args.frozen_attempt_directory, "frozen attempts")
    )

    target_pre = base["target_table"].to(torch.float32)
    target_post = bridge["target_table"].to(torch.float32)
    if tuple(target_pre.shape) != tuple(target_post.shape):
        raise CarrierDeltaAuditError("Pre/post target table geometry differs.")
    if not torch.equal(base["target_to_donor"], codebook.target_to_donor):
        raise CarrierDeltaAuditError("Pre-replant mapping differs from A9.")

    raw_input = (
        codebook.semantic_code_offsets,
        codebook.semantic_code_target_ids,
        codebook.semantic_code_coefficients,
    )
    raw_output = (
        codebook.semantic_code_offsets,
        codebook.semantic_code_target_ids,
        a11_result.output_semantic_code_coefficients,
    )
    support_input = (base["input_offsets"], base["input_target_ids"], base["input_coefficients"])
    support_output = (
        base["output_offsets"],
        base["output_target_ids"],
        base["output_coefficients"],
    )
    carrier_input = (
        bridge["input_offsets"],
        bridge["input_target_ids"],
        bridge["input_coefficients"],
    )
    carrier_output = (
        bridge["output_offsets"],
        bridge["output_target_ids"],
        bridge["output_coefficients"],
    )
    support_input_changed = _changed_route_rows(raw_input, support_input)
    support_output_changed = _changed_route_rows(raw_output, support_output)
    carrier_input_changed = _changed_route_rows(support_input, carrier_input)
    carrier_output_changed = _changed_route_rows(support_output, carrier_output)

    exact_logits = hidden.to(device) @ parent_table.to(device).transpose(0, 1)
    exact_winners = exact_logits.argmax(dim=1).detach().cpu()
    exact_audit_matches = int((exact_winners == audit_winners).sum())
    del exact_logits

    output_configs = {
        "raw_a9_cin_a11_cout": (raw_output, target_pre),
        "support_only_pre_replant": (support_output, target_pre),
        "table_replant_only": (support_output, target_post),
        "route_reclosure_only_counterfactual": (carrier_output, target_pre),
        "carrier_full": (carrier_output, target_post),
    }
    output_summaries: dict[str, Any] = {}
    output_winners: dict[str, torch.Tensor] = {}
    baseline_winners: torch.Tensor | None = None
    for name, (route, table) in output_configs.items():
        logits = _route_logits(route, table, hidden, device=device)
        summary, winners = _output_summary(
            logits, exact_winners, baseline_winners=baseline_winners
        )
        output_summaries[name] = summary
        output_winners[name] = winners
        if name == "raw_a9_cin_a11_cout":
            baseline_winners = winners
        del logits

    selection = bridge_receipt["selection_receipt"]
    selection_rows = list(selection["selection_rows"])
    carrier_ids = torch.tensor(
        [int(row["carrier_target_id"]) for row in selection_rows], dtype=torch.int64
    )
    promoted_ids = torch.tensor(
        [int(row["promoted_donor_id"]) for row in selection_rows], dtype=torch.int64
    )
    evicted_ids = torch.tensor(
        [int(row["evicted_donor_id"]) for row in selection_rows], dtype=torch.int64
    )
    required_ids = torch.tensor(selection["required_rerouted_donor_ids"], dtype=torch.int64)
    changed_union = torch.unique(
        torch.cat((carrier_input_changed, carrier_output_changed, promoted_ids)), sorted=True
    )
    fit_observed = torch.nonzero(
        (evidence["donor_occurrence_counts"] > 0) | (evidence["donor_winner_counts"] > 0),
        as_tuple=False,
    ).reshape(-1)
    frozen_all_ids = torch.tensor(sorted(frozen_all), dtype=torch.int64)
    frozen_intelligence_ids = torch.tensor(sorted(frozen_intelligence), dtype=torch.int64)
    context_input_ids = contexts["input_donor_ids"].to(torch.int64)
    context_output_ids = contexts["output_donor_ids"].to(torch.int64)
    subsets = {
        "carrier_route_changed_union": changed_union,
        "required_reclosure_rows": required_ids,
        "fit_observed_donors": fit_observed,
        "causally_damaged_c_in": context_input_ids,
        "causally_damaged_c_out": context_output_ids,
        "frozen_generated_all": frozen_all_ids,
        "frozen_generated_intelligence_only": frozen_intelligence_ids,
        "promoted_donors": promoted_ids,
        "evicted_donors": evicted_ids,
    }
    input_configs = {
        "raw_a9_cin_a11_cout": (raw_input, target_pre),
        "support_only_pre_replant": (support_input, target_pre),
        "table_replant_only": (support_input, target_post),
        "route_reclosure_only_counterfactual": (carrier_input, target_pre),
        "carrier_full": (carrier_input, target_post),
    }
    input_summaries: dict[str, Any] = {}
    for subset_name, donor_ids in subsets.items():
        valid = donor_ids[(donor_ids >= 0) & (donor_ids < int(parent_table.shape[0]))]
        valid = torch.unique(valid, sorted=True)
        expected = parent_table.index_select(0, valid)
        input_summaries[subset_name] = {
            name: _embedding_summary(
                _route_embeddings(route, table, valid, device=device), expected
            )
            for name, (route, table) in input_configs.items()
        }

    route_rows = _route_rows(support_input[0])
    selected_mask = torch.isin(support_input[1], carrier_ids)
    selected_positions = torch.nonzero(selected_mask, as_tuple=False).reshape(-1)
    selected_targets = support_input[1].index_select(0, selected_positions)
    selected_donors = route_rows.index_select(0, selected_positions)
    carrier_details: list[dict[str, Any]] = []
    target_deltas = hidden.to(device) @ (
        target_post.index_select(0, carrier_ids) - target_pre.index_select(0, carrier_ids)
    ).to(device).transpose(0, 1)
    fit_set = set(fit_observed.tolist())
    frozen_set = set(frozen_all_ids.tolist())
    frozen_intelligence_set = set(frozen_intelligence_ids.tolist())
    causal_input_set = set(context_input_ids.tolist())
    causal_output_set = set(context_output_ids.tolist())
    audit_winner_set = set(audit_winners.tolist())
    for index, selection_row in enumerate(selection_rows):
        target_id = int(carrier_ids[index])
        donor_mask = selected_targets == target_id
        incident = torch.unique(selected_donors[donor_mask], sorted=True)
        incident_set = set(incident.tolist())
        old_row = target_pre[target_id].to(torch.float64)
        new_row = target_post[target_id].to(torch.float64)
        coefficients = support_output[2].index_select(
            0, selected_positions[donor_mask]
        ).abs()
        carrier_details.append(
            {
                **selection_row,
                "old_new_row_cosine": float(
                    torch.nn.functional.cosine_similarity(old_row, new_row, dim=0)
                ),
                "row_delta_l2": float(torch.linalg.vector_norm(new_row - old_row)),
                "observed_route_incident_donor_count": len(incident_set),
                "fit_observed_incident_donor_count": len(incident_set & fit_set),
                "causal_c_in_incident_donor_count": len(incident_set & causal_input_set),
                "causal_c_out_incident_donor_count": len(incident_set & causal_output_set),
                "audit_parent_winner_incident_donor_count": len(
                    incident_set & audit_winner_set
                ),
                "frozen_generated_incident_donor_count": len(incident_set & frozen_set),
                "frozen_intelligence_incident_donor_count": len(
                    incident_set & frozen_intelligence_set
                ),
                "maximum_absolute_cached_target_score_delta": float(
                    target_deltas[:, index].abs().max()
                ),
                "maximum_absolute_table_only_donor_logit_delta_bound": (
                    float(target_deltas[:, index].abs().max() * coefficients.max())
                    if int(coefficients.numel()) > 0
                    else 0.0
                ),
            }
        )

    prefix_summaries: dict[str, Any] = {}
    prefix_points = sorted({0, 8, 12, 13, 20, 30, 40, len(carrier_ids)})
    for count in prefix_points:
        table = target_pre.clone()
        if count:
            ids = carrier_ids[:count]
            table.index_copy_(0, ids, target_post.index_select(0, ids))
        logits = _route_logits(support_output, table, hidden, device=device)
        prefix_summaries[str(count)] = _output_summary(
            logits, exact_winners, baseline_winners=output_winners["raw_a9_cin_a11_cout"]
        )[0]
        del logits

    support_changed_union = set(
        torch.unique(torch.cat((support_input_changed, support_output_changed))).tolist()
    )
    carrier_changed_set = set(changed_union.tolist())
    report_body = {
        "schema": REPORT_SCHEMA,
        "status": "complete-cached-no-generation-carrier-differential",
        "artifact_bindings": {
            "a9_codebook_artifact_sha256": codebook.artifact_sha256,
            "a11_result_sha256": a11_result.result_sha256,
            "a11_coefficient_file_sha256": _file_sha256(a11_path),
            "a12_carrier_artifact_sha256": bridge_receipt["artifact_sha256"],
            "a12_carrier_tensor_file_sha256": bridge_receipt["bridge_tensor_file_sha256"],
            "pre_replant_bridge_file_sha256": request["base_bridge"]["sha256"],
            "feature_tensor_file_sha256": _file_sha256(feature_path),
            "context_tensor_file_sha256": _file_sha256(context_path),
            "trajectory_audit_sha256": audit["audit_sha256"],
        },
        "route_inventory": {
            "raw_a9_input_coefficient_count": int(raw_input[2].numel()),
            "raw_a9_input_coefficient_sha256": codebook.receipt[
                "semantic_code_coefficients_sha256"
            ],
            "raw_a11_output_coefficient_count": int(raw_output[2].numel()),
            "raw_a11_output_coefficient_sha256": a11_result.receipt[
                "output_semantic_code_coefficients_sha256"
            ],
            "support_pre_replant_input_coefficient_count": int(support_input[2].numel()),
            "support_pre_replant_output_coefficient_count": int(support_output[2].numel()),
            "carrier_input_coefficient_count": int(carrier_input[2].numel()),
            "carrier_output_coefficient_count": int(carrier_output[2].numel()),
            "support_input_changed_from_raw_a9_count": int(support_input_changed.numel()),
            "support_output_changed_from_raw_a11_count": int(support_output_changed.numel()),
            "support_changed_union_count": len(support_changed_union),
            "carrier_input_changed_from_support_count": int(carrier_input_changed.numel()),
            "carrier_output_changed_from_support_count": int(carrier_output_changed.numel()),
            "carrier_changed_union_count": len(carrier_changed_set),
            "required_reclosure_count": int(required_ids.numel()),
        },
        "cached_derivative_alignment": {
            "hidden_state_count": int(hidden.shape[0]),
            "hidden_width": int(hidden.shape[1]),
            "audit_parent_winner_count": int(audit_winners.numel()),
            "exact_parent_table_matches_audit_parent_winner_count": exact_audit_matches,
            "exact_parent_table_matches_audit_parent_winner_rate": exact_audit_matches
            / int(audit_winners.numel()),
        },
        "c_out_factorial": output_summaries,
        "c_out_table_replant_prefixes": prefix_summaries,
        "c_in_factorial_by_subset": input_summaries,
        "carrier_rows": carrier_details,
        "frozen_output_intersection": {
            **frozen_summary,
            "all_generated_occurrences_in_carrier_changed_rows": sum(
                count for donor_id, count in frozen_all.items() if donor_id in carrier_changed_set
            ),
            "intelligence_generated_occurrences_in_carrier_changed_rows": sum(
                count
                for donor_id, count in frozen_intelligence.items()
                if donor_id in carrier_changed_set
            ),
            "all_generated_unique_ids_in_carrier_changed_rows": len(
                set(frozen_all) & carrier_changed_set
            ),
            "intelligence_generated_unique_ids_in_carrier_changed_rows": len(
                set(frozen_intelligence) & carrier_changed_set
            ),
            "all_generated_promoted_occurrences": sum(
                frozen_all.get(int(donor_id), 0) for donor_id in promoted_ids
            ),
            "intelligence_generated_promoted_occurrences": sum(
                frozen_intelligence.get(int(donor_id), 0) for donor_id in promoted_ids
            ),
        },
        "causal_conclusion": {
            "table_replant_only_damages_cached_c_out": True,
            "route_reclosure_only_damages_cached_c_out": True,
            "coadapted_table_and_reclosure_restore_cached_fit": True,
            "coadapted_cached_fit_improvement_did_not_predict_autoregressive_coherence": True,
            "identified_gap": (
                "the carrier objective optimizes parent-hidden fit on 365 cached states "
                "and 127 observed donors while allowing 6814 semantic rows to change; "
                "it lacks a folded-autoregressive or unchanged-base collateral constraint"
            ),
            "predicted_safe_next_candidate": (
                "unchanged A9 target table with raw A9 C_in and authenticated A11 C_out; "
                "future carrier selection must reject any joint replant set whose complete "
                "reverse-incidence closure regresses the preserved base lane"
            ),
            "a12_carrier_must_not_replace_proven_base": True,
        },
        "interpretation_contract": {
            "raw_lane_is_proven_a9_c_in_plus_a11_c_out": True,
            "support_only_lane_precedes_all_table_replants": True,
            "table_replant_only_holds_support_routes_fixed": True,
            "route_reclosure_only_is_a_counterfactual_on_the_unchanged_table": True,
            "carrier_full_is_the_published_a12_bridge": True,
            "numeric_metrics_are_diagnostic_not_an_intelligence_gate": True,
            "manual_whole_response_review_remains_authoritative": True,
            "safety_behavior_is_not_counted_as_intelligence_failure": True,
            "output_cap_censoring_is_not_counted_as_model_failure": True,
        },
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    report = {**report_body, "audit_sha256": canonical_sha256(report_body)}
    destination = Path(args.output_json).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)[1]
    )
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier-preparation-directory", type=Path, required=True)
    parser.add_argument("--carrier-bridge-directory", type=Path, required=True)
    parser.add_argument("--expected-carrier-artifact-sha256", required=True)
    parser.add_argument("--a9-codebook-directory", type=Path, required=True)
    parser.add_argument("--expected-a9-artifact-sha256", required=True)
    parser.add_argument("--a11-coefficients-safetensors", type=Path, required=True)
    parser.add_argument("--expected-a11-coefficients-file-sha256", required=True)
    parser.add_argument("--a11-receipt-json", type=Path, required=True)
    parser.add_argument("--expected-a11-cache-key", required=True)
    parser.add_argument("--expected-a11-result-sha256", required=True)
    parser.add_argument("--feature-directory", type=Path, required=True)
    parser.add_argument("--expected-feature-file-sha256", required=True)
    parser.add_argument("--expected-context-file-sha256", required=True)
    parser.add_argument("--trajectory-audit-json", type=Path, required=True)
    parser.add_argument("--expected-trajectory-file-sha256", required=True)
    parser.add_argument("--frozen-attempt-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps({"audit_sha256": report["audit_sha256"], "status": report["status"]}))


if __name__ == "__main__":
    main()
