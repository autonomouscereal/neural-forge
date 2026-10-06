#!/usr/bin/env python3
"""Separate nonaccepting QIMG development refold: JSON diagnostics, never tensors.

The tensor API uses caller-supplied loaded tensors. The development CLI reuses
the original verified capture/policy loaders; their historical preregistration
flags do not establish fresh-data acceptance for this experimental selector.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import torch

TOOLS = Path(__file__).resolve().parent
MEDIA = TOOLS.parents[1] / "media"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(MEDIA))
from c16_qimg_fold import (STREAMS, POLICY_SCHEMA, MIN_FIT_ROWS_PER_KEEP,
    load_policy, load_captures, row_preflight, mlp_weights,
    mlp_forward as _source_mlp_forward, score as _source_score,
    numerical_pass, biased_gelu_mlp_parameters, file_sha, tensor_sha)
from mediafold.linalg import law_abiding_solve
from mediafold.experimental_fit_selector import RECIPE_ID, select_fit_output_weighted_columns

DEVELOPMENT_SCHEMA = "neural_forge.qimg_development_refold.v1"
MAX_SOURCE_REPLAY_REL_L2 = 0.005
MAX_DEVELOPMENT_HELD_REL_L2 = 0.30
MIN_DEVELOPMENT_HELD_COS = 0.95


def _nonaccepting():
    return {"schema": DEVELOPMENT_SCHEMA, "recipe_id": RECIPE_ID,
        "scope": "development_diagnostics_only",
        "accepted": False, "acceptance_enabled": False, "candidate_written": False,
        "selection_tensors_written": False,
        "fresh_data_acceptance_validated": False,
        "historical_holdout_scope": "previously observed holdouts are development data",
        "capability_claim": "none: no accepted fold, installed model, complete child or retained intelligence proof"}


class DevelopmentRejected(ValueError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = {**receipt, **_nonaccepting()}
        status = self.receipt.get("status", "development_refused")
        self.receipt["status"] = status if status.startswith("development_") else "development_" + status


class NonFiniteDiagnosticError(ValueError):
    """Non-finite arithmetic is refused, never serialized as a passing score."""


def _require_finite_diagnostics(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise NonFiniteDiagnosticError("non-finite solver or receipt diagnostic")
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("diagnostic metadata keys must be strings")
        for item in value.values(): _require_finite_diagnostics(item)
    elif isinstance(value, (list, tuple)):
        for item in value: _require_finite_diagnostics(item)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError("diagnostics must contain JSON metadata only, never tensors")


def mlp_forward(x, weights):
    try:
        hidden, predicted = _source_mlp_forward(x, weights)
    except ValueError as exc:
        if "non-finite" in str(exc):
            raise NonFiniteDiagnosticError(str(exc)) from exc
        raise
    if not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(predicted).all()):
        raise NonFiniteDiagnosticError("non-finite native BF16 source/candidate MLP arithmetic")
    return hidden, predicted


def score(actual, target):
    try:
        scores = _source_score(actual, target)
    except ValueError as exc:
        if "non-finite" in str(exc):
            raise NonFiniteDiagnosticError(str(exc)) from exc
        raise
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) for value in scores.values()):
        raise NonFiniteDiagnosticError("non-finite numerical diagnostic; no numeric gate comparison made")
    return scores


def _validate_policy(policy):
    # Same schema, exact gate fields and preregistration assertion as the original
    # loader, plus ceilings/floor that prohibit weakening the frozen broad gates.
    allowed = {"schema", "preregistered_before_scoring", "gate_scope", "streams", "policy_id", "description"}
    if not isinstance(policy, dict) or set(policy) - allowed:
        raise ValueError("unsupported development policy fields")
    for field in ("policy_id", "description"):
        if field in policy and not isinstance(policy[field], str):
            raise ValueError("optional policy metadata must be strings")
    if policy.get("schema") != POLICY_SCHEMA or policy.get("preregistered_before_scoring") is not True:
        raise ValueError("original caller-pinned policy contract required; it does not establish fresh acceptance")
    if policy.get("gate_scope") != "pooled_and_each_heldout_forward" or set(policy.get("streams", {})) != set(STREAMS):
        raise ValueError("both streams and pooled/every-forward numeric gates required")
    for stream in STREAMS:
        entry = policy["streams"][stream]
        if not isinstance(entry, dict) or set(entry) != {"keep", "max_source_replay_rel_l2", "max_heldout_rel_l2", "min_heldout_cos"}:
            raise ValueError("all original stream gates must be explicit")
        keep = entry["keep"]
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            raise ValueError("keep must be a positive integer")
        for key in ("max_source_replay_rel_l2", "max_heldout_rel_l2", "min_heldout_cos"):
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("finite numerical gates required")
        if not 0 <= entry["max_source_replay_rel_l2"] <= MAX_SOURCE_REPLAY_REL_L2:
            raise ValueError("source replay gate cannot weaken the frozen 0.005 ceiling")
        if not 0 <= entry["max_heldout_rel_l2"] <= MAX_DEVELOPMENT_HELD_REL_L2:
            raise ValueError("development error gate cannot weaken the frozen 0.30 ceiling")
        if not MIN_DEVELOPMENT_HELD_COS <= entry["min_heldout_cos"] <= 1:
            raise ValueError("development cosine gate cannot weaken the frozen 0.95 floor")


@torch.no_grad()
def _measure_loaded(data, policy, *, device="cpu"):
    rows = row_preflight(data, policy)
    baseline, features = {}, {}
    # BOTH streams must replay before EITHER stream reaches pivoting/refitting.
    for stream in STREAMS:
        weights = mlp_weights(data["frozen"], data["block"], stream, device)
        baseline[stream], fit_hidden = [], []
        for item in data["forwards"]:
            captured = item[stream]
            hidden, predicted = mlp_forward(captured["x"], weights)
            scores = score(predicted, captured["y"])
            baseline[stream].append({"file": item["file"], "split": item["split"], "scores": scores})
            if scores["rel_l2"] > policy["streams"][stream]["max_source_replay_rel_l2"]:
                raise DevelopmentRejected("source BF16/FP8 replay does not match captured MLP output; no fold attempted",
                                   {"status": "rejected_source_replay", "source_replay": baseline, "fixture": data["fixture"]})
            if item["split"] == "fit":
                fit_hidden.append(hidden.squeeze(0).cpu()[captured["valid"]])
        features[stream] = torch.cat(fit_hidden)
        del weights, fit_hidden
    candidate = dict(data["frozen"])
    measured = {}
    for stream in STREAMS:
        entry = policy["streams"][stream]
        keep, width, inner = entry["keep"], data["width"], data["inner"]
        hidden = features.pop(stream)
        native_parent_w2 = data["frozen"][f"transformer_blocks.{data['block']}.{stream}_mlp.net.2.weight"].to(torch.bfloat16)
        selection = select_fit_output_weighted_columns(hidden, native_parent_w2, keep)
        del native_parent_w2
        prefix = f"transformer_blocks.{data['block']}.{stream}_mlp."
        # Bias is a counted coefficient and has its own prior column. Neither
        # the target bias nor the parent anchor is silently discarded/zeroed.
        prior_w = data["frozen"][prefix + "net.2.weight"].to(torch.bfloat16).double()[:, selection]
        prior_b = data["frozen"][prefix + "net.2.bias"].to(torch.bfloat16).double()[:, None]
        prior = torch.cat((prior_w, prior_b), dim=1)
        design = torch.cat((hidden[:, selection].double(), torch.ones((len(hidden), 1), dtype=torch.float64)), dim=1)
        targets = torch.cat([item[stream]["y"].squeeze(0)[item[stream]["valid"]]
                             for item in data["forwards"] if item["split"] == "fit"]).double()
        fitted, solve = law_abiding_solve(design, targets, prior)
        _require_finite_diagnostics(solve)
        if not bool(torch.isfinite(fitted).all()):
            raise NonFiniteDiagnosticError("non-finite closed-form solution")
        stored = fitted.to(torch.bfloat16)
        if not bool(torch.isfinite(stored).all()):
            raise DevelopmentRejected("finite solver coefficients overflowed native BF16 storage",
                {"status": "development_rejected_nonfinite_bf16_storage", "stream": stream,
                 "nonfinite_arithmetic_refused": True, "stored_update_scale": None,
                 "stored_update_classification": "not computed: BF16 coefficients are non-finite"})
        stored_update = float((stored.double() - prior).norm() / prior.norm().clamp_min(1e-30))
        if not math.isfinite(stored_update):
            raise DevelopmentRejected("non-finite native BF16 update diagnostic",
                {"status": "development_rejected_nonfinite_update", "stream": stream,
                 "nonfinite_arithmetic_refused": True, "stored_update_scale": None,
                 "stored_update_classification": "non-finite; no gate comparison made"})
        if stored_update > 1:
            raise DevelopmentRejected("stored BF16 refit violates the existing update_scale <= 1 veto",
                               {"status": "rejected_stored_update", "stream": stream, "stored_update_scale": stored_update})
        for name in ("net.0.proj.weight", "net.0.proj.bias"):
            original = data["frozen"][prefix + name]
            # CPU advanced indexing lacks FP8 support in some pinned runtimes;
            # gather the original one-byte representation without quantization.
            candidate[prefix + name] = original.view(torch.uint8).index_select(0, selection).contiguous().view(original.dtype)
        candidate[prefix + "net.2.weight"] = stored[:, :-1].contiguous()
        candidate[prefix + "net.2.bias"] = stored[:, -1].contiguous()
        del fitted, design, targets, hidden, prior_w, prior_b, prior
        weights = mlp_weights(candidate, data["block"], stream, device)
        per_forward, pooled = [], {split: [[], []] for split in ("fit", "holdout")}
        for item in data["forwards"]:
            captured = item[stream]
            _, prediction = mlp_forward(captured["x"], weights)
            prediction = prediction.squeeze(0).cpu()[captured["valid"]]
            target = captured["y"].squeeze(0)[captured["valid"]]
            scores = score(prediction, target)
            per_forward.append({"file": item["file"], "image_sha256": item["image_sha256"],
                                "split": item["split"], "sigma": item["sigma"], "scores": scores})
            pooled[item["split"]][0].append(prediction)
            pooled[item["split"]][1].append(target)
        fit_scores, held_scores = (score(torch.cat(pooled[split][0]), torch.cat(pooled[split][1])) for split in ("fit", "holdout"))
        passed = numerical_pass(held_scores, entry) and all(numerical_pass(row["scores"], entry) for row in per_forward if row["split"] == "holdout")
        parent_count = biased_gelu_mlp_parameters(width, inner)
        child_count = sum(candidate[prefix + name].numel() for name in
                          ("net.0.proj.weight", "net.0.proj.bias", "net.2.weight", "net.2.bias"))
        if 4 * child_count > parent_count:
            raise ValueError("stored component does not satisfy the bias-inclusive fourfold floor")
        measured[stream] = {"keep": keep, "parent_parameters": parent_count, "child_parameters": child_count,
            "component_compression_x": parent_count / child_count, "solve": solve,
            "stored_update_scale": stored_update, "fit_scores": fit_scores, "heldout_scores": held_scores,
            "per_forward": per_forward, "development_numeric_gates_passed": passed}
        measured[stream]["selected_channel_indices"] = selection.tolist()
        measured[stream]["ephemeral_refit_sha256"] = {name: tensor_sha(candidate[prefix + name])
            for name in ("net.2.weight", "net.2.bias")}
        measured[stream]["ephemeral_verbatim_projection_sha256"] = {name: tensor_sha(candidate[prefix + name])
            for name in ("net.0.proj.weight", "net.0.proj.bias")}
        del weights, pooled
    result = {"schema": DEVELOPMENT_SCHEMA, "status": "development_complete", "fixture": data["fixture"],
        "block": data["block"], "source_replay": baseline, "streams": measured, "row_preflight": rows,
        "policy": policy, "image_splits": data["image_splits"], "capture_receipts": data["capture_receipts"],
        "source_frozen_file_sha256": data["frozen_file_sha256"],
        "source_parent_identity_scope": "capture parent header plus complete selected block/time and captured-file hashes; not a newly verified full parent-file digest",
        "development_numeric_gates_passed": all(m["development_numeric_gates_passed"] for m in measured.values()),
        "runtime": {"device": device, "torch": torch.__version__, "compute_dtype": "torch.bfloat16",
                    "solver_dtype": "torch.float64", "torch_threads": torch.get_num_threads()},
        "law": {"selection": RECIPE_ID,
                "refit": "existing law_abiding_solve, with parent-anchored affine bias column",
                "min_fit_rows_per_keep": MIN_FIT_ROWS_PER_KEEP, "heldout_used_for_selection_or_refit": False,
                "text_rows": "valid context rows only for fit/heldout; source replay covers all captured text rows"},
        "accounting": {"parent_block_and_time_parameters": sum(t.numel() for t in data["frozen"].values()),
                       "child_block_and_time_parameters": sum(t.numel() for t in candidate.values()),
                       "selection_indices_are_parameters": False},
        "runtime_requirements": {"native_qimg_dual_stream_runtime": "still required; generic adaln6 is insufficient",
                                 "paired_host_source_charts": "not supplied or invented", "full_joint_block_replay": "not performed by this token-local MLP fold",
                                 "weight_storage": "preserved FP8 source plus BF16 refit; a loader that globally recasts this candidate to FP8 is not equivalent"},
        "capability_claim": "none: measured MLP candidate is not installed, behaviorally accepted, or a media/all-source child proof"}
    if not result["development_numeric_gates_passed"]:
        result["status"] = "development_rejected_numerical_policy"
        raise DevelopmentRejected("development numerical gates failed; no acceptance or tensors written", result)
    return result


def implementation_sha256():
    return {"c16_qimg_development_refold.py": file_sha(Path(__file__)),
        "mediafold/experimental_fit_selector.py": file_sha(MEDIA / "mediafold/experimental_fit_selector.py"),
        "c16_qimg_fold.py": file_sha(TOOLS / "c16_qimg_fold.py"),
        "c16_qimg_capture.py": file_sha(TOOLS / "c16_qimg_capture.py"),
        "c16_qimg_preflight.py": file_sha(TOOLS / "c16_qimg_preflight.py"),
        "mediafold/linalg.py": file_sha(MEDIA / "mediafold/linalg.py")}


def fold_loaded(data, policy, *, device="cpu"):
    """Tensor-only diagnostic API: return metadata/scores, never candidate tensors.

    Caller must establish loaded-tensor provenance. Image/split and file identities
    are independently checked only by the CLI's original verified capture loader.
    Both streams replay before either is selected or fitted. Old/new held-out rows
    are scored only after FIT-only selection/refit; all outcomes remain nonaccepting.
    """
    if device not in {"cpu", "cuda:0"}:
        raise ValueError("only the original cpu/cuda:0 replay devices are supported")
    provenance = {"implementation_sha256": implementation_sha256(),
        "input_contract": "tensor API: caller-supplied loaded tensors; file/split provenance not established by this API",
        "selector_chunk_rows": 512}
    try:
        _validate_policy(policy)
        result = _measure_loaded(data, policy, device=device)
    except DevelopmentRejected as exc:
        exc.receipt.update(provenance)
        try:
            _require_finite_diagnostics(exc.receipt)
        except (ValueError, TypeError) as bad:
            raise DevelopmentRejected(str(bad), {"status": "development_rejected_nonfinite_receipt" if isinstance(bad, NonFiniteDiagnosticError) else "development_refused_receipt_contract",
                "original_refusal_status": exc.receipt["status"],
                "nonfinite_arithmetic_refused": isinstance(bad, NonFiniteDiagnosticError), **provenance}) from exc
        raise
    except NonFiniteDiagnosticError as exc:
        raise DevelopmentRejected(str(exc), {"status": "development_rejected_nonfinite_arithmetic",
            "nonfinite_arithmetic_refused": True, "error_type": type(exc).__name__, **provenance}) from exc
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise DevelopmentRejected(str(exc), {"status": "development_refused_tensor_contract",
            "error_type": type(exc).__name__, **provenance}) from exc
    result = {**result, **provenance, **_nonaccepting()}
    try:
        _require_finite_diagnostics(result)
    except NonFiniteDiagnosticError as exc:
        raise DevelopmentRejected(str(exc), {"status": "development_rejected_nonfinite_receipt",
            "nonfinite_arithmetic_refused": True, **provenance}) from exc
    except (ValueError, TypeError) as exc:
        raise DevelopmentRejected(str(exc), {"status": "development_refused_receipt_contract",
            "error_type": type(exc).__name__, **provenance}) from exc
    return result


def report_preflight(receipts, policy_path, report):
    """New separate JSON only; refuse overwrites/source overlap before tensor I/O."""
    if report.exists() or report.is_symlink():
        raise ValueError("report must be a new file; no overwrite or cleanup")
    destination = report.resolve()
    def overlaps(left, right):
        return left == right or left in right.parents or right in left.parents
    sources = [*(p.parent.resolve() for p in receipts), TOOLS.resolve(), MEDIA.resolve(),
               *(p.resolve() for p in receipts), policy_path.resolve()]
    if any(overlaps(destination, source) for source in sources):
        raise ValueError("development report overlaps an input or code/source directory")
    return destination


def _write_report(receipts, policy_path, report, receipt):
    payload = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    report = report_preflight(receipts, policy_path, report)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def run_development(receipts, policy_path, policy_sha256, report, *,
                    development_only=False, device="cpu", validate_only=False, fixture=False):
    """Original verified load gates plus exclusively written nonaccepting JSON.

    ``fixture`` exists for synthetic tests; the CLI exposes no fixture bypass.
    There is no candidate output path, tensor writer, installer or acceptance mode.
    Validate-only still loads/verifies capture tensors and applies row/count gates.
    """
    report = report_preflight(receipts, policy_path, report)
    common = {**_nonaccepting(), "report": str(report),
        "implementation_sha256": implementation_sha256(),
        "policy_sha256": policy_sha256,
        "policy_preregistration_scope": "caller-pinned bytes and historical assertion; new recipe chronology/freshness not established",
        "fixture": fixture}
    policy, data = None, None
    try:
        if development_only is not True:
            raise ValueError("explicit development_only=True is required; acceptance is unavailable")
        policy = load_policy(policy_path, policy_sha256)
        _validate_policy(policy)
        data = load_captures(receipts, fixture=fixture)
        common.update({"policy": policy, "capture_receipts": data["capture_receipts"],
            "source_frozen_file_sha256": data["frozen_file_sha256"],
            "input_contract": "original verified capture/policy loaders; no fresh-data acceptance attestation"})
        preflight = row_preflight(data, policy)
        if validate_only:
            result = {"status": "development_inputs_validated", "row_preflight": preflight,
                "source_replay_performed": False, "fold_launched": False,
                "development_numeric_gates_passed": None}
        else:
            result = fold_loaded(data, policy, device=device)
        result.update(common)
    except DevelopmentRejected as exc:
        exc.receipt.update(common)
        exc.receipt["rejection_reason"] = str(exc)
        _write_report(receipts, policy_path, report, exc.receipt)
        raise
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        result = {"status": "development_refused_inputs_or_runtime", "error_type": type(exc).__name__,
            "rejection_reason": str(exc), **common}
        _write_report(receipts, policy_path, report, result)
        raise DevelopmentRejected(str(exc), result) from exc
    _write_report(receipts, policy_path, report, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-only", action="store_true", required=True,
                        help="explicit nonaccepting diagnostics; no acceptance mode exists")
    parser.add_argument("--captures", nargs="+", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True, help="new separate JSON receipt; never tensors")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_development(args.captures, args.policy, args.policy_sha256, args.report,
            development_only=args.development_only, device=args.device, validate_only=args.validate_only)
    except DevelopmentRejected as exc:
        print(json.dumps({"status": exc.receipt["status"], "report": str(args.report),
            "report_sha256": file_sha(args.report), **_nonaccepting()}, allow_nan=False))
        return 2
    print(json.dumps({"status": result["status"], "report": str(args.report),
        "report_sha256": file_sha(args.report), **_nonaccepting()}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
