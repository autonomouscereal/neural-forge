#!/usr/bin/env python3
"""Fresh-capture numerical paired-MLP artifact writer; no model installation.

Only a separately verified fresh capture contract admits real inputs. The
development API remains permanently nonaccepting. A synthetic private test path
exercises actual serialization/readback and always publishes accepted=false.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import torch
import c16_qimg_development_refold as development
import c16_qimg_fold as original
import c16_qimg_fresh_capture as fresh
import c16_qimg_fresh_preregistration as gate
import c16_qimg_joint_replay as joint

TOOLS = Path(__file__).resolve().parent
SCOPE = "numerically accepted fresh paired token-local MLP candidate only"
SERIALIZATION_COMPLETE = "complete: shapes/dtypes/byte identity/verbatim/frozen/update/count/held gates"


def _no_larger_claims():
    return {"native_block_accepted": False, "media_installed": False,
            "child_accepted": False, "capability_accepted": False,
            "full_joint_block_replay_performed": False,
            "capability_claim": "none; no native full-block/media/installed-child/retained-intelligence acceptance"}


def _json_exclusive(path, body):
    development._require_finite_diagnostics(body)
    raw = (json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    return {"path": str(path.resolve()), "sha256": original.file_sha(path)}


def _publish_final(path, body):
    """Fsync complete staged bytes, then atomically publish without overwrite.

    Only canonical fold_receipt.json is an authoritative publication marker.
    Pending files and provisional receipts never grant acceptance. No failure-
    prone I/O follows the final atomic link; interrupted stages are preserved.
    """
    path = Path(path)
    development._require_finite_diagnostics(body)
    stage = path.with_name(".fold_receipt.publish_pending.json")
    _json_exclusive(stage, body)
    if hasattr(os, "O_DIRECTORY"):
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    os.link(stage, path)  # atomic and fails if destination exists; same directory


def _finite_tensors(state):
    for name, value in state.items():
        if value.is_floating_point():
            if any(not bool(torch.isfinite(chunk.float()).all()) for chunk in value.reshape(-1).split(1 << 20)):
                raise ValueError("nonfinite stored candidate tensor: " + name)


@torch.no_grad()
def _rescore_persisted(data, candidate, policy, *, device):
    """Score read-back BF16 tensors; no selection, solving, refitting or retry."""
    streams = {}
    for stream in original.STREAMS:
        weights = original.mlp_weights(candidate, data["block"], stream, device)
        per_forward, pools = [], {side: [[], []] for side in ("fit", "holdout")}
        for item in data["forwards"]:
            captured = item[stream]
            _, prediction = development.mlp_forward(captured["x"], weights)
            prediction = prediction.squeeze(0).cpu()[captured["valid"]]
            target = captured["y"].squeeze(0)[captured["valid"]]
            scores = development.score(prediction, target)
            per_forward.append({"file": item["file"], "image_sha256": item["image_sha256"],
                                "split": item["split"], "sigma": item["sigma"], "scores": scores})
            pools[item["split"]][0].append(prediction); pools[item["split"]][1].append(target)
        fit, held = (development.score(torch.cat(pools[side][0]), torch.cat(pools[side][1])) for side in ("fit", "holdout"))
        passed = original.numerical_pass(held, policy["streams"][stream]) and all(
            original.numerical_pass(row["scores"], policy["streams"][stream]) for row in per_forward if row["split"] == "holdout")
        streams[stream] = {"fit_scores": fit, "heldout_scores": held, "per_forward": per_forward,
                           "numerical_policy_passed": passed}
    development._require_finite_diagnostics(streams)
    if not all(s["numerical_policy_passed"] for s in streams.values()):
        raise original.FoldRejected("persisted BF16 held-out gates failed; no accepted receipt", {"streams": streams})
    return streams


def _component_receipt(data, policy, measured, *, accepted=False):
    result = dict(measured)
    result.update(schema=original.FOLD_SCHEMA, status="measured_candidate", fixture=data["fixture"],
                  recipe_id=development.RECIPE_ID, numerical_policy_passed=True,
                  acceptance_enabled=False, accepted=accepted,
                  acceptance_scope=SCOPE if accepted else "provisional or synthetic numerical paired-MLP artifact; not accepted",
                  candidate_written=True, selection_tensors_written=True, policy=policy,
                  **_no_larger_claims())
    # Keep original joint.load_candidate schema compatibility without presenting
    # the development report as an accepted receipt.
    result.pop("development_numeric_gates_passed", None)
    result["streams"] = {stream: dict(values) for stream, values in measured["streams"].items()}
    for values in result["streams"].values():
        values["numerical_policy_passed"] = values.pop("development_numeric_gates_passed")
    return result


def _persist_checked(data, policy, out, rejection, *, device, revalidate, provenance, real_admission):
    candidate = selection = None
    try:
        development._validate_policy(policy)
        original.row_preflight(data, policy)
        revalidate()
        # One FIT-only selection and affine solve. Both full source streams replay
        # first; the reviewed kernel performs all finite/update/count/held gates.
        candidate, selection, measured = development._fit_loaded_candidate(data, policy, device=device)
        _finite_tensors(candidate)
        revalidate()
        out.mkdir(parents=True, exist_ok=False)
        result = _component_receipt(data, policy, measured)
        result.update(provenance)
        result["serialization_validation"] = "pending; no accepted receipt"
        result["output"] = original.write_tensors(out / "folded_source_block_and_time.safetensors", candidate)
        result["selection"] = original.write_tensors(out / "selection.safetensors", selection)
        provisional = _json_exclusive(out / "provisional_fold_receipt.json", result)
        loaded, _ = joint.load_candidate(Path(provisional["path"]), provisional["sha256"], data)
        loaded_selection, _ = original.verified_tensors(out, result["selection"], 1 << 20)
        _finite_tensors(loaded)
        if set(loaded) != set(candidate) or any(not joint.same_bytes(loaded[k], v) for k, v in candidate.items()):
            raise ValueError("serialized/read-back candidate is not byte-identical to fitted coefficients")
        if set(loaded_selection) != set(selection) or any(not joint.same_bytes(loaded_selection[k], v) for k, v in selection.items()):
            raise ValueError("serialized/read-back source-channel selection changed")
        # Structural/frozen/verbatim/update checks came from load_candidate above.
        # Recompute the bias-inclusive count and stored update independently.
        for stream in original.STREAMS:
            prefix = f"transformer_blocks.{data['block']}.{stream}_mlp."
            index = loaded_selection[stream + ".selected_channels"]
            count = sum(loaded[prefix + suffix].numel() for suffix in
                        ("net.0.proj.weight", "net.0.proj.bias", "net.2.weight", "net.2.bias"))
            parent_count = original.biased_gelu_mlp_parameters(data["width"], data["inner"])
            if 4 * count > parent_count: raise ValueError("persisted bias-inclusive fourfold floor failed")
            prior = torch.cat((data["frozen"][prefix + "net.2.weight"].bfloat16().double()[:, index],
                               data["frozen"][prefix + "net.2.bias"].bfloat16().double()[:, None]), dim=1)
            stored = torch.cat((loaded[prefix + "net.2.weight"].double(), loaded[prefix + "net.2.bias"].double()[:, None]), dim=1)
            update = float((stored-prior).norm() / prior.norm().clamp_min(1e-30))
            if not math.isfinite(update) or update > 1: raise ValueError("persisted BF16 update veto failed")
            if count != result["streams"][stream]["child_parameters"]: raise ValueError("persisted parameter count changed")
            result["streams"][stream]["stored_update_scale"] = update
        persisted = _rescore_persisted(data, loaded, policy, device=device)
        for stream, scores in persisted.items(): result["streams"][stream].update(scores)
        revalidate()
        # Recheck both files after scoring and before final publication.
        final_candidate, _ = original.verified_tensors(out, result["output"], 2 << 30)
        final_selection, _ = original.verified_tensors(out, result["selection"], 1 << 20)
        if any(not joint.same_bytes(final_candidate[k], v) for k, v in loaded.items()) or any(
                not joint.same_bytes(final_selection[k], v) for k, v in loaded_selection.items()):
            raise ValueError("persisted tensors changed before final receipt")
        if real_admission and data["fixture"]: raise ValueError("synthetic fixture cannot receive production acceptance")
        result.update(accepted=bool(real_admission and not data["fixture"]),
                      acceptance_enabled=bool(real_admission and not data["fixture"]),
                      serialization_validation=SERIALIZATION_COMPLETE,
                      persisted_numeric_gates_passed=True, persisted_tensors_rescored_without_refit=True,
                      fit_or_selector_retry=False, finalized_at_utc=datetime.now(timezone.utc).isoformat())
        result["acceptance_scope"] = SCOPE if result["accepted"] else "synthetic numerical paired-MLP artifact only; not accepted"
        result["publication_protocol"] = "only canonical fold_receipt.json is an atomic exclusive publication marker; pending/provisional files grant no acceptance"
        # Flush both actual artifact bodies before publishing any acceptance.
        for record in (result["output"], result["selection"]):
            with original.artifact(out, record["file"]).open("rb") as handle: os.fsync(handle.fileno())
        _publish_final(out / "fold_receipt.json", result)
        return result
    except Exception as exc:
        # Never publish an accepted receipt for a failed reload/recheck/score.
        receipt = {"schema": original.FOLD_SCHEMA, "status": "rejected_fresh_fold_or_serialization",
                   "fixture": bool(data["fixture"]), "accepted": False, "acceptance_enabled": False,
                   "candidate_written": (out / "folded_source_block_and_time.safetensors").is_file(),
                   "selection_tensors_written": (out / "selection.safetensors").is_file(),
                   "staged_receipt_present": (out / ".fold_receipt.publish_pending.json").is_file(),
                   "staged_receipt_authoritative": False,
                   "canonical_receipt_present": (out / "fold_receipt.json").exists(),
                   "canonical_receipt_published_by_this_writer": False,
                   "publication_scope": "pending bytes may contain completed diagnostics; only the canonical acceptance reader grants component acceptance",
                   "rejection_reason": str(exc), "error_type": type(exc).__name__, **_no_larger_claims(), **provenance}
        if isinstance(exc, (original.FoldRejected, development.DevelopmentRejected)):
            receipt["measured_refusal"] = exc.receipt
        development._require_finite_diagnostics(receipt)
        _json_exclusive(rejection, receipt)
        raise original.FoldRejected(str(exc), receipt) from exc


def _code_and_policy(ctx, kwargs, policy_path, policy_sha):
    for module in (sys.modules[__name__], development, sys.modules["mediafold.experimental_fit_selector"]):
        matches = [p for p in ctx["recipe"]["source_tool_pins"] if Path(p["path"]).name == Path(module.__file__).name]
        gate.meta.require(len(matches) == 1 and gate.meta.contained(kwargs["code_root"], matches[0]["path"]).resolve()
                          == Path(module.__file__).resolve(), "writer/kernel/selector must be pinned actual recipe source before freeze")
    matching = [p for p in ctx["recipe"]["source_tool_pins"] if gate.meta.contained(kwargs["code_root"], p["path"]).resolve() == Path(policy_path).resolve()]
    gate.meta.require(len(matching) == 1 and matching[0]["sha256"] == policy_sha, "exact policy bytes must be in frozen recipe source pins")
    policy = original.load_policy(Path(policy_path), policy_sha)
    development._validate_policy(policy)
    gate.meta.require(gate.meta.same(policy["streams"], ctx["recipe"]["streams"]), "policy differs from frozen selector recipe gates")
    return policy


def convert_fresh(*, run_receipt_path, run_receipt_sha, policy_path, policy_sha, out,
                  rejection_receipt=None, device="cpu", **kwargs):
    """Only production entry point; captures derive solely from admitted run."""
    out, policy_path = Path(out), Path(policy_path)
    rejection = Path(rejection_receipt) if rejection_receipt is not None else out.with_name(out.name + ".rejected.json")
    # Early destination guards precede heavy native capture or solver I/O.
    if out.exists() or out.is_symlink() or rejection.exists() or rejection.is_symlink():
        raise ValueError("new separate output/rejection required; no overwrite/cleanup")
    def overlaps(left, right):
        left, right = Path(left).resolve(), Path(right).resolve()
        return left == right or left in right.parents or right in left.parents
    protected = [kwargs["code_root"], kwargs["evidence_root"], run_receipt_path, policy_path,
                 *(v for k, v in kwargs.items() if k.endswith("_path"))]
    if overlaps(out, rejection) or any(overlaps(destination, source) for destination in (out, rejection) for source in protected):
        raise ValueError("output/rejection overlaps source, authority or evidence inputs")
    if device not in {"cpu", "cuda:0"}: raise ValueError("only original CPU/cuda:0 numerical devices supported")
    ctx = gate.load(**kwargs)
    policy = _code_and_policy(ctx, kwargs, policy_path, policy_sha)
    admission = fresh.admit(run_receipt_path=run_receipt_path, run_receipt_sha=run_receipt_sha, **kwargs)
    run = gate.meta.strict_json(run_receipt_path, run_receipt_sha)
    native_ref = run["native_audited_run"]
    native = gate.meta.strict_json(native_ref["path"], native_ref["sha256"])
    refs = [batch["capture_receipt"] for batch in native["captures"]]
    receipts = [Path(ref["path"]) for ref in refs]
    rejection = original.destinations_preflight(receipts, policy_path, out, rejection)
    data = original.load_captures(receipts, fixture=False)
    if data["capture_receipts"] != refs: raise ValueError("loaded capture identities differ from admitted fresh run")
    def revalidate():
        current_ctx = gate.load(**kwargs)
        current_policy = _code_and_policy(current_ctx, kwargs, policy_path, policy_sha)
        current = fresh.admit(run_receipt_path=run_receipt_path, run_receipt_sha=run_receipt_sha, **kwargs)
        if current_policy != policy or current != admission: raise ValueError("fresh admission/policy changed around fitting or serialization")
    provenance = {"fresh_capture_run": {"path": str(Path(run_receipt_path).resolve()), "sha256": run_receipt_sha},
                  "fresh_capture_admission": admission, "policy_sha256": policy_sha,
                  "contract_sha256": ctx["contract_sha256"], "recipe_sha256": ctx["request"]["recipe_sha256"],
                  "source_commit": ctx["recipe"]["source_commit"],
                  "capture_contract_scope": "current versioned capture-only proposal; full QA scope is not an original fold-policy requirement",
                  "runtime_resource_authorization": "independent coordinator obligation; writer does not allocate, stop or authenticate runtime"}
    return _persist_checked(data, policy, out, rejection, device=device, revalidate=revalidate,
                            provenance=provenance, real_admission=True)


def _published_header(receipt_path, receipt_sha):
    """Publication identity first; a renamed/provisional path is never a marker."""
    path = Path(receipt_path)
    if (path.name != "fold_receipt.json" or path.is_symlink() or not path.is_file()
            or path.absolute() != path.resolve()):
        raise ValueError("canonical regular non-symlink fold_receipt.json required")
    receipt = gate.meta.strict_json(path, receipt_sha)
    required_true = ("accepted", "acceptance_enabled", "numerical_policy_passed",
                     "persisted_numeric_gates_passed", "persisted_tensors_rescored_without_refit",
                     "candidate_written", "selection_tensors_written")
    required_false = ("fixture", "fit_or_selector_retry", "native_block_accepted", "media_installed",
                      "child_accepted", "capability_accepted", "full_joint_block_replay_performed")
    if (receipt.get("schema") != original.FOLD_SCHEMA or receipt.get("status") != "measured_candidate"
            or receipt.get("recipe_id") != development.RECIPE_ID
            or receipt.get("acceptance_scope") != SCOPE
            or receipt.get("serialization_validation") != SERIALIZATION_COMPLETE
            or any(receipt.get(key) is not True for key in required_true)
            or any(receipt.get(key) is not False for key in required_false)):
        raise ValueError("canonical receipt does not grant the narrow persisted nonfixture component acceptance")
    development._require_finite_diagnostics(receipt)
    return path, receipt


@torch.no_grad()
def load_accepted_component(*, receipt_path, receipt_sha, run_receipt_path, run_receipt_sha,
                            policy_path, policy_sha, device="cpu", **kwargs):
    """Read only canonical accepted paired MLPs, re-admitting real capture inputs.

    This is the acceptance consumer. The original joint.load_candidate remains
    a structural loader and grants no publication/component acceptance itself.
    No full-block/child installation is performed or authorized by this API.
    """
    path, receipt = _published_header(receipt_path, receipt_sha)
    if device not in {"cpu", "cuda:0"}: raise ValueError("only original CPU/cuda:0 numerical devices supported")
    def identities():
        ctx = gate.load(**kwargs)
        policy = _code_and_policy(ctx, kwargs, policy_path, policy_sha)
        admission = fresh.admit(run_receipt_path=run_receipt_path, run_receipt_sha=run_receipt_sha, **kwargs)
        expected = {"fresh_capture_run": {"path": str(Path(run_receipt_path).resolve()), "sha256": run_receipt_sha},
                    "fresh_capture_admission": admission, "policy_sha256": policy_sha,
                    "contract_sha256": ctx["contract_sha256"], "recipe_sha256": ctx["request"]["recipe_sha256"],
                    "source_commit": ctx["recipe"]["source_commit"], "policy": policy}
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("published component identities differ from fresh admission/frozen policy/source")
        return policy, admission
    policy, admission = identities()
    run = gate.meta.strict_json(run_receipt_path, run_receipt_sha)
    native_ref = run["native_audited_run"]
    native = gate.meta.strict_json(native_ref["path"], native_ref["sha256"])
    refs = [batch["capture_receipt"] for batch in native["captures"]]
    data = original.load_captures([Path(ref["path"]) for ref in refs], fixture=False)
    if data["fixture"] is not False or data["capture_receipts"] != refs:
        raise ValueError("component source differs from admitted nonfixture capture identities")
    if receipt.get("capture_receipts") != refs or any(
            receipt.get("streams", {}).get(stream, {}).get("keep") != policy["streams"][stream]["keep"]
            for stream in original.STREAMS):
        raise ValueError("component capture references or widths differ from frozen policy")
    original.row_preflight(data, policy)
    candidate, identity = joint.load_candidate(path, receipt_sha, data)
    _finite_tensors(candidate)
    persisted = _rescore_persisted(data, candidate, policy, device=device)
    if identities() != (policy, admission): raise ValueError("source/admission changed during component load")
    if _published_header(path, receipt_sha)[1] != receipt: raise ValueError("publication marker changed during component load")
    checked, _ = original.verified_tensors(path.parent, receipt["output"], 2 << 30)
    original.verified_tensors(path.parent, receipt["selection"], 1 << 20)
    if set(checked) != set(candidate) or any(not joint.same_bytes(checked[k], value) for k, value in candidate.items()):
        raise ValueError("component tensors changed during persisted scoring")
    return candidate, {"receipt": {"path": str(path.resolve()), "sha256": receipt_sha},
                       "acceptance_scope": SCOPE, "structural_identity": identity,
                       "persisted_scores_revalidated": persisted, **_no_larger_claims()}


def _convert_fixture_for_tests(data, policy, out, *, device="cpu", revalidate=lambda: None):
    """Synthetic internal test API only; no production acceptance or CLI flag."""
    if data.get("fixture") is not True: raise ValueError("explicit synthetic fixture required")
    out = Path(out); rejection = out.with_name(out.name + ".rejected.json")
    if out.exists() or rejection.exists() or out.is_symlink() or rejection.is_symlink(): raise ValueError("fixture outputs must be new")
    return _persist_checked(data, policy, out, rejection, device=device, revalidate=revalidate,
                            provenance={"input_contract": "synthetic CPU tensor fixture; no real capture/authority"}, real_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract-path", "contract-sha", "request-path", "request-sha", "code-root", "evidence-root", "anchor-path", "anchor-sha",
                 "run-receipt-path", "run-receipt-sha", "policy-path", "policy-sha", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--rejection-receipt")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    try:
        result = convert_fresh(**vars(parser.parse_args()))
    except original.FoldRejected as exc:
        print(json.dumps(exc.receipt, sort_keys=True, indent=2, allow_nan=False)); return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
