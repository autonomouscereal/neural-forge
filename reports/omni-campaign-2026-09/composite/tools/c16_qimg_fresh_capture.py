#!/usr/bin/env python3
"""Explicit fresh capture-only entry point; no generation, fit or accepted writer.

The coordinator owns actual runtime identity, GPU approval/allocation/restoration
and hard-stop enforcement. This program validates its independently pinned input
authorization and every byte commitment; it does not authenticate that authority.
All fresh stages are distinct from the permanently nonlaunching v2 metadata draft.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib
from pathlib import Path
import sys
import c16_qimg_fresh_preregistration as gate

SCHEMA = "neural_forge.qimg_fresh_capture_bundle.v1"
RUN_SCHEMA = "neural_forge.qimg_fresh_capture_run.v1"


def _native(ctx, code_root):
    """Called only after all metadata/source pins pass; no model loading here."""
    broad = importlib.import_module("c16_qimg_broad_capture")
    preflight = importlib.import_module("c16_qimg_preflight")
    linalg = importlib.import_module("mediafold.linalg")
    for module in (sys.modules[__name__], broad, broad.capture, broad.prepare,
                   broad.parent, broad.fold, preflight, broad.joint, linalg, sys.modules["mediafold"]):
        matches = [p for p in ctx["recipe"]["source_tool_pins"] if Path(p["path"]).name == Path(module.__file__).name]
        gate.meta.require(len(matches) == 1 and gate.meta.contained(code_root, matches[0]["path"]).resolve()
                          == Path(module.__file__).resolve(), "imported native helper is not pinned recipe source")
    # The recipe also fixes the later development selector without importing or
    # executing it. Its pin must be the real canonical source, not a shadow.
    selector = broad.TOOLS.parents[1] / "media" / "mediafold" / "experimental_fit_selector.py"
    matching = [p for p in ctx["recipe"]["source_tool_pins"] if Path(p["path"]).name == selector.name]
    gate.meta.require(len(matching) == 1 and gate.meta.contained(code_root, matching[0]["path"]).resolve()
                      == selector.resolve(), "selector source pin is a shadow or alias")
    union = {**broad.capture.SOURCE_FILES, **broad.prepare.SOURCE_FILES, **broad.parent.SAMPLING_SOURCE}
    gate.meta.require(union == gate.NATIVE_SOURCE_PINS, "actual native source-code union differs")
    return broad


def _references(root, refs):
    result = []
    for ref in refs:
        gate.meta.reference(root, ref)  # bounded regular strict JSON, no aliases
        result.append({"path": str(gate.meta.contained(root, ref["path"]).resolve()), "sha256": ref["sha256"]})
    return result


def _derive(ctx, broad, parent_receipts, exclusions):
    root, req = ctx["evidence_root"], ctx["request"]
    gate.meta.require(isinstance(parent_receipts, list) and len(parent_receipts) == 4,
                      "four complete parent receipt references required")
    plans = [gate.meta.reference(root, ref) for ref in req["plans"]]
    plan_refs = _references(root, req["plans"])
    receipt_refs = _references(root, parent_receipts)
    batches = [{"number": i, "generation_plan": gp, "parent_receipt": rp}
               for i, (gp, rp) in enumerate(zip(plan_refs, receipt_refs), 1)]
    gate.meta.require(isinstance(exclusions, list), "explicit historical capture exclusions required")
    exclude_refs = _references(root, exclusions)
    # All parent/PNG/source/runtime/seed audits remain in the original shared code.
    derived, sources, old = broad._derive_plans_against(batches, exclude_refs, plans)
    actual = [{"id": entry["id"], "image_sha256": entry["image"]["sha256"],
               "parent_receipt_sha256": entry["image_provenance"]["parent_receipt"]["sha256"]}
              for plan in derived for entry in plan["entries"]]
    gate.meta.require(actual == req["generated_images"], "actual native PNGs/receipts differ from authoritative generated identities")
    gate.meta.require(all(broad.source_content(sources[role]) == expected
                          for role, expected in ctx["generation"]["source_headers"].items()),
                      "actual parent header identities differ from external generation commitments")
    return derived, sources, old, batches, exclude_refs


def _inputs(kwargs):
    # Absolute paths are provenance. Explicit hashes bind bytes at every dispatch.
    return {key: str(Path(value).resolve()) if key.endswith("_path") or key.endswith("_root") else value
            for key, value in kwargs.items()}


def seal(*, parent_receipts, exclusions, out, **kwargs):
    out = Path(out)
    gate.meta.require(not out.exists() and not out.is_symlink(), "seal output exists; no overwrite/cleanup")
    ctx, stage = gate.precapture(**kwargs)
    broad = _native(ctx, kwargs["code_root"])
    derived, sources, old, batches, exclude_refs = _derive(ctx, broad, parent_receipts, exclusions)
    out.mkdir(parents=True, exist_ok=False)
    refs = [broad.exclusive_json(out / f"encoding_batch{i}.json", plan) for i, plan in enumerate(derived, 1)]
    bundle = {"schema": SCHEMA, "status": "sealed_before_encoding_and_capture", "stage_inputs": _inputs(kwargs),
              "stage_audit": stage, "parent_receipts": parent_receipts, "exclusions": exclusions,
              "batches": batches, "excluded_image_sha256": old, "encoding_plans": refs,
              "source_identities": sources, "tool_sha256": broad.tool_hashes(),
              "calibration_seed_rule": broad.SEED_RULE, "sigmas": broad.SIGMAS,
              "expected": {"images": 32, "fit_images": 24, "holdout_images": 8, "forwards": 128,
                           "image_rows": {"fit": 98304, "holdout": 32768}, "image_row_cap": 131072,
                           "minimum_actual_fit_text_rows": 6142},
              "accepted": False, "acceptance_enabled": False,
              "scope": "new staged authority contract: capture-only; no fitting, accepted fold or child claim"}
    return broad.exclusive_json(out / "fresh_capture_bundle.json", bundle)


def validate_bundle(path, sha):
    body = gate.meta.strict_json(path, sha)
    fields = {"schema", "status", "stage_inputs", "stage_audit", "parent_receipts", "exclusions", "batches",
              "excluded_image_sha256", "encoding_plans", "source_identities", "tool_sha256", "calibration_seed_rule",
              "sigmas", "expected", "accepted", "acceptance_enabled", "scope"}
    gate.meta.exact(body, fields, "fresh capture bundle")
    gate.meta.require(body["schema"] == SCHEMA and body["status"] == "sealed_before_encoding_and_capture"
                      and body["accepted"] is False and body["acceptance_enabled"] is False,
                      "fresh capture-only sealed bundle required")
    gate.meta.exact(body["stage_inputs"], {"contract_path", "contract_sha", "request_path", "request_sha", "code_root",
                                          "evidence_root", "anchor_path", "anchor_sha"}, "sealed stage inputs")
    ctx, stage = gate.precapture(**body["stage_inputs"])
    broad = _native(ctx, body["stage_inputs"]["code_root"])
    plans, sources, old, batches, exclude_refs = _derive(ctx, broad, body["parent_receipts"], body["exclusions"])
    gate.meta.require(body["stage_audit"] == stage and body["batches"] == batches
                      and body["source_identities"] == sources and body["excluded_image_sha256"] == old
                      and body["tool_sha256"] == broad.tool_hashes() and body["sigmas"] == broad.SIGMAS
                      and body["calibration_seed_rule"] == broad.SEED_RULE
                      and body["expected"] == {"images": 32, "fit_images": 24, "holdout_images": 8, "forwards": 128,
                          "image_rows": {"fit": 98304, "holdout": 32768}, "image_row_cap": 131072, "minimum_actual_fit_text_rows": 6142}
                      and len(body["encoding_plans"]) == 4, "fresh sealed source/stage/geometry contract changed")
    for expected, ref in zip(plans, body["encoding_plans"]):
        gate.meta.require(gate.meta.strict_json(ref["path"], ref["sha256"]) == expected, "fresh encoding plan bytes/content changed")
    return body, plans, ctx, broad


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(bundle_path, bundle_sha, *, out, weights, vae, text_encoder, comfy_root, threads,
        runtime_authorization_path, runtime_authorization_sha, runtime_anchor_path,
        runtime_anchor_sha):
    out = Path(out)
    gate.meta.require(not out.exists() and not out.is_symlink(), "run output exists; no overwrite/retry/cleanup")
    gate.meta.require(type(threads) is int and 1 <= threads <= 22, "bounded threads 1..22 required")
    bundle, plans, ctx, broad = validate_bundle(bundle_path, bundle_sha)
    def authorize(context):
        return gate.runtime_authorization(context, runtime_authorization_path, runtime_authorization_sha,
            bundle_sha=bundle_sha, runtime_anchor_path=runtime_anchor_path, runtime_anchor_sha=runtime_anchor_sha)
    authorization = authorize(ctx)
    def check():
        current, _, current_ctx, current_broad = validate_bundle(bundle_path, bundle_sha)
        authorize(current_ctx)
        gate.meta.require(current == bundle and current_broad is broad, "sealed bundle/native module changed before dispatch")
    started = _utc_now()
    result = broad._run_audited(bundle_path, bundle_sha, out, weights, vae, text_encoder, comfy_root, threads,
                                bundle, plans, check, before_dispatch=check)
    completed = _utc_now()
    receipt = {"schema": RUN_SCHEMA, "status": "complete", "contract_sha256": ctx["contract_sha256"],
               "recipe_sha256": ctx["request"]["recipe_sha256"], "bundle": broad.reference(bundle_path, bundle_sha),
               "native_audited_run": result, "started_at_utc": started, "completed_at_utc": completed,
               "timestamp_scope": "encode-and-capture pipeline; external start provenance requires postcapture admission",
               "runtime_authorization": broad.reference(runtime_authorization_path, runtime_authorization_sha),
               "runtime_authority_anchor": broad.reference(runtime_anchor_path, runtime_anchor_sha),
               "coordinator_runtime_image_id": authorization["runtime_image_id"],
               "coordinator_physical_gpu_uuid": authorization["physical_gpu_uuid"],
               "external_runtime_state_independently_verified": False,
               "accepted": False, "acceptance_enabled": False,
               "scope": "native encodings and source captures only; no candidate/fitting/accepted writer"}
    return broad.exclusive_json(out / "fresh_capture_run_receipt.json", receipt)


def admit(*, run_receipt_path, run_receipt_sha, **kwargs):
    """Re-audit actual tensor receipts and rows, then check external start binding."""
    ctx, stage = gate.postcapture(run_receipt_path=run_receipt_path, run_receipt_sha=run_receipt_sha, **kwargs)
    receipt = gate.meta.strict_json(run_receipt_path, run_receipt_sha)
    gate.meta.exact(receipt, {"schema", "status", "contract_sha256", "recipe_sha256", "bundle", "native_audited_run",
                             "started_at_utc", "completed_at_utc", "timestamp_scope", "runtime_authorization",
                             "runtime_authority_anchor",
                             "coordinator_runtime_image_id", "coordinator_physical_gpu_uuid",
                             "external_runtime_state_independently_verified", "accepted", "acceptance_enabled", "scope"}, "fresh run receipt")
    ref = receipt["bundle"]
    bundle, plans, original_ctx, broad = validate_bundle(ref["path"], ref["sha256"])
    wanted = dict(ctx["request"]); wanted["capture_start_chronology"] = None
    gate.meta.require(wanted == original_ctx["request"], "postcapture request changed the frozen precapture identities")
    auth_ref, auth_anchor = receipt["runtime_authorization"], receipt["runtime_authority_anchor"]
    auth = gate.runtime_authorization(original_ctx, auth_ref["path"], auth_ref["sha256"], bundle_sha=ref["sha256"],
        runtime_anchor_path=auth_anchor["path"], runtime_anchor_sha=auth_anchor["sha256"],
        now=gate.meta.utc(receipt["started_at_utc"]))
    gate.meta.require(gate.meta.utc(receipt["completed_at_utc"]) <= gate.meta.utc(auth["not_after_utc"])
                      and receipt["coordinator_runtime_image_id"] == auth["runtime_image_id"]
                      and receipt["coordinator_physical_gpu_uuid"] == auth["physical_gpu_uuid"]
                      and receipt["external_runtime_state_independently_verified"] is False,
                      "historical runtime authorization/declared identities differ or expired during run")
    native_ref = receipt["native_audited_run"]
    native = gate.meta.strict_json(native_ref["path"], native_ref["sha256"])
    gate.meta.require(native.get("schema") == broad.RUN_SCHEMA and native.get("status") == "complete"
                      and native.get("bundle") == ref, "native audited run receipt identity differs")
    root = Path(native_ref["path"]).parent
    encodings, captures = [], []
    for i, (plan, plan_ref) in enumerate(zip(plans, bundle["encoding_plans"]), 1):
        encoded = broad.audit_encoding(root / f"encoded_batch{i}", plan, plan_ref, bundle["source_identities"])
        encodings.append(encoded)
        captures.append(broad.audit_capture(root / f"capture_batch{i}", plan, encoded, bundle["source_identities"],
                                            captures[0]["source_frozen_sha256"] if captures else None))
    rows = broad.require_rows(captures)
    hashes = [sha for item in captures for sha in item["forward_file_sha256"]]
    gate.meta.require(native.get("encodings") == encodings and native.get("captures") == captures
                      and native.get("actual_row_counts") == rows and native.get("actual_forwards") == 128
                      and len(hashes) == len(set(hashes)) == 128, "actual capture receipts/forward identities/rows differ")
    stage.update(actual_native_capture_audits_revalidated=True, actual_forwards=128, actual_row_counts=rows)
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    def stage_args(p):
        for name in ("contract-path", "contract-sha", "request-path", "request-sha", "code-root", "evidence-root", "anchor-path", "anchor-sha"):
            p.add_argument("--" + name, required=True)
    p = sub.add_parser("pregeneration"); stage_args(p)
    p = sub.add_parser("seal"); stage_args(p)
    p.add_argument("--parent-receipt", action="append", nargs=2, required=True)
    p.add_argument("--exclude-capture", action="append", nargs=2, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    for name in ("bundle-path", "bundle-sha", "out", "weights", "vae", "text-encoder", "comfy-root", "threads",
                 "runtime-authorization-path", "runtime-authorization-sha", "runtime-anchor-path", "runtime-anchor-sha"):
        p.add_argument("--" + name, required=True, type=int if name == "threads" else str)
    p = sub.add_parser("admit"); stage_args(p)
    p.add_argument("--run-receipt-path", required=True); p.add_argument("--run-receipt-sha", required=True)
    args = vars(parser.parse_args()); command = args.pop("command")
    if command == "pregeneration":
        _, result = gate.pregeneration(**args)
    elif command == "seal":
        args["parent_receipts"] = [{"path": p, "sha256": s} for p, s in args.pop("parent_receipt")]
        args["exclusions"] = [{"path": p, "sha256": s} for p, s in args.pop("exclude_capture")]
        result = seal(**args)
    elif command == "run":
        result = run(**args)
    else:
        result = admit(**args)
    print(gate.json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
