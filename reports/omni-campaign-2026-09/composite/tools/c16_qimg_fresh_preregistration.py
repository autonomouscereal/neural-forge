#!/usr/bin/env python3
"""Staged capture-only authority contract. No model imports or launcher.

The old metadata proposal and its permanently false readiness flags are unchanged.
This distinct API validates commitments, not the independent origin, freshness or
truth of caller-supplied authority. Real execution additionally requires a pinned
coordinator authorization; no stage can authorize accepted-fold artifacts.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import c16_qimg_fresh_metadata as meta

CONTRACT_SCHEMA = "neural_forge.qimg_fresh_capture_contract.v1"
REQUEST_SCHEMA = "neural_forge.qimg_fresh_capture_request.v1"
REPORT_SCHEMA = "neural_forge.qimg_fresh_capture_stage.v1"
FREEZE_SCHEMA = "neural_forge.qimg_fresh_capture_freeze.v1"
GENERATION_SCHEMA = "neural_forge.qimg_fresh_capture_generation.v1"
START_SCHEMA = "neural_forge.qimg_fresh_capture_start.v1"
AUTH_SCHEMA = "neural_forge.qimg_capture_runtime_authorization.v1"
ADAPTER_NAMES = {"c16_qimg_broad_capture.py", "c16_qimg_fresh_metadata.py",
                 "c16_qimg_fresh_preregistration.py", "c16_qimg_fresh_capture.py"}
AUTHORITY_SCOPE = dict(meta.COVERAGE)
AUTHORITY_LIMIT = "caller must independently establish authority origin/currentness; adapter verifies byte commitments only"
NATIVE_SOURCE_PINS = {'comfy/k_diffusion/sampling.py': 'cc5f944efd85c566484c3999beb74e8c19c894f2a50ca574d090c3a46ac6bd06', 'comfy/latent_formats.py': 'f13259cbd815a339a474976522eb6aae10fb16c8397f13b41e50ddd107f8caf8', 'comfy/ldm/flux/layers.py': '35f2dfbadb8b59de79c306f365f26c2baa0a7d54836414737cd781fc24e6a2bc', 'comfy/ldm/flux/math.py': 'ee3473e262894884b12eb3af4caa22d069273eb87db3549fb46c42c37f910ece', 'comfy/ldm/lightricks/model.py': 'c9ae1a8505aa0eabefdffe409083141e5e8461dd9d55f452c13f0f7b0c6466d8', 'comfy/ldm/qwen_image/model.py': '14c805af8da13d31094c2c704e413cc282ef43f4afe17f77de9070bdca301f28', 'comfy/ldm/wan/vae.py': 'b18fdd6d95d9c7093fbc8f2c831dfb36322aa1951c2dd17d32733dac53d689e4', 'comfy/model_management.py': 'b7741d049c2c749c316e5769ef520846bc7dd444e67de08403a4ee13a9c034c2', 'comfy/model_sampling.py': '8559ac2f700b788babf30cc16f410c93890e1f71485ac8f6a299ea36e2cb8717', 'comfy/sample.py': 'ee3485747727f9a182b9e7dd5a32a81e9429c329f2dcd248996ea0eba191faea', 'comfy/samplers.py': 'b9f5e22d0d65b5e78cd4dcc763e702369869dc92cf4371dc52bbecaf0733fe49', 'comfy/sd.py': '7efb10f22252446069a0b88f0a8bc2d94713696d9f946d7757a0981ae8101b58', 'comfy/sd1_clip.py': 'cada45bf2cd1e2e20d24a46b959d86fc9f5529953456bb21471735f036e2dc70', 'comfy/text_encoders/llama.py': 'ad7d59be8f4732eee5b898591330adc875d69fcbd9987b9aa6c8473aeb423a11', 'comfy/text_encoders/qwen_image.py': '9eab4371988f5b5e55260aba00e3984996d8efd4851569cc7e54b0e39811c1a9'}


def load(contract_path, contract_sha, request_path, request_sha, *, code_root,
         evidence_root, anchor_path, anchor_sha):
    """Revalidate every input and source pin on every call; no cached authority."""
    contract = meta.strict_json(contract_path, contract_sha)
    meta.exact(contract, {"schema", "id", "recipe", "authority_scope", "authority_limit",
                          "accepted", "acceptance_enabled"}, "fresh capture contract")
    meta.require(contract["schema"] == CONTRACT_SCHEMA and contract["accepted"] is False
                 and contract["acceptance_enabled"] is False, "capture-only contract required")
    meta.token(contract["id"], "contract ID")
    meta.require(meta.same(contract["authority_scope"], AUTHORITY_SCOPE)
                 and contract["authority_limit"] == AUTHORITY_LIMIT,
                 "complete current authority scope and truthful limit required")
    recipe = meta.reference(evidence_root, contract["recipe"])
    history = meta.validate_recipe(recipe, code_root, evidence_root)
    meta.require(recipe["native_source_code_pins"] == NATIVE_SOURCE_PINS,
                 "complete reviewed native source-code union differs")
    pin_names = {Path(p["path"]).name for p in recipe["source_tool_pins"]}
    meta.require(ADAPTER_NAMES <= pin_names, "new recipe must pin all adapter/helper bytes before freeze")
    meta.require("__init__.py" in pin_names, "actual imported mediafold package initializer must be pinned")
    # All four modules must be uniquely pinned; an alias cannot bless another import.
    for name in ADAPTER_NAMES:
        matches = [p for p in recipe["source_tool_pins"] if Path(p["path"]).name == name]
        meta.require(len(matches) == 1, "ambiguous adapter/helper pin")
        if name in {Path(__file__).name, Path(meta.__file__).name}:
            meta.require(meta.contained(code_root, matches[0]["path"]).resolve()
                         == Path(__file__ if name == Path(__file__).name else meta.__file__).resolve(),
                         "imported helper is not the pinned code-root module")
    anchor = meta.strict_json(anchor_path, anchor_sha)
    meta.exact(anchor, {"schema", "authority_id", "revision", "allowed_raw_sha256", "allowed_qa_methods"}, "authority anchor")
    meta.require(anchor["schema"] == "chimera_omni.qimg_owner_authority_anchor.v1", "explicit owner-pinned authority anchor required")
    meta.token(anchor["authority_id"], "authority ID"); meta.token(anchor["revision"], "authority revision")
    meta.require(isinstance(anchor["allowed_raw_sha256"], dict), "authority raw pins required")
    for name, sha in anchor["allowed_raw_sha256"].items(): meta.token(name, "authority role"); meta.digest(sha)
    meta.require(isinstance(anchor["allowed_qa_methods"], list) and 1 <= len(anchor["allowed_qa_methods"]) <= 8
                 and all(isinstance(x, str) and x for x in anchor["allowed_qa_methods"]), "explicit owner QA methods required")
    req = meta.strict_json(request_path, request_sha)
    fields = {"schema", "contract_sha256", "recipe_sha256", "plans", "assignments", "family_registry",
              "qa_registry", "qa_result", "freeze_receipt", "generated_images", "generation_chronology",
              "capture_start_chronology"}
    meta.exact(req, fields, "staged request")
    meta.require(req["schema"] == REQUEST_SCHEMA and req["contract_sha256"] == contract_sha
                 and req["recipe_sha256"] == contract["recipe"]["sha256"], "contract/recipe/request identity differs")
    entries = meta.read_plans(evidence_root, req["plans"], history)
    assignment_sha = meta.validate_assignments(req["assignments"], entries)
    qa = meta.check_authorities(evidence_root, req, anchor, assignment_sha,
                               req["recipe_sha256"], recipe["historical_exclusions"]["sha256"])
    commitment = meta.canonical_sha({k: req[k] for k in ("contract_sha256", "recipe_sha256", "plans", "assignments")})
    freeze = meta.trusted(evidence_root, req["freeze_receipt"], "freeze_receipt", anchor)
    meta.exact(freeze, {"schema", "event_id", "recorded_at_utc", "contract_sha256", "recipe_sha256",
                        "preparation_commitment_sha256", "qa_result_sha256"}, "external freeze")
    meta.require(freeze["schema"] == FREEZE_SCHEMA and freeze["contract_sha256"] == contract_sha
                 and freeze["recipe_sha256"] == req["recipe_sha256"]
                 and freeze["preparation_commitment_sha256"] == commitment
                 and freeze["qa_result_sha256"] == req["qa_result"]["sha256"], "freeze commitments differ")
    meta.token(freeze["event_id"], "freeze event")
    meta.require(meta.utc(qa["checked_at_utc"]) <= meta.utc(freeze["recorded_at_utc"]), "QA must precede freeze")
    return {"contract": contract, "request": req, "recipe": recipe, "history": history, "anchor": anchor,
            "freeze": freeze, "commitment": commitment, "contract_sha256": contract_sha,
            "request_sha256": request_sha, "assignment_sha256": assignment_sha,
            "source_authority_anchor_sha256": anchor_sha,
            "evidence_root": Path(evidence_root)}


def report(ctx, stage):
    return {"schema": REPORT_SCHEMA, "stage": stage, "contract_sha256": ctx["contract_sha256"],
            "request_sha256": ctx["request_sha256"], "recipe_sha256": ctx["request"]["recipe_sha256"],
            "stage_prerequisites_satisfied": True,
            "authority_byte_commitments_verified": True,
            "external_authority_provenance_independently_verified": False,
            "actual_execution_blocked_without_pinned_coordinator_authorization": True,
            "accepted": False, "acceptance_enabled": False,
            "scope": "distinct staged capture-only contract; old proposal readiness flags confer no authority"}


def pregeneration(**kwargs):
    ctx = load(**kwargs)
    req = ctx["request"]
    meta.require(all(req[k] is None for k in ("generated_images", "generation_chronology", "capture_start_chronology")),
                 "pregeneration-only request must not assert later generation or capture evidence")
    return ctx, report(ctx, "pregeneration")


def _generation(ctx):
    req, root, anchor = ctx["request"], ctx["evidence_root"], ctx["anchor"]
    images = req["generated_images"]
    meta.require(isinstance(images, list) and len(images) == 32, "all 32 actual generated image identities required")
    seen, receipts = set(), {}
    for actual, planned in zip(images, req["assignments"]):
        meta.exact(actual, {"id", "image_sha256", "parent_receipt_sha256"}, "generated identity")
        meta.require(actual["id"] == planned["id"], "generated image order changed")
        sha = meta.digest(actual["image_sha256"]); meta.digest(actual["parent_receipt_sha256"])
        meta.require(sha not in seen and sha not in ctx["history"]["image_sha256"], "duplicate or historical generated image")
        seen.add(sha)
        batch = planned["batch"]
        meta.require(batch not in receipts or receipts[batch] == actual["parent_receipt_sha256"], "one complete parent receipt per batch required")
        receipts[batch] = actual["parent_receipt_sha256"]
    meta.require(len(receipts) == 4 and len(set(receipts.values())) == 4, "four distinct parent receipts required")
    event = meta.trusted(root, req["generation_chronology"], "generation_chronology", anchor)
    fields = {"schema", "event_id", "started_at_utc", "completed_at_utc", "contract_sha256", "recipe_sha256",
              "preparation_commitment_sha256", "freeze_receipt_sha256", "generated_images_sha256", "source_headers", "postgeneration_qa"}
    meta.exact(event, fields, "external generation")
    meta.require(event["schema"] == GENERATION_SCHEMA and event["contract_sha256"] == ctx["contract_sha256"]
                 and event["recipe_sha256"] == req["recipe_sha256"] and event["preparation_commitment_sha256"] == ctx["commitment"]
                 and event["freeze_receipt_sha256"] == req["freeze_receipt"]["sha256"]
                 and event["generated_images_sha256"] == meta.canonical_sha(images), "generation commitments differ")
    meta.token(event["event_id"], "generation event")
    meta.require(meta.utc(ctx["freeze"]["recorded_at_utc"]) < meta.utc(event["started_at_utc"])
                 <= meta.utc(event["completed_at_utc"]), "freeze must strictly precede generation")
    meta.exact(event["source_headers"], {"diffusion", "vae", "text_encoder"}, "teacher headers")
    for role, ident in event["source_headers"].items():
        meta.exact(ident, {"header_sha256", "file_bytes", "header_bytes", "full_file_sha256", "scope"}, "teacher header")
        meta.digest(ident["header_sha256"])
        meta.require(type(ident["file_bytes"]) is int and type(ident["header_bytes"]) is int
                     and 2 <= ident["header_bytes"] < ident["file_bytes"] and ident["full_file_sha256"] is None
                     and ident["scope"] == "header only; tensor bodies not read", "truthful bounded teacher header required")
    meta.require(event["source_headers"]["diffusion"]["header_sha256"] == meta.DIFFUSION_HEADER, "wrong diffusion source")
    post = meta.trusted(root, event["postgeneration_qa"], "postgeneration_qa", anchor)
    meta.exact(post, {"schema", "recipe_sha256", "generated_images_sha256", "qa_registry_sha256", "checked_at_utc",
                      "coverage", "comparison_method", "collisions"}, "postgeneration QA")
    meta.require(post["schema"] == "chimera_omni.qimg_postgeneration_hash_only_qa_response.v1"
                 and post["recipe_sha256"] == req["recipe_sha256"] and post["generated_images_sha256"] == meta.canonical_sha(images)
                 and post["qa_registry_sha256"] == req["qa_registry"]["sha256"] and meta.same(post["coverage"], AUTHORITY_SCOPE)
                 and post["comparison_method"] in anchor["allowed_qa_methods"] and isinstance(post["collisions"], dict)
                 and set(post["collisions"]) == set(AUTHORITY_SCOPE)
                 and all(type(v) is int and v == 0 for v in post["collisions"].values()), "postgeneration QA scope/identity/method/collision differs")
    meta.require(meta.utc(event["completed_at_utc"]) <= meta.utc(post["checked_at_utc"]), "QA must follow generation completion")
    ctx.update(generation=event, post=post)


def precapture(**kwargs):
    ctx = load(**kwargs)
    meta.require(ctx["request"]["capture_start_chronology"] is None,
                 "precapture request must not fabricate or reuse an actual capture-start event")
    _generation(ctx)
    return ctx, report(ctx, "precapture")


def postcapture(*, run_receipt_path, run_receipt_sha, **kwargs):
    """External start-byte link only; runner admit() separately re-audits tensors.

    This metadata helper alone does not establish a complete native capture.
    """
    ctx = load(**kwargs); _generation(ctx)
    req, anchor, root = ctx["request"], ctx["anchor"], ctx["evidence_root"]
    start = meta.trusted(root, req["capture_start_chronology"], "capture_start_chronology", anchor)
    meta.exact(start, {"schema", "event_id", "started_at_utc", "contract_sha256", "recipe_sha256",
                       "generated_images_sha256", "generation_chronology_sha256", "postgeneration_qa_sha256", "run_receipt_sha256"}, "actual capture start")
    meta.require(start["schema"] == START_SCHEMA and start["contract_sha256"] == ctx["contract_sha256"]
                 and start["recipe_sha256"] == req["recipe_sha256"]
                 and start["generated_images_sha256"] == meta.canonical_sha(req["generated_images"])
                 and start["generation_chronology_sha256"] == req["generation_chronology"]["sha256"]
                 and start["postgeneration_qa_sha256"] == ctx["generation"]["postgeneration_qa"]["sha256"]
                 and start["run_receipt_sha256"] == run_receipt_sha, "capture-start commitments differ")
    meta.token(start["event_id"], "capture event")
    run = meta.strict_json(run_receipt_path, run_receipt_sha)
    meta.require(run.get("schema") == "neural_forge.qimg_fresh_capture_run.v1" and run.get("status") == "complete"
                 and run.get("contract_sha256") == ctx["contract_sha256"] and run.get("recipe_sha256") == req["recipe_sha256"]
                 and run.get("accepted") is False and run.get("acceptance_enabled") is False, "fresh capture-only run receipt required")
    meta.require(meta.utc(ctx["post"]["checked_at_utc"]) < meta.utc(start["started_at_utc"])
                 == meta.utc(run["started_at_utc"]) <= meta.utc(run["completed_at_utc"]), "actual capture chronology differs or precedes QA")
    result = report(ctx, "postcapture")
    result["external_capture_start_byte_commitments_verified"] = True
    return ctx, result


def runtime_authorization(ctx, path, sha, *, bundle_sha, runtime_anchor_path,
                          runtime_anchor_sha, now=None):
    """Strict input dependency. This does not inspect GPU/provider/container state."""
    # This new independently pinned anchor is created AFTER sealing. It must not
    # mutate the source/freeze anchor inside the bundle or create a hash cycle.
    anchor = meta.strict_json(runtime_anchor_path, runtime_anchor_sha)
    meta.exact(anchor, {"schema", "authority_id", "revision", "source_authority_anchor_sha256",
                        "contract_sha256", "allowed_runtime_authorization_sha256"}, "runtime authority anchor")
    meta.require(anchor["schema"] == "neural_forge.qimg_capture_runtime_authority_anchor.v1"
                 and anchor["source_authority_anchor_sha256"] == ctx["source_authority_anchor_sha256"]
                 and anchor["contract_sha256"] == ctx["contract_sha256"], "runtime anchor does not bind frozen source authority/contract")
    meta.token(anchor["authority_id"], "runtime authority ID"); meta.token(anchor["revision"], "runtime authority revision")
    meta.require(anchor["allowed_runtime_authorization_sha256"] == meta.digest(sha),
                 "independently pinned coordinator runtime authorization required")
    body = meta.strict_json(path, sha)
    meta.exact(body, {"schema", "event_id", "action", "contract_sha256", "bundle_sha256", "approved_at_utc",
                      "not_after_utc", "runtime_image_id", "physical_gpu_uuid", "scope"}, "runtime authorization")
    meta.require(body["schema"] == AUTH_SCHEMA and body["action"] == "encode_and_capture_only"
                 and body["contract_sha256"] == ctx["contract_sha256"] and body["bundle_sha256"] == bundle_sha
                 and body["scope"] == "external coordinator manages and verifies runtime/resources; no fold acceptance",
                 "runtime authorization scope/identity differs")
    meta.token(body["event_id"], "runtime event")
    meta.require(isinstance(body["runtime_image_id"], str) and body["runtime_image_id"].startswith("sha256:")
                 and len(body["runtime_image_id"]) == 71, "pinned image identity required")
    meta.digest(body["runtime_image_id"][7:])
    meta.require(isinstance(body["physical_gpu_uuid"], str) and body["physical_gpu_uuid"].startswith("GPU-"), "physical GPU UUID required")
    meta.token(body["physical_gpu_uuid"], "GPU UUID")
    current = now or datetime.now(timezone.utc)
    approved, expiry = meta.utc(body["approved_at_utc"]), meta.utc(body["not_after_utc"])
    meta.require(approved <= current < expiry and (expiry - approved).total_seconds() <= 3600,
                 "authorization not current or exceeds bounded one-hour window")
    return body
