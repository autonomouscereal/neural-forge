#!/usr/bin/env python3
"""CPU metadata-only proposal gate. No model imports, payload reads or launcher.

The caller pins the recipe, submission and an independently supplied trust
anchor. An arbitrary local file calling itself authoritative is not a trust
anchor. This module verifies bytes/relations, not the provenance of that trust.
Even complete metadata cannot enable capture or acceptance: current production
broad capture is bound to Sep7 plans and the changed selector has no accepted
artifact writer. A reviewed adapter is a separate, required implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "chimera_omni.qimg_fresh_preparation.v1"
RECIPE_SCHEMA = "chimera_omni.qimg_fresh_recipe_draft.v1"
REPORT_SCHEMA = "chimera_omni.qimg_fresh_preparation_report.v1"
PARENT_SCHEMA = "neural_forge.qimg_parent_image_plan.v1"
SAMPLING = {"width": 512, "height": 512, "steps": 20, "cfg": 4,
            "sampler": "euler", "scheduler": "normal", "denoise": 1}
SIGMAS = [.15, .35, .65, .9]
DIFFUSION_HEADER = "bd90f4d6af8d235cb3b46ccf5f426abb351995e28e852b04fc678a541d87158e"
GATES = {"keep": 3071, "max_source_replay_rel_l2": .005,
         "max_heldout_rel_l2": .30, "min_heldout_cos": .95}
COVERAGE = {name: True for name in ("historical_recipes_and_captures",
    "source_families", "frozen_benchmarks", "hidden_gold", "media")}
BOUNDARY = {"proposed_metadata_adapter_only": True,
    "existing_broad_capture_accepts_new_plans": False,
    "reviewed_fresh_capture_adapter_available": False,
    "reviewed_changed_selector_acceptance_writer_available": False}
RULES = {"selector_recipe_id": "fit-centered-native-bf16-parent-output-norm-v1",
    "calibration_seed_rule": "sha256(qimg-broad-calibration-v1 NUL plan_sha256 NUL id NUL generation_seed); first8 big-endian masked int63",
    "selector_scope": "fit_valid_rows_only", "selected_channels": "verbatim_source_order",
    "solver": "existing_parent_anchored_affine_ridge_ladder",
    "native_replay_dtype": "torch.bfloat16", "solve_dtype": "torch.float64",
    "stored_update_relative_l2_max": 1, "bias_inclusive_mlp_reduction_min": 4,
    "gate_scope": "pooled_and_each_heldout_forward", "no_held_selection_refit_or_retry": True,
    "minimum_actual_fit_text_rows": 6142, "maximum_rows_per_stream": 131072,
    "expected_images": 32, "fit_images": 24, "held_images": 8, "expected_forwards": 128,
    "image_rows": {"fit": 98304, "holdout": 32768}}
MAX_JSON = 2 << 20


class Refused(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Refused(message)


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "lowercase SHA256 required")
    return value


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def raw_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(value, keys, label):
    require(isinstance(value, dict) and set(value) == set(keys), label + " fields differ")


def same(actual, expected):
    """Numeric equality may allow 4/4.0, but never bool masquerading as 1/0."""
    if type(expected) is bool:
        return type(actual) is bool and actual == expected
    if isinstance(expected, (int, float)):
        return type(actual) in {int, float} and actual == expected
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def token(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value), label + " safe identifier required")


def utc(value):
    require(isinstance(value, str) and value.endswith("Z"), "externally recorded UTC timestamp ending Z required")
    try:
        out = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Refused("invalid UTC timestamp") from exc
    require(out.tzinfo == timezone.utc, "UTC timestamp required")
    return out


def strict_json(path, expected_sha=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular metadata JSON file required")
    require(path.suffix == ".json" and path.stat().st_size <= MAX_JSON, "bounded .json metadata only")
    data = path.read_bytes()
    require(len(data) <= MAX_JSON, "metadata grew past bound")
    if expected_sha is not None:
        require(hashlib.sha256(data).hexdigest() == digest(expected_sha), "raw metadata SHA256 mismatch")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def finite(value):
        raise Refused("nonfinite JSON constant")
    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=finite)
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, Refused):
            raise
        raise Refused("invalid bounded metadata JSON") from exc
    # 1e999 is parsed as infinity without parse_constant being invoked.
    def walk(item, depth=0):
        require(depth <= 32, "metadata nesting exceeds bound")
        if isinstance(item, float):
            require(math.isfinite(item), "nonfinite numeric JSON value")
        if isinstance(item, dict):
            for child in item.values(): walk(child, depth + 1)
        elif isinstance(item, list):
            require(len(item) <= 4096, "metadata list exceeds bound")
            for child in item: walk(child, depth + 1)
    walk(value)
    return value


def contained(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "POSIX relative metadata path required")
    value = Path(relative)
    require(not value.is_absolute() and all(p not in {"", ".", ".."} for p in value.parts)
            and ":" not in relative, "path escape refused")
    base = Path(root).resolve()
    target = base / value
    require(not target.is_symlink() and target.resolve().is_relative_to(base), "metadata path escape/symlink refused")
    return target


def reference(root, ref):
    exact(ref, {"path", "sha256"}, "metadata reference")
    return strict_json(contained(root, ref["path"]), ref["sha256"])


def trusted(root, ref, role, anchor):
    require(anchor is not None, "independent owner-pinned authority anchor missing")
    exact(ref, {"path", "sha256"}, "authority metadata reference")
    require(role in anchor["allowed_raw_sha256"]
            and digest(ref["sha256"]) == anchor["allowed_raw_sha256"][role],
            "metadata authority/reference pin not authorized for " + role)
    return reference(root, ref)


def validate_recipe(body, code_root, evidence_root):
    exact(body, {"schema", "id", "state", "source_commit", "sampling", "sigmas", "block",
                 "streams", "rules", "diffusion_header_sha256", "source_tool_pins",
                 "native_source_code_pins", "historical_exclusions", "integration_boundary"}, "recipe")
    require(body["schema"] == RECIPE_SCHEMA and body["state"] == "recipe_only_frozen_draft", "recipe-only draft schema required")
    token(body["id"], "recipe ID")
    require(isinstance(body["source_commit"], str) and re.fullmatch(r"[0-9a-f]{40}", body["source_commit"]), "source commit required")
    require(same(body["sampling"], SAMPLING) and same(body["sigmas"], SIGMAS) and type(body["block"]) is int
            and body["block"] == 30, "exact reviewed sampling/grid/block required")
    require(same(body["streams"], {"img": GATES, "txt": GATES}) and same(body["rules"], RULES),
            "fixed FIT-only recipe/gates differ; no implicit acceptance change")
    require(body["diffusion_header_sha256"] == DIFFUSION_HEADER, "wrong QIMG parent header")
    require(same(body["integration_boundary"], BOUNDARY), "proposal cannot self-enable runner/acceptance integration")
    pins = body["source_tool_pins"]
    require(isinstance(pins, list) and 10 <= len(pins) <= 32, "bounded complete source-tool pin list required")
    names = set()
    for pin in pins:
        exact(pin, {"path", "bytes", "sha256"}, "tool pin")
        require(pin["path"] not in names and type(pin["bytes"]) is int and 0 < pin["bytes"] <= MAX_JSON,
                "unique bounded tool pin required")
        names.add(pin["path"])
        path = contained(code_root, pin["path"])
        require(path.is_file() and path.stat().st_size == pin["bytes"] and raw_sha(path) == digest(pin["sha256"]),
                "raw checkout tool identity changed: " + pin["path"])
    required = {"c16_qimg_parent_images.py", "c16_qimg_prepare_inputs.py", "c16_qimg_broad_capture.py",
                "c16_qimg_capture.py", "c16_qimg_preflight.py", "c16_qimg_fold.py",
                "c16_qimg_joint_replay.py", "c16_qimg_development_refold.py", "experimental_fit_selector.py", "linalg.py"}
    require(required <= {Path(name).name for name in names}, "missing actual capture/selector/solver source pins")
    native = body["native_source_code_pins"]
    require(isinstance(native, dict) and 10 <= len(native) <= 32, "native Comfy source pins required")
    for name, sha in native.items():
        require(name.startswith("comfy/") and ".." not in Path(name).parts, "native source path refused")
        digest(sha)
    history = reference(evidence_root, body["historical_exclusions"])
    exact(history, {"schema", "scope", "authoritative_complete", "recipe_sha256", "capture_sha256",
                    "image_sha256", "prompt_sha256", "image_ids", "generation_seeds", "evidence_pins"}, "known history")
    require(history["schema"] == "chimera_omni.qimg_known_history.v1"
            and history["authoritative_complete"] is False, "local history must remain bounded/non-authoritative")
    require(isinstance(history["scope"], str) and history["scope"].strip(), "known-history scope required")
    for kind in ("recipe_sha256", "capture_sha256", "image_sha256", "prompt_sha256"):
        require(isinstance(history[kind], list) and history[kind] and len(history[kind]) == len(set(history[kind])),
                "unique known history hashes required")
        for sha in history[kind]: digest(sha)
    require(len(history["image_sha256"]) >= 40, "both observed image experiments must be excluded")
    for name in ("image_ids", "generation_seeds"):
        require(isinstance(history[name], list) and history[name], "known-history IDs/seeds required")
    for name in history["image_ids"]: token(name, "known image ID")
    for seed in history["generation_seeds"]:
        require(type(seed) is int and 0 <= seed < 2**63, "known-history int63 seed required")
    for name in ("image_ids", "generation_seeds"):
        require(len(history[name]) == len(set(history[name])), "unique known-history IDs/seeds required")
    require(isinstance(history["evidence_pins"], list) and 1 <= len(history["evidence_pins"]) <= 32,
            "bounded known-history metadata evidence pins required")
    history_pins = set()
    for ref in history["evidence_pins"]:
        require(ref["sha256"] not in history_pins, "duplicate known-history evidence pin")
        history_pins.add(ref["sha256"])
        reference(evidence_root, ref)
    return history


def read_plans(root, refs, history):
    require(isinstance(refs, list) and len(refs) == 4, "four pinned new generation plans required")
    entries, seen = [], {name: set() for name in ("id", "prompt_sha256", "seed", "plan_sha256")}
    for number, ref in enumerate(refs, 1):
        exact(ref, {"path", "sha256"}, "generation plan reference")
        sha = digest(ref["sha256"])
        require(sha not in seen["plan_sha256"] and sha not in history["recipe_sha256"], "duplicate or historical generation plan")
        seen["plan_sha256"].add(sha)
        plan = reference(root, ref)
        exact(plan, {"schema", "scope", "sampling", "entries"}, "actual parent plan")
        require(plan["schema"] == PARENT_SCHEMA and same(plan["sampling"], SAMPLING) and isinstance(plan["scope"], str),
                "actual parent plan contract differs")
        require(isinstance(plan["entries"], list) and len(plan["entries"]) == 8
                and all(isinstance(e, dict) for e in plan["entries"])
                and all(isinstance(e.get("split"), str) and e["split"] in {"fit", "holdout"} for e in plan["entries"])
                and Counter(e.get("split") for e in plan["entries"]) == {"fit": 6, "holdout": 2}, "each batch requires six FIT/two held")
        for entry in plan["entries"]:
            exact(entry, {"id", "split", "seed", "prompt"}, "parent entry")
            token(entry["id"], "image ID")
            require(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", entry["id"]), "actual runner safe image ID required")
            require(type(entry["seed"]) is int and 0 <= entry["seed"] < 2**63, "int63 generation seed required")
            prompt = entry["prompt"]
            require(isinstance(prompt, str) and prompt.strip() and "<|" not in prompt
                    and len(prompt.encode()) <= 8192, "bounded plain parent prompt required")
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            values = {"id": entry["id"], "seed": entry["seed"], "prompt_sha256": prompt_sha}
            for key, value in values.items():
                old_key = {"id": "image_ids", "seed": "generation_seeds", "prompt_sha256": "prompt_sha256"}[key]
                require(value not in seen[key] and value not in history[old_key], "duplicate/historical " + key)
                seen[key].add(value)
            entries.append({"id": entry["id"], "split": entry["split"], "generation_seed": entry["seed"],
                            "prompt_sha256": prompt_sha, "batch": number})
    return entries


def validate_assignments(assignments, entries):
    require(isinstance(assignments, list) and len(assignments) == 32, "exact 32 image/family assignments required")
    fields = {"id", "split", "generation_seed", "prompt_sha256", "batch", "family_id", "source_uid"}
    families = {"fit": set(), "holdout": set()}
    source_uids = set()
    for actual, planned in zip(assignments, entries):
        exact(actual, fields, "assignment")
        require(same({key: actual[key] for key in planned}, planned), "assignment order/plan identity/split changed")
        token(actual["family_id"], "family ID"); token(actual["source_uid"], "source UID")
        require(actual["source_uid"] not in source_uids, "duplicate source UID")
        source_uids.add(actual["source_uid"])
        families[actual["split"]].add(actual["family_id"])
    require(not families["fit"] & families["holdout"], "source-family leakage across FIT/held")
    return canonical_sha(assignments)


def check_authorities(root, submission, anchor, assignment_sha, recipe_sha, history_sha):
    family = trusted(root, submission["family_registry"], "family_registry", anchor)
    exact(family, {"schema", "registry_id", "revision", "assignments_sha256", "assignments"}, "family registry")
    require(family["schema"] == "chimera_omni.qimg_family_registry_response.v1"
            and family["assignments_sha256"] == assignment_sha
            and same(family["assignments"], submission["assignments"]), "authoritative family mapping differs")
    token(family["registry_id"], "family registry ID"); token(family["revision"], "family registry revision")
    registry = trusted(root, submission["qa_registry"], "qa_registry", anchor)
    exact(registry, {"schema", "registry_id", "revision", "scope_sha256", "coverage"}, "QA registry")
    require(registry["schema"] == "chimera_omni.qimg_exclusion_registry_response.v1"
            and same(registry["coverage"], COVERAGE), "full frozen/hidden/gold/media/history/family exclusion coverage missing")
    token(registry["registry_id"], "QA registry ID"); token(registry["revision"], "QA registry revision"); digest(registry["scope_sha256"])
    qa = trusted(root, submission["qa_result"], "qa_result", anchor)
    exact(qa, {"schema", "recipe_sha256", "assignments_sha256", "known_history_sha256", "family_registry_sha256",
               "qa_registry_sha256", "checked_at_utc", "coverage", "comparison_method", "collisions"}, "hash-only QA result")
    require(qa["schema"] == "chimera_omni.qimg_hash_only_qa_response.v1"
            and qa["recipe_sha256"] == recipe_sha and qa["assignments_sha256"] == assignment_sha
            and qa["known_history_sha256"] == history_sha
            and qa["family_registry_sha256"] == submission["family_registry"]["sha256"]
            and qa["qa_registry_sha256"] == submission["qa_registry"]["sha256"]
            and same(qa["coverage"], COVERAGE) and qa["comparison_method"] in anchor["allowed_qa_methods"], "QA identity/coverage/method differs")
    require(isinstance(qa["collisions"], dict) and set(qa["collisions"]) == set(COVERAGE)
            and all(type(v) is int and v == 0 for v in qa["collisions"].values()), "exclusion collision or incomplete comparison")
    utc(qa["checked_at_utc"])
    return qa


def validate_preparation(recipe_path, recipe_sha, submission_path, submission_sha, *, code_root,
                         evidence_root, trust_anchor_path=None, trust_anchor_sha=None):
    recipe = strict_json(recipe_path, recipe_sha)
    history = validate_recipe(recipe, code_root, evidence_root)
    submission = strict_json(submission_path, submission_sha)
    keys = {"schema", "recipe_sha256", "plans", "assignments", "family_registry", "qa_registry", "qa_result",
            "freeze_receipt", "generated_images", "generation_chronology", "capture_start_chronology"}
    exact(submission, keys, "preparation")
    require(submission["schema"] == SCHEMA and submission["recipe_sha256"] == recipe_sha, "preparation recipe identity differs")
    anchor = None
    if trust_anchor_path is not None or trust_anchor_sha is not None:
        require(trust_anchor_path is not None and trust_anchor_sha is not None, "both out-of-band trust anchor path/hash required")
        anchor = strict_json(trust_anchor_path, trust_anchor_sha)
        exact(anchor, {"schema", "authority_id", "revision", "allowed_raw_sha256", "allowed_qa_methods"}, "authority anchor")
        require(anchor["schema"] == "chimera_omni.qimg_owner_authority_anchor.v1", "owner authority anchor schema required")
        token(anchor["authority_id"], "authority ID"); token(anchor["revision"], "authority revision")
        require(isinstance(anchor["allowed_raw_sha256"], dict), "explicit authority pins required")
        for role, sha in anchor["allowed_raw_sha256"].items(): token(role, "authority role"); digest(sha)
        require(isinstance(anchor["allowed_qa_methods"], list) and 1 <= len(anchor["allowed_qa_methods"]) <= 8
                and all(isinstance(v, str) and v for v in anchor["allowed_qa_methods"]), "owner-pinned QA methods required")
    blockers = []
    prepared_sha = canonical_sha({key: submission[key] for key in ("recipe_sha256", "plans", "assignments")})
    assignment_sha = None
    if submission["plans"] is None or submission["assignments"] is None:
        require(submission["plans"] is None and submission["assignments"] is None, "plan/assignment pair must both be pending")
        blockers.append("four new frozen generation plans and 24 FIT/8 held authoritative image-family assignments")
    else:
        entries = read_plans(evidence_root, submission["plans"], history)
        assignment_sha = validate_assignments(submission["assignments"], entries)
    authority_keys = ("family_registry", "qa_registry", "qa_result")
    qa = None
    if any(submission[key] is None for key in authority_keys):
        require(all(submission[key] is None for key in authority_keys), "family/QA evidence cannot be partially asserted")
        blockers.append("owner-pinned source-family registry and current frozen/hidden/gold/media/history hash-only QA authority/result")
    else:
        require(assignment_sha is not None, "authority evidence requires complete assignments")
        qa = check_authorities(evidence_root, submission, anchor, assignment_sha, recipe_sha, recipe["historical_exclusions"]["sha256"])
    freeze = None
    if submission["freeze_receipt"] is None:
        blockers.append("external immutable freeze event binding exact recipe and complete plan/assignment commitment before generation")
    else:
        require(assignment_sha is not None and qa is not None, "freeze cannot substitute for missing plan/QA authority")
        freeze = trusted(evidence_root, submission["freeze_receipt"], "freeze_receipt", anchor)
        exact(freeze, {"schema", "event_id", "recorded_at_utc", "recipe_sha256", "preparation_commitment_sha256",
                       "qa_result_sha256"}, "freeze event")
        require(freeze["schema"] == "chimera_omni.qimg_external_freeze_response.v1"
                and freeze["recipe_sha256"] == recipe_sha and freeze["preparation_commitment_sha256"] == prepared_sha
                and freeze["qa_result_sha256"] == submission["qa_result"]["sha256"], "freeze byte commitments differ")
        token(freeze["event_id"], "freeze event ID")
        require(utc(qa["checked_at_utc"]) <= utc(freeze["recorded_at_utc"]), "QA must precede immutable plan freeze")
    generated = submission["generated_images"]
    generation = submission["generation_chronology"]
    if generated is None or generation is None:
        require(generated is None and generation is None, "generation identities and external chronology must be paired")
        blockers.append("32 immutable generated image hashes and external generation chronology, with postgeneration hash-only QA")
    else:
        require(freeze is not None, "generation cannot establish prior registration without frozen plan evidence")
        require(isinstance(generated, list) and len(generated) == 32, "all 32 generated image hash identities required")
        image_shas, batch_receipts = set(), {}
        for actual, planned in zip(generated, submission["assignments"]):
            exact(actual, {"id", "image_sha256", "parent_receipt_sha256"}, "generated image identity")
            require(actual["id"] == planned["id"], "generated image order/identity differs")
            sha = digest(actual["image_sha256"]); digest(actual["parent_receipt_sha256"])
            require(sha not in image_shas and sha not in history["image_sha256"], "duplicate/historical generated image")
            image_shas.add(sha)
            batch = planned["batch"]
            require(batch not in batch_receipts or batch_receipts[batch] == actual["parent_receipt_sha256"],
                    "one complete parent-generation receipt per eight-image batch required")
            batch_receipts[batch] = actual["parent_receipt_sha256"]
        require(len(batch_receipts) == 4 and len(set(batch_receipts.values())) == 4,
                "four distinct complete parent-generation receipt identities required")
        generation = trusted(evidence_root, generation, "generation_chronology", anchor)
        exact(generation, {"schema", "event_id", "started_at_utc", "completed_at_utc", "recipe_sha256",
                           "preparation_commitment_sha256", "freeze_receipt_sha256", "generated_images_sha256",
                           "source_headers", "postgeneration_qa"}, "generation chronology")
        require(generation["schema"] == "chimera_omni.qimg_external_generation_response.v1"
                and generation["recipe_sha256"] == recipe_sha and generation["preparation_commitment_sha256"] == prepared_sha
                and generation["freeze_receipt_sha256"] == submission["freeze_receipt"]["sha256"]
                and generation["generated_images_sha256"] == canonical_sha(generated), "generation commitments differ")
        token(generation["event_id"], "generation event ID")
        require(utc(freeze["recorded_at_utc"]) < utc(generation["started_at_utc"]) <= utc(generation["completed_at_utc"]),
                "external freeze must strictly precede generation start")
        headers = generation["source_headers"]
        exact(headers, {"diffusion", "vae", "text_encoder"}, "generation source headers")
        for role, identity in headers.items():
            exact(identity, {"header_sha256", "file_bytes", "header_bytes", "full_file_sha256", "scope"}, "source header")
            digest(identity["header_sha256"])
            require(type(identity["file_bytes"]) is int and type(identity["header_bytes"]) is int
                    and 2 <= identity["header_bytes"] < identity["file_bytes"] and identity["full_file_sha256"] is None
                    and identity["scope"] == "header only; tensor bodies not read", "truthful exact header-only source evidence required")
        require(headers["diffusion"]["header_sha256"] == DIFFUSION_HEADER, "generation used wrong parent")
        post = trusted(evidence_root, generation["postgeneration_qa"], "postgeneration_qa", anchor)
        exact(post, {"schema", "recipe_sha256", "generated_images_sha256", "qa_registry_sha256", "checked_at_utc",
                     "coverage", "comparison_method", "collisions"}, "postgeneration hash-only QA")
        require(post["schema"] == "chimera_omni.qimg_postgeneration_hash_only_qa_response.v1"
                and post["recipe_sha256"] == recipe_sha and post["generated_images_sha256"] == canonical_sha(generated)
                and post["qa_registry_sha256"] == submission["qa_registry"]["sha256"] and same(post["coverage"], COVERAGE)
                and post["comparison_method"] in anchor["allowed_qa_methods"]
                and isinstance(post["collisions"], dict) and set(post["collisions"]) == set(COVERAGE)
                and all(type(v) is int and v == 0 for v in post["collisions"].values()), "postgeneration QA scope/method/collision mismatch")
        require(utc(generation["completed_at_utc"]) <= utc(post["checked_at_utc"]), "postgeneration QA must follow generation completion")
    capture = submission["capture_start_chronology"]
    if capture is None:
        blockers.append("externally pinned actual capture start after freeze, generation and postgeneration QA; required for future receipt admission")
    else:
        require(generated is not None, "capture chronology requires complete generated identities")
        capture = trusted(evidence_root, capture, "capture_start_chronology", anchor)
        exact(capture, {"schema", "event_id", "started_at_utc", "recipe_sha256", "generated_images_sha256",
                        "generation_chronology_sha256", "postgeneration_qa_sha256"}, "capture chronology")
        require(capture["schema"] == "chimera_omni.qimg_external_capture_start_response.v1"
                and capture["recipe_sha256"] == recipe_sha and capture["generated_images_sha256"] == canonical_sha(generated)
                and capture["generation_chronology_sha256"] == submission["generation_chronology"]["sha256"]
                and capture["postgeneration_qa_sha256"] == generation["postgeneration_qa"]["sha256"], "capture chronology commitments differ")
        token(capture["event_id"], "capture event ID")
        require(utc(post["checked_at_utc"]) < utc(capture["started_at_utc"]), "capture start must strictly follow postgeneration QA")
    metadata_complete = not blockers
    blockers += ["reviewed fresh-plan broad capture adapter (current wrapper hardcodes September 7 plans)",
                 "reviewed durable accepted-artifact writer/integration for changed FIT-only selector"]
    return {"schema": REPORT_SCHEMA, "status": "metadata_complete_integration_blocked" if metadata_complete else "recipe_draft_pending",
            "recipe_sha256": recipe_sha, "submission_sha256": submission_sha,
            "preparation_commitment_sha256": prepared_sha, "assignments_sha256": assignment_sha,
            "source_tool_pins_verified": len(recipe["source_tool_pins"]),
            "known_excluded_images": len(history["image_sha256"]), "known_history_complete": False,
            "metadata_prerequisites_satisfied": metadata_complete, "ready_for_generation": False,
            "ready_for_capture": False, "acceptance_enabled": False, "accepted": False,
            "model_tensor_payloads_read": False, "benchmark_gold_media_payloads_read": False,
            "external_authority_provenance_independently_verified": False,
            "blockers": blockers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "submission", "code-root", "evidence-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("recipe-sha256", "submission-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--trust-anchor", type=Path)
    parser.add_argument("--trust-anchor-sha256")
    args = parser.parse_args()
    try:
        report = validate_preparation(args.recipe, args.recipe_sha256, args.submission, args.submission_sha256,
            code_root=args.code_root, evidence_root=args.evidence_root,
            trust_anchor_path=args.trust_anchor, trust_anchor_sha=args.trust_anchor_sha256)
    except (Refused, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"schema": REPORT_SCHEMA, "status": "refused", "error": str(exc),
                          "ready_for_capture": False, "accepted": False}, sort_keys=True))
        raise SystemExit(2)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
