"""Pinned D4 preparation and replay/ownership/merge-plan validation, no training.

All validators operate on supplied immutable evidence. A valid plan is not a
parity/cache seal, a registered child, a production driver or a weight merge.
This command neither imports Torch nor touches model or optimizer weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import tempfile

from .corpus import canonical, digest, file_sha256, iter_jsonl

FROZEN_SHA = "c24d21e08815d7acd2ce14aea989d16f84f167eb24a40597966be19284cb1370"
FROZEN_BYTES = 15339
QUOTAS = {"short_text": 2500000, "long_text": 500000, "k3": 1000000,
          "vision": 300, "ace": 300, "qimg": 300, "hy3d": 300, "h3": 300}
OPTIMIZER_FAMILIES = {"D2.lora", "D2.k3_lora", "D2.gain", "D3.unit", "D3.emission", "D3.bridge", "D3.music_embed", "D3.music_head", "D3.shared_lora"}
STAGE_ORDER = ("D2", "D3a", "D3b.ace", "D3b.qimg", "D3b.hy3d", "D3b.h3", "D3c", "D3d")
REQUIRED_STAGE_ROLES = set(STAGE_ORDER)


class Error(ValueError):
    pass


def sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def pin(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def pinned(ref, root):
    if not sha256(ref.get("sha256")):
        raise Error("pinned_source_hash_missing")
    path = Path(ref["path"])
    path = path if path.is_absolute() else root / path
    if file_sha256(path) != ref["sha256"]:
        raise Error("pinned_source_bytes_changed")
    return path


def preregistration(path):
    # Reproduce the campaign receipt algorithm exactly, including its first
    # literal Scope match in the introduction and Amendments heading's LF.
    # File bytes remain separately pinned; only the frozen-body rule normalizes LF.
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    start, end = raw.find(b"## 1. Scope"), raw.find(b"\n## Amendments")
    if start < 0 or end <= start:
        raise Error("frozen_preregistration_markers_missing")
    body = raw[start:end + 1]
    import hashlib
    if len(body) != FROZEN_BYTES or hashlib.sha256(body).hexdigest() != FROZEN_SHA:
        raise Error("frozen_D1_D4_preregistration_body_changed")
    return {**pin(path), "frozen_body_sha256": FROZEN_SHA, "frozen_body_bytes": FROZEN_BYTES,
            "nominal_quotas": QUOTAS, "lr_multiplier": "0.3", "text_vocab": [0, 248320],
            "music_allocated_rows": [252416, 317952], "music_written_rows": [252416, 316416]}


def select_replay(lane, rows, held_groups=(), *, whole_row_overshoot_amended=False):
    """Deterministic, unique, admitted whole-row selection, never padding/repetition."""
    if lane not in QUOTAS:
        raise Error("unknown_D4_replay_lane")
    ids, pairs, selected, amount = set(), set(), [], 0
    held_groups = set(held_groups)
    quota, unit = QUOTAS[lane], "real_tokens" if lane in {"short_text", "long_text", "k3"} else "pairs"
    for row in rows:
        if amount >= quota:
            break
        identifier = row.get("id")
        group = row.get("recovery_group_id")
        if not identifier or identifier in ids or not group:
            raise Error("missing_or_duplicate_replay_identity")
        ids.add(identifier)
        if row.get("split") != "train" or row.get("held") is not False or group in held_groups:
            raise Error("heldout_or_unknown_split_in_replay")
        if row.get("lane") != lane or row.get("sealed_target") is not True:
            raise Error("replay_requires_matching_existing_sealed_targets")
        # The descriptor is supplied only after authentic native cache traversal;
        # a metadata label alone never certifies the underlying tensor contents.
        for field in ("corpus_sha256", "teacher_manifest_sha256", "target_shard_sha256"):
            value = row.get(field)
            if not sha256(value):
                raise Error("replay_source_pin_missing")
        count = row.get("real_tokens") if unit == "real_tokens" else 1
        if type(count) is not int or count < 1:
            raise Error("real_token_count_missing")
        if unit == "pairs" and row.get("pair_id") in pairs:
            raise Error("duplicate_native_pair_does_not_fill_replay_quota")
        if unit == "pairs" and not row.get("pair_id"):
            raise Error("native_pair_identity_missing")
        if unit == "pairs":
            identity = {key: row.get(key) for key in ("input_sha256", "target_sha256", "conditioning_sha256")}
            if any(not sha256(value) for value in identity.values()) or row["pair_id"] != digest(identity):
                raise Error("native_pair_identity_must_bind_actual_input_target_conditioning")
            if identity["input_sha256"] == identity["target_sha256"]:
                raise Error("target_cannot_be_the_replay_input")
            pairs.add(row["pair_id"])
        selected.append(row)
        amount += count
    deficit = max(0, quota - amount)
    excess = max(0, amount - quota)
    blockers = []
    if deficit:
        blockers.append("insufficient_distinct_admitted_existing_targets")
    if excess and not whole_row_overshoot_amended:
        blockers.append("whole_row_token_budget_overshoot_requires_dated_amendment")
    return {"lane": lane, "quota": quota, "unit": unit, "selected_amount": amount,
            "deficit": deficit, "overshoot": excess, "selected_ids": [row["id"] for row in selected],
            "groups": sorted({row["recovery_group_id"] for row in selected}), "blockers": blockers,
            "production_ready": False, "scope": "selection logic only; source tensors/parity checked by native preflight"}


def optimizer_union(groups):
    """Require one optimizer owner per actual parameter, all prior rates scaled .3."""
    owners, families = {}, set()
    for group in groups:
        family = group.get("family")
        if family not in OPTIMIZER_FAMILIES:
            raise Error("unknown_or_missing_D4_optimizer_family")
        families.add(family)
        try:
            previous, current = Decimal(str(group["previous_base_lr"])), Decimal(str(group["lr"]))
        except (InvalidOperation, KeyError):
            raise Error("D4_optimizer_rate_missing_or_invalid") from None
        if not previous.is_finite() or previous <= 0 or not current.is_finite() or current != previous * Decimal("0.3"):
            raise Error("D4_lr_must_equal_pinned_previous_base_lr_times_point3")
        if not group.get("parameters"):
            raise Error("D4_optimizer_family_has_no_actual_parameters")
        for parameter in group["parameters"]:
            identifier = parameter.get("parameter_id")
            if not identifier or identifier in owners:
                raise Error("physical_parameter_has_duplicate_optimizer_owners")
            if type(parameter.get("numel")) is not int or parameter["numel"] < 1:
                raise Error("actual_parameter_census_missing")
            owners[identifier] = (family, parameter["numel"])
    if families != OPTIMIZER_FAMILIES:
        raise Error("D4_missing_D2_or_D3_trainable_family")
    return {"families": sorted(families), "parameters": len(owners), "numel": sum(value[1] for value in owners.values()),
            "ownership_sha256": digest(owners), "scope": "declared actual census; runtime must verify tensor identity/gradient ownership"}


def sequential_states(states, *, initial_shared_bank_sha256):
    """Reject sibling full adapter replacements presented as a sequential chain."""
    if not sha256(initial_shared_bank_sha256):
        raise Error("initial_shared_r16_identity_required")
    if [state.get("stage_role") for state in states] != list(STAGE_ORDER):
        raise Error("D4_prior_stages_require_preregistered_order_once_each")
    previous, shared_bank, roles, seen = None, initial_shared_bank_sha256, set(), set()
    for state in states:
        role = state.get("stage_role")
        state_sha = state.get("state_sha256")
        if role not in REQUIRED_STAGE_ROLES or role in roles or not sha256(state_sha) or state_sha in seen:
            raise Error("stage_state_role_or_identity_invalid")
        if previous is None and role != "D2":
            raise Error("D4_prior_stage_lineage_must_begin_with_D2")
        if previous is not None and state.get("parent_state_sha256") != previous:
            raise Error("stage_states_are_not_one_verified_sequential_lineage")
        if state.get("semantics") not in {"full_replacement", "additive_native_delta"}:
            raise Error("stage_state_semantics_missing")
        if state.get("shared_lora_semantics") == "add":
            raise Error("complete_shared_LoRA_banks_cannot_be_summed")
        if role.startswith("D3"):
            expected = "unchanged" if role == "D3a" else "full_replacement"
            if (state.get("shared_lora_semantics") != expected or not sha256(state.get("shared_lora_state_sha256"))
                    or state.get("shared_lora_parent_sha256") != shared_bank):
                raise Error("shared_r16_continuation_not_proven")
            if expected == "unchanged" and state["shared_lora_state_sha256"] != shared_bank:
                raise Error("unit_only_refit_cannot_replace_shared_r16_bank")
            shared_bank = state["shared_lora_state_sha256"]
        elif state.get("shared_lora_state_sha256") is not None:
            raise Error("D2_r64_bank_cannot_be_claimed_as_D3_shared_r16")
        previous = state_sha
        roles.add(role)
        seen.add(state_sha)
    if roles != REQUIRED_STAGE_ROLES:
        raise Error("D4_missing_prior_stage_state_roles")
    return {"terminal_state_sha256": previous, "shared_r16_state_sha256": shared_bank,
            "lineage_sha256": digest(states), "stages": len(states), "scope": "declared chain; actual state import/compose and parity still required"}


def merge_plan(operations, tensor_shapes, aliases=None, *, protected_row_targets=(), owner_regions=None, require_complete=False):
    """LoRA contributions first, K3 gain exactly once on down columns, one cast."""
    aliases, touched, adapters, gains = aliases or {}, {}, set(), set()
    protected_row_targets = {aliases.get(name, name) for name in protected_row_targets}
    if not operations:
        raise Error("empty_merge_is_not_a_D4_merge")
    for op in operations:
        target = aliases.get(op.get("target"), op.get("target"))
        if target not in tensor_shapes:
            raise Error("merge_target_not_in_actual_native_tensor_inventory")
        shape = tensor_shapes[target]
        kind, axis, start, stop = op.get("kind"), op.get("axis"), op.get("start"), op.get("stop")
        if (not isinstance(shape, list) or not shape or any(type(n) is not int or n < 1 for n in shape)
                or type(axis) is not int or not 0 <= axis < len(shape)
                or type(start) is not int or type(stop) is not int or not 0 <= start < stop <= shape[axis]):
            raise Error("merge_region_outside_actual_tensor")
        if op.get("compute_dtype") != "float32" or op.get("cast") != "defer_until_target_complete":
            raise Error("D4_requires_fp32_accumulation_and_one_final_cast")
        bank = op.get("bank")
        if bank not in OPTIMIZER_FAMILIES:
            raise Error("merge_operation_requires_known_trainable_owner")
        if owner_regions is not None and not any(
                aliases.get(region.get("target"), region.get("target")) == target and region.get("axis") == axis
                and region.get("start", stop) <= start < stop <= region.get("stop", start)
                for region in owner_regions.get(bank, [])):
            raise Error("merge_write_outside_pinned_owner_region")
        if target in protected_row_targets and bank not in {"D3.music_embed", "D3.music_head"}:
            raise Error("protected_vocab_tensor_requires_music_row_owner")
        if bank in {"D3.music_embed", "D3.music_head"} and (axis != 0 or start != 252416 or stop != 316416):
            raise Error("music_merge_must_preserve_text_ledger_and_reserved_rows")
        if kind == "lora":
            if len(shape) != 2:
                raise Error("LoRA_target_must_be_an_actual_matrix")
            if bank not in {"D2.lora", "D2.k3_lora", "D3.shared_lora"}:
                raise Error("LoRA_merge_requires_adapter_owner")
            if bank == "D2.k3_lora" and (axis != 1 or start != 4352 or stop != 16640 or not target.endswith("ffn.down.weight")):
                raise Error("K3_LoRA_must_own_K3_down_columns")
            ident = op.get("adapter_state_id")
            if not ident or ident in adapters:
                raise Error("adapter_state_would_be_folded_more_than_once")
            if target in gains:
                raise Error("LoRA_added_after_gain_would_miss_gain_scaling")
            adapters.add(ident)
        elif kind == "k3_gain":
            if (bank != "D2.gain" or target in gains or axis != 1 or start != 4352 or stop != 16640
                    or len(shape) != 2 or not target.endswith("ffn.down.weight") or shape[1] != 16640
                    or not any(item.get("kind") == "lora" and item.get("bank") == "D2.k3_lora" for item in touched.get(target, []))):
                raise Error("K3_gain_must_fold_once_after_K3_LoRA_on_down_columns")
            gains.add(target)
        elif kind not in {"additive_native_delta", "replace_full_native_state"}:
            raise Error("unknown_merge_operation_semantics")
        elif bank in {"D2.lora", "D2.k3_lora", "D3.shared_lora", "D2.gain"}:
            raise Error("adapter_or_gain_cannot_use_native_replacement_semantics")
        if kind == "replace_full_native_state" and any(
                (item["axis"] != axis
                or max(item["start"], start) < min(item["stop"], stop)) for item in touched.get(target, [])):
            raise Error("full_state_replacement_would_overwrite_prior_merge_write")
        if kind != "lora" and target in gains and kind != "k3_gain":
            raise Error("native_weight_write_after_gain_is_not_final_merge")
        touched.setdefault(target, []).append(op)
    k3_targets = {target for target, ops in touched.items() if any(op.get("bank") == "D2.k3_lora" for op in ops)}
    if gains != k3_targets:
        raise Error("missing_or_extra_K3_gain_fold")
    if require_complete:
        if owner_regions is None or set(owner_regions) != OPTIMIZER_FAMILIES or any(not regions for regions in owner_regions.values()):
            raise Error("complete_D4_merge_requires_every_owned_family_and_region")
        declared = {(bank, aliases.get(region["target"], region["target"]), region["axis"], region["start"], region["stop"])
                    for bank, regions in owner_regions.items() for region in regions}
        applied = {(op["bank"], aliases.get(op["target"], op["target"]), op["axis"], op["start"], op["stop"]) for op in operations}
        if declared != applied:
            raise Error("complete_D4_merge_must_cover_exact_pinned_owner_regions")
    return {"targets": len(touched), "adapter_states": len(adapters), "gain_folds": len(gains),
            "merge_plan_sha256": digest(operations), "scope": "operation order/region contract; no numerical merge or production writes performed"}


def validate_job(path):
    """Executable pre-merge job check of pinned files; never a runtime/parity seal."""
    job = json.loads(path.read_text(encoding="utf-8"))
    if job.get("schema") != "chimera_recovery.d4_validation_plan.v1":
        raise Error("D4_validation_plan_schema_required")
    required = {"preregistration", "replay_catalog", "optimizer_groups", "stage_states", "merge_operations", "tensor_inventory", "owner_regions", "initial_shared_bank"}
    if set(job.get("sources", {})) != required:
        raise Error("complete_pinned_D4_validation_inputs_required")
    sources = {name: pinned(ref, path.parent) for name, ref in job["sources"].items()}
    prereg = preregistration(sources["preregistration"])
    load = lambda name: json.loads(sources[name].read_text(encoding="utf-8"))
    states = load("stage_states")
    for state in states:
        ref = state.get("state_file", {})
        state_file = pinned(ref, sources["stage_states"].parent)
        if file_sha256(state_file) != state.get("state_sha256"):
            raise Error("stage_state_identity_does_not_bind_actual_file")
        if state.get("shared_lora_semantics") == "full_replacement":
            shared = pinned(state.get("shared_lora_file", {}), sources["stage_states"].parent)
            if file_sha256(shared) != state.get("shared_lora_state_sha256"):
                raise Error("shared_bank_identity_does_not_bind_actual_file")
    rows = list(iter_jsonl(sources["replay_catalog"]))
    if any(row.get("lane") not in QUOTAS for row in rows):
        raise Error("unknown_D4_replay_lane")
    # A heldout in even an unused suffix is refused. Do not let a quota stop
    # hide an invalid split or duplicate identity elsewhere in the catalog.
    if len({row.get("id") for row in rows}) != len(rows):
        raise Error("duplicate_replay_catalog_identity")
    if any(row.get("split") != "train" or row.get("held") is not False or row.get("recovery_group_id") in job.get("heldout_groups", []) for row in rows):
        raise Error("heldout_or_unknown_split_in_replay_catalog")
    replay = {lane: select_replay(lane, [row for row in rows if row["lane"] == lane], job.get("heldout_groups", []),
                              whole_row_overshoot_amended=False) for lane in QUOTAS}
    tensor_inventory = load("tensor_inventory")
    result = {"schema": "chimera_recovery.d4_validation_result.v1", "job": pin(path), "sources": job["sources"],
              "preregistration": prereg, "replay": replay, "optimizer": optimizer_union(load("optimizer_groups")),
              "lineage": sequential_states(states, initial_shared_bank_sha256=job["sources"]["initial_shared_bank"]["sha256"]),
              "merge": merge_plan(load("merge_operations"), tensor_inventory["shapes"], tensor_inventory.get("aliases", {}),
                                  protected_row_targets=tensor_inventory["protected_vocab_targets"], owner_regions=load("owner_regions"), require_complete=True),
              "production_ready": False, "training_started": False, "weights_changed": False,
              "source_code_sha256": file_sha256(Path(__file__)),
              "limitations": ["declared census/regions still require actual native runtime tensor identity and gradient proof",
                  "catalog file pins are not a teacher cache full-integrity traversal or numerical merge parity",
                  "whole-row overshoot blocks; this validator accepts no implicit budget amendment"]}
    result["seal_sha256"] = digest(result)
    return result


def prepare(prereg, inventory, text_jobs, admission, sources, output, *, native_manifests=(), progress=None, modality_jobs=None, supersedes=None):
    if output.exists():
        raise Error("retain_prior_D4_preparation")
    pins = {"preregistration": preregistration(prereg), "inventory": pin(inventory), "text_jobs": pin(text_jobs), "admission": pin(admission),
            "stage_sources": {path.name: pin(path) for path in sources}}
    if supersedes is not None:
        pins["supersedes"] = pin(supersedes)
    if len(pins["stage_sources"]) != len(sources):
        raise Error("stage_source_basename_collision")
    old = json.loads(inventory.read_text(encoding="utf-8"))
    jobs = json.loads(text_jobs.read_text(encoding="utf-8"))
    media = json.loads(admission.read_text(encoding="utf-8"))
    body = {key: value for key, value in media.items() if key != "manifest_sha256"}
    if digest(body) != media.get("manifest_sha256"):
        raise Error("media_admission_seal_mismatch")
    admission_rows = pinned(media["outputs"]["admission"], admission.parent)
    admitted, admitted_groups = Counter(), {}
    for row in iter_jsonl(admission_rows):
        if row.get("train_admitted") is True and row.get("split") == "train":
            admitted[row["kind"]] += 1
            admitted_groups.setdefault(row["kind"], set()).add(row["recovery_group_id"])
    # These are source/admission inventories, never substituted for native seals.
    packed = [row for row in jobs.get("sources", []) if row.get("split") == "train"]
    short = sum(row.get("real_tokens", 0) for row in packed if row.get("shape", [0, 0])[1] < 32768 and "long" not in row.get("name", ""))
    long = sum(row.get("real_tokens", 0) for row in packed if "long" in row.get("name", ""))
    raw = old.get("manifests", {})
    coverage = {
        "short_text": {"quota": QUOTAS["short_text"], "prepared_native_input_real_tokens": short, "verified_replay_target_tokens": None},
        "long_text": {"quota": QUOTAS["long_text"], "prepared_recovery_long_input_real_tokens": long,
                      "recovery_long_max_packed_context": max((row["shape"][1] for row in packed if "long" in row.get("name", "")), default=0),
                      "original_32k_corpus_real_tokens": raw.get("long", {}).get("real_tokens"), "verified_replay_target_tokens": None},
        "k3": {"quota": QUOTAS["k3"], "prereg_Qwen_tokens_reported": 1991482, "prereg_aligned_pairs_reported": 1867899, "verified_replay_target_tokens": None},
        **{lane: {"quota": 300, "source_admitted_items": admitted.get(kind, 0), "verified_native_pairs": None}
           for lane, kind in (("vision", "qvl"), ("ace", "ace"), ("qimg", "qimg"))},
        "hy3d": {"quota": 300, "original_raw_items": raw.get("media/hy3d", {}).get("items"), "verified_native_pairs": None},
        "h3": {"quota": 300, "original_unrelated_audio_pairs_excluded": raw.get("media/h3", {}).get("items"), "verified_native_pairs": None},
    }
    pins["native_manifests"] = {}
    for lane, path in native_manifests:
        if lane not in {"ace", "qimg", "vision", "hy3d", "h3"} or lane in pins["native_manifests"]:
            raise Error("invalid_or_duplicate_native_manifest_lane")
        value = json.loads(path.read_text(encoding="utf-8"))
        items = value.get("batches", [])
        if (value.get("schema") != "chimera_recovery.native_batches.v1" or value.get("sealed") is not True or value.get("split") != "train"
                or value.get("complete_source_traversal") is not True or value.get("heldout_groups_excluded") is not True
                or value.get("corpus_sha256") != file_sha256(admission) or not items or len({item["id"] for item in items}) != len(items)):
            raise Error("native_manifest_requires_current_admission_complete_unique_train_batches")
        for item in items:
            if (not item.get("recovery_group_id") or not sha256(item.get("tensors", {}).get("sha256"))
                    or not sha256(item.get("teacher_receipt", {}).get("sha256"))
                    or not item.get("tensors", {}).get("path") or not item.get("teacher_receipt", {}).get("path")):
                raise Error("native_batch_file_group_pins_missing")
            routes = {"ace": {"music_generate"}, "qimg": {"image_generate", "image_edit"}, "vision": {"vision_grounding", "video_understand"},
                      "hy3d": {"volume_generate"}, "h3": {"joint_av"}}
            if item.get("route") not in routes[lane]:
                raise Error("native_manifest_route_does_not_match_D4_lane")
            kind = "qvl" if lane == "vision" else lane
            if item["recovery_group_id"] not in admitted_groups.get(kind, set()):
                raise Error("native_batch_group_not_current_train_admitted")
        pins["native_manifests"][lane] = pin(path)
        groups = {item["recovery_group_id"] for item in items}
        coverage[lane].update(native_sealed_manifest_batches=len(items), native_manifest_distinct_groups=len(groups),
                              distinct_group_quota_deficit=max(0, 300 - len(groups)),
                              tensor_bytes_verified_in_this_local_audit=False, batch_ids_sha256=digest([item["id"] for item in items]))
    if progress is not None:
        pins["live_cache_progress"] = pin(progress)
        observed = json.loads(progress.read_text(encoding="utf-8"))
        data = observed.get("data", observed)
        for item in data.get("current_cache_progress", []):
            if "/ace_lm_codes/" in item["cache"] or "/hy3d_trajectories/" in item["cache"]:
                name = "music_lm" if "/ace_lm_codes/" in item["cache"] else "hy3d"
                coverage.setdefault(name, {}).update(teacher_shards_observed=item["shards"], teacher_cache_sealed_observed=item["sealed"],
                                                     observed_at=data["captured_at"], partial_targets_count_toward_quota=False)
    if modality_jobs is not None:
        pins["operator_modality_jobs"] = pin(modality_jobs)
        value = json.loads(modality_jobs.read_text(encoding="utf-8"))
        for route, lane in (("joint_av", "h3"), ("volume_generate", "hy3d")):
            count = value["lanes"][route]["counts"]["train"]
            coverage[lane].update(operator_source_train_items=count, distinct_source_item_gap_vs_300=max(0, 300 - count),
                                  source_scope="operator source items only; this is not a native input/target pair count or normative pair deficit")
    blockers = ["actual accepted D1/D2/D3 registered state lineage and immutable pre-D4 clone not supplied",
                "sequential D3 shared-r16 state import/compose not implemented by current stage drivers",
                "joint D4 union optimizer/forward/backward/exact-resume driver and native parity not supplied",
                "actual replay row/pair catalog with sealed target files, source groups and holdout exclusions not supplied",
                "300 native vision pairs require independent aligned assistant-answer KD, not ViT features alone",
                "H3 original arbitrary video/audio pairing cannot fill synchronized replay; genuine native joint cache required",
                "native HY3D geometry/conditioning target cache and authentic converter proof required",
                "music active64000/reserved1536 row protection and shared bank continuation must be preserved",
                "fp32 final merge must use exact actual tensor census and K3 LoRA-then-gain once; no checkpoint state summation",
                "whole-row nominal replay budget overshoot must be recorded and amended before an affected step"]
    result = {"schema": "chimera_recovery.d4_preparation.v1", "stage": "D4", "pins": pins, "quotas": QUOTAS,
              "coverage": coverage, "production_ready": False, "training_started": False, "weights_changed": False,
              "joint_driver_implemented": False, "blockers": blockers,
              "validators": ["select_replay", "optimizer_union", "sequential_states", "merge_plan"],
              "ownership_requirements": sorted(OPTIMIZER_FAMILIES), "stage_roles_required": sorted(REQUIRED_STAGE_ROLES),
              "runtime_receipts_required": ["actual native tensor/gradient optimizer census", "same-base union forward and gradient parity",
                  "exact sampler/window optimizer/scaler resume", "teacher full-integrity native cache traversal", "actual no-regression evaluation and decoded media",
                  "one fp32 merge numerical parity and final registered child/clone identities"],
              "source_code_sha256": file_sha256(Path(__file__)),
              "text_input_totals": {"train_rows": sum(row["rows"] for row in packed),
                  "real_tokens": sum(row["real_tokens"] for row in packed), "assistant_tokens": sum(row["supervised_tokens"] for row in packed)},
              "limitations": ["counts are pinned reported source/input availability, not target-cache verification",
                  "source items/groups and sigma batches differ from native input/target pairs; only a pinned actual x/y/conditioning catalog proves the 300-pair budget",
                  "no false seal/clone/parity receipt is generated", "no model, GPU, registry or active queue is touched"]}
    result["seal_sha256"] = digest(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".", dir=output.parent) as temp:
        root = Path(temp)
        (root / "report.json").write_bytes(canonical(result) + b"\n")
        (root / "source_pins.json").write_bytes(canonical(pins) + b"\n")
        root.rename(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-job", type=Path, help="check pinned replay/lineage/optimizer/merge plan; no training")
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--text-jobs", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--stage-source", type=Path, action="append")
    parser.add_argument("--native-manifest", action="append", default=[], help="lane=actual native batches manifest")
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--modality-jobs", type=Path)
    parser.add_argument("--supersedes", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.validate_job:
        print(json.dumps(validate_job(args.validate_job)))
        return
    if any(value is None for value in (args.preregistration, args.inventory, args.text_jobs, args.admission, args.stage_source, args.out)):
        parser.error("preparation requires --preregistration --inventory --text-jobs --admission --stage-source --out")
    if any("=" not in value for value in args.native_manifest):
        parser.error("--native-manifest must be lane=path")
    native = [(value.split("=", 1)[0], Path(value.split("=", 1)[1])) for value in args.native_manifest]
    report = prepare(args.preregistration, args.inventory, args.text_jobs, args.admission, args.stage_source, args.out,
                     native_manifests=native, progress=args.progress, modality_jobs=args.modality_jobs, supersedes=args.supersedes)
    print(json.dumps({key: report[key] for key in ("production_ready", "coverage", "blockers", "seal_sha256")}))


if __name__ == "__main__":
    main()
