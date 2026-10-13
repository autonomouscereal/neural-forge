"""Prepare dated D3a/c/d contracts and materialize configs from actual seals.

Standard library only. This helper never imports a model, loads tensor targets,
launches training, allocates a runtime or changes an existing D3b contract.
Native driver preflight and decoded-content review remain separate executions.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

DATE = "2026-10-13"
STAGES = {
    "D3a": ("refit", "teacher_delta_host_write", "train_refit.py"),
    "D3c": ("music", "music_code_band_kl", "train_music.py"),
    "D3d": ("vision", "text_answer_kd_plus_vit_feature_mse", "train_vision.py"),
}
SOURCES = ("train_stages.py", "train_media.py", "training.py", "media_cache.py",
           "train_refit.py", "train_music.py", "music_cache.py", "train_vision.py", "vision_cache.py")
UNITS = ("ace_dit_L12", "h3_dit_L25_curve6", "qimg_mmdit_L30_img", "hy3d_dit_L10", "qvl32_vit_L13")
SAFETY = {"minutes": 30, "keep": 2, "max_consecutive_skips": 8, "window": 100, "max_skip_fraction": .2}
FROZEN_SHA = "c24d21e08815d7acd2ce14aea989d16f84f167eb24a40597966be19284cb1370"
FROZEN_BYTES = 15339
# Newly dated initial choices, adapted from adjacent D3b. These are neither
# historical approved D3a/c/d defaults nor experimentally measured optima.
ADAM = {"betas": [.9, .95], "eps": 1e-8, "weight_decay": 0., "clip": 1.}
OPTIMIZERS = {
    "D3a": {"lr": 2e-5, **ADAM},
    "D3c": {"band_lr": 2e-5, "lora_lr": 1e-4, **ADAM, "warmup_steps": 20, "min_lr_ratio": .1},
    "D3d": {"unit_lr": 2e-5, "bridge_lr": 2e-5, "lora_lr": 1e-4, **ADAM, "warmup_steps": 20, "min_lr_ratio": .1},
}
DEPENDENCIES = {
    "D3a": ["registered immutable exact D2 child and hybrid parity",
            "host rows recaptured on that child; independent sealed parent input/output and host-write targets",
            "selected unit source positions, row types and teacher provenance; full native batch seals"],
    "D3c": ["complete finite unchanged ACE-LM teacher source; partial captured traversal is insufficient",
            "authentic native music conditioning; shifted conditional top64 and tail band targets",
            "64000 written code rows within the allocated music band; no same-position target leakage"],
    "D3d": ["screened full unchanged QVL ViT-block13 teacher features; diagnostic ref32 is excluded",
            "actual native patch_flat input and independently captured assistant-only answer KD",
            "exact native/teacher vocabulary equality, answer positions and matching image/frame provenance"],
}


class Refusal(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_sha(value):
    if not isinstance(value, str) or re.fullmatch("[a-f0-9]{64}", value) is None:
        raise Refusal("actual lowercase SHA256 required")
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pin(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise Refusal("actual file required: " + str(path))
    return {"path": str(path), "sha256": sha(path)}


def check_pin(entry, root):
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        raise Refusal("file reference must contain only path and sha256")
    path = Path(entry["path"])
    path = path if path.is_absolute() else Path(root) / path
    if not path.is_file() or sha(path) != require_sha(entry["sha256"]):
        raise Refusal("pinned file missing or changed: " + str(path))
    return path.resolve()


def literal(path, name):
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise Refusal("current engine constant is absent: " + name)


def engine_inventory(root):
    root = Path(root).resolve()
    if (literal(root / "train_stages.py", "STAGES") != STAGES or
            literal(root / "train_stages.py", "SAFETY") != SAFETY or
            literal(root / "train_media.py", "FROZEN_BODY_SHA256") != FROZEN_SHA or
            literal(root / "train_media.py", "FROZEN_BODY_BYTES") != FROZEN_BYTES):
        raise Refusal("current native stage dispatch/frozen contract changed; review before preparing")
    return {name: pin(root / name) for name in SOURCES}


def preregistration_parts(raw):
    normalized = raw.replace(b"\r\n", b"\n")
    start, end = normalized.find(b"## 1. Scope"), normalized.find(b"\n## Amendments")
    if start < 0 or end <= start:
        raise Refusal("frozen preregistration markers absent")
    body, tail = normalized[start:end + 1], normalized[end:]
    if len(body) != FROZEN_BYTES or hashlib.sha256(body).hexdigest() != FROZEN_SHA:
        raise Refusal("frozen preregistration body differs")
    if b"guided_velocity" not in tail or DATE.encode() not in tail:
        raise Refusal("preserve existing dated D3b amendment before appending stage contracts")
    return normalized, body, tail


def amendment(stage):
    common = (f"\n\n### {DATE}: independently dated {stage} native recovery contract\n\n"
              "These explicit first-pass settings are new recovery choices adapted from the adjacent "
              "D3b contract, not approved historical defaults or measured optima. No existing D3b "
              "preregistration, cache seal or optimizer identity is rewritten.\n\n")
    objectives = {
        "D3a": "D3a objective `teacher_delta_host_write` retains relative-MSE chart-in and block refit and adds "
                "relative MSE of `(parent_out - parent_in) @ chart_out.weight.T` against independently "
                "sealed host-write targets on unit-owned rows. Teacher tensors are detached. Host rows "
                "must be recaptured on the exact registered D2 child; copying current predictions as "
                "targets is forbidden. Explicit term weights are chart_in=1, block=1, chart_out=1. "
                "Overall unit weight is 0.1 for QVL type6 and 1 for other units. Train all parameters "
                "of the selected unit only; other units and all LoRA are frozen. AdamW LR is constant; "
                "this engine has no warmup or cosine schedule fields.",
        "D3c": "D3c objective `music_code_band_kl` uses the actual sealed conditional ACE-LM top64 plus "
                "tail targets normalized as `recorded_full_vocab_logprobs + lse_full - lse_band`, "
                "shifted to native next-code positions. Parent prompt IDs are not reused and target "
                "codes are absent from the same input position. Only 64000 music rows [252416,316416) "
                "are written within allocated [252416,317952); reserves stay unchanged. Train those "
                "embedding/head row masters with LazyRowAdam plus all-layer native r16 LoRA with "
                "AdamW. The incomplete teacher traversal is blocked, regardless of candidate count.",
        "D3d": "D3d objective `text_answer_kd_plus_vit_feature_mse` combines assistant-only text answer KD "
                "weight1 over 248320 native text vocabulary rows with weight0.1 absolute MSE of actual "
                "QVL ViT-block13 features. Native patch_flat, matching image/frame input, independent "
                "answer teacher, answer-position masks and identical teacher/native vocabulary are "
                "required. Train the selected QVL type6 unit, C11 bridge delta and all-layer native "
                "r16 LoRA. A ViT-only cache, diagnostic ref32, an audio-generation codebook or raw "
                "fixture metadata cannot supply this objective.",
    }
    return (common + objectives[stage] + "\n\nOptimizer (exact engine fields):\n```json\n" +
            json.dumps(OPTIMIZERS[stage], sort_keys=True, indent=2) + "\n```\n\n"
            "Seed20261013. One pass over each accepted train batch exactly once; steps equals its "
            "actual sealed manifest count. D3c/d use 20 warmup steps followed by cosine decay to "
            "0.1 of initial rates. Rank16/alpha16 LoRA, no extra K3 and no gain; trunk FFN width is "
            "read from the actual native base rather than inferred. Preserve 30-minute rolling "
            "checkpoints (latest2), halt on8 consecutive nonfinite skips or more than20/100 skips, "
            "and quarter-point prompt-conditioned decode review. Frozen holdout/source groups "
            "never enter the sampler. Pin immutable base, parity, active runtime, current driver "
            "and engine bytes, corpus/cache/amendment identities and exact resume state. A final "
            "delta is unregistered until native checkpoint composition and independent content/no-"
            "regression review. Preparation does not establish trained-model recovery.\n").encode()


def write_new(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def template(stage, unit, prereg_entry, dtype):
    value = {"schema": "chimera_recovery.native_stage_config.v1", "stage": stage,
             "objective": STAGES[stage][1], "dtype": dtype, "optimizer": copy.deepcopy(OPTIMIZERS[stage]),
             "safety": dict(SAFETY), "seed": 20261013, "sampling_policy": "one_pass", "steps": None,
             "lora": {"rank": 16, "alpha": 16, "k3_rank": 0, "gain": False, "trunk_ffn_width": None},
             "preregistration": prereg_entry,
             "hybrid_module": None, "immutable_base_receipt": None, "parity_receipt": None,
             "runtime_assignment": None, "batches_manifest": None,
             "omni": None, "parent_config": None, "device": None, "kernels": None}
    if unit is not None:
        value["unit"] = unit
    if stage == "D3a":
        value["refit_weights"] = {"chart_in": 1., "block": 1., "chart_out": 1.}
    return value


def prepare(engine_dir, preregistration, out, *, dtype, python=sys.executable, evidence=()):
    if dtype not in {"float32", "bfloat16"} or not python:
        raise Refusal("explicit supported dtype and Python runtime required")
    out = Path(out).resolve()
    if out.exists():
        raise Refusal("new preparation directory required; preserve previous seals")
    inventory = engine_inventory(engine_dir)
    original = pin(preregistration)
    normalized, frozen_body, old_tail = preregistration_parts(Path(preregistration).read_bytes())
    evidence_pins = [pin(path) for path in evidence]
    jobs = []
    for stage in STAGES:
        raw = normalized + amendment(stage)
        _, after_body, after_tail = preregistration_parts(raw)
        if after_body != frozen_body or not after_tail.startswith(old_tail):
            raise Refusal("stage append moved frozen body or existing D3b tail")
        doc = out / f"PREREGISTRATION_{stage}_{DATE}.md"
        write_new(doc, raw)
        prereg_entry = {**pin(doc), "amendment_sha256": hashlib.sha256(after_tail).hexdigest()}
        units = UNITS if stage == "D3a" else (None,) if stage == "D3c" else ("qvl32_vit_L13",)
        for unit in units:
            ident = stage + ("_" + unit if unit else "_music_band")
            path = out / (ident + ".template.json")
            write_new(path, canonical(template(stage, unit, prereg_entry, dtype)) + b"\n")
            jobs.append({"id": ident, "stage": stage, "unit": unit, "template": pin(path),
                         "preregistration": prereg_entry, "status": "blocked_pending_actual_bindings",
                         "native_train_ready": False, "required_data": DEPENDENCIES[stage],
                         "preflight_argv": [python, str(Path(engine_dir).resolve() / "train_stages.py"), "preflight",
                                            "--config", str(out / (ident + ".config.json")), "--out", str(out / (ident + ".preflight.json"))]})
    result = {"schema": "chimera_recovery.native_stage_jobs.v1", "date": DATE, "stage_contracts": list(STAGES),
              "engine_sources": inventory, "source_preregistration": original, "source_evidence": evidence_pins,
              "frozen_body_sha256": FROZEN_SHA, "frozen_body_bytes": FROZEN_BYTES,
              "old_d3b_amendment_sha256": hashlib.sha256(old_tail).hexdigest(), "jobs": jobs,
              "numeric_settings_status": "new dated initial choices; no measured optimum claim",
              "production_ready": False, "training_launched": False, "tensor_targets_loaded": False,
              "existing_d3b_seals_changed": False}
    write_new(out / "manifest.json", canonical(result) + b"\n")
    return {"manifest": pin(out / "manifest.json"), "jobs": len(jobs), "stage_contracts": 3,
            "production_ready": False, "training_launched": False}


def verify_plan(plan_path, expected_sha):
    plan_path = Path(plan_path)
    if sha(plan_path) != require_sha(expected_sha):
        raise Refusal("prepared job manifest changed")
    plan = read_json(plan_path)
    if plan.get("schema") != "chimera_recovery.native_stage_jobs.v1" or plan.get("date") != DATE:
        raise Refusal("prepared job contract differs")
    for entry in plan["engine_sources"].values():
        check_pin(entry, plan_path.parent)
    check_pin(plan["source_preregistration"], plan_path.parent)
    for entry in plan["source_evidence"]:
        check_pin(entry, plan_path.parent)
    return plan


def materialize(plan_path, expected_sha, job_id, bindings_path, out):
    """Pin actual existing artifacts; never turn metadata readiness into a seal."""
    plan_path, out = Path(plan_path).resolve(), Path(out).resolve()
    if out.exists() or out.with_suffix(".binding.json").exists():
        raise Refusal("new materialized config and binding receipt required")
    plan = verify_plan(plan_path, expected_sha)
    matches = [job for job in plan["jobs"] if job["id"] == job_id]
    if len(matches) != 1:
        raise Refusal("prepared job ID absent or ambiguous")
    job = matches[0]
    config = read_json(check_pin(job["template"], plan_path.parent))
    stage, lane = config["stage"], STAGES[config["stage"]][0]
    bindings = read_json(bindings_path)
    required = {"hybrid_module", "immutable_base_receipt", "parity_receipt", "runtime_assignment", "batches_manifest",
                "omni", "parent_config", "device", "kernels", "trunk_ffn_width", "python"}
    if not required.issubset(bindings) or set(bindings) - required - {"compose_plan", "decode_review_receipt"}:
        raise Refusal("bindings need exact actual base/runtime/cache/module paths and explicit native layout")
    if type(bindings["trunk_ffn_width"]) is not int or bindings["trunk_ffn_width"] <= 0:
        raise Refusal("actual positive native trunk FFN width required")
    for key in ("device", "kernels", "python"):
        if not isinstance(bindings[key], str) or not bindings[key]:
            raise Refusal("explicit runtime/kernel/device required")
    for key in ("hybrid_module", "immutable_base_receipt", "parity_receipt", "runtime_assignment", "batches_manifest"):
        config[key] = pin(bindings[key])
    parent_config = pin(bindings["parent_config"])
    omni = Path(bindings["omni"]).resolve()
    if not omni.is_dir():
        raise Refusal("actual immutable native clone directory required")
    config.update({"omni": str(omni), "parent_config": parent_config["path"],
                   "device": bindings["device"], "kernels": bindings["kernels"]})
    config["lora"]["trunk_ffn_width"] = bindings["trunk_ffn_width"]
    for key in ("compose_plan", "decode_review_receipt"):
        if key in bindings:
            config[key] = bindings[key]
    doc_path = check_pin({k: config["preregistration"][k] for k in ("path", "sha256")}, plan_path.parent)
    _, _, tail = preregistration_parts(doc_path.read_bytes())
    amendment_sha = hashlib.sha256(tail).hexdigest()
    if amendment_sha != config["preregistration"]["amendment_sha256"]:
        raise Refusal("new stage amendment moved")
    base, parity, assignment, data = (read_json(config[key]["path"]) for key in
                                    ("immutable_base_receipt", "parity_receipt", "runtime_assignment", "batches_manifest"))
    base_sha = require_sha(base.get("base_sha256"))
    if (base.get("immutable_clone") is not True or base.get("registered") is not True or
            base.get("verified_equal_to_registered_parent") is not True or base.get("clone_sha256") != base_sha or
            Path(base.get("clone_path", "")).resolve() != omni):
        raise Refusal("actual registered immutable clone proof differs")
    if parity.get("verdict") != "PASS" or parity.get("base_sha256") != base_sha:
        raise Refusal("actual native hybrid parity differs")
    if assignment.get("active") is not True or assignment.get("device") != config["device"]:
        raise Refusal("actual active exclusive runtime assignment differs")
    if (data.get("schema") != f"chimera_recovery.{lane}_batches.v1" or data.get("stage") != stage or
            data.get("sealed") is not True or data.get("complete_source_traversal") is not True or
            data.get("split") != "train" or data.get("heldout_groups_excluded") is not True or
            data.get("amendment_sha256") != amendment_sha or not data.get("batches")):
        raise Refusal("actual complete train batches with this new stage amendment are required")
    for key in ("corpus_sha256", "source_manifest_sha256"):
        require_sha(data.get(key))
    if stage == "D3a":
        # train_stages constructs the refit engine identity from these manifest
        # pins, not solely the individual sidecars. Refuse before model loading
        # if that identity would contain missing/different child/target bytes.
        for key in ("D2_child_sha256", "recaptured_host_cache_sha256", "host_write_target_sha256"):
            require_sha(data.get(key))
    data_root, ids = Path(config["batches_manifest"]["path"]).parent, set()
    for item in data["batches"]:
        ident, group = item.get("id"), item.get("recovery_group_id")
        if not isinstance(ident, str) or not ident or ident in ids or not isinstance(group, str) or not group:
            raise Refusal("native batch stable ID/source group absent or duplicate")
        ids.add(ident)
        tensor_path = check_pin(item["tensors"], data_root)
        receipt = read_json(check_pin(item["teacher_receipt"], data_root))
        if (receipt.get("schema") != f"chimera_recovery.{lane}_cache.v1" or receipt.get("sealed") is not True or
                receipt.get("all_finite") is not True or receipt.get("split") != "train" or receipt.get("held") is not False or
                receipt.get("tensors_sha256") != sha(tensor_path) or receipt.get("corpus_sha256") != data["corpus_sha256"] or
                receipt.get("amendment_sha256") != amendment_sha or receipt.get("recovery_group_id") != group):
            raise Refusal("actual native batch/teacher/corpus/group seal differs")
        if stage == "D3a":
            for key in ("D2_child_sha256", "recaptured_host_cache_sha256", "teacher_cache_sha256", "host_write_target_sha256"):
                require_sha(receipt.get(key))
            if (data.get("D2_child_sha256") != base_sha or receipt["D2_child_sha256"] != base_sha or
                    receipt["recaptured_host_cache_sha256"] != data["recaptured_host_cache_sha256"] or
                    receipt["host_write_target_sha256"] != data["host_write_target_sha256"] or
                    receipt["teacher_cache_sha256"] != data["source_manifest_sha256"] or
                    receipt.get("chart_out_objective") != "teacher_delta_host_write" or receipt.get("host_write_target_independent") is not True):
                raise Refusal("D3a independent outward targets must bind this exact D2 child")
        elif stage == "D3d":
            if (receipt.get("teacher_feature_block") != 13 or receipt.get("feature_weight") != .1 or
                    receipt.get("text_weight") != 1. or receipt.get("assistant_only") is not True or
                    receipt.get("text_vocab_rows") != 248320 or receipt.get("native_image_patches_verified") is not True):
                raise Refusal("D3d feature-only cache cannot substitute for aligned native answer KD")
        else:
            if (receipt.get("teacher_branch") != "conditional" or
                    receipt.get("normalization") != "recorded_full_vocab_logprobs + lse_full - lse_band" or
                    receipt.get("target_code_in_same_input_position") is not False or
                    receipt.get("prompt", {}).get("parent_prompt_ids_reused") is not False):
                raise Refusal("D3c requires actual shifted conditional music targets")
    config["steps"] = len(ids)
    config["preparation_sources"] = plan["engine_sources"]
    config["preparation_plan"] = pin(plan_path)
    write_new(out, canonical(config) + b"\n")
    argv = [bindings["python"], plan["engine_sources"]["train_stages.py"]["path"], "preflight",
            "--config", str(out), "--out", str(out.with_suffix(".preflight.json"))]
    result = {"schema": "chimera_recovery.native_stage_binding.v1", "stage": stage, "job_id": job_id,
              "config": pin(out), "bindings_source": pin(bindings_path), "plan": pin(plan_path),
              "parent_config": parent_config, "actual_accepted_batches": len(ids), "preflight_argv": argv,
              "production_ready": False, "status": "actual artifacts bound; production native preflight still required",
              "training_launched": False, "tensor_targets_loaded": False}
    write_new(out.with_suffix(".binding.json"), canonical(result) + b"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--engine-dir", type=Path, required=True)
    prep.add_argument("--preregistration", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    prep.add_argument("--python", default=sys.executable)
    prep.add_argument("--evidence", type=Path, action="append", default=[])
    bind = commands.add_parser("bind")
    bind.add_argument("--plan", type=Path, required=True)
    bind.add_argument("--plan-sha256", required=True)
    bind.add_argument("--job-id", required=True)
    bind.add_argument("--bindings", type=Path, required=True)
    bind.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (prepare(args.engine_dir, args.preregistration, args.out, dtype=args.dtype, python=args.python, evidence=args.evidence)
                  if args.command == "prepare" else materialize(args.plan, args.plan_sha256, args.job_id, args.bindings, args.out))
        print(json.dumps(result, sort_keys=True))
        return 0
    except (Refusal, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "refused", "reason": str(error), "training_launched": False, "production_ready": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
