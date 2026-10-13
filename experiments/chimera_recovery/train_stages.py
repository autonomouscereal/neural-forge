"""Production dispatch for native D3a refit, D3c music and D3d vision recovery.

No GPU or checkpoint is loaded by preflight. Each stage needs a sealed native
batch manifest, exact source-group exclusion and independently pinned objective
inputs. The current ViT-only source cannot satisfy D3d answer KD by itself.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chimera_recovery_stage_common", HERE / "train_media.py")
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)
R, Error = T.R, T.Error
STAGES = {"D3a": ("refit", "teacher_delta_host_write", "train_refit.py"),
          "D3c": ("music", "music_code_band_kl", "train_music.py"),
          "D3d": ("vision", "text_answer_kd_plus_vit_feature_mse", "train_vision.py")}
SAFETY = {"minutes": 30, "keep": 2, "max_consecutive_skips": 8, "window": 100, "max_skip_fraction": .2}


def load_engine_module(stage):
    return T.load_module(HERE / STAGES[stage][2], "chimera_recovery_production_" + stage)


def stage_optimizer(stage, value):
    engine = load_engine_module(stage)
    if stage == "D3c":
        return engine.options(value)
    if stage == "D3d":
        mapped = dict(value)
        if "bridge_lr" not in mapped:
            raise Error("D3d requires explicit C11 bridge rate")
        mapped["emission_lr"] = mapped.pop("bridge_lr")
        T.optimizer_config(mapped)
        return dict(value)
    if set(value) != {"lr", "betas", "eps", "weight_decay", "clip"}:
        raise Error("D3a requires explicit unit-only Adam settings")
    if (any(type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] <= 0 for key in ("lr", "eps", "clip")) or
            not isinstance(value["betas"], list) or len(value["betas"]) != 2 or any(not 0 <= beta < 1 for beta in value["betas"]) or
            not math.isfinite(value["weight_decay"]) or value["weight_decay"] < 0):
        raise Error("D3a invalid Adam settings")
    return dict(value)


def stage_data(stage, manifest_path, amendment_sha):
    lane, _, _ = STAGES[stage]
    data = T.read_json(manifest_path)
    if (data.get("schema") != f"chimera_recovery.{lane}_batches.v1" or data.get("stage") != stage or data.get("sealed") is not True or
            data.get("complete_source_traversal") is not True or data.get("split") != "train" or data.get("heldout_groups_excluded") is not True or
            data.get("amendment_sha256") != amendment_sha or not data.get("batches")):
        raise Error(f"{stage} needs complete sealed admitted native {lane} batches; raw/partial teacher cache is insufficient")
    R.require_sha(data.get("corpus_sha256"), "admitted_corpus_sha256")
    R.require_sha(data.get("source_manifest_sha256"), "source_teacher_manifest_sha256")
    if stage == "D3a":
        # These manifest pins become the actual engine's resume identity.
        # Checking sidecars alone would admit a config that fails only after
        # the immutable base is opened and the hybrid is constructed.
        for key in ("D2_child_sha256", "recaptured_host_cache_sha256", "host_write_target_sha256"):
            R.require_sha(data.get(key), "D3a manifest " + key)
    if stage == "D3c":
        music = T.load_module(HERE / "music_cache.py", "chimera_recovery_music_stage_source")
        music.sealed_source(Path(data["source_cache"]), Path(data["source_items"]), data["source_manifest_sha256"])
    ids, groups = set(), set()
    for item in data["batches"]:
        ident = item.get("id")
        if not isinstance(ident, str) or not ident or ident in ids or not item.get("recovery_group_id"):
            raise Error("native batch stable IDs/groups are missing or duplicate")
        ids.add(ident)
        groups.add(item["recovery_group_id"])
        T.pinned_file(item["tensors"], manifest_path.parent)
        receipt = T.read_json(T.pinned_file(item["teacher_receipt"], manifest_path.parent))
        if (receipt.get("schema") != f"chimera_recovery.{lane}_cache.v1" or receipt.get("sealed") is not True or receipt.get("all_finite") is not True or
                receipt.get("split") != "train" or receipt.get("held") is not False or
                receipt.get("tensors_sha256") != item["tensors"]["sha256"] or receipt.get("corpus_sha256") != data["corpus_sha256"] or
                receipt.get("amendment_sha256") != amendment_sha or receipt.get("recovery_group_id") != item["recovery_group_id"]):
            raise Error("native stage receipt does not bind exact batch/corpus/group/objective")
        if stage == "D3c":
            if (receipt.get("teacher_branch") != "conditional" or receipt.get("normalization") != "recorded_full_vocab_logprobs + lse_full - lse_band" or
                    receipt.get("target_code_in_same_input_position") is not False or receipt.get("prompt", {}).get("parent_prompt_ids_reused") is not False or
                    receipt.get("written_code_rows") != [252416, 316416] or receipt.get("allocated_music_rows") != [252416, 317952] or
                    receipt.get("source_manifest_sha256") != data.get("source_manifest_sha256") or
                    receipt.get("adapter_sha256") != R.sha(HERE / "music_cache.py")):
                raise Error("D3c requires reviewed native ACE-LM conditional band adapter")
        elif stage == "D3d":
            if (receipt.get("teacher_feature_block") != 13 or receipt.get("feature_weight") != .1 or receipt.get("text_weight") != 1. or
                    receipt.get("assistant_only") is not True or receipt.get("text_vocab_rows") != 248320 or
                    receipt.get("native_image_patches_verified") is not True):
                raise Error("D3d needs authentic native patches, assistant-only text KD and weight0.1 ViT-block13 features")
            native_vocab = T.pinned_file(receipt["native_vocab"], manifest_path.parent)
            teacher_vocab = T.pinned_file(receipt["teacher_vocab"], manifest_path.parent)
            if T.read_json(native_vocab) != T.read_json(teacher_vocab):
                raise Error("D3d teacher answer token IDs require exact native vocabulary alignment; no guessed remapping")
            for key in ("source_feature_manifest", "source_feature_shard", "source_answer_manifest", "source_answer_shard", "source_image"):
                T.pinned_file(receipt[key], manifest_path.parent)
            feature_manifest = T.read_json(T.pinned_file(receipt["source_feature_manifest"], manifest_path.parent))
            if (feature_manifest.get("schema") != "omni-campaign.teacher_cache.v2" or feature_manifest.get("teacher") != "qvl32_vit_l13" or
                    feature_manifest.get("finite") is not True or feature_manifest.get("teacher_integrity", {}).get("equal") is not True or
                    feature_manifest.get("teacher_integrity", {}).get("mode") != "full" or
                    feature_manifest.get("corpus", {}).get("contamination_screen", {}).get("applied") is not True):
                raise Error("D3d excludes diagnostic QVLref32; training feature source needs a full finite screened teacher seal")
        else:
            for key in ("D2_child_sha256", "recaptured_host_cache_sha256", "teacher_cache_sha256", "host_write_target_sha256"):
                R.require_sha(receipt.get(key), "D3a " + key)
            if (any(receipt[key] != data[key] for key in ("D2_child_sha256", "recaptured_host_cache_sha256", "host_write_target_sha256")) or
                    receipt["teacher_cache_sha256"] != data["source_manifest_sha256"]):
                raise Error("D3a batch child/host/teacher/write target pins differ from manifest engine identity")
            if receipt.get("chart_out_objective") != "teacher_delta_host_write" or receipt.get("host_write_target_independent") is not True:
                raise Error("D3a outward chart needs independent sealed write targets recaptured on this child")
    return data, {"admitted_groups": len(groups), "accepted_batches": len(ids)}


def preflight(config_path):
    config, root = T.read_json(config_path), config_path.parent
    blockers, verified = [], {}
    stage = config.get("stage")
    if stage not in STAGES or config.get("schema") != "chimera_recovery.native_stage_config.v1":
        return {"schema": "chimera_recovery.native_stage_preflight.v1", "production_ready": False, "training_started": False,
                "blockers": ["config must select explicit D3a/D3c/D3d native stage"], "config_sha256": R.sha(config_path)}
    def check(label, fn):
        try:
            verified[label] = fn()
        except (Error, KeyError, ValueError, TypeError, OSError) as error:
            blockers.append(label + ": " + str(error))
    check("optimizer", lambda: stage_optimizer(stage, config["optimizer"]))
    check("preregistration", lambda: T.verify_preregistration(config["preregistration"], root))
    check("hybrid_module", lambda: str(T.pinned_file(config["hybrid_module"], root)))
    check("base", lambda: T.read_json(T.pinned_file(config["immutable_base_receipt"], root)))
    check("parity", lambda: T.read_json(T.pinned_file(config["parity_receipt"], root)))
    check("runtime_assignment", lambda: T.read_json(T.pinned_file(config["runtime_assignment"], root)))
    check("batches_path", lambda: str(T.pinned_file(config["batches_manifest"], root)))
    try:
        if config.get("objective") != STAGES[stage][1] or config.get("dtype") not in {"float32", "bfloat16"}:
            raise Error("stage objective/dtype must be explicit")
        if config.get("safety") != SAFETY or type(config.get("seed")) is not int or type(config.get("steps")) is not int or config["steps"] < 1:
            raise Error("stage needs frozen rolling checkpoints/skip guard, seed and positive budget")
        if config.get("lora", {}).get("rank") != 16 or config["lora"].get("alpha") != 16 or config["lora"].get("k3_rank") != 0 or config["lora"].get("gain") is not False:
            raise Error("stage hybrid must explicitly select r16/alpha16/noK3/no gain")
        base = verified["base"]
        if (base.get("immutable_clone") is not True or base.get("registered") is not True or base.get("verified_equal_to_registered_parent") is not True or
                base.get("base_sha256") != base.get("clone_sha256") or Path(base.get("clone_path", "")) != Path(config["omni"])):
            raise Error("stage base must be an immutable registered verified clone")
        R.require_sha(base["base_sha256"], "base_sha256")
        verified["shared_r16"] = T.I.preflight_config(config, root, base["base_sha256"])
        if verified["parity"].get("verdict") != "PASS" or verified["parity"].get("base_sha256") != base["base_sha256"]:
            raise Error("hybrid parity must pass on exact stage base")
        if verified["runtime_assignment"].get("active") is not True or verified["runtime_assignment"].get("device") != config.get("device"):
            raise Error("explicit exclusive runtime/device assignment differs")
        data, summary = stage_data(stage, Path(verified["batches_path"]), verified["preregistration"]["amendment_sha256"])
        if config.get("sampling_policy") != "one_pass" or config["steps"] != len(data["batches"]):
            raise Error("native stage must consume each accepted batch exactly once")
        verified["batches"], verified["admission"] = data, summary
        if stage == "D3a":
            weights = config.get("refit_weights", {})
            if (data.get("D2_child_sha256") != base["base_sha256"] or set(weights) != {"chart_in", "block", "chart_out"} or
                    any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in weights.values())):
                raise Error("D3a captures must bind this exact D2 child and all explicit refit weights")
    except (Error, KeyError, ValueError, TypeError, OSError) as error:
        blockers.append(str(error))
    if stage == "D3d" and "batches" not in verified:
        blockers.append("ViT-only Stage-C QVL cache has no native patch_flat or assistant-answer KD; those independent aligned artifacts are required")
    return {"schema": "chimera_recovery.native_stage_preflight.v1", "stage": stage, "production_ready": not blockers,
            "training_started": False, "config_sha256": R.sha(config_path), "config": config, "verified": verified, "blockers": blockers,
            "driver_sha256": R.sha(Path(__file__)), "engine_sha256": R.sha(HERE / STAGES[stage][2])}


def dispatch_step(engine, stage, batch, receipt):
    """Concrete calls into actual native engines, shared by production/CPU proof."""
    if stage == "D3c":
        return engine.train_step({"ids": batch["input_ids"]}, prompt_length=receipt["prompt_length"],
                                 topk_codes=batch["topk_codes"], topk_logprobs_band=batch["topk_logprobs_band"])
    if stage == "D3a":
        return engine.train_step(batch["host_rows"], batch["parent_in"], batch["parent_out"], batch["host_write_target"],
                                 levels=batch.get("levels"), row_types=batch.get("row_types"), positions=batch.get("source_positions"))
    if stage == "D3d":
        targets = {key: batch["answer." + key] for key in ("next_ids", "ce_w", "eos_w", "q_w", "q_ids", "q_lp")}
        return engine.train_step(T.make_segment(batch, media_placeholder=engine.hybrid.media_placeholder), targets, batch["teacher_features"],
                                 answer_positions=receipt["answer_positions"])
    raise Error("unsupported native stage dispatch")


def validate_music_batch(batch, receipt):
    """Reproduce native targets from the actual sealed raw source before update."""
    import torch
    from safetensors.torch import load_file
    music = T.load_module(HERE / "music_cache.py", "chimera_recovery_music_stage_validate")
    source = Path(receipt["source_cache"]) / receipt["source_shard"]
    if source.parent != Path(receipt["source_cache"]) or R.sha(source) != receipt["source_shard_sha256"]:
        raise Error("D3c original sealed raw source changed")
    sidecar_path = source.with_suffix(".json")
    if R.sha(sidecar_path) != receipt["source_sidecar_sha256"]:
        raise Error("D3c original source sidecar changed")
    prefix = batch["input_ids"][0, :receipt["prompt_length"]].tolist()
    original, proof = music.native_music_batch(load_file(str(source), device="cpu"), T.read_json(sidecar_path), prefix)
    if set(batch) != set(original) or any(not torch.equal(batch[key], value) for key, value in original.items()) or proof["teacher_positions"] != receipt["teacher_positions"]:
        raise Error("D3c prepared targets differ from actual sealed raw source recipe")


def save_dispatch_checkpoint(engine, path, policy):
    """Bind engine state to wrapper skip/review policy for exact resumed safety."""
    engine.save(path)
    policy_path = path / "dispatch_policy.json"
    policy_path.write_bytes(R.canonical(policy) + b"\n")
    (path / "dispatch_receipt.json").write_bytes(R.canonical({"schema": "chimera_recovery.dispatch_checkpoint.v1",
        "identity_sha256": R.digest(engine.identity), "engine_receipt_sha256": R.sha(path / "receipt.json"),
        "policy_sha256": R.sha(policy_path), "step": engine.step, "sampler_counter": engine.sampler_counter}) + b"\n")


def resume_dispatch_policy(engine, path):
    receipt = T.read_json(path / "dispatch_receipt.json")
    if (receipt.get("schema") != "chimera_recovery.dispatch_checkpoint.v1" or receipt.get("identity_sha256") != R.digest(engine.identity) or
            receipt.get("engine_receipt_sha256") != R.sha(path / "receipt.json") or receipt.get("policy_sha256") != R.sha(path / "dispatch_policy.json") or
            receipt.get("step") != engine.step or receipt.get("sampler_counter") != engine.sampler_counter):
        raise Error("native dispatch resume safety policy missing/changed")
    policy = T.read_json(path / "dispatch_policy.json")
    if (set(policy) != {"consecutive_skips", "window", "decode_review_passed"} or type(policy["consecutive_skips"]) is not int or
            policy["consecutive_skips"] < 0 or len(policy["window"]) > 100 or any(type(value) is not bool for value in policy["window"]) or
            type(policy["decode_review_passed"]) is not bool):
        raise Error("native dispatch safety policy invalid")
    return policy


def train(config_path, out, resume=None):
    check = preflight(config_path)
    if not check["production_ready"]:
        raise Error("native stage preflight blocked: " + "; ".join(check["blockers"]))
    if out.exists():
        raise Error("preserve existing run; use new output and explicit resume")
    import torch
    config, verified = check["config"], check["verified"]
    stage = config["stage"]
    module = T.load_module(Path(verified["hybrid_module"]), "d3_hybrid_remaining_production")
    engine_module = load_engine_module(stage)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    plan = module.trainer_module().SitePlanConfig(rank=16, alpha=16., k3_rank=0, trunk_ffn_width=config["lora"]["trunk_ffn_width"], gain=False)
    hybrid = module.HybridD3(config["omni"], config["parent_config"], module.HybridOptions(dtype=getattr(torch, config["dtype"]), device=config["device"],
                            kernels=config["kernels"], lora=plan, lora_seed=config["seed"], plan=config.get("compose_plan")))
    inherited = T.I.initialize_stage(hybrid, verified["shared_r16"], config_path.parent)
    identity = {"stage": stage, "config_sha256": check["config_sha256"], "base_sha256": verified["base"]["base_sha256"],
                "cache_manifest_sha256": config["batches_manifest"]["sha256"], "teacher_cache_sha256": verified["batches"].get("source_manifest_sha256"),
                "amendment_sha256": verified["preregistration"]["amendment_sha256"], "hybrid_module_sha256": config["hybrid_module"]["sha256"],
                "driver_sha256": check["driver_sha256"], "engine_sha256": check["engine_sha256"], "heldout_groups_excluded": True, "shared_r16": inherited}
    if stage == "D3c":
        identity["teacher_kind"] = "sealed_ACE_LM_top64_tail"
        engine = engine_module.MusicBandEngine(module, hybrid, config["optimizer"], identity, total_steps=config["steps"])
    elif stage == "D3d":
        identity["teacher_feature_block"] = 13
        engine = engine_module.VisionEngine(module, hybrid, config["unit"], config["optimizer"], identity, total_steps=config["steps"])
    else:
        identity.update({key: verified["batches"].get(key) for key in ("D2_child_sha256", "recaptured_host_cache_sha256", "host_write_target_sha256")})
        identity["chart_out_objective"] = "teacher_delta_host_write"
        engine = engine_module.UnitRefitEngine(module, hybrid, config["unit"], config["optimizer"], identity, total_steps=config["steps"], weights=config["refit_weights"])
    identity = engine.identity
    policy = {"consecutive_skips": 0, "window": [], "decode_review_passed": False}
    if resume:
        engine.resume(resume)
        policy = resume_dispatch_policy(engine, resume)
    out.mkdir(parents=True)
    (out / "preflight.json").write_bytes(R.canonical(check) + b"\n")
    save_dispatch_checkpoint(engine, out / "checkpoint-initial", policy)
    (out / "initial_receipt.json").write_bytes(R.canonical({"schema": "chimera_recovery.native_stage_initial.v1", "identity": identity,
        "trainable_census": engine.ownership, "step": engine.step, "sampler_counter": engine.sampler_counter}) + b"\n")
    batches = verified["batches"]["batches"]
    root = Path(verified["batches_path"]).parent
    last_safety = time.monotonic()
    while engine.step < config["steps"]:
        quarter = max(1, math.ceil(config["steps"] * .25))
        if engine.step >= quarter and not policy["decode_review_passed"]:
            filename = config.get("decode_review_receipt")
            review = T.read_json(Path(filename)) if filename and Path(filename).is_file() else {}
            if (review.get("meaningful_prompt_conditioned_content") is not True or review.get("run_identity_sha256") != R.digest(identity) or
                    not quarter <= review.get("trained_step", -1) <= engine.step):
                save_dispatch_checkpoint(engine, out / f"checkpoint-decode-review-{engine.step:08d}", policy)
                return {"stage": stage, "status": "awaiting_decode_review", "step": engine.step, "training_complete": False}
            policy["decode_review_passed"] = True
        if engine.sampler_counter >= len(batches):
            raise Error("one-pass native stage sampler exhausted early")
        item = batches[engine.sampler_counter]
        receipt = T.read_json(T.pinned_file(item["teacher_receipt"], root))
        # Music tables need CPU targets; vision/refit execute on native device.
        batch = T.load_native_batch(item, root, "cpu" if stage == "D3c" else config["device"])
        try:
            if stage == "D3c":
                validate_music_batch(batch, receipt)
            elif stage == "D3d":
                vision = T.load_module(HERE / "vision_cache.py", "chimera_recovery_vision_stage_validate")
                vision.validate_vision_batch(batch, receipt, root)
            result = dispatch_step(engine, stage, batch, receipt)
        except Exception:
            save_dispatch_checkpoint(engine, out / f"checkpoint-failure-{engine.step:08d}", policy)
            raise
        skipped = result.get("skipped") is True
        policy["consecutive_skips"] = policy["consecutive_skips"] + 1 if skipped else 0
        policy["window"] = (policy["window"] + [skipped])[-100:]
        with (out / "steps.jsonl").open("ab") as stream:
            stream.write(R.canonical(result) + b"\n")
        if policy["consecutive_skips"] >= 8 or (len(policy["window"]) == 100 and sum(policy["window"]) > 20):
            save_dispatch_checkpoint(engine, out / f"checkpoint-skip-halt-{engine.step:08d}", policy)
            raise Error("persistent nonfinite skips require a ruling")
        if time.monotonic() - last_safety >= 1800:
            save_dispatch_checkpoint(engine, out / f"checkpoint-safety-{engine.step:08d}", policy)
            T.prune_safety_checkpoints(out, keep=2)
            last_safety = time.monotonic()
    save_dispatch_checkpoint(engine, out / "checkpoint-final", policy)
    shared = T.I.export_from_engine(engine, out / "checkpoint-final", out / "shared-r16", verified["shared_r16"]["context"], stage, config.get("unit"))
    return {"schema": "chimera_recovery.native_stage_run.v1", "stage": stage, "status": "complete", "step": engine.step,
            "training_complete": True, "registered": False, "requires_checkpoint_compose_adapter": True, "shared_r16": shared}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name == "train":
            command.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    result = preflight(args.config) if args.command == "preflight" else train(args.config, args.out, args.resume)
    if args.command == "preflight":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(R.canonical(result) + b"\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"config", "verified"}}))
    return 0 if result.get("production_ready", result.get("training_complete", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
