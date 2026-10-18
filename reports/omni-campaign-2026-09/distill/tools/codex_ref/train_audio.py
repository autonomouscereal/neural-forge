"""Distinct clean-audio observation -> authored assistant CE engine.

The current H3 unit accepts audio-only observations, but its joint attention
must see prompt text only, never teacher-forced answers. Authentic full-coverage
clean encoding and measured source positions remain production prerequisites.
This module creates no encoder, teacher logits, runtime allocation or GPU job.
"""
from __future__ import annotations
from contextlib import contextmanager
import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chimera_recovery_audio_driver", HERE / "train_media.py")
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)
R, Error = T.R, T.Error
OBJECTIVE = "authored_audio_assistant_ce"
READ_POLICY = "h3_prompt_and_clean_observation_only"
STAGE = "D3d_audio_ce"
SAFETY = {"minutes": 30, "keep": 2, "max_consecutive_skips": 8, "window": 100, "max_skip_fraction": .2}


def assistant_ce_targets(model, segment, target_mask, prompt_length, completion_eos_id):
    """Own only real next assistant tokens; shift target ownership exactly once."""
    import torch
    ids = segment["ids"]
    if (ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.int64 or target_mask.shape != ids.shape or
            target_mask.dtype != torch.bool or type(prompt_length) is not int or not 0 < prompt_length < ids.shape[1] or
            type(completion_eos_id) is not int or not 0 <= completion_eos_id < model.base_vocab):
        raise Error("audio CE needs native IDs, boolean target ownership and exact generation-prompt boundary/EOS")
    owned = target_mask[0].nonzero().flatten()
    if (owned.numel() < 2 or int(owned[0]) != prompt_length or
            not torch.equal(owned, torch.arange(prompt_length, int(owned[-1]) + 1, device=owned.device)) or
            int(ids[0, owned[-1]]) != completion_eos_id or int(ids[target_mask].eq(completion_eos_id).sum()) != 1 or
            bool(((ids[target_mask] < 0) | (ids[target_mask] >= model.base_vocab)).any())):
        raise Error("audio CE supervision must be one terminal native assistant body plus completion EOS")
    next_ids = torch.full((ids.numel(),), -1, dtype=torch.int64, device=ids.device)
    weights = torch.zeros(ids.numel(), dtype=torch.float32, device=ids.device)
    selected = target_mask[0, 1:]
    next_ids[:-1] = torch.where(selected, ids[0, 1:], -1)
    weights[:-1] = selected.float()
    return {"next_ids": next_ids, "ce_w": weights, "eos_w": torch.zeros_like(weights)}, int(owned.numel())


def validate_clean_observation(model, segment, prompt_length):
    import torch
    ids, types, indices, features = (segment[key] for key in ("ids", "types", "indices", "features"))
    rows = types.numel()
    if (types.shape != (1, rows) or types.dtype != torch.int64 or rows < 1 or bool((types != 3).any()) or indices.shape != types.shape or
            indices.dtype != torch.int64 or bool(((indices < 0) | (indices >= prompt_length)).any()) or
            torch.unique(indices).numel() != rows or features.shape != (1, rows, model.cfg.native_feature_dim) or
            segment["sigma"].shape != types.shape or bool((segment["sigma"] != 0).any()) or
            segment["positions"].shape != (1, rows, 4)):
        raise Error("audio CE requires clean type3 observation rows wholly before the authentic prompt boundary")
    if bool((ids[0, indices[0]] != model.media_placeholder).any()):
        raise Error("audio CE observation indices must select genuine native media placeholder IDs")
    lo, hi = model.bands[3]["feature"]
    outside = features.clone()
    outside[..., lo:hi] = 0
    source = segment.get("source_positions", {}).get("h3")
    if (bool(outside.any()) or source is None or source.shape != (*ids.shape, 3) or
            any(not bool(torch.isfinite(value).all()) for value in (features, segment["positions"], source)) or
            any(value.requires_grad for value in (features, segment["positions"], source))):
        raise Error("audio CE needs detached genuine H3-band observations and finite measured native/source positions")


@contextmanager
def prompt_only_h3(model, unit_name, prompt_length):
    """Prevent answer -> joint audio attention -> earlier answer target leakage."""
    import torch
    prior_override, original = "unit_write" in model.__dict__, model.unit_write
    def write(placement, x, context):
        if placement.container != unit_name:
            return original(placement, x, context)
        types = context["row_types"]
        if types is None:
            raise Error("audio CE H3 context lacks actual row ownership")
        positions = context["positions"][placement.spec.source]
        prefix = torch.arange(types.shape[1], device=types.device).unsqueeze(0) < prompt_length
        visible = context["valid"] & prefix & (types.eq(0) | types.eq(3))
        result = model.units[unit_name](x.to(model.options.unit_dtype), types, context["levels"], visible, source_positions=positions)
        return result.to(x.dtype)
    model.unit_write = write
    try:
        yield
    finally:
        if prior_override:
            model.unit_write = original
        else:
            delattr(model, "unit_write")


def ce_head_loss(model, targets, positions):
    import torch
    config = model.T.ObjectiveConfig(w_q38=0., lambda_k3=0., w_ce=1., w_eos=0., w_ul=0.,
                                    text_vocab_rows=model.base_vocab, head_chunk_positions=256)
    def loss(normed, context):
        head = model.engine.head_weight[:model.base_vocab]
        parts, gradient = model.T.chunked_objective(normed.to(head.device, head.dtype), head, targets, config,
                                                    denominator=float(positions), backward=True)
        surrogate = (normed * gradient.to(normed.device, normed.dtype)).sum()
        return surrogate - surrogate.detach() + torch.tensor(float(parts["total"]), device=normed.device)
    return loss


class AudioCEEngine(T.D3bEngine):
    checkpoint_schema = "chimera_recovery.audio_ce_checkpoint.v1"

    def __init__(self, module, hybrid, unit, optimizer, identity, *, total_steps):
        import torch
        value = dict(optimizer)
        if "bridge_lr" not in value:
            raise Error("audio CE requires explicit type3 bridge_lr")
        value["emission_lr"] = value.pop("bridge_lr")
        self.module, self.hybrid, self.options = module, hybrid, T.optimizer_config(value)
        self.identity = {**identity, "objective": OBJECTIVE, "read_policy": READ_POLICY,
                         "optimizer_sha256": R.digest(optimizer), "total_steps": total_steps}
        self.placement = next((p for p in hybrid.layout if p.container == unit and p.spec.source == "h3" and 3 in p.type_ids), None)
        if self.placement is None or hybrid.head_deltas or hybrid.bridge_deltas or total_steps < 1:
            raise Error("audio CE needs a fresh genuine H3 audio unit and positive exact budget")
        if identity.get("fixture") is not True:
            for key in ("base_sha256", "clean_encoder_receipt_sha256", "batch_manifest_sha256", "amendment_sha256", "hybrid_module_sha256"):
                R.require_sha(identity.get(key), "audio CE " + key)
            if (identity.get("objective") != OBJECTIVE or identity.get("read_policy") != READ_POLICY or
                    identity.get("heldout_groups_excluded") is not True or hybrid.base_vocab != 248320 or
                    hybrid.bands[3]["feature"] != (480, 512)):
                raise Error("audio CE production requires the distinct amended clean observation/assistant CE contract")
        for native_unit in hybrid.units.values():
            native_unit.requires_grad_(False)
        hybrid.unit_parameters([unit])
        deltas = hybrid.add_bridge_deltas([3])
        for tensor in hybrid.bank.tensors.values():
            tensor.requires_grad_(True)
        self.census = {f"unit.{unit}.{name}": value for name, value in hybrid.units[unit].named_parameters() if value.requires_grad}
        self.census.update({f"bridge.{delta.name}": delta.delta for delta in deltas})
        self.census.update({"lora." + name: tensor for name, tensor in sorted(hybrid.bank.tensors.items()) if tensor.requires_grad})
        groups = [{"name": kind, "params": [value for name, value in self.census.items() if name.startswith(kind + ".")],
                   "lr": rate, "base_lr": rate} for kind, rate in (("unit", self.options["unit_lr"]), ("bridge", self.options["emission_lr"]), ("lora", self.options["lora_lr"]))]
        if any(not group["params"] for group in groups):
            raise Error("audio CE ownership requires the H3 unit, type3 input bridge and shared LoRA")
        self.optimizer = torch.optim.AdamW(groups, betas=tuple(self.options["betas"]), eps=self.options["eps"], weight_decay=self.options["weight_decay"])
        self.ownership = R.optimizer_ownership(self.census, self.optimizer)
        self.initial = {name: value.detach().cpu().clone() for name, value in self.census.items()}
        self.step, self.sampler_counter, self.consecutive_skips, self.skip_window, self.total_steps = 0, 0, 0, [], total_steps
        self.decode_review_passed = False

    def train_step(self, segment, target_mask, *, prompt_length, completion_eos_id):
        import torch
        if self.step >= self.total_steps:
            raise Error("audio CE exhausted one-pass budget")
        targets, count = assistant_ce_targets(self.hybrid, segment, target_mask, prompt_length, completion_eos_id)
        validate_clean_observation(self.hybrid, segment, prompt_length)
        self.optimizer.zero_grad(set_to_none=True)
        try:
            with prompt_only_h3(self.hybrid, self.placement.container, prompt_length):
                loss, _, _ = self.module.full_depth_forward_backward(self.hybrid, segment, ce_head_loss(self.hybrid, targets, count))
        except self.module.D3Error as error:
            if "not finite" not in str(error):
                raise
            return self.nonfinite_skip()
        missing = [name for name, value in self.census.items() if value.grad is None]
        if missing:
            raise Error(f"audio CE trainable received no gradient: {missing[:3]}")
        if not math.isfinite(loss) or any(not bool(torch.isfinite(value.grad).all()) for value in self.census.values()):
            return self.nonfinite_skip()
        norms = {group["name"]: float(torch.sqrt(sum(value.grad.float().square().sum() for value in group["params"]))) for group in self.optimizer.param_groups}
        if any(value <= 0 for value in norms.values()):
            raise Error("audio CE declared optimizer group received zero gradient")
        factor = self.module.trainer_module().lr_factor(self.step, self.options["warmup_steps"], self.total_steps, self.options["min_lr_ratio"])
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        norm = torch.nn.utils.clip_grad_norm_(list(self.census.values()), self.options["clip"], error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1
        self.sampler_counter += 1
        self.consecutive_skips, self.skip_window = 0, (self.skip_window + [False])[-100:]
        return {"loss": loss, "assistant_targets": count, "skipped": False, "step": self.step, "gradient_norm": float(norm), "group_norms": norms,
                "objective": OBJECTIVE, "read_policy": READ_POLICY, "teacher_logits_used": False}

    def export(self, out):
        raise Error("audio CE checkpoint needs a reviewed sequential native composer; no generic D3b emission export")

    def nonfinite_skip(self):
        self.consecutive_skips += 1
        self.skip_window = (self.skip_window + [True])[-100:]
        return {"loss": None, "skipped": True, "step": self.step,
                "halt": self.consecutive_skips >= 8 or (len(self.skip_window) == 100 and sum(self.skip_window) > 20)}


def audio_optimizer(value):
    mapped = dict(value)
    if "bridge_lr" not in mapped:
        raise Error("D3d_audio_ce needs an explicit type3 input bridge rate")
    mapped["emission_lr"] = mapped.pop("bridge_lr")
    T.optimizer_config(mapped)
    expected = {"unit_lr": 2e-5, "bridge_lr": 2e-5, "lora_lr": 1e-4, "betas": [.9, .95], "eps": 1e-8,
                "weight_decay": 0., "clip": 1., "warmup_steps": 20, "min_lr_ratio": .1}
    if value != expected:
        raise Error("initial audio optimizer must match the dated candidate; changed rates need a pinned successor amendment")
    return dict(value)


def audio_batches(path, amendment_sha):
    C = T.load_module(HERE / "audio_cache.py", "chimera_recovery_audio_driver_cache")
    data = T.read_json(path)
    if (data.get("schema") != C.SCHEMA or data.get("stage") != STAGE or data.get("objective") != OBJECTIVE or
            data.get("sealed") is not True or data.get("split") != "train" or data.get("complete_source_traversal") is not True or
            data.get("heldout_groups_excluded") is not True or data.get("amendment_sha256") != amendment_sha or not data.get("batches") or
            data.get("adapter_sha256") != R.sha(HERE / "audio_cache.py") or data.get("native_parent_xyz_parity") is not False):
        raise Error("audio needs complete sealed native CE batches under the explicitly amended chart")
    source = T.pinned_file(data["source"], path.parent)
    C.admission(data["source_admission"], path.parent, source)
    T.pinned_file(data["clean_manifest"], path.parent)
    T.pinned_file(data["charted_attention"], path.parent)
    ids = set()
    for item in data["batches"]:
        if not item.get("id") or item["id"] in ids:
            raise Error("audio batches need unique exact source row IDs")
        ids.add(item["id"])
        T.pinned_file(item["tensors"], path.parent)
        receipt = T.read_json(T.pinned_file(item["teacher_receipt"], path.parent))
        if (receipt.get("schema") != "chimera_recovery.audio_cache.v1" or receipt.get("sealed") is not True or
                receipt.get("all_finite") is not True or receipt.get("held") is not False or receipt.get("split") != "train" or
                receipt.get("id") != item["id"] or receipt.get("recovery_group_id") != item.get("recovery_group_id") or
                receipt.get("objective") != OBJECTIVE or receipt.get("assistant_only") is not True or receipt.get("teacher_logits_used") is not False or
                receipt.get("tensors_sha256") != item["tensors"]["sha256"] or receipt.get("corpus_sha256") != data["corpus_sha256"] or
                receipt.get("amendment_sha256") != amendment_sha or receipt.get("charted_attention_sha256") != data["charted_attention"]["sha256"]):
            raise Error("audio batch/source/assistant/chart receipt differs")
        for key in ("source_record", "dialogue", "encoder_proof", "source_tensors", "source_admission"):
            T.pinned_file(receipt[key], path.parent)
        for name, ref in receipt["tokenizer_files"].items():
            if name not in C.D.TOKENIZER_PINS or ref["sha256"] != C.D.TOKENIZER_PINS[name]:
                raise Error("audio parent tokenizer identity differs")
            T.pinned_file(ref, path.parent)
        if set(receipt["tokenizer_files"]) != set(C.D.TOKENIZER_PINS):
            raise Error("audio parent tokenizer source inventory incomplete")
    return data


def preflight(config_path):
    config, root = T.read_json(config_path), config_path.parent
    verified, blockers = {}, []
    if config.get("schema") != "chimera_recovery.audio_stage_config.v1" or config.get("stage") != STAGE:
        return {"schema": "chimera_recovery.audio_stage_preflight.v1", "production_ready": False, "training_started": False,
                "blockers": ["explicit D3d_audio_ce config is required"], "config_sha256": R.sha(config_path)}
    def check(label, fn):
        try:
            verified[label] = fn()
        except (KeyError, ValueError, TypeError, OSError) as exc:
            blockers.append(label + ": " + str(exc))
    check("optimizer", lambda: audio_optimizer(config["optimizer"]))
    check("preregistration", lambda: T.verify_preregistration(config["preregistration"], root))
    check("hybrid_module", lambda: str(T.pinned_file(config["hybrid_module"], root)))
    check("base", lambda: T.read_json(T.pinned_file(config["immutable_base_receipt"], root)))
    check("parity", lambda: T.read_json(T.pinned_file(config["parity_receipt"], root)))
    check("runtime_assignment", lambda: T.read_json(T.pinned_file(config["runtime_assignment"], root)))
    check("batches_path", lambda: str(T.pinned_file(config["batches_manifest"], root)))
    try:
        if (config.get("after_stage"), config.get("before_stage")) != ("D3d", "D3d_speech_flow"):
            raise Error("audio CE must follow completed vision D3d and precede selected D3d_speech_flow")
        if (config.get("objective") != OBJECTIVE or config.get("read_policy") != READ_POLICY or config.get("dtype") not in {"float32", "bfloat16"} or
                config.get("safety") != SAFETY or
                type(config.get("seed")) is not int or config["seed"] != 20261013):
            raise Error("audio objective, stage order, read policy, dtype, seed and safety must be explicitly amended")
        body = T.pinned_file(config["preregistration"], root).read_bytes().replace(b"\r\n", b"\n")
        tail = body[body.find(b"\n## Amendments"):]
        for required in (STAGE, OBJECTIVE, READ_POLICY, "audio40hz_prompt_offset_categorical_stereo_v1", "native_audio_before_user_close_v1"):
            if required.encode() not in tail:
                raise Error("actual dated preregistration lacks explicit audio selection: " + required)
        if (config.get("lora", {}).get("rank") != 16 or config["lora"].get("alpha") != 16 or
                config["lora"].get("k3_rank") != 0 or config["lora"].get("gain") is not False):
            raise Error("audio stage requires the existing shared r16/alpha16/noK3/no gain bank")
        base = verified["base"]
        if (base.get("immutable_clone") is not True or base.get("registered") is not True or
                base.get("verified_equal_to_registered_parent") is not True or base.get("base_sha256") != base.get("clone_sha256") or
                Path(base.get("clone_path", "")) != Path(config["omni"])):
            raise Error("audio stage base must be the current immutable verified registered clone")
        R.require_sha(base["base_sha256"], "base_sha256")
        verified["shared_r16"] = T.I.preflight_config(config, root, base["base_sha256"])
        verified["native_state"] = T.N.preflight_config(config, root, verified["shared_r16"])
        producer = T.read_json(T.pinned_file(config["native_state"]["source"], root))
        if producer.get("stage") != "D3d":
            raise Error("audio CE must inherit the preceding completed vision D3d cumulative state")
        if (verified["parity"].get("verdict") != "PASS" or verified["parity"].get("base_sha256") != base["base_sha256"] or
                verified["parity"].get("audio_prompt_read_policy") != READ_POLICY or
                verified["parity"].get("audio_full_depth_gradients") is not True):
            raise Error("audio requires actual base hybrid parity and amended read-policy/full-depth gradient proof")
        if verified["runtime_assignment"].get("active") is not True or verified["runtime_assignment"].get("device") != config.get("device"):
            raise Error("audio stage needs an active exclusive allocator device assignment")
        data = audio_batches(Path(verified["batches_path"]), verified["preregistration"]["amendment_sha256"])
        if config.get("sampling_policy") != "one_pass" or type(config.get("steps")) is not int or config["steps"] != len(data["batches"]):
            raise Error("audio budget must equal admitted training rows, one successful update each")
        verified["batches"] = data
    except (KeyError, ValueError, TypeError, OSError) as exc:
        blockers.append(str(exc))
    return {"schema": "chimera_recovery.audio_stage_preflight.v1", "stage": STAGE, "production_ready": not blockers,
            "training_started": False, "blockers": blockers, "config_sha256": R.sha(config_path), "config": config,
            "verified": verified, "driver_sha256": R.sha(Path(__file__)), "native_parent_xyz_parity": False}


def dispatch_step(engine, batch, receipt):
    segment = T.make_segment(batch, media_placeholder=engine.hybrid.media_placeholder)
    return engine.train_step(segment, batch["assistant_target_mask"], prompt_length=receipt["prompt_length"],
                             completion_eos_id=receipt["completion_eos_id"])


def save_audio_checkpoint(engine, path, review=None):
    if engine.decode_review_passed and review is None:
        raise Error("audio completed review must retain its exact free-answer evidence")
    engine.save(path)
    if review is not None:
        (path / "free_answer_review.json").write_bytes(R.canonical(review) + b"\n")
    guard = {"schema": "chimera_recovery.audio_checkpoint_guard.v1", "identity_sha256": R.digest(engine.identity),
        "engine_receipt_sha256": R.sha(path / "receipt.json"), "review_sha256": R.sha(path / "free_answer_review.json") if review is not None else None,
        "step": engine.step, "sampler_counter": engine.sampler_counter, "decode_review_passed": engine.decode_review_passed}
    (path / "audio_guard.json").write_bytes(R.canonical(guard) + b"\n")


def resume_audio_checkpoint(engine, path):
    engine.resume(path)
    guard = T.read_json(path / "audio_guard.json")
    if (guard.get("schema") != "chimera_recovery.audio_checkpoint_guard.v1" or guard.get("identity_sha256") != R.digest(engine.identity) or
            guard.get("engine_receipt_sha256") != R.sha(path / "receipt.json") or guard.get("step") != engine.step or
            guard.get("sampler_counter") != engine.sampler_counter or guard.get("decode_review_passed") != engine.decode_review_passed):
        raise Error("audio exact resume guard/engine/read-policy identity differs")
    if engine.decode_review_passed:
        review_path = T.pinned_file({"path": str(path / "free_answer_review.json"), "sha256": guard["review_sha256"]}, path)
        return T.read_json(review_path)
    if guard.get("review_sha256") is not None:
        raise Error("uncompleted audio review has unexpected accepted evidence")
    return None


def train(config_path, out, resume=None):
    check = preflight(config_path)
    if not check["production_ready"]:
        raise Error("audio preflight blocked: " + "; ".join(check["blockers"]))
    if out.exists():
        raise Error("preserve existing run; select new output and explicit checkpoint resume")
    import torch
    C = T.load_module(HERE / "audio_cache.py", "chimera_recovery_audio_production_cache")
    config, verified = check["config"], check["verified"]
    module = T.load_module(Path(verified["hybrid_module"]), "d3_hybrid_audio_production")
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    plan = module.trainer_module().SitePlanConfig(rank=16, alpha=16., k3_rank=0, trunk_ffn_width=config["lora"]["trunk_ffn_width"], gain=False)
    hybrid = module.HybridD3(config["omni"], config["parent_config"], module.HybridOptions(dtype=getattr(torch, config["dtype"]),
        device=config["device"], kernels=config["kernels"], lora=plan, lora_seed=config["seed"], plan=config.get("compose_plan")))
    inherited = T.I.initialize_stage(hybrid, verified["shared_r16"], config_path.parent)
    native = T.N.initialize_stage(hybrid, verified["native_state"], config_path.parent)
    identity = {"stage": STAGE, "objective": OBJECTIVE, "read_policy": READ_POLICY, "config_sha256": check["config_sha256"],
        "base_sha256": verified["base"]["base_sha256"], "batch_manifest_sha256": config["batches_manifest"]["sha256"],
        "clean_encoder_receipt_sha256": verified["batches"]["encoder_receipt_inventory_sha256"],
        "amendment_sha256": verified["preregistration"]["amendment_sha256"], "hybrid_module_sha256": config["hybrid_module"]["sha256"],
        "driver_sha256": check["driver_sha256"], "cache_adapter_sha256": R.sha(HERE / "audio_cache.py"),
        "heldout_groups_excluded": True, "shared_r16": inherited, "native_state": native}
    engine = AudioCEEngine(module, hybrid, config["unit"], config["optimizer"], identity, total_steps=config["steps"])
    accepted_review = resume_audio_checkpoint(engine, resume) if resume else None
    out.mkdir(parents=True)
    (out / "preflight.json").write_bytes(R.canonical(check) + b"\n")
    save_audio_checkpoint(engine, out / "checkpoint-initial", accepted_review)
    (out / "initial_receipt.json").write_bytes(R.canonical({"schema": "chimera_recovery.audio_stage_initial.v1",
        "identity": engine.identity, "trainable_census": engine.ownership, "step": engine.step, "sampler_counter": engine.sampler_counter}) + b"\n")
    batches, root = verified["batches"]["batches"], Path(verified["batches_path"]).parent
    last_safety = time.monotonic()
    while engine.step < engine.total_steps:
        quarter = max(1, math.ceil(engine.total_steps * .25))
        if engine.step >= quarter and not engine.decode_review_passed:
            review_path = config.get("free_answer_review")
            # The stable configured path is populated later by the real heldout
            # review. Its exact accepted contents are pinned in each checkpoint.
            candidate = Path(review_path) if review_path else None
            candidate = candidate if candidate is None or candidate.is_absolute() else config_path.parent / candidate
            review = T.read_json(candidate) if candidate and candidate.is_file() else {}
            if (review.get("schema") != "chimera_recovery.audio_free_answer_review.v1" or review.get("passed") is not True or
                    review.get("run_identity_sha256") != R.digest(engine.identity) or not quarter <= review.get("trained_step", -1) <= engine.step or
                    review.get("teacher_forced") is not False or review.get("heldout_only") is not True or not review.get("answers")):
                save_audio_checkpoint(engine, out / f"checkpoint-answer-review-{engine.step:08d}", accepted_review)
                return {"stage": STAGE, "status": "awaiting_audio_free_answer_review", "step": engine.step, "training_complete": False}
            engine.decode_review_passed = True
            accepted_review = review
        if engine.sampler_counter >= len(batches):
            raise Error("audio one-pass sampler exhausted early")
        item = batches[engine.sampler_counter]
        receipt = T.read_json(T.pinned_file(item["teacher_receipt"], root))
        cpu_batch = T.load_native_batch(item, root, "cpu")
        try:
            C.validate_batch(cpu_batch, receipt, root)
            batch = {key: value.to(config["device"]) for key, value in cpu_batch.items()}
            result = dispatch_step(engine, batch, receipt)
        except Exception:
            save_audio_checkpoint(engine, out / f"checkpoint-failure-{engine.step:08d}", accepted_review)
            raise
        with (out / "steps.jsonl").open("ab") as stream:
            stream.write(R.canonical(result) + b"\n")
        if result.get("halt") is True:
            save_audio_checkpoint(engine, out / f"checkpoint-skip-halt-{engine.step:08d}", accepted_review)
            raise Error("audio persistent nonfinite skips require a ruling")
        if time.monotonic() - last_safety >= 1800:
            save_audio_checkpoint(engine, out / f"checkpoint-safety-{engine.step:08d}", accepted_review)
            T.prune_safety_checkpoints(out, keep=2)
            last_safety = time.monotonic()
    save_audio_checkpoint(engine, out / "checkpoint-final", accepted_review)
    shared = T.I.export_from_engine(engine, out / "checkpoint-final", out / "shared-r16", verified["shared_r16"]["context"], STAGE, config["unit"])
    native = T.N.export_from_engine(engine, out / "checkpoint-final", out / "native-state", verified["shared_r16"]["context"], STAGE, shared)
    return {"schema": "chimera_recovery.audio_stage_run.v1", "stage": STAGE, "status": "complete", "step": engine.step,
        "training_complete": True, "registered": False, "shared_r16": shared, "native_state": native,
        "objective": OBJECTIVE, "native_parent_xyz_parity": False, "speech_generation_trained": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train"):
        command = sub.add_parser(name)
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
