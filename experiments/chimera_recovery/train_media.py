"""D3b native-media recovery driver and explicit CPU fixture proof.

Production requires sealed accepted native batches, an immutable registered base,
hybrid parity, and a pinned dated amendment. Nothing starts from a raw sidecar or
an unverified teacher. D3a/c/d are separate objectives and are not implemented.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time
from typing import Any

HERE = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = load_module(HERE / "training.py", "chimera_recovery_training_media")
Error = R.ReadinessError
FROZEN_BODY_SHA256 = "c24d21e08815d7acd2ce14aea989d16f84f167eb24a40597966be19284cb1370"
FROZEN_BODY_BYTES = 15339
SCHEMA = "chimera_recovery.d3b_run.v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pinned_file(entry: dict[str, Any], root: Path) -> Path:
    path = Path(entry["path"])
    path = path if path.is_absolute() else root / path
    if not path.is_file() or R.sha(path) != R.require_sha(entry.get("sha256"), "file.sha256"):
        raise Error(f"pinned file missing or changed: {path}")
    return path.resolve()


def verify_preregistration(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    path = pinned_file(entry, root)
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    start, end = raw.find(b"## 1. Scope"), raw.find(b"\n## Amendments")
    if start < 0 or end <= start:
        raise Error("frozen preregistration markers missing")
    body = raw[start:end]
    if __import__("hashlib").sha256(body).hexdigest() != FROZEN_BODY_SHA256 or len(body) != FROZEN_BODY_BYTES:
        raise Error("frozen D1-D4 preregistration body moved")
    amendment = raw[end:]
    observed = __import__("hashlib").sha256(amendment).hexdigest()
    if observed != R.require_sha(entry.get("amendment_sha256"), "amendment_sha256"):
        raise Error("dated recovery amendment digest moved")
    if b"guided_velocity" not in amendment or b"2026-10-13" not in amendment:
        raise Error("dated amendment must explicitly select guided_velocity")
    return {"document_sha256": R.sha(path), "frozen_sha256": FROZEN_BODY_SHA256,
            "frozen_bytes": FROZEN_BODY_BYTES, "amendment_sha256": observed, "newline_normalization": "CRLF to LF"}


def optimizer_config(value: dict[str, Any]) -> dict[str, Any]:
    # Frozen D3 gives trainable sites/objective, but no separate optimizer/LR
    # defaults. Require the amendment's values instead of inventing them.
    required = {"unit_lr", "emission_lr", "lora_lr", "betas", "eps", "weight_decay", "clip", "warmup_steps", "min_lr_ratio"}
    if set(value) != required:
        raise Error(f"optimizer must pin exactly {sorted(required)}")
    for key in ("unit_lr", "emission_lr", "lora_lr", "eps", "clip"):
        if not isinstance(value[key], (int, float)) or isinstance(value[key], bool) or not math.isfinite(value[key]) or value[key] <= 0:
            raise Error(f"optimizer {key} must be finite and positive")
    if not isinstance(value["betas"], list) or len(value["betas"]) != 2 or any(not 0 <= beta < 1 for beta in value["betas"]):
        raise Error("optimizer betas must be explicit in [0,1)")
    if value["weight_decay"] < 0 or not math.isfinite(value["weight_decay"]) or not 0 < value["min_lr_ratio"] <= 1:
        raise Error("invalid weight decay or minimum LR ratio")
    if type(value["warmup_steps"]) is not int or value["warmup_steps"] < 0:
        raise Error("warmup_steps must be a nonnegative integer")
    return dict(value)


def preflight(config_path: Path) -> dict[str, Any]:
    config, root = read_json(config_path), config_path.parent
    blockers, verified = [], {}
    if config.get("schema") != "chimera_recovery.d3b_config.v1" or config.get("stage") != "D3b":
        blockers.append("only D3b production driver is implemented; D3a/c/d remain separate blocked lanes")
    try:
        verified["optimizer"] = optimizer_config(config["optimizer"])
        verified["preregistration"] = verify_preregistration(config["preregistration"], root)
        if config.get("objective") != "guided_velocity":
            raise Error("objective must explicitly be guided_velocity")
        if config.get("lora", {}).get("rank") != 16 or config["lora"].get("alpha") != 16 or config["lora"].get("k3_rank") != 0 or config["lora"].get("gain") is not False:
            raise Error("D3b requires declared r16 LoRA, alpha16, no K3 extra adapter or gain")
        if type(config.get("steps")) is not int or config["steps"] < 1 or type(config.get("seed")) is not int:
            raise Error("positive step budget and deterministic integer seed required")
        if config.get("dtype") not in {"float32", "bfloat16"}:
            raise Error("D3b supports explicit fp32/bf16 frozen forwards, fp32 trainable masters")
        if config.get("safety") != {"minutes": 30, "keep": 2, "max_consecutive_skips": 8, "window": 100, "max_skip_fraction": 0.2}:
            raise Error("rolling checkpoint and finite-skip policy differs from frozen section 2")
        if config.get("device") is None or config.get("runtime_assignment") is None:
            raise Error("explicit device and exclusive runtime assignment required")
        assignment = read_json(pinned_file(config["runtime_assignment"], root))
        if assignment.get("active") is not True or assignment.get("device") != config["device"]:
            raise Error("runtime assignment is inactive or device differs")
        module = pinned_file(config["hybrid_module"], root)
        verified["hybrid_module"] = str(module)
        base = read_json(pinned_file(config["immutable_base_receipt"], root))
        if base.get("immutable_clone") is not True or base.get("registered") is not True or base.get("verified_equal_to_registered_parent") is not True:
            raise Error("base must be a registered immutable byte clone with verified identity")
        for key in ("base_sha256", "clone_sha256"):
            R.require_sha(base.get(key), key)
        if base["base_sha256"] != base["clone_sha256"] or Path(base.get("clone_path", "")) != Path(config.get("omni", "")):
            raise Error("configured base path/digest differs from immutable clone receipt")
        verified["base"] = base
        parity = read_json(pinned_file(config["parity_receipt"], root))
        if parity.get("verdict") != "PASS" or parity.get("base_sha256") != base["base_sha256"]:
            raise Error("native hybrid parity must pass on this exact base")
        verified["parity"] = parity
        batches_path = pinned_file(config["batches_manifest"], root)
        batches = read_json(batches_path)
        if batches.get("schema") != "chimera_recovery.native_batches.v1" or batches.get("sealed") is not True or batches.get("split") != "train" or batches.get("heldout_groups_excluded") is not True:
            raise Error("accepted sealed training batches must exclude heldout source groups")
        if not batches.get("batches"):
            raise Error("no accepted native media batches; cache adapters have not supplied training data")
        if config.get("sampling_policy") != "one_pass" or config["steps"] != len(batches["batches"]):
            raise Error("D3b recovery must pin one_pass and one successful step per accepted native batch")
        if len({item.get("id") for item in batches["batches"]}) != len(batches["batches"]) or any(not isinstance(item.get("id"), str) for item in batches["batches"]):
            raise Error("accepted native batches require unique stable ids")
        for item in batches["batches"]:
            if item.get("unit") != config.get("unit") or R.ROUTES[R.route_name(item.get("route", ""))][0] != "D3b":
                raise Error("one D3b run owns exactly one native unit/lane")
            pinned_file(item["tensors"], batches_path.parent)
            receipt = read_json(pinned_file(item["teacher_receipt"], batches_path.parent))
            R.ReadyObjectiveAdapter(receipt)
            if receipt["cache_sha256"] != item["tensors"]["sha256"] or receipt["corpus_sha256"] != batches.get("corpus_sha256"):
                raise Error("native teacher receipt does not bind this exact tensor shard and admitted corpus")
            if receipt["recovery_amendment_sha256"] != verified["preregistration"]["amendment_sha256"]:
                raise Error("native cache objective amendment differs from this run")
            if receipt["route"] != R.route_name(item["route"]):
                raise Error("teacher cache route differs from accepted batch")
            if item.get("target_layout") != "emission" or item.get("input_layout") != "feature":
                raise Error("accepted batch must state feature inputs and emission targets")
        verified["batches_path"] = str(batches_path)
        verified["batches"] = batches
    except (Error, KeyError, TypeError, ValueError, OSError) as error:
        blockers.append(str(error))
    return {"schema": "chimera_recovery.d3b_preflight.v1", "production_ready": not blockers, "training_started": False,
            "config_sha256": R.sha(config_path), "config": config, "verified": verified, "blockers": blockers,
            "unsupported_stages": ["D3a", "D3c", "D3d"], "driver_sha256": R.sha(Path(__file__)),
            "objective_adapter_sha256": R.sha(HERE / "training.py")}


def configure_trainables(hybrid: Any, unit: str, config: dict[str, Any]):
    import torch
    options = optimizer_config(config)
    placement = next((p for p in hybrid.layout if p.container == unit), None)
    if placement is None:
        raise Error("selected unit absent from native block schedule")
    if hybrid.head_deltas or hybrid.bridge_deltas:
        raise Error("configure once on a fresh hybrid; unexpected interface deltas already present")
    for module in hybrid.units.values():
        module.requires_grad_(False)
    hybrid.unit_parameters([unit])
    deltas = hybrid.add_emission_deltas(list(placement.type_ids))
    for tensor in hybrid.bank.tensors.values():
        tensor.requires_grad_(True)
    hybrid.freeze_lora_outside(range(placement.depth, hybrid.num_layers))
    census, categories = {}, {"unit": [], "emission": [], "lora": []}
    for name, tensor in hybrid.units[unit].named_parameters():
        if tensor.requires_grad:
            census[f"unit.{unit}.{name}"] = tensor
            categories["unit"].append(tensor)
    for delta in deltas:
        census[f"emission.{delta.name}"] = delta.delta
        categories["emission"].append(delta.delta)
    for name in sorted(hybrid.bank.tensors):
        tensor = hybrid.bank.tensors[name]
        if tensor.requires_grad:
            census[f"lora.{name}"] = tensor
            categories["lora"].append(tensor)
    if any(not tensors for tensors in categories.values()):
        raise Error("D3b requires nonempty unit, interface-out, and LoRA trainable groups")
    groups = [{"params": tensors, "lr": options[f"{name}_lr"], "base_lr": options[f"{name}_lr"], "name": name} for name, tensors in categories.items()]
    optimizer = torch.optim.AdamW(groups, betas=tuple(options["betas"]), eps=options["eps"], weight_decay=options["weight_decay"])
    ownership = R.optimizer_ownership(census, optimizer)
    return placement, census, optimizer, ownership


def make_segment(batch: dict[str, Any], *, media_placeholder: int | None = None) -> dict[str, Any]:
    if batch["input_ids"].shape[0] != 1 or not bool(batch["native_media_valid_mask"].all()):
        raise Error("HybridD3 requires one unpadded sample per call")
    if media_placeholder is not None:
        actual = batch["input_ids"].gather(1, batch["native_media_indices"])
        if not bool((actual == media_placeholder).all()):
            raise Error("native media placeholder IDs differ from actual immutable base ledger")
    segment = {"ids": batch["input_ids"], "features": batch["native_media_features"], "types": batch["native_media_type_ids"],
               "positions": batch["native_media_positions"], "indices": batch["native_media_indices"],
               "sigma": batch["native_media_noise_levels"], "media_rows": batch["native_media_type_ids"].shape[1]}
    if "charted_source_positions" in batch:
        segment["source_positions"] = {"h3": batch["charted_source_positions"]}
    return segment


def prune_safety_checkpoints(out: Path, keep: int = 2):
    """Keep the pinned rolling count, checking every resolved deletion target."""
    root = out.resolve()
    candidates = sorted(path for path in root.iterdir() if re.fullmatch(r"checkpoint-safety-[0-9]{8}", path.name))
    for path in candidates[:-keep]:
        resolved = path.resolve()
        if path.is_symlink() or resolved.parent != root or not resolved.is_dir() or not re.fullmatch(r"checkpoint-safety-[0-9]{8}", resolved.name):
            raise Error("refuse recursive removal outside this run's rolling checkpoint directory")
        shutil.rmtree(resolved)


def capture_xd(module, engine, batches, root: Path, out: Path, device: str):
    """Capture exact frozen lower-stack state once into the existing sealed store."""
    import torch
    from safetensors.torch import load_file
    identity = {"base": engine.identity["base"], "batches_manifest_sha256": engine.identity["batches_manifest_sha256"],
                "unit": engine.identity["unit"], "depth": engine.placement.depth,
                "lower_lora_sha256": engine.hybrid.lora_digest(range(engine.placement.depth)),
                "hybrid_module_sha256": engine.identity["hybrid_module_sha256"]}
    writer = module.XdChunkWriter(out, identity=identity, max_chunk_bytes=20 * 10**9, dtype=engine.hybrid.dtype)
    for item in batches:
        batch = load_file(str(pinned_file(item["tensors"], root)), device=device)
        R.validate_native_batch(item["route"], batch)
        segment = make_segment(batch, media_placeholder=engine.hybrid.media_placeholder)
        context = engine.hybrid.prepare(segment)
        with torch.no_grad():
            x_d = engine.hybrid.run_layers(context["x"], 0, engine.placement.depth, context, units_at_stop=False)
        writer.add(item["id"], x_d[0], {"batch_sha256": item["tensors"]["sha256"], "route": item["route"]})
    writer.close()
    return module.XdChunkReader(out, expect_identity=identity, verify=True)


class D3bEngine:
    def __init__(self, module, hybrid, unit: str, optimizer: dict[str, Any], identity: dict[str, Any], *, total_steps: int):
        import torch
        self.module, self.hybrid, self.options, self.identity = module, hybrid, optimizer_config(optimizer), identity
        self.placement, self.census, self.optimizer, self.ownership = configure_trainables(hybrid, unit, optimizer)
        self.step, self.sampler_counter, self.consecutive_skips, self.skip_window, self.total_steps = 0, 0, 0, [], total_steps
        self.decode_review_passed = False
        self.initial = {name: tensor.detach().cpu().clone() for name, tensor in self.census.items()}

    def train_step(self, segment, loss_fn, *, x_d=None):
        import torch
        self.optimizer.zero_grad(set_to_none=True)
        context = self.hybrid.prepare(segment)
        if x_d is None:
            # Exact recapture from this base and frozen lower-layer LoRA; the
            # layers below d remain frozen for the entire one-unit D3b run.
            with torch.no_grad():
                x_d = self.hybrid.run_layers(context["x"], 0, self.placement.depth, context, units_at_stop=False)
        try:
            loss = self.module.d3b_forward_backward(self.hybrid, context, x_d, self.placement.depth, loss_fn)
        except self.module.D3Error as error:
            if "not finite" not in str(error):
                raise
            self.consecutive_skips += 1
            self.skip_window = (self.skip_window + [True])[-100:]
            return {"loss": None, "skipped": True, "halt": self.consecutive_skips >= 8 or (len(self.skip_window) == 100 and sum(self.skip_window) > 20)}
        missing = [name for name, tensor in self.census.items() if tensor.grad is None]
        finite = math.isfinite(loss) and all(tensor.grad is not None and bool(torch.isfinite(tensor.grad).all()) for tensor in self.census.values())
        if missing:
            raise Error(f"declared trainable tensors received no gradient: {missing[:3]}")
        if not finite:
            self.consecutive_skips += 1
            self.skip_window = (self.skip_window + [True])[-100:]
            return {"loss": loss if math.isfinite(loss) else None, "skipped": True, "halt": self.consecutive_skips >= 8 or (len(self.skip_window) == 100 and sum(self.skip_window) > 20)}
        norms = {group["name"]: float(torch.sqrt(sum((tensor.grad.float().square().sum() for tensor in group["params"])))) for group in self.optimizer.param_groups}
        if any(value <= 0 for value in norms.values()):
            raise Error("a D3b trainable group received zero gradient")
        factor = self.module.trainer_module().lr_factor(self.step, self.options["warmup_steps"], self.total_steps, self.options["min_lr_ratio"])
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        total_norm = torch.nn.utils.clip_grad_norm_(list(self.census.values()), self.options["clip"], error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1
        self.sampler_counter += 1
        self.consecutive_skips = 0
        self.skip_window = (self.skip_window + [False])[-100:]
        return {"loss": loss, "skipped": False, "step": self.step, "gradient_norm": float(total_norm), "group_norms": norms}

    def save(self, out: Path):
        import torch
        if out.exists():
            raise Error("checkpoint output exists; preserve exact resume artifact")
        out.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".d3b-checkpoint-", dir=out.parent))
        state = {"schema": SCHEMA, "identity": self.identity, "ownership": self.ownership,
                 "trainables": {name: tensor.detach().cpu() for name, tensor in self.census.items()},
                 "optimizer": self.optimizer.state_dict(), "step": self.step, "sampler_counter": self.sampler_counter,
                 "consecutive_skips": self.consecutive_skips, "skip_window": self.skip_window,
                 "decode_review_passed": self.decode_review_passed,
                 "rng_cpu": torch.get_rng_state(), "rng_python": random.getstate(),
                 "rng_cuda": torch.cuda.get_rng_state_all() if self.hybrid.device.type == "cuda" else []}
        torch.save(state, stage / "state.pt")
        (stage / "receipt.json").write_bytes(R.canonical({"schema": SCHEMA, "identity_sha256": R.digest(self.identity),
            "state_sha256": R.sha(stage / "state.pt"), "step": self.step, "sampler_counter": self.sampler_counter,
            "trainable_census": self.ownership, "complete": True}) + b"\n")
        stage.rename(out)

    def resume(self, checkpoint: Path):
        import torch
        receipt = read_json(checkpoint / "receipt.json")
        if receipt.get("complete") is not True or R.sha(checkpoint / "state.pt") != receipt.get("state_sha256"):
            raise Error("checkpoint incomplete or digest changed")
        state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
        if state["schema"] != SCHEMA or state["identity"] != self.identity or state["ownership"] != self.ownership:
            raise Error("exact resume refused changed base, data, objective/config, or optimizer census")
        if set(state["trainables"]) != set(self.census):
            raise Error("resume trainable inventory differs")
        with torch.no_grad():
            for name, tensor in self.census.items():
                tensor.copy_(state["trainables"][name].to(tensor.device, tensor.dtype))
        self.optimizer.load_state_dict(state["optimizer"])
        self.step, self.sampler_counter = state["step"], state["sampler_counter"]
        self.consecutive_skips, self.skip_window = state["consecutive_skips"], state["skip_window"]
        self.decode_review_passed = state["decode_review_passed"]
        torch.set_rng_state(state["rng_cpu"])
        random.setstate(state["rng_python"])
        if state["rng_cuda"]:
            torch.cuda.set_rng_state_all(state["rng_cuda"])

    def export(self, out: Path):
        from safetensors.torch import save_file
        if out.exists():
            raise Error("delta output exists")
        out.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".d3b-delta-", dir=out.parent))
        tensors = {}
        for name, tensor in self.census.items():
            tensors[name] = (tensor.detach().cpu() - self.initial[name]).contiguous() if name.startswith("unit.") else tensor.detach().cpu().contiguous()
        save_file(tensors, str(stage / "delta_state.safetensors"))
        (stage / "manifest.json").write_bytes(R.canonical({"schema": "chimera_recovery.d3b_delta.v1", "identity": self.identity,
            "sha256": R.sha(stage / "delta_state.safetensors"), "step": self.step, "trainable_census": self.ownership,
            "semantics": {"unit": "additive delta from immutable base", "emission": "scatter additive slice delta using native emission ranges",
                          "lora": "adapter bank replacement; apply A/B product at declared sites"},
            "registered": False, "requires_oc_checkpoint_compose_adapter": True}) + b"\n")
        stage.rename(out)


def cpu_fixture(module_path: Path, out: Path) -> dict[str, Any]:
    """Actual HybridD3 tiny random child, update and exact interrupted resume."""
    import torch
    if torch.cuda.is_initialized():
        raise Error("CPU fixture refuses an initialized CUDA runtime")
    torch.set_num_threads(1)
    module = load_module(module_path, "d3_hybrid_module_recovery_fixture")
    built = module.build_tiny_child(out / "tiny-base", seed=17)
    config = {"unit_lr": 0.001, "emission_lr": 0.001, "lora_lr": 0.001, "betas": [0.9, 0.95], "eps": 1e-8,
              "weight_decay": 0., "clip": 1., "warmup_steps": 0, "min_lr_ratio": 1.}
    # This random tiny fixture explicitly defines its last six ACE bridge
    # channels as noisy latent; it does not assert native ACE's 384-wide recipe.
    fixture_noise = {4: (6, 12)}
    identity = {"fixture": True, "seed": 17, "objective": "guided_velocity", "module_sha256": R.sha(module_path), "optimizer": config,
                "fixture_noisy_feature_bands": fixture_noise}
    def construct():
        plan = module.trainer_module().SitePlanConfig(rank=4, alpha=4., k3_rank=0, trunk_ffn_width=module.TINY.TRUNK, gain=False)
        hybrid = module.HybridD3(built["dir"], built["parent_config"], module.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch",
                                lora=plan, lora_seed=3, pin_boundaries=False, prefetch=False))
        return D3bEngine(module, hybrid, hybrid.layout[0].container, config, identity, total_steps=3)
    full = construct()
    segment = module.tiny_segment(full.hybrid.cfg, "ace", rows=4, seed=18)
    generator = torch.Generator().manual_seed(19)
    target = torch.randn(1, 4, module.TINY.FEATURE, generator=generator)
    def objective(engine):
        return lambda prediction: R.flow_velocity_loss(prediction, target, segment["types"], torch.ones(1, 4), engine.hybrid.bands,
                                                        param="x0", x_t=segment["features"], sigma=segment["sigma"],
                                                        noisy_feature_bands=fixture_noise)
    full.save(out / "initial")
    first = full.train_step(segment, objective(full))
    full.save(out / "interrupted")
    second = full.train_step(segment, objective(full))
    third = full.train_step(segment, objective(full))
    resumed = construct()
    resumed.resume(out / "interrupted")
    second_resumed = resumed.train_step(segment, objective(resumed))
    third_resumed = resumed.train_step(segment, objective(resumed))
    exact = all(torch.equal(full.census[name], resumed.census[name]) for name in full.census)
    changed = any(not torch.equal(tensor.detach().cpu(), full.initial[name]) for name, tensor in full.census.items())
    bad = construct()
    bad.identity = {**identity, "objective": "wrong"}
    refused = False
    try:
        bad.resume(out / "interrupted")
    except Error:
        refused = True
    full.export(out / "delta")
    report = {"schema": "chimera_recovery.d3b_cpu_fixture.v1", "production_model_weights_loaded": False,
              "random_tiny_child": True, "actual_hybrid_forward_backward": True, "optimizer_updates": full.step,
              "fixture_noisy_feature_bands": fixture_noise,
              "updated_parameters": changed, "resume_tensor_exact": exact, "changed_identity_refused": refused,
              "uninterrupted_losses": [first["loss"], second["loss"], third["loss"]],
              "resumed_losses": [second_resumed["loss"], third_resumed["loss"]], "ownership": full.ownership,
              "production_ready": False, "passed": exact and changed and refused and second == second_resumed and third == third_resumed}
    (out / "cpu_fixture.json").write_bytes(R.canonical(report) + b"\n")
    return report


def train(config_path: Path, out: Path, resume: Path | None = None) -> dict[str, Any]:
    """Production entry point; all external prerequisites must pass before load."""
    import torch
    from safetensors.torch import load_file
    check = preflight(config_path)
    if not check["production_ready"]:
        raise Error("D3b preflight blocked: " + "; ".join(check["blockers"]))
    if out.exists():
        raise Error("run output exists; use a new run directory and explicit --resume")
    out.mkdir(parents=True)
    (out / "preflight.json").write_bytes(R.canonical(check) + b"\n")
    config, verified = check["config"], check["verified"]
    module = load_module(Path(verified["hybrid_module"]), "d3_hybrid_module_recovery_production")
    plan = module.trainer_module().SitePlanConfig(rank=16, alpha=16., k3_rank=0, trunk_ffn_width=config["lora"]["trunk_ffn_width"], gain=False)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    hybrid = module.HybridD3(config["omni"], config["parent_config"], module.HybridOptions(dtype=getattr(torch, config["dtype"]),
                            device=config["device"], kernels=config["kernels"], lora=plan, lora_seed=config["seed"], plan=config.get("compose_plan")))
    identity = {"config_sha256": check["config_sha256"], "base": verified["base"], "batches_manifest_sha256": config["batches_manifest"]["sha256"],
                "hybrid": hybrid.identity(), "preregistration": verified["preregistration"], "driver_sha256": check["driver_sha256"],
                "objective_adapter_sha256": check["objective_adapter_sha256"], "optimizer": config["optimizer"], "unit": config["unit"],
                "hybrid_module_sha256": config["hybrid_module"]["sha256"]}
    engine = D3bEngine(module, hybrid, config["unit"], config["optimizer"], identity, total_steps=config["steps"])
    if resume:
        engine.resume(resume)
    engine.save(out / "checkpoint-initial")
    batches = verified["batches"]["batches"]
    root = Path(verified["batches_path"]).parent
    # Recapture from this exact base; the run stores/reuses x_d on disk instead
    # of retaining every sample in RAM or trusting stale S3/S5 prefix captures.
    xd = capture_xd(module, engine, batches, root, out / "xd", config["device"])
    last_safety = time.monotonic()
    while engine.step < config["steps"]:
        quarter = max(1, math.ceil(config["steps"] * .25))
        if engine.step >= quarter and not engine.decode_review_passed:
            review_path = config.get("decode_review_receipt")
            review = read_json(Path(review_path)) if review_path and Path(review_path).is_file() else {}
            if (review.get("meaningful_prompt_conditioned_content") is not True or
                    review.get("run_identity_sha256") != R.digest(identity) or
                    not quarter <= review.get("trained_step", -1) <= engine.step):
                engine.save(out / f"checkpoint-decode-review-{engine.step:08d}")
                return {"schema": SCHEMA, "status": "awaiting_decode_review", "step": engine.step, "training_complete": False}
            engine.decode_review_passed = True
        if engine.sampler_counter >= len(batches):
            raise Error("one-pass sampler exhausted before declared step budget")
        item = batches[engine.sampler_counter]
        batch = load_file(str(pinned_file(item["tensors"], root)), device=config["device"])
        receipt = read_json(pinned_file(item["teacher_receipt"], root))
        compute = R.ReadyObjectiveAdapter(receipt).bind(item["route"], batch, hybrid.bands,
                                                    parent_signs={int(k): v for k, v in item.get("parent_signs", {}).items()})
        segment = make_segment(batch, media_placeholder=hybrid.media_placeholder)
        try:
            result = engine.train_step(segment, compute, x_d=xd.get(item["id"]).unsqueeze(0))
        except Exception:
            engine.save(out / f"checkpoint-failure-{engine.step:08d}")
            raise
        with (out / "steps.jsonl").open("ab") as stream:
            stream.write(R.canonical(result) + b"\n")
        if result.get("halt"):
            engine.save(out / f"checkpoint-skip-halt-{engine.step:08d}")
            raise Error("persistent non-finite skips require a ruling")
        if time.monotonic() - last_safety >= 1800:
            engine.save(out / f"checkpoint-safety-{engine.step:08d}")
            prune_safety_checkpoints(out, keep=2)
            last_safety = time.monotonic()
    engine.save(out / "checkpoint-final")
    engine.export(out / "delta")
    return {"schema": SCHEMA, "status": "complete", "step": engine.step, "training_complete": True, "registered": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("preflight")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    f = commands.add_parser("cpu-fixture")
    f.add_argument("--hybrid-module", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    t = commands.add_parser("train")
    t.add_argument("--config", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.config)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(R.canonical(result) + b"\n")
        print(json.dumps({"production_ready": result["production_ready"], "blockers": result["blockers"]}))
        return 0 if result["production_ready"] else 3
    if args.command == "cpu-fixture":
        if args.out.exists():
            raise Error("fixture output must be new")
        args.out.mkdir(parents=True)
        result = cpu_fixture(args.hybrid_module, args.out)
        print(json.dumps({"passed": result["passed"], "resume_tensor_exact": result["resume_tensor_exact"], "production_ready": False}))
        return 0 if result["passed"] else 3
    result = train(args.config, args.out, args.resume)
    (args.out / "run_result.json").write_bytes(R.canonical(result) + b"\n")
    print(json.dumps(result))
    return 0 if result["training_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
