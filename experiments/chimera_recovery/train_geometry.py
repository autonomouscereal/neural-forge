"""Distinct full-depth source-authenticated supervised geometry stage.

This candidate never allocates a runtime, generates targets, changes production
weights or accepts its outputs. Canonical sigma and native progress stay separate;
GeometryFlowAdapter reconstructs genuine pixels/targets before every backward.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import time

from . import train_media as T, consolidation_objectives as O

I, N, Error = T.I, T.N, T.Error
STAGE = "D3b_geometry_supervised_flow"
OBJECTIVE = O.GEOMETRY
SCHEMA = "chimera_recovery.geometry_stage_config.v1"
OPTIMIZER = dict(unit_lr=2e-5, bridge_lr=2e-5, emission_lr=2e-5, lora_lr=1e-4,
                 betas=[.9, .95], eps=1e-8, weight_decay=0., clip=1., warmup_steps=20, min_lr_ratio=.1)


def geometry_module():
    from . import geometry_flow
    return geometry_flow


def optimizer_options(value):
    options = dict(value)
    bridge = options.pop("bridge_lr", None)
    if type(bridge) not in (float, int) or not math.isfinite(bridge) or bridge <= 0:
        raise Error("geometry optimizer needs an explicit positive input bridge rate")
    T.optimizer_config(options)
    return dict(value)


def cache_manifest(path, amendment_sha):
    """Metadata/source proof before opening a model; tensor proof remains per call."""
    G = geometry_module()
    path = Path(path)
    data = T.read_json(path)
    if (data.get("schema") != G.SCHEMA or data.get("stage") != STAGE or data.get("objective") != OBJECTIVE
            or any(data.get(k) is not True for k in ("sealed", "complete_source_traversal", "heldout_groups_excluded", "training_admission_selected"))
            or data.get("amendment", {}).get("sha256") != amendment_sha or data.get("storage") != G.COMPACT
            or data.get("adapter_sha256") != I.sha(Path(G.__file__)) or data.get("dependencies") != G.dependency_pins()
            or data.get("sigmas") != G.SIGMAS or data.get("policy") != G.POLICY
            or data.get("manifest_sha256") != I.digest({k: v for k, v in data.items() if k != "manifest_sha256"})):
        raise Error("geometry requires the complete selected source-authenticated compact cache")
    seen, pairs, selected = set(), set(), set()
    for item in data.get("batches", []):
        if not item.get("id") or item["id"] in seen or not item.get("recovery_group_id"):
            raise Error("geometry source batch identity/group missing or duplicated")
        seen.add(item["id"])
        I.pinned(item["tensors"], path.parent)
        receipt_path = I.pinned(item["receipt"], path.parent)
        adapter = G.GeometryFlowAdapter(receipt_path, item["receipt"]["sha256"])
        r = adapter.receipt
        if (r["id"] != item["id"] or r["source_id"] != item["source_id"]
                or r["recovery_group_id"] != item["recovery_group_id"]
                or r["tensors"]["sha256"] != item["tensors"]["sha256"]
                or r["amendment"] != data["amendment"] or r["selection"] != data["selection"]):
            raise Error("geometry selected manifest membership differs from genuine receipt")
        pair = r["source_id"], r["sigma"]
        if pair in pairs:
            raise Error("geometry source/sigma replay cannot be repeated")
        pairs.add(pair)
        selected.add(r["source_id"])
    if not seen or pairs != {(source, sigma) for source in selected for sigma in G.SIGMAS}:
        raise Error("geometry must use every selected genuine source/sigma pair exactly once")
    _, _, inventory = G.inventory(Path(data["intersection"]["path"]), data["intersection"]["sha256"])
    _, ids = G.selection(I.pinned(data["selection"], path.parent), amendment_sha, data["intersection"]["sha256"], inventory)
    if selected != set(ids):
        raise Error("geometry cache omitted a selected eligible genuine source")
    return data


class GeometryFlowEngine(T.D3bEngine):
    """One fresh Adam owner; complete r16/native handoff precedes construction."""
    checkpoint_schema = "chimera_recovery.geometry_stage_checkpoint.v1"

    def __init__(self, module, hybrid, unit, optimizer, identity, *, total_steps):
        import torch
        self.module, self.hybrid, self.options = module, hybrid, optimizer_options(optimizer)
        self.identity = {**identity, "stage": STAGE, "objective": OBJECTIVE,
                         "optimizer_sha256": I.digest(optimizer), "total_steps": total_steps}
        placements = [p for p in hybrid.layout if p.spec.source == "hy3d" and 7 in p.type_ids]
        self.placement = placements[0] if len(placements) == 1 and placements[0].container == unit else None
        if self.placement is None or hybrid.head_deltas or hybrid.bridge_deltas or type(total_steps) is not int or total_steps < 1:
            raise Error("geometry needs a fresh genuine HY3D unit and exact positive update budget")
        if (hybrid.bands[7]["feature"] != (576, 640) or hybrid.bands[7]["emission"] != (320, 384)):
            raise Error("geometry exact native type7 feature/emission allocation differs")
        if identity.get("fixture") is not True:
            if self.options != OPTIMIZER:
                raise Error("geometry initial rates/options must equal the explicitly selected dated policy")
            for key in ("base_sha256", "batch_manifest_sha256", "amendment_sha256", "hybrid_module_sha256", "config_sha256"):
                if not I.valid_sha(identity.get(key)):
                    raise Error("geometry production identity missing: " + key)
            if identity.get("heldout_groups_excluded") is not True or hybrid.base_vocab != 248320:
                raise Error("geometry production needs exact native vocabulary/source exclusion")
        for native in hybrid.units.values():
            native.requires_grad_(False)
        hybrid.unit_parameters([unit])
        bridges, emissions = hybrid.add_bridge_deltas([7]), hybrid.add_emission_deltas([7])
        for value in hybrid.bank.tensors.values():
            value.requires_grad_(True)
        self.census = {"unit." + unit + "." + name: value for name, value in hybrid.units[unit].named_parameters() if value.requires_grad}
        self.census.update({"bridge." + d.name: d.delta for d in bridges})
        self.census.update({"emission." + d.name: d.delta for d in emissions})
        self.census.update({"lora." + name: value for name, value in sorted(hybrid.bank.tensors.items()) if value.requires_grad})
        groups = [dict(name=kind, params=[v for k, v in self.census.items() if k.startswith(kind + ".")],
                       lr=self.options[kind + "_lr"], base_lr=self.options[kind + "_lr"])
                  for kind in ("unit", "bridge", "emission", "lora")]
        if any(not g["params"] for g in groups):
            raise Error("geometry ownership requires HY3D, type7 bridge/emission and shared r16")
        self.optimizer = torch.optim.AdamW(groups, betas=tuple(self.options["betas"]), eps=self.options["eps"], weight_decay=self.options["weight_decay"])
        self.ownership = T.R.optimizer_ownership(self.census, self.optimizer)
        self.initial = {name: value.detach().cpu().clone() for name, value in self.census.items()}
        self.step = self.sampler_counter = self.consecutive_skips = 0
        self.skip_window, self.total_steps, self.decode_review_passed = [], total_steps, False

    def train_step(self, batch, compute, *, gate):
        import torch
        if self.step >= self.total_steps or not callable(gate):
            raise Error("geometry needs remaining one-pass budget and live root optimizer authority")
        rng = torch.get_rng_state(), random.getstate(), torch.cuda.get_rng_state_all() if self.hybrid.device.type == "cuda" else []
        def rollback():
            self.optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(rng[0]); random.setstate(rng[1])
            if rng[2]:
                torch.cuda.set_rng_state_all(rng[2])
        self.optimizer.zero_grad(set_to_none=True)
        try:
            mask = batch["native_media_loss_weights"]
            units = O.scalar_flow_count(batch["native_media_type_ids"], mask, {7: self.hybrid.bands[7]})
            loss = O.geometry_backward(self.module, self.hybrid, batch, compute, units, 1.)
        except self.module.D3Error as exc:
            if "not finite" not in str(exc):
                rollback(); raise
            loss = float("nan")
        except BaseException:
            rollback(); raise
        missing = [name for name, v in self.census.items() if v.grad is None]
        if math.isfinite(loss) and missing:
            rollback()
            raise Error("geometry declared owner received no gradient: " + repr(missing[:3]))
        if not math.isfinite(loss) or any(v.grad is None or not bool(torch.isfinite(v.grad).all()) for v in self.census.values()):
            rollback()
            self.consecutive_skips += 1
            self.skip_window = (self.skip_window + [True])[-100:]
            return dict(loss=None, skipped=True, step=self.step, halt=self.consecutive_skips >= 8 or (len(self.skip_window) == 100 and sum(self.skip_window) > 20))
        norms = {g["name"]: float(torch.sqrt(sum(v.grad.float().square().sum() for v in g["params"]))) for g in self.optimizer.param_groups}
        if any(v <= 0 for v in norms.values()):
            rollback()
            raise Error("geometry declared optimizer family received zero gradient")
        factor = self.module.trainer_module().lr_factor(self.step, self.options["warmup_steps"], self.total_steps, self.options["min_lr_ratio"])
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        try:
            norm = torch.nn.utils.clip_grad_norm_(list(self.census.values()), self.options["clip"], error_if_nonfinite=True)
            gate()  # immediately before the only physical Adam update
        except BaseException:
            rollback(); raise
        self.optimizer.step()
        self.step += 1
        self.sampler_counter += 1
        self.consecutive_skips, self.skip_window = 0, (self.skip_window + [False])[-100:]
        return dict(loss=loss, skipped=False, step=self.step, sampler_counter=self.sampler_counter, supervised_units=units,
                    group_norms=norms, gradient_norm=float(norm), objective=OBJECTIVE, guided_teacher=False)

    def resume(self, checkpoint):
        super().resume(checkpoint)
        if self.step != self.sampler_counter or not 0 <= self.step <= self.total_steps:
            raise Error("geometry resumed applied step/one-pass source cursor differs")
        expected = set(self.census)
        state = self.optimizer.state
        if self.step and (len(state) != len(expected) or any(int(v.get("step", -1)) != self.step for v in state.values())):
            raise Error("geometry resumed physical Adam step ownership differs")

    def export(self, out):
        raise Error("geometry exports authenticated complete shared-r16/native state, never a generic guided slice")


def preflight(config_path):
    from . import consolidation_dispatch as D
    path = Path(config_path)
    config, root = T.read_json(path), path.parent
    checked, blockers = {}, []
    try:
        if config.get("schema") != SCHEMA or config.get("stage") != STAGE or config.get("objective") != OBJECTIVE:
            raise Error("explicit distinct geometry stage/config/objective required")
        checked["preregistration"] = T.verify_preregistration(config["preregistration"], root)
        checked["hybrid"] = str(I.pinned(config["hybrid_module"], root))
        checked["base"] = D.read(config["immutable_base_receipt"], root)[0]
        base = checked["base"]
        if (any(base.get(k) is not True for k in ("immutable_clone", "registered", "verified_equal_to_registered_parent"))
                or base.get("base_sha256") != base.get("clone_sha256") or base.get("clone_path") != config["omni"]):
            raise Error("geometry needs exact immutable registered D2 clone")
        checked["shared_r16"] = I.preflight_config(config, root, base["base_sha256"])
        checked["native_state"] = N.preflight_config(config, root, checked["shared_r16"])
        accepted, accepted_path = D.read(config["parent_acceptance"], root)
        producer = D.read(config["native_state"]["source"], root)[0]
        if (accepted.get("schema") != "chimera_recovery.accepted_stage.v1" or accepted.get("accepted") is not True
                or accepted.get("stage") != "D3b.qimg" or producer["producer"]["checkpoint_state"] != accepted["checkpoint_state"]
                or producer["producer"]["checkpoint_receipt"] != accepted["checkpoint_receipt"]):
            raise Error("geometry must inherit the actual accepted completed qimg predecessor")
        for field in ("full_answers", "outcomes", "evaluation_protocol"):
            I.pinned(accepted[field], accepted_path.parent)
        checked["cache_path"] = str(I.pinned(config["batches_manifest"], root))
        checked["cache"] = cache_manifest(checked["cache_path"], config["geometry_amendment"]["sha256"])
        if checked["cache"]["amendment"] != config["geometry_amendment"]:
            raise Error("geometry cache belongs to a different selected amendment")
        options = optimizer_options(config["optimizer"])
        if options != OPTIMIZER:
            raise Error("geometry selected initial policy requires native2e-5/shared1e-4 and exact optimizer settings")
        optimizer_selection = D.read(config["geometry_amendment"], root)[0].get("optimizer_scope", {})
        if (optimizer_selection.get("optimizer") != options
                or optimizer_selection.get("ownership") != ["HY3D_unit", "type7_bridge", "type7_emission", "shared_r16"]):
            raise Error("geometry optimizer/site rates need exact dated owner selection")
        if (config.get("steps") != len(checked["cache"]["batches"]) or config.get("sampling_policy") != "one_pass"
                or config.get("safety") != D.SAFETY or type(config.get("seed")) is not int
                or config.get("dtype") not in {"bfloat16", "float32"}
                or config.get("kernels") not in {"torch", "triton"} or not str(config.get("device", "")).startswith("cuda:")
                or config.get("lora", {}) != dict(rank=16, alpha=16, k3_rank=0, gain=False, trunk_ffn_width=config["lora"]["trunk_ffn_width"])):
            raise Error("geometry exact one-pass budget/shared-r16/frozen safety required")
        parity = D.read(config["parity_receipt"], root)[0]
        if (parity.get("verdict") != "PASS" or parity.get("base_sha256") != base["base_sha256"]
                or parity.get("geometry_full_depth_forward_backward") is not True
                or parity.get("raw_sign_and_native_progress_verified") is not True):
            raise Error("geometry needs actual current-base full-depth objective/native bridge parity")
        checked["runtime"] = D.read(config["runtime_assignment"], root)[0]
        for key in ("helpers",):
            for ref in config[key].values():
                I.pinned(ref, root)
        required = [Path(__file__), Path(O.__file__), Path(T.__file__), Path(I.__file__), Path(N.__file__),
                    Path(D.__file__), Path(geometry_module().__file__)]
        if any(config["helpers"].get(p.name, {}).get("sha256") != I.sha(p) for p in required):
            raise Error("geometry running driver/objective/loader/importer/helper differs from exact source pin")
    except (KeyError, ValueError, TypeError, OSError, ImportError) as exc:
        blockers.append(str(exc))
    return dict(schema=SCHEMA, candidate_preflight_passed=not blockers, production_ready=False,
                training_started=False, blockers=blockers, config=config, verified=checked, config_sha256=I.sha(path))


def load_row(item, root):
    from safetensors.torch import load_file
    G = geometry_module()
    cpu = G.inflate(load_file(str(I.pinned(item["tensors"], root)), device="cpu"))
    compute = G.GeometryFlowAdapter(I.pinned(item["receipt"], root), item["receipt"]["sha256"]).bind(cpu)
    return cpu, compute


def train(config_path, out, resume=None):
    """Explicit runtime-bound native stage; imports no teacher or historical command."""
    from . import consolidation_dispatch as D
    check = preflight(config_path)
    if not check["candidate_preflight_passed"]:
        raise Error("geometry candidate preflight blocked: " + "; ".join(check["blockers"]))
    out, root = Path(out), Path(config_path).parent
    if out.exists():
        raise Error("preserve geometry run; exact resume requires fresh output namespace")
    import torch
    import sys
    config, verified = check["config"], check["verified"]
    def gate():
        for ref in config["helpers"].values():
            I.pinned(ref, root)
        assignment = D.read(config["runtime_assignment"], root)[0]
        proof = T.read_json(Path(config["runtime_proof_path"]))
        return D.runtime_gate(assignment, proof, D.job_digest(config))
    gate()
    assignment = verified["runtime"]
    if (sys.executable != assignment["image_python"] or str(torch.__version__) != assignment["torch_version"]
            or not torch.cuda.is_available() or torch.cuda.device_count() != len(assignment["gpu_uuids"])
            or [D.canonical_gpu_uuid(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())] != assignment["gpu_uuids"]):
        raise Error("geometry actual allocated image/interpreter/CUDA membership differs")
    mountinfo = Path("/proc/self/mountinfo").read_text()
    mounts = {line.split()[4].replace("\\040", " "): "ro" in line.split()[5].split(",")
              for line in mountinfo.splitlines() if len(line.split()) >= 10 and "-" in line.split()}
    if any(mounts.get(item["target"]) is not item["read_only"] for item in assignment["mounts"]):
        raise Error("geometry in-container readonly source access differs from actual root attestation")
    module = T.load_module(Path(verified["hybrid"]), "chimera_geometry_full_native")
    torch.manual_seed(config["seed"]); random.seed(config["seed"])
    plan = module.trainer_module().SitePlanConfig(rank=16, alpha=16., k3_rank=0, gain=False, trunk_ffn_width=config["lora"]["trunk_ffn_width"])
    hybrid = module.HybridD3(config["omni"], config["parent_config"], module.HybridOptions(dtype=getattr(torch, config["dtype"]),
        device=config["device"], kernels=config["kernels"], lora=plan, lora_seed=config["seed"], plan=config.get("compose_plan")))
    shared = I.initialize_stage(hybrid, verified["shared_r16"], root)
    native = N.initialize_stage(hybrid, verified["native_state"], root)
    identity = dict(stage=STAGE, objective=OBJECTIVE, config_sha256=check["config_sha256"], base_sha256=verified["base"]["base_sha256"],
        batch_manifest_sha256=config["batches_manifest"]["sha256"], amendment_sha256=verified["preregistration"]["amendment_sha256"],
        geometry_amendment=config["geometry_amendment"], hybrid_module_sha256=config["hybrid_module"]["sha256"],
        helpers=config["helpers"], heldout_groups_excluded=True, shared_r16=shared, native_state=native)
    engine = GeometryFlowEngine(module, hybrid, config["unit"], config["optimizer"], identity, total_steps=config["steps"])
    return execute(engine, verified["cache"]["batches"], Path(verified["cache_path"]).parent, out, gate=gate,
                   review_pointer=config.get("decode_review_pointer"), resume=resume, context=verified["shared_r16"]["context"])


def execute(engine, items, root, out, *, gate, review_pointer=None, resume=None, context=None, export=True, clock=time.monotonic):
    """Exact applied-row cursor, rollback, persistent skip guard and quarter review."""
    from . import consolidation_dispatch as D
    out = Path(out)
    if out.exists() or len(items) != engine.total_steps:
        raise Error("geometry fresh namespace/exact complete source row count required")
    policy = dict(consecutive_skips=0, window=[], decode_review=None, attempts=0)
    if resume:
        engine.resume(Path(resume))
        policy = restored_policy(engine, Path(resume))
    if policy["consecutive_skips"] >= 8 or (len(policy["window"]) == 100 and sum(policy["window"]) > 20):
        raise Error("geometry exact resume cannot erase persistent skip halt")
    if policy["decode_review"]:
        review, path = D.read(policy["decode_review"], Path(resume))
        if review.get("run_identity_sha256") != I.digest(engine.identity) or review.get("verdict") != "PASS":
            raise Error("geometry retained review changed")
        for field in ("full_answers", "outcomes", "evaluation_protocol", "baseline_answers"):
            I.pinned(review[field], path.parent)
    out.mkdir(parents=True)
    save(engine, out / "checkpoint-initial", policy)
    last_safety = clock()
    while True:
        if engine.step >= max(1, math.ceil(engine.total_steps * .25)) and not policy["decode_review"]:
            checkpoint = out / f"checkpoint-review-{engine.step:08d}"
            save(engine, checkpoint, policy)
            if not review_pointer or not Path(review_pointer).is_file():
                return dict(stage=STAGE, status="awaiting_decode_review", step=engine.step, checkpoint=str(checkpoint), training_complete=False, production_ready=False)
            pointer = T.read_json(Path(review_pointer))
            expected = Path(resume) if resume and engine.step == T.read_json(Path(resume) / "receipt.json")["step"] else checkpoint
            policy["decode_review"] = D.decode_review(pointer, engine, out, expected)
        if engine.step >= engine.total_steps:
            break
        item = items[engine.sampler_counter]
        try:
            batch, compute = load_row(item, root)
            batch = {key: value.to(engine.hybrid.device) for key, value in batch.items()}
            step = engine.train_step(batch, compute, gate=gate)
        except BaseException:
            save(engine, out / f"checkpoint-failure-{engine.step:08d}", policy)
            raise
        skipped = step.get("skipped") is True
        policy["attempts"] += 1
        policy["consecutive_skips"] = policy["consecutive_skips"] + 1 if skipped else 0
        policy["window"] = (policy["window"] + [skipped])[-100:]
        with (out / "steps.jsonl").open("ab") as stream:
            stream.write(I.canonical(dict(attempt=policy["attempts"], **step)) + b"\n")
        if step.get("halt"):
            save(engine, out / f"checkpoint-skip-halt-{engine.step:08d}", policy)
            raise Error("geometry persistent nonfinite skips halted without consuming source pairs")
        if clock() - last_safety >= 1800:
            save(engine, out / f"checkpoint-safety-{policy['attempts']:08d}", policy)
            T.prune_safety_checkpoints(out, keep=2)
            last_safety = clock()
    checkpoint = out / "checkpoint-final"
    save(engine, checkpoint, policy)
    result = dict(stage=STAGE, status="complete", step=engine.step, training_complete=True, accepted=False,
                  registered=False, production_ready=False, checkpoint=str(checkpoint), guided_teacher=False)
    if export:
        shared = I.export_from_engine(engine, checkpoint, out / "shared-r16", context, STAGE, engine.placement.container)
        result["shared_r16"] = shared
        result["native_state"] = N.export_from_engine(engine, checkpoint, out / "native-state", context, STAGE, shared)
    return result


def save(engine, path, policy):
    from . import consolidation_dispatch as D
    path = Path(path)
    engine.save(path)
    (path / "dispatch_policy.json").write_bytes(I.canonical(D.policy_record(policy)) + b"\n")
    (path / "geometry_dispatch_receipt.json").write_bytes(I.canonical(dict(schema="chimera_recovery.geometry_dispatch_checkpoint.v1",
        identity_sha256=I.digest(engine.identity), engine_receipt=I.pin(path / "receipt.json"), policy=I.pin(path / "dispatch_policy.json"),
        step=engine.step, cursor=engine.sampler_counter)) + b"\n")


def restored_policy(engine, path):
    from . import consolidation_dispatch as D
    value = T.read_json(path / "geometry_dispatch_receipt.json")
    if (value.get("schema") != "chimera_recovery.geometry_dispatch_checkpoint.v1" or value.get("identity_sha256") != I.digest(engine.identity)
            or value.get("step") != engine.step or value.get("cursor") != engine.sampler_counter
            or I.pinned(value["engine_receipt"], path) != (path / "receipt.json").resolve()):
        raise Error("geometry checkpoint/policy/one-pass cursor differs")
    return D.policy_record(T.read_json(I.pinned(value["policy"], path)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train"):
        command = sub.add_parser(name, allow_abbrev=False)
        command.add_argument("--config", type=Path, required=True)
        if name == "train":
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    result = preflight(args.config) if args.command == "preflight" else train(args.config, args.out, args.resume)
    print(json.dumps({k: result[k] for k in ("candidate_preflight_passed", "blockers", "status", "production_ready", "training_complete") if k in result}))
    return 0 if result.get("candidate_preflight_passed", result.get("training_complete", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
