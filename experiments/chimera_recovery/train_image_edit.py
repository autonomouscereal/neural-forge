"""Distinct full-depth source-authenticated supervised image_edit stage.

This candidate never allocates a runtime, generates targets, changes production
weights or accepts its outputs. The genuine before image conditions output rows;
ImageEditAdapter reconstructs genuine input/target/tokenizer proofs before backward.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import time

from . import train_media as T, image_edit as E

I, N, Error = T.I, T.N, T.Error
STAGE = E.STAGE
OBJECTIVE = E.OBJECTIVE
SCHEMA = "chimera_recovery.image_edit_stage_config.v1"
OPTIMIZER = dict(unit_lr=2e-5, bridge_lr=2e-5, emission_lr=2e-5, lora_lr=1e-4,
                 betas=[.9, .95], eps=1e-8, weight_decay=0., clip=1., warmup_steps=20, min_lr_ratio=.1)


def image_edit_module():
    return E


def optimizer_options(value):
    options = dict(value)
    bridge = options.pop("bridge_lr", None)
    if type(bridge) not in (float, int) or not math.isfinite(bridge) or bridge <= 0:
        raise Error("image_edit optimizer needs an explicit positive input bridge rate")
    T.optimizer_config(options)
    return dict(value)


def cache_manifest(path, amendment_sha, *, aliases=None):
    """Bind all actual source/sigma pairs before model or optimizer allocation."""
    path=Path(path);data=T.read_json(path)
    if (data.get('schema')!=E.SCHEMA or data.get('stage')!=STAGE or data.get('objective')!=OBJECTIVE or
            data.get('sealed') is not True or data.get('complete_source_traversal') is not True or
            data.get('amendment_sha256')!=amendment_sha or data.get('adapter_sha256')!=I.sha(Path(E.__file__)) or
            data.get('dependencies')!=E.dependency_pins() or data.get('sigma_grid')!=E.SIGMAS or
            data.get('guided_teacher') is not False or data.get('native_parent_full_parity') is not False):
        raise Error('image editing needs the complete selected source-authenticated supervised cache')
    seen,pairs,selected=set(),set(),set()
    for item in data.get('batches',[]):
        if not item.get('id') or item['id'] in seen or not item.get('recovery_group_id'):
            raise Error('editing batch source/group identity missing or duplicated')
        seen.add(item['id']);I.pinned(item['tensors'],path.parent)
        r=T.read_json(I.pinned(item['teacher_receipt'],path.parent))
        if (r['source_id']!=item['source_id'] or r['recovery_group_id']!=item['recovery_group_id'] or
                r['tensors_sha256']!=item['tensors']['sha256'] or r['storage']!=item['storage'] or
                r['amendment_sha256']!=amendment_sha or r['source_admission']!=data['source_admission'] or
                r['vae_proof']!=data['vae_proof'] or r['tokenizer_files']!=data['tokenizer_files']):
            raise Error('editing manifest membership differs from its authentic source receipt')
        pair=r['source_id'],r['sigma']
        if pair in pairs:raise Error('editing cannot repeat an actual source/sigma pair')
        pairs.add(pair);selected.add(r['source_id'])
        # Fail before opening any model when source, target, tokenizer or chart changed.
        load_row(item,path.parent,aliases=aliases)
    admission_path=E.N.pinned(data['source_admission'],path.parent,aliases)
    admission=E.N.read(admission_path)
    records=E.N.rows(E.N.pinned(admission['sources']['source'],admission_path.parent,aliases))
    edit_ids={r['id'] for r in records if r['route']=='image_edit'}
    eligible={r['id'] for r in E.N.rows(E.N.pinned(admission['outputs']['admission'],admission_path.parent,aliases))
              if r['id'] in edit_ids and r['train_admitted'] is True}
    if not seen or selected!=eligible or pairs!={(source,sigma) for source in eligible for sigma in E.SIGMAS}:
        raise Error('editing must cover every eligible original source/sigma pair exactly once')
    return data


class ImageEditEngine(T.D3bEngine):
    """One fresh Adam owner; complete r16/native handoff precedes construction."""
    checkpoint_schema = "chimera_recovery.image_edit_stage_checkpoint.v1"

    def __init__(self, module, hybrid, unit, optimizer, identity, *, total_steps):
        import torch
        self.module, self.hybrid, self.options = module, hybrid, optimizer_options(optimizer)
        self.identity = {**identity, "stage": STAGE, "objective": OBJECTIVE,
                         "optimizer_sha256": I.digest(optimizer), "total_steps": total_steps}
        placements = [p for p in hybrid.layout if p.spec.source == "qimg" and 1 in p.type_ids]
        self.placement = placements[0] if len(placements) == 1 and placements[0].container == unit else None
        if self.placement is None or hybrid.head_deltas or hybrid.bridge_deltas or type(total_steps) is not int or total_steps < 1:
            raise Error("image_edit needs a fresh genuine QIMG unit and exact positive update budget")
        if (hybrid.bands[1]["feature"] != (512, 576) or hybrid.bands[1]["emission"] != (256, 320)):
            raise Error("image_edit exact native type1 feature/emission allocation differs")
        if identity.get("fixture") is not True:
            if self.options != OPTIMIZER:
                raise Error("image_edit initial rates/options must equal the explicitly selected dated policy")
            for key in ("base_sha256", "batch_manifest_sha256", "amendment_sha256", "hybrid_module_sha256", "config_sha256"):
                if not I.valid_sha(identity.get(key)):
                    raise Error("image_edit production identity missing: " + key)
            if identity.get("heldout_groups_excluded") is not True or hybrid.base_vocab != 248320:
                raise Error("image_edit production needs exact native vocabulary/source exclusion")
        for native in hybrid.units.values():
            native.requires_grad_(False)
        hybrid.unit_parameters([unit])
        bridges, emissions = hybrid.add_bridge_deltas([1]), hybrid.add_emission_deltas([1])
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
            raise Error("image_edit ownership requires QIMG, type1 bridge/emission and shared r16")
        self.optimizer = torch.optim.AdamW(groups, betas=tuple(self.options["betas"]), eps=self.options["eps"], weight_decay=self.options["weight_decay"])
        self.ownership = T.R.optimizer_ownership(self.census, self.optimizer)
        self.initial = {name: value.detach().cpu().clone() for name, value in self.census.items()}
        self.step = self.sampler_counter = self.consecutive_skips = 0
        self.skip_window, self.total_steps, self.decode_review_passed = [], total_steps, False

    def train_step(self, batch, compute, *, gate):
        import torch
        if self.step >= self.total_steps or not callable(gate):
            raise Error("image_edit needs remaining one-pass budget and live root optimizer authority")
        rng = torch.get_rng_state(), random.getstate(), torch.cuda.get_rng_state_all() if self.hybrid.device.type == "cuda" else []
        def rollback():
            self.optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(rng[0]); random.setstate(rng[1])
            if rng[2]:
                torch.cuda.set_rng_state_all(rng[2])
        self.optimizer.zero_grad(set_to_none=True)
        try:
            mask = batch["native_media_loss_mask"]
            units = int(mask.sum()) * 64
            loss = E.full_depth_backward(self.module, self.hybrid, batch, compute)
        except self.module.D3Error as exc:
            if "not finite" not in str(exc):
                rollback(); raise
            loss = float("nan")
        except BaseException:
            rollback(); raise
        missing = [name for name, v in self.census.items() if v.grad is None]
        if math.isfinite(loss) and missing:
            rollback()
            raise Error("image_edit declared owner received no gradient: " + repr(missing[:3]))
        if not math.isfinite(loss) or any(v.grad is None or not bool(torch.isfinite(v.grad).all()) for v in self.census.values()):
            rollback()
            self.consecutive_skips += 1
            self.skip_window = (self.skip_window + [True])[-100:]
            return dict(loss=None, skipped=True, step=self.step, halt=self.consecutive_skips >= 8 or (len(self.skip_window) == 100 and sum(self.skip_window) > 20))
        norms = {g["name"]: float(torch.sqrt(sum(v.grad.float().square().sum() for v in g["params"]))) for g in self.optimizer.param_groups}
        if any(v <= 0 for v in norms.values()):
            rollback()
            raise Error("image_edit declared optimizer family received zero gradient")
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
            raise Error("image_edit resumed applied step/one-pass source cursor differs")
        expected = set(self.census)
        state = self.optimizer.state
        if self.step and (len(state) != len(expected) or any(int(v.get("step", -1)) != self.step for v in state.values())):
            raise Error("image_edit resumed physical Adam step ownership differs")

    def export(self, out):
        raise Error("image_edit exports authenticated complete shared-r16/native state, never a generic guided slice")


def preflight(config_path):
    path = Path(config_path)
    config, root = T.read_json(path), path.parent
    checked, blockers = {}, []
    try:
        if config.get("schema") != SCHEMA or config.get("stage") != STAGE or config.get("objective") != OBJECTIVE:
            raise Error("explicit distinct image_edit stage/config/objective required")
        from . import consolidation_dispatch as D
        checked["preregistration"] = T.verify_preregistration(config["preregistration"], root)
        checked["hybrid"] = str(I.pinned(config["hybrid_module"], root))
        checked["base"] = D.read(config["immutable_base_receipt"], root)[0]
        base = checked["base"]
        if (any(base.get(k) is not True for k in ("immutable_clone", "registered", "verified_equal_to_registered_parent"))
                or base.get("base_sha256") != base.get("clone_sha256") or base.get("clone_path") != config["omni"]):
            raise Error("image_edit needs exact immutable registered D2 clone")
        checked["shared_r16"] = I.preflight_config(config, root, base["base_sha256"])
        checked["native_state"] = N.preflight_config(config, root, checked["shared_r16"])
        accepted, accepted_path = D.read(config["parent_acceptance"], root)
        producer = D.read(config["native_state"]["source"], root)[0]
        if (accepted.get("schema") != "chimera_recovery.accepted_stage.v1" or accepted.get("accepted") is not True
                or accepted.get("stage") != "D3b_geometry_supervised_flow" or producer["producer"]["checkpoint_state"] != accepted["checkpoint_state"]
                or producer["producer"]["checkpoint_receipt"] != accepted["checkpoint_receipt"]):
            raise Error("image_edit must inherit the actual accepted completed supervised geometry predecessor")
        for field in ("full_answers", "outcomes", "evaluation_protocol"):
            I.pinned(accepted[field], accepted_path.parent)
        checked["cache_path"] = str(I.pinned(config["batches_manifest"], root))
        checked["cache"] = cache_manifest(checked["cache_path"], config["image_edit_amendment"]["amendment_sha256"], aliases=config.get("source_aliases"))
        if checked["cache"]["amendment_sha256"] != config["image_edit_amendment"]["amendment_sha256"]:
            raise Error("image_edit cache belongs to a different selected amendment")
        options = optimizer_options(config["optimizer"])
        if options != OPTIMIZER:
            raise Error("image_edit selected initial policy requires native2e-5/shared1e-4 and exact optimizer settings")
        selected_amendment=D.read(config["image_edit_amendment"]["selection"],root)[0]
        if (selected_amendment.get("selected") is not True or selected_amendment.get("amendment_sha256") != config["image_edit_amendment"]["amendment_sha256"]):
            raise Error("editing needs the exact owner-selected amendment receipt")
        selected_document=I.pinned(config['image_edit_amendment']['document'],root)
        if (I.sha(selected_document)!=selected_amendment['amendment_sha256'] or
                selected_amendment.get('schema')!='chimera_recovery.image_edit_selection.v1' or
                selected_amendment.get('prior_amendment_exact_prefix_preserved') is not True):
            raise Error("editing exact dated document/preserved predecessor selection differs")
        if (config.get("steps") != len(checked["cache"]["batches"]) or config.get("sampling_policy") != "one_pass"
                or config.get("safety") != D.SAFETY or type(config.get("seed")) is not int
                or config.get("dtype") not in {"bfloat16", "float32"}
                or config.get("kernels") not in {"torch", "triton"} or not str(config.get("device", "")).startswith("cuda:")
                or config.get("lora", {}) != dict(rank=16, alpha=16, k3_rank=0, gain=False, trunk_ffn_width=config["lora"]["trunk_ffn_width"])):
            raise Error("image_edit exact one-pass budget/shared-r16/frozen safety required")
        parity = D.read(config["parity_receipt"], root)[0]
        if (parity.get("verdict") != "PASS" or parity.get("base_sha256") != base["base_sha256"]
                or parity.get("image_edit_full_depth_forward_backward") is not True
                or parity.get("raw_sign_and_clean_reference_policy_verified") is not True):
            raise Error("image_edit needs actual current-base full-depth objective/native bridge parity")
        checked["runtime"] = D.read(config["runtime_assignment"], root)[0]
        for key in ("helpers",):
            for ref in config[key].values():
                I.pinned(ref, root)
        required = [Path(__file__), Path(T.__file__), Path(I.__file__), Path(N.__file__),
                    Path(D.__file__), Path(image_edit_module().__file__)]
        if any(config["helpers"].get(p.name, {}).get("sha256") != I.sha(p) for p in required):
            raise Error("image_edit running driver/objective/loader/importer/helper differs from exact source pin")
    except (KeyError, ValueError, TypeError, OSError, ImportError) as exc:
        blockers.append(str(exc))
    return dict(schema=SCHEMA, candidate_preflight_passed=not blockers, production_ready=False,
                training_started=False, blockers=blockers, config=config, verified=checked, config_sha256=I.sha(path))


def load_row(item,root,*,aliases=None):
    from safetensors.torch import load_file
    cpu=E.M.inflate_native_batch(load_file(str(I.pinned(item['tensors'],root)),device='cpu'),item['storage'])
    receipt=T.read_json(I.pinned(item['teacher_receipt'],root))
    compute=E.ImageEditAdapter(receipt,root,aliases=aliases).bind(cpu)
    return cpu,compute


def train(config_path, out, resume=None):
    """Explicit runtime-bound native stage; imports no teacher or historical command."""
    from . import consolidation_dispatch as D
    check = preflight(config_path)
    if not check["candidate_preflight_passed"]:
        raise Error("image_edit candidate preflight blocked: " + "; ".join(check["blockers"]))
    out, root = Path(out), Path(config_path).parent
    if out.exists():
        raise Error("preserve image_edit run; exact resume requires fresh output namespace")
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
        raise Error("image_edit actual allocated image/interpreter/CUDA membership differs")
    mountinfo = Path("/proc/self/mountinfo").read_text()
    mounts = {line.split()[4].replace("\\040", " "): "ro" in line.split()[5].split(",")
              for line in mountinfo.splitlines() if len(line.split()) >= 10 and "-" in line.split()}
    if any(mounts.get(item["target"]) is not item["read_only"] for item in assignment["mounts"]):
        raise Error("image_edit in-container readonly source access differs from actual root attestation")
    module = T.load_module(Path(verified["hybrid"]), "chimera_image_edit_full_native")
    torch.manual_seed(config["seed"]); random.seed(config["seed"])
    plan = module.trainer_module().SitePlanConfig(rank=16, alpha=16., k3_rank=0, gain=False, trunk_ffn_width=config["lora"]["trunk_ffn_width"])
    hybrid = module.HybridD3(config["omni"], config["parent_config"], module.HybridOptions(dtype=getattr(torch, config["dtype"]),
        device=config["device"], kernels=config["kernels"], lora=plan, lora_seed=config["seed"], plan=config.get("compose_plan")))
    shared = I.initialize_stage(hybrid, verified["shared_r16"], root)
    native = N.initialize_stage(hybrid, verified["native_state"], root)
    identity = dict(stage=STAGE, objective=OBJECTIVE, config_sha256=check["config_sha256"], base_sha256=verified["base"]["base_sha256"],
        batch_manifest_sha256=config["batches_manifest"]["sha256"], amendment_sha256=verified["preregistration"]["amendment_sha256"],
        image_edit_amendment=config["image_edit_amendment"], hybrid_module_sha256=config["hybrid_module"]["sha256"],
        helpers=config["helpers"], heldout_groups_excluded=True, shared_r16=shared, native_state=native)
    engine = ImageEditEngine(module, hybrid, config["unit"], config["optimizer"], identity, total_steps=config["steps"])
    return execute(engine, verified["cache"]["batches"], Path(verified["cache_path"]).parent, out, gate=gate,
                   review_pointer=config.get("decode_review_pointer"), resume=resume, context=verified["shared_r16"]["context"], aliases=config.get("source_aliases"))


def execute(engine, items, root, out, *, gate, review_pointer=None, resume=None, context=None, export=True, clock=time.monotonic, aliases=None):
    """Exact applied-row cursor, rollback, persistent skip guard and quarter review."""
    from . import consolidation_dispatch as D
    out = Path(out)
    if out.exists() or len(items) != engine.total_steps:
        raise Error("image_edit fresh namespace/exact complete source row count required")
    policy = dict(consecutive_skips=0, window=[], decode_review=None, attempts=0)
    if resume:
        engine.resume(Path(resume))
        policy = restored_policy(engine, Path(resume))
    if policy["consecutive_skips"] >= 8 or (len(policy["window"]) == 100 and sum(policy["window"]) > 20):
        raise Error("image_edit exact resume cannot erase persistent skip halt")
    if policy["decode_review"]:
        review, path = D.read(policy["decode_review"], Path(resume))
        if review.get("run_identity_sha256") != I.digest(engine.identity) or review.get("verdict") != "PASS":
            raise Error("image_edit retained review changed")
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
            batch, compute = load_row(item, root, aliases=aliases)
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
            raise Error("image_edit persistent nonfinite skips halted without consuming source pairs")
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
    (path / "image_edit_dispatch_receipt.json").write_bytes(I.canonical(dict(schema="chimera_recovery.image_edit_dispatch_checkpoint.v1",
        identity_sha256=I.digest(engine.identity), engine_receipt=I.pin(path / "receipt.json"), policy=I.pin(path / "dispatch_policy.json"),
        step=engine.step, cursor=engine.sampler_counter)) + b"\n")


def restored_policy(engine, path):
    from . import consolidation_dispatch as D
    value = T.read_json(path / "image_edit_dispatch_receipt.json")
    if (value.get("schema") != "chimera_recovery.image_edit_dispatch_checkpoint.v1" or value.get("identity_sha256") != I.digest(engine.identity)
            or value.get("step") != engine.step or value.get("cursor") != engine.sampler_counter
            or I.pinned(value["engine_receipt"], path) != (path / "receipt.json").resolve()):
        raise Error("image_edit checkpoint/policy/one-pass cursor differs")
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
