"""Exact full-bank D3 r16 handoff; no bank summation or checkpoint composition.

Metadata verification is CPU/stdlib only. Runtime import copies into the existing
physical bank before a stage constructs its census/optimizer. Unit, bridge,
emission and music-table states remain separate compose obligations.
"""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import struct
import tempfile

SCHEMA = "chimera_recovery.shared_r16.v1"


class Error(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def pinned(ref, root):
    if not isinstance(ref, dict) or not valid_sha(ref.get("sha256")) or not ref.get("path"):
        raise Error("shared bank source needs an exact file pin")
    path = Path(ref["path"])
    path = path if path.is_absolute() else Path(root) / path
    if not path.is_file() or sha(path) != ref["sha256"]:
        raise Error("shared bank source file missing or changed")
    return path


def stage_context(config, base_sha256, root=Path(".")):
    path = Path(config["parent_config"])
    path = path if path.is_absolute() else Path(root) / path
    path = path / "config.json" if path.is_dir() else path
    return checked_context(dict(base_sha256=base_sha256, parent_config_sha256=sha(path),
                                hybrid_module_sha256=config["hybrid_module"]["sha256"], fixture=False))


def checked_context(value):
    if (set(value) != {"base_sha256", "parent_config_sha256", "hybrid_module_sha256", "fixture"}
            or type(value["fixture"]) is not bool or any(not valid_sha(value[key]) for key in value if key != "fixture")):
        raise Error("exact immutable base/config/hybrid context required")
    return dict(value)


def tensor_header(path):
    """Read safetensors descriptors without importing Torch or opening a model."""
    with path.open("rb") as stream:
        size_raw = stream.read(8)
        if len(size_raw) != 8:
            raise Error("shared bank safetensors header missing")
        size = struct.unpack("<Q", size_raw)[0]
        if not 2 <= size <= 16 * 1024 * 1024:
            raise Error("shared bank safetensors header outside bounded size")
        header = json.loads(stream.read(size))
    result, spans = {}, []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        shape, offsets = entry.get("shape"), entry.get("data_offsets")
        if (entry.get("dtype") != "F32" or not isinstance(shape, list) or not shape
                or any(type(n) is not int or n < 1 for n in shape) or not isinstance(offsets, list)
                or len(offsets) != 2 or any(type(n) is not int for n in offsets)):
            raise Error("shared bank requires actual fp32 tensor census")
        count = 1
        for n in shape:
            count *= n
        if not 0 <= offsets[0] < offsets[1] or offsets[1] - offsets[0] != count * 4:
            raise Error("shared bank tensor byte extent differs from census")
        result[name] = dict(shape=shape, dtype="float32", numel=count)
        spans.append(tuple(offsets))
    spans.sort()
    if not spans or spans[0][0] != 0 or any(left[1] != right[0] for left, right in zip(spans, spans[1:])) or spans[-1][1] + 8 + size != path.stat().st_size:
        raise Error("shared bank tensor byte extents overlap or do not cover file")
    return result


def verify_entry(entry, root, expected_context):
    path = pinned(entry, root)
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("schema") != SCHEMA or value.get("complete_bank") is not True or value.get("all_finite") is not True
            or value.get("semantics") != "full_bank_replacement" or value.get("context") != checked_context(expected_context)
            or value.get("exporter_sha256") != sha(Path(__file__))):
        raise Error("shared bank schema/context/source/complete replacement contract differs")
    if digest({k: v for k, v in value.items() if k != "seal_sha256"}) != value.get("seal_sha256"):
        raise Error("shared bank manifest seal changed")
    data = pinned(value["tensors"], path.parent)
    if tensor_header(data) != value.get("tensor_census") or digest(value.get("site_plan")) != value.get("site_plan_sha256"):
        raise Error("shared bank actual header/site census differs")
    producer = value.get("producer", {})
    receipt_path = pinned(producer["checkpoint_receipt"], path.parent)
    state_path = pinned(producer["checkpoint_state"], path.parent)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("complete") is not True or receipt.get("state_sha256") != sha(state_path)
            or receipt.get("identity_sha256") != producer.get("run_identity_sha256") or receipt.get("step") != producer.get("step")):
        raise Error("shared bank producer checkpoint is incomplete or not this run")
    return dict(manifest_path=str(path), manifest_sha256=entry["sha256"], tensors_path=str(data), manifest=value)


def preflight_config(config, root, base_sha256):
    context = stage_context(config, base_sha256, root)
    policy = config.get("shared_r16")
    if not isinstance(policy, dict):
        raise Error("explicit shared_r16 initialization or previous full-bank source required")
    if policy.get("mode") == "initialize":
        if config.get("stage") != "D3a" or set(policy) != {"mode", "first_refit"} or policy["first_refit"] is not True:
            raise Error("fresh shared r16 initialization is restricted to explicit first D3a refit")
        return dict(mode="initialize", context=context, importer_sha256=sha(Path(__file__)))
    if policy.get("mode") != "inherit" or set(policy) != {"mode", "source"}:
        raise Error("shared_r16 must select one pinned full-bank inheritance source")
    verified = verify_entry(policy["source"], root, context)
    return dict(mode="inherit", context=context, source=policy["source"],
                source_manifest_sha256=verified["manifest_sha256"], importer_sha256=sha(Path(__file__)))


def bank_description(hybrid, context):
    import torch
    bank = hybrid.bank
    if bank is None or not bank.tensors or set(bank.names) != set(bank.tensors):
        raise Error("full physical shared bank is missing")
    plan = {kind: [asdict(spec) if is_dataclass(spec) else dict(spec) for spec in specs] for kind, specs in sorted(bank.plan.items())}
    if not context["fixture"] and any(spec.get("rank") != 16 or spec.get("alpha") != 16 or spec.get("k3_rank") != 0 or spec.get("gain") is not False for specs in plan.values() for spec in specs):
        raise Error("production shared bank must be r16/alpha16/noK3/no gain")
    census, storage = {}, set()
    for name, tensor in sorted(bank.tensors.items()):
        if (tensor.dtype != torch.float32 or not tensor.is_contiguous() or not tensor.is_leaf or not bool(torch.isfinite(tensor).all())):
            raise Error("shared bank tensors must be finite contiguous fp32 leaves")
        owner = (str(tensor.device), tensor.untyped_storage().data_ptr())
        if owner in storage:
            raise Error("shared bank has duplicate physical tensor ownership")
        storage.add(owner)
        census[name] = dict(shape=list(tensor.shape), dtype="float32", numel=tensor.numel())
    site_plan = dict(specs=plan, layer_types=list(bank.layer_types))
    return dict(tensor_census=census, site_plan=site_plan, site_plan_sha256=digest(site_plan))


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def import_bank(hybrid, entry, root, context):
    """Validate all tensors before copying; retain physical objects and grad flags."""
    import torch
    from safetensors.torch import load_file
    verified = verify_entry(entry, root, context)
    description = bank_description(hybrid, context)
    source = verified["manifest"]
    if any(description[key] != source[key] for key in description):
        raise Error("shared bank site/rank/shape census differs from actual hybrid")
    values = load_file(verified["tensors_path"], device="cpu")
    if set(values) != set(hybrid.bank.tensors) or any(not bool(torch.isfinite(value).all()) for value in values.values()):
        raise Error("shared bank loaded tensors incomplete or nonfinite")
    if any(tensor.grad is not None for tensor in hybrid.bank.tensors.values()):
        raise Error("shared bank import must precede gradient/optimizer use")
    before = {name: (id(tensor), tensor.requires_grad) for name, tensor in hybrid.bank.tensors.items()}
    with torch.no_grad():
        for name, tensor in hybrid.bank.tensors.items():
            tensor.copy_(values[name].to(tensor.device))
    if before != {name: (id(tensor), tensor.requires_grad) for name, tensor in hybrid.bank.tensors.items()}:
        raise Error("shared bank import replaced a physical owner or gradient policy")
    hybrid._recovery_shared_r16_parent = verified["manifest_sha256"]
    return dict(source_manifest_sha256=verified["manifest_sha256"], complete_bank=True,
                tensor_census_sha256=digest(description["tensor_census"]), physical_owners_preserved=True,
                optimizer_state_imported=False, semantics="full_bank_replacement")


def initialize_stage(hybrid, verified, root):
    description = bank_description(hybrid, verified["context"])
    if verified["mode"] == "inherit":
        result = import_bank(hybrid, verified["source"], root, verified["context"])
    else:
        hybrid._recovery_shared_r16_parent = None
        result = dict(source_manifest_sha256=None, complete_bank=True, optimizer_state_imported=False, semantics="first_D3a_initialization")
    hybrid._recovery_shared_r16_initial_hashes = {name: tensor_sha(tensor) for name, tensor in hybrid.bank.tensors.items()}
    return {**result, "context": verified["context"], "site_plan_sha256": description["site_plan_sha256"],
            "initial_bank_tensor_hashes_sha256": digest(hybrid._recovery_shared_r16_initial_hashes), "importer_sha256": sha(Path(__file__))}


def export_from_engine(engine, checkpoint, out, context, stage, unit=None):
    """Export every bank leaf, not just the current trainable optimizer census."""
    import torch
    from safetensors.torch import save_file
    if out.exists():
        raise Error("retain prior shared bank export")
    hybrid = getattr(engine, "hybrid", getattr(engine, "model", None))
    description = bank_description(hybrid, context)
    inherited = engine.identity.get("shared_r16", {})
    initial = getattr(hybrid, "_recovery_shared_r16_initial_hashes", None)
    if (initial is None or inherited.get("context") != context or inherited.get("initial_bank_tensor_hashes_sha256") != digest(initial)
            or inherited.get("source_manifest_sha256") != getattr(hybrid, "_recovery_shared_r16_parent", None)):
        raise Error("engine identity must bind pre-optimizer full-bank inheritance")
    if any(not tensor.requires_grad and tensor_sha(tensor) != initial[name] for name, tensor in hybrid.bank.tensors.items()):
        raise Error("frozen shared bank leaf changed during this stage")
    receipt = json.loads((checkpoint / "receipt.json").read_text(encoding="utf-8"))
    if (receipt.get("complete") is not True or receipt.get("state_sha256") != sha(checkpoint / "state.pt")
            or receipt.get("identity_sha256") != digest(engine.identity) or receipt.get("step") != engine.step):
        raise Error("shared bank export must bind actual complete engine checkpoint")
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
    if state.get("identity") != engine.identity or state.get("step") != engine.step:
        raise Error("checkpoint internal identity/step differs from engine")
    checkpoint_bank = {name[5:]: value for name, value in state.get("trainables", state.get("lora", {})).items() if name.startswith("lora.")}
    actual_owned = {name[5:] for name in engine.census if name.startswith("lora.")}
    if set(checkpoint_bank) != actual_owned or any(not torch.equal(value, hybrid.bank.tensors[name].detach().cpu()) for name, value in checkpoint_bank.items()):
        raise Error("shared bank export differs from recorded trained checkpoint leaves")
    tensors = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in sorted(hybrid.bank.tensors.items())}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=out.name + ".", dir=out.parent) as temporary:
        root = Path(temporary)
        save_file(tensors, str(root / "shared_r16.safetensors"), metadata={"schema": SCHEMA})
        value = dict(schema=SCHEMA, complete_bank=True, all_finite=True, semantics="full_bank_replacement", context=checked_context(context),
                     **description, tensors=dict(path="shared_r16.safetensors", sha256=sha(root / "shared_r16.safetensors")),
                     parent_manifest_sha256=getattr(hybrid, "_recovery_shared_r16_parent", None), exporter_sha256=sha(Path(__file__)),
                     producer=dict(stage=stage, unit=unit, step=engine.step, run_identity_sha256=digest(engine.identity),
                                   checkpoint_receipt=pin(checkpoint / "receipt.json"), checkpoint_state=pin(checkpoint / "state.pt")),
                     compose_required_nonbank_states=["learned native units", "input bridge deltas", "emission deltas", "music embed/head master tables"],
                     optimizer_state_exported=False, complete_native_checkpoint_composed=False, production_ready=False)
        value["seal_sha256"] = digest(value)
        (root / "manifest.json").write_bytes(canonical(value) + b"\n")
        root.rename(out)
    return pin(out / "manifest.json")
