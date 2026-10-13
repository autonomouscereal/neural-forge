"""Cumulative native-state handoff between recovery stages.

Units are replacements, interfaces are cumulative fp32 deltas against the
immutable native base, and music tables are fp32 masters. Nothing is added to
production files. The forward adapter combines old and current interface deltas
before one dtype cast; independent stage banks are never summed.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types

spec = importlib.util.spec_from_file_location("chimera_native_state_bank_contract", Path(__file__).with_name("stage_inheritance.py"))
I = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = I
spec.loader.exec_module(I)
Error = I.Error
SCHEMA = "chimera_recovery.native_state.v1"


def verify_entry(entry, root, context, shared_source_sha256):
    path = I.pinned(entry, root)
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("schema") != SCHEMA or value.get("complete_cumulative_state") is not True
            or value.get("stage_training_complete") is not True
            or value.get("all_finite") is not True or value.get("context") != I.checked_context(context)
            or value.get("exporter_sha256") != I.sha(Path(__file__))
            or value.get("shared_r16", {}).get("sha256") != shared_source_sha256
            or value.get("semantics") != {"unit": "replacement", "bridge": "cumulative_fp32_delta", "emission": "cumulative_fp32_delta", "music": "fp32_master_replacement"}):
        raise Error("native state context/source/cumulative/shared-bank contract differs")
    if I.digest({k: v for k, v in value.items() if k != "seal_sha256"}) != value.get("seal_sha256"):
        raise Error("native state manifest seal changed")
    bank = I.verify_entry(value["shared_r16"], path.parent, context)
    producer = value["producer"]
    if producer != bank["manifest"]["producer"]:
        raise Error("native and shared bank must come from the exact same checkpoint")
    tensors = I.pinned(value["tensors"], path.parent)
    if I.tensor_header(tensors) != value.get("tensor_census") or I.digest(value.get("native_inventory")) != value.get("native_inventory_sha256"):
        raise Error("native actual header/inventory differs")
    if set(value.get("descriptors", {})) != set(value["tensor_census"]):
        raise Error("native tensor semantics inventory incomplete")
    return dict(manifest=value, manifest_path=str(path), manifest_sha256=entry["sha256"], tensors_path=str(tensors))


def preflight_config(config, root, shared):
    policy = config.get("native_state")
    if not isinstance(policy, dict):
        raise Error("explicit native_state initialization or cumulative source required")
    if policy.get("mode") == "initialize":
        if config.get("stage") != "D3a" or policy != {"mode": "initialize", "first_refit": True} or shared["mode"] != "initialize":
            raise Error("native initialization is restricted to the explicit first D3a refit")
        return dict(mode="initialize", context=shared["context"], importer_sha256=I.sha(Path(__file__)))
    if policy.get("mode") != "inherit" or set(policy) != {"mode", "source"} or shared["mode"] != "inherit":
        raise Error("native and shared bank must inherit one exact producing stage")
    checked = verify_entry(policy["source"], root, shared["context"], shared["source_manifest_sha256"])
    return dict(mode="inherit", source=policy["source"], source_manifest_sha256=checked["manifest_sha256"],
                shared_r16_sha256=shared["source_manifest_sha256"], context=shared["context"], importer_sha256=I.sha(Path(__file__)))


def native_inventory(hybrid):
    def describe(module):
        return {name: dict(shape=list(value.shape), dtype=str(value.dtype), frozen_coefficient=name in ("h3_time.curve_table", "attn.rope_inv_freq"))
                for name, value in sorted(module.named_parameters())}
    result = dict(units={name: describe(unit) for name, unit in sorted(hybrid.units.items())}, bridge=describe(hybrid.bridge),
                emission=describe(hybrid.media_head), bands={str(k): v for k, v in hybrid.bands.items()},
                source=hybrid.source.identity(),
                layout=[dict(container=p.container, depth=p.depth, type_ids=list(p.type_ids)) for p in hybrid.layout],
                music=dict(begin=hybrid.music_band_begin, base_vocab=hybrid.base_vocab, embed_shape=list(hybrid.band_embed.shape),
                           head_shape=None if hybrid.band_head is None else list(hybrid.band_head.shape)))
    return json.loads(I.canonical(result))


def unit_parameters(hybrid):
    return {f"unit.{unit}.{name}": value for unit, module in hybrid.units.items() for name, value in module.named_parameters()}


def fixed_parameters(hybrid):
    return {**{"bridge." + name: value for name, value in hybrid.bridge.named_parameters()},
            **{"emission." + name: value for name, value in hybrid.media_head.named_parameters()}}


def descriptor_for_delta(delta, kind):
    return dict(kind=kind, name=delta.name, shape=list(delta.shape), axis=delta.axis, ranges=[list(pair) for pair in delta.ranges])


def checked_descriptor(desc, hybrid):
    kind, name, shape, axis, ranges = (desc.get(k) for k in ("kind", "name", "shape", "axis", "ranges"))
    module = hybrid.bridge if kind == "bridge" else hybrid.media_head if kind == "emission" else None
    allowed = {"bridge": {"feature_proj.weight": (1, "feature"), "type_embed.weight": (0, None)},
               "emission": {"out.weight": (0, "emission"), "type_affine.weight": (0, None)}}
    if module is None or name not in allowed[kind] or axis != allowed[kind][name][0] or shape != list(dict(module.named_parameters())[name].shape):
        raise Error("interface descriptor does not identify an actual native weight/axis")
    band = allowed[kind][name][1]
    valid_ranges = {tuple(hybrid.bands[t][band]) if band else (t, t + 1) for t in hybrid.bands}
    if not isinstance(ranges, list) or not ranges or any(not isinstance(pair, list) or len(pair) != 2 or any(type(n) is not int for n in pair) or tuple(pair) not in valid_ranges for pair in ranges):
        raise Error("interface range is outside actual native ownership bands")
    ordered = sorted(ranges)
    if ranges != ordered or any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise Error("interface ownership ranges overlap or repeat")
    return desc


def checked_sparse(value, desc, hybrid):
    import torch
    checked_descriptor(desc, hybrid)
    if list(value.shape) != desc["shape"] or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise Error("native interface master shape/dtype/finite check failed")
    mask = torch.zeros(desc["shape"][desc["axis"]], dtype=torch.bool, device=value.device)
    for a, b in desc["ranges"]:
        mask[a:b] = True
    outside = value[~mask] if desc["axis"] == 0 else value[:, ~mask]
    if bool((outside != 0).any()):
        raise Error("native interface master writes outside owned ranges")


def scatter(delta):
    import torch
    value = torch.zeros(delta.shape, dtype=torch.float32, device=delta.delta.device)
    offset = 0
    for a, b in delta.ranges:
        width = b - a
        if delta.axis == 0:
            value[a:b] = delta.delta[offset:offset + width]
        else:
            value[:, a:b] = delta.delta[:, offset:offset + width]
        offset += width
    return value


def cumulative_interfaces(hybrid, kind, *, validate=True):
    import torch
    old = hybrid._recovery_native_interfaces
    masters = {key: value for key, value in old.items() if key.startswith(kind + ".")}
    descriptors = {key: value for key, value in hybrid._recovery_native_descriptors.items() if key.startswith(kind + ".")}
    deltas = hybrid.bridge_deltas if kind == "bridge" else hybrid.head_deltas
    seen = set()
    for delta in deltas:
        key = kind + "." + delta.name
        if key in seen:
            raise Error("one physical stage delta owner required per interface parameter")
        seen.add(key)
        desc = descriptor_for_delta(delta, kind)
        checked_descriptor(desc, hybrid)
        value = scatter(delta)
        if key in masters:
            previous = descriptors[key]
            if previous["axis"] != desc["axis"] or previous["shape"] != desc["shape"]:
                raise Error("interface cumulative shape/axis differs")
            desc["ranges"] = sorted([list(pair) for pair in {tuple(pair) for pair in previous["ranges"] + desc["ranges"]}])
            value = masters[key].to(value.device) + value
        if validate:
            checked_sparse(value, desc, hybrid)
        masters[key], descriptors[key] = value, desc
    return masters, descriptors


def interface_parameters(hybrid, kind):
    masters, _ = cumulative_interfaces(hybrid, kind, validate=False)
    module = hybrid.bridge if kind == "bridge" else hybrid.media_head
    named = dict(module.named_parameters())
    prefix = "bridge." if kind == "bridge" else "head."
    # Cast the combined delta exactly once, matching native SliceDelta.full;
    # do not fold intermediate stages into a low-precision base tensor.
    return {prefix + key[len(kind) + 1:]: named[key[len(kind) + 1:]] + value.to(named[key[len(kind) + 1:]].dtype)
            for key, value in masters.items()}


def install_forward_adapter(hybrid):
    import torch
    def shim(self):
        class Embedder:
            def embed(_self, *args):
                params = interface_parameters(self, "bridge")
                return torch.func.functional_call(self._bridge_call, params, args) if params else self._bridge_call(*args)
        return types.SimpleNamespace(has_charted_blocks=bool(self.layout), native_media_bridge=Embedder(), cfg=self.cfg)
    def emission(self, normed, context):
        rows = self.media_hidden(normed, context)
        params = interface_parameters(self, "emission")
        return torch.func.functional_call(self._head_call, params, (rows, context["types"])) if params else self._head_call(rows, context["types"])
    hybrid._shim = types.MethodType(shim, hybrid)
    hybrid.emission = types.MethodType(emission, hybrid)


def music_contract(hybrid, code_rows=None):
    offset = hybrid.music_band_begin - hybrid.base_vocab
    rows = hybrid.band_embed.shape[0] - offset
    if code_rows is not None and (type(code_rows) is not int or not 1 <= code_rows <= rows):
        raise Error("music code rows outside allocated native band")
    return dict(offset=offset, rows=rows, code_rows=64000 if code_rows is None else code_rows,
                ledger_sha256=I.tensor_sha(hybrid.band_embed[:offset].float()),
                reserved_embed_sha256=I.tensor_sha(hybrid.band_embed[offset + (64000 if code_rows is None else code_rows):].float()),
                reserved_head_sha256=I.tensor_sha(hybrid.band_head[offset + (64000 if code_rows is None else code_rows):].float()))


def initialize_stage(hybrid, verified, root):
    """Copy validated replacements before any stage optimizer is constructed."""
    import torch
    from safetensors.torch import load_file
    if hybrid.head_deltas or hybrid.bridge_deltas or hybrid.band_embed_override is not None or hasattr(hybrid, "_recovery_native_parent"):
        raise Error("native inheritance must run once on a fresh hybrid before optimizer setup")
    inventory = native_inventory(hybrid)
    parameters = unit_parameters(hybrid)
    if any(value.grad is not None for value in parameters.values()):
        raise Error("native inheritance must precede gradient/optimizer use")
    values, descriptors, music, parent = {}, {}, None, None
    if verified["mode"] == "inherit":
        if verified["shared_r16_sha256"] != getattr(hybrid, "_recovery_shared_r16_parent", None):
            raise Error("import shared bank from the exact native producing stage first")
        source = verify_entry(verified["source"], root, verified["context"], verified["shared_r16_sha256"])
        manifest = source["manifest"]
        if inventory != manifest["native_inventory"]:
            raise Error("native source layout/table/shape inventory differs from actual hybrid")
        values = load_file(source["tensors_path"], device="cpu")
        descriptors, music, parent = manifest["descriptors"], manifest.get("music_contract"), source["manifest_sha256"]
        if set(values) != set(descriptors) or any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in values.values()):
            raise Error("native loaded state incomplete, nonfinite or not fp32")
        for key, value in values.items():
            desc = descriptors[key]
            if desc["kind"] == "unit":
                if key not in parameters or list(value.shape) != list(parameters[key].shape) or any(key.endswith("." + name) for name in ("h3_time.curve_table", "attn.rope_inv_freq")):
                    raise Error("native unit replacement missing, wrong shape or source-frozen coefficient")
            elif desc["kind"] in ("bridge", "emission"):
                if key != desc["kind"] + "." + desc["name"]:
                    raise Error("interface tensor name differs from descriptor")
                checked_sparse(value, desc, hybrid)
            elif desc["kind"] == "music":
                if key not in {"music.embed", "music.head"} or not music or list(value.shape) != [music["rows"], hybrid.hidden]:
                    raise Error("native music master contract differs")
            else:
                raise Error("unknown native state family")
        if music:
            if not verified["context"]["fixture"] and (music["code_rows"] != 64000 or music["rows"] != 65536 or hybrid.music_band_begin != 252416 or hybrid.hidden != 5120):
                raise Error("production music must retain exact allocated/written row contract")
            if {"music.embed", "music.head"} - set(values) or music != music_contract(hybrid, music["code_rows"]):
                raise Error("native music ledger/reserved rows do not match immutable base")
            for key, label in (("music.embed", "reserved_embed_sha256"), ("music.head", "reserved_head_sha256")):
                if I.tensor_sha(values[key][music["code_rows"]:]) != music[label]:
                    raise Error("native music replacement changes reserved rows")
    # All checks precede any mutation; objects and stage ownership are retained.
    with torch.no_grad():
        for key, value in values.items():
            if descriptors[key]["kind"] == "unit":
                parameters[key].copy_(value.to(parameters[key].device, parameters[key].dtype))
    hybrid._recovery_native_interfaces = {key: value.to(hybrid.device) for key, value in values.items() if descriptors[key]["kind"] in ("bridge", "emission")}
    hybrid._recovery_native_descriptors = descriptors
    hybrid._recovery_native_unit_keys = {key for key in values if descriptors[key]["kind"] == "unit"}
    hybrid._recovery_native_music_contract = music
    if music:
        hybrid.band_embed_override = hybrid.band_embed.float().cpu().clone()
        hybrid.band_embed_override[music["offset"]:] = values["music.embed"]
        hybrid._recovery_music_head_master = values["music.head"]
    hybrid._recovery_native_parent = parent
    hybrid._recovery_native_initial_hashes = {key: I.tensor_sha(value) for key, value in {**parameters, **fixed_parameters(hybrid)}.items()}
    hybrid._recovery_native_overlay_hashes = {key: I.tensor_sha(value) for key, value in hybrid._recovery_native_interfaces.items()}
    if music:
        hybrid._recovery_native_overlay_hashes.update({"music.embed": I.tensor_sha(values["music.embed"]), "music.head": I.tensor_sha(values["music.head"])})
    install_forward_adapter(hybrid)
    return dict(source_manifest_sha256=parent, context=verified["context"], native_inventory_sha256=I.digest(inventory),
                initial_native_hashes_sha256=I.digest(hybrid._recovery_native_initial_hashes), forward_adapter_sha256=I.sha(Path(__file__)),
                initial_overlay_hashes_sha256=I.digest(hybrid._recovery_native_overlay_hashes),
                optimizer_state_imported=False, complete_native_checkpoint_composed=False)


def export_from_engine(engine, checkpoint, out, context, stage, shared_r16):
    """Export cumulative states, including earlier families outside this census."""
    import torch
    from safetensors.torch import save_file
    if out.exists():
        raise Error("preserve prior cumulative native export")
    hybrid = getattr(engine, "hybrid", getattr(engine, "model", None))
    inherited = engine.identity.get("native_state", {})
    inventory, initial = native_inventory(hybrid), getattr(hybrid, "_recovery_native_initial_hashes", None)
    if (initial is None or inherited.get("context") != context or inherited.get("source_manifest_sha256") != hybrid._recovery_native_parent
            or inherited.get("initial_native_hashes_sha256") != I.digest(initial) or inherited.get("forward_adapter_sha256") != I.sha(Path(__file__))
            or inherited.get("initial_overlay_hashes_sha256") != I.digest(hybrid._recovery_native_overlay_hashes)
            or inherited.get("native_inventory_sha256") != I.digest(inventory)):
        raise Error("engine identity must bind actual pre-optimizer native state inheritance")
    if any(I.tensor_sha(value) != hybrid._recovery_native_overlay_hashes[key] for key, value in hybrid._recovery_native_interfaces.items()):
        raise Error("frozen inherited native interface master changed")
    bank = I.verify_entry(shared_r16, out.parent, context)["manifest"]
    producer = bank["producer"]
    if (producer["run_identity_sha256"] != I.digest(engine.identity) or producer["step"] != engine.step
            or producer["checkpoint_state"] != I.pin(checkpoint / "state.pt") or producer["checkpoint_receipt"] != I.pin(checkpoint / "receipt.json")):
        raise Error("native export requires matching current checkpoint and shared bank")
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
    if type(engine.total_steps) is not int or engine.total_steps < 1 or engine.step != engine.total_steps or state.get("total_steps") != engine.total_steps:
        raise Error("native successor export requires the complete declared stage budget")
    recorded = state.get("trainables", state.get("lora", {}))
    if state.get("identity") != engine.identity or state.get("step") != engine.step or set(recorded) != set(engine.census):
        raise Error("native checkpoint census/identity/step differs")
    for key, value in engine.census.items():
        if not torch.equal(value.detach().cpu(), recorded[key]):
            raise Error("native engine trainable differs from actual checkpoint")
    params, values, descriptors = unit_parameters(hybrid), {}, {}
    current_units = {key for key in engine.census if key.startswith("unit.")}
    for key, value in {**params, **fixed_parameters(hybrid)}.items():
        if key not in current_units and I.tensor_sha(value) != initial[key]:
            raise Error("frozen native unit/base interface weight changed")
    for key in sorted(hybrid._recovery_native_unit_keys | current_units):
        value = params[key]
        if any(key.endswith("." + name) for name in ("h3_time.curve_table", "attn.rope_inv_freq")):
            raise Error("source-frozen unit coefficient cannot be exported as learned state")
        values[key] = value.detach().float().cpu().contiguous().clone()
        descriptors[key] = dict(kind="unit", shape=list(value.shape))
    for kind in ("bridge", "emission"):
        masters, descs = cumulative_interfaces(hybrid, kind)
        values.update({key: value.detach().float().cpu().contiguous() for key, value in masters.items()})
        descriptors.update(descs)
    music = hybrid._recovery_native_music_contract
    if hasattr(engine, "embed_master") and hasattr(engine, "head_master"):
        music = music_contract(hybrid, engine.code_rows)
        if (state.get("rows") != music["rows"] or state.get("code_rows") != music["code_rows"]
                or not torch.equal(state["embed"], engine.embed_master.cpu()) or not torch.equal(state["head"], engine.head_master.cpu())):
            raise Error("native music masters differ from recorded checkpoint")
        values.update({"music.embed": engine.embed_master.detach().cpu().contiguous(), "music.head": engine.head_master.detach().cpu().contiguous()})
    elif music:
        values.update({"music.embed": hybrid.band_embed_override[music["offset"]:].detach().cpu().contiguous(),
                       "music.head": hybrid._recovery_music_head_master.detach().cpu().contiguous()})
        if any(I.tensor_sha(values[key]) != hybrid._recovery_native_overlay_hashes[key] for key in ("music.embed", "music.head")):
            raise Error("frozen inherited music master changed outside music stage")
    if music:
        for key, label in (("music.embed", "reserved_embed_sha256"), ("music.head", "reserved_head_sha256")):
            if I.tensor_sha(values[key][music["code_rows"]:]) != music[label]:
                raise Error("native music reserved rows changed")
            descriptors[key] = dict(kind="music", shape=list(values[key].shape))
        if I.tensor_sha(hybrid.band_embed_override[:music["offset"]]) != music["ledger_sha256"]:
            raise Error("native music ledger rows changed")
    if not values or any(not bool(torch.isfinite(value).all()) for value in values.values()):
        raise Error("native cumulative export must contain actual finite learned state")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=out.name + ".", dir=out.parent) as temporary:
        root = Path(temporary)
        save_file(values, str(root / "native_state.safetensors"), metadata={"schema": SCHEMA})
        value = dict(schema=SCHEMA, complete_cumulative_state=True, stage_training_complete=True, all_finite=True, context=I.checked_context(context),
                     semantics=dict(unit="replacement", bridge="cumulative_fp32_delta", emission="cumulative_fp32_delta", music="fp32_master_replacement"),
                     tensors=dict(path="native_state.safetensors", sha256=I.sha(root / "native_state.safetensors")), tensor_census=I.tensor_header(root / "native_state.safetensors"),
                     descriptors=descriptors, native_inventory=inventory, native_inventory_sha256=I.digest(inventory), music_contract=music,
                     parent_manifest_sha256=hybrid._recovery_native_parent, shared_r16=shared_r16, producer=producer,
                     stage=stage, exporter_sha256=I.sha(Path(__file__)), optimizer_state_exported=False,
                     complete_native_checkpoint_composed=False, production_ready=False)
        value["seal_sha256"] = I.digest(value)
        (root / "manifest.json").write_bytes(I.canonical(value) + b"\n")
        root.rename(out)
    return I.pin(out / "manifest.json")
