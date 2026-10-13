"""Executable D4 tensor/runtime core; no production dispatcher or GPU launch.

Joint adapters run on authenticated pre-D2 text weights. Complete D2 and D3
banks remain distinct physical leaves, bound together at the genuine trainer's
sites. Numerical merge accumulates fp32 and folds K3 gain after both banks once.
Native cumulative interfaces can become one active owner without adding their
inherited values twice. Full replay/cache/lease/lineage admission remains outside
this core, and absence of an authentic D2 checkpoint refuses production binding.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import types

from . import stage_inheritance as I, native_state_inheritance as N

Error = I.Error
FAMILIES = {"D2.lora", "D2.k3_lora", "D2.gain", "D3.unit", "D3.emission", "D3.bridge", "D3.music_embed", "D3.music_head", "D3.shared_lora"}


def bank_plan(bank):
    return dict(specs={kind: [asdict(spec) for spec in specs] for kind, specs in sorted(bank.plan.items())}, layer_types=list(bank.layer_types))


def verify_d2_source(entry, root, source, bank):
    """Verify future authentic full D2 checkpoint, never infer it from a clone.

    The binding document must pin the actual COMPLETE.json, trainer state and
    original uncomposed source identity. No metadata-only partial cache fallback.
    The caller constructs a fresh original D2-shaped bank before this function.
    """
    path = I.pinned(entry, root)
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("schema") != "chimera_recovery.d4_d2_binding.v1" or value.get("stage") != "D2"
            or value.get("accepted_complete_stage") is not True or value.get("weights_semantics") != "uncomposed_pre_D2"
            or value.get("d2_already_embedded") is not False or value.get("optimizer_state_imported") is not False
            or value.get("source_identity") != source.identity() or value.get("site_plan") != bank_plan(bank)):
        raise Error("D4 requires authentic complete D2 bank on the exact uncomposed pre-D2 source")
    if I.digest({key: val for key, val in value.items() if key != "seal_sha256"}) != value.get("seal_sha256"):
        raise Error("D4 D2 binding seal differs")
    specs = [spec for group in bank.plan.values() for spec in group]
    if any(spec.rank != 64 or spec.alpha != 64 or (spec.kind == "ffn_out" and (spec.k3_rank != 32 or spec.k3_alpha != 32))
           or (spec.site == "mlp.up_proj" and not spec.gain) for spec in specs):
        raise Error("D4 production D2 bank must preserve actual r64/r32/gain sites")
    import torch
    from safetensors.torch import load_file
    marker_path = I.pinned(value["complete_json"], path.parent)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    producer = {}
    for name in ("adapters.safetensors", "optimizer.pt", "trainer_state.json"):
        target = marker_path.parent / name
        if not target.is_file() or I.sha(target) != marker.get("sha256", {}).get(name):
            raise Error("D4 actual D2 checkpoint files missing or changed")
        producer[name] = I.pin(target)
    state = json.loads((marker_path.parent / "trainer_state.json").read_text(encoding="utf-8"))
    if (state.get("step") != marker.get("step") or state.get("step") != value.get("completed_step")
            or state.get("source_identity") != source.identity()
            or state.get("source_identity_sha256") != hashlib.sha256(json.dumps(source.identity(), sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
            or state.get("data_identity_sha256") != value.get("data_identity_sha256") or state.get("step", 0) < 1):
        raise Error("D4 D2 producer step/model/data identity differs")
    values = load_file(str(marker_path.parent / "adapters.safetensors"), device="cpu")
    if set(values) != set(bank.tensors) or any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in values.values()):
        raise Error("D4 full D2 tensor inventory/dtype/finiteness differs")
    if any(list(values[name].shape) != list(tensor.shape) or tensor.grad is not None for name, tensor in bank.tensors.items()):
        raise Error("D4 D2 bank must be shape-exact and imported before optimizer/gradient use")
    with torch.no_grad():
        for name, tensor in bank.tensors.items():
            tensor.copy_(values[name].to(tensor.device))
    return dict(binding=entry, source_identity_sha256=I.digest(source.identity()), bank_sha256=producer["adapters.safetensors"]["sha256"],
                bank_state_sha256=I.digest({name: I.tensor_sha(value) for name, value in sorted(values.items())}),
                producer=producer, optimizer_state_imported=False, weights_semantics="uncomposed_pre_D2", d2_already_embedded=False)


class JointAdapterBank:
    """One bound site, separate D2/D3 leaves, one D2 gain; no full-bank sum."""
    def __init__(self, d2, shared, source_identity, binding, *, fixture=False):
        import torch
        if (binding.get("weights_semantics") != "uncomposed_pre_D2" or binding.get("d2_already_embedded") is not False
                or binding.get("source_identity_sha256") != I.digest(source_identity)
                or (not fixture and (not binding.get("producer") or not binding.get("bank_sha256")))):
            raise Error("joint D4 bank refuses composed D2 base or missing authenticated original source")
        if d2 is shared or d2.layer_types != shared.layer_types or set(d2.plan) != set(shared.plan):
            raise Error("D4 needs separate geometrically matching D2 and shared-D3 banks")
        self.d2, self.shared, self.layer_types = d2, shared, list(d2.layer_types)
        if not fixture:
            for ref in binding["producer"].values():
                I.pinned(ref, Path("."))
            if I.digest({name: I.tensor_sha(value) for name, value in sorted(d2.tensors.items())}) != binding.get("bank_state_sha256"):
                raise Error("D4 D2 bank no longer equals authenticated producer state")
        self.plan, self.specs = {}, {}
        for kind in d2.plan:
            left, right = {spec.site: spec for spec in d2.plan[kind]}, {spec.site: spec for spec in shared.plan[kind]}
            if set(left) != set(right):
                raise Error("D4 joint adapter sites differ")
            self.plan[kind], self.specs[kind] = [], (left, right)
            for site, first in left.items():
                second = right[site]
                if (any(getattr(first, key) != getattr(second, key) for key in ("kind", "in_features", "out_features", "trunk"))
                        or second.k3_rank != 0 or second.gain or not first.scaling > 0 or not second.scaling > 0):
                    raise Error("D4 D3 bank must match site geometry and own neither K3 adapter nor gain")
                self.plan[kind].append(replace(first, rank=first.rank + second.rank, alpha=float(first.rank + second.rank)))
        self.tensors = {**{"d2." + name: value for name, value in d2.tensors.items()},
                        **{"d3." + name: value for name, value in shared.tensors.items()}}
        self.names = sorted(self.tensors)
        self.binding, self.fixture = binding, fixture
        storage = set()
        for tensor in self.tensors.values():
            owner = (str(tensor.device), tensor.untyped_storage().data_ptr())
            if (owner in storage or not tensor.is_leaf or not tensor.is_contiguous() or tensor.dtype != torch.float32
                    or tensor.grad is not None or not bool(torch.isfinite(tensor).all())):
                raise Error("D4 banks require unique finite contiguous fp32 physical leaves before optimizer use")
            storage.add(owner)
            tensor.requires_grad_(True)

    def bound_site(self, index, site):
        import torch
        kind = self.layer_types[index]
        first, second = [part[site] for part in self.specs[kind]]
        a, b = self.d2.layer_sites[index][site], self.shared.layer_sites[index][site]
        result = dict(a=torch.cat([a["a"], b["a"]], dim=0),
                      b=torch.cat([a["b"] * first.scaling, b["b"] * second.scaling], dim=1))
        # K3 A/B and the activation gain belong only to D2.
        result.update({key: value for key, value in a.items() if key not in {"a", "b"}})
        return result

    def bind(self, layer, index):
        for spec in self.plan[self.layer_types[index]]:
            target = layer.get_submodule(spec.site)
            target._kd_spec = spec
            target._kd_bound = self.bound_site(index, spec.site)

    def census(self):
        return {"parameters_total": sum(value.numel() for value in self.tensors.values()), "tensors": len(self.tensors),
                "two_complete_bank_owners": True, "gain_owner": "D2 only", "source": self.binding}

    def trainable(self):
        return [self.tensors[name] for name in self.names]

    def zero_grad(self):
        for tensor in self.tensors.values():
            tensor.grad = None


def install_joint_bank(hybrid, joint):
    """Bind fresh concatenations per layer use, including resident recompute.

    The genuine streamed engine already binds every reload. Its resident mode
    normally binds once, which would retain a stale concatenation graph after
    the first backward; this wrapper refreshes the actual site references.
    """
    if (I.digest(hybrid.engine.source.identity()) != joint.binding["source_identity_sha256"]
            or any(value.grad is not None for value in joint.tensors.values()) or hasattr(hybrid, "_recovery_joint_bank")):
        raise Error("D4 joint bank must attach once to exact pre-D2 source before optimizer use")
    hybrid.bank = hybrid.engine.bank = joint
    layer = hybrid.engine.layer
    if hybrid.engine.mode == "resident":
        def fresh(self, index, next_index=None):
            value = layer(index, next_index)
            joint.bind(value, index)
            return value
        hybrid.engine.layer = types.MethodType(fresh, hybrid.engine)
    hybrid._recovery_joint_bank = joint


def merged_site_weights(base_weights, joint, index):
    """Return fresh merged tensors: fp32 accumulation, gain once, final cast.

    Inputs are original uncomposed weights in the genuine qwen3_5 layer naming.
    No source tensor or bank is mutated. Unit/media/music replacement remains a
    separate exact native-name writer; this function cannot certify a checkpoint.
    """
    import torch
    specs = joint.plan[joint.layer_types[index]]
    if set(base_weights) != {spec.site for spec in specs}:
        raise Error("D4 merge needs every exact original adapter site weight")
    result, gain = {}, None
    for spec in specs:
        original, parts = base_weights[spec.site], joint.bound_site(index, spec.site)
        if list(original.shape) != [spec.out_features, spec.in_features] or not bool(torch.isfinite(original).all()):
            raise Error("D4 merge original site shape/finiteness differs")
        weight = original.detach().float().clone()
        delta = parts["b"].float() @ parts["a"].float()
        if spec.kind == "full":
            weight += delta
        elif spec.kind == "ffn_in":
            weight[:spec.trunk] += delta
            if "gain" in parts:
                if gain is not None:
                    raise Error("D4 gain would be folded more than once")
                gain = parts["gain"].detach().float()
        elif spec.kind == "ffn_out":
            weight[:, :spec.trunk] += delta
            if "k3_a" in parts:
                weight[:, spec.trunk:] += spec.k3_scaling * (parts["k3_b"].float() @ parts["k3_a"].float())
        else:
            raise Error("D4 merge unknown native site semantics")
        result[spec.site] = weight
    if gain is None or "mlp.down_proj" not in result:
        raise Error("D4 requires exactly one authenticated D2 K3 gain per layer")
    down = next(spec for spec in specs if spec.site == "mlp.down_proj")
    if gain.shape != (down.in_features - down.trunk,) or not bool(torch.isfinite(gain).all()):
        raise Error("D4 gain geometry/finiteness differs")
    result[down.site][:, down.trunk:] *= gain
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise Error("D4 merge nonfinite result")
    return {site: value.to(base_weights[site].dtype).detach().contiguous() for site, value in result.items()}


def activate_native_interfaces(module, hybrid):
    """Promote cumulative learned fp32 interfaces to one compact active owner.

    Move each inherited cumulative total into the actual native SliceDelta's
    compact owned ranges and remove its old frozen overlay. Its forward still
    scatters/combines fp32 before one cast. No fresh increment is added twice.
    Must run before constructing any union optimizer.
    """
    import torch
    if hybrid.bridge_deltas or hybrid.head_deltas or not hasattr(hybrid, "_recovery_native_parent"):
        raise Error("D4 native activation needs fresh inherited stage, no active deltas")
    created = {"bridge": [], "emission": []}
    for key, value in sorted(hybrid._recovery_native_interfaces.items()):
        desc = hybrid._recovery_native_descriptors[key]
        N.checked_sparse(value, desc, hybrid)
        # Preserve adjacent type-band boundaries. Native descriptors authorize
        # each exact band, not an invented combined (type3,type4) interval.
        ranges = [tuple(pair) for pair in desc["ranges"]]
        delta = module.SliceDelta(desc["name"], tuple(desc["shape"]), ranges, desc["axis"], hybrid.device)
        selected = [value[lo:hi] if desc["axis"] == 0 else value[:, lo:hi] for lo, hi in ranges]
        with torch.no_grad():
            delta.delta.copy_(torch.cat(selected, dim=desc["axis"]))
        created[desc["kind"]].append(delta)
    hybrid.bridge_deltas = created["bridge"]
    hybrid.head_deltas = created["emission"]
    hybrid._recovery_native_interfaces = {}
    hybrid._recovery_native_overlay_hashes = {key: value for key, value in hybrid._recovery_native_overlay_hashes.items() if not key.startswith(("bridge.", "emission."))}
    return {kind: {kind + "." + delta.name: delta.delta for delta in deltas} for kind, deltas in created.items()}


def fresh_union_optimizer(groups, previous_rates, *, betas=(.9, .95), eps=1e-8, require_complete=True):
    """One fresh AdamW with one owner per actual leaf, prior rates times .3."""
    import torch
    if require_complete and set(groups) != FAMILIES:
        raise Error("D4 union misses an actual D2/D3 trainable family")
    if set(groups) != set(previous_rates) or set(groups) - FAMILIES:
        raise Error("D4 source optimizer rates must bind every actual family")
    owners, parameters, storage = {}, [], set()
    for family, named in sorted(groups.items()):
        previous = Decimal(str(previous_rates[family]))
        if not previous.is_finite() or previous <= 0 or not named:
            raise Error("D4 positive pinned previous optimizer rate and nonempty group required")
        for name, tensor in sorted(named.items()):
            token = (str(tensor.device), tensor.untyped_storage().data_ptr())
            if (name in owners or token in storage or not tensor.is_leaf or not tensor.requires_grad or tensor.dtype != torch.float32
                    or not tensor.is_contiguous() or tensor.numel() < 1 or not bool(torch.isfinite(tensor).all())):
                raise Error("D4 tensor has duplicate physical optimizer owner or is not a trainable fp32 leaf")
            owners[name], storage = family, storage | {token}
        parameters.append(dict(name=family, params=list(named.values()), lr=float(previous * Decimal("0.3")),
                               base_lr=float(previous * Decimal("0.3")), previous_base_lr=float(previous)))
    optimizer = torch.optim.AdamW(parameters, betas=betas, eps=eps, weight_decay=0.)
    if optimizer.state:
        raise Error("D4 union optimizer must start without previous Adam state")
    return optimizer, dict(owners=owners, ownership_sha256=I.digest(owners), single_optimizer=True, previous_state_imported=False)
