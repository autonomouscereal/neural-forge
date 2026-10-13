"""Source-pinned D4 candidate tensor export through the genuine P1 map.

Produces immutable replacement patches, not a registered production checkpoint.
Patches replace complete native tensors and must not be added to an already
composed D2/D3 child. The future publication adapter must bind the authenticated
pre-D2/base lineage, consume these patches exactly once, and rerun native parity.
"""
from __future__ import annotations
import json
from pathlib import Path
import tempfile

from . import consolidation_runtime as R

I, N, D, Error = R.I, R.N, R.D, R.Error
SCHEMA = "chimera_recovery.d4_native_replacements.v1"


def text_layer_updates(reader, joint, index, *, prefix):
    """Reverse actual P1 view/cat/interleave specs, scatter fp32, then cast once."""
    import torch
    kind = joint.layer_types[index]
    sites = joint.plan[kind]
    qnames = {spec.site: f"{prefix}.layers.{index}.{spec.site}.weight" for spec in sites}
    weights = {site: reader.get(name) for site, name in qnames.items()}
    merged = D.merged_site_weights(weights, joint, index, cast=False)
    updates, written = {}, {}
    def assign(name, rows, value):
        original = reader._raw(name)
        if name not in updates:
            updates[name] = original.detach().float().cpu().clone()
            written[name] = torch.zeros(original.shape[0], dtype=torch.bool)
        if rows is None:
            lo, hi = 0, original.shape[0]
        else:
            lo, hi = rows
        if (type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= original.shape[0]
                or list(value.shape) != list(original[lo:hi].shape) or bool(written[name][lo:hi].any())
                or not bool(torch.isfinite(value).all())):
            raise Error("D4 P1 inverse mapping writes overlapping/outside/mismatched native rows")
        updates[name][lo:hi] = value.detach().float().cpu()
        written[name][lo:hi] = True
    for site, qname in qnames.items():
        spec = reader.specs.get(qname)
        if not spec or list(merged[site].shape) != spec["shape"]:
            raise Error("D4 actual P1 adapter site inventory differs")
        parts = spec["parts"]
        if spec["kind"] == "view":
            if len(parts) != 1:
                raise Error("D4 P1 view requires its exact single source part")
            assign(*parts[0], merged[site])
        elif spec["kind"] == "cat":
            offset = 0
            for name, rows in parts:
                base = reader._raw(name)
                size = base.shape[0] if rows is None else rows[1] - rows[0]
                assign(name, rows, merged[site][offset:offset + size])
                offset += size
            if offset != merged[site].shape[0]:
                raise Error("D4 P1 cat source sizes differ")
        elif spec["kind"] == "interleave":
            if len(parts) != 2 or any(rows is not None for _, rows in parts):
                raise Error("D4 P1 interleave requires exact query and gate parts")
            heads, dim = spec["heads"], spec["head_dim"]
            shaped = merged[site].reshape(heads, 2, dim, -1)
            assign(*parts[0], shaped[:, 0].reshape(heads * dim, -1))
            assign(*parts[1], shaped[:, 1].reshape(heads * dim, -1))
        else:
            raise Error("D4 unsupported P1 inverse mapping at trained site")
    # All rows of each touched physical matrix must be completed. Fused gate/up
    # share a target but use disjoint halves; no untouched-row inference.
    if any(not bool(mask.all()) for mask in written.values()):
        raise Error("D4 incomplete fused native text matrix")
    return {name: value.to(reader._raw(name).dtype).contiguous() for name, value in updates.items()}


def native_updates(hybrid):
    """Yield genuine unit replacements, cumulative interfaces and music codes."""
    import torch
    if not hasattr(hybrid, "_recovery_d4_music"):
        raise Error("D4 native export requires activated complete union music state")
    inventory = N.unit_parameters(hybrid)
    placements = {p.container: p for p in hybrid.layout}
    for container, unit in sorted(hybrid.units.items()):
        placement = placements[container]
        for name, value in sorted(unit.named_parameters()):
            key = f"unit.{container}.{name}"
            if key not in hybrid._recovery_native_unit_keys:
                continue
            target = f"blocks.{placement.block}.charted.{name}"
            original = hybrid.reader.extra(target)
            if value.dtype != torch.float32 or value.shape != original.shape or not bool(torch.isfinite(value).all()):
                raise Error("D4 native unit shape/dtype/finiteness differs")
            yield target, value.detach().cpu().to(original.dtype).contiguous()
    if not hybrid._recovery_native_unit_keys <= set(inventory):
        raise Error("D4 native unit target inventory missing")
    for kind, prefix in (("bridge", "native_media_bridge."), ("emission", "native_media_head.")):
        masters, descs = N.cumulative_interfaces(hybrid, kind)
        for key, master in sorted(masters.items()):
            desc = descs[key]
            N.checked_sparse(master, desc, hybrid)
            target = prefix + desc["name"]
            original = hybrid.reader.extra(target)
            # One cumulative fp32 scatter is combined with the immutable native
            # base before the final stored dtype conversion; no intermediate cast.
            yield target, (original.detach().float().cpu() + master.detach().float().cpu()).to(original.dtype).contiguous()
    music = hybrid._recovery_d4_music
    contract = music["contract"]
    begin, codes = hybrid.music_band_begin, contract["code_rows"]
    if hybrid.reader.tied:
        raise Error("D4 separate learned embed/head masters require actual untied native tables")
    for target, master, reserved, label in (("embed.weight", music["embed"], music["reserved_embed"], "reserved_embed_sha256"),
                                           ("lm_head.weight", music["head"], music["reserved_head"], "reserved_head_sha256")):
        original = hybrid.reader._raw(target)
        if (begin != hybrid.base_vocab + contract["offset"] or list(master.shape) != [codes, hybrid.hidden]
                or I.tensor_sha(reserved) != contract[label] or I.tensor_sha(music["ledger"]) != contract["ledger_sha256"]
                or not bool(torch.isfinite(master).all())):
            raise Error("D4 native music allocation/protected row/finite contract differs")
        value = original.detach().cpu().clone()
        value[begin:begin + codes] = master.detach().cpu().to(value.dtype)
        if not torch.equal(value[:begin], original[:begin]) or not torch.equal(value[begin + codes:], original[begin + codes:]):
            raise Error("D4 native music writer changed protected rows")
        yield target, value


def iter_replacements(engine):
    """Bounded one-layer/one-native-tensor iterator over the actual union."""
    hybrid = engine.hybrid
    if (not isinstance(hybrid.bank, D.JointAdapterBank) or not hasattr(hybrid, "_recovery_d4_rebase")
            or hybrid.engine.source.identity() != hybrid._recovery_d4_rebase["original_source"]):
        raise Error("D4 writer needs exact once-only joint banks on authenticated original text source")
    source = hybrid.engine.source
    prefix = getattr(source, "text_prefix", hybrid.T.DEFAULT_TEXT_PREFIX)
    seen = set()
    for index in range(hybrid.num_layers):
        for name, value in sorted(text_layer_updates(source.reader, hybrid.bank, index, prefix=prefix).items()):
            if name in seen:
                raise Error("D4 native writer repeats physical target")
            seen.add(name)
            yield name, value
    for name, value in native_updates(hybrid):
        if name in seen:
            raise Error("D4 native writer overlaps text and native owners")
        seen.add(name)
        yield name, value


def export_candidate(engine, checkpoint, output):
    """Export immutable completed candidate patches; do not register a model."""
    import torch
    from safetensors.torch import save_file
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise Error("preserve previous D4 replacement artifact")
    engine.verify_sources()
    engine.verify_frozen()
    receipt = json.loads((checkpoint / "receipt.json").read_text(encoding="utf-8"))
    if (receipt.get("complete") is not True or receipt.get("stage_training_complete") is not True
            or receipt.get("state_sha256") != I.sha(checkpoint / "state.pt")
            or receipt.get("identity_sha256") != I.digest(engine.identity)
            or engine.step != engine.total_steps or engine.cursor != len(engine.rows)):
        raise Error("D4 writer requires actual complete declared-budget checkpoint")
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
    if (state.get("identity") != engine.identity or state.get("step") != engine.step or state.get("cursor") != engine.cursor
            or set(state.get("trainables", {})) != set(engine.census)
            or any(not torch.equal(value.detach().cpu(), state["trainables"][key]) for key, value in engine.census.items())):
        raise Error("D4 native writer current state differs from sealed checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".", dir=output.parent) as temporary:
        root = Path(temporary)
        updates = []
        for index, (name, value) in enumerate(iter_replacements(engine)):
            if not bool(torch.isfinite(value).all()):
                raise Error("D4 native writer nonfinite replacement")
            filename = f"tensor-{index:05d}.safetensors"
            save_file({name: value}, str(root / filename), metadata={"schema": SCHEMA})
            updates.append(dict(target=name, shape=list(value.shape), dtype=str(value.dtype),
                                tensor_sha256=I.tensor_sha(value), file=dict(path=filename, sha256=I.sha(root / filename))))
        if not updates:
            raise Error("D4 native writer has no actual targets")
        value = dict(schema=SCHEMA, semantics="complete_native_tensor_replacement_on_pre_D2_base_once", fixture=engine.identity["fixture"],
                     identity_sha256=I.digest(engine.identity), checkpoint_receipt=I.pin(checkpoint / "receipt.json"), checkpoint_state=I.pin(checkpoint / "state.pt"),
                     original_text_source=engine.hybrid._recovery_d4_rebase["original_source"],
                     immutable_native_source=engine.hybrid._recovery_d4_rebase["d2_native_source"],
                     updates=updates, compute_dtype="float32", cast="once_after_target_complete", gain="D2_only_after_both_banks",
                     helper_sha256=I.sha(Path(__file__)), native_music_protected_rows=engine.hybrid._recovery_d4_music["contract"],
                     complete_replacement_inventory=True, stage_training_complete=True,
                     complete_native_checkpoint_composed=False, production_ready=False,
                     required_next_checks=["selected replay and allocator/objective/safety bindings", "source-bound registered checkpoint publication adapter", "native stored-dtype parity and held-out capability evaluation"])
        value["seal_sha256"] = I.digest(value)
        (root / "manifest.json").write_bytes(I.canonical(value) + b"\n")
        root.rename(output)
    return I.pin(output / "manifest.json")
