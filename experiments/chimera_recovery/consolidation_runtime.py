"""Bounded D4 union execution on genuine HybridD3; no launch dispatcher.

The native reader retains the immutable D2 child while text layers are reopened
on verified pre-D2 weights, with both banks bound once. A caller must supply
authentic completed D2 and terminal D3d_speech_flow exports. Fixture mode is
explicit and cannot produce a production-ready receipt. No historical commands
are executed; replay callbacks are trusted native objective functions supplied
by the candidate dispatcher, which is still required before production use.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import tempfile
import types

from . import train_consolidation as D

I, N, Error = D.I, D.N, D.Error
SCHEMA = "chimera_recovery.d4_union_runtime.v1"
TERMINAL = "D3d_speech_flow"


def model_files(source):
    """Pin actual P1 roots and all composing files, not only shard sizes."""
    reader = source.reader
    roots = [reader.shards]
    if reader.overlay is not None:
        roots.extend(shards for _, shards in reader.overlay.links)
    paths = {Path(source.parent_config_path)}
    if source.plan is not None:
        paths.add(Path(source.plan))
    for shards in roots:
        paths.update(shards.directory / name for name in shards.files)
        paths.update(shards.directory.glob("*.json"))
    for ref in (source.identity().get("p1_tool"),):
        if ref:
            paths.add(Path(ref["path"]))
    return {str(path.resolve()): I.pin(path.resolve()) for path in sorted(paths, key=str)}


def rebase_text(hybrid, original, d2):
    """Prove every actual text tensor before changing the engine's source.

    Adapter sites in the D2 source must equal the exact numerical D2-only
    composition; all other text tensors must remain bit-identical. The proof
    hashes tensor bytes, rather than trusting P1's shard-size identity. Native
    unit/interface/band tables retain the authenticated D2 reader and exports.
    """
    import torch
    engine = hybrid.engine
    if (engine.layer_loads or engine._pending or hasattr(hybrid, "_recovery_joint_bank")
            or original.config_dict() != hybrid.source.config_dict()):
        raise Error("D4 rebase must precede layer/optimizer use on identical text geometry")
    prefix = getattr(original, "text_prefix", hybrid.T.DEFAULT_TEXT_PREFIX)
    if prefix != getattr(hybrid.source, "text_prefix", hybrid.T.DEFAULT_TEXT_PREFIX):
        raise Error("D4 original and composed text prefixes differ")
    # A D2-only view carries the genuine site scaling/K3/gain; no shared bank.
    def bound(index, site):
        spec = next(s for s in d2.plan[d2.layer_types[index]] if s.site == site)
        parts = d2.layer_sites[index][site]
        return {**parts, "b": parts["b"] * spec.scaling}
    view = types.SimpleNamespace(plan=d2.plan, layer_types=d2.layer_types, bound_site=bound)
    hashes = {}
    names = set()
    for index, kind in enumerate(engine.layer_types):
        base = f"{prefix}.layers.{index}."
        listed = set(original.names_under(base))
        if listed != set(hybrid.source.names_under(base)):
            raise Error("D4 text rebase inventory differs")
        weights = {spec.site: original.get(base + spec.site + ".weight") for spec in d2.plan[kind]}
        merged = D.merged_site_weights(weights, view, index)
        replacements = {base + site + ".weight": tensor for site, tensor in merged.items()}
        for name in sorted(listed):
            before, observed = original.get(name), hybrid.source.get(name)
            expected = replacements.get(name, before)
            if (observed.dtype != expected.dtype or observed.shape != expected.shape
                    or not torch.equal(observed, expected) or not bool(torch.isfinite(observed).all())):
                raise Error("D4 D2 child differs from exact once-only numerical composition: " + name)
            hashes[name] = {"original": I.tensor_sha(before), "d2": I.tensor_sha(observed)}
            names.add(name)
    for name in (f"{prefix}.embed_tokens.weight", f"{prefix}.norm.weight", hybrid.T.head_weight_name(original)):
        before, observed = original.get(name), hybrid.source.get(name)
        if before.dtype != observed.dtype or not torch.equal(before, observed) or not bool(torch.isfinite(before).all()):
            raise Error("D4 frozen text embed/head/norm differs during rebase: " + name)
        hashes[name] = {"original": I.tensor_sha(before), "d2": I.tensor_sha(observed)}
        names.add(name)
    # Mutation follows the completed tensor proof. Resident layers are refreshed
    # from the original source; streamed buffers reload on their next use.
    receipt = dict(original_source=original.identity(), d2_native_source=hybrid.source.identity(),
                   model_files={**model_files(original), **model_files(hybrid.source)},
                   tensor_hashes=hashes, tensor_proof_sha256=I.digest(hashes),
                   text_tensors=len(names), d2_applied_in_source=False, native_reader_retained=True)
    engine.source = original
    if engine.mode == "resident":
        for index, layer in enumerate(engine.layers):
            engine._load_into(layer, index, engine._fetch_state(index))
    hybrid._recovery_d4_rebase = receipt
    return receipt


def activate_native(hybrid, module):
    """Activate complete native replacement state with protected rows frozen."""
    import torch
    units = N.unit_parameters(hybrid)
    expected = {key for key in units if not any(key.endswith("." + suffix) for suffix in ("h3_time.curve_table", "attn.rope_inv_freq"))}
    music = getattr(hybrid, "_recovery_native_music_contract", None)
    if (not expected or expected != getattr(hybrid, "_recovery_native_unit_keys", set()) or not music
            or hybrid.band_embed_override is None or not hasattr(hybrid, "_recovery_music_head_master")):
        raise Error("D4 requires complete learned unit replacements and genuine music masters")
    groups = {"D3.unit": {key: units[key] for key in sorted(expected)}}
    for value in groups["D3.unit"].values():
        if value.dtype != torch.float32 or not value.is_leaf or value.grad is not None:
            raise Error("D4 unit master must remain an unused physical fp32 leaf")
    offset, rows, codes = (music[key] for key in ("offset", "rows", "code_rows"))
    embed, head = hybrid.band_embed_override, hybrid._recovery_music_head_master
    if (type(codes) is not int or not 1 <= codes <= rows or embed.shape[0] != offset + rows
            or list(head.shape) != [rows, hybrid.hidden]
            or I.tensor_sha(embed[:offset]) != music["ledger_sha256"]
            or I.tensor_sha(embed[offset + codes:]) != music["reserved_embed_sha256"]
            or I.tensor_sha(head[codes:]) != music["reserved_head_sha256"]):
        raise Error("D4 music ledger/reserved row protection differs")
    for key, value in groups["D3.unit"].items():
        value.requires_grad_(True)
    interfaces = D.activate_native_interfaces(module, hybrid)
    groups.update({"D3." + kind: named for kind, named in interfaces.items()})
    # Only 64000 genuine code rows are optimizer leaves; ledger/reserved rows
    # cannot receive decay/moments or updates. Keep host masters; copies are
    # differentiable. Build a fresh embedding graph for each actual microbatch.
    embed_code = torch.nn.Parameter(embed[offset:offset + codes].detach().float().cpu().contiguous().clone())
    head_code = torch.nn.Parameter(head[:codes].detach().float().cpu().contiguous().clone())
    ledger, reserved_embed = embed[:offset].detach().clone(), embed[offset + codes:].detach().clone()
    reserved_head = head[codes:].detach().clone()
    old_embed = hybrid.embed
    def dynamic_embed(self, ids):
        self.band_embed_override = torch.cat([ledger, embed_code, reserved_embed], dim=0)
        return old_embed(ids)
    hybrid.embed = types.MethodType(dynamic_embed, hybrid)
    hybrid._recovery_d4_music = dict(embed=embed_code, head=head_code, ledger=ledger,
                                   reserved_embed=reserved_embed, reserved_head=reserved_head, contract=music)
    # Frozen inherited hashes refer to complete masters, not the newly active
    # leaves. D4 has a new checkpoint/export contract and must not reuse D3's.
    hybrid._recovery_native_overlay_hashes = {}
    groups["D3.music_embed"] = {"music.embed": embed_code}
    groups["D3.music_head"] = {"music.head": head_code}
    return groups


def construct_union(module, hybrid, original, d2, binding, native, shared, rates, *, fixture=False):
    """Construct complete genuine union; production requires terminal exports."""
    if not fixture:
        nr, sr = native.get("receipt", {}), shared.get("receipt", {})
        if (set(native) != {"source", "receipt"} or set(shared) != {"source", "receipt"}
                or nr.get("source_manifest_sha256") != getattr(hybrid, "_recovery_native_parent", None)
                or sr.get("source_manifest_sha256") != getattr(hybrid, "_recovery_shared_r16_parent", None)
                or not nr.get("source_manifest_sha256") or not sr.get("source_manifest_sha256")
                or nr.get("context") != sr.get("context")
                or nr.get("initial_native_hashes_sha256") != I.digest(hybrid._recovery_native_initial_hashes)
                or nr.get("initial_overlay_hashes_sha256") != I.digest(hybrid._recovery_native_overlay_hashes)
                or sr.get("initial_bank_tensor_hashes_sha256") != I.digest(hybrid._recovery_shared_r16_initial_hashes)):
            raise Error("D4 needs authenticated terminal native and shared-bank inheritance receipts")
        # Re-read the actual exports, including their producer checkpoint pins.
        value = N.verify_entry(native["source"], Path("."), nr["context"], sr["source_manifest_sha256"])
        if value["manifest"]["stage"] != TERMINAL:
            raise Error("D4 terminal stage must include distinct audio CE then speech flow")
        bank = I.verify_entry(shared["source"], Path("."), sr["context"])
        if (value["manifest_sha256"] != nr["source_manifest_sha256"] or bank["manifest_sha256"] != sr["source_manifest_sha256"]
                or bank["manifest"]["producer"] != value["manifest"]["producer"]):
            raise Error("D4 terminal unit/music and bank must bind the same producing checkpoint")
    joint = D.JointAdapterBank(d2, hybrid.bank, original.identity(), binding, fixture=fixture)
    receipt = rebase_text(hybrid, original, d2)
    D.install_joint_bank(hybrid, joint)
    groups = activate_native(hybrid, module)
    for family in ("D2.lora", "D2.k3_lora", "D2.gain", "D3.shared_lora"):
        groups[family] = {}
    for name, value in joint.tensors.items():
        family = "D3.shared_lora" if name.startswith("d3.") else "D2.gain" if name.endswith(".gain") else "D2.k3_lora" if ".k3_" in name else "D2.lora"
        groups[family][name] = value
    rates = rates(groups) if callable(rates) else rates
    optimizer, ownership = D.fresh_union_optimizer(groups, rates)
    return dict(hybrid=hybrid, groups=groups, optimizer=optimizer, ownership=ownership,
                identity=dict(schema=SCHEMA, fixture=fixture, terminal_stage=TERMINAL,
                              native=native, shared=shared, d2=binding, rebase=receipt, ownership=ownership,
                              previous_rates=rates, runtime_sha256=I.sha(Path(__file__)),
                              helper_sha256=I.sha(Path(D.__file__)), native_sha256=I.sha(Path(N.__file__)),
                              dense_host_music_adam=True, production_ready=False))


def replay_identity(rows, sources, root):
    """Admit pinned train-only rows; sampling is one physical pass in row order."""
    if not rows or len({row.get("id") for row in rows}) != len(rows) or not sources:
        raise Error("D4 replay needs unique real rows and actual pinned source files")
    for ref in sources.values():
        I.pinned(ref, root)
    for row in rows:
        if (not row.get("id") or not row.get("recovery_group_id") or row.get("split") != "train"
                or row.get("held") is not False or row.get("train_admitted") is not True
                or row.get("blockers") or not row.get("objective")
                or type(row.get("supervised_units")) is not int or row["supervised_units"] < 1
                or type(row.get("objective_weight")) not in (float, int) or not math.isfinite(row["objective_weight"]) or row["objective_weight"] <= 0
                or not row.get("active_families") or set(row["active_families"]) - D.FAMILIES):
            raise Error("D4 replay row is not admitted train-only supervised data")
        for name in ("input", "target"):
            I.pinned(row[name], root)
    return dict(rows=rows, sources=sources, row_order_sha256=I.digest(rows), exclusive_one_pass=True)


class UnionEngine:
    """Actual optimizer/cursor/checkpoint core; native objectives stay explicit.

    Calls one trusted native backward callback per immutable admitted row. A
    nonfinite window does not step or consume rows. Dense host music AdamW is
    a declared candidate implementation; its measured host budget and exact
    replay/objective/lease dispatcher remain prerequisites for production.
    """
    def __init__(self, union, rows, sources, root, *, accumulation=1, clip_norm=1.):
        import torch
        if type(accumulation) is not int or accumulation < 1 or not math.isfinite(clip_norm) or clip_norm <= 0:
            raise Error("D4 requires valid accumulation and clipping")
        self.hybrid, self.optimizer = union["hybrid"], union["optimizer"]
        self.census = {name: tensor for named in union["groups"].values() for name, tensor in named.items()}
        self.owners = union["ownership"]["owners"]
        if self.optimizer.state or any(value.grad is not None for value in self.census.values()):
            raise Error("D4 starts with fresh Adam state and unused leaves")
        replay = replay_identity(copy.deepcopy(rows), sources, root)
        self.identity = {**union["identity"], "replay": replay, "accumulation": accumulation, "clip_norm": clip_norm,
                         "loss_normalization": "per_objective_supervised_units_then_declared_weight"}
        self.rows, self.root = replay["rows"], Path(root)
        self.accumulation, self.clip_norm, self.cursor, self.step = accumulation, clip_norm, 0, 0
        self.optimizer_steps = {name: 0 for name in self.census}
        self.total_steps = (len(rows) + accumulation - 1) // accumulation
        self.initial_frozen = {key: I.tensor_sha(value) for key, value in N.unit_parameters(self.hybrid).items() if key not in self.census}
        # Compact active deltas deliberately share native names with their
        # frozen full base matrices. Their base still needs its own guard.
        self.initial_frozen.update({key: I.tensor_sha(value) for key, value in N.fixed_parameters(self.hybrid).items()})
        self.initial_native = N.native_inventory(self.hybrid)

    def verify_sources(self):
        for ref in self.identity["rebase"]["model_files"].values():
            I.pinned(ref, self.root)
        replay_identity(self.rows, self.identity["replay"]["sources"], self.root)
        if I.digest(self.rows) != self.identity["replay"]["row_order_sha256"]:
            raise Error("D4 ordered source rows changed")

    def window(self, backward, *, before_apply=None):
        import torch
        if not self.identity["fixture"] and not callable(before_apply):
            raise Error("D4 production requires a live update gate at the actual optimizer boundary")
        self.verify_sources()
        if self.cursor >= len(self.rows):
            raise Error("D4 exclusive replay already consumed")
        end = min(self.cursor + self.accumulation, len(self.rows))
        rows = self.rows[self.cursor:end]
        rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
        def rejected(reason):
            self.optimizer.zero_grad(set_to_none=True)
            torch.random.set_rng_state(rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
            return dict(applied=False, reason=reason, cursor=self.cursor, step=self.step)
        self.optimizer.zero_grad(set_to_none=True)
        denominators, weights = {}, {}
        for row in rows:
            objective = row["objective"]
            if objective in weights and weights[objective] != row["objective_weight"]:
                raise Error("D4 objective weight differs inside one accumulation window")
            weights[objective] = row["objective_weight"]
            denominators[objective] = denominators.get(objective, 0) + row["supervised_units"]
        stats = []
        for row in rows:
            # Callback returns a scalar diagnostic, and performs the authentic
            # backward of its mean loss scaled by the measured supervised
            # count over its entire objective's accumulation-window denominator.
            weight = row["objective_weight"] * row["supervised_units"] / denominators[row["objective"]]
            try:
                value = float(backward(self.hybrid, row, weight))
            except BaseException:
                rejected("objective_error")
                raise
            if not math.isfinite(value):
                return rejected("nonfinite_loss")
            stats.append(value)
        expected = set().union(*(set(row["active_families"]) for row in rows))
        present = {self.owners[name] for name, value in self.census.items() if value.grad is not None}
        if not expected <= present:
            rejected("missing_active_family")
            raise Error("D4 native objective omitted an active optimizer family")
        if any(value.grad is not None and not bool(torch.isfinite(value.grad).all()) for value in self.census.values()):
            return rejected("nonfinite_gradient")
        norm = torch.nn.utils.clip_grad_norm_(list(self.census.values()), self.clip_norm)
        if not bool(torch.isfinite(norm)):
            return rejected("nonfinite_norm")
        try:
            if before_apply is not None:
                before_apply()
        except BaseException:
            rejected("update_gate_refused")
            raise
        self.optimizer.step()
        for name, value in self.census.items():
            if value.grad is not None:
                self.optimizer_steps[name] += 1
        self.cursor, self.step = end, self.step + 1
        self.optimizer.zero_grad(set_to_none=True)
        return dict(applied=True, cursor=self.cursor, step=self.step, rows=len(rows), mean_row_loss=sum(stats) / len(stats),
                    supervised_denominators=denominators, objective_weights=weights, grad_norm=float(norm))

    def verify_frozen(self):
        for key, value in {**N.unit_parameters(self.hybrid), **N.fixed_parameters(self.hybrid)}.items():
            if key in self.initial_frozen and I.tensor_sha(value) != self.initial_frozen[key]:
                raise Error("D4 frozen native coefficient/base interface changed")
        music = self.hybrid._recovery_d4_music
        for name, label in (("ledger", "ledger_sha256"), ("reserved_embed", "reserved_embed_sha256"), ("reserved_head", "reserved_head_sha256")):
            if I.tensor_sha(music[name]) != music["contract"][label]:
                raise Error("D4 frozen music ledger/reserved rows changed")

    def checkpoint(self, out):
        import torch
        out = Path(out)
        if out.exists():
            raise Error("preserve previous D4 checkpoint")
        self.verify_sources()
        self.verify_frozen()
        if self.cursor != min(self.step * self.accumulation, len(self.rows)):
            raise Error("D4 checkpoint sampler cursor disagrees with applied windows")
        if any(not bool(torch.isfinite(value).all()) for value in self.census.values()):
            raise Error("D4 cannot save nonfinite trainables")
        state = dict(schema=SCHEMA, identity=self.identity, step=self.step, cursor=self.cursor, total_steps=self.total_steps,
                     trainables={name: value.detach().cpu().clone() for name, value in self.census.items()},
                     optimizer=self.optimizer.state_dict(), optimizer_steps=self.optimizer_steps, rng=torch.random.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=out.name + ".", dir=out.parent) as temporary:
            directory = Path(temporary)
            torch.save(state, directory / "state.pt")
            receipt = dict(schema=SCHEMA, identity_sha256=I.digest(self.identity), state_sha256=I.sha(directory / "state.pt"),
                           complete=True, step=self.step, cursor=self.cursor, stage_training_complete=self.cursor == len(self.rows),
                           production_ready=False, fixture=self.identity["fixture"])
            (directory / "receipt.json").write_bytes(I.canonical(receipt) + b"\n")
            directory.rename(out)
        return I.pin(out / "receipt.json")

    def resume(self, checkpoint):
        import torch
        checkpoint = Path(checkpoint)
        self.verify_sources()
        receipt = json.loads((checkpoint / "receipt.json").read_text(encoding="utf-8"))
        if (receipt.get("complete") is not True or receipt.get("identity_sha256") != I.digest(self.identity)
                or receipt.get("state_sha256") != I.sha(checkpoint / "state.pt")):
            raise Error("D4 resume receipt/source/identity differs")
        state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
        step, cursor = state.get("step"), state.get("cursor")
        if (state.get("schema") != SCHEMA or state.get("identity") != self.identity or state.get("total_steps") != self.total_steps
                or type(step) is not int or not 0 <= step <= self.total_steps
                or cursor != min(step * self.accumulation, len(self.rows)) or receipt.get("step") != step or receipt.get("cursor") != cursor
                or set(state.get("trainables", {})) != set(self.census)):
            raise Error("D4 resume exact sampler/optimizer/census state differs")
        values = state["trainables"]
        if any(value.dtype != torch.float32 or value.shape != self.census[name].shape or not bool(torch.isfinite(value).all()) for name, value in values.items()):
            raise Error("D4 resume nonfinite/shape/dtype trainable")
        saved_optimizer = state["optimizer"]
        if len(saved_optimizer["param_groups"]) != len(self.optimizer.param_groups):
            raise Error("D4 resume optimizer groups differ")
        actual = self.optimizer.state_dict()
        if any({k: v for k, v in a.items() if k != "params"} != {k: v for k, v in b.items() if k != "params"}
               or a["params"] != b["params"] for a, b in zip(saved_optimizer["param_groups"], actual["param_groups"])):
            raise Error("D4 resume optimizer ownership/rates differ")
        if any(not bool(torch.isfinite(value).all()) for record in saved_optimizer["state"].values() for value in record.values() if isinstance(value, torch.Tensor)):
            raise Error("D4 resume optimizer state nonfinite")
        counters = state.get("optimizer_steps", {})
        if set(counters) != set(self.census) or any(type(value) is not int or not 0 <= value <= step for value in counters.values()):
            raise Error("D4 per-owner optimizer counters differ")
        tensor_names = {id(value): name for name, value in self.census.items()}
        for saved, physical in zip(saved_optimizer["param_groups"], self.optimizer.param_groups):
            for pid, value in zip(saved["params"], physical["params"]):
                count, record = counters[tensor_names[id(value)]], saved_optimizer["state"].get(pid)
                if (count == 0 and record) or (count > 0 and (not record or set(record) != {"step", "exp_avg", "exp_avg_sq"}
                        or float(record["step"]) != count or record["exp_avg"].shape != value.shape or record["exp_avg_sq"].shape != value.shape)):
                    raise Error("D4 applied per-owner counters disagree with actual Adam state")
        if state["cuda_rng"] and not torch.cuda.is_initialized():
            raise Error("D4 CUDA checkpoint cannot resume into a CPU-only runtime")
        # All admissions precede tensor or optimizer mutation; preserve actual
        # leaf object identity for the joint hooks and native/music references.
        with torch.no_grad():
            for name, value in self.census.items():
                value.copy_(values[name].to(value.device))
        self.optimizer.load_state_dict(saved_optimizer)
        self.optimizer_steps = dict(counters)
        self.step, self.cursor = step, cursor
        torch.random.set_rng_state(state["rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
