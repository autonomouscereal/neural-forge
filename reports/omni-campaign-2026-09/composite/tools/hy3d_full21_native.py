"""Source-faithful FULL21 load/replay and complete native branch installation.

The dense union is a storage/ownership contract. Native expert routing, grouping,
mass multiplication, scatter-reduce and shared addition execute unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
import c16_capture_hy3d as capture
import c16_replay_hy3d as dense_replay
from c16_hy3d_contract import CAPTURE_DIMS, inspect_source, tensor_header, bounded_bytes, unique_json, stamp, validate_capture
from hy3d_full21_contract import block_contract, require, union_manifest, validate_union, FOLD_SCHEMA
from hy3d_full21_linalg import scores


def bit_exact(actual, reference):
    return actual.dtype == reference.dtype and actual.shape == reference.shape and torch.equal(
        actual.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8))


def tensor_identity(value):
    value = capture.cpu(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype), "elements": value.numel(),
            "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def native_components(native, config, index):
    contract = block_contract(index)
    p = config["model"]["params"]
    require(p["depth"] == 21 and p["num_moe_layers"] == 6 and p["num_experts"] == 8 and p["moe_top_k"] == 2 and
            not p["use_pos_emb"] and not p["use_attention_pooling"] and not p["with_decoupled_ca"] and
            p["hidden_size"] == 2048 and p["num_heads"] == 16 and p["context_dim"] == 1024,
            "exact audited native FULL21 geometry required")
    dit, width = native["dit"], p["hidden_size"]
    # Local construction context avoids default F32 random parameter allocation.
    # It changes no native forward class or source; strict assign materializes
    # every persistent state tensor and rejects remaining meta parameters/buffers.
    with torch.device("meta"):
        block = dit.HunYuanDiTBlock(hidden_size=width, c_emb_size=width, num_heads=p["num_heads"],
            text_states_dim=p["context_dim"], qk_norm=p["qk_norm"], norm_layer=torch.nn.LayerNorm,
            qk_norm_layer=torch.nn.RMSNorm, qkv_bias=p["qkv_bias"], skip_connection=contract["skip_required"],
            with_decoupled_ca=False, timested_modulate=False, use_moe=contract["use_moe"],
            num_experts=8, moe_top_k=2)
        time = dit.TimestepEmbedder(width, width * 4, cond_proj_dim=None)
        embedding = torch.nn.Linear(p["in_channels"], width, bias=True)
    return block, time, embedding


def strict_load(module, parent, prefix, device):
    loaded = capture.strict_component(module, parent, prefix, device)
    require(not any(t.is_meta for t in (*loaded.parameters(), *loaded.buffers())),
            "native component has an unmaterialized parameter/buffer")
    return loaded


def branch_modules(block, index):
    contract = block_contract(index)
    require(block.use_moe == contract["use_moe"] and not block.timested_modulate and
            (block.skip_linear is not None) == contract["skip_required"], "native variant/skip changed")
    result = []
    if block.use_moe:
        moe = block.moe
        require(len(moe.experts) == 8 and moe.moe_top_k == 2 and moe.gate.n_routed_experts == 8 and
                moe.gate.top_k == 2 and moe.gate.scoring_func == "softmax" and not moe.gate.norm_topk_prob,
                "native eight-expert top2 unnormalized router required")
        modules = [*moe.experts, moe.shared_experts]
        for info, ff in zip(contract["branches"], modules):
            require(len(ff.net) == 3 and ff.net[0].__class__.__name__ == "GELU" and
                    getattr(ff.net[0], "approximate", None) == "none" and
                    isinstance(ff.net[1], torch.nn.Dropout) and ff.net[1].p == 0,
                    "native exact biased diffusers GELU FFN required")
            first, second = ff.net[0].proj, ff.net[2]
            require(set(ff.state_dict()) == {"net.0.proj.weight", "net.0.proj.bias", "net.2.weight", "net.2.bias"},
                    "unexpected native FFN computational branch/state")
            result.append((info, ff, first, second))
    else:
        mlp = block.mlp
        require(isinstance(mlp.gelu, torch.nn.GELU) and mlp.gelu.approximate == "none" and
                set(mlp.state_dict()) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"},
                "native exact biased dense GELU MLP required")
        result.append((contract["branches"][0], mlp, mlp.fc1, mlp.fc2))
    for _, _, first, second in result:
        require(isinstance(first, torch.nn.Linear) and isinstance(second, torch.nn.Linear) and
                first.bias is not None and second.bias is not None and
                first.in_features == second.out_features == 2048 and first.out_features == second.in_features,
                "native biased branch geometry mismatch")
    return result


def load_capture(run_directory, index, source):
    """Extend the historical sealed join to all21; exact router I64 is retained."""
    contract = block_contract(index)
    root = Path(run_directory).resolve(strict=True)
    run_path = root / "capture-run.json"
    run = unique_json(bounded_bytes(run_path))
    require(isinstance(run, dict) and run.get("schema") == capture.SCHEMA and
            run.get("source_identity") == source["source_identity"] and run.get("producer") == capture.PRODUCER,
            "matching actual native capture run required")
    directory = root / f"block-{index:02d}"
    checked = validate_capture(directory, source)
    body = unique_json(bounded_bytes(directory / "capture.json"))
    require(body.get("kind") == "dit_block" and body.get("block_index") == index and run.get("dit_capture") == checked,
            "capture-run/block validation disagreement")
    rows = run.get("samples")
    require(isinstance(rows, list) and len(rows) == 4 and len(body["samples"]) == 4, "exact 2FIT/2HELD complete native contexts required")
    tracked = {run_path: capture.digest_file(run_path), directory / "capture.json": capture.digest_file(directory / "capture.json")}
    samples, geometry_splits = [], {}
    for i, (row, block_row) in enumerate(zip(rows, body["samples"])):
        require(all(row.get(k) == block_row.get(k) for k in
                ("object_sha256", "conditioning_sha256", "seed", "split", "geometry_identity")), "object/image/split order changed")
        geometry = row.get("geometry_identity", {})
        require(geometry.get("schema") == capture.GEOMETRY_SCHEMA and
                geometry_splits.setdefault(geometry.get("sha256"), row["split"]) == row["split"], "geometry split ownership changed")
        sample_path = dense_replay.sealed_path(root, row.get("file"))
        require(tensor_header(sample_path)["bytes"] <= 512 << 20 and capture.digest_file(sample_path) == row.get("file_sha256"),
                "bounded sealed sample identity mismatch")
        sample_json = root / f"sample-{i:04d}.json"
        require(unique_json(bounded_bytes(sample_json)) == row, "sample receipt mismatch")
        tracked.update({sample_path: row["file_sha256"], sample_json: capture.digest_file(sample_json)})
        with safe_open(str(sample_path), framework="pt", device="cpu") as handle:
            required = {"self_attention_write", "cross_attention_write", "dit_latent_input", "surface", "query_source_index", "anchor_xyz"}
            require(required <= set(handle.keys()), "missing genuine native per-object evidence")
            samples.append({name: handle.get_tensor(name) for name in required})
    path = dense_replay.sealed_path(directory, body["file"])
    tracked[path] = body["file_sha256"]
    names = {"block_input", "block_output", "context", "sigma", "timestep", "time_token", "mlp_input", "mlp_output"}
    if contract["skip_required"]:
        names.add("skip_value")
    if contract["use_moe"]:
        names.update(("router_logits", "router_topk_ids", "router_topk_weights", "shared_output"))
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in names}
        for sample_name, combined_name in (("surface", "surface"), ("query_source_index", "query_source_index"), ("anchor_xyz", "latent_xyz")):
            require(bit_exact(torch.cat([sample[sample_name] for sample in samples]), handle.get_tensor(combined_name)),
                    "combined/per-object source surface/anchor ownership changed")
    for name in ("self_attention_write", "cross_attention_write", "dit_latent_input"):
        tensors[name] = torch.cat([sample[name] for sample in samples])
    for i, row in enumerate(rows):
        require(bit_exact(torch.tensor([row["requested_sigma"]], dtype=torch.float16), tensors["sigma"][i:i + 1]), "source sigma changed")
    for name, value in tensors.items():
        require(value.dtype == (torch.int64 if name == "router_topk_ids" else torch.float16), "native capture dtype changed: " + name)
        capture.finite(value, name)
    require(tensors["dit_latent_input"].shape == (4, 4096, 64) and
            tensors["self_attention_write"].shape == tensors["cross_attention_write"].shape == (4, 4097, 2048),
            "native full context missing")
    return tensors, rows, tracked, body


@torch.inference_mode()
def replay_one(block, time, embedding, tensors, row_index, index, device, fit_features=None):
    contract = block_contract(index)
    branches = branch_modules(block, index)
    require(not any(module.training or any(p.requires_grad for p in module.parameters()) for module in (block, time, embedding)),
            "native replay must be frozen/eval")
    row = {name: tensor[row_index:row_index + 1].to(device) for name, tensor in tensors.items()}
    observed, feature_trace = {}, {}
    def put(name, value):
        require(name not in observed, "unexpected repeated native replay hook: " + name)
        capture.finite(value, name)
        observed[name] = capture.cpu(value)
    ffn = block.moe if contract["use_moe"] else block.mlp
    with ExitStack() as hooks:
        for name, module in (("self_attention_write", block.attn1), ("cross_attention_write", block.attn2), ("mlp_output", ffn)):
            hooks.callback(module.register_forward_hook(lambda m, a, out, name=name: put(name, out)).remove)
        hooks.callback(ffn.register_forward_pre_hook(lambda m, a: put("mlp_input", a[0])).remove)
        if contract["skip_required"]:
            hooks.callback(block.skip_linear.register_forward_pre_hook(lambda m, a: put("skip_concat", a[0])).remove)
        if contract["use_moe"]:
            gate = ffn.gate
            namespace, native_f = gate.forward.__func__.__globals__, gate.forward.__func__.__globals__["F"]
            require(not isinstance(native_f, capture.Proxy), "nested native gate observation")
            def linear(x, weight, *args, **kwargs):
                out = native_f.linear(x, weight, *args, **kwargs)
                if weight is gate.weight:
                    put("router_logits", out.reshape(1, row["block_input"].shape[1], 8))
                return out
            namespace["F"] = capture.Proxy(native_f, linear=linear)
            hooks.callback(namespace.__setitem__, "F", native_f)
            def gate_out(m, a, out):
                put("router_topk_ids", out[0].reshape(1, -1, 2))
                put("router_topk_weights", out[1].reshape(1, -1, 2))
            hooks.callback(gate.register_forward_hook(gate_out).remove)
            hooks.callback(ffn.shared_experts.register_forward_hook(lambda m, a, out: put("shared_output", out)).remove)
        if fit_features is not None:
            for info, ff, first, second in branches:
                branch = info["id"]
                feature_trace[branch] = {}
                def save(which, value, branch=branch):
                    require(which not in feature_trace[branch], "native branch repeated; grouping changed")
                    feature_trace[branch][which] = capture.cpu(value).reshape(-1, value.shape[-1])
                hooks.callback(second.register_forward_pre_hook(lambda m, a, save=save: save("hidden", a[0])).remove)
                # Hook is before native in-place expert mass multiplication.
                hooks.callback(ff.register_forward_hook(lambda m, a, out, save=save: save("target", out)).remove)
        token = time(row["sigma"], condition=None)
        put("time_token", token)
        if index == 0:
            put("initial_block_input", torch.cat((token, embedding(row["dit_latent_input"])), dim=1))
        output = block(row["block_input"], token, row["context"], skip_value=row.get("skip_value"))
        put("block_output", output)
    references = {name: tensors[name][row_index:row_index + 1] for name in observed if name not in ("initial_block_input", "skip_concat")}
    if index == 0:
        references["initial_block_input"] = tensors["block_input"][row_index:row_index + 1]
    if contract["skip_required"]:
        references["skip_concat"] = torch.cat((tensors["skip_value"][row_index:row_index + 1],
                                               tensors["block_input"][row_index:row_index + 1]), dim=-1)
    metrics = {name: {"byte_exact": bit_exact(observed[name], ref),
                     **({} if ref.dtype == torch.int64 else scores(observed[name].reshape(-1, ref.shape[-1]),
                                                                    ref.reshape(-1, ref.shape[-1])))}
               for name, ref in references.items()}
    if fit_features is not None:
        for info, _, _, _ in branches:
            trace = feature_trace[info["id"]]
            require(not trace or set(trace) == {"hidden", "target"}, "partial native branch observation")
            if trace:
                fit_features[info["id"]].append(trace)
    return observed, metrics


@contextmanager
def installed_union(block, index, states):
    """Restore original linears even on error; retain actual FFN/MoE classes."""
    branches = branch_modules(block, index)
    require(set(states) == {info["id"] for info, *_ in branches}, "complete dense union branch state required")
    restore = []
    try:
        for info, ff, first, second in branches:
            state = states[info["id"]]
            require(set(state) == {"first.weight", "first.bias", "second.weight", "second.bias"}, "both biased linears required")
            keep, width = state["first.weight"].shape
            require(width == 2048 and 0 < keep <= 8192 and
                    state["first.bias"].shape == (keep,) and state["second.weight"].shape == (width, keep) and
                    state["second.bias"].shape == (width,), "invalid dense union branch geometry")
            linears = []
            for name, shape in (("first", (width, keep)), ("second", (keep, width))):
                for suffix in ("weight", "bias"):
                    value = state[name + "." + suffix]
                    require(value.dtype == torch.float16, "union native F16 storage required")
                    capture.finite(value, "saved branch " + name + suffix)
                with torch.device("meta"):
                    linear = torch.nn.Linear(*shape, bias=True)
                linear.load_state_dict({key: state[name + "." + key] for key in ("weight", "bias")}, strict=True, assign=True)
                linear.to(device=first.weight.device, dtype=torch.float16).eval().requires_grad_(False)
                linears.append(linear)
            restore.append((ff, first, second))
            if block.use_moe:
                ff.net[0].proj, ff.net[2] = linears
            else:
                ff.fc1, ff.fc2 = linears
        yield union_manifest(index, {info["id"]: states[info["id"]]["first.weight"].shape[0] for info, *_ in branches})
    finally:
        for ff, first, second in restore:
            if block.use_moe:
                ff.net[0].proj, ff.net[2] = first, second
            else:
                ff.fc1, ff.fc2 = first, second


def component_identity(block, time, embedding, index):
    return {prefix + name: tensor_identity(value) for prefix, module in
            ((f"model.blocks.{index}.", block), ("model.t_embedder.", time), ("model.x_embedder.", embedding))
            for name, value in module.state_dict().items()}


def load_saved_union(receipt_path, source, body, rows):
    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt = unique_json(bounded_bytes(receipt_path, 16 << 20))
    require(receipt.get("schema") == FOLD_SCHEMA and receipt.get("source_identity") == source["source_identity"] and
            receipt.get("block_index") == body["block_index"] and receipt.get("capture_sha256") == body["file_sha256"] and
            receipt.get("fit_object_sha256") == sorted(row["object_sha256"] for row in rows if row["split"] == "fit"),
            "saved dense union source/capture/FIT identity mismatch")
    manifest = validate_union(receipt.get("union"))
    path = dense_replay.sealed_path(receipt_path.parent, receipt.get("file"))
    require(tensor_header(path)["bytes"] <= 768 << 20 and capture.digest_file(path) == receipt.get("file_sha256"),
            "saved dense union hash/bound mismatch")
    states = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        expected = {branch["id"] + "." + name for branch in manifest["branches"]
                    for name in ("first.weight", "first.bias", "second.weight", "second.bias")}
        require(set(handle.keys()) == expected, "saved dense union missing/added source branch tensors")
        require(set(receipt.get("tensor_identity", {})) == expected, "saved tensor readback identity inventory incomplete")
        for branch in manifest["branches"]:
            state = {}
            for name in ("first.weight", "first.bias", "second.weight", "second.bias"):
                key = branch["id"] + "." + name
                value = handle.get_tensor(key)
                require(tensor_identity(value) == receipt["tensor_identity"][key], "saved dense union readback mismatch")
                state[name] = value
            require(state["first.weight"].shape[0] == branch["width"], "dense union branch width/readback mismatch")
            states[branch["id"]] = state
    return states, receipt, {receipt_path: capture.digest_file(receipt_path), path: receipt["file_sha256"]}
