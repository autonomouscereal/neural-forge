"""Source-complete sharded FULL21 dense child serializer and strict reload.

Every original non-FFN tensor is physically retained. Original router/expert
weights are replaced explicitly, not falsely reported as dense native fidelity.
The immutable donor checkpoint remains an external protected source artifact.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
import c16_capture_hy3d as capture
from hy3d_full21_contract import require, source_groups
from hy3d_full21_native import tensor_identity
from hy3d_dense_child_contract import SCHEMA, STATE_NAMES, coverage21
from hy3d_dense_child_native import DenseChild

BUNDLE_SCHEMA = "chimera.hy3d-full21-source-complete-dense-bundle.v1"


def ffn_prefix(index):
    return f"model.blocks.{index}." + ("moe." if index >= 15 else "mlp.")


def materialize_fourier_buffer(vae, config):
    params = config["vae"]["params"]
    require(params.get("num_freqs") == 8 and params.get("include_pi") is False,
            "only audited native Fourier factory configuration supported")
    original = vae.fourier_embedder
    require(type(original).__name__ == "FourierEmbedder" and original.num_freqs == 8 and original.include_input is True and
            original.out_dim == 51 and original.frequencies.shape == (8,) and
            vae.encoder.fourier_embedder is original and vae.geo_decoder.fourier_embedder is original and
            "frequencies" in original._non_persistent_buffers_set, "native shared Fourier buffer topology changed")
    with torch.device("cpu"):
        reference = type(original)(num_freqs=8, logspace=True, input_dim=3, include_input=True, include_pi=False)
    require(not reference.frequencies.is_meta and reference.frequencies.dtype == torch.float32,
            "native deterministic Fourier factory did not materialize")
    original.frequencies = reference.frequencies.detach().clone()
    return {"native_factory_arguments": {"num_freqs": 8, "logspace": True, "input_dim": 3,
             "include_input": True, "include_pi": False}, "buffer_identity_before_native_f16_move": tensor_identity(original.frequencies),
             "native_shared_aliases_preserved": True, "checkpoint_tensor": False, "trained_weight": False, "random_init": False}


def pack(parent, source, receipt_paths, output, shard_cap=256 << 20):
    require(type(shard_cap) is int and 0 < shard_cap <= 512 << 20, "bounded shard cap required")
    parent = Path(parent).resolve(strict=True); stamp = capture.stamp(parent)
    header = capture.tensor_header(parent); groups = source_groups(header["tensors"])
    require(header["header_sha256"] == source["source_identity"]["checkpoint_header_sha256"] and
            header["bytes"] == source["source_identity"]["checkpoint_bytes"], "donor header/bytes differ from inspected source")
    paths = [Path(p).resolve(strict=True) for p in receipt_paths]
    receipt_raw = [capture.bounded_bytes(p, 16 << 20) for p in paths]
    receipt_pins = [hashlib.sha256(raw).hexdigest() for raw in receipt_raw]
    rows = [capture.unique_json(raw) for raw in receipt_raw]
    coverage21(rows); require(rows[0]["source_identity"] == source["source_identity"], "bundle donor identity mismatch")
    out = Path(output).resolve(); require(not out.exists(), "fresh bundle output required"); out.mkdir(parents=True)
    prefixes = [ffn_prefix(i) for i in range(21)]
    require(all(any(n.startswith(p) for n in header["tensors"]) for p in prefixes), "donor FFN source coverage incomplete")
    retained = [n for n in sorted(header["tensors"]) if not any(n.startswith(p) for p in prefixes)]
    shards, index, identities, state, used = [], {}, {}, {}, 0
    def flush():
        nonlocal state, used
        if not state: return
        artifact = capture.sealed_tensors(out / f"weights-{len(shards):04d}.safetensors", state)
        with safe_open(str(out / artifact["file"]), framework="pt", device="cpu") as check:
            require(set(check.keys()) == set(state) and all(tensor_identity(check.get_tensor(n)) == identities[n] for n in state), "bundle shard byte readback mismatch")
        shards.append(artifact); state = {}; used = 0
    def add(name, value):
        nonlocal used
        size = value.numel() * value.element_size()
        require(name not in index and size <= shard_cap, "duplicate/overlength source tensor")
        if used + size > shard_cap: flush()
        index[name] = f"weights-{len(shards):04d}.safetensors"; identities[name] = tensor_identity(value)
        state[name] = value; used += size
    with safe_open(str(parent), framework="pt", device="cpu") as handle:
        original_ffn_names = {n for n in header["tensors"] if any(n.startswith(p) for p in prefixes)}
        original_ids = {}
        for row in rows:
            require(isinstance(row.get("source_tensor_identity"), dict), "actual original block tensor identities required")
            for name, identity in row["source_tensor_identity"].items():
                if name in original_ids: require(original_ids[name] == identity, "inconsistent original source identity")
                original_ids[name] = identity
        require(original_ffn_names <= set(original_ids), "original FFN body identity coverage incomplete")
        for name, identity in original_ids.items():
            require(name in header["tensors"] and tensor_identity(handle.get_tensor(name)) == identity, "original source body differs from actual fold")
        for name in retained: add(name, handle.get_tensor(name))
    for path, row in zip(paths, rows):
        require(row.get("schema") == SCHEMA and row.get("stored_tensor_readback_verified") is True, "saved actual dense child required")
        name = row.get("file"); require(isinstance(name, str) and Path(name).name == name and name.endswith(".safetensors"), "unsafe child file")
        artifact = path.parent / name
        require(not artifact.is_symlink() and capture.digest_file(artifact) == row.get("file_sha256"), "child artifact hash mismatch")
        with safe_open(str(artifact), framework="pt", device="cpu") as handle:
            require(set(handle.keys()) == STATE_NAMES and set(row.get("tensor_identity", {})) == STATE_NAMES, "child tensor inventory mismatch")
            child_state = {n: handle.get_tensor(n) for n in STATE_NAMES}
        require(all(tensor_identity(v) == row["tensor_identity"][n] for n, v in child_state.items()), "child body changed")
        DenseChild(child_state, 2048, row["dense_child"]["keep"])
        for name, value in sorted(child_state.items()): add(ffn_prefix(row["block_index"]) + name, value)
    flush(); require(capture.stamp(parent) == stamp and all(capture.digest_file(p) == h for p, h in zip(paths, receipt_pins)), "donor/receipt changed during serialization")
    expected = set(retained) | {ffn_prefix(i) + n for i in range(21) for n in STATE_NAMES}
    require(set(index) == expected, "source-complete tensor accounting mismatch")
    body = {"schema": BUNDLE_SCHEMA, "source_identity": source["source_identity"], "source_groups": groups,
            "dense_child_contracts": [r["dense_child"] for r in rows], "shards": shards, "weight_map": index,
            "tensor_identity": identities, "retained_non_ffn_tensors": retained,
            "retained_non_ffn_body_identity": {n: identities[n] for n in retained},
            "child_receipts": [{"path": str(p), "sha256": h} for p, h in zip(paths, receipt_pins)],
            "original_ffn_disposition": "replaced by explicit approximate dense children; immutable original donor retained externally",
            "physical_non_ffn_readback_verified": True, "whole_native_load_forward_proven": False,
            "capability_claim": False, "training": False}
    capture.sealed_json(out / "bundle.json", body); return body


def load_components(bundle_path, modules, config, device="cpu"):
    path = Path(bundle_path).resolve(strict=True); body = capture.unique_json(capture.bounded_bytes(path, 64 << 20))
    require(body.get("schema") == BUNDLE_SCHEMA, "wrong full21 bundle schema")
    contracts = body.get("dense_child_contracts")
    require(isinstance(contracts, list) and [c.get("block_index") for c in contracts] == list(range(21)), "all21 load contract required")
    from hy3d_dense_child_contract import validate_child
    for contract in contracts: validate_child(contract)
    with torch.device("meta"):
        model = modules["dit"].HunYuanDiTPlain(**config["model"]["params"])
        vae = modules["vae"].ShapeVAE(**config["vae"]["params"])
        conditioner = modules["conditioner"].SingleImageEncoder(**config["conditioner"]["params"])
    require(len(model.blocks) == 21, "native all21 model required")
    # Replace the FFN modules before strict full-state matching, without random
    # runtime tensors. Actual child states are assigned from the sealed shards.
    for i, contract in enumerate(contracts):
        child = DenseChild.meta(2048, contract["keep"])
        setattr(model.blocks[i], "moe" if i >= 15 else "mlp", child)
    components = {"model": model, "vae": vae, "conditioner": conditioner}
    derived_fourier = materialize_fourier_buffer(vae, config)
    expected = {prefix + "." + n for prefix, module in components.items() for n in module.state_dict()}
    shapes = {prefix + "." + n: list(value.shape) for prefix, module in components.items() for n, value in module.state_dict().items()}
    require(set(body.get("weight_map", {})) == expected and set(body.get("tensor_identity", {})) == expected, "complete native state inventory mismatch")
    shards = body.get("shards")
    require(isinstance(shards, list) and 1 <= len(shards) <= 256 and len({s.get("file") for s in shards}) == len(shards), "bounded unique shard inventory required")
    seen = set()
    for shard in shards:
        name = shard.get("file"); require(isinstance(name, str) and Path(name).name == name and name.endswith(".safetensors"), "unsafe bundle shard")
        file = path.parent / name; require(not file.is_symlink() and capture.digest_file(file) == shard.get("file_sha256"), "bundle shard hash changed")
        header = capture.tensor_header(file)
        require(header["bytes"] <= (512 << 20) + (8 << 20), "bounded shard payload required")
        require(set(header["tensors"]) == {key for key, value in body["weight_map"].items() if value == name}, "shard/header inventory differs")
        require(all(item["dtype"] == "F16" and item["shape"] == shapes[key] for key, item in header["tensors"].items()), "actual native shard shape/F16 mismatch")
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            states = {p: {} for p in components}
            for key in handle.keys():
                require(key not in seen and body["weight_map"].get(key) == name, "duplicate/wrong shard tensor")
                value = handle.get_tensor(key); require(tensor_identity(value) == body["tensor_identity"][key], "bundle tensor body changed")
                prefix, local = key.split(".", 1); states[prefix][local] = value; seen.add(key)
            for prefix, state in states.items():
                result = components[prefix].load_state_dict(state, strict=False, assign=True)
                require(not result.unexpected_keys, "unexpected native state key")
    require(seen == expected, "missing assembled native state")
    for module in components.values():
        require(not any(t.is_meta for t in (*module.parameters(), *module.buffers())), "unmaterialized native component")
        module.to(device=device, dtype=torch.float16).eval().requires_grad_(False)
    return components, {**body, "runtime_derived_fourier_buffer": {**derived_fourier,
                        "buffer_identity_after_native_f16_move": tensor_identity(vae.fourier_embedder.frequencies)}}
