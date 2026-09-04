"""Independent Sep 4 S0 serialization verifier; no model construction or forward.

Run on the AI server with the import tree on PYTHONPATH. Paths intentionally bind
this receipt to one probe. Refuses to overwrite an existing verification result.
"""
import collections, hashlib, json, math, pathlib, resource, struct, time
from chimera_omni.importing.safetensors_io import read_header, read_safetensors_file
root = pathlib.Path("/home/cereal/omni-campaign/composite/s0/q38-27bu-chimera-bf16-sep04")
job = pathlib.Path("/home/cereal/omni-campaign/composite/codex-sep04-safety")
started = time.time()
assert "Exit status: 0" in (job / "s0_import.time.log").read_text(), "import process did not exit successfully"
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()
receipt_path = root / "import_receipt.json"
receipt_sha = sha(receipt_path)
assert receipt_sha == (root / "import_receipt.json.sha256").read_text().split()[0], "receipt SHA mismatch"
r = json.loads(receipt_path.read_text())
assert sha(root / "config.json") == r["checkpoint"]["config_sha256"]
assert sha(root / "model.safetensors.index.json") == r["checkpoint"]["index_sha256"]
index = json.loads((root / "model.safetensors.index.json").read_text())
wm = index["weight_map"]
expected = index["metadata"]["tensor_sha256"]
assert set(wm) == set(expected), "missing tensor hashes"
assert not index["metadata"]["tied_weights"], "this verifier targets the untied 27B probe"
shards = sorted(set(wm.values()))
assert shards == sorted(p.name for p in root.glob("*.safetensors"))
state = hashlib.sha256()
names_seen = []
params = 0
dtypes = collections.Counter()
files = {}
zero_band_tables = []
for filename in shards:
    p = root / filename
    assert p.resolve().is_relative_to(root), "escaping shard"
    assert p.stat().st_size == r["checkpoint"]["shard_files"][filename]["bytes"]
    file_sha = sha(p)
    assert file_sha == r["checkpoint"]["shard_files"][filename]["sha256"], filename
    header, base = read_header(p)
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            assert key not in obj, f"duplicate header key: {key}"
            obj[key] = value
        return obj
    with p.open("rb") as raw:
        header_length = struct.unpack("<Q", raw.read(8))[0]
        strict_header = json.loads(raw.read(header_length), object_pairs_hook=unique_object)
    assert strict_header == header
    cursor = 0
    for name, meta in sorted(((n, m) for n, m in header.items() if n != "__metadata__"), key=lambda item: item[1]["data_offsets"][0]):
        begin, end = meta["data_offsets"]
        assert meta["dtype"] == "BF16", "unexpected serialized dtype"
        assert all(type(d) is int and d >= 0 for d in meta["shape"])
        assert type(begin) is int and type(end) is int
        assert begin == cursor and end - begin == 2 * math.prod(meta["shape"]), f"invalid payload offsets: {name}"
        cursor = end
    assert base + cursor == p.stat().st_size, "trailing or missing payload bytes"
    names = sorted(name for name, f in wm.items() if f == filename)
    assert set(header) - {"__metadata__"} == set(names), "index/header disagreement"
    views = read_safetensors_file(p, names)
    assert set(views) == set(names), "missing serialized tensor"
    for name in names:
        view = views[name]
        assert view.sha256() == expected[name], name
        state.update(name.encode("utf-8"))
        state.update(view.dtype.encode("ascii"))
        state.update(json.dumps(list(view.shape), separators=(",", ":")).encode("ascii"))
        state.update(view.buffer)
        if name in {"embed.weight", "lm_head.weight"}:
            base_vocab = r["ledger_bands"]["base_vocab"]
            hidden = r["ledger_bands"]["hidden_size"]
            assert list(view.shape) == [r["ledger_bands"]["vocab_size"], hidden]
            assert view.dtype == "BF16"
            tail = memoryview(view.buffer)[base_vocab * hidden * 2:]
            assert len(tail) == (r["ledger_bands"]["vocab_size"] - base_vocab) * hidden * 2
            for offset in range(0, len(tail), 8 * 1024 * 1024):
                assert not bytes(tail[offset:offset + 8 * 1024 * 1024]).strip(b"\x00"), f"{name} nonzero ledger band"
            del tail
            zero_band_tables.append(name)
        params += math.prod(view.shape)
        dtypes[view.dtype] += 1
        names_seen.append(name)
    del view, views
    files[filename] = file_sha
    print(f"verified {filename}: {len(names)} tensors", flush=True)
assert names_seen == sorted(wm), "non-monotone shard/name layout"
assert state.hexdigest() == r["checkpoint"]["state_sha256"], "full serialized state SHA mismatch"
assert params == r["checkpoint"]["header_element_count"] == r["checkpoint"]["model_parameter_count"] == 27699193344
assert set(dtypes) == {"BF16"}, dtypes
assert r["ledger_bands"]["declared_extra"] == [["music_codes", 65536]]
assert r["ledger_bands"]["band_rows_zero"] and r["ledger_bands"]["extra_band_rows_zero"]
assert sorted(zero_band_tables) == ["embed.weight", "lm_head.weight"]
out = {
    "schema": "chimera.s0.serialization-verification.v1",
    "mode": "serialization-only-no-model-forward",
    "directory": str(root), "receipt_sha256": receipt_sha,
    "state_sha256": state.hexdigest(), "header_parameters": params,
    "tensor_count": len(names_seen), "dtypes": dict(dtypes),
    "shard_sha256": files, "music_codes_rows": 65536,
    "all_serialized_tensor_hashes_verified": True,
    "all_serialized_ledger_band_rows_verified_zero": True,
    "full_state_hash_reproduced": True,
    "contiguous_offsets_exact_lengths_unique_header_keys": True,
    "wall_seconds": round(time.time() - started, 2),
    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
}
encoded = (json.dumps(out, indent=2, sort_keys=True) + "\n").encode()
target = job / "s0_serialization_verification.json"
assert not target.exists(), "verification result already exists; inspect it instead of overwriting"
target.write_bytes(encoded)
print(json.dumps(out, sort_keys=True))
