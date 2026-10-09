"""C18 development tables from a verified ACE component and two directed charts.

This module never constructs a chart, fits a model, installs a child, or accepts
capability/training claims. The CLI uses the existing ACE proof-closure reader.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import struct
import sys

SCHEMA = "chimera.ace_host_tables.development.v1"
CHART_SCHEMA = "chimera.ace_host_directed_charts.development.v1"
SOURCE_START, ELIGIBLE, RESERVED = 151669, 64000, 65536
DIRECTIONS = {
    "host_to_source.weight": "source_final_hidden = host_final_hidden @ weight.T",
    "source_to_host.weight": "host_embedding = source_embedding @ weight.T",
}
HOST_BINDING_HELPER_SHA = "b86d86a24e2bfedb229c071fff3ed89cc75d27aeaca6ed61a87db8af46a571d5"


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    def real(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON number")
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject, parse_float=real)


def file_pin(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return {"bytes": Path(path).stat().st_size, "sha256": h.hexdigest()}


def checked_file(binding):
    if not isinstance(binding, dict) or set(binding) != {"path", "bytes", "sha256"}:
        raise ValueError("binding requires exact path/bytes/sha256")
    path = Path(binding["path"]).resolve(strict=True)
    if type(binding["bytes"]) is not int or binding["bytes"] < 0:
        raise ValueError("invalid bound byte count")
    if file_pin(path) != {k: binding[k] for k in ("bytes", "sha256")}:
        raise ValueError(f"bound file changed: {path.name}")
    return path


def guard_output(path, paths, reader):
    target = Path(path).resolve()
    sidecar = Path(str(target) + ".json")
    donor_run = paths["ace_component_manifest"].parent.parent.resolve()
    protected = {donor_run, Path(reader).resolve(), *(p.parent.resolve() for p in paths.values())}
    for destination in (target, sidecar):
        for root in protected:
            if destination == root or root in destination.parents:
                raise ValueError("output overlaps donor/reader/bound source directory")
        if destination.exists():
            raise FileExistsError("output or sidecar already exists")
    if not target.parent.is_dir():
        raise ValueError("fresh output parent must already exist")
    return target


def metadata_geometry(ace, host):
    """Validate current actual source/host descriptors without opening weights."""
    if ace.get("schema") != "chimera.ace_native_component.development.v1":
        raise ValueError("unsupported ACE component descriptor")
    cfg = ace.get("source_config", {})
    if (cfg.get("hidden_size"), cfg.get("vocab_size"), cfg.get("num_hidden_layers"),
        cfg.get("tie_word_embeddings")) != (2048, 217204, 28, True):
        raise ValueError("ACE source constructor differs")
    if ace.get("intended_layers") != list(range(28)) or ace.get("widths") != [1536] * 28:
        raise ValueError("complete folded ACE28 coverage required")
    expected_artifacts = {f"layer-{i:02}/{name}" for i in range(28) for name in
                          ("capture.safetensors", "capture_receipt.json", "folded.safetensors", "fold_receipt.json")}
    if set(ace.get("verified_layer_artifacts", {})) != expected_artifacts:
        raise ValueError("ACE complete portable proof inventory required")
    if ace.get("source_named_tensor_count") != 310 or len(ace.get("tensors", {})) != 310:
        raise ValueError("ACE complete tensor inventory required")
    protected = ace.get("preserved_attention_and_global", {})
    if len(protected) != 226 or "embed_tokens.weight" not in protected:
        raise ValueError("ACE protected table inventory missing")
    if protected["embed_tokens.weight"] != ace["tensors"]["embed_tokens.weight"]:
        raise ValueError("ACE protected embedding identity differs")
    if set(ace.get("tokenizer_assets", {})) != {"tokenizer/" + n for n in
        ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
         "vocab.json", "merges.txt", "added_tokens.json")}:
        raise ValueError("ACE tokenizer inventory incomplete")
    bridge = ace.get("codec_bridge", {})
    if bridge.get("source_generator_interval") != [SOURCE_START, SOURCE_START + ELIGIBLE]:
        raise ValueError("ACE generator ID interval differs")
    if host.get("schema") != "chimera_omni.checkpoint.v1":
        raise ValueError("actual native host checkpoint descriptor required")
    h = host.get("config", {})
    if h.get("hidden_size") != 5120 or h.get("num_layers") != 64 or h.get("intermediate_size") != 4352:
        raise ValueError("actual S3 native host geometry differs")
    ledger = host.get("ledger", {})
    ranges = ledger.get("ranges", [])
    if ledger.get("schema") != "chimera_omni.ledger.v1" or ledger.get("vocab_size") != h.get("vocab_size"):
        raise ValueError("host ledger identity differs")
    cursor = 0
    names = set()
    for band in ranges:
        if band.get("name") in names or type(band.get("begin")) is not int or type(band.get("end")) is not int:
            raise ValueError("invalid or duplicate ledger band")
        if band["begin"] != cursor or band["end"] <= cursor or band.get("width") != band["end"] - cursor:
            raise ValueError("ledger gaps/overlap/width mismatch")
        names.add(band["name"]); cursor = band["end"]
    if cursor != h["vocab_size"]:
        raise ValueError("ledger coverage differs")
    music = [b for b in ranges if b["name"] == "music_codes"]
    if len(music) != 1 or [music[0]["begin"], music[0]["end"]] != [252416, 317952]:
        raise ValueError("actual host music reservation differs")
    if bridge.get("ledger") != ledger or bridge.get("native_unused_interval") != [316416, 317952]:
        raise ValueError("source/host codec ledger identity differs")
    if type(h.get("tie_word_embeddings")) is not bool:
        raise ValueError("explicit host table tying required")
    return {"source_hidden": 2048, "host_hidden": 5120, "source_start": SOURCE_START,
            "eligible_rows": ELIGIBLE, "reserved_rows": RESERVED,
            "native_begin": music[0]["begin"], "host_tied": h["tie_word_embeddings"]}


def validate_chart_receipt(receipt, bindings):
    if receipt.get("schema") != CHART_SCHEMA or receipt.get("directions") != DIRECTIONS:
        raise ValueError("explicit independent directed chart contract required")
    if receipt.get("bias_policy") != "zero_no_bias_tensors":
        raise ValueError("only explicit zero-bias charts supported")
    if receipt.get("numerical_policy") != "diagnostic_only":
        raise ValueError("finite chart diagnostics must not gate")
    if receipt.get("bindings") != bindings:
        raise ValueError("chart source/host/capture binding differs")
    if receipt.get("pairing") != "exact_record_and_position" or receipt.get("fit_split") != "FIT":
        raise ValueError("exact paired FIT capture identity required")
    if receipt.get("synthetic_fixture") is not False:
        raise ValueError("CLI export requires actual source charts, not fixtures")


def host_bundle_join(bindings, paths):
    """Reuse the adopted metadata guard; do not trust unrelated pinned files."""
    import importlib.util
    helper = paths["host_binding_reader"]
    if bindings["host_binding_reader"]["sha256"] != HOST_BINDING_HELPER_SHA:
        raise ValueError("adopted host binding helper identity differs")
    spec = importlib.util.spec_from_file_location("c18_bound_host_guard", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    references = {short: {"path": str(paths[long]), "sha256": bindings[long]["sha256"]}
                  for short, long in (("config", "host_config"), ("index", "host_index"),
                                      ("runtime_contract", "host_runtime_contract"))}
    return module.host_checkpoint_binding(strict_json(paths["host_bundle_receipt"]),
                    source_binding_policy=module.OWNER_DIAGNOSTIC_POLICY, metadata_refs=references)


def paired_capture_join(receipt, bindings, geometry):
    expected = {k: bindings[k] for k in ("ace_component_manifest", "host_bundle_receipt", "host_config",
                                        "host_index", "host_runtime_contract", "paired_capture_rows")}
    if receipt.get("schema") != "chimera.ace_host_paired_capture.development.v1":
        raise ValueError("typed ACE/host paired capture receipt required")
    if receipt.get("bindings") != expected or receipt.get("synthetic_fixture") is not False:
        raise ValueError("paired capture source/host/rows identities differ")
    if receipt.get("row_semantics") != {"source_embedding_rows": "source_codec_embedding",
                "host_embedding_rows": "declared_host_codec_embedding_target",
                "source_final_hidden_rows": "ACE_final_normalized_hidden",
                "host_final_hidden_rows": "S3_final_normalized_hidden"}:
        raise ValueError("paired input/output boundary semantics differ")
    pairs = receipt.get("pairs")
    if not isinstance(pairs, list) or not pairs or type(receipt.get("row_count")) is not int or receipt["row_count"] != len(pairs):
        raise ValueError("paired record/position inventory missing")
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {"record_id", "position", "source_id", "native_id", "split"}:
            raise ValueError("paired row identity fields differ")
        if not isinstance(pair["record_id"], str) or not pair["record_id"] or type(pair["position"]) is not int or pair["position"] < 0:
            raise ValueError("invalid paired record/position")
        key = (pair["record_id"], pair["position"])
        if key in seen or pair["split"] != "FIT":
            raise ValueError("duplicate or non-FIT paired row")
        seen.add(key)
        source_id, native_id = pair["source_id"], pair["native_id"]
        if type(source_id) is not int or type(native_id) is not int or not SOURCE_START <= source_id < SOURCE_START + ELIGIBLE:
            raise ValueError("paired codec ID is outside generator interval")
        if native_id != geometry["native_begin"] + source_id - SOURCE_START:
            raise ValueError("paired source/native codec IDs differ")
    return pairs


def tensor_digest(value):
    import torch
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b"F32" + json.dumps(list(value.shape), separators=(",", ":")).encode() + raw).hexdigest()


def validate_capture_and_fit(rows, pairs, embedding, charts, geometry, chart_receipt, capture_receipt, bindings):
    import torch
    p, h, n = geometry["source_hidden"], geometry["host_hidden"], len(pairs)
    shapes = {"source_embedding_rows": (n, p), "host_embedding_rows": (n, h),
              "source_final_hidden_rows": (n, p), "host_final_hidden_rows": (n, h)}
    if set(rows) != set(shapes):
        raise ValueError("exact paired embedding/final-hidden row tensors required")
    for name, shape in shapes.items():
        value = rows[name]
        if tuple(value.shape) != shape or value.dtype != torch.float32 or value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise ValueError("paired capture geometry/dtype/nonfinite fault")
    ids = torch.tensor([pair["source_id"] for pair in pairs], dtype=torch.long)
    if not torch.equal(rows["source_embedding_rows"], embedding[ids].float()):
        raise ValueError("paired source embedding is not the actual protected table row")
    if capture_receipt.get("tensor_sha256") != {k: tensor_digest(v) for k, v in rows.items()}:
        raise ValueError("paired capture tensor identity differs")
    if chart_receipt.get("tensor_sha256") != {k: tensor_digest(v) for k, v in charts.items()}:
        raise ValueError("directed chart tensor identity differs")
    fit = chart_receipt.get("fit")
    if not isinstance(fit, dict) or fit.get("method") != "ridge_through_origin" or fit.get("maps_independently_fitted") is not True:
        raise ValueError("independent zero-bias fit provenance missing")
    if fit.get("row_count") != n or fit.get("rows_file_sha256") != bindings["paired_capture_rows"]["sha256"]:
        raise ValueError("fit/capture row identity differs")
    if fit.get("input_direction") != "source_embedding_rows_to_host_embedding_rows" or fit.get("output_direction") != "host_final_hidden_rows_to_source_final_hidden_rows":
        raise ValueError("fit direction/boundary differs")
    # Error/cosine/compression are diagnostics only. Finite fitted maps can be
    # imperfect. These joins establish provenance/geometry, not observed behavior.


def validate_tensors(embedding, charts, geometry):
    import torch
    p, h = geometry["source_hidden"], geometry["host_hidden"]
    if set(charts) != set(DIRECTIONS):
        raise ValueError("exact two chart tensors required; biases are unsupported")
    if embedding.ndim != 2 or embedding.shape[1] != p or embedding.shape[0] < geometry["source_start"] + geometry["eligible_rows"]:
        raise ValueError("source embedding geometry differs")
    if embedding.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("source embedding must be floating-point")
    for name, shape in (("host_to_source.weight", (p, h)), ("source_to_host.weight", (h, p))):
        value = charts[name]
        if tuple(value.shape) != shape or value.dtype != torch.float32 or value.device.type != "cpu":
            raise ValueError(f"directed F32 CPU chart geometry differs: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite chart: {name}")
    if embedding.device.type != "cpu":
        raise ValueError("CPU-only export requires CPU source embedding")


def projected_batches(embedding, charts, geometry, *, name, chunk_rows=256):
    import torch
    if type(chunk_rows) is not int or chunk_rows <= 0 or chunk_rows > 1024:
        raise ValueError("chunk_rows must be 1..1024")
    if name not in ("embed.weight", "lm_head.weight"):
        raise ValueError("unknown table")
    validate_tensors(embedding, charts, geometry)
    # Separate maps are essential: the source's tied output does not tie host tables.
    matrix = (charts["source_to_host.weight"].T if name == "embed.weight"
              else charts["host_to_source.weight"])
    start, count, reserved = geometry["source_start"], geometry["eligible_rows"], geometry["reserved_rows"]
    if not 0 < count <= reserved:
        raise ValueError("eligible rows must fit reservation")
    with torch.inference_mode():
        for begin in range(0, reserved, chunk_rows):
            length = min(chunk_rows, reserved - begin)
            active = max(0, min(length, count - begin))
            value = torch.zeros((length, geometry["host_hidden"]), dtype=torch.float32)
            if active:
                source = embedding[start + begin:start + begin + active].float()
                if not bool(torch.isfinite(source).all()):
                    raise ValueError("nonfinite eligible source embedding")
                value[:active] = source @ matrix
            if not bool(torch.isfinite(value).all()):
                raise ValueError("nonfinite projected table")
            yield value


def publish_tables(path, embedding, charts, geometry, provenance, *, chunk_rows=256):
    """Stream an exclusive plain F32 safetensors file; sidecar commits last.

    A failed publication retains the partial file and has no committed sidecar.
    No retry overwrites an existing path. Caller chooses a new output path.
    """
    import torch
    validate_tensors(embedding, charts, geometry)
    path = Path(path)
    if path.exists() or Path(str(path) + ".json").exists():
        raise FileExistsError("publication path already exists")
    shape = [geometry["reserved_rows"], geometry["host_hidden"]]
    nbytes = math.prod(shape) * 4
    header = {name: {"dtype": "F32", "shape": shape, "data_offsets": [i*nbytes, (i+1)*nbytes]}
              for i, name in enumerate(("embed.weight", "lm_head.weight"))}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    digests, table_blocks = {}, {}
    with path.open("xb") as f:
        f.write(struct.pack("<Q", len(encoded))); f.write(encoded)
        for name in header:
            digest = hashlib.sha256(b"F32" + json.dumps(shape, separators=(",", ":")).encode())
            written = 0
            for batch in projected_batches(embedding, charts, geometry, name=name, chunk_rows=chunk_rows):
                raw = batch.contiguous().view(torch.uint8).numpy().tobytes()
                f.write(raw); digest.update(raw); written += len(raw)
            if written != nbytes:
                raise ValueError("incomplete streamed table")
            digests[name] = digest.hexdigest()
            table_blocks[name] = nbytes
        f.flush(); os.fsync(f.fileno())
    # Read every output byte, independently of the streaming writer digest.
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        read_header = json.loads(f.read(size))
        if read_header != header:
            raise ValueError("serialized table header readback differs")
        for name in header:
            digest = hashlib.sha256(b"F32" + json.dumps(shape, separators=(",", ":")).encode())
            remaining = nbytes
            unused_start = geometry["eligible_rows"] * geometry["host_hidden"] * 4
            position = 0
            while remaining:
                raw = f.read(min(1 << 20, remaining))
                if not raw:
                    raise ValueError("truncated serialized table")
                if position + len(raw) > unused_start and any(raw[max(0, unused_start - position):]):
                    raise ValueError("unused music positions are not exactly zero")
                digest.update(raw); position += len(raw); remaining -= len(raw)
            if digest.hexdigest() != digests[name]:
                raise ValueError("serialized table tensor readback differs")
        if f.read(1):
            raise ValueError("unexpected table trailing bytes")
    if geometry["host_tied"] and digests["embed.weight"] != digests["lm_head.weight"]:
        raise ValueError("tied host requires exactly equal projected tables")
    receipt = {"schema": SCHEMA, "geometry": geometry, "tensor_sha256": digests,
               "table_file": file_pin(path), "table_payload_bytes": sum(table_blocks.values()),
               "provenance": provenance, "numerical_policy": "diagnostic_only",
               "unused_rows_exact_zero": geometry["reserved_rows"] - geometry["eligible_rows"],
               "composite_child_installed": False, "capability_evaluated": False, "training_eligibility": 0}
    encoded_receipt = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    sidecar = Path(str(path) + ".json")
    with sidecar.open("xb") as f:
        f.write(encoded_receipt); f.flush(); os.fsync(f.fileno())
    if sidecar.read_bytes() != encoded_receipt:
        raise ValueError("sidecar readback differs")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--ace-reader-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.threads <= 2:
        raise ValueError("CPU thread cap is two")
    bindings = strict_json(args.bindings)
    required = {"ace_component_manifest", "host_bundle_receipt", "host_binding_reader", "host_config",
                "host_index", "host_runtime_contract", "paired_capture_receipt", "paired_capture_rows",
                "charts", "chart_receipt"}
    if set(bindings) != required:
        raise ValueError("exact source/host/chart/capture binding inventory required")
    paths = {name: checked_file(binding) for name, binding in bindings.items()}
    reader = Path(args.ace_reader_dir).resolve(strict=True)
    output = guard_output(args.out, paths, reader)
    ace, host = strict_json(paths["ace_component_manifest"]), strict_json(paths["host_config"])
    geometry = metadata_geometry(ace, host)
    host_join = host_bundle_join(bindings, paths)
    capture_receipt = strict_json(paths["paired_capture_receipt"])
    pairs = paired_capture_join(capture_receipt, bindings, geometry)
    chart_receipt = strict_json(paths["chart_receipt"])
    chart_bindings = {k: v for k, v in bindings.items() if k != "chart_receipt"}
    validate_chart_receipt(chart_receipt, chart_bindings)
    import torch
    torch.set_num_threads(args.threads)
    from safetensors.torch import load_file
    charts = load_file(str(paths["charts"]), device="cpu")
    for name in ("ace_fold.py", "ace_native.py", "qf_solve.py"):
        if file_pin(reader / name) != ace["implementation"][name]:
            raise ValueError("ACE reader implementation differs from completed artifact")
    # The existing reader checks original runtime, full portable proof closure,
    # source tensor identities, unchanged protected tensors and all folded FFNs.
    sys.path.insert(0, str(reader))
    sys.path.insert(0, str(reader / "native_src"))
    component, loaded = importlib.import_module("ace_fold").load_component(paths["ace_component_manifest"].parent, device="cpu")
    if loaded != ace:
        raise ValueError("ACE reader descriptor differs")
    validate_tensors(component.embed_tokens.weight, charts, geometry)
    rows = load_file(str(paths["paired_capture_rows"]), device="cpu")
    validate_capture_and_fit(rows, pairs, component.embed_tokens.weight, charts, geometry,
                             chart_receipt, capture_receipt, bindings)
    result = publish_tables(output, component.embed_tokens.weight, charts, geometry,
                            {"bindings": bindings, "reader_directory": str(reader),
                             "host_binding": host_join, "paired_row_count": len(pairs),
                             "source_reader": "ace_fold.load_component full proof closure"})
    print(json.dumps({"schema": result["schema"], "table_file": result["table_file"],
                      "table_payload_bytes": result["table_payload_bytes"], "training_eligibility": 0}))


if __name__ == "__main__":
    main()
