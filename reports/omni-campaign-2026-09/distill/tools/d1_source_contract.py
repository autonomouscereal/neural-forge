"""Fail-closed D1 successor admission, usable before importing Torch/model code.

This module validates genuine safetensors bytes, source spans and all source
pins. It does not build teacher caches, open checkpoints or perform updates.
"""
from __future__ import annotations
import argparse
import array
import hashlib
import json
from pathlib import Path
import struct
import sys

SCHEMA = "chimera_recovery.d1_source_successor.v1"
_SOURCE_MAP = None

def resolved_source(path):
    if _SOURCE_MAP is None:
        return str(path)
    from d1_allocated_contract import map_source
    return map_source(_SOURCE_MAP, path)

def configure_source_map(args):
    global _SOURCE_MAP
    path = getattr(args, "source_path_map", None)
    digest = getattr(args, "source_path_map_sha256", None)
    if bool(path) != bool(digest):
        raise ValueError("D1 source mapping path/SHA must be supplied together")
    if path:
        from d1_allocated_contract import source_map_record
        _SOURCE_MAP = source_map_record(path, digest)
    else:
        _SOURCE_MAP = None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_pin(pin):
    path = Path(resolved_source(pin["path"]))
    if not path.is_file() or sha(path) != pin["sha256"]:
        raise ValueError(f"D1 source pin missing/changed: {path}")
    return path


def same_path(left, right):
    return Path(resolved_source(left)).resolve() == Path(resolved_source(right)).resolve()


def manifest_record(path, expected_sha):
    path = resolved_source(path)
    if not expected_sha or sha(path) != expected_sha:
        raise ValueError("D1 source manifest SHA differs from explicit launcher/trainer pin")
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    unsealed = {k: v for k, v in manifest.items() if k != "seal_sha256"}
    if manifest.get("schema") != SCHEMA or hashlib.sha256(canonical(unsealed)).hexdigest() != manifest.get("seal_sha256"):
        raise ValueError("D1 source successor schema/seal differs")
    counts = manifest["actual_counts"]
    for key in ("fit_rows", "held_rows", "total_rows", "width", "fit_real_tokens", "diagnostic_real_tokens", "total_real_tokens"):
        if type(counts.get(key)) is not int or counts[key] < 1:
            raise ValueError("D1 source successor positive real counts required")
    if (counts["width"] != 4096 or counts["fit_rows"] + counts["held_rows"] != counts["total_rows"]
            or counts["fit_real_tokens"] + counts["diagnostic_real_tokens"] != counts["total_real_tokens"]
            or counts["total_rows"] * counts["width"] - counts["total_real_tokens"] != counts["padded_positions"]):
        raise ValueError("D1 source successor row/real token census differs")
    fit, held = manifest["fit_groups"], manifest["diagnostic_groups"]
    if not fit or not held or len(fit) != len(set(fit)) or len(held) != len(set(held)) or set(fit) & set(held):
        raise ValueError("D1 fit/diagnostic connected groups missing, repeated or overlapping")
    return manifest


def tensor_lengths(path, counts):
    """Read the fixed successor tensor contract without an optional framework.

    Inspect every real ID and every padding position; the row masks must agree
    with the explicit fit/held boundary. No implicit full-width fallback.
    """
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("D1 tensor header truncated")
        size = struct.unpack("<Q", prefix)[0]
        if size > 1048576:
            raise ValueError("D1 tensor header unreasonably large")
        header = json.loads(stream.read(size))
        offset = 8 + size
        actual = path.stat().st_size - offset
        rows, width, fit = counts["total_rows"], counts["width"], counts["fit_rows"]
        specs = {"input_ids": ("I64", [rows, width], 8 * rows * width), "lengths": ("I64", [rows], 8 * rows),
                 "fit_mask": ("BOOL", [rows], rows), "held_mask": ("BOOL", [rows], rows)}
        if set(header) - {"__metadata__"} != set(specs):
            raise ValueError("D1 explicit input_ids/lengths/fit_mask/held_mask required")
        spans = []
        for key, (dtype, shape, nbytes) in specs.items():
            record = header[key]
            lo, hi = record["data_offsets"]
            if (record["dtype"] != dtype or record["shape"] != shape or type(lo) is not int or type(hi) is not int
                    or not 0 <= lo < hi <= actual or hi - lo != nbytes):
                raise ValueError("D1 tensor shape/dtype/byte span differs")
            spans.append((lo, hi))
        spans.sort()
        if spans[0][0] != 0 or spans[-1][1] != actual or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError("D1 tensor byte spans overlap or leave unowned bytes")
        def read(key):
            lo, hi = header[key]["data_offsets"]
            stream.seek(offset + lo)
            value = stream.read(hi - lo)
            if len(value) != hi - lo:
                raise ValueError("D1 tensor payload truncated")
            return value
        lengths = [entry[0] for entry in struct.iter_unpack("<q", read("lengths"))]
        if (any(not 1 <= length <= width for length in lengths)
                or sum(lengths[:fit]) != counts["fit_real_tokens"] or sum(lengths[fit:]) != counts["diagnostic_real_tokens"]):
            raise ValueError("D1 real lengths/fit-held token counts differ")
        if read("fit_mask") != bytes([1] * fit + [0] * (rows - fit)) or read("held_mask") != bytes([0] * fit + [1] * (rows - fit)):
            raise ValueError("D1 masks disagree with explicit fit/held boundary")
        stream.seek(offset + header["input_ids"]["data_offsets"][0])
        for length in lengths:
            ids = array.array("q")
            ids.frombytes(stream.read(width * 8))
            if sys.byteorder != "little":
                ids.byteswap()
            if len(ids) != width or any(value < 0 or value >= 248320 for value in ids[:length]) or any(ids[length:]):
                raise ValueError("D1 real token vocabulary or zero padding differs")
    return lengths


def validate_manifest(path, expected_sha, *, ids, seq_len, lengths_key, held_rows, preregistration, max_rows=None):
    path = resolved_source(path)
    manifest = manifest_record(path, expected_sha)
    if (not same_path(ids, manifest["dataset"]["path"]) or seq_len != manifest["actual_counts"]["width"]
            or lengths_key != "lengths" or held_rows != manifest["actual_counts"]["held_rows"] or max_rows is not None
            or not same_path(preregistration, manifest["preregistration"]["path"])):
        raise ValueError("D1 trainer arguments differ from exact successor dataset/lengths/held/preregistration contract")
    pins = dict(manifest["sources"], dataset=manifest["dataset"], documents=manifest["documents"],
                preregistration=manifest["preregistration"], source_audit=manifest["source_audit"])
    paths = {key: checked_pin(value) for key, value in pins.items()}
    lengths = tensor_lengths(paths["dataset"], manifest["actual_counts"])
    counts = manifest["actual_counts"]
    totals, observed, docs = {"train": 0, "validation": 0}, {"train": set(), "validation": set()}, {"train": 0, "validation": 0}
    identifiers = set()
    with paths["documents"].open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            split = row["split"]
            if (split not in totals or row["id"] in identifiers or not row.get("group") or row["start"] != totals[split]
                    or type(row["length"]) is not int or row["length"] < 1 or row["end"] != row["start"] + row["length"]):
                raise ValueError("D1 source spans/order/split/conversation IDs differ")
            identifiers.add(row["id"])
            totals[split] = row["end"]
            observed[split].add(row["group"])
            docs[split] += 1
    if (totals != {"train": counts["fit_real_tokens"], "validation": counts["diagnostic_real_tokens"]}
            or observed["train"] != set(manifest["fit_groups"]) or observed["validation"] != set(manifest["diagnostic_groups"])
            or docs != {"train": counts["fit_documents"], "validation": counts["diagnostic_documents"]}):
        raise ValueError("D1 source document/group census differs")
    # Recheck byte pins after reading: admission is a concrete point-in-time proof.
    if sha(path) != expected_sha or any(sha(paths[key]) != pins[key]["sha256"] for key in paths):
        raise ValueError("D1 source changed during admission")
    return dict(schema="chimera_recovery.d1_source_contract.v1", manifest_path=str(Path(path).resolve()),
                manifest_sha256=expected_sha, source_seal_sha256=manifest["seal_sha256"], actual_counts=counts,
                template_sha256=manifest["template_sha256"], source_audit=manifest["source_audit"], dataset=manifest["dataset"],
                preregistration=manifest["preregistration"], fit_groups=manifest["fit_groups"], diagnostic_groups=manifest["diagnostic_groups"],
                all_source_pins_verified=True, exact_real_lengths_verified=True, parent_teacher="direct immutable parent hidden-state forward",
                qtext_teacher_cache_required=False)


def validate_args(args, trainer_source):
    configure_source_map(args)
    if not args.source_manifest:
        if args.source_manifest_sha256 or args.source_validator_sha256:
            raise ValueError("D1 source flags must be provided together")
        return None
    if not args.source_validator_sha256 or sha(__file__) != args.source_validator_sha256:
        raise ValueError("D1 source validator code pin differs")
    record = validate_manifest(Path(args.source_manifest), args.source_manifest_sha256, ids=args.ids, seq_len=args.seq_len,
                              lengths_key=args.lengths_key, held_rows=args.held_rows, preregistration=args.preregistration,
                              max_rows=args.max_rows)
    record["trainer_sha256"] = sha(trainer_source)
    record["validator_sha256"] = sha(__file__)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--source-path-map")
    parser.add_argument("--source-path-map-sha256")
    args = parser.parse_args()
    configure_source_map(args)
    manifest = manifest_record(args.manifest, args.sha256)
    result = validate_manifest(args.manifest, args.sha256, ids=manifest["dataset"]["path"], seq_len=4096, lengths_key="lengths",
                               held_rows=manifest["actual_counts"]["held_rows"], preregistration=manifest["preregistration"]["path"])
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
