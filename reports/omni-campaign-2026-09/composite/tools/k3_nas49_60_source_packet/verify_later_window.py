"""Verify and retain actual K3 window ancestry; never launch or alter native files.

Run with the pinned native image's Python, while its fitter is terminal. FFNs
remain subject to the separately pinned full finite/load scanner. This isolated
49-60 proposal requires the actual sealed37-48 read-only predecessor and does
not publish a later-window plan or launch native work.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
from k3_recovery_ancestry import LEGACY_TOOL_SHA
from ro_resume_runtime import nas_resume_from_arguments

TOOL = "reports/omni-campaign-2026-09/composite/tools/k3_dense_pass.py"
LANE = "reports/omni-campaign-2026-09/k3/"
TOOL_SHA = "44ccfba059f0da1a5b2c63dfcbbb54e2f4e2b641fed5d0273cc00437cb6d8dde"
BASELINE_SHA = "c54e2fd0ade6fb8bbc9a111cb76fc5469abc1c197001b137e70f1c1156e976ff"
DIGEST_SHA = "58bc54f0c635ec78f53a4575155d249ebd10bde4ea302f65872bebe253d7a91f"
CONFIG_SHA = "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213"
INDEX_SHA = "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd"
CORPUS_SHA = "7578ff9e7b55586c382150782ca1210ba5e671f6a3a6fe94b6b58c9135c50998"
GPU_UUID = "GPU-f5a09535-fc85-a027-50f7-734b344747dd"
LINEAGE = "k3_dense_pass.prefix_lineage.v1"
WINDOWS = [(5, 12)] + [(s, min(s + 11, 92)) for s in range(13, 93, 12)]
_RESUME_PATHS = None
FORWARD_CHECKS = {"folded latent_in (fit input)", "teacher y_lat (fit target)",
                  "teacher prefix after the layer", "folded prefix after the layer"}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def canonical(value):
    return json.dumps(value, indent=1, sort_keys=True, default=str)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    require(isinstance(value, dict), "JSON object required: " + str(path))
    return value


class Pins:
    def __init__(self):
        self.stamps = {}

    def file(self, path, expected):
        path = Path(path)
        require(not path.is_symlink() and path.is_file(), "regular nonsymlink file required: " + str(path))
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "explicit SHA256 required")
        before = path.stat()
        require(sha(path) == expected, "full file hash mismatch: " + str(path))
        stamp = (before.st_size, before.st_mtime_ns)
        require(stamp == (path.stat().st_size, path.stat().st_mtime_ns), "input changed during hash")
        require(path not in self.stamps or self.stamps[path] == stamp, "input changed during verification")
        self.stamps[path] = stamp
        return path

    def observed(self, path):
        path = Path(path)
        value = sha(path)
        self.file(path, value)
        return value

    def json(self, path, expected):
        return read(self.file(path, expected))

    def stable(self):
        for path, stamp in self.stamps.items():
            require(stamp == (path.stat().st_size, path.stat().st_mtime_ns), "input changed: " + str(path))


def mapped(root, logical, prefix="/run-output"):
    if prefix == "/run-output":
        require(_RESUME_PATHS is not None and root == _RESUME_PATHS.output,
                "exact RO-prefix/new-output contract required")
        return _RESUME_PATHS.read_path(logical)
    logical_path, prefix_path = PurePosixPath(logical), PurePosixPath(prefix)
    require(logical_path.is_absolute() and str(logical_path) == logical and ".." not in logical_path.parts,
            "canonical absolute recorded path required")
    require(logical_path.is_relative_to(prefix_path) and logical_path != prefix_path, "recorded path aliases its mount")
    path = root.joinpath(*logical_path.relative_to(prefix_path).parts)
    require(not path.is_symlink() and path.resolve(strict=True).is_relative_to(root), "recorded file escapes mount")
    return path


def declared_write(output, path):
    """Every new ancestry leaf must be in the same native finite write manifest."""
    path = Path(path)
    require(_RESUME_PATHS is not None and output == _RESUME_PATHS.output,
            "exact owned NAS output root required")
    require(path.is_relative_to(output), "new verifier leaf escapes output")
    logical = "/run-output/" + path.relative_to(output).as_posix()
    destination = _RESUME_PATHS.write_path(logical)
    require(destination == path and not _RESUME_PATHS.fs.exists(destination) and not destination.is_symlink(),
            "exclusive declared verifier output required")
    return destination


def note_output(output, path):
    logical = "/run-output/" + Path(path).relative_to(output).as_posix()
    _RESUME_PATHS.note_created_output(logical)


def header(path):
    with Path(path).open("rb") as handle:
        lead = handle.read(8)
        require(len(lead) == 8, "truncated safetensors header")
        length = struct.unpack("<Q", lead)[0]
        require(2 <= length <= 16 << 20 and 8 + length <= Path(path).stat().st_size, "invalid safetensors header length")
        result = json.loads(handle.read(length), object_pairs_hook=unique)
    require(isinstance(result, dict), "safetensors header object required")
    return result


def native_ffn_shapes(width, cfg):
    latent, hidden = cfg["routed_expert_hidden_size"], cfg["hidden_size"]
    shared = cfg["num_shared_experts"] * cfg["moe_intermediate_size"]
    prefix = "block_sparse_moe."
    return {"w1_s": [width, latent], "w3_s": [width, latent], "w2p": [latent, width], "channels": [width, 2],
            prefix + "routed_expert_down_proj.weight": [latent, hidden],
            prefix + "routed_expert_up_proj.weight": [hidden, latent],
            prefix + "routed_expert_norm.weight": [latent],
            prefix + "shared_experts.gate_proj.weight": [shared, hidden],
            prefix + "shared_experts.up_proj.weight": [shared, hidden],
            prefix + "shared_experts.down_proj.weight": [hidden, shared]}


def finite_library_readback(path, expected_shapes=None, expected_meta=None, expected_dtypes=None):
    """Read every stored coordinate in bounded actual safe_open CPU slices."""
    import torch
    from safetensors import safe_open

    path = Path(path)
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    hdr = header(path)
    metadata = hdr.pop("__metadata__", {})
    require(isinstance(metadata, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()),
            "invalid safetensors metadata")
    if expected_shapes is not None:
        require(set(hdr) == set(expected_shapes), "exact endpoint tensor keys required")
        if expected_dtypes is None:
            require(len({record.get("dtype") for record in hdr.values()}) == 1, "endpoint tensor dtypes disagree")
    if expected_meta is not None:
        for key, value in expected_meta.items():
            require(metadata.get(key) == value, "endpoint metadata mismatch: " + key)
    records = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        require(set(handle.keys()) == set(hdr), "actual library/header tensor set mismatch")
        for name, description in hdr.items():
            dimensions = description["shape"]
            require(isinstance(dimensions, list) and all(type(x) is int and x >= 0 for x in dimensions), "invalid tensor shape")
            if expected_shapes is not None:
                require(dimensions == expected_shapes[name], "endpoint shape mismatch: " + name)
                if expected_dtypes is None:
                    require(description["dtype"] in ("F16", "BF16", "F32"), "native endpoint float dtype required")
                else:
                    require(description["dtype"] == expected_dtypes[name], "native saved-output dtype mismatch: " + name)
            view = handle.get_slice(name)
            require(view.get_shape() == dimensions, "actual library tensor shape mismatch")
            # One first-axis slice stays near 4 MiB; no full residual stack is loaded.
            trailing = math.prod(dimensions[1:]) if dimensions else 1
            rows = max(1, (1 << 20) // max(1, trailing))
            count, nonfinite = 0, 0
            for start in range(0, dimensions[0] if dimensions else 1, rows):
                tensor = view[start:min(start + rows, dimensions[0])] if dimensions else handle.get_tensor(name)
                count += tensor.numel()
                if tensor.is_floating_point():
                    nonfinite += int((~torch.isfinite(tensor)).sum().item())
                del tensor
            require(count == math.prod(dimensions) and nonfinite == 0, "actual nonfinite or incomplete tensor: " + name)
            records[name] = {"shape": dimensions, "dtype": description["dtype"], "elements": count, "nonfinite": nonfinite}
    require(before == (path.stat().st_size, path.stat().st_mtime_ns), "file changed during actual library read")
    return {"path": str(path), "bytes": before[0], "tensors": records,
            "actual_library_load_verified": True, "all_coordinates_finite": True,
            "reader": "safetensors.safe_open framework=pt CPU slices; no full model load"}


def fresh_observation(observed, image, context, check_age=True):
    require(observed.get("schema") == "neural-forge.k3-runtime-observation.v1" and observed.get("image_id") == image
            and observed.get("gpu_uuid") == GPU_UUID and observed.get("runtime") == context["runtime"]
            and observed.get("software") == context["software"], "actual native runtime/source context changed")
    at = datetime.fromisoformat(observed["observed_utc"].replace("Z", "+00:00"))
    require(at.tzinfo is not None, "native observation needs timezone")
    if check_age:
        require(0 <= (datetime.now(timezone.utc) - at).total_seconds() <= 300,
                "native observation stale/future; collect a fresh actual observation")


def verify(args):
    global _RESUME_PATHS
    pins = Pins()
    paths, guard = nas_resume_from_arguments(args, args.nf / TOOL)
    require(paths.first == 49 and paths.last == 60, "only authorized NAS window49-60 required")
    _RESUME_PATHS = paths
    output = args.output.resolve(strict=True)
    require(output == paths.output, "canonical separate new NAS output required")
    require(output.is_dir(), "owned continuation directory required")
    first = pins.json(args.first_plan, args.first_plan_sha256)
    require(first.get("schema") == "neural_forge.k3_owner_diagnostic_continuation_plan.v1"
            and first.get("required_gpu_uuid") == GPU_UUID and first.get("next_layers") == "5-12", "first actual record plan required")
    baseline = pins.json(args.baseline_receipt, BASELINE_SHA)
    legacy_context = baseline["resume_context"]
    require(guard.legacy_context == legacy_context, "manifest legacy context differs from genuine original accepted context")
    context = guard.current_context
    require(context.get("version") == "k3_dense_pass.resume_context.v1" and context["model"]["source_directory"] == "/parent"
            and context["model"]["config_sha256"] == CONFIG_SHA and context["model"]["index_sha256"] == INDEX_SHA
            and context["corpus"] == CORPUS_SHA and legacy_context["software"]["tool_sha256"] == LEGACY_TOOL_SHA,
            "original accepted context required")
    require(first["installed_ffn_sha256"] == {str(i): baseline["layers"][str(i)]["ffn"]["sha256"] for i in range(1, 5)},
            "initial prefix differs from original accepted receipt")
    require(context["runtime"]["num_threads"] == 22 and context["runtime"]["device"] == context["runtime"]["experts_device"] == "cuda:0"
            and context["runtime"]["resid_device"] == "cpu" and context["runtime"]["kda_backend"] == "torch"
            and context["runtime"]["devices"] == [{"capability": "sm75", "index": 0, "name": "Quadro RTX 8000", "total_GB": 47.6}],
            "original GPU6 sm75 native runtime required")
    runtime = pins.json(args.runtime, args.runtime_sha256)
    fresh_observation(runtime, first["image_id"], context)
    nf, parent, corpus = (getattr(args, key).resolve(strict=True) for key in ("nf", "parent", "corpus"))
    for source in (nf, parent, corpus):
        require(source.is_dir() and not (output == source or output.is_relative_to(source) or source.is_relative_to(output)), "source/output overlap")
    pins.file(nf / TOOL, guard.current_sha)
    require(set(context["software"]["lane_sha256"]) == {"k3_common.py", "k3_layer.py", "k3_stream.py",
            "c14_union_coreset.py", "k3_receipt.py", "k3_review_stream.py"}, "all six native lane pins required")
    for name, digest in context["software"]["lane_sha256"].items():
        require(re.fullmatch(r"[a-z0-9_]+\.py", name), "native lane basename required")
        pins.file(nf / LANE / name, digest)
    config = pins.json(parent / "config.json", CONFIG_SHA)
    index = pins.json(parent / "model.safetensors.index.json", INDEX_SHA)
    cfg = config.get("text_config", config)
    require(cfg["num_hidden_layers"] == 93 and cfg["first_k_dense_replace"] == 1 and cfg.get("moe_layer_freq", 1) == 1, "native 92-MoE layout required")
    metadata = context["model"]["source_file_metadata"]
    require(set(metadata) == set(index["weight_map"].values()) and len(metadata) == 96, "all 96 original indexed shards required")
    shard_stamps = {}
    for name, record in metadata.items():
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", name) and name not in (".", ".."), "source shard basename required")
        path = parent / name
        require(not path.is_symlink() and path.is_file(), "regular source shard required")
        stamp = (path.stat().st_size, path.stat().st_mtime_ns)
        require(stamp == (record["bytes"], record["mtime_ns"]), "source shard metadata changed: " + name)
        shard_stamps[path] = stamp
    digest = pins.json(args.digest, DIGEST_SHA)
    require(digest.get("complete") is True and digest.get("all_verified") is True and len(digest["files"]) == 98, "dated 98-file digest required")
    names = [r["file"] for r in digest["files"]]
    require(len(set(names)) == 98 and set(metadata).issubset(names), "digest does not cover exact source shards")
    for record in digest["files"]:
        name = record["file"]
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", name) and name not in (".", ".."), "digest basename required")
        path = parent / name
        require(not path.is_symlink() and path.is_file() and path.stat().st_size == record["local_size"], "dated digest current-size mismatch")
    require(baseline["corpus"]["corpus_sha256"] == CORPUS_SHA, "accepted corpus digest mismatch")
    for segment in baseline["corpus"]["segments"]:
        pins.file(mapped(corpus, segment["path"], "/inputs/corpus"), segment["file_sha256"])
    completed = tuple(int(x) for x in args.completed_window.split("-"))
    require(completed in WINDOWS and completed == (49, 60), "only authorized ordinary NAS window49-60 required")
    lo, hi = completed
    receipt = pins.json(args.receipt, args.receipt_sha256)
    require(args.receipt.resolve() == (output / "receipts/k3_dense_pass_receipt.json").resolve(), "canonical terminal full receipt required")
    require(receipt.get("resume_context") == context and receipt.get("corpus", {}).get("corpus_sha256") == CORPUS_SHA
            and receipt.get("arguments", {}).get("rung") == 17408 and receipt["arguments"].get("acceptance") == "record",
            "completed receipt source/runtime/corpus/rung/record context mismatch")
    processed = receipt.get("layers_processed_this_run")
    require(isinstance(processed, list) and processed and all(type(x) is int for x in processed)
            and lo <= processed[0] <= hi and processed == list(range(processed[0], hi + 1)), "completed actual native suffix coverage mismatch")
    start = receipt.get("start", {})
    require(start.get("from") == "dual_snapshot" and start.get("layer") == processed[0]
            and start.get("installed_layers") == list(range(1, processed[0])), "native dual-snapshot resume ancestry mismatch")
    if processed[0] > lo:
        carried = receipt.get("resumed_from_receipt", {})
        require(carried.get("carried") is True and carried.get("layers_dropped") == {}
                and set(range(lo, processed[0])).issubset(carried.get("layers_carried_forward", [])),
                "partial-window recovery did not retain its actual validated native entries")
    require(receipt["arguments"].get("layers") == args.completed_window and receipt["arguments"].get("resume") is True
            and receipt["arguments"].get("src_dir") == "/parent" and receipt["arguments"].get("out") == "/run-output/fold"
            and receipt["arguments"].get("receipt_dir") == "/run-output/receipts"
            and receipt["arguments"].get("body_dir") == "/run-output/body", "actual native window/mount arguments changed")
    require(receipt.get("refusal") is None, "functional refusal remains in terminal receipt")
    require(receipt.get("peak_rss_bytes", 0) > 0, "actual native process peak measurement missing")
    require(set(k for k, v in receipt["layers"].items() if v.get("moe")) == {str(i) for i in range(1, hi + 1)}, "complete prior native MoE receipt chain required")
    prep = pins.json(args.prep_receipt, args.prep_receipt_sha256)
    require(prep.get("ok") is True and prep.get("all_files_fullhash_verified") is True, "actual separate-copy proof required")
    original_copies = {str(PurePosixPath(row["copy"]).relative_to(PurePosixPath(prep["identity"]["output_root"]))): row for row in prep["copies"]}
    installed, prefix, specs, native_checks, body = {}, [], [], [], []
    for layer in range(1, hi + 1):
        key = str(layer)
        entry = receipt["layers"][key]
        logical = f"/run-output/fold/dense/L{layer:02d}/ffn.safetensors"
        require(entry.get("moe") is True and entry["ffn"]["path"] == logical, "canonical native FFN path required")
        ffn_path = pins.file(mapped(output, logical), entry["ffn"]["sha256"])
        done_path = mapped(output, logical.replace("ffn.safetensors", "done.json"))
        done_sha = pins.observed(done_path)
        done = read(done_path)
        lineage = {"version": LINEAGE, "upstream_ffn_sha256": dict(installed)}
        require(done.get("layer") == layer and type(done.get("layer")) is int and done.get("moe") is True
                and done.get("artefact") == logical and done.get("artefact_sha256") == entry["ffn"]["sha256"]
                and guard.matches(done.get("resume_context"), done_path) and done.get("corpus_sha256") == CORPUS_SHA
                and type(done.get("rung")) is int and done["rung"] == 17408
                and done.get("prefix_lineage") == entry.get("prefix_lineage") == lineage, "done/receipt prefix identity mismatch")
        recorded_done_sha = entry["ffn"].get("done_marker_sha256")
        if recorded_done_sha is None:
            reused = entry.get("fit_skipped") or {}
            require(entry["ffn"].get("reused") is True and reused.get("done_marker") == logical.replace("ffn.safetensors", "done.json")
                    and reused.get("artefact") == logical and reused.get("artefact_sha256") == entry["ffn"]["sha256"],
                    "native reused-marker proof required for absent receipt marker hash")
        else:
            require(recorded_done_sha == done_sha, "actual done marker full hash mismatch")
        acceptance = done.get("fold_metrics", {}).get("acceptance")
        entry_acceptance = entry.get("acceptance")
        if entry["ffn"].get("reused") is True and isinstance(entry_acceptance, dict):
            entry_acceptance = {k: v for k, v in entry_acceptance.items() if k not in
                ("zero_child_prefix_this_run", "mode_this_run", "enforced_this_run")}
        require(isinstance(acceptance, dict) and canonical(acceptance) == canonical(entry_acceptance)
                and acceptance.get("version") == "k3_dense_pass.acceptance.v1"
                and acceptance.get("constants") == baseline["layers"]["1"]["acceptance"]["constants"],
                "historical numerical verdict not consistently retained")
        if layer <= 4:
            baseline_acceptance = baseline["layers"][key]["acceptance"]
            if baseline["layers"][key]["ffn"].get("reused") is True:
                baseline_acceptance = {k: v for k, v in baseline_acceptance.items() if k not in
                    ("zero_child_prefix_this_run", "mode_this_run", "enforced_this_run")}
            require(entry["ffn"]["sha256"] == baseline["layers"][key]["ffn"]["sha256"]
                    and done_sha == original_copies[f"fold/dense/L{layer:02d}/done.json"]["sha256"]
                    and acceptance.get("verdict") == "accepted"
                    and canonical(acceptance) == canonical(baseline_acceptance), "original accepted prefix changed")
        else:
            require(acceptance.get("mode") == "record" and acceptance.get("enforced") is False,
                    "new layer numeric diagnostics were enforced")
        h = header(ffn_path)
        meta = h["__metadata__"]
        require(meta.get("input_ids_sha256") == CORPUS_SHA and meta.get("K") == "17408" and meta.get("layer") == key,
                "FFN native corpus/rung/layer metadata mismatch")
        width = int(meta["width_installed"])
        require(0 < width <= cfg["num_experts"] * cfg["moe_intermediate_size"], "native installed width outside source channel pool")
        expected_shapes = native_ffn_shapes(width, cfg)
        require(set(h) == {"__metadata__", *expected_shapes}, "native exact ten-key Form-E tensor set required")
        for name, shape in expected_shapes.items():
            expected_dtype = "I32" if name == "channels" else "F32" if name == "w2p" else "BF16"
            require(h[name].get("shape") == shape and h[name].get("dtype") == expected_dtype,
                    "native Form-E exact shape/dtype mismatch: " + name)
        installed[key] = entry["ffn"]["sha256"]
        prefix.append({"layer": layer, "ffn_sha256": installed[key], "done_sha256": done_sha,
                       "numerical_verdict": acceptance.get("verdict"), "numerical_policy_enforced": acceptance.get("enforced")})
        if lo <= layer <= hi:
            checks = entry.get("finite_checks", [])
            require({x.get("tensor") for x in checks} >= FORWARD_CHECKS
                    and all(type(x.get("elements")) is int and x["elements"] > 0 and x.get("non_finite") == 0 for x in checks),
                    "actual native fit-input/target/teacher/folded output finite checks missing")
            native_checks.append({"layer": layer, "artifact_creation_run_id": done.get("run_id"),
                                  "forward_checks_run_id": receipt.get("run_id") if layer in processed else entry.get("run_id"),
                                  "forward_checks_carried_from_prior_receipt": layer not in processed, "finite_checks": checks})
            specs.append({"layer": layer, "path": str(ffn_path), "sha256": installed[key],
                          "width": width, "latent": cfg["routed_expert_hidden_size"],
                          "hidden": cfg["hidden_size"], "shared_channels": cfg["num_shared_experts"] * cfg["moe_intermediate_size"],
                          "source_routed_channels": cfg["num_experts"] * cfg["moe_intermediate_size"],
                          "source_experts": cfg["num_experts"], "source_expert_channels": cfg["moe_intermediate_size"],
                          "requested_rung": 17408})
        b = entry.get("body")
        require(isinstance(b, dict) and b.get("byte_exact") is True, "native verbatim body receipt missing")
        bpath = pins.file(mapped(output, b["path"]), b["sha256"])
        body.append({"layer": layer, "sha256": b["sha256"], "byte_exact": True,
                     "readback": finite_library_readback(bpath) if lo <= layer <= hi else None})
    require(receipt.get("installed_ffn_sha256") == installed, "terminal installed prefix digest set mismatch")
    prior_continuity = None
    if lo > 5:
        require(args.prior_continuity is not None and args.prior_continuity_sha256 is not None,
                "actual previously validated window continuity receipt required")
        require(args.prior_continuity_sha256 == guard.prior["sha256"]
                and args.prior_continuity.resolve(strict=True) == Path(guard.prior["path"]).resolve(strict=True),
                "exact authenticated sealed37-48 predecessor required")
        prior_continuity = pins.json(args.prior_continuity, args.prior_continuity_sha256)
        require(prior_continuity.get("schema") == "neural_forge.k3_actual_window_continuity.v1"
                and prior_continuity.get("completed_window") == "37-48"
                and prior_continuity.get("full_prefix_layers") == lo - 1 and prior_continuity.get("ffn_full_scan_complete") is True
                and guard.matches(prior_continuity.get("resume_context"), args.prior_continuity)
                and prior_continuity.get("original_record_plan_sha256") == args.first_plan_sha256
                and prior_continuity.get("original_accepted_receipt_sha256") == BASELINE_SHA,
                "preceding window was not functionally validated under this original context")
        prior_prefix = prior_continuity.get("prefix", [])
        require(len(prior_prefix) == lo - 1 and all(old.get("layer") == current["layer"]
                and old.get("ffn_sha256") == current["ffn_sha256"] and old.get("done_sha256") == current["done_sha256"]
                for old, current in zip(prior_prefix, prefix)), "previously validated prefix bytes changed")
        prior_scan_path = args.prior_continuity.parent / "inputs/full_ffn_validation.json"
        prior_scan = pins.json(prior_scan_path, prior_continuity["ffn_validation_sha256"])
        require(prior_scan.get("functional_artifacts_usable") is True and all(x.get("actual_library_load_verified") is True
                and x.get("all_coefficients_finite") is True and installed[str(x["layer"])] == x.get("sha256") for x in prior_scan["ffns"]),
                "retained preceding actual FFN scanner proof changed")
    partial_path = mapped(output, "/run-output/receipts/k3_dense_pass_receipt.partial.json")
    require(partial_path.is_file(), "terminal native partial receipt absent")
    pins.observed(partial_path)
    partial = read(partial_path)
    require(partial.get("receipt_sha256") == args.receipt_sha256 and partial.get("resume_context") == context
            and partial.get("installed_ffn_sha256") == installed and canonical(partial.get("layers")) == canonical(receipt["layers"])
            and partial.get("layers_processed_this_run") == processed,
            "newer native partial receipt could select a different context/prefix")
    preserved = []
    for row in prep["copies"]:
        relative = PurePosixPath(row["copy"]).relative_to(PurePosixPath(prep["identity"]["output_root"]))
        if relative.parts[0] == "body" or str(relative).startswith("fold/dense/L00/"):
            path = mapped(output, "/run-output/" + str(relative))
            pins.file(path, row["sha256"])
            preserved.append({"path": str(relative), "sha256": row["sha256"], "bytes": row["bytes"],
                              "readback": finite_library_readback(path) if hi == 92 and path.suffix == ".safetensors" else None})
    require(any(r["path"] == "body/model_level.safetensors" for r in preserved)
            and any(r["path"] == "fold/dense/L00/mlp.safetensors" for r in preserved), "original globals/L00 preservation proof absent")
    # The original before-L03/before-L05 pairs have already been copied outside
    # native snapshot retention. Reattest those preserved bytes, too.
    original_endpoints = []
    for row in prep["copies"]:
        name = PurePosixPath(row["copy"]).name
        if re.fullmatch(r"before_L(?:03|05)(?:_folded)?\.safetensors", name):
            path = guard.historical_path("ancestry/original_endpoints/" + name)
            pins.file(path, row["sha256"])
            original_endpoints.append({"path": str(path), "sha256": row["sha256"], "bytes": row["bytes"]})
    require({PurePosixPath(r["path"]).name for r in original_endpoints} == {"before_L03.safetensors", "before_L03_folded.safetensors",
            "before_L05.safetensors", "before_L05_folded.safetensors"}, "all required original captured endpoints must remain retained")
    manifest_path = mapped(output, "/run-output/fold/snapshots/snapshots_manifest.json")
    manifest_sha = pins.observed(manifest_path)
    manifest = read(manifest_path)
    endpoints, saved_head_outputs = [], []
    if hi < 92:
        next_layer = hi + 1
        require(receipt.get("end_of_window_snapshot", {}).get("layer") == next_layer, "actual end-of-window checkpoint absent")
        tokens = baseline["corpus"]["tokens"]
        blocks = (next_layer + cfg["attn_res_block_size"] - 1) // cfg["attn_res_block_size"]
        for stream, suffix in (("teacher", ""), ("folded", "_folded")):
            logical = f"/run-output/fold/snapshots/before_L{next_layer:02d}{suffix}.safetensors"
            m = manifest["files"][logical]
            require(m.get("layer") == next_layer and m.get("stream") == stream and m.get("installed") == installed
                    and m.get("corpus_sha256") == CORPUS_SHA and m.get("rung") == 17408
                    and m.get("run_id") == receipt["run_id"], "actual endpoint manifest lineage mismatch")
            end_ref = receipt["end_of_window_snapshot"]["files"][stream]
            require(end_ref["path"] == logical and end_ref["sha256"] == m["sha256"], "receipt endpoint full hash mismatch")
            path = pins.file(mapped(output, logical), m["sha256"])
            meta = {"resume_context": canonical(context), "installed_ffn_sha256": canonical(installed),
                    "installed_layers": str(hi), "prefix_lineage_version": LINEAGE, "input_ids_sha256": CORPUS_SHA,
                    "layer": str(next_layer), "stream": stream, "rung": "17408", "run_id": m["run_id"], "blocks": str(blocks)}
            evidence = finite_library_readback(path, {"prefix_sum": [tokens, cfg["hidden_size"]],
                                                       "block_residual": [tokens, blocks, cfg["hidden_size"]]}, meta)
            evidence.update(sha256=m["sha256"], stream=stream, layer=next_layer)
            endpoints.append(evidence)
        require(header(mapped(output, f"/run-output/fold/snapshots/before_L{next_layer:02d}.safetensors"))["prefix_sum"]["dtype"]
                == header(mapped(output, f"/run-output/fold/snapshots/before_L{next_layer:02d}_folded.safetensors"))["prefix_sum"]["dtype"],
                "teacher/folded endpoint dtype mismatch")
    else:
        require(receipt.get("end_of_stack") is not None and receipt.get("layers_folded") == list(range(1, 93)),
                "actual terminal full-stack native diagnostics/coverage absent")
        from safetensors import safe_open
        review_files = receipt["end_of_stack"]["review"]["files"]
        review_segments = [s for s in baseline["corpus"]["segments"] if s["role"] == "review"]
        require(set(review_files) == {PurePosixPath(s["path"]).stem for s in review_segments}, "complete saved native review corpus outputs required")
        for segment in review_segments:
            name = PurePosixPath(segment["path"]).stem
            with safe_open(str(mapped(corpus, segment["path"], "/inputs/corpus")), framework="pt", device="cpu") as handle:
                lengths = handle.get_tensor("real_lengths")
                positions = int((lengths - 1).clamp_min(0).sum().item())
                del lengths
            record = review_files[name]
            logical = f"/run-output/fold/review/{name}_teacher_forced.safetensors"
            require(record["path"] == logical and record.get("positions") == positions and positions > 0,
                    "native saved review path/position accounting mismatch")
            path = pins.file(mapped(output, logical), record["sha256"])
            shapes = {k: [positions] for k in ("teacher_next_token_logprob", "folded_next_token_logprob", "teacher_argmax",
                      "folded_argmax", "kl_teacher_folded", "row_index", "position")}
            shapes.update({k: [positions, 16] for k in ("teacher_topk_ids", "teacher_topk_logprobs", "folded_topk_ids", "folded_topk_logprobs")})
            dtypes = {k: "I32" if k.endswith("_ids") or k.endswith("argmax") or k in ("row_index", "position") else "F32" for k in shapes}
            evidence = finite_library_readback(path, shapes,
                {"input_ids_sha256": CORPUS_SHA, "file": name, "topk": "16"}, dtypes)
            evidence.update(sha256=record["sha256"], positions=positions,
                            scope="all stored teacher-forced head diagnostic tensors; unretained full-vocabulary logits are not reread")
            saved_head_outputs.append(evidence)
    request = {"schema": "neural_forge.k3_functional_ffn_request.v2", "ffns": specs}
    scanner = None
    if args.ffn_validation is not None:
        scanner = pins.json(args.ffn_validation, args.ffn_validation_sha256)
        require(scanner.get("schema") == "neural_forge.k3_functional_ffn_receipt.v2" and scanner.get("functional_artifacts_usable") is True,
                "actual full finite/load FFN scanner result required")
        found = {x["layer"]: x for x in scanner["ffns"]}
        require(set(found) == set(range(lo, hi + 1)), "scanner actual completed-window coverage mismatch")
        for spec in specs:
            record = found[spec["layer"]]
            require(record.get("sha256") == spec["sha256"] and Path(record.get("path", "")).resolve() == Path(spec["path"]).resolve()
                    and record.get("all_coefficients_finite") is True and record.get("actual_library_load_verified") is True
                    and record.get("functional_artifact_usable") is True, "scanner actual FFN hash/load/finite proof mismatch")
            native_entry = receipt["layers"][str(spec["layer"])]["ffn"]
            require(record.get("parameters") == native_entry.get("header_parameters")
                    == native_entry.get("parameters", {}).get("child_moe_total")
                    and record.get("bytes") == native_entry.get("bytes") and record.get("channels") == spec["width"]
                    and record.get("tensors", {}).get("channels", {}).get("elements") == 2 * spec["width"],
                    "scanner/native receipt exact tensor accounting mismatch")
            shapes = native_ffn_shapes(spec["width"], cfg)
            require(set(record.get("tensors", {})) == set(shapes) and all(record["tensors"][name].get("shape") == shape
                    and record["tensors"][name].get("nonfinite") == 0 for name, shape in shapes.items()),
                    "scanner exact native tensor coverage/shape mismatch")
    pins.stable()
    for path, stamp in shard_stamps.items():
        require(stamp == (path.stat().st_size, path.stat().st_mtime_ns), "source shard changed during verification")
    # Long full readback/copy work cannot invalidate finite retained artifacts.
    # Parent collects another authentic observation immediately before dispatch.
    fresh_observation(runtime, first["image_id"], context, check_age=False)
    # Preserve overwritten receipts and the validated next-window boundary.
    # Ordinary intermediate snapshots remain hash-referenced in the sealed
    # manifest; unchanged native retention may prune those owned intermediates.
    # Original captured L03/L05 anchors were independently sealed at preparation.
    seal = args.seal_dir
    require(seal.resolve().is_relative_to(output) and not seal.exists() and not seal.is_symlink(), "exclusive owned ancestry directory required")
    seal.mkdir(parents=True, exist_ok=False)
    files = []
    for path in [args.receipt, manifest_path, *sorted((output / "receipts").glob("k3_dense_pass_receipt*")),
                 *[Path(endpoint["path"]) for endpoint in endpoints]]:
        path = Path(path)
        if any(x["source"] == str(path) for x in files):
            continue
        rel = path.relative_to(output)
        destination = declared_write(output, seal / rel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = pins.observed(path)
        shutil.copy2(path, destination)
        note_output(output, destination)
        require(not destination.samefile(path) and sha(destination) == expected, "ancestry copy is not separate or full-byte identical")
        files.append({"source": str(path), "retained": str(destination), "sha256": expected, "bytes": destination.stat().st_size})
    for label, path in (("original_record_plan", args.first_plan), ("original_accepted_receipt", args.baseline_receipt),
                        ("native_runtime_observation", args.runtime), ("separate_copy_receipt", args.prep_receipt),
                        ("full_ffn_validation", args.ffn_validation), ("previous_window_continuity", args.prior_continuity)):
        if path is None:
            continue
        destination = declared_write(output, seal / "inputs" / (label + ".json"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = pins.observed(path)
        shutil.copy2(path, destination)
        note_output(output, destination)
        require(not destination.samefile(path) and sha(destination) == expected, "ancestry metadata copy mismatch")
        files.append({"source": str(path), "retained": str(destination), "sha256": expected, "bytes": destination.stat().st_size})
    result = {"schema": "neural_forge.k3_actual_window_continuity.v1", "observed_utc": datetime.now(timezone.utc).isoformat(),
              "completed_window": args.completed_window, "full_prefix_layers": hi, "prefix": prefix,
              "actual_layers_processed_this_terminal_run": processed,
              "receipt_sha256": args.receipt_sha256, "snapshot_manifest_sha256": manifest_sha,
              "original_accepted_receipt_sha256": BASELINE_SHA, "original_record_plan_sha256": args.first_plan_sha256,
              "prior_window_continuity_sha256": args.prior_continuity_sha256,
              "runtime_observation_sha256": args.runtime_sha256, "resume_context": context,
              "recovery_ancestry": guard.record(), "legacy_producer_context": legacy_context,
              "preserved_original_body_globals_L00": preserved, "preserved_original_capture_endpoints": original_endpoints,
              "native_forward_checks": native_checks,
              "body": body, "endpoint_readback": endpoints, "retained_ancestry": files,
              "snapshot_retention_scope": "dual verified next-window boundary plus independently sealed original L03/L05; ordinary intermediate hashes/metadata retained in manifest without duplicate payload copies",
              "saved_native_head_output_readback": saved_head_outputs,
              "ffn_full_scan_complete": scanner is not None, "capability_accepted": False, "training_eligible": False,
              "ffn_validation_sha256": args.ffn_validation_sha256,
              "numerical_targets_enforced": False, "native_whole_child_capability_checked": False,
              "source_digest_scope": "pinned dated full 98-file verification plus current 96 indexed shard size/mtime; no new shard rehash"}
    request_path = declared_write(output, seal / "ffn_request.json")
    with request_path.open("x", encoding="utf-8") as handle:
        json.dump(request, handle, indent=2, sort_keys=True); handle.write("\n")
    note_output(output, request_path)
    result["ffn_request_sha256"] = sha(request_path)
    result["ffn_request_path"] = str(seal / "ffn_request.json")
    result["authorized_NAS_window_only"] = "49-60"
    result["RO_resume_mapping"] = paths.record()
    continuity_path = declared_write(output, seal / "window_continuity.json")
    with continuity_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True); handle.write("\n")
    note_output(output, continuity_path)
    pins.stable()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("output", "first-plan", "baseline-receipt", "runtime", "nf", "parent", "corpus", "digest",
                "receipt", "prep-receipt", "seal-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("first-plan-sha256", "runtime-sha256", "receipt-sha256", "prep-receipt-sha256", "completed-window"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--ffn-validation", type=Path)
    parser.add_argument("--ffn-validation-sha256")
    parser.add_argument("--prior-continuity", type=Path)
    parser.add_argument("--prior-continuity-sha256")
    from ro_resume_runtime import add_resume_source_arguments
    add_resume_source_arguments(parser)
    args = parser.parse_args()
    require((args.ffn_validation is None) == (args.ffn_validation_sha256 is None), "scanner path/hash must be supplied together")
    require((args.prior_continuity is None) == (args.prior_continuity_sha256 is None), "preceding continuity path/hash must be supplied together")
    try:
        result = verify(args)
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        parser.exit(2, "actual window verification refused: " + str(exc) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
