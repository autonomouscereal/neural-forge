"""Read-only planner for the captured K3 L01..L04 prefix, never a GPU launcher.

The caller prepares a separate continuation copy and a fresh runtime observation.
This tool hashes/reads files, including safetensors headers, using only stdlib.
It cannot establish a new upstream model identity or change native acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct


TOOL = "reports/omni-campaign-2026-09/composite/tools/k3_dense_pass.py"
LANE = "reports/omni-campaign-2026-09/k3/"
CAPTURE_SHA = "853c6ed70bf8b8a4845da7597425ef7061585f5b8abe528509faf23aa39b3d8b"
PLAN_SHA = "0500111bbf2e334cf4e9b65017795997c4654ca87f69935313e54e59ce213542"
DIGEST_SHA = "58bc54f0c635ec78f53a4575155d249ebd10bde4ea302f65872bebe253d7a91f"
GPU_UUID = "GPU-f5a09535-fc85-a027-50f7-734b344747dd"
PREFIX_VERSION = "k3_dense_pass.prefix_lineage.v1"
CONSTANTS = {"G1_signal_margin": 0.9, "G2_memorisation_ratio_max": 1.5,
             "G3_prefix_margin": 0.95, "G4_window_kl_per_layer_budget_x": 2.0,
             "G4_window_top1_floor_x": 0.5, "exact_fold_floor_relL2": 1e-3}
RUNTIME_MAX_AGE_SECONDS = 300


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def canonical(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True)


def read_json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    require(isinstance(result, dict), f"JSON object required: {path}")
    return result


class Pins:
    def __init__(self):
        self.stamps = {}

    def file(self, path: Path, expected: str) -> Path:
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                "SHA256 digest required")
        path = path.resolve(strict=True)
        before = path.stat()
        require(path.is_file(), f"regular input file required: {path}")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        require(actual == expected, f"hash mismatch: {path}")
        stamp = (before.st_size, before.st_mtime_ns)
        require(stamp == (path.stat().st_size, path.stat().st_mtime_ns), f"input changed while reading: {path}")
        if path in self.stamps:
            require(self.stamps[path] == stamp, f"input changed during preflight: {path}")
        self.stamps[path] = stamp
        return path

    def json(self, path: Path, expected: str) -> dict:
        return read_json(self.file(path, expected))

    def stable(self):
        for path, stamp in self.stamps.items():
            require(stamp == (path.stat().st_size, path.stat().st_mtime_ns), f"input changed during preflight: {path}")


def mapped(root: Path, logical: str, prefix: str) -> Path:
    require(isinstance(logical, str), "recorded path required")
    path, base = PurePosixPath(logical), PurePosixPath(prefix)
    require(path.is_absolute() and str(path) == logical and ".." not in path.parts,
            "recorded paths must be canonical absolute POSIX paths")
    require(path.is_relative_to(base) and path != base, f"source alias refused: expected {prefix}")
    physical = root.joinpath(*path.relative_to(base).parts).resolve(strict=True)
    require(physical.is_relative_to(root), "input path escapes its mount")
    return physical


def header(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(8)
        require(len(raw) == 8, "truncated safetensors header")
        length = struct.unpack("<Q", raw)[0]
        require(2 <= length <= 16 * 1024 * 1024 and length + 8 <= path.stat().st_size,
                "invalid safetensors header length")
        value = json.loads(handle.read(length))
    require(isinstance(value, dict), "safetensors header object required")
    return value


def accepted(value: dict) -> bool:
    return (isinstance(value, dict) and value.get("version") == "k3_dense_pass.acceptance.v1"
            and value.get("constants") == CONSTANTS and value.get("verdict") == "accepted")


def build_plan(args: argparse.Namespace) -> dict:
    roots = {key: Path(getattr(args, key)).resolve(strict=True)
             for key in ("nf", "parent", "corpus", "accepted", "sealed", "output")}
    for path in roots.values():
        require(path.is_dir(), f"directory required: {path}")
    output = roots["output"]
    for key in ("nf", "parent", "corpus", "accepted", "sealed"):
        path = roots[key]
        require(not (output == path or output.is_relative_to(path) or path.is_relative_to(output)),
                "continuation output must not overlap an accepted/source/input mount")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", args.image) is not None,
            "immutable inspected image ID required")
    pins = Pins()
    capture = pins.json(Path(args.capture), CAPTURE_SHA)
    plan = pins.json(Path(args.capture_plan), PLAN_SHA)
    require(capture.get("schema") == "neural-forge.c17-k3-boundaries.v1" and capture.get("fixture") is False
            and capture.get("accepted_prefix_replay_exact") is True and capture.get("full_corpus_replayed") is True,
            "pinned production prefix replay required")
    require(plan.get("schema") == "neural-forge.c17-k3-capture-plan.v1" and plan.get("fixture") is False
            and plan.get("layers") == [3, 4], "only the pinned L03..L04 capture plan is supported")
    require(capture["capture_plan"]["sha256"] == PLAN_SHA, "capture plan binding mismatch")
    for key in ("accepted_receipt", "snapshot_manifest"):
        require(plan[key] == capture[key], f"{key} binding mismatch")
    receipt = pins.json(mapped(roots["sealed"], plan["accepted_receipt"]["path"], "/sealed"),
                        plan["accepted_receipt"]["sha256"])
    manifest = pins.json(mapped(roots["sealed"], plan["snapshot_manifest"]["path"], "/sealed"),
                         plan["snapshot_manifest"]["sha256"])
    context = capture["accepted_resume_context"]
    require(context.get("version") == "k3_dense_pass.resume_context.v1" and receipt.get("resume_context") == context,
            "accepted source/runtime/corpus context mismatch")
    require(context["model"]["source_directory"] == "/parent", "unresolved/aliased parent source identity")
    require(context["corpus"] == capture["corpus_sha256"] == receipt["corpus"]["corpus_sha256"],
            "corpus context mismatch")
    require(type(receipt["arguments"]["rung"]) is int and receipt["arguments"]["rung"] == 17408,
            "accepted native rung mismatch")
    rung = receipt["arguments"]["rung"]
    runtime, software = context["runtime"], context["software"]
    require(runtime["devices"] == [{"capability": "sm75", "index": 0, "name": "Quadro RTX 8000", "total_GB": 47.6}]
            and runtime["device"] == runtime["experts_device"] == "cuda:0"
            and runtime["resid_device"] == "cpu" and runtime["kda_backend"] == "torch",
            "original RTX 8000 native runtime required; device substitution refused")
    observed = pins.json(Path(args.runtime), args.runtime_sha256)
    require(observed.get("schema") == "neural-forge.k3-runtime-observation.v1"
            and observed.get("image_id") == args.image and observed.get("gpu_uuid") == GPU_UUID
            and observed.get("runtime") == runtime and observed.get("software") == software,
            "runtime observation differs from accepted image/card/software/context")
    observed_at = datetime.fromisoformat(observed["observed_utc"].replace("Z", "+00:00"))
    require(observed_at.tzinfo is not None, "runtime observation needs timezone")
    age = (datetime.now(timezone.utc) - observed_at).total_seconds()
    require(0 <= age <= RUNTIME_MAX_AGE_SECONDS, "runtime observation stale or future-dated; collect a fresh observation")
    pins.file(roots["nf"] / TOOL, software["tool_sha256"])
    require(set(software["lane_sha256"]) == {"k3_common.py", "k3_layer.py", "k3_stream.py",
            "c14_union_coreset.py", "k3_receipt.py", "k3_review_stream.py"}, "complete pinned native lane required")
    for name, sha in software["lane_sha256"].items():
        pins.file(roots["nf"] / LANE / name, sha)
    config = pins.json(roots["parent"] / "config.json", context["model"]["config_sha256"])
    index = pins.json(roots["parent"] / "model.safetensors.index.json", context["model"]["index_sha256"])
    cfg = config.get("text_config", config)
    require(config.get("synthetic") is not True and cfg["num_hidden_layers"] == 93
            and cfg.get("first_k_dense_replace", 0) == 1 and cfg.get("moe_layer_freq", 1) == 1
            and cfg.get("num_experts") is not None, "pinned 93-layer source layout required")
    metadata = context["model"]["source_file_metadata"]
    require(set(metadata) == set(index["weight_map"].values()) and len(metadata) == 96,
            "complete accepted source shard metadata required")
    shard_stamps = {}
    for name, row in metadata.items():
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", name) is not None and name not in (".", ".."),
                "source shard names must be plain basenames")
        shard = (roots["parent"] / name).resolve(strict=True)
        require(shard.parent == roots["parent"] and shard.is_file(), "source shard escapes parent root")
        stat = shard.stat()
        require(stat.st_size == row["bytes"] and stat.st_mtime_ns == row["mtime_ns"],
                f"source metadata changed since accepted resume context: {name}")
        shard_stamps[shard] = (stat.st_size, stat.st_mtime_ns)
    digest = pins.json(Path(args.digest), DIGEST_SHA)
    require(digest.get("complete") is True and digest.get("all_verified") is True
            and len(digest["files"]) == 98, "complete dated 98-file digest receipt required")
    digest_names = [row["file"] for row in digest["files"]]
    require(all(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                and name not in (".", "..") for name in digest_names), "digest filenames must be plain basenames")
    require(len(set(digest_names)) == 98 and set(metadata).issubset(digest_names),
            "digest does not cover the exact source shards or has duplicate names")
    for row in digest["files"]:
        path = (roots["parent"] / row["file"]).resolve(strict=True)
        require(path.parent == roots["parent"] and path.is_file(), "digest file escapes parent root")
        require(path.stat().st_size == row["local_size"], "source size changed since dated digest receipt")
    corpus_refs = [plan["input_ids"], *plan["review_input_ids"]]
    require(len(corpus_refs) == 3 and [r["path"] for r in corpus_refs] ==
            [s["path"] for s in capture["corpus"]["segments"]], "corpus order/basename identity mismatch")
    for ref in corpus_refs:
        pins.file(mapped(roots["corpus"], ref["path"], "/inputs/corpus"), ref["sha256"])
    require(set(plan["accepted_layers"]) == set(capture["installed_ffn_sha256"]) == {"1", "2", "3", "4"},
            "complete accepted L01..L04 prefix required")
    installed = {}
    copied = []

    def copy_proof(source: Path, logical: str, sha: str):
        dest = mapped(output, logical, "/run-output")
        require(not dest.samefile(source), "continuation copy must not alias an accepted input file")
        pins.file(dest, sha)
        copied.append(logical)

    for i in range(1, 5):
        key = str(i)
        record, refs = receipt["layers"][key], plan["accepted_layers"][key]
        sha = capture["installed_ffn_sha256"][key]
        require(refs["ffn"]["sha256"] == record["ffn"]["sha256"] == sha, "accepted FFN pin mismatch")
        source = pins.file(mapped(roots["accepted"], refs["ffn"]["path"], "/accepted"), sha)
        done_path = mapped(roots["accepted"], refs["done"]["path"], "/accepted")
        done = pins.json(done_path, refs["done"]["sha256"])
        logical = f"/run-output/fold/dense/L{i:02d}/ffn.safetensors"
        lineage = {"version": PREFIX_VERSION, "upstream_ffn_sha256": dict(installed)}
        require(type(done.get("layer")) is int and done["layer"] == i and done.get("moe") is True
                and done.get("artefact") == logical and done.get("artefact_sha256") == sha
                and record["ffn"]["path"] == logical, "done marker artifact/layer path mismatch or alias")
        require(done.get("resume_context") == context and done.get("corpus_sha256") == context["corpus"]
                and type(done.get("rung")) is int and done["rung"] == rung, "done marker source/runtime/corpus/rung mismatch")
        require(done.get("prefix_lineage") == record.get("prefix_lineage") == lineage,
                "done marker/receipt prefix ancestry mismatch")
        require(accepted(done.get("fold_metrics", {}).get("acceptance")) and accepted(record.get("acceptance")),
                "prefix native acceptance missing or changed")
        require(record["ffn"].get("done_marker_sha256", refs["done"]["sha256"]) == refs["done"]["sha256"],
                "done marker receipt hash mismatch")
        copy_proof(source, logical, sha)
        copy_proof(done_path, logical.replace("ffn.safetensors", "done.json"), refs["done"]["sha256"])
        installed[key] = sha
    for stream in ("teacher", "folded"):
        key = "end_" + stream
        ref, captured = plan[key], capture["snapshots"][key]
        require(ref["path"] == captured["path"] and ref["manifest_key"] == captured["manifest_key"],
                "snapshot path/manifest binding mismatch")
        entry = manifest["files"][ref["manifest_key"]]
        require(type(entry["layer"]) is int and entry["layer"] == 5 and entry["stream"] == stream
                and entry["installed"] == installed and entry["corpus_sha256"] == context["corpus"]
                and type(entry["rung"]) is int and entry["rung"] == rung
                and entry["sha256"] == captured["sha256"], "snapshot manifest context/ancestry mismatch")
        source = pins.file(mapped(roots["accepted"], ref["path"], "/accepted"), entry["sha256"])
        got = header(source)
        meta = got.get("__metadata__", {})
        require(meta == captured["metadata"] and meta.get("resume_context") == canonical(context)
                and meta.get("installed_ffn_sha256") == canonical(installed) and meta.get("installed_layers") == "4"
                and meta.get("prefix_lineage_version") == PREFIX_VERSION and meta.get("input_ids_sha256") == context["corpus"]
                and meta.get("layer") == "5" and meta.get("stream") == stream
                and meta.get("rung") == str(rung) and meta.get("run_id") == entry["run_id"], "snapshot metadata mismatch")
        tokens, hidden = capture["corpus"]["tokens"], cfg["hidden_size"]
        blocks = (5 + cfg["attn_res_block_size"] - 1) // cfg["attn_res_block_size"]
        require(set(got) == {"__metadata__", "prefix_sum", "block_residual"}
                and got["prefix_sum"]["shape"] == [tokens, hidden]
                and got["block_residual"]["shape"] == [tokens, blocks, hidden]
                and got["prefix_sum"]["dtype"] in ("F16", "BF16", "F32")
                and got["prefix_sum"]["dtype"] == got["block_residual"]["dtype"]
                and meta.get("blocks") == str(blocks), "snapshot shape/dtype mismatch")
        copy_proof(source, ref["manifest_key"], entry["sha256"])
    copy_proof(mapped(roots["sealed"], plan["accepted_receipt"]["path"], "/sealed"),
               "/run-output/receipts/k3_dense_pass_receipt.json", plan["accepted_receipt"]["sha256"])
    require(not (output / "receipts/k3_dense_pass_receipt.partial.json").exists(),
            "partial receipt could override pinned accepted receipt; prepare a clean continuation copy")
    copy_proof(mapped(roots["sealed"], plan["snapshot_manifest"]["path"], "/sealed"),
               "/run-output/fold/snapshots/snapshots_manifest.json", plan["snapshot_manifest"]["sha256"])
    for layer in range(5, 93):
        require(not (output / f"fold/dense/L{layer:02d}").exists()
                and not (output / f"body/L{layer:02d}").exists(), "continuation copy already contains later layer artifacts")
    for path in (output / "fold/snapshots").glob("before_L*.safetensors"):
        match = re.fullmatch(r"before_L(\d+)(?:_folded)?\.safetensors", path.name)
        require(match is not None and int(match[1]) <= 5, "later snapshot could select a different resume point")
    pins.stable()
    for path, stamp in shard_stamps.items():
        require(stamp == (path.stat().st_size, path.stat().st_mtime_ns), "source shard changed during preflight")
    require(0 <= (datetime.now(timezone.utc) - observed_at).total_seconds() <= RUNTIME_MAX_AGE_SECONDS,
            "runtime observation expired during preflight; collect a fresh observation")
    mounts = [{"source": str(roots[key]), "target": target, "read_only": True}
              for key, target in (("nf", "/code/neural-forge"), ("parent", "/parent"), ("corpus", "/inputs/corpus"))]
    mounts += [{"source": str(Path(args.digest).resolve()), "target": "/inputs/digest.json", "read_only": True},
               {"source": str(output), "target": "/run-output", "read_only": False}]
    argv = ["env", *[f"{name}={runtime['num_threads']}" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")],
            "python", "/code/neural-forge/" + TOOL, "--src-dir", "/parent", "--input-ids", corpus_refs[0]["path"]]
    for ref in corpus_refs[1:]:
        argv += ["--review-input-ids", ref["path"]]
    argv += ["--out", "/run-output/fold", "--receipt-dir", "/run-output/receipts", "--body-dir", "/run-output/body",
             "--digest-receipt", "/inputs/digest.json", "--no-share-read", "--resume", "--layers", "5-12", "--rung", str(rung)]
    for key in ("device", "experts_device", "resid_device", "compute_dtype", "kda_backend", "k_pool_factor",
                "row_chunk", "token_chunk", "chunk", "holdout_blocks", "divergence_rows", "review_topk"):
        argv += ["--" + key.replace("_", "-"), str(runtime[key])]
    argv += ["--max-gram-gb", "160", "--max-device-gb", "-1", "--snapshot-every", "12", "--snapshot-keep", "2", "--acceptance", "gate"]
    return {"schema": "neural-forge.k3-continuation-preflight.v1", "action": "plan_only_no_allocation_or_launch",
            "image_id": args.image, "required_gpu_uuid": GPU_UUID, "mounts": mounts, "command_argv": argv,
            "next_layers": "5-12", "remaining_moe_layers": 88, "installed_ffn_sha256": installed,
            "capture_sha256": CAPTURE_SHA, "accepted_receipt_sha256": plan["accepted_receipt"]["sha256"],
            "runtime_observation_sha256": args.runtime_sha256, "runtime_observed_utc": observed["observed_utc"],
            "source_identity": {"config_sha256": context["model"]["config_sha256"], "index_sha256": context["model"]["index_sha256"],
                                "scope": context["source_identity_scope"], "digest_scope": "dated full verification plus current size/mtime; no new shard rehash"},
            "corpus_sha256": context["corpus"], "verified_continuation_files": copied,
            "later_windows_conditional_on_new_receipt_and_snapshot_review": [f"{start}-{min(start+11,92)}" for start in range(13, 93, 12)],
            "limitations": ["runtime observation is caller-collected and hash-pinned; no live device inspection or allocation",
                            "future windows need fresh complete prefix proofs; this plan approves no future artifacts",
                            "continuation does not establish upstream Kimi K3 identity or an accepted whole dense child",
                            "preserve the existing L00/body/model-level artifacts separately; a late window does not regenerate them",
                            "the native first G4 window calibrates its baseline; calibration alone is not independent whole-child proof"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("nf", "parent", "corpus", "accepted", "sealed", "output", "capture", "capture-plan", "digest",
                "runtime", "runtime-sha256", "image"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    try:
        result = build_plan(args)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.exit(2, f"continuation preflight refused: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
