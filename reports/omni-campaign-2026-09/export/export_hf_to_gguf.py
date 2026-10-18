#!/usr/bin/env python3
"""HF checkpoint (qwen3_5 layout, or any arch llama.cpp converts) -> GGUF -> Q4_K_M / IQ4_XS.

    convert_hf_to_gguf.py  ->  <tag>.<outtype>.gguf   (the full-precision GGUF)
    llama-quantize         ->  <tag>.Q4_K_M.gguf, <tag>.IQ4_XS.gguf

Writes one canonical-JSON receipt per export with: the source directory digest (same
construction the fold/train receipts use, so the export joins to its lineage), every
command line verbatim, per-stage wall clock, output sizes and sha256s, and the tool
versions. Nothing is deleted and nothing is overwritten unless --force is given.

The quantization rule for this campaign: Q4 happens ONCE, at the very end, and the child
is re-graded after it. This script does not enforce that (it is a tool, not a gate) but it
records `quantization_stage` in the receipt so the ledger can see which children were
quantized and when.

Examples
--------
  python3 export_hf_to_gguf.py \
      --hf-dir /home/cereal/models/Qwen3.5-0.8B \
      --out-dir /home/cereal/omni-campaign/export/gguf \
      --tag Q35-0.8B-stock \
      --outtype bf16 --quants Q4_K_M,IQ4_XS

  # dry run: print the exact commands, touch nothing
  python3 export_hf_to_gguf.py --hf-dir ... --out-dir ... --tag ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_probe import (  # noqa: E402
    directory_digest,
    run,
    safe_tag,
    sha256_file,
    utc_now,
    write_receipt,
)

DEFAULT_ROOT = "/home/cereal/omni-campaign/export/llamacpp"
KNOWN_QUANTS = (
    "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q4_0",
    "IQ4_XS", "IQ4_NL", "Q3_K_M", "MOSTLY_F16",
)
KNOWN_OUTTYPES = ("f32", "f16", "bf16", "q8_0", "auto")


def parse_quants(text):
    quants = []
    for piece in str(text).replace(" ", "").split(","):
        if not piece:
            continue
        name = piece.upper()
        if name not in KNOWN_QUANTS:
            raise ValueError(
                "unknown quant type %r (known: %s)" % (piece, ", ".join(KNOWN_QUANTS))
            )
        quants.append(name)
    # preserve order, drop duplicates
    seen = set()
    ordered = []
    for name in quants:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


# safe_tag lives in llama_probe now: envelope_bench, parity_check, p40probe_check and
# fa_negative_control interpolate --tag into paths too, and only this file validated it.

def gguf_name(tag, kind):
    return "%s.%s.gguf" % (safe_tag(tag), kind)


# Multimodal / conditional-generation checkpoints (Qwen3_5ForConditionalGeneration is
# exactly one) keep the text tower's architecture in a nested sub-config. A flat
# config.get() reads None for every field on precisely the family this campaign targets,
# and a receipt full of nulls is not a lineage record. Search order matters: the first
# sub-config that has the field wins, and where it came from is recorded.
SUB_CONFIG_KEYS = ("text_config", "llm_config", "thinker_config", "language_config",
                   "decoder_config", "llm_cfg")
REQUIRED_ARCH_FIELDS = ("num_hidden_layers", "hidden_size", "vocab_size",
                        "max_position_embeddings")
OPTIONAL_ARCH_FIELDS = ("tie_word_embeddings", "num_attention_heads", "num_key_value_heads",
                        "head_dim", "intermediate_size", "attn_output_gate",
                        "full_attention_interval", "rope_parameters", "dtype", "torch_dtype")


def _resolve_field(config, field):
    """(value, source) for `field`, looking at the top level then each sub-config.

    A field present at BOTH levels is reported from the sub-config, because that is the
    value the text tower actually uses -- but the top-level value is kept separately in the
    receipt so a disagreement is visible instead of silently resolved.
    """
    for key in SUB_CONFIG_KEYS:
        sub = config.get(key)
        if isinstance(sub, dict) and field in sub:
            return sub[field], key
    if field in config:
        return config[field], "config.json"
    return None, None


def read_hf_arch(hf_dir):
    """Architecture record for the export receipt, with nested sub-configs resolved.

    Returns resolved values, where each came from, the raw top-level values, the raw
    config.json sha256, and a LOUD warnings list for anything that could not be resolved.
    """
    config_path = os.path.join(hf_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    sub_configs = [key for key in SUB_CONFIG_KEYS if isinstance(config.get(key), dict)]
    resolved = {}
    sources = {}
    warnings = []
    for field in REQUIRED_ARCH_FIELDS + OPTIONAL_ARCH_FIELDS:
        value, source = _resolve_field(config, field)
        resolved[field] = value
        sources[field] = source
        if value is None and field in REQUIRED_ARCH_FIELDS:
            warnings.append(
                "REQUIRED architecture field %r resolved to None (top level and %s); the "
                "export receipt has no usable %s for lineage joins"
                % (field, sub_configs or "no sub-configs", field)
            )

    layer_types, layer_types_source = _resolve_field(config, "layer_types")
    record = {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "config_sha256": sha256_file(config_path),
        "sub_configs_present": sub_configs,
        "resolved": resolved,
        "resolved_from": sources,
        "top_level": {
            field: config.get(field)
            for field in REQUIRED_ARCH_FIELDS + OPTIONAL_ARCH_FIELDS
        },
        "layer_types_len": len(layer_types) if isinstance(layer_types, list) else None,
        "layer_types_from": layer_types_source,
        "layer_types_counts": (
            {name: layer_types.count(name) for name in sorted(set(layer_types))}
            if isinstance(layer_types, list) else None
        ),
        "warnings": warnings,
    }
    # Flat aliases so older readers of this receipt keep working, now filled from the
    # RESOLVED value rather than the top level.
    for field in REQUIRED_ARCH_FIELDS + ("tie_word_embeddings",):
        record[field] = resolved.get(field)
    return record


def load_prior_receipt(path):
    """The receipt a previous run of this tag left behind, or None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (ValueError, OSError):
        return None


def prior_source_matches(prior, hf_dir, source_digest):
    """Does the prior receipt describe the SAME source checkpoint as this run?

    Reuse is what makes an export receipt cheap to re-emit, and it is also how an export
    receipt ends up describing a checkpoint its artifacts did not come from: same --tag,
    different --hf-dir, old GGUFs reused, fresh source digest written over them. Returns
    (verdict, reasons) with verdict True / False / None (nothing recorded to compare).
    """
    if not prior:
        return None, ["no prior receipt for this tag"]
    reasons = []
    compared = 0
    mismatch = False
    prior_dir = prior.get("hf_dir")
    if prior_dir:
        compared += 1
        if os.path.abspath(prior_dir) != os.path.abspath(hf_dir):
            mismatch = True
            reasons.append("hf_dir differs: prior %s != now %s" % (prior_dir, hf_dir))
    else:
        reasons.append("prior receipt records no hf_dir")
    prior_digest = ((prior.get("source_directory_digest") or {}).get("directory_sha256"))
    now_digest = (source_digest or {}).get("directory_sha256")
    if prior_digest and now_digest:
        compared += 1
        if prior_digest != now_digest:
            mismatch = True
            reasons.append("source_directory_sha256 differs: prior %s != now %s"
                           % (prior_digest[:16], now_digest[:16]))
    else:
        reasons.append("no source directory digest on both sides to compare")
    if mismatch:
        return False, reasons
    if compared:
        reasons.append("%d source field(s) matched" % compared)
        return True, reasons
    return None, reasons


def carry_provenance(prior, key, path, digest, source_matches=True):
    """Provenance for an artifact this run did NOT produce.

    Stamping the CURRENT build's hash onto a file an earlier build produced is an
    unverified lineage join -- the shipped export receipt did exactly that, carrying a
    build_receipt_sha256 written after its own GGUFs existed. Provenance is carried forward
    from the prior receipt only when that receipt's recorded sha256 still matches the bytes
    on disk; otherwise the artifact is honestly marked unknown.
    """
    record = {
        "path": path,
        "bytes": os.path.getsize(path),
        "sha256": digest,
        "produced_by": "reused_unknown_provenance",
        "produced_at": None,
        "produced_by_build_receipt_sha256": None,
        "produced_by_llama_cpp_commit": None,
        "produced_from_hf_dir": None,
        "produced_from_source_directory_sha256": None,
    }
    previous = ((prior or {}).get("artifacts") or {}).get(key) or {}
    if previous.get("sha256") == digest:
        # `reused_from_a_different_source` is the --allow-mismatched-reuse case: the
        # carried provenance is TRUE (that is where the bytes came from) and it is not
        # this run's source, which is exactly what a reader has to see.
        record["produced_by"] = "reused" if source_matches else "reused_from_a_different_source"
        record["produced_at"] = previous.get("produced_at") or (prior or {}).get("created_at")
        record["produced_by_build_receipt_sha256"] = (
            previous.get("produced_by_build_receipt_sha256")
            or (prior or {}).get("build_receipt_sha256")
        )
        record["produced_by_llama_cpp_commit"] = (
            previous.get("produced_by_llama_cpp_commit") or (prior or {}).get("llama_cpp_commit")
        )
        record["produced_from_hf_dir"] = (
            previous.get("produced_from_hf_dir") or (prior or {}).get("hf_dir")
        )
        record["produced_from_source_directory_sha256"] = (
            previous.get("produced_from_source_directory_sha256")
            or ((prior or {}).get("source_directory_digest") or {}).get("directory_sha256")
        )
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-dir", required=True, help="HF checkpoint directory")
    parser.add_argument("--out-dir", required=True, help="where the GGUFs and the receipt go")
    parser.add_argument("--tag", required=True, help="child tag, becomes the filename stem")
    parser.add_argument("--llamacpp-root", default=DEFAULT_ROOT,
                        help="root produced by build_llamacpp_cuda.sh (bin/ + llama.cpp/)")
    parser.add_argument("--python", default=sys.executable,
                        help="python that runs convert_hf_to_gguf.py (needs torch + gguf)")
    parser.add_argument("--gguf-py-path", default=None,
                        help="prepend this to PYTHONPATH if gguf is not pip-installed")
    parser.add_argument("--outtype", default="bf16", choices=KNOWN_OUTTYPES)
    parser.add_argument("--quants", default="Q4_K_M,IQ4_XS")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument("--convert-timeout", type=int, default=10800)
    parser.add_argument("--quantize-timeout", type=int, default=10800)
    parser.add_argument("--skip-digest", action="store_true",
                        help="skip the source directory digest (it re-hashes every shard)")
    parser.add_argument("--quantization-stage", default="final",
                        help="recorded in the receipt: which campaign stage authorised this Q4")
    parser.add_argument("--force", action="store_true", help="overwrite existing GGUFs")
    parser.add_argument("--allow-mismatched-reuse", action="store_true",
                        help="reuse GGUFs whose recorded source does not match --hf-dir; "
                             "each such artifact is stamped provenance "
                             "'reused_unknown_provenance' instead of this run's build hash")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    tag = safe_tag(args.tag)
    quants = parse_quants(args.quants)
    hf_dir = os.path.abspath(args.hf_dir)
    out_dir = os.path.abspath(args.out_dir)
    bin_dir = os.path.join(args.llamacpp_root, "bin")
    converter = os.path.join(args.llamacpp_root, "llama.cpp", "convert_hf_to_gguf.py")
    quantizer = os.path.join(bin_dir, "llama-quantize")

    full_path = os.path.join(out_dir, gguf_name(tag, args.outtype))
    convert_cmd = [
        args.python, converter,
        "--outtype", args.outtype,
        "--outfile", full_path,
        hf_dir,
    ]
    plan = {
        "convert": convert_cmd,
        "quantize": [
            [quantizer, full_path, os.path.join(out_dir, gguf_name(tag, quant)), quant, str(args.threads)]
            for quant in quants
        ],
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    for path in (converter, quantizer):
        if not os.path.exists(path):
            print("MISSING TOOL: %s  (run build_llamacpp_cuda.sh first)" % path, file=sys.stderr)
            return 3
    if not os.path.isdir(hf_dir):
        print("MISSING HF DIR: %s" % hf_dir, file=sys.stderr)
        return 3
    os.makedirs(out_dir, exist_ok=True)

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = bin_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    # gguf-py travels with the source tree we built. Using it via PYTHONPATH keeps the
    # shared lab venv untouched (the build script no longer pip-installs into it by
    # default) and guarantees the converter runs against the gguf-py of the SAME commit.
    gguf_py_path = args.gguf_py_path
    if not gguf_py_path:
        bundled = os.path.join(args.llamacpp_root, "llama.cpp", "gguf-py")
        if os.path.isdir(bundled):
            gguf_py_path = bundled
    if gguf_py_path:
        env["PYTHONPATH"] = gguf_py_path + ":" + env.get("PYTHONPATH", "")

    hf_config = read_hf_arch(hf_dir)
    for warning in hf_config["warnings"]:
        print("WARNING (recorded in the receipt): %s" % warning, file=sys.stderr)
    print("hf arch: %s (%s) layers=%s hidden=%s vocab=%s ctx=%s  [sub-configs: %s]" % (
        (hf_config["architectures"] or ["?"])[0], hf_config["model_type"],
        hf_config["num_hidden_layers"], hf_config["hidden_size"],
        hf_config["vocab_size"], hf_config["max_position_embeddings"],
        ", ".join(hf_config["sub_configs_present"]) or "none"))

    stages = []
    receipt = {
        "kind": "gguf_export",
        "hf_config_warnings": hf_config["warnings"],
        "created_at": utc_now(),
        "tag": tag,
        "hf_dir": hf_dir,
        "out_dir": out_dir,
        "outtype": args.outtype,
        "quants": quants,
        "threads": args.threads,
        "quantization_stage": args.quantization_stage,
        "llamacpp_root": args.llamacpp_root,
        "gguf_py_path_used": gguf_py_path,
        "hf_config": hf_config,
        "stages": stages,
        "artifacts": {},
    }

    build_receipt_path = os.path.join(args.llamacpp_root, "build-receipt.json")
    if os.path.exists(build_receipt_path):
        receipt["build_receipt_sha256"] = sha256_file(build_receipt_path)
        with open(build_receipt_path, "r", encoding="utf-8") as handle:
            built = json.load(handle)
        receipt["llama_cpp_commit"] = built.get("llama_cpp_commit")
        receipt["cuda_architectures"] = built.get("cuda_architectures")

    if not args.skip_digest:
        started = time.time()
        receipt["source_directory_digest"] = directory_digest(hf_dir)
        stages.append({"stage": "source_digest", "rc": 0, "wall_s": round(time.time() - started, 2)})

    # Reuse is only safe against the SAME source. A prior receipt for this tag is loaded
    # and compared before any existing file is adopted.
    receipt_path = os.path.join(out_dir, "export-receipt-%s.json" % tag)
    prior = load_prior_receipt(receipt_path)
    source_match, source_reasons = prior_source_matches(
        prior, hf_dir, receipt.get("source_directory_digest"))
    receipt["prior_receipt"] = {
        "path": receipt_path if prior else None,
        "created_at": (prior or {}).get("created_at"),
        "sha256": sha256_file(receipt_path) if prior else None,
        "source_matches_this_run": source_match,
        "reasons": source_reasons,
    }
    reuse_allowed = not (source_match is False) or args.allow_mismatched_reuse
    if source_match is False:
        print("PRIOR RECEIPT SOURCE MISMATCH: %s" % "; ".join(source_reasons), file=sys.stderr)
        if not args.allow_mismatched_reuse:
            print("refusing to reuse artifacts produced from a different source; pass "
                  "--force to rebuild them, a new --tag, or --allow-mismatched-reuse to "
                  "keep them with provenance 'reused_unknown_provenance'.", file=sys.stderr)
    receipt["reuse_allowed"] = reuse_allowed
    artifact_stamp = {
        "produced_by": "this_run",
        "produced_at": utc_now(),
        "produced_by_build_receipt_sha256": receipt.get("build_receipt_sha256"),
        "produced_by_llama_cpp_commit": receipt.get("llama_cpp_commit"),
        "produced_from_hf_dir": hf_dir,
        "produced_from_source_directory_sha256": (
            (receipt.get("source_directory_digest") or {}).get("directory_sha256")),
    }

    # ---------------------------------------------------------------- convert
    convert_reused = False
    if os.path.exists(full_path) and not args.force and not reuse_allowed:
        print("MISMATCHED REUSE REFUSED for %s" % full_path, file=sys.stderr)
        stages.append({"stage": "convert", "rc": 5, "wall_s": 0.0,
                       "note": "refused to reuse an artifact from a different source"})
        write_receipt(receipt_path, receipt)
        return 5
    if os.path.exists(full_path) and not args.force:
        print("reusing existing %s (pass --force to rebuild)" % full_path)
        stages.append({"stage": "convert", "rc": 0, "wall_s": 0.0, "note": "reused existing file"})
        convert_reused = True
    else:
        started = time.time()
        rc, out, err = run(convert_cmd, timeout=args.convert_timeout, env=env)
        wall = round(time.time() - started, 2)
        stages.append({
            "stage": "convert",
            "cmd": convert_cmd,
            "rc": rc,
            "wall_s": wall,
            "stdout_tail": out.strip().splitlines()[-12:],
            "stderr_tail": err.strip().splitlines()[-12:],
        })
        print("convert rc=%s wall=%ss -> %s" % (rc, wall, full_path))
        if rc != 0:
            write_receipt(os.path.join(out_dir, "export-receipt-%s.json" % tag), receipt)
            print(err[-4000:], file=sys.stderr)
            return 4

    full_digest = sha256_file(full_path)
    if convert_reused:
        receipt["artifacts"][args.outtype] = carry_provenance(
            prior, args.outtype, full_path, full_digest,
            source_matches=source_match is not False)
    else:
        record = {"path": full_path, "bytes": os.path.getsize(full_path), "sha256": full_digest}
        record.update(artifact_stamp)
        receipt["artifacts"][args.outtype] = record

    # ---------------------------------------------------------------- quantize
    for quant in quants:
        target = os.path.join(out_dir, gguf_name(tag, quant))
        quant_reused = False
        if os.path.exists(target) and not args.force and not reuse_allowed:
            print("MISMATCHED REUSE REFUSED for %s" % target, file=sys.stderr)
            stages.append({"stage": "quantize:%s" % quant, "rc": 5, "wall_s": 0.0,
                           "note": "refused to reuse an artifact from a different source"})
            continue
        if os.path.exists(target) and not args.force:
            print("reusing existing %s" % target)
            # A reused quant used to record an artifact and NO stage row, so the receipt
            # could not tell "quantized by this run" from "found on disk" -- the shipped
            # receipt had two artifacts and zero quantize stages. The row is written here.
            stages.append({"stage": "quantize:%s" % quant, "rc": 0, "wall_s": 0.0,
                           "note": "reused existing file"})
            quant_reused = True
        else:
            cmd = [quantizer, full_path, target, quant, str(args.threads)]
            started = time.time()
            rc, out, err = run(cmd, timeout=args.quantize_timeout, env=env)
            wall = round(time.time() - started, 2)
            stages.append({
                "stage": "quantize:%s" % quant,
                "cmd": cmd,
                "rc": rc,
                "wall_s": wall,
                "stdout_tail": out.strip().splitlines()[-8:],
                "stderr_tail": err.strip().splitlines()[-8:],
            })
            print("quantize %s rc=%s wall=%ss -> %s" % (quant, rc, wall, target))
            if rc != 0:
                continue
        target_digest = sha256_file(target)
        if quant_reused:
            receipt["artifacts"][quant] = carry_provenance(
                prior, quant, target, target_digest,
                source_matches=source_match is not False)
        else:
            record = {"path": target, "bytes": os.path.getsize(target), "sha256": target_digest}
            record.update(artifact_stamp)
            receipt["artifacts"][quant] = record

    receipt["artifacts_all_produced_by_this_run"] = bool(receipt["artifacts"]) and all(
        info.get("produced_by") == "this_run" for info in receipt["artifacts"].values())
    receipt["artifact_provenance_note"] = (
        "build_receipt_sha256 / llama_cpp_commit at the top level describe the build THIS "
        "RUN used. Per-artifact produced_by_* fields describe the build that actually "
        "wrote each file; on reuse they are carried forward from the prior receipt, and "
        "are null when that join could not be made."
    )
    digest = write_receipt(receipt_path, receipt)
    print("export receipt: %s sha256=%s" % (receipt_path, digest))
    for kind, info in sorted(receipt["artifacts"].items()):
        print("  %-8s %10.2f GB  %s  [%s]" % (
            kind, info["bytes"] / 1e9, info["sha256"][:16], info.get("produced_by")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
