"""Source-pinned assistant-only adapter around the authentic streamed VL writer.

Preparation and inspection are CPU-only. `build` is an explicit teacher-cache
entry point for the allocated runtime; it never launches or allocates a GPU.
Image/video expansion and M-RoPE inputs stay owned by the original processor.
Audio is refused because this teacher has no authentic audio-input route.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chimera_recovery_vl_common", HERE / "training.py")
R = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
spec.loader.exec_module(R)
Error = R.ReadinessError
SCHEMA = "chimera_recovery.vl_jobs.v1"
INSPECTION = "chimera_recovery.vl_processor_inspection.v1"
CONTRACT = "chimera_recovery.vl_assistant_prefix_anchor.v1"
MEDIA_KEYS = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")


def load_writer(path, expected_sha):
    if R.sha(path) != R.require_sha(expected_sha, "writer_sha256"):
        raise Error("original streamed VL writer bytes differ")
    spec = importlib.util.spec_from_file_location("chimera_recovery_original_vl_writer", path)
    writer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = writer
    spec.loader.exec_module(writer)
    if writer.TOOL_VERSION != "1.0.5":
        raise Error("VL prefix adapter was audited against writer1.0.5")
    return writer


def anchor_assistant(processor, row, full, prefix):
    """Require identical expanded native prefix; own assistant body and EOS only."""
    import torch
    ids, before = full["input_ids"], prefix["input_ids"]
    if (ids.ndim != 2 or ids.shape[0] != 1 or before.ndim != 2 or before.shape[0] != 1 or
            ids.dtype != torch.int64 or before.dtype != torch.int64 or not 0 < before.shape[1] < ids.shape[1] or
            not torch.equal(ids[:, :before.shape[1]], before)):
        raise Error(f"VL row {row.get('row_id')}: native expanded prompt IDs do not anchor the full assistant turn")
    types, prompt_types = full["mm_token_type_ids"], prefix["mm_token_type_ids"]
    if (types.shape != ids.shape or prompt_types.shape != before.shape or
            not torch.equal(types[:, :before.shape[1]], prompt_types) or bool(types[:, before.shape[1]:].any())):
        raise Error("VL native media placeholders/types changed or enter assistant supervision")
    for key in MEDIA_KEYS:
        if (key in full) != (key in prefix) or (key in full and not torch.equal(full[key], prefix[key])):
            raise Error("VL assistant rendering changed actual image/video expansion tensors")
    eos = processor.tokenizer.eos_token_id
    eos_ids = [eos] if type(eos) is int else eos
    if not isinstance(eos_ids, (list, tuple)) or not eos_ids or any(type(token) is not int or not 0 <= token < R.TEXT_VOCAB for token in eos_ids):
        raise Error("VL native tokenizer lacks an in-vocabulary completion EOS")
    start = before.shape[1]
    ends = torch.isin(ids[0, start:], torch.tensor(eos_ids, dtype=ids.dtype)).nonzero().flatten()
    if ends.numel() != 1:
        raise Error("VL terminal assistant turn needs exactly one authentic completion EOS")
    stop = start + int(ends[0]) + 1
    if stop <= start + 1 or processor.tokenizer.decode(ids[0, stop:].tolist(), skip_special_tokens=False).strip():
        raise Error("VL EOS has no assistant body or leaves non-whitespace trailing content")
    targets = torch.zeros_like(ids, dtype=torch.bool)
    targets[:, start:stop] = True
    if bool(((ids < 0) | (ids >= R.TEXT_VOCAB)).any()):
        raise Error("VL native processor IDs leave parent text vocabulary")
    full["assistant_target_mask"] = targets
    return full


@contextmanager
def assistant_mask_adapter(writer):
    """Bounded wrapper; restore both authentic functions on every exit."""
    encode, collate = writer.encode_vl_row, writer.collate_vl
    def encode_masked(processor, row, base_dir, template_kwargs=None, generation_prompt=False):
        if dict(template_kwargs or {}) != {"enable_thinking": False}:
            raise Error("VL recovery requires pinned enable_thinking=False on both renders")
        messages = row["messages"]
        if any(item.get("type") not in {"image", "video", "text"} or
               message.get("role") == "assistant" and item.get("type") != "text"
               for message in messages for item in message.get("content", []) if isinstance(message.get("content"), list)):
            raise Error("VL recovery refuses unsupported audio inputs or assistant media targets")
        full = encode(processor, row, base_dir, template_kwargs, generation_prompt)
        if generation_prompt:
            return full
        if (not messages or messages[-1].get("role") != "assistant" or
                sum(message.get("role") == "assistant" for message in messages) != 1 or
                any(message.get("role") not in {"system", "user", "assistant"} for message in messages)):
            raise Error("VL fixture adapter requires one terminal assistant target and authentic prompt history")
        previous = copy.deepcopy(row)
        previous["messages"] = previous["messages"][:-1]
        prefix = encode(processor, previous, base_dir, template_kwargs, True)
        return anchor_assistant(processor, row, full, prefix)
    def collate_masked(encoded_rows, rows, pad_id):
        import torch
        batch = collate(encoded_rows, rows, pad_id)
        mask = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
        for index, encoded in enumerate(encoded_rows):
            target = encoded.get("assistant_target_mask")
            length = encoded["input_ids"].shape[1]
            if target is None or target.shape != (1, length) or target.dtype != torch.bool or bool(target[:, 0].any()):
                raise Error("VL collate lacks authentic target-indexed assistant token ownership")
            mask[index, :length - 1] = target[0, 1:]
            if bool((mask[index, :length] & batch["mm_token_type_ids"][index, :length].ne(0)).any()):
                raise Error("VL stored prediction positions include visual placeholders")
        batch["store_mask"] = mask
        return batch
    writer.encode_vl_row, writer.collate_vl = encode_masked, collate_masked
    try:
        yield writer
    finally:
        writer.encode_vl_row, writer.collate_vl = encode, collate


def prepare_fixtures(source_path, source_report, asset_root, out, runtime_root, *, writer_path, writer_sha):
    """Copy real supported assets and retain source, held split and group identities."""
    if out.exists():
        raise Error("VL preparation needs a new output directory")
    report = json.loads(source_report.read_text())
    artifact = report["artifacts"][source_path.name]
    if R.sha(source_path) != artifact["sha256"] or report.get("cross_split_media_conflicts") != 0:
        raise Error("VL authored source receipt or media split identity differs")
    records = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    if len(records) != artifact["rows"]:
        raise Error("VL authored source row count differs")
    R.require_sha(writer_sha, "writer_sha256")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".vl-fixtures-", dir=out.parent))
    by_split = {key: [] for key in ("train", "validation", "test")}
    assets, blocked, groups, seen = {}, [], {}, set()
    for record in records:
        route, split = record["route"], record["split"]
        ident = record["id"]
        group = record["provenance"]["group_id"]
        if (ident in seen or split not in by_split or record["provenance"].get("split") != split or
                group in groups and groups[group] != split or record.get("schema") != "recovery_corpus.v1" or
                record.get("quality", {}).get("source_ready") is not True or
                record.get("quality", {}).get("train_ready") is not True or
                record.get("quality", {}).get("groundtruth_verified") is not True or
                record.get("supervision", {}).get("policy") != "assistant_only" or
                record["supervision"].get("message_mask") != [False, True]):
            raise Error("VL source IDs/connected groups/split/source admission differs")
        seen.add(ident)
        groups[group] = split
        if route == "audio_understand":
            blocked.append({"id": ident, "route": route, "split": split, "recovery_group_id": group,
                            "blocker": "Qwen3.8 VL teacher has no authentic audio input; no waveform-to-text substitute"})
            continue
        if route not in {"vision_grounding", "video_understand"} or len(record["media"]) != 1:
            raise Error("unsupported understanding fixture route/media cardinality")
        media = record["media"][0]
        kind = "image" if route == "vision_grounding" else "video"
        case = record["provenance"]["case_id"]
        basename = str(media["path"]).replace("\\", "/").split("/")[-1]
        if (media.get("type") != kind or media.get("role") != "input" or not basename or
                basename in {".", ".."} or case in {"", ".", ".."} or "/" in case or "\\" in case):
            raise Error("VL source asset type/role/path differs")
        asset = asset_root / case / basename
        if R.sha(asset) != R.require_sha(media["sha256"], "asset_sha256") or asset.stat().st_size != media["bytes"]:
            raise Error("VL original fixture asset bytes changed")
        truth_name = str(record["provenance"]["groundtruth_path"]).replace("\\", "/").split("/")[-1]
        if truth_name != "groundtruth.json" or R.sha(asset_root / case / truth_name) != R.require_sha(record["provenance"]["groundtruth_sha256"], "groundtruth_sha256"):
            raise Error("VL authored groundtruth bytes changed")
        relative = f"assets/{case}/{basename}"
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(asset, target)
        assets[relative] = {"sha256": media["sha256"], "bytes": media["bytes"], "kind": kind}
        messages = record["messages"]
        if (len(messages) != 2 or [message.get("role") for message in messages] != ["user", "assistant"] or
                any(not isinstance(message.get("content"), str) or not message["content"] for message in messages)):
            raise Error("VL fixture requires actual authored user instruction and assistant groundtruth")
        entry = {"type": kind, "path": relative, "sha256": media["sha256"]}
        if kind == "video":
            if media.get("fps") != 8 or media.get("duration_seconds") != 1.:
                raise Error("VL video fixture must preserve its actual full eight-frame interval")
            entry["num_frames"] = 8
        by_split[split].append({"row_id": ident, "route": route, "split": split, "recovery_group_id": group,
                               "messages": [{"role": "user", "content": [entry, {"type": "text", "text": messages[0]["content"]}]},
                                            {"role": "assistant", "content": [{"type": "text", "text": messages[1]["content"]}]}],
                               "source_row_sha256": R.digest(record), "groundtruth_sha256": record["provenance"]["groundtruth_sha256"],
                               "teacher_forced_authored_reference": True, "adapter_contract": CONTRACT,
                               "adapter_sha256": R.sha(Path(__file__))})
    sources = []
    for split, rows in by_split.items():
        filename = f"{split}.jsonl"
        (stage / filename).write_bytes(b"".join(R.canonical(row) + b"\n" for row in rows))
        sources.append({"split": split, "rows": len(rows), "path": filename, "sha256": R.sha(stage / filename),
                        "routes": {route: sum(row["route"] == route for row in rows) for route in ("vision_grounding", "video_understand")}})
    plan = {"schema": SCHEMA, "production_training_ready": False, "teacher_cache_started": False,
            "source_sha256": R.sha(source_path), "source_report_sha256": R.sha(source_report), "source_rows": len(records),
            "adapter_contract": CONTRACT, "adapter_sha256": R.sha(Path(__file__)), "writer_path": str(writer_path), "writer_sha256": writer_sha,
            "runtime_root": runtime_root, "sources": sources, "assets": assets, "blocked": blocked,
            "accepted_rows": sum(len(rows) for rows in by_split.values()), "connected_groups_disjoint": True,
            "teacher": "q38-vl-recovery-assistant-" + R.sha(Path(__file__))[:12],
            "processor_options": {"enable_thinking": False, "max_pixels": 262144},
            "scope": "teacher-forced authored groundtruth; no free-generation or native student capability claim",
            "launch_constraints": ["exclusive GPU/runtime allocation", "previous Qwen streamer stopped and host RAM policy satisfied",
                                   "actual processor inspection must pass and be pinned", "GPU-native parity and capacity must pass"],
            "native_student_remaining": ["source-aligned QVL feature targets for images and paired video frames",
                                         "D3d native answer-cache conversion and full-response evaluation",
                                         "audio semantic teacher and native audio-to-answer input contract"]}
    (stage / "manifest.json").write_bytes(R.canonical(plan) + b"\n")
    stage.rename(out)
    return plan


def load_plan(path, expected_sha):
    if R.sha(path) != R.require_sha(expected_sha, "plan_sha256"):
        raise Error("VL job plan bytes changed")
    plan = json.loads(path.read_text())
    if plan.get("schema") != SCHEMA or plan.get("adapter_contract") != CONTRACT or plan.get("adapter_sha256") != R.sha(Path(__file__)):
        raise Error("VL job plan adapter/source contract differs")
    def contained(name):
        target = (path.parent / name).resolve()
        if Path(name).is_absolute() or not target.is_relative_to(path.parent.resolve()):
            raise Error("VL source asset/row path leaves private preparation")
        return target
    for source in plan["sources"]:
        if R.sha(contained(source["path"])) != source["sha256"]:
            raise Error("VL normalized source rows changed")
    for name, asset in plan["assets"].items():
        target = contained(name)
        if R.sha(target) != asset["sha256"] or target.stat().st_size != asset["bytes"]:
            raise Error("VL visible source asset changed")
    return plan


def inspect_processor(writer, plan_path, plan_sha, model_dir, processor_dir, out):
    import torch
    if torch.cuda.is_initialized() or out.exists():
        raise Error("VL processor inspection is CPU-only and needs a new report")
    plan = load_plan(plan_path, plan_sha)
    processor = writer.load_processor(processor_dir, plan["processor_options"]["max_pixels"])
    identity = writer.processor_identity(processor, processor_dir, {"enable_thinking": False})
    rows = []
    with assistant_mask_adapter(writer):
        for source in plan["sources"]:
            corpus = writer.load_vl_corpus(plan_path.parent / source["path"])
            for index, row in enumerate(corpus["rows_data"]):
                encoded = writer.encode_vl_row(processor, row, plan_path.parent, {"enable_thinking": False})
                batch = writer.collate_vl([encoded], [index], processor.tokenizer.pad_token_id or 0)
                if encoded["input_ids"].shape[1] > 4096:
                    raise Error("VL fixture expanded context exceeds declared4096 budget; no truncation")
                rows.append({"row_id": row["row_id"], "split": source["split"], "length": encoded["input_ids"].shape[1],
                             "assistant_targets": int(encoded["assistant_target_mask"].sum()), "stored_positions": int(batch["store_mask"].sum()),
                             "input_ids_sha256": R.tensor_sha256(encoded["input_ids"]), "store_mask_sha256": R.tensor_sha256(batch["store_mask"]),
                             "media_tensors_sha256": {key: R.tensor_sha256(encoded[key]) for key in MEDIA_KEYS if key in encoded}})
    receipt = {"schema": INSPECTION, "passed": True, "plan_sha256": plan_sha, "adapter_sha256": R.sha(Path(__file__)),
               "writer_sha256": plan["writer_sha256"], "processor": identity, "tokenizer_class": type(processor.tokenizer).__name__,
               "chat_template_sha256": R.digest(processor.chat_template), "model_config_sha256": R.sha(model_dir / "config.json"),
               "rows": rows, "model_weights_loaded": False, "training_started": False}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(R.canonical(receipt) + b"\n")
    return receipt


def build_teacher(writer, plan_path, plan_sha, inspection_path, inspection_sha, split, *, model_dir, processor_dir,
                  out_root, device, allocator_receipt, receipt_path, max_new_shards=0):
    plan = load_plan(plan_path, plan_sha)
    if not allocator_receipt.is_file() or max_new_shards < 0 or device != "cuda:0":
        raise Error("VL teacher build requires a real allocator receipt, remapped sole cuda:0 and nonnegative bound")
    if receipt_path.exists() or R.sha(inspection_path) != R.require_sha(inspection_sha, "inspection_sha256"):
        raise Error("VL build needs a new receipt and pinned CPU processor inspection")
    inspection = json.loads(inspection_path.read_text())
    if (inspection.get("schema") != INSPECTION or inspection.get("passed") is not True or
            inspection.get("plan_sha256") != plan_sha or inspection.get("writer_sha256") != plan["writer_sha256"] or
            inspection.get("adapter_sha256") != R.sha(Path(__file__)) or
            inspection["processor"]["files_sha256"] != writer.processor_files(processor_dir) or
            inspection.get("model_config_sha256") != R.sha(model_dir / "config.json")):
        raise Error("VL inspected processor/tokenizer/config/source pins changed")
    source = next(row for row in plan["sources"] if row["split"] == split)
    argv = ["build-vl", "--model-dir", str(model_dir), "--processor-dir", str(processor_dir), "--teacher", plan["teacher"],
            "--out-root", str(out_root), "--corpus", str(plan_path.parent / source["path"]), "--device", device,
            "--dtype", "bfloat16", "--top-k", "64", "--vocab-rows", "248320", "--micro-batch", "2",
            "--max-batch-tokens", "8192", "--head-chunk-rows", "256", "--rows-per-shard", "16", "--batch-order", "length",
            "--head-precision", "fp32", "--teacher-digest", "full", "--digest-threads", "4", "--host-resident", "pageable",
            "--max-pixels", "262144", "--chat-template-kwargs", '{"enable_thinking":false}', "--verify-existing",
            "--allocator-receipt", str(allocator_receipt), "--purpose", "source-pinned assistant-only VL recovery cache",
            "--max-new-shards", str(max_new_shards)]
    args = writer.parser().parse_args(argv)
    corpus = writer.load_vl_corpus(Path(args.corpus[0]))
    failure, code = None, 2
    try:
        with assistant_mask_adapter(writer):
            code = writer.build_cache(args, "vl")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    cache = out_root / plan["teacher"] / corpus["sha256"]
    sealed = (cache / "manifest.json").is_file()
    verified = sealed and writer.verify_cache(cache) == 0
    receipt = {"schema": "chimera_recovery.vl_teacher_run.v1", "plan_sha256": plan_sha, "inspection_sha256": inspection_sha,
               "adapter_sha256": R.sha(Path(__file__)), "writer_sha256": plan["writer_sha256"], "split": split,
               "argv": argv, "cache": str(cache), "writer_exitcode": code, "cache_sealed_and_verified": verified,
               "cache_manifest_sha256": R.sha(cache / "manifest.json") if sealed else None,
               "assistant_only": True, "teacher_forced_authored_reference": True, "native_student_training_started": False}
    receipt["failure"] = failure
    receipt["allocator_receipt_sha256"] = R.sha(allocator_receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(R.canonical(receipt) + b"\n")
    if failure:
        raise Error("VL teacher writer failed; private run receipt records the exact refusal: " + failure["message"])
    return code if code else (0 if verified or max_new_shards else 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    for name in ("source", "source-report", "asset-root", "out", "writer"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--writer-sha256", required=True)
    prepare.add_argument("--runtime-root", required=True)
    for name in ("inspect", "build"):
        p = sub.add_parser(name)
        for field in ("plan", "writer", "model-dir", "processor-dir"):
            p.add_argument("--" + field, type=Path, required=True)
        for field in ("plan-sha256", "writer-sha256"):
            p.add_argument("--" + field, required=True)
        if name == "inspect":
            p.add_argument("--out", type=Path, required=True)
        else:
            for field in ("inspection", "out-root", "allocator-receipt", "receipt"):
                p.add_argument("--" + field, type=Path, required=True)
            p.add_argument("--inspection-sha256", required=True)
            p.add_argument("--split", choices=("train", "validation", "test"), required=True)
            p.add_argument("--device", required=True)
            p.add_argument("--max-new-shards", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_fixtures(args.source, args.source_report, args.asset_root, args.out, args.runtime_root,
                                  writer_path=args.writer, writer_sha=args.writer_sha256)
        print(json.dumps({"accepted": result["accepted_rows"], "blocked": len(result["blocked"]), "production_training_ready": False}))
        return 0
    writer = load_writer(args.writer, args.writer_sha256)
    plan = load_plan(args.plan, args.plan_sha256)
    if plan["writer_sha256"] != args.writer_sha256:
        raise Error("VL job plan and loaded writer pin disagree")
    if args.command == "inspect":
        result = inspect_processor(writer, args.plan, args.plan_sha256, args.model_dir, args.processor_dir, args.out)
        print(json.dumps({"passed": result["passed"], "rows": len(result["rows"])}))
        return 0
    return build_teacher(writer, args.plan, args.plan_sha256, args.inspection, args.inspection_sha256, args.split,
                         model_dir=args.model_dir, processor_dir=args.processor_dir, out_root=args.out_root, device=args.device,
                         allocator_receipt=args.allocator_receipt, receipt_path=args.receipt, max_new_shards=args.max_new_shards)


if __name__ == "__main__":
    raise SystemExit(main())
