"""Prepare a new source-isolated D1 dataset from genuine parent-tokenized rows.

CPU preparation only. Complete conversations are selected separately from train
and validation; test/frozen sources never enter the optimizer. Original corpus,
preregistration and models stay unchanged. D1 uses all real hidden-state positions,
not assistant-only answer loss. Padding is recorded and never counted as tokens.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

try:
    from . import d1_source_audit as A
except ImportError:
    import d1_source_audit as A

FROZEN_SHA = "c24d21e08815d7acd2ce14aea989d16f84f167eb24a40597966be19284cb1370"
SCHEMA = "chimera_recovery.d1_source_successor.v1"


def digest(value):
    return hashlib.sha256(A.canonical(value)).hexdigest()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=A.sha(path))


def group(row):
    return row.get("provenance", {}).get("recovery_group_id")


def packed_source(manifest_path, records_path, split, excluded_ids, excluded_groups):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = list(A.read_rows(records_path))
    if (manifest.get("schema") != "chimera_recovery.assistant_supervision.v1" or manifest.get("split") != split
            or manifest.get("cross_split_groups") or manifest.get("input_records_sha256") != digest(records)):
        raise ValueError("actual native packed source/input record/split contract differs")
    by_id = {row["id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate source IDs in native input records")
    receipts, accepted, candidates, removed = manifest["row_receipts"], [], [], []
    if len(receipts) != manifest["rows"]:
        raise ValueError("native source row receipt count differs")
    for index, receipt in enumerate(receipts):
        ident = receipt["row_id"]
        row = by_id.get(ident)
        if (row is None or receipt["source_record_sha256"] != digest(row) or row.get("split") != split
                or row.get("provenance", {}).get("split", split) != split or not group(row)
                or row.get("quality", {}).get("train_ready") is not True or row.get("provenance", {}).get("source_verified") is not True
                or receipt.get("template_sha256") != manifest["template_sha256"]):
            raise ValueError("native receipt must bind a complete admitted source row/group/template")
        accepted.append(row)
        item = dict(id=ident, group=group(row), source_index=index, source_record_sha256=digest(row), split=split,
                    route=row["route"],
                    rendered_sha256=receipt["rendered_sha256"], native_messages_sha256=receipt["native_messages_sha256"])
        (removed if ident in excluded_ids or group(row) in excluded_groups else candidates).append(item)
    if digest(accepted) != manifest["records_sha256"] or len({r["id"] for r in candidates}) != len(candidates):
        raise ValueError("native admitted source order/census differs")
    shard = Path(manifest["shard"])
    if not shard.is_file() or A.sha(shard) != manifest["sha256"]:
        raise ValueError("actual parent-tokenized source shard missing or changed")
    return manifest, candidates, removed, shard


def select_complete(rows, lengths, minimum, *, unavailable_groups=()):
    if type(minimum) is not int or minimum < 1:
        raise ValueError("positive real-token minimum required")
    selected, total = [], 0
    for row in rows:
        if row["group"] in unavailable_groups:
            raise ValueError("fit/diagnostic connected source group overlaps")
        length = lengths[row["source_index"]]
        if type(length) is not int or length < 1:
            raise ValueError("positive original real length required")
        selected.append({**row, "length": length, "start": total, "end": total + length})
        total += length
        if total >= minimum:
            return selected, total
    raise ValueError("insufficient real source tokens; never repeat rows or count padding")


def amended_preregistration(raw, audit, dataset_pin, counts, source_pins):
    normalized = raw.replace(b"\r\n", b"\n")
    start, end = normalized.find(b"## 1. Scope"), normalized.find(b"\n## Amendments")
    if start < 0 or end <= start or hashlib.sha256(normalized[start:end + 1]).hexdigest() != FROZEN_SHA:
        raise ValueError("preserved original frozen preregistration body differs")
    extra = ("\n\n### 2026-10-13: D1 source-isolated successor after measured source-span audit\n\n"
             "This dated correction supersedes the earlier amendment's instruction to keep the sealed D1 "
             "input as the active training input. The original sealed prefix and original frozen body are "
             "preserved as evidence. The measured original fit prefix includes 1,498,741 tokens across "
             "3,010 documents incompatible with current recovery splits/quality checks, including "
             "371,547 test and 351,492 validation tokens; one document crosses its fit/diagnostic boundary. "
             "There were zero original frozen-carve assignment violations. New-split overlap is not "
             "retroactive proof the original frozen evaluation leaked; structural answer quarantine is "
             "also distinct from hidden-state regression target correctness.\n\n"
             "Use only verified complete parent-tokenized admitted train conversations for D1 optimizer "
             "positions and admitted validation conversations for diagnostics. Exclude current global "
             "cross-route/frozen-source groups from both pools, and prove fit/diagnostic source groups "
             "disjoint. No test row enters either pool. Select whole conversations in existing source "
             "order until at least 3,751,936 fit and 249,856 diagnostic real tokens; pack the two streams "
             "separately at width4,096 with explicit real lengths and an explicit --held-rows count. "
             "D1 retains all-real-position prefix-fed hidden-state regression and the original optimizer, "
             "layer objectives and immutable base. Whole-document overshoot and final-row padding are "
             "reported below; padding is never a training target or claimed real budget. This replaces "
             "the exact977-row budget where whole conversations require additional rows. Pin this new "
             "dataset/config and amendment before the first update; the original-prefix extractor must "
             "not overwrite or substitute this input. Preparation does not launch or establish recovery.\n\n")
    proof = dict(source_audit=audit, dataset=dataset_pin, actual_counts=counts, sources=source_pins)
    return normalized + extra.encode() + b"```json\n" + json.dumps(proof, sort_keys=True, indent=2).encode() + b"\n```\n"


def prepare(train_manifest, train_records, validation_manifest, validation_records, admission, exclusions, source_audit, preregistration, out):
    import torch
    from safetensors.torch import load_file, save_file
    if out.exists():
        raise ValueError("preserve prior dataset; new output directory required")
    global_admission = json.loads(admission.read_text(encoding="utf-8"))
    if (global_admission.get("connected_groups_verified") is not True or global_admission.get("eval_disjoint") is not True
            or global_admission["outputs"]["cross_route_training_exclusions"]["sha256"] != A.sha(exclusions)):
        raise ValueError("actual global connected-group/frozen exclusion receipt differs")
    audit = json.loads(source_audit.read_text(encoding="utf-8"))
    if (audit.get("schema") != "chimera_recovery.d1_source_span_audit.v1" or audit.get("current_incompatible_fit_tokens") != 1498741
            or audit.get("current_incompatible_fit_documents") != 3010 or audit.get("original_frozen_carve_violation_documents") != 0):
        raise ValueError("dated amendment must bind the actual measured D1 source audit")
    if digest({key: value for key, value in audit.items() if key != "seal_sha256"}) != audit.get("seal_sha256"):
        raise ValueError("measured source-audit seal changed")
    excluded = list(A.read_rows(exclusions))
    excluded_ids = {row["id"] for row in excluded}
    excluded_groups = {row[key] for row in excluded for key in ("source_group_id", "connected_group_id") if row.get(key)}
    lanes = []
    source_pins = {key: pin(path) for key, path in dict(train_manifest=train_manifest, train_records=train_records, validation_manifest=validation_manifest,
                    validation_records=validation_records, global_admission=admission, cross_route_exclusions=exclusions, original_preregistration=preregistration).items()}
    for manifest_path, records_path, split in ((train_manifest, train_records, "train"), (validation_manifest, validation_records, "validation")):
        manifest, candidates, removed, shard = packed_source(manifest_path, records_path, split, excluded_ids, excluded_groups)
        data = load_file(str(shard), device="cpu")
        ids, lengths = data["input_ids"], data["lengths"]
        if (ids.dtype != torch.int64 or lengths.dtype != torch.int64 or list(ids.shape) != manifest["shape"]
                or list(lengths.shape) != [manifest["rows"]] or bool(((lengths < 1) | (lengths > ids.shape[1])).any())
                or int(lengths.sum()) != manifest["tokens"] or manifest["shape"][1] > 4096):
            raise ValueError("native tensor shapes/real lengths/token count differ")
        valid = torch.arange(ids.shape[1])[None, :] < lengths[:, None]
        if bool(((ids[valid] < 0) | (ids[valid] >= 248320)).any()):
            raise ValueError("D1 source IDs must stay in native text vocabulary")
        source_pins[split + "_shard"] = pin(shard)
        lanes.append(dict(candidates=candidates, removed=removed, ids=ids, lengths=lengths.tolist(), template=manifest["template_sha256"]))
    if lanes[0]["template"] != lanes[1]["template"]:
        raise ValueError("train/diagnostic parent templates differ")
    train_groups = {row["group"] for row in lanes[0]["candidates"]}
    held_groups = {row["group"] for row in lanes[1]["candidates"]}
    if train_groups & held_groups:
        raise ValueError("fit/diagnostic connected source group overlaps")
    fit, fit_tokens = select_complete(lanes[0]["candidates"], lanes[0]["lengths"], 3751936)
    held, held_tokens = select_complete(lanes[1]["candidates"], lanes[1]["lengths"], 249856, unavailable_groups=train_groups)
    fit_rows, held_rows = math.ceil(fit_tokens / 4096), math.ceil(held_tokens / 4096)
    ids = torch.zeros(fit_rows + held_rows, 4096, dtype=torch.int64)
    lengths = torch.zeros(fit_rows + held_rows, dtype=torch.int64)
    for lane, selected, n_tokens, offset, n_rows in ((lanes[0], fit, fit_tokens, 0, fit_rows), (lanes[1], held, held_tokens, fit_rows, held_rows)):
        stream = torch.cat([lane["ids"][row["source_index"], :row["length"]] for row in selected])
        for index in range(n_rows):
            part = stream[index * 4096:(index + 1) * 4096]
            ids[offset + index, :part.numel()] = part
            lengths[offset + index] = part.numel()
    counts = dict(fit_documents=len(fit), diagnostic_documents=len(held), fit_rows=fit_rows, held_rows=held_rows, total_rows=fit_rows+held_rows,
                  fit_real_tokens=fit_tokens, diagnostic_real_tokens=held_tokens, total_real_tokens=fit_tokens+held_tokens,
                  padded_positions=ids.numel()-fit_tokens-held_tokens, width=4096, original_real_budget=4001792,
                  whole_document_real_overshoot=fit_tokens+held_tokens-4001792)
    counts["fit_real_tokens_by_route"] = {route: sum(row["length"] for row in fit if row["route"] == route) for route in sorted({row["route"] for row in fit})}
    counts["diagnostic_real_tokens_by_route"] = {route: sum(row["length"] for row in held if row["route"] == route) for route in sorted({row["route"] for row in held})}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=out.name + ".", dir=out.parent) as temporary:
        root = Path(temporary)
        save_file(dict(input_ids=ids, lengths=lengths, fit_mask=torch.arange(ids.shape[0]) < fit_rows, held_mask=torch.arange(ids.shape[0]) >= fit_rows),
                  str(root / "d1_source_ids.safetensors"), metadata=dict(schema=SCHEMA, padding="zero id; lengths determine all real positions"))
        dataset_pin = dict(path=str((out / "d1_source_ids.safetensors").resolve()), sha256=A.sha(root / "d1_source_ids.safetensors"))
        raw = amended_preregistration(preregistration.read_bytes(), pin(source_audit), dataset_pin, counts, source_pins)
        (root / "PREREGISTRATION_D1_SOURCE_SUCCESSOR_2026-10-13.md").write_bytes(raw)
        with (root / "documents.jsonl").open("wb") as stream:
            for row in fit + held:
                stream.write(A.canonical(row) + b"\n")
        config = dict(schema=SCHEMA, actual_counts=counts, sources=source_pins, source_audit=pin(source_audit), dataset=dataset_pin,
                      documents=dict(path=str((out / "documents.jsonl").resolve()), sha256=A.sha(root / "documents.jsonl")),
                      preregistration=dict(path=str((out / "PREREGISTRATION_D1_SOURCE_SUCCESSOR_2026-10-13.md").resolve()), sha256=A.sha(root / "PREREGISTRATION_D1_SOURCE_SUCCESSOR_2026-10-13.md")),
                      fit_groups=sorted({row["group"] for row in fit}), diagnostic_groups=sorted({row["group"] for row in held}),
                      source_exclusions={"train": len(lanes[0]["removed"]), "validation": len(lanes[1]["removed"])},
                      template_sha256=lanes[0]["template"], builder_sha256=A.sha(Path(__file__)), production_training_ready=False,
                      training_launched=False, teacher_runtime_verified=False, original_artifacts_changed=False,
                      trainer_required_args=["--ids", dataset_pin["path"], "--seq-len", "4096", "--lengths-key", "lengths", "--held-rows", str(held_rows),
                                             "--preregistration", str((out / "PREREGISTRATION_D1_SOURCE_SUCCESSOR_2026-10-13.md").resolve())])
        config["seal_sha256"] = digest(config)
        (root / "manifest.json").write_bytes(A.canonical(config) + b"\n")
        # Recheck every source seal before atomic publication.
        if any(A.sha(Path(value["path"])) != value["sha256"] for value in source_pins.values()):
            raise ValueError("source changed during CPU preparation")
        root.rename(out)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-manifest", "train-records", "validation-manifest", "validation-records", "admission", "exclusions", "source-audit", "preregistration", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = prepare(**vars(args))
    print(json.dumps({key: result[key] for key in ("actual_counts", "source_exclusions", "production_training_ready", "seal_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
