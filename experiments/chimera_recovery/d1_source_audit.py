"""Read-only source-span audit of the preserved original D1 prefix.

Reports compatibility with current recovery holdouts and exclusions. A newer
split intersection is not retroactive proof the original frozen eval leaked.
No tokenizer, model, cache builder or training command is executed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def audit(documents, text_manifest, normalized, carve, exclusions, out, *, total_rows=977, fit_rows=916, width=4096):
    if out.exists() or not 0 < fit_rows < total_rows or width < 1:
        raise ValueError("new audit directory and valid original prefix required")
    manifest = json.loads(text_manifest.read_text(encoding="utf-8"))
    if sha(documents) != manifest["documents_file"]["sha256"] or manifest["sequence_length"] != width:
        raise ValueError("original source-span file/row width differs from sealed text manifest")
    assignment = json.loads(carve.read_text(encoding="utf-8"))["assignment"]
    source = {}
    for row in read_rows(normalized):
        origin = row["provenance"].get("registry_origin", {})
        rid = origin.get("rid")
        if not rid or rid in source:
            raise ValueError("normalized registry RID missing or repeated")
        source[rid] = dict(uid=row["id"], group=row["provenance"]["recovery_group_id"], split=row["split"],
                           ready=row["quality"]["train_ready"], blockers=row["quality"]["blockers"], route=row["route"])
    excluded = {row["rid"]: row for row in read_rows(exclusions)}
    fit_end, end = fit_rows * width, total_rows * width
    entries, fit_groups, held_groups, doc_ids = [], set(), set(), set()
    counts, tokens, routes = Counter(), Counter(), Counter()
    for row in read_rows(documents):
        lo, hi = row["start"], row["end"]
        if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi or hi - lo != row["kept"]:
            raise ValueError("source document span/kept token extent differs")
        if lo >= end:
            continue
        uid = row["uid"]
        rid = row["rid"]
        if (uid, rid) in doc_ids or rid not in source or source[rid]["uid"] != uid:
            raise ValueError("D1 source UID/RID missing, mismatched or repeated")
        doc_ids.add((uid, rid))
        item = source[rid]
        fit = max(0, min(hi, fit_end) - lo)
        held = max(0, min(hi, end) - max(lo, fit_end))
        risks = []
        if item["split"] in {"validation", "test"}:
            risks.append("current_recovery_" + item["split"])
        if not item["ready"]:
            risks.append("current_structural_quarantine")
            risks.extend("quality:" + key for key in item["blockers"])
        if rid in excluded:
            risks.append("global_cross_route_exclusion")
            risks.extend("cross_route:" + key for key in excluded[rid]["blockers"])
        old = assignment.get(rid)
        if old != "train":
            risks.append("original_carve:" + str(old))
        entry = dict(uid=uid, rid=row["rid"], original_document=row["doc"], source_group=item["group"],
                     normalized_route=item["route"], current_split=item["split"], original_carve=old,
                     start=lo, end=hi, fit_tokens=fit, diagnostic_tokens=held, risks=sorted(set(risks)))
        entries.append(entry)
        if fit:
            fit_groups.add(item["group"])
            counts["fit_documents"] += 1
            tokens["fit_tokens"] += fit
            routes[item["route"]] += fit
            for risk in set(risks):
                counts[risk] += 1
                tokens[risk] += fit
        if held:
            held_groups.add(item["group"])
            counts["diagnostic_documents"] += 1
            tokens["diagnostic_tokens"] += held
    if tokens["fit_tokens"] != fit_end or tokens["diagnostic_tokens"] != end - fit_end:
        raise ValueError("source-span ledger does not cover the exact original prefix")
    shared = fit_groups & held_groups
    crossing = [row for row in entries if row["fit_tokens"] and row["diagnostic_tokens"]]
    fit_risk = [row for row in entries if row["fit_tokens"] and row["risks"]]
    out.mkdir(parents=True)
    with (out / "source_spans.jsonl").open("wb") as stream:
        for row in entries:
            stream.write(canonical(row) + b"\n")
    report = dict(schema="chimera_recovery.d1_source_span_audit.v1", date="2026-10-13", original_artifacts_changed=False,
                  sources={str(path): sha(path) for path in (documents, text_manifest, normalized, carve, exclusions)},
                  auditor_sha256=sha(Path(__file__)), original_prefix=dict(rows=total_rows, fit_rows=fit_rows, held_rows=total_rows-fit_rows, width=width),
                  counts=dict(counts), fit_tokens_by_risk=dict(tokens), fit_tokens_by_route=dict(routes),
                  fit_diagnostic_shared_connected_groups=len(shared), boundary_crossing_documents=len(crossing),
                  current_incompatible_fit_documents=len(fit_risk), current_incompatible_fit_tokens=sum(row["fit_tokens"] for row in fit_risk),
                  original_frozen_carve_violation_documents=sum(bool(row["fit_tokens"] and row["original_carve"] != "train") for row in entries),
                  compatible_with_current_holdouts_and_quarantine=not fit_risk and not shared,
                  interpretation=["Risk counts overlap; union counts/tokens count each fit document once.",
                                  "Current recovery split overlap is a prospective holdout compatibility failure, not retroactive proof of original frozen-eval contamination.",
                                  "D1 hidden-state regression differs from assistant-only answer learning, but source-group evaluation isolation still applies.",
                                  "Any corrected D1 successor needs new source/teacher-cache pins and a dated amendment; preserve original sealed prefix and frozen evaluation."],
                  source_spans=dict(path=str((out / "source_spans.jsonl").resolve()), sha256=sha(out / "source_spans.jsonl")))
    report["seal_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    (out / "report.json").write_bytes(canonical(report) + b"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("documents", "text-manifest", "normalized", "carve", "exclusions", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.documents, args.text_manifest, args.normalized, args.carve, args.exclusions, args.out)
    print(json.dumps({key: result[key] for key in ("compatible_with_current_holdouts_and_quarantine", "counts", "fit_diagnostic_shared_connected_groups", "current_incompatible_fit_tokens", "seal_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
