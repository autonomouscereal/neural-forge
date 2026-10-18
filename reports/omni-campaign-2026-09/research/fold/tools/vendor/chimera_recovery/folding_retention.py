"""Score real paired free-running continuations for future folding comparisons.

This is an offline measurement tool, not a training loss or an intelligence
benchmark. Parent agreement and independently established task correctness are
different evidence. It never changes a checkpoint or chooses a replacement fold.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HORIZONS = (5, 8, 16, 32)
SHA = re.compile(r"[a-f0-9]{64}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def tokens(value, name, vocab):
    if not isinstance(value, list) or any(type(t) is not int or not 0 <= t < vocab for t in value):
        raise ValueError(name + " requires actual native token IDs")
    return value


def completion(run, *, eos_ids, max_new_tokens, vocab):
    ids = tokens(run.get("generated_ids"), "continuation", vocab)
    if len(ids) > max_new_tokens or run.get("mode") != "free_running":
        raise ValueError("bounded independently free-running generation required")
    reason = run.get("stop_reason")
    if reason not in {"eos", "max_tokens", "stop_sequence"}:
        raise ValueError("actual generation stop reason required")
    if reason == "max_tokens" and len(ids) != max_new_tokens:
        raise ValueError("cap stop must bind the complete requested generation budget")
    if reason == "eos":
        if not ids or ids[-1] not in eos_ids or any(t in eos_ids for t in ids[:-1]):
            raise ValueError("EOS stop requires one final actual EOS")
        content = ids[:-1]
    else:
        if any(t in eos_ids for t in ids):
            raise ValueError("EOS token contradicts recorded non-EOS stop")
        content = ids
    return ids, content, reason


def longest_run(reference, actual):
    prefix = run = longest = matches = 0
    prefix_open = True
    for expected, got in zip(reference, actual):
        if expected == got:
            matches += 1
            run += 1
            longest = max(longest, run)
            if prefix_open:
                prefix += 1
        else:
            prefix_open = False
            run = 0
    return prefix, longest, matches


def repeated_ngrams(ids, n=4):
    counts = Counter(tuple(ids[i:i+n]) for i in range(max(0, len(ids)-n+1)))
    total = sum(counts.values())
    return {"n": n, "windows": total, "duplicate_windows": sum(c-1 for c in counts.values()),
            "max_occurrences": max(counts.values(), default=0)}


def score(records, contract):
    if (contract.get("schema") != "chimera_recovery.fold_rollout_contract.v1"
            or contract.get("decoding") != {"strategy": "greedy", "do_sample": False}
            or type(contract.get("vocab_size")) is not int or contract["vocab_size"] < 2
            or type(contract.get("max_new_tokens")) is not int or contract["max_new_tokens"] < 32):
        raise ValueError("explicit shared tokenizer, greedy decoding and 32+ token budget required")
    for name in ("tokenizer_sha256", "evaluation_source_sha256", "parent_checkpoint_sha256", "candidate_checkpoint_sha256"):
        if not SHA.fullmatch(str(contract.get(name, ""))):
            raise ValueError("missing actual pinned " + name)
    eos = set(tokens(contract.get("eos_ids"), "EOS IDs", contract["vocab_size"]))
    if not eos:
        raise ValueError("explicit native EOS set required")
    forbidden = set(contract.get("training_source_groups", []))
    seen, rows, groups = set(), [], set()
    for record in records:
        ident, group = record.get("id"), record.get("source_group_id")
        if (not isinstance(ident, str) or not ident or ident in seen or not isinstance(group, str) or not group
                or group in forbidden or record.get("split") not in {"validation", "test"}):
            raise ValueError("unique held identities and training-disjoint source groups required")
        seen.add(ident); groups.add(group)
        prompt = tokens(record.get("prompt_ids"), "prompt", contract["vocab_size"])
        if not prompt:
            raise ValueError("nonempty actual common prompt required")
        runs = []
        for name in ("parent", "candidate"):
            run = record.get(name, {})
            if run.get("prompt_sha256") != digest(prompt):
                raise ValueError("parent and candidate must bind identical full prompt IDs")
            if run.get("checkpoint_sha256") != contract[name + "_checkpoint_sha256"]:
                raise ValueError("rollout checkpoint differs from comparison contract")
            runs.append(completion(run, eos_ids=eos, max_new_tokens=contract["max_new_tokens"], vocab=contract["vocab_size"]))
        parent, child = runs
        prefix, longest, matches = longest_run(parent[1], child[1])
        rows.append({"id": ident, "source_group_id": group, "split": record["split"], "prompt_sha256": digest(prompt),
                     "parent_content_tokens": len(parent[1]), "candidate_content_tokens": len(child[1]),
                     "parent_stop_reason": parent[2], "candidate_stop_reason": child[2],
                     "longest_correct_prefix": prefix, "longest_aligned_match_run": longest,
                     "aligned_content_matches": matches,
                     "prefix_matches": {str(h): (prefix >= h if len(parent[1]) >= h else None) for h in HORIZONS},
                     "exact_complete_generation": bool(parent[1]) and parent[0] == child[0] and parent[2] == child[2] and parent[2] != "max_tokens",
                     "candidate_empty": not child[1], "candidate_capped": child[2] == "max_tokens",
                     "candidate_repetition": repeated_ngrams(child[1]),
                     "parent_repetition": repeated_ngrams(parent[1]),
                     "task_correctness": "not assessed by token agreement"})
    if not rows:
        raise ValueError("real paired rollouts required; empty report is not a retention result")
    horizons = {}
    for h in HORIZONS:
        eligible = [r["prefix_matches"][str(h)] for r in rows if r["prefix_matches"][str(h)] is not None]
        matched = sum(eligible)
        horizons[str(h)] = {"eligible": len(eligible), "matches": matched,
                            "rate": matched / len(eligible) if eligible else None,
                            "ineligible_short_parent": len(rows)-len(eligible)}
    output = {"schema": "chimera_recovery.fold_retention_report.v1", "contract": contract,
              "paired_rollouts_sha256": digest(records), "held_source_groups": sorted(groups), "rows": rows,
              "summary": {"pairs": len(rows), "consecutive_prefix": horizons,
                          "exact_complete_generations": sum(r["exact_complete_generation"] for r in rows),
                          "candidate_empty": sum(r["candidate_empty"] for r in rows),
                          "candidate_capped": sum(r["candidate_capped"] for r in rows),
                          "mean_correct_prefix": sum(r["longest_correct_prefix"] for r in rows)/len(rows)},
              "checkpoint_changed": False, "selection_performed": False,
              "limitations": ["Parent agreement is not independently verified task correctness or intelligence.",
                              "Top-k rank hits and teacher-forced logits are not consecutive free-running matches.",
                              "Short/EOS-only parent continuations are ineligible for longer horizons, not successful examples.",
                              "Source-group declarations must come from actual corpus admission; this tool does not discover undeclared leakage.",
                              "Use identical budgets, source sets and parent/model-size constraints before comparing folds."]}
    output["seal_sha256"] = digest(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("preserve previous measurement; new output path required")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    records = [json.loads(s) for s in args.rollouts.read_text(encoding="utf-8").splitlines() if s.strip()]
    result = score(records, contract)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical(result) + b"\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
