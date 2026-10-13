"""Opt-in structured recovery for the exact audited streamed D2 engine.

The original D2 lane keeps its sampler, padding and normalization. This lane
reads authentic sealed assistant-only Q-text caches, visits every accepted row
once, crops right padding, and normalizes the entire accumulation window by
the number of assistant prediction positions. No model or GPU is loaded here.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

from .training import ReadinessError, canonical, digest, sha, streamed_supervision_patch

POLICY = "chimera_recovery.structured_one_pass.v1"


def apply_exact_diff(source: str, patch: str) -> str:
    lines = source.splitlines(True)
    out, pos = [], 0
    for block in re.split(r"(?=^@@ )", patch, flags=re.M)[1:]:
        header, *body = block.splitlines(True)
        match = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", header)
        if not match:
            raise ReadinessError("invalid patch hunk")
        start = int(match[1]) - 1
        if start < pos:
            raise ReadinessError("overlapping patch hunks")
        out.extend(lines[pos:start]); pos = start
        for line in body:
            if line[:1] in (" ", "-"):
                if pos >= len(lines) or lines[pos] != line[1:]:
                    raise ReadinessError("patch context differs from pinned source")
                if line[0] == " ":
                    out.append(lines[pos])
                pos += 1
            elif line.startswith("+"):
                out.append(line[1:])
            else:
                raise ReadinessError("unexpected patch line")
    out.extend(lines[pos:])
    return "".join(out)


class OnePassSampler:
    """Hash-sorted exclusive row plan: stable across Python/Torch versions."""
    def __init__(self, sources, seed):
        self.seed = int(seed)
        self.sources = sources
        names = [s.name for s in sources]
        if not sources or len(set(names)) != len(names):
            raise ReadinessError("one-pass sources must be nonempty and uniquely named")
        rows = [(i, int(r)) for i, s in enumerate(sources) for r in s.row_ids]
        if not rows or len(rows) != len(set(rows)):
            raise ReadinessError("one-pass row plan is empty or repeats a cache row")
        self.plan = sorted(rows, key=lambda x: (digest([self.seed, names[x[0]], x[1]]), x))
        self.plan_sha256 = digest([[names[i], r] for i, r in self.plan])
        self.cursor = 0

    @property
    def remaining(self):
        return len(self.plan) - self.cursor

    def next_rows(self, count, tokens_seen=0):
        if int(count) <= 0:
            raise ReadinessError("row count must be positive")
        end = min(self.cursor + int(count), len(self.plan))
        result = self.plan[self.cursor:end]
        self.cursor = end
        return result

    def state(self):
        return {"policy": POLICY, "seed": self.seed, "plan_sha256": self.plan_sha256,
                "cursor": self.cursor, "rows": len(self.plan)}

    def load_state(self, state):
        expected = self.state()
        for name in ("policy", "seed", "plan_sha256", "rows"):
            if state.get(name) != expected[name]:
                raise ReadinessError(f"one-pass resume {name} differs")
        cursor = state.get("cursor")
        if type(cursor) is not int or not 0 <= cursor <= len(self.plan):
            raise ReadinessError("invalid one-pass resume cursor")
        self.cursor = cursor


def load_sources(path, api):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    sources, records = [], []
    verifier = api["load_peer_module"]("qtext_teacher_cache")
    verifier_identity = {"path": verifier.__file__, "sha256": sha(Path(verifier.__file__))}
    for item in spec["sources"]:
        if item.get("kind", "text") != "text" or item.get("supervision") != "assistant_only":
            raise ReadinessError("structured lane requires assistant-only text sources; K3/on-policy stay in their replay lanes")
        for key in ("tokens", "fraction", "start_after_tokens"):
            if item.get(key) not in (None, 0):
                raise ReadinessError("structured one-pass cannot use mixture budgets or delayed sources")
        if item.get("weight", 1.0) != 1.0:
            raise ReadinessError("one-pass rows cannot be reweighted by repeating a source")
        cache = api["QTextCache"](Path(item["qtext_cache"]), corpus_path=Path(item["corpus"]) if item.get("corpus") else None)
        # QTextCache verifies metadata, not the five raw target files. Audit all
        # bytes and structural contents before any update or resume is allowed.
        raw_records = []
        for entry in cache.manifest["shards"]:
            name = entry["shard"]
            if Path(name).name != name:
                raise ReadinessError("Q-text shard escapes its sealed cache")
            checked = verifier.verify_shard(cache.dir / name)
            if checked.get("ok") is not True:
                raise ReadinessError(f"Q-text raw shard verification failed: {name}: {checked.get('problems')}")
            raw_records.append({"shard": name, "files": cache.records[name]["files"]})
        if cache.corpus_assistant_mask is None:
            raise ReadinessError("structured cache lacks assistant ownership")
        source = api["DataSource"](name=item["name"], kind="text", cache=cache, row_ids=sorted(cache.index))
        sources.append(source)
        records.append({"name": source.name, "kind": source.kind, "rows": source.rows,
                        "qtext_manifest_sha256": cache.manifest_sha256,
                        "corpus_sha256": cache.manifest["corpus"]["sha256"],
                        "native_seq_len": cache.seq_len, "supervision": "assistant_only",
                        "length_source": cache.length_source, "rows_cut_to_real_lengths": cache.rows_cut_to_real_lengths})
        records[-1]["raw_shards_verified_sha256"] = digest(raw_records)
    if len({record["corpus_sha256"] for record in records}) != len(records):
        raise ReadinessError("same native corpus is aliased under multiple structured sources")
    OnePassSampler(sources, 0)  # validate exclusive row identities before model updates
    if len({s.cache.top_k for s in sources}) != 1:
        raise ReadinessError("structured sources must share teacher top-k")
    return sources, {"config_sha256": sha(Path(path)), "policy": POLICY,
                     "helper_sha256": sha(Path(__file__)), "cache_verifier": verifier_identity, "sources": records}


def cropped_batch(sources, picks, objective, builder):
    import torch
    if not picks:
        raise ReadinessError("cannot build an empty accumulation batch")
    rows = [sources[s].cache.row(r) for s, r in picks]
    length = max(int(row["length"]) for row in rows)
    if length < 2:
        raise ReadinessError("a conversation has no prediction positions")
    cropped, supervised = {}, 0
    for (s, r), row in zip(picks, rows):
        real = int(row["length"])
        mask = row.get("assistant_mask")
        if mask is None or mask.dtype != torch.bool or real > len(mask) or bool(mask[real:].any()):
            raise ReadinessError("assistant ownership is missing or extends into padding")
        count = int(mask[1:real].sum())
        if count == 0:
            raise ReadinessError("row has no supervised next-token assistant position")
        supervised += count
        output = {}
        for key, value in row.items():
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == row["ids"].shape[0]:
                if value.shape[0] >= length:
                    output[key] = value[:length]
                else:
                    fill = float("-inf") if key == "topk_logprobs" else -1 if key == "topk_ids" else 0
                    pad = torch.full((length - value.shape[0], *value.shape[1:]), fill, dtype=value.dtype, device=value.device)
                    output[key] = torch.cat((value, pad))
            else:
                output[key] = value
        cropped[(s, r)] = output
    proxies = [SimpleNamespace(**{**vars(source), "cache": SimpleNamespace(row=lambda rid, i=i: cropped[(i, rid)])})
               for i, source in enumerate(sources)]
    ids, targets = builder(proxies, picks, objective)
    stats = {"rows": len(rows), "real_tokens": sum(int(r["length"]) for r in rows),
             "compute_tokens": int(ids.numel()), "supervised_positions": supervised}
    return ids, targets, stats


def accumulation_window(sources, sampler, micro_batch, accumulation, objective, builder):
    picks = sampler.next_rows(micro_batch * accumulation)
    batches, total = [], {"rows": 0, "real_tokens": 0, "compute_tokens": 0, "supervised_positions": 0}
    for offset in range(0, len(picks), micro_batch):
        ids, targets, stats = cropped_batch(sources, picks[offset:offset + micro_batch], objective, builder)
        batches.append((ids, targets))
        for key in total:
            total[key] += stats[key]
    if not total["supervised_positions"]:
        raise ReadinessError("one-pass accumulation window exhausted")
    return batches, total


def verify_amendment(document, amendment):
    raw = Path(document).read_bytes().replace(b"\r\n", b"\n")
    extra = Path(amendment).read_bytes().replace(b"\r\n", b"\n")
    marker = raw.find(b"\n## Amendments")
    if marker < 0 or extra not in raw[marker:]:
        raise ReadinessError("structured recovery amendment is absent from the preregistration tail")
    return {"document_sha256": sha(Path(document)), "appendix_sha256": hashlib.sha256(extra).hexdigest(),
            "amendment_tail_sha256": hashlib.sha256(raw[marker:]).hexdigest()}


def verify_progress(sampler, progress, step, tokens_seen, micro_batch, accumulation, optimizer):
    expected_cursor = min(int(step) * int(micro_batch) * int(accumulation), len(sampler.plan))
    if sampler.cursor != expected_cursor:
        raise ReadinessError("one-pass cursor disagrees with applied optimizer windows")
    if (not isinstance(progress, dict) or progress.get("rows") != expected_cursor or
            progress.get("real_tokens") != int(tokens_seen) or progress.get("applied_steps") != int(step)):
        raise ReadinessError("structured progress disagrees with checkpoint counters")
    if any(type(progress.get(k)) is not int or progress[k] < 0 for k in
           ("rows", "real_tokens", "compute_tokens", "supervised_positions", "applied_steps")):
        raise ReadinessError("invalid structured progress counters")
    if progress["compute_tokens"] < progress["real_tokens"] or progress["supervised_positions"] > progress["real_tokens"]:
        raise ReadinessError("structured token counts are inconsistent")
    # Derive counters independently from the immutable source plan, so editing
    # two agreeing checkpoint fields cannot falsify the number of consumed rows.
    actual = {"real_tokens": 0, "compute_tokens": 0, "supervised_positions": 0}
    for offset in range(0, expected_cursor, int(micro_batch)):
        lengths = []
        for source_index, row_id in sampler.plan[offset:min(offset + int(micro_batch), expected_cursor)]:
            cache = sampler.sources[source_index].cache
            length = int(cache.index[row_id][3])
            lengths.append(length)
            actual["real_tokens"] += length
            actual["supervised_positions"] += int(cache.corpus_assistant_mask[row_id, 1:length].sum())
        actual["compute_tokens"] += len(lengths) * max(lengths)
    if any(progress[key] != value for key, value in actual.items()):
        raise ReadinessError("structured counters differ from the immutable row-plan prefix")
    adam_steps = [int(value["step"]) for value in optimizer.state.values() if "step" in value]
    if step and (not adam_steps or max(adam_steps) != int(step) or any(n > step for n in adam_steps)):
        raise ReadinessError("Adam state disagrees with applied steps")
    return dict(progress)


def patch_trainer(source: Path):
    patch, mask_receipt = streamed_supervision_patch(source)
    before = source.read_text(encoding="utf-8").replace("\r\n", "\n")
    candidate = apply_exact_diff(before, patch)
    if hashlib.sha256(candidate.encode()).hexdigest() != mask_receipt["patched_lf_sha256"]:
        raise ReadinessError("assistant patch reconstruction digest differs")
    changes = [
        ('    tr.set_defaults(func=command_train)\n',
         '    tr.add_argument("--structured-recovery", action="store_true", help="one pass, dynamic right padding, assistant-position normalization; one replica only")\n'
         '    tr.add_argument("--recovery-amendment", help="exact appended structured recovery amendment; required by --structured-recovery")\n'
         '    tr.set_defaults(func=command_train)\n'),
        ('    kernels = select_linear_attention_kernels(args.kernels)\n    dtype = resolve_dtype(args.dtype)\n    source = build_source(args)\n',
         '    structured = bool(getattr(args, "structured_recovery", False))\n'
         '    if structured:\n'
         '        if world != 1 or args.allow_config_change or args.allow_identity_change or args.no_preregistration:\n'
         '            raise TrainerError("structured recovery requires one replica, pinned preregistration and exact resume")\n'
         '        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n'
         '        from experiments.chimera_recovery import train_text as recovery_text\n'
         '        if not args.recovery_amendment or not args.preregistration:\n'
         '            raise TrainerError("structured recovery requires its appended --recovery-amendment")\n'
         '        recovery_amendment = recovery_text.verify_amendment(args.preregistration, args.recovery_amendment)\n'
         '    kernels = select_linear_attention_kernels(args.kernels)\n    dtype = resolve_dtype(args.dtype)\n    source = build_source(args)\n'),
        ('    sources, data_identity = load_data_config(Path(args.data_config))\n    if sources[0].cache.seq_len != args.seq_len:\n',
         '    sources, data_identity = (recovery_text.load_sources(Path(args.data_config), globals()) if structured else load_data_config(Path(args.data_config)))\n'
         '    if structured and any(s.cache.seq_len > args.seq_len for s in sources):\n'
         '        raise TrainerError("native corpus width exceeds structured context limit")\n'
         '    if not structured and sources[0].cache.seq_len != args.seq_len:\n'),
        ('    sampler = MixtureSampler(sources, args.seed)\n',
         '    sampler = recovery_text.OnePassSampler(sources, args.seed) if structured else MixtureSampler(sources, args.seed)\n'),
        ('    denominator = float(args.micro_batch * (args.seq_len - 1) * args.accumulation * world)\n',
         '    denominator = float(args.micro_batch * (args.seq_len - 1) * args.accumulation * world)\n'
         '    if structured:\n'
         '        exact_steps = math.ceil(len(sampler.plan) / (args.micro_batch * args.accumulation))\n'
         '        if args.total_steps and int(args.total_steps) != exact_steps:\n'
         '            raise TrainerError("structured total steps must equal the exclusive row plan")\n'
         '        total_steps = exact_steps\n'
         '        data_identity["row_plan_sha256"] = sampler.plan_sha256\n'
         '        data_identity["trainer_sha256"] = file_sha256(Path(__file__))\n'
         '        data_identity["recovery_amendment"] = recovery_amendment\n'),
        ('    halt_reason: str | None = None\n    while step < stop_at:\n',
         '    halt_reason: str | None = None\n'
         '    if structured and rank == 0:\n'
         '        out.mkdir(parents=True, exist_ok=True)\n'
         '        write_json_atomic(out / f"initial-receipt-segment-{len(segments)-1}.json", {"schema": recovery_text.POLICY, "before_first_update": True, "config_digest": config_digest, "preregistration": preregistration, "source_identity": source_identity, "data_identity": data_identity, "sampler": sampler.state(), "normalization": "sum assistant predicting positions across accumulation window", "total_steps": total_steps})\n'
         '    while step < stop_at:\n'),
        ('    step, tokens_seen, history = 0, 0, []\n',
         '    step, tokens_seen, history = 0, 0, []\n'
         '    structured_progress = {"rows": 0, "real_tokens": 0, "compute_tokens": 0, "supervised_positions": 0, "applied_steps": 0}\n'),
        ('            step, tokens_seen, history = int(state["step"]), int(state["tokens_seen"]), list(state.get("history_tail", []))\n',
         '            step, tokens_seen, history = int(state["step"]), int(state["tokens_seen"]), list(state.get("history_tail", []))\n'
         '            if structured:\n'
         '                structured_progress = recovery_text.verify_progress(sampler, state.get("structured_progress"), step, tokens_seen, args.micro_batch, args.accumulation, optimizer)\n'),
        ('                "segments": segments, "halt_reason": halt_reason}\n',
         '                "segments": segments, "halt_reason": halt_reason,\n'
         '                **({"structured_progress": dict(structured_progress)} if structured else {})}\n'),
        ('        for _ in range(args.accumulation):\n            picks = sampler.next_rows(args.micro_batch * world, tokens_seen)\n            mine = picks[rank * args.micro_batch : (rank + 1) * args.micro_batch]\n            ids, targets = build_batch(sources, mine, objective)\n            parts = engine.forward_backward(ids, targets, objective, denominator=denominator, loss_scale=scale)\n            for k, v in parts.items():\n                parts_sum[k] = parts_sum.get(k, 0.0) + float(v)\n',
         '        step_stats = None\n'
         '        if structured:\n'
         '            batches, step_stats = recovery_text.accumulation_window(sources, sampler, args.micro_batch, args.accumulation, objective, build_batch)\n'
         '            step_denominator = float(step_stats["supervised_positions"])\n'
         '            for ids, targets in batches:\n'
         '                parts = engine.forward_backward(ids, targets, objective, denominator=step_denominator, loss_scale=scale)\n'
         '                for k, v in parts.items():\n'
         '                    parts_sum[k] = parts_sum.get(k, 0.0) + float(v)\n'
         '        else:\n'
         '            for _ in range(args.accumulation):\n'
         '                picks = sampler.next_rows(args.micro_batch * world, tokens_seen)\n'
         '                mine = picks[rank * args.micro_batch : (rank + 1) * args.micro_batch]\n'
         '                ids, targets = build_batch(sources, mine, objective)\n'
         '                parts = engine.forward_backward(ids, targets, objective, denominator=denominator, loss_scale=scale)\n'
         '                for k, v in parts.items():\n'
         '                    parts_sum[k] = parts_sum.get(k, 0.0) + float(v)\n'),
        ('        step += 1\n        tokens_seen += tokens_per_step\n',
         '        if structured:\n'
         '            if finite:\n'
         '                step += 1\n'
         '                tokens_seen += step_stats["real_tokens"]\n'
         '                for key, value in step_stats.items():\n'
         '                    structured_progress[key] += value\n'
         '                structured_progress["applied_steps"] = step\n'
         '            else:\n'
         '                sampler.cursor -= step_stats["rows"]\n'
         '        else:\n'
         '            step += 1\n'
         '            tokens_seen += tokens_per_step\n'),
        ('        history.append(record)\n        receipt_steps.append(record)\n',
         '        if structured:\n'
         '            record.update(step_stats)\n'
         '            record["tok_per_s"] = round(step_stats["real_tokens"] / seconds, 2)\n'
         '            record["normalization_positions"] = step_stats["supervised_positions"]\n'
         '        history.append(record)\n        receipt_steps.append(record)\n'),
        ('        receipt_path = Path(args.receipt) if args.receipt else out / "training-receipt.json"\n',
         '        if structured:\n'
         '            receipt["structured_policy"] = recovery_text.POLICY\n'
         '            receipt["tokens_per_step"] = None\n'
         '            receipt["tokens_per_second_this_run"] = round(sum(r["real_tokens"] for r in receipt_steps) / wall, 3) if wall > 0 else None\n'
         '            receipt["row_plan"] = sampler.state()\n'
         '            receipt["segment_counts"] = {k: sum(r[k] for r in receipt_steps) for k in ("rows", "real_tokens", "compute_tokens", "supervised_positions")}\n'
         '            receipt["applied_supervised_positions"] = sum(r["supervised_positions"] for r in applied)\n'
         '            receipt["cumulative_applied_progress"] = dict(structured_progress)\n'
         '            receipt["nonfinite_row_policy"] = "retry exact window without advancing sampler or optimizer; frozen skip guard halts repeated failures"\n'
         '        receipt_path = Path(args.receipt) if args.receipt else out / "training-receipt.json"\n'),
    ]
    parser_old, parser_new = changes[0]
    if candidate.count(parser_old) != 1:
        raise ReadinessError("structured parser anchor absent or ambiguous")
    candidate = candidate.replace(parser_old, parser_new, 1)
    node = next(n for n in ast.parse(candidate).body if isinstance(n, ast.FunctionDef) and n.name == "command_train")
    lines = candidate.splitlines(True)
    command = "".join(lines[node.lineno - 1:node.end_lineno])
    for index, (old, new) in enumerate(changes[1:], 1):
        if command.count(old) != 1:
            raise ReadinessError(f"structured command patch anchor {index} absent or ambiguous")
        command = command.replace(old, new, 1)
    candidate = "".join(lines[:node.lineno - 1]) + command + "".join(lines[node.end_lineno:])
    ast.parse(candidate)
    receipt = {"schema": POLICY, "baseline_sha256": sha(source),
               "assistant_patch": mask_receipt, "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
               "helper_sha256": sha(Path(__file__)), "legacy_lane_preserved": True,
               "world_size": 1, "dynamic_right_padding": True, "single_exclusive_row_pass": True,
               "normalization": "assistant prediction positions over full accumulation window",
               "runtime_validation_required": True, "production_training_ready": False}
    return candidate, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".receipt.json").exists():
        raise ReadinessError("candidate outputs already exist; retain earlier evidence")
    candidate, receipt = patch_trainer(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(candidate.encode())
    args.out.with_suffix(".receipt.json").write_bytes(canonical(receipt) + b"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
