"""A separate bounded student-prefix cache round; no sampling or GPU launch here.

The frozen structured off-policy corpus is unchanged. Native student continuations
must be captured on admitted train-only prompts, then independently rescored by
the immutable parent. A dedicated candidate trains JSD over Q64+tail at those
student prefixes. CE/EOS/unlikelihood are zero: wrong sampled tokens are inputs,
not hard targets. This is snapshot on-policy recovery, not a live rollout loop.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

from . import corpus as C, text_eval_seed as E, train_text as T

Error = T.ReadinessError
POLICY = "chimera_recovery.student_prefix_one_pass.v1"
ROUTES = {"text_instruction", "text_reasoning", "text_arithmetic", "text_code"}
OBJECTIVE = {"w_q38": 1.0, "w_ce": 0.0, "w_eos": 0.0, "w_ul": 0.0,
             "onpolicy_divergence": "jsd", "jsd_beta": 0.5, "text_vocab_rows": 248320}
canonical, digest, sha = C.canonical, C.digest, C.file_sha256


def pin(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha(path)}


def checked(record):
    path = Path(record["path"])
    if not path.is_file() or sha(path) != record.get("sha256"):
        raise Error("student-prefix evidence bytes changed: " + str(path))
    return path


def group(row):
    p = row.get("provenance") or {}
    return p.get("recovery_group_id") or p.get("group_id") or row.get("group_id")


def prepare_prompts(training_paths, holdout_paths, exclusion_paths, output,
                    per_route=64, max_new_tokens=256, seed=20261013, split_inventory=None):
    """One prompt per source group, screened against full pinned held-out text.

    No source assistant answer is published in inputs. Source text remains
    ungraded; lexical/group screening is bounded, not semantic equivalence proof.
    """
    output = Path(output)
    if output.exists() or not 1 <= per_route <= 512 or not 32 <= max_new_tokens <= 512:
        raise Error("new output and bounded per-route/max-new-token limits required")
    training_paths, holdout_paths, exclusion_paths = [list(map(Path, v)) for v in
                                                   (training_paths, holdout_paths, exclusion_paths)]
    if not training_paths or not holdout_paths or not exclusion_paths or split_inventory is None:
        raise Error("train, full split inventory/frozen inputs and global exclusion sources required")
    split_inventory = Path(split_inventory)
    source_pins = {str(p.resolve()): sha(p) for p in training_paths + holdout_paths + exclusion_paths + [split_inventory]}
    excluded_ids, excluded_groups = set(), set()
    for path in exclusion_paths:
        for row in C.iter_jsonl(path):
            excluded_ids.add(row.get("id"))
            excluded_groups.update(v for v in (row.get("source_group_id"), row.get("recovery_group_id"), group(row)) if v)
    candidates, reasons = [], Counter()
    for path in training_paths:
        for row in C.iter_jsonl(path):
            p, q = row.get("provenance") or {}, row.get("quality") or {}
            route, g = row.get("route"), group(row)
            if (row.get("schema") != "recovery_corpus.v1" or row.get("split", p.get("split")) != "train"
                    or q.get("train_ready") is not True or q.get("blockers") or p.get("source_verified") is not True):
                reasons["source_not_explicitly_admitted_train"] += 1
                continue
            if route not in ROUTES or row.get("media") or row.get("tools"):
                reasons["outside_text_prompt_scope"] += 1
                continue
            if not g or row.get("id") in excluded_ids or g in excluded_groups:
                reasons["global_cross_route_exclusion_or_missing_group"] += 1
                continue
            messages = []
            for message in row.get("messages", []):
                if message.get("role") == "assistant":
                    break
                if message.get("role") not in {"system", "developer", "user"} or not isinstance(message.get("content"), str):
                    messages = []
                    break
                messages.append({"role": message["role"], "content": message["content"]})
            if not messages or messages[-1]["role"] != "user" or len(C.words(messages[-1]["content"])) < 8:
                reasons["incomplete_or_too_short_prompt"] += 1
                continue
            candidates.append({"schema": "chimera_recovery.onpolicy_prompt.v1", "id": digest(["student-prefix", row["id"]]),
                "route": route, "group_id": g, "split": "train", "input": {"messages": messages, "tools": [], "media": []},
                "input_messages_sha256": digest(messages), "source_row_id": row["id"], "source_row_sha256": digest(row),
                "source_file": {"path": str(path.resolve()), "sha256": source_pins[str(path.resolve())]}, "provenance": p})
    # Bound the lexical candidate index before streaming all heldout rows.
    # Select distinct real prompt/groups; never manufacture a missing route.
    candidates.sort(key=lambda r: (digest([int(seed), r["id"]]), r["id"]))
    bounded, seen_groups, seen_prompts, route_candidates = [], set(), set(), Counter()
    for row in candidates:
        prompt = digest(row["input"])
        if row["group_id"] in seen_groups or prompt in seen_prompts:
            reasons["duplicate_source_group_or_prompt"] += 1
            continue
        if route_candidates[row["route"]] >= per_route * 8:
            continue
        bounded.append(row); seen_groups.add(row["group_id"]); seen_prompts.add(prompt); route_candidates[row["route"]] += 1
    candidates = bounded
    # The existing candidate-index scanner consumes all heldout source text,
    # including individual messages, answer text and full conversations.
    # Multiple message prompts are joined only for the lexical comparison.
    screens = [{**r, "input": {"messages": [{"role": "user", "content": "\n".join(m["content"] for m in r["input"]["messages"])}]}}
               for r in candidates]
    # Include every normalized holdout row, including quarantined/blocked rows.
    # Filter only by recorded split so train prompts do not screen themselves.
    with tempfile.TemporaryDirectory(prefix="student-prefix-held-inventory.") as temporary:
        held_snapshot = Path(temporary) / "held.jsonl"
        held_count = 0
        with held_snapshot.open('wb') as handle:
            for row in C.iter_jsonl(split_inventory):
                if row.get('split', (row.get('provenance') or {}).get('split')) in {'validation', 'test'}:
                    handle.write(canonical(row) + b'\n'); held_count += 1
        if not held_count:
            raise Error("full split inventory contains no recorded held-out rows")
        held_snapshot_sha256 = sha(held_snapshot)
        blockers, held_pins, counts, _ = E.screen(screens, holdout_paths + [held_snapshot])
        temporary_key, inventory_key = str(held_snapshot.resolve()), str(split_inventory.resolve())
        held_pins.pop(temporary_key); counts.pop(temporary_key)
        held_pins[inventory_key], counts[inventory_key] = source_pins[inventory_key], held_count
        for values in blockers.values():
            for hit in values:
                if hit.get('source') == temporary_key:
                    hit['source'] = inventory_key
    candidates.sort(key=lambda r: (digest([int(seed), r["id"]]), r["id"]))
    selected, groups, prompts, coverage = [], set(), set(), Counter()
    for row in candidates:
        if row["id"] in blockers:
            reasons["heldout_text_or_group_overlap"] += 1
            continue
        prompt = digest(row["input"])
        if row["group_id"] in groups or prompt in prompts:
            reasons["duplicate_source_group_or_prompt"] += 1
            continue
        if coverage[row["route"]] >= per_route:
            continue
        selected.append(row); groups.add(row["group_id"]); prompts.add(prompt); coverage[row["route"]] += 1
    if not selected:
        raise Error("no screened train-only student-prefix prompts")
    for path, wanted in source_pins.items():
        if sha(Path(path)) != wanted:
            raise Error("prompt source changed during screening")
    manifest = {"schema": "chimera_recovery.onpolicy_prompts.v1", "date": "2026-10-13", "rows": len(selected),
        "routes": dict(coverage), "seed": int(seed), "max_new_tokens": max_new_tokens,
        "maximum_sampled_tokens": len(selected) * max_new_tokens, "actual_sampled_tokens": 0,
        "objective": OBJECTIVE, "source_pins": source_pins, "heldout_source_rows": counts,
        "heldout_source_pins": held_pins, "screened_out": dict(reasons), "train_only": True,
        "full_split_inventory": {"path": str(split_inventory.resolve()), "sha256": source_pins[str(split_inventory.resolve())],
                                 "heldout_rows": held_count, "heldout_snapshot_sha256": held_snapshot_sha256,
                                 "filter": "recorded split validation or test, including blocked rows"},
        "frozen_original_corpus_changed": False, "student_generation_completed": False,
        "production_training_ready": False,
        "blockers": ["Exact accepted student/checkpoint and native tokenizer/template receipt required",
                     "No student continuations or parent rescoring caches have been produced",
                     "Separate dated selected amendment and matched baseline/after full-answer review required"],
        "measurement": {"frozen_validation_and_test_optimizer_exclusion": True,
                        "free_run_horizons": [5, 8, 16, 32],
                        "separate_metrics": ["semantic correctness", "positional match", "LCP", "exact prefix horizon", "EOS", "cap", "loops", "teacher-forced top-k position coverage"],
                        "alternate_correct_answers": "semantic oracle accepts valid alternatives; exact-token retention is a separate diagnostic"},
        "limits": ["One bounded snapshot round; no claim of full online student-distribution coverage",
                   "Source content is structurally admitted and ungraded unless separately oracle-verified",
                   "Lexical/group exclusion does not prove absence of semantic duplicates",
                   "Maximum sampled tokens is a ceiling, never padded or duplicated to fill a budget"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".", dir=output.parent) as temporary:
        root = Path(temporary)
        (root / "prompts.jsonl").write_bytes(b"".join(canonical(r) + b"\n" for r in selected))
        manifest["prompts_sha256"] = sha(root / "prompts.jsonl")
        (root / "manifest.json").write_bytes(canonical(manifest) + b"\n")
        root.rename(output)
    return manifest


def validate_trace(seed, row, expected):
    """Validate recorded native IDs, never infer an unobserved student answer."""
    if (row.get("schema") != "chimera_recovery.student_trace.v1" or row.get("id") != seed["id"]
            or row.get("seed_sha256") != digest(seed) or row.get("input_messages_sha256") != seed["input_messages_sha256"]
            or seed.get("split") != "train" or seed.get("group_id") != row.get("group_id")):
        raise Error("student trace prompt/source/group identity differs")
    for key in ("student_checkpoint_sha256", "tokenizer_sha256", "template_sha256", "generation"):
        if row.get(key) != expected.get(key) or expected.get(key) is None:
            raise Error("student trace native checkpoint/template/generation pin differs")
    for key in ("student_checkpoint_sha256", "tokenizer_sha256", "template_sha256"):
        value = expected[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise Error("invalid student trace native identity SHA")
    prompt, generated = row.get("prompt_ids"), row.get("generated_ids")
    if (not isinstance(prompt, list) or not prompt or not isinstance(generated, list) or not generated
            or any(type(v) is not int or not 0 <= v < 248320 for v in prompt + generated)):
        raise Error("student trace has missing/out-of-text-band native IDs")
    cfg = expected["generation"]
    if (type(cfg.get("max_new_tokens")) is not int or not 32 <= cfg["max_new_tokens"] <= 512
            or cfg.get("temperature") != 0 or cfg.get("do_sample") is not False
            or not cfg.get("eos_ids") or any(type(v) is not int or not 0 <= v < 248320 for v in cfg["eos_ids"])):
        raise Error("bounded deterministic native generation config required")
    if len(generated) > cfg["max_new_tokens"] or any(v in cfg["eos_ids"] for v in generated[:-1]):
        raise Error("trace exceeds cap or continues after EOS")
    stop = "eos" if generated[-1] in cfg["eos_ids"] else "cap"
    if row.get("stop_reason") != stop or (stop == "cap" and len(generated) != cfg["max_new_tokens"]):
        raise Error("trace actual stop reason differs")
    if row.get("prompt_ids_sha256") != digest(prompt) or row.get("generated_ids_sha256") != digest(generated):
        raise Error("trace token content digest differs")
    if not isinstance(row.get("answer_text"), str) or row.get("answer_text_sha256") != digest(row["answer_text"]):
        raise Error("full recorded student answer bytes required")
    return prompt + generated, [False] * len(prompt) + [True] * len(generated)


def pack_traces(prompts, prompt_manifest, traces, generation_receipt, output):
    """Native CPU packing only after full captured generation receipt pins agree."""
    prompts, prompt_manifest, traces, generation_receipt, output = map(Path, (prompts, prompt_manifest, traces, generation_receipt, output))
    if output.exists():
        raise Error("retain prior student-prefix corpora")
    manifest, receipt = json.loads(prompt_manifest.read_text()), json.loads(generation_receipt.read_text())
    seeds = {r["id"]: r for r in C.iter_jsonl(prompts)}
    rows = list(C.iter_jsonl(traces))
    if (manifest.get("schema") != "chimera_recovery.onpolicy_prompts.v1" or manifest.get("prompts_sha256") != sha(prompts)
            or receipt.get("schema") != "chimera_recovery.student_generation_receipt.v1"
            or receipt.get("prompts_sha256") != sha(prompts) or receipt.get("prompt_manifest_sha256") != sha(prompt_manifest)
            or receipt.get("traces_sha256") != sha(traces) or receipt.get("completed") is not True
            or receipt.get("accepted_student") is not True or receipt.get("native_template_verified") is not True
            or len(rows) != len(seeds) or {r.get("id") for r in rows} != set(seeds)):
        raise Error("captured student generation must cover exact complete prompt inventory")
    for key in ("student_acceptance_receipt", "generator_code", "runtime_receipt"):
        checked(receipt[key])
    expected = receipt["identity"]
    if expected.get("generation", {}).get("max_new_tokens") != manifest["max_new_tokens"]:
        raise Error("trace budget differs from frozen prompt plan")
    accepted = json.loads(checked(receipt["student_acceptance_receipt"]).read_text())
    if accepted.get("accepted_complete_stage") is not True or accepted.get("checkpoint_sha256") != expected.get("student_checkpoint_sha256"):
        raise Error("missing exact accepted complete student state")
    data = [validate_trace(seeds[r["id"]], r, expected) for r in rows]
    import torch
    from safetensors.torch import save_file
    width = max(len(ids) for ids, _ in data)
    if width > 32768:
        raise Error("student trace exceeds reviewed context; never truncate")
    ids, masks = torch.zeros((len(rows), width), dtype=torch.int64), torch.zeros((len(rows), width), dtype=torch.bool)
    for i, (tokens, mask) in enumerate(data):
        ids[i, :len(tokens)] = torch.tensor(tokens); masks[i, :len(mask)] = torch.tensor(mask)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".", dir=output.parent) as temporary:
        root = Path(temporary)
        native = root / "student_prefix.safetensors"
        save_file({"input_ids": ids, "real_lengths": torch.tensor([len(tokens) for tokens, _ in data]), "assistant_mask": masks}, str(native))
        evidence = {"schema": "chimera_recovery.student_prefix_corpus.v1", "policy": POLICY,
            "corpus": {"path": str((output / native.name).resolve()), "sha256": sha(native)},
            "source_prompts": pin(prompts), "source_prompt_manifest": pin(prompt_manifest), "generation_receipt": pin(generation_receipt),
            "traces": pin(traces), "identity": expected, "rows": len(rows), "shape": list(ids.shape),
            "real_tokens": sum(len(tokens) for tokens, _ in data), "supervised_positions": sum(sum(mask) for _, mask in data),
            "objective": OBJECTIVE, "row_ids": [r["id"] for r in rows], "groups": [seeds[r["id"]]["group_id"] for r in rows],
            "teacher_cache_sealed": False, "production_training_ready": False, "training_started": False}
        (root / "manifest.json").write_bytes(canonical(evidence) + b"\n"); root.rename(output)
    return evidence


class OnePassSampler(T.OnePassSampler):
    def state(self):
        return {**super().state(), "policy": POLICY}


accumulation_window, verify_progress, verify_amendment = T.accumulation_window, T.verify_progress, T.verify_amendment


def verify_objective(args):
    if not getattr(args, "structured_recovery", False):
        raise Error("dedicated student-prefix candidate requires structured one-pass")
    for key, value in OBJECTIVE.items():
        if getattr(args, key, None) != value:
            raise Error("student-prefix objective differs: " + key)


def verify_round(args):
    """Fresh round imports an accepted complete D2 bank; exact resume is separate."""
    path = getattr(args, "student_prefix_round", None)
    if not path or not getattr(args, "init_adapters", None):
        raise Error("student-prefix round needs exact accepted student and --init-adapters")
    evidence = json.loads(Path(path).read_text())
    if evidence.get("schema") != "chimera_recovery.student_prefix_round.v1":
        raise Error("student-prefix round schema differs")
    for key in ('student_checkpoint_sha256', 'base_source_identity_sha256'):
        value = evidence.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise Error("student-prefix round complete native identity missing")
    accepted = json.loads(checked(evidence["student_acceptance_receipt"]).read_text())
    if (accepted.get("accepted_complete_stage") is not True or accepted.get("stage") != "D2"
            or accepted.get("checkpoint_sha256") != evidence.get("student_checkpoint_sha256")
            or accepted.get("base_source_identity_sha256") != evidence.get("base_source_identity_sha256")
            or accepted.get("bank_scope") != "D2.full_bank_replacement"
            or accepted.get("adapters") != evidence.get("init_adapters")
            or sha(Path(args.init_adapters)) != evidence["init_adapters"]["sha256"]):
        raise Error("student-prefix round accepted student/base/full-bank binding differs")
    checked(evidence["init_adapters"])
    spec = json.loads(Path(args.data_config).read_text())
    if spec.get("student_prefix_round") != pin(path):
        raise Error("data config references another student-prefix round")
    corpus_receipt = json.loads(checked(spec["student_prefix_corpus"]).read_text())
    if (corpus_receipt.get("identity", {}).get("student_checkpoint_sha256") != evidence.get("student_checkpoint_sha256")
            or corpus_receipt.get("generation_receipt") != evidence.get("generation_receipt")
            or corpus_receipt.get("objective") != OBJECTIVE):
        raise Error("student-prefix round was not sampled by this accepted student")
    return evidence


def verify_source(args, source):
    evidence = verify_round(args)
    if digest(source.identity()) != evidence["base_source_identity_sha256"]:
        raise Error("onpolicy base source changed or already composed; adapter must apply exactly once")


def load_sources(path, api):
    spec = json.loads(Path(path).read_text())
    corpus_receipt = json.loads(checked(spec["student_prefix_corpus"]).read_text())
    if (corpus_receipt.get("schema") != "chimera_recovery.student_prefix_corpus.v1" or corpus_receipt.get("policy") != POLICY
            or corpus_receipt.get("objective") != OBJECTIVE or len(spec.get("sources", [])) != 1):
        raise Error("dedicated student-prefix round/corpus evidence differs")
    checked(corpus_receipt["generation_receipt"])
    checked(corpus_receipt["source_prompts"]); checked(corpus_receipt["source_prompt_manifest"]); checked(corpus_receipt["traces"])
    teacher_sha = spec.get("teacher_sha256")
    if not isinstance(teacher_sha, str) or len(teacher_sha) != 64 or any(v not in '0123456789abcdef' for v in teacher_sha):
        raise Error("immutable parent teacher SHA required")
    item = spec["sources"][0]
    if (item.get("kind") != "onpolicy" or item.get("supervision") != "assistant_only"
            or any(item.get(key) not in (None, 0) for key in ("fraction", "start_after_tokens", "tokens"))
            or item.get("weight", 1.0) != 1.0 or sha(Path(item["corpus"])) != corpus_receipt["corpus"]["sha256"]):
        raise Error("onpolicy corpus/mixing/delay contract differs")
    sources, identity = api["load_data_config"](Path(path))
    verifier = api["load_peer_module"]("qtext_teacher_cache")
    source = sources[0]
    teacher = source.cache.manifest.get("teacher", {})
    if (source.cache.corpus_assistant_mask is None or source.cache.top_k != 64
            or source.cache.manifest_sha256 != spec.get("cache_manifest_sha256")
            or source.cache.manifest.get("cache_config", {}).get("vocab_rows") != 248320
            or teacher.get("equal_before_and_after_every_session") is not True
            or teacher.get("tree_sha256_before") != spec.get("teacher_sha256")
            or teacher.get("tree_sha256_after") != spec.get("teacher_sha256")
            or source.rows != corpus_receipt["rows"]):
        raise Error("immutable parent Q64 teacher/cache/full student inventory differs")
    raw = []
    for entry in source.cache.manifest["shards"]:
        name = entry["shard"]
        if Path(name).name != name or verifier.verify_shard(source.cache.dir / name).get("ok") is not True:
            raise Error("actual student-prefix parent raw shard failed verification")
        raw.append(source.cache.records[name]["files"])
    OnePassSampler(sources, 0)
    identity.update(policy=POLICY, helper_sha256=sha(Path(__file__)), structured_helper_sha256=sha(Path(T.__file__)),
                    student_prefix_corpus_sha256=spec["student_prefix_corpus"]["sha256"],
                    raw_shards_verified_sha256=digest(raw), teacher_sha256=spec["teacher_sha256"],
                    cache_verifier={"path": verifier.__file__, "sha256": sha(Path(verifier.__file__))})
    return sources, identity


def bind_cache(corpus_manifest, cache, teacher_sha256, round_receipt, output):
    """Bind authentic parent rescoring to the exact new student-prefix corpus.

    Raw shards are verified by the real cache helper again at train/resume.
    This metadata binding never certifies a GPU runtime or decoder correctness.
    """
    corpus_manifest, cache, round_receipt, output = map(Path, (corpus_manifest, cache, round_receipt, output))
    if output.exists():
        raise Error("retain earlier student-prefix data config")
    corpus = json.loads(corpus_manifest.read_text())
    if corpus.get("schema") != "chimera_recovery.student_prefix_corpus.v1" or corpus.get("objective") != OBJECTIVE:
        raise Error("student-prefix native corpus schema/objective differs")
    checked(corpus["corpus"]); checked(corpus["generation_receipt"])
    manifest = json.loads((cache / "manifest.json").read_text())
    teacher = manifest.get("teacher", {})
    if (manifest.get("schema") != "omni-distill-teacher-cache-manifest-v1"
            or manifest.get("all_finite") is not True or manifest.get("all_structural_ok") is not True
            or manifest.get("cache_config", {}).get("top_k") != 64
            or manifest.get("cache_config", {}).get("vocab_rows") != 248320
            or manifest.get("corpus", {}).get("sha256") != corpus["corpus"]["sha256"]
            or teacher.get("equal_before_and_after_every_session") is not True
            or teacher.get("tree_sha256_before") != teacher_sha256 or teacher.get("tree_sha256_after") != teacher_sha256
            or not manifest.get("shards")):
        raise Error("parent cache is incomplete or not exact immutable Q64 student-prefix rescoring")
    for entry in manifest["shards"]:
        name = entry["shard"]
        path = cache / name / "shard.json"
        if Path(name).name != name or sha(path) != entry["shard_json_sha256"]:
            raise Error("sealed teacher shard metadata differs")
    spec = {"sources": [{"name": "student-prefix-snapshot", "kind": "onpolicy", "supervision": "assistant_only",
                         "qtext_cache": str(cache.resolve()), "corpus": corpus["corpus"]["path"]}],
            "student_prefix_corpus": pin(corpus_manifest), "student_prefix_round": pin(round_receipt),
            "teacher_sha256": teacher_sha256, "cache_manifest_sha256": sha(cache / "manifest.json")}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exercise the actual round initialization contract before publishing config.
    with tempfile.TemporaryDirectory(prefix=output.name + ".", dir=output.parent) as temporary:
        path = Path(temporary) / output.name; path.write_bytes(canonical(spec) + b"\n")
        round_record = json.loads(round_receipt.read_text())
        verify_round(argparse.Namespace(student_prefix_round=str(round_receipt), init_adapters=round_record["init_adapters"]["path"], data_config=str(path)))
        path.rename(output)
    return {"schema": POLICY, "data_config": pin(output), "parent_cache_manifest": pin(cache / "manifest.json"),
            "raw_teacher_validation_required_at_every_run_resume": True, "production_training_ready": False}


def patch_trainer(source):
    candidate, original = T.patch_trainer(Path(source))
    old = '        from experiments.chimera_recovery import train_text as recovery_text\n'
    new = ('        from experiments.chimera_recovery import train_onpolicy as recovery_text\n'
           '        recovery_text.verify_objective(args)\n'
           '        recovery_text.verify_round(args)\n')
    if candidate.count(old) != 1:
        raise Error("exact structured trainer anchor differs")
    candidate = candidate.replace(old, new, 1)
    old = '    source = build_source(args)\n'
    node = next(n for n in ast.parse(candidate).body if isinstance(n, ast.FunctionDef) and n.name == 'command_train')
    lines = candidate.splitlines(True)
    command = ''.join(lines[node.lineno-1:node.end_lineno])
    if command.count(old) != 1:
        raise Error("onpolicy source binding anchor differs")
    command = command.replace(old, old + '    recovery_text.verify_source(args, source)\n', 1)
    candidate = ''.join(lines[:node.lineno-1]) + command + ''.join(lines[node.end_lineno:])
    parser = '    tr.set_defaults(func=command_train)\n'
    if candidate.count(parser) != 1:
        raise Error("onpolicy parser binding anchor differs")
    candidate = candidate.replace(parser, '    tr.add_argument("--student-prefix-round", required=True, help="exact accepted student/full-bank and generation round receipt")\n' + parser, 1)
    # Refuse the legacy branch too, before source weights/model construction.
    anchor = '    structured = bool(getattr(args, "structured_recovery", False))\n'
    if candidate.count(anchor) != 1:
        raise Error("dedicated candidate mode anchor differs")
    candidate = candidate.replace(anchor, anchor + '    if not structured:\n        raise TrainerError("dedicated student-prefix candidate requires --structured-recovery")\n', 1)
    compile(candidate, "student-prefix-candidate", "exec")
    return candidate, {"schema": POLICY, "baseline": original, "helper_sha256": sha(Path(__file__)),
                        "objective": OBJECTIVE, "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
                        "runtime_validation_required": True, "production_training_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="mode", required=True)
    prompts = sub.add_parser("prepare-prompts")
    for key in ("training-paths", "holdout-paths", "exclusion-paths"):
        prompts.add_argument("--" + key, nargs="+", type=Path, required=True)
    prompts.add_argument("--output", type=Path, required=True)
    prompts.add_argument("--split-inventory", type=Path, required=True)
    prompts.add_argument("--per-route", type=int, default=64)
    pack = sub.add_parser("pack-traces")
    for key in ("prompts", "prompt-manifest", "traces", "generation-receipt", "output"):
        pack.add_argument("--" + key, type=Path, required=True)
    candidate = sub.add_parser("prepare-trainer")
    candidate.add_argument("--source", type=Path, required=True); candidate.add_argument("--output", type=Path, required=True)
    binding = sub.add_parser("bind-cache")
    for key in ("corpus-manifest", "cache", "round-receipt", "output"):
        binding.add_argument("--" + key, type=Path, required=True)
    binding.add_argument("--teacher-sha256", required=True)
    args = vars(parser.parse_args()); mode = args.pop("mode")
    if mode == "prepare-prompts":
        result = prepare_prompts(**args)
    elif mode == "pack-traces":
        result = pack_traces(**args)
    elif mode == "bind-cache":
        result = bind_cache(**args)
    else:
        output = args.pop("output")
        if output.exists() or output.with_suffix(".receipt.json").exists():
            raise Error("retain earlier candidate source")
        candidate, result = patch_trainer(**args); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(candidate.encode()); output.with_suffix(".receipt.json").write_bytes(canonical(result) + b"\n")
    print(json.dumps({key: result[key] for key in ("schema", "rows", "routes", "production_training_ready") if key in result}))


if __name__ == "__main__":
    main()
