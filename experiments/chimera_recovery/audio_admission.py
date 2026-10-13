"""Seal global audio source eligibility; never run a teacher or grant native readiness.

Only recorded source files, actual PCM, frozen evaluation and the complete prior
connected admission are inspected. H3 encoder/tensor admission remains separate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import wave

SCHEMA = "recovery_audio_admission.v1"
SPLITS = {"train", "validation", "test"}
SHA = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("source JSONL row must be an object")
                yield value


def pinned(spec, label):
    path = Path(spec["path"])
    if not SHA.fullmatch(str(spec.get("sha256", ""))) or file_sha(path) != spec["sha256"]:
        raise ValueError("missing or changed source pin: " + label)
    return path


def tokens(text):
    return tuple(re.findall(r"\w+", str(text).casefold()))


def texts(record):
    return [m["content"] for m in record.get("messages", []) if isinstance(m.get("content"), str)]


def identities(record):
    p = record.get("provenance") or {}
    result = {"uid:" + str(record["id"])} if record.get("id") else set()
    for group in (record.get("group_id"), p.get("group_id"), p.get("recovery_group_id")):
        if group:
            result.add("group:" + str(group))
    result.update("source:" + str(p.get("source", "")) + ":" + str(sid) for sid in p.get("source_ids", []))
    rid = (p.get("registry_origin") or {}).get("rid")
    if rid:
        result.add("rid:" + str(rid))
    for field in ("source_payload_sha256s", "archive_payload_sha256s"):
        result.update("payload:" + str(v) for v in p.get(field, []))
    for media in record.get("media", []):
        if SHA.fullmatch(str(media.get("sha256", ""))):
            result.add("asset:" + media["sha256"])
        if media.get("path"):
            result.add("asset-path:" + str(media["path"]).replace("\\", "/"))
    # An observation's generic instruction is not its content identity: its
    # actual media and family must participate. Text-only task identity is full.
    task = tokens("\n".join(texts(record)))
    if len(task) >= 4:
        result.add("task:" + digest(task))
    if not record.get("media"):
        result.update("wording:" + digest(tokens(t)) for t in texts(record) if len(tokens(t)) >= 4)
    return result


class Groups:
    def __init__(self):
        self.parent, self.keys = [], {}

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def add(self, keys):
        i = len(self.parent)
        self.parent.append(i)
        for key in keys:
            if key in self.keys:
                a, b = self.find(i), self.find(self.keys[key])
                self.parent[a] = b
            self.keys[key] = i
        return i


class Frozen:
    def __init__(self, prompts):
        if not isinstance(prompts, list) or any(not isinstance(p, str) for p in prompts):
            raise ValueError("frozen exclusion file must be the complete string list")
        self.exact = {tokens(p) for p in prompts if len(tokens(p)) >= 2}
        self.grams = {t[i:i+13] for t in self.exact for i in range(len(t)-12)}
        self.fours = {t[i:i+4] for t in self.exact for i in range(len(t)-3)}

    def add(self, values):
        for value in values:
            t = tokens(value)
            if len(t) >= 2:
                self.exact.add(t)
            self.grams.update(t[i:i+13] for i in range(len(t)-12))
            self.fours.update(t[i:i+4] for i in range(len(t)-3))

    def hits(self, values, *, speech=False):
        result = set()
        for value in values:
            t = tokens(value)
            if t in self.exact:
                result.add("frozen_eval_exact_text")
            n = len(t)-12
            if n > 0 and sum(t[i:i+13] in self.grams for i in range(n))/n > .5:
                result.add("frozen_eval_13gram_near_duplicate")
            n = len(t)-3
            if speech and n > 0 and sum(t[i:i+4] in self.fours for i in range(n))/n > .5:
                result.add("frozen_eval_speech_4gram_overlap")
        return result


def pcm(path):
    with wave.open(str(path), "rb") as w:
        rate, channels, width, count = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        if w.getcomptype() != "NONE" or channels not in (1, 2) or width != 2 or rate not in (16000, 24000, 32000) or not 0 < count <= rate*52:
            raise ValueError("actual source must be bounded PCM16 mono/stereo at16/24/32kHz")
        data = w.readframes(count)
    if len(data) != count*channels*width or not any(data):
        raise ValueError("waveform sample coverage is incomplete or silent")
    return {"sample_rate": rate, "channels": channels, "samples": count,
            "duration_seconds": count/rate, "pcm_sha256": hashlib.sha256(data).hexdigest()}, data


def validate_tail_derivation(receipt, source, derived):
    """Verify explicit minimal25ms zero-tail without altering original PCM."""
    if receipt.get("schema") != "chimera.speech-h3-minimal-tail-padding.v1":
        raise ValueError("unsupported source-preserving padding receipt")
    a, raw = pcm(source)
    b, out = pcm(derived)
    hop = a["sample_rate"]//40
    extra = (-a["samples"]) % hop
    if (file_sha(source) != receipt.get("source_sha256") or file_sha(derived) != receipt.get("derived_sha256") or
        a["sample_rate"] != b["sample_rate"] or a["channels"] != b["channels"] or
        receipt.get("source_rate") != a["sample_rate"] or receipt.get("source_samples") != a["samples"] or
        receipt.get("derived_samples") != b["samples"] or receipt.get("tail_padding_samples") != extra or
        b["samples"] != a["samples"]+extra or receipt.get("trimmed_samples") != 0 or
        receipt.get("prefix_pcm_bitwise_equal") is not True or out[:len(raw)] != raw or any(out[len(raw):])):
        raise ValueError("padding is not the verified minimal zero-tail with identical PCM prefix")
    return {"original": a, "derived": b, "tail_padding_samples": extra,
            "original_asset_sha256": file_sha(source), "derived_asset_sha256": file_sha(derived),
            "original_duration_seconds": a["duration_seconds"], "derived_duration_seconds": b["duration_seconds"],
            "synthetic_tail_is_not_original_audio": True, "native_train_ready": False}


def procedural_candidates(paths, source_specs, asset_root):
    source = list(rows(paths["source"]))
    inventory = list(rows(paths["source_inventory"]))
    report = json.loads(paths["source_report"].read_text(encoding="utf-8"))
    preflight = json.loads(paths["source_preflight"].read_text(encoding="utf-8"))
    for name in ("source", "source_inventory"):
        seal = report["artifacts"][paths[name].name]
        if seal["sha256"] != source_specs[name]["sha256"] or seal["rows"] != (len(source) if name == "source" else len(inventory)):
            raise ValueError("source report does not seal the complete source inventory")
    if (report.get("schema") != "operator_media_seed_report.v1" or not str(report.get("generated_at", "")).startswith("2026-") or
        report.get("examples") != len(inventory) or report.get("cross_split_media_conflicts") != 0 or
        preflight.get("schema") != "chimera_recovery.audio_understanding_preflight.v1" or preflight.get("source_ready") is not True or
        preflight.get("source_sha256") != source_specs["source"]["sha256"] or
        preflight.get("source_report_sha256") != source_specs["source_report"]["sha256"]):
        raise ValueError("source/preflight identities are not verified")
    by_id, assets = {}, set()
    root = asset_root.resolve()
    for r in inventory:
        if r.get("id") in by_id or r.get("schema") != "recovery_corpus.v1" or r.get("split") not in SPLITS:
            raise ValueError("source inventory missing unique IDs/explicit splits")
        p = r.get("provenance", {})
        if (p.get("source_verified") is not True or p.get("source_year") != 2026 or p.get("split") != r["split"] or
            p.get("source_code_sha256") != report.get("source_code_sha256") or not p.get("group_id")):
            raise ValueError("source inventory provenance changed")
        for m in r.get("media", []):
            path = Path(m["path"])
            if not path.resolve().is_relative_to(root) or path.is_symlink() or file_sha(path) != m.get("sha256") or path.stat().st_size != m.get("bytes"):
                raise ValueError("actual media asset path/hash/bytes differs")
            assets.add(str(path.resolve()))
        by_id[r["id"]] = r
    if len(assets) != report.get("verified_assets"):
        raise ValueError("complete media asset traversal count differs")
    if any(r.get("id") not in by_id or digest(r) != digest(by_id[r["id"]]) for r in source) or len({r["id"] for r in source}) != len(source):
        raise ValueError("source subset is not exact original inventory")
    pf = {r["id"]: r for r in preflight["rows"]}
    candidates = []
    for r in source:
        if r.get("route") != "audio_understand":
            continue
        p, q = r["provenance"], r.get("quality", {})
        if (len(r["media"]) != 1 or r["media"][0].get("type") != "audio" or r["media"][0].get("role") != "input" or
            [m.get("role") for m in r["messages"]] != ["user", "assistant"] or
            r.get("supervision", {}).get("policy") != "assistant_only" or r["supervision"].get("message_mask") != [0,1] or
            q.get("groundtruth_verified") is not True or q.get("source_ready") is not True):
            raise ValueError("audio source supervision/input identity differs")
        m = r["media"][0]
        truth_path = Path(p["groundtruth_path"])
        if not truth_path.resolve().is_relative_to(root) or file_sha(truth_path) != p["groundtruth_sha256"]:
            raise ValueError("authored groundtruth bytes changed")
        answer = json.loads(r["messages"][-1]["content"])
        truth = json.loads(truth_path.read_text(encoding="utf-8"))["groundtruth"]
        if (set(answer) != {"contains_speech", "count", "events"} or answer.get("contains_speech") is not False or
            type(answer.get("count")) is not int or answer["count"] != len(answer["events"]) or
            any(truth.get(k) != v for k,v in answer.items()) or
            r["messages"][-1]["content"] in r["messages"][0]["content"]):
            raise ValueError("source reference is not actual non-speech groundtruth")
        measure, _ = pcm(Path(m["path"]))
        proof = pf.get(r["id"], {})
        if (proof.get("source_row_sha256") != digest(r) or proof.get("split") != r["split"] or
            proof.get("recovery_group_id") != p["group_id"] or proof.get("waveform_sha256") != m["sha256"] or
            proof.get("groundtruth_sha256") != p["groundtruth_sha256"] or proof.get("authored_answer_sha256") != digest(answer) or
            any(proof.get(k) != v for k,v in measure.items() if k != "pcm_sha256") or
            measure["sample_rate"] != m.get("sample_rate") or measure["duration_seconds"] != m.get("duration_seconds")):
            raise ValueError("actual source/preflight waveform/reference join differs")
        candidates.append({"record": r, "asset_path": m["path"], "asset_sha256": m["sha256"], "measure": measure,
                           "groundtruth_sha256": p["groundtruth_sha256"], "groundtruth_answer_sha256": digest(answer)})
    if len(pf) != len(candidates) or preflight.get("audio_rows") != len(candidates):
        raise ValueError("complete audio preflight/source join count differs")
    return candidates, inventory


def seal_admission(*, specs, candidates, inventory, output):
    """Traverse full normalized/prior/eval inventories and propagate exclusions."""
    if output.exists():
        raise ValueError("select a new admission output directory")
    required = {"normalized", "corpus_manifest", "eval_rows", "frozen_exclusions", "prior_admission"}
    if not required.issubset(specs):
        raise ValueError("global admission requires full normalized/prior/frozen inputs")
    paths = {k:pinned(v,k) for k,v in specs.items()}
    sources = {k:{"path": str(p.resolve()), "sha256": specs[k]["sha256"]} for k,p in paths.items()}
    corpus = json.loads(paths["corpus_manifest"].read_text(encoding="utf-8"))
    prior = json.loads(paths["prior_admission"].read_text(encoding="utf-8"))
    if (corpus.get("schema") != "recovery_corpus.manifest.v1" or
        corpus.get("outputs", {}).get("normalized", {}).get("sha256") != specs["normalized"]["sha256"] or
        specs["eval_rows"]["sha256"] not in {s["sha256"] for s in corpus.get("eval_sources", [])}):
        raise ValueError("global corpus/eval manifest source seals changed")
    if (prior.get("schema") != "recovery_media_admission.v1" or prior.get("connected_groups_verified") is not True or
        prior.get("eval_disjoint") is not True or digest({k:v for k,v in prior.items() if k != "manifest_sha256"}) != prior.get("manifest_sha256")):
        raise ValueError("prior connected admission seal is invalid")
    for name in ("normalized", "corpus_manifest", "eval_rows"):
        if prior["sources"][name]["sha256"] != specs[name]["sha256"]:
            raise ValueError("global prior/corpus source identities differ")
    # Verify original registry files, not just an opaque complete-traversal flag.
    for i, source in enumerate(corpus.get("sources", [])):
        original = pinned(source, "original_registry")
        sources["original_registry_"+str(i)] = {"path":str(original.resolve()), "sha256":source["sha256"]}
    if not corpus.get("sources"):
        raise ValueError("original registry source pin missing")
    for name in ("admission", "cross_route_training_exclusions"):
        ref = prior["outputs"][name]
        path = pinned(ref, "prior_"+name)
        alias = "prior_admission_rows" if name == "admission" else "prior_"+name
        paths[alias] = path
        sources[alias] = {"path":str(path.resolve()), "sha256":ref["sha256"]}
    frozen = Frozen(json.loads(paths["frozen_exclusions"].read_text(encoding="utf-8")))
    groups, nodes = Groups(), []
    counts = Counter()

    def add(keys, split, blockers, group=None, ident=None):
        index = groups.add(keys)
        nodes.append({"ix":index,"split":split,"blockers":set(blockers),"group":group,"id":ident})
        return index

    # Evaluation source identities connect even when text is abbreviated.
    for r in rows(paths["eval_rows"]):
        frozen.add(texts(r))
        add(identities(r), "evaluation", {"frozen_evaluation_source"}, ident=r.get("id"))
        counts["eval_rows"] += 1
    for r in rows(paths["normalized"]):
        p = r.get("provenance", {})
        if r.get("schema") != "recovery_corpus.v1" or r.get("split") not in SPLITS or not p.get("recovery_group_id"):
            raise ValueError("complete normalized traversal lacks explicit group/split")
        blockers = {b for b in r.get("quality", {}).get("blockers", []) if b.startswith("eval_") or b.startswith("frozen_carve_") and b != "frozen_carve_duplicate"}
        if r["split"] != "train":
            blockers.add("global_recovery_holdout")
            if not r.get("media"):
                frozen.add(texts(r))
        add(identities(r), r["split"], blockers, p["recovery_group_id"], r["id"])
        counts["normalized"] += 1
    for r in rows(paths["prior_admission_rows"]):
        keys = {"asset:"+r["asset_sha256"]} if SHA.fullmatch(str(r.get("asset_sha256", ""))) else set()
        if r.get("asset_path"):
            keys.add("asset-path:"+str(r["asset_path"]).replace("\\", "/"))
        if r.get("uid"):
            keys.add("uid:"+r["uid"])
        if r.get("rid"):
            keys.add("rid:"+str(r["rid"]))
        keys.update("group:"+g for g in r.get("connected_source_groups", []))
        blockers = set(r.get("blockers", [])) if r.get("source_train_eligible") is not True else set()
        # Partial teacher caches are operational, not source contamination.
        blockers.discard("teacher_cache_not_sealed_or_diagnostic")
        add(keys, r.get("split"), blockers, r.get("recovery_group_id"), r.get("item_id"))
        counts["prior_admission"] += 1
    for r in rows(paths["prior_cross_route_training_exclusions"]):
        keys = {"uid:"+r["id"], "group:"+r["source_group_id"]}
        if r.get("rid"):
            keys.add("rid:"+str(r["rid"]))
        add(keys, "excluded", r["blockers"], r["source_group_id"], r["id"])
        counts["prior_cross_route_training_exclusions"] += 1
    for name in ("admission", "cross_route_training_exclusions"):
        if counts["prior_"+name] != prior["outputs"][name]["rows"]:
            raise ValueError("complete prior admission/exclusion traversal count differs")
    for r in inventory:
        p = r["provenance"]
        blockers = frozen.hits(texts(r))
        if r["split"] != "train":
            blockers.add("source_family_holdout")
        add(identities(r), r["split"], blockers, p["group_id"], r["id"])
        counts["source_inventory"] += 1
    for candidate in candidates:
        r = candidate["record"]
        p = r["provenance"]
        keys = identities(r)
        # Speech target wording is the actual utterance identity across voices.
        speech = r["route"] == "speech_understand"
        if speech:
            keys.add("wording:"+digest(tokens(r["messages"][-1]["content"])))
        blockers = set(candidate.get("blockers", [])) | frozen.hits(texts(r), speech=speech)
        if r["split"] != "train":
            blockers.add("source_family_holdout")
        candidate["ix"] = add(keys,r["split"],blockers,p["group_id"],r["id"])
    root_blocks, root_splits, root_names = defaultdict(set), defaultdict(set), defaultdict(set)
    for node in nodes:
        root = groups.find(node["ix"])
        root_blocks[root].update(node["blockers"])
        if node["split"]:
            root_splits[root].add(node["split"])
        if node["group"]:
            root_names[root].add(node["group"])
    out = []
    for candidate in candidates:
        r = candidate["record"]
        root = groups.find(candidate["ix"])
        blockers = set(root_blocks[root])
        if root_splits[root] != {"train"}:
            blockers.add("connected_group_not_train_exclusive")
        p = r["provenance"]
        row = {k:v for k,v in candidate.items() if k not in {"record", "ix", "blockers"}}
        row.update(schema=SCHEMA,id=r["id"],item_id=r["id"],route=r["route"],split=r["split"],
                   recovery_group_id=digest(sorted(root_names[root])),connected_source_groups=sorted(root_names[root]),
                   source_group_id=p["group_id"],source_row_sha256=digest(r),
                   authored_answer_sha256=digest(r["messages"][-1]["content"]),
                   prompt_sha256=hashlib.sha256(r["messages"][0]["content"].encode()).hexdigest(),
                   source_train_eligible=not blockers,train_admitted=not blockers,blockers=sorted(blockers),
                   native_train_ready=False,encoder_status="per_source_actual_H3_proof_required",source_pins=sources)
        out.append(row)
    output.mkdir(parents=True)
    outputs = {}
    for name, values in (("admission",out),("train_admitted",[r for r in out if r["train_admitted"]])):
        path = output/(name+".jsonl")
        path.write_bytes(b"".join(canonical(v)+b"\n" for v in values))
        outputs[name] = {"path":str(path.resolve()),"sha256":file_sha(path),"rows":len(values)}
    manifest = {"schema":SCHEMA,"created_utc":datetime.now(timezone.utc).isoformat(),"sealed":True,
                "complete_source_traversal":True,"frozen_exclusions_verified":True,"connected_groups_verified":True,
                "sources":sources,"code_sha256":file_sha(Path(__file__)),"traversal_counts":dict(counts),
                "rows":len(out),"train_admitted":sum(r["train_admitted"] for r in out),
                "split_counts":dict(Counter(r["split"] for r in out)),
                "blocker_counts":dict(Counter(b for r in out for b in r["blockers"])),"outputs":outputs,
                "native_train_ready":False,"training_started":False,
                "policy":"connected source/family/asset/full-task/utterance groups; all held/eval exclusions propagate across routes",
                "limitations":["Source eligibility only; actual per-source clean H3 tensors, dialogue masks, amendment and native batch proof remain required.",
                               "Lexical frozen13gram and speech4gram screens cannot prove exhaustive semantic separation.",
                               "Shared generic media instructions do not equate distinct observations; their actual media and source family are identities."]}
    manifest["manifest_sha256"] = digest(manifest)
    (output/"manifest.json").write_bytes(canonical(manifest)+b"\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins",type=Path,required=True,help="exact source {name:{path,sha256}} JSON")
    parser.add_argument("--asset-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(argv)
    specs = json.loads(args.pins.read_text(encoding="utf-8"))
    paths = {k:pinned(v,k) for k,v in specs.items()}
    candidates, inventory = procedural_candidates(paths,specs,args.asset_root)
    result = seal_admission(specs=specs,candidates=candidates,inventory=inventory,output=args.output)
    print(json.dumps({k:result[k] for k in ("rows","train_admitted","split_counts","blocker_counts","outputs","manifest_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
