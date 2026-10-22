"""ingest.foldrec -- the complete Neural Forge folding record (inventory/<date>-foldrec/*.json, schema foldrec-v1)
-> experiment, fold_arm, fold_component, fold_behaviour, fold_diagnostic, fold_source, experiment_alias,
   experiment_link, fold_finding (+ _experiment, _source), technique (+ 0007 extension columns), directive,
   fold_shard, fold_coverage.

Every *.json in the shard directory (--inventory, default inventory/2026-10-22-foldrec in this repo) is one
shard written by one extraction agent.  Nothing is paraphrased: text columns hold the shard's text, numeric
columns are parsed copies (NULL when the text is not one number), and every raw shard record is kept in
fold_arm.record.

Validation.  A shard that is not valid JSON, is not schema foldrec-v1, or repeats another shard's name is a
hard error (the ingest refuses; --dry-run lists it).  A record without a key is skipped.  An enum value outside
the foldrec-v1 list is a soft problem: the column gets NULL ('other' for a component kind), the raw value
stays in fold_arm.record, and the problem is noted on the batch.

Dedup across shards.  Records are the same arm only when they carry the same key (the format makes the key
the source's own stable id).  An alias never merges two records: the extractors list a base id as the alias
of each separate attempt ('C04.T1.M16.L00#wave2' -> 'C04.T1.M16.L00'), and one row per attempt is the rule.
Merging fills missing fields from later members (shard name order).  When members disagree, every variant is
kept in fold_arm.conflicts with its shard, key and sources; a disagreeing number (ratio_body, ratio_net, axes)
or verdict_class is left NULL in its column, never picked.  evidence_level takes the strongest stated level.
Lists (components, behaviour, diagnostics, sources) are unioned; identical entries collapse, each row says
which shard(s) gave it.

Linking.
  * An arm whose canonical key is already an experiment written by another ingester (a foldhist entry, a
    proof_eval experiment) attaches its fold_arm and child rows to that experiment; the experiment's own
    columns stay as that ingester wrote them.  Otherwise foldrec owns the experiment row.
  * Aliases -> experiment_alias.  An alias that is another arm's key links the two arms: '<A> rerun_of <B>'
    when A's key is B's key + '#suffix' or '@suffix' (a wave or a rerun), else '<A> same_as <B>'.  An alias
    that is the key of an experiment outside this record (a foldhist id) links it to the arm: '<E> same_as
    <arm>' when one arm carries the alias, '<E> summarizes <arm>' when several do.
  * Technique names (components and techniques[]) resolve to technique rows by name, case and whitespace
    insensitive; unknown names become new rows (family 'fold').  foldrec owns the 0007 columns code_location,
    first_date, first_date_text, status_text and detail (all shard variants are in detail.foldrec).
  * child.ref / parent.name ledger ids (OCnn-..., PARENT-...) resolve to model rows; a source path or child.ref
    equal to a run's receipt_path or run_dir links that run (fold_arm.run_ids; experiment.run_id when one).
  * Findings keep their key; two shards using one key for different statements both get '<shard>:<key>'.
    experiment_keys resolve through arm keys, arm aliases and existing experiment keys; unresolved ones stay
    with experiment_id NULL and fill on a re-run.
  * Directives go into directive only after the quote is found in its source (agent_memory_events user prompt
    via read-only SQL, or the cited file) and only when no directive already contains the exact quote.  Keys
    follow ingest/directives.py ('ev:<id>:<sha8>' / 'doc:<sha8 path>:<line>:<sha8>'), so the two ingesters
    never write the same words twice.  A quote with '...' is checked fragment by fragment, in order, and stored
    as the exact source spans joined by ' ... '.  When a cited event does not hold the words but a cited file
    does, the file is the source and the summary says so.  Unverifiable quotes are skipped and noted.

Re-runnable: rows are keyed on natural keys, child rows on (owner, ordinal) with surplus ordinals deleted, so
an unchanged shard set writes nothing.  Rows of arms that disappeared from the shards are reported, not deleted.

    python -m ingest.foldrec [--inventory DIR] [--dry-run]
    omni_lab.py ingest foldrec [--inventory DIR] [--dry-run]
--dry-run parses, validates, merges and resolves against the database read-only (omni_lab_ro); it writes no
row, not even an ingest_batch.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from . import common as C
from . import knowledge_common as K
from .common import db
from .directives import AGENT_CHANNEL, anchor, date_from_doc
from .foldhist import LEDGER_ID_RE

SOURCE = "foldrec"
SCHEMA = "foldrec-v1"
DEFAULT_DIR = K.REPO / "inventory" / "2026-10-22-foldrec"

ENUMS = {
    "program": ("early-surgery", "neural-forge", "residual-attention", "lfm2-dynamic", "cpcf", "sparse-semantic",
                "cross-width", "cross-family", "kernel", "stage1", "stage2", "matrix-campaign", "chimera-omni-toy",
                "chimera-omni", "k3", "media", "density", "other"),
    "owner": ("claude", "codex", "both", "unknown"),
    "exists_now": ("yes", "deleted", "unknown"),
    "fit": ("closed_form", "ridge", "coreset", "low_rank", "eigen", "pca", "omp", "gradient", "kd", "on_policy",
            "search", "other"),
    "target": ("activation", "output_write", "top1", "topk", "kl", "multi_token", "endpoint_residual", "energy",
               "attention_pattern", "other"),
    "training_used": ("none", "recovery_after_fold", "gradient_fit_of_fold", "unknown"),
    "verdict_class": ("works", "partial", "fails", "mechanics_only", "design_only", "unmeasured", "closed_path",
                      "superseded"),
    "evidence_level": ("receipt", "report", "notes", "memory", "transcript", "commit_msg"),   # strongest first
    "scorer": ("cyril", "claude", "codex", "machine", "mixed", "none"),
    "kind": ("hidden_width", "mlp_width", "depth_excision", "depth_merge", "attn_q", "attn_k", "attn_v", "attn_o",
             "attn_qkvo", "heads", "head_dim", "conv", "embed", "vocab", "lm_head", "experts_to_dense", "cross_width",
             "cross_family", "kind_translation", "modality_unit", "whole_model", "other"),
    "finding_status": ("proven", "supported", "refuted", "closed_path", "open", "superseded"),
}
DIAG_NAMES = ("tf_top1", "top5", "top10", "topk_overlap", "free_run_match", "lcp", "exact_h_token", "kl", "nll", "ppl",
              "retained_energy", "relL2", "cos", "mse", "db", "rank", "other")
PROGRAM_PROJECT = {"chimera-omni": "chimera-omni", "chimera-omni-toy": "chimera-omni", "k3": "k3", "media": "media"}

# arm scalar fields: (column, path into the record, kind).  kind 'num' and 'verdict' go NULL on disagreement.
ARM_FIELDS = (
    ("title", ("title",), "text"), ("date", ("date",), "text"), ("date_text", ("date_text",), "text"),
    ("program", ("program",), "enum:program"), ("wave", ("wave",), "text"), ("owner", ("owner",), "enum:owner"),
    ("parent_name", ("parent", "name"), "text"), ("parent_params", ("parent", "params"), "text"),
    ("child_ref", ("child", "ref"), "text"), ("child_params", ("child", "params"), "text"),
    ("child_exists_now", ("child", "exists_now"), "enum:exists_now"),
    ("axes", ("axes",), "int"), ("ratio_body", ("ratio_body",), "num"), ("ratio_net", ("ratio_net",), "num"),
    ("ratio_text", ("ratio_text",), "text"),
    ("objective_fit", ("objective", "fit"), "enum:fit"), ("objective_target", ("objective", "target"), "enum:target"),
    ("objective_k", ("objective", "k"), "text"), ("objective_horizon", ("objective", "horizon"), "text"),
    ("objective_text", ("objective", "text"), "text"),
    ("training_used", ("training_used",), "enum:training_used"), ("training_text", ("training_text",), "text"),
    ("verdict_class", ("verdict_class",), "verdict"), ("verdict_text", ("verdict_text",), "text"),
    ("cyril", ("cyril",), "text"), ("evidence_level", ("evidence_level",), "evidence"),
)
NUM_RE = re.compile(r"^\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+|\d*\.\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(%|x|×|dB|db)?\s*$")
FRAC_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")
EV_RE = re.compile(r"^(?:ev[:#=\s]*|agent_memory_events\.id=)?(\d+)\b", re.I)


# ---------------------------------------------------------------------------
# value helpers (pure)
# ---------------------------------------------------------------------------

def jsonable(value: Any) -> Any:
    """Decimal -> int/float so a value can go into JSONB (the text form stays in the text columns)."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() and "." not in str(value) and "E" not in str(value) \
            else float(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def as_text(value: Any) -> str | None:
    """A scalar as text exactly as the shard wrote it (JSON numbers keep their digits); None stays None."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, Decimal)):
        return str(value)
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)


def as_number(value: Any) -> Decimal | None:
    """A JSON number, or a string that is exactly one plain number; else None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, Decimal)):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(repr(value))
    if isinstance(value, str):
        m = NUM_RE.match(value)
        if m and not m.group(2):
            try:
                return Decimal(m.group(1).replace(",", ""))
            except InvalidOperation:
                return None
    return None


def as_int(value: Any) -> int | None:
    n = as_number(value)
    return int(n) if n is not None and n == n.to_integral_value() else None


def parse_value(value: Any) -> tuple[float | None, str | None]:
    """(value_num, value_unit) for a diagnostic value: one number ('0.91', '1,024', '31.2%', '-12 dB', '4x') or a
    fraction 'a/b' (-> a/b, unit 'fraction').  Anything else -> (None, None)."""
    if isinstance(value, bool) or value is None:
        return None, None
    if isinstance(value, (int, Decimal, float)):
        return float(value), None
    text = str(value)
    m = FRAC_RE.match(text)
    if m:
        den = float(m.group(2))
        return (float(m.group(1)) / den if den else None), "fraction"
    m = NUM_RE.match(text)
    if m:
        try:
            num = float(m.group(1).replace(",", ""))
        except ValueError:
            return None, None
        unit = m.group(2)
        return num, ({"×": "x", "db": "dB"}.get(unit, unit) if unit else None)
    return None, None


def parse_score(score: Any) -> tuple[Decimal | None, Decimal | None]:
    """'12/15' -> (12, 15); a bare number -> (n, None); else (None, None)."""
    if score is None or isinstance(score, bool):
        return None, None
    if isinstance(score, (int, Decimal)):
        return Decimal(score), None
    m = FRAC_RE.match(str(score))
    if m:
        return Decimal(m.group(1)), Decimal(m.group(2))
    n = as_number(score)
    return (n, None) if n is not None else (None, None)


def iso_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()[:10]) if re.match(r"^\d{4}-\d\d-\d\d$", value.strip()) else None
    except ValueError:
        return None


def norm_name(name: str) -> str:
    return " ".join(str(name).split()).casefold()


def dig(record: dict, path: tuple[str, ...]) -> Any:
    cur: Any = record
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (isinstance(value, (list, dict)) and not value)


def ev_id(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, Decimal)):
        return int(value)
    m = EV_RE.match(str(value).strip())
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# loading + validation (pure)
# ---------------------------------------------------------------------------

class Shard:
    def __init__(self, path: Path, doc: dict):
        self.path = path
        self.doc = doc
        self.name = doc["shard"]


class Plan:
    """Everything one ingest would write, computed from the shard files alone (no database)."""

    def __init__(self):
        self.shards: list[Shard] = []
        self.errors: list[str] = []        # hard: the ingest refuses
        self.problems: list[str] = []      # soft: coerced, noted
        self.skipped: list[str] = []
        self.arms: list[dict] = []
        self.findings: list[dict] = []
        self.techniques: dict[str, dict] = {}
        self.directives: list[tuple[Shard, dict]] = []
        self.alias_only: dict[str, list[str]] = {}
        self.records_seen = 0


def load_shards(directory: Path, plan: Plan) -> None:
    if not directory.is_dir():
        plan.errors.append(f"shard directory not found: {directory}")
        return
    names: dict[str, Path] = {}
    for p in sorted(directory.glob("*.json")):
        try:
            doc = json.loads(p.read_bytes().decode("utf-8"), parse_float=Decimal)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            plan.errors.append(f"{p.name}: not readable JSON ({type(exc).__name__}: {str(exc)[:120]})")
            continue
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            plan.errors.append(f"{p.name}: not schema {SCHEMA}")
            continue
        if not isinstance(doc.get("shard"), str) or not doc["shard"].strip():
            plan.errors.append(f"{p.name}: no shard name")
            continue
        bad = [k for k in ("sources_read", "sources_not_read", "experiments", "findings", "techniques", "directives")
               if not isinstance(doc.get(k, []), list)]
        if bad:
            plan.errors.append(f"{p.name}: {', '.join(bad)} must be lists")
            continue
        if doc["shard"] in names:
            plan.errors.append(f"{p.name}: shard name {doc['shard']!r} already used by {names[doc['shard']].name}")
            continue
        if p.stem != doc["shard"]:
            plan.problems.append(f"{p.name}: file name differs from shard name {doc['shard']!r}")
        names[doc["shard"]] = p
        plan.shards.append(Shard(p, doc))


def check_enum(plan: Plan, where: str, field: str, value: Any, enum: str) -> str | None:
    if is_empty(value):
        return None
    if isinstance(value, str) and value in ENUMS[enum]:
        return value
    plan.problems.append(f"{where}: {field}={as_text(value)!r} is not a foldrec-v1 value (column NULL, kept in record)")
    return None


def clean_list(plan: Plan, where: str, field: str, value: Any) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        plan.problems.append(f"{where}: {field} is not a list (ignored)")
        return []
    out = [v for v in value if isinstance(v, dict)]
    if len(out) != len(value):
        plan.problems.append(f"{where}: {len(value) - len(out)} non-object {field} entries ignored")
    return out


def clean_aliases(plan: Plan, where: str, value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        plan.problems.append(f"{where}: aliases is not a list (ignored)")
        return []
    out = []
    for a in value:
        if isinstance(a, str) and a.strip():
            if a.strip() not in out:
                out.append(a.strip())
        else:
            plan.problems.append(f"{where}: non-text alias {as_text(a)!r} ignored")
    return out


# ---------------------------------------------------------------------------
# merging (pure)
# ---------------------------------------------------------------------------

def field_value(plan: Plan, where: str, column: str, kind: str, raw: Any) -> Any:
    if is_empty(raw):
        return None
    if kind == "text":
        return as_text(raw)
    if kind == "num":
        n = as_number(raw)
        if n is None:
            plan.problems.append(f"{where}: {column}={as_text(raw)!r} is not a number (column NULL, kept in record)")
        return n
    if kind == "int":
        n = as_int(raw)
        if n is None:
            plan.problems.append(f"{where}: {column}={as_text(raw)!r} is not an integer (column NULL, kept in record)")
        return n
    enum = {"verdict": "verdict_class", "evidence": "evidence_level"}.get(kind) or kind.split(":", 1)[1]
    return check_enum(plan, where, column, raw, enum)


def merge_arm(plan: Plan, members: list[dict]) -> dict:
    """One arm from its member records (members sorted by shard name, then record index)."""
    key = members[0]["key"]
    arm: dict[str, Any] = {"key": key, "members": members, "conflicts": {},
                           "shards": sorted({m["shard"].name for m in members}),
                           "member_keys": sorted({m["key"] for m in members})}
    aliases: list[str] = []
    for m in members:
        for a in m["aliases"]:
            if a != key and a not in aliases:
                aliases.append(a)
    arm["aliases"] = aliases
    for column, path, kind in ARM_FIELDS:
        variants: list[tuple[Any, dict]] = []
        for m in members:
            v = field_value(plan, m["where"], column, kind, dig(m["rec"], path))
            if v is not None:
                variants.append((v, m))
        distinct: list[Any] = []
        for v, _m in variants:
            if v not in distinct:
                distinct.append(v)
        if len(distinct) > 1:
            arm["conflicts"][column] = [{"value": jsonable(v), "shard": m["shard"].name, "key": m["key"],
                                         "sources": jsonable(m["rec"].get("sources") or [])} for v, m in variants]
        if not distinct:
            arm[column] = None
        elif kind == "evidence":
            arm[column] = min(distinct, key=ENUMS["evidence_level"].index)
        elif len(distinct) > 1 and kind in ("num", "int", "verdict"):
            arm[column] = None
        else:
            arm[column] = distinct[0]
    arm["date_value"] = iso_date(arm["date"])
    if arm["date"] is not None and arm["date_value"] is None:
        plan.problems.append(f"{key}: date {arm['date']!r} is not YYYY-MM-DD (date NULL, date_text/record keep it)")

    def union(field: str, build) -> list[dict]:
        out: list[dict] = []
        index: dict[str, dict] = {}
        for m in members:
            for entry in clean_list(plan, m["where"], field, m["rec"].get(field)):
                row = build(m, entry)
                if row is None:
                    continue
                ident = db.canonical_json(jsonable({k: v for k, v in row.items() if k != "shards"}))
                if ident in index:
                    if m["shard"].name not in index[ident]["shards"]:
                        index[ident]["shards"].append(m["shard"].name)
                    continue
                row["shards"] = [m["shard"].name]
                index[ident] = row
                out.append(row)
        return out

    def component(m, c):
        kind = c.get("kind")
        if kind not in ENUMS["kind"]:
            plan.problems.append(f"{m['where']}: component kind={as_text(kind)!r} is not a foldrec-v1 value (stored 'other')")
            kind = "other"
        tech = as_text(c.get("technique"))
        tech = tech.strip() if tech and tech.strip() else None
        return {"kind": kind, "layers": as_text(c.get("layers")), "technique_text": tech,
                "ratio": field_value(plan, m["where"], "component ratio", "num", c.get("ratio")),
                "ratio_text": as_text(c.get("ratio_text")), "rank_or_k": as_text(c.get("rank_or_k")),
                "notes": as_text(c.get("notes"))}

    def behaviour(m, b):
        fields = ("suite", "score", "useful", "partial", "miss", "total", "column_b", "parent_score", "text")
        if all(is_empty(b.get(f)) for f in fields) and b.get("scorer") in (None, "none"):
            return None                                   # 'nothing measured': kept only in fold_arm.record
        value, den = parse_score(b.get("score"))
        row = {"suite": as_text(b.get("suite")),
               "scorer": check_enum(plan, m["where"], "behaviour.scorer", b.get("scorer"), "scorer"),
               "score": as_text(b.get("score")), "score_value": value, "score_denominator": den,
               "column_b": as_text(b.get("column_b")), "parent_score": as_text(b.get("parent_score")),
               "text": as_text(b.get("text"))}
        for f in ("useful", "partial", "miss", "total"):
            row[f] = as_int(b.get(f))
            if row[f] is None and not is_empty(b.get(f)):
                plan.problems.append(f"{m['where']}: behaviour.{f}={as_text(b.get(f))!r} is not an integer")
        return row

    def diagnostic(m, d):
        if is_empty(d.get("value")):
            plan.problems.append(f"{m['where']}: diagnostic {as_text(d.get('name'))!r} without a value ignored")
            return None
        name = as_text(d.get("name")) or "other"
        num, unit = parse_value(d.get("value"))
        return {"name": name, "value_text": as_text(d.get("value")), "value_num": num, "value_unit": unit,
                "horizon": as_text(d.get("horizon")), "denominator": as_text(d.get("denominator")),
                "split": as_text(d.get("split")), "note": as_text(d.get("note"))}

    def source(m, s):
        row = {"path": as_text(s.get("path")), "lines": as_text(s.get("lines")), "commit_ref": as_text(s.get("commit")),
               "ev": as_text(s.get("ev")), "ev_id": ev_id(s.get("ev")), "quote": as_text(s.get("quote"))}
        if all(is_empty(v) for v in row.values()):
            return None
        return row

    arm["components"] = union("components", component)
    arm["behaviour"] = union("behaviour", behaviour)       # one object per record; distinct ones all kept
    arm["diagnostics"] = union("diagnostics", diagnostic)
    arm["sources"] = union("sources", source)
    if not arm["sources"]:
        plan.problems.append(f"{key}: no sources (every record must cite path + lines, a commit or an event)")
    arm["record"] = [{"shard": m["shard"].name, "index": m["index"], "record": jsonable(m["rec"])} for m in members]
    return arm


def build_plan(directory: Path) -> Plan:
    plan = Plan()
    load_shards(directory, plan)
    records: list[dict] = []
    for sh in plan.shards:
        for i, rec in enumerate(sh.doc.get("experiments") or []):
            plan.records_seen += 1
            where = f"{sh.name}#{i}"
            if not isinstance(rec, dict) or not isinstance(rec.get("key"), str) or not rec["key"].strip():
                plan.skipped.append(f"{where}: experiment without a key")
                continue
            records.append({"shard": sh, "index": i, "rec": rec, "key": rec["key"].strip(),
                            "where": f"{sh.name}:{rec['key'].strip()}",
                            "aliases": clean_aliases(plan, where, rec.get("aliases"))})
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["key"]].append(r)
    for members in groups.values():
        members.sort(key=lambda m: (m["shard"].name, m["index"]))
        plan.arms.append(merge_arm(plan, members))
    plan.arms.sort(key=lambda a: a["key"])
    keys = {a["key"] for a in plan.arms}
    holders: dict[str, list[str]] = defaultdict(list)
    for a in plan.arms:
        for al in a["aliases"]:
            if al not in keys:
                holders[al].append(a["key"])
    plan.alias_only = {al: ks for al, ks in holders.items() if len(ks) > 1}
    build_findings(plan)
    build_techniques(plan)
    for sh in plan.shards:
        for i, d in enumerate(sh.doc.get("directives") or []):
            plan.records_seen += 1
            if not isinstance(d, dict):
                plan.skipped.append(f"{sh.name}#directive{i}: not an object")
                continue
            plan.directives.append((sh, d))
    return plan


def build_findings(plan: Plan) -> None:
    raw: list[tuple[Shard, int, dict]] = []
    for sh in plan.shards:
        for i, f in enumerate(sh.doc.get("findings") or []):
            plan.records_seen += 1
            if not isinstance(f, dict) or is_empty(f.get("key")) or is_empty(f.get("statement")):
                plan.skipped.append(f"{sh.name}#finding{i}: finding without key or statement")
                continue
            raw.append((sh, i, f))
    by_key: dict[str, list[tuple[Shard, int, dict]]] = defaultdict(list)
    for item in raw:
        by_key[as_text(item[2]["key"]).strip()].append(item)
    out: dict[str, dict] = {}
    for key, items in by_key.items():
        statements = {as_text(f["statement"]) for _s, _i, f in items}
        if len(statements) == 1:
            groups = [(key, items)]
        else:
            plan.problems.append(f"finding key {key!r} names different statements in {len(items)} records; "
                                 f"stored as '<shard>:{key}'")
            per: dict[str, list] = defaultdict(list)
            for it in items:
                per[f"{it[0].name}:{key}"].append(it)
            groups = list(per.items())
        for fkey, its in groups:
            its.sort(key=lambda t: (t[0].name, t[1]))
            fkey = fkey if fkey not in out else f"{fkey}#{its[0][1]}"
            first = its[0][2]
            rec = {"key": fkey, "statement": as_text(first["statement"]), "conflicts": {},
                   "shards": sorted({s.name for s, _i, _f in its}), "experiment_keys": [], "sources": []}
            for field in ("status", "scope"):
                vals = []
                for s, _i, f in its:
                    v = check_enum(plan, f"{s.name}:{fkey}", "status", f.get(field), "finding_status") \
                        if field == "status" else as_text(f.get(field))
                    if v is not None and v not in vals:
                        vals.append(v)
                if len(vals) > 1:
                    rec["conflicts"][field] = [{"value": jsonable(f.get(field)), "shard": s.name} for s, _i, f in its]
                rec[field] = vals[0] if len(vals) == 1 or field == "scope" and vals else None
            seen_src: dict[str, dict] = {}
            for s, _i, f in its:
                for k in f.get("experiment_keys") or []:
                    if isinstance(k, str) and k.strip() and k.strip() not in rec["experiment_keys"]:
                        rec["experiment_keys"].append(k.strip())
                for src in clean_list(plan, f"{s.name}:{fkey}", "sources", f.get("sources")):
                    row = {"path": as_text(src.get("path")), "lines": as_text(src.get("lines")),
                           "commit_ref": as_text(src.get("commit")), "ev": as_text(src.get("ev")),
                           "quote": as_text(src.get("quote"))}
                    ident = db.canonical_json(row)
                    if ident in seen_src:
                        if s.name not in seen_src[ident]["shards"]:
                            seen_src[ident]["shards"].append(s.name)
                        continue
                    row["shards"] = [s.name]
                    seen_src[ident] = row
                    rec["sources"].append(row)
            out[fkey] = rec
    plan.findings = [out[k] for k in sorted(out)]


def build_techniques(plan: Plan) -> None:
    techs: dict[str, dict] = {}

    def get(name: str) -> dict:
        n = norm_name(name)
        if n not in techs:
            techs[n] = {"name": " ".join(name.split()), "entries": [], "arms": []}
        return techs[n]

    for sh in plan.shards:
        for i, t in enumerate(sh.doc.get("techniques") or []):
            plan.records_seen += 1
            if not isinstance(t, dict) or is_empty(t.get("name")):
                plan.skipped.append(f"{sh.name}#technique{i}: technique without a name")
                continue
            get(as_text(t["name"]))["entries"].append((sh.name, t))
    for arm in plan.arms:
        for c in arm["components"]:
            if c["technique_text"]:
                t = get(c["technique_text"])
                if arm["key"] not in t["arms"]:
                    t["arms"].append(arm["key"])
    for t in techs.values():
        t["entries"].sort(key=lambda e: e[0])
        cols: dict[str, Any] = {}
        for col, field in (("code_location", "code"), ("first_date_text", "first_date"), ("status_text", "status_text"),
                           ("description", "description"), ("family_text", "family")):
            vals = []
            for _s, e in t["entries"]:
                v = as_text(e.get(field))
                if v is not None and v.strip() and v not in vals:
                    vals.append(v)
            cols[col] = vals[0] if vals else None
            if len(vals) > 1:
                cols.setdefault("_conflicting", []).append(col)
        cols["first_date"] = iso_date(cols["first_date_text"]) if cols["first_date_text"] else None
        t["cols"] = cols
    plan.techniques = techs


# ---------------------------------------------------------------------------
# database lookups (read-only)
# ---------------------------------------------------------------------------

def existing_experiments(conn) -> dict[str, tuple[int, str | None]]:
    """key -> (experiment id, ingest_batch.source of the row's writer)."""
    with conn.cursor() as cur:
        cur.execute("SELECT e.key, e.id, b.source FROM experiment e LEFT JOIN ingest_batch b ON b.id = e.ingest_batch_id")
        return {k: (i, s) for k, i, s in cur.fetchall()}


def existing_techniques(conn) -> dict[str, tuple[int, str]]:
    """normalised name -> (id, name); an exact-case name wins over a later case variant."""
    out: dict[str, tuple[int, str]] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT id, name FROM technique ORDER BY id")
        for i, name in cur.fetchall():
            out.setdefault(norm_name(name), (i, name))
    return out


def directive_present(conn, texts: list[str]) -> str | None:
    """directive_key of a directive NOT written by foldrec whose verbatim already contains one of the texts exactly.
    (foldrec's own rows are left out so a re-run sees the same answer as the first run.)"""
    with conn.cursor() as cur:
        for t in texts:
            if not t:
                continue
            cur.execute("SELECT d.directive_key FROM directive d LEFT JOIN ingest_batch b ON b.id = d.ingest_batch_id "
                        "WHERE strpos(d.verbatim, %s) > 0 AND b.source IS DISTINCT FROM %s ORDER BY d.id LIMIT 1",
                        (t, SOURCE))
            row = cur.fetchone()
            if row:
                return row[0]
    return None


class EventReader:
    """agent_memory_events rows by id, read-only (admin profile, a READ ONLY session); None = not readable here."""

    def __init__(self, ids: list[int], notes: list[str]):
        self.rows: dict[int, tuple] | None = None
        if not ids:
            self.rows = {}
            return
        try:
            mem = db.connect("admin", dbname="agent_memory", app="ingest-foldrec-readonly")
        except Exception as exc:                       # no credential / no route
            notes.append(f"agent_memory not readable ({type(exc).__name__}); event-sourced directives skipped")
            return
        try:
            mem.set_session(readonly=True)
            with mem.cursor() as cur:
                cur.execute("SELECT id, agent, event_type, created_at, content FROM agent_memory_events "
                            "WHERE id = ANY(%s)", (ids,))
                self.rows = {r[0]: r for r in cur.fetchall()}
            mem.rollback()
        finally:
            mem.close()


ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|…)\s*")


def anchor_quote(quote: str, text: str | None) -> str | None:
    """The exact source span(s) a quote stands for.  A quote with '...' / an ellipsis character is matched
    fragment by fragment, in order; the result is the exact span of each fragment joined by ' ... '."""
    if not text:
        return None
    parts = [p for p in ELLIPSIS_RE.split(quote) if p.strip()]
    if not parts:
        return None
    if parts == [quote]:                                  # no ellipsis
        return anchor(quote, text)
    spans, pos = [], 0
    for part in parts:
        m = re.compile(r"\s+".join(re.escape(tok) for tok in part.split())).search(text, pos)
        if m is None:
            return None
        spans.append(text[m.start():m.end()])
        pos = m.end()
    return " ... ".join(spans)


def resolve_directive(sh: Shard, d: dict, events: EventReader) -> tuple[dict | None, str]:
    """(directive row values, state).  state: verified | not-found | unreadable | no-source | not-a-prompt | ...
    An event-sourced quote is checked in the event (a user prompt); when the event cannot confirm it and the
    record also cites a file, the file is checked instead and the row says why."""
    quote = (as_text(d.get("quote")) or "").strip()
    if not quote:
        return None, "empty-quote"
    src = d.get("source") if isinstance(d.get("source"), dict) else {}
    qsha = K.sha_text(quote, 8)
    eid = ev_id(src.get("ev"))
    topic = as_text(d.get("topic"))
    meta = {"_quote": quote, "_shard": sh.name, "_check": as_text(d.get("speaker_check")) or "not stated",
            "_date_note": None}
    path = as_text(src.get("path"))
    ev_state = None
    if eid is not None:
        row = None if events.rows is None else events.rows.get(eid)
        if events.rows is None:
            ev_state = "unreadable"
        elif row is None:
            ev_state = "no-such-event"
        elif row[2] != "user_prompt":
            ev_state = "not-a-prompt"
        elif AGENT_CHANNEL.get(str(row[1]).lower().replace(" ", "")) is None:
            ev_state = "unknown-agent"
        else:
            _id, agent, event_type, created_at, content = row
            channel = AGENT_CHANNEL[str(agent).lower().replace(" ", "")]
            span = anchor_quote(quote, content)
            if span is not None:
                return {"directive_key": f"ev:{eid}:{qsha}", "said_at": created_at, "said_at_verified": channel != "dot",
                        "author": "cyril", "channel": channel, "source_event_id": eid, "verbatim": span,
                        "topics": [topic] if topic else [],
                        "src_path": f"agent_memory_events.id={eid} (event_type={event_type}, agent={agent})",
                        **meta}, "verified"
            ev_state = "not-found"
        if not path:
            return None, ev_state
        meta["_check"] += f" [event {eid}: {ev_state}; verified in the cited file instead]"
    if path:
        line = as_int(src.get("line")) or 0
        text = C.read_text(path)
        if not text:
            return None, "unreadable"
        span = anchor_quote(quote, text)
        if span is None:
            return None, "not-found"
        said = iso_date(as_text(d.get("date")) or "")
        where = "shard date"
        if said is None:
            found = date_from_doc(path, line) if line else None
            if not found:
                return None, "no-date"
            said, where = found
        return {"directive_key": f"doc:{K.sha_text(path, 8)}:{line}:{qsha}", "said_at": K.utc_midnight(said),
                "said_at_verified": False, "author": "cyril", "channel": "doc", "source_event_id": None,
                "verbatim": span, "topics": [topic] if topic else [],
                "src_path": f"{path}:{line}" if line else path,
                **{**meta, "_date_note": f"date {said} from {where} (day only)"}}, "verified"
    return None, "no-source"


def plan_directives(conn, plan: Plan, events: EventReader) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """(rows to write, [(shard, state, detail)]) for every shard directive, decided the same way on every run:
    resolve + verify each quote; one row per directive_key (shards and speaker checks merged); drop a quote that a
    directive from another ingester already contains; drop a quote contained in a longer foldrec quote."""
    outcomes: list[tuple[str, str, str]] = []
    cands: dict[str, dict] = {}
    for sh, d in plan.directives:
        values, how = resolve_directive(sh, d, events)
        if values is None:
            outcomes.append((sh.name, how, f"{(as_text(d.get('quote')) or '')[:70]!r} source={jsonable(d.get('source'))}"))
            continue
        c = cands.get(values["directive_key"])
        if c is None:
            values["_shards"], values["_checks"] = [values["_shard"]], [values["_check"]]
            cands[values["directive_key"]] = values
        else:
            c["_shards"].append(values["_shard"])
            c["_checks"].append(values["_check"])
            c["topics"] = sorted(set(c["topics"]) | set(values["topics"]))
            outcomes.append((sh.name, "same-key-merged", values["directive_key"]))
    kept: list[dict] = []
    for c in sorted(cands.values(), key=lambda v: (-len(v["verbatim"]), v["directive_key"])):
        other = directive_present(conn, [c["_quote"], c["verbatim"]])
        if other:
            outcomes.append((c["_shards"][0], "already-present", f"as {other}: {c['_quote'][:70]!r}"))
            continue
        outer = next((k for k in kept if c["_quote"] in k["verbatim"] or c["verbatim"] in k["verbatim"]), None)
        if outer is not None:
            outcomes.append((c["_shards"][0], "contained", f"in {outer['directive_key']}: {c['_quote'][:70]!r}"))
            continue
        kept.append(c)
    rows = []
    for c in sorted(kept, key=lambda v: v["directive_key"]):
        shards = sorted(set(c["_shards"]))
        plural = "s" if len(shards) > 1 else ""
        summary = (f"foldrec shard{plural} {', '.join(shards)}; speaker_check: "
                   + " | ".join(dict.fromkeys(c["_checks"])) + (f"; {c['_date_note']}" if c["_date_note"] else ""))
        row = {k: v for k, v in c.items() if not k.startswith("_")}
        row["summary"] = summary
        row["_shard"] = shards[0]
        rows.append(row)
    return rows, outcomes


def runs_for_paths(conn, paths: list[str]) -> list[int]:
    paths = sorted({p.rstrip("/") for p in paths if p})
    if not paths:
        return []
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM run WHERE rtrim(receipt_path, '/') = ANY(%s) OR rtrim(run_dir, '/') = ANY(%s) "
                    "ORDER BY id", (paths, paths))
        return [r[0] for r in cur.fetchall()]


def model_links(conn, arm: dict) -> tuple[int | None, int | None]:
    child = parent = None
    ref = arm.get("child_ref") or ""
    m = LEDGER_ID_RE.search(ref)
    if m and not m.group(1).startswith("PARENT-"):
        child = db.model_id(conn, m.group(1))
    elif ref.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", ref):
        child = C.model_id_by_path(conn, ref)
    pm = LEDGER_ID_RE.search(arm.get("parent_name") or "")
    if pm and pm.group(1).startswith("PARENT-"):
        parent = db.model_id(conn, pm.group(1))
    return child, parent


def alias_links(plan: Plan, existing: dict[str, tuple[int, str | None]]) -> list[tuple[str, str, str, dict]]:
    """(from key, to key, relation, detail) implied by aliases (see module docstring)."""
    arm_keys = {a["key"] for a in plan.arms}
    out = []
    holders: dict[str, list[str]] = defaultdict(list)
    for a in plan.arms:
        for al in a["aliases"]:
            if al in arm_keys:
                rel = "rerun_of" if a["key"].startswith((al + "#", al + "@")) else "same_as"
                out.append((a["key"], al, rel, {"via": "alias", "alias": al}))
            elif al in existing:
                holders[al].append(a["key"])
    for al, arms in sorted(holders.items()):
        rel = "same_as" if len(arms) == 1 else "summarizes"
        out.extend((al, k, rel, {"via": "alias", "alias": al, "arms_sharing_alias": len(arms)}) for k in arms)
    return sorted(out)


# ---------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------

def replace_children(conn, tally: K.Tally, table: str, owner_col: str, owner_id: Any, rows: list[dict]) -> None:
    """Rows keyed (owner, ordinal) in list order; ordinals past the list are deleted."""
    from psycopg2 import sql
    for i, row in enumerate(rows):
        _id, state = K.upsert(conn, table, (owner_col, "ordinal"), {owner_col: owner_id, "ordinal": i, **row},
                              id_col=owner_col)
        tally.add(table, state)
    with conn.cursor() as cur:
        cur.execute(sql.SQL("DELETE FROM {} WHERE {} = %s AND ordinal >= %s").format(
            sql.Identifier(table), sql.Identifier(owner_col)), (owner_id, len(rows)))
        if cur.rowcount:
            tally.c[(table, "deleted")] += cur.rowcount
            tally.batch.wrote(cur.rowcount)


def ensure_technique(conn, tally: K.Tally, techs: dict[str, tuple[int, str]], t: dict, batch_id: int) -> int:
    cols = t["cols"]
    detail = {"foldrec": {"variants": [{"shard": s, "entry": jsonable(e)} for s, e in t["entries"]],
                          "arms": sorted(t["arms"]), "family_text": cols.get("family_text"),
                          "conflicting": cols.get("_conflicting", [])}}
    ext = {"code_location": cols["code_location"], "first_date": cols["first_date"],
           "first_date_text": cols["first_date_text"], "status_text": cols["status_text"], "detail": detail}
    n = norm_name(t["name"])
    if n in techs:
        tid, name = techs[n]
        _id, state = K.upsert(conn, "technique", ("name",), {"name": name, **ext})
        tally.add("technique", state)
        return tid
    first = t["entries"][0][0] if t["entries"] else (t["arms"][0] if t["arms"] else "?")
    tid, state = K.upsert(conn, "technique", ("name",), {
        "name": t["name"], "family": "fold",
        "description": cols["description"] or f"first seen in foldrec ({first})", **ext},
        ignore_in_compare=("description",))
    tally.add("technique", state)
    techs[n] = (tid, t["name"])
    return tid


def run(conn, batch: C.Batch, args) -> None:
    directory = shard_dir(args)
    plan = build_plan(directory)
    if plan.errors:
        raise db.LabError("foldrec refused: " + "; ".join(plan.errors[:10]))
    tally = K.Tally(batch)
    batch.saw(plan.records_seen)
    for s in plan.skipped:
        batch.skip(s)
    host = db.this_host()
    prov = {sh.name: batch.prov(sh.path, host=host) for sh in plan.shards}

    # shards + coverage
    for sh in plan.shards:
        p = prov[sh.name]
        doc = sh.doc
        read = clean_list(plan, sh.name, "sources_read", doc.get("sources_read"))
        not_read = clean_list(plan, sh.name, "sources_not_read", doc.get("sources_not_read"))
        _id, state = K.upsert(conn, "fold_shard", ("shard",), {
            "shard": sh.name, "schema": doc["schema"], "extracted_at": C.parse_ts(as_text(doc.get("extracted_at"))),
            "extracted_at_text": as_text(doc.get("extracted_at")),
            "n_experiments": len(doc.get("experiments") or []), "n_findings": len(doc.get("findings") or []),
            "n_techniques": len(doc.get("techniques") or []), "n_directives": len(doc.get("directives") or []),
            "n_sources_read": len(read), "n_sources_not_read": len(not_read),
            "src_path": p[0], "src_sha256": p[1], "src_host": p[2], "ingest_batch_id": batch.id}, id_col="shard")
        tally.add("fold_shard", state)
        rows = [{"state": "read", "path": as_text(s.get("path")) or "?", "lines": as_text(s.get("lines")),
                 "note": as_text(s.get("note"))} for s in read]
        rows += [{"state": "not_read", "path": as_text(s.get("path")) or "?", "lines": None,
                  "note": as_text(s.get("reason"))} for s in not_read]
        replace_children(conn, tally, "fold_coverage", "shard", sh.name, rows)

    techs = existing_techniques(conn)
    tech_ids = {n: ensure_technique(conn, tally, techs, t, batch.id) for n, t in plan.techniques.items()}
    existing = existing_experiments(conn)
    arm_ids: dict[str, int] = {}
    attached = 0
    for arm in plan.arms:
        first = arm["members"][0]["shard"].name
        p = prov[first]
        child_id, parent_id = model_links(conn, arm)
        run_ids = runs_for_paths(conn, [s["path"] for s in arm["sources"] if s["path"]] + [arm.get("child_ref") or ""])
        comp_tech = [tech_ids.get(norm_name(c["technique_text"])) if c["technique_text"] else None
                     for c in arm["components"]]
        row = existing.get(arm["key"])
        owned = row is None or row[1] == SOURCE
        if owned:
            ratios = [c["ratio"] for c in arm["components"] if c["ratio"] is not None]
            exp_id, state = K.upsert(conn, "experiment", ("key",), {
                "project": PROGRAM_PROJECT.get(arm["program"] or "", "neural-forge"), "key": arm["key"],
                "kind": "kernel" if arm["program"] == "kernel" else "fold",
                "technique_id": next((t for t in comp_tech if t is not None), None),
                "parent_id": parent_id, "child_id": child_id, "run_id": run_ids[0] if len(run_ids) == 1 else None,
                "ratio_body": arm["ratio_body"], "ratio_component": ratios[0] if len(ratios) == 1 else None,
                "params": {"ratio_text": arm["ratio_text"], "ratio_net": jsonable(arm["ratio_net"]), "axes": arm["axes"],
                           "objective": {"fit": arm["objective_fit"], "target": arm["objective_target"],
                                         "k": arm["objective_k"], "horizon": arm["objective_horizon"],
                                         "text": arm["objective_text"]},
                           "training_used": arm["training_used"]},
                "budget": {}, "behaviour_legacy": arm["behaviour"][0]["score"] if arm["behaviour"] else None,
                "verdict": arm["verdict_text"], "evidence_level": arm["evidence_level"] or "unverified",
                "results_path": next((s["path"] for s in arm["sources"] if s["path"]), None),
                "date": arm["date_value"],
                "detail": {"record": SCHEMA, "program": arm["program"], "wave": arm["wave"], "title": arm["title"],
                           "date_text": arm["date_text"], "owner": arm["owner"], "verdict_class": arm["verdict_class"]},
                "src_path": p[0], "src_sha256": p[1], "src_host": p[2], "ingest_batch_id": batch.id},
                jsonb_cols=("params", "detail", "budget"))
            tally.add("experiment", state)
            existing[arm["key"]] = (exp_id, SOURCE)
        else:
            exp_id = row[0]
            attached += 1
        arm_ids[arm["key"]] = exp_id
        _id, state = K.upsert(conn, "fold_arm", ("experiment_id",), {
            "experiment_id": exp_id, "arm_key": arm["key"], "title": arm["title"], "program": arm["program"],
            "wave": arm["wave"], "owner": arm["owner"], "date_text": arm["date_text"] or arm["date"],
            "parent_name": arm["parent_name"], "parent_params": arm["parent_params"], "child_ref": arm["child_ref"],
            "child_params": arm["child_params"], "child_exists_now": arm["child_exists_now"], "axes": arm["axes"],
            "ratio_body": arm["ratio_body"], "ratio_net": arm["ratio_net"], "ratio_text": arm["ratio_text"],
            "objective_fit": arm["objective_fit"], "objective_target": arm["objective_target"],
            "objective_k": arm["objective_k"], "objective_horizon": arm["objective_horizon"],
            "objective_text": arm["objective_text"], "training_used": arm["training_used"],
            "training_text": arm["training_text"], "verdict_class": arm["verdict_class"],
            "verdict_text": arm["verdict_text"], "cyril": arm["cyril"], "evidence_level": arm["evidence_level"],
            "shards": arm["shards"], "member_keys": arm["member_keys"], "run_ids": run_ids,
            "conflicts": arm["conflicts"], "record": arm["record"],
            "src_path": p[0], "src_sha256": p[1], "src_host": p[2], "ingest_batch_id": batch.id},
            jsonb_cols=("conflicts", "record"), id_col="experiment_id")
        tally.add("fold_arm", state)
        comps = [{**c, "technique_id": t} for c, t in zip(arm["components"], comp_tech)]
        replace_children(conn, tally, "fold_component", "experiment_id", exp_id, comps)
        replace_children(conn, tally, "fold_behaviour", "experiment_id", exp_id, arm["behaviour"])
        replace_children(conn, tally, "fold_diagnostic", "experiment_id", exp_id, arm["diagnostics"])
        replace_children(conn, tally, "fold_source", "experiment_id", exp_id, arm["sources"])

    # aliases and links (the whole foldrec-written set is replaced: rows no shard states any more are deleted)
    arm_by_key = {a["key"]: a for a in plan.arms}
    want_alias = {(al, arm_ids[a["key"]]): a for a in plan.arms for al in a["aliases"]}
    for (al, eid), a in sorted(want_alias.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        p = prov[a["members"][0]["shard"].name]
        _id, state = K.upsert(conn, "experiment_alias", ("alias", "experiment_id"), {
            "alias": al, "experiment_id": eid, "source": SOURCE, "src_path": p[0], "src_sha256": p[1],
            "src_host": p[2], "ingest_batch_id": batch.id}, id_col="experiment_id")
        tally.add("experiment_alias", state)
    links = alias_links(plan, existing)
    want_link = set()
    for from_key, to_key, rel, detail in links:
        frm, to = existing[from_key][0], existing[to_key][0]
        want_link.add((frm, to, rel))
        holder = arm_by_key.get(from_key) or arm_by_key[to_key]
        p = prov[holder["members"][0]["shard"].name]
        _id, state = K.upsert(conn, "experiment_link", ("from_id", "to_id", "relation"), {
            "from_id": frm, "to_id": to, "relation": rel, "source": SOURCE, "detail": detail,
            "src_path": p[0], "src_sha256": p[1], "src_host": p[2], "ingest_batch_id": batch.id},
            jsonb_cols=("detail",), id_col="from_id")
        tally.add("experiment_link", state)
    with conn.cursor() as cur:
        cur.execute("SELECT alias, experiment_id FROM experiment_alias WHERE source = %s", (SOURCE,))
        stale = [r for r in cur.fetchall() if (r[0], r[1]) not in want_alias]
        for al, eid in stale:
            cur.execute("DELETE FROM experiment_alias WHERE alias = %s AND experiment_id = %s", (al, eid))
        cur.execute("SELECT from_id, to_id, relation FROM experiment_link WHERE source = %s", (SOURCE,))
        stale_l = [r for r in cur.fetchall() if (r[0], r[1], r[2]) not in want_link]
        for frm, to, rel in stale_l:
            cur.execute("DELETE FROM experiment_link WHERE from_id = %s AND to_id = %s AND relation = %s", (frm, to, rel))
    for table, n in (("experiment_alias", len(stale)), ("experiment_link", len(stale_l))):
        if n:
            tally.c[(table, "deleted")] += n
            batch.wrote(n)

    # findings
    name_to_arm = {}
    for a in plan.arms:
        for n in [a["key"]] + a["aliases"]:
            name_to_arm.setdefault(n, arm_ids[a["key"]])
    for f in plan.findings:
        p = prov[f["shards"][0]]
        fid, state = K.upsert(conn, "fold_finding", ("finding_key",), {
            "finding_key": f["key"], "statement": f["statement"], "status": f["status"], "scope": f["scope"],
            "shards": f["shards"], "conflicts": f["conflicts"], "src_path": p[0], "src_sha256": p[1], "src_host": p[2],
            "ingest_batch_id": batch.id}, jsonb_cols=("conflicts",))
        tally.add("fold_finding", state)
        for k in f["experiment_keys"]:
            eid = name_to_arm.get(k) or (existing[k][0] if k in existing else None)
            _id, state = K.upsert(conn, "fold_finding_experiment", ("finding_id", "experiment_key"),
                                  {"finding_id": fid, "experiment_key": k, "experiment_id": eid}, id_col="finding_id")
            tally.add("fold_finding_experiment", state)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM fold_finding_experiment WHERE finding_id = %s AND NOT (experiment_key = ANY(%s))",
                        (fid, f["experiment_keys"]))
            if cur.rowcount:
                tally.c[("fold_finding_experiment", "deleted")] += cur.rowcount
                batch.wrote(cur.rowcount)
        replace_children(conn, tally, "fold_finding_source", "finding_id", fid, f["sources"])

    # directives
    states: Counter = Counter()
    events = EventReader(sorted({e for _s, d in plan.directives
                                 if (e := ev_id((d.get("source") or {}).get("ev") if isinstance(d.get("source"), dict)
                                                else None)) is not None}), batch.notes)
    rows, outcomes = plan_directives(conn, plan, events)
    for shard, how, detail in outcomes:
        states[how] += 1
        batch.skip(f"directive {shard}: {how}: {detail}")
    for values in rows:
        p = prov[values.pop("_shard")]
        _id, state = K.upsert(conn, "directive", ("directive_key",), {
            **values, "src_sha256": p[1], "src_host": p[2], "ingest_batch_id": batch.id})
        tally.add("directive", state)
        states["written"] += 1

    with conn.cursor() as cur:
        cur.execute("SELECT arm_key FROM fold_arm WHERE NOT (arm_key = ANY(%s)) ORDER BY arm_key",
                    ([a["key"] for a in plan.arms],))
        gone = [r[0] for r in cur.fetchall()]
    for g in gone[:20]:
        batch.notes.append(f"fold_arm {g} is no longer in any shard (kept)")
    for pr in plan.problems[:150]:
        batch.notes.append(pr)
    K.finish(batch, tally)
    pruned = sorted((t, n) for (t, s), n in tally.c.items() if s == "deleted")
    if pruned:
        batch.tally_lines.append("deleted (surplus ordinals / stale aliases+links): "
                                 + " ".join(f"{t}={n}" for t, n in pruned))
    batch.tally_lines.append(f"arms={len(plan.arms)} attached-to-existing={attached} shards={len(plan.shards)} "
                             f"problems={len(plan.problems)} directives: "
                             + (" ".join(f"{k}={v}" for k, v in sorted(states.items())) or "none"))


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------

def shard_dir(args) -> Path:
    given = getattr(args, "inventory", None)
    return Path(given) if given else DEFAULT_DIR


def dry_run(args, conn=None) -> dict:
    """Parse, validate, merge and resolve read-only.  Returns the report (also used by tests)."""
    directory = shard_dir(args)
    plan = build_plan(directory)
    rep: dict[str, Any] = {
        "directory": str(directory), "errors": plan.errors, "skipped": plan.skipped, "problems": plan.problems,
        "shards": {sh.name: {k: len(sh.doc.get(k) or []) for k in ("experiments", "findings", "techniques", "directives",
                                                                     "sources_read", "sources_not_read")}
                   for sh in plan.shards},
        "records_seen": plan.records_seen, "arms": len(plan.arms),
        "merged_arms": {a["key"]: a["member_keys"] for a in plan.arms if len(a["members"]) > 1},
        "conflicting_arms": {a["key"]: sorted(a["conflicts"]) for a in plan.arms if a["conflicts"]},
        "alias_only_shared": plan.alias_only,
        "components": sum(len(a["components"]) for a in plan.arms),
        "behaviour_rows": sum(len(a["behaviour"]) for a in plan.arms),
        "diagnostics": sum(len(a["diagnostics"]) for a in plan.arms),
        "sources": sum(len(a["sources"]) for a in plan.arms),
        "findings": len(plan.findings), "techniques": len(plan.techniques), "directives": len(plan.directives),
        "verdict_class": dict(Counter(a["verdict_class"] or "(null)" for a in plan.arms)),
        "program": dict(Counter(a["program"] or "(null)" for a in plan.arms)),
    }
    own = conn is None
    if own:
        conn = db.connect("ro", app="ingest-foldrec-dry-run")
    try:
        existing = existing_experiments(conn)
        techs = existing_techniques(conn)
        rep["attach_to_existing"] = sorted(a["key"] for a in plan.arms
                                           if a["key"] in existing and existing[a["key"]][1] != SOURCE)
        rep["new_or_foldrec_experiments"] = len(plan.arms) - len(rep["attach_to_existing"])
        rep["alias_links"] = [f"{f} {rel} {t}" for f, t, rel, _d in alias_links(plan, {**existing, **{
            a["key"]: (0, SOURCE) for a in plan.arms}})]
        rep["techniques_existing"] = sorted(t["name"] for n, t in plan.techniques.items() if n in techs)
        rep["techniques_new"] = len(plan.techniques) - len(rep["techniques_existing"])
        rep["run_links"] = {a["key"]: runs_for_paths(conn, [s["path"] for s in a["sources"] if s["path"]]
                                                      + [a.get("child_ref") or ""]) for a in plan.arms}
        rep["run_links"] = {k: v for k, v in rep["run_links"].items() if v}
        states: Counter = Counter()
        events = EventReader(sorted({e for _s, d in plan.directives
                                     if (e := ev_id((d.get("source") or {}).get("ev")
                                                    if isinstance(d.get("source"), dict) else None)) is not None}),
                             rep.setdefault("notes", []))
        rows, outcomes = plan_directives(conn, plan, events)
        states["would-write"] = len(rows)
        for shard, how, _detail in outcomes:
            states[how] += 1
        rep["directive_states"] = dict(states)
        rep["directive_failures"] = [f"{shard}: {how}: {detail}" for shard, how, detail in outcomes]
        conn.rollback()
    finally:
        if own:
            conn.close()
    return rep


def print_report(rep: dict, limit: int = 40) -> None:
    print(f"[foldrec dry-run] {rep['directory']}")
    for name, counts in rep["shards"].items():
        print(f"  shard {name}: " + " ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"  records seen={rep['records_seen']} arms={rep['arms']} (merged across shards: {len(rep['merged_arms'])}, "
          f"with conflicts: {len(rep['conflicting_arms'])})")
    print(f"  components={rep['components']} behaviour={rep['behaviour_rows']} diagnostics={rep['diagnostics']} "
          f"sources={rep['sources']} findings={rep['findings']} techniques={rep['techniques']} "
          f"(existing {len(rep.get('techniques_existing', []))}, new {rep.get('techniques_new')}) "
          f"directives={rep['directives']} {rep.get('directive_states', {})}")
    print(f"  experiments: new/foldrec-owned={rep.get('new_or_foldrec_experiments')} "
          f"attach-to-existing={rep.get('attach_to_existing')}")
    print(f"  alias links: {len(rep.get('alias_links', []))} "
          f"{dict(Counter(x.split(' ')[1] for x in rep.get('alias_links', [])))} {rep.get('alias_links', [])[:limit]}")
    print(f"  run links: {len(rep.get('run_links', {}))} arms")
    print(f"  verdict_class: {rep['verdict_class']}")
    print(f"  program: {rep['program']}")
    for k, v in list(rep["merged_arms"].items())[:limit]:
        print(f"  merged {k}: {v}")
    for k, v in list(rep["conflicting_arms"].items())[:limit]:
        print(f"  conflict {k}: {v}")
    for k, v in list(rep["alias_only_shared"].items())[:limit]:
        print(f"  alias shared (not merged) {k!r}: {v}")
    for f in rep.get("directive_failures", [])[:limit]:
        print(f"  directive {f}")
    for e in rep["errors"]:
        print(f"  ERROR {e}")
    for s in rep["skipped"][:limit]:
        print(f"  skipped {s}")
    for p in rep["problems"][:limit]:
        print(f"  problem {p}")
    if len(rep["problems"]) > limit:
        print(f"  ... {len(rep['problems']) - limit} more problems")
    for n in rep.get("notes", []):
        print(f"  note {n}")


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m ingest.foldrec", description=__doc__.split("\n")[0])
    ap.add_argument("--inventory", help=f"shard directory (default {DEFAULT_DIR})")
    ap.add_argument("--dry-run", action="store_true", help="parse + validate + resolve read-only; write nothing")
    args = ap.parse_args()
    if args.dry_run:
        rep = dry_run(args)
        print_report(rep)
        return 1 if rep["errors"] else 0
    return K.standalone_main(sys.modules[__name__])


if __name__ == "__main__":
    sys.exit(main())
