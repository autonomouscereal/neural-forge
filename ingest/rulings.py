"""ingest.rulings -- owner rulings and agent decisions -> ruling.

Sources (AI server, read only):
    $OC/rulings/*.json                         sealed owner rulings (schema neural-forge-omni-owner-ruling-v1)
    $OC/distill/STAGE_D_RULINGS_2026-10-12.md  the file of record: Cyril's words, prior rulings, the
                                               'What these rulings settle' table, and every dated entry
    $Q/*_ruling.txt                            machine-read ruling files (first line = ruling id) -> attachments
    $Q/D2_DECISIONS_2026-10-15.json            machine-readable Claude-lead D2 decisions -> attachment
    $Q/backups/rulings_append_2026-10-17.md    the 2026-10-17 append as written -> attachment

Authority is decided from the entry's own words, never guessed:
    'not Cyril's words' / 'Claude-lead decision'      -> who 'claude-lead', authority 'lead'
    'Cyril to ...' / 'His words'                      -> who 'cyril',       authority 'owner'
    'approved by Cyril' (no words of his own quoted
    beyond the approval)                              -> who 'claude-lead', authority 'owner'
    the 2026-10-12 decision table (written by Claude under the owner's GO, basis column kept)
                                                      -> who 'claude',      authority 'agent'
verbatim holds only text inside double quotes in an owner entry (Cyril's words); decision holds the entry
as written.  directive_id links the ruling to the directive row whose verbatim matches one of its quotes
(run ingest.directives first; a re-run fills links added later).  Re-runnable.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import common as C
from . import knowledge_common as K
from .common import db

SOURCE = "rulings"

RULINGS_MD = "distill/STAGE_D_RULINGS_2026-10-12.md"
ID_IN_HEADER_RE = re.compile(r"(?:ruling|decision)\s+`([a-z0-9][a-z0-9.:-]+)`|\*\*`([a-z0-9][a-z0-9.:-]+)`", re.I)
TS_RE = re.compile(r"(20\d\d-[01]\d-[0-3]\d)T(\d\d):(\d\d)(?::(\d\d))?Z")
APPROX_RE = re.compile(r"(20\d\d-[01]\d-[0-3]\d)\s*\(~(\d\d):(\d\d)Z\)")
QUOTE_RE = re.compile(r"\"([^\"]{6,})\"")
PATH_REF_RE = re.compile(r"`(\$(?:OC|Q|T)/[^`\s]+|distill/[^`\s]+\.(?:md|json|txt)|reports/[^`\s]+\.md)`")
NOT_CYRIL_RE = re.compile(r"not cyril'?s words|claude-lead decision", re.I)
OWNER_RE = re.compile(r"cyril to \w+|his words|approved by cyril|cyril's words", re.I)
WORDS_RE = re.compile(r"cyril to \w+|his words", re.I)            # the entry quotes him, not only his approval


def roots() -> dict[str, str]:
    C.roots_from_env()
    return {"$OC": C.OC, "$Q": C.Q, "$T": C.OC + "/distill/tools"}


def resolve_ref(ref: str) -> str:
    r = roots()
    for var, base in r.items():
        if ref.startswith(var + "/"):
            return base + ref[len(var):]
    if ref.startswith("distill/"):
        return C.OC + "/" + ref
    return ref


def attachment(path: str) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    return {"path": p.as_posix(), "sha256": db.file_sha256(p)}


def decided_at(text: str, fallback: datetime) -> datetime:
    m = TS_RE.search(text)
    if m:
        return datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4) or '00'}+00:00")
    m = APPROX_RE.search(text)
    if m:
        return datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}:{m.group(3)}:00+00:00")
    d = K.first_date(text)
    return K.utc_midnight(d) if d else fallback


def slug(text: str, n: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", re.sub(r"\([^)]*\)", "", text.lower())).strip("-")
    return s[:n].rstrip("-")


def strip_md(text: str) -> str:
    return re.sub(r"\*\*|`", "", text).strip()


# ---------------------------------------------------------------------------
# parsing STAGE_D_RULINGS (pure: text in, records out; tested with fixtures)
# ---------------------------------------------------------------------------

def parse_rulings_md(text: str) -> list[dict]:
    """Records {key, decided_at, who, authority, verbatim, decision, scope, refs[]} in file order."""
    lines = text.splitlines()
    out: list[dict] = []
    section = ""
    section_date = None
    i = 0

    def block(start: int) -> tuple[str, int]:
        """A top-level bullet plus its indented continuation lines."""
        j = start + 1
        while j < len(lines) and (lines[j].startswith("  ") or lines[j].strip() == "" and j + 1 < len(lines)
                                   and lines[j + 1].startswith("  ")):
            j += 1
        return "\n".join(lines[start:j]).rstrip(), j

    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            section_date = K.first_date(section) or section_date
            i += 1
            continue
        low = section.lower()
        base = K.utc_midnight(section_date) if section_date else datetime(2026, 10, 12, tzinfo=timezone.utc)
        if low.startswith("cyril's words") and "realignment" not in low and re.match(r"^\d+\.\s", line):
            # the 2026-10-12 owner words: one owner ruling per numbered item, merged with the sealed JSON later
            n = re.match(r"^(\d+)\.", line).group(1)
            quotes = QUOTE_RE.findall(line)
            out.append({"key": f"OWNER-WORDS-2026-10-12-{n}", "decided_at": base, "who": "cyril", "authority": "owner",
                        "verbatim": " | ".join(quotes) or None, "decision": strip_md(line), "scope": ["stage-d"],
                        "refs": []})
            i += 1
            continue
        if line.startswith("- ") and not line.startswith("- **") and low.startswith("cyril's words")                 and "realignment" not in low:
            # 'Prior rulings that still hold' bullets sit under the first section
            d = K.first_date(line)
            if d:
                quotes = QUOTE_RE.findall(line)
                out.append({"key": f"OWNER-PRIOR-{d.isoformat()}-{slug(line.split(':', 1)[-1], 40)}",
                            "decided_at": K.utc_midnight(d), "who": "cyril", "authority": "owner",
                            "verbatim": " | ".join(quotes) or None, "decision": strip_md(line[2:]),
                            "scope": ["stage-d", "prior"], "refs": []})
            i += 1
            continue
        if low.startswith("what these rulings settle") and line.startswith("|") and not re.match(r"^\|[-\s|]+\|$", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 3 and cells[0].lower() != "item":
                out.append({"key": f"STAGE-D-2026-10-12/{slug(cells[0])}", "decided_at": base, "who": "claude",
                            "authority": "agent", "verbatim": None,
                            "decision": f"{strip_md(cells[0])}: {strip_md(cells[1])} (basis: {strip_md(cells[2])})",
                            "scope": ["stage-d"], "refs": PATH_REF_RE.findall(line)})
            i += 1
            continue
        if line.startswith("- **"):
            text_block, j = block(i)
            header = text_block.split("\n", 1)[0]
            m = ID_IN_HEADER_RE.search(header)
            if m and not header.lower().startswith("- **plan of record"):
                key = m.group(1) or m.group(2)
                if NOT_CYRIL_RE.search(text_block):
                    who, authority = "claude-lead", "lead"
                elif WORDS_RE.search(text_block):
                    who, authority = "cyril", "owner"
                elif OWNER_RE.search(text_block) or "cyril's words" in low:
                    who, authority = "claude-lead", "owner"     # Claude-written text the owner approved
                else:
                    who, authority = "claude-lead", "lead"
                quotes = QUOTE_RE.findall(text_block) if authority == "owner" else []
                scope = sorted({s for s in ("k3", "d2", "h3", "lanes", "speech", "data", "distill", "media")
                                if re.search(rf"\b{s}\b", text_block, re.I)})
                out.append({"key": key, "decided_at": decided_at(header, base), "who": who, "authority": authority,
                            "verbatim": " | ".join(quotes) or None, "decision": strip_md(text_block[2:]),
                            "scope": scope, "refs": PATH_REF_RE.findall(text_block)})
            i = j
            continue
        i += 1
    return out


# ---------------------------------------------------------------------------

def directive_index(conn) -> list[tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, verbatim FROM directive ORDER BY said_at, id")
        return [(i, norm(v)) for i, v in cur.fetchall()]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("…", "...")).strip().lower()


def match_directive(index: list[tuple[int, str]], verbatim: str | None) -> int | None:
    """The directive whose words contain (or are contained in) a quoted fragment of >= 25 chars."""
    if not verbatim:
        return None
    for part in verbatim.split(" | "):
        for frag in re.split(r"\.\.\.|…", norm(part)):
            frag = frag.strip(" .")
            if len(frag) < 25:
                continue
            for did, words in index:
                if frag in words or (len(words) >= 25 and words in frag):
                    return did
    return None


def run(conn, batch: C.Batch, args) -> None:
    C.roots_from_env()
    tally = K.Tally(batch)
    index = directive_index(conn)

    # machine-read ruling files: id (first line) -> attachment
    by_id_attach: dict[str, list[dict]] = {}
    for p in sorted(Path(C.Q).glob("*_ruling.txt")):
        first = C.read_text(p, limit=4096).splitlines()[:1]
        if first and re.fullmatch(r"[a-z0-9][a-z0-9.:-]+", first[0].strip(), re.I):
            att = attachment(p.as_posix())
            batch.file_shas.append(att["sha256"])
            by_id_attach.setdefault(first[0].strip(), []).append(att)
    d2 = attachment(C.Q + "/D2_DECISIONS_2026-10-15.json")
    d2_ids: list[str] = []
    if d2:
        d2_doc = C.load_json(d2["path"]) or {}
        d2_ids = list((d2_doc.get("rulings") or {}).get("ids") or [])
        batch.file_shas.append(d2["sha256"])
    append_1017 = attachment(C.Q + "/backups/rulings_append_2026-10-17.md")

    # 1. sealed owner rulings
    owner_json_keys: list[str] = []
    owner_json_words: list[str] = []
    for p in sorted(Path(C.OC, "rulings").glob("*.json")):
        rec = C.load_json(p)
        batch.saw()
        if not isinstance(rec, dict) or not rec.get("ruling_id"):
            batch.skip(f"{p.name}: not a ruling record")
            continue
        prov = batch.prov(p)
        verbatim = " | ".join(rec.get("verbatim") or []) or None
        turns = (rec.get("provenance") or {}).get("user_turn_timestamps_utc") or []
        said = C.parse_ts(turns[0].split(" ")[0]) if turns else None     # first owner turn, else the seal time
        owner_json_words.extend(rec.get("verbatim") or [])
        rid, state = K.upsert(conn, "ruling", ("ruling_key",), {
            "ruling_key": rec["ruling_id"],
            "decided_at": said or C.parse_ts(rec.get("sealed_at")),
            "who": "cyril", "authority": "owner", "verbatim": verbatim,
            "decision": f"{rec.get('kind')}: {rec.get('source')}; sealed {rec.get('sealed_at')} by {rec.get('sealer')}",
            "scope": ["stage-d"], "directive_id": match_directive(index, verbatim),
            "attachments": list(rec.get("attachments") or []),
            "src_path": prov[0], "src_sha256": prov[1], "src_host": prov[2], "ingest_batch_id": batch.id},
            jsonb_cols=("attachments",))
        tally.add("ruling", state)
        owner_json_keys.append(rec["ruling_id"])

    # 2. the file of record
    md_path = Path(C.OC, RULINGS_MD)
    prov = batch.prov(md_path)
    for rec in parse_rulings_md(C.read_text(md_path)):
        batch.saw()
        if rec["key"].startswith("OWNER-WORDS-") and rec["verbatim"] and all(
                any(norm(q) in norm(w) for w in owner_json_words) for q in rec["verbatim"].split(" | ")):
            batch.skip(f"{rec['key']}: same words as the sealed owner ruling {', '.join(owner_json_keys)}")
            continue
        atts = []
        for ref in rec["refs"]:
            a = attachment(resolve_ref(ref))
            if a and a not in atts:
                atts.append(a)
        for a in by_id_attach.get(rec["key"], []):
            if a not in atts:
                atts.append(a)
        if d2 and rec["key"] in d2_ids and d2 not in atts:
            atts.append(d2)
        if append_1017 and rec["key"].endswith("2026-10-17") and append_1017 not in atts:
            atts.append(append_1017)
        if rec["key"].startswith("OWNER-WORDS-2026-10-12") and owner_json_keys:
            rec["decision"] += f" (sealed as {', '.join(owner_json_keys)})"
        _id, state = K.upsert(conn, "ruling", ("ruling_key",), {
            "ruling_key": rec["key"], "decided_at": rec["decided_at"], "who": rec["who"],
            "authority": rec["authority"], "verbatim": rec["verbatim"], "decision": rec["decision"],
            "scope": rec["scope"], "directive_id": match_directive(index, rec["verbatim"]), "attachments": atts,
            "src_path": prov[0], "src_sha256": prov[1], "src_host": prov[2], "ingest_batch_id": batch.id},
            jsonb_cols=("attachments",))
        tally.add("ruling", state)
    K.finish(batch, tally)


if __name__ == "__main__":
    K.main_guard(sys.modules[__name__])
