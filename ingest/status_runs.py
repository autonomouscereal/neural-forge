"""ingest.status_runs -- queue status files -> run_event (every timestamped line) + run (paired jobs) + run_device.

The campaign's *.status files are append-only '<ISO-UTC> <free text>' lines in at least eight grammars
(job=.. state=.., '<job> START|PASS|FAIL', 'segment N exit', '<tool>-<ver> ...', ...).  This parser:
  1. stores every timestamped line as a run_event (natural key: file + line);
  2. assigns each line a job key and a state (start | pass | fail | refused | stop | skip | wait | end | info);
  3. pairs a job's start with its next terminal line into one run; a second start before a terminal
     closes the first as 'superseded'; a start with no terminal line is 'unknown' (open at end of file);
     a terminal line with no start is a run of its own.
Pairing is APPROXIMATE (heuristic grammar); every run carries its start/end lines in config and
run.notes says so.  GPU placement comes from the index named on the start line (mapping recorded as
'index-from-status') or from GPU-UUIDs when the line has them.  Files owned by trainer_runs (full-KD /
media M1 queue.status, d2.status) and untimestamped files (*.go, *.verdict, rulings, intake.status) are
not paired here; the intake cron heartbeat log is skipped.
"""
from __future__ import annotations

import glob
import os
import re

from . import common as C
from .common import db

SOURCE = "status_runs"

STATE_PATTERNS = [   # first match wins
    ("refused", r"\bREFUSED\b|\brefused\b|state=refused"),
    ("fail", r"\bFAIL(?:ED)?\b|\bfailed\b|state=failed|\bexit[= ](?!0\b)\d+|\brc=(?!0\b)\d+|Traceback|crash \d"),
    ("end", r"\bqueue[ =_]?(?:end|END)\b|state=end\b|event=queue_end|QUEUE end|queue end\b|\bd3/queue end"),
    ("pass", r"\bPASS(?:ED)?\b|state=done\b|\bDONE\b|\bdone\b|\bSEALED\b|\bsealed\b|\bexit[= ]0\b|\brc=0\b|\bOK\b"),
    ("start", r"\bSTART\b|\bstart\b|\bstarted\b|state=started|\bARMED\b|\barmed\b|state=armed|queue=start|event=queue_start"),
    ("wait", r"\bWAIT\b|\bwaiting\b|state=waiting"),
    ("stop", r"\bSTOPPED\b|\bstopped\b|STOP file|preempted|\bEND\b|\bSUPERSEDED\b"),
    ("skip", r"\bSKIP\b|\bskipped\b"),
]
STATE_RX = [(s, re.compile(p)) for s, p in STATE_PATTERNS]
TERMINAL = {"pass", "fail", "refused", "stop", "skip", "end"}
RUN_STATUS = {"pass": "pass", "end": "pass", "fail": "fail", "refused": "refused", "stop": "stopped", "skip": "skipped"}

# (regex on file path, kind, project, lane)
FILE_RULES = [
    (r"/d1[_.][^/]*$", "distill", "chimera-omni", "L-D1"),
    (r"/d2clean\.status$|/d2refill\.status$", "cache", "corpus", "L-DATA"),
    (r"/d3\.status$", "smoke", "chimera-omni", None),
    (r"/export\.status$", "export", "chimera-omni", None),
    (r"/gpu0\.status$", "cache", "chimera-omni", None),
    (r"/gpu4\.status$", "cache", "media", "L-MEDIA-DATA"),
    (r"/gpu6\.status$", "cache", "k3", "L-K3"),
    (r"/j6\.(status|done)$", "cache", "chimera-omni", None),
    (r"/k3_seal_decide\.status$", "review", "k3", "L-K3"),
    (r"/k3mc-gpu\d\.status$", "cache", "k3", "L-K3ACC"),
    (r"/p11\.status$|/p2\.status$", "probe", "chimera-omni", None),
    (r"/p40\.status$", "cache", "media", "L-MEDIA-DATA"),
    (r"/post_d1\.status$", "probe", "chimera-omni", "L-D1"),
    (r"/toolcall/logs/", "gen", "chimera-omni", "L-TOOL"),
    (r"/flash_lane\.log$", "gen", "chimera-omni", "L-TOOL"),
    (r"/vmath-pilot", "gen", "corpus", "L-DATA"),
    (r"/research/fold/", "fold", "neural-forge", "L-FOLD"),
    (r"/research/", "other", "neural-forge", "L-RESEARCH"),
]
JOB_KIND = [(re.compile(r"test", re.I), "smoke"), (re.compile(r"probe|parity|\bb2\b|tps|bench", re.I), "probe")]

JOB_RX = [
    re.compile(r"\bjob=(\S+)"),
    re.compile(r"^(?:queue\s+)?(?:\S+\s+)?step=(\S+)"),
    re.compile(r"^(segment \d+)\b"),
    re.compile(r"^(?:start\s+)?(probe|train)\b"),
    re.compile(r"^(\w+)\s+(gpu\d)\b"),                       # 'd2clean gpu2 ARMED' -> d2clean/gpu2
    re.compile(r"^([A-Za-z0-9][\w.-]*/[\w.-]+)\s"),          # 'd3/tests start'
    re.compile(r"^([A-Za-z0-9][\w.-]*)\s+(?:START|PASS|FAIL|WAIT|SKIP|DONE|REFUSED|STOPPED)\b"),
]
QUEUE_RX = re.compile(r"^(?:queue|QUEUE|DONE)\b|\b(?:queue|QUEUE)\b[ =_]?(?:start|end|START|END)\b|"
                      r"event=queue_(?:start|end)|\bstate=(?:armed|end)\b|\brunner START\b|\brunner STOPPED\b")
GPU_RX = re.compile(r"\bgpu[= ]?(\d)\b|\bGPU(\d)\b", re.I)
UUID_RX = re.compile(r"\bGPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
KERNEL_RX = re.compile(r"\bkernel[= ]([\w.-]+)")
TOOLVER_RX = re.compile(r"\b([a-z][a-z0-9_]*(?:\.sh|\.py)?)[- ](\d+\.\d+\.\d+[a-z]?)\b")
SHA_RX = re.compile(r"\b(?:sha256?|tool_sha|trainer|frozen|tool)[= ]([0-9a-f]{12,64})\b")
PATH_RX = re.compile(r"(/(?:home|mnt)/[^\s,;)\"'\]]+)")


def status_files() -> list[str]:
    pats = [C.Q + "/*.status", C.Q + "/*.done", C.Q + "/*.sealed", C.Q + "/*.reviewed",
            C.OC + "/distill/toolcall/logs/*.status", C.OC + "/distill/flash_lane.log",
            C.LX + "/research/*/queue.status", C.LX + "/research/*/*/queue.status",
            C.LX + "/research/*/*/*/queue.status", C.LX + "/distill/*/*/queue.status", C.LX + "/distill/*/queue.status"]
    found = sorted({p for pat in pats for p in glob.glob(pat)})
    return [p for p in found if not any(rx.search(p) for rx in C.TRAINER_OWNED_STATUS)
            and not p.endswith("/intake.status")]


def line_state(text: str) -> str:
    return next((s for s, rx in STATE_RX if rx.search(text)), "info")


def line_job(text: str) -> str:
    m = JOB_RX[0].search(text)          # an explicit job= always wins
    if m:
        return m.group(1)
    if QUEUE_RX.search(text):
        return "_queue"
    for rx in JOB_RX:
        m = rx.search(text)
        if m:
            return "/".join(g for g in m.groups() if g)
    first = text.split(" ", 1)[0]
    return first if first else "_lane"


def rel(path: str) -> str:
    if path.startswith(C.OC + "/"):
        return "$OC/" + path[len(C.OC) + 1:]
    if path.startswith(C.LX + "/"):
        return "$L/" + path[len(C.LX) + 1:]
    return path


def parse_file(path: str) -> list[dict]:
    """[{ts, text, line, state, job}] for every timestamped line, in file order."""
    out = []
    for raw in C.read_text(path).splitlines():
        m = C.TS_RE.match(raw.strip())
        if not m:
            continue
        text = m.group(2)
        out.append({"ts": C.parse_ts(m.group(1)), "ts_text": m.group(1), "text": text, "line": raw.rstrip(),
                    "state": line_state(text), "job": line_job(text)})
    return out


def pair_runs(events: list[dict]) -> list[dict]:
    """Group events into job runs: [{job, start, end, status, events[]}]."""
    runs, open_runs = [], {}
    for ev in events:
        job, state = ev["job"], ev["state"]
        if state == "start":
            if job in open_runs:
                prev = open_runs.pop(job)
                prev["status"] = "superseded"
                runs.append(prev)
            open_runs[job] = {"job": job, "start": ev, "end": None, "status": None, "events": [ev]}
        elif state in TERMINAL:
            r = open_runs.pop(job, None) or {"job": job, "start": None, "end": None, "status": None, "events": []}
            r["end"] = ev
            r["events"].append(ev)
            r["status"] = RUN_STATUS[state]
            runs.append(r)
        else:
            target = open_runs.get(job) or open_runs.get("_queue")
            if target is not None:
                target["events"].append(ev)
            else:
                ev["unattached"] = True
    for r in open_runs.values():
        r["status"] = "unknown"
        runs.append(r)
    return runs


def file_rule(path: str) -> tuple[str, str, str | None]:
    for rx, kind, project, lane in FILE_RULES:
        if re.search(rx, path):
            return kind, project, lane
    return "other", "chimera-omni", None


def _devices(conn, text: str) -> list[tuple[int, str]]:
    """[(gpu_id, mapping)] named on a line: GPU-UUIDs exactly, else gpu indexes (backfill mapping)."""
    out = []
    for u in UUID_RX.findall(text):
        gid = C.gpu_for(conn, uuid=u)
        if gid:
            out.append((gid, "uuid-from-status"))
    if out:
        return out
    seen = set()
    for m in GPU_RX.finditer(text):
        idx = int(next(g for g in m.groups() if g))
        gid = C.gpu_for(conn, index=idx)
        if gid and gid not in seen:
            seen.add(gid)
            out.append((gid, "index-from-status"))
    return out


def run(conn, batch: C.Batch, args) -> None:
    host_id = C.host_id(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT rtrim(path, '/'), id FROM dataset_version WHERE path IS NOT NULL")
        dv_by_path = dict(cur.fetchall())
        cur.execute("SELECT rtrim(path, '/'), id FROM model WHERE path IS NOT NULL ORDER BY (ledger_checkpoint_id IS NULL)")
        model_by_path: dict[str, int] = {}
        for p, mid in cur.fetchall():
            model_by_path.setdefault(p, mid)
    for path in status_files():
        events = parse_file(path)
        if not events:
            batch.skip(f"no timestamped lines: {rel(path)}")
            continue
        prov = batch.prov(path)
        kind0, project, lane = file_rule(path)
        for r in pair_runs(events):
            batch.saw()
            first = r["start"] or r["end"]
            last = r["end"]
            job = r["job"]
            kind = kind0
            if job != "_queue":
                kind = next((k for rx, k in JOB_KIND if rx.search(job)), kind0)
            key_line = (r["start"] or r["end"])["text"]
            all_text = " ".join(e["text"] for e in r["events"])
            tv = TOOLVER_RX.search(key_line) or TOOLVER_RX.search(all_text)
            sha = SHA_RX.search(key_line)
            kern = KERNEL_RX.search(key_line)
            refs = sorted({p.rstrip(".") for p in PATH_RX.findall(all_text)})[:40]
            name_file = os.path.basename(path)
            config = {"source_file": rel(path), "job": job, "start_line": r["start"]["text"] if r["start"] else None,
                      "end_line": last["text"] if last else None, "n_lines": len(r["events"]),
                      "tool": tv.group(1) if tv else None, "tool_version": tv.group(2) if tv else None,
                      "sha_prefix": sha.group(1) if sha else None, "kernel": kern.group(1) if kern else None,
                      "refs": refs, "pairing": "status-line heuristic"}
            tool_vid = None
            if tv:
                tname = re.sub(r"\.(sh|py)$", "", tv.group(1))
                tool_vid = C.tool_version_for(conn, tname, tv.group(2), sha.group(1) if sha else None)
            started = r["start"]["ts"] if r["start"] else None
            ended = last["ts"] if last else None
            run_id = db.upsert_run(
                conn, f"status:{rel(path)}:{job}:{first['ts_text']}", project, kind, f"{name_file} {job}", r["status"],
                tool_version_id=tool_vid, host_id=host_id, started_at=started or (ended if r["start"] is None else None),
                ended_at=ended, wall_seconds=C.seconds_between(started, ended), config=config, log_path=path,
                notes=("paired from status lines (approximate)" + ("; no terminal line in the file (process state "
                       "UNVERIFIED)" if r["status"] == "unknown" else "")),
                prov=prov, batch_id=batch.id)
            C.set_run_lane(conn, run_id, lane)
            for gid, mapping in _devices(conn, key_line):
                db.add_run_device(conn, run_id, gid, "compute", kernel_path=config["kernel"],
                                  detail={"mapping": mapping, "from": "start line" if r["start"] else "end line"})
            for ev in r["events"]:
                C.put_run_event(conn, run_id, ev["ts"], ev["state"], ev["line"], prov, batch.id)
            for ref in refs:
                ref_dir = ref[:-len("/manifest.json")] if ref.endswith("/manifest.json") else ref.rstrip("/")
                if ref_dir in dv_by_path:
                    C.link_run_dataset(conn, run_id, dv_by_path[ref_dir], "output" if kind == "cache" else "input",
                                       {"from": "path named in status line"})
                if ref_dir in model_by_path:
                    db.link_run_model(conn, run_id, model_by_path[ref_dir], "named")
            batch.wrote()
        for ev in events:
            if ev.get("unattached"):
                C.put_run_event(conn, None, ev["ts"], ev["state"], ev["line"], prov, batch.id)
