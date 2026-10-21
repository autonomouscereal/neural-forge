"""omni_lab_run -- the control side of the Omni Lab runner (psycopg2 only, never torch).

    run        one model x one suite (or a saved test) -> claim cards -> GPU worker -> ingest -> table + pivot
    compare    runs x cases pivot from the database (blank '-' = that run never had that case)
    list       runs | suites | models | tests | cases <suite>
    add-suite  a bank / task file -> a new suite (cases are global; an identical question is one row)
    add-cases  new cases -> the next suite version (old cases carried forward; older runs show '-' for the new ones)
    grade      a reader's grade on one result (human:<name>), artifact-tolerant
    log-run    start / end / metric rows for training, fold and cache jobs (never on their critical path)

Card rules (run): cards are named by nvidia-smi index; each is claimed with flock on
$OC/distill/queues/gpu<N>.lock (the queues' own lock files, non-blocking: a held lock refuses the run),
then refused if it already uses more than --busy-limit MiB (default 2000); GPU5 is refused unless
--allow-gpu5; host MemAvailable must stay above --mem-floor-gib.  The locks are held until the worker exits.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import omni_lab_db as db  # noqa: E402
from ingest import common as C  # noqa: E402
from ingest import evalkit as K  # noqa: E402

GPU_PYTHON = os.environ.get("OMNI_LAB_GPU_PYTHON", "/home/cereal/model-surgery-lab/venv/bin/python")
SPEC_SCHEMA = "omni-lab-run-spec-v1"
DEFAULT_GEN = {"cap": 1024, "thinking": "off", "extra_eos_tokens": ["<|im_end|>", "<|endoftext|>"],
               "ledger_band_mask": True, "seed": 1234}
ENGINE_FOR_LAYOUT = {"hf": "hf", "qwen3_5": "qwen35-student", "chimera.checkpoint.v1": "chimera", "gguf": "gguf"}
GRADE_ABBR = {"correct": "C", "partial": "P", "miss": "M", "none": "N", "pass": "pass", "fail": "FAIL"}
FLAG_ABBR = {"loop": "~loop", "empty": "~empty", "cap-ride": "~cap", "echo": "~echo", "engine-unsupported": "~unsup",
             "oom": "~oom", "grader-error": "~gerr"}


def tool_paths() -> dict[str, str]:
    """Campaign files the worker imports (read at call time so OMNI_LAB_OC / OMNI_LAB_LEXAR overrides apply)."""
    return {
        "calibration_dir": f"{C.OC}/calibration",
        "toolcall_selfgen": f"{C.OC}/distill/tools/toolcall_selfgen.py",
        "student_test": os.environ.get("OMNI_LAB_STUDENT_TEST", f"{C.LX}/scratch/student-tests-20261021/student_test.py"),
        "qf_arch": f"{C.OC}/qwen-fold/qf_arch.py",
        "chimera_runner": f"{C.OC}/composite/tools/run_omni_frozen_review.py",
        "chimera_src": os.environ.get("CHIMERA_OMNI_SRC", f"{C.OC}/import/rq27b/tree_c5e6052"),
        "probe_prompts": f"{C.OC}/calibration/banks/matrix_campaign_probe_prompts_v1.json",
        "recovery_vendor": os.environ.get("OMNI_LAB_RECOVERY_VENDOR",
                                          f"{C.LX}/research/fold/tools/vendor/chimera_recovery"),
    }


def runs_root() -> Path:
    return Path(os.environ.get("OMNI_LAB_RUNS", f"{C.LX}/scratch/omni-lab-runs"))


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def print_table(headers: list[str], rows: list, max_width: int = 60) -> None:
    def cell(v) -> str:
        s = "-" if v is None else str(v)
        s = s.replace("\n", " ")
        return s if len(s) <= max_width else s[: max_width - 1] + "~"
    cells = [[cell(v) for v in r] for r in rows]
    widths = [max([len(h)] + [len(r[i]) for r in cells]) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in cells:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))


# ---------------------------------------------------------------------------
# models / suites / cases
# ---------------------------------------------------------------------------

MODEL_COLS = "id, model_key, layout, path, family, stage, role, project, ledger_checkpoint_id"


def resolve_model(conn, ref: str, create: bool = True) -> dict:
    """A model row by model_key, ledger checkpoint id or path; a path not yet in the DB gets a 'path:<sha16>' row
    (the key model_scan uses) with its layout read off the directory (oc_ledger.layout_of)."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT {MODEL_COLS} FROM model WHERE model_key = %s OR ledger_checkpoint_id = %s "
                    "ORDER BY (model_key = %s) DESC LIMIT 1", (ref, ref, ref))
        row = cur.fetchone()
        if row is None and ("/" in ref or "\\" in ref):
            cur.execute(f"SELECT {MODEL_COLS} FROM model WHERE rtrim(path, '/') = %s "
                        "ORDER BY (ledger_checkpoint_id IS NOT NULL) DESC, id LIMIT 1", (ref.rstrip("/"),))
            row = cur.fetchone()
    if row is None:
        if not create or not ("/" in ref or "\\" in ref) or not os.path.exists(ref):
            raise db.LabError(f"model {ref!r}: not a model_key / ledger id in the database and not an existing path")
        from ingest import oc_ledger as L
        layout = "gguf" if ref.endswith(".gguf") else L.layout_of(ref)
        if os.path.isfile(os.path.join(ref, "student_manifest.json")):
            layout = "qwen3_5"
        key = C.sha16_path_key(ref.rstrip("/"), db.this_host())
        db.upsert_model(conn, key, "chimera-omni", "other", layout, path=ref.rstrip("/"),
                        host_id=db.ensure_host(conn, db.this_host()),
                        notes="registered by omni_lab run (path given on the command line)")
        return resolve_model(conn, key, create=False)
    return dict(zip([c.strip() for c in MODEL_COLS.split(",")], row))


def engine_for(model: dict) -> str:
    path = model.get("path") or ""
    if path and os.path.isfile(os.path.join(path, "student_manifest.json")):
        return "qwen35-student"
    engine = ENGINE_FOR_LAYOUT.get(model.get("layout") or "")
    if engine is None:
        raise db.LabError(f"model {model['model_key']}: layout {model.get('layout')!r} has no engine "
                          "(a raw delta must be composed first, e.g. d1_compose_eval.py); pass --engine to force one")
    return engine


def parse_suite_ref(ref: str) -> tuple[str, str | None]:
    if "@" in ref:
        name, version = ref.split("@", 1)
        return name, version or None
    return ref, None


def suite_row(conn, ref: str) -> dict:
    name, version = parse_suite_ref(ref)
    sid = db.suite_id(conn, name, version)
    if sid is None:
        raise db.LabError(f"suite {ref!r} not found (omni_lab.py list suites)")
    with conn.cursor() as cur:
        cur.execute("SELECT id, name, version, status, default_grader, bank_sha256, modality_in, modality_out "
                    "FROM test_suite WHERE id = %s", (sid,))
        r = cur.fetchone()
    return dict(zip(("id", "name", "version", "status", "default_grader", "bank_sha256", "modality_in", "modality_out"), r))


def suite_case_rows(conn, suite_pk: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT tc.id, tc.case_id, tc.content_sha256, tc.category, tc.lane, tc.prompt, tc.tools, tc.media, "
                    "tc.reference, COALESCE(tc.grader, ts.default_grader), tc.modality_in, tc.modality_out, sm.ordinal "
                    "FROM suite_member sm JOIN test_case tc ON tc.id = sm.case_pk JOIN test_suite ts ON ts.id = sm.suite_id "
                    "WHERE sm.suite_id = %s ORDER BY sm.ordinal", (suite_pk,))
        cols = ("case_pk", "case_id", "content_sha256", "category", "lane", "prompt", "tools", "media", "reference",
                "grader", "modality_in", "modality_out", "ordinal")
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def filter_cases(cases: list[dict], patterns: list[str] | None, category: str | None, limit: int | None) -> list[dict]:
    out = cases
    if patterns:
        out = [c for c in out if any(fnmatch.fnmatchcase(c["case_id"], p) for p in patterns)]
    if category:
        out = [c for c in out if fnmatch.fnmatchcase(c.get("category") or "", category)]
    if limit:
        out = out[:limit]
    return out


# ---------------------------------------------------------------------------
# cards
# ---------------------------------------------------------------------------

def nvidia_smi() -> dict[int, dict]:
    fields = "index,uuid,name,memory.used,memory.total,compute_cap,driver_version"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise db.LabError(f"nvidia-smi failed: {exc}") from exc
    cards = {}
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 7:
            continue
        cards[int(p[0])] = {"index": int(p[0]), "uuid": p[1], "name": p[2], "used_mib": int(float(p[3])),
                            "vram_mib": int(float(p[4])), "sm": p[5], "driver": p[6]}
    return cards


def mem_available_gib() -> float | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        return None
    return None


def check_cards(smi: dict[int, dict], indexes: list[int], *, allow_gpu5: bool, busy_limit: int) -> list[dict]:
    """The card rules without the locks (dry-run uses this alone): GPU5 only with --allow-gpu5, every card
    present in nvidia-smi and using no more than busy_limit MiB.  Raises LabError on the first violation."""
    if 5 in indexes and not allow_gpu5:
        raise db.LabError("GPU5 is never used unless told: pass --allow-gpu5")
    if len(set(indexes)) != len(indexes):
        raise db.LabError(f"a card is named twice: {indexes}")
    cards = []
    for i in indexes:
        if i not in smi:
            raise db.LabError(f"GPU{i} is not in nvidia-smi")
        if smi[i]["used_mib"] > busy_limit:
            raise db.LabError(f"GPU{i} {smi[i]['name']} uses {smi[i]['used_mib']} MiB > {busy_limit}: not idle")
        cards.append(smi[i])
    return cards


def claim_cards(indexes: list[int], *, allow_gpu5: bool, busy_limit: int) -> tuple[list[dict], list[Any]]:
    """flock each $Q/gpu<N>.lock (non-blocking) and check the card is idle; returns (cards, open lock files).
    Refuses (LabError) on the first card that is locked, busy, missing, or GPU5 without --allow-gpu5."""
    import fcntl
    check_cards(nvidia_smi(), indexes, allow_gpu5=allow_gpu5, busy_limit=busy_limit)
    held: list = []
    try:
        for i in indexes:
            fh = open(f"{C.Q}/gpu{i}.lock", "a")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                raise db.LabError(f"gpu{i}.lock is held by another job ({C.Q}/gpu{i}.lock)") from None
            held.append(fh)
        cards = check_cards(nvidia_smi(), indexes, allow_gpu5=allow_gpu5, busy_limit=busy_limit)  # re-read under the locks
    except BaseException:
        for fh in held:
            fh.close()
        raise
    return cards, held


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def code_shas() -> dict[str, str]:
    files = [HERE / "omni_lab.py", HERE / "omni_lab_run.py", HERE / "omni_lab_db.py", HERE / "omni_lab_worker.py",
             HERE / "graders" / "registry.py", HERE / "ingest" / "lab_runs.py"] + sorted((HERE / "adapters").glob("*.py"))
    return {str(p.relative_to(HERE)).replace("\\", "/"): db.file_sha256(p) for p in files if p.is_file()}


def tool_pin_rows(conn, paths: dict[str, str]) -> dict[str, dict]:
    """sha256 of each campaign file the worker will import, and the TOOL_PINS row (tool_version) it matches."""
    out = {}
    files = {"calibration_dir": "oc_calib.py", "recovery_vendor": "evaluation.py"}
    for key, p in paths.items():
        if key in ("chimera_src", "probe_prompts"):
            continue
        f = Path(p) / files[key] if key in files else Path(p)
        if not f.is_file():
            out[key] = {"path": str(f), "sha256": None, "present": False}
            continue
        sha = db.file_sha256(f)
        with conn.cursor() as cur:
            cur.execute("SELECT t.name, tv.version, tv.pinned FROM tool_version tv JOIN tool t ON t.id = tv.tool_id "
                        "WHERE tv.sha256 = %s ORDER BY tv.pinned DESC LIMIT 1", (sha,))
            hit = cur.fetchone()
        out[key] = {"path": str(f), "sha256": sha, "pin": {"tool": hit[0], "version": hit[1], "pinned": hit[2]} if hit else None}
    return out


def default_tokenizer_dir(conn, model: dict) -> str | None:
    """The model's own directory when it carries a tokenizer, else the tokenizer the newest earlier eval of this
    model recorded (the student export has none; student_test used the S3 directory)."""
    path = model.get("path") or ""
    if path and os.path.isfile(os.path.join(path, "tokenizer.json")):
        return path
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(r.config->>'tokenizer_directory', r.config->>'tokenizer_dir') FROM eval_run er "
                    "JOIN run r ON r.id = er.run_id WHERE er.model_id = %s AND (r.config ? 'tokenizer_directory' "
                    "OR r.config ? 'tokenizer_dir') ORDER BY r.started_at DESC NULLS LAST LIMIT 1", (model["id"],))
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def auto_baselines(conn, run_id: int, model_id: int, case_pks: list[int], limit: int = 2) -> list[int]:
    """Up to `limit` earlier eval runs that share cases with this one: a lineage parent's newest run first,
    then this model's own previous run, then the newest run of any other model."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT er.run_id, er.model_id FROM eval_run er JOIN run r ON r.id = er.run_id "
            "WHERE er.run_id <> %s AND EXISTS (SELECT 1 FROM eval_result res WHERE res.eval_run_id = er.run_id "
            "AND res.case_pk = ANY(%s)) "
            "ORDER BY (er.model_id IN (SELECT parent_id FROM model_lineage WHERE child_id = %s)) DESC, "
            "(er.model_id = %s) DESC, r.started_at DESC NULLS LAST, er.run_id DESC LIMIT 20",
            (run_id, case_pks, model_id, model_id))
        rows = cur.fetchall()
    out, seen_models = [], set()
    for rid, mid in rows:
        if mid in seen_models:
            continue
        seen_models.add(mid)
        out.append(rid)
        if len(out) >= limit:
            break
    return out


def build_spec(conn, args) -> dict:
    """Everything the worker and the ingest need, resolved from the database and the command line."""
    test = None
    if args.test:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, suite_name, suite_version, engine, gen, devices, case_filter FROM test "
                        "WHERE name = %s", (args.test,))
            r = cur.fetchone()
        if r is None and not args.save_test:
            raise db.LabError(f"test {args.test!r} not found (omni_lab.py list tests)")
        if r is not None:
            test = dict(zip(("id", "name", "suite_name", "suite_version", "engine", "gen", "devices", "case_filter"), r))
    suite_ref = args.suite or (test and (test["suite_name"] + ("@" + test["suite_version"] if test["suite_version"] else "")))
    if not suite_ref:
        raise db.LabError("name a --suite (or a --test that names one)")
    if not args.model:
        raise db.LabError("name a --model (model_key, ledger checkpoint id or path)")
    model = resolve_model(conn, args.model)
    suite = suite_row(conn, suite_ref)
    engine = args.engine or (test["engine"] if test and test["engine"] != "auto" else None) or engine_for(model)
    gen = dict(DEFAULT_GEN)
    if test:
        gen.update(test["gen"] or {})
    for key in ("cap", "thinking", "seed"):
        if getattr(args, key, None) is not None:
            gen[key] = getattr(args, key)
    if args.no_band_mask:
        gen["ledger_band_mask"] = False
    dtype = args.dtype or gen.pop("dtype", None) or ("float32" if args.cpu else "bfloat16")
    gen.pop("dtype", None)
    patterns = [p for p in (args.cases or "").split(",") if p] or (test["case_filter"] if test else None)
    cases = filter_cases(suite_case_rows(conn, suite["id"]), patterns, args.category, args.limit)
    if not cases:
        raise db.LabError(f"no cases selected from {suite['name']}@{suite['version']}")
    devices = [int(x) for x in args.gpus.split(",") if x.strip()] if args.gpus else (
        list(test["devices"]) if test and test["devices"] and not args.cpu else [])
    if engine not in ("echo", "gguf") and not devices and not args.cpu:
        raise db.LabError("name cards with --gpus 4,0 (nvidia-smi indexes) or run on --cpu")
    tok_dir = args.tokenizer_dir or default_tokenizer_dir(conn, model)
    paths = tool_paths()
    stamp = utc_stamp()
    short = re.sub(r"[^A-Za-z0-9.-]+", "-", model["model_key"])[-40:]
    core = {"model": model["model_key"], "suite": [suite["name"], suite["version"]], "engine": engine, "gen": gen,
            "dtype": dtype, "cases": [c["case_pk"] for c in cases], "devices": devices, "cpu": bool(args.cpu), "t": stamp}
    run_key = f"eval:omni_lab:{stamp}:{db.sha256_json(core)[:12]}"
    name = args.name or f"{short} {suite['name']}@{suite['version']} {engine}" + (f" [{args.test}]" if args.test else "")
    shas = code_shas()
    spec = {
        "schema": SPEC_SCHEMA, "run_key": run_key, "name": name, "created_utc": stamp, "host": db.this_host(),
        "lab_dir": str(HERE), "owner": args.owner, "notes": args.notes, "project": model.get("project") or "chimera-omni",
        "model": {k: model.get(k) for k in ("model_key", "layout", "path", "family", "stage", "role", "project")},
        "model_path": model.get("path"), "tokenizer_dir": tok_dir, "engine": engine, "dtype": dtype,
        "label": args.label or f"LAB-{short}-{suite['name']}", "loader_class": args.loader_class,
        "suite": {"name": suite["name"], "version": suite["version"], "bank_sha256": suite["bank_sha256"],
                  "default_grader": suite["default_grader"]},
        "case_filter": patterns, "test_name": args.test if (test or args.save_test) else None,
        "gen": gen,
        "device": {"kind": "server" if engine == "gguf" else "cpu" if args.cpu or (engine == "echo" and not devices)
                   else "cuda", "cards": [], "threads": args.threads, "split": args.split, "server_url": args.server_url},
        "paths": paths, "code_sha256": shas, "code_commit": "lab:" + db.sha256_json(shas)[:12],
        "worker_version": "1.0.0",
        "cases": [{k: c[k] for k in ("case_pk", "case_id", "content_sha256", "category", "lane", "prompt", "tools", "media",
                                     "reference", "grader", "modality_in", "modality_out")} for c in cases],
    }
    spec["_devices_requested"] = devices
    return spec


def write_pre_run(conn, spec: dict, model_id: int) -> int:
    """The run row exists from the moment the worker starts (status running), so a crash is still visible."""
    sid = db.suite_id(conn, spec["suite"]["name"], spec["suite"]["version"])
    rid = db.upsert_run(conn, spec["run_key"], spec["project"], "eval", spec["name"], "running", model_id_=model_id,
                        host_id=db.ensure_host(conn, spec["host"]) if spec["host"] != C.SERVER_HOST else C.host_id(conn),
                        started_at=datetime.now(timezone.utc), config=spec["config"], run_dir=spec["run_dir"],
                        log_path=spec["run_dir"] + "/worker.log", owner=spec.get("owner"), notes=spec.get("notes"))
    db.upsert_eval_run(conn, rid, model_id, sid, spec["engine"], gen=spec["gen"], label=spec["label"],
                       thinking_requested=spec["gen"]["thinking"], case_filter=spec["case_filter"],
                       bank_file_sha256=spec["suite"]["bank_sha256"])
    return rid


def worker_command(spec_path: Path, run_dir: Path, nice: int | None, python: str) -> list[str]:
    cmd = [python, str(HERE / "omni_lab_worker.py"), "--spec", str(spec_path), "--run-dir", str(run_dir)]
    if nice and os.name != "nt":
        cmd = ["nice", "-n", str(int(nice))] + cmd
    return cmd


def worker_env(spec: dict) -> dict:
    env = dict(os.environ)
    cards = spec["device"]["cards"]
    env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": ",".join(str(c["index"]) for c in cards) if spec["device"]["kind"] == "cuda" else "",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TRITON_CACHE_DIR": str(runs_root() / ".triton_cache")})
    if spec["device"].get("threads"):
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = str(spec["device"]["threads"])
    return env


def cmd_run(args) -> int:
    t0 = time.monotonic()
    conn = db.connect("rw", app="run")
    try:
        return _run(conn, args, t0)
    finally:
        conn.close()


def _run(conn, args, t0: float) -> int:
    held: list = []
    try:
        spec = build_spec(conn, args)
        devices = spec.pop("_devices_requested")
        model_id = db.model_id(conn, spec["model"]["model_key"])
        if args.save_test:
            tid = db.upsert_test(conn, args.save_test, spec["suite"]["name"], spec["engine"],
                                 gen={**spec["gen"], "dtype": spec["dtype"]}, suite_version=spec["suite"]["version"],
                                 devices=devices or None, case_filter=spec["case_filter"],
                                 notes=f"saved by omni_lab run {spec['created_utc']}")
            spec["test_name"] = args.save_test
            conn.commit()
            print(f"saved test {args.save_test!r} (id {tid}): {spec['suite']['name']}@{spec['suite']['version']} "
                  f"engine {spec['engine']} gen {spec['gen']} cards {devices or '-'}")
        floor = args.mem_floor_gib
        avail = mem_available_gib()
        if avail is not None and avail < floor and spec["engine"] != "echo":
            raise db.LabError(f"MemAvailable {avail:.1f} GiB < floor {floor} GiB")
        if spec["device"]["kind"] == "cuda":
            if args.dry_run:
                spec["device"]["cards"] = check_cards(nvidia_smi(), devices, allow_gpu5=args.allow_gpu5,
                                                      busy_limit=args.busy_limit)
            else:
                spec["device"]["cards"], held = claim_cards(devices, allow_gpu5=args.allow_gpu5, busy_limit=args.busy_limit)
            spec["device"]["driver"] = (spec["device"]["cards"] or [{}])[0].get("driver")
        run_dir = runs_root() / f"{spec['created_utc']}-{spec['engine']}-{re.sub(r'[^A-Za-z0-9.-]+', '-', spec['model']['model_key'])[-32:]}-{spec['suite']['name']}"
        spec["run_dir"] = str(run_dir)
        spec["config"] = {
            "engine": spec["engine"], "dtype": spec["dtype"], "argv": sys.argv, "device_kind": spec["device"]["kind"],
            "cards": [{k: c.get(k) for k in ("index", "uuid", "name", "sm", "vram_mib", "used_mib")} for c in spec["device"]["cards"]],
            "threads": spec["device"].get("threads"), "split": spec["device"].get("split"),
            "server_url": spec["device"].get("server_url"), "gen": spec["gen"], "model_path": spec["model_path"],
            "tokenizer_dir": spec["tokenizer_dir"], "suite": f"{spec['suite']['name']}@{spec['suite']['version']}",
            "n_cases": len(spec["cases"]), "case_filter": spec["case_filter"], "test": spec["test_name"],
            "lab_code": spec["code_commit"], "tool_pins": tool_pin_rows(conn, spec["paths"]),
            "gpu_python": args.gpu_python, "nice": args.nice, "mem_available_gib_at_start": round(avail, 1) if avail else None,
        }
        cmd = worker_command(run_dir / "spec.json", run_dir, args.nice, args.gpu_python)
        print(f"run      {spec['name']}\nmodel    {spec['model']['model_key']}  ({spec['model']['layout']}, {spec['model_path']})\n"
              f"suite    {spec['suite']['name']}@{spec['suite']['version']}  {len(spec['cases'])} cases\n"
              f"engine   {spec['engine']}  dtype {spec['dtype']}  gen {json.dumps(spec['gen'])}\n"
              f"device   {spec['device']['kind']} " + ", ".join(f"GPU{c['index']} {c['name']} {c['uuid']} ({c['used_mib']} MiB used)"
                                                           for c in spec["device"]["cards"]) +
              f"\ntokenizer {spec['tokenizer_dir']}\nrun dir  {run_dir}")
        if args.dry_run:
            print("worker   " + " ".join(shlex.quote(x) for x in cmd))
            print("[dry-run] nothing claimed, written or started")
            return 0
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False, default=str) + "\n",
                                           encoding="utf-8")
        run_id = write_pre_run(conn, spec, model_id)
        conn.commit()
        print(f"run id   {run_id}   ({spec['run_key']})\n")
        exit_code, stopped = launch_worker(cmd, spec, run_dir)
    finally:
        for fh in held:
            fh.close()
    run_id = ingest_run_dir(conn, run_dir, exit_code, stopped)
    print_run_table(conn, run_id)
    baselines = [] if args.baseline == "none" else (
        [int(x) for x in args.baseline.split(",")] if args.baseline else
        auto_baselines(conn, run_id, model_id, [c["case_pk"] for c in spec["cases"]]))
    print()
    compare(conn, baselines + [run_id], show=args.show, lead_run=run_id,
            case_pks={c["case_pk"] for c in spec["cases"]})
    print(f"\n({time.monotonic() - t0:.1f} s total; worker exit {exit_code}; run dir {run_dir})")
    return 0 if exit_code == 0 else 1


def launch_worker(cmd: list[str], spec: dict, run_dir: Path) -> tuple[int, bool]:
    """Run the worker, tee its console to worker.log; Ctrl-C stops it and the run is recorded as stopped."""
    stopped = False
    with open(run_dir / "worker.log", "w", encoding="utf-8") as log, \
            subprocess.Popen(cmd, env=worker_env(spec), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1) as proc:
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            stopped = True
            proc.terminate()
            try:
                code = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait()
            print("\n[stopped] worker terminated by Ctrl-C", flush=True)
    (run_dir / "worker.exit").write_text(f"{code}\n", encoding="utf-8")
    return code, stopped


def ingest_run_dir(conn, run_dir: Path, exit_code: int | None, stopped: bool = False) -> int:
    from ingest import lab_runs
    batch_id = db.start_batch(conn, lab_runs.SOURCE, parser="ingest/lab_runs.py",
                              parser_sha256=db.file_sha256(HERE / "ingest" / "lab_runs.py"), argv=list(sys.argv))
    conn.commit()
    batch = C.Batch(batch_id, lab_runs.SOURCE)
    try:
        K.seed_host(conn)
        rid = lab_runs.ingest_dir(conn, batch, str(run_dir), exit_code)
        if rid is None:
            raise db.LabError(f"{run_dir}: nothing ingested ({'; '.join(batch.notes)})")
        if stopped:
            with conn.cursor() as cur:
                cur.execute("UPDATE run SET status = 'stopped' WHERE id = %s", (rid,))
        db.end_batch(conn, batch_id, "ok", batch.seen, batch.written, batch.skipped,
                     error="; ".join(batch.notes[:20]) or None, file_shas=batch.file_shas)
        conn.commit()
        return rid
    except BaseException as exc:
        conn.rollback()
        db.end_batch(conn, batch_id, "failed", batch.seen, 0, batch.skipped, error=f"{type(exc).__name__}: {exc}")
        conn.commit()
        raise


def print_run_table(conn, run_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT tc.case_id, res.output_tokens, res.termination, res.form, eg.grade, eg.human_grade IS NOT NULL, "
                    "array_to_string(res.artifact_flags, ','), res.latency_ms, round(res.tok_s::numeric, 2), "
                    "left(regexp_replace(COALESCE(res.output_text, ''), '\\s+', ' ', 'g'), 70) "
                    "FROM eval_result res JOIN test_case tc ON tc.id = res.case_pk "
                    "JOIN v_effective_grade eg ON eg.eval_result_id = res.id "
                    "LEFT JOIN suite_member sm ON sm.case_pk = tc.id AND sm.suite_id = (SELECT suite_id FROM eval_run WHERE run_id = %s) "
                    "WHERE res.eval_run_id = %s ORDER BY sm.ordinal NULLS LAST, tc.case_id", (run_id, run_id))
        rows = cur.fetchall()
        cur.execute("SELECT test, model_key, suite, engine, devices, env, correct, partial, miss, \"none\", pass, fail, "
                    "ungraded, unsupported, n_results, n_cases, status FROM v_results WHERE eval_run_id = %s", (run_id,))
        summary = cur.fetchone()
    print_table(["case", "tok", "term", "form", "grade", "flags", "ms", "tok/s", "response"],
                [(r[0], r[1], r[2], r[3], (r[4] or "ungraded") + ("*" if r[5] else ""), r[6] or "", r[7], r[8], r[9])
                 for r in rows], max_width=70)
    if summary:
        (test, mk, suite, engine, devices, env, c, p, m, n, ps, f, ug, us, nr, nc, st) = summary
        print(f"\nrun {run_id} [{st}] {mk} on {suite} via {engine}; devices {devices or 'cpu'}; {env or ''}")
        print(f"results {nr}/{nc}: correct {c} partial {p} miss {m} none {n} | pass {ps} fail {f} | ungraded {ug} "
              f"| unsupported {us}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def select_runs(conn, args) -> list[int]:
    if args.runs:
        return [int(x) for r in args.runs for x in str(r).split(",") if x]
    where, params = ["true"], []
    if args.model:
        where.append("(m.model_key = ANY(%s) OR m.ledger_checkpoint_id = ANY(%s))")
        params += [args.model, args.model]
    if args.engine:
        where.append("er.engine = %s")
        params.append(args.engine)
    if args.test:
        where.append("t.name = %s")
        params.append(args.test)
    if args.suite:
        name, version = parse_suite_ref(args.suite)
        where.append("EXISTS (SELECT 1 FROM eval_result res JOIN suite_member sm ON sm.case_pk = res.case_pk "
                     "JOIN test_suite ts2 ON ts2.id = sm.suite_id WHERE res.eval_run_id = er.run_id AND ts2.name = %s"
                     + (" AND ts2.version = %s" if version else "") + ")")
        params += [name] + ([version] if version else [])
    with conn.cursor() as cur:
        cur.execute("SELECT er.run_id FROM eval_run er JOIN run r ON r.id = er.run_id JOIN model m ON m.id = er.model_id "
                    "LEFT JOIN test t ON t.id = er.test_id WHERE " + " AND ".join(where) +
                    " ORDER BY r.started_at DESC NULLS LAST, er.run_id DESC LIMIT %s", params + [args.last])
        ids = [r[0] for r in cur.fetchall()]
    return list(reversed(ids))


def compare(conn, run_ids: list[int], *, suite: str | None = None, case_patterns: list[str] | None = None,
            show: str = "grade", width: int = 40, lead_run: int | None = None,
            case_pks: set[int] | None = None) -> None:
    """Pivot: one row per case, one column per run.  '-' = that run has no result for the case (it never ran it,
    e.g. a case added after the run); '?' = ran, nobody graded it yet (machine flags follow '~')."""
    t0 = time.monotonic()
    run_ids = list(dict.fromkeys(run_ids))
    if not run_ids:
        print("(no runs selected)")
        return
    with conn.cursor() as cur:
        cur.execute("SELECT er.run_id, m.model_key, m.stage, ts.name || '@' || ts.version, er.engine, "
                    "to_char(r.started_at, 'YYYY-MM-DD'), COALESCE(t.name, er.label), r.status, er.suite_id "
                    "FROM eval_run er JOIN run r ON r.id = er.run_id JOIN model m ON m.id = er.model_id "
                    "JOIN test_suite ts ON ts.id = er.suite_id LEFT JOIN test t ON t.id = er.test_id "
                    "WHERE er.run_id = ANY(%s)", (run_ids,))
        meta = {r[0]: r for r in cur.fetchall()}
        missing = [r for r in run_ids if r not in meta]
        if missing:
            raise db.LabError(f"not eval runs: {missing}")
        if suite:
            name, version = parse_suite_ref(suite)
            cur.execute("SELECT id FROM test_suite WHERE name = %s" + (" AND version = %s" if version else "") +
                        " ORDER BY created_at DESC, id DESC", [name] + ([version] if version else []))
            suite_ids = [r[0] for r in cur.fetchall()]
        else:
            lead = [meta[lead_run][8]] if lead_run in meta else []
            suite_ids = list(dict.fromkeys(lead + [meta[rid][8] for rid in run_ids]))
        cur.execute("SELECT sm.suite_id, tc.id, tc.case_id, tc.category, sm.ordinal FROM suite_member sm "
                    "JOIN test_case tc ON tc.id = sm.case_pk WHERE sm.suite_id = ANY(%s)", (suite_ids,))
        members = cur.fetchall()
        cur.execute("SELECT res.eval_run_id, res.case_pk, eg.grade, eg.human_grade IS NOT NULL, res.termination, "
                    "res.artifact_flags, res.output_tokens, "
                    "left(regexp_replace(COALESCE(res.output_text, ''), '\\s+', ' ', 'g'), 60) "
                    "FROM eval_result res JOIN v_effective_grade eg ON eg.eval_result_id = res.id "
                    "WHERE res.eval_run_id = ANY(%s)", (run_ids,))
        cells = {(r[0], r[1]): r for r in cur.fetchall()}
        # row order: the first run's suite order, then the next suites' (a --suite ref: newest version first)
        rank = {sid: i for i, sid in enumerate(suite_ids)}
        best: dict[int, tuple] = {}
        for sid, pk, cid, cat, ordinal in members:
            key = (rank[sid], ordinal, cid, cat)
            if pk not in best or key < best[pk]:
                best[pk] = key
        extra = [] if suite else sorted({k[1] for k in cells} - set(best))
        if extra:          # a result on a case outside the runs' suites still gets its row
            cur.execute("SELECT id, case_id, category FROM test_case WHERE id = ANY(%s)", (extra,))
            for pk, cid, cat in cur.fetchall():
                best[pk] = (len(rank), 0, cid, cat)
    cases = sorted(((pk, k[2], k[3]) for pk, k in best.items()), key=lambda c: best[c[0]][:3])
    if case_patterns:
        cases = [c for c in cases if any(fnmatch.fnmatchcase(c[1], p) for p in case_patterns)]
    if case_pks is not None:
        cases = [c for c in cases if c[0] in case_pks]
    dup: dict[str, int] = {}
    for c in cases:
        dup[c[1]] = dup.get(c[1], 0) + 1

    def render(cell) -> str:
        if cell is None:
            return "-"
        _, _, grade, human, term, flags, toks, text = cell
        if show == "tokens":
            return f"{toks}/{term or '?'}"
        if show == "text":
            return text or "''"
        if show == "term":
            return term or "?"
        base = GRADE_ABBR.get(grade, grade) if grade else ("unsup" if term == "unsupported-modality" else "?")
        if human:
            base += "*"
        marks = "".join(FLAG_ABBR.get(f, "") for f in (flags or []) if f in FLAG_ABBR)
        return base + marks

    print("runs:")
    print_table(["col", "run", "model", "stage", "suite", "engine", "date", "test/label", "status"],
                [(f"r{rid}", rid, *meta[rid][1:8]) for rid in run_ids], max_width=42)
    print()
    headers = ["case", "category"] + [f"r{rid}" for rid in run_ids]
    body = []
    tallies = {rid: {} for rid in run_ids}
    for c in cases:
        label = c[1] if dup[c[1]] == 1 else f"{c[1]}#{c[0]}"
        line = [label, c[2]]
        for rid in run_ids:
            cell = cells.get((rid, c[0]))
            s = render(cell)
            line.append(s)
            key = "-" if cell is None else (cell[2] or ("unsup" if cell[4] == "unsupported-modality" else "?"))
            tallies[rid][key] = tallies[rid].get(key, 0) + 1
        body.append(line)
    print_table(headers, body, max_width=max(width, 40) if show != "text" else 60)
    print()
    order_keys = ["correct", "partial", "miss", "none", "pass", "fail", "?", "unsup", "-"]
    tot = []
    for rid in run_ids:
        t = tallies[rid]
        tot.append(" ".join(f"{GRADE_ABBR.get(k, k)}={t[k]}" for k in order_keys if t.get(k)))
    print_table(["col", "totals (C/P/M/N reader grades, pass/FAIL machine, ?=ungraded, unsup, -=never ran)"],
                [(f"r{rid}", s) for rid, s in zip(run_ids, tot)], max_width=120)
    print(f"({len(cases)} cases x {len(run_ids)} runs, {time.monotonic() - t0:.2f} s query+render; * = human grade)")


def cmd_compare(args) -> int:
    t0 = time.monotonic()
    conn = db.connect("ro", app="compare")
    try:
        ids = select_runs(conn, args)
        compare(conn, ids, suite=args.suite, case_patterns=[p for p in (args.cases or "").split(",") if p] or None,
                show=args.show, width=args.width)
        print(f"({time.monotonic() - t0:.2f} s including connect)")
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def cmd_list(args) -> int:
    conn = db.connect("ro", app="list")
    try:
        with conn.cursor() as cur:
            if args.what == "suites":
                cur.execute("SELECT ts.name, ts.version, ts.status, ts.default_grader, "
                            "(SELECT count(*) FROM suite_member sm WHERE sm.suite_id = ts.id), "
                            "(SELECT count(*) FROM eval_run er WHERE er.suite_id = ts.id), ts.created_at::date, "
                            "left(ts.description, 70) FROM test_suite ts ORDER BY ts.name, ts.created_at")
                print_table(["suite", "version", "status", "default grader", "cases", "runs", "created", "description"],
                            cur.fetchall(), max_width=70)
            elif args.what == "tests":
                cur.execute("SELECT t.name, t.suite_name || COALESCE('@' || t.suite_version, ''), t.engine, t.gen::text, "
                            "t.devices::text, t.case_filter::text, (SELECT count(*) FROM eval_run er WHERE er.test_id = t.id), "
                            "t.created_at::date FROM test t ORDER BY t.name")
                print_table(["test", "suite", "engine", "gen", "cards", "cases", "runs", "created"], cur.fetchall(), 70)
            elif args.what == "models":
                like = f"%{args.filter}%" if args.filter else "%"
                cur.execute("SELECT m.model_key, m.role, m.layout, m.family, m.stage, m.status, "
                            "(SELECT count(*) FROM eval_run er WHERE er.model_id = m.id), m.path FROM model m "
                            "WHERE (m.model_key ILIKE %s OR m.path ILIKE %s OR m.family ILIKE %s OR m.stage ILIKE %s) "
                            "AND (%s OR m.ledger_checkpoint_id IS NOT NULL OR m.role IN ('parent','student','composite') "
                            "OR EXISTS (SELECT 1 FROM eval_run er WHERE er.model_id = m.id)) "
                            "ORDER BY m.project, m.family, m.stage, m.model_key LIMIT %s",
                            (like, like, like, like, bool(args.filter) or args.all, args.limit))
                print_table(["model", "role", "layout", "family", "stage", "status", "evals", "path"], cur.fetchall(), 70)
            elif args.what == "cases":
                if not args.filter:
                    raise db.LabError("list cases <suite[@version]>")
                s = suite_row(conn, args.filter)
                cur.execute("SELECT sm.ordinal, tc.case_id, tc.category, COALESCE(tc.grader, %s), tc.modality_in::text, "
                            "left(regexp_replace(CASE WHEN jsonb_typeof(tc.prompt) = 'array' THEN tc.prompt->-1->>'content' "
                            "ELSE tc.prompt::text END, '\\s+', ' ', 'g'), 80) FROM suite_member sm "
                            "JOIN test_case tc ON tc.id = sm.case_pk WHERE sm.suite_id = %s ORDER BY sm.ordinal",
                            (s["default_grader"], s["id"]))
                print(f"{s['name']}@{s['version']} [{s['status']}]")
                print_table(["#", "case", "category", "grader", "in", "prompt (last turn)"], cur.fetchall(), 80)
            else:   # runs
                cur.execute("SELECT r.id, r.kind, r.status, r.name, m.model_key, r.started_at::timestamp(0), "
                            "round(r.wall_seconds::numeric), (SELECT string_agg(g.gpu_index::text, ',' ORDER BY g.gpu_index) "
                            "FROM run_device rd JOIN gpu g ON g.id = rd.gpu_id WHERE rd.run_id = r.id) "
                            "FROM run r LEFT JOIN model m ON m.id = r.model_id "
                            "WHERE (%s IS NULL OR r.kind = %s) AND (%s IS NULL OR r.name ILIKE %s OR m.model_key ILIKE %s) "
                            "ORDER BY r.started_at DESC NULLS LAST, r.id DESC LIMIT %s",
                            (args.kind, args.kind, args.filter, f"%{args.filter}%", f"%{args.filter}%", args.limit))
                print_table(["id", "kind", "status", "name", "model", "started", "wall s", "gpus"], cur.fetchall(), 60)
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# add-suite / add-cases
# ---------------------------------------------------------------------------

def _messages(obj: dict) -> Any:
    if isinstance(obj.get("messages"), list):
        return obj["messages"]
    if isinstance(obj.get("input"), dict) and isinstance(obj["input"].get("messages"), list):
        return obj["input"]["messages"]
    if isinstance(obj.get("raw_text"), str):
        return {"raw_text": obj["raw_text"]}
    if isinstance(obj.get("prompt"), str):
        return [{"role": "user", "content": obj["prompt"]}]
    raise db.LabError(f"case {obj.get('id')!r}: no messages / prompt / raw_text / input.messages")


def parse_case_file(path: Path, prefix: str) -> list[dict]:
    """Bank and task layouts already in the estate -> case dicts:
       {"prompts": [...]} (bank v1/v2), {"cases": [...]} (toolcall bank), a JSON list of objects or strings,
       or JSONL tasks (Codex recovery: id, input{messages,tools,media}, oracle).
    Per object: id, category, lane, prompt|messages|raw_text|input, tools, media, expected, asserts, oracle,
    context, grader, modality_in, modality_out, tags."""
    text = path.read_text(encoding="utf-8-sig")
    try:
        doc = json.loads(text)
    except ValueError:
        doc = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(doc, dict):
        rows = doc.get("prompts") if isinstance(doc.get("prompts"), list) else doc.get("cases")
        if not isinstance(rows, list):
            raise db.LabError(f"{path}: no 'prompts' or 'cases' list")
    elif isinstance(doc, list):
        rows = doc
    else:
        raise db.LabError(f"{path}: not a JSON object, list or JSONL")
    out = []
    for n, row in enumerate(rows, 1):
        if isinstance(row, str):
            row = {"id": f"{prefix}-{n}", "prompt": row}
        if not isinstance(row, dict):
            raise db.LabError(f"{path}: row {n} is not an object or a string")
        cid = str(row.get("id") or f"{prefix}-{n}")
        tools = row.get("tools") or (row.get("input") or {}).get("tools") or None
        media = row.get("media") or (row.get("input") or {}).get("media") or None
        reference = {k: row[k] for k in ("expected", "asserts", "oracle", "context", "evaluation_scope", "echo_response")
                     if row.get(k) is not None}
        grader = row.get("grader")
        if grader is None:
            if tools and isinstance(row.get("expected"), dict):
                grader = "toolcall_selfgen.grade_case"
            elif row.get("asserts"):
                grader = "student_test.code_check"
            elif row.get("oracle"):
                grader = "chimera_recovery.evaluate"
        modality_in = row.get("modality_in") or (sorted({"text"} | {str(m.get("kind")) for m in media
                                                                    if isinstance(m, dict) and m.get("kind")})
                                                 if isinstance(media, list) else ["text"])
        out.append({"case_id": cid, "prompt": _messages(row), "tools": tools, "media": media,
                    "reference": reference or None, "category": row.get("category"), "lane": row.get("lane"),
                    "grader": grader, "tags": list(row.get("tags") or []), "modality_in": modality_in,
                    "modality_out": row.get("modality_out") or ["text"]})
    ids = [c["case_id"] for c in out]
    if len(set(ids)) != len(ids):
        raise db.LabError(f"{path}: duplicate case ids")
    return out


def next_version(conn, name: str) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM test_suite WHERE name = %s", (name,))
        nums = [int(v) for (v,) in cur.fetchall() if re.fullmatch(r"\d+", v)]
    return str(max(nums) + 1) if nums else "1"


def _write_cases(conn, batch_id: int, cases: list[dict], prov) -> list[int]:
    for c in cases:
        if c.get("grader") and c["grader"] in K.GRADERS:
            K.grader_id(conn, c["grader"])
    return [db.ensure_case(conn, c["case_id"], c["prompt"], tools=c["tools"], media=c["media"], reference=c["reference"],
                           category=c["category"], lane=c["lane"], tags=c["tags"], modality_in=c["modality_in"],
                           modality_out=c["modality_out"], grader=c["grader"], prov=prov, batch_id=batch_id)
            for c in cases]


def cmd_add_suite(args) -> int:
    path = Path(args.file)
    cases = parse_case_file(path, args.name)
    conn = db.connect("rw", app="add-suite")
    try:
        version = args.version or next_version(conn, args.name)
        if db.suite_id(conn, args.name, version) is not None:
            raise db.LabError(f"suite {args.name}@{version} exists; suites are immutable (add-cases makes the next version)")
        batch = db.start_batch(conn, "add_suite", parser="omni_lab_run.py", parser_sha256=db.file_sha256(Path(__file__)),
                               argv=list(sys.argv))
        prov = db.provenance(path)
        pks = _write_cases(conn, batch, cases, prov)
        mods_in = sorted({m for c in cases for m in c["modality_in"]})
        sid = db.ensure_suite(conn, args.name, version, status="draft" if args.draft else "sealed",
                              modality_in=mods_in, default_grader=args.grader, bank_sha256=prov[1],
                              description=args.description or f"added from {path.name}", prov=prov, batch_id=batch)
        K.set_membership(conn, sid, pks)
        db.end_batch(conn, batch, "ok", len(cases), len(cases), 0, file_shas=[prov[1]])
        conn.commit()
        print(f"suite {args.name}@{version}: {len(cases)} cases ({len(set(pks))} distinct rows; identical questions "
              f"already in the DB were reused) from {path}")
        return 0
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def cmd_add_cases(args) -> int:
    path = Path(args.file)
    conn = db.connect("rw", app="add-cases")
    try:
        old = suite_row(conn, args.suite)
        cases = parse_case_file(path, old["name"])
        batch = db.start_batch(conn, "add_cases", parser="omni_lab_run.py", parser_sha256=db.file_sha256(Path(__file__)),
                               argv=list(sys.argv))
        prov = db.provenance(path)
        new_pks = _write_cases(conn, batch, cases, prov)
        current = suite_case_rows(conn, old["id"])
        new_ids = {c["case_id"] for c in cases}
        # same case_id with a new question replaces the old row in the NEW version only (old runs keep theirs)
        members = [c["case_pk"] for c in current if c["case_id"] not in new_ids] + new_pks
        replaced = sorted(new_ids & {c["case_id"] for c in current})
        if old["status"] == "draft":
            sid, version = old["id"], old["version"]
        else:
            version = args.version or next_version(conn, old["name"])
            sid = db.ensure_suite(conn, old["name"], version, status="sealed",
                                  modality_in=sorted(set(old["modality_in"] or ["text"]) | {m for c in cases for m in c["modality_in"]}),
                                  default_grader=old["default_grader"], bank_sha256=prov[1],
                                  description=f"{old['name']}@{old['version']} + {len(cases)} cases from {path.name}"
                                              + (f" (replaced: {', '.join(replaced)})" if replaced else ""),
                                  prov=prov, batch_id=batch)
        K.set_membership(conn, sid, list(dict.fromkeys(members)))
        db.end_batch(conn, batch, "ok", len(cases), len(cases), 0, file_shas=[prov[1]])
        conn.commit()
        print(f"suite {old['name']}@{version}: {len(set(members))} cases ({len(current)} carried from "
              f"@{old['version']}, {len(cases)} added{', replaced ' + ', '.join(replaced) if replaced else ''}). "
              f"Runs made before show '-' for the new cases.")
        return 0
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------

def cmd_grade(args) -> int:
    if not args.grader.startswith("human:"):
        raise db.LabError("reader grades use a human:<name> grader (machine grades come from the worker)")
    entries: list[tuple[str, dict]] = []
    if args.file:
        doc = json.loads(Path(args.file).read_text(encoding="utf-8"))
        items = doc.get("cases", doc) if isinstance(doc, dict) else {}
        for cid, v in items.items():
            entries.append((cid, v if isinstance(v, dict) else {"grade": v}))
    else:
        if not (args.case and args.grade):
            raise db.LabError("grade RUN CASE GRADE [--why ...]  or  grade RUN --file grades.json")
        entries.append((args.case, {"grade": args.grade, "why": args.why,
                                    "artifact_flags": [f for f in (args.flags or "").split(",") if f]}))
    conn = db.connect("rw", app="grade")
    try:
        if args.grader not in K.GRADERS:
            K.GRADERS[args.grader] = ("human", "correct/partial/miss/none", f"reader grades by {args.grader[6:]}")
        batch_id = db.start_batch(conn, "grade", parser="omni_lab_run.py", argv=list(sys.argv))
        batch = C.Batch(batch_id, "grade")
        prov = db.provenance(args.file) if args.file else (f"cli:{args.grader}", None, db.this_host())
        wrote = 0
        for cid, entry in entries:
            res = K.result_id(conn, args.run, cid)
            if res is None:
                print(f"  {cid}: run {args.run} has no result for this case (blank cell: nothing to grade)")
                continue
            if K.put_reader_grade(conn, batch, res, args.grader, entry, prov):
                wrote += 1
                print(f"  {cid}: {entry.get('grade')} ({args.grader})")
            else:
                print(f"  {cid}: grade {entry.get('grade')!r} not one of correct / partially correct / content miss / "
                      "no content (or partial / miss / none)")
        db.end_batch(conn, batch_id, "ok", len(entries), wrote, len(entries) - wrote, error="; ".join(batch.notes) or None)
        conn.commit()
        print(f"{wrote}/{len(entries)} grades written to run {args.run}")
        return 0 if wrote == len(entries) else 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# log-run (training / fold / cache jobs)
# ---------------------------------------------------------------------------

def _json_arg(value: str | None) -> dict:
    if not value:
        return {}
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def _run_ref(conn, ref: str) -> int:
    with conn.cursor() as cur:
        if ref.isdigit():
            cur.execute("SELECT id FROM run WHERE id = %s", (int(ref),))
        else:
            cur.execute("SELECT id FROM run WHERE run_key = %s", (ref,))
        row = cur.fetchone()
    if row is None:
        raise db.LabError(f"run {ref!r} not found")
    return row[0]


def _metrics(items: list[str] | None) -> list[tuple[str, int, float]]:
    out = []
    for item in items or []:
        m = re.fullmatch(r"([^=@]+)=([-+0-9.eE]+|nan|inf)(?:@(\d+))?", item.strip())
        if not m:
            raise db.LabError(f"metric {item!r}: use name=value or name=value@step")
        out.append((m.group(1), int(m.group(3) or 0), float(m.group(2))))
    return out


def cmd_log_run(args) -> int:
    conn = db.connect("rw", app="log-run", statement_timeout_ms=10000, lock_timeout_ms=3000)
    try:
        if args.action == "start":
            if not (args.kind and args.name):
                raise db.LabError("log-run start --kind KIND --name NAME [...]")
            config = _json_arg(args.config)
            stamp = utc_stamp()
            key = args.run_key or f"{args.kind}:{re.sub(r'[^A-Za-z0-9.-]+', '-', args.name)}:{stamp}:{db.sha256_json(config)[:8]}"
            model = resolve_model(conn, args.model)["id"] if args.model else None
            hid = db.ensure_host(conn, db.this_host()) if db.this_host() != C.SERVER_HOST else C.host_id(conn)
            rid = db.upsert_run(conn, key, args.project, args.kind, args.name, "running", model_id_=model, host_id=hid,
                                started_at=datetime.now(timezone.utc), config=config, log_path=args.log,
                                run_dir=args.run_dir, owner=args.owner, notes=args.notes)
            if args.lane:
                C.set_run_lane(conn, rid, args.lane)
            if args.gpus:
                smi = nvidia_smi()
                for i in (int(x) for x in args.gpus.split(",") if x.strip()):
                    card = smi[i]
                    gid = db.gpu_id_by_uuid(conn, hid, card["uuid"]) or db.upsert_gpu(
                        conn, hid, i, card["uuid"], card["name"], card["sm"], card["vram_mib"])
                    db.add_run_device(conn, rid, gid, kernel_path=args.kernel_path,
                                      detail={"mapping": "uuid-at-log-run", "index": i})
            conn.commit()
            print(rid)
            return 0
        rid = _run_ref(conn, args.run)
        rows = _metrics(args.metric)
        if args.metrics_jsonl:
            for line in Path(args.metrics_jsonl).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    rows += C.step_metrics(rec)
        db.put_metrics(conn, rid, rows)
        if args.action == "end":
            receipt_sha = db.file_sha256(args.receipt) if args.receipt and os.path.isfile(args.receipt) else None
            db.finish_run(conn, rid, args.status, exit_code=args.exit_code, receipt_path=args.receipt,
                          receipt_sha256=receipt_sha, log_path=args.log)
        conn.commit()
        print(f"run {rid}: {len(rows)} metric rows" + (f", status {args.status}" if args.action == "end" else ""))
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# argparse (wired into omni_lab.py)
# ---------------------------------------------------------------------------

def add_parsers(sub) -> None:
    p = sub.add_parser("run", help="evaluate one model on one suite / saved test -> console + database")
    p.add_argument("--model", help="model_key, ledger checkpoint id (OC02-F-Q38-27BU-OMNI1) or a path")
    p.add_argument("--suite", help="suite name or name@version (default: newest version)")
    p.add_argument("--test", help="a saved test (suite + engine + gen + cards + case filter)")
    p.add_argument("--save-test", help="save this run's suite/engine/gen/cards/case filter as a named test")
    p.add_argument("--gpus", help="nvidia-smi indexes, e.g. 6,4,0 (first card holds embed/head for split engines)")
    p.add_argument("--cpu", action="store_true", help="run on CPU (engine hf / echo)")
    p.add_argument("--engine", choices=("hf", "qwen35-student", "chimera", "gguf", "echo"), help="default: from the model layout")
    p.add_argument("--cases", help="comma list of case ids or globs")
    p.add_argument("--category", help="category glob")
    p.add_argument("--limit", type=int, help="first N cases of the suite")
    p.add_argument("--cap", type=int, help="max new tokens (default 1024)")
    p.add_argument("--thinking", choices=("off", "template-default"))
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--seed", type=int)
    p.add_argument("--no-band-mask", action="store_true", help="turn off the S8 ledger-band logit mask (student/chimera)")
    p.add_argument("--tokenizer-dir", help="default: the model dir, else the tokenizer an earlier eval of the model used")
    p.add_argument("--loader-class")
    p.add_argument("--split", help="chimera: GiB per visible card, e.g. 46,24")
    p.add_argument("--server-url", help="gguf: URL of a RUNNING llama-server")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--nice", type=int, default=15)
    p.add_argument("--allow-gpu5", action="store_true")
    p.add_argument("--busy-limit", type=int, default=2000, help="refuse a card using more than this many MiB")
    p.add_argument("--mem-floor-gib", type=float, default=40.0)
    p.add_argument("--baseline", help="run ids to compare against (default: auto; 'none' to skip)")
    p.add_argument("--show", default="grade", choices=("grade", "tokens", "term", "text"))
    p.add_argument("--name")
    p.add_argument("--label")
    p.add_argument("--owner", default="claude")
    p.add_argument("--notes")
    p.add_argument("--gpu-python", default=GPU_PYTHON)
    p.add_argument("--dry-run", action="store_true", help="resolve and print; claim nothing, start nothing")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="runs x cases pivot ('-' = never ran that case)")
    p.add_argument("runs", nargs="*", help="eval run ids (default: select with the filters)")
    p.add_argument("--suite", help="rows = this suite's cases (all versions); runs = those with results on them")
    p.add_argument("--model", action="append", help="model_key / ledger id (repeatable)")
    p.add_argument("--engine")
    p.add_argument("--test")
    p.add_argument("--last", type=int, default=8)
    p.add_argument("--cases", help="comma list of case-id globs")
    p.add_argument("--show", default="grade", choices=("grade", "tokens", "term", "text"))
    p.add_argument("--width", type=int, default=40, help="max column width")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("list", help="runs | suites | models | tests | cases <suite>")
    p.add_argument("what", choices=("runs", "suites", "models", "tests", "cases"))
    p.add_argument("filter", nargs="?")
    p.add_argument("--kind")
    p.add_argument("--all", action="store_true", help="models: include every scanned artifact")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("add-suite", help="a bank/task file -> a new suite")
    p.add_argument("name")
    p.add_argument("--file", required=True)
    p.add_argument("--version")
    p.add_argument("--grader", help="suite default grader (per-case graders are inferred: tools+expected, asserts, oracle)")
    p.add_argument("--draft", action="store_true", help="draft suites take add-cases in place")
    p.add_argument("--description")
    p.set_defaults(func=cmd_add_suite)

    p = sub.add_parser("add-cases", help="new cases -> the next suite version (old cases carried forward)")
    p.add_argument("suite")
    p.add_argument("--file", required=True)
    p.add_argument("--version")
    p.set_defaults(func=cmd_add_cases)

    p = sub.add_parser("grade", help="reader grade(s) on a run's results")
    p.add_argument("run", type=int)
    p.add_argument("case", nargs="?")
    p.add_argument("grade", nargs="?", help="correct | partial | miss | none (or the long forms)")
    p.add_argument("--why")
    p.add_argument("--flags", help="artifact flags the reader saw, comma list")
    p.add_argument("--file", help="{case_id: grade} or {case_id: {grade, why}} or {cases: {...}}")
    p.add_argument("--grader", default="human:cyril")
    p.set_defaults(func=cmd_grade)

    p = sub.add_parser("log-run", help="start / end / metric rows for a training, fold or cache job")
    p.add_argument("action", choices=("start", "end", "metric"))
    p.add_argument("run", nargs="?", help="end/metric: run id or run_key")
    p.add_argument("--kind", choices=("fold", "distill", "cache", "gen", "eval", "probe", "build", "export", "review",
                                      "smoke", "other"))
    p.add_argument("--name")
    p.add_argument("--project", default="chimera-omni")
    p.add_argument("--model")
    p.add_argument("--lane")
    p.add_argument("--config", help="JSON text or @file")
    p.add_argument("--gpus")
    p.add_argument("--kernel-path")
    p.add_argument("--run-key")
    p.add_argument("--run-dir")
    p.add_argument("--log")
    p.add_argument("--owner", default="claude")
    p.add_argument("--notes")
    p.add_argument("--status", default="pass", choices=("pass", "fail", "stopped", "refused", "skipped", "unknown"))
    p.add_argument("--exit-code", type=int)
    p.add_argument("--receipt")
    p.add_argument("--metric", action="append", help="name=value[@step] (repeatable)")
    p.add_argument("--metrics-jsonl", help="trainer STEP records (one JSON per line)")
    p.set_defaults(func=cmd_log_run)
