#!/usr/bin/env python3
"""d2clean_recache.py -- lane d2clean's runner: the Q38 top-64 teacher cache of the clean re-pack (d2_clean_repack.py),
one card per runner, built only on cards v1 does not need, yielding to v1 at once.  Written 2026-10-15 by Claude.
Standard library only: run with /usr/bin/python3.

WHAT IT RUNS.  The frozen qtext_teacher_cache 1.0.5 (`$Q/frozen/qtext_teacher_cache-1.0.5.py`, sha f2465a11: the tool
and engine (tps_probe 449eecc4) that sealed the 20 registry caches) with the sealed caches' flags: --dtype bfloat16
--attn-implementation sdpa --max-batch-tokens 32768 --batch-order length --rows-per-shard 64 --head-precision fp32
--digest-threads 8, teacher q38-27bu, out-root /mnt/lexar/omni-campaign/teacher_cache/qtext (so the cache of file F is
<root>/q38-27bu/<sha256(F)>, where d2_plan looks).  Per card class (measured 2026-10-15, lane d2clean P40 probe):
  P40 (GPU1/2/3, Pascal venv)  --linear-attn-kernel torch: fla's Triton kernels do not compile for sm_61 ("Feature
                               '.bf16' requires .target sm_80"), so `auto` (fla, the sealed caches' path) is refused by
                               the card; the torch GDN reference is the tool's own alternative and is recorded in the
                               cache config ("chunk_gated_delta_rule": "torch").  Against an fp32 reference of the same
                               rows it is as close as the sealed 3090/fla cache (KL mean 0.0116 vs 0.0109, top-1 0.989 vs
                               0.988; receipts scratch/d2clean/p40probe-*).  --micro-batch 2 (mb8 and mb4 run out of memory:
                               the math SDPA's scores at T = 4,096), 53.21 tok/s.  --host-resident off (host RAM is v1's).
  RTX 3090 (GPU0/4, lab venv)  exactly the sealed path (--linear-attn-kernel auto = fla, --micro-batch 8); only when the
                               GO file names the card AND $Q/d1.reviewed exists AND its gpuN.lock is free.
GPU5 and GPU6 are refused.  Micro-batch and host residency are not part of a cache's locked semantics (cache_config).

GATES, re-read before every file and every 60 s while a file builds:
  go        $Q/d2clean.go: line 1 `d2clean-recache-go`, `manifest_sha256=<the clean manifest's>`, `cards=1,2,3`; a card
            not listed (or the file gone) stops the runner after it yields
  corpus    d2_clean_repack `state`: built and verified for that manifest_sha256
  card      < 2,000 MiB used and no compute app; P40: no live foreign line in p40.claims (d1_seal_lib.claim_state);
            $Q/gpuN.lock taken with flock (non-inheritable) only while a file builds, released on every yield
  v1        x3_v1_yield_watchdog --probe (the fold lane's v1-pressure reader, its read-only CLI): any v1 RAM wait
            (gpu6 queue, D1 entry, post_d1, d1_seal, d2), a K3-chain job queued below 75 GiB, and -- on a P40 -- the
            d1_seal bank v1 review waiting for or holding the P40s.  `swap_full` alone is recorded, not obeyed (it is
            the steady state while D1 and K3 both run; this job holds no large anonymous RAM).
  ram       MemAvailable >= START_GIB to start a file; below FLOOR_GIB while building = yield
On pressure: SIGTERM the build (qtext 1.0.5 turns it into SystemExit(143), writes its "after" digest, and keeps every
finished 64-row shard: resume skips them), wait for it, release the claim and the lock, then wait for RESUME_HOLD_S of
no pressure before taking the card again.  It never sends SIGKILL.  A build that fails for any other reason is retried
once; a second failure on the same file halts the runner (status FAILED, the file's lock released).

FILES: one flock per clean file under $Q/d2clean/claims/ (two runners never build one file); status lines to
$Q/d2clean.status; log $Q/logs/d2clean/gpu<N>.log; heartbeats lane L-TRAIN (detail only, the D2 lane owns the input).

    setsid nohup /usr/bin/python3 $T/d2clean_recache.py run --card 3 >> $Q/logs/d2clean/gpu3.runner.log 2>&1 < /dev/null &
    /usr/bin/python3 $T/d2clean_recache.py probe --card 3     # print every gate once; take nothing
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TOOL_NAME = "d2clean_recache"
TOOL_VERSION = "1.0.1"  # 2026-10-15: P40 micro-batch default 2 (the probe: mb8 and mb4 run out of memory in the math SDPA)
# 1.0.0 2026-10-15: first version (lane d2clean), never run; _superseded/d2clean_recache_1.0.0.py

OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
LEX = Path(os.environ.get("LEX", "/mnt/lexar/omni-campaign"))
Q = OC / "distill" / "queues"
T = OC / "distill" / "tools"
FROZEN = Q / "frozen" / "qtext_teacher_cache-1.0.5.py"
FROZEN_SHA256 = "f2465a11b7a2242f167191fca8583a11ff74a50d33d7bba84b45c2d123ce6cb9"
ENGINE = OC / "distill" / "tps_probe.py"
ENGINE_SHA256 = "449eecc4acf985556a2454f4af25ead6379d2d515a7d77879f27d9c9d80f9a1a"
MODEL = Path("/home/cereal/models/Qwen3.8-27B-Uncensored")
OUT_ROOT = LEX / "teacher_cache" / "qtext"
TEACHER = "q38-27bu"
CLEAN_DIR = OC / "distill" / "corpus" / "text_clean_v1"
REPACK = T / "d2_clean_repack.py"
LAB_PY = "/home/cereal/model-surgery-lab/venv/bin/python"
PASCAL_PY = "/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python"
WATCHDOG = LEX / "research" / "fold" / "tools" / "x3_v1_yield_watchdog.py"
SEAL_LIB = T / "d1_seal_lib.py"
LANES = T / "lanes.py"
GO_FILE = Q / "d2clean.go"
STATUS = Q / "d2clean.status"
CLAIMS = Q / "p40.claims"
LANE = "d2clean"
JOB = "qtext-clean"

#: card -> (uuid, class, python, linear-attention kernel, micro-batch).  P40 micro-batch: the probe's largest that ran
#: (2026-10-15, GPU3, 8 sealed rows: mb8 OOM 12 GiB SDPA, mb4 OOM 6 GiB, mb2 53.21 tok/s peak 20.9 GB; fp32 mb1 47.26 tok/s)
MB_P40 = int(os.environ.get("D2CLEAN_MB_P40", "2"))
CARDS = {
    0: ("GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561", "rtx3090", LAB_PY, "auto", 8),
    1: ("GPU-c95fb5b3-f97c-bfb2-9760-79fe107ffe6d", "p40", PASCAL_PY, "torch", MB_P40),
    2: ("GPU-a92b0dfe-83f8-11ff-db0d-c65921f365b4", "p40", PASCAL_PY, "torch", MB_P40),
    3: ("GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8", "p40", PASCAL_PY, "torch", MB_P40),
    4: ("GPU-2ce6a95c-7b1f-644a-3808-d781732a0582", "rtx3090", LAB_PY, "auto", 8),
}
REFUSED_CARDS = {5: "GPU5 is never used", 6: "GPU6 is lane L-K3's (K3 passes, H3, j6)"}
SEALED_FLAGS = ["--dtype", "bfloat16", "--attn-implementation", "sdpa", "--digest-threads", "8", "--teacher", TEACHER,
                "--max-batch-tokens", "32768", "--batch-order", "length", "--rows-per-shard", "64", "--head-precision", "fp32",
                "--host-resident", "off"]
CARD_BUSY_MIB = 2000
START_GIB = float(os.environ.get("D2CLEAN_START_GIB", "60"))
FLOOR_GIB = float(os.environ.get("D2CLEAN_FLOOR_GIB", "24"))
RESUME_HOLD_S = int(os.environ.get("D2CLEAN_RESUME_HOLD_S", "600"))
POLL_S = int(os.environ.get("D2CLEAN_POLL_S", "60"))
TERM_WAIT_S = 1800  # the "after" digest reads 55.6 GB (105 s measured on a warm page cache)
OBEYED = ("gpu6_wait_ram", "d1_wait_mem", "post_d1_wait_mem", "d1_seal_wait_mem", "d2_wait_mem", "k3_queued_low_mem")
P40_ONLY = ("gpu3_wait",)  # the watchdog's name; it means "the bank v1 review waits for / holds the P40s 1+2+3"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def mem_available_gib() -> float:
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    return 0.0


def cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


# ------------------------------------------------------------------------------------------------ pure readers (tested)
def read_go(text: str | None, now: str | None = None) -> dict:
    """$Q/d2clean.go -> {'ok', 'manifest_sha256', 'cards', 'pending', 'why'}.  `cards=3` lists cards now;
    `cards_after=<UTC>:1,2` (any number of lines) adds cards from that time on (a card another lane holds until a
    declared stop: the tool-call refill's GPU1/2)."""
    if text is None:
        return {"ok": False, "why": "no GO file"}
    now = now or utc()
    lines = [x.strip() for x in text.splitlines() if x.strip() and not x.strip().startswith("#")]
    if not lines or lines[0] != "d2clean-recache-go":
        return {"ok": False, "why": "GO file line 1 is not d2clean-recache-go"}
    kv = dict(x.split("=", 1) for x in lines[1:] if "=" in x and not x.startswith("cards_after="))
    cards, pending = set(), []
    try:
        cards |= {int(c) for c in kv.get("cards", "").split(",") if c.strip()}
        for x in lines[1:]:
            if x.startswith("cards_after="):
                when, _, rest = x[len("cards_after="):].rpartition(":")
                if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", when):
                    return {"ok": False, "why": f"GO file {x!r}: want cards_after=<YYYY-MM-DDTHH:MM:SSZ>:<cards>"}
                group = {int(c) for c in rest.split(",") if c.strip()}
                if now >= when:
                    cards |= group
                else:
                    pending.append({"after": when, "cards": sorted(group)})
    except ValueError:
        return {"ok": False, "why": "GO file cards are not lists of integers"}
    sha = kv.get("manifest_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        return {"ok": False, "why": "GO file has no manifest_sha256=<64 hex>"}
    return {"ok": True, "manifest_sha256": sha, "cards": sorted(cards), "pending": pending, "why": None}


def pressure(signals: dict, card_class: str) -> list[str]:
    """The watchdog signals this runner obeys on a card of this class (swap_full is recorded, never obeyed)."""
    names = OBEYED + (P40_ONLY if card_class == "p40" else ())
    return [f"{k}: {signals[k]}" for k in names if k in signals]


def pick_file(files: list[dict], sealed: dict[str, bool], failed: set[str]) -> list[dict]:
    """Files still to build, in manifest order (a failed-twice file is skipped by this runner)."""
    return [f for f in files if not sealed.get(f["sha256"]) and f["sha256"] not in failed]


def claim_line(card: int, pid: int, log: str) -> str:
    # never the words "release"/"released": d1_seal_lib.claim_state reads them as a release (the x3 claim-line lesson)
    return (f"{utc()} CLAIM GPU{card} owner=claude lane={LANE} job={JOB} pid={pid} log={log} "
            f"preempt: kill -TERM {pid} (the build stops at a 64-row shard boundary, finished shards are kept; the runner "
            f"yields by itself when the d1_seal review waits for the P40s or v1 waits for RAM)")


def release_line(card: int) -> str:
    return f"{utc()} GPU{card} lane={LANE} job={JOB} released"


# ------------------------------------------------------------------------------------------------ the runner
class Runner:
    def __init__(self, a):
        self.a = a
        self.card = a.card
        self.uuid, self.cls, self.py, self.kernel, self.mb = CARDS[self.card]
        self.log_dir = Q / "logs" / "d2clean"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"gpu{self.card}.log"
        self.claims_dir = Q / "d2clean" / "claims"
        self.claims_dir.mkdir(parents=True, exist_ok=True)
        self.card_fd = None
        self.file_fd = None
        self.claimed = False
        self.failed: dict[str, int] = {}
        self.wd_sha = None
        self.stop = False
        signal.signal(signal.SIGTERM, self._sigterm)

    def _sigterm(self, *_):
        self.stop = True

    # ---- output
    def say(self, text: str) -> None:
        line = f"{utc()} gpu{self.card} {text}"
        print(line, flush=True)
        with open(self.log_path, "a") as handle:
            handle.write(line + "\n")

    def status(self, text: str) -> None:
        with open(STATUS, "a") as handle:
            handle.write(f"{utc()} d2clean gpu{self.card} {text}\n")
        self.say(text)

    def heartbeat(self, detail: str) -> None:
        try:
            subprocess.run(["/usr/bin/python3", str(LANES), "heartbeat", "--lane", "L-TRAIN", "--agent", "claude",
                            "--detail", detail[:480], "--source", f"{TOOL_NAME}-{TOOL_VERSION}"],
                           capture_output=True, timeout=45, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass

    # ---- gates
    def go(self) -> dict:
        try:
            return read_go(GO_FILE.read_text(encoding="utf-8"))
        except OSError:
            return read_go(None)

    def clean_state(self) -> dict:
        r = subprocess.run([LAB_PY, str(REPACK), "state", "--out", str(self.a.clean_dir)], capture_output=True, text=True, timeout=300)
        for line in r.stdout.splitlines():
            if line.startswith("D2CLEAN state "):
                return json.loads(line[len("D2CLEAN state "):])
        return {"built": False, "why": (r.stdout + r.stderr)[-300:]}

    def signals(self) -> dict:
        """The fold lane's v1-pressure reader through its own read-only CLI (`--probe`: "print the signals ... act on
        nothing"), so this runner obeys exactly the signals the fold job yields to: {name: reason, '_facts': {...}}."""
        r = subprocess.run(["/usr/bin/python3", str(WATCHDOG), "--probe", "--lanes", ""], capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            raise RuntimeError(f"watchdog --probe rc {r.returncode}: {(r.stderr or r.stdout)[-200:]}")
        doc = json.loads(r.stdout)
        self.wd_sha = sha256_file(WATCHDOG)
        return {**doc["pressure"], "_facts": doc.get("facts") or {}}

    def card_state(self) -> list[str]:
        why = []
        q = subprocess.run(["nvidia-smi", "-i", str(self.card), "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60)
        parts = [x.strip() for x in q.stdout.strip().split(",")]
        if len(parts) < 2 or parts[0] != self.uuid:
            why.append(f"nvidia-smi -i {self.card} names {parts[:1]} not {self.uuid}")
        elif int(parts[1]) >= CARD_BUSY_MIB:
            why.append(f"GPU{self.card} holds {parts[1]} MiB")
        apps = subprocess.run(["nvidia-smi", "-i", str(self.card), "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60).stdout
        if re.search(r"\d", apps):
            why.append(f"GPU{self.card} has compute apps {apps.split()}")
        if self.cls == "p40":
            r = subprocess.run(["/usr/bin/python3", str(SEAL_LIB), "p40-free", "--claims", str(CLAIMS), "--gpus", str(self.card)],
                               capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                why.append(f"p40.claims: {r.stdout.strip()[-200:]}")
        else:
            if not (Q / "d1.reviewed").is_file():
                why.append("3090: $Q/d1.reviewed does not exist (the D1 review comes first)")
        return why

    def take_card_lock(self) -> bool:
        fd = os.open(str(Q / f"gpu{self.card}.lock"), os.O_RDWR | os.O_CREAT, 0o664)  # non-inheritable (PEP 446)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        self.card_fd = fd
        return True

    def take_file_lock(self, f: dict) -> bool:
        fd = os.open(str(self.claims_dir / f"{f['file']}.lock"), os.O_RDWR | os.O_CREAT, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        self.file_fd = fd
        return True

    def release_all(self, why: str) -> None:
        if self.claimed:
            with open(CLAIMS, "a") as handle:
                handle.write(release_line(self.card) + "\n")
            self.claimed = False
        for name in ("file_fd", "card_fd"):
            fd = getattr(self, name)
            if fd is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                setattr(self, name, None)
        self.say(f"card and file locks released ({why})")

    def gates(self, *, starting: bool) -> tuple[list[str], dict]:
        """(reasons not to run, facts)."""
        why, facts = [], {}
        go = self.go()
        if not go["ok"]:
            return [f"go: {go['why']}"], facts
        if self.card not in go["cards"]:
            return [f"go: card {self.card} not in cards={go['cards']}"], facts
        try:
            sig = self.signals()
        except Exception as exc:  # the v1 reader is the safety: unknown pressure never starts a build
            return [f"v1 pressure reader failed: {exc!r}"], facts
        facts = sig.get("_facts", {})
        facts["swap_full_recorded"] = sig.get("swap_full")
        why += pressure(sig, self.cls)
        mem = mem_available_gib()
        facts["mem_available_gib"] = round(mem, 1)
        if starting and mem < START_GIB:
            why.append(f"MemAvailable {mem:.1f} GiB < {START_GIB}")
        if not starting and mem < FLOOR_GIB:
            why.append(f"MemAvailable {mem:.1f} GiB < floor {FLOOR_GIB}")
        facts["go"] = go
        return why, facts

    # ---- one file
    def build_command(self, f: dict) -> tuple[list[str], dict]:
        purpose = f"d2clean-recache-gpu{self.card}: Q38 top-64 of {f['file']} (D2-X1 clean re-pack)"
        cmd = ["nice", "-n", "10", self.py, str(FROZEN), "build-text", "--model-dir", str(MODEL), "--device", "cuda:0",
               "--linear-attn-kernel", self.kernel, "--micro-batch", str(self.mb), *SEALED_FLAGS,
               "--corpus", str(Path(self.a.clean_dir) / f["file"]), "--out-root", str(OUT_ROOT), "--purpose", purpose,
               "--allocator-receipt", str(OC / "composite" / "receipts" / "allocator" / f"d2clean-gpu{self.card}.json")]
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=self.uuid, OMP_NUM_THREADS="8",
                   TOKENIZERS_PARALLELISM="false", QTEXT_TPS_PROBE=str(ENGINE), PYTHONDONTWRITEBYTECODE="1")
        return cmd, env

    def run_file(self, f: dict) -> str:
        """'sealed' | 'yielded' | 'failed' | 'stopped'."""
        cmd, env = self.build_command(f)
        out = self.log_dir / f"gpu{self.card}-{f['file']}.log"
        self.status(f"START {f['file']} sha {f['sha256'][:16]} kernel={self.kernel} mb={self.mb} log {out}")
        with open(out, "a") as handle:
            handle.write(f"===== {utc()} {' '.join(cmd)}\n")
            handle.flush()
            proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True, close_fds=True)
        termed = None
        last_hb = 0.0
        while proc.poll() is None:
            time.sleep(min(POLL_S, 30))
            if termed is not None:
                if time.time() - termed > TERM_WAIT_S:
                    self.say(f"build pid {proc.pid} still alive {TERM_WAIT_S} s after SIGTERM; waiting (never SIGKILL)")
                    termed = time.time()
                continue
            why, facts = self.gates(starting=False)
            if self.stop:
                why.append("runner got SIGTERM")
            if why:
                self.status(f"YIELD {f['file']}: {'; '.join(why)[:600]}")
                proc.send_signal(signal.SIGTERM)
                termed = time.time()
            elif time.time() - last_hb > 1800:
                last_hb = time.time()
                prog = self.progress(f)
                self.heartbeat(f"{utc()} d2clean gpu{self.card} building {f['file']}: {prog}")
        rc = proc.returncode
        sealed = self.cache_sealed(f)
        if sealed:
            vrc, vline = self.verify_cache(f, env)
            self.status(f"SEALED {f['file']} rc {rc}; verify rc {vrc}: {vline}")
            return "sealed" if vrc == 0 else "failed"
        if termed is not None:
            self.status(f"YIELDED {f['file']} rc {rc}; progress {self.progress(f)}")
            return "stopped" if self.stop else "yielded"
        self.status(f"FAILED {f['file']} rc {rc}; see {out}")
        return "failed"

    def verify_cache(self, f: dict, env: dict) -> tuple[int, str]:
        """The frozen tool's own byte-for-byte re-verify of the sealed cache (CPU)."""
        v = subprocess.run([self.py, str(FROZEN), "verify", "--cache-dir", str(OUT_ROOT / TEACHER / f["sha256"])],
                           env=dict(env, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True, timeout=3600)
        return v.returncode, (v.stdout.strip().splitlines() or [""])[-1][:300]

    def cache_sealed(self, f: dict) -> bool:
        m = OUT_ROOT / TEACHER / f["sha256"] / "manifest.json"
        if not m.is_file():
            return False
        d = json.loads(m.read_text(encoding="utf-8"))
        return bool(d.get("all_finite")) and bool(d.get("all_structural_ok")) and bool((d.get("teacher") or {}).get("equal_before_and_after_every_session"))

    def progress(self, f: dict) -> str:
        p = OUT_ROOT / TEACHER / f["sha256"] / "progress.json"
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return f"shards {d.get('shards_present')}/{d.get('shards_total')}, {d.get('session_forward_tokens_per_second')} tok/s"
        except (OSError, ValueError):
            return "no progress.json yet"

    # ---- main loop
    def run(self) -> int:
        if self.card in REFUSED_CARDS or self.card not in CARDS:
            print(f"REFUSED card {self.card}: {REFUSED_CARDS.get(self.card, 'unknown card')}", flush=True)
            return 3
        for path, want in ((FROZEN, FROZEN_SHA256), (ENGINE, ENGINE_SHA256)):
            got = sha256_file(path) if path.is_file() else None
            if got != want:
                self.status(f"REFUSED {path} hashes {got} (want {want})")
                return 2
        self.status(f"ARMED {TOOL_NAME} {TOOL_VERSION} pid {os.getpid()} class {self.cls} kernel {self.kernel} mb {self.mb} "
                    f"frozen {FROZEN_SHA256[:16]} watchdog {WATCHDOG}")
        ok_since = None
        last_wait = ""
        while not self.stop:
            st = self.clean_state()
            go = self.go()
            if not st.get("built") or not st.get("verified"):
                why = ["clean corpus not built + verified"]
            elif go.get("ok") and go["manifest_sha256"] != st.get("manifest_sha256"):
                why = [f"go names manifest {go['manifest_sha256'][:16]}, the clean corpus is {str(st.get('manifest_sha256'))[:16]}"]
            else:
                why = []
            todo = []
            if not why:
                sealed = {c["corpus_sha256"]: c["sealed"] for c in st["caches"]}
                files = [{"file": c["file"], "sha256": c["corpus_sha256"]} for c in st["caches"]]
                todo = pick_file(files, sealed, {s for s, n in self.failed.items() if n >= 2})
                if not todo:
                    self.status(f"DONE: {st.get('caches_sealed')}/{st.get('files')} clean caches sealed")
                    return 0
                g, facts = self.gates(starting=True)
                why += g
                if not g:
                    why += self.card_state()
            later = any(self.card in p["cards"] for p in go.get("pending") or [])
            if go.get("ok") is False or (go.get("ok") and self.card not in go["cards"] and not later):
                if not self.claimed:
                    self.status(f"STOP: {go.get('why') or f'card {self.card} is not in the GO file'}")
                    return 0
            if why:
                ok_since = None
                text = "; ".join(why)[:500]
                if text != last_wait:
                    self.status(f"WAIT {text}")
                    last_wait = text
                time.sleep(POLL_S)
                continue
            if ok_since is None:
                ok_since = time.time()
                self.say(f"gates open; holding {RESUME_HOLD_S} s before taking the card")
            if time.time() - ok_since < RESUME_HOLD_S:
                time.sleep(POLL_S)
                continue
            target = None
            for f in todo:
                if self.take_file_lock(f):
                    target = f
                    break
            if target is None:
                self.status("WAIT every remaining file is being built by another runner")
                ok_since = None
                time.sleep(POLL_S * 5)
                continue
            if not self.take_card_lock():
                self.release_all("gpu lock busy")
                self.status(f"WAIT $Q/gpu{self.card}.lock is held")
                ok_since = None
                time.sleep(POLL_S)
                continue
            if self.cls == "p40":
                with open(CLAIMS, "a") as handle:
                    handle.write(claim_line(self.card, os.getpid(), str(self.log_path)) + "\n")
                self.claimed = True
            last_wait = ""
            result = self.run_file(target)
            self.release_all(result)
            ok_since = None
            if result == "failed":
                self.failed[target["sha256"]] = self.failed.get(target["sha256"], 0) + 1
                if self.failed[target["sha256"]] >= 2:
                    self.status(f"HALT {target['file']} failed twice on gpu{self.card}: a decision is needed")
                    return 1
            if result == "sealed":
                self.heartbeat(f"{utc()} d2clean gpu{self.card}: {target['file']} sealed")
        self.release_all("runner stopped")
        self.status("STOPPED (SIGTERM)")
        return 0

    def probe(self) -> int:
        out = {"card": self.card, "class": self.cls, "kernel": self.kernel, "mb": self.mb}
        if self.card in REFUSED_CARDS:
            out["refused"] = REFUSED_CARDS[self.card]
        out["frozen_ok"] = FROZEN.is_file() and sha256_file(FROZEN) == FROZEN_SHA256
        out["engine_ok"] = ENGINE.is_file() and sha256_file(ENGINE) == ENGINE_SHA256
        out["go"] = self.go()
        out["clean_state"] = {k: v for k, v in self.clean_state().items() if k != "caches"}
        g, facts = self.gates(starting=True)
        out["gates"] = g
        out["card"] = self.card_state()
        out["facts"] = facts
        out["watchdog_sha256"] = self.wd_sha
        print("D2CLEAN probe " + json.dumps(out, sort_keys=True, default=str), flush=True)
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=TOOL_NAME, description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--card", type=int, required=True)
        p.add_argument("--clean-dir", default=str(CLEAN_DIR))
    a = ap.parse_args(argv)
    if a.card in REFUSED_CARDS or a.card not in CARDS:
        print(f"REFUSED card {a.card}: {REFUSED_CARDS.get(a.card, 'unknown card')}", flush=True)
        return 3
    r = Runner(a)
    return r.run() if a.command == "run" else r.probe()


if __name__ == "__main__":
    sys.exit(main())
