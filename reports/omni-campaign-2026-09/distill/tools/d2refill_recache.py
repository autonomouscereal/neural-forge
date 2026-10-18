#!/usr/bin/env python3
"""d2refill_recache.py -- the Q38 top-64 teacher caches of the D2-X1 refill corpora (ranks 2-3, d2_refill_build.py), one
card per runner, built only on cards v1 does not need, yielding to v1 at once.  Written 2026-10-15 by Claude (lane refill).
Standard library only: run with /usr/bin/python3.

WHAT IT IS.  Lane d2clean's runner (d2clean_recache.Runner, imported, not copied) pointed at another file list: the
corpora of $OC/distill/corpus/d2_refill that `d2_refill_build.py state` reports ELIGIBLE (their omni-d2x1-refill-screen-v1
receipt passes d2_plan's own refill_screen_state) and whose cache is not sealed, rank order.  Everything else is the
d2clean runner's: the frozen qtext_teacher_cache 1.0.5 (f2465a11) with the sealed caches' flags, the P40 path (torch GDN
kernel, Pascal venv, --host-resident off; lane d2clean's measured class, decision d2-clean-repack-2026-10-15), the v1
pressure reader (x3_v1_yield_watchdog --probe: any v1 RAM wait, a K3-chain job queued low, the bank v1 review waiting for
or holding the P40s), p40.claims through d1_seal_lib.claim_state, the SAME $Q/gpuN.lock (so this runner and the d2clean
runner never share a card), SIGTERM-only yields.  GPU5 and GPU6 are refused; 3090s only after $Q/d1.reviewed.

WHAT DIFFERS (and why):
  go          $Q/d2refill.go: line 1 `d2refill-recache-go`, then `cards=` and/or `cards_after=<UTC>:<cards>` (no manifest
              line: the eligibility receipt is the identity).  Today: cards_after=2026-10-15T21:40:00Z:1,2 (the tool-call
              refill's declared server stop), the cards the task names.
  hold        RESUME_HOLD_S 300 s (d2clean's is 600 s): when a card frees, both runners see it; the refill corpora are about
              350k tokens (about 2 P40-hours), d2clean's 12 files about 64 P40-hours, so the refill takes the card first and
              hands it on within hours.  After a refill file the d2clean runner gets the card by the same lock.
  micro-batch P40: largest of 2..16 rows with micro x width <= 8,192 tokens (d2clean's measured-safe pass, mb2 x 4,096):
              the refill rows are 296 / 817 wide, so a pass holds the same tokens with SDPA scores far below the measured
              peak (16 x 24 heads x 296^2 x 4 B = 0.13 GB vs 2 x 24 x 4,096^2 x 4 B = 3.2 GB).  Not a cache semantic
              (d2clean_recache docstring).
  release     the claim is released in a `finally` (review d2clean finding 1: an exception between claim and release left a
              stale CLAIM that blocks the bank v1 review for 72 h); a build still alive there gets SIGTERM and is waited for.
  verify      the frozen tool's byte re-verify runs AFTER the card is released (it is CPU work: review d2clean finding 1);
              its result goes to $OC/distill/corpus/d2_refill/receipts/cache-<sha16>.verify.json (finding 3's receipt).
  done        exit 0 once every eligible refill corpus has a sealed, verified cache.  A corpus built later (a new self-gen
              run) needs a new runner.

FILES: status $Q/d2refill.status; logs $Q/logs/d2refill/gpu<N>.log (+ one per corpus); file flocks $Q/d2refill/claims/;
heartbeats lane L-TRAIN (detail only).

    setsid nohup /usr/bin/python3 $T/d2refill_recache.py run --card 1 >> $Q/logs/d2refill/gpu1.runner.log 2>&1 < /dev/null &
    /usr/bin/python3 $T/d2refill_recache.py probe --card 1     # print every gate once; take nothing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d2clean_recache as base  # noqa: E402  (the d2clean runner: gates, locks, claims, the frozen tool)

TOOL_NAME = "d2refill_recache"
TOOL_VERSION = "1.0.0"  # 2026-10-15: first version (lane refill, Claude lead)

OC, LEX, Q, T = base.OC, base.LEX, base.Q, base.T
REFILL_DIR = OC / "distill" / "corpus" / "d2_refill"
REFILL_BUILD = T / "d2_refill_build.py"
GO_FILE = Q / "d2refill.go"
STATUS = Q / "d2refill.status"
LANE = "d2refill"
JOB = "qtext-refill"
RESUME_HOLD_S = int(os.environ.get("D2REFILL_RESUME_HOLD_S", "300"))
POLL_S = int(os.environ.get("D2REFILL_POLL_S", "60"))
PASS_TOKENS = 8192  # d2clean's measured-safe P40 pass: micro-batch 2 x 4,096
MB_MAX = 16


def utc() -> str:
    return base.utc()


def read_go(text: str | None, now: str | None = None) -> dict:
    """$Q/d2refill.go -> {'ok', 'cards', 'pending', 'why'}: `cards=` now, `cards_after=<UTC>:<cards>` from that time on."""
    if text is None:
        return {"ok": False, "why": "no GO file"}
    now = now or utc()
    lines = [x.strip() for x in text.splitlines() if x.strip() and not x.strip().startswith("#")]
    if not lines or lines[0] != "d2refill-recache-go":
        return {"ok": False, "why": "GO file line 1 is not d2refill-recache-go"}
    cards, pending = set(), []
    try:
        for x in lines[1:]:
            if x.startswith("cards="):
                cards |= {int(c) for c in x[len("cards="):].split(",") if c.strip()}
            elif x.startswith("cards_after="):
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
    return {"ok": True, "cards": sorted(cards), "pending": pending, "why": None}


def micro_batch(card_class: str, width: int, default: int) -> int:
    if card_class != "p40":
        return default
    return max(2, min(MB_MAX, PASS_TOKENS // max(int(width), 1)))


def pick(candidates: list[dict], failed: set[str]) -> list[dict]:
    """Eligible corpora whose cache is not sealed (or sealed but not yet verified by this lane), rank then name order."""
    todo = [c for c in candidates if c.get("eligible") and c.get("corpus_sha256") not in failed
            and not (c.get("cache_sealed") and c.get("verified"))]
    return sorted(todo, key=lambda c: (int(c["rank"]), c["name"]))


def claim_line(card: int, pid: int, log: str) -> str:
    # never the words "release"/"released" (d1_seal_lib.claim_state reads them as a release)
    return (f"{utc()} CLAIM GPU{card} owner=claude lane={LANE} job={JOB} pid={pid} log={log} "
            f"preempt: kill -TERM {pid} stops this runner for good (the shard in flight, up to about 80 min on a P40, is lost; "
            f"finished shards are kept and a new runner resumes them); it yields by itself when the d1_seal review waits for "
            f"the P40s or v1 waits for RAM")


def release_line(card: int) -> str:
    return f"{utc()} GPU{card} lane={LANE} job={JOB} released"


def verify_receipt_path(sha: str) -> Path:
    return REFILL_DIR / "receipts" / f"cache-{sha[:16]}.verify.json"


class RefillRunner(base.Runner):
    def __init__(self, a):
        super().__init__(a)
        self.log_dir = Q / "logs" / "d2refill"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"gpu{self.card}.log"
        self.claims_dir = Q / "d2refill" / "claims"
        self.claims_dir.mkdir(parents=True, exist_ok=True)
        self.proc = None

    # ---- output
    def status(self, text: str) -> None:
        with open(STATUS, "a") as handle:
            handle.write(f"{utc()} d2refill gpu{self.card} {text}\n")
        self.say(text)

    def heartbeat(self, detail: str) -> None:
        try:
            subprocess.run(["/usr/bin/python3", str(base.LANES), "heartbeat", "--lane", "L-TRAIN", "--agent", "claude",
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

    def refill_state(self) -> list[dict]:
        r = subprocess.run([base.LAB_PY, str(REFILL_BUILD), "state", "--out", str(self.a.refill_dir)],
                           capture_output=True, text=True, timeout=600, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        for line in r.stdout.splitlines():
            if line.startswith("D2REFILL state "):
                rows = json.loads(line[len("D2REFILL state "):])["candidates"]
                for c in rows:
                    c["verified"] = bool(c.get("corpus_sha256")) and (json.loads(verify_receipt_path(c["corpus_sha256"]).read_text())
                                                                      .get("rc") == 0 if verify_receipt_path(c["corpus_sha256"]).is_file() else False)
                return rows
        raise RuntimeError(f"d2_refill_build state rc {r.returncode}: {(r.stderr or r.stdout)[-300:]}")

    def release_all(self, why: str) -> None:
        if self.claimed:
            with open(base.CLAIMS, "a") as handle:
                handle.write(release_line(self.card) + "\n")
            self.claimed = False
        for name in ("file_fd", "card_fd"):
            fd = getattr(self, name)
            if fd is not None:
                base.fcntl.flock(fd, base.fcntl.LOCK_UN)
                os.close(fd)
                setattr(self, name, None)
        self.say(f"card and file locks released ({why})")

    # ---- one corpus
    def build_command(self, f: dict) -> tuple[list[str], dict]:
        mb = micro_batch(self.cls, int(f.get("width") or 4096), self.mb)
        purpose = f"d2refill-recache-gpu{self.card}: Q38 top-64 of {f['file']} (D2-X1 refill rank {f['rank']}, {f['name']})"
        cmd = ["nice", "-n", "10", self.py, str(base.FROZEN), "build-text", "--model-dir", str(base.MODEL), "--device", "cuda:0",
               "--linear-attn-kernel", self.kernel, "--micro-batch", str(mb), *base.SEALED_FLAGS,
               "--corpus", f["corpus"], "--out-root", str(base.OUT_ROOT), "--purpose", purpose,
               "--allocator-receipt", str(OC / "composite" / "receipts" / "allocator" / f"d2refill-gpu{self.card}.json")]
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=self.uuid, OMP_NUM_THREADS="8",
                   TOKENIZERS_PARALLELISM="false", QTEXT_TPS_PROBE=str(base.ENGINE), PYTHONDONTWRITEBYTECODE="1")
        return cmd, env

    def cache_sealed(self, f: dict) -> bool:
        return super().cache_sealed({"sha256": f["corpus_sha256"]})

    def progress(self, f: dict) -> str:
        return super().progress({"sha256": f["corpus_sha256"]})

    def run_file(self, f: dict) -> str:
        """'sealed' | 'yielded' | 'stopped' | 'failed'.  The build is never left running when this returns."""
        self.proc = None
        cmd, env = self.build_command(f)
        out = self.log_dir / f"gpu{self.card}-{f['file']}.log"
        self.status(f"START {f['file']} sha {f['corpus_sha256'][:16]} rank {f['rank']} kernel={self.kernel} "
                    f"mb={cmd[cmd.index('--micro-batch') + 1]} log {out}")
        termed = None
        try:
            with open(out, "a") as handle:
                handle.write(f"===== {utc()} {' '.join(cmd)}\n")
                handle.flush()
                self.proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                             start_new_session=True, close_fds=True)
            last_hb = 0.0
            while self.proc.poll() is None:
                time.sleep(min(POLL_S, 30))
                if termed is not None:
                    if time.time() - termed > base.TERM_WAIT_S:
                        self.say(f"build pid {self.proc.pid} still alive {base.TERM_WAIT_S} s after SIGTERM; waiting (never SIGKILL)")
                        termed = time.time()
                    continue
                why, _facts = self.gates(starting=False)
                if self.stop:
                    why.append("runner got SIGTERM")
                if why:
                    self.status(f"YIELD {f['file']}: {'; '.join(why)[:600]}")
                    self.proc.send_signal(signal.SIGTERM)
                    termed = time.time()
                elif time.time() - last_hb > 1800:
                    last_hb = time.time()
                    self.heartbeat(f"{utc()} d2refill gpu{self.card} building {f['file']}: {self.progress(f)}")
        finally:
            if self.proc is not None and self.proc.poll() is None:  # an exception above: never leave the build behind
                self.say(f"stopping build pid {self.proc.pid} (runner leaving run_file)")
                self.proc.send_signal(signal.SIGTERM)
                termed = termed or time.time()
                while self.proc.poll() is None:
                    time.sleep(5)
        rc = self.proc.returncode
        if self.cache_sealed(f):
            self.status(f"BUILT {f['file']} rc {rc}; cache sealed (verify follows, card released first)")
            return "sealed"
        if termed is not None:
            self.status(f"YIELDED {f['file']} rc {rc}; progress {self.progress(f)}")
            return "stopped" if self.stop else "yielded"
        self.status(f"FAILED {f['file']} rc {rc}; see {out}")
        return "failed"

    def verify(self, f: dict) -> int:
        """The frozen tool's byte re-verify, CPU, with no card or claim held; its receipt is what `pick` reads."""
        cmd, env = self.build_command(f)
        try:
            vrc, vline = self.verify_cache({"sha256": f["corpus_sha256"]}, env)
        except (OSError, subprocess.TimeoutExpired) as exc:
            vrc, vline = 99, f"verify did not run: {exc!r}"
        cache = base.OUT_ROOT / base.TEACHER / f["corpus_sha256"]
        body = {"schema": "omni-d2x1-refill-cache-verify-v1", "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "utc": utc(),
                "corpus": f["corpus"], "corpus_sha256": f["corpus_sha256"], "cache_dir": str(cache),
                "cache_manifest_sha256": base.sha256_file(cache / "manifest.json") if (cache / "manifest.json").is_file() else None,
                "frozen_tool_sha256": base.FROZEN_SHA256, "rc": vrc, "last_line": vline}
        path = verify_receipt_path(f["corpus_sha256"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        self.status(f"SEALED {f['file']} verify rc {vrc}: {vline} (receipt {path})")
        return vrc

    # ---- main loop
    def run(self) -> int:
        if self.card in base.REFUSED_CARDS or self.card not in base.CARDS:
            print(f"REFUSED card {self.card}: {base.REFUSED_CARDS.get(self.card, 'unknown card')}", flush=True)
            return 3
        for path, want in ((base.FROZEN, base.FROZEN_SHA256), (base.ENGINE, base.ENGINE_SHA256)):
            got = base.sha256_file(path) if path.is_file() else None
            if got != want:
                self.status(f"REFUSED {path} hashes {got} (want {want})")
                return 2
        self.status(f"ARMED {TOOL_NAME} {TOOL_VERSION} pid {os.getpid()} class {self.cls} kernel {self.kernel} "
                    f"frozen {base.FROZEN_SHA256[:16]} hold {RESUME_HOLD_S} s; base {base.TOOL_NAME} {base.TOOL_VERSION}")
        ok_since = None
        last_wait = ""
        while not self.stop:
            go = self.go()
            try:
                cands = self.refill_state()
                why = []
            except Exception as exc:  # unknown eligibility never starts a build
                cands, why = [], [f"refill state failed: {exc!r}"[:300]]
            halted = {s for s, n in self.failed.items() if n >= 2}
            todo = pick(cands, halted)
            if not why and not todo and halted:
                self.status(f"HALT {len(halted)} refill cache(s) failed twice ({sorted(h[:16] for h in halted)}): a decision is needed")
                return 1
            if not why and not todo:
                done = [c["name"] for c in cands if c.get("eligible")]
                self.status(f"DONE: every eligible refill corpus has a sealed, verified cache ({done})")
                return 0
            later = any(self.card in p["cards"] for p in go.get("pending") or [])
            if go.get("ok") is False or (go.get("ok") and self.card not in go["cards"] and not later):
                self.status(f"STOP: {go.get('why') or f'card {self.card} is not in the GO file'}")
                return 0
            if not why:
                sealed_unverified = [c for c in todo if c.get("cache_sealed")]
                if sealed_unverified:  # a cache another session sealed: verify it here (CPU), no card needed
                    c = sealed_unverified[0]
                    if self.verify(c) != 0:
                        self.failed[c["corpus_sha256"]] = self.failed.get(c["corpus_sha256"], 0) + 1
                    continue
                g, _facts = self.gates(starting=True)
                why += g
                if not g:
                    why += self.card_state()
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
                if self.take_file_lock({"file": f["file"]}):
                    target = f
                    break
            if target is None:
                self.status("WAIT every remaining refill corpus is being built by another runner")
                ok_since = None
                time.sleep(POLL_S * 5)
                continue
            if not self.take_card_lock():
                self.release_all("gpu lock busy")
                self.status(f"WAIT $Q/gpu{self.card}.lock is held")
                ok_since = None
                time.sleep(POLL_S)
                continue
            result = "failed"
            try:
                if self.cls == "p40":
                    with open(base.CLAIMS, "a") as handle:
                        handle.write(claim_line(self.card, os.getpid(), str(self.log_path)) + "\n")
                    self.claimed = True
                last_wait = ""
                result = self.run_file(target)
            finally:
                self.release_all(result)
            ok_since = None
            if result == "sealed":
                result = "sealed" if self.verify(target) == 0 else "failed"
            if result == "failed":
                self.failed[target["corpus_sha256"]] = self.failed.get(target["corpus_sha256"], 0) + 1
                if self.failed[target["corpus_sha256"]] >= 2:
                    self.status(f"HALT {target['file']} failed twice on gpu{self.card}: a decision is needed")
                    return 1
            if result == "sealed":
                self.heartbeat(f"{utc()} d2refill gpu{self.card}: {target['file']} sealed + verified")
        self.release_all("runner stopped")
        self.status("STOPPED (SIGTERM)")
        return 0

    def probe(self) -> int:
        out = {"card": self.card, "class": self.cls, "kernel": self.kernel}
        out["frozen_ok"] = base.FROZEN.is_file() and base.sha256_file(base.FROZEN) == base.FROZEN_SHA256
        out["engine_ok"] = base.ENGINE.is_file() and base.sha256_file(base.ENGINE) == base.ENGINE_SHA256
        out["go"] = self.go()
        try:
            out["refill"] = [{k: c.get(k) for k in ("rank", "name", "eligible", "why", "cache_sealed", "verified", "corpus_sha256")}
                             for c in self.refill_state()]
        except Exception as exc:
            out["refill"] = f"state failed: {exc!r}"
        out["todo"] = [c["name"] for c in pick(out["refill"], set())] if isinstance(out["refill"], list) else None
        g, facts = self.gates(starting=True)
        out["gates"] = g
        out["card_state"] = self.card_state()
        out["facts"] = facts
        print("D2REFILL probe " + json.dumps(out, sort_keys=True, default=str), flush=True)
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=TOOL_NAME, description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--card", type=int, required=True)
        p.add_argument("--refill-dir", default=str(REFILL_DIR))
    a = ap.parse_args(argv)
    if a.card in base.REFUSED_CARDS or a.card not in base.CARDS:
        print(f"REFUSED card {a.card}: {base.REFUSED_CARDS.get(a.card, 'unknown card')}", flush=True)
        return 3
    r = RefillRunner(a)
    return r.run() if a.command == "run" else r.probe()


if __name__ == "__main__":
    sys.exit(main())
