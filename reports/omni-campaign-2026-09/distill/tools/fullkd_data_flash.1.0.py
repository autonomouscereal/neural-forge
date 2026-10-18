#!/usr/bin/env python3
"""fullkd_data_flash.py -- bring Qwen3.8-Flash-Next into the full-KD text teacher, corpus by corpus, as it is scored.

  mix-ready   for every source of a full-KD data config whose Q38 cache has a SEALED Flash Next cache over the same
              corpus (gguf_teacher_cache, teacher flash-next-strata-iq3s), build the Q38 + Flash Next mixture cache
              (mix_teacher_cache.py; weights 0.25 Q38 / 0.75 Flash Next: Cyril's "Q38 small weight") if it is missing.
              Idempotent; meant to be re-run (flash_mix_queue.sh polls it while the Flash Next scoring runs).
  config      write a new data config: every source whose mixture is sealed points its qtext_cache at the mixture
              (text and K3 sources alike: on K3 rows the mixture is the same-tokenizer anchor beside K3); every other
              source is unchanged.  Prints what changed.  Adopting it in a running job is a planned resume under
              --allow-config-change --allow-identity-change (recorded in every later receipt).

The mixture is gradient-exact for the weighted sum of the two teachers' KL terms (mix_teacher_cache.py docstring), so
the trainer and its objective weights are unchanged: w_q38 (text rows) and lambda_k3 * anchor (K3 rows) now weigh the
two-teacher mixture instead of Q38 alone.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sealed(path: Path) -> bool:
    try:
        m = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(m.get("all_finite") and m.get("all_structural_ok") and m.get("teacher", {}).get("equal_before_and_after_every_session"))


def corpus_sha(cache: Path) -> str:
    return json.loads((cache / "manifest.json").read_text(encoding="utf-8"))["corpus"]["sha256"]


def plan(args: argparse.Namespace) -> list[dict]:
    spec = json.loads(Path(args.data_config).read_text(encoding="utf-8"))
    rows = []
    for source in spec["sources"]:
        qc = source.get("qtext_cache")
        if not qc or not (Path(qc) / "manifest.json").is_file():
            rows.append({"name": source["name"], "status": "no sealed Q38 cache (deferred or missing)"})
            continue
        sha = corpus_sha(Path(qc))
        flash = Path(args.flash_root) / args.flash_teacher / sha
        mix = Path(args.mix_root) / args.mix_teacher / sha
        rows.append({"name": source["name"], "q38": qc, "corpus_sha256": sha, "flash": str(flash), "mix": str(mix),
                     "flash_sealed": sealed(flash), "mix_sealed": sealed(mix)})
    return rows


def mix_ready(args: argparse.Namespace) -> int:
    rc = 0
    done = set()
    for row in plan(args):
        if "flash" not in row or not row["flash_sealed"] or row["mix_sealed"] or row["corpus_sha256"] in done:
            continue
        done.add(row["corpus_sha256"])
        log(f"mix {row['name']}: {row['q38']} + {row['flash']}")
        out = subprocess.run([PY, str(HERE / "mix_teacher_cache.py"), "mix", "--a", row["q38"], "--b", row["flash"],
                              "--weight-a", str(args.weight_q38), "--weight-b", str(args.weight_flash),
                              "--teacher", args.mix_teacher, "--out-root", args.mix_root], capture_output=True, text=True)
        log(out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr.strip()[-300:])
        if out.returncode != 0:
            rc = out.returncode
            continue
        check = subprocess.run([PY, str(HERE / "mix_teacher_cache.py"), "check", "--cache", row["mix"], "--positions", "200"],
                               capture_output=True, text=True)
        if check.returncode != 0:
            log(f"CHECK FAILED for {row['mix']}: {check.stdout[-400:]}")
            rc = 2
    rows = plan(args)
    log(f"mixtures sealed: {sum(1 for r in rows if r.get('mix_sealed'))} / {sum(1 for r in rows if 'mix' in r)} sources")
    return rc


def make_config(args: argparse.Namespace) -> int:
    spec = json.loads(Path(args.data_config).read_text(encoding="utf-8"))
    by_name = {r["name"]: r for r in plan(args)}
    changed = []
    for source in spec["sources"]:
        row = by_name.get(source["name"], {})
        if row.get("mix_sealed"):
            changed.append({"name": source["name"], "from": source["qtext_cache"], "to": row["mix"]})
            source["qtext_cache"] = row["mix"]
    spec.setdefault("notes", []).append(
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: fullkd_data_flash.py: {len(changed)} sources moved from the "
        f"Q38 cache to the Q38 {args.weight_q38} / Flash Next {args.weight_flash} mixture ({args.mix_teacher}); "
        "gradient-exact for the weighted sum of both teachers' KL terms (mix_teacher_cache.py).")
    Path(args.out).write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(json.dumps({"out": args.out, "changed": changed}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("mix-ready", "config", "status"))
    ap.add_argument("--data-config", default="/mnt/lexar/omni-campaign/distill/fullkd/fullkd_data_v3.json")
    ap.add_argument("--flash-root", default="/home/cereal/omni-campaign/distill/teacher_cache")
    ap.add_argument("--flash-teacher", default="flash-next-strata-iq3s")
    ap.add_argument("--mix-root", default="/home/cereal/omni-campaign/distill/teacher_cache")
    ap.add_argument("--mix-teacher", default="mix-q38w0.25-flashnextw0.75")
    ap.add_argument("--weight-q38", type=float, default=0.25)
    ap.add_argument("--weight-flash", type=float, default=0.75)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.command == "mix-ready":
        return mix_ready(args)
    if args.command == "config":
        if not args.out:
            raise SystemExit("--out is required for config")
        return make_config(args)
    rows = plan(args)
    for r in rows:
        print(f"{r['name'][:34]:34s} flash {'Y' if r.get('flash_sealed') else '-'} mix {'Y' if r.get('mix_sealed') else '-'} {r.get('status', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
