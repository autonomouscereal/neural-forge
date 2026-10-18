#!/usr/bin/env python3
"""Register lane L-RESEARCH in $Q/LANES.json with lanes.py's documented `init --seed ... --replace` (lanes.py has no
add-lane verb).  Every existing lane object is carried verbatim; lanes.py backs the old registry up itself; this script
refuses to call it a success unless that backup's sha256 equals the registry it read (no concurrent write lost).
Run under flock $Q/.lanes_pulse.lock so the 10-min pulse cannot interleave."""
import hashlib
import json
import re
import subprocess
import sys
import time

Q = "/home/cereal/omni-campaign/distill/queues"
F = f"{Q}/LANES.json"
LANES = "/home/cereal/omni-campaign/distill/tools/lanes.py"
X = "/mnt/lexar/omni-campaign/research/fold/x3-08b"
stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

pre = open(F, "rb").read()
pre_sha = hashlib.sha256(pre).hexdigest()
reg = json.loads(pre)
if "L-RESEARCH" in reg["lanes"]:
    sys.exit("L-RESEARCH already registered")
copy = f"{Q}/lanes_backups/LANES.pre-L-RESEARCH.{stamp}.json"
open(copy, "wb").write(pre)

lane = {
    "lane": "L-RESEARCH",
    "owner": "claude",
    "scope": ("Intelligence-density and data research program (reports/omni-campaign-2026-09/research/"
              "DENSITY_AND_DATA_PROGRAM_2026-10-14.md): track A fold experiments on ladder parents (x3_queue.sh on "
              "GPU3: X1-X3, H4, H8), track B data scoring and synthetic-data pilots (quality_scorer, vmath_queue.sh "
              "on GPU1/GPU2 after L-TOOL), track C efficiency readings.  Background only: never takes a card, RAM or "
              "disk a v1 lane needs; findings reach v1 only as written amendments to L-TRAIN / L-DATA."),
    "lock_paths": [f"{Q}/p40.claims", f"{Q}/gpu3.lock"],
    "status": (f"registered {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by Claude (lead): x3_queue.sh 1.0.0 "
               "pid 772203 (x3-08b, GPU3 via p40.claims) and vmath_queue.sh 1.0.1 pid 836018 (waits for GPU1/GPU2); "
               "x3_v1_yield_watchdog 1.0.0 being armed (the next heartbeat names its pid)"),
    "transferable": False,
    "transferable_reason": ("research queues supervised by x3_v1_yield_watchdog (it owns $X/STOP_QUEUE and "
                            "$X/V1_YIELD.json and relaunches the x3 queue); a second agent's launch could fork the x3 "
                            "lineage on GPU3 or undo a v1 yield.  Moves to Codex only on Cyril's word."),
    "next_command": ("none by hand: the watchdog stops the x3 queue on v1 pressure and relaunches it; read "
                     f"{X}/v1_yield_watchdog.log and v1_yield_watchdog.state.json; vmath_queue.sh starts itself when "
                     "L-TOOL releases GPU1/GPU2"),
    "do_not_touch": (f"the x3 queue, its stages and the watchdog (stop for good: touch {X}/STOP_QUEUE, then kill -TERM "
                     f"the watchdog); {X}/STOP_QUEUE and V1_YIELD.json while they are the watchdog's; GPU0/4/6 (v1), "
                     "GPU5 (never)"),
    "probe": {"checks": [{"kind": "pid_cmdline", "match": "research/fold/tools/x3_v1_yield_watchdog.py"}],
              "receipt_globs": [f"{X}/eval/*/grade.json", f"{X}/SUMMARY.json",
                                "/mnt/lexar/omni-campaign/research/data/*/receipt.json",
                                "/mnt/lexar/omni-campaign/research/data/*/d2_*.json"],
              "detail": {"path": f"{X}/v1_yield_watchdog.log"}},
    "lead": {"agent": "claude", "ruling": "cyril-claude-leads-all-lanes-2026-10-14",
             "rulings_file": "/home/cereal/omni-campaign/distill/STAGE_D_RULINGS_2026-10-12.md"},
    "codex_role": ("contributor: open fold hypotheses go in section 11 of DENSITY_AND_DATA_PROGRAM_2026-10-14.md; "
                   f"work only through {Q}/CODEX_TASKS.md"),
}
seed = {"created_by": reg.get("created_by", "unknown"), "about": reg.get("about", ""), "rules": reg.get("rules", {}),
        "lanes": list(reg["lanes"].values()) + [lane]}
seed_path = f"{Q}/lanes_backups/LANES.seed-L-RESEARCH-{stamp}.json"
with open(seed_path, "w") as handle:
    json.dump(seed, handle, indent=1)
r = subprocess.run(["/usr/bin/python3", LANES, "--file", F, "init", "--seed", seed_path, "--replace"],
                   capture_output=True, text=True, timeout=60)
print(r.stdout.strip(), r.stderr.strip(), f"rc={r.returncode}")
m = re.search(r"backup (\S+) sha256 ([0-9a-f]{64})", r.stdout)
if r.returncode != 0 or not m:
    sys.exit("init failed")
print(f"pre-read sha {pre_sha}\ninit backup  {m.group(2)} ({m.group(1)})\ncopy {copy}\nseed {seed_path}")
if m.group(2) != pre_sha:
    sys.exit("MISMATCH: the registry changed between the read and init; re-merge from the backup")
new = json.load(open(F))
same = all(new["lanes"][k] == v for k, v in reg["lanes"].items())
print(f"lanes now: {list(new['lanes'])}; existing lanes byte-equal in content: {same}; revision {new['revision']}")
sys.exit(0 if same else 3)
