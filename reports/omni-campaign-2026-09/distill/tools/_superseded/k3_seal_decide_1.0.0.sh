#!/bin/bash
# $OC/distill/queues/k3_seal_decide.sh -- the K3-first / D2-first decision at the D1 seal, UNATTENDED.  Lane sealdecide,
# Claude, 2026-10-15.  Automates K3_ACCELERATION_PLAN_2026-10-14 section 4.5 (and section 4.4's planning-engine line,
# install step 6) so no agent has to be awake when D1 seals.  post_d1_queue.sh >= 1.4.0 runs it as job `k3decide`,
# right after the P2 verdict (job p2).
#
#   k3_seal_decide.sh              decide and apply (exit 0: decided; 75: a gate is not ready, nothing written; 5: another
#                                  instance; 2: refused)
#   k3_seal_decide.sh --gate-once  the waiting gates only (d1, p2, then B2 pending and a live GPU0 runner): prints "ready"
#                                  (exit 0) or the reasons (exit 75); writes nothing.  post_d1 calls it before each try
#   k3_seal_decide.sh --dry-run    gates, the resolved k3_multicard plan, the arithmetic on whatever eqcheck log exists
#                                  and the branch it would take; writes NOTHING
#   k3_seal_decide.sh --stagger <gpu> <after gpu>   (internal, detached) start `k3_multicard.sh <gpu>` once the runner on
#                                  <after gpu> has started its first pass and its RAM ballast has settled (or it ended)
#
# GATES (each failing gate: exit 75, nothing written; post_d1 defers the job and retries):
#   d1      $Q/d1.sealed exists AND claude-d1-hardened.service is not active / activating (D1 never holds the cards)
#   p2      $Q/post_d1/p2.json exists (any rc: "after the P2 verdict") and no run_p2_v133.sh process lives
#   b2      the B2 PASS of the frozen 1.2.4 (k3_lib b2_ok on k3_multicard's own B2 dir).  B2 FAILED (a parity or identity
#           receipt that is not a pass) is not a wait: the decision is D2-first (no card can run K3)
# STEPS:
#   1. eqcheck: writes $Q/k3_multicard.go = "k3-multicard-go / gpus=0 / eqcheck_only=1" and runs `k3_multicard.sh 0`
#      (its own gates, gpu0.lock, the card-class check of ruling k3-card-class-equivalence-2026-10-15).  The runner is
#      watched: if it has not printed "eqcheck START" / "eqcheck PASS" within K3D_EQ_START_MAX_S (6 h) it is stopped
#      (no GPU job has begun) and the job defers.
#   2. arithmetic (section 4.5, constants below, every input and number written to the decision):
#      3090 pass   = sum over the eqcheck's 93 printed layers of  c' + max(0, 17 GB / 250 MB/s - c'),
#                    c' = compute_s x (pass positions / eqcheck positions) x 1.10      (rule step 2)
#      GPU6        = layers per hour from its current pass log (the last K3D_RATE_WINDOW layer lines), remaining =
#                    (93 - layers done) / rate; a full GPU6 pass = 93 / rate, else the last completed GPU6 pass from
#                    gpu6.status, else K3D_GPU6_FALLBACK_H (plan section 3)                  (rule step 3)
#      D2-first    = max(D2 start + 20.75 h, K3 sealed by GPU6 alone) + 15.7 h               (rule step 4)
#      K3-first    = K3 sealed with GPU6's current pass + GPU0 + GPU4, at most K3D_MAX_K3_PASSES (2) K3 passes at once
#                    (host RAM, plan section 4.3), greedy, + 36.4 h                          (rule step 4)
#      D2 start = the decision time (+ K3D_D2_START_DELAY_H).  The earlier finish wins; a tie is D2-first.
#   3. apply, $Q/k3_seal_decision.json written first (applied=false), then each action, then applied=true:
#      k3-first  $Q/k3_gpu6.max_passes = the passes the LIVE gpu6 queue has started (queue_gpu6 1.4.0 re-reads it before
#                every pick: GPU6 ends its current pass, then goes on to H3 and j6), $Q/k3_multicard.go = gpus=0,4 (no
#                eqcheck_only), `k3_multicard.sh 0` detached at once, `k3_multicard.sh 4` detached through --stagger
#                (after GPU0's first pass has started and its ballast settled: two runners never pass the MemAvailable
#                gate together); the runners' own wait_ram (pass footprint + headroom) does the rest
#      d2-first  $Q/k3_multicard.go removed (copied to $Q/backups/ first): GPU0 / GPU4 are D2's.  A $Q/k3_gpu6.max_passes
#                below 4 is moved to $Q/backups/ (GPU6 must run every remaining pass)
#      planning  (P2 verdict engine=split: D2 runs on GPU6 + GPU4, plan section 4.4 / install step 6) max_passes as
#                k3-first, $Q/k3_multicard.go = gpus=0, `k3_multicard.sh 0` only
#      manual    a GO file this script did not write (no "by=k3_seal_decide" line), or a K3 pass already live on GPU0 /
#                GPU4: the lead decided by hand; recorded, nothing changed
#   K3-first / planning are REFUSED (-> d2-first, the reason recorded) unless: the 3090 class passed its eqcheck; no
#   live gpu6 queue older than queue_gpu6.sh 1.4.0 (1.3.x ignores max_passes and the pass claims: a 3090 runner would
#   collide with it); and, when the 3090 passes would run k3_teacher_cache 1.3.0, d2_plan reads $Q/k3_pass_tools.json
#   (d2_plan 1.2.0 counts only passes sealed by the frozen 1.2.4; plan section 6).
# A decision already applied is never redone (exit 0, printed).  Status lines: $Q/k3_seal_decide.status; lane L-K3 is
# heartbeated with each (lanes.py, 20 s cap, never fatal); post_d1's own lines heartbeat L-D1.
# Never kills a job it did not start; never touches GPU5, GPU6's jobs, the gpu6 queue, D1 or the P40s.
set -u
VERSION="k3_seal_decide.sh 1.0.0"
OC=${K3D_OC:-/home/cereal/omni-campaign}
Q=${K3D_Q:-$OC/distill/queues}
T=${K3D_T:-$OC/distill/tools}
LEXAR=/mnt/lexar/omni-campaign
MC=${K3D_MC:-$Q/k3_multicard.sh}
K3LIB=${K3D_K3LIB:-$T/k3_lib.sh}
K3TOOL=${K3D_K3TOOL:-$Q/frozen/k3_teacher_cache-gpu6.py}
STATUS=${K3D_STATUS:-$Q/k3_seal_decide.status}
DECISION=${K3D_DECISION:-$Q/k3_seal_decision.json}
GO_FILE=${K3D_GO:-$Q/k3_multicard.go}
MAXP_FILE=${K3D_MAXP:-$Q/k3_gpu6.max_passes}
D1_SEALED=${K3D_D1_SEALED:-$Q/d1.sealed}
D1_UNIT=${K3D_D1_UNIT:-claude-d1-hardened.service}
SYSTEMCTL=${K3D_SYSTEMCTL:-systemctl}
P2_RECEIPT=${K3D_P2_RECEIPT:-$Q/post_d1/p2.json}
P2_VERDICT=${K3D_P2_VERDICT:-$Q/p2.verdict}
GPU6_STATUS=${K3D_GPU6_STATUS:-$Q/gpu6.status}
GPU6_LOGS=${K3D_GPU6_LOGS:-$Q/logs/gpu6}
MC_LOGS=${K3D_MC_LOGS:-$Q/logs}          # k3_multicard writes $MC_LOGS/k3mc-gpu<N>/ and $Q/k3mc-gpu<N>.status
MC_STATUS_DIR=${K3D_MC_STATUS_DIR:-$Q}
KC=${K3D_KC:-$OC/distill/corpus/k3}
D2_PLAN=${K3D_D2_PLAN:-$T/d2_plan.py}
BACKUPS=${K3D_BACKUPS:-$Q/backups}
LOCK=${K3D_LOCK:-$Q/k3_seal_decide.lock}
LANES_PY=${LANES_PY:-$T/lanes.py}
LANES_FILE=${LANES_FILE:-$Q/LANES.json}
POLL_S=${K3D_POLL_S:-60}
EQ_START_MAX_S=${K3D_EQ_START_MAX_S:-21600}
STAGGER_SETTLE_S=${K3D_STAGGER_SETTLE_S:-1800}
V130_PIN=${K3_V130_SHA:-434ced6d7e8e22e4179bb2332d4695803dd89a38407805126261500c112fa3c7}
# the section 4.5 constants (plan, stated before the numbers existed); env overrides exist for tests and a lead's rerun
export K3D_ATTN_FACTOR=${K3D_ATTN_FACTOR:-1.10} K3D_LAYER_GB=${K3D_LAYER_GB:-17} K3D_READ_MBS=${K3D_READ_MBS:-250} \
       K3D_D2_TO_20M_H=${K3D_D2_TO_20M_H:-20.75} K3D_D2_REST_H=${K3D_D2_REST_H:-15.7} K3D_D2_FULL_H=${K3D_D2_FULL_H:-36.4} \
       K3D_D2_START_DELAY_H=${K3D_D2_START_DELAY_H:-0} K3D_GPU6_FALLBACK_H=${K3D_GPU6_FALLBACK_H:-36.85} \
       K3D_RATE_WINDOW=${K3D_RATE_WINDOW:-12} K3D_MAX_K3_PASSES=${K3D_MAX_K3_PASSES:-2} K3D_LAYERS=${K3D_LAYERS:-93}
MODE=armed
case "${1:-}" in
  --gate-once) MODE=gate ;;
  --dry-run) MODE=dry ;;
  --stagger) MODE=stagger; ST_GPU=${2:-}; ST_AFTER=${3:-} ;;
  "") ;;
  *) echo "usage: k3_seal_decide.sh [--gate-once | --dry-run | --stagger <gpu> <after gpu>]"; exit 2 ;;
esac

utc() { date -u +%FT%TZ; }
say() { echo "[$(utc)] $*"; }
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
lane_hb() {  # $1 status $2 detail: lane L-K3, armed only; never fatal, at most 20 s, silent with no registry
  [ "$MODE" = armed ] || [ "$MODE" = stagger ] || return 0
  [ -f "$LANES_FILE" ] && [ -f "$LANES_PY" ] || return 0
  timeout 20 python3 "$LANES_PY" --file "$LANES_FILE" heartbeat --lane L-K3 --agent claude --source "$VERSION" \
    --status "$1" --detail "${2:0:400}" > /dev/null 2>&1 || say "lane heartbeat L-K3 failed (ignored)"
  return 0
}
status() {  # one line in $STATUS (armed / stagger only) + the L-K3 heartbeat
  say "STATUS $*"
  if [ "$MODE" = armed ] || [ "$MODE" = stagger ]; then
    echo "$(utc) k3decide $*" >> "$STATUS"; lane_hb "K3 seal decision ($VERSION): ${*:0:120}" "$*"
  fi
  return 0
}
lock_free() { [ -e "$1" ] || return 0; ( exec 3< "$1" && flock -n 3 ) 2>/dev/null; }
procs() {  # "pid args" of every process; K3D_PROCS (a file) replaces the process table in tests
  if [ -n "${K3D_PROCS:-}" ]; then cat "$K3D_PROCS" 2>/dev/null; else ps -eo pid=,args= 2>/dev/null; fi
}
runner_pids() {  # $1 gpu: live `k3_multicard.sh <gpu>` runners and --stagger <gpu> helpers
  procs | awk -v g="$1" '($0 ~ "k3_multicard\\.sh " g "( |$)") || ($0 ~ "k3_seal_decide\\.sh --stagger " g " ") {print $1}'
}
d1_active() {  # prints the unit state when D1 still holds (or may hold) the cards
  local st
  st=$($SYSTEMCTL show "$D1_UNIT" -p ActiveState --value 2>/dev/null)
  case "$st" in active|activating|reloading|deactivating) echo "$st" ;; esac
}

# ------------------------------------------------------------------------------------------------ k3_multicard plan
# the runner resolves its own tool, B2 dir and K3 out dir; this script reads them from its plan selftest (one source)
resolve_plan() {  # sets PLAN_OK PLAN_KIND PLAN_SHA16 PLAN_CLASS B2_DIR REBASE K3OUT EQV PLAN_TEXT
  PLAN_OK=0
  [ -s "$MC" ] || { PLAN_WHY="$MC missing"; return 1; }
  PLAN_TEXT=$(K3MC_SELFTEST=plan timeout 300 bash "$MC" 0 2>&1)
  local l1
  l1=$(echo "$PLAN_TEXT" | grep -m1 "^k3_multicard\.sh .* gpu 0 (")
  [ -n "$l1" ] || { PLAN_WHY="k3_multicard plan unreadable: $(echo "$PLAN_TEXT" | tail -2 | tr '\n' ' ')"; return 1; }
  PLAN_CLASS=$(echo "$l1" | sed -n 's/^.* gpu 0 (\([^)]*\)).*$/\1/p')
  PLAN_KIND=$(echo "$l1" | sed -n 's/^.*: tool \(v[0-9]*\) .*$/\1/p')
  PLAN_SHA16=$(echo "$l1" | sed -n 's/^.* \([0-9a-f]\{16\}\), group tokens.*$/\1/p')
  B2_DIR=$(echo "$l1" | sed -n 's/^.*, B2 dir \(.*\) (rebase \([01]\))$/\1/p')
  REBASE=$(echo "$l1" | sed -n 's/^.* (rebase \([01]\))$/\1/p')
  K3OUT=$(echo "$PLAN_TEXT" | sed -n 's/^K3 out \([^;]*\);.*$/\1/p' | head -1)
  EQV=$(echo "$PLAN_TEXT" | sed -n 's/^gates: .* eqcheck \(.*\) in .*$/\1/p' | head -1)
  if [ -z "$PLAN_CLASS" ] || [ -z "$PLAN_KIND" ] || [ -z "$PLAN_SHA16" ] || [ -z "$B2_DIR" ] || [ -z "$K3OUT" ]; then
    PLAN_WHY="k3_multicard plan incomplete: class '$PLAN_CLASS' tool '$PLAN_KIND' '$PLAN_SHA16' b2 '$B2_DIR' out '$K3OUT'"; return 1
  fi
  PLAN_OK=1
}
b2_state() {  # prints pass | failed | pending (B2 dir from the plan, frozen 1.2.4 sha, rebase)
  local k3sha
  k3sha=$(sha "$K3TOOL")
  [ -n "$k3sha" ] || { echo "pending"; return; }
  ( . "$K3LIB" >/dev/null 2>&1 && b2_ok "$B2_DIR" "$k3sha" "$REBASE" ) && { echo pass; return; }
  python3 - "$B2_DIR" <<'PY' 2>/dev/null || echo pending
import json, os, sys
d = sys.argv[1]
def verdict(path, key):
    try:
        return json.load(open(path)).get(key)
    except (OSError, ValueError):
        return "absent"
p = verdict(os.path.join(d, "parity_b2_receipt.json"), "verdict")
i = verdict(os.path.join(d, "identity", "b2_identity.json"), "verdict")
failed = (p not in ("absent", None, "pass")) or (isinstance(i, dict) and i.get("pass") is False)
print("failed" if failed else "pending")
PY
}

# ------------------------------------------------------------------------------------------------ gates
gate_why() {  # prints why the decision cannot be made now (empty: ready); never writes
  local why="" st
  [ -e "$D1_SEALED" ] || why="$why no_$(basename "$D1_SEALED")"
  st=$(d1_active); [ -n "$st" ] && why="$why D1_unit_$st"
  [ -e "$P2_RECEIPT" ] || why="$why no_P2_receipt($P2_RECEIPT)"
  procs | grep -q "run_p2_v133\.sh" && why="$why P2_running"
  echo "${why# }"
}

# ------------------------------------------------------------------------------------------------ the arithmetic
estimate() {  # env in: EQ_LOG PLAN_SHA16 K3OUT KC GPU6_STATUS GPU6_LOGS P2_VERDICT PRE_* ; prints the estimate JSON
  python3 - <<'PY'
import calendar, json, os, re, struct, time
E = os.environ.get
F = lambda k: float(E(k))
now = float(E("K3D_NOW") or time.time())
iso = lambda t: None if t is None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
H = 3600.0
NL = int(F("K3D_LAYERS"))
out = {"schema": "omni-k3-seal-estimate-v1", "now_utc": iso(now),
       "constants": {k: F(k) for k in ("K3D_ATTN_FACTOR", "K3D_LAYER_GB", "K3D_READ_MBS", "K3D_D2_TO_20M_H",
                                       "K3D_D2_REST_H", "K3D_D2_FULL_H", "K3D_D2_START_DELAY_H", "K3D_GPU6_FALLBACK_H",
                                       "K3D_RATE_WINDOW", "K3D_MAX_K3_PASSES", "K3D_LAYERS")}}
LINE = re.compile(r"\] L(\d+) (KDA|MLA): read ([\d.]+)s \(([\d.]+) MB/s\), compute ([\d.]+)s over ([\d,]+) positions"
                  r" .*?; ([\d,]+) pos/s\)")
STAMP = re.compile(r"^\[\S+ (\d\d):(\d\d):(\d\d)\]")
def load(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None

# -- 1. the 3090 pass from the eqcheck's printed layers (segments of this tool sha only; a resumed run continues)
eq_log, sha16 = E("EQ_LOG", ""), E("PLAN_SHA16", "")
layers, take = {}, False
if os.path.isfile(eq_log):
    for line in open(eq_log, errors="replace"):
        if line.startswith("== eqcheck "):
            take = f" {sha16} on " in line
            continue
        m = LINE.search(line) if take else None
        if m:
            layers[int(m.group(1))] = {"kind": m.group(2), "read_s": float(m.group(3)), "read_MBps": float(m.group(4)),
                                       "compute_s": float(m.group(5)), "positions": int(m.group(6).replace(",", "")),
                                       "pos_per_s": int(m.group(7).replace(",", ""))}
missing = [l for l in range(NL) if l not in layers]
eq = {"log": eq_log, "tool_sha16": sha16, "layers_printed": len(layers), "layers_missing": missing[:10],
      "compute_s_total": round(sum(v["compute_s"] for v in layers.values()), 1),
      "read_s_total": round(sum(v["read_s"] for v in layers.values()), 1),
      "positions": sorted({v["positions"] for v in layers.values()})}
if layers:
    mbs = sorted(v["read_MBps"] for v in layers.values())
    eq["read_MBps_median_diagnostic"] = mbs[len(mbs) // 2]
    eq["kda_pos_per_s"] = [min((v["pos_per_s"] for v in layers.values() if v["kind"] == "KDA"), default=None),
                           max((v["pos_per_s"] for v in layers.values() if v["kind"] == "KDA"), default=None)]
    eq["mla_pos_per_s"] = [min((v["pos_per_s"] for v in layers.values() if v["kind"] == "MLA"), default=None),
                           max((v["pos_per_s"] for v in layers.values() if v["kind"] == "MLA"), default=None)]
out["eqcheck"] = eq
read_s = F("K3D_LAYER_GB") * 1e9 / (F("K3D_READ_MBS") * 1e6)
def pass_3090_h(positions):
    if missing:
        return None
    tot = 0.0
    for v in layers.values():
        c = v["compute_s"] * (positions / v["positions"]) * F("K3D_ATTN_FACTOR")
        tot += c + max(0.0, read_s - c)
    return tot / H

# -- 2. the K3 passes: positions, sealed, where they run
allowed = {a for a in (E("K3SHA", ""), E("V130_PIN", "")) if a}
k3out, kc = E("K3OUT", ""), E("KC", "")
procs = []
if E("K3D_PROCS"):
    procs = open(E("K3D_PROCS")).read().splitlines() if os.path.isfile(E("K3D_PROCS")) else []
else:
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/cmdline", "rb") as handle:
                    procs.append(d + " " + handle.read().replace(b"\0", b" ").decode("utf-8", "replace"))
            except OSError:
                pass
pid_args = {}
for p in procs:
    pid, _, args = p.strip().partition(" ")
    if pid.isdigit():
        pid_args[int(pid)] = args + " "
RUN = re.compile(r"--pass-dir (\S+)/pass-(\d)-of-4 .*--state-dir (\S+) ")
passes = {}
override = [int(x) for x in E("K3D_PASS_POSITIONS", "").split(",") if x.strip()]
for k in range(4):
    pd = os.path.join(k3out, f"pass-{k}-of-4")
    man = load(os.path.join(pd, "manifest.json")) or {}
    if override:
        positions = override[k]
    else:
        try:
            with open(os.path.join(kc, f"pass-{k}", "k3_input_ids.safetensors"), "rb") as handle:
                head = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
            rows, T = head["input_ids"]["shape"][:2]
            positions = int(rows) * int(T)
        except (OSError, ValueError, KeyError):
            positions = None
    where = None
    for pid, args in pid_args.items():
        m = RUN.search(args)
        if m and "k3_teacher_cache" in args and " run " in args and m.group(1) == k3out and int(m.group(2)) == k:
            g = re.search(r"-gpu(\d+)$", m.group(3))
            where = {"gpu": int(g.group(1)) if g else 6, "pid": pid, "state": "running"}
    claim = load(pd + ".claim")
    if where is None and claim and int(claim.get("pid", 0)) in pid_args and claim.get("match", "\0") in pid_args[int(claim["pid"])]:
        where = {"gpu": int(claim.get("gpu", -1)), "pid": int(claim["pid"]), "state": "claimed"}
    passes[k] = {"positions": positions, "sealed": man.get("verdict") == "sealed" and man.get("tool_sha256") in allowed,
                 "sealed_by": (man.get("tool_sha256") or "")[:16] or None, "live": where}
out["passes"] = passes

# -- 3. GPU6: the live queue (version, passes started), its current pass and rate
st_lines = open(E("GPU6_STATUS")).read().splitlines() if os.path.isfile(E("GPU6_STATUS", "")) else []
qidx = max((i for i, l in enumerate(st_lines) if " queue START " in l), default=-1)
qver = None
if qidx >= 0:
    m = re.search(r"queue_gpu6\.sh (\d+)\.(\d+)\.(\d+)", st_lines[qidx])
    qver = [int(x) for x in m.groups()] if m else None
q_alive = any(re.search(r"bash \S*/gpu6\.sh( |$)", a) for a in pid_args.values())
started_live = sum(1 for l in st_lines[qidx + 1:] if re.match(r"^\S+ k3-pass-[0-3] START ", l))
def stamp(line):
    try:
        return calendar.timegm(time.strptime(line.split()[0], "%Y-%m-%dT%H:%M:%SZ"))
    except (ValueError, IndexError):
        return None
done_h = []
for k in range(4):
    starts = [stamp(l) for l in st_lines if re.match(rf"^\S+ k3-pass-{k} START ", l)]
    ends = [stamp(l) for l in st_lines if re.match(rf"^\S+ k3-pass-{k} PASS ", l)]
    if starts and ends and starts[0] is not None and ends[-1] is not None and ends[-1] > starts[0]:
        done_h.append({"pass": k, "hours": round((ends[-1] - starts[0]) / H, 2)})
cur = next((k for k, p in passes.items() if p["live"] and p["live"]["gpu"] == 6 and not p["sealed"]), None)
g6 = {"queue_alive": q_alive, "queue_version": ".".join(map(str, qver)) if qver else None,
      "passes_started_by_live_queue": started_live, "current_pass": cur, "completed_pass_hours": done_h}
rate = None
if cur is not None:
    logf = os.path.join(E("GPU6_LOGS", ""), f"k3-pass-{cur}.log")
    pts, seen = [], -1
    if os.path.isfile(logf):
        for line in open(logf, errors="replace"):
            m, s = LINE.search(line), STAMP.match(line)
            if m and s:
                pts.append((int(m.group(1)), int(s.group(1)) * 3600 + int(s.group(2)) * 60 + int(s.group(3))))
    done = (max(l for l, _ in pts) + 1) if pts else 0
    win = pts[-int(F("K3D_RATE_WINDOW")):]
    t, prev, span = 0, None, 0
    for _, s in win:                       # the log prints host-local HH:MM:SS without a date: unwrap midnight
        if prev is not None:
            span += (s - prev) % 86400
        prev = s
    if len(win) >= 3 and span > 0:
        rate = (len(win) - 1) * H / span
    g6.update({"log": logf, "layers_done": done, "rate_window_lines": len(win),
               "layers_per_hour": round(rate, 3) if rate else None})
if rate:
    full6, src6 = NL / rate, "current pass log"
elif done_h:
    full6, src6 = done_h[-1]["hours"], "last completed GPU6 pass (gpu6.status START->PASS)"
else:
    full6, src6 = F("K3D_GPU6_FALLBACK_H"), "fallback constant (plan section 3: 1.2.4 throttled 29.8-43.9 h, midpoint)"
if cur is not None:
    rem6 = (NL - g6["layers_done"]) / rate if rate else full6
    if passes[cur]["live"]["state"] == "claimed":
        rem6 = full6                       # claimed, waiting for RAM / the card: the whole pass is ahead
else:
    rem6 = 0.0
g6.update({"full_pass_hours": round(full6, 2), "full_pass_source": src6, "current_remaining_hours": round(rem6, 2)})
out["gpu6"] = g6

# -- 4. the two finishes (section 4.5 step 4)
todo = [k for k, p in passes.items() if not p["sealed"]]
elsewhere = [k for k in todo if passes[k]["live"] and passes[k]["live"]["gpu"] != 6]
rest = [k for k in todo if k != cur]
c = out["constants"]
d2_start = c["K3D_D2_START_DELAY_H"]
p3 = {k: pass_3090_h(passes[k]["positions"]) for k in rest if passes[k]["positions"]}
fin = {"todo": todo, "gpu6_current": cur, "for_3090": rest, "live_elsewhere": elsewhere,
       "pass_3090_hours": {k: round(v, 2) for k, v in p3.items() if v is not None}}
if not todo:
    fin["k3_complete"] = True
k3_g6 = rem6 + len(rest) * full6 if q_alive or cur is not None else None
if not todo:
    k3_g6 = 0.0
d2f = None if k3_g6 is None else max(d2_start + c["K3D_D2_TO_20M_H"], k3_g6) + c["K3D_D2_REST_H"]
k3f, sched = None, []
if rest and len(p3) == len(rest) and all(v is not None for v in p3.values()):
    busy = [(0.0, rem6)] if cur is not None else []
    free = {0: 0.0, 4: 0.0}
    for k in sorted(rest, reverse=True):          # k3_multicard claims from the top
        g = min(free, key=lambda x: (free[x], x))
        t0 = free[g]
        def over(t):                               # K3 passes running over [t, t + p3[k])
            pts = [t] + [s for s, e in busy if t < s < t + p3[k]]
            return max(sum(1 for s, e in busy if s <= x < e) for x in pts)
        cands = sorted({t0} | {e for s, e in busy if e > t0})
        t0 = next(t for t in cands if over(t) < int(c["K3D_MAX_K3_PASSES"]))
        busy.append((t0, t0 + p3[k]))
        free[g] = t0 + p3[k]
        sched.append({"pass": k, "gpu": g, "start_h": round(t0, 2), "end_h": round(t0 + p3[k], 2)})
    k3_3090 = max(e for s, e in busy)
    k3f = k3_3090 + c["K3D_D2_FULL_H"]
    fin["k3_sealed_3090_h"] = round(k3_3090, 2)
elif not rest and todo:
    k3f = rem6 + c["K3D_D2_FULL_H"]               # GPU6 holds the only pass left: nothing for the 3090s
    fin["k3_sealed_3090_h"] = round(rem6, 2)
fin.update({"schedule_3090": sched, "d2_start_h": d2_start,
            "k3_sealed_gpu6_only_h": None if k3_g6 is None else round(k3_g6, 2),
            "d2_first_finish_h": None if d2f is None else round(d2f, 2),
            "k3_first_finish_h": None if k3f is None else round(k3f, 2),
            "d2_first_finish_utc": iso(now + d2f * H) if d2f is not None else None,
            "k3_first_finish_utc": iso(now + k3f * H) if k3f is not None else None,
            "formulas": {"pass_3090": "sum_l c' + max(0, LAYER_GB/READ_MBS - c'), c' = compute_s x pos_pass/pos_eq x ATTN",
                         "d2_first": "max(D2 start + D2_TO_20M_H, K3 sealed by GPU6 alone) + D2_REST_H",
                         "k3_first": "K3 sealed with GPU6's current pass + GPU0 + GPU4 (<= MAX_K3_PASSES at once) + D2_FULL_H"}})
out["finish"] = fin

# -- 5. the branch
verdict = (open(E("P2_VERDICT")).read() if os.path.isfile(E("P2_VERDICT", "")) else "")
engine = next((l.split("=", 1)[1].strip() for l in verdict.splitlines() if l.startswith("engine=")), None)
out["engine"] = {"p2_verdict": E("P2_VERDICT"), "engine": engine}
blocked = []
if E("PRE_EQ") != "pass":
    blocked.append(f"3090 class eqcheck: {E('PRE_EQ')}")
if qver is not None and q_alive and qver < [1, 4, 0]:
    blocked.append(f"live gpu6 queue is queue_gpu6.sh {g6['queue_version']} (< 1.4.0 ignores k3_gpu6.max_passes and the pass claims)")
if E("PLAN_KIND") == "v130" and E("PRE_D2PLAN_V130") != "1":
    blocked.append("the 3090 passes would run 1.3.0 and d2_plan does not read k3_pass_tools.json (plan section 6)")
if elsewhere:
    blocked.append(f"K3 pass(es) {elsewhere} already live on a 3090")
out["k3_on_3090_blocked"] = blocked
if elsewhere:
    branch, why = "manual", f"K3 pass(es) {elsewhere} already run on a 3090: a manual launch, not overridden"
elif not todo:
    branch, why = "d2-first", "all four K3 passes are sealed: nothing for the 3090s"
elif engine == "split":
    if blocked:
        branch, why = "d2-first", "P2 engine split (planning, plan 4.4) but K3 on GPU0 is blocked: " + "; ".join(blocked)
    else:
        branch, why = "planning", "P2 engine split: D2 runs on GPU6 + GPU4, so GPU6 ends its current pass and GPU0 runs the rest (plan 4.4, install step 6)"
elif k3f is None:
    branch, why = "d2-first", "no K3-first estimate (eqcheck layers missing: %s)" % (missing[:5] or "pass positions unreadable")
elif d2f is not None and not k3f < d2f:
    branch, why = "d2-first", f"D2-first finishes first or ties ({d2f:.2f} h vs K3-first {k3f:.2f} h)"
elif blocked:
    branch, why = "d2-first", f"K3-first is earlier ({k3f:.2f} h vs {'unbounded' if d2f is None else f'{d2f:.2f} h'}) but blocked: " + "; ".join(blocked)
else:
    branch, why = "k3-first", f"K3-first finishes first ({k3f:.2f} h vs D2-first {'unbounded (no live GPU6 K3)' if d2f is None else f'{d2f:.2f} h'})"
out["branch"], out["reason"] = branch, why
out["gpu6_max_passes"] = started_live if branch in ("k3-first", "planning") else None
print(json.dumps(out, indent=1, sort_keys=True))
PY
}
jget() { python3 -c 'import json,sys; v=json.load(open(sys.argv[1])); [v := (v or {}).get(k) for k in sys.argv[2].split(".")]; print("" if v is None else v)' "$1" "$2" 2>/dev/null; }

# ------------------------------------------------------------------------------------------------ actions
write_atomic() {  # $1 path, stdin = content
  local tmp="$1.tmp$$"
  cat > "$tmp" && mv -f "$tmp" "$1"
}
backup() {  # $1 file -> $BACKUPS/<name>.<tag>-<utc>, prints the copy
  local b
  [ -e "$1" ] || return 0
  mkdir -p "$BACKUPS"
  b=$BACKUPS/$(basename "$1").$2-$(date -u +%Y%m%dT%H%M%SZ)
  cp -p "$1" "$b" && echo "$b"
}
launch() {  # $1 runner|stagger, rest: args.  Detached (setsid, own session), every queue lock fd closed.  K3D_LAUNCH stubs it.
  local kind=$1; shift
  if [ -n "${K3D_LAUNCH:-}" ]; then "$K3D_LAUNCH" "$kind" "$@"; return; fi
  mkdir -p "$MC_LOGS/k3mc-gpu$1"
  if [ "$kind" = runner ]; then
    setsid nohup bash "$MC" "$1" > "$MC_LOGS/k3mc-gpu$1/runner.log" 2>&1 < /dev/null 5>&- 7>&- 8>&- 9>&- &
  else
    K3D_STAGGER_FROM_LINE=${3:-} setsid nohup bash "$0" --stagger "$1" "$2" > "$MC_LOGS/k3mc-gpu$1/stagger.log" 2>&1 < /dev/null 5>&- 7>&- 8>&- 9>&- &
  fi
  echo $!
}
go_ours() {  # the GO file was written by this script (or is absent)
  [ ! -e "$GO_FILE" ] || grep -q "^by=k3_seal_decide" "$GO_FILE"
}
record() {  # $1 decision json in, $2 actions json (list) -> $DECISION (atomic)
  python3 - "$1" "$2" "$DECISION" <<'PY'
import json, os, sys
doc = json.load(open(sys.argv[1]))
doc["actions"] = json.loads(sys.argv[2]) if sys.argv[2] else doc.get("actions", [])
tmp = sys.argv[3] + ".tmp"
with open(tmp, "w") as handle:
    json.dump(doc, handle, indent=1, sort_keys=True)
os.replace(tmp, sys.argv[3])
PY
}

# ------------------------------------------------------------------------------------------------ --stagger
if [ "$MODE" = stagger ]; then
  case "$ST_GPU:$ST_AFTER" in 4:0|0:4) ;; *) echo "refuse: --stagger takes 4 0 or 0 4"; exit 2 ;; esac
  ST_FILE=$MC_STATUS_DIR/k3mc-gpu$ST_AFTER.status
  # only lines written after this decision launched the <after gpu> runner count (the eqcheck run left a "runner END")
  ST_N0=${K3D_STAGGER_FROM_LINE:-$(wc -l < "$ST_FILE" 2>/dev/null || echo 0)}
  status "stagger gpu$ST_GPU waits for gpu$ST_AFTER's first pass START + ballast (or its runner's end), $ST_FILE"
  t_start=""
  while :; do
    [ -n "$(d1_active)" ] && { sleep "$POLL_S"; continue; }
    pass=$(tail -n +$((ST_N0 + 1)) "$ST_FILE" 2>/dev/null | grep -o "gpu$ST_AFTER k3-pass-[0-3] START" | tail -1 | awk '{print $2}')
    if [ -n "$pass" ]; then
      [ -n "$t_start" ] || t_start=$(date +%s)
      # ram_ballast prints "ballast N GiB (want N)" once it holds the pass's untouched footprint
      if grep -Eq "ballast ([0-9]+) GiB \(want \1\)" "$MC_LOGS/k3mc-gpu$ST_AFTER/$pass.ram.log" 2>/dev/null; then
        why="gpu$ST_AFTER $pass started and its ballast settled"; break; fi
      [ $(( $(date +%s) - t_start )) -ge "$STAGGER_SETTLE_S" ] && { why="gpu$ST_AFTER $pass started ${STAGGER_SETTLE_S}s ago (ballast not settled)"; break; }
    fi
    [ -z "$(runner_pids "$ST_AFTER" | grep -vx "$$")" ] && tail -n +$((ST_N0 + 1)) "$ST_FILE" 2>/dev/null | grep -q "gpu$ST_AFTER runner \(END\|EXIT\)" && {
      why="gpu$ST_AFTER runner ended"; break; }
    sleep "$POLL_S"
  done
  if [ -n "$(runner_pids "$ST_GPU" | grep -vx "$$")" ]; then status "stagger gpu$ST_GPU: a runner already lives; not started"; exit 0; fi
  status "stagger gpu$ST_GPU: $why; exec k3_multicard.sh $ST_GPU (its own gates: d1, go, b2, ruling, lock, eqcheck verdict, RAM)"
  if [ -n "${K3D_LAUNCH:-}" ]; then exec "$K3D_LAUNCH" exec "$ST_GPU"; fi
  exec bash "$MC" "$ST_GPU" 5>&- 7>&- 8>&- 9>&-
fi

# ------------------------------------------------------------------------------------------------ --gate-once
if [ "$MODE" = gate ]; then
  if [ -s "$DECISION" ] && [ "$(jget "$DECISION" applied)" = True ]; then echo "ready (decision already applied: $DECISION)"; exit 0; fi
  why=$(gate_why)
  if [ -z "$why" ] && resolve_plan; then   # past the cheap gates: B2 pending and a live GPU0 runner are waits too
    [ "$(b2_state)" = pending ] && why="B2_not_passed_yet($B2_DIR)"
    [ -n "$(runner_pids 0)" ] && why="${why:+$why }k3_multicard_0_already_live"
  fi
  [ -z "$why" ] && { echo ready; exit 0; }
  echo "$why"; exit 75
fi

# ------------------------------------------------------------------------------------------------ --dry-run
if [ "$MODE" = dry ]; then
  say "$VERSION dry run"
  why=$(gate_why); say "gates: ${why:-ready}"
  say "decision file $DECISION: $([ -s "$DECISION" ] && echo "present, branch $(jget "$DECISION" branch) applied $(jget "$DECISION" applied)" || echo absent)"
  say "go file $GO_FILE: $([ -e "$GO_FILE" ] && tr '\n' ' ' < "$GO_FILE" || echo absent); max_passes $MAXP_FILE: $(cat "$MAXP_FILE" 2>/dev/null || echo absent)"
  if resolve_plan; then
    say "k3_multicard plan: class $PLAN_CLASS tool $PLAN_KIND $PLAN_SHA16, B2 $B2_DIR (rebase $REBASE) -> b2 $(b2_state); K3 out $K3OUT; eqcheck verdict $EQV ($([ -s "$EQV" ] && jget "$EQV" verdict || echo absent))"
    EQ_LOG=$MC_LOGS/k3mc-gpu0/eqcheck.log PLAN_SHA16=$PLAN_SHA16 PLAN_KIND=$PLAN_KIND K3OUT=$K3OUT KC=$KC K3SHA=$(sha "$K3TOOL") \
      V130_PIN=$V130_PIN GPU6_STATUS=$GPU6_STATUS GPU6_LOGS=$GPU6_LOGS P2_VERDICT=$P2_VERDICT \
      PRE_EQ=$([ -s "$EQV" ] && jget "$EQV" verdict || echo absent) PRE_D2PLAN_V130=$(grep -q k3_pass_tools "$D2_PLAN" 2>/dev/null && echo 1 || echo 0) \
      estimate > "${TMPDIR:-/tmp}/k3d-dry-$$.json"
    python3 - "${TMPDIR:-/tmp}/k3d-dry-$$.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
f, g, e = d["finish"], d["gpu6"], d["eqcheck"]
print(f"  eqcheck layers printed {e['layers_printed']} of 93 (compute {e['compute_s_total']} s); 3090 pass hours {f['pass_3090_hours']}")
print(f"  gpu6: queue {g['queue_version']} alive {g['queue_alive']}, started by it {g['passes_started_by_live_queue']}, current pass {g['current_pass']}, remaining {g['current_remaining_hours']} h, full pass {g['full_pass_hours']} h ({g['full_pass_source']})")
print(f"  D2-first {f['d2_first_finish_h']} h, K3-first {f['k3_first_finish_h']} h; blocked {d['k3_on_3090_blocked']}")
print(f"  would take: {d['branch']} ({d['reason']})")
PY
    rm -f "${TMPDIR:-/tmp}/k3d-dry-$$.json"
  else
    say "k3_multicard plan: $PLAN_WHY"
  fi
  say "DRY RUN: nothing was written"
  exit 0
fi

# ------------------------------------------------------------------------------------------------ armed
exec 5>> "$LOCK"
flock -n 5 || { say "another k3_seal_decide holds $LOCK"; exit 5; }
if [ -s "$DECISION" ] && [ "$(jget "$DECISION" applied)" = True ]; then
  say "decision already applied: branch $(jget "$DECISION" branch) ($DECISION, sha $(sha "$DECISION" | cut -c1-16)); nothing redone"
  exit 0
fi
why=$(gate_why)
if [ -n "$why" ]; then say "not ready: $why (nothing written)"; exit 75; fi
status "start $VERSION pid $$"
mkdir -p "$(dirname "$DECISION")"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/k3d.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
PRE_EQ=not-run; EQ_RC=""; EQ_NOTE=""

decide_now() {  # $1 PRE_EQ: estimate + branch into $WORK/decision.json (status line), never applies
  local pre=$1 v130=0
  grep -q k3_pass_tools "$D2_PLAN" 2>/dev/null && v130=1
  EQ_LOG=$MC_LOGS/k3mc-gpu0/eqcheck.log PLAN_SHA16=${PLAN_SHA16:-} PLAN_KIND=${PLAN_KIND:-} K3OUT=${K3OUT:-} KC=$KC \
    K3SHA=$(sha "$K3TOOL") V130_PIN=$V130_PIN GPU6_STATUS=$GPU6_STATUS GPU6_LOGS=$GPU6_LOGS P2_VERDICT=$P2_VERDICT \
    PRE_EQ=$pre PRE_D2PLAN_V130=$v130 estimate > "$WORK/estimate.json" || return 1
  python3 - "$WORK/estimate.json" "$WORK/decision.json" "$VERSION" "$(sha "$0")" "$pre" "${EQ_RC:-}" "${EQ_NOTE:-}" \
    "${EQV:-}" "${PLAN_CLASS:-}" "${PLAN_KIND:-}" "${PLAN_SHA16:-}" "${B2_DIR:-}" "${B2:-}" "$GO_FILE" "$MAXP_FILE" "${FORCE_BRANCH:-}" "${FORCE_WHY:-}" <<'PY'
import hashlib, json, os, sys, time
(est_p, out, ver, me, pre, eq_rc, eq_note, eqv, cls, kind, sha16, b2dir, b2, go, maxp, fb, fw) = sys.argv[1:18]
est = json.load(open(est_p))
def body(p):
    try:
        return open(p).read()
    except OSError:
        return None
def sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError:
        return None
branch, reason = est["branch"], est["reason"]
if fb:
    branch, reason = fb, fw
doc = {"schema": "omni-k3-seal-decision-v1", "by": ver, "script_sha256": me,
       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "rule": "K3_ACCELERATION_PLAN_2026-10-14 section 4.5 (planning engine: section 4.4 / install step 6)",
       "branch": branch, "reason": reason, "applied": False,
       "card_class_check": {"precondition": pre, "runner_rc": eq_rc or None, "note": eq_note or None,
                            "verdict_path": eqv or None, "verdict_sha256": sha(eqv) if eqv else None,
                            "card_class": cls or None, "tool_kind": kind or None, "tool_sha16": sha16 or None,
                            "eqcheck_log_sha256": sha(est["eqcheck"]["log"]) if est["eqcheck"]["log"] else None},
       "b2": {"dir": b2dir or None, "state": b2 or None},
       "before": {"go_file": body(go), "gpu6_max_passes": body(maxp)},
       "gpu6_max_passes": est.get("gpu6_max_passes") if branch in ("k3-first", "planning") else None,
       "estimate": est}
json.dump(doc, open(out, "w"), indent=1, sort_keys=True)
print(f"{branch}: {reason}")
PY
}
finish() {  # $1 actions json: write the decision with applied=true, status, heartbeat; exit 0
  python3 - "$WORK/decision.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d["applied"] = True; json.dump(d, open(sys.argv[1], "w"), indent=1, sort_keys=True)
PY
  record "$WORK/decision.json" "$1"
  local e=$WORK/decision.json
  status "decision=$(jget "$e" branch) applied=1 d2_first_h=$(jget "$e" estimate.finish.d2_first_finish_h) k3_first_h=$(jget "$e" estimate.finish.k3_first_finish_h) reason=\"$(jget "$e" reason | cut -c1-200)\" receipt=$DECISION"
  exit 0
}

# -- the plan, B2 and a manual GO
if ! resolve_plan; then
  FORCE_BRANCH=d2-first FORCE_WHY="k3_multicard plan unresolvable: $PLAN_WHY"
  decide_now "plan-unresolvable"
elif ! go_ours; then
  FORCE_BRANCH=manual FORCE_WHY="$GO_FILE was not written by k3_seal_decide ($(tr '\n' ' ' < "$GO_FILE" | cut -c1-160)): the lead decided by hand; nothing changed"
  decide_now "manual-go"
  finish '[{"action": "none", "why": "manual GO present"}]'
else
  B2=$(b2_state)
  case "$B2" in
    pending) status "deferred: B2 has not passed yet for the frozen 1.2.4 ($B2_DIR); nothing written"; exit 75 ;;
    failed)  FORCE_BRANCH=d2-first FORCE_WHY="B2 FAILED ($B2_DIR): no card can run a K3 pass; GPU0 / GPU4 are D2's"
             decide_now "b2-failed" ;;
  esac
fi

# -- step 1: the card-class check on GPU0 (only when nothing above already decided).  Always through the runner: it
# reads its own verdict (k3_gate_ok: this tool sha AND this card class) and exits 0 at once after an earlier PASS, so a
# verdict of another tool or class is never taken for this one.
if [ ! -s "$WORK/decision.json" ]; then
  EQ_STATUS=$MC_STATUS_DIR/k3mc-gpu0.status
  if [ -n "$(runner_pids 0)" ]; then status "deferred: a k3_multicard.sh 0 runner already lives ($(runner_pids 0 | tr '\n' ' ')); nothing written"; exit 75; fi
  printf 'k3-multicard-go\ngpus=0\neqcheck_only=1\nby=k3_seal_decide %s %s (plan section 4.5 step 1)\n' "$VERSION" "$(utc)" | write_atomic "$GO_FILE"
  n0=$(wc -l < "$EQ_STATUS" 2>/dev/null || echo 0)
  status "eqcheck: GO gpus=0 eqcheck_only=1 written; k3_multicard.sh 0 ($PLAN_CLASS, $PLAN_KIND $PLAN_SHA16), log $MC_LOGS/k3mc-gpu0/eqcheck.log"
  mkdir -p "$MC_LOGS/k3mc-gpu0"
  bash "$MC" 0 >> "$MC_LOGS/k3mc-gpu0/runner-eqcheck.log" 2>&1 < /dev/null 5>&- 7>&- 8>&- 9>&- &
  EQ_PID=$!; waited=0; begun=0
  while [ -d /proc/$EQ_PID ]; do
    if [ $begun = 0 ]; then
      tail -n +$((n0 + 1)) "$EQ_STATUS" 2>/dev/null | grep -Eq "gpu0 eqcheck (START|PASS|FAIL|REFUSED|HALT)" && begun=1
      if [ $begun = 0 ] && [ $waited -ge "$EQ_START_MAX_S" ]; then   # still at a gate: no GPU job has begun
        pkill -TERM -P "$EQ_PID" 2>/dev/null; kill -TERM "$EQ_PID" 2>/dev/null; wait "$EQ_PID" 2>/dev/null
        rm -f "$GO_FILE"
        status "deferred: k3_multicard.sh 0 still at its gates after ${waited}s ($(tail -1 "$EQ_STATUS" 2>/dev/null | cut -c1-200)); runner stopped before any GPU job, eqcheck GO removed"
        exit 75
      fi
    fi
    sleep "$POLL_S"; waited=$((waited + POLL_S))
  done
  wait "$EQ_PID"; EQ_RC=$?
  eq_ok=$(python3 - "$EQV" "$PLAN_SHA16" "$PLAN_CLASS" <<'PY' 2>/dev/null
import json, sys
v = json.load(open(sys.argv[1]))
print(1 if v.get("verdict") == "pass" and str(v.get("tool_sha256", "")).startswith(sys.argv[2]) and v.get("card_class") == sys.argv[3] else 0)
PY
)
  [ "$EQ_RC" = 0 ] && [ "$eq_ok" != 1 ] && EQ_RC=70   # exit 0 without a PASS verdict for this tool + class
  case "$EQ_RC" in
    0) PRE_EQ=pass; EQ_NOTE="$(tail -n +$((n0 + 1)) "$EQ_STATUS" 2>/dev/null | grep -E 'eqcheck (PASS|START)' | tail -1 | cut -c1-240)" ;;
    3) PRE_EQ=fail; EQ_NOTE="$(tail -n +$((n0 + 1)) "$EQ_STATUS" 2>/dev/null | grep -E 'eqcheck (FAIL|REFUSED)' | tail -1 | cut -c1-240)" ;;
    *) PRE_EQ="incomplete-rc$EQ_RC"; EQ_NOTE="$(tail -2 "$EQ_STATUS" 2>/dev/null | tr '\n' ' ' | cut -c1-240)" ;;
  esac
  status "eqcheck: runner exit $EQ_RC -> $PRE_EQ ($EQ_NOTE)"
  decide_now "$PRE_EQ" || { status "FAILED: the estimate did not run ($WORK/estimate.json)"; exit 1; }
fi

# -- step 3: apply (the decision is on disk first, applied=false)
record "$WORK/decision.json" "[]"
BRANCH=$(jget "$WORK/decision.json" branch)
status "decision=$BRANCH applied=0 $(jget "$WORK/decision.json" reason | cut -c1-200)"
[ -z "$(d1_active)" ] || { status "REFUSED: D1 unit is $(d1_active) again; nothing applied"; exit 2; }
ACTS=$WORK/actions.jsonl; : > "$ACTS"
act() { python3 -c 'import json,sys; print(json.dumps(dict(zip(sys.argv[1::2], sys.argv[2::2]))))' "$@" >> "$ACTS"; }
case "$BRANCH" in
  k3-first|planning)
    MAXP=$(jget "$WORK/decision.json" gpu6_max_passes)
    b=$(backup "$MAXP_FILE" pre-k3decide)
    echo "$MAXP" | write_atomic "$MAXP_FILE"; act action write path "$MAXP_FILE" value "$MAXP" backup "${b:-none}"
    gpus=0; [ "$BRANCH" = k3-first ] && gpus=0,4
    b=$(backup "$GO_FILE" eqcheck)
    printf 'k3-multicard-go\ngpus=%s\nby=k3_seal_decide %s %s (%s; %s)\n' "$gpus" "$VERSION" "$(utc)" "$BRANCH" "$DECISION" | write_atomic "$GO_FILE"
    act action write path "$GO_FILE" gpus "$gpus" backup "${b:-none}"
    S0=$MC_STATUS_DIR/k3mc-gpu0.status; n0=$(wc -l < "$S0" 2>/dev/null || echo 0)
    if [ -n "$(runner_pids 0)" ]; then act action skip what "runner gpu0" why "already live: $(runner_pids 0 | tr '\n' ' ')"
    else
      p=$(launch runner 0); act action start what "k3_multicard.sh 0" pid "$p" log "$MC_LOGS/k3mc-gpu0/runner.log"
      # hold the post_d1 queue until runner 0 owns gpu0.lock, so its next GPU0 job (c1 / c2) never takes the card first
      w=0; got=""
      while [ $w -lt "${K3D_LOCK_WAIT_S:-900}" ]; do
        got=$(tail -n +$((n0 + 1)) "$S0" 2>/dev/null | grep -E "gpu0 (gate OK lock|runner END|runner EXIT|eqcheck (REFUSED|FAIL|HALT))" | head -1)
        [ -n "$got" ] && break
        sleep 5; w=$((w + 5))
      done
      act action wait what "runner gpu0 takes gpu0.lock" result "${got:-not seen after ${w}s (the runner keeps waiting on its own)}"
    fi
    if [ "$BRANCH" = k3-first ]; then
      if [ -n "$(runner_pids 4)" ]; then act action skip what "runner gpu4" why "already live: $(runner_pids 4 | tr '\n' ' ')"
      else p=$(launch stagger 4 0 "$n0"); act action start what "k3_seal_decide.sh --stagger 4 0 (then k3_multicard.sh 4)" pid "$p" log "$MC_LOGS/k3mc-gpu4/stagger.log"; fi
    fi ;;
  d2-first)
    if [ -e "$GO_FILE" ]; then
      if go_ours; then b=$(backup "$GO_FILE" d2first); rm -f "$GO_FILE"; act action remove path "$GO_FILE" backup "${b:-none}"
      else act action keep path "$GO_FILE" why "not written by k3_seal_decide"; fi
    else act action none path "$GO_FILE" why absent; fi
    m=$(tr -dc 0-9 < "$MAXP_FILE" 2>/dev/null)
    if [ -n "$m" ] && [ "$m" -lt 4 ]; then
      b=$(backup "$MAXP_FILE" d2first); rm -f "$MAXP_FILE"; act action remove path "$MAXP_FILE" value "$m" backup "${b:-none}" why "GPU6 runs every remaining pass under D2-first"
    fi ;;
  manual) act action none why "manual" ;;
esac
finish "$(python3 -c 'import json,sys; print(json.dumps([json.loads(l) for l in open(sys.argv[1]) if l.strip()]))' "$ACTS")"
