#!/usr/bin/env python3
"""k3_schedule_projection.py -- the arithmetic behind K3_ACCELERATION_PLAN_2026-10-14.md (lane k3plan, 2026-10-15).

Every input is a printed measurement (source in INPUTS) or a labelled assumption; every output is EST.  No GPU, no
server access: python k3_schedule_projection.py [--json]."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta

H = 3600.0
POS = 122 * 4096          # positions per pass file (k3_reserve: rows x T; trim can only lower it)
KDA, MLA = 69, 24          # Kimi K3: 93 layers, MLA at L03, L07, ..., L91 and L92
INPUTS = {
    "gpu6_124_throttled": {"kda": (303, 411), "mla": (474, 801), "read_s": (75, 190), "ckpt_h": 0.4,
                           "src": "b2-identity_pass.log, production CachePass.layer at 143,360 positions: KDA L01 400, L02 374, "
                                  "L12 411, L24 303; MLA L03 539 .. L27 491, range 474-801 pos/s; 1.2.4 reads serial 75-190 s"},
    "v130_layer_speedup": {"range": (1.15, 1.46), "src": "lane k3accel P40: production 8-group layer 1.15-1.25x, steady L85 1.46x"},
    "p40_v130": {"kda": (946, 1048), "mla": (935, 1816),
                 "src": "w1/w4 L84-85 63,952 positions 946-1,048 pos/s; w2 L83 MLA T=2048 1,816, b2geo 1,760; "
                        "935 = k3accel's MLA floor for T=4096 (unmeasured)"},
    "gpu6_124_cool": {"kda": (750, 1068), "mla": (853, 853), "src": "b2_pass.log L01 KDA 1,068 pos/s at 1,605 MHz 254 W "
                      "(one cool layer, 1 block); L03 MLA 853 at 1,035 MHz; 750 = assumed floor for late layers"},
    "d2": {"upside_tok_s": 348.0, "planning_tok_s": 151.24, "overhead": 1.3, "short_tokens": 24.9e6, "k3_from": 20e6,
           "long_up_h": 1.0e6 / 86 / H * 1.3, "long_plan_h": 1.0e6 / 75.6 / H * 1.3, "seal_h": 6.4,
           "src": "DISTILL_DESIGN 4 (D2 row); A2 draft item 3 (~24.9M short) and item 8 (K3 rows after 20M)"},
}


def span(lo_hi_rate, layers):
    lo, hi = lo_hi_rate
    return layers * POS / hi / H, layers * POS / lo / H


def per_pass():
    g = INPUTS["gpu6_124_throttled"]
    k1, k2 = span(g["kda"], KDA)
    m1, m2 = span(g["mla"], MLA)
    r1, r2 = 93 * g["read_s"][0] / H, 93 * g["read_s"][1] / H
    s124 = (k1 + m1 + r1 + g["ckpt_h"], k2 + m2 + r2 + g["ckpt_h"])
    sp = INPUTS["v130_layer_speedup"]["range"]
    s130 = ((k1 + m1) / sp[1], (k2 + m2) / sp[0])
    p = INPUTS["p40_v130"]
    pk = span(p["kda"], KDA); pm = span(p["mla"], MLA)
    cool130 = (pk[0] + pm[0] + 0.3, pk[1] + pm[1] + 0.3)
    c = INPUTS["gpu6_124_cool"]
    ck = span(c["kda"], KDA); cm = span(c["mla"], MLA)
    cool124 = (ck[0] + cm[0] + r1 + g["ckpt_h"], ck[1] + cm[1] + r2 + g["ckpt_h"])
    return {"gpu6_124_throttled": s124, "gpu6_130_throttled": s130, "gpu6_130_cool": cool130,
            "gpu6_124_cool": cool124, "rtx3090_130": (4.2, cool130[1]),
            "parts_124_throttled_h": {"kda": (k1, k2), "mla": (m1, m2), "reads": (r1, r2), "ckpt": g["ckpt_h"]}}


T0 = datetime(2026, 10, 15, 1, 30)              # this projection
B2 = (datetime(2026, 10, 15, 16, 30), datetime(2026, 10, 15, 18, 0))   # identity ~12:30Z + fp32 3.4-5 h + parity 0.5 h
GATE_H = (1.0, 1.5)                               # k3-v130-gate on GPU6 (windows w1, w2p, w4, NEG)
D1REG = (datetime(2026, 10, 16, 1, 20), datetime(2026, 10, 16, 10, 20))  # 45 layers x 28-40 min + 3.1 h seal
P2_H = 1.0
EQ_H = (1.5, 2.5)                                 # eqcheck on a 3090: 93 layers of 17 GB reads at ~250 MB/s + compute


def d2_tail(engine):
    d = INPUTS["d2"]
    rate = d[f"{engine}_tok_s"]
    to20 = d["k3_from"] / rate / H * d["overhead"]
    after = (d["short_tokens"] - d["k3_from"]) / rate / H * d["overhead"]
    long_h = d["long_up_h"] if engine == "upside" else d["long_plan_h"]
    return to20, after, long_h, d["seal_h"]


def fmt(t):
    return t.strftime("%m-%d %H:%MZ")


def schedule():
    pp = per_pass()
    out = {"per_pass_h": {k: [round(v[0], 1), round(v[1], 1)] for k, v in pp.items() if k != "parts_124_throttled_h"},
           "parts_124_throttled_h": {k: ([round(x, 1) for x in v] if isinstance(v, tuple) else v)
                                      for k, v in pp["parts_124_throttled_h"].items()}}
    to20, after, long_h, seal = d2_tail("upside")
    pto20, pafter, plong, _ = d2_tail("planning")
    out["d2_h"] = {"upside_to_20M": round(to20, 2), "upside_after_20M": round(after, 2), "upside_long": round(long_h, 2),
                   "planning_to_20M": round(pto20, 2), "planning_after_20M": round(pafter, 2), "planning_long": round(plong, 2),
                   "seal": seal}
    d2start = (D1REG[0] + timedelta(hours=P2_H), D1REG[1] + timedelta(hours=P2_H))
    out["d1_registered"] = [fmt(D1REG[0]), fmt(D1REG[1])]
    out["d2_start"] = [fmt(d2start[0]), fmt(d2start[1])]
    out["d2_first_k3_row_upside"] = [fmt(d2start[0] + timedelta(hours=to20)), fmt(d2start[1] + timedelta(hours=to20))]
    out["d2_first_k3_row_planning"] = [fmt(d2start[0] + timedelta(hours=pto20)), fmt(d2start[1] + timedelta(hours=pto20))]
    tail = after + long_h + seal
    opts = {}
    # O1 / O2 / O3: GPU6 only, 4 passes back to back; D2 (upside) on GPU0+GPU4 meanwhile, waiting at 20M
    for name, key, gate in (("O1 1.2.4, GPU6 only, throttled", "gpu6_124_throttled", (0, 0)),
                            ("O2 1.3.0, GPU6 only, throttled", "gpu6_130_throttled", GATE_H),
                            ("O3 1.3.0, GPU6 only, cooling fixed", "gpu6_130_cool", GATE_H)):
        lo = B2[0] + timedelta(hours=gate[0] + 4 * pp[key][0])
        hi = B2[1] + timedelta(hours=gate[1] + 4 * pp[key][1])
        d2a = max(lo, d2start[0] + timedelta(hours=to20)) + timedelta(hours=tail)
        d2b = max(hi, d2start[1] + timedelta(hours=to20)) + timedelta(hours=tail)
        opts[name] = {"k3_sealed": [fmt(lo), fmt(hi)], "d2_done_upside": [fmt(d2a), fmt(d2b)],
                      "arith": f"B2 {fmt(B2[0])}..{fmt(B2[1])} + gate {gate[0]}-{gate[1]} h + 4 x {pp[key][0]:.1f}-{pp[key][1]:.1f} h"}
    # O4: K3-first on the 3090s after D1 (P2 first, then the eqcheck), GPU6 capped at pass 0, <= 2 passes at once (RAM);
    # D2 starts after the seal (+ the P2 hour already spent)
    g6 = pp["gpu6_130_throttled"]; r3090 = pp["rtx3090_130"]
    res = []
    for i in (0, 1):
        g6_end = B2[i] + timedelta(hours=GATE_H[i] + g6[i])
        g0_start = D1REG[i] + timedelta(hours=P2_H + EQ_H[i])
        g0_p3 = g0_start + timedelta(hours=r3090[i])
        g0_p2 = g0_p3 + timedelta(hours=r3090[i])
        g4_p1 = max(g6_end, g0_start) + timedelta(hours=r3090[i])   # second 3090 starts once GPU6's pass frees the RAM
        sealed = max(g6_end, g0_p2, g4_p1)
        done = sealed + timedelta(hours=to20 + after + long_h + seal)
        res.append((sealed, done, g6_end, g0_p3, g0_p2, g4_p1))
    opts["O4 1.3.0 GPU6 pass 0 + 3090s K3-first (D2 after the seal)"] = {
        "k3_sealed": [fmt(res[0][0]), fmt(res[1][0])], "d2_done_upside": [fmt(res[0][1]), fmt(res[1][1])],
        "arith": (f"GPU6 pass 0 ends {fmt(res[0][2])}..{fmt(res[1][2])}; GPU0 passes 3, 2 end {fmt(res[0][4])}..{fmt(res[1][4])}; "
                  f"GPU4 pass 1 ends {fmt(res[0][5])}..{fmt(res[1][5])}; 3090 pass {r3090[0]:.1f}-{r3090[1]:.1f} h (unmeasured)")}
    out["options"] = opts
    return out


if __name__ == "__main__":
    o = schedule()
    if "--json" in sys.argv:
        print(json.dumps(o, indent=1))
    else:
        print("per pass (h):", json.dumps(o["per_pass_h"]))
        print("1.2.4 throttled parts (h):", json.dumps(o["parts_124_throttled_h"]))
        print("D2 (h):", json.dumps(o["d2_h"]))
        print("D1 registered", o["d1_registered"], "D2 start", o["d2_start"])
        print("D2 first K3 row: upside", o["d2_first_k3_row_upside"], "planning", o["d2_first_k3_row_planning"])
        for k, v in o["options"].items():
            print(f"{k}: K3 sealed {v['k3_sealed']}, D2 done (upside) {v['d2_done_upside']}  [{v['arith']}]")
