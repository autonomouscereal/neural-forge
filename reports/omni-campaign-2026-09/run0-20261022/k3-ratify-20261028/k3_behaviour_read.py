#!/usr/bin/env python3
"""k3_behaviour_read 1.0.0: decode what the provisional bf16 K3 cache predicts, for a lead read.

Read-only. Part A: sample documents from the four provisional passes (stratified by route),
decode the real text and K3's teacher-forced top-1 choice at every position, list the
disagreements with K3's top-3. Part B: on the B2 review rows, every position where the GPU6
bf16 review pass and the fp32 reference pass disagree on top-1, with both top-3 lists.
"""
import json
import random
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tiktoken.load
from safetensors.numpy import load_file

K = Path("/mnt/lexar/omni-campaign/teacher_cache/k3/eb35851b2496a628-provisional-bf16")
CORPUS = Path("/home/cereal/omni-campaign/distill/corpus/k3")
B2 = Path("/mnt/lexar/omni-campaign/distill/parity/k3cache-2026-10-14-b2-gpu6")
REVIEW = Path("/home/cereal/omni-campaign/composite/corpus")
K3DIR = Path("/mnt/lexar/models/moonshotai/Kimi-K3")
OUT = Path(sys.argv[1])
SEED = 20261028

ranks = tiktoken.load.load_tiktoken_bpe(str(K3DIR / "tiktoken.model"))
PIECE = {r: b for b, r in ranks.items()}
cfg = json.loads((K3DIR / "tokenizer_config.json").read_text())
for k, v in cfg.get("added_tokens_decoder", {}).items():
    PIECE[int(k)] = v["content"].encode()


def piece(i):
    return PIECE.get(int(i), f"<?{int(i)}>".encode())


def text(ids):
    return b"".join(piece(i) for i in ids).decode("utf-8", "replace")


def tok(i):
    return repr(piece(i).decode("utf-8", "replace"))


def bf16(a):
    # safetensors numpy has no bf16: shards load as uint16 views when read raw
    return (a.astype(np.uint32) << 16).view(np.float32)


def load_shard(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n))
        base = 8 + n
        out = {}
        for name, meta in head.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            fh.seek(base + a)
            raw = fh.read(b - a)
            dt = {"F32": np.float32, "I32": np.int32, "I64": np.int64, "BF16": np.uint16,
                  "U8": np.uint8, "I8": np.int8, "BOOL": np.bool_}[meta["dtype"]]
            arr = np.frombuffer(raw, dtype=dt).reshape(meta["shape"])
            out[name] = bf16(arr) if meta["dtype"] == "BF16" else arr
        return out


def row_index_map(pass_dir):
    """row -> (shard tensors path, start offset, length)."""
    m = {}
    for shard in sorted((pass_dir / "shards").glob("group-*.safetensors")):
        t = load_shard(shard)
        off = 0
        for ri, rl in zip(t["row_index"], t["real_lengths"]):
            m[int(ri)] = (shard, off, int(rl))
            off += int(rl)
    return m


SHARD_CACHE = {}


def shard(path):
    if path not in SHARD_CACHE:
        if len(SHARD_CACHE) > 6:
            SHARD_CACHE.clear()
        SHARD_CACHE[path] = load_shard(path)
    return SHARD_CACHE[path]


def part_a(lines, summary):
    probe = json.loads((K / "route_probe_k3.json").read_text())
    docs = probe["docs"]
    by_route = defaultdict(list)
    for d in docs:
        if d["k3_tokens"] >= 24:
            by_route[d["route"]].append(d)
    rng = random.Random(SEED)
    picks = []
    for route in sorted(by_route):
        pool = by_route[route]
        n = 3 if len(pool) >= 40 else (2 if len(pool) >= 4 else 1)
        picks += rng.sample(pool, min(n, len(pool)))
    ids_by_pass, rows_by_pass = {}, {}
    for p in range(4):
        ids_by_pass[p] = load_file(str(CORPUS / f"pass-{p}" / "k3_input_ids.safetensors"))["input_ids"]
        rows_by_pass[p] = row_index_map(K / f"pass-{p}-of-4")
    lines.append(f"# PART A: {len(picks)} cached documents, stratified over {len(by_route)} routes (seed {SEED})\n")
    agg = defaultdict(lambda: [0, 0, 0])
    for d in picks:
        p, r = d["pass"], d["row"]
        a, b = d["k3_span"]
        ids = ids_by_pass[p][r]
        sp, off, rl = rows_by_pass[p][r]
        t = shard(sp)
        tk_ids = t["topk_ids"][off:off + rl]
        tk_lp = t["topk_logprobs"][off:off + rl]
        span = ids[a:b + 1].tolist()
        hits, n, miss, conf_wrong = 0, 0, [], 0
        for pos in range(a, b):
            actual = int(ids[pos + 1])
            top1 = int(tk_ids[pos, 0])
            n += 1
            if top1 == actual:
                hits += 1
            else:
                p1 = float(np.exp(tk_lp[pos, 0]))
                conf_wrong += p1 > 0.9
                rank = [int(x) for x in tk_ids[pos]].index(actual) if actual in tk_ids[pos] else -1
                ctx = text(ids[max(a, pos - 14):pos + 1].tolist())
                alts = ", ".join(f"{tok(tk_ids[pos, j])} {np.exp(tk_lp[pos, j]):.2f}" for j in range(3))
                miss.append(f"    ...{ctx[-70:]!r} -> real {tok(actual)} (K3 rank {rank}) | K3: {alts}")
        acc = hits / max(n, 1)
        agg[d["route"]][0] += hits
        agg[d["route"]][1] += n
        agg[d["route"]][2] += conf_wrong
        greedy = text([int(tk_ids[pos, 0]) for pos in range(a, min(b, a + 160))])
        lines.append(f"## pass {p} row {r} doc {d['doc']} route={d['route']} tokens={d['k3_tokens']} bpb={d['k3_bpb']:.3f} top1={acc:.3f} confident_wrong(p>0.9)={conf_wrong}")
        lines.append("REAL TEXT (first 1500 chars):")
        lines.append(text(span)[:1500])
        lines.append("K3 TOP-1 STREAM (token-by-token prediction under the real prefix, first 160 tokens):")
        lines.append(greedy)
        lines.append(f"DISAGREEMENTS ({len(miss)}; first 25):")
        lines += miss[:25]
        lines.append("")
        summary["docs"].append({"pass": p, "row": r, "doc": d["doc"], "route": d["route"],
                                "tokens": d["k3_tokens"], "bpb": d["k3_bpb"], "top1": acc,
                                "confident_wrong": conf_wrong})
    summary["route_top1"] = {k: {"top1": v[0] / max(v[1], 1), "positions": v[1], "confident_wrong": v[2]}
                             for k, v in agg.items()}


def part_b(lines, summary):
    lines.append("\n# PART B: B2 review rows, GPU6 bf16 review pass vs fp32 reference pass, every top-1 disagreement\n")
    tot, dis, near = 0, 0, 0
    for name in ("k3_review_input_ids", "k3_review_native_input_ids"):
        ids = load_file(str(REVIEW / f"{name}.safetensors"))["input_ids"]
        b16, f32 = {}, {}
        for src, dst in ((B2 / "review-pass", b16), (B2 / "review-pass-fp32", f32)):
            for sh in sorted((src / "shards").glob("group-*.safetensors")):
                t = load_shard(sh)
                meta_seg = None
                with open(sh, "rb") as fh:
                    n = struct.unpack("<Q", fh.read(8))[0]
                    meta_seg = json.loads(fh.read(n))["__metadata__"]["segment"]
                if meta_seg != name:
                    continue
                off = 0
                for ri, rl in zip(t["row_index"], t["real_lengths"]):
                    dst[int(ri)] = (t["topk_ids"][off:off + rl], t["topk_logprobs"][off:off + rl])
                    off += int(rl)
        for r in sorted(b16):
            bi, bl = b16[r]
            fi, fl = f32[r]
            for pos in range(len(bi) - 1):
                tot += 1
                if int(bi[pos, 0]) == int(fi[pos, 0]):
                    continue
                dis += 1
                gap = abs(float(fl[pos, 0]) - float(fl[pos, 1]))
                near += gap < 0.5
                actual = int(ids[r, pos + 1])
                ctx = text(ids[r, max(0, pos - 16):pos + 1].tolist())
                fa = ", ".join(f"{tok(fi[pos, j])} {np.exp(fl[pos, j]):.2f}" for j in range(3))
                ba = ", ".join(f"{tok(bi[pos, j])} {np.exp(bl[pos, j]):.2f}" for j in range(3))
                lines.append(f"[{name} r{r} p{pos}] ...{ctx[-70:]!r} real={tok(actual)}\n    fp32: {fa}\n    bf16: {ba}   (fp32 top1-top2 gap {gap:.2f} nats)")
    lines.append(f"\nPART B totals: positions {tot}, top-1 disagreements {dis} ({dis / max(tot, 1):.4f}), of which fp32 top-1/top-2 gap < 0.5 nats: {near}")
    summary["part_b"] = {"positions": tot, "disagreements": dis, "near_ties_lt_0p5_nats": near}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines, summary = [], {"tool": "k3_behaviour_read 1.0.0", "seed": SEED, "docs": []}
    part_a(lines, summary)
    part_b(lines, summary)
    (OUT / "k3_behaviour_read.txt").write_text("\n".join(lines))
    (OUT / "k3_behaviour_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "docs"}, indent=1))


if __name__ == "__main__":
    main()
