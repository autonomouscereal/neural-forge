#!/usr/bin/env python3
"""F1: why the BETTER body composes WORSE, measured two ways.

The fact this file exists to explain, from waves 30-32's sealed rows:

| | body alone | K=16,384 | 12,288 | 9,216 | 8,192 |
|---|---:|---:|---:|---:|---:|
| `ARCH.CL1792`  (stream-selected) | 7 | **8** | **7** | **6** | **5** |
| `ARCH.CL1792M` (reader-metric)   | **8** | 3 | 4 | 2 | 3 |

Both bodies are `intermediate_size` 1792 at byte-identical storage 0.856199x.
The M body reads ONE BETTER alone and **three and a half cases worse on average
under a vocabulary gather that touches no weight either body owns**.  Wave 32
measured all four rungs and could not say why, and the answer decides whether
any future body improvement can be expected to survive composition at all.

Two measurements, and they answer different halves.

**`--lanes` is free and says WHICH capability dies.**  Every sealed cell carries
a `per-case.json` with a verdict per lane, so the death is readable per lane
rather than per count: for each family, which lanes the body holds, which the
composed child holds, and -- the discriminating quantity -- which lanes the M
family loses that the plain family keeps at the same K and the same storage.  A
count that falls by five can fall in fifteen different ways and only some of
them have a mechanism.

**`--channels` costs one parent forward pass per id stream and tests the
candidate.**  The `intermediate_size` fold is a channel SELECTION
(`mx34_blend_body.py`): `w1`/`w3` rows are verbatim parent rows in the pivot's
order.  The two bodies share 23,015 of 25,088 selected channels and each keeps
2,073 the other does not.  The vocabulary gather changes no weight -- it changes
the SEGMENTATION, so every kept channel sees a different input distribution.
The hypothesis this tests is therefore precise and falsifiable:

> the M body's uniquely-kept channels are the ones whose input statistics move
> most under retokenization, so the M selection is the one the vocabulary gather
> invalidates.

Measured as: run the PARENT over the sealed corpus twice through
`mx34_retok.retokenize` -- once on `parent_ids_original` (the parent's own
segmentation) and once on `parent_ids` (the child's segmentation, injected back
into the parent's id space, which is the same text cut differently) -- capture
the SwiGLU intermediate at every feed-forward site, and take each parent
channel's mean square under each stream.  A channel's **retokenization
sensitivity** is the relative move.  Then compare that sensitivity across three
disjoint groups: channels both bodies keep, channels only `CL1792` keeps, and
channels only `CL1792M` keeps.  If the M-unique group is not more sensitive, the
hypothesis is dead and the file says so.

Nothing here grades anything.  Usage::

    mx34_mforensics.py --lanes --matrix /mnt/ai_data/neural-forge-matrix --out <json>
    mx34_mforensics.py --channels --parent <dir> --retok-child <dir> \\
        --body plain=<dir> --body metric=<dir> --device cuda:0 --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: The two families, body first, then the composed rungs in descending K.  Each
#: entry is (label, wave) so the cell receipt can be found without a glob that
#: might match a rebuild in another wave's tree.
FAMILIES: dict[str, list[tuple[str, str]]] = {
    "plain": [
        ("CL1792", "wave30"),
        ("V16CL1792", "wave31"),
        ("V12CL1792", "wave31"),
        ("V9CL1792", "wave31"),
        ("V8CL1792", "wave31"),
    ],
    "metric": [
        ("CL1792M", "wave30"),
        ("V16CL1792M", "wave32"),
        ("V12CL1792M", "wave32"),
        ("V9CL1792M", "wave32"),
        ("V8CL1792M", "wave32"),
    ],
}
HELD = ("Useful", "Partial")


def read_cases(matrix: Path, label: str, wave: str) -> dict[str, str]:
    path = matrix / wave / "cells" / f"ARCH.{label}" / "per-case.json"
    if not path.is_file():
        raise SystemExit(f"REFUSED: no per-case verdicts at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def lanes_report(matrix: Path) -> dict[str, Any]:
    families: dict[str, Any] = {}
    cases: dict[str, dict[str, dict[str, str]]] = {}
    for family, members in FAMILIES.items():
        cases[family] = {label: read_cases(matrix, label, wave) for label, wave in members}
    lanes = sorted(next(iter(cases["plain"].values())))

    for family, members in FAMILIES.items():
        body_label = members[0][0]
        body = cases[family][body_label]
        rows = []
        for label, _ in members:
            verdicts = cases[family][label]
            rows.append({
                "label": label,
                "count": sum(1 for v in verdicts.values() if v in HELD),
                "held": sorted(l for l in lanes if verdicts.get(l) in HELD),
            })
        body_held = set(rows[0]["held"])
        for row in rows[1:]:
            row["lost_from_body"] = sorted(body_held - set(row["held"]))
            row["gained_over_body"] = sorted(set(row["held"]) - body_held)
        families[family] = {"body": body_label, "rungs": rows}

    # The discriminating comparison: at each K, which lanes does the M family
    # hold and the plain family not, and the other way round.
    per_k = []
    for i in range(1, len(FAMILIES["plain"])):
        p_label = FAMILIES["plain"][i][0]
        m_label = FAMILIES["metric"][i][0]
        p_held = {l for l in lanes if cases["plain"][p_label].get(l) in HELD}
        m_held = {l for l in lanes if cases["metric"][m_label].get(l) in HELD}
        per_k.append({
            "plain": p_label, "metric": m_label,
            "plain_count": len(p_held), "metric_count": len(m_held),
            "delta": len(m_held) - len(p_held),
            "plain_only": sorted(p_held - m_held),
            "metric_only": sorted(m_held - p_held),
            "both": sorted(p_held & m_held),
        })

    # Which lanes account for the M family's loss, summed over the four rungs.
    charge: dict[str, int] = {l: 0 for l in lanes}
    for row in per_k:
        for lane in row["plain_only"]:
            charge[lane] += 1
        for lane in row["metric_only"]:
            charge[lane] -= 1
    return {
        "lanes": lanes,
        "families": families,
        "per_k": per_k,
        "lane_charge_plain_minus_metric": dict(
            sorted(charge.items(), key=lambda kv: -kv[1])
        ),
        "note": (
            "A positive charge means the plain family holds that lane at more of the "
            "four composed rungs than the M family does.  The sum of the charges is "
            "the total case difference the four rungs carry."
        ),
    }


# ------------------------------------------------------------------- channels
def channels_report(parent_dir: Path, retok_child: Path, bodies: dict[str, Path],
                    device: str) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file
    from transformers import Lfm2ForCausalLM

    import mx34_blend_body as BB
    import mx34_retok

    rows, meta = mx34_retok.retokenize(retok_child)
    print(f"[mx34forensic] retok: {len(rows)} batches, "
          f"{meta['sequence_length_ratio']:.4f}x the parent's positions")

    parent_tensors = load_file(str(parent_dir / "model.safetensors"))
    sites = BB._ff_sites(parent_tensors)
    print(f"[mx34forensic] {len(sites)} feed-forward sites")

    selections: dict[str, dict[str, list[int]]] = {}
    for name, body in bodies.items():
        child_tensors = load_file(str(body / "model.safetensors"))
        per_site = {}
        for prefix in sites:
            index = BB.parent_channel_index(parent_tensors, prefix)
            per_site[prefix] = BB.recover_selection(child_tensors, index, prefix, name)
        selections[name] = per_site
        total = sum(len(v) for v in per_site.values())
        print(f"[mx34forensic] {name}: recovered {total} channels over {len(sites)} sites")

    model = (Lfm2ForCausalLM.from_pretrained(str(parent_dir), dtype=torch.float32)
             .to(device).eval())
    # The SwiGLU intermediate is exactly the input to w2, so one pre-hook per site
    # captures the per-channel activation without re-deriving silu(w1 x) * (w3 x).
    modules = {}
    for prefix in sites:
        layer = int(prefix.split(".")[2])
        modules[prefix] = model.model.layers[layer].feed_forward.w2

    @torch.no_grad()
    def energies(key: str) -> dict[str, Any]:
        state: dict[str, torch.Tensor] = {}
        handles = [
            module.register_forward_pre_hook(
                (lambda n: lambda mod, args, kwargs: state.__setitem__(
                    n, (args[0] if args else kwargs["input"]).detach()))(name),
                with_kwargs=True)
            for name, module in modules.items()
        ]
        total = {name: None for name in modules}
        positions = 0
        try:
            for row in rows:
                ids = torch.tensor([row[key]], dtype=torch.long, device=device)
                model(input_ids=ids, use_cache=False)
                positions += len(row[key])
                for name in modules:
                    act = state[name].reshape(-1, state[name].shape[-1]).to(torch.float64)
                    block = (act * act).sum(dim=0).cpu()
                    total[name] = block if total[name] is None else total[name] + block
                state.clear()
        finally:
            for handle in handles:
                handle.remove()
        return {"sums": total, "positions": positions}

    print("[mx34forensic] parent forward on its OWN segmentation")
    base = energies("parent_ids_original")
    print("[mx34forensic] parent forward on the CHILD's segmentation, injected")
    shifted = energies("parent_ids")

    groups: dict[str, list[float]] = {"shared": [], "plain_only": [], "metric_only": []}
    per_site_rows = []
    names = list(bodies)
    for prefix in sites:
        a = set(selections[names[0]][prefix])
        b = set(selections[names[1]][prefix])
        e0 = base["sums"][prefix] / base["positions"]
        e1 = shifted["sums"][prefix] / shifted["positions"]
        scale = float(e0.mean())
        sens = ((e1 - e0).abs() / (e0 + 1e-12 + 0.01 * scale)).tolist()
        site_groups = {
            "shared": [sens[c] for c in sorted(a & b)],
            "plain_only": [sens[c] for c in sorted(a - b)],
            "metric_only": [sens[c] for c in sorted(b - a)],
        }
        for key, values in site_groups.items():
            groups[key].extend(values)
        per_site_rows.append({
            "site": prefix,
            "kept_plain": len(a), "kept_metric": len(b),
            "shared": len(a & b), "plain_only": len(a - b), "metric_only": len(b - a),
            "median_sensitivity": {
                k: (round(sorted(v)[len(v) // 2], 6) if v else None)
                for k, v in site_groups.items()
            },
        })

    def stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        ordered = sorted(values)
        return {
            "n": len(ordered),
            "median": round(ordered[len(ordered) // 2], 6),
            "mean": round(sum(ordered) / len(ordered), 6),
            "p90": round(ordered[int(0.90 * (len(ordered) - 1))], 6),
        }

    # ------------------------------------------------------------------
    # THE SECOND CANDIDATE, and the one the receipts point at.  `ARCH.CL1792M`'s
    # own axis records its selection as a pivoted-Cholesky coreset on
    # READER-METRIC-RESCALED channels, `s_j = sqrt(w2[:,j]^T M w2[:,j])`, where M
    # is built from the PARENT's 65,536-row head.  So the M selection ranks a
    # channel by its effect on the OUTPUT LOGITS and the plain selection ranks it
    # in the residual stream.  Two quantities decide between the readings, and
    # neither needs M itself:
    #
    #   a_j = w2[:,j]^T (H^T H) w2[:,j]   -- the channel's logit leverage through
    #                                        the FULL head;
    #   b_j = w2[:,j]^T (H_K^T H_K) w2[:,j] -- the same through the KEPT rows only;
    #   s_j = ||w2[:,j]||^2                -- what it writes into the stream.
    #
    # `a_j / s_j` says whether the M selection really did prefer logit-heavy,
    # stream-light channels.  `b_j / a_j` says whether the vocabulary restriction
    # moves those channels' leverage more than the others'.  Both are exact in
    # closed form off a 1024x1024 Gram, so the 65,536 x 2,560 product is never
    # materialised.
    import mx34_retok as _retok
    keep = _retok.injection(retok_child)
    head = None
    for key in ("lm_head.weight", "model.embed_tokens.weight"):
        if key in parent_tensors:
            head = parent_tensors[key].to(torch.float64)
            break
    leverage: dict[str, Any] = {"available": head is not None}
    if head is not None:
        G_full = head.T @ head
        rows = torch.tensor(sorted(keep), dtype=torch.long)
        sub = head[rows]
        G_keep = sub.T @ sub
        buckets: dict[str, dict[str, list[float]]] = {
            g: {"logit_per_stream": [], "restriction_ratio": []}
            for g in ("shared", "plain_only", "metric_only")
        }
        for prefix in sites:
            a_sel = set(selections[names[0]][prefix])
            b_sel = set(selections[names[1]][prefix])
            w2 = parent_tensors[prefix + "feed_forward.w2.weight"].to(torch.float64)
            a = ((G_full @ w2) * w2).sum(dim=0)
            b = ((G_keep @ w2) * w2).sum(dim=0)
            s = (w2 * w2).sum(dim=0)
            per_stream = (a / (s + 1e-30)).tolist()
            ratio = (b / (a + 1e-30)).tolist()
            for name, members in (("shared", a_sel & b_sel),
                                  ("plain_only", a_sel - b_sel),
                                  ("metric_only", b_sel - a_sel)):
                for c in sorted(members):
                    buckets[name]["logit_per_stream"].append(per_stream[c])
                    buckets[name]["restriction_ratio"].append(ratio[c])
        leverage["head_rows"] = int(head.shape[0])
        leverage["kept_rows"] = len(keep)
        leverage["groups"] = {
            g: {k: stats(v) for k, v in d.items()} for g, d in buckets.items()
        }

    summary = {k: stats(v) for k, v in groups.items()}
    plain_med = summary["plain_only"].get("median")
    metric_med = summary["metric_only"].get("median")
    verdict = "not decidable: a group is empty"
    if plain_med is not None and metric_med is not None:
        ratio = metric_med / plain_med if plain_med else None
        verdict = (
            f"M-unique channels are {ratio:.3f}x as retokenization-sensitive as "
            f"plain-unique channels (median).  "
            + ("HYPOTHESIS SUPPORTED: the M selection is the one the gather "
               "invalidates" if ratio and ratio >= 1.10 else
               "HYPOTHESIS FALSIFIED: the M selection is NOT more exposed to "
               "retokenization, so the composition penalty is not a property of "
               "which channels were chosen" if ratio and ratio <= 0.91 else
               "HYPOTHESIS NOT SUPPORTED: the two groups are within 10 per cent of "
               "one another, which is not a mechanism")
        )
    return {
        "retok": {
            "child": str(retok_child),
            "batches": len(rows),
            "sequence_length_ratio": meta["sequence_length_ratio"],
        },
        "bodies": {k: str(v) for k, v in bodies.items()},
        "sites": len(sites),
        "positions": {"parent_segmentation": base["positions"],
                      "child_segmentation": shifted["positions"]},
        "sensitivity_by_group": summary,
        "logit_leverage": leverage,
        "per_site": per_site_rows,
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", action="store_true")
    ap.add_argument("--channels", action="store_true")
    ap.add_argument("--matrix", type=Path, default=Path("/mnt/ai_data/neural-forge-matrix"))
    ap.add_argument("--parent", type=Path)
    ap.add_argument("--retok-child", type=Path)
    ap.add_argument("--body", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    body: dict[str, Any] = {
        "schema": "neural-forge-matrix-wave34-m-composition-forensics-v1",
        "purpose": (
            "F1.  Why ARCH.CL1792M reads one case BETTER alone and three and a half "
            "cases WORSE composed, over four vocabulary rungs at byte-identical "
            "storage.  Reads sealed per-case verdicts and, on the channel arm, runs "
            "the parent twice over one corpus under two segmentations."
        ),
    }
    if args.lanes:
        body["lanes"] = lanes_report(args.matrix)
    if args.channels:
        bodies = {}
        for spec in args.body:
            name, _, path = spec.partition("=")
            bodies[name] = Path(path)
        if len(bodies) != 2:
            raise SystemExit("--channels needs exactly two --body NAME=DIR arguments")
        body["channels"] = channels_report(args.parent, args.retok_child, bodies, args.device)
    if not args.lanes and not args.channels:
        raise SystemExit("--lanes or --channels")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1), encoding="utf-8", newline="")

    if "lanes" in body:
        for family, data in body["lanes"]["families"].items():
            print(f"\n{family} family (body {data['body']}):")
            for row in data["rungs"]:
                lost = row.get("lost_from_body")
                print(f"  {row['label']:<14} {row['count']:>2}/15"
                      + (f"   lost from body: {','.join(lost)}" if lost else ""))
        print("\nat each K, plain vs metric:")
        for row in body["lanes"]["per_k"]:
            print(f"  {row['plain']:<12} {row['plain_count']:>2}  vs  "
                  f"{row['metric']:<14} {row['metric_count']:>2}   delta {row['delta']:+d}")
            print(f"      plain only : {','.join(row['plain_only']) or '-'}")
            print(f"      metric only: {','.join(row['metric_only']) or '-'}")
        print("\nlane charge (plain holds it at N more rungs than metric):")
        for lane, value in body["lanes"]["lane_charge_plain_minus_metric"].items():
            if value:
                print(f"  {lane:<32} {value:+d}")
    if "channels" in body:
        print(json.dumps(body["channels"]["sensitivity_by_group"], indent=1))
        print(body["channels"]["verdict"])
        lev = body["channels"].get("logit_leverage") or {}
        if lev.get("available"):
            print(f"\nlogit leverage through the head ({lev['head_rows']} rows, "
                  f"{lev['kept_rows']} kept):")
            for group, d in lev["groups"].items():
                lps = d["logit_per_stream"]
                rr = d["restriction_ratio"]
                print(f"  {group:<12} n {lps['n']:>5}  logit-per-stream median "
                      f"{lps['median']:>10}  restriction ratio median {rr['median']:>8}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
