#!/usr/bin/env python3
"""d2_register_check.py -- the two checks between OMNI1D2's register DRY RUN and its --confirm (queues/d2_launch.sh).

Lane L-TRAIN (Claude, d2prep), 2026-10-15, from the adversarial review of d2_launch.sh 1.1.1: the queue went from the
dry run to --confirm on its exit code alone.  The ledger already refuses an unknown target, a LoRA whose A/B do not
multiply out to its region, a scale of the wrong length, orphan tensors and a base that is not its seal.  What it
cannot see is a recipe that maps an adapter onto the WRONG tensor of the right shape: `mlp.gate_proj` and
`mlp.up_proj` are the two row halves of ONE Chimera tensor `blocks.N.ffn.up.weight`, and qwen3_5 layer t is the t-th
NON-charted Chimera block (OMNI1 has 5 charted media blocks at 32-36), not `blocks.t`.  A swapped half or a shifted
layer passes every ledger check and registers a model that is not the one D2 trained.

  log    --log DRYRUN.log --checkpoint-id ID --base-id BASE --base-sha256 SHA --base-record BASE.json --delta-dir DIR
         the dry run printed exactly the expected plan: stage distill, the base pinned at its sealed digest, the chain
         BASE -> ID, ops lora / scale only (no replace) in the counts DELTA.json holds, `added` = every element of the
         delta's tensors, and parameters = the base record's count + added (a LoRA delta ADDS parameters until the D4
         merge folds them: composed = base is the wrong rule for D2), status candidate, nothing written.
  sites  --delta-dir DIR --trained D2.TRAINED --chimera ROOT
         every trained adapter, mapped through the P1 reader's OWN qwen3_5 <- Chimera map (p1_qwen35_reader, imported,
         not copied) to the Chimera tensors and regions it acts on, must be EXACTLY one DELTA.json entry: lora on
         (target, region) with A and B equal to the trained fp32 tensors bit for bit (B split by rows where the map
         assembles a qwen tensor from several Chimera parts) and alpha / rank = the trainer's scaling; scale (dim 0) on
         the K3 rows of up_proj with the trained gain.  Nothing else may be in the recipe.  The adapters and the
         training receipt are the long rung's, pinned by d2.trained; the site plan (ranks, alphas, sites, trunk) is
         re-parsed from that receipt's argv with the trainer's own parser.

Exit 0 = OK, 3 = mismatch (the queue refuses --confirm), 2 = an input is unreadable.  Prints one `D2 REGISTER-CHECK`
or `D2 SITE-CHECK` line.  CPU only; reads headers, the adapters and the delta's tensors, never a base weight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

TOOL_NAME = "d2_register_check"
TOOL_VERSION = "1.0.0"  # 2026-10-15: first version (adversarial review of d2_launch.sh 1.1.1, finding 4)
HERE = Path(__file__).resolve().parent
QWEN_LAYER_PREFIX = "model.language_model.layers."
B_OF = {"a": "b", "k3_a": "k3_b"}  # a LoRA's A part -> its B part (adapter names `layers.{i}.{site}.{part}`)


class Refusal(Exception):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_header(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        n = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(n))
    header.pop("__metadata__", None)
    return header


def delta_tensor_elements(delta_dir: Path) -> int:
    total = 0
    for shard in sorted(Path(delta_dir).glob("*.safetensors")):
        for spec in safetensors_header(shard).values():
            total += math.prod(int(n) for n in spec["shape"])
    return total


# ---------------------------------------------------------------------------------------------------------------- log
def log_check(log: str, *, checkpoint_id: str, base_id: str, base_sha256: str, base_parameters: int,
              recipe: dict[str, Any], added: int) -> list[str]:
    """The expected dry-run lines that are missing (empty = OK)."""

    ops = [e.get("op") for e in recipe.get("entries") or []]
    lora, scale, replace = ops.count("lora"), ops.count("scale"), ops.count("replace")
    problems = []
    if replace or not lora or len(ops) != lora + scale:
        problems.append(f"DELTA.json ops {sorted(set(map(str, ops)))}: D2 is lora / scale only, with at least one lora")
    if recipe.get("base_checkpoint_id") != base_id or recipe.get("base_directory_sha256") != base_sha256:
        problems.append("DELTA.json does not name the sealed D2 base")

    def num(n: int) -> str:
        return f"{int(n):,}"

    want = [
        rf"=== REGISTER {re.escape(checkpoint_id)} ===",
        r"stage\s+distill$",
        r"directory_sha256\s+[0-9a-f]{64}$",
        rf"parameters\s+{re.escape(num(base_parameters + added))}\s+\[derived-from-safetensors-headers\]",
        rf"delta base\s+{re.escape(base_id)} pinned at {re.escape(base_sha256[:16])} \(matches its seal\)",
        rf"delta chain\s+.*{re.escape(base_id)} -> {re.escape(checkpoint_id)}$",
        rf"delta ops\s+\{{'replace': 0, 'lora': {lora}, 'scale': {scale}\}} on \d+ target tensor\(s\); "
        rf"replaced 0, added {re.escape(num(added))}$",
        r"status\s+candidate\b",
        r"DRY RUN: nothing was written\.",
    ]
    lines = [ln.strip() for ln in log.splitlines()]
    problems += [f"dry-run line missing: /{w}/" for w in want if not any(re.search(w, ln) for ln in lines)]
    return problems


# -------------------------------------------------------------------------------------------------------------- sites
def load_peer(name: str) -> Any:
    sys.path.insert(0, str(HERE))
    return __import__(name)


def normalize_region(slices: list[list[int]] | None, shape: list[int]) -> tuple[tuple[int, int, int], ...]:
    region = {d: (0, int(n)) for d, n in enumerate(shape)}
    for dim, start, stop in slices or []:
        region[int(dim)] = (int(start), int(stop))
    return tuple((d, region[d][0], region[d][1]) for d in range(len(shape)))


def chimera_rows(reader: Any, qname: str) -> list[tuple[str, int, int, int]]:
    """qwen rows -> Chimera rows: [(chimera name, qwen row start, qwen row stop, chimera row offset)] for the map's
    `view` and `cat` kinds (the only kinds a trained site can be: q_proj (interleave) and norms are frozen)."""

    if qname not in reader.specs:
        raise Refusal(f"{qname} is not in the P1 map of {reader.directory}")
    spec = reader.specs[qname]
    if spec["kind"] == "view":
        (cname, rows), = spec["parts"]
        start = int(rows[0]) if rows else 0
        return [(cname, 0, int(spec["shape"][0]), start)]
    if spec["kind"] == "cat":
        out, at = [], 0
        for cname, _rows in spec["parts"]:
            n = int(reader.shards.spec(cname)["shape"][0])
            out.append((cname, at, at + n, 0))
            at += n
        return out
    raise Refusal(f"{qname} is served by the map as {spec['kind']!r}; a trained site must be a view or a row cat")


def expected_pieces(reader: Any, site_plan: dict[str, list[Any]], layer_types: list[str], adapters: dict[str, Any]) -> list[dict[str, Any]]:
    """Every (op, Chimera target, Chimera region) a correct recipe must hold, with the trained tensors it must carry.

    Trainer semantics (streamed_kd_trainer.site_forward_hook, on the qwen3_5 Linear W [out, in]):
      full    y += s B (A x)                     -> lora on W[:, :]
      ffn_in  y[:trunk] += s B (A x); y[trunk:] *= gain   -> lora on W[:trunk, :], scale dim 0 on W[trunk:, :]
      ffn_out y += s B (A x[:trunk]) + s3 B3 (A3 x[trunk:])  -> lora on W[:, :trunk] and on W[:, trunk:]
    with s = alpha / rank (oc_checkpoint composes lora as alpha / A.rows * B @ A, scale as W[region] *= gain).
    """

    pieces: list[dict[str, Any]] = []

    def lora(layer: int, site: str, parts: tuple[str, str], alpha: float, qrows: tuple[int, int], qcols: tuple[int, int], qname: str):
        a, b = adapters[f"layers.{layer}.{site}.{parts[0]}"], adapters[f"layers.{layer}.{site}.{B_OF[parts[0]]}"]
        for cname, q0, q1, off in chimera_rows(reader, qname):
            lo, hi = max(q0, qrows[0]), min(q1, qrows[1])
            if lo >= hi:
                continue
            pieces.append({"op": "lora", "layer": layer, "site": site, "part": parts[0], "target": cname,
                           "region": ((0, off + lo - q0, off + hi - q0), (1, qcols[0], qcols[1])),
                           "a": a, "b": b[lo - qrows[0]: hi - qrows[0]], "alpha": float(alpha), "rank": int(a.shape[0])})

    for layer, layer_type in enumerate(layer_types):
        for spec in site_plan[layer_type]:
            qname = f"{QWEN_LAYER_PREFIX}{layer}.{spec.site}.weight"
            out_f, in_f, trunk = int(spec.out_features), int(spec.in_features), int(spec.trunk)
            if spec.kind == "full":
                lora(layer, spec.site, ("a", "b"), spec.alpha, (0, out_f), (0, in_f), qname)
            elif spec.kind == "ffn_in":
                lora(layer, spec.site, ("a", "b"), spec.alpha, (0, trunk), (0, in_f), qname)
                gain_key = f"layers.{layer}.{spec.site}.gain"
                if gain_key in adapters:
                    gain = adapters[gain_key]
                    for cname, q0, q1, off in chimera_rows(reader, qname):
                        lo, hi = max(q0, trunk), min(q1, out_f)
                        if lo < hi:
                            pieces.append({"op": "scale", "layer": layer, "site": spec.site, "part": "gain", "target": cname,
                                           "region": ((0, off + lo - q0, off + hi - q0), (1, 0, in_f)), "dim": 0,
                                           "tensor": gain[lo - trunk: hi - trunk]})
            elif spec.kind == "ffn_out":
                lora(layer, spec.site, ("a", "b"), spec.alpha, (0, out_f), (0, trunk), qname)
                if f"layers.{layer}.{spec.site}.k3_a" in adapters:
                    lora(layer, spec.site, ("k3_a", "k3_b"), spec.k3_alpha, (0, out_f), (trunk, in_f), qname)
            else:
                raise Refusal(f"unknown site kind {spec.kind!r}")
    return pieces


def sites_check(pieces: list[dict[str, Any]], recipe: dict[str, Any], delta: dict[str, Any], shapes: dict[str, list[int]]) -> tuple[list[str], dict[str, Any]]:
    """Each recipe entry against the expected pieces: (problems, counts)."""

    import torch

    want = {(p["op"], p["target"], p["region"]): p for p in pieces}
    seen: dict[tuple, int] = {}
    problems: list[str] = []
    for number, entry in enumerate(recipe.get("entries") or []):
        target = entry.get("target")
        if target not in shapes:
            problems.append(f"entry {number}: target {target!r} is not a Chimera tensor of the base")
            continue
        key = (entry.get("op"), target, normalize_region(entry.get("slice"), shapes[target]))
        piece = want.get(key)
        if piece is None:
            problems.append(f"entry {number}: {key[0]} on {target} {list(key[2])} is not where any trained adapter acts")
            continue
        if key in seen:
            problems.append(f"entry {number}: {key[0]} on {target} {list(key[2])} repeats entry {seen[key]}")
            continue
        seen[key] = number
        where = f"entry {number} (layer {piece['layer']} {piece['site']} {piece['part']} -> {target})"
        if piece["op"] == "lora":
            a, b = delta.get(entry.get("a")), delta.get(entry.get("b"))
            if a is None or b is None:
                problems.append(f"{where}: A / B tensor missing from the delta")
                continue
            if a.dtype != torch.float32 or b.dtype != torch.float32:
                problems.append(f"{where}: A / B stored as {a.dtype} / {b.dtype}, not the trained fp32")
            if tuple(a.shape) != tuple(piece["a"].shape) or not torch.equal(a.float(), piece["a"].float()):
                problems.append(f"{where}: A is not the trained {piece['part']}")
            if tuple(b.shape) != tuple(piece["b"].shape) or not torch.equal(b.float(), piece["b"].float()):
                problems.append(f"{where}: B is not the trained rows of {B_OF[piece['part']]}")
            alpha = entry.get("alpha")
            if not isinstance(alpha, (int, float)) or not math.isclose(float(alpha) / int(a.shape[0]), piece["alpha"] / piece["rank"],
                                                                          rel_tol=1e-12):
                problems.append(f"{where}: alpha {alpha!r} / rank {int(a.shape[0])} is not the trained scaling "
                                f"{piece['alpha']} / {piece['rank']}")
        else:
            gain = delta.get(entry.get("tensor"))
            if entry.get("dim") != 0:
                problems.append(f"{where}: scale dim {entry.get('dim')!r}, the gain scales rows (dim 0)")
            if gain is None or gain.dtype != torch.float32 or tuple(gain.shape) != tuple(piece["tensor"].shape) \
                    or not torch.equal(gain.float(), piece["tensor"].float()):
                problems.append(f"{where}: the gain is not the trained fp32 gain")
    missing = [p for k, p in want.items() if k not in seen]
    problems += [f"no entry for layer {p['layer']} {p['site']} {p['part']}: {p['op']} on {p['target']} {list(p['region'])}"
                 for p in missing]
    counts = {"pieces": len(pieces), "entries": len(recipe.get("entries") or []), "matched": len(seen), "missing": len(missing),
              "lora": sum(1 for p in pieces if p["op"] == "lora"), "scale": sum(1 for p in pieces if p["op"] == "scale"),
              "layers": len({p["layer"] for p in pieces})}
    return problems, counts


def load_delta(delta_dir: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    out: dict[str, Any] = {}
    for shard in sorted(Path(delta_dir).glob("*.safetensors")):
        out.update(load_file(str(shard), device="cpu"))
    return out


def trained_inputs(trained_path: Path) -> tuple[Path, dict[str, Any]]:
    """(the long rung's adapters, its training receipt), both checked against the shas d2.trained recorded."""

    trained = json.loads(Path(trained_path).read_text(encoding="utf-8"))
    final = trained["long"]
    adapters = Path(final["adapters"])
    if sha256_file(adapters) != final["sha256"]:
        raise Refusal(f"{adapters} does not digest to the sha d2.trained recorded")
    receipt_path = adapters.parent / "training-receipt.json"
    if sha256_file(receipt_path) != final["receipt_sha256"]:
        raise Refusal(f"{receipt_path} does not digest to the sha d2.trained recorded")
    return adapters, json.loads(receipt_path.read_text(encoding="utf-8"))


def run_sites(delta_dir: Path, trained_path: Path, chimera: Path) -> tuple[list[str], dict[str, Any]]:
    T = load_peer("streamed_kd_trainer")
    adapters_path, receipt = trained_inputs(trained_path)
    argv = list(receipt.get("argv") or [])
    if "train" not in argv:
        raise Refusal("the long rung's receipt has no `train` argv")
    ns = T.parser().parse_args(argv[argv.index("train"):])
    source = T.P1Source(Path(chimera), Path(ns.p1_parent_config), None)  # the map only: no plan, no base weight read
    config = T.text_config_from_source(source, T.resolve_dtype(ns.dtype), ns.attn_implementation)
    plan = T.build_site_plan(config, T.site_plan_from_args(ns))
    layer_types = [str(t) for t in config.layer_types[: int(config.num_hidden_layers)]]
    adapters = T.load_adapters(adapters_path)
    pieces = expected_pieces(source.reader, plan, layer_types, adapters)
    used = {f"layers.{p['layer']}.{p['site']}.{part}" for p in pieces for part in (p["part"], B_OF.get(p["part"], p["part"]))}
    unmapped = sorted(set(adapters) - used)
    recipe = json.loads((Path(delta_dir) / "DELTA.json").read_text(encoding="utf-8"))
    shapes = {name: list(source.reader.shards.spec(name)["shape"]) for name in source.reader.shards.names()}
    problems, counts = sites_check(pieces, recipe, load_delta(delta_dir), shapes)
    problems = [f"trained adapter tensor {n} maps to no piece" for n in unmapped] + problems
    counts.update(adapters=str(adapters_path), adapters_tensors=len(adapters), chimera=str(chimera))
    return problems, counts


# ---------------------------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("log")
    s.add_argument("--log", required=True, type=Path)
    for k in ("checkpoint-id", "base-id", "base-sha256"):
        s.add_argument(f"--{k}", required=True)
    s.add_argument("--base-record", required=True, type=Path, help="`oc_checkpoint.py show` of the D2 base")
    s.add_argument("--delta-dir", required=True, type=Path)
    s = sub.add_parser("sites")
    s.add_argument("--delta-dir", required=True, type=Path)
    s.add_argument("--trained", required=True, type=Path, help="$D2/d2.trained (pins the long rung's adapters + receipt)")
    s.add_argument("--chimera", required=True, type=Path, help="the D2 base's root Chimera directory (its map)")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "log":
            record = json.loads(args.base_record.read_text(encoding="utf-8"))
            base_parameters = int((record.get("checkpoint") or record)["parameter_count"])
            recipe = json.loads((args.delta_dir / "DELTA.json").read_text(encoding="utf-8"))
            added = delta_tensor_elements(args.delta_dir)
            problems = log_check(args.log.read_text(encoding="utf-8", errors="replace"), checkpoint_id=args.checkpoint_id,
                                 base_id=args.base_id, base_sha256=args.base_sha256, base_parameters=base_parameters,
                                 recipe=recipe, added=added)
            body = {"checkpoint_id": args.checkpoint_id, "base_parameters": base_parameters, "added": added,
                    "composed": base_parameters + added, "problems": problems}
            print(f"D2 REGISTER-CHECK {'OK' if not problems else 'MISMATCH'} " + json.dumps(body, sort_keys=True), flush=True)
            return 0 if not problems else 3
        problems, counts = run_sites(args.delta_dir, args.trained, args.chimera)
        body = {**counts, "problems": problems[:20], "problems_total": len(problems)}
        print(f"D2 SITE-CHECK {'OK' if not problems else 'MISMATCH'} " + json.dumps(body, sort_keys=True, default=str), flush=True)
        return 0 if not problems else 3
    except (Refusal, OSError, KeyError, ValueError) as exc:
        print(f"REFUSED: {TOOL_NAME}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
