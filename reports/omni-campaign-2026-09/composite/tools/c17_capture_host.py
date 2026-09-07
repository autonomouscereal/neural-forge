"""Observed host attention/residual boundaries for C17 (forward only, not charts).

Runs the existing native StreamedChimera loader/block path, stopping after the
last requested layer. One block and one unpadded evidence sequence are on the
device; residuals live on CPU. No logits, fitted weights, training or source
attention substitutes. The original evidence fit/held groups are retained.

Example (inside an independently allocated, read-only mounted runtime):
  python c17_capture_host.py --model /host --config-sha256 SHA --index-sha256 SHA \
    --host-receipt /inputs/s3-import.json --host-receipt-sha256 SHA \
    --evidence /inputs/evidence.json --evidence-sha256 SHA \
    --boundary-map /inputs/boundary_map.json --boundary-sha256 SHA \
    --chimera-tree /code/chimera --chimera-commit COMMIT --layers 0,1 \
    --device cuda:0 --dtype bfloat16 --out /run-output/c17-host

Outputs Lxx/host_boundaries.safetensors plus capture_receipt.json. These are
HOST-ONLY observations, not completed K3/host pairs or intelligence evidence.
K3's native KDA/MLA inputs/writes, BlockAttnRes state and source positions still
need an authenticated folded-only replay. Existing held_moe_x is insufficient.
Do not feed these seven boundaries into the old FFN-only C17 fitter implicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

SCHEMA = "neural-forge.c17-host-boundaries.v1"
BOUNDARIES = ("residual_in", "mixer_input", "mixer_output", "post_attention",
              "ffn_input", "ffn_output", "residual_out")
SHA = re.compile(r"[0-9a-f]{64}\Z")


class Refusal(ValueError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(path: Path) -> tuple[int, ...]:
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def pinned_json(path: Path, expected: str) -> dict[str, Any]:
    if not SHA.fullmatch(expected):
        raise Refusal(f"invalid expected SHA256 for {path}")
    before = stamp(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected or stamp(path) != before:
        raise Refusal(f"SHA256 mismatch or changing input: {path}")
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise Refusal(f"not a JSON object: {path}")
    return body


def canonical_sha(body: Any) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False, separators=(",", ":")).encode("ascii")).hexdigest()


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Refusal(f"{label} must be an integer >= {minimum}")
    return value


def evidence_rows(evidence: dict[str, Any], boundary: dict[str, Any], evidence_sha: str) -> list[dict[str, Any]]:
    """Pair common byte boundaries only, retaining row-rank and batch identities."""
    if evidence.get("schema") != "neural-forge-omni-evidence-v1":
        raise Refusal("unsupported evidence manifest schema")
    if boundary.get("schema") != "neural-forge-k3-boundary-map-v2":
        raise Refusal("unsupported boundary map schema")
    seal = boundary.get("boundary_map_sha256")
    if seal != canonical_sha({k: v for k, v in boundary.items() if k != "boundary_map_sha256"}):
        raise Refusal("boundary map seal mismatch")
    if boundary.get("evidence_manifest_file_sha256") != evidence_sha:
        raise Refusal("boundary map is not bound to this evidence manifest")
    seq = integer(boundary.get("sequence_length"), "sequence_length", minimum=1)
    raw = evidence.get("rows")
    if not isinstance(raw, list) or not raw:
        raise Refusal("evidence rows missing")
    batches = [integer(r.get("batch_index"), "batch_index") for r in raw]
    if len(set(batches)) != len(batches):
        raise Refusal("duplicate evidence batch_index")
    ordered = sorted(raw, key=lambda r: r["batch_index"])
    mapped = [r for r in boundary.get("rows", []) if r.get("file") == "k3_input_ids"]
    ranks = [integer(r.get("row_index"), "row_index") for r in mapped]
    if len(ranks) != len(set(ranks)) or set(ranks) != set(range(len(ordered))):
        raise Refusal("boundary row ranks must cover evidence exactly once")
    by_rank = dict(zip(ranks, mapped))
    result = []
    for rank, row in enumerate(ordered):
        b = by_rank[rank]
        split = row.get("split")
        if split not in ("fit", "held") or b.get("split") != split or b.get("source") != "evidence":
            raise Refusal("evidence split/source mismatch (review prompts cannot calibrate charts)")
        ids = row.get("ids")
        if not isinstance(ids, list) or not 0 < len(ids) <= seq:
            raise Refusal("empty or oversized evidence sequence")
        for token in ids:
            integer(token, "token id")
        q_real = integer(b.get("qwen_tokens_real"), "qwen_tokens_real", minimum=1)
        k_real = integer(b.get("k3_tokens_real"), "k3_tokens_real", minimum=1)
        if q_real > len(ids) or k_real > seq:
            raise Refusal("boundary real-token lengths exceed supplied sequence")
        pairs = b.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise Refusal("row has no common byte-boundary pairs")
        seen_q, seen_k, checked = set(), set(), []
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise Refusal("malformed boundary pair")
            q, k = integer(pair[0], "qwen position"), integer(pair[1], "K3 position")
            if q >= q_real or k >= k_real or q in seen_q or k in seen_k:
                raise Refusal("duplicate or non-real boundary position")
            seen_q.add(q)
            seen_k.add(k)
            checked.append((q, k))
        checked.sort()
        result.append({"row_index": rank, "batch_index": row["batch_index"], "split": split,
                       "ids": ids, "pairs": checked, "sequence_length": seq})
    if {r["split"] for r in result} != {"fit", "held"}:
        raise Refusal("both original fit and held groups are required")
    # Identical token sequences cannot cross the held boundary, even with different IDs.
    groups: dict[tuple[int, ...], str] = {}
    for row in result:
        key = tuple(row["ids"])
        if key in groups and groups[key] != row["split"]:
            raise Refusal("identical token sequence crosses fit/held groups")
        groups[key] = row["split"]
    return result


class BoundaryHooks:
    """Observational hooks; copy selected rows immediately, never alter outputs."""
    def __init__(self, block: Any, positions: torch.Tensor):
        self.positions = positions
        self.rows: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []
        try:
            self.handles.append(block.register_forward_pre_hook(lambda _m, x: self.take("residual_in", x[0])))
            self.handles.append(block.register_forward_hook(lambda _m, _x, y: self.take("residual_out", y)))
            self.handles.append(block.mixer.register_forward_hook(self.mixer))
            self.handles.append(block.norm2.register_forward_pre_hook(lambda _m, x: self.take("post_attention", x[0])))
            self.handles.append(block.ffn.register_forward_hook(self.ffn))
        except Exception:
            self.close()
            raise

    def take(self, key: str, value: torch.Tensor) -> None:
        if key in self.rows or not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1:
            raise Refusal(f"unexpected native boundary call: {key}")
        picked = value[0].index_select(0, self.positions.to(value.device)).detach().cpu().clone()
        if not torch.isfinite(picked).all():
            raise Refusal(f"nonfinite native boundary: {key}")
        self.rows[key] = picked

    def mixer(self, _module: Any, x: tuple[Any, ...], y: Any) -> None:
        self.take("mixer_input", x[0])
        self.take("mixer_output", y)

    def ffn(self, _module: Any, x: tuple[Any, ...], y: Any) -> None:
        self.take("ffn_input", x[0])
        self.take("ffn_output", y)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def capture(engine: Any, rows: list[dict[str, Any]], layers: list[int], out: Path) -> list[dict[str, Any]]:
    """CPU residual cache, one native block at a time; no LM-head evaluation."""
    if not layers or sorted(set(layers)) != layers or layers[0] < 0 or layers[-1] >= int(engine.cfg.num_layers):
        raise Refusal("layers must be a sorted unique in-range nonempty list")
    if engine.cfg.residual_mode != "none":
        raise Refusal("host capture requires plain residual host")
    if any(len(r["ids"]) > int(engine.cfg.max_sequence_length) for r in rows):
        raise Refusal("evidence exceeds native host context length")
    residuals = []
    with torch.no_grad():
        for row in rows:
            ids = torch.tensor([row["ids"]], dtype=torch.long, device=engine.device)
            engine.model._validate_token_ids(ids)
            residuals.append(engine.model.embed(ids).detach().cpu())
        reports = []
        for layer in range(layers[-1] + 1):
            block = engine._build_block(layer)
            parts: dict[str, list[torch.Tensor]] = {k: [] for k in BOUNDARIES}
            for i, row in enumerate(rows):
                x = residuals[i].to(engine.device)
                valid = torch.ones(x.shape[:2], dtype=torch.bool, device=engine.device)
                hooks = BoundaryHooks(block, torch.tensor([q for q, _ in row["pairs"]])) if layer in layers else None
                try:
                    y = block(x, token_valid_mask=valid)
                    if not torch.isfinite(y).all():
                        raise Refusal(f"nonfinite host residual at layer {layer}")
                    residuals[i] = y.detach().cpu()
                    if hooks is not None:
                        if set(hooks.rows) != set(BOUNDARIES):
                            raise Refusal("native hooks did not capture every boundary")
                        for key in BOUNDARIES:
                            parts[key].append(hooks.rows[key])
                finally:
                    if hooks is not None:
                        hooks.close()
                del x, y, valid, hooks
            if layer in layers:
                payload = {key: torch.cat(parts[key]).contiguous() for key in BOUNDARIES}
                identities = [(r["row_index"], r["batch_index"], q, k, int(r["split"] == "held"))
                              for r in rows for q, k in r["pairs"]]
                names = ("evidence_row_index", "batch_index", "host_position", "k3_position", "held")
                for column, name in enumerate(names):
                    payload[name] = torch.tensor([r[column] for r in identities], dtype=torch.int64)
                for side in ("host", "k3"):
                    payload[f"{side}_flat_position"] = (payload["evidence_row_index"] * rows[0]["sequence_length"]
                                                          + payload[f"{side}_position"])
                dest = out / f"L{layer:02d}"
                dest.mkdir()
                path = dest / "host_boundaries.safetensors"
                save_file(payload, str(path), metadata={"schema": SCHEMA, "layer": str(layer), "side": "host-only"})
                reports.append({"layer": layer, "kind": block.kind, "path": path.relative_to(out).as_posix(),
                                "sha256": digest(path), "bytes": path.stat().st_size, "rows": len(identities),
                                "boundaries": {k: {"shape": list(payload[k].shape), "dtype": str(payload[k].dtype)} for k in BOUNDARIES}})
                del payload
            del parts, block
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return reports


def checkpoint_inputs(root: Path, config_sha: str, index_sha: str) -> dict[str, Any]:
    pinned_json(root / "config.json", config_sha)
    index = pinned_json(root / "model.safetensors.index.json", index_sha)
    weights = index.get("weight_map")
    hashes = index.get("metadata", {}).get("tensor_sha256", {})
    if not isinstance(weights, dict) or not weights:
        raise Refusal("host index has no weight map")
    if any(not SHA.fullmatch(str(hashes.get(key, ""))) for key in weights):
        raise Refusal("every indexed host tensor needs an explicit digest before native verified load")
    files = {"config.json", "model.safetensors.index.json"}
    for name in weights.values():
        if not isinstance(name, str) or not name or "\\" in name or ":" in name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise Refusal("unsafe checkpoint shard path")
        if not (root / name).resolve(strict=True).is_relative_to(root):
            raise Refusal("checkpoint shard escapes model directory")
        files.add(name)
    return {name: stamp(root / name) for name in sorted(files)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import run_omni_frozen_review as native

    torch.set_num_threads(args.threads)
    root, out = args.model.resolve(strict=True), args.out.resolve()
    if out.exists():
        raise Refusal("output must be new; partial captures are preserved, never resumed or overwritten")
    sources = [root, args.chimera_tree.resolve(strict=True)]
    sources += [p.resolve(strict=True) for p in (args.evidence, args.boundary_map, args.host_receipt)]
    if any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in sources):
        raise Refusal("output overlaps a source")
    files = checkpoint_inputs(root, args.config_sha256, args.index_sha256)
    receipt = pinned_json(args.host_receipt, args.host_receipt_sha256)
    evidence = pinned_json(args.evidence, args.evidence_sha256)
    boundary = pinned_json(args.boundary_map, args.boundary_sha256)
    rows = evidence_rows(evidence, boundary, args.evidence_sha256)
    tree = native.bind_chimera_tree(args.chimera_tree, args.chimera_commit)
    code = {str(p.resolve()): digest(p) for p in [Path(__file__), Path(native.__file__),
            Path(native.__file__).with_name("s0_import_27b.py"), *sorted((args.chimera_tree / "src" / "chimera_omni").rglob("*.py"))]}
    device = str(args.device)
    inventory = native.cuda_inventory([device]) if device.startswith("cuda") else []
    if inventory:
        native.check_gpu_free(inventory)
    engine = native.StreamedChimera(root, device=device, dtype=getattr(torch, args.dtype),
                                   ledger_band_logit_mask=False, verify_tensors=True)
    layers = [int(x) for x in args.layers.split(",")]
    paired_positions = sum(len(r["pairs"]) for r in rows)
    per_layer_bytes = paired_positions * (7 * int(engine.cfg.hidden_size) * torch.empty((), dtype=engine.dtype).element_size() + 7 * 8)
    projected = {"paired_positions_per_layer": paired_positions,
                 "tensor_bytes_per_layer": per_layer_bytes,
                 "tensor_bytes_requested": per_layer_bytes * len(layers),
                 "excludes": "safetensors headers and JSON; seven full boundary arrays, no subsampling"}
    if per_layer_bytes * len(layers) > args.max_output_bytes:
        raise Refusal(f"projected capture tensors exceed --max-output-bytes: {projected}")
    out.mkdir(parents=True)
    reports = capture(engine, rows, layers, out)
    if files != checkpoint_inputs(root, args.config_sha256, args.index_sha256):
        raise Refusal("host checkpoint changed during capture")
    for p, sha in code.items():
        if digest(Path(p)) != sha:
            raise Refusal("native source changed during capture")
    for path, sha in [(args.evidence, args.evidence_sha256), (args.boundary_map, args.boundary_sha256),
                      (args.host_receipt, args.host_receipt_sha256)]:
        pinned_json(path, sha)
    result = {"schema": SCHEMA, "host_only": True, "source_host_pairs_complete": False,
              "folded_child_proven": False, "gradients_or_training": False,
              "model": str(root), "config_sha256": args.config_sha256, "index_sha256": args.index_sha256,
              "host_receipt": {"path": str(args.host_receipt), "sha256": args.host_receipt_sha256,
                               "schema": receipt.get("schema"), "acceptance_inferred": False},
              "evidence_sha256": args.evidence_sha256, "boundary_sha256": args.boundary_sha256,
              "boundary_seal": boundary["boundary_map_sha256"], "tree": tree, "code_sha256": code,
              "checkpoint_stat_stable": True, "consumed_tensor_hashes_verified": True,
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "cuda": inventory, "device": device, "dtype": args.dtype, "threads": args.threads,
              "micro_batch": 1, "residual_device": "cpu", "last_forward_layer": layers[-1],
              "prefix_only": layers[-1] != int(engine.cfg.num_layers) - 1,
              "position_sequence_length": boundary["sequence_length"],
              "projected_output": projected,
              "layers": reports, "groups": [{k: r[k] for k in ("row_index", "batch_index", "split")} for r in rows],
              "conventions": {"residual_in": "native block input", "mixer_input": "actual native mixer input after norm/mask",
                              "mixer_output": "native mixer write before final validity mask (selected positions all valid)",
                              "post_attention": "native norm2 input, after residual add", "ffn_input": "actual normalized native FFN input",
                              "ffn_output": "native FFN write before validity mask (selected positions all valid)",
                              "residual_out": "native block output; never inferred from subtracting rounded tensors"},
              "split_policy": "original evidence fit/held groups unchanged; no chart fitting; K3 gate holdout must not be relabeled final evaluation",
              "missing": ["authenticated native K3 attention/residual captures", "paired chart solve and installation", "held functional all-source evaluation"]}
    (out / "capture_receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "host-receipt", "evidence", "boundary-map", "chimera-tree", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    for name in ("config", "index", "host-receipt", "evidence", "boundary"):
        p.add_argument(f"--{name}-sha256", required=True)
    p.add_argument("--chimera-commit", required=True)
    p.add_argument("--layers", required=True, help="sorted comma-separated zero-based native host layers")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    p.add_argument("--threads", type=int, default=22)
    p.add_argument("--max-output-bytes", type=int, default=16 * 1024**3,
                   help="refuse excessive requested full captures; reduce layer window, never silently sample")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
