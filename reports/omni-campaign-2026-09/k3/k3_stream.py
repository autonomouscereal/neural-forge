#!/usr/bin/env python3
"""K3-1(a): the streamed per-layer calibration forward.

One pass over Kimi-K3.  Layer i is read off the share, run, cached, and
dropped; the only thing that crosses a layer boundary is the residual PAIR
(`prefix_sum`, `block_residual`), because `attn_res_block_size = 12` makes the
residual stream a softmax read over up to eight snapshots rather than a sum.

What it writes per layer, under ``--cache-dir/T{tokens}/L{i:02d}/``:

    tensors.safetensors   moe_x bf16 [T,H], latent_in bf16 [T,3584],
                          y_lat fp32 [T,3584] (the PRE-norm routed accumulator,
                          i.e. the C14 ridge target -- fp32 is mandatory),
                          router_topk_idx i16 [T,16], router_topk_w bf16 [T,16]
                          (+ attn_in/attn_out/moe_out/prefix_out with --cache-set full)
    receipt.json          shapes, dtypes, sha256 per tensor, read MB/s, per-half
                          seconds, experts touched, router mass concentration

and a restart snapshot (`prefix_sum` + the block stack) every
``--snapshot-every`` layers so a killed pass resumes without re-reading what it
already paid for.

Cost model, measured (`k3_layer1_probe_cpu.json`): the read dominates.  Layer 1
was 16.99 GB at 5.3 MB/s = 3207 s while the share was contended; compute was
245 s of dequant and 2.5 s of MoE.  Nothing here is compute-bound, so the
harness logs MB/s per shard and puts it in every receipt.

Usage (server, lab venv):
  python k3_stream.py --model-dir /mnt/ai_data/models/moonshotai/Kimi-K3 \\
      --input-ids ids.safetensors --layers 0-92 --device cuda:0 \\
      --kda-backend fla --cache-dir /home/cereal/omni-campaign/k3/cache
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from k3_common import (
    K3Index,
    apply_attn_res,
    canonical_device,
    device_inventory,
    log,
    read_tensors,
    rms_norm,
    save_tensors,
    sha256_tensor,
    write_receipt,
)
from k3_layer import LayerWeights, StreamedLayer, load_layer

MINIMAL_TAPS = ("moe_x", "latent_in", "y_lat", "router_topk_idx", "router_topk_w")
FULL_TAPS = MINIMAL_TAPS + ("attn_in", "attn_out", "moe_out", "prefix_out")


# --------------------------------------------------------------------------
def parse_layers(spec: str, total: int) -> list[int]:
    out: list[int] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo, hi = piece.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(piece))
    bad = [i for i in out if not 0 <= i < total]
    if bad:
        raise SystemExit(f"layers out of range for a {total}-layer model: {bad}")
    return out


def load_input_ids(args) -> torch.Tensor:
    if args.input_ids:
        from safetensors.torch import load_file

        ids = load_file(args.input_ids)["input_ids"].to(torch.long)
        return ids if ids.dim() == 2 else ids.unsqueeze(0)
    generator = torch.Generator().manual_seed(args.seed)
    vocab = args.vocab_size
    return torch.randint(0, vocab, (1, args.random_tokens), generator=generator, dtype=torch.long)


def staged_body_path(body_dir: str, layer_idx: int) -> str:
    return os.path.join(body_dir, f"L{layer_idx:02d}.safetensors")


def load_layer_maybe_staged(index: K3Index, layer_idx: int, args) -> tuple[LayerWeights, dict]:
    """Read the layer; take the non-expert half from local staging when present.

    Staging the body (~1.3 GB/layer, ~100 GB total) off the CIFS share is the
    one caching that pays for itself immediately: a second pass over the same
    layer then only re-reads the packed experts.
    """
    keys = index.layer_keys(layer_idx)
    body_path = staged_body_path(args.body_dir, layer_idx) if args.body_dir else None
    if body_path and os.path.exists(body_path):
        from safetensors.torch import load_file

        staged = load_file(body_path)
        expert_keys = [k for k in keys if ".experts." in k]
        started = time.time()
        tensors, timing = index.read_keys(expert_keys, progress_tag=f"L{layer_idx:02d} experts")
        tensors.update(staged)
        timing["staged_body_seconds"] = round(time.time() - started, 1)
        timing["staged_body"] = body_path
        weights = LayerWeights(layer_idx, tensors, index)
        weights.to(args.device, args.experts_device or args.device)
        return weights, timing

    weights, timing = load_layer(index, layer_idx, args.device, args.experts_device or args.device)
    if args.body_dir:
        os.makedirs(args.body_dir, exist_ok=True)
        prefix = index.layer_prefix_for(layer_idx)
        body = {}
        for group, group_name in ((weights.attn, "self_attn."), (weights.res, ""), (weights.body, "")):
            for short, tensor in group.items():
                body[f"{prefix}{group_name}{short}"] = tensor.cpu()
        timing["staged_body_sha256"] = save_tensors(staged_body_path(args.body_dir, layer_idx), body)
        timing["staged_body"] = staged_body_path(args.body_dir, layer_idx)
    return weights, timing


# --------------------------------------------------------------------------
class StreamRun:
    def __init__(self, args):
        self.args = args
        self.index = K3Index(args.model_dir)
        self.cfg = self.index.text_config
        # canonical (indexed) form -- `cuda` and `cuda:0` are one card, and
        # the receipt should name the card the tensors actually landed on
        self.device = canonical_device(args.device)
        self.experts_device = canonical_device(args.experts_device or args.device)
        self.cache_root = os.path.join(args.cache_dir, f"T{args.tokens_tag}")
        os.makedirs(self.cache_root, exist_ok=True)
        self.taps = () if args.cache_set == "none" else (
            MINIMAL_TAPS if args.cache_set == "minimal" else FULL_TAPS
        )
        self.receipt = {
            "model_dir": args.model_dir,
            "device": str(self.device),
            "experts_device": str(self.experts_device),
            "device_requested": args.device,
            "experts_device_requested": args.experts_device or args.device,
            "kda_backend": args.kda_backend,
            "cache_set": args.cache_set,
            "devices": device_inventory(),
            "torch": torch.__version__,
            "layers": {},
        }
        # A resumed pass used to start `layers` empty and overwrite the receipt
        # after every layer, so a 93-layer run resumed three times kept only the
        # per-layer read/sha record of the last segment.  Carry the earlier
        # segments forward instead.
        existing = os.path.join(self.cache_root, "stream_receipt.json")
        if args.resume and os.path.exists(existing):
            try:
                with open(existing, "r", encoding="utf-8") as handle:
                    prior = json.load(handle)
                self.receipt["layers"].update(prior.get("layers", {}))
                self.receipt["resumed_from_receipt"] = {
                    "layers_carried_forward": sorted(int(k) for k in prior.get("layers", {})),
                    "previous_receipt_sha256": prior.get("receipt_sha256"),
                }
            except (OSError, ValueError) as exc:
                self.receipt["resumed_from_receipt"] = {"error": repr(exc)[:200]}

    # -- model-level tensors ----------------------------------------------
    def _model_tensor(self, name_candidates: list[str]) -> torch.Tensor:
        for name in name_candidates:
            if name in self.index.weight_map:
                shard = self.index.weight_map[name]
                got = read_tensors(os.path.join(self.args.model_dir, shard), [name])
                return got[name]
        raise SystemExit(f"none of {name_candidates} is in the index")

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        prefix = self.index.model_prefix
        table = self._model_tensor([f"{prefix}embed_tokens.weight", "model.embed_tokens.weight"])
        embedded = table[input_ids.flatten()].reshape(*input_ids.shape, -1)
        del table
        return embedded.to(self.device)

    def head(self, hidden: torch.Tensor, block_residual: torch.Tensor) -> torch.Tensor:
        """output attn-res read -> final norm -> lm_head."""
        prefix = self.index.model_prefix
        batch, seq, size = hidden.shape
        proj = self._model_tensor([f"{prefix}output_attn_res_proj.weight"]).to(self.device)
        norm = self._model_tensor([f"{prefix}output_attn_res_norm.weight"]).to(self.device)
        hidden = apply_attn_res(
            hidden.view(-1, size), block_residual, proj, norm, float(self.cfg["rms_norm_eps"])
        ).view(batch, seq, size)
        final = self._model_tensor([f"{prefix}norm.weight"]).to(self.device)
        hidden = rms_norm(hidden, final, float(self.cfg["rms_norm_eps"]))
        lm_head = self._model_tensor(
            [f"{self.index.layer_prefix.split('model.layers.')[0]}lm_head.weight", "lm_head.weight"]
        ).to(self.device)
        return torch.nn.functional.linear(hidden, lm_head)

    # -- snapshots ---------------------------------------------------------
    def snapshot_path(self, layer_idx: int) -> str:
        return os.path.join(self.cache_root, "blocks", f"before_L{layer_idx:02d}.safetensors")

    def save_snapshot(self, layer_idx: int, prefix_sum: torch.Tensor, block_residual: torch.Tensor) -> str:
        return save_tensors(
            self.snapshot_path(layer_idx),
            {"prefix_sum": prefix_sum.cpu(), "block_residual": block_residual.cpu()},
            metadata={"layer": str(layer_idx)},
        )

    def load_snapshot(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from safetensors.torch import load_file

        got = load_file(self.snapshot_path(layer_idx))
        return got["prefix_sum"].to(self.device), got["block_residual"].to(self.device)

    def resume_point(self, layers: list[int]) -> int | None:
        available = [i for i in layers if os.path.exists(self.snapshot_path(i))]
        return max(available) if available else None

    # -- the pass ----------------------------------------------------------
    def run(self) -> dict:
        args = self.args
        layers = parse_layers(args.layers, self.index.num_layers)
        input_ids = load_input_ids(args)
        tokens = int(input_ids.numel())
        self.receipt["tokens"] = tokens
        self.receipt["input_ids_sha256"] = sha256_tensor(input_ids)

        start_layer = layers[0]
        if args.resume:
            found = self.resume_point(layers)
            if found is not None and found >= start_layer:
                start_layer = found
                log(f"resuming from the snapshot before layer {start_layer}")
        if start_layer == layers[0] and start_layer == 0:
            prefix_sum = self.embed(input_ids)
            block_residual = prefix_sum.new_zeros(prefix_sum.shape[0] * prefix_sum.shape[1], 0,
                                                  prefix_sum.shape[2])
        else:
            prefix_sum, block_residual = self.load_snapshot(start_layer)

        wall_start = time.time()
        processed = [i for i in layers if i >= start_layer]
        for layer_idx in processed:
            if args.snapshot_every and layer_idx % args.snapshot_every == 0:
                self.save_snapshot(layer_idx, prefix_sum, block_residual)
            layer_start = time.time()
            weights, timing = load_layer_maybe_staged(self.index, layer_idx, args)
            runner = StreamedLayer(weights, self.cfg, kda_backend=args.kda_backend)
            compute_start = time.time()
            prefix_sum, block_residual, taps = runner.forward(
                prefix_sum, block_residual, collect=bool(self.taps)
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            compute_seconds = time.time() - compute_start
            entry = self.write_layer_cache(layer_idx, taps, weights)
            entry["read"] = timing
            entry["compute_seconds"] = round(compute_seconds, 2)
            entry["layer_seconds"] = round(time.time() - layer_start, 1)
            entry["is_kda"] = weights.is_kda
            entry["is_moe"] = weights.is_moe
            entry["parameters"] = weights.parameter_counts()
            entry["prefix_out_sha256"] = sha256_tensor(prefix_sum)
            entry["block_residual_blocks"] = int(block_residual.shape[1])
            self.receipt["layers"][str(layer_idx)] = entry
            log(f"L{layer_idx:02d} done: read {timing['gigabytes']} GB at {timing['MB_per_s']} MB/s, "
                f"compute {compute_seconds:.1f}s, blocks={int(block_residual.shape[1])}")
            del weights, runner, taps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            write_receipt(os.path.join(self.cache_root, "stream_receipt.json"), self.receipt)

        # A snapshot for the layer AFTER the window, so the next `--layers` range
        # has something to resume from.  Without it a pass over 0-11 left nothing
        # a pass over 12-23 could start from: `resume_point` found no snapshot
        # inside [12, 23] and `load_snapshot(12)` raised.  That was documented as
        # a limitation; it is cheaper to remove than to document.
        if processed and args.snapshot_every:
            nxt = processed[-1] + 1
            if nxt < self.index.num_layers:
                self.receipt["end_of_window_snapshot"] = {
                    "layer": nxt, "sha256": self.save_snapshot(nxt, prefix_sum, block_residual)
                }

        if args.emit_logits and layers[-1] == self.index.num_layers - 1:
            logits = self.head(prefix_sum, block_residual)
            top = torch.topk(logits[0, -1].float(), k=min(args.topk_logits, logits.shape[-1]))
            path = os.path.join(self.cache_root, "logits.safetensors")
            self.receipt["logits"] = {
                "path": path,
                "sha256": save_tensors(path, {"logits": logits[0].to(torch.float32).cpu()}),
                "last_token_top_ids": top.indices.cpu().tolist(),
                "last_token_top_values": [round(float(v), 5) for v in top.values.cpu()],
            }

        self.receipt["total_seconds"] = round(time.time() - wall_start, 1)
        self.receipt["final_prefix_sha256"] = sha256_tensor(prefix_sum)
        path = os.path.join(self.cache_root, "stream_receipt.json")
        self.receipt["receipt_sha256"] = write_receipt(path, self.receipt)
        log(f"receipt -> {path}")
        return self.receipt

    def write_layer_cache(self, layer_idx: int, taps: dict, weights: LayerWeights) -> dict:
        entry: dict = {}
        if not self.taps:
            return entry
        out_dir = os.path.join(self.cache_root, f"L{layer_idx:02d}")
        payload = {}
        for name in self.taps:
            if name not in taps:
                continue
            tensor = taps[name]
            tensor = tensor.reshape(-1, tensor.shape[-1]) if tensor.dim() == 3 else tensor
            payload[name] = tensor.cpu().contiguous()
        if not payload:
            return entry
        path = os.path.join(out_dir, "tensors.safetensors")
        entry["cache_path"] = path
        entry["cache_sha256"] = save_tensors(path, payload)
        entry["cached"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                               "sha256": sha256_tensor(v)[:16]} for k, v in payload.items()}
        if "router_counts" in taps:
            counts = taps["router_counts"].tolist()
            mass = taps["router_mass"].tolist()
            total = sum(mass) or 1.0
            ordered = sorted(range(len(mass)), key=lambda e: -mass[e])
            stats = {
                "experts": len(mass),
                "experts_touched": int(sum(1 for c in counts if c > 0)),
                "max_tokens_on_one_expert": int(max(counts)) if counts else 0,
                "expert_count": counts,
                "expert_mass": mass,
                "top16_mass_fraction": round(sum(mass[e] for e in ordered[:16]) / total, 5),
            }
            with open(os.path.join(out_dir, "router_stats.json"), "w", encoding="utf-8") as handle:
                json.dump(stats, handle)
            entry["router"] = {k: stats[k] for k in
                               ("experts", "experts_touched", "max_tokens_on_one_expert", "top16_mass_fraction")}
        return entry


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--cache-dir", default="/home/cereal/omni-campaign/k3/cache")
    ap.add_argument("--body-dir", default=None, help="stage/reuse the non-expert half here")
    ap.add_argument("--layers", default="0-92")
    ap.add_argument("--input-ids", default=None, help="safetensors with an `input_ids` tensor")
    ap.add_argument("--random-tokens", type=int, default=256)
    ap.add_argument("--vocab-size", type=int, default=163840)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--experts-device", default=None)
    ap.add_argument("--kda-backend", default="torch", choices=("torch", "fla"))
    ap.add_argument("--cache-set", default="minimal", choices=("none", "minimal", "full"))
    ap.add_argument("--snapshot-every", type=int, default=12)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--emit-logits", action="store_true")
    ap.add_argument("--topk-logits", type=int, default=16)
    ap.add_argument("--tokens-tag", default=None, help="cache subdirectory name; defaults to the token count")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.tokens_tag is None:
        args.tokens_tag = str(args.random_tokens if not args.input_ids else "custom")
    run = StreamRun(args)
    run.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
