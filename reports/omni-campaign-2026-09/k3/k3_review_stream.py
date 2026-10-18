#!/usr/bin/env python3
"""K3-1(d): the streamed whole-model review runner, N layers folded to dense.

A hard fact this file is built around, and it is worth writing down because it
kills the plan's original phrasing: the MoE parent CANNOT generate here.  Ninety-two
layers of packed experts are 1.37 TB, RAM+VRAM is 435 GB, and per-token expert
fetch off the CIFS share is ~24 GB per token.  So there are exactly two
instruments, and only one of them is a behaviour instrument:

  --mode proxy  (R-proxy, ANY N, DIAGNOSTIC ONLY, never a gate)
      One streamed teacher-forced pass over prompt + fixed reference
      continuation, layers in --dense-layers swapped for their C14 children,
      every other layer read off the share exactly once.  Records
      KL(parent || child) on the logits, top-1/top-5 agreement, and the drift
      of the final hidden state.  Its ONLY job is to ORDER layers and rungs.

      STREAMED means streamed: `ReviewModel.stream_proxy` holds ONE layer at a
      time and carries only the per-case residual pair.  It has to.  An
      unfolded K3 MoE layer is ~15.7 GB packed, so the resident loader would
      need ~1.44 TB for 92 of them against 251 GiB of RAM -- i.e. the parent
      baseline (`--emit-baseline`, no folds at all) and every N below "all MoE
      layers folded" are exactly the cases a resident proxy cannot run, which
      is most of what the instrument is for.  All cases go through a layer
      together, so each layer is still read once per invocation, not once per
      case.

  --mode decode  (R-full, the behaviour instrument, Cyril-scored)
      Requires every MoE layer to have a dense fold, at which point the whole
      model is RAM/VRAM resident and greedy decode is a normal forward loop
      with a per-layer state (conv + recurrent for KDA, K/V for MLA).  With a
      fold present for a layer this runner never reads that layer's packed
      experts at all -- which is the difference between 16 GB and 1.3 GB of
      share traffic per layer.

A BASELINE IS PART OF THE MEASUREMENT, so it is stamped and checked.
`--emit-baseline` writes the run's logits with a `__metadata__` block naming
the mode, the fold census and the rung that produced them, and it REFUSES to
write at all while any layer is folded unless `--allow-folded-baseline` says
so out loud.  `--baseline` reads that stamp back, records the file's path and
sha256 in the receipt, and refuses a baseline that was itself a child.  The
hole this closes is the mirror image of `_check_requested_folds`: copy the
child command line, forget to drop `--dense-dir/--rung`, and
`parent_logits.safetensors` is a rung-65,536 child.  The next run then scores
the 17,408 child against it, reports KL ~ 1e-4 with high top-1 agreement --
a far better child than the truth -- and nothing in `review_child.json` said
which baseline it was.

Output is the transcript shape `seal_matrix_behavior_review.py` expects:
verbatim responses per case, termination noted, artifact-tolerant (loops,
echo, cap-rides and a missing EOS are NOT content misses and this file does
not score them).
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
    peak_rss_bytes,
    read_metadata,
    read_tensors,
    rms_norm,
    save_tensors,
    sha256_file,
    sha256_tensor,
    write_receipt,
)
from k3_layer import LayerWeights, StreamedLayer, new_state


# --------------------------------------------------------------------------
def scan_fold_dir(dense_dir: str | None) -> list[tuple[int, int, str]]:
    """Every form-E fold in a directory as (layer, rung, path).

    The `K` in the filename is the REQUESTED rung, which is the key `--rung`
    is matched against; `c14_union_coreset.py` writes the width it actually
    installed into the safetensors metadata, because rank starvation can land
    below the rung and the two used to disagree silently.
    """
    if not dense_dir or not os.path.isdir(dense_dir):
        return []
    found = []
    for name in sorted(os.listdir(dense_dir)):
        if not name.endswith("_formE.safetensors") or not name.startswith("L"):
            continue
        try:
            # the layer digits, however many there are.  `int(name[1:3])` was
            # safe only because c14 writes `L{idx:02d}` and 93 layers never
            # need three -- two conventions coupled by nothing, and a
            # three-digit layer would have been silently DROPPED, which is the
            # same silent-parent failure `_check_requested_folds` exists for.
            digits = name[1:].split("_", 1)[0]
            layer = int(digits)
            rung = int(name.split("_K")[1].split("_")[0])
        except (ValueError, IndexError):
            continue
        found.append((layer, rung, os.path.join(dense_dir, name)))
    return found


def discover_folds(dense_dir: str | None, rung: int | None) -> dict[int, str]:
    """Map layer index -> form-E fold file, honouring an explicit rung."""
    found: dict[int, tuple[int, str]] = {}
    for layer, width, path in scan_fold_dir(dense_dir):
        if rung is not None and width != rung:
            continue
        # widest fold wins unless a rung was named
        if layer not in found or width > found[layer][0]:
            found[layer] = (width, path)
    return {layer: path for layer, (_, path) in found.items()}


def load_fold(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    got = load_file(path)
    return {
        "w1_s": got["w1_s"].to(device),
        "w3_s": got["w3_s"].to(device),
        "w2p": got["w2p"].to(device),
    }


# --------------------------------------------------------------------------
class ReviewModel:
    """Every layer, loaded once, with folds substituted where they exist."""

    def __init__(self, index: K3Index, args):
        self.index = index
        self.cfg = index.text_config
        self.args = args
        self.device = canonical_device(args.device)
        self.available = scan_fold_dir(args.dense_dir)
        self.folds = discover_folds(args.dense_dir, args.rung)
        self.requested_layers: set[int] | None = None
        if args.dense_layers:
            self.requested_layers = set(int(x) for x in args.dense_layers.split(",") if x.strip() != "")
            self.folds = {k: v for k, v in self.folds.items() if k in self.requested_layers}
        self.layers: list[StreamedLayer] = []
        self.dense: dict[int, dict[str, torch.Tensor]] = {}
        self.load_report: list[dict] = []
        self.states: list = []
        self.model_tensors: dict[str, torch.Tensor] = {}
        self.load_seconds: float | None = None
        self._check_requested_folds()

    def _check_requested_folds(self) -> None:
        """A fold request that selected NOTHING is an error, not a quiet parent run.

        `--rung` is matched against the `K` in the filename.  When the two keys
        drifted apart -- the fold file used to be named by the width installed,
        not the rung requested -- `--rung 17408` silently matched no file, the
        proxy then ran the UNFOLDED parent against the parent baseline, and the
        receipt read as a perfect child (KL ~ 0, top-1 agreement 1.0).  Only
        `layers_folded: []` betrayed it and nothing checked that field.  Decode
        mode was safe because `plan()` refuses; proxy mode is most of what the
        instrument is for.
        """
        asked = bool(self.args.dense_dir) or bool(self.args.dense_layers) or self.args.rung is not None
        if not asked:
            return
        widths = sorted({w for _, w, _ in self.available})
        layers = sorted({layer for layer, _, _ in self.available})
        if not self.folds:
            raise SystemExit(
                f"a fold was requested (--dense-dir {self.args.dense_dir!r}, "
                f"--rung {self.args.rung}, --dense-layers {self.args.dense_layers!r}) but NO fold "
                f"file matched, so this run would review the unfolded parent and read as a perfect "
                f"child.  Rungs present in --dense-dir: {widths or 'none'}; layers: {layers or 'none'}."
            )
        if self.requested_layers is not None:
            missing = sorted(self.requested_layers - set(self.folds))
            if missing:
                raise SystemExit(
                    f"--dense-layers asked for {sorted(self.requested_layers)} but no fold matched "
                    f"layers {missing} at rung {self.args.rung}.  Rungs present: {widths or 'none'}; "
                    f"layers present: {layers or 'none'}."
                )

    # -- planning (no weights read) ----------------------------------------
    def plan(self) -> dict:
        """What this run WOULD do, computed from the index alone.

        Deliberately cheap and deliberately called before any weight read: the
        decode refusal below depends only on `is_moe_layer` and the discovered
        folds, and running it after `load()` made it unreachable on the real
        model -- the process would OOM during the load it was supposed to
        prevent.
        """
        moe_layers = [i for i in range(self.index.num_layers) if self.index.is_moe_layer(i)]
        unfolded = [i for i in moe_layers if i not in self.folds]
        return {
            "layers_folded": sorted(self.folds),
            "fold_rung_requested": self.args.rung,
            "folds_available_in_dense_dir": sorted({w for _, w, _ in self.available}),
            "moe_layers": len(moe_layers),
            "moe_layers_unfolded": len(unfolded),
            "unfolded_moe_layers_head": unfolded[:8],
        }

    # -- loading -----------------------------------------------------------
    def _layer_keys(self, layer_idx: int) -> tuple[list[str], str | None]:
        """Keys for one layer, minus the packed experts when a fold replaces them."""
        keys = self.index.layer_keys(layer_idx)
        fold_path = self.folds.get(layer_idx)
        if fold_path:
            keys = [k for k in keys if ".experts." not in k]
        return keys, fold_path

    def _load_one_layer(self, layer_idx: int) -> tuple[StreamedLayer, dict[str, torch.Tensor] | None, dict]:
        keys, fold_path = self._layer_keys(layer_idx)
        tensors, timing = self.index.read_keys(keys, progress_tag=None)
        weights = LayerWeights(layer_idx, tensors, self.index)
        weights.to(self.device)
        runner = StreamedLayer(weights, self.cfg, kda_backend=self.args.kda_backend)
        dense = load_fold(fold_path, self.device) if fold_path else None
        report = {
            "layer": layer_idx,
            "folded": bool(fold_path),
            "fold": fold_path,
            "bytes": timing["bytes"],
            "gigabytes": timing["gigabytes"],
            "MB_per_s": timing["MB_per_s"],
            "dense_channels": int(dense["w1_s"].shape[0]) if dense is not None else None,
        }
        log(f"L{layer_idx:02d} loaded {'(folded)' if fold_path else ''} "
            f"{timing['gigabytes']} GB at {timing['MB_per_s']} MB/s")
        return runner, dense, report

    def load(self) -> None:
        """RESIDENT load -- every layer at once.  Only `--mode decode` may use it.

        Decode needs every layer simultaneously anyway (a token walks the whole
        trunk), and decode already refuses unless every MoE layer is folded, so
        the packed experts are never what is being held here.
        """
        started = time.time()
        for layer_idx in range(self.index.num_layers):
            runner, dense, report = self._load_one_layer(layer_idx)
            self.layers.append(runner)
            if dense is not None:
                self.dense[layer_idx] = dense
            self.load_report.append(report)
        self.model_tensors = self._read_model_level()
        self.load_seconds = round(time.time() - started, 1)

    def _logical_model_keys(self) -> dict[str, str]:
        """Logical name -> checkpoint key, for the tensors outside the layers.

        Mapped to logical names on the way in: `norm.weight` and
        `output_attn_res_norm.weight` both end in "norm.weight", so suffix
        matching would silently pick the wrong one.
        """
        prefix = self.index.model_prefix
        root = self.index.layer_prefix.split("model.layers.")[0]
        logical = {
            "embed": f"{prefix}embed_tokens.weight",
            "final_norm": f"{prefix}norm.weight",
            "out_res_norm": f"{prefix}output_attn_res_norm.weight",
            "out_res_proj": f"{prefix}output_attn_res_proj.weight",
            "lm_head": f"{root}lm_head.weight",
        }
        return {name: key for name, key in logical.items() if key in self.index.weight_map}

    def _read_model_level(self, names: list[str] | None = None) -> dict[str, torch.Tensor]:
        """Read the model-level tensors, optionally only some of them.

        The streamed proxy reads `embed` first and drops it before the trunk,
        then reads the head at the end; on the real K3 those are 2.35 GB each
        and holding both across a 93-layer pass buys nothing.
        """
        logical = self._logical_model_keys()
        if names is not None:
            logical = {name: key for name, key in logical.items() if name in names}
        raw: dict[str, torch.Tensor] = {}
        for shard in self.index.shards_for(logical.values()):
            wanted = [k for k in logical.values() if self.index.weight_map[k] == shard]
            raw.update(read_tensors(os.path.join(self.index.model_dir, shard), wanted))
        return {name: raw[key].to(self.device) for name, key in logical.items()}

    def _mt(self, name: str) -> torch.Tensor:
        if name not in self.model_tensors:
            raise SystemExit(f"model-level tensor {name!r} is not in the index")
        return self.model_tensors[name]

    # -- forward -----------------------------------------------------------
    def reset_states(self) -> None:
        self.states = [new_state(self.index, i) for i in range(self.index.num_layers)]

    def trunk(self, embedded: torch.Tensor, use_state: bool) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_sum = embedded
        block_residual = prefix_sum.new_zeros(
            prefix_sum.shape[0] * prefix_sum.shape[1], 0, prefix_sum.shape[2]
        )
        for layer_idx, runner in enumerate(self.layers):
            prefix_sum, block_residual, _ = runner.forward(
                prefix_sum, block_residual,
                state=self.states[layer_idx] if use_state else None,
                collect=False, dense=self.dense.get(layer_idx),
            )
        return prefix_sum, block_residual

    def head(self, hidden: torch.Tensor, block_residual: torch.Tensor,
             tensors: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """output attn-res read -> final norm -> lm_head."""
        tensors = self.model_tensors if tensors is None else tensors
        eps = float(self.cfg["rms_norm_eps"])
        batch, seq, size = hidden.shape
        hidden = apply_attn_res(
            hidden.view(-1, size), block_residual,
            tensors["out_res_proj"], tensors["out_res_norm"], eps,
        ).view(batch, seq, size)
        hidden = rms_norm(hidden, tensors["final_norm"], eps)
        return torch.nn.functional.linear(hidden, tensors["lm_head"])

    def embed(self, ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        return table[ids.flatten().to(table.device)].reshape(*ids.shape, -1).to(self.device)

    def logits(self, ids: torch.Tensor, use_state: bool = False) -> torch.Tensor:
        embedded = self.embed(ids, self._mt("embed"))
        hidden, block_residual = self.trunk(embedded, use_state)
        return self.head(hidden, block_residual)

    # -- streamed proxy ----------------------------------------------------
    @torch.no_grad()
    def stream_proxy(self, case_ids: list[torch.Tensor]) -> tuple[list[torch.Tensor], dict]:
        """Teacher-forced logits for every case, ONE layer resident at a time.

        The loop is layer-major, not case-major: all cases cross a layer before
        it is dropped, so a layer's packed experts are read once per invocation
        no matter how many cases there are.  What crosses a layer boundary is
        only K3's residual PAIR per case, which is [T, 7168] plus up to eight
        [T, 7168] snapshots -- megabytes, not gigabytes.
        """
        started = time.time()
        table = self._read_model_level(["embed"])["embed"]
        carried = []
        for ids in case_ids:
            prefix_sum = self.embed(ids, table)
            blocks = prefix_sum.new_zeros(
                prefix_sum.shape[0] * prefix_sum.shape[1], 0, prefix_sum.shape[2]
            )
            carried.append((prefix_sum, blocks))
        del table

        for layer_idx in range(self.index.num_layers):
            runner, dense, report = self._load_one_layer(layer_idx)
            for i, (prefix_sum, blocks) in enumerate(carried):
                prefix_sum, blocks, _ = runner.forward(
                    prefix_sum, blocks, state=None, collect=False, dense=dense
                )
                carried[i] = (prefix_sum, blocks)
            self.load_report.append(report)
            del runner, dense
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        tensors = self._read_model_level(["final_norm", "out_res_norm", "out_res_proj", "lm_head"])
        out = [self.head(prefix_sum, blocks, tensors) for prefix_sum, blocks in carried]
        seconds = round(time.time() - started, 1)
        self.load_seconds = seconds
        return out, {
            "mode": "streamed (one layer resident at a time)",
            "layers_streamed": self.index.num_layers,
            "cases_per_layer": len(case_ids),
            "bytes_read": int(sum(r["bytes"] for r in self.load_report)),
            "gigabytes_read": round(sum(r["bytes"] for r in self.load_report) / 1e9, 3),
            "seconds": seconds,
        }

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int, eos_id: int | None) -> dict:
        """Greedy decode.  Artifact-tolerant: loops and a missing EOS are recorded, never scored."""
        self.reset_states()
        started = time.time()
        logits = self.logits(ids, use_state=True)
        produced: list[int] = []
        terminated = "length"
        for _ in range(max_new_tokens):
            token = int(torch.argmax(logits[0, -1]))
            produced.append(token)
            if eos_id is not None and token == eos_id:
                terminated = "eos"
                break
            logits = self.logits(torch.tensor([[token]], dtype=torch.long), use_state=True)
        seconds = time.time() - started
        return {
            "tokens": produced,
            "termination": terminated,
            "seconds": round(seconds, 2),
            "tokens_per_second": round(len(produced) / max(seconds, 1e-9), 3),
        }


# --------------------------------------------------------------------------
def kl_top_agreement(child: torch.Tensor, parent: torch.Tensor) -> dict:
    """KL(parent || child) plus top-1/top-5 agreement, fp32, per position."""
    child = child.float()
    parent = parent.float()
    log_child = torch.log_softmax(child, dim=-1)
    log_parent = torch.log_softmax(parent, dim=-1)
    kl = (log_parent.exp() * (log_parent - log_child)).sum(-1)
    top1 = (child.argmax(-1) == parent.argmax(-1)).float().mean()
    k = min(5, child.shape[-1])
    child_top = child.topk(k, dim=-1).indices.reshape(-1, k)
    parent_top = parent.topk(k, dim=-1).indices.reshape(-1, k)
    top5 = (child_top.unsqueeze(-1) == parent_top.unsqueeze(-2)).any(-1).float().mean()
    return {
        "kl_parent_child_mean": float(kl.mean()),
        "kl_parent_child_max": float(kl.max()),
        "top1_agreement": float(top1),
        "top5_set_agreement": float(top5),
    }


def baseline_provenance(args, plan: dict) -> dict:
    """Refuse a mislabelled baseline, and record which one was actually used.

    Two refusals and one record, all of them from the review's own gap list:

      EMIT  A run with any layer folded may not write `--emit-baseline`.  The
            failure this stops is one forgotten flag: copy the child command
            line, leave `--dense-dir`/`--rung` on, and
            `parent_logits.safetensors` is a rung-65,536 CHILD.
            `_check_requested_folds` is satisfied (folds WERE found), the run
            streams happily, and every later comparison against that file
            reports a far better child than the truth.
      READ  A `--baseline` whose stamp says it was folded is refused for the
            same reason, in the same direction.
      RECORD The consuming run's receipt now carries `baseline_path`,
            `baseline_sha256` and the baseline's own stamp.  Before this, a
            receipt said nothing at all about what it was compared against, so
            the mistake above left no trace anywhere.

    `--allow-folded-baseline` turns both refusals into recorded facts, for the
    deliberate child-vs-child case.  An UNSTAMPED baseline (written before this
    check existed, or by hand) is reported as unstamped and allowed: it cannot
    be proven guilty, and refusing it would strand every earlier receipt.
    """
    folded = plan["layers_folded"]
    record: dict = {
        "this_run_layers_folded": folded,
        "allow_folded_baseline": bool(args.allow_folded_baseline),
    }
    if args.emit_baseline:
        if args.mode != "proxy":
            raise SystemExit(
                f"--emit-baseline only has anything to write in --mode proxy; --mode {args.mode} "
                "would leave the file untouched and the run would look like it had emitted one."
            )
        if folded and not args.allow_folded_baseline:
            raise SystemExit(
                f"--emit-baseline refuses to write a FOLDED run as a baseline: layers {folded} are "
                f"folded at rung {args.rung} from --dense-dir {args.dense_dir!r}.  A child written "
                f"to {args.emit_baseline!r} reads as a parent to every run after it, and the "
                "comparison then reports a far better child than the truth.  Drop --dense-dir/"
                "--rung/--dense-layers for the parent baseline, or pass --allow-folded-baseline "
                "if a child-vs-child baseline is what you actually meant."
            )
        record["emit_baseline_path"] = args.emit_baseline
        record["emit_baseline_is_parent"] = not folded
    if args.baseline:
        stamp = read_metadata(args.baseline)
        record["baseline_path"] = args.baseline
        record["baseline_sha256"] = sha256_file(args.baseline)
        record["baseline_stamp"] = stamp or None
        baseline_folded = json.loads(stamp.get("layers_folded", "[]")) if stamp else None
        record["baseline_layers_folded"] = baseline_folded
        record["baseline_is_stamped"] = bool(stamp)
        if not stamp:
            record["baseline_stamp_note"] = (
                "UNSTAMPED: written before --emit-baseline stamped its output, or by hand.  It "
                "cannot be shown to be a parent; read the comparison accordingly."
            )
            log(f"[warn] baseline {args.baseline} carries no stamp; it cannot be shown to be a parent")
        elif baseline_folded and not args.allow_folded_baseline:
            raise SystemExit(
                f"--baseline {args.baseline!r} was itself produced with layers {baseline_folded} "
                f"folded (rung {stamp.get('fold_rung_requested')}), so it is a CHILD, not a "
                "parent, and every KL/agreement number against it would flatter this run.  "
                "Re-emit the baseline with no folds, or pass --allow-folded-baseline for a "
                "deliberate child-vs-child comparison."
            )
    return record


def load_cases(path: str | None, args) -> list[dict]:
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            cases = json.load(handle)
        return cases if isinstance(cases, list) else cases["cases"]
    generator = torch.Generator().manual_seed(args.seed)
    return [
        {
            "id": f"synthetic-{i}",
            "input_ids": torch.randint(0, args.vocab_size, (args.case_tokens,),
                                       generator=generator, dtype=torch.long).tolist(),
        }
        for i in range(args.cases)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dense-dir", default=None, help="directory of L##_K####_formE.safetensors folds")
    ap.add_argument("--dense-layers", default=None, help="comma list; default = every fold found")
    ap.add_argument("--rung", type=int, default=None,
                    help="pick folds at exactly this REQUESTED rung (the K in the filename); "
                         "a request that matches nothing is refused, not silently ignored")
    ap.add_argument("--mode", default="proxy", choices=("proxy", "decode"))
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--case-file", default=None, help="JSON list of {id, input_ids[, reference_ids]}")
    ap.add_argument("--case-tokens", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--eos-id", type=int, default=None)
    ap.add_argument("--baseline", default=None,
                    help="parent logits safetensors for the proxy comparison; its stamp is read "
                         "back and a baseline that was itself folded is refused")
    ap.add_argument("--emit-baseline", default=None,
                    help="write this run's logits here (make a parent baseline).  Refused while "
                         "any layer is folded, because a child written under this name reads as "
                         "a parent forever after")
    ap.add_argument("--allow-folded-baseline", action="store_true",
                    help="deliberately emit or consume a FOLDED baseline (child-vs-child "
                         "comparisons).  Recorded in the receipt either way")
    ap.add_argument("--vocab-size", type=int, default=163840)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kda-backend", default="torch", choices=("torch", "fla"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    index = K3Index(args.model_dir)
    model = ReviewModel(index, args)
    receipt = {
        "model_dir": args.model_dir,
        "mode": args.mode,
        "dense_dir": args.dense_dir,
        "device": str(canonical_device(args.device)),
        "device_requested": args.device,
        "kda_backend": args.kda_backend,
        "devices": device_inventory(),
        "torch": torch.__version__,
        "instrument_note": (
            "proxy = teacher-forced logit drift, DIAGNOSTIC ONLY, orders layers and rungs, "
            "streamed one layer at a time so an unfolded MoE layer never has to be resident; "
            "decode = the behaviour instrument, resident, and needs every MoE layer folded"
        ),
        "agreement_note": (
            "top1/top5 agreement is computed over EVERY position including the prompt; "
            "vs_parent_continuation_only repeats it on the reference continuation alone"
        ),
    }
    # PLAN FIRST -- nothing below here may read a weight before the refusal has
    # had its chance.  `plan()` touches only the index and the fold directory.
    plan = model.plan()
    receipt.update(plan)

    if args.mode == "decode" and plan["moe_layers_unfolded"]:
        raise SystemExit(
            f"--mode decode needs every MoE layer folded; {plan['moe_layers_unfolded']} are still "
            f"packed (e.g. {plan['unfolded_moe_layers_head'][:5]}).  The MoE parent cannot generate "
            "on this box -- use --mode proxy."
        )

    # --- baseline provenance, decided BEFORE any weight is read ----------
    # `_check_requested_folds` closes the "a fold was asked for and none was
    # found" direction; this closes the other one.  Both refusals happen up
    # here, next to the decode refusal, because on the real model everything
    # below reads 1.3 GB a layer.
    receipt["baseline_provenance"] = baseline_provenance(args, plan)
    cases = load_cases(args.case_file, args)
    results = []
    baseline = None
    if args.baseline:
        from safetensors.torch import load_file

        baseline = load_file(args.baseline)
    emitted: dict[str, torch.Tensor] = {}

    if args.mode == "proxy":
        # teacher-forced ids per case, prompt + reference continuation
        prepared = []
        for case in cases:
            ids = torch.tensor(case["input_ids"], dtype=torch.long).unsqueeze(0)
            entry = {"id": case.get("id", "case"), "prompt_tokens": int(ids.numel())}
            if "reference_ids" in case:
                ids = torch.cat(
                    [ids, torch.tensor(case["reference_ids"], dtype=torch.long).unsqueeze(0)], dim=1
                )
                entry["reference_tokens"] = len(case["reference_ids"])
            prepared.append((entry, ids))
        logits_per_case, stream_report = model.stream_proxy([ids for _, ids in prepared])
        receipt["stream"] = stream_report
        for (entry, ids), logits in zip(prepared, logits_per_case):
            entry["logits_sha256"] = sha256_tensor(logits)
            entry["final_top1"] = int(torch.argmax(logits[0, -1]))
            emitted[entry["id"]] = logits[0].to(torch.float32).cpu()
            if baseline is not None and entry["id"] in baseline:
                reference = baseline[entry["id"]].to(logits.device)
                entry["vs_parent"] = kl_top_agreement(logits[0], reference)
                # scored over EVERY position including the prompt: an ordering
                # diagnostic, not a quality number.  When the case carries a
                # reference continuation the continuation-only slice is scored
                # separately, because that is the part a reader will misread as
                # quality if only one number is printed.
                span = entry.get("reference_tokens")
                if span:
                    entry["vs_parent_continuation_only"] = kl_top_agreement(
                        logits[0, -span:], reference[-span:]
                    )
            results.append(entry)
            log(f"{entry['id']}: {json.dumps({k: v for k, v in entry.items() if k != 'tokens'})[:200]}")
    else:
        model.load()
        receipt["load"] = {"seconds": model.load_seconds, "per_layer": model.load_report}
        with torch.no_grad():
            for case in cases:
                ids = torch.tensor(case["input_ids"], dtype=torch.long).unsqueeze(0)
                entry = {"id": case.get("id", "case"), "prompt_tokens": int(ids.numel())}
                entry.update(model.generate(ids, args.max_new_tokens, args.eos_id))
                entry["transcript_note"] = (
                    "verbatim token ids; loops/echo/cap-rides/missing EOS are artifacts and are NOT content misses"
                )
                results.append(entry)
                log(f"{entry['id']}: {json.dumps({k: v for k, v in entry.items() if k != 'tokens'})[:200]}")

    if args.mode == "proxy":
        receipt["load"] = {"seconds": model.load_seconds, "per_layer": model.load_report}
    if args.emit_baseline and emitted:
        # STAMPED.  A logits file with no stamp cannot say whether it is a
        # parent, and the consuming run has no other way to find out.
        stamp = {
            "produced_by": "k3_review_stream.py",
            "mode": args.mode,
            "model_dir": args.model_dir,
            "layers_folded": json.dumps(plan["layers_folded"]),
            "layers_folded_count": str(len(plan["layers_folded"])),
            "fold_rung_requested": str(args.rung),
            "dense_dir": str(args.dense_dir),
            "is_parent": str(not plan["layers_folded"]),
        }
        receipt["emitted_baseline_stamp"] = stamp
        receipt["baseline_sha256"] = save_tensors(args.emit_baseline, emitted, metadata=stamp)
        receipt["baseline_path"] = args.emit_baseline
    receipt["cases"] = results
    peak = peak_rss_bytes()
    receipt["peak_rss_gigabytes"] = round(peak / 1e9, 3) if peak else None
    receipt["receipt_sha256"] = write_receipt(args.out, receipt)
    log(f"receipt -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
