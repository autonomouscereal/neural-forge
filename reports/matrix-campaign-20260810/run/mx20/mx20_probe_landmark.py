#!/usr/bin/env python3
"""F-LANDMARK -- verbatim-state landmark memory for attn_k/attn_v, slate §A8.

**The claim under test.**  The deepest sealed insight in this campaign is that
VERBATIM parent rows survive folds that synthesised rows do not -- the token head
and the SwiGLU channel coreset both live on it.  Every instance so far has been
verbatim rows of a *table*.  F-LANDMARK asks whether the same law holds for
verbatim *states*: replace a KV group's key and value projections with m = 256
ACTUAL cached key/value vectors, taken unmodified from the parent's own run, and
let the converted head address them by content:

    y_h = softmax(beta * q_h M_k^T) M_v            (q_h with NO RoPE)

Storage per converted group is ``2 * 256 * 64 = 32,768`` against the parent's
``2 * 64 * 1024 = 131,072`` -- **4.0x exactly**, with no index storage because
the rows are copied into parameters and the betas counted.

**Status: PROBE, research-only, and this one is not arguable.**  There is no
config that expresses a landmark memory, `Lfm2ForCausalLM.from_pretrained`
cannot produce this model, and R3 makes custom runtime research-only by
construction.  The frozen review still runs, through
`mx20_probe_review.py`, against the same protocol digest and the same sixteen
cases, because what the class is being asked is behavioural.

**One deviation from the spec, stated rather than smuggled.**  §A8 selects the
landmarks by attended-mass-weighted pivoted QR on ``diag(sqrt(w)) K_h``.  That
matrix is ``N x 64``: its rank is at most 64, so pivoted QR is exhausted after 64
picks and the remaining 192 are chosen on a numerically empty residual.  The
selection here is therefore **the QR pivots while they are meaningful, then the
highest attended mass among the rows not yet taken** -- which is the same
weighting the spec names, applied by the rule that still has content at m = 256.
The count of each is in the receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mx20_build as mx  # noqa: E402

SCHEMA = "neural-forge-matrix-wave20-landmark-probe-v1"
LANDMARKS = 256
#: KV groups kept fully verbatim -- the ones their own landmark set reconstructs
#: WORST, per §A8 step 6.  Converting the best-reconstructed groups first is the
#: pre-registered order and it is what makes the kept pair a control.
KEEP_GROUPS = 2
#: The beta grid, deterministic and pinned, selected on a fit-side split.
BETA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


class LandmarkAttention(nn.Module):
    """The parent's attention block with some KV groups replaced by a memory.

    Kept groups run the parent's own path -- RoPE, causal mask, KV cache.
    Converted groups run content addressing over fixed landmarks, need no cache
    at all (their keys and values do not depend on the sequence), and drop RoPE
    on the query because a landmark has no position to rotate against.
    """

    def __init__(self, parent_attn: Any, converted: list[int], kept: list[int]) -> None:
        super().__init__()
        self.head_dim = int(parent_attn.head_dim)
        self.scaling = float(parent_attn.scaling)
        self.num_key_value_groups = int(parent_attn.num_key_value_groups)
        self.layer_idx = int(parent_attn.layer_idx)
        self.converted = list(converted)
        self.kept = list(kept)
        self.q_proj = parent_attn.q_proj
        self.out_proj = parent_attn.out_proj
        self.q_layernorm = parent_attn.q_layernorm
        self.k_layernorm = parent_attn.k_layernorm
        # Kept groups keep their projections; the converted groups' rows are
        # GONE -- the probe stores landmarks instead, which is the whole 4x.
        device = parent_attn.k_proj.weight.device
        dtype = parent_attn.k_proj.weight.dtype
        rows = torch.cat(
            [
                torch.arange(g * self.head_dim, (g + 1) * self.head_dim)
                for g in self.kept
            ]
        ).to(device)
        self.k_proj = nn.Linear(
            parent_attn.k_proj.in_features, len(rows), bias=False, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            parent_attn.v_proj.in_features, len(rows), bias=False, device=device, dtype=dtype
        )
        self.k_proj.weight.data.copy_(parent_attn.k_proj.weight.data[rows])
        self.v_proj.weight.data.copy_(parent_attn.v_proj.weight.data[rows])
        self.register_buffer(
            "landmark_k",
            torch.zeros(len(converted), LANDMARKS, self.head_dim, device=device, dtype=dtype),
        )
        self.register_buffer(
            "landmark_v",
            torch.zeros(len(converted), LANDMARKS, self.head_dim, device=device, dtype=dtype),
        )
        self.register_buffer(
            "beta", torch.ones(len(converted), device=device, dtype=torch.float32)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb

        shape = hidden_states.shape[:-1]
        hidden_shape = (*shape, -1, self.head_dim)
        q = self.q_layernorm(self.q_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q_rope, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        heads = q.shape[1]
        per_head = torch.zeros_like(q)
        # ---- the kept groups: the parent's own attention, unchanged.
        if self.kept:
            kept_heads = [
                h
                for h in range(heads)
                if (h // self.num_key_value_groups) in self.kept
            ]
            index = {g: i for i, g in enumerate(self.kept)}
            k_rep = torch.stack(
                [k[:, index[h // self.num_key_value_groups]] for h in kept_heads], dim=1
            )
            v_rep = torch.stack(
                [v[:, index[h // self.num_key_value_groups]] for h in kept_heads], dim=1
            )
            scores = torch.matmul(q_rope[:, kept_heads], k_rep.transpose(-1, -2)) * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask[..., : scores.shape[-1]].to(scores.dtype)
            probabilities = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            per_head[:, kept_heads] = torch.matmul(probabilities, v_rep)
        # ---- the converted groups: content addressing over fixed landmarks.
        for slot, group in enumerate(self.converted):
            group_heads = [
                h
                for h in range(heads)
                if (h // self.num_key_value_groups) == group
            ]
            memory_k = self.landmark_k[slot]
            memory_v = self.landmark_v[slot]
            scores = torch.matmul(q[:, group_heads], memory_k.T.unsqueeze(0)) * float(
                self.beta[slot]
            )
            probabilities = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            per_head[:, group_heads] = torch.matmul(probabilities, memory_v)

        output = per_head.transpose(1, 2).reshape(*shape, -1).contiguous()
        return self.out_proj(output), None


def weighted_landmarks(
    keys: torch.Tensor, mass: torch.Tensor, count: int
) -> tuple[torch.Tensor, dict[str, int]]:
    """Attended-mass-weighted pivoted QR while it has content, then top mass.

    ``keys`` is (N, head_dim); ``mass`` is the attended mass of each source row.
    Returns the selected row indices and how many came from each rule.
    """

    weighted = keys * mass.clamp_min(0.0).sqrt()[:, None]
    # `torch.linalg.qr` has no pivoting, so the pivots are computed by the greedy
    # rule QR with column pivoting IS -- take the column with the largest
    # residual norm, project it out, repeat.  Deterministic, fp64, CPU, which is
    # wave 15's portability lesson applied to a selection.
    residual = weighted.T.clone()
    taken = torch.zeros(residual.shape[1], dtype=torch.bool)
    order: list[int] = []
    limit = min(int(keys.shape[1]), count)
    for _ in range(limit):
        norms = residual.pow(2).sum(dim=0)
        norms[taken] = -1.0
        pick = int(norms.argmax())
        if float(norms[pick]) <= 1e-24:
            break
        order.append(pick)
        taken[pick] = True
        column = residual[:, pick]
        column = column / column.norm()
        residual = residual - column[:, None] * (column[None, :] @ residual)
    qr_picks = len(order)
    for index in torch.argsort(mass, descending=True).tolist():
        if len(order) >= count:
            break
        if not bool(taken[index]):
            order.append(index)
            taken[index] = True
    return torch.tensor(sorted(order), dtype=torch.long), {
        "qr_pivots": qr_picks,
        "mass_filled": len(order) - qr_picks,
        "requested": count,
    }


def install(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_grad_enabled(False)
    started = time.time()
    parent = mx.Harness(mx.PARENT, args.device, torch.bfloat16)
    config = json.loads((mx.PARENT / "config.json").read_text())
    if config["layer_types"][args.layer] != "full_attention":
        raise SystemExit(f"L{args.layer:02d} is not an attention block")
    block = parent.layers()[args.layer]
    attn = block.self_attn
    head_dim, groups = int(attn.head_dim), int(config["num_key_value_heads"])
    per_group = int(attn.num_key_value_groups)

    # ---- the evidence: the block's own input, its pre-RoPE keys and values, and
    # the attention mass the parent puts on each source row.
    mx.log(f"capturing the attention block's input at L{args.layer:02d}")
    captured = parent.capture({"a": attn})
    X_fit, X_hld = captured["a"]["fit"], captured["a"]["hld"]
    kw = attn.k_proj.weight.data.to(torch.float64).cpu()
    vw = attn.v_proj.weight.data.to(torch.float64).cpu()
    kgain = attn.k_layernorm.weight.data.to(torch.float64).cpu()
    eps = float(config["norm_eps"])

    def per_group_kv(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = (X @ kw.T).reshape(X.shape[0], groups, head_dim)
        k = k * k.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt() * kgain
        v = (X @ vw.T).reshape(X.shape[0], groups, head_dim)
        return k, v

    K_fit, V_fit = per_group_kv(X_fit)

    mx.log("recomputing the parent's own attention mass per source row")
    mass = attention_mass(parent, args.layer, groups, per_group)

    # ---- selection, then the pre-registered keep/convert order.
    selections: dict[int, Any] = {}
    residuals: list[tuple[float, int]] = []
    for g in range(groups):
        picked, how = weighted_landmarks(K_fit[:, g], mass[g], LANDMARKS)
        memory = K_fit[picked, g]
        # How well does this group's own landmark set explain its keys?  The
        # worst-explained groups are the ones §A8 keeps verbatim.
        solution = torch.linalg.lstsq(memory.T, K_fit[:, g].T).solution
        error = float(
            (K_fit[:, g].T - memory.T @ solution).norm() / K_fit[:, g].norm()
        )
        selections[g] = {"rows": picked, "how": how, "reconstruction_relL2": error}
        residuals.append((error, g))
    residuals.sort(reverse=True)
    kept = sorted(g for _, g in residuals[:KEEP_GROUPS])
    converted = sorted(g for g in range(groups) if g not in kept)
    mx.log(
        f"keeping groups {kept} verbatim (worst reconstructed), converting {converted}"
    )

    # ---- beta, on an internal fit-side split, against the parent's own per-head
    # attention output for the converted heads.
    check = mx.fit_check_rows(parent.batches, parent.split)
    # The parent's own out_proj rows, captured ONCE while the block is still the
    # parent's: per head they are beta's target, and pushed through the parent's
    # out_proj they are the block output the refit has to reproduce.
    parent_o = parent.capture({"o": attn.out_proj})
    ow = attn.out_proj.weight.data.to(torch.float64).cpu()
    parent_block_out = {
        "fit": parent_o["o"]["fit"] @ ow.T,
        "hld": parent_o["o"]["hld"] @ ow.T,
    }
    parent_out = [
        parent_o["o"]["fit"][:, h * head_dim : (h + 1) * head_dim].contiguous()
        for h in range(groups * per_group)
    ]
    betas: dict[int, float] = {}
    beta_report: dict[str, Any] = {}
    for g in converted:
        rows = selections[g]["rows"]
        memory_k, memory_v = K_fit[rows, g], V_fit[rows, g]
        heads = [h for h in range(groups * per_group) if h // per_group == g]
        best: tuple[float, float] | None = None
        ladder = []
        for beta in BETA_GRID:
            total = 0.0
            for h in heads:
                q = parent_query(parent, args.layer, X_fit, h, head_dim)
                pred = torch.softmax(beta * (q @ memory_k.T), dim=-1) @ memory_v
                total += float((pred[check] - parent_out[h][check]).pow(2).sum())
            ladder.append({"beta": beta, "check_sse": total})
            if best is None or total < best[0]:
                best = (total, beta)
        betas[g] = float(best[1])
        beta_report[str(g)] = {"chosen": best[1], "ladder": ladder}
    mx.log(f"betas chosen on the internal split: {betas}")

    # ---- install, then re-solve out_proj in place against the parent's block.
    module = LandmarkAttention(attn, converted, kept).to(args.device)
    for slot, g in enumerate(converted):
        rows = selections[g]["rows"]
        module.landmark_k[slot] = K_fit[rows, g].to(module.landmark_k.dtype).to(args.device)
        module.landmark_v[slot] = V_fit[rows, g].to(module.landmark_v.dtype).to(args.device)
        module.beta[slot] = betas[g]
    block.self_attn = module

    context = parent.capture({"o": module.out_proj})
    prior = module.out_proj.weight.data.to(torch.float64).cpu()
    prior_scores = mx.write_scores(context["o"]["hld"] @ prior.T, parent_block_out["hld"])
    solved, guard = mx.law_abiding_solve_split(
        context["o"]["fit"], parent_block_out["fit"], check, prior
    )
    module.out_proj.weight.data.copy_(
        solved.to(device=module.out_proj.weight.device, dtype=module.out_proj.weight.dtype)
    )
    solved_scores = mx.write_scores(context["o"]["hld"] @ solved.T, parent_block_out["hld"])
    mx.log(
        f"out_proj re-solved in place: held-out cos {prior_scores['cos']:.4f} -> "
        f"{solved_scores['cos']:.4f}, update_scale {guard['update_scale']:.3f}"
    )

    outdir = Path(args.state)
    outdir.mkdir(parents=True, exist_ok=True)
    state = {
        "landmark_k": module.landmark_k.detach().cpu(),
        "landmark_v": module.landmark_v.detach().cpu(),
        "beta": module.beta.detach().cpu(),
        "k_proj": module.k_proj.weight.detach().cpu(),
        "v_proj": module.v_proj.weight.detach().cpu(),
        "out_proj": module.out_proj.weight.detach().cpu(),
    }
    from safetensors.torch import save_file

    save_file(
        {k: v.contiguous() for k, v in state.items()},
        str(outdir / "landmark-state.safetensors"),
        metadata={"schema": SCHEMA},
    )
    parent_group_numbers = 2 * head_dim * int(config["hidden_size"])
    child_group_numbers = 2 * LANDMARKS * head_dim
    receipt = {
        "schema": SCHEMA,
        "class": "F-LANDMARK",
        "spec": "docs/FOLD_CLASS_CANDIDATES_2026-08-12.md section A8",
        "status": "probe",
        "export_class": "custom-runtime",
        "research_only": True,
        "research_only_reason": (
            "a landmark memory has no config that expresses it and "
            "Lfm2ForCausalLM.from_pretrained cannot produce this model; R3 makes "
            "custom runtime research-only"
        ),
        "site": {
            "component": "C07+C08 attn_k/attn_v, with C09 refit in place",
            "layer": args.layer,
            "kv_groups": groups,
            "kept_verbatim": kept,
            "converted": converted,
            "landmarks": LANDMARKS,
            "rope_on_converted_queries": False,
        },
        "selection": {
            str(g): {
                "qr_pivots": selections[g]["how"]["qr_pivots"],
                "mass_filled": selections[g]["how"]["mass_filled"],
                "reconstruction_relL2": selections[g]["reconstruction_relL2"],
            }
            for g in range(groups)
        },
        "beta": beta_report,
        "storage": {
            "parent_numbers_per_group": parent_group_numbers,
            "child_numbers_per_group": child_group_numbers,
            "ratio_per_converted_group": parent_group_numbers / child_group_numbers,
            "converted_groups": len(converted),
            "beta_scalars": len(converted),
            "parameters_removed": len(converted) * (parent_group_numbers - child_group_numbers),
        },
        "out_proj_refit": {
            "prior_only": prior_scores,
            "solved": solved_scores,
            "guard": {k: v for k, v in guard.items() if k != "ladder"},
        },
        "rows": {"fit": int(X_fit.shape[0]), "held_out": int(X_hld.shape[0])},
        "parent_forward_calls": parent.forward_calls,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_seconds": round(time.time() - started, 1),
        "state_sha256": mx.file_sha256(outdir / "landmark-state.safetensors"),
    }
    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(json.dumps(receipt, indent=1))
    mx.log(f"F-LANDMARK probe state -> {outdir}")
    return receipt


def attention_mass(
    parent: mx.Harness, layer: int, groups: int, per_group: int
) -> torch.Tensor:
    """Per-KV-group attended mass on each source row, from the parent's own run.

    This architecture's eager path returns its attention weights, so no
    recomputation and no second capture is needed -- wave 18's gate probe
    established the same access.
    """

    store: list[torch.Tensor] = []

    def hook(module, args_, kwargs_, output):
        weights = output[1]
        if weights is None:
            raise SystemExit("the eager attention path returned no weights")
        store.append(weights.detach().to("cpu", torch.float64))

    handle = parent.layers()[layer].self_attn.register_forward_hook(
        hook, with_kwargs=True
    )
    try:
        for _index, ids in parent.batches:
            parent.model(input_ids=ids.to(parent.device), use_cache=False)
            parent.forward_calls += 1
    finally:
        handle.remove()
    per_batch = []
    for weights in store:
        # (B, heads, q, kv) -> mass each source row receives, per KV group.
        mass = weights.sum(dim=2)[0]
        per_batch.append(
            torch.stack(
                [mass[g * per_group : (g + 1) * per_group].sum(dim=0) for g in range(groups)]
            )
        )
    return torch.cat(per_batch, dim=1)


_QUERY_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def parent_query(
    parent: mx.Harness, layer: int, X: torch.Tensor, head: int, head_dim: int
) -> torch.Tensor:
    """The parent's post-layernorm query rows for one head, WITHOUT RoPE.

    Computed once for every head on the first call and cached: the beta grid asks
    for them fourteen times per group and they do not change between asks.
    """

    key = (layer, head)
    if key not in _QUERY_CACHE:
        attn = parent.layers()[layer].self_attn
        qw = attn.q_proj.weight.data.to(torch.float64).cpu()
        gain = attn.q_layernorm.weight.data.to(torch.float64).cpu()
        eps = float(parent.model.config.norm_eps)
        rows = (X @ qw.T).reshape(X.shape[0], -1, head_dim)
        rows = rows * rows.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt() * gain
        for h in range(rows.shape[1]):
            _QUERY_CACHE[(layer, h)] = rows[:, h].contiguous()
    return _QUERY_CACHE[key]


def load_probe(
    state: Path, *, parent: Path, device: str, dtype: Any, receipt: dict[str, Any]
) -> Any:
    """Rebuild the probe model for review: the parent with one block replaced."""

    from safetensors.torch import load_file
    from transformers import Lfm2ForCausalLM

    observed = mx.file_sha256(Path(state) / "landmark-state.safetensors")
    if observed != str(receipt["state_sha256"]):
        raise SystemExit(
            f"{state} digests {observed}; the probe receipt sealed "
            f"{receipt['state_sha256']}"
        )
    model = Lfm2ForCausalLM.from_pretrained(str(parent), dtype=dtype).to(device).eval()
    layer = int(receipt["site"]["layer"])
    tensors = load_file(str(Path(state) / "landmark-state.safetensors"))
    module = LandmarkAttention(
        model.model.layers[layer].self_attn,
        list(receipt["site"]["converted"]),
        list(receipt["site"]["kept_verbatim"]),
    ).to(device)
    module.landmark_k.copy_(tensors["landmark_k"].to(device=device, dtype=module.landmark_k.dtype))
    module.landmark_v.copy_(tensors["landmark_v"].to(device=device, dtype=module.landmark_v.dtype))
    module.beta.copy_(tensors["beta"].to(device))
    module.k_proj.weight.data.copy_(tensors["k_proj"].to(device=device, dtype=dtype))
    module.v_proj.weight.data.copy_(tensors["v_proj"].to(device=device, dtype=dtype))
    module.out_proj.weight.data.copy_(tensors["out_proj"].to(device=device, dtype=dtype))
    model.model.layers[layer].self_attn = module
    mx.log(
        f"F-LANDMARK probe rebuilt at L{layer:02d}: "
        f"{len(receipt['site']['converted'])} groups converted, custom runtime"
    )
    return model.eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    install(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
