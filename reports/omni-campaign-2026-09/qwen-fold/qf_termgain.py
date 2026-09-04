"""C9 -- terminal gain.  The cheapest fold-repair in the campaign and the last one run.

A folded child's residual stream is not the parent's, but the reader that
consumes it -- the final RMSNorm's per-dimension gain -- is still the parent's.
C9 re-solves that one vector in closed form so the reader is the CHILD's.

The system, derived rather than fitted.  Let ``Z`` be the child's RMS-normalised
reader input, ``N`` the parent's on the same positions, ``W`` the reader
(``lm_head`` when the head is untied, the embedding when it is tied) and
``C = W^T W``.  Minimising ``|| Z diag(d) W^T - N diag(p) W^T ||_F^2`` in ``d``
gives

    ``[ (Z^T Z) o C ] d = [ (Z^T N) o C ] p``

-- one Hadamard-modified ``H x H`` SPD solve, no iteration, and an exact
identity: at ``Z == N`` the two matrices are the same matrix, so ``d == p``
to machine precision.  That identity is the unit test.

Everything here works in EFFECTIVE gains (``1 + w`` for ``Qwen3_5RMSNorm``) and
writes back through :func:`qf_arch.store_gain`, because a solve that returns a
gain and a parameter that stores ``gain - 1`` is exactly the kind of seam a port
loses a wave to.
"""

from __future__ import annotations

from typing import Any

import torch

import qf_arch
from qf_corpus import fit_check_rows
from qf_solve import LAMBDA_LADDER


def reader_gram(model: torch.nn.Module, which: str, layer_index: int | None = None) -> torch.Tensor:
    """``W^T W`` for the reader that consumes a given norm's output.

    * ``final`` -- the head (or the tied embedding).
    * ``ffn`` at layer i -- the MLP's two readers, stacked.
    * ``operator`` at layer i -- the mixer's readers: q/k/v for a full layer
      (query AND gate rows, since both read the same normed residual), the four
      ``in_proj_*`` for a GDN layer.
    """

    def gram(*tensors: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat([t.detach().to("cpu", torch.float64) for t in tensors], dim=0)
        return stacked.T @ stacked

    if which == "final":
        head = qf_arch.lm_head(model).weight
        return gram(head)
    layer = qf_arch.layers(model)[int(layer_index)]
    if which == "ffn":
        return gram(layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight)
    if which == "operator":
        if qf_arch.layer_kind(layer) == qf_arch.FULL:
            attn = layer.self_attn
            return gram(attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight)
        mixer = layer.linear_attn
        return gram(
            mixer.in_proj_qkv.weight, mixer.in_proj_z.weight,
            mixer.in_proj_b.weight, mixer.in_proj_a.weight,
        )
    raise SystemExit(f"unknown reader site {which!r}")


def solve_gain(
    C: torch.Tensor,
    Z: torch.Tensor,
    N: torch.Tensor,
    prior: torch.Tensor,
    *,
    check: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """The Hadamard normal equations above, with lambda on an internal split.

    ``prior`` and the return value are EFFECTIVE gains.
    """

    left = (Z.T @ Z) * C
    right = ((Z.T @ N) * C) @ prior
    scale = float(torch.diagonal(left).mean())

    def solve(A: torch.Tensor, b: torch.Tensor, rel: float) -> torch.Tensor:
        lam = rel * scale
        M = A + lam * torch.eye(A.shape[0], dtype=A.dtype)
        return torch.linalg.solve(M, b + lam * prior)

    def sse(d: torch.Tensor, rows: torch.Tensor) -> float:
        residual = (Z[rows] * d) - (N[rows] * prior)
        return float(((residual @ C) * residual).sum())

    if check is None:
        # Reported, not assumed: at Z == N the system is exact and lambda is
        # literally zero (that is the identity gate); otherwise the floor rung
        # is used, and the receipt must say the number that ran.
        used = 0.0 if _is_exact(Z, N) else LAMBDA_LADDER[0]
        gain = solve(left, right, used)
        return gain, {
            "lambda_relative": used,
            "selected_on": "all rows (no internal check split)",
            "update_scale": float(
                (gain - prior).norm() / prior.norm().clamp_min(1e-30)
            ),
        }

    inner = ~check
    left_in = (Z[inner].T @ Z[inner]) * C
    right_in = ((Z[inner].T @ N[inner]) * C) @ prior
    rungs = []
    for rel in LAMBDA_LADDER:
        candidate_in = solve(left_in, right_in, rel)
        rungs.append(
            {
                "lambda_relative": rel,
                "check_sse": sse(candidate_in, check),
                "update_scale": float(
                    (solve(left, right, rel) - prior).norm() / prior.norm().clamp_min(1e-30)
                ),
            }
        )
    admitted = [row for row in rungs if row["update_scale"] <= 1.0]
    if not admitted:
        raise SystemExit("no lambda on the ladder admits the gain update; THE LAW refuses")
    chosen = min(admitted, key=lambda row: row["check_sse"])
    gain = solve(left, right, chosen["lambda_relative"])
    return gain, {
        "lambda_relative": chosen["lambda_relative"],
        "selected_on": "internal fit split",
        "update_scale": chosen["update_scale"],
        "ladder": rungs,
    }


def _is_exact(Z: torch.Tensor, N: torch.Tensor) -> bool:
    return Z.shape == N.shape and bool(torch.equal(Z, N))


@torch.no_grad()
def _normalised_reader_input(harness, which: str, layer_index: int | None) -> dict[str, torch.Tensor]:
    """Capture a norm's input and RMS-normalise it exactly as the module does."""

    if which == "final":
        module = harness.final_norm()
    else:
        layer = harness.layers()[int(layer_index)]
        module = layer.post_attention_layernorm if which == "ffn" else layer.input_layernorm
    name = "gain-site"
    captured = harness.capture({name: module})[name]
    eps = harness.eps
    out = {}
    for key in ("fit", "hld"):
        if key not in captured:
            continue
        x = captured[key]
        out[key] = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)
    return out


def terminal_gain(
    child_harness,
    parent_harness,
    *,
    which: str = "final",
    layer_index: int | None = None,
    lambda_split: bool = True,
    log=print,
) -> dict[str, Any]:
    """Re-solve one norm's gain on the child against the parent's own reader output.

    The parent harness must be running the SAME corpus -- the two normalised
    streams are compared position by position and a different corpus makes the
    comparison meaningless rather than merely noisy.
    """

    if child_harness.manifest["manifest_sha256"] != parent_harness.manifest["manifest_sha256"]:
        raise SystemExit(
            "the child and the parent are on different evidence manifests; the "
            "terminal gain compares them position by position"
        )
    Z = _normalised_reader_input(child_harness, which, layer_index)
    N = _normalised_reader_input(parent_harness, which, layer_index)
    if Z["fit"].shape != N["fit"].shape:
        raise SystemExit(
            f"reader input is {tuple(Z['fit'].shape)} on the child and "
            f"{tuple(N['fit'].shape)} on the parent; a width fold needs the "
            f"transported form, not this one"
        )

    if which == "final":
        module = child_harness.final_norm()
        parent_module = parent_harness.final_norm()
    else:
        layer = child_harness.layers()[int(layer_index)]
        parent_layer = parent_harness.layers()[int(layer_index)]
        module = layer.post_attention_layernorm if which == "ffn" else layer.input_layernorm
        parent_module = (
            parent_layer.post_attention_layernorm if which == "ffn"
            else parent_layer.input_layernorm
        )

    C = reader_gram(parent_harness.model, which, layer_index)
    prior = qf_arch.effective_gain(module).detach().to("cpu", torch.float64)
    parent_gain = qf_arch.effective_gain(parent_module).detach().to("cpu", torch.float64)
    if not torch.allclose(prior, parent_gain):
        raise SystemExit(
            "the child's gain at this site is not the parent's; C9 re-solves a "
            "CARRIED gain and this one has already been moved"
        )

    check = fit_check_rows(child_harness.manifest) if lambda_split else None
    gain, guard = solve_gain(C, Z["fit"], N["fit"], prior, check=check)

    def relative_sse(d: torch.Tensor, key: str) -> float:
        target = N[key] * prior
        residual = (Z[key] * d) - target
        num = float(((residual @ C) * residual).sum())
        den = float(((target @ C) * target).sum())
        return (num / max(den, 1e-300)) ** 0.5

    before = {key: relative_sse(prior, key) for key in Z}
    after = {key: relative_sse(gain, key) for key in Z}
    qf_arch.store_gain(module, gain)

    log(
        f"[qf] C9 {which}{'' if layer_index is None else f' L{layer_index:03d}'}: "
        f"reader relSSE {before['fit']:.4f} -> {after['fit']:.4f} (fit)"
    )
    return {
        "axis": "terminal_gain",
        "kind": "closed-form gain re-solve in the reader's own metric",
        "site": which,
        "layer": layer_index,
        "solve": guard,
        "reader_relative_sse_before": before,
        "reader_relative_sse_after": after,
        "gain_update_scale": float((gain - prior).norm() / prior.norm().clamp_min(1e-30)),
        "parameters_removed": 0,
        "note": "C9 moves values, never shapes; it is a repair, never a compression",
    }
