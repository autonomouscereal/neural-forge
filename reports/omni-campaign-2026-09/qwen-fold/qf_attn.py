"""C6 NoPE-dimension contraction and C7 kv-head merge on the gated GQA layers.

A Qwen3.5 attention head is 256 coordinates of which only the first 64 rotate
(``partial_rotary_factor`` 0.25; ``apply_rotary_pos_emb`` splits at
``rotary_dim`` and concatenates the rest back untouched, modeling 570-605).  The
192 NoPE coordinates contribute a PLAIN bilinear form ``q_n^T k_n`` to every
score -- no position dependence, no plane structure, nothing that forbids a
two-sided low-rank contraction of them.  That is C6, and it has no LFM2
ancestor at all, because LFM2's heads were all-rotary and its ``head_dim`` axis
died on the plane arithmetic that is simply absent here.

C7 merges kv heads.  Two arms:

* **kv-only** -- ``num_key_value_heads`` shrinks, ``num_attention_heads`` does
  not.  Every query head keeps its own gate rows, so the gate is EXACT and the
  whole cost of the transaction sits in the key relation and the value
  contraction.  This is the KV-cache lever: each merge divides the cache.
* **full-group** -- query heads and their gates merge too.  Cheaper in
  parameters; the gate rows must be averaged in the value frame, which
  ``sigmoid`` cannot be pushed through (``gate_frame_mismatch``).

Both classes install first and solve ``o_proj`` second, on the child's own
post-gate stream, because the gate is part of the writer's design matrix and
approximating it would put an avoidable approximation inside a solve that has an
exact form available.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import qf_arch
from qf_arch import FULL
from qf_corpus import fit_check_rows
from qf_harness import pristine_copy
from qf_solve import (
    gated_out_proj_solve,
    joint_relation_solve,
    matrix_sqrt,
    project_writer,
    whitened_lra,
    write_scores,
)


def attention_slices(attn: nn.Module, config: Any) -> dict[str, Any]:
    """Row and column blocks of a ``Qwen3_5Attention``.

    ``q_proj`` is ``[num_heads * head_dim * 2, H]`` and the two halves are
    INTERLEAVED PER HEAD, not stacked: ``forward`` views it as
    ``(..., -1, head_dim*2)`` and chunks the last axis (modeling 683-685), so
    head ``h`` owns rows ``[h*2d, h*2d+d)`` = QUERY and ``[h*2d+d, h*2d+2d)`` =
    GATE.  Reading it as "all queries, then all gates" is the easiest possible
    way to build a child that runs and is wrong.
    """

    tc = qf_arch.text_config(config)
    d = int(attn.head_dim)
    nh = int(tc.num_attention_heads)
    nkv = int(tc.num_key_value_heads)
    return {
        "head_dim": d,
        "heads": nh,
        "kv_heads": nkv,
        "group": nh // nkv,
        "q_rows": lambda h: torch.arange(h * 2 * d, h * 2 * d + d),
        "gate_rows": lambda h: torch.arange(h * 2 * d + d, h * 2 * d + 2 * d),
        "k_rows": lambda m: torch.arange(m * d, (m + 1) * d),
        "v_rows": lambda m: torch.arange(m * d, (m + 1) * d),
        "o_cols": lambda h: torch.arange(h * d, (h + 1) * d),
    }


def _rms(values: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(values.pow(2).mean(dim=1, keepdim=True) + eps)


def _post_norm(X: torch.Tensor, rows: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """What the score actually sees: ``(X W^T / rms) * gain``."""

    raw = X @ rows.T
    return (raw / _rms(raw, eps)) * gain.unsqueeze(0)


def _second_moment(values: torch.Tensor) -> torch.Tensor:
    return (values.T @ values) / max(values.shape[0], 1)


def install_attention(
    attn: nn.Module,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_gain: torch.Tensor,
    k_gain: torch.Tensor,
    head_dim: int,
    heads: int,
    kv_heads: int,
    dtype: torch.dtype,
) -> None:
    """Replace every attention tensor and every cached shape attribute together.

    ``head_dim``, ``num_key_value_groups`` and ``scaling`` are read off the
    MODULE inside ``forward`` (modeling 652-654, 681, 709), not off the config,
    so writing weights without writing these leaves a module whose reshapes
    belong to the parent.

    ``scaling`` is written here as ``head_dim ** -0.5`` because that is what the
    stock class recomputes from the child's own config on re-load (modeling 654)
    -- a child whose module scaling disagreed with its config would measure one
    thing here and another after export.  The consequence is that CHANGING
    ``head_dim`` CHANGES THE SOFTMAX TEMPERATURE: a caller that preserves the
    score ``q^T k`` and does nothing else ships a child whose logits are the
    parent's times ``sqrt(parent_head_dim/child_head_dim)``.  That is not
    absorbable downstream -- ``o_proj`` sits after the softmax -- so any caller
    that moves ``head_dim`` must fold ``sqrt(child_head_dim/parent_head_dim)``
    into ``q_gain``, which is exact because the ``q_norm`` gain is a free
    per-dimension weight.  C6 does; C7 does not move ``head_dim`` at all.
    """

    device = attn.q_proj.weight.device
    hidden = int(attn.q_proj.weight.shape[1])

    def linear(rows: torch.Tensor) -> nn.Linear:
        layer = nn.Linear(hidden, rows.shape[0], bias=False, dtype=dtype, device=device)
        layer.weight.data.copy_(rows.to(device=device, dtype=dtype))
        return layer

    attn.q_proj, attn.k_proj, attn.v_proj = linear(q), linear(k), linear(v)
    attn.o_proj = nn.Linear(heads * head_dim, hidden, bias=False, dtype=dtype, device=device)
    attn.o_proj.weight.data.copy_(o.to(device=device, dtype=dtype))

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    if int(attn.q_norm.weight.shape[0]) != head_dim:
        eps = float(attn.q_norm.eps)
        attn.q_norm = Qwen3_5RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
        attn.k_norm = Qwen3_5RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    qf_arch.store_gain(attn.q_norm, q_gain)
    qf_arch.store_gain(attn.k_norm, k_gain)

    attn.head_dim = head_dim
    attn.scaling = head_dim ** -0.5
    attn.num_key_value_groups = heads // kv_heads


#: The largest fraction of a site's captured rows that may be non-finite and
#: still be handled by dropping them.  Above this line the child is not
#: producing numbers on its own evidence at all, and a writer fitted on what
#: survives would be a fit to a MINORITY of the stream recorded as a fit to the
#: whole of it -- so the fold refuses instead, with the census.  Proposed
#: decision line, recorded either way; it is not a grade.
NONFINITE_ROW_CEILING = 0.02


def unfold_gain(scaled, gain, parent_rows, *, record=None, where=""):
    """Divide an RMS-norm gain back out of solved rows -- EXCEPT where it is zero.

    C7 solves the key/query relation on GAIN-SCALED rows, because
    ``Qwen3_5RMSNorm.forward`` multiplies by ``1.0 + weight`` (modeling: ``output
    = self._norm(x.float()) * (1.0 + self.weight.float())``) and that factor is
    what the softmax sees.  The solved rows are then divided by the same gain
    before installation, which round-trips exactly -- until a coordinate's gain
    is exactly zero.

    **Qwen3.5-4B layer 27's ``q_norm`` has one: coordinate 1, whose bf16 weight
    is exactly -1.0, so ``1 + w`` is exactly 0.**  No other full-attention layer
    in that parent has one, which is why the K2KV build folded layers 3 through
    23 and died on 27.  Coordinate 1 is the REAL half of rotary plane 1 (paired
    with coordinate 33), so the scaled row is zero but the plane is not, and the
    complex coefficient :func:`qf_solve.joint_relation_solve` multiplies the
    plane by leaks a NON-zero real part straight back into that row.  Dividing
    it by zero installs +-Inf into 16 rows of ``q_proj``; ``q_norm``'s own
    per-token RMS then divides the whole head by Inf, ``Inf * 0`` is NaN, and
    every number after it -- the logits, the softmax, the gated writer stream,
    its Gram, and the ridge computed from that Gram's diagonal mean -- is NaN.
    ``torch.linalg.cholesky`` reports that as "the leading minor of order 1 is
    not positive-definite", which is a true statement about a matrix of NaNs and
    says nothing about where they came from.

    What to install there instead is not a free choice and it is not zero.  A
    zero-gain coordinate contributes nothing to any score -- the child's
    ``q_norm`` multiplies it by zero exactly as the parent's did -- but it DOES
    contribute its square to the per-token RMS that the norm divides the whole
    head by.  Writing zero would change that RMS; writing the row the PARENT had
    there preserves it, which is the only property of that coordinate the model
    ever uses.  So that is what goes in, and ``zero_gain_coordinates`` in the
    layer's report says which coordinates it happened to and why.
    """

    live = gain != 0
    divisor = torch.where(live, gain, torch.ones_like(gain))
    rows = scaled / divisor.unsqueeze(1)
    dead = ~live
    count = int(dead.sum())
    if count:
        rows = torch.where(dead.unsqueeze(1), parent_rows, rows)
        if record is not None:
            record.append(
                {
                    "where": where,
                    "coordinates": torch.nonzero(dead).flatten().tolist(),
                    "count": count,
                    "gain_convention": "effective gain is 1 + weight (Qwen3_5RMSNorm)",
                    "installed": (
                        "the parent's own raw row, not the divided one and not "
                        "zero: the norm multiplies this coordinate by zero either "
                        "way, and the parent's value is what its per-token RMS saw"
                    ),
                    "would_have_been": (
                        "+-Inf from a division by zero, which q_norm's RMS turns "
                        "into NaN across the entire head"
                    ),
                }
            )
    return rows


def _finite_rows(*blocks) -> torch.Tensor:
    """Rows that are finite in EVERY supplied block."""

    mask = None
    for block in blocks:
        if block is None:
            continue
        finite = torch.isfinite(block).all(dim=1)
        mask = finite if mask is None else (mask & finite)
    return mask


def _writer_stream_repair(name, A, Y, check, *, log, where):
    """Keep the captured rows that are numbers; name and count the ones that are not.

    The writer's design matrix is the child's OWN forward, in bf16, on the GPU.
    A fold that moves a key or a value frame can put a token's attention output
    outside what the rest of that bf16 arithmetic can represent, and once one
    element of one row is NaN or Inf, ``A^T A`` spreads it over EVERY entry of
    the Gram -- the diagonal mean goes non-finite, the ridge goes with it, and
    ``linalg.cholesky`` reports "the leading minor of order 1 is not
    positive-definite", which names neither the row nor the reason.  That is how
    the K2KV build died at full-attention layer 27.

    A ridge cannot repair a row defect, so it is repaired here, where the rows
    are: the offending rows are REMOVED from the normal equations and counted,
    with the channels they touched, into the receipt.  Nothing is clamped,
    imputed or rescaled -- a clamped NaN is a fabricated observation -- and the
    fit/held scores are computed on the same kept rows, so the number in the
    receipt is a number about the rows the receipt says it is about.
    """

    total = int(A.shape[0])
    keep = _finite_rows(A, Y)
    dropped = total - int(keep.sum())
    record = {
        "where": where,
        "rows_captured": total,
        "rows_nonfinite": dropped,
        "repaired": bool(dropped),
        "policy": (
            "rows the child's own bf16 forward did not produce a number for are "
            "REMOVED from the normal equations and counted here; nothing is "
            "clamped, imputed or rescaled, and every score reported for this "
            "site is computed on the kept rows"
        ),
    }
    if not dropped:
        return A, Y, check, record

    bad_design = ~torch.isfinite(A)
    columns = torch.nonzero(bad_design.any(dim=0)).flatten()
    record.update(
        {
            "fraction_nonfinite": dropped / max(total, 1),
            "ceiling": NONFINITE_ROW_CEILING,
            "design_nan": int(torch.isnan(A).sum()),
            "design_posinf": int(torch.isposinf(A).sum()),
            "design_neginf": int(torch.isneginf(A).sum()),
            "target_nonfinite": int((~torch.isfinite(Y)).sum()),
            "design_channels_touched": int(columns.numel()),
            "design_channels_first": columns[:16].tolist(),
            "first_rows": torch.nonzero(~keep).flatten()[:8].tolist(),
        }
    )
    log(
        f"[qf] {name}: {dropped}/{total} captured {where} rows are not finite "
        f"({record['fraction_nonfinite']:.4%}, {record['design_channels_touched']} "
        f"channels touched); dropped from the normal equations and recorded"
    )
    if dropped >= total:
        raise SystemExit(
            f"{name}: every one of the {total} captured {where} rows is "
            f"non-finite; the child produces no numbers on its own evidence at "
            f"this site and there is nothing to solve against"
        )
    if record["fraction_nonfinite"] > NONFINITE_ROW_CEILING:
        raise SystemExit(
            f"{name}: {dropped}/{total} ({record['fraction_nonfinite']:.4%}) of the "
            f"captured {where} rows are non-finite, above the "
            f"{NONFINITE_ROW_CEILING:.2%} ceiling; channels touched "
            f"{record['design_channels_first']}.  Dropping this many rows would "
            f"report a fit to a minority of the stream as a fit to all of it -- "
            f"the fold that produced these activations is what needs fixing"
        )
    A, Y = A[keep], Y[keep]
    record["rows_kept"] = int(A.shape[0])
    record["design_absmax_kept"] = float(A.abs().max()) if A.numel() else None
    if check is not None:
        check = check[keep]
        record["internal_check_rows_after"] = int(check.sum())
        if not bool(check.any()) or bool(check.all()):
            raise SystemExit(
                f"{name}: dropping {dropped} non-finite rows collapses the "
                f"internal fit/check split to {int(check.sum())} of "
                f"{check.numel()} rows; the lambda ladder would be selected on "
                f"nothing"
            )
    return A, Y, check, record


def _resolve_writer(harness, attn, name, Y, Y_hld, lambda_split, dtype, log=print):
    """Capture the child's own post-gate stream and re-solve ``o_proj`` on it."""

    stream = harness.capture({name: attn.o_proj})[name]
    prior = attn.o_proj.weight.detach().to("cpu", torch.float64)
    check = fit_check_rows(harness.manifest) if lambda_split else None
    fit, Y_fit, check, repair = _writer_stream_repair(
        name, stream["fit"], Y, check, log=log, where="fit"
    )
    writer, guard = gated_out_proj_solve(fit, Y_fit, prior, check)
    guard["stream_repair"] = repair
    attn.o_proj.weight.data.copy_(writer.to(device=harness.device, dtype=dtype))
    scores = {"fit": write_scores(fit @ writer.T, Y_fit)}
    if "hld" in stream and Y_hld is not None:
        held, Y_held, _, held_repair = _writer_stream_repair(
            name, stream["hld"], Y_hld, None, log=log, where="held"
        )
        scores["held"] = write_scores(held @ writer.T, Y_held)
        guard["stream_repair_held"] = held_repair
    scores["no_repair"] = write_scores(fit @ prior.T, Y_fit)
    return guard, scores


# ---------------------------------------------------------------- C6: NoPE dim
def nope_dim_contraction(
    harness,
    *,
    keep_nope: int,
    lambda_split: bool = True,
    dtype: torch.dtype | None = None,
    log=print,
) -> dict[str, Any]:
    """Contract the non-rotating coordinates of every attention head.

    The NoPE part of a score is ``q_n^T k_n`` with no rotation between the two
    sides, so for ONE head the optimal rank-``d'`` replacement is a truncated
    SVD in the evidence's own metric: with ``q~ = Cq^{-1/2} q`` and
    ``k~ = Ck^{-1/2} k`` the form is ``q~^T (Cq^{1/2} Ck^{1/2}) k~`` and
    truncating THAT matrix is the answer.  ``head_dim`` becomes
    ``rotary_dim + d'`` and ``partial_rotary_factor`` is re-derived so the
    rotary width -- and with it ``mrope_section`` -- is untouched.

    A GQA group does not have one head.  The child holds one key row block per
    group, so ``Pk`` is shared whether or not that is convenient, while ``Pq``
    is free per head.  Two facts make the construction below the right one and
    BOTH are measured per layer rather than asserted:

    * writing the installed operator as ``M = Pq^T Pk``, the group's total score
      error is ``sum_h ||Cq_h^{1/2}(M - I) Ck^{1/2}||_F^2``, which is exactly
      ``||(sum_h Cq_h)^{1/2}(M - I) Ck^{1/2}||_F^2``.  The sum of the per-head
      NoPE second moments is therefore the exact metric for choosing the shared
      row space -- averaging is the aggregation of the per-head objectives, not
      a shortcut past them;
    * for a fixed ``Pk`` the free per-head ``Pq`` buys nothing: the normal
      equation ``W Pk Ck Pk^T = Ck Pk^T`` has ``Cq_h`` cancel, so the
      conditional optimum ``(Pk Ck Pk^T)^-1 Pk Ck`` is the same frame for every
      head in the group.  The loop solves it anyway and records the gap against
      what is installed, per head, under ``query_metric_aggregation``.

    What the sharing DOES cost each head individually is
    ``own_metric_relerror_installed`` -- the relative residual of ``M`` against
    the identity in that head's own second moment -- and its spread across the
    group sits in the same report field.

    The value side and the GATE carry no rotary structure at all (the gate
    multiplies the head's output elementwise, modeling 714), so both are
    contracted WHOLE by the value frame, rotary coordinates included.

    ``head_dim`` is also the softmax TEMPERATURE (``scaling = head_dim ** -0.5``,
    modeling 654, recomputed by the stock class on re-load), so preserving the
    score is not enough: the constant ``sqrt(child_head_dim/parent_head_dim)``
    is folded into ``q_gain`` below, exactly and in closed form.

    What this costs and cannot repair: ``q_norm``/``k_norm`` take an RMS over
    ``head_dim`` coordinates and the child's is over fewer.  The per-token ratio
    is not a weight.  Its MEASURED mean is folded into the child's gain and the
    residual spread is recorded as ``rms_frame_mismatch``.
    """

    dtype = dtype or harness.dtype
    tc = harness.text_config
    layers = harness.layers()
    positions = [i for i, layer in enumerate(layers) if qf_arch.layer_kind(layer) == FULL]
    if not positions:
        raise SystemExit("this model has no full-attention layers")

    rotary_dim = qf_arch.rotary_dim_of(harness.model.config)
    parent_head_dim = int(tc.head_dim)
    parent_nope = parent_head_dim - rotary_dim
    if keep_nope >= parent_nope or keep_nope <= 0:
        raise SystemExit(f"keep_nope {keep_nope} is not a fold of {parent_nope}")
    child_head_dim = rotary_dim + keep_nope
    eps = harness.eps

    per_layer: list[dict[str, Any]] = []
    for position in positions:
        layer = layers[position]
        attn = layer.self_attn
        name = f"L{position:03d}"
        slices = attention_slices(attn, harness.model.config)
        d, nh, nkv, group = slices["head_dim"], slices["heads"], slices["kv_heads"], slices["group"]
        pristine = pristine_copy(attn)

        captured = harness.capture({name: attn}, pristine={name: pristine})[name]
        X, Y = captured["fit"], captured["local_fit"]
        Y_hld = captured.get("local_hld")

        q = attn.q_proj.weight.detach().to("cpu", torch.float64)
        k = attn.k_proj.weight.detach().to("cpu", torch.float64)
        v = attn.v_proj.weight.detach().to("cpu", torch.float64)
        o = attn.o_proj.weight.detach().to("cpu", torch.float64)
        gq = qf_arch.effective_gain(attn.q_norm).detach().to("cpu", torch.float64)
        gk = qf_arch.effective_gain(attn.k_norm).detach().to("cpu", torch.float64)
        rot = torch.arange(rotary_dim)
        nope = torch.arange(rotary_dim, d)

        # Gains are folded into the rows before anything is contracted, and the
        # child's gain becomes a single measured RMS correction.  Mixing the two
        # conventions -- contracting raw rows but keeping the parent's per-dim
        # gain -- would apply the parent's dimension-indexed weights to
        # coordinates that no longer mean what those weights were fitted for.
        q_scaled = torch.stack(
            [q[slices["q_rows"](h)] * gq.unsqueeze(1) for h in range(nh)]
        )
        k_scaled = torch.stack(
            [k[slices["k_rows"](m)] * gk.unsqueeze(1) for m in range(nkv)]
        )

        child_q = torch.zeros(nh * child_head_dim * 2, q.shape[1], dtype=torch.float64)
        child_k = torch.zeros(nkv * child_head_dim, k.shape[1], dtype=torch.float64)
        child_v = torch.zeros(nkv * child_head_dim, v.shape[1], dtype=torch.float64)
        child_o = torch.zeros(o.shape[0], nh * child_head_dim, dtype=torch.float64)
        detail = []

        for m in range(nkv):
            heads = list(range(m * group, (m + 1) * group))
            k_normed = _post_norm(X, k[slices["k_rows"](m)], gk, eps)
            Ck = _second_moment(k_normed[:, nope])
            # The per-head second moments are kept, not just their sum: the sum
            # is what picks the SHARED row space (proof in the docstring) and
            # the individual ones are what price that choice per head below.
            Cq_heads = {}
            for h in heads:
                q_normed = _post_norm(X, q[slices["q_rows"](h)], gq, eps)
                Cq_heads[h] = _second_moment(q_normed[:, nope])
            Cq = sum(Cq_heads.values()) / len(heads)

            root_q, inverse_q = matrix_sqrt(Cq)
            root_k, inverse_k = matrix_sqrt(Ck)
            U, S, Vh = torch.linalg.svd(root_q @ root_k)
            scale = torch.diag(S[:keep_nope].clamp_min(0.0).sqrt())
            Pq = scale @ U[:, :keep_nope].T @ inverse_q
            Pk = scale @ Vh[:keep_nope] @ inverse_k

            # ---- the per-head query frame, MEASURED rather than asserted.
            # `Pq` above is built from the group-averaged `Cq` and the child's
            # query rows are per head, so a reader is owed the number that says
            # what the sharing cost.  For a FIXED `Pk` the per-head optimum is
            # the weighted least squares
            #     argmin_W || Cq_h^{1/2} (W Pk - I) Ck^{1/2} ||_F,
            # whose normal equation is `W Pk Ck Pk^T = Ck Pk^T`.  `Cq_h` is
            # square and invertible (matrix_sqrt floors its spectrum) so it
            # CANCELS, and the conditional optimum
            #     Pq* = (Pk Ck Pk^T)^-1 Pk Ck
            # is the same frame for every head in the group.  `Pq^T Pk` and
            # `Pq*^T Pk` are equal in exact arithmetic; the gap recorded below
            # is the floating-point residual, and it is there so a reviewer can
            # CHECK that claim rather than trust it.  The SVD-scaled form stays
            # the installed one because it splits the singular values evenly
            # between the two sides, where `Pq*` carries `S^-1` and amplifies
            # the smallest retained direction.
            k_pull = Ck @ Pk.T
            conditional = torch.linalg.solve(Pk @ k_pull, k_pull.T)
            operator_installed = Pq.T @ Pk
            operator_conditional = conditional.T @ Pk
            identity_nope = torch.eye(int(len(nope)), dtype=torch.float64)
            per_head_frame = []
            for h in heads:
                root_h, _ = matrix_sqrt(Cq_heads[h])
                reference = max(float((root_h @ root_k).norm()), 1e-300)
                installed_error = float(
                    (root_h @ (operator_installed - identity_nope) @ root_k).norm()
                    / reference
                )
                optimum_error = float(
                    (root_h @ (operator_conditional - identity_nope) @ root_k).norm()
                    / reference
                )
                per_head_frame.append(
                    {
                        "head": h,
                        "own_metric_relerror_installed": installed_error,
                        "own_metric_relerror_conditional_optimum": optimum_error,
                        "own_metric_retained": 1.0 - installed_error ** 2,
                    }
                )
            frame_gap = max(
                abs(
                    row["own_metric_relerror_installed"]
                    - row["own_metric_relerror_conditional_optimum"]
                )
                for row in per_head_frame
            )

            # The value/gate frame: one contraction of the whole head, weighted
            # by every writer that consumes it.
            v_rows = v[slices["v_rows"](m)]
            writers = torch.cat([o[:, slices["o_cols"](h)] for h in heads], dim=0)
            gram = _second_moment(X @ v_rows.T)
            Pv, lra = whitened_lra(writers, gram, child_head_dim)

            base_k = m * child_head_dim
            child_k[base_k : base_k + rotary_dim] = k_scaled[m][rot]
            child_k[base_k + rotary_dim : base_k + child_head_dim] = Pk @ k_scaled[m][nope]
            child_v[base_k : base_k + child_head_dim] = Pv @ v_rows

            for h in heads:
                base = h * 2 * child_head_dim
                child_q[base : base + rotary_dim] = q_scaled[h][rot]
                child_q[base + rotary_dim : base + child_head_dim] = Pq @ q_scaled[h][nope]
                child_q[base + child_head_dim : base + 2 * child_head_dim] = (
                    Pv @ q[slices["gate_rows"](h)]
                )
                child_o[:, h * child_head_dim : (h + 1) * child_head_dim] = project_writer(
                    o[:, slices["o_cols"](h)], Pv
                )
            detail.append(
                {
                    "kv_group": m,
                    "nope_retained_energy": float(
                        S[:keep_nope].pow(2).sum() / S.pow(2).sum().clamp_min(1e-300)
                    ),
                    "value_lra": lra,
                    "query_metric_aggregation": {
                        "shared_row_space_metric": (
                            "sum of the group's per-head NoPE second moments"
                        ),
                        "why_the_sum_is_exact": (
                            "the group's total score error is "
                            "sum_h ||Cq_h^{1/2}(M - I)Ck^{1/2}||_F^2 for ONE shared "
                            "M = Pq^T Pk (the child has one k row block per kv group), "
                            "and that equals ||(sum_h Cq_h)^{1/2}(M - I)Ck^{1/2}||_F^2 "
                            "-- so summing the per-head metrics is the exact "
                            "aggregation of the per-head objectives, not a convenience"
                        ),
                        "per_head": per_head_frame,
                        "installed_vs_per_head_conditional_optimum_max_gap": frame_gap,
                        "spread_own_metric_relerror": [
                            min(
                                row["own_metric_relerror_installed"]
                                for row in per_head_frame
                            ),
                            max(
                                row["own_metric_relerror_installed"]
                                for row in per_head_frame
                            ),
                        ],
                    },
                }
            )

        # The RMS correction, MEASURED rather than assumed: the child's norm
        # divides by an RMS over child_head_dim coordinates of the new rows, the
        # parent's divided by an RMS over head_dim of the old ones.
        ratios: dict[str, list[float]] = {"q": [], "k": []}
        for h in range(nh):
            parent_rms = _rms(X @ q[slices["q_rows"](h)].T, eps)
            child_rows = child_q[h * 2 * child_head_dim : h * 2 * child_head_dim + child_head_dim]
            ratios["q"].append(float((_rms(X @ child_rows.T, eps) / parent_rms).mean()))
        for m in range(nkv):
            parent_rms = _rms(X @ k[slices["k_rows"](m)].T, eps)
            child_rows = child_k[m * child_head_dim : (m + 1) * child_head_dim]
            ratios["k"].append(float((_rms(X @ child_rows.T, eps) / parent_rms).mean()))
        q_ratio = sum(ratios["q"]) / len(ratios["q"])
        k_ratio = sum(ratios["k"]) / len(ratios["k"])

        # THE TEMPERATURE.  The whole construction above preserves the score
        # `q^T k`; the module then multiplies it by `head_dim ** -0.5`, and the
        # child's head_dim is smaller, so a child that stopped here would run a
        # softmax `sqrt(parent_head_dim/child_head_dim)` times sharper than the
        # parent's at every full-attention layer.  Nothing downstream can undo
        # it -- o_proj is on the far side of the softmax.  The q_norm gain is a
        # free per-dimension weight, so folding the constant into it is exact
        # and closed-form.  (This is precisely the correction the C5 head-dim
        # arm does NOT need: the delta rule is linear in q, softmax is not.)
        temperature = (child_head_dim / float(parent_head_dim)) ** 0.5

        install_attention(
            attn,
            q=child_q, k=child_k, v=child_v, o=child_o,
            q_gain=torch.full((child_head_dim,), q_ratio * temperature, dtype=torch.float64),
            k_gain=torch.full((child_head_dim,), k_ratio, dtype=torch.float64),
            head_dim=child_head_dim, heads=nh, kv_heads=nkv, dtype=dtype,
        )
        guard, scores = _resolve_writer(
            harness, attn, name, Y, Y_hld, lambda_split, dtype, log=log
        )
        per_layer.append(
            {
                "layer": position,
                "kv_groups": detail,
                "solve": guard,
                "scores": scores,
                "rms_correction": {"query": q_ratio, "key": k_ratio,
                                   "query_spread": ratios["q"], "key_spread": ratios["k"]},
                "softmax_temperature_correction": {
                    "factor": temperature,
                    "installed_in": "q_norm gain (multiplied into every coordinate)",
                    "why": (
                        f"the child's module scaling is {child_head_dim} ** -0.5 where the "
                        f"parent's was {parent_head_dim} ** -0.5; the construction preserves "
                        f"q^T k, so without this factor every attention logit would be "
                        f"{(parent_head_dim / child_head_dim) ** 0.5:.4f}x the parent's"
                    ),
                },
                "installed_query_gain": q_ratio * temperature,
                "rms_frame_mismatch": (
                    f"q_norm/k_norm take an RMS over {child_head_dim} coordinates where "
                    f"the parent took it over {parent_head_dim}; the measured mean ratio "
                    f"is folded into the child gains, the per-token residual is not "
                    f"repairable by any weight"
                ),
                "gate_frame_mismatch": (
                    "gate rows contracted by the value frame Pv; sigmoid is elementwise "
                    "and does not commute with Pv"
                ),
            }
        )
        log(
            f"[qf] C6 {name}: head_dim {parent_head_dim} -> {child_head_dim} "
            f"(NoPE {parent_nope} -> {keep_nope})  fit relL2 {scores['fit']['relL2']:.4f}"
        )

    qf_arch.config_field(tc, "head_dim", child_head_dim)
    params = dict(tc.rope_parameters or {})
    params["partial_rotary_factor"] = qf_arch.exact_partial_rotary_factor(
        rotary_dim, child_head_dim
    )
    qf_arch.config_field(tc, "rope_parameters", params)
    if qf_arch.rotary_dim_of(tc) != rotary_dim:
        raise SystemExit(
            f"the child's config would build a {qf_arch.rotary_dim_of(tc)}-wide rotary "
            f"ladder where the parent's is {rotary_dim}; the rotary frequencies would "
            f"not be the parent's and the fold would not be a fold"
        )

    hidden = int(tc.hidden_size)
    nh, nkv = int(tc.num_attention_heads), int(tc.num_key_value_heads)
    delta = parent_head_dim - child_head_dim
    removed = (2 * nh + 2 * nkv + nh) * hidden * delta + 2 * delta
    return {
        "axis": "head_dim (NoPE coordinates)",
        "kind": "two-sided evidence-whitened contraction of the non-rotating score form",
        "sites": len(per_layer),
        "parent_head_dim": parent_head_dim,
        "child_head_dim": child_head_dim,
        "rotary_dim": rotary_dim,
        "component_ratio": child_head_dim / float(parent_head_dim),
        "kv_cache_ratio": child_head_dim / float(parent_head_dim),
        "softmax_temperature_correction": (child_head_dim / float(parent_head_dim)) ** 0.5,
        "parameters_removed": removed * len(per_layer),
        "per_layer": per_layer,
    }


# ------------------------------------------------------------- C7: kv merge
def kv_head_merge(
    harness,
    *,
    kv_heads: int,
    arm: str = "kv-only",
    lambda_split: bool = True,
    dtype: torch.dtype | None = None,
    log=print,
) -> dict[str, Any]:
    """Merge key/value heads with a per-plane joint relation solve.

    The merged key is constrained to the SPAN of its parents and chosen by the
    Hermitian eigenproblem in :func:`qf_solve.joint_relation_solve`; every query
    head that read a merged parent is then re-fitted in closed form against that
    choice.  Both are computed on the GAIN-SCALED PRE-NORM rows -- the ``q_norm``
    /``k_norm`` weights are folded in before the solve and divided back out
    after, so the relation preserved is the one the softmax sees UP TO the
    per-token RMS division, which is not a weight and is not in the solve.  The
    resulting key RMS shift is MEASURED per child head (``key_rms_ratio``) and
    reported as ``rms_frame_mismatch``; it is deliberately not installed,
    because ``k_norm``'s gain is shared by every query head reading that key and
    scaling it would change the temperature of heads whose relation was already
    solved for.  ``head_dim`` does not move here, so the softmax temperature
    ``head_dim ** -0.5`` is the parent's exactly.
    """

    dtype = dtype or harness.dtype
    tc = harness.text_config
    layers = harness.layers()
    positions = [i for i, layer in enumerate(layers) if qf_arch.layer_kind(layer) == FULL]
    if not positions:
        raise SystemExit("this model has no full-attention layers")

    parent_kv = int(tc.num_key_value_heads)
    parent_heads = int(tc.num_attention_heads)
    if parent_kv % kv_heads:
        raise SystemExit(f"{parent_kv} kv heads do not merge into {kv_heads}")
    reduction = parent_kv // kv_heads
    if reduction == 1:
        raise SystemExit("nothing to merge")
    if arm not in {"kv-only", "full-group"}:
        raise SystemExit(f"unknown arm {arm!r}")
    group = parent_heads // parent_kv
    child_heads = parent_heads if arm == "kv-only" else kv_heads * group

    rotary_dim = qf_arch.rotary_dim_of(harness.model.config)
    head_dim = int(tc.head_dim)
    layout = qf_arch.plane_layout(head_dim, rotary_dim)
    eps = harness.eps

    per_layer: list[dict[str, Any]] = []
    for position in positions:
        layer = layers[position]
        attn = layer.self_attn
        name = f"L{position:03d}"
        slices = attention_slices(attn, harness.model.config)
        pristine = pristine_copy(attn)
        captured = harness.capture({name: attn}, pristine={name: pristine})[name]
        X, Y = captured["fit"], captured["local_fit"]
        Y_hld = captured.get("local_hld")

        q = attn.q_proj.weight.detach().to("cpu", torch.float64)
        k = attn.k_proj.weight.detach().to("cpu", torch.float64)
        v = attn.v_proj.weight.detach().to("cpu", torch.float64)
        o = attn.o_proj.weight.detach().to("cpu", torch.float64)
        gq = qf_arch.effective_gain(attn.q_norm).detach().to("cpu", torch.float64)
        gk = qf_arch.effective_gain(attn.k_norm).detach().to("cpu", torch.float64)

        child_q = torch.zeros(child_heads * head_dim * 2, q.shape[1], dtype=torch.float64)
        child_k = torch.zeros(kv_heads * head_dim, k.shape[1], dtype=torch.float64)
        child_v = torch.zeros(kv_heads * head_dim, v.shape[1], dtype=torch.float64)
        child_o = torch.zeros(o.shape[0], child_heads * head_dim, dtype=torch.float64)
        detail = []
        zero_gain: list[dict[str, Any]] = []

        for m_child in range(kv_heads):
            parents = list(range(m_child * reduction, (m_child + 1) * reduction))
            heads = [h for m in parents for h in range(m * group, (m + 1) * group)]
            q_rows = [q[slices["q_rows"](h)] * gq.unsqueeze(1) for h in heads]
            k_rows = [k[slices["k_rows"](m)] * gk.unsqueeze(1) for m in parents]
            group_of = [parents.index(h // group) for h in heads]
            weights = [
                max(float(o[:, slices["o_cols"](h)].pow(2).sum()), 1e-30) for h in heads
            ]
            k_hat, q_hats, relation = joint_relation_solve(
                q_rows, k_rows, group_of, weights, layout
            )
            # A zero-gain KEY coordinate has no single parent row to fall back
            # to -- the merged key has `reduction` of them -- so the fallback is
            # their mean, which is what the merged key's RMS contribution should
            # resemble.  Qwen3.5-4B has no zero-gain k_norm coordinate at any
            # full-attention layer (the smallest is 0.00390625 at layer 27), so
            # this branch is dormant on this parent; it is here because the
            # division is the same division.
            child_k[m_child * head_dim : (m_child + 1) * head_dim] = unfold_gain(
                k_hat,
                gk,
                sum(k[slices["k_rows"](m)] for m in parents) / len(parents),
                record=zero_gain,
                where=f"k_norm, child kv head {m_child}",
            )

            # The joint value space of the merged parents is `reduction *
            # head_dim` wide, and each query head's writer acts on ONE parent's
            # block of it, so the writer stack is placed block-wise.
            v_rows = torch.cat([v[slices["v_rows"](m)] for m in parents], dim=0)
            placed = []
            for local, h in enumerate(heads):
                block = torch.zeros(o.shape[0], reduction * head_dim, dtype=torch.float64)
                start = group_of[local] * head_dim
                block[:, start : start + head_dim] = o[:, slices["o_cols"](h)]
                placed.append(block)
            gram = _second_moment(X @ v_rows.T)
            Pv, lra = whitened_lra(torch.cat(placed, dim=0), gram, head_dim)
            child_v[m_child * head_dim : (m_child + 1) * head_dim] = Pv @ v_rows

            if arm == "kv-only":
                for local, h in enumerate(heads):
                    base = h * 2 * head_dim
                    child_q[base : base + head_dim] = unfold_gain(
                        q_hats[local],
                        gq,
                        q[slices["q_rows"](h)],
                        record=zero_gain,
                        where=f"q_norm, query slot {h}",
                    )
                    # EXACT: the gate reads the residual and multiplies the
                    # head's own output; nothing about it changed.
                    child_q[base + head_dim : base + 2 * head_dim] = q[slices["gate_rows"](h)]
                    child_o[:, h * head_dim : (h + 1) * head_dim] = project_writer(
                        placed[local], Pv
                    )
            else:
                for t in range(group):
                    child_head = m_child * group + t
                    members = [local for local, h in enumerate(heads) if h % group == t]
                    energy = torch.tensor([weights[i] for i in members], dtype=torch.float64)
                    energy = energy / energy.sum()
                    base = child_head * 2 * head_dim
                    child_q[base : base + head_dim] = sum(
                        float(w) * unfold_gain(
                            q_hats[i],
                            gq,
                            q[slices["q_rows"](heads[i])],
                            record=zero_gain,
                            where=f"q_norm, merged query head {child_head} <- slot {heads[i]}",
                        )
                        for w, i in zip(energy, members)
                    )
                    child_q[base + head_dim : base + 2 * head_dim] = sum(
                        float(w) * q[slices["gate_rows"](heads[i])]
                        for w, i in zip(energy, members)
                    )
                    child_o[:, child_head * head_dim : (child_head + 1) * head_dim] = sum(
                        float(w) * project_writer(placed[i], Pv) for w, i in zip(energy, members)
                    )
            detail.append(
                {
                    "child_kv_head": m_child,
                    "parent_kv_heads": parents,
                    "query_slots": heads,
                    "relation": relation,
                    "value_lra": lra,
                }
            )

        key_ratio = []
        for m in range(kv_heads):
            child_rows = child_k[m * head_dim : (m + 1) * head_dim]
            parents = list(range(m * reduction, (m + 1) * reduction))
            parent_rms = sum(
                _rms(X @ k[slices["k_rows"](p)].T, eps) for p in parents
            ) / len(parents)
            key_ratio.append(float((_rms(X @ child_rows.T, eps) / parent_rms).mean()))

        install_attention(
            attn, q=child_q, k=child_k, v=child_v, o=child_o,
            q_gain=gq, k_gain=gk, head_dim=head_dim, heads=child_heads,
            kv_heads=kv_heads, dtype=dtype,
        )
        guard, scores = _resolve_writer(
            harness, attn, name, Y, Y_hld, lambda_split, dtype, log=log
        )
        per_layer.append(
            {
                "layer": position,
                "per_kv_head": detail,
                "solve": guard,
                "scores": scores,
                "key_rms_ratio": key_ratio,
                "zero_gain_coordinates": zero_gain,
                "zero_gain_policy": (
                    "a coordinate whose effective norm gain (1 + weight) is "
                    "exactly zero cannot have the gain divided back out of the "
                    "solved rows; the parent's own raw row is installed there "
                    "instead.  The score is unchanged (the norm multiplies that "
                    "coordinate by zero), the per-token RMS the norm divides the "
                    "head by is the parent's, and nothing is +-Inf.  Empty list "
                    "means the parent had no such coordinate at this layer."
                ),
                "rms_frame_mismatch": (
                    "the relation is solved on gain-scaled PRE-norm rows; k_norm's "
                    f"per-token RMS division is not in the solve and the merged key's "
                    f"measured RMS ratio is {['%.4f' % r for r in key_ratio]}.  It is "
                    "NOT installed: k_norm's gain is shared by every query head that "
                    "reads this key, so folding it in would retune heads whose "
                    "relation was already solved.  Unrepaired approximation."
                ),
                "gate": (
                    "exact: gate rows copied verbatim"
                    if arm == "kv-only"
                    else "gate_frame_mismatch: gate rows averaged across merged query heads"
                ),
            }
        )
        log(
            f"[qf] C7 {name} ({arm}): kv {parent_kv} -> {kv_heads}, heads "
            f"{parent_heads} -> {child_heads}  fit relL2 {scores['fit']['relL2']:.4f}"
            + (
                f"  [zero-gain coordinates "
                f"{sorted({c for row in zero_gain for c in row['coordinates']})} "
                f"held at the parent's rows]"
                if zero_gain
                else ""
            )
        )

    qf_arch.config_field(tc, "num_key_value_heads", kv_heads)
    qf_arch.config_field(tc, "num_attention_heads", child_heads)
    hidden = int(tc.hidden_size)
    removed = (
        2 * (parent_kv - kv_heads) * head_dim * hidden
        + (parent_heads - child_heads) * head_dim * 3 * hidden
    )
    return {
        "axis": "num_key_value_heads" if arm == "kv-only"
        else "num_key_value_heads+num_attention_heads",
        "kind": f"kv-head merge ({arm}) by per-plane joint relation solve",
        "arm": arm,
        "sites": len(per_layer),
        "parent_kv_heads": parent_kv,
        "child_kv_heads": kv_heads,
        "parent_attention_heads": parent_heads,
        "child_attention_heads": child_heads,
        "kv_cache_ratio": kv_heads / float(parent_kv),
        "parameters_removed": removed * len(per_layer),
        "per_layer": per_layer,
    }


@torch.no_grad()
def attention_gate_probe(
    harness, layer_index: int, batches: int = 1, *, keep_tensors: bool = False
) -> dict[str, Any]:
    """Recompute one layer's attention probabilities and compare to the module's own.

    The primitive the causal assertion and every attention arm's recompute check
    stand on: if this disagrees, the recompute is wrong and every number derived
    from it is void.  It is the ONLY instrument in the lane that reads the
    query/gate row split, the rotary/NoPE split, the per-head placement, the
    norm gains and ``scaling`` all at once, which is why C6 and C7 call it on
    the parent (as an assertion) and on the child (as a measurement) rather than
    trusting shapes and a re-load.

    With ``keep_tensors`` the pre-mask logits and the probabilities come back so
    a caller can compare a child against its parent -- see
    :func:`logit_scale_ratio`, which is what catches a temperature change that a
    shape check cannot see.
    """

    layer = harness.layers()[layer_index]
    attn = layer.self_attn
    grabbed: dict[str, Any] = {}

    def pre(module, args, kwargs):
        grabbed["hidden"] = (args[0] if args else kwargs["hidden_states"]).detach()
        grabbed["pos"] = kwargs.get("position_embeddings")
        return None

    def post(module, args, kwargs, output):
        grabbed["weights"] = output[1].detach()
        return None

    handles = [
        attn.register_forward_pre_hook(pre, with_kwargs=True),
        attn.register_forward_hook(post, with_kwargs=True),
    ]
    try:
        for _, ids in harness.batches[:batches]:
            harness.model(input_ids=ids.to(harness.device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    hidden = grabbed["hidden"].to(torch.float64)
    cos, sin = (t.to(torch.float64) for t in grabbed["pos"])
    d = int(attn.head_dim)
    q_pair = (hidden @ attn.q_proj.weight.detach().to(torch.float64).T).view(
        *hidden.shape[:-1], -1, d * 2
    )
    qs, _ = torch.chunk(q_pair, 2, dim=-1)
    ks = (hidden @ attn.k_proj.weight.detach().to(torch.float64).T).view(
        *hidden.shape[:-1], -1, d
    )
    gq = qf_arch.effective_gain(attn.q_norm).to(torch.float64)
    gk = qf_arch.effective_gain(attn.k_norm).to(torch.float64)
    eps = float(attn.q_norm.eps)
    qs = qs * torch.rsqrt(qs.pow(2).mean(-1, keepdim=True) + eps) * gq
    ks = ks * torch.rsqrt(ks.pow(2).mean(-1, keepdim=True) + eps) * gk
    qs, ks = apply_rotary_pos_emb(qs.transpose(1, 2), ks.transpose(1, 2), cos, sin)
    ks = ks.repeat_interleave(attn.num_key_value_groups, dim=1)
    logits = qs @ ks.transpose(2, 3) * attn.scaling
    causal = torch.full(logits.shape[-2:], float("-inf"), dtype=logits.dtype).triu(1)
    probabilities = torch.softmax(logits + causal, dim=-1)
    reference = grabbed["weights"].to(torch.float64)
    report = {
        "layer": layer_index,
        "head_dim": d,
        "scaling": float(attn.scaling),
        "recompute_relL2": float((probabilities - reference).norm() / reference.norm()),
        "above_diagonal_mass_module": float(
            torch.triu(reference, diagonal=1).sum() / reference.sum()
        ),
        "mean_max_probability": float(probabilities.max(dim=-1).values.mean()),
    }
    if keep_tensors:
        report["logits"] = logits
        report["probabilities"] = probabilities
    return report


def logit_scale_ratio(child_logits: torch.Tensor, parent_logits: torch.Tensor) -> float:
    """The one scalar ``a`` minimising ``|| a*S_child - S_parent ||`` on the causal support.

    Rows are centred over their own causal support first, because a softmax is
    invariant to a per-row additive constant and the question here is only about
    SHARPNESS.  A contraction that truncates in the evidence metric is a
    projection, so its own error is orthogonal to what it kept and ``a`` comes
    back at 1; a change of softmax temperature moves ``a`` by exactly that
    factor and nothing else in this lane can see it.
    """

    def centred(t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float64)
        mask = torch.ones(t.shape[-2:], dtype=torch.bool, device=t.device).tril()
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = (t * mask).sum(dim=-1, keepdim=True) / count
        return (t - mean) * mask

    child, parent = centred(child_logits), centred(parent_logits)
    return float((child * parent).sum() / (child * child).sum().clamp_min(1e-300))
