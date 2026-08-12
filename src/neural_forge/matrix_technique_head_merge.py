#!/usr/bin/env python3
"""M5: merge attention heads onto a shared basis by orthogonal Procrustes.

`reports/matrix-campaign-20260810/wave12/NOTES-MX-P.md` §8 is this kernel's
registered specification and it exists because of a verdict, not a hunch.  C10's
other kernel -- `matrix_technique_head_drop` -- **cannot express S4**: a dropped
head removes ``D`` rows of ``q_proj`` and ``D`` columns of ``out_proj`` and
stores ``k_proj``/``v_proj`` whole, so fifteen of sixteen heads store 0.958x and
the storage model bottoms out around 0.5x with one head left.  ADDENDUM 6 calls
that ``cannot-express-4x`` and no behavioural claim follows from it.

A *merge* is a different claim.  Write the attention block per head::

    Q_h = q_proj[h*D:(h+1)*D, :]        (D x hidden)     16 query heads
    K_j = k_proj[j*D:(j+1)*D, :]        (D x hidden)      8 kv heads
    V_j = v_proj[j*D:(j+1)*D, :]        (D x hidden)
    O_h = out_proj[:, h*D:(h+1)*D]      (hidden x D)

and store, instead of 24 head blocks, ``g_q`` query-group bases with their O
partners, ``g_kv`` kv-group bases, and one ``D x D`` orthogonal alignment per
head::

    q_h = R_h @ Qb[a]        o_h = Ob[a] @ R_h.T          a = group of head h
    k_j = S_j @ Kb[b]        v_j = S_j @ Vb[b]            b = group of kv head j

The alignments are what make this a merge and not a drop: every head keeps its
own frame and only the *content* is shared.  `stored(g)` is §8's arithmetic::

    stored = g_q*(D*hidden + hidden*D) + g_kv*2*(D*hidden) + (H + H_kv)*D*D

    g_q = 4, g_kv = 2  ->  884,736  = 0.28125x   (over the 4x bar)
    g_q = 3, g_kv = 2  ->  753,664  = 0.239583x  (under it)

**One correction to §8 of record.**  §8 priced ``g = 3`` at 688,128 = 0.21875x by
writing the kv term as ``(g/2) * 2 * (64*1024)`` and evaluating ``g/2 = 1.5``.
There is no such thing as one and a half kv groups.  Eight kv heads split into
``ceil(3 * 8/16) = 2`` groups, so the implementable ``g_q = 3`` config stores
**753,664 = 0.239583x**, not 0.21875x.  It is still under the S4 bar -- the
verdict §8 reached survives -- but the number it reached it with was 8.7% low and
this kernel refuses to inherit it.  ``g_q = 4`` at 0.28125x is unchanged and is
still over the bar; it may be measured, and a receipt that seals it is a
**sub-4x diagnostic** and never a frontier claim.

How the fit uses the sealed evidence
------------------------------------

C10's endpoints are C09's (`matrix_technique_registry:181`): the cell reads
``attention.context`` and writes ``attention.o``, and
``attention.o = attention.context @ out_proj.T`` is a genuine single matmul
between two sealed endpoints.  That is the only place in an attention block
where evidence and weights meet exactly, and this kernel spends it twice:

1. **The parent gate.**  ``held_context @ out_proj.T`` against the sealed
   ``attention.o``; a parent that does not reproduce its own write means the
   cell is measuring something else.  Sealed as
   ``parent_matmul_relative_l2``, exactly as the head-drop kernel seals it.
2. **The O bases are re-solved against the sealed write.**  With the alignments
   fixed, the child's O contribution is *linear in the stored bases*::

       context @ o_child.T = sum_a (sum_{h in a} context_h @ R_h) @ Ob[a].T
                           = Z @ W.T,   W = [Ob_1 | ... | Ob_g]

   so the group bases are a prior-anchored ridge fit over the same production
   solver every dense cell uses, with the Procrustes bases as the prior.  This
   is `matrix_technique_head_drop`'s absorption step generalized from "the
   surviving columns" to "the shared bases", and it is why the merge is graded
   on what the block actually wrote rather than on how close two weight matrices
   look.

Everything the evidence cannot reach directly -- ``q_proj``, ``k_proj``,
``v_proj``, whose writes are behind a softmax and are not endpoints of a C10
cell -- is fit in the weight metric, with the **o term weighted by the head's
own measured context energy** ``lambda_h = E||context_h||^2 / mean_h(...)``.  A
head that writes more of the block's output is worth more of the shared basis,
and that ranking comes from the sealed rows rather than from the tensors.

The grouping is not chosen either
---------------------------------

Which heads share a basis is decided by agglomerative clustering whose linkage
is *the merge objective itself*: at every step the pair of clusters whose union
costs the least additional residual is merged, where a cluster's cost is the
converged alternating Procrustes objective over that cluster.  Sixteen heads to
three groups is thirteen merges and 676 cluster evaluations of 64x64 SVDs, which
costs seconds; a similarity heuristic would have cost the same and answered a
different question.

Three properties worth knowing before reading a receipt
-------------------------------------------------------

1. **The identity gate is measured, not asserted.**  At ``g_q = H`` and
   ``g_kv = H_kv`` every group holds one head, the Procrustes step returns the
   identity, the basis update returns the head, and the O refit's target is
   the parent's own write, so a prior-anchored fit with a zero residual returns
   its prior.  All four installed tensors must come back **bit-equal to the
   parent** and ``identity_gate_max_abs_deviation`` is that measurement over the
   whole four-tensor group.
2. **The alternation is monotone and its trace is sealed.**  Each half-step is
   the exact minimizer of the other's argument (orthogonal Procrustes; a
   weighted mean), so the objective cannot rise.  ``objective_trace`` is sealed
   so a reader can see that it did not.
3. **The storage claim covers all four projections and nothing is stored
   twice.**  Unlike head drop -- which stores ``k_proj``/``v_proj`` whole and
   says so -- a merge replaces every one of C10's four tensors, so
   ``installed`` and ``owned`` coincide and `matrix_composition.net_storage`'s
   coverage rule is inert here.

An open question this kernel does **not** resolve, stated because it bounds the
result.  §8 ties one alignment to the (q, o) pair of a head and one to the
(k, v) pair of a kv head, which is what makes 24 alignments and what makes
``g_q = 3`` fit under the bar.  The *gauge* of a single attention head pairs
(q, k) and (v, o) instead -- rotating q and k together leaves the scores
invariant, and rotating v and o together leaves the write invariant -- so §8's
pairing asks one rotation to serve two reconstructions that the architecture
does not couple.  Pricing the gauge-faithful pairing means a different storage
table (a separate alignment per projection is 48 of them, 196,608 parameters,
which pushes ``g_q = 3, g_kv = 2`` to 0.2708x and out of S4), so it is a
different cell and not a knob.  This kernel implements what was registered.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "C10_COMPONENT_ID",
    "HEAD_MERGE_HEAD_DIM",
    "HEAD_MERGE_TECHNIQUE_ID",
    "c10_head_merge_storage",
    "c10_head_merge_technique",
    "describe_head_merge_technique",
    "head_merge_group_plan",
    "head_merge_stored_parameters",
]

C10_COMPONENT_ID = "C10"

#: Registered under an existing technique id, pinned to one component, which is
#: the shape `matrix_technique_registry` already documents for T1-at-C14 (the
#: standalone norm refit) and T1-at-C10 (head drop).  T7 is
#: ``attention_relation_matching`` and is registered for C06/C07/C06+C07; a head
#: merge is relation matching between heads, and pinning it to C10 collides with
#: no registered pair.  Minting a T10 would have widened `TECHNIQUES` and
#: therefore `known_cell_ids`, which every one of the campaign's 442 sealed rows
#: is verified against -- ADDENDUM 10 went out of its way to keep those objects
#: byte-identical and this kernel is not the thing that breaks them.
HEAD_MERGE_TECHNIQUE_ID = "T7"

#: The head width of the LFM2 parent family, pinned as this technique's own
#: constant for exactly the reason `_c01_token_route_storage` pins C01's route
#: head size: a storage model is a function of ONE width, and
#: `storage_width_for_ratio` cannot resolve a level against a model that needs a
#: second parameter.  The *kernel* never trusts it -- it reads ``head_dim`` off
#: the parent's own ``q_layernorm`` gain and refuses when the two disagree.
HEAD_MERGE_HEAD_DIM = 64


def _torch() -> Any:
    import torch

    return torch


# --------------------------------------------------------------------------
# the storage algebra (NOTES-MX-P §8, with §8's kv term made integral)
# --------------------------------------------------------------------------


def head_merge_group_plan(
    width: int,
    *,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[int, int]:
    """``width`` -> ``(g_q, g_kv)``, the two group counts the level buys.

    The level's width is the *stored query-basis height*: ``g_q`` bases of
    ``head_dim`` rows each, which is the same reading `matrix_technique_head_drop`
    gives C10's level axis (kept heads x head_dim) and keeps the two C10 kernels
    talking about the same number.

    ``g_kv`` is derived, not chosen: the kv side is merged in the same
    proportion the parent's GQA already uses, rounded **up** so a query group
    never has fewer kv groups to attend through than the ratio implies.  For
    LFM2's 16/8 that is ``ceil(g_q / 2)``, which is where §8's non-integral
    ``g/2 = 1.5`` becomes 2 and its 0.21875x becomes 0.239583x.
    """

    if head_dim <= 0 or query_heads <= 0 or kv_heads <= 0:
        raise ValueError("head geometry must be positive")
    groups = int(width) // int(head_dim)
    if groups < 1:
        groups = 1
    groups = min(groups, int(query_heads))
    kv_groups = -(-groups * int(kv_heads) // int(query_heads))  # ceil
    kv_groups = max(1, min(int(kv_heads), kv_groups))
    return groups, kv_groups


def head_merge_stored_parameters(
    *,
    hidden_size: int,
    head_dim: int,
    query_heads: int,
    kv_heads: int,
    query_groups: int,
    kv_groups: int,
) -> int:
    """§8's ``stored(g)``, with one alignment per head charged in full.

    An orthogonal ``D x D`` matrix has ``D*(D-1)/2`` degrees of freedom and could
    be stored as Givens angles at 2,016 instead of 4,096.  The child does not do
    that, so this does not claim it: `NOTES-MX-E` §4b exists to catch exactly the
    saving a model banks and an artifact does not take.
    """

    basis = int(head_dim) * int(hidden_size)
    return int(
        int(query_groups) * (basis + int(hidden_size) * int(head_dim))
        + int(kv_groups) * 2 * basis
        + (int(query_heads) + int(kv_heads)) * int(head_dim) * int(head_dim)
    )


def c10_head_merge_admissible_width(
    spec: Any, ratio: float, *, head_dim: int | None = None
) -> int | None:
    """The widest **head-aligned** width whose stored form fits ``ratio``.

    `storage_width_for_ratio` searches every integer and returns the widest one
    that fits, which for this technique is a width in the middle of a group --
    255 buys the same three query groups 192 does.  The kernel refuses a level
    that does not name whole groups (a caller who asks for 255 has almost
    certainly mis-derived something), so a planner needs the aligned answer and
    this is it.  Returns ``None`` when no width fits, which is ADDENDUM 6's
    ``cannot-express``.
    """

    from .matrix_technique_registry import storage_width_for_ratio

    width = storage_width_for_ratio(spec, HEAD_MERGE_TECHNIQUE_ID, ratio)
    if width is None:
        return None
    resolved = HEAD_MERGE_HEAD_DIM if head_dim is None else int(head_dim)
    aligned = int(width) - int(width) % resolved
    if aligned < resolved:
        return None
    return aligned


def _c10_geometry(spec: Any, head_dim: int) -> tuple[int, int, int, int]:
    """``(hidden, head_dim, query_heads, kv_heads)`` from C10's own numbers.

    ``ComponentSpec`` carries the primary tensor's two widths and the owned
    parameter count over all four projections, which is enough: with
    ``q_proj`` at ``(q_width, hidden)`` and ``out_proj`` at ``(hidden, q_width)``
    the remaining ``P - 2*q_width*hidden`` parameters are ``k_proj`` and
    ``v_proj``, both ``(kv_width, hidden)``.
    """

    hidden = int(spec.parent_input_width)
    q_width = int(spec.parent_output_width)
    owned = int(spec.parent_parameter_count)
    remainder = owned - 2 * q_width * hidden
    if hidden <= 0 or q_width <= 0 or remainder <= 0 or remainder % (2 * hidden) != 0:
        raise ValueError(
            f"C10's owned parameter count {owned} is not q/k/v/o over a "
            f"{q_width}x{hidden} query projection."
        )
    kv_width = remainder // (2 * hidden)
    if q_width % head_dim != 0 or kv_width % head_dim != 0:
        raise ValueError(
            f"head_dim {head_dim} does not divide the {q_width}-wide query and "
            f"{kv_width}-wide kv projections."
        )
    return hidden, int(head_dim), q_width // head_dim, kv_width // head_dim


def c10_head_merge_storage(spec: Any, width: int, *, head_dim: int | None = None) -> int:
    """The registry's storage model for ``(T7, C10)``.

    Registered in `matrix_technique_registry._STORAGE_MODELS_BY_COMPONENT` the
    same way ``(T5, C01)`` is: through a lazy import, so the frozen dispatch
    table never imports a technique lane at module scope.

    ``head_dim`` defaults to this technique's pinned constant because
    `storage_width_for_ratio` calls a storage model with exactly one width and
    nothing else -- planning a level cannot depend on a second parameter.  The
    *kernel* always passes the head width it read off the parent, so the
    agreement check it runs after the solve is a check against the geometry the
    child was actually built at and not against the constant.
    """

    hidden, head_dim, query_heads, kv_heads = _c10_geometry(
        spec, HEAD_MERGE_HEAD_DIM if head_dim is None else int(head_dim)
    )
    query_groups, kv_groups = head_merge_group_plan(
        width, query_heads=query_heads, kv_heads=kv_heads, head_dim=head_dim
    )
    return head_merge_stored_parameters(
        hidden_size=hidden,
        head_dim=head_dim,
        query_heads=query_heads,
        kv_heads=kv_heads,
        query_groups=query_groups,
        kv_groups=kv_groups,
    )


# --------------------------------------------------------------------------
# the merge itself
# --------------------------------------------------------------------------


def _procrustes(cross: Any) -> Any:
    """``argmax_R <R, cross>`` over orthogonal ``R``: ``U V.T`` of ``cross``."""

    torch = _torch()
    left, _singular, right_t = torch.linalg.svd(cross, full_matrices=False)
    return left @ right_t


def _blocks(matrix: Any, count: int, size: int, *, axis: int) -> list[Any]:
    """Split a projection into its per-head blocks along ``axis``."""

    if axis == 0:
        return [matrix[index * size : (index + 1) * size, :] for index in range(count)]
    return [matrix[:, index * size : (index + 1) * size] for index in range(count)]


class _MergePair:
    """One (primary, partner) family of head blocks merged under one alignment.

    Two instances cover the whole component.  The query family carries ``Q_h``
    as the primary and ``O_h`` as the partner, with the partner entering
    transposed because ``o_h = Ob @ R.T`` and ``q_h = R @ Qb``; the kv family
    carries ``K_j`` with ``V_j``, both untransposed.  Writing it once means the
    alternation, the objective and the clustering are the same code for both
    and cannot drift apart between them.
    """

    def __init__(
        self,
        primary: Sequence[Any],
        partner: Sequence[Any],
        weights: Sequence[float],
        *,
        partner_transposed: bool,
    ) -> None:
        torch = _torch()
        self.primary = list(primary)
        # The partner is carried in the orientation the Procrustes step wants:
        # for the query family that is O_h.T, so that both terms of the cross
        # matrix are ``block @ basis.T``.
        self.partner = [
            value.T.contiguous() if partner_transposed else value for value in partner
        ]
        self.weights = [float(value) for value in weights]
        self.partner_transposed = bool(partner_transposed)
        self.count = len(self.primary)
        self.dtype = self.primary[0].dtype
        self.identity = torch.eye(
            int(self.primary[0].shape[0]), dtype=self.dtype
        )

    # -- one alternation ---------------------------------------------------

    def _align(self, index: int, primary_basis: Any, partner_basis: Any) -> Any:
        cross = self.primary[index] @ primary_basis.T + self.weights[index] * (
            self.partner[index] @ partner_basis.T
        )
        return _procrustes(cross)

    def _bases(self, members: Sequence[int], rotations: Sequence[Any]) -> tuple[Any, Any]:
        torch = _torch()
        primary = torch.zeros_like(self.primary[0])
        partner = torch.zeros_like(self.partner[0])
        weight_total = 0.0
        for position, index in enumerate(members):
            primary = primary + rotations[position].T @ self.primary[index]
            partner = partner + self.weights[index] * (
                rotations[position].T @ self.partner[index]
            )
            weight_total += self.weights[index]
        primary = primary / float(len(members))
        if weight_total > 0.0:
            partner = partner / weight_total
        return primary, partner

    def _residual(
        self,
        members: Sequence[int],
        rotations: Sequence[Any],
        primary_basis: Any,
        partner_basis: Any,
    ) -> float:
        total = 0.0
        for position, index in enumerate(members):
            delta = self.primary[index] - rotations[position] @ primary_basis
            total += float((delta * delta).sum().item())
            delta = self.partner[index] - rotations[position] @ partner_basis
            total += self.weights[index] * float((delta * delta).sum().item())
        return total

    def fit_group(
        self,
        members: Sequence[int],
        *,
        sweeps: int,
        improvement_floor: float,
    ) -> tuple[float, list[Any], Any, Any, list[float]]:
        """Alternating Procrustes over one group.  Monotone by construction.

        The basis is initialized to the group's **lowest-indexed head** rather
        than to a mean: a mean of unaligned heads is a smaller object than any
        of them and starts the alternation inside a basin no head is in.  The
        choice is deterministic, which the campaign needs more than it needs a
        marginally better initialization -- the same recipe must produce the
        same child on any run.
        """

        members = list(members)
        anchor = members[0]
        primary_basis = self.primary[anchor].clone()
        partner_basis = self.partner[anchor].clone()
        rotations = [self.identity.clone() for _ in members]
        trace: list[float] = []
        previous = math.inf
        for _sweep in range(max(1, int(sweeps))):
            rotations = [
                self._align(index, primary_basis, partner_basis) for index in members
            ]
            primary_basis, partner_basis = self._bases(members, rotations)
            value = self._residual(members, rotations, primary_basis, partner_basis)
            trace.append(value)
            if previous < math.inf and previous - value <= abs(previous) * float(
                improvement_floor
            ):
                break
            previous = value
        return trace[-1], rotations, primary_basis, partner_basis, trace

    def energy(self) -> float:
        """The objective a partition of singletons would leave: zero.

        Kept as the denominator every reported residual is relative to.
        """

        total = 0.0
        for index in range(self.count):
            block = self.primary[index]
            total += float((block * block).sum().item())
            block = self.partner[index]
            total += self.weights[index] * float((block * block).sum().item())
        return total

    # -- grouping ----------------------------------------------------------

    def cluster(
        self,
        groups: int,
        *,
        sweeps: int,
        improvement_floor: float,
    ) -> tuple[list[list[int]], list[dict[str, Any]]]:
        """Agglomerate to ``groups`` clusters, linkage = the merge objective.

        The cost of a candidate union is the converged alternating objective
        over the union minus the two costs it replaces, so the clustering
        optimizes the same quantity the fit does instead of a proxy for it.
        Ties break on the lowest member indices, which makes the whole grouping
        a deterministic function of the parent's weights and the sealed rows.
        """

        clusters: list[list[int]] = [[index] for index in range(self.count)]
        cache: dict[tuple[int, ...], float] = {}

        def cost(members: Sequence[int]) -> float:
            key = tuple(sorted(members))
            found = cache.get(key)
            if found is None:
                found = self.fit_group(
                    key, sweeps=sweeps, improvement_floor=improvement_floor
                )[0]
                cache[key] = found
            return found

        journal: list[dict[str, Any]] = []
        while len(clusters) > int(groups):
            best: tuple[float, int, int] | None = None
            for left in range(len(clusters)):
                for right in range(left + 1, len(clusters)):
                    union = sorted(clusters[left] + clusters[right])
                    increase = (
                        cost(union) - cost(clusters[left]) - cost(clusters[right])
                    )
                    candidate = (increase, clusters[left][0], clusters[right][0])
                    if best is None or candidate < best:
                        best = candidate
                        best_pair = (left, right)
            assert best is not None
            left, right = best_pair
            union = sorted(clusters[left] + clusters[right])
            journal.append(
                {
                    "merged": [list(clusters[left]), list(clusters[right])],
                    "objective_increase": float(best[0]),
                    "clusters_after": len(clusters) - 1,
                }
            )
            clusters = [
                entry
                for position, entry in enumerate(clusters)
                if position not in (left, right)
            ]
            clusters.append(union)
            clusters.sort(key=lambda entry: entry[0])
        return clusters, journal


def _relative_l2(candidate: Any, reference: Any) -> float:
    torch = _torch()
    numerator = float(torch.linalg.matrix_norm(candidate - reference).item())
    denominator = float(torch.linalg.matrix_norm(reference).item())
    if denominator <= 0.0 or not math.isfinite(denominator):
        return float("nan")
    return numerator / denominator


def _head_dimension(request: Any, spec: Any, output_width: int) -> int:
    """The head width, read off the parent's own per-head RMSNorm gain.

    Identical policy to `matrix_technique_head_drop._head_dimension`, restated
    rather than imported so the two C10 kernels do not acquire a dependency on
    each other's internals: the checkpoint decides, the knob only covers a
    parent that carries no such tensor, and a knob never overrides one that
    does.
    """

    from .matrix_technique_registry import CellTechniqueError

    key = f"model.layers.{int(spec.layer_index)}.self_attn.q_layernorm.weight"
    gain = request.parent_weights.get(key)
    if gain is not None:
        head_dim = int(gain.shape[0])
    else:
        declared = request.knob("head_dim", HEAD_MERGE_HEAD_DIM)
        head_dim = int(declared)
    if head_dim <= 0 or output_width % head_dim != 0:
        raise CellTechniqueError(
            f"C10 resolved head_dim {head_dim}, which does not divide the "
            f"{output_width}-wide query projection."
        )
    return head_dim


def c10_head_merge_technique(request: Any) -> Any:
    """C10 under T7: merge ``H`` heads onto ``g`` shared bases by Procrustes."""

    from .hf_lfm2_structural_residual_attention_fold import _fit_linear
    from .matrix_technique_registry import (
        CellTechniqueError,
        CellTechniqueResult,
        sealable_report,
        sealable_solve_number,
    )

    torch = _torch()
    spec = request.component
    if spec.component_id != C10_COMPONENT_ID or spec.contraction_axis != "heads":
        raise CellTechniqueError(
            "c10_head_merge_technique is the attn_per_head kernel only; "
            f"{spec.component_id} has axis {spec.contraction_axis!r}."
        )
    if spec.layer_index is None:
        raise CellTechniqueError("C10 needs a layer index to name its tensors.")
    if spec.input_endpoint is None or spec.write_endpoint is None:
        raise CellTechniqueError("C10 has no sealed input/write endpoints.")

    layer = int(spec.layer_index)
    keys = {
        "q": f"model.layers.{layer}.self_attn.q_proj.weight",
        "k": f"model.layers.{layer}.self_attn.k_proj.weight",
        "v": f"model.layers.{layer}.self_attn.v_proj.weight",
        "o": f"model.layers.{layer}.self_attn.out_proj.weight",
    }
    for name in keys.values():
        if name not in spec.weight_keys:
            raise CellTechniqueError(
                f"C10 at {spec.site} does not own {name}; the component's key "
                f"set is {list(spec.weight_keys)}."
            )
    parent = {
        role: request.parent_weights[name].detach().to(dtype=torch.float64)
        for role, name in keys.items()
    }
    hidden = int(parent["q"].shape[1])
    q_width = int(parent["q"].shape[0])
    kv_width = int(parent["k"].shape[0])
    if int(parent["v"].shape[0]) != kv_width or int(parent["o"].shape[1]) != q_width:
        raise CellTechniqueError(
            "C10's four projections do not share one head layout: "
            f"q {tuple(parent['q'].shape)}, k {tuple(parent['k'].shape)}, "
            f"v {tuple(parent['v'].shape)}, o {tuple(parent['o'].shape)}."
        )
    head_dim = _head_dimension(request, spec, q_width)
    if kv_width % head_dim != 0:
        raise CellTechniqueError(
            f"head_dim {head_dim} does not divide the {kv_width}-wide kv "
            "projections."
        )
    query_heads = q_width // head_dim
    kv_heads = kv_width // head_dim

    query_groups, kv_groups = head_merge_group_plan(
        int(request.level_width),
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    if int(request.level_width) % head_dim != 0:
        raise CellTechniqueError(
            f"Level {request.cell.level} resolves to {int(request.level_width)} "
            f"channels, which is not a whole number of {head_dim}-wide query "
            "group bases; C10's level axis is head counts and this kernel will "
            "not round to one."
        )
    if query_groups >= query_heads and kv_groups >= kv_heads:
        raise CellTechniqueError(
            f"Level {request.cell.level} buys {query_groups} query groups and "
            f"{kv_groups} kv groups, which is one basis per head and not a "
            "contraction."
        )

    fit_context = request.evidence.fit_rows(spec.input_endpoint).to(dtype=torch.float64)
    fit_write = request.evidence.fit_rows(spec.write_endpoint).to(dtype=torch.float64)
    held_context = request.evidence.heldout_rows(spec.input_endpoint).to(
        dtype=torch.float64
    )
    held_write = request.evidence.heldout_rows(spec.write_endpoint).to(
        dtype=torch.float64
    )
    if int(fit_context.shape[1]) != q_width:
        raise CellTechniqueError(
            f"C10's input endpoint {spec.input_endpoint!r} is "
            f"{int(fit_context.shape[1])} wide; the O projection reads "
            f"{q_width} concatenated head channels."
        )

    tolerance = float(
        request.knob(
            "head_merge_parent_tolerance",
            request.knob("calibration_tolerance", 5e-3),
        )
    )
    parent_relative = _relative_l2(held_context @ parent["o"].T, held_write)
    request.log(
        f"C10 {request.cell.text}: {query_heads}q/{kv_heads}kv heads of "
        f"{head_dim} -> {query_groups}/{kv_groups} groups; parent write replay "
        f"relL2 {parent_relative:.6g}"
    )
    if not math.isfinite(parent_relative) or parent_relative > tolerance:
        raise CellTechniqueError(
            f"C10 cannot reproduce {spec.write_endpoint!r} from "
            f"{spec.input_endpoint!r} through the parent O projection: relL2 "
            f"{parent_relative:.6g} against a {tolerance:g} tolerance."
        )

    # -- the evidence weights ---------------------------------------------
    rows = int(fit_context.shape[0])
    context_blocks = _blocks(fit_context, query_heads, head_dim, axis=1)
    head_energy = [
        float((block * block).sum().item()) / float(max(rows, 1))
        for block in context_blocks
    ]
    mean_energy = sum(head_energy) / float(query_heads)
    if mean_energy <= 0.0 or not math.isfinite(mean_energy):
        raise CellTechniqueError(
            "The sealed attention context carries no energy at this site; the "
            "per-head weighting would be undefined."
        )
    query_weights = [value / mean_energy for value in head_energy]
    per_kv = query_heads // kv_heads
    kv_energy = [
        sum(head_energy[j * per_kv : (j + 1) * per_kv]) for j in range(kv_heads)
    ]
    mean_kv = sum(kv_energy) / float(kv_heads)
    kv_weights = [value / mean_kv for value in kv_energy]

    sweeps = int(request.knob("head_merge_als_sweeps", 24))
    floor = float(request.knob("head_merge_improvement_floor", 1e-10))
    linkage_sweeps = int(request.knob("head_merge_linkage_sweeps", 4))

    query_family = _MergePair(
        _blocks(parent["q"], query_heads, head_dim, axis=0),
        _blocks(parent["o"], query_heads, head_dim, axis=1),
        query_weights,
        partner_transposed=True,
    )
    kv_family = _MergePair(
        _blocks(parent["k"], kv_heads, head_dim, axis=0),
        _blocks(parent["v"], kv_heads, head_dim, axis=0),
        kv_weights,
        partner_transposed=False,
    )

    def solve_family(
        family: _MergePair, groups: int
    ) -> tuple[list[list[int]], list[Any], list[Any], list[Any], dict[str, Any]]:
        partition, journal = family.cluster(
            groups, sweeps=linkage_sweeps, improvement_floor=floor
        )
        rotations: list[Any] = [None] * family.count
        primary_bases: list[Any] = []
        partner_bases: list[Any] = []
        traces: list[list[float]] = []
        objective = 0.0
        for members in partition:
            value, group_rotations, primary, partner, trace = family.fit_group(
                members, sweeps=sweeps, improvement_floor=floor
            )
            for position, index in enumerate(members):
                rotations[index] = group_rotations[position]
            primary_bases.append(primary)
            partner_bases.append(partner)
            traces.append([float(item) for item in trace])
            objective += value
        energy = family.energy()
        telemetry = {
            "partition": [list(members) for members in partition],
            "merge_journal": journal,
            "objective": float(objective),
            "objective_energy": float(energy),
            "objective_relative": (
                None if energy <= 0.0 else float(objective) / float(energy)
            ),
            "objective_trace": traces,
            "monotone": all(
                all(
                    trace[step + 1] <= trace[step] + 1e-12 * max(abs(trace[step]), 1.0)
                    for step in range(len(trace) - 1)
                )
                for trace in traces
            ),
        }
        return partition, rotations, primary_bases, partner_bases, telemetry

    query_partition, query_rotations, query_bases, o_bases_internal, query_telemetry = (
        solve_family(query_family, query_groups)
    )
    # The query family carries its partner as ``O.T`` so both Procrustes terms
    # read ``block @ basis.T``; the installed tensor wants ``Ob`` the way
    # ``out_proj`` is shaped, so the orientation is undone exactly once, here.
    o_bases = [basis.T.contiguous() for basis in o_bases_internal]
    kv_partition, kv_rotations, k_bases, v_bases, kv_telemetry = solve_family(
        kv_family, kv_groups
    )

    group_of_head = {
        head: position
        for position, members in enumerate(query_partition)
        for head in members
    }
    group_of_kv = {
        head: position
        for position, members in enumerate(kv_partition)
        for head in members
    }

    # -- the O bases, re-solved against the sealed write -------------------
    #
    # context @ o_child.T = sum_a (sum_{h in a} context_h @ R_h) @ Ob[a].T, so
    # with the alignments fixed the stored O bases are the coefficients of a
    # plain linear system between two sealed endpoints.  Solving them here is
    # head drop's absorption step with "the surviving columns" replaced by "the
    # shared bases", and it is the only place in this kernel where the fit is
    # against what the block WROTE rather than against what it stores.
    def design(context_rows: Any) -> Any:
        blocks = _blocks(context_rows, query_heads, head_dim, axis=1)
        columns = []
        for members in query_partition:
            accumulated = None
            for head in members:
                term = blocks[head] @ query_rotations[head]
                accumulated = term if accumulated is None else accumulated + term
            columns.append(accumulated)
        return torch.cat(columns, dim=1).contiguous()

    ridge = float(request.knob("head_merge_ridge", 1e-6))
    max_delta = request.knob("max_prior_delta_ratio", 1.0)
    procrustes_prior = torch.cat(o_bases, dim=1).contiguous()
    fit_design = design(fit_context)
    solved, report = _fit_linear(
        fit_design,
        fit_write,
        role=f"{spec.name}_c10_head_merge_o_bases",
        ridge=ridge,
        prior=procrustes_prior,
        max_prior_delta_ratio=max_delta,
        solver_diagnostics=True,
        relative_ridge_lambda=request.knob("relative_ridge_lambda", None),
    )

    # The refit is not adopted on faith.  A least-squares solve is *guaranteed*
    # to beat its prior on the rows it was fit on and is not guaranteed to beat
    # it anywhere else -- the O design has ``g_q * head_dim`` columns against a
    # 1,024-wide target, and a merge with few groups is exactly the regime where
    # that gap opens.  The choice between the Procrustes bases and the refit is
    # therefore made on an INNER split of the fit rows, never on the held-out
    # rows: those are the campaign's boundary score and selecting on them would
    # be scoring a choice against the evidence that chose it.
    fraction = float(request.knob("head_merge_inner_fit_fraction", 0.8))
    columns = int(fit_design.shape[1])
    inner = int(rows * fraction)
    inner_applicable = inner > columns and rows - inner >= 8
    inner_refit = None
    inner_prior = None
    if inner_applicable:
        inner_solved, _inner_report = _fit_linear(
            fit_design[:inner],
            fit_write[:inner],
            role=f"{spec.name}_c10_head_merge_o_bases_inner",
            ridge=ridge,
            prior=procrustes_prior,
            max_prior_delta_ratio=max_delta,
            solver_diagnostics=False,
            relative_ridge_lambda=request.knob("relative_ridge_lambda", None),
        )
        validation_design = fit_design[inner:]
        validation_write = fit_write[inner:]
        inner_refit = _relative_l2(
            validation_design @ inner_solved.T, validation_write
        )
        inner_prior = _relative_l2(
            validation_design @ procrustes_prior.T, validation_write
        )
        adopt = math.isfinite(inner_refit) and inner_refit < inner_prior
    else:
        adopt = True
    installed_o_weight = solved if adopt else procrustes_prior
    refit_o_bases = [
        installed_o_weight[
            :, position * head_dim : (position + 1) * head_dim
        ].contiguous()
        for position in range(query_groups)
    ]

    def materialize(
        o_group_bases: Sequence[Any],
    ) -> dict[str, Any]:
        child_q = torch.zeros_like(parent["q"])
        child_o = torch.zeros_like(parent["o"])
        child_k = torch.zeros_like(parent["k"])
        child_v = torch.zeros_like(parent["v"])
        for head in range(query_heads):
            group = group_of_head[head]
            rotation = query_rotations[head]
            child_q[head * head_dim : (head + 1) * head_dim, :] = (
                rotation @ query_bases[group]
            )
            child_o[:, head * head_dim : (head + 1) * head_dim] = (
                o_group_bases[group] @ rotation.T
            )
        for head in range(kv_heads):
            group = group_of_kv[head]
            rotation = kv_rotations[head]
            child_k[head * head_dim : (head + 1) * head_dim, :] = (
                rotation @ k_bases[group]
            )
            child_v[head * head_dim : (head + 1) * head_dim, :] = (
                rotation @ v_bases[group]
            )
        return {"q": child_q, "k": child_k, "v": child_v, "o": child_o}

    child = materialize(refit_o_bases)
    procrustes_child = materialize(o_bases)

    # -- the identity gate -------------------------------------------------
    #
    # One basis per head, and the O refit aimed at the parent's OWN write:
    # every step of this pipeline must then be an identity, and the deviation
    # over all four tensors is the measurement of whether it is.
    identity_deviation = _identity_gate(
        request=request,
        spec=spec,
        parent=parent,
        fit_context=fit_context,
        head_dim=head_dim,
        query_heads=query_heads,
        kv_heads=kv_heads,
        query_weights=query_weights,
        kv_weights=kv_weights,
        sweeps=sweeps,
        floor=floor,
        ridge=ridge,
        max_delta=max_delta,
        fit_linear=_fit_linear,
    )

    stored = head_merge_stored_parameters(
        hidden_size=hidden,
        head_dim=head_dim,
        query_heads=query_heads,
        kv_heads=kv_heads,
        query_groups=query_groups,
        kv_groups=kv_groups,
    )
    modelled = c10_head_merge_storage(
        spec, int(request.level_width), head_dim=head_dim
    )
    if stored != modelled:
        raise CellTechniqueError(
            f"The realized storage {stored} disagrees with the registered "
            f"storage model's {modelled}; one of them is wrong and a receipt "
            "may not seal either until they agree."
        )

    installed = {
        keys["q"]: child["q"],
        keys["k"]: child["k"],
        keys["v"]: child["v"],
        keys["o"]: child["o"],
    }

    held_design = design(held_context)
    telemetry = report.as_dict()
    conditioning = telemetry.get("solver_conditioning", {})
    clamp = telemetry.get("prior_delta_clamp", {})
    relative = telemetry.get("relative_ridge", {})
    condition = conditioning.get("system_condition_number")
    condition_reported = condition is not None and math.isfinite(float(condition))
    report_body, report_non_finite = sealable_report(telemetry)

    solve: dict[str, Any] = {
        "solver_path": conditioning.get("solver_path") or "unreported",
        "ridge_relative": float(relative.get("relative_ridge_lambda") or 0.0),
        "condition_number": sealable_solve_number(condition),
        "condition_cap_saturated": not condition_reported,
        "condition_number_unbounded_reason": (
            None
            if condition_reported
            else ("solver-reported-none" if condition is None else "singular-system")
        ),
        "prior_norm": float(telemetry.get("prior_norm", 0.0)),
        "correction_norm": float(telemetry.get("correction_norm", 0.0)),
        "requested_clamp_ratio": sealable_solve_number(
            clamp.get("requested_prior_delta_ratio")
        ),
        "allowed_clamp_ratio": sealable_solve_number(
            clamp.get("allowed_prior_delta_ratio")
        ),
        "clamp_engaged": bool(clamp.get("clamp_engaged") or False),
        "omp": None,
        "prior_correction_report": report_body,
        "prior_correction_report_non_finite": report_non_finite,
        "contraction_axis": "heads",
        "mechanism": "M5-svd-procrustes-head-merge",
        "algorithm": (
            "objective-linkage-agglomeration + alternating orthogonal "
            "Procrustes over (q,o) and (k,v) head families + prior-anchored "
            "ridge refit of the O group bases against the sealed attention "
            "write (v1)"
        ),
        "parent_heads": query_heads,
        "parent_kv_heads": kv_heads,
        "head_dim": head_dim,
        "query_groups": query_groups,
        "kv_groups": kv_groups,
        "query_partition": [list(members) for members in query_partition],
        "kv_partition": [list(members) for members in kv_partition],
        "query_family": query_telemetry,
        "kv_family": kv_telemetry,
        "context_head_energy": [float(value) for value in head_energy],
        "query_head_weights": [float(value) for value in query_weights],
        "kv_head_weights": [float(value) for value in kv_weights],
        "weight_reconstruction_relative_l2": {
            role: float(_relative_l2(child[role], parent[role]))
            for role in ("q", "k", "v", "o")
        },
        "weight_reconstruction_relative_l2_procrustes_only": {
            role: float(_relative_l2(procrustes_child[role], parent[role]))
            for role in ("q", "k", "v", "o")
        },
        "write_fit_relative_l2": float(
            _relative_l2(fit_design @ installed_o_weight.T, fit_write)
        ),
        "write_fit_relative_l2_refit": float(
            _relative_l2(fit_design @ solved.T, fit_write)
        ),
        "write_fit_relative_l2_procrustes_only": float(
            _relative_l2(fit_design @ procrustes_prior.T, fit_write)
        ),
        "write_heldout_relative_l2": float(
            _relative_l2(held_design @ installed_o_weight.T, held_write)
        ),
        "write_heldout_relative_l2_refit": float(
            _relative_l2(held_design @ solved.T, held_write)
        ),
        "write_heldout_relative_l2_procrustes_only": float(
            _relative_l2(held_design @ procrustes_prior.T, held_write)
        ),
        "o_refit_adopted": bool(adopt),
        "o_refit_inner_split_applicable": bool(inner_applicable),
        "o_refit_inner_fit_rows": int(inner) if inner_applicable else None,
        "o_refit_inner_validation_relative_l2": (
            None if inner_refit is None else sealable_solve_number(inner_refit)
        ),
        "o_prior_inner_validation_relative_l2": (
            None if inner_prior is None else sealable_solve_number(inner_prior)
        ),
        "parent_matmul_relative_l2": sealable_solve_number(parent_relative),
        "parent_matmul_tolerance": tolerance,
        "identity_gate_max_abs_deviation": float(identity_deviation),
        "als_sweeps_cap": sweeps,
        "linkage_sweeps": linkage_sweeps,
        "improvement_floor": floor,
        "stored_parameter_count": int(stored),
        "parent_parameter_count": int(spec.parent_parameter_count),
        "storage_realized_ratio": float(stored) / float(spec.parent_parameter_count),
        "storage_model_agrees": True,
        "alignment_parameter_count": int(
            (query_heads + kv_heads) * head_dim * head_dim
        ),
        "basis_parameter_count": int(stored)
        - int((query_heads + kv_heads) * head_dim * head_dim),
    }

    def predict(rows: Any) -> Any:
        return rows.to(dtype=torch.float64) @ child["o"].T

    return CellTechniqueResult(
        installed_weights=installed,
        stored_parameter_count=int(stored),
        solve=solve,
        predict=predict,
        notes=(
            f"C10 head merge: {query_heads} query heads -> {query_groups} bases "
            f"{[list(members) for members in query_partition]}, {kv_heads} kv "
            f"heads -> {kv_groups} bases {[list(m) for m in kv_partition]}, one "
            f"{head_dim}x{head_dim} orthogonal alignment per head; stored "
            f"{stored} of {int(spec.parent_parameter_count)} "
            f"({float(stored) / float(spec.parent_parameter_count):.6f}x); "
            f"held-out write relL2 {solve['write_heldout_relative_l2']:.6g} "
            f"(Procrustes bases alone "
            f"{solve['write_heldout_relative_l2_procrustes_only']:.6g}; O refit "
            f"{'adopted' if adopt else 'REJECTED on the inner split'}); "
            f"identity gate {identity_deviation:.3g}."
        ),
    )


def _identity_gate(
    *,
    request: Any,
    spec: Any,
    parent: Mapping[str, Any],
    fit_context: Any,
    head_dim: int,
    query_heads: int,
    kv_heads: int,
    query_weights: Sequence[float],
    kv_weights: Sequence[float],
    sweeps: int,
    floor: float,
    ridge: float,
    max_delta: Any,
    fit_linear: Any,
) -> float:
    """Run the whole pipeline at one basis per head and measure the deviation.

    Not a re-derivation of the algebra in a cheaper form: the same clustering,
    the same alternation and the same refit, with the group counts set so that
    every group holds exactly one head and the refit's target is the parent's
    own write.  A pipeline that is not an identity there is not a merge, it is
    an edit, and the receipt says which by a number.
    """

    torch = _torch()
    query_family = _MergePair(
        _blocks(parent["q"], query_heads, head_dim, axis=0),
        _blocks(parent["o"], query_heads, head_dim, axis=1),
        query_weights,
        partner_transposed=True,
    )
    kv_family = _MergePair(
        _blocks(parent["k"], kv_heads, head_dim, axis=0),
        _blocks(parent["v"], kv_heads, head_dim, axis=0),
        kv_weights,
        partner_transposed=False,
    )
    deviation = 0.0
    rotations: list[Any] = []
    bases: list[Any] = []
    for head in range(query_heads):
        _value, group_rotations, primary, partner, _trace = query_family.fit_group(
            [head], sweeps=sweeps, improvement_floor=floor
        )
        rotation = group_rotations[0]
        rotations.append(rotation)
        bases.append(primary)
        deviation = max(
            deviation,
            float(
                (
                    rotation @ primary
                    - parent["q"][head * head_dim : (head + 1) * head_dim, :]
                )
                .abs()
                .max()
                .item()
            ),
        )
        # ``partner`` is O_h.T in this family's orientation.
        deviation = max(
            deviation,
            float(
                (
                    partner.T @ rotation.T
                    - parent["o"][:, head * head_dim : (head + 1) * head_dim]
                )
                .abs()
                .max()
                .item()
            ),
        )
    for head in range(kv_heads):
        _value, group_rotations, primary, partner, _trace = kv_family.fit_group(
            [head], sweeps=sweeps, improvement_floor=floor
        )
        rotation = group_rotations[0]
        for role, basis in (("k", primary), ("v", partner)):
            deviation = max(
                deviation,
                float(
                    (
                        rotation @ basis
                        - parent[role][head * head_dim : (head + 1) * head_dim, :]
                    )
                    .abs()
                    .max()
                    .item()
                ),
            )
    # ``o_h = Ob @ R.T`` inverts to ``Ob = o_h @ R``, which is the prior the
    # refit is anchored on.  At the gate every ``R`` is the identity and the two
    # forms coincide, which is exactly why writing the wrong one here would
    # never be caught by the gate it belongs to.
    prior = torch.cat(
        [
            (query_family.partner[head].T @ rotations[head])
            for head in range(query_heads)
        ],
        dim=1,
    ).contiguous()
    design = torch.cat(
        [
            fit_context[:, head * head_dim : (head + 1) * head_dim] @ rotations[head]
            for head in range(query_heads)
        ],
        dim=1,
    ).contiguous()
    solved, _report = fit_linear(
        design,
        fit_context @ parent["o"].T,
        role=f"{spec.name}_c10_head_merge_identity_gate",
        ridge=ridge,
        prior=prior,
        max_prior_delta_ratio=max_delta,
        solver_diagnostics=False,
    )
    child_o = torch.zeros_like(parent["o"])
    for head in range(query_heads):
        child_o[:, head * head_dim : (head + 1) * head_dim] = (
            solved[:, head * head_dim : (head + 1) * head_dim] @ rotations[head].T
        )
    return max(deviation, float((child_o - parent["o"]).abs().max().item()))


def _register_head_merge_technique() -> None:
    """Wire the M5 kernel into the harness dispatch table at import time."""

    try:
        from .matrix_technique_registry import register
    except ImportError:  # pragma: no cover - the harness lane may not be present
        return
    register(
        HEAD_MERGE_TECHNIQUE_ID,
        c10_head_merge_technique,
        component=C10_COMPONENT_ID,
        replace=True,
    )


_register_head_merge_technique()


def describe_head_merge_technique() -> Mapping[str, Any]:
    """Serializable inventory of what this kernel wires, for a cell receipt."""

    return {
        "schema": "neural-forge-matrix-technique-head-merge-v1",
        "technique": HEAD_MERGE_TECHNIQUE_ID,
        "component": C10_COMPONENT_ID,
        "mechanism": "M5",
        "contraction_axis": "heads",
        "specification": (
            "reports/matrix-campaign-20260810/wave12/NOTES-MX-P.md §8, with the "
            "kv group count made integral (ceil) -- g_q=3 stores 0.239583x, not "
            "§8's 0.21875x"
        ),
        "grouping": "agglomerative, linkage = the converged merge objective",
        "alignment": "closed-form orthogonal Procrustes (SVD), one per head",
        "refit": "prior-anchored ridge over the O group bases vs the sealed write",
        "parent_gate": "context @ out_proj.T against the sealed attention write",
        "identity_gate": "one basis per head must reproduce all four projections",
        "head_dim_constant": HEAD_MERGE_HEAD_DIM,
    }
