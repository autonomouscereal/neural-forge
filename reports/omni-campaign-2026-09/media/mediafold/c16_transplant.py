"""C16 DiT block transplant: residual charts, the five arms, and the 4x coreset.

Charts
------
``B_in : R^hidden -> R^2048`` and ``B_out : R^2048 -> R^hidden`` are fitted closed form
on PAIRED rows -- host residual-stream rows and parent block-input rows taken at the
same media frame and the same sigma. Four methods:

``ridge``       prior-free ridge in fp64 (the default; works at any width pair)
``procrustes``  orthogonal Procrustes, equal widths only (the isometry control)
``svd``         one global orthonormal chart from the pooled parent rows (the
                ``run_chimera_omni_toy_fold.build_residual_chart`` control)
``random``      gaussian charts -- the ARM control, not a fit

Arms (the T1 protocol)
----------------------
``replace``      ``x + B_out(block(B_in(x), temb, ctx) - B_in(x))``
``insert``       the same delta, added as an EXTRA block rather than replacing one
``control``      a random-init block of identical shape through the same FITTED charts
``identity``     charts only, no block: ``B_out(B_in(x))`` -- prices chart round-trip
``random_chart`` the real block through RANDOM charts -- prices the chart fit

Only ``replace`` and ``insert`` are candidate folds; the other three exist so a win can
be attributed. Nothing here is a gate: the gate is Cyril's blind score on decoded audio.

``control`` is a NORM-MATCHED random block, and the two words matter. Re-initialising
every parameter with ``dim < 2`` to zero also zeroes the five RMSNorm weights
(self_attn_norm, cross_attn_norm, mlp_norm, q_norm, k_norm); with a zero norm weight the
normalised activations are identically zero and the block emits essentially nothing, so
the arm measures "a dead block writes nothing" rather than "a random block of this shape
carries nothing". 1-D parameters therefore keep their PARENT values, only the 2-D
projections are redrawn, and the resulting delta is rescaled to the parent block's own
delta norm on the same rows (``match_delta_to``), so replace-vs-control is a comparison
of CONTENT at equal write strength.

RoPE: an ``AceStepDiTLayer`` takes ``position_embeddings`` and block 12 is a
``sliding_attention`` layer whose self-attention is meaningless without its rotary phase.
:class:`TransplantedBlock` therefore builds the parent's own rope table for the host
sequence length (``rope_tables(L, head_dim, rope_theta, ...)``, identical to
``AceStepDiTModel.block_inputs``) and passes it in. Running the block without it -- which
this class used to do -- makes every arm score a different function of the same weights
than the parent reference it is compared against.

4x channel coreset
------------------
``coreset_block_mlp`` narrows the transplanted block's SwiGLU from 6144 to ``keep``
channels: ``gate_proj`` / ``up_proj`` rows are VERBATIM (LFM2 law 40/42 -- never SVD on
a gated FFN), ``down_proj`` is refit by prior-anchored ridge with the update_scale veto.
6144 -> 1536 is 4x on the MLP component and about 2.1x on the whole block; reaching 4x
on the WHOLE block additionally needs the kv-head merge (rung 2, not built here).
Per-component and whole-block ratios are always reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn

from .ace_dit import MLP, AceStepDiTLayer, rope_tables
from .linalg import (
    LinearMap,
    compression_receipt,
    fit_ridge_lambda_ladder,
    law_abiding_solve,
    orthogonal_procrustes,
    pivoted_columns,
    random_chart,
    svd_chart,
    write_scores,
)

ARMS = ("replace", "insert", "control", "identity", "random_chart")


# --------------------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------------------


@dataclass
class ChartPair:
    chart_in: LinearMap  # hidden -> parent width
    chart_out: LinearMap  # parent width -> hidden
    method: str
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def parameters(self) -> int:
        return int(
            self.chart_in.weight.numel()
            + self.chart_in.bias.numel()
            + self.chart_out.weight.numel()
            + self.chart_out.bias.numel()
        )

    def state(self) -> dict[str, torch.Tensor]:
        return {
            "chart_in.weight": self.chart_in.weight.to(torch.float32),
            "chart_in.bias": self.chart_in.bias.to(torch.float32),
            "chart_out.weight": self.chart_out.weight.to(torch.float32),
            "chart_out.bias": self.chart_out.bias.to(torch.float32),
        }


def _as_map(weight: torch.Tensor) -> LinearMap:
    return LinearMap(weight=weight.to(torch.float64), bias=torch.zeros(weight.shape[0], dtype=torch.float64))


def fit_charts(
    host_rows: torch.Tensor,
    parent_rows: torch.Tensor,
    *,
    method: str = "ridge",
    seed: int = 0,
    holdout_fraction: float = 0.2,
) -> ChartPair:
    """Fit ``B_in`` and ``B_out`` on paired residual rows.

    Rows are split before fitting. ``chart_out`` carries ``parent_width + 1`` free
    parameters per output coordinate, so scoring the round-trip on the rows it was fitted
    on is optimistic by an unknown margin; ``roundtrip_scores`` / ``forward_scores`` are
    therefore HELD-OUT numbers and the in-sample pair is reported beside them.
    """

    host = host_rows.detach().to(torch.float64)
    parent = parent_rows.detach().to(torch.float64)
    if host.shape[0] != parent.shape[0]:
        raise ValueError("chart fitting needs paired rows with equal counts")

    count = int(host.shape[0])
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(count, generator=generator)
    held = max(1, int(round(float(holdout_fraction) * count))) if holdout_fraction > 0 else 0
    hold_index, fit_index = order[:held], order[held:]
    if int(fit_index.numel()) == 0:
        raise ValueError("holdout_fraction leaves no rows to fit on")
    host_fit, parent_fit = host[fit_index], parent[fit_index]

    if method == "ridge":
        chart_in = fit_ridge_lambda_ladder(host_fit, parent_fit, seed=seed)
        chart_out = fit_ridge_lambda_ladder(parent_fit, host_fit, seed=seed)
        notes = {"in": chart_in.notes, "out": chart_out.notes}
    elif method == "procrustes":
        if host.shape[1] != parent.shape[1]:
            raise ValueError("orthogonal Procrustes needs equal widths; use method='ridge'")
        weight_in, notes_in = orthogonal_procrustes(host_fit, parent_fit)
        chart_in = _as_map(weight_in)
        chart_out = _as_map(weight_in.transpose(0, 1).contiguous())
        notes = {"in": notes_in, "out": {"transpose_of_in": True}}
    elif method == "svd":
        chart, chart_notes = svd_chart(parent_fit, host.shape[1])
        chart_in = _as_map(chart.transpose(0, 1).contiguous())  # hidden -> parent
        chart_out = _as_map(chart.contiguous())  # parent -> hidden
        notes = {"chart": chart_notes}
    elif method == "random":
        chart_in = _as_map(random_chart(parent.shape[1], host.shape[1], seed=seed))
        chart_out = _as_map(random_chart(host.shape[1], parent.shape[1], seed=seed + 1))
        notes = {"random": True, "seed": int(seed)}
    else:
        raise ValueError(f"unknown chart method {method!r}")

    pair = ChartPair(chart_in=chart_in, chart_out=chart_out, method=method, notes=notes)
    scored_index = hold_index if held else fit_index
    host_scored, parent_scored = host[scored_index], parent[scored_index]
    pair.notes["roundtrip_scores"] = write_scores(chart_out.apply(chart_in.apply(host_scored)), host_scored)
    pair.notes["forward_scores"] = write_scores(chart_in.apply(host_scored), parent_scored)
    pair.notes["in_sample_roundtrip_scores"] = write_scores(
        chart_out.apply(chart_in.apply(host_fit)), host_fit
    )
    pair.notes["in_sample_forward_scores"] = write_scores(chart_in.apply(host_fit), parent_fit)
    pair.notes["split"] = {
        "rows": count,
        "fit_rows": int(fit_index.numel()),
        "holdout_rows": int(held),
        "scored_on": "holdout" if held else "fit (holdout_fraction=0)",
        "split_by": "row",
    }
    pair.notes["parameters"] = pair.parameters
    return pair


# --------------------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------------------


class TransplantedBlock(nn.Module):
    """A parent DiT block wired into a host trunk through residual charts.

    The block itself is FROZEN (``requires_grad_(False)``): a fold installs the parent's
    computation, it does not retrain it. The host trunk trains around it.
    """

    def __init__(
        self,
        block: AceStepDiTLayer | None,
        charts: ChartPair,
        *,
        arm: str = "replace",
        temb_source: nn.Module | None = None,
        dtype: torch.dtype = torch.float32,
        rope_theta: float = 1.0e6,
        delta_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        self.block = block
        if block is not None:
            block.requires_grad_(False)
        self.rope_theta = float(rope_theta)
        self.head_dim = None if block is None else int(block.self_attn.head_dim)
        self.delta_scale = float(delta_scale)
        self.register_buffer("chart_in_weight", charts.chart_in.weight.to(dtype))
        self.register_buffer("chart_in_bias", charts.chart_in.bias.to(dtype))
        self.register_buffer("chart_out_weight", charts.chart_out.weight.to(dtype))
        self.register_buffer("chart_out_bias", charts.chart_out.bias.to(dtype))
        self.temb_source = temb_source

    def project_in(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.chart_in_weight.to(x.dtype).transpose(0, 1) + self.chart_in_bias.to(x.dtype)

    def project_out(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.chart_out_weight.to(x.dtype).transpose(0, 1) + self.chart_out_bias.to(x.dtype)

    def position_embeddings(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The parent's own rope table for this sequence length.

        Same call the parent makes in ``AceStepDiTModel.block_inputs``; built here rather
        than passed in so a deployment of this class can never silently run the block
        without its rotary phase.
        """

        if self.head_dim is None:
            return None
        return rope_tables(int(rows.shape[1]), int(self.head_dim), self.rope_theta, rows.device, rows.dtype)

    def block_delta(
        self,
        projected: torch.Tensor,
        temb: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The parent-space residual write, BEFORE ``delta_scale``."""

        if self.block is None:
            return torch.zeros_like(projected)
        if temb is None:
            temb = torch.zeros(
                projected.shape[0], 6, projected.shape[-1], device=projected.device, dtype=projected.dtype
            )
        if position_embeddings is None:
            position_embeddings = self.position_embeddings(projected)
        return (
            self.block(
                projected,
                temb,
                encoder_hidden_states=context,
                position_embeddings=position_embeddings,
            )
            - projected
        )

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        projected = self.project_in(x)
        if self.arm == "identity" or self.block is None:
            return self.project_out(projected)
        delta = self.block_delta(projected, temb, context, position_embeddings)
        if self.delta_scale != 1.0:
            delta = delta * self.delta_scale
        # the chart's bias belongs to the residual stream, not to the delta: a zero delta
        # must leave the host untouched, so B_out is applied without its bias here.
        return x + delta @ self.chart_out_weight.to(x.dtype).transpose(0, 1)

    @property
    def frozen_parameters(self) -> int:
        block_parameters = 0 if self.block is None else sum(int(p.numel()) for p in self.block.parameters())
        chart_parameters = int(
            self.chart_in_weight.numel()
            + self.chart_in_bias.numel()
            + self.chart_out_weight.numel()
            + self.chart_out_bias.numel()
        )
        return block_parameters + chart_parameters


def build_arm(
    arm: str,
    block: AceStepDiTLayer,
    fitted: ChartPair,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    rope_theta: float = 1.0e6,
    norm_match_rows: torch.Tensor | None = None,
    temb: torch.Tensor | None = None,
    context: torch.Tensor | None = None,
) -> TransplantedBlock:
    """Materialise one arm from a fitted chart pair and a loaded parent block.

    ``norm_match_rows`` (with ``temb`` / ``context``) makes ``control`` a write-strength
    matched control: its delta is rescaled to the parent block's delta norm on those same
    rows, so replace-vs-control is a contrast of CONTENT and not of magnitude.
    """

    import copy

    if arm == "replace" or arm == "insert":
        return TransplantedBlock(block, fitted, arm=arm, dtype=dtype, rope_theta=rope_theta)
    if arm == "identity":
        return TransplantedBlock(None, fitted, arm="identity", dtype=dtype, rope_theta=rope_theta)
    if arm == "control":
        blank = copy.deepcopy(block)
        generator = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            parent_state = dict(block.state_dict())
            for name, parameter in blank.named_parameters():
                if parameter.dim() >= 2:
                    scale = (parameter.shape[-1]) ** -0.5
                    parameter.copy_(
                        torch.randn(parameter.shape, generator=generator, dtype=torch.float32).to(parameter.dtype)
                        * scale
                    )
                else:
                    # 1-D parameters are the five RMSNorm gains (self_attn_norm,
                    # cross_attn_norm, mlp_norm, q_norm, k_norm). Zeroing a norm gain
                    # makes every normalised activation identically zero -- a DEAD block,
                    # which attributes nothing. They keep the PARENT values; only the 2-D
                    # projections are redrawn, which is what "a random block of this
                    # shape" has to mean for the contrast to be about content.
                    parameter.copy_(parent_state[name].to(parameter.dtype))
        installed = TransplantedBlock(blank, fitted, arm="control", dtype=dtype, rope_theta=rope_theta)
        if norm_match_rows is not None:
            reference = TransplantedBlock(block, fitted, arm="replace", dtype=dtype, rope_theta=rope_theta)
            with torch.no_grad():
                projected = installed.project_in(norm_match_rows)
                parent_norm = reference.block_delta(projected, temb, context).norm()
                control_norm = installed.block_delta(projected, temb, context).norm()
            installed.delta_scale = float((parent_norm / control_norm.clamp_min(1.0e-30)).item())
        return installed
    if arm == "random_chart":
        host_width = int(fitted.chart_in.weight.shape[1])
        parent_width = int(fitted.chart_in.weight.shape[0])
        scrambled = ChartPair(
            chart_in=_as_map(random_chart(parent_width, host_width, seed=seed + 101)),
            chart_out=_as_map(random_chart(host_width, parent_width, seed=seed + 102)),
            method="random",
            notes={"derived_from": fitted.method},
        )
        return TransplantedBlock(block, scrambled, arm="random_chart", dtype=dtype, rope_theta=rope_theta)
    raise ValueError(f"unknown arm {arm!r}")


def score_arm(
    installed: TransplantedBlock,
    host_rows: torch.Tensor,
    target_rows: torch.Tensor,
    *,
    temb: torch.Tensor | None = None,
    context: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Diagnostic: how close does the arm's output sit to the host's own next-layer rows.

    DIAGNOSTIC ONLY. The behavioural instrument is decoded audio scored blind; this
    orders arms, it never closes one.
    """

    with torch.no_grad():
        produced = installed(host_rows, temb, context)
    record: dict[str, Any] = {
        "arm": installed.arm,
        "scores": write_scores(produced.reshape(-1, produced.shape[-1]), target_rows.reshape(-1, target_rows.shape[-1])),
        "frozen_parameters": installed.frozen_parameters,
        "delta_scale": float(installed.delta_scale),
        "rope": installed.head_dim is not None,
    }
    if installed.arm == "insert":
        # honest label: at ONE transplanted block the insert and replace forward paths are
        # the same expression, so the two rows are identical by construction and must not
        # be read as two independent arms. They separate only once the host keeps its own
        # block as well, which this scaffold does not build.
        record["identical_forward_to"] = "replace"
    return record


# --------------------------------------------------------------------------------------
# 4x channel coreset
# --------------------------------------------------------------------------------------


@dataclass
class CoresetResult:
    block: AceStepDiTLayer
    selection: torch.Tensor
    receipt: dict[str, Any]


def coreset_block_mlp(
    block: AceStepDiTLayer,
    mlp_rows: torch.Tensor,
    *,
    keep: int,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    sequence_index: torch.Tensor | None = None,
    group_name: str = "sequence",
    row_membership: torch.Tensor | None = None,
    min_fit_rows_per_kept_channel: float = 2.0,
) -> CoresetResult:
    """Fold the block's SwiGLU to ``keep`` verbatim channels with a refit down-projection.

    ``mlp_rows`` are the block's own ``mlp_norm`` outputs captured on the calibration
    stimulus, [rows, hidden]. Selection ranks the gated activations; the refit target is
    the parent's exact MLP output on the same rows.

    Two things that used to be post-hoc error messages are now PRECONDITIONS, because a
    run that violates them still produces a plausible-looking receipt:

    * ``min_fit_rows_per_kept_channel`` -- the pivoted selection needs a Gram of rank far
      above ``keep`` and the ``down_proj`` refit has ``keep + 1`` free parameters per
      output row, so at ~1x rows the last pivots sit on numerically exhausted residual and
      the refit is at its interpolation threshold. Checked BEFORE pivoting, on the FIT
      rows, not after selection silently succeeds.
    * ``sequence_index`` -- neighbouring positions inside one group are strongly
      correlated, so a row-wise holdout leaks. When the caller supplies the per-row group
      id the split is by GROUP, the same discipline the C15 side applies by clip. A row
      split is still allowed (and labelled) when there is nothing to group by.
      ``group_name`` only names the grouping in the receipt ("clip", "sequence", ...).
    * ``row_membership`` -- the ids each row's SOURCE material actually draws from,
      ``[rows, k]``, ``-1`` for padding. Splitting by ``sequence_index`` is disjoint in
      the ids by construction, which proves nothing at all about the material: the defect
      this argument exists to catch is a group id that is not the id it is named after
      (the old code called the DiT batch row a holdout unit while a batch row spanned
      several clips). With membership supplied the guard re-derives the held-out ids from
      the realised split and refuses if a single FIT row draws on one of them, so it fires
      on exactly that mislabelling. ``group_name="clip"`` therefore REQUIRES it.

    What the group split buys is row-level disjointness of the source material. It is not
    statistical independence of the fit and holdout activations: if the rows were produced
    by a model whose receptive field spans several groups, a held-out row's activation
    still encodes fit material. Only the caller knows the receptive field, so the receipt
    says what was checked and never calls the holdout independent.
    """

    import copy

    with torch.no_grad():
        rows = mlp_rows.detach().to(torch.float32)
        hidden = block.mlp.hidden(rows.to(block.mlp.gate_proj.weight.dtype)).to(torch.float64)
        target = block.mlp(rows.to(block.mlp.gate_proj.weight.dtype)).to(torch.float64)

    intermediate = int(block.mlp.gate_proj.weight.shape[0])
    keep = int(keep)
    if not 0 < keep <= intermediate:
        raise ValueError(f"keep must be in (0, {intermediate}]")

    count = rows.shape[0]
    generator = torch.Generator().manual_seed(int(seed))
    if sequence_index is None:
        order = torch.randperm(count, generator=generator)
        held = max(1, int(round(holdout_fraction * count))) if holdout_fraction > 0 else 0
        fit_index, hold_index = order[held:], order[:held]
        split_by = "row (no sequence_index supplied; neighbouring positions leak)"
        group_counts = {
            "group_name": "row", "groups": int(count), "fit_groups": None, "holdout_groups": None,
        }
    else:
        sequences = sequence_index.reshape(-1).to(torch.long)
        if int(sequences.numel()) != count:
            raise ValueError(f"sequence_index has {int(sequences.numel())} entries for {count} rows")
        unique = torch.unique(sequences)
        shuffled = unique[torch.randperm(int(unique.numel()), generator=generator)]
        held_sequences = (
            max(1, int(round(holdout_fraction * int(unique.numel())))) if holdout_fraction > 0 else 0
        )
        hold_set = shuffled[:held_sequences]
        hold_mask = torch.isin(sequences, hold_set)
        hold_index = torch.nonzero(hold_mask).flatten()
        fit_index = torch.nonzero(~hold_mask).flatten()
        held = int(hold_index.numel())
        split_by = (
            f"{group_name} ({int(unique.numel())} {group_name}s, {int(held_sequences)} held out)"
        )
        fit_groups = torch.unique(sequences[fit_index])
        hold_groups = torch.unique(sequences[hold_index])
        group_counts = {
            "group_name": group_name,
            "groups": int(unique.numel()),
            "fit_groups": int(fit_groups.numel()),
            "holdout_groups": int(hold_groups.numel()),
        }
        # The check that can actually fail. Splitting `sequences` makes the two id sets
        # disjoint by construction, so comparing them proves nothing; what has to hold is
        # that no FIT row draws on material a held-out group owns. That is a statement
        # about `row_membership`, which is derived independently of the group ids.
        if row_membership is None:
            if group_name == "clip":
                raise ValueError(
                    "group_name='clip' without row_membership: the split would be disjoint in "
                    "the ids by construction and nothing would check that those ids really are "
                    "clips. Pass mediafold.c16_capture.token_clip_membership(...) for these rows."
                )
            group_counts["membership_checked"] = False
            group_counts["membership_note"] = (
                "no row_membership supplied: the split is disjoint in the group ids, which is "
                "true by construction and is not evidence that the ids partition the material"
            )
        else:
            membership = row_membership.reshape(count, -1).to(torch.long)
            held_material = torch.unique(membership[hold_index])
            held_material = held_material[held_material >= 0]
            touching = int(
                torch.isin(membership[fit_index], held_material).any(dim=-1).sum().item()
            )
            if touching:
                raise RuntimeError(
                    f"{touching} of {int(fit_index.numel())} fit rows draw on a held-out "
                    f"{group_name}: the group ids do not partition the material they are named "
                    "after, so the holdout score would be part fit score"
                )
            group_counts["membership_checked"] = True
            group_counts["membership_width"] = int(membership.shape[1])
            group_counts["fit_rows_drawing_on_a_held_out_group"] = 0

    required = int(round(float(min_fit_rows_per_kept_channel) * keep))
    if int(fit_index.numel()) < required:
        raise ValueError(
            f"{int(fit_index.numel())} fit rows for a {keep}-channel coreset: the Gram has rank "
            f"<= {int(fit_index.numel())} for {intermediate} columns and the down_proj refit has "
            f"{keep + 1} free parameters per output row, so the selection and the refit both sit "
            f"at the interpolation threshold. Capture at least {required} fit rows "
            f"({min_fit_rows_per_kept_channel}x keep) -- raise --batch / --length."
        )

    selection = pivoted_columns(hidden[fit_index], keep)
    if int(selection.numel()) < keep:
        raise RuntimeError(
            f"pivoted selection returned {int(selection.numel())} of {keep} channels: the Gram is "
            f"rank starved at {int(fit_index.numel())} fit rows."
        )
    selection = torch.sort(selection).values

    prior = block.mlp.down_proj.weight.detach().to(torch.float64)[:, selection]
    weight, notes = law_abiding_solve(hidden[fit_index][:, selection], target[fit_index], prior)

    child = copy.deepcopy(block)
    narrowed = MLP(int(block.mlp.gate_proj.weight.shape[1]), keep)
    with torch.no_grad():
        narrowed.gate_proj.weight.copy_(block.mlp.gate_proj.weight.detach()[selection])
        narrowed.up_proj.weight.copy_(block.mlp.up_proj.weight.detach()[selection])
        narrowed.down_proj.weight.copy_(weight.to(block.mlp.down_proj.weight.dtype))
    child.mlp = narrowed.to(device=block.mlp.gate_proj.weight.device, dtype=block.mlp.gate_proj.weight.dtype)

    with torch.no_grad():
        child_output = child.mlp(rows.to(child.mlp.gate_proj.weight.dtype)).to(torch.float64)

    parent_mlp_parameters = sum(int(p.numel()) for p in block.mlp.parameters())
    child_mlp_parameters = sum(int(p.numel()) for p in child.mlp.parameters())
    receipt = {
        "keep": keep,
        "intermediate": intermediate,
        "selection_sha256": None,
        "selected_channels": selection.tolist() if keep <= 64 else f"{keep} channels (list omitted)",
        "solve": notes,
        "component": compression_receipt(parent_mlp_parameters, child_mlp_parameters, "ace_dit_block.mlp"),
        "whole_block": compression_receipt(
            block.block_parameters(), child.block_parameters(), "ace_dit_block"
        ),
        "fit_scores": write_scores(child_output[fit_index], target[fit_index]),
        "holdout_scores": write_scores(child_output[hold_index], target[hold_index]) if held else {},
        "rows": int(count),
        "fit_rows": int(fit_index.numel()),
        "holdout_rows": int(held),
        "split_by": split_by,
        "split_groups": group_counts,
        "split_guarantee": (
            "row-level disjointness of the source material between fit and holdout"
            + (
                " (re-derived from row_membership, not from the group ids)"
                if group_counts.get("membership_checked")
                else ""
            )
            + ". This is NOT statistical independence of the two sets of activations: if the "
            "rows were produced by a model whose receptive field spans several groups, a "
            "held-out row's activation still encodes fit material. The caller's capture "
            "receipt is where the receptive field is recorded."
        ),
        "fit_rows_per_kept_channel": float(int(fit_index.numel()) / max(keep, 1)),
        "note": (
            "per-COMPONENT compression is the >=4x claim; the whole-block ratio is coverage "
            "driven and reaches 4x only with the kv-head merge (rung 2)"
        ),
    }
    from . import receipts as receipt_module

    receipt["selection_sha256"] = receipt_module.hash_tensor(selection)
    return CoresetResult(block=child, selection=selection, receipt=receipt)


def paired_rows_from_capture(
    host_rows: torch.Tensor,
    capture_records: dict[str, torch.Tensor],
    *,
    key: str = "x_in",
    limit: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Line host residual rows up with parent block rows, truncating to the common count."""

    parent = capture_records[key]
    parent_flat = parent.reshape(-1, parent.shape[-1]).to(torch.float64)
    host_flat = host_rows.reshape(-1, host_rows.shape[-1]).to(torch.float64)
    count = min(parent_flat.shape[0], host_flat.shape[0])
    if limit:
        count = min(count, int(limit))
    if count == 0:
        raise ValueError("no paired rows available")
    return host_flat[:count], parent_flat[:count]


def arm_table(
    block: AceStepDiTLayer,
    fitted: ChartPair,
    host_rows: torch.Tensor,
    target_rows: torch.Tensor,
    *,
    temb: torch.Tensor | None = None,
    context: torch.Tensor | None = None,
    arms: Sequence[str] = ARMS,
    seed: int = 0,
    rope_theta: float = 1.0e6,
    norm_match_control: bool = True,
) -> list[dict[str, Any]]:
    """Build and score every arm. Ordering only -- behaviour is the gate."""

    table: list[dict[str, Any]] = []
    for arm in arms:
        installed = build_arm(
            arm,
            block,
            fitted,
            seed=seed,
            rope_theta=rope_theta,
            norm_match_rows=host_rows if (arm == "control" and norm_match_control) else None,
            temb=temb,
            context=context,
        )
        table.append(score_arm(installed, host_rows, target_rows, temb=temb, context=context))
    return table
