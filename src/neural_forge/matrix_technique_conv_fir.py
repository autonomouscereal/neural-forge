#!/usr/bin/env python3
"""C04: the short-conv FIR tap contraction -- the campaign's first ``kernel`` axis.

`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` §1.1 gives C04 the contraction axis
``kernel`` -- "the FIR taps themselves ... not a matrix factorization" -- and
every technique the campaign owns refuses it by name.  Wave 1 sealed
`C04.T1.M16.L00` ``infra`` on a shape crash; wave 2 sealed it ``infra`` on
``T1's reference kernel contracts linear maps; C04 has axis 'kernel'``
(`NOTES-MX-F` §4).  Two waves, no evidence about the component.  This module is
the missing kernel, and it is deliberately small: keep ``t`` of the parent's
FIR taps and refit the survivors by least squares against the parent filter's
own response on the sealed evidence.

Mechanically, an LFM2 short conv is

    B, C, x = chunk(in_proj(u), 3)      # (rows, 3*W) -> three (rows, W) blocks
    bx      = B * x
    filt[t] = sum_i taps[:, i] * bx[t - (L-1-i)]     # depthwise, causal, L taps
    context = C * filt
    output  = out_proj(context)

so the component this cell contracts is ``taps`` (``conv.conv.weight``, shaped
``(channels, 1, L)``), the only tensor it installs.  ``in_proj``/``out_proj``
are read as read-only neighbours; the cell never installs them and never folds
two components.

Three things this kernel has to solve that no other campaign technique faces,
each recorded rather than absorbed:

1. **The level axis is not the axis the harness resolves.**
   `resolve_level_width` measures ``ComponentSpec.own_width``, which for a
   ``(channels, 1, taps)`` tensor is the *channel* count -- 1024, not 3.  The
   level's meaning at C04 is the tap count, so the kernel re-resolves the level
   against ``LEVELS`` on the tap axis and seals **both** numbers
   (``harness_level_width`` beside ``kept_taps``).  Re-resolving silently, or
   contracting the channel axis because that is what the harness handed over,
   would produce a confident cell measuring a different component.
   On a 3-tap parent the marginal family collapses: M16, M8 and M4 all floor to
   2 taps and only M2 reaches 1.  That is a property of a 3-tap filter, and the
   receipt says so instead of pretending four distinct levels exist.

2. **A FIR needs the token axis, and the evidence rows do not carry one.**
   `run_lfm2_matrix_cell.TraceSiteEvidence` concatenates every batch and phase
   into one ``(rows, width)`` matrix; row adjacency is real *inside* a captured
   sequence (prefill tokens 0..P-1 followed by the forced decode steps, whose
   conv state continues the prefill's) and meaningless *across* one.  Fitting
   taps across a sequence boundary mixes one prompt's tail into the next
   prompt's head -- a small number of rows, a completely wrong delay, and no
   crash to notice it by.  The row counts prove the blocks are not even uniform:
   the sealed base's FIT half is 4,751 rows, one short of 18x264.

   So the kernel **derives** the segmentation and then **proves** it.  Every
   captured sequence starts at the same first token, and a position-0 row is a
   function of that token alone, so the sealed rows that equal row 0 exactly are
   the sequence starts.  That is a hypothesis, and it is tested the way §2.1
   tests everything else -- by reproducing the parent: the full short-conv
   forward is replayed from ``operator.input`` under the derived segmentation
   and scored against the sealed ``operator.output``.  A wrong segmentation
   cannot pass that check, and the kernel additionally replays under the
   deliberately wrong "one long sequence" hypothesis and refuses unless the
   correct one is materially better -- a gate nothing can fail is not a gate.

   This is also the *first* parent-calibration measurement C04 has ever had.
   `run_lfm2_matrix_cell.NO_PARENT_MATMUL_PREDICTOR` declares the generic gate
   inapplicable for C04 because ``held_input @ parent_weight.T`` is not this
   component; the receipt keeps that declaration, and the number the generic
   gate could not produce is sealed under ``solve`` where it belongs.

3. **The refit target is the parent operator, not a sealed endpoint.**
   No sealed endpoint carries the FIR output on its own -- the cache for this
   component holds ``operator.input`` and ``operator.output``, and recovering
   ``filt`` from the output would mean inverting ``out_proj`` and dividing by a
   gate that passes through zero.  The parent's taps and the reconstructed
   ``bx`` give ``filt`` directly and exactly, so the least-squares target is the
   parent filter's response on the evidence's own input distribution.  The
   pleasant consequence is an *exact* identity gate: at zero drop the residual
   is identically zero, so the prior-anchored correction is identically zero and
   the refit returns the parent taps bit-for-bit.  Every C04 receipt seals that
   measurement (``identity_gate_max_abs_deviation``) rather than asserting it.

   Channels are weighted by the gate ``C`` by default.  ``filt`` is multiplied
   elementwise by ``C`` before it leaves the operator, so an unweighted fit
   optimizes a quantity the model never reads; ``conv_fir_channel_weighting =
   "none"`` recovers the plain objective for contrast.

Storage is the one place this component is unambiguously honest.  The stored
form is ``channels * kept_taps`` floats plus one global tap mask of ``L`` bits,
against a dense ``channels * L``: 2 of 3 taps stores 0.667x the parent tensor.
Every marginal T1/T2/T3 cell in this campaign stores *more* than its parent
(`NOTES-MX-E` §4.1); this one does not.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "C04_COMPONENT_ID",
    "c04_fir_tap_contraction_technique",
    "describe_conv_fir_technique",
]

C04_COMPONENT_ID = "C04"

#: The technique id this kernel answers for.  C04's cell in design §4.1 row 21
#: is a ``T1`` cell -- "conv kernel-width marginal (the shipped wall's
#: component, solver now exonerated)" -- and both sealed C04 instances carry
#: that id.  Registering under a *new* technique id would orphan those two
#: receipts from their successor instead of superseding them, so this kernel is
#: T1 pinned to C04, exactly as the standalone norm refit is T1 pinned to C14.
C04_TECHNIQUE_ID = "T1"

#: The estate's own tolerance for reproducing a short-conv's pre-output context
#: from its taps: `hf_lfm2_circuit_trace_cache._finalize_short_conv` accepts the
#: FIR reconstruction at ``rtol=2e-2, atol=2e-3`` against the captured context.
#: The replay below is that same reconstruction plus two projections, scored on
#: bf16-quantized rows, so it inherits that band rather than inventing a
#: tighter one.
_DEFAULT_RECONSTRUCTION_TOLERANCE = 2e-2

#: How much worse the deliberately-wrong "one long sequence" replay has to be
#: before the segmentation check counts as evidence.  Two rows per sequence
#: carry a wrong delay under that hypothesis, so the ratio is large when the
#: derived segmentation is right and ~1 when it is not.
_DEFAULT_DISCRIMINATION_RATIO = 3.0

#: Refuse to enumerate tap subsets beyond this many.  An LFM2 short conv has
#: three taps (three subsets of size two); the cap exists so a wider filter
#: fails loudly instead of silently running an exponential search inside a
#: 20-minute cell cap.
_MAX_TAP_SUBSETS = 256


def _torch() -> Any:
    import torch

    return torch


def _causal_segments(
    rows: Any,
    *,
    minimum_length: int,
    maximum_length: int,
) -> tuple[tuple[int, int], ...]:
    """Split flattened evidence rows back into the causal sequences that made them.

    Returns ``((start, stop), ...)`` half-open row spans.  Each span is one
    captured sequence: the prefill tokens in order, followed by the forced
    decode steps, whose conv state continues the prefill's.

    The rule is the one property of the capture that survives flattening: every
    captured sequence begins with the same first token, and the layer input at
    position 0 is a function of that token and nothing else, so it is bitwise
    identical in every sequence.  Rows equal to row 0 are therefore the sequence
    starts.  A coincidental interior duplicate would produce a span outside the
    length band, which is refused here, and would fail the replay gate above it
    anyway.
    """

    torch = _torch()
    from .matrix_technique_registry import CellTechniqueError

    if int(rows.shape[0]) < 2:
        raise CellTechniqueError("C04 needs at least two evidence rows.")
    anchor = rows[0]
    starts = torch.nonzero((rows == anchor).all(dim=1), as_tuple=False)
    indices = [int(value) for value in starts.reshape(-1).tolist()]
    if not indices or indices[0] != 0:
        raise CellTechniqueError(
            "C04 could not locate the sequence starts in the sealed rows: the "
            "first row is not the anchor its own equality test found."
        )
    if len(indices) < 2:
        raise CellTechniqueError(
            "C04 found a single sequence start in the sealed rows; the evidence "
            "concatenates many captured sequences and treating them as one "
            "would fit the FIR across prompt boundaries."
        )
    bounds = indices + [int(rows.shape[0])]
    spans: list[tuple[int, int]] = []
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        length = stop - start
        if length < minimum_length or length > maximum_length:
            raise CellTechniqueError(
                f"C04 derived a {length}-row sequence at row {start}, outside "
                f"the accepted [{minimum_length}, {maximum_length}] band; the "
                "row layout is not the one this kernel knows how to segment."
            )
        spans.append((int(start), int(stop)))
    return tuple(spans)


def _delay_embedding(
    signal: Any,
    taps: int,
    segments: Sequence[tuple[int, int]],
) -> Any:
    """``(rows, taps, channels)`` where ``[t, i]`` is ``signal[t - (taps-1-i)]``.

    Tap index ``taps-1`` multiplies the current token and tap index ``0`` the
    oldest, which is `torch.nn.functional.conv1d`'s cross-correlation ordering
    under ``padding=L-1`` -- the exact call the frozen collector replays when it
    re-derives a short conv's context.  Rows before a sequence's start are zero,
    which is the same left padding the runtime uses.
    """

    torch = _torch()
    rows, channels = int(signal.shape[0]), int(signal.shape[1])
    embedding = torch.zeros((rows, taps, channels), dtype=signal.dtype)
    for start, stop in segments:
        block = signal[start:stop]
        length = stop - start
        for index in range(taps):
            lag = taps - 1 - index
            if lag == 0:
                embedding[start:stop, index, :] = block
            elif lag < length:
                embedding[start + lag : stop, index, :] = block[: length - lag]
    return embedding


def _filter(embedding: Any, taps: Any) -> Any:
    """Apply per-channel taps to a delay embedding: ``(rows, channels)``."""

    torch = _torch()
    return torch.einsum("rtc,ct->rc", embedding, taps)


def _weighted_channel_solve(
    embedding: Any,
    target: Any,
    prior: Any,
    weights: Any | None,
    ridge: float,
) -> tuple[Any, Any, Any]:
    """Prior-anchored per-channel ridge over the kept taps.

    One independent ``k x k`` system per channel -- a depthwise filter has no
    cross-channel coupling -- solving the same objective `_fit_linear` solves
    for the dense techniques, ``||X S - R||^2 + ridge*||S||^2`` around a
    source-derived prior, with an optional diagonal row weighting.  Returns the
    solved taps, the per-channel normal matrices (for conditioning) and the
    per-channel residual sum of squares.
    """

    torch = _torch()
    if weights is None:
        gram = torch.einsum("rtc,ruc->ctu", embedding, embedding)
        cross = torch.einsum("rtc,rc->ct", embedding, target)
        target_energy = (target * target).sum(dim=0)
    else:
        scaled = embedding * weights.unsqueeze(1)
        gram = torch.einsum("rtc,ruc->ctu", scaled, embedding)
        cross = torch.einsum("rtc,rc->ct", scaled, target)
        target_energy = (weights * target * target).sum(dim=0)
    kept = int(embedding.shape[1])
    identity = torch.eye(kept, dtype=embedding.dtype).unsqueeze(0)
    system = gram + ridge * identity
    residual_rhs = cross - torch.einsum("ctu,cu->ct", gram, prior)
    correction = torch.linalg.solve(system, residual_rhs.unsqueeze(-1)).squeeze(-1)
    solved = prior + correction
    residual = (
        target_energy
        - 2.0 * (solved * cross).sum(dim=-1)
        + torch.einsum("ct,ctu,cu->c", solved, gram, solved)
    )
    return solved, system, residual.clamp_min(0.0)


def c04_fir_tap_contraction_technique(request: Any) -> Any:
    """C04: keep ``t`` of the parent's FIR taps and refit the survivors.

    The subset is chosen by exhaustive search over every subset of the requested
    size, scored by the same weighted least-squares residual the refit
    minimizes, with a lexicographic tie-break so two runs of the same cell are
    the same cell.  Three taps make that three candidates; the enumeration is
    capped so a wider filter refuses instead of running an exponential search.
    """

    from .matrix_cell_receipt import LEVELS
    from .matrix_technique_registry import (
        CellTechniqueError,
        CellTechniqueResult,
        sealable_solve_number,
    )

    torch = _torch()
    spec = request.component
    if spec.component_id != C04_COMPONENT_ID or spec.contraction_axis != "kernel":
        raise CellTechniqueError(
            "c04_fir_tap_contraction_technique is the conv_kernel kernel only; "
            f"{spec.component_id} has axis {spec.contraction_axis!r}."
        )
    if spec.input_endpoint is None or spec.write_endpoint is None:
        raise CellTechniqueError("C04 has no sealed input/write endpoints.")
    if spec.layer_index is None:
        raise CellTechniqueError("C04 needs a layer index to name its neighbours.")

    key = spec.weight_keys[0]
    parent = request.parent_weights[key].detach().to(dtype=torch.float64)
    if parent.ndim != 3 or int(parent.shape[1]) != 1:
        raise CellTechniqueError(
            f"{key} is {tuple(int(v) for v in parent.shape)}; C04 expects a "
            "depthwise (channels, 1, taps) FIR kernel."
        )
    channels = int(parent.shape[0])
    taps = int(parent.shape[2])
    parent_taps = parent[:, 0, :].contiguous()

    # (1) the level, re-resolved on the axis it actually names.
    level = str(request.cell.level)
    if level == "X4":
        kept_taps = request.knob("conv_fir_x4_taps", None)
        if kept_taps is None:
            raise CellTechniqueError(
                "X4 for C04 is an option-bank lookup this campaign has not "
                "settled on the tap axis; pass conv_fir_x4_taps explicitly."
            )
        kept_taps = int(kept_taps)
    else:
        fraction = LEVELS.get(level)
        if not isinstance(fraction, float):
            raise CellTechniqueError(f"C04 has no tap-axis reading of level {level!r}.")
        kept_taps = int(math.floor(taps * float(fraction)))
    if kept_taps < 1 or kept_taps >= taps:
        raise CellTechniqueError(
            f"Level {level} on {taps} FIR taps resolves to {kept_taps}, which is "
            "not a contraction of the tap axis."
        )

    layer = int(spec.layer_index)
    in_proj_key = f"model.layers.{layer}.conv.in_proj.weight"
    out_proj_key = f"model.layers.{layer}.conv.out_proj.weight"
    for neighbour in (in_proj_key, out_proj_key):
        if neighbour not in request.parent_weights:
            raise CellTechniqueError(
                f"C04 needs read-only access to {neighbour} to replay the "
                "operator it lives inside; the parent state does not carry it."
            )
    in_proj = request.parent_weights[in_proj_key].detach().to(dtype=torch.float64)
    out_proj = request.parent_weights[out_proj_key].detach().to(dtype=torch.float64)
    if int(in_proj.shape[0]) != 3 * channels:
        raise CellTechniqueError(
            f"{in_proj_key} emits {int(in_proj.shape[0])} channels; a B/C/x "
            f"projection over {channels} FIR channels emits {3 * channels}."
        )

    minimum_rows = int(request.knob("conv_fir_min_segment_rows", 8))
    maximum_rows = int(request.knob("conv_fir_max_segment_rows", 4096))
    tolerance = float(
        request.knob("conv_fir_reconstruction_tolerance", _DEFAULT_RECONSTRUCTION_TOLERANCE)
    )
    discrimination = float(
        request.knob("conv_fir_discrimination_ratio", _DEFAULT_DISCRIMINATION_RATIO)
    )
    ridge = float(request.knob("conv_fir_ridge", 1e-6))
    weighting = str(request.knob("conv_fir_channel_weighting", "gate"))
    if weighting not in ("gate", "none"):
        raise CellTechniqueError(
            f"conv_fir_channel_weighting must be 'gate' or 'none', not {weighting!r}."
        )

    def split(rows: Any) -> tuple[Any, Any]:
        """``operator.input`` rows -> the FIR's input and its output gate."""

        projected = rows.to(dtype=torch.float64) @ in_proj.T
        b_block, c_block, x_block = projected.chunk(3, dim=-1)
        return (b_block * x_block).contiguous(), c_block.contiguous()

    def operator_forward(rows: Any, kernel: Any) -> Any:
        signal, gate = split(rows)
        segments = _causal_segments(
            rows.to(dtype=torch.float64),
            minimum_length=minimum_rows,
            maximum_length=maximum_rows,
        )
        embedding = _delay_embedding(signal, taps, segments)
        return (gate * _filter(embedding, kernel)) @ out_proj.T

    # (2) the segmentation, derived and then proven against the parent.
    held_input = request.evidence.heldout_rows(spec.input_endpoint).to(
        dtype=torch.float64
    )
    held_write = request.evidence.heldout_rows(spec.write_endpoint).to(
        dtype=torch.float64
    )
    held_segments = _causal_segments(
        held_input, minimum_length=minimum_rows, maximum_length=maximum_rows
    )
    held_signal, held_gate = split(held_input)
    replayed = (
        held_gate * _filter(_delay_embedding(held_signal, taps, held_segments), parent_taps)
    ) @ out_proj.T
    reconstruction = _relative_l2(replayed, held_write)
    single = (
        held_gate
        * _filter(
            _delay_embedding(held_signal, taps, ((0, int(held_input.shape[0])),)),
            parent_taps,
        )
    ) @ out_proj.T
    unsegmented = _relative_l2(single, held_write)
    request.log(
        f"C04 {request.cell.text}: {len(held_segments)} held-out sequences, "
        f"parent replay relL2 {reconstruction:.6g} (unsegmented {unsegmented:.6g}), "
        f"taps {taps} -> {kept_taps}"
    )
    if not math.isfinite(reconstruction) or reconstruction > tolerance:
        raise CellTechniqueError(
            f"C04 cannot reproduce {spec.write_endpoint!r} from "
            f"{spec.input_endpoint!r} under its derived segmentation: relL2 "
            f"{reconstruction:.6g} against a {tolerance:g} tolerance.  The cell "
            "is void rather than a measurement of the technique."
        )
    if not math.isfinite(unsegmented) or unsegmented < discrimination * reconstruction:
        raise CellTechniqueError(
            "C04's segmentation check has no power on this evidence: the "
            f"deliberately unsegmented replay scores {unsegmented:.6g} against "
            f"the segmented {reconstruction:.6g}, short of the required "
            f"{discrimination:g}x separation."
        )

    # (3) the contraction, against the parent filter's own response.
    fit_input = request.evidence.fit_rows(spec.input_endpoint).to(dtype=torch.float64)
    fit_segments = _causal_segments(
        fit_input, minimum_length=minimum_rows, maximum_length=maximum_rows
    )
    fit_signal, fit_gate = split(fit_input)
    fit_embedding = _delay_embedding(fit_signal, taps, fit_segments)
    fit_target = _filter(fit_embedding, parent_taps)
    weights = (fit_gate * fit_gate).contiguous() if weighting == "gate" else None

    identity_solved, _identity_system, _identity_residual = _weighted_channel_solve(
        fit_embedding,
        fit_target,
        parent_taps,
        weights,
        ridge,
    )
    identity_deviation = float((identity_solved - parent_taps).abs().max().item())

    if taps > 24 or math.comb(taps, kept_taps) > _MAX_TAP_SUBSETS:
        raise CellTechniqueError(
            f"C04 would enumerate {math.comb(taps, kept_taps)} tap subsets, past "
            f"the {_MAX_TAP_SUBSETS} cap; a wider filter needs a selection "
            "policy this kernel does not have."
        )
    best: tuple[float, tuple[int, ...], Any, Any] | None = None
    scores: dict[str, float] = {}
    for subset in itertools.combinations(range(taps), kept_taps):
        index = torch.tensor(subset, dtype=torch.int64)
        solved, system, residual = _weighted_channel_solve(
            fit_embedding[:, index, :],
            fit_target,
            parent_taps[:, index],
            weights,
            ridge,
        )
        total = float(residual.sum().item())
        scores["+".join(str(v) for v in subset)] = total
        if best is None or total < best[0]:
            best = (total, tuple(int(v) for v in subset), solved, system)
    assert best is not None  # itertools.combinations is non-empty for 1<=k<taps
    total_residual, kept_index, solved_taps, solved_system = best

    child_taps = torch.zeros_like(parent_taps)
    child_taps[:, torch.tensor(kept_index, dtype=torch.int64)] = solved_taps
    weight = child_taps.unsqueeze(1).contiguous()
    if tuple(weight.shape) != tuple(parent.shape):
        raise CellTechniqueError(
            f"C04 materialized {tuple(int(v) for v in weight.shape)} for a parent "
            f"of {tuple(int(v) for v in parent.shape)}."
        )

    condition = float(torch.linalg.cond(solved_system).max().item())
    target_energy = float((fit_target * fit_target).sum().item())
    solve: dict[str, Any] = {
        "solver_path": "c04_depthwise_fir_tap_subset_prior_anchored_ridge",
        "ridge_relative": 0.0,
        "condition_number": sealable_solve_number(condition),
        "condition_cap_saturated": not math.isfinite(condition),
        "prior_norm": float(torch.linalg.matrix_norm(parent_taps).item()),
        "correction_norm": float(torch.linalg.matrix_norm(child_taps - parent_taps).item()),
        "requested_clamp_ratio": None,
        "allowed_clamp_ratio": None,
        "clamp_engaged": False,
        "omp": None,
        # The tap axis, named beside the axis the harness resolved, so nobody
        # has to guess which 960 the cell id's M16 refers to.
        "contraction_axis": "kernel",
        "parent_taps": taps,
        "kept_taps": int(kept_taps),
        "kept_tap_indices": list(kept_index),
        "harness_level_width": int(request.level_width),
        "harness_level_axis_width": int(spec.own_width),
        "tap_subset_scores": scores,
        "tap_subset_selected_residual": total_residual,
        "tap_subset_relative_residual": (
            None if target_energy <= 0.0 else total_residual / target_energy
        ),
        "channel_weighting": weighting,
        "channel_count": channels,
        # The parent-calibration measurement the generic gate could not make.
        "parent_operator_replay_relative_l2": sealable_solve_number(reconstruction),
        "parent_operator_replay_tolerance": tolerance,
        "unsegmented_replay_relative_l2": sealable_solve_number(unsegmented),
        "segmentation_discrimination_ratio": sealable_solve_number(
            None if reconstruction <= 0.0 else unsegmented / reconstruction
        ),
        "fit_sequences": len(fit_segments),
        "heldout_sequences": len(held_segments),
        "fit_sequence_lengths": sorted({stop - start for start, stop in fit_segments}),
        "heldout_sequence_lengths": sorted(
            {stop - start for start, stop in held_segments}
        ),
        # Zero drop keeps every tap, the residual against a parent-derived
        # target is identically zero, and a prior-anchored solve with a zero
        # residual returns the prior.  Measured, not asserted.
        "identity_gate_max_abs_deviation": identity_deviation,
        # The mask is one global subset of tap positions, not one per channel,
        # so its cost is `parent_taps` bits and not `channels * parent_taps`.
        "selection_only_parameters": int(taps),
    }

    def predict(rows: Any) -> Any:
        return operator_forward(rows, child_taps)

    return CellTechniqueResult(
        installed_weights={key: weight},
        stored_parameter_count=int(channels * kept_taps),
        solve=solve,
        predict=predict,
        notes=(
            f"C04 depthwise FIR tap contraction: kept taps {list(kept_index)} of "
            f"{taps} with a {weighting}-weighted prior-anchored refit; parent "
            f"operator replay relL2 {reconstruction:.6g}, identity gate "
            f"{identity_deviation:.3g}."
        ),
    )


def _relative_l2(candidate: Any, reference: Any) -> float:
    torch = _torch()
    numerator = float(torch.linalg.matrix_norm(candidate - reference).item())
    denominator = float(torch.linalg.matrix_norm(reference).item())
    if denominator <= 0.0 or not math.isfinite(denominator):
        return float("nan")
    return numerator / denominator


def _register_conv_fir_technique() -> None:
    """Wire the C04 kernel into the harness dispatch table at import time."""

    try:
        from .matrix_technique_registry import register
    except ImportError:  # pragma: no cover - the harness lane may not be present
        return
    register(
        C04_TECHNIQUE_ID,
        c04_fir_tap_contraction_technique,
        component=C04_COMPONENT_ID,
        replace=True,
    )


_register_conv_fir_technique()


def describe_conv_fir_technique() -> Mapping[str, Any]:
    """Serializable inventory of what this kernel wires, for a cell receipt."""

    return {
        "schema": "neural-forge-matrix-technique-conv-fir-v1",
        "technique": C04_TECHNIQUE_ID,
        "component": C04_COMPONENT_ID,
        "contraction_axis": "kernel",
        "selection": "exhaustive tap subsets, weighted least-squares residual",
        "refit": "per-channel prior-anchored ridge over the kept taps",
        "target": "the parent filter's response on the sealed evidence input",
        "parent_gate": "short-conv replay of operator.input -> operator.output",
    }
