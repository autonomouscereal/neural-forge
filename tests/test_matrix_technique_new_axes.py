"""Contract tests for the three wave-3 kernels: C04 taps, C10 heads, T9 schedules.

`NOTES-MX-F` §4 left two components localized to a *missing kernel* rather than
to a technique verdict -- C04's ``kernel`` axis and C10's ``heads`` axis -- and
`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` §1.2 left T9 unrun entirely.  These
are the first tests for all three, and what they pin is the set of ways a new
kernel can be wrong in a manner the sealed receipt would *not* show:

1. **The identity gate.**  At zero contraction each kernel's own algebra has to
   return the parent bit-for-bit.  Both C04 and C10 fit against a
   parent-derived target for exactly this reason: a prior-anchored solve with
   an identically zero residual returns its prior, so "reconstructs exactly" is
   a testable claim rather than a hopeful one.  A kernel that silently
   perturbed the surviving coefficients would still produce a plausible
   receipt.
2. **The axis is the axis the cell id names.**  C04's level means taps, not the
   960 channels the harness's generic resolver hands it; C10's means whole
   heads and refuses to round to one.  Contracting the wrong axis is the
   failure mode that produces a confident number about a different component.
3. **Storage is the stored form.**  Both kernels genuinely contract (unlike
   every marginal factorization in this campaign, `NOTES-MX-E` §4.1), and the
   count has to be the child's, not the parent-shaped materialization's.
4. **A refusal is a refusal.**  Wrong component, wrong axis, a level that does
   not name whole heads, a segmentation the parent replay does not confirm --
   each raises `CellTechniqueError` with the reason, because the campaign reads
   a technique refusal as a result and a silent fallback as a lie.
5. **T9 conditions the teacher and nothing else.**  Its rung-2 evidence view
   has to answer the *write* endpoint from the previous child and pass every
   other endpoint through untouched, and its final width has to equal the
   cell's own level width so a T9 cell is comparable with its direct twin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

torch = pytest.importorskip("torch")

from neural_forge.matrix_cell_receipt import CellId  # noqa: E402
from neural_forge.matrix_technique_conv_fir import (  # noqa: E402
    c04_fir_tap_contraction_technique,
    describe_conv_fir_technique,
)
from neural_forge.matrix_technique_head_drop import (  # noqa: E402
    c10_head_drop_technique,
    describe_head_drop_technique,
)
from neural_forge.matrix_technique_progressive import (  # noqa: E402
    describe_progressive_technique,
    t9_progressive_multipass_technique,
)
from neural_forge.matrix_technique_registry import (  # noqa: E402
    CellTechniqueError,
    CellTechniqueRequest,
    component_spec,
    resolve,
    validate_result,
)

CHANNELS = 24
TAPS = 3
HEADS = 8
HEAD_DIM = 4
ATTENTION_WIDTH = HEADS * HEAD_DIM
KV_HEADS = 4
SEQUENCES = 6
SEQUENCE_ROWS = 20


class _Evidence:
    def __init__(self, fit, heldout):
        self._fit = dict(fit)
        self._heldout = dict(heldout)

    @property
    def endpoints(self):
        return tuple(sorted(self._fit))

    @property
    def selection_sha256(self):
        return "0" * 64

    @property
    def fit_row_count(self):
        return int(next(iter(self._fit.values())).shape[0])

    @property
    def heldout_row_count(self):
        return int(next(iter(self._heldout.values())).shape[0])

    def fit_rows(self, endpoint):
        return self._fit[endpoint]

    def heldout_rows(self, endpoint):
        return self._heldout[endpoint]


def _request(cell_text, state, evidence, *, level_width, knobs=None):
    cell = CellId.parse(cell_text)
    spec = component_spec(cell, state)
    return CellTechniqueRequest(
        cell=cell,
        component=spec,
        level=cell.level,
        level_width=level_width,
        seed=0,
        parent_weights=state,
        evidence=evidence,
        knobs=dict(knobs or {}),
        workspace=Path("."),
        log=lambda message: None,
    )


# ------------------------------------------------------------- C04 fixtures


def _conv_state(generator, *, layer=0):
    def randn(*shape):
        return torch.randn(shape, dtype=torch.float64, generator=generator) / (
            shape[-1] ** 0.5
        )

    return {
        f"model.layers.{layer}.conv.conv.weight": randn(CHANNELS, 1, TAPS),
        f"model.layers.{layer}.conv.in_proj.weight": randn(3 * CHANNELS, CHANNELS),
        f"model.layers.{layer}.conv.out_proj.weight": randn(CHANNELS, CHANNELS),
    }


def _conv_rows(generator, sequences, *, anchor=None):
    """Concatenated captured sequences: every one starts on the same first row.

    That shared first row is the property the C04 kernel segments on -- position
    zero is a function of the first token alone, so it is bitwise identical in
    every captured sequence.
    """

    if anchor is None:
        anchor = torch.randn((CHANNELS,), dtype=torch.float64, generator=generator)
    blocks = []
    for _ in range(sequences):
        body = torch.randn(
            (SEQUENCE_ROWS - 1, CHANNELS), dtype=torch.float64, generator=generator
        )
        blocks.append(torch.cat((anchor.unsqueeze(0), body), dim=0))
    return torch.cat(blocks, dim=0).contiguous(), anchor


def _conv_forward(rows, state, taps, *, layer=0, segment_rows=SEQUENCE_ROWS):
    """The reference short conv, written independently of the kernel under test."""

    in_proj = state[f"model.layers.{layer}.conv.in_proj.weight"]
    out_proj = state[f"model.layers.{layer}.conv.out_proj.weight"]
    projected = rows @ in_proj.T
    b_block, c_block, x_block = projected.chunk(3, dim=-1)
    signal = b_block * x_block
    filtered = torch.zeros_like(signal)
    for start in range(0, int(rows.shape[0]), segment_rows):
        stop = min(start + segment_rows, int(rows.shape[0]))
        block = signal[start:stop]
        for index in range(TAPS):
            lag = TAPS - 1 - index
            if lag == 0:
                filtered[start:stop] += block * taps[:, index]
            elif lag < stop - start:
                filtered[start + lag : stop] += block[: stop - start - lag] * taps[
                    :, index
                ]
    return (c_block * filtered) @ out_proj.T


def _conv_evidence(generator, state, *, layer=0):
    fit_rows, anchor = _conv_rows(generator, SEQUENCES)
    held_rows, _ = _conv_rows(generator, SEQUENCES - 2, anchor=anchor)
    taps = state[f"model.layers.{layer}.conv.conv.weight"][:, 0, :]
    return _Evidence(
        {
            "operator.input": fit_rows,
            "operator.output": _conv_forward(fit_rows, state, taps, layer=layer),
        },
        {
            "operator.input": held_rows,
            "operator.output": _conv_forward(held_rows, state, taps, layer=layer),
        },
    )


# ------------------------------------------------------------- C10 fixtures


def _attention_state(generator, *, layer=2):
    def randn(*shape):
        return torch.randn(shape, dtype=torch.float64, generator=generator) / (
            shape[-1] ** 0.5
        )

    return {
        f"model.layers.{layer}.self_attn.q_proj.weight": randn(
            ATTENTION_WIDTH, ATTENTION_WIDTH
        ),
        f"model.layers.{layer}.self_attn.k_proj.weight": randn(
            KV_HEADS * HEAD_DIM, ATTENTION_WIDTH
        ),
        f"model.layers.{layer}.self_attn.v_proj.weight": randn(
            KV_HEADS * HEAD_DIM, ATTENTION_WIDTH
        ),
        f"model.layers.{layer}.self_attn.out_proj.weight": randn(
            ATTENTION_WIDTH, ATTENTION_WIDTH
        ),
        f"model.layers.{layer}.self_attn.q_layernorm.weight": torch.ones(
            HEAD_DIM, dtype=torch.float64
        ),
    }


def _attention_evidence(generator, state, *, layer=2, redundant_head=None):
    """Context rows and their exact O-projection write.

    ``redundant_head`` makes one head's context a *near* image of head 0's --
    ``0.5 * head_0 + 0.01 * noise``.  Near, not exact, and the distinction is
    the point: an exact dependency makes head 0 and head 5 equally droppable
    (each spans the other), so the selection's choice between them would be a
    rounding coin flip.  Under a near dependency the asymmetry is real and
    ordered -- absorbing head 5 costs the noise, absorbing head 0 costs twice
    the noise, absorbing anything else costs a whole head -- so the test can
    assert *which* head a residual-minimizing selection has to pick.
    """

    o_proj = state[f"model.layers.{layer}.self_attn.out_proj.weight"]

    def context(rows):
        value = torch.randn(
            (rows, ATTENTION_WIDTH), dtype=torch.float64, generator=generator
        )
        if redundant_head is not None:
            source = slice(0, HEAD_DIM)
            target = slice(redundant_head * HEAD_DIM, (redundant_head + 1) * HEAD_DIM)
            value[:, target] = value[:, source] * 0.5 + 0.01 * value[:, target]
        return value

    fit = context(320)
    held = context(96)
    return _Evidence(
        {"attention.context": fit, "attention.o": fit @ o_proj.T},
        {"attention.context": held, "attention.o": held @ o_proj.T},
    )


# ------------------------------------------------------------------ wiring


def test_the_two_missing_kernels_and_the_schedule_are_wired():
    assert resolve("T1", "C04") is c04_fir_tap_contraction_technique
    assert resolve("T1", "C10") is c10_head_drop_technique
    assert resolve("T9", "C05") is t9_progressive_multipass_technique
    # The generic T1 is untouched: the two new entries are component-pinned, the
    # same shape as T1-at-C14, so every other T1 column keeps its baseline.
    assert resolve("T1", "C05") is not c04_fir_tap_contraction_technique
    assert describe_conv_fir_technique()["contraction_axis"] == "kernel"
    assert describe_head_drop_technique()["contraction_axis"] == "heads"
    assert describe_progressive_technique()["does_not_condition"] == (
        "input rows",
        "evidence split",
        "row order",
    )


# ---------------------------------------------------------------------- C04


def test_c04_contracts_the_tap_axis_not_the_channel_axis():
    generator = torch.Generator().manual_seed(11)
    state = _conv_state(generator)
    evidence = _conv_evidence(generator, state)
    # 960 is what the harness's generic resolver produces for a (24, 1, 3)
    # tensor at M16 -- the channel axis.  The kernel must ignore it.
    request = _request("C04.T1.M16.L00", state, evidence, level_width=960)
    result = c04_fir_tap_contraction_technique(request)
    validate_result(request, result)

    weight = result.installed_weights["model.layers.0.conv.conv.weight"]
    assert tuple(weight.shape) == (CHANNELS, 1, TAPS)
    assert result.solve["kept_taps"] == 2
    assert result.solve["parent_taps"] == TAPS
    assert result.solve["harness_level_width"] == 960
    assert result.solve["harness_level_axis_width"] == CHANNELS
    # A dropped tap is dropped: the whole column is zero in every channel.
    dropped = [index for index in range(TAPS) if index not in result.solve["kept_tap_indices"]]
    assert len(dropped) == 1
    assert torch.count_nonzero(weight[:, 0, dropped[0]]).item() == 0
    # Storage is the child's, and it is a real contraction -- which is not true
    # of a single marginal factorization anywhere else in this campaign.
    assert result.stored_parameter_count == CHANNELS * 2
    assert result.stored_parameter_count < weight.numel()


def test_c04_identity_gate_returns_the_parent_taps_exactly():
    generator = torch.Generator().manual_seed(12)
    state = _conv_state(generator)
    evidence = _conv_evidence(generator, state)
    request = _request("C04.T1.M16.L00", state, evidence, level_width=960)
    result = c04_fir_tap_contraction_technique(request)
    # Zero drop is not a level, so the gate is measured inside the kernel and
    # sealed: keep every tap, fit against the parent's own response, and the
    # prior-anchored solve has to return the prior to machine precision.
    assert result.solve["identity_gate_max_abs_deviation"] < 1e-10


def test_c04_segments_the_rows_it_was_given_and_proves_it_against_the_parent():
    generator = torch.Generator().manual_seed(13)
    state = _conv_state(generator)
    evidence = _conv_evidence(generator, state)
    request = _request("C04.T1.M16.L00", state, evidence, level_width=960)
    result = c04_fir_tap_contraction_technique(request)
    assert result.solve["fit_sequences"] == SEQUENCES
    assert result.solve["heldout_sequences"] == SEQUENCES - 2
    assert result.solve["fit_sequence_lengths"] == [SEQUENCE_ROWS]
    # The replay is exact here because the fixture's target *is* the parent
    # operator; on sealed bf16 evidence it lands in the campaign's calibration
    # band instead, which is what the tolerance knob is for.
    assert result.solve["parent_operator_replay_relative_l2"] < 1e-12
    # And the check has power: the deliberately unsegmented replay is worse by
    # far more than the required separation.
    assert result.solve["unsegmented_replay_relative_l2"] > 1e-3


def test_c04_refuses_when_the_segmentation_cannot_be_confirmed():
    generator = torch.Generator().manual_seed(14)
    state = _conv_state(generator)
    evidence = _conv_evidence(generator, state)
    # Corrupt the parent's own taps *after* the evidence was built: the replay
    # can no longer reproduce the sealed write, so the cell is void rather than
    # a measurement of the technique.
    state["model.layers.0.conv.conv.weight"] = state[
        "model.layers.0.conv.conv.weight"
    ] + 1.0
    request = _request("C04.T1.M16.L00", state, evidence, level_width=960)
    with pytest.raises(CellTechniqueError, match="cannot reproduce"):
        c04_fir_tap_contraction_technique(request)


def test_c04_predicts_the_child_operator_through_the_whole_short_conv():
    generator = torch.Generator().manual_seed(15)
    state = _conv_state(generator)
    evidence = _conv_evidence(generator, state)
    request = _request("C04.T1.M16.L00", state, evidence, level_width=960)
    result = c04_fir_tap_contraction_technique(request)
    held = evidence.heldout_rows("operator.input")
    predicted = result.predict(held)
    # The harness scores the write with this callable, so it has to be the
    # child's *operator* output and not the raw tensor product a linear map
    # would give.  Independent reference forward, child taps.
    reference = _conv_forward(
        held, state, result.installed_weights["model.layers.0.conv.conv.weight"][:, 0, :]
    )
    assert torch.allclose(predicted, reference, atol=1e-12)
    assert predicted.shape == evidence.heldout_rows("operator.output").shape


def test_c04_refuses_a_component_that_is_not_the_fir_kernel():
    generator = torch.Generator().manual_seed(16)
    state = _conv_state(generator)
    state["model.layers.0.conv.out_proj.weight"] = state[
        "model.layers.0.conv.out_proj.weight"
    ]
    evidence = _conv_evidence(generator, state)
    request = _request("C05.T1.M16.L00", state, evidence, level_width=20)
    with pytest.raises(CellTechniqueError, match="conv_kernel kernel only"):
        c04_fir_tap_contraction_technique(request)


# ---------------------------------------------------------------------- C10


def test_c10_drops_whole_heads_and_absorbs_them_into_the_o_projection():
    generator = torch.Generator().manual_seed(21)
    state = _attention_state(generator)
    evidence = _attention_evidence(generator, state, redundant_head=5)
    keep = (HEADS - 1) * HEAD_DIM
    request = _request("C10.T1.M16.L02", state, evidence, level_width=keep)
    result = c10_head_drop_technique(request)
    validate_result(request, result)

    o_key = "model.layers.2.self_attn.out_proj.weight"
    q_key = "model.layers.2.self_attn.q_proj.weight"
    assert set(result.installed_weights) == {q_key, o_key}
    assert result.solve["kept_heads"] == HEADS - 1
    assert result.solve["dropped_head_count"] == 1
    assert result.solve["head_dim"] == HEAD_DIM
    # Head 5's context is 0.5*head_0 plus 1% noise, so absorbing it is the
    # cheapest drop available and a residual-minimizing selection has to say so.
    assert result.solve["dropped_heads"] == [5]
    scores = result.solve["head_drop_set_scores"]
    # The full ordering the fixture builds, asserted rather than the winner
    # alone: head 5 costs the noise, head 0 costs twice the noise because it is
    # recovered *through* head 5, and every independent head costs a whole head.
    assert scores["5"] < scores["0"] < min(
        value for key, value in scores.items() if key not in ("0", "5")
    )
    assert scores["0"] > 3.0 * scores["5"]
    energy = result.solve["fit_write_energy"]
    assert result.solve["selected_drop_relative_residual"] < 1e-3
    assert min(
        value for key, value in scores.items() if key not in ("0", "5")
    ) / energy > 0.01

    child_o = result.installed_weights[o_key]
    child_q = result.installed_weights[q_key]
    assert torch.count_nonzero(child_o[:, 5 * HEAD_DIM : 6 * HEAD_DIM]).item() == 0
    assert torch.count_nonzero(child_q[5 * HEAD_DIM : 6 * HEAD_DIM, :]).item() == 0
    # Every other head's query rows are untouched -- a head drop is a block
    # edit, not a re-solve of the whole projection.
    assert torch.equal(child_q[: 5 * HEAD_DIM, :], state[q_key][: 5 * HEAD_DIM, :])


def test_c10_head_drop_storage_is_a_real_contraction():
    generator = torch.Generator().manual_seed(22)
    state = _attention_state(generator)
    evidence = _attention_evidence(generator, state)
    keep = (HEADS - 2) * HEAD_DIM
    request = _request("C10.T1.M8.L02", state, evidence, level_width=keep)
    result = c10_head_drop_technique(request)
    kv = 2 * KV_HEADS * HEAD_DIM * ATTENTION_WIDTH
    expected = 2 * ATTENTION_WIDTH * (HEADS - 2) * HEAD_DIM + kv
    assert result.stored_parameter_count == expected
    assert result.stored_parameter_count < request.component.parent_parameter_count
    # The kv rows a fully-orphaned group would free are reported and not
    # banked, because this child still stores them.
    assert "dead_kv_groups" in result.solve
    assert result.solve["kv_group_size"] == HEADS // KV_HEADS


def test_c10_identity_gate_returns_the_parent_projection_exactly():
    generator = torch.Generator().manual_seed(23)
    state = _attention_state(generator)
    evidence = _attention_evidence(generator, state)
    request = _request(
        "C10.T1.M16.L02", state, evidence, level_width=(HEADS - 1) * HEAD_DIM
    )
    result = c10_head_drop_technique(request)
    assert result.solve["identity_gate_max_abs_deviation"] < 1e-10
    # And the parent gate the generic harness gate could not run.
    assert result.solve["parent_matmul_relative_l2"] < 1e-12


def test_c10_refuses_a_level_that_is_not_a_whole_number_of_heads():
    generator = torch.Generator().manual_seed(24)
    state = _attention_state(generator)
    evidence = _attention_evidence(generator, state)
    request = _request("C10.T1.M16.L02", state, evidence, level_width=ATTENTION_WIDTH - 1)
    with pytest.raises(CellTechniqueError, match="whole number of"):
        c10_head_drop_technique(request)


def test_c10_refuses_when_the_parent_does_not_reproduce_its_own_write():
    generator = torch.Generator().manual_seed(25)
    state = _attention_state(generator)
    evidence = _attention_evidence(generator, state)
    state["model.layers.2.self_attn.out_proj.weight"] = (
        state["model.layers.2.self_attn.out_proj.weight"] + 1.0
    )
    request = _request(
        "C10.T1.M16.L02", state, evidence, level_width=(HEADS - 1) * HEAD_DIM
    )
    with pytest.raises(CellTechniqueError, match="cannot reproduce"):
        c10_head_drop_technique(request)


# ----------------------------------------------------------------------- T9


def _writer_state(generator):
    def randn(*shape):
        return torch.randn(shape, dtype=torch.float64, generator=generator) / (
            shape[-1] ** 0.5
        )

    return {"model.layers.5.conv.out_proj.weight": randn(CHANNELS, CHANNELS)}


def _writer_evidence(generator, state):
    parent = state["model.layers.5.conv.out_proj.weight"]
    scale = torch.logspace(0.0, -2.0, CHANNELS, dtype=torch.float64)
    fit = torch.randn((256, CHANNELS), dtype=torch.float64, generator=generator) * scale
    held = torch.randn((64, CHANNELS), dtype=torch.float64, generator=generator) * scale
    return _Evidence(
        {"short_conv.context": fit, "operator.output": fit @ parent.T},
        {"short_conv.context": held, "operator.output": held @ parent.T},
    )


def test_t9_lands_on_the_cells_own_level_width_through_an_intermediate_rung():
    generator = torch.Generator().manual_seed(31)
    state = _writer_state(generator)
    evidence = _writer_evidence(generator, state)
    final = 18
    request = _request(
        "C05.T9.M8.L05",
        state,
        evidence,
        level_width=final,
        knobs={"progressive_intermediate_levels": ["M16"]},
    )
    result = t9_progressive_multipass_technique(request)
    validate_result(request, result)

    schedule = result.solve["progressive_schedule"]
    assert [entry["level"] for entry in schedule] == ["M16", "M8"]
    assert schedule[0]["width"] == int(CHANNELS * 15 // 16)
    # The last rung is the cell's own level width, so a T9 cell and its direct
    # twin store the same rank and differ only in what taught the fit.
    assert schedule[-1]["width"] == final
    assert result.solve["progressive_matches_direct_level_width"] is True
    assert result.solve["progressive_rungs"] == 2
    assert result.solve["progressive_passes"][0]["teacher"] == "parent"
    assert result.solve["progressive_passes"][1]["teacher"] == "rung-1"
    assert result.stored_parameter_count == final * 2 * CHANNELS


def test_t9_second_rung_is_taught_by_the_first_childs_write():
    generator = torch.Generator().manual_seed(32)
    state = _writer_state(generator)
    evidence = _writer_evidence(generator, state)
    seen: list[tuple[str, int]] = []

    class _Recorder:
        """A base technique that records what its second rung was handed."""

        def __call__(self, request):
            key = request.component.weight_keys[0]
            parent = request.parent_weights[key]
            target = request.evidence.fit_rows(request.component.write_endpoint)
            rows = request.evidence.fit_rows(request.component.input_endpoint)
            seen.append(
                (
                    request.level,
                    int(request.level_width),
                )
            )
            # The teacher's write, to the precision the rung's own parent gives.
            residual = float(
                torch.linalg.matrix_norm(target - rows @ parent.T).item()
            )
            assert residual < 1e-9, "rung target is not its own parent's write"
            from neural_forge.matrix_technique_registry import CellTechniqueResult

            basis = torch.linalg.qr(
                torch.randn(
                    (CHANNELS, int(request.level_width)),
                    dtype=torch.float64,
                    generator=torch.Generator().manual_seed(int(request.level_width)),
                )
            )[0]
            weight = (parent @ basis) @ basis.T
            return CellTechniqueResult(
                installed_weights={key: weight.contiguous()},
                stored_parameter_count=int(request.level_width) * 2 * CHANNELS,
                solve={
                    "solver_path": "recorder",
                    "ridge_relative": 0.0,
                    "condition_number": 1.0,
                    "condition_cap_saturated": False,
                    "prior_norm": 0.0,
                    "correction_norm": 0.0,
                    "requested_clamp_ratio": None,
                    "allowed_clamp_ratio": None,
                    "clamp_engaged": False,
                    "omp": None,
                },
            )

    from neural_forge.matrix_technique_registry import register

    register("T6", _Recorder(), component="C05", replace=True)
    request = _request(
        "C05.T9.M8.L05",
        state,
        evidence,
        level_width=18,
        knobs={
            "progressive_base_technique": "T6",
            "progressive_intermediate_levels": ["M16"],
        },
    )
    result = t9_progressive_multipass_technique(request)
    assert seen == [("M16", int(CHANNELS * 15 // 16)), ("M8", 18)]
    assert result.solve["progressive_base_technique"] == "T6"


def test_t9_refuses_a_non_monotone_schedule_and_refuses_to_schedule_itself():
    generator = torch.Generator().manual_seed(33)
    state = _writer_state(generator)
    evidence = _writer_evidence(generator, state)
    request = _request(
        "C05.T9.M16.L05",
        state,
        evidence,
        level_width=int(CHANNELS * 15 // 16),
        knobs={"progressive_intermediate_levels": ["M4"]},
    )
    with pytest.raises(CellTechniqueError, match="not monotone"):
        t9_progressive_multipass_technique(request)

    request = _request(
        "C05.T9.M8.L05",
        state,
        evidence,
        level_width=18,
        knobs={"progressive_base_technique": "T9"},
    )
    with pytest.raises(CellTechniqueError, match="cannot schedule itself"):
        t9_progressive_multipass_technique(request)


def test_t9_teacher_evidence_passes_every_other_endpoint_through():
    from neural_forge.matrix_technique_progressive import _TeacherEvidence

    generator = torch.Generator().manual_seed(34)
    state = _writer_state(generator)
    evidence = _writer_evidence(generator, state)
    child = torch.randn((CHANNELS, CHANNELS), dtype=torch.float64, generator=generator)
    view = _TeacherEvidence(
        evidence, "operator.output", child, "short_conv.context"
    )
    rows = evidence.fit_rows("short_conv.context")
    assert torch.equal(view.fit_rows("short_conv.context"), rows)
    assert torch.allclose(view.fit_rows("operator.output"), rows @ child.T)
    assert view.selection_sha256 == evidence.selection_sha256
    assert view.fit_row_count == evidence.fit_row_count
