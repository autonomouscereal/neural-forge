"""The technique lane against the harness lane's real dispatch contract.

`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` §4.3 freezes exactly two objects
between the two lanes: the receipt schema and the dispatch table.  These tests
drive T4/T5/T7/T8 through the second one — a real ``CellTechniqueRequest``, a
real ``ComponentSpec`` resolved against a parent-shaped state dict, and the
registry's own ``validate_result`` — so a contract drift on either side fails
here rather than during a GPU cell.

The load-bearing claim in every case is the same: the installed tensor is
**parent-shaped and functionally identical to the contracted child**, so every
untouched neighbour really is copy_exact and ``stored_parameter_count`` is the
only place the contraction is declared.
"""

from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from neural_forge.matrix_cell_receipt import CellId  # noqa: E402
from neural_forge.matrix_technique_registry import (  # noqa: E402
    CellTechniqueError,
    CellTechniqueRequest,
    component_spec,
    registered_pairs,
    resolve,
    resolve_level_width,
    validate_result,
)

# Importing the technique modules is what registers them (design §4.3: "the
# technique lane registers from src/neural_forge/matrix_technique_*.py at
# import time").
from neural_forge.matrix_technique_body_routes import (  # noqa: E402
    t5_body_route_technique,
)
from neural_forge.matrix_technique_lowrank_writers import (  # noqa: E402,F401
    t4_lowrank_writer_technique,
)
from neural_forge.matrix_technique_relation_matching import (  # noqa: E402
    t7_attention_relation_technique,
)


HIDDEN = 48
INTERMEDIATE = 96
HEAD_DIM = 8
QUERY_HEADS = 6
KV_HEADS = 3
LAYER = 5
ATTENTION_LAYER = 2


def _normal(seed: int, rows: int, columns: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, columns, generator=generator, dtype=torch.float64) * 0.1


def _captured(seed: int, rows: int, channels: int, latent: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    factors = torch.randn(rows, latent, generator=generator, dtype=torch.float64)
    mixing = torch.randn(latent, channels, generator=generator, dtype=torch.float64)
    noise = torch.randn(rows, channels, generator=generator, dtype=torch.float64)
    return factors @ mixing / latent**0.5 + 0.05 * noise


def _parent_state() -> dict[str, torch.Tensor]:
    return {
        f"model.layers.{LAYER}.conv.in_proj.weight": _normal(1, 3 * HIDDEN, HIDDEN),
        f"model.layers.{LAYER}.conv.out_proj.weight": _normal(2, HIDDEN, HIDDEN),
        f"model.layers.{LAYER}.feed_forward.w1.weight": _normal(
            3, INTERMEDIATE, HIDDEN
        ),
        f"model.layers.{LAYER}.feed_forward.w2.weight": _normal(
            4, HIDDEN, INTERMEDIATE
        ),
        f"model.layers.{ATTENTION_LAYER}.self_attn.q_proj.weight": _normal(
            5, QUERY_HEADS * HEAD_DIM, HIDDEN
        ),
        f"model.layers.{ATTENTION_LAYER}.self_attn.k_proj.weight": _normal(
            6, KV_HEADS * HEAD_DIM, HIDDEN
        ),
        f"model.layers.{ATTENTION_LAYER}.self_attn.out_proj.weight": _normal(
            7, HIDDEN, KV_HEADS * HEAD_DIM
        ),
        f"model.layers.{ATTENTION_LAYER}.self_attn.v_proj.weight": _normal(
            8, KV_HEADS * HEAD_DIM, HIDDEN
        ),
        "model.layers.0.operator_norm.weight": torch.ones(
            HIDDEN, dtype=torch.float64
        ),
        "model.layers.0.ffn_norm.weight": torch.ones(HIDDEN, dtype=torch.float64),
    }


class _Evidence:
    """A minimal ``MatrixCellEvidence`` over synthetic captured-shaped rows."""

    def __init__(self, rows: dict[str, torch.Tensor], heldout: int = 0) -> None:
        self._fit = rows
        self._heldout = {
            name: value[:heldout].contiguous() for name, value in rows.items()
        }
        self._heldout_rows = heldout

    @property
    def endpoints(self) -> tuple[str, ...]:
        return tuple(sorted(self._fit))

    @property
    def selection_sha256(self) -> str:
        return "0" * 64

    @property
    def fit_row_count(self) -> int:
        return int(next(iter(self._fit.values())).shape[0])

    @property
    def heldout_row_count(self) -> int:
        return self._heldout_rows

    def fit_rows(self, endpoint: str) -> torch.Tensor:
        if endpoint not in self._fit:
            raise KeyError(endpoint)
        return self._fit[endpoint]

    def heldout_rows(self, endpoint: str) -> torch.Tensor:
        if endpoint not in self._heldout:
            raise KeyError(endpoint)
        return self._heldout[endpoint]


def _evidence(parent: dict[str, torch.Tensor], heldout: int = 32) -> _Evidence:
    residual = _captured(10, 320, HIDDEN, 20)
    context = _captured(11, 320, HIDDEN, 20)
    post_swiglu = _captured(12, 320, INTERMEDIATE, 24)
    attention_context = _captured(13, 320, KV_HEADS * HEAD_DIM, 12)
    rows = {
        "operator.input": residual,
        "short_conv.context": context,
        "mlp.input": residual,
        "mlp.post_swiglu": post_swiglu,
        "attention.context": attention_context,
        "residual.pre": residual,
    }
    rows["operator.output"] = context @ parent[
        f"model.layers.{LAYER}.conv.out_proj.weight"
    ].transpose(0, 1)
    rows["mlp.gate"] = residual @ parent[
        f"model.layers.{LAYER}.feed_forward.w1.weight"
    ].transpose(0, 1)
    rows["mlp.down"] = post_swiglu @ parent[
        f"model.layers.{LAYER}.feed_forward.w2.weight"
    ].transpose(0, 1)
    rows["attention.o"] = attention_context @ parent[
        f"model.layers.{ATTENTION_LAYER}.self_attn.out_proj.weight"
    ].transpose(0, 1)
    rows["attention.v"] = residual @ parent[
        f"model.layers.{ATTENTION_LAYER}.self_attn.v_proj.weight"
    ].transpose(0, 1)
    rows["residual.post_layer"] = residual
    return _Evidence(rows, heldout=heldout)


def _request(
    cell_text: str,
    *,
    knobs: dict[str, object] | None = None,
    heldout: int = 32,
) -> CellTechniqueRequest:
    parent = _parent_state()
    cell = CellId.parse(cell_text)
    spec = component_spec(cell, parent)
    return CellTechniqueRequest(
        cell=cell,
        component=spec,
        level=cell.level,
        level_width=resolve_level_width(spec, cell.level),
        seed=0,
        parent_weights=parent,
        evidence=_evidence(parent, heldout=heldout),
        knobs=dict(knobs or {}),
        workspace=Path("."),
        log=lambda _message: None,
    )


def _run(cell_text: str, **kwargs: object):
    request = _request(cell_text, **kwargs)
    result = resolve(request.cell.technique, request.cell.component)(request)
    validate_result(request, result)
    return request, result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_every_wave_one_technique_component_pair_is_wired():
    pairs = set(registered_pairs())
    for component in ("C03", "C05", "C08", "C09", "C11", "C12", "C13"):
        assert ("T5", component) in pairs
    for component in ("C05", "C09", "C13"):
        assert ("T4", component) in pairs
        assert ("T8", component) in pairs
    assert ("T7", "C06") in pairs and ("T7", "C07") in pairs


# ---------------------------------------------------------------------------
# T5 through the contract
# ---------------------------------------------------------------------------


def test_t5_writer_cell_installs_a_functionally_identical_parent_shape():
    request, result = _run(f"C13.T5.M16.L{LAYER:02d}", knobs={"active_k": 6})
    key = request.component.weight_keys[0]
    installed = result.installed_weights[key]
    assert installed.shape == request.parent_weights[key].shape
    kept = int(request.level_width)
    assert result.stored_parameter_count == HIDDEN * kept
    # Exactly ``level_width`` input columns survive; the rest are zero, which is
    # what makes "the child gathers the kept channels" and "the parent-shaped
    # tensor" the same function.
    nonzero = int((installed.abs().sum(dim=0) > 0).sum().item())
    assert nonzero == kept
    rows = request.evidence.fit_rows(request.component.input_endpoint)
    assert torch.allclose(
        result.predict(rows), rows @ installed.transpose(0, 1), atol=1e-10
    )
    assert result.solve["omp"]["double_mgs"] is True
    assert result.solve["write_solve"]["coupled"] is True


def test_t5_reader_cell_composes_the_route_into_the_parent_shape():
    request, result = _run(f"C11.T5.M16.L{LAYER:02d}", knobs={"active_k": 6})
    key = request.component.weight_keys[0]
    installed = result.installed_weights[key]
    assert installed.shape == (INTERMEDIATE, HIDDEN)
    kept = int(request.level_width)
    assert result.stored_parameter_count == kept * HIDDEN + int(
        result.solve["route_support_entries"]
    )
    rows = request.evidence.fit_rows(request.component.input_endpoint)
    assert torch.allclose(
        result.predict(rows), rows @ installed.transpose(0, 1), atol=1e-9
    )
    # The kept channels are carried exactly; only the dropped ones are routed.
    parent = request.parent_weights[key]
    reference = rows @ parent.transpose(0, 1)
    routed = rows @ installed.transpose(0, 1)
    kept_ids = torch.nonzero(
        (installed - parent).abs().sum(dim=1) <= 1e-12, as_tuple=False
    ).flatten()
    assert int(kept_ids.numel()) >= kept
    assert torch.allclose(
        routed.index_select(1, kept_ids),
        reference.index_select(1, kept_ids),
        atol=1e-9,
    )


def test_t5_conv_in_proj_shares_one_route_across_the_three_blocks():
    request, result = _run(f"C03.T5.M16.L{LAYER:02d}", knobs={"active_k": 6})
    assert result.solve["block_count"] == 3
    assert result.solve["kept_width"] * 3 == int(request.level_width)
    key = request.component.weight_keys[0]
    installed = result.installed_weights[key]
    assert installed.shape == (3 * HIDDEN, HIDDEN)
    assert result.stored_parameter_count == (
        result.solve["kept_width"] * 3 * HIDDEN
        + int(result.solve["route_support_entries"])
    )


def test_t5_refuses_a_component_axis_it_cannot_route():
    request = _request("C14.T1.NA.L00")
    with pytest.raises(CellTechniqueError, match="linear channel axis"):
        t5_body_route_technique(request)


# ---------------------------------------------------------------------------
# T7 through the contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("component", "side"), [("C06", "query"), ("C07", "key")])
def test_t7_installs_one_side_against_a_bit_identical_partner(component, side):
    knobs = {
        "head_dim": HEAD_DIM,
        "query_heads": QUERY_HEADS,
        "key_value_heads": KV_HEADS,
    }
    request, result = _run(
        f"{component}.T7.M4.L{ATTENTION_LAYER:02d}", knobs=knobs
    )
    key = request.component.weight_keys[0]
    installed = result.installed_weights[key]
    assert installed.shape == request.parent_weights[key].shape
    assert result.solve["installed_side"] == side
    assert result.solve["partner_is_copy_exact"] is True
    # The single-side realization attains the *joint* optimum, not a fallback.
    assert result.solve["realized_score_kernel_relative_error_max"] == pytest.approx(
        result.solve["calibration"]["score_kernel_relative_error_max"],
        rel=1e-6,
        abs=1e-9,
    )
    # The contraction is the rank of the installed tensor, per head.
    heads = QUERY_HEADS if side == "query" else KV_HEADS
    child_dim = int(request.level_width) // heads
    for head in range(heads):
        block = installed[head * HEAD_DIM : (head + 1) * HEAD_DIM, :]
        assert int(torch.linalg.matrix_rank(block, tol=1e-9).item()) <= child_dim
    assert result.stored_parameter_count == heads * child_dim * (
        HIDDEN + HEAD_DIM
    )


def test_t7_at_no_contraction_reproduces_the_parent_score_kernel():
    knobs = {
        "head_dim": HEAD_DIM,
        "query_heads": QUERY_HEADS,
        "key_value_heads": KV_HEADS,
    }
    request = _request(f"C06.T7.M4.L{ATTENTION_LAYER:02d}", knobs=knobs)
    # Drive the kernel at the parent's own head_dim by handing it the full width.
    full = CellTechniqueRequest(
        cell=request.cell,
        component=request.component,
        level=request.level,
        level_width=QUERY_HEADS * HEAD_DIM,
        seed=request.seed,
        parent_weights=request.parent_weights,
        evidence=request.evidence,
        knobs=request.knobs,
        workspace=request.workspace,
        log=request.log,
    )
    result = t7_attention_relation_technique(full)
    assert result.solve["realized_score_kernel_relative_error_max"] < 1e-6


def test_t7_refuses_a_component_that_is_not_the_qk_pair():
    request = _request(f"C08.T5.M16.L{ATTENTION_LAYER:02d}")
    with pytest.raises(CellTechniqueError, match="bilinear Q/K cell"):
        t7_attention_relation_technique(request)


# ---------------------------------------------------------------------------
# T8 through the contract
# ---------------------------------------------------------------------------


def test_t8_input_axis_cell_installs_a_contracted_writer():
    request, result = _run(f"C05.T8.M4.L{LAYER:02d}")
    key = request.component.weight_keys[0]
    installed = result.installed_weights[key]
    assert installed.shape == (HIDDEN, HIDDEN)
    kept = int(request.level_width)
    assert result.stored_parameter_count == HIDDEN * kept
    assert int((installed.abs().sum(dim=0) > 0).sum().item()) == kept
    assert result.solve["row_weighting"] == "inverse-parent-row-norm-squared"


def test_t8_output_axis_cell_declares_a_clamped_relation_rank():
    # An output-axis level asks for a stored width (72) wider than this
    # component's own input (48), so the rank cap cannot be the binding
    # constraint.  The cell still runs — the scale-free weighting is live — and
    # says plainly that the truncation did nothing.
    request, result = _run(f"C11.T8.M4.L{LAYER:02d}")
    key = request.component.weight_keys[0]
    installed = result.installed_weights[key]
    assert installed.shape == (INTERMEDIATE, HIDDEN)
    assert result.solve["requested_relation_rank"] == int(request.level_width)
    assert result.solve["relation_rank_clamped"] is True
    assert result.solve["relation_rank"] <= HIDDEN
    assert result.solve["gram_truncation_active"] is False
    rank = int(torch.linalg.matrix_rank(installed, tol=1e-9).item())
    assert rank <= result.solve["relation_rank"]
    assert result.stored_parameter_count == result.solve["relation_rank"] * (
        INTERMEDIATE + HIDDEN
    )


# ---------------------------------------------------------------------------
# T4 through the contract
# ---------------------------------------------------------------------------


def test_t4_marginal_level_on_a_square_writer_fails_closed():
    with pytest.raises(CellTechniqueError, match="only shrinks at rank"):
        _run(f"C05.T4.M16.L{LAYER:02d}")


def test_t4_rectangular_writer_clamps_to_the_full_rank_and_round_trips():
    # C13's level is a fraction of its 96-wide *input*, which exceeds the
    # writer's rank ceiling of 48; the kernel clamps and declares it.
    request, result = _run(
        f"C13.T4.M4.L{LAYER:02d}", knobs={"writer_rank": 20}
    )
    key = request.component.weight_keys[0]
    installed = result.installed_weights[key]
    assert installed.shape == (HIDDEN, INTERMEDIATE)
    assert result.solve["writer_rank"] == 20
    assert result.stored_parameter_count == 20 * (HIDDEN + INTERMEDIATE)
    assert result.solve["shrinks_the_writer"] is True
    clamped = _run(f"C13.T4.M4.L{LAYER:02d}", knobs={
        "t4_allow_parameter_increase": True
    })[1]
    assert clamped.solve["writer_rank_clamped"] is True
    assert clamped.solve["requested_writer_rank"] == int(request.level_width)
    rows = request.evidence.fit_rows(request.component.input_endpoint)
    assert torch.allclose(
        result.predict(rows), rows @ installed.transpose(0, 1), atol=1e-9
    )


def test_t4_override_admits_a_deliberate_non_shrinking_probe():
    _request_, result = _run(
        f"C05.T4.M16.L{LAYER:02d}", knobs={"t4_allow_parameter_increase": True}
    )
    assert result.solve["shrinks_the_writer"] is False
    assert result.stored_parameter_count > HIDDEN * HIDDEN
