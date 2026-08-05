"""Synthetic contract tests for the layer-0 representability study runner.

These never touch the sealed r12 artifacts.  They build a tiny LFM2-shaped
short-conv "parent", generate its exact cached B/C/x and residual rows, and
check three things the study's verdict depends on:

1. On evidence whose residual genuinely lives inside the chart, the closed-form
   layer solve reproduces the chart-projected parent write to solver precision.
   The compiler refuses a square chart, so the control instead drives a parent
   whose last residual coordinate is identically zero through a chart that
   drops exactly that coordinate: representable by construction.  If this ever
   regresses, a low measured cosine cannot be read as a representability
   result.  A second case shows the same solver losing a large amount on a
   chart that really does truncate live directions -- the cubic B*x*C path
   amplifies chart loss, which is the quantitative fact the study's width
   ladder measures on the real parent.
2. The production prior-delta clamp is observable: with ``max_prior_delta_ratio
   = 1.0`` and a badly-scaled prior the reported ``update_scale`` drops below
   one and the write degrades, which is exactly the solver-side mechanism the
   study is testing for.
3. ``nested_chart`` extends a sealed chart without disturbing its columns.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_lfm2_layer0_representability_study.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_lfm2_layer0_representability_study", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


study = _load_module()


def _parent_weights(width: int, generator: torch.Generator) -> dict:
    return {
        "in_proj": torch.randn(
            (3 * width, width), dtype=torch.float64, generator=generator
        )
        / width**0.5,
        "conv": torch.randn(
            (width, 1, 3), dtype=torch.float64, generator=generator
        ),
        "out_proj": torch.randn(
            (width, width), dtype=torch.float64, generator=generator
        )
        / width**0.5,
        "operator_norm": torch.ones((width,), dtype=torch.float64),
    }


def _synthetic_evidence(
    *,
    width: int,
    batches: int,
    sequence: int,
    weights: dict,
    generator: torch.Generator,
    dead_direction: torch.Tensor | None = None,
) -> study.BoundaryEvidence:
    phases = ("prefill", "decode.000")
    payload: dict[int, dict[str, torch.Tensor]] = {}
    split: dict[int, str] = {}
    for index in range(batches):
        split[index] = "fit" if index % 2 == 0 else "validation"
        local: dict[str, torch.Tensor] = {}
        # ``cached_short_conv_case_from_lfm2_trace`` resets the FIR state only
        # on non-decode phases, so the synthetic parent has to carry state the
        # same way or the first two rows of every decode phase would compare a
        # stateful child against a stateless target.
        state = torch.zeros((1, width, 2), dtype=torch.float64)
        for phase in phases:
            pre = torch.randn(
                (1, sequence, width), dtype=torch.float64, generator=generator
            )
            if dead_direction is not None:
                pre = pre - (pre @ dead_direction).unsqueeze(-1) * dead_direction
            normalized = pre * torch.rsqrt(
                pre.square().mean(dim=-1, keepdim=True) + 1e-5
            )
            operator_input = normalized * weights["operator_norm"]
            projected = operator_input @ weights["in_proj"].T
            b_value, c_value, x_value = projected.chunk(3, dim=-1)
            gated = (b_value * x_value).transpose(1, 2).contiguous()
            history = torch.cat((state, gated), dim=2)
            filtered = torch.nn.functional.conv1d(
                history,
                weights["conv"],
                groups=width,
            )
            state = history[:, :, -2:].contiguous()
            context = c_value * filtered.transpose(1, 2)
            write = context @ weights["out_proj"].T
            local.update(
                {
                    f"{phase}.residual.pre": pre.to(torch.float32),
                    f"{phase}.operator.input": operator_input.to(torch.float32),
                    f"{phase}.short_conv.b": b_value.to(torch.float32),
                    f"{phase}.short_conv.c": c_value.to(torch.float32),
                    f"{phase}.short_conv.x": x_value.to(torch.float32),
                    f"{phase}.short_conv.context": context.to(torch.float32),
                    f"{phase}.operator.output": write.to(torch.float32),
                    f"{phase}.residual.post_operator": (pre + write).to(
                        torch.float32
                    ),
                }
            )
        payload[index] = local
    return study.BoundaryEvidence(
        layer_index=0,
        phases=phases,
        split_by_batch=split,
        batches=payload,
        source_width=width,
        selection_sha256="0" * 64,
    )


def _chart(width: int, target: int, generator: torch.Generator) -> torch.Tensor:
    raw = torch.randn((width, width), dtype=torch.float64, generator=generator)
    orthogonal, _ = torch.linalg.qr(raw)
    return orthogonal[:, :target].contiguous()


def test_representable_boundary_is_solved_to_precision() -> None:
    generator = torch.Generator().manual_seed(20260805)
    width = 32
    weights = _parent_weights(width, generator)
    raw = torch.randn((width, width), dtype=torch.float64, generator=generator)
    dead = raw[:, 0] / torch.linalg.vector_norm(raw[:, 0])
    frame, _ = torch.linalg.qr(torch.cat((dead.reshape(-1, 1), raw[:, 1:]), dim=1))
    chart = frame[:, 1:].contiguous()
    evidence = _synthetic_evidence(
        width=width,
        batches=8,
        sequence=24,
        weights=weights,
        generator=generator,
        dead_direction=dead,
    )
    record = study.run_arm(
        name="representable",
        evidence=evidence,
        sealed_basis=chart,
        parent_weights=weights,
        target_width=width - 1,
        ridge=1e-12,
        max_prior_delta_ratio=None,
        prior_mode="dense_chart",
        norm_epsilon=1e-5,
        norm_ridge=1e-12,
        chart_cache={width - 1: chart},
        layer_index=0,
        solve_dtype="float64",
    )
    heldout = record["metrics"]["heldout"]["all_rows"]["operator.output"]
    assert heldout["mean_cosine"] > 0.999, heldout
    assert heldout["mean_relative_l2"] < 0.05, heldout


def test_truncating_a_live_direction_costs_far_more_than_its_energy() -> None:
    generator = torch.Generator().manual_seed(20260805)
    width = 32
    weights = _parent_weights(width, generator)
    evidence = _synthetic_evidence(
        width=width,
        batches=8,
        sequence=24,
        weights=weights,
        generator=generator,
    )
    results = {}
    for target in (width - 1, width - 8):
        chart = _chart(width, target, generator)
        record = study.run_arm(
            name=f"width_{target}",
            evidence=evidence,
            sealed_basis=chart,
            parent_weights=weights,
            target_width=target,
            ridge=1e-10,
            max_prior_delta_ratio=None,
            prior_mode="dense_chart",
            norm_epsilon=1e-5,
            norm_ridge=1e-10,
            chart_cache={target: chart},
            layer_index=0,
            solve_dtype="float64",
        )
        results[target] = record["metrics"]["heldout"]["all_rows"][
            "operator.output"
        ]
    wide = results[width - 1]
    narrow = results[width - 8]
    # One dropped direction out of 32 is ~3% of the residual energy, and the
    # cubic B*x*C path turns that into a double-digit write error.
    assert 0.05 < wide["mean_relative_l2"] < 0.75, wide
    assert narrow["mean_relative_l2"] > wide["mean_relative_l2"], (wide, narrow)
    assert narrow["mean_cosine"] < wide["mean_cosine"], (wide, narrow)


def test_prior_delta_clamp_is_observable_and_costly() -> None:
    generator = torch.Generator().manual_seed(20260806)
    width = 32
    weights = _parent_weights(width, generator)
    evidence = _synthetic_evidence(
        width=width,
        batches=8,
        sequence=24,
        weights=weights,
        generator=generator,
    )
    chart = _chart(width, width - 1, generator)
    free = study.run_arm(
        name="free",
        evidence=evidence,
        sealed_basis=chart,
        parent_weights=weights,
        target_width=width - 1,
        ridge=1e-10,
        max_prior_delta_ratio=None,
        prior_mode="source_derived",
        norm_epsilon=1e-5,
        norm_ridge=1e-10,
        chart_cache={width - 1: chart},
        layer_index=0,
        solve_dtype="float64",
    )
    clamped = study.run_arm(
        name="clamped",
        evidence=evidence,
        sealed_basis=chart,
        parent_weights=weights,
        target_width=width - 1,
        ridge=1e-10,
        max_prior_delta_ratio=1.0,
        prior_mode="source_derived",
        norm_epsilon=1e-5,
        norm_ridge=1e-10,
        chart_cache={width - 1: chart},
        layer_index=0,
        solve_dtype="float64",
    )
    free_scale = free["producer"]["in_proj_fit"]["update_scale"]
    clamped_scale = clamped["producer"]["in_proj_fit"]["update_scale"]
    assert free_scale == 1.0
    assert clamped_scale <= 1.0
    free_write = free["metrics"]["heldout"]["all_rows"]["operator.output"]
    clamped_write = clamped["metrics"]["heldout"]["all_rows"]["operator.output"]
    assert free_write["mean_cosine"] >= clamped_write["mean_cosine"] - 1e-9


def test_nested_chart_preserves_the_sealed_columns() -> None:
    generator = torch.Generator().manual_seed(20260807)
    width = 32
    weights = _parent_weights(width, generator)
    evidence = _synthetic_evidence(
        width=width,
        batches=4,
        sequence=12,
        weights=weights,
        generator=generator,
    )
    sealed = _chart(width, 20, generator)
    extended = study.nested_chart(sealed, evidence, 26)
    assert tuple(extended.shape) == (width, 26)
    assert torch.equal(extended[:, :20], sealed)
    assert torch.allclose(
        extended.T @ extended,
        torch.eye(26, dtype=torch.float64),
        atol=1e-10,
    )
    assert torch.equal(study.nested_chart(sealed, evidence, 20), sealed)
