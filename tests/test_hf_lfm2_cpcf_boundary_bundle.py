from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_cpcf_boundary_bundle import (  # noqa: E402
    fit_lfm2_cpcf_boundary_bundle,
)
from neural_forge.hf_lfm2_cpcf_installability import (  # noqa: E402
    Lfm2CpcfInstallableLayerEvidence,
    Lfm2CpcfInstallablePhaseEvidence,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (  # noqa: E402
    build_lfm2_cpcf_target_phases,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Record:
    ordinal: int
    record_id: str
    split: str
    donor_cache_input_sha256: str
    alignment_sha256: str
    target_input_ids: object
    attention_mask: object
    target_forced_decode_ids: object
    position_ids: object = None


class _Adapter:
    def __init__(self, records):
        self.records = tuple(records)

    def verify(self):
        return self


_STATE_MAPS = (
    torch.tensor(
        [
            [1.0, 0.2, -0.4, 0.7],
            [-0.3, 0.9, 0.5, -0.2],
        ],
        dtype=torch.float64,
    ),
    torch.tensor(
        [
            [0.6, -0.8, 0.4, 1.1],
            [0.5, 0.3, -0.7, 0.2],
        ],
        dtype=torch.float64,
    ),
    torch.tensor(
        [
            [0.8, 0.4, -0.1, 0.5],
            [-0.2, 0.7, 0.6, -0.4],
        ],
        dtype=torch.float64,
    ),
    torch.tensor(
        [
            [0.3, -0.5, 0.9, 0.8],
            [0.7, 0.4, -0.2, 0.6],
        ],
        dtype=torch.float64,
    ),
    torch.tensor(
        [
            [0.9, -0.1, 0.3, 0.6],
            [0.4, 0.8, -0.5, 0.2],
        ],
        dtype=torch.float64,
    ),
)


def _parent_phase(
    *,
    ordinal: int,
    sampled: int,
    selection_scale: float,
):
    coordinate = torch.stack(
        (
            torch.linspace(
                0.4 + ordinal * 0.17,
                1.1 + ordinal * 0.17,
                sampled,
                dtype=torch.float64,
            ),
            torch.linspace(
                -0.8 + ordinal * 0.11,
                0.6 + ordinal * 0.11,
                sampled,
                dtype=torch.float64,
            ),
        ),
        dim=1,
    )
    if ordinal == 0:
        coordinate = coordinate * selection_scale
    values = [
        (coordinate @ mapping).unsqueeze(0).contiguous()
        for mapping in _STATE_MAPS
    ]
    layers = (
        Lfm2CpcfInstallableLayerEvidence(
            layer_index=0,
            operator_kind="short_conv",
            source_width=4,
            tensors={
                "residual.pre": values[0],
                "residual.post_operator": values[1],
                "residual.post_layer": values[2],
            },
            evidence_sha256=_sha(
                f"layer-0-{ordinal}-{selection_scale}"
            ),
        ),
        Lfm2CpcfInstallableLayerEvidence(
            layer_index=1,
            operator_kind="full_attention",
            source_width=4,
            tensors={
                "residual.pre": values[2],
                "residual.post_operator": values[3],
                "residual.post_layer": values[4],
            },
            evidence_sha256=_sha(
                f"layer-1-{ordinal}-{selection_scale}"
            ),
        ),
    )
    return Lfm2CpcfInstallablePhaseEvidence(
        cache_key=_sha("synthetic-cache"),
        batch_index=ordinal,
        input_sha256=_sha(f"donor-{ordinal}"),
        phase="prefill",
        layers=layers,
        final_norm_tensors={
            "output.final_norm_output": values[4],
        },
        evidence_sha256=_sha(
            f"phase-{ordinal}-{selection_scale}"
        ),
    )


def _phases(*, selection_scale: float = 1.0):
    specs = (
        ("selection", 3),
        ("fit", 2),
        ("fit", 5),
        ("validation", 3),
        ("validation", 4),
    )
    records = []
    evidence = []
    for ordinal, (split, sampled) in enumerate(specs):
        donor_sha = _sha(f"donor-{ordinal}")
        records.append(
            _Record(
                ordinal=ordinal,
                record_id=f"row-{ordinal}",
                split=split,
                donor_cache_input_sha256=donor_sha,
                alignment_sha256=_sha(f"alignment-{ordinal}"),
                target_input_ids=torch.arange(
                    2,
                    2 + sampled,
                    dtype=torch.long,
                ).unsqueeze(0),
                attention_mask=torch.ones(
                    1,
                    sampled,
                    dtype=torch.long,
                ),
                target_forced_decode_ids=torch.tensor(
                    [[17]],
                    dtype=torch.long,
                ),
            )
        )
        evidence.append(
            _parent_phase(
                ordinal=ordinal,
                sampled=sampled,
                selection_scale=selection_scale,
            )
        )
    return build_lfm2_cpcf_target_phases(
        _Adapter(records),
        tuple(evidence),
    )


def _source_table():
    return torch.tensor(
        [
            [0.8, -0.3, 0.5, 0.1],
            [-0.4, 0.9, 0.2, 0.7],
            [0.6, 0.4, -0.8, 0.3],
            [0.1, -0.7, 0.9, 0.5],
            [-0.9, 0.2, 0.4, -0.6],
            [0.3, 0.8, 0.1, -0.5],
            [0.7, -0.2, -0.4, 0.9],
            [-0.1, 0.5, 0.8, -0.7],
        ],
        dtype=torch.float64,
    )


def _bundle(*, selection_scale: float = 1.0):
    return fit_lfm2_cpcf_boundary_bundle(
        _phases(selection_scale=selection_scale),
        _source_table(),
        target_hidden_width=2,
        state_reconstruction_weight=0.0,
        minimum_predictive_energy_fraction=0.90,
        minimum_validation_block_retention=0.90,
    )


def test_bundle_charts_absorb_the_post_operator_to_post_layer_transition():
    phases = _phases()
    bundle = fit_lfm2_cpcf_boundary_bundle(
        phases,
        _source_table(),
        target_hidden_width=2,
        state_reconstruction_weight=0.0,
        minimum_predictive_energy_fraction=0.90,
        minimum_validation_block_retention=0.90,
    )
    validation = next(phase for phase in phases if phase.split == "validation")
    layer = bundle.layers[0]
    parent_post_operator = validation.parent_evidence.layers[0].tensors[
        "residual.post_operator"
    ]
    parent_post_layer = validation.parent_evidence.layers[0].tensors[
        "residual.post_layer"
    ]
    actual_child_pre = layer.post_operator.encode_shaped(
        parent_post_operator
    )
    encoded_endpoint, write = layer.post_layer.residual_write_target(
        actual_child_pre_state=actual_child_pre,
        parent_endpoint=parent_post_layer,
    )

    torch.testing.assert_close(
        actual_child_pre + write,
        encoded_endpoint,
    )
    torch.testing.assert_close(
        encoded_endpoint,
        layer.post_layer.encode_shaped(parent_post_layer),
    )
    assert bundle.report["coordinate_contract"][
        "one_global_residual_basis_used"
    ] is False
    assert bundle.report["coordinate_contract"][
        "residual_write_formula"
    ] == "encoded_parent_endpoint_minus_actual_child_pre_state"
    assert bundle.report["source_coverage"]["full_source_coverage"] is True
    assert bundle.report["operation_audit"]["parent_forward_count"] == 0
    assert bundle.report["operation_audit"]["training_step_count"] == 0


def test_fit_validation_are_disjoint_and_selection_cannot_affect_fits():
    baseline = _bundle(selection_scale=1.0)
    changed_selection = _bundle(selection_scale=10_000.0)

    assert (
        baseline.split_phase_sha256["fit"]
        != baseline.split_phase_sha256["validation"]
    )
    assert (
        baseline.split_row_sha256["fit"]
        != baseline.split_row_sha256["validation"]
    )
    assert baseline.report["split_row_counts"] == {
        "selection": 3,
        "fit": 7,
        "validation": 7,
    }
    assert baseline.report["selection"][
        "used_to_fit_boundary_charts"
    ] is False
    assert baseline.report["selection"]["used_for_validation_gates"] is False
    phase_weights = baseline.report["phase_weighting"]
    assert phase_weights["balanced_phase_weights"] is True
    assert [
        value["total_phase_weight"]
        for value in phase_weights["fit_phases"]
    ] == pytest.approx([0.5, 0.5])
    assert baseline.report["interface_top1"]["diagnostic_only"] is True
    assert baseline.report["interface_top1"]["capacity_gate_used"] is False

    torch.testing.assert_close(
        baseline.interface.target_weight,
        changed_selection.interface.target_weight,
    )
    torch.testing.assert_close(
        baseline.interface.producer_encoder,
        changed_selection.interface.producer_encoder,
    )
    for left, right in zip(
        baseline.layers,
        changed_selection.layers,
        strict=True,
    ):
        torch.testing.assert_close(
            left.post_operator.mapping.encoder,
            right.post_operator.mapping.encoder,
        )
        torch.testing.assert_close(
            left.post_layer.mapping.encoder,
            right.post_layer.mapping.encoder,
        )
    assert (
        baseline.report["source_coverage"]["source_observation_sha256"]
        != changed_selection.report["source_coverage"][
            "source_observation_sha256"
        ]
    )
