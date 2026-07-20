from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_cpcf_boundary_bundle import (  # noqa: E402
    LFM2_CPCF_LAYER_CHARTS_SCHEMA,
    Lfm2CpcfBoundaryBundle,
    Lfm2CpcfBoundaryBundleError,
    Lfm2CpcfLayerBoundaryCharts,
    build_lfm2_cpcf_causal_anchor_splits,
    fit_lfm2_cpcf_boundary_bundle,
)
from neural_forge.hf_cpcf_boundary_chart import (  # noqa: E402
    fit_cpcf_orthogonal_chart_transport,
)
from neural_forge.hf_lfm2_cpcf_installability import (  # noqa: E402
    Lfm2CpcfInstallableLayerEvidence,
    Lfm2CpcfInstallablePhaseEvidence,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (  # noqa: E402
    build_lfm2_cpcf_target_phases,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (  # noqa: E402
    Lfm2CpcfModelFoldPolicy,
)
from neural_forge.hf_lfm2_cpcf_v5_model_compiler import (  # noqa: E402
    _SelectiveBoundaryCompiler,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


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
    phase: str,
    sampled: int,
    batch_size: int,
    selection_scale: float,
):
    phase_index = (
        0
        if phase in {"prefill", "cached_prefill"}
        else int(phase.removeprefix("decode.")) + 1
    )
    coordinate = torch.stack(
        (
            torch.linspace(
                0.4 + ordinal * 0.17 + phase_index * 0.07,
                1.1 + ordinal * 0.17 + phase_index * 0.07,
                sampled,
                dtype=torch.float64,
            ),
            torch.linspace(
                -0.8 + ordinal * 0.11 - phase_index * 0.05,
                0.6 + ordinal * 0.11 - phase_index * 0.05,
                sampled,
                dtype=torch.float64,
            ),
        ),
        dim=1,
    )
    if ordinal == 0:
        coordinate = coordinate * selection_scale
    batch_coordinates = torch.stack(
        [
            coordinate
            + torch.tensor(
                [batch_row * 0.03, -batch_row * 0.02],
                dtype=torch.float64,
            )
            for batch_row in range(batch_size)
        ],
        dim=0,
    )
    values = [(batch_coordinates @ mapping).contiguous() for mapping in _STATE_MAPS]
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
                f"layer-0-{ordinal}-{phase}-{batch_size}-{selection_scale}"
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
                f"layer-1-{ordinal}-{phase}-{batch_size}-{selection_scale}"
            ),
        ),
    )
    return Lfm2CpcfInstallablePhaseEvidence(
        cache_key=_sha("synthetic-cache"),
        batch_index=ordinal,
        input_sha256=_sha(f"donor-{ordinal}"),
        phase=phase,
        layers=layers,
        final_norm_tensors={
            "output.final_norm_output": values[4],
        },
        evidence_sha256=_sha(f"phase-{ordinal}-{phase}-{batch_size}-{selection_scale}"),
    )


def _phases(
    *,
    selection_scale: float = 1.0,
    validation_forced_shift: int = 0,
    batch_size: int = 1,
    cached_prefill: bool = False,
):
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
                )
                .unsqueeze(0)
                .expand(batch_size, -1),
                attention_mask=torch.ones(
                    batch_size,
                    sampled,
                    dtype=torch.long,
                ),
                target_forced_decode_ids=torch.tensor(
                    [
                        [
                            (
                                ordinal
                                + batch_row
                                + 1
                                + (
                                    validation_forced_shift
                                    if split == "validation"
                                    else 0
                                )
                            )
                            % 8,
                            (
                                ordinal
                                + batch_row
                                + 2
                                + (
                                    validation_forced_shift
                                    if split == "validation"
                                    else 0
                                )
                            )
                            % 8,
                        ]
                        for batch_row in range(batch_size)
                    ],
                    dtype=torch.long,
                ),
            )
        )
        for phase, phase_sampled in (
            ("cached_prefill" if cached_prefill else "prefill", sampled),
            ("decode.000", 1),
            ("decode.001", 1),
        ):
            evidence.append(
                _parent_phase(
                    ordinal=ordinal,
                    phase=phase,
                    sampled=phase_sampled,
                    batch_size=batch_size,
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


def _bundle(
    *,
    selection_scale: float = 1.0,
    validation_forced_shift: int = 0,
):
    return fit_lfm2_cpcf_boundary_bundle(
        _phases(
            selection_scale=selection_scale,
            validation_forced_shift=validation_forced_shift,
        ),
        _source_table(),
        target_hidden_width=2,
        state_reconstruction_weight=0.0,
        minimum_predictive_energy_fraction=0.90,
        minimum_validation_block_retention=0.90,
    )


def test_causal_anchor_extraction_uses_one_endpoint_per_labeled_phase():
    phases = _phases()
    splits = build_lfm2_cpcf_causal_anchor_splits(phases)

    assert {name: split.row_count for name, split in splits.items()} == {
        "selection": 2,
        "fit": 4,
        "validation": 4,
    }
    selection = splits["selection"]
    assert selection.row_ids == (
        "selection/batch.000000/record.row-0/batch_row.0000/"
        "causal.0000/prefix.00000002",
        "selection/batch.000000/record.row-0/batch_row.0000/"
        "causal.0001/prefix.00000003",
    )
    assert selection.prefix_positions.tolist() == [2, 3]
    assert selection.target_next_token_ids.tolist() == [1, 2]
    record_phases = [phase for phase in phases if phase.record_id == "row-0"]
    expected = torch.stack(
        [
            record_phases[0].parent_evidence.final_norm_tensors[
                "output.final_norm_output"
            ][0, -1],
            record_phases[1].parent_evidence.final_norm_tensors[
                "output.final_norm_output"
            ][0, -1],
        ]
    )
    torch.testing.assert_close(selection.source_producer_rows, expected)
    final_endpoint = record_phases[2].parent_evidence.final_norm_tensors[
        "output.final_norm_output"
    ][0, -1]
    assert not any(
        torch.equal(row, final_endpoint) for row in selection.source_producer_rows
    )


def test_causal_anchor_extraction_supports_batched_cached_prefill():
    phases = _phases(batch_size=2, cached_prefill=True)
    splits = build_lfm2_cpcf_causal_anchor_splits(phases)

    assert {name: split.row_count for name, split in splits.items()} == {
        "selection": 4,
        "fit": 8,
        "validation": 8,
    }
    selection = splits["selection"]
    assert selection.row_ids == (
        "selection/batch.000000/record.row-0/batch_row.0000/"
        "causal.0000/prefix.00000002",
        "selection/batch.000000/record.row-0/batch_row.0001/"
        "causal.0000/prefix.00000002",
        "selection/batch.000000/record.row-0/batch_row.0000/"
        "causal.0001/prefix.00000003",
        "selection/batch.000000/record.row-0/batch_row.0001/"
        "causal.0001/prefix.00000003",
    )
    assert selection.prefix_positions.tolist() == [2, 2, 3, 3]
    assert selection.target_next_token_ids.tolist() == [1, 2, 2, 3]
    record_phases = [phase for phase in phases if phase.record_id == "row-0"]
    assert record_phases[0].phase == "cached_prefill"
    expected = torch.cat(
        [
            record_phases[0].parent_evidence.final_norm_tensors[
                "output.final_norm_output"
            ][:, -1],
            record_phases[1].parent_evidence.final_norm_tensors[
                "output.final_norm_output"
            ][:, -1],
        ],
        dim=0,
    )
    torch.testing.assert_close(selection.source_producer_rows, expected)


def test_causal_anchor_extraction_rejects_noncanonical_phase_order():
    phases = list(_phases())
    phases[0], phases[1] = phases[1], phases[0]
    with pytest.raises(
        Lfm2CpcfBoundaryBundleError,
        match="prefill/decode order|canonical prefill",
    ):
        build_lfm2_cpcf_causal_anchor_splits(phases)


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
    actual_child_pre = layer.post_operator.encode_shaped(parent_post_operator)
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
    assert (
        bundle.report["coordinate_contract"]["one_global_residual_basis_used"] is False
    )
    assert (
        bundle.report["coordinate_contract"]["residual_write_formula"]
        == "encoded_parent_endpoint_minus_actual_child_pre_state"
    )
    assert bundle.report["source_coverage"]["full_source_coverage"] is True
    assert bundle.report["operation_audit"]["parent_forward_count"] == 0
    assert bundle.report["operation_audit"]["training_step_count"] == 0


def test_v5_selective_boundary_compiler_is_byte_exact_to_v4():
    phases = _phases()
    source_table = _source_table()
    policy = Lfm2CpcfModelFoldPolicy(
        state_reconstruction_weight=0.0,
        minimum_predictive_energy_fraction=0.90,
        minimum_validation_block_retention=0.90,
    )
    expected = fit_lfm2_cpcf_boundary_bundle(
        phases,
        source_table,
        target_hidden_width=2,
        state_reconstruction_weight=0.0,
        ridge=policy.ridge,
        minimum_predictive_energy_fraction=0.90,
        minimum_validation_block_retention=0.90,
    )
    layer_kinds = tuple(
        layer.operator_kind for layer in phases[0].parent_evidence.layers
    )
    catalog = SimpleNamespace(
        source_width=4,
        layer_count=len(layer_kinds),
        phase_references=tuple(
            SimpleNamespace(layer_operator_kinds=layer_kinds) for _phase in phases
        ),
    )
    compiler = _SelectiveBoundaryCompiler(
        phases,
        catalog,
        source_table,
        target_hidden_width=2,
        model_policy=policy,
    )
    compiler.fit_interface(phases, target_norm_epsilon=1e-5)
    for layer_index in range(len(layer_kinds)):
        compiler.fit_layer(phases, layer_index=layer_index)
    observed = compiler.finalize()

    assert observed.bundle_sha256 == expected.bundle_sha256
    assert observed.report == expected.report
    assert [layer.as_dict() for layer in observed.layers] == [
        layer.as_dict() for layer in expected.layers
    ]


def test_v5_combined_profile_persists_only_upstream_transported_chart():
    phases = _phases()
    source_table = _source_table()
    policy = Lfm2CpcfModelFoldPolicy(
        state_reconstruction_weight=0.0,
        minimum_predictive_energy_fraction=0.90,
        minimum_validation_block_retention=0.90,
    )
    layer_kinds = tuple(
        layer.operator_kind for layer in phases[0].parent_evidence.layers
    )
    catalog = SimpleNamespace(
        source_width=4,
        layer_count=len(layer_kinds),
        phase_references=tuple(
            SimpleNamespace(layer_operator_kinds=layer_kinds) for _phase in phases
        ),
    )
    compiler = _SelectiveBoundaryCompiler(
        phases,
        catalog,
        source_table,
        target_hidden_width=2,
        model_policy=policy,
    )
    compiler.fit_interface(phases, target_norm_epsilon=1e-5)
    source = compiler.fit_layer(phases, layer_index=0)
    parent_rows = torch.randn(
        12,
        source.source_width,
        generator=torch.Generator().manual_seed(9773),
        dtype=torch.float64,
    )
    rotation = torch.tensor(
        [[0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float64,
    )
    transport = fit_cpcf_orthogonal_chart_transport(
        source.post_operator,
        fit_actual_child_rows=source.post_operator.encode(parent_rows) @ rotation,
        fit_parent_states=parent_rows,
        fit_provenance="test/upstream-fit-only",
    )
    assert transport.chart is not None
    layer_body = {
        "schema": LFM2_CPCF_LAYER_CHARTS_SCHEMA,
        "layer_index": source.layer_index,
        "operator_kind": source.operator_kind,
        "source_width": source.source_width,
        "target_width": source.target_width,
        "post_operator_report_sha256": transport.chart.report["report_sha256"],
        "post_layer_report_sha256": source.post_layer.report["report_sha256"],
        "independent_boundary_coordinates": True,
    }
    effective = Lfm2CpcfLayerBoundaryCharts(
        layer_index=source.layer_index,
        operator_kind=source.operator_kind,
        post_operator=transport.chart,
        post_layer=source.post_layer,
        layer_charts_sha256=_canonical_sha256(layer_body),
    )
    controller_body = {
        "component": "fit_only_post_operator_chart_transport",
        "production_profile_authorized": True,
        "source_layer_charts_sha256": source.layer_charts_sha256,
        "transported_layer_charts_sha256": effective.layer_charts_sha256,
        "unchanged_post_layer_report_sha256": (
            source.post_layer.report["report_sha256"]
        ),
        "transported_post_operator_report_sha256": (
            effective.post_operator.report["report_sha256"]
        ),
    }
    controller = {
        **controller_body,
        "report_sha256": _canonical_sha256(controller_body),
    }

    compiler.replace_latest_layer_charts(
        source,
        effective,
        controller_report=controller,
    )
    for layer_index in range(1, len(layer_kinds)):
        compiler.fit_layer(phases, layer_index=layer_index)
    observed = compiler.finalize()

    assert observed.layers[0] is effective
    assert observed.layers[0].layer_charts_sha256 == effective.layer_charts_sha256
    assert observed.layers[0].post_layer is source.post_layer
    assert observed.layers[0].post_operator is not source.post_operator


def test_fit_validation_are_disjoint_and_selection_cannot_affect_fits():
    baseline = _bundle(selection_scale=1.0)
    changed_selection = _bundle(selection_scale=10_000.0)

    assert (
        baseline.split_phase_sha256["fit"] != baseline.split_phase_sha256["validation"]
    )
    assert baseline.split_row_sha256["fit"] != baseline.split_row_sha256["validation"]
    assert baseline.report["split_row_counts"] == {
        "selection": 5,
        "fit": 11,
        "validation": 11,
    }
    assert baseline.report["selection"]["used_to_fit_boundary_charts"] is False
    assert baseline.report["selection"]["used_to_fit_causal_anchor_values"] is False
    assert (
        baseline.report["selection"]["used_to_select_causal_anchor_ridge_margin"]
        is True
    )
    assert baseline.report["selection"]["used_for_validation_gates"] is False
    phase_weights = baseline.report["phase_weighting"]
    assert phase_weights["balanced_phase_weights"] is True
    assert [
        value["total_phase_weight"] for value in phase_weights["fit_phases"]
    ] == pytest.approx([1.0 / 6.0] * 6)
    assert baseline.report["interface_top1"]["diagnostic_only"] is True
    assert baseline.report["interface_top1"]["capacity_gate_used"] is False

    assert (
        baseline.report["interface_lineage"]["base_interface_report"]
        == changed_selection.report["interface_lineage"]["base_interface_report"]
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
        != changed_selection.report["source_coverage"]["source_observation_sha256"]
    )


def test_anchor_integration_is_deterministic_and_preserves_both_seals():
    first = _bundle()
    second = _bundle()

    torch.testing.assert_close(
        first.interface.target_weight,
        second.interface.target_weight,
    )
    torch.testing.assert_close(
        first.interface.producer_encoder,
        second.interface.producer_encoder,
    )
    assert first.report == second.report
    lineage = first.report["interface_lineage"]
    assert (
        lineage["base_interface_report_sha256"]
        == (first.report["base_interface_report_sha256"])
    )
    assert (
        lineage["causal_anchor_report_sha256"]
        == first.report["causal_anchor_report_sha256"]
    )
    assert (
        lineage["causal_anchor_report_sha256"]
        == first.report["interface_report_sha256"]
    )
    assert lineage["active_interface_is_anchor_repaired"] is False
    assert lineage["active_interface_is_base_control"] is True
    assert lineage["causal_anchor_report"]["selected_candidate_is_base_control"] is True
    assert (
        lineage["causal_anchor_report"][
            "target_anchor_metrics_used_for_candidate_selection"
        ]
        is False
    )
    assert lineage["additional_runtime_parameter_count"] == 0
    assert (
        first.report["causal_anchor_extraction"]["no_sampled_prefix_row_duplication"]
        is True
    )
    assert (
        first.report["causal_anchor_extraction"][
            "row_id_binds_batch_index_batch_row_and_prefix_position"
        ]
        is True
    )
    assert (
        first.report["coordinate_contract"]["active_interface_is_anchor_repaired"]
        is False
    )
    assert (
        first.report["coordinate_contract"]["active_interface_is_base_control"] is True
    )


def test_bundle_rejects_a_resealed_substitute_base_interface_report():
    bundle = _bundle()
    report = copy.deepcopy(dict(bundle.report))
    lineage = report["interface_lineage"]
    substitute_body = dict(lineage["base_interface_report"])
    substitute_body.pop("report_sha256")
    substitute_body["substituted_after_anchor_fit"] = True
    substitute = {
        **substitute_body,
        "report_sha256": _canonical_sha256(substitute_body),
    }
    lineage["base_interface_report"] = substitute
    lineage["base_interface_report_sha256"] = substitute["report_sha256"]
    report["base_interface_report_sha256"] = substitute["report_sha256"]
    report_body = dict(report)
    report_body.pop("bundle_sha256")
    resealed_bundle_sha256 = _canonical_sha256(report_body)
    report["bundle_sha256"] = resealed_bundle_sha256

    with pytest.raises(
        Lfm2CpcfBoundaryBundleError,
        match="lineage seal is inconsistent",
    ):
        Lfm2CpcfBoundaryBundle(
            interface=bundle.interface,
            layers=bundle.layers,
            split_phase_sha256=bundle.split_phase_sha256,
            split_row_sha256=bundle.split_row_sha256,
            report=report,
            bundle_sha256=resealed_bundle_sha256,
        )


def test_validation_labels_cannot_change_anchor_or_fitted_coordinates():
    baseline = _bundle(validation_forced_shift=0)
    changed = _bundle(validation_forced_shift=1)

    torch.testing.assert_close(
        baseline.interface.target_weight,
        changed.interface.target_weight,
    )
    torch.testing.assert_close(
        baseline.interface.producer_encoder,
        changed.interface.producer_encoder,
    )
    baseline_anchor = baseline.report["interface_lineage"]["causal_anchor_report"]
    changed_anchor = changed.report["interface_lineage"]["causal_anchor_report"]
    assert (
        baseline_anchor["selected_candidate_index"]
        == (changed_anchor["selected_candidate_index"])
    )
    assert baseline_anchor["candidate_reports"] == changed_anchor["candidate_reports"]
    assert (
        baseline_anchor["residual_weight_sha256"]
        == changed_anchor["residual_weight_sha256"]
    )
    assert (
        baseline_anchor["split_metrics"]["validation"]
        != changed_anchor["split_metrics"]["validation"]
    )
    for left, right in zip(baseline.layers, changed.layers, strict=True):
        torch.testing.assert_close(
            left.post_operator.mapping.encoder,
            right.post_operator.mapping.encoder,
        )
        torch.testing.assert_close(
            left.post_layer.mapping.encoder,
            right.post_layer.mapping.encoder,
        )
