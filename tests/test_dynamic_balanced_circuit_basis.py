from __future__ import annotations

import ast
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_balanced_circuit_basis import (  # noqa: E402
    BalancedCircuitBasisError,
    BalancedCircuitBasisEvidenceError,
    BalancedCircuitBasisRankError,
    DownstreamObservability,
    EvidenceSplit,
    ResidualSnapshotBlock,
    build_balanced_circuit_basis,
)
from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    CircuitKind,
    CircuitNode,
    CircuitPort,
    ModalityInterface,
    ModelCircuitGraph,
    PortKind,
)
from neural_forge.dynamic_weight_complete_basis import (  # noqa: E402
    FeatureOrientation,
    residual_facing_operator,
)


def _graph() -> ModelCircuitGraph:
    width = 2
    attention = CircuitNode(
        node_id="layer.000.attention",
        kind=CircuitKind.ATTENTION,
        order=0,
        path="layers.0.attention",
        layer_index=0,
        inputs=(
            CircuitPort(
                "residual",
                PortKind.NORMALIZED_RESIDUAL,
                ("batch", "sequence", width),
            ),
        ),
        outputs=(
            CircuitPort(
                "delta",
                PortKind.BRANCH_DELTA,
                ("batch", "sequence", width),
            ),
        ),
    )
    mlp = CircuitNode(
        node_id="layer.000.mlp",
        kind=CircuitKind.GATED_MLP,
        order=1,
        path="layers.0.mlp",
        layer_index=0,
        inputs=(
            CircuitPort(
                "residual",
                PortKind.NORMALIZED_RESIDUAL,
                ("batch", "sequence", width),
            ),
        ),
        outputs=(
            CircuitPort(
                "delta",
                PortKind.BRANCH_DELTA,
                ("batch", "sequence", width),
            ),
        ),
    )
    head = CircuitNode(
        node_id="output.head",
        kind=CircuitKind.OUTPUT_HEAD,
        order=2,
        path="head",
        layer_index=None,
        inputs=(
            CircuitPort(
                "residual",
                PortKind.NORMALIZED_RESIDUAL,
                ("batch", "sequence", width),
            ),
        ),
        outputs=(
            CircuitPort(
                "logits",
                PortKind.LOGITS,
                ("batch", "sequence", 7),
                modality="text",
            ),
        ),
    )
    return ModelCircuitGraph(
        model_name="synthetic.balanced",
        family="synthetic",
        model_class="SyntheticBalancedCircuit",
        nodes=(attention, mlp, head),
        edges=(),
        interfaces=(
            ModalityInterface(
                interface_id="text.logits.output",
                modality="text",
                direction="output",
                node_id=head.node_id,
                port_name="logits",
                representation="logits",
            ),
        ),
        logical_parameter_count=0,
        unique_parameter_count=0,
        unique_parameter_bytes=0,
        config_fingerprint="1" * 64,
    )


def _structural():
    # Raw weight energy is 100,000,000x larger on x than y.
    return (
        residual_facing_operator(
            torch.diag(
                torch.tensor([100.0, 0.01], dtype=torch.float64)
            ),
            role="whole_model_weights",
            name="dominant_raw_weight",
            orientation=FeatureOrientation.INPUT,
        ),
    )


def _snapshots(*, heldout_y_scale: float = 1.0):
    dominant_rows = torch.tensor(
        [
            [20.0, 0.02],
            [20.0, -0.02],
            [-20.0, 0.02],
            [-20.0, -0.02],
        ],
        dtype=torch.float64,
    )
    fit = (
        ResidualSnapshotBlock(
            block_id="prefill.layer0.pre",
            split=EvidenceSplit.FIT,
            phase="prefill",
            layer="layer.000",
            endpoint="residual.pre",
            rows=dominant_rows,
            record_ids=("fit.prompt.0", "fit.prompt.1"),
        ),
        ResidualSnapshotBlock(
            block_id="decode.layer0.post",
            split=EvidenceSplit.FIT,
            phase="decode.000",
            layer="layer.000",
            endpoint="residual.post_layer",
            rows=dominant_rows * torch.tensor(
                [0.5, 2.0], dtype=torch.float64
            ),
            record_ids=("fit.prompt.0", "fit.prompt.1"),
        ),
    )
    heldout = (
        ResidualSnapshotBlock(
            block_id="heldout.layer0.post",
            split=EvidenceSplit.HELDOUT,
            phase="decode.001",
            layer="layer.000",
            endpoint="residual.post_layer",
            rows=torch.tensor(
                [
                    [0.0, heldout_y_scale],
                    [0.0, -heldout_y_scale],
                ],
                dtype=torch.float64,
            ),
            record_ids=("heldout.prompt.0",),
        ),
    )
    return fit, heldout


def _observability():
    attention_writer = residual_facing_operator(
        torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        role="attention_writer",
        name="attention.output",
        orientation=FeatureOrientation.OUTPUT,
    )
    mlp_writer = residual_facing_operator(
        torch.tensor([[0.0], [2.0]], dtype=torch.float64),
        role="mlp_writer",
        name="mlp.down",
        orientation=FeatureOrientation.OUTPUT,
    )
    tied_output = residual_facing_operator(
        torch.tensor([[0.0, 10.0]], dtype=torch.float64),
        role="tied_output_interface",
        name="tied.embedding_head",
        orientation=FeatureOrientation.INPUT,
    )
    return (
        DownstreamObservability(
            evidence_id="attention.writer",
            node_id="layer.000.attention",
            circuit_kind=CircuitKind.ATTENTION,
            operator=attention_writer,
        ),
        DownstreamObservability(
            evidence_id="mlp.writer",
            node_id="layer.000.mlp",
            circuit_kind=CircuitKind.GATED_MLP,
            operator=mlp_writer,
        ),
        DownstreamObservability(
            evidence_id="tied.output",
            node_id="output.head",
            circuit_kind=CircuitKind.OUTPUT_HEAD,
            operator=tied_output,
            interface_ids=("text.logits.output",),
        ),
    )


def _build(*, heldout_y_scale: float = 1.0):
    fit, heldout = _snapshots(heldout_y_scale=heldout_y_scale)
    return build_balanced_circuit_basis(
        _structural(),
        fit,
        heldout,
        _observability(),
        target_width=1,
        circuit_graph=_graph(),
        gram_regularization=1e-12,
        required_observability_kinds=(
            CircuitKind.ATTENTION,
            CircuitKind.GATED_MLP,
            CircuitKind.OUTPUT_HEAD,
        ),
        require_all_ir_writers=True,
        minimum_heldout_retained_energy_fraction=0.99,
    )


def test_balanced_circuit_basis_preserves_low_energy_output_critical_axis() -> None:
    result = _build()
    fit, _heldout = _snapshots()
    activation_gram = sum(
        (value.rows.T @ value.rows for value in fit),
        torch.zeros((2, 2), dtype=torch.float64),
    )
    raw_weight_gram = (
        _structural()[0].tensor.T @ _structural()[0].tensor
    )
    activation_pca = torch.linalg.eigh(activation_gram).eigenvectors[:, -1]
    raw_weight_basis = torch.linalg.eigh(raw_weight_gram).eigenvectors[:, -1]

    assert abs(float(activation_pca[0])) > 0.999999
    assert abs(float(raw_weight_basis[0])) > 0.999999
    assert abs(float(result.basis[1, 0])) > 0.999999
    assert abs(float(result.basis[1, 0])) > abs(float(result.basis[0, 0]))
    assert result.report["not_activation_pca"] is True
    assert result.report["not_raw_weight_svd"] is True
    assert result.report["not_independent_local_regression"] is True
    assert result.report["retention"][
        "heldout_cached_residual_fraction"
    ] == pytest.approx(1.0, abs=1e-12)
    assert result.report["regularized_balanced_truncation"][
        "unregularized_hankel_rank"
    ] == 1
    assert result.report["regularized_balanced_truncation"][
        "captured_hankel_squared_energy_fraction"
    ] == pytest.approx(1.0, abs=1e-12)
    assert result.report["circuit_ir"]["all_writer_nodes_covered"] is True
    assert result.report["circuit_ir"][
        "all_output_interfaces_covered"
    ] is True
    assert set(
        result.report["downstream_observability"][
            "required_circuit_kinds"
        ]
    ) == {"attention", "gated_mlp", "output_head"}


def test_runtime_maps_accept_empty_row_matrices_without_relaxing_width() -> None:
    result = _build()

    encoded = result.encode(torch.empty((0, result.source_width)))
    decoded = result.decode(torch.empty((0, result.target_width)))
    balanced_encoded = result.balanced_encode(
        torch.empty((0, result.source_width))
    )
    balanced_decoded = result.balanced_decode(
        torch.empty((0, result.target_width))
    )

    assert encoded.shape == (0, result.target_width)
    assert decoded.shape == (0, result.source_width)
    assert balanced_encoded.shape == (0, result.target_width)
    assert balanced_decoded.shape == (0, result.source_width)
    assert encoded.dtype == torch.float64
    assert decoded.dtype == torch.float64

    sequence_rows = torch.arange(
        2 * 3 * result.source_width,
        dtype=torch.float64,
    ).reshape(2, 3, result.source_width)
    encoded_sequences = result.encode(sequence_rows)
    balanced_encoded_sequences = result.balanced_encode(sequence_rows)
    decoded_sequences = result.decode(encoded_sequences)
    balanced_decoded_sequences = result.balanced_decode(
        balanced_encoded_sequences
    )

    assert encoded_sequences.shape == (2, 3, result.target_width)
    assert balanced_encoded_sequences.shape == (
        2,
        3,
        result.target_width,
    )
    assert decoded_sequences.shape == (2, 3, result.source_width)
    assert balanced_decoded_sequences.shape == (
        2,
        3,
        result.source_width,
    )
    assert torch.equal(
        encoded_sequences.reshape(-1, result.target_width),
        result.encode(
            sequence_rows.reshape(-1, result.source_width)
        ),
    )

    empty_sequences = torch.empty((2, 0, result.source_width))
    assert result.encode(empty_sequences).shape == (
        2,
        0,
        result.target_width,
    )
    assert result.balanced_encode(empty_sequences).shape == (
        2,
        0,
        result.target_width,
    )
    assert result.decode(
        torch.empty((2, 0, result.target_width))
    ).shape == (2, 0, result.source_width)
    assert result.balanced_decode(
        torch.empty((2, 0, result.target_width))
    ).shape == (2, 0, result.source_width)

    with pytest.raises(
        BalancedCircuitBasisError,
        match="source residual width",
    ):
        result.encode(torch.empty((0, result.source_width + 1)))
    with pytest.raises(
        BalancedCircuitBasisError,
        match="target residual width",
    ):
        result.decode(torch.empty((0, result.target_width + 1)))


def test_basis_is_deterministic_under_input_order_and_thread_count() -> None:
    fit, heldout = _snapshots()
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        first = build_balanced_circuit_basis(
            _structural(),
            fit,
            heldout,
            _observability(),
            1,
            circuit_graph=_graph(),
            gram_regularization=1e-12,
            require_all_ir_writers=True,
        )
        torch.set_num_threads(2)
        second = build_balanced_circuit_basis(
            tuple(reversed(_structural())),
            tuple(reversed(fit)),
            tuple(reversed(heldout)),
            tuple(reversed(_observability())),
            1,
            circuit_graph=_graph(),
            gram_regularization=1e-12,
            require_all_ir_writers=True,
        )
    finally:
        torch.set_num_threads(original_threads)

    assert torch.equal(first.basis, second.basis)
    assert torch.equal(first.balanced_encoder, second.balanced_encoder)
    assert torch.equal(first.balanced_decoder, second.balanced_decoder)
    assert first.map_sha256 == second.map_sha256
    assert first.report["hashes"] == second.report["hashes"]
    assert first.report["run_sha256"] == second.report["run_sha256"]
    assert first.report["determinism"] == {
        "cpu_float64": True,
        "internal_torch_threads": 1,
        "canonical_input_order": True,
        "canonical_balanced_subspace_projector": True,
        "canonical_coordinate_axis_order": True,
    }


def test_heldout_is_excluded_from_selection_but_changes_heldout_audit() -> None:
    first = _build(heldout_y_scale=1.0)
    second = _build(heldout_y_scale=3.0)

    assert torch.equal(first.basis, second.basis)
    assert (
        first.report["hashes"]["selection_sha256"]
        == second.report["hashes"]["selection_sha256"]
    )
    assert (
        first.report["hashes"]["heldout_evaluation_sha256"]
        != second.report["hashes"]["heldout_evaluation_sha256"]
    )
    assert first.report["run_sha256"] != second.report["run_sha256"]
    assert first.report["fit_heldout_isolation"] == {
        "fit_record_count": 2,
        "heldout_record_count": 1,
        "shared_record_count": 0,
        "record_ids_disjoint": True,
        "heldout_used_for_selection": False,
        "selection_sha256_excludes_heldout": True,
    }
    assert first.report["cached_controllability"]["phases"] == [
        "decode.000",
        "prefill",
    ]
    assert first.report["cached_controllability"]["block_count"] == 2


def test_dense_fold_protocol_and_zero_training_audit() -> None:
    result = _build()
    rows = torch.tensor(
        [[1.0, 2.0], [-3.0, 4.0]], dtype=torch.float64
    )
    encoded = result.encode(rows)

    assert torch.equal(encoded, result(rows))
    assert torch.equal(encoded, result.apply(rows))
    assert torch.allclose(
        result.decode(encoded),
        rows @ result.basis @ result.basis.T,
        atol=1e-14,
        rtol=0.0,
    )
    assert torch.allclose(
        result.balanced_encoder @ result.balanced_decoder,
        torch.eye(1, dtype=torch.float64),
        atol=1e-10,
        rtol=0.0,
    )
    consumer = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    producer = torch.tensor([[2.0], [3.0]], dtype=torch.float64)
    assert torch.equal(
        result.fold_consumer_weight(consumer),
        consumer @ result.basis,
    )
    assert torch.equal(
        result.fold_producer_weight(producer),
        result.basis.T @ producer,
    )
    assert torch.equal(
        result.materialize_dense(
            max_elements=2,
            dtype=torch.float32,
        ),
        result.basis.to(torch.float32),
    )
    with pytest.raises(
        BalancedCircuitBasisError,
        match="materialization bound",
    ):
        result.materialize_dense(max_elements=1)
    assert result.report["execution"] == {
        "precomputed_parent_cache_only": True,
        "model_forward_calls": 0,
        "parent_forward_calls": 0,
        "trace_capture_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "autograd_used": False,
        "filesystem_writes": 0,
    }

    source = Path(
        "src/neural_forge/dynamic_balanced_circuit_basis.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert "forward" not in called_attributes
    assert "backward" not in called_attributes
    assert "step" not in called_attributes
    assert "save" not in called_attributes


def test_nonfinite_overlap_missing_roles_and_rank_fail_closed() -> None:
    with pytest.raises(
        BalancedCircuitBasisEvidenceError,
        match="finite",
    ):
        ResidualSnapshotBlock(
            block_id="bad",
            split=EvidenceSplit.FIT,
            phase="prefill",
            layer="layer.000",
            endpoint="residual.pre",
            rows=torch.tensor([[float("nan"), 0.0]]),
            record_ids=("fit.bad",),
        )

    fit, heldout = _snapshots()
    overlapping = (
        ResidualSnapshotBlock(
            block_id="heldout.overlap",
            split=EvidenceSplit.HELDOUT,
            phase="decode.001",
            layer="layer.000",
            endpoint="residual.post_layer",
            rows=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            record_ids=("fit.prompt.0",),
        ),
    )
    with pytest.raises(
        BalancedCircuitBasisEvidenceError,
        match="share record IDs",
    ):
        build_balanced_circuit_basis(
            _structural(),
            fit,
            overlapping,
            _observability(),
            1,
        )

    with pytest.raises(
        BalancedCircuitBasisEvidenceError,
        match="Missing required observability",
    ):
        build_balanced_circuit_basis(
            _structural(),
            fit,
            heldout,
            _observability(),
            1,
            required_observability_kinds=(
                CircuitKind.MODALITY_OUTPUT,
            ),
        )

    with pytest.raises(
        BalancedCircuitBasisRankError,
        match="intersection ranks",
    ):
        build_balanced_circuit_basis(
            _structural(),
            fit,
            heldout,
            _observability(),
            2,
            gram_regularization=1e-12,
        )
