from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack  # noqa: E402
from neural_forge.analytic_execution_profile import (  # noqa: E402
    resolve_analytic_execution_profile,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_weight_complete_basis import (  # noqa: E402
    FeatureOrientation,
    ResidualFacingOperator,
    compile_role_balanced_operator_geometry,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (  # noqa: E402
    Lfm2CpcfTerminalActionDownWeightPrior,
    Lfm2CpcfTerminalActionEvidence,
)
from neural_forge.hf_lfm2_semantic_body_recipe import (  # noqa: E402
    lfm2_semantic_body_recipe_ids_for_component_lane,
)
from neural_forge.hf_lfm2_semantic_mlp_refold_adapter import (  # noqa: E402
    Lfm2DirectLocalMlpSourceInventory,
    Lfm2SemanticMlpEndpointEvidence,
)
from neural_forge.hf_lfm2_semantic_postfold import (  # noqa: E402
    Lfm2SemanticPostfoldError,
    authenticate_lfm2_semantic_layer_stack_attention_receipts,
)
from neural_forge.hf_tied_causal_interface_fold import (  # noqa: E402
    TiedCausalInterfaceFold,
)
from neural_forge.hf_tied_residual_congruence import (  # noqa: E402
    TIED_RESIDUAL_CONGRUENCE_SCHEMA,
    TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    compile_tied_residual_congruence_candidate,
)


_BASE_INTERFACE_SCHEMA = "neural-forge-tied-causal-interface-fold-v1"
_ANCHOR_INTERFACE_SCHEMA = "neural-forge-tied-causal-anchor-residual-v1"


@pytest.mark.parametrize(
    "operator_kinds",
    (
        ("short_conv", "full_attention"),
        (
            "short_conv",
            "full_attention",
            "short_conv",
            "full_attention",
            "short_conv",
        ),
        tuple(
            "full_attention" if index in {2, 4, 6, 8, 10, 12} else "short_conv"
            for index in range(14)
        ),
    ),
    ids=("depth-2", "depth-5", "depth-14"),
)
def test_operator_topology_is_derived_for_multiple_blank_child_depths(
    monkeypatch: pytest.MonkeyPatch,
    operator_kinds: tuple[str, ...],
) -> None:
    class Lineage:
        def __init__(self, layer_index: int) -> None:
            self.layer_index = layer_index

    monkeypatch.setattr(stack, "Lfm2SemanticCompilerLayerLineage", Lineage)
    topology = [
        {"contract": {"kind": operator_kind}} for operator_kind in operator_kinds
    ]
    target = SimpleNamespace(
        model=SimpleNamespace(
            layers=tuple(
                SimpleNamespace(is_attention_layer=kind == "full_attention")
                for kind in operator_kinds
            )
        ),
        config=SimpleNamespace(
            layer_types=tuple(
                "conv" if kind == "short_conv" else kind for kind in operator_kinds
            )
        ),
    )
    attention_indices = tuple(
        index for index, kind in enumerate(operator_kinds) if kind == "full_attention"
    )
    lineage = {index: Lineage(index) for index in attention_indices}

    assert (
        stack._validated_operator_topology(
            SimpleNamespace(manifest={"layer_topology": {"layers": topology}}),
            target,
            lineage,
        )
        == operator_kinds
    )


def test_normalized_operator_decoder_uses_exact_residual_chart_fit_batch_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_ids = (10, 20, 30, 40)
    alignments = tuple(
        SimpleNamespace(
            source_batch_index=batch_index,
            phases=(SimpleNamespace(target_to_donor_boundary_positions=(0,)),),
        )
        for batch_index in batch_ids
    )
    chart = torch.eye(2, dtype=torch.float64)
    partition_batch_indices = {
        "fit": [30],
        "fit_check": [10],
        "selection": [40],
        "validation": [20],
    }
    chart_body = {
        "schema": "neural-forge-lfm2-layer-stack-residual-chart-v1",
        "layer_index": 0,
        "chart_sha256": stack._tensor_sha256(chart),
        "partition_batch_indices": partition_batch_indices,
    }
    chart_receipt = MappingProxyType(
        {**chart_body, "result_sha256": canonical_sha256(chart_body)}
    )
    calls: list[tuple[int, tuple[float, ...]]] = []

    def fake_fit(
        *,
        parent_raw_rows: object,
        fit_trajectory_count: int,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, MappingProxyType]:
        rows = tuple(parent_raw_rows)
        values = tuple(float(value[0, 0].item()) for value in rows)
        calls.append((fit_trajectory_count, values))
        decoder = torch.cat(rows[:fit_trajectory_count], dim=0).mean(dim=0)
        body = {
            "schema": "test-normalized-decoder",
            "fit_trajectory_count": fit_trajectory_count,
        }
        return decoder, MappingProxyType(
            {**body, "decoder_sha256": canonical_sha256(body)}
        )

    monkeypatch.setattr(
        stack,
        "_fit_normalized_operator_input_decoder",
        fake_fit,
    )

    def run(fit_check_value: float) -> tuple[torch.Tensor, MappingProxyType]:
        values = {10: fit_check_value, 20: 2.0, 30: 3.0, 40: 4.0}
        parent = {
            batch_index: SimpleNamespace(
                final_histories={
                    "residual.pre": torch.tensor(
                        [[[values[batch_index], 0.0]]],
                        dtype=torch.float64,
                    )
                }
            )
            for batch_index in batch_ids
        }
        child = SimpleNamespace(
            replay_by_batch={
                batch_index: SimpleNamespace(
                    operator_input=torch.tensor(
                        [[[values[batch_index], 0.0]]],
                        dtype=torch.float64,
                    )
                )
                for batch_index in batch_ids
            }
        )
        return stack._fit_layer_normalized_operator_input_decoder(
            alignments=alignments,
            parent=parent,
            child_bank=child,
            parent_operator_norm_weight=torch.ones(2),
            target_operator_norm_weight=torch.ones(2),
            parent_norm_epsilon=1e-5,
            target_norm_epsilon=1e-5,
            chart=chart,
            chart_receipt=chart_receipt,
        )

    first, first_receipt = run(1.0)
    second, second_receipt = run(999.0)

    assert all(count == 1 and values[0] == 3.0 for count, values in calls)
    assert torch.equal(first, second)
    assert first_receipt["fit_batch_indices"] == [30]
    assert second_receipt["fit_batch_indices"] == [30]
    assert first_receipt["trajectory_batch_indices"] == [30, 10, 40, 20]
    assert first_receipt["heldout_batch_indices"] == [10, 40, 20]
    assert first_receipt["fit_batch_indices_authenticated_from_residual_chart"]


def test_body_global_fit_rows_are_fit_only_trajectory_balanced_and_deterministic() -> (
    None
):
    batch_ids = (10, 20, 30, 40)
    partition_batch_indices = {
        "fit": [30, 10],
        "fit_check": [20],
        "selection": [40],
        "validation": [],
    }
    chart_body = {
        "schema": "neural-forge-lfm2-layer-stack-residual-chart-v1",
        "layer_index": 0,
        "partition_batch_indices": partition_batch_indices,
    }
    chart_receipt = MappingProxyType(
        {**chart_body, "result_sha256": canonical_sha256(chart_body)}
    )
    lengths = {10: 1, 20: 2, 30: 4, 40: 3}

    def run(
        *,
        alignment_order: tuple[int, ...],
        heldout_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor, MappingProxyType]:
        alignments = tuple(
            SimpleNamespace(
                source_batch_index=batch_index,
                phases=(
                    SimpleNamespace(
                        target_to_donor_boundary_positions=tuple(
                            range(lengths[batch_index])
                        )
                    ),
                ),
                alignment_sha256=f"{batch_index:064x}",
            )
            for batch_index in alignment_order
        )
        parent = {
            batch_index: SimpleNamespace(
                final_histories={
                    "residual.pre": torch.full(
                        (1, lengths[batch_index], 3),
                        1.0
                        if batch_index in partition_batch_indices["fit"]
                        else heldout_value,
                        dtype=torch.float64,
                    )
                }
            )
            for batch_index in reversed(alignment_order)
        }
        child_bank = SimpleNamespace(
            pre_residual_by_batch={
                batch_index: torch.full(
                    (1, lengths[batch_index], 2),
                    2.0
                    if batch_index in partition_batch_indices["fit"]
                    else -heldout_value,
                    dtype=torch.float64,
                )
                for batch_index in alignment_order
            }
        )
        return stack._body_global_fit_residual_rows(
            alignments=alignments,
            parent=parent,
            child_bank=child_bank,
            chart_receipt=chart_receipt,
            layer_index=0,
        )

    first_child, first_parent, first_receipt = run(
        alignment_order=(40, 30, 20, 10),
        heldout_value=17.0,
    )
    second_child, second_parent, second_receipt = run(
        alignment_order=batch_ids,
        heldout_value=999_999.0,
    )

    torch.testing.assert_close(first_child, second_child, rtol=0, atol=0)
    torch.testing.assert_close(first_parent, second_parent, rtol=0, atol=0)
    assert first_receipt == second_receipt
    assert first_receipt["fit_batch_indices"] == [30, 10]
    assert [row["batch_index"] for row in first_receipt["row_manifest"]] == [30, 10]
    assert first_receipt["fit_row_count"] == 5
    assert first_receipt["fit_partition_only"] is True
    assert first_receipt["heldout_rows_used"] == 0
    assert first_receipt["trajectory_balanced_rows"] is True
    torch.testing.assert_close(
        first_child[:4].square().sum(),
        first_child[4:].square().sum(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first_parent[:4].square().sum(),
        first_parent[4:].square().sum(),
        rtol=0,
        atol=0,
    )


def test_real_normalized_operator_decoder_is_invariant_to_every_nonfit_split() -> None:
    batch_ids = (10, 20, 30, 40)
    fit_batch_index = 30
    alignments = tuple(
        SimpleNamespace(
            source_batch_index=batch_index,
            phases=(SimpleNamespace(target_to_donor_boundary_positions=(0,)),),
        )
        for batch_index in batch_ids
    )
    chart = torch.eye(2, dtype=torch.float64)
    partition_batch_indices = {
        "fit": [fit_batch_index],
        "fit_check": [10],
        "selection": [40],
        "validation": [20],
    }
    chart_body = {
        "schema": "neural-forge-lfm2-layer-stack-residual-chart-v1",
        "layer_index": 0,
        "chart_sha256": stack._tensor_sha256(chart),
        "partition_batch_indices": partition_batch_indices,
    }
    chart_receipt = MappingProxyType(
        {**chart_body, "result_sha256": canonical_sha256(chart_body)}
    )

    def run(
        overrides: dict[int, float] | None = None,
    ) -> tuple[torch.Tensor, MappingProxyType]:
        values = {10: 1.0, 20: 2.0, 30: 3.0, 40: 4.0}
        values.update({} if overrides is None else overrides)
        parent = {
            batch_index: SimpleNamespace(
                final_histories={
                    "residual.pre": torch.tensor(
                        [[[values[batch_index], values[batch_index] + 0.5]]],
                        dtype=torch.float64,
                    )
                }
            )
            for batch_index in batch_ids
        }
        child = SimpleNamespace(
            replay_by_batch={
                batch_index: SimpleNamespace(
                    operator_input=torch.tensor(
                        [[[0.25 * values[batch_index], 0.5]]],
                        dtype=torch.float64,
                    )
                )
                for batch_index in batch_ids
            }
        )
        return stack._fit_layer_normalized_operator_input_decoder(
            alignments=alignments,
            parent=parent,
            child_bank=child,
            parent_operator_norm_weight=torch.ones(2, dtype=torch.bfloat16),
            target_operator_norm_weight=torch.ones(2, dtype=torch.bfloat16),
            parent_norm_epsilon=1e-5,
            target_norm_epsilon=1e-5,
            chart=chart,
            chart_receipt=chart_receipt,
        )

    baseline, baseline_receipt = run()
    for nonfit_batch_index in (10, 40, 20):
        perturbed, receipt = run({nonfit_batch_index: 10_000.0})
        torch.testing.assert_close(perturbed, baseline, rtol=0, atol=0)
        assert receipt["fit_batch_indices"] == [fit_batch_index]
        assert receipt["heldout_rows_used_for_fit"] == 0
    assert baseline_receipt["fit_trajectory_count"] == 1


def _body_recipe_fields(recipe_id: str = "fit_guided") -> dict[str, object]:
    recipe = stack.resolve_lfm2_semantic_body_recipe(recipe_id)
    return {
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
    }


def _qdeim_compute_execution(
    *,
    schema: str,
    requested_device: str,
) -> dict[str, object]:
    cuda = requested_device != "cpu"
    body: dict[str, object] = {
        "schema": schema,
        "requested_device": requested_device,
        "resolved_device": requested_device,
        "backend": "cuda" if cuda else "cpu",
        "compute_dtype": "float64",
        "authenticated_source_archive_device": "cpu",
        "active_layer_staged_to_compute_device": cuda,
        "materialized_candidate_device": "cpu",
        "system_memory_is_immutable_archive": True,
        "gpu_is_ephemeral_dense_algebra_workspace": cuda,
        "silent_device_fallback_performed": False,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    if cuda:
        body.update(
            {
                "device_index": int(requested_device.split(":", 1)[1]),
                "device_name": "test CUDA device",
                "compute_capability": [6, 1],
            }
        )
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _mlp_functional_provenance_receipt(layer_index: int) -> dict[str, object]:
    policy_body = {
        "schema": "neural-forge-bf16-cross-kernel-equivalence-policy-v1",
        "diagnostic_only": True,
        "numeric_admission_authority": False,
        "fold_selection_or_quality_objective": False,
    }
    policy = {
        **policy_body,
        "policy_sha256": canonical_sha256(policy_body),
    }
    direct_names = (
        "ffn_norm.weight",
        "feed_forward.w1.weight",
        "feed_forward.w3.weight",
        "feed_forward.w2.weight",
    )
    endpoint_names = (
        "mlp.input",
        "mlp.gate",
        "mlp.up",
        "mlp.post_swiglu",
        "residual.post_layer",
    )
    parent_identity = {
        "schema": "neural-forge-lfm2-parent-fingerprint-v2",
        "model_class": "tests.Parent",
        "config_sha256": "0" * 64,
        "state_dict_sha256": "1" * 64,
        "state_tensor_count": 4,
        "runtime_buffer_sha256": "2" * 64,
        "runtime_buffer_count": 0,
    }
    cache_parent_fingerprint = canonical_sha256(parent_identity)
    metric = {
        "maximum_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
        "error_rms": 0.0,
        "cached_endpoint_rms": 1.0,
        "relative_rmse": 0.0,
        "cached_maximum_absolute_value": 1.0,
        "maximum_elementwise_tolerance_at_cached_maximum": 0.01,
        "replayed_all_finite": True,
        "nonfinite_replayed_element_count": 0,
        "within_legacy_bf16_envelope": True,
        "diagnostics_finite": True,
        "diagnostic_only": True,
        "numeric_admission_authority": False,
    }
    body = {
        "schema": "neural-forge-lfm2-direct-parent-mlp-functional-provenance-v3",
        "layer_index": layer_index,
        "cache_key": "4" * 64,
        "cache_parent_fingerprint": cache_parent_fingerprint,
        "cache_manifest_sha256": "6" * 64,
        "parent_checkpoint_manifest_sha256": "7" * 64,
        "parent_session_open_receipt_sha256": "8" * 64,
        "parent_config_file_sha256": "9" * 64,
        "parent_normalized_config_sha256": "0" * 64,
        "parent_identity_reconstruction_sha256": "f" * 64,
        "reconstructed_parent_identity": parent_identity,
        "parent_norm_epsilon": 1e-5,
        "execution_device": "cpu",
        "direct_tensor_sha256": {name: "1" * 64 for name in direct_names},
        "cached_endpoint_sha256": {name: "2" * 64 for name in endpoint_names},
        "replayed_endpoint_sha256": {name: "3" * 64 for name in endpoint_names},
        "reconstructed_normalized_mlp_input_sha256": "a" * 64,
        "cached_normalized_mlp_input_sha256": "b" * 64,
        "semantic_endpoint_row_ids_sha256": "c" * 64,
        "cross_kernel_equivalence_policy": policy,
        "endpoint_equivalence_metrics": {
            name: {"field": name, **metric} for name in endpoint_names
        },
        "outside_legacy_bf16_envelope": {},
        "semantic_endpoint_rows": 1,
        "all_cached_endpoints_bit_exact": True,
        "all_cached_endpoints_within_legacy_bf16_envelope": True,
        "all_cross_kernel_replay_diagnostics_finite": True,
        "cached_mlp_input_is_normalized_evidence_source_of_truth": True,
        "reconstructed_mlp_input_is_diagnostic_only": True,
        "direct_parent_tensors_within_legacy_numeric_envelope": True,
        "semantic_endpoint_scope_is_complete_tensor_identity_proof": False,
        "cryptographic_parent_tensor_identity_proven_by_this_receipt": True,
        "exact_parent_identity_is_source_admission_authority": True,
        "numeric_replay_has_source_admission_authority": False,
        "complete_state_dict_identity_reconstructed": True,
        "runtime_buffer_identity_reconstructed": True,
        "normalized_config_identity_reconstructed": True,
        "parent_checkpoint_files_hashed_by_session": True,
        "direct_tensor_digests_recorded": True,
        "complete_state_dict_digest_compared_to_cache": True,
        "numerical_equivalence_is_diagnostic_only": True,
        "numeric_parent_child_weight_match_objective": False,
        "parent_model_forward_calls": 0,
        "training_steps": 0,
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def _anchor_tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(
        json.dumps(
            descriptor,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class _TiedInterfaceTarget(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        vocabulary, width = weight.shape
        self.embedding = torch.nn.Embedding(
            vocabulary,
            width,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            self.embedding.weight.copy_(weight)
        self.output = torch.nn.Linear(
            width,
            vocabulary,
            bias=False,
            dtype=weight.dtype,
        )
        self.output.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output


def _interface_for_schema(
    schema: str,
) -> tuple[_TiedInterfaceTarget, TiedCausalInterfaceFold]:
    target_weight = torch.tensor(
        [[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]],
        dtype=torch.float64,
    )
    producer_encoder = torch.tensor(
        [[1.0, 0.5], [-0.25, 2.0], [3.0, -4.0], [0.75, 1.25]],
        dtype=torch.float64,
    )
    tensor_sha256 = (
        _anchor_tensor_sha256
        if schema == _ANCHOR_INTERFACE_SCHEMA
        else stack._interface_tensor_sha256
    )
    report_body = {
        "schema": schema,
        "producer_encoder_sha256": tensor_sha256(producer_encoder),
        "target_weight_sha256": tensor_sha256(target_weight),
    }
    interface = TiedCausalInterfaceFold(
        target_weight=target_weight,
        producer_encoder=producer_encoder,
        report={
            **report_body,
            "report_sha256": canonical_sha256(report_body),
        },
    )
    return _TiedInterfaceTarget(target_weight), interface


def test_validated_interface_lineage_accepts_anchor_active_tensor_seals() -> None:
    target, interface = _interface_for_schema(_ANCHOR_INTERFACE_SCHEMA)
    assert _anchor_tensor_sha256(
        interface.producer_encoder
    ) != stack._interface_tensor_sha256(interface.producer_encoder)

    receipt = stack._validated_interface_lineage(target, interface)

    assert receipt["interface_report_schema"] == _ANCHOR_INTERFACE_SCHEMA
    assert (
        receipt["interface_tensor_seal_schema"] == "canonical-descriptor-prefix-raw-v1"
    )
    assert (
        receipt["producer_encoder_sha256"]
        == interface.report["producer_encoder_sha256"]
    )


def test_validated_interface_lineage_preserves_base_interface_tensor_seals() -> None:
    target, interface = _interface_for_schema(_BASE_INTERFACE_SCHEMA)

    receipt = stack._validated_interface_lineage(target, interface)

    assert receipt["interface_report_schema"] == _BASE_INTERFACE_SCHEMA
    assert receipt["interface_tensor_seal_schema"] == "dtype-nul-shape-nul-raw-v1"
    assert receipt["target_weight_sha256"] == interface.report["target_weight_sha256"]


def test_congruence_lineage_is_accepted_and_forces_one_frozen_global_chart() -> None:
    generator = torch.Generator().manual_seed(721_318)
    source = torch.randn((5, 3), generator=generator, dtype=torch.float64)
    target = torch.randn((5, 3), generator=generator, dtype=torch.float64)
    encoder = torch.randn((3, 3), generator=generator, dtype=torch.float64)
    report_body = {
        "schema": _BASE_INTERFACE_SCHEMA,
        "producer_encoder_sha256": stack._interface_tensor_sha256(encoder),
        "target_weight_sha256": stack._interface_tensor_sha256(target),
    }
    raw = TiedCausalInterfaceFold(
        target_weight=target,
        producer_encoder=encoder,
        report={
            **report_body,
            "report_sha256": canonical_sha256(report_body),
        },
    )
    candidate = compile_tied_residual_congruence_candidate(
        raw,
        source,
        mode=TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    )
    target_model = _TiedInterfaceTarget(candidate.interface.target_weight)

    lineage = stack._validated_interface_lineage(
        target_model,
        candidate.interface,
    )
    alignments, parent, bank = _chart_inputs(reject_validation=False)
    chart, chart_receipt = stack._fit_residual_chart(
        alignments=alignments,
        parent=parent,
        child_bank=bank,
        producer_encoder=candidate.interface.producer_encoder,
        layer_index=0,
        global_gauge_required=lineage["global_residual_gauge_required"],
    )

    assert lineage["interface_report_schema"] == TIED_RESIDUAL_CONGRUENCE_SCHEMA
    assert lineage["tied_residual_congruence_authenticated"] is True
    assert lineage["global_residual_gauge_required"] is True
    torch.testing.assert_close(
        chart,
        candidate.interface.producer_encoder,
        rtol=0.0,
        atol=0.0,
    )
    assert chart_receipt["selected_candidate"] == "base_identity"
    assert chart_receipt["global_self_dual_gauge_required"] is True
    assert chart_receipt["independent_layer_rotation_allowed"] is False


@pytest.mark.parametrize(
    "schema",
    [_ANCHOR_INTERFACE_SCHEMA, _BASE_INTERFACE_SCHEMA],
)
def test_validated_interface_lineage_rejects_schema_specific_tensor_tampering(
    schema: str,
) -> None:
    target, interface = _interface_for_schema(schema)
    tampered_body = dict(interface.report)
    tampered_body.pop("report_sha256")
    tampered_body["target_weight_sha256"] = "0" * 64
    tampered = TiedCausalInterfaceFold(
        target_weight=interface.target_weight,
        producer_encoder=interface.producer_encoder,
        report={
            **tampered_body,
            "report_sha256": canonical_sha256(tampered_body),
        },
    )

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="lack exact authenticated child lineage",
    ):
        stack._validated_interface_lineage(target, tampered)


def _terminal_banks():
    return {
        split: Lfm2CpcfTerminalActionEvidence(
            split=split,
            row_ids=(f"{split}:row",),
            mlp_input=torch.ones((1, 2), dtype=torch.bfloat16),
            pre_mlp_residual=torch.zeros((1, 2), dtype=torch.bfloat16),
            parent_final_norm_output=torch.ones((1, 2), dtype=torch.bfloat16),
            sample_weights=torch.ones((1,), dtype=torch.float64),
        )
        for split in stack._PARTITIONS
    }


class _TinyTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mutable = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        self.final_w2 = torch.nn.Parameter(torch.tensor([7.0], dtype=torch.float64))
        self._neural_forge_quarantine_tokenizer = object()


class _ReceiptStateLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.operator_norm = torch.nn.Module()
        self.operator_norm.weight = torch.nn.Parameter(
            torch.ones((1,), dtype=torch.float64)
        )
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Module()
        self.self_attn.q_proj.weight = torch.nn.Parameter(
            torch.ones((1,), dtype=torch.float64)
        )
        self.conv = torch.nn.Module()
        self.conv.in_proj = torch.nn.Module()
        self.conv.in_proj.weight = torch.nn.Parameter(
            torch.ones((1,), dtype=torch.float64)
        )
        self.feed_forward = torch.nn.Module()
        self.feed_forward.w1 = torch.nn.Module()
        self.feed_forward.w1.weight = torch.nn.Parameter(
            torch.ones((1,), dtype=torch.float64)
        )


class _FourteenLayerReceiptStateTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [_ReceiptStateLayer() for _ in range(14)]
        )
        self.config = SimpleNamespace(norm_eps=1e-5)
        self.final_w2 = torch.nn.Parameter(torch.tensor([7.0], dtype=torch.float64))
        self._neural_forge_quarantine_tokenizer = object()


class _DenseCoverageLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(3, 3, bias=False)
        self.self_attn.k_proj = torch.nn.Linear(3, 2, bias=False)
        self.self_attn.v_proj = torch.nn.Linear(3, 2, bias=False)
        self.self_attn.out_proj = torch.nn.Linear(2, 3, bias=False)
        self.conv = torch.nn.Module()
        self.conv.in_proj = torch.nn.Linear(3, 6, bias=False)
        self.conv.conv = torch.nn.Conv1d(6, 6, 3, groups=6, bias=False)
        self.conv.out_proj = torch.nn.Linear(6, 3, bias=False)
        self.feed_forward = torch.nn.Module()
        self.feed_forward.w1 = torch.nn.Linear(3, 5, bias=False)
        self.feed_forward.w3 = torch.nn.Linear(3, 5, bias=False)
        self.feed_forward.w2 = torch.nn.Linear(5, 3, bias=False)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()


class _DenseCoverageTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_DenseCoverageLayer()])


def _state_identity(model: _TinyTarget) -> str:
    return canonical_sha256(
        {
            name: stack._tensor_sha256(value)
            for name, value in sorted(model.state_dict().items())
        }
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_live_layer_runtime_accepts_uniform_supported_target_dtype(dtype) -> None:
    layer = torch.nn.Sequential(
        torch.nn.Linear(3, 4, bias=False, dtype=dtype),
        torch.nn.Linear(4, 3, bias=False, dtype=dtype),
    )

    device, observed_dtype = stack._layer_runtime(layer)

    assert device.type == "cpu"
    assert observed_dtype == dtype


def test_live_layer_runtime_rejects_mixed_target_dtypes() -> None:
    layer = torch.nn.Module()
    layer.first = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    layer.second = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="one materialized BF16 or FP32 dtype",
    ):
        stack._layer_runtime(layer)


def test_initial_residual_bank_preserves_float32_target_coordinates() -> None:
    class _EmbeddingTarget(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 3, dtype=torch.float32)

        def get_input_embeddings(self):
            return self.embedding

    alignment = SimpleNamespace(
        source_batch_index=7,
        phases=(
            SimpleNamespace(
                canonical_target_input_ids=torch.tensor([[0, 2, 3]], dtype=torch.long)
            ),
        ),
    )

    bank = stack._initial_residual_bank(_EmbeddingTarget(), (alignment,))

    assert bank[7].dtype == torch.float32
    assert tuple(bank[7].shape) == (1, 3, 3)


def test_changed_state_keys_canonicalizes_fourteen_layer_model_order() -> None:
    names = [
        f"model.layers.{layer_index}.feed_forward.w1.weight"
        for layer_index in range(14)
    ]
    before = {name: "before" for name in names}
    after = {name: "after" for name in sorted(names)}

    changed = stack._changed_state_keys(before, after)

    assert changed == tuple(sorted(names))
    assert changed.index("model.layers.10.feed_forward.w1.weight") < changed.index(
        "model.layers.2.feed_forward.w1.weight"
    )


def test_dense_producer_coverage_authenticates_each_structural_axis_and_deferred_w2() -> (
    None
):
    target = _DenseCoverageTarget()
    layer = target.model.layers[0]

    blank_attention = stack._lfm2_dense_producer_chain_coverage(
        target,
        layer_index=0,
        producer_kind="full_attention",
    )
    assert blank_attention["all_required_roles_structurally_nonzero"] is False
    assert blank_attention["blank_or_partially_blank_required_role_count"] == 4

    with torch.no_grad():
        for projection in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.out_proj,
        ):
            projection.weight.fill_(1.0)
        layer.feed_forward.w1.weight.fill_(1.0)
        layer.feed_forward.w3.weight.fill_(1.0)

    planted_attention = stack._lfm2_dense_producer_chain_coverage(
        target,
        layer_index=0,
        producer_kind="full_attention",
    )
    assert planted_attention["all_required_roles_structurally_nonzero"] is True
    assert planted_attention["blank_or_partially_blank_required_role_count"] == 0

    deferred_mlp = stack._lfm2_dense_producer_chain_coverage(
        target,
        layer_index=0,
        producer_kind="mlp",
        defer_down_weight=True,
    )
    assert deferred_mlp["all_required_roles_structurally_nonzero"] is True
    assert deferred_mlp["deferred_roles"] == ["feed_forward.w2.weight"]

    incomplete_full_mlp = stack._lfm2_dense_producer_chain_coverage(
        target,
        layer_index=0,
        producer_kind="mlp",
    )
    assert incomplete_full_mlp["all_required_roles_structurally_nonzero"] is False
    assert incomplete_full_mlp["blank_or_partially_blank_required_role_count"] == 1

    with torch.no_grad():
        layer.feed_forward.w2.weight.fill_(1.0)
        layer.self_attn.q_proj.weight.zero_()
        layer.self_attn.q_proj.weight[0, 0] = 1.0
    partial_attention = stack._lfm2_dense_producer_chain_coverage(
        target,
        layer_index=0,
        producer_kind="full_attention",
    )
    q_receipt = partial_attention["role_receipts"]["self_attn.q_proj.weight"]
    assert q_receipt["nonzero_element_count"] == 1
    assert q_receipt["all_output_structural_units_covered"] is False
    assert q_receipt["all_input_structural_units_covered"] is False
    assert partial_attention["all_required_roles_structurally_nonzero"] is False

    full_mlp = stack._lfm2_dense_producer_chain_coverage(
        target,
        layer_index=0,
        producer_kind="mlp",
    )
    assert full_mlp["all_required_roles_structurally_nonzero"] is True


def _persistable_attention_operator_receipt(
    *,
    layer_index: int,
    before: str,
    after: str,
    installed: bool,
    parameter_count: int = 2,
    body_recipe: str = "fit_guided",
) -> dict[str, object]:
    recipe = stack.resolve_lfm2_semantic_body_recipe(body_recipe)
    roles = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
    )
    bundle = {
        role: canonical_sha256(["attention-role", layer_index, role, after])
        for role in roles
    }
    output_proposal_body = {
        "schema": "neural-forge-lfm2-semantic-output-writer-proposal-v1",
        "output_weight_sha256": bundle["out_proj.weight"],
        "changed": False,
        "value_weight_fixed": True,
        "fit_improved_is_diagnostic_only": True,
        "numeric_metric_used_for_step_size_selection": False,
        "numeric_metric_used_for_publication": False,
    }
    output_proposal = {
        **output_proposal_body,
        "proposal_sha256": canonical_sha256(output_proposal_body),
    }
    output_writer_body = {
        "accepted": False,
        "proposal": output_proposal,
        "heldout_metrics_are_diagnostic_only": True,
        "numeric_metric_used_for_publication": False,
    }
    output_writer = {
        **output_writer_body,
        "gate_sha256": canonical_sha256(output_writer_body),
    }
    joint_body = {
        "value_weight_sha256": bundle["v_proj.weight"],
        "output_weight_sha256": bundle["out_proj.weight"],
        "changed": False,
        "diagnostic_only": False,
        "publication_allowed": True,
        "numeric_metric_used_for_publication": False,
    }
    joint = {
        **joint_body,
        "proposal_sha256": canonical_sha256(joint_body),
    }
    metrics = {
        "endpoint_relative_rmse": 1.0,
        "endpoint_rmse": 1.0,
        "endpoint_maximum_absolute_error": 1.0,
        "endpoint_cosine_similarity_mean": 0.0,
        "endpoint_cosine_similarity_worst": 0.0,
    }
    proposal_sha256 = {
        "incumbent": canonical_sha256(["attention-incumbent", layer_index]),
        "fixed_v_output_only": output_proposal["proposal_sha256"],
        "joint_value_output": joint["proposal_sha256"],
    }
    candidate_reports = {}
    for candidate in ("incumbent", "fixed_v_output_only", "joint_value_output"):
        partitions = {}
        for partition in ("fit", "fit_check", "selection", "validation"):
            partitions[partition] = {
                "candidate": dict(metrics),
                "diagnostic_only": partition != "fit",
                "used_for_candidate_ordering": False,
                "used_for_structural_eligibility": False,
                "used_for_publication_threshold": False,
                "used_for_publication": False,
            }
        candidate_reports[candidate] = {
            "proposal_changed": False,
            "proposal_sha256": proposal_sha256[candidate],
            "finite": True,
            "dense_strided": True,
            "bfloat16": True,
            "exact_shape": True,
            "effectful": False,
            "eligible": False,
            "eligibility_basis": (
                "finite_dense_bfloat16_exact_shape_and_nonincumbent_effect"
            ),
            "numeric_metric_used_for_eligibility": False,
            "eligible_after_fit_structural_check": False,
            "eligible_after_fit_fit_check_selection": False,
            "structurally_publishable": False,
            "partitions": partitions,
        }
    value_output_gate_body = {
        **_body_recipe_fields(recipe.recipe_id),
        "recipe_component_lane": recipe.component_lanes["value_output"],
        "schema": "neural-forge-lfm2-semantic-vo-candidate-gate-v1",
        "candidate_reports": candidate_reports,
        "selected_candidate": "incumbent",
        "published_candidate": "incumbent",
        "structurally_eligible_candidate_count": 0,
        "candidate_eligibility_basis": (
            "finite_dense_bfloat16_exact_shape_and_nonincumbent_effect"
        ),
        "structural_safety_is_only_candidate_veto": True,
        "fit_metrics_used_for_candidate_ordering": (
            recipe.fit_metrics_order_candidates
        ),
        "component_selection_policy_equivalent_recipe_ids": list(
            lfm2_semantic_body_recipe_ids_for_component_lane(
                "value_output",
                recipe.component_lanes["value_output"],
            )
        ),
        "component_selection_policy_equivalent_to_other_recipe": (
            len(
                lfm2_semantic_body_recipe_ids_for_component_lane(
                    "value_output",
                    recipe.component_lanes["value_output"],
                )
            )
            > 1
        ),
        "fit_ordering_metrics": [
            "endpoint_relative_rmse",
            "endpoint_rmse",
            "endpoint_maximum_absolute_error",
            "negative_endpoint_cosine_similarity_mean",
            "negative_endpoint_cosine_similarity_worst",
        ],
        "exact_fit_tie_breaker": "proposal_sha256",
        "fit_metrics_used_for_structural_eligibility": False,
        "fit_metric_threshold_used_for_candidate_selection": False,
        "incumbent_noop_fallback_used": True,
        "incumbent_noop_fallback_reason": (
            "all-analytic-proposals-structurally-invalid-or-identical"
        ),
        "selected_candidate_frozen_before_validation": True,
        "selected_candidate_frozen_before_heldout_diagnostics": True,
        "frozen_candidate_sha256": canonical_sha256(
            {
                "candidate_name": "incumbent",
                "value_weight_sha256": bundle["v_proj.weight"],
                "output_weight_sha256": bundle["out_proj.weight"],
                "proposal_sha256": proposal_sha256["incumbent"],
            }
        ),
        "validation_triggered_refit_or_reselection": False,
        "validation": {
            "diagnostic_only": True,
            "used_for_publication": False,
        },
        "selected_correction_rejected_on_validation": False,
        "validation_publication_veto_applied": False,
        "rollback_authorized": False,
        "alternate_candidate_selected_after_validation": False,
        "published_value_weight_sha256": bundle["v_proj.weight"],
        "published_output_weight_sha256": bundle["out_proj.weight"],
        "fit_only_analytic_solves": True,
        "heldout_metrics_are_diagnostic_only": True,
        "numeric_metric_used_for_candidate_selection": False,
        "numeric_metric_used_for_publication_or_rollback": False,
        "joint_value_output_publication_allowed": True,
        "fixed_value_weight_preserved": True,
        "operation_audit": {
            **dict(stack._ZERO_OPERATIONS),
            "model_forward_calls": 0,
        },
    }
    value_output_gate = {
        **value_output_gate_body,
        "gate_sha256": canonical_sha256(value_output_gate_body),
    }
    changed = (
        [f"model.layers.{layer_index}.self_attn.q_proj.weight"] if installed else []
    )
    analytic = {
        **_body_recipe_fields(recipe.recipe_id),
        "recipe_qk_lane": recipe.component_lanes["qk"],
        "recipe_value_output_lane": recipe.component_lanes["value_output"],
        "schema": "neural-forge-lfm2-semantic-attention-refold-v1",
        "cache_key": "a" * 64,
        "bridge_artifact_sha256": "b" * 64,
        "frozen_attention_six_tensor_bundle": bundle,
        "frozen_attention_six_tensor_bundle_roles": list(roles),
        "frozen_attention_six_tensor_bundle_sha256": canonical_sha256(bundle),
        "frozen_qk_sha256": canonical_sha256(
            {"q": bundle["q_proj.weight"], "k": bundle["k_proj.weight"]}
        ),
        "frozen_qkvo_sha256": canonical_sha256(
            {
                "q": bundle["q_proj.weight"],
                "k": bundle["k_proj.weight"],
                "v": bundle["v_proj.weight"],
                "o": bundle["out_proj.weight"],
            }
        ),
        "qk_compiler_accepted": installed,
        "qk_compiler_changed": installed,
        "fixed_value_source": "test-authenticated-fixed-value",
        "output_writer_source": "test-authenticated-output-writer",
        "output_writer": output_writer,
        "joint_value_output": joint,
        "value_output_candidate_gate": value_output_gate,
        "heldout_metrics_are_diagnostic_only": True,
        "numeric_metric_used_for_publication_or_rollback": False,
        "joint_value_output_candidate_diagnostic_only": False,
        "joint_value_output_publication_allowed": True,
        "joint_value_output_candidate_published": False,
        "complete_qkvo_bundle_installed_transactionally": installed,
        "complete_qkvo_bundle_transaction_verified": True,
        "authenticated_incumbent_qkvo_retained_without_installation": not installed,
        "projection_parameter_copy_calls": len(changed),
        "analytically_refit_projection_roles": (["q_proj"] if installed else []),
        "fixed_inherited_projection_roles": (
            ["k_proj", "v_proj", "out_proj"]
            if installed
            else ["q_proj", "k_proj", "v_proj", "out_proj"]
        ),
        "fresh_output_writer_refit_performed": False,
        "all_semantically_exact_endpoint_rows_used_for_attention_fit": True,
        "terminal_endpoint_only_supervision": False,
        "analytic_fp64_target_construction_distinct_from_bfloat16_runtime_replay": (
            True
        ),
        "runtime_replay_numeric_contract": (
            "bfloat16-linears-fp32-qk-softmax-bfloat16-probability-context"
        ),
        "cross_head_dimension_bridge_deferred": True,
        "refold_operation_audit": {
            **dict(stack._ZERO_OPERATIONS),
            "model_forward_calls": 0,
        },
    }
    topology = canonical_sha256(["test-attention-topology", layer_index])
    body = {
        **_body_recipe_fields(recipe.recipe_id),
        "recipe_component_lane": recipe.component_lanes["qk"],
        "schema": stack.LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA,
        "status": (
            "installed-frozen-semantic-attention-candidate"
            if installed
            else stack.LFM2_SEMANTIC_ATTENTION_STRUCTURAL_NOOP_STATUS
        ),
        "layer_index": layer_index,
        "installed": installed,
        "install_count": int(installed),
        "model_state_before_sha256": before,
        "model_state_after_sha256": after,
        "transport_evidence_sha256": [canonical_sha256(["attention", layer_index])],
        "changed_state_keys": changed,
        "parameter_element_count_before": parameter_count,
        "parameter_element_count_after": parameter_count,
        "parameter_count_changed": False,
        "target_topology_sha256_before": topology,
        "target_topology_sha256_after": topology,
        "topology_changed": False,
        "parameter_identities_preserved": True,
        "parameter_storage_preserved": True,
        "gradient_state_preserved": True,
        "exact_noop_state_proven": not installed,
        "selected_candidate": (
            "frozen-semantic-attention-candidate" if installed else "incumbent-noop"
        ),
        "selected_candidate_state_identical_to_incumbent": not installed,
        "incumbent_noop_fallback_used": not installed,
        "incumbent_noop_fallback_reason": (
            None
            if installed
            else "all-analytic-proposals-structurally-invalid-or-identical"
        ),
        "structurally_eligible_candidate_count": int(installed),
        "candidate_eligibility_basis": (
            "finite_dense_bfloat16_exact_shape_and_nonincumbent_effect"
        ),
        "structural_safety_is_only_candidate_veto": True,
        "fit_metrics_used_for_candidate_ordering": True,
        "fit_metrics_used_for_structural_eligibility": False,
        "incumbent_selected_by_heldout_gates": False,
        "heldout_metrics_are_diagnostic_only": True,
        "validation_publication_veto_applied": False,
        "rollback_authorized": False,
        "numeric_metrics_have_final_intelligence_authority": False,
        "validation_rejected_candidate": None,
        "numeric_metric_used_for_publication_or_rollback": False,
        "candidate_frozen_before_validation": True,
        "validation_triggered_refit": False,
        "analytic_cached_evidence_only": True,
        "operation_audit": dict(stack._ZERO_OPERATIONS),
        "analytic_refold": analytic,
    }
    return {**body, "result_sha256": canonical_sha256(body)}


@pytest.mark.parametrize(
    "body_recipe",
    (
        "fit_guided",
        "full_strength_structural",
        "residual_writer",
        "source_producer_residual_writer",
        "structural_sequence_residual_mlp_writer",
    ),
)
def test_attention_binding_authenticates_complete_recipe_component_equivalence(
    body_recipe: str,
) -> None:
    recipe = stack.resolve_lfm2_semantic_body_recipe(body_recipe)
    receipt = _persistable_attention_operator_receipt(
        layer_index=2,
        before="1" * 64,
        after="1" * 64,
        installed=False,
        body_recipe=body_recipe,
    )

    stack._require_attention_body_recipe_binding(
        receipt,
        recipe,
        label="test attention receipt",
    )

    gate = receipt["analytic_refold"]["value_output_candidate_gate"]
    equivalent = list(
        lfm2_semantic_body_recipe_ids_for_component_lane(
            "value_output",
            recipe.component_lanes["value_output"],
        )
    )
    assert gate["component_selection_policy_equivalent_recipe_ids"] == equivalent
    assert gate["component_selection_policy_equivalent_to_other_recipe"] is (
        len(equivalent) > 1
    )


def test_attention_binding_accepts_legacy_residual_writer_equivalence_receipt() -> None:
    recipe = stack.resolve_lfm2_semantic_body_recipe("residual_writer")
    receipt = _persistable_attention_operator_receipt(
        layer_index=2,
        before="1" * 64,
        after="1" * 64,
        installed=False,
        body_recipe=recipe.recipe_id,
    )
    gate = receipt["analytic_refold"]["value_output_candidate_gate"]
    gate.pop("gate_sha256")
    gate["component_selection_policy_equivalent_recipe_ids"] = ["residual_writer"]
    gate["component_selection_policy_equivalent_to_other_recipe"] = False
    gate["gate_sha256"] = canonical_sha256(gate)

    stack._require_attention_body_recipe_binding(
        receipt,
        recipe,
        label="legacy residual-writer attention receipt",
    )


def _patch_complete_transaction(
    monkeypatch,
    *,
    failing_layer: int | None = None,
    noop_operator_layers: frozenset[int] = frozenset(),
    noop_mlp_layers: frozenset[int] = frozenset(),
    blank_operator_layers: frozenset[int] = frozenset(),
    blank_mlp_layers: frozenset[int] = frozenset(),
    incomplete_operator_after_layers: frozenset[int] = frozenset(),
    incomplete_mlp_after_layers: frozenset[int] = frozenset(),
    use_live_fourteen_layer_state_chain: bool = False,
):
    events: list[str] = []

    class Cache:
        cache_key = "a" * 64
        parent_fingerprint = "b" * 64
        layer_topology_fingerprint = "c" * 64
        batch_count = 4
        manifest = {"manifest_sha256": "d" * 64}

        def selective_authenticated_session(self):
            class Session:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                def verify(self):
                    return self

                def as_dict(self):
                    return {"session_sha256": "e" * 64}

            return Session()

    class Bridge:
        artifact_sha256 = "f" * 64

        def verify_live_tokenizers(self, _donor, _target):
            return self

    class ParentSession:
        manifest_sha256 = "1" * 64
        open_receipt_sha256 = "2" * 64

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def tensor(self, _key):
            return torch.ones((2, 1, 2), dtype=torch.bfloat16)

        def optional_tensor(self, _key):
            return None

        def receipt(self):
            return {"session_sha256": "3" * 64}

    monkeypatch.setattr(stack, "Lfm2CircuitTraceCache", Cache)
    monkeypatch.setattr(stack, "StrictTokenizerBridgeArtifact", Bridge)
    monkeypatch.setattr(
        stack,
        "fingerprint_hf_tokenizer_semantics",
        lambda _value: "4" * 64,
    )
    monkeypatch.setattr(
        stack,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_state_identity(model)),
    )
    monkeypatch.setattr(
        stack,
        "_validated_operator_topology",
        lambda *_args: tuple(
            "full_attention"
            if index in stack._EXPECTED_ATTENTION_INDICES
            else "short_conv"
            for index in range(14)
        ),
    )
    monkeypatch.setattr(
        stack,
        "_validated_interface_lineage",
        lambda *_args, **_kwargs: {
            "interface_report_sha256": "5" * 64,
            "producer_encoder_sha256": "6" * 64,
        },
    )
    monkeypatch.setattr(
        stack,
        "_resolve_short_conv_adapter",
        lambda: (object, lambda **_kwargs: None),
    )
    monkeypatch.setattr(
        stack,
        "_resolve_mlp_adapter",
        lambda: (object, object, lambda **_kwargs: None),
    )
    monkeypatch.setattr(stack, "_final_w2_parameter", lambda model: model.final_w2)
    synthetic_changed_state_keys = []
    for layer_index in range(14):
        if layer_index not in noop_operator_layers:
            synthetic_changed_state_keys.append(
                (
                    f"model.layers.{layer_index}.self_attn.q_proj.weight"
                    if layer_index in stack._EXPECTED_ATTENTION_INDICES
                    else f"model.layers.{layer_index}.conv.in_proj.weight"
                )
            )
        if layer_index < 13 and layer_index not in noop_mlp_layers:
            synthetic_changed_state_keys.append(
                f"model.layers.{layer_index}.feed_forward.w1.weight"
            )
    if not use_live_fourteen_layer_state_chain:
        monkeypatch.setattr(
            stack,
            "_changed_state_keys",
            lambda _before, _after: tuple(sorted(synthetic_changed_state_keys)),
        )
    coverage_calls: dict[tuple[int, str], int] = {}

    def producer_coverage(
        model,
        *,
        layer_index,
        producer_kind,
        defer_down_weight=False,
    ):
        key = (layer_index, producer_kind)
        ordinal = coverage_calls.get(key, 0)
        coverage_calls[key] = ordinal + 1
        is_before = ordinal % 2 == 0
        blank_at_entry = (
            layer_index in blank_mlp_layers
            if producer_kind == "mlp"
            else layer_index in blank_operator_layers
        )
        incomplete_after = (
            layer_index in incomplete_mlp_after_layers
            if producer_kind == "mlp"
            else layer_index in incomplete_operator_after_layers
        )
        complete = not (
            (is_before and blank_at_entry) or (not is_before and incomplete_after)
        )
        state_sha = canonical_sha256(
            ["coverage-state", layer_index, producer_kind, _state_identity(model)]
        )
        body = {
            "schema": "neural-forge-lfm2-dense-producer-chain-coverage-v1",
            "layer_index": layer_index,
            "producer_kind": producer_kind,
            "deferred_roles": (["feed_forward.w2.weight"] if defer_down_weight else []),
            "required_role_state_sha256": state_sha,
            "model_state_tensor_hashes_sha256": _state_identity(model),
            "all_required_roles_structurally_nonzero": complete,
            "blank_or_partially_blank_required_role_count": int(not complete),
            "dense_existing_topology_only": True,
            "operation_audit": dict(stack._ZERO_OPERATIONS),
        }
        return stack._sealed(body)

    monkeypatch.setattr(
        stack,
        "_lfm2_dense_producer_chain_coverage",
        producer_coverage,
    )
    monkeypatch.setattr(
        stack,
        "_target_rope",
        lambda _model: (torch.ones((1,), dtype=torch.float64), 1.0),
    )
    monkeypatch.setattr(stack, "_DirectParentSafetensorsSession", ParentSession)
    alignments = tuple(
        SimpleNamespace(source_batch_index=index, alignment_sha256=str(index) * 64)
        for index in range(4)
    )
    monkeypatch.setattr(
        stack,
        "_semantic_alignments_from_session",
        lambda **_kwargs: (
            alignments,
            {"receipt_sha256": "8" * 64},
        ),
    )
    monkeypatch.setattr(
        stack,
        "audit_lfm2_semantic_span_cpcf_refinement",
        lambda *_args, **_kwargs: {
            "probability_preserving_attention_transport_go": True,
            "blocking_reasons": [],
            "report_sha256": "7" * 64,
        },
    )
    monkeypatch.setattr(
        stack,
        "_validate_direct_attention_sources",
        lambda *_args: {
            index: {"result_sha256": canonical_sha256(["qkvo", index])}
            for index in stack._EXPECTED_ATTENTION_INDICES
        },
    )
    monkeypatch.setattr(
        stack,
        "_initial_residual_bank",
        lambda *_args: MappingProxyType({0: "initial"}),
    )
    replay_count = {index: 0 for index in range(14)}

    def replay(model, *, layer_index, **_kwargs):
        ordinal = replay_count[layer_index]
        replay_count[layer_index] += 1
        events.append(f"{layer_index}:replay:{ordinal}")
        return SimpleNamespace(
            captured_child_model_state_sha256=_state_identity(model),
            layer_index=layer_index,
            bank_sha256=canonical_sha256(["bank", layer_index, ordinal]),
        )

    monkeypatch.setattr(stack, "_replay_current_layer", replay)

    def load_parent(_session, *, layer_index, **_kwargs):
        events.append(f"{layer_index}:parent")
        return {
            0: SimpleNamespace(
                evidence_sha256=canonical_sha256(["parent", layer_index])
            )
        }

    monkeypatch.setattr(stack, "_load_parent_layer_trajectories", load_parent)

    def chart(*, layer_index, **_kwargs):
        events.append(f"{layer_index}:chart")
        body = {"layer_index": layer_index, "fit_only": True}
        return torch.eye(2, dtype=torch.float64), {
            **body,
            "result_sha256": canonical_sha256(body),
        }

    monkeypatch.setattr(stack, "_fit_residual_chart", chart)

    def evidence(label: str, layer_index: int):
        return (
            SimpleNamespace(evidence_sha256=canonical_sha256([label, layer_index])),
        )

    def build_attention(*, layer_index, **_kwargs):
        events.append(f"{layer_index}:build:full_attention")
        return evidence("attention", layer_index)

    def build_short(*, layer_index, **_kwargs):
        events.append(f"{layer_index}:build:short_conv")
        return evidence("short", layer_index)

    monkeypatch.setattr(stack, "_build_attention_evidence", build_attention)
    monkeypatch.setattr(stack, "_build_short_conv_evidence", build_short)

    def mutate_operator(
        model,
        layer_index: int,
        kind: str,
        *,
        fold_compute_device: str = "cpu",
        body_recipe: object = "fit_guided",
    ):
        recipe = stack.resolve_lfm2_semantic_body_recipe(body_recipe)
        before = _state_identity(model)
        events.append(f"{layer_index}:install:{kind}")
        installed = layer_index not in noop_operator_layers
        if installed:
            with torch.no_grad():
                if use_live_fourteen_layer_state_chain:
                    layer = model.model.layers[layer_index]
                    parameter = (
                        layer.self_attn.q_proj.weight
                        if kind == "full_attention"
                        else layer.conv.in_proj.weight
                    )
                    parameter.add_(1.0)
                else:
                    model.mutable.add_(1.0)
        if failing_layer == layer_index:
            raise RuntimeError("injected partial operator failure")
        after = _state_identity(model)
        receipt = {
            "status": (
                "installed-analytic-candidate"
                if installed
                else (
                    stack._SHORT_CONV_STRUCTURAL_NOOP_STATUS
                    if kind == "short_conv"
                    else "heldout-selected-incumbent-noop"
                )
            ),
            "installed": installed,
            "install_count": int(installed),
            "model_state_before_sha256": before,
            "model_state_after_sha256": after,
            "result_sha256": canonical_sha256(["operator", layer_index, after]),
        }
        if kind == "full_attention":
            receipt = _persistable_attention_operator_receipt(
                layer_index=layer_index,
                before=before,
                after=after,
                installed=installed,
                parameter_count=stack._parameter_count(model),
                body_recipe=recipe.recipe_id,
            )
        else:
            receipt_body = {
                **{
                    key: value
                    for key, value in receipt.items()
                    if key != "result_sha256"
                },
                **_body_recipe_fields(recipe.recipe_id),
                "recipe_component_lane": recipe.component_lanes["short_conv"],
            }
            receipt = {
                **receipt_body,
                "result_sha256": canonical_sha256(receipt_body),
            }
        if kind == "short_conv" and fold_compute_device != "cpu":
            receipt.update(
                {
                    "fold_compute_device": fold_compute_device,
                    "requested_cuda_short_conv_routes_proved": True,
                    "structural_cluster_compute_execution": {
                        "backend": "cuda",
                        "requested_device": fold_compute_device,
                        "compute_dtype": "float64",
                        "execution_sha256": "a" * 64,
                    },
                    "writer_compute_execution": {
                        "backend": "cuda",
                        "requested_device": fold_compute_device,
                        "compute_dtype": "float64",
                        "execution_sha256": "b" * 64,
                    },
                }
            )
        resealed_body = {
            key: value for key, value in receipt.items() if key != "result_sha256"
        }
        return {
            **resealed_body,
            "result_sha256": canonical_sha256(resealed_body),
        }

    monkeypatch.setattr(
        stack,
        "_install_attention_operator",
        lambda target_model, layer_index, **_kwargs: mutate_operator(
            target_model,
            layer_index,
            "full_attention",
            body_recipe=_kwargs.get("body_recipe", "fit_guided"),
        ),
    )
    monkeypatch.setattr(
        stack,
        "_install_short_conv_operator",
        lambda target_model, layer_index, **_kwargs: mutate_operator(
            target_model,
            layer_index,
            "short_conv",
            fold_compute_device=_kwargs.get("fold_compute_device", "cpu"),
            body_recipe=_kwargs.get("body_recipe", "fit_guided"),
        ),
    )

    def validate_fir(*, layer_index, **_kwargs):
        events.append(f"{layer_index}:fir")
        body = {"layer_index": layer_index}
        return {**body, "result_sha256": canonical_sha256(body)}

    monkeypatch.setattr(stack, "_validate_cached_fir", validate_fir)

    def inventory(*, layer_index, **_kwargs):
        events.append(f"{layer_index}:inventory")
        return object(), _mlp_functional_provenance_receipt(layer_index)

    monkeypatch.setattr(stack, "_direct_mlp_inventory", inventory)

    def build_mlp(*, layer_index, **_kwargs):
        events.append(f"{layer_index}:build:mlp")
        return evidence("mlp", layer_index)

    monkeypatch.setattr(stack, "_build_mlp_evidence", build_mlp)

    def install_mlp(*, target_model, layer_index, **_kwargs):
        recipe = stack.resolve_lfm2_semantic_body_recipe(
            _kwargs.get("body_recipe", "fit_guided")
        )
        before = _state_identity(target_model)
        events.append(f"{layer_index}:install:mlp")
        installed = layer_index < 13 and layer_index not in noop_mlp_layers
        if installed:
            with torch.no_grad():
                if use_live_fourteen_layer_state_chain:
                    target_model.model.layers[layer_index].feed_forward.w1.weight.add_(
                        1.0
                    )
                else:
                    target_model.mutable.add_(1.0)
        after = _state_identity(target_model)
        fold_compute_device = _kwargs.get("fold_compute_device", "cpu")
        source_derived_planting = {
            "fold_compute_device": fold_compute_device,
        }
        if recipe.component_lanes["mlp"] in stack._GLOBAL_RESIDUAL_MLP_LANES:
            source_derived_planting.update(
                {
                    "normalized_decoder_control_compute_device": "cpu",
                    "qdeim_pod_and_pivot_compute_device": fold_compute_device,
                    "qdeim_compute_execution": _qdeim_compute_execution(
                        schema="neural-forge-lfm2-mlp-qdeim-compute-v1",
                        requested_device=fold_compute_device,
                    ),
                }
            )
        else:
            source_derived_planting["weight_complete_signature_compute_execution"] = (
                None
                if fold_compute_device == "cpu"
                else {
                    "backend": "cuda",
                    "requested_device": fold_compute_device,
                    "compute_dtype": "float64",
                    "execution_sha256": "9" * 64,
                }
            )
        receipt_body = {
            **_body_recipe_fields(recipe.recipe_id),
            "recipe_component_lane": recipe.component_lanes["mlp"],
            "status": (
                "installed-frozen-semantic-mlp-candidate"
                if installed
                else stack._MLP_STRUCTURAL_NOOP_STATUS
            ),
            "installed": installed,
            "install_count": int(installed),
            "model_state_before_sha256": before,
            "model_state_after_sha256": after,
            "source_derived_planting": source_derived_planting,
        }
        receipt = {
            **receipt_body,
            "result_sha256": canonical_sha256(receipt_body),
        }
        return receipt, (_terminal_banks() if layer_index == 13 else {}), None

    monkeypatch.setattr(stack, "_install_mlp", install_mlp)

    def next_bank(bank):
        events.append(f"{bank.layer_index}:next")
        return MappingProxyType({0: f"after:{bank.layer_index}"})

    monkeypatch.setattr(stack, "_next_residual_bank", next_bank)
    target = (
        _FourteenLayerReceiptStateTarget()
        if use_live_fourteen_layer_state_chain
        else _TinyTarget()
    )
    return target, Cache(), Bridge(), events


def test_complete_transaction_executes_exact_causal_order(monkeypatch) -> None:
    target, cache, bridge, events = _patch_complete_transaction(monkeypatch)
    before_w2 = target.final_w2.detach().clone()

    result = stack.compile_install_lfm2_semantic_layer_stack_postfold(
        cache=cache,
        bridge=bridge,
        target_model=target,
        interface=SimpleNamespace(producer_encoder=torch.eye(2)),
        compiler_lineage={
            index: object() for index in stack._EXPECTED_ATTENTION_INDICES
        },
        parent_model_directory="unused",
        donor_tokenizer=object(),
        target_tokenizer=object(),
        attention_installer=lambda **_kwargs: None,
    )

    assert result.receipt["complete_causal_layer_stack_adjudicated"] is True
    assert result.receipt["complete_causal_layer_stack_installed"] is False
    assert result.receipt["attention_operator_install_count"] == 6
    assert result.receipt["short_conv_operator_install_count"] == 8
    assert result.receipt["replay_fresh_mlp_evidence_compile_count"] == 14
    assert result.receipt["final_layer_mlp_w2_install_count"] == 0
    assert result.receipt["complete_nonzero_structural_producer_coverage"] is True
    assert result.receipt["blank_or_partially_blank_producer_chain_count"] == 0
    assert result.receipt["installed_complete_producer_chain_count"] == 27
    assert result.receipt["tokenizer_session_receipt_sha256"] == "8" * 64
    assert result.receipt["tokenizer_session"] == {"receipt_sha256": "8" * 64}
    assert result.receipt["authenticated_nonzero_folded_seed_retained_noop_count"] == 1
    assert result.receipt["final_mlp_producer_incumbent_noop_count"] == 1
    assert result.receipt["heldout_selected_incumbent_noop_count"] == 1
    assert (
        result.receipt["layer_installations"][13]["mlp_incumbent_noop_selected"] is True
    )
    attention_authentication = (
        authenticate_lfm2_semantic_layer_stack_attention_receipts(result.receipt)
    )
    assert attention_authentication["attention_receipt_count"] == 6
    assert attention_authentication["complete_six_tensor_bundle_count"] == 6
    assert attention_authentication["attention_install_count"] == 6
    for layer in result.receipt["layer_installations"]:
        if layer["operator_kind"] != "full_attention":
            assert "attention_operator_decision_receipt" not in layer
            assert "attention_operator_installation_receipt" not in layer
            continue
        assert (
            layer["attention_operator_decision_receipt"]
            == layer["attention_operator_installation_receipt"]
        )
        analytic = layer["attention_operator_decision_receipt"]["analytic_refold"]
        assert len(analytic["frozen_attention_six_tensor_bundle"]) == 6
        assert analytic["joint_value_output_publication_allowed"] is True
        assert analytic["joint_value_output_candidate_diagnostic_only"] is False
    torch.testing.assert_close(target.final_w2, before_w2, rtol=0, atol=0)
    for layer_index in range(14):
        kind = (
            "full_attention"
            if layer_index in stack._EXPECTED_ATTENTION_INDICES
            else "short_conv"
        )
        observed = [value for value in events if value.startswith(f"{layer_index}:")]
        expected = [
            f"{layer_index}:replay:0",
            f"{layer_index}:parent",
            f"{layer_index}:chart",
            *([] if kind == "full_attention" else [f"{layer_index}:fir"]),
            f"{layer_index}:build:{kind}",
            f"{layer_index}:install:{kind}",
            f"{layer_index}:replay:1",
            f"{layer_index}:inventory",
            f"{layer_index}:build:mlp",
            f"{layer_index}:install:mlp",
            f"{layer_index}:replay:2",
            f"{layer_index}:next",
        ]
        assert observed == expected


def test_complete_transaction_canonicalizes_real_fourteen_layer_state_chain(
    monkeypatch,
) -> None:
    target, cache, bridge, _events = _patch_complete_transaction(
        monkeypatch,
        use_live_fourteen_layer_state_chain=True,
    )

    result = stack.compile_install_lfm2_semantic_layer_stack_postfold(
        cache=cache,
        bridge=bridge,
        target_model=target,
        interface=SimpleNamespace(producer_encoder=torch.eye(2)),
        compiler_lineage={
            index: object() for index in stack._EXPECTED_ATTENTION_INDICES
        },
        parent_model_directory="unused",
        donor_tokenizer=object(),
        target_tokenizer=object(),
        attention_installer=lambda **_kwargs: None,
    )

    changed = result.receipt["changed_state_keys"]
    assert changed == sorted(set(changed))
    assert changed.index("model.layers.10.feed_forward.w1.weight") < changed.index(
        "model.layers.2.feed_forward.w1.weight"
    )
    authenticated = authenticate_lfm2_semantic_layer_stack_attention_receipts(
        result.receipt
    )
    assert authenticated["causal_model_state_chain_authenticated"] is True
    assert authenticated["layer_stack_changed_state_key_count"] == len(changed)


def test_complete_transaction_routes_and_authenticates_cuda_mlp_folding(
    monkeypatch,
) -> None:
    target, cache, bridge, _events = _patch_complete_transaction(monkeypatch)

    result = stack.compile_install_lfm2_semantic_layer_stack_postfold(
        cache=cache,
        bridge=bridge,
        target_model=target,
        interface=SimpleNamespace(producer_encoder=torch.eye(2)),
        compiler_lineage={
            index: object() for index in stack._EXPECTED_ATTENTION_INDICES
        },
        parent_model_directory="unused",
        donor_tokenizer=object(),
        target_tokenizer=object(),
        attention_installer=lambda **_kwargs: None,
        fold_compute_device="cuda:1",
    )

    assert result.receipt["stack_compute_device_router_present"] is True
    assert result.receipt["fold_compute_device"] == "cuda:1"
    assert result.receipt["all_mlp_layers_proved_requested_fold_compute_device"] is True
    assert (
        result.receipt["all_short_conv_layers_proved_requested_fold_compute_device"]
        is True
    )
    assert all(
        item["mlp_fold_compute_device"] == "cuda:1"
        and item["mlp_weight_signature_compute_execution_sha256"] == "9" * 64
        for item in result.receipt["layer_installations"]
    )
    short_conv_layers = [
        item
        for item in result.receipt["layer_installations"]
        if item["operator_kind"] == "short_conv"
    ]
    assert len(short_conv_layers) == 8
    assert all(
        item["short_conv_fold_compute_device"] == "cuda:1"
        and item["short_conv_structural_cluster_compute_execution_sha256"] == "a" * 64
        and item["short_conv_writer_compute_execution_sha256"] == "b" * 64
        for item in short_conv_layers
    )


def _complete_attention_receipt_document(monkeypatch) -> dict[str, object]:
    target, cache, bridge, _events = _patch_complete_transaction(monkeypatch)
    result = stack.compile_install_lfm2_semantic_layer_stack_postfold(
        cache=cache,
        bridge=bridge,
        target_model=target,
        interface=SimpleNamespace(producer_encoder=torch.eye(2)),
        compiler_lineage={
            index: object() for index in stack._EXPECTED_ATTENTION_INDICES
        },
        parent_model_directory="unused",
        donor_tokenizer=object(),
        target_tokenizer=object(),
        attention_installer=lambda **_kwargs: None,
    )
    return json.loads(json.dumps(dict(result.receipt)))


def _reseal_attention_operator_in_stack(
    document: dict[str, object],
    *,
    layer_index: int,
    operator_body: dict[str, object],
    update_layer_installation_binding: bool = True,
) -> dict[str, object]:
    resealed_operator = {
        **operator_body,
        "result_sha256": canonical_sha256(operator_body),
    }
    layer = dict(document["layer_installations"][layer_index])
    layer.pop("result_sha256")
    layer["attention_operator_decision_receipt"] = resealed_operator
    layer["attention_operator_installation_receipt"] = resealed_operator
    layer["operator_complete_candidate_frontier_receipt"] = resealed_operator
    layer["operator_decision_sha256"] = resealed_operator["result_sha256"]
    layer["operator_installation_sha256"] = resealed_operator["result_sha256"]
    resealed_layer = {
        **layer,
        "result_sha256": canonical_sha256(layer),
    }
    document["layer_installations"][layer_index] = resealed_layer
    if update_layer_installation_binding:
        document["layer_installation_sha256"][layer_index] = resealed_layer[
            "result_sha256"
        ]
    document.pop("result_sha256")
    return {
        **document,
        "result_sha256": canonical_sha256(document),
    }


def _reseal_layer_transaction_in_stack(
    document: dict[str, object],
    *,
    layer_index: int,
    updates: dict[str, object],
) -> dict[str, object]:
    layer = dict(document["layer_installations"][layer_index])
    layer.pop("result_sha256")
    layer.update(updates)
    resealed_layer = {
        **layer,
        "result_sha256": canonical_sha256(layer),
    }
    document["layer_installations"][layer_index] = resealed_layer
    document["layer_installation_sha256"][layer_index] = resealed_layer["result_sha256"]
    document.pop("result_sha256")
    return {
        **document,
        "result_sha256": canonical_sha256(document),
    }


def test_attention_receipt_persistence_rejects_a_resealed_missing_six_tensor_bundle(
    monkeypatch,
) -> None:
    document = _complete_attention_receipt_document(monkeypatch)
    layer = document["layer_installations"][2]
    operator = dict(layer["attention_operator_decision_receipt"])
    operator.pop("result_sha256")
    analytic = dict(operator["analytic_refold"])
    analytic.pop("frozen_attention_six_tensor_bundle")
    operator["analytic_refold"] = analytic
    resealed_stack = _reseal_attention_operator_in_stack(
        document,
        layer_index=2,
        operator_body=operator,
    )

    with pytest.raises(
        Lfm2SemanticPostfoldError,
        match="omitted its six-tensor bundle",
    ):
        authenticate_lfm2_semantic_layer_stack_attention_receipts(resealed_stack)


@pytest.mark.parametrize("layer_index", (0, 7))
def test_attention_receipt_persistence_rejects_resealed_layer_state_chain_drift(
    monkeypatch,
    layer_index: int,
) -> None:
    document = _complete_attention_receipt_document(monkeypatch)
    resealed_stack = _reseal_layer_transaction_in_stack(
        document,
        layer_index=layer_index,
        updates={"model_state_before_operator_sha256": "f" * 64},
    )

    with pytest.raises(
        Lfm2SemanticPostfoldError,
        match=f"layer {layer_index} causal state chain changed",
    ):
        authenticate_lfm2_semantic_layer_stack_attention_receipts(resealed_stack)


def test_attention_receipt_persistence_rejects_resealed_terminal_state_chain_drift(
    monkeypatch,
) -> None:
    document = _complete_attention_receipt_document(monkeypatch)
    document.pop("result_sha256")
    document["model_state_after_sha256"] = "f" * 64
    resealed_stack = {
        **document,
        "result_sha256": canonical_sha256(document),
    }

    with pytest.raises(
        Lfm2SemanticPostfoldError,
        match="final causal state binding changed",
    ):
        authenticate_lfm2_semantic_layer_stack_attention_receipts(resealed_stack)


def test_attention_receipt_persistence_rejects_resealed_unowned_changed_state_key(
    monkeypatch,
) -> None:
    document = _complete_attention_receipt_document(monkeypatch)
    document.pop("result_sha256")
    document["changed_state_keys"] = sorted(
        [*document["changed_state_keys"], "model.layers.0.unowned.weight"]
    )
    resealed_stack = {
        **document,
        "result_sha256": canonical_sha256(document),
    }

    with pytest.raises(
        Lfm2SemanticPostfoldError,
        match="changed-key or parameter mutation bindings changed",
    ):
        authenticate_lfm2_semantic_layer_stack_attention_receipts(resealed_stack)


def test_attention_receipt_persistence_names_noncanonical_stack_state_keys(
    monkeypatch,
) -> None:
    document = _complete_attention_receipt_document(monkeypatch)
    document.pop("result_sha256")
    document["changed_state_keys"] = list(reversed(document["changed_state_keys"]))
    resealed_stack = {
        **document,
        "result_sha256": canonical_sha256(document),
    }

    with pytest.raises(
        Lfm2SemanticPostfoldError,
        match="changed_state_keys_canonical",
    ):
        authenticate_lfm2_semantic_layer_stack_attention_receipts(resealed_stack)


def test_attention_receipt_persistence_rejects_resealed_parameter_count_drift(
    monkeypatch,
) -> None:
    document = _complete_attention_receipt_document(monkeypatch)
    document.pop("result_sha256")
    document["parameter_element_count"] += 1
    resealed_stack = {
        **document,
        "result_sha256": canonical_sha256(document),
    }

    with pytest.raises(
        Lfm2SemanticPostfoldError,
        match="receipt bindings disagree",
    ):
        authenticate_lfm2_semantic_layer_stack_attention_receipts(resealed_stack)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("model_state_before_sha256", "f" * 64),
        ("transport_evidence_sha256", ["f" * 64]),
    ),
)
def test_attention_receipt_persistence_rejects_resealed_layer_cross_binding_drift(
    monkeypatch,
    field: str,
    replacement: object,
) -> None:
    document = _complete_attention_receipt_document(monkeypatch)
    operator = dict(
        document["layer_installations"][2]["attention_operator_decision_receipt"]
    )
    operator.pop("result_sha256")
    operator[field] = replacement
    resealed_stack = _reseal_attention_operator_in_stack(
        document,
        layer_index=2,
        operator_body=operator,
    )

    with pytest.raises(
        Lfm2SemanticPostfoldError,
        match="receipt bindings disagree",
    ):
        authenticate_lfm2_semantic_layer_stack_attention_receipts(resealed_stack)


def test_attention_receipt_persistence_rejects_resealed_layer_index_hash_drift(
    monkeypatch,
) -> None:
    document = _complete_attention_receipt_document(monkeypatch)
    layer = dict(document["layer_installations"][2])
    layer.pop("result_sha256")
    layer["operator_decision_count"] = 2
    document["layer_installations"][2] = {
        **layer,
        "result_sha256": canonical_sha256(layer),
    }
    document.pop("result_sha256")
    resealed_stack = {
        **document,
        "result_sha256": canonical_sha256(document),
    }

    with pytest.raises(
        Lfm2SemanticPostfoldError,
        match="transaction binding changed",
    ):
        authenticate_lfm2_semantic_layer_stack_attention_receipts(resealed_stack)


def test_attention_receipt_persistence_rejects_resealed_numeric_vo_publication(
    monkeypatch,
) -> None:
    document = _complete_attention_receipt_document(monkeypatch)
    operator = dict(
        document["layer_installations"][2]["attention_operator_decision_receipt"]
    )
    operator.pop("result_sha256")
    analytic = dict(operator["analytic_refold"])
    value_output_gate = dict(analytic["value_output_candidate_gate"])
    value_output_gate.pop("gate_sha256")
    value_output_gate["numeric_metric_used_for_publication_or_rollback"] = True
    analytic["value_output_candidate_gate"] = {
        **value_output_gate,
        "gate_sha256": canonical_sha256(value_output_gate),
    }
    operator["analytic_refold"] = analytic
    resealed_stack = _reseal_attention_operator_in_stack(
        document,
        layer_index=2,
        operator_body=operator,
    )

    with pytest.raises(
        Lfm2SemanticPostfoldError,
        match="V/O gate permits behavioral publication",
    ):
        authenticate_lfm2_semantic_layer_stack_attention_receipts(resealed_stack)


def test_closing_tokenizer_reauthentication_fails_before_model_mutation(
    monkeypatch,
) -> None:
    target, cache, bridge, events = _patch_complete_transaction(monkeypatch)
    before = {
        name: value.detach().clone() for name, value in target.state_dict().items()
    }
    before_identity = _state_identity(target)

    def fail_closing_authentication(**_kwargs):
        raise RuntimeError("closing tokenizer reauthentication failed")

    monkeypatch.setattr(
        stack,
        "_semantic_alignments_from_session",
        fail_closing_authentication,
    )

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="exact entry bytes were restored",
    ) as captured:
        stack.compile_install_lfm2_semantic_layer_stack_postfold(
            cache=cache,
            bridge=bridge,
            target_model=target,
            interface=SimpleNamespace(producer_encoder=torch.eye(2)),
            compiler_lineage={
                index: object() for index in stack._EXPECTED_ATTENTION_INDICES
            },
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            attention_installer=lambda **_kwargs: None,
        )

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert "closing tokenizer reauthentication failed" in str(captured.value.__cause__)
    assert events == []
    assert _state_identity(target) == before_identity
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_complete_transaction_truthfully_counts_mixed_incumbent_noops(
    monkeypatch,
) -> None:
    target, cache, bridge, _events = _patch_complete_transaction(
        monkeypatch,
        noop_operator_layers=frozenset({0, 2}),
        noop_mlp_layers=frozenset({1}),
    )

    result = stack.compile_install_lfm2_semantic_layer_stack_postfold(
        cache=cache,
        bridge=bridge,
        target_model=target,
        interface=SimpleNamespace(producer_encoder=torch.eye(2)),
        compiler_lineage={
            index: object() for index in stack._EXPECTED_ATTENTION_INDICES
        },
        parent_model_directory="unused",
        donor_tokenizer=object(),
        target_tokenizer=object(),
        attention_installer=lambda **_kwargs: None,
    )

    receipt = result.receipt
    assert receipt["attention_operator_decision_count"] == 6
    assert receipt["attention_operator_install_count"] == 5
    assert receipt["attention_operator_incumbent_noop_count"] == 1
    assert receipt["short_conv_operator_decision_count"] == 8
    assert receipt["short_conv_operator_install_count"] == 7
    assert receipt["short_conv_operator_incumbent_noop_count"] == 1
    assert receipt["nonterminal_mlp_writer_decision_count"] == 13
    assert receipt["nonterminal_mlp_writer_install_count"] == 12
    assert receipt["nonterminal_mlp_writer_incumbent_noop_count"] == 1
    assert receipt["final_mlp_producer_incumbent_noop_count"] == 1
    assert receipt["heldout_selected_incumbent_noop_count"] == 4
    assert receipt["layer_installations"][1]["mlp_incumbent_noop_selected"] is True
    assert receipt["layer_installations"][13]["mlp_incumbent_noop_selected"] is True
    assert receipt["complete_causal_layer_stack_adjudicated"] is True
    assert receipt["complete_causal_layer_stack_installed"] is False
    assert receipt["whole_stage_contains_meaningful_parameter_changes"] is True
    assert receipt["complete_nonzero_structural_producer_coverage"] is True
    assert receipt["blank_or_partially_blank_producer_chain_count"] == 0
    assert receipt["authenticated_nonzero_folded_seed_retained_noop_count"] == 4


def test_blank_producer_chain_cannot_be_accepted_as_an_incumbent_noop(
    monkeypatch,
) -> None:
    target, cache, bridge, _events = _patch_complete_transaction(
        monkeypatch,
        noop_operator_layers=frozenset({0}),
        blank_operator_layers=frozenset({0}),
    )
    before = _state_identity(target)

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="exact entry bytes were restored",
    ):
        stack.compile_install_lfm2_semantic_layer_stack_postfold(
            cache=cache,
            bridge=bridge,
            target_model=target,
            interface=SimpleNamespace(producer_encoder=torch.eye(2)),
            compiler_lineage={
                index: object() for index in stack._EXPECTED_ATTENTION_INDICES
            },
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            attention_installer=lambda **_kwargs: None,
        )

    assert _state_identity(target) == before


def test_installed_but_structurally_partial_producer_chain_rolls_back(
    monkeypatch,
) -> None:
    target, cache, bridge, _events = _patch_complete_transaction(
        monkeypatch,
        incomplete_mlp_after_layers=frozenset({2}),
    )
    before = _state_identity(target)

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="exact entry bytes were restored",
    ):
        stack.compile_install_lfm2_semantic_layer_stack_postfold(
            cache=cache,
            bridge=bridge,
            target_model=target,
            interface=SimpleNamespace(producer_encoder=torch.eye(2)),
            compiler_lineage={
                index: object() for index in stack._EXPECTED_ATTENTION_INDICES
            },
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            attention_installer=lambda **_kwargs: None,
        )

    assert _state_identity(target) == before


def test_complete_transaction_rejects_an_all_incumbent_noop_stack(
    monkeypatch,
) -> None:
    target, cache, bridge, _events = _patch_complete_transaction(
        monkeypatch,
        noop_operator_layers=frozenset(range(14)),
        noop_mlp_layers=frozenset(range(13)),
    )
    before = _state_identity(target)

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="exact entry bytes were restored",
    ):
        stack.compile_install_lfm2_semantic_layer_stack_postfold(
            cache=cache,
            bridge=bridge,
            target_model=target,
            interface=SimpleNamespace(producer_encoder=torch.eye(2)),
            compiler_lineage={
                index: object() for index in stack._EXPECTED_ATTENTION_INDICES
            },
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            attention_installer=lambda **_kwargs: None,
        )

    assert _state_identity(target) == before


def test_partial_layer_failure_restores_exact_whole_model(monkeypatch) -> None:
    target, cache, bridge, _events = _patch_complete_transaction(
        monkeypatch,
        failing_layer=3,
    )
    before = {
        name: value.detach().clone() for name, value in target.state_dict().items()
    }
    before_identity = _state_identity(target)

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="exact entry bytes were restored",
    ):
        stack.compile_install_lfm2_semantic_layer_stack_postfold(
            cache=cache,
            bridge=bridge,
            target_model=target,
            interface=SimpleNamespace(producer_encoder=torch.eye(2)),
            compiler_lineage={
                index: object() for index in stack._EXPECTED_ATTENTION_INDICES
            },
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            attention_installer=lambda **_kwargs: None,
        )

    assert _state_identity(target) == before_identity
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def _chart_inputs(*, reject_validation: bool):
    alignments = []
    parent = {}
    child = {}
    for batch in range(8):
        phase = SimpleNamespace(
            target_to_donor_boundary_positions=(0, 1),
        )
        alignments.append(SimpleNamespace(source_batch_index=batch, phases=(phase,)))
        source = torch.tensor(
            [[[batch + 1.0, 0.0, 0.0], [batch + 2.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        factor = -2.0 if reject_validation and batch >= 6 else 2.0
        parent[batch] = SimpleNamespace(final_histories={"residual.pre": source})
        child[batch] = source * factor
    bank = SimpleNamespace(
        captured_child_model_state_sha256="8" * 64,
        pre_residual_by_batch=child,
    )
    return tuple(alignments), parent, bank


def test_residual_chart_preserves_unsupported_subspace_with_diagnostic_metrics() -> (
    None
):
    alignments, parent, bank = _chart_inputs(reject_validation=False)

    chart, receipt = stack._fit_residual_chart(
        alignments=alignments,
        parent=parent,
        child_bank=bank,
        producer_encoder=torch.eye(3, dtype=torch.float64),
        layer_index=0,
    )

    assert receipt["selected_candidate"] != "base_identity"
    assert receipt["supported_rank"] == 1
    assert receipt["candidate_family_constructed_from_fit_only"] is True
    assert receipt["candidate_frozen_before_validation"] is True
    assert receipt["validation_triggered_refit"] is False
    assert receipt["all_scalar_metrics_diagnostic_only"] is True
    assert receipt["numeric_metric_used_for_candidate_selection"] is False
    assert receipt["numeric_metric_used_for_publication_or_rollback"] is False
    assert set(receipt["partition_batch_indices"]) == set(stack._PARTITIONS)
    torch.testing.assert_close(
        chart[1:],
        torch.eye(3, dtype=torch.float64)[1:],
        rtol=0,
        atol=1e-12,
    )


def test_residual_chart_validation_regression_cannot_rollback_fit_chart() -> None:
    alignments, parent, bank = _chart_inputs(reject_validation=True)

    chart, receipt = stack._fit_residual_chart(
        alignments=alignments,
        parent=parent,
        child_bank=bank,
        producer_encoder=torch.eye(3, dtype=torch.float64),
        layer_index=0,
    )

    assert receipt["selected_before_validation"] != "base_identity"
    assert receipt["selected_candidate"] == "rank_supported_ridge_0.0001"
    assert receipt["validation_nonregression_passed"] is False
    assert receipt["validation_rejection_fell_back_to_base"] is False
    assert receipt["all_scalar_metrics_diagnostic_only"] is True
    assert receipt["numeric_metric_used_for_publication_or_rollback"] is False
    assert not torch.equal(chart, torch.eye(3, dtype=torch.float64))


def test_self_dual_global_chart_forbids_independent_layer_rotation() -> None:
    alignments, parent, bank = _chart_inputs(reject_validation=False)
    encoder = torch.eye(3, dtype=torch.float64)

    chart, receipt = stack._fit_residual_chart(
        alignments=alignments,
        parent=parent,
        child_bank=bank,
        producer_encoder=encoder,
        layer_index=0,
        global_gauge_required=True,
    )

    torch.testing.assert_close(chart, encoder, rtol=0, atol=0)
    assert receipt["selected_candidate"] == "base_identity"
    assert receipt["global_self_dual_gauge_required"] is True
    assert receipt["independent_layer_rotation_allowed"] is False
    assert receipt["global_gauge_preserved_without_layer_local_rotation"] is True


def test_attention_decode_uses_reconstructed_history_not_suffix_rows(
    monkeypatch,
) -> None:
    phases = (
        SimpleNamespace(
            phase="prefill",
            donor_full_prefix_tokens=2,
            target_full_prefix_tokens=2,
            alignment_sha256="9" * 64,
        ),
        SimpleNamespace(
            phase="decode.000",
            donor_full_prefix_tokens=3,
            target_full_prefix_tokens=3,
            alignment_sha256="0" * 64,
        ),
    )
    alignment = SimpleNamespace(
        source_batch_index=0,
        phase_order=("prefill", "decode.000"),
        phases=phases,
    )
    reference = SimpleNamespace(index=0)
    tensors = {
        "prefill.layer.002.attention.q": torch.ones((1, 1, 2, 1)),
        "decode.000.layer.002.attention.q": torch.ones((1, 1, 1, 1)),
        "prefill.layer.002.attention.k": torch.ones((1, 1, 2, 1)),
        "decode.000.layer.002.attention.k": torch.ones((1, 1, 1, 1)),
    }

    class Session:
        batch_references = (reference,)

        def load_selected_tensors(self, *_args):
            return SimpleNamespace(tensors=tensors)

        def release_selected_tensors(self, _payload):
            pass

    class Capability:
        capability_sha256 = "1" * 64

        def __init__(self, **_kwargs):
            pass

        def require_exact_history(self):
            return self

        def to_evidence_dict(self):
            return {"capability_sha256": self.capability_sha256}

    class History:
        def __init__(
            self,
            query_normalized_pre_rope,
            key_normalized_pre_rope,
            value,
            source_capability_sha256="1" * 64,
            source_tensor_names=(),
        ):
            self.query_normalized_pre_rope = query_normalized_pre_rope
            self.key_normalized_pre_rope = key_normalized_pre_rope
            self.value = value
            self.source_capability_sha256 = source_capability_sha256
            self.source_tensor_names = source_tensor_names

        def to_evidence_dict(self):
            return {"history_sha256": "2" * 64}

    complete_history = History(
        torch.ones((1, 1, 3, 1)),
        torch.ones((1, 1, 3, 1)),
        torch.ones((1, 1, 3, 1)),
        source_tensor_names=(
            "prefill.layer.002.attention.q",
            "decode.000.layer.002.attention.q",
        ),
    )

    def transport(phase, **_kwargs):
        if phase.phase == "prefill":
            return SimpleNamespace(
                query_target_positions=(0, 1),
                query_donor_positions=(0, 1),
            )
        return SimpleNamespace(
            query_target_positions=(0, 2),
            query_donor_positions=(0, 2),
        )

    endpoints = []

    class Endpoint:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            endpoints.append(self)

    class Evidence:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.evidence_sha256 = canonical_sha256(
                ["attention", kwargs["alignment_sha256"]]
            )

    monkeypatch.setattr(stack, "_attention_tensor_names", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        stack,
        "inspect_lfm2_semantic_attention_cache_capability",
        lambda *_args, **_kwargs: Capability(),
    )
    monkeypatch.setattr(
        stack,
        "assemble_lfm2_explicit_pre_rope_history",
        lambda *_args, **_kwargs: complete_history,
    )
    monkeypatch.setattr(stack, "Lfm2SemanticAttentionCacheCapability", Capability)
    monkeypatch.setattr(stack, "SemanticAttentionPreRopeHistory", History)
    monkeypatch.setattr(
        stack,
        "transport_lfm2_phase_probability_preserving_attention",
        transport,
    )
    monkeypatch.setattr(stack, "Lfm2SemanticAttentionReplayEndpoint", Endpoint)
    monkeypatch.setattr(stack, "Lfm2SemanticAttentionTransportEvidence", Evidence)
    monkeypatch.setattr(
        stack,
        "_attention_tensor_names",
        lambda *_args, **_kwargs: tuple(tensors),
    )
    full_post = torch.tensor(
        [[[10.0, 100.0], [11.0, 110.0], [12.0, 120.0]]],
        dtype=torch.float32,
    )
    parent = {
        0: SimpleNamespace(
            phase_rows={
                "prefill": {"residual.post_operator": full_post[:, :2]},
                "decode.000": {"residual.post_operator": full_post[:, 2:]},
            },
            phase_histories={
                "prefill": {"residual.post_operator": full_post[:, :2]},
                "decode.000": {"residual.post_operator": full_post},
            },
        )
    }
    replay = SimpleNamespace(operator_input=torch.ones((1, 3, 2)))
    child_bank = SimpleNamespace(
        captured_child_model_state_sha256="3" * 64,
        bank_sha256="4" * 64,
        replay_by_batch={0: replay},
        pre_residual_by_batch={0: torch.zeros((1, 3, 2))},
    )
    lineage = SimpleNamespace(
        source_inventory=object(),
        query_groups=((0,),),
        query_group_weights=((1.0,),),
        compiler_lineage_sha256="5" * 64,
    )

    result = stack._build_attention_evidence(
        cache=SimpleNamespace(cache_key="6" * 64),
        session=Session(),
        bridge=SimpleNamespace(artifact_sha256="7" * 64),
        alignments=(alignment,),
        parent=parent,
        child_bank=child_bank,
        chart=torch.eye(2, dtype=torch.float64),
        chart_receipt={"result_sha256": "8" * 64},
        layer_index=2,
        lineage=lineage,
        target_rope_inv_freq=torch.ones((1,)),
        target_rope_attention_scaling=1.0,
    )

    assert len(result) == 2
    torch.testing.assert_close(
        endpoints[1].parent_post_attention_residual_target_space,
        full_post.index_select(1, torch.tensor((0, 2))).to(torch.float32),
        rtol=0,
        atol=0,
    )


def test_parent_loader_reconstructs_complete_final_norm_history() -> None:
    phases = (
        SimpleNamespace(
            phase="prefill",
            donor_full_prefix_tokens=2,
            target_full_prefix_tokens=2,
            alignment_sha256="1" * 64,
        ),
        SimpleNamespace(
            phase="decode.000",
            donor_full_prefix_tokens=3,
            target_full_prefix_tokens=3,
            alignment_sha256="2" * 64,
        ),
    )
    alignment = SimpleNamespace(
        source_batch_index=0,
        alignment_sha256="3" * 64,
        phases=phases,
    )
    names = {
        f"{phase.phase}.layer.000.{field}"
        for phase in phases
        for field in stack._COMMON_PARENT_FIELDS
    }
    names.update(f"{phase.phase}.output.final_norm_output" for phase in phases)
    tensors = {}
    for field_index, field in enumerate(stack._COMMON_PARENT_FIELDS):
        full = torch.arange(6, dtype=torch.bfloat16).reshape(1, 3, 2) + 10 * field_index
        tensors[f"prefill.layer.000.{field}"] = full[:, :2]
        tensors[f"decode.000.layer.000.{field}"] = full[:, 2:]
    final_norm = torch.tensor(
        [[[101.0, 102.0], [103.0, 104.0], [105.0, 106.0]]],
        dtype=torch.bfloat16,
    )
    tensors["prefill.output.final_norm_output"] = final_norm[:, :2]
    tensors["decode.000.output.final_norm_output"] = final_norm[:, 2:]

    class Session:
        batch_references = (
            SimpleNamespace(index=0, ordered_tensor_names=tuple(sorted(names))),
        )

        def load_selected_tensors(self, *_args):
            return SimpleNamespace(tensors=tensors)

        def release_selected_tensors(self, _payload):
            pass

    result = stack._load_parent_layer_trajectories(
        Session(),
        alignments=(alignment,),
        layer_index=0,
        operator_kind="full_attention",
        include_final_norm=True,
    )[0]

    torch.testing.assert_close(
        result.phase_histories["decode.000"]["output.final_norm_output"],
        final_norm,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result.final_histories["output.final_norm_output"],
        final_norm,
        rtol=0,
        atol=0,
    )


def test_mlp_evidence_expands_exact_boundaries_and_skips_insertions() -> None:
    phases = (
        SimpleNamespace(
            phase="prefill",
            donor_full_prefix_tokens=3,
            target_full_prefix_tokens=4,
            target_to_donor_boundary_positions=(0, None, 1, 2),
            semantic_phase_sha256="4" * 64,
            alignment_sha256="5" * 64,
        ),
        SimpleNamespace(
            phase="decode.000",
            donor_full_prefix_tokens=4,
            target_full_prefix_tokens=6,
            target_to_donor_boundary_positions=(0, None, 1, 2, None, 3),
            semantic_phase_sha256="6" * 64,
            alignment_sha256="7" * 64,
        ),
    )
    alignment = SimpleNamespace(
        source_batch_index=0,
        alignment_sha256="8" * 64,
        phases=phases,
    )
    parent_gate = torch.arange(16, dtype=torch.bfloat16).reshape(1, 4, 4)
    parent_up = parent_gate + 20
    parent_post = parent_gate + 40
    parent_layer = torch.arange(12, dtype=torch.bfloat16).reshape(1, 4, 3)
    parent_norm = parent_layer + 60
    parent_post_operator = parent_layer + 80
    ffn_norm_weight = torch.tensor(
        [1.0, 0.5, 1.5],
        dtype=torch.bfloat16,
    )
    norm_epsilon = 1e-5
    residual_float = parent_post_operator.to(torch.float32)
    reconstructed_parent_input = ffn_norm_weight * (
        residual_float
        * torch.rsqrt(residual_float.square().mean(dim=-1, keepdim=True) + norm_epsilon)
    ).to(torch.bfloat16)
    cached_parent_input = reconstructed_parent_input.clone()
    cached_parent_input[0, 0, 0] += torch.tensor(
        torch.finfo(torch.bfloat16).eps,
        dtype=torch.bfloat16,
    )

    def histories(tokens: int):
        return {
            "residual.post_operator": parent_post_operator[:, :tokens],
            "mlp.input": cached_parent_input[:, :tokens],
            "mlp.gate": parent_gate[:, :tokens],
            "mlp.up": parent_up[:, :tokens],
            "mlp.post_swiglu": parent_post[:, :tokens],
            "residual.post_layer": parent_layer[:, :tokens],
            "output.final_norm_output": parent_norm[:, :tokens],
        }

    trajectory = SimpleNamespace(
        phase_histories={
            "prefill": histories(3),
            "decode.000": histories(4),
        }
    )
    child_input = torch.tensor(
        [[[10.0 + row, 20.0 + row] for row in range(6)]],
        dtype=torch.bfloat16,
    )
    child_pre = child_input + 100
    child_bank = SimpleNamespace(
        captured_child_model_state_sha256="9" * 64,
        replay_by_batch={
            0: SimpleNamespace(
                mlp_input=child_input,
                post_operator_residual=child_pre,
            )
        },
    )
    chart = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
        dtype=torch.float64,
    )
    inventory = Lfm2DirectLocalMlpSourceInventory(
        layer_index=0,
        parent_model_identity_sha256="a" * 64,
        parent_checkpoint_manifest_sha256="b" * 64,
        direct_local_loader_receipt_sha256="c" * 64,
        residual_assignment_sha256="d" * 64,
        compiler_lineage_sha256="e" * 64,
        residual_chart=chart,
        ffn_norm_weight=ffn_norm_weight,
        norm_epsilon=norm_epsilon,
        gate_weight=torch.ones((4, 3), dtype=torch.bfloat16),
        up_weight=torch.ones((4, 3), dtype=torch.bfloat16),
        down_weight=torch.ones((3, 4), dtype=torch.bfloat16),
    )
    result = stack._build_mlp_evidence(
        evidence_type=Lfm2SemanticMlpEndpointEvidence,
        cache=SimpleNamespace(
            cache_key="f" * 64,
            manifest={"manifest_sha256": "0" * 64},
        ),
        bridge=SimpleNamespace(artifact_sha256="1" * 64),
        interface=SimpleNamespace(producer_encoder=chart),
        alignments=(alignment,),
        parent={0: trajectory},
        child_bank=child_bank,
        chart=chart,
        source_inventory=inventory,
        layer_index=0,
        final_layer=True,
    )

    assert len(result) == 4
    assert [item.batch_index for item in result] == [0, 0, 0, 0]
    assert [item.phase_index for item in result] == [0, 1, 2, 3]
    assert len({item.phase_sha256 for item in result}) == 4
    assert [item.sample_weight for item in result] == pytest.approx(
        [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0]
    )
    assert sum(item.sample_weight for item in result) == pytest.approx(1.0)
    expected_target_positions = torch.tensor((0, 2, 3, 5), dtype=torch.long)
    torch.testing.assert_close(
        torch.cat([item.actual_child_mlp_input for item in result]),
        child_input[0].index_select(0, expected_target_positions),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.cat([item.parent_gate for item in result]),
        parent_gate[0],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.cat([item.parent_normalized_mlp_input for item in result]),
        cached_parent_input[0],
        rtol=0,
        atol=0,
    )
    assert not torch.equal(cached_parent_input, reconstructed_parent_input)
    expected_parent_layer = (parent_layer.to(torch.float64) @ chart).to(torch.bfloat16)[
        0
    ]
    torch.testing.assert_close(
        torch.cat([item.parent_post_mlp_residual_target_space for item in result]),
        expected_parent_layer,
        rtol=0,
        atol=0,
    )
    expected_parent_norm = (parent_norm.to(torch.float64) @ chart).to(torch.bfloat16)[0]
    torch.testing.assert_close(
        torch.cat([item.parent_final_norm_output_target_space for item in result]),
        expected_parent_norm,
        rtol=0,
        atol=0,
    )
    assert all(item.verify() for item in result)


def _parent_mlp_provenance_fixture(*, corrupt_down: bool = False):
    torch.manual_seed(913)
    hidden = 4
    intermediate = 6
    residual = torch.randn((1, 1, hidden), dtype=torch.bfloat16)
    norm = torch.randn((hidden,), dtype=torch.bfloat16)
    gate = torch.randn((intermediate, hidden), dtype=torch.bfloat16)
    up = torch.randn((intermediate, hidden), dtype=torch.bfloat16)
    down = torch.randn((hidden, intermediate), dtype=torch.bfloat16)
    epsilon = 1e-5
    residual_float = residual.to(torch.float32)
    mlp_input = norm * (
        residual_float
        * torch.rsqrt(residual_float.square().mean(dim=-1, keepdim=True) + epsilon)
    ).to(torch.bfloat16)
    expected_gate = torch.nn.functional.linear(mlp_input, gate)
    expected_up = torch.nn.functional.linear(mlp_input, up)
    expected_post = torch.nn.functional.silu(expected_gate) * expected_up
    expected_layer = residual + torch.nn.functional.linear(expected_post, down)
    phase = SimpleNamespace(
        phase="prefill",
        semantic_phase_sha256="a" * 64,
    )
    alignment = SimpleNamespace(
        source_batch_index=0,
        phases=(phase,),
    )
    trajectory = SimpleNamespace(
        evidence_sha256="b" * 64,
        phase_rows={
            "prefill": {
                "residual.post_operator": residual,
                "mlp.input": mlp_input,
                "mlp.gate": expected_gate,
                "mlp.up": expected_up,
                "mlp.post_swiglu": expected_post,
                "residual.post_layer": expected_layer,
            }
        },
    )
    direct_down = down.clone()
    if corrupt_down:
        direct_down[0, 0] += torch.tensor(1.0, dtype=torch.bfloat16)
    tensors = {
        "model.layers.0.feed_forward.w1.weight": gate,
        "model.layers.0.feed_forward.w3.weight": up,
        "model.layers.0.feed_forward.w2.weight": direct_down,
        "model.layers.0.ffn_norm.weight": norm,
    }
    tensor_hashes = {
        name: stack._cache_tensor_sha256(torch, value)
        for name, value in tensors.items()
    }
    state_digest = hashlib.sha256(b"neural-forge-lfm2-parent-state-v1\0")
    for name in sorted(tensor_hashes):
        state_digest.update(name.encode("utf-8"))
        state_digest.update(b"\0")
        state_digest.update(tensor_hashes[name].encode("ascii"))
        state_digest.update(b"\0")
    parent_identity_body = {
        "schema": "neural-forge-lfm2-parent-fingerprint-v2",
        "model_class": "tests.DirectParent",
        "config_sha256": "4" * 64,
        "state_dict_sha256": state_digest.hexdigest(),
        "state_tensor_count": 4,
        "runtime_buffer_sha256": hashlib.sha256(
            b"neural-forge-lfm2-parent-runtime-buffers-v1\0"
        ).hexdigest(),
        "runtime_buffer_count": 0,
    }
    parent_fingerprint = canonical_sha256(parent_identity_body)
    parent_record = {
        key: value for key, value in parent_identity_body.items() if key != "schema"
    }
    parent_record["identity_sha256"] = parent_fingerprint
    identity_receipt_body = {
        "schema": "neural-forge-direct-parent-exact-identity-reconstruction-v1",
        "cache_parent_identity_sha256": parent_fingerprint,
        "reconstructed_parent_identity_sha256": parent_fingerprint,
        "reconstructed_parent_identity": parent_identity_body,
        "exact_parent_identity_authenticated": True,
    }
    exact_identity_receipt = {
        **identity_receipt_body,
        "result_sha256": canonical_sha256(identity_receipt_body),
    }

    class ParentSession:
        manifest_sha256 = "c" * 64
        open_receipt_sha256 = "d" * 64
        config_file_sha256 = "e" * 64
        normalized_config_sha256 = parent_record["config_sha256"]
        parent_identity_reconstruction_sha256 = exact_identity_receipt["result_sha256"]
        norm_epsilon = epsilon

        def tensor(self, key):
            return tensors[key].detach().cpu().contiguous().clone()

        def identity_receipt(self):
            return exact_identity_receipt

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.ones((1,), dtype=torch.bfloat16))

    target = SimpleNamespace(
        model=SimpleNamespace(layers=(Layer(),)),
    )
    cache = SimpleNamespace(
        cache_key="f" * 64,
        parent_fingerprint=parent_fingerprint,
        layer_topology_fingerprint="1" * 64,
        manifest={"manifest_sha256": "2" * 64, "parent": parent_record},
    )
    bridge = SimpleNamespace(artifact_sha256="3" * 64)
    chart = torch.eye(hidden, dtype=torch.float64)
    chart_body = {
        "schema": "neural-forge-lfm2-layer-stack-residual-chart-v1",
        "layer_index": 0,
        "chart_sha256": stack._tensor_sha256(chart),
    }
    chart_receipt = {
        **chart_body,
        "result_sha256": canonical_sha256(chart_body),
    }
    return (
        target,
        cache,
        bridge,
        ParentSession(),
        (alignment,),
        {0: trajectory},
        chart,
        chart_receipt,
    )


def _exact_parent_checkpoint_fixture(directory: Path):
    from safetensors.torch import save_file
    from transformers import Lfm2Config, Lfm2ForCausalLM

    directory.mkdir()
    config = Lfm2Config(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
        layer_types=["full_attention"],
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
        dtype="bfloat16",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2117)
        model = Lfm2ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    payload = model.config.to_dict()
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    tensors = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in model.state_dict().items()
    }
    save_file(tensors, directory / "model.safetensors")
    parent_record = stack.fingerprint_lfm2_parent(model).as_dict()
    return payload, tensors, parent_record


def test_direct_parent_session_accepts_exact_complete_parent_identity(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "parent"
    _payload, tensors, parent_record = _exact_parent_checkpoint_fixture(directory)

    with stack._DirectParentSafetensorsSession(
        directory,
        expected_parent_identity_sha256=parent_record["identity_sha256"],
        expected_parent_record=parent_record,
    ) as session:
        torch.testing.assert_close(
            session.tensor("model.layers.0.feed_forward.w2.weight"),
            tensors["model.layers.0.feed_forward.w2.weight"],
            rtol=0,
            atol=0,
        )
        identity = session.identity_receipt()
        receipt = session.receipt()

    assert identity["exact_parent_identity_authenticated"] is True
    assert (
        identity["reconstructed_parent_identity_sha256"]
        == parent_record["identity_sha256"]
    )
    assert identity["reconstructed_state_tensor_count"] == len(tensors)
    assert (
        identity["reconstructed_runtime_buffer_count"]
        == parent_record["runtime_buffer_count"]
    )
    assert [row["name"] for row in identity["checkpoint_tensor_records"]] == sorted(
        tensors
    )
    assert receipt["checkpoint_files_rehashed_after_materialization"] is True


def test_direct_parent_session_reconstructs_tied_alias_and_runtime_buffers(
    tmp_path: Path,
) -> None:
    from safetensors.torch import save_file
    from transformers import Lfm2Config, Lfm2ForCausalLM

    directory = tmp_path / "parent"
    directory.mkdir()
    config = Lfm2Config(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
        layer_types=["full_attention"],
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
        dtype="bfloat16",
    )
    model = Lfm2ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    (directory / "config.json").write_text(
        json.dumps(model.config.to_dict()),
        encoding="utf-8",
    )
    save_file(
        {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in model.state_dict().items()
            if name != "lm_head.weight"
        },
        directory / "model.safetensors",
    )
    parent_record = stack.fingerprint_lfm2_parent(model).as_dict()

    with stack._DirectParentSafetensorsSession(
        directory,
        expected_parent_identity_sha256=parent_record["identity_sha256"],
        expected_parent_record=parent_record,
    ) as session:
        identity = session.identity_receipt()

    assert (
        identity["checkpoint_tensor_count"] + 1 == parent_record["state_tensor_count"]
    )
    assert (
        identity["reconstructed_state_tensor_count"]
        == parent_record["state_tensor_count"]
    )
    assert identity["reconstructed_runtime_buffer_count"] == 2
    assert [row["name"] for row in identity["runtime_buffer_records"]] == [
        "model.rotary_emb.inv_freq",
        "model.rotary_emb.original_inv_freq",
    ]
    assert (
        identity["reconstructed_parent_identity_sha256"]
        == parent_record["identity_sha256"]
    )


def test_direct_parent_session_rejects_one_wrong_w2_before_numeric_replay(
    tmp_path: Path,
) -> None:
    from safetensors.torch import save_file

    directory = tmp_path / "parent"
    _payload, tensors, parent_record = _exact_parent_checkpoint_fixture(directory)
    wrong = {name: value.clone() for name, value in tensors.items()}
    wrong_w2 = wrong["model.layers.0.feed_forward.w2.weight"]
    wrong_w2[0, 0] += torch.tensor(1.0, dtype=torch.bfloat16)
    save_file(wrong, directory / "model.safetensors")

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="full parent model identity",
    ):
        with stack._DirectParentSafetensorsSession(
            directory,
            expected_parent_identity_sha256=parent_record["identity_sha256"],
            expected_parent_record=parent_record,
        ):
            pass


@pytest.mark.parametrize(
    "tamper",
    ("config", "cache_record", "model_class", "identity"),
)
def test_direct_parent_session_rejects_cache_manifest_or_identity_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    directory = tmp_path / "parent"
    payload, _tensors, parent_record = _exact_parent_checkpoint_fixture(directory)
    expected_identity = parent_record["identity_sha256"]
    if tamper == "config":
        payload["norm_eps"] = 0.125
        (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    elif tamper in {"cache_record", "model_class"}:
        if tamper == "cache_record":
            parent_record["state_dict_sha256"] = "f" * 64
        else:
            parent_record["model_class"] = "evil.Impostor"
        body = {
            "schema": "neural-forge-lfm2-parent-fingerprint-v2",
            **{
                key: parent_record[key]
                for key in (
                    "model_class",
                    "config_sha256",
                    "state_dict_sha256",
                    "state_tensor_count",
                    "runtime_buffer_sha256",
                    "runtime_buffer_count",
                )
            },
        }
        expected_identity = canonical_sha256(body)
        parent_record["identity_sha256"] = expected_identity
    else:
        expected_identity = "f" * 64

    with pytest.raises(stack.Lfm2SemanticLayerStackRefoldError):
        with stack._DirectParentSafetensorsSession(
            directory,
            expected_parent_identity_sha256=expected_identity,
            expected_parent_record=parent_record,
        ):
            pass


def test_direct_parent_session_rejects_file_change_after_materialization(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "parent"
    payload, _tensors, parent_record = _exact_parent_checkpoint_fixture(directory)
    with stack._DirectParentSafetensorsSession(
        directory,
        expected_parent_identity_sha256=parent_record["identity_sha256"],
        expected_parent_record=parent_record,
    ) as session:
        payload["architectures"] = ["tampered-after-open"]
        (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(
            stack.Lfm2SemanticLayerStackRefoldError,
            match="changed after exact identity reconstruction",
        ):
            session.receipt()


def test_direct_parent_session_rejects_index_shard_path_escape(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "parent"
    _payload, tensors, parent_record = _exact_parent_checkpoint_fixture(directory)
    outside = tmp_path / "outside.safetensors"
    (directory / "model.safetensors").replace(outside)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {name: "../outside.safetensors" for name in tensors}}
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="must remain inside",
    ):
        with stack._DirectParentSafetensorsSession(
            directory,
            expected_parent_identity_sha256=parent_record["identity_sha256"],
            expected_parent_record=parent_record,
        ):
            pass


def test_direct_parent_mlp_inventory_is_functionally_bound_to_cached_endpoints() -> (
    None
):
    target, cache, bridge, session, alignments, parent, chart, chart_receipt = (
        _parent_mlp_provenance_fixture()
    )

    inventory, receipt = stack._direct_mlp_inventory(
        inventory_type=Lfm2DirectLocalMlpSourceInventory,
        parent_session=session,
        cache=cache,
        bridge=bridge,
        layer_index=0,
        operator_kind="short_conv",
        residual_chart=chart,
        chart_receipt=chart_receipt,
        attention_lineage=None,
        alignments=alignments,
        parent=parent,
        target_model=target,
    )

    assert receipt["all_cached_endpoints_bit_exact"] is True
    assert receipt["all_cached_endpoints_within_legacy_bf16_envelope"] is True
    assert receipt["numerical_equivalence_is_diagnostic_only"] is True
    assert (
        receipt["cross_kernel_equivalence_policy"][
            "fold_selection_or_quality_objective"
        ]
        is False
    )
    assert receipt["direct_parent_tensors_within_legacy_numeric_envelope"] is True
    assert receipt["semantic_endpoint_scope_is_complete_tensor_identity_proof"] is False
    assert (
        receipt["cryptographic_parent_tensor_identity_proven_by_this_receipt"] is True
    )
    assert receipt["complete_state_dict_digest_compared_to_cache"] is True
    assert receipt["exact_parent_identity_is_source_admission_authority"] is True
    assert receipt["numeric_replay_has_source_admission_authority"] is False
    assert receipt["semantic_endpoint_rows"] == 1
    assert (
        receipt["endpoint_equivalence_metrics"]["mlp.input"][
            "within_legacy_bf16_envelope"
        ]
        is True
    )
    assert receipt["cached_endpoint_sha256"]["mlp.input"] == stack._tensor_sha256(
        parent[0].phase_rows["prefill"]["mlp.input"].reshape(1, -1)
    )
    assert (
        receipt["cached_normalized_mlp_input_sha256"]
        == (receipt["cached_endpoint_sha256"]["mlp.input"])
    )
    assert (
        receipt["replayed_endpoint_sha256"]["mlp.input"]
        == (receipt["reconstructed_normalized_mlp_input_sha256"])
    )
    assert receipt["cached_mlp_input_is_normalized_evidence_source_of_truth"] is True
    assert receipt["reconstructed_mlp_input_is_diagnostic_only"] is True
    assert inventory.direct_local_loader_receipt_sha256 == receipt["result_sha256"]
    assert inventory.residual_assignment_sha256 == chart_receipt["result_sha256"]
    assert (
        inventory.authentication_body()["residual_chart_sha256"]
        == (chart_receipt["chart_sha256"])
    )
    assert inventory.compiler_lineage_sha256 == stack._mlp_lineage_sha256(
        cache=cache,
        bridge=bridge,
        layer_index=0,
        operator_kind="short_conv",
        chart_receipt=chart_receipt,
        attention_lineage=None,
    )
    torch.testing.assert_close(inventory.residual_chart, chart, rtol=0, atol=0)
    assert inventory.verify() is True


def test_real_v3_parent_mlp_provenance_reaches_layer0_receipt() -> None:
    target, cache, bridge, session, alignments, parent, chart, chart_receipt = (
        _parent_mlp_provenance_fixture()
    )
    _inventory, provenance = stack._direct_mlp_inventory(
        inventory_type=Lfm2DirectLocalMlpSourceInventory,
        parent_session=session,
        cache=cache,
        bridge=bridge,
        layer_index=0,
        operator_kind="short_conv",
        residual_chart=chart,
        chart_receipt=chart_receipt,
        attention_lineage=None,
        alignments=alignments,
        parent=parent,
        target_model=target,
    )

    def replay(state_sha256: str, bank_sha256: str):
        return stack._LayerReplayBank(
            captured_child_model_state_sha256=state_sha256,
            layer_index=0,
            pre_residual_by_batch=MappingProxyType({}),
            replay_by_batch=MappingProxyType({}),
            bank_sha256=bank_sha256,
        )

    operator_receipt = stack._sealed(
        {
            **_body_recipe_fields(),
            "recipe_component_lane": stack.resolve_lfm2_semantic_body_recipe(
                "fit_guided"
            ).component_lanes["short_conv"],
            "status": "installed-analytic-candidate",
            "installed": True,
        }
    )
    producer_coverage = stack._sealed({"all_required_roles_structurally_nonzero": True})
    mlp_receipt = stack._sealed(
        {
            **_body_recipe_fields(),
            "recipe_component_lane": stack.resolve_lfm2_semantic_body_recipe(
                "fit_guided"
            ).component_lanes["mlp"],
            "status": "installed-frozen-semantic-mlp-candidate",
            "installed": True,
        }
    )
    receipt = stack._layer_receipt(
        layer_index=0,
        operator_kind="short_conv",
        before_operator_replay=replay("4" * 64, "5" * 64),
        operator_receipt=operator_receipt,
        operator_coverage_decision=producer_coverage,
        after_operator_replay=replay("6" * 64, "7" * 64),
        mlp_receipt=mlp_receipt,
        mlp_coverage_decision=producer_coverage,
        mlp_source_provenance=provenance,
        propagation_replay=replay("8" * 64, "9" * 64),
        chart_receipt=chart_receipt,
        parent=parent,
        operator_evidence_sha256=("a" * 64,),
        mlp_evidence_sha256=("b" * 64,),
        fir_receipt=None,
        fold_compute_device="cpu",
    )

    assert (
        receipt["mlp_source_functional_provenance_sha256"]
        == (provenance["result_sha256"])
    )
    assert receipt["mlp_source_functional_provenance_schema"] == (
        "neural-forge-lfm2-direct-parent-mlp-functional-provenance-v3"
    )
    assert receipt["mlp_source_direct_tensors_within_legacy_numeric_envelope"] is True
    assert (
        receipt["mlp_source_all_cached_endpoints_within_legacy_bf16_envelope"] is True
    )
    assert (
        receipt["mlp_source_semantic_endpoint_scope_is_complete_tensor_identity_proof"]
        is False
    )
    assert (
        receipt[
            "mlp_source_cryptographic_parent_tensor_identity_proven_by_this_receipt"
        ]
        is True
    )
    assert (
        receipt["mlp_source_exact_parent_identity_is_source_admission_authority"]
        is True
    )
    assert receipt["mlp_source_numeric_replay_has_source_admission_authority"] is False
    assert stack._verify_sealed(receipt, "layer 0") == dict(receipt)


def _shared_body_completion_layer_receipt_fixture(
    operator_kind: str,
) -> tuple[dict[str, object], dict[str, str]]:
    recipe = stack.resolve_lfm2_semantic_body_recipe("full_strength_structural")
    private_map_sha256 = "c" * 64
    public_map_sha256 = "d" * 64
    completion_candidate = stack._sealed(
        {
            "schema": "test-body-global-coordinate-completion-candidate-v1",
            "coordinate_map_sha256": private_map_sha256,
        }
    )
    candidate_sha256 = completion_candidate["result_sha256"]
    completion_wrapper = stack._sealed(
        {
            "schema": "test-body-global-coordinate-completion-wrapper-v1",
            "selected_candidate_result_sha256": candidate_sha256,
            "coordinate_map_sha256": private_map_sha256,
            "public_coordinate_map_sha256": public_map_sha256,
        }
    )
    planting = {
        "fold_compute_device": "cpu",
        "shared_body_global_completion_sha256": candidate_sha256,
        "body_global_raw_decoder_public_sha256": public_map_sha256,
        "body_global_coordinate_map_sha256": private_map_sha256,
    }
    mlp_receipt = stack._sealed(
        {
            **_body_recipe_fields(recipe.recipe_id),
            "recipe_component_lane": recipe.component_lanes["mlp"],
            "status": "installed-frozen-semantic-mlp-candidate",
            "installed": True,
            "source_derived_planting": planting,
        }
    )
    if operator_kind == "full_attention":
        attention = _persistable_attention_operator_receipt(
            layer_index=0,
            before="4" * 64,
            after="6" * 64,
            installed=True,
            body_recipe=recipe.recipe_id,
        )
        attention_body = dict(attention)
        attention_body.pop("result_sha256")
        attention_body["normalized_input_decoder_receipt"] = {
            "raw_structural_decoder_provenance": {
                "coordinate_completion_receipt_sha256": candidate_sha256,
                "raw_structural_decoder_sha256": public_map_sha256,
                "coordinate_completion_coordinate_map_sha256": (private_map_sha256),
            }
        }
        operator_receipt = stack._sealed(attention_body)
    else:
        construction = stack._resolve_short_conv_construction(
            "body-global-coordinate-complete-structural-v1"
        )
        source_body = {
            "schema": "test-coordinate-complete-short-conv-source-v1",
            "coordinate_completion_candidate_sha256": candidate_sha256,
        }
        source_sha256 = canonical_sha256(source_body)
        operator_receipt = stack._sealed(
            {
                **_body_recipe_fields(recipe.recipe_id),
                "recipe_component_lane": recipe.component_lanes["short_conv"],
                "status": "installed-analytic-candidate",
                "installed": True,
                "short_conv_construction": construction.as_dict(),
                "short_conv_construction_id": construction.construction_id,
                "short_conv_construction_sha256": (construction.construction_sha256),
                "coordinate_complete_direct_source": {
                    **source_body,
                    "source_sha256": source_sha256,
                },
                "coordinate_complete_direct_source_sha256": source_sha256,
                "coordinate_complete_candidate_state_sha256": "e" * 64,
                "coordinate_completion_candidate_sha256": candidate_sha256,
                "shared_body_global_completion_sha256": candidate_sha256,
                "raw_target_to_parent_decoder_sha256": public_map_sha256,
                "coordinate_completion_coordinate_map_sha256": (private_map_sha256),
                "body_global_coordinate_complete_structural_fold": True,
                "coordinate_complete_candidate_selected_by_metrics": False,
                "coordinate_complete_candidate_selected_by_construction": True,
            }
        )

    def replay(state_sha256: str, bank_sha256: str) -> stack._LayerReplayBank:
        return stack._LayerReplayBank(
            captured_child_model_state_sha256=state_sha256,
            layer_index=0,
            pre_residual_by_batch=MappingProxyType({}),
            replay_by_batch=MappingProxyType({}),
            bank_sha256=bank_sha256,
        )

    coverage = stack._sealed({"all_required_roles_structurally_nonzero": True})
    chart_receipt = stack._sealed(
        {
            "schema": "test-global-gauge-residual-chart-v1",
            "global_self_dual_gauge_required": True,
            "global_gauge_preserved_without_layer_local_rotation": True,
        }
    )
    kwargs: dict[str, object] = {
        "layer_index": 0,
        "operator_kind": operator_kind,
        "before_operator_replay": replay("4" * 64, "5" * 64),
        "operator_receipt": operator_receipt,
        "operator_coverage_decision": coverage,
        "after_operator_replay": replay("6" * 64, "7" * 64),
        "mlp_receipt": mlp_receipt,
        "mlp_coverage_decision": coverage,
        "mlp_source_provenance": _mlp_functional_provenance_receipt(0),
        "propagation_replay": replay("8" * 64, "9" * 64),
        "chart_receipt": chart_receipt,
        "parent": {0: SimpleNamespace(evidence_sha256="f" * 64)},
        "operator_evidence_sha256": ("a" * 64,),
        "mlp_evidence_sha256": ("b" * 64,),
        "fir_receipt": None,
        "fold_compute_device": "cpu",
        "body_recipe": recipe,
        "body_coordinate_completion_receipt": completion_wrapper,
        "body_coordinate_completion_candidate_receipt": completion_candidate,
    }
    expected = {
        "wrapper": completion_wrapper["result_sha256"],
        "candidate": candidate_sha256,
        "public": public_map_sha256,
        "private": private_map_sha256,
    }
    return kwargs, expected


def _preaggregated_completion_source_fixture():
    operators = (
        ResidualFacingOperator(
            role="consumer",
            name="input",
            tensor=torch.tensor(
                [[1.0, 0.0], [0.0, 2.0]],
                dtype=torch.float64,
            ),
            orientation=FeatureOrientation.INPUT,
        ),
        ResidualFacingOperator(
            role="writer",
            name="output",
            tensor=torch.tensor(
                [[2.0, 1.0], [0.0, 1.0]],
                dtype=torch.float64,
            ),
            orientation=FeatureOrientation.OUTPUT,
        ),
    )
    geometry = compile_role_balanced_operator_geometry(operators)
    profile = resolve_analytic_execution_profile("cpu")
    source = stack._body_coordinate_completion_source(
        parent_operators=None,
        parent_geometry=geometry,
        least_squares_execution_profile=profile,
        recipe="bridge-procrustes-isometric",
    )
    assert source is not None
    return operators, geometry, profile, source


def _preaggregated_body_completion_layer_receipt_fixture(
    operator_kind: str,
):
    kwargs, expected = _shared_body_completion_layer_receipt_fixture(operator_kind)
    _operators, geometry, profile, source = _preaggregated_completion_source_fixture()
    legacy_wrapper = dict(kwargs["body_coordinate_completion_receipt"])
    legacy_wrapper.pop("result_sha256")
    hashes = geometry.report["hashes"]
    geometry_wrapper = stack._sealed(
        {
            **legacy_wrapper,
            "schema": (
                stack.LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_RECEIPT_SCHEMA
            ),
            "parent_geometry_sha256": geometry.geometry_sha256,
            "parent_geometry_result_sha256": geometry.geometry_sha256,
            "parent_operator_sha256s": list(geometry.ordered_operator_sha256s),
            "parent_operator_manifest_sha256": canonical_sha256(
                list(geometry.ordered_operator_sha256s)
            ),
            "parent_raw_gram_sha256": hashes["raw_gram_sha256"],
            "parent_balanced_gram_sha256": hashes["balanced_gram_sha256"],
            "least_squares_execution_profile": profile.as_dict(),
            "least_squares_execution_profile_sha256": profile.result_sha256,
            "preaggregated_parent_geometry": True,
            "parent_operator_reconstruction_calls": 0,
        }
    )
    kwargs["body_coordinate_completion_receipt"] = geometry_wrapper
    kwargs["body_coordinate_completion_source"] = source
    expected.update(
        {
            "geometry": geometry.geometry_sha256,
            "raw": hashes["raw_gram_sha256"],
            "balanced": hashes["balanced_gram_sha256"],
            "profile": profile.result_sha256,
            "operator_count": len(geometry.ordered_operator_sha256s),
        }
    )
    return kwargs, expected


def test_preaggregated_completion_source_is_authenticated_and_operator_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operators, geometry, profile, expected = _preaggregated_completion_source_fixture()

    def forbidden_operator_reconstruction(*_args, **_kwargs):
        raise AssertionError("the core reconstructed a residual-facing operator")

    monkeypatch.setattr(
        stack,
        "ResidualFacingOperator",
        forbidden_operator_reconstruction,
    )
    observed = stack._body_coordinate_completion_source(
        parent_operators=None,
        parent_geometry=geometry,
        least_squares_execution_profile=profile,
        recipe="bridge-procrustes-isometric",
    )

    assert observed is not None
    assert observed.operators == ()
    assert observed.geometry is geometry
    assert observed.least_squares_execution_profile == profile
    assert observed.ordered_operator_sha256s == geometry.ordered_operator_sha256s
    assert observed.operator_manifest_sha256 == canonical_sha256(
        list(geometry.ordered_operator_sha256s)
    )
    assert observed.raw_gram_sha256 == geometry.report["hashes"]["raw_gram_sha256"]
    assert (
        observed.balanced_gram_sha256
        == geometry.report["hashes"]["balanced_gram_sha256"]
    )
    assert len(operators) == len(observed.ordered_operator_sha256s)
    assert expected.operators == ()


def test_completion_source_preserves_legacy_and_no_completion_contracts() -> None:
    operators, _geometry, _profile, _source = _preaggregated_completion_source_fixture()

    assert (
        stack._body_coordinate_completion_source(
            parent_operators=None,
            parent_geometry=None,
            least_squares_execution_profile=None,
            recipe=None,
        )
        is None
    )
    legacy = stack._body_coordinate_completion_source(
        parent_operators=operators,
        parent_geometry=None,
        least_squares_execution_profile=None,
        recipe="bridge-procrustes-isometric",
    )
    assert legacy is not None
    assert legacy.operators == operators
    assert legacy.geometry is None
    assert legacy.least_squares_execution_profile is None


def test_completion_source_rejects_xor_recipe_and_profile_violations() -> None:
    operators, geometry, profile, _source = _preaggregated_completion_source_fixture()
    cases = (
        {
            "parent_operators": None,
            "parent_geometry": None,
            "least_squares_execution_profile": None,
            "recipe": "bridge-procrustes-isometric",
        },
        {
            "parent_operators": operators,
            "parent_geometry": geometry,
            "least_squares_execution_profile": profile,
            "recipe": "bridge-procrustes-isometric",
        },
        {
            "parent_operators": None,
            "parent_geometry": geometry,
            "least_squares_execution_profile": None,
            "recipe": "bridge-procrustes-isometric",
        },
        {
            "parent_operators": operators,
            "parent_geometry": None,
            "least_squares_execution_profile": profile,
            "recipe": "bridge-procrustes-isometric",
        },
        {
            "parent_operators": None,
            "parent_geometry": geometry,
            "least_squares_execution_profile": profile,
            "recipe": None,
        },
        {
            "parent_operators": None,
            "parent_geometry": geometry,
            "least_squares_execution_profile": profile,
            "recipe": "unknown-completion",
        },
    )

    for values in cases:
        with pytest.raises(stack.Lfm2SemanticLayerStackRefoldError):
            stack._body_coordinate_completion_source(**values)

    forged_profile = object.__new__(stack.AnalyticExecutionProfile)
    object.__setattr__(
        forged_profile,
        "receipt",
        MappingProxyType({"result_sha256": "0" * 64}),
    )
    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="profile authentication failed",
    ):
        stack._body_coordinate_completion_source(
            parent_operators=None,
            parent_geometry=geometry,
            least_squares_execution_profile=forged_profile,
            recipe="bridge-procrustes-isometric",
        )


@pytest.mark.parametrize("operator_kind", ("full_attention", "short_conv"))
def test_layer_receipt_binds_preaggregated_geometry_profile_and_manifest(
    operator_kind: str,
) -> None:
    kwargs, expected = _preaggregated_body_completion_layer_receipt_fixture(
        operator_kind
    )

    receipt = stack._layer_receipt(**kwargs)

    assert receipt["body_global_parent_geometry_sha256"] == expected["geometry"]
    assert receipt["body_global_parent_raw_gram_sha256"] == expected["raw"]
    assert receipt["body_global_parent_balanced_gram_sha256"] == expected["balanced"]
    assert (
        receipt["body_global_least_squares_execution_profile_sha256"]
        == expected["profile"]
    )
    assert receipt["body_global_parent_operator_count"] == expected["operator_count"]
    assert receipt["body_global_parent_operator_reconstruction_calls"] == 0


@pytest.mark.parametrize(
    "mutation",
    ("geometry", "raw", "profile", "drop-source", "drop-geometry-schema"),
)
def test_layer_receipt_rejects_geometry_substitution_or_drop(mutation: str) -> None:
    kwargs, _expected = _preaggregated_body_completion_layer_receipt_fixture(
        "full_attention"
    )
    if mutation == "drop-source":
        kwargs["body_coordinate_completion_source"] = None
    else:
        wrapper = dict(kwargs["body_coordinate_completion_receipt"])
        wrapper.pop("result_sha256")
        if mutation == "geometry":
            wrapper["parent_geometry_sha256"] = "9" * 64
            wrapper["parent_geometry_result_sha256"] = "9" * 64
        elif mutation == "raw":
            wrapper["parent_raw_gram_sha256"] = "9" * 64
        elif mutation == "profile":
            wrapper["least_squares_execution_profile_sha256"] = "9" * 64
        else:
            wrapper["schema"] = "test-body-global-coordinate-completion-wrapper-v1"
        kwargs["body_coordinate_completion_receipt"] = stack._sealed(wrapper)

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="geometry|source|profile",
    ):
        stack._layer_receipt(**kwargs)


def test_core_compiler_routes_geometry_profile_only_at_singleton_layer_zero() -> None:
    source = Path(stack.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    compiler = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "compile_install_lfm2_semantic_layer_stack_postfold"
    )
    parents = {
        child: parent
        for parent in ast.walk(compiler)
        for child in ast.iter_child_nodes(parent)
    }

    def call_name(call: ast.Call) -> str | None:
        return call.func.id if isinstance(call.func, ast.Name) else None

    geometry_calls = [
        call
        for call in ast.walk(compiler)
        if isinstance(call, ast.Call)
        and any(
            keyword.arg == "parent_geometry"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "completion_geometry"
            for keyword in call.keywords
        )
    ]
    assert {call_name(call) for call in geometry_calls} == {
        "lfm2_body_global_coordinate_completion_evidence_sha256",
        "compile_lfm2_body_global_coordinate_completion",
    }
    assert len(geometry_calls) == 2

    for call in geometry_calls:
        profile_keywords = [
            keyword
            for keyword in call.keywords
            if keyword.arg == "least_squares_execution_profile"
        ]
        assert len(profile_keywords) == 1
        assert isinstance(profile_keywords[0].value, ast.Name)
        assert profile_keywords[0].value.id == "completion_least_squares_profile"

        ancestors: list[ast.AST] = []
        ancestor = parents.get(call)
        while ancestor is not None:
            ancestors.append(ancestor)
            ancestor = parents.get(ancestor)
        singleton_guards = [
            candidate.test
            for candidate in ancestors
            if isinstance(candidate, ast.If)
            and isinstance(candidate.test, ast.BoolOp)
            and isinstance(candidate.test.op, ast.And)
        ]
        assert any(
            any(
                isinstance(part, ast.Name)
                and part.id == "coordinate_completion_enabled"
                for part in guard.values
            )
            and any(
                isinstance(part, ast.Compare)
                and isinstance(part.left, ast.Name)
                and part.left.id == "layer_index"
                and len(part.ops) == 1
                and isinstance(part.ops[0], ast.Eq)
                and len(part.comparators) == 1
                and isinstance(part.comparators[0], ast.Constant)
                and part.comparators[0].value == 0
                for part in guard.values
            )
            for guard in singleton_guards
        )


@pytest.mark.parametrize("operator_kind", ("full_attention", "short_conv"))
def test_layer_receipt_authenticates_one_shared_body_completion_across_components(
    operator_kind: str,
) -> None:
    kwargs, expected = _shared_body_completion_layer_receipt_fixture(operator_kind)

    receipt = stack._layer_receipt(**kwargs)

    assert receipt["body_global_coordinate_completion_sha256"] == expected["wrapper"]
    assert receipt["shared_body_global_completion_sha256"] == expected["candidate"]
    assert receipt["body_global_raw_decoder_public_sha256"] == expected["public"]
    assert receipt["body_global_coordinate_map_sha256"] == expected["private"]
    assert receipt["body_global_completion_shared_by_operator_and_mlp"] is True
    assert receipt["body_global_global_gauge_preserved"] is True


@pytest.mark.parametrize("operator_kind", ("full_attention", "short_conv"))
def test_layer_receipt_rejects_substituted_body_completion_candidate_sha256(
    operator_kind: str,
) -> None:
    kwargs, _expected = _shared_body_completion_layer_receipt_fixture(operator_kind)
    candidate = dict(kwargs["body_coordinate_completion_candidate_receipt"])
    candidate.pop("result_sha256")
    candidate["substitution_marker"] = True
    kwargs["body_coordinate_completion_candidate_receipt"] = stack._sealed(candidate)

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="one shared body-global coordinate completion",
    ):
        stack._layer_receipt(**kwargs)


def test_direct_parent_mlp_inventory_rejects_mismatched_residual_chart_receipt() -> (
    None
):
    target, cache, bridge, session, alignments, parent, chart, chart_receipt = (
        _parent_mlp_provenance_fixture()
    )
    mismatched_body = dict(chart_receipt)
    mismatched_body.pop("result_sha256")
    mismatched_body["chart_sha256"] = "9" * 64
    mismatched_receipt = {
        **mismatched_body,
        "result_sha256": canonical_sha256(mismatched_body),
    }

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="does not authenticate the selected MLP chart",
    ):
        stack._direct_mlp_inventory(
            inventory_type=Lfm2DirectLocalMlpSourceInventory,
            parent_session=session,
            cache=cache,
            bridge=bridge,
            layer_index=0,
            operator_kind="short_conv",
            residual_chart=chart,
            chart_receipt=mismatched_receipt,
            attention_lineage=None,
            alignments=alignments,
            parent=parent,
            target_model=target,
        )


def test_parent_mlp_functional_provenance_accepts_bf16_backend_rounding_drift() -> None:
    target, cache, bridge, session, alignments, parent, chart, chart_receipt = (
        _parent_mlp_provenance_fixture()
    )
    phase_rows = parent[0].phase_rows["prefill"]
    gate = phase_rows["mlp.gate"].clone()
    gate.view(-1)[0] = (
        gate.view(-1)[0].to(torch.float32)
        + torch.tensor(
            torch.finfo(torch.bfloat16).eps / 2.0,
            dtype=torch.float32,
        )
    ).to(torch.bfloat16)
    phase_rows["mlp.gate"] = gate

    _inventory, receipt = stack._direct_mlp_inventory(
        inventory_type=Lfm2DirectLocalMlpSourceInventory,
        parent_session=session,
        cache=cache,
        bridge=bridge,
        layer_index=0,
        operator_kind="short_conv",
        residual_chart=chart,
        chart_receipt=chart_receipt,
        attention_lineage=None,
        alignments=alignments,
        parent=parent,
        target_model=target,
    )

    assert receipt["all_cached_endpoints_bit_exact"] is False
    assert receipt["all_cached_endpoints_within_legacy_bf16_envelope"] is True
    assert (
        receipt["endpoint_equivalence_metrics"]["mlp.gate"][
            "within_legacy_bf16_envelope"
        ]
        is True
    )


def test_cross_kernel_nonfinite_replay_is_saturated_into_finite_diagnostics() -> None:
    policy = stack._bf16_cross_kernel_equivalence_policy()
    replayed = torch.tensor([[float("inf"), float("nan")]], dtype=torch.bfloat16)
    cached = torch.zeros_like(replayed)

    metrics = stack._bf16_cross_kernel_equivalence_metrics(
        replayed=replayed,
        cached=cached,
        policy=policy,
        field_name="mlp.gate",
    )

    assert metrics["diagnostics_finite"] is True
    assert metrics["replayed_all_finite"] is False
    assert metrics["nonfinite_replayed_element_count"] == 2
    assert metrics["within_legacy_bf16_envelope"] is False
    assert metrics["diagnostic_only"] is True
    assert metrics["numeric_admission_authority"] is False


def test_direct_parent_mlp_inventory_records_large_kernel_drift_without_veto() -> None:
    target, cache, bridge, session, alignments, parent, chart, chart_receipt = (
        _parent_mlp_provenance_fixture(corrupt_down=True)
    )

    _inventory, receipt = stack._direct_mlp_inventory(
        inventory_type=Lfm2DirectLocalMlpSourceInventory,
        parent_session=session,
        cache=cache,
        bridge=bridge,
        layer_index=0,
        operator_kind="short_conv",
        residual_chart=chart,
        chart_receipt=chart_receipt,
        attention_lineage=None,
        alignments=alignments,
        parent=parent,
        target_model=target,
    )
    assert receipt["all_cross_kernel_replay_diagnostics_finite"] is True
    assert receipt["all_cached_endpoints_within_legacy_bf16_envelope"] is False
    assert receipt["numeric_replay_has_source_admission_authority"] is False
    assert receipt["exact_parent_identity_is_source_admission_authority"] is True


def _invoke_mlp_noop_receipt(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str,
    final_layer: bool,
) -> MappingProxyType | dict[str, object]:
    target = _DenseCoverageTarget()
    state = _state_identity(target)
    monkeypatch.setattr(
        stack,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_state_identity(model)),
    )
    cache = SimpleNamespace(
        cache_key="a" * 64,
        manifest={"manifest_sha256": "c" * 64},
    )
    bridge = SimpleNamespace(artifact_sha256="b" * 64)
    evidence = (SimpleNamespace(captured_child_model_state_sha256=state),)
    parameter_count = stack._parameter_count(target)
    terminal_evidence = _terminal_banks() if final_layer else {}
    recipe = stack.resolve_lfm2_semantic_body_recipe("fit_guided")
    body: dict[str, object] = {
        **_body_recipe_fields(recipe.recipe_id),
        "recipe_component_lane": recipe.component_lanes["mlp"],
        "schema": stack._MLP_INSTALLATION_SCHEMA,
        "status": status,
        "installed": False,
        "install_count": 0,
        "layer_index": 0,
        "model_state_before_sha256": state,
        "model_state_after_sha256": state,
        "parameter_count_before": parameter_count,
        "parameter_count_after": parameter_count,
        "parameter_count_changed": False,
        "topology_changed": False,
        "parameter_identities_preserved": True,
        "parameter_storage_preserved": True,
        "gradient_state_preserved": True,
        "exact_noop_state_proven": True,
        "changed_parameter_roles": [],
        "candidate_frozen_before_validation": True,
        "validation_triggered_refit": False,
        "analytic_linear_and_cluster_solves_only": True,
        "final_layer_mlp_w2_deferred_to_fused_terminal": final_layer,
        "downstream_terminal_replay_and_validation_required": final_layer,
        "terminal_action_evidence_sha256": {
            split: value.authentication_sha256
            for split, value in terminal_evidence.items()
        },
        "cache_key": cache.cache_key,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "evidence_endpoint_count": len(evidence),
        "operation_audit": dict(stack._ZERO_OPERATIONS),
    }
    if status == stack._MLP_STRUCTURAL_NOOP_STATUS:
        body.update(
            {
                "selected_candidate": "incumbent-noop",
                "selected_candidate_state_identical_to_incumbent": True,
                "incumbent_noop_fallback_used": True,
                "incumbent_noop_fallback_reason": (
                    "all-analytic-proposals-structurally-invalid-or-identical"
                ),
                "incumbent_selected_by_heldout_gates": False,
                "structurally_eligible_candidate_count": 0,
                "candidate_eligibility_basis": (
                    "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
                ),
                "structural_safety_is_only_candidate_veto": True,
                "fit_metrics_used_for_candidate_ordering": True,
                "fit_metrics_used_for_structural_eligibility": False,
                "heldout_metrics_are_diagnostic_only": True,
                "validation_publication_veto_applied": False,
                "rollback_authorized": False,
                "numeric_metrics_have_final_intelligence_authority": False,
                "validation_rejected_candidate": None,
            }
        )
    elif status == "heldout-selected-incumbent-noop":
        body.update(
            {
                "incumbent_selected_by_heldout_gates": True,
                "fit_check_nonregression_passed": True,
                "selection_nonregression_passed": True,
                "validation_nonregression_passed": True,
            }
        )
    elif status == "deterministic-fit-correction-rounded-to-incumbent-noop":
        body.update(
            {
                "incumbent_selected_by_heldout_gates": False,
                "heldout_metrics_are_diagnostic_only": True,
                "numeric_metric_used_for_publication_or_rollback": False,
            }
        )
    elif status == "prepared-final-w2-deferred" and final_layer:
        body.update(
            {
                "selected_candidate": "incumbent-producer-w2",
                "selected_candidate_state_identical_to_incumbent": False,
                "incumbent_noop_fallback_used": False,
                "incumbent_noop_fallback_reason": None,
                "incumbent_selected_by_heldout_gates": False,
                "structurally_eligible_candidate_count": 2,
                "candidate_eligibility_basis": (
                    "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
                ),
                "structural_safety_is_only_candidate_veto": True,
                "fit_metrics_used_for_candidate_ordering": True,
                "fit_metrics_used_for_structural_eligibility": False,
                "heldout_metrics_are_diagnostic_only": True,
                "validation_publication_veto_applied": False,
                "rollback_authorized": False,
                "numeric_metrics_have_final_intelligence_authority": False,
                "validation_rejected_candidate": None,
            }
        )
    else:  # pragma: no cover - constrained by focused callers below
        raise AssertionError(f"unsupported test status: {status}")
    receipt = {**body, "result_sha256": canonical_sha256(body)}
    result = SimpleNamespace(
        receipt=receipt,
        final_layer_mlp_evidence=terminal_evidence,
    )
    accepted, _terminal, terminal_prior = stack._install_mlp(
        target_model=target,
        layer_index=0,
        evidence=evidence,
        cache=cache,
        bridge=bridge,
        installer=lambda **_kwargs: result,
        final_layer=final_layer,
        parameter_count=parameter_count,
    )
    assert terminal_prior is None
    return accepted


@pytest.mark.parametrize("final_layer", (False, True))
def test_mlp_install_accepts_only_structural_no_candidate_noop_reason(
    monkeypatch: pytest.MonkeyPatch,
    final_layer: bool,
) -> None:
    receipt = _invoke_mlp_noop_receipt(
        monkeypatch,
        status=stack._MLP_STRUCTURAL_NOOP_STATUS,
        final_layer=final_layer,
    )

    assert receipt["status"] == stack._MLP_STRUCTURAL_NOOP_STATUS
    assert receipt["incumbent_noop_fallback_used"] is True
    assert receipt["structurally_eligible_candidate_count"] == 0


def test_final_mlp_install_accepts_structurally_eligible_analytic_w2_deferral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _invoke_mlp_noop_receipt(
        monkeypatch,
        status="prepared-final-w2-deferred",
        final_layer=True,
    )

    assert receipt["selected_candidate"] == "incumbent-producer-w2"
    assert receipt["incumbent_noop_fallback_used"] is False
    assert receipt["structurally_eligible_candidate_count"] == 2


@pytest.mark.parametrize("final_layer", (False, True))
@pytest.mark.parametrize(
    "status",
    (
        "heldout-selected-incumbent-noop",
        "deterministic-fit-correction-rounded-to-incumbent-noop",
    ),
)
def test_mlp_install_rejects_legacy_noop_reasons(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    final_layer: bool,
) -> None:
    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="MLP.*(?:defer|install)",
    ):
        _invoke_mlp_noop_receipt(
            monkeypatch,
            status=status,
            final_layer=final_layer,
        )


def test_attention_install_accepts_only_structural_no_candidate_noop_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _FourteenLayerReceiptStateTarget()
    state = _state_identity(target)
    monkeypatch.setattr(
        stack,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_state_identity(model)),
    )
    layer_index = 2
    evidence_sha256 = canonical_sha256(["attention", layer_index])
    evidence = (
        SimpleNamespace(
            evidence_sha256=evidence_sha256,
            actual_child_replay=SimpleNamespace(
                captured_child_model_state_sha256=state
            ),
        ),
    )
    parameter_count = stack._parameter_count(target)
    receipt = _persistable_attention_operator_receipt(
        layer_index=layer_index,
        before=state,
        after=state,
        installed=False,
        parameter_count=parameter_count,
    )

    accepted = stack._install_attention_operator(
        target_model=target,
        layer_index=layer_index,
        evidence=evidence,
        cache_key="a" * 64,
        bridge_artifact_sha256="b" * 64,
        installer=lambda **_kwargs: receipt,
        parameter_count=parameter_count,
    )

    assert accepted["status"] == stack.LFM2_SEMANTIC_ATTENTION_STRUCTURAL_NOOP_STATUS
    assert accepted["incumbent_noop_fallback_used"] is True
    assert accepted["structurally_eligible_candidate_count"] == 0


@pytest.mark.parametrize(
    "status",
    (
        "heldout-selected-incumbent-noop",
        "deterministic-fit-correction-rounded-to-incumbent-noop",
    ),
)
def test_attention_install_rejects_legacy_metric_noop_reasons(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    target = _FourteenLayerReceiptStateTarget()
    state = _state_identity(target)
    monkeypatch.setattr(
        stack,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_state_identity(model)),
    )
    layer_index = 2
    evidence = (
        SimpleNamespace(
            evidence_sha256=canonical_sha256(["attention", layer_index]),
            actual_child_replay=SimpleNamespace(
                captured_child_model_state_sha256=state
            ),
        ),
    )
    recipe = stack.resolve_lfm2_semantic_body_recipe("fit_guided")
    body = {
        **_body_recipe_fields(),
        "recipe_component_lane": recipe.component_lanes["qk"],
        "status": status,
    }
    receipt = {**body, "result_sha256": canonical_sha256(body)}

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="obsolete metric-based incumbent no-op",
    ):
        stack._install_attention_operator(
            target_model=target,
            layer_index=layer_index,
            evidence=evidence,
            cache_key="a" * 64,
            bridge_artifact_sha256="b" * 64,
            installer=lambda **_kwargs: receipt,
            parameter_count=stack._parameter_count(target),
        )


def _invoke_short_conv_noop_receipt(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str,
) -> MappingProxyType | dict[str, object]:
    target = _FourteenLayerReceiptStateTarget()
    state = _state_identity(target)
    monkeypatch.setattr(
        stack,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_state_identity(model)),
    )
    evidence = (
        SimpleNamespace(
            captured_child_model_state_sha256=state,
            evidence_sha256="d" * 64,
        ),
    )
    cache = SimpleNamespace(cache_key="a" * 64, parent_fingerprint="c" * 64)
    bridge = SimpleNamespace(artifact_sha256="b" * 64)
    parameter_count = stack._parameter_count(target)
    recipe = stack.resolve_lfm2_semantic_body_recipe("fit_guided")
    body: dict[str, object] = {
        **_body_recipe_fields(recipe.recipe_id),
        "recipe_component_lane": recipe.component_lanes["short_conv"],
        "schema": stack._SHORT_CONV_INSTALLATION_SCHEMA,
        "status": status,
        "installed": False,
        "candidate_frozen_before_validation": True,
        "validation_triggered_refit": False,
        "model_state_before_sha256": state,
        "model_state_after_sha256": state,
        "changed_state_keys": [],
        "parameter_count_changed": False,
        "topology_changed": False,
        "parameter_identities_preserved": True,
        "parameter_storage_preserved": True,
        "gradient_state_preserved": True,
        "exact_noop_state_proven": True,
        "analytic_cached_evidence_only": True,
        "operation_audit": dict(stack._ZERO_OPERATIONS),
        "layer_index": 0,
        "cache_key": cache.cache_key,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "source_model_identity_sha256": cache.parent_fingerprint,
        "transport_evidence_sha256": [evidence[0].evidence_sha256],
        "parameter_element_count": parameter_count,
    }
    if status == "heldout-selected-incumbent-noop":
        body.update(
            {
                "incumbent_selected_by_heldout_gates": True,
                "fit_check_nonregression_passed": True,
                "selection_nonregression_passed": True,
                "validation_nonregression_passed": True,
            }
        )
    elif status == "deterministic-fit-correction-rounded-to-incumbent-noop":
        body.update(
            {
                "incumbent_selected_by_heldout_gates": False,
                "heldout_metrics_are_diagnostic_only": True,
                "numeric_metric_used_for_publication_or_rollback": False,
            }
        )
    elif status == stack._SHORT_CONV_STRUCTURAL_NOOP_STATUS:
        body.update(
            {
                "selected_candidate": "incumbent-noop",
                "selected_candidate_state_identical_to_incumbent": True,
                "incumbent_noop_fallback_used": True,
                "incumbent_noop_fallback_reason": (
                    "all-analytic-proposals-structurally-invalid-or-identical"
                ),
                "incumbent_selected_by_heldout_gates": False,
                "structurally_eligible_candidate_count": 0,
                "candidate_eligibility_basis": (
                    "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
                ),
                "structural_safety_is_only_candidate_veto": True,
                "fit_metrics_used_for_candidate_ordering": True,
                "fit_metrics_used_for_structural_eligibility": False,
                "heldout_metrics_are_diagnostic_only": True,
                "validation_publication_veto_applied": False,
                "rollback_authorized": False,
                "numeric_metrics_have_final_intelligence_authority": False,
                "validation_rejected_candidate": None,
            }
        )
    else:  # pragma: no cover - constrained by focused callers below
        raise AssertionError(f"unsupported test status: {status}")
    receipt = {**body, "result_sha256": canonical_sha256(body)}

    return stack._install_short_conv_operator(
        target_model=target,
        layer_index=0,
        evidence=evidence,
        source_fir_weight=torch.ones((2, 1, 2), dtype=torch.bfloat16),
        source_fir_bias=None,
        cache=cache,
        bridge=bridge,
        installer=lambda **_kwargs: receipt,
        parameter_count=parameter_count,
    )


@pytest.mark.parametrize(
    "status",
    (
        "heldout-selected-incumbent-noop",
        "deterministic-fit-correction-rounded-to-incumbent-noop",
    ),
)
def test_short_conv_install_rejects_legacy_noop_reasons(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="incomplete or over-broad receipt",
    ):
        _invoke_short_conv_noop_receipt(monkeypatch, status=status)


def test_short_conv_install_accepts_structural_no_candidate_noop_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _invoke_short_conv_noop_receipt(
        monkeypatch,
        status=stack._SHORT_CONV_STRUCTURAL_NOOP_STATUS,
    )

    assert receipt["status"] == stack._SHORT_CONV_STRUCTURAL_NOOP_STATUS
    assert receipt["incumbent_noop_fallback_used"] is True
    assert receipt["structurally_eligible_candidate_count"] == 0


def test_stack_never_implements_its_own_cuda_bf16_router() -> None:
    source = stack.__file__
    text = open(source, encoding="utf-8").read()
    assert "torch.cuda.is_bf16_supported" not in text
    assert "cuda.get_device_capability" not in text


def _patch_coupled_global_closure_transaction(
    monkeypatch: pytest.MonkeyPatch,
    *,
    closure_raises: bool = False,
):
    target, cache, bridge, _events = _patch_complete_transaction(
        monkeypatch,
        use_live_fourteen_layer_state_chain=True,
    )
    target.model.layers[0].conv.out_proj = torch.nn.Module()
    target.model.layers[0].conv.out_proj.weight = torch.nn.Parameter(
        torch.ones((1, 1), dtype=torch.float64)
    )
    closure_writer = target.model.layers[0].conv.out_proj.weight
    physical_site_id = "model.layers.0.conv.out_proj.weight"
    monkeypatch.setattr(
        stack,
        "_global_live_physical_state",
        lambda _target: MappingProxyType({physical_site_id: closure_writer}),
    )
    recipe = stack.resolve_lfm2_semantic_body_recipe(
        "nonlinear_qdeim_coupled_attention_structural"
    )
    prior = Lfm2CpcfTerminalActionDownWeightPrior(
        down_weight=torch.ones((1, 1), dtype=torch.bfloat16),
        body_recipe_id=recipe.recipe_id,
        body_recipe_sha256=recipe.recipe_sha256,
        mlp_lane=recipe.component_lanes["mlp"],
        selected_role="full-w1-w3-w2",
        selected_candidate_sha256="a" * 64,
        preparation_sha256="b" * 64,
        mlp_installation_core_sha256="c" * 64,
        live_deferred_w2_sha256=(
            stack.lfm2_cpcf_terminal_action_tensor_sha256(target.final_w2)
        ),
    )

    stack._DirectParentSafetensorsSession.norm_epsilon = 1e-5
    original_chart = stack._fit_residual_chart

    def chart_with_fit_batches(**kwargs):
        chart, receipt = original_chart(**kwargs)
        body = dict(receipt)
        body.pop("result_sha256")
        body["partition_batch_indices"] = {
            "fit": [0],
            "fit_check": [1],
            "selection": [2],
            "validation": [3],
        }
        return chart, {**body, "result_sha256": canonical_sha256(body)}

    monkeypatch.setattr(stack, "_fit_residual_chart", chart_with_fit_batches)
    monkeypatch.setattr(
        stack,
        "_fit_layer_normalized_operator_input_decoder",
        lambda **_kwargs: (
            torch.eye(2, dtype=torch.float64),
            {"result_sha256": "d" * 64},
        ),
    )
    monkeypatch.setattr(
        stack,
        "seal_lfm2_semantic_attention_group_qdeim_norm_inventory",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        stack,
        "_require_attention_body_recipe_binding",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        stack,
        "_authenticate_lfm2_semantic_attention_operator_receipt",
        lambda *_args, **_kwargs: {},
    )

    def coupled_installer(**kwargs):
        receipt = stack._install_attention_operator(
            target_model=kwargs["target_model"],
            layer_index=kwargs["layer_index"],
            body_recipe=kwargs["body_recipe"],
        )
        fold_compute_device = kwargs.get("fold_compute_device", "cpu")
        compute_execution = _qdeim_compute_execution(
            schema="neural-forge-lfm2-attention-qdeim-compute-v1",
            requested_device=fold_compute_device,
        )
        body = {key: value for key, value in receipt.items() if key != "result_sha256"}
        body.update(
            {
                "fold_compute_device": fold_compute_device,
                "qdeim_pod_and_pivot_compute_device": fold_compute_device,
                "qdeim_compute_execution": compute_execution,
                "qdeim_compute_execution_sha256": compute_execution["receipt_sha256"],
                "requested_fold_compute_device_proven": True,
            }
        )
        return stack._sealed(body)

    monkeypatch.setattr(
        stack,
        "_install_coupled_attention_group_qdeim_operator",
        coupled_installer,
    )
    original_mlp_installer = stack._install_mlp

    def mlp_with_prior(**kwargs):
        receipt, evidence, _unused_prior = original_mlp_installer(**kwargs)
        return (
            receipt,
            evidence,
            prior if kwargs["layer_index"] == 13 else None,
        )

    monkeypatch.setattr(stack, "_install_mlp", mlp_with_prior)
    row_map = stack._sealed(
        {
            "schema": stack.LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
            "kind": "test-global-row-map",
        }
    )
    monkeypatch.setattr(
        stack,
        "_prepare_global_residual_accumulators",
        lambda _alignments: (MappingProxyType({}), row_map),
    )
    monkeypatch.setattr(
        stack,
        "_capture_global_parent_layer_boundaries",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        stack,
        "_capture_global_child_layer_primals",
        lambda *_args, **_kwargs: None,
    )
    closure_calls = []

    def closure(_accumulators, **_kwargs):
        closure_calls.append(
            (
                _kwargs["mapped_parent_tied_table"].detach().clone(),
                _kwargs["requested_basis_rank"],
            )
        )
        before = stack._tensor_sha256(closure_writer)
        with torch.no_grad():
            closure_writer.add_(0.5)
        if closure_raises:
            raise RuntimeError("injected global closure failure")
        after = stack._tensor_sha256(closure_writer)
        scale_policy = stack.Lfm2ProducerAwareCorrectionScalePolicy(
            materialization_frontier=tuple(_kwargs["correction_scale_frontier"]),
            materialization_scale=_kwargs["correction_scale"],
        )
        closure_policy = _kwargs["closure_policy"]
        sites = [
            {
                "site_id": f"test.layer.{index // 4}.site.{index % 4}",
                "layer_index": index // 4,
                "causal_order": 6 * (index // 4) + (0, 1, 3, 4)[index % 4],
                "writer_kind": "test",
                "input_width": 1,
            }
            for index in range(56)
        ]
        topology_body = {
            "schema": "test",
            "kind": "writer_topology",
            "sites": sites,
        }
        topology = {
            **topology_body,
            "topology_sha256": canonical_sha256(topology_body),
        }
        materialization_device = _kwargs["materialization_compute_device"]
        compute_body = {
            "schema": ("neural-forge-lfm2-producer-aware-dense-absorption-compute-v1"),
            "kind": "producer_aware_dense_absorption_compute_execution",
            "requested_device": materialization_device,
            "resolved_device": materialization_device,
            "backend": materialization_device.partition(":")[0],
        }
        compute_receipt = {
            **compute_body,
            "report_sha256": canonical_sha256(compute_body),
        }
        representability_frontier = [
            {
                "level": _kwargs["target_dtype_representability_level"],
                "ray_multiplier": 1.0,
                "selected_for_this_transaction": True,
            }
        ]
        report_body = {
            "evidence_package_sha256": "e" * 64,
            "closure_result_sha256": "f" * 64,
            "closure_scope": "56_producer_aware_input_output_sites_v2a",
            "closure_backend": "producer_aware_v2a",
            "global_residual_backend": closure_policy.backend,
            "closure_policy": closure_policy.as_dict(),
            "closure_policy_sha256": closure_policy.authentication_sha256,
            "layer_count": 14,
            "writer_site_count": 56,
            "four_sites_per_layer": True,
            "dynamic_layer_count_derived_from_authenticated_evidence": True,
            "site_specifications": sites,
            "topology_receipt": topology,
            "topology_sha256": topology["topology_sha256"],
            "materialization_compute_device": materialization_device,
            "materialization_compute_execution_receipt_sha256": compute_receipt[
                "report_sha256"
            ],
            "materialization_compute_execution_receipt": compute_receipt,
            "dense_transaction_receipt": {
                "compute_execution_receipt_sha256": compute_receipt["report_sha256"],
                "compute_execution_receipt": compute_receipt,
            },
            "target_dtype_representability_level": _kwargs[
                "target_dtype_representability_level"
            ],
            "target_dtype_representability_ray_multiplier": 1.0,
            "target_dtype_representability_frontier": representability_frontier,
            "installed_parameter_ids": [physical_site_id],
            "installed_parameter_count": 1,
            "all_nondeferred_physical_parameters_installed": True,
            "dense_live_physical_sha256_before": {physical_site_id: before},
            "dense_live_physical_sha256_after": {physical_site_id: after},
            "correction_scale_policy": {
                **scale_policy.authentication_body(),
                "authentication_sha256": scale_policy.authentication_sha256,
            },
            "correction_scale_policy_sha256": scale_policy.authentication_sha256,
            "materialization_frontier": list(scale_policy.materialization_frontier),
            "materialization_scale": scale_policy.materialization_scale,
            "materialization_scale_is_execution_variant_not_winner": True,
            "frontier_member_selected_as_winner": False,
            "separate_fold_decode_retain_required_for_each_frontier_scale": True,
            "all_frontier_candidates_eligible_for_decoded_behavioral_review": True,
            "numeric_or_manual_quality_gate": False,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "final_w2_parameter_id": ("model.layers.13.feed_forward.w2.weight"),
            "final_w2_unchanged": True,
            "final_layer_mlp_down_weight_prior_authentication_sha256": (
                prior.authentication_sha256
            ),
            "final_layer_mlp_down_weight_prior_tensor_sha256": (
                stack.lfm2_cpcf_terminal_action_tensor_sha256(prior.down_weight)
            ),
            "final_layer_mlp_down_weight_prior_install_count": 0,
            "corrected_final_w2_closure_lineage_sha256": "1" * 64,
            "final_layer_mlp_down_weight_prior_closure_lineage_sha256": "1" * 64,
            "quality_gate": False,
            "numeric_diagnostics_used_for_auto_veto": False,
        }
        report = MappingProxyType(
            {**report_body, "report_sha256": canonical_sha256(report_body)}
        )
        installation = SimpleNamespace(
            corrected_final_w2_prior=prior,
            report=report,
            verify_authentication=lambda: True,
        )
        return installation, SimpleNamespace(evidence_sha256="e" * 64)

    monkeypatch.setattr(
        stack,
        "_compile_install_global_residual_from_accumulators",
        closure,
    )
    return target, cache, bridge, recipe, closure_writer, closure_calls


@pytest.mark.parametrize("fold_compute_device", ["cpu", "cuda:1"])
def test_coupled_stack_invokes_global_closure_and_returns_corrected_prior(
    monkeypatch: pytest.MonkeyPatch,
    fold_compute_device: str,
) -> None:
    target, cache, bridge, recipe, closure_writer, closure_calls = (
        _patch_coupled_global_closure_transaction(monkeypatch)
    )
    before_final_w2 = target.final_w2.detach().clone()
    before_writer = closure_writer.detach().clone()
    child_tied_table = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float64,
    )
    parent_coordinate_table = torch.ones((2, 3), dtype=torch.float64)

    result = stack.compile_install_lfm2_semantic_layer_stack_postfold(
        cache=cache,
        bridge=bridge,
        target_model=target,
        interface=SimpleNamespace(
            producer_encoder=torch.ones((3, 2), dtype=torch.float64),
            target_weight=child_tied_table,
        ),
        compiler_lineage={
            index: SimpleNamespace(source_inventory=object())
            for index in stack._EXPECTED_ATTENTION_INDICES
        },
        parent_model_directory="unused",
        donor_tokenizer=object(),
        target_tokenizer=object(),
        attention_installer=None,
        fold_compute_device=fold_compute_device,
        global_residual_basis_rank=2,
        body_recipe=recipe,
        mapped_parent_tied_table=parent_coordinate_table,
    )

    assert len(closure_calls) == 1
    torch.testing.assert_close(closure_calls[0][0], child_tied_table, rtol=0, atol=0)
    assert closure_calls[0][1] == 2
    assert result.receipt["global_residual_requested_basis_rank"] == 2
    assert result.receipt["global_residual_tied_table_source"] == (
        "authenticated_live_interface_target_weight_child_coordinates"
    )
    assert result.receipt["global_residual_closure_applied"] is True
    assert result.receipt["global_residual_closure_numeric_quality_gate"] is False
    assert result.receipt["global_residual_closure_manual_quality_gate"] is False
    assert result.receipt["fold_compute_device"] == fold_compute_device
    assert (
        result.receipt[
            "all_mlp_qdeim_pod_and_pivot_layers_proved_requested_fold_compute_device"
        ]
        is True
    )
    assert (
        result.receipt[
            "all_attention_qdeim_pod_and_pivot_layers_proved_requested_fold_compute_device"
        ]
        is True
    )
    assert all(
        layer["mlp_qdeim_pod_and_pivot_compute_device"] == fold_compute_device
        and layer["mlp_normalized_decoder_control_compute_device"] == "cpu"
        for layer in result.receipt["layer_installations"]
    )
    assert all(
        layer["mlp_weight_signature_compute_execution_sha256"] is None
        for layer in result.receipt["layer_installations"]
    )
    assert all(
        layer["attention_qdeim_pod_and_pivot_compute_device"] == fold_compute_device
        for layer in result.receipt["layer_installations"]
        if layer["operator_kind"] == "full_attention"
    )
    assert result.final_layer_mlp_down_weight_prior is not None
    assert not torch.equal(closure_writer, before_writer)
    torch.testing.assert_close(target.final_w2, before_final_w2, rtol=0, atol=0)


def test_global_closure_failure_rolls_back_local_and_closure_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, cache, bridge, recipe, _closure_writer, closure_calls = (
        _patch_coupled_global_closure_transaction(
            monkeypatch,
            closure_raises=True,
        )
    )
    before = {
        name: value.detach().clone() for name, value in target.state_dict().items()
    }
    child_tied_table = torch.eye(2, dtype=torch.float64)

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="exact entry bytes were restored",
    ):
        stack.compile_install_lfm2_semantic_layer_stack_postfold(
            cache=cache,
            bridge=bridge,
            target_model=target,
            interface=SimpleNamespace(
                producer_encoder=torch.ones((3, 2), dtype=torch.float64),
                target_weight=child_tied_table,
            ),
            compiler_lineage={
                index: SimpleNamespace(source_inventory=object())
                for index in stack._EXPECTED_ATTENTION_INDICES
            },
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            attention_installer=None,
            body_recipe=recipe,
            mapped_parent_tied_table=torch.ones((2, 3), dtype=torch.float64),
        )

    assert len(closure_calls) == 1
    torch.testing.assert_close(closure_calls[0][0], child_tied_table, rtol=0, atol=0)
    assert closure_calls[0][1] is None
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
