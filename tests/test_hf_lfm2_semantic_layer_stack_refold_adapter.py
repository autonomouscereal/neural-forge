from __future__ import annotations

import hashlib
import json
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (  # noqa: E402
    Lfm2CpcfTerminalActionEvidence,
)
from neural_forge.hf_lfm2_semantic_mlp_refold_adapter import (  # noqa: E402
    Lfm2DirectLocalMlpSourceInventory,
    Lfm2SemanticMlpEndpointEvidence,
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


def _mlp_functional_provenance_receipt(layer_index: int) -> dict[str, object]:
    policy_body = {
        "schema": "neural-forge-bf16-cross-kernel-equivalence-policy-v1",
        "authentication_only": True,
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
    body = {
        "schema": "neural-forge-lfm2-direct-parent-mlp-functional-provenance-v2",
        "layer_index": layer_index,
        "cache_key": "4" * 64,
        "cache_parent_fingerprint": "5" * 64,
        "cache_manifest_sha256": "6" * 64,
        "parent_checkpoint_manifest_sha256": "7" * 64,
        "parent_session_open_receipt_sha256": "8" * 64,
        "parent_config_file_sha256": "9" * 64,
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
            name: {"field": name, "accepted": True} for name in endpoint_names
        },
        "semantic_endpoint_rows": 1,
        "all_cached_endpoints_bit_exact": True,
        "all_cached_endpoints_within_authenticated_bf16_envelope": True,
        "cached_mlp_input_is_normalized_evidence_source_of_truth": True,
        "reconstructed_mlp_input_is_authentication_check_only": True,
        "direct_parent_tensors_numerically_consistent_on_cached_endpoints": True,
        "semantic_endpoint_scope_is_complete_tensor_identity_proof": False,
        "cryptographic_parent_tensor_identity_proven_by_this_receipt": False,
        "parent_checkpoint_files_hashed_by_session": True,
        "direct_tensor_digests_recorded": True,
        "expected_per_tensor_digest_comparison_performed": False,
        "numerical_equivalence_is_provenance_only": True,
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
        lambda *_args: {
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
    ):
        before = _state_identity(model)
        events.append(f"{layer_index}:install:{kind}")
        installed = layer_index not in noop_operator_layers
        if installed:
            with torch.no_grad():
                model.mutable.add_(1.0)
        if failing_layer == layer_index:
            raise RuntimeError("injected partial operator failure")
        after = _state_identity(model)
        receipt = {
            "status": (
                "installed-analytic-candidate"
                if installed
                else "heldout-selected-incumbent-noop"
            ),
            "installed": installed,
            "install_count": int(installed),
            "model_state_before_sha256": before,
            "model_state_after_sha256": after,
            "result_sha256": canonical_sha256(["operator", layer_index, after]),
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
        return receipt

    monkeypatch.setattr(
        stack,
        "_install_attention_operator",
        lambda target_model, layer_index, **_kwargs: mutate_operator(
            target_model, layer_index, "full_attention"
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
        before = _state_identity(target_model)
        events.append(f"{layer_index}:install:mlp")
        installed = layer_index < 13 and layer_index not in noop_mlp_layers
        if installed:
            with torch.no_grad():
                target_model.mutable.add_(1.0)
        after = _state_identity(target_model)
        fold_compute_device = _kwargs.get("fold_compute_device", "cpu")
        source_derived_planting = {
            "fold_compute_device": fold_compute_device,
            "weight_complete_signature_compute_execution": (
                None
                if fold_compute_device == "cpu"
                else {
                    "backend": "cuda",
                    "requested_device": fold_compute_device,
                    "compute_dtype": "float64",
                    "execution_sha256": "9" * 64,
                }
            ),
        }
        receipt_body = {
            "status": (
                "installed-frozen-semantic-mlp-candidate"
                if installed
                else "heldout-selected-incumbent-noop"
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
        return receipt, (_terminal_banks() if layer_index == 13 else {})

    monkeypatch.setattr(stack, "_install_mlp", install_mlp)

    def next_bank(bank):
        events.append(f"{bank.layer_index}:next")
        return MappingProxyType({0: f"after:{bank.layer_index}"})

    monkeypatch.setattr(stack, "_next_residual_bank", next_bank)
    target = _TinyTarget()
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


def test_residual_chart_preserves_unsupported_subspace_with_diagnostic_metrics() -> None:
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
        execution_device="cpu",
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

    class ParentSession:
        manifest_sha256 = "c" * 64
        open_receipt_sha256 = "d" * 64
        config_file_sha256 = "e" * 64
        norm_epsilon = epsilon

        def tensor(self, key):
            return tensors[key].detach().cpu().contiguous().clone()

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.ones((1,), dtype=torch.bfloat16))

    target = SimpleNamespace(
        model=SimpleNamespace(layers=(Layer(),)),
    )
    cache = SimpleNamespace(
        cache_key="f" * 64,
        parent_fingerprint="0" * 64,
        layer_topology_fingerprint="1" * 64,
        manifest={"manifest_sha256": "2" * 64},
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
    assert receipt["all_cached_endpoints_within_authenticated_bf16_envelope"] is True
    assert receipt["numerical_equivalence_is_provenance_only"] is True
    assert (
        receipt["cross_kernel_equivalence_policy"][
            "fold_selection_or_quality_objective"
        ]
        is False
    )
    assert (
        receipt["direct_parent_tensors_numerically_consistent_on_cached_endpoints"]
        is True
    )
    assert receipt["semantic_endpoint_scope_is_complete_tensor_identity_proof"] is False
    assert (
        receipt["cryptographic_parent_tensor_identity_proven_by_this_receipt"] is False
    )
    assert receipt["expected_per_tensor_digest_comparison_performed"] is False
    assert receipt["semantic_endpoint_rows"] == 1
    assert receipt["endpoint_equivalence_metrics"]["mlp.input"]["accepted"] is True
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
    assert receipt["reconstructed_mlp_input_is_authentication_check_only"] is True
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


def test_real_v2_parent_mlp_provenance_reaches_layer0_receipt() -> None:
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
            "status": "installed-analytic-candidate",
            "installed": True,
        }
    )
    producer_coverage = stack._sealed({"all_required_roles_structurally_nonzero": True})
    mlp_receipt = stack._sealed(
        {
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
        "neural-forge-lfm2-direct-parent-mlp-functional-provenance-v2"
    )
    assert (
        receipt["mlp_source_direct_tensors_numerically_consistent_on_cached_endpoints"]
        is True
    )
    assert (
        receipt["mlp_source_all_cached_endpoints_within_authenticated_bf16_envelope"]
        is True
    )
    assert (
        receipt["mlp_source_semantic_endpoint_scope_is_complete_tensor_identity_proof"]
        is False
    )
    assert (
        receipt[
            "mlp_source_cryptographic_parent_tensor_identity_proven_by_this_receipt"
        ]
        is False
    )
    assert "mlp_source_direct_tensors_functionally_bound_to_cache" not in receipt
    assert stack._verify_sealed(receipt, "layer 0") == dict(receipt)


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
    assert receipt["all_cached_endpoints_within_authenticated_bf16_envelope"] is True
    assert receipt["endpoint_equivalence_metrics"]["mlp.gate"]["accepted"] is True


def test_direct_parent_mlp_inventory_rejects_self_labeled_wrong_w2() -> None:
    target, cache, bridge, session, alignments, parent, chart, chart_receipt = (
        _parent_mlp_provenance_fixture(corrupt_down=True)
    )

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="outside the authenticated BF16 cross-kernel numerical envelope",
    ):
        stack._direct_mlp_inventory(
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


def test_heldout_selected_incumbent_noop_contract_is_strict() -> None:
    state = "4" * 64
    receipt = {
        "status": "heldout-selected-incumbent-noop",
        "installed": False,
        "incumbent_selected_by_heldout_gates": True,
        "fit_check_nonregression_passed": True,
        "selection_nonregression_passed": True,
        "validation_nonregression_passed": True,
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
    }

    assert stack._authenticated_heldout_incumbent_noop(
        receipt,
        before_sha256=state,
        after_sha256=state,
        changed_state_keys=(),
    )
    assert stack._authenticated_heldout_incumbent_noop(
        {
            **receipt,
            "operation_audit": {
                **receipt["operation_audit"],
                "parameters_sparsified": 0,
                "parameter_aliases_created": 0,
            },
        },
        before_sha256=state,
        after_sha256=state,
        changed_state_keys=(),
    )
    assert not stack._authenticated_heldout_incumbent_noop(
        {**receipt, "validation_nonregression_passed": False},
        before_sha256=state,
        after_sha256=state,
        changed_state_keys=(),
    )
    assert not stack._authenticated_heldout_incumbent_noop(
        {
            **receipt,
            "operation_audit": {
                **receipt["operation_audit"],
                "parameters_sparsified": 1,
            },
        },
        before_sha256=state,
        after_sha256=state,
        changed_state_keys=(),
    )


def test_stack_never_implements_its_own_cuda_bf16_router() -> None:
    source = stack.__file__
    text = open(source, encoding="utf-8").read()
    assert "torch.cuda.is_bf16_supported" not in text
    assert "cuda.get_device_capability" not in text
