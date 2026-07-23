from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_compiler import DynamicFoldingCompiler
from neural_forge.dynamic_fold_executor import FoldDecision, fingerprint_model_state
from neural_forge.dynamic_lfm2_adapter import (
    HYBRID_MODEL_METHOD,
    DynamicLfm2AdapterError,
    Lfm2PreparedHybridProducerEvidence,
    Lfm2TargetSynthesisAdapter,
    build_lfm2_dynamic_compiler_bridge,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import default_inspector_registry
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesizer,
)
from neural_forge.hf_lfm2_canonical import inspect_lfm2_layer
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_closed_loop_fold import Lfm2ClosedLoopFoldConfig
from neural_forge.hf_lfm2_hybrid_producer_first_model_fold import (
    Lfm2HybridProducerFirstFoldError,
    install_lfm2_hybrid_producer_first_model_fold,
)
from neural_forge.hf_lfm2_model_fold import (
    Lfm2ModelFoldConfig,
    Lfm2ModelFoldEvidenceError,
    Lfm2ModelFoldInvariantError,
    Lfm2ModelFoldSpan,
    fingerprint_lfm2_model_fold_tensor,
    fingerprint_lfm2_model_feature_state,
)
from neural_forge.hf_lfm2_producer_first_model_fold import (
    Lfm2ProducerFirstCapture,
    Lfm2ProducerFirstEvidence,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _model(
    seed: int,
    *,
    layer_types: tuple[str, ...],
    intermediate_size: int,
    query_heads: int,
    key_value_heads: int,
):
    torch.manual_seed(seed)
    config = transformers.Lfm2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=intermediate_size,
        num_hidden_layers=len(layer_types),
        num_attention_heads=query_heads,
        num_key_value_heads=key_value_heads,
        head_dim=2,
        layer_types=list(layer_types),
        conv_L_cache=3,
        conv_bias=True,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    return transformers.Lfm2ForCausalLM(config).eval()


class _Rows:
    def __init__(
        self,
        *,
        layer_kinds: tuple[str, ...],
        layer_fields: dict[int, dict[str, torch.Tensor]],
        cache_key: str,
        trace_rows_sha256: str,
    ) -> None:
        self.paths = tuple(f"row-{index}" for index in range(12))
        self.path_splits = (
            *("selection" for _ in range(4)),
            *("fit" for _ in range(4)),
            *("validation" for _ in range(4)),
        )
        self.layer_kinds = layer_kinds
        self.cache_key = cache_key
        self.corpus_fingerprint = "b" * 64
        self.trace_rows_sha256 = trace_rows_sha256
        self.layer_fields = MappingProxyType(
            {
                index: MappingProxyType(dict(values))
                for index, values in layer_fields.items()
            }
        )

    def field(self, layer_index: int, name: str) -> torch.Tensor:
        return self.layer_fields[layer_index][name]


def _parent_rows(parent) -> _Rows:
    generator = torch.Generator().manual_seed(103)
    row_count = 12
    hidden = 8
    source_neurons = 6
    residual_pre = torch.zeros(row_count, hidden, dtype=torch.float64)
    fields: dict[int, dict[str, torch.Tensor]] = {}
    for layer_index, layer in enumerate(parent.model.layers):
        contract = inspect_lfm2_layer(layer)
        post_operator = residual_pre + torch.randn(
            row_count,
            hidden,
            generator=generator,
            dtype=torch.float64,
        )
        residual_post = post_operator + torch.randn(
            row_count,
            hidden,
            generator=generator,
            dtype=torch.float64,
        )
        gate = torch.randn(
            row_count,
            source_neurons,
            generator=generator,
            dtype=torch.float64,
        )
        up = torch.randn(
            row_count,
            source_neurons,
            generator=generator,
            dtype=torch.float64,
        )
        values = {
            "residual.pre": residual_pre,
            "residual.post_operator": post_operator,
            "residual.post_layer": residual_post,
            "mlp.gate": gate,
            "mlp.up": up,
            "mlp.post_swiglu": torch.nn.functional.silu(gate) * up,
        }
        if contract.kind == "full_attention":
            head_dim = int(parent.config.head_dim)
            query_heads = int(contract.operator.query_heads)
            key_value_heads = int(contract.operator.key_value_heads)
            epsilon = float(parent.config.norm_eps)
            operator_input = residual_pre / torch.sqrt(
                residual_pre.square().mean(dim=-1, keepdim=True) + epsilon
            )
            operator_input = operator_input * layer.operator_norm.weight.detach().to(
                torch.float64
            )
            query_linear = torch.nn.functional.linear(
                operator_input,
                layer.self_attn.q_proj.weight.detach().to(torch.float64),
            ).reshape(row_count, query_heads, head_dim)
            key_linear = torch.nn.functional.linear(
                operator_input,
                layer.self_attn.k_proj.weight.detach().to(torch.float64),
            ).reshape(row_count, key_value_heads, head_dim)
            query_pre_rope = query_linear / torch.sqrt(
                query_linear.square().mean(dim=-1, keepdim=True) + epsilon
            )
            query_pre_rope = (
                query_pre_rope
                * layer.self_attn.q_layernorm.weight.detach().to(torch.float64)
            )
            key_pre_rope = key_linear / torch.sqrt(
                key_linear.square().mean(dim=-1, keepdim=True) + epsilon
            )
            key_pre_rope = (
                key_pre_rope
                * layer.self_attn.k_layernorm.weight.detach().to(torch.float64)
            )
            values.update(
                {
                    "attention.q": torch.randn(
                        row_count,
                        4,
                        2,
                        generator=generator,
                        dtype=torch.float64,
                    ),
                    "attention.k": torch.randn(
                        row_count,
                        2,
                        2,
                        generator=generator,
                        dtype=torch.float64,
                    ),
                    "attention.v": torch.randn(
                        row_count,
                        2,
                        2,
                        generator=generator,
                        dtype=torch.float64,
                    ),
                    "attention.context": torch.randn(
                        row_count,
                        4,
                        2,
                        generator=generator,
                        dtype=torch.float64,
                    ),
                    "attention.q_linear": query_linear,
                    "attention.k_linear": key_linear,
                    "attention.q_normalized_pre_rope": query_pre_rope,
                    "attention.k_normalized_pre_rope": key_pre_rope,
                }
            )
        fields[layer_index] = values
        residual_pre = residual_post
    return _Rows(
        layer_kinds=tuple(
            inspect_lfm2_layer(layer).kind for layer in parent.model.layers
        ),
        layer_fields=fields,
        cache_key="a" * 64,
        trace_rows_sha256="c" * 64,
    )


def _child_rows(child) -> _Rows:
    generator = torch.Generator().manual_seed(104)
    row_count = 12
    hidden = 8
    target_neurons = 4
    fields: dict[int, dict[str, torch.Tensor]] = {}
    for layer_index, layer in enumerate(child.model.layers):
        contract = inspect_lfm2_layer(layer)
        values = {
            "residual.pre": torch.randn(
                row_count,
                hidden,
                generator=generator,
                dtype=torch.float64,
            ),
            "residual.post_operator": torch.randn(
                row_count,
                hidden,
                generator=generator,
                dtype=torch.float64,
            ),
            "mlp.post_swiglu": torch.randn(
                row_count,
                target_neurons,
                generator=generator,
                dtype=torch.float64,
            ),
        }
        if contract.kind == "full_attention":
            values["attention.context"] = torch.randn(
                row_count,
                2,
                2,
                generator=generator,
                dtype=torch.float64,
            )
        fields[layer_index] = values
    return _Rows(
        layer_kinds=tuple(
            inspect_lfm2_layer(layer).kind for layer in child.model.layers
        ),
        layer_fields=fields,
        cache_key="d" * 64,
        trace_rows_sha256="e" * 64,
    )


def _state(model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }


def _assert_state_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> None:
    assert set(left) == set(right)
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _unique_parameters(model) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            total += int(parameter.numel())
    return total


def _fixture(
    *,
    failing_capture: bool = False,
    misaligned_capture: bool = False,
    misbound_capture: bool = False,
    dtype=None,
):
    parent = _model(
        100,
        layer_types=("conv", "full_attention", "conv"),
        intermediate_size=6,
        query_heads=4,
        key_value_heads=2,
    )
    child = _model(
        101,
        layer_types=("full_attention", "conv"),
        intermediate_size=4,
        query_heads=2,
        key_value_heads=1,
    )
    if dtype is not None:
        parent = parent.to(dtype=dtype)
        child = child.to(dtype=dtype)
    parent_rows = _parent_rows(parent)
    captured_rows = _child_rows(child)
    if misaligned_capture:
        captured_rows.corpus_fingerprint = "f" * 64
    blank_state = fingerprint_lfm2_model_feature_state(child)
    captured_states: list[str] = []

    def capture(model):
        state = fingerprint_lfm2_model_feature_state(model)
        captured_states.append(state)
        assert state != blank_state
        assert getattr(
            model,
            "_neural_forge_producer_plan_fingerprint",
            None,
        )
        if failing_capture:
            raise Lfm2HybridProducerFirstFoldError("synthetic capture failure")
        return Lfm2ProducerFirstCapture(
            rows=captured_rows,
            cache_key=("f" * 64 if misbound_capture else captured_rows.cache_key),
            child_forward_count=1,
        )

    evidence = Lfm2ProducerFirstEvidence(
        parent_rows=parent_rows,
        selection_indices=(0, 1, 2, 3),
        fit_indices=(4, 5, 6, 7),
        validation_indices=(8, 9, 10, 11),
        capture_child=capture,
        parent_cache_fingerprint=parent_rows.cache_key,
        parent_identity_sha256=(fingerprint_lfm2_parent(parent).identity_sha256),
        capture_builder_id="test.hybrid-producer-first.v1",
    )
    spans = (
        Lfm2ModelFoldSpan((0, 1), 0),
        Lfm2ModelFoldSpan((2,), 1),
    )
    config = Lfm2ModelFoldConfig(
        layer_config=Lfm2ClosedLoopFoldConfig(
            endpoint_ridge=1e-8,
            max_endpoint_relative_rmse=1e9,
            min_endpoint_cosine_similarity=-1.0,
            max_endpoint_prior_delta_ratio=1e12,
        ),
        max_span_correction_anchors=4,
        max_span_endpoint_prior_delta_ratio=1e12,
    )
    return parent, child, spans, evidence, config, captured_states


def test_hybrid_depth_fold_captures_only_complete_producer_and_rolls_back_blank():
    parent, child, spans, evidence, config, captured_states = _fixture()
    parent_before = _state(parent)
    blank_before = _state(child)
    blank_fingerprint = fingerprint_lfm2_model_feature_state(child)

    transaction = install_lfm2_hybrid_producer_first_model_fold(
        parent,
        child,
        spans,
        evidence,
        config=config,
    )

    assert len(captured_states) == 1
    assert captured_states[0] == transaction.plan.producer_state_sha256
    assert captured_states[0] != blank_fingerprint
    assert transaction.plan.bindings[0].anchor_source_layer_index == 1
    assert transaction.plan.bindings[1].anchor_source_layer_index == 2
    assert transaction.plan.model_plan.parameters_removed > 0
    assert transaction.plan.child_forward_count == 1
    assert transaction.plan.as_dict()["blank_features_used"] is False
    _assert_state_equal(parent_before, _state(parent))

    report = transaction.rollback()
    assert report["rollback_exact"] is True
    assert report["restored_state_sha256"] == blank_fingerprint
    _assert_state_equal(blank_before, _state(child))
    assert not hasattr(child, "_neural_forge_producer_plan_fingerprint")


def test_active_bf16_depth_overlay_leaves_singleton_candidate_unchanged():
    parent, child, spans, evidence, config, _captured_states = _fixture(
        dtype=torch.bfloat16
    )
    blank_before = _state(child)

    transaction = install_lfm2_hybrid_producer_first_model_fold(
        parent,
        child,
        spans,
        evidence,
        config=config,
    )

    plan = transaction.plan.model_plan
    assert tuple(index for index, _receipt in plan.span_coupled_overlay_receipts) == (
        0,
    )
    assert plan.span_coupled_overlay_receipts[0][1]["installed_any"] is True
    singleton = plan.span_plans[1]
    assert singleton.span.source_layer_indices == (2,)
    assert singleton.span.is_depth_fold is False
    expected = singleton.layer_plan.candidate_state_dict()
    observed = child.model.layers[1].state_dict()
    assert set(expected) == set(observed)
    for name in expected:
        assert torch.equal(expected[name], observed[name].detach().cpu()), name

    report = transaction.rollback()
    assert report["rollback_exact"] is True
    _assert_state_equal(blank_before, _state(child))


@pytest.mark.parametrize(
    "heterogeneous_norms",
    (False, True),
    ids=("shared-qk-norm-installed", "heterogeneous-qk-norm-closed-form-refit"),
)
def test_hybrid_bf16_two_attention_layers_invoke_span_compilers_atomically(
    heterogeneous_norms,
    monkeypatch,
):
    import neural_forge.hf_lfm2_attention_group_qdeim as attention_qdeim

    parent = _model(
        120,
        layer_types=("full_attention", "full_attention"),
        intermediate_size=6,
        query_heads=4,
        key_value_heads=2,
    ).to(dtype=torch.bfloat16)
    child = _model(
        121,
        layer_types=("full_attention",),
        intermediate_size=4,
        query_heads=4,
        key_value_heads=2,
    ).to(dtype=torch.bfloat16)
    with torch.no_grad():
        if heterogeneous_norms:
            parent.model.layers[0].self_attn.q_layernorm.weight.fill_(1.0)
            parent.model.layers[0].self_attn.k_layernorm.weight.fill_(1.0)
            parent.model.layers[1].self_attn.q_layernorm.weight.fill_(2.0)
            parent.model.layers[1].self_attn.k_layernorm.weight.fill_(3.0)
        else:
            parent.model.layers[1].self_attn.q_layernorm.weight.copy_(
                parent.model.layers[0].self_attn.q_layernorm.weight
            )
            parent.model.layers[1].self_attn.k_layernorm.weight.copy_(
                parent.model.layers[0].self_attn.k_layernorm.weight
            )
    parent_rows = _parent_rows(parent)
    generator = torch.Generator().manual_seed(122)
    captured_rows = _Rows(
        layer_kinds=("full_attention",),
        layer_fields={
            0: {
                "residual.pre": torch.randn(
                    12, 8, generator=generator, dtype=torch.float64
                ),
                "residual.post_operator": torch.randn(
                    12, 8, generator=generator, dtype=torch.float64
                ),
                "mlp.post_swiglu": torch.randn(
                    12, 4, generator=generator, dtype=torch.float64
                ),
                "attention.context": torch.randn(
                    12, 4, 2, generator=generator, dtype=torch.float64
                ),
            }
        },
        cache_key="d" * 64,
        trace_rows_sha256="e" * 64,
    )
    blank_state = _state(child)
    blank_sha256 = fingerprint_lfm2_model_feature_state(child)
    parent_state = _state(parent)
    capture_count = 0

    def capture(model):
        nonlocal capture_count
        capture_count += 1
        assert fingerprint_lfm2_model_feature_state(model) != blank_sha256
        return Lfm2ProducerFirstCapture(
            rows=captured_rows,
            cache_key=captured_rows.cache_key,
            child_forward_count=1,
        )

    evidence = Lfm2ProducerFirstEvidence(
        parent_rows=parent_rows,
        selection_indices=(0, 1, 2, 3),
        fit_indices=(4, 5, 6, 7),
        validation_indices=(8, 9, 10, 11),
        capture_child=capture,
        parent_cache_fingerprint=parent_rows.cache_key,
        parent_identity_sha256=fingerprint_lfm2_parent(parent).identity_sha256,
        capture_builder_id="test.span-coupled-production.v1",
    )
    config = Lfm2ModelFoldConfig(
        layer_config=Lfm2ClosedLoopFoldConfig(
            endpoint_ridge=0.0 if heterogeneous_norms else 1e-8,
            max_endpoint_relative_rmse=0.0,
            min_endpoint_cosine_similarity=1.0,
            max_endpoint_prior_delta_ratio=1e12,
        ),
        max_span_correction_anchors=4,
        max_span_endpoint_prior_delta_ratio=1e12,
    )
    original_attention_compiler = (
        attention_qdeim.compile_lfm2_attention_group_span_qdeim
    )
    compiled_attention = {}

    def capture_compiled_attention(*args, **kwargs):
        result = original_attention_compiler(*args, **kwargs)
        compiled_attention["result"] = result
        return result

    monkeypatch.setattr(
        attention_qdeim,
        "compile_lfm2_attention_group_span_qdeim",
        capture_compiled_attention,
    )
    transaction = install_lfm2_hybrid_producer_first_model_fold(
        parent,
        child,
        (Lfm2ModelFoldSpan((0, 1), 0),),
        evidence,
        config=config,
    )

    assert capture_count == 1
    plan = transaction.plan.model_plan
    assert len(plan.span_coupled_overlay_receipts) == 1
    receipt = dict(plan.span_coupled_overlay_receipts[0][1])
    assert receipt["installed"] is True
    assert receipt["installed_any"] is True
    assert receipt["installed_all"] is True
    assert receipt["mlp"]["installed"] is True
    assert receipt["attention"]["installed"] is True
    assert receipt["attention"]["shared_qk_norm_directly_representable"] is (
        not heterogeneous_norms
    )
    assert receipt["attention"]["heterogeneous_norm_rows_averaged"] is False
    assert ("shared_qk_norm_refit" in receipt["attention"]) is heterogeneous_norms
    endpoint = plan.span_plans[0].layer_plan.mlp.endpoint_fit
    assert endpoint.gate.accepted is False
    assert endpoint.accepted is True
    assert endpoint.endpoint_metric_policy == "diagnostic_only"
    assert {
        item["source_layer_index"]
        for item in receipt["mlp"]["compiler"]["writer_transport"][
            "source_contributions"
        ]
    } == {0, 1}
    assert {
        item["source_layer_index"]
        for item in receipt["attention"]["compiler"]["writer_transport"][
            "source_contributions"
        ]
    } == {0, 1}
    assert all(
        item["writer_contribution_frobenius_norm"] > 0.0
        for branch in (receipt["mlp"], receipt["attention"])
        for item in branch["compiler"]["writer_transport"]["source_contributions"]
    )
    assert plan.as_dict()["span_coupled_dense_state_installed"] is True
    assert plan.as_dict()["span_coupled_all_requested_operators_installed"] is True
    span_attention = compiled_attention["result"]
    span_state = span_attention.state_dict()
    installed = child.model.layers[0].state_dict()
    for name in ("v_proj.weight", "out_proj.weight"):
        assert torch.equal(
            installed[f"self_attn.{name}"].detach().cpu(),
            span_state[name],
        )
    if heterogeneous_norms:
        import neural_forge.hf_lfm2_model_fold as model_fold

        legacy = plan.span_plans[0].layer_plan.candidate_state_dict()
        refit = receipt["attention"]["shared_qk_norm_refit"]
        assert receipt["schema"].endswith("production-receipt-v2")
        assert refit["heterogeneous_source_norms_averaged"] is False
        assert refit["numeric_diagnostics_have_install_authority"] is False
        assert refit["rowspace_ridge"] == 0.0
        assert refit["zero_ridge_uses_minimum_frobenius_norm_pseudoinverse"] is True
        assert (
            refit["query_refit"]["analytic_precast_rowspace_solver"]["method"]
            == "zero-ridge-moore-penrose-minimum-frobenius-norm"
        )
        for name in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.q_layernorm.weight",
            "self_attn.k_layernorm.weight",
        ):
            assert not torch.equal(installed[name].detach().cpu(), legacy[name]), name
        for name in ("v_proj.weight", "out_proj.weight"):
            assert (
                fingerprint_lfm2_model_fold_tensor(
                    installed[f"self_attn.{name}"].detach().cpu()
                )
                == refit["unchanged_state_sha256"][name]
            )
        tampered_receipt = plan.as_dict()["span_coupled_overlay_receipts"]["0"]
        tampered_refit = tampered_receipt["attention"]["shared_qk_norm_refit"]
        tampered_refit["installed_attention_state_sha256"]["q_proj.weight"] = "0" * 64
        tampered_refit["refit_state_sha256"]["q_proj.weight"] = "0" * 64
        tampered_refit.pop("result_sha256")
        tampered_refit["result_sha256"] = canonical_sha256(tampered_refit)
        tampered_receipt.pop("receipt_sha256")
        tampered_receipt["receipt_sha256"] = canonical_sha256(tampered_receipt)
        tampered_plan = replace(
            plan,
            span_coupled_overlay_receipts=((0, MappingProxyType(tampered_receipt)),),
        )
        with pytest.raises(
            Lfm2ModelFoldInvariantError,
            match="failed nested authentication",
        ):
            model_fold._validate_span_coupled_overlay_plan(
                tampered_plan,
                plan.candidate_state_dict(),
            )
    else:
        assert receipt["schema"].endswith("production-receipt-v1")
        for name in ("q_proj.weight", "k_proj.weight"):
            assert torch.equal(
                installed[f"self_attn.{name}"].detach().cpu(),
                span_state[name],
            )
        assert torch.equal(
            installed["self_attn.q_layernorm.weight"].detach().cpu(),
            span_attention.selected_query_norm_weight_rows[0],
        )
        assert torch.equal(
            installed["self_attn.k_layernorm.weight"].detach().cpu(),
            span_attention.selected_key_norm_weight_rows[0],
        )
    _assert_state_equal(parent_state, _state(parent))

    report = transaction.rollback()
    assert report["rollback_exact"] is True
    _assert_state_equal(blank_state, _state(child))


def test_hybrid_capture_failure_restores_compiler_zero_child_exactly():
    parent, child, spans, evidence, config, captured_states = _fixture(
        failing_capture=True
    )
    blank_before = _state(child)

    with pytest.raises(
        Lfm2HybridProducerFirstFoldError,
        match="synthetic capture failure",
    ):
        install_lfm2_hybrid_producer_first_model_fold(
            parent,
            child,
            spans,
            evidence,
            config=config,
        )

    assert len(captured_states) == 1
    _assert_state_equal(blank_before, _state(child))
    assert not hasattr(child, "_neural_forge_producer_plan_fingerprint")


def test_hybrid_late_evidence_failure_restores_compiler_zero_child_exactly():
    parent, child, spans, evidence, config, captured_states = _fixture(
        misaligned_capture=True
    )
    blank_before = _state(child)

    with pytest.raises(
        Lfm2ModelFoldEvidenceError,
        match="not exactly aligned",
    ):
        install_lfm2_hybrid_producer_first_model_fold(
            parent,
            child,
            spans,
            evidence,
            config=config,
        )

    assert len(captured_states) == 1
    _assert_state_equal(blank_before, _state(child))
    assert not hasattr(child, "_neural_forge_producer_plan_fingerprint")


def test_hybrid_capture_cache_key_mismatch_restores_blank_exactly():
    parent, child, spans, evidence, config, captured_states = _fixture(
        misbound_capture=True
    )
    blank_before = _state(child)

    with pytest.raises(
        Lfm2HybridProducerFirstFoldError,
        match="not bound to their declared cache key",
    ):
        install_lfm2_hybrid_producer_first_model_fold(
            parent,
            child,
            spans,
            evidence,
            config=config,
        )

    assert len(captured_states) == 1
    _assert_state_equal(blank_before, _state(child))
    assert not hasattr(child, "_neural_forge_producer_plan_fingerprint")


@pytest.mark.parametrize(
    "tamper_kind",
    ("unrelated-provenance", "retained-base-lineage"),
)
def test_self_resealed_unrelated_span_provenance_fails_and_restores_blank(
    monkeypatch,
    tamper_kind,
):
    import neural_forge.hf_lfm2_hybrid_producer_first_model_fold as hybrid_fold

    parent, child, spans, evidence, config, _captured_states = _fixture(
        dtype=torch.bfloat16
    )
    parent_before = _state(parent)
    blank_before = _state(child)
    original_overlay = hybrid_fold.overlay_lfm2_span_coupled_candidates

    def unrelated_overlay(*args, **kwargs):
        plan = original_overlay(*args, **kwargs)
        if tamper_kind == "retained-base-lineage":
            changed = replace(
                plan.span_plans[0],
                cached_span_pre_sha256="f" * 64,
            )
            return replace(
                plan,
                span_plans=(changed, *plan.span_plans[1:]),
            )
        provenance = plan.as_dict()["span_coupled_provenance_receipt"]
        assert isinstance(provenance, dict)
        provenance.pop("provenance_root_sha256")
        provenance["parent_cache_fingerprint"] = "f" * 64
        provenance["provenance_root_sha256"] = canonical_sha256(provenance)
        return replace(
            plan,
            span_coupled_provenance_receipt=MappingProxyType(provenance),
        )

    monkeypatch.setattr(
        hybrid_fold,
        "overlay_lfm2_span_coupled_candidates",
        unrelated_overlay,
    )
    with pytest.raises(
        Lfm2ModelFoldInvariantError,
        match=(
            "not bound to this cache, child, and receipts"
            if tamper_kind == "unrelated-provenance"
            else "retained base model-fold plan"
        ),
    ):
        install_lfm2_hybrid_producer_first_model_fold(
            parent,
            child,
            spans,
            evidence,
            config=config,
        )

    _assert_state_equal(parent_before, _state(parent))
    _assert_state_equal(blank_before, _state(child))
    assert not hasattr(child, "_neural_forge_producer_plan_fingerprint")


class _TopologyFinalGate:
    def __init__(
        self,
        *,
        reject: bool,
        parent_depth: int = 3,
        child_depth: int = 2,
    ) -> None:
        self.reject = reject
        self.parent_depth = parent_depth
        self.child_depth = child_depth
        self.calls = 0

    def evaluate(self, context):
        self.calls += 1
        assert (
            len(context.execution_context.parent_graphs["parent"].layer_indices)
            == self.parent_depth
        )
        assert (
            len(context.execution_context.child_graph.layer_indices) == self.child_depth
        )
        if self.reject:
            return FoldDecision.reject("synthetic mapped trajectory rejection")
        return FoldDecision.accept("test final gate")


@pytest.mark.parametrize(
    "reject_final",
    (False, True),
    ids=("commit", "final-reject-exact-rollback"),
)
def test_dynamic_operator_invokes_both_span_compilers_then_rolls_back(
    monkeypatch,
    reject_final,
):
    import neural_forge.hf_lfm2_attention_group_qdeim as attention_qdeim
    import neural_forge.hf_lfm2_mlp_qdeim as mlp_qdeim

    parent = _model(
        130,
        layer_types=("full_attention", "full_attention"),
        intermediate_size=6,
        query_heads=4,
        key_value_heads=2,
    ).to(dtype=torch.bfloat16)
    with torch.no_grad():
        parent.model.layers[1].self_attn.q_layernorm.weight.copy_(
            parent.model.layers[0].self_attn.q_layernorm.weight
        )
        parent.model.layers[1].self_attn.k_layernorm.weight.copy_(
            parent.model.layers[0].self_attn.k_layernorm.weight
        )
    parent_state = _state(parent)
    tokenizer_fingerprint = canonical_sha256(
        {"tokenizer": "dynamic-span-coupled-production"}
    )
    cache_fingerprint = "a" * 64
    exact_target = _unique_parameters(
        _model(
            131,
            layer_types=("full_attention",),
            intermediate_size=4,
            query_heads=4,
            key_value_heads=2,
        )
    )
    target_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(4, 2, 4),
        evidence_fingerprint=cache_fingerprint,
    )
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((target_adapter,)),
    ).synthesize(
        request=DynamicTargetRequest(
            child_name="span-coupled-child",
            target_family="lfm2",
            target_modalities=("text",),
            exact_unique_parameters=exact_target,
            tokenizer_fingerprint=tokenizer_fingerprint,
            adapter_id=target_adapter.adapter_id,
            fold_method_id=HYBRID_MODEL_METHOD,
            blank_seed=19,
            configuration=(
                ("source_parent", "parent"),
                ("target_depth", 1),
                ("target_layer_types", "full_attention"),
                ("ordered_source_spans", "0,1"),
            ),
        ),
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer_fingerprint},
    )
    child = target.model
    assert all(parameter.dtype == torch.bfloat16 for parameter in child.parameters())
    blank = fingerprint_model_state(child)
    blank_state = _state(child)
    parent_rows = _parent_rows(parent)
    generator = torch.Generator().manual_seed(132)
    captured_rows = _Rows(
        layer_kinds=("full_attention",),
        layer_fields={
            0: {
                "residual.pre": torch.randn(
                    12, 8, generator=generator, dtype=torch.float64
                ),
                "residual.post_operator": torch.randn(
                    12, 8, generator=generator, dtype=torch.float64
                ),
                "mlp.post_swiglu": torch.randn(
                    12, 4, generator=generator, dtype=torch.float64
                ),
                "attention.context": torch.randn(
                    12, 4, 2, generator=generator, dtype=torch.float64
                ),
            }
        },
        cache_key="d" * 64,
        trace_rows_sha256="e" * 64,
    )

    evidence = Lfm2ProducerFirstEvidence(
        parent_rows=parent_rows,
        selection_indices=(0, 1, 2, 3),
        fit_indices=(4, 5, 6, 7),
        validation_indices=(8, 9, 10, 11),
        capture_child=lambda _model: Lfm2ProducerFirstCapture(
            rows=captured_rows,
            cache_key=captured_rows.cache_key,
            child_forward_count=1,
        ),
        parent_cache_fingerprint=cache_fingerprint,
        parent_identity_sha256=fingerprint_lfm2_parent(parent).identity_sha256,
        capture_builder_id="test.dynamic-span-coupled-production.v1",
    )
    compiler_reports: dict[str, dict] = {}
    original_mlp_compiler = mlp_qdeim.compile_lfm2_mlp_span_coupled_nonlinear_qdeim
    original_attention_compiler = (
        attention_qdeim.compile_lfm2_attention_group_span_qdeim
    )

    def compile_mlp(*args, **kwargs):
        result = original_mlp_compiler(*args, **kwargs)
        compiler_reports["mlp"] = result.as_dict()
        return result

    def compile_attention(*args, **kwargs):
        result = original_attention_compiler(*args, **kwargs)
        compiler_reports["attention"] = result.as_dict()
        return result

    monkeypatch.setattr(
        mlp_qdeim,
        "compile_lfm2_mlp_span_coupled_nonlinear_qdeim",
        compile_mlp,
    )
    monkeypatch.setattr(
        attention_qdeim,
        "compile_lfm2_attention_group_span_qdeim",
        compile_attention,
    )
    model_config = Lfm2ModelFoldConfig(
        layer_config=Lfm2ClosedLoopFoldConfig(
            endpoint_ridge=1e-8,
            max_endpoint_relative_rmse=1e9,
            min_endpoint_cosine_similarity=-1.0,
            max_endpoint_prior_delta_ratio=1e12,
        ),
        max_span_correction_anchors=4,
        max_span_endpoint_prior_delta_ratio=1e12,
    )
    bridge = build_lfm2_dynamic_compiler_bridge(
        parent_cache_fingerprint=cache_fingerprint,
        parent_cache_payload={"sealed": "parent-cache"},
        evidence_builder=lambda _request: None,
        model_evidence_builder=lambda _request: Lfm2PreparedHybridProducerEvidence(
            evidence=evidence
        ),
        model_fold_config=model_config,
        selected_target=target.adapter_payload,
        final_evaluator=(
            final_gate := _TopologyFinalGate(
                reject=reject_final,
                parent_depth=2,
                child_depth=1,
            )
        ),
        blank_state_fingerprint=blank,
        parent_cache_integrity_check=lambda _payload: cache_fingerprint,
    )
    compilation = DynamicFoldingCompiler(operators=bridge.operators).compile(
        parent_models={"parent": parent},
        child_model=child,
        child_name="span-coupled-child",
        evidence_provider=bridge.evidence_provider,
        evaluator=bridge.evaluator,
        final_evaluator=bridge.final_evaluator,
        evidence_fingerprint=cache_fingerprint,
        parent_tokenizer_fingerprints={"parent": tokenizer_fingerprint},
        child_tokenizer_fingerprint=tokenizer_fingerprint,
        exact_target_unique_parameters=exact_target,
        initializer=bridge.initializer,
    )

    assert final_gate.calls == 1
    assert set(compiler_reports) == {"mlp", "attention"}
    for report in compiler_reports.values():
        contributions = report["writer_transport"]["source_contributions"]
        assert {item["source_layer_index"] for item in contributions} == {0, 1}
        assert all(
            item["writer_contribution_frobenius_norm"] > 0.0 for item in contributions
        )
    assert compilation.run.observed_parent_module_forward_calls == 0
    assert compilation.run.observed_parent_root_forward_calls == 0
    if reject_final:
        assert compilation.complete is False
        assert compilation.run.status == "final_gate_rejected"
        assert fingerprint_model_state(child) == blank
        _assert_state_equal(blank_state, _state(child))
    else:
        assert compilation.complete is True
        assert fingerprint_model_state(child) != blank
        assert all(
            value.layout == torch.strided and not value.is_sparse
            for value in child.state_dict().values()
        )
    _assert_state_equal(parent_state, _state(parent))


@pytest.mark.parametrize(
    "reject_final",
    (False, True),
    ids=("commit", "final-reject-exact-rollback"),
)
def test_dynamic_compiler_runs_hybrid_parent_only_provider_to_one_commit(
    reject_final,
):
    parent = _model(
        110,
        layer_types=("conv", "full_attention", "conv"),
        intermediate_size=6,
        query_heads=4,
        key_value_heads=2,
    )
    tokenizer_fingerprint = canonical_sha256(
        {"tokenizer": "hybrid-producer-first-integration"}
    )
    cache_fingerprint = "a" * 64
    exact_target = _unique_parameters(
        _model(
            111,
            layer_types=("full_attention", "conv"),
            intermediate_size=4,
            query_heads=2,
            key_value_heads=1,
        )
    )
    target_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(2, 1, 4),
        evidence_fingerprint=cache_fingerprint,
    )
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((target_adapter,)),
    ).synthesize(
        request=DynamicTargetRequest(
            child_name="hybrid-child",
            target_family="lfm2",
            target_modalities=("text",),
            exact_unique_parameters=exact_target,
            tokenizer_fingerprint=tokenizer_fingerprint,
            adapter_id=target_adapter.adapter_id,
            fold_method_id=HYBRID_MODEL_METHOD,
            blank_seed=17,
            configuration=(
                ("source_parent", "parent"),
                ("target_depth", 2),
                ("target_layer_types", "full_attention,conv"),
                ("ordered_source_spans", "0,1;2"),
            ),
        ),
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer_fingerprint},
    )
    child = target.model
    blank = fingerprint_model_state(child)
    blank_state = _state(child)
    blank_parameter_ids = tuple(id(parameter) for parameter in child.parameters())
    parent_rows = _parent_rows(parent)
    captured_rows = _child_rows(child)
    captured_states: list[str] = []

    def capture(model):
        state = fingerprint_lfm2_model_feature_state(model)
        captured_states.append(state)
        assert fingerprint_model_state(model) != blank
        return Lfm2ProducerFirstCapture(
            rows=captured_rows,
            cache_key=captured_rows.cache_key,
            child_forward_count=1,
        )

    evidence = Lfm2ProducerFirstEvidence(
        parent_rows=parent_rows,
        selection_indices=(0, 1, 2, 3),
        fit_indices=(4, 5, 6, 7),
        validation_indices=(8, 9, 10, 11),
        capture_child=capture,
        parent_cache_fingerprint=cache_fingerprint,
        parent_identity_sha256=(fingerprint_lfm2_parent(parent).identity_sha256),
        capture_builder_id="test.dynamic-hybrid-provider.v1",
    )
    builder_calls = 0

    def model_builder(request):
        nonlocal builder_calls
        builder_calls += 1
        assert request.child_state_fingerprint == blank
        assert fingerprint_model_state(request.child_model) == blank
        return Lfm2PreparedHybridProducerEvidence(evidence=evidence)

    model_config = Lfm2ModelFoldConfig(
        layer_config=Lfm2ClosedLoopFoldConfig(
            endpoint_ridge=1e-8,
            max_endpoint_relative_rmse=1e9,
            min_endpoint_cosine_similarity=-1.0,
            max_endpoint_prior_delta_ratio=1e12,
        ),
        max_span_correction_anchors=4,
        max_span_endpoint_prior_delta_ratio=1e12,
    )
    bridge = build_lfm2_dynamic_compiler_bridge(
        parent_cache_fingerprint=cache_fingerprint,
        parent_cache_payload={"sealed": "parent-cache"},
        evidence_builder=lambda _request: None,
        model_evidence_builder=model_builder,
        model_fold_config=model_config,
        selected_target=target.adapter_payload,
        final_evaluator=(final_gate := _TopologyFinalGate(reject=reject_final)),
        blank_state_fingerprint=blank,
        parent_cache_integrity_check=lambda _payload: cache_fingerprint,
    )
    compilation = DynamicFoldingCompiler(
        operators=bridge.operators,
    ).compile(
        parent_models={"parent": parent},
        child_model=child,
        child_name="hybrid-child",
        evidence_provider=bridge.evidence_provider,
        evaluator=bridge.evaluator,
        final_evaluator=bridge.final_evaluator,
        evidence_fingerprint=cache_fingerprint,
        parent_tokenizer_fingerprints={
            "parent": tokenizer_fingerprint,
        },
        child_tokenizer_fingerprint=tokenizer_fingerprint,
        exact_target_unique_parameters=exact_target,
        initializer=bridge.initializer,
    )

    assert builder_calls == 1
    assert len(captured_states) == 1
    assert bridge.evidence_provider.build_count == 1
    assert final_gate.calls == 1
    assert compilation.run.observed_parent_module_forward_calls == 0
    assert compilation.run.observed_parent_root_forward_calls == 0
    if reject_final:
        assert compilation.complete is False
        assert compilation.run.status == "final_gate_rejected"
        assert compilation.run.accepted_action_ids == ()
        assert compilation.run.completed_target_parameter_paths == ()
        assert fingerprint_model_state(child) == blank
        _assert_state_equal(blank_state, _state(child))
        assert tuple(id(parameter) for parameter in child.parameters()) == (
            blank_parameter_ids
        )
        assert not hasattr(child, "_neural_forge_producer_plan_fingerprint")
        setattr(
            child,
            "_neural_forge_producer_plan_fingerprint",
            "synthetic-unauthenticated-marker",
        )
        with pytest.raises(
            DynamicLfm2AdapterError,
            match="not authenticated by its parent-only cache",
        ):
            bridge.evidence_provider.restore(
                SimpleNamespace(
                    child_model=child,
                    child_state_fingerprint=blank,
                ),
                blank,
                ("unknown-provisional-action",),
            )
        assert hasattr(child, "_neural_forge_producer_plan_fingerprint")
        delattr(child, "_neural_forge_producer_plan_fingerprint")
    else:
        assert compilation.complete is True
        assert fingerprint_model_state(child) != blank
