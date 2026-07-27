from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_post_causal_o_global_reclosure as post  # noqa: E402
import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402


_KINDS = ("full_attention", "short_conv", "full_attention")


def _sealed(
    body: dict[str, Any],
    *,
    field: str = "result_sha256",
) -> MappingProxyType:
    return MappingProxyType({**body, field: canonical_sha256(body)})


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.k_proj = torch.nn.Linear(4, 4, bias=False)
        self.v_proj = torch.nn.Linear(4, 4, bias=False)
        self.out_proj = torch.nn.Linear(4, 4, bias=False)


class _Conv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj = torch.nn.Linear(4, 4, bias=False)
        self.out_proj = torch.nn.Linear(4, 4, bias=False)


class _Mlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(4, 4, bias=False)
        self.w3 = torch.nn.Linear(4, 4, bias=False)
        self.w2 = torch.nn.Linear(4, 4, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.is_attention_layer = kind == "full_attention"
        if self.is_attention_layer:
            self.self_attn = _Attention()
        else:
            self.conv = _Conv()
        self.feed_forward = _Mlp()


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(kind) for kind in _KINDS])
        self.embedding_norm = torch.nn.LayerNorm(4)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()
        self.embedding = torch.nn.Embedding(11, 4)
        self.output = torch.nn.Linear(4, 11, bias=False)
        with torch.no_grad():
            self.output.weight.copy_(self.embedding.weight)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.output

    def forward(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("The post-causal-O stage must not call model.forward().")


class _Session:
    def __init__(self) -> None:
        self.verified = False

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def verify(self) -> None:
        self.verified = True

    def as_dict(self) -> MappingProxyType:
        return MappingProxyType({"session_sha256": "9" * 64})


class _Cache:
    cache_key = "c" * 64
    parent_fingerprint = "d" * 64
    layer_topology_fingerprint = "e" * 64
    batch_count = 4

    def __init__(self) -> None:
        self.session = _Session()

    def selective_authenticated_session(self) -> _Session:
        return self.session


class _Bridge:
    artifact_sha256 = "b" * 64


@dataclass
class _Prior:
    down_weight: Any
    live_deferred_w2_sha256: str
    authentication_sha256: str = "a" * 64

    def verify_authentication(self) -> bool:
        return True


@dataclass
class _Rebase:
    terminal_prior: _Prior
    receipt: MappingProxyType

    def verify(self) -> bool:
        return True


@dataclass
class _Causal:
    receipt: MappingProxyType

    def verify(self) -> bool:
        return True


@dataclass
class _Evidence:
    evidence_sha256: str = "7" * 64

    def verify_authentication(self) -> bool:
        return True


@dataclass
class _Installation:
    corrected_final_w2_prior: _Prior
    report: MappingProxyType

    def verify_authentication(self) -> bool:
        return True


@dataclass
class _Replay:
    captured_child_model_state_sha256: str
    layer_index: int
    pre_residual_by_batch: MappingProxyType
    bank_sha256: str


@dataclass
class _Parent:
    evidence_sha256: str


def _identity(model: _Model) -> str:
    return canonical_sha256(dict(post._state_tensor_hashes(model)))


def _writer_parameters(model: _Model) -> MappingProxyType:
    values: dict[str, Any] = {}
    for index, layer in enumerate(model.model.layers):
        if layer.is_attention_layer:
            values[f"model.layers.{index}.self_attn.out_proj.weight"] = (
                layer.self_attn.out_proj.weight
            )
        else:
            values[f"model.layers.{index}.conv.out_proj.weight"] = (
                layer.conv.out_proj.weight
            )
        if index < len(model.model.layers) - 1:
            values[f"model.layers.{index}.feed_forward.w2.weight"] = (
                layer.feed_forward.w2.weight
            )
    return MappingProxyType(values)


def _install_harness(
    monkeypatch: pytest.MonkeyPatch,
    model: _Model,
    *,
    closure_mode: str = "valid",
) -> tuple[dict[str, Any], list[tuple[Any, ...]]]:
    cache = _Cache()
    bridge = _Bridge()
    monkeypatch.setattr(post, "Lfm2CircuitTraceCache", _Cache)
    monkeypatch.setattr(post, "StrictTokenizerBridgeArtifact", _Bridge)
    monkeypatch.setattr(post, "Lfm2CausalPostQkOutputFoldInstallation", _Causal)
    monkeypatch.setattr(post, "Lfm2PostfoldTerminalW2Rebase", _Rebase)
    monkeypatch.setattr(post, "Lfm2CpcfTerminalActionDownWeightPrior", _Prior)
    monkeypatch.setattr(post, "Lfm2SemanticGlobalResidualEvidencePackage", _Evidence)
    monkeypatch.setattr(post, "Lfm2SemanticGlobalResidualInstallation", _Installation)
    def identity(value: _Model) -> SimpleNamespace:
        return SimpleNamespace(identity_sha256=_identity(value))

    monkeypatch.setattr(post, "fingerprint_lfm2_parent", identity)
    monkeypatch.setattr(stack, "fingerprint_lfm2_parent", identity)

    encoder = torch.eye(4, dtype=torch.float64)
    tied = torch.arange(44, dtype=torch.float64).reshape(11, 4).div(43.0)
    live_target = model.get_input_embeddings().weight.detach().to(
        device="cpu"
    ).contiguous()
    interface_body = {
        "schema": "test-interface-v1",
        "producer_encoder_sha256": post._interface_tensor_sha256(encoder),
        "target_weight_sha256": post._interface_tensor_sha256(live_target),
    }
    interface = _sealed(interface_body, field="report_sha256")
    archive_body = {
        "schema": "test-reclosure-seed-archive-v1",
        "foundation_active_interface_report_sha256": interface["report_sha256"],
        "mapped_parent_tied_table_foundation_sha256": (
            post._foundation_tensor_sha256(tied)
        ),
        "producer_encoder_foundation_sha256": (
            post._foundation_tensor_sha256(encoder)
        ),
        "tensor_records": {
            "mapped_parent_tied_table": {
                "sha256": post.lfm2_post_causal_o_global_coordinate_sha256(tied)
            },
            "producer_encoder": {
                "sha256": post.lfm2_post_causal_o_global_coordinate_sha256(encoder)
            },
        },
    }
    archive_manifest_sha256 = canonical_sha256(archive_body)
    archive_manifest = MappingProxyType(
        {
            **archive_body,
            "manifest_sha256": archive_manifest_sha256,
            "fresh_reload_verified": True,
        }
    )
    lineage = {
        "interface_report_sha256": interface["report_sha256"],
        "producer_encoder_sha256": interface["producer_encoder_sha256"],
        "target_weight_sha256": interface["target_weight_sha256"],
        "producer_encoder_taken_from_authenticated_compiler_lineage": True,
        "global_residual_gauge_required": False,
        "target_weight_matches_live_input_child": True,
        "interface_runtime_mode": "independent_output",
        "live_output_parameter_independent": True,
        "output_table_matches_structural_input_at_body_entry": True,
    }
    stack_body = {
        "schema": post.LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
        "cache_key": cache.cache_key,
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "cache_layer_topology_sha256": cache.layer_topology_fingerprint,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "layer_count": len(_KINDS),
        "terminal_layer_index": len(_KINDS) - 1,
        "operator_layer_kinds": list(_KINDS),
        "attention_layer_indices": [0, 2],
        "short_conv_layer_indices": [1],
        "complete_causal_layer_stack_adjudicated": True,
        "final_layer_mlp_w2_deferred_to_fused_terminal": True,
        "final_layer_mlp_w2_install_count": 0,
        "body_global_coordinate_completion_enabled": True,
        "tied_interface_lineage": lineage,
    }
    stack_receipt = _sealed(stack_body)
    causal_body = {
        "schema": post.LFM2_CAUSAL_POST_QK_OUTPUT_FOLD_SCHEMA,
        "semantic_layer_stack_result_sha256": stack_receipt["result_sha256"],
        "cache_key": cache.cache_key,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "model_state_after_sha256": _identity(model),
        "all_attention_o_decisions_completed": True,
        "qkv_byte_identical_before_and_after": True,
        "causal_downstream_replay_after_each_o_install": True,
    }
    causal = _Causal(_sealed(causal_body))
    final_name = "model.layers.2.feed_forward.w2.weight"
    final_w2 = dict(model.named_parameters())[final_name]
    prior = _Prior(
        down_weight=final_w2.detach().clone(),
        live_deferred_w2_sha256=(
            post.lfm2_cpcf_terminal_action_tensor_sha256(final_w2)
        ),
    )
    rebase_body = {
        "schema": post.LFM2_POSTFOLD_TERMINAL_W2_REBASE_SCHEMA,
        "semantic_layer_stack_result_sha256": stack_receipt["result_sha256"],
        "terminal_prior_authentication_sha256": prior.authentication_sha256,
        "terminal_prior_tensor_sha256": (
            post.lfm2_cpcf_terminal_action_tensor_sha256(prior.down_weight)
        ),
        "restored_live_w2_sha256": prior.live_deferred_w2_sha256,
        "final_w2_parameter_name": final_name,
    }
    rebase = _Rebase(prior, _sealed(rebase_body))
    alignments = tuple(SimpleNamespace(source_batch_index=index) for index in range(4))
    monkeypatch.setattr(
        post,
        "_semantic_alignments_from_session",
        lambda **_kwargs: (
            alignments,
            MappingProxyType({"receipt_sha256": "8" * 64}),
        ),
    )
    monkeypatch.setattr(
        post,
        "audit_lfm2_semantic_span_cpcf_refinement",
        lambda *_args, **_kwargs: {
            "probability_preserving_attention_transport_go": True,
            "report_sha256": "6" * 64,
        },
    )
    accumulators = {index: SimpleNamespace() for index in range(4)}
    monkeypatch.setattr(
        post,
        "_prepare_global_residual_accumulators",
        lambda _alignments: (
            accumulators,
            _sealed({"schema": "test-row-maps-v1"}),
        ),
    )
    monkeypatch.setattr(
        post,
        "_initial_residual_bank",
        lambda *_args: MappingProxyType(
            {index: torch.full((1, 1, 4), float(index)) for index in range(4)}
        ),
    )
    calls: list[tuple[Any, ...]] = []

    def replay(
        target: _Model,
        *,
        alignments: Any,
        layer_index: int,
        pre_residual_by_batch: MappingProxyType,
    ) -> _Replay:
        del alignments
        calls.append(
            (
                "replay",
                layer_index,
                tuple(float(value[0, 0, 0]) for value in pre_residual_by_batch.values()),
            )
        )
        return _Replay(
            captured_child_model_state_sha256=_identity(target),
            layer_index=layer_index,
            pre_residual_by_batch=MappingProxyType(dict(pre_residual_by_batch)),
            bank_sha256=canonical_sha256({"replay": layer_index}),
        )

    monkeypatch.setattr(post, "_replay_current_layer", replay)

    def next_bank(replay_bank: _Replay) -> MappingProxyType:
        calls.append(("next", replay_bank.layer_index))
        return MappingProxyType(
            {
                index: value + 1.0
                for index, value in replay_bank.pre_residual_by_batch.items()
            }
        )

    monkeypatch.setattr(post, "_next_residual_bank", next_bank)

    def load_parent(
        _session: _Session,
        *,
        alignments: Any,
        layer_index: int,
        operator_kind: str,
        include_final_norm: bool,
    ) -> MappingProxyType:
        del alignments
        calls.append(("parent", layer_index, operator_kind, include_final_norm))
        return MappingProxyType(
            {
                index: _Parent(canonical_sha256({"parent": layer_index, "batch": index}))
                for index in range(4)
            }
        )

    monkeypatch.setattr(post, "_load_parent_layer_trajectories", load_parent)

    def fit_chart(**kwargs: Any) -> tuple[Any, MappingProxyType]:
        layer_index = kwargs["layer_index"]
        assert kwargs["producer_encoder"] is encoder
        assert kwargs["global_gauge_required"] is True
        calls.append(("chart", layer_index))
        return torch.eye(4, dtype=torch.float64), _sealed(
            {"schema": "test-chart-v1", "layer_index": layer_index}
        )

    monkeypatch.setattr(post, "_fit_residual_chart", fit_chart)

    def capture_parent(_accumulators: Any, **kwargs: Any) -> None:
        assert kwargs["capture_source_local_pre"] is True
        calls.append(("capture_parent", kwargs["layer_index"]))

    monkeypatch.setattr(post, "_capture_global_parent_layer_boundaries", capture_parent)

    def capture_child(_accumulators: Any, **kwargs: Any) -> None:
        layer_index = kwargs["layer_index"]
        expected = prior if layer_index == len(_KINDS) - 1 else None
        assert kwargs["deferred_final_w2_prior"] is expected
        calls.append(("capture_child", layer_index))

    monkeypatch.setattr(post, "_capture_global_child_layer_primals", capture_child)
    evidence = _Evidence()
    writers = _writer_parameters(model)

    def build_evidence(_accumulators: Any, **kwargs: Any) -> tuple[Any, Any]:
        assert torch.equal(kwargs["mapped_parent_tied_table"], live_target)
        assert kwargs["deferred_final_w2_prior"] is prior
        assert kwargs["source_local"] is True
        calls.append(("build_evidence",))
        return evidence, writers

    monkeypatch.setattr(
        post,
        "_build_global_residual_evidence_from_accumulators",
        build_evidence,
    )

    def close(
        supplied_evidence: _Evidence,
        *,
        live_writer_parameters: MappingProxyType,
        deferred_final_writer_parameter: Any,
        policy: Any,
    ) -> _Installation:
        del policy
        assert supplied_evidence is evidence
        assert live_writer_parameters is writers
        assert deferred_final_writer_parameter is final_w2
        calls.append(("closure",))
        with torch.no_grad():
            if closure_mode == "protected":
                model.model.layers[0].self_attn.q_proj.weight.add_(3.0)
            else:
                model.model.layers[0].self_attn.out_proj.weight.add_(2.0)
            if closure_mode == "runtime":
                model.model.layers[0].self_attn.q_proj.weight.requires_grad_(False)
                raise RuntimeError("synthetic runtime-topology failure")
            if closure_mode == "raise":
                raise RuntimeError("synthetic closure failure")
        report = _sealed(
            {
                "schema": (
                    post.LFM2_SEMANTIC_SOURCE_LOCAL_OUTPUT_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA
                ),
                "final_w2_unchanged": True,
                "final_w2_installation_count": 0,
            },
            field="report_sha256",
        )
        return _Installation(
            corrected_final_w2_prior=_Prior(
                prior.down_weight.add(0.125),
                prior.live_deferred_w2_sha256,
                authentication_sha256="f" * 64,
            ),
            report=report,
        )

    monkeypatch.setattr(
        post,
        "compile_install_lfm2_semantic_source_local_output_residual_closure",
        close,
    )
    kwargs = {
        "cache": cache,
        "bridge": bridge,
        "donor_tokenizer": object(),
        "target_tokenizer": object(),
        "target_model": model,
        "causal_o_installation": causal,
        "expected_causal_o_receipt_sha256": causal.receipt["result_sha256"],
        "semantic_layer_stack_receipt": stack_receipt,
        "expected_semantic_layer_stack_receipt_sha256": stack_receipt[
            "result_sha256"
        ],
        "mapped_parent_tied_table": tied,
        "mapped_parent_tied_table_receipt": archive_manifest,
        "expected_mapped_parent_tied_table_sha256": (
            archive_manifest["tensor_records"]["mapped_parent_tied_table"]["sha256"]
        ),
        "expected_mapped_parent_tied_table_receipt_sha256": archive_manifest[
            "manifest_sha256"
        ],
        "producer_encoder": encoder,
        "producer_encoder_receipt": archive_manifest,
        "expected_producer_encoder_sha256": (
            archive_manifest["tensor_records"]["producer_encoder"]["sha256"]
        ),
        "expected_producer_encoder_receipt_sha256": archive_manifest[
            "manifest_sha256"
        ],
        "terminal_rebase": rebase,
        "expected_terminal_rebase_receipt_sha256": rebase.receipt[
            "result_sha256"
        ],
        "expected_terminal_prior_authentication_sha256": (
            prior.authentication_sha256
        ),
    }
    return kwargs, calls


def test_replays_live_child_causally_and_recloses_only_output_writers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    kwargs, calls = _install_harness(monkeypatch, model)
    before = post._state_tensor_hashes(model)
    qkv_before = {
        name: digest
        for name, digest in before.items()
        if any(role in name for role in ("q_proj", "k_proj", "v_proj"))
    }

    result = post.compile_install_lfm2_post_causal_o_source_local_global_reclosure(
        **kwargs
    )

    assert result.verify()
    assert result.receipt["tokenizer_session_sha256"] == "8" * 64
    assert result.receipt["semantic_refinement_sha256"] == "6" * 64
    assert result.receipt["cache_session_sha256"] == "9" * 64
    assert result.receipt["changed_state_keys"] == [
        "model.layers.0.self_attn.out_proj.weight"
    ]
    after = post._state_tensor_hashes(model)
    assert {
        name: after[name] for name in qkv_before
    } == qkv_before
    assert [value[:2] for value in calls if value[0] == "replay"] == [
        ("replay", 0),
        ("replay", 1),
        ("replay", 2),
    ]
    assert [value[2] for value in calls if value[0] == "replay"] == [
        (0.0, 1.0, 2.0, 3.0),
        (1.0, 2.0, 3.0, 4.0),
        (2.0, 3.0, 4.0, 5.0),
    ]
    assert [value for value in calls if value[0] == "parent"][-1][-1] is True
    assert result.receipt["current_live_child_replayed_in_causal_layer_order"] is True
    assert result.receipt["live_terminal_w2_unchanged_and_deferred"] is True


@pytest.mark.parametrize(
    "invalid_topology",
    (
        "same_parameter",
        "shared_storage",
        "offset_shared_storage",
        "config_tied",
    ),
)
def test_live_interface_rejects_tied_or_shared_output_topology(
    invalid_topology: str,
) -> None:
    model = _Model()
    if invalid_topology == "same_parameter":
        model.output.weight = model.embedding.weight
    elif invalid_topology == "shared_storage":
        model.output.weight = torch.nn.Parameter(model.embedding.weight.detach())
        assert model.output.weight is not model.embedding.weight
        assert model.output.weight.data_ptr() == model.embedding.weight.data_ptr()
    elif invalid_topology == "offset_shared_storage":
        source = model.embedding.weight.detach().reshape(-1)
        backing = torch.cat((source, source)).contiguous()
        elements = source.numel()
        model.embedding.weight = torch.nn.Parameter(
            backing[:elements].view_as(model.embedding.weight)
        )
        model.output.weight = torch.nn.Parameter(
            backing[elements:].view_as(model.output.weight)
        )
        assert model.output.weight.data_ptr() != model.embedding.weight.data_ptr()
        assert (
            model.output.weight.untyped_storage().data_ptr()
            == model.embedding.weight.untyped_storage().data_ptr()
        )
    else:
        model.config = SimpleNamespace(tie_word_embeddings=True)
    lineage = {
        "target_weight_sha256": "0" * 64,
        "target_weight_matches_live_input_child": True,
        "interface_runtime_mode": "independent_output",
        "live_output_parameter_independent": True,
        "output_table_matches_structural_input_at_body_entry": True,
    }

    with pytest.raises(post.Lfm2PostCausalOGlobalReclosureError):
        post._live_interface_target_table(model, interface_lineage=lineage)


@pytest.mark.parametrize(
    "field",
    (
        "expected_causal_o_receipt_sha256",
        "expected_producer_encoder_sha256",
        "expected_mapped_parent_tied_table_receipt_sha256",
        "expected_terminal_prior_authentication_sha256",
    ),
)
def test_independent_provenance_tamper_fails_before_publication_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    model = _Model()
    kwargs, calls = _install_harness(monkeypatch, model)
    before = post._state_tensor_hashes(model)
    kwargs[field] = "0" * 64

    with pytest.raises(post.Lfm2PostCausalOGlobalReclosureError):
        post.compile_install_lfm2_post_causal_o_source_local_global_reclosure(
            **kwargs
        )

    assert post._state_tensor_hashes(model) == before
    assert ("closure",) not in calls


def test_protected_producer_mutation_is_rejected_and_whole_model_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    kwargs, _calls = _install_harness(
        monkeypatch,
        model,
        closure_mode="protected",
    )
    before = post._state_tensor_hashes(model)

    with pytest.raises(post.Lfm2PostCausalOGlobalReclosureError):
        post.compile_install_lfm2_post_causal_o_source_local_global_reclosure(
            **kwargs
        )

    assert post._state_tensor_hashes(model) == before


def test_closure_exception_restores_exact_entry_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    kwargs, _calls = _install_harness(monkeypatch, model, closure_mode="raise")
    before = post._state_tensor_hashes(model)
    topology = post._parameter_topology(model)

    with pytest.raises(post.Lfm2PostCausalOGlobalReclosureError):
        post.compile_install_lfm2_post_causal_o_source_local_global_reclosure(
            **kwargs
        )

    assert post._state_tensor_hashes(model) == before
    assert post._parameter_topology(model) == topology


def test_unrestorable_runtime_mutation_is_not_misreported_as_exact_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    kwargs, _calls = _install_harness(monkeypatch, model, closure_mode="runtime")

    with pytest.raises(
        post.Lfm2PostCausalOGlobalReclosureError,
        match="exact rollback could not be proven",
    ):
        post.compile_install_lfm2_post_causal_o_source_local_global_reclosure(
            **kwargs
        )

    assert model.model.layers[0].self_attn.q_proj.weight.requires_grad is False
