from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_causal_post_qk_output_fold as causal  # noqa: E402
from neural_forge import (  # noqa: E402
    hf_lfm2_post_target_lattice_qk_output_fold as legacy,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import fingerprint_model_state  # noqa: E402
from neural_forge.hf_lfm2_attention_group_qdeim import (  # noqa: E402
    LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
    Lfm2AttentionGroupQdeimSourceLocalOutputFitCase,
    Lfm2CompiledAttentionGroupQdeim,
    lfm2_attention_group_qdeim_candidate_state_sha256,
    lfm2_attention_group_qdeim_source_local_output_fit_evidence_sha256,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_saved_body_target_lattice_qk import (  # noqa: E402
    LFM2_SAVED_BODY_TARGET_LATTICE_QK_SCHEMA,
)
from neural_forge.hf_lfm2_semantic_attention_refold_adapter import (  # noqa: E402
    LFM2_SEMANTIC_TARGET_LATTICE_QK_ONLY_SCHEMA,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import (  # noqa: E402
    LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
    _LayerReplayBank,
)
from neural_forge.hf_lfm2_structural_attention_candidate_archive import (  # noqa: E402
    load_lfm2_structural_attention_candidate_archive,
    write_lfm2_structural_attention_candidate_archive,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


_ATTENTION_INDICES = (2, 10)
_LAYER_COUNT = 11


def _sealed(
    body: dict[str, object],
    *,
    seal: str = "result_sha256",
) -> MappingProxyType:
    return MappingProxyType({**body, seal: canonical_sha256(body)})


def _tensor_sha(value: torch.Tensor) -> str:
    return lfm2_authenticated_tensor_sha256(value)


class _Config:
    model_type = "lfm2"
    layer_types = [
        "full_attention" if index in _ATTENTION_INDICES else "short_conv"
        for index in range(_LAYER_COUNT)
    ]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "hidden_size": 4,
            "layer_types": list(self.layer_types),
        }


class _Norm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))


class _Attention(torch.nn.Module):
    def __init__(self, layer_index: int) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.float32)
        self.k_proj = torch.nn.Linear(4, 2, bias=False, dtype=torch.float32)
        self.v_proj = torch.nn.Linear(4, 2, bias=False, dtype=torch.float32)
        self.out_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.float32)
        self.q_layernorm = _Norm()
        self.k_layernorm = _Norm()
        with torch.no_grad():
            self.q_proj.weight.copy_(
                torch.eye(4, dtype=torch.float32).mul(2.0 + layer_index)
            )
            self.k_proj.weight.copy_(
                torch.tensor(
                    [
                        [1.5 + layer_index, 0.0, 0.0, 0.0],
                        [0.0, 1.75 + layer_index, 0.0, 0.0],
                    ]
                )
            )
            self.v_proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0 + 0.1 * layer_index, 0.0, 0.0, 0.0],
                        [0.0, 1.0 + 0.1 * layer_index, 0.0, 0.0],
                    ]
                )
            )
            self.out_proj.weight.copy_(
                torch.eye(4, dtype=torch.float32).mul(1.0 + 0.05 * layer_index)
            )


class _AttentionLayer(torch.nn.Module):
    is_attention_layer = True

    def __init__(self, layer_index: int) -> None:
        super().__init__()
        self.self_attn = _Attention(layer_index)


class _ShortConvLayer(torch.nn.Module):
    is_attention_layer = False

    def __init__(self, layer_index: int) -> None:
        super().__init__()
        self.writer = torch.nn.Parameter(
            torch.eye(4, dtype=torch.float32).mul(0.01 * (layer_index + 1))
        )


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                (
                    _AttentionLayer(index)
                    if index in _ATTENTION_INDICES
                    else _ShortConvLayer(index)
                )
                for index in range(_LAYER_COUNT)
            ]
        )


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.model = _Backbone()
        self.embedding = torch.nn.Embedding(19, 4, dtype=torch.float32)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embedding

    def forward(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("Causal O folding must not call model.forward().")


class _Session:
    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def as_dict(self) -> MappingProxyType:
        return MappingProxyType({"session_sha256": "9" * 64})


class _Cache:
    cache_key = "c" * 64
    parent_fingerprint = "d" * 64
    batch_count = 1

    def __init__(self) -> None:
        self.manifest = {
            "layer_topology": {
                "layers": [
                    {
                        "contract": {
                            "kind": (
                                "full_attention"
                                if index in _ATTENTION_INDICES
                                else "short_conv"
                            )
                        }
                    }
                    for index in range(_LAYER_COUNT)
                ]
            }
        }

    def selective_authenticated_session(self) -> _Session:
        return _Session()


class _Bridge:
    artifact_sha256 = "b" * 64


def _candidate(model: _Model, layer_index: int) -> Lfm2CompiledAttentionGroupQdeim:
    attention = model.model.layers[layer_index].self_attn
    query = torch.eye(4, dtype=torch.float32).mul(0.5 + layer_index)
    key = torch.tensor(
        [
            [0.5 + layer_index, 0.0, 0.0, 0.0],
            [0.0, 0.75 + layer_index, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    value = attention.v_proj.weight.detach().cpu().clone()
    out = attention.out_proj.weight.detach().cpu().clone()
    basis = torch.tensor([[1.0], [0.5]], dtype=torch.float64)
    interpolation = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    reconstruction = torch.eye(4, dtype=torch.float64)
    decoder = torch.eye(4, dtype=torch.float64)
    chart = torch.eye(4, dtype=torch.float64)
    body = {
        "schema": LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
        "selection": {
            "selected_source_key_value_groups": [0],
            "selected_source_query_heads": [0, 1],
        },
        "interpolation": {
            "group_pod_basis_sha256": _tensor_sha(basis),
            "group_interpolation_sha256": _tensor_sha(interpolation),
            "context_reconstruction_sha256": _tensor_sha(reconstruction),
        },
        "authentication": {
            "normalized_input_decoder_sha256": _tensor_sha(decoder),
            "residual_output_chart_sha256": _tensor_sha(chart),
        },
        "materialization": {
            "candidate_state_sha256": (
                lfm2_attention_group_qdeim_candidate_state_sha256(
                    query,
                    key,
                    value,
                    out,
                )
            ),
            "candidate_dtype": "float32",
        },
    }
    return Lfm2CompiledAttentionGroupQdeim(
        selected_source_key_value_groups=(0,),
        selected_source_query_heads=(0, 1),
        query_weight=query,
        key_weight=key,
        value_weight=value,
        out_weight=out,
        group_pod_basis=basis,
        group_interpolation=interpolation,
        context_reconstruction=reconstruction,
        normalized_input_decoder=decoder,
        residual_output_chart=chart,
        report=_sealed(body),
    )


def _stack(
    candidates: dict[int, Lfm2CompiledAttentionGroupQdeim],
) -> MappingProxyType:
    layers = []
    for index in range(_LAYER_COUNT):
        kind = "full_attention" if index in _ATTENTION_INDICES else "short_conv"
        body: dict[str, object] = {
            "schema": "neural-forge-lfm2-semantic-layer-transaction-v1",
            "layer_index": index,
            "operator_kind": kind,
        }
        if kind == "full_attention":
            chart_body = {
                "schema": "neural-forge-lfm2-layer-stack-residual-chart-v1",
                "layer_index": index,
                "partition_batch_indices": {"fit": [0]},
                "chart_sha256": _tensor_sha(
                    candidates[index].residual_output_chart
                ),
            }
            chart = _sealed(chart_body)
            structural_body = {
                "schema": (
                    "neural-forge-lfm2-semantic-attention-group-qdeim-"
                    "layer-installation-v1"
                ),
                "layer_index": index,
                "fit_batch_indices": [0],
                "fit_batch_sample_weights": [
                    {"batch_index": 0, "sample_weight": 1.0}
                ],
                "cache_key": _Cache.cache_key,
                "bridge_artifact_sha256": _Bridge.artifact_sha256,
                "coupled_attention_compiler_receipt": dict(
                    candidates[index].report
                ),
                "residual_chart_receipt": dict(chart),
            }
            structural = _sealed(structural_body)
            operator = (
                _sealed(
                    {
                        "schema": (
                            "neural-forge-lfm2-semantic-attention-source-local-"
                            "output-composite-v1"
                        ),
                        "layer_index": index,
                        "structural_attention_operator_receipt": dict(structural),
                    }
                )
                if index == 10
                else structural
            )
            body["attention_operator_installation_receipt"] = dict(operator)
        layers.append(dict(_sealed(body)))
    return _sealed(
        {
            "schema": LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
            "layer_installations": layers,
        }
    )


def _qk_treatment(model: _Model) -> MappingProxyType:
    identity = fingerprint_lfm2_parent(model).identity_sha256
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    changed: list[str] = []
    installations = []
    steps = []
    for index in _ATTENTION_INDICES:
        attention = model.model.layers[index].self_attn
        prefix = f"model.layers.{index}.self_attn."
        local_changed = [prefix + "k_proj.weight", prefix + "q_proj.weight"]
        changed.extend(local_changed)
        fixed = {
            role: _tensor_sha(getattr(getattr(attention, module), "weight"))
            for role, module in (
                ("v_proj.weight", "v_proj"),
                ("out_proj.weight", "out_proj"),
                ("q_layernorm.weight", "q_layernorm"),
                ("k_layernorm.weight", "k_layernorm"),
            )
        }
        installation = _sealed(
            {
                "schema": LFM2_SEMANTIC_TARGET_LATTICE_QK_ONLY_SCHEMA,
                "layer_index": index,
                "installed": True,
                "qk_only": True,
                "changed_state_keys": local_changed,
                "actual_changed_state_keys_are_qk_only": True,
                "parameter_dtype": "float32",
                "checkpoint_dtype_preserved": True,
                "frozen_query_weight_sha256": _tensor_sha(
                    attention.q_proj.weight
                ),
                "frozen_key_weight_sha256": _tensor_sha(attention.k_proj.weight),
                "fixed_tensor_sha256_before_and_after": fixed,
            }
        )
        installations.append(dict(installation))
        steps.append(
            dict(
                _sealed(
                    {
                        "schema": (
                            "neural-forge-lfm2-saved-body-qk-layer-step-v1"
                        ),
                        "layer_index": index,
                        "operator_kind": "full_attention",
                        "entry_replay_bank_sha256": canonical_sha256(
                            ["entry", index]
                        ),
                        "entry_model_state_sha256": canonical_sha256(
                            ["entry-state", index]
                        ),
                        "qk_installation_sha256": installation["result_sha256"],
                        "transport_evidence_sha256": [
                            canonical_sha256(["transport", index])
                        ],
                        "propagation_replay_bank_sha256": canonical_sha256(
                            ["propagation", index]
                        ),
                        "propagation_model_state_sha256": canonical_sha256(
                            ["propagation-state", index]
                        ),
                        "qk_modified": True,
                        "non_qk_tensor_modified": False,
                    }
                )
            )
        )
    body = {
        "schema": LFM2_SAVED_BODY_TARGET_LATTICE_QK_SCHEMA,
        "model_state_before_sha256": "a" * 64,
        "model_state_after_sha256": identity,
        "parameter_dtype": "float32",
        "checkpoint_dtype_preserved": True,
        "changed_state_keys": sorted(changed),
        "actual_changed_state_keys_are_qk_only": True,
        "attention_layer_indices": list(_ATTENTION_INDICES),
        "attention_layer_count": len(_ATTENTION_INDICES),
        "attention_install_count": len(_ATTENTION_INDICES),
        "layer_steps": steps,
        "attention_installations": installations,
        "causal_layer_order_proven": True,
        "later_layers_replayed_from_treated_prefix": True,
        "fixed_child_value_output_used_as_endpoint_teacher": True,
        "all_non_qk_tensors_byte_identical": True,
        "parameter_element_count_before": parameter_count,
        "parameter_element_count_after": parameter_count,
        "parameter_count_changed": False,
        "operation_audit": dict(legacy._QK_TREATMENT_ZERO_OPERATION_AUDIT),
    }
    return _sealed(body)


def _provenance(model: _Model, treatment: MappingProxyType) -> MappingProxyType:
    body = {
        "schema": legacy._QK_TREATED_BODY_PROVENANCE_SCHEMA,
        "status": "saved-reloaded-qk-only-treated-body",
        "source_checkpoint_inventory_sha256": "1" * 64,
        "source_checkpoint_dtype": "float32",
        "source_body_model_state_sha256": "2" * 64,
        "source_body_lfm2_identity_sha256": "3" * 64,
        "source_causal_body_inventory_sha256": "4" * 64,
        "treated_checkpoint_inventory_sha256": "5" * 64,
        "treated_checkpoint_dtype": "float32",
        "checkpoint_dtype_preserved": True,
        "treated_body_model_state_sha256": fingerprint_model_state(model),
        "treated_body_lfm2_identity_sha256": (
            fingerprint_lfm2_parent(model).identity_sha256
        ),
        "treated_causal_body_inventory_sha256": "6" * 64,
        "changed_checkpoint_relative_paths": ["model.safetensors"],
        "changed_state_keys": list(treatment["changed_state_keys"]),
        "target_lattice_qk_treatment_sha256": treatment["result_sha256"],
        "causal_output_head_installed": False,
        "standalone_fresh_reload_verified": True,
        "operation_audit": dict(legacy._QK_FRESH_RELOAD_ZERO_OPERATION_AUDIT),
    }
    return _sealed(body, seal="receipt_sha256")


@dataclass(frozen=True)
class _Parent:
    layer_index: int
    evidence_sha256: str


def _install_internal_replay_fakes(
    monkeypatch: pytest.MonkeyPatch,
    model: _Model,
    *,
    no_effect_indices: frozenset[int] = frozenset(),
) -> dict[str, object]:
    alignment = SimpleNamespace(
        source_batch_index=0,
        alignment_sha256="e" * 64,
    )
    observations: dict[str, object] = {
        "replay": [],
        "parent_loads": [],
        "builder_parents": [],
    }

    monkeypatch.setattr(causal, "Lfm2CircuitTraceCache", _Cache)
    monkeypatch.setattr(causal, "StrictTokenizerBridgeArtifact", _Bridge)
    monkeypatch.setattr(
        causal,
        "_semantic_alignments_from_session",
        lambda **_kwargs: ((alignment,), {"receipt_sha256": "9" * 64}),
    )
    monkeypatch.setattr(
        causal,
        "audit_lfm2_semantic_span_cpcf_refinement",
        lambda *_args, **_kwargs: {
            "probability_preserving_attention_transport_go": True,
            "report_sha256": "f" * 64,
        },
    )
    monkeypatch.setattr(
        causal,
        "_initial_residual_bank",
        lambda *_args, **_kwargs: MappingProxyType(
            {0: torch.eye(4, dtype=torch.float32).unsqueeze(0)}
        ),
    )

    def replay(
        target_model: _Model,
        *,
        alignments: object,
        layer_index: int,
        pre_residual_by_batch: MappingProxyType,
    ) -> _LayerReplayBank:
        del alignments
        pre = pre_residual_by_batch[0].detach().cpu().clone()
        layer = target_model.model.layers[layer_index]
        if layer.is_attention_layer:
            context = pre.clone()
            local = context @ layer.self_attn.out_proj.weight.detach().cpu().T
            post = pre + local
        else:
            context = pre.clone()
            post = pre + layer.writer.detach().cpu().unsqueeze(0)
        identity = fingerprint_lfm2_parent(target_model).identity_sha256
        replay_sha = canonical_sha256(
            {
                "layer_index": layer_index,
                "identity": identity,
                "pre": _tensor_sha(pre),
                "post": _tensor_sha(post),
            }
        )
        replay_value = SimpleNamespace(
            operator_input=pre,
            operator_output_input=context,
            post_layer_residual=post,
            replay_sha256=replay_sha,
        )
        bank_body = {
            "schema": "neural-forge-lfm2-semantic-layer-replay-bank-v1",
            "layer_index": layer_index,
            "captured_child_model_state_sha256": identity,
            "batch_replays": [
                {
                    "batch_index": 0,
                    "alignment_sha256": alignment.alignment_sha256,
                    "replay_sha256": replay_sha,
                }
            ],
            "model_forward_calls": 0,
            "layer_forward_calls": 0,
            "training_steps": 0,
        }
        observations["replay"].append(
            {
                "layer_index": layer_index,
                "identity": identity,
                "pre_sha256": _tensor_sha(pre),
                "layer_2_o_sha256": _tensor_sha(
                    target_model.model.layers[2].self_attn.out_proj.weight
                ),
            }
        )
        return _LayerReplayBank(
            captured_child_model_state_sha256=identity,
            layer_index=layer_index,
            pre_residual_by_batch=MappingProxyType({0: pre}),
            replay_by_batch=MappingProxyType({0: replay_value}),
            bank_sha256=canonical_sha256(bank_body),
        )

    monkeypatch.setattr(causal, "_replay_current_layer", replay)

    def load_parent(
        _session: object,
        *,
        alignments: object,
        layer_index: int,
        operator_kind: str,
    ) -> MappingProxyType:
        del alignments
        assert operator_kind == "full_attention"
        observations["parent_loads"].append(layer_index)
        return MappingProxyType(
            {
                0: _Parent(
                    layer_index=layer_index,
                    evidence_sha256=canonical_sha256(["parent", layer_index]),
                )
            }
        )

    monkeypatch.setattr(causal, "_load_parent_layer_trajectories", load_parent)

    def build_cases(
        *,
        alignments: object,
        parent: MappingProxyType,
        structural_replay: _LayerReplayBank,
        fit_batch_indices: tuple[int, ...],
        fit_batch_sample_weights: MappingProxyType,
        layer_index: int,
    ) -> tuple[tuple[Lfm2AttentionGroupQdeimSourceLocalOutputFitCase, ...], object]:
        del alignments
        assert fit_batch_indices == (0,)
        assert dict(fit_batch_sample_weights) == {0: 1.0}
        assert parent[0].layer_index == layer_index
        observations["builder_parents"].append(parent[0].evidence_sha256)
        replay_value = structural_replay.replay_by_batch[0]
        context = replay_value.operator_output_input[0].to(torch.float64)
        attention = model.model.layers[layer_index].self_attn
        prior = context @ attention.out_proj.weight.detach().to(torch.float64).T
        row_count = int(context.shape[0])
        target_delta = torch.eye(row_count, 4, dtype=torch.float64).mul(
            0.0 if layer_index in no_effect_indices else 0.5
        )
        row_identity = canonical_sha256(
            {
                "schema": (
                    "neural-forge-lfm2-source-local-attention-aligned-fit-rows-v1"
                ),
                "partition": "fit",
                "layer_index": layer_index,
                "batch_index": 0,
                "alignment_sha256": alignment.alignment_sha256,
                "target_positions": list(range(row_count)),
                "donor_positions": list(range(row_count)),
                "row_count": row_count,
            }
        )
        case = Lfm2AttentionGroupQdeimSourceLocalOutputFitCase(
            case_id=f"layer-{layer_index:03d}.batch-0000",
            parent_pre_operator_residual=torch.zeros_like(prior),
            parent_post_operator_residual=prior + target_delta,
            live_child_pre_operator_residual=(
                replay_value.operator_input[0].to(torch.float64)
            ),
            live_child_pre_o_context=context,
            row_sample_weights=tuple(1.0 for _ in range(row_count)),
            parent_trajectory_receipt_sha256=parent[0].evidence_sha256,
            live_child_replay_receipt_sha256=replay_value.replay_sha256,
            aligned_row_identity_sha256=row_identity,
        )
        fit_sha = (
            lfm2_attention_group_qdeim_source_local_output_fit_evidence_sha256(
                (case,)
            )
        )
        receipt = _sealed(
            {
                "schema": (
                    "neural-forge-lfm2-semantic-attention-source-local-"
                    "fit-cases-v1"
                ),
                "layer_index": layer_index,
                "partition": "fit",
                "fit_batch_indices": [0],
                "fit_evidence_sha256": fit_sha,
                "structural_replay_bank_sha256": structural_replay.bank_sha256,
            }
        )
        return (case,), receipt

    monkeypatch.setattr(
        causal,
        "_build_attention_source_local_output_fit_cases",
        build_cases,
    )
    return observations


def _arguments(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
    *,
    no_effect_indices: frozenset[int] = frozenset(),
) -> tuple[_Model, dict[str, object], dict[str, object]]:
    model = _Model()
    candidates = {index: _candidate(model, index) for index in _ATTENTION_INDICES}
    stack = _stack(candidates)
    archive_root = tmp_path / "attention-candidates"
    written = write_lfm2_structural_attention_candidate_archive(
        archive_root,
        candidates,
        semantic_layer_stack_receipt=stack,
    )
    loaded, manifest = load_lfm2_structural_attention_candidate_archive(
        archive_root,
        semantic_layer_stack_receipt=stack,
        expected_manifest_sha256=written["manifest_sha256"],
    )
    treatment = _qk_treatment(model)
    provenance = _provenance(model, treatment)
    observations = _install_internal_replay_fakes(
        monkeypatch,
        model,
        no_effect_indices=no_effect_indices,
    )
    arguments: dict[str, object] = {
        "cache": _Cache(),
        "bridge": _Bridge(),
        "donor_tokenizer": object(),
        "target_tokenizer": object(),
        "target_model": model,
        "qk_treatment_receipt": treatment,
        "qk_fresh_reload_provenance_receipt": provenance,
        "expected_qk_fresh_reload_provenance_sha256": provenance[
            "receipt_sha256"
        ],
        "semantic_layer_stack_receipt": stack,
        "expected_semantic_layer_stack_receipt_sha256": stack["result_sha256"],
        "structural_candidates": loaded,
        "structural_candidate_archive_manifest": manifest,
        "expected_structural_candidate_archive_manifest_sha256": manifest[
            "manifest_sha256"
        ],
        "ridge": 1.0e-6,
        "fold_compute_device": "cpu",
    }
    return model, arguments, observations


def _state(model: _Model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def test_causal_fold_orders_multidigit_layers_and_uses_internal_cached_rows(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, arguments, observations = _arguments(tmp_path, monkeypatch)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    before = _state(model)
    before_layer_2_o = _tensor_sha(
        model.model.layers[2].self_attn.out_proj.weight
    )
    parameter_ids = {name: id(value) for name, value in model.named_parameters()}
    pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
    gradient_ids = {
        name: id(value.grad) for name, value in model.named_parameters()
    }
    gradients = {
        name: value.grad.detach().clone() for name, value in model.named_parameters()
    }

    result = causal.compile_install_lfm2_causal_post_qk_output_fold(**arguments)

    assert result.verify()
    assert result.receipt["tokenizer_session_sha256"] == "9" * 64
    assert result.receipt["semantic_refinement_sha256"] == "f" * 64
    assert [value["layer_index"] for value in result.layer_receipts] == [2, 10]
    assert [
        value["causal_attention_sequence_number"] for value in result.layer_receipts
    ] == [0, 1]
    assert observations["parent_loads"] == [2, 10]
    assert len(observations["builder_parents"]) == 2
    assert "fit_cases" not in inspect.signature(
        causal.compile_install_lfm2_causal_post_qk_output_fold
    ).parameters
    assert "replay_evidence_by_layer" not in inspect.signature(
        causal.compile_install_lfm2_causal_post_qk_output_fold
    ).parameters
    layer_10_entry = next(
        value
        for value in observations["replay"]
        if value["layer_index"] == 10
    )
    assert layer_10_entry["layer_2_o_sha256"] != before_layer_2_o
    step_10 = result.receipt["causal_replay_steps"][10]
    assert step_10["upstream_installed_o_state_keys"] == [
        "model.layers.2.self_attn.out_proj.weight",
        "model.layers.10.self_attn.out_proj.weight",
    ]
    expected_changed = {
        f"model.layers.{index}.self_attn.out_proj.weight"
        for index in _ATTENTION_INDICES
    }
    after = _state(model)
    assert {
        name for name in before if not torch.equal(before[name], after[name])
    } == expected_changed
    assert {
        name: id(value) for name, value in model.named_parameters()
    } == parameter_ids
    assert {
        name: value.data_ptr() for name, value in model.named_parameters()
    } == pointers
    assert {
        name: id(value.grad) for name, value in model.named_parameters()
    } == gradient_ids
    assert all(
        torch.equal(value.grad, gradients[name])
        for name, value in model.named_parameters()
    )


def test_mutated_or_forged_chart_is_rejected_before_publication(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, arguments, _observations = _arguments(tmp_path, monkeypatch)
    before = _state(model)
    candidates = arguments["structural_candidates"]
    candidates[2].residual_output_chart[0, 0] += 0.25

    with pytest.raises(
        causal.Lfm2CausalPostQkOutputFoldError,
        match="exact entry state was restored",
    ):
        causal.compile_install_lfm2_causal_post_qk_output_fold(**arguments)

    assert all(
        torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )


def test_consistently_resealed_forged_chart_breaks_stack_archive_binding(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, arguments, _observations = _arguments(tmp_path, monkeypatch)
    stack_body = causal._deep_thaw(arguments["semantic_layer_stack_receipt"])
    stack_body.pop("result_sha256")
    layer = stack_body["layer_installations"][2]
    layer.pop("result_sha256")
    operator = layer["attention_operator_installation_receipt"]
    operator.pop("result_sha256")
    chart = operator["residual_chart_receipt"]
    chart.pop("result_sha256")
    chart["chart_sha256"] = "0" * 64
    operator["residual_chart_receipt"] = dict(_sealed(chart))
    layer["attention_operator_installation_receipt"] = dict(_sealed(operator))
    stack_body["layer_installations"][2] = dict(_sealed(layer))
    forged_stack = _sealed(stack_body)

    manifest_body = causal._deep_thaw(
        arguments["structural_candidate_archive_manifest"]
    )
    manifest_body.pop("fresh_reload_verified")
    manifest_body.pop("manifest_sha256")
    manifest_body["semantic_layer_stack_result_sha256"] = forged_stack[
        "result_sha256"
    ]
    forged_manifest = {
        **manifest_body,
        "manifest_sha256": canonical_sha256(manifest_body),
        "fresh_reload_verified": True,
    }
    arguments["semantic_layer_stack_receipt"] = forged_stack
    arguments["expected_semantic_layer_stack_receipt_sha256"] = forged_stack[
        "result_sha256"
    ]
    arguments["structural_candidate_archive_manifest"] = forged_manifest
    arguments["expected_structural_candidate_archive_manifest_sha256"] = (
        forged_manifest["manifest_sha256"]
    )

    with pytest.raises(
        causal.Lfm2CausalPostQkOutputFoldError,
        match="exact entry state was restored",
    ):
        causal.compile_install_lfm2_causal_post_qk_output_fold(**arguments)


def test_authenticated_noop_still_replays_and_advances_causally(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, arguments, observations = _arguments(
        tmp_path,
        monkeypatch,
        no_effect_indices=frozenset({10}),
    )

    result = causal.compile_install_lfm2_causal_post_qk_output_fold(**arguments)

    assert result.receipt["attention_o_decision_count"] == 2
    assert result.receipt["attention_o_install_count"] == 1
    assert result.receipt["attention_o_authenticated_noop_count"] == 1
    assert result.layer_receipts[1]["changed_state_keys"] == []
    assert result.layer_receipts[1]["authenticated_o_representational_noop"]
    assert result.layer_receipts[1]["post_install_replay_precedes_downstream_layer"]
    assert sum(
        value["layer_index"] == 10 for value in observations["replay"]
    ) == 2


def test_outer_transaction_rolls_back_upstream_o_when_later_layer_fails(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, arguments, _observations = _arguments(tmp_path, monkeypatch)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    before = _state(model)
    parameter_ids = {name: id(value) for name, value in model.named_parameters()}
    pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
    gradients = {
        name: value.grad.detach().clone() for name, value in model.named_parameters()
    }
    original = causal.compile_install_lfm2_semantic_attention_source_local_output
    calls = 0

    def fail_second(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            with torch.no_grad():
                kwargs["target_model"].model.layers[10].self_attn.out_proj.weight.add_(
                    7.0
                )
            raise RuntimeError("injected later-layer failure")
        return original(**kwargs)

    monkeypatch.setattr(
        causal,
        "compile_install_lfm2_semantic_attention_source_local_output",
        fail_second,
    )

    with pytest.raises(
        causal.Lfm2CausalPostQkOutputFoldError,
        match="exact entry state was restored",
    ):
        causal.compile_install_lfm2_causal_post_qk_output_fold(**arguments)

    assert calls == 2
    assert all(
        torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )
    assert {
        name: id(value) for name, value in model.named_parameters()
    } == parameter_ids
    assert {
        name: value.data_ptr() for name, value in model.named_parameters()
    } == pointers
    assert all(
        torch.equal(value.grad, gradients[name])
        for name, value in model.named_parameters()
    )
