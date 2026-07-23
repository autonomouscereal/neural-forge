from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (  # noqa: E402
    seal_lfm2_dense_qkvo_source_inventory,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CACHE_SCHEMA,
    SELECTIVE_AUTHENTICATED_SESSION_SCHEMA,
)
from neural_forge.hf_lfm2_semantic_attention_group_qdeim_adapter import (  # noqa: E402
    Lfm2SemanticAttentionGroupQdeimAdapterError,
    Lfm2SemanticAttentionGroupQdeimCandidate,
    Lfm2SemanticAttentionGroupQdeimEvidenceDeficit,
    compile_lfm2_semantic_attention_group_qdeim_from_cache,
    inspect_lfm2_semantic_attention_group_qdeim_cache,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


def _sealed(body: dict[str, object], seal: str) -> dict[str, object]:
    return {**body, seal: canonical_sha256(body)}


class _Payload:
    def __init__(self, session_sha: str, index: int, names, tensors) -> None:
        self.batch_index = index
        self.input_sha256 = f"{index + 1:064x}"
        self.ordered_tensor_names = tuple(names)
        self.logical_bytes = sum(
            int(tensors[name].numel() * tensors[name].element_size()) for name in names
        )
        self.session_sha256 = session_sha
        self.selection_sha256 = canonical_sha256({"batch": index, "names": list(names)})
        self.released = False
        self.tensors = MappingProxyType({name: tensors[name] for name in names})


class _Session:
    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        cache_key = "c" * 64
        contract = {
            "schema": "neural-forge-lfm2-layer-contract-v1",
            "kind": "full_attention",
            "layer_index": 0,
            "operator": {
                "operator": "full_attention",
                "hidden_size": 4,
                "query_heads": 10,
                "key_value_heads": 5,
                "head_dim": 2,
                "query_projection_shape": [20, 4],
                "key_projection_shape": [10, 4],
                "value_projection_shape": [10, 4],
                "output_projection_shape": [4, 20],
            },
        }
        manifest_body = {
            "schema": CACHE_SCHEMA,
            "cache_key": cache_key,
            "parent": {"identity_sha256": "d" * 64},
            "corpus": {"batch_count": len(batches)},
            "implementation": {},
            "layer_topology": {"layers": [{"contract": contract}]},
            "capture": {},
            "runtime": {},
            "execution": {},
            "storage": {},
        }
        self.manifest = _sealed(manifest_body, "manifest_sha256")
        self.cache_key = cache_key
        self._batches = batches
        self.batch_references = tuple(
            SimpleNamespace(
                index=index,
                input_sha256=f"{index + 1:064x}",
                record_sha256=f"{index + 17:064x}",
                ordered_tensor_names=tuple(sorted(tensors)),
            )
            for index, tensors in enumerate(batches)
        )
        receipt_body = {
            "schema": SELECTIVE_AUTHENTICATED_SESSION_SCHEMA,
            "cache_key": cache_key,
            "manifest_sha256": self.manifest["manifest_sha256"],
            "completion_sha256": "e" * 64,
            "batch_count": len(batches),
            "ordered_authenticated_batches": [
                {"batch_index": index} for index in range(len(batches))
            ],
            "authentication_contract": {
                "parent_forward_calls": 0,
                "training_steps": 0,
            },
        }
        self._receipt = _sealed(receipt_body, "session_sha256")
        self.evidence_sha256 = self._receipt["session_sha256"]
        self.loaded: list[tuple[int, tuple[str, ...]]] = []
        self.released = 0

    def as_dict(self):
        return dict(self._receipt)

    def load_selected_tensors(self, index: int, names):
        names = tuple(names)
        self.loaded.append((index, names))
        return _Payload(self.evidence_sha256, index, names, self._batches[index])

    def release_selected_tensors(self, payload):
        payload.released = True
        self.released += 1

    def verify(self):
        return self


def _contexts(*, rank_deficient: bool = False) -> torch.Tensor:
    rows = 4
    groups = 5
    q_per_group = 2
    head_dim = 2
    if rank_deficient:
        common = torch.arange(rows * q_per_group * head_dim, dtype=torch.float64)
        features = common[:, None].repeat(1, groups)
    else:
        latent = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, -1.0, 0.0],
                [0.0, 2.0, -1.0],
                [-1.0, 0.0, 2.0],
                [1.0, 2.0, 3.0],
                [3.0, 1.0, 2.0],
                [2.0, 3.0, 1.0],
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=torch.float64,
        )
        mixing = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 2.0],
                [0.0, 1.0, 0.0, 1.0, -1.0],
                [0.0, 0.0, 1.0, 0.5, 1.0],
            ],
            dtype=torch.float64,
        )
        features = latent @ mixing
    rows_first = (
        features.reshape(rows, q_per_group, head_dim, groups)
        .permute(0, 3, 1, 2)
        .reshape(rows, groups * q_per_group, head_dim)
    )
    return rows_first.permute(1, 0, 2).unsqueeze(0).contiguous()


def _source_and_maps() -> dict[str, object]:
    source_q = torch.arange(20 * 4, dtype=torch.float64).reshape(20, 4).div(8.0)
    source_k = (
        torch.arange(10 * 4, dtype=torch.float64).reshape(10, 4).add(2.0).div(16.0)
    )
    source_v = source_k.mul(-0.5).add(1.0)
    source_o = (
        torch.arange(4 * 20, dtype=torch.float64).reshape(4, 20).sub(5.0).div(32.0)
    )
    decoder = torch.tensor(
        [
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, -0.5],
        ],
        dtype=torch.float64,
    )
    chart = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 1.0],
            [0.0, -0.5, 1.0],
        ],
        dtype=torch.float64,
    )
    chart_body = {
        "schema": "neural-forge-lfm2-layer-stack-residual-chart-v1",
        "layer_index": 0,
        "chart_sha256": lfm2_authenticated_tensor_sha256(chart),
        "partition_batch_indices": {
            "fit": [0],
            "fit_check": [1],
            "selection": [2],
            "validation": [3],
        },
        "fit_only": True,
        "validation_rows_used_for_fit": 0,
        "numeric_metric_used_for_candidate_selection": False,
        "numeric_metric_used_for_publication_or_rollback": False,
        "training_steps": 0,
    }
    chart_receipt = _sealed(chart_body, "result_sha256")
    decoder_body = {
        "schema": "neural-forge-lfm2-normalized-operator-input-decoder-v1",
        "residual_chart_sha256": chart_receipt["result_sha256"],
        "residual_chart_observed_sha256": lfm2_authenticated_tensor_sha256(chart),
        "published_decoder_sha256": lfm2_authenticated_tensor_sha256(decoder),
        "trajectory_batch_indices": [0, 1, 2, 3],
        "fit_batch_indices": [0],
        "heldout_batch_indices": [1, 2, 3],
        "fit_trajectory_count": 1,
        "residual_chart_partition_batch_indices_sha256": canonical_sha256(
            {
                "fit": [0],
                "fit_check": [1],
                "selection": [2],
                "validation": [3],
            }
        ),
        "fit_batch_indices_authenticated_from_residual_chart": True,
        "fit_rows_only_used_for_residual": True,
        "heldout_rows_used_for_fit": 0,
        "gradients_used": False,
        "training_steps": 0,
    }
    decoder_receipt = _sealed(decoder_body, "decoder_sha256")
    templates = {
        "target_query_weight_template": torch.empty(12, 3, dtype=torch.bfloat16),
        "target_key_weight_template": torch.empty(6, 3, dtype=torch.bfloat16),
        "target_value_weight_template": torch.empty(6, 3, dtype=torch.bfloat16),
        "target_out_weight_template": torch.empty(3, 12, dtype=torch.bfloat16),
    }
    return {
        "source_query_weight": source_q,
        "source_key_weight": source_k,
        "source_value_weight": source_v,
        "source_out_weight": source_o,
        "source_inventory": seal_lfm2_dense_qkvo_source_inventory(
            source_query_weight=source_q,
            source_key_weight=source_k,
            source_value_weight=source_v,
            source_output_weight=source_o,
        ),
        **templates,
        "normalized_input_decoder": decoder,
        "normalized_input_decoder_receipt": decoder_receipt,
        "residual_output_chart": chart,
        "residual_chart_receipt": chart_receipt,
    }


def _arguments(session: _Session) -> dict[str, object]:
    return {
        "session": session,
        "layer_index": 0,
        "fit_batch_indices": (0,),
        "fit_batch_sample_weights": {0: 2.0},
        **_source_and_maps(),
    }


def test_true_context_cache_route_emits_dense_candidate_and_never_loads_holdout() -> (
    None
):
    context = _contexts()
    fit = {
        "prefill.layer.000.attention.q": torch.zeros(1, 10, 3, 2),
        "prefill.layer.000.attention.context": context[:, :, :3],
        "decode.000.layer.000.attention.q": torch.zeros(1, 10, 1, 2),
        "decode.000.layer.000.attention.context": context[:, :, 3:],
    }
    heldout = {
        "prefill.layer.000.attention.q": torch.zeros(1, 10, 3, 2),
        "prefill.layer.000.attention.context": context[:, :, :3].mul(7.0),
        "decode.000.layer.000.attention.q": torch.zeros(1, 10, 1, 2),
        "decode.000.layer.000.attention.context": context[:, :, 3:].mul(7.0),
    }
    session = _Session([fit, heldout])
    result = compile_lfm2_semantic_attention_group_qdeim_from_cache(
        **_arguments(session)
    )
    assert isinstance(result, Lfm2SemanticAttentionGroupQdeimCandidate)
    assert session.loaded == [
        (
            0,
            (
                "decode.000.layer.000.attention.context",
                "prefill.layer.000.attention.context",
            ),
        ),
    ]
    assert session.released == 1
    state = result.state_dict()
    assert set(state) == {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
    }
    assert all(value.dtype == torch.bfloat16 for value in state.values())
    assert all(value.layout == torch.strided for value in state.values())
    receipt = result.report
    assert receipt["captured_true_post_softmax_context_only"] is True
    assert receipt["heldout_batch_indices"] == [1]
    assert receipt["heldout_tensor_payload_loads"] == 0
    assert receipt["qkv_producer_averaging_performed"] is False
    assert receipt["dense_all_parent_output_columns_used"] is True
    assert receipt["fold_compute_device"] == "cpu"
    assert receipt["qdeim_compute_execution"]["requested_device"] == "cpu"
    assert receipt["qdeim_compute_execution"]["resolved_device"] == "cpu"
    assert (
        receipt["qdeim_compute_execution"]["authenticated_source_archive_device"]
        == "cpu"
    )
    assert receipt["qdeim_compute_execution"]["materialized_candidate_device"] == (
        "cpu"
    )
    assert receipt["numeric_metrics_have_semantic_decision_authority"] is False
    assert receipt["binary_manual_threshold_has_iteration_veto_authority"] is False
    assert receipt["binary_manual_acceptance_gate_required"] is False
    assert receipt["intelligence_retention_target_is_objective_not_numeric_cutoff"]
    weights = receipt["context_bank_receipt"]["row_sample_weights"]
    assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1.0])
    assert receipt["context_bank_receipt"]["phase_weight_sums"] == [
        {
            "batch_index": 0,
            "phase": "prefill",
            "phase_weight_budget": 1.0,
            "materialized_row_weight_sum": 1.0,
        },
        {
            "batch_index": 0,
            "phase": "decode.000",
            "phase_weight_budget": 1.0,
            "materialized_row_weight_sum": 1.0,
        },
    ]
    sealed = dict(receipt)
    digest = sealed.pop("result_sha256")
    assert digest == canonical_sha256(sealed)


def test_capability_and_missing_true_context_return_explicit_structural_proxy() -> None:
    structural = _contexts()
    batch = {
        "prefill.layer.000.attention.q": torch.zeros(1, 10, 4, 2),
        "prefill.layer.000.attention.context_from_probabilities": structural,
    }
    session = _Session([batch])
    capability = inspect_lfm2_semantic_attention_group_qdeim_cache(
        session=session,
        layer_index=0,
        fit_batch_indices=(0,),
    )
    assert capability["true_post_softmax_context_complete"] is False
    assert capability["nearest_structural_context_complete"] is True
    assert capability["tensor_payloads_loaded"] == 0

    result = compile_lfm2_semantic_attention_group_qdeim_from_cache(
        **_arguments(session)
    )
    assert isinstance(result, Lfm2SemanticAttentionGroupQdeimEvidenceDeficit)
    assert result.deficit_kind == "missing-captured-true-post-softmax-context"
    assert result.candidate_materialized is False
    assert torch.equal(
        result.structural_context_rows,
        structural.permute(0, 2, 1, 3).reshape(4, 10, 2),
    )
    report = result.report
    assert report["dense_qkvo_candidate_materialized"] is False
    assert report["nearest_structural_context_materialized"] is True
    assert report["binary_manual_acceptance_gate_required"] is False
    assert report["intelligence_retention_target_is_objective_not_numeric_cutoff"]
    assert report[
        "nearest_structural_context_is_not_silently_admitted_as_true_fit_context"
    ]
    methods = report["nearest_structural_context_receipt"]["phase_materializations"]
    assert methods[0]["method"] == "cached_context_from_probabilities"


def test_nearest_structural_context_can_be_derived_only_from_cached_probability_and_v() -> (
    None
):
    torch.manual_seed(31)
    logits = torch.randn(1, 10, 2, 3, dtype=torch.float64)
    probabilities = torch.softmax(logits, dim=-1)
    values = torch.randn(1, 5, 3, 2, dtype=torch.float64)
    batch = {
        "prefill.layer.000.attention.q": torch.zeros(1, 10, 2, 2),
        "prefill.layer.000.attention.probabilities": probabilities,
        "prefill.layer.000.attention.value_history": values,
    }
    session = _Session([batch])
    result = compile_lfm2_semantic_attention_group_qdeim_from_cache(
        **_arguments(session)
    )
    assert isinstance(result, Lfm2SemanticAttentionGroupQdeimEvidenceDeficit)
    expected = (
        (probabilities @ values.repeat_interleave(2, dim=1))
        .permute(0, 2, 1, 3)
        .reshape(2, 10, 2)
    )
    assert torch.allclose(result.structural_context_rows, expected)
    phase = result.report["nearest_structural_context_receipt"][
        "phase_materializations"
    ][0]
    assert phase["method"] == "probabilities_matmul_repeated_value_history"
    assert result.report["dense_qkvo_candidate_materialized"] is False


def test_true_context_rank_deficit_is_typed_and_does_not_complete_randomly() -> None:
    batch = {
        "prefill.layer.000.attention.q": torch.zeros(1, 10, 4, 2),
        "prefill.layer.000.attention.context": _contexts(rank_deficient=True),
    }
    session = _Session([batch])
    result = compile_lfm2_semantic_attention_group_qdeim_from_cache(
        **_arguments(session)
    )
    assert isinstance(result, Lfm2SemanticAttentionGroupQdeimEvidenceDeficit)
    assert result.deficit_kind == "insufficient-authenticated-true-context-group-rank"
    deficit = result.report["compiler_rank_deficit"]
    assert deficit["observed_context_group_rank"] == 1
    assert deficit["minimum_context_group_rank"] == 3
    assert deficit["random_rank_completion_permitted"] is False
    assert deficit["heldout_rank_completion_permitted"] is False
    assert result.report["dense_qkvo_candidate_materialized"] is False


def test_decoder_receipt_must_match_existing_chart_lineage() -> None:
    batch = {
        "prefill.layer.000.attention.q": torch.zeros(1, 10, 4, 2),
        "prefill.layer.000.attention.context": _contexts(),
    }
    session = _Session([batch])
    arguments = _arguments(session)
    decoder_receipt = dict(arguments["normalized_input_decoder_receipt"])
    decoder_receipt["residual_chart_sha256"] = "f" * 64
    body = dict(decoder_receipt)
    body.pop("decoder_sha256")
    decoder_receipt["decoder_sha256"] = canonical_sha256(body)
    arguments["normalized_input_decoder_receipt"] = decoder_receipt
    with pytest.raises(
        Lfm2SemanticAttentionGroupQdeimAdapterError,
        match="not bound",
    ):
        compile_lfm2_semantic_attention_group_qdeim_from_cache(**arguments)
