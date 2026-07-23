from __future__ import annotations

from dataclasses import replace
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
import neural_forge.hf_lfm2_semantic_attention_group_qdeim_installer as qdeim_installer  # noqa: E402, I001
import neural_forge.hf_lfm2_semantic_attention_group_qdeim_adapter as qdeim_adapter  # noqa: E402, I001


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
        "parent_norm_epsilon": 1e-5,
        "target_norm_epsilon": 1e-5,
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
    query_norm = torch.tensor([1.125, 0.875], dtype=torch.float64)
    key_norm = torch.tensor([0.75, 1.25], dtype=torch.float64)
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
        "source_query_norm_weight": query_norm,
        "source_key_norm_weight": key_norm,
        "expected_source_query_norm_weight_sha256": (
            lfm2_authenticated_tensor_sha256(query_norm)
        ),
        "expected_source_key_norm_weight_sha256": (
            lfm2_authenticated_tensor_sha256(key_norm)
        ),
        **templates,
        "normalized_input_decoder": decoder,
        "normalized_input_decoder_receipt": decoder_receipt,
        "residual_output_chart": chart,
        "residual_chart_receipt": chart_receipt,
    }


def _enhanced_causal_phase(prefix: str) -> dict[str, torch.Tensor]:
    epsilon = 1e-5
    query_heads = 10
    key_value_heads = 5
    head_dim = 2
    query_rows = 4
    query_norm = torch.tensor([1.125, 0.875], dtype=torch.float64)
    key_norm = torch.tensor([0.75, 1.25], dtype=torch.float64)
    q_linear = (
        torch.arange(
            query_heads * query_rows * head_dim,
            dtype=torch.float64,
        )
        .reshape(1, query_heads, query_rows, head_dim)
        .sub(17.0)
        .div(13.0)
    )
    k_linear = (
        torch.arange(
            key_value_heads * query_rows * head_dim,
            dtype=torch.float64,
        )
        .reshape(1, key_value_heads, query_rows, head_dim)
        .sub(11.0)
        .div(9.0)
    )
    q_pre = (
        q_linear
        * torch.rsqrt(q_linear.square().mean(dim=-1, keepdim=True) + epsilon)
        * query_norm
    )
    k_pre = (
        k_linear
        * torch.rsqrt(k_linear.square().mean(dim=-1, keepdim=True) + epsilon)
        * key_norm
    )
    angle = torch.linspace(0.1, 0.7, query_rows, dtype=torch.float64)
    angle = angle.reshape(1, query_rows, 1).repeat(1, 1, head_dim)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        return torch.cat((-value[..., 1:], value[..., :1]), dim=-1)

    def rope(value: torch.Tensor) -> torch.Tensor:
        rows_first = value.permute(0, 2, 1, 3)
        rotated = rows_first * cos.unsqueeze(2) + rotate_half(
            rows_first
        ) * sin.unsqueeze(2)
        return rotated.permute(0, 2, 1, 3).contiguous()

    query = rope(q_pre)
    key_history = rope(k_pre)
    value_history = (
        torch.arange(
            key_value_heads * query_rows * head_dim,
            dtype=torch.float64,
        )
        .reshape(1, key_value_heads, query_rows, head_dim)
        .add(3.0)
        .div(19.0)
    )
    mask = torch.zeros((1, 1, query_rows, query_rows), dtype=torch.float64)
    qk_scores = (
        query
        @ key_history.repeat_interleave(2, dim=1).transpose(2, 3)
        / (head_dim**0.5)
    )
    probability = torch.softmax(qk_scores + mask, dim=-1)
    return {
        "input.input_ids": torch.tensor([[11, 12, 13, 14]], dtype=torch.long),
        "input.forced_decode_ids": torch.empty((1, 0), dtype=torch.long),
        f"{prefix}.q_linear": q_linear,
        f"{prefix}.k_linear": k_linear,
        f"{prefix}.q_normalized_pre_rope": q_pre,
        f"{prefix}.k_normalized_pre_rope": k_pre,
        f"{prefix}.rope_cos": cos,
        f"{prefix}.rope_sin": sin,
        f"{prefix}.q": query,
        f"{prefix}.key_history": key_history,
        f"{prefix}.value_history": value_history,
        f"{prefix}.qk_scores": qk_scores,
        f"{prefix}.probabilities": probability,
        f"{prefix}.additive_mask": mask,
    }


def _arguments(session: _Session) -> dict[str, object]:
    return {
        "session": session,
        "layer_index": 0,
        "fit_batch_indices": (0,),
        "fit_batch_sample_weights": {0: 2.0},
        **_source_and_maps(),
    }


def _enhanced_prefill_decode_batch() -> dict[str, torch.Tensor]:
    prefill_prefix = "prefill.layer.000.attention"
    decode_prefix = "decode.000.layer.000.attention"
    batch = _enhanced_causal_phase(prefill_prefix)
    batch[f"{prefill_prefix}.context"] = _contexts()
    batch["input.forced_decode_ids"] = torch.tensor([[21]], dtype=torch.long)
    batch["decode.000.forced_token_ids"] = torch.tensor([[21]], dtype=torch.long)
    query_heads = 10
    key_value_heads = 5
    head_dim = 2
    epsilon = 1e-5
    query_norm = torch.tensor([1.125, 0.875], dtype=torch.float64)
    key_norm = torch.tensor([0.75, 1.25], dtype=torch.float64)
    q_linear = (
        torch.arange(query_heads * head_dim, dtype=torch.float64)
        .reshape(1, query_heads, 1, head_dim)
        .sub(7.0)
        .div(11.0)
    )
    k_linear = (
        torch.arange(key_value_heads * head_dim, dtype=torch.float64)
        .reshape(1, key_value_heads, 1, head_dim)
        .sub(4.0)
        .div(7.0)
    )
    q_pre = (
        q_linear
        * torch.rsqrt(q_linear.square().mean(dim=-1, keepdim=True) + epsilon)
        * query_norm
    )
    k_pre = (
        k_linear
        * torch.rsqrt(k_linear.square().mean(dim=-1, keepdim=True) + epsilon)
        * key_norm
    )
    angle = torch.full((1, 1, head_dim), 0.9, dtype=torch.float64)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        return torch.cat((-value[..., 1:], value[..., :1]), dim=-1)

    def rope(value: torch.Tensor) -> torch.Tensor:
        rows_first = value.permute(0, 2, 1, 3)
        result = rows_first * cos.unsqueeze(2) + rotate_half(
            rows_first
        ) * sin.unsqueeze(2)
        return result.permute(0, 2, 1, 3).contiguous()

    query = rope(q_pre)
    key = rope(k_pre)
    key_history = torch.cat(
        (batch[f"{prefill_prefix}.key_history"], key),
        dim=2,
    )
    decode_value = (
        torch.arange(key_value_heads * head_dim, dtype=torch.float64)
        .reshape(1, key_value_heads, 1, head_dim)
        .add(5.0)
        .div(17.0)
    )
    value_history = torch.cat(
        (batch[f"{prefill_prefix}.value_history"], decode_value),
        dim=2,
    )
    qk_scores = (
        query
        @ key_history.repeat_interleave(2, dim=1).transpose(2, 3)
        / (head_dim**0.5)
    )
    mask = torch.zeros((1, 1, 1, int(key_history.shape[2])), dtype=torch.float64)
    probability = torch.softmax(qk_scores + mask, dim=-1)
    batch.update(
        {
            f"{decode_prefix}.q_linear": q_linear,
            f"{decode_prefix}.k_linear": k_linear,
            f"{decode_prefix}.q_normalized_pre_rope": q_pre,
            f"{decode_prefix}.k_normalized_pre_rope": k_pre,
            f"{decode_prefix}.rope_cos": cos,
            f"{decode_prefix}.rope_sin": sin,
            f"{decode_prefix}.q": query,
            f"{decode_prefix}.key_history": key_history,
            f"{decode_prefix}.value_history": value_history,
            f"{decode_prefix}.qk_scores": qk_scores,
            f"{decode_prefix}.probabilities": probability,
            f"{decode_prefix}.additive_mask": mask,
            f"{decode_prefix}.context": _contexts(),
        }
    )
    return batch


def test_legacy_context_cache_returns_typed_tangent_capability_deficit() -> None:
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
    assert isinstance(result, Lfm2SemanticAttentionGroupQdeimEvidenceDeficit)
    assert result.deficit_kind == (
        "missing-authenticated-causal-attention-tangent-or-cross-phase-identity-"
        "evidence"
    )
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
    receipt = result.report
    assert receipt["dense_qkvo_candidate_materialized"] is False
    capability = receipt["cache_capability_receipt"]
    assert capability["analytic_attention_tangent_phase_bundle_complete"] is False
    assert (
        "input.input_ids"
        in capability["missing_cross_phase_stitch_identity_tensor_names"]
    )
    assert capability["cross_phase_stitch_identity_evidence_complete"] is False


def test_enhanced_fit_cache_invokes_authenticated_causal_tangents_in_real_adapter() -> (
    None
):
    prefix = "prefill.layer.000.attention"
    fit = {
        f"{prefix}.context": _contexts(),
        **_enhanced_causal_phase(prefix),
    }
    heldout = {
        f"{prefix}.context": _contexts().mul(9.0),
        **{
            name: value.mul(0.5)
            for name, value in _enhanced_causal_phase(prefix).items()
        },
    }
    session = _Session([fit, heldout])
    result = compile_lfm2_semantic_attention_group_qdeim_from_cache(
        **_arguments(session)
    )
    assert isinstance(result, Lfm2SemanticAttentionGroupQdeimCandidate)
    report = result.report
    assert report["analytic_attention_tangent_evidence_complete"] is True
    assert report["attention_tangent_bank_materialized"] is True
    assert report["functional_snapshot_block_count"] == 2
    assert report["functional_pod_group_metric"] == (
        "residual-output-balanced-empirical-and-causal-tangent"
    )
    assert report[
        "empirical_post_softmax_context_mapped_through_source_o_and_child_"
        "residual_chart_before_group_selection"
    ]
    assert report["raw_empirical_context_used_directly_for_group_selection"] is False
    residual_pod = report["residual_output_balanced_functional_pod"]
    assert residual_pod["empirical_post_softmax_context_mapped_through_source_o"]
    assert residual_pod[
        "empirical_post_softmax_context_mapped_through_residual_output_chart"
    ]
    assert (
        residual_pod["empirical_direct_feature_gram"][
            "covariance_eigendecomposition_performed"
        ]
        is False
    )
    assert (
        residual_pod["causal_tangent_balanced_direct_feature_gram"][
            "covariance_refactor_performed"
        ]
        is False
    )
    assert residual_pod["positive_finite_feature_modes_removed"] == 0
    assert residual_pod["combined_gram_within_fp64_roundoff"]
    assert report["functional_tangent_roles_bound_to_bank_reduction_receipts"]
    assert set(report["functional_tangent_perturbation_role_receipts"]) == {
        "joint_sampled_token_query_key_value",
        "older_prefix_key_value",
    }
    assert (
        report["residual_output_balanced_functional_pod_sha256"]
        == (result.compiled.report["evidence"]["functional_pod_receipt_sha256"])
    )
    assert report["missing_analytic_attention_tangent_tensor_names"] == []
    assert report["query_local_and_sampled_suffix_history_kv_tangents_separate"]
    assert report["cross_phase_older_key_anchor_stitch_implemented"] is True
    assert report["functional_tangent_blocks_are_joint_qkv_and_older_prefix_key_value"]
    tangent = report["attention_tangent_bank"]
    assert tangent["geometry"]["authenticated_child_decoder_direction_count"] == 3
    assert tangent["heldout_tensor_payloads_used"] == 0
    assert tangent["query_local_reduction_receipt"][
        "covariance_gram_within_fp64_roundoff"
    ]
    assert tangent["sampled_suffix_history_key_value_reduction_receipt"][
        "covariance_gram_within_fp64_roundoff"
    ]
    assert (
        report["functional_snapshot_evidence_sha256"]
        == result.compiled.report["authentication"][
            "functional_snapshot_evidence_sha256"
        ]
    )
    assert {index for index, _names in session.loaded} == {0}
    assert len(session.loaded) == 2


def test_typed_tangent_role_rejects_reduction_receipt_swapped_from_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "prefill.layer.000.attention"
    session = _Session(
        [{f"{prefix}.context": _contexts(), **_enhanced_causal_phase(prefix)}]
    )
    original_builder = qdeim_adapter.build_lfm2_attention_tangent_bank

    def tampered_builder(**kwargs):
        bank = original_builder(**kwargs)
        tampered_joint = replace(
            bank.joint_sampled_token_query_key_value,
            reduction_receipt=bank.older_prefix_key_value.reduction_receipt,
        )
        return replace(
            bank,
            joint_sampled_token_query_key_value=tampered_joint,
        )

    monkeypatch.setattr(
        qdeim_adapter,
        "build_lfm2_attention_tangent_bank",
        tampered_builder,
    )
    with pytest.raises(
        Lfm2SemanticAttentionGroupQdeimAdapterError,
        match="does not match its bank/reduction evidence",
    ):
        compile_lfm2_semantic_attention_group_qdeim_from_cache(**_arguments(session))


def test_cross_dtype_probability_replay_is_diagnostic_not_numeric_veto() -> None:
    prefix = "prefill.layer.000.attention"
    phase = _enhanced_causal_phase(prefix)
    probability_name = f"{prefix}.probabilities"
    phase[probability_name] = torch.full_like(
        phase[probability_name],
        1.0 / int(phase[probability_name].shape[-1]),
    )
    session = _Session([{f"{prefix}.context": _contexts(), **phase}])
    result = compile_lfm2_semantic_attention_group_qdeim_from_cache(
        **_arguments(session)
    )
    assert isinstance(result, Lfm2SemanticAttentionGroupQdeimCandidate)
    tangent = result.report["attention_tangent_bank"]
    diagnostic = tangent["cross_dtype_fp64_replay_diagnostic_summary"][
        "cached_probability_from_sealed_qk_scores_mask"
    ]
    assert diagnostic["outside_legacy_allclose_envelope_count"] == 1
    assert diagnostic["hard_authentication_authority"] is False
    assert tangent["fp64_replay_mismatch_does_not_reject_authenticated_capture"]


def test_decode_history_k_anchors_are_stitched_from_authenticated_prior_phases() -> (
    None
):
    session = _Session([_enhanced_prefill_decode_batch()])
    result = compile_lfm2_semantic_attention_group_qdeim_from_cache(
        **_arguments(session)
    )
    assert isinstance(result, Lfm2SemanticAttentionGroupQdeimCandidate)
    report = result.report
    assert report["cross_phase_older_key_anchor_stitch_required"] is True
    assert report["cross_phase_older_key_anchor_stitch_implemented"] is True
    tangent = report["attention_tangent_bank"]
    assert tangent["complete_key_history_tangent_coverage"] is True
    assert tangent["unsupported_historical_prefix_key_position_count"] == 0
    assert tangent["cross_phase_stitched_historical_prefix_key_position_count"] == 4
    loader = report["attention_tangent_fit_loader_receipt"]
    assert loader["cross_phase_k_anchor_stitch_implemented"] is True
    decode_source = next(
        value for value in loader["phase_sources"] if value["phase"] == "decode.000"
    )
    stitch = decode_source["cross_phase_k_anchor_stitch"]
    assert stitch["history_count"] == 5
    assert len(stitch["ordered_contributing_anchor_source_sha256"]) == 2
    assert stitch["phase_order_and_forced_token_identity_verified"] is True


def test_cross_phase_stitch_rejects_forced_token_identity_tamper() -> None:
    batch = _enhanced_prefill_decode_batch()
    batch["decode.000.forced_token_ids"] = torch.tensor([[22]], dtype=torch.long)
    session = _Session([batch])
    with pytest.raises(
        Lfm2SemanticAttentionGroupQdeimAdapterError,
        match="phase/token identity is not contiguous",
    ):
        compile_lfm2_semantic_attention_group_qdeim_from_cache(**_arguments(session))


class _InstallerConfig:
    model_type = "lfm2"
    norm_eps = 1e-5

    def to_dict(self) -> dict[str, object]:
        return {"model_type": self.model_type, "norm_eps": self.norm_eps}


class _InstallerNorm(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(width, dtype=torch.bfloat16))


class _InstallerAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head_dim = 2
        self.q_proj = torch.nn.Linear(3, 12, bias=False, dtype=torch.bfloat16)
        self.k_proj = torch.nn.Linear(3, 6, bias=False, dtype=torch.bfloat16)
        self.v_proj = torch.nn.Linear(3, 6, bias=False, dtype=torch.bfloat16)
        self.out_proj = torch.nn.Linear(12, 3, bias=False, dtype=torch.bfloat16)
        self.q_layernorm = _InstallerNorm(2)
        self.k_layernorm = _InstallerNorm(2)
        with torch.no_grad():
            self.q_proj.weight.zero_()
            self.k_proj.weight.zero_()
            self.v_proj.weight.zero_()
            self.out_proj.weight.zero_()


class _InstallerLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _InstallerAttention()


class _InstallerBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_InstallerLayer()])


class _InstallerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _InstallerConfig()
        self.model = _InstallerBackbone()

    def forward(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("The analytic installer must not execute the child model.")


def test_enhanced_cache_runs_real_compile_and_six_parameter_installer() -> None:
    prefix = "prefill.layer.000.attention"
    session = _Session(
        [
            {
                f"{prefix}.context": _contexts(),
                **_enhanced_causal_phase(prefix),
            }
        ]
    )
    maps = _source_and_maps()
    source_q = maps["source_query_weight"].to(torch.bfloat16)
    source_k = maps["source_key_weight"].to(torch.bfloat16)
    source_v = maps["source_value_weight"].to(torch.bfloat16)
    source_o = maps["source_out_weight"].to(torch.bfloat16)
    source_inventory = seal_lfm2_dense_qkvo_source_inventory(
        source_query_weight=source_q,
        source_key_weight=source_k,
        source_value_weight=source_v,
        source_output_weight=source_o,
    )
    query_norm = torch.tensor([1.125, 0.875], dtype=torch.bfloat16)
    key_norm = torch.tensor([0.75, 1.25], dtype=torch.bfloat16)
    norm_inventory = (
        qdeim_installer.seal_lfm2_semantic_attention_group_qdeim_norm_inventory(
            layer_index=0,
            source_parent_identity_sha256="d" * 64,
            source_inventory=source_inventory,
            source_query_norm_weight=query_norm,
            source_key_norm_weight=key_norm,
        )
    )
    model = _InstallerModel()
    result = qdeim_installer.compile_install_lfm2_semantic_attention_group_qdeim(
        target_model=model,
        session=session,
        layer_index=0,
        fit_batch_indices=(0,),
        fit_batch_sample_weights={0: 1.0},
        source_query_weight=source_q,
        source_key_weight=source_k,
        source_value_weight=source_v,
        source_out_weight=source_o,
        source_inventory=source_inventory,
        source_query_norm_weight=query_norm,
        source_key_norm_weight=key_norm,
        source_norm_inventory=norm_inventory,
        normalized_input_decoder=maps["normalized_input_decoder"],
        normalized_input_decoder_receipt=maps["normalized_input_decoder_receipt"],
        residual_output_chart=maps["residual_output_chart"],
        residual_chart_receipt=maps["residual_chart_receipt"],
    )
    receipt = result.receipt
    assert receipt["status"] == "installed-coupled-intact-group-qdeim-attention"
    assert len(receipt["changed_state_keys"]) == 6
    assert result.prepared.candidate.report["attention_tangent_bank_materialized"]
    assert result.prepared.candidate.report["functional_snapshot_block_count"] == 2
    attention = model.model.layers[0].self_attn
    for value in (
        attention.q_proj.weight,
        attention.k_proj.weight,
        attention.v_proj.weight,
        attention.out_proj.weight,
        attention.q_layernorm.weight,
        attention.k_layernorm.weight,
    ):
        assert int(torch.count_nonzero(value).item()) > 0


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


def test_cached_probability_row_sum_drift_is_normalized_not_hard_gated() -> None:
    torch.manual_seed(37)
    logits = torch.randn(1, 10, 2, 3, dtype=torch.float64)
    normalized_probabilities = torch.softmax(logits, dim=-1)
    cached_probabilities = normalized_probabilities * 1.01
    values = torch.randn(1, 5, 3, 2, dtype=torch.float64)
    batch = {
        "prefill.layer.000.attention.q": torch.zeros(1, 10, 2, 2),
        "prefill.layer.000.attention.probabilities": cached_probabilities,
        "prefill.layer.000.attention.value_history": values,
    }
    result = compile_lfm2_semantic_attention_group_qdeim_from_cache(
        **_arguments(_Session([batch]))
    )
    assert isinstance(result, Lfm2SemanticAttentionGroupQdeimEvidenceDeficit)
    expected = (
        (normalized_probabilities @ values.repeat_interleave(2, dim=1))
        .permute(0, 2, 1, 3)
        .reshape(2, 10, 2)
    )
    assert torch.allclose(result.structural_context_rows, expected)
    phase = result.report["nearest_structural_context_receipt"][
        "phase_materializations"
    ][0]
    assert phase["cached_probability_row_normalization_applied"] is True
    assert (
        phase["cached_probability_row_sum_max_absolute_deviation_before_normalization"]
        > 0.009
    )
    assert (
        phase["cached_probability_row_sum_diagnostics_have_no_acceptance_authority"]
        is True
    )


def test_missing_tangent_precedes_context_rank_materialization() -> None:
    batch = {
        "prefill.layer.000.attention.q": torch.zeros(1, 10, 4, 2),
        "prefill.layer.000.attention.context": _contexts(rank_deficient=True),
    }
    session = _Session([batch])
    result = compile_lfm2_semantic_attention_group_qdeim_from_cache(
        **_arguments(session)
    )
    assert isinstance(result, Lfm2SemanticAttentionGroupQdeimEvidenceDeficit)
    assert result.deficit_kind == (
        "missing-authenticated-causal-attention-tangent-or-cross-phase-identity-"
        "evidence"
    )
    assert result.report["compiler_rank_deficit"] is None
    assert result.report["dense_qkvo_candidate_materialized"] is False


def test_residual_output_rank_deficit_is_not_mislabeled_as_true_context_rank() -> None:
    prefix = "prefill.layer.000.attention"
    captured_context = _contexts()
    raw_rows = captured_context.permute(0, 2, 1, 3).reshape(4, 10, 2)
    raw_group_features = raw_rows.reshape(4, 5, 2, 2).permute(0, 2, 3, 1).reshape(16, 5)
    assert int(torch.linalg.matrix_rank(raw_group_features).item()) >= 3

    session = _Session(
        [{f"{prefix}.context": captured_context, **_enhanced_causal_phase(prefix)}]
    )
    arguments = _arguments(session)
    source_o = arguments["source_out_weight"].clone()
    source_o[:, 8:] = 0.0
    arguments["source_out_weight"] = source_o
    arguments["source_inventory"] = seal_lfm2_dense_qkvo_source_inventory(
        source_query_weight=arguments["source_query_weight"],
        source_key_weight=arguments["source_key_weight"],
        source_value_weight=arguments["source_value_weight"],
        source_output_weight=source_o,
    )

    result = compile_lfm2_semantic_attention_group_qdeim_from_cache(**arguments)
    assert isinstance(result, Lfm2SemanticAttentionGroupQdeimEvidenceDeficit)
    assert result.deficit_kind == (
        "insufficient-authenticated-residual-output-functional-group-rank"
    )
    compiler = result.report["compiler_rank_deficit"]
    assert compiler["status"] == (
        "insufficient-authenticated-residual-output-functional-group-rank"
    )
    assert compiler["functional_pod_group_metric"] == (
        "residual-output-balanced-empirical-and-causal-tangent"
    )
    assert compiler["observed_context_group_rank"] <= 2


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
