from __future__ import annotations

import ast
import hashlib
import json
import math
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_mlp_refold_adapter as adapter  # noqa: E402
import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack_adapter  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_mlp_qdeim import (  # noqa: E402
    Lfm2MlpQdeimInsufficientRankError,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _completion_tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


class _Config:
    model_type = "lfm2"
    norm_eps = 1e-5

    def to_dict(self):
        return {
            "model_type": "lfm2",
            "hidden_size": 4,
            "intermediate_size": 3,
            "num_hidden_layers": 1,
            "norm_eps": self.norm_eps,
        }


class _Mlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
        self.w3 = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
        self.w2 = torch.nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.w1.weight.copy_(
                torch.tensor(
                    [
                        [0.50, -0.25, 0.10, 0.20],
                        [-0.20, 0.40, 0.30, -0.10],
                        [0.15, 0.05, -0.35, 0.45],
                    ],
                    dtype=torch.bfloat16,
                )
            )
            self.w3.weight.copy_(
                torch.tensor(
                    [
                        [0.30, 0.10, -0.20, 0.40],
                        [0.25, -0.30, 0.45, 0.15],
                        [-0.10, 0.35, 0.20, 0.50],
                    ],
                    dtype=torch.bfloat16,
                )
            )
            self.w2.weight.copy_(
                torch.tensor(
                    [
                        [0.30, -0.20, 0.10],
                        [-0.10, 0.25, 0.35],
                        [0.20, 0.15, -0.30],
                        [0.05, -0.40, 0.25],
                    ],
                    dtype=torch.bfloat16,
                )
            )


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ffn_norm = torch.nn.Module()
        self.ffn_norm.weight = torch.nn.Parameter(
            torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.bfloat16)
        )
        self.feed_forward = _Mlp()


class _Inner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])


class _Target(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.model = _Inner()


def _source() -> adapter.Lfm2DirectLocalMlpSourceInventory:
    gate = torch.tensor(
        [
            [0.20, -0.10, 0.30, 0.15],
            [-0.25, 0.35, 0.10, -0.20],
            [0.40, 0.05, -0.15, 0.25],
            [0.10, 0.30, 0.20, -0.35],
            [-0.30, 0.20, 0.45, 0.05],
        ],
        dtype=torch.bfloat16,
    )
    up = torch.tensor(
        [
            [0.10, 0.25, -0.20, 0.35],
            [0.30, -0.15, 0.40, 0.10],
            [-0.20, 0.45, 0.15, -0.10],
            [0.35, 0.05, -0.30, 0.20],
            [0.15, -0.25, 0.30, 0.40],
        ],
        dtype=torch.bfloat16,
    )
    down = torch.tensor(
        [
            [0.20, -0.10, 0.30, 0.15, -0.20],
            [-0.15, 0.25, 0.05, 0.35, 0.10],
            [0.30, 0.20, -0.25, 0.10, 0.15],
            [0.05, -0.30, 0.20, 0.25, 0.40],
        ],
        dtype=torch.bfloat16,
    )
    return adapter.Lfm2DirectLocalMlpSourceInventory(
        layer_index=0,
        parent_model_identity_sha256=_sha("parent"),
        parent_checkpoint_manifest_sha256=_sha("manifest"),
        direct_local_loader_receipt_sha256=_sha("loader"),
        residual_assignment_sha256=_sha("residual-chart"),
        compiler_lineage_sha256=_sha("compiler-lineage"),
        residual_chart=torch.eye(4, dtype=torch.float64),
        ffn_norm_weight=torch.tensor(
            [1.0, 0.9, 1.1, 0.8],
            dtype=torch.bfloat16,
        ),
        norm_epsilon=1e-5,
        gate_weight=gate,
        up_weight=up,
        down_weight=down,
    )


def _parent_normalized(
    source: adapter.Lfm2DirectLocalMlpSourceInventory,
    parent_pre_mlp_residual: torch.Tensor,
) -> torch.Tensor:
    return adapter._reconstruct_parent_rmsnorm_bf16(
        parent_pre_mlp_residual,
        norm_weight=source.ffn_norm_weight,
        norm_epsilon=source.norm_epsilon,
    )


def _parent_pair_for_desired_normalized(
    source: adapter.Lfm2DirectLocalMlpSourceInventory,
    desired_normalized: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    quotient = desired_normalized.to(torch.float64) / source.ffn_norm_weight.to(
        torch.float64
    )
    mean_square = float(quotient.square().mean().item())
    assert mean_square < 1.0
    scale = math.sqrt(source.norm_epsilon / (1.0 - mean_square))
    parent_pre = (quotient * scale).to(torch.bfloat16)
    return parent_pre, _parent_normalized(source, parent_pre)


def test_source_inventory_authenticates_dense_residual_chart_transport() -> None:
    source = _source()
    body = source.authentication_body()

    assert source.verify() is True
    assert source.target_hidden_size == 4
    assert body["schema"] == adapter.LFM2_SEMANTIC_MLP_SOURCE_SCHEMA
    assert body["residual_chart_shape"] == [4, 4]
    assert body["residual_chart_sha256"] == adapter._tensor_sha256(
        source.residual_chart
    )
    assert body["residual_chart_receipt_sha256"] == source.residual_assignment_sha256
    assert body["residual_chart_and_receipt_jointly_hashed"] is True
    assert body["ffn_norm_weight_sha256"] == adapter._tensor_sha256(
        source.ffn_norm_weight
    )
    assert body["norm_epsilon"] == pytest.approx(1e-5)
    assert body["normalization_lineage_sha256"] == (source.normalization_lineage_sha256)
    assert body["normalization_and_loader_lineage_jointly_hashed"] is True
    with pytest.raises(
        adapter.Lfm2SemanticMlpRefoldError,
        match="source-hidden-to-target-hidden",
    ):
        replace(source, residual_chart=torch.ones((3, 4), dtype=torch.float64))
    nonfinite_chart = torch.eye(4, dtype=torch.float64)
    nonfinite_chart[0, 0] = float("nan")
    with pytest.raises(
        adapter.Lfm2SemanticMlpRefoldError,
        match="detached finite rank-2",
    ):
        replace(source, residual_chart=nonfinite_chart)

    source.residual_chart[0, 0] += 1.0
    assert source.verify() is False

    with pytest.raises(
        adapter.Lfm2SemanticMlpRefoldError,
        match="cannot be serialized with Python pickle",
    ):
        pickle.dumps(_source())


def test_source_inventory_rejects_or_detects_normalization_tampering() -> None:
    source = _source()
    with pytest.raises(
        adapter.Lfm2SemanticMlpRefoldError,
        match="positive norm epsilon",
    ):
        replace(source, norm_epsilon=0.0)
    with pytest.raises(
        adapter.Lfm2SemanticMlpRefoldError,
        match="positive norm epsilon",
    ):
        replace(
            source,
            ffn_norm_weight=torch.ones((3,), dtype=torch.bfloat16),
        )
    with pytest.raises(
        adapter.Lfm2SemanticMlpRefoldError,
        match="source-hidden-to-target-hidden",
    ):
        replace(
            source,
            ffn_norm_weight=torch.tensor(
                [1.0, 0.0, 1.0, 1.0],
                dtype=torch.bfloat16,
            ),
        )

    source.ffn_norm_weight[0] += torch.tensor(0.25, dtype=torch.bfloat16)
    assert source.verify() is False


def test_endpoint_authenticates_parent_normalized_mlp_input() -> None:
    valid = _evidence(_Target())[0]
    body = valid.authentication_body()
    assert body["tensor_sha256"]["parent_normalized_mlp_input"] == (
        adapter._tensor_sha256(valid.parent_normalized_mlp_input)
    )
    assert body["normalization_lineage_sha256"] == (
        valid.source_inventory.normalization_lineage_sha256
    )
    assert body["parent_normalized_mlp_input_is_exact_cached_mlp_input"] is True
    assert (
        body["parent_normalized_mlp_input_reconstruction_independently_replayed"]
        is True
    )
    assert body["normalization_reconstruction_diagnostic_only"] is True
    assert body["normalization_reconstruction_numeric_admission_authority"] is False
    assert body["parent_model_forward_calls_for_normalized_mlp_input"] == 0
    assert body["tensor_sha256"]["parent_pre_mlp_residual"] == (
        adapter._tensor_sha256(valid.parent_pre_mlp_residual)
    )
    assert len(body["normalization_reconstruction_sha256"]) == 64
    with pytest.raises(
        adapter.Lfm2SemanticMlpRefoldError,
        match="compatible endpoint",
    ):
        replace(
            valid,
            parent_normalized_mlp_input=torch.ones(
                (1, 3),
                dtype=torch.bfloat16,
            ),
        )
    drifted = replace(
        valid,
        parent_normalized_mlp_input=torch.zeros_like(valid.parent_normalized_mlp_input),
    )
    assert drifted.verify() is True
    assert drifted.normalization_reconstruction_within_legacy_envelope is False
    assert (
        drifted.authentication_body()[
            "normalization_reconstruction_numeric_admission_authority"
        ]
        is False
    )

    with pytest.raises(
        adapter.Lfm2SemanticMlpRefoldError,
        match="cannot be serialized with Python pickle",
    ):
        pickle.dumps(valid)

    valid.parent_normalized_mlp_input[0, 0] += torch.tensor(
        0.25,
        dtype=torch.bfloat16,
    )
    assert valid.verify() is False


def test_nonfinite_rmsnorm_kernel_replay_is_a_finite_nonveto_diagnostic() -> None:
    replayed = torch.tensor([[float("inf"), float("nan")]], dtype=torch.bfloat16)
    cached = torch.zeros_like(replayed)

    metrics = adapter._bf16_reconstruction_metrics(replayed, cached)

    assert metrics["diagnostics_finite"] is True
    assert metrics["replayed_all_finite"] is False
    assert metrics["nonfinite_replayed_element_count"] == 2
    assert metrics["within_legacy_bf16_envelope"] is False
    assert metrics["diagnostic_only"] is True
    assert metrics["numeric_admission_authority"] is False


def test_endpoint_rejects_residual_chart_with_wrong_target_width() -> None:
    model = _Target()
    valid = _evidence(model)[0]
    wrong_width = replace(
        valid.source_inventory,
        residual_chart=torch.ones((4, 3), dtype=torch.float64),
    )
    assert wrong_width.verify()
    with pytest.raises(
        adapter.Lfm2SemanticMlpRefoldError,
        match="compatible endpoint",
    ):
        replace(valid, source_inventory=wrong_width)


def test_mlp_parameter_bundle_rejects_mixed_target_dtypes() -> None:
    model = _Target().to(torch.float32)
    mlp = model.model.layers[0].feed_forward
    mlp.w3.weight = torch.nn.Parameter(mlp.w3.weight.detach().to(torch.bfloat16))

    with pytest.raises(
        adapter.Lfm2SemanticMlpRefoldError,
        match="share one BF16 or FP32 dtype",
    ):
        adapter._parameter_bundle(model, 0)


def _evidence(
    model: _Target,
    *,
    batches: int = 4,
    total_phases: int | None = None,
    terminal: bool = True,
) -> tuple[adapter.Lfm2SemanticMlpEndpointEvidence, ...]:
    source = _source()
    state = fingerprint_lfm2_parent(model).identity_sha256
    mlp = model.model.layers[0].feed_forward
    target_dtype = mlp.w1.weight.dtype
    child_input = torch.tensor([[0.75, -0.50, 0.25, 1.00]], dtype=target_dtype)
    child_pre = torch.tensor([[0.20, -0.10, 0.30, 0.05]], dtype=target_dtype)
    desired_down = mlp.w2.weight.detach().to(torch.float64) + torch.tensor(
        [
            [0.08, -0.02, 0.04],
            [-0.03, 0.05, 0.06],
            [0.04, 0.03, -0.05],
            [0.02, -0.06, 0.05],
        ],
        dtype=torch.float64,
    )
    inputs64 = child_input.to(torch.float64)
    child_post = torch.nn.functional.silu(
        inputs64 @ mlp.w1.weight.detach().to(torch.float64).T
    ) * (inputs64 @ mlp.w3.weight.detach().to(torch.float64).T)
    desired_endpoint = (child_pre.to(torch.float64) + child_post @ desired_down.T).to(
        target_dtype
    )
    desired_parent_input = torch.tensor(
        [[0.40, -0.20, 0.60, 0.10]],
        dtype=torch.bfloat16,
    )
    parent_pre, parent_input = _parent_pair_for_desired_normalized(
        source,
        desired_parent_input,
    )
    parent_gate = (
        parent_input.to(torch.float64) @ source.gate_weight.to(torch.float64).T
    ).to(torch.bfloat16)
    parent_up = (
        parent_input.to(torch.float64) @ source.up_weight.to(torch.float64).T
    ).to(torch.bfloat16)
    parent_post = (
        torch.nn.functional.silu(parent_gate.to(torch.float64))
        * parent_up.to(torch.float64)
    ).to(torch.bfloat16)
    if total_phases is None:
        per_batch = [1] * batches
    else:
        quotient, remainder = divmod(total_phases, batches)
        per_batch = [quotient + int(index < remainder) for index in range(batches)]
    values = []
    phase_global = 0
    for batch_index, phase_count in enumerate(per_batch):
        for phase_index in range(phase_count):
            values.append(
                adapter.Lfm2SemanticMlpEndpointEvidence(
                    batch_index=batch_index,
                    trajectory_sha256=_sha(f"trajectory:{batch_index}"),
                    phase_index=phase_index,
                    phase_sha256=_sha(f"phase:{batch_index}:{phase_index}"),
                    layer_index=0,
                    alignment_sha256=_sha(f"alignment:{batch_index}:{phase_index}"),
                    cache_key=_sha("cache"),
                    cache_manifest_sha256=_sha("cache-manifest"),
                    bridge_artifact_sha256=_sha("bridge"),
                    captured_child_model_state_sha256=state,
                    source_inventory=source,
                    parent_gate=parent_gate,
                    parent_up=parent_up,
                    parent_post_swiglu=parent_post,
                    parent_pre_mlp_residual=parent_pre,
                    parent_normalized_mlp_input=parent_input,
                    actual_child_mlp_input=child_input,
                    actual_child_pre_mlp_residual=child_pre,
                    parent_post_mlp_residual_target_space=desired_endpoint,
                    parent_final_norm_output_target_space=(
                        desired_endpoint if terminal else None
                    ),
                )
            )
            phase_global += 1
    assert phase_global == (total_phases if total_phases is not None else batches)
    return tuple(values)


def _prepare(
    model: _Target,
    evidence: tuple[adapter.Lfm2SemanticMlpEndpointEvidence, ...],
    *,
    terminal: bool = False,
    maximum_delta_ratio: float | None = 1.0,
    body_recipe: str = "fit_guided",
    fold_compute_device: str = "cpu",
) -> adapter.Lfm2PreparedSemanticMlpRefold:
    return adapter.prepare_lfm2_semantic_mlp_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key=_sha("cache"),
        cache_manifest_sha256=_sha("cache-manifest"),
        bridge_artifact_sha256=_sha("bridge"),
        maximum_delta_ratio=maximum_delta_ratio,
        rmse_relative_tolerance=1e-4,
        cosine_tolerance=1e-5,
        require_terminal_action_evidence=terminal,
        body_recipe=body_recipe,
        fold_compute_device=fold_compute_device,
    )


def test_prepare_rejects_zero_child_ffn_norm_coordinate() -> None:
    model = _Target()
    with torch.no_grad():
        model.model.layers[0].ffn_norm.weight[0] = 0.0
    with pytest.raises(
        adapter.Lfm2SemanticMlpRefoldError,
        match="FFN RMSNorm Parameter",
    ):
        _prepare(model, _evidence(model))


def _material_full_win_case(model: _Target):
    """Build repeated heldouts where W2 helps but changed producers help more."""

    source = _source()
    state_sha = fingerprint_lfm2_parent(model).identity_sha256
    mlp = model.model.layers[0].feed_forward
    desired = {
        "w1.weight": (
            mlp.w1.weight.detach().to(torch.float32)
            + torch.tensor(
                [
                    [0.12, -0.08, 0.06, 0.04],
                    [-0.05, 0.10, 0.03, -0.07],
                    [0.08, 0.02, -0.09, 0.11],
                ]
            )
        ).to(torch.bfloat16),
        "w3.weight": (
            mlp.w3.weight.detach().to(torch.float32)
            + torch.tensor(
                [
                    [-0.07, 0.09, 0.04, -0.05],
                    [0.08, 0.03, -0.06, 0.10],
                    [-0.04, 0.07, 0.12, -0.08],
                ]
            )
        ).to(torch.bfloat16),
        "w2.weight": (
            mlp.w2.weight.detach().to(torch.float32)
            + torch.tensor(
                [
                    [0.03, -0.02, 0.01],
                    [-0.01, 0.02, 0.03],
                    [0.02, 0.01, -0.025],
                    [0.01, -0.03, 0.02],
                ]
            )
        ).to(torch.bfloat16),
    }
    inputs = (
        torch.tensor([[0.75, -0.50, 0.25, 1.00]], dtype=torch.bfloat16),
        torch.tensor([[-0.40, 0.80, 0.50, -0.20]], dtype=torch.bfloat16),
        torch.tensor([[0.30, 0.20, -0.90, 0.70]], dtype=torch.bfloat16),
        torch.tensor([[1.00, 0.40, 0.10, -0.60]], dtype=torch.bfloat16),
    )
    child_pre = torch.tensor([[0.20, -0.10, 0.30, 0.05]], dtype=torch.bfloat16)
    values = []
    for batch_index in range(8):
        child_input = inputs[batch_index % len(inputs)]
        parent_pre, parent_input = _parent_pair_for_desired_normalized(
            source,
            child_input,
        )
        input64 = child_input.to(torch.float64)
        desired_post = torch.nn.functional.silu(
            input64 @ desired["w1.weight"].to(torch.float64).T
        ) * (input64 @ desired["w3.weight"].to(torch.float64).T)
        endpoint = (
            child_pre.to(torch.float64)
            + desired_post @ desired["w2.weight"].to(torch.float64).T
        ).to(torch.bfloat16)
        parent_input64 = parent_input.to(torch.float64)
        parent_gate = (parent_input64 @ source.gate_weight.to(torch.float64).T).to(
            torch.bfloat16
        )
        parent_up = (parent_input64 @ source.up_weight.to(torch.float64).T).to(
            torch.bfloat16
        )
        parent_post = (
            torch.nn.functional.silu(parent_gate.to(torch.float64))
            * parent_up.to(torch.float64)
        ).to(torch.bfloat16)
        values.append(
            adapter.Lfm2SemanticMlpEndpointEvidence(
                batch_index=batch_index,
                trajectory_sha256=_sha(f"full-win-trajectory:{batch_index}"),
                phase_index=0,
                phase_sha256=_sha(f"full-win-phase:{batch_index}"),
                layer_index=0,
                alignment_sha256=_sha(f"full-win-alignment:{batch_index}"),
                cache_key=_sha("cache"),
                cache_manifest_sha256=_sha("cache-manifest"),
                bridge_artifact_sha256=_sha("bridge"),
                captured_child_model_state_sha256=state_sha,
                source_inventory=source,
                parent_gate=parent_gate,
                parent_up=parent_up,
                parent_post_swiglu=parent_post,
                parent_pre_mlp_residual=parent_pre,
                parent_normalized_mlp_input=parent_input,
                actual_child_mlp_input=child_input,
                actual_child_pre_mlp_residual=child_pre,
                parent_post_mlp_residual_target_space=endpoint,
            )
        )
    return tuple(values), desired


def _cold_start_parent_endpoint_evidence(
    model: _Target,
) -> tuple[adapter.Lfm2SemanticMlpEndpointEvidence, ...]:
    source = _source()
    state_sha = fingerprint_lfm2_parent(model).identity_sha256
    target_dtype = model.model.layers[0].feed_forward.w1.weight.dtype
    inputs = torch.tensor(
        [
            [0.75, -0.50, 0.25, 1.00],
            [-0.40, 0.80, 0.50, -0.20],
            [0.30, 0.20, -0.90, 0.70],
            [1.00, 0.40, 0.10, -0.60],
            [-0.70, -0.30, 0.80, 0.20],
            [0.20, 1.00, -0.40, 0.60],
            [0.60, -0.80, -0.10, 0.50],
            [-0.20, 0.30, 0.90, -0.70],
        ],
        dtype=target_dtype,
    )
    child_pre = torch.tensor(
        [[0.20, -0.10, 0.30, 0.05]],
        dtype=target_dtype,
    )
    values = []
    for batch_index in range(4):
        for phase_index, input_row in enumerate(inputs):
            child_input = input_row.unsqueeze(0)
            parent_pre, parent_input = _parent_pair_for_desired_normalized(
                source,
                child_input,
            )
            parent_input64 = parent_input.to(torch.float64)
            parent_gate = parent_input64 @ source.gate_weight.to(torch.float64).T
            parent_up = parent_input64 @ source.up_weight.to(torch.float64).T
            parent_post = torch.nn.functional.silu(parent_gate) * parent_up
            parent_endpoint = (
                child_pre.to(torch.float64)
                + parent_post @ source.down_weight.to(torch.float64).T
            )
            values.append(
                adapter.Lfm2SemanticMlpEndpointEvidence(
                    batch_index=batch_index,
                    trajectory_sha256=_sha(f"cold-trajectory:{batch_index}"),
                    phase_index=phase_index,
                    phase_sha256=_sha(f"cold-phase:{batch_index}:{phase_index}"),
                    layer_index=0,
                    alignment_sha256=_sha(
                        f"cold-alignment:{batch_index}:{phase_index}"
                    ),
                    cache_key=_sha("cache"),
                    cache_manifest_sha256=_sha("cache-manifest"),
                    bridge_artifact_sha256=_sha("bridge"),
                    captured_child_model_state_sha256=state_sha,
                    source_inventory=source,
                    parent_gate=parent_gate.to(torch.bfloat16),
                    parent_up=parent_up.to(torch.bfloat16),
                    parent_post_swiglu=parent_post.to(torch.bfloat16),
                    parent_pre_mlp_residual=parent_pre,
                    parent_normalized_mlp_input=parent_input,
                    actual_child_mlp_input=child_input,
                    actual_child_pre_mlp_residual=child_pre,
                    parent_post_mlp_residual_target_space=(
                        parent_endpoint.to(target_dtype)
                    ),
                )
            )
    return tuple(values)


def test_typed_source_and_endpoint_seals_reject_inserted_rows() -> None:
    model = _Target()
    source = _source()
    assert source.verify()
    assert source.authentication_body()["loader_mode"] == "direct-local-safetensors"
    valid = _evidence(model)[0]
    assert valid.verify()
    assert valid.authentication_body()["parent_exact_semantic_endpoint"] is True
    with pytest.raises(adapter.Lfm2SemanticMlpRefoldError, match="noninserted"):
        adapter.Lfm2SemanticMlpEndpointEvidence(
            batch_index=valid.batch_index,
            trajectory_sha256=valid.trajectory_sha256,
            phase_index=valid.phase_index,
            phase_sha256=valid.phase_sha256,
            layer_index=valid.layer_index,
            alignment_sha256=valid.alignment_sha256,
            cache_key=valid.cache_key,
            cache_manifest_sha256=valid.cache_manifest_sha256,
            bridge_artifact_sha256=valid.bridge_artifact_sha256,
            captured_child_model_state_sha256=(valid.captured_child_model_state_sha256),
            source_inventory=valid.source_inventory,
            parent_gate=valid.parent_gate,
            parent_up=valid.parent_up,
            parent_post_swiglu=valid.parent_post_swiglu,
            parent_pre_mlp_residual=valid.parent_pre_mlp_residual,
            parent_normalized_mlp_input=valid.parent_normalized_mlp_input,
            actual_child_mlp_input=valid.actual_child_mlp_input,
            actual_child_pre_mlp_residual=valid.actual_child_pre_mlp_residual,
            parent_post_mlp_residual_target_space=(
                valid.parent_post_mlp_residual_target_space
            ),
            inserted_target_row=True,
        )


def test_prepare_freezes_two_fit_only_candidates_and_selects_by_fit_order() -> None:
    model = _Target()
    before = fingerprint_lfm2_parent(model).identity_sha256
    prepared = _prepare(model, _evidence(model))
    assert prepared.verify()
    assert fingerprint_lfm2_parent(model).identity_sha256 == before
    assert prepared.selected_role == "full-w1-w3-w2"
    assert prepared.writer_candidate.solver_receipt["cluster_solve"] is False
    assert prepared.full_candidate.solver_receipt["cluster_solve"] is True
    assert prepared.full_candidate.solver_receipt["full_source_coverage"] is True
    assert prepared.selection_receipt["fit_partition_only_used_for_solves"] is True
    assert prepared.selection_receipt["fit_ordering_keys"][
        prepared.selected_role
    ] == min(prepared.selection_receipt["fit_ordering_keys"].values())
    metrics = prepared.selection_receipt["endpoint_metrics"]
    for role in ("incumbent", "incumbent-producer-w2", "full-w1-w3-w2"):
        for partition in ("fit", "fit_check", "selection", "validation"):
            assert metrics[role][partition]["endpoint_rows"] > 0
            assert "endpoint_rmse" in metrics[role][partition]
            assert "endpoint_maximum_absolute_error" in metrics[role][partition]
            assert "mean_endpoint_cosine" in metrics[role][partition]
    heldout = prepared.selection_receipt["heldout_nonregression"][
        "incumbent-producer-w2"
    ]
    assert heldout == {
        "fit_check": True,
        "selection": True,
        "validation": True,
    }


def test_exact_zero_incumbent_uses_unclipped_authenticated_structural_escape() -> None:
    model = _Target()
    mlp = model.model.layers[0].feed_forward
    with torch.no_grad():
        for parameter in mlp.parameters():
            parameter.zero_()
    before = fingerprint_lfm2_parent(model).identity_sha256
    prepared = _prepare(model, _cold_start_parent_endpoint_evidence(model))

    assert prepared.verify()
    assert fingerprint_lfm2_parent(model).identity_sha256 == before
    assert prepared.selected_role == "full-w1-w3-w2"
    assert prepared.selection_receipt["full_changed_roles"] == [
        "w1.weight",
        "w3.weight",
        "w2.weight",
    ]
    assert all(
        int(torch.count_nonzero(value).item()) > 0
        for value in (
            prepared.full_candidate.gate_weight,
            prepared.full_candidate.up_weight,
            prepared.full_candidate.down_weight,
        )
    )
    cold = prepared.full_candidate.solver_receipt["cold_start_escape"]
    assert cold == {
        "incumbent_w1_w3_w2_exactly_zero": True,
        "requested_maximum_delta_ratio": 1.0,
        "effective_full_structural_maximum_delta_ratio": None,
        "relative_incumbent_trust_cap_applied": False,
        "authenticated_full_source_structural_solve_used": True,
        "authenticated_normalized_input_bridge_used_for_w1_w3": True,
        "authenticated_raw_residual_chart_used_only_for_w2": True,
        "producer_consumer_bridge_fields_separate": True,
        "coupled_real_medoid_gate_up_used": True,
        "shared_normalized_input_decoder_used": True,
        "per_neuron_producer_interpolation_used": False,
        "every_source_multi_anchor_consumer_used": True,
        "materialized_target_dtype_producers_used_for_w2": True,
        "random_initialization_used": False,
        "gradient_training_used": False,
    }
    assert prepared.selection_receipt["cold_start_escape"] == cold
    planting = prepared.full_candidate.solver_receipt["source_derived_planting"]
    solver = prepared.full_candidate.solver_receipt
    assert planting["source_inventory_sha256"] == _source().inventory_sha256
    assert planting["compiler_lineage_sha256"] == _sha("compiler-lineage")
    assert planting["residual_assignment_sha256"] == _sha("residual-chart")
    assert planting["normalization_lineage_sha256"] == (
        _source().normalization_lineage_sha256
    )
    assert planting["residual_chart_sha256"] == adapter._tensor_sha256(
        _source().residual_chart
    )
    assert planting["parent_ffn_norm_weight_sha256"] == adapter._tensor_sha256(
        _source().ffn_norm_weight
    )
    assert planting["target_ffn_norm_weight_sha256"] == adapter._tensor_sha256(
        model.model.layers[0].ffn_norm.weight
    )
    assert (
        planting["normalized_input_decoder_sha256"]
        == solver["producer_bridge"]["decoder_sha256"]
    )
    assert planting["multi_anchor_plan_sha256"] == solver["multi_anchor_plan_sha256"]
    assert planting["producer_prior_sha256"] == solver["producer_prior_sha256"]
    assert planting["consumer_prior_sha256"] == solver["consumer_prior_sha256"]
    assert planting["full_source_coverage"] is True
    assert planting["producer_averaging"] is False
    assert planting["producer_per_neuron_interpolation"] is False
    assert planting["producer_shared_decoder"] is True
    assert planting["source_neurons_dropped"] == 0
    assert planting["fit_case_weight_sum"] == 8.0
    assert planting["fit_row_weight_sum"] == 8.0
    assert planting["fit_row_weights_sha256"] == solver["fit_row_weights_sha256"]
    assert planting["heldout_rows_used_by_solver"] == 0
    assert planting["planted_roles"] == ["w1.weight", "w3.weight", "w2.weight"]
    assert planting["all_dense_w1_w3_w2_roles_planted"] is True
    assert planting["fit_partition_only_used_for_solve"] is True
    assert prepared.selection_receipt["source_derived_planting"] == planting
    correction = solver["consumer_correction_report"]
    assert correction["role"] == "mlp_normalized_bridge_multi_anchor_down"
    assert correction["prior_norm"] > 0.0
    assert correction["update_scale"] == 1.0
    assert solver["producer_bridge"]["role"] == (
        "normalized_mlp_input_target_to_parent"
    )
    assert solver["consumer_bridge"]["role"] == "raw_residual_source_to_target"
    assert solver["producer_consumer_bridge_fields_separate"] is True
    assert solver["producer_shared_decoder"] is True
    assert solver["producer_per_neuron_interpolation"] is False
    assert solver["producer_averaging"] is False
    assert solver["full_source_coverage"] is True
    assert solver["source_neurons_dropped"] == 0
    assert solver["endpoint_feature_source"] == (
        "materialized_target_dtype_w1_w3_cast_back_to_fp64"
    )
    assert solver["plan_evidence_scope"] == (
        "all_source_w1_w3_w2_weights_in_global_residual_basis"
    )
    assert solver["fit_cases_used_for_plan_diagnostics_only"] is True
    assert solver["heldout_rows_used_for_plan"] == 0
    assert solver["heldout_rows_used_by_solver"] == 0
    assert solver["multi_anchor_plan"]["calibration_provenance"] == (
        "cached_fit_rows_diagnostics_only"
    )
    assert solver["weight_complete_signature_used_for_structural_selection"] is True
    assert solver["cached_activation_rows_used_for_structural_selection"] is False
    assert solver["cached_activation_rows_used_for_sparse_anchor_fit"] is False
    assert solver["multi_anchor_plan"]["planning_signature_source"] == (
        "full-weight-complete-structural-signature"
    )
    assert (
        solver["multi_anchor_plan"]["planning_signature_sha256"]
        == solver["weight_complete_signature_sha256"]
    )
    assert (
        solver["multi_anchor_plan"]["planning_signature_provenance"]
        == solver["weight_complete_signature_report_sha256"]
    )
    assert (
        planting["weight_complete_signature_sha256"]
        == (solver["weight_complete_signature_sha256"])
    )
    assert (
        planting["weight_complete_signature_report_sha256"]
        == (solver["weight_complete_signature_report_sha256"])
    )
    assert planting["weight_complete_signature_used_for_structural_selection"] is True
    assert planting["cached_activation_rows_used_for_structural_selection"] is False
    assert planting["cached_activation_rows_used_for_sparse_anchor_fit"] is False
    assert solver["multi_anchor_plan"]["invariants"]["all_hold"] is True
    assert set(solver["operation_audit"].values()) == {0}

    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
    )
    assert result.receipt["selected_role"] == "full-w1-w3-w2"
    assert result.receipt["cold_start_escape"] == cold
    assert result.receipt["source_derived_planting"] == planting
    assert all(
        int(torch.count_nonzero(parameter).item()) > 0 for parameter in mlp.parameters()
    )


def _prepared_blank_qdeim(
    recipe_id: str,
    *,
    fold_compute_device: str = "cpu",
    target_dtype: torch.dtype = torch.bfloat16,
) -> tuple[_Target, adapter.Lfm2PreparedSemanticMlpRefold]:
    model = _Target().to(target_dtype)
    with torch.no_grad():
        for parameter in model.model.layers[0].feed_forward.parameters():
            parameter.zero_()
    prepared = _prepare(
        model,
        _cold_start_parent_endpoint_evidence(model),
        body_recipe=recipe_id,
        fold_compute_device=fold_compute_device,
    )
    return model, prepared


def test_blank_qdeim_accepts_sealed_negative_alignment_coordinate_completion() -> None:
    model = _Target()
    with torch.no_grad():
        for parameter in model.model.layers[0].feed_forward.parameters():
            parameter.zero_()
    evidence = _cold_start_parent_endpoint_evidence(model)
    parent_gamma = evidence[0].source_inventory.ffn_norm_weight.to(torch.float64)
    raw_structural_decoder = -torch.diag(parent_gamma.reciprocal()).contiguous()
    coordinate_map_sha256 = _completion_tensor_sha256(raw_structural_decoder)
    completion_body = {
        "schema": "neural-forge-analytic-coordinate-completion-candidate-v1",
        "recipe": "test-full-rank-negative-alignment-completion",
        "evidence_sha256": _sha("body-global-completion-evidence"),
        "coordinate_map_shape": [4, 4],
        "coordinate_map_sha256": coordinate_map_sha256,
        "materialized_rank": 4,
        "full_rank_materialized": True,
    }
    completion_sha256 = adapter.canonical_sha256(completion_body)

    prepared = adapter.prepare_lfm2_semantic_mlp_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key=_sha("cache"),
        cache_manifest_sha256=_sha("cache-manifest"),
        bridge_artifact_sha256=_sha("bridge"),
        maximum_delta_ratio=1.0,
        rmse_relative_tolerance=1e-4,
        cosine_tolerance=1e-5,
        require_terminal_action_evidence=False,
        body_recipe="nonlinear_qdeim_structural",
        fold_compute_device="cpu",
        normalized_input_correction_rho=0.5,
        raw_structural_decoder=raw_structural_decoder,
        raw_structural_decoder_sha256=(
            adapter.lfm2_authenticated_tensor_sha256(raw_structural_decoder)
        ),
        coordinate_completion_receipt={
            **completion_body,
            "result_sha256": completion_sha256,
        },
        coordinate_completion_receipt_sha256=completion_sha256,
    )

    assert prepared.verify()
    solver = prepared.full_candidate.solver_receipt
    control = solver["normalized_decoder_control"]
    producer_bridge = control["producer_bridge"]
    provenance = control["coordinate_completion_provenance"]
    planting = solver["source_derived_planting"]
    assert producer_bridge["fit_signed_scalar_gain_advisory"] < 0.0
    assert producer_bridge["fit_structural_gain_used_for_acceptance_or_veto"] is False
    assert control["normalized_input_correction_rho"] == 0.5
    assert producer_bridge["supported_relative_cap_rho"] == 0.5
    assert producer_bridge["normalized_input_correction_rho"] == 0.5
    assert producer_bridge["structural_scale_policy"] == (
        "authenticated-coordinate-completion-prior-scale-preserved"
    )
    assert provenance["receipt_sha256"] == completion_sha256
    assert provenance["coordinate_map_sha256"] == coordinate_map_sha256
    assert planting["shared_body_global_completion_sha256"] == completion_sha256
    assert planting["body_global_coordinate_map_sha256"] == coordinate_map_sha256
    assert planting["normalized_input_correction_rho"] == 0.5
    assert set(solver["operation_audit"].values()) == {0}
    assert control["operation_audit"]["training_steps"] == 0
    assert solver["cold_start_escape"]["gradient_training_used"] is False


def test_blank_qdeim_recipes_select_separate_authenticated_dense_candidates() -> None:
    structural_model, structural = _prepared_blank_qdeim("nonlinear_qdeim_structural")
    corrected_model, corrected = _prepared_blank_qdeim("nonlinear_qdeim_fit_corrected")

    assert structural.verify()
    assert corrected.verify()
    assert structural.selected_role == corrected.selected_role == "full-w1-w3-w2"
    structural_solver = structural.full_candidate.solver_receipt
    corrected_solver = corrected.full_candidate.solver_receipt
    assert structural_solver["selected_qdeim_candidate_id"] == (
        "qdeim_structural_prior"
    )
    assert corrected_solver["selected_qdeim_candidate_id"] == (
        "qdeim_fit_residual_corrected"
    )
    assert structural_solver["qdeim_compiler_receipt"]["candidate_ids"] == [
        "qdeim_structural_prior",
        "qdeim_fit_residual_corrected",
    ]
    compute = structural_solver["qdeim_compiler_receipt"]["compute_execution"]
    assert compute["requested_device"] == "cpu"
    assert compute["resolved_device"] == "cpu"
    assert compute["authenticated_source_archive_device"] == "cpu"
    assert compute["materialized_candidate_device"] == "cpu"
    assert structural_solver["qdeim_integration"]["qdeim_compute_execution"] == (
        compute
    )
    tangent_compute = structural_solver["tangent_bank"]["compute_execution"]
    assert tangent_compute["requested_device"] == "cpu"
    assert tangent_compute["resolved_device"] == "cpu"
    assert tangent_compute["authenticated_source_archive_device"] == "cpu"
    assert tangent_compute["materialized_tangent_bank_device"] == "cpu"
    assert (
        structural_solver["qdeim_integration"]["tangent_bank_compute_execution"]
        == tangent_compute
    )
    assert (
        structural_solver["source_derived_planting"][
            "qdeim_pod_and_pivot_compute_device"
        ]
        == "cpu"
    )
    assert (
        structural_solver["source_derived_planting"]["tangent_bank_compute_device"]
        == "cpu"
    )
    assert structural_solver["tangent_bank"]["geometry"]["anchor_count"] == 2
    assert structural_solver["tangent_bank"]["anchor_indices"] == [0, 7]
    assert (
        structural_solver["qdeim_integration"][
            "tangent_rows_used_for_endpoint_w2_correction"
        ]
        == 0
    )
    assert (
        structural_solver["qdeim_integration"]["heldout_rows_used_for_pod_or_tangent"]
        == 0
    )
    assert (
        structural_solver["qdeim_integration"][
            "numeric_metrics_used_for_semantic_acceptance_or_veto"
        ]
        is False
    )
    assert (
        structural_solver["qdeim_integration"][
            "whole_decoded_responses_are_primary_semantic_evidence"
        ]
        is True
    )
    assert structural_solver["qdeim_integration"]["binary_manual_gate_created"] is False
    assert structural_solver["child_coordinate_rms"]["structural_floor_policy"] == (
        "sqrt_float64_machine_epsilon"
    )
    assert structural_solver["normalized_decoder_control"]["schema"] == (
        "neural-forge-lfm2-normalized-input-decoder-only-v1"
    )
    assert structural_solver["normalized_decoder_control"]["decoder_only_scope"] is True
    assert (
        structural_solver["normalized_decoder_control"][
            "cluster_or_medoid_state_constructed"
        ]
        is False
    )
    decoder_operations = structural_solver["normalized_decoder_control"][
        "operation_audit"
    ]
    assert decoder_operations["activation_cluster_plan_builds"] == 0
    assert decoder_operations["medoid_selection_operations"] == 0
    assert decoder_operations["producer_materializations"] == 0
    assert decoder_operations["consumer_materializations"] == 0
    assert (
        structural_solver["qdeim_integration"][
            "normalized_decoder_control_candidate_weights_used"
        ]
        is False
    )
    assert structural_solver["rank_completion"]["random_completion_modes"] == 0
    assert structural_solver["rank_completion"]["silent_completion_performed"] is False
    assert (
        structural.selection_receipt[
            "blank_target_selected_state_uses_real_qdeim_producers"
        ]
        is True
    )
    assert (
        structural.selection_receipt[
            "whole_decoded_responses_are_primary_semantic_evidence"
        ]
        is True
    )
    assert structural.selection_receipt["binary_manual_gate_created"] is False
    assert torch.equal(
        structural.full_candidate.gate_weight,
        corrected.full_candidate.gate_weight,
    )
    assert torch.equal(
        structural.full_candidate.up_weight,
        corrected.full_candidate.up_weight,
    )
    assert not torch.equal(
        structural.full_candidate.down_weight,
        corrected.full_candidate.down_weight,
    )
    assert all(
        value.dtype == torch.bfloat16 and not value.requires_grad
        for value in structural.selected_candidate.state_dict().values()
    )

    structural_repeat_model, structural_repeat = _prepared_blank_qdeim(
        "nonlinear_qdeim_structural"
    )
    assert structural_repeat.full_candidate.candidate_sha256 == (
        structural.full_candidate.candidate_sha256
    )
    assert (
        structural_repeat.full_candidate.solver_receipt["qdeim_integration_sha256"]
        == structural_solver["qdeim_integration_sha256"]
    )
    assert fingerprint_lfm2_parent(structural_model).identity_sha256 == (
        fingerprint_lfm2_parent(structural_repeat_model).identity_sha256
    )
    assert fingerprint_lfm2_parent(corrected_model).identity_sha256 == (
        fingerprint_lfm2_parent(structural_model).identity_sha256
    )


def test_blank_qdeim_float32_target_prepares_and_installs_float32_state() -> None:
    model, prepared = _prepared_blank_qdeim(
        "nonlinear_qdeim_structural",
        target_dtype=torch.float32,
    )

    assert prepared.verify()
    solver = prepared.full_candidate.solver_receipt
    qdeim = solver["qdeim_compiler_receipt"]
    assert qdeim["target_materialization_dtype"] == "float32"
    assert qdeim["dense_supported_target_dtype_state"] is True
    assert qdeim["dense_bfloat16_target_state"] is False
    assert solver["qdeim_integration"]["target_materialization_dtype"] == "float32"
    assert all(
        value.dtype == torch.float32
        for value in prepared.selected_candidate.state_dict().values()
    )

    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
    )

    assert result.receipt["installed"] is True
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_blank_qdeim_cuda_stages_tangent_and_qdeim_without_gpu_output_leak() -> None:
    _model, prepared = _prepared_blank_qdeim(
        "nonlinear_qdeim_structural",
        fold_compute_device="cuda:0",
    )
    assert prepared.verify()
    solver = prepared.full_candidate.solver_receipt
    qdeim_compute = solver["qdeim_compiler_receipt"]["compute_execution"]
    tangent_compute = solver["tangent_bank"]["compute_execution"]
    for compute in (qdeim_compute, tangent_compute):
        assert compute["requested_device"] == "cuda:0"
        assert compute["resolved_device"] == "cuda:0"
        assert compute["backend"] == "cuda"
        assert compute["active_layer_staged_to_compute_device"] is True
        assert compute["authenticated_source_archive_device"] == "cpu"
        assert compute["silent_device_fallback_performed"] is False
    assert qdeim_compute["materialized_candidate_device"] == "cpu"
    assert tangent_compute["materialized_tangent_bank_device"] == "cpu"
    assert solver["qdeim_integration"]["qdeim_compute_execution"] == qdeim_compute
    assert (
        solver["qdeim_integration"]["tangent_bank_compute_execution"] == tangent_compute
    )
    assert solver["source_derived_planting"]["fold_compute_device"] == "cuda:0"
    assert all(
        value.device.type == "cpu" and value.dtype == torch.bfloat16
        for value in prepared.selected_candidate.state_dict().values()
    )


def test_blank_qdeim_rank_deficit_propagates_without_medoid_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Target()
    with torch.no_grad():
        for parameter in model.model.layers[0].feed_forward.parameters():
            parameter.zero_()
    report = {
        "observed_snapshot_rank": 2,
        "minimum_snapshot_rank": 3,
        "snapshot_rank_deficit": 1,
        "selected_source_neurons": [],
        "interpolation_matrix_materialized": False,
        "random_mode_completion_permitted": False,
    }

    def insufficient_rank(**_kwargs):
        raise Lfm2MlpQdeimInsufficientRankError(report)

    monkeypatch.setattr(
        adapter,
        "compile_lfm2_mlp_nonlinear_qdeim",
        insufficient_rank,
    )
    with pytest.raises(Lfm2MlpQdeimInsufficientRankError) as caught:
        _prepare(
            model,
            _cold_start_parent_endpoint_evidence(model),
            body_recipe="nonlinear_qdeim_structural",
        )
    assert caught.value.report["snapshot_rank_deficit"] == 1
    assert caught.value.report["selected_source_neurons"] == []


def test_blank_qdeim_never_enters_legacy_medoid_or_writer_solvers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Q-DEIM entered an unselected legacy solver.")

    monkeypatch.setattr(
        adapter,
        "compile_lfm2_normalized_input_real_medoid_structural_mlp_fold",
        forbidden,
    )
    monkeypatch.setattr(adapter, "_solve_writer", forbidden)

    _model, prepared = _prepared_blank_qdeim("nonlinear_qdeim_structural")

    assert prepared.verify()
    assert prepared.full_candidate.solver_receipt["decoder_control_only_solve"] is True
    assert prepared.writer_candidate.solver_receipt["residual_writer_solve_calls"] == 0
    assert (
        prepared.writer_candidate.solver_receipt["placeholder_state_is_exact_incumbent"]
        is True
    )


def test_blank_residual_writer_is_distinct_complete_full_source_plant() -> None:
    prepared_by_recipe: dict[str, adapter.Lfm2PreparedSemanticMlpRefold] = {}
    state_sha256: dict[str, str] = {}
    for recipe_id in (
        "full_strength_structural",
        "residual_writer",
        "source_producer_residual_writer",
        "structural_sequence_residual_mlp_writer",
    ):
        model = _Target()
        with torch.no_grad():
            for parameter in model.model.layers[0].feed_forward.parameters():
                parameter.zero_()
        prepared = _prepare(
            model,
            _cold_start_parent_endpoint_evidence(model),
            body_recipe=recipe_id,
        )
        prepared_by_recipe[recipe_id] = prepared
        state_sha256[recipe_id] = adapter._state_sha256(
            prepared.selected_candidate.state_dict()
        )
        assert prepared.verify()
        assert (
            prepared.selection_receipt[
                "blank_target_selected_state_uses_full_source_structural_producers"
            ]
            is True
        )
        assert adapter._nonzero_swiglu_roles(
            prepared.selected_candidate.state_dict()
        ) == ("w1.weight", "w3.weight", "w2.weight")

    assert prepared_by_recipe["full_strength_structural"].selected_role == (
        "full-w1-w3-w2"
    )
    residual = prepared_by_recipe["residual_writer"]
    assert residual.selected_role == "incumbent-producer-w2"
    assert residual.writer_candidate.solver_receipt["fixed_producer_state_source"] == (
        "full-source-structural-w1-w3"
    )
    assert (
        residual.selection_receipt[
            "blank_target_residual_writer_component_equivalent_to_full_structural"
        ]
        is False
    )
    assert state_sha256["full_strength_structural"] != state_sha256["residual_writer"]
    hybrid = prepared_by_recipe["source_producer_residual_writer"]
    assert hybrid.selected_role == "incumbent-producer-w2"
    assert hybrid.writer_candidate.solver_receipt["fixed_producer_state_source"] == (
        "full-source-structural-w1-w3"
    )
    assert (
        state_sha256["source_producer_residual_writer"]
        == state_sha256["residual_writer"]
    )
    structural_sequence_writer = prepared_by_recipe[
        "structural_sequence_residual_mlp_writer"
    ]
    assert structural_sequence_writer.selected_role == "incumbent-producer-w2"
    assert (
        state_sha256["structural_sequence_residual_mlp_writer"]
        == state_sha256["residual_writer"]
    )


def test_prepared_recipe_and_selected_lane_cannot_be_resealed_independently() -> None:
    recipe_ids = (
        "fit_guided",
        "full_strength_structural",
        "residual_writer",
        "source_producer_residual_writer",
        "structural_sequence_residual_mlp_writer",
    )
    for source_recipe_id in recipe_ids:
        model = _Target()
        prepared = _prepare(
            model,
            _evidence(model),
            body_recipe=source_recipe_id,
        )
        for replacement_recipe_id in recipe_ids:
            if replacement_recipe_id == source_recipe_id:
                continue
            with pytest.raises(
                adapter.Lfm2SemanticMlpRefoldError,
                match="recipe binding",
            ):
                replace(
                    prepared,
                    body_recipe=adapter.resolve_lfm2_semantic_body_recipe(
                        replacement_recipe_id
                    ),
                )
        forged_selected_role = (
            "incumbent-producer-w2"
            if prepared.selected_role == "full-w1-w3-w2"
            else "full-w1-w3-w2"
        )
        with pytest.raises(
            adapter.Lfm2SemanticMlpRefoldError,
            match="recipe binding",
        ):
            replace(prepared, selected_role=forged_selected_role)
        with pytest.raises(
            adapter.Lfm2SemanticMlpRefoldError,
            match="recipe binding",
        ):
            replace(prepared, writer_candidate=prepared.full_candidate)


def test_blank_dispatch_uses_only_weighted_fit_chart_multi_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Target()
    with torch.no_grad():
        for parameter in model.model.layers[0].feed_forward.parameters():
            parameter.zero_()
    evidence = tuple(
        replace(item, sample_weight=(index + 1) / 32.0)
        for index, item in enumerate(_cold_start_parent_endpoint_evidence(model))
    )
    fit = adapter._split_evidence(evidence)["fit"]
    observed = []
    real_compile = adapter.compile_lfm2_normalized_input_real_medoid_structural_mlp_fold

    def recording_compile(**kwargs):
        observed.append(kwargs)
        return real_compile(**kwargs)

    def forbidden_legacy(**_kwargs):
        raise AssertionError("Blank child reached the legacy barycentric compiler.")

    monkeypatch.setattr(
        adapter,
        "compile_lfm2_normalized_input_real_medoid_structural_mlp_fold",
        recording_compile,
    )
    monkeypatch.setattr(
        adapter,
        "compile_lfm2_structural_residual_mlp_fold",
        forbidden_legacy,
    )
    prepared = _prepare(model, evidence)

    assert prepared.selected_role == "full-w1-w3-w2"
    assert len(observed) == 1
    call = observed[0]
    assert call["case_sample_weights"] == tuple(item.sample_weight for item in fit)
    assert len(call["cases"]) == len(fit)
    assert all(
        torch.equal(case.parent_normalized_input, item.parent_normalized_mlp_input)
        for case, item in zip(call["cases"], fit, strict=True)
    )
    assert call["residual_chart_sha256"] == adapter._tensor_sha256(
        fit[0].source_inventory.residual_chart
    )
    assert call["parent_ffn_norm_weight_sha256"] == adapter._tensor_sha256(
        fit[0].source_inventory.ffn_norm_weight
    )
    assert call["target_ffn_norm_weight_sha256"] == adapter._tensor_sha256(
        model.model.layers[0].ffn_norm.weight
    )
    assert call["normalized_decoder_ridge"] == adapter._NORMALIZED_DECODER_RIDGE
    assert prepared.full_candidate.solver_receipt["fit_case_weight_sum"] == (
        pytest.approx(sum(item.sample_weight for item in fit))
    )
    assert prepared.full_candidate.solver_receipt["heldout_rows_used_for_plan"] == 0
    assert prepared.full_candidate.solver_receipt["heldout_rows_used_by_solver"] == 0


def test_mutated_source_chart_is_rejected_before_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Target()
    evidence = _evidence(model)
    before = fingerprint_lfm2_parent(model).identity_sha256
    compile_called = False

    def forbidden_compile(**_kwargs):
        nonlocal compile_called
        compile_called = True
        raise AssertionError("Unauthenticated source reached the compiler.")

    monkeypatch.setattr(
        adapter,
        "compile_lfm2_structural_residual_mlp_fold",
        forbidden_compile,
    )
    evidence[0].source_inventory.residual_chart[0, 0] += 0.5
    assert evidence[0].verify() is True
    assert evidence[0].source_inventory.verify() is False

    with pytest.raises(
        adapter.Lfm2SemanticMlpRefoldError,
        match="authenticated typed endpoint evidence",
    ):
        _prepare(model, evidence)
    assert compile_called is False
    assert fingerprint_lfm2_parent(model).identity_sha256 == before


def test_blank_multi_weight_plant_matches_canonical_layer_stack_diff() -> None:
    model = _Target()
    model.untouched = torch.nn.Parameter(
        torch.tensor([1.25], dtype=torch.bfloat16),
    )
    mlp = model.model.layers[0].feed_forward
    with torch.no_grad():
        for parameter in mlp.parameters():
            parameter.zero_()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    evidence = _cold_start_parent_endpoint_evidence(model)
    before_hashes = stack_adapter._state_tensor_hashes(model)
    before_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    pointers = {
        name: parameter.data_ptr() for name, parameter in model.named_parameters()
    }
    gradient_ids = {
        name: id(parameter.grad) for name, parameter in model.named_parameters()
    }
    gradient_state = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }

    receipt, terminal_evidence, terminal_prior = stack_adapter._install_mlp(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache=SimpleNamespace(
            cache_key=_sha("cache"),
            manifest={"manifest_sha256": _sha("cache-manifest")},
        ),
        bridge=SimpleNamespace(artifact_sha256=_sha("bridge")),
        installer=adapter.compile_install_lfm2_semantic_mlp_postfold,
        final_layer=False,
        parameter_count=stack_adapter._parameter_count(model),
    )

    live_changed = stack_adapter._changed_state_keys(
        before_hashes,
        stack_adapter._state_tensor_hashes(model),
    )
    prefix = "model.layers.0.feed_forward."
    assert receipt["installed"] is True
    assert receipt["selected_role"] == "full-w1-w3-w2"
    assert receipt["changed_parameter_roles"] == [
        name.removeprefix(prefix) for name in live_changed
    ]
    assert receipt["changed_parameter_roles"] == [
        "w1.weight",
        "w2.weight",
        "w3.weight",
    ]
    assert live_changed == tuple(sorted(live_changed))
    assert terminal_evidence == {}
    assert terminal_prior is None
    assert torch.equal(model.untouched.detach().cpu(), before_state["untouched"])
    assert all(
        torch.equal(value.detach().cpu(), before_state[name])
        for name, value in model.state_dict().items()
        if name not in live_changed
    )
    assert {
        name: id(parameter) for name, parameter in model.named_parameters()
    } == identities
    assert {
        name: parameter.data_ptr() for name, parameter in model.named_parameters()
    } == pointers
    assert {
        name: id(parameter.grad) for name, parameter in model.named_parameters()
    } == gradient_ids
    assert all(
        torch.equal(parameter.grad.detach().cpu(), gradient_state[name])
        for name, parameter in model.named_parameters()
    )
    assert all(
        int(torch.count_nonzero(parameter).item()) > 0 for parameter in mlp.parameters()
    )


def test_exact_zero_incumbent_continues_complete_plant_past_local_quality_gates() -> (
    None
):
    model = _Target()
    with torch.no_grad():
        for parameter in model.model.layers[0].feed_forward.parameters():
            parameter.zero_()
    before = fingerprint_lfm2_parent(model).identity_sha256

    prepared = _prepare(model, _evidence(model))

    assert prepared.verify()
    assert prepared.selected_role == "full-w1-w3-w2"
    diagnostics = dict(prepared.selection_receipt)
    assert set(diagnostics["endpoint_metrics"]) == {
        "incumbent",
        "incumbent-producer-w2",
        "full-w1-w3-w2",
    }
    for role_metrics in diagnostics["endpoint_metrics"].values():
        assert set(role_metrics) == {"fit", "fit_check", "selection", "validation"}
        assert all(metrics["endpoint_rows"] > 0 for metrics in role_metrics.values())
    assert (
        diagnostics["heldout_selection_improvement"]["incumbent-producer-w2"] is False
    )
    assert diagnostics["heldout_nonregression"]["full-w1-w3-w2"] == {
        "fit_check": False,
        "selection": False,
        "validation": False,
    }
    assert diagnostics["selected_role"] == "full-w1-w3-w2"
    assert diagnostics["component_quality_gates_block_installation"] is False
    assert diagnostics["blank_target_requires_complete_structural_plant"] is True
    assert diagnostics["blank_target_continues_to_whole_response_evaluation"] is True
    assert diagnostics["heldout_quality_retained_as_diagnostics"] is True
    assert diagnostics["incumbent_noop_fallback_used"] is False
    assert diagnostics["validation_used_for_candidate_choice"] is False
    assert (
        diagnostics["validation_used_only_for_acceptance_or_incumbent_fallback"]
        is False
    )
    assert diagnostics["rationale"] == (
        "blank-target-complete-structural-plant-continues-to-whole-response-evaluation"
    )
    assert prepared.partition_batch_indices == {
        "fit": (0,),
        "fit_check": (1,),
        "selection": (2,),
        "validation": (3,),
    }
    assert adapter._nonzero_swiglu_roles(prepared.selected_candidate.state_dict()) == (
        "w1.weight",
        "w3.weight",
        "w2.weight",
    )
    assert fingerprint_lfm2_parent(model).identity_sha256 == before


def test_tiny_requested_delta_ratio_is_diagnostic_and_never_caps_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Target()
    observed_caps = []
    real_compile = adapter.compile_lfm2_structural_residual_mlp_fold

    def recording_compile(**kwargs):
        observed_caps.append(kwargs["max_prior_delta_ratio"])
        return real_compile(**kwargs)

    monkeypatch.setattr(
        adapter,
        "compile_lfm2_structural_residual_mlp_fold",
        recording_compile,
    )

    def forbidden_blank_compile(**_kwargs):
        raise AssertionError("Nonzero child reached the blank-start compiler.")

    monkeypatch.setattr(
        adapter,
        "compile_lfm2_normalized_input_real_medoid_structural_mlp_fold",
        forbidden_blank_compile,
    )
    tiny_goal = 1e-12
    prepared = _prepare(
        model,
        _evidence(model),
        maximum_delta_ratio=tiny_goal,
    )

    assert observed_caps == [None]
    assert "cold_start_escape" not in prepared.full_candidate.solver_receipt
    assert "cold_start_escape" not in prepared.selection_receipt
    assert prepared.selected_role in {"incumbent-producer-w2", "full-w1-w3-w2"}
    selection = prepared.selection_receipt
    assert selection["requested_maximum_delta_ratio_goal"] == tiny_goal
    assert selection["effective_maximum_delta_ratio_cap"] is None
    assert selection["delta_ratio_cap_applied"] is False
    assert selection["delta_ratio_used_to_scale_candidate"] is False
    assert selection["delta_ratio_diagnostic_only"] is True
    assert selection["numeric_delta_ratio_used_for_candidate_eligibility"] is False
    assert selection["numeric_delta_ratio_used_for_candidate_ordering"] is False
    assert selection["numeric_delta_ratio_used_for_publication_or_rollback"] is False
    assert selection["numeric_delta_ratio_has_final_intelligence_authority"] is False
    writer_strength = selection["planting_strength_diagnostics"][
        "incumbent-producer-w2"
    ]
    assert writer_strength["observed_maximum_delta_ratio"] > tiny_goal
    assert writer_strength["requested_delta_ratio_goal_met"] is False
    assert writer_strength["effective_maximum_delta_ratio_cap"] is None
    assert writer_strength["delta_ratio_cap_applied"] is False
    writer_solver = prepared.writer_candidate.solver_receipt
    assert writer_solver["unclipped_delta_ratio"] > tiny_goal
    assert writer_solver["applied_delta_scale"] == 1.0
    assert writer_solver["requested_maximum_delta_ratio_goal"] == tiny_goal
    assert writer_solver["requested_delta_ratio_goal_met"] is False
    assert writer_solver["effective_maximum_delta_ratio_cap"] is None
    assert writer_solver["delta_ratio_cap_applied"] is False
    full_solver = prepared.full_candidate.solver_receipt
    assert full_solver["requested_maximum_delta_ratio_goal"] == tiny_goal
    assert full_solver["effective_maximum_delta_ratio_cap"] is None
    assert full_solver["delta_ratio_cap_applied"] is False
    assert all(
        report["update_scale"] == 1.0
        for report in full_solver["prior_correction_reports"]
    )
    writer_report = selection["candidate_reports"]["incumbent-producer-w2"]
    assert writer_report["effectful_nonincumbent"] is True
    assert writer_report["structurally_eligible"] is True
    selected_strength = selection["planting_strength_diagnostics"][
        prepared.selected_role
    ]
    assert selected_strength["observed_maximum_delta_ratio"] > tiny_goal
    assert selected_strength["requested_delta_ratio_goal_met"] is False
    selected_report = selection["candidate_reports"][prepared.selected_role]
    assert selected_report["effectful_nonincumbent"] is True
    assert selected_report["structurally_eligible"] is True
    eligible_fit_keys = [
        selection["fit_ordering_keys"][role]
        for role in ("incumbent-producer-w2", "full-w1-w3-w2")
        if selection["candidate_reports"][role]["structurally_eligible"] is True
    ]
    assert selection["fit_ordering_keys"][prepared.selected_role] == min(
        eligible_fit_keys
    )


def test_materially_better_full_candidate_wins_after_frozen_selection(
    monkeypatch,
) -> None:
    model = _Target()
    evidence, desired = _material_full_win_case(model)

    def exact_full_candidate(*_args, **_kwargs):
        return desired, {
            "solver": "test-authenticated-exact-full-candidate",
            "cluster_solve": True,
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    monkeypatch.setattr(adapter, "_full_candidate_state", exact_full_candidate)
    prepared = _prepare(model, evidence)
    gates = prepared.selection_receipt
    assert gates["heldout_nonregression"]["incumbent-producer-w2"] == {
        "fit_check": True,
        "selection": True,
        "validation": True,
    }
    assert gates["writer_selection_nonregressing_versus_full"] is False
    assert prepared.selected_role == "full-w1-w3-w2"
    assert (
        prepared.full_candidate.metrics["selection"].endpoint_maximum_absolute_error
        < prepared.writer_candidate.metrics["selection"].endpoint_maximum_absolute_error
    )
    assert gates["candidate_frozen_before_validation"] is True
    assert gates["validation_triggered_refit"] is False


def test_transactional_install_preserves_topology_identity_and_gradients() -> None:
    model = _Target()
    mlp = model.model.layers[0].feed_forward
    for parameter in mlp.parameters():
        parameter.grad = torch.ones_like(parameter)
    evidence = _evidence(model)
    prepared = _prepare(model, evidence)
    count = sum(parameter.numel() for parameter in model.parameters())
    identities = tuple(id(parameter) for parameter in model.parameters())
    pointers = tuple(parameter.data_ptr() for parameter in model.parameters())
    gradient_ids = tuple(id(parameter.grad) for parameter in model.parameters())
    before = fingerprint_lfm2_parent(model).identity_sha256
    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
    )
    assert result.receipt["installed"] is True
    assert result.receipt["selected_role"] == "full-w1-w3-w2"
    assert result.receipt["changed_parameter_roles"] == [
        "w1.weight",
        "w2.weight",
        "w3.weight",
    ]
    assert result.receipt["model_state_before_sha256"] == before
    assert result.receipt["model_state_after_sha256"] != before
    assert result.receipt["parameter_count_changed"] is False
    assert sum(parameter.numel() for parameter in model.parameters()) == count
    assert tuple(id(parameter) for parameter in model.parameters()) == identities
    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == pointers
    assert tuple(id(parameter.grad) for parameter in model.parameters()) == gradient_ids
    assert all(value == 0 for value in result.receipt["operation_audit"].values())


def test_regressing_metrics_do_not_veto_structurally_valid_effectful_candidate(
    monkeypatch,
) -> None:
    model = _Target()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    evidence = _evidence(model)
    before = fingerprint_lfm2_parent(model).identity_sha256
    identities = tuple(id(parameter) for parameter in model.parameters())
    pointers = tuple(parameter.data_ptr() for parameter in model.parameters())
    gradient_ids = tuple(id(parameter.grad) for parameter in model.parameters())

    def regressing_writer(_fit, incumbent, **_policy):
        state = {
            name: value.detach().cpu().clone() for name, value in incumbent.items()
        }
        state["w2.weight"] = torch.full_like(state["w2.weight"], 64.0)
        return state, {
            "solver": "test-regressing-writer",
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    def regressing_full(_fit, _inventory, incumbent, **_policy):
        state = {
            name: torch.full_like(value, -64.0) for name, value in incumbent.items()
        }
        return state, {
            "solver": "test-regressing-full",
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    monkeypatch.setattr(adapter, "_solve_writer", regressing_writer)
    monkeypatch.setattr(adapter, "_full_candidate_state", regressing_full)
    prepared = _prepare(model, evidence)
    assert prepared.selected_role == "full-w1-w3-w2"
    assert prepared.selection_receipt["incumbent_noop_fallback_used"] is False
    assert prepared.selection_receipt["structurally_eligible_candidate_count"] == 2
    assert prepared.selection_receipt["structural_safety_is_only_candidate_veto"]
    assert prepared.selection_receipt["heldout_metrics_are_diagnostic_only"]
    assert prepared.selection_receipt["validation_publication_veto_applied"] is False
    assert prepared.selection_receipt["rollback_authorized"] is False
    assert prepared.selection_receipt["validation_rejected_role"] is None
    assert (
        prepared.selection_receipt["fit_ordering_keys"]["full-w1-w3-w2"]
        < (prepared.selection_receipt["fit_ordering_keys"]["incumbent-producer-w2"])
    )
    for role in ("incumbent-producer-w2", "full-w1-w3-w2"):
        report = prepared.selection_receipt["candidate_reports"][role]
        assert report["structurally_eligible"] is True
        assert report["numeric_diagnostic_miss"] is True
        assert report["numeric_diagnostic_miss_can_exclude_candidate"] is False
    assert (
        prepared.selection_receipt["heldout_nonregression"]["full-w1-w3-w2"][
            "validation"
        ]
        is False
    )

    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
    )
    receipt = dict(result.receipt)
    seal = receipt.pop("result_sha256")
    assert adapter.canonical_sha256(receipt) == seal
    assert receipt["status"] == "installed-frozen-semantic-mlp-candidate"
    assert receipt["selected_role"] == "full-w1-w3-w2"
    assert receipt["incumbent_selected_by_heldout_gates"] is False
    assert receipt["fit_check_nonregression_passed"] is False
    assert receipt["selection_nonregression_passed"] is False
    assert receipt["validation_nonregression_passed"] is False
    assert receipt["validation_diagnostic_would_have_failed_legacy_gate"] is True
    assert receipt["validation_publication_veto_applied"] is False
    assert receipt["rollback_authorized"] is False
    assert receipt["numeric_metrics_have_final_intelligence_authority"] is False
    assert receipt["installed"] is True
    assert receipt["install_count"] == 1
    assert receipt["changed_parameter_roles"] == [
        "w1.weight",
        "w2.weight",
        "w3.weight",
    ]
    assert receipt["parameter_identities_preserved"] is True
    assert receipt["parameter_storage_preserved"] is True
    assert receipt["gradient_state_preserved"] is True
    assert (
        receipt["source_inventory_sha256"]
        == evidence[0].source_inventory.inventory_sha256
    )
    assert (
        receipt["compiler_lineage_sha256"]
        == evidence[0].source_inventory.compiler_lineage_sha256
    )
    assert receipt["cache_key"] == _sha("cache")
    assert receipt["cache_manifest_sha256"] == _sha("cache-manifest")
    assert receipt["bridge_artifact_sha256"] == _sha("bridge")
    assert receipt["evidence_sha256"] == [item.evidence_sha256 for item in evidence]
    assert all(value == 0 for value in receipt["operation_audit"].values())
    assert fingerprint_lfm2_parent(model).identity_sha256 != before
    assert tuple(id(parameter) for parameter in model.parameters()) == identities
    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == pointers
    assert tuple(id(parameter.grad) for parameter in model.parameters()) == gradient_ids
    assert all(
        torch.equal(
            parameter.detach().cpu(),
            torch.full_like(parameter, -64.0).cpu(),
        )
        for parameter in model.model.layers[0].feed_forward.parameters()
    )
    assert not result.final_layer_mlp_evidence


@pytest.mark.parametrize(
    ("body_recipe", "expected_role"),
    (
        ("fit_guided", "full-w1-w3-w2"),
        ("full_strength_structural", "full-w1-w3-w2"),
        ("residual_writer", "incumbent-producer-w2"),
        ("source_producer_residual_writer", "incumbent-producer-w2"),
        ("structural_sequence_residual_mlp_writer", "incumbent-producer-w2"),
    ),
)
def test_recipe_priority_survives_poor_heldout_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    body_recipe: str,
    expected_role: str,
) -> None:
    model = _Target()
    evidence = _evidence(model)

    def regressing_writer(_fit, incumbent, **_policy):
        state = {
            name: value.detach().cpu().clone() for name, value in incumbent.items()
        }
        state["w2.weight"] = torch.full_like(state["w2.weight"], 64.0)
        return state, {
            "solver": "test-recipe-regressing-writer",
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    def regressing_full(_fit, _inventory, incumbent, **_policy):
        return {
            name: torch.full_like(value, -64.0) for name, value in incumbent.items()
        }, {
            "solver": "test-recipe-regressing-full",
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    monkeypatch.setattr(adapter, "_solve_writer", regressing_writer)
    monkeypatch.setattr(adapter, "_full_candidate_state", regressing_full)
    prepared = _prepare(model, evidence, body_recipe=body_recipe)

    assert prepared.selected_role == expected_role
    assert prepared.selection_receipt["body_recipe_id"] == body_recipe
    assert (
        prepared.selection_receipt["heldout_nonregression"][expected_role]["validation"]
        is False
    )
    assert prepared.selection_receipt["incumbent_noop_fallback_used"] is False
    assert prepared.selection_receipt["validation_publication_veto_applied"] is False
    assert (
        prepared.selection_receipt["numeric_metrics_have_final_intelligence_authority"]
        is False
    )

    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
    )
    assert result.receipt["installed"] is True
    assert result.receipt["selected_role"] == expected_role
    assert result.receipt["body_recipe_id"] == body_recipe


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
def test_identical_mlp_candidates_select_incumbent_and_perform_no_copy(
    monkeypatch,
    body_recipe: str,
) -> None:
    model = _Target()
    evidence = _evidence(model)
    before = fingerprint_lfm2_parent(model).identity_sha256

    def identical_writer(_fit, incumbent, **_policy):
        return {
            name: value.detach().cpu().clone() for name, value in incumbent.items()
        }, {
            "solver": "test-identical-writer",
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    def identical_full(_fit, _inventory, incumbent, **_policy):
        return {
            name: value.detach().cpu().clone() for name, value in incumbent.items()
        }, {
            "solver": "test-identical-full",
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    monkeypatch.setattr(adapter, "_solve_writer", identical_writer)
    monkeypatch.setattr(adapter, "_full_candidate_state", identical_full)
    prepared = _prepare(model, evidence, body_recipe=body_recipe)
    assert prepared.verify()
    assert prepared.selected_role == "incumbent"
    assert prepared.selection_receipt["writer_changed_roles"] == []
    assert prepared.selection_receipt["full_changed_roles"] == []
    assert prepared.selection_receipt["heldout_selection_improvement"] == {
        "incumbent-producer-w2": False,
        "full-w1-w3-w2": False,
    }
    assert prepared.selection_receipt["structurally_eligible_candidate_count"] == 0
    assert prepared.selection_receipt["incumbent_noop_fallback_reason"] == (
        "all-analytic-proposals-structurally-invalid-or-identical"
    )
    for role in ("incumbent-producer-w2", "full-w1-w3-w2"):
        report = prepared.selection_receipt["candidate_reports"][role]
        assert report["structural_safety_passed"] is True
        assert report["effectful_nonincumbent"] is False
        assert report["structurally_eligible"] is False

    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
    )
    assert result.receipt["installed"] is False
    assert result.receipt["install_count"] == 0
    assert result.receipt["selected_role"] == "incumbent"
    assert result.receipt["selected_install_roles_identical_to_live_state"] is True
    assert result.receipt["incumbent_noop_fallback_used"] is True
    assert result.receipt["status"] == adapter._STRUCTURAL_NOOP_STATUS
    assert result.receipt["selected_candidate"] == "incumbent-noop"
    assert result.receipt["incumbent_selected_by_heldout_gates"] is False
    assert result.receipt["body_recipe_id"] == body_recipe
    assert result.receipt["heldout_metrics_are_diagnostic_only"] is True
    assert result.receipt["numeric_metric_used_for_publication_or_rollback"] is False
    assert stack_adapter._authenticated_heldout_incumbent_noop(
        result.receipt,
        component="mlp",
        before_sha256=before,
        after_sha256=before,
        changed_state_keys=(),
    )
    assert fingerprint_lfm2_parent(model).identity_sha256 == before


def test_metric_evaluation_failure_is_diagnostic_not_structural_veto(
    monkeypatch,
) -> None:
    model = _Target()
    evidence = _evidence(model)

    def unavailable_metrics(*_args, **_kwargs):
        raise ArithmeticError("injected diagnostic overflow")

    monkeypatch.setattr(adapter, "_metrics", unavailable_metrics)
    prepared = _prepare(model, evidence)

    assert prepared.verify()
    assert prepared.selected_role == "incumbent-producer-w2"
    assert prepared.selection_receipt["structurally_eligible_candidate_count"] == 2
    assert (
        prepared.selection_receipt[
            "metric_evaluation_failure_can_exclude_structural_candidate"
        ]
        is False
    )
    for role in ("incumbent", "incumbent-producer-w2", "full-w1-w3-w2"):
        errors = prepared.selection_receipt["partition_metric_errors"][role]
        assert set(errors) == {"fit", "fit_check", "selection", "validation"}
        assert all("injected diagnostic overflow" in error for error in errors.values())
    assert prepared.writer_candidate.metrics["fit"].endpoint_relative_rmse == (
        adapter._MAX_FINITE_FLOAT
    )

    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
    )
    assert result.receipt["installed"] is True
    assert result.receipt["selected_role"] == "incumbent-producer-w2"
    assert (
        result.receipt["metric_evaluation_failure_can_exclude_structural_candidate"]
        is False
    )


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
def test_invalid_mlp_proposals_are_sealed_and_use_structural_incumbent_fallback(
    monkeypatch,
    body_recipe: str,
) -> None:
    model = _Target()
    evidence = _evidence(model)
    before = fingerprint_lfm2_parent(model).identity_sha256

    def wrong_dtype_writer(_fit, incumbent, **_policy):
        state = {
            name: value.detach().cpu().clone() for name, value in incumbent.items()
        }
        state["w2.weight"] = state["w2.weight"].to(torch.float32).add_(1.0)
        return state, {
            "solver": "test-wrong-dtype-writer",
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    def nonfinite_full(_fit, _inventory, incumbent, **_policy):
        state = {
            name: value.detach().cpu().clone() for name, value in incumbent.items()
        }
        state["w1.weight"][0, 0] = torch.nan
        return state, {
            "solver": "test-nonfinite-full",
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    monkeypatch.setattr(adapter, "_solve_writer", wrong_dtype_writer)
    monkeypatch.setattr(adapter, "_full_candidate_state", nonfinite_full)
    prepared = _prepare(model, evidence, body_recipe=body_recipe)

    assert prepared.verify()
    assert prepared.selected_role == "incumbent"
    assert prepared.selection_receipt["structurally_eligible_candidate_count"] == 0
    assert prepared.selection_receipt["incumbent_noop_fallback_reason"] == (
        "all-analytic-proposals-structurally-invalid-or-identical"
    )
    assert prepared.selection_receipt["invalid_proposal_metric_placeholder_roles"] == [
        "incumbent-producer-w2",
        "full-w1-w3-w2",
    ]
    for candidate in (prepared.writer_candidate, prepared.full_candidate):
        report = prepared.selection_receipt["candidate_reports"][candidate.role]
        assert report["structural_safety_passed"] is False
        assert report["structurally_eligible"] is False
        assert report["frozen_candidate_uses_incumbent_placeholder"] is True
        assert report["partition_metrics_represent_original_analytic_proposal"] is False
        assert candidate.solver_receipt[
            "invalid_proposal_materialized_as_incumbent_placeholder"
        ]
        assert all(
            torch.equal(
                candidate.state_dict()[name], prepared.incumbent.state_dict()[name]
            )
            for name in candidate.state_dict()
        )

    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
    )
    assert result.receipt["status"] == adapter._STRUCTURAL_NOOP_STATUS
    assert result.receipt["installed"] is False
    assert result.receipt["install_count"] == 0
    assert result.receipt["incumbent_selected_by_heldout_gates"] is False
    assert result.receipt["body_recipe_id"] == body_recipe
    assert stack_adapter._authenticated_heldout_incumbent_noop(
        result.receipt,
        component="mlp",
        before_sha256=before,
        after_sha256=before,
        changed_state_keys=(),
    )
    assert fingerprint_lfm2_parent(model).identity_sha256 == before


def test_final_w2_deferral_exports_four_banks_and_installs_selected_producers() -> None:
    model = _Target()
    evidence = _evidence(model, terminal=True)
    before = fingerprint_lfm2_parent(model).identity_sha256
    prepared = _prepare(model, evidence, terminal=True)
    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
        defer_down_weight=True,
    )
    assert result.receipt["status"] == "installed-w1-w3-final-w2-deferred"
    assert result.receipt["installed"] is True
    assert result.receipt["install_count"] == 1
    assert result.receipt["changed_parameter_roles"] == ["w1.weight", "w3.weight"]
    assert result.receipt["final_layer_mlp_w2_deferred_to_fused_terminal"] is True
    assert fingerprint_lfm2_parent(model).identity_sha256 != before
    assert result.receipt["cache_key"] == _sha("cache")
    assert result.receipt["cache_manifest_sha256"] == _sha("cache-manifest")
    assert result.receipt["evidence_sha256"] == [
        item.evidence_sha256 for item in evidence
    ]
    assert set(result.final_layer_mlp_evidence) == {
        "fit",
        "fit_check",
        "selection",
        "validation",
    }
    for partition, bank in result.final_layer_mlp_evidence.items():
        assert bank.split == partition
        assert bank.verify_authentication()
        assert (
            result.receipt["terminal_action_evidence_sha256"][partition]
            == bank.authentication_sha256
        )


def test_qdeim_final_layer_keeps_terminal_w2_deferral_semantics() -> None:
    model = _Target()
    with torch.no_grad():
        for parameter in model.model.layers[0].feed_forward.parameters():
            parameter.zero_()
    evidence = tuple(
        replace(
            item,
            parent_final_norm_output_target_space=(
                item.parent_post_mlp_residual_target_space
            ),
        )
        for item in _cold_start_parent_endpoint_evidence(model)
    )
    prepared = _prepare(
        model,
        evidence,
        terminal=True,
        body_recipe="nonlinear_qdeim_fit_corrected",
    )
    before_w2 = model.model.layers[0].feed_forward.w2.weight.detach().clone()

    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
        defer_down_weight=True,
    )

    assert result.receipt["status"] == "installed-w1-w3-final-w2-deferred"
    assert result.receipt["changed_parameter_roles"] == ["w1.weight", "w3.weight"]
    assert result.receipt["final_layer_mlp_w2_deferred_to_fused_terminal"] is True
    assert (
        result.receipt["analytic_linear_cluster_tangent_and_qdeim_solves_only"] is True
    )
    assert result.receipt["analytic_linear_and_cluster_solves_only"] is False
    assert torch.equal(
        model.model.layers[0].feed_forward.w2.weight.detach(),
        before_w2,
    )
    assert set(result.final_layer_mlp_evidence) == set(adapter._PARTITIONS)
    prior = result.deferred_selected_w2_prior
    assert prior is not None
    assert prior.verify_authentication()
    assert prior.body_recipe_id == "nonlinear_qdeim_fit_corrected"
    assert prior.selected_candidate_sha256 == (
        prepared.selected_candidate.candidate_sha256
    )
    assert torch.equal(prior.down_weight, prepared.selected_candidate.down_weight)
    assert prior.live_deferred_w2_sha256 == (
        adapter.lfm2_cpcf_terminal_action_tensor_sha256(before_w2)
    )
    assert result.receipt["deferred_selected_w2_prior_installed_before_terminal"] is (
        False
    )
    assert (
        result.receipt["deferred_selected_w2_prior_authentication_sha256"]
        == prior.authentication_sha256
    )


def test_layer_stack_mlp_seam_threads_qdeim_prior_without_installing_w2() -> None:
    model = _Target()
    with torch.no_grad():
        for parameter in model.model.layers[0].feed_forward.parameters():
            parameter.zero_()
    evidence = tuple(
        replace(
            item,
            parent_final_norm_output_target_space=(
                item.parent_post_mlp_residual_target_space
            ),
        )
        for item in _cold_start_parent_endpoint_evidence(model)
    )
    before_w2 = model.model.layers[0].feed_forward.w2.weight.detach().clone()

    receipt, terminal_evidence, prior = stack_adapter._install_mlp(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache=SimpleNamespace(
            cache_key=_sha("cache"),
            manifest={"manifest_sha256": _sha("cache-manifest")},
        ),
        bridge=SimpleNamespace(artifact_sha256=_sha("bridge")),
        installer=adapter.compile_install_lfm2_semantic_mlp_postfold,
        final_layer=True,
        parameter_count=stack_adapter._parameter_count(model),
        body_recipe="nonlinear_qdeim_fit_corrected",
    )

    assert prior is not None
    assert prior.verify_authentication()
    assert (
        receipt["deferred_selected_w2_prior_authentication_sha256"]
        == prior.authentication_sha256
    )
    assert set(terminal_evidence) == set(adapter._PARTITIONS)
    assert torch.equal(
        model.model.layers[0].feed_forward.w2.weight.detach(),
        before_w2,
    )


def test_stale_replay_fails_closed_and_copy_failure_rolls_back_exactly(
    monkeypatch,
) -> None:
    stale_model = _Target()
    stale_prepared = _prepare(stale_model, _evidence(stale_model))
    with torch.no_grad():
        stale_model.model.layers[0].feed_forward.w1.weight.add_(0.01)
    with pytest.raises(adapter.Lfm2SemanticMlpRefoldError, match="stale"):
        adapter.install_lfm2_prepared_semantic_mlp_postfold(
            target_model=stale_model,
            prepared=stale_prepared,
        )

    model = _Target()
    prepared = _prepare(model, _evidence(model))
    before = fingerprint_lfm2_parent(model).identity_sha256
    target_w2 = model.model.layers[0].feed_forward.w2.weight
    original = adapter._copy_candidate

    def copy_then_fail(parameter, value):
        original(parameter, value)
        if parameter is target_w2:
            raise RuntimeError("injected after W2 copy")

    monkeypatch.setattr(adapter, "_copy_candidate", copy_then_fail)
    with pytest.raises(RuntimeError, match="injected"):
        adapter.install_lfm2_prepared_semantic_mlp_postfold(
            target_model=model,
            prepared=prepared,
        )
    assert fingerprint_lfm2_parent(model).identity_sha256 == before


def test_repeated_complete_cache_scale_accepts_all_365_phases() -> None:
    model = _Target()
    evidence = _evidence(model, batches=16, total_phases=365)
    prepared = _prepare(model, evidence)
    assert len(prepared.evidence_sha256) == 365
    assert (
        sum(
            prepared.incumbent.metrics[name].endpoint_rows
            for name in ("fit", "fit_check", "selection", "validation")
        )
        == 365
    )
    assert set(prepared.partition_batch_indices) == {
        "fit",
        "fit_check",
        "selection",
        "validation",
    }
    assert prepared.selection_receipt["candidate_frozen_before_validation"] is True


def test_module_ast_contains_no_execution_or_training_calls() -> None:
    path = Path(adapter.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_attributes = {
        "forward",
        "generate",
        "backward",
        "step",
        "zero_grad",
        "grad",
        "from_pretrained",
    }
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called.isdisjoint(forbidden_attributes)
