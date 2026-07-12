from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neural_forge.hf_arch_layout import (
    AttentionLayout,
    DecoderBlockLayout,
    DecoderLayout,
    EmbeddingLayout,
    MLPLayout,
    NormLayout,
    PositionLayout,
    SERIAL_PRE_NORM_RESIDUAL,
)
from neural_forge.hf_dense_gqa_qkvo_residual_adapter import DenseGQASite
from neural_forge.hf_dense_swiglu_projection_residual_adapter import (
    DenseSwiGLUResidualProjection,
    SparseDenseSwiGLUProjectionResidualRegistry,
    install_dense_swiglu_projection_residual_adapter,
)
from neural_forge.hf_factorized_vocab import exact_unique_parameter_count
from neural_forge.hf_parent_compiler import HFParentCompilerResult
from neural_forge.hf_progressive_target_bridge import (
    ExactTargetCompilerTransaction,
    ExternalBehaviorGateReceipt,
    ProgressiveTargetBridgeError,
    ProgressiveTargetPublicationError,
    certify_native_dense_gqa_target,
    install_exact_target_compiler_transaction,
    install_parent_compiler_result_target_native_transaction,
)


class _ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        query = self.q_proj(hidden)
        key = self.k_proj(hidden).repeat_interleave(2, dim=-1)
        value = self.v_proj(hidden).repeat_interleave(2, dim=-1)
        return self.o_proj(query + key + value)


class _ToySwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 12, bias=False)
        self.up_proj = nn.Linear(8, 12, bias=False)
        self.down_proj = nn.Linear(12, 8, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _ToyAttention()
        self.mlp = _ToySwiGLU()

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.self_attn(hidden)
        return hidden + self.mlp(hidden)


class _ToyTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(19, 8)
        self.layers = nn.ModuleList([_ToyBlock(), _ToyBlock()])
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 19, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(self.norm(hidden))


def _layout() -> DecoderLayout:
    attention_norm = NormLayout(
        path="norm",
        kind="rms_norm",
        placement="pre_attention",
        epsilon=1e-5,
        has_bias=False,
    )
    mlp_norm = NormLayout(
        path="norm",
        kind="rms_norm",
        placement="pre_mlp",
        epsilon=1e-5,
        has_bias=False,
    )
    blocks = []
    for index in range(2):
        block = f"layers.{index}"
        attention = f"{block}.self_attn"
        mlp = f"{block}.mlp"
        blocks.append(
            DecoderBlockLayout(
                index=index,
                path=block,
                attention_input_norm=attention_norm,
                attention=AttentionLayout(
                    module_path=attention,
                    projection_style="split_qkv",
                    fused_qkv_path=None,
                    query_path=f"{attention}.q_proj",
                    key_path=f"{attention}.k_proj",
                    value_path=f"{attention}.v_proj",
                    output_path=f"{attention}.o_proj",
                    query_heads=4,
                    key_value_heads=2,
                    head_dim=2,
                    hidden_size=8,
                    projection_weight_layout="output_input",
                    projection_bias=False,
                    attention_dropout=0.0,
                ),
                mlp_input_norm=mlp_norm,
                mlp=MLPLayout(
                    module_path=mlp,
                    kind="gated",
                    input_projection_path=None,
                    gate_projection_path=f"{mlp}.gate_proj",
                    up_projection_path=f"{mlp}.up_proj",
                    down_projection_path=f"{mlp}.down_proj",
                    activation_path=f"{mlp}.act_fn",
                    activation_name="silu",
                    activation_module_type="torch.nn.SiLU",
                    hidden_size=8,
                    intermediate_size=12,
                    projection_weight_layout="output_input",
                    projection_bias=False,
                ),
                residual=SERIAL_PRE_NORM_RESIDUAL,
            )
        )
    return DecoderLayout(
        family="llama",
        model_type="toy_target_native",
        model_class="_ToyTarget",
        hidden_size=8,
        intermediate_size=12,
        blocks_path="layers",
        embeddings=EmbeddingLayout(
            token_path="embedding",
            position_path=None,
            dropout_path=None,
            vocab_size=19,
            hidden_size=8,
        ),
        position=PositionLayout(
            kind="rope",
            operator_paths=(),
            max_positions=64,
            rotary_dim=2,
            rope_theta=10_000.0,
            rope_type="default",
        ),
        blocks=tuple(blocks),
        final_norm=NormLayout(
            path="norm",
            kind="rms_norm",
            placement="final",
            epsilon=1e-5,
            has_bias=True,
        ),
        lm_head_path="lm_head",
        tie_word_embeddings=False,
    )


def _approval(*, passed: bool = True) -> ExternalBehaviorGateReceipt:
    return ExternalBehaviorGateReceipt(
        evidence_sha256="a" * 64,
        evaluation_contract_sha256="b" * 64,
        heldout_action_count=12,
        free_running_rollout_count=6,
        passed=passed,
    )


def _activate(transaction: ExactTargetCompilerTransaction) -> None:
    with torch.no_grad():
        for role in ("q", "k", "v", "o"):
            getattr(transaction.qkvo_handle.adapter_registry, f"{role}_left").fill_(
                0.015
            )
        for role in ("gate", "up", "down"):
            getattr(transaction.swiglu_handle.registry, f"{role}_left").fill_(0.02)


def test_dense_swiglu_adapter_is_zero_identity_then_bakes_all_three_roles() -> None:
    torch.manual_seed(21)
    model = _ToyTarget().eval()
    inputs = torch.tensor([[1, 2, 3]])
    baseline = model(inputs).detach().clone()
    exact = exact_unique_parameter_count(model)
    keys = tuple(sorted(model.state_dict()))
    handle = install_dense_swiglu_projection_residual_adapter(
        model,
        {index: block.mlp for index, block in enumerate(model.layers)},
        layer_indices=(1,),
        hidden_size=8,
        intermediate_size=12,
        rank=2,
        initialization_seed=9,
    )
    assert torch.equal(model(inputs), baseline)
    assert handle.report.adapter_parameter_count == 3 * 2 * (8 + 12)
    assert handle.report.model_parameter_count_installed == exact + 120
    with torch.no_grad():
        for role in ("gate", "up", "down"):
            getattr(handle.registry, f"{role}_left").fill_(0.025)
    live = model(inputs).detach().clone()
    bake = handle.bake_for_publication()
    baked = model(inputs).detach().clone()
    torch.testing.assert_close(baked, live, rtol=2e-5, atol=2e-5)
    assert bake.affected_projection_count == 3
    assert exact_unique_parameter_count(model) == exact
    assert tuple(sorted(model.state_dict())) == keys
    assert not any(
        isinstance(module, DenseSwiGLUResidualProjection) for module in model.modules()
    )
    assert not any(
        isinstance(module, SparseDenseSwiGLUProjectionResidualRegistry)
        for module in model.modules()
    )


def test_combined_transaction_counts_temporary_state_and_publishes_exact_native() -> (
    None
):
    torch.manual_seed(22)
    model = _ToyTarget().eval()
    layout = _layout()
    inputs = torch.tensor([[2, 4, 6]])
    baseline = model(inputs).detach().clone()
    exact = exact_unique_parameter_count(model)
    keys = tuple(sorted(model.state_dict()))
    transaction = install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact,
        qkvo_sites=(DenseGQASite(0, 1), DenseGQASite(1, 0)),
        qkvo_rank=2,
        swiglu_layer_indices=(0, 1),
        swiglu_rank=2,
        layout=layout,
        initialization_seed=13,
    )
    assert torch.equal(model(inputs), baseline)
    report = transaction.installation_report()
    assert report["zero_step_identity"] is True
    assert report["temporary_installed_parameters"] == (
        exact + report["temporary_factor_parameters"]
    )
    assert report["published_parameters_if_promoted"] == exact
    scope_report = transaction.output_training_scope().as_dict()
    assert scope_report["weight_matching_allowed"] is False
    assert scope_report["activation_matching_allowed"] is False
    with transaction.output_training_scope() as trainable:
        trainable_ids = {id(value) for value in trainable}
        assert trainable_ids
        assert all(
            not value.requires_grad
            for value in model.parameters()
            if id(value) not in trainable_ids
        )
        model(inputs).sum().backward()
        assert all(value.grad is not None for value in trainable)
    _activate(transaction)
    live = model(inputs).detach().clone()
    publication = transaction.bake_for_publication(_approval())
    baked = model(inputs).detach().clone()
    torch.testing.assert_close(baked, live, rtol=3e-5, atol=3e-5)
    assert tuple(sorted(model.state_dict())) == keys
    assert exact_unique_parameter_count(model) == exact
    assert publication["exact_budget_match"] is True
    assert publication["published_adapter_parameter_count"] == 0
    assert publication["output_action_gate_required"] is True
    assert publication["free_running_rollout_gate_required"] is True
    assert publication["weight_or_activation_similarity_can_promote"] is False
    certificate = certify_native_dense_gqa_target(
        model,
        exact_unique_parameters=exact,
        layout=layout,
        require_dense_swiglu=True,
    )
    assert certificate.dense_qkvo_projection_count == 8
    assert certificate.dense_swiglu_projection_count == 6


def test_combined_rejection_rollback_is_bit_exact() -> None:
    torch.manual_seed(23)
    model = _ToyTarget().eval()
    inputs = torch.tensor([[3, 5, 7]])
    baseline = model(inputs).detach().clone()
    exact = exact_unique_parameter_count(model)
    transaction = install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact,
        qkvo_sites=(DenseGQASite(0, 0),),
        qkvo_rank=1,
        swiglu_layer_indices=(1,),
        swiglu_rank=1,
        layout=_layout(),
    )
    _activate(transaction)
    transaction.rollback()
    assert torch.equal(model(inputs), baseline)
    assert exact_unique_parameter_count(model) == exact


def test_gate_rejection_does_not_mutate_live_transaction() -> None:
    model = _ToyTarget()
    exact = exact_unique_parameter_count(model)
    transaction = install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact,
        qkvo_sites=(DenseGQASite(0, 0),),
        qkvo_rank=1,
        swiglu_layer_indices=(0,),
        swiglu_rank=1,
        layout=_layout(),
    )
    with pytest.raises(ProgressiveTargetPublicationError):
        transaction.bake_for_publication(_approval(passed=False))
    assert transaction.is_live
    transaction.rollback()


def test_late_bake_failure_restores_native_child_atomically(monkeypatch) -> None:
    torch.manual_seed(24)
    model = _ToyTarget().eval()
    inputs = torch.tensor([[4, 8]])
    baseline = model(inputs).detach().clone()
    exact = exact_unique_parameter_count(model)
    keys = tuple(sorted(model.state_dict()))
    transaction = install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact,
        qkvo_sites=(DenseGQASite(1, 1),),
        qkvo_rank=1,
        swiglu_layer_indices=(1,),
        swiglu_rank=1,
        layout=_layout(),
    )
    _activate(transaction)

    def fail_after_qkvo_bake():
        raise RuntimeError("synthetic later bake failure")

    monkeypatch.setattr(
        transaction.swiglu_handle,
        "bake_for_publication",
        fail_after_qkvo_bake,
    )
    with pytest.raises(RuntimeError, match="synthetic later bake failure"):
        transaction.bake_for_publication(_approval())
    assert exact_unique_parameter_count(model) == exact
    assert tuple(sorted(model.state_dict())) == keys
    assert torch.equal(model(inputs), baseline)


@dataclass(frozen=True)
class _AcceptedCompilerReport:
    count: int
    target_budget_exact: bool = True
    bootstrap_accepted: bool = True

    @property
    def target_unique_parameter_count(self) -> int:
        return self.count

    @property
    def target_parameter_budget(self) -> int:
        return self.count

    def as_dict(self) -> dict[str, object]:
        return {
            "target_budget_exact": self.target_budget_exact,
            "bootstrap_accepted": self.bootstrap_accepted,
            "target_unique_parameter_count": self.count,
            "target_parameter_budget": self.count,
        }


def test_parent_compiler_result_enters_full_target_native_transaction() -> None:
    model = _ToyTarget()
    exact = exact_unique_parameter_count(model)
    result = HFParentCompilerResult(
        child_model=model,
        execution_graph=None,  # type: ignore[arg-type]
        bootstrap_result=None,  # type: ignore[arg-type]
        report=_AcceptedCompilerReport(exact),  # type: ignore[arg-type]
    )
    transaction = install_parent_compiler_result_target_native_transaction(
        result,
        qkvo_sites=(DenseGQASite(0, 0),),
        qkvo_rank=1,
        swiglu_layer_indices=(0,),
        swiglu_rank=1,
        layout=_layout(),
    )
    assert transaction.parent_compiler_report_sha256 is not None
    transaction.rollback()


def test_wrong_budget_fails_before_any_combined_adapter_is_installed() -> None:
    model = _ToyTarget()
    exact = exact_unique_parameter_count(model)
    keys = tuple(sorted(model.state_dict()))
    with pytest.raises(ProgressiveTargetBridgeError, match="expected"):
        install_exact_target_compiler_transaction(
            model,
            exact_unique_parameters=exact - 1,
            qkvo_sites=(DenseGQASite(0, 0),),
            qkvo_rank=1,
            swiglu_layer_indices=(0,),
            swiglu_rank=1,
            layout=_layout(),
        )
    assert tuple(sorted(model.state_dict())) == keys
    assert not any(
        isinstance(module, SparseDenseSwiGLUProjectionResidualRegistry)
        for module in model.modules()
    )
