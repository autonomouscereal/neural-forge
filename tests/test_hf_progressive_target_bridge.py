from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
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
from neural_forge.hf_dense_gqa_qkvo_residual_adapter import (
    DenseGQAQKVOResidualProjection,
    DenseGQASite,
    SparseDenseGQAQKVOResidualRegistry,
)
from neural_forge.hf_factorized_vocab import exact_unique_parameter_count
from neural_forge.hf_parent_compiler import HFParentCompilerResult
from neural_forge.hf_progressive_target_bridge import (
    QWEN3_1P4B_EXACT_PARAMETERS,
    QWEN3_4B_EXACT_PARAMETERS,
    QWEN3_600M_EXACT_PARAMETERS,
    SMOLLM2_135M_EXACT_PARAMETERS,
    DenseGQATargetShape,
    ExternalBehaviorGateReceipt,
    ProgressiveDenseGQAStage,
    ProgressiveTargetBridgeError,
    ProgressiveTargetPublicationError,
    certify_native_dense_gqa_target,
    default_qwen4b_progressive_target_plan,
    install_exact_target_qkvo_bridge,
    install_parent_compiler_result_qkvo_bridge,
)


def test_default_progressive_plan_has_exact_native_counts_and_bounded_steps() -> None:
    plan = default_qwen4b_progressive_target_plan()
    assert plan.source_unique_parameters == QWEN3_4B_EXACT_PARAMETERS
    assert [stage.exact_unique_parameters for stage in plan.stages] == [
        QWEN3_1P4B_EXACT_PARAMETERS,
        QWEN3_600M_EXACT_PARAMETERS,
        SMOLLM2_135M_EXACT_PARAMETERS,
    ]
    assert [stage.shape.unique_parameter_count for stage in plan.stages] == [
        QWEN3_1P4B_EXACT_PARAMETERS,
        QWEN3_600M_EXACT_PARAMETERS,
        SMOLLM2_135M_EXACT_PARAMETERS,
    ]
    report = plan.as_dict()
    assert report["minimum_25x_endpoint"] is True
    assert report["overall_compression_ratio"] == pytest.approx(29.9034892523)
    assert [
        stage["single_stage_compression_ratio"] for stage in report["stages"]
    ] == pytest.approx([2.851667, 2.348485, 4.465139], rel=2e-6)
    assert report["every_stage_is_native_target_sized"] is True
    assert report["capability_proven_by_plan"] is False


def test_target_shape_formula_separates_qwen_qk_norms_from_smollm() -> None:
    smollm = default_qwen4b_progressive_target_plan().stages[-1].shape
    assert smollm.parameter_breakdown()["per_layer"]["query_key_head_norms"] == 0
    assert smollm.unique_parameter_count == 134_515_008

    with_qk_norms = DenseGQATargetShape(
        **{
            **smollm.as_dict(),
            "family": "qwen3",
            "query_key_head_norms": True,
        }
    )
    assert with_qk_norms.unique_parameter_count == 134_518_848


def test_stage_rejects_a_declared_count_that_shape_cannot_realize() -> None:
    shape = default_qwen4b_progressive_target_plan().stages[0].shape
    with pytest.raises(ProgressiveTargetBridgeError, match="native shape contains"):
        ProgressiveDenseGQAStage(
            name="wrong",
            shape=shape,
            exact_unique_parameters=shape.unique_parameter_count - 1,
            tokenizer_transition="preserve",
            purpose="must fail",
        )


class _ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        q = self.q_proj(hidden)
        k = self.k_proj(hidden).repeat_interleave(2, dim=-1)
        v = self.v_proj(hidden).repeat_interleave(2, dim=-1)
        return self.o_proj(q + k + v)


class _ToyDenseGQAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(17, 8)
        self.attentions = nn.ModuleList([_ToyAttention(), _ToyAttention()])
        self.norm = nn.LayerNorm(8, elementwise_affine=True)
        self.lm_head = nn.Linear(8, 17, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden = self.embedding(input_ids)
        for attention in self.attentions:
            hidden = hidden + attention(hidden)
        return self.lm_head(self.norm(hidden))


def _toy_layout() -> DecoderLayout:
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
        prefix = f"attentions.{index}"
        blocks.append(
            DecoderBlockLayout(
                index=index,
                path=prefix,
                attention_input_norm=attention_norm,
                attention=AttentionLayout(
                    module_path=prefix,
                    projection_style="split_qkv",
                    fused_qkv_path=None,
                    query_path=f"{prefix}.q_proj",
                    key_path=f"{prefix}.k_proj",
                    value_path=f"{prefix}.v_proj",
                    output_path=f"{prefix}.o_proj",
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
                    module_path="norm",
                    kind="gated",
                    input_projection_path=None,
                    gate_projection_path="norm",
                    up_projection_path="norm",
                    down_projection_path="norm",
                    activation_path="norm",
                    activation_name="silu",
                    activation_module_type="SiLUActivation",
                    hidden_size=8,
                    intermediate_size=16,
                    projection_weight_layout="output_input",
                    projection_bias=False,
                ),
                residual=SERIAL_PRE_NORM_RESIDUAL,
            )
        )
    return DecoderLayout(
        family="llama",
        model_type="toy_dense_gqa",
        model_class="_ToyDenseGQAModel",
        hidden_size=8,
        intermediate_size=16,
        blocks_path="attentions",
        embeddings=EmbeddingLayout(
            token_path="embedding",
            position_path=None,
            dropout_path=None,
            vocab_size=17,
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
            has_bias=False,
        ),
        lm_head_path="lm_head",
        tie_word_embeddings=False,
    )


def _approval(*, passed: bool = True, internal_only: bool = False):
    return ExternalBehaviorGateReceipt(
        evidence_sha256="a" * 64,
        evaluation_contract_sha256="b" * 64,
        heldout_action_count=16,
        free_running_rollout_count=8,
        passed=passed,
        internal_similarity_only=internal_only,
    )


def _activate(session) -> None:
    with torch.no_grad():
        for role in ("q", "k", "v", "o"):
            getattr(session.handle.adapter_registry, f"{role}_left").fill_(0.02)
            getattr(session.handle.adapter_registry, f"{role}_right").fill_(0.03)


def test_exact_target_bridge_bakes_to_original_native_topology_without_sidecar() -> (
    None
):
    torch.manual_seed(7)
    model = _ToyDenseGQAModel().eval()
    layout = _toy_layout()
    exact = exact_unique_parameter_count(model)
    keys_before = tuple(sorted(model.state_dict()))
    inputs = torch.tensor([[1, 2, 3]])
    baseline = model(inputs).detach().clone()

    session = install_exact_target_qkvo_bridge(
        model,
        exact_unique_parameters=exact,
        sites=(DenseGQASite(1, 1),),
        rank=2,
        layout=layout,
        initialization_seed=11,
    )
    assert torch.equal(model(inputs), baseline)
    assert session.handle.report.model_parameter_count_before == exact
    assert session.handle.report.model_parameter_count_installed > exact
    assert (
        session.installation_report()["checkpoint_publication_allowed_while_installed"]
        is False
    )

    _activate(session)
    live_logits = model(inputs).detach().clone()
    publication = session.bake_for_publication(_approval())
    baked_logits = model(inputs).detach().clone()

    torch.testing.assert_close(baked_logits, live_logits, rtol=2e-5, atol=2e-5)
    assert tuple(sorted(model.state_dict())) == keys_before
    assert exact_unique_parameter_count(model) == exact
    assert all(type(attention.q_proj) is nn.Linear for attention in model.attentions)
    assert all(type(attention.k_proj) is nn.Linear for attention in model.attentions)
    assert all(type(attention.v_proj) is nn.Linear for attention in model.attentions)
    assert all(type(attention.o_proj) is nn.Linear for attention in model.attentions)
    assert not any(
        isinstance(module, DenseGQAQKVOResidualProjection) for module in model.modules()
    )
    assert not any(
        isinstance(module, SparseDenseGQAQKVOResidualRegistry)
        for module in model.modules()
    )
    assert publication["exact_budget_match"] is True
    assert publication["published_adapter_parameter_count"] == 0
    assert publication["published_parent_runtime_reference_count"] == 0
    assert publication["native_state_key_topology_restored"] is True
    assert publication["capability_proven_by_this_module"] is False

    reloaded_certificate = certify_native_dense_gqa_target(
        model,
        exact_unique_parameters=exact,
        layout=layout,
    )
    assert reloaded_certificate.exact_unique_parameters == exact
    assert reloaded_certificate.dense_qkvo_projection_count == 8


@pytest.mark.parametrize(
    "receipt",
    [_approval(passed=False), _approval(internal_only=True)],
)
def test_publication_requires_real_external_behavior_and_rollout_gate(receipt) -> None:
    model = _ToyDenseGQAModel()
    exact = exact_unique_parameter_count(model)
    session = install_exact_target_qkvo_bridge(
        model,
        exact_unique_parameters=exact,
        sites=(DenseGQASite(0, 0),),
        rank=1,
        layout=_toy_layout(),
    )
    with pytest.raises(ProgressiveTargetPublicationError):
        session.bake_for_publication(receipt)
    assert session.is_live
    session.rollback()
    assert exact_unique_parameter_count(model) == exact


def test_wrong_exact_budget_fails_before_installing_any_adapter() -> None:
    model = _ToyDenseGQAModel()
    exact = exact_unique_parameter_count(model)
    keys = tuple(sorted(model.state_dict()))
    with pytest.raises(ProgressiveTargetBridgeError, match="expected"):
        install_exact_target_qkvo_bridge(
            model,
            exact_unique_parameters=exact - 1,
            sites=(DenseGQASite(0, 0),),
            rank=1,
            layout=_toy_layout(),
        )
    assert tuple(sorted(model.state_dict())) == keys
    assert not any(
        isinstance(module, SparseDenseGQAQKVOResidualRegistry)
        for module in model.modules()
    )


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

    def as_dict(self):
        return {
            "target_budget_exact": self.target_budget_exact,
            "bootstrap_accepted": self.bootstrap_accepted,
            "target_unique_parameter_count": self.count,
            "target_parameter_budget": self.count,
        }


def test_parent_compiler_result_connects_directly_to_native_qkvo_transaction() -> None:
    model = _ToyDenseGQAModel()
    exact = exact_unique_parameter_count(model)
    result = HFParentCompilerResult(
        child_model=model,
        execution_graph=None,  # type: ignore[arg-type]
        bootstrap_result=None,  # type: ignore[arg-type]
        report=_AcceptedCompilerReport(exact),  # type: ignore[arg-type]
    )
    session = install_parent_compiler_result_qkvo_bridge(
        result,
        sites=(DenseGQASite(0, 0),),
        rank=1,
        layout=_toy_layout(),
    )
    assert session.parent_compiler_report_sha256 is not None
    assert session.installation_report()["parent_compiler_report_sha256"] is not None
    session.rollback()
