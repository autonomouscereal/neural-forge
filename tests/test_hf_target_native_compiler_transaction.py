from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import neural_forge.hf_progressive_target_bridge as target_bridge
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
from neural_forge.hf_dense_lexical_residual_adapter import (
    DenseLexicalResidualEmbedding,
    DenseLexicalResidualLMHead,
    DenseLexicalResidualRegistry,
)
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

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


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
        if transaction.lexical_handle is not None:
            transaction.lexical_handle.registry.embedding_left.fill_(0.01)
            transaction.lexical_handle.registry.lm_head_left.fill_(0.012)


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


def test_composite_lexical_transaction_counts_trains_and_publishes_exact_native() -> (
    None
):
    torch.manual_seed(220)
    model = _ToyTarget().eval()
    layout = _layout()
    inputs = torch.tensor([[1, 7, 11]])
    baseline = model(inputs).detach().clone()
    exact = exact_unique_parameter_count(model)
    keys = tuple(sorted(model.state_dict()))
    transaction = install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact,
        qkvo_sites=(DenseGQASite(0, 0), DenseGQASite(1, 1)),
        qkvo_rank=2,
        swiglu_layer_indices=(0, 1),
        swiglu_rank=2,
        lexical_rank=2,
        layout=layout,
        initialization_seed=31,
    )
    assert transaction.lexical_handle is not None
    assert torch.equal(model(inputs), baseline)
    report = transaction.installation_report()
    lexical_parameters = 2 * 2 * (19 + 8)
    assert report["temporary_lexical_factor_parameters"] == lexical_parameters
    assert report["dense_lexical_installation"]["rank"] == 2
    assert report["role_complete_lexical_boundary"] is True
    assert report["temporary_installed_parameters"] == (
        exact + report["temporary_factor_parameters"]
    )
    assert exact_unique_parameter_count(model) == (
        exact + report["temporary_factor_parameters"]
    )
    assert len(transaction.output_trainable_parameters()) == 18
    scope = transaction.output_training_scope().as_dict()
    assert scope["trainable_lexical_factor_parameters"] == lexical_parameters
    assert scope["trainable_lexical_factor_tensor_count"] == 4
    assert scope["role_complete_lexical_boundary"] is True

    _activate(transaction)
    live = model(inputs).detach().clone()
    publication = transaction.bake_for_publication(_approval())
    baked = model(inputs).detach().clone()
    torch.testing.assert_close(baked, live, rtol=4e-5, atol=4e-5)
    assert publication["dense_lexical_bake"]["affected_dense_matrix_count"] == 2
    assert publication["published_lexical_adapter_parameter_count"] == 0
    assert publication["role_complete_lexical_boundary"] is True
    assert exact_unique_parameter_count(model) == exact
    assert tuple(sorted(model.state_dict())) == keys
    assert not any(
        isinstance(
            module,
            (
                DenseLexicalResidualEmbedding,
                DenseLexicalResidualLMHead,
                DenseLexicalResidualRegistry,
            ),
        )
        for module in model.modules()
    )


def test_composite_success_bakes_lexical_then_qkvo_then_gud(monkeypatch) -> None:
    model = _ToyTarget().eval()
    transaction = install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact_unique_parameter_count(model),
        qkvo_sites=(DenseGQASite(0, 0),),
        qkvo_rank=1,
        swiglu_layer_indices=(0,),
        swiglu_rank=1,
        lexical_rank=1,
        layout=_layout(),
    )
    assert transaction.lexical_handle is not None
    order: list[str] = []
    lexical_bake = transaction.lexical_handle.bake_for_publication
    qkvo_bake = transaction.qkvo_handle.bake_for_publication
    gud_bake = transaction.swiglu_handle.bake_for_publication

    def record_lexical_bake():
        order.append("lexical")
        return lexical_bake()

    def record_qkvo_bake():
        order.append("qkvo")
        return qkvo_bake()

    def record_gud_bake():
        order.append("gud")
        return gud_bake()

    monkeypatch.setattr(
        transaction.lexical_handle,
        "bake_for_publication",
        record_lexical_bake,
    )
    monkeypatch.setattr(
        transaction.qkvo_handle,
        "bake_for_publication",
        record_qkvo_bake,
    )
    monkeypatch.setattr(
        transaction.swiglu_handle,
        "bake_for_publication",
        record_gud_bake,
    )

    transaction.bake_for_publication(_approval())
    assert order == ["lexical", "qkvo", "gud"]


def test_two_guided_owners_are_nested_around_older_publication_baselines(
    monkeypatch,
) -> None:
    model = _ToyTarget().eval()
    transaction = install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact_unique_parameter_count(model),
        qkvo_sites=(DenseGQASite(0, 0),),
        qkvo_rank=1,
        swiglu_layer_indices=(0,),
        swiglu_rank=1,
        lexical_rank=1,
        layout=_layout(),
    )
    assert transaction.lexical_handle is not None
    events: list[str] = []

    class _GuidedOwnerProxy:
        def __init__(self, name, handle) -> None:
            self.name = name
            self.handle = handle
            self.suspended = False
            self.required_suspension = None

        def __getattr__(self, name):
            return getattr(self.handle, name)

        @property
        def is_applied(self) -> bool:
            return self.handle.is_applied

        @property
        def is_baked(self) -> bool:
            return self.handle.is_baked

        @contextmanager
        def suspend_guided_registry_owner_for_nested_publication(self):
            assert self.is_applied
            assert self.suspended is False
            events.append(f"enter-{self.name}")
            self.suspended = True
            try:
                yield
            finally:
                self.suspended = False
                events.append(f"exit-{self.name}")

        def bake_for_publication(self):
            if self.required_suspension is not None:
                assert self.required_suspension.suspended is True
            events.append(f"bake-{self.name}")
            return self.handle.bake_for_publication()

    qkvo = _GuidedOwnerProxy("qkvo", transaction.qkvo_handle)
    gud = _GuidedOwnerProxy("gate-up-down", transaction.swiglu_handle)
    qkvo.required_suspension = gud
    transaction.qkvo_handle = qkvo
    transaction.swiglu_handle = gud
    lexical_bake = transaction.lexical_handle.bake_for_publication

    def assert_nested_lexical_bake():
        assert qkvo.suspended is True
        assert gud.suspended is True
        events.append("bake-lexical")
        return lexical_bake()

    monkeypatch.setattr(
        transaction.lexical_handle,
        "bake_for_publication",
        assert_nested_lexical_bake,
    )

    publication = transaction.bake_for_publication(_approval())

    assert events == [
        "enter-qkvo",
        "enter-gate-up-down",
        "bake-lexical",
        "exit-gate-up-down",
        "exit-qkvo",
        "enter-gate-up-down",
        "bake-qkvo",
        "exit-gate-up-down",
        "bake-gate-up-down",
    ]
    assert (
        publication["guided_gate_up_down_registry_suspended_for_lexical_bake"] is True
    )
    assert publication["guided_gate_up_down_registry_suspended_for_qkvo_bake"] is True
    assert publication["nested_publication_bake_order"] == [
        "lexical",
        "qkvo-inherited-then-guided",
        "gate-up-down-inherited-then-guided",
    ]


def test_two_guided_owners_are_nested_during_direct_rejection(monkeypatch) -> None:
    model = _ToyTarget().eval()
    exact = exact_unique_parameter_count(model)
    transaction = install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact,
        qkvo_sites=(DenseGQASite(0, 0),),
        qkvo_rank=1,
        swiglu_layer_indices=(0,),
        swiglu_rank=1,
        lexical_rank=1,
        layout=_layout(),
    )
    assert transaction.lexical_handle is not None
    events: list[str] = []
    suspended = {"qkvo": False, "gate-up-down": False}

    @contextmanager
    def suspend(name):
        events.append(f"enter-{name}")
        suspended[name] = True
        try:
            yield
        finally:
            suspended[name] = False
            events.append(f"exit-{name}")

    monkeypatch.setattr(
        transaction.qkvo_handle,
        "suspend_guided_registry_owner_for_nested_publication",
        lambda: suspend("qkvo"),
        raising=False,
    )
    monkeypatch.setattr(
        transaction.swiglu_handle,
        "suspend_guided_registry_owner_for_nested_publication",
        lambda: suspend("gate-up-down"),
        raising=False,
    )
    lexical_rollback = transaction.lexical_handle.rollback
    qkvo_rollback = transaction.qkvo_handle.rollback
    gud_rollback = transaction.swiglu_handle.rollback

    def nested_lexical_rollback():
        assert all(suspended.values())
        events.append("rollback-lexical")
        return lexical_rollback()

    def nested_qkvo_rollback():
        assert suspended["gate-up-down"] is True
        events.append("rollback-qkvo")
        return qkvo_rollback()

    def record_gud_rollback():
        assert not any(suspended.values())
        events.append("rollback-gate-up-down")
        return gud_rollback()

    monkeypatch.setattr(transaction.lexical_handle, "rollback", nested_lexical_rollback)
    monkeypatch.setattr(transaction.qkvo_handle, "rollback", nested_qkvo_rollback)
    monkeypatch.setattr(transaction.swiglu_handle, "rollback", record_gud_rollback)

    restored = transaction.rollback()

    assert restored.exact_unique_parameters == exact
    assert events == [
        "enter-qkvo",
        "enter-gate-up-down",
        "rollback-lexical",
        "exit-gate-up-down",
        "exit-qkvo",
        "enter-gate-up-down",
        "rollback-qkvo",
        "exit-gate-up-down",
        "rollback-gate-up-down",
    ]


def test_composite_public_restore_unwinds_baked_handles_in_reverse_order(
    monkeypatch,
) -> None:
    model = _ToyTarget().eval()
    inputs = torch.tensor([[3, 8, 13]])
    baseline = model(inputs).detach().clone()
    exact = exact_unique_parameter_count(model)
    keys = tuple(sorted(model.state_dict()))
    transaction = install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact,
        qkvo_sites=(DenseGQASite(0, 0),),
        qkvo_rank=1,
        swiglu_layer_indices=(0,),
        swiglu_rank=1,
        lexical_rank=1,
        layout=_layout(),
    )
    assert transaction.lexical_handle is not None
    _activate(transaction)
    transaction.bake_for_publication(_approval())
    order: list[str] = []
    gud_restore = transaction.swiglu_handle.restore_prepublication_native
    qkvo_restore = transaction.qkvo_handle.restore_prepublication_native
    lexical_restore = transaction.lexical_handle.restore_prepublication_native

    def record_gud_restore():
        order.append("gud")
        return gud_restore()

    def record_qkvo_restore(*, expected_model_parameter_count=None):
        order.append("qkvo")
        return qkvo_restore(
            expected_model_parameter_count=expected_model_parameter_count
        )

    def record_lexical_restore(*, expected_model_parameter_count=None):
        order.append("lexical")
        return lexical_restore(
            expected_model_parameter_count=expected_model_parameter_count
        )

    monkeypatch.setattr(
        transaction.swiglu_handle,
        "restore_prepublication_native",
        record_gud_restore,
    )
    monkeypatch.setattr(
        transaction.qkvo_handle,
        "restore_prepublication_native",
        record_qkvo_restore,
    )
    monkeypatch.setattr(
        transaction.lexical_handle,
        "restore_prepublication_native",
        record_lexical_restore,
    )

    restored = transaction.restore_prepublication_native()
    assert order == ["gud", "qkvo", "lexical"]
    assert restored.as_dict() == transaction.native_before.as_dict()
    assert transaction.publication_report is None
    assert exact_unique_parameter_count(model) == exact
    assert tuple(sorted(model.state_dict())) == keys
    assert torch.equal(model(inputs), baseline)


def test_composite_publication_report_failure_restores_all_baked_handles(
    monkeypatch,
) -> None:
    model = _ToyTarget().eval()
    inputs = torch.tensor([[4, 9]])
    baseline = model(inputs).detach().clone()
    exact = exact_unique_parameter_count(model)
    keys = tuple(sorted(model.state_dict()))
    transaction = install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact,
        qkvo_sites=(DenseGQASite(0, 0),),
        qkvo_rank=1,
        swiglu_layer_indices=(0,),
        swiglu_rank=1,
        lexical_rank=1,
        layout=_layout(),
    )
    _activate(transaction)

    def fail_gate_report(_self):
        raise RuntimeError("synthetic outer publication report failure")

    monkeypatch.setattr(ExternalBehaviorGateReceipt, "as_dict", fail_gate_report)
    with pytest.raises(RuntimeError, match="outer publication report failure"):
        transaction.bake_for_publication(_approval())

    assert transaction.publication_report is None
    assert exact_unique_parameter_count(model) == exact
    assert tuple(sorted(model.state_dict())) == keys
    assert torch.equal(model(inputs), baseline)


def test_context_manager_restores_baked_transaction_after_later_save_failure() -> None:
    model = _ToyTarget().eval()
    inputs = torch.tensor([[5, 11]])
    baseline = model(inputs).detach().clone()
    exact = exact_unique_parameter_count(model)
    keys = tuple(sorted(model.state_dict()))
    transaction = install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact,
        qkvo_sites=(DenseGQASite(0, 0),),
        qkvo_rank=1,
        swiglu_layer_indices=(0,),
        swiglu_rank=1,
        lexical_rank=1,
        layout=_layout(),
    )
    _activate(transaction)

    with pytest.raises(RuntimeError, match="synthetic artifact save failure"):
        with transaction:
            transaction.bake_for_publication(_approval())
            raise RuntimeError("synthetic artifact save failure")

    assert transaction.publication_report is None
    assert exact_unique_parameter_count(model) == exact
    assert tuple(sorted(model.state_dict())) == keys
    assert torch.equal(model(inputs), baseline)


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


def test_composite_lexical_rejection_rolls_back_in_reverse_install_order() -> None:
    torch.manual_seed(230)
    model = _ToyTarget().eval()
    inputs = torch.tensor([[2, 9]])
    baseline = model(inputs).detach().clone()
    exact = exact_unique_parameter_count(model)
    keys = tuple(sorted(model.state_dict()))
    transaction = install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact,
        qkvo_sites=(DenseGQASite(0, 1),),
        qkvo_rank=1,
        swiglu_layer_indices=(1,),
        swiglu_rank=1,
        lexical_rank=1,
        layout=_layout(),
    )
    _activate(transaction)
    transaction.rollback()
    assert torch.equal(model(inputs), baseline)
    assert exact_unique_parameter_count(model) == exact
    assert tuple(sorted(model.state_dict())) == keys


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


def test_lexical_then_qkvo_bake_failure_restores_all_three_operators(
    monkeypatch,
) -> None:
    torch.manual_seed(240)
    model = _ToyTarget().eval()
    inputs = torch.tensor([[5, 10]])
    baseline = model(inputs).detach().clone()
    exact = exact_unique_parameter_count(model)
    keys = tuple(sorted(model.state_dict()))
    transaction = install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact,
        qkvo_sites=(DenseGQASite(1, 0),),
        qkvo_rank=1,
        swiglu_layer_indices=(0,),
        swiglu_rank=1,
        lexical_rank=1,
        layout=_layout(),
    )
    assert transaction.lexical_handle is not None
    _activate(transaction)

    def fail_after_lexical_and_qkvo_bake():
        raise RuntimeError("synthetic failure after lexical and qkvo bake")

    monkeypatch.setattr(
        transaction.swiglu_handle,
        "bake_for_publication",
        fail_after_lexical_and_qkvo_bake,
    )
    with pytest.raises(RuntimeError, match="after lexical and qkvo bake"):
        transaction.bake_for_publication(_approval())
    assert exact_unique_parameter_count(model) == exact
    assert tuple(sorted(model.state_dict())) == keys
    assert torch.equal(model(inputs), baseline)


def test_lexical_install_failure_unwinds_qkvo_and_gud(monkeypatch) -> None:
    torch.manual_seed(250)
    model = _ToyTarget().eval()
    inputs = torch.tensor([[6, 12]])
    baseline = model(inputs).detach().clone()
    exact = exact_unique_parameter_count(model)
    keys = tuple(sorted(model.state_dict()))

    def fail_lexical_install(*_args, **_kwargs):
        raise RuntimeError("synthetic lexical installation failure")

    monkeypatch.setattr(
        target_bridge,
        "install_dense_lexical_residual_adapter",
        fail_lexical_install,
    )
    with pytest.raises(RuntimeError, match="lexical installation failure"):
        install_exact_target_compiler_transaction(
            model,
            exact_unique_parameters=exact,
            qkvo_sites=(DenseGQASite(0, 0),),
            qkvo_rank=1,
            swiglu_layer_indices=(0,),
            swiglu_rank=1,
            lexical_rank=1,
            layout=_layout(),
        )
    assert exact_unique_parameter_count(model) == exact
    assert tuple(sorted(model.state_dict())) == keys
    assert torch.equal(model(inputs), baseline)


def test_composite_installation_report_failure_unwinds_every_handle(
    monkeypatch,
) -> None:
    torch.manual_seed(260)
    model = _ToyTarget().eval()
    inputs = torch.tensor([[7, 13]])
    baseline = model(inputs).detach().clone()
    exact = exact_unique_parameter_count(model)
    keys = tuple(sorted(model.state_dict()))

    def fail_report(_self):
        raise RuntimeError("synthetic composite report failure")

    monkeypatch.setattr(
        ExactTargetCompilerTransaction,
        "installation_report",
        fail_report,
    )
    with pytest.raises(RuntimeError, match="composite report failure"):
        install_exact_target_compiler_transaction(
            model,
            exact_unique_parameters=exact,
            qkvo_sites=(DenseGQASite(0, 0),),
            qkvo_rank=1,
            swiglu_layer_indices=(0,),
            swiglu_rank=1,
            lexical_rank=1,
            layout=_layout(),
        )
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
        lexical_rank=1,
        layout=_layout(),
    )
    assert transaction.parent_compiler_report_sha256 is not None
    assert transaction.lexical_handle is not None
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
