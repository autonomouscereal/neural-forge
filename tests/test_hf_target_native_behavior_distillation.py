from __future__ import annotations

import copy
from dataclasses import replace
from types import MappingProxyType

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import neural_forge.hf_target_native_behavior_distillation as behavior_distillation
import neural_forge.hf_dense_gqa_qkvo_residual_adapter as guided_qkvo
import neural_forge.hf_dense_swiglu_projection_residual_adapter as guided_gud
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
    GUIDED_QKVO_RIGHT_BASIS_V2_SCHEMA,
    QKVO_OUTPUT_JACOBIAN_AUDIT_V2_SCHEMA,
    DenseGQAQKVOResidualError,
    DenseGQASite,
    guided_qkvo_basis_payload_sha256,
    install_guided_dense_gqa_qkvo_residual_tranche,
)
from neural_forge.hf_dense_swiglu_projection_residual_adapter import (
    install_guided_dense_swiglu_projection_residual_tranche,
)
from neural_forge.hf_factorized_vocab import exact_unique_parameter_count
from neural_forge.hf_progressive_target_bridge import (
    install_exact_target_compiler_transaction,
)
from neural_forge.hf_target_native_behavior_distillation import (
    BehaviorCheckpoint,
    BehaviorDistillationSchedule,
    CandidateBehaviorThresholds,
    CausalBehaviorMetrics,
    TargetNativeBehaviorDistillationError,
    capture_target_native_factor_state,
    load_behavior_domain_suite,
    restore_target_native_factor_state,
    restore_target_native_factor_payload,
    select_behavior_checkpoint,
    train_target_native_behavior_distillation,
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


class _StrictToyTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(19, 8)
        self.layers = nn.ModuleList([_ToyBlock(), _ToyBlock()])
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 19, bias=False)
        self.internal_output_request_count = 0

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        use_cache: bool = False,
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        logits_to_keep: int | None = None,
    ) -> Tensor:
        del attention_mask, return_dict, logits_to_keep
        if use_cache or output_attentions or output_hidden_states:
            self.internal_output_request_count += 1
            raise AssertionError("Output-only trainer requested model internals.")
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
        model_type="strict_output_only_toy",
        model_class="_StrictToyTarget",
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


def _domain(split: str, prompts: list[list[int]], artifact: str):
    return load_behavior_domain_suite(
        split,
        {"prompts": [{"token_ids": value} for value in prompts]},
        artifact_sha256=artifact,
    )


def _transaction(
    model: _StrictToyTarget,
    *,
    lexical_rank: int | None = None,
):
    return install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact_unique_parameter_count(model),
        qkvo_sites=(DenseGQASite(0, 0), DenseGQASite(1, 1)),
        qkvo_rank=2,
        swiglu_layer_indices=(0, 1),
        swiglu_rank=2,
        layout=_layout(),
        initialization_seed=31,
        lexical_rank=lexical_rank,
    )


def _real_guided_v4_transaction(
    model: _StrictToyTarget,
):
    return install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact_unique_parameter_count(model),
        qkvo_sites=tuple(
            DenseGQASite(layer, group) for layer in range(2) for group in range(2)
        ),
        qkvo_rank=2,
        swiglu_layer_indices=(0, 1),
        swiglu_rank=2,
        layout=_layout(),
        initialization_seed=31,
        lexical_rank=2,
    )


def _install_real_guided_qkvo(transaction, monkeypatch: pytest.MonkeyPatch):
    geometry = transaction.qkvo_handle.report.geometry
    sites = transaction.qkvo_handle.report.sites
    accounting = guided_qkvo.DenseGQAQKVOResidualAccounting.build(
        logical_layers=geometry.logical_layers,
        hidden_size=geometry.hidden_size,
        query_heads=geometry.query_heads,
        key_value_heads=geometry.key_value_heads,
        head_dim=geometry.head_dim,
        rank=transaction.qkvo_handle.report.rank,
        site_count=len(sites),
    )
    right_parameter_count = (
        len(sites)
        * transaction.qkvo_handle.report.rank
        * (geometry.hidden_size * 3 + geometry.query_group_width)
    )
    patches = {
        "_GUIDED_STAGE1_LOGICAL_LAYERS": geometry.logical_layers,
        "_GUIDED_STAGE1_HIDDEN_SIZE": geometry.hidden_size,
        "_GUIDED_STAGE1_QUERY_HEADS": geometry.query_heads,
        "_GUIDED_STAGE1_KEY_VALUE_HEADS": geometry.key_value_heads,
        "_GUIDED_STAGE1_HEAD_DIM": geometry.head_dim,
        "_GUIDED_STAGE1_RANK": transaction.qkvo_handle.report.rank,
        "_GUIDED_STAGE1_SITE_COUNT": len(sites),
        "_GUIDED_STAGE1_RIGHT_PARAMETER_COUNT": right_parameter_count,
        "_GUIDED_STAGE1_PARAMETER_COUNT": accounting.adapter_parameters,
    }
    for name, value in patches.items():
        monkeypatch.setattr(guided_qkvo, name, value)
    monkeypatch.setattr(
        behavior_distillation,
        "GUIDED_QKVO_PARAMETER_COUNT",
        accounting.adapter_parameters,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(871)
    factor_tensors = {
        "q_orthogonal_right": torch.randn(
            len(sites),
            transaction.qkvo_handle.report.rank,
            geometry.hidden_size,
            generator=generator,
        ),
        "k_orthogonal_right": torch.randn(
            len(sites),
            transaction.qkvo_handle.report.rank,
            geometry.hidden_size,
            generator=generator,
        ),
        "v_orthogonal_right": torch.randn(
            len(sites),
            transaction.qkvo_handle.report.rank,
            geometry.hidden_size,
            generator=generator,
        ),
        "o_orthogonal_right": torch.randn(
            len(sites),
            transaction.qkvo_handle.report.rank,
            geometry.query_group_width,
            generator=generator,
        ),
    }
    manifest = guided_qkvo._guided_basis_tensor_manifest(factor_tensors)
    manifest_sha = guided_qkvo._canonical_sha256(manifest)
    site_indices = [[site.layer_index, site.physical_group] for site in sites]
    site_sha = guided_qkvo._canonical_sha256(site_indices)
    inherited_sha = guided_qkvo._qkvo_factor_state_sha256(
        transaction.qkvo_handle.adapter_registry
    )
    roles = ("q", "k", "v", "o")
    audit = {
        "schema": QKVO_OUTPUT_JACOBIAN_AUDIT_V2_SCHEMA,
        "passed": True,
        "site_count": len(sites),
        "site_indices": site_indices,
        "site_indices_sha256": site_sha,
        "factor_state_sha256_before": "2" * 64,
        "inherited_qkvo_state_sha256_before": inherited_sha,
        "inherited_qkvo_state_sha256_after": inherited_sha,
        "diagnostic_passed_roles": list(roles),
        "authorized_roles": list(roles),
        "installable_roles": list(roles),
        "per_role": {role: {"passed": True} for role in roles},
        "installer_must_ignore_unauthorized_role_tensors": True,
        "all_four_role_tensors_retained_for_diagnostics": True,
        "basis_manifest": manifest,
        "basis_manifest_sha256": manifest_sha,
    }
    audit_sha = guided_qkvo._canonical_sha256(audit)
    audit["report_sha256"] = audit_sha
    pack = {
        "schema": GUIDED_QKVO_RIGHT_BASIS_V2_SCHEMA,
        "factor_state_sha256": "2" * 64,
        "site_indices": site_indices,
        "site_indices_sha256": site_sha,
        "basis_manifest_sha256": manifest_sha,
        "tensor_count": 4,
        "total_numel": sum(value.numel() for value in factor_tensors.values()),
        "total_nbytes": sum(
            value.numel() * value.element_size() for value in factor_tensors.values()
        ),
        "factors": manifest,
        "factor_tensors": factor_tensors,
        "diagnostic_passed_roles": list(roles),
        "authorized_roles": list(roles),
        "installable_roles": list(roles),
        "authorized_for_training": True,
        "installer_must_ignore_unauthorized_role_tensors": True,
        "all_four_role_tensors_retained_for_diagnostics": True,
        "contains_model_state": False,
        "contains_parent_state": False,
        "contains_optimizer_state": False,
        "contains_gradients": False,
        "contains_activations": False,
        "contains_hidden_states": False,
    }
    return install_guided_dense_gqa_qkvo_residual_tranche(
        transaction.qkvo_handle,
        audit_report=audit,
        basis_pack=pack,
        expected_audit_report_sha256=audit_sha,
        expected_basis_envelope_file_sha256="a" * 64,
        expected_basis_payload_sha256=guided_qkvo_basis_payload_sha256(pack),
        expected_basis_manifest_sha256=manifest_sha,
    )


def _install_real_guided_gud(transaction):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(947)
    rows = {
        "gate": torch.randn(2, 2, 8, generator=generator),
        "up": torch.randn(2, 2, 8, generator=generator),
        "down": torch.randn(2, 2, 12, generator=generator),
    }
    hashes = {role: guided_gud._tensor_sha256(value) for role, value in rows.items()}
    return install_guided_dense_swiglu_projection_residual_tranche(
        transaction.swiglu_handle,
        guided_right_rows=rows,
        source_basis_tensor_sha256s=hashes,
        expected_audit_report_sha256="1" * 64,
        expected_basis_envelope_file_sha256="2" * 64,
        expected_basis_payload_sha256="3" * 64,
        expected_basis_manifest_sha256="4" * 64,
    )


class _TinyGuidedRegistry(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for role_index, role in enumerate(("q", "k", "v", "o"), start=1):
            setattr(
                self,
                f"{role}_left",
                nn.Parameter(torch.zeros(3, dtype=torch.float32)),
            )
            setattr(
                self,
                f"{role}_right",
                nn.Parameter(
                    torch.tensor([0.125, -0.0625, 0.03125], dtype=torch.float32)
                    * role_index
                ),
            )


class _TinyGuidedTranche:
    def __init__(self, transaction, registry: _TinyGuidedRegistry) -> None:
        self.inherited_handle = transaction.qkvo_handle
        self.guided_registry = registry
        self.training_stage = "guided-left"
        self._stages = ("guided-left", "guided", "joint")
        self._source_binding = MappingProxyType(
            {
                "audit_report_sha256": "1" * 64,
                "basis_envelope_file_sha256": "2" * 64,
                "basis_payload_sha256": "3" * 64,
                "basis_manifest_sha256": "4" * 64,
                "source_basis_tensor_sha256s": MappingProxyType(
                    {
                        role: behavior_distillation._output_jacobian_tensor_sha256(
                            getattr(registry, f"{role}_right")
                        )
                        for role in ("q", "k", "v", "o")
                    }
                ),
                "inherited_installation_sha256": "5" * 64,
                "inherited_factor_state_sha256_at_install": "6" * 64,
                "inherited_handle_identity": id(transaction.qkvo_handle),
                "inherited_registry_identity": id(
                    transaction.qkvo_handle.adapter_registry
                ),
                "adapter_site_indices_sha256": "7" * 64,
                "audit_site_indices_sha256": "8" * 64,
            }
        )

    def assert_live(self):
        return None

    def assert_structurally_live(self):
        return None

    def factor_source_binding(self):
        return self._source_binding

    def set_training_stage(self, stage):
        if self._stages.index(stage) < self._stages.index(self.training_stage):
            raise RuntimeError("guided training stages cannot move backward")
        self.training_stage = stage
        inherited_trainable = stage == "joint"
        for value in self.inherited_handle.adapter_registry.parameters(recurse=False):
            value.requires_grad_(inherited_trainable)
        guided_right_trainable = stage in {"guided", "joint"}
        for name, value in self.guided_registry.named_parameters(recurse=False):
            value.requires_grad_(name.endswith("_left") or guided_right_trainable)
        return self.output_trainable_parameters()

    def output_trainable_parameters(self):
        inherited = tuple(
            self.inherited_handle.adapter_registry.parameters(recurse=False)
        )
        guided = tuple(self.guided_registry.parameters(recurse=False))
        if self.training_stage == "guided-left":
            return tuple(
                value
                for name, value in self.guided_registry.named_parameters(recurse=False)
                if name.endswith("_left")
            )
        if self.training_stage == "guided":
            return guided
        return (*inherited, *guided)


class _TinyGuidedGUDRegistry(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for role_index, role in enumerate(("gate", "up", "down"), start=1):
            setattr(
                self,
                f"{role}_left",
                nn.Parameter(torch.zeros(3, dtype=torch.float32)),
            )
            setattr(
                self,
                f"{role}_right",
                nn.Parameter(
                    torch.tensor([0.09375, -0.046875, 0.0234375]) * role_index
                ),
            )


class _TinyGuidedGUDTranche:
    def __init__(self, transaction, registry: _TinyGuidedGUDRegistry) -> None:
        self.inherited_handle = transaction.swiglu_handle
        self.guided_registry = registry
        self.training_stage = "guided-left"
        self._stages = ("guided-left", "guided", "joint")
        self._source_binding = MappingProxyType(
            {
                "audit_report_sha256": "a" * 64,
                "basis_envelope_file_sha256": "b" * 64,
                "basis_payload_sha256": "c" * 64,
                "basis_manifest_sha256": "d" * 64,
                "source_basis_tensor_sha256s": MappingProxyType(
                    {
                        role: behavior_distillation._guided_gud_basis_tensor_sha256(
                            getattr(registry, f"{role}_right")
                        )
                        for role in ("gate", "up", "down")
                    }
                ),
                "inherited_installation_sha256": "e" * 64,
                "inherited_factor_state_sha256_at_install": "f" * 64,
                "inherited_handle_identity": id(transaction.swiglu_handle),
                "inherited_registry_identity": id(transaction.swiglu_handle.registry),
                "adapter_site_indices_sha256": "a" * 64,
                "audit_site_indices_sha256": "b" * 64,
            }
        )

    def assert_live(self):
        return None

    def assert_structurally_live(self):
        return None

    def factor_source_binding(self):
        return self._source_binding

    def set_training_stage(self, stage):
        if self._stages.index(stage) < self._stages.index(self.training_stage):
            raise RuntimeError("guided-GUD training stages cannot move backward")
        self.training_stage = stage
        inherited_trainable = stage == "joint"
        for value in self.inherited_handle.registry.parameters(recurse=False):
            value.requires_grad_(inherited_trainable)
        right_trainable = stage in {"guided", "joint"}
        for name, value in self.guided_registry.named_parameters(recurse=False):
            value.requires_grad_(name.endswith("_left") or right_trainable)
        return self.output_trainable_parameters()

    def output_trainable_parameters(self):
        inherited = tuple(self.inherited_handle.registry.parameters(recurse=False))
        guided = tuple(self.guided_registry.parameters(recurse=False))
        if self.training_stage == "guided-left":
            return tuple(
                value
                for name, value in self.guided_registry.named_parameters(recurse=False)
                if name.endswith("_left")
            )
        if self.training_stage == "guided":
            return guided
        return (*inherited, *guided)


def _tiny_guided_v3_case(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(behavior_distillation, "GUIDED_QKVO_PARAMETER_COUNT", 24)
    monkeypatch.setattr(behavior_distillation, "GUIDED_CONTINUATION_STEPS", 1)
    monkeypatch.setattr(behavior_distillation, "GUIDED_LEFT_ONLY_END_STEP", 0)
    monkeypatch.setattr(behavior_distillation, "GUIDED_ONLY_END_STEP", 0)
    monkeypatch.setattr(behavior_distillation, "GUIDED_JOINT_START_STEP", 1)

    torch.manual_seed(176)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 2, dims=0))
    transaction = _transaction(child, lexical_rank=2)
    registry = _TinyGuidedRegistry()
    child.add_module("_tiny_guided_registry", registry)
    tranche = _TinyGuidedTranche(transaction, registry)
    source_v2_state = capture_target_native_factor_state(transaction)
    binding = behavior_distillation.authenticate_guided_factor_sources(
        transaction,
        tranche,
        source_v2_state,
    )
    zero_state = capture_target_native_factor_state(
        transaction,
        guided_tranche=tranche,
        guided_source_binding=binding,
    )
    recovered_tensors = tuple(
        tensor + torch.full_like(tensor, (index + 1) * 1e-4)
        for index, tensor in enumerate(zero_state.tensors)
    )
    recovered_hashes = tuple(
        behavior_distillation._tensor_sha256(tensor) for tensor in recovered_tensors
    )
    recovered_rows = behavior_distillation._factor_manifest_rows(
        zero_state.names,
        recovered_tensors,
        recovered_hashes,
    )
    recovered_state = replace(
        zero_state,
        tensors=recovered_tensors,
        tensor_sha256s=recovered_hashes,
        factor_state_sha256=behavior_distillation._factor_state_sha256(
            recovered_rows,
            schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            lexical_rank=zero_state.lexical_rank,
            guided_source_binding=binding,
        ),
    )

    def add_guided_product(_module, _inputs, output):
        correction = sum(
            (
                getattr(registry, f"{role}_left") * getattr(registry, f"{role}_right")
            ).sum()
            for role in ("q", "k", "v", "o")
        )
        channel = torch.linspace(
            -1.0,
            1.0,
            output.shape[-1],
            device=output.device,
            dtype=output.dtype,
        )
        return output + correction.to(dtype=output.dtype) * channel

    hook = child.register_forward_hook(add_guided_product)
    domains = {
        "fit": _domain("fit", [[1, 2, 3]], "a" * 64),
        "calibration": _domain("calibration", [[4, 5, 6]], "b" * 64),
        "sealed": _domain("sealed", [[7, 8, 9]], "c" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=1,
        learning_rate=0.01,
        fit_horizon=1,
        evaluation_horizon=1,
        top_k=2,
        guided_continuation_enabled=True,
    )
    return (
        child,
        parent,
        transaction,
        tranche,
        hook,
        domains,
        schedule,
        zero_state,
        recovered_state,
    )


def _tiny_guided_v4_case(monkeypatch: pytest.MonkeyPatch):
    values = _tiny_guided_v3_case(monkeypatch)
    (
        child,
        parent,
        transaction,
        guided_tranche,
        qkvo_hook,
        domains,
        schedule,
        zero_v3_state,
        recovered_v3_state,
    ) = values
    monkeypatch.setattr(behavior_distillation, "GUIDED_CONTINUATION_STEPS", 3)
    monkeypatch.setattr(behavior_distillation, "GUIDED_LEFT_ONLY_END_STEP", 1)
    monkeypatch.setattr(behavior_distillation, "GUIDED_ONLY_END_STEP", 2)
    monkeypatch.setattr(behavior_distillation, "GUIDED_JOINT_START_STEP", 3)
    gud_registry = _TinyGuidedGUDRegistry()
    child.add_module("_tiny_guided_gud_registry", gud_registry)
    gud_tranche = _TinyGuidedGUDTranche(transaction, gud_registry)

    def add_guided_gud_product(_module, _inputs, output):
        correction = sum(
            (
                getattr(gud_registry, f"{role}_left")
                * getattr(gud_registry, f"{role}_right")
            ).sum()
            for role in ("gate", "up", "down")
        )
        channel = torch.linspace(
            1.0,
            -1.0,
            output.shape[-1],
            device=output.device,
            dtype=output.dtype,
        )
        return output + correction.to(dtype=output.dtype) * channel

    gud_hook = child.register_forward_hook(add_guided_gud_product)
    return (
        child,
        parent,
        transaction,
        guided_tranche,
        gud_tranche,
        qkvo_hook,
        gud_hook,
        domains,
        replace(schedule, steps=3, checkpoint_steps=(1, 2)),
        zero_v3_state,
        recovered_v3_state,
    )


def _real_guided_v4_case(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(behavior_distillation, "GUIDED_CONTINUATION_STEPS", 3)
    monkeypatch.setattr(behavior_distillation, "GUIDED_LEFT_ONLY_END_STEP", 1)
    monkeypatch.setattr(behavior_distillation, "GUIDED_ONLY_END_STEP", 2)
    monkeypatch.setattr(behavior_distillation, "GUIDED_JOINT_START_STEP", 3)
    torch.manual_seed(1051)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 2, dims=0))
    transaction = _real_guided_v4_transaction(child)
    source_v2 = capture_target_native_factor_state(transaction)
    qkvo_tranche = _install_real_guided_qkvo(transaction, monkeypatch)
    qkvo_binding = behavior_distillation.authenticate_guided_factor_sources(
        transaction,
        qkvo_tranche,
        source_v2,
    )
    qkvo_tranche.set_training_stage("joint")
    entries = behavior_distillation._factor_parameters(
        transaction,
        guided_tranche=qkvo_tranche,
    )
    with torch.no_grad():
        for index, (_name, value) in enumerate(entries, start=1):
            value.add_(torch.full_like(value, index * 1e-5))
    recovered_v3 = capture_target_native_factor_state(
        transaction,
        guided_tranche=qkvo_tranche,
        guided_source_binding=qkvo_binding,
    )
    payload = recovered_v3.as_payload()
    restored = restore_target_native_factor_payload(
        transaction,
        payload,
        guided_tranche=qkvo_tranche,
        expected_guided_source_binding=qkvo_binding,
    )
    assert restored.factor_state_sha256 == recovered_v3.factor_state_sha256
    gud_tranche = _install_real_guided_gud(transaction)
    domains = {
        "fit": _domain("fit", [[1, 2, 3]], "a" * 64),
        "calibration": _domain("calibration", [[4, 5, 6]], "b" * 64),
        "sealed": _domain("sealed", [[7, 8, 9]], "c" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=3,
        learning_rate=0.01,
        checkpoint_steps=(1, 2),
        fit_horizon=1,
        evaluation_horizon=1,
        top_k=2,
        guided_continuation_enabled=True,
    )
    return (
        child,
        parent,
        transaction,
        qkvo_tranche,
        gud_tranche,
        domains,
        schedule,
        payload,
        recovered_v3,
    )


def _selection_metrics(
    *,
    exact_rollouts: int,
    action_agreements: int,
    token_matches: int,
    lcp_tokens: int,
    mean_logprob: float,
) -> CausalBehaviorMetrics:
    return CausalBehaviorMetrics(
        split="calibration-selection",
        action_count=10,
        action_agreement_count=action_agreements,
        action_accuracy=action_agreements / 10,
        parent_action_mean_logprob=mean_logprob,
        parent_action_topk_fraction=0.4 if mean_logprob > -5.0 else 0.3,
        parent_action_mean_rank=4.0 if mean_logprob > -5.0 else 5.0,
        parent_action_worst_rank=8 if mean_logprob > -5.0 else 9,
        rollout_count=2,
        exact_rollout_count=exact_rollouts,
        exact_rollout_fraction=exact_rollouts / 2,
        rollout_token_count=8,
        token_match_count=token_matches,
        token_match_fraction=token_matches / 8,
        lcp_token_count=lcp_tokens,
        lcp_fraction=lcp_tokens / 8,
        lcp_at_least_4_count=int(lcp_tokens >= 4),
        repeat_fraction_excess=0.0 if mean_logprob > -5.0 else 0.25,
        adjacent_repeat_fraction_excess=0.0 if mean_logprob > -5.0 else 0.25,
        longest_run_excess=0 if mean_logprob > -5.0 else 1,
        report_sha256=("a" if mean_logprob > -5.0 else "b") * 64,
    )


def test_loader_derives_prompt_identity_and_rejects_false_binding() -> None:
    suite = _domain("fit", [[1, 2, 3], [4, 5]], "a" * 64)
    assert suite.prompts[0].prompt_id == (f"prompt-{suite.prompts[0].token_ids_sha256}")
    assert suite.as_dict()["raw_prompt_text_retained"] is False
    assert _domain("candidate", [[6, 7]], "b" * 64).split_name == "candidate"
    with pytest.raises(TargetNativeBehaviorDistillationError, match="does not match"):
        load_behavior_domain_suite(
            "fit",
            {"prompts": [{"token_ids": [1, 2], "prompt_id": "invented"}]},
            artifact_sha256="a" * 64,
        )


def test_rollout_first_policy_acquires_behavior_without_strict_dominance() -> None:
    baseline = BehaviorCheckpoint(
        step=0,
        metrics=_selection_metrics(
            exact_rollouts=0,
            action_agreements=2,
            token_matches=2,
            lcp_tokens=1,
            mean_logprob=-1.0,
        ),
        factor_state_sha256="1" * 64,
        pareto_safe_vs_step0=True,
    )
    acquired = BehaviorCheckpoint(
        step=1,
        metrics=_selection_metrics(
            exact_rollouts=1,
            action_agreements=3,
            token_matches=3,
            lcp_tokens=2,
            mean_logprob=-10.0,
        ),
        factor_state_sha256="2" * 64,
        pareto_safe_vs_step0=False,
    )
    assert (
        select_behavior_checkpoint(
            (baseline, acquired), selection_policy="strict-nonregression"
        ).step
        == 0
    )
    assert (
        select_behavior_checkpoint(
            (baseline, acquired), selection_policy="rollout-first-acquisition"
        ).step
        == 0
    )
    safe_acquired = replace(
        acquired,
        metrics=_selection_metrics(
            exact_rollouts=1,
            action_agreements=3,
            token_matches=3,
            lcp_tokens=2,
            mean_logprob=-2.0,
        ),
    )
    assert (
        select_behavior_checkpoint(
            (baseline, safe_acquired), selection_policy="rollout-first-acquisition"
        ).step
        == 1
    )
    schedule = BehaviorDistillationSchedule(
        steps=1,
        learning_rate=1e-3,
        selection_policy="rollout-first-acquisition",
    )
    assert schedule.as_dict()["selection_policy"] == "rollout-first-acquisition"

    first_action_only = BehaviorCheckpoint(
        step=1,
        metrics=_selection_metrics(
            exact_rollouts=0,
            action_agreements=4,
            token_matches=2,
            lcp_tokens=1,
            mean_logprob=-0.5,
        ),
        factor_state_sha256="3" * 64,
        pareto_safe_vs_step0=False,
    )
    sequential_progress = BehaviorCheckpoint(
        step=2,
        metrics=_selection_metrics(
            exact_rollouts=0,
            action_agreements=3,
            token_matches=4,
            lcp_tokens=3,
            mean_logprob=-2.0,
        ),
        factor_state_sha256="4" * 64,
        pareto_safe_vs_step0=False,
    )
    assert (
        select_behavior_checkpoint(
            (baseline, first_action_only, sequential_progress),
            selection_policy="rollout-first-acquisition",
        ).step
        == 2
    )

    repetitive_isolated_matches = BehaviorCheckpoint(
        step=3,
        metrics=_selection_metrics(
            exact_rollouts=0,
            action_agreements=8,
            token_matches=7,
            lcp_tokens=1,
            mean_logprob=-10.0,
        ),
        factor_state_sha256="5" * 64,
        pareto_safe_vs_step0=False,
    )
    assert (
        select_behavior_checkpoint(
            (baseline, repetitive_isolated_matches),
            selection_policy="rollout-first-acquisition",
        ).step
        == 0
    )


def test_divergence_curriculum_and_staged_factor_rates_are_exact() -> None:
    schedule = BehaviorDistillationSchedule(
        steps=32,
        learning_rate=3e-4,
        fit_horizon=8,
        parent_only_steps=8,
        warmup_fit_horizon=4,
        solved_prefix_loss_multiplier=0.5,
        divergence_loss_multiplier=4.0,
        post_divergence_loss_multiplier=2.0,
        post_divergence_positions=1,
        child_lane_starts_at_divergence=True,
        max_child_lane_prefixes=3,
        child_repeat_stop_run_length=3,
        qk_learning_rate_scale=0.5,
        vo_learning_rate_scale=1.0,
        gud_learning_rate_scale=0.5,
        lexical_embedding_learning_rate_scale=0.25,
        lexical_head_learning_rate_scale=1.5,
        right_factor_learning_rate_scale=0.25,
        lexical_right_factor_learning_rate_scale=0.2,
        learning_rate_decay_step_1=8,
        learning_rate_decay_scale_1=2.0 / 3.0,
        learning_rate_decay_step_2=20,
        learning_rate_decay_scale_2=0.5,
        qkvo_frozen_steps=8,
        gud_frozen_steps=20,
        lexical_embedding_frozen_steps=20,
        lexical_head_frozen_steps=0,
    )
    parent = (1, 2, 3, 4, 5, 6, 7, 8)
    child = (1, 9, 9, 9, 9, 9, 9, 9)
    divergence = behavior_distillation._first_divergence(parent, child)
    assert divergence == 1
    assert behavior_distillation._divergence_position_weights(
        len(parent), divergence, schedule
    )[:4] == pytest.approx((0.5, 4.0, 2.0, 1.0))
    assert behavior_distillation._child_lane_indices(child, divergence, schedule) == (
        1,
        2,
        3,
    )
    assert behavior_distillation._factor_learning_rate(
        "qkvo.q_left", schedule, step=1
    ) == pytest.approx(1.5e-4)
    assert behavior_distillation._factor_learning_rate(
        "qkvo.v_right", schedule, step=9
    ) == pytest.approx(5e-5)
    assert behavior_distillation._factor_learning_rate(
        "gud.gate_left", schedule, step=21
    ) == pytest.approx(5e-5)
    assert behavior_distillation._factor_learning_rate(
        "lexical.embedding_right", schedule, step=1
    ) == pytest.approx(1.5e-5)
    assert behavior_distillation._factor_learning_rate(
        "lexical.lm_head_left", schedule, step=1
    ) == pytest.approx(4.5e-4)
    assert (
        behavior_distillation._factor_frozen("lexical.lm_head_left", schedule, step=1)
        is False
    )
    assert behavior_distillation._factor_frozen("qkvo.q_left", schedule, step=8) is True
    assert (
        behavior_distillation._factor_frozen("qkvo.q_left", schedule, step=9) is False
    )
    assert (
        behavior_distillation._factor_frozen("gud.gate_left", schedule, step=20) is True
    )
    assert (
        behavior_distillation._factor_frozen(
            "lexical.embedding_left", schedule, step=21
        )
        is False
    )
    three_decay = BehaviorDistillationSchedule(
        steps=48,
        learning_rate=1e-4,
        learning_rate_decay_step_1=8,
        learning_rate_decay_scale_1=0.75,
        learning_rate_decay_step_2=24,
        learning_rate_decay_scale_2=2.0 / 3.0,
        learning_rate_decay_step_3=40,
        learning_rate_decay_scale_3=0.5,
    )
    assert behavior_distillation._factor_learning_rate(
        "lexical.lm_head_left", three_decay, step=1
    ) == pytest.approx(1e-4)
    assert behavior_distillation._factor_learning_rate(
        "lexical.lm_head_left", three_decay, step=9
    ) == pytest.approx(7.5e-5)
    assert behavior_distillation._factor_learning_rate(
        "lexical.lm_head_left", three_decay, step=25
    ) == pytest.approx(5e-5)
    assert behavior_distillation._factor_learning_rate(
        "lexical.lm_head_left", three_decay, step=41
    ) == pytest.approx(2.5e-5)
    assert three_decay.as_dict()["learning_rate_decay_step_3"] == 40
    legacy_schedule = BehaviorDistillationSchedule(
        steps=2,
        learning_rate=1e-3,
        fit_horizon=8,
    )
    assert behavior_distillation._child_lane_indices(
        child, divergence, legacy_schedule
    ) == tuple(range(8))


def test_legacy_schedule_canonical_hash_is_unchanged_by_guided_extension() -> None:
    schedule = BehaviorDistillationSchedule(steps=2, learning_rate=1e-3)
    assert not any(name.startswith("guided") for name in schedule.as_dict())
    assert "sealed_evaluation_policy" not in schedule.as_dict()
    assert behavior_distillation._canonical_sha256(schedule.as_dict()) == (
        "da96ebef088cfab8813aaba8571f0afe4c48f8e472494a78c6b7874612f42105"
    )
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="guided_continuation_enabled",
    ):
        BehaviorDistillationSchedule(
            steps=64,
            learning_rate=1e-3,
            guided_lexical_head_unlock_step=41,
        )


def test_guided_schedule_has_exact_three_phase_trainability_counts() -> None:
    schedule = BehaviorDistillationSchedule(
        steps=64,
        learning_rate=4e-4,
        guided_continuation_enabled=True,
        guided_qk_learning_rate_scale=1.0,
        guided_vo_learning_rate_scale=1.25,
        guided_right_factor_learning_rate_scale=0.5,
        guided_joint_inherited_learning_rate_scale=0.2,
        guided_lexical_head_unlock_step=41,
    )
    names = tuple(
        sorted(
            behavior_distillation._QKVO_GUD_FACTOR_NAMES
            | behavior_distillation._LEXICAL_FACTOR_NAMES
            | behavior_distillation._GUIDED_QKVO_FACTOR_NAMES
        )
    )
    entries = tuple((name, nn.Parameter(torch.zeros(()))) for name in names)

    assert len(entries) == behavior_distillation.GUIDED_COMBINED_FACTOR_TENSOR_COUNT
    assert behavior_distillation._guided_training_stage(1) == "guided-left"
    assert behavior_distillation._guided_training_stage(9) == "guided"
    assert behavior_distillation._guided_training_stage(25) == "joint"
    authorized = {
        step: behavior_distillation._authorized_factor_names(
            entries,
            schedule,
            step=step,
            guided_continuation=True,
        )
        for step in (1, 8, 9, 24, 25, 40, 41, 64)
    }
    assert [len(authorized[step]) for step in (1, 8)] == [4, 4]
    assert [len(authorized[step]) for step in (9, 24)] == [8, 8]
    assert [len(authorized[step]) for step in (25, 40)] == [16, 16]
    assert [len(authorized[step]) for step in (41, 64)] == [18, 18]
    assert all(
        name.startswith("guided_qkvo.") and name.endswith("_left")
        for name in authorized[1]
    )
    assert not any(
        name.startswith(("gud.", "lexical.embedding_")) for name in authorized[64]
    )
    assert behavior_distillation._factor_learning_rate(
        "qkvo.q_left",
        schedule,
        step=25,
        guided_continuation=True,
    ) == pytest.approx(8e-5)
    assert behavior_distillation._factor_learning_rate(
        "guided_qkvo.v_right",
        schedule,
        step=9,
        guided_continuation=True,
    ) == pytest.approx(2.5e-4)


def test_guided_gud_v4_schedule_has_exact_3_6_22_authorized_tensors() -> None:
    schedule = BehaviorDistillationSchedule(
        steps=64,
        learning_rate=4e-4,
        guided_continuation_enabled=True,
    )
    names = tuple(
        sorted(
            behavior_distillation._QKVO_GUD_FACTOR_NAMES
            | behavior_distillation._LEXICAL_FACTOR_NAMES
            | behavior_distillation._GUIDED_QKVO_FACTOR_NAMES
            | behavior_distillation._GUIDED_GUD_FACTOR_NAMES
        )
    )
    entries = tuple((name, nn.Parameter(torch.zeros(()))) for name in names)
    authorized = {
        step: behavior_distillation._authorized_factor_names(
            entries,
            schedule,
            step=step,
            guided_continuation=True,
            guided_gud_continuation=True,
        )
        for step in (1, 8, 9, 24, 25, 64)
    }
    assert len(entries) == behavior_distillation.GUIDED_GUD_COMBINED_FACTOR_TENSOR_COUNT
    assert [len(authorized[step]) for step in (1, 8)] == [3, 3]
    assert [len(authorized[step]) for step in (9, 24)] == [6, 6]
    assert [len(authorized[step]) for step in (25, 64)] == [22, 22]
    assert all(
        name.startswith("guided_gud.") and name.endswith("_left")
        for name in authorized[1]
    )
    assert set(authorized[25]) == (
        behavior_distillation._GUIDED_GUD_FACTOR_NAMES
        | behavior_distillation._GUIDED_QKVO_FACTOR_NAMES
        | {
            name
            for name in behavior_distillation._QKVO_GUD_FACTOR_NAMES
            if name.startswith("qkvo.")
        }
    )
    assert not any(name.startswith(("gud.", "lexical.")) for name in authorized[64])


def test_guided_method_gate_requires_one_contiguous_lcp4_and_repeat_safety() -> None:
    baseline = _selection_metrics(
        exact_rollouts=0,
        action_agreements=2,
        token_matches=2,
        lcp_tokens=1,
        mean_logprob=-1.0,
    )
    four_isolated_one_token_matches = replace(
        baseline,
        lcp_token_count=4,
        lcp_fraction=0.5,
        lcp_at_least_4_count=0,
    )
    isolated_gate = behavior_distillation._guided_candidate_gate(
        selected_step=24,
        step0_metrics=baseline,
        candidate_metrics=four_isolated_one_token_matches,
    )
    assert isolated_gate.observed_lcp_at_least_4_count == 0
    assert isolated_gate.passed is False

    contiguous_but_repetitive = replace(
        four_isolated_one_token_matches,
        lcp_at_least_4_count=1,
        repeat_fraction_excess=baseline.repeat_fraction_excess + 0.01,
    )
    repeat_gate = behavior_distillation._guided_candidate_gate(
        selected_step=24,
        step0_metrics=baseline,
        candidate_metrics=contiguous_but_repetitive,
    )
    assert repeat_gate.repeat_fraction_nonregression is False
    assert repeat_gate.passed is False

    safe = replace(
        contiguous_but_repetitive,
        repeat_fraction_excess=baseline.repeat_fraction_excess,
        adjacent_repeat_fraction_excess=(baseline.adjacent_repeat_fraction_excess),
    )
    assert (
        behavior_distillation._guided_candidate_gate(
            selected_step=24,
            step0_metrics=baseline,
            candidate_metrics=safe,
        ).passed
        is True
    )
    assert (
        behavior_distillation._guided_candidate_gate(
            selected_step=0,
            step0_metrics=baseline,
            candidate_metrics=safe,
        ).passed
        is False
    )


def test_v3_pack_authenticates_26_tensors_and_rejects_source_tamper() -> None:
    inherited_names = tuple(
        sorted(
            behavior_distillation._QKVO_GUD_FACTOR_NAMES
            | behavior_distillation._LEXICAL_FACTOR_NAMES
        )
    )
    guided_names = tuple(sorted(behavior_distillation._GUIDED_QKVO_FACTOR_NAMES))
    names = tuple(sorted((*inherited_names, *guided_names)))
    entries = tuple(
        (
            name,
            nn.Parameter(torch.full((2,), float(index + 1), dtype=torch.float32)),
        )
        for index, name in enumerate(names)
    )
    source_v2_hashes = {
        name: behavior_distillation._tensor_sha256(value)
        for name, value in entries
        if name in inherited_names
    }
    basis_hashes = {
        role: behavior_distillation._tensor_sha256(
            dict(entries)[f"guided_qkvo.{role}_right"]
        )
        for role in ("q", "k", "v", "o")
    }
    binding = behavior_distillation.GuidedFactorSourceBinding(
        source_v2_factor_state_sha256="1" * 64,
        source_v2_tensor_sha256s=tuple(source_v2_hashes.items()),
        audit_report_sha256="2" * 64,
        basis_envelope_file_sha256="3" * 64,
        basis_payload_sha256="4" * 64,
        basis_manifest_sha256="5" * 64,
        source_basis_tensor_sha256s=tuple(basis_hashes.items()),
        inherited_installation_sha256="6" * 64,
        inherited_factor_state_sha256_at_install="7" * 64,
        adapter_site_indices_sha256="8" * 64,
        audit_site_indices_sha256="a" * 64,
    )
    tensors = tuple(value.detach().cpu().clone() for _name, value in entries)
    hashes = tuple(behavior_distillation._tensor_sha256(value) for value in tensors)
    rows = behavior_distillation._factor_manifest_rows(names, tensors, hashes)
    state = behavior_distillation.TargetNativeFactorState(
        names=names,
        tensors=tensors,
        tensor_sha256s=hashes,
        factor_state_sha256=behavior_distillation._factor_state_sha256(
            rows,
            schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            lexical_rank=3,
            guided_source_binding=binding,
        ),
        parameter_ids=tuple(id(value) for _name, value in entries),
        parameter_data_ptrs=tuple(int(value.data_ptr()) for _name, value in entries),
        schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        lexical_rank=3,
        guided_source_binding=binding,
    )
    payload = state.as_payload()
    assert payload["factor_count"] == 26
    assert payload["inherited_product_regrouped"] is False
    assert payload["guided_product_regrouped"] is False
    authenticated = behavior_distillation._authenticate_factor_payload_state(
        payload,
        entries,
        schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        lexical_rank=3,
        expected_guided_source_binding=binding,
    )
    assert authenticated.factor_state_sha256 == state.factor_state_sha256
    durable_event = behavior_distillation._durable_behavior_checkpoint(
        step=7,
        metrics=_selection_metrics(
            exact_rollouts=1,
            action_agreements=2,
            token_matches=3,
            lcp_tokens=3,
            mean_logprob=-4.0,
        ),
        state=state,
        pareto_safe_vs_step0=True,
    )
    durable_payload = durable_event.factor_state.as_factor_payload()
    durable_authenticated = behavior_distillation._authenticate_factor_payload_state(
        durable_payload,
        entries,
        schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        lexical_rank=3,
        expected_guided_source_binding=binding,
    )
    assert durable_authenticated.factor_state_sha256 == state.factor_state_sha256
    assert durable_event.factor_state.guided_source_binding == binding

    tampered = copy.deepcopy(payload)
    tampered_binding_payload = dict(tampered["guided_source_binding"])
    tampered_source_hashes = dict(tampered_binding_payload["source_v2_tensor_sha256s"])
    tampered_source_hashes[inherited_names[0]] = "9" * 64
    tampered_binding_payload["source_v2_tensor_sha256s"] = tampered_source_hashes
    tampered_binding = behavior_distillation.GuidedFactorSourceBinding(
        **tampered_binding_payload
    )
    tampered["guided_source_binding"] = tampered_binding.as_dict()
    tampered["guided_source_binding_sha256"] = tampered_binding.binding_sha256
    tampered["factor_state_sha256"] = behavior_distillation._factor_state_sha256(
        tampered["factors"],
        schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        lexical_rank=3,
        guided_source_binding=tampered_binding,
    )
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="preregistered sources",
    ):
        behavior_distillation._authenticate_factor_payload_state(
            tampered,
            entries,
            schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            lexical_rank=3,
            expected_guided_source_binding=binding,
        )

    tensor_tamper = copy.deepcopy(payload)
    tensor_tamper["factor_tensors"][names[0]][0] += 1.0
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="shape/dtype/hash authentication",
    ):
        behavior_distillation._authenticate_factor_payload_state(
            tensor_tamper,
            entries,
            schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            lexical_rank=3,
            expected_guided_source_binding=binding,
        )


def test_v4_pack_authenticates_32_tensors_durable_bytes_and_nested_sources() -> None:
    v3_names = tuple(
        sorted(
            behavior_distillation._QKVO_GUD_FACTOR_NAMES
            | behavior_distillation._LEXICAL_FACTOR_NAMES
            | behavior_distillation._GUIDED_QKVO_FACTOR_NAMES
        )
    )
    names = tuple(sorted((*v3_names, *behavior_distillation._GUIDED_GUD_FACTOR_NAMES)))
    entries = tuple(
        (
            name,
            nn.Parameter(torch.full((2,), (index + 1) / 32, dtype=torch.float32)),
        )
        for index, name in enumerate(names)
    )
    entry_by_name = dict(entries)
    qkvo_binding = behavior_distillation.GuidedFactorSourceBinding(
        source_v2_factor_state_sha256="1" * 64,
        source_v2_tensor_sha256s=tuple(
            (name, behavior_distillation._tensor_sha256(entry_by_name[name]))
            for name in sorted(
                behavior_distillation._QKVO_GUD_FACTOR_NAMES
                | behavior_distillation._LEXICAL_FACTOR_NAMES
            )
        ),
        audit_report_sha256="2" * 64,
        basis_envelope_file_sha256="3" * 64,
        basis_payload_sha256="4" * 64,
        basis_manifest_sha256="5" * 64,
        source_basis_tensor_sha256s=tuple(
            (
                role,
                behavior_distillation._tensor_sha256(
                    entry_by_name[f"guided_qkvo.{role}_right"]
                ),
            )
            for role in ("q", "k", "v", "o")
        ),
        inherited_installation_sha256="6" * 64,
        inherited_factor_state_sha256_at_install="7" * 64,
        adapter_site_indices_sha256="8" * 64,
        audit_site_indices_sha256="9" * 64,
    )
    v3_tensors = tuple(entry_by_name[name].detach().clone() for name in v3_names)
    v3_hashes = tuple(
        behavior_distillation._tensor_sha256(value) for value in v3_tensors
    )
    v3_rows = behavior_distillation._factor_manifest_rows(
        v3_names, v3_tensors, v3_hashes
    )
    v3_sha = behavior_distillation._factor_state_sha256(
        v3_rows,
        schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        lexical_rank=3,
        guided_source_binding=qkvo_binding,
    )
    gud_binding = behavior_distillation.GuidedGUDFactorSourceBinding(
        source_v3_factor_state_sha256=v3_sha,
        source_v3_factor_manifest_sha256=behavior_distillation._canonical_sha256(
            v3_rows
        ),
        source_v3_tensor_sha256s=tuple(zip(v3_names, v3_hashes, strict=True)),
        source_v3_lexical_rank=3,
        source_guided_qkvo_binding=qkvo_binding,
        audit_report_sha256="a" * 64,
        basis_envelope_file_sha256="b" * 64,
        basis_payload_sha256="c" * 64,
        basis_manifest_sha256="d" * 64,
        source_basis_tensor_sha256s=tuple(
            (
                role,
                behavior_distillation._guided_gud_basis_tensor_sha256(
                    entry_by_name[f"guided_gud.{role}_right"]
                ),
            )
            for role in ("gate", "up", "down")
        ),
        inherited_installation_sha256="e" * 64,
        inherited_factor_state_sha256_at_install="f" * 64,
        adapter_site_indices_sha256="a" * 64,
        audit_site_indices_sha256="b" * 64,
    )
    tensors = tuple(value.detach().clone() for _name, value in entries)
    hashes = tuple(behavior_distillation._tensor_sha256(value) for value in tensors)
    rows = behavior_distillation._factor_manifest_rows(names, tensors, hashes)
    state = behavior_distillation.TargetNativeFactorState(
        names=names,
        tensors=tensors,
        tensor_sha256s=hashes,
        factor_state_sha256=behavior_distillation._factor_state_sha256(
            rows,
            schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
            lexical_rank=3,
            guided_source_binding=qkvo_binding,
            guided_gud_source_binding=gud_binding,
        ),
        parameter_ids=tuple(id(value) for _name, value in entries),
        parameter_data_ptrs=tuple(int(value.data_ptr()) for _name, value in entries),
        schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
        lexical_rank=3,
        guided_source_binding=qkvo_binding,
        guided_gud_source_binding=gud_binding,
    )
    payload = state.as_payload()
    assert payload["factor_count"] == 32
    assert payload["guided_gud_product_regrouped"] is False
    authenticated = behavior_distillation._authenticate_factor_payload_state(
        payload,
        entries,
        schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
        lexical_rank=3,
        expected_guided_source_binding=qkvo_binding,
        expected_guided_gud_source_binding=gud_binding,
    )
    assert authenticated.factor_state_sha256 == state.factor_state_sha256
    durable = behavior_distillation._durable_behavior_checkpoint(
        step=9,
        metrics=_selection_metrics(
            exact_rollouts=0,
            action_agreements=2,
            token_matches=3,
            lcp_tokens=2,
            mean_logprob=-4.0,
        ),
        state=state,
        pareto_safe_vs_step0=True,
    )
    assert durable.factor_state.schema == (
        behavior_distillation.BEHAVIOR_FACTOR_PACK_V4_SCHEMA
    )
    assert len(durable.factor_state.tensors) == 32
    assert durable.factor_state.as_factor_payload()["factor_state_sha256"] == (
        state.factor_state_sha256
    )

    tampered = copy.deepcopy(payload)
    nested = dict(tampered["guided_gud_source_binding"])
    nested["source_v3_content_sha256"] = "0" * 64
    tampered["guided_gud_source_binding"] = nested
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="source binding",
    ):
        behavior_distillation._authenticate_factor_payload_state(
            tampered,
            entries,
            schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
            lexical_rank=3,
            expected_guided_source_binding=qkvo_binding,
            expected_guided_gud_source_binding=gud_binding,
        )


def test_guided_v3_pack_resumes_exact_authenticated_step_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        child,
        parent,
        transaction,
        tranche,
        hook,
        domains,
        schedule,
        _zero_state,
        recovered_state,
    ) = _tiny_guided_v3_case(monkeypatch)
    assert isinstance(tranche.factor_source_binding(), MappingProxyType)
    assert isinstance(
        tranche.factor_source_binding()["source_basis_tensor_sha256s"],
        MappingProxyType,
    )
    durable = []
    result = train_target_native_behavior_distillation(
        transaction,
        parent,
        domains,
        schedule,
        initial_factor_payload=recovered_state.as_payload(),
        guided_tranche=tranche,
        durable_checkpoint_callback=lambda event: durable.append(event),
    )
    assert result.initial_factor_state_sha256 == recovered_state.factor_state_sha256
    assert (
        result.initial_factor_payload_schema
        == behavior_distillation.BEHAVIOR_FACTOR_PACK_V3_SCHEMA
    )
    assert result.initial_factor_restore_mode == "guided-v3-full"
    assert durable[0].step == 0
    assert (
        durable[0].factor_state.factor_state_sha256
        == recovered_state.factor_state_sha256
    )
    assert (
        result.checkpoints[0].factor_state_sha256 == recovered_state.factor_state_sha256
    )
    hook.remove()
    delattr(child, "_tiny_guided_registry")
    transaction.rollback()


def test_guided_v3_pack_rebinds_only_cross_process_installation_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        child,
        parent,
        transaction,
        tranche,
        hook,
        domains,
        schedule,
        _zero_state,
        recovered_state,
    ) = _tiny_guided_v3_case(monkeypatch)
    live_binding = recovered_state.guided_source_binding
    assert isinstance(live_binding, behavior_distillation.GuidedFactorSourceBinding)
    historical_binding_payload = live_binding.as_dict()
    historical_binding_payload["inherited_installation_sha256"] = "9" * 64
    historical_binding = behavior_distillation.GuidedFactorSourceBinding(
        **historical_binding_payload
    )
    historical_state = replace(
        recovered_state,
        guided_source_binding=historical_binding,
        factor_state_sha256=behavior_distillation._factor_state_sha256(
            behavior_distillation._factor_manifest_rows(
                recovered_state.names,
                recovered_state.tensors,
                recovered_state.tensor_sha256s,
            ),
            schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            lexical_rank=recovered_state.lexical_rank,
            guided_source_binding=historical_binding,
        ),
    )
    durable = []
    result = train_target_native_behavior_distillation(
        transaction,
        parent,
        domains,
        schedule,
        initial_factor_payload=historical_state.as_payload(),
        guided_tranche=tranche,
        durable_checkpoint_callback=lambda event: durable.append(event),
    )
    assert historical_state.factor_state_sha256 != recovered_state.factor_state_sha256
    assert result.initial_factor_state_sha256 == recovered_state.factor_state_sha256
    assert durable[0].factor_state.factor_state_sha256 == (
        recovered_state.factor_state_sha256
    )
    assert durable[0].factor_state.guided_source_binding == live_binding
    hook.remove()
    delattr(child, "_tiny_guided_registry")
    transaction.rollback()


@pytest.mark.parametrize("tamper", ["binding", "tensor", "foreign-tranche"])
def test_guided_v3_tamper_fails_before_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    (
        child,
        parent,
        transaction,
        tranche,
        hook,
        domains,
        schedule,
        zero_state,
        recovered_state,
    ) = _tiny_guided_v3_case(monkeypatch)
    payload = copy.deepcopy(recovered_state.as_payload())
    if tamper == "binding":
        binding_payload = dict(payload["guided_source_binding"])
        binding_payload["audit_report_sha256"] = "f" * 64
        binding = behavior_distillation.GuidedFactorSourceBinding(**binding_payload)
        payload["guided_source_binding"] = binding.as_dict()
        payload["guided_source_binding_sha256"] = binding.binding_sha256
        payload["factor_state_sha256"] = behavior_distillation._factor_state_sha256(
            payload["factors"],
            schema=behavior_distillation.BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            lexical_rank=recovered_state.lexical_rank,
            guided_source_binding=binding,
        )
    elif tamper == "tensor":
        payload["factor_tensors"][recovered_state.names[0]].reshape(-1)[0] += 1.0
    else:
        tranche.inherited_handle = object()

    optimizer_calls = 0

    def optimizer_factory(parameters):
        nonlocal optimizer_calls
        optimizer_calls += 1
        return torch.optim.AdamW(parameters, lr=schedule.learning_rate)

    with pytest.raises((TargetNativeBehaviorDistillationError, AttributeError)):
        train_target_native_behavior_distillation(
            transaction,
            parent,
            domains,
            schedule,
            initial_factor_payload=payload,
            guided_tranche=tranche,
            optimizer_factory=optimizer_factory,
        )
    assert optimizer_calls == 0
    if tamper != "foreign-tranche":
        assert (
            capture_target_native_factor_state(
                transaction,
                guided_tranche=tranche,
                guided_source_binding=zero_state.guided_source_binding,
            ).factor_state_sha256
            == zero_state.factor_state_sha256
        )
    hook.remove()
    delattr(child, "_tiny_guided_registry")
    transaction.rollback()


def test_guided_v3_training_failure_restores_exact_recovered_step_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        child,
        parent,
        transaction,
        tranche,
        hook,
        domains,
        schedule,
        _zero_state,
        recovered_state,
    ) = _tiny_guided_v3_case(monkeypatch)

    def fail_after_step(step: int) -> None:
        raise RuntimeError(f"stop after {step}")

    with pytest.raises(RuntimeError, match="stop after 1"):
        train_target_native_behavior_distillation(
            transaction,
            parent,
            domains,
            schedule,
            initial_factor_payload=recovered_state.as_payload(),
            guided_tranche=tranche,
            post_optimizer_step_callback=fail_after_step,
        )
    assert (
        capture_target_native_factor_state(
            transaction,
            guided_tranche=tranche,
            guided_source_binding=recovered_state.guided_source_binding,
        ).factor_state_sha256
        == recovered_state.factor_state_sha256
    )
    hook.remove()
    delattr(child, "_tiny_guided_registry")
    transaction.rollback()


def test_guided_gud_v4_trains_3_6_22_and_emits_full_durable_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        child,
        parent,
        transaction,
        guided_tranche,
        gud_tranche,
        qkvo_hook,
        gud_hook,
        domains,
        schedule,
        _zero_v3_state,
        recovered_v3_state,
    ) = _tiny_guided_v4_case(monkeypatch)
    events = []
    result = train_target_native_behavior_distillation(
        transaction,
        parent,
        domains,
        schedule,
        initial_factor_payload=recovered_v3_state.as_payload(),
        guided_tranche=guided_tranche,
        guided_gud_tranche=gud_tranche,
        durable_checkpoint_callback=events.append,
    )
    assert result.initial_factor_payload_schema == (
        behavior_distillation.BEHAVIOR_FACTOR_PACK_V3_SCHEMA
    )
    assert result.initial_factor_restore_mode == "guided-v3-to-v4"
    assert result.factor_pack.schema == (
        behavior_distillation.BEHAVIOR_FACTOR_PACK_V4_SCHEMA
    )
    assert result.factor_pack.as_manifest()["factor_count"] == 32
    assert [event.step for event in events] == [0, 1, 2, 3]
    assert all(
        event.factor_state.schema
        == behavior_distillation.BEHAVIOR_FACTOR_PACK_V4_SCHEMA
        and len(event.factor_state.tensors) == 32
        for event in events
    )
    stage_counts = [
        len(names)
        for _stage, names in result.training_audit.gradient_activation.stage_authorized_factor_names
    ]
    assert stage_counts == [3, 6, 22]
    assert len(result.training_audit.gradient_activation.authorized_factor_names) == 22
    binding = result.guided_gud_source_binding
    assert isinstance(binding, behavior_distillation.GuidedGUDFactorSourceBinding)
    step1 = {
        value.name: value.materialize_tensor()
        for value in events[1].factor_state.tensors
    }
    assert all(
        behavior_distillation._guided_gud_basis_tensor_sha256(
            step1[f"guided_gud.{role}_right"]
        )
        == digest
        for role, digest in binding.source_basis_tensor_sha256s
    )
    step2_hashes = {
        value.name: value.tensor_sha256 for value in events[2].factor_state.tensors
    }
    assert all(
        step2_hashes[name] == digest
        for name, digest in binding.source_v3_tensor_sha256s
    )
    assert result.evidence_payload()["guided_gud_schedule"][
        "authorized_tensor_counts"
    ] == [3, 6, 22]

    qkvo_hook.remove()
    gud_hook.remove()
    delattr(child, "_tiny_guided_gud_registry")
    delattr(child, "_tiny_guided_registry")
    transaction.rollback()


def test_guided_gud_v4_failure_rolls_back_exact_32_factor_step_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        child,
        parent,
        transaction,
        guided_tranche,
        gud_tranche,
        qkvo_hook,
        gud_hook,
        domains,
        schedule,
        _zero_v3_state,
        recovered_v3_state,
    ) = _tiny_guided_v4_case(monkeypatch)
    events = []

    def fail_after_step(step: int) -> None:
        if step == 2:
            raise RuntimeError("stop v4 after step 2")

    with pytest.raises(RuntimeError, match="stop v4 after step 2"):
        train_target_native_behavior_distillation(
            transaction,
            parent,
            domains,
            schedule,
            initial_factor_payload=recovered_v3_state.as_payload(),
            guided_tranche=guided_tranche,
            guided_gud_tranche=gud_tranche,
            durable_checkpoint_callback=events.append,
            post_optimizer_step_callback=fail_after_step,
        )
    assert [event.step for event in events] == [0, 1]
    step0 = events[0].factor_state
    restored = capture_target_native_factor_state(
        transaction,
        guided_tranche=guided_tranche,
        guided_source_binding=step0.guided_source_binding,
        guided_gud_tranche=gud_tranche,
        guided_gud_source_binding=step0.guided_gud_source_binding,
    )
    assert restored.factor_state_sha256 == step0.factor_state_sha256

    qkvo_hook.remove()
    gud_hook.remove()
    delattr(child, "_tiny_guided_gud_registry")
    delattr(child, "_tiny_guided_registry")
    transaction.rollback()


def test_real_guided_adapters_accept_prerestored_v3_handoff_and_train_v4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        child,
        parent,
        transaction,
        qkvo_tranche,
        gud_tranche,
        domains,
        schedule,
        payload,
        recovered_v3,
    ) = _real_guided_v4_case(monkeypatch)
    pretrain_v3_hashes = {
        name: behavior_distillation._tensor_sha256(value)
        for name, value in behavior_distillation._factor_parameters(
            transaction,
            guided_tranche=qkvo_tranche,
        )
    }
    events = []
    result = train_target_native_behavior_distillation(
        transaction,
        parent,
        domains,
        schedule,
        initial_factor_payload=payload,
        guided_tranche=qkvo_tranche,
        guided_gud_tranche=gud_tranche,
        durable_checkpoint_callback=events.append,
    )
    assert result.initial_factor_state_sha256 == recovered_v3.factor_state_sha256
    assert result.initial_factor_restore_mode == "guided-v3-to-v4"
    assert result.combined_step0_factor_state_sha256 == (
        events[0].factor_state.factor_state_sha256
    )
    assert dict(result.guided_gud_source_binding.source_v3_tensor_sha256s) == (
        pretrain_v3_hashes
    )
    assert [
        len(names)
        for _stage, names in result.training_audit.gradient_activation.stage_authorized_factor_names
    ] == [3, 6, 22]
    assert [event.step for event in events] == [0, 1, 2, 3]
    assert all(
        event.factor_state.schema
        == behavior_distillation.BEHAVIOR_FACTOR_PACK_V4_SCHEMA
        and len(event.factor_state.tensors) == 32
        for event in events
    )

    gud_tranche.rollback_guided_only()
    qkvo_tranche.rollback_guided_only()
    transaction.rollback()
    assert exact_unique_parameter_count(child) == exact_unique_parameter_count(parent)


def test_real_guided_prerestored_v4_failure_restores_step_zero_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        child,
        parent,
        transaction,
        qkvo_tranche,
        gud_tranche,
        domains,
        schedule,
        payload,
        _recovered_v3,
    ) = _real_guided_v4_case(monkeypatch)
    events = []

    def fail_after_joint_step(step: int) -> None:
        if step == 3:
            raise RuntimeError("stop real v4 after joint step")

    with pytest.raises(RuntimeError, match="stop real v4 after joint step"):
        train_target_native_behavior_distillation(
            transaction,
            parent,
            domains,
            schedule,
            initial_factor_payload=payload,
            guided_tranche=qkvo_tranche,
            guided_gud_tranche=gud_tranche,
            durable_checkpoint_callback=events.append,
            post_optimizer_step_callback=fail_after_joint_step,
        )
    assert [event.step for event in events] == [0, 1, 2]
    step0 = events[0].factor_state
    restored = capture_target_native_factor_state(
        transaction,
        guided_tranche=qkvo_tranche,
        guided_source_binding=step0.guided_source_binding,
        guided_gud_tranche=gud_tranche,
        guided_gud_source_binding=step0.guided_gud_source_binding,
    )
    assert restored.factor_state_sha256 == step0.factor_state_sha256

    gud_tranche.rollback_guided_only()
    qkvo_tranche.rollback_guided_only()
    transaction.rollback()
    assert exact_unique_parameter_count(child) == exact_unique_parameter_count(parent)


def test_real_guided_prerestored_mismatch_fails_before_any_factor_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        child,
        parent,
        transaction,
        qkvo_tranche,
        gud_tranche,
        domains,
        schedule,
        payload,
        _recovered_v3,
    ) = _real_guided_v4_case(monkeypatch)
    before = {
        name: behavior_distillation._tensor_sha256(value)
        for name, value in behavior_distillation._factor_parameters(
            transaction,
            guided_tranche=qkvo_tranche,
            guided_gud_tranche=gud_tranche,
        )
    }
    tampered = copy.deepcopy(payload)
    first_name = tampered["factors"][0]["name"]
    tampered["factor_tensors"][first_name].reshape(-1)[0] += 1.0
    optimizer_calls = 0

    def optimizer_factory(parameters):
        nonlocal optimizer_calls
        optimizer_calls += 1
        return torch.optim.AdamW(parameters, lr=schedule.learning_rate)

    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="shape/dtype/hash authentication",
    ):
        train_target_native_behavior_distillation(
            transaction,
            parent,
            domains,
            schedule,
            initial_factor_payload=tampered,
            guided_tranche=qkvo_tranche,
            guided_gud_tranche=gud_tranche,
            optimizer_factory=optimizer_factory,
        )
    after = {
        name: behavior_distillation._tensor_sha256(value)
        for name, value in behavior_distillation._factor_parameters(
            transaction,
            guided_tranche=qkvo_tranche,
            guided_gud_tranche=gud_tranche,
        )
    }
    assert optimizer_calls == 0
    assert after == before

    gud_tranche.rollback_guided_only()
    qkvo_tranche.rollback_guided_only()
    transaction.rollback()
    assert exact_unique_parameter_count(child) == exact_unique_parameter_count(parent)


def test_v2_toy_factor_pack_hash_and_restore_evidence_remain_legacy_exact() -> None:
    torch.manual_seed(175)
    child = _StrictToyTarget()
    transaction = _transaction(child, lexical_rank=2)
    historical = capture_target_native_factor_state(transaction)
    assert historical.factor_state_sha256 == (
        "cfda1512ec46a8b43c33821e66c5744a38c61c6387dd31f5e34ef8bf2d7d3530"
    )
    restored = behavior_distillation._restore_target_native_factor_payload(
        transaction,
        historical.as_payload(),
    )
    assert restored.source_factor_state_sha256 == historical.factor_state_sha256
    assert (
        restored.source_schema == behavior_distillation.BEHAVIOR_FACTOR_PACK_V2_SCHEMA
    )
    assert restored.restore_mode == "full"
    transaction.rollback()


def test_guided_source_authentication_binds_live_v2_and_exact_basis_bytes() -> None:
    class _ExactGuidedRegistry(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shapes = {
                "q_left": (120, 512, 8),
                "q_right": (120, 8, 1536),
                "k_left": (120, 128, 8),
                "k_right": (120, 8, 1536),
                "v_left": (120, 128, 8),
                "v_right": (120, 8, 1536),
                "o_left": (120, 1536, 8),
                "o_right": (120, 8, 512),
            }
            for name, shape in shapes.items():
                value = torch.zeros(shape, dtype=torch.float32)
                if name.endswith("_right"):
                    value.reshape(-1)[::257] = 0.125
                setattr(self, name, nn.Parameter(value))

    torch.manual_seed(173)
    child = _StrictToyTarget()
    transaction = _transaction(child, lexical_rank=2)
    source_v2_state = capture_target_native_factor_state(transaction)
    registry = _ExactGuidedRegistry()
    child.add_module("_test_guided_registry", registry)

    source_basis_hashes = {
        role: guided_qkvo._tensor_sha256(getattr(registry, f"{role}_right"))
        for role in ("q", "k", "v", "o")
    }
    assert all(
        source_basis_hashes[role]
        == behavior_distillation._output_jacobian_tensor_sha256(
            getattr(registry, f"{role}_right")
        )
        != behavior_distillation._tensor_sha256(getattr(registry, f"{role}_right"))
        for role in ("q", "k", "v", "o")
    )

    class _GuidedTranche:
        inherited_handle = transaction.qkvo_handle
        guided_registry = registry
        training_stage = "joint"

        @staticmethod
        def assert_live():
            return None

        @staticmethod
        def assert_structurally_live():
            return None

        def set_training_stage(self, stage):
            assert stage == "joint"
            for value in self.inherited_handle.adapter_registry.parameters(
                recurse=False
            ):
                value.requires_grad_(True)
            for value in self.guided_registry.parameters(recurse=False):
                value.requires_grad_(True)

        def output_trainable_parameters(self):
            return (
                *self.inherited_handle.adapter_registry.parameters(recurse=False),
                *self.guided_registry.parameters(recurse=False),
            )

        @staticmethod
        def factor_source_binding():
            return {
                "audit_report_sha256": "1" * 64,
                "basis_envelope_file_sha256": "2" * 64,
                "basis_payload_sha256": "3" * 64,
                "basis_manifest_sha256": "4" * 64,
                "source_basis_tensor_sha256s": source_basis_hashes,
                "inherited_installation_sha256": "5" * 64,
                "inherited_factor_state_sha256_at_install": "6" * 64,
                "inherited_handle_identity": id(transaction.qkvo_handle),
                "inherited_registry_identity": id(
                    transaction.qkvo_handle.adapter_registry
                ),
                "adapter_site_indices_sha256": "7" * 64,
                "audit_site_indices_sha256": "8" * 64,
            }

    tranche = _GuidedTranche()
    entries = behavior_distillation._factor_parameters(
        transaction,
        guided_tranche=tranche,
    )
    embedding_requires_grad = child.embedding.weight.requires_grad
    with behavior_distillation._GuidedCombinedOutputTrainingScope(
        transaction,
        tranche,
        entries,
    ) as trainable:
        assert len(trainable) == 26
        assert {id(value) for value in trainable} == {
            id(value) for _name, value in entries
        }
        assert child.embedding.weight.requires_grad is False
    assert child.embedding.weight.requires_grad is embedding_requires_grad

    binding = behavior_distillation.authenticate_guided_factor_sources(
        transaction,
        tranche,
        source_v2_state,
    )
    assert binding.source_v2_factor_state_sha256 == (
        source_v2_state.factor_state_sha256
    )
    assert len(binding.source_v2_tensor_sha256s) == 18
    assert len(binding.source_basis_tensor_sha256s) == 4
    assert sum(value.numel() for value in registry.parameters()) == (
        behavior_distillation.GUIDED_QKVO_PARAMETER_COUNT
    )

    with torch.no_grad():
        registry.k_right.reshape(-1)[0] += 1.0
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="does not byte-match",
    ):
        behavior_distillation.authenticate_guided_factor_sources(
            transaction,
            tranche,
            source_v2_state,
        )
    delattr(child, "_test_guided_registry")
    transaction.rollback()


def test_durable_checkpoint_callback_emits_every_authenticated_snapshot() -> None:
    torch.manual_seed(120)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 3, dims=0))
    transaction = _transaction(child)
    domains = {
        "fit": _domain("fit", [[1, 2, 3]], "1" * 64),
        "calibration": _domain("calibration", [[4, 5, 6]], "2" * 64),
        "sealed": _domain("sealed", [[7, 8, 9]], "3" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=2,
        learning_rate=0.02,
        checkpoint_steps=(1,),
        fit_horizon=1,
        evaluation_horizon=1,
        top_k=2,
    )
    events: list[behavior_distillation.DurableBehaviorCheckpoint] = []

    def sink(event: behavior_distillation.DurableBehaviorCheckpoint) -> object:
        events.append(event)
        return {"selected_step": 999, "unlock_sealed": False}

    result = train_target_native_behavior_distillation(
        transaction,
        parent,
        domains,
        schedule,
        durable_checkpoint_callback=sink,
    )

    assert [event.step for event in events] == [0, 1, 2]
    assert [value.step for value in result.checkpoints] == [0, 1, 2]
    assert all(event.metrics.action_count > 0 for event in events)
    assert all(event.factor_state.factor_state_sha256 for event in events)
    assert all(event.checkpoint_sha256 for event in events)
    assert all(
        len(event.factor_state.tensors) == 14
        and event.factor_state.as_manifest()["factor_count"] == 14
        for event in events
    )
    assert result.selected_step != 999
    with pytest.raises(TypeError):
        events[0].factor_state.tensors[0].raw_bytes[0] = 0  # type: ignore[index]
    with pytest.raises(AttributeError):
        events[0].step = 99  # type: ignore[misc]

    first_tensor = events[0].factor_state.tensors[0]
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="invalid byte count",
    ):
        replace(first_tensor, raw_bytes=first_tensor.raw_bytes[:-1])
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="Unsupported durable tensor dtype",
    ):
        replace(first_tensor, dtype="torch.not_a_dtype")
    nonfinite = torch.full(first_tensor.shape, float("nan"), dtype=torch.float32)
    nonfinite_bytes = nonfinite.view(torch.uint8).numpy().tobytes()
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="contains NaN or infinity",
    ):
        replace(
            first_tensor,
            dtype=str(nonfinite.dtype),
            raw_bytes=nonfinite_bytes,
            tensor_sha256=behavior_distillation._raw_tensor_sha256(
                dtype=str(nonfinite.dtype),
                shape=tuple(nonfinite.shape),
                raw_bytes=nonfinite_bytes,
            ),
        )
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="names/count",
    ):
        replace(
            events[0].factor_state,
            tensors=events[0].factor_state.tensors[:-1],
        )

    step0_payload = events[0].factor_state.as_factor_payload()
    restored = restore_target_native_factor_payload(transaction, step0_payload)
    assert restored.factor_state_sha256 == events[0].factor_state.factor_state_sha256
    assert capture_target_native_factor_state(transaction).factor_state_sha256 == (
        events[0].factor_state.factor_state_sha256
    )
    transaction.rollback()


def test_post_optimizer_step_callback_receives_every_completed_step() -> None:
    torch.manual_seed(126)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 3, dims=0))
    transaction = _transaction(child)
    domains = {
        "fit": _domain("fit", [[1, 2, 3]], "3" * 64),
        "calibration": _domain("calibration", [[4, 5, 6]], "4" * 64),
        "sealed": _domain("sealed", [[7, 8, 9]], "5" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=3,
        learning_rate=0.02,
        checkpoint_steps=(2,),
        fit_horizon=1,
        evaluation_horizon=1,
        top_k=2,
    )
    completed_steps: list[int] = []

    def invariant_hook(step: int) -> object:
        completed_steps.append(step)
        return {"selected_step": 999, "unlock_sealed": True}

    result = train_target_native_behavior_distillation(
        transaction,
        parent,
        domains,
        schedule,
        post_optimizer_step_callback=invariant_hook,
    )
    assert completed_steps == [1, 2, 3]
    assert result.selected_step != 999
    assert [value.step for value in result.checkpoints] == [0, 2, 3]
    transaction.rollback()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_post_optimizer_step_callback_failure_exactly_restores_step_zero(
    error_type: type[BaseException],
) -> None:
    torch.manual_seed(127)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 4, dims=0))
    transaction = _transaction(child)
    step0 = capture_target_native_factor_state(transaction)
    domains = {
        "fit": _domain("fit", [[1, 2, 3]], "6" * 64),
        "calibration": _domain("calibration", [[4, 5, 6]], "7" * 64),
        "sealed": _domain("sealed", [[7, 8, 9]], "8" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=3,
        learning_rate=0.02,
        checkpoint_steps=(1, 2),
        fit_horizon=1,
        evaluation_horizon=1,
        top_k=2,
    )
    completed_steps: list[int] = []

    def interrupted_hook(step: int) -> None:
        completed_steps.append(step)
        if step == 2:
            raise error_type("post-step invariant failed")

    with pytest.raises(error_type, match="post-step invariant failed"):
        train_target_native_behavior_distillation(
            transaction,
            parent,
            domains,
            schedule,
            post_optimizer_step_callback=interrupted_hook,
        )
    assert completed_steps == [1, 2]
    transaction.assert_live()
    assert capture_target_native_factor_state(transaction).factor_state_sha256 == (
        step0.factor_state_sha256
    )
    transaction.rollback()


def test_post_optimizer_step_callback_must_be_callable() -> None:
    torch.manual_seed(128)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    transaction = _transaction(child)
    domains = {
        "fit": _domain("fit", [[1, 2]], "9" * 64),
        "calibration": _domain("calibration", [[3, 4]], "a" * 64),
        "sealed": _domain("sealed", [[5, 6]], "b" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=1,
        learning_rate=0.02,
        fit_horizon=1,
        evaluation_horizon=1,
        top_k=2,
    )
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="post_optimizer_step_callback must be callable",
    ):
        train_target_native_behavior_distillation(
            transaction,
            parent,
            domains,
            schedule,
            post_optimizer_step_callback=object(),  # type: ignore[arg-type]
        )
    transaction.assert_live()
    transaction.rollback()


def test_durable_checkpoint_callback_failure_exactly_restores_step_zero() -> None:
    torch.manual_seed(121)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 5, dims=0))
    transaction = _transaction(child)
    step0 = capture_target_native_factor_state(transaction)
    domains = {
        "fit": _domain("fit", [[1, 2, 3]], "4" * 64),
        "calibration": _domain("calibration", [[4, 5, 6]], "5" * 64),
        "sealed": _domain("sealed", [[7, 8, 9]], "6" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=2,
        learning_rate=0.02,
        checkpoint_steps=(1,),
        fit_horizon=1,
        evaluation_horizon=1,
        top_k=2,
    )
    observed_steps: list[int] = []

    def failing_sink(event: behavior_distillation.DurableBehaviorCheckpoint) -> None:
        observed_steps.append(event.step)
        if event.step == 1:
            raise RuntimeError("durable sink failed")

    with pytest.raises(RuntimeError, match="durable sink failed"):
        train_target_native_behavior_distillation(
            transaction,
            parent,
            domains,
            schedule,
            durable_checkpoint_callback=failing_sink,
        )
    assert observed_steps == [0, 1]
    transaction.assert_live()
    assert capture_target_native_factor_state(transaction).factor_state_sha256 == (
        step0.factor_state_sha256
    )
    transaction.rollback()


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_durable_checkpoint_base_exception_exactly_restores_step_zero(
    error_type: type[BaseException],
) -> None:
    torch.manual_seed(124)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 4, dims=0))
    transaction = _transaction(child)
    step0 = capture_target_native_factor_state(transaction)
    domains = {
        "fit": _domain("fit", [[1, 2, 3]], "d" * 64),
        "calibration": _domain("calibration", [[4, 5, 6]], "e" * 64),
        "sealed": _domain("sealed", [[7, 8, 9]], "f" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=2,
        learning_rate=0.02,
        checkpoint_steps=(1,),
        fit_horizon=1,
        evaluation_horizon=1,
        top_k=2,
    )

    def interrupted_sink(
        event: behavior_distillation.DurableBehaviorCheckpoint,
    ) -> None:
        if event.step == 1:
            raise error_type("durable sink interrupted")

    with pytest.raises(error_type, match="durable sink interrupted"):
        train_target_native_behavior_distillation(
            transaction,
            parent,
            domains,
            schedule,
            durable_checkpoint_callback=interrupted_sink,
        )
    transaction.assert_live()
    assert capture_target_native_factor_state(transaction).factor_state_sha256 == (
        step0.factor_state_sha256
    )
    transaction.rollback()


def test_durable_checkpoint_callback_cannot_mutate_live_factors() -> None:
    torch.manual_seed(122)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    transaction = _transaction(child)
    step0 = capture_target_native_factor_state(transaction)
    domains = {
        "fit": _domain("fit", [[1, 2, 3]], "7" * 64),
        "calibration": _domain("calibration", [[4, 5, 6]], "8" * 64),
        "sealed": _domain("sealed", [[7, 8, 9]], "9" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=1,
        learning_rate=0.02,
        fit_horizon=1,
        evaluation_horizon=1,
        top_k=2,
    )

    def malicious_sink(
        event: behavior_distillation.DurableBehaviorCheckpoint,
    ) -> None:
        if event.step == 1:
            with torch.no_grad():
                transaction.output_trainable_parameters()[0].add_(1.0)

    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="callback altered live compiler factors",
    ):
        train_target_native_behavior_distillation(
            transaction,
            parent,
            domains,
            schedule,
            durable_checkpoint_callback=malicious_sink,
        )
    transaction.assert_live()
    assert capture_target_native_factor_state(transaction).factor_state_sha256 == (
        step0.factor_state_sha256
    )
    transaction.rollback()


@pytest.mark.parametrize("mutation", ["parameter", "storage"])
def test_durable_checkpoint_callback_repairs_factor_ownership_and_storage(
    mutation: str,
) -> None:
    torch.manual_seed(125)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    transaction = _transaction(child)
    step0 = capture_target_native_factor_state(transaction)
    registry = transaction.qkvo_handle.adapter_registry
    parameter_name, original = next(registry.named_parameters(recurse=False))
    original_data_ptr = int(original.data_ptr())
    domains = {
        "fit": _domain("fit", [[1, 2, 3]], "0" * 64),
        "calibration": _domain("calibration", [[4, 5, 6]], "1" * 64),
        "sealed": _domain("sealed", [[7, 8, 9]], "2" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=1,
        learning_rate=0.02,
        fit_horizon=1,
        evaluation_horizon=1,
        top_k=2,
    )

    def structural_sink(
        event: behavior_distillation.DurableBehaviorCheckpoint,
    ) -> None:
        if event.step != 1:
            return
        if mutation == "parameter":
            setattr(
                registry,
                parameter_name,
                nn.Parameter(original.detach().clone()),
            )
        else:
            original.data = original.detach().clone()

    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="callback altered live compiler factors",
    ):
        train_target_native_behavior_distillation(
            transaction,
            parent,
            domains,
            schedule,
            durable_checkpoint_callback=structural_sink,
        )
    transaction.assert_live()
    assert dict(registry.named_parameters(recurse=False))[parameter_name] is original
    assert int(original.data_ptr()) == original_data_ptr
    assert capture_target_native_factor_state(transaction).factor_state_sha256 == (
        step0.factor_state_sha256
    )
    transaction.rollback()


def test_candidate_only_policy_never_accepts_or_evaluates_a_sealed_suite(
    monkeypatch,
) -> None:
    torch.manual_seed(123)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 5, dims=0))
        parent.layers[0].self_attn.q_proj.weight.mul_(1.35)
    transaction = _transaction(child)
    domains = {
        "fit": _domain("fit", [[1, 2, 3], [4, 5, 6]], "a" * 64),
        "calibration": _domain("calibration", [[7, 8], [9, 10]], "b" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=3,
        learning_rate=0.04,
        checkpoint_steps=(1, 2),
        fit_horizon=2,
        evaluation_horizon=2,
        fit_prompts_per_step=2,
        top_k=3,
        gradient_clip_norm=5.0,
        sealed_evaluation_policy="disabled",
    )
    evaluated_domains: list[str] = []
    evaluator = behavior_distillation.evaluate_behavior_domain

    def audited_evaluator(*args, **kwargs):
        evaluated_domains.append(args[2].split_name)
        return evaluator(*args, **kwargs)

    monkeypatch.setattr(
        behavior_distillation, "evaluate_behavior_domain", audited_evaluator
    )
    result = train_target_native_behavior_distillation(
        transaction,
        parent,
        domains,
        schedule,
    )
    assert result.candidate_gate_passed is True
    assert result.sealed_evaluation_policy == "disabled"
    assert result.sealed_evaluation_unlocked is False
    assert result.sealed_metrics is None
    assert "sealed" not in evaluated_domains
    assert schedule.as_dict()["sealed_evaluation_policy"] == "disabled"
    assert result.evidence_payload()["candidate_only_evaluation"] is True
    assert result.evidence_payload()["sealed_suite_supplied"] is False
    assert result.evidence_payload()["sealed_evaluation_attempted"] is False
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="exactly fit, calibration",
    ):
        train_target_native_behavior_distillation(
            transaction,
            parent,
            {
                **domains,
                "sealed": _domain("sealed", [[11, 12]], "c" * 64),
            },
            schedule,
        )
    transaction.rollback()


def test_gud_freeze_skips_adam_state_until_thaw() -> None:
    torch.manual_seed(109)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 3, dims=0))
    transaction = _transaction(child)
    domains = {
        "fit": _domain("fit", [[1, 2]], "a" * 64),
        "calibration": _domain("calibration", [[3, 4]], "b" * 64),
        "sealed": _domain("sealed", [[5, 6]], "c" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=3,
        learning_rate=1e-3,
        fit_horizon=1,
        evaluation_horizon=1,
        gud_frozen_steps=1,
        top_k=2,
    )
    gud_ids = {
        id(value)
        for value in transaction.swiglu_handle.registry.parameters(recurse=False)
    }
    snapshots: list[dict[int, float | None]] = []

    class _AuditedAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            result = super().step(closure)
            snapshot: dict[int, float | None] = {}
            for group in self.param_groups:
                for parameter in group["params"]:
                    raw = self.state.get(parameter, {}).get("step")
                    snapshot[id(parameter)] = None if raw is None else float(raw.item())
            snapshots.append(snapshot)
            return result

    train_target_native_behavior_distillation(
        transaction,
        parent,
        domains,
        schedule,
        optimizer_factory=lambda parameters: _AuditedAdamW(
            parameters,
            lr=schedule.learning_rate,
            weight_decay=0.0,
        ),
    )
    assert all(snapshots[0][parameter_id] is None for parameter_id in gud_ids)
    assert all(snapshots[1][parameter_id] == 1.0 for parameter_id in gud_ids)
    assert all(snapshots[2][parameter_id] == 2.0 for parameter_id in gud_ids)
    transaction.rollback()


def test_factor_state_capture_restore_is_exact_and_factor_only() -> None:
    torch.manual_seed(101)
    model = _StrictToyTarget()
    transaction = _transaction(model)
    state = capture_target_native_factor_state(transaction)
    assert len(state.names) == 14
    assert all(name.startswith(("qkvo.", "gud.")) for name in state.names)
    manifest = state.as_manifest()
    assert manifest["schema"] == behavior_distillation.BEHAVIOR_FACTOR_PACK_SCHEMA
    assert "lexical_rank" not in manifest
    assert set(manifest) == {
        "schema",
        "factor_count",
        "factors",
        "factor_state_sha256",
        "parent_state_included",
        "native_child_state_included",
        "optimizer_state_included",
    }
    assert manifest["native_child_state_included"] is False
    with torch.no_grad():
        transaction.qkvo_handle.adapter_registry.q_left.add_(0.125)
        transaction.swiglu_handle.registry.gate_left.sub_(0.25)
    assert capture_target_native_factor_state(transaction).factor_state_sha256 != (
        state.factor_state_sha256
    )
    restore_target_native_factor_state(transaction, state)
    assert capture_target_native_factor_state(transaction).factor_state_sha256 == (
        state.factor_state_sha256
    )
    transaction.rollback()


def test_factor_only_payload_restores_exact_continuation_and_rejects_tamper() -> None:
    torch.manual_seed(107)
    model = _StrictToyTarget()
    transaction = _transaction(model)
    with torch.no_grad():
        for index, value in enumerate(transaction.output_trainable_parameters()):
            value.add_(float(index + 1) / 1000.0)
    selected = capture_target_native_factor_state(transaction)
    payload = selected.as_payload()
    with torch.no_grad():
        for value in transaction.output_trainable_parameters():
            value.zero_()
    restored = restore_target_native_factor_payload(transaction, payload)
    assert restored.factor_state_sha256 == selected.factor_state_sha256

    tampered = selected.as_payload()
    tampered["factor_tensors"][selected.names[0]].reshape(-1)[0] += 1.0
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="shape/dtype/hash authentication",
    ):
        restore_target_native_factor_payload(transaction, tampered)

    extra_field = selected.as_payload()
    extra_field["lexical_rank"] = 2
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="exactly the schema fields",
    ):
        restore_target_native_factor_payload(transaction, extra_field)
    transaction.rollback()


def test_v1_pack_partially_restores_combined_transaction_without_touching_lexical() -> (
    None
):
    torch.manual_seed(110)
    source_model = _StrictToyTarget()
    source_transaction = _transaction(source_model)
    with torch.no_grad():
        for index, (_name, value) in enumerate(
            behavior_distillation._factor_parameters(source_transaction)
        ):
            value.add_(float(index + 1) / 1000.0)
    source = capture_target_native_factor_state(source_transaction)
    assert source.schema == behavior_distillation.BEHAVIOR_FACTOR_PACK_SCHEMA
    assert len(source.names) == 14

    combined_model = _StrictToyTarget()
    combined_transaction = _transaction(combined_model, lexical_rank=2)
    lexical_registry = combined_transaction.lexical_handle.registry
    lexical_before = {
        name: value.detach().clone()
        for name, value in lexical_registry.named_parameters(recurse=False)
    }
    restored = restore_target_native_factor_payload(
        combined_transaction, source.as_payload()
    )
    assert restored.schema == behavior_distillation.BEHAVIOR_FACTOR_PACK_V2_SCHEMA
    assert restored.lexical_rank == 2
    assert len(restored.names) == 18
    assert restored.factor_state_sha256 != source.factor_state_sha256
    combined_entries = dict(
        behavior_distillation._factor_parameters(combined_transaction)
    )
    for name, value in zip(source.names, source.tensors, strict=True):
        assert torch.equal(combined_entries[name].detach().cpu(), value)
    assert all(
        torch.equal(value.detach(), lexical_before[name])
        for name, value in lexical_registry.named_parameters(recurse=False)
    )

    with torch.no_grad():
        lexical_registry.embedding_left.reshape(-1)[0] = 0.25
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="untouched exact-zero lexical initialization",
    ):
        restore_target_native_factor_payload(combined_transaction, source.as_payload())
    combined_transaction.rollback()
    source_transaction.rollback()


def test_v2_pack_restores_all_18_factors_and_binds_lexical_rank() -> None:
    torch.manual_seed(111)
    model = _StrictToyTarget()
    transaction = _transaction(model, lexical_rank=3)
    entries = behavior_distillation._factor_parameters(transaction)
    with torch.no_grad():
        for index, (_name, value) in enumerate(entries):
            value.add_(float(index + 1) / 2000.0)
    selected = capture_target_native_factor_state(transaction)
    payload = selected.as_payload()
    assert selected.schema == behavior_distillation.BEHAVIOR_FACTOR_PACK_V2_SCHEMA
    assert selected.lexical_rank == 3
    assert payload["schema"] == behavior_distillation.BEHAVIOR_FACTOR_PACK_V2_SCHEMA
    assert payload["factor_count"] == 18
    assert payload["lexical_rank"] == 3
    assert {name for name in selected.names if name.startswith("lexical.")} == {
        "lexical.embedding_left",
        "lexical.embedding_right",
        "lexical.lm_head_left",
        "lexical.lm_head_right",
    }

    with torch.no_grad():
        for _name, value in entries:
            value.zero_()
    restored = restore_target_native_factor_payload(transaction, payload)
    assert restored.factor_state_sha256 == selected.factor_state_sha256
    assert all(
        torch.equal(actual.detach().cpu(), expected)
        for (_name, actual), expected in zip(
            behavior_distillation._factor_parameters(transaction),
            selected.tensors,
            strict=True,
        )
    )

    wrong_rank = selected.as_payload()
    wrong_rank["lexical_rank"] = 4
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="lexical rank",
    ):
        restore_target_native_factor_payload(transaction, wrong_rank)
    transaction.rollback()


def test_low_precision_child_uses_fp32_factors_and_adam_moments() -> None:
    torch.manual_seed(104)
    model = _StrictToyTarget().to(dtype=torch.float16)
    transaction = _transaction(model)
    factors = transaction.output_trainable_parameters()
    installation = transaction.installation_report()
    assert factors
    assert {value.dtype for value in factors} == {torch.float32}
    assert installation["dense_qkvo_installation"]["factor_dtype"] == ("torch.float32")
    assert installation["dense_gate_up_down_installation"]["factor_dtype"] == (
        "torch.float32"
    )

    optimizer = torch.optim.AdamW(factors, lr=1e-3, weight_decay=0.0, eps=1e-8)
    for index, value in enumerate(factors):
        value.grad = torch.zeros_like(value)
        if index % 2:
            value.grad.reshape(-1)[0] = 1e-4
    optimizer.step()

    assert all(bool(torch.isfinite(value).all()) for value in factors)
    moment_dtypes = {
        state_value.dtype
        for value in factors
        for state_value in optimizer.state[value].values()
        if isinstance(state_value, Tensor) and state_value.is_floating_point()
    }
    assert moment_dtypes == {torch.float32}
    transaction.rollback()


def test_poisoned_factor_snapshot_restore_and_combined_rollback_are_exact() -> None:
    torch.manual_seed(105)
    model = _StrictToyTarget()
    exact = exact_unique_parameter_count(model)
    transaction = _transaction(model)
    step0 = capture_target_native_factor_state(transaction)
    with torch.no_grad():
        transaction.qkvo_handle.adapter_registry.q_left.reshape(-1)[0] = torch.nan
        transaction.swiglu_handle.registry.gate_left.reshape(-1)[0] = torch.inf
    with pytest.raises(DenseGQAQKVOResidualError, match="NaN|infinity"):
        transaction.assert_live()

    restore_target_native_factor_state(transaction, step0)
    transaction.assert_live()
    assert capture_target_native_factor_state(transaction).factor_state_sha256 == (
        step0.factor_state_sha256
    )

    with torch.no_grad():
        transaction.qkvo_handle.adapter_registry.k_left.reshape(-1)[0] = torch.nan
        transaction.swiglu_handle.registry.up_left.reshape(-1)[0] = torch.inf
    transaction.rollback()
    assert exact_unique_parameter_count(model) == exact


def test_poisoning_optimizer_preserves_original_error_and_restores_step_zero() -> None:
    class _PoisoningOptimizer(torch.optim.Optimizer):
        def __init__(self, parameters, *, lr: float) -> None:
            super().__init__(parameters, {"lr": lr})

        @torch.no_grad()
        def step(self, closure=None):
            del closure
            self.param_groups[0]["params"][0].reshape(-1)[0] = torch.nan

    torch.manual_seed(106)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 3, dims=0))
    transaction = _transaction(child)
    step0 = capture_target_native_factor_state(transaction)
    domains = {
        "fit": _domain("fit", [[1, 2, 3]], "a" * 64),
        "calibration": _domain("calibration", [[4, 5, 6]], "b" * 64),
        "sealed": _domain("sealed", [[7, 8, 9]], "c" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=1,
        learning_rate=1e-3,
        fit_horizon=1,
        evaluation_horizon=1,
        top_k=2,
    )
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="Optimizer produced non-finite compiler factors",
    ):
        train_target_native_behavior_distillation(
            transaction,
            parent,
            domains,
            schedule,
            optimizer_factory=lambda parameters: _PoisoningOptimizer(
                parameters, lr=schedule.learning_rate
            ),
        )
    transaction.assert_live()
    assert capture_target_native_factor_state(transaction).factor_state_sha256 == (
        step0.factor_state_sha256
    )
    transaction.rollback()


def test_output_only_training_audits_all_factors_and_retains_step_zero() -> None:
    torch.manual_seed(102)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 5, dims=0))
        parent.layers[0].self_attn.q_proj.weight.mul_(1.35)
    transaction = _transaction(child)
    initial_factor_payload = capture_target_native_factor_state(
        transaction
    ).as_payload()
    domains = {
        "fit": _domain("fit", [[1, 2, 3], [4, 5, 6]], "1" * 64),
        "calibration": _domain("calibration", [[7, 8], [9, 10]], "2" * 64),
        "sealed": _domain("sealed", [[11, 12], [13, 14]], "3" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=3,
        learning_rate=0.04,
        checkpoint_steps=(1, 2),
        fit_horizon=2,
        evaluation_horizon=2,
        fit_prompts_per_step=2,
        top_k=3,
        gradient_clip_norm=5.0,
    )
    result = train_target_native_behavior_distillation(
        transaction,
        parent,
        domains,
        schedule,
        seed=77,
        initial_factor_payload=initial_factor_payload,
        optimizer_factory=lambda parameters: torch.optim.SGD(
            parameters, lr=schedule.learning_rate
        ),
    )
    assert [value.step for value in result.checkpoints] == [0, 1, 2, 3]
    assert result.checkpoints[0].factor_state_sha256
    assert result.selected_step in {0, 1, 2, 3}
    selected_checkpoint = next(
        value for value in result.checkpoints if value.step == result.selected_step
    )
    assert result.factor_pack.factor_state_sha256 == (
        selected_checkpoint.factor_state_sha256
    )
    assert result.training_audit.parent_greedy_prefix_count == 12
    assert result.training_audit.child_on_policy_prefix_count == 12
    assert result.training_audit.parent_greedy_cache_prompt_count == 2
    assert result.training_audit.parent_greedy_cache_forward_count == 4
    assert result.training_audit.parent_output_forward_count == 16
    assert result.training_audit.gradient_activation.complete is True
    assert len(result.training_audit.gradient_activation.factor_names) == 14
    assert result.candidate_metrics.action_count == 4
    assert result.candidate_metrics.rollout_count == 2
    assert result.fit_metrics.split == "calibration-exploratory"
    assert result.sealed_metrics is not None
    assert result.sealed_metrics.split == "heldout-evaluation"
    assert result.factor_pack_payload()["optimizer_state_included"] is False
    evidence = result.evidence_payload()
    assert result.evidence_sha256 == (
        "77bb6ff79bed156a848ddbfe054607bd66f6b7d8a5e6556711853ec0ae7d3bf3"
    )
    assert behavior_distillation._canonical_sha256(schedule.as_dict()) == (
        "ffa5255ce8daf43f896b79492c914db977e993cfc87592d9334599d03d29b7f4"
    )
    assert behavior_distillation._canonical_sha256(evidence) == result.evidence_sha256
    assert evidence["factor_pack_manifest"] == result.factor_pack.as_manifest()
    assert evidence["continued_from_factor_pack"] is True
    assert (
        evidence["initial_factor_state_sha256"]
        == (initial_factor_payload["factor_state_sha256"])
    )
    assert (
        evidence["source_factor_state_sha256"]
        == (initial_factor_payload["factor_state_sha256"])
    )
    assert evidence["initial_factor_restore_mode"] == "full"
    assert (
        evidence["combined_step0_factor_state_sha256"]
        == (initial_factor_payload["factor_state_sha256"])
    )
    assert evidence["lexical_initialization_unchanged"] is None
    assert (
        evidence["training_audit"]["gradient_activation"][
            "every_qkvo_gud_factor_gradient_activated"
        ]
        is True
    )
    assert evidence["internal_numeric_matching_used"] is False
    assert not any(name.startswith("guided_") for name in evidence)
    assert "selected_factor_pack_retained_on_acceptance" not in evidence
    assert (
        "every_authorized_factor_gradient_activated"
        not in (evidence["training_audit"]["gradient_activation"])
    )
    assert result.as_dict()["evidence_payload"] == evidence
    assert parent.internal_output_request_count == 0
    assert child.internal_output_request_count == 0
    transaction.rollback()


def test_combined_training_restores_v1_then_audits_and_emits_v2_evidence() -> None:
    torch.manual_seed(112)
    source_model = _StrictToyTarget()
    source_transaction = _transaction(source_model)
    with torch.no_grad():
        for index, value in enumerate(source_transaction.output_trainable_parameters()):
            value.add_(float(index + 1) / 1000.0)
    source_payload = capture_target_native_factor_state(source_transaction).as_payload()
    source_sha = source_payload["factor_state_sha256"]

    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 4, dims=0))
        parent.layers[1].self_attn.o_proj.weight.mul_(1.2)
    transaction = _transaction(child, lexical_rank=2)
    domains = {
        "fit": _domain("fit", [[1, 2, 3], [4, 5, 6]], "a" * 64),
        "calibration": _domain("calibration", [[7, 8], [9, 10]], "b" * 64),
        "sealed": _domain("sealed", [[11, 12], [13, 14]], "c" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=4,
        learning_rate=0.03,
        checkpoint_steps=(1, 2, 3),
        fit_horizon=2,
        evaluation_horizon=2,
        fit_prompts_per_step=2,
        qkvo_frozen_steps=1,
        gud_frozen_steps=2,
        lexical_embedding_frozen_steps=2,
        lexical_head_frozen_steps=0,
        top_k=3,
        gradient_clip_norm=5.0,
    )
    result = train_target_native_behavior_distillation(
        transaction,
        parent,
        domains,
        schedule,
        seed=13,
        initial_factor_payload=source_payload,
        optimizer_factory=lambda parameters: torch.optim.SGD(
            parameters, lr=schedule.learning_rate
        ),
    )
    audit = result.training_audit.gradient_activation
    assert len(audit.factor_names) == 18
    assert audit.complete is True
    assert audit.qkvo_gud_complete is True
    assert audit.lexical_complete is True
    assert (
        result.factor_pack.schema
        == behavior_distillation.BEHAVIOR_FACTOR_PACK_V2_SCHEMA
    )
    assert result.factor_pack.lexical_rank == 2
    assert result.factor_pack.as_manifest()["factor_count"] == 18
    assert result.factor_pack.as_manifest()["lexical_rank"] == 2

    evidence = result.evidence_payload()
    combined_step0_sha = result.checkpoints[0].factor_state_sha256
    assert evidence["source_factor_state_sha256"] == source_sha
    assert evidence["initial_factor_state_sha256"] == source_sha
    assert evidence["initial_factor_payload_schema"] == (
        behavior_distillation.BEHAVIOR_FACTOR_PACK_SCHEMA
    )
    assert evidence["initial_factor_restore_mode"] == "qkvo-gud-partial"
    assert evidence["combined_step0_factor_state_sha256"] == combined_step0_sha
    assert combined_step0_sha != source_sha
    assert evidence["lexical_initialization_state_sha256"]
    assert evidence["lexical_initialization_unchanged"] is True
    assert (
        evidence["training_audit"]["gradient_activation"][
            "every_qkvo_gud_factor_gradient_activated"
        ]
        is True
    )
    assert (
        evidence["training_audit"]["gradient_activation"][
            "every_lexical_factor_gradient_activated"
        ]
        is True
    )
    assert (
        evidence["training_audit"]["gradient_activation"][
            "every_combined_factor_gradient_activated"
        ]
        is True
    )
    transaction.rollback()
    source_transaction.rollback()


def test_disjointness_and_candidate_gate_keep_sealed_locked(monkeypatch) -> None:
    torch.manual_seed(103)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 7, dims=0))
    transaction = _transaction(child)
    shared = list(range(1, 9))
    overlap = {
        "fit": _domain("fit", [shared + [9]], "4" * 64),
        "calibration": _domain("calibration", [shared + [10]], "5" * 64),
        "sealed": _domain("sealed", [[11, 12]], "6" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=2,
        learning_rate=0.02,
        fit_horizon=1,
        evaluation_horizon=1,
        top_k=2,
    )
    with pytest.raises(TargetNativeBehaviorDistillationError, match="prefix overlaps"):
        train_target_native_behavior_distillation(
            transaction, parent, overlap, schedule
        )

    disjoint = {
        "fit": _domain("fit", [[1, 2]], "7" * 64),
        "calibration": _domain("calibration", [[3, 4]], "8" * 64),
        "sealed": _domain("sealed", [[5, 6]], "9" * 64),
    }
    evaluated_domains: list[str] = []
    evaluator = behavior_distillation.evaluate_behavior_domain

    def audited_evaluator(*args, **kwargs):
        evaluated_domains.append(args[2].split_name)
        return evaluator(*args, **kwargs)

    monkeypatch.setattr(
        behavior_distillation, "evaluate_behavior_domain", audited_evaluator
    )
    result = train_target_native_behavior_distillation(
        transaction,
        parent,
        disjoint,
        schedule,
        candidate_thresholds=CandidateBehaviorThresholds(
            minimum_action_accuracy=1.0,
            minimum_exact_rollout_fraction=1.0,
            minimum_token_match_fraction=1.0,
            minimum_lcp_fraction=1.0,
        ),
    )
    assert result.candidate_gate_passed is False
    assert result.sealed_evaluation_unlocked is False
    assert result.sealed_metrics is None
    assert result.as_dict()["sealed_used_for_selection"] is False
    assert "sealed" not in evaluated_domains
    transaction.rollback()
