from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import neural_forge.hf_gud_output_jacobian_replay as replay
from neural_forge.hf_dense_swiglu_projection_residual_adapter import (
    install_dense_swiglu_projection_residual_adapter,
)
from neural_forge.hf_gud_output_jacobian_audit import (
    GUD_ROLE_ORDER,
    HARD_MAX_AUXILIARY_BYTES,
    GUDStageGeometry,
)
from neural_forge.hf_output_jacobian_sketch import (
    OutputJacobianSketchPhaseError,
)
from neural_forge.hf_qkvo_output_jacobian_audit import _LossContext
from neural_forge.hf_target_native_behavior_distillation import (
    BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
    BehaviorDistillationSchedule,
    BehaviorDomainSuite,
    BehaviorPrompt,
    GuidedFactorSourceBinding,
    TargetNativeFactorState,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        )


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _ToyMLP()


def _toy_replay(*, tamper_pass2: bool) -> tuple[str, str, object, object]:
    torch.manual_seed(11)
    model = _ToyModel()
    handle = install_dense_swiglu_projection_residual_adapter(
        model,
        (model.mlp,),
        layer_indices=(0,),
        hidden_size=4,
        intermediate_size=6,
        rank=1,
        initialization_seed=17,
    )
    with torch.no_grad():
        for role in GUD_ROLE_ORDER:
            getattr(handle.registry, f"{role}_left").normal_(0.0, 0.1)
            getattr(handle.registry, f"{role}_right").normal_(0.0, 0.1)
    transaction = SimpleNamespace(swiglu_handle=handle)
    geometry = GUDStageGeometry(1, 4, 6)
    config = replay.GUDOutputJacobianReplayConfig(
        prompt_count=1,
        horizon=1,
        rank=1,
        sketch_width=2,
        balanced_envelope_width=2,
        balanced_stream_rows=2,
        fingerprint_lanes=2,
        fingerprint_chunk_elements=64,
        max_callback_rows=32,
        max_bytes=64 * 1024 * 1024,
        seed=7,
    )
    sketches = replay._build_sketches(transaction, geometry, config)
    collector = replay._GUDReplayCollector(
        transaction=transaction,
        geometry=geometry,
        sketches=sketches,
        config=config,
    )
    source = torch.randn(
        (8, 4), generator=torch.Generator(device="cpu").manual_seed(33)
    )

    def run_pass(phase: str, inputs: Tensor) -> str:
        collector.begin_pass(phase, 1)  # type: ignore[arg-type]
        context = _LossContext(
            lane="parent-greedy",
            prompt_id="toy-prompt",
            prefix_sha256=_digest("toy-prefix"),
            loss_weight=1.0,
            block_ordinal=0,
            prefix_length=8,
        )
        collector.prepare_context(context)
        collector.current = context
        output = model.mlp(inputs.clone().requires_grad_())
        collector.current = None
        output.square().sum().backward()
        digest = collector.finish_pass()
        model.zero_grad(set_to_none=True)
        return digest

    collector.install()
    try:
        pass1 = run_pass("pass1", source)
        for sketch in sketches.values():
            sketch.finish_pass1()
        collector.finalize_pass1_envelopes()
        second = source.clone()
        if tamper_pass2:
            second[3, 2].add_(0.25)
        pass2 = run_pass("pass2", second)
        return pass1, pass2, collector, sketches
    finally:
        collector.remove()
        handle.rollback()


def _source_binding() -> GuidedFactorSourceBinding:
    inherited_names = sorted(
        {
            *(
                f"qkvo.{role}_{side}"
                for role in ("q", "k", "v", "o")
                for side in ("left", "right")
            ),
            *(
                f"gud.{role}_{side}"
                for role in ("gate", "up", "down")
                for side in ("left", "right")
            ),
            *(
                f"lexical.{role}_{side}"
                for role in ("embedding", "lm_head")
                for side in ("left", "right")
            ),
        }
    )
    return GuidedFactorSourceBinding(
        source_v2_factor_state_sha256=_digest("source-v2"),
        source_v2_tensor_sha256s=tuple(
            (name, _digest(f"v2:{name}")) for name in inherited_names
        ),
        audit_report_sha256=_digest("qkvo-audit"),
        basis_envelope_file_sha256=_digest("qkvo-envelope"),
        basis_payload_sha256=_digest("qkvo-payload"),
        basis_manifest_sha256=_digest("qkvo-manifest"),
        source_basis_tensor_sha256s=tuple(
            (role, _digest(f"basis:{role}")) for role in ("q", "k", "v", "o")
        ),
        inherited_installation_sha256=_digest("qkvo-installation"),
        inherited_factor_state_sha256_at_install=_digest("qkvo-state"),
        adapter_site_indices_sha256=_digest("adapter-sites"),
        audit_site_indices_sha256=_digest("audit-sites"),
    )


def test_mixed_precision_role_local_vjp_matches_live_autograd_exactly() -> None:
    generator = torch.Generator(device="cpu").manual_seed(91)
    inputs = torch.randn(
        (7, 5), generator=generator, dtype=torch.float16, requires_grad=True
    )
    weight = torch.randn((4, 5), generator=generator, dtype=torch.float16)
    left = torch.randn((4, 2), generator=generator, dtype=torch.float32)
    right = torch.randn((2, 5), generator=generator, dtype=torch.float32)
    delta = torch.randn((7, 4), generator=generator, dtype=torch.float16)
    output = F.linear(inputs, weight) + F.linear(
        F.linear(inputs.to(torch.float32), right), left
    ).to(torch.float16)
    (output * delta).sum().backward()

    actual = replay.role_local_mixed_precision_input_vjp_prevalidated(
        delta,
        delta.to(torch.float32),
        weight,
        left,
        right,
    )

    assert inputs.grad is not None
    assert torch.equal(actual, inputs.grad.to(torch.float32))
    assert actual.dtype == torch.float32
    assert weight.dtype == torch.float16


def test_toy_collector_replays_all_separate_gud_sites_and_finalizes() -> None:
    pass1, pass2, collector, sketches = _toy_replay(tamper_pass2=False)

    assert pass1 == pass2
    assert collector.active_hook_count == 0
    assert set(collector.observations) == {
        (role, 0) for role in GUD_ROLE_ORDER
    }
    assert all(counts == [1, 1] for counts in collector.observations.values())
    results = {key: sketch.finalize() for key, sketch in sketches.items()}
    assert results[("gate", 0)].guided_right_rows.shape == (1, 4)
    assert results[("up", 0)].guided_right_rows.shape == (1, 4)
    assert results[("down", 0)].guided_right_rows.shape == (1, 6)


def test_full_element_replay_authenticator_rejects_one_element_tamper() -> None:
    pass1, pass2, _collector, sketches = _toy_replay(tamper_pass2=True)

    assert pass1 != pass2
    with pytest.raises(OutputJacobianSketchPhaseError, match="identities"):
        sketches[("gate", 0)].finalize()


def test_exact_production_memory_plan_is_bounded_and_has_no_full_tables() -> None:
    config = replay.GUDOutputJacobianReplayConfig()
    plan = replay.plan_gud_real_replay_memory(
        GUDStageGeometry(30, 1536, 5120),
        replay_blocks=384,
        max_rows=135,
        config=config,
    )

    assert plan["accounted_auxiliary_peak_bytes"] == 254_496_064
    assert plan["accounted_auxiliary_peak_bytes"] < HARD_MAX_AUXILIARY_BYTES
    assert plan["within_limit"] is True
    assert plan["site_count"] == 90
    assert plan["full_all_sites_replay_tensor_table_bytes"] == 0
    assert plan["full_lanes_rows_width_coefficient_table_bytes"] == 0
    with pytest.raises(replay.GUDOutputJacobianReplayLimitError, match="exceeds"):
        replay.plan_gud_real_replay_memory(
            GUDStageGeometry(30, 1536, 5120),
            replay_blocks=5000,
            max_rows=135,
            config=config,
        )


def test_behavior_identity_binds_candidate24_prompt_order_and_exact_schedule() -> None:
    prompts = tuple(BehaviorPrompt((index + 1, index + 101)) for index in range(24))
    domain = BehaviorDomainSuite(
        split_name="fit",
        prompts=prompts,
        artifact_sha256=_digest("candidate24-artifact"),
    )
    schedule = BehaviorDistillationSchedule(
        steps=1,
        learning_rate=1e-4,
        fit_horizon=8,
        evaluation_horizon=8,
    )
    plans = tuple(
        SimpleNamespace(
            prompt=prompt,
            parent_tokens=tuple(range(index, index + 8)),
            child_tokens=tuple(range(index + 1, index + 9)),
            position_weights=(1.0,) * 8,
            child_indices=tuple(range(8)),
        )
        for index, prompt in enumerate(prompts)
    )
    fields = replay._behavior_replay_report_fields(
        domain,
        schedule,
        plans,
        replay_blocks=384,
        pass1_loss_sha256=_digest("loss"),
        pass2_loss_sha256=_digest("loss"),
        pass1_authenticator_sha256=_digest("auth"),
        pass2_authenticator_sha256=_digest("auth"),
    )

    assert fields["basis_domain"]["prompt_count"] == 24
    assert [row["prompt_id"] for row in fields["basis_domain"]["prompts"]] == [
        prompt.prompt_id for prompt in prompts
    ]
    assert fields["basis_domain_sha256"] == domain.domain_sha256
    assert fields["basis_artifact_sha256"] == domain.artifact_sha256
    assert fields["behavior_schedule"] == schedule.as_dict()
    assert len(fields["behavior_schedule_sha256"]) == 64
    assert len(fields["replay_plan_sha256"]) == 64

    reordered = BehaviorDomainSuite(
        split_name="fit",
        prompts=tuple(reversed(prompts)),
        artifact_sha256=domain.artifact_sha256,
    )
    assert reordered.domain_sha256 != fields["basis_domain_sha256"]
    sealed = BehaviorDomainSuite(
        split_name="heldout-evaluation",
        prompts=prompts,
        artifact_sha256=domain.artifact_sha256,
    )
    assert sealed.domain_sha256 != fields["basis_domain_sha256"]


def test_live_v3_capture_derives_portable_identity_and_rejects_binding_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _source_binding()
    factor_names = sorted(
        {
            *(name for name, _digest_value in binding.source_v2_tensor_sha256s),
            *(
                f"guided_qkvo.{role}_{side}"
                for role in ("q", "k", "v", "o")
                for side in ("left", "right")
            ),
        }
    )
    parameters = tuple(nn.Parameter(torch.tensor([float(index + 1)])) for index in range(26))
    entries = tuple(zip(factor_names, parameters, strict=True))
    tensors = tuple(value.detach().clone() for value in parameters)
    tensor_hashes = tuple(_digest(f"factor:{name}") for name in factor_names)
    state = TargetNativeFactorState(
        names=tuple(factor_names),
        tensors=tensors,
        tensor_sha256s=tensor_hashes,
        factor_state_sha256=_digest("runtime-v3"),
        parameter_ids=tuple(id(value) for value in parameters),
        parameter_data_ptrs=tuple(int(value.data_ptr()) for value in parameters),
        schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        lexical_rank=8,
        guided_source_binding=binding,
    )
    registry = nn.Module()
    qkvo_handle = SimpleNamespace(adapter_registry=registry)
    transaction = SimpleNamespace(qkvo_handle=qkvo_handle)
    expected_raw = {
        "audit_report_sha256": binding.audit_report_sha256,
        "basis_envelope_file_sha256": binding.basis_envelope_file_sha256,
        "basis_payload_sha256": binding.basis_payload_sha256,
        "basis_manifest_sha256": binding.basis_manifest_sha256,
        "source_basis_tensor_sha256s": dict(binding.source_basis_tensor_sha256s),
        "inherited_installation_sha256": binding.inherited_installation_sha256,
        "inherited_factor_state_sha256_at_install": (
            binding.inherited_factor_state_sha256_at_install
        ),
        "inherited_handle_identity": id(qkvo_handle),
        "inherited_registry_identity": id(registry),
        "adapter_site_indices_sha256": binding.adapter_site_indices_sha256,
        "audit_site_indices_sha256": binding.audit_site_indices_sha256,
    }
    tranche = SimpleNamespace(
        inherited_handle=qkvo_handle,
        training_stage="joint",
        assert_live=lambda: None,
        factor_source_binding=lambda: expected_raw,
    )
    monkeypatch.setattr(replay._trainer, "_factor_parameters", lambda *_args, **_kwargs: entries)
    monkeypatch.setattr(
        replay,
        "capture_target_native_factor_state",
        lambda *_args, **_kwargs: state,
    )

    source = replay._capture_live_v3_provenance(transaction, tranche, binding)
    fields = replay._source_v3_report_fields(source, source, binding)

    assert fields["source_v3_runtime_factor_state_sha256_before"] == _digest(
        "runtime-v3"
    )
    assert fields["source_v3_runtime_factor_state_sha256_after"] == _digest(
        "runtime-v3"
    )
    assert fields["source_v3_portable_factor_state_sha256"] == (
        replay.portable_v3_factor_state_sha256(state.as_manifest(), binding)
    )
    assert fields["source_v3_factor_manifest_sha256"] == source.factor_manifest_sha256
    assert fields["source_v3_tensor_sha256s"] == dict(zip(factor_names, tensor_hashes, strict=True))
    assert fields["source_guided_qkvo_binding_sha256"] == binding.binding_sha256
    assert fields["source_v3_state_unchanged"] is True

    tranche.factor_source_binding = lambda: {**expected_raw, "audit_report_sha256": _digest("tampered")}
    with pytest.raises(
        replay.GUDOutputJacobianReplayTopologyError,
        match="does not authenticate",
    ):
        replay._capture_live_v3_provenance(transaction, tranche, binding)
