from __future__ import annotations

import hashlib
import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import neural_forge.hf_target_native_behavior_distillation as trainer_module
from neural_forge.hf_dense_gqa_qkvo_residual_adapter import (
    DenseGQAGeometry,
    DenseGQASite,
    DenseGQAQKVOResidualProjection,
    SparseDenseGQAQKVOResidualRegistry,
)
from neural_forge.hf_qkvo_output_jacobian_audit import (
    GUIDED_QKVO_RIGHT_BASIS_SCHEMA,
    QKVO_OUTPUT_JACOBIAN_AUDIT_SCHEMA,
    QKVOOutputJacobianAuditConfig,
    QKVOOutputJacobianAuditError,
    QKVOOutputJacobianAuditLimitError,
    QKVOOutputJacobianAuditReport,
    QKVOOutputJacobianAuditReplayError,
    QKVOOutputJacobianAuditResult,
    QKVOOutputJacobianAuditTopologyError,
    _aggregate_reports,
    _audit_decision,
    _build_replay_plans,
    _callback_hotpath_source_audit,
    _canonical_sha256,
    _compact_numeric_fingerprint,
    _diagnostic_passed_roles,
    _greedy_token_path,
    _LossContext,
    _QKVOHookCollector,
    _ReplayPlan,
    _replay_loss_pass,
    _require_memory_within_limit,
    _sketch_memory_plan,
    _slice_qkvo_blocks,
    _streamed_factor_state_evidence,
    _tensor_manifest_row,
    _tensor_sha256,
    _trainer_compatible_tensor_sha256,
    _validate_exact_topology,
    _validate_inherited_qkvo_factors,
    _write_device_tensor_fingerprint,
)
from neural_forge.hf_target_native_behavior_distillation import (
    BehaviorDistillationSchedule,
    BehaviorDomainSuite,
    BehaviorPrompt,
)


def test_config_is_exact_and_hash_bound() -> None:
    config = QKVOOutputJacobianAuditConfig()
    assert config.to_dict() == {
        "prompt_count": 24,
        "horizon": 8,
        "lanes": ["parent-greedy", "child-greedy"],
        "rank": 8,
        "sketch_width": 16,
        "seed": 0,
        "min_normal_fraction": 0.2,
        "min_guided_over_random_ratio": 1.5,
        "max_bytes": 268_435_456,
    }
    assert config.sha256() == _canonical_sha256(config.to_dict())
    with pytest.raises(QKVOOutputJacobianAuditError, match="24 prompts"):
        QKVOOutputJacobianAuditConfig(prompt_count=23)
    with pytest.raises(QKVOOutputJacobianAuditError, match="256 MiB"):
        QKVOOutputJacobianAuditConfig(max_bytes=268_435_457)


def test_qkvo_group_slicing_is_exact_and_view_only() -> None:
    geometry = DenseGQAGeometry(2, 6, 4, 2, 2)
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    query_delta = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    d, x = _slice_qkvo_blocks("q", 1, geometry, hidden, query_delta)
    assert torch.equal(d, query_delta[:, 4:8])
    assert x is hidden

    kv_delta = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    d, x = _slice_qkvo_blocks("k", 1, geometry, hidden, kv_delta)
    assert torch.equal(d, kv_delta[:, 2:4])
    assert x is hidden

    context = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    output_delta = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    d, x = _slice_qkvo_blocks("o", 1, geometry, context, output_delta)
    assert d is output_delta
    assert torch.equal(x, context[:, 4:8])


def test_probabilistic_authenticator_binds_every_element_and_is_finite() -> None:
    inputs = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    delta = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    first = _compact_numeric_fingerprint(inputs, delta)
    assert first == _compact_numeric_fingerprint(inputs.clone(), delta.clone())

    # The retired implementation sampled only 16 locations.  Swapping these
    # two unsampled values preserves every scalar summary it used while the
    # full-element, order-sensitive authenticator must still change.
    retired_sample_indices = set(
        torch.linspace(0, 63, steps=16).round().to(torch.int64).tolist()
    )
    assert {18, 19}.isdisjoint(retired_sample_indices)
    changed = delta.clone()
    flat_changed = changed.reshape(-1)
    flat_changed[18], flat_changed[19] = (
        flat_changed[19].clone(),
        flat_changed[18].clone(),
    )
    assert changed.sum() == delta.sum()
    assert changed.square().sum() == delta.square().sum()
    assert changed.abs().sum() == delta.abs().sum()
    assert changed.min() == delta.min()
    assert changed.max() == delta.max()
    assert first != _compact_numeric_fingerprint(inputs, changed)

    strided = torch.arange(20, dtype=torch.float32)[::2]
    assert _tensor_sha256(strided) == _tensor_sha256(strided.clone())
    poisoned = delta.clone()
    poisoned[0, 0] = float("nan")
    with pytest.raises(QKVOOutputJacobianAuditReplayError, match="non-finite"):
        _compact_numeric_fingerprint(inputs, poisoned)


def test_tensor_hashes_stream_through_bounded_cpu_staging() -> None:
    value = torch.arange((5 * 1024 * 1024) // 4 + 17, dtype=torch.float32)
    staging: list[int] = []
    actual = _tensor_sha256(value, staging_observer=staging.append)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(memoryview(value.view(torch.uint8).numpy()))
    assert actual == digest.hexdigest()
    assert len(staging) == 2
    assert max(staging) <= 4 * 1024 * 1024

    compatible_staging: list[int] = []
    compatible = _trainer_compatible_tensor_sha256(
        value, staging_observer=compatible_staging.append
    )
    assert compatible == trainer_module._tensor_sha256(value)  # noqa: SLF001
    assert len(compatible_staging) == 2
    assert max(compatible_staging) <= 4 * 1024 * 1024


def test_factor_state_evidence_is_streamed_and_trainer_compatible() -> None:
    lexical_registry = nn.Module()
    lexical_registry.rank = 2
    transaction = SimpleNamespace(
        lexical_handle=SimpleNamespace(registry=lexical_registry)
    )
    entries = (
        (
            "qkvo.large",
            nn.Parameter(
                torch.arange((5 * 1024 * 1024) // 4 + 11, dtype=torch.float32)
            ),
        ),
        ("qkvo.small", nn.Parameter(torch.arange(13, dtype=torch.float32))),
    )
    staging: list[int] = []
    evidence = _streamed_factor_state_evidence(
        transaction,  # type: ignore[arg-type]
        entries,
        staging_observer=staging.append,
    )
    expected_hashes = tuple(
        trainer_module._tensor_sha256(value)  # noqa: SLF001
        for _name, value in entries
    )
    expected_rows = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "tensor_sha256": tensor_hash,
        }
        for (name, value), tensor_hash in zip(entries, expected_hashes, strict=True)
    ]
    expected_state = trainer_module._factor_state_sha256(  # noqa: SLF001
        expected_rows,
        schema=trainer_module.BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
        lexical_rank=2,
    )
    assert evidence["factor_state_sha256"] == expected_state
    assert evidence["tensor_sha256s"] == expected_hashes
    assert evidence["capture_mode"] == "streamed-live-manifest-no-full-cpu-clone"
    assert evidence["retained_cpu_clone_bytes"] == 0
    assert max(staging) <= 4 * 1024 * 1024
    source = inspect.getsource(_streamed_factor_state_evidence)
    assert ".clone(" not in source
    assert ".cpu(" not in source


class _FakeSketch:
    def __init__(self) -> None:
        self.pass1: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.pass2: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.identity_chains: dict[int, str] = {}

    def add_prevalidated_pass1(
        self,
        delta: torch.Tensor,
        inputs: torch.Tensor,
    ) -> None:
        self.pass1.append((delta.detach().clone(), inputs.detach().clone()))

    def add_prevalidated_pass2(
        self,
        delta: torch.Tensor,
        inputs: torch.Tensor,
    ) -> None:
        self.pass2.append((delta.detach().clone(), inputs.detach().clone()))

    def bind_prevalidated_identity_chain(
        self, pass_number: int, chain_sha256: str
    ) -> None:
        self.identity_chains[pass_number] = chain_sha256


def _toy_hook_collector() -> tuple[
    _QKVOHookCollector,
    dict[int, nn.Module],
    dict[tuple[str, int], _FakeSketch],
]:
    geometry = DenseGQAGeometry(30, 6, 4, 2, 2)
    sites = tuple(
        DenseGQASite(layer, group) for layer in range(30) for group in range(2)
    )
    registry = SparseDenseGQAQKVOResidualRegistry(
        sites,
        geometry=geometry,
        rank=2,
        dtype=torch.float32,
        device="cpu",
    )
    attention_modules: dict[int, nn.Module] = {}
    for layer in range(30):
        attention = nn.Module()
        attention.q_proj = DenseGQAQKVOResidualProjection(
            nn.Linear(6, 8, bias=False), registry, layer_index=layer, role="q"
        )
        attention.k_proj = DenseGQAQKVOResidualProjection(
            nn.Linear(6, 4, bias=False), registry, layer_index=layer, role="k"
        )
        attention.v_proj = DenseGQAQKVOResidualProjection(
            nn.Linear(6, 4, bias=False), registry, layer_index=layer, role="v"
        )
        attention.o_proj = DenseGQAQKVOResidualProjection(
            nn.Linear(8, 6, bias=False), registry, layer_index=layer, role="o"
        )
        attention_modules[layer] = attention
    handle = SimpleNamespace(
        attention_modules=attention_modules,
        report=SimpleNamespace(
            projection_names={
                "q": "q_proj",
                "k": "k_proj",
                "v": "v_proj",
                "o": "o_proj",
            }
        ),
    )
    transaction = SimpleNamespace(qkvo_handle=handle)
    site_pairs = tuple((site.layer_index, site.physical_group) for site in sites)
    sketches = {
        (role, row): _FakeSketch()
        for role in ("q", "k", "v", "o")
        for row in range(len(sites))
    }
    collector = _QKVOHookCollector(
        transaction=transaction,
        geometry=geometry,
        sites=site_pairs,
        sketches=sketches,  # type: ignore[arg-type]
        max_rows=3,
        fingerprint_seed=0,
    )
    return collector, attention_modules, sketches


def test_hook_slices_once_cleans_up_and_does_not_mutate_projection() -> None:
    collector, attention_modules, sketches = _toy_hook_collector()
    before = {
        (layer, role): getattr(attention, f"{role}_proj").weight.detach().clone()
        for layer, attention in attention_modules.items()
        for role in ("q", "k", "v", "o")
    }
    collector.install()
    assert collector.active_hook_count == 120

    def replay(phase: str) -> str:
        context = _LossContext(
            "parent-greedy", "prompt-x", "0" * 64, 0.25, 0, 3
        )
        collector.begin_pass(phase, 1)  # type: ignore[arg-type]
        collector.prepare_context(context)
        collector.current = context
        total = torch.zeros(())
        for attention in attention_modules.values():
            hidden = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
            query = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
            total = total + attention.q_proj(hidden).sum()
            total = total + attention.k_proj(hidden).sum()
            total = total + attention.v_proj(hidden).sum()
            total = total + attention.o_proj(query).sum()
        collector.current = None
        total.backward()
        return collector.finish_pass()

    first_content = replay("pass1")
    second_content = replay("pass2")
    assert first_content == second_content

    first = sketches[("q", 0)].pass1[0]
    second = sketches[("q", 1)].pass1[0]
    assert first[0].shape == (3, 4)
    assert first[1].shape == (3, 6)
    assert second[0].shape == (3, 4)
    assert sketches[("q", 0)].identity_chains[1] != sketches[("q", 1)].identity_chains[1]
    assert sketches[("q", 0)].identity_chains[1] == sketches[("q", 0)].identity_chains[2]
    instrumentation = collector.hotpath_instrumentation
    assert instrumentation["wrapper_gradient_callback_count"] == 240
    assert instrumentation["callback_sync_evidence"] == "callback_source_audit"
    assert instrumentation["pass_boundary_device_to_host_transfer_call_count"] == 2
    assert instrumentation["fingerprint_records_hashed"] == 240
    assert instrumentation["fingerprint_values_per_wrapper"] == 44
    assert instrumentation["replay_authenticator_binds_every_element"] is True
    assert instrumentation["replay_authenticator_is_collision_free"] is False
    assert instrumentation["replay_authenticator_projection_lanes"] == 16
    assert instrumentation["replay_authenticator_coefficient_bytes"] == 1_536
    assert set(instrumentation["pass_full_element_replay_authenticator_sha256"]) == {
        "pass1",
        "pass2",
    }
    assert "callback_host_sync_count" not in instrumentation

    collector.remove()
    assert collector.active_hook_count == 0
    for attention in attention_modules.values():
        for role in ("q", "k", "v", "o"):
            assert not getattr(attention, f"{role}_proj")._forward_hooks
    assert all(
        torch.equal(getattr(attention_modules[layer], f"{role}_proj").weight, value)
        for (layer, role), value in before.items()
    )


def test_production_hotpath_has_bounded_transfers_and_no_sync_primitives() -> None:
    maximum_blocks_per_pass = 24 * (8 + 3)
    wrapper_callbacks = 2 * maximum_blocks_per_pass * 120
    assert wrapper_callbacks == 63_360
    assert wrapper_callbacks * 14 == 887_040
    audit = _callback_hotpath_source_audit()
    assert audit["kind"] == "source-audited-complete-python-callback-call-graph-v1"
    assert audit["target_count"] == len(audit["target_source_sha256"])
    assert audit["target_count"] >= 18
    assert audit["banned_host_sync_primitive_count"] == 0
    assert audit["runtime_cuda_profiler_used"] is False
    assert "internal to PyTorch kernels" in audit["limitation"]
    assert {
        "collector.forward_hook",
        "collector.record_wrapper_fingerprint",
        "audit.write_device_authenticator",
        "sketch.add_prevalidated_pass1",
        "sketch.add_prevalidated_pass2",
        "sketch.accumulate_pass1",
        "sketch.accumulate_pass2",
    }.issubset(audit["target_source_sha256"])
    source = inspect.getsource(_write_device_tensor_fingerprint)
    assert all(
        token not in source
        for token in (".item(", ".cpu(", ".numpy(", ".tolist(", "memoryview(")
    )


class _TinyLM(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(7, 7, generator=generator))
        self.forward_count = 0

    def forward(
        self, input_ids: torch.Tensor, **_kwargs: object
    ) -> dict[str, torch.Tensor]:
        self.forward_count += 1
        values = nn.functional.one_hot(input_ids, num_classes=7).to(torch.float32)
        return {"logits": values @ self.weight}


def test_behavior_loss_replay_is_identical_without_an_optimizer() -> None:
    parent = _TinyLM(1)
    child = _TinyLM(2)
    transaction = SimpleNamespace(model=child)
    domain = BehaviorDomainSuite(
        split_name="fit",
        prompts=(BehaviorPrompt((1, 2)),),
        artifact_sha256="a" * 64,
    )
    schedule = BehaviorDistillationSchedule(
        steps=1,
        learning_rate=1e-3,
        fit_horizon=8,
        repetition_excess_loss_weight=0.5,
    )
    plans = _build_replay_plans(
        transaction,  # type: ignore[arg-type]
        parent,
        domain,
        schedule,
        QKVOOutputJacobianAuditConfig(),
    )
    assert "parent_rows" not in _ReplayPlan.__dataclass_fields__
    assert all(
        not isinstance(getattr(plan, name), torch.Tensor)
        for plan in plans
        for name in _ReplayPlan.__dataclass_fields__
    )
    assert "parent_rows" not in inspect.getsource(_ReplayPlan)
    assert "_greedy_path" not in inspect.getsource(_build_replay_plans)
    assert ".parent_rows" not in inspect.getsource(_replay_loss_pass)
    assert "with torch.no_grad()" in inspect.getsource(_greedy_token_path)
    assert "with torch.no_grad()" in inspect.getsource(_replay_loss_pass)
    collector = SimpleNamespace(current=None)
    parent_after_planning = parent.forward_count
    staging: list[int] = []
    first, blocks, first_parent_recomputations = _replay_loss_pass(
        transaction=transaction,  # type: ignore[arg-type]
        parent_model=parent,
        plans=plans,
        schedule=schedule,
        collector=collector,  # type: ignore[arg-type]
        factors=(child.weight,),
        hash_staging_observer=staging.append,
    )
    assert parent.forward_count - parent_after_planning == blocks
    parent_after_first_pass = parent.forward_count
    second, second_blocks, second_parent_recomputations = _replay_loss_pass(
        transaction=transaction,  # type: ignore[arg-type]
        parent_model=parent,
        plans=plans,
        schedule=schedule,
        collector=collector,  # type: ignore[arg-type]
        factors=(child.weight,),
        hash_staging_observer=staging.append,
    )
    assert blocks == second_blocks > 0
    assert first_parent_recomputations == second_parent_recomputations == blocks
    assert parent.forward_count - parent_after_first_pass == blocks
    assert first == second
    assert staging and max(staging) <= 4 * 1024 * 1024
    assert child.weight.grad is None


def _fake_report(*, raw: float, normal: float, guided: float, random: float) -> object:
    return SimpleNamespace(
        raw_frobenius_sq_estimate=raw,
        normal_frobenius_sq_estimate=normal,
        guided_capture_energy=guided,
        random_capture_energy=random,
    )


def _role_decisions(*authorized: str) -> dict[str, dict[str, bool]]:
    return {
        role: {"passed": role in authorized} for role in ("q", "k", "v", "o")
    }


def test_energy_weighted_aggregation_and_global_gates() -> None:
    reports = [
        (role, _fake_report(raw=100.0, normal=40.0, guided=20.0, random=10.0))
        for role in ("q", "k", "v", "o")
    ]
    aggregate, per_role = _aggregate_reports(  # type: ignore[arg-type]
        reports, config=QKVOOutputJacobianAuditConfig()
    )
    assert aggregate["normal_fraction"] == pytest.approx(0.4)
    assert aggregate["guided_capture_fraction"] == pytest.approx(0.5)
    assert aggregate["guided_over_random_ratio"] == pytest.approx(2.0)
    assert aggregate["passed"] is True
    assert all(value["passed"] is True for value in per_role.values())

    rejected = [
        (role, _fake_report(raw=100, normal=10, guided=10, random=10))
        for role in ("q", "k", "v", "o")
    ]
    aggregate, _ = _aggregate_reports(  # type: ignore[arg-type]
        rejected, config=QKVOOutputJacobianAuditConfig()
    )
    assert aggregate["passed"] is False


def test_aggregate_pass_without_one_installable_role_is_rejected() -> None:
    reports = [
        *[
            (role, _fake_report(raw=100, normal=10, guided=10, random=1))
            for role in ("q", "k")
        ],
        *[
            (role, _fake_report(raw=100, normal=100, guided=10, random=10))
            for role in ("v", "o")
        ],
    ]
    aggregate, per_role = _aggregate_reports(  # type: ignore[arg-type]
        reports, config=QKVOOutputJacobianAuditConfig()
    )
    assert aggregate["passed"] is True
    assert all(value["passed"] is False for value in per_role.values())
    diagnostic_roles = _diagnostic_passed_roles(per_role)
    assert diagnostic_roles == []
    assert _audit_decision(
        aggregate,
        diagnostic_roles,
        replay_authenticator_match=True,
        mutation_safe=True,
        gradients_cleared=True,
        hooks_removed=True,
        transaction_live=True,
    ) is False


def test_wrong_topology_nonzero_factor_and_memory_fail_closed() -> None:
    with pytest.raises(QKVOOutputJacobianAuditTopologyError, match="transaction"):
        _validate_exact_topology(object())  # type: ignore[arg-type]

    registry = nn.Module()
    registry.rank = 2
    for role in ("q", "k", "v", "o"):
        setattr(registry, f"{role}_left", nn.Parameter(torch.randn(120, 4, 2)))
        setattr(registry, f"{role}_right", nn.Parameter(torch.randn(120, 2, 6)))
    registry.q_left.data.zero_()
    with pytest.raises(
        QKVOOutputJacobianAuditTopologyError, match="finite and nonzero"
    ):
        _validate_inherited_qkvo_factors(registry)

    tiny_limit = QKVOOutputJacobianAuditConfig(max_bytes=1)
    memory = _sketch_memory_plan(
        geometry=DenseGQAGeometry(30, 1536, 16, 4, 128),
        site_count=120,
        inherited_rank=8,
        config=tiny_limit,
        max_rows=16,
    )
    with pytest.raises(QKVOOutputJacobianAuditLimitError, match="before backward"):
        _require_memory_within_limit(memory, tiny_limit, phase="test")

    production = _sketch_memory_plan(
        geometry=DenseGQAGeometry(30, 1536, 16, 4, 128),
        site_count=120,
        inherited_rank=8,
        config=QKVOOutputJacobianAuditConfig(),
        max_rows=135,
    )
    assert production["deferred_fingerprint_buffer_upper_bound_bytes"] == 8_110_080
    assert production["deferred_fingerprint_host_staging_upper_bound_bytes"] == 8_110_080
    assert production["replay_authenticator_coefficient_bytes"] == 17_694_720
    assert production["tensor_hash_cpu_staging_limit_bytes"] == 4_194_304
    assert production["final_guided_cpu_pack_bytes"] == 19_660_800
    assert production["basis_payload_cpu_clone_bytes"] == 19_660_800
    assert production["replay_plan_cached_parent_logit_row_count"] == 0
    assert production["replay_plan_cached_parent_logit_bytes"] == 0
    assert production["replay_plan_cached_child_logit_row_count"] == 0
    assert production["replay_plan_cached_child_logit_bytes"] == 0
    assert production["replay_plan_cached_tensor_count"] == 0
    assert production["replay_plan_cached_tensor_bytes"] == 0
    assert production["token_path_scalar_transient_bytes"] == 8
    component_names = (
        "persistent_upper_bound_bytes",
        "largest_single_operation_bytes",
        "wrapper_hook_transient_bytes",
        "deferred_fingerprint_buffer_upper_bound_bytes",
        "deferred_fingerprint_host_staging_upper_bound_bytes",
        "replay_authenticator_coefficient_bytes",
        "tensor_hash_cpu_staging_limit_bytes",
        "final_guided_cpu_pack_bytes",
        "basis_payload_cpu_clone_bytes",
        "token_path_scalar_transient_bytes",
    )
    assert production["accounted_auxiliary_peak_bytes"] == sum(
        production[name] for name in component_names
    )
    assert production["accounted_auxiliary_peak_bytes"] == 214_001_640
    assert production["accounted_auxiliary_peak_bytes"] < 268_435_456
    assert production["memory_scope"]["bounded"] == (
        "audit-explicit-auxiliary-tensors-only"
    )
    assert "parent-and-child-model-weights" in production["memory_scope"]["excludes"]
    assert "caller-owned-authenticated-continuation-factor-payload" in production[
        "memory_scope"
    ]["excludes"]
    assert "standard-forward-backward-autograd-activations" in production[
        "memory_scope"
    ]["excludes"]


def test_globally_rejected_basis_pack_retains_diagnostics_but_authorizes_nothing() -> None:
    guided = {
        "q": torch.randn(2, 2, 3),
        "k": torch.randn(2, 2, 3),
        "v": torch.randn(2, 2, 3),
        "o": torch.randn(2, 2, 1),
    }
    named = {
        "q_orthogonal_right": guided["q"],
        "k_orthogonal_right": guided["k"],
        "v_orthogonal_right": guided["v"],
        "o_orthogonal_right": guided["o"],
    }
    manifest = [_tensor_manifest_row(name, value) for name, value in named.items()]
    report = QKVOOutputJacobianAuditReport(
        {
            "schema": QKVO_OUTPUT_JACOBIAN_AUDIT_SCHEMA,
            "passed": False,
            "report_sha256": "b" * 64,
            "factor_state_sha256_before": "c" * 64,
            "basis_manifest_sha256": _canonical_sha256(manifest),
            "diagnostic_passed_roles": ["q", "o"],
            "authorized_roles": [],
            "installable_roles": [],
            "per_role": _role_decisions("q", "o"),
            "site_indices": [[0, 0], [0, 1]],
            "site_indices_sha256": _canonical_sha256([[0, 0], [0, 1]]),
            "memory_accounting": {"basis_payload_cpu_clone_bytes": 160},
        }
    )
    pack = QKVOOutputJacobianAuditResult(guided, report).basis_pack_payload()
    assert pack["schema"] == GUIDED_QKVO_RIGHT_BASIS_SCHEMA
    assert pack["authorized_for_training"] is False
    assert pack["diagnostic_passed_roles"] == ["q", "o"]
    assert pack["authorized_roles"] == []
    assert pack["installable_roles"] == []
    assert pack["installer_must_ignore_unauthorized_role_tensors"] is True
    assert pack["contains_gradients"] is False
    assert pack["contains_activations"] is False
    assert tuple(pack["factor_tensors"]) == tuple(named)


def test_partial_role_authorization_keeps_all_tensors_but_limits_installers() -> None:
    guided = {
        "q": torch.randn(2, 2, 3),
        "k": torch.randn(2, 2, 3),
        "v": torch.randn(2, 2, 3),
        "o": torch.randn(2, 2, 1),
    }
    named = {_PACK_NAME: guided[role] for role, _PACK_NAME in (
        ("q", "q_orthogonal_right"),
        ("k", "k_orthogonal_right"),
        ("v", "v_orthogonal_right"),
        ("o", "o_orthogonal_right"),
    )}
    manifest = [_tensor_manifest_row(name, value) for name, value in named.items()]
    site_indices = [[0, 0], [0, 1]]
    report = QKVOOutputJacobianAuditReport(
        {
            "schema": QKVO_OUTPUT_JACOBIAN_AUDIT_SCHEMA,
            "passed": True,
            "report_sha256": "b" * 64,
            "factor_state_sha256_before": "c" * 64,
            "basis_manifest_sha256": _canonical_sha256(manifest),
            "diagnostic_passed_roles": ["q", "o"],
            "authorized_roles": ["q", "o"],
            "installable_roles": ["q", "o"],
            "per_role": _role_decisions("q", "o"),
            "site_indices": site_indices,
            "site_indices_sha256": _canonical_sha256(site_indices),
            "memory_accounting": {"basis_payload_cpu_clone_bytes": 160},
        }
    )
    pack = QKVOOutputJacobianAuditResult(guided, report).basis_pack_payload()
    assert pack["authorized_for_training"] is True
    assert pack["diagnostic_passed_roles"] == ["q", "o"]
    assert pack["authorized_roles"] == ["q", "o"]
    assert pack["installable_roles"] == ["q", "o"]
    assert tuple(pack["factor_tensors"]) == tuple(named)
    assert pack["all_four_role_tensors_retained_for_diagnostics"] is True

    failed_aggregate = dict(report.to_dict())
    failed_aggregate["passed"] = False
    failed_aggregate["authorized_roles"] = []
    failed_aggregate["installable_roles"] = []
    failed_pack = QKVOOutputJacobianAuditResult(
        guided, QKVOOutputJacobianAuditReport(failed_aggregate)
    ).basis_pack_payload()
    assert failed_pack["diagnostic_passed_roles"] == ["q", "o"]
    assert failed_pack["authorized_roles"] == []
    assert failed_pack["installable_roles"] == []
    assert failed_pack["authorized_for_training"] is False

    malformed = dict(report.to_dict())
    malformed["authorized_roles"] = ["o", "q"]
    malformed["installable_roles"] = ["o", "q"]
    with pytest.raises(QKVOOutputJacobianAuditError, match="ordered role"):
        QKVOOutputJacobianAuditResult(
            guided, QKVOOutputJacobianAuditReport(malformed)
        ).basis_pack_payload()


def test_v1_audit_report_cannot_publish_a_v2_basis_pack() -> None:
    guided = {
        "q": torch.randn(2, 2, 3),
        "k": torch.randn(2, 2, 3),
        "v": torch.randn(2, 2, 3),
        "o": torch.randn(2, 2, 1),
    }
    report = QKVOOutputJacobianAuditReport(
        {
            "schema": "neural-forge-qkvo-output-jacobian-audit-v1",
            "passed": False,
        }
    )
    with pytest.raises(QKVOOutputJacobianAuditError, match="v2 report"):
        QKVOOutputJacobianAuditResult(guided, report).basis_pack_payload()
