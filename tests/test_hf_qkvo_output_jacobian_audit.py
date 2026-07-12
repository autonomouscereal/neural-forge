from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from neural_forge.hf_dense_gqa_qkvo_residual_adapter import (
    DenseGQAGeometry,
    DenseGQASite,
    DenseGQAQKVOResidualProjection,
    SparseDenseGQAQKVOResidualRegistry,
)
from neural_forge.hf_qkvo_output_jacobian_audit import (
    GUIDED_QKVO_RIGHT_BASIS_SCHEMA,
    QKVOOutputJacobianAuditConfig,
    QKVOOutputJacobianAuditError,
    QKVOOutputJacobianAuditLimitError,
    QKVOOutputJacobianAuditReport,
    QKVOOutputJacobianAuditReplayError,
    QKVOOutputJacobianAuditResult,
    QKVOOutputJacobianAuditTopologyError,
    _aggregate_reports,
    _build_replay_plans,
    _canonical_sha256,
    _compact_numeric_fingerprint,
    _LossContext,
    _QKVOHookCollector,
    _replay_loss_pass,
    _require_memory_within_limit,
    _sketch_memory_plan,
    _slice_qkvo_blocks,
    _tensor_manifest_row,
    _tensor_sha256,
    _validate_exact_topology,
    _validate_inherited_qkvo_factors,
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


def test_compact_fingerprint_and_tensor_hash_are_deterministic_and_finite() -> None:
    inputs = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    delta = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    first = _compact_numeric_fingerprint(inputs, delta)
    assert first == _compact_numeric_fingerprint(inputs.clone(), delta.clone())
    changed = delta.clone()
    changed[2, 1] += 1.0
    assert first != _compact_numeric_fingerprint(inputs, changed)

    strided = torch.arange(20, dtype=torch.float32)[::2]
    assert _tensor_sha256(strided) == _tensor_sha256(strided.clone())
    poisoned = delta.clone()
    poisoned[0, 0] = float("nan")
    with pytest.raises(QKVOOutputJacobianAuditReplayError, match="non-finite"):
        _compact_numeric_fingerprint(inputs, poisoned)


class _FakeSketch:
    def __init__(self) -> None:
        self.pass1: list[tuple[torch.Tensor, torch.Tensor, object]] = []

    def add_pass1(
        self,
        delta: torch.Tensor,
        inputs: torch.Tensor,
        *,
        block_identity: object,
    ) -> None:
        self.pass1.append(
            (delta.detach().clone(), inputs.detach().clone(), block_identity)
        )

    def add_pass2(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("pass two was not selected")


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
    collector.current = _LossContext("parent-greedy", "prompt-x", "0" * 64, 0.25, 0)
    values = torch.randn(1, 3, 6, requires_grad=True)
    output = attention_modules[0].q_proj(values)
    collector.current = None
    output.sum().backward()

    first = sketches[("q", 0)].pass1[0]
    second = sketches[("q", 1)].pass1[0]
    assert first[0].shape == (3, 4)
    assert first[1].shape == (3, 6)
    assert second[0].shape == (3, 4)
    assert first[2].numeric_fingerprint_sha256 == second[2].numeric_fingerprint_sha256
    assert first[2].semantic_identity != second[2].semantic_identity

    collector.remove()
    assert collector.active_hook_count == 0
    for attention in attention_modules.values():
        for role in ("q", "k", "v", "o"):
            assert not getattr(attention, f"{role}_proj")._forward_hooks
    assert all(
        torch.equal(getattr(attention_modules[layer], f"{role}_proj").weight, value)
        for (layer, role), value in before.items()
    )


class _TinyLM(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(7, 7, generator=generator))

    def forward(
        self, input_ids: torch.Tensor, **_kwargs: object
    ) -> dict[str, torch.Tensor]:
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
    collector = SimpleNamespace(current=None)
    first, blocks = _replay_loss_pass(
        transaction=transaction,  # type: ignore[arg-type]
        parent_model=parent,
        plans=plans,
        schedule=schedule,
        collector=collector,  # type: ignore[arg-type]
        factors=(child.weight,),
    )
    second, second_blocks = _replay_loss_pass(
        transaction=transaction,  # type: ignore[arg-type]
        parent_model=parent,
        plans=plans,
        schedule=schedule,
        collector=collector,  # type: ignore[arg-type]
        factors=(child.weight,),
    )
    assert blocks == second_blocks > 0
    assert first == second
    assert child.weight.grad is None


def _fake_report(*, raw: float, normal: float, guided: float, random: float) -> object:
    return SimpleNamespace(
        raw_frobenius_sq_estimate=raw,
        normal_frobenius_sq_estimate=normal,
        guided_capture_energy=guided,
        random_capture_energy=random,
    )


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


def test_rejected_basis_pack_is_diagnostic_and_never_authorized() -> None:
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
            "passed": False,
            "report_sha256": "b" * 64,
            "factor_state_sha256_before": "c" * 64,
            "basis_manifest_sha256": _canonical_sha256(manifest),
        }
    )
    pack = QKVOOutputJacobianAuditResult(guided, report).basis_pack_payload()
    assert pack["schema"] == GUIDED_QKVO_RIGHT_BASIS_SCHEMA
    assert pack["authorized_for_training"] is False
    assert pack["contains_gradients"] is False
    assert pack["contains_activations"] is False
    assert tuple(pack["factor_tensors"]) == tuple(named)
