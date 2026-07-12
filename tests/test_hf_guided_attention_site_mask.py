from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

import pytest
import torch
from torch import nn

import neural_forge.hf_dense_gqa_qkvo_residual_adapter as adapter
from neural_forge.hf_dense_gqa_qkvo_residual_adapter import (
    DenseGQAGeometry,
    DenseGQASite,
    GuidedSparseDenseGQAQKVOResidualRegistry,
)
from neural_forge.hf_guided_attention_site_mask import (
    GradientSiteMaskGuard,
    GradientSiteMaskGuardError,
    GuidedAttentionSiteMaskAuthenticationError,
    ROLES,
    build_guided_attention_site_mask,
)
from neural_forge.hf_output_jacobian_sketch import OUTPUT_JACOBIAN_SKETCH_SCHEMA


EXPECTED_CURRENT_COUNTS = {"q": 36, "k": 34, "v": 57, "o": 67}
ROLE_AUTHORIZATION_POLICY = (
    "installers-must-ignore-all-tensors-for-roles-not-listed-in-"
    "authorized_roles"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _with_report_sha(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["report_sha256"] = _canonical_sha256(result)
    return result


def _energy_profile(target_count: int) -> list[float]:
    total = 1.25 * target_count - 0.1
    tail = (total - target_count) / (120 - target_count)
    assert 0.0 < tail < 1.0
    return [1.0] * target_count + [tail] * (120 - target_count)


def _sketch_report(
    energy: float,
    *,
    gate_passed: bool = True,
    normal_effective_rank: int = 8,
    block_replay: bool = True,
) -> dict[str, object]:
    return _with_report_sha(
        {
            "schema": OUTPUT_JACOBIAN_SKETCH_SCHEMA,
            "gate_passed": gate_passed,
            "normal_effective_rank": normal_effective_rank,
            "normal_frobenius_sq_estimate": energy,
            "block_identity_replay_authenticated": block_replay,
        }
    )


def _audit_report(
    *,
    energies: Mapping[str, list[float]] | None = None,
    sketch_overrides: Mapping[tuple[str, int], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if energies is None:
        energies = {
            role: _energy_profile(EXPECTED_CURRENT_COUNTS[role]) for role in ROLES
        }
    overrides = {} if sketch_overrides is None else dict(sketch_overrides)
    sites = [[layer, group] for layer in range(30) for group in range(4)]
    site_reports: list[dict[str, object]] = []
    for role in ROLES:
        assert len(energies[role]) == 120
        for row, (layer, group) in enumerate(sites):
            sketch_options = {
                "energy": energies[role][row],
                "gate_passed": True,
                "normal_effective_rank": 8,
                "block_replay": True,
            }
            sketch_options.update(overrides.get((role, row), {}))
            site_reports.append(
                {
                    "role": role,
                    "site_row": row,
                    "layer": layer,
                    "physical_group": group,
                    "inherited_right_median_row_norm": 1.0,
                    "sketch_report": _sketch_report(**sketch_options),
                }
            )
    config = {
        "prompt_count": 24,
        "horizon": 8,
        "lanes": ["parent-greedy", "child-greedy"],
        "rank": 8,
        "sketch_width": 16,
        "seed": 0,
        "min_normal_fraction": 0.2,
        "min_guided_over_random_ratio": 1.5,
        "max_bytes": 256 * 1024 * 1024,
    }
    paired = {
        "pass1_replay_sha256": "1" * 64,
        "pass2_replay_sha256": "1" * 64,
        "pass1_full_element_replay_authenticator_sha256": "2" * 64,
        "pass2_full_element_replay_authenticator_sha256": "2" * 64,
        "native_state_sha256_before": "3" * 64,
        "native_state_sha256_after": "3" * 64,
        "factor_state_sha256_before": "4" * 64,
        "factor_state_sha256_after": "4" * 64,
        "inherited_qkvo_state_sha256_before": "5" * 64,
        "inherited_qkvo_state_sha256_after": "5" * 64,
        "factor_parameter_versions_sha256_before": "6" * 64,
        "factor_parameter_versions_sha256_after": "6" * 64,
    }
    return _with_report_sha(
        {
            "schema": adapter.QKVO_OUTPUT_JACOBIAN_AUDIT_V2_SCHEMA,
            "passed": True,
            "config": config,
            "config_sha256": _canonical_sha256(config),
            "site_count": 120,
            "role_count": 4,
            "sketch_count": 480,
            "site_indices": sites,
            "site_indices_sha256": _canonical_sha256(sites),
            "probabilistic_replay_authenticator_match": True,
            "replay_authentication_binds_every_element": True,
            "replay_authentication_is_collision_free": False,
            "replay_authentication_projection_lanes": 16,
            "factor_gradients_cleared": True,
            "hooks_removed": True,
            "active_hook_count_after": 0,
            "transaction_live_after": True,
            "rollback_compatible_after": True,
            "optimizer_created": False,
            "optimizer_used": False,
            "optimizer_steps": 0,
            "new_candidate_accessed": False,
            "sealed_accessed": False,
            "aggregate": {"passed": True},
            "per_role": {role: {"passed": True} for role in ROLES},
            "diagnostic_passed_roles": list(ROLES),
            "authorized_roles": list(ROLES),
            "installable_roles": list(ROLES),
            "role_authorization_policy": ROLE_AUTHORIZATION_POLICY,
            "installer_must_ignore_unauthorized_role_tensors": True,
            "all_four_role_tensors_retained_for_diagnostics": True,
            "site_reports": site_reports,
            **paired,
        }
    )


def _rehash_report(report: dict[str, object]) -> str:
    report.pop("report_sha256", None)
    report["report_sha256"] = _canonical_sha256(report)
    return str(report["report_sha256"])


def _mask(report: dict[str, object]):
    return build_guided_attention_site_mask(
        report, expected_report_sha256=str(report["report_sha256"])
    )


def _registry(mask) -> GuidedSparseDenseGQAQKVOResidualRegistry:
    geometry = DenseGQAGeometry(
        logical_layers=30,
        hidden_size=8,
        query_heads=4,
        key_value_heads=4,
        head_dim=2,
    )
    sites = tuple(
        DenseGQASite(layer, group) for layer in range(30) for group in range(4)
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(910)
    right_rows = {
        "q": torch.randn(120, 8, 8, generator=generator),
        "k": torch.randn(120, 8, 8, generator=generator),
        "v": torch.randn(120, 8, 8, generator=generator),
        "o": torch.randn(120, 8, 2, generator=generator),
    }
    hashes = {role: adapter._tensor_sha256(right_rows[role]) for role in ROLES}
    return GuidedSparseDenseGQAQKVOResidualRegistry(
        sites,
        geometry=geometry,
        rank=8,
        guided_right_rows=right_rows,
        source_basis_tensor_sha256s=hashes,
        audit_report_sha256=mask.audit_report_sha256,
        basis_envelope_file_sha256="a" * 64,
        basis_payload_sha256="b" * 64,
        basis_manifest_sha256="c" * 64,
        dtype=torch.float32,
        device="cpu",
    )


def _hook_count(registry: nn.Module) -> int:
    return sum(len(value._backward_hooks or {}) for value in registry.parameters())


def test_current_audit_energy_fixture_selects_exact_role_counts() -> None:
    report = _audit_report()
    mask = _mask(report)

    assert {
        role: mask.for_role(role).selected_site_count for role in ROLES
    } == EXPECTED_CURRENT_COUNTS
    assert all(
        value.selected_energy_coverage >= 0.80 for value in mask.role_masks
    )
    assert mask.audit_report_sha256 == report["report_sha256"]
    assert mask.as_dict()["mask_contract_sha256"] == mask.mask_contract_sha256
    with pytest.raises(AttributeError):
        mask.audit_report_sha256 = "0" * 64  # type: ignore[misc]


def test_energy_ties_use_site_row_and_prefix_is_minimal() -> None:
    report = _audit_report(energies={role: [1.0] * 120 for role in ROLES})
    mask = _mask(report)

    for role in ROLES:
        contract = mask.for_role(role)
        assert contract.selected_site_count == 96
        assert contract.selected_site_rows == tuple(range(96))
        assert contract.selected_energy_coverage == pytest.approx(0.8)


def test_eligibility_requires_gate_rank_energy_and_block_replay() -> None:
    overrides = {
        ("q", 0): {"gate_passed": False},
        ("q", 1): {"normal_effective_rank": 7},
        ("q", 2): {"energy": 0.0},
        ("q", 3): {"block_replay": False},
    }
    report = _audit_report(
        energies={role: [1.0] * 120 for role in ROLES},
        sketch_overrides=overrides,
    )
    mask = _mask(report)
    q_mask = mask.for_role("q")

    assert q_mask.eligible_site_count == 116
    assert not {0, 1, 2, 3} & {
        row for row, _energy in q_mask.eligible_site_energies
    }


def test_outer_nested_order_site_and_role_tamper_fail_closed() -> None:
    report = _audit_report()

    outer = copy.deepcopy(report)
    outer["passed"] = False
    with pytest.raises(
        GuidedAttentionSiteMaskAuthenticationError, match="canonical SHA"
    ):
        build_guided_attention_site_mask(
            outer, expected_report_sha256=str(report["report_sha256"])
        )

    nested = copy.deepcopy(report)
    nested_sketch = nested["site_reports"][0]["sketch_report"]  # type: ignore[index]
    nested_sketch["normal_frobenius_sq_estimate"] = 99.0  # type: ignore[index]
    nested_sha = _rehash_report(nested)
    with pytest.raises(
        GuidedAttentionSiteMaskAuthenticationError, match="sketch_report canonical"
    ):
        build_guided_attention_site_mask(nested, expected_report_sha256=nested_sha)

    reordered = copy.deepcopy(report)
    reordered["site_reports"][0], reordered["site_reports"][1] = (  # type: ignore[index]
        reordered["site_reports"][1],  # type: ignore[index]
        reordered["site_reports"][0],  # type: ignore[index]
    )
    reordered_sha = _rehash_report(reordered)
    with pytest.raises(
        GuidedAttentionSiteMaskAuthenticationError, match="exact role/site-index order"
    ):
        build_guided_attention_site_mask(
            reordered, expected_report_sha256=reordered_sha
        )

    site_tamper = copy.deepcopy(report)
    site_tamper["site_indices_sha256"] = "f" * 64
    site_sha = _rehash_report(site_tamper)
    with pytest.raises(GuidedAttentionSiteMaskAuthenticationError, match="site-index"):
        build_guided_attention_site_mask(
            site_tamper, expected_report_sha256=site_sha
        )

    role_tamper = copy.deepcopy(report)
    role_tamper["authorized_roles"] = ["q", "k", "v"]
    role_sha = _rehash_report(role_tamper)
    with pytest.raises(
        GuidedAttentionSiteMaskAuthenticationError, match="role authorization"
    ):
        build_guided_attention_site_mask(
            role_tamper, expected_report_sha256=role_sha
        )


def test_gradient_hooks_mask_unselected_rows_and_optimizer_preserves_sources() -> None:
    mask = _mask(_audit_report())
    registry = _registry(mask)
    optimizer = torch.optim.SGD(registry.parameters(), lr=0.05)
    right_before = {
        role: getattr(registry, f"{role}_right").detach().clone() for role in ROLES
    }

    with GradientSiteMaskGuard(registry, mask) as guard:
        assert guard.active_hook_count == 8
        loss = sum(value.sum() for value in registry.parameters())
        loss.backward()
        for role in ROLES:
            selected = set(mask.for_role(role).selected_site_rows)
            unselected = [row for row in range(120) if row not in selected]
            selected_rows = sorted(selected)
            for side in ("left", "right"):
                gradient = getattr(registry, f"{role}_{side}").grad
                assert gradient is not None
                assert torch.count_nonzero(gradient[unselected]).item() == 0
                assert torch.all(gradient[selected_rows] == 1)
        optimizer.step()
        guard.verify_step()
        for role in ROLES:
            selected = set(mask.for_role(role).selected_site_rows)
            unselected = [row for row in range(120) if row not in selected]
            selected_rows = sorted(selected)
            left = getattr(registry, f"{role}_left")
            right = getattr(registry, f"{role}_right")
            assert torch.count_nonzero(left[unselected]).item() == 0
            assert torch.equal(right[unselected], right_before[role][unselected])
            assert torch.count_nonzero(left[selected_rows]).item() > 0
            assert not torch.equal(right[selected_rows], right_before[role][selected_rows])

    assert _hook_count(registry) == 0


@pytest.mark.parametrize("side", ("left", "right"))
def test_unselected_direct_mutation_is_detected_and_hooks_are_removed(side: str) -> None:
    mask = _mask(_audit_report())
    registry = _registry(mask)
    role = "q"
    selected = set(mask.for_role(role).selected_site_rows)
    row = next(value for value in range(120) if value not in selected)

    with pytest.raises(GradientSiteMaskGuardError, match="Unselected Q"):
        with GradientSiteMaskGuard(registry, mask):
            with torch.no_grad():
                getattr(registry, f"{role}_{side}")[row].add_(1.0)
    assert _hook_count(registry) == 0


def test_hook_cleanup_preserves_user_baseexception() -> None:
    mask = _mask(_audit_report())
    registry = _registry(mask)

    with pytest.raises(KeyboardInterrupt, match="stop"):
        with GradientSiteMaskGuard(registry, mask):
            assert _hook_count(registry) == 8
            raise KeyboardInterrupt("stop")
    assert _hook_count(registry) == 0


def test_registry_site_index_tamper_fails_before_hooks() -> None:
    mask = _mask(_audit_report())
    registry = _registry(mask)
    with torch.no_grad():
        registry.site_indices[0, 0] = 1

    with pytest.raises(GradientSiteMaskGuardError, match="topology/audit"):
        with GradientSiteMaskGuard(registry, mask):
            pass
    assert _hook_count(registry) == 0
