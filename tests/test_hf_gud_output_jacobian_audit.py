from __future__ import annotations

import hashlib
import json

import pytest
import torch

from neural_forge.hf_gud_output_jacobian_audit import (
    GUD_CANDIDATE_BASIS_SCHEMA,
    GUD_OUTPUT_JACOBIAN_AUDIT_SCHEMA,
    GUD_ROLE_ORDER,
    HARD_MAX_AUXILIARY_BYTES,
    GUDJacobianBlock,
    GUDOutputJacobianAuditConfig,
    GUDOutputJacobianAuditFiniteError,
    GUDOutputJacobianAuditLimitError,
    GUDOutputJacobianAuditShapeError,
    GUDRoleSpec,
    GUDSiteAuditInput,
    GUDStageGeometry,
    audit_gud_site_basis,
    audit_gud_stage_bases,
    ordered_gud_role_specs,
    plan_gud_auxiliary_memory,
    role_local_input_vjp,
    validate_qwen_30_layer_stage_geometry,
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _empty_factors(
    output_width: int,
    input_width: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty((output_width, 0), dtype=dtype),
        torch.empty((0, input_width), dtype=dtype),
    )


def _config(**overrides: object) -> GUDOutputJacobianAuditConfig:
    values: dict[str, object] = {
        "rank": 1,
        "sketch_width": 4,
        "balanced_envelope_width": 2,
        "seed": 73,
        "max_blocks_per_site": 8,
        "max_rows_per_site": 256,
        "max_bytes": 64 * 1024 * 1024,
    }
    values.update(overrides)
    return GUDOutputJacobianAuditConfig(**values)  # type: ignore[arg-type]


def _low_variance_site() -> tuple[GUDSiteAuditInput, GUDRoleSpec, torch.Tensor]:
    rows = 32
    width = 16
    generator = torch.Generator(device="cpu").manual_seed(5)
    delta = (
        torch.where(
            torch.arange(rows) % 2 == 0,
            1.0,
            -1.0,
        )
        .to(torch.float64)
        .reshape(-1, 1)
    )
    inputs = 10.0 * torch.randn((rows, width), generator=generator, dtype=torch.float64)
    # Remove every nuisance column's correlation with D.  The last coordinate
    # is four orders of magnitude lower variance than the nuisance block but
    # is the only direction that can affect this local output/loss.
    inputs -= delta @ ((delta.T @ inputs) / (delta.T @ delta))
    inputs[:, -1] = 0.01 * delta[:, 0]
    base_weight = torch.zeros((1, width), dtype=torch.float64)
    base_weight[0, -1] = 1.0
    left, right = _empty_factors(1, width)
    site = GUDSiteAuditInput(
        layer_index=0,
        role="gate",
        base_weight=base_weight,
        inherited_left=left,
        inherited_right=right,
        blocks=(GUDJacobianBlock(delta, inputs, "fit:block:0"),),
    )
    return site, GUDRoleSpec("gate", width, 1, "hidden", "gate"), delta.T @ inputs


def test_ordered_role_specs_and_exact_stage_geometry_are_explicit() -> None:
    specs = ordered_gud_role_specs(1536, 5120)

    assert tuple(spec.role for spec in specs) == GUD_ROLE_ORDER
    assert [(spec.output_width, spec.input_width) for spec in specs] == [
        (5120, 1536),
        (5120, 1536),
        (1536, 5120),
    ]
    geometry = GUDStageGeometry(30, 1536, 5120)
    assert validate_qwen_30_layer_stage_geometry(geometry) == specs
    assert geometry.to_dict()["ordered_role_specs"] == [
        spec.to_dict() for spec in specs
    ]

    with pytest.raises(GUDOutputJacobianAuditShapeError, match="30 layers"):
        validate_qwen_30_layer_stage_geometry(GUDStageGeometry(29, 1536, 5120))
    with pytest.raises(GUDOutputJacobianAuditShapeError, match="hidden size 1536"):
        validate_qwen_30_layer_stage_geometry(GUDStageGeometry(30, 1024, 5120))


def test_role_local_input_vjp_matches_autograd_exactly() -> None:
    generator = torch.Generator(device="cpu").manual_seed(19)
    rows, inputs, outputs, latent = 7, 5, 4, 2
    x = torch.randn(
        (rows, inputs), generator=generator, dtype=torch.float64, requires_grad=True
    )
    weight = torch.randn((outputs, inputs), generator=generator, dtype=torch.float64)
    left = torch.randn((outputs, latent), generator=generator, dtype=torch.float64)
    right = torch.randn((latent, inputs), generator=generator, dtype=torch.float64)
    delta = torch.randn((rows, outputs), generator=generator, dtype=torch.float64)
    local_output = x @ weight.T + (x @ right.T) @ left.T
    (local_output * delta).sum().backward()
    expected = x.grad.detach().clone()

    actual = role_local_input_vjp(delta, weight, left, right)

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        actual,
        delta @ weight + (delta @ left) @ right,
        atol=1e-12,
        rtol=1e-12,
    )
    assert actual.grad_fn is None
    assert x.grad is not None


def test_guided_candidates_capture_output_relevant_low_variance_direction() -> None:
    site, role_spec, jacobian = _low_variance_site()

    result = audit_gud_site_basis(site, role_spec, _config())

    candidates = result.candidates()
    capture = {
        name: float(
            torch.linalg.vector_norm(jacobian @ rows.to(torch.float64).T).square()
        )
        for name, rows in candidates.items()
    }
    assert capture["output-jacobian"] > 0.999 * float(jacobian.square().sum())
    assert capture["balanced"] > 0.999 * float(jacobian.square().sum())
    assert capture["procrustes-hybrid"] > 0.999 * float(jacobian.square().sum())
    assert capture["output-jacobian"] > 8.0 * capture["random-control"]
    assert capture["balanced"] > 8.0 * capture["random-control"]
    assert result.output_jacobian_right_rows[0, -1] == pytest.approx(1.0)
    assert result.balanced_right_rows[0, -1] == pytest.approx(1.0)
    assert result.report.payload["internal_weight_matching_used"] is False
    assert result.report.payload["shared_input_hook_used"] is False


def test_all_candidates_exclude_inherited_right_row_space() -> None:
    generator = torch.Generator(device="cpu").manual_seed(31)
    rows, input_width, output_width = 24, 6, 3
    inputs = torch.randn((rows, input_width), generator=generator, dtype=torch.float64)
    delta = torch.randn((rows, output_width), generator=generator, dtype=torch.float64)
    weight = torch.randn(
        (output_width, input_width), generator=generator, dtype=torch.float64
    )
    inherited_left = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
    inherited_right = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    site = GUDSiteAuditInput(
        0,
        "down",
        weight,
        inherited_left,
        inherited_right,
        (GUDJacobianBlock(delta, inputs, "fit:block:0"),),
    )
    result = audit_gud_site_basis(
        site,
        GUDRoleSpec("down", input_width, output_width, "post", "residual"),
        _config(),
    )

    for candidate in result.candidates().values():
        torch.testing.assert_close(
            candidate @ candidate.T,
            torch.eye(1),
            atol=1e-6,
            rtol=0.0,
        )
        assert (
            float((candidate @ inherited_right.to(torch.float32).T).abs().max()) < 1e-6
        )
    assert result.report.payload["occupied_inherited_right_row_rank"] == 1
    assert (
        result.report.payload[
            "all_candidates_projected_out_of_inherited_right_row_space"
        ]
        is True
    )


def test_determinism_and_separable_authenticated_manifests() -> None:
    site, role_spec, _jacobian = _low_variance_site()
    config = _config()

    first = audit_gud_site_basis(site, role_spec, config)
    second = audit_gud_site_basis(site, role_spec, config)

    assert first.report.to_dict() == second.report.to_dict()
    assert first.report.report_sha256 == second.report.report_sha256
    for kind in first.candidates():
        assert torch.equal(first.candidates()[kind], second.candidates()[kind])
    report = first.report.to_dict()
    assert report["schema"] == GUD_OUTPUT_JACOBIAN_AUDIT_SCHEMA
    assert report["candidate_schema"] == GUD_CANDIDATE_BASIS_SCHEMA
    assert report["report_sha256"] == _canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    manifest = report["input_manifest"]
    assert manifest["input_manifest_sha256"] == _canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "input_manifest_sha256"
        }
    )
    block = manifest["blocks"][0]
    assert block["output_delta"]["sha256"] != block["projection_input"]["sha256"]
    assert block["output_delta"]["every_element_bound"] is True
    assert block["projection_input"]["every_element_bound"] is True
    assert block["role_local_input_vjp"]["every_element_bound"] is True
    assert manifest["shared_projection_fingerprint_used"] is False
    assert manifest["full_lanes_rows_width_table_used"] is False
    assert report["determinism_verified_by_exact_repeat"] is True


def test_memory_plan_is_explicit_bounded_and_fails_before_work() -> None:
    site, role_spec, _jacobian = _low_variance_site()
    geometry = GUDStageGeometry(1, role_spec.input_width, role_spec.output_width)
    plan = plan_gud_auxiliary_memory(geometry, (site,), _config())

    assert plan["accounted_auxiliary_peak_bytes"] <= plan["max_auxiliary_bytes"]
    assert plan["max_auxiliary_bytes"] <= HARD_MAX_AUXILIARY_BYTES
    assert plan["within_limit"] is True
    assert plan["balanced_centered"] is False
    assert plan["full_width_balanced_projector_bytes"] == 0
    assert plan["full_lanes_rows_width_fingerprint_table_bytes"] == 0
    assert plan["memory_plan_sha256"] == _canonical_sha256(
        {key: value for key, value in plan.items() if key != "memory_plan_sha256"}
    )
    assert all(
        row["full_width_balanced_projector_bytes"] == 0
        and row["full_lanes_rows_width_fingerprint_table_bytes"] == 0
        for row in plan["site_working_plans"]
    )

    with pytest.raises(GUDOutputJacobianAuditLimitError, match="256 MiB"):
        _config(max_bytes=HARD_MAX_AUXILIARY_BYTES + 1)
    with pytest.raises(GUDOutputJacobianAuditLimitError, match="exceed"):
        plan_gud_auxiliary_memory(geometry, (site,), _config(max_bytes=1024))


def test_stage_requires_complete_layer_major_gate_up_down_order() -> None:
    site, _role_spec, _jacobian = _low_variance_site()
    geometry = GUDStageGeometry(1, 16, 1)
    wrong = (
        site,
        GUDSiteAuditInput(
            0,
            "down",
            site.base_weight,
            site.inherited_left,
            site.inherited_right,
            site.blocks,
        ),
    )

    with pytest.raises(GUDOutputJacobianAuditShapeError, match="Gate, Up, Down"):
        audit_gud_stage_bases(geometry, wrong, _config())


def test_nonfinite_source_fails_closed() -> None:
    site, role_spec, _jacobian = _low_variance_site()
    bad_inputs = site.blocks[0].projection_input.clone()
    bad_inputs[0, 0] = float("nan")
    bad = GUDSiteAuditInput(
        site.layer_index,
        site.role,
        site.base_weight,
        site.inherited_left,
        site.inherited_right,
        (
            GUDJacobianBlock(
                site.blocks[0].output_delta,
                bad_inputs,
                site.blocks[0].semantic_identity,
            ),
        ),
    )

    with pytest.raises(GUDOutputJacobianAuditFiniteError, match="non-finite"):
        audit_gud_site_basis(bad, role_spec, _config())
