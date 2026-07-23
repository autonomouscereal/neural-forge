from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_global_residual_closure import (  # noqa: E402
    Lfm2GlobalResidualBoundaryBundle,
    Lfm2GlobalResidualClosureAuthenticationError,
    Lfm2GlobalResidualClosureOperator,
    Lfm2GlobalResidualClosurePolicy,
    Lfm2GlobalResidualLinearizationKernel,
    Lfm2GlobalResidualWriterSite,
    Lfm2GlobalResidualWriterTopology,
    adjoint_lfm2_global_residual_writer_block,
    apply_lfm2_global_residual_writer_block,
    audit_lfm2_global_residual_closure_adjoint,
    audit_lfm2_global_residual_closure_causality,
    build_lfm2_global_residual_basis,
    estimate_lfm2_global_residual_streaming_memory,
    fingerprint_lfm2_global_residual_tensor,
    materialize_lfm2_global_residual_corrected_writers,
    seal_lfm2_global_residual_boundary,
    seal_lfm2_global_residual_linearization_receipt,
    solve_lfm2_global_residual_closure,
    verify_lfm2_global_residual_closure_report,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _hashes(values):
    return {
        name: fingerprint_lfm2_global_residual_tensor(value)
        for name, value in values.items()
    }


def _topology(*, hidden: int = 4, head_dimension: int = 2):
    return Lfm2GlobalResidualWriterTopology(
        hidden_size=hidden,
        head_dimension=head_dimension,
        sites=(
            Lfm2GlobalResidualWriterSite(
                site_id="layer.000.attention.out_proj",
                layer_index=0,
                causal_order=0,
                writer_kind="attention_output",
                input_width=3,
            ),
            Lfm2GlobalResidualWriterSite(
                site_id="layer.001.mlp.w2",
                layer_index=1,
                causal_order=2,
                writer_kind="mlp_down",
                input_width=2,
            ),
        ),
    )


def _basis(topology, *, seed: int = 712):
    generator = torch.Generator().manual_seed(seed)
    hidden = topology.hidden_size
    defects = {
        "layer.000.post_operator": torch.randn(
            (7, hidden), generator=generator, dtype=torch.float64
        ),
        "layer.001.post_layer": torch.randn(
            (5, hidden), generator=generator, dtype=torch.float64
        ),
    }
    writers = {
        site.site_id: torch.randn(
            (hidden, site.input_width),
            generator=generator,
            dtype=torch.float64,
        )
        for site in topology.sites
    }
    terminal = {
        "winner_minus_competitor": torch.randn(
            (9, hidden), generator=generator, dtype=torch.float64
        )
    }
    return build_lfm2_global_residual_basis(
        topology,
        mapped_fit_residual_defects=defects,
        mapped_fit_residual_defects_sha256=_hashes(defects),
        source_writer_weights=writers,
        source_writer_weights_sha256=_hashes(writers),
        terminal_tied_table_directions=terminal,
        terminal_tied_table_directions_sha256=_hashes(terminal),
    )


class _TwoSiteKernel(Lfm2GlobalResidualLinearizationKernel):
    __slots__ = ()

    def _jvp_chunk_tensor_only(
        self,
        state,
        coefficient_blocks,
        output_basis,
        *,
        boundary_index,
        row_start,
        row_stop,
    ):
        producer0 = state["producer0"][row_start:row_stop].to(
            device=output_basis.device
        )
        effect0 = apply_lfm2_global_residual_writer_block(
            producer0,
            output_basis,
            coefficient_blocks[0],
        )
        if boundary_index == 0:
            return effect0
        producer1 = state["producer1"][row_start:row_stop].to(
            device=output_basis.device
        )
        transition = state["transition"].to(device=output_basis.device)
        return effect0 @ transition.transpose(0, 1) + (
            apply_lfm2_global_residual_writer_block(
                producer1,
                output_basis,
                coefficient_blocks[1],
            )
        )

    def _vjp_chunk_tensor_only(
        self,
        state,
        boundary_cotangent,
        output_basis,
        *,
        boundary_index,
        row_start,
        row_stop,
    ):
        producer0 = state["producer0"][row_start:row_stop].to(
            device=output_basis.device
        )
        if boundary_index == 0:
            cotangent0 = boundary_cotangent
            second = torch.zeros(
                (output_basis.shape[1], self.topology.sites[1].input_width),
                device=output_basis.device,
                dtype=output_basis.dtype,
            )
        else:
            transition = state["transition"].to(device=output_basis.device)
            cotangent0 = boundary_cotangent @ transition
            producer1 = state["producer1"][row_start:row_stop].to(
                device=output_basis.device
            )
            second = adjoint_lfm2_global_residual_writer_block(
                producer1,
                output_basis,
                boundary_cotangent,
            )
        first = adjoint_lfm2_global_residual_writer_block(
            producer0,
            output_basis,
            cotangent0,
        )
        return first, second


class _BundleTwoSiteKernel(_TwoSiteKernel):
    __slots__ = ()

    def _jvp_bundle_tensor_only(
        self,
        state,
        coefficient_blocks,
        output_basis,
        *,
        maximum_workspace_bytes,
    ):
        del maximum_workspace_bytes
        outputs = []
        for boundary_index, boundary in enumerate(self.boundaries.boundaries):
            rows = boundary.defect.numel() // self.topology.hidden_size
            outputs.append(
                self._jvp_chunk_tensor_only(
                    state,
                    coefficient_blocks,
                    output_basis,
                    boundary_index=boundary_index,
                    row_start=0,
                    row_stop=rows,
                )
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
                .clone()
            )
        return tuple(outputs)

    def _vjp_bundle_tensor_only(
        self,
        state,
        boundary_cotangents_cpu,
        output_basis,
        *,
        boundary_row_scales_cpu,
        maximum_workspace_bytes,
    ):
        del maximum_workspace_bytes
        accumulated = [
            torch.zeros(
                (output_basis.shape[1], site.input_width),
                device=output_basis.device,
                dtype=output_basis.dtype,
            )
            for site in self.topology.sites
        ]
        for boundary_index, cotangent_cpu in enumerate(boundary_cotangents_cpu):
            cotangent = cotangent_cpu.to(device=output_basis.device)
            if boundary_row_scales_cpu is not None:
                cotangent = cotangent * boundary_row_scales_cpu[boundary_index].to(
                    device=output_basis.device
                ).unsqueeze(-1)
            contribution = self._vjp_chunk_tensor_only(
                state,
                cotangent,
                output_basis,
                boundary_index=boundary_index,
                row_start=0,
                row_stop=int(cotangent.shape[0]),
            )
            accumulated = [
                value + delta
                for value, delta in zip(
                    accumulated,
                    contribution,
                    strict=True,
                )
            ]
        return tuple(accumulated)


class _WrongBundleVjpKernel(_BundleTwoSiteKernel):
    __slots__ = ()

    def _vjp_bundle_tensor_only(
        self,
        _state,
        _boundary_cotangents_cpu,
        output_basis,
        *,
        boundary_row_scales_cpu,
        maximum_workspace_bytes,
    ):
        del boundary_row_scales_cpu, maximum_workspace_bytes
        return tuple(
            torch.zeros(
                (output_basis.shape[1], site.input_width),
                device=output_basis.device,
                dtype=output_basis.dtype,
            )
            for site in self.topology.sites
        )


class _MutatingBundleCotangentKernel(_BundleTwoSiteKernel):
    __slots__ = ()

    def _vjp_bundle_tensor_only(
        self,
        state,
        boundary_cotangents_cpu,
        output_basis,
        *,
        boundary_row_scales_cpu,
        maximum_workspace_bytes,
    ):
        boundary_cotangents_cpu[0].add_(1.0)
        return super()._vjp_bundle_tensor_only(
            state,
            boundary_cotangents_cpu,
            output_basis,
            boundary_row_scales_cpu=boundary_row_scales_cpu,
            maximum_workspace_bytes=maximum_workspace_bytes,
        )


class _AliasedBundleOutputKernel(_BundleTwoSiteKernel):
    __slots__ = ()

    def _jvp_bundle_tensor_only(self, *args, **kwargs):
        outputs = super()._jvp_bundle_tensor_only(*args, **kwargs)
        return outputs[0], outputs[0]


class _AliasedBundleVjpResultKernel(_BundleTwoSiteKernel):
    __slots__ = ()

    def _vjp_bundle_tensor_only(self, *args, **kwargs):
        blocks = super()._vjp_bundle_tensor_only(*args, **kwargs)
        backing = torch.cat([value.reshape(-1) for value in blocks])
        first_count = blocks[0].numel()
        return (
            backing[:first_count].reshape(blocks[0].shape),
            backing[first_count:].reshape(blocks[1].shape),
        )


class _WrongVjpKernel(_TwoSiteKernel):
    __slots__ = ()

    def _vjp_chunk_tensor_only(
        self,
        _state,
        _boundary_cotangent,
        output_basis,
        *,
        boundary_index: int,
        row_start: int,
        row_stop: int,
    ):
        del boundary_index, row_start, row_stop
        return tuple(
            torch.zeros(
                (output_basis.shape[1], site.input_width),
                device=output_basis.device,
                dtype=output_basis.dtype,
            )
            for site in self.topology.sites
        )


class _MutatingStateKernel(_TwoSiteKernel):
    __slots__ = ()

    def _jvp_chunk_tensor_only(self, state, *args, **kwargs):
        state["producer0"].add_(1.0)
        return super()._jvp_chunk_tensor_only(state, *args, **kwargs)


class _MutatingBasisKernel(_TwoSiteKernel):
    __slots__ = ()

    def _jvp_chunk_tensor_only(
        self,
        state,
        coefficient_blocks,
        output_basis,
        **kwargs,
    ):
        output_basis.add_(1.0)
        return super()._jvp_chunk_tensor_only(
            state,
            coefficient_blocks,
            output_basis,
            **kwargs,
        )


class _BackwardKernel(_TwoSiteKernel):
    __slots__ = ()

    def _jvp_chunk_tensor_only(self, state, *args, **kwargs):
        torch.ones((), requires_grad=True).backward()
        return super()._jvp_chunk_tensor_only(state, *args, **kwargs)


def _problem(
    *,
    pcg_iterations: int = 64,
    kernel_type=_TwoSiteKernel,
    boundary_chunk_rows: int = 3,
    conditioning_scale: float | None = None,
):
    topology = _topology()
    basis = _basis(topology)
    rows = 6
    generator = torch.Generator().manual_seed(9917)
    defect0 = torch.randn((rows, 4), generator=generator, dtype=torch.float64)
    defect1 = torch.randn((rows, 4), generator=generator, dtype=torch.float64)
    mass0 = torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0, 2.0], dtype=torch.float64)
    mass1 = torch.tensor([2.0, 1.0, 1.0, 1.0, 3.0, 2.0], dtype=torch.float64)
    boundary0 = seal_lfm2_global_residual_boundary(
        boundary_id="layer.000.post_operator",
        layer_index=0,
        causal_order=1,
        boundary_kind="post_operator_residual",
        defect=defect0,
        defect_sha256=fingerprint_lfm2_global_residual_tensor(defect0),
        row_mass=mass0,
        row_mass_sha256=fingerprint_lfm2_global_residual_tensor(mass0),
    )
    boundary1 = seal_lfm2_global_residual_boundary(
        boundary_id="layer.001.post_layer",
        layer_index=1,
        causal_order=3,
        boundary_kind="post_layer_residual",
        defect=defect1,
        defect_sha256=fingerprint_lfm2_global_residual_tensor(defect1),
        row_mass=mass1,
        row_mass_sha256=fingerprint_lfm2_global_residual_tensor(mass1),
    )
    boundaries = Lfm2GlobalResidualBoundaryBundle((boundary0, boundary1))
    producer0 = torch.randn((rows, 3), generator=generator, dtype=torch.float64)
    producer1 = torch.randn((rows, 2), generator=generator, dtype=torch.float64)
    transition = torch.randn((4, 4), generator=generator, dtype=torch.float64) * 0.2
    transition += torch.eye(4, dtype=torch.float64) * 0.7

    state = {
        "producer0": producer0,
        "producer1": producer1,
        "transition": transition,
    }
    conditioning_kwargs = (
        {}
        if conditioning_scale is None
        else {
            "operator_squared_conditioning_scale": conditioning_scale,
            "block_squared_conditioning_scales": (
                conditioning_scale / 2.0,
                conditioning_scale / 2.0,
            ),
            "conditioning_scale_method": (
                "weighted-local-direct-test-scale-not-certified-bound"
            ),
        }
    )
    receipt = seal_lfm2_global_residual_linearization_receipt(
        topology,
        boundaries,
        kernel_id="test.manual-causal-two-site.v1",
        source_sha256={
            "producer0": fingerprint_lfm2_global_residual_tensor(producer0),
            "producer1": fingerprint_lfm2_global_residual_tensor(producer1),
            "transition": fingerprint_lfm2_global_residual_tensor(transition),
        },
        analytic_math_components=(
            "neural_forge.global_writer_block.apply",
            "neural_forge.global_writer_block.adjoint",
            "test.manual_residual_transition",
        ),
        operator_squared_norm_upper_bound=100.0,
        block_squared_norm_upper_bounds=(100.0, 100.0),
        **conditioning_kwargs,
    )
    policy = Lfm2GlobalResidualClosurePolicy(
        backend="cpu",
        condition_cap=10_000.0,
        pcg_iterations=pcg_iterations,
        boundary_chunk_rows=boundary_chunk_rows,
    )
    kernel = kernel_type(
        topology,
        boundaries,
        tensor_state=state,
        tensor_state_sha256=_hashes(state),
        linearization_receipt=receipt,
        transpose_contract_body={
            "derivation": "manual_reverse_composition",
            "jvp_order": ["writer0", "transition", "writer1"],
            "vjp_order": ["writer1.T", "transition.T", "writer0.T"],
        },
    )
    operator = Lfm2GlobalResidualClosureOperator(
        topology,
        basis,
        boundaries,
        kernel=kernel,
        policy=policy,
    )
    return {
        "topology": topology,
        "basis": basis,
        "boundaries": boundaries,
        "operator": operator,
        "kernel": kernel,
        "receipt": receipt,
        "producer0": producer0,
        "producer1": producer1,
        "transition": transition,
    }


def test_canonical_topology_builds_exact_rank_128_mixed_basis() -> None:
    hidden = 132
    topology = Lfm2GlobalResidualWriterTopology(
        hidden_size=hidden,
        head_dimension=64,
        sites=(
            Lfm2GlobalResidualWriterSite(
                site_id="layer.000.attention.out_proj",
                layer_index=0,
                causal_order=0,
                writer_kind="attention_output",
                input_width=64,
            ),
            Lfm2GlobalResidualWriterSite(
                site_id="layer.001.short_conv.out_proj",
                layer_index=1,
                causal_order=1,
                writer_kind="short_conv_output",
                input_width=132,
            ),
            Lfm2GlobalResidualWriterSite(
                site_id="layer.001.mlp.w2",
                layer_index=1,
                causal_order=2,
                writer_kind="mlp_down",
                input_width=71,
            ),
        ),
    )
    first = _basis(topology, seed=81)
    second = _basis(topology, seed=81)

    assert topology.topology_basis_rank == 128
    assert topology.effective_basis_rank == 128
    assert first.output_basis.shape == (hidden, 128)
    assert len(first.qdeim_coordinates) == 128
    assert len(set(first.qdeim_coordinates)) == 128
    assert first.report["canonical_lfm2_rank_128"] is True
    assert first.report["all_three_source_groups_present"] is True
    assert [value["group"] for value in first.report["group_reports"]] == [
        "mapped_fit_residual_defects",
        "source_writer_weights",
        "terminal_tied_table_directions",
    ]
    gram = first.output_basis.T @ first.output_basis
    assert torch.allclose(gram, torch.eye(128, dtype=torch.float64), atol=2e-12)
    assert first.verify_authentication()
    assert torch.equal(first.output_basis, second.output_basis)
    assert first.qdeim_coordinates == second.qdeim_coordinates
    assert first.report["report_sha256"] == second.report["report_sha256"]


def test_basis_authenticates_sources_and_detects_result_tampering() -> None:
    topology = _topology()
    generator = torch.Generator().manual_seed(13)
    defects = {"d": torch.randn((3, 4), generator=generator)}
    writers = {"w": torch.randn((4, 3), generator=generator)}
    terminal = {"t": torch.randn((2, 4), generator=generator)}
    bad_hashes = _hashes(defects)
    bad_hashes["d"] = _digest("not-the-defect")

    with pytest.raises(
        Lfm2GlobalResidualClosureAuthenticationError,
        match="mapped_fit_residual_defects.d differs",
    ):
        build_lfm2_global_residual_basis(
            topology,
            mapped_fit_residual_defects=defects,
            mapped_fit_residual_defects_sha256=bad_hashes,
            source_writer_weights=writers,
            source_writer_weights_sha256=_hashes(writers),
            terminal_tied_table_directions=terminal,
            terminal_tied_table_directions_sha256=_hashes(terminal),
        )

    basis = _basis(topology)
    tampered_tensor = basis.output_basis.clone()
    tampered_tensor[0, 0] += 0.25
    tampered = replace(basis, output_basis=tampered_tensor)
    assert tampered.verify_authentication() is False


def test_writer_block_parameterization_has_an_exact_transpose() -> None:
    generator = torch.Generator().manual_seed(21)
    rows = torch.randn((2, 5, 3), generator=generator, dtype=torch.float64)
    basis = torch.linalg.qr(
        torch.randn((6, 4), generator=generator, dtype=torch.float64)
    ).Q
    coefficients = torch.randn((4, 3), generator=generator, dtype=torch.float64)
    cotangent = torch.randn((2, 5, 6), generator=generator, dtype=torch.float64)

    effect = apply_lfm2_global_residual_writer_block(
        rows,
        basis,
        coefficients,
    )
    transpose = adjoint_lfm2_global_residual_writer_block(
        rows,
        basis,
        cotangent,
    )

    assert effect.shape == (2, 5, 6)
    assert transpose.shape == (4, 3)
    assert float((effect * cotangent).sum().item()) == pytest.approx(
        float((coefficients * transpose).sum().item()),
        abs=2e-12,
    )


def _dense_oracle(operator):
    shapes = [tuple(value.shape) for value in operator.zero_coefficients()]
    coordinate_count = sum(torch.empty(shape).numel() for shape in shapes)
    columns = []
    for coordinate in range(coordinate_count):
        blocks = list(operator.zero_coefficients())
        remaining = coordinate
        for block_index, block in enumerate(blocks):
            if remaining < block.numel():
                block.reshape(-1)[remaining] = 1.0
                break
            remaining -= block.numel()
        effects = operator.jvp(tuple(blocks))
        weighted = []
        for boundary, effect in zip(
            operator.boundaries.boundaries,
            effects,
            strict=True,
        ):
            weight = boundary.row_mass / (
                len(operator.boundaries.boundaries) * operator.topology.hidden_size
            )
            weighted.append((effect * weight.sqrt().unsqueeze(-1)).reshape(-1))
        columns.append(torch.cat(weighted))
    design = torch.stack(columns, dim=1)
    target_parts = []
    for boundary in operator.boundaries.boundaries:
        weight = boundary.row_mass / (
            len(operator.boundaries.boundaries) * operator.topology.hidden_size
        )
        target_parts.append((boundary.defect * weight.sqrt().unsqueeze(-1)).reshape(-1))
    target = torch.cat(target_parts)
    gram = design.T @ design
    exact = torch.linalg.solve(
        gram + operator.ridge * torch.eye(coordinate_count, dtype=torch.float64),
        design.T @ target,
    )
    return exact


def test_global_matrix_free_pcg_matches_dense_ridge_oracle() -> None:
    problem = _problem(pcg_iterations=96)
    operator = problem["operator"]
    exact = _dense_oracle(operator)
    result = solve_lfm2_global_residual_closure(operator)
    actual = torch.cat([value.reshape(-1) for value in result.coefficient_blocks])
    report = result.report

    assert torch.allclose(actual, exact, atol=2e-8, rtol=2e-8)
    assert result.verify_authentication()
    assert report["basis_rank"] == 4
    assert report["coefficient_coordinate_count"] == 20
    assert report["dense_writer_coordinate_count"] == 20
    assert report["standalone_dense_writer_corrections_materialized"] is True
    assert report["shared_or_sparse_runtime_sidecar_required"] is False
    assert report["solve"]["normal_design_materialized"] is False
    assert report["solve"]["gram_materialized"] is False
    assert report["solve"]["early_stopping"] is False
    assert len(report["solve"]["residual_norms"]) == 97
    assert report["conditioning"]["condition_cap_passed"] is True
    assert report["weighted_fit_objective_improved_diagnostic"] is True
    assert report["quality_gate"] is False
    assert report["numeric_diagnostics_used_for_auto_veto"] is False
    assert report["execution"]["backward_calls"] == 0
    assert report["execution"]["optimizer_steps"] == 0
    assert report["execution"]["pruning_operations"] == 0
    assert report["execution"]["quantization_operations"] == 0
    assert report["execution"]["guard_reports_clean"] is True
    assert report["fold_only_guard"]["kernel_call_count"] > 0
    assert report["fold_only_guard"]["hardcoded_zero_counter_claims_used"] is False
    assert report["authentication_checks"]["cheap_tensor_version_checks"] > 0
    assert report["authentication_checks"]["force_raw_tensor_hash_checks"] == 4
    assert report["authentication_checks"]["force_raw_hash_per_bundle_call"] is False
    assert report["streaming"]["boundary_archive_storage_device"] == "cpu"
    assert report["streaming"]["full_boundary_compute_staging"] is False
    assert report["streaming"]["workspace_budget_passed"] is True
    assert report["transpose_contract_sha256"]


def test_nonbound_conditioning_scale_controls_ridge_without_quality_authority() -> None:
    problem = _problem(pcg_iterations=4, conditioning_scale=0.25)
    operator = problem["operator"]

    assert operator.ridge == pytest.approx(0.25 / 9_999.0)
    assert operator.operator_squared_conditioning_scale == pytest.approx(0.25)
    assert operator.regularized_condition_upper_bound > operator.policy.condition_cap

    result = solve_lfm2_global_residual_closure(operator)
    conditioning = result.report["conditioning"]
    assert conditioning["conditioning_scale_is_certified_norm_bound"] is False
    assert conditioning["conditioning_scale_has_quality_gate_authority"] is False
    assert conditioning["regularization_scale_ratio_target_passed"] is True
    assert conditioning["condition_cap_passed"] is False
    assert result.report["quality_gate"] is False


def test_whole_stack_adjoint_and_causal_proof_seams_are_authenticated() -> None:
    operator = _problem(pcg_iterations=4)["operator"]
    adjoint = audit_lfm2_global_residual_closure_adjoint(operator, probe_count=3)
    causality = audit_lfm2_global_residual_closure_causality(operator)

    assert verify_lfm2_global_residual_closure_report(adjoint)
    assert verify_lfm2_global_residual_closure_report(causality)
    assert adjoint["all_probes_within_declared_tolerance"] is True
    assert adjoint["maximum_relative_error"] < 2e-14
    assert causality["all_sites_within_declared_tolerance"] is True
    assert adjoint["diagnostic_only"] is True
    assert adjoint["used_to_auto_veto_publication"] is False


def test_wrong_vjp_fails_structural_contract_before_pcg() -> None:
    bad = _problem(pcg_iterations=2, kernel_type=_WrongVjpKernel)["operator"]

    audit = audit_lfm2_global_residual_closure_adjoint(bad)
    assert audit["all_probes_within_declared_tolerance"] is False
    assert audit["diagnostic_only"] is True

    with pytest.raises(Exception, match="structural transpose proof"):
        solve_lfm2_global_residual_closure(bad)
    assert bad.normal_calls == 0


def test_bundle_normal_uses_one_global_jvp_and_one_global_vjp() -> None:
    chunk_operator = _problem(
        pcg_iterations=2,
        kernel_type=_TwoSiteKernel,
        boundary_chunk_rows=1,
    )["operator"]
    bundle_operator = _problem(
        pcg_iterations=2,
        kernel_type=_BundleTwoSiteKernel,
        boundary_chunk_rows=1,
    )["operator"]
    coefficients = tuple(
        torch.arange(
            block.numel(),
            device=block.device,
            dtype=block.dtype,
        ).reshape(block.shape)
        / 19.0
        for block in bundle_operator.zero_coefficients()
    )

    expected = chunk_operator.normal(coefficients)
    actual = bundle_operator.normal(coefficients)
    audit = bundle_operator.fold_only_guard_audit()

    assert bundle_operator.kernel.supports_bundle_linearization is True
    assert all(
        torch.allclose(left, right, atol=2e-12, rtol=2e-12)
        for left, right in zip(actual, expected, strict=True)
    )
    assert bundle_operator.jvp_calls == 1
    assert bundle_operator.vjp_calls == 1
    assert bundle_operator.jvp_bundle_calls == 1
    assert bundle_operator.vjp_bundle_calls == 1
    assert bundle_operator.jvp_chunk_calls == 0
    assert bundle_operator.vjp_chunk_calls == 0
    assert audit["kernel_call_count"] == 1
    assert audit["guard_phase_counts"] == {"global-residual-closure-normal-0": 1}


def test_bundle_transpose_contract_audits_actual_fast_path() -> None:
    bad = _problem(
        pcg_iterations=2,
        kernel_type=_WrongBundleVjpKernel,
    )["operator"]

    audit = audit_lfm2_global_residual_closure_adjoint(bad)

    assert audit["all_probes_within_declared_tolerance"] is False
    assert bad.jvp_bundle_calls == 2
    assert bad.vjp_bundle_calls == 2
    assert bad.jvp_chunk_calls == 0
    assert bad.vjp_chunk_calls == 0
    with pytest.raises(Exception, match="structural transpose proof"):
        solve_lfm2_global_residual_closure(bad)
    assert bad.normal_calls == 0


def test_bundle_vjp_cannot_corrupt_operator_boundary_archives() -> None:
    operator = _problem(
        pcg_iterations=2,
        kernel_type=_MutatingBundleCotangentKernel,
    )["operator"]
    before = tuple(
        fingerprint_lfm2_global_residual_tensor(value)
        for value in operator._defects_cpu
    )

    with pytest.raises(
        Lfm2GlobalResidualClosureAuthenticationError,
        match="attempted to mutate",
    ):
        operator.right_hand_side()

    after = tuple(
        fingerprint_lfm2_global_residual_tensor(value)
        for value in operator._defects_cpu
    )
    assert after == before
    operator.assert_internal_tensor_authentication(
        "adversarial bundle test",
        force_raw=True,
    )


@pytest.mark.parametrize(
    ("kernel_type", "operation"),
    [
        (
            _AliasedBundleOutputKernel,
            lambda operator: operator.jvp(operator.zero_coefficients()),
        ),
        (
            _AliasedBundleVjpResultKernel,
            lambda operator: operator.right_hand_side(),
        ),
    ],
)
def test_bundle_outputs_must_own_independent_storage(
    kernel_type,
    operation,
) -> None:
    operator = _problem(pcg_iterations=2, kernel_type=kernel_type)["operator"]

    with pytest.raises(
        Lfm2GlobalResidualClosureAuthenticationError,
        match="mutually independent",
    ):
        operation(operator)


@pytest.mark.parametrize(
    ("kernel_type", "message"),
    [
        (_MutatingStateKernel, "kernel mutated"),
        (_MutatingBasisKernel, "mutate cloned basis"),
    ],
)
def test_kernel_state_and_basis_mutation_fail_authentication(
    kernel_type,
    message,
) -> None:
    problem = _problem(pcg_iterations=2, kernel_type=kernel_type)

    with pytest.raises(
        Lfm2GlobalResidualClosureAuthenticationError,
        match=message,
    ):
        problem["operator"].jvp(problem["operator"].zero_coefficients())


def test_actual_backward_attempt_is_stopped_by_fold_only_guard() -> None:
    problem = _problem(pcg_iterations=2, kernel_type=_BackwardKernel)

    with pytest.raises(Exception, match="Backward calls are forbidden"):
        problem["operator"].jvp(problem["operator"].zero_coefficients())


def test_external_state_and_public_basis_are_non_aliasing_clones() -> None:
    problem = _problem(pcg_iterations=2)
    original_kernel_sha = problem["kernel"].authentication_sha256
    original_basis = problem["operator"].output_basis

    problem["producer0"].add_(100.0)
    exposed_basis = problem["operator"].output_basis
    exposed_basis.add_(100.0)

    assert problem["kernel"].verify_authentication(force_raw=True)
    assert problem["kernel"].authentication_sha256 == original_kernel_sha
    assert torch.equal(problem["operator"].output_basis, original_basis)
    assert problem["operator"].verify_authentication()


def test_core_owned_boundary_mutation_is_detected_before_publication() -> None:
    problem = _problem(pcg_iterations=2)
    problem["operator"]._weights_cpu[0].add_(1.0)

    with pytest.raises(
        Lfm2GlobalResidualClosureAuthenticationError,
        match="operator tensor state mutated",
    ):
        solve_lfm2_global_residual_closure(problem["operator"])


def test_all_zero_basis_evidence_canonical_completes_without_veto() -> None:
    topology = _topology()
    defects = {"zero-defect": torch.zeros((4, 4), dtype=torch.float64)}
    writers = {
        site.site_id: torch.zeros(
            (topology.hidden_size, site.input_width), dtype=torch.float64
        )
        for site in topology.sites
    }
    terminal = {"zero-terminal": torch.zeros((3, 4), dtype=torch.float64)}

    basis = build_lfm2_global_residual_basis(
        topology,
        mapped_fit_residual_defects=defects,
        mapped_fit_residual_defects_sha256=_hashes(defects),
        source_writer_weights=writers,
        source_writer_weights_sha256=_hashes(writers),
        terminal_tied_table_directions=terminal,
        terminal_tied_table_directions_sha256=_hashes(terminal),
    )

    assert basis.verify_authentication()
    assert basis.report["nonzero_source_group_count"] == 0
    assert basis.report["all_zero_evidence_uses_canonical_completion"] is True
    assert basis.report["deterministic_completion_columns"] == basis.rank
    assert torch.equal(basis.output_basis, torch.eye(4, dtype=torch.float64))
    assert len(basis.report["zero_source_groups_skipped"]) == 3
    assert basis.report["numeric_diagnostics_used_for_auto_veto"] is False


def test_h776_real_geometry_streaming_stays_below_workspace_budget() -> None:
    report = estimate_lfm2_global_residual_streaming_memory(
        hidden_size=776,
        boundary_row_counts=(40_656,) * 29,
        requested_chunk_rows=256,
        maximum_workspace_bytes=512 * 1024 * 1024,
    )

    assert verify_lfm2_global_residual_closure_report(report)
    assert report["unstreamed_defect_effect_weighted_bytes"] > 20 * 1024**3
    assert report["estimated_peak_boundary_workspace_bytes"] < 5 * 1024**2
    assert report["boundary_archive_storage_device"] == "cpu"
    assert report["whole_stack_bundle_output_storage_device"] == "cpu"
    assert report["whole_stack_bundle_effect_archive_bytes"] > 6 * 1024**3
    assert report["whole_stack_bundle_weighted_effect_copy_materialized"] is False
    assert report["full_boundary_compute_staging"] is False
    assert report["defect_effect_weighted_triple_copy_on_gpu"] is False
    assert report["workspace_budget_passed"] is True


def test_chunk_count_does_not_multiply_fold_only_guard_setup() -> None:
    operator = _problem(
        pcg_iterations=2,
        boundary_chunk_rows=1,
    )["operator"]
    operator.jvp(operator.zero_coefficients())
    audit = operator.fold_only_guard_audit()

    assert operator.jvp_calls == 12
    assert audit["kernel_call_count"] == 1
    assert audit["guard_phase_counts"] == {"global-residual-closure-full-jvp": 1}


def test_solved_corrections_bake_into_dense_target_dtype_writers() -> None:
    problem = _problem(pcg_iterations=16)
    result = solve_lfm2_global_residual_closure(problem["operator"])
    generator = torch.Generator().manual_seed(44)
    seeds = {
        site.site_id: torch.randn(
            (problem["topology"].hidden_size, site.input_width),
            generator=generator,
            dtype=torch.float32,
        )
        for site in problem["topology"].sites
    }
    before = {name: value.clone() for name, value in seeds.items()}
    state = materialize_lfm2_global_residual_corrected_writers(
        problem["topology"],
        result,
        seed_writer_weights=seeds,
        seed_writer_weights_sha256=_hashes(seeds),
    )

    assert state.verify_authentication()
    assert state.site_ids == tuple(site.site_id for site in problem["topology"].sites)
    assert all(value.dtype == torch.float32 for value in state.writer_weights)
    assert all(value.layout == torch.strided for value in state.writer_weights)
    assert all(torch.equal(value, before[name]) for name, value in seeds.items())
    assert state.report["all_tensors_dense"] is True
    assert state.report["runtime_sidecar_required"] is False
    assert state.report["execution"]["training_steps"] == 0


def test_linearization_receipt_must_bind_exact_topology_and_boundaries() -> None:
    problem = _problem(pcg_iterations=2)
    different_topology = replace(
        problem["topology"],
        head_dimension=1,
        topology_sha256="",
    )

    with pytest.raises(
        Lfm2GlobalResidualClosureAuthenticationError,
        match="does not bind",
    ):
        Lfm2GlobalResidualClosureOperator(
            different_topology,
            _basis(different_topology),
            problem["boundaries"],
            kernel=problem["kernel"],
            policy=problem["operator"].policy,
        )
