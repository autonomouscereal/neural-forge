"""Stage-1 proofs for T5 sparse residual-OMP routes on body components.

Four things are proven here, all on synthetic sealed-evidence-shaped tensors
with no optimizer, no gradient, and no model execution:

* the exact-identity gates — at no contraction the child weight is the parent
  weight bit-for-bit, and at ``k=full`` on evidence whose column rank the kept
  channels span, the OMP reconstruction is exact;
* the A11 split is real — read and write coefficients share supports, differ
  when the write metric couples the dropped channels, and the write solve
  strictly improves the induced-output objective it claims to;
* every refusal the module names actually closes; and
* on captured-evidence-shaped input (low-rank correlated activations, a real
  ``mlp_down``-shaped writer) the routes beat plain truncation.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest


torch = pytest.importorskip("torch")

import neural_forge.matrix_technique_body_routes as body_routes  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_feature_fold import FeatureFoldError  # noqa: E402
from neural_forge.matrix_technique_body_routes import (  # noqa: E402
    MATRIX_TECHNIQUE_ENTRY_SCHEMA,
    T5_BODY_ROUTE_RECORD_SCHEMA,
    T5_BODY_ROUTE_SOURCE_WITNESS_SCHEMA,
    T5BodyRouteConditioningError,
    T5BodyRouteDisciplineError,
    T5BodyRouteError,
    T5BodyRouteEvidenceError,
    T5BodyRouteGeometryError,
    T5BodyRouteAuthenticationError,
    build_t5_body_route_source_witness,
    fit_body_component_routes,
    fit_from_request,
    matrix_technique_registry_entry,
    resolve_component,
    resolve_kept_width,
    select_kept_channels,
    sparse_feature_fold_map_from_body_route,
    t5_tensor_sha256,
    verify_sealed_body_route_record,
    verify_t5_body_route_source_witness,
)


def _normal(seed: int, rows: int, columns: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, columns, generator=generator, dtype=torch.float64)


def _captured_evidence(
    seed: int,
    rows: int,
    channels: int,
    latent: int,
    *,
    noise: float = 0.02,
) -> torch.Tensor:
    """Low-rank correlated rows shaped like a captured activation cache."""

    return _captured_split(seed, rows, 0, channels, latent, noise=noise)[0]


def _captured_split(
    seed: int,
    fit_rows: int,
    heldout_rows: int,
    channels: int,
    latent: int,
    *,
    noise: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One captured cache split into fit and held-out rows.

    Both halves share the latent subspace — a held-out split drawn from a
    *different* random subspace would not test transfer, it would test
    extrapolation into directions the evidence never carried, which is not
    what the campaign's held-out split is.
    """

    generator = torch.Generator().manual_seed(seed)
    total = fit_rows + heldout_rows
    factors = torch.randn(total, latent, generator=generator, dtype=torch.float64)
    mixing = torch.randn(latent, channels, generator=generator, dtype=torch.float64)
    perturbation = torch.randn(
        total, channels, generator=generator, dtype=torch.float64
    )
    rows = factors @ mixing + noise * perturbation
    return rows[:fit_rows].contiguous(), rows[fit_rows:].contiguous()


def _exact_sparse_adapter_fixture(role: str):
    rows = torch.tensor(
        [
            [8.0, 0.0, 1.0, 0.0],
            [0.0, 8.0, 0.0, 1.0],
            [8.0, 0.0, 1.0, 0.0],
            [0.0, 8.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    if role == "writer":
        weight = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            dtype=torch.float64,
        )
        component = "mlp_down"
        extra = {"write_metric": torch.eye(4, dtype=torch.float64)}
    else:
        weight = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [4.0, 5.0, 6.0],
                [8.0, 10.0, 12.0],
            ],
            dtype=torch.float64,
        )
        component = "mlp_gate"
        extra = {}
    fold = fit_body_component_routes(
        parent_weight=weight,
        component_rows=rows,
        component=component,
        level="X4",
        kept_width=2,
        active_k=1,
        channel_selection="explicit",
        kept_channel_ids=[0, 1],
        refit_ridge_relative=0.0,
        write_ridge_relative=0.0,
        **extra,
    )
    return weight, fold


def _reseal_adapter_fixture(fold, **changes):
    record = dict(fold.record)
    if "kept_channel_ids" in changes:
        record["kept_channel_ids_sha256"] = body_routes._tensor_sha256(
            changes["kept_channel_ids"]
        )
    if "support_channel_ids" in changes:
        record["support_channel_ids_sha256"] = body_routes._tensor_sha256(
            changes["support_channel_ids"]
        )
    if "support_offsets" in changes:
        record["support_offsets_sha256"] = body_routes._tensor_sha256(
            changes["support_offsets"]
        )
    if "read_coefficients" in changes:
        record["read_coefficients_sha256"] = body_routes._tensor_sha256(
            changes["read_coefficients"]
        )
    if "write_coefficients" in changes:
        record["write_coefficients_sha256"] = body_routes._tensor_sha256(
            changes["write_coefficients"]
        )
    if "expansion_route" in changes:
        record["expansion_route_sha256"] = body_routes._tensor_sha256(
            changes["expansion_route"]
        )
    record.pop("record_sha256")
    record["record_sha256"] = canonical_sha256(record)
    return replace(fold, record=MappingProxyType(record), **changes)


_SOURCE_BINDING = {
    "parameter_path": "model.layers.5.feed_forward.w2.weight",
    "graph_fingerprint": "a" * 64,
    "parent_state_fingerprint": "b" * 64,
    "component_id": "C13",
    "layer_index": 5,
    "site": "L05",
    "role": "writer",
    "evidence_cache_sha256": "c" * 64,
    "evidence_selection_sha256": "d" * 64,
    "evidence_endpoint": "L05.mlp_down.input",
    "coefficient_set": "write",
}


def _source_witness_fixture():
    parent = _normal(60, 7, 12)
    rows, heldout = _captured_split(61, 80, 24, 12, 6)
    fold = fit_body_component_routes(
        parent_weight=parent,
        component_rows=rows,
        heldout_component_rows=heldout,
        component="C13",
        level="M4",
        site="L05",
        active_k=3,
    )
    witness = build_t5_body_route_source_witness(
        fold,
        parent_weight=parent,
        component_rows=rows,
        heldout_component_rows=heldout,
        **_SOURCE_BINDING,
    )
    return parent, rows, heldout, fold, witness


# ---------------------------------------------------------------------------
# Level and component algebra
# ---------------------------------------------------------------------------


def test_levels_match_the_design_width_table():
    assert resolve_kept_width(level="M16", component_width=2560) == 2400
    assert resolve_kept_width(level="M8", component_width=2560) == 2240
    assert resolve_kept_width(level="M4", component_width=2560) == 1920
    assert resolve_kept_width(level="M16", component_width=1024) == 960
    assert resolve_kept_width(level="M8", component_width=1024) == 896
    assert resolve_kept_width(level="M4", component_width=1024) == 768


def test_x4_refuses_without_an_option_bank_width():
    with pytest.raises(T5BodyRouteGeometryError, match="option bank"):
        resolve_kept_width(level="X4", component_width=1024)
    assert resolve_kept_width(level="X4", component_width=1024, kept_width=256) == 256


def test_component_roles_follow_the_campaign_table():
    assert resolve_component("C13") == ("C13", "mlp_down", "writer")
    assert resolve_component("mlp_gate") == ("C11", "mlp_gate", "reader")
    assert resolve_component("C09") == ("C09", "attn_o", "writer")
    with pytest.raises(T5BodyRouteGeometryError, match="no channel-route"):
        resolve_component("C04")
    with pytest.raises(T5BodyRouteGeometryError, match="mis-orient"):
        resolve_component("mlp_down", "reader")


# ---------------------------------------------------------------------------
# Exact-identity gates
# ---------------------------------------------------------------------------


def test_no_contraction_reproduces_the_parent_writer_bit_for_bit():
    weight = _normal(11, 32, 64)
    rows = _captured_evidence(12, 160, 64, 20)
    fold = fit_body_component_routes(
        parent_weight=weight,
        component_rows=rows,
        component="mlp_down",
        level="M1",
        site="L05",
        active_k=4,
    )
    assert fold.kept_width == 64
    assert torch.equal(fold.child_weight, weight)
    assert fold.calibration["component_write_relative_l2"] == 0.0
    assert fold.calibration["dropped_channel_count"] == 0
    assert int(fold.support_channel_ids.numel()) == 0
    reference = rows @ weight.transpose(0, 1)
    assert torch.allclose(fold.apply(rows), reference, atol=0.0, rtol=0.0)


def test_no_contraction_reproduces_the_parent_reader_bit_for_bit():
    weight = _normal(13, 96, 32)
    rows = _captured_evidence(14, 160, 96, 24)
    fold = fit_body_component_routes(
        parent_weight=weight,
        component_rows=rows,
        component="mlp_gate",
        level="M1",
        site="L00",
        active_k=4,
    )
    assert torch.equal(fold.child_weight, weight)
    inputs = _normal(15, 24, 32)
    assert torch.allclose(
        fold.apply(inputs), inputs @ weight.transpose(0, 1), atol=1e-12
    )


def test_k_full_omp_reconstructs_spanned_channels_below_one_microunit():
    kept, width, rows = 24, 32, 96
    basis = _normal(16, rows, kept)
    mixing = _normal(17, kept, width)
    # Plant the kept columns as the dominant-energy exact basis so the default
    # evidence-energy policy selects a spanning set; every dropped column then
    # lies exactly in their span.
    mixing[:, :kept] = torch.eye(kept, dtype=torch.float64) * 8.0
    evidence = basis @ mixing
    fold = fit_body_component_routes(
        parent_weight=_normal(18, 16, width),
        component_rows=evidence,
        component="mlp_down",
        level="M4",
        kept_width=kept,
        active_k=kept,
        site="L00",
    )
    assert fold.calibration["channel_reconstruction_relative_l2"] < 1e-6
    assert fold.calibration["component_write_relative_l2"] < 1e-6
    assert int(fold.solve["omp"]["dependent_atom_rejections"]) >= 0


# ---------------------------------------------------------------------------
# The A11 split
# ---------------------------------------------------------------------------


def test_read_and_write_coefficients_share_supports_and_differ_when_coupled():
    weight = _normal(19, 48, 96)
    rows = _captured_evidence(20, 300, 96, 40)
    fold = fit_body_component_routes(
        parent_weight=weight,
        component_rows=rows,
        component="mlp_down",
        level="M8",
        site="L13",
        active_k=6,
        write_solve_sweeps=4,
    )
    assert fold.read_coefficients.shape == fold.write_coefficients.shape
    assert int(fold.support_offsets[-1]) == int(fold.read_coefficients.numel())
    assert fold.solve["write_solve"]["coupled"] is True
    assert fold.solve["write_solve"]["objective_improved"] is True
    assert (
        fold.solve["write_solve"]["final_objective"]
        < (fold.solve["write_solve"]["initial_objective"])
    )
    assert not torch.allclose(fold.read_coefficients, fold.write_coefficients)
    assert fold.installed_coefficients == "write"


def test_identity_write_metric_collapses_the_split_and_says_so():
    weight = _normal(21, 40, 64)
    rows = _captured_evidence(22, 200, 64, 24)
    fold = fit_body_component_routes(
        parent_weight=weight,
        component_rows=rows,
        component="mlp_down",
        level="M8",
        site="L00",
        active_k=6,
        write_metric=torch.eye(64, dtype=torch.float64),
    )
    assert fold.solve["write_solve"]["coupled"] is False
    assert fold.solve["write_solve"]["sweeps_run"] == 0
    assert torch.equal(fold.read_coefficients, fold.write_coefficients)


def test_write_coefficients_beat_read_coefficients_at_the_writer_output():
    weight = _normal(23, 48, 96)
    rows = _captured_evidence(24, 300, 96, 40)
    common = {
        "parent_weight": weight,
        "component_rows": rows,
        "component": "mlp_down",
        "level": "M8",
        "site": "L13",
        "active_k": 6,
    }
    read = fit_body_component_routes(**common, install_coefficients="read")
    write = fit_body_component_routes(**common, install_coefficients="write")
    assert torch.equal(read.support_channel_ids, write.support_channel_ids)
    assert (
        write.calibration["component_write_relative_l2"]
        < read.calibration["component_write_relative_l2"]
    )


# ---------------------------------------------------------------------------
# Captured-evidence-shaped behaviour
# ---------------------------------------------------------------------------


def test_routes_beat_plain_truncation_on_captured_shaped_evidence():
    width, out_features = 160, 64
    weight = _normal(25, out_features, width)
    rows, held = _captured_split(26, 512, 128, width, 28)
    fold = fit_body_component_routes(
        parent_weight=weight,
        component_rows=rows,
        component="mlp_down",
        level="M16",
        site="L05",
        active_k=8,
        heldout_component_rows=held,
    )
    kept = fold.kept_channel_ids
    truncated = held.index_select(1, kept) @ weight.index_select(1, kept).transpose(
        0, 1
    )
    reference = held @ weight.transpose(0, 1)
    truncation_error = float(
        torch.linalg.matrix_norm(truncated - reference, ord="fro").item()
    ) / float(torch.linalg.matrix_norm(reference, ord="fro").item())
    routed = fold.calibration["heldout"]["component_write_relative_l2"]
    assert routed < truncation_error
    assert fold.calibration["heldout"]["component_write_cosine"] > 0.9


def test_reader_expansion_carries_dropped_channels_on_held_out_rows():
    width, in_features = 128, 48
    weight = _normal(28, width, in_features)
    inputs, held_inputs = _captured_split(29, 400, 96, in_features, 20)
    rows = inputs @ weight.transpose(0, 1)
    fold = fit_body_component_routes(
        parent_weight=weight,
        component_rows=rows,
        component="mlp_gate",
        level="M16",
        site="L00",
        active_k=8,
    )
    assert fold.child_weight.shape == (fold.kept_width, in_features)
    reference = held_inputs @ weight.transpose(0, 1)
    routed = fold.apply(held_inputs)
    kept = fold.kept_channel_ids
    truncated = torch.zeros_like(reference)
    truncated[:, kept] = reference[:, kept]
    assert torch.linalg.matrix_norm(
        routed - reference, ord="fro"
    ) < torch.linalg.matrix_norm(truncated - reference, ord="fro")


def test_reader_block_count_shares_one_route_across_projection_blocks():
    internal, in_features, blocks = 64, 32, 3
    weight = _normal(31, blocks * internal, in_features)
    inputs = _captured_evidence(32, 300, in_features, 16)
    rows = _captured_evidence(33, 300, internal, 16)
    fold = fit_body_component_routes(
        parent_weight=weight,
        component_rows=rows,
        component="conv_in_proj",
        level="M8",
        site="L00",
        block_count=blocks,
        active_k=6,
    )
    assert fold.child_weight.shape == (blocks * fold.kept_width, in_features)
    expanded = fold.apply(inputs)
    assert expanded.shape == (300, blocks * internal)
    # Each block sees the same expansion, so the kept channels of every block
    # reproduce the parent exactly.
    reference = inputs @ weight.transpose(0, 1)
    for block in range(blocks):
        offset = block * internal
        columns = fold.kept_channel_ids + offset
        assert torch.allclose(
            expanded.index_select(1, columns),
            reference.index_select(1, columns),
            atol=1e-9,
        )


def test_public_t5_tensor_digest_is_the_record_digest_dialect():
    tensor = _normal(58, 3, 5)
    assert t5_tensor_sha256(tensor) == body_routes._tensor_sha256(tensor)


def test_t5_source_witness_api_is_publicly_exported():
    assert {
        "T5_BODY_ROUTE_SOURCE_WITNESS_SCHEMA",
        "T5BodyRouteSourceWitness",
        "build_t5_body_route_source_witness",
        "t5_tensor_sha256",
        "verify_t5_body_route_source_witness",
    }.issubset(body_routes.__all__)


def test_writer_source_witness_rederives_and_reauthenticates_live_sources():
    parent, rows, heldout, fold, witness = _source_witness_fixture()
    assert witness.record["schema"] == T5_BODY_ROUTE_SOURCE_WITNESS_SCHEMA
    assert witness.body_route_record_sha256 == fold.record_sha256
    assert witness.report["parent_weight_sha256"] == fold.record["parent_weight_sha256"]
    assert (
        witness.report["component_rows_sha256"] == fold.record["component_rows_sha256"]
    )
    assert (
        witness.report["selected_child_weight_sha256"]
        == fold.record["child_weight_sha256"]
    )
    assert (
        witness.report["selected_expansion_route_sha256"]
        == fold.record["expansion_route_sha256"]
    )
    assert witness.report["recomputed_calibration"] == dict(fold.calibration)
    assert witness.report["authority"] == {
        "live_source_tensor_identity": True,
        "writer_tensor_algebra": True,
        "fit_and_heldout_metrics": True,
        "solver_origin": "external-artifact-bound-not-attested",
        "support_selection_optimality": False,
    }

    authenticated = verify_t5_body_route_source_witness(
        witness,
        fold,
        parent_weight=parent,
        component_rows=rows,
        heldout_component_rows=heldout,
        **_SOURCE_BINDING,
    )
    assert authenticated.witness_sha256 == witness.witness_sha256
    assert authenticated.parent_weight.data_ptr() != witness.parent_weight.data_ptr()
    assert authenticated.component_rows.data_ptr() != witness.component_rows.data_ptr()


def test_writer_source_witness_rejects_same_shape_other_parent_and_rows():
    parent, rows, heldout, fold, _ = _source_witness_fixture()
    with pytest.raises(T5BodyRouteAuthenticationError, match="parent_weight"):
        build_t5_body_route_source_witness(
            fold,
            parent_weight=_normal(62, *parent.shape),
            component_rows=rows,
            heldout_component_rows=heldout,
            **_SOURCE_BINDING,
        )
    with pytest.raises(T5BodyRouteAuthenticationError, match="component_rows"):
        build_t5_body_route_source_witness(
            fold,
            parent_weight=parent,
            component_rows=_normal(63, *rows.shape),
            heldout_component_rows=heldout,
            **_SOURCE_BINDING,
        )


def test_writer_source_witness_rejects_genuine_fold_from_another_source():
    parent, rows, heldout, _, witness = _source_witness_fixture()
    other_parent = _normal(64, *parent.shape)
    other_rows, other_heldout = _captured_split(65, 80, 24, 12, 6)
    other_fold = fit_body_component_routes(
        parent_weight=other_parent,
        component_rows=other_rows,
        heldout_component_rows=other_heldout,
        component="C13",
        level="M4",
        site="L05",
        active_k=3,
    )
    with pytest.raises(T5BodyRouteAuthenticationError, match="different body-route"):
        verify_t5_body_route_source_witness(
            witness,
            other_fold,
            parent_weight=other_parent,
            component_rows=other_rows,
            heldout_component_rows=other_heldout,
            **_SOURCE_BINDING,
        )


def test_writer_source_witness_rejects_coherently_resealed_zero_route():
    parent, rows, heldout, fold, _ = _source_witness_fixture()
    zero_read = torch.zeros_like(fold.read_coefficients)
    zero_write = torch.zeros_like(fold.write_coefficients)
    child = parent.index_select(1, fold.kept_channel_ids)
    expansion = torch.zeros_like(fold.expansion_route)
    expansion[
        torch.arange(fold.kept_width, dtype=torch.long),
        fold.kept_channel_ids,
    ] = 1
    record = dict(fold.record)
    record.update(
        {
            "read_coefficients_sha256": t5_tensor_sha256(zero_read),
            "write_coefficients_sha256": t5_tensor_sha256(zero_write),
            "child_weight_sha256": t5_tensor_sha256(child),
            "expansion_route_sha256": t5_tensor_sha256(expansion),
        }
    )
    record.pop("record_sha256")
    record["record_sha256"] = canonical_sha256(record)
    tampered = replace(
        fold,
        read_coefficients=zero_read,
        write_coefficients=zero_write,
        child_weight=child,
        expansion_route=expansion,
        record=MappingProxyType(record),
    )
    with pytest.raises(T5BodyRouteAuthenticationError, match="calibration is stale"):
        build_t5_body_route_source_witness(
            tampered,
            parent_weight=parent,
            component_rows=rows,
            heldout_component_rows=heldout,
            **_SOURCE_BINDING,
        )


def test_writer_source_witness_requires_exact_heldout_source():
    parent, rows, heldout, fold, witness = _source_witness_fixture()
    with pytest.raises(T5BodyRouteAuthenticationError, match="has no heldout"):
        build_t5_body_route_source_witness(
            fold,
            parent_weight=parent,
            component_rows=rows,
            **_SOURCE_BINDING,
        )
    changed = heldout.clone()
    changed[0, 0] += 1.0
    with pytest.raises(T5BodyRouteAuthenticationError, match="calibration is stale"):
        build_t5_body_route_source_witness(
            fold,
            parent_weight=parent,
            component_rows=rows,
            heldout_component_rows=changed,
            **_SOURCE_BINDING,
        )
    with pytest.raises(T5BodyRouteAuthenticationError, match="differs from"):
        verify_t5_body_route_source_witness(
            witness,
            fold,
            parent_weight=parent,
            component_rows=rows,
            heldout_component_rows=changed,
            **_SOURCE_BINDING,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("parameter_path", "model.layers.5.feed_forward.other.weight"),
        ("graph_fingerprint", "e" * 64),
        ("parent_state_fingerprint", "f" * 64),
        ("component_id", "C05"),
        ("evidence_cache_sha256", "1" * 64),
        ("evidence_selection_sha256", "2" * 64),
        ("evidence_endpoint", "L05.other.input"),
        ("coefficient_set", "read"),
    ),
)
def test_writer_source_witness_binds_every_external_authority(field, value):
    parent, rows, heldout, fold, witness = _source_witness_fixture()
    binding = {**_SOURCE_BINDING, field: value}
    with pytest.raises(T5BodyRouteAuthenticationError, match=field):
        verify_t5_body_route_source_witness(
            witness,
            fold,
            parent_weight=parent,
            component_rows=rows,
            heldout_component_rows=heldout,
            **binding,
        )


def test_writer_source_witness_binds_layer_site_and_role():
    parent, rows, heldout, fold, witness = _source_witness_fixture()
    wrong_layer = {**_SOURCE_BINDING, "layer_index": 6, "site": "L06"}
    with pytest.raises(T5BodyRouteAuthenticationError, match="layer_index"):
        verify_t5_body_route_source_witness(
            witness,
            fold,
            parent_weight=parent,
            component_rows=rows,
            heldout_component_rows=heldout,
            **wrong_layer,
        )
    wrong_site = {**_SOURCE_BINDING, "site": "L06"}
    with pytest.raises(T5BodyRouteAuthenticationError, match="exact layer site"):
        verify_t5_body_route_source_witness(
            witness,
            fold,
            parent_weight=parent,
            component_rows=rows,
            heldout_component_rows=heldout,
            **wrong_site,
        )
    wrong_role = {**_SOURCE_BINDING, "role": "reader"}
    with pytest.raises(T5BodyRouteAuthenticationError, match="writer routes only"):
        verify_t5_body_route_source_witness(
            witness,
            fold,
            parent_weight=parent,
            component_rows=rows,
            heldout_component_rows=heldout,
            **wrong_role,
        )


def test_writer_source_witness_clones_sources_before_hash_callbacks(monkeypatch):
    parent, rows, heldout, fold, baseline = _source_witness_fixture()
    raced_parent = parent.clone()
    raced_rows = rows.clone()
    raced_heldout = heldout.clone()
    digest = body_routes.t5_tensor_sha256
    calls = 0

    def digest_then_mutate_sources(value):
        nonlocal calls
        result = digest(value)
        calls += 1
        if calls == 1:
            raced_parent.zero_()
            raced_rows.zero_()
            raced_heldout.zero_()
        return result

    monkeypatch.setattr(
        body_routes,
        "t5_tensor_sha256",
        digest_then_mutate_sources,
    )
    raced = build_t5_body_route_source_witness(
        fold,
        parent_weight=raced_parent,
        component_rows=raced_rows,
        heldout_component_rows=raced_heldout,
        **_SOURCE_BINDING,
    )
    assert calls > 1
    assert not bool(raced_parent.any().item())
    assert not bool(raced_rows.any().item())
    assert not bool(raced_heldout.any().item())
    assert raced.witness_sha256 == baseline.witness_sha256
    assert torch.equal(raced.parent_weight, parent)
    assert torch.equal(raced.component_rows, rows)
    assert torch.equal(raced.heldout_component_rows, heldout)


@pytest.mark.parametrize(
    ("role", "coefficient_set"),
    (("reader", "read"), ("writer", "write")),
)
def test_sparse_feature_adapter_has_exact_reader_and_writer_weight_parity(
    role, coefficient_set
):
    weight, fold = _exact_sparse_adapter_fixture(role)
    caller_binding = {"compiler_plan_sha256": "a" * 64, "phase": "component"}
    feature_map = sparse_feature_fold_map_from_body_route(
        fold,
        coefficient_set=coefficient_set,
        admission_binding=caller_binding,
    )

    assert feature_map.kept_source_ids.tolist() == [0, 1]
    assert feature_map.dropped_source_ids.tolist() == [2, 3]
    # T5 stored source ids [0, 1]; the generic map stores target positions.
    assert feature_map.support_target_ids.tolist() == [0, 1]
    expected_binding = {
        **caller_binding,
        "body_route_record_sha256": fold.record_sha256,
        "coefficient_set": coefficient_set,
        "component_id": fold.component_id,
        "site": fold.site,
        "level": fold.level,
        "role": fold.role,
        "installed_coefficients": fold.installed_coefficients,
        "source_width": fold.parent_width,
        "target_width": fold.kept_width,
    }
    assert feature_map.admission_binding_sha256 == canonical_sha256(expected_binding)
    assert feature_map.verify(
        expected_map_sha256=feature_map.map_sha256
    ).map_sha256 == (feature_map.map_sha256)
    if role == "reader":
        compiled = feature_map.fold_producer_weight(weight)
    else:
        compiled = feature_map.fold_consumer_weight(weight)
    assert torch.equal(compiled, fold.child_weight)


def test_sparse_feature_adapter_removes_zero_edges_but_retains_empty_routes():
    _, fold = _exact_sparse_adapter_fixture("reader")
    read = fold.read_coefficients.clone()
    read[0] = 0.0
    expansion = fold.expansion_route.clone()
    expansion[:, 2] = 0.0
    zero_route_fold = _reseal_adapter_fixture(
        fold,
        read_coefficients=read,
        expansion_route=expansion,
    )

    feature_map = sparse_feature_fold_map_from_body_route(
        zero_route_fold,
        coefficient_set="read",
    )
    assert feature_map.support_offsets.tolist() == [0, 0, 1]
    assert feature_map.support_target_ids.tolist() == [1]
    assert feature_map.coefficients.tolist() == [0.125]
    decoded = feature_map.decode(torch.tensor([[2.0, 3.0]], dtype=torch.float64))
    assert torch.equal(
        decoded,
        torch.tensor([[2.0, 3.0, 0.0, 0.375]], dtype=torch.float64),
    )


def test_sparse_feature_adapter_rejects_authenticated_unkept_support():
    _, fold = _exact_sparse_adapter_fixture("reader")
    supports = fold.support_channel_ids.clone()
    supports[0] = 2
    malformed = _reseal_adapter_fixture(fold, support_channel_ids=supports)
    with pytest.raises(T5BodyRouteAuthenticationError, match="not admitted"):
        sparse_feature_fold_map_from_body_route(
            malformed,
            coefficient_set="read",
        )


def test_sparse_feature_adapter_rejects_coherent_unsealed_route_repartition():
    _, fold = _exact_sparse_adapter_fixture("reader")
    offsets = torch.tensor([0, 2, 2], dtype=torch.long)
    expansion = torch.zeros_like(fold.expansion_route)
    expansion[0, 0] = 1.0
    expansion[1, 1] = 1.0
    expansion[:, 2] = fold.read_coefficients
    tampered = replace(
        fold,
        support_offsets=offsets,
        expansion_route=expansion,
    )
    with pytest.raises(
        T5BodyRouteAuthenticationError,
        match="support_offsets failed authentication",
    ):
        sparse_feature_fold_map_from_body_route(
            tampered,
            coefficient_set="read",
        )


def test_sparse_feature_adapter_fails_closed_on_legacy_unbound_route_record():
    _, fold = _exact_sparse_adapter_fixture("reader")
    legacy_record = dict(fold.record)
    legacy_record.pop("support_offsets_sha256")
    legacy_record.pop("expansion_route_sha256")
    legacy_record.pop("record_sha256")
    legacy_record["record_sha256"] = canonical_sha256(legacy_record)
    legacy = replace(fold, record=MappingProxyType(legacy_record))
    with pytest.raises(
        T5BodyRouteAuthenticationError,
        match="support_offsets failed authentication",
    ):
        sparse_feature_fold_map_from_body_route(
            legacy,
            coefficient_set="read",
        )


def test_sparse_feature_adapter_uses_authenticated_snapshot_after_boundary(
    monkeypatch,
):
    _, clean_fold = _exact_sparse_adapter_fixture("reader")
    baseline = sparse_feature_fold_map_from_body_route(
        clean_fold,
        coefficient_set="read",
        admission_binding={"compiler_plan_sha256": "c" * 64},
    )
    _, raced_fold = _exact_sparse_adapter_fixture("reader")
    authenticate = body_routes._authenticated_body_route_fold

    def authenticate_then_mutate(candidate):
        snapshot = authenticate(candidate)
        candidate.read_coefficients[0] = 9.0
        candidate.support_offsets[1] = 2
        return snapshot

    monkeypatch.setattr(
        body_routes,
        "_authenticated_body_route_fold",
        authenticate_then_mutate,
    )
    raced = sparse_feature_fold_map_from_body_route(
        raced_fold,
        coefficient_set="read",
        admission_binding={"compiler_plan_sha256": "c" * 64},
    )
    assert raced_fold.read_coefficients[0].item() == 9.0
    assert raced_fold.support_offsets.tolist() == [0, 2, 2]
    assert raced.map_sha256 == baseline.map_sha256
    assert raced.admission_binding_sha256 == baseline.admission_binding_sha256
    assert torch.equal(raced.coefficients, baseline.coefficients)
    assert torch.equal(raced.support_offsets, baseline.support_offsets)


def test_sparse_feature_authentication_clones_before_tensor_hash_callbacks(
    monkeypatch,
):
    _, clean_fold = _exact_sparse_adapter_fixture("reader")
    baseline = sparse_feature_fold_map_from_body_route(
        clean_fold,
        coefficient_set="read",
        admission_binding={"compiler_plan_sha256": "d" * 64},
    )
    _, raced_fold = _exact_sparse_adapter_fixture("reader")
    tensor_sha256 = body_routes._tensor_sha256
    hash_calls = 0

    def hash_then_mutate_caller(value):
        nonlocal hash_calls
        digest = tensor_sha256(value)
        hash_calls += 1
        # The fifth sealed-route hash is the non-installed write lane.  Mutate
        # every original route boundary immediately after that digest, while
        # authentication is still executing.
        if hash_calls == 5:
            raced_fold.write_coefficients[0] = 9.0
            raced_fold.support_offsets[1] = 2
            raced_fold.support_channel_ids[0] = 1
            raced_fold.expansion_route.zero_()
        return digest

    monkeypatch.setattr(body_routes, "_tensor_sha256", hash_then_mutate_caller)
    raced = sparse_feature_fold_map_from_body_route(
        raced_fold,
        coefficient_set="read",
        admission_binding={"compiler_plan_sha256": "d" * 64},
    )
    assert hash_calls == 7
    assert raced_fold.write_coefficients[0].item() == 9.0
    assert raced_fold.support_offsets.tolist() == [0, 2, 2]
    assert raced_fold.support_channel_ids.tolist() == [1, 1]
    assert not bool(raced_fold.expansion_route.any().item())
    assert raced.map_sha256 == baseline.map_sha256
    assert raced.admission_binding_sha256 == baseline.admission_binding_sha256
    assert torch.equal(raced.coefficients, baseline.coefficients)
    assert torch.equal(raced.support_offsets, baseline.support_offsets)
    assert torch.equal(raced.support_target_ids, baseline.support_target_ids)


def test_sparse_feature_adapter_is_deterministic_and_provenance_bound():
    _, fold = _exact_sparse_adapter_fixture("writer")
    first = sparse_feature_fold_map_from_body_route(
        fold,
        coefficient_set="write",
        admission_binding={"compiler_plan_sha256": "b" * 64},
    )
    second = sparse_feature_fold_map_from_body_route(
        fold,
        coefficient_set="write",
        admission_binding={"compiler_plan_sha256": "b" * 64},
    )
    read_lane = sparse_feature_fold_map_from_body_route(
        fold,
        coefficient_set="read",
        admission_binding={"compiler_plan_sha256": "b" * 64},
    )
    assert first.map_sha256 == second.map_sha256
    assert first.admission_binding_sha256 == second.admission_binding_sha256
    assert first.map_sha256 != read_lane.map_sha256
    assert first.admission_binding_sha256 != read_lane.admission_binding_sha256

    changed_record = dict(fold.record)
    changed_record["site"] = "L99"
    changed_record.pop("record_sha256")
    changed_record["record_sha256"] = canonical_sha256(changed_record)
    changed_origin = replace(
        fold,
        site="L99",
        record=MappingProxyType(changed_record),
    )
    changed_map = sparse_feature_fold_map_from_body_route(
        changed_origin,
        coefficient_set="write",
        admission_binding={"compiler_plan_sha256": "b" * 64},
    )
    assert first.map_sha256 != changed_map.map_sha256
    assert first.admission_binding_sha256 != changed_map.admission_binding_sha256

    with pytest.raises(T5BodyRouteAuthenticationError, match="reserved origin key"):
        sparse_feature_fold_map_from_body_route(
            fold,
            coefficient_set="write",
            admission_binding={"body_route_record_sha256": "0" * 64},
        )
    with pytest.raises(FeatureFoldError, match="external seal"):
        first.verify(expected_map_sha256="0" * 64)


def test_sparse_feature_adapter_rejects_tampering_and_requires_lane_choice():
    _, fold = _exact_sparse_adapter_fixture("reader")
    tampered_record = dict(fold.record)
    tampered_record["site"] = "tampered"
    tampered = replace(fold, record=MappingProxyType(tampered_record))
    with pytest.raises(T5BodyRouteAuthenticationError, match="seal is invalid"):
        sparse_feature_fold_map_from_body_route(
            tampered,
            coefficient_set="read",
        )
    with pytest.raises(T5BodyRouteGeometryError, match="explicitly"):
        sparse_feature_fold_map_from_body_route(fold, coefficient_set="installed")


def test_writer_block_count_is_refused():
    with pytest.raises(T5BodyRouteGeometryError, match="single component axis"):
        fit_body_component_routes(
            parent_weight=_normal(34, 32, 64),
            component_rows=_captured_evidence(35, 120, 64, 20),
            component="mlp_down",
            level="M8",
            block_count=2,
        )


# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------


def test_pivoted_qr_and_evidence_energy_are_both_deterministic_and_sorted():
    rows = _captured_evidence(36, 200, 48, 24)
    for policy in ("evidence-energy", "pivoted-qr"):
        first = select_kept_channels(rows, kept_width=20, policy=policy)
        second = select_kept_channels(rows, kept_width=20, policy=policy)
        assert torch.equal(first, second)
        assert torch.equal(first, torch.sort(first).values)
        assert int(first.numel()) == 20


def test_pivoted_qr_refuses_beyond_the_evidence_rank():
    rows = _captured_evidence(37, 200, 48, 10, noise=0.0)
    with pytest.raises(T5BodyRouteConditioningError, match="exhausted"):
        select_kept_channels(rows, kept_width=40, policy="pivoted-qr")


def test_explicit_channel_ids_are_honoured_and_validated():
    rows = _captured_evidence(38, 120, 32, 16)
    chosen = select_kept_channels(
        rows, kept_width=4, policy="explicit", explicit_ids=[7, 2, 19, 30]
    )
    assert chosen.tolist() == [2, 7, 19, 30]
    with pytest.raises(T5BodyRouteGeometryError, match="outside the component"):
        select_kept_channels(
            rows, kept_width=2, policy="explicit", explicit_ids=[1, 99]
        )
    with pytest.raises(T5BodyRouteGeometryError, match="requires kept_channel_ids"):
        select_kept_channels(rows, kept_width=2, policy="explicit")


# ---------------------------------------------------------------------------
# The refusal ladder
# ---------------------------------------------------------------------------


def test_autograd_state_is_refused():
    rows = _captured_evidence(39, 120, 64, 20).requires_grad_(True)
    with pytest.raises(T5BodyRouteDisciplineError, match="fold-only"):
        fit_body_component_routes(
            parent_weight=_normal(40, 32, 64),
            component_rows=rows,
            component="mlp_down",
            level="M8",
        )


def test_non_finite_evidence_is_refused():
    rows = _captured_evidence(41, 120, 64, 20)
    rows[3, 5] = float("inf")
    with pytest.raises(T5BodyRouteEvidenceError, match="not finite"):
        fit_body_component_routes(
            parent_weight=_normal(42, 32, 64),
            component_rows=rows,
            component="mlp_down",
            level="M8",
        )


def test_evidence_width_mismatch_is_refused():
    with pytest.raises(T5BodyRouteEvidenceError, match="component axis width"):
        fit_body_component_routes(
            parent_weight=_normal(43, 32, 64),
            component_rows=_captured_evidence(44, 120, 48, 20),
            component="mlp_down",
            level="M8",
        )


def test_dead_kept_channel_is_refused_as_a_conditioning_failure():
    rows = _captured_evidence(45, 200, 16, 8)
    rows[:, :] = rows
    dead = torch.zeros(200, 4, dtype=torch.float64)
    evidence = torch.cat((rows, dead), dim=1)
    with pytest.raises(T5BodyRouteConditioningError, match="zero evidence energy"):
        fit_body_component_routes(
            parent_weight=_normal(46, 8, 20),
            component_rows=evidence,
            component="mlp_down",
            level="M4",
            channel_selection="explicit",
            kept_channel_ids=[0, 1, 16, 17, 18, 19, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            active_k=4,
        )


def test_unsupported_block_reader_geometry_is_refused():
    with pytest.raises(T5BodyRouteGeometryError, match="not divisible"):
        fit_body_component_routes(
            parent_weight=_normal(47, 97, 32),
            component_rows=_captured_evidence(48, 120, 32, 16),
            component="mlp_gate",
            level="M8",
            block_count=3,
        )


# ---------------------------------------------------------------------------
# Sealing and the registry contract
# ---------------------------------------------------------------------------


def test_record_seals_and_detects_tampering():
    fold = fit_body_component_routes(
        parent_weight=_normal(49, 32, 64),
        component_rows=_captured_evidence(50, 200, 64, 24),
        component="mlp_down",
        level="M16",
        site="L05",
    )
    verified = verify_sealed_body_route_record(fold.record)
    assert verified["schema"] == T5_BODY_ROUTE_RECORD_SCHEMA
    assert verified["component_id"] == "C13"
    assert verified["record_sha256"] == fold.record_sha256
    assert verified["operation_audit"]["backward_calls"] == 0
    assert verified["operation_audit"]["parent_forward_calls"] == 0
    assert verified["support_offsets_sha256"] == body_routes._tensor_sha256(
        fold.support_offsets
    )
    assert verified["expansion_route_sha256"] == body_routes._tensor_sha256(
        fold.expansion_route
    )
    tampered = dict(fold.record)
    tampered["kept_width"] = int(tampered["kept_width"]) - 1
    with pytest.raises(Exception, match="seal is invalid"):
        verify_sealed_body_route_record(tampered)


def test_records_are_reproducible_for_the_same_request():
    request = {
        "parent_weight": _normal(51, 32, 64),
        "component_rows": _captured_evidence(52, 200, 64, 24),
        "component": "C13",
        "level": "M16",
        "site": "L05",
    }
    first = fit_from_request(MappingProxyType(dict(request)))
    second = fit_from_request(MappingProxyType(dict(request)))
    assert (
        first["technique_record"]["record_sha256"]
        == second["technique_record"]["record_sha256"]
    )


def test_registry_entry_matches_the_dispatch_contract():
    entry = matrix_technique_registry_entry()
    assert entry.schema == MATRIX_TECHNIQUE_ENTRY_SCHEMA
    assert entry.technique_id == "T5"
    assert entry.technique == "sparse_omp_routes"
    assert "C13" in entry.components and "C05" in entry.components
    assert entry.levels == ("M16", "M8", "M4", "X4")
    description = entry.describe()
    assert description["fold_only"] is True
    assert description["record_schema"] == T5_BODY_ROUTE_RECORD_SCHEMA
    result = entry.fit(
        {
            "parent_weight": _normal(53, 32, 64),
            "component_rows": _captured_evidence(54, 200, 64, 24),
            "component": "C13",
            "level": "M16",
        }
    )
    assert result["technique_id"] == "T5"
    assert set(result) >= {
        "child_parameters",
        "route",
        "apply",
        "calibration",
        "solve",
        "technique_record",
    }
    assert result["solve"]["omp"]["double_mgs"] is True
    assert callable(result["apply"])


def test_registry_request_validation_closes():
    entry = matrix_technique_registry_entry()
    with pytest.raises(T5BodyRouteError, match="Missing T5 request keys"):
        entry.fit({"parent_weight": _normal(55, 8, 16)})
    with pytest.raises(T5BodyRouteError, match="Unsupported T5 request keys"):
        entry.fit(
            {
                "parent_weight": _normal(56, 8, 16),
                "component_rows": _captured_evidence(57, 40, 16, 8),
                "component": "C13",
                "level": "M8",
                "learning_rate": 0.1,
            }
        )
    with pytest.raises(T5BodyRouteError, match="must be a mapping"):
        entry.fit(["not", "a", "mapping"])
