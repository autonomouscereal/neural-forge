from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import neural_forge.analytic_coordinate_completion as coordinate_completion
import neural_forge.lfm2_body_global_coordinate_completion as body_completion
from neural_forge.analytic_execution_profile import (
    AnalyticExecutionProfile,
    resolve_analytic_execution_profile,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_weight_complete_basis import (
    FeatureOrientation,
    ResidualFacingOperator,
    compile_role_balanced_operator_geometry,
)
from neural_forge.lfm2_body_global_coordinate_completion import (
    LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_BALANCED_EVIDENCE_SCHEMA,
    LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_BALANCED_RECEIPT_SCHEMA,
    LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_EVIDENCE_SCHEMA,
    LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_RECEIPT_SCHEMA,
    LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECEIPT_SCHEMA,
    Lfm2BodyGlobalCoordinateCompletion,
    Lfm2BodyGlobalCoordinateCompletionError,
    compile_lfm2_body_global_coordinate_completion,
    lfm2_body_global_coordinate_completion_evidence_sha256,
)
from neural_forge.lfm2_circuit_balanced_coordinate_recipe import (
    LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE,
)


_RECIPE = "bridge-procrustes-isometric"


def _case(
    *,
    target_width: int = 3,
    source_width: int = 7,
    trace_rank: int = 2,
    seed: int = 4100,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    child_basis = torch.linalg.qr(
        torch.randn(
            target_width,
            target_width,
            generator=generator,
            dtype=torch.float64,
        )
    ).Q
    child_rows = torch.cat(
        (
            child_basis[:, :trace_rank].T,
            1.7 * child_basis[:, :trace_rank].T,
        ),
        dim=0,
    ).contiguous()
    desired_decoder = torch.randn(
        target_width,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    parent_rows = (child_rows @ desired_decoder).contiguous()
    producer_encoder = torch.randn(
        source_width,
        target_width,
        generator=generator,
        dtype=torch.float64,
    )
    operators = (
        ResidualFacingOperator(
            role="attention_consumer",
            name="qkv",
            tensor=torch.randn(
                source_width + 2,
                source_width,
                generator=generator,
                dtype=torch.float64,
            ),
            orientation=FeatureOrientation.INPUT,
        ),
        ResidualFacingOperator(
            role="residual_writer",
            name="output",
            tensor=torch.randn(
                source_width,
                source_width + 1,
                generator=generator,
                dtype=torch.float64,
            ),
            orientation=FeatureOrientation.OUTPUT,
        ),
    )
    evidence = lfm2_body_global_coordinate_completion_evidence_sha256(
        child_rows,
        parent_rows,
        producer_encoder,
        operators,
        recipe=_RECIPE,
    )
    return {
        "child_rows": child_rows,
        "parent_rows": parent_rows,
        "producer_encoder": producer_encoder,
        "operators": operators,
        "evidence": evidence,
    }


def _compile(case: dict[str, object]):
    return compile_lfm2_body_global_coordinate_completion(
        case["child_rows"],
        case["parent_rows"],
        case["producer_encoder"],
        case["operators"],
        recipe=_RECIPE,
        expected_evidence_sha256=case["evidence"],
    )


def _geometry_case(
    *,
    target_width: int = 3,
    source_width: int = 7,
    trace_rank: int = 2,
    seed: int = 4100,
    profile: AnalyticExecutionProfile | None = None,
) -> dict[str, object]:
    case = _case(
        target_width=target_width,
        source_width=source_width,
        trace_rank=trace_rank,
        seed=seed,
    )
    geometry = compile_role_balanced_operator_geometry(case["operators"])
    selected_profile = profile or resolve_analytic_execution_profile("cpu")
    evidence = lfm2_body_global_coordinate_completion_evidence_sha256(
        case["child_rows"],
        case["parent_rows"],
        case["producer_encoder"],
        parent_geometry=geometry,
        least_squares_execution_profile=selected_profile,
        recipe=_RECIPE,
    )
    return {
        **case,
        "geometry": geometry,
        "profile": selected_profile,
        "geometry_evidence": evidence,
    }


def _compile_geometry(case: dict[str, object]):
    return compile_lfm2_body_global_coordinate_completion(
        case["child_rows"],
        case["parent_rows"],
        case["producer_encoder"],
        parent_geometry=case["geometry"],
        least_squares_execution_profile=case["profile"],
        recipe=_RECIPE,
        expected_evidence_sha256=case["geometry_evidence"],
    )


def _fake_balanced_evidence(
    monkeypatch: pytest.MonkeyPatch,
    case: dict[str, object],
):
    source_width = int(case["producer_encoder"].shape[0])
    target_width = int(case["producer_encoder"].shape[1])
    generator = torch.Generator().manual_seed(7712)
    basis = torch.linalg.qr(
        torch.randn(
            source_width,
            target_width,
            generator=generator,
            dtype=torch.float64,
        )
    ).Q
    projector = (basis @ basis.T).contiguous()
    projector_sha256 = body_completion._tensor_sha256(projector)
    basis_sha256 = body_completion._tensor_sha256(basis)
    completion_weight = 0.4
    evidence_body = {
        "schema": "test-circuit-balanced-evidence-v1",
        "recipe": LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE,
        "basis_sha256": basis_sha256,
        "source_projector_sha256": projector_sha256,
        "completion_weight": completion_weight,
        "foundation_interface": {
            "producer_encoder_sha256": body_completion._tensor_sha256(
                case["producer_encoder"]
            ),
            "producer_encoder_preserved": True,
            "structural_target_table_preserved": True,
        },
        "rmsnorm_isometry_contract": {
            "orthonormal_basis_isometry": True,
            "orthogonal_projector_only_crosses_norm_boundary": True,
            "nonorthogonal_balanced_factors_cross_rmsnorm": False,
        },
    }
    evidence_receipt = {
        **evidence_body,
        "result_sha256": canonical_sha256(evidence_body),
    }

    class FakeBalancedEvidence:
        def __init__(self):
            self.source_width = source_width
            self.target_width = target_width
            self.source_projector = projector
            self.completion_weight = completion_weight
            self.receipt = evidence_receipt
            self.result_sha256 = evidence_receipt["result_sha256"]
            self.source_projector_sha256 = projector_sha256
            self.balanced_basis = SimpleNamespace(map_sha256=basis_sha256)

        def verify(self):
            return self

        def as_dict(self):
            return dict(self.receipt)

    monkeypatch.setattr(
        body_completion,
        "Lfm2CircuitBalancedCoordinateEvidence",
        FakeBalancedEvidence,
    )
    return FakeBalancedEvidence()


@pytest.fixture(scope="module")
def sealed_geometry_result() -> Lfm2BodyGlobalCoordinateCompletion:
    return _compile_geometry(_geometry_case(seed=4190))


def _unsealed_receipt(
    result: Lfm2BodyGlobalCoordinateCompletion,
) -> dict[str, object]:
    receipt = copy.deepcopy(dict(result.receipt))
    receipt.pop("result_sha256")
    return receipt


def _seal_receipt(receipt: dict[str, object]) -> dict[str, object]:
    return {**receipt, "result_sha256": canonical_sha256(receipt)}


def test_compiles_prior_centred_fit_and_preserves_its_fit_action() -> None:
    case = _case()
    child_rows = case["child_rows"]
    parent_rows = case["parent_rows"]
    encoder = case["producer_encoder"]
    prior = torch.linalg.pinv(encoder, atol=0.0, rtol=1e-12)
    expected_fit = prior + torch.linalg.pinv(
        child_rows,
        atol=0.0,
        rtol=1e-12,
    ) @ (parent_rows - child_rows @ prior)

    result = _compile(case)

    assert result.raw_decoder.shape == expected_fit.shape
    assert torch.linalg.matrix_rank(result.raw_decoder, rtol=1e-12).item() == 3
    assert torch.allclose(
        child_rows @ result.raw_decoder,
        child_rows @ expected_fit,
        rtol=2e-12,
        atol=2e-12,
    )
    assert torch.allclose(
        child_rows @ result.raw_decoder,
        parent_rows,
        rtol=2e-12,
        atol=2e-12,
    )
    assert result.receipt["schema"] == (
        LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECEIPT_SCHEMA
    )
    assert result.receipt["requested_recipe"] == _RECIPE
    assert result.receipt["recipe_selection_resolution"] in {
        "requested-recipe-materialized",
        "requested-recipe-materialized-as-identical-alias",
    }
    if result.receipt["materialized_candidate_recipe"] != _RECIPE:
        assert result.receipt["recipe_selection_resolution"] == (
            "requested-recipe-materialized-as-identical-alias"
        )
    assert result.receipt["full_rank_materialized"] is True
    assert result.receipt["coordinate_map_shape"] == list(result.raw_decoder.shape)
    assert (
        result.receipt["coordinate_map_sha256"]
        == (result.receipt["raw_decoder_sha256"])
    )
    assert (
        result.receipt["public_coordinate_map_sha256"]
        != (result.receipt["coordinate_map_sha256"])
    )
    assert result.receipt["evidence_sha256"] == case["evidence"]
    assert result.receipt["structural_rank"] == 3
    assert result.receipt["fit_partition_only"] is True
    assert result.receipt["quality_score_gate_applied"] is False
    assert result.receipt["model_forward_calls"] == 0
    assert result.receipt["training_steps"] == 0
    assert len(result.receipt["result_sha256"]) == 64
    with pytest.raises(TypeError):
        result.receipt["status"] = "tampered"


def test_legacy_operator_api_remains_byte_identical() -> None:
    case = _case(seed=4200)

    positional = _compile(case)
    explicit_evidence = lfm2_body_global_coordinate_completion_evidence_sha256(
        case["child_rows"],
        case["parent_rows"],
        case["producer_encoder"],
        parent_operators=case["operators"],
        recipe=_RECIPE,
    )
    explicit = compile_lfm2_body_global_coordinate_completion(
        case["child_rows"],
        case["parent_rows"],
        case["producer_encoder"],
        parent_operators=case["operators"],
        recipe=_RECIPE,
        expected_evidence_sha256=explicit_evidence,
    )

    assert explicit_evidence == case["evidence"]
    assert torch.equal(positional.raw_decoder, explicit.raw_decoder)
    assert dict(positional.receipt) == dict(explicit.receipt)
    assert positional.receipt["schema"] == (
        LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_RECEIPT_SCHEMA
    )
    assert "parent_geometry_sha256" not in positional.receipt
    assert "least_squares_execution_profile" not in positional.receipt


def test_preaggregated_geometry_cpu_is_legacy_numerically_equivalent() -> None:
    case = _geometry_case(seed=4250)

    legacy = _compile(case)
    geometry = _compile_geometry(case)

    assert torch.equal(geometry.raw_decoder, legacy.raw_decoder)
    assert geometry.receipt["schema"] == (
        LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_RECEIPT_SCHEMA
    )
    assert geometry.receipt["input_evidence_sha256"] == (case["geometry_evidence"])
    assert case["geometry_evidence"] != case["evidence"]
    assert LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_EVIDENCE_SCHEMA.endswith(
        "evidence-v2"
    )
    parent_geometry = case["geometry"]
    profile = case["profile"]
    assert geometry.receipt["parent_geometry_sha256"] == (
        parent_geometry.geometry_sha256
    )
    assert geometry.receipt["parent_operator_sha256s"] == list(
        parent_geometry.ordered_operator_sha256s
    )
    assert (
        geometry.receipt["parent_raw_gram_sha256"]
        == (parent_geometry.report["hashes"]["raw_gram_sha256"])
    )
    assert (
        geometry.receipt["parent_balanced_gram_sha256"]
        == (parent_geometry.report["hashes"]["balanced_gram_sha256"])
    )
    assert geometry.receipt["least_squares_execution_profile_sha256"] == (
        profile.result_sha256
    )
    assert geometry.receipt["preaggregated_parent_geometry"] is True
    assert geometry.receipt["parent_operator_reconstruction_calls"] == 0
    assert geometry.receipt["least_squares_pseudoinverse_count"] == 2
    assert geometry.receipt["least_squares_normal_equations_used"] is False


def test_circuit_balanced_recipe_adds_authenticated_geometry_without_changing_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _geometry_case(seed=4260)
    balanced = _fake_balanced_evidence(monkeypatch, case)
    evidence = lfm2_body_global_coordinate_completion_evidence_sha256(
        case["child_rows"],
        case["parent_rows"],
        case["producer_encoder"],
        parent_geometry=case["geometry"],
        least_squares_execution_profile=case["profile"],
        circuit_balanced_coordinate_evidence=balanced,
        recipe=LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE,
    )
    result = compile_lfm2_body_global_coordinate_completion(
        case["child_rows"],
        case["parent_rows"],
        case["producer_encoder"],
        parent_geometry=case["geometry"],
        least_squares_execution_profile=case["profile"],
        circuit_balanced_coordinate_evidence=balanced,
        recipe=LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE,
        expected_evidence_sha256=evidence,
    )

    assert result.receipt["schema"] == (
        LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_BALANCED_RECEIPT_SCHEMA
    )
    assert evidence != case["geometry_evidence"]
    assert result.receipt["requested_recipe"] == (
        LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE
    )
    if result.receipt["materialized_candidate_recipe"] == (
        "bridge-procrustes-isometric"
    ):
        assert result.receipt["recipe_selection_resolution"] == (
            "circuit-balanced-source-geometry-plus-bridge-procrustes-materialized"
        )
    else:
        assert result.receipt["recipe_selection_resolution"] == (
            "circuit-balanced-source-geometry-plus-bridge-procrustes-"
            "identical-alias"
        )
    assert result.receipt["circuit_balanced_coordinate_evidence_sha256"] == (
        balanced.result_sha256
    )
    assert result.receipt["circuit_balanced_source_projector_sha256"] == (
        balanced.source_projector_sha256
    )
    assert result.receipt["circuit_balanced_rmsnorm_isometry_bound"] is True
    assert result.receipt["foundation_structural_target_table_preserved"] is True
    assert result.receipt["foundation_producer_encoder_preserved"] is True
    assert result.receipt["nonorthogonal_balanced_factors_cross_rmsnorm"] is False
    fisher = result.frontier.report["parent_geometry"]["cached_fisher"]
    assert fisher["supplied"] is True
    assert fisher["weight"] == balanced.completion_weight
    assert 0.0 <= fisher["positive_semidefinite_shift"] <= 1e-10
    assert torch.allclose(
        case["child_rows"] @ result.raw_decoder,
        case["parent_rows"],
        rtol=2e-12,
        atol=2e-12,
    )
    assert LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_BALANCED_EVIDENCE_SCHEMA.endswith(
        "evidence-v3"
    )


def test_balanced_recipe_and_evidence_are_an_authenticated_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _geometry_case(seed=4261)
    balanced = _fake_balanced_evidence(monkeypatch, case)

    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="must be supplied together",
    ):
        lfm2_body_global_coordinate_completion_evidence_sha256(
            case["child_rows"],
            case["parent_rows"],
            case["producer_encoder"],
            parent_geometry=case["geometry"],
            least_squares_execution_profile=case["profile"],
            recipe=LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE,
        )
    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="must be supplied together",
    ):
        lfm2_body_global_coordinate_completion_evidence_sha256(
            case["child_rows"],
            case["parent_rows"],
            case["producer_encoder"],
            parent_geometry=case["geometry"],
            least_squares_execution_profile=case["profile"],
            circuit_balanced_coordinate_evidence=balanced,
            recipe=_RECIPE,
        )


@pytest.mark.parametrize(
    "field",
    sorted(body_completion._GEOMETRY_RECEIPT_REQUIRED_FIELDS),
)
def test_standalone_v2_wrapper_rejects_each_dropped_geometry_field(
    sealed_geometry_result: Lfm2BodyGlobalCoordinateCompletion,
    field: str,
) -> None:
    receipt = _unsealed_receipt(sealed_geometry_result)
    receipt.pop(field)

    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="geometry receipt",
    ):
        Lfm2BodyGlobalCoordinateCompletion(
            raw_decoder=sealed_geometry_result.raw_decoder,
            frontier=sealed_geometry_result.frontier,
            receipt=_seal_receipt(receipt),
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "geometry_result",
        "geometry_pair",
        "operator_manifest",
        "operator_order",
        "preaggregated_flag",
        "reconstruction_count",
        "raw_gram",
        "balanced_gram",
        "profile_sha256",
        "profile_receipt",
        "requested_device",
        "resolved_device",
        "compute_dtype",
        "pseudoinverse_count",
        "normal_equations",
        "settings_applied",
        "settings_restored",
        "report_device",
        "child_public_rows",
        "parent_public_rows",
        "input_evidence",
    ),
)
def test_standalone_v2_wrapper_rejects_resealed_geometry_contradictions(
    sealed_geometry_result: Lfm2BodyGlobalCoordinateCompletion,
    mutation: str,
) -> None:
    receipt = _unsealed_receipt(sealed_geometry_result)
    replacement_digest = "0" * 64
    if mutation == "geometry_result":
        receipt["parent_geometry_result_sha256"] = replacement_digest
    elif mutation == "geometry_pair":
        receipt["parent_geometry_sha256"] = replacement_digest
        receipt["parent_geometry_result_sha256"] = replacement_digest
    elif mutation == "operator_manifest":
        receipt["parent_operator_manifest_sha256"] = replacement_digest
    elif mutation == "operator_order":
        operator_sha256s = list(reversed(receipt["parent_operator_sha256s"]))
        receipt["parent_operator_sha256s"] = operator_sha256s
        receipt["parent_operator_manifest_sha256"] = canonical_sha256(operator_sha256s)
    elif mutation == "preaggregated_flag":
        receipt["preaggregated_parent_geometry"] = False
    elif mutation == "reconstruction_count":
        receipt["parent_operator_reconstruction_calls"] = 1
    elif mutation == "raw_gram":
        receipt["parent_raw_gram_sha256"] = replacement_digest
    elif mutation == "balanced_gram":
        receipt["parent_balanced_gram_sha256"] = replacement_digest
    elif mutation == "profile_sha256":
        receipt["least_squares_execution_profile_sha256"] = replacement_digest
    elif mutation == "profile_receipt":
        replacement_profile = resolve_analytic_execution_profile(
            "cuda:999999",
            allow_cpu_fallback=True,
        )
        receipt["least_squares_execution_profile"] = replacement_profile.as_dict()
        receipt["least_squares_execution_profile_sha256"] = (
            replacement_profile.result_sha256
        )
        receipt["least_squares_requested_device"] = replacement_profile.requested_device
        receipt["least_squares_resolved_device"] = replacement_profile.resolved_device
    elif mutation == "requested_device":
        receipt["least_squares_requested_device"] = "cuda:999999"
    elif mutation == "resolved_device":
        receipt["least_squares_resolved_device"] = "cuda:999999"
    elif mutation == "compute_dtype":
        receipt["least_squares_compute_dtype"] = "float32"
    elif mutation == "pseudoinverse_count":
        receipt["least_squares_pseudoinverse_count"] = 3
    elif mutation == "normal_equations":
        receipt["least_squares_normal_equations_used"] = True
    elif mutation == "settings_applied":
        receipt["least_squares_deterministic_settings_applied"] = False
    elif mutation == "settings_restored":
        receipt["least_squares_deterministic_settings_restored"] = False
    elif mutation == "report_device":
        receipt["least_squares_canonical_report_device"] = "cuda:0"
    elif mutation == "child_public_rows":
        receipt["fit_child_residual_rows_public_sha256"] = replacement_digest
    elif mutation == "parent_public_rows":
        receipt["fit_parent_residual_rows_public_sha256"] = replacement_digest
    else:
        assert mutation == "input_evidence"
        receipt["input_evidence_sha256"] = replacement_digest
        receipt["evidence_sha256"] = replacement_digest

    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="geometry receipt",
    ):
        Lfm2BodyGlobalCoordinateCompletion(
            raw_decoder=sealed_geometry_result.raw_decoder,
            frontier=sealed_geometry_result.frontier,
            receipt=_seal_receipt(receipt),
        )


def test_geometry_fit_applies_and_restores_cpu_execution_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _geometry_case(seed=4275)
    initial = {
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "benchmark": torch.backends.cudnn.benchmark,
    }
    observations: list[dict[str, object]] = []
    original_pinv = torch.linalg.pinv

    def observed_pinv(*args, **kwargs):
        observations.append(
            {
                "device": args[0].device.type,
                "threads": torch.get_num_threads(),
                "deterministic": (torch.are_deterministic_algorithms_enabled()),
                "warn_only": (torch.is_deterministic_algorithms_warn_only_enabled()),
                "tf32": torch.backends.cuda.matmul.allow_tf32,
                "benchmark": torch.backends.cudnn.benchmark,
                "atol": kwargs.get("atol"),
                "rtol": kwargs.get("rtol"),
                "hermitian": kwargs.get("hermitian"),
            }
        )
        return original_pinv(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "pinv", observed_pinv)
    _compile_geometry(case)

    assert len(observations) == 2
    assert all(value["device"] == "cpu" for value in observations)
    assert all(value["threads"] == 1 for value in observations)
    assert all(value["deterministic"] is True for value in observations)
    assert all(value["warn_only"] is False for value in observations)
    assert all(value["tf32"] is False for value in observations)
    assert all(value["benchmark"] is False for value in observations)
    assert all(value["atol"] == 0.0 for value in observations)
    assert all(value["rtol"] == 1e-12 for value in observations)
    assert all(value["hermitian"] is False for value in observations)
    assert torch.get_num_threads() == initial["threads"]
    assert torch.are_deterministic_algorithms_enabled() == (initial["deterministic"])
    assert (
        torch.is_deterministic_algorithms_warn_only_enabled() == (initial["warn_only"])
    )
    assert torch.backends.cuda.matmul.allow_tf32 == initial["tf32"]
    assert torch.backends.cudnn.benchmark == initial["benchmark"]


def test_geometry_accepts_an_authenticated_explicit_cpu_fallback_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _geometry_case(seed=4290)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    profile = resolve_analytic_execution_profile(
        "cuda:0",
        allow_cpu_fallback=True,
    )
    case["profile"] = profile
    case["geometry_evidence"] = lfm2_body_global_coordinate_completion_evidence_sha256(
        case["child_rows"],
        case["parent_rows"],
        case["producer_encoder"],
        parent_geometry=case["geometry"],
        least_squares_execution_profile=profile,
        recipe=_RECIPE,
    )

    result = _compile_geometry(case)

    assert result.receipt["least_squares_requested_device"] == "cuda:0"
    assert result.receipt["least_squares_resolved_device"] == "cpu"
    assert (
        result.receipt["least_squares_execution_profile"]["cpu_fallback_used"] is True
    )
    assert (
        result.receipt["least_squares_execution_profile"]["cpu_fallback_reason"]
        == "cuda-runtime-unavailable"
    )


def test_geometry_path_rejects_source_xor_profile_and_width_errors() -> None:
    case = _geometry_case(seed=4310)
    common = (
        case["child_rows"],
        case["parent_rows"],
        case["producer_encoder"],
    )
    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="Exactly one",
    ):
        lfm2_body_global_coordinate_completion_evidence_sha256(
            *common,
            recipe=_RECIPE,
        )
    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="Exactly one",
    ):
        lfm2_body_global_coordinate_completion_evidence_sha256(
            *common,
            case["operators"],
            parent_geometry=case["geometry"],
            least_squares_execution_profile=case["profile"],
            recipe=_RECIPE,
        )
    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="requires least_squares_execution_profile",
    ):
        lfm2_body_global_coordinate_completion_evidence_sha256(
            *common,
            parent_geometry=case["geometry"],
            recipe=_RECIPE,
        )
    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="legacy parent_operators path",
    ):
        lfm2_body_global_coordinate_completion_evidence_sha256(
            *common,
            case["operators"],
            least_squares_execution_profile=case["profile"],
            recipe=_RECIPE,
        )

    wrong_width_geometry = _geometry_case(
        source_width=8,
        seed=4311,
    )["geometry"]
    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="geometry width",
    ):
        lfm2_body_global_coordinate_completion_evidence_sha256(
            *common,
            parent_geometry=wrong_width_geometry,
            least_squares_execution_profile=case["profile"],
            recipe=_RECIPE,
        )


def test_geometry_and_profile_tampering_cannot_reuse_evidence() -> None:
    geometry_case = _geometry_case(seed=4330)
    geometry = geometry_case["geometry"]
    geometry.raw_gram[0, 0] += 0.5
    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="geometry authentication failed",
    ):
        _compile_geometry(geometry_case)

    profile_case = _geometry_case(seed=4331)
    profile = profile_case["profile"]
    tampered_profile_receipt = profile.as_dict()
    tampered_profile_receipt["cpu_thread_count"] = 2
    object.__setattr__(profile, "receipt", tampered_profile_receipt)
    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="execution_profile authentication failed",
    ):
        _compile_geometry(profile_case)


def test_geometry_path_never_reconstructs_or_reaggregates_parent_operators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _geometry_case(seed=4350)
    gram_calls = 0

    def forbidden_operator_gram(*_args, **_kwargs):
        nonlocal gram_calls
        gram_calls += 1
        raise AssertionError("geometry path reconstructed an operator Gram")

    def forbidden_operator_snapshot(*_args, **_kwargs):
        raise AssertionError("geometry path reconstructed parent operators")

    monkeypatch.setattr(
        coordinate_completion,
        "_operator_gram",
        forbidden_operator_gram,
    )
    monkeypatch.setattr(
        body_completion,
        "_operator_snapshots",
        forbidden_operator_snapshot,
    )

    result = _compile_geometry(case)

    assert result.receipt["parent_operator_reconstruction_calls"] == 0
    assert gram_calls == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_geometry_fit_matches_cpu_with_tight_fp64_tolerance() -> None:
    cpu_case = _geometry_case(seed=4370)
    cuda_profile = resolve_analytic_execution_profile(
        "cuda:0",
        require_cuda=True,
    )
    cuda_case = {
        **cpu_case,
        "profile": cuda_profile,
    }
    cuda_case["geometry_evidence"] = (
        lfm2_body_global_coordinate_completion_evidence_sha256(
            cuda_case["child_rows"],
            cuda_case["parent_rows"],
            cuda_case["producer_encoder"],
            parent_geometry=cuda_case["geometry"],
            least_squares_execution_profile=cuda_profile,
            recipe=_RECIPE,
        )
    )

    cpu_result = _compile_geometry(cpu_case)
    cuda_result = _compile_geometry(cuda_case)

    assert torch.allclose(
        cuda_result.raw_decoder,
        cpu_result.raw_decoder,
        rtol=5e-10,
        atol=5e-11,
    )
    assert cuda_result.receipt["least_squares_resolved_device"] == "cuda:0"


@pytest.mark.parametrize(
    ("target_width", "source_width", "trace_rank"),
    (
        (3, 9, 2),
        (7, 4, 2),
        (6, 6, 2),
    ),
    ids=(
        "narrow-child-wide-parent",
        "wide-child-narrow-parent",
        "equal-width-cross-family",
    ),
)
def test_architecture_neutral_rectangular_maps_reach_maximum_rank(
    target_width: int,
    source_width: int,
    trace_rank: int,
) -> None:
    case = _case(
        target_width=target_width,
        source_width=source_width,
        trace_rank=trace_rank,
        seed=4300 + target_width * 10 + source_width,
    )

    result = _compile(case)

    assert result.raw_decoder.shape == (target_width, source_width)
    assert torch.linalg.matrix_rank(result.raw_decoder, rtol=1e-12).item() == min(
        target_width,
        source_width,
    )
    assert result.receipt["required_full_rank"] == min(
        target_width,
        source_width,
    )
    assert torch.allclose(
        case["child_rows"] @ result.raw_decoder,
        case["parent_rows"],
        rtol=2e-12,
        atol=2e-12,
    )


def test_operator_order_and_caller_thread_count_do_not_change_result() -> None:
    case = _case(seed=4500)
    initial_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        first = _compile(case)
        reversed_operators = tuple(reversed(case["operators"]))
        reversed_evidence = lfm2_body_global_coordinate_completion_evidence_sha256(
            case["child_rows"],
            case["parent_rows"],
            case["producer_encoder"],
            reversed_operators,
            recipe=_RECIPE,
        )
        torch.set_num_threads(max(2, min(4, initial_threads)))
        second = compile_lfm2_body_global_coordinate_completion(
            case["child_rows"],
            case["parent_rows"],
            case["producer_encoder"],
            reversed_operators,
            recipe=_RECIPE,
            expected_evidence_sha256=reversed_evidence,
        )
    finally:
        torch.set_num_threads(initial_threads)

    assert case["evidence"] == reversed_evidence
    assert torch.equal(first.raw_decoder, second.raw_decoder)
    assert first.receipt["result_sha256"] == second.receipt["result_sha256"]


@pytest.mark.parametrize("field", ("child_rows", "parent_rows", "producer_encoder"))
def test_tampered_cached_tensor_cannot_reuse_evidence(field: str) -> None:
    case = _case(seed=4700)
    changed = dict(case)
    changed[field] = case[field].clone()
    changed[field][0, 0] += 0.125

    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="authentication failed",
    ):
        _compile(changed)


def test_mutated_parent_operator_cannot_reuse_authenticated_evidence() -> None:
    case = _case(seed=4800)
    case["operators"][0].tensor[0, 0] += 0.5

    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="changed after authentication",
    ):
        _compile(case)


def test_recipe_is_explicit_and_unknown_recipe_is_rejected() -> None:
    case = _case(seed=4900)
    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="Unsupported body-global completion recipe",
    ):
        compile_lfm2_body_global_coordinate_completion(
            case["child_rows"],
            case["parent_rows"],
            case["producer_encoder"],
            case["operators"],
            recipe="implicit-best-score",
            expected_evidence_sha256=case["evidence"],
        )


def test_inputs_are_owned_and_never_mutated() -> None:
    case = _case(seed=5000)
    child_before = case["child_rows"].clone()
    parent_before = case["parent_rows"].clone()
    encoder_before = case["producer_encoder"].clone()
    operator_before = tuple(value.tensor.clone() for value in case["operators"])

    result = _compile(case)

    assert torch.equal(case["child_rows"], child_before)
    assert torch.equal(case["parent_rows"], parent_before)
    assert torch.equal(case["producer_encoder"], encoder_before)
    assert all(
        torch.equal(operator.tensor, before)
        for operator, before in zip(
            case["operators"],
            operator_before,
            strict=True,
        )
    )
    assert result.raw_decoder.device.type == "cpu"
    assert result.raw_decoder.dtype == torch.float64
    assert result.raw_decoder.requires_grad is False


def test_result_wrapper_rejects_a_decoder_that_no_longer_matches_its_receipt() -> None:
    result = _compile(_case(seed=5050))
    tampered_decoder = result.raw_decoder.clone()
    tampered_decoder[0, 0] += 0.25

    with pytest.raises(
        Lfm2BodyGlobalCoordinateCompletionError,
        match="does not authenticate raw_decoder",
    ):
        Lfm2BodyGlobalCoordinateCompletion(
            raw_decoder=tampered_decoder,
            frontier=result.frontier,
            receipt=result.receipt,
        )


def test_module_has_no_model_execution_training_or_filesystem_writes() -> None:
    source = Path(
        "src/neural_forge/lfm2_body_global_coordinate_completion.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called_names.intersection(
        {"forward", "generate", "backward", "step", "save", "open"}
    )
    assert not called_attributes.intersection(
        {
            "forward",
            "generate",
            "backward",
            "step",
            "save",
            "write_text",
            "write_bytes",
            "requires_grad_",
        }
    )
    assert "transformers" not in source.lower()
