from __future__ import annotations

import copy
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_cpcf_layer_compiler as layer_compiler  # noqa: E402
from neural_forge.hf_lfm2_cpcf_short_conv_fold import (  # noqa: E402
    Lfm2CpcfShortConvGatePolicy,
    Lfm2CpcfShortConvValidationError,
)
from neural_forge.hf_lfm2_cpcf_candidate_replay import (  # noqa: E402
    CpuReferenceLfm2CpcfCandidateReplayEvaluator,
    Lfm2CpcfCandidateReplayError,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (  # noqa: E402
    Lfm2CpcfShortConvV4SolverPolicy,
    causal_depthwise_context,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4_fold import (  # noqa: E402
    LFM2_CPCF_SHORT_CONV_V4_FAILURE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_FOLD_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FAILURE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FOLD_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_POOL_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_PROVENANCE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FAILURE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FOLD_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_SELECTOR_RECEIPT_SCHEMA,
    Lfm2CpcfShortConvNumericalFrontier,
    Lfm2CpcfShortConvNumericalFrontierRequest,
    Lfm2CpcfShortConvNumericalRawCandidate,
    Lfm2CpcfShortConvV4FoldCase,
    Lfm2CpcfShortConvV4FoldError,
    _sha,
    compile_lfm2_cpcf_short_conv_v4_fold,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5 import (  # noqa: E402
    LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
    rescore_sealed_v4_short_conv_selection_frontier,
)


def _fixture(
    *,
    validation_offset: float = 0.0,
    target_dtype: Any = torch.float32,
):
    generator = torch.Generator().manual_seed(7019)
    source_hidden = 4
    target_hidden = 2
    fir_length = 3
    source_in = (
        torch.randn(
            (3 * source_hidden, source_hidden),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.25
    )
    source_in_bias = (
        torch.randn(
            (3 * source_hidden,),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.05
    )
    source_fir = (
        torch.randn(
            (source_hidden, fir_length),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.25
    )
    source_fir_bias = (
        torch.randn(
            (source_hidden,),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.03
    )
    source_out = (
        torch.randn(
            (source_hidden, source_hidden),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.2
    )
    source_out_bias = (
        torch.randn(
            (source_hidden,),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.02
    )
    input_lift = torch.randn(
        (target_hidden, source_hidden),
        generator=generator,
        dtype=torch.float64,
    )
    input_lift_bias = torch.randn(
        (source_hidden,),
        generator=generator,
        dtype=torch.float64,
    )
    output_encoder = torch.randn(
        (source_hidden, target_hidden),
        generator=generator,
        dtype=torch.float64,
    )
    cases = []
    originals = []
    for ordinal, split in enumerate(("fit", "fit", "selection", "validation")):
        child = torch.randn(
            (2, 9 + ordinal, target_hidden),
            generator=generator,
            dtype=torch.float64,
        )
        parent_input = child @ input_lift + input_lift_bias
        projected = parent_input @ source_in.T + source_in_bias
        b_value, c_value, x_value = projected.chunk(3, dim=-1)
        context = causal_depthwise_context(
            b_value,
            c_value,
            x_value,
            source_fir,
            source_fir_bias,
        )
        parent_write = context @ source_out.T + source_out_bias
        actual_pre = torch.randn(
            (2, int(child.shape[1]), target_hidden),
            generator=generator,
            dtype=torch.float64,
        )
        endpoint = actual_pre + parent_write @ output_encoder
        if split == "validation":
            endpoint = endpoint + validation_offset
        mask = torch.ones(
            tuple(child.shape[:2]),
            dtype=torch.bool,
        )
        originals.append(child)
        cases.append(
            Lfm2CpcfShortConvV4FoldCase(
                case_id=f"{split}-{ordinal}",
                split=split,
                suffix_start=0,
                actual_child_operator_input_history=child,
                parent_operator_input_history=parent_input,
                parent_b_history=b_value,
                parent_c_history=c_value,
                parent_x_history=x_value,
                parent_post_operator_residual_target_space_suffix=endpoint,
                actual_child_pre_operator_residual_suffix=actual_pre,
                valid_prefix_mask=mask,
            )
        )
    templates = {
        "target_in_proj_weight_template": torch.zeros(
            (3 * target_hidden, target_hidden),
            dtype=target_dtype,
        ),
        "target_in_proj_bias_template": torch.zeros(
            (3 * target_hidden,),
            dtype=target_dtype,
        ),
        "target_conv_weight_template": torch.zeros(
            (target_hidden, 1, fir_length),
            dtype=target_dtype,
        ),
        "target_conv_bias_template": torch.zeros(
            (target_hidden,),
            dtype=target_dtype,
        ),
        "target_out_proj_weight_template": torch.zeros(
            (target_hidden, target_hidden),
            dtype=target_dtype,
        ),
        "target_out_proj_bias_template": torch.zeros(
            (target_hidden,),
            dtype=target_dtype,
        ),
    }
    return {
        "source_in_proj_weight": source_in,
        "source_in_proj_bias": source_in_bias,
        "source_fir_weight": source_fir[:, None, :],
        "source_fir_bias": source_fir_bias,
        "source_out_proj_weight": source_out,
        "source_out_proj_bias": source_out_bias,
        "output_encoder": output_encoder,
        "cases": tuple(cases),
        "originals": originals,
        **templates,
    }


def _solver_policy(
    target_dtype: str = "float32",
) -> Lfm2CpcfShortConvV4SolverPolicy:
    return Lfm2CpcfShortConvV4SolverPolicy(
        ridge_alphas=(1e-6,),
        variable_projection_passes=2,
        target_dtype=target_dtype,
    )


def _permissive_gate() -> Lfm2CpcfShortConvGatePolicy:
    return Lfm2CpcfShortConvGatePolicy(
        maximum_fit_endpoint_relative_rmse=1e6,
        maximum_validation_endpoint_relative_rmse=1e6,
        minimum_fit_endpoint_cosine_similarity=-1.0,
        minimum_validation_endpoint_cosine_similarity=-1.0,
        enforce=True,
    )


def _compile(
    fixture,
    *,
    gate=None,
    candidate_replay_evaluator=None,
    candidate_selector=None,
    numerical_frontier_adapter=None,
):
    values = {key: value for key, value in fixture.items() if key != "originals"}
    dtype_name = str(fixture["target_in_proj_weight_template"].dtype).removeprefix(
        "torch."
    )
    return compile_lfm2_cpcf_short_conv_v4_fold(
        **values,
        solver_policy=_solver_policy(dtype_name),
        gate_policy=_permissive_gate() if gate is None else gate,
        candidate_replay_evaluator=candidate_replay_evaluator,
        candidate_selector=candidate_selector,
        numerical_frontier_adapter=numerical_frontier_adapter,
    )


class _DeterministicToyNumericalFrontierAdapter:
    def __init__(self) -> None:
        self.requests = []
        self.frontier = None

    def generate(
        self,
        request: Lfm2CpcfShortConvNumericalFrontierRequest,
    ) -> Lfm2CpcfShortConvNumericalFrontier:
        self.requests.append(request)
        source_hidden = int(request.report["source_hidden_size"])
        target_hidden = int(request.report["target_hidden_size"])
        context_encoder = torch.zeros(
            (source_hidden, target_hidden),
            dtype=torch.float64,
        )
        for source_index in range(source_hidden):
            context_encoder[
                source_index,
                source_index % target_hidden,
            ] = 1.0

        def value(role: str, amount: float):
            template = request.target_templates[role]
            return (
                None
                if template is None
                else torch.full_like(
                    template,
                    amount,
                    dtype=torch.float64,
                )
            )

        candidates = tuple(
            Lfm2CpcfShortConvNumericalRawCandidate(
                request_sha256=request.request_sha256,
                candidate_id=candidate_id,
                candidate_kind="deterministic_test_alternate",
                in_proj_weight=value("in_proj_weight", amount),
                in_proj_bias=value("in_proj_bias", amount),
                fir_kernel=value("fir_kernel", amount),
                fir_bias=value("fir_bias", amount),
                writer_weight=value("writer_weight", amount),
                writer_bias=value("writer_bias", amount),
                method_provenance={
                    "method": "deterministic_constant_test_frontier",
                    "constant": amount,
                },
            )
            for candidate_id, amount in (
                ("toy-alternate-a", 0.03125),
                ("toy-alternate-z", -0.0625),
            )
        )
        self.frontier = Lfm2CpcfShortConvNumericalFrontier(
            adapter_id="tests.deterministic_toy_frontier",
            request_sha256=request.request_sha256,
            context_encoder=context_encoder,
            raw_candidates=candidates,
            fit_case_authentication_sha256=tuple(
                request.report["fit_case_authentication_sha256"]
            ),
            adapter_provenance={
                "implementation": "tests.deterministic_toy_frontier.v1",
                "real_v5_numerical_algorithm": False,
            },
        )
        return self.frontier


def test_v4_orchestrator_compiles_target_shaped_state_and_seals_roles() -> None:
    fixture = _fixture()
    artifact = _compile(fixture)

    assert artifact.report["schema"] == LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA
    assert artifact.behavior_gate_passed
    assert artifact.full_source_coverage
    assert artifact.report["selected_candidate_id"].startswith("ridge-alpha-00")
    assert artifact.report["fit_only_parameter_solves"] is True
    assert artifact.report["selection_used_only_for_candidate_ranking"] is True
    assert artifact.report["validation_used_only_after_candidate_freeze"] is True
    assert artifact.report["selection_ranking"]["context_state_evidence_ranked"] is True
    assert (
        artifact.report["selection_ranking"][
            "gate_feasibility_ranked_before_metric_optimization"
        ]
        is True
    )
    pool = artifact.report["candidate_pool_report"]
    assert pool["accepted_phase_anchors_included"] is True
    assert pool["rejected_phase_states_included"] is False
    assert pool["prior_to_final_trust_path_fractions"] == (
        1 / 64,
        1 / 32,
        1 / 16,
        1 / 8,
        1 / 4,
        1 / 2,
    )
    assert pool["prior_to_first_writer_trust_path_fractions"] == (
        1 / 64,
        1 / 32,
        1 / 16,
        1 / 8,
        1 / 4,
        1 / 2,
    )
    ridge_report = next(iter(artifact.report["ridge_fit_reports"].values()))
    assert pool["raw_candidate_count"] == ridge_report["path_state_count"] + 12
    assert pool["unique_target_dtype_candidate_count"] == len(
        artifact.report["candidate_selection_reports"]
    )
    assert set(artifact.report["candidate_fit_reports"]) == set(
        artifact.report["candidate_selection_reports"]
    )
    selected_provenance = artifact.report["candidate_fit_reports"][
        artifact.report["selected_candidate_id"]
    ]
    assert selected_provenance["fit_only"] is True
    assert selected_provenance["selection_cases_consumed"] == 0
    assert selected_provenance["validation_cases_consumed"] == 0
    assert artifact.report["fit_evaluation"]["context_evidence_present"] is True
    assert artifact.report["validation_evaluation"]["context_evidence_present"] is True
    assert (
        artifact.report["behavior_gate"]["checks"]["fit_context_relative_rmse"] is True
    )
    assert (
        artifact.report["behavior_gate"]["checks"]["validation_context_relative_rmse"]
        is True
    )
    assert artifact.report["execution"]["training_steps"] == 0
    assert artifact.report["execution"]["parameters_pruned"] == 0
    assert artifact.report["execution"]["parameters_quantized"] == 0
    state = artifact.state_dict()
    assert tuple(state["in_proj.weight"].shape) == (6, 2)
    assert tuple(state["conv.weight"].shape) == (2, 1, 3)
    assert tuple(state["out_proj.weight"].shape) == (2, 2)
    assert all(value.dtype == torch.float32 for value in state.values())
    state["in_proj.weight"].zero_()
    assert artifact.verify_authentication()
    artifact.in_proj_weight.zero_()
    assert not artifact.verify_authentication()
    with pytest.raises(
        Lfm2CpcfShortConvV4FoldError,
        match="changed after authentication",
    ):
        artifact.as_dict()


def test_injected_replay_uses_explicit_v5_receipt_without_changing_math() -> None:
    fixture = _fixture()
    legacy = _compile(fixture)
    scheduled = _compile(
        fixture,
        candidate_replay_evaluator=(
            CpuReferenceLfm2CpcfCandidateReplayEvaluator()
        ),
    )

    assert legacy.report["schema"] == LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA
    assert scheduled.report["schema"] == LFM2_CPCF_SHORT_CONV_V5_FOLD_SCHEMA
    assert scheduled.report["legacy_v4_report_claimed"] is False
    assert (
        scheduled.report["candidate_replay_backend"]["kind"]
        == "injected_cpu_reference"
    )
    assert (
        scheduled.report["candidate_replay_backend_report_sha256"]
        == scheduled.report["candidate_replay_backend"]["report_sha256"]
    )
    assert (
        scheduled.report["candidate_pool_report"][
            "scheduled_replay_changes_closed_form_solve_math"
        ]
        is False
    )
    assert (
        scheduled.report["candidate_selection_reports"]
        == legacy.report["candidate_selection_reports"]
    )
    assert scheduled.report["selection_ranking"] == legacy.report[
        "selection_ranking"
    ]
    assert scheduled.report["selected_candidate_id"] == legacy.report[
        "selected_candidate_id"
    ]
    for name, value in legacy.state_dict().items():
        assert torch.equal(value, scheduled.state_dict()[name])
    assert scheduled.verify_authentication()


def test_default_none_selector_is_byte_exact_v4() -> None:
    fixture = _fixture()
    implicit = _compile(fixture)
    explicit = _compile(
        fixture,
        candidate_selector=None,
        numerical_frontier_adapter=None,
    )

    assert implicit.as_dict() == explicit.as_dict()
    assert implicit.report["schema"] == LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA
    for name, value in implicit.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])


def test_typed_numerical_frontier_reuses_replay_and_installs_candidate() -> None:
    adapter = _DeterministicToyNumericalFrontierAdapter()
    artifact = _compile(
        _fixture(),
        numerical_frontier_adapter=adapter,
    )

    assert len(adapter.requests) == 1
    request = adapter.requests[0]
    assert {case.split for case in request.cases} == {"fit"}
    assert set(request.parent_contexts) == {
        case.case_id for case in request.cases
    }
    assert request.report["selection_cases_exposed"] is False
    assert request.report["validation_cases_exposed"] is False
    assert request.verify_authentication()
    assert adapter.frontier.verify_authentication()
    assert artifact.report["schema"] == (
        LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FOLD_SCHEMA
    )
    assert artifact.report["numerical_frontier_adapter_active"] is True
    assert artifact.report["legacy_v4_numerical_generation_claimed"] is False
    assert artifact.report["legacy_v4_report_claimed"] is False
    assert artifact.report["candidate_pool_report"]["schema"] == (
        LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_POOL_SCHEMA
    )
    assert all(
        report["schema"]
        == LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_PROVENANCE_SCHEMA
        for report in artifact.report["candidate_fit_reports"].values()
    )
    selected_id = artifact.report["selected_candidate_id"]
    selected_raw = next(
        candidate
        for candidate in adapter.frontier.raw_candidates
        if candidate.candidate_id == selected_id
    )
    selected_state = selected_raw.state_dict()
    installed_roles = {
        "in_proj.weight": "in_proj_weight",
        "in_proj.bias": "in_proj_bias",
        "conv.weight": "fir_kernel",
        "conv.bias": "fir_bias",
        "out_proj.weight": "writer_weight",
        "out_proj.bias": "writer_bias",
    }
    for installed_name, raw_role in installed_roles.items():
        expected = selected_state[raw_role]
        if installed_name == "conv.weight":
            expected = expected.unsqueeze(1)
        assert torch.equal(
            artifact.state_dict()[installed_name],
            expected.to(dtype=torch.float32),
        )
    assert artifact.verify_authentication()


def test_numerical_frontier_request_mutation_fails_closed() -> None:
    class MutatingAdapter:
        def generate(self, request):
            request.source_tensors["source_fir_kernel"].zero_()
            raise RuntimeError("mutated detached request")

    with pytest.raises(
        Lfm2CpcfShortConvV4FoldError,
        match="changed its detached request before failing",
    ):
        _compile(
            _fixture(),
            numerical_frontier_adapter=MutatingAdapter(),
        )


def test_preinstall_selector_rescores_retained_states_before_install() -> None:
    class EndpointFirstSelector:
        def __init__(self) -> None:
            self.requests = []

        def select(self, request):
            self.requests.append(copy.deepcopy(request))
            return rescore_sealed_v4_short_conv_selection_frontier(
                request["candidate_selection_reports"]
            )

    selector = EndpointFirstSelector()
    artifact = _compile(
        _fixture(),
        candidate_selector=selector,
    )

    assert len(selector.requests) == 1
    request = selector.requests[0]
    assert request["candidate_state_tensors_exposed_to_selector"] is False
    assert request["fit_reports_exposed_to_selector"] is False
    assert request["validation_reports_exposed_to_selector"] is False
    assert artifact.report["schema"] == (
        LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FOLD_SCHEMA
    )
    receipt = artifact.report["preinstall_candidate_selector_receipt"]
    assert receipt["schema"] == (
        LFM2_CPCF_SHORT_CONV_V5_SELECTOR_RECEIPT_SCHEMA
    )
    assert receipt["request_sha256"] == request["report_sha256"]
    assert receipt["all_live_candidate_states_verified_before_callback"] is True
    assert receipt["all_live_candidate_states_verified_after_callback"] is True
    assert receipt["live_candidate_state_bytes_unchanged"] is True
    assert receipt["post_install_rescoring"] is False
    assert artifact.report["selected_candidate_id"] == (
        receipt["selected_candidate_id"]
    )
    assert artifact.report["selection_evaluation"][
        "candidate_state_sha256"
    ] == (
        receipt["selected_candidate_state_sha256"]
    )
    assert artifact.report["legacy_v4_report_claimed"] is False
    assert artifact.verify_authentication()


def test_preinstall_selector_can_override_and_install_a_retained_state() -> None:
    class LastCandidateSelector:
        def select(self, request):
            candidate_id = request["candidate_ids"][-1]
            body = {
                "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
                "kind": "authenticated_last_candidate_override",
                "candidate_count": len(request["candidate_ids"]),
                "selection_reports_consumed": len(
                    request["candidate_ids"]
                ),
                "fit_reports_consumed": 0,
                "validation_reports_consumed": 0,
                "validation_accessible": False,
                "selected_candidate_id": candidate_id,
                "selected_candidate_state_sha256": request[
                    "candidate_state_sha256"
                ][candidate_id],
                "execution": {
                    "training_steps": 0,
                    "backward_calls": 0,
                    "optimizer_steps": 0,
                    "parameters_pruned": 0,
                    "parameters_quantized": 0,
                },
            }
            return {**body, "report_sha256": _sha(body)}

    baseline = _compile(_fixture())
    overridden = _compile(
        _fixture(),
        candidate_selector=LastCandidateSelector(),
    )
    receipt = overridden.report["preinstall_candidate_selector_receipt"]

    assert receipt["selection_changed_from_v4"] is True
    assert receipt["default_v4_selected_candidate_id"] == (
        baseline.report["selected_candidate_id"]
    )
    assert overridden.report["selected_candidate_id"] == (
        receipt["selected_candidate_id"]
    )
    assert overridden.report["selection_evaluation"][
        "candidate_state_sha256"
    ] == receipt["selected_candidate_state_sha256"]
    assert overridden.verify_authentication()


def test_preinstall_selector_request_or_state_hash_tamper_fails_closed() -> None:
    class MutatingRequestSelector:
        def select(self, request):
            request["candidate_ids"].append("forged-candidate")
            return rescore_sealed_v4_short_conv_selection_frontier(
                request["candidate_selection_reports"]
            )

    with pytest.raises(
        Lfm2CpcfShortConvV4FoldError,
        match="changed its authenticated request",
    ):
        _compile(
            _fixture(),
            candidate_selector=MutatingRequestSelector(),
        )

    class WrongStateSelector:
        def select(self, request):
            report = dict(
                rescore_sealed_v4_short_conv_selection_frontier(
                    request["candidate_selection_reports"]
                )
            )
            report.pop("report_sha256")
            report["selected_candidate_state_sha256"] = "f" * 64
            return {**report, "report_sha256": _sha(report)}

    with pytest.raises(
        Lfm2CpcfShortConvV4FoldError,
        match="state-bound v5",
    ):
        _compile(
            _fixture(),
            candidate_selector=WrongStateSelector(),
        )


def test_injected_replay_failure_cannot_publish_a_compiled_artifact() -> None:
    class BrokenEvaluator:
        def evaluate(self, _items):
            raise RuntimeError("synthetic backend failure")

    with pytest.raises(
        Lfm2CpcfCandidateReplayError,
        match="failed before publication",
    ):
        _compile(
            _fixture(),
            candidate_replay_evaluator=BrokenEvaluator(),
        )


def test_cases_snapshot_caller_tensors_and_validation_cannot_select_state() -> None:
    base = _fixture(validation_offset=0.0)
    shifted = _fixture(validation_offset=100.0)
    base_artifact = _compile(base)
    shifted_artifact = _compile(shifted)

    for name, value in base_artifact.state_dict().items():
        assert torch.equal(value, shifted_artifact.state_dict()[name])
    assert (
        base_artifact.report["selection_ranking"]
        == shifted_artifact.report["selection_ranking"]
    )
    base_pool = dict(base_artifact.report["candidate_pool_report"])
    shifted_pool = dict(shifted_artifact.report["candidate_pool_report"])
    for key in (
        "report_sha256",
        "fold_case_authentication_sha256",
        "projection_case_authentication_sha256",
    ):
        base_pool.pop(key)
        shifted_pool.pop(key)
    assert base_pool == shifted_pool
    for key in (
        "fold_case_authentication_sha256",
        "projection_case_authentication_sha256",
    ):
        base_bindings = base_artifact.report["candidate_pool_report"][key]
        shifted_bindings = shifted_artifact.report["candidate_pool_report"][key]
        assert base_bindings["fit"] == shifted_bindings["fit"]
        assert base_bindings["selection"] == shifted_bindings["selection"]
        assert base_bindings["validation"] != shifted_bindings["validation"]
    assert (
        base_artifact.report["candidate_fit_reports"]
        == shifted_artifact.report["candidate_fit_reports"]
    )
    assert (
        base_artifact.report["validation_evaluation"]["report_sha256"]
        != shifted_artifact.report["validation_evaluation"]["report_sha256"]
    )
    case = base["cases"][0]
    before = case.authentication_sha256
    base["originals"][0].zero_()
    assert case.verify_authentication()
    assert case.authentication_sha256 == before


def test_enforced_gate_emits_nonpromotable_self_sealed_failure() -> None:
    fixture = _fixture(validation_offset=10.0)
    strict = Lfm2CpcfShortConvGatePolicy(
        maximum_fit_endpoint_relative_rmse=0.0,
        maximum_validation_endpoint_relative_rmse=0.0,
        minimum_fit_endpoint_cosine_similarity=1.0,
        minimum_validation_endpoint_cosine_similarity=1.0,
        enforce=True,
    )

    with pytest.raises(Lfm2CpcfShortConvValidationError) as raised:
        _compile(fixture, gate=strict)

    report = copy.deepcopy(dict(raised.value.report))
    seal = report.pop("report_sha256")

    assert seal == _sha(report)
    assert report["schema"] == LFM2_CPCF_SHORT_CONV_V4_FAILURE_SCHEMA
    assert report["behavior_gate"]["passed"] is False
    assert report["compiled_short_conv_fold_issued"] is False
    assert report["promotion_claimed"] is False
    assert report["promotable_candidate"] is False
    assert report["failure_evidence_only"] is True


def test_selector_failure_keeps_authenticated_v5_receipt_through_layer_gate() -> None:
    class EndpointFirstSelector:
        def select(self, request):
            return rescore_sealed_v4_short_conv_selection_frontier(
                request["candidate_selection_reports"]
            )

    strict = Lfm2CpcfShortConvGatePolicy(
        maximum_fit_endpoint_relative_rmse=0.0,
        maximum_validation_endpoint_relative_rmse=0.0,
        minimum_fit_endpoint_cosine_similarity=1.0,
        minimum_validation_endpoint_cosine_similarity=1.0,
        enforce=True,
    )
    with pytest.raises(Lfm2CpcfShortConvValidationError) as raised:
        _compile(
            _fixture(validation_offset=10.0),
            gate=strict,
            candidate_selector=EndpointFirstSelector(),
        )

    report = dict(raised.value.report)
    assert report["schema"] == (
        LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FAILURE_SCHEMA
    )
    assert report["preinstall_candidate_selector_active"] is True
    assert report["post_install_rescoring_performed"] is False
    assert layer_compiler._valid_short_conv_validation_report(report)


def test_numerical_frontier_failure_uses_explicit_v5_validator() -> None:
    class LastCandidateSelector:
        def select(self, request):
            candidate_id = request["candidate_ids"][-1]
            body = {
                "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
                "kind": "numerical_frontier_failure_selector",
                "candidate_count": len(request["candidate_ids"]),
                "selection_reports_consumed": len(
                    request["candidate_ids"]
                ),
                "fit_reports_consumed": 0,
                "validation_reports_consumed": 0,
                "validation_accessible": False,
                "selected_candidate_id": candidate_id,
                "selected_candidate_state_sha256": request[
                    "candidate_state_sha256"
                ][candidate_id],
                "execution": {
                    "training_steps": 0,
                    "backward_calls": 0,
                    "optimizer_steps": 0,
                    "parameters_pruned": 0,
                    "parameters_quantized": 0,
                },
            }
            return {**body, "report_sha256": _sha(body)}

    strict = Lfm2CpcfShortConvGatePolicy(
        maximum_fit_endpoint_relative_rmse=0.0,
        maximum_validation_endpoint_relative_rmse=0.0,
        minimum_fit_endpoint_cosine_similarity=1.0,
        minimum_validation_endpoint_cosine_similarity=1.0,
        enforce=True,
    )
    for selector in (None, LastCandidateSelector()):
        with pytest.raises(Lfm2CpcfShortConvValidationError) as raised:
            _compile(
                _fixture(validation_offset=10.0),
                gate=strict,
                numerical_frontier_adapter=(
                    _DeterministicToyNumericalFrontierAdapter()
                ),
                candidate_selector=selector,
            )

        report = dict(raised.value.report)
        assert report["schema"] == (
            LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FAILURE_SCHEMA
        )
        assert report["numerical_frontier_adapter_active"] is True
        assert report["legacy_v4_report_claimed"] is False
        assert (
            "preinstall_candidate_selector_receipt" in report
        ) is (selector is not None)
        assert layer_compiler._valid_short_conv_validation_report(report)

        def reseal_in_place(value):
            value.pop("report_sha256", None)
            value["report_sha256"] = _sha(value)

        def reseal_with(mutator):
            body = copy.deepcopy(report)
            body.pop("report_sha256")
            mutator(body)
            return {**body, "report_sha256": _sha(body)}

        def assert_rejected(mutator):
            assert not layer_compiler._valid_short_conv_validation_report(
                reseal_with(mutator)
            )

        def rebind_frontier(body):
            frontier = body["numerical_frontier"]
            reseal_in_place(frontier)
            frontier_sha = frontier["report_sha256"]
            body["numerical_frontier_report_sha256"] = frontier_sha
            pool = body["candidate_pool_report"]
            pool["numerical_frontier_report_sha256"] = frontier_sha
            reseal_in_place(pool)
            for provenance in body["candidate_fit_reports"].values():
                provenance[
                    "numerical_frontier_report_sha256"
                ] = frontier_sha
                reseal_in_place(provenance)

        def rebind_request(body):
            request = body["numerical_frontier_request"]
            reseal_in_place(request)
            request_sha = request["report_sha256"]
            body["numerical_frontier_request_sha256"] = request_sha
            frontier = body["numerical_frontier"]
            frontier["request_sha256"] = request_sha
            for candidate_id, raw in frontier[
                "raw_candidate_reports"
            ].items():
                raw["request_sha256"] = request_sha
                reseal_in_place(raw)
                frontier["raw_candidate_report_sha256"][candidate_id] = raw[
                    "report_sha256"
                ]
            body["candidate_pool_report"]["request_sha256"] = request_sha
            for provenance in body["candidate_fit_reports"].values():
                provenance["request_sha256"] = request_sha
                for source in provenance["sources"]:
                    raw = source["numerical_raw_candidate_report"]
                    raw["request_sha256"] = request_sha
                    reseal_in_place(raw)
                    source["numerical_raw_candidate_report_sha256"] = raw[
                        "report_sha256"
                    ]
            rebind_frontier(body)

        for execution in ({}, {"training_steps": 0}):
            assert_rejected(
                lambda body, value=execution: body.__setitem__(
                    "execution",
                    value,
                )
            )
        for split in ("fit", "selection", "validation"):
            assert_rejected(
                lambda body, value=split: body[
                    "case_authentication_sha256"
                ].__setitem__(value, [])
            )
        for split in ("selection", "validation"):
            assert_rejected(
                lambda body, value=split: body[
                    "case_authentication_sha256"
                ].__setitem__(value, ["f" * 64])
            )

        def delete_pool_evidence(body, field):
            pool = body["candidate_pool_report"]
            pool.pop(field)
            reseal_in_place(pool)

        for field in (
            "fold_case_authentication_sha256",
            "projection_case_authentication_sha256",
        ):
            assert_rejected(
                lambda body, value=field: delete_pool_evidence(body, value)
            )

        def empty_evaluation(body, split):
            evaluation = body[f"{split}_evaluation"]
            evaluation["cases"] = []
            evaluation["case_count"] = 0
            reseal_in_place(evaluation)

        for split in ("fit", "selection", "validation"):
            assert_rejected(
                lambda body, value=split: empty_evaluation(body, value)
            )

        def mismatch_pool_policy(body):
            pool = body["candidate_pool_report"]
            pool["policy_sha256"] = "e" * 64
            reseal_in_place(pool)

        assert_rejected(mismatch_pool_policy)
        assert_rejected(
            lambda body: body.__setitem__("source_hidden_size", 999)
        )

        def nonzero_provenance_execution(body):
            provenance = next(
                iter(body["candidate_fit_reports"].values())
            )
            provenance["execution"]["training_steps"] = 1
            reseal_in_place(provenance)

        assert_rejected(nonzero_provenance_execution)

        def nonzero_provenance_source_execution(body):
            provenance = next(
                iter(body["candidate_fit_reports"].values())
            )
            source = provenance["sources"][0]
            raw = source["numerical_raw_candidate_report"]
            raw["execution"]["training_steps"] = 1
            reseal_in_place(raw)
            source["numerical_raw_candidate_report_sha256"] = raw[
                "report_sha256"
            ]
            reseal_in_place(provenance)

        assert_rejected(nonzero_provenance_source_execution)

        def nonzero_frontier_execution(body):
            body["numerical_frontier"]["execution"]["training_steps"] = 1
            rebind_frontier(body)

        assert_rejected(nonzero_frontier_execution)

        def incomplete_request_execution(body):
            body["numerical_frontier_request"]["execution"].pop(
                "optimizer_steps"
            )
            rebind_request(body)

        assert_rejected(incomplete_request_execution)

        def nonzero_raw_execution(body):
            frontier = body["numerical_frontier"]
            candidate_id = frontier["raw_candidate_ids"][0]
            raw = frontier["raw_candidate_reports"][candidate_id]
            raw["execution"]["training_steps"] = 1
            reseal_in_place(raw)
            frontier["raw_candidate_report_sha256"][candidate_id] = raw[
                "report_sha256"
            ]
            for provenance in body["candidate_fit_reports"].values():
                for source in provenance["sources"]:
                    if source["candidate_id"] == candidate_id:
                        source["numerical_raw_candidate_report"] = (
                            copy.deepcopy(raw)
                        )
                        source[
                            "numerical_raw_candidate_report_sha256"
                        ] = raw["report_sha256"]
            rebind_frontier(body)

        assert_rejected(nonzero_raw_execution)

        if selector is not None:

            def nonzero_selector_execution(body):
                receipt = body["preinstall_candidate_selector_receipt"]
                selector_report = receipt["selector_report"]
                selector_report["execution"]["training_steps"] = 1
                reseal_in_place(selector_report)
                receipt["selector_report_sha256"] = selector_report[
                    "report_sha256"
                ]
                reseal_in_place(receipt)
                body[
                    "preinstall_candidate_selector_receipt_sha256"
                ] = receipt["report_sha256"]

            assert_rejected(nonzero_selector_execution)


def test_orchestrator_rejects_mutated_authenticated_case() -> None:
    fixture = _fixture()
    fixture["cases"][0].parent_b_history.zero_()

    with pytest.raises(
        Lfm2CpcfShortConvV4FoldError,
        match="changed after authentication",
    ):
        _compile(fixture)


def test_nonzero_suffix_is_aligned_without_using_hidden_prefix_targets() -> None:
    fixture = _fixture()
    suffix_cases = []
    for case in fixture["cases"]:
        suffix_cases.append(
            Lfm2CpcfShortConvV4FoldCase(
                case_id=case.case_id,
                split=case.split,
                suffix_start=2,
                actual_child_operator_input_history=(
                    case.actual_child_operator_input_history
                ),
                parent_operator_input_history=(case.parent_operator_input_history),
                parent_b_history=case.parent_b_history,
                parent_c_history=case.parent_c_history,
                parent_x_history=case.parent_x_history,
                parent_post_operator_residual_target_space_suffix=(
                    case.parent_post_operator_residual_target_space_suffix[:, 2:]
                ),
                actual_child_pre_operator_residual_suffix=(
                    case.actual_child_pre_operator_residual_suffix[:, 2:]
                ),
                valid_prefix_mask=case.valid_prefix_mask,
            )
        )
    fixture["cases"] = tuple(suffix_cases)

    artifact = _compile(fixture)

    assert artifact.verify_authentication()
    assert artifact.report["fit_evaluation"]["case_count"] == 2


def test_orchestrator_materializes_real_bfloat16_parameters_and_arithmetic() -> None:
    artifact = _compile(_fixture(target_dtype=torch.bfloat16))

    assert all(
        value.dtype == torch.bfloat16 for value in artifact.state_dict().values()
    )
    assert (
        artifact.report["fit_evaluation"][
            "actual_parameter_and_intermediate_target_dtype_replay"
        ]
        is True
    )
    assert artifact.report["fit_evaluation"]["target_dtype"] == "bfloat16"


def test_bfloat16_two_ridge_frontier_deduplicates_authenticated_states() -> None:
    fixture = _fixture(target_dtype=torch.bfloat16)
    values = {key: value for key, value in fixture.items() if key != "originals"}
    artifact = compile_lfm2_cpcf_short_conv_v4_fold(
        **values,
        solver_policy=Lfm2CpcfShortConvV4SolverPolicy(
            ridge_alphas=(1e-8, 1e-4),
            variable_projection_passes=2,
            target_dtype="bfloat16",
        ),
        gate_policy=_permissive_gate(),
    )

    pool = artifact.report["candidate_pool_report"]
    provenance = artifact.report["candidate_fit_reports"]
    raw_count = sum(
        report["source_alias_count"] for report in provenance.values()
    )
    assert pool["ridge_candidate_count"] == 2
    assert pool["raw_candidate_count"] == raw_count
    assert pool["raw_candidate_count"] > pool["unique_target_dtype_candidate_count"]
    assert pool["target_dtype_duplicate_count"] == (
        pool["raw_candidate_count"]
        - pool["unique_target_dtype_candidate_count"]
    )
    assert any(
        len({source["ridge_id"] for source in report["sources"]}) > 1
        for report in provenance.values()
    )
    assert all(
        value.dtype == torch.bfloat16 for value in artifact.state_dict().values()
    )
    selected = artifact.report["selected_candidate_id"]
    assert (
        pool["candidate_state_sha256"][selected]
        == artifact.report["selection_evaluation"]["candidate_state_sha256"]
    )
    assert artifact.verify_authentication()


def test_non_enforced_failed_gate_returns_nonpromotable_diagnostic_artifact() -> None:
    fixture = _fixture(validation_offset=10.0)
    diagnostic_gate = Lfm2CpcfShortConvGatePolicy(
        maximum_fit_endpoint_relative_rmse=0.0,
        maximum_validation_endpoint_relative_rmse=0.0,
        minimum_fit_endpoint_cosine_similarity=1.0,
        minimum_validation_endpoint_cosine_similarity=1.0,
        enforce=False,
    )

    artifact = _compile(fixture, gate=diagnostic_gate)

    assert artifact.verify_authentication()
    assert artifact.behavior_gate_passed is False
    assert artifact.report["behavior_gate"]["enforced"] is False
    assert artifact.report["promotable_candidate"] is False
    assert artifact.report["compiled_short_conv_fold_issued"] is True
