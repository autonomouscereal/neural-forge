from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_lfm2_cpcf_v5_selective_blank_fold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_cpcf_v5_selective_blank_fold",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _catalog_preparation() -> dict[str, object]:
    digests = {
        name: MODULE._sha({"fixture": name})
        for name in (
            "preparation_receipt_sha256",
            "binding_sha256",
            "artifact_manifest_sha256",
            "artifact_catalog_sha256",
            "selective_catalog_sha256",
            "selective_source_sha256",
            "selective_provider_sha256",
            "codec_sha256",
            "provider_implementation_sha256",
            "artifact_store_source_sha256",
            "selective_provider_source_sha256",
        )
    }
    return {
        "schema": MODULE.SELECTIVE_CATALOG_PREPARATION_SCHEMA,
        "status": "prepared_authenticated_tensor_free_catalog",
        **digests,
        "warm_hit_required_for_production_fold": True,
        "cold_rebuild_permitted_in_production_fold": False,
    }


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--parent-model-directory",
        str(tmp_path / "parent"),
        "--donor-tokenizer-directory",
        str(tmp_path / "donor-tokenizer"),
        "--target-tokenizer-directory",
        str(tmp_path / "target-tokenizer"),
        "--bridge-directory",
        str(tmp_path / "bridge"),
        "--trace-cache-directory",
        str(tmp_path / "cache"),
        "--trajectory-corpus",
        str(tmp_path / "trajectories.json"),
        "--diagnostic-json",
        str(tmp_path / "diagnostic.json"),
        "--target-unique-parameters",
        "2291840",
        "--maximum-process-rss-gib",
        "24",
    ]


def test_module_import_is_cpu_only_and_keeps_heavy_imports_lazy() -> None:
    command = (
        "import runpy,sys;"
        f"runpy.run_path({str(SCRIPT)!r});"
        "forbidden=[name for name in sys.modules "
        "if name=='torch' or name.startswith('transformers') "
        "or name.startswith('neural_forge')];"
        "print(forbidden)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "[]"


def test_parser_reuses_shared_compression_request_and_bounds_memory(
    tmp_path: Path,
) -> None:
    args = MODULE.build_parser().parse_args(_required_args(tmp_path))
    config = MODULE.config_from_args(args)

    assert config.compression.target_unique_parameters == 2_291_840
    assert config.compression.compression_ratio is None
    assert config.maximum_process_rss_bytes == 24 * 1024**3
    assert config.maximum_cuda_gib_per_device == 20.0
    assert config.maximum_cuda_bytes_per_device == 20 * 1024**3
    assert config.maximum_logical_resident_gib == 0.0
    assert config.minimum_physical_rss_reduction_fraction == 0.10
    assert config.model_device == "cuda:0"
    assert config.dtype == "bfloat16"
    assert config.short_conv_v4_ridge_alphas == (1e-8, 1e-4)
    assert config.resume_store_directory is None
    assert config.resume_admission_sha256 is None
    assert config.attention_sequential_transport_profile is None
    assert config.composed_qk_attention_profile is None
    assert config.endpoint_residual_attention_profile is None
    assert config.endpoint_terminal_action_profile is None
    assert config.endpoint_sequential_terminal_action_profile is None
    assert config.forbidden_requests == ()


def test_default_attention_transport_profile_is_exact_legacy_noop() -> None:
    pytest.importorskip("torch")
    from neural_forge.hf_lfm2_cpcf_model_fold import (
        Lfm2CpcfModelFoldPolicy,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
        Lfm2CpcfShortConvV4SolverPolicy,
    )

    args = MODULE.build_parser().parse_args(
        [
            "--parent-model-directory",
            "parent",
            "--donor-tokenizer-directory",
            "donor",
            "--target-tokenizer-directory",
            "target",
            "--bridge-directory",
            "bridge",
            "--trace-cache-directory",
            "cache",
            "--trajectory-corpus",
            "rows.json",
            "--diagnostic-json",
            "diag.json",
            "--target-unique-parameters",
            "2291840",
            "--maximum-process-rss-gib",
            "24",
        ]
    )
    config = MODULE.config_from_args(args)
    config_document = config.as_dict()
    actual = MODULE._build_configured_model_fold_policy(config)
    expected_short_conv = Lfm2CpcfShortConvV4SolverPolicy(
        ridge_alphas=(1e-8, 1e-4),
        variable_projection_passes=2,
        writer_trust_radius=4.0,
        fir_trust_radius=1.0,
        bcx_trust_radius=0.25,
        correction_rms_floor=1e-6,
        context_objective_weight=0.05,
        target_dtype="bfloat16",
    )
    expected = Lfm2CpcfModelFoldPolicy(
        ridge=1e-8,
        state_reconstruction_weight=0.05,
        minimum_predictive_energy_fraction=0.90,
        minimum_validation_block_retention=0.0,
        enforce_validation_block_floor=True,
        short_conv_v4_solver_policy=expected_short_conv,
        promotion_compression_ratio=1.0,
        require_promotion_ratio=True,
    )

    assert MODULE._sha(config_document) == (
        "98ecbc677c2352b457d3aebcfe100223512307d946a78e37265e0b814bfc4c1a"
    )
    assert not set(MODULE._OPT_IN_POLICY_IDENTITY_FIELDS) & set(config_document)
    assert not set(MODULE._COMPOSED_QK_POLICY_IDENTITY_FIELDS) & set(config_document)
    assert not set(MODULE._ENDPOINT_RESIDUAL_POLICY_IDENTITY_FIELDS) & set(
        config_document
    )
    assert not set(MODULE._ENDPOINT_TERMINAL_ACTION_POLICY_IDENTITY_FIELDS) & set(
        config_document
    )
    assert actual.as_dict() == expected.as_dict()
    assert actual.attention_sequential_transport_policy is None
    assert MODULE._configured_model_fold_policy_identity(config) is None
    assert (
        MODULE._build_resume_policy_binding(
            None,
            runtime_policy=SimpleNamespace(as_dict=lambda: {"legacy": True}),
        )
        is None
    )
    assert (
        MODULE._build_composed_qk_runtime_policy_binding(
            None,
            runtime_policy=SimpleNamespace(as_dict=lambda: {"legacy": True}),
        )
        is None
    )


def test_existing_attention_profile_hashes_remain_byte_stable() -> None:
    pytest.importorskip("torch")
    expected = {
        MODULE.V5AttentionSequentialTransportRunnerProfile: (
            "a16010a680cad00e8ceadd596b4aeefe070d2eafa498700213966584ba12d241"
        ),
        MODULE.V5ComposedQkAttentionRunnerProfile: (
            "eee3ef8b8474227c3031147776ad1a8ba5d457e689925d19bbf4f0716f33e3f5"
        ),
        MODULE.V5EndpointResidualAttentionRunnerProfile: (
            "c2ae580fafb3c76b97ba858e50154503e7e8a648e840c5ecc8cfd47c8051af62"
        ),
        MODULE.V5EndpointTerminalActionRunnerProfile: (
            "9ff4adb2c26daaaee4a4843efc438d410642dc23aaa1acf1aab263d1adc999b5"
        ),
    }

    for profile_type, profile_sha256 in expected.items():
        assert profile_type().as_dict()["profile_sha256"] == profile_sha256


def test_opt_in_profile_builds_typed_attention_only_forwarding_policy(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from neural_forge import hf_lfm2_cpcf_v5_model_compiler as v5_compiler
    from neural_forge.hf_lfm2_cpcf_layer_compiler import (
        Lfm2CpcfAttentionSequentialTransportPolicy,
    )

    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--attention-sequential-transport-profile",
            MODULE.ATTENTION_SEQUENTIAL_TRANSPORT_RUNNER_PROFILE_ID,
        ]
    )
    config = MODULE.config_from_args(args)
    policy = MODULE._build_configured_model_fold_policy(config)
    transport = policy.attention_sequential_transport_policy
    config_document = config.as_dict()
    identity = MODULE._configured_model_fold_policy_identity(
        config,
        model_fold_policy=policy,
    )

    assert type(config.attention_sequential_transport_profile) is (
        MODULE.V5AttentionSequentialTransportRunnerProfile
    )
    assert type(transport) is Lfm2CpcfAttentionSequentialTransportPolicy
    assert transport.enabled is True
    assert (
        policy.resolved_attention_sequential_transport_policy("full_attention")
        is transport
    )
    assert policy.resolved_attention_sequential_transport_policy("short_conv") is None
    assert policy.as_dict()["attention_sequential_transport_policy"] == (
        transport.as_dict()
    )
    assert identity is not None
    assert config_document["attention_sequential_transport_profile"] == (
        config.attention_sequential_transport_profile.as_dict()
    )
    assert config_document["complete_model_fold_policy_sha256"] == MODULE._sha(
        policy.as_dict()
    )
    assert (
        config_document["model_fold_policy_identity_sha256"]
        == (identity["identity_sha256"])
    )
    compiler_source = inspect.getsource(
        v5_compiler.compile_lfm2_cpcf_v5_selective_model_fold
    )
    assert "resolved_attention_sequential_transport_policy" in compiler_source
    assert "attention_sequential_transport_policy=(" in compiler_source


def test_composed_profile_builds_exact_all_layer_runtime_policy(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from neural_forge import hf_lfm2_cpcf_v5_model_compiler as v5_compiler
    from neural_forge.hf_lfm2_composed_qk_attention_fold import (
        Lfm2ComposedQkAttentionFoldPolicy,
    )

    args = MODULE.build_parser().parse_args(
        [*_required_args(tmp_path), "--composed-qk-attention-profile"]
    )
    config = MODULE.config_from_args(args)
    model_policy = MODULE._build_configured_model_fold_policy(config)
    composed = MODULE._build_configured_composed_qk_attention_policy(config)
    identity = MODULE._configured_composed_qk_policy_identity(
        config,
        composed_policy=composed,
    )
    runtime = v5_compiler.Lfm2CpcfV5RuntimePolicy(
        model_fold_policy=model_policy,
        maximum_resident_evidence_bytes=1,
        full_phase_tree_logical_bytes=2,
        composed_qk_attention_policy=composed,
    )
    binding = MODULE._build_composed_qk_runtime_policy_binding(
        identity,
        runtime_policy=runtime,
    )
    request = config.as_dict()

    assert type(config.composed_qk_attention_profile) is (
        MODULE.V5ComposedQkAttentionRunnerProfile
    )
    assert type(composed) is Lfm2ComposedQkAttentionFoldPolicy
    assert composed.verify_authentication()
    assert composed.score_policy.solver_device == "cuda:1"
    assert composed.score_policy.compute_devices == ("cuda:0", "cuda:1")
    assert composed.global_policy.maximum_total_score_elements == 134_217_728
    assert composed.global_policy.maximum_total_output_elements == 201_326_592
    assert identity is not None and binding is not None
    assert runtime.resolved_composed_qk_attention_policy("full_attention") is composed
    assert runtime.resolved_composed_qk_attention_policy("short_conv") is None
    assert request["composed_qk_attention_profile"] == (
        config.composed_qk_attention_profile.as_dict()
    )
    assert request["composed_qk_attention_policy_sha256"] == (
        composed.authentication_sha256
    )
    assert binding["runtime_policy_sha256"] == MODULE._sha(runtime.as_dict())
    assert binding["cross_profile_resume_permitted"] is False

    compiler_source = inspect.getsource(
        v5_compiler.compile_lfm2_cpcf_v5_selective_model_fold
    )
    assert "resolved_composed_qk_attention_policy" in compiler_source
    assert "composed_qk_attention_policy=(" in compiler_source


def test_composed_and_sequential_attention_profiles_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--attention-sequential-transport-profile",
            MODULE.ATTENTION_SEQUENTIAL_TRANSPORT_RUNNER_PROFILE_ID,
            "--composed-qk-attention-profile",
        ]
    )
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="mutually exclusive",
    ):
        MODULE.config_from_args(args)


def test_endpoint_profile_builds_exact_five_arm_runtime_policy(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from neural_forge import hf_lfm2_cpcf_v5_model_compiler as v5_compiler
    from neural_forge.hf_lfm2_composed_qk_attention_fold import (
        Lfm2ComposedEndpointResidualAttentionFoldPolicy,
        Lfm2ComposedQkAttentionFoldPolicy,
    )

    args = MODULE.build_parser().parse_args(
        [*_required_args(tmp_path), "--endpoint-residual-attention-profile"]
    )
    config = MODULE.config_from_args(args)
    model_policy = MODULE._build_configured_model_fold_policy(config)
    composed = MODULE._build_configured_composed_qk_attention_policy(config)
    endpoint = MODULE._build_configured_endpoint_residual_attention_policy(config)
    identity = MODULE._configured_endpoint_residual_policy_identity(
        config,
        composed_policy=composed,
        endpoint_policy=endpoint,
    )
    runtime = v5_compiler.Lfm2CpcfV5RuntimePolicy(
        model_fold_policy=model_policy,
        maximum_resident_evidence_bytes=1,
        full_phase_tree_logical_bytes=2,
        composed_qk_attention_policy=composed,
        endpoint_residual_attention_policy=endpoint,
    )
    binding = MODULE._build_endpoint_residual_runtime_policy_binding(
        identity,
        runtime_policy=runtime,
    )
    request = config.as_dict()

    assert type(config.endpoint_residual_attention_profile) is (
        MODULE.V5EndpointResidualAttentionRunnerProfile
    )
    assert type(composed) is Lfm2ComposedQkAttentionFoldPolicy
    assert type(endpoint) is Lfm2ComposedEndpointResidualAttentionFoldPolicy
    assert composed.verify_authentication() is True
    assert endpoint.verify_authentication() is True
    assert endpoint.maximum_outer_blocks == 2
    assert endpoint.endpoint_policy.maximum_cg_iterations == 64
    assert endpoint.endpoint_policy.cg_restart_iterations == 64
    assert endpoint.endpoint_policy.cg_relative_tolerance == 1e-2
    assert endpoint.endpoint_policy.solver_device == "cuda:1"
    assert endpoint.endpoint_policy.compute_devices == ("cuda:0", "cuda:1")
    assert endpoint.allow_legacy_recomputed_parent_probabilities is False
    assert config.endpoint_residual_attention_profile.as_dict()[
        "allowed_candidate_count_per_attention_layer"
    ] == [4, 5]
    assert (
        runtime.resolved_endpoint_residual_attention_policy("full_attention")
        is endpoint
    )
    assert runtime.resolved_endpoint_residual_attention_policy("short_conv") is None
    assert identity is not None and binding is not None
    assert request["endpoint_residual_attention_profile"] == (
        config.endpoint_residual_attention_profile.as_dict()
    )
    assert request["endpoint_residual_attention_policy_sha256"] == (
        endpoint.authentication_sha256
    )
    assert binding["runtime_policy_sha256"] == MODULE._sha(runtime.as_dict())
    assert binding["cross_profile_resume_permitted"] is False


def _mixed_endpoint_coverage_fixture() -> tuple[dict[str, object], dict[str, object]]:
    from neural_forge.hf_lfm2_cpcf_layer_compiler import (
        LFM2_CPCF_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS,
        LFM2_CPCF_ENDPOINT_WITHOUT_COMPOSED_ATTENTION_CANDIDATE_IDS,
    )

    profile = MODULE.V5EndpointResidualAttentionRunnerProfile().as_dict()
    identity_body = {
        "schema": MODULE.ENDPOINT_RESIDUAL_POLICY_IDENTITY_SCHEMA,
        "endpoint_residual_attention_profile": profile,
        "endpoint_residual_composed_qk_attention_policy_sha256": profile[
            "composed_qk_attention_policy"
        ]["authentication_sha256"],
        "endpoint_residual_attention_policy_sha256": profile[
            "endpoint_residual_attention_policy"
        ]["authentication_sha256"],
        "expected_attention_layer_indices": list(
            MODULE.COMPOSED_QK_ATTENTION_LAYER_INDICES
        ),
        "runtime_policy_must_embed_exact_composed_and_endpoint_policies": True,
        "cross_profile_resume_permitted": False,
    }
    identity = {**identity_body, "identity_sha256": MODULE._sha(identity_body)}
    candidate_counts = [4, 5, 4, 5, 4, 5]
    rows = []
    for layer_index, candidate_count in zip(
        MODULE.COMPOSED_QK_ATTENTION_LAYER_INDICES,
        candidate_counts,
        strict=True,
    ):
        eligible = candidate_count == 5
        ids = (
            LFM2_CPCF_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS
            if eligible
            else LFM2_CPCF_ENDPOINT_WITHOUT_COMPOSED_ATTENTION_CANDIDATE_IDS
        )
        digest = MODULE._sha({"layer": layer_index})
        rows.append(
            {
                "layer_index": layer_index,
                "layer_result_sha256": digest,
                "endpoint_artifact_sha256": digest,
                "candidate_evaluation_report_sha256": digest,
                "candidate_selection_report_sha256": digest,
                "candidate_count": candidate_count,
                "candidate_ids": list(ids),
                "composed_candidate_eligible": eligible,
                "source_inventory_sha256": digest,
                "evaluation_fit_evidence_sha256": digest,
                "evaluation_fit_check_evidence_sha256": digest,
                "blank_independent_output_prior_sha256": digest,
                "outer_block_count": 1,
                "ordered_block_sha256s": [digest],
                "ordered_qk_fit_proposal_sha256s": [digest],
                "ordered_execution_receipt_sha256s": [digest],
            }
        )
    expected_layers = list(MODULE.COMPOSED_QK_ATTENTION_LAYER_INDICES)
    coverage_body = {
        "schema": MODULE.ENDPOINT_RESIDUAL_ATTENTION_COVERAGE_SCHEMA,
        "policy_authentication_sha256": identity_body[
            "endpoint_residual_attention_policy_sha256"
        ],
        "expected_attention_layer_indices": expected_layers,
        "observed_attention_layer_indices": expected_layers,
        "covered_attention_layer_indices": expected_layers,
        "attention_layer_count": len(expected_layers),
        "ordered_layer_coverage": rows,
        "exact_ordered_attention_layer_coverage": True,
        "full_attention_only_forwarding": True,
        "candidate_count_per_attention_layer": candidate_counts,
        "candidate_ids_per_attention_layer": [row["candidate_ids"] for row in rows],
        "composed_candidate_eligible_per_attention_layer": [
            row["composed_candidate_eligible"] for row in rows
        ],
        "candidate_local_fit_only_downstream_mlp_evaluation": True,
        "fit_check_final_one_shot_nonregression_only": True,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "blank_target_values_read": False,
    }
    coverage = {
        **coverage_body,
        "report_sha256": MODULE._sha(coverage_body),
    }
    return {"endpoint_residual_attention_coverage": coverage}, identity


def test_endpoint_coverage_accepts_exact_mixed_four_and_five_arm_layers() -> None:
    compiled_report, identity = _mixed_endpoint_coverage_fixture()

    coverage = MODULE._authenticate_endpoint_residual_coverage(
        compiled_report,
        policy_identity=identity,
    )

    assert coverage["candidate_count_per_attention_layer"] == [4, 5, 4, 5, 4, 5]
    assert coverage["composed_candidate_eligible_per_attention_layer"] == [
        False,
        True,
        False,
        True,
        False,
        True,
    ]


def test_endpoint_coverage_rejects_resealed_mixed_inventory_tamper() -> None:
    compiled_report, identity = _mixed_endpoint_coverage_fixture()
    tampered = copy.deepcopy(compiled_report)
    coverage_body = tampered["endpoint_residual_attention_coverage"]
    coverage_body.pop("report_sha256")
    coverage_body["ordered_layer_coverage"][0]["composed_candidate_eligible"] = True
    coverage_body["composed_candidate_eligible_per_attention_layer"][0] = True
    coverage_body["report_sha256"] = MODULE._sha(coverage_body)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="dynamic four/five-arm",
    ):
        MODULE._authenticate_endpoint_residual_coverage(
            tampered,
            policy_identity=identity,
        )


def test_endpoint_profile_is_mutually_exclusive_with_prior_attention_profiles(
    tmp_path: Path,
) -> None:
    parser = MODULE.build_parser()
    for prior_flag in (
        "--composed-qk-attention-profile",
        "--attention-sequential-transport-profile",
    ):
        args = [*_required_args(tmp_path / prior_flag.removeprefix("--"))]
        args.append("--endpoint-residual-attention-profile")
        if prior_flag == "--attention-sequential-transport-profile":
            args.extend(
                [
                    prior_flag,
                    MODULE.ATTENTION_SEQUENTIAL_TRANSPORT_RUNNER_PROFILE_ID,
                ]
            )
        else:
            args.append(prior_flag)
        with pytest.raises(
            MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
            match="mutually exclusive",
        ):
            MODULE.config_from_args(parser.parse_args(args))


def test_endpoint_terminal_profile_pins_policy_sources_runtime_and_receipt(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from neural_forge import hf_lfm2_cpcf_v5_model_compiler as v5_compiler
    from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
        Lfm2CpcfTerminalActionPolicy,
    )
    from neural_forge.hf_lfm2_cpcf_v5_terminal_action_integration import (
        LFM2_CPCF_V5_TERMINAL_INTEGRATION_SCHEMA,
    )

    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [*_required_args(tmp_path), "--endpoint-terminal-action-profile"]
        )
    )
    model = MODULE._build_configured_model_fold_policy(config)
    composed = MODULE._build_configured_composed_qk_attention_policy(config)
    endpoint = MODULE._build_configured_endpoint_residual_attention_policy(config)
    terminal = MODULE._build_configured_terminal_action_residual_policy(config)
    identity = MODULE._configured_endpoint_terminal_action_policy_identity(
        config,
        composed_policy=composed,
        endpoint_policy=endpoint,
        terminal_policy=terminal,
    )
    fields = MODULE._endpoint_terminal_action_policy_identity_fields(identity)
    runtime = v5_compiler.Lfm2CpcfV5RuntimePolicy(
        model_fold_policy=model,
        maximum_resident_evidence_bytes=1,
        full_phase_tree_logical_bytes=2,
        composed_qk_attention_policy=composed,
        endpoint_residual_attention_policy=endpoint,
        terminal_action_residual_policy=terminal,
    )
    binding = MODULE._build_endpoint_terminal_action_runtime_policy_binding(
        identity,
        runtime_policy=runtime,
    )

    assert type(config.endpoint_terminal_action_profile) is (
        MODULE.V5EndpointTerminalActionRunnerProfile
    )
    assert type(terminal) is Lfm2CpcfTerminalActionPolicy
    assert terminal.verify_authentication() is True
    assert fields["terminal_action_residual_policy_sha256"] == (
        terminal.authentication_sha256
    )
    assert fields["terminal_action_policy_source_sha256"] == (
        MODULE._file_sha256(MODULE._TERMINAL_ACTION_POLICY_SOURCE)
    )
    assert fields["terminal_action_integration_source_sha256"] == (
        MODULE._file_sha256(MODULE._TERMINAL_ACTION_INTEGRATION_SOURCE)
    )
    assert binding["runtime_policy_sha256"] == MODULE._sha(runtime.as_dict())
    assert binding["terminal_integration_receipt_required"] is True
    assert runtime.as_dict()["terminal_action_residual_policy"] == terminal.as_dict()

    pure_body = {"policy_sha256": terminal.authentication_sha256}
    pure = {**pure_body, "result_sha256": MODULE._sha(pure_body)}
    operation_audit = {
        "parent_model_forward_calls": 0,
        "target_full_model_forward_calls": 0,
        "target_final_layer_replay_calls": 1,
        "generation_calls": 0,
        "parent_cache_generation_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_replaced": 0,
        "new_parameters_created": 0,
    }
    receipt_body = {
        "schema": LFM2_CPCF_V5_TERMINAL_INTEGRATION_SCHEMA,
        "policy_sha256": terminal.authentication_sha256,
        "pure_tarc_result_sha256": pure["result_sha256"],
        "pure_tarc_report": pure,
        "writer_parameter_identity_preserved": True,
        "parameter_count_changed": False,
        "final_layer_only_local_replay": True,
        "parent_final_output_payload_live_during_compile": True,
        "rollback_exact": True,
        "operation_audit": operation_audit,
    }
    receipt = {
        **receipt_body,
        "result_sha256": MODULE._sha(receipt_body),
    }
    publication_body = {
        "schema": v5_compiler.LFM2_CPCF_V5_TERMINAL_RESUME_PUBLICATION_SCHEMA,
        "next_layer_index": 14,
        "expected_layer_count": 14,
        "terminal_action_enabled": True,
        "preintegration_final_layer_prefix": True,
        "publish_resume_prefix": False,
        "latest_safe_recovery_boundary": (
            "preceding_admitted_prefix_then_reexecute_final_layer_and_terminal"
        ),
        "postintegration_checkpoint_published": False,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    publication = {
        **publication_body,
        "report_sha256": MODULE._sha(publication_body),
    }
    compiled = {
        "terminal_action_residual_enabled": True,
        "terminal_action_residual_integrated": True,
        "terminal_action_residual_report": receipt,
        "terminal_action_residual_report_sha256": receipt["result_sha256"],
        "ordered_layer_result_sha256": [
            MODULE._sha({"layer": index}) for index in range(14)
        ],
        "resume_admitted_prefixes": [
            {"next_layer_index": index} for index in range(14)
        ],
        "terminal_action_preintegration_final_prefix_publication_suppressed": True,
        "terminal_action_resume_publication_receipt": publication,
        "terminal_action_suppressed_resume_prefix_indices": [14],
        "terminal_action_resume_requires_final_layer_reexecution": True,
        "terminal_action_postintegration_resume_checkpoint_published": False,
    }
    authenticated = MODULE._authenticate_terminal_action_integration_receipt(
        compiled,
        policy_identity=identity,
    )
    assert authenticated["result_sha256"] == receipt["result_sha256"]
    leaked_final_prefix = copy.deepcopy(compiled)
    leaked_final_prefix["resume_admitted_prefixes"].append({"next_layer_index": 14})
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="Terminal-action integration receipt",
    ):
        MODULE._authenticate_terminal_action_integration_receipt(
            leaked_final_prefix,
            policy_identity=identity,
        )
    tampered = copy.deepcopy(compiled)
    tampered["terminal_action_residual_report"]["operation_audit"]["training_steps"] = 1
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="Terminal-action integration receipt",
    ):
        MODULE._authenticate_terminal_action_integration_receipt(
            tampered,
            policy_identity=identity,
        )


def test_endpoint_terminal_profile_is_mutually_exclusive_with_every_prior_profile(
    tmp_path: Path,
) -> None:
    parser = MODULE.build_parser()
    prior_arguments = (
        ("--composed-qk-attention-profile",),
        ("--endpoint-residual-attention-profile",),
        (
            "--attention-sequential-transport-profile",
            MODULE.ATTENTION_SEQUENTIAL_TRANSPORT_RUNNER_PROFILE_ID,
        ),
    )
    for index, prior in enumerate(prior_arguments):
        args = [*_required_args(tmp_path / str(index)), *prior]
        args.append("--endpoint-terminal-action-profile")
        with pytest.raises(
            MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
            match="mutually exclusive",
        ):
            MODULE.config_from_args(parser.parse_args(args))


def test_endpoint_sequential_terminal_profile_binds_every_exact_component(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from neural_forge import hf_lfm2_cpcf_v5_model_compiler as v5_compiler

    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_required_args(tmp_path),
                "--endpoint-sequential-terminal-action-profile",
            ]
        )
    )
    profile = config.endpoint_sequential_terminal_action_profile
    model = MODULE._build_configured_model_fold_policy(config)
    composed = MODULE._build_configured_composed_qk_attention_policy(config)
    endpoint = MODULE._build_configured_endpoint_residual_attention_policy(config)
    combination = MODULE._build_configured_endpoint_sequential_transport_policy(config)
    terminal = MODULE._build_configured_terminal_action_residual_policy(config)
    post_operator = profile.build_post_operator_transport_policy()
    identity = MODULE._configured_endpoint_sequential_terminal_action_policy_identity(
        config,
        model_fold_policy=model,
        composed_policy=composed,
        endpoint_policy=endpoint,
        combination_policy=combination,
        terminal_policy=terminal,
    )
    fields = MODULE._endpoint_sequential_terminal_action_policy_identity_fields(
        identity
    )
    runtime = v5_compiler.Lfm2CpcfV5RuntimePolicy(
        model_fold_policy=model,
        maximum_resident_evidence_bytes=1,
        full_phase_tree_logical_bytes=2,
        composed_qk_attention_policy=composed,
        endpoint_residual_attention_policy=endpoint,
        terminal_action_residual_policy=terminal,
        endpoint_sequential_transport_policy=combination,
        post_operator_transport_policy=post_operator,
    )
    binding = MODULE._build_endpoint_sequential_terminal_action_runtime_policy_binding(
        identity,
        runtime_policy=runtime,
    )

    assert type(profile) is MODULE.V5EndpointSequentialTerminalActionRunnerProfile
    assert combination.authenticates_components(
        composed_qk_attention_policy=composed,
        endpoint_residual_attention_policy=endpoint,
        attention_sequential_transport_policy=(
            model.attention_sequential_transport_policy
        ),
    )
    assert fields["post_operator_transport_policy_sha256"] == (
        post_operator.authentication_sha256
    )
    assert fields["endpoint_sequential_transport_combination_policy_sha256"] == (
        combination.authentication_sha256
    )
    assert binding["runtime_policy_sha256"] == MODULE._sha(runtime.as_dict())
    assert binding["runtime_policy_contains_all_exact_component_policies"] is True
    assert binding["two_sided_transport_coverage_receipt_required"] is True
    assert binding["terminal_integration_receipt_required"] is True
    assert (
        runtime.resolved_endpoint_sequential_transport_policy("full_attention")
        is combination
    )
    assert runtime.resolved_endpoint_sequential_transport_policy("short_conv") is None


def test_endpoint_sequential_terminal_profile_catalog_round_trip_and_exclusion(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_required_args(tmp_path),
                "--endpoint-sequential-terminal-action-profile",
            ]
        )
    )
    identity = MODULE._configured_endpoint_sequential_terminal_action_policy_identity(
        config
    )
    fields = MODULE._endpoint_sequential_terminal_action_policy_identity_fields(
        identity
    )
    preparation = {**_catalog_preparation(), **fields}

    MODULE._authenticate_preparation_policy_identity(preparation, config=config)
    for prior in (
        ("--composed-qk-attention-profile",),
        ("--endpoint-residual-attention-profile",),
        ("--endpoint-terminal-action-profile",),
        (
            "--attention-sequential-transport-profile",
            MODULE.ATTENTION_SEQUENTIAL_TRANSPORT_RUNNER_PROFILE_ID,
        ),
    ):
        with pytest.raises(
            MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
            match="mutually exclusive",
        ):
            MODULE.config_from_args(
                MODULE.build_parser().parse_args(
                    [
                        *_required_args(tmp_path / prior[0][2:]),
                        *prior,
                        "--endpoint-sequential-terminal-action-profile",
                    ]
                )
            )


def test_endpoint_sequential_transport_coverage_is_exact_and_tamper_closed(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from neural_forge.hf_lfm2_cpcf_v5_model_compiler import (
        LFM2_CPCF_V5_ENDPOINT_SEQUENTIAL_TRANSPORT_COVERAGE_SCHEMA,
    )

    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_required_args(tmp_path),
                "--endpoint-sequential-terminal-action-profile",
            ]
        )
    )
    identity = MODULE._configured_endpoint_sequential_terminal_action_policy_identity(
        config
    )
    fields = MODULE._endpoint_sequential_terminal_action_policy_identity_fields(
        identity
    )
    indices = list(MODULE.COMPOSED_QK_ATTENTION_LAYER_INDICES)
    rows = [
        {
            "layer_index": layer_index,
            "layer_result_sha256": MODULE._sha({"result": layer_index}),
            "post_operator_transport_report_sha256": MODULE._sha(
                {"upstream": layer_index}
            ),
            "candidate_selection_report_sha256": MODULE._sha(
                {"selection": layer_index}
            ),
            "candidate_count": 5,
            "selected_candidate_id": "endpoint_residual_alternating_qkvo",
            "selected_effective_boundary_charts_sha256": MODULE._sha(
                {"downstream": layer_index}
            ),
            "fit_required_write_rms_reduction_fraction": 0.33,
            "selection_required_write_rms_reduction_fraction": 0.16,
        }
        for layer_index in indices
    ]
    body = {
        "schema": LFM2_CPCF_V5_ENDPOINT_SEQUENTIAL_TRANSPORT_COVERAGE_SCHEMA,
        "policy_authentication_sha256": fields[
            "endpoint_sequential_transport_combination_policy_sha256"
        ],
        "expected_attention_layer_indices": indices,
        "observed_attention_layer_indices": indices,
        "covered_attention_layer_indices": indices,
        "ordered_layer_coverage": rows,
        "fit_only_post_operator_transport_per_attention_layer": True,
        "candidate_local_post_layer_transport_per_qkvo_candidate": True,
        "candidate_specific_closed_form_mlp_per_qkvo_candidate": True,
        "winner_bundle_frozen_before_validation": True,
        "actual_child_post_layer_state_propagated": True,
        "short_conv_transport_enabled": False,
        "validation_rows_read_by_transport_solvers": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    coverage = {**body, "report_sha256": MODULE._sha(body)}

    observed = MODULE._authenticate_endpoint_sequential_transport_coverage(
        {"endpoint_sequential_transport_coverage": coverage},
        policy_identity=identity,
    )
    assert observed["covered_attention_layer_indices"] == indices
    tampered = copy.deepcopy(coverage)
    tampered["training_steps"] = 1
    tampered_body = dict(tampered)
    tampered_body.pop("report_sha256")
    tampered["report_sha256"] = MODULE._sha(tampered_body)
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="six-attention-layer no-training receipt",
    ):
        MODULE._authenticate_endpoint_sequential_transport_coverage(
            {"endpoint_sequential_transport_coverage": tampered},
            policy_identity=identity,
        )


def test_composed_profile_catalog_identity_is_exact_and_tamper_closed(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [*_required_args(tmp_path), "--composed-qk-attention-profile"]
        )
    )
    identity = MODULE._configured_composed_qk_policy_identity(config)
    fields = MODULE._composed_qk_policy_identity_fields(identity)
    preparation = {**_catalog_preparation(), **fields}

    MODULE._authenticate_preparation_policy_identity(
        preparation,
        config=config,
    )
    tampered = dict(preparation)
    tampered["composed_qk_attention_policy_sha256"] = "0" * 64
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="differs from this request",
    ):
        MODULE._authenticate_preparation_policy_identity(
            tampered,
            config=config,
        )


def test_opt_in_profile_changes_sealed_request_and_preparation_identity(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    off = MODULE.config_from_args(
        MODULE.build_parser().parse_args(_required_args(tmp_path))
    )
    on = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_required_args(tmp_path),
                "--attention-sequential-transport-profile",
                MODULE.ATTENTION_SEQUENTIAL_TRANSPORT_RUNNER_PROFILE_ID,
            ]
        )
    )
    identity = MODULE._configured_model_fold_policy_identity(on)
    assert identity is not None
    fields = MODULE._opt_in_policy_identity_fields(identity)
    common_preparation = {
        "schema": MODULE.SELECTIVE_CATALOG_PREPARATION_SCHEMA,
        "fixture": "same-catalog",
    }
    off_preparation = dict(common_preparation)
    on_preparation = {**common_preparation, **fields}

    assert MODULE._canonical_json(off.as_dict()) != MODULE._canonical_json(on.as_dict())
    assert MODULE._sha(off.as_dict()) != MODULE._sha(on.as_dict())
    assert MODULE._sha(off_preparation) != MODULE._sha(on_preparation)
    MODULE._authenticate_preparation_policy_identity(
        off_preparation,
        config=off,
    )
    MODULE._authenticate_preparation_policy_identity(
        on_preparation,
        config=on,
    )
    orphaned_off = {
        **off_preparation,
        "complete_model_fold_policy_sha256": MODULE._sha({"orphan": "model"}),
    }
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="opt-in fold policy profile",
    ):
        MODULE._authenticate_preparation_policy_identity(
            orphaned_off,
            config=off,
        )
    foreign_active = {
        **on_preparation,
        "terminal_action_residual_policy_sha256": MODULE._sha({"orphan": "terminal"}),
    }
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="differs from this request",
    ):
        MODULE._authenticate_preparation_policy_identity(
            foreign_active,
            config=on,
        )
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="opt-in fold policy profile",
    ):
        MODULE._authenticate_preparation_policy_identity(
            on_preparation,
            config=off,
        )
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="differs from this request",
    ):
        MODULE._authenticate_preparation_policy_identity(
            off_preparation,
            config=on,
        )


def test_opt_in_profile_tampering_fails_closed(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_required_args(tmp_path),
                "--attention-sequential-transport-profile",
                MODULE.ATTENTION_SEQUENTIAL_TRANSPORT_RUNNER_PROFILE_ID,
            ]
        )
    )
    identity = MODULE._configured_model_fold_policy_identity(config)
    assert identity is not None
    fields = MODULE._opt_in_policy_identity_fields(identity)

    tampered_identity = dict(identity)
    tampered_identity["complete_model_fold_policy_sha256"] = "0" * 64
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="schema or seal",
    ):
        MODULE._opt_in_policy_identity_fields(tampered_identity)

    tampered_preparation = dict(fields)
    tampered_preparation["complete_model_fold_policy_sha256"] = "0" * 64
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="differs from this request",
    ):
        MODULE._authenticate_preparation_policy_identity(
            tampered_preparation,
            config=config,
        )


def test_resume_binding_rejects_nested_model_policy_ridge_mismatch(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from neural_forge.hf_lfm2_cpcf_v5_model_compiler import (
        Lfm2CpcfV5RuntimePolicy,
    )

    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_required_args(tmp_path),
                "--attention-sequential-transport-profile",
                MODULE.ATTENTION_SEQUENTIAL_TRANSPORT_RUNNER_PROFILE_ID,
            ]
        )
    )
    selected_policy = MODULE._build_configured_model_fold_policy(config)
    identity = MODULE._configured_model_fold_policy_identity(
        config,
        model_fold_policy=selected_policy,
    )
    mismatched_runtime = Lfm2CpcfV5RuntimePolicy(
        model_fold_policy=replace(
            selected_policy,
            ridge=selected_policy.ridge * 10.0,
        ),
        maximum_resident_evidence_bytes=1,
        full_phase_tree_logical_bytes=1,
    )

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="embeds a different complete model fold policy",
    ):
        MODULE._build_resume_policy_binding(
            identity,
            runtime_policy=mismatched_runtime,
        )


def test_catalog_store_overlap_is_rejected_at_configuration(
    tmp_path: Path,
) -> None:
    args = _required_args(tmp_path)
    args.extend(
        (
            "--selective-catalog-store-directory",
            str(tmp_path / "cache" / "catalogs"),
            "--selective-catalog-receipt-json",
            str(tmp_path / "catalog-preparation.json"),
        )
    )

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="cannot equal, contain, or be contained",
    ):
        MODULE.config_from_args(MODULE.build_parser().parse_args(args))


def test_catalog_receipt_accepts_exact_safe_sibling_layout(
    tmp_path: Path,
) -> None:
    trajectory_root = tmp_path / "trajectory-input"
    trajectory_root.mkdir()
    trajectory = trajectory_root / "trajectories.json"
    trajectory.write_text('{"fixture":"trajectory"}\n', "ascii")
    args = _required_args(tmp_path)
    args[args.index("--trajectory-corpus") + 1] = str(trajectory)
    args[args.index("--diagnostic-json") + 1] = str(
        tmp_path / "work" / "diagnostic.json"
    )
    args.extend(
        (
            "--resume-store-directory",
            str(tmp_path / "resume"),
            "--selective-catalog-store-directory",
            str(tmp_path / "catalog-store"),
            "--selective-catalog-receipt-json",
            str(tmp_path / "catalog-control" / "preparation.json"),
        )
    )

    config = MODULE.config_from_args(MODULE.build_parser().parse_args(args))

    assert config.selective_catalog_receipt_json == (
        tmp_path / "catalog-control" / "preparation.json"
    )


def test_catalog_receipt_rejects_nested_ancestor_and_equal_paths(
    tmp_path: Path,
) -> None:
    trajectory_root = tmp_path / "trajectory-input"
    trajectory_root.mkdir()
    trajectory = trajectory_root / "trajectories.json"
    trajectory.write_text('{"fixture":"trajectory"}\n', "ascii")
    args = _required_args(tmp_path)
    args[args.index("--trajectory-corpus") + 1] = str(trajectory)
    args[args.index("--diagnostic-json") + 1] = str(
        tmp_path / "work" / "diagnostic.json"
    )
    args.extend(
        (
            "--resume-store-directory",
            str(tmp_path / "resume"),
            "--selective-catalog-store-directory",
            str(tmp_path / "catalog-store"),
            "--selective-catalog-receipt-json",
            str(tmp_path / "catalog-control" / "preparation.json"),
        )
    )
    config = MODULE.config_from_args(MODULE.build_parser().parse_args(args))
    invalid_receipts = (
        config.parent_model_directory / "nested-receipt.json",
        tmp_path,
        config.trajectory_corpus,
        config.resume_store_directory / "nested-receipt.json",
        config.diagnostic_json,
        config.selective_catalog_store_directory,
    )

    for receipt in invalid_receipts:
        with pytest.raises(
            MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
            match="cannot equal, contain, or be contained",
        ):
            replace(
                config,
                selective_catalog_receipt_json=receipt,
            )


def test_production_fold_authenticates_catalog_before_torch_or_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "parent",
        "donor-tokenizer",
        "target-tokenizer",
        "bridge",
        "cache",
    ):
        (tmp_path / name).mkdir()
    trajectory_root = tmp_path / "trajectory-input"
    trajectory_root.mkdir()
    trajectory = trajectory_root / "trajectories.json"
    trajectory.write_text(
        '{"fixture":"trajectory"}\n',
        "ascii",
    )
    args = _required_args(tmp_path)
    trajectory_index = args.index("--trajectory-corpus") + 1
    args[trajectory_index] = str(trajectory)
    diagnostic_index = args.index("--diagnostic-json") + 1
    args[diagnostic_index] = str(tmp_path / "work" / "diagnostic.json")
    args.extend(
        (
            "--selective-catalog-store-directory",
            str(tmp_path / "catalog-store"),
            "--selective-catalog-receipt-json",
            str(tmp_path / "catalog-preparation.json"),
            "--expected-selective-catalog-receipt-sha256",
            "0" * 64,
        )
    )
    config = MODULE.config_from_args(MODULE.build_parser().parse_args(args))

    def forbidden_torch_configuration():
        raise AssertionError("torch configured before catalog authentication")

    monkeypatch.setattr(
        MODULE,
        "_configure_torch",
        forbidden_torch_configuration,
    )
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="receipt is missing",
    ):
        MODULE._run(config)


def test_catalog_prepare_and_fold_source_order_is_fail_closed() -> None:
    run_source = inspect.getsource(MODULE._run)
    prepare_source = inspect.getsource(MODULE.prepare_selective_catalog)

    assert run_source.index(
        "_read_selective_catalog_preparation(config)"
    ) < run_source.index("_configure_torch()")
    assert "open_lfm2_cpcf_selective_evidence_provider_from_catalog" in run_source
    assert "open_or_build_lfm2_cpcf_selective_evidence_provider" not in run_source
    assert "load_local_lfm2_model" not in prepare_source
    assert "plan_and_synthesize_lfm2_cpcf_blank" not in prepare_source
    assert "torch.cuda.is_initialized()" in prepare_source


def test_catalog_post_run_reverification_record_is_exact() -> None:
    preparation = _catalog_preparation()
    record = dict(MODULE.selective_catalog_post_run_reverification_record(preparation))

    assert (
        dict(
            MODULE.authenticate_selective_catalog_post_run_reverification(
                record,
                catalog_preparation=preparation,
            )
        )
        == record
    )
    tampered = dict(record)
    tampered["artifact_catalog_sha256"] = "0" * 64
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="exact authenticated runner record",
    ):
        MODULE.authenticate_selective_catalog_post_run_reverification(
            tampered,
            catalog_preparation=preparation,
        )


def test_cuda_allocator_cap_is_enforced_on_both_logical_devices() -> None:
    class Properties:
        total_memory = 24 * 1024**3
        name = MODULE.EXPECTED_GPU_NAME

    class Cuda:
        def __init__(self) -> None:
            self.calls = []

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_properties(_index):
            return Properties()

        def set_per_process_memory_fraction(self, fraction, *, device):
            self.calls.append((device, fraction))

    cuda = Cuda()
    receipt = MODULE._enforce_cuda_allocator_caps(
        SimpleNamespace(cuda=cuda),
        maximum_cuda_bytes_per_device=20 * 1024**3,
    )

    assert [device for device, _fraction in cuda.calls] == [0, 1]
    assert all(0.0 < fraction < 1.0 for _device, fraction in cuda.calls)
    assert receipt["all_devices_enforced"] is True
    assert receipt["requested_maximum_cuda_bytes_per_device"] == 20 * 1024**3
    assert all(
        row["effective_allocator_maximum_bytes"] <= 20 * 1024**3
        and row["allowed_bytes"] == 20 * 1024**3
        and row["observed_total_memory_bytes"] == 24 * 1024**3
        and row["enforced_before_substantive_cuda_allocations"] is True
        for row in receipt["device_receipts"]
    )
    body = dict(receipt)
    claimed = body.pop("receipt_sha256")
    assert claimed == MODULE._sha(body)


def test_cuda_allocator_cap_fails_when_runtime_cannot_enforce_it() -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=24 * 1024**3,
            name=MODULE.EXPECTED_GPU_NAME,
        ),
        set_per_process_memory_fraction=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("allocator refused")
        ),
    )

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="Could not enforce",
    ):
        MODULE._enforce_cuda_allocator_caps(
            SimpleNamespace(cuda=cuda),
            maximum_cuda_bytes_per_device=20 * 1024**3,
        )


def test_parser_enables_cold_checkpointing_or_exact_admitted_resume(
    tmp_path: Path,
) -> None:
    store = tmp_path / "resume"
    digest = hashlib.sha256(b"admitted-v5-prefix").hexdigest()

    cold_args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(store),
        ]
    )
    cold = MODULE.config_from_args(cold_args)
    assert cold.resume_store_directory == store
    assert cold.resume_admission_sha256 is None
    assert cold.as_dict()["resume_checkpointing_requested"] is True
    assert cold.as_dict()["resume_requested"] is False

    resume_args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(store),
            "--resume-admission-sha256",
            digest,
        ]
    )
    resumed = MODULE.config_from_args(resume_args)
    assert resumed.resume_store_directory == store
    assert resumed.resume_admission_sha256 == digest
    assert resumed.as_dict()["resume_requested"] is True


def test_catalog_plan_pin_and_request_derived_ratio_are_preserved(
    tmp_path: Path,
) -> None:
    plan_sha256 = hashlib.sha256(b"catalog-plan").hexdigest()
    argv = _required_args(tmp_path)
    target_flag = argv.index("--target-unique-parameters")
    argv[target_flag : target_flag + 2] = ["--compression-ratio", "4"]
    args = MODULE.build_parser().parse_args(
        [
            *argv,
            "--expected-target-plan-fingerprint",
            plan_sha256,
        ]
    )
    config = MODULE.config_from_args(args)

    assert config.expected_target_plan_fingerprint == plan_sha256
    assert config.as_dict()["expected_target_plan_fingerprint"] == plan_sha256
    assert (
        MODULE._request_derived_structural_compression_floor(config.compression) == 4.0
    )

    exact = MODULE.config_from_args(
        MODULE.build_parser().parse_args(_required_args(tmp_path))
    )
    assert (
        MODULE._request_derived_structural_compression_floor(exact.compression) == 1.0
    )


def test_catalog_plan_pin_requires_an_exact_digest(tmp_path: Path) -> None:
    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--expected-target-plan-fingerprint",
            "not-a-plan",
        ]
    )
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="exact lowercase SHA-256",
    ):
        MODULE.config_from_args(args)


def test_resume_admission_requires_exact_digest_and_local_store(
    tmp_path: Path,
) -> None:
    parser = MODULE.build_parser()

    missing_store = parser.parse_args(
        [
            *_required_args(tmp_path),
            "--resume-admission-sha256",
            hashlib.sha256(b"head").hexdigest(),
        ]
    )
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="requires an explicit local",
    ):
        MODULE.config_from_args(missing_store)

    invalid_digest = parser.parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(tmp_path / "resume"),
            "--resume-admission-sha256",
            "not-a-v5-admission",
        ]
    )
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="exact lowercase SHA-256",
    ):
        MODULE.config_from_args(invalid_digest)


def test_real_v5_adapter_uses_explicit_preregistered_measurement_gates() -> None:
    pytest.importorskip("torch")
    adapter = MODULE._build_preregistered_real_v5_numerical_adapter()

    assert (
        adapter.basis_policy.maximum_projected_fir_offdiagonal_relative_frobenius
        == MODULE.PREREGISTERED_PROJECTED_FIR_OFFDIAGONAL_GATE
        == 0.0
    )
    assert (
        adapter.basis_policy.maximum_per_lag_commutator_relative_frobenius
        == MODULE.PREREGISTERED_PER_LAG_COMMUTATOR_GATE
        == 1.0
    )
    assert adapter.basis_policy.verify_authentication()
    assert adapter.ridge_policy.verify_authentication()


@pytest.mark.parametrize(
    ("flag", "value"),
    (
        ("--resume", None),
        ("--resume-state", "resume.json"),
        ("--resume-checkpoint-reference", "checkpoint.json"),
        ("--publish", None),
        ("--output-directory", "output"),
        ("--publication-directory", "published"),
        ("--save-child", None),
        ("--regenerate-traces", None),
        ("--train", None),
        ("--prune", None),
        ("--quantize", None),
        ("--download", None),
    ),
)
def test_forbidden_actions_fail_closed_with_an_explicit_reason(
    tmp_path: Path,
    flag: str,
    value: str | None,
) -> None:
    argv = [*_required_args(tmp_path), flag]
    if value is not None:
        argv.append(str(tmp_path / value))
    args = MODULE.build_parser().parse_args(argv)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="forbids requested",
    ):
        MODULE.config_from_args(args)


def test_cold_resume_store_constructs_and_passes_typed_coordinator(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from neural_forge.hf_lfm2_cpcf_v5_resume import (
        Lfm2CpcfV5ResumeCoordinator,
    )

    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(tmp_path / "resume"),
        ]
    )
    runtime = MODULE._build_resume_runtime(MODULE.config_from_args(args))
    observed = {}

    def compile_stub(source, target, **kwargs):
        observed.update(kwargs)
        return (source, target)

    result = MODULE._compile_with_authenticated_resume(
        compile_stub,
        "source",
        "target",
        resume_runtime=runtime,
        compile_kwargs={"sealed_input": True},
    )

    assert result == ("source", "target")
    assert type(runtime.coordinator) is Lfm2CpcfV5ResumeCoordinator
    assert runtime.state is None
    assert observed["resume_coordinator"] is runtime.coordinator
    assert observed["resume_state"] is None
    report = runtime.progress.as_dict()
    assert report["checkpointing_enabled"] is True
    assert report["resume_requested"] is False
    assert report["child_publication_attempted"] is False
    assert report["parent_or_cache_tensor_sidecars"] == 0


def test_exact_coordinator_accepts_real_publish_pending_contract(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")

    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(tmp_path / "resume"),
        ]
    )
    runtime = MODULE._build_resume_runtime(MODULE.config_from_args(args))

    with pytest.raises(TypeError, match="event must be Lfm2CpcfV5ProgressEvent"):
        runtime.coordinator.publish_pending(
            object(),
            rss_checkpoint_expectation={},
        )
    with pytest.raises(TypeError, match="rss_checkpoint_expectation"):
        runtime.coordinator.publish_pending(object())

    report = runtime.progress.as_dict()
    assert report["published_pending_prefix_count"] == 0
    assert report["outstanding_pending_prefix_count"] == 0


def test_real_runtime_exact_coordinator_completes_prefix_zero_admission_proof(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from neural_forge import hf_lfm2_cpcf_v5_admission_proof as admission_proof
    from neural_forge import hf_lfm2_cpcf_v5_model_compiler as v5_compiler
    from neural_forge import hf_lfm2_cpcf_v5_resume as v5_resume
    from tests.test_hf_lfm2_cpcf_resume_bridge import (
        _v5_progress,
        _v5_rss_expectation,
        _v5_rss_receipt,
    )

    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(tmp_path / "resume"),
        ]
    )
    runtime = MODULE._build_resume_runtime(MODULE.config_from_args(args))
    coordinator = runtime.coordinator

    assert type(coordinator) is v5_resume.Lfm2CpcfV5ResumeCoordinator
    assert (
        v5_compiler.Lfm2CpcfV5ResumeCoordinator is v5_resume.Lfm2CpcfV5ResumeCoordinator
    )
    assert (
        admission_proof.resume.Lfm2CpcfV5ResumeCoordinator
        is v5_resume.Lfm2CpcfV5ResumeCoordinator
    )

    prefix_zero = _v5_progress(0)
    expectation = _v5_rss_expectation(0)

    def compile_prefix_zero(_source, _target, **kwargs):
        received = kwargs["resume_coordinator"]
        assert received is coordinator
        pending = received.publish_pending(
            prefix_zero["event"],
            rss_checkpoint_expectation=expectation,
        )
        admission = received.admit(
            pending,
            rss_checkpoint_receipt=_v5_rss_receipt(
                expectation,
                passed=True,
            ),
        )
        proof = admission_proof.build_lfm2_cpcf_v5_admission_chain_proof(
            received,
            admission["admission_sha256"],
        )
        record = {
            "next_layer_index": 0,
            "frontier_reference": pending.frontier_reference.as_dict(),
            "checkpoint_reference": pending.checkpoint.reference.as_dict(),
            "frontier_sha256": pending.frontier_sha256,
            "admission_sha256": admission["admission_sha256"],
            "admission_chain_proof": dict(proof),
            "admission_chain_proof_sha256": proof["proof_sha256"],
        }
        return SimpleNamespace(report={"resume_admitted_prefixes": [record]})

    compiled = MODULE._compile_with_authenticated_resume(
        compile_prefix_zero,
        object(),
        object(),
        resume_runtime=runtime,
        compile_kwargs={},
    )

    assert compiled.report["resume_admitted_prefixes"][0]["next_layer_index"] == 0
    report = runtime.progress.as_dict()
    assert report["published_pending_prefix_count"] == 1
    assert report["admitted_prefix_count"] == 1
    assert report["outstanding_pending_prefix_count"] == 0
    assert report["rejected_prefix_count"] == 0
    assert report["admitted_prefixes"] == (compiled.report["resume_admitted_prefixes"])


def test_exact_admission_opens_only_typed_v5_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from neural_forge.hf_lfm2_cpcf_resume_artifact import (
        Lfm2CpcfArtifactReference,
    )
    from neural_forge.hf_lfm2_cpcf_v5_resume import (
        Lfm2CpcfV5ResumeCoordinator,
        Lfm2CpcfV5ResumeState,
    )

    admission_sha = hashlib.sha256(b"admitted-v5-prefix").hexdigest()
    frontier_sha = hashlib.sha256(b"frontier").hexdigest()
    frontier_reference = Lfm2CpcfArtifactReference(
        artifact_kind="boundary",
        artifact_key=hashlib.sha256(b"boundary-key").hexdigest(),
        manifest_sha256=hashlib.sha256(b"boundary-manifest").hexdigest(),
    )
    checkpoint_reference = Lfm2CpcfArtifactReference(
        artifact_kind="checkpoint",
        artifact_key=hashlib.sha256(b"checkpoint-key").hexdigest(),
        manifest_sha256=hashlib.sha256(b"checkpoint-manifest").hexdigest(),
    )

    def open_admitted(self, exact_digest):
        assert exact_digest == admission_sha
        return Lfm2CpcfV5ResumeState(
            coordinator=self,
            frontier=SimpleNamespace(
                next_layer_index=3,
                frontier_sha256=frontier_sha,
            ),
            frontier_reference=frontier_reference,
            checkpoint_reference=checkpoint_reference,
            admission={"admission_sha256": admission_sha},
        )

    monkeypatch.setattr(
        Lfm2CpcfV5ResumeCoordinator,
        "open_admitted",
        open_admitted,
    )
    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(tmp_path / "resume"),
            "--resume-admission-sha256",
            admission_sha,
        ]
    )

    runtime = MODULE._build_resume_runtime(MODULE.config_from_args(args))

    assert type(runtime.state) is Lfm2CpcfV5ResumeState
    assert runtime.state.coordinator is runtime.coordinator
    head = runtime.progress.as_dict()["invocation_head"]
    assert head["admission_sha256"] == admission_sha
    assert head["next_layer_index"] == 3
    assert head["typed_v5_state_opened"] is True
    assert head["checkpoint_reference"] == checkpoint_reference.as_dict()


def test_opt_in_resume_requires_runtime_digest_before_store_open(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from neural_forge.hf_lfm2_cpcf_v5_model_compiler import (
        Lfm2CpcfV5RuntimePolicy,
    )

    store = tmp_path / "must-not-open"
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_required_args(tmp_path),
                "--resume-store-directory",
                str(store),
                "--attention-sequential-transport-profile",
                MODULE.ATTENTION_SEQUENTIAL_TRANSPORT_RUNNER_PROFILE_ID,
            ]
        )
    )
    policy = MODULE._build_configured_model_fold_policy(config)
    identity = MODULE._configured_model_fold_policy_identity(
        config,
        model_fold_policy=policy,
    )
    runtime = Lfm2CpcfV5RuntimePolicy(
        model_fold_policy=policy,
        maximum_resident_evidence_bytes=1,
        full_phase_tree_logical_bytes=1,
    )
    binding = MODULE._build_resume_policy_binding(
        identity,
        runtime_policy=runtime,
    )

    assert not store.exists()
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="before the resume store may be opened",
    ):
        MODULE._build_resume_runtime(
            config,
            expected_runtime_policy_sha256=None,
            policy_binding=binding,
        )
    assert not store.exists()


def test_off_and_on_profiles_cannot_cross_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from neural_forge.hf_lfm2_cpcf_resume_artifact import (
        Lfm2CpcfArtifactReference,
    )
    from neural_forge.hf_lfm2_cpcf_v5_model_compiler import (
        Lfm2CpcfV5RuntimePolicy,
    )
    from neural_forge.hf_lfm2_cpcf_v5_resume import (
        Lfm2CpcfV5ResumeCoordinator,
        Lfm2CpcfV5ResumeState,
    )

    admission_sha = hashlib.sha256(b"profile-cross-resume").hexdigest()

    def configured(*, enabled: bool, store_name: str):
        profile_args = (
            [
                "--attention-sequential-transport-profile",
                MODULE.ATTENTION_SEQUENTIAL_TRANSPORT_RUNNER_PROFILE_ID,
            ]
            if enabled
            else []
        )
        return MODULE.config_from_args(
            MODULE.build_parser().parse_args(
                [
                    *_required_args(tmp_path),
                    "--resume-store-directory",
                    str(tmp_path / store_name),
                    "--resume-admission-sha256",
                    admission_sha,
                    *profile_args,
                ]
            )
        )

    off = configured(enabled=False, store_name="resume-off")
    on = configured(enabled=True, store_name="resume-on")

    def runtime(config):
        return Lfm2CpcfV5RuntimePolicy(
            model_fold_policy=MODULE._build_configured_model_fold_policy(config),
            maximum_resident_evidence_bytes=1,
            full_phase_tree_logical_bytes=1,
        )

    off_runtime = runtime(off)
    on_runtime = runtime(on)
    off_sha256 = MODULE._sha(off_runtime.as_dict())
    on_sha256 = MODULE._sha(on_runtime.as_dict())
    assert off_sha256 != on_sha256
    on_identity = MODULE._configured_model_fold_policy_identity(
        on,
        model_fold_policy=on_runtime.model_fold_policy,
    )
    on_binding = MODULE._build_resume_policy_binding(
        on_identity,
        runtime_policy=on_runtime,
    )
    frontier_reference = Lfm2CpcfArtifactReference(
        artifact_kind="boundary",
        artifact_key=hashlib.sha256(b"profile-boundary-key").hexdigest(),
        manifest_sha256=hashlib.sha256(b"profile-boundary-manifest").hexdigest(),
    )
    checkpoint_reference = Lfm2CpcfArtifactReference(
        artifact_kind="checkpoint",
        artifact_key=hashlib.sha256(b"profile-checkpoint-key").hexdigest(),
        manifest_sha256=hashlib.sha256(b"profile-checkpoint-manifest").hexdigest(),
    )

    def cross_state(self, _exact_digest, *, state_runtime_sha256):
        return Lfm2CpcfV5ResumeState(
            coordinator=self,
            frontier=SimpleNamespace(
                next_layer_index=2,
                frontier_sha256=hashlib.sha256(b"profile-frontier").hexdigest(),
                context={"runtime_policy_sha256": state_runtime_sha256},
            ),
            frontier_reference=frontier_reference,
            checkpoint_reference=checkpoint_reference,
            admission={"admission_sha256": admission_sha},
        )

    monkeypatch.setattr(
        Lfm2CpcfV5ResumeCoordinator,
        "open_admitted",
        lambda self, digest: cross_state(
            self,
            digest,
            state_runtime_sha256=off_sha256,
        ),
    )
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="runtime policy differs",
    ):
        MODULE._build_resume_runtime(
            on,
            expected_runtime_policy_sha256=on_sha256,
            policy_binding=on_binding,
        )

    monkeypatch.setattr(
        Lfm2CpcfV5ResumeCoordinator,
        "open_admitted",
        lambda self, digest: cross_state(
            self,
            digest,
            state_runtime_sha256=on_sha256,
        ),
    )
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="runtime policy differs",
    ):
        MODULE._build_resume_runtime(
            off,
            expected_runtime_policy_sha256=off_sha256,
            policy_binding=None,
        )


@pytest.mark.parametrize("kind", ("pending", "rejected"))
def test_pending_or_rejected_digest_cannot_open_as_admitted_resume(
    tmp_path: Path,
    kind: str,
) -> None:
    pytest.importorskip("torch")
    from neural_forge.hf_lfm2_cpcf_v5_resume import (
        Lfm2CpcfV5ResumeError,
    )

    digest = hashlib.sha256(kind.encode("ascii")).hexdigest()
    store = tmp_path / f"{kind}-store"
    cold_args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(store),
        ]
    )
    MODULE._build_resume_runtime(MODULE.config_from_args(cold_args))
    if kind == "rejected":
        (store / "v5-rejections").mkdir(exist_ok=True)
    resume_args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(store),
            "--resume-admission-sha256",
            digest,
        ]
    )

    with pytest.raises(Lfm2CpcfV5ResumeError) as caught:
        MODULE._build_resume_runtime(MODULE.config_from_args(resume_args))

    assert "admission" in str(caught.value).lower()
    progress = caught.value.resume_progress_report
    assert progress["requested_admission_sha256"] == digest
    assert progress["invocation_head"] is None


def test_wrong_live_resume_context_refusal_is_preserved(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(tmp_path / "resume"),
        ]
    )
    runtime = MODULE._build_resume_runtime(MODULE.config_from_args(args))

    def reject_context(_source, _target, **kwargs):
        assert kwargs["resume_coordinator"] is runtime.coordinator
        raise RuntimeError("Live resume context differs from the admitted frontier.")

    with pytest.raises(RuntimeError, match="context differs") as caught:
        MODULE._compile_with_authenticated_resume(
            reject_context,
            object(),
            object(),
            resume_runtime=runtime,
            compile_kwargs={"runtime_policy": "different"},
        )

    progress = caught.value.resume_progress_report
    assert progress["checkpointing_enabled"] is True
    assert progress["admitted_prefix_count"] == 0
    assert progress["child_publication_attempted"] is False


def test_failed_rss_preserves_pending_and_rejected_prefix_references(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")

    class Reference:
        def __init__(self, kind: str, label: str) -> None:
            self.value = {
                "artifact_kind": kind,
                "artifact_key": hashlib.sha256(
                    f"{label}-key".encode("ascii")
                ).hexdigest(),
                "manifest_sha256": hashlib.sha256(
                    f"{label}-manifest".encode("ascii")
                ).hexdigest(),
            }

        def as_dict(self):
            return dict(self.value)

    pending = SimpleNamespace(
        next_layer_index=1,
        frontier_reference=Reference("boundary", "frontier"),
        checkpoint=SimpleNamespace(reference=Reference("checkpoint", "checkpoint")),
        frontier_sha256=hashlib.sha256(b"frontier").hexdigest(),
        event_report_sha256=hashlib.sha256(b"event").hexdigest(),
    )
    rejection = {"rejection_sha256": hashlib.sha256(b"rejection").hexdigest()}
    args = MODULE.build_parser().parse_args(
        [
            *_required_args(tmp_path),
            "--resume-store-directory",
            str(tmp_path / "resume"),
        ]
    )
    runtime = MODULE._build_resume_runtime(MODULE.config_from_args(args))

    def fail_rss(_source, _target, **_kwargs):
        runtime.progress.record_pending(pending)
        runtime.progress.record_rejected(pending, rejection)
        raise RuntimeError("physical RSS gate failed")

    with pytest.raises(RuntimeError, match="RSS gate failed") as caught:
        MODULE._compile_with_authenticated_resume(
            fail_rss,
            object(),
            object(),
            resume_runtime=runtime,
            compile_kwargs={},
        )

    diagnostic = MODULE.build_failure_diagnostic(caught.value)
    progress = diagnostic["resume_progress_reports"][0]
    assert progress["published_pending_prefix_count"] == 1
    assert progress["rejected_prefix_count"] == 1
    assert progress["outstanding_pending_prefix_count"] == 0
    assert progress["rejected_prefixes"][0]["promotable"] is False
    assert diagnostic["resume_artifacts_preserved_for_diagnosis"] is True
    assert diagnostic["publication_attempted"] is False
    assert diagnostic["resume_parent_or_cache_tensor_sidecars"] == 0


def test_untyped_or_v4_resume_cannot_enter_compiler_kwargs(
    tmp_path: Path,
) -> None:
    args = MODULE.build_parser().parse_args(_required_args(tmp_path))
    runtime = MODULE._build_resume_runtime(MODULE.config_from_args(args))

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="authenticated runner seam",
    ):
        MODULE._compile_with_authenticated_resume(
            lambda *_args, **_kwargs: None,
            object(),
            object(),
            resume_runtime=runtime,
            compile_kwargs={"resume_state": {"schema": "v4-resume"}},
        )


def _consumer_digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _consumer_candidate_saver(model, destination: Path) -> None:
    torch = pytest.importorskip("torch")
    torch.save(model.state_dict(), destination / "weights.pt")
    (destination / "config.json").write_text(
        json.dumps({"input_width": 3, "output_width": 2}, sort_keys=True),
        encoding="utf-8",
    )


def _consumer_candidate_loader(source: Path):
    torch = pytest.importorskip("torch")
    if not source.is_dir():
        raise FileNotFoundError(source)
    model = torch.nn.Linear(3, 2).requires_grad_(False).eval()
    state = torch.load(
        source / "weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    return model


def _consumer_candidate_evaluator(model):
    torch = pytest.importorskip("torch")
    from neural_forge.dynamic_candidate_capture import (
        QuarantineEvaluation,
    )

    value = model(torch.tensor([[1.0, -2.0, 0.5]]))
    return QuarantineEvaluation(
        output={"rows": value.detach().cpu().tolist()},
        evaluation={
            "finite": bool(torch.isfinite(value).all()),
            "shape": list(value.shape),
        },
    )


def _capture_consumer_candidate(
    context,
    destination: Path,
    *,
    parent_sha256: str | None = None,
):
    from neural_forge.dynamic_candidate_capture import (
        QuarantineCaptureRequest,
        QuarantineZeroOperationAudit,
        capture_quarantined_candidate,
        fingerprint_quarantine_callable,
    )

    parent_sha = context.parent_sha256 if parent_sha256 is None else parent_sha256
    counters = context.zero_operation_audit
    request = QuarantineCaptureRequest(
        plan_sha256=context.target_plan_fingerprint,
        parent_sha256=parent_sha,
        cache_sha256=context.cache_sha256,
        bridge_sha256=context.bridge_sha256,
        corpus_sha256=context.corpus_sha256,
        compiler_sha256=context.compiler_receipt_sha256,
        source_sha256=context.source_binding_sha256,
        saver_sha256=fingerprint_quarantine_callable(
            _consumer_candidate_saver,
            role="saver",
        ),
        loader_sha256=fingerprint_quarantine_callable(
            _consumer_candidate_loader,
            role="loader",
        ),
        evaluator_sha256=fingerprint_quarantine_callable(
            _consumer_candidate_evaluator,
            role="evaluator",
        ),
        expected_model_state_sha256=context.target_state_sha256,
        prior_audit=QuarantineZeroOperationAudit(
            optimizer_constructions=counters["optimizer_constructions"],
            optimizer_steps=counters["optimizer_steps"],
            backward_calls=counters["backward_calls"],
            autograd_grad_calls=counters["autograd_grad_calls"],
            training_steps=counters["training_steps"],
            pruning_operations=counters["pruning_operations"],
            quantization_operations=counters["quantization_operations"],
            parent_forward_calls=counters["parent_forward_calls"],
            parent_inference_calls=counters["parent_inference_calls"],
        ),
    )
    return capture_quarantined_candidate(
        destination,
        candidate=context.target_model,
        request=request,
        saver=_consumer_candidate_saver,
        loader=_consumer_candidate_loader,
        evaluator=_consumer_candidate_evaluator,
        forbidden_parent_models=(context.forbidden_parent_model,),
    )


def _compiled_consumer_context():
    torch = pytest.importorskip("torch")
    parent = torch.nn.Linear(3, 3).requires_grad_(False).eval()
    target = torch.nn.Linear(3, 2).requires_grad_(False).eval()
    parent_auth = MODULE._live_model_authentication(parent)
    target_auth = MODULE._live_model_authentication(target)
    plan_sha = _consumer_digest("plan")
    parent_sha = _consumer_digest("parent")
    cache_sha = _consumer_digest("cache")
    bridge_sha = _consumer_digest("bridge")
    corpus_sha = _consumer_digest("corpus")
    source_sha = _consumer_digest("selective-source")
    compiler_sha = _consumer_digest("compiler-implementation")
    runner_sha = MODULE._file_sha256(SCRIPT)
    compiler_source_files = [
        {
            "module": "tests.fake_v5_compiler",
            "filename": "fake_v5_compiler.py",
            "sha256": _consumer_digest("fake-v5-compiler-source"),
        }
    ]
    parent_tokenizer_sha = _consumer_digest("parent-tokenizer")
    tokenizer_sha = _consumer_digest("target-tokenizer")
    blank_body = {
        "schema": "test-blank-v1",
        "target_plan_fingerprint": plan_sha,
        "target_unique_parameters": target_auth["parameter_inventory"][
            "unique_parameters"
        ],
    }
    blank = {**blank_body, "report_sha256": MODULE._sha(blank_body)}
    compiled_body = {
        "schema": "test-compiled-v1",
        "blank_target_report_sha256": blank["report_sha256"],
        "selective_source_sha256": source_sha,
        "compiler_implementation": {
            "implementation_sha256": compiler_sha,
            "ordered_source_files": compiler_source_files,
        },
    }
    compiled = {
        **compiled_body,
        "result_sha256": MODULE._sha(compiled_body),
    }
    zero_audit = {
        "schema": MODULE.COMPILED_CANDIDATE_ZERO_AUDIT_SCHEMA,
        **{name: 0 for name in MODULE._CONSUMER_ZERO_COUNTER_FIELDS},
        "clean": True,
    }
    from neural_forge import dynamic_fold_executor

    guard_identity = MODULE._fold_only_guard_implementation_identity(
        dynamic_fold_executor
    )
    static_audit_body = {
        "schema": ("neural-forge-lfm2-cpcf-v5-static-fold-mutation-source-audit-v1"),
        "compiler_implementation_sha256": compiler_sha,
        "authenticated_compiler_source_files": compiler_source_files,
        "runner_source_sha256": runner_sha,
        "audited_source_count": 2,
        "source_audits": [],
        "forbidden_findings": [],
        "forbidden_finding_count": 0,
        "audit_scope": "test authenticated sources",
        "clean": True,
    }
    static_audit = {
        **static_audit_body,
        "report_sha256": MODULE._sha(static_audit_body),
    }
    guard_body = {
        "schema": "neural-forge-lfm2-cpcf-v5-compiler-runtime-guard-v1",
        "guard_implementation": dict(guard_identity),
        "static_pruning_quantization_source_audit": static_audit,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "autograd_grad_calls": 0,
        "pruning_operations": 0,
        "quantization_operations": 0,
        "parent_module_forward_calls": 0,
        "parent_root_forward_calls": 0,
        "parent_inference_entrypoint_calls": 0,
        "target_module_forward_calls": 2,
        "target_root_forward_calls": 0,
        "target_inference_entrypoint_calls": 0,
        "training_steps": 0,
        "training_steps_runtime_observed_directly": False,
        "training_steps_zero_derivation": (
            "zero optimizer/backward/autograd API attempts under no-grad"
        ),
        "pruning_observation_scope": (
            "common-public-pytorch-pruning-mutation-entrypoints"
        ),
        "quantization_observation_scope": (
            "common-public-pytorch-quantization-mutation-entrypoints"
        ),
        "ordinary_bfloat16_and_float32_casts_blocked": False,
        "clean": True,
    }
    guard_report = {
        **guard_body,
        "report_sha256": MODULE._sha(guard_body),
    }
    zero_audit["compiler_runtime_guard"] = guard_report
    catalog_preparation = _catalog_preparation()
    source_binding_sha = MODULE._source_binding_sha256(
        selective_source_sha256=source_sha,
        compiler_implementation_sha256=compiler_sha,
        runner_source_sha256=runner_sha,
        fold_only_runtime_guard_identity_sha256=guard_identity["identity_sha256"],
        catalog_preparation=catalog_preparation,
    )
    seam = {
        "schema": MODULE.COMPILED_CANDIDATE_CONTEXT_SCHEMA,
        "target_plan_fingerprint": plan_sha,
        "blank_target_report_sha256": blank["report_sha256"],
        "compiler_receipt_sha256": compiled["result_sha256"],
        "parent_sha256": parent_sha,
        "cache_sha256": cache_sha,
        "bridge_sha256": bridge_sha,
        "corpus_sha256": corpus_sha,
        "source_sha256": source_sha,
        "compiler_sha256": compiler_sha,
        "runner_source_sha256": runner_sha,
        "source_binding_sha256": source_binding_sha,
        "selective_catalog_preparation_receipt_sha256": (
            catalog_preparation["preparation_receipt_sha256"]
        ),
        "selective_catalog_binding_sha256": catalog_preparation["binding_sha256"],
        "selective_catalog_artifact_manifest_sha256": (
            catalog_preparation["artifact_manifest_sha256"]
        ),
        "selective_catalog_artifact_sha256": catalog_preparation[
            "artifact_catalog_sha256"
        ],
        "selective_catalog_codec_sha256": catalog_preparation["codec_sha256"],
        "selective_catalog_provider_implementation_sha256": (
            catalog_preparation["provider_implementation_sha256"]
        ),
        "fold_only_runtime_guard_identity_sha256": guard_identity["identity_sha256"],
        "parent_tokenizer_semantics_sha256": parent_tokenizer_sha,
        "target_tokenizer_semantics_sha256": tokenizer_sha,
        "target_tokenizer_package_sha256": None,
        "target_state_sha256": target_auth["state_sha256"],
        "target_runtime_sha256": target_auth["runtime_sha256"],
        "parent_state_sha256": parent_auth["state_sha256"],
        "parent_runtime_sha256": parent_auth["runtime_sha256"],
        "zero_operation_audit": zero_audit,
    }
    diagnostic_body = {
        "schema": MODULE.RUN_SCHEMA,
        "inputs": {
            "parent_fingerprint": parent_sha,
            "cache_key": cache_sha,
            "bridge_artifact_sha256": bridge_sha,
            "trajectory_corpus_sha256": corpus_sha,
            "donor_tokenizer_semantics_sha256": parent_tokenizer_sha,
            "target_tokenizer_semantics_sha256": tokenizer_sha,
        },
        "blank_target": blank,
        "compiled_model_fold": compiled,
        "compiler_runtime_fold_only_guard": guard_report,
        "parent_parameter_inventory": parent_auth["parameter_inventory"],
        "blank_parameter_inventory": target_auth["parameter_inventory"],
        "compiled_parameter_inventory": target_auth["parameter_inventory"],
        "compiled_candidate_consumer_seam": seam,
        "selective_catalog_artifact": catalog_preparation,
        "selective_catalog_post_run_reverification": dict(
            MODULE.selective_catalog_post_run_reverification_record(catalog_preparation)
        ),
        "claims": {
            "child_checkpoint_saved": False,
            "publication_attempted": False,
        },
    }
    diagnostic = {
        **diagnostic_body,
        "receipt_sha256": MODULE._sha(diagnostic_body),
    }
    context = MODULE.V5CompiledCandidateConsumerContext(
        target_model=target,
        target_tokenizer=SimpleNamespace(name="target-tokenizer"),
        forbidden_parent_model=parent,
        target_plan_fingerprint=plan_sha,
        blank_target_report=blank,
        compiled_model_fold_report=compiled,
        compiler_receipt_sha256=compiled["result_sha256"],
        compile_diagnostic=diagnostic,
        parent_sha256=parent_sha,
        cache_sha256=cache_sha,
        bridge_sha256=bridge_sha,
        corpus_sha256=corpus_sha,
        source_sha256=source_sha,
        compiler_sha256=compiler_sha,
        runner_source_sha256=runner_sha,
        source_binding_sha256=source_binding_sha,
        parent_tokenizer_semantics_sha256=parent_tokenizer_sha,
        target_tokenizer_semantics_sha256=tokenizer_sha,
        target_tokenizer_package_sha256=None,
        parent_parameter_inventory=parent_auth["parameter_inventory"],
        blank_parameter_inventory=target_auth["parameter_inventory"],
        compiled_parameter_inventory=target_auth["parameter_inventory"],
        zero_operation_audit=zero_audit,
        target_state_sha256=target_auth["state_sha256"],
        target_runtime_sha256=target_auth["runtime_sha256"],
        parent_state_sha256=parent_auth["state_sha256"],
        parent_runtime_sha256=parent_auth["runtime_sha256"],
    )
    return context, target, parent


def _compiled_consumer_result(
    context,
    *,
    candidate,
    parent_sha256: str | None = None,
    verification_receipt=None,
):
    parent_sha = context.parent_sha256 if parent_sha256 is None else parent_sha256
    capture = candidate.receipt.as_dict()
    return MODULE.V5CompiledCandidateConsumerResult(
        candidate=candidate,
        capture_receipt=capture,
        capture_receipt_sha256=capture["receipt_sha256"],
        compiler_receipt_sha256=context.compiler_receipt_sha256,
        target_plan_fingerprint=context.target_plan_fingerprint,
        target_state_sha256=context.target_state_sha256,
        parent_sha256=parent_sha,
        cache_sha256=context.cache_sha256,
        bridge_sha256=context.bridge_sha256,
        corpus_sha256=context.corpus_sha256,
        source_sha256=context.source_sha256,
        compiler_sha256=context.compiler_sha256,
        runner_source_sha256=context.runner_source_sha256,
        source_binding_sha256=context.source_binding_sha256,
        parent_tokenizer_semantics_sha256=(context.parent_tokenizer_semantics_sha256),
        target_tokenizer_semantics_sha256=(context.target_tokenizer_semantics_sha256),
        compile_diagnostic=context.compile_diagnostic,
        verification_receipt=(
            candidate.verification_receipt
            if verification_receipt is None
            else verification_receipt
        ),
    )


def test_static_fold_audit_finds_prebound_prune_and_quantization_aliases() -> None:
    findings = MODULE._fold_only_static_source_findings(
        "\n".join(
            (
                "import torch",
                "from torch.nn.utils.prune import L1Unstructured as Plant",
                "plant = Plant.apply",
                "quant = torch.quantize_per_tensor",
                "plant(model, 'weight', amount=0.5)",
                "quant(value, 0.1, 0, torch.qint8)",
            )
        ),
        source_label="prebound_aliases.py",
    )

    kinds = {finding["kind"] for finding in findings}
    symbols = {finding["symbol"] for finding in findings}
    assert "forbidden-import-alias" in kinds
    assert "forbidden-prebound-alias" in kinds
    assert "forbidden-call" in kinds
    assert "torch.nn.utils.prune.L1Unstructured.apply" in symbols
    assert "torch.quantize_per_tensor" in symbols


def test_authenticated_compiler_source_audit_rejects_prebound_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "prebound_compiler.py"
    source_path.write_text(
        "from torch import quantize_per_tensor as hidden_quantize\n",
        encoding="utf-8",
    )
    module_name = "tests.prebound_compiler"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(__file__=str(source_path)),
    )
    implementation_body = {
        "schema": "test-compiler-implementation-v1",
        "ordered_source_files": [
            {
                "module": module_name,
                "filename": source_path.name,
                "sha256": MODULE._file_sha256(source_path),
            }
        ],
    }
    implementation = {
        **implementation_body,
        "implementation_sha256": MODULE._sha(implementation_body),
    }

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="alias that could evade",
    ):
        MODULE._audit_authenticated_compiler_sources(implementation)


def test_compiled_candidate_context_exposes_exact_authenticated_bindings() -> None:
    context, target, parent = _compiled_consumer_context()

    assert context.target_model is target
    assert context.forbidden_parent_model is parent
    assert (
        context.blank_target_report["target_plan_fingerprint"]
        == context.target_plan_fingerprint
    )
    assert (
        context.compiled_model_fold_report["result_sha256"]
        == context.compiler_receipt_sha256
    )
    assert context.zero_operation_audit["clean"] is True
    assert all(
        context.zero_operation_audit[name] == 0
        for name in MODULE._CONSUMER_ZERO_COUNTER_FIELDS
    )


def test_compiled_candidate_consumer_accepts_ordinary_quarantine_capture(
    tmp_path: Path,
) -> None:
    context, _target, _parent = _compiled_consumer_context()
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "ordinary-quarantine",
    )

    result = MODULE._invoke_compiled_candidate_consumer(
        context,
        lambda inner: _compiled_consumer_result(
            inner,
            candidate=candidate,
        ),
    )

    assert type(result) is MODULE.V5CompiledCandidateConsumerResult
    assert result.compiler_receipt_sha256 == context.compiler_receipt_sha256
    assert result.capture_receipt_sha256 != result.compiler_receipt_sha256
    assert result.compiler_child_checkpoint_saved is False
    assert result.quarantine_candidate_saved is True
    assert result.quarantine_publication_nonpromotable is True
    assert dict(result.compile_diagnostic) == dict(context.compile_diagnostic)
    assert dict(result.verification_receipt) == dict(candidate.verification_receipt)
    assert result.consumer_runtime_guard_receipt["clean"] is True
    assert (
        result.consumer_runtime_guard_receipt["capture_verification_sha256"]
        == result.verification_receipt["verification_sha256"]
    )


def test_compiled_candidate_result_rejects_duck_typed_candidate(
    tmp_path: Path,
) -> None:
    context, _target, _parent = _compiled_consumer_context()
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "exact-type-quarantine",
    )
    fake = SimpleNamespace(
        path=candidate.path,
        artifact_path=candidate.artifact_path,
        model=candidate.model,
        receipt=candidate.receipt,
        verification_receipt=candidate.verification_receipt,
        promotable=False,
        publication_authorized=False,
        requires_reviewed_rerun=True,
    )

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="exact generic quarantined candidate type",
    ):
        _compiled_consumer_result(context, candidate=fake)


def test_compiled_candidate_result_rejects_resealed_verification_tamper(
    tmp_path: Path,
) -> None:
    context, _target, _parent = _compiled_consumer_context()
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "verification-tamper-quarantine",
    )
    tampered = copy.deepcopy(dict(candidate.verification_receipt))
    tampered["output_sha256"] = _consumer_digest("unrelated-output")
    tampered_body = dict(tampered)
    tampered_body.pop("verification_sha256")
    tampered["verification_sha256"] = MODULE._sha(tampered_body)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="Final retained-destination reload verification",
    ):
        _compiled_consumer_result(
            context,
            candidate=candidate,
            verification_receipt=tampered,
        )


def test_outer_consumer_guard_rejects_swallowed_pruning_attempt(
    tmp_path: Path,
) -> None:
    from torch.nn.utils import prune

    context, target, _parent = _compiled_consumer_context()
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "consumer-pruning-quarantine",
    )

    def attempt_pruning(inner):
        try:
            prune.l1_unstructured(target, "weight", amount=0.5)
        except Exception:
            pass
        return _compiled_consumer_result(inner, candidate=candidate)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="Consumer runtime fold-only guard receipt",
    ):
        MODULE._invoke_compiled_candidate_consumer(
            context,
            attempt_pruning,
        )


def test_compiled_candidate_consumer_rejects_target_mutation(
    tmp_path: Path,
) -> None:
    context, target, _parent = _compiled_consumer_context()
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "mutation-quarantine",
    )

    def mutate(inner):
        with pytest.importorskip("torch").no_grad():
            next(target.parameters()).add_(1.0)
        return _compiled_consumer_result(inner, candidate=candidate)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="mutated",
    ):
        MODULE._invoke_compiled_candidate_consumer(context, mutate)


def test_compiled_candidate_consumer_rejects_target_gradients(
    tmp_path: Path,
) -> None:
    context, target, _parent = _compiled_consumer_context()
    torch = pytest.importorskip("torch")
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "gradient-quarantine",
    )

    def attach_gradient(inner):
        parameter = next(target.parameters())
        parameter.grad = torch.ones_like(parameter)
        return _compiled_consumer_result(inner, candidate=candidate)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="mutated",
    ):
        MODULE._invoke_compiled_candidate_consumer(context, attach_gradient)


@pytest.mark.parametrize("direct_forward", (False, True))
def test_compiled_candidate_consumer_rejects_parent_inference(
    direct_forward: bool,
    tmp_path: Path,
) -> None:
    context, _target, parent = _compiled_consumer_context()
    torch = pytest.importorskip("torch")
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / f"parent-call-{direct_forward}",
    )

    def call_parent(inner):
        if direct_forward:
            parent.forward(torch.zeros(1, 3))
        else:
            parent(torch.zeros(1, 3))
        return _compiled_consumer_result(inner, candidate=candidate)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="forbidden live parent inference",
    ):
        MODULE._invoke_compiled_candidate_consumer(context, call_parent)


def test_compiled_candidate_consumer_rejects_wrong_result_type() -> None:
    context, _target, _parent = _compiled_consumer_context()

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="wrong exact result type",
    ):
        MODULE._invoke_compiled_candidate_consumer(
            context,
            lambda _inner: SimpleNamespace(capture="untyped"),
        )


def test_compiled_candidate_consumer_rejects_unrelated_typed_receipt(
    tmp_path: Path,
) -> None:
    context, _target, _parent = _compiled_consumer_context()
    wrong_parent = _consumer_digest("wrong-parent")
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "wrong-parent-quarantine",
        parent_sha256=wrong_parent,
    )

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="differs from the exact",
    ):
        MODULE._invoke_compiled_candidate_consumer(
            context,
            lambda inner: _compiled_consumer_result(
                inner,
                candidate=candidate,
                parent_sha256=wrong_parent,
            ),
        )


def test_compiled_candidate_result_rejects_unrelated_candidate_model(
    tmp_path: Path,
) -> None:
    context, _target, _parent = _compiled_consumer_context()
    candidate = _capture_consumer_candidate(
        context,
        tmp_path / "unrelated-model-quarantine",
    )
    with pytest.importorskip("torch").no_grad():
        next(candidate.model.parameters()).add_(1.0)

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="candidate model state differs",
    ):
        _compiled_consumer_result(
            context,
            candidate=candidate,
        )


def test_current_plus_next_budget_includes_final_norm() -> None:
    references = (
        SimpleNamespace(logical_bytes=10),
        SimpleNamespace(logical_bytes=20),
        SimpleNamespace(logical_bytes=15),
    )

    resolved, required = MODULE._resolved_logical_resident_bytes(
        references=references,
        final_norm_logical_bytes=44,
        requested_gib=0.0,
    )

    assert MODULE._maximum_current_plus_next_bytes(references) == 35
    assert required == 44
    assert resolved == 44


def test_absolute_rss_cap_tightens_the_relative_compiler_policy() -> None:
    fraction, allowed = MODULE._derive_effective_rss_reduction_fraction(
        full_phase_tree_logical_bytes=1_000,
        compiler_baseline_rss_bytes=9_700,
        maximum_process_rss_bytes=10_000,
        requested_minimum_reduction_fraction=0.10,
    )

    assert allowed == 300
    assert fraction >= 0.70
    assert int(1_000 * (1.0 - fraction)) <= allowed


def test_absolute_rss_policy_rejects_an_over_limit_baseline() -> None:
    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="baseline RSS already exceeds",
    ):
        MODULE._derive_effective_rss_reduction_fraction(
            full_phase_tree_logical_bytes=1_000,
            compiler_baseline_rss_bytes=10_001,
            maximum_process_rss_bytes=10_000,
            requested_minimum_reduction_fraction=0.10,
        )


def test_current_two_gpu_policy_receipt_matches_live_replay_sources() -> None:
    receipt = MODULE._load_proven_gpu_parity_receipt()

    assert receipt["receipt_sha256"] == (
        "2a80198832f265a5f1621d57c03081c1f612fc13848de99e3786727b8e2b26a3"
    )
    assert receipt["file_sha256"] == (
        "1431076192c4596244f92f4b46b52a0447e328cd3752ba6ff97a24ec5298f55b"
    )
    assert receipt["real_two_gpu_parity_passed"] is True


def test_historical_two_gpu_policy_receipt_rejects_changed_sources() -> None:
    historical = MODULE._REPOSITORY_ROOT / "reports" / "12061b4-two-rtx3090-bf16.json"

    with pytest.raises(
        MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="GPU replay source changed",
    ):
        MODULE._load_proven_gpu_parity_receipt(historical)


def test_failure_diagnostic_preserves_structured_reports_from_causes() -> None:
    nested_body = {
        "schema": "test-fold-failure-v1",
        "stage": "short_conv",
        "promotion_claimed": False,
    }
    nested = RuntimeError("inner failure")
    nested.report = {
        **nested_body,
        "report_sha256": MODULE._sha(nested_body),
    }
    outer = MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError("outer failure")
    outer.diagnostic_report = {
        "schema": MODULE.FAILURE_SCHEMA,
        "failure_evidence_only": True,
    }
    outer.__cause__ = nested

    diagnostic = MODULE.build_failure_diagnostic(outer)

    assert diagnostic["status"] == "failed"
    assert diagnostic["structured_failure_report_count"] == 2
    assert (
        diagnostic["preserved_structured_failure_reports"][1]["report"] == nested.report
    )
    body = dict(diagnostic)
    claimed = body.pop("report_sha256")
    assert claimed == MODULE._sha(body)
    assert diagnostic["publication_claimed"] is False
    assert diagnostic["child_checkpoint_saved"] is False


def test_failure_diagnostic_preserves_real_adapter_diagnostics() -> None:
    inner = RuntimeError("real adapter gate failed")
    inner.diagnostics = {
        "schema": "test-real-v5-adapter-diagnostics-v1",
        "kind": "fir_commutator_gate_failure_evidence",
        "requested_gate": 1.0,
        "observed": 1.25,
    }
    outer = MODULE.Lfm2CpcfV5SelectiveBlankFoldRunnerError("outer failure")
    outer.__cause__ = inner

    diagnostic = MODULE.build_failure_diagnostic(outer)

    preserved = diagnostic["preserved_structured_failure_reports"]
    assert len(preserved) == 1
    assert preserved[0]["name"] == "diagnostics"
    assert preserved[0]["report"] == inner.diagnostics


def test_main_persists_failure_diagnostic_without_loading_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic_path = tmp_path / "failure.json"
    argv = _required_args(tmp_path)
    argv[argv.index("--diagnostic-json") + 1] = str(diagnostic_path)
    nested_body = {
        "schema": "test-runtime-failure-v1",
        "stage": "catalog",
        "promotion_claimed": False,
    }

    def fail(_config):
        error = RuntimeError("synthetic pre-CUDA failure")
        error.report = {
            **nested_body,
            "report_sha256": MODULE._sha(nested_body),
        }
        raise error

    monkeypatch.setattr(MODULE, "run", fail)

    assert MODULE.main(argv) == 2
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert diagnostic["status"] == "failed"
    assert diagnostic["structured_failure_report_count"] == 1
    assert (
        diagnostic["preserved_structured_failure_reports"][0]["report"]["stage"]
        == "catalog"
    )
    assert diagnostic["publication_claimed"] is False


def test_runner_source_has_no_disallowed_execution_or_loader_lane() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        ".back" + "ward(",
        "torch." + "optim",
        "from_" + "pretrained(",
        "build_lfm2_" + "circuit_trace_cache(",
        "save_" + "pretrained(",
    )

    assert all(token not in source for token in forbidden)
    assert "v5_compiler.compile_lfm2_cpcf_v5_selective_model_fold," in source
    assert "Lfm2CpcfV5EndpointFirstCandidateSelector()" in source
    assert "_build_preregistered_real_v5_numerical_adapter()" in source
    assert '"numerical_frontier_adapter": numerical_adapter' in source
    assert "require_short_conv_v5_real_numerical_frontier_adapter=True" in source
    assert "require_short_conv_v5_authenticated_whole_macrocell_selector=True" in source
    assert (
        '"short_conv_v5_authenticated_whole_macrocell_selection_integrated"' in source
    )
    assert '"short_conv_v5_macrocell_selector_final_authority"' in source
    assert '"short_conv_v5_endpoint_first_selector_final_authority"' in source
    assert '"short_conv_v5_endpoint_prefilter_control_integrated"' in source
    assert '"macrocell_candidate_selector_layer_indices"' in source
    assert "open_lfm2_cpcf_selective_evidence_provider_from_catalog(" in source
    assert "prepare_selective_catalog(" in source
    assert "expected_selective_catalog_receipt_sha256" in source
    assert "selective_catalog_post_run_reverification" in source
    assert "plan_and_synthesize_lfm2_cpcf_blank(" in source
    assert "Lfm2CpcfV5ResumeCoordinator" in source
    assert "coordinator.open_admitted(" in source
    assert "resume_state=resume_runtime.state" in source
    assert "resume_coordinator=resume_runtime.coordinator" in source
    assert '"resume_supported") is not True' in source
    assert '"resume_fail_closed") is not True' in source
    assert '"resume_parent_or_cache_tensor_sidecars": 0' in source
    assert '"child_checkpoint_saved": False' in source
    assert '"publication_attempted": False' in source
    assert "GPU_LOGICAL_DEVICES = (0, 1)" in source
    assert hashlib.sha256(source.encode("utf-8")).hexdigest()
