from __future__ import annotations

from types import MappingProxyType

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack  # noqa: E402
import neural_forge.hf_lfm2_semantic_short_conv_refold_adapter as adapter  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_semantic_body_recipe import (  # noqa: E402
    resolve_lfm2_semantic_body_recipe,
)
from tests.test_hf_lfm2_semantic_short_conv_refold_adapter import (  # noqa: E402
    _ToyModel,
    _compile,
    _digest,
    _evidence_bank,
    _fingerprint,
)


_RECIPE_ID = "nonlinear_qdeim_coupled_attention_source_local_residual_fit_corrected"


def _reseal(value: object) -> MappingProxyType:
    body = dict(value)
    body.pop("result_sha256", None)
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


@pytest.fixture()
def composite_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> MappingProxyType:
    monkeypatch.setattr(adapter, "fingerprint_lfm2_parent", _fingerprint)
    monkeypatch.setattr(stack, "fingerprint_lfm2_parent", _fingerprint)
    model = _ToyModel()
    structural = _compile(
        model,
        _evidence_bank(model),
        fold_compute_device="cpu",
        body_recipe=_RECIPE_ID,
    )
    parameter_count = stack._parameter_count(model)
    structural_replay_bank_sha256 = "7" * 64
    structural_replay = stack._LayerReplayBank(
        captured_child_model_state_sha256=structural["model_state_after_sha256"],
        layer_index=0,
        pre_residual_by_batch=MappingProxyType({}),
        replay_by_batch=MappingProxyType({}),
        bank_sha256=structural_replay_bank_sha256,
    )

    residual_chart = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, -0.25]],
        dtype=torch.float64,
    )
    writer_design = torch.tensor(
        [[-2.0, 0.25], [-1.0, -0.5], [0.5, 1.0], [1.5, -1.0]],
        dtype=torch.bfloat16,
    )
    desired_weight = torch.tensor(
        [[1.25, -0.75], [-0.5, 1.5]],
        dtype=torch.float64,
    )
    desired_bias = torch.tensor([0.5, -0.25], dtype=torch.float64)
    mapped_local_action = writer_design.to(torch.float64) @ desired_weight.T
    mapped_local_action = mapped_local_action + desired_bias
    parent_pre = torch.tensor(
        [
            [0.125, 0.25, 0.5],
            [0.25, 0.5, 1.0],
            [0.5, 1.0, 2.0],
            [1.0, 2.0, 4.0],
        ],
        dtype=torch.float64,
    )
    parent_local_action = torch.cat(
        (
            mapped_local_action,
            torch.zeros((writer_design.shape[0], 1), dtype=torch.float64),
        ),
        dim=1,
    )
    fit_case = adapter.Lfm2SemanticShortConvSourceLocalWriterFitCase(
        case_id="layer-000.batch-0000",
        parent_pre_operator_residual=parent_pre,
        parent_post_operator_residual=parent_pre + parent_local_action,
        live_child_pre_operator_residual=torch.tensor(
            [[0.125, -0.25], [0.25, -0.5], [0.5, -1.0], [1.0, -2.0]],
            dtype=torch.bfloat16,
        ),
        live_child_writer_design=writer_design,
        row_sample_weights=(0.25, 0.25, 0.25, 0.25),
        parent_trajectory_receipt_sha256=_digest("source-local-parent-trajectory"),
        live_child_replay_receipt_sha256=_digest("source-local-child-replay"),
        aligned_row_identity_sha256=_digest("source-local-row-identity"),
    )
    fit_evidence_sha256 = (
        adapter.lfm2_semantic_short_conv_source_local_writer_fit_evidence_sha256(
            (fit_case,)
        )
    )
    fit_receipt = _reseal(
        {
            "schema": (
                "neural-forge-lfm2-semantic-short-conv-source-local-fit-cases-v1"
            ),
            "layer_index": 0,
            "partition": "fit",
            "fit_batch_indices": [0],
            "heldout_batch_indices_used": [],
            "fit_case_count": 1,
            "fit_row_count": 4,
            "fit_cases": [
                {
                    "case_id": fit_case.case_id,
                    "batch_index": 0,
                    "row_count": 4,
                    "total_row_mass": 1.0,
                    "parent_trajectory_receipt_sha256": (
                        fit_case.parent_trajectory_receipt_sha256
                    ),
                    "live_child_replay_receipt_sha256": (
                        fit_case.live_child_replay_receipt_sha256
                    ),
                    "aligned_row_identity_sha256": (
                        fit_case.aligned_row_identity_sha256
                    ),
                }
            ],
            "fit_evidence_sha256": fit_evidence_sha256,
            "structural_replay_bank_sha256": structural_replay_bank_sha256,
            "structural_replay_model_state_sha256": structural[
                "model_state_after_sha256"
            ],
            "parent_forward_calls": 0,
            "target_model_forward_calls": 0,
            "training_steps": 0,
        }
    )
    compiled, installation = stack._compile_install_short_conv_source_local_writer(
        target_model=model,
        layer_index=0,
        structural_operator_receipt=structural,
        structural_replay=structural_replay,
        fit_cases=(fit_case,),
        fit_cases_receipt=fit_receipt,
        residual_output_chart=residual_chart,
        parameter_count=parameter_count,
        fold_compute_device="cpu",
        body_recipe=_RECIPE_ID,
    )
    assert installation["installed"] is True
    post_replay = stack._LayerReplayBank(
        captured_child_model_state_sha256=installation["model_state_after_sha256"],
        layer_index=0,
        pre_residual_by_batch=MappingProxyType({}),
        replay_by_batch=MappingProxyType({}),
        bank_sha256="8" * 64,
    )
    return stack._compose_source_local_short_conv_operator_receipt(
        structural_operator_receipt=structural,
        source_local_compiled=compiled,
        source_local_installation_receipt=installation,
        fit_cases_receipt=fit_receipt,
        structural_replay=structural_replay,
        post_source_local_replay=post_replay,
        layer_index=0,
        parameter_count=parameter_count,
        body_recipe=resolve_lfm2_semantic_body_recipe(_RECIPE_ID),
    )


def _authenticate(receipt: object) -> object:
    return postfold._authenticate_lfm2_source_local_short_conv_output_composite_operator_receipt(
        receipt,
        layer_index=0,
        body_recipe=_RECIPE_ID,
    )


def test_authenticates_structural_plus_source_local_writer_composite(
    composite_receipt: MappingProxyType,
) -> None:
    authenticated = _authenticate(composite_receipt)

    assert authenticated["installed"] is True
    assert authenticated["source_local_short_conv_output"] is True
    assert authenticated["source_local_writer_only_second_stage"] is True
    assert (
        authenticated["transport_evidence_sha256"]
        == composite_receipt["transport_evidence_sha256"]
    )
    assert (
        authenticated["changed_state_keys"] == composite_receipt["changed_state_keys"]
    )
    assert (
        authenticated["model_state_intermediate_sha256"]
        == composite_receipt["structural_replay_model_state_sha256"]
    )


def test_rejects_resealed_outer_state_chain_mismatch(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    body["model_state_intermediate_sha256"] = "0" * 64
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="inconsistent"):
        _authenticate(_reseal(body))


def test_rejects_resealed_second_transaction_outside_writer(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    installation = dict(body["source_local_writer_installation_receipt"])
    installation["changed_state_keys"] = ["model.layers.0.conv.in_proj.weight"]
    body["source_local_writer_installation_receipt"] = dict(_reseal(installation))
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="inconsistent"):
        _authenticate(_reseal(body))


def test_rejects_resealed_structural_plus_fit_evidence_omission(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    body["transport_evidence_sha256"] = body["transport_evidence_sha256"][:-1]
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="inconsistent"):
        _authenticate(_reseal(body))


def test_rejects_nested_installation_tamper_even_if_outer_is_resealed(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    installation = dict(body["source_local_writer_installation_receipt"])
    installation["training_steps"] = 1
    body["source_local_writer_installation_receipt"] = installation
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="self-seal"):
        _authenticate(_reseal(body))


def test_rejects_resealed_compiler_local_target_tamper(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    compiler = dict(body["source_local_writer_compiler_receipt"])
    target = dict(compiler["target"])
    target["correction_target"] = "absolute-parent-endpoint"
    compiler["target"] = target
    body["source_local_writer_compiler_receipt"] = dict(_reseal(compiler))
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="inconsistent"):
        _authenticate(_reseal(body))


def test_rejects_resealed_compiler_prohibited_operation(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    compiler = dict(body["source_local_writer_compiler_receipt"])
    prohibited = dict(compiler["prohibited_operations"])
    prohibited["training_steps"] = 1
    compiler["prohibited_operations"] = prohibited
    body["source_local_writer_compiler_receipt"] = dict(_reseal(compiler))
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="inconsistent"):
        _authenticate(_reseal(body))


def test_rejects_resealed_fit_replay_lineage_tamper(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    fit = dict(body["source_local_writer_fit_cases_receipt"])
    fit["structural_replay_bank_sha256"] = "f" * 64
    body["source_local_writer_fit_cases_receipt"] = dict(_reseal(fit))
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="inconsistent"):
        _authenticate(_reseal(body))


def test_rejects_resealed_cache_or_source_root_tamper(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    body["source_model_identity_sha256"] = "e" * 64
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="inconsistent"):
        _authenticate(_reseal(body))
