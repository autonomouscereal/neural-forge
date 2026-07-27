from __future__ import annotations

from types import MappingProxyType

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack  # noqa: E402
import neural_forge.hf_lfm2_semantic_attention_group_qdeim_installer as installer  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_semantic_attention_source_local_output import (  # noqa: E402
    compile_install_lfm2_semantic_attention_source_local_output,
)
from neural_forge.hf_lfm2_semantic_body_recipe import (  # noqa: E402
    resolve_lfm2_semantic_body_recipe,
)
from tests.test_hf_lfm2_attention_group_qdeim import (  # noqa: E402
    _source_local_output_fit_case,
)
from tests.test_hf_lfm2_semantic_attention_group_qdeim_installer import (  # noqa: E402
    _Model,
    _candidate,
)
from tests.test_hf_lfm2_semantic_coupled_attention_layer_installation import (  # noqa: E402
    _wrapper_arguments,
)


_RECIPE_ID = "nonlinear_qdeim_coupled_attention_source_local_residual_fit_corrected"


def _reseal(value: object) -> MappingProxyType:
    body = dict(value)
    body.pop("result_sha256", None)
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


@pytest.fixture()
def composite_receipt(monkeypatch: pytest.MonkeyPatch) -> MappingProxyType:
    model = _Model()
    captured: dict[str, object] = {}

    def compiler(**kwargs: object) -> object:
        candidate = _candidate(**kwargs)
        captured["structural"] = candidate.compiled
        return candidate

    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        compiler,
    )
    arguments = _wrapper_arguments(model)
    arguments["body_recipe"] = _RECIPE_ID
    structural_receipt = stack._install_coupled_attention_group_qdeim_operator(
        **arguments
    )
    structural = captured["structural"]
    fit_case = _source_local_output_fit_case(structural, action_scale=64.0)
    output_result = compile_install_lfm2_semantic_attention_source_local_output(
        target_model=model,
        layer_index=0,
        structural_candidate=structural,
        fit_cases=(fit_case,),
    )
    output = output_result.receipt
    compiler_receipt = output_result.compiled.report
    recipe = resolve_lfm2_semantic_body_recipe(_RECIPE_ID)
    structural_replay_bank_sha256 = "7" * 64
    post_o_replay_bank_sha256 = "8" * 64
    compiler_cases = compiler_receipt["evidence"]["cases"]
    fit_cases_receipt = _reseal(
        {
            "schema": (
                "neural-forge-lfm2-semantic-attention-source-local-fit-cases-v1"
            ),
            "layer_index": 0,
            "partition": "fit",
            "fit_batch_indices": [0],
            "heldout_batch_indices_used": [],
            "fit_case_count": len(compiler_cases),
            "fit_row_count": compiler_receipt["topology"]["fit_row_count"],
            "fit_cases": [
                {
                    "case_id": value["case_id"],
                    "batch_index": 0,
                    "row_count": value["row_count"],
                    "total_row_mass": sum(value["row_sample_weights"]),
                    "parent_trajectory_receipt_sha256": value[
                        "parent_trajectory_receipt_sha256"
                    ],
                    "live_child_replay_receipt_sha256": value[
                        "live_child_replay_receipt_sha256"
                    ],
                    "aligned_row_identity_sha256": value["aligned_row_identity_sha256"],
                }
                for value in compiler_cases
            ],
            "fit_evidence_sha256": output["source_local_fit_evidence_sha256"],
            "structural_replay_bank_sha256": structural_replay_bank_sha256,
            "structural_replay_model_state_sha256": output["model_state_before_sha256"],
            "parent_forward_calls": 0,
            "target_model_forward_calls": 0,
            "training_steps": 0,
        }
    )
    structural_replay = stack._LayerReplayBank(
        captured_child_model_state_sha256=output["model_state_before_sha256"],
        layer_index=0,
        pre_residual_by_batch=MappingProxyType({}),
        replay_by_batch=MappingProxyType({}),
        bank_sha256=structural_replay_bank_sha256,
    )
    post_o_replay = stack._LayerReplayBank(
        captured_child_model_state_sha256=output["model_state_after_sha256"],
        layer_index=0,
        pre_residual_by_batch=MappingProxyType({}),
        replay_by_batch=MappingProxyType({}),
        bank_sha256=post_o_replay_bank_sha256,
    )
    return stack._compose_source_local_attention_operator_receipt(
        structural_operator_receipt=structural_receipt,
        source_local_installation=output_result,
        fit_cases_receipt=fit_cases_receipt,
        structural_replay=structural_replay,
        post_source_local_replay=post_o_replay,
        layer_index=0,
        parameter_count=structural_receipt["parameter_element_count_before"],
        body_recipe=recipe,
    )


def _authenticate(receipt: object) -> object:
    return postfold._authenticate_lfm2_semantic_attention_operator_receipt(
        receipt,
        layer_index=0,
        body_recipe=_RECIPE_ID,
    )


def test_authenticates_composite_and_returns_final_corrected_o(
    composite_receipt: MappingProxyType,
) -> None:
    authenticated = _authenticate(composite_receipt)
    structural = composite_receipt["structural_attention_operator_receipt"]
    output = composite_receipt["source_local_output_installation_receipt"]
    structural_authenticated = _authenticate(structural)

    assert authenticated["installed"] is True
    assert authenticated["coupled_attention_group_qdeim"] is True
    assert authenticated["source_local_attention_output"] is True
    assert (
        authenticated["changed_state_keys"]
        == structural_authenticated["changed_state_keys"]
    )
    assert authenticated["six_tensor_bundle"] == output["attention_tensor_sha256_after"]
    assert (
        authenticated["six_tensor_bundle"]["out_proj.weight"]
        != (structural_authenticated["six_tensor_bundle"]["out_proj.weight"])
    )
    for role in (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
    ):
        assert (
            authenticated["six_tensor_bundle"][role]
            == (structural_authenticated["six_tensor_bundle"][role])
        )
    assert authenticated["transport_evidence_sha256"] == [
        *structural_authenticated["transport_evidence_sha256"],
        output["source_local_fit_evidence_sha256"],
    ]


def test_rejects_resealed_outer_state_chain_mismatch(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    body["model_state_before_sha256"] = "0" * 64
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="inconsistent"):
        _authenticate(_reseal(body))


def test_rejects_resealed_second_transaction_that_claims_non_o_mutation(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    output = dict(body["source_local_output_installation_receipt"])
    output["changed_state_keys"] = ["model.layers.0.self_attn.q_proj.weight"]
    body["source_local_output_installation_receipt"] = dict(_reseal(output))
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="inconsistent"):
        _authenticate(_reseal(body))


def test_rejects_resealed_transport_evidence_omission(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    body["transport_evidence_sha256"] = body["transport_evidence_sha256"][:-1]
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="inconsistent"):
        _authenticate(_reseal(body))


def test_rejects_resealed_source_local_lineage_tampering(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    output = dict(body["source_local_output_installation_receipt"])
    output["source_local_corrected_candidate_state_sha256"] = "f" * 64
    body["source_local_output_installation_receipt"] = dict(_reseal(output))
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="inconsistent"):
        _authenticate(_reseal(body))


def test_rejects_nested_receipt_tampering_even_when_outer_is_resealed(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    output = dict(body["source_local_output_installation_receipt"])
    output["training_steps"] = 1
    body["source_local_output_installation_receipt"] = output
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="self-seal"):
        _authenticate(_reseal(body))


def test_rejects_resealed_compiler_that_claims_a_prohibited_operation(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    compiler = dict(body["source_local_output_compiler_receipt"])
    prohibited = dict(compiler["prohibited_operations"])
    prohibited["training_steps"] = 1
    compiler["prohibited_operations"] = prohibited
    body["source_local_output_compiler_receipt"] = dict(_reseal(compiler))
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="inconsistent"):
        _authenticate(_reseal(body))


def test_rejects_resealed_fit_receipt_with_wrong_replay_lineage(
    composite_receipt: MappingProxyType,
) -> None:
    body = dict(composite_receipt)
    fit_cases = dict(body["source_local_output_fit_cases_receipt"])
    fit_cases["structural_replay_bank_sha256"] = "e" * 64
    body["source_local_output_fit_cases_receipt"] = dict(_reseal(fit_cases))
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="inconsistent"):
        _authenticate(_reseal(body))
