from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from neural_forge.hf_lfm2_fused_attention_diagnostic import (
    Lfm2FusedAttentionDiagnosticError,
    Lfm2FusedAttentionDiagnosticInput,
    authenticate_lfm2_fused_attention_candidate,
    authenticate_lfm2_fused_attention_reconstruction_descriptor,
    compile_lfm2_fused_attention_diagnostic_bank,
    fingerprint_lfm2_fused_attention_diagnostic_input,
    lfm2_fused_attention_candidate_state_sha256,
    lfm2_fused_attention_parameter_count,
    reconstruct_lfm2_fused_attention_candidate,
    substitute_lfm2_fused_attention_gqa_group,
)


def _tensor(shape: tuple[int, ...], *, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, dtype=torch.float64)


def _source() -> Lfm2FusedAttentionDiagnosticInput:
    generator = torch.Generator(device="cpu").manual_seed(1701)
    parent_hidden = 5
    compact_hidden = 3
    source_query_heads = 6
    source_key_value_heads = 3
    target_query_heads = 4
    target_key_value_heads = 2
    head_dim = 2
    source_query_width = source_query_heads * head_dim
    source_key_value_width = source_key_value_heads * head_dim
    target_query_width = target_query_heads * head_dim
    target_key_value_width = target_key_value_heads * head_dim
    pristine_out = _tensor(
        (compact_hidden, target_query_width),
        generator=generator,
    )
    final_out = pristine_out + 0.125 * _tensor(
        (compact_hidden, target_query_width),
        generator=generator,
    )
    pristine_query = _tensor(
        (target_query_width, compact_hidden),
        generator=generator,
    )
    pristine_key = _tensor(
        (target_key_value_width, compact_hidden),
        generator=generator,
    )
    pristine_value = _tensor(
        (target_key_value_width, compact_hidden),
        generator=generator,
    )
    final_query = pristine_query + 0.125 * _tensor(
        (target_query_width, compact_hidden),
        generator=generator,
    )
    final_key = pristine_key + 0.125 * _tensor(
        (target_key_value_width, compact_hidden),
        generator=generator,
    )
    final_value = pristine_value + 0.125 * _tensor(
        (target_key_value_width, compact_hidden),
        generator=generator,
    )
    parent_query_norm = _tensor((head_dim,), generator=generator)
    parent_key_norm = _tensor((head_dim,), generator=generator)
    return Lfm2FusedAttentionDiagnosticInput(
        layer_index=1,
        source_query_head_count=source_query_heads,
        source_key_value_head_count=source_key_value_heads,
        target_query_head_count=target_query_heads,
        target_key_value_head_count=target_key_value_heads,
        head_dim=head_dim,
        selected_source_key_value_groups=(2, 0),
        selected_source_query_heads=(4, 5, 0, 1),
        norm_eps=1e-5,
        attention_implementation="eager",
        parent_query_weight=_tensor(
            (source_query_width, parent_hidden),
            generator=generator,
        ),
        parent_key_weight=_tensor(
            (source_key_value_width, parent_hidden),
            generator=generator,
        ),
        parent_value_weight=_tensor(
            (source_key_value_width, parent_hidden),
            generator=generator,
        ),
        parent_out_weight=_tensor(
            (parent_hidden, source_query_width),
            generator=generator,
        ),
        parent_query_norm_weight=parent_query_norm,
        parent_key_norm_weight=parent_key_norm,
        compact_pristine_query_weight=pristine_query,
        compact_pristine_key_weight=pristine_key,
        compact_pristine_value_weight=pristine_value,
        compact_pristine_out_weight=pristine_out,
        compact_final_query_weight=final_query,
        compact_final_key_weight=final_key,
        compact_final_value_weight=final_value,
        compact_final_out_weight=final_out,
        compact_query_norm_weight=parent_query_norm.clone(),
        compact_key_norm_weight=parent_key_norm.clone(),
        input_encoder=_tensor(
            (parent_hidden, compact_hidden),
            generator=generator,
        ),
        output_decoder=_tensor(
            (compact_hidden, parent_hidden),
            generator=generator,
        ),
        context_reconstruction=_tensor(
            (target_query_width, source_query_width),
            generator=generator,
        ),
    )


def _pins(source: Lfm2FusedAttentionDiagnosticInput) -> tuple[str, dict[str, str]]:
    authentication = fingerprint_lfm2_fused_attention_diagnostic_input(source)
    return authentication["input_bundle_sha256"], {
        name: record["sha256"]
        for name, record in authentication["tensor_records"].items()
    }


def _bank(source: Lfm2FusedAttentionDiagnosticInput):
    bundle, pins = _pins(source)
    return compile_lfm2_fused_attention_diagnostic_bank(
        source,
        expected_input_bundle_sha256=bundle,
        expected_tensor_sha256=pins,
    )


def _parent_config(source: Lfm2FusedAttentionDiagnosticInput) -> Any:
    from transformers import Lfm2Config

    config = Lfm2Config(
        vocab_size=32,
        hidden_size=source.parent_query_weight.shape[1],
        intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=source.source_query_head_count,
        num_key_value_heads=source.source_key_value_head_count,
        head_dim=source.head_dim,
        layer_types=["full_attention"] * 3,
        norm_eps=source.norm_eps,
    )
    config._attn_implementation = source.attention_implementation
    return config


def _reconstruct(
    candidate: Any,
    source: Lfm2FusedAttentionDiagnosticInput,
    *,
    parent_config: Any | None = None,
) -> Any:
    return reconstruct_lfm2_fused_attention_candidate(
        candidate=candidate,
        parent_config=(
            _parent_config(source) if parent_config is None else parent_config
        ),
        expected_candidate_result_sha256=candidate.receipt["result_sha256"],
        expected_descriptor_sha256=(
            candidate.reconstruction_descriptor["descriptor_sha256"]
        ),
        device="cpu",
    )


def _storage_alias_group_count(tensors: dict[str, torch.Tensor]) -> int:
    groups: dict[tuple[str, str, int, int], list[str]] = {}
    for name, tensor in tensors.items():
        identity = (
            str(tensor.device),
            str(tensor.dtype),
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.untyped_storage().nbytes()),
        )
        groups.setdefault(identity, []).append(name)
    return sum(len(names) > 1 for names in groups.values())


def test_bank_materializes_native_qk_vo_o_and_fused_variants() -> None:
    source = _source()
    bank = _bank(source)
    assert tuple(bank.candidates) == (
        "native-selected",
        "qk-structural-s0-only",
        "qk-final-s2-only",
        "v-structural-s0-only",
        "v-final-s2-only",
        "o-structural-s0-only",
        "o-final-s2-only",
        "vo-structural-s0",
        "vo-final-s2",
        "qk-structural-s0-vo-final-s2",
        "qk-final-s2-vo-structural-s0",
        "fused-qkvo-structural-s0",
        "fused-qkvo-final-s2",
    )
    assert bank.receipt["compact_qkvo_artifact_lanes"] == {
        "pristine_qkvo_s0_supplied_and_authenticated": True,
        "final_qkvo_post_global_s2_supplied_and_authenticated": True,
        "source_local_only_s1_bytes_supplied_and_authenticated": False,
        "source_local_only_s1_inferred_from_receipt_hash_without_bytes": False,
    }
    assert bank.receipt["compact_s0_s2_byte_equality_by_role"] == {
        "q": False,
        "k": False,
        "v": False,
        "out": False,
    }
    native = bank.candidates["native-selected"].state_dict()
    qk_pristine = bank.candidates["qk-structural-s0-only"].state_dict()
    qk_final = bank.candidates["qk-final-s2-only"].state_dict()
    vo_pristine = bank.candidates["vo-structural-s0"].state_dict()
    vo_final = bank.candidates["vo-final-s2"].state_dict()
    fused = bank.candidates["fused-qkvo-final-s2"].state_dict()
    query_rows = [8, 9, 10, 11, 0, 1, 2, 3]
    key_value_rows = [4, 5, 0, 1]
    torch.testing.assert_close(
        native["q_proj.weight"],
        source.parent_query_weight[query_rows],
    )
    torch.testing.assert_close(
        native["k_proj.weight"],
        source.parent_key_weight[key_value_rows],
    )
    torch.testing.assert_close(
        native["v_proj.weight"],
        source.parent_value_weight[key_value_rows],
    )
    torch.testing.assert_close(
        native["out_proj.weight"],
        source.parent_out_weight @ source.context_reconstruction.T,
    )
    torch.testing.assert_close(
        qk_pristine["q_proj.weight"],
        source.compact_pristine_query_weight @ source.input_encoder.T,
    )
    torch.testing.assert_close(
        qk_pristine["k_proj.weight"],
        source.compact_pristine_key_weight @ source.input_encoder.T,
    )
    torch.testing.assert_close(
        qk_final["q_proj.weight"],
        source.compact_final_query_weight @ source.input_encoder.T,
    )
    torch.testing.assert_close(qk_pristine["v_proj.weight"], native["v_proj.weight"])
    torch.testing.assert_close(
        qk_pristine["out_proj.weight"], native["out_proj.weight"]
    )
    torch.testing.assert_close(
        vo_pristine["v_proj.weight"],
        source.compact_pristine_value_weight @ source.input_encoder.T,
    )
    torch.testing.assert_close(
        vo_pristine["out_proj.weight"],
        source.output_decoder.T @ source.compact_pristine_out_weight,
    )
    torch.testing.assert_close(
        vo_final["out_proj.weight"],
        source.output_decoder.T @ source.compact_final_out_weight,
    )
    torch.testing.assert_close(
        vo_final["v_proj.weight"],
        source.compact_final_value_weight @ source.input_encoder.T,
    )
    assert not torch.equal(
        vo_pristine["out_proj.weight"],
        vo_final["out_proj.weight"],
    )
    torch.testing.assert_close(
        fused["q_proj.weight"],
        source.compact_final_query_weight @ source.input_encoder.T,
    )
    torch.testing.assert_close(
        fused["v_proj.weight"],
        source.compact_final_value_weight @ source.input_encoder.T,
    )
    for candidate in bank.candidates.values():
        state = candidate.state_dict()
        torch.testing.assert_close(
            state["q_layernorm.weight"],
            source.parent_query_norm_weight,
        )
        torch.testing.assert_close(
            state["k_layernorm.weight"],
            source.parent_key_norm_weight,
        )
    assert bank.receipt["operation_audit"]["parent_model_forward_calls"] == 0
    assert bank.receipt["operation_audit"]["training_steps"] == 0


def test_source_local_only_o_lane_requires_authenticated_external_bytes() -> None:
    source = _source()
    local_out = (
        source.compact_pristine_out_weight
        + 0.25 * torch.ones_like(source.compact_pristine_out_weight)
    ).contiguous()
    source = replace(
        source,
        compact_local_corrected_out_weight=local_out,
    )
    bundle, pins = _pins(source)
    bank = compile_lfm2_fused_attention_diagnostic_bank(
        source,
        expected_input_bundle_sha256=bundle,
        expected_tensor_sha256=pins,
    )
    assert tuple(bank.candidates)[-3:] == (
        "o-local-s1-only",
        "vo-structural-v-s0-local-o-s1",
        "fused-structural-qkv-s0-local-o-s1",
    )
    assert bank.receipt["compact_qkvo_artifact_lanes"] == {
        "pristine_qkvo_s0_supplied_and_authenticated": True,
        "final_qkvo_post_global_s2_supplied_and_authenticated": True,
        "source_local_only_s1_bytes_supplied_and_authenticated": True,
        "source_local_only_s1_inferred_from_receipt_hash_without_bytes": False,
    }
    torch.testing.assert_close(
        bank.candidates["o-local-s1-only"].out_weight,
        source.output_decoder.T @ local_out,
    )

    local_out[0, 0] += 1.0
    with pytest.raises(Lfm2FusedAttentionDiagnosticError, match="independent pins"):
        compile_lfm2_fused_attention_diagnostic_bank(
            source,
            expected_input_bundle_sha256=bundle,
            expected_tensor_sha256=pins,
        )


def test_byte_identical_s0_s2_qkvo_is_authenticated_and_recorded_as_noop() -> None:
    source = _source()
    source = replace(
        source,
        compact_final_query_weight=source.compact_pristine_query_weight.clone(),
        compact_final_key_weight=source.compact_pristine_key_weight.clone(),
        compact_final_value_weight=source.compact_pristine_value_weight.clone(),
        compact_final_out_weight=source.compact_pristine_out_weight.clone(),
    )
    bank = _bank(source)
    assert bank.receipt["compact_s0_s2_byte_equality_by_role"] == {
        "q": True,
        "k": True,
        "v": True,
        "out": True,
    }
    assert bank.receipt["compact_s0_s2_full_qkvo_byte_equal"] is True
    assert bank.receipt["compact_s0_s2_stage_transition_is_authenticated_noop"] is True
    assert bank.receipt["candidate_deduplication_performed"] is False
    assert (
        bank.candidates["fused-qkvo-structural-s0"].receipt["candidate_state_sha256"]
        == bank.candidates["fused-qkvo-final-s2"].receipt["candidate_state_sha256"]
    )


def test_candidates_are_literal_smaller_unshared_states_without_sidecars() -> None:
    source = _source()
    bank = _bank(source)
    original = lfm2_fused_attention_parameter_count(
        hidden_size=5,
        query_head_count=6,
        key_value_head_count=3,
        head_dim=2,
    )
    fused = lfm2_fused_attention_parameter_count(
        hidden_size=5,
        query_head_count=4,
        key_value_head_count=2,
        head_dim=2,
    )
    assert original == 184
    assert fused == 124
    for candidate in bank.candidates.values():
        authenticated = authenticate_lfm2_fused_attention_candidate(candidate)
        physical = authenticated["physical_parameter_accounting"]
        state = dict(candidate.state_dict())
        assert sum(tensor.numel() for tensor in state.values()) == fused
        assert _storage_alias_group_count(state) == 0
        assert physical == {
            "original_parent_attention_parameter_elements": original,
            "fused_candidate_parameter_elements": fused,
            "physical_parameter_reduction": original - fused,
            "fused_candidate_parameter_bytes": fused * 8,
            "input_bridge_parameter_elements": 0,
            "output_bridge_parameter_elements": 0,
            "sidecar_parameter_elements": 0,
            "runtime_shared_parameter_elements": 0,
            "stored_tensor_elements": fused,
            "parameter_reduction_attributed_to_sharing": False,
            "parameter_reduction_attributed_to_quantization": False,
            "parameter_reduction_attributed_to_pruning": False,
            "parameter_reduction_attributed_to_sparsity": False,
        }
        descriptor = candidate.reconstruction_descriptor
        assert descriptor["input_bridge_module_present"] is False
        assert descriptor["output_bridge_module_present"] is False
        assert descriptor["runtime_parameter_alias_group_count"] == 0
        assert authenticated["operation_audit"]["parameters_quantized"] == 0
        assert authenticated["operation_audit"]["parameters_pruned"] == 0
        assert authenticated["operation_audit"]["training_steps"] == 0


def test_actual_exact4x_lfm2_attention_counts_show_stronger_fused_reduction() -> None:
    parent = lfm2_fused_attention_parameter_count(
        hidden_size=1024,
        query_head_count=16,
        key_value_head_count=8,
        head_dim=64,
    )
    child = lfm2_fused_attention_parameter_count(
        hidden_size=776,
        query_head_count=10,
        key_value_head_count=5,
        head_dim=64,
    )
    fused_parent_io = lfm2_fused_attention_parameter_count(
        hidden_size=1024,
        query_head_count=10,
        key_value_head_count=5,
        head_dim=64,
    )
    explicit_bridge_wrapper = child + 2 * 1024 * 776
    assert parent == 3_145_856
    assert child == 1_490_048
    assert explicit_bridge_wrapper == 3_079_296
    assert parent - explicit_bridge_wrapper == 66_560
    assert fused_parent_io == 1_966_208
    assert parent - fused_parent_io == 1_179_648


def test_fp64_fused_qkvo_matches_explicit_bridge_runtime() -> None:
    from transformers import Lfm2Config
    from transformers.models.lfm2.modeling_lfm2 import Lfm2Attention

    source = _source()
    bank = _bank(source)
    candidate = bank.candidates["fused-qkvo-final-s2"]
    reconstructed = _reconstruct(candidate, source)
    compact_config = Lfm2Config(
        vocab_size=32,
        hidden_size=source.compact_final_query_weight.shape[1],
        intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=source.target_query_head_count,
        num_key_value_heads=source.target_key_value_head_count,
        head_dim=source.head_dim,
        layer_types=["full_attention"] * 3,
        norm_eps=source.norm_eps,
    )
    compact_config._attn_implementation = "eager"
    compact = Lfm2Attention(compact_config, source.layer_index).double().eval()
    compact.load_state_dict(
        {
            "q_proj.weight": source.compact_final_query_weight,
            "k_proj.weight": source.compact_final_key_weight,
            "v_proj.weight": source.compact_final_value_weight,
            "out_proj.weight": source.compact_final_out_weight,
            "q_layernorm.weight": source.compact_query_norm_weight,
            "k_layernorm.weight": source.compact_key_norm_weight,
        },
        strict=True,
    )
    generator = torch.Generator(device="cpu").manual_seed(27)
    hidden = torch.randn(1, 5, 5, generator=generator, dtype=torch.float64)
    cosine = torch.ones(1, 5, source.head_dim, dtype=torch.float64)
    sine = torch.zeros_like(cosine)
    compact_output, _ = compact(
        hidden @ source.input_encoder,
        (cosine, sine),
        None,
    )
    explicit_output = compact_output @ source.output_decoder
    fused_output, _ = reconstructed.module(hidden, (cosine, sine), None)
    torch.testing.assert_close(
        fused_output,
        explicit_output,
        rtol=1e-12,
        atol=1e-12,
    )
    assert reconstructed.receipt["model_forward_calls"] == 0
    assert reconstructed.receipt["training_steps"] == 0


@pytest.mark.parametrize("role_bundle", ["qk", "vo", "qkvo"])
def test_one_intact_gqa_group_substitution_is_exact_and_count_preserving(
    role_bundle: str,
) -> None:
    source = _source()
    bank = _bank(source)
    base = bank.candidates["native-selected"]
    donor = bank.candidates["fused-qkvo-final-s2"]
    candidate = substitute_lfm2_fused_attention_gqa_group(
        base=base,
        donor=donor,
        target_group_index=1,
        role_bundle=role_bundle,  # type: ignore[arg-type]
        variant_id=f"group-001-{role_bundle}",
    )
    authenticated = authenticate_lfm2_fused_attention_candidate(candidate)
    base_state = base.state_dict()
    donor_state = donor.state_dict()
    state = candidate.state_dict()
    if role_bundle in {"qk", "qkvo"}:
        torch.testing.assert_close(
            state["q_proj.weight"][4:8],
            donor_state["q_proj.weight"][4:8],
        )
        torch.testing.assert_close(
            state["k_proj.weight"][2:4],
            donor_state["k_proj.weight"][2:4],
        )
    else:
        torch.testing.assert_close(
            state["q_proj.weight"],
            base_state["q_proj.weight"],
        )
        torch.testing.assert_close(
            state["k_proj.weight"],
            base_state["k_proj.weight"],
        )
    if role_bundle in {"vo", "qkvo"}:
        torch.testing.assert_close(
            state["v_proj.weight"][2:4],
            donor_state["v_proj.weight"][2:4],
        )
        torch.testing.assert_close(
            state["out_proj.weight"][:, 4:8],
            donor_state["out_proj.weight"][:, 4:8],
        )
    else:
        torch.testing.assert_close(
            state["v_proj.weight"],
            base_state["v_proj.weight"],
        )
        torch.testing.assert_close(
            state["out_proj.weight"],
            base_state["out_proj.weight"],
        )
    torch.testing.assert_close(
        state["q_proj.weight"][:4],
        base_state["q_proj.weight"][:4],
    )
    torch.testing.assert_close(
        state["q_layernorm.weight"],
        base_state["q_layernorm.weight"],
    )
    assert (
        authenticated["physical_parameter_accounting"][
            "fused_candidate_parameter_elements"
        ]
        == 124
    )
    assert _storage_alias_group_count(dict(state)) == 0


def test_hash_geometry_gqa_alias_dtype_and_nonfinite_inputs_fail_closed() -> None:
    source = _source()
    bundle, pins = _pins(source)
    source.parent_query_weight[0, 0] += 1.0
    with pytest.raises(
        Lfm2FusedAttentionDiagnosticError,
        match="independent pins",
    ):
        compile_lfm2_fused_attention_diagnostic_bank(
            source,
            expected_input_bundle_sha256=bundle,
            expected_tensor_sha256=pins,
        )

    source = _source()
    bad_heads = replace(
        source,
        selected_source_query_heads=(4, 5, 1, 0),
    )
    with pytest.raises(Lfm2FusedAttentionDiagnosticError, match="intact GQA"):
        fingerprint_lfm2_fused_attention_diagnostic_input(bad_heads)

    aliased = replace(
        source,
        compact_pristine_out_weight=source.compact_final_out_weight,
    )
    with pytest.raises(Lfm2FusedAttentionDiagnosticError, match="aliased"):
        fingerprint_lfm2_fused_attention_diagnostic_input(aliased)

    nonfinite_encoder = source.input_encoder.clone()
    nonfinite_encoder[0, 0] = torch.inf
    nonfinite = replace(source, input_encoder=nonfinite_encoder)
    with pytest.raises(Lfm2FusedAttentionDiagnosticError, match="finite dense"):
        fingerprint_lfm2_fused_attention_diagnostic_input(nonfinite)

    requires_grad = replace(
        source,
        compact_final_query_weight=(
            source.compact_final_query_weight.clone().requires_grad_(True)
        ),
    )
    with pytest.raises(Lfm2FusedAttentionDiagnosticError, match="detached"):
        fingerprint_lfm2_fused_attention_diagnostic_input(requires_grad)

    mixed_dtype = replace(
        source,
        compact_pristine_out_weight=source.compact_pristine_out_weight.float(),
    )
    with pytest.raises(Lfm2FusedAttentionDiagnosticError, match="share one dtype"):
        fingerprint_lfm2_fused_attention_diagnostic_input(mixed_dtype)

    wrong_context = replace(
        source,
        context_reconstruction=source.context_reconstruction[:, :-1].clone(),
    )
    with pytest.raises(Lfm2FusedAttentionDiagnosticError, match="geometry"):
        fingerprint_lfm2_fused_attention_diagnostic_input(wrong_context)


def test_receipt_descriptor_state_and_parent_config_tampering_fail_closed() -> None:
    source = _source()
    bank = _bank(source)
    candidate = bank.candidates["fused-qkvo-final-s2"]
    descriptor = dict(candidate.reconstruction_descriptor)
    descriptor["variant_id"] = "tampered"
    with pytest.raises(Lfm2FusedAttentionDiagnosticError, match="seal"):
        authenticate_lfm2_fused_attention_reconstruction_descriptor(descriptor)

    bad_config = _parent_config(source)
    bad_config.hidden_size = 6
    with pytest.raises(Lfm2FusedAttentionDiagnosticError, match="source geometry"):
        _reconstruct(candidate, source, parent_config=bad_config)

    bad_seam = _parent_config(source)
    bad_seam.norm_eps = 1e-4
    with pytest.raises(
        Lfm2FusedAttentionDiagnosticError,
        match="normalization/attention seam",
    ):
        _reconstruct(candidate, source, parent_config=bad_seam)

    with pytest.raises(Lfm2FusedAttentionDiagnosticError, match="external pin"):
        reconstruct_lfm2_fused_attention_candidate(
            candidate=candidate,
            parent_config=_parent_config(source),
            expected_candidate_result_sha256="0" * 64,
            expected_descriptor_sha256=(
                candidate.reconstruction_descriptor["descriptor_sha256"]
            ),
        )

    candidate.query_weight[0, 0] += 1.0
    with pytest.raises(Lfm2FusedAttentionDiagnosticError, match="not bound"):
        authenticate_lfm2_fused_attention_candidate(candidate)


def test_producer_and_reconstruction_paths_do_not_call_attention_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers.models.lfm2.modeling_lfm2 import Lfm2Attention

    source = _source()
    calls = 0

    def forbidden_forward(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError("producer/reconstruction must not forward attention")

    monkeypatch.setattr(Lfm2Attention, "forward", forbidden_forward)
    bank = _bank(source)
    reconstructed = _reconstruct(
        bank.candidates["fused-qkvo-final-s2"],
        source,
    )
    assert calls == 0
    assert reconstructed.receipt["model_forward_calls"] == 0
    assert reconstructed.receipt["training_steps"] == 0
    assert (
        lfm2_fused_attention_candidate_state_sha256(reconstructed.module.state_dict())
        == reconstructed.receipt["candidate_state_sha256"]
    )
