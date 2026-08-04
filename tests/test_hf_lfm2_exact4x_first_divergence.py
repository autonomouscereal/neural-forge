from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_counted_child_state import (  # noqa: E402
    ContinuousChildStateAllocation,
)
from neural_forge.dynamic_fold_phase_cache import DynamicFoldPhaseCache  # noqa: E402
from neural_forge.dynamic_model_inspector import inspect_model_circuits  # noqa: E402
import neural_forge.hf_lfm2_exact4x_first_divergence as diagnostic  # noqa: E402
from neural_forge.hf_lfm2_balanced_basis_phase_cache import (  # noqa: E402
    LFM2_BALANCED_BASIS_PHASE,
    Lfm2BalancedBasisPhaseBuildRequest,
    Lfm2BalancedBasisPhaseCacheError,
    Lfm2BalancedBasisPhaseCacheContext,
    PhaseAnchorAuthority,
    load_existing_lfm2_circuit_balanced_basis_phase,
    load_or_build_lfm2_circuit_balanced_basis,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCorruptionError,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceCaptureError,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
    build_lfm2_circuit_trace_cache,
    capture_lfm2_child_embedding_replay,
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_exact4x_first_divergence import (  # noqa: E402
    LFM2_EXACT4X_BUDGET_OBSERVATION_SCHEMA,
    LFM2_EXACT4X_FIRST_DIVERGENCE_IMPLEMENTATION_SCHEMA,
    Lfm2Exact4xFirstDivergenceError,
    aggregate_lfm2_exact4x_first_divergence,
    audit_lfm2_exact4x_first_divergence_batch,
    authoritative_split_by_batch,
    build_lfm2_exact4x_first_divergence_request,
    parent_selection_logical_bytes,
    parent_tensor_names_for_first_divergence,
    phase_names,
    verify_complete_exact_budget_child,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    Lfm2CanonicalFp32RuntimeRelocationReceipt,
    count_lfm2_direct_local_unique_parameter_elements,
)
from neural_forge.hf_lfm2_semantic_step_token_fold import (  # noqa: E402
    bind_lfm2_separate_sparse_semantic_code_runtime,
    seal_lfm2_semantic_step_span_evidence,
)
from neural_forge.lfm2_carrier_bridge_publication import (  # noqa: E402
    PACKAGED_ROUTE_SCHEMA,
    AuthenticatedPackagedLfm2CarrierRoutes,
    Lfm2PortableCarrierRuntimeReceipt,
)


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


def _budget_observation() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": LFM2_EXACT4X_BUDGET_OBSERVATION_SCHEMA,
        "one_complete_child_observed": True,
    }
    return {**body, "budget_observation_sha256": canonical_sha256(body)}


def _implementation_identity() -> dict[str, object]:
    file_names = (
        "balanced_basis_phase",
        "canonical_lfm2",
        "carrier_child_package",
        "carrier_relocation_package",
        "core",
        "direct_local_loader",
        "dynamic_blank_execution",
        "routing_runtime",
        "runner",
        "teacher_forced_readout",
        "trace_cache",
        "transformers_lfm2_runtime",
    )
    body = {
        "schema": LFM2_EXACT4X_FIRST_DIVERGENCE_IMPLEMENTATION_SCHEMA,
        "files": {
            name: {"path": f"{name}.py", "sha256": _digest(name)} for name in file_names
        },
        "packages": {
            "python": "test",
            "safetensors": "test",
            "torch": "test",
            "transformers": "test",
        },
        "numeric_contract": {
            "runtime_device": "cpu",
            "runtime_dtype": "float32",
        },
        "execution_contract": {
            "network_allowed": False,
            "parent_model_loaded": False,
            "parent_forward_calls": 0,
            "training_steps": 0,
        },
    }
    return {**body, "implementation_sha256": canonical_sha256(body)}


def _parent_logit_domain_contract(
    *,
    child_package_sha256: str = "",
    route_package_sha256: str = "",
    trace_cache_key: str = "",
    trace_selective_session_sha256: str = "",
    defined_count: int = 31,
    physical_count: int = 31,
) -> dict[str, object]:
    return diagnostic.build_lfm2_exact4x_parent_logit_domain_contract(
        child_package_sha256=(child_package_sha256 or _digest("child-package")),
        route_package_sha256=(route_package_sha256 or _digest("route-package")),
        semantic_source_tokenizer_sha256=_digest("semantic-source-tokenizer"),
        semantic_source_lexicon_package_sha256=_digest("semantic-source-lexicon"),
        packaged_bridge_reference_sha256=_digest("bridge-reference-file"),
        bridge_reference_schema="test-bridge-reference-v1",
        bridge_artifact_sha256=_digest("strict-bridge"),
        bridge_profile_sha256=_digest("strict-bridge-profile"),
        bridge_target_tokenizer_semantics_sha256=_digest("target-tokenizer"),
        bridge_target_tokenizer_package_fingerprint=_digest("target-package"),
        trace_cache_key=(trace_cache_key or _digest("trace-cache")),
        trace_selective_session_sha256=(
            trace_selective_session_sha256 or _digest("trace-session")
        ),
        trace_logit_descriptor_inventory_sha256=_digest("trace-logit-descriptors"),
        trace_batch_count=2,
        trace_phase_count=4,
        defined_semantic_row_count=defined_count,
        maximum_defined_token_id=defined_count - 1,
        route_donor_row_count=defined_count,
        parent_physical_row_capacity=physical_count,
    )


def _model(*, seed: int = 2401):
    config = transformers.Lfm2Config(
        vocab_size=31,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return transformers.Lfm2ForCausalLM(config)


def _source_model(*, seed: int = 2399, vocab_size: int = 31):
    config = transformers.Lfm2Config(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return transformers.Lfm2ForCausalLM(config)


def _compact_model(*, seed: int = 2404):
    config = transformers.Lfm2Config(
        vocab_size=31,
        hidden_size=2,
        intermediate_size=3,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return transformers.Lfm2ForCausalLM(config)


def _module_modes(model):
    return {name: bool(module.training) for name, module in model.named_modules()}


def _embedding_replay(model):
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    embeddings = model.get_input_embeddings()(token_ids).detach().clone()
    return embeddings, capture_lfm2_child_embedding_replay(
        model,
        embeddings,
        attention_mask=torch.ones_like(token_ids),
        position_ids=torch.arange(3, dtype=torch.long).unsqueeze(0),
        selected_layer_indices=(0, 1),
        sample_last_tokens=2,
        input_device="cpu",
    )


def test_embedding_replay_is_one_state_preserving_inputs_embeds_forward(
    monkeypatch,
) -> None:
    model = _model()
    model.train()
    model.model.layers[0].eval()
    modes_before = _module_modes(model)
    state_before = fingerprint_lfm2_parent(model)
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    embeddings = model.get_input_embeddings()(token_ids).detach().clone()
    calls = []

    def observe(_module, args, kwargs):
        calls.append(
            {
                "positional_count": len(args),
                "input_ids": kwargs.get("input_ids"),
                "inputs_embeds": kwargs.get("inputs_embeds"),
                "use_cache": kwargs.get("use_cache"),
                "grad_enabled": torch.is_grad_enabled(),
                "training": bool(model.training),
            }
        )

    model.register_forward_pre_hook(observe, with_kwargs=True)

    def forbidden_embedding_lookup(*_args, **_kwargs):
        raise AssertionError("embedding replay performed a token-table lookup")

    monkeypatch.setattr(
        model.get_input_embeddings(), "forward", forbidden_embedding_lookup
    )
    replay = capture_lfm2_child_embedding_replay(
        model,
        embeddings,
        attention_mask=torch.ones_like(token_ids),
        position_ids=torch.arange(3, dtype=torch.long).unsqueeze(0),
        selected_layer_indices=(0, 1),
        sample_last_tokens=2,
        input_device="cpu",
    )

    assert len(calls) == 1
    assert calls[0]["positional_count"] == 0
    assert calls[0]["input_ids"] is None
    assert calls[0]["inputs_embeds"] is not None
    assert calls[0]["use_cache"] is False
    assert calls[0]["grad_enabled"] is False
    assert calls[0]["training"] is False
    assert replay.selected_layer_indices == (0, 1)
    assert replay.record["operation_audit"] == {
        "backward_calls": 0,
        "child_forward_calls": 1,
        "input_ids_used": False,
        "inputs_embeds_used": True,
        "model_mutations": 0,
        "network_calls": 0,
        "optimizer_steps": 0,
        "parent_forward_calls": 0,
        "tokenizer_calls": 0,
        "training_steps": 0,
        "use_cache": False,
    }
    state_endpoints = {
        "trajectory.layer.000.operator.output",
        "trajectory.layer.000.residual.post_operator",
        "trajectory.layer.000.mlp.down",
        "trajectory.layer.000.residual.post_layer",
        "trajectory.layer.001.operator.output",
        "trajectory.layer.001.residual.post_operator",
        "trajectory.layer.001.mlp.down",
        "trajectory.layer.001.residual.post_layer",
        "trajectory.output.lm_head_input",
    }
    assert state_endpoints.issubset(replay.tensors)
    assert all(
        tuple(replay.tensors[name].shape[:2]) == (1, 2) for name in state_endpoints
    )
    assert fingerprint_lfm2_parent(model) == state_before
    assert _module_modes(model) == modes_before
    mask = torch.ones_like(token_ids)
    positions = torch.arange(3, dtype=torch.long).unsqueeze(0)
    assert (
        replay.authenticate_for_inputs(
            embeddings,
            attention_mask=mask,
            position_ids=positions,
        )["embedding_replay_sha256"]
        == replay.replay_sha256
    )
    with pytest.raises(Lfm2CircuitTraceCorruptionError, match="different donor"):
        replay.authenticate_for_inputs(
            embeddings + 1.0,
            attention_mask=mask,
            position_ids=positions,
        )
    with pytest.raises(Lfm2CircuitTraceCorruptionError, match="causal input geometry"):
        replay.authenticate_for_inputs(
            embeddings,
            attention_mask=mask,
            position_ids=positions + 1,
        )

    with torch.inference_mode():
        replay.tensors["trajectory.layer.000.operator.output"].add_(1.0)
    with pytest.raises(Lfm2CircuitTraceCorruptionError, match="tensor inventory"):
        replay.authenticate_for_inputs(
            embeddings,
            attention_mask=mask,
            position_ids=positions,
        )

    with pytest.raises(Lfm2CircuitTraceCaptureError, match="NaN or infinity"):
        capture_lfm2_child_embedding_replay(
            model,
            torch.full_like(embeddings, float("nan")),
            attention_mask=torch.ones_like(token_ids),
            selected_layer_indices=(0, 1),
            sample_last_tokens=2,
            input_device="cpu",
        )
    assert len(calls) == 1


def _reference(index: int, split: str) -> SimpleNamespace:
    del split
    input_sha = _digest(f"input-{index}")
    phases = phase_names(1)
    names = {
        "input.input_ids",
        "input.attention_mask",
        "input.forced_decode_ids",
    }
    for phase in phases:
        names.add(f"{phase}.logits")
        for layer_index in range(2):
            for suffix in (
                "operator.output",
                "residual.post_operator",
                "mlp.down",
                "residual.post_layer",
            ):
                names.add(f"{phase}.layer.{layer_index:03d}.{suffix}")
        names.add(f"{phase}.layer.000.residual.pre")
    inventory = {name: {"size_bytes": 8} for name in names}
    return SimpleNamespace(
        index=index,
        input_sha256=input_sha,
        record_sha256=_digest(f"record-{index}"),
        ordered_tensor_names=tuple(sorted(names)),
        record={"capture": {"decode_steps": 1}, "tensors": inventory},
    )


def test_parent_selection_and_basis_split_are_complete_and_authoritative() -> None:
    references = (_reference(0, "fit"), _reference(1, "heldout"))
    names = parent_tensor_names_for_first_divergence(
        references[0],
        layer_count=2,
        focus_layer_indices=(1,),
    )
    assert parent_selection_logical_bytes(references[0], names) == len(names) * 8
    assert "prefill.layer.000.residual.pre" in names
    assert "prefill.layer.001.residual.pre" not in names
    assert "prefill.layer.000.operator.output" not in names
    assert "prefill.layer.001.operator.output" in names
    assert "prefill.layer.000.residual.post_layer" in names
    report = {
        "cache": {
            "record_horizons": [
                {
                    "batch_index": 0,
                    "record_id": references[0].input_sha256,
                    "split": "fit",
                },
                {
                    "batch_index": 1,
                    "record_id": references[1].input_sha256,
                    "split": "heldout",
                },
            ]
        }
    }
    assert dict(authoritative_split_by_batch(report, references)) == {
        0: "fit",
        1: "heldout",
    }
    report["cache"]["record_horizons"][1]["record_id"] = _digest("foreign")
    with pytest.raises(Lfm2Exact4xFirstDivergenceError, match="differ"):
        authoritative_split_by_batch(report, references)
    with pytest.raises(Lfm2Exact4xFirstDivergenceError, match="only integers"):
        parent_tensor_names_for_first_divergence(
            references[0],
            layer_count=2,
            focus_layer_indices=(0, "1"),
        )


def test_exact_budget_rejects_caller_constructed_parameter_assertions() -> None:
    with pytest.raises(Lfm2Exact4xFirstDivergenceError, match="authenticated"):
        verify_complete_exact_budget_child(
            object(),
            basis_phase=SimpleNamespace(parent_parameter_elements=400),
            route_package=SimpleNamespace(parameter_elements=25),
            portable_runtime_receipt=SimpleNamespace(parameter_elements=100),
        )


class _CoordinateMap:
    source_width = 16
    target_width = 8

    def __init__(self, map_sha256: str) -> None:
        self.map_sha256 = map_sha256

    def encode(self, rows):
        return rows[..., : self.target_width]


def _route_allocations(
    input_elements: int,
    output_elements: int,
) -> tuple[ContinuousChildStateAllocation, ContinuousChildStateAllocation]:
    artifact_sha256 = _digest("route-source-artifact")
    return (
        ContinuousChildStateAllocation(
            state_id="semantic-carrier-route.c_in",
            parameter_elements=input_elements,
            tensor_sha256=_digest(f"c-in-{input_elements}"),
            artifact_sha256=artifact_sha256,
        ),
        ContinuousChildStateAllocation(
            state_id="semantic-carrier-route.c_out",
            parameter_elements=output_elements,
            tensor_sha256=_digest(f"c-out-{output_elements}"),
            artifact_sha256=artifact_sha256,
        ),
    )


def _route_package(
    allocations: tuple[
        ContinuousChildStateAllocation,
        ContinuousChildStateAllocation,
    ],
) -> AuthenticatedPackagedLfm2CarrierRoutes:
    body = {
        "schema": PACKAGED_ROUTE_SCHEMA,
        "status": "complete",
        "input_route_allocation": allocations[0].as_dict(),
        "output_route_allocation": allocations[1].as_dict(),
        "continuous_semantic_coefficient_count": sum(
            value.parameter_elements for value in allocations
        ),
    }
    receipt = {**body, "package_sha256": canonical_sha256(body)}
    return AuthenticatedPackagedLfm2CarrierRoutes(
        root=Path("."),
        receipt=receipt,
        tensors={},
        allocations=allocations,
    )


def _portable_receipt(
    route_package: AuthenticatedPackagedLfm2CarrierRoutes,
    *,
    runtime_sha256: str,
    dense_parameter_elements: int,
) -> Lfm2PortableCarrierRuntimeReceipt:
    direct = Lfm2CanonicalFp32RuntimeRelocationReceipt(
        provenance_resource_fingerprint=_digest("implementation"),
        canonical_device="cpu",
        runtime_device="cpu",
        runtime_dtype="float32",
        canonical_state_sha256=_digest("canonical-state"),
        expected_runtime_state_sha256=_digest("runtime-state"),
        runtime_state_sha256=_digest("runtime-state"),
        canonical_state_alias_abi_sha256=_digest("canonical-alias"),
        runtime_state_alias_abi_sha256=_digest("canonical-alias"),
        canonical_unique_parameter_elements=dense_parameter_elements,
        runtime_unique_parameter_elements=dense_parameter_elements,
        executable_forward_smoke={"logits_finite": True, "use_cache": False},
    )
    return Lfm2PortableCarrierRuntimeReceipt(
        route_package_sha256=route_package.package_sha256,
        source_bridge_artifact_sha256=_digest("route-source-artifact"),
        direct_local_relocation=direct,
        canonical_input_table_binding={},
        canonical_output_table_binding={},
        runtime_input_table_binding={},
        runtime_output_table_binding={},
        route_allocations=route_package.allocations,
        carrier_runtime_sha256=runtime_sha256,
    )


def _routing_context(model):
    donor_count = int(model.config.vocab_size)
    spans = [[index] for index in range(donor_count)]
    evidence = seal_lfm2_semantic_step_span_evidence(
        spans,
        target_row_count=donor_count,
    )
    offsets = torch.arange(donor_count + 1, dtype=torch.int64)
    ids = torch.arange(donor_count, dtype=torch.int64)
    coefficients = torch.ones(donor_count, dtype=torch.float32)
    routing = bind_lfm2_separate_sparse_semantic_code_runtime(
        evidence,
        model.get_input_embeddings().weight,
        live_child_output_table=model.get_output_embeddings().weight,
        input_semantic_code_offsets=offsets,
        input_semantic_code_target_ids=ids,
        input_semantic_code_coefficients=coefficients,
        output_semantic_code_offsets=offsets,
        output_semantic_code_target_ids=ids,
        output_semantic_code_coefficients=coefficients.clone(),
        target_to_donor=ids,
    )
    allocations = _route_allocations(donor_count, donor_count)
    route_package = _route_package(allocations)
    portable = _portable_receipt(
        route_package,
        runtime_sha256=routing.as_dict()["runtime_sha256"],
        dense_parameter_elements=count_lfm2_direct_local_unique_parameter_elements(
            model
        ),
    )
    return routing, route_package, portable


def _projectable(row):
    value = row.detach().to(device="cpu", dtype=torch.float32)
    return torch.cat((value, torch.zeros_like(value))).reshape(1, 1, 16)


def _parent_tensors_for_replay(replay, embeddings, routing, output_session):
    prompt_ids = torch.tensor([[1, 2]], dtype=torch.long)
    forced_ids = torch.tensor([[3]], dtype=torch.long)
    result = {
        "input.input_ids": prompt_ids,
        "input.attention_mask": torch.ones_like(prompt_ids),
        "input.position_ids": torch.arange(2, dtype=torch.long).unsqueeze(0),
        "input.forced_decode_ids": forced_ids,
    }
    for phase_index, phase in enumerate(phase_names(1)):
        state_position = phase_index + 1
        result[f"{phase}.layer.000.residual.pre"] = _projectable(
            embeddings[0, state_position]
        )
        for layer_index in range(2):
            for suffix in (
                "operator.output",
                "residual.post_operator",
                "mlp.down",
                "residual.post_layer",
            ):
                child_row = replay.tensors[
                    f"trajectory.layer.{layer_index:03d}.{suffix}"
                ][0, phase_index]
                result[f"{phase}.layer.{layer_index:03d}.{suffix}"] = _projectable(
                    child_row
                )
        child_hidden = replay.tensors["trajectory.output.lm_head_input"][0, phase_index]
        target_scores = routing.score_target_tokens(output_session, child_hidden)
        donor_scores = routing.compute_all_donor_logits_from_target_scores(
            output_session,
            target_scores,
            maximum_donor_rows_per_chunk=7,
        )
        result[f"{phase}.logits"] = donor_scores.reshape(1, 1, 31)
    return result


def _compact_payload(tensors):
    return {
        name: (value if name.startswith("input.") else value[:, -1:, :])
        .detach()
        .to(device="cpu")
        .contiguous()
        .clone()
        for name, value in tensors.items()
    }


def test_parent_logit_domain_excludes_undefined_physical_capacity() -> None:
    padded_contract = _parent_logit_domain_contract(
        defined_count=3,
        physical_count=4,
    )
    defined, evidence = diagnostic._defined_parent_logits_and_capacity_evidence(
        torch.tensor([1.0, 4.0, 2.0, 100.0]),
        padded_contract,
    )
    assert torch.equal(defined, torch.tensor([1.0, 4.0, 2.0], dtype=torch.float64))
    assert int(defined.argmax().item()) == 1
    assert evidence == diagnostic._authenticate_parent_logit_domain(evidence)
    assert evidence["physical_row_count"] == 4
    assert evidence["defined_semantic_row_count"] == 3
    assert evidence["undefined_capacity_row_count"] == 1
    assert evidence["physical_top1_row_id"] == 3
    assert evidence["physical_top1_is_defined_semantic_row"] is False
    assert evidence["defined_top1_row_id"] == 1
    assert evidence["undefined_capacity_rows_excluded_from_semantic_scoring"]
    assert (
        evidence["parent_logit_domain_contract_sha256"]
        == padded_contract["contract_sha256"]
    )

    exact_contract = _parent_logit_domain_contract(
        defined_count=3,
        physical_count=3,
    )
    exact, unpadded = diagnostic._defined_parent_logits_and_capacity_evidence(
        torch.tensor([1.0, 4.0, 2.0]),
        exact_contract,
    )
    assert torch.equal(exact, defined)
    assert unpadded["undefined_capacity_row_count"] == 0
    assert unpadded["maximum_undefined_capacity_score"] is None
    assert unpadded["undefined_minus_defined_top1_score"] is None
    assert unpadded["undefined_physical_softmax_mass"] == 0.0
    assert unpadded == diagnostic._authenticate_parent_logit_domain(unpadded)

    with pytest.raises(Lfm2Exact4xFirstDivergenceError, match="capacity"):
        diagnostic._defined_parent_logits_and_capacity_evidence(
            torch.tensor([1.0, 2.0]),
            padded_contract,
        )
    for invalid in (
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j]),
    ):
        with pytest.raises(Lfm2Exact4xFirstDivergenceError, match="finite real"):
            diagnostic._defined_parent_logits_and_capacity_evidence(
                invalid,
                padded_contract,
            )


def test_parent_logit_domain_authenticates_prefix_tie_breaks_and_winners() -> None:
    rank = diagnostic._deterministic_parent_winner_rank_margin(
        torch.tensor([5.0, 5.0, 4.0]),
        0,
    )
    assert rank["top1_token_id"] == 0
    assert rank["best_competing_token_id"] == 1
    assert rank["parent_winner_margin"] == 0.0

    contract = _parent_logit_domain_contract(
        defined_count=2,
        physical_count=4,
    )
    _defined, tied = diagnostic._defined_parent_logits_and_capacity_evidence(
        torch.tensor([5.0, 4.0, 5.0, 5.0]),
        contract,
    )
    assert tied["defined_top1_row_id"] == 0
    assert tied["maximum_undefined_capacity_row_id"] == 2
    assert tied["physical_top1_row_id"] == 0
    assert tied["physical_top1_is_defined_semantic_row"] is True
    diagnostic._authenticate_parent_logit_domain(tied, contract=contract)

    _defined, undefined_wins = diagnostic._defined_parent_logits_and_capacity_evidence(
        torch.tensor([5.0, 4.0, 6.0, 6.0]),
        contract,
    )
    assert undefined_wins["maximum_undefined_capacity_row_id"] == 2
    assert undefined_wins["physical_top1_row_id"] == 2
    assert undefined_wins["physical_top1_is_defined_semantic_row"] is False
    diagnostic._authenticate_parent_logit_domain(
        undefined_wins,
        contract=contract,
    )

    contradictory = dict(tied)
    contradictory["physical_top1_row_id"] = 2
    contradictory["physical_top1_is_defined_semantic_row"] = False
    with pytest.raises(Lfm2Exact4xFirstDivergenceError, match="winner evidence"):
        diagnostic._authenticate_parent_logit_domain(
            contradictory,
            contract=contract,
        )


def test_real_selected_cache_and_typed_routes_drive_complete_child_diagnostic(
    tmp_path,
) -> None:
    parent = _source_model(seed=2402, vocab_size=33)
    model = _model(seed=2403)
    map_sha = _digest("basis-map")
    coordinate = _CoordinateMap(map_sha)
    routing, route_package, portable = _routing_context(model)
    runtime_receipt = routing.as_dict()
    portable_receipt = portable.as_dict()
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 7]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "position_ids": torch.arange(3, dtype=torch.long).unsqueeze(0),
            "forced_decode_ids": torch.tensor([[3]], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[4, 5, 8]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "forced_decode_ids": torch.tensor([[6]], dtype=torch.long),
        },
    ]
    trace_cache = build_lfm2_circuit_trace_cache(
        tmp_path / "trace-cache",
        parent,
        batches,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=1,
            max_decode_steps=1,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    results = []
    with trace_cache.selective_authenticated_session() as trace_session:
        request = build_lfm2_exact4x_first_divergence_request(
            child_package_sha256=_digest("child-package"),
            child_state_sha256=fingerprint_lfm2_parent(model).identity_sha256,
            route_package_sha256=route_package.package_sha256,
            route_runtime_sha256=runtime_receipt["runtime_sha256"],
            portable_runtime_receipt_sha256=portable_receipt["receipt_sha256"],
            trace_cache_key=trace_cache.cache_key,
            trace_selective_session_sha256=trace_session.evidence_sha256,
            basis_phase_key_sha256=_digest("basis-phase"),
            basis_completion_sha256=_digest("basis-completion"),
            basis_map_sha256=map_sha,
            budget_observation=_budget_observation(),
            layer_count=2,
            focus_layer_indices=(1,),
            runtime_device="cpu",
            runtime_dtype="float32",
            maximum_selected_parent_bytes=1_000_000,
            implementation_identity=_implementation_identity(),
            parent_logit_domain_contract=_parent_logit_domain_contract(
                child_package_sha256=_digest("child-package"),
                route_package_sha256=route_package.package_sha256,
                trace_cache_key=trace_cache.cache_key,
                trace_selective_session_sha256=trace_session.evidence_sha256,
                defined_count=31,
                physical_count=33,
            ),
        )
        paired = routing.authenticated_live_table_sessions(
            model.get_input_embeddings().weight,
            model.get_output_embeddings().weight,
        )
        with paired as (input_session, output_session):
            for reference, split in zip(
                trace_session.batch_references,
                ("fit", "heldout"),
                strict=True,
            ):
                names = parent_tensor_names_for_first_divergence(
                    reference,
                    layer_count=2,
                    focus_layer_indices=(1,),
                )
                payload = trace_session.load_selected_tensors(reference.index, names)
                try:
                    parent_tensors = _compact_payload(payload.tensors)
                    ids = torch.cat(
                        (
                            parent_tensors["input.input_ids"][0],
                            parent_tensors["input.forced_decode_ids"][0],
                        )
                    )
                    embeddings = routing.reconstruct_selected_donor_embeddings(
                        input_session,
                        ids,
                    ).unsqueeze(0)
                    replay = capture_lfm2_child_embedding_replay(
                        model,
                        embeddings,
                        attention_mask=torch.ones((1, 4), dtype=torch.long),
                        position_ids=(
                            torch.arange(4, dtype=torch.long).unsqueeze(0)
                            if "input.position_ids" in parent_tensors
                            else None
                        ),
                        selected_layer_indices=(0, 1),
                        sample_last_tokens=2,
                        input_device="cpu",
                    )
                    result = audit_lfm2_exact4x_first_divergence_batch(
                        request=request,
                        batch_index=reference.index,
                        trace_input_sha256=reference.input_sha256,
                        trace_record_sha256=reference.record_sha256,
                        split=split,
                        parent_tensors=parent_tensors,
                        selected_parent_payload=payload,
                        trace_selective_session=trace_session,
                        reconstructed_trajectory_embeddings=embeddings,
                        child_replay=replay,
                        coordinate_map=coordinate,
                        route_package=route_package,
                        portable_runtime_receipt=portable,
                        routing=routing,
                        input_routing_session=input_session,
                        output_routing_session=output_session,
                        maximum_donor_rows_per_score_chunk=7,
                    )
                    assert result["numeric_decision"] is None
                    assert result["safety_observations"] == []
                    assert result["generation_cap_events"] == []
                    assert result["complete_child_observed_layer_indices"] == [0, 1]
                    assert result["focused_branch_layer_indices"] == [1]
                    assert result["selected_payload_sha256"] == payload.selection_sha256
                    assert result["selected_parent_authentication"][
                        "typed_selected_payload_authenticated"
                    ]
                    assert result["replay_input_authentication"][
                        "position_ids_present"
                    ] is (reference.index == 0)
                    assert all(
                        {row["layer_index"] for row in phase["branch_components"]}
                        == {1}
                        and {
                            row["layer_index"]
                            for row in phase["state_path"]
                            if row["component_id"] != "c_in"
                        }
                        == {0, 1}
                        for phase in result["phases"]
                    )
                    assert all(
                        phase["c_out"]["parent_logit_domain"]["physical_row_count"]
                        == 33
                        and phase["c_out"]["parent_logit_domain"][
                            "defined_semantic_row_count"
                        ]
                        == 31
                        and phase["c_out"]["parent_logit_domain"][
                            "undefined_capacity_row_count"
                        ]
                        == 2
                        and phase["c_out"]["parent_logit_domain"][
                            "undefined_capacity_rows_excluded_from_semantic_scoring"
                        ]
                        is True
                        for phase in result["phases"]
                    )
                    results.append(result)

                    tampered = dict(parent_tensors)
                    tampered["prefill.logits"] = (
                        tampered["prefill.logits"].clone().add_(1.0)
                    )
                    with pytest.raises(
                        Lfm2Exact4xFirstDivergenceError,
                        match="Retained parent tensor",
                    ):
                        audit_lfm2_exact4x_first_divergence_batch(
                            request=request,
                            batch_index=reference.index,
                            trace_input_sha256=reference.input_sha256,
                            trace_record_sha256=reference.record_sha256,
                            split=split,
                            parent_tensors=tampered,
                            selected_parent_payload=payload,
                            trace_selective_session=trace_session,
                            reconstructed_trajectory_embeddings=embeddings,
                            child_replay=replay,
                            coordinate_map=coordinate,
                            route_package=route_package,
                            portable_runtime_receipt=portable,
                            routing=routing,
                            input_routing_session=input_session,
                            output_routing_session=output_session,
                        )
                finally:
                    trace_session.release_selected_tensors(payload)

            reference = trace_session.batch_references[0]
            names = parent_tensor_names_for_first_divergence(
                reference,
                layer_count=2,
                focus_layer_indices=(1,),
            )
            payload = trace_session.load_selected_tensors(reference.index, names)
            try:
                parent_tensors = _compact_payload(payload.tensors)
                ids = torch.cat(
                    (
                        parent_tensors["input.input_ids"][0],
                        parent_tensors["input.forced_decode_ids"][0],
                    )
                )
                embeddings = routing.reconstruct_selected_donor_embeddings(
                    input_session, ids
                ).unsqueeze(0)
                replay = capture_lfm2_child_embedding_replay(
                    model,
                    embeddings,
                    attention_mask=torch.ones((1, 4), dtype=torch.long),
                    position_ids=torch.arange(4, dtype=torch.long).unsqueeze(0),
                    selected_layer_indices=(0, 1),
                    sample_last_tokens=2,
                    input_device="cpu",
                )
                with pytest.raises(
                    Lfm2Exact4xFirstDivergenceError, match="typed semantic"
                ):
                    audit_lfm2_exact4x_first_divergence_batch(
                        request=request,
                        batch_index=reference.index,
                        trace_input_sha256=reference.input_sha256,
                        trace_record_sha256=reference.record_sha256,
                        split="fit",
                        parent_tensors=parent_tensors,
                        selected_parent_payload=payload,
                        trace_selective_session=trace_session,
                        reconstructed_trajectory_embeddings=embeddings,
                        child_replay=replay,
                        coordinate_map=coordinate,
                        route_package=route_package,
                        portable_runtime_receipt=portable,
                        routing=object(),
                        input_routing_session=input_session,
                        output_routing_session=output_session,
                    )
            finally:
                trace_session.release_selected_tensors(payload)

            with pytest.raises(Lfm2Exact4xFirstDivergenceError, match="open typed"):
                diagnostic._authenticate_route_execution_context(
                    request=request,
                    route_package=route_package,
                    portable_runtime_receipt=portable,
                    routing=routing,
                    input_routing_session=object(),
                    output_routing_session=output_session,
                )

    ledger = aggregate_lfm2_exact4x_first_divergence(request, results)
    assert ledger["numeric_acceptance_decision"] is None
    assert ledger["manual_whole_response_intelligence_review_is_authoritative"]
    assert ledger["safety_observations_are_separate_not_failures"]
    assert ledger["generation_cap_saturation_is_censored_not_failure"]
    assert ledger["operation_audit"] == {
        "backward_calls": 0,
        "child_forward_calls": 2,
        "child_forward_call_scope": ("persisted-batch-evidence-across-producing-runs"),
        "current_run_child_forward_calls": None,
        "current_vs_historical_forward_counts_are_runner_scoped": True,
        "model_mutations": 0,
        "optimizer_steps": 0,
        "parent_forward_calls": 0,
        "training_steps": 0,
    }
    mismatch = ledger["first_discrete_c_out_parent_winner_mismatch"]
    assert mismatch is None or mismatch["batch_index"] in {0, 1}
    for split in ("fit", "heldout"):
        domain = ledger["split_summaries"][split]["parent_logit_domain"]
        assert domain["physical_row_counts"] == [33]
        assert domain["defined_semantic_row_counts"] == [31]
        assert domain["undefined_capacity_row_counts"] == [2]
        assert domain["undefined_capacity_rows_excluded_from_semantic_scoring"]


def test_balanced_basis_hit_only_loader_miss_creates_nothing(tmp_path) -> None:
    root = tmp_path / "missing-phase-root"
    with pytest.raises(Lfm2BalancedBasisPhaseCacheError, match="hit-only"):
        load_existing_lfm2_circuit_balanced_basis_phase(
            DynamicFoldPhaseCache(root),
            _digest("missing-phase"),
            anchor_authority=PhaseAnchorAuthority(
                key_id="first-divergence-test-key",
                secret=b"first-divergence-test-secret-32b!",
            ),
            expected_parent_identity_sha256=_digest("parent"),
            expected_trace_cache_key=_digest("trace"),
            expected_trace_selective_session_sha256=_digest("session"),
            expected_source_width=16,
            expected_target_width=8,
        )
    assert not root.exists()


def test_balanced_basis_hit_only_loader_authenticates_existing_map_without_trace_reads(
    tmp_path,
    monkeypatch,
) -> None:
    parent = _model(seed=2403)
    compact_child = _compact_model(seed=2404)
    graph = inspect_model_circuits(
        parent,
        model_name="first-divergence-basis-parent",
        tokenizer_fingerprint=_digest("parent-tokenizer"),
    )
    batches = []
    for index in range(4):
        input_ids = torch.tensor([[1, 2, 3, 4 + index]], dtype=torch.long)
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "forced_decode_ids": torch.tensor(
                    [[10 + index, 15 + index]],
                    dtype=torch.long,
                ),
            }
        )
    trace_cache = build_lfm2_circuit_trace_cache(
        tmp_path / "trace-cache",
        parent,
        batches,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=2,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    phase_root = tmp_path / "phases"
    phase_cache = DynamicFoldPhaseCache(phase_root)
    authority = PhaseAnchorAuthority(
        key_id="first-divergence-integration-key",
        secret=b"first-divergence-integration-secret!",
    )
    context = Lfm2BalancedBasisPhaseCacheContext(
        cache=phase_cache,
        anchor_authority=authority,
        dependency_provider=lambda: {"carrier.route": _digest("route")},
    )
    target = {
        "schema": "test-selected-target-v1",
        "source_graph_fingerprint": graph.fingerprint,
        "source_state_fingerprint": _digest("executor-state"),
        "hidden_size": 2,
        "exact_target_unique_parameters": (
            count_lfm2_direct_local_unique_parameter_elements(compact_child)
        ),
    }
    tokenizer = {
        "schema": "test-tokenizer-binding-v1",
        "parent_tokenizer_semantics_sha256": graph.tokenizer_fingerprint,
        "child_tokenizer_semantics_sha256": graph.tokenizer_fingerprint,
        "bridge_required": False,
    }
    session = trace_cache.selective_authenticated_session()
    try:
        built = load_or_build_lfm2_circuit_balanced_basis(
            context,
            parent,
            trace_cache,
            2,
            circuit_graph=graph,
            selective_session=session,
            build_request=Lfm2BalancedBasisPhaseBuildRequest(
                adapter_source_sha256=_digest("basis-adapter"),
                validation_fraction=0.25,
                split_seed="first-divergence-basis-test-v1",
                max_rows_per_block=8,
            ),
            binding_provider=lambda: {
                "selected_target": target,
                "tokenizer": tokenizer,
            },
            live_vocabulary_candidate_gate_required=False,
        )
        entry = next((phase_root / LFM2_BALANCED_BASIS_PHASE).iterdir())

        def forbidden(*_args, **_kwargs):
            raise AssertionError("hit-only basis loader rebuilt or read a trace tensor")

        monkeypatch.setattr(DynamicFoldPhaseCache, "begin", forbidden)
        monkeypatch.setattr(
            Lfm2CircuitTraceSelectiveAuthenticatedSession,
            "load_selected_tensors",
            forbidden,
        )
        loaded = load_existing_lfm2_circuit_balanced_basis_phase(
            phase_cache,
            entry.name,
            anchor_authority=authority,
            expected_parent_identity_sha256=trace_cache.parent_fingerprint,
            expected_trace_cache_key=trace_cache.cache_key,
            expected_trace_selective_session_sha256=session.evidence_sha256,
            expected_source_width=8,
            expected_target_width=2,
            expected_map_sha256=built.basis_result.map_sha256,
        )
        assert loaded.map_sha256 == built.basis_result.map_sha256
        assert loaded.source_width == 8
        assert loaded.target_width == 2
        assert loaded.as_dict()["cache_miss_build_allowed"] is False
        assert loaded.as_dict()["trace_tensor_loads"] == 0
        split = dict(
            authoritative_split_by_batch(
                loaded.report,
                session.batch_references,
            )
        )
        assert set(split) == {0, 1, 2, 3}
        assert tuple(split.values()).count("fit") == 3
        assert tuple(split.values()).count("heldout") == 1
        allocations = _route_allocations(1, 1)
        route_package = _route_package(allocations)
        portable = _portable_receipt(
            route_package,
            runtime_sha256=_digest("budget-only-runtime"),
            dense_parameter_elements=(
                count_lfm2_direct_local_unique_parameter_elements(compact_child)
            ),
        )
        budget = verify_complete_exact_budget_child(
            compact_child,
            basis_phase=loaded,
            route_package=route_package,
            portable_runtime_receipt=portable,
        )
        assert budget["parent_parameter_elements"] == sum(
            parameter.numel() for parameter in parent.parameters()
        )
        assert budget["physical_dense_parameter_elements"] == (
            count_lfm2_direct_local_unique_parameter_elements(compact_child)
        )
        assert budget["continuous_route_parameter_elements"] == 2
        assert budget["at_least_fourfold_parameter_compaction"] is True
        assert (
            budget["parameter_count_authority"]["caller_parameter_count_assertion_used"]
            is False
        )
    finally:
        session.close()
