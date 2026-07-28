from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import (
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from neural_forge.dynamic_model_inspector import default_inspector_registry
from neural_forge.dynamic_target_synthesis import (
    serialized_inference_state_inventory,
)
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_direct_local import (
    Lfm2DirectLocalCheckpointIO,
    load_lfm2_direct_local_tokenizer,
)
import neural_forge.hf_lfm2_pre_causal_body_resume_capsule as capsule
from neural_forge.hf_lfm2_interface_frontier import (
    fingerprint_lfm2_causal_body_inventory,
)
from neural_forge.hf_tokenizer_bridge import (
    seal_strict_tokenizer_bridge_artifact,
    strict_tokenizer_bridge_target_package_fingerprint,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")


def _tokenizer_package(path: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    path.mkdir(parents=True)
    vocabulary = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "hello": 4,
        "world": 5,
        "small": 6,
        "model": 7,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    return load_lfm2_direct_local_tokenizer(path)


def _untied_model(*, value_equal: bool = True):
    config = transformers.Lfm2Config(
        vocab_size=8,
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
    torch.manual_seed(24601)
    model = transformers.Lfm2ForCausalLM(config)
    if value_equal:
        with torch.no_grad():
            model.get_output_embeddings().weight.copy_(
                model.get_input_embeddings().weight
            )
    return model.eval()


def _fixture_digest(label: str) -> str:
    return canonical_sha256({"fixture_lineage": label})


def _problem(
    tmp_path: Path,
    *,
    value_equal: bool = True,
    freeze_before_publication: bool = True,
):
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer = _tokenizer_package(tokenizer_root)
    bridge = seal_strict_tokenizer_bridge_artifact(
        tokenizer,
        tokenizer,
        tokenizer_root,
        donor_row_capacity=8,
        target_row_capacity=8,
    )
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        bridge,
        device="cpu",
        dtype=torch.float32,
    )
    model = _untied_model(value_equal=value_equal)
    child_name = "tiny-post-moving-gauge-body"
    graph = default_inspector_registry().inspect(
        model,
        model_name=child_name,
        tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
    )
    if freeze_before_publication:
        model.requires_grad_(False)
        model.eval()
    inventory = serialized_inference_state_inventory(model)
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    bindings = capsule.Lfm2PreCausalBodyResumeBindings(
        child_name=child_name,
        target_unique_parameters=graph.unique_parameter_count,
        target_logical_parameters=graph.logical_parameter_count,
        parent_identity_sha256=_fixture_digest("parent-identity"),
        target_synthesis_fingerprint=_fixture_digest("target-synthesis"),
        target_topology_fingerprint=_fixture_digest("target-topology"),
        target_configuration_sha256=_fixture_digest("target-configuration"),
        target_state_inventory_sha256=canonical_sha256(inventory),
        native_target_contract_sha256=_fixture_digest("native-target-contract"),
        child_graph_fingerprint=graph.fingerprint,
        child_structural_topology_fingerprint=(
            graph.structural_topology_fingerprint
        ),
        bridge_artifact_sha256=bridge.artifact_sha256,
        bridge_profile_sha256=bridge.profile_fingerprint,
        target_tokenizer_semantics_sha256=(
            bridge.target_tokenizer_semantics_sha256
        ),
        target_tokenizer_package_fingerprint=(
            strict_tokenizer_bridge_target_package_fingerprint(bridge)
        ),
        body_fold_realization_sha256=_fixture_digest("body-fold-realization"),
        global_residual_closure_selection_sha256=_fixture_digest(
            "global-residual-closure-selection"
        ),
        structural_interface_install_sha256=_fixture_digest(
            "structural-interface-install"
        ),
        semantic_layer_stack_sha256=_fixture_digest("semantic-layer-stack"),
        structural_attention_candidate_archive_manifest_sha256=(
            _fixture_digest("structural-attention-archive")
        ),
        postfold_reclosure_seed_archive_manifest_sha256=_fixture_digest(
            "postfold-reclosure-seed-archive"
        ),
        postfold_reclosure_seed_publication_sha256=_fixture_digest(
            "postfold-reclosure-seed-publication"
        ),
        deferred_final_w2_install_sha256=_fixture_digest("deferred-final-w2"),
        moving_gauge_post_w2_finalization_sha256=_fixture_digest(
            "moving-gauge-post-w2-finalization"
        ),
        body_model_state_sha256=fingerprint_model_state(model),
        body_runtime_state_sha256=fingerprint_model_runtime_state(model),
        body_lfm2_identity_sha256=fingerprint_lfm2_parent(model).identity_sha256,
        causal_body_inventory_sha256=(
            fingerprint_lfm2_causal_body_inventory(model)
        ),
        installed_input_table_sha256=capsule._interface_tensor_sha256(
            input_weight
        ),
        installed_output_baseline_sha256=capsule._interface_tensor_sha256(
            output_weight
        ),
    )
    return model, checkpoint_io, bindings


def test_pre_causal_body_capsule_atomically_publishes_and_fresh_loads(
    tmp_path: Path,
) -> None:
    model, checkpoint_io, bindings = _problem(tmp_path)
    output = tmp_path / "body-capsule"
    incoming_runtime = fingerprint_model_runtime_state(model)
    frozen_graph = default_inspector_registry().inspect(
        model,
        model_name=bindings.child_name,
        tokenizer_fingerprint=bindings.target_tokenizer_semantics_sha256,
    )

    assert incoming_runtime == bindings.body_runtime_state_sha256
    assert frozen_graph.fingerprint != bindings.child_graph_fingerprint
    assert frozen_graph.structural_topology_fingerprint == (
        bindings.child_structural_topology_fingerprint
    )

    reference = capsule.publish_lfm2_pre_causal_body_resume_capsule(
        model=model,
        checkpoint_io=checkpoint_io,
        bindings=bindings,
        output_directory=output,
    )
    loaded = capsule.load_lfm2_pre_causal_body_resume_capsule(
        reference,
        checkpoint_io=checkpoint_io,
    )

    assert reference.path == output
    assert {path.name for path in output.iterdir()} == {
        "checkpoint",
        "capsule-manifest.json",
        "capsule-receipt.json",
    }
    assert loaded.model is not model
    assert fingerprint_model_state(loaded.model) == bindings.body_model_state_sha256
    assert loaded.model.get_input_embeddings().weight is not (
        loaded.model.get_output_embeddings().weight
    )
    assert torch.equal(
        loaded.model.get_input_embeddings().weight,
        loaded.model.get_output_embeddings().weight,
    )
    assert loaded.manifest["status"] == (
        "quarantined-pre-causal-body-resume-only"
    )
    assert loaded.manifest["accepted_artifact"] is False
    assert loaded.manifest["causal_output_head_installed"] is False
    assert loaded.manifest["parent_sidecar_parameters"] == 0
    assert loaded.manifest["runtime_shared_parent_parameters"] == 0
    assert loaded.manifest["parent_tensor_payloads"] == 0
    assert loaded.manifest["operation_audit"]["training_steps"] == 0
    assert loaded.manifest["operation_audit"]["parent_model_forward_calls"] == 0
    assert loaded.manifest["body_runtime_state_sha256"] == incoming_runtime
    assert loaded.manifest["runtime_boundary_audit"]["boundary_action"] == (
        "authenticated-inference-policy-noop"
    )
    assert loaded.manifest["runtime_boundary_audit"]["state_preserved"] is True
    assert loaded.manifest["compiled_child_graph_fingerprint"] == (
        bindings.child_graph_fingerprint
    )
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
    assert loaded.receipt["publication_fresh_reload_materializations"] == 1


def test_pre_causal_body_capsule_authenticates_trainable_to_frozen_transition(
    tmp_path: Path,
) -> None:
    model, checkpoint_io, bindings = _problem(
        tmp_path,
        freeze_before_publication=False,
    )
    state_before = fingerprint_model_state(model)
    runtime_before = fingerprint_model_runtime_state(model)
    storage_before = capsule._model_binding_signature(model)
    assert any(parameter.requires_grad for parameter in model.parameters())

    reference = capsule.publish_lfm2_pre_causal_body_resume_capsule(
        model=model,
        checkpoint_io=checkpoint_io,
        bindings=bindings,
        output_directory=tmp_path / "body-capsule",
    )
    loaded = capsule.load_lfm2_pre_causal_body_resume_capsule(
        reference,
        checkpoint_io=checkpoint_io,
    )

    assert bindings.body_runtime_state_sha256 == runtime_before
    assert fingerprint_model_state(model) == state_before
    assert capsule._model_binding_signature(model) == storage_before
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert loaded.manifest["runtime_boundary_audit"]["boundary_action"] == (
        "authenticated-inference-policy-transition"
    )
    assert loaded.manifest["runtime_boundary_audit"][
        "object_and_storage_bindings_preserved"
    ] is True
    assert loaded.manifest["runtime_boundary_audit"][
        "structural_topology_preserved"
    ] is True


def test_pre_causal_body_capsule_rejects_unset_lineage_digest(
    tmp_path: Path,
) -> None:
    _model, _checkpoint_io, bindings = _problem(tmp_path)

    with pytest.raises(
        capsule.Lfm2PreCausalBodyResumeCapsuleError,
        match="unset all-zero digest",
    ):
        replace(bindings, parent_identity_sha256="0" * 64)


def test_pre_causal_body_capsule_rejects_byte_tamper(
    tmp_path: Path,
) -> None:
    model, checkpoint_io, bindings = _problem(tmp_path)
    reference = capsule.publish_lfm2_pre_causal_body_resume_capsule(
        model=model,
        checkpoint_io=checkpoint_io,
        bindings=bindings,
        output_directory=tmp_path / "body-capsule",
    )
    tensor_path = reference.path / "checkpoint" / "model.safetensors"
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)

    with pytest.raises(
        capsule.Lfm2PreCausalBodyResumeCapsuleError,
        match="checkpoint bytes differ",
    ):
        capsule.load_lfm2_pre_causal_body_resume_capsule(
            reference,
            checkpoint_io=checkpoint_io,
        )


def test_pre_causal_body_capsule_is_no_replace(
    tmp_path: Path,
) -> None:
    model, checkpoint_io, bindings = _problem(tmp_path)
    output = tmp_path / "body-capsule"
    reference = capsule.publish_lfm2_pre_causal_body_resume_capsule(
        model=model,
        checkpoint_io=checkpoint_io,
        bindings=bindings,
        output_directory=output,
    )
    receipt_before = (output / "capsule-receipt.json").read_bytes()

    with pytest.raises(
        capsule.Lfm2PreCausalBodyResumeCapsuleError,
        match="Refusing to replace",
    ):
        capsule.publish_lfm2_pre_causal_body_resume_capsule(
            model=model,
            checkpoint_io=checkpoint_io,
            bindings=bindings,
            output_directory=output,
        )

    assert reference.path == output
    assert (output / "capsule-receipt.json").read_bytes() == receipt_before


def test_pre_causal_body_capsule_rejects_interface_and_topology_drift(
    tmp_path: Path,
) -> None:
    unequal, checkpoint_io, unequal_bindings = _problem(
        tmp_path / "unequal",
        value_equal=False,
    )
    with pytest.raises(
        capsule.Lfm2PreCausalBodyResumeCapsuleError,
        match="value-equal untied token tables",
    ):
        capsule.publish_lfm2_pre_causal_body_resume_capsule(
            model=unequal,
            checkpoint_io=checkpoint_io,
            bindings=unequal_bindings,
            output_directory=tmp_path / "unequal-capsule",
        )

    model, checkpoint_io, bindings = _problem(tmp_path / "topology")
    stale = replace(
        bindings,
        child_structural_topology_fingerprint="f" * 64,
    )
    with pytest.raises(
        capsule.Lfm2PreCausalBodyResumeCapsuleError,
        match="target/body bindings",
    ):
        capsule.publish_lfm2_pre_causal_body_resume_capsule(
            model=model,
            checkpoint_io=checkpoint_io,
            bindings=stale,
            output_directory=tmp_path / "stale-capsule",
        )


def test_pre_causal_body_capsule_requires_publication_checkpoint_resource(
    tmp_path: Path,
) -> None:
    model, checkpoint_io, bindings = _problem(tmp_path)
    reference = capsule.publish_lfm2_pre_causal_body_resume_capsule(
        model=model,
        checkpoint_io=checkpoint_io,
        bindings=bindings,
        output_directory=tmp_path / "body-capsule",
    )
    wrong_resource = Lfm2DirectLocalCheckpointIO(
        checkpoint_io.tokenizer_bridge,
        device="cpu",
        dtype=torch.bfloat16,
    )

    with pytest.raises(
        capsule.Lfm2PreCausalBodyResumeCapsuleError,
        match="publication resource",
    ):
        capsule.load_lfm2_pre_causal_body_resume_capsule(
            reference,
            checkpoint_io=wrong_resource,
        )


def test_pre_causal_body_reference_exactly_parses_and_authenticates_without_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, checkpoint_io, bindings = _problem(tmp_path)
    reference = capsule.publish_lfm2_pre_causal_body_resume_capsule(
        model=model,
        checkpoint_io=checkpoint_io,
        bindings=bindings,
        output_directory=tmp_path / "body-capsule",
    )
    parsed = capsule.Lfm2PreCausalBodyResumeReference.from_mapping(
        reference.as_dict()
    )
    assert parsed == reference

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("authenticate-only path materialized a model")

    monkeypatch.setattr(Lfm2DirectLocalCheckpointIO, "load", forbidden_load)
    authenticated = capsule.authenticate_lfm2_pre_causal_body_resume_capsule(
        parsed,
        checkpoint_io=checkpoint_io,
    )
    assert authenticated.bindings == bindings
    assert authenticated.reference == reference
    assert authenticated.receipt["publication_fresh_reload_materializations"] == 1

    unknown = {**reference.as_dict(), "unknown": True}
    with pytest.raises(
        capsule.Lfm2PreCausalBodyResumeCapsuleError,
        match="missing or unknown",
    ):
        capsule.Lfm2PreCausalBodyResumeReference.from_mapping(unknown)


def test_pre_causal_body_capsule_rejects_nonportable_runtime_hooks(
    tmp_path: Path,
) -> None:
    model, checkpoint_io, bindings = _problem(tmp_path)
    handle = model.model.layers[0].register_forward_hook(
        lambda _module, _args, output: output
    )
    try:
        with pytest.raises(
            capsule.Lfm2PreCausalBodyResumeCapsuleError,
            match="hook registries",
        ):
            capsule.publish_lfm2_pre_causal_body_resume_capsule(
                model=model,
                checkpoint_io=checkpoint_io,
                bindings=bindings,
                output_directory=tmp_path / "body-capsule",
            )
    finally:
        handle.remove()
