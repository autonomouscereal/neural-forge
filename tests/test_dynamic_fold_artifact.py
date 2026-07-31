from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace

import pytest

import neural_forge.dynamic_fold_artifact as artifact
from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.dynamic_behavior_gate import (
    BehaviorInferenceExecutionContract,
)
from neural_forge.dynamic_fold_artifact import (
    DynamicFoldArtifactError,
    LEGACY_SCHEMA,
    PREVIOUS_SCHEMA,
    _artifact_tree,
    _declared_lfm2_file,
    _structural_topology_differences,
    load_dynamic_fold_artifact_certificate,
    publish_dynamic_fold_artifact,
    scan_standalone_fold_artifact_tree,
)
from neural_forge.dynamic_fold_compiler import compile_loaded_models
from neural_forge.dynamic_fold_executor import (
    ExactCopyOperator,
    FoldDecision,
    FoldOperatorRegistry,
    StaticEvidenceProvider,
)
from neural_forge.dynamic_model_inspector import default_inspector_registry


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _model(seed: int):
    torch.manual_seed(seed)
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=True,
        )
    )


class _Accept:
    def evaluate(self, _context):
        return FoldDecision.accept("accepted")


def _compiled(*, include_parent: bool = False):
    parent = _model(1)
    child = _model(2)
    evidence = canonical_sha256({"cache": "once"})
    tokenizer = canonical_sha256({"tokenizer": "same"})
    compilation = compile_loaded_models(
        parent_models={"parent": parent},
        child_model=child,
        child_name="child",
        operators=FoldOperatorRegistry((ExactCopyOperator(),)),
        evidence_provider=StaticEvidenceProvider(evidence),
        evaluator=_Accept(),
        final_evaluator=_Accept(),
        evidence_fingerprint=evidence,
        parent_tokenizer_fingerprints={"parent": tokenizer},
        child_tokenizer_fingerprint=tokenizer,
    )
    if include_parent:
        return child, compilation, tokenizer, parent
    return child, compilation, tokenizer


def _saver(model, destination):
    torch.save(model.state_dict(), destination / "model.pt")
    (destination / "config.json").write_text(
        json.dumps(model.config.to_dict()),
        encoding="utf-8",
    )


def _loader(source):
    config = transformers.LlamaConfig(
        **json.loads((source / "config.json").read_text(encoding="utf-8"))
    )
    model = transformers.LlamaForCausalLM(config)
    model.load_state_dict(
        torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    model.requires_grad_(False)
    return model


class _DriftingInspectors:
    def __init__(self, mutate):
        self._base = default_inspector_registry()
        self._mutate = mutate

    def inspect(self, model, **kwargs):
        return self._mutate(self._base.inspect(model, **kwargs))


def _drift_node_path(graph):
    first = replace(graph.nodes[0], path=f"{graph.nodes[0].path}.drifted")
    return replace(graph, nodes=(first, *graph.nodes[1:]))


def _drift_config(graph):
    return replace(
        graph,
        config_fingerprint=canonical_sha256({"drift": "configuration"}),
    )


def _drift_parameter_shape(graph):
    for node_index, node in enumerate(graph.nodes):
        for parameter_index, parameter in enumerate(node.parameters):
            if parameter.numel <= 1:
                continue
            shape = (parameter.numel, 1)
            if shape == parameter.shape:
                shape = (1, parameter.numel)
            parameters = list(node.parameters)
            parameters[parameter_index] = replace(parameter, shape=shape)
            nodes = list(graph.nodes)
            nodes[node_index] = replace(node, parameters=tuple(parameters))
            return replace(graph, nodes=tuple(nodes))
    raise AssertionError("Test graph has no reshapeable parameter.")


def _fresh_accept(model):
    with torch.no_grad():
        model(
            input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            use_cache=False,
        )
    return FoldDecision.accept("fresh reload accepted")


def test_artifact_tree_rejects_resume_cache_sidecar_filename(tmp_path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    (child / "resume-cache.bin").write_bytes(b"not-a-child-weight")

    with pytest.raises(
        DynamicFoldArtifactError,
        match="Forbidden sidecar-like artifact name",
    ):
        _artifact_tree(child)


def test_public_standalone_tree_scanner_hashes_the_complete_clean_tree(
    tmp_path,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    (child / "config.json").write_text("{}", encoding="utf-8")
    (child / "model.safetensors").write_bytes(b"test")

    tree = scan_standalone_fold_artifact_tree(
        child,
        expected_file_roles={
            "config.json": "model_config",
            "model.safetensors": "model_tensor",
        },
    )

    assert [row["path"] for row in tree] == ["config.json", "model.safetensors"]
    assert canonical_sha256(tree) == canonical_sha256([dict(row) for row in tree])


@pytest.mark.parametrize("extra_name", ("auxiliary.dat", "payload.pkl"))
def test_public_standalone_tree_scanner_rejects_neutrally_named_extra_files(
    tmp_path,
    extra_name: str,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    (child / "config.json").write_text("{}", encoding="utf-8")
    (child / "model.safetensors").write_bytes(b"test")
    (child / extra_name).write_bytes(b"unlisted runtime material")

    with pytest.raises(
        DynamicFoldArtifactError,
        match="canonical typed inventory",
    ):
        scan_standalone_fold_artifact_tree(
            child,
            expected_file_roles={
                "config.json": "model_config",
                "model.safetensors": "model_tensor",
            },
        )


def test_lfm2_declared_inventory_rejects_an_unhashable_path_cleanly(tmp_path) -> None:
    with pytest.raises(
        DynamicFoldArtifactError,
        match="declared-file path is malformed",
    ):
        _declared_lfm2_file(
            tmp_path,
            {"path": [], "size_bytes": 0, "sha256": "0" * 64},
            size_field="size_bytes",
            role="model_tensor",
        )


def test_strict_standalone_scanner_rejects_oversized_structured_json(
    tmp_path,
    monkeypatch,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    (child / "config.json").write_text("{}", encoding="utf-8")
    (child / "model.safetensors").write_bytes(b"test")
    monkeypatch.setattr(artifact, "_MAX_STRUCTURED_INVENTORY_BYTES", 1)

    with pytest.raises(DynamicFoldArtifactError, match="JSON exceeds the scan limit"):
        scan_standalone_fold_artifact_tree(
            child,
            expected_file_roles={
                "config.json": "model_config",
                "model.safetensors": "model_tensor",
            },
        )


def test_strict_standalone_scanner_rejects_hardlinked_files(tmp_path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"test")
    try:
        os.link(source, child / "model.safetensors")
    except OSError as exc:  # pragma: no cover - filesystem capability
        pytest.skip(f"hard links unavailable: {exc}")
    (child / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DynamicFoldArtifactError, match="cannot be a hard link"):
        scan_standalone_fold_artifact_tree(
            child,
            expected_file_roles={
                "config.json": "model_config",
                "model.safetensors": "model_tensor",
            },
        )


@pytest.mark.parametrize(
    "writer",
    (
        lambda root: (root / "config.json").write_text(
            json.dumps(
                {
                    "resume_coordinator_reference": {
                        "checkpoint_reference": "checkpoint/store.json",
                    }
                }
            ),
            encoding="utf-8",
        ),
        lambda root: torch.save(
            {"resume_coordinator.payload": torch.zeros(1)},
            root / "model.pt",
        ),
    ),
)
def test_artifact_tree_rejects_serialized_resume_resource_inventory(
    tmp_path,
    writer,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    writer(child)

    with pytest.raises(
        DynamicFoldArtifactError,
        match="Forbidden resume/cache/coordinator/checkpoint/pending/admission",
    ):
        _artifact_tree(child)


def _wrapper_behavior_manifest(
    contract: BehaviorInferenceExecutionContract,
) -> tuple[str, int, str, str, dict]:
    gate_id = "test.wrapper.behavior"
    gate_version = 1
    baseline = canonical_sha256({"cached": "wrapper-output"})
    body = {
        "schema": "test-wrapper-behavior-gate-v1",
        "gate_id": gate_id,
        "gate_version": gate_version,
        "baseline_fingerprint": baseline,
        "inference_execution": contract.as_dict(),
    }
    fingerprint = canonical_sha256(body)
    return (
        gate_id,
        gate_version,
        baseline,
        fingerprint,
        {**body, "manifest_fingerprint": fingerprint},
    )


def _loader_with_chat(source):
    model = _loader(source)

    def chat(input_ids):
        return model.model(input_ids=input_ids, use_cache=False)

    object.__setattr__(model, "chat", chat)
    return model


def _wrapper_fresh_accept(model):
    with torch.no_grad():
        model.chat(torch.tensor([[1, 2]], dtype=torch.long))
    return FoldDecision.accept("wrapper inference accepted")


def test_publication_is_atomic_exact_and_freshly_reloaded(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()

    destination = tmp_path / "published"
    certificate = publish_dynamic_fold_artifact(
        compilation=compilation,
        child_model=child,
        destination=destination,
        model_saver=_saver,
        model_loader=_loader,
        fresh_reload_evaluator=_fresh_accept,
        saver_id="test-local-torch-save-v1",
        loader_id="test-local-torch-load-v1",
        tokenizer_fingerprint=tokenizer,
    )

    assert destination.is_dir()
    assert (destination / "child" / "model.pt").is_file()
    assert (destination / "certificate.json").is_file()
    assert {value.name for value in (destination / "provenance").iterdir()} == {
        "child_graph.json",
        "parent_graphs.json",
        "plan.json",
        "run.json",
        "target.json",
    }
    assert certificate.exact_unique_parameters == (
        compilation.plan.exact_target_unique_parameters
    )
    assert certificate.fresh_reload_decision.accepted is True
    assert certificate.artifact_file_count == 2
    assert certificate.artifact_total_bytes > 0
    assert certificate.compilation_optimizer_steps == 0
    assert certificate.compilation_backward_calls == 0
    assert certificate.loader_path_challenge_performed is True
    assert certificate.loader_path_challenge_exception_type.endswith(
        "FileNotFoundError"
    )
    assert certificate.provenance_plan_sha256 != "0" * 64
    assert certificate.provenance_target_sha256 != "0" * 64
    assert certificate.provenance_target_fingerprint != "0" * 64
    assert certificate.dynamic_target_provenance_present is False
    assert certificate.saver_implementation_fingerprint != "0" * 64
    assert certificate.loader_implementation_fingerprint != "0" * 64
    assert certificate.fresh_reload_module_forward_calls > 0
    assert certificate.fresh_reload_root_forward_calls > 0
    assert certificate.fresh_reload_inference_entrypoint_calls > 0
    assert certificate.body()["fresh_reload_source"] == ("retained-final-destination")
    assert certificate.body()["final_destination_reload_performed"] is True
    assert certificate.preparation_parent_module_forward_calls == 0
    assert certificate.publication_parent_module_forward_calls == 0
    assert certificate.fresh_reload_state_inventory is not None
    assert (
        certificate.fresh_reload_state_inventory["runtime_state_bytes"]
        >= certificate.fresh_reload_state_inventory["serialized_state_bytes"]
    )
    assert load_dynamic_fold_artifact_certificate(destination) == certificate


def test_publication_accepts_only_reload_trainability_policy_difference(
    tmp_path,
) -> None:
    child, compilation, tokenizer = _compiled()
    assert any(parameter.requires_grad for parameter in child.parameters())

    destination = tmp_path / "published"
    certificate = publish_dynamic_fold_artifact(
        compilation=compilation,
        child_model=child,
        destination=destination,
        model_saver=_saver,
        model_loader=_loader,
        fresh_reload_evaluator=_fresh_accept,
        saver_id="test-saver",
        loader_id="frozen-inference-loader",
        tokenizer_fingerprint=tokenizer,
    )

    reloaded = _loader(destination / "child")
    assert all(not parameter.requires_grad for parameter in reloaded.parameters())
    reload_graph = default_inspector_registry().inspect(
        reloaded,
        model_name=compilation.child_graph.model_name,
        tokenizer_fingerprint=tokenizer,
    )
    assert reload_graph.fingerprint != compilation.child_graph.fingerprint
    assert reload_graph.structural_topology_fingerprint == (
        compilation.child_graph.structural_topology_fingerprint
    )
    assert certificate.child_graph_fingerprint == compilation.child_graph.fingerprint
    assert certificate.fresh_reload_structural_topology_fingerprint == (
        compilation.child_graph.structural_topology_fingerprint
    )
    assert certificate.fresh_reload_parameters_frozen is True
    child_provenance = json.loads(
        (destination / "provenance" / "child_graph.json").read_text(encoding="ascii")
    )
    assert child_provenance["fingerprint"] == certificate.child_graph_fingerprint


def test_publication_rejects_trainable_fresh_reload_parameters(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"

    def trainable_loader(source):
        model = _loader(source)
        model.requires_grad_(True)
        return model

    with pytest.raises(
        DynamicFoldArtifactError,
        match="inference parameters are not all frozen",
    ):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=trainable_loader,
            fresh_reload_evaluator=_fresh_accept,
            saver_id="test-saver",
            loader_id="trainable-loader",
            tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()


def test_structural_topology_certificate_is_bound_to_compiled_provenance(
    tmp_path,
) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"
    publish_dynamic_fold_artifact(
        compilation=compilation,
        child_model=child,
        destination=destination,
        model_saver=_saver,
        model_loader=_loader,
        fresh_reload_evaluator=_fresh_accept,
        saver_id="test-saver",
        loader_id="frozen-inference-loader",
        tokenizer_fingerprint=tokenizer,
    )
    certificate_path = destination / "certificate.json"
    payload = json.loads(certificate_path.read_text(encoding="ascii"))
    payload["fresh_reload_structural_topology_fingerprint"] = "f" * 64
    payload.pop("certificate_sha256")
    payload["certificate_sha256"] = canonical_sha256(payload)
    certificate_path.write_bytes(canonical_json_bytes(payload) + b"\n")

    with pytest.raises(
        DynamicFoldArtifactError,
        match="differs from compiled child provenance",
    ):
        load_dynamic_fold_artifact_certificate(destination)


@pytest.mark.parametrize(
    "mutate",
    (_drift_node_path, _drift_config, _drift_parameter_shape),
    ids=("topology", "config", "shape"),
)
def test_publication_rejects_nontrainability_graph_drift(tmp_path, mutate) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"

    with pytest.raises(
        DynamicFoldArtifactError,
        match="structural topology differs from the compiled child",
    ):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=_loader,
            fresh_reload_evaluator=_fresh_accept,
            saver_id="test-saver",
            loader_id="structurally-drifting-loader",
            inspectors=_DriftingInspectors(mutate),
            tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()


def test_structural_topology_drift_error_names_the_differing_leaves(
    tmp_path,
) -> None:
    """A fail-closed reload must say which graph leaf moved, not just that one did."""

    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"

    with pytest.raises(DynamicFoldArtifactError) as failure:
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=_loader,
            fresh_reload_evaluator=_fresh_accept,
            saver_id="test-saver",
            loader_id="config-drifting-loader",
            inspectors=_DriftingInspectors(_drift_config),
            tokenizer_fingerprint=tokenizer,
        )

    message = str(failure.value)
    assert "Differing leaves:" in message
    assert "config_fingerprint" in message
    assert not destination.exists()


def test_structural_topology_differences_reports_every_leaf_shape() -> None:
    assert _structural_topology_differences({"a": 1}, {"a": 1}) == []
    # A reload that fabricates a configuration key the compiled child never had
    # is the exact shape of the LFM2 carrier-lane block_ff_dim alias defect.
    assert _structural_topology_differences(
        {"config": {"intermediate_size": 11}},
        {"config": {"intermediate_size": 11, "block_ff_dim": 11}},
    ) == ["config.block_ff_dim: missing in compiled child"]
    assert _structural_topology_differences(
        {"config": {"intermediate_size": 11, "block_ff_dim": 11}},
        {"config": {"intermediate_size": 11}},
    ) == ["config.block_ff_dim: missing in {observed}"]
    assert _structural_topology_differences(
        {"nodes": [{"path": "a"}, {"path": "b"}]},
        {"nodes": [{"path": "a"}, {"path": "b.drifted"}]},
    ) == ["nodes[1].path: 'b' != 'b.drifted'"]
    assert _structural_topology_differences(
        {"nodes": [1, 2]},
        {"nodes": [1]},
    ) == ["nodes: length 2 != 1"]


def test_legacy_v2_certificate_remains_loadable(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"
    publish_dynamic_fold_artifact(
        compilation=compilation,
        child_model=child,
        destination=destination,
        model_saver=_saver,
        model_loader=_loader,
        fresh_reload_evaluator=_fresh_accept,
        saver_id="test-local-torch-save-v1",
        loader_id="test-local-torch-load-v1",
        tokenizer_fingerprint=tokenizer,
    )
    certificate_path = destination / "certificate.json"
    payload = json.loads(certificate_path.read_text(encoding="ascii"))
    payload["schema"] = LEGACY_SCHEMA
    payload.pop("fresh_reload_inference_entrypoint_calls")
    payload.pop("fresh_reload_declared_inference_path_observed")
    payload.pop("fresh_reload_structural_topology_fingerprint")
    payload.pop("fresh_reload_parameters_frozen")
    payload.pop("structural_topology_excluded_parameter_fields")
    payload.pop("fresh_reload_structural_topology_matches_compiled_child")
    payload.pop("certificate_sha256")
    payload["certificate_sha256"] = canonical_sha256(payload)
    certificate_path.write_bytes(canonical_json_bytes(payload) + b"\n")

    loaded = load_dynamic_fold_artifact_certificate(destination)

    assert loaded.certificate_schema == LEGACY_SCHEMA
    assert loaded.fresh_reload_inference_entrypoint_calls == (
        loaded.fresh_reload_root_forward_calls
    )


def test_previous_v3_certificate_remains_loadable(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"
    publish_dynamic_fold_artifact(
        compilation=compilation,
        child_model=child,
        destination=destination,
        model_saver=_saver,
        model_loader=_loader,
        fresh_reload_evaluator=_fresh_accept,
        saver_id="test-local-torch-save-v1",
        loader_id="test-local-torch-load-v1",
        tokenizer_fingerprint=tokenizer,
    )
    certificate_path = destination / "certificate.json"
    payload = json.loads(certificate_path.read_text(encoding="ascii"))
    payload["schema"] = PREVIOUS_SCHEMA
    payload.pop("fresh_reload_structural_topology_fingerprint")
    payload.pop("fresh_reload_parameters_frozen")
    payload.pop("structural_topology_excluded_parameter_fields")
    payload.pop("fresh_reload_structural_topology_matches_compiled_child")
    payload.pop("certificate_sha256")
    payload["certificate_sha256"] = canonical_sha256(payload)
    certificate_path.write_bytes(canonical_json_bytes(payload) + b"\n")

    loaded = load_dynamic_fold_artifact_certificate(destination)

    assert loaded.certificate_schema == PREVIOUS_SCHEMA
    assert loaded.fresh_reload_structural_topology_fingerprint == "0" * 64
    assert loaded.fresh_reload_parameters_frozen is False


def test_loader_that_ignores_path_and_deepcopies_child_is_rejected(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"

    def ignores_path(_source):
        return copy.deepcopy(child)

    with pytest.raises(
        DynamicFoldArtifactError,
        match="path-consumption challenge",
    ):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=ignores_path,
            fresh_reload_evaluator=_fresh_accept,
            saver_id="test-saver",
            loader_id="path-ignoring-loader",
            tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()


def test_provenance_tamper_is_rejected_on_load(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"
    publish_dynamic_fold_artifact(
        compilation=compilation,
        child_model=child,
        destination=destination,
        model_saver=_saver,
        model_loader=_loader,
        fresh_reload_evaluator=_fresh_accept,
        saver_id="test-saver",
        loader_id="test-loader",
        tokenizer_fingerprint=tokenizer,
    )
    plan_path = destination / "provenance" / "plan.json"
    payload = json.loads(plan_path.read_text(encoding="ascii"))
    payload["evidence_fingerprint"] = "f" * 64
    plan_path.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="ascii",
    )

    with pytest.raises(
        DynamicFoldArtifactError,
        match="provenance (tree|document digests)",
    ):
        load_dynamic_fold_artifact_certificate(destination)


def test_semantic_provenance_tamper_is_rejected_after_outer_recertification(
    tmp_path,
) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"
    publish_dynamic_fold_artifact(
        compilation=compilation,
        child_model=child,
        destination=destination,
        model_saver=_saver,
        model_loader=_loader,
        fresh_reload_evaluator=_fresh_accept,
        saver_id="test-saver",
        loader_id="test-loader",
        tokenizer_fingerprint=tokenizer,
    )
    provenance = destination / "provenance"
    plan_path = provenance / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="ascii"))
    plan["evidence_fingerprint"] = "f" * 64
    plan_path.write_bytes(canonical_json_bytes(plan) + b"\n")

    certificate_path = destination / "certificate.json"
    certificate = json.loads(certificate_path.read_text(encoding="ascii"))
    rows = []
    for path in sorted(provenance.iterdir(), key=lambda value: value.name):
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    hashes = {row["path"]: row["sha256"] for row in rows}
    certificate["provenance_plan_sha256"] = hashes["plan.json"]
    certificate["provenance_tree_sha256"] = canonical_sha256(rows)
    certificate.pop("certificate_sha256")
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    certificate_path.write_bytes(canonical_json_bytes(certificate) + b"\n")

    with pytest.raises(
        DynamicFoldArtifactError,
        match="Plan provenance semantic fingerprint",
    ):
        load_dynamic_fold_artifact_certificate(destination)


def test_no_dynamic_target_marker_tamper_fails_after_outer_recertification(
    tmp_path,
) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"
    publish_dynamic_fold_artifact(
        compilation=compilation,
        child_model=child,
        destination=destination,
        model_saver=_saver,
        model_loader=_loader,
        fresh_reload_evaluator=_fresh_accept,
        saver_id="test-saver",
        loader_id="test-loader",
        tokenizer_fingerprint=tokenizer,
    )
    provenance = destination / "provenance"
    target_path = provenance / "target.json"
    target = json.loads(target_path.read_text(encoding="ascii"))
    target["reason"] = "forged-dynamic-target"
    target_path.write_bytes(canonical_json_bytes(target) + b"\n")

    rows = []
    for path in sorted(provenance.iterdir(), key=lambda value: value.name):
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    certificate_path = destination / "certificate.json"
    certificate = json.loads(certificate_path.read_text(encoding="ascii"))
    certificate["provenance_target_sha256"] = next(
        row["sha256"] for row in rows if row["path"] == "target.json"
    )
    certificate["provenance_target_fingerprint"] = canonical_sha256(target)
    certificate["provenance_tree_sha256"] = canonical_sha256(rows)
    certificate.pop("certificate_sha256")
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    certificate_path.write_bytes(canonical_json_bytes(certificate) + b"\n")

    with pytest.raises(
        DynamicFoldArtifactError,
        match="invalid no-dynamic-target marker",
    ):
        load_dynamic_fold_artifact_certificate(destination)


def test_publication_rejects_a_swallowed_compilation_training_attempt(
    tmp_path,
) -> None:
    child, compilation, tokenizer = _compiled()
    forged = replace(
        compilation,
        run=replace(compilation.run, backward_calls=1),
    )
    destination = tmp_path / "published"

    with pytest.raises(
        DynamicFoldArtifactError,
        match="forbidden training API attempt",
    ):
        publish_dynamic_fold_artifact(
            compilation=forged,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=_loader,
            fresh_reload_evaluator=_fresh_accept,
            saver_id="test-saver",
            loader_id="test-loader",
            tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()


def test_publication_rejects_a_mutating_saver_atomically(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"

    def mutating_saver(model, output):
        _saver(model, output)
        model.lm_head.register_forward_hook(
            lambda _module, _inputs, value: torch.zeros_like(value)
        )

    with pytest.raises(DynamicFoldArtifactError, match="saver mutated"):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=mutating_saver,
            model_loader=_loader,
            fresh_reload_evaluator=_fresh_accept,
            saver_id="mutating-saver",
            loader_id="test-loader",
            tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()
    assert not child.lm_head._forward_hooks


def test_publication_rejects_a_mutating_reload_evaluator(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"

    def mutating_evaluator(model):
        _fresh_accept(model)
        model.train()
        return True

    with pytest.raises(DynamicFoldArtifactError, match="evaluator mutated"):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=_loader,
            fresh_reload_evaluator=mutating_evaluator,
            saver_id="test-saver",
            loader_id="test-loader",
            tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()


def test_publication_forbids_backward_during_reload_evaluation(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"

    def backward_evaluator(_model):
        value = torch.tensor(2.0, requires_grad=True)
        (value * value).backward()
        return True

    with pytest.raises(Exception, match="Backward calls are forbidden"):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=_loader,
            fresh_reload_evaluator=backward_evaluator,
            saver_id="test-saver",
            loader_id="test-loader",
            tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()


def test_prebound_backward_alias_cannot_train_reloaded_child(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"
    backward_alias = torch.autograd.backward
    shared = {}

    def graph_building_loader(source):
        model = _loader(source)
        shared["model"] = model
        shared["loss"] = next(model.parameters()).sum()
        return model

    def alias_evaluator(_model):
        backward_alias(shared["loss"])
        return True

    with pytest.raises(
        RuntimeError,
        match="does not require grad",
    ):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=graph_building_loader,
            fresh_reload_evaluator=alias_evaluator,
            saver_id="test-saver",
            loader_id="graph-building-loader",
            tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()
    assert all(parameter.grad is None for parameter in shared["model"].parameters())


def test_fresh_loader_cannot_return_in_memory_child(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"

    def aliased_loader(source):
        if not (source / "config.json").is_file():
            raise FileNotFoundError("challenge has no serialized child")
        return child

    with pytest.raises(
        DynamicFoldArtifactError,
        match="in-memory child object",
    ):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=aliased_loader,
            fresh_reload_evaluator=_fresh_accept,
            saver_id="test-saver",
            loader_id="aliased-loader",
            tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()


def test_fresh_reload_acceptance_requires_observed_model_execution(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"

    with pytest.raises(
        DynamicFoldArtifactError,
        match="without executing the published child",
    ):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=_loader,
            fresh_reload_evaluator=lambda _model: True,
            saver_id="test-saver",
            loader_id="cached-output-only-loader",
            tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()


def test_fresh_reload_rejects_leaf_only_cached_acceptance(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"

    def leaf_only_evaluator(model):
        with torch.no_grad():
            model.model.embed_tokens(torch.tensor([[1, 2]], dtype=torch.long))
        return True

    with pytest.raises(
        DynamicFoldArtifactError,
        match="without executing the published child root",
    ):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=_loader,
            fresh_reload_evaluator=leaf_only_evaluator,
            saver_id="test-saver",
            loader_id="leaf-only-loader",
            tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()


def test_fresh_reload_accepts_authenticated_wrapper_module_root(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"
    contract = BehaviorInferenceExecutionContract(
        entrypoint="chat",
        required_module_path="model",
    )
    gate_id, gate_version, baseline, fingerprint, manifest = _wrapper_behavior_manifest(
        contract
    )

    certificate = publish_dynamic_fold_artifact(
        compilation=compilation,
        child_model=child,
        destination=destination,
        model_saver=_saver,
        model_loader=_loader_with_chat,
        fresh_reload_evaluator=_wrapper_fresh_accept,
        saver_id="test-saver",
        loader_id="wrapper-loader",
        behavior_gate_id=gate_id,
        behavior_gate_version=gate_version,
        behavior_baseline_fingerprint=baseline,
        behavior_gate_manifest_fingerprint=fingerprint,
        behavior_gate_manifest=manifest,
        tokenizer_fingerprint=tokenizer,
    )

    assert certificate.fresh_reload_decision.accepted
    assert certificate.fresh_reload_root_forward_calls > 0
    assert certificate.fresh_reload_inference_entrypoint_calls > 0
    assert certificate.behavior_gate_manifest["inference_execution"] == (
        contract.as_dict()
    )
    assert load_dynamic_fold_artifact_certificate(destination) == certificate


def test_fresh_reload_wrapper_rejects_unrelated_descendant_execution(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"
    contract = BehaviorInferenceExecutionContract(
        entrypoint="chat",
        required_module_path="model",
    )
    gate_id, gate_version, baseline, fingerprint, manifest = _wrapper_behavior_manifest(
        contract
    )

    def unrelated_loader(source):
        model = _loader(source)

        def chat(input_ids):
            embedded = model.model.embed_tokens(input_ids)
            return model.lm_head(embedded)

        object.__setattr__(model, "chat", chat)
        return model

    with pytest.raises(
        DynamicFoldArtifactError,
        match="declared inference entrypoint and module root",
    ):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=unrelated_loader,
            fresh_reload_evaluator=_wrapper_fresh_accept,
            saver_id="test-saver",
            loader_id="unrelated-wrapper-loader",
            behavior_gate_id=gate_id,
            behavior_gate_version=gate_version,
            behavior_baseline_fingerprint=baseline,
            behavior_gate_manifest_fingerprint=fingerprint,
            behavior_gate_manifest=manifest,
            tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()


def test_loader_must_reload_from_retained_final_destination(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    destination = tmp_path / "published"
    valid_load_paths = []

    def stage_only_loader(source):
        if not (source / "config.json").is_file():
            return _loader(source)
        valid_load_paths.append(source.absolute())
        if source.parent.name.startswith(".dynamic-fold-stage-"):
            return _loader(source)
        raise RuntimeError("loader refuses retained final destination")

    with pytest.raises(
        RuntimeError,
        match="refuses retained final destination",
    ):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=stage_only_loader,
            fresh_reload_evaluator=_fresh_accept,
            saver_id="test-saver",
            loader_id="stage-only-loader",
            tokenizer_fingerprint=tokenizer,
        )

    assert len(valid_load_paths) == 2
    assert valid_load_paths[0].parent.name.startswith(".dynamic-fold-stage-")
    assert valid_load_paths[1] == (destination / "child").absolute()
    assert not destination.exists()
    assert not any(
        path.name.startswith(".dynamic-fold-stage-") for path in tmp_path.iterdir()
    )


def test_publication_rejects_live_parent_forward_from_saver(tmp_path) -> None:
    child, compilation, tokenizer, parent = _compiled(include_parent=True)
    destination = tmp_path / "published"

    def parent_forwarding_saver(model, output):
        with torch.no_grad():
            parent(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                use_cache=False,
            )
        _saver(model, output)

    with pytest.raises(
        DynamicFoldArtifactError,
        match="publication executed a live parent model",
    ):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=parent_forwarding_saver,
            model_loader=_loader,
            fresh_reload_evaluator=_fresh_accept,
            saver_id="parent-forwarding-saver",
            loader_id="test-loader",
            tokenizer_fingerprint=tokenizer,
            parent_models={"parent": parent},
        )

    assert not destination.exists()


def test_fresh_reload_must_be_disjoint_from_every_live_parent(tmp_path) -> None:
    child, compilation, tokenizer, parent = _compiled(include_parent=True)
    destination = tmp_path / "published"

    def returns_parent(source):
        if not (source / "config.json").is_file():
            raise FileNotFoundError("challenge has no serialized child")
        return parent

    with pytest.raises(DynamicFoldArtifactError, match="live parent 'parent' object"):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=returns_parent,
            fresh_reload_evaluator=_fresh_accept,
            saver_id="test-saver",
            loader_id="parent-returning-loader",
            parent_models={"parent": parent},
            tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()


def test_fresh_reload_rejects_changed_buffer_persistence_inventory(tmp_path) -> None:
    child, compilation, tokenizer = _compiled()
    child.register_buffer(
        "publication_marker",
        torch.tensor([4.25], dtype=torch.float32),
        persistent=True,
    )
    destination = tmp_path / "published"

    def loader_with_nonpersistent_marker(source):
        config = transformers.LlamaConfig(
            **json.loads((source / "config.json").read_text(encoding="utf-8"))
        )
        model = transformers.LlamaForCausalLM(config)
        model.register_buffer(
            "publication_marker",
            torch.zeros(1, dtype=torch.float32),
            persistent=False,
        )
        state = torch.load(
            source / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        marker = state.pop("publication_marker")
        model.load_state_dict(state)
        model.publication_marker.copy_(marker)
        model.eval()
        model.requires_grad_(False)
        return model

    with pytest.raises(
        DynamicFoldArtifactError,
        match="inference-state inventory differs",
    ):
        publish_dynamic_fold_artifact(
            compilation=compilation,
            child_model=child,
            destination=destination,
            model_saver=_saver,
            model_loader=loader_with_nonpersistent_marker,
            fresh_reload_evaluator=_fresh_accept,
            saver_id="test-saver",
            loader_id="persistence-changing-loader",
            tokenizer_fingerprint=tokenizer,
        )

    assert not destination.exists()
