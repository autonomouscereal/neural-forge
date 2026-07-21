from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402


class _TiedModel:
    def __init__(self, weight):
        self.weight = torch.nn.Parameter(weight.clone())
        self.model = SimpleNamespace(
            layers=(),
            rotary_emb=SimpleNamespace(
                inv_freq=torch.tensor([1.0], dtype=torch.float32),
                attention_scaling=1.0,
            ),
        )

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.weight)

    def get_output_embeddings(self):
        return SimpleNamespace(weight=self.weight)

    def parameters(self):
        yield self.weight

    def state_dict(self):
        return {"model.embed_tokens.weight": self.weight}


def _terminal_evidence(split: str, tag: str):
    return postfold.Lfm2CpcfTerminalActionEvidence(
        split=split,
        row_ids=(f"{tag}-row",),
        mlp_input=torch.ones((1, 2), dtype=torch.bfloat16),
        pre_mlp_residual=torch.zeros((1, 2), dtype=torch.bfloat16),
        parent_final_norm_output=torch.ones((1, 2), dtype=torch.bfloat16),
        sample_weights=torch.ones((1,), dtype=torch.float64),
    )


def test_reconstructs_saved_mean_preserving_interface_without_parent_model() -> None:
    source = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, -1.0, 1.0],
            [0.5, 4.0, -2.0],
            [-1.0, 0.25, 2.0],
        ],
        dtype=torch.bfloat16,
    )
    target = torch.tensor(
        [
            [0.5, 1.0],
            [0.5, -1.0],
            [0.5, 0.5],
            [0.5, -0.5],
        ],
        dtype=torch.bfloat16,
    )
    model = _TiedModel(target)

    interface = postfold.reconstruct_lfm2_saved_tied_interface(model, source)

    assert interface.target_width == 2
    assert interface.source_width == 3
    assert interface.vocabulary_size == 4
    assert interface.report["checkpoint_target_bytes_are_authoritative"] is True
    assert interface.report["parent_model_calls"] == 0
    body = dict(interface.report)
    seal = body.pop("report_sha256")
    assert seal == canonical_sha256(body)
    torch.testing.assert_close(
        interface.target_weight.to(torch.bfloat16),
        model.weight.detach(),
        rtol=0,
        atol=0,
    )


def _seal(value, field):
    return {**value, field: canonical_sha256(value)}


def test_compiler_lineage_accepts_distinct_semantic_cache_key(monkeypatch) -> None:
    compiler_key = "1" * 64
    semantic_key = "2" * 64
    parent = "3" * 64
    bridge_sha = "4" * 64
    target_state = "5" * 64
    artifact = {
        "query_groups": [[0]],
        "query_barycentric_weights": [[1.0]],
    }
    direct = _seal(
        {
            "candidate_id": "direct_monolithic",
            "operator_artifact": artifact,
            "operator_artifact_sha256": canonical_sha256(artifact),
            "selection_context_metrics": {
                "query_groups_sha256": canonical_sha256(
                    {"groups": [[0]], "weights": [[1.0]]}
                )
            },
        },
        "report_sha256",
    )
    endpoint = _seal(
        {
            "source_inventory": {"fixture": True},
            "source_inventory_sha256": "6" * 64,
        },
        "artifact_sha256",
    )
    layer = _seal(
        {
            "layer_index": 2,
            "operator_kind": "full_attention",
            "solver_evidence_policy": {
                "attention_candidate_evaluations": {
                    "direct_monolithic": direct,
                },
                "endpoint_residual_attention_artifact": endpoint,
                "endpoint_residual_attention_artifact_sha256": endpoint[
                    "artifact_sha256"
                ],
            },
        },
        "result_sha256",
    )
    compiled = _seal({"fixture": "compiled"}, "result_sha256")
    diagnostic = {
        "receipt_sha256": "7" * 64,
        "inputs": {
            "cache_key": compiler_key,
            "parent_fingerprint": parent,
            "bridge_artifact_sha256": bridge_sha,
        },
        "compiled_candidate_consumer_seam": {
            "cache_sha256": compiler_key,
            "parent_sha256": parent,
            "bridge_sha256": bridge_sha,
            "target_state_sha256": target_state,
        },
        "compiled_model_fold": compiled,
        "layer_results": [layer],
    }
    inventory = SimpleNamespace(inventory_sha256="6" * 64)
    monkeypatch.setattr(
        postfold,
        "_authenticated_c61_diagnostic",
        lambda *_a, **_k: (MappingProxyType(diagnostic), "8" * 64),
    )
    monkeypatch.setattr(postfold, "_typed_qkvo_inventory", lambda _value: inventory)
    monkeypatch.setattr(
        postfold,
        "Lfm2SemanticCompilerLayerLineage",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    result = postfold.load_lfm2_semantic_compiler_lineage(
        "unused.json",
        expected_diagnostic_sha256=diagnostic["receipt_sha256"],
        cache=SimpleNamespace(cache_key=semantic_key, parent_fingerprint=parent),
        bridge=SimpleNamespace(artifact_sha256=bridge_sha),
        input_metadata={"model_state_sha256": target_state},
    )

    assert result[2].report["compiler_cache_key"] == compiler_key
    assert result[2].report["semantic_evidence_cache_key"] == semantic_key
    assert (
        result[2].report["compiler_and_semantic_cache_identities_bound_separately"]
        is True
    )
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="not the compiler ancestry",
    ):
        postfold.load_lfm2_semantic_compiler_lineage(
            "unused.json",
            expected_diagnostic_sha256=diagnostic["receipt_sha256"],
            cache=SimpleNamespace(
                cache_key=semantic_key,
                parent_fingerprint="f" * 64,
            ),
            bridge=SimpleNamespace(artifact_sha256=bridge_sha),
            input_metadata={"model_state_sha256": target_state},
        )


def test_anchor_active_interface_rehydrates_exact_producer_encoder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compiler_key = "1" * 64
    semantic_key = "2" * 64
    parent = "3" * 64
    bridge_sha = "4" * 64
    final_norm_sha = "5" * 64
    source = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, -1.0, 1.0], [0.5, 4.0, -2.0]],
        dtype=torch.bfloat16,
    )
    source64 = source.to(torch.float64)
    encoder = torch.tensor(
        [[0.2, 0.1], [0.3, -0.4], [0.5, 0.2]],
        dtype=torch.float64,
    )
    base_target = torch.tensor(
        [[0.5, 1.0], [0.5, -1.0], [0.5, 0.25]],
        dtype=torch.float64,
    )
    # Anchor repair intentionally destroys the constant first column.  An
    # inverse of the saved table is therefore invalid, while the producer map
    # remains exactly the base compiler's map.
    active_target = base_target + torch.tensor(
        [[0.125, 0.0], [0.0, 0.25], [-0.125, 0.0]],
        dtype=torch.float64,
    )
    base_report = _seal(
        {
            "source_table_sha256": postfold._interface_tensor_sha256(source64),
            "target_weight_sha256": postfold._interface_tensor_sha256(base_target),
            "producer_encoder_sha256": postfold._interface_tensor_sha256(encoder),
        },
        "report_sha256",
    )
    anchor_report = _seal(
        {
            "base_interface_report_sha256": base_report["report_sha256"],
            "source_tied_table_sha256": postfold._interface_tensor_sha256(source64),
            "target_weight_sha256": postfold._interface_tensor_sha256(active_target),
            "producer_encoder_sha256": postfold._interface_tensor_sha256(encoder),
            "selected_candidate_is_base_control": False,
            "active_anchor_residual_selected": True,
        },
        "report_sha256",
    )
    interface = postfold.TiedCausalInterfaceFold(
        target_weight=active_target,
        producer_encoder=encoder,
        report=anchor_report,
    )
    base_interface = postfold.TiedCausalInterfaceFold(
        target_weight=base_target,
        producer_encoder=encoder,
        report=base_report,
    )
    model = _TiedModel(active_target.to(torch.bfloat16))
    model.config = SimpleNamespace(hidden_size=2)

    store = tmp_path / "catalog-store"
    artifact = store / "artifact"
    artifact.mkdir(parents=True)
    compiler_cache = tmp_path / "compiler-cache"
    compiler_cache.mkdir()
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text("[]", encoding="ascii")
    trajectory_sha = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    preparation_path = tmp_path / "catalog-receipt.json"
    preparation_body = {
        "catalog_receipt_json": str(preparation_path.resolve()),
        "catalog_store_directory": str(store.resolve()),
        "trace_cache_directory": str(compiler_cache.resolve()),
        "artifact_path": str(artifact.resolve()),
        "cache_key": compiler_key,
        "binding_sha256": "6" * 64,
        "artifact_manifest_sha256": "7" * 64,
        "artifact_catalog_sha256": "8" * 64,
        "selective_catalog_sha256": "9" * 64,
        "selective_source_sha256": "a" * 64,
        "selective_provider_sha256": "b" * 64,
    }
    preparation = _seal(preparation_body, "preparation_receipt_sha256")
    preparation_path.write_bytes(
        (
            json.dumps(
                preparation,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
    )
    lineage = {
        "base_interface_report": base_report,
        "causal_anchor_report": anchor_report,
        "base_interface_report_sha256": base_report["report_sha256"],
        "causal_anchor_report_sha256": anchor_report["report_sha256"],
        "active_interface_report_sha256": anchor_report["report_sha256"],
        "active_interface_is_anchor_repaired": True,
        "active_interface_is_base_control": False,
        "producer_encoder_fixed_by_anchor_selection": True,
    }
    bundle = _seal(
        {
            "interface_report_sha256": anchor_report["report_sha256"],
            "base_interface_report_sha256": base_report["report_sha256"],
            "causal_anchor_report_sha256": anchor_report["report_sha256"],
            "interface_lineage": lineage,
        },
        "bundle_sha256",
    )
    compiled = _seal(
        {
            "selective_catalog_sha256": preparation["selective_catalog_sha256"],
            "selective_source_sha256": preparation["selective_source_sha256"],
            "selective_provider_sha256": preparation["selective_provider_sha256"],
            "final_norm_authentication_sha256": final_norm_sha,
            "boundary_bundle_sha256": bundle["bundle_sha256"],
        },
        "result_sha256",
    )
    install = _seal(
        {
            "interface_report_sha256": anchor_report["report_sha256"],
            "target_weight_sha256": postfold.cpcf_model_compiler._tensor_sha256(
                model.weight.detach()
            ),
        },
        "report_sha256",
    )
    inputs = {
        "cache_key": compiler_key,
        "parent_fingerprint": parent,
        "bridge_artifact_sha256": bridge_sha,
        "trace_cache_directory": str(compiler_cache.resolve()),
        "trajectory_corpus": str(trajectory.resolve()),
        "trajectory_corpus_sha256": trajectory_sha,
        "trace_adapter_sha256": "c" * 64,
        "selective_catalog_preparation_receipt_sha256": preparation[
            "preparation_receipt_sha256"
        ],
        "selective_catalog_binding_sha256": preparation["binding_sha256"],
        "selective_catalog_artifact_manifest_sha256": preparation[
            "artifact_manifest_sha256"
        ],
        "selective_catalog_artifact_sha256": preparation["artifact_catalog_sha256"],
    }
    diagnostic = {
        "receipt_sha256": "d" * 64,
        "inputs": inputs,
        "request": {
            "selective_catalog_store_directory": str(store.resolve()),
        },
        "selective_catalog_artifact": preparation,
        "compiled_model_fold": compiled,
        "boundary_bundle": bundle,
        "interface_install": install,
        "compiled_candidate_consumer_seam": {
            "cache_sha256": compiler_key,
            "parent_sha256": parent,
            "bridge_sha256": bridge_sha,
        },
    }

    class Provider:
        catalog = SimpleNamespace(
            catalog_sha256=preparation["selective_catalog_sha256"],
            source_sha256=preparation["selective_source_sha256"],
            provider_sha256=preparation["selective_provider_sha256"],
        )

        def load_final_norm_payload(self):
            return object()

        def authenticate_final_norm_payload(self, _payload):
            return final_norm_sha

        def release_final_norm_payload(self, _payload):
            return None

        def close(self):
            return None

    class BoundaryCompiler:
        def __init__(self, *_args, **_kwargs):
            self.base_interface = base_interface
            self.anchor_fold = SimpleNamespace(report=anchor_report)

        def fit_interface(self, *_args, **_kwargs):
            return interface

    monkeypatch.setattr(
        postfold,
        "_authenticated_c61_diagnostic",
        lambda *_a, **_k: (MappingProxyType(diagnostic), "e" * 64),
    )
    monkeypatch.setattr(
        postfold,
        "open_lfm2_circuit_trace_cache",
        lambda *_a, **_k: SimpleNamespace(
            cache_key=compiler_key,
            parent_fingerprint=parent,
        ),
    )
    monkeypatch.setattr(postfold.cpcf_resources, "_read_trajectory_rows", lambda _p: ())
    monkeypatch.setattr(
        postfold.cpcf_resources,
        "_build_trace_adapter",
        lambda **_k: SimpleNamespace(evidence_sha256="c" * 64),
    )
    monkeypatch.setattr(
        postfold,
        "open_lfm2_cpcf_selective_evidence_provider_from_catalog",
        lambda *_a, **_k: Provider(),
    )
    monkeypatch.setattr(
        postfold.cpcf_v5_compiler,
        "build_lfm2_cpcf_v5_target_phase_shells",
        lambda *_a, **_k: (),
    )
    monkeypatch.setattr(
        postfold.cpcf_v5_compiler,
        "materialize_lfm2_cpcf_v5_final_norm_phases",
        lambda *_a, **_k: (),
    )
    monkeypatch.setattr(
        postfold.cpcf_v5_compiler,
        "_SelectiveBoundaryCompiler",
        BoundaryCompiler,
    )
    monkeypatch.setattr(
        postfold.cpcf_model_fold, "_final_norm_epsilon", lambda _m: 1e-5
    )

    rehydrated, receipt = postfold.rehydrate_lfm2_c61_tied_interface(
        model,
        source,
        semantic_cache=SimpleNamespace(
            cache_key=semantic_key,
            parent_fingerprint=parent,
        ),
        bridge=SimpleNamespace(artifact_sha256=bridge_sha),
        target_tokenizer=object(),
        compiler_diagnostic_json="unused.json",
        expected_compiler_diagnostic_sha256=diagnostic["receipt_sha256"],
    )

    torch.testing.assert_close(rehydrated.producer_encoder, encoder, rtol=0, atol=0)
    assert receipt["active_interface_is_anchor_repaired"] is True
    assert receipt["producer_encoder_recovered_from_repaired_target_table"] is False
    assert receipt["compiler_cache_key"] == compiler_key
    assert receipt["semantic_evidence_cache_key"] == semantic_key


def test_attention_installation_receipt_requires_a_real_layer_local_change() -> None:
    receipt = postfold.seal_lfm2_semantic_attention_installation_receipt(
        layer_index=2,
        transport_evidence_sha256=("1" * 64, "2" * 64),
        model_state_before_sha256="3" * 64,
        model_state_after_sha256="4" * 64,
        changed_state_keys=("model.layers.2.self_attn.q_proj.weight",),
        parameter_element_count=57_405_368,
    )

    assert receipt["installed"] is True
    assert receipt["parameter_count_changed"] is False
    assert receipt["operation_audit"]["training_steps"] == 0
    body = dict(receipt)
    seal = body.pop("result_sha256")
    assert seal == canonical_sha256(body)

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="must change model state",
    ):
        postfold.seal_lfm2_semantic_attention_installation_receipt(
            layer_index=2,
            transport_evidence_sha256=("1" * 64,),
            model_state_before_sha256="3" * 64,
            model_state_after_sha256="3" * 64,
            changed_state_keys=("model.layers.2.self_attn.q_proj.weight",),
            parameter_element_count=10,
        )


def test_layer_stack_defers_final_w2_and_fuses_three_equal_mass_lanes() -> None:
    evidence = {
        split: _terminal_evidence(split, "final-mlp")
        for split in ("fit", "fit_check", "selection", "validation")
    }
    evidence_sha = {
        split: evidence[split].authentication_sha256 for split in sorted(evidence)
    }
    body = {
        "final_layer_mlp_w2_deferred_to_fused_terminal": True,
        "final_layer_mlp_w2_install_count": 0,
        "final_layer_mlp_evidence_sha256": evidence_sha,
    }
    installation = postfold.Lfm2SemanticLayerStackInstallation(
        receipt={**body, "result_sha256": canonical_sha256(body)},
        final_layer_mlp_evidence=evidence,
    )

    assert installation.receipt["final_layer_mlp_w2_install_count"] == 0
    merged = postfold._merge_terminal_evidence(
        {
            "final_layer_mlp": evidence["fit"],
            "lexical": _terminal_evidence("fit", "lexical"),
            "parent_endpoint": _terminal_evidence("fit", "parent"),
        }
    )
    assert merged.rows == 3
    assert float(merged.sample_weights.sum().item()) == pytest.approx(1.0)
    assert tuple(merged.sample_weights.tolist()) == pytest.approx((1.0 / 3.0,) * 3)


def test_lexical_coordinates_cover_all_87_inserted_rows() -> None:
    multiplicities = (1,) * 273 + (2,) * 66 + (3,) * 9 + (4,)
    phases = [
        SimpleNamespace(
            phase="prefill",
            target_full_prefix_length=10,
        )
    ]
    length = 10
    token = 100
    for index, multiplicity in enumerate(multiplicities):
        replacement = tuple(range(token, token + multiplicity))
        phases.append(
            SimpleNamespace(
                phase=f"decode_{index:03d}",
                target_replacement_token_ids=replacement,
                target_replaced_suffix_token_count=0,
                target_rewrite_start=length,
                target_full_prefix_length=length + multiplicity,
            )
        )
        length += multiplicity
        token += multiplicity
    semantic = SimpleNamespace(source_batch_index=7, phases=tuple(phases))

    rows, desired, row_ids, observed = (
        postfold._append_only_lexical_supervision_coordinates(semantic)
    )

    assert len(observed) == 349
    assert len(rows) == len(desired) == len(row_ids) == 87
    assert length - 10 == 436
    assert max(observed) == 4
    assert {value: observed.count(value) for value in set(observed)} == {
        1: 273,
        2: 66,
        3: 9,
        4: 1,
    }
    assert len(set(row_ids)) == 87


def test_attention_stage_fails_closed_with_exact_missing_compiler(
    monkeypatch,
) -> None:
    class FakeCache:
        cache_key = "a" * 64
        parent_fingerprint = "b" * 64
        batch_count = 1

        def selective_authenticated_session(self):
            class Session:
                batch_references = ()

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                def verify(self):
                    return self

                def as_dict(self):
                    return {"session_sha256": "c" * 64}

            return Session()

    class FakeBridge:
        artifact_sha256 = "d" * 64

        def verify_live_tokenizers(self, _donor, _target):
            return self

    alignment = SimpleNamespace(
        alignment_sha256="e" * 64,
        to_evidence_dict=lambda: {
            "alignment_sha256": "e" * 64,
        },
    )
    target = SimpleNamespace(_neural_forge_quarantine_tokenizer=object())
    monkeypatch.setattr(postfold, "Lfm2CircuitTraceCache", FakeCache)
    monkeypatch.setattr(postfold, "StrictTokenizerBridgeArtifact", FakeBridge)
    monkeypatch.setattr(
        postfold,
        "fingerprint_hf_tokenizer_semantics",
        lambda _value: "f" * 64,
    )
    monkeypatch.setattr(
        postfold,
        "_attention_layer_indices",
        lambda _cache, _model: (2,),
    )
    monkeypatch.setattr(
        postfold,
        "_target_rope",
        lambda _model: (torch.tensor([1.0], dtype=torch.float64), 1.0),
    )
    monkeypatch.setattr(
        postfold,
        "_semantic_alignments_from_session",
        lambda **_kwargs: (alignment,),
    )
    monkeypatch.setattr(
        postfold,
        "audit_lfm2_semantic_span_cpcf_refinement",
        lambda *_args, **_kwargs: {
            "probability_preserving_attention_transport_go": True,
            "blocking_reasons": [],
            "report_sha256": "0" * 64,
        },
    )
    monkeypatch.setattr(postfold, "_resolve_attention_installer", lambda _value: None)
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256="9" * 64),
    )
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 57_405_368)
    monkeypatch.setattr(
        postfold,
        "load_lfm2_parent_qkvo_inventory_from_local_safetensors",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(postfold.Lfm2SemanticPostfoldBlockedError) as captured:
        postfold.compile_install_lfm2_semantic_attention_stage(
            cache=FakeCache(),
            bridge=FakeBridge(),
            target_model=target,
            interface=object(),
            compiler_lineage={
                2: SimpleNamespace(source_inventory=object()),
            },
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
        )

    report = dict(captured.value.report)
    assert report["status"] == "blocked-no-checkpoint-published"
    assert report["terminal_refold_called"] is False
    assert report["checkpoint_save_called"] is False
    assert report["missing_callable"] == (
        "neural_forge.hf_lfm2_semantic_attention_refold_adapter."
        "compile_install_lfm2_semantic_attention_postfold"
    )
    assert report["operation_audit"]["training_steps"] == 0
    body = dict(report)
    seal = body.pop("failure_sha256")
    assert seal == canonical_sha256(body)


def test_checkpoint_publication_is_staged_then_reloaded_after_atomic_move(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _TiedModel(torch.full((4, 2), 0.5, dtype=torch.bfloat16))
    tokenizer = object()
    events = []
    model_state = "1" * 64
    inventory_sha = "2" * 64
    tokenizer_sha = "3" * 64

    class Saver:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _model, directory):
            events.append(("save", directory.name))
            body = {
                "schema": postfold.LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA,
                "dense_safetensors_only": True,
                "target_tokenizer_saved_with_child": True,
                "local_reload_only": True,
                "network_access": False,
                "hub_loader_used": False,
            }
            value = {**body, "artifact_sha256": canonical_sha256(body)}
            (directory / postfold.LFM2_CPCF_V5_ARTIFACT_METADATA).write_text(
                json.dumps(value),
                encoding="utf-8",
            )

    class Loader:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, directory):
            events.append(("load", directory.name))
            return model

    monkeypatch.setattr(postfold, "Lfm2CpcfV5QuarantineSaver", Saver)
    monkeypatch.setattr(postfold, "Lfm2CpcfV5StrictLocalLoader", Loader)
    monkeypatch.setattr(postfold, "fingerprint_model_state", lambda _model: model_state)
    monkeypatch.setattr(
        postfold,
        "independent_parameter_inventory",
        lambda _model: {"inventory_sha256": inventory_sha},
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_hf_tokenizer_semantics",
        lambda _tokenizer: tokenizer_sha,
    )
    monkeypatch.setattr(postfold, "_ffn_width", lambda _model: 7)
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 100)
    output = tmp_path / "standalone-postfold"

    reloaded, receipt = postfold.publish_lfm2_semantic_postfold_checkpoint(
        target_model=model,
        target_tokenizer=tokenizer,
        output_directory=output,
        context_binding_sha256="4" * 64,
        device="cpu",
    )

    assert reloaded is model
    assert output.is_dir()
    assert events[0][0] == "save"
    assert events[1][0] == "load"
    assert events[1][1].startswith(".semantic-postfold-")
    assert events[2] == ("load", "standalone-postfold")
    assert receipt["staged_reload_verified"] is True
    assert receipt["published_reload_verified"] is True
    assert receipt["operation_audit"]["optimizer_steps"] == 0


def test_failed_post_move_reload_removes_only_new_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _TiedModel(torch.full((4, 2), 0.5, dtype=torch.bfloat16))

    class Saver:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _model, directory):
            body = {
                "schema": postfold.LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA,
                "dense_safetensors_only": True,
                "target_tokenizer_saved_with_child": True,
                "local_reload_only": True,
                "network_access": False,
                "hub_loader_used": False,
            }
            (directory / postfold.LFM2_CPCF_V5_ARTIFACT_METADATA).write_text(
                json.dumps({**body, "artifact_sha256": canonical_sha256(body)}),
                encoding="utf-8",
            )

    class Loader:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, directory):
            if directory.name == "failed-publication":
                raise RuntimeError("synthetic final reload failure")
            return model

    monkeypatch.setattr(postfold, "Lfm2CpcfV5QuarantineSaver", Saver)
    monkeypatch.setattr(postfold, "Lfm2CpcfV5StrictLocalLoader", Loader)
    monkeypatch.setattr(postfold, "fingerprint_model_state", lambda _model: "1" * 64)
    monkeypatch.setattr(
        postfold,
        "independent_parameter_inventory",
        lambda _model: {"inventory_sha256": "2" * 64},
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_hf_tokenizer_semantics",
        lambda _tokenizer: "3" * 64,
    )
    monkeypatch.setattr(postfold, "_ffn_width", lambda _model: 7)
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 100)
    output = tmp_path / "failed-publication"

    with pytest.raises(RuntimeError, match="synthetic final reload failure"):
        postfold.publish_lfm2_semantic_postfold_checkpoint(
            target_model=model,
            target_tokenizer=object(),
            output_directory=output,
            context_binding_sha256="4" * 64,
            device="cpu",
        )

    assert not output.exists()


def test_complete_layer_stack_fails_closed_on_conv_and_mlp_gaps(
    monkeypatch,
) -> None:
    topology = [
        {
            "contract": {
                "kind": "full_attention" if index in {2, 5, 8, 11} else "short_conv"
            }
        }
        for index in range(14)
    ]
    cache = SimpleNamespace(manifest={"layer_topology": {"layers": topology}})
    monkeypatch.setattr(
        postfold,
        "_resolve_layer_stack_installer",
        lambda _value: None,
    )

    with pytest.raises(postfold.Lfm2SemanticPostfoldBlockedError) as captured:
        postfold.compile_install_lfm2_semantic_layer_stack_stage(
            cache=cache,
            bridge=object(),
            target_model=object(),
            interface=object(),
            compiler_lineage={},
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
        )

    report = dict(captured.value.report)
    assert report["required_causal_layer_count"] == 14
    assert report["required_attention_operator_count"] == 4
    assert report["required_short_conv_operator_count"] == 10
    assert "long-horizon-short-conv-refold-missing" in report["blocking_reasons"]
    assert "replay-fresh-per-layer-mlp-refold-missing" in report["blocking_reasons"]
    assert report["fused_terminal_refold_called"] is False
    assert report["checkpoint_save_called"] is False
    body = dict(report)
    seal = body.pop("failure_sha256")
    assert seal == canonical_sha256(body)


def test_top_level_orders_layer_stack_terminal_and_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events = []
    cache = SimpleNamespace(cache_key="1" * 64, parent_fingerprint="2" * 64)
    bridge = SimpleNamespace(artifact_sha256="3" * 64)
    model = object()
    tokenizer = object()
    parent_norm = object()
    source_table = torch.ones((2, 2), dtype=torch.bfloat16)
    interface = SimpleNamespace(report={"report_sha256": "6" * 64})
    interface_rehydration = MappingProxyType(
        {
            "compiler_cache_key": "8" * 64,
            "semantic_evidence_cache_key": cache.cache_key,
            "result_sha256": "9" * 64,
        }
    )
    layer_body = {
        "complete_causal_layer_stack_installed": True,
    }
    layer_receipt = MappingProxyType(
        {**layer_body, "result_sha256": canonical_sha256(layer_body)}
    )
    layer_stack = SimpleNamespace(receipt=layer_receipt)
    terminal_body = {"accepted": True}
    terminal = MappingProxyType(
        {**terminal_body, "result_sha256": canonical_sha256(terminal_body)}
    )
    cache_open_calls = []

    def open_cache(*_args, **kwargs):
        cache_open_calls.append(kwargs.get("verify"))
        return cache

    monkeypatch.setattr(postfold, "open_lfm2_circuit_trace_cache", open_cache)
    monkeypatch.setattr(
        postfold,
        "load_strict_tokenizer_bridge_artifact",
        lambda *_a, **_k: bridge,
    )
    monkeypatch.setattr(
        postfold,
        "load_lfm2_cpcf_v5_standalone_child",
        lambda *_a, **_k: (
            model,
            tokenizer,
            MappingProxyType(
                {
                    "artifact_sha256": "4" * 64,
                    "model_state_sha256": "5" * 64,
                }
            ),
        ),
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_hf_tokenizer_semantics",
        lambda _tokenizer: "a" * 64,
    )
    monkeypatch.setattr(
        postfold,
        "load_lfm2_parent_final_rmsnorm_from_local_safetensors",
        lambda *_a, **_k: parent_norm,
    )
    monkeypatch.setattr(
        postfold,
        "load_lfm2_mapped_parent_tied_table_from_local_safetensors",
        lambda *_a, **_k: (
            source_table,
            MappingProxyType({"result_sha256": "7" * 64}),
        ),
    )
    monkeypatch.setattr(
        postfold,
        "rehydrate_lfm2_c61_tied_interface",
        lambda *_a, **_k: (interface, interface_rehydration),
    )
    monkeypatch.setattr(
        postfold,
        "load_lfm2_semantic_compiler_lineage",
        lambda *_a, **_k: MappingProxyType({}),
    )

    def layer_stack_compiler(**_kwargs):
        events.append("layer_stack")
        return layer_stack

    def terminal_compiler(**_kwargs):
        events.append("terminal")
        assert _kwargs["layer_stack"] is layer_stack
        return terminal

    def publish(**_kwargs):
        events.append("checkpoint")
        return object(), MappingProxyType({"result_sha256": "c" * 64})

    monkeypatch.setattr(
        postfold,
        "compile_install_lfm2_semantic_layer_stack_stage",
        layer_stack_compiler,
    )
    monkeypatch.setattr(
        postfold,
        "publish_lfm2_semantic_postfold_checkpoint",
        publish,
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256="d" * 64),
    )
    input_checkpoint = tmp_path / "input"
    input_checkpoint.mkdir()
    parent_directory = tmp_path / "parent"
    parent_directory.mkdir()
    bridge_directory = tmp_path / "bridge"
    bridge_directory.mkdir()
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    compiler_diagnostic = tmp_path / "compiler.json"
    compiler_diagnostic.write_text("{}", encoding="utf-8")

    result = postfold.run_lfm2_semantic_postfold(
        input_checkpoint_directory=input_checkpoint,
        expected_input_artifact_sha256="4" * 64,
        output_checkpoint_directory=tmp_path / "output",
        cache_directory=cache_directory,
        expected_cache_key=cache.cache_key,
        bridge_directory=bridge_directory,
        expected_bridge_artifact_sha256=bridge.artifact_sha256,
        donor_tokenizer=object(),
        target_tokenizer=tokenizer,
        parent_model_directory=parent_directory,
        compiler_diagnostic_json=compiler_diagnostic,
        expected_compiler_diagnostic_sha256="e" * 64,
        device="cpu",
        fused_terminal_compiler=terminal_compiler,
    )

    assert cache_open_calls == [True]
    assert events == ["layer_stack", "terminal", "checkpoint"]
    assert result["compiler_cache_key"] == "8" * 64
    assert result["semantic_evidence_cache_key"] == cache.cache_key
    assert (
        result["context_binding"][
            "compiler_and_semantic_cache_identities_bound_separately"
        ]
        is True
    )
    assert result["claims"]["training_performed"] is False
    assert result["claims"]["attention_correction_installed"] is True
    body = dict(result)
    seal = body.pop("result_sha256")
    assert seal == canonical_sha256(body)
