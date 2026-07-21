from __future__ import annotations

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
        "reconstruct_lfm2_saved_tied_interface",
        lambda *_a, **_k: interface,
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
    assert result["claims"]["training_performed"] is False
    assert result["claims"]["attention_correction_installed"] is True
    body = dict(result)
    seal = body.pop("result_sha256")
    assert seal == canonical_sha256(body)
