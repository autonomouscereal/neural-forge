from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

import neural_forge.hf_lfm2_semantic_foundation as foundation  # noqa: E402
import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402


def _seal(body, field="result_sha256"):
    return {**body, field: canonical_sha256(body)}


def _write_parent(directory: Path, *, changed_q: bool = False):
    directory.mkdir(parents=True, exist_ok=True)
    layer_types = list(foundation.EXACT_C61_TOPOLOGY["layer_types"])
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "lfm2",
                "tie_word_embeddings": True,
                "layer_types": layer_types,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    tensors = {
        "model.embed_tokens.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "model.embedding_norm.weight": torch.ones(4, dtype=torch.float32),
    }
    for layer_index, kind in enumerate(layer_types):
        prefix = f"model.layers.{layer_index}."
        tensors[prefix + "ffn_norm.weight"] = torch.ones(4)
        tensors[prefix + "feed_forward.w1.weight"] = torch.ones(2, 4)
        tensors[prefix + "feed_forward.w2.weight"] = torch.ones(4, 2)
        tensors[prefix + "feed_forward.w3.weight"] = torch.ones(2, 4)
        if kind != "full_attention":
            continue
        attention = prefix + "self_attn."
        q = torch.arange(4_096, dtype=torch.float32).reshape(1_024, 4)
        if changed_q and layer_index == 2:
            q = q.clone()
            q[0, 0] += 1.0
        tensors[attention + "q_proj.weight"] = q
        tensors[attention + "k_proj.weight"] = torch.ones(512, 4)
        tensors[attention + "v_proj.weight"] = torch.full((512, 4), 2.0)
        tensors[attention + "out_proj.weight"] = torch.ones(4, 1_024)
    safetensors.save_file(tensors, directory / "model.safetensors")


def _parameter_inventory():
    body = {
        "method": "fixture_unique_storage_inventory",
        "unique_parameters": foundation.EXACT_C61_TARGET_PARAMETERS,
        "raw_named_parameter_occurrences": 1,
        "raw_named_parameter_numel": foundation.EXACT_C61_TARGET_PARAMETERS,
        "parameter_iterator_unique_numel": foundation.EXACT_C61_TARGET_PARAMETERS,
        "independent_methods_agree": True,
        "alias_group_count": 0,
        "alias_groups": [],
        "checkpoint_byte_size_used": False,
        "quantization_adjustment_used": False,
        "pruning_mask_adjustment_used": False,
    }
    return _seal(body, "inventory_sha256")


def _foundation_fixture_receipt(parent_directory, parent_inventory, attention_lineage):
    parent_sha = "a" * 64
    cache_key = "b" * 64
    bridge_sha = "c" * 64
    target_plan = "d" * 64
    candidate = "e" * 64
    blank_state = "f" * 64
    installed_state = "1" * 64
    inventory = _parameter_inventory()
    blank = _seal(
        {
            "schema": "neural-forge-lfm2-cpcf-blank-target-v1",
            "target_plan_fingerprint": target_plan,
            "selected_candidate_fingerprint": candidate,
            "target_unique_parameters": foundation.EXACT_C61_TARGET_PARAMETERS,
            "blank_state_fingerprint": blank_state,
            "all_parameters_zero": True,
            "dense_materialized_parameters": True,
            "training_used": False,
            "pruning_used": False,
            "quantization_used": False,
        },
        "report_sha256",
    )
    interface_solve = _seal(
        {
            "schema": foundation.LFM2_SEMANTIC_FOUNDATION_INTERFACE_SCHEMA,
            "legacy_layer_compiler_calls": 0,
            "training_steps": 0,
        }
    )
    interface_install = _seal({"accepted": True}, "report_sha256")
    norm_install = _seal({"accepted": True}, "report_sha256")
    preparation = _seal(
        {"schema": foundation.SELECTIVE_CATALOG_PREPARATION_SCHEMA},
        "preparation_receipt_sha256",
    )
    execution_counts = {
        "module_forward_calls": 0,
        "root_forward_calls": 0,
        "inference_entrypoint_calls": 0,
    }
    guard = _seal(
        {
            "parent_execution_counts": execution_counts,
            "child_execution_counts": execution_counts,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "autograd_grad_calls": 0,
            "pruning_operations": 0,
            "quantization_operations": 0,
            "all_forbidden_operation_counts_zero": True,
        }
    )
    context_body = {
        "schema": "neural-forge-lfm2-semantic-foundation-context-v1",
        "parent_identity_sha256": parent_sha,
        "parent_inventory_sha256": parent_inventory["result_sha256"],
        "compiler_cache_key": cache_key,
        "cache_manifest_sha256": "2" * 64,
        "bridge_artifact_sha256": bridge_sha,
        "catalog_preparation_receipt_sha256": preparation[
            "preparation_receipt_sha256"
        ],
        "target_plan_fingerprint": target_plan,
        "selected_candidate_fingerprint": candidate,
        "blank_report_sha256": blank["report_sha256"],
        "blank_state_sha256": blank_state,
        "interface_solve_sha256": interface_solve["result_sha256"],
        "interface_install_sha256": interface_install["report_sha256"],
        "neutral_norm_install_sha256": norm_install["report_sha256"],
        "attention_lineage_sha256": canonical_sha256(dict(attention_lineage)),
        "installed_model_state_sha256": installed_state,
        "parameter_inventory_sha256": inventory["inventory_sha256"],
        "operation_audit": dict(foundation._ZERO_FOLD_OPERATIONS),
        "fold_only_guard_sha256": guard["result_sha256"],
    }
    context_sha = canonical_sha256(context_body)
    artifact_metadata = _seal(
        {
            "schema": "fixture-standalone-child-v1",
            "context_binding_sha256": context_sha,
            "model_state_sha256": installed_state,
            "parameter_inventory_sha256": inventory["inventory_sha256"],
        },
        "artifact_sha256",
    )
    artifact = {
        "checkpoint_directory": "fixture",
        "artifact_metadata": artifact_metadata,
        "artifact_sha256": artifact_metadata["artifact_sha256"],
        "model_state_sha256": installed_state,
        "parameter_inventory_sha256": inventory["inventory_sha256"],
        "staged_reload_verified": True,
        "published_reload_verified": True,
    }
    body = {
        "schema": foundation.LFM2_SEMANTIC_FOUNDATION_SCHEMA,
        "status": foundation.LFM2_SEMANTIC_FOUNDATION_STATUS,
        "parent_model_directory": str(parent_directory.resolve()),
        "parent_identity_sha256": parent_sha,
        "parent_source_inventory": dict(parent_inventory),
        "compiler_cache_key": cache_key,
        "cache_parent_fingerprint": parent_sha,
        "cache_manifest_sha256": "2" * 64,
        "bridge_artifact_sha256": bridge_sha,
        "catalog_preparation": preparation,
        "target_plan_fingerprint": target_plan,
        "selected_candidate_fingerprint": candidate,
        "blank_seed": foundation.EXACT_C61_BLANK_SEED,
        "blank_target": blank,
        "blank_model_state_sha256": blank_state,
        "target_topology": dict(foundation.EXACT_C61_TOPOLOGY),
        "target_unique_parameters": foundation.EXACT_C61_TARGET_PARAMETERS,
        "blank_parameter_inventory": inventory,
        "interface_solve": interface_solve,
        "interface_install": interface_install,
        "neutral_norm_install": norm_install,
        "attention_lineage": dict(attention_lineage),
        "installed_model_state_sha256": installed_state,
        "installed_parameter_inventory": inventory,
        "foundation_context": {**context_body, "context_sha256": context_sha},
        "fold_only_guard": guard,
        "standalone_artifact": artifact,
        "legacy_cpcf_layer_compiler_executed": False,
        "operation_audit": {
            **dict(foundation._ZERO_FOLD_OPERATIONS),
            "parent_model_objects_constructed": 1,
            "child_model_objects_constructed": 3,
        },
    }
    receipt = _seal(body, "receipt_sha256")
    metadata = {
        "artifact_sha256": artifact["artifact_sha256"],
        "model_state_sha256": installed_state,
        "parameter_inventory_sha256": inventory["inventory_sha256"],
        "context_binding_sha256": context_sha,
    }
    cache = SimpleNamespace(parent_fingerprint=parent_sha, cache_key=cache_key)
    bridge = SimpleNamespace(artifact_sha256=bridge_sha)
    return receipt, metadata, cache, bridge


def test_head_groups_preserve_every_source_head_and_gqa_ownership():
    groups, weights, report = foundation.build_lfm2_semantic_foundation_head_groups(
        source_query_heads=16,
        source_key_value_heads=8,
        target_query_heads=10,
        target_key_value_heads=5,
    )
    assert len(groups) == 10
    assert sorted(value for group in groups for value in group) == list(range(16))
    assert all(sum(weight) == pytest.approx(1.0) for weight in weights)
    assert report["full_source_query_coverage"] is True
    assert report["result_sha256"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "result_sha256"}
    )


def test_direct_local_parent_inventory_detects_one_changed_q_weight(tmp_path):
    parent = tmp_path / "parent"
    _write_parent(parent)
    before = foundation.inventory_lfm2_semantic_foundation_parent(
        parent,
        expected_parent_identity_sha256="a" * 64,
    )
    _write_parent(parent, changed_q=True)
    after = foundation.inventory_lfm2_semantic_foundation_parent(
        parent,
        expected_parent_identity_sha256="a" * 64,
    )
    assert before["tensor_count"] == after["tensor_count"]
    assert before["result_sha256"] != after["result_sha256"]
    assert (
        before["attention_qkvo_inventories"]["2"]["inventory_sha256"]
        != after["attention_qkvo_inventories"]["2"]["inventory_sha256"]
    )


def test_postfold_dispatch_accepts_foundation_and_rejects_changed_parent(tmp_path):
    parent = tmp_path / "parent"
    _write_parent(parent)
    parent_inventory = foundation.inventory_lfm2_semantic_foundation_parent(
        parent,
        expected_parent_identity_sha256="a" * 64,
    )
    target = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=foundation.EXACT_C61_TOPOLOGY["hidden_size"],
            num_attention_heads=foundation.EXACT_C61_TOPOLOGY["query_heads"],
            num_key_value_heads=foundation.EXACT_C61_TOPOLOGY["key_value_heads"],
            head_dim=foundation.EXACT_C61_TOPOLOGY["head_dim"],
        )
    )
    attention_lineage = foundation.build_lfm2_semantic_foundation_attention_lineage(
        parent_inventory=parent_inventory,
        target_model=target,
    )
    receipt, metadata, cache, bridge = _foundation_fixture_receipt(
        parent,
        parent_inventory,
        attention_lineage,
    )
    receipt_path = tmp_path / "foundation.json"
    foundation.write_lfm2_semantic_foundation_receipt(receipt_path, receipt)
    loaded = postfold.load_lfm2_semantic_compiler_lineage(
        receipt_path,
        expected_diagnostic_sha256=receipt["receipt_sha256"],
        cache=cache,
        bridge=bridge,
        input_metadata=metadata,
        parent_model_directory=parent,
    )
    assert set(loaded) == set(foundation.EXACT_C61_TOPOLOGY["attention_layer_indices"])
    assert all(
        value.report["foundation_receipt_sha256"] == receipt["receipt_sha256"]
        for value in loaded.values()
    )

    _write_parent(parent, changed_q=True)
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="ancestry authentication failed",
    ):
        postfold.load_lfm2_semantic_compiler_lineage(
            receipt_path,
            expected_diagnostic_sha256=receipt["receipt_sha256"],
            cache=cache,
            bridge=bridge,
            input_metadata=metadata,
            parent_model_directory=parent,
        )


def test_foundation_lineage_rejects_resealed_duplicate_head_coverage(tmp_path):
    parent = tmp_path / "parent"
    _write_parent(parent)
    parent_inventory = foundation.inventory_lfm2_semantic_foundation_parent(
        parent,
        expected_parent_identity_sha256="a" * 64,
    )
    target = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=foundation.EXACT_C61_TOPOLOGY["hidden_size"],
            num_attention_heads=foundation.EXACT_C61_TOPOLOGY["query_heads"],
            num_key_value_heads=foundation.EXACT_C61_TOPOLOGY["key_value_heads"],
            head_dim=foundation.EXACT_C61_TOPOLOGY["head_dim"],
        )
    )
    lineages = dict(
        foundation.build_lfm2_semantic_foundation_attention_lineage(
            parent_inventory=parent_inventory,
            target_model=target,
        )
    )
    forged = dict(lineages["2"])
    forged.pop("lineage_sha256")
    groups = [list(group) for group in forged["query_groups"]]
    groups[0][0] = groups[1][0]
    forged["query_groups"] = groups
    group_report = dict(forged["head_group_report"])
    group_report.pop("result_sha256")
    group_report["query_groups"] = groups
    forged["head_group_report"] = _seal(group_report)
    forged["head_group_report_sha256"] = forged["head_group_report"][
        "result_sha256"
    ]
    lineages["2"] = _seal(forged, "lineage_sha256")
    receipt, metadata, cache, bridge = _foundation_fixture_receipt(
        parent,
        parent_inventory,
        lineages,
    )
    receipt_path = tmp_path / "forged-foundation.json"
    foundation.write_lfm2_semantic_foundation_receipt(receipt_path, receipt)
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="internally inconsistent",
    ):
        postfold.load_lfm2_semantic_compiler_lineage(
            receipt_path,
            expected_diagnostic_sha256=receipt["receipt_sha256"],
            cache=cache,
            bridge=bridge,
            input_metadata=metadata,
            parent_model_directory=parent,
        )


def test_run_dispatches_foundation_interface_without_legacy_c61(monkeypatch, tmp_path):
    cache = SimpleNamespace(cache_key="a" * 64, parent_fingerprint="b" * 64)
    bridge = SimpleNamespace(artifact_sha256="c" * 64)
    tokenizer = object()
    interface = SimpleNamespace(report={"report_sha256": "d" * 64})
    interface_rehydration = {
        "compiler_cache_key": "e" * 64,
        "result_sha256": "f" * 64,
    }
    layer_stack = SimpleNamespace(
        receipt={
            "result_sha256": "1" * 64,
            "complete_causal_layer_stack_installed": False,
            "whole_stage_contains_meaningful_parameter_changes": True,
            "model_state_before_sha256": "9" * 64,
            "model_state_after_sha256": "7" * 64,
            "attention_operator_decision_count": 6,
            "attention_operator_install_count": 5,
            "short_conv_operator_decision_count": 8,
            "short_conv_operator_install_count": 7,
            "replay_fresh_mlp_evidence_compile_count": 14,
            "nonterminal_mlp_writer_decision_count": 13,
            "nonterminal_mlp_writer_install_count": 12,
            "final_mlp_producer_install_count": 0,
        }
    )
    terminal = _seal(
        {
            "schema": postfold.LFM2_SEMANTIC_LEXICAL_COMPLETION_SCHEMA,
            "semantic_layer_stack_result_sha256": "1" * 64,
            "accepted": True,
            "terminal_publication_decision_accepted": True,
            "terminal_nonincumbent_candidate_accepted": False,
            "terminal_incumbent_retained": True,
            "terminal_writer_decision_count": 1,
            "terminal_writer_install_count": 0,
            "terminal_writer_mutation_count": 0,
            "terminal_writer_incumbent_retention_count": 1,
            "installation": {
                "decision_count": 1,
                "installed": False,
                "install_count": 0,
                "mutation_count": 0,
                "incumbent_retention_count": 1,
                "exact_zero_copy_noop": True,
                "parameter_copy_calls": 0,
                "changed_state_keys": [],
                "parameter_topology_preserved": True,
                "parameter_identities_preserved": True,
                "parameter_storage_preserved": True,
                "gradient_state_preserved": True,
                "parameter_count_changed": False,
                "model_state_before_sha256": "7" * 64,
                "model_state_after_sha256": "7" * 64,
            },
            "operation_audit": dict(postfold._ZERO_OPERATIONS),
        }
    )
    ancestry_calls = []

    monkeypatch.setattr(postfold, "open_lfm2_circuit_trace_cache", lambda *_a, **_k: cache)
    monkeypatch.setattr(postfold, "load_strict_tokenizer_bridge_artifact", lambda *_a, **_k: bridge)
    monkeypatch.setattr(
        postfold,
        "load_lfm2_cpcf_v5_standalone_child",
        lambda *_a, **_k: (
            object(),
            tokenizer,
            {"artifact_sha256": "2" * 64, "model_state_sha256": "3" * 64},
        ),
    )
    monkeypatch.setattr(postfold, "fingerprint_hf_tokenizer_semantics", lambda _v: "4" * 64)
    monkeypatch.setattr(
        postfold,
        "load_lfm2_parent_final_rmsnorm_from_local_safetensors",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        postfold,
        "load_lfm2_mapped_parent_tied_table_from_local_safetensors",
        lambda *_a, **_k: (object(), {"result_sha256": "5" * 64}),
    )

    def lineage_loader(*_args, **kwargs):
        ancestry_calls.append(("lineage", kwargs.get("parent_model_directory")))
        return MappingProxyType({})

    def foundation_interface(*_args, **kwargs):
        ancestry_calls.append(("foundation", kwargs["parent_model_directory"]))
        return interface, interface_rehydration

    monkeypatch.setattr(postfold, "load_lfm2_semantic_compiler_lineage", lineage_loader)
    monkeypatch.setattr(
        postfold,
        "rehydrate_lfm2_semantic_foundation_interface",
        foundation_interface,
    )
    monkeypatch.setattr(
        postfold,
        "rehydrate_lfm2_c61_tied_interface",
        lambda *_a, **_k: pytest.fail("legacy c61 path was selected"),
    )
    monkeypatch.setattr(
        postfold,
        "compile_install_lfm2_semantic_layer_stack_stage",
        lambda **_k: layer_stack,
    )
    monkeypatch.setattr(
        postfold,
        "publish_lfm2_semantic_postfold_checkpoint",
        lambda **_k: (object(), {"result_sha256": "6" * 64}),
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256="7" * 64),
    )
    for name in ("input", "cache", "bridge", "parent"):
        (tmp_path / name).mkdir()
    ancestry = tmp_path / "foundation.json"
    ancestry.write_text(
        json.dumps({"schema": foundation.LFM2_SEMANTIC_FOUNDATION_SCHEMA}) + "\n",
        encoding="ascii",
    )
    result = postfold.run_lfm2_semantic_postfold(
        input_checkpoint_directory=tmp_path / "input",
        expected_input_artifact_sha256="2" * 64,
        output_checkpoint_directory=tmp_path / "output",
        cache_directory=tmp_path / "cache",
        expected_cache_key=cache.cache_key,
        bridge_directory=tmp_path / "bridge",
        expected_bridge_artifact_sha256=bridge.artifact_sha256,
        donor_tokenizer=object(),
        target_tokenizer=tokenizer,
        parent_model_directory=tmp_path / "parent",
        compiler_diagnostic_json=ancestry,
        expected_compiler_diagnostic_sha256="8" * 64,
        device="cpu",
        fused_terminal_compiler=lambda **_k: terminal,
    )
    assert [name for name, _path in ancestry_calls] == ["lineage", "foundation"]
    assert all(path == tmp_path / "parent" for _name, path in ancestry_calls)
    assert result["compiler_ancestry_schema"] == foundation.LFM2_SEMANTIC_FOUNDATION_SCHEMA


class _TinyLfm2(torch.nn.Module):
    def __init__(self, *, fill=0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((4, 4), fill))
        self.config = SimpleNamespace(
            model_type="lfm2",
            vocab_size=4,
            hidden_size=4,
            intermediate_size=1,
            layer_types=[],
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
        )
        self.model = SimpleNamespace(layers=[])

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.weight)

    def get_output_embeddings(self):
        return SimpleNamespace(weight=self.weight)


def test_materializer_saves_and_freshly_reloads_twice(monkeypatch, tmp_path):
    topology = MappingProxyType(
        {
            "vocabulary_size": 4,
            "hidden_size": 4,
            "intermediate_size": 1,
            "layer_count": 0,
            "layer_types": [],
            "query_heads": 1,
            "key_value_heads": 1,
            "head_dim": 4,
            "attention_layer_indices": [],
        }
    )
    monkeypatch.setattr(foundation, "EXACT_C61_TARGET_PARAMETERS", 16)
    monkeypatch.setattr(foundation, "EXACT_C61_TOPOLOGY", topology)
    parent = _TinyLfm2(fill=2.0)
    parent_sha = "a" * 64
    plan_sha = "b" * 64
    candidate_sha = "c" * 64
    target = _TinyLfm2()

    class Blank:
        model = target
        unique_parameters = 16
        plan = SimpleNamespace(fingerprint=plan_sha)

        def as_dict(self):
            body = {
                "target_plan_fingerprint": plan_sha,
                "selected_candidate_fingerprint": candidate_sha,
                "target_unique_parameters": 16,
                "blank_state_fingerprint": foundation.fingerprint_model_state(target),
                "all_parameters_zero": True,
                "dense_materialized_parameters": True,
                "training_used": False,
                "pruning_used": False,
                "quantization_used": False,
            }
            return _seal(body, "report_sha256")

    monkeypatch.setattr(foundation, "plan_and_synthesize_lfm2_cpcf_blank", lambda *_a, **_k: Blank())
    monkeypatch.setattr(
        foundation,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256=parent_sha),
    )
    parent_inventory = _seal({"schema": "fixture-parent-inventory-v1"})
    monkeypatch.setattr(
        foundation,
        "inventory_lfm2_semantic_foundation_parent",
        lambda *_a, **_k: parent_inventory,
    )
    interface_report = _seal({"schema": "fixture-interface-v1"}, "report_sha256")
    interface = SimpleNamespace(
        target_weight=torch.ones(4, 4),
        producer_encoder=torch.eye(4),
        report=interface_report,
    )
    solve = foundation.Lfm2SemanticFoundationInterfaceSolve(
        interface=interface,
        report=MappingProxyType(
            _seal(
                {
                    "legacy_layer_compiler_calls": 0,
                    "training_steps": 0,
                }
            )
        ),
    )
    monkeypatch.setattr(foundation, "fit_lfm2_semantic_foundation_interface", lambda **_k: solve)

    def install_interface(model, folded):
        with torch.no_grad():
            model.weight.copy_(folded.target_weight)
        return _seal({"accepted": True}, "report_sha256")

    monkeypatch.setattr(foundation, "install_lfm2_cpcf_tied_interface", install_interface)
    monkeypatch.setattr(
        foundation,
        "install_lfm2_cpcf_neutral_norms",
        lambda _model: _seal({"accepted": True}, "report_sha256"),
    )
    monkeypatch.setattr(
        foundation,
        "build_lfm2_semantic_foundation_attention_lineage",
        lambda **_k: MappingProxyType({}),
    )
    reload_paths = []

    class Saver:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, model, directory):
            directory = Path(directory)
            torch.save(model.state_dict(), directory / "model.pt")
            body = {
                "context_binding_sha256": self.kwargs["context_binding_sha256"],
                "model_state_sha256": self.kwargs["expected_model_state_sha256"],
                "parameter_inventory_sha256": self.kwargs[
                    "expected_parameter_inventory_sha256"
                ],
                "tokenizer_semantics_sha256": self.kwargs[
                    "expected_tokenizer_semantics_sha256"
                ],
            }
            metadata = _seal(body, "artifact_sha256")
            (directory / foundation.LFM2_CPCF_V5_ARTIFACT_METADATA).write_text(
                json.dumps(metadata, sort_keys=True),
                encoding="utf-8",
            )

    class Loader:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, directory):
            directory = Path(directory)
            reload_paths.append(directory)
            model = _TinyLfm2()
            model.load_state_dict(torch.load(directory / "model.pt", weights_only=True))
            return model

    monkeypatch.setattr(foundation, "Lfm2CpcfV5QuarantineSaver", Saver)
    monkeypatch.setattr(foundation, "Lfm2CpcfV5StrictLocalLoader", Loader)
    monkeypatch.setattr(
        foundation,
        "fingerprint_hf_tokenizer_semantics",
        lambda tokenizer: tokenizer.semantic_sha,
    )
    parent_directory = tmp_path / "parent"
    parent_directory.mkdir()
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text("[]", encoding="ascii")
    donor_tokenizer = SimpleNamespace(semantic_sha="d" * 64)
    target_tokenizer = SimpleNamespace(semantic_sha="e" * 64)
    bridge = SimpleNamespace(
        donor_tokenizer_semantics_sha256=donor_tokenizer.semantic_sha,
        target_tokenizer_semantics_sha256=target_tokenizer.semantic_sha,
        artifact_sha256="f" * 64,
        transport_profile=SimpleNamespace(
            target_to_donor_row_indices=(0, 1, 2, 3),
            donor_row_capacity=4,
            target_row_capacity=4,
        ),
    )
    cache = SimpleNamespace(
        parent_fingerprint=parent_sha,
        cache_key="1" * 64,
        manifest={"manifest_sha256": "2" * 64},
        path=cache_directory,
    )
    preparation = _seal({}, "preparation_receipt_sha256")
    output = tmp_path / "child"
    receipt = foundation.materialize_lfm2_semantic_foundation(
        parent_model=parent,
        parent_model_directory=parent_directory,
        parent_name="fixture-parent",
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        bridge=bridge,
        compiler_cache=cache,
        trajectory_corpus=trajectory,
        catalog_preparation=preparation,
        output_checkpoint_directory=output,
        expected_parent_identity_sha256=parent_sha,
        expected_target_plan_fingerprint=plan_sha,
        expected_selected_candidate_fingerprint=candidate_sha,
        device="cpu",
        dtype="bfloat16",
    )
    assert output.is_dir()
    assert len(reload_paths) == 2
    assert reload_paths[0].name.startswith(".semantic-foundation-")
    assert reload_paths[1] == output.resolve()
    assert receipt["standalone_artifact"]["staged_reload_verified"] is True
    assert receipt["standalone_artifact"]["published_reload_verified"] is True
    assert receipt["operation_audit"]["training_steps"] == 0
