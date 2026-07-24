from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_foundation as foundation  # noqa: E402
import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
import neural_forge.hf_tokenizer_bridge as tokenizer_bridge  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (  # noqa: E402
    Lfm2DenseQkvoSourceInventory,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCache,
)
from neural_forge.hf_token_align import (  # noqa: E402
    StrictTokenRowMapping,
    StrictTokenTransportProfile,
)


def _seal(body, field="result_sha256"):
    return {**body, field: canonical_sha256(body)}


def _bridge(tmp_path: Path):
    root = tmp_path / "tokenizer"
    root.mkdir()
    payload = b"{}\n"
    (root / "tokenizer.json").write_bytes(payload)
    record = tokenizer_bridge.TokenizerArtifactFile(
        relative_path="tokenizer.json",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    profile = StrictTokenTransportProfile(
        donor_tokenizer_class="DonorTokenizer",
        target_tokenizer_class="TargetTokenizer",
        donor_vocab_size=8,
        target_vocab_size=4,
        donor_row_capacity=8,
        target_row_capacity=4,
        alignment_fingerprint="1" * 64,
        row_mappings=tuple(
            StrictTokenRowMapping(
                target_row_index=index,
                donor_row_index=index,
                resolution="decoded-token-surface",
            )
            for index in range(4)
        ),
    )
    body = {
        "schema": tokenizer_bridge.STRICT_TOKENIZER_BRIDGE_SCHEMA,
        "kind": "sealed-strict-target-tokenizer",
        "profile_fingerprint": profile.fingerprint(),
        "transport_profile": profile.to_dict(),
        "donor_tokenizer_semantics_sha256": "2" * 64,
        "target_tokenizer_semantics_sha256": "3" * 64,
        "target_files": [record.to_dict()],
        "model_parameter_operations": {
            "training_steps": 0,
            "backward_calls": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
            "random_target_rows": 0,
        },
    }
    return tokenizer_bridge.StrictTokenizerBridgeArtifact(
        root_dir=root,
        transport_profile=profile,
        donor_tokenizer_semantics_sha256="2" * 64,
        target_tokenizer_semantics_sha256="3" * 64,
        target_files=(record,),
        artifact_sha256=tokenizer_bridge._canonical_sha256(body),
    )


def _parent_inventory(parent_sha: str, attention_layers=(1, 3)):
    shapes = {
        "q_proj.weight": [8, 4],
        "k_proj.weight": [4, 4],
        "v_proj.weight": [4, 4],
        "out_proj.weight": [4, 8],
    }

    def source_inventory(layer_index):
        receipts = {}
        source_digest = hashlib.sha256()
        structural_digest = hashlib.sha256()
        for name in sorted(shapes):
            tensor_sha = hashlib.sha256(
                f"layer={layer_index};tensor={name}".encode()
            ).hexdigest()
            structural_sha = hashlib.sha256(
                f"layer={layer_index};structural={name}".encode()
            ).hexdigest()
            receipts[name] = {
                "shape": shapes[name],
                "dtype": "torch.float32",
                "logical_parameters": shapes[name][0] * shapes[name][1],
                "tensor_sha256": tensor_sha,
                "structural_tensor_sha256": structural_sha,
            }
            source_digest.update(name.encode())
            source_digest.update(bytes.fromhex(tensor_sha))
            structural_digest.update(name.encode())
            structural_digest.update(bytes.fromhex(structural_sha))
        return Lfm2DenseQkvoSourceInventory(
            tensor_receipts=receipts,
            source_state_sha256=source_digest.hexdigest(),
            structural_source_state_sha256=structural_digest.hexdigest(),
            logical_parameters=sum(
                value[0] * value[1] for value in shapes.values()
            ),
        )

    attention = {
        str(index): source_inventory(index).as_dict()
        for index in attention_layers
    }
    tensor_receipts = {
        "model.embed_tokens.weight": {
            "dtype": "float32",
            "shape": [8, 4],
            "numel": 32,
            "tensor_sha256": "4" * 64,
        }
    }
    checkpoint_manifest = {
        "schema": "neural-forge-direct-parent-checkpoint-manifest-v1",
        "parent_model_identity_sha256": parent_sha,
        "config_file_sha256": "5" * 64,
        "index": None,
        "shards": [{"name": "model.safetensors"}],
    }
    body = {
        "schema": foundation.LFM2_SEMANTIC_FOUNDATION_PARENT_SCHEMA,
        "parent_model_directory": "C:/fixture/parent",
        "expected_parent_identity_sha256": parent_sha,
        "checkpoint_manifest": checkpoint_manifest,
        "checkpoint_manifest_sha256": canonical_sha256(checkpoint_manifest),
        "tensor_key_inventory_sha256": canonical_sha256(
            sorted(tensor_receipts)
        ),
        "full_tensor_inventory_sha256": canonical_sha256(tensor_receipts),
        "tensor_count": len(tensor_receipts),
        "tensor_receipts": tensor_receipts,
        "attention_layer_indices": list(attention_layers),
        "attention_qkvo_inventories": attention,
        "ffn_inventories": {},
        "all_checkpoint_tensors_hashed": True,
        "direct_local_safetensors": True,
        "parent_model_forward_calls": 0,
        "network_access_performed": False,
    }
    return _seal(body)


def _cache(tmp_path: Path, parent_sha: str, *, attention_layers=(1, 3)):
    layers = []
    for index in range(4):
        layers.append(
            {
                "layer_index": index,
                "contract": {
                    "kind": (
                        "full_attention"
                        if index in attention_layers
                        else "short_conv"
                    )
                },
            }
        )
    topology_body = {"layers": layers}
    topology = {
        **topology_body,
        "layer_topology_sha256": canonical_sha256(topology_body),
    }
    manifest_body = {
        "schema": "neural-forge-lfm2-circuit-trace-cache-v1",
        "cache_key": "6" * 64,
        "parent": {"identity_sha256": parent_sha},
        "layer_topology": topology,
        "runtime": {
            "network_access_performed": False,
            "local_loaded_parent_only": True,
        },
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": canonical_sha256(manifest_body),
    }
    return Lfm2CircuitTraceCache(
        path=tmp_path / "cache",
        _manifest=MappingProxyType(manifest),
        _completion=MappingProxyType({}),
    )


def _target(bridge, *, bridge_artifact_sha256=None):
    config = SimpleNamespace(
        model_type="lfm2",
        vocab_size=4,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        layer_types=("conv", "full_attention", "conv", "full_attention"),
    )
    setattr(
        config,
        tokenizer_bridge.TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
        bridge.artifact_sha256
        if bridge_artifact_sha256 is None
        else bridge_artifact_sha256,
    )
    setattr(
        config,
        tokenizer_bridge.TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY,
        bridge.profile_fingerprint,
    )
    setattr(
        config,
        tokenizer_bridge.PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY,
        bridge.donor_tokenizer_semantics_sha256,
    )
    setattr(
        config,
        tokenizer_bridge.CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY,
        bridge.target_tokenizer_semantics_sha256,
    )
    return SimpleNamespace(
        config=config,
        model=SimpleNamespace(
            layers=tuple(
                SimpleNamespace(is_attention_layer=index in {1, 3})
                for index in range(4)
            )
        ),
    )


def _fixture(tmp_path: Path):
    parent_sha = "7" * 64
    bridge = _bridge(tmp_path)
    parent = _parent_inventory(parent_sha)
    cache = _cache(tmp_path, parent_sha)
    target = _target(bridge)
    raw = foundation.build_lfm2_semantic_foundation_attention_lineage(
        parent_inventory=parent,
        target_model=target,
    )
    return raw, parent, cache, bridge, target


def test_normalizes_fresh_lineage_without_saved_foundation(tmp_path: Path):
    raw, parent, cache, bridge, target = _fixture(tmp_path)

    result = postfold.normalize_lfm2_semantic_foundation_attention_lineage(
        raw,
        parent_inventory=parent,
        cache=cache,
        bridge=bridge,
        target_model=target,
    )

    assert set(result) == {1, 3}
    for layer in result.values():
        report = layer.report
        assert report["schema"] == postfold.LFM2_SEMANTIC_COMPILER_LINEAGE_SCHEMA
        assert report["parent_inventory_sha256"] == parent["result_sha256"]
        assert report["bridge_artifact_sha256"] == bridge.artifact_sha256
        assert report["single_process_in_memory_normalization"] is True
        assert report["saved_foundation_checkpoint_required"] is False
        assert report["operation_audit"]["training_steps"] == 0
        assert report["operation_audit"]["parent_model_forward_calls"] == 0
        assert report["operation_audit"]["child_model_forward_calls"] == 0


def test_rejects_resealed_duplicate_query_head_coverage(tmp_path: Path):
    raw, parent, cache, bridge, target = _fixture(tmp_path)
    forged_all = dict(raw)
    forged = dict(forged_all["1"])
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
    forged_all["1"] = _seal(forged, "lineage_sha256")

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="internally inconsistent",
    ):
        postfold.normalize_lfm2_semantic_foundation_attention_lineage(
            forged_all,
            parent_inventory=parent,
            cache=cache,
            bridge=bridge,
            target_model=target,
        )


def test_rejects_cache_parent_and_target_bridge_drift(tmp_path: Path):
    raw, parent, cache, bridge, target = _fixture(tmp_path)
    wrong_cache = _cache(tmp_path, "8" * 64)
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="ancestry diverged",
    ):
        postfold.normalize_lfm2_semantic_foundation_attention_lineage(
            raw,
            parent_inventory=parent,
            cache=wrong_cache,
            bridge=bridge,
            target_model=target,
        )

    wrong_target = _target(bridge, bridge_artifact_sha256="9" * 64)
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="target, cache topology, and tokenizer bridge diverged",
    ):
        postfold.normalize_lfm2_semantic_foundation_attention_lineage(
            raw,
            parent_inventory=parent,
            cache=cache,
            bridge=bridge,
            target_model=wrong_target,
        )
