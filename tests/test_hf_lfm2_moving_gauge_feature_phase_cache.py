from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_phase_cache import DynamicFoldPhaseCache  # noqa: E402
from neural_forge.hf_lfm2_moving_gauge_boundary_phase_cache import (  # noqa: E402
    AuthenticatedLfm2MovingGaugeBoundaryPhase,
    Lfm2MovingGaugeBoundaryPhaseAuthority,
    load_or_extract_lfm2_moving_gauge_boundary_phase,
)
from neural_forge.hf_lfm2_moving_gauge_feature_phase_cache import (  # noqa: E402
    AuthenticatedLfm2MovingGaugeFeaturePhase,
    Lfm2MovingGaugeFeatureExtractionProgress,
    Lfm2MovingGaugeFeatureExtractionSources,
    Lfm2MovingGaugeFeatureImportRequest,
    Lfm2MovingGaugeFeaturePhaseAuthenticationError,
    Lfm2MovingGaugeFeaturePhaseAuthority,
    derive_lfm2_moving_gauge_writer_feature_bindings,
    lfm2_moving_gauge_feature_implementation_sha256,
    load_existing_lfm2_moving_gauge_feature_phase,
    load_or_extract_lfm2_moving_gauge_feature_phase,
)
from neural_forge import (  # noqa: E402
    hf_lfm2_moving_gauge_feature_phase_cache as feature_phase_module,
)
from tests import test_hf_lfm2_moving_gauge_boundary_phase_cache as boundary_test  # noqa: E402


_FEATURE_SECRET = b"writer-feature-test-secret-32-bytes"


def _layer_contract(
    *,
    residual_width: int = 2,
    context_width: int = 4,
    intermediate_width: int = 5,
):
    return {
        "schema": "neural-forge-lfm2-heterogeneous-layer-contract-v1",
        "implementation": "test.dynamic_lfm2",
        "model_type": "lfm2",
        "layer_index": 0,
        "kind": "short_conv",
        "hidden_size": residual_width,
        "ffn_intermediate_size": intermediate_width,
        "selection_evidence_sha256": boundary_test._sha("selection"),
        "physical_layout_descriptor_sha256": boundary_test._sha("layout"),
        "operator": {
            "operator": "short_conv",
            "external_hidden_size": residual_width,
            "internal_channel_width": context_width,
            "fir_length": 3,
            "bias": True,
            "input_projection_shape": [3 * context_width, residual_width],
            "input_projection_chunk_order": ["b", "c", "x"],
            "depthwise_kernel_shape": [context_width, 1, 3],
            "output_projection_shapes": [[3, context_width], [residual_width, 3]],
            "writer_topology": "factorized",
            "writer_rank": 3,
            "groups": context_width,
            "not_attention_qkv": True,
        },
        "structurally_validated": True,
        "sealed_heterogeneous_physical_layout": True,
    }


def _feature_tensors(*, context_width: int = 4, intermediate_width: int = 5):
    batches = []
    for batch in range(2):
        tensors = {}
        for phase_index, (phase, rows) in enumerate(
            (("prefill", 2), ("decode.000", 1))
        ):
            context = torch.arange(rows * context_width, dtype=torch.float32).reshape(
                1, rows, context_width
            )
            post = torch.arange(rows * intermediate_width, dtype=torch.float32).reshape(
                1, rows, intermediate_width
            )
            tensors[f"{phase}.layer.000.short_conv.context"] = (
                context + batch * 1000 + phase_index * 100
            )
            tensors[f"{phase}.layer.000.mlp.post_swiglu"] = (
                post + batch * 2000 + phase_index * 200
            )
        batches.append(tensors)
    return tuple(batches)


def _case(tmp_path):
    boundary_request, boundary_provider, _boundary_calls = boundary_test._case(
        tmp_path / "boundary"
    )
    boundary_cache = DynamicFoldPhaseCache(tmp_path / "boundary-phase-cache")
    boundary_authority = Lfm2MovingGaugeBoundaryPhaseAuthority(
        key_id=boundary_request.checkpoint_hmac_key_id,
        secret=b"a" * 32,
    )
    boundary = load_or_extract_lfm2_moving_gauge_boundary_phase(
        boundary_cache,
        boundary_request,
        authority=boundary_authority,
        checkpoint_root=tmp_path / "boundary-checkpoints",
        source_provider=boundary_provider,
    )
    assert isinstance(boundary, AuthenticatedLfm2MovingGaugeBoundaryPhase)
    boundary_sources = boundary_provider()
    child_cache = boundary_sources.child_cache
    contract = _layer_contract()
    child_cache.manifest["layer_topology"]["layers"][0]["contract"] = contract
    tensors = _feature_tensors()
    rows = tuple(dict(row) for row in boundary.row_index["rows"])

    def references():
        result = []
        for index, batch in enumerate(tensors):
            input_sha = next(
                row["child_input_sha256"]
                for row in rows
                if row["record_ordinal"] == index
            )
            result.append(
                boundary_test._Reference(
                    index=index,
                    input_sha256=input_sha,
                    record_sha256=boundary_test._sha(f"feature-record-{index}"),
                    ordered_tensor_names=tuple(sorted(batch)),
                    _inventory=MappingProxyType(
                        {
                            name: boundary_test._descriptor(value)
                            for name, value in batch.items()
                        }
                    ),
                )
            )
        return tuple(result)

    refs = references()
    bindings = derive_lfm2_moving_gauge_writer_feature_bindings(
        contract,
        source_dtype_by_field={
            "short_conv.context": "float32",
            "mlp.post_swiglu": "float32",
        },
    )
    request = Lfm2MovingGaugeFeatureImportRequest(
        child_cache_key=boundary_request.child_cache_key,
        child_manifest_sha256=boundary_request.provenance_sha256s["child_manifest"],
        child_completion_sha256=boundary_request.provenance_sha256s["child_completion"],
        child_identity_sha256=boundary_request.provenance_sha256s["child_identity"],
        child_state_dict_sha256=boundary_request.provenance_sha256s["child_state_dict"],
        child_topology_sha256=boundary_request.provenance_sha256s["child_topology"],
        child_selective_session_sha256=boundary_request.provenance_sha256s[
            "child_selective_session"
        ],
        child_layer_contract_sha256=canonical_sha256(contract),
        boundary_phase_key_sha256=boundary.entry.phase_key.phase_key_sha256,
        boundary_completion_sha256=boundary.entry.completion_sha256,
        boundary_request_sha256=boundary.request.request_sha256,
        boundary_receipt_sha256=boundary.receipt_sha256,
        boundary_row_index_sha256=boundary.row_index["row_index_sha256"],
        row_alignment_sha256=boundary.row_alignment_sha256,
        split_identity_sha256=boundary.split_identity_sha256,
        expected_batch_count=2,
        expected_aligned_row_count=6,
        logical_layer_index=0,
        child_layer_index=0,
        feature_bindings=bindings,
        implementation_sha256=lfm2_moving_gauge_feature_implementation_sha256(),
        checkpoint_hmac_key_id="writer-feature-test",
        shard_batch_count=1,
    )
    calls = {"provider": 0, "session": 0}

    def session_factory():
        calls["session"] += 1
        return boundary_test._Session(
            request.child_cache_key,
            request.child_selective_session_sha256,
            refs,
            tensors,
        )

    def provider():
        calls["provider"] += 1
        return Lfm2MovingGaugeFeatureExtractionSources(
            boundary_phase=boundary,
            child_cache=child_cache,
            child_session_factory=session_factory,
        )

    return request, provider, calls, boundary, child_cache, refs, tensors


def _authority(request):
    return Lfm2MovingGaugeFeaturePhaseAuthority(
        key_id=request.checkpoint_hmac_key_id,
        secret=_FEATURE_SECRET,
    )


def _reseal_checkpoint(checkpoint, authority):
    body = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"checkpoint_sha256", "checkpoint_hmac_sha256"}
    }
    sealed = {**body, "checkpoint_sha256": canonical_sha256(body)}
    return {
        **sealed,
        "checkpoint_hmac_sha256": authority.sign_checkpoint(sealed),
    }


def test_variable_width_contract_uses_writer_geometry_not_residual_width() -> None:
    contract = _layer_contract(
        residual_width=816,
        context_width=1024,
        intermediate_width=321,
    )
    bindings = derive_lfm2_moving_gauge_writer_feature_bindings(
        contract,
        source_dtype_by_field={
            "short_conv.context": "float32",
            "mlp.post_swiglu": "float32",
        },
    )
    assert bindings[0].feature_width == 1024
    assert bindings[0].writer_weight_shapes == ((3, 1024), (816, 3))
    assert bindings[1].feature_width == 321
    assert bindings[1].writer_weight_shapes == ((816, 321),)
    assert bindings[0].feature_width != contract["hidden_size"]
    assert bindings[1].feature_width != contract["hidden_size"]


@pytest.mark.parametrize("nonliteral", [True, "1024"])
def test_writer_topology_rejects_nonliteral_shape_entries(nonliteral) -> None:
    contract = _layer_contract()
    contract["operator"]["output_projection_shapes"][0][1] = nonliteral
    with pytest.raises(
        Lfm2MovingGaugeFeaturePhaseAuthenticationError,
        match="shapes are malformed",
    ):
        derive_lfm2_moving_gauge_writer_feature_bindings(
            contract,
            source_dtype_by_field={
                "short_conv.context": "float32",
                "mlp.post_swiglu": "float32",
            },
        )


def test_child_only_feature_phase_resumes_reuses_boundary_and_hits_before_provider(
    tmp_path,
) -> None:
    request, provider, calls, boundary, _cache, _refs, _tensors = _case(tmp_path)
    cache = DynamicFoldPhaseCache(tmp_path / "feature-phase-cache")
    checkpoint_root = tmp_path / "feature-checkpoints"
    authority = _authority(request)
    partial = load_or_extract_lfm2_moving_gauge_feature_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=checkpoint_root,
        source_provider=provider,
        maximum_new_shards=1,
    )
    assert isinstance(partial, Lfm2MovingGaugeFeatureExtractionProgress)
    assert (partial.completed_shards, partial.total_shards) == (1, 2)
    completed = load_or_extract_lfm2_moving_gauge_feature_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=checkpoint_root,
        source_provider=provider,
    )
    assert isinstance(completed, AuthenticatedLfm2MovingGaugeFeaturePhase)
    assert calls == {"provider": 2, "session": 2}
    assert completed.verify_authentication()
    assert completed.row_index == boundary.row_index
    assert completed.row_index["row_index_sha256"] == request.boundary_row_index_sha256
    assert completed.receipt["boundary_row_index_reused_exactly"] is True
    assert (
        completed.receipt["tensor_bytes_authenticated_lazily_on_field_access"] is True
    )
    assert completed.receipt["execution"]["child_model_forward_calls"] == 0
    assert completed.receipt["execution"]["training_steps"] == 0
    context = completed.feature_rows("short_conv.context", split="fit")
    post = completed.feature_rows("mlp.post_swiglu", split="fit")
    assert context.rows.shape == (3, 4)
    assert post.rows.shape == (3, 5)
    assert context.rows.dtype == post.rows.dtype == torch.float32
    assert context.global_row_indices == post.global_row_indices == (0, 1, 2)
    assert context.feature_width != boundary.request.child_width
    assert post.feature_width != boundary.request.child_width
    assert float(context.weights.sum()) == pytest.approx(1.0)

    def forbidden_provider():
        raise AssertionError("feature phase hit touched its source provider")

    hit = load_or_extract_lfm2_moving_gauge_feature_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=tmp_path / "unused",
        source_provider=forbidden_provider,
    )
    assert isinstance(hit, AuthenticatedLfm2MovingGaugeFeaturePhase)
    assert hit.receipt_sha256 == completed.receipt_sha256


def test_selected_feature_tensor_must_byte_match_sealed_descriptor(tmp_path) -> None:
    request, provider, _calls, _boundary, _cache, _refs, _tensors = _case(tmp_path)
    sources = provider()

    def poisoned_provider():
        return replace(
            sources,
            child_session_factory=lambda: boundary_test._PoisoningSession(
                sources.child_session_factory()
            ),
        )

    checkpoint_root = tmp_path / "poisoned-checkpoints"
    with pytest.raises(
        Lfm2MovingGaugeFeaturePhaseAuthenticationError,
        match="sealed bytes",
    ):
        load_or_extract_lfm2_moving_gauge_feature_phase(
            DynamicFoldPhaseCache(tmp_path / "poisoned-cache"),
            request,
            authority=_authority(request),
            checkpoint_root=checkpoint_root,
            source_provider=poisoned_provider,
        )
    checkpoint_dir = (
        checkpoint_root / request.phase_key.phase_key_sha256 / "checkpoints"
    )
    assert not checkpoint_dir.exists() or not tuple(checkpoint_dir.iterdir())


def test_unverified_source_session_never_admits_checkpoint(tmp_path) -> None:
    request, provider, _calls, _boundary, _cache, _refs, _tensors = _case(tmp_path)
    sources = provider()

    def rejecting_provider():
        return replace(
            sources,
            child_session_factory=lambda: boundary_test._RejectingSession(
                sources.child_session_factory()
            ),
        )

    checkpoint_root = tmp_path / "rejected-checkpoints"
    with pytest.raises(RuntimeError, match="synthetic selective-session"):
        load_or_extract_lfm2_moving_gauge_feature_phase(
            DynamicFoldPhaseCache(tmp_path / "rejected-cache"),
            request,
            authority=_authority(request),
            checkpoint_root=checkpoint_root,
            source_provider=rejecting_provider,
        )
    work_root = checkpoint_root / request.phase_key.phase_key_sha256
    assert not (work_root / "checkpoints").exists()
    assert not (work_root / "shards").exists() or not tuple(
        (work_root / "shards").iterdir()
    )


def test_each_verified_feature_shard_is_durable_before_later_failure(tmp_path) -> None:
    request, provider, _calls, _boundary, _cache, _refs, _tensors = _case(tmp_path)
    valid_sources = provider()
    failing_sources = replace(
        valid_sources,
        child_session_factory=lambda: boundary_test._RejectAfterSuccessfulVerifications(
            valid_sources.child_session_factory(),
            successful_verifications=1,
        ),
    )
    cache = DynamicFoldPhaseCache(tmp_path / "durable-feature-cache")
    checkpoint_root = tmp_path / "durable-feature-checkpoints"
    authority = _authority(request)

    with pytest.raises(RuntimeError, match="after durable shard verification"):
        load_or_extract_lfm2_moving_gauge_feature_phase(
            cache,
            request,
            authority=authority,
            checkpoint_root=checkpoint_root,
            source_provider=lambda: failing_sources,
        )

    work_root = checkpoint_root / request.phase_key.phase_key_sha256
    checkpoint_paths = tuple(sorted((work_root / "checkpoints").glob("*.json")))
    assert tuple(path.name for path in checkpoint_paths) == ("checkpoint-0000.json",)
    admitted_checkpoint_bytes = checkpoint_paths[0].read_bytes()
    admitted = json.loads(admitted_checkpoint_bytes)
    assert admitted["checkpoint_ordinal"] == 0
    admitted_shard_path = work_root / "shards" / admitted["shard"]["filename"]
    admitted_shard_bytes = admitted_shard_path.read_bytes()
    assert len(tuple((work_root / "shards").glob("*.safetensors"))) == 1

    resumed = load_or_extract_lfm2_moving_gauge_feature_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=checkpoint_root,
        source_provider=provider,
    )
    assert isinstance(resumed, AuthenticatedLfm2MovingGaugeFeaturePhase)
    assert resumed.verify_authentication()
    assert checkpoint_paths[0].read_bytes() == admitted_checkpoint_bytes
    assert admitted_shard_path.read_bytes() == admitted_shard_bytes
    assert resumed.receipt["ordered_checkpoints"][0]["checkpoint_hmac_sha256"] == (
        admitted["checkpoint_hmac_sha256"]
    )


def test_wrong_writer_topology_rejected_before_selective_session_open(tmp_path) -> None:
    request, provider, calls, _boundary, _cache, _refs, _tensors = _case(tmp_path)
    wrong = replace(
        request.feature_bindings[0],
        feature_width=3,
        writer_weight_shapes=((3, 3), (2, 3)),
        binding_sha256="",
    )
    altered = replace(
        request,
        feature_bindings=(wrong, request.feature_bindings[1]),
        request_sha256="",
    )
    with pytest.raises(
        Lfm2MovingGaugeFeaturePhaseAuthenticationError,
        match="widths or writer weight topology",
    ):
        load_or_extract_lfm2_moving_gauge_feature_phase(
            DynamicFoldPhaseCache(tmp_path / "wrong-topology-cache"),
            altered,
            authority=_authority(altered),
            checkpoint_root=tmp_path / "wrong-topology-checkpoints",
            source_provider=provider,
        )
    assert calls["session"] == 0


def test_checkpoint_hmac_and_source_row_claims_fail_closed(tmp_path) -> None:
    request, provider, _calls, _boundary, _cache, _refs, _tensors = _case(tmp_path)
    cache = DynamicFoldPhaseCache(tmp_path / "partial-cache")
    checkpoint_root = tmp_path / "partial-checkpoints"
    authority = _authority(request)
    partial = load_or_extract_lfm2_moving_gauge_feature_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=checkpoint_root,
        source_provider=provider,
        maximum_new_shards=1,
    )
    assert isinstance(partial, Lfm2MovingGaugeFeatureExtractionProgress)
    checkpoint_path = (
        checkpoint_root
        / request.phase_key.phase_key_sha256
        / "checkpoints"
        / "checkpoint-0000.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["source_records"][0]["alignment_sha256"] = boundary_test._sha(
        "fabricated-alignment"
    )
    checkpoint = _reseal_checkpoint(checkpoint, authority)
    checkpoint_path.write_bytes(canonical_json_bytes(checkpoint) + b"\n")
    with pytest.raises(
        Lfm2MovingGaugeFeaturePhaseAuthenticationError,
        match="source record differs",
    ):
        load_or_extract_lfm2_moving_gauge_feature_phase(
            cache,
            request,
            authority=authority,
            checkpoint_root=checkpoint_root,
            source_provider=provider,
        )


@pytest.mark.parametrize("source_field", ["record_sha256", "selected_payload_sha256"])
def test_resumed_prefix_rebinds_source_records_and_tensor_descriptors(
    tmp_path,
    source_field,
) -> None:
    request, provider, _calls, _boundary, _cache, _refs, _tensors = _case(tmp_path)
    cache = DynamicFoldPhaseCache(tmp_path / "rebound-cache")
    checkpoint_root = tmp_path / "rebound-checkpoints"
    authority = _authority(request)
    partial = load_or_extract_lfm2_moving_gauge_feature_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=checkpoint_root,
        source_provider=provider,
        maximum_new_shards=1,
    )
    assert isinstance(partial, Lfm2MovingGaugeFeatureExtractionProgress)
    checkpoint_path = (
        checkpoint_root
        / request.phase_key.phase_key_sha256
        / "checkpoints"
        / "checkpoint-0000.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["source_records"][0][source_field] = boundary_test._sha(
        f"forged-{source_field}"
    )
    checkpoint = _reseal_checkpoint(checkpoint, authority)
    checkpoint_path.write_bytes(canonical_json_bytes(checkpoint) + b"\n")

    with pytest.raises(
        Lfm2MovingGaugeFeaturePhaseAuthenticationError,
        match="live selective source record or tensor descriptors",
    ):
        load_or_extract_lfm2_moving_gauge_feature_phase(
            cache,
            request,
            authority=authority,
            checkpoint_root=checkpoint_root,
            source_provider=provider,
        )
    assert cache.lookup(request.phase_key) is None
    assert not (
        checkpoint_root
        / request.phase_key.phase_key_sha256
        / "checkpoints"
        / "checkpoint-0001.json"
    ).exists()


def test_malformed_verified_session_cannot_poison_cache_and_valid_retry_succeeds(
    tmp_path,
) -> None:
    request, provider, _calls, boundary, child_cache, refs, tensors = _case(tmp_path)
    malformed_refs = (replace(refs[0], record_sha256="not-a-digest"), *refs[1:])

    def malformed_provider():
        return Lfm2MovingGaugeFeatureExtractionSources(
            boundary_phase=boundary,
            child_cache=child_cache,
            child_session_factory=lambda: boundary_test._Session(
                request.child_cache_key,
                request.child_selective_session_sha256,
                malformed_refs,
                tensors,
            ),
        )

    cache = DynamicFoldPhaseCache(tmp_path / "malformed-cache")
    checkpoint_root = tmp_path / "malformed-checkpoints"
    with pytest.raises(
        Lfm2MovingGaugeFeaturePhaseAuthenticationError,
        match="source records are invalid",
    ):
        load_or_extract_lfm2_moving_gauge_feature_phase(
            cache,
            request,
            authority=_authority(request),
            checkpoint_root=checkpoint_root,
            source_provider=malformed_provider,
        )
    assert cache.lookup(request.phase_key) is None
    work_root = checkpoint_root / request.phase_key.phase_key_sha256
    assert not (work_root / "checkpoints").exists()
    assert not (work_root / "shards").exists() or not tuple(
        (work_root / "shards").iterdir()
    )

    completed = load_or_extract_lfm2_moving_gauge_feature_phase(
        cache,
        request,
        authority=_authority(request),
        checkpoint_root=checkpoint_root,
        source_provider=provider,
    )
    assert isinstance(completed, AuthenticatedLfm2MovingGaugeFeaturePhase)
    assert completed.verify_authentication()


def test_malformed_verified_session_reference_index_cannot_complete_cache(
    tmp_path,
) -> None:
    request, _provider, _calls, boundary, child_cache, refs, tensors = _case(tmp_path)
    malformed_refs = (replace(refs[0], index=1), *refs[1:])

    def malformed_provider():
        return Lfm2MovingGaugeFeatureExtractionSources(
            boundary_phase=boundary,
            child_cache=child_cache,
            child_session_factory=lambda: boundary_test._Session(
                request.child_cache_key,
                request.child_selective_session_sha256,
                malformed_refs,
                tensors,
            ),
        )

    cache = DynamicFoldPhaseCache(tmp_path / "malformed-index-cache")
    checkpoint_root = tmp_path / "malformed-index-checkpoints"
    with pytest.raises(
        Lfm2MovingGaugeFeaturePhaseAuthenticationError,
        match="wrong batch index",
    ):
        load_or_extract_lfm2_moving_gauge_feature_phase(
            cache,
            request,
            authority=_authority(request),
            checkpoint_root=checkpoint_root,
            source_provider=malformed_provider,
        )
    assert cache.lookup(request.phase_key) is None
    work_root = checkpoint_root / request.phase_key.phase_key_sha256
    assert not (work_root / "checkpoints").exists()
    assert not (work_root / "shards").exists()


def test_staged_checkpoint_chain_is_authenticated_before_cache_completion(
    tmp_path,
    monkeypatch,
) -> None:
    request, provider, _calls, _boundary, _cache, _refs, _tensors = _case(tmp_path)
    cache = DynamicFoldPhaseCache(tmp_path / "staged-cache")
    checkpoint_root = tmp_path / "staged-checkpoints"
    original_copyfile = feature_phase_module.shutil.copyfile
    corrupted = False

    def corrupt_staged_checkpoint(source, destination):
        nonlocal corrupted
        result = original_copyfile(source, destination)
        destination_path = Path(destination)
        if not corrupted and destination_path.parent.name == "checkpoints":
            checkpoint = json.loads(destination_path.read_text(encoding="utf-8"))
            checkpoint["source_records"][0]["record_sha256"] = "not-a-digest"
            destination_path.write_bytes(canonical_json_bytes(checkpoint) + b"\n")
            corrupted = True
        return result

    monkeypatch.setattr(
        feature_phase_module.shutil,
        "copyfile",
        corrupt_staged_checkpoint,
    )
    with pytest.raises(
        Lfm2MovingGaugeFeaturePhaseAuthenticationError,
        match="Feature checkpoint 0 differs from its request",
    ):
        load_or_extract_lfm2_moving_gauge_feature_phase(
            cache,
            request,
            authority=_authority(request),
            checkpoint_root=checkpoint_root,
            source_provider=provider,
        )
    assert corrupted is True
    assert cache.lookup(request.phase_key) is None


def test_complete_phase_rejects_wrong_hmac_authority(tmp_path) -> None:
    request, provider, _calls, _boundary, _cache, _refs, _tensors = _case(tmp_path)
    cache = DynamicFoldPhaseCache(tmp_path / "complete-cache")
    completed = load_or_extract_lfm2_moving_gauge_feature_phase(
        cache,
        request,
        authority=_authority(request),
        checkpoint_root=tmp_path / "complete-checkpoints",
        source_provider=provider,
    )
    assert isinstance(completed, AuthenticatedLfm2MovingGaugeFeaturePhase)
    wrong = Lfm2MovingGaugeFeaturePhaseAuthority(
        key_id=request.checkpoint_hmac_key_id,
        secret=b"different-writer-feature-secret-32",
    )
    with pytest.raises(
        Lfm2MovingGaugeFeaturePhaseAuthenticationError,
        match="HMAC",
    ):
        load_existing_lfm2_moving_gauge_feature_phase(
            cache,
            request,
            authority=wrong,
        )


def test_lazy_field_access_rejects_changed_tensor_bytes(tmp_path) -> None:
    request, provider, _calls, _boundary, _cache, _refs, _tensors = _case(tmp_path)
    completed = load_or_extract_lfm2_moving_gauge_feature_phase(
        DynamicFoldPhaseCache(tmp_path / "lazy-cache"),
        request,
        authority=_authority(request),
        checkpoint_root=tmp_path / "lazy-checkpoints",
        source_provider=provider,
    )
    assert isinstance(completed, AuthenticatedLfm2MovingGaugeFeaturePhase)
    summary = completed.receipt["ordered_checkpoints"][0]
    shard_path = completed.entry.payload_path / "shards" / summary["shard"]["filename"]
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(shard_path), device="cpu")
    key = "layer.000.short_conv.context"
    tensors[key] = tensors[key] + 1
    save_file(
        tensors,
        str(shard_path),
        metadata={
            "schema": "neural-forge-lfm2-moving-gauge-writer-feature-shard-v1",
            "request_sha256": request.request_sha256,
            "checkpoint_ordinal": "0",
        },
    )
    with pytest.raises(
        Lfm2MovingGaugeFeaturePhaseAuthenticationError,
        match="tensor .* changed",
    ):
        completed.feature_rows("short_conv.context", split="fit")
