from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache  # noqa: E402
import neural_forge.hf_lfm2_cpcf_model_compiler as model_compiler  # noqa: E402
import neural_forge.hf_lfm2_cpcf_selective_evidence as selective  # noqa: E402
import neural_forge.hf_lfm2_cpcf_v5_model_compiler as v5  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_json_bytes  # noqa: E402
from neural_forge.dynamic_evidence_catalog_artifact import (  # noqa: E402
    CATALOG_FILENAME,
    DynamicEvidenceCatalogArtifactIntegrityError,
)
from neural_forge.dynamic_evidence_residency import (  # noqa: E402
    DynamicEvidenceLayerReference,
    DynamicEvidenceResidencyError,
    DynamicEvidenceResidencyWindow,
)
from neural_forge.dynamic_fold_executor import _FoldOnlyTorchGuard  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    BATCH_RECORD_FILENAME,
    TRACE_FILENAME,
    Lfm2CircuitTraceCacheError,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceCorruptionError,
    build_lfm2_circuit_trace_cache,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_cpcf_installability import (  # noqa: E402
    iter_lfm2_cpcf_compiler_inputs,
)
from neural_forge.hf_lfm2_cpcf_selective_evidence import (  # noqa: E402
    Lfm2CpcfSelectiveEvidenceError,
    build_lfm2_cpcf_selective_evidence_provider,
    lfm2_cpcf_selective_catalog_from_artifact_payload,
    open_lfm2_cpcf_selective_evidence_provider_from_catalog,
    open_or_build_lfm2_cpcf_selective_evidence_provider,
)


def _parent() -> Any:
    config = transformers.Lfm2Config(
        vocab_size=31,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(909)
        return transformers.Lfm2ForCausalLM(config)


@pytest.fixture(scope="module")
def sealed_cache_path(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("selective-lfm2-cache")
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "forced_decode_ids": torch.tensor([[5]], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[1, 4, 3, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "forced_decode_ids": torch.tensor([[6]], dtype=torch.long),
        },
    ]
    cache = build_lfm2_circuit_trace_cache(
        root,
        _parent(),
        batches,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=1,
            capture_causal_circuit_evidence=True,
        ),
        input_device="cpu",
    )
    return cache.path


def _contains_tensor(value: Any) -> bool:
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def _assert_same_tensors(
    left: dict[str, Any] | Any,
    right: dict[str, Any] | Any,
) -> None:
    assert tuple(left) == tuple(right)
    for name in left:
        assert left[name].dtype == right[name].dtype
        assert tuple(left[name].shape) == tuple(right[name].shape)
        assert torch.equal(left[name], right[name])


def test_selective_provider_matches_every_full_phase_layer_and_final_norm(
    sealed_cache_path,
) -> None:
    full_phases = tuple(
        iter_lfm2_cpcf_compiler_inputs(
            open_lfm2_circuit_trace_cache(
                sealed_cache_path,
                verify=False,
            )
        )
    )
    provider = build_lfm2_cpcf_selective_evidence_provider(
        open_lfm2_circuit_trace_cache(
            sealed_cache_path,
            verify=False,
        )
    )
    try:
        catalog = provider.catalog
        assert catalog.layer_count == 2
        assert catalog.phase_count == len(full_phases) == 4
        assert _contains_tensor(catalog.as_dict()) is False
        assert [
            reference.existing_phase_evidence_sha256
            for reference in catalog.phase_references
        ] == [phase.evidence_sha256 for phase in full_phases]

        for dynamic_reference in provider.dynamic_catalog.references:
            payload = provider.load_layer_payload(dynamic_reference)
            try:
                receipt = provider.authenticate_layer_payload(
                    dynamic_reference,
                    payload,
                )
                assert receipt.evidence_sha256 == dynamic_reference.evidence_sha256
                assert receipt.logical_bytes == dynamic_reference.logical_bytes
                assert len(payload.phases) == len(full_phases)
                expected_bytes = 0
                for phase_index, phase_payload in enumerate(payload.phases):
                    actual = phase_payload.evidence
                    expected = full_phases[phase_index].layers[
                        dynamic_reference.layer_index
                    ]
                    assert actual.evidence_sha256 == expected.evidence_sha256
                    assert actual.operator_kind == expected.operator_kind
                    _assert_same_tensors(actual.tensors, expected.tensors)
                    expected_bytes += sum(
                        int(value.numel() * value.element_size())
                        for value in expected.tensors.values()
                    )
                assert payload.logical_bytes == expected_bytes
            finally:
                provider.release_layer_payload(
                    dynamic_reference,
                    payload,
                )
            assert payload.released is True
            with pytest.raises(
                Lfm2CpcfSelectiveEvidenceError,
                match="released",
            ):
                _ = payload.phases

        final_norm = provider.load_final_norm_payload()
        try:
            assert (
                provider.authenticate_final_norm_payload(final_norm)
                == catalog.final_norm_evidence_sha256
            )
            assert len(final_norm.phases) == len(full_phases)
            for phase_index, phase_payload in enumerate(final_norm.phases):
                _assert_same_tensors(
                    phase_payload.tensors,
                    full_phases[phase_index].final_norm_tensors,
                )
        finally:
            provider.release_final_norm_payload(final_norm)
        assert final_norm.released is True

        maximum_bytes = sum(
            reference.logical_bytes
            for reference in provider.dynamic_catalog.references
        )
        with DynamicEvidenceResidencyWindow(
            provider.dynamic_catalog,
            provider,
            maximum_resident_bytes=maximum_bytes,
        ) as window:
            current = window.start()
            assert current.layer_index == 0
            assert window.wait_for_prefetch() == 1
            current = window.advance(completed_layer_index=0)
            assert current is not None and current.layer_index == 1
            assert window.advance(completed_layer_index=1) is None
            assert window.peak_occupied_layers == 2
            assert window.peak_resident_bytes == maximum_bytes
    finally:
        provider.close()


def test_selective_session_hashes_each_trace_once_and_never_rehashes_per_layer(
    sealed_cache_path,
    monkeypatch,
) -> None:
    trace_hashes: list[Path] = []
    original = trace_cache._file_sha256

    def counted(path: Path) -> str:
        if Path(path).name == TRACE_FILENAME:
            trace_hashes.append(Path(path))
        return original(path)

    monkeypatch.setattr(trace_cache, "_file_sha256", counted)
    provider = build_lfm2_cpcf_selective_evidence_provider(
        open_lfm2_circuit_trace_cache(
            sealed_cache_path,
            verify=False,
        )
    )
    try:
        assert len(trace_hashes) == 2
        for reference in provider.dynamic_catalog.references:
            payload = provider.load_layer_payload(reference)
            provider.authenticate_layer_payload(reference, payload)
            provider.release_layer_payload(reference, payload)
        final_norm = provider.load_final_norm_payload()
        provider.authenticate_final_norm_payload(final_norm)
        provider.release_final_norm_payload(final_norm)
        provider._session.verify()
        assert len(trace_hashes) == 2
        evidence = provider.selective_session_evidence
        assert evidence["authentication_contract"][
            "trace_file_sha256_passes_per_session"
        ] == 1
        assert evidence["authentication_contract"][
            "unselected_tensor_payloads_loaded"
        ] is False
        assert "safe_open path/stat interval" in evidence[
            "authentication_contract"
        ]["toctou_boundary"]
    finally:
        provider.close()


def test_catalog_artifact_cold_then_warm_skips_only_catalog_tensor_sweep(
    sealed_cache_path,
    tmp_path,
    monkeypatch,
) -> None:
    store_root = tmp_path / "selective-catalogs"
    trace_hashes: list[Path] = []
    original_file_sha256 = trace_cache._file_sha256
    original_layer_builder = (
        selective.Lfm2CpcfSelectiveEvidenceProvider._build_layer_payload
    )
    original_final_builder = (
        selective.Lfm2CpcfSelectiveEvidenceProvider._build_final_norm_payload
    )
    layer_builds: list[int] = []
    final_builds = 0

    def counted_file_sha256(path: Path) -> str:
        if Path(path).name == TRACE_FILENAME:
            trace_hashes.append(Path(path))
        return original_file_sha256(path)

    def counted_layer_builder(self, layer_index):
        layer_builds.append(layer_index)
        return original_layer_builder(self, layer_index)

    def counted_final_builder(self, *, expected_last_shapes):
        nonlocal final_builds
        final_builds += 1
        return original_final_builder(
            self,
            expected_last_shapes=expected_last_shapes,
        )

    monkeypatch.setattr(trace_cache, "_file_sha256", counted_file_sha256)
    monkeypatch.setattr(
        selective.Lfm2CpcfSelectiveEvidenceProvider,
        "_build_layer_payload",
        counted_layer_builder,
    )
    monkeypatch.setattr(
        selective.Lfm2CpcfSelectiveEvidenceProvider,
        "_build_final_norm_payload",
        counted_final_builder,
    )
    with _FoldOnlyTorchGuard() as cold_guard:
        cold = open_or_build_lfm2_cpcf_selective_evidence_provider(
            open_lfm2_circuit_trace_cache(
                sealed_cache_path,
                verify=False,
            ),
            artifact_store=store_root,
        )
        cold_catalog = cold.catalog.as_dict()
        assert cold.catalog_artifact_receipt is not None
        assert cold.catalog_artifact_receipt["disposition"] == "published"
        cold.close()
    assert layer_builds == [0, 1]
    assert final_builds == 1
    assert len(trace_hashes) == 2
    assert cold_guard.optimizer_constructions == 0
    assert cold_guard.optimizer_steps == 0
    assert cold_guard.backward_calls == 0
    assert cold_guard.autograd_grad_calls == 0
    assert cold_guard.pruning_operations == 0
    assert cold_guard.quantization_operations == 0

    def forbidden_catalog_sweep(*args, **kwargs):
        del args, kwargs
        raise AssertionError("warm catalog hit performed a tensor sweep")

    trace_hashes.clear()
    monkeypatch.setattr(
        selective.Lfm2CpcfSelectiveEvidenceProvider,
        "_build_layer_payload",
        forbidden_catalog_sweep,
    )
    monkeypatch.setattr(
        selective.Lfm2CpcfSelectiveEvidenceProvider,
        "_build_final_norm_payload",
        forbidden_catalog_sweep,
    )
    with _FoldOnlyTorchGuard() as warm_guard:
        warm = open_or_build_lfm2_cpcf_selective_evidence_provider(
            open_lfm2_circuit_trace_cache(
                sealed_cache_path,
                verify=False,
            ),
            artifact_store=store_root,
        )
        assert warm.catalog.as_dict() == cold_catalog
        assert warm.catalog_artifact_receipt is not None
        assert warm.catalog_artifact_receipt["disposition"] == "loaded"
        assert len(trace_hashes) == 2

        monkeypatch.setattr(
            selective.Lfm2CpcfSelectiveEvidenceProvider,
            "_build_layer_payload",
            original_layer_builder,
        )
        monkeypatch.setattr(
            selective.Lfm2CpcfSelectiveEvidenceProvider,
            "_build_final_norm_payload",
            original_final_builder,
        )
        reference = warm.dynamic_catalog.references[0]
        payload = warm.load_layer_payload(reference)
        try:
            receipt = warm.authenticate_layer_payload(reference, payload)
            assert receipt.evidence_sha256 == reference.evidence_sha256
        finally:
            warm.release_layer_payload(reference, payload)
        warm.close()
    assert len(trace_hashes) == 2
    assert warm_guard.optimizer_constructions == 0
    assert warm_guard.optimizer_steps == 0
    assert warm_guard.backward_calls == 0
    assert warm_guard.autograd_grad_calls == 0
    assert warm_guard.pruning_operations == 0
    assert warm_guard.quantization_operations == 0


def test_production_warm_open_requires_exact_existing_catalog(
    sealed_cache_path,
    tmp_path,
    monkeypatch,
) -> None:
    store_root = tmp_path / "selective-catalogs"
    cold = open_or_build_lfm2_cpcf_selective_evidence_provider(
        open_lfm2_circuit_trace_cache(
            sealed_cache_path,
            verify=False,
        ),
        artifact_store=store_root,
    )
    assert cold.catalog_artifact is not None
    binding_sha256 = cold.catalog_artifact.binding.binding_sha256
    manifest_sha256 = cold.catalog_artifact.manifest_sha256
    catalog_sha256 = cold.catalog_artifact.catalog_sha256
    expected_catalog = cold.catalog.as_dict()
    cold.close()

    def forbidden_catalog_sweep(*args, **kwargs):
        del args, kwargs
        raise AssertionError("production warm open ran the cold builder")

    monkeypatch.setattr(
        selective,
        "_build_catalog_from_authenticated_source",
        forbidden_catalog_sweep,
    )
    warm = open_lfm2_cpcf_selective_evidence_provider_from_catalog(
        open_lfm2_circuit_trace_cache(
            sealed_cache_path,
            verify=False,
        ),
        artifact_store=store_root,
        expected_binding_sha256=binding_sha256,
        expected_manifest_sha256=manifest_sha256,
        expected_catalog_sha256=catalog_sha256,
    )
    try:
        assert warm.catalog.as_dict() == expected_catalog
        assert warm.catalog_artifact_receipt is not None
        assert warm.catalog_artifact_receipt["disposition"] == "loaded"
    finally:
        warm.close()

    with pytest.raises(
        Lfm2CpcfSelectiveEvidenceError,
        match="required catalog binding",
    ):
        open_lfm2_cpcf_selective_evidence_provider_from_catalog(
            open_lfm2_circuit_trace_cache(
                sealed_cache_path,
                verify=False,
            ),
            artifact_store=store_root,
            expected_binding_sha256="0" * 64,
            expected_manifest_sha256=manifest_sha256,
            expected_catalog_sha256=catalog_sha256,
        )

    with pytest.raises(
        Lfm2CpcfSelectiveEvidenceError,
        match="not prepared",
    ):
        open_lfm2_cpcf_selective_evidence_provider_from_catalog(
            open_lfm2_circuit_trace_cache(
                sealed_cache_path,
                verify=False,
            ),
            artifact_store=tmp_path / "empty-store",
            expected_binding_sha256=binding_sha256,
            expected_manifest_sha256=manifest_sha256,
            expected_catalog_sha256=catalog_sha256,
        )


def test_runner_prepare_only_creates_one_authenticated_pre_cuda_catalog(
    sealed_cache_path: Path,
    tmp_path: Path,
) -> None:
    from experiments import (
        run_lfm2_cpcf_v5_selective_blank_fold as runner,
    )

    for name in (
        "parent",
        "donor-tokenizer",
        "target-tokenizer",
        "bridge",
    ):
        (tmp_path / name).mkdir()
    trajectory_root = tmp_path / "trajectory-input"
    trajectory_root.mkdir()
    trajectory = trajectory_root / "trajectories.json"
    trajectory.write_text('{"fixture":"unused-by-preparation"}\n', "ascii")
    store = tmp_path / "catalog-store"
    receipt_path = tmp_path / "catalog-preparation.json"
    diagnostic = tmp_path / "work" / "failure-diagnostic.json"
    args = runner.build_parser().parse_args(
        [
            "--parent-model-directory",
            str(tmp_path / "parent"),
            "--donor-tokenizer-directory",
            str(tmp_path / "donor-tokenizer"),
            "--target-tokenizer-directory",
            str(tmp_path / "target-tokenizer"),
            "--bridge-directory",
            str(tmp_path / "bridge"),
            "--trace-cache-directory",
            str(sealed_cache_path),
            "--trajectory-corpus",
            str(trajectory),
            "--diagnostic-json",
            str(diagnostic),
            "--compression-ratio",
            "4",
            "--maximum-process-rss-gib",
            "24",
            "--selective-catalog-store-directory",
            str(store),
            "--selective-catalog-receipt-json",
            str(receipt_path),
            "--prepare-selective-catalog-only",
        ]
    )
    config = runner.config_from_args(args)

    preparation = dict(runner.prepare_selective_catalog(config))

    assert preparation["parent_model_loaded"] is False
    assert preparation["blank_target_allocated"] is False
    assert preparation["cuda_initialized"] is False
    assert preparation["tensor_payloads_resident"] is False
    assert preparation["warm_hit_required_for_production_fold"] is True
    assert (
        preparation["cold_rebuild_permitted_in_production_fold"] is False
    )
    assert all(
        count == 0
        for count in preparation["operation_audit"].values()
    )
    assert receipt_path.is_file()
    assert not diagnostic.exists()
    reopened = runner.authenticate_prepared_selective_catalog(
        trace_cache_directory=sealed_cache_path,
        selective_catalog_store_directory=store,
        selective_catalog_receipt_json=receipt_path,
        expected_selective_catalog_receipt_sha256=preparation[
            "preparation_receipt_sha256"
        ],
    )
    assert dict(reopened) == preparation

    with pytest.raises(
        runner.Lfm2CpcfV5SelectiveBlankFoldRunnerError,
        match="already exists",
    ):
        runner.prepare_selective_catalog(config)


def test_catalog_artifact_typed_and_persisted_tamper_fail_without_rebuild(
    sealed_cache_path,
    tmp_path,
    monkeypatch,
) -> None:
    store_root = tmp_path / "selective-catalogs"
    provider = open_or_build_lfm2_cpcf_selective_evidence_provider(
        open_lfm2_circuit_trace_cache(
            sealed_cache_path,
            verify=False,
        ),
        artifact_store=store_root,
    )
    assert provider.catalog_artifact is not None
    artifact_path = provider.catalog_artifact.path
    artifact_payload = copy.deepcopy(dict(provider.catalog_artifact.payload))
    provider.close()

    artifact_payload["dynamic_references"].reverse()
    with pytest.raises(
        (Lfm2CpcfSelectiveEvidenceError, DynamicEvidenceResidencyError),
        match="canonical|contiguous|seal|changed",
    ):
        lfm2_cpcf_selective_catalog_from_artifact_payload(
            artifact_payload
        )

    document_path = artifact_path / CATALOG_FILENAME
    document = json.loads(document_path.read_text(encoding="ascii"))
    document["payload"]["catalog"]["source_width"] += 1
    document_path.write_bytes(canonical_json_bytes(document) + b"\n")

    def forbidden_rebuild(*args, **kwargs):
        del args, kwargs
        raise AssertionError("corrupt immutable path was rebuilt")

    monkeypatch.setattr(
        selective,
        "_build_catalog_from_authenticated_source",
        forbidden_rebuild,
    )
    with pytest.raises(
        DynamicEvidenceCatalogArtifactIntegrityError,
        match="seal|binding",
    ):
        open_or_build_lfm2_cpcf_selective_evidence_provider(
            open_lfm2_circuit_trace_cache(
                sealed_cache_path,
                verify=False,
            ),
            artifact_store=store_root,
        )


def test_toy_cache_v5_phase_shells_and_active_layers_are_v4_exact(
    sealed_cache_path,
) -> None:
    full_parent_phases = tuple(
        iter_lfm2_cpcf_compiler_inputs(
            open_lfm2_circuit_trace_cache(
                sealed_cache_path,
                verify=False,
            )
        )
    )
    records = []
    target_inputs = (
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        torch.tensor([[1, 4, 3, 2]], dtype=torch.long),
    )
    forced = (
        torch.tensor([[5]], dtype=torch.long),
        torch.tensor([[6]], dtype=torch.long),
    )
    for batch_index, reference in enumerate(
        (
            full_parent_phases[0],
            full_parent_phases[2],
        )
    ):
        records.append(
            SimpleNamespace(
                ordinal=batch_index,
                record_id=f"toy-{batch_index}",
                split="fit" if batch_index == 0 else "validation",
                donor_cache_input_sha256=reference.input_sha256,
                alignment_sha256=v5._sha(
                    {"toy-alignment": batch_index}
                ),
                target_input_ids=target_inputs[batch_index],
                attention_mask=torch.ones_like(
                    target_inputs[batch_index]
                ),
                target_forced_decode_ids=forced[batch_index],
                position_ids=None,
            )
        )

    class _TraceAdapter:
        def __init__(self, values):
            self.records = tuple(values)

        def verify(self):
            return self

    adapter = _TraceAdapter(records)
    full_target_phases = model_compiler.build_lfm2_cpcf_target_phases(
        adapter,
        full_parent_phases,
    )
    provider = build_lfm2_cpcf_selective_evidence_provider(
        open_lfm2_circuit_trace_cache(
            sealed_cache_path,
            verify=False,
        )
    )
    try:
        reference0 = provider.dynamic_catalog.references[0]
        bootstrap = provider.load_layer_payload(reference0)
        provider.authenticate_layer_payload(reference0, bootstrap)
        try:
            shells = v5.build_lfm2_cpcf_v5_target_phase_shells(
                adapter,
                provider.catalog,
                bootstrap,
            )
        finally:
            provider.release_layer_payload(reference0, bootstrap)
        assert [phase.as_dict() for phase in shells] == [
            phase.as_dict() for phase in full_target_phases
        ]
        assert all(
            not layer.tensors
            for phase in shells
            for layer in phase.parent_evidence.layers
        )
        assert all(
            not phase.parent_evidence.final_norm_tensors
            for phase in shells
        )

        state_bank = model_compiler._phase_state_bank(
            "toy-input",
            0,
            shells,
            {
                phase.identity: torch.zeros(
                    phase.batch_size,
                    phase.full_prefix_tokens,
                    3,
                )
                for phase in shells
            },
        )
        original_state_pointers = {
            identity: int(value.data_ptr())
            for identity, value in state_bank.states.items()
        }
        maximum_bytes = sum(
            reference.logical_bytes
            for reference in provider.dynamic_catalog.references
        )
        with DynamicEvidenceResidencyWindow(
            provider.dynamic_catalog,
            provider,
            maximum_resident_bytes=maximum_bytes,
        ) as window:
            resident = window.start()
            for layer_index in range(provider.catalog.layer_count):
                materialized = (
                    v5.materialize_lfm2_cpcf_v5_layer_phases(
                        shells,
                        provider.catalog,
                        resident.payload,
                    )
                )
                for phase_index, phase in enumerate(materialized):
                    actual = phase.parent_evidence.layers[layer_index]
                    expected = full_target_phases[
                        phase_index
                    ].parent_evidence.layers[layer_index]
                    assert actual.evidence_sha256 == expected.evidence_sha256
                    _assert_same_tensors(actual.tensors, expected.tensors)
                    assert [
                        layer.layer_index
                        for layer in phase.parent_evidence.layers
                        if layer.tensors
                    ] == [layer_index]
                rebound = model_compiler.rebind_lfm2_cpcf_phase_state_bank(
                    state_bank,
                    materialized,
                )
                assert rebound.state_bank_sha256 == (
                    state_bank.state_bank_sha256
                )
                assert {
                    identity: int(value.data_ptr())
                    for identity, value in rebound.states.items()
                } == original_state_pointers
                resident = window.advance(
                    completed_layer_index=layer_index
                )
            snapshot = window.snapshot()
            assert snapshot["state"] == "complete"
            assert snapshot["peak_occupied_layers"] == 2
            assert snapshot["peak_resident_bytes"] == maximum_bytes
            assert snapshot["requested_layer_indices"] == [0, 1]
            assert snapshot["promoted_layer_indices"] == [0, 1]
            assert snapshot["released_layer_indices"] == [0, 1]
    finally:
        provider.close()


def test_selective_payload_order_content_and_byte_bounds_fail_closed(
    sealed_cache_path,
) -> None:
    provider = build_lfm2_cpcf_selective_evidence_provider(
        open_lfm2_circuit_trace_cache(
            sealed_cache_path,
            verify=False,
        )
    )
    try:
        reference = provider.dynamic_catalog.references[0]
        payload = provider.load_layer_payload(reference)
        try:
            phases = payload.phases
            payload._phases = tuple(reversed(phases))
            with pytest.raises(
                Lfm2CpcfSelectiveEvidenceError,
                match="phase reference|canonical|differs",
            ):
                provider.authenticate_layer_payload(reference, payload)
            payload._phases = phases

            first_tensor = next(
                iter(payload.phases[0].evidence.tensors.values())
            )
            first_tensor.reshape(-1)[0].add_(1)
            with pytest.raises(
                Lfm2CpcfSelectiveEvidenceError,
                match="differs|does not match",
            ):
                provider.authenticate_layer_payload(reference, payload)
        finally:
            provider.release_layer_payload(reference, payload)

        changed_bytes = DynamicEvidenceLayerReference.create(
            source_sha256=reference.source_sha256,
            provider_sha256=reference.provider_sha256,
            layer_index=reference.layer_index,
            evidence_sha256=reference.evidence_sha256,
            logical_bytes=reference.logical_bytes + 1,
        )
        with pytest.raises(
            Lfm2CpcfSelectiveEvidenceError,
            match="differs from the sealed catalog",
        ):
            provider.load_layer_payload(changed_bytes)

        pair_bytes = sum(
            value.logical_bytes
            for value in provider.dynamic_catalog.references[:2]
        )
        window = DynamicEvidenceResidencyWindow(
            provider.dynamic_catalog,
            provider,
            maximum_resident_bytes=pair_bytes - 1,
        )
        with pytest.raises(
            DynamicEvidenceResidencyError,
            match="above maximum_resident_bytes",
        ):
            window.start()
        window.close()

        batch_reference = provider._session.batch_references[0]
        names = batch_reference.ordered_tensor_names[:2]
        with pytest.raises(
            Lfm2CircuitTraceCacheError,
            match="canonical sorted order",
        ):
            provider._session.load_selected_tensors(
                0,
                tuple(reversed(names)),
            )
    finally:
        provider.close()


def test_selective_session_detects_post_authentication_record_tamper(
    sealed_cache_path,
    tmp_path,
) -> None:
    copied = tmp_path / sealed_cache_path.name
    shutil.copytree(sealed_cache_path, copied)
    provider = build_lfm2_cpcf_selective_evidence_provider(
        open_lfm2_circuit_trace_cache(copied, verify=False)
    )
    record_path = (
        copied
        / "batches"
        / "batch-000000"
        / BATCH_RECORD_FILENAME
    )
    original = record_path.read_bytes()
    record_path.write_bytes(original + b" ")
    try:
        with pytest.raises(
            Lfm2CircuitTraceCorruptionError,
            match="record identity changed",
        ):
            provider.load_layer_payload(
                provider.dynamic_catalog.references[0]
            )
    finally:
        with pytest.raises(
            Lfm2CircuitTraceCorruptionError,
            match="identity changed",
        ):
            provider.close()
