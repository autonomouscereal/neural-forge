from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch import nn

from neural_forge.hf_lfm2_direct_local import (
    Lfm2DirectLocalCheckpointIO,
    Lfm2DirectLocalError,
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_model,
    relocate_authenticated_lfm2_canonical_fp32_model,
)
from neural_forge.hf_lfm2_semantic_step_token_fold import (
    seal_lfm2_semantic_step_span_evidence,
)
from neural_forge.hf_tokenizer_bridge import (
    seal_strict_tokenizer_bridge_artifact,
)
from neural_forge.lfm2_carrier_bridge_publication import (
    Lfm2CarrierBridgePublicationError,
    PORTABLE_CARRIER_RUNTIME_SCHEMA,
    fingerprint_lfm2_carrier_runtime_owned_tensor_state,
    relocate_and_bind_packaged_lfm2_carrier_routes,
)
from neural_forge.lfm2_carrier_child_package import (
    Lfm2CarrierChildCheckpointIO,
    Lfm2CarrierChildPackageError,
    Lfm2CarrierChildPackageRegistration,
    load_authenticated_lfm2_carrier_child_package,
)
from test_lfm2_carrier_child_package_fresh_process import (
    _BaseCheckpointIO,
    _Child,
    _build_child_package,
    _file_sha256,
    _source_publication,
    _tokenizer_package,
)


transformers = pytest.importorskip("transformers")


class _ExecutableChild(_Child):
    def __init__(self, rows: int = 2, hidden_size: int = 3) -> None:
        super().__init__(rows, hidden_size)
        self.config = SimpleNamespace(model_type="lfm2")

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        use_cache,
        return_dict,
    ):
        assert torch.is_grad_enabled() is False
        assert use_cache is False
        assert return_dict is True
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        hidden = self.input_table(input_ids)
        return SimpleNamespace(logits=self.output_table(hidden))


class _ExecutableBase(_BaseCheckpointIO):
    child_type = _ExecutableChild

    def __init__(self, tokenizer_source: Path | None) -> None:
        super().__init__(tokenizer_source)
        self.last_loaded = None

    def load(self, source: Path) -> _ExecutableChild:
        loaded = super().load(source)
        rows, hidden_size = loaded.input_table.weight.shape
        child = self.child_type(int(rows), int(hidden_size))
        child.load_state_dict(loaded.state_dict())
        self.last_loaded = child
        return child


def test_real_lfm2_package_publish_load_relocate_and_execute(
    tmp_path: Path,
) -> None:
    semantic_tokenizer_directory = tmp_path / "semantic-source-tokenizer"
    semantic_tokenizer, semantic_sha256 = _tokenizer_package(
        semantic_tokenizer_directory,
        {"<unk>": 0, "<eos>": 1, "hello": 2},
    )
    target_tokenizer_directory = tmp_path / "target-tokenizer"
    target_tokenizer, _target_sha256 = _tokenizer_package(
        target_tokenizer_directory,
        {"<unk>": 0, "<eos>": 1},
    )
    bridge = seal_strict_tokenizer_bridge_artifact(
        semantic_tokenizer,
        target_tokenizer,
        target_tokenizer_directory,
        donor_row_capacity=3,
        target_row_capacity=2,
    )
    publication, artifact_sha256 = _source_publication(tmp_path)
    spans = seal_lfm2_semantic_step_span_evidence(
        [[0], [1], [0, 1]],
        target_row_count=2,
        lineage={"real_lfm2_interface_integration": True},
    )
    span_path = tmp_path / "spans.json"
    span_path.write_text(
        json.dumps(spans, allow_nan=False, sort_keys=True) + "\n",
        encoding="ascii",
    )
    registration = Lfm2CarrierChildPackageRegistration(
        publication_directory=publication,
        expected_artifact_sha256=artifact_sha256,
        span_evidence_json=span_path,
        expected_span_evidence_sha256=str(spans["evidence_sha256"]),
        expected_span_evidence_file_sha256=_file_sha256(span_path),
        expected_target_rows=2,
        expected_hidden_size=2,
        expected_input_route_elements=4,
        expected_output_route_elements=4,
    )
    config = transformers.Lfm2Config(
        vocab_size=2,
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
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(65537)
    source_child = transformers.Lfm2ForCausalLM(config).eval()
    source_state_sha256 = fingerprint_lfm2_direct_local_state(source_child)
    direct_io = Lfm2DirectLocalCheckpointIO(
        bridge,
        device="cpu",
        dtype=torch.float32,
    )
    writer = Lfm2CarrierChildCheckpointIO(
        base=direct_io,
        registration=registration,
        semantic_source_tokenizer_directory=semantic_tokenizer_directory,
        semantic_source_tokenizer_sha256=semantic_sha256,
        in_memory_semantic_source_tokenizer=semantic_tokenizer,
        in_memory_target_tokenizer=target_tokenizer,
    )
    child_directory = tmp_path / "real-lfm2-child"
    child_directory.mkdir()
    writer.save(source_child, child_directory)

    package_before = load_authenticated_lfm2_carrier_child_package(child_directory)
    canonical_child = load_lfm2_direct_local_model(
        child_directory,
        device="cpu",
        dtype=torch.float32,
    ).eval()
    package = load_authenticated_lfm2_carrier_child_package(
        child_directory,
        child_model=canonical_child,
        expected_package_sha256=package_before.package_sha256,
    )
    child, runtime, receipt = relocate_and_bind_packaged_lfm2_carrier_routes(
        canonical_child,
        package.routes,
        provenance_resource_fingerprint="real-lfm2-package-integration-v1",
        runtime_device="cpu",
        runtime_dtype=torch.float32,
        expected_target_rows=2,
        expected_hidden_size=2,
    )

    assert isinstance(child, transformers.Lfm2ForCausalLM)
    assert package.package_sha256 == package_before.package_sha256
    assert fingerprint_lfm2_direct_local_state(canonical_child) == (source_state_sha256)
    receipt_body = receipt.as_dict()
    assert receipt_body["carrier_route_package_sha256"] == (
        package.routes.package_sha256
    )
    assert receipt_body["canonical_artifact_rewritten"] is False
    assert receipt_body["canonical_route_package_rewritten"] is False
    assert receipt_body["dtype_or_device_parameter_compaction_credit"] == 0
    assert (
        receipt_body["direct_local_relocation"]["executable_forward_smoke"][
            "logits_finite"
        ]
        is True
    )
    input_table = child.get_input_embeddings().weight
    output_table = child.get_output_embeddings().weight
    with runtime.authenticated_live_table_sessions(
        input_table,
        output_table,
    ) as (input_session, output_session):
        donor_embeddings = runtime.reconstruct_selected_donor_embeddings(
            input_session,
            [0, 1, 2],
        )
        donor_logits = runtime.compute_all_donor_logits_from_target_scores(
            output_session,
            torch.tensor([[2.0, 4.0]], dtype=torch.float32),
        )
    assert tuple(donor_embeddings.shape) == (3, 2)
    torch.testing.assert_close(
        donor_logits,
        torch.tensor([[2.0, 4.0, 3.5]], dtype=torch.float32),
    )


@pytest.mark.parametrize(
    "runtime_dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
def test_portable_carrier_runtime_preserves_canonical_package_and_counts(
    tmp_path: Path,
    runtime_dtype: torch.dtype,
) -> None:
    child_directory, _sources = _build_child_package(tmp_path)
    package_before = load_authenticated_lfm2_carrier_child_package(child_directory)
    file_hashes_before = {
        str(row["relative_path"]): str(row["sha256"])
        for row in package_before.receipt["child_file_records"]
    }
    loader = Lfm2CarrierChildCheckpointIO.from_child_directory(
        base=_ExecutableBase(None),
        child_directory=child_directory,
    )

    child, receipt = loader.load_authenticated_portable(
        child_directory,
        runtime_device="cpu",
        runtime_dtype=runtime_dtype,
    )
    body = receipt.as_dict()
    package_after = load_authenticated_lfm2_carrier_child_package(
        child_directory,
        expected_package_sha256=package_before.package_sha256,
    )

    assert body["schema"] == PORTABLE_CARRIER_RUNTIME_SCHEMA
    assert body["carrier_child_package_sha256"] == (package_before.package_sha256)
    assert body["carrier_route_package_sha256"] == (
        package_before.routes.package_sha256
    )
    assert package_after.package_sha256 == package_before.package_sha256
    assert {
        str(row["relative_path"]): _file_sha256(
            child_directory / str(row["relative_path"])
        )
        for row in package_after.receipt["child_file_records"]
    } == file_hashes_before
    assert body["canonical_dense_unique_parameter_elements"] == 12
    assert body["runtime_dense_unique_parameter_elements"] == 12
    assert body["counted_carrier_parameter_elements"] == 8
    assert body["counted_total_parameter_elements"] == 20
    assert body["dtype_or_device_parameter_compaction_credit"] == 0
    assert body["canonical_artifact_rewritten"] is False
    assert body["canonical_route_package_rewritten"] is False
    assert body["carrier_coefficient_artifact_dtype"] == "float32"
    assert body["carrier_coefficient_owned_runtime_dtype"] == "float32"
    assert body["direct_local_relocation"]["exact_parameter_count_invariant"] is True
    assert (
        body["direct_local_relocation"]["deterministic_post_cast_state_match"] is True
    )
    assert (
        body["direct_local_relocation"]["state_names_shapes_and_alias_abi_preserved"]
        is True
    )
    assert (
        body["direct_local_relocation"]["executable_forward_smoke"]["logits_finite"]
        is True
    )
    assert body["direct_local_relocation"]["runtime_dtype"] == str(runtime_dtype)
    assert {
        value.dtype
        for value in child.state_dict().values()
        if value.is_floating_point()
    } == {runtime_dtype}
    assert all(
        allocation.dtype == "float32" for allocation in package_after.routes.allocations
    )
    assert all(
        package_after.routes.tensors[name].dtype is torch.float32
        for name in ("input_coefficients", "output_coefficients")
    )
    assert callable(getattr(child, "semantic_step_generate", None))


@pytest.mark.parametrize(
    "runtime_dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
def test_portable_carrier_executes_both_routes_at_requested_dtype(
    tmp_path: Path,
    runtime_dtype: torch.dtype,
) -> None:
    child_directory, _sources = _build_child_package(tmp_path)
    canonical_child = _ExecutableBase(None).load(child_directory)
    package = load_authenticated_lfm2_carrier_child_package(
        child_directory,
        child_model=canonical_child,
    ).routes

    child, runtime, receipt = relocate_and_bind_packaged_lfm2_carrier_routes(
        canonical_child,
        package,
        provenance_resource_fingerprint="test-canonical-package",
        runtime_device="cpu",
        runtime_dtype=runtime_dtype,
        expected_target_rows=2,
        expected_hidden_size=3,
    )
    input_table = child.get_input_embeddings().weight
    output_table = child.get_output_embeddings().weight
    route_state_before = fingerprint_lfm2_carrier_runtime_owned_tensor_state(runtime)
    with runtime.authenticated_live_table_sessions(
        input_table,
        output_table,
    ) as (input_session, output_session):
        donor_embeddings = runtime.reconstruct_selected_donor_embeddings(
            input_session,
            [0, 1, 2],
        )
        donor_logits = runtime.compute_all_donor_logits_from_target_scores(
            output_session,
            torch.tensor([[2.0, 4.0]], dtype=runtime_dtype),
        )
    route_state_after = fingerprint_lfm2_carrier_runtime_owned_tensor_state(runtime)

    assert donor_embeddings.dtype is runtime_dtype
    assert torch.isfinite(donor_embeddings).all()
    assert torch.allclose(
        donor_embeddings[2].float(),
        (0.4 * input_table[0].detach().float() + 0.6 * input_table[1].detach().float()),
        atol=2e-3,
        rtol=2e-3,
    )
    expected_logit_dtype = (
        torch.float32
        if runtime_dtype in {torch.float16, torch.bfloat16}
        else runtime_dtype
    )
    assert donor_logits.dtype is expected_logit_dtype
    assert torch.isfinite(donor_logits).all()
    assert torch.allclose(
        donor_logits.float(),
        torch.tensor([[2.0, 4.0, 3.5]]),
        atol=2e-3,
        rtol=2e-3,
    )
    body = receipt.as_dict()
    assert body["carrier_coefficient_owned_runtime_dtype"] == "float32"
    assert body["dtype_or_device_parameter_compaction_credit"] == 0
    assert route_state_after == route_state_before
    assert route_state_after["continuous_coefficient_tensor_count"] == 2
    assert route_state_after["continuous_coefficients_remain_fp32"] is True


def test_portable_route_fingerprint_requires_both_fp32_coefficient_tensors(
    tmp_path: Path,
) -> None:
    child_directory, _sources = _build_child_package(tmp_path)
    canonical_child = _ExecutableBase(None).load(child_directory)
    package = load_authenticated_lfm2_carrier_child_package(
        child_directory,
        child_model=canonical_child,
    ).routes
    _child, runtime, _receipt = relocate_and_bind_packaged_lfm2_carrier_routes(
        canonical_child,
        package,
        provenance_resource_fingerprint="test-canonical-package",
        runtime_device="cpu",
        runtime_dtype=torch.float32,
        expected_target_rows=2,
        expected_hidden_size=3,
    )

    runtime._semantic_code_coefficients = None
    state = fingerprint_lfm2_carrier_runtime_owned_tensor_state(runtime)

    assert state["continuous_coefficient_tensor_count"] == 1
    assert state["continuous_coefficients_remain_fp32"] is False


@pytest.mark.parametrize(
    ("runtime_device", "runtime_dtype"),
    (
        ("cpu", torch.int8),
        ("not-a-real-device", torch.float32),
        ("cuda", torch.float32),
        ("cuda:999999", torch.float32),
    ),
)
def test_portable_carrier_runtime_has_no_device_or_dtype_fallback(
    tmp_path: Path,
    runtime_device: str,
    runtime_dtype: torch.dtype,
) -> None:
    child_directory, _sources = _build_child_package(tmp_path)
    loader = Lfm2CarrierChildCheckpointIO.from_child_directory(
        base=_ExecutableBase(None),
        child_directory=child_directory,
    )

    with pytest.raises(
        Lfm2CarrierChildPackageError,
        match="portable runtime relocation",
    ):
        loader.load_authenticated_portable(
            child_directory,
            runtime_device=runtime_device,
            runtime_dtype=runtime_dtype,
        )


def test_portable_carrier_rejects_noncanonical_model_forgery(
    tmp_path: Path,
) -> None:
    child_directory, _sources = _build_child_package(tmp_path)

    class _ForgedBf16Base(_ExecutableBase):
        def load(self, source: Path) -> _Child:
            return super().load(source).to(dtype=torch.bfloat16)

    loader = Lfm2CarrierChildCheckpointIO.from_child_directory(
        base=_ForgedBf16Base(None),
        child_directory=child_directory,
    )
    with pytest.raises(
        (Lfm2CarrierChildPackageError, Lfm2CarrierBridgePublicationError),
    ):
        loader.load_authenticated_portable(
            child_directory,
            runtime_device="cpu",
            runtime_dtype=torch.bfloat16,
        )


def test_portable_carrier_rejects_model_table_mutation_before_relocation(
    tmp_path: Path,
) -> None:
    child_directory, _sources = _build_child_package(tmp_path)

    class _MutatedTableBase(_ExecutableBase):
        def load(self, source: Path) -> _Child:
            child = super().load(source)
            with torch.no_grad():
                child.input_table.weight[0, 0].add_(1.0)
            return child

    loader = Lfm2CarrierChildCheckpointIO.from_child_directory(
        base=_MutatedTableBase(None),
        child_directory=child_directory,
    )
    with pytest.raises(Lfm2CarrierBridgePublicationError):
        loader.load_authenticated_portable(
            child_directory,
            runtime_device="cpu",
            runtime_dtype=torch.float16,
        )


@pytest.mark.parametrize("forgery", ("tensor", "receipt"))
def test_low_level_portable_binding_reauthenticates_retained_package(
    tmp_path: Path,
    forgery: str,
) -> None:
    child_directory, _sources = _build_child_package(tmp_path)
    canonical_child = _ExecutableBase(None).load(child_directory)
    package = load_authenticated_lfm2_carrier_child_package(
        child_directory,
        child_model=canonical_child,
    ).routes
    if forgery == "tensor":
        package.tensors["input_coefficients"][0].add_(1.0)
    else:
        forged_receipt = dict(package.receipt)
        forged_receipt["package_sha256"] = "0" * 64
        package = replace(
            package,
            receipt=MappingProxyType(forged_receipt),
        )

    with pytest.raises(
        Lfm2CarrierBridgePublicationError,
        match="changed before canonical FP32 relocation",
    ):
        relocate_and_bind_packaged_lfm2_carrier_routes(
            canonical_child,
            package,
            provenance_resource_fingerprint="test-canonical-package",
            runtime_device="cpu",
            runtime_dtype=torch.float16,
            expected_target_rows=2,
            expected_hidden_size=3,
        )
    assert {
        value.dtype
        for value in canonical_child.state_dict().values()
        if value.is_floating_point()
    } == {torch.float32}


def test_direct_relocation_rejects_parameter_count_mutation() -> None:
    class _CountChangingChild(_ExecutableChild):
        def to(self, *args, **kwargs):
            result = super().to(*args, **kwargs)
            reference = next(self.parameters())
            self.forged_parameter = nn.Parameter(
                torch.ones(
                    1,
                    device=reference.device,
                    dtype=reference.dtype,
                )
            )
            return result

    canonical_child = _CountChangingChild()
    canonical_sha256 = {
        name: value.detach().clone()
        for name, value in canonical_child.state_dict().items()
    }
    with pytest.raises(
        Lfm2DirectLocalError,
        match="parameter count",
    ):
        relocate_authenticated_lfm2_canonical_fp32_model(
            canonical_child,
            provenance_resource_fingerprint="test-count-mutation",
            runtime_device="cpu",
            runtime_dtype=torch.float16,
            fresh_loaded_runtime_copy=True,
        )
    assert all(
        torch.equal(canonical_child.state_dict()[name], value)
        for name, value in canonical_sha256.items()
    )
    assert not hasattr(canonical_child, "forged_parameter")


def test_direct_relocation_rejects_same_count_value_corrupting_to() -> None:
    class _ValueCorruptingChild(_ExecutableChild):
        def to(self, *args, **kwargs):
            result = super().to(*args, **kwargs)
            with torch.no_grad():
                next(self.parameters()).add_(1.0)
            return result

    canonical_child = _ValueCorruptingChild()
    canonical_state = {
        name: value.detach().clone()
        for name, value in canonical_child.state_dict().items()
    }
    with pytest.raises(
        Lfm2DirectLocalError,
        match="deterministic CPU FP32 post-cast bytes",
    ):
        relocate_authenticated_lfm2_canonical_fp32_model(
            canonical_child,
            provenance_resource_fingerprint="test-value-corruption",
            runtime_device="cpu",
            runtime_dtype=torch.float16,
            fresh_loaded_runtime_copy=True,
        )
    assert all(
        torch.equal(canonical_child.state_dict()[name], value)
        for name, value in canonical_state.items()
    )


def test_direct_relocation_seals_nonpersistent_buffer_cast_bytes() -> None:
    class _BufferCorruptingChild(_ExecutableChild):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "runtime_only_buffer",
                torch.tensor([1.0]),
                persistent=False,
            )

        def to(self, *args, **kwargs):
            result = super().to(*args, **kwargs)
            self.runtime_only_buffer.add_(1.0)
            return result

    canonical_child = _BufferCorruptingChild()
    with pytest.raises(
        Lfm2DirectLocalError,
        match="deterministic CPU FP32 post-cast bytes",
    ):
        relocate_authenticated_lfm2_canonical_fp32_model(
            canonical_child,
            provenance_resource_fingerprint="test-buffer-cast-corruption",
            runtime_device="cpu",
            runtime_dtype=torch.float16,
            fresh_loaded_runtime_copy=True,
        )
    assert canonical_child.runtime_only_buffer.item() == 1.0


def test_direct_forward_smoke_rejects_nonpersistent_buffer_mutation() -> None:
    class _BufferMutatingForwardChild(_ExecutableChild):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "runtime_only_buffer",
                torch.tensor([1.0]),
                persistent=False,
            )

        def forward(self, **kwargs):
            result = super().forward(**kwargs)
            self.runtime_only_buffer.add_(1.0)
            return result

    canonical_child = _BufferMutatingForwardChild()
    with pytest.raises(
        Lfm2DirectLocalError,
        match="mutated runtime model state",
    ):
        relocate_authenticated_lfm2_canonical_fp32_model(
            canonical_child,
            provenance_resource_fingerprint="test-buffer-forward-mutation",
            runtime_device="cpu",
            runtime_dtype=torch.float32,
            fresh_loaded_runtime_copy=True,
        )
    assert canonical_child.runtime_only_buffer.item() == 1.0


def test_direct_relocation_fails_closed_when_deepcopy_breaks_overlap_alias() -> None:
    class _OverlappingParameterChild(_ExecutableChild):
        def __init__(self) -> None:
            super().__init__()
            shared = torch.arange(6, dtype=torch.float32)
            self.overlap_left = nn.Parameter(shared[:4])
            self.overlap_right = nn.Parameter(shared[1:5])

    canonical_child = _OverlappingParameterChild()
    left = canonical_child.overlap_left
    right = canonical_child.overlap_right
    assert left.untyped_storage().data_ptr() == (right.untyped_storage().data_ptr())
    assert (left.storage_offset(), right.storage_offset()) == (0, 1)

    with pytest.raises(
        Lfm2DirectLocalError,
        match="Transactional runtime copy changed",
    ):
        relocate_authenticated_lfm2_canonical_fp32_model(
            canonical_child,
            provenance_resource_fingerprint="test-overlap-alias",
            runtime_device="cpu",
            runtime_dtype=torch.float32,
            fresh_loaded_runtime_copy=True,
        )

    assert left.untyped_storage().data_ptr() == (right.untyped_storage().data_ptr())
    assert (left.storage_offset(), right.storage_offset()) == (0, 1)


def test_cuda_bfloat16_rejects_p40_capability_before_storage_mutation(
    monkeypatch,
) -> None:
    canonical_child = _ExecutableChild()
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (6, 1),
    )

    with pytest.raises(
        Lfm2DirectLocalError,
        match="native compute capability 8.0",
    ):
        relocate_authenticated_lfm2_canonical_fp32_model(
            canonical_child,
            provenance_resource_fingerprint="test-p40-bfloat16",
            runtime_device="cuda:0",
            runtime_dtype=torch.bfloat16,
            fresh_loaded_runtime_copy=True,
        )
    assert {
        value.dtype
        for value in canonical_child.state_dict().values()
        if value.is_floating_point()
    } == {torch.float32}


def test_storage_success_forward_failure_leaves_disk_and_caller_canonical(
    tmp_path: Path,
) -> None:
    class _ForwardFailingChild(_ExecutableChild):
        def forward(self, **_kwargs):
            raise RuntimeError("simulated executable forward failure")

    class _ForwardFailingBase(_ExecutableBase):
        child_type = _ForwardFailingChild

    child_directory, _sources = _build_child_package(tmp_path)
    package_before = load_authenticated_lfm2_carrier_child_package(child_directory)
    file_hashes_before = {
        str(row["relative_path"]): _file_sha256(
            child_directory / str(row["relative_path"])
        )
        for row in package_before.receipt["child_file_records"]
    }
    base = _ForwardFailingBase(None)
    loader = Lfm2CarrierChildCheckpointIO.from_child_directory(
        base=base,
        child_directory=child_directory,
    )

    with pytest.raises(
        Lfm2CarrierChildPackageError,
        match="portable runtime relocation",
    ):
        loader.load_authenticated_portable(
            child_directory,
            runtime_device="cpu",
            runtime_dtype=torch.float16,
        )

    assert base.last_loaded is not None
    assert {
        value.dtype
        for value in base.last_loaded.state_dict().values()
        if value.is_floating_point()
    } == {torch.float32}
    package_after = load_authenticated_lfm2_carrier_child_package(
        child_directory,
        expected_package_sha256=package_before.package_sha256,
    )
    assert {
        str(row["relative_path"]): _file_sha256(
            child_directory / str(row["relative_path"])
        )
        for row in package_after.receipt["child_file_records"]
    } == file_hashes_before
    assert not any(
        "portable" in path.name.lower() for path in child_directory.rglob("*.json")
    )
