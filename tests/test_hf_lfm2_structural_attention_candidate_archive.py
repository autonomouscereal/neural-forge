from __future__ import annotations

from types import MappingProxyType

import pytest

torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_attention_group_qdeim import (  # noqa: E402
    LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
    Lfm2CompiledAttentionGroupQdeim,
    lfm2_attention_group_qdeim_candidate_state_sha256,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import (  # noqa: E402
    LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
)
from neural_forge.hf_lfm2_structural_attention_candidate_archive import (  # noqa: E402
    Lfm2StructuralAttentionCandidateArchiveError,
    TENSOR_FILENAME,
    load_lfm2_structural_attention_candidate_archive,
    write_lfm2_structural_attention_candidate_archive,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


def _sealed(body: dict[str, object], field: str = "result_sha256") -> MappingProxyType:
    return MappingProxyType({**body, field: canonical_sha256(body)})


def _candidate(layer_index: int) -> Lfm2CompiledAttentionGroupQdeim:
    offset = float(layer_index + 1)
    query = torch.eye(4, dtype=torch.float32).mul(offset)
    key = torch.eye(4, dtype=torch.float32).mul(offset + 0.1)
    value = torch.eye(4, dtype=torch.float32).mul(offset + 0.2)
    out = torch.eye(4, dtype=torch.float32).mul(offset + 0.3)
    basis = torch.tensor([[1.0], [0.5]], dtype=torch.float64)
    interpolation = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    reconstruction = torch.eye(4, dtype=torch.float64)
    decoder = torch.eye(4, dtype=torch.float64)
    chart = torch.eye(4, dtype=torch.float64)
    body = {
        "schema": LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
        "selection": {
            "selected_source_key_value_groups": [layer_index],
            "selected_source_query_heads": [2 * layer_index, 2 * layer_index + 1],
        },
        "interpolation": {
            "group_pod_basis_sha256": lfm2_authenticated_tensor_sha256(basis),
            "group_interpolation_sha256": lfm2_authenticated_tensor_sha256(
                interpolation
            ),
            "context_reconstruction_sha256": lfm2_authenticated_tensor_sha256(
                reconstruction
            ),
        },
        "authentication": {
            "normalized_input_decoder_sha256": lfm2_authenticated_tensor_sha256(
                decoder
            ),
            "residual_output_chart_sha256": lfm2_authenticated_tensor_sha256(chart),
        },
        "materialization": {
            "candidate_state_sha256": (
                lfm2_attention_group_qdeim_candidate_state_sha256(
                    query, key, value, out
                )
            )
        },
    }
    return Lfm2CompiledAttentionGroupQdeim(
        selected_source_key_value_groups=(layer_index,),
        selected_source_query_heads=(2 * layer_index, 2 * layer_index + 1),
        query_weight=query,
        key_weight=key,
        value_weight=value,
        out_weight=out,
        group_pod_basis=basis,
        group_interpolation=interpolation,
        context_reconstruction=reconstruction,
        normalized_input_decoder=decoder,
        residual_output_chart=chart,
        report=_sealed(body),
    )


def _stack(candidates: dict[int, Lfm2CompiledAttentionGroupQdeim]):
    layers = []
    for index, candidate in sorted(candidates.items()):
        operator = _sealed(
            {
                "schema": "test-structural-attention-installation-v1",
                "coupled_attention_compiler_receipt": dict(candidate.report),
            }
        )
        layers.append(
            {
                "layer_index": index,
                "operator_kind": "full_attention",
                "attention_operator_installation_receipt": dict(operator),
            }
        )
    layers.insert(1, {"layer_index": 1, "operator_kind": "short_conv"})
    return _sealed(
        {
            "schema": LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
            "layer_installations": layers,
        }
    )


def test_archive_roundtrip_reconstructs_exact_typed_candidates(tmp_path) -> None:
    candidates = {0: _candidate(0), 2: _candidate(2)}
    stack = _stack(candidates)
    root = tmp_path / "candidate-archive"

    manifest = write_lfm2_structural_attention_candidate_archive(
        root,
        candidates,
        semantic_layer_stack_receipt=stack,
    )
    loaded, authenticated = load_lfm2_structural_attention_candidate_archive(
        root,
        semantic_layer_stack_receipt=stack,
        expected_manifest_sha256=manifest["manifest_sha256"],
    )

    assert authenticated["fresh_reload_verified"] is True
    assert set(loaded) == {0, 2}
    for index in loaded:
        assert loaded[index].report == candidates[index].report
        assert loaded[index].selected_source_query_heads == (
            candidates[index].selected_source_query_heads
        )
        for field in (
            "query_weight",
            "key_weight",
            "value_weight",
            "out_weight",
            "group_pod_basis",
            "group_interpolation",
            "context_reconstruction",
            "normalized_input_decoder",
            "residual_output_chart",
        ):
            assert torch.equal(getattr(loaded[index], field), getattr(candidates[index], field))


def test_archive_rejects_changed_tensor_file(tmp_path) -> None:
    candidates = {0: _candidate(0), 2: _candidate(2)}
    stack = _stack(candidates)
    root = tmp_path / "candidate-archive"
    manifest = write_lfm2_structural_attention_candidate_archive(
        root,
        candidates,
        semantic_layer_stack_receipt=stack,
    )
    tensor_path = root / TENSOR_FILENAME
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)

    with pytest.raises(
        Lfm2StructuralAttentionCandidateArchiveError,
        match="manifest, stack ancestry, or tensor file changed",
    ):
        load_lfm2_structural_attention_candidate_archive(
            root,
            semantic_layer_stack_receipt=stack,
            expected_manifest_sha256=manifest["manifest_sha256"],
        )


def test_archive_rejects_candidate_not_bound_to_stack(tmp_path) -> None:
    candidates = {0: _candidate(0), 2: _candidate(2)}
    stack = _stack(candidates)
    changed = _candidate(0)
    changed.query_weight.add_(1.0)

    with pytest.raises(
        Lfm2StructuralAttentionCandidateArchiveError,
        match="candidate tensors differ",
    ):
        write_lfm2_structural_attention_candidate_archive(
            tmp_path / "candidate-archive",
            {0: changed, 2: candidates[2]},
            semantic_layer_stack_receipt=stack,
        )
