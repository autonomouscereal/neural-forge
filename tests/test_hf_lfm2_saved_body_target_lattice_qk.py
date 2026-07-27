from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_saved_body_target_lattice_qk as closure  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)


class _Config:
    model_type = "lfm2"
    norm_eps = 1e-5

    def to_dict(self) -> dict[str, object]:
        return {"model_type": self.model_type, "norm_eps": self.norm_eps}


class _Attention(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False, dtype=dtype)
        self.k_proj = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
        self.v_proj = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
        self.out_proj = torch.nn.Linear(4, 4, bias=False, dtype=dtype)


class _Layer(torch.nn.Module):
    def __init__(
        self,
        *,
        attention: bool,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.is_attention_layer = attention
        if attention:
            self.self_attn = _Attention(dtype=dtype)
        else:
            self.conv_weight = torch.nn.Parameter(torch.ones(4, dtype=dtype))


class _Model(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.config = _Config()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [
                _Layer(attention=True, dtype=dtype),
                _Layer(attention=False, dtype=dtype),
                _Layer(attention=True, dtype=dtype),
            ]
        )

    def forward(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("Saved-body Q/K closure must not call model.forward().")


def _selection(layer_index: int) -> closure.Lfm2QdeimHeadSelection:
    return closure.Lfm2QdeimHeadSelection(
        layer_index=layer_index,
        source_query_head_count=4,
        source_key_value_head_count=2,
        target_query_head_count=2,
        target_key_value_head_count=1,
        selected_source_key_value_groups=(1,),
        selected_source_query_heads=(2, 3),
        source_receipt_sha256=str(layer_index + 1) * 64,
    )


def test_qdeim_selection_rejects_partial_reordered_or_duplicate_groups() -> None:
    selected = _selection(0)
    assert selected.selected_source_query_heads == (2, 3)
    assert selected.as_dict()["head_averaging_performed"] is False

    with pytest.raises(
        closure.Lfm2SavedBodyTargetLatticeQkError,
        match="intact source GQA groups",
    ):
        closure.Lfm2QdeimHeadSelection(
            layer_index=0,
            source_query_head_count=4,
            source_key_value_head_count=2,
            target_query_head_count=2,
            target_key_value_head_count=1,
            selected_source_key_value_groups=(1,),
            selected_source_query_heads=(3, 2),
            source_receipt_sha256="1" * 64,
        )

    with pytest.raises(
        closure.Lfm2SavedBodyTargetLatticeQkError,
        match="intact source GQA groups",
    ):
        closure.Lfm2QdeimHeadSelection(
            layer_index=0,
            source_query_head_count=4,
            source_key_value_head_count=2,
            target_query_head_count=4,
            target_key_value_head_count=2,
            selected_source_key_value_groups=(1, 1),
            selected_source_query_heads=(2, 3, 2, 3),
            source_receipt_sha256="1" * 64,
        )


def test_target_lattice_evidence_rejects_selection_target_topology_mismatch() -> None:
    model = _Model()
    model.model.layers[0].self_attn.head_dim = 2
    lineage = SimpleNamespace(
        layer_index=0,
        query_groups=((0, 1), (2, 3)),
        key_value_groups=((0,), (1,)),
    )
    mismatched = closure.Lfm2QdeimHeadSelection(
        layer_index=0,
        source_query_head_count=4,
        source_key_value_head_count=2,
        target_query_head_count=2,
        target_key_value_head_count=1,
        selected_source_key_value_groups=(1,),
        selected_source_query_heads=(2, 3),
        source_receipt_sha256="1" * 64,
    )
    model.model.layers[0].self_attn.k_proj = torch.nn.Linear(
        4,
        4,
        bias=False,
        dtype=torch.bfloat16,
    )

    with pytest.raises(
        closure.Lfm2SavedBodyTargetLatticeQkError,
        match="live target GQA topology",
    ):
        closure._build_target_lattice_evidence(
            cache=SimpleNamespace(),
            session=SimpleNamespace(),
            bridge=SimpleNamespace(),
            alignments=(),
            child_bank=SimpleNamespace(),
            target_model=model,
            layer_index=0,
            lineage=lineage,
            selection=mismatched,
            target_rope_inv_freq=torch.ones(1),
            target_rope_attention_scaling=1.0,
        )


class _FakeCache:
    cache_key = "a" * 64
    parent_fingerprint = "b" * 64
    batch_count = 4


class _FakeBridge:
    artifact_sha256 = "c" * 64


def _patch_causal_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    inject_non_qk_change: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[_Model, list[tuple[int, str]]]:
    model = _Model(dtype=dtype)
    replay_calls: list[tuple[int, str]] = []
    monkeypatch.setattr(closure, "Lfm2CircuitTraceCache", _FakeCache)
    monkeypatch.setattr(closure, "StrictTokenizerBridgeArtifact", _FakeBridge)
    monkeypatch.setattr(
        closure,
        "_validated_operator_topology",
        lambda *_args, **_kwargs: (
            "full_attention",
            "short_conv",
            "full_attention",
        ),
    )
    monkeypatch.setattr(
        closure,
        "_target_rope",
        lambda _model: (torch.tensor([1.0]), 1.0),
    )
    monkeypatch.setattr(
        closure,
        "_semantic_alignments_from_session",
        lambda **_kwargs: (
            (SimpleNamespace(source_batch_index=0),),
            {"receipt_sha256": "d" * 64},
        ),
    )
    monkeypatch.setattr(
        closure,
        "audit_lfm2_semantic_span_cpcf_refinement",
        lambda *_args, **_kwargs: {
            "probability_preserving_attention_transport_go": True,
            "blocking_reasons": [],
            "report_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(closure, "_initial_residual_bank", lambda *_args: {0: "r0"})

    def replay(
        target_model: _Model,
        *,
        layer_index: int,
        **_kwargs: object,
    ) -> SimpleNamespace:
        state = fingerprint_lfm2_parent(target_model).identity_sha256
        replay_calls.append((layer_index, state))
        return SimpleNamespace(
            captured_child_model_state_sha256=state,
            bank_sha256=canonical_sha256(
                {"layer_index": layer_index, "state": state, "ordinal": len(replay_calls)}
            ),
            replay_by_batch={0: SimpleNamespace(post_layer_residual=torch.zeros(1))},
            pre_residual_by_batch={0: torch.zeros(1)},
        )

    monkeypatch.setattr(closure, "_replay_current_layer", replay)
    monkeypatch.setattr(closure, "_next_residual_bank", lambda bank: {0: bank.bank_sha256})
    monkeypatch.setattr(
        closure,
        "_build_target_lattice_evidence",
        lambda **kwargs: (
            SimpleNamespace(
                evidence_sha256=canonical_sha256(
                    {"layer_index": kwargs["layer_index"]}
                )
            ),
        ),
    )

    def install(*, target_model: _Model, layer_index: int, **_kwargs: object) -> dict[str, object]:
        before = fingerprint_lfm2_parent(target_model).identity_sha256
        layer = target_model.model.layers[layer_index]
        with torch.no_grad():
            layer.self_attn.q_proj.weight[0, 0] += 1.0
            if inject_non_qk_change and layer_index == 2:
                layer.self_attn.v_proj.weight[0, 0] += 1.0
        after = fingerprint_lfm2_parent(target_model).identity_sha256
        body = {
            "schema": closure.LFM2_SEMANTIC_TARGET_LATTICE_QK_ONLY_SCHEMA,
            "installed": True,
            "model_state_before_sha256": before,
            "model_state_after_sha256": after,
            "changed_state_keys": [
                f"model.layers.{layer_index}.self_attn.q_proj.weight"
            ],
            "actual_changed_state_keys_are_qk_only": True,
            "parameter_dtype": str(dtype).removeprefix("torch."),
            "checkpoint_dtype_preserved": True,
            "numeric_metric_used_for_publication_or_rollback": False,
        }
        return {**body, "result_sha256": canonical_sha256(body)}

    monkeypatch.setattr(
        closure,
        "compile_install_lfm2_semantic_target_lattice_qk_only",
        install,
    )
    return model, replay_calls


def test_saved_body_closure_replays_in_causal_order_and_changes_only_qk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, replay_calls = _patch_causal_harness(
        monkeypatch,
        inject_non_qk_change=False,
    )
    before = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    cache = _FakeCache()
    receipt = closure.compile_install_lfm2_saved_body_target_lattice_qk_closure(
        cache=cache,
        bridge=_FakeBridge(),
        target_model=model,
        compiler_lineage={0: object(), 2: object()},
        qdeim_head_selections={0: _selection(0), 2: _selection(2)},
        donor_tokenizer=object(),
        target_tokenizer=object(),
        authenticated_cache_session=SimpleNamespace(),
    )

    assert receipt["actual_changed_state_keys_are_qk_only"] is True
    assert receipt["tokenizer_session_sha256"] == "d" * 64
    assert receipt["semantic_refinement_sha256"] == "e" * 64
    assert receipt["changed_state_keys"] == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.2.self_attn.q_proj.weight",
    ]
    assert [layer for layer, _state in replay_calls] == [0, 0, 1, 2, 2]
    assert replay_calls[1][1] == replay_calls[2][1] == replay_calls[3][1]
    assert replay_calls[0][1] != replay_calls[1][1]
    assert replay_calls[3][1] != replay_calls[4][1]
    for name, value in model.state_dict().items():
        if name in receipt["changed_state_keys"]:
            assert not torch.equal(value.detach().cpu(), before[name])
        else:
            assert torch.equal(value.detach().cpu(), before[name]), name
    assert all(value == 0 for value in receipt["operation_audit"].values())


def test_fp32_qk_only_treatment_survives_safetensors_fresh_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    model, _replay_calls = _patch_causal_harness(
        monkeypatch,
        inject_non_qk_change=False,
        dtype=torch.float32,
    )
    before_hashes = closure._state_tensor_hashes(model)
    receipt = closure.compile_install_lfm2_saved_body_target_lattice_qk_closure(
        cache=_FakeCache(),
        bridge=_FakeBridge(),
        target_model=model,
        compiler_lineage={0: object(), 2: object()},
        qdeim_head_selections={0: _selection(0), 2: _selection(2)},
        donor_tokenizer=object(),
        target_tokenizer=object(),
        authenticated_cache_session=SimpleNamespace(),
    )
    treated_hashes = closure._state_tensor_hashes(model)
    checkpoint = tmp_path / "model.safetensors"
    safetensors.save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        },
        str(checkpoint),
    )
    reloaded = _Model(dtype=torch.float32)
    reloaded.load_state_dict(safetensors.load_file(str(checkpoint)), strict=True)

    assert receipt["parameter_dtype"] == "float32"
    assert receipt["checkpoint_dtype_preserved"] is True
    assert fingerprint_lfm2_parent(reloaded).identity_sha256 == receipt[
        "model_state_after_sha256"
    ]
    assert closure._state_tensor_hashes(reloaded) == treated_hashes
    assert list(
        closure._changed_state_keys(
            before_hashes,
            closure._state_tensor_hashes(reloaded),
        )
    ) == receipt["changed_state_keys"]
    assert all(parameter.dtype == torch.float32 for parameter in reloaded.parameters())


def test_saved_body_closure_rolls_back_malicious_non_qk_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _replay_calls = _patch_causal_harness(
        monkeypatch,
        inject_non_qk_change=True,
    )
    before_identity = fingerprint_lfm2_parent(model).identity_sha256
    before = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }

    with pytest.raises(
        closure.Lfm2SavedBodyTargetLatticeQkError,
        match="outside Q/K",
    ):
        closure.compile_install_lfm2_saved_body_target_lattice_qk_closure(
            cache=_FakeCache(),
            bridge=_FakeBridge(),
            target_model=model,
            compiler_lineage={0: object(), 2: object()},
            qdeim_head_selections={0: _selection(0), 2: _selection(2)},
            donor_tokenizer=object(),
            target_tokenizer=object(),
            authenticated_cache_session=SimpleNamespace(),
        )

    assert fingerprint_lfm2_parent(model).identity_sha256 == before_identity
    assert all(
        torch.equal(value.detach().cpu(), before[name])
        for name, value in model.state_dict().items()
    )
