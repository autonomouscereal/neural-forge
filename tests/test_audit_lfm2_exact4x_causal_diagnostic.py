from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments import audit_lfm2_exact4x_causal_diagnostic as runner


class _FakeShortConv(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.out_proj = torch.nn.Linear(width, width, bias=False)

    def forward(self, hidden_states: torch.Tensor, **_kwargs) -> torch.Tensor:
        context = torch.cumsum(torch.tanh(hidden_states), dim=1)
        return self.out_proj(context)


class _FakeMlp(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(width, width + 1, bias=False)
        self.w3 = torch.nn.Linear(width, width + 1, bias=False)
        self.w2 = torch.nn.Linear(width + 1, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(value)) * self.w3(value))


class _FakeLayer(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv = _FakeShortConv(width)
        self.feed_forward = _FakeMlp(width)

    def forward(self, hidden_states: torch.Tensor, **_kwargs) -> torch.Tensor:
        post_operator = hidden_states + self.conv(hidden_states=hidden_states)
        return post_operator + self.feed_forward(post_operator)


class _FakeBody(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer(width), _FakeLayer(width)])


class _FakeCausalLm(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.model = _FakeBody(width)
        self.lm_head = torch.nn.Linear(width, 11, bias=False)

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert tuple(attention_mask.shape) == tuple(inputs_embeds.shape[:2])
        assert position_ids is None or tuple(position_ids.shape) == tuple(
            inputs_embeds.shape[:2]
        )
        assert use_cache is False
        assert return_dict is True
        hidden = inputs_embeds
        for layer in self.model.layers:
            hidden = layer(hidden_states=hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_eight_subset_replays_capture_full_base_and_phase_only_interventions() -> None:
    torch.manual_seed(7)
    width = 4
    child = _FakeCausalLm(width)
    before = _state(child)
    embeddings = torch.randn(1, 6, width)
    positions = torch.tensor([2, 3, 4], dtype=torch.long)
    targets = {
        "C": torch.randn(6, width),
        "S": torch.randn(6, width),
        "M": torch.randn(6, width),
    }
    specs = runner.default_lfm2_l0_csm_operator_specs()
    call_count = 0

    def counted(_module, _args, _output) -> None:
        nonlocal call_count
        call_count += 1

    handle = child.register_forward_hook(counted)
    try:
        observations = []
        for subset in runner.canonical_intervention_subsets(("C", "S", "M")):
            observations.append(
                runner._run_causal_replay(
                    child=child,
                    reconstructed_embeddings=embeddings,
                    attention_mask=torch.ones(1, 6, dtype=torch.long),
                    position_ids=torch.arange(6).unsqueeze(0),
                    intervention_positions=torch.arange(6),
                    semantic_positions=positions,
                    operator_specs=specs,
                    intervention_members=subset["members"],
                    projected_parent_targets=targets,
                    capture_full_features=subset["mask"] == 0,
                )
            )
    finally:
        handle.remove()
    assert call_count == 8
    assert set(observations[0]) == {
        "feature.S",
        "feature.M",
        "terminal_residual",
        "lm_head_input",
    }
    assert observations[0]["feature.S"].shape == (1, 6, width)
    assert observations[0]["feature.M"].shape == (1, 6, width + 1)
    assert observations[0]["terminal_residual"].shape == (1, 6, width)
    for observation in observations[1:]:
        assert set(observation) == {"terminal_residual", "lm_head_input"}
        assert observation["terminal_residual"].shape == (1, 3, width)
    after = _state(child)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(not module._forward_hooks for module in child.modules())
    assert all(not module._forward_pre_hooks for module in child.modules())


def test_replay_removes_temporary_hooks_when_forward_raises() -> None:
    child = _FakeCausalLm(4)

    def fail(_module, _args, _kwargs):
        raise RuntimeError("injected failure")

    handle = child.model.layers[1].register_forward_pre_hook(fail, with_kwargs=True)
    with pytest.raises(RuntimeError, match="injected failure"):
        runner._run_causal_replay(
            child=child,
            reconstructed_embeddings=torch.randn(1, 4, 4),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            position_ids=None,
            intervention_positions=torch.arange(4),
            semantic_positions=torch.tensor([1, 2], dtype=torch.long),
            operator_specs=runner.default_lfm2_l0_csm_operator_specs(),
            intervention_members=("C",),
            projected_parent_targets={
                "C": torch.zeros(4, 4),
                "S": torch.zeros(4, 4),
                "M": torch.zeros(4, 4),
            },
            capture_full_features=False,
        )
    handle.remove()
    assert all(not module._forward_hooks for module in child.modules())
    assert all(not module._forward_pre_hooks for module in child.modules())


def test_early_nonsemantic_prompt_intervention_changes_later_semantic_output() -> None:
    torch.manual_seed(13)
    child = _FakeCausalLm(4)
    embeddings = torch.randn(1, 6, 4)
    intervention_positions = torch.arange(6)
    semantic_positions = torch.tensor([4, 5], dtype=torch.long)
    targets = {
        "C": embeddings[0].detach().clone(),
        "S": torch.zeros(6, 4),
        "M": torch.zeros(6, 4),
    }
    targets["C"][0] += 25.0
    common = {
        "child": child,
        "reconstructed_embeddings": embeddings,
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "position_ids": torch.arange(6).unsqueeze(0),
        "intervention_positions": intervention_positions,
        "semantic_positions": semantic_positions,
        "operator_specs": runner.default_lfm2_l0_csm_operator_specs(),
        "projected_parent_targets": targets,
        "capture_full_features": False,
    }
    base = runner._run_causal_replay(
        **common,
        intervention_members=(),
    )
    intervened = runner._run_causal_replay(
        **common,
        intervention_members=("C",),
    )
    assert not torch.equal(
        base["terminal_residual"][:, -1],
        intervened["terminal_residual"][:, -1],
    )


def test_full_parent_trajectory_uses_every_prefill_row_and_one_row_per_decode() -> None:
    parent = {
        "prefill.layer.000.operator.output": torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
        ),
        "decode.000.layer.000.operator.output": torch.tensor([[[7.0, 8.0]]]),
        "decode.001.layer.000.operator.output": torch.tensor(
            [[[90.0, 91.0], [9.0, 10.0]]]
        ),
    }
    rows = runner._parent_full_trajectory_matrix(
        parent,
        phases=("prefill", "decode.000", "decode.001"),
        layer_index=0,
        endpoint="operator.output",
        source_width=2,
        prompt_count=3,
        forced_count=2,
    )
    torch.testing.assert_close(
        rows,
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            dtype=torch.float64,
        ),
    )


def test_operator_specs_support_conv_and_attention_without_path_rewrite() -> None:
    conv = runner._operator_specs_for_layer(0, "short_conv")
    assert [spec.group_id for spec in conv] == ["C", "S", "M"]
    assert conv[1].target_module_path == "model.layers.0.conv"
    assert conv[1].consumer_module_path == "model.layers.0.conv.out_proj"
    for layer_index in (10, 12):
        attention = runner._operator_specs_for_layer(
            layer_index,
            "full_attention",
        )
        assert [spec.group_id for spec in attention] == ["C", "A", "M"]
        assert attention[1].target_module_path == (
            f"model.layers.{layer_index}.self_attn"
        )
        assert attention[1].consumer_module_path == (
            f"model.layers.{layer_index}.self_attn.out_proj"
        )
        assert attention[2].target_module_path == (
            f"model.layers.{layer_index}.feed_forward.w2"
        )


def test_operator_kind_is_authenticated_against_trace_and_child_topology() -> None:
    topology = [
        {"contract": {"kind": "short_conv"}},
        {"contract": {"kind": "full_attention"}},
    ]
    child_layers = [
        SimpleNamespace(is_attention_layer=False),
        SimpleNamespace(is_attention_layer=True),
    ]
    assert (
        runner._authenticated_operator_kind(
            topology_layers=topology,
            child_layers=child_layers,
            layer_index=1,
            requested_kind="auto",
        )
        == "full_attention"
    )
    with pytest.raises(
        runner.Lfm2Exact4xCausalRunnerError,
        match="differs from the authenticated trace topology",
    ):
        runner._authenticated_operator_kind(
            topology_layers=topology,
            child_layers=child_layers,
            layer_index=1,
            requested_kind="short_conv",
        )
    with pytest.raises(
        runner.Lfm2Exact4xCausalRunnerError,
        match="Child operator kind differs",
    ):
        runner._authenticated_operator_kind(
            topology_layers=topology,
            child_layers=[
                SimpleNamespace(is_attention_layer=False),
                SimpleNamespace(is_attention_layer=False),
            ],
            layer_index=1,
            requested_kind="auto",
        )


def test_v3_implementation_identity_allows_paths_only_to_relocate() -> None:
    def identity(path: str, *, file_sha256: str = "a" * 64) -> dict[str, object]:
        body = {
            "schema": "v3-implementation-test",
            "files": {
                "runner": {"path": path, "sha256": file_sha256},
                "core": {"path": f"{path}.core", "sha256": "b" * 64},
            },
            "packages": {"torch": "test"},
            "numeric_contract": {"runtime_device": "cuda:0"},
            "execution_contract": {"parent_forward_calls": 0},
        }
        return {**body, "implementation_sha256": runner.canonical_sha256(body)}

    sealed = identity("/sealed/deployment/runner.py")
    relocated = identity("/new/deployment/runner.py")
    receipt = runner._authenticate_relocated_v3_implementation_identity(
        sealed_identity=sealed,
        current_identity=relocated,
    )
    assert receipt["all_file_sha256s_equal"] is True
    assert receipt["non_path_runtime_contract_equal"] is True
    assert all(row["path_relocated"] for row in receipt["files"])

    changed = identity("/new/deployment/runner.py", file_sha256="c" * 64)
    with pytest.raises(
        runner.Lfm2Exact4xCausalRunnerError,
        match="not byte-identical",
    ):
        runner._authenticate_relocated_v3_implementation_identity(
            sealed_identity=sealed,
            current_identity=changed,
        )


def test_feature_file_receipt_authenticates_tensors_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    feature_path = tmp_path / "batch-000000.safetensors"
    tensors = {
        "base.feature.S": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "semantic_positions": torch.tensor([1, 2, 3], dtype=torch.int64),
    }
    runner._atomic_publish_safetensors(feature_path, tensors)
    receipt = runner._build_feature_receipt(
        path=feature_path,
        tensors=tensors,
        request_sha256="1" * 64,
        batch_index=0,
        trace_input_sha256="2" * 64,
        trace_record_sha256="3" * 64,
    )
    authenticated, loaded = runner._authenticate_feature_receipt(
        receipt,
        feature_directory=tmp_path,
        expected_request_sha256="1" * 64,
        expected_batch_index=0,
        expected_trace_input_sha256="2" * 64,
        expected_trace_record_sha256="3" * 64,
    )
    assert authenticated == receipt
    assert all(torch.equal(loaded[name], tensors[name]) for name in tensors)
    raw = bytearray(feature_path.read_bytes())
    raw[-1] ^= 1
    feature_path.write_bytes(raw)
    with pytest.raises(
        runner.Lfm2Exact4xCausalRunnerError,
        match="differs from its immutable receipt",
    ):
        runner._authenticate_feature_receipt(
            receipt,
            feature_directory=tmp_path,
            expected_request_sha256="1" * 64,
            expected_batch_index=0,
            expected_trace_input_sha256="2" * 64,
            expected_trace_record_sha256="3" * 64,
        )


def test_completed_capture_resume_performs_zero_processor_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = {
        "request_sha256": "1" * 64,
        "record_bindings": [
            {
                "trace_input_sha256": "2" * 64,
                "trace_record_sha256": "3" * 64,
            }
        ],
    }
    reference = SimpleNamespace(
        index=0,
        input_sha256="2" * 64,
        record_sha256="3" * 64,
    )
    checkpoint_directory = tmp_path / "checkpoints"
    feature_directory = tmp_path / "features"

    monkeypatch.setattr(
        runner.causal_module,
        "authenticate_lfm2_exact4x_causal_diagnostic_request",
        lambda value: value,
    )
    monkeypatch.setattr(
        runner.causal_module,
        "authenticate_lfm2_exact4x_causal_batch_result",
        lambda value, request: value,
    )
    monkeypatch.setattr(
        runner.causal_module,
        "build_lfm2_exact4x_causal_batch_checkpoint",
        lambda request, result, authority: {"result": result},
    )

    def authenticate_resume(values, *, request, authority):
        return tuple(values)

    monkeypatch.setattr(
        runner.causal_module,
        "authenticate_lfm2_exact4x_causal_resume",
        authenticate_resume,
    )

    calls = 0

    def processor(_reference, _record):
        nonlocal calls
        calls += 1
        feature_directory.mkdir(parents=True, exist_ok=True)
        path = feature_directory / "batch-000000.safetensors"
        tensors = {"feature.S": torch.ones(2, 3)}
        runner._atomic_publish_safetensors(path, tensors)
        receipt = runner._build_feature_receipt(
            path=path,
            tensors=tensors,
            request_sha256="1" * 64,
            batch_index=0,
            trace_input_sha256="2" * 64,
            trace_record_sha256="3" * 64,
        )
        result = {
            "oracle_evidence": {"feature_receipt": receipt},
            "operation_audit": {"child_forward_calls": 8},
        }
        return result, receipt, tensors

    first = runner.resume_causal_batches(
        request=request,
        references=(reference,),
        checkpoint_directory=checkpoint_directory,
        feature_directory=feature_directory,
        checkpoint_authority=object(),
        processor=processor,
    )
    assert calls == 1
    assert first.current_run_child_forward_calls == 8

    def forbidden(_reference, _record):
        raise AssertionError("completed capture must not call processor")

    second = runner.resume_causal_batches(
        request=copy.deepcopy(request),
        references=(reference,),
        checkpoint_directory=checkpoint_directory,
        feature_directory=feature_directory,
        checkpoint_authority=object(),
        processor=forbidden,
    )
    assert second.checkpoint_hit_count == 1
    assert second.current_run_child_forward_calls == 0
    assert second.historical_checkpoint_child_forward_calls == 8


def test_restart_quarantines_uncheckpointed_feature_before_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = {
        "request_sha256": "1" * 64,
        "record_bindings": [
            {
                "trace_input_sha256": "2" * 64,
                "trace_record_sha256": "3" * 64,
            }
        ],
    }
    reference = SimpleNamespace(
        index=0,
        input_sha256="2" * 64,
        record_sha256="3" * 64,
    )
    checkpoint_directory = tmp_path / "checkpoints"
    feature_directory = tmp_path / "features"
    feature_directory.mkdir()
    orphan_path = feature_directory / "batch-000000.safetensors"
    runner._atomic_publish_safetensors(
        orphan_path,
        {"stale": torch.zeros(1, 1)},
    )
    stale_sha256 = runner._file_sha256(orphan_path)
    monkeypatch.setattr(
        runner.causal_module,
        "authenticate_lfm2_exact4x_causal_diagnostic_request",
        lambda value: value,
    )
    monkeypatch.setattr(
        runner.causal_module,
        "authenticate_lfm2_exact4x_causal_resume",
        lambda values, request, authority: tuple(values),
    )
    monkeypatch.setattr(
        runner.causal_module,
        "authenticate_lfm2_exact4x_causal_batch_result",
        lambda value, request: value,
    )
    monkeypatch.setattr(
        runner.causal_module,
        "build_lfm2_exact4x_causal_batch_checkpoint",
        lambda request, result, authority: {"result": result},
    )
    calls = 0

    def processor(_reference, _record):
        nonlocal calls
        calls += 1
        assert not orphan_path.exists()
        tensors = {"fresh": torch.ones(2, 2)}
        runner._atomic_publish_safetensors(orphan_path, tensors)
        receipt = runner._build_feature_receipt(
            path=orphan_path,
            tensors=tensors,
            request_sha256="1" * 64,
            batch_index=0,
            trace_input_sha256="2" * 64,
            trace_record_sha256="3" * 64,
        )
        return (
            {
                "oracle_evidence": {"feature_receipt": receipt},
                "operation_audit": {"child_forward_calls": 8},
            },
            receipt,
            tensors,
        )

    runner.resume_causal_batches(
        request=request,
        references=(reference,),
        checkpoint_directory=checkpoint_directory,
        feature_directory=feature_directory,
        checkpoint_authority=object(),
        processor=processor,
    )
    assert calls == 1
    quarantined = list(
        (feature_directory / "orphaned-incomplete-captures").glob(
            "*.orphaned-*.safetensors"
        )
    )
    assert len(quarantined) == 1
    assert stale_sha256 in quarantined[0].name
    assert orphan_path.exists()


def test_repeated_identical_orphans_use_collision_safe_quarantine_names(
    tmp_path: Path,
) -> None:
    feature_directory = tmp_path / "features"
    feature_directory.mkdir()
    orphan_path = feature_directory / "batch-000000.safetensors"
    receipts = []
    for _ in range(2):
        runner._atomic_publish_safetensors(
            orphan_path,
            {"stale": torch.zeros(1, 1)},
        )
        receipts.append(runner._quarantine_orphan_feature(orphan_path))
    assert receipts[0] is not None
    assert receipts[1] is not None
    assert receipts[0]["file_sha256"] == receipts[1]["file_sha256"]
    assert (
        receipts[0]["quarantine_relative_path"]
        != receipts[1]["quarantine_relative_path"]
    )
    for receipt in receipts:
        assert (feature_directory / receipt["quarantine_relative_path"]).is_file()
    assert not orphan_path.exists()


def test_bounded_resume_computes_only_limit_then_reuses_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = {
        "request_sha256": "1" * 64,
        "record_bindings": [
            {
                "trace_input_sha256": "2" * 64,
                "trace_record_sha256": "3" * 64,
            },
            {
                "trace_input_sha256": "4" * 64,
                "trace_record_sha256": "5" * 64,
            },
        ],
    }
    references = tuple(
        SimpleNamespace(
            index=index,
            input_sha256=record["trace_input_sha256"],
            record_sha256=record["trace_record_sha256"],
        )
        for index, record in enumerate(request["record_bindings"])
    )
    checkpoint_directory = tmp_path / "checkpoints"
    feature_directory = tmp_path / "features"
    monkeypatch.setattr(
        runner.causal_module,
        "authenticate_lfm2_exact4x_causal_diagnostic_request",
        lambda value: value,
    )
    monkeypatch.setattr(
        runner.causal_module,
        "authenticate_lfm2_exact4x_causal_resume",
        lambda values, request, authority: tuple(values),
    )
    monkeypatch.setattr(
        runner.causal_module,
        "authenticate_lfm2_exact4x_causal_batch_result",
        lambda value, request: value,
    )
    monkeypatch.setattr(
        runner.causal_module,
        "build_lfm2_exact4x_causal_batch_checkpoint",
        lambda request, result, authority: {"result": result},
    )
    calls = []

    def processor(reference, _record):
        calls.append(reference.index)
        feature_directory.mkdir(parents=True, exist_ok=True)
        path = feature_directory / f"batch-{reference.index:06d}.safetensors"
        tensors = {"feature.S": torch.full((1, 1), float(reference.index))}
        runner._atomic_publish_safetensors(path, tensors)
        record = request["record_bindings"][reference.index]
        receipt = runner._build_feature_receipt(
            path=path,
            tensors=tensors,
            request_sha256=request["request_sha256"],
            batch_index=reference.index,
            trace_input_sha256=record["trace_input_sha256"],
            trace_record_sha256=record["trace_record_sha256"],
        )
        return (
            {
                "oracle_evidence": {"feature_receipt": receipt},
                "operation_audit": {"child_forward_calls": 8},
            },
            receipt,
            tensors,
        )

    partial = runner.resume_causal_batches(
        request=request,
        references=references,
        checkpoint_directory=checkpoint_directory,
        feature_directory=feature_directory,
        checkpoint_authority=object(),
        processor=processor,
        maximum_new_batches=1,
    )
    assert calls == [0]
    assert partial.completed_batch_count == 1
    assert partial.remaining_batch_count == 1
    assert partial.computed_batch_count == 1
    assert partial.is_complete is False

    complete = runner.resume_causal_batches(
        request=copy.deepcopy(request),
        references=references,
        checkpoint_directory=checkpoint_directory,
        feature_directory=feature_directory,
        checkpoint_authority=object(),
        processor=processor,
    )
    assert calls == [0, 1]
    assert complete.checkpoint_hit_count == 1
    assert complete.computed_batch_count == 1
    assert complete.completed_batch_count == 2
    assert complete.is_complete is True
