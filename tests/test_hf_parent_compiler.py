from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_parent_compiler as compiler_module  # noqa: E402
from neural_forge.hf_byte_span_trace import ByteSpanTraceInput  # noqa: E402
from neural_forge.hf_canonical_graph import (  # noqa: E402
    CanonicalChildSpec,
    CanonicalParentLogits,
    CanonicalTokenContract,
)
from neural_forge.hf_canonical_kernel import (  # noqa: E402
    CanonicalParentTrace,
    CanonicalRowIdentity,
    build_proportional_depth_cover,
)
from neural_forge.hf_factorized_vocab import exact_parameter_report  # noqa: E402
from neural_forge.hf_parent_compiler import (  # noqa: E402
    HFParentCompilerError,
    compile_hf_parents_to_blank_child,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _TinyDecoder(torch.nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        vocabulary_size: int = 3,
        carrier: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, hidden_size)
        self.body = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.head = torch.nn.Linear(hidden_size, vocabulary_size, bias=False)
        self.head.weight = self.embedding.weight
        self._compiler_carrier = carrier

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head


def _layout(model: _TinyDecoder):
    carrier = model._compiler_carrier
    hidden = int(model.embedding.weight.shape[1])
    return SimpleNamespace(
        family="llama",
        model_type="llama",
        model_class=type(model).__name__,
        hidden_size=hidden,
        intermediate_size=6 if carrier else 3,
        layer_count=2 if carrier else 1,
        embeddings=SimpleNamespace(vocab_size=int(model.embedding.weight.shape[0])),
        position=SimpleNamespace(kind="rope"),
        tie_word_embeddings=True,
    )


def _models():
    torch.manual_seed(101)
    carrier = _TinyDecoder(hidden_size=4, carrier=True)
    torch.manual_seed(102)
    target = _TinyDecoder(hidden_size=2)
    return carrier, target


def _child(target: _TinyDecoder, *, budget_delta: int = 0) -> CanonicalChildSpec:
    budget = int(exact_parameter_report(target)["unique_parameter_count"])
    return CanonicalChildSpec(
        family="llama",
        layer_count=1,
        hidden_size=2,
        parameter_budget=budget + budget_delta,
        vocabulary_size=3,
        tokenizer_fingerprint=_digest("child-tokenizer"),
        tie_embeddings=True,
        position_policy="rope",
        architecture_fingerprint=_digest("tiny-llama-structure"),
    )


def _identity(
    name: str,
    split: str,
    prompts: tuple[str, ...],
    *,
    row_prefix: str | None = None,
) -> CanonicalRowIdentity:
    prefix = split if row_prefix is None else row_prefix
    return CanonicalRowIdentity(
        split_name=split,
        pooling="last_valid",
        row_ids=tuple(f"{prefix}-row-{index}" for index in range(len(prompts))),
        raw_prompt_sha256=tuple(_digest(prompt) for prompt in prompts),
        corpus_sha256=_digest("\n".join(prompts)),
        tokenizer_fingerprint=_digest(f"{name}-tokenizer"),
        cache_fingerprint=_digest(f"{name}-{split}-cache"),
    )


def _states(seed: int, rows: int, width: int, depth: int):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(rows, width, generator=generator, dtype=torch.float64)
    return tuple(base + 0.05 * index * torch.sin(base) for index in range(depth + 1))


def _parent(
    name: str,
    *,
    split: str,
    prompts: tuple[str, ...],
    seed: int,
    width: int,
    depth: int,
    weight: float = 1.0,
    identity_prompts: tuple[str, ...] | None = None,
) -> CanonicalParentTrace:
    rows = len(prompts)
    return CanonicalParentTrace(
        name=name,
        layer_states=_states(seed, rows, width, depth),
        row_identity=_identity(
            name,
            split,
            prompts if identity_prompts is None else identity_prompts,
        ),
        depth_cover=build_proportional_depth_cover(depth, 1),
        weight=weight,
    )


def _contract(name: str) -> CanonicalTokenContract:
    return CanonicalTokenContract(
        parent_name=name,
        parent_tokenizer_fingerprint=_digest(f"{name}-tokenizer"),
        child_tokenizer_fingerprint=_digest("child-tokenizer"),
        method="byte-span-likelihood",
        mapping_fingerprint=_digest(f"{name}-child-columns"),
    )


def _logits(
    name: str,
    calibration_rows: int,
    validation_rows: int,
    *,
    seed: int,
) -> CanonicalParentLogits:
    generator = torch.Generator().manual_seed(seed)
    return CanonicalParentLogits(
        parent_name=name,
        calibration_logits=torch.randn(
            calibration_rows,
            3,
            generator=generator,
            dtype=torch.float64,
        ),
        validation_logits=torch.randn(
            validation_rows,
            3,
            generator=generator,
            dtype=torch.float64,
        ),
        token_contract=_contract(name),
    )


def _fixture(parent_names=("alpha",)):
    calibration_prompts = ("cal-a", "cal-b", "cal-c", "cal-d", "cal-e")
    validation_prompts = ("val-a", "val-b", "val-c", "val-d")
    calibration = tuple(
        _parent(
            name,
            split="calibration",
            prompts=calibration_prompts,
            seed=201 + index,
            width=3 + index,
            depth=1 + index,
        )
        for index, name in enumerate(parent_names)
    )
    validation = tuple(
        _parent(
            name,
            split="validation",
            prompts=validation_prompts,
            seed=211 + index,
            width=3 + index,
            depth=1 + index,
        )
        for index, name in enumerate(parent_names)
    )
    logits = tuple(
        _logits(
            name,
            len(calibration_prompts),
            len(validation_prompts),
            seed=221 + index,
        )
        for index, name in enumerate(parent_names)
    )
    return calibration, validation, logits


def _install_stubs(monkeypatch):
    calls = []

    def fake_bootstrap(source, target, calibration_inputs, **kwargs):
        calls.append((source, target, calibration_inputs, kwargs))
        with torch.no_grad():
            target.body.weight.fill_(0.125)
        return SimpleNamespace(
            transaction=SimpleNamespace(accepted=True),
            fit_reports={"stub": True},
        )

    monkeypatch.setattr(compiler_module, "build_decoder_layout", _layout)
    monkeypatch.setattr(
        compiler_module,
        "bootstrap_blank_decoder_models",
        fake_bootstrap,
    )
    return calls


def _compile(
    monkeypatch,
    calibration,
    validation,
    logits,
    *,
    weights,
    child=None,
):
    calls = _install_stubs(monkeypatch)
    carrier, target = _models()
    carrier_inputs = {"input_ids": torch.tensor([[0, 1]])}
    carrier_validation = {"input_ids": torch.tensor([[1, 2]])}
    child_inputs = {"input_ids": torch.tensor([[2, 0]])}
    child_validation = {"input_ids": torch.tensor([[0, 2]])}
    result = compile_hf_parents_to_blank_child(
        calibration,
        validation,
        logits,
        structural_carrier_model=carrier,
        blank_target_model=target,
        carrier_calibration_inputs=carrier_inputs,
        carrier_validation_inputs=carrier_validation,
        child_calibration_inputs=child_inputs,
        child_validation_inputs=child_validation,
        child_spec=_child(target) if child is None else child,
        parent_weights=weights,
        bootstrap_kwargs={"ridge": 1e-5},
    )
    return (
        result,
        calls,
        carrier,
        target,
        (
            carrier_inputs,
            carrier_validation,
            child_inputs,
            child_validation,
        ),
    )


def test_one_parent_compiles_with_exact_wiring_and_truthful_report(monkeypatch) -> None:
    calibration, validation, logits = _fixture()
    carrier_snapshot = None
    parent_snapshots = [value.detach().clone() for value in calibration[0].layer_states]
    result, calls, carrier, target, inputs = _compile(
        monkeypatch,
        calibration,
        validation,
        logits,
        weights={"alpha": 2.0},
    )
    carrier_snapshot = result.report.structural_carrier_state_sha256_before

    assert result.child_model is target
    assert result.target_model is target
    assert len(calls) == 1
    source_arg, target_arg, calibration_arg, kwargs = calls[0]
    assert source_arg is carrier
    assert target_arg is target
    assert calibration_arg is inputs[0]
    assert kwargs["validation_inputs"] is inputs[1]
    assert kwargs["target_calibration_inputs"] is inputs[2]
    assert kwargs["target_validation_inputs"] is inputs[3]
    expected = result.execution_graph.bootstrap_inputs()
    for name, value in expected.items():
        if isinstance(value, tuple):
            assert len(kwargs[name]) == len(value)
            assert all(
                actual is wanted
                for actual, wanted in zip(kwargs[name], value, strict=True)
            )
        else:
            assert kwargs[name] is value
    assert kwargs["ridge"] == pytest.approx(1e-5)
    assert (
        result.report.execution_graph_fingerprint == result.execution_graph.fingerprint
    )
    assert result.report.target_budget_exact is True
    assert result.report.target_unique_learned_weight_bytes == 40
    assert result.report.target_logical_learned_weight_bytes == 64
    assert result.report.target_unique_buffer_bytes == 0
    assert result.report.target_logical_buffer_bytes == 0
    assert result.report.target_unique_parameter_and_buffer_bytes == 40
    assert result.report.serialized_artifact_bytes_proven is False
    assert "serializer metadata" in result.report.byte_accounting_scope
    assert result.report.bootstrap_accepted is True
    assert result.report.zero_recovery is True
    assert result.report.recovery_steps == 0
    assert result.report.capability_status == "unproven"
    assert result.report.capability_proven is False
    assert result.report.whole_model_exactness_proven is False
    assert result.report.parent_models_invoked is False
    assert result.report.structural_carrier_unchanged is True
    assert result.report.parent_payloads_unchanged is True
    assert result.report.structural_carrier_state_sha256_after == carrier_snapshot
    assert len(result.report.parent_contributions) == 1
    assert result.report.parent_contributions[0].normalized_weight == pytest.approx(1.0)
    assert torch.equal(target.body.weight, torch.full_like(target.body.weight, 0.125))
    for before, after in zip(
        parent_snapshots,
        calibration[0].layer_states,
        strict=True,
    ):
        assert torch.equal(before, after)
    with pytest.raises(FrozenInstanceError):
        result.report.capability_proven = True
    with pytest.raises(FrozenInstanceError):
        result.report.parent_contributions[0].normalized_weight = 0.0
    with pytest.raises(FrozenInstanceError):
        result.child_model = carrier


def test_two_parent_compilation_is_order_invariant(monkeypatch) -> None:
    calibration, validation, logits = _fixture(("zeta", "alpha"))
    first, _, _, _, _ = _compile(
        monkeypatch,
        calibration,
        validation,
        logits,
        weights={"zeta": 3.0, "alpha": 1.0},
    )
    second, _, _, _, _ = _compile(
        monkeypatch,
        tuple(reversed(calibration)),
        tuple(reversed(validation)),
        tuple(reversed(logits)),
        weights={"alpha": 1.0, "zeta": 3.0},
    )

    assert first.execution_graph.fingerprint == second.execution_graph.fingerprint
    assert first.report.parent_contributions == second.report.parent_contributions
    assert tuple(value.parent_name for value in first.report.parent_contributions) == (
        "alpha",
        "zeta",
    )
    assert tuple(
        value.normalized_weight for value in first.report.parent_contributions
    ) == pytest.approx((0.25, 0.75))
    torch.testing.assert_close(
        first.execution_graph.calibration_target_log_probs,
        second.execution_graph.calibration_target_log_probs,
        atol=0.0,
        rtol=0.0,
    )


def test_exact_budget_mismatch_fails_before_bootstrap(monkeypatch) -> None:
    calibration, validation, logits = _fixture()
    calls = _install_stubs(monkeypatch)
    carrier, target = _models()

    with pytest.raises(HFParentCompilerError, match="unique parameter count"):
        compile_hf_parents_to_blank_child(
            calibration,
            validation,
            logits,
            structural_carrier_model=carrier,
            blank_target_model=target,
            carrier_calibration_inputs={"input_ids": torch.ones((1, 1))},
            carrier_validation_inputs={"input_ids": torch.ones((1, 1))},
            child_calibration_inputs={"input_ids": torch.ones((1, 1))},
            child_validation_inputs={"input_ids": torch.ones((1, 1))},
            child_spec=_child(target, budget_delta=1),
            parent_weights={"alpha": 1.0},
        )

    assert calls == []


def test_failed_bootstrap_restores_the_exact_blank_target(monkeypatch) -> None:
    calibration, validation, logits = _fixture()
    monkeypatch.setattr(compiler_module, "build_decoder_layout", _layout)
    carrier, target = _models()
    before = {
        name: value.detach().clone() for name, value in target.state_dict().items()
    }
    body_parameter = target.body.weight

    def mutate_then_fail(_source, failed_target, _inputs, **_kwargs):
        with torch.no_grad():
            failed_target.embedding.weight.fill_(9.0)
            failed_target.body.weight.fill_(-7.0)
        raise RuntimeError("synthetic bootstrap failure")

    monkeypatch.setattr(
        compiler_module,
        "bootstrap_blank_decoder_models",
        mutate_then_fail,
    )

    with pytest.raises(RuntimeError, match="synthetic bootstrap failure"):
        compile_hf_parents_to_blank_child(
            calibration,
            validation,
            logits,
            structural_carrier_model=carrier,
            blank_target_model=target,
            carrier_calibration_inputs={},
            carrier_validation_inputs={},
            child_calibration_inputs={},
            child_validation_inputs={},
            child_spec=_child(target),
            parent_weights={"alpha": 1.0},
        )

    assert target.body.weight is body_parameter
    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_stale_or_misaligned_split_rows_fail_closed(monkeypatch) -> None:
    calibration, validation, logits = _fixture()
    calls = _install_stubs(monkeypatch)
    carrier, target = _models()
    stale_validation = (
        _parent(
            "alpha",
            split="validation",
            prompts=("val-a", "val-b", "val-c", "val-d"),
            identity_prompts=("cal-a", "cal-b", "cal-c", "cal-d"),
            seed=301,
            width=3,
            depth=1,
        ),
    )

    with pytest.raises(HFParentCompilerError, match="stale or overlap"):
        compile_hf_parents_to_blank_child(
            calibration,
            stale_validation,
            logits,
            structural_carrier_model=carrier,
            blank_target_model=target,
            carrier_calibration_inputs={},
            carrier_validation_inputs={},
            child_calibration_inputs={},
            child_validation_inputs={},
            child_spec=_child(target),
            parent_weights={"alpha": 1.0},
        )

    second = _parent(
        "beta",
        split="calibration",
        prompts=("other-a", "other-b", "other-c", "other-d", "other-e"),
        seed=302,
        width=4,
        depth=1,
    )
    second_validation = _parent(
        "beta",
        split="validation",
        prompts=("val-a", "val-b", "val-c", "val-d"),
        seed=303,
        width=4,
        depth=1,
    )
    with pytest.raises(HFParentCompilerError, match="misaligned"):
        compile_hf_parents_to_blank_child(
            (calibration[0], second),
            (validation[0], second_validation),
            (logits[0], _logits("beta", 5, 4, seed=304)),
            structural_carrier_model=carrier,
            blank_target_model=target,
            carrier_calibration_inputs={},
            carrier_validation_inputs={},
            child_calibration_inputs={},
            child_validation_inputs={},
            child_spec=_child(target),
            parent_weights={"alpha": 1.0, "beta": 1.0},
        )

    assert calls == []


class _ExactTokenizer:
    def __init__(self, rows: dict[int, str]) -> None:
        self.rows = rows

    def decode(self, token_ids, **_kwargs):
        return "".join(self.rows[int(value)] for value in token_ids)


def _byte_trace(
    prompts: tuple[str, ...],
    *,
    seed: int,
    token_offset: int,
) -> ByteSpanTraceInput:
    ids = torch.arange(token_offset, token_offset + len(prompts)).reshape(-1, 1)
    offsets = torch.tensor([[[0, len(prompt)]] for prompt in prompts])
    mask = torch.ones_like(ids)
    generator = torch.Generator().manual_seed(seed)
    hidden = tuple(
        torch.randn(
            len(prompts),
            1,
            3,
            generator=generator,
            dtype=torch.float64,
        )
        for _ in range(2)
    )
    return ByteSpanTraceInput(
        parent_name="alpha",
        raw_prompts=prompts,
        tokenizer=_ExactTokenizer(
            {token_offset + index: prompt for index, prompt in enumerate(prompts)}
        ),
        input_ids=ids,
        offset_mapping=offsets,
        attention_mask=mask,
        hidden_states=hidden,
        tokenizer_fingerprint=_digest("alpha-tokenizer"),
        cache_fingerprint=_digest(f"alpha-byte-cache-{seed}"),
    )


def test_byte_span_trace_sets_use_the_alignment_helper_path(monkeypatch) -> None:
    calibration_prompts = ("cal-one", "cal-two")
    validation_prompts = ("val-one", "val-two")
    calibration = (_byte_trace(calibration_prompts, seed=401, token_offset=10),)
    validation = (_byte_trace(validation_prompts, seed=402, token_offset=20),)
    logits = (_logits("alpha", 2, 2, seed=403),)
    result, _, _, _, _ = _compile(
        monkeypatch,
        calibration,
        validation,
        logits,
        weights={"alpha": 1.0},
    )

    assert result.report.parent_input_kind == "byte-span"
    assert result.report.parent_contributions[0].calibration_row_count == 2
    assert result.report.parent_contributions[0].validation_row_count == 2
    assert result.report.parent_payloads_unchanged is True
