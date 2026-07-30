from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_semantic_step_model_runtime import (
    LFM2_SEMANTIC_STEP_MODEL_PROGRESS_SCHEMA,
    Lfm2SemanticStepModelRuntimeError,
    run_lfm2_semantic_step_model_review,
)
from neural_forge.hf_lfm2_semantic_step_token_fold import (
    bind_lfm2_semantic_step_hard_anchor_runtime,
    bind_lfm2_separate_sparse_semantic_code_runtime,
    bind_lfm2_sparse_semantic_code_runtime,
    seal_lfm2_semantic_step_span_evidence,
)


class _FakeLiveTableSession:
    def __init__(self, owner: "_FakeRouting", table: torch.Tensor) -> None:
        self.owner = owner
        self.table = table
        self.full_hash_verification_count = 0
        self.closed = False

    def __enter__(self) -> "_FakeLiveTableSession":
        assert not self.closed
        self.full_hash_verification_count = 1
        self.owner.session_entries += 1
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.full_hash_verification_count += 1
        self.closed = True
        self.owner.session_exits += 1

    def as_dict(self):
        return {
            "schema": "fake-live-table-session-v1",
            "closed": self.closed,
            "full_hash_verification_count": self.full_hash_verification_count,
        }


class _FakeRouting:
    def __init__(self, *, always_donor_id: int | None = None) -> None:
        self.donor_row_count = 4
        self.target_row_count = 5
        self.hidden_size = 2
        self._spans = ((0,), (3,), (0, 1, 2), (2,))
        self.always_donor_id = always_donor_id
        self.session_entries = 0
        self.session_exits = 0
        self.score_widths: list[int] = []
        self.reconstructed_ids: list[tuple[int, ...]] = []

    def as_dict(self):
        return {
            "schema": "fake-parameterless-semantic-routing-v1",
            "donor_row_count": self.donor_row_count,
            "target_row_count": self.target_row_count,
            "hidden_size": self.hidden_size,
            "owned_learned_parameter_count": 0,
            "owns_or_registers_target_table": False,
            "routing_metadata_is_integer_only": True,
        }

    def authenticated_live_table_session(self, table: torch.Tensor):
        return _FakeLiveTableSession(self, table)

    def span_for_donor(self, donor_id: int):
        return self._spans[donor_id]

    @staticmethod
    def _table(session: _FakeLiveTableSession) -> torch.Tensor:
        assert isinstance(session, _FakeLiveTableSession)
        assert not session.closed
        assert session.full_hash_verification_count == 1
        return session.table

    def reconstruct_selected_donor_embeddings(
        self,
        session: _FakeLiveTableSession,
        donor_ids,
        *,
        maximum_rows_per_chunk: int,
    ) -> torch.Tensor:
        del maximum_rows_per_chunk
        table = self._table(session)
        values = tuple(int(value) for value in donor_ids)
        self.reconstructed_ids.append(values)
        rows = []
        for donor_id in values:
            ids = torch.tensor(self._spans[donor_id], device=table.device)
            rows.append(table.index_select(0, ids).mean(dim=0))
        return torch.stack(rows) if rows else table.new_empty((0, self.hidden_size))

    def score_target_tokens(
        self,
        session: _FakeLiveTableSession,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        table = self._table(session)
        scores = table.new_zeros((*hidden_states.shape[:-1], self.target_row_count))
        scores[..., 0] = hidden_states[..., 0].to(dtype=table.dtype)
        return scores

    def compute_all_donor_logits_from_target_scores(
        self,
        session: _FakeLiveTableSession,
        target_scores: torch.Tensor,
        *,
        maximum_donor_rows_per_chunk: int,
    ) -> torch.Tensor:
        del maximum_donor_rows_per_chunk
        table = self._table(session)
        self.score_widths.append(self.donor_row_count)
        result = table.new_full(
            (*target_scores.shape[:-1], self.donor_row_count),
            -100.0,
        )
        if self.always_donor_id is not None:
            result[..., self.always_donor_id] = 100.0
        else:
            early = target_scores[..., 0] < 0.5
            result[..., 2] = torch.where(
                early,
                result.new_tensor(100.0),
                result.new_tensor(-100.0),
            )
            result[..., 1] = torch.where(
                early,
                result.new_tensor(-100.0),
                result.new_tensor(100.0),
            )
        return result


class _FakeBody(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        past_key_values=None,
    ):
        assert use_cache is True
        assert inputs_embeds.shape[0] == 1
        assert inputs_embeds.shape[2] == self.hidden_size
        assert attention_mask.shape[1] == (
            inputs_embeds.shape[1]
            if past_key_values is None
            else int(past_key_values) + inputs_embeds.shape[1]
        )
        if past_key_values is None:
            next_cache = int(inputs_embeds.shape[1])
            signal = 0.0
        else:
            next_cache = int(past_key_values) + int(inputs_embeds.shape[1])
            generated_steps = next_cache - (int(attention_mask.shape[1]) - 1)
            del generated_steps
            signal = float(len(self.calls))
        hidden = inputs_embeds.new_zeros(inputs_embeds.shape)
        # Prefill predicts the first response token.  Every one-token decode
        # call increments the signal used by the fake route.
        hidden[..., 0] = (
            0.0
            if past_key_values is None
            else float(sum(1 for call in self.calls if call["past"] is not None) + 1)
        )
        self.calls.append(
            {
                "sequence_length": int(inputs_embeds.shape[1]),
                "attention_length": int(attention_mask.shape[1]),
                "past": past_key_values,
                "dtype": inputs_embeds.dtype,
                "device": inputs_embeds.device,
                "signal": signal,
            }
        )
        return SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=next_cache,
        )


class _FakeChild(torch.nn.Module):
    def __init__(
        self, *, dtype: torch.dtype = torch.float32, tied: bool = True
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 2, dtype=dtype)
        self.lm_head = torch.nn.Linear(2, 5, bias=False, dtype=dtype)
        if tied:
            self.lm_head.weight = self.embedding.weight
        self.model = _FakeBody(hidden_size=2)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [2.0, 0.0],
                        [3.0, 0.0],
                        [4.0, 0.0],
                    ],
                    dtype=dtype,
                )
            )
        self.generate_invocations = 0

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def generate(self, *args, **kwargs):  # pragma: no cover - must never run
        self.generate_invocations += 1
        raise AssertionError("semantic-step runtime must not call generate()")


class _DonorTokenizer:
    # Deliberately larger nominal capacity: only get_vocab() rows are defined.
    vocab_size = 6

    def __init__(self, *, eos_token_id=1) -> None:
        self.eos_token_id = eos_token_id
        self.prompt_calls = 0
        self.decode_calls = 0

    def get_vocab(self):
        return {"<prompt>": 0, "<eos>": 1, "x": 2, "<user>": 3}

    def __call__(self, text, *, return_tensors: str, add_special_tokens: bool):
        assert isinstance(text, str)
        assert return_tensors == "pt"
        assert add_special_tokens is True
        self.prompt_calls += 1
        return {
            "input_ids": torch.tensor([[0, 3]], dtype=torch.int64),
            "attention_mask": torch.ones((1, 2), dtype=torch.int64),
        }

    def decode(self, ids, *, skip_special_tokens: bool):
        assert skip_special_tokens is True
        self.decode_calls += 1
        return "".join("x" for value in ids if int(value) == 2)


class _TargetTokenizer:
    def __init__(self) -> None:
        self.encode_calls: list[str] = []
        self.decode_calls: list[tuple[int, ...]] = []
        self._last_text = ""

    def get_vocab(self):
        return {"a": 0, "b": 1, "c": 2, "d": 3, "whole": 4}

    def encode(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        self.encode_calls.append(text)
        self._last_text = text
        return [] if not text else [4]

    def decode(self, ids, *, skip_special_tokens: bool):
        assert skip_special_tokens is True
        self.decode_calls.append(tuple(int(value) for value in ids))
        return self._last_text


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_semantic_runtime_advances_once_per_donor_action_and_retokenizes_whole_text(
    dtype: torch.dtype,
) -> None:
    child = _FakeChild(dtype=dtype)
    routing = _FakeRouting()
    donor = _DonorTokenizer()
    target = _TargetTokenizer()
    progress = []
    before_parameter_ids = tuple(id(parameter) for parameter in child.parameters())
    before_named = tuple(child.named_parameters(remove_duplicate=False))

    result = run_lfm2_semantic_step_model_review(
        child,
        routing,
        donor,
        target,
        [{"id": "hello", "category": "greeting", "prompt": "Hello"}],
        progress_callback=progress.append,
    )

    response = result["responses"][0]
    assert response["raw_donor_response_token_ids"] == [2, 1]
    # The first donor action has a three-target-token spelling, but only one
    # one-token body call advances it.  EOS stops without an unused cache call.
    assert response["semantic_steps"][0]["target_span_token_ids"] == [0, 1, 2]
    assert response["semantic_steps"][0]["body_semantic_tokens_advanced"] == 1
    assert [call["sequence_length"] for call in child.model.calls] == [2, 1]
    assert all(call["dtype"] == dtype for call in child.model.calls)
    assert response["diagnostics"]["body_generated_forward_calls"] == 1
    assert response["semantic_steps"][1]["body_forward_calls"] == 0
    assert (
        response["semantic_steps"][1]["terminal_token_not_fed_to_unused_cache"] is True
    )
    assert response["diagnostics"]["target_span_tokens_used_as_body_steps"] == 0

    # Concatenating routed spans would yield [0, 1, 2, 3].  The runtime instead
    # decodes the complete donor response and invokes target encode exactly once.
    assert target.encode_calls == ["x"]
    assert response["final_target_response_token_ids"] == [4]
    assert response["response_token_ids"] == [4]
    assert response["final_target_response_text"] == "x"
    assert response["target_roundtrip_exact"] is True
    assert result["whole_response_target_retokenization"] is True

    # Defined donor IDs are 0..3 even though nominal tokenizer capacity is six.
    assert routing.score_widths == [4, 4]
    assert result["donor_defined_row_count"] == 4
    assert result["undefined_donor_capacity_rows_scored"] == 0
    assert result["operation_audit"]["undefined_donor_rows_scored"] == 0

    # The routing hashes once on entry and once on exit, and neither the model
    # nor the runtime gains parameters/modules.
    assert routing.session_entries == 1
    assert routing.session_exits == 1
    assert result["routing_authentication_mode"] == "hash-once-live-table-session"
    assert result["routing_full_hash_verification_count"] == 2
    assert (
        tuple(id(parameter) for parameter in child.parameters()) == before_parameter_ids
    )
    assert tuple(child.named_parameters(remove_duplicate=False)) == before_named
    assert child.generate_invocations == 0
    assert result["operation_audit"]["parent_forward_calls"] == 0
    assert result["operation_audit"]["training_steps"] == 0
    assert result["operation_audit"]["owned_learned_parameter_count"] == 0

    assert progress[0]["stage"] == "runtime-start"
    assert progress[-1]["stage"] == "runtime-complete"
    assert sum(row["stage"] == "semantic-step-complete" for row in progress) == 2
    previous = None
    for ordinal, immutable in enumerate(progress):
        assert immutable["schema"] == LFM2_SEMANTIC_STEP_MODEL_PROGRESS_SCHEMA
        assert immutable["ordinal"] == ordinal
        assert immutable["previous_progress_sha256"] == previous
        plain = dict(immutable)
        supplied = plain.pop("progress_sha256")
        assert supplied == canonical_sha256(plain)
        previous = supplied
        with pytest.raises(TypeError):
            immutable["stage"] = "tampered"
    assert result["progress_record_count"] == len(progress)
    assert result["progress_final_sha256"] == previous
    body = dict(result)
    supplied = body.pop("runtime_sha256")
    assert supplied == canonical_sha256(body)


def test_generation_can_cross_sixty_four_without_using_span_tokens_as_body_clock() -> (
    None
):
    assert (
        inspect.signature(run_lfm2_semantic_step_model_review)
        .parameters["maximum_new_tokens"]
        .default
        == 256
    )
    child = _FakeChild()
    routing = _FakeRouting(always_donor_id=2)
    donor = _DonorTokenizer(eos_token_id=None)
    target = _TargetTokenizer()
    progress = []

    result = run_lfm2_semantic_step_model_review(
        child,
        routing,
        donor,
        target,
        [{"id": "long", "prompt": "Continue"}],
        maximum_new_tokens=65,
        progress_callback=progress.append,
    )

    response = result["responses"][0]
    assert len(response["raw_donor_response_token_ids"]) == 65
    assert response["diagnostics"]["generation_limit_reached"] is True
    assert response["diagnostics"]["body_generated_forward_calls"] == 64
    assert len(child.model.calls) == 65
    assert all(call["sequence_length"] == 1 for call in child.model.calls[1:])
    assert result["operation_audit"]["generated_donor_semantic_tokens_advanced"] == 64
    assert result["operation_audit"]["target_span_metadata_tokens_observed"] == 195
    assert result["operation_audit"]["target_span_tokens_used_as_body_steps"] == 0
    assert target.encode_calls == ["x" * 65]
    assert sum(row["stage"] == "semantic-step-complete" for row in progress) == 65


def test_runtime_fails_closed_on_untied_child_and_noncontiguous_defined_vocab() -> None:
    with pytest.raises(Lfm2SemanticStepModelRuntimeError, match="tied"):
        run_lfm2_semantic_step_model_review(
            _FakeChild(tied=False),
            _FakeRouting(),
            _DonorTokenizer(),
            _TargetTokenizer(),
            [{"id": "x", "prompt": "x"}],
        )

    class _HoleyDonor(_DonorTokenizer):
        def get_vocab(self):
            return {"zero": 0, "two": 2, "three": 3, "four": 4}

    with pytest.raises(Lfm2SemanticStepModelRuntimeError, match="contiguous"):
        run_lfm2_semantic_step_model_review(
            _FakeChild(),
            _FakeRouting(),
            _HoleyDonor(),
            _TargetTokenizer(),
            [{"id": "x", "prompt": "x"}],
        )


def test_runtime_rejects_logits_that_include_undefined_capacity_rows() -> None:
    class _CapacityScoringRouting(_FakeRouting):
        def compute_all_donor_logits_from_target_scores(
            self,
            session,
            target_scores,
            *,
            maximum_donor_rows_per_chunk,
        ):
            del maximum_donor_rows_per_chunk
            table = self._table(session)
            return table.new_zeros((int(target_scores.shape[0]), 6))

    with pytest.raises(Lfm2SemanticStepModelRuntimeError, match="exactly"):
        run_lfm2_semantic_step_model_review(
            _FakeChild(),
            _CapacityScoringRouting(),
            _DonorTokenizer(),
            _TargetTokenizer(),
            [{"id": "x", "prompt": "x"}],
        )


def test_runtime_consumes_the_real_parameterless_hash_once_routing_api() -> None:
    child = _FakeChild()
    evidence = seal_lfm2_semantic_step_span_evidence(
        [[0], [3], [0, 1, 2], [2]],
        target_row_count=5,
    )
    routing = bind_lfm2_semantic_step_hard_anchor_runtime(
        evidence,
        child.embedding.weight,
    )

    result = run_lfm2_semantic_step_model_review(
        child,
        routing,
        _DonorTokenizer(eos_token_id=None),
        _TargetTokenizer(),
        [{"id": "real-routing", "prompt": "x"}],
        maximum_new_tokens=1,
    )

    assert result["routing_authentication_mode"] == "hash-once-live-table-session"
    assert result["routing_full_hash_verification_count"] == 2
    assert result["routing_session_receipt"]["full_hash_verification_count"] == 2
    assert result["operation_audit"]["child_body_generated_forward_calls"] == 0
    assert result["operation_audit"]["target_span_tokens_used_as_body_steps"] == 0


def test_runtime_counts_sparse_semantic_coefficients_exactly_once() -> None:
    class _FiveDonorTokenizer(_DonorTokenizer):
        def __init__(self) -> None:
            super().__init__(eos_token_id=None)

        def get_vocab(self):
            return {
                "<prompt>": 0,
                "<eos>": 1,
                "x": 2,
                "<user>": 3,
                "y": 4,
            }

    child = _FakeChild()
    evidence = seal_lfm2_semantic_step_span_evidence(
        [[0], [1], [2], [3], [4]],
        target_row_count=5,
    )
    routing = bind_lfm2_sparse_semantic_code_runtime(
        evidence,
        child.embedding.weight,
        semantic_code_offsets=torch.arange(6, dtype=torch.int64),
        semantic_code_target_ids=torch.arange(5, dtype=torch.int64),
        semantic_code_coefficients=torch.ones(5, dtype=torch.float32),
        target_to_donor=torch.arange(5, dtype=torch.int64),
    )
    result = run_lfm2_semantic_step_model_review(
        child,
        routing,
        _FiveDonorTokenizer(),
        _TargetTokenizer(),
        [{"id": "counted", "prompt": "x"}],
        maximum_new_tokens=1,
    )
    assert result["child_unique_parameter_count"] == 10
    assert result["routing_fixed_continuous_state_count"] == 5
    assert result["effective_child_parameter_count"] == 15
    assert (
        result["effective_child_parameter_count_includes_semantic_code_coefficients"]
        is True
    )
    assert result["operation_audit"]["routing_fixed_continuous_state_count"] == 5
    assert result["operation_audit"]["effective_child_parameter_count"] == 15
    assert (
        result["operation_audit"]["semantic_code_coefficients_counted_exactly_once"]
        is True
    )


def test_runtime_counts_separate_input_output_codes_exactly_once() -> None:
    class _FiveDonorTokenizer(_DonorTokenizer):
        def __init__(self) -> None:
            super().__init__(eos_token_id=None)

        def get_vocab(self):
            return {
                "<prompt>": 0,
                "<eos>": 1,
                "x": 2,
                "<user>": 3,
                "y": 4,
            }

    child = _FakeChild()
    evidence = seal_lfm2_semantic_step_span_evidence(
        [[0], [1], [2], [3], [4]], target_row_count=5
    )
    offsets = torch.arange(6, dtype=torch.int64)
    ids = torch.arange(5, dtype=torch.int64)
    coefficients = torch.ones(5, dtype=torch.float32)
    routing = bind_lfm2_separate_sparse_semantic_code_runtime(
        evidence,
        child.embedding.weight,
        input_semantic_code_offsets=offsets,
        input_semantic_code_target_ids=ids,
        input_semantic_code_coefficients=coefficients,
        output_semantic_code_offsets=offsets,
        output_semantic_code_target_ids=ids,
        output_semantic_code_coefficients=coefficients.clone(),
        target_to_donor=torch.arange(5, dtype=torch.int64),
    )
    result = run_lfm2_semantic_step_model_review(
        child,
        routing,
        _FiveDonorTokenizer(),
        _TargetTokenizer(),
        [{"id": "separate-counted", "prompt": "x"}],
        maximum_new_tokens=1,
    )
    assert result["child_unique_parameter_count"] == 10
    assert result["routing_input_fixed_continuous_state_count"] == 5
    assert result["routing_output_fixed_continuous_state_count"] == 5
    assert result["routing_fixed_continuous_state_count"] == 10
    assert result["effective_child_parameter_count"] == 20
    assert result["routing_input_output_coefficient_storage_shared"] is False
    audit = result["operation_audit"]
    assert audit["routing_input_fixed_continuous_state_count"] == 5
    assert audit["routing_output_fixed_continuous_state_count"] == 5
    assert audit["routing_fixed_continuous_state_count"] == 10


def test_target_roundtrip_mismatch_makes_actual_target_decode_authoritative() -> None:
    class _NormalizingTarget(_TargetTokenizer):
        def decode(self, ids, *, skip_special_tokens: bool):
            assert skip_special_tokens is True
            self.decode_calls.append(tuple(int(value) for value in ids))
            return "normalized"

    result = run_lfm2_semantic_step_model_review(
        _FakeChild(),
        _FakeRouting(),
        _DonorTokenizer(),
        _NormalizingTarget(),
        [{"id": "mismatch", "prompt": "x"}],
    )
    response = result["responses"][0]
    assert response["donor_response_text"] == "x"
    assert response["final_target_response_text"] == "normalized"
    assert response["response"] == "normalized"
    assert response["target_roundtrip_exact"] is False
    assert response["diagnostics"]["target_roundtrip_changed_text"] is True
    assert response["diagnostics"]["authoritative_response_text_source"] == (
        "whole-target-sequence-retokenization-decode"
    )


def test_hash_once_session_is_required_by_default() -> None:
    class _LegacyRouting(_FakeRouting):
        authenticated_live_table_session = None

    with pytest.raises(
        Lfm2SemanticStepModelRuntimeError,
        match="authenticated_live_table_session",
    ):
        run_lfm2_semantic_step_model_review(
            _FakeChild(),
            _LegacyRouting(),
            _DonorTokenizer(),
            _TargetTokenizer(),
            [{"id": "legacy", "prompt": "x"}],
        )
