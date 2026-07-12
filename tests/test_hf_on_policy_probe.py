from __future__ import annotations

import copy
import hashlib
import json
from types import MappingProxyType

import pytest

from neural_forge.hf_on_policy_probe import (
    GreedyProbeSettings,
    OnPolicyProbeConflictError,
    OnPolicyProbeError,
    OnPolicyProbeIntegrityError,
    build_on_policy_probe_corpus,
    load_on_policy_generation_cache,
)


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


class ByteTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.encode_calls = 0
        self.decode_calls = 0

    def __len__(self) -> int:
        return 258

    def get_vocab(self):
        return {
            "<pad>": 0,
            "<eos>": 1,
            **{f"byte-{value:03d}": value + 2 for value in range(256)},
        }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        self.encode_calls += 1
        return [value + 2 for value in text.encode("utf-8")]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        self.decode_calls += 1
        return bytes(token_id - 2 for token_id in token_ids).decode("utf-8")


class NibbleTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self) -> int:
        return 18

    def get_vocab(self):
        return {
            "<pad>": 0,
            "<eos>": 1,
            **{f"nibble-{value:x}": value + 2 for value in range(16)},
        }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        result = []
        for value in text.encode("utf-8"):
            result.extend((2 + (value >> 4), 2 + (value & 15)))
        return result

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        if len(token_ids) % 2:
            raise ValueError("nibble sequence must be even")
        values = []
        for index in range(0, len(token_ids), 2):
            values.append(((token_ids[index] - 2) << 4) | (token_ids[index + 1] - 2))
        return bytes(values).decode("utf-8")


def _child_model(torch):
    class TinyChild(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.25]))
            self.generate_calls = 0
            self.grad_enabled = []
            self.training_seen = []

        def generate(
            self,
            input_ids,
            attention_mask,
            max_new_tokens,
            do_sample,
            num_beams,
            **_kwargs,
        ):
            self.generate_calls += 1
            self.grad_enabled.append(torch.is_grad_enabled())
            self.training_seen.append(self.training)
            assert do_sample is False
            assert num_beams == 1
            assert attention_mask.tolist() == [[1] * int(input_ids.shape[1])]
            suffix_values = "!?".encode("utf-8")[:max_new_tokens]
            suffix = torch.tensor(
                [[value + 2 for value in suffix_values]],
                dtype=torch.long,
                device=input_ids.device,
            )
            return torch.cat((input_ids, suffix), dim=1)

    return TinyChild()


def _settings(**changes):
    values = {
        "max_new_tokens": 2,
        "max_context_tokens": 64,
        "trace_batch_size": 2,
        "seed": 17,
        "strict_byte_identity": True,
        "include_seed_prefix": True,
    }
    values.update(changes)
    return GreedyProbeSettings(**values)


def test_cross_tokenizer_rows_are_aligned_by_exact_utf8_hash_and_batched() -> None:
    torch = pytest.importorskip("torch")
    child = _child_model(torch)
    child.train()
    child.weight.grad = torch.tensor([0.5])
    child_tokenizer = ByteTokenizer()
    parent_tokenizer = NibbleTokenizer()
    parameter_before = child.weight.detach().clone()
    gradient_before = child.weight.grad.clone()

    corpus = build_on_policy_probe_corpus(
        child,
        child_tokenizer,
        {"parent-nibble": parent_tokenizer},
        ["A", "B"],
        settings=_settings(),
    )

    assert len(corpus.rows) == 6
    assert tuple(corpus.tokenizations) == ("child", "parent-nibble")
    expected_ids = tuple(row.row_id for row in corpus.rows)
    assert tuple(row.row_id for row in corpus.tokenizations["child"]) == expected_ids
    assert (
        tuple(row.row_id for row in corpus.tokenizations["parent-nibble"])
        == expected_ids
    )
    assert all(
        row.row_id == f"utf8-sha256:{hashlib.sha256(row.utf8_bytes).hexdigest()}"
        and row.text.encode("utf-8") == row.utf8_bytes
        for row in corpus.rows
    )
    assert all(
        len(parent.token_ids) == 2 * len(child_row.token_ids)
        for child_row, parent in zip(
            corpus.tokenizations["child"],
            corpus.tokenizations["parent-nibble"],
        )
    )

    batch = corpus.batches_for("child")[0]
    first = batch.as_trace_cache_batch(torch_module=torch)
    second = batch.as_trace_cache_batch(torch_module=torch)
    assert isinstance(first, MappingProxyType)
    assert first["input_ids"].tolist() == [list(row) for row in batch.input_ids]
    assert first["attention_mask"].tolist() == [
        list(row) for row in batch.attention_mask
    ]
    assert first["input_ids"] is not second["input_ids"]
    assert child.generate_calls == 2
    assert child.grad_enabled == [False, False]
    assert child.training_seen == [False, False]
    assert child.training is True
    torch.testing.assert_close(child.weight, parameter_before)
    torch.testing.assert_close(child.weight.grad, gradient_before)
    assert corpus.report["execution"]["optimizer_steps"] == 0
    assert corpus.report["execution"]["backward_calls"] == 0
    assert corpus.report["execution"]["parent_forward_calls"] == 0
    assert corpus.report["immutability"]["child_parameters_unchanged"] is True
    with pytest.raises(TypeError):
        corpus.report["schema"] = "changed"


def test_generation_cache_is_reused_without_regenerating_and_deduplicates(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    child = _child_model(torch)
    child_tokenizer = ByteTokenizer()
    cache_path = tmp_path / "child-prefixes.json"

    first = build_on_policy_probe_corpus(
        child,
        child_tokenizer,
        {"parent": NibbleTokenizer()},
        ["same", "same"],
        settings=_settings(max_new_tokens=1),
        generation_cache_path=cache_path,
    )
    calls_after_first = child.generate_calls
    second = build_on_policy_probe_corpus(
        child,
        child_tokenizer,
        {"parent": NibbleTokenizer()},
        ["same", "same"],
        settings=_settings(max_new_tokens=1),
        generation_cache_path=cache_path,
    )

    assert calls_after_first == 2
    assert child.generate_calls == calls_after_first
    assert tuple(row.row_id for row in first.rows) == tuple(
        row.row_id for row in second.rows
    )
    assert len(first.rows) == 2
    assert [len(row.origins) for row in first.rows] == [2, 2]
    assert first.report["generation_cache"]["hit"] is False
    assert first.report["generation_cache"]["generation_calls"] == 2
    assert second.report["generation_cache"]["hit"] is True
    assert second.report["generation_cache"]["generation_calls"] == 0
    loaded = load_on_policy_generation_cache(cache_path)
    assert isinstance(loaded, MappingProxyType)
    assert loaded["cache_sha256"] == second.report["generation_cache"]["cache_sha256"]


def test_generation_cache_fails_closed_on_tampering(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    child = _child_model(torch)
    cache_path = tmp_path / "tampered.json"
    kwargs = {
        "settings": _settings(max_new_tokens=1),
        "generation_cache_path": cache_path,
    }
    build_on_policy_probe_corpus(
        child, ByteTokenizer(), {"parent": NibbleTokenizer()}, ["A"], **kwargs
    )
    value = json.loads(cache_path.read_text(encoding="utf-8"))
    value["records"][0]["text"] = "tampered"
    cache_path.write_bytes(_canonical_json(value) + b"\n")

    with pytest.raises(OnPolicyProbeIntegrityError, match="cache_sha256"):
        build_on_policy_probe_corpus(
            child, ByteTokenizer(), {"parent": NibbleTokenizer()}, ["A"], **kwargs
        )


def test_generation_cache_rejects_duplicate_rows_even_with_resealed_hashes(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    child = _child_model(torch)
    cache_path = tmp_path / "duplicate.json"
    kwargs = {
        "settings": _settings(max_new_tokens=1),
        "generation_cache_path": cache_path,
    }
    build_on_policy_probe_corpus(
        child, ByteTokenizer(), {"parent": NibbleTokenizer()}, ["A"], **kwargs
    )
    value = json.loads(cache_path.read_text(encoding="utf-8"))
    duplicate = copy.deepcopy(value["records"][0])
    duplicate["row_index"] = len(value["records"])
    value["records"].append(duplicate)
    value["record_count"] = len(value["records"])
    value["records_sha256"] = _sha256(value["records"])
    core = {key: item for key, item in value.items() if key != "cache_sha256"}
    value["cache_sha256"] = _sha256(core)
    cache_path.write_bytes(_canonical_json(value) + b"\n")

    with pytest.raises(OnPolicyProbeIntegrityError, match="duplicate rows"):
        build_on_policy_probe_corpus(
            child, ByteTokenizer(), {"parent": NibbleTokenizer()}, ["A"], **kwargs
        )


def test_cache_identity_rejects_different_prompts_without_regeneration(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    child = _child_model(torch)
    cache_path = tmp_path / "conflict.json"
    build_on_policy_probe_corpus(
        child,
        ByteTokenizer(),
        {"parent": NibbleTokenizer()},
        ["A"],
        settings=_settings(max_new_tokens=1),
        generation_cache_path=cache_path,
    )
    calls = child.generate_calls

    with pytest.raises(OnPolicyProbeConflictError, match="different child"):
        build_on_policy_probe_corpus(
            child,
            ByteTokenizer(),
            {"parent": NibbleTokenizer()},
            ["B"],
            settings=_settings(max_new_tokens=1),
            generation_cache_path=cache_path,
        )
    assert child.generate_calls == calls


def test_strict_byte_identity_rejects_lossy_parent_tokenizer() -> None:
    torch = pytest.importorskip("torch")

    class LossyTokenizer(ByteTokenizer):
        def encode(self, text, add_special_tokens=False):
            return super().encode(text.lower(), add_special_tokens=add_special_tokens)

    with pytest.raises(OnPolicyProbeIntegrityError, match="loses UTF-8 bytes"):
        build_on_policy_probe_corpus(
            _child_model(torch),
            ByteTokenizer(),
            {"lossy": LossyTokenizer()},
            ["UPPER"],
            settings=_settings(max_new_tokens=1),
        )


def test_context_overflow_fails_before_child_generation() -> None:
    torch = pytest.importorskip("torch")
    child = _child_model(torch)
    with pytest.raises(OnPolicyProbeError, match="plus max_new_tokens exceeds"):
        build_on_policy_probe_corpus(
            child,
            ByteTokenizer(),
            {"parent": NibbleTokenizer()},
            ["AB"],
            settings=_settings(max_new_tokens=2, max_context_tokens=3),
        )
    assert child.generate_calls == 0


def test_parent_context_overflow_and_vocabulary_mutation_fail_closed() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(OnPolicyProbeError, match="tokenization length"):
        build_on_policy_probe_corpus(
            _child_model(torch),
            ByteTokenizer(),
            {"parent": NibbleTokenizer()},
            ["A"],
            settings=_settings(max_new_tokens=1, max_context_tokens=3),
        )

    class MutatingTokenizer(NibbleTokenizer):
        def __init__(self) -> None:
            self.mutated = False

        def __len__(self):
            return 19 if self.mutated else 18

        def get_vocab(self):
            vocabulary = super().get_vocab()
            if self.mutated:
                vocabulary["new"] = 18
            return vocabulary

        def encode(self, text, add_special_tokens=False):
            result = super().encode(text, add_special_tokens=add_special_tokens)
            self.mutated = True
            return result

    with pytest.raises(OnPolicyProbeIntegrityError, match="vocabulary changed"):
        build_on_policy_probe_corpus(
            _child_model(torch),
            ByteTokenizer(),
            {"mutating": MutatingTokenizer()},
            ["A"],
            settings=_settings(max_new_tokens=1),
        )


def test_child_parameter_mutation_during_generate_is_detected() -> None:
    torch = pytest.importorskip("torch")

    class MutatingChild(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def generate(self, input_ids, **_kwargs):
            self.weight.add_(1)
            return input_ids

    child = MutatingChild()
    before = child.weight.detach().clone()
    with pytest.raises(OnPolicyProbeIntegrityError, match="parameters or gradients"):
        build_on_policy_probe_corpus(
            child,
            ByteTokenizer(),
            {"parent": NibbleTokenizer()},
            ["A"],
            settings=_settings(max_new_tokens=1),
        )
    assert not torch.equal(child.weight, before)
