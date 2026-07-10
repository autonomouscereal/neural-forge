from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from neural_forge.hf_native_raw_eval import (  # noqa: E402
    NativeRawEvaluationHarness,
    NativeRawHarnessError,
    ParentOutputCache,
    RawModalitySample,
    RawTextProbe,
    build_raw_modality_corpus,
    canonical_json_sha256,
    load_parent_output_cache,
    load_raw_token_corpus,
)
from neural_forge.hf_native_raw_tokens import HFNativeRawByteCodec  # noqa: E402


def _bytelevel_tokenizer(model_max_length: int = 4096) -> Any:
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {token: token_id for token_id, token in enumerate(alphabet)}
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocabulary, merges=[]))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=False,
    )
    backend.decoder = tokenizers.decoders.ByteLevel()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        model_max_length=model_max_length,
    )


def _samples() -> tuple[RawModalitySample, ...]:
    return (
        RawModalitySample(
            sample_id="image-01",
            modality="image",
            tensor=np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3),
            caption="small color grid",
            probes=(RawTextProbe("How many rows?", "3", probe_id="rows"),),
            metadata={"layout": "HWC"},
        ),
        RawModalitySample(
            sample_id="audio-01",
            modality="audio",
            tensor=np.array([-32768, -3, 0, 9, 32767], dtype=np.int16),
            caption="five signed samples",
            probes=(RawTextProbe("What is the sample count?", "5"),),
            metadata={"sample_rate": 16000},
        ),
        RawModalitySample(
            sample_id="video-01",
            modality="video",
            tensor=np.arange(2 * 2 * 3 * 3, dtype=np.uint8).reshape(2, 2, 3, 3),
            caption="two tiny frames",
            probes=(RawTextProbe("How many frames?", "2"),),
            metadata={"layout": "THWC"},
        ),
        RawModalitySample(
            sample_id="pointcloud-01",
            modality="3d.point-cloud",
            tensor=np.array(
                [[-1.5, 0.0, 2.25], [3.0, -4.5, 8.0]],
                dtype="<f4",
            ),
            caption="two 3D points",
            probes=(RawTextProbe("How many points?", "2"),),
            metadata={"axes": ["x", "y", "z"]},
        ),
    )


def _build_fixture():
    tokenizer = _bytelevel_tokenizer()
    codec = HFNativeRawByteCodec(tokenizer, context_limit=4096)
    corpus = build_raw_modality_corpus(
        codec,
        tokenizer,
        _samples(),
        continuation_tokens=12,
    )
    return tokenizer, codec, corpus


def test_corpus_uses_one_deterministic_existing_byte_token_path(
    tmp_path: Path,
) -> None:
    tokenizer, codec, corpus = _build_fixture()
    reversed_corpus = build_raw_modality_corpus(
        codec,
        tokenizer,
        reversed(_samples()),
        continuation_tokens=12,
    )

    assert corpus.corpus_sha256 == reversed_corpus.corpus_sha256
    assert len(corpus.examples) == 12
    assert corpus.report()["task_counts"] == {
        "next-token": 4,
        "caption": 4,
        "probe": 4,
    }
    assert corpus.report()["modalities"] == [
        "3d.point-cloud",
        "audio",
        "image",
        "video",
    ]
    assert corpus.report()["parameter_count"] == 0
    assert corpus.report()["learned_components"] == []
    assert corpus.report()["encoder"] is None
    assert corpus.report()["decoder"] is None
    assert corpus.report()["adapter"] is None
    assert corpus.report()["projector"] is None
    assert corpus.report()["bridge"] is None

    samples_by_id = {sample.sample_id: sample for sample in _samples()}
    for example in corpus.examples:
        assert max(example.sequence_token_ids) < len(tokenizer)
        payload_offset = example.source_report["payload_token_offset"]
        expected_payload_ids = codec.mapping.encode_bytes(
            samples_by_id[example.sample_id].tensor.tobytes(order="C")
        )
        assert example.raw_token_ids[payload_offset:] == expected_payload_ids
        training = example.training_record()
        prompt_count = len(example.prompt_token_ids)
        assert training["labels"][:prompt_count] == [-100] * prompt_count
        assert training["labels"][prompt_count:] == list(example.target_token_ids)
        assert training["attention_mask"] == [1] * len(training["input_ids"])

    corpus_path = tmp_path / "corpus.json"
    file_hash = corpus.write(corpus_path)
    assert hashlib.sha256(corpus_path.read_bytes()).hexdigest() == file_hash
    assert (tmp_path / "corpus.json.sha256").is_file()
    loaded = load_raw_token_corpus(corpus_path)
    assert loaded.corpus_sha256 == corpus.corpus_sha256
    assert loaded.document() == corpus.document()


class _ReplayCausalLM(torch.nn.Module):
    def __init__(self, vocabulary_size: int, examples: Any) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, 4)
        self.config = SimpleNamespace(max_position_embeddings=4096)
        self.transitions: dict[tuple[int, ...], int] = {}
        self.fail_on_forward = False
        for example in examples:
            sequence = example.sequence_token_ids
            prompt_count = len(example.prompt_token_ids)
            for target_index, token_id in enumerate(example.target_token_ids):
                prefix = sequence[: prompt_count + target_index]
                previous = self.transitions.setdefault(prefix, token_id)
                assert previous == token_id

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask):
        if self.fail_on_forward:
            raise AssertionError("cached evaluation must not call the model")
        batch, length = input_ids.shape
        logits = torch.full(
            (batch, length, self.embedding.num_embeddings),
            -8.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        values = input_ids.detach().cpu().tolist()
        for row in range(batch):
            for position in range(length):
                prefix = tuple(values[row][: position + 1])
                expected = self.transitions.get(prefix, 0)
                logits[row, position, expected] = 8.0
        return SimpleNamespace(logits=logits)


def test_evaluation_scores_raw_reconstruction_and_text_without_parameters(
    tmp_path: Path,
) -> None:
    tokenizer, codec, corpus = _build_fixture()
    model = _ReplayCausalLM(len(tokenizer), corpus.examples)
    model.train()
    parameters_before = tuple(id(parameter) for parameter in model.parameters())
    report_path = tmp_path / "evaluation.json"

    report = NativeRawEvaluationHarness(codec, tokenizer).evaluate(
        corpus,
        model,
        model_fingerprint="synthetic-replay-parent-v1",
        report_path=report_path,
    )

    assert report["status"] == "complete"
    assert report["metrics"]["overall"]["token_exact_accuracy"] == 1.0
    assert report["metrics"]["overall"]["exact_byte_continuations"] == 4
    assert report["metrics"]["overall"]["exact_frame_reconstructions"] == 4
    assert report["metrics"]["overall"]["exact_tensor_reconstructions"] == 4
    assert report["metrics"]["overall"]["text_answer_matches"] == 8
    assert report["metrics"]["overall"]["text_answer_accuracy"] == 1.0
    assert report["metrics"]["overall"]["cross_entropy_mean"] < 0.001
    assert report["audit"]["parameter_count_added_by_harness"] == 0
    assert report["audit"]["learned_components_added"] == []
    assert report["audit"]["parameter_audit"]["parameter_structure_unchanged"] is True
    assert tuple(id(parameter) for parameter in model.parameters()) == parameters_before
    assert model.training is True
    assert report["report_sha256"] == canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_parent_outputs_are_reused_and_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    tokenizer, codec, corpus = _build_fixture()
    model = _ReplayCausalLM(len(tokenizer), corpus.examples)
    harness = NativeRawEvaluationHarness(codec, tokenizer)
    cache_path = tmp_path / "parent_outputs.json"

    first = harness.evaluate(
        corpus,
        model,
        model_fingerprint="synthetic-replay-parent-v1",
        parent_cache_path=cache_path,
    )
    cache = load_parent_output_cache(cache_path)
    assert isinstance(cache, ParentOutputCache)
    assert cache.is_complete
    assert cache.corpus_sha256 == corpus.corpus_sha256
    assert first["cache"]["model_outputs_this_evaluation"] == len(corpus.examples)

    cached = harness.evaluate(corpus, parent_outputs=cache_path)

    assert cached["cache"]["cache_hits_this_evaluation"] == len(corpus.examples)
    assert cached["cache"]["model_outputs_this_evaluation"] == 0
    assert cached["audit"]["model_call_count"] == 0
    assert cached["metrics"] == first["metrics"]

    tampered_path = tmp_path / "tampered.json"
    tampered = json.loads(cache_path.read_text(encoding="utf-8"))
    tampered["entries"][0]["generated_token_ids"][0] ^= 1
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(NativeRawHarnessError, match="cache_sha256"):
        load_parent_output_cache(tampered_path)

    incomplete = ParentOutputCache(
        corpus_sha256=cache.corpus_sha256,
        model_fingerprint=cache.model_fingerprint,
        evaluation_spec=cache.evaluation_spec,
        expected_example_count=cache.expected_example_count,
        entries=cache.entries[:-1],
    )
    with pytest.raises(NativeRawHarnessError, match="model is required"):
        harness.evaluate(corpus, parent_outputs=incomplete)
