"""Deterministic corpus and causal-LM evaluation over native raw byte tokens.

The harness composes :mod:`neural_forge.hf_native_raw_tokens` with an ordinary
causal language model.  Tensor bytes and text both enter through token IDs that
already exist in the model vocabulary.  This module creates no learned module,
adds no vocabulary row, and has no modality encoder, decoder, adapter,
projector, or bridge.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import os
import re
import struct
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .hf_native_raw_tokens import (
    HFNativeRawByteCodec,
    HFNativeRawTokenError,
)


CORPUS_SCHEMA = "neural-forge-hf-native-raw-corpus-v1"
EXAMPLE_SCHEMA = "neural-forge-hf-native-raw-example-v1"
PARENT_CACHE_SCHEMA = "neural-forge-hf-native-parent-outputs-v1"
EVALUATION_SCHEMA = "neural-forge-hf-native-raw-evaluation-v1"
HARNESS_PROTOCOL = "native-existing-byte-token-causal-lm-v1"

TASK_NEXT_TOKEN = "next-token"
TASK_CAPTION = "caption"
TASK_PROBE = "probe"
TASKS = (TASK_NEXT_TOKEN, TASK_CAPTION, TASK_PROBE)

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_TOKEN_ID = (1 << 63) - 1
_CAPTION_PROMPT = "\nTask: caption\nCaption: "
_PROBE_PREFIX = "\nTask: probe\nQuestion: "
_PROBE_ANSWER = "\nAnswer: "


class NativeRawHarnessError(ValueError):
    """Raised when a corpus, cache, or evaluation contract is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical ASCII JSON representation used by every hash."""

    try:
        return json.dumps(
            _thaw_json(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise NativeRawHarnessError("value is not canonical JSON data") from exc


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def token_ids_sha256(
    token_ids: Iterable[int],
    *,
    domain: bytes = b"native-raw-eval-token-ids-v1",
) -> str:
    """Hash token IDs without an ambiguous textual serialization."""

    values = tuple(
        _token_id(value, f"token_ids[{index}]") for index, value in enumerate(token_ids)
    )
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(struct.pack(">Q", len(values)))
    for value in values:
        digest.update(struct.pack(">Q", value))
    return digest.hexdigest()


def write_auditable_json(path: str | Path, document: Mapping[str, Any]) -> str:
    """Atomically write canonical JSON and a matching ``.sha256`` sidecar."""

    target = Path(path)
    payload = canonical_json_bytes(document) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    sidecar = target.with_name(f"{target.name}.sha256")
    sidecar_temporary = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, target)
        sidecar_temporary.write_bytes(f"{digest}  {target.name}\n".encode("ascii"))
        os.replace(sidecar_temporary, sidecar)
    finally:
        if temporary.exists():
            temporary.unlink()
        if sidecar_temporary.exists():
            sidecar_temporary.unlink()
    return digest


@dataclass(frozen=True, slots=True)
class RawTextProbe:
    """One deterministic question/answer target for a raw tensor."""

    question: str
    answer: str
    probe_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        question = _nonempty_text(self.question, "probe question")
        answer = _nonempty_text(self.answer, "probe answer")
        probe_id = self.probe_id
        if probe_id:
            probe_id = _identifier(probe_id, "probe_id")
        metadata = _canonical_mapping(self.metadata, "probe metadata")
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "answer", answer)
        object.__setattr__(self, "probe_id", probe_id)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True, slots=True)
class RawModalitySample:
    """A tensor plus optional text supervision for corpus construction."""

    sample_id: str
    modality: str
    tensor: np.ndarray
    caption: str | None = None
    probes: tuple[RawTextProbe, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_id = _identifier(self.sample_id, "sample_id")
        modality = _identifier(self.modality, "modality")
        if not isinstance(self.tensor, np.ndarray):
            raise NativeRawHarnessError("sample tensor must be a NumPy ndarray")
        caption = self.caption
        if caption is not None:
            caption = _nonempty_text(caption, "caption")
        probes = tuple(self.probes)
        if any(not isinstance(probe, RawTextProbe) for probe in probes):
            raise NativeRawHarnessError(
                "sample probes must contain RawTextProbe values"
            )
        metadata = _canonical_mapping(self.metadata, "sample metadata")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "caption", caption)
        object.__setattr__(self, "probes", probes)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True, slots=True)
class RawTokenExample:
    """One prompt/target sequence with enough provenance to audit its bytes."""

    example_id: str
    task: str
    sample_id: str
    modality: str
    prompt_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    raw_token_ids: tuple[int, ...]
    source_report: Mapping[str, Any]
    expected_text: str | None = None
    question: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        example_id = _identifier(self.example_id, "example_id")
        sample_id = _identifier(self.sample_id, "sample_id")
        modality = _identifier(self.modality, "modality")
        if self.task not in TASKS:
            raise NativeRawHarnessError(f"task must be one of {list(TASKS)!r}")
        prompt = _token_ids(self.prompt_token_ids, "prompt_token_ids")
        target = _token_ids(self.target_token_ids, "target_token_ids")
        raw = _token_ids(self.raw_token_ids, "raw_token_ids")
        if not prompt or not target or not raw:
            raise NativeRawHarnessError(
                "prompt_token_ids, target_token_ids, and raw_token_ids must be nonempty"
            )
        source_report = _canonical_mapping(self.source_report, "source_report")
        metadata = _canonical_mapping(self.metadata, "example metadata")
        expected_text = self.expected_text
        question = self.question
        if self.task == TASK_NEXT_TOKEN:
            if prompt + target != raw:
                raise NativeRawHarnessError(
                    "next-token prompt and target must exactly partition raw_token_ids"
                )
            if expected_text is not None or question is not None:
                raise NativeRawHarnessError(
                    "next-token examples must not contain text supervision"
                )
        else:
            if prompt[: len(raw)] != raw:
                raise NativeRawHarnessError(
                    "text-task prompts must begin with the complete raw tensor frame"
                )
            expected_text = _nonempty_text(expected_text, "expected_text")
            if self.task == TASK_CAPTION:
                if question is not None:
                    raise NativeRawHarnessError(
                        "caption examples must not contain a question"
                    )
            else:
                question = _nonempty_text(question, "question")
        object.__setattr__(self, "example_id", example_id)
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "prompt_token_ids", prompt)
        object.__setattr__(self, "target_token_ids", target)
        object.__setattr__(self, "raw_token_ids", raw)
        object.__setattr__(self, "source_report", MappingProxyType(source_report))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        object.__setattr__(self, "expected_text", expected_text)
        object.__setattr__(self, "question", question)

    @property
    def sequence_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.target_token_ids

    @property
    def example_sha256(self) -> str:
        return canonical_json_sha256(self._record_core())

    def _record_core(self) -> dict[str, Any]:
        return {
            "schema": EXAMPLE_SCHEMA,
            "example_id": self.example_id,
            "task": self.task,
            "sample_id": self.sample_id,
            "modality": self.modality,
            "prompt_token_ids": list(self.prompt_token_ids),
            "target_token_ids": list(self.target_token_ids),
            "raw_token_ids": list(self.raw_token_ids),
            "expected_text": self.expected_text,
            "question": self.question,
            "metadata": _thaw_json(self.metadata),
            "source_report": _thaw_json(self.source_report),
            "prompt_token_ids_sha256": token_ids_sha256(
                self.prompt_token_ids, domain=b"native-raw-prompt-v1"
            ),
            "target_token_ids_sha256": token_ids_sha256(
                self.target_token_ids, domain=b"native-raw-target-v1"
            ),
            "raw_token_ids_sha256": token_ids_sha256(
                self.raw_token_ids, domain=b"native-raw-source-v1"
            ),
            "sequence_token_ids_sha256": token_ids_sha256(
                self.sequence_token_ids, domain=b"native-raw-sequence-v1"
            ),
        }

    def record(self) -> dict[str, Any]:
        core = self._record_core()
        return {**core, "example_sha256": canonical_json_sha256(core)}

    def training_record(self) -> dict[str, Any]:
        """Return ordinary causal-LM inputs with loss masked over the prompt."""

        sequence = self.sequence_token_ids
        labels = (-100,) * len(self.prompt_token_ids) + self.target_token_ids
        return {
            "example_id": self.example_id,
            "task": self.task,
            "input_ids": list(sequence),
            "attention_mask": [1] * len(sequence),
            "labels": list(labels),
            "input_ids_sha256": token_ids_sha256(
                sequence, domain=b"native-raw-training-input-v1"
            ),
            "labels_sha256": _signed_ints_sha256(labels),
            "example_sha256": self.example_sha256,
        }


@dataclass(frozen=True, slots=True)
class RawTokenCorpus:
    """An ordered, JSON-serializable set of raw-token causal examples."""

    examples: tuple[RawTokenExample, ...]
    protocol_report: Mapping[str, Any]
    build_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        examples = tuple(self.examples)
        if not examples:
            raise NativeRawHarnessError("corpus must contain at least one example")
        if any(not isinstance(example, RawTokenExample) for example in examples):
            raise NativeRawHarnessError(
                "corpus examples must contain RawTokenExample values"
            )
        identifiers = [example.example_id for example in examples]
        if len(set(identifiers)) != len(identifiers):
            raise NativeRawHarnessError("corpus example_id values must be unique")
        protocol = _canonical_mapping(self.protocol_report, "protocol_report")
        config = _canonical_mapping(self.build_config, "build_config")
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "protocol_report", MappingProxyType(protocol))
        object.__setattr__(self, "build_config", MappingProxyType(config))

    def _document_core(self) -> dict[str, Any]:
        return {
            "schema": CORPUS_SCHEMA,
            "protocol": HARNESS_PROTOCOL,
            "protocol_report": _thaw_json(self.protocol_report),
            "build_config": _thaw_json(self.build_config),
            "example_count": len(self.examples),
            "examples": [example.record() for example in self.examples],
        }

    @property
    def corpus_sha256(self) -> str:
        return canonical_json_sha256(self._document_core())

    def document(self) -> dict[str, Any]:
        core = self._document_core()
        return {**core, "corpus_sha256": canonical_json_sha256(core)}

    def report(self) -> dict[str, Any]:
        task_counts = {
            task: sum(example.task == task for example in self.examples)
            for task in TASKS
        }
        modalities = sorted({example.modality for example in self.examples})
        return {
            "schema": CORPUS_SCHEMA,
            "protocol": HARNESS_PROTOCOL,
            "corpus_sha256": self.corpus_sha256,
            "example_count": len(self.examples),
            "task_counts": task_counts,
            "modalities": modalities,
            "example_sha256": [example.example_sha256 for example in self.examples],
            "ordinary_existing_vocabulary_ids": True,
            "parameter_count": 0,
            "learned_components": [],
            "encoder": None,
            "decoder": None,
            "adapter": None,
            "projector": None,
            "bridge": None,
        }

    def write(self, path: str | Path) -> str:
        return write_auditable_json(path, self.document())


class NativeRawCorpusBuilder:
    """Build deterministic next-token, caption, and probe examples."""

    def __init__(
        self,
        codec: HFNativeRawByteCodec,
        tokenizer: Any,
        *,
        continuation_tokens: int = 32,
    ) -> None:
        if not isinstance(codec, HFNativeRawByteCodec):
            raise NativeRawHarnessError("codec must be HFNativeRawByteCodec")
        self.codec = codec
        self.tokenizer = tokenizer
        self.continuation_tokens = _positive_int(
            continuation_tokens, "continuation_tokens"
        )
        self._vocabulary = _tokenizer_vocabulary(tokenizer)
        self._tokenizer_length = _tokenizer_length(tokenizer)
        if self._tokenizer_length != codec.mapping.tokenizer_length:
            raise NativeRawHarnessError(
                "tokenizer length does not match the codec byte mapping"
            )
        self._vocabulary_sha256 = canonical_json_sha256(self._vocabulary)

    def build(self, samples: Iterable[RawModalitySample]) -> RawTokenCorpus:
        values = tuple(samples)
        if not values:
            raise NativeRawHarnessError("at least one sample is required")
        if any(not isinstance(sample, RawModalitySample) for sample in values):
            raise NativeRawHarnessError("samples must contain RawModalitySample values")
        sample_ids = [sample.sample_id for sample in values]
        if len(set(sample_ids)) != len(sample_ids):
            raise NativeRawHarnessError("sample_id values must be unique")

        examples: list[RawTokenExample] = []
        for sample in sorted(values, key=lambda item: item.sample_id):
            examples.extend(self._build_sample(sample))

        after_vocabulary = _tokenizer_vocabulary(self.tokenizer)
        if after_vocabulary != self._vocabulary:
            raise NativeRawHarnessError(
                "tokenizer vocabulary changed during corpus construction"
            )
        protocol_report = {
            "harness_protocol": HARNESS_PROTOCOL,
            "raw_codec": self.codec.protocol_report(),
            "tensor_path": "one generic ndarray C-order byte path",
            "text_path": "ordinary tokenizer IDs with add_special_tokens=False",
            "tokenizer_class": _type_name(self.tokenizer),
            "tokenizer_length": self._tokenizer_length,
            "tokenizer_vocabulary_sha256": self._vocabulary_sha256,
            "byte_mapping_sha256": self.codec.mapping.mapping_sha256,
            "vocabulary_mutated": False,
            "ordinary_existing_vocabulary_ids": True,
            "parameter_count": 0,
            "learned_modules": [],
            "encoder": None,
            "decoder": None,
            "adapter": None,
            "projector": None,
            "bridge": None,
            "runtime_donor": None,
        }
        build_config = {
            "continuation_tokens": self.continuation_tokens,
            "sample_order": "lexicographic sample_id",
            "task_order": list(TASKS),
            "caption_prompt": _CAPTION_PROMPT,
            "probe_prefix": _PROBE_PREFIX,
            "probe_answer_prefix": _PROBE_ANSWER,
            "text_special_tokens_added": False,
        }
        return RawTokenCorpus(
            examples=tuple(examples),
            protocol_report=protocol_report,
            build_config=build_config,
        )

    def _build_sample(self, sample: RawModalitySample) -> list[RawTokenExample]:
        encoded = self.codec.encode_tensor(
            sample.tensor,
            modality=sample.modality,
            metadata=sample.metadata,
        )
        raw = encoded.input_ids
        continuation_count = min(self.continuation_tokens, len(raw) - 1)
        if continuation_count <= 0:
            raise NativeRawHarnessError(
                "encoded tensor must contain at least two raw tokens"
            )
        source_report = encoded.report()
        examples = [
            RawTokenExample(
                example_id=f"{sample.sample_id}.next-token",
                task=TASK_NEXT_TOKEN,
                sample_id=sample.sample_id,
                modality=sample.modality,
                prompt_token_ids=raw[:-continuation_count],
                target_token_ids=raw[-continuation_count:],
                raw_token_ids=raw,
                source_report=source_report,
                metadata={"continuation_tokens": continuation_count},
            )
        ]
        if sample.caption is not None:
            examples.append(
                RawTokenExample(
                    example_id=f"{sample.sample_id}.caption",
                    task=TASK_CAPTION,
                    sample_id=sample.sample_id,
                    modality=sample.modality,
                    prompt_token_ids=raw + self._encode_text(_CAPTION_PROMPT),
                    target_token_ids=self._encode_text(sample.caption),
                    raw_token_ids=raw,
                    source_report=source_report,
                    expected_text=sample.caption,
                )
            )
        indexed_probes = []
        for index, probe in enumerate(sample.probes):
            probe_id = probe.probe_id or f"{index:04d}"
            indexed_probes.append((probe_id, index, probe))
        for probe_id, _, probe in sorted(indexed_probes):
            prompt_text = f"{_PROBE_PREFIX}{probe.question}{_PROBE_ANSWER}"
            examples.append(
                RawTokenExample(
                    example_id=f"{sample.sample_id}.probe.{probe_id}",
                    task=TASK_PROBE,
                    sample_id=sample.sample_id,
                    modality=sample.modality,
                    prompt_token_ids=raw + self._encode_text(prompt_text),
                    target_token_ids=self._encode_text(probe.answer),
                    raw_token_ids=raw,
                    source_report=source_report,
                    expected_text=probe.answer,
                    question=probe.question,
                    metadata={
                        "probe_id": probe_id,
                        "probe_metadata": _thaw_json(probe.metadata),
                    },
                )
            )
        return examples

    def _encode_text(self, text: str) -> tuple[int, ...]:
        encode = getattr(self.tokenizer, "encode", None)
        if not callable(encode):
            raise NativeRawHarnessError("tokenizer must provide encode()")
        try:
            result = encode(text, add_special_tokens=False)
        except TypeError as exc:
            raise NativeRawHarnessError(
                "tokenizer.encode must support add_special_tokens=False"
            ) from exc
        if hasattr(result, "ids"):
            result = result.ids
        ids = _token_ids(result, "encoded text token IDs")
        if not ids:
            raise NativeRawHarnessError("text encoded to an empty token sequence")
        if max(ids) >= self._tokenizer_length:
            raise NativeRawHarnessError(
                "text encoding produced an ID outside the existing vocabulary"
            )
        return ids


def build_raw_modality_corpus(
    codec: HFNativeRawByteCodec,
    tokenizer: Any,
    samples: Iterable[RawModalitySample],
    *,
    continuation_tokens: int = 32,
) -> RawTokenCorpus:
    """Convenience wrapper for :class:`NativeRawCorpusBuilder`."""

    return NativeRawCorpusBuilder(
        codec,
        tokenizer,
        continuation_tokens=continuation_tokens,
    ).build(samples)


def load_raw_token_corpus(path: str | Path) -> RawTokenCorpus:
    """Load and strictly verify a corpus written by :meth:`RawTokenCorpus.write`."""

    value = _read_json_object(path, "raw-token corpus")
    claimed_hash = _sha256_value(value.get("corpus_sha256"), "corpus_sha256")
    core = {key: item for key, item in value.items() if key != "corpus_sha256"}
    if canonical_json_sha256(core) != claimed_hash:
        raise NativeRawHarnessError("corpus_sha256 does not match corpus contents")
    if core.get("schema") != CORPUS_SCHEMA or core.get("protocol") != HARNESS_PROTOCOL:
        raise NativeRawHarnessError("unsupported raw-token corpus schema or protocol")
    raw_examples = core.get("examples")
    if not isinstance(raw_examples, list):
        raise NativeRawHarnessError("corpus examples must be a list")
    examples = tuple(_example_from_record(record) for record in raw_examples)
    if core.get("example_count") != len(examples):
        raise NativeRawHarnessError("corpus example_count does not match examples")
    corpus = RawTokenCorpus(
        examples=examples,
        protocol_report=core.get("protocol_report"),
        build_config=core.get("build_config"),
    )
    if corpus.corpus_sha256 != claimed_hash:
        raise NativeRawHarnessError("reconstructed corpus hash does not match")
    return corpus


@dataclass(frozen=True, slots=True)
class CachedParentOutput:
    """Model outputs sufficient to reproduce all evaluation metrics."""

    example_id: str
    example_sha256: str
    generated_token_ids: tuple[int, ...]
    cross_entropy_sum: float
    cross_entropy_tokens: int
    context_window_tokens: int
    context_truncated: bool

    def __post_init__(self) -> None:
        example_id = _identifier(self.example_id, "cached example_id")
        example_hash = _sha256_value(self.example_sha256, "example_sha256")
        generated = _token_ids(self.generated_token_ids, "cached generated_token_ids")
        loss = _finite_nonnegative_float(self.cross_entropy_sum, "cross_entropy_sum")
        tokens = _positive_int(self.cross_entropy_tokens, "cross_entropy_tokens")
        context = _positive_int(self.context_window_tokens, "context_window_tokens")
        if not isinstance(self.context_truncated, bool):
            raise NativeRawHarnessError("context_truncated must be boolean")
        object.__setattr__(self, "example_id", example_id)
        object.__setattr__(self, "example_sha256", example_hash)
        object.__setattr__(self, "generated_token_ids", generated)
        object.__setattr__(self, "cross_entropy_sum", loss)
        object.__setattr__(self, "cross_entropy_tokens", tokens)
        object.__setattr__(self, "context_window_tokens", context)

    def _record_core(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "example_sha256": self.example_sha256,
            "generated_token_ids": list(self.generated_token_ids),
            "generated_token_ids_sha256": token_ids_sha256(
                self.generated_token_ids, domain=b"native-raw-parent-generation-v1"
            ),
            "cross_entropy_sum": self.cross_entropy_sum,
            "cross_entropy_tokens": self.cross_entropy_tokens,
            "context_window_tokens": self.context_window_tokens,
            "context_truncated": self.context_truncated,
        }

    @property
    def output_sha256(self) -> str:
        return canonical_json_sha256(self._record_core())

    def record(self) -> dict[str, Any]:
        core = self._record_core()
        return {**core, "output_sha256": canonical_json_sha256(core)}


@dataclass(frozen=True, slots=True)
class ParentOutputCache:
    """A corpus- and evaluator-bound cache of immutable parent outputs."""

    corpus_sha256: str
    model_fingerprint: str
    evaluation_spec: Mapping[str, Any]
    expected_example_count: int
    entries: tuple[CachedParentOutput, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        corpus_hash = _sha256_value(self.corpus_sha256, "corpus_sha256")
        model_hash = _sha256_value(self.model_fingerprint, "model_fingerprint")
        evaluation_spec = _canonical_mapping(self.evaluation_spec, "evaluation_spec")
        expected = _positive_int(self.expected_example_count, "expected_example_count")
        entries = tuple(self.entries)
        if any(not isinstance(entry, CachedParentOutput) for entry in entries):
            raise NativeRawHarnessError(
                "parent cache entries must contain CachedParentOutput values"
            )
        identifiers = [entry.example_id for entry in entries]
        if len(set(identifiers)) != len(identifiers):
            raise NativeRawHarnessError("parent cache example_id values must be unique")
        if len(entries) > expected:
            raise NativeRawHarnessError(
                "parent cache has more entries than expected_example_count"
            )
        metadata = _canonical_mapping(self.metadata, "parent cache metadata")
        object.__setattr__(self, "corpus_sha256", corpus_hash)
        object.__setattr__(self, "model_fingerprint", model_hash)
        object.__setattr__(self, "evaluation_spec", MappingProxyType(evaluation_spec))
        object.__setattr__(self, "expected_example_count", expected)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def evaluation_fingerprint(self) -> str:
        return canonical_json_sha256(_thaw_json(self.evaluation_spec))

    @property
    def is_complete(self) -> bool:
        return len(self.entries) == self.expected_example_count

    def by_example_id(self) -> Mapping[str, CachedParentOutput]:
        return MappingProxyType({entry.example_id: entry for entry in self.entries})

    def _document_core(self) -> dict[str, Any]:
        return {
            "schema": PARENT_CACHE_SCHEMA,
            "protocol": HARNESS_PROTOCOL,
            "corpus_sha256": self.corpus_sha256,
            "model_fingerprint": self.model_fingerprint,
            "evaluation_spec": _thaw_json(self.evaluation_spec),
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "expected_example_count": self.expected_example_count,
            "entry_count": len(self.entries),
            "complete": self.is_complete,
            "entries": [entry.record() for entry in self.entries],
            "metadata": _thaw_json(self.metadata),
        }

    @property
    def cache_sha256(self) -> str:
        return canonical_json_sha256(self._document_core())

    def document(self) -> dict[str, Any]:
        core = self._document_core()
        return {**core, "cache_sha256": canonical_json_sha256(core)}

    def write(self, path: str | Path) -> str:
        return write_auditable_json(path, self.document())


def load_parent_output_cache(path: str | Path) -> ParentOutputCache:
    value = _read_json_object(path, "parent output cache")
    claimed_hash = _sha256_value(value.get("cache_sha256"), "cache_sha256")
    core = {key: item for key, item in value.items() if key != "cache_sha256"}
    if canonical_json_sha256(core) != claimed_hash:
        raise NativeRawHarnessError("cache_sha256 does not match cache contents")
    if (
        core.get("schema") != PARENT_CACHE_SCHEMA
        or core.get("protocol") != HARNESS_PROTOCOL
    ):
        raise NativeRawHarnessError("unsupported parent output cache schema")
    raw_entries = core.get("entries")
    if not isinstance(raw_entries, list):
        raise NativeRawHarnessError("parent cache entries must be a list")
    entries = tuple(_cached_output_from_record(record) for record in raw_entries)
    if core.get("entry_count") != len(entries):
        raise NativeRawHarnessError("parent cache entry_count does not match entries")
    cache = ParentOutputCache(
        corpus_sha256=core.get("corpus_sha256"),
        model_fingerprint=core.get("model_fingerprint"),
        evaluation_spec=core.get("evaluation_spec"),
        expected_example_count=core.get("expected_example_count"),
        entries=entries,
        metadata=core.get("metadata"),
    )
    if cache.evaluation_fingerprint != core.get("evaluation_fingerprint"):
        raise NativeRawHarnessError("evaluation_fingerprint does not match")
    if cache.is_complete is not core.get("complete"):
        raise NativeRawHarnessError("parent cache complete flag does not match")
    if cache.cache_sha256 != claimed_hash:
        raise NativeRawHarnessError("reconstructed parent cache hash does not match")
    return cache


class NativeRawEvaluationHarness:
    """Evaluate a causal LM without adding or training any model component."""

    def __init__(self, codec: HFNativeRawByteCodec, tokenizer: Any) -> None:
        if not isinstance(codec, HFNativeRawByteCodec):
            raise NativeRawHarnessError("codec must be HFNativeRawByteCodec")
        self.codec = codec
        self.tokenizer = tokenizer
        self._tokenizer_length = _tokenizer_length(tokenizer)
        if self._tokenizer_length != codec.mapping.tokenizer_length:
            raise NativeRawHarnessError(
                "tokenizer length does not match the codec byte mapping"
            )

    def evaluate(
        self,
        corpus: RawTokenCorpus,
        model: Any | None = None,
        *,
        model_fingerprint: str | Mapping[str, Any] | None = None,
        parent_outputs: ParentOutputCache | str | Path | None = None,
        parent_cache_path: str | Path | None = None,
        report_path: str | Path | None = None,
        max_context_tokens: int | None = None,
    ) -> dict[str, Any]:
        if not isinstance(corpus, RawTokenCorpus):
            raise NativeRawHarnessError("corpus must be RawTokenCorpus")
        vocabulary_before = _tokenizer_vocabulary(self.tokenizer)
        vocabulary_sha256 = canonical_json_sha256(vocabulary_before)
        self._validate_corpus(corpus, vocabulary_sha256=vocabulary_sha256)
        cache = _coerce_parent_cache(parent_outputs)

        if cache is not None and cache.corpus_sha256 != corpus.corpus_sha256:
            raise NativeRawHarnessError(
                "parent cache corpus_sha256 does not match the corpus"
            )

        if cache is not None and model is None and max_context_tokens is None:
            cached_context = cache.evaluation_spec.get("context_window_tokens")
            context_window = _positive_int(
                cached_context, "cached context_window_tokens"
            )
        else:
            context_window = self._resolve_context_window(
                model, max_context_tokens=max_context_tokens
            )

        evaluation_spec = {
            "algorithm": "teacher-forced-target-nll-and-fixed-length-greedy-v1",
            "context_window_tokens": context_window,
            "generation": {
                "method": "greedy_argmax",
                "do_sample": False,
                "generated_tokens": "exact target token count",
                "kv_cache": False,
            },
            "text_normalization": "Unicode NFKC + strip + casefold",
            "raw_reconstruction": "strict codec hash validation",
        }
        evaluation_fingerprint = canonical_json_sha256(evaluation_spec)
        if cache is not None and cache.evaluation_fingerprint != evaluation_fingerprint:
            raise NativeRawHarnessError(
                "parent cache evaluation contract does not match this evaluation"
            )

        fingerprint, fingerprint_report = _resolve_model_fingerprint(
            model,
            model_fingerprint,
            cache=cache,
            cache_requested=(parent_cache_path is not None),
            partial_cache_requires_model=(
                cache is not None and not cache.is_complete and model is not None
            ),
        )
        if cache is not None and cache.model_fingerprint != fingerprint:
            raise NativeRawHarnessError("parent cache model_fingerprint does not match")

        cached_by_id = cache.by_example_id() if cache is not None else {}
        expected_by_id = {example.example_id: example for example in corpus.examples}
        unknown_cache_ids = sorted(set(cached_by_id).difference(expected_by_id))
        if unknown_cache_ids:
            raise NativeRawHarnessError(
                f"parent cache contains unknown examples: {unknown_cache_ids!r}"
            )
        for example_id, cached in cached_by_id.items():
            if cached.example_sha256 != expected_by_id[example_id].example_sha256:
                raise NativeRawHarnessError(
                    f"parent cache example hash mismatch for {example_id}"
                )
            if cached.context_window_tokens != context_window:
                raise NativeRawHarnessError(
                    f"parent cache context mismatch for {example_id}"
                )
            self._validate_output_contract(
                expected_by_id[example_id],
                cached,
                context_window=context_window,
            )

        missing = [
            example
            for example in corpus.examples
            if example.example_id not in cached_by_id
        ]
        if missing and model is None:
            raise NativeRawHarnessError(
                "model is required because the parent output cache is incomplete"
            )

        outputs: list[CachedParentOutput] = []
        cache_hits = 0
        model_outputs = 0
        model_call_count = 0
        parameter_audit = {
            "model_supplied": model is not None,
            "model_used": bool(missing),
            "parameter_count_before": None,
            "parameter_count_after": None,
            "parameter_structure_unchanged": None,
            "parameter_count_added": 0,
            "learned_modules_added": [],
        }
        embedding_validation: Mapping[str, Any] | None = None

        torch = None
        signature_before = None
        was_training = None
        if missing:
            torch = _import_torch()
            signature_before = _parameter_signature(model)
            parameter_audit["parameter_count_before"] = sum(
                item[3] for item in signature_before
            )
            was_training = bool(getattr(model, "training", False))
            eval_method = getattr(model, "eval", None)
            if callable(eval_method):
                eval_method()
            embedding_validation = self.codec.validate_model_input_embedding(model)
            maximum_corpus_id = max(
                token_id
                for example in corpus.examples
                for token_id in example.sequence_token_ids
            )
            if maximum_corpus_id >= int(embedding_validation["input_embedding_rows"]):
                raise NativeRawHarnessError(
                    "model input embedding does not contain every corpus token ID"
                )

        try:
            inference_context = (
                torch.inference_mode() if torch is not None else _NullContext()
            )
            with inference_context:
                for example in corpus.examples:
                    cached = cached_by_id.get(example.example_id)
                    if cached is not None:
                        output = cached
                        cache_hits += 1
                    else:
                        output, calls = self._run_model_example(
                            model,
                            example,
                            torch=torch,
                            context_window=context_window,
                        )
                        model_call_count += calls
                        model_outputs += 1
                    self._validate_output_contract(
                        example,
                        output,
                        context_window=context_window,
                    )
                    outputs.append(output)
        finally:
            if missing:
                train_method = getattr(model, "train", None)
                if callable(train_method):
                    train_method(was_training)

        if missing:
            signature_after = _parameter_signature(model)
            unchanged = signature_after == signature_before
            parameter_audit["parameter_count_after"] = sum(
                item[3] for item in signature_after
            )
            parameter_audit["parameter_structure_unchanged"] = unchanged
            if not unchanged:
                raise NativeRawHarnessError(
                    "model parameter structure changed during evaluation"
                )

        complete_cache = ParentOutputCache(
            corpus_sha256=corpus.corpus_sha256,
            model_fingerprint=fingerprint,
            evaluation_spec=evaluation_spec,
            expected_example_count=len(corpus.examples),
            entries=tuple(outputs),
            metadata={
                "role": "cached-parent-causal-lm-outputs",
                "ordinary_existing_vocabulary_ids": True,
                "parameter_count_added": 0,
            },
        )
        cache_file_sha256 = None
        if parent_cache_path is not None:
            cache_file_sha256 = complete_cache.write(parent_cache_path)

        results = [
            self._build_result(example, output)
            for example, output in zip(corpus.examples, outputs, strict=True)
        ]
        vocabulary_after = _tokenizer_vocabulary(self.tokenizer)
        if vocabulary_after != vocabulary_before:
            raise NativeRawHarnessError(
                "tokenizer vocabulary changed during evaluation"
            )
        metrics = _aggregate_metrics(results)
        report_core = {
            "schema": EVALUATION_SCHEMA,
            "protocol": HARNESS_PROTOCOL,
            "status": "complete",
            "corpus": corpus.report(),
            "model": {
                "fingerprint": fingerprint,
                "fingerprint_report": fingerprint_report,
                "embedding_validation": _thaw_json(embedding_validation),
            },
            "evaluation_spec": evaluation_spec,
            "evaluation_fingerprint": evaluation_fingerprint,
            "cache": {
                "parent_cache_sha256": complete_cache.cache_sha256,
                "parent_cache_file_sha256": cache_file_sha256,
                "cache_hits_this_evaluation": cache_hits,
                "model_outputs_this_evaluation": model_outputs,
                "complete": complete_cache.is_complete,
            },
            "metrics": metrics,
            "examples": results,
            "audit": {
                "model_call_count": model_call_count,
                "parameter_audit": parameter_audit,
                "parameter_count_added_by_harness": 0,
                "learned_components_added": [],
                "encoder": None,
                "decoder": None,
                "adapter": None,
                "projector": None,
                "bridge": None,
                "runtime_donor_required": False,
                "tokenizer_vocabulary_sha256": vocabulary_sha256,
                "tokenizer_vocabulary_unchanged": True,
                "corpus_order_preserved": True,
                "stochastic_generation": False,
            },
        }
        report = {
            **report_core,
            "report_sha256": canonical_json_sha256(report_core),
        }
        if report_path is not None:
            write_auditable_json(report_path, report)
        return report

    def _validate_corpus(
        self,
        corpus: RawTokenCorpus,
        *,
        vocabulary_sha256: str,
    ) -> None:
        claimed_vocabulary_hash = corpus.protocol_report.get(
            "tokenizer_vocabulary_sha256"
        )
        if (
            claimed_vocabulary_hash is not None
            and claimed_vocabulary_hash != vocabulary_sha256
        ):
            raise NativeRawHarnessError(
                "corpus tokenizer vocabulary hash does not match the evaluator"
            )
        validated_samples: dict[str, str] = {}
        for example in corpus.examples:
            if max(example.sequence_token_ids) >= self._tokenizer_length:
                raise NativeRawHarnessError(
                    f"corpus example {example.example_id} contains an ID outside "
                    "the existing tokenizer vocabulary"
                )
            raw_hash = token_ids_sha256(
                example.raw_token_ids, domain=b"native-raw-source-v1"
            )
            previous = validated_samples.get(example.sample_id)
            if previous is not None:
                if previous != raw_hash:
                    raise NativeRawHarnessError(
                        f"sample {example.sample_id} has inconsistent raw frames"
                    )
                continue
            try:
                self.codec.decode_tensor(
                    example.raw_token_ids,
                    expected_report=example.source_report,
                )
            except HFNativeRawTokenError as exc:
                raise NativeRawHarnessError(
                    f"corpus raw frame validation failed for {example.example_id}"
                ) from exc
            validated_samples[example.sample_id] = raw_hash

    def _validate_output_contract(
        self,
        example: RawTokenExample,
        output: CachedParentOutput,
        *,
        context_window: int,
    ) -> None:
        target_count = len(example.target_token_ids)
        if len(output.generated_token_ids) != target_count:
            raise NativeRawHarnessError(
                f"cached generated token count mismatch for {example.example_id}"
            )
        if output.cross_entropy_tokens != target_count:
            raise NativeRawHarnessError(
                f"cached cross-entropy token count mismatch for {example.example_id}"
            )
        if output.context_window_tokens != context_window:
            raise NativeRawHarnessError(
                f"cached context window mismatch for {example.example_id}"
            )
        expected_truncation = (
            len(example.prompt_token_ids) + target_count - 1 > context_window
        )
        if output.context_truncated is not expected_truncation:
            raise NativeRawHarnessError(
                f"cached context truncation flag mismatch for {example.example_id}"
            )
        if max(output.generated_token_ids) >= self._tokenizer_length:
            raise NativeRawHarnessError(
                f"generated ID outside tokenizer vocabulary for {example.example_id}"
            )

    def _resolve_context_window(
        self,
        model: Any | None,
        *,
        max_context_tokens: int | None,
    ) -> int:
        if max_context_tokens is not None:
            return _positive_int(max_context_tokens, "max_context_tokens")
        candidates = []
        if self.codec.context_limit is not None:
            candidates.append(self.codec.context_limit)
        model_limit = _model_context_limit(model) if model is not None else None
        if model_limit is not None:
            candidates.append(model_limit)
        if not candidates:
            raise NativeRawHarnessError(
                "max_context_tokens is required when no trusted context limit exists"
            )
        return min(candidates)

    def _run_model_example(
        self,
        model: Any,
        example: RawTokenExample,
        *,
        torch: Any,
        context_window: int,
    ) -> tuple[CachedParentOutput, int]:
        device = _model_device(model, torch)
        loss_sum, context_truncated, score_calls = _target_cross_entropy(
            model,
            example.prompt_token_ids,
            example.target_token_ids,
            torch=torch,
            device=device,
            context_window=context_window,
        )
        generated, generation_calls = _greedy_continue(
            model,
            example.prompt_token_ids,
            len(example.target_token_ids),
            torch=torch,
            device=device,
            context_window=context_window,
        )
        return (
            CachedParentOutput(
                example_id=example.example_id,
                example_sha256=example.example_sha256,
                generated_token_ids=generated,
                cross_entropy_sum=loss_sum,
                cross_entropy_tokens=len(example.target_token_ids),
                context_window_tokens=context_window,
                context_truncated=context_truncated,
            ),
            score_calls + generation_calls,
        )

    def _build_result(
        self,
        example: RawTokenExample,
        output: CachedParentOutput,
    ) -> dict[str, Any]:
        expected = example.target_token_ids
        generated = output.generated_token_ids
        token_exact = generated == expected
        mean_loss = output.cross_entropy_sum / output.cross_entropy_tokens
        core: dict[str, Any] = {
            "example_id": example.example_id,
            "example_sha256": example.example_sha256,
            "task": example.task,
            "sample_id": example.sample_id,
            "modality": example.modality,
            "target_token_count": len(expected),
            "generated_token_count": len(generated),
            "target_token_ids_sha256": token_ids_sha256(
                expected, domain=b"native-raw-eval-expected-v1"
            ),
            "generated_token_ids_sha256": token_ids_sha256(
                generated, domain=b"native-raw-eval-generated-v1"
            ),
            "token_exact_match": token_exact,
            "cross_entropy_sum": output.cross_entropy_sum,
            "cross_entropy_tokens": output.cross_entropy_tokens,
            "cross_entropy_mean": mean_loss,
            "context_window_tokens": output.context_window_tokens,
            "context_truncated": output.context_truncated,
            "cached_output_sha256": output.output_sha256,
        }
        if example.task == TASK_NEXT_TOKEN:
            expected_bytes = self.codec.mapping.decode_token_ids(expected)
            generated_bytes = None
            generated_byte_error = None
            try:
                generated_bytes = self.codec.mapping.decode_token_ids(generated)
            except HFNativeRawTokenError as exc:
                generated_byte_error = str(exc)
            byte_exact = generated_bytes == expected_bytes
            frame_exact = False
            tensor_exact = False
            reconstruction_error = None
            if generated_bytes is not None:
                reconstructed_ids = example.prompt_token_ids + generated
                frame_exact = reconstructed_ids == example.raw_token_ids
                try:
                    self.codec.decode_tensor(
                        reconstructed_ids,
                        expected_report=example.source_report,
                    )
                    tensor_exact = True
                except HFNativeRawTokenError as exc:
                    reconstruction_error = str(exc)
            core["raw_continuation"] = {
                "expected_bytes": len(expected_bytes),
                "generated_bytes": (
                    len(generated_bytes) if generated_bytes is not None else None
                ),
                "expected_bytes_sha256": hashlib.sha256(expected_bytes).hexdigest(),
                "generated_bytes_sha256": (
                    hashlib.sha256(generated_bytes).hexdigest()
                    if generated_bytes is not None
                    else None
                ),
                "exact_byte_continuation": byte_exact,
                "exact_frame_reconstruction": frame_exact,
                "exact_tensor_reconstruction": tensor_exact,
                "generated_byte_error": generated_byte_error,
                "reconstruction_error": reconstruction_error,
            }
        else:
            generated_text = _decode_text(self.tokenizer, generated)
            expected_decoded = _decode_text(self.tokenizer, expected)
            normalized_expected = _normalize_answer(example.expected_text)
            normalized_generated = _normalize_answer(generated_text)
            core["text_behavior"] = {
                "question": example.question,
                "expected_text": example.expected_text,
                "expected_decoded_text": expected_decoded,
                "generated_text": generated_text,
                "generated_text_sha256": hashlib.sha256(
                    generated_text.encode("utf-8")
                ).hexdigest(),
                "decoded_text_exact_match": generated_text == expected_decoded,
                "normalized_expected": normalized_expected,
                "normalized_generated": normalized_generated,
                "answer_match": normalized_generated == normalized_expected,
            }
        return {**core, "result_sha256": canonical_json_sha256(core)}


def evaluate_raw_modality_causal_lm(
    codec: HFNativeRawByteCodec,
    tokenizer: Any,
    corpus: RawTokenCorpus,
    model: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper for :class:`NativeRawEvaluationHarness`."""

    return NativeRawEvaluationHarness(codec, tokenizer).evaluate(
        corpus,
        model,
        **kwargs,
    )


def _target_cross_entropy(
    model: Any,
    prompt: tuple[int, ...],
    target: tuple[int, ...],
    *,
    torch: Any,
    device: Any,
    context_window: int,
) -> tuple[float, bool, int]:
    complete_input = prompt + target[:-1]
    if len(complete_input) <= context_window:
        logits = _model_logits(model, complete_input, torch=torch, device=device)
        start = len(prompt) - 1
        selected = logits[0, start : start + len(target), :].float()
        if selected.shape[0] != len(target):
            raise NativeRawHarnessError(
                "causal LM logits do not cover every target token"
            )
        target_tensor = torch.tensor(target, dtype=torch.long, device=selected.device)
        _validate_target_vocabulary(selected, target_tensor)
        loss = (
            -torch.log_softmax(selected, dim=-1)
            .gather(-1, target_tensor.unsqueeze(-1))
            .sum()
        )
        return _finite_loss(loss), False, 1

    combined = prompt + target
    total = 0.0
    for index, token_id in enumerate(target):
        history = combined[: len(prompt) + index]
        context = history[-context_window:]
        logits = _model_logits(model, context, torch=torch, device=device)
        selected = logits[0, -1, :].float()
        target_tensor = torch.tensor(token_id, dtype=torch.long, device=selected.device)
        _validate_target_vocabulary(selected.unsqueeze(0), target_tensor.unsqueeze(0))
        loss = -torch.log_softmax(selected, dim=-1)[target_tensor]
        total += _finite_loss(loss)
    return total, True, len(target)


def _greedy_continue(
    model: Any,
    prompt: tuple[int, ...],
    token_count: int,
    *,
    torch: Any,
    device: Any,
    context_window: int,
) -> tuple[tuple[int, ...], int]:
    history = list(prompt)
    generated: list[int] = []
    for _ in range(token_count):
        context = tuple(history[-context_window:])
        logits = _model_logits(model, context, torch=torch, device=device)
        token_id = int(torch.argmax(logits[0, -1, :]).item())
        generated.append(_token_id(token_id, "generated token ID"))
        history.append(token_id)
    return tuple(generated), token_count


def _model_logits(
    model: Any,
    input_ids: tuple[int, ...],
    *,
    torch: Any,
    device: Any,
) -> Any:
    if not input_ids:
        raise NativeRawHarnessError("causal LM input must not be empty")
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    outputs = model(input_ids=ids, attention_mask=mask)
    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, Mapping):
        logits = outputs.get("logits")
    if logits is None and isinstance(outputs, Sequence) and outputs:
        logits = outputs[0]
    if logits is None or getattr(logits, "ndim", None) != 3:
        raise NativeRawHarnessError("causal LM must return rank-3 logits as .logits")
    if int(logits.shape[0]) != 1 or int(logits.shape[1]) != len(input_ids):
        raise NativeRawHarnessError(
            "causal LM logits batch/sequence shape does not match input_ids"
        )
    return logits


def _validate_target_vocabulary(logits: Any, target: Any) -> None:
    vocabulary = int(logits.shape[-1])
    minimum = int(target.min().item())
    maximum = int(target.max().item())
    if minimum < 0 or maximum >= vocabulary:
        raise NativeRawHarnessError(
            "target token ID is outside the causal LM output vocabulary"
        )


def _finite_loss(value: Any) -> float:
    result = float(value.detach().cpu().item())
    if not math.isfinite(result) or result < 0:
        raise NativeRawHarnessError("causal LM produced non-finite cross-entropy")
    return result


def _aggregate_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _metric_group(results),
        "by_task": {
            task: _metric_group([item for item in results if item["task"] == task])
            for task in TASKS
        },
        "by_modality": {
            modality: _metric_group(
                [item for item in results if item["modality"] == modality]
            )
            for modality in sorted({str(item["modality"]) for item in results})
        },
    }


def _metric_group(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(results)
    loss_sum = sum(float(item["cross_entropy_sum"]) for item in results)
    loss_tokens = sum(int(item["cross_entropy_tokens"]) for item in results)
    mean_loss = loss_sum / loss_tokens if loss_tokens else None
    if mean_loss is None:
        perplexity = None
        perplexity_overflow = False
    elif mean_loss > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        perplexity = None
        perplexity_overflow = True
    else:
        perplexity = math.exp(mean_loss)
        perplexity_overflow = False
    token_exact = sum(item["token_exact_match"] is True for item in results)
    raw = [item["raw_continuation"] for item in results if "raw_continuation" in item]
    text = [item["text_behavior"] for item in results if "text_behavior" in item]
    return {
        "examples": count,
        "cross_entropy_sum": loss_sum,
        "cross_entropy_tokens": loss_tokens,
        "cross_entropy_mean": mean_loss,
        "perplexity": perplexity,
        "perplexity_overflow": perplexity_overflow,
        "token_exact_matches": token_exact,
        "token_exact_accuracy": token_exact / count if count else None,
        "raw_examples": len(raw),
        "exact_byte_continuations": sum(
            item["exact_byte_continuation"] is True for item in raw
        ),
        "exact_frame_reconstructions": sum(
            item["exact_frame_reconstruction"] is True for item in raw
        ),
        "exact_tensor_reconstructions": sum(
            item["exact_tensor_reconstruction"] is True for item in raw
        ),
        "text_examples": len(text),
        "text_answer_matches": sum(item["answer_match"] is True for item in text),
        "text_answer_accuracy": (
            sum(item["answer_match"] is True for item in text) / len(text)
            if text
            else None
        ),
    }


def _resolve_model_fingerprint(
    model: Any | None,
    supplied: str | Mapping[str, Any] | None,
    *,
    cache: ParentOutputCache | None,
    cache_requested: bool,
    partial_cache_requires_model: bool,
) -> tuple[str, dict[str, Any]]:
    if supplied is not None:
        if isinstance(supplied, str):
            if not supplied:
                raise NativeRawHarnessError("model_fingerprint must not be empty")
            if _SHA256.fullmatch(supplied):
                return supplied, {"basis": "caller-supplied-sha256"}
            descriptor = {"caller_supplied_model_id": supplied}
        elif isinstance(supplied, Mapping):
            descriptor = _canonical_mapping(supplied, "model_fingerprint")
        else:
            raise NativeRawHarnessError("model_fingerprint must be a string or mapping")
        return canonical_json_sha256(descriptor), {
            "basis": "canonical caller-supplied descriptor",
            "descriptor": descriptor,
        }
    if cache is not None:
        if partial_cache_requires_model:
            raise NativeRawHarnessError(
                "model_fingerprint is required to fill a partial parent cache"
            )
        return cache.model_fingerprint, {"basis": "validated parent output cache"}
    if cache_requested:
        raise NativeRawHarnessError(
            "model_fingerprint is required when writing a reusable parent cache"
        )
    if model is None:
        raise NativeRawHarnessError("model or parent_outputs is required")
    descriptor = _runtime_model_descriptor(model)
    return canonical_json_sha256(descriptor), {
        "basis": "runtime model structure; supply a checkpoint hash for cache reuse",
        "descriptor": descriptor,
    }


def _runtime_model_descriptor(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    name_or_path = getattr(config, "_name_or_path", None)
    if name_or_path is None:
        name_or_path = getattr(config, "name_or_path", None)
    signature = _parameter_signature(model)
    return {
        "model_class": _type_name(model),
        "name_or_path": str(name_or_path) if name_or_path is not None else None,
        "parameter_count": sum(item[3] for item in signature),
        "parameter_shapes_sha256": canonical_json_sha256(
            [[item[0], item[2], item[3]] for item in signature]
        ),
    }


def _parameter_signature(
    model: Any,
) -> tuple[tuple[str, int, tuple[int, ...], int], ...]:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise NativeRawHarnessError("model must provide named_parameters()")
    signature = []
    for name, parameter in named_parameters():
        shape = tuple(int(value) for value in parameter.shape)
        signature.append((str(name), id(parameter), shape, int(parameter.numel())))
    return tuple(signature)


def _model_device(model: Any, torch: Any) -> Any:
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(iter(parameters())).device
        except StopIteration:
            pass
    get_input = getattr(model, "get_input_embeddings", None)
    if callable(get_input):
        weight = getattr(get_input(), "weight", None)
        if weight is not None:
            return weight.device
    return torch.device("cpu")


def _model_context_limit(model: Any) -> int | None:
    config = getattr(model, "config", None)
    candidates = []
    for name in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, name, None)
        if value is None or isinstance(value, bool):
            continue
        try:
            candidate = operator.index(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 < candidate <= 10_000_000:
            candidates.append(candidate)
    return min(candidates) if candidates else None


def _decode_text(tokenizer: Any, token_ids: Iterable[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise NativeRawHarnessError("tokenizer must provide decode()")
    ids = list(token_ids)
    try:
        value = decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        value = decode(ids)
    if not isinstance(value, str):
        raise NativeRawHarnessError("tokenizer.decode must return text")
    return value


def _normalize_answer(value: Any) -> str:
    if not isinstance(value, str):
        raise NativeRawHarnessError("text answer must be a string")
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _example_from_record(value: Any) -> RawTokenExample:
    if not isinstance(value, Mapping):
        raise NativeRawHarnessError("corpus example record must be an object")
    claimed = _sha256_value(value.get("example_sha256"), "example_sha256")
    core = {key: item for key, item in value.items() if key != "example_sha256"}
    if canonical_json_sha256(core) != claimed:
        raise NativeRawHarnessError("example_sha256 does not match example record")
    if core.get("schema") != EXAMPLE_SCHEMA:
        raise NativeRawHarnessError("unsupported example schema")
    example = RawTokenExample(
        example_id=core.get("example_id"),
        task=core.get("task"),
        sample_id=core.get("sample_id"),
        modality=core.get("modality"),
        prompt_token_ids=core.get("prompt_token_ids"),
        target_token_ids=core.get("target_token_ids"),
        raw_token_ids=core.get("raw_token_ids"),
        source_report=core.get("source_report"),
        expected_text=core.get("expected_text"),
        question=core.get("question"),
        metadata=core.get("metadata"),
    )
    if example.record() != dict(value):
        raise NativeRawHarnessError("example record is noncanonical or inconsistent")
    return example


def _cached_output_from_record(value: Any) -> CachedParentOutput:
    if not isinstance(value, Mapping):
        raise NativeRawHarnessError("cached output record must be an object")
    claimed = _sha256_value(value.get("output_sha256"), "output_sha256")
    core = {key: item for key, item in value.items() if key != "output_sha256"}
    if canonical_json_sha256(core) != claimed:
        raise NativeRawHarnessError("output_sha256 does not match cached output")
    output = CachedParentOutput(
        example_id=core.get("example_id"),
        example_sha256=core.get("example_sha256"),
        generated_token_ids=core.get("generated_token_ids"),
        cross_entropy_sum=core.get("cross_entropy_sum"),
        cross_entropy_tokens=core.get("cross_entropy_tokens"),
        context_window_tokens=core.get("context_window_tokens"),
        context_truncated=core.get("context_truncated"),
    )
    if output.record() != dict(value):
        raise NativeRawHarnessError("cached output record is noncanonical")
    return output


def _coerce_parent_cache(
    value: ParentOutputCache | str | Path | None,
) -> ParentOutputCache | None:
    if value is None:
        return None
    if isinstance(value, ParentOutputCache):
        return value
    if isinstance(value, (str, Path)):
        return load_parent_output_cache(value)
    raise NativeRawHarnessError(
        "parent_outputs must be ParentOutputCache, a JSON path, or None"
    )


def _tokenizer_vocabulary(tokenizer: Any) -> dict[str, int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise NativeRawHarnessError("tokenizer must provide get_vocab()")
    raw = get_vocab()
    if not isinstance(raw, Mapping):
        raise NativeRawHarnessError("tokenizer vocabulary must be a mapping")
    vocabulary = {}
    for token, value in raw.items():
        if not isinstance(token, str):
            raise NativeRawHarnessError("tokenizer vocabulary keys must be strings")
        vocabulary[token] = _token_id(value, f"vocabulary ID for {token!r}")
    return vocabulary


def _tokenizer_length(tokenizer: Any) -> int:
    try:
        value = operator.index(len(tokenizer))
    except (TypeError, ValueError, OverflowError) as exc:
        raise NativeRawHarnessError("tokenizer must provide a valid length") from exc
    if value <= 0:
        raise NativeRawHarnessError("tokenizer length must be positive")
    return value


def _read_json_object(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeRawHarnessError(f"cannot read valid JSON from {label}") from exc
    if not isinstance(value, dict):
        raise NativeRawHarnessError(f"{label} must contain one JSON object")
    return value


def _canonical_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeRawHarnessError(f"{label} must be a mapping")
    try:
        copied = json.loads(canonical_json_bytes(value).decode("ascii"))
    except NativeRawHarnessError as exc:
        raise NativeRawHarnessError(
            f"{label} must contain canonical JSON data"
        ) from exc
    if not isinstance(copied, dict):
        raise NativeRawHarnessError(f"{label} must be a JSON object")
    return copied


def _thaw_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _token_ids(value: Any, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise NativeRawHarnessError(f"{label} must be an iterable of integers")
    try:
        return tuple(
            _token_id(item, f"{label}[{index}]") for index, item in enumerate(value)
        )
    except TypeError as exc:
        raise NativeRawHarnessError(f"{label} must be iterable") from exc


def _token_id(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise NativeRawHarnessError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NativeRawHarnessError(f"{label} must be an integer") from exc
    if result < 0 or result > _MAX_TOKEN_ID:
        raise NativeRawHarnessError(f"{label} must be between 0 and {_MAX_TOKEN_ID}")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise NativeRawHarnessError(f"{label} must be a positive integer")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NativeRawHarnessError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise NativeRawHarnessError(f"{label} must be a positive integer")
    return result


def _finite_nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise NativeRawHarnessError(f"{label} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NativeRawHarnessError(
            f"{label} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(result) or result < 0:
        raise NativeRawHarnessError(f"{label} must be a finite non-negative number")
    return result


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NativeRawHarnessError(f"{label} must be nonempty text")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise NativeRawHarnessError(
            f"{label} must match [A-Za-z0-9][A-Za-z0-9._:-]{{0,127}}"
        )
    return value


def _sha256_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise NativeRawHarnessError(f"{label} must be lowercase SHA-256")
    return value


def _signed_ints_sha256(values: Iterable[int]) -> str:
    items = tuple(values)
    digest = hashlib.sha256()
    digest.update(b"native-raw-signed-labels-v1")
    digest.update(struct.pack(">Q", len(items)))
    for value in items:
        digest.update(struct.pack(">q", int(value)))
    return digest.hexdigest()


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise NativeRawHarnessError(
            "PyTorch is required to evaluate a causal LM"
        ) from exc
    return torch


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


__all__ = [
    "CORPUS_SCHEMA",
    "EVALUATION_SCHEMA",
    "EXAMPLE_SCHEMA",
    "HARNESS_PROTOCOL",
    "PARENT_CACHE_SCHEMA",
    "TASK_CAPTION",
    "TASK_NEXT_TOKEN",
    "TASK_PROBE",
    "TASKS",
    "CachedParentOutput",
    "NativeRawCorpusBuilder",
    "NativeRawEvaluationHarness",
    "NativeRawHarnessError",
    "ParentOutputCache",
    "RawModalitySample",
    "RawTextProbe",
    "RawTokenCorpus",
    "RawTokenExample",
    "build_raw_modality_corpus",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "evaluate_raw_modality_causal_lm",
    "load_parent_output_cache",
    "load_raw_token_corpus",
    "token_ids_sha256",
    "write_auditable_json",
]
