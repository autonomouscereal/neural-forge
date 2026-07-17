"""Conservative cached-output intelligence gate for large-ratio LFM2 folds.

The gate never calls a parent model and never trains a candidate.  It consumes
an authenticated cache of parent responses plus already-generated candidate
responses.  Topical relevance is retained as a diagnostic only: promotion is
decided from separately reported correctness, capability, termination,
degeneration, code, arithmetic, instruction, and manual-review evidence.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256


BASELINE_SCHEMA = "neural-forge-lfm2-intelligence-baseline-v1"
REPORT_SCHEMA = "neural-forge-lfm2-intelligence-report-v1"
MANUAL_PACKET_SCHEMA = "neural-forge-lfm2-manual-intelligence-packet-v1"
MANUAL_RATINGS_SCHEMA = "neural-forge-lfm2-manual-intelligence-ratings-v1"
HOLDOUT_MANIFEST_SCHEMA = "neural-forge-lfm2-intelligence-holdout-manifest-v1"
HOLDOUT_SPLIT = "promotion_holdout_v1"
HOLDOUT_CASE_COUNT = 64

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*\Z")
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d[\d,]*)(?:\.\d+)?(?!\w)")
_WORD = re.compile(r"\b[\w'-]+\b")
_CODE_FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

_COMPONENT_NAMES = (
    "task_correctness",
    "parent_capability",
    "nonempty_retention",
    "fragment_retention",
    "eos_generation_limit",
    "repetition_degeneration",
    "code_syntax",
    "code_behavior",
    "arithmetic_exactness",
    "instruction_compliance",
)

_RESERVED_CALIBRATION_CASE_IDS = frozenset(
    {
        "recovery_math_00607",
        "recovery_math_00106",
        "recovery_math_01098",
        "recovery_math_00912",
        "recovery_math_01135",
        "recovery_math_00520",
        "recovery_math_00045",
        "recovery_math_00254",
        "recovery_code_00057",
        "recovery_code_00036",
        "recovery_code_00060",
        "recovery_code_00096",
        "recovery_code_00146",
        "recovery_code_00037",
        "recovery_code_00238",
        "recovery_code_00145",
        "manual_chat_00",
        "manual_fact_00",
        "manual_safety_00",
        "manual_math_00",
        "manual_code_00",
        "manual_ai_00",
        "manual_compare_00",
        "manual_instruction_00",
        "manual_reason_00",
        "manual_summary_00",
        "manual_creative_00",
        "manual_fold_00",
    }
)
_RESERVED_CALIBRATION_PROMPTS = (
    "Question: What is 26 * 3? Answer with only the number.\nAnswer:",
    "Question: What is 29 * 16? Answer with only the number.\nAnswer:",
    "Question: What is 96 + 47? Answer with only the number.\nAnswer:",
    "Question: What is 86 + 73? Answer with only the number.\nAnswer:",
    "Question: What is 30 * 10? Answer with only the number.\nAnswer:",
    "Question: What is 27 * 10? Answer with only the number.\nAnswer:",
    "Question: What is 82 + 71? Answer with only the number.\nAnswer:",
    "Question: What is 465 / 15? Answer with only the number.\nAnswer:",
    "Write a tiny Python function named subtract_057 that returns the -.\n```python\n",
    "Write a tiny Python function named add_036 that returns the +.\n```python\n",
    "Write a tiny Python function named add_060 that returns the +.\n```python\n",
    "Write a tiny Python function named add_096 that returns the +.\n```python\n",
    "Write a tiny Python function named multiply_146 that returns the *.\n```python\n",
    "Write a tiny Python function named subtract_037 that returns the -.\n```python\n",
    "Write a tiny Python function named multiply_238 that returns the *.\n```python\n",
    "Write a tiny Python function named subtract_145 that returns the -.\n```python\n",
    "Hi! Briefly introduce yourself and ask how you can help.",
    "Why does rain fall from clouds? Explain in two clear sentences.",
    "You smell gas in a house. What should you do first? Give a concise safe answer.",
    "Calculate 27 times 14. Show the short reasoning, then give the answer.",
    (
        "Write a small Python function is_palindrome(text) that ignores spaces "
        "and capitalization."
    ),
    (
        "Explain why matching one next token is not enough to prove a compressed "
        "language model is intelligent."
    ),
    "What is the practical difference between RAM and disk storage? Keep it brief.",
    "Reply with exactly one word naming the color of a clear daytime sky.",
    (
        "All bloops are razzies. No razzies are green. Can any bloop be green? "
        "Explain."
    ),
    (
        "Summarize this in one sentence: A model can preserve local activations "
        "yet fail after several generated tokens because small state errors "
        "compound through attention and the residual stream."
    ),
    "Write a coherent three-sentence story about a robot finding a lost key.",
    "In plain language, describe residual-attention-aware neuron folding.",
)


class Lfm2IntelligenceGateError(RuntimeError):
    """Cached evidence, candidate evidence, or a gate policy is invalid."""


def _canonical_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("ascii"))
    except (TypeError, ValueError) as exc:
        raise Lfm2IntelligenceGateError(f"{name} must be canonical JSON.") from exc


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise Lfm2IntelligenceGateError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a nonempty string"
        raise Lfm2IntelligenceGateError(f"{name} must be {qualifier}.")
    return value if allow_empty else value.strip()


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2IntelligenceGateError(f"{name} must be a non-negative integer.")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise Lfm2IntelligenceGateError(f"{name} must be positive.")
    return result


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2IntelligenceGateError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise Lfm2IntelligenceGateError(
            f"{name} must be finite and between zero and one."
        )
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise Lfm2IntelligenceGateError(f"{name} must be boolean.")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise Lfm2IntelligenceGateError(f"{name} must be a sequence.")
    result = tuple(_text(item, name) for item in value)
    if len(set(result)) != len(result):
        raise Lfm2IntelligenceGateError(f"{name} contains duplicates.")
    return result


def _term_groups(value: Any, name: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise Lfm2IntelligenceGateError(f"{name} must be a sequence of groups.")
    groups = []
    for index, group in enumerate(value):
        values = _string_tuple(group, f"{name}[{index}]")
        if not values:
            raise Lfm2IntelligenceGateError(f"{name}[{index}] cannot be empty.")
        groups.append(values)
    return tuple(groups)


def _words(value: str) -> tuple[str, ...]:
    return tuple(match.casefold() for match in _WORD.findall(value))


def _decimal(value: Any, name: str) -> Decimal:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2IntelligenceGateError(f"{name} must be a decimal string.")
    try:
        result = Decimal(value.replace(",", "").strip())
    except InvalidOperation as exc:
        raise Lfm2IntelligenceGateError(f"{name} is not a decimal.") from exc
    if not result.is_finite():
        raise Lfm2IntelligenceGateError(f"{name} must be finite.")
    return result


def _numbers(value: str) -> tuple[Decimal, ...]:
    result = []
    for match in _NUMBER.findall(value):
        try:
            number = Decimal(match.replace(",", ""))
        except InvalidOperation:
            continue
        if number.is_finite():
            result.append(number)
    return tuple(result)


def _document_from(value: Mapping[str, Any] | str | Path, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _canonical_copy(dict(value), name)
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise Lfm2IntelligenceGateError(f"{name} path is missing or unsafe.")
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2IntelligenceGateError(f"{name} is not canonical ASCII JSON.") from exc
    if not isinstance(document, Mapping):
        raise Lfm2IntelligenceGateError(f"{name} must be a JSON object.")
    if raw not in (
        canonical_json_bytes(document),
        canonical_json_bytes(document) + b"\n",
    ):
        raise Lfm2IntelligenceGateError(f"{name} encoding is not canonical.")
    return dict(document)


@dataclass(frozen=True, slots=True)
class GenerationEvidence:
    """Mechanical metadata for one cached or candidate generation."""

    generated_token_count: int
    generation_limit: int
    eos_observed: bool
    fragment_like: bool
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        count = _nonnegative_int(self.generated_token_count, "generated_token_count")
        limit = _positive_int(self.generation_limit, "generation_limit")
        if count > limit:
            raise Lfm2IntelligenceGateError(
                "generated_token_count cannot exceed generation_limit."
            )
        _bool(self.eos_observed, "eos_observed")
        _bool(self.fragment_like, "fragment_like")
        tokens = tuple(self.token_ids)
        if len(tokens) != count or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in tokens
        ):
            raise Lfm2IntelligenceGateError(
                "token_ids must contain exactly generated_token_count "
                "non-negative integer IDs."
            )
        object.__setattr__(self, "token_ids", tokens)

    @property
    def generation_limit_saturated(self) -> bool:
        return self.generated_token_count == self.generation_limit

    @property
    def unfinished_at_generation_limit(self) -> bool:
        return self.generation_limit_saturated and not self.eos_observed

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_token_count": self.generated_token_count,
            "generation_limit": self.generation_limit,
            "eos_observed": self.eos_observed,
            "fragment_like": self.fragment_like,
            "token_ids": list(self.token_ids),
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, name: str
    ) -> "GenerationEvidence":
        if not isinstance(value, Mapping):
            raise Lfm2IntelligenceGateError(f"{name} must be an object.")
        expected = {
            "generated_token_count",
            "generation_limit",
            "eos_observed",
            "fragment_like",
            "token_ids",
        }
        if set(value) != expected:
            raise Lfm2IntelligenceGateError(
                f"{name} fields differ from the generation-evidence schema."
            )
        token_ids = value["token_ids"]
        if not isinstance(token_ids, (list, tuple)):
            raise Lfm2IntelligenceGateError(f"{name}.token_ids must be a sequence.")
        return cls(
            generated_token_count=value["generated_token_count"],
            generation_limit=value["generation_limit"],
            eos_observed=value["eos_observed"],
            fragment_like=value["fragment_like"],
            token_ids=tuple(token_ids),
        )


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """A transparent, parent-demonstrated semantic capability contract."""

    required_concept_groups: tuple[tuple[str, ...], ...]
    minimum_characters: int = 1
    minimum_words: int = 1
    forbidden_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        groups = _term_groups(self.required_concept_groups, "required_concept_groups")
        if not groups:
            raise Lfm2IntelligenceGateError(
                "A capability spec requires at least one concept group."
            )
        object.__setattr__(self, "required_concept_groups", groups)
        object.__setattr__(
            self,
            "minimum_characters",
            _positive_int(self.minimum_characters, "minimum_characters"),
        )
        object.__setattr__(
            self,
            "minimum_words",
            _positive_int(self.minimum_words, "minimum_words"),
        )
        object.__setattr__(
            self,
            "forbidden_terms",
            _string_tuple(self.forbidden_terms, "forbidden_terms"),
        )

    def passes(self, output: str) -> bool:
        normalized = output.casefold()
        return (
            len(output.strip()) >= self.minimum_characters
            and len(_words(output)) >= self.minimum_words
            and all(
                any(term.casefold() in normalized for term in group)
                for group in self.required_concept_groups
            )
            and not any(term.casefold() in normalized for term in self.forbidden_terms)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_concept_groups": [
                list(group) for group in self.required_concept_groups
            ],
            "minimum_characters": self.minimum_characters,
            "minimum_words": self.minimum_words,
            "forbidden_terms": list(self.forbidden_terms),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CapabilitySpec":
        if not isinstance(value, Mapping):
            raise Lfm2IntelligenceGateError("capability_spec must be an object.")
        allowed = {
            "required_concept_groups",
            "minimum_characters",
            "minimum_words",
            "forbidden_terms",
        }
        if not set(value).issubset(allowed) or "required_concept_groups" not in value:
            raise Lfm2IntelligenceGateError(
                "capability_spec fields differ from its schema."
            )
        return cls(
            required_concept_groups=value["required_concept_groups"],
            minimum_characters=value.get("minimum_characters", 1),
            minimum_words=value.get("minimum_words", 1),
            forbidden_terms=value.get("forbidden_terms", ()),
        )


def _safe_json_value(value: Any, name: str, *, depth: int = 0) -> Any:
    if depth > 5:
        raise Lfm2IntelligenceGateError(f"{name} is nested too deeply.")
    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str) and len(value) > 1024:
            raise Lfm2IntelligenceGateError(f"{name} string is too long.")
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > 10**12:
            raise Lfm2IntelligenceGateError(f"{name} integer is too large.")
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > 10**12:
            raise Lfm2IntelligenceGateError(f"{name} float is invalid.")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            raise Lfm2IntelligenceGateError(f"{name} has too many items.")
        return [_safe_json_value(item, f"{name}[]", depth=depth + 1) for item in value]
    if isinstance(value, Mapping):
        if len(value) > 64 or any(not isinstance(key, str) for key in value):
            raise Lfm2IntelligenceGateError(f"{name} object is invalid.")
        return {
            key: _safe_json_value(item, f"{name}.{key}", depth=depth + 1)
            for key, item in sorted(value.items())
        }
    raise Lfm2IntelligenceGateError(f"{name} is not a bounded JSON value.")


@dataclass(frozen=True, slots=True)
class PureFunctionExample:
    arguments: tuple[Any, ...]
    expected: Any

    def __post_init__(self) -> None:
        arguments = _safe_json_value(list(self.arguments), "arguments")
        expected = _safe_json_value(self.expected, "expected")
        if len(canonical_json_bytes({"a": arguments, "e": expected})) > 8192:
            raise Lfm2IntelligenceGateError(
                "Pure-function example exceeds the size bound."
            )
        object.__setattr__(self, "arguments", tuple(arguments))
        object.__setattr__(self, "expected", expected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arguments": list(self.arguments),
            "expected": self.expected,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PureFunctionExample":
        if not isinstance(value, Mapping) or set(value) != {"arguments", "expected"}:
            raise Lfm2IntelligenceGateError(
                "Pure-function example fields differ from its schema."
            )
        arguments = value["arguments"]
        if not isinstance(arguments, (list, tuple)):
            raise Lfm2IntelligenceGateError(
                "Pure-function arguments must be a sequence."
            )
        return cls(arguments=tuple(arguments), expected=value["expected"])


@dataclass(frozen=True, slots=True)
class ReliableValidatorSpec:
    """An explicit validator whose result is eligible for correctness metrics."""

    kind: str
    expected_numeric: str | None = None
    function_name: str | None = None
    examples: tuple[PureFunctionExample, ...] = ()
    required_groups: tuple[tuple[str, ...], ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    minimum_words: int = 1
    maximum_words: int | None = None
    exact_word: str | None = None

    def __post_init__(self) -> None:
        kind = _text(self.kind, "validator kind")
        if kind not in {
            "arithmetic_exact",
            "python_pure_function",
            "instruction_constraints",
        }:
            raise Lfm2IntelligenceGateError(
                f"Unsupported reliable validator kind {kind!r}."
            )
        object.__setattr__(self, "kind", kind)
        examples = tuple(self.examples)
        if any(not isinstance(item, PureFunctionExample) for item in examples):
            raise Lfm2IntelligenceGateError(
                "examples must contain PureFunctionExample values."
            )
        object.__setattr__(self, "examples", examples)
        object.__setattr__(
            self,
            "required_groups",
            _term_groups(self.required_groups, "required_groups"),
        )
        object.__setattr__(
            self,
            "forbidden_terms",
            _string_tuple(self.forbidden_terms, "forbidden_terms"),
        )
        object.__setattr__(
            self,
            "minimum_words",
            _positive_int(self.minimum_words, "minimum_words"),
        )
        if self.maximum_words is not None:
            maximum = _positive_int(self.maximum_words, "maximum_words")
            if maximum < self.minimum_words:
                raise Lfm2IntelligenceGateError(
                    "maximum_words cannot be smaller than minimum_words."
                )
            object.__setattr__(self, "maximum_words", maximum)
        if kind == "arithmetic_exact":
            expected = _text(self.expected_numeric, "expected_numeric")
            _decimal(expected, "expected_numeric")
            object.__setattr__(self, "expected_numeric", expected)
        elif self.expected_numeric is not None:
            raise Lfm2IntelligenceGateError(
                "expected_numeric is only valid for arithmetic_exact."
            )
        if kind == "python_pure_function":
            function_name = _text(self.function_name, "function_name")
            if _IDENTIFIER.fullmatch(function_name) is None or not examples:
                raise Lfm2IntelligenceGateError(
                    "python_pure_function requires a valid name and examples."
                )
            object.__setattr__(self, "function_name", function_name)
        elif self.function_name is not None or examples:
            raise Lfm2IntelligenceGateError(
                "Function fields are only valid for python_pure_function."
            )
        if kind == "instruction_constraints":
            if not self.required_groups and self.exact_word is None:
                raise Lfm2IntelligenceGateError(
                    "Instruction validators require explicit constraints."
                )
            if self.exact_word is not None:
                object.__setattr__(
                    self, "exact_word", _text(self.exact_word, "exact_word")
                )
        elif (
            self.required_groups
            or self.forbidden_terms
            or self.maximum_words is not None
            or self.exact_word is not None
            or self.minimum_words != 1
        ):
            raise Lfm2IntelligenceGateError(
                "Instruction fields are only valid for instruction_constraints."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "expected_numeric": self.expected_numeric,
            "function_name": self.function_name,
            "examples": [item.as_dict() for item in self.examples],
            "required_groups": [list(group) for group in self.required_groups],
            "forbidden_terms": list(self.forbidden_terms),
            "minimum_words": self.minimum_words,
            "maximum_words": self.maximum_words,
            "exact_word": self.exact_word,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReliableValidatorSpec":
        if not isinstance(value, Mapping):
            raise Lfm2IntelligenceGateError("validator must be an object.")
        allowed = {
            "kind",
            "expected_numeric",
            "function_name",
            "examples",
            "required_groups",
            "forbidden_terms",
            "minimum_words",
            "maximum_words",
            "exact_word",
        }
        if "kind" not in value or not set(value).issubset(allowed):
            raise Lfm2IntelligenceGateError("validator fields differ from its schema.")
        examples = value.get("examples", ())
        if not isinstance(examples, (list, tuple)):
            raise Lfm2IntelligenceGateError("validator examples must be a sequence.")
        return cls(
            kind=value["kind"],
            expected_numeric=value.get("expected_numeric"),
            function_name=value.get("function_name"),
            examples=tuple(PureFunctionExample.from_mapping(item) for item in examples),
            required_groups=value.get("required_groups", ()),
            forbidden_terms=value.get("forbidden_terms", ()),
            minimum_words=value.get("minimum_words", 1),
            maximum_words=value.get("maximum_words"),
            exact_word=value.get("exact_word"),
        )


def _normalized_prompt(value: str) -> str:
    return " ".join(value.casefold().split())


def _prompt_sha256(value: str) -> str:
    return canonical_sha256(
        {
            "schema": "neural-forge-normalized-prompt-v1",
            "prompt": _normalized_prompt(value),
        }
    )


@dataclass(frozen=True, slots=True)
class IntelligenceHoldoutCaseSpec:
    """One immutable task in the independent promotion holdout."""

    case_id: str
    prompt: str
    category: str
    split: str
    generation_limit: int
    validator: ReliableValidatorSpec | None
    capability: CapabilitySpec | None
    manual_review_required: bool
    longer_prefix: bool = False

    def __post_init__(self) -> None:
        case_id = _text(self.case_id, "holdout case_id")
        if not case_id.startswith("holdout_v1_"):
            raise Lfm2IntelligenceGateError(
                "Holdout case IDs must use the holdout_v1_ namespace."
            )
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "prompt", _text(self.prompt, "holdout prompt"))
        object.__setattr__(self, "category", _text(self.category, "holdout category"))
        split = _text(self.split, "holdout split")
        if split != HOLDOUT_SPLIT:
            raise Lfm2IntelligenceGateError(
                f"Holdout cases must use split {HOLDOUT_SPLIT!r}."
            )
        object.__setattr__(self, "split", split)
        limit = _positive_int(self.generation_limit, "generation_limit")
        if not 16 <= limit <= 128:
            raise Lfm2IntelligenceGateError(
                "Holdout generation limits must be between 16 and 128."
            )
        object.__setattr__(self, "generation_limit", limit)
        if self.validator is not None and not isinstance(
            self.validator, ReliableValidatorSpec
        ):
            raise Lfm2IntelligenceGateError(
                "Holdout validator must be ReliableValidatorSpec or None."
            )
        if self.capability is not None and not isinstance(
            self.capability, CapabilitySpec
        ):
            raise Lfm2IntelligenceGateError(
                "Holdout capability must be CapabilitySpec or None."
            )
        _bool(self.manual_review_required, "manual_review_required")
        _bool(self.longer_prefix, "longer_prefix")
        if self.validator is None and not self.manual_review_required:
            raise Lfm2IntelligenceGateError(
                "Unvalidated holdout cases require manual review."
            )
        if self.manual_review_required and self.category != "open_ended_manual":
            raise Lfm2IntelligenceGateError(
                "Manual holdout cases must use open_ended_manual category."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "prompt": self.prompt,
            "category": self.category,
            "split": self.split,
            "generation_limit": self.generation_limit,
            "validator": (
                None if self.validator is None else self.validator.as_dict()
            ),
            "capability": (
                None if self.capability is None else self.capability.as_dict()
            ),
            "manual_review_required": self.manual_review_required,
            "longer_prefix": self.longer_prefix,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "IntelligenceHoldoutCaseSpec":
        if not isinstance(value, Mapping) or set(value) != {
            "case_id",
            "prompt",
            "category",
            "split",
            "generation_limit",
            "validator",
            "capability",
            "manual_review_required",
            "longer_prefix",
        }:
            raise Lfm2IntelligenceGateError(
                "Holdout case fields differ from the manifest schema."
            )
        validator = value["validator"]
        capability = value["capability"]
        return cls(
            case_id=value["case_id"],
            prompt=value["prompt"],
            category=value["category"],
            split=value["split"],
            generation_limit=value["generation_limit"],
            validator=(
                None
                if validator is None
                else ReliableValidatorSpec.from_mapping(validator)
            ),
            capability=(
                None
                if capability is None
                else CapabilitySpec.from_mapping(capability)
            ),
            manual_review_required=value["manual_review_required"],
            longer_prefix=value["longer_prefix"],
        )


@dataclass(frozen=True, slots=True)
class IntelligenceHoldoutManifest:
    """Typed, sealed 64-case promotion manifest."""

    cases: tuple[IntelligenceHoldoutCaseSpec, ...]
    manifest_sha256: str
    document_sha256: str
    reserved_case_ids_sha256: str
    reserved_prompts_sha256: str

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        if (
            len(cases) != HOLDOUT_CASE_COUNT
            or any(not isinstance(item, IntelligenceHoldoutCaseSpec) for item in cases)
        ):
            raise Lfm2IntelligenceGateError(
                f"Holdout manifest requires exactly {HOLDOUT_CASE_COUNT} cases."
            )
        if len({item.case_id for item in cases}) != len(cases):
            raise Lfm2IntelligenceGateError(
                "Holdout manifest case IDs must be unique."
            )
        if len({_prompt_sha256(item.prompt) for item in cases}) != len(cases):
            raise Lfm2IntelligenceGateError(
                "Holdout manifest prompts must be unique."
            )
        object.__setattr__(self, "cases", cases)
        for name in (
            "manifest_sha256",
            "document_sha256",
            "reserved_case_ids_sha256",
            "reserved_prompts_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    @property
    def case_map(self) -> Mapping[str, IntelligenceHoldoutCaseSpec]:
        return MappingProxyType({item.case_id: item for item in self.cases})


def _arithmetic_validator(expected: str) -> ReliableValidatorSpec:
    return ReliableValidatorSpec(
        kind="arithmetic_exact",
        expected_numeric=expected,
    )


def _instruction_validator(
    *groups: Sequence[str],
    minimum_words: int = 1,
    maximum_words: int | None = None,
    exact_word: str | None = None,
    forbidden_terms: Sequence[str] = (),
) -> ReliableValidatorSpec:
    return ReliableValidatorSpec(
        kind="instruction_constraints",
        required_groups=tuple(tuple(group) for group in groups),
        forbidden_terms=tuple(forbidden_terms),
        minimum_words=minimum_words,
        maximum_words=maximum_words,
        exact_word=exact_word,
    )


def _capability(
    *groups: Sequence[str],
    minimum_words: int = 1,
    minimum_characters: int = 1,
    forbidden_terms: Sequence[str] = (),
) -> CapabilitySpec:
    return CapabilitySpec(
        required_concept_groups=tuple(tuple(group) for group in groups),
        minimum_words=minimum_words,
        minimum_characters=minimum_characters,
        forbidden_terms=tuple(forbidden_terms),
    )


def _code_validator(
    name: str,
    examples: Sequence[tuple[Sequence[Any], Any]],
) -> ReliableValidatorSpec:
    return ReliableValidatorSpec(
        kind="python_pure_function",
        function_name=name,
        examples=tuple(
            PureFunctionExample(arguments=tuple(arguments), expected=expected)
            for arguments, expected in examples
        ),
    )


def _holdout_cases() -> tuple[IntelligenceHoldoutCaseSpec, ...]:
    cases: list[IntelligenceHoldoutCaseSpec] = []

    def add(
        stem: str,
        prompt: str,
        category: str,
        *,
        validator: ReliableValidatorSpec | None,
        capability: CapabilitySpec | None = None,
        manual: bool = False,
        limit: int = 64,
        longer_prefix: bool = False,
    ) -> None:
        cases.append(
            IntelligenceHoldoutCaseSpec(
                case_id=f"holdout_v1_{stem}",
                prompt=prompt,
                category=category,
                split=HOLDOUT_SPLIT,
                generation_limit=limit,
                validator=validator,
                capability=capability,
                manual_review_required=manual,
                longer_prefix=longer_prefix,
            )
        )

    arithmetic = (
        ("arithmetic_00", "A shipment has 37 cartons with 18 items each. What is the total? End with the number.", "666"),
        ("arithmetic_01", "Compute 1,250 minus 487. End with the numeric result.", "763"),
        ("arithmetic_02", "Divide 864 by 27. Give the final number.", "32"),
        ("arithmetic_03", "Evaluate (48 + 72) / 6. End with the result.", "20"),
        ("arithmetic_04", "What is 15 percent of 340? End with the number.", "51"),
        ("arithmetic_05", "Calculate 7.5 times 8. End with the result.", "60"),
        ("arithmetic_06", "Find the average of 18, 24, 33, and 45. End with the number.", "30"),
        ("arithmetic_07", "Compute 9 times 9 minus 17. End with the result.", "64"),
        ("arithmetic_08", "Add 2.4 and 3.75. End with the decimal result.", "6.15"),
        ("arithmetic_09", "Evaluate 14 times 13 plus 6. End with the result.", "188"),
    )
    for stem, prompt, expected in arithmetic:
        add(
            stem,
            prompt,
            "arithmetic",
            validator=_arithmetic_validator(expected),
            limit=32,
        )

    code_rows = (
        (
            "python_00",
            "holdout_is_even",
            "Write only a pure Python function holdout_is_even(n) that returns whether n is even. Use no imports.",
            (((2,), True), ((7,), False), ((0,), True)),
        ),
        (
            "python_01",
            "holdout_clamp",
            "Write only a pure Python function holdout_clamp(x, low, high) that clamps x to the inclusive range. Use one return expression and no imports.",
            (((-2, 0, 5), 0), ((3, 0, 5), 3), ((8, 0, 5), 5)),
        ),
        (
            "python_02",
            "holdout_last",
            "Write only a pure Python function holdout_last(items) that returns the last item. No imports.",
            ((([1, 2, 3],), 3), ((["a", "b"],), "b")),
        ),
        (
            "python_03",
            "holdout_reverse",
            "Write only a pure Python function holdout_reverse(text) that returns the string reversed. No imports.",
            ((("abc",), "cba"), (("",), ""), (("level",), "level")),
        ),
        (
            "python_04",
            "holdout_between",
            "Write only a pure Python function holdout_between(x, low, high) that returns True exactly when low <= x <= high. No imports.",
            (((3, 1, 5), True), ((0, 1, 5), False), ((5, 1, 5), True)),
        ),
        (
            "python_05",
            "holdout_choose",
            "Write only a pure Python function holdout_choose(flag, yes, no) that returns yes when flag is true and no otherwise. No imports.",
            (((True, "up", "down"), "up"), ((False, 1, 2), 2)),
        ),
        (
            "python_06",
            "holdout_difference",
            "Write only a pure Python function holdout_difference(a, b) that returns a minus b. No imports.",
            (((9, 4), 5), ((-2, 3), -5), ((1.5, 0.5), 1.0)),
        ),
        (
            "python_07",
            "holdout_join",
            "Write only a pure Python function holdout_join(left, right) that concatenates the two strings. No imports.",
            ((("neural", "forge"), "neuralforge"), (("", "x"), "x")),
        ),
    )
    for stem, name, prompt, examples in code_rows:
        add(
            stem,
            prompt,
            "python",
            validator=_code_validator(name, examples),
            limit=96,
        )

    factual = (
        (
            "factual_00",
            "A metal spoon and a wooden spoon sit in the same room. Why does the metal one feel colder?",
            (("heat", "thermal"), ("conduct", "transfer"), ("hand", "skin")),
        ),
        (
            "factual_01",
            "What causes Earth's seasons? Give a concise explanation.",
            (("tilt", "axis"), ("sunlight", "sun"), ("hemisphere", "angle")),
        ),
        (
            "factual_02",
            "Why does water boil at a lower temperature high on a mountain?",
            (("pressure",), ("lower", "less"), ("boil", "vapor")),
        ),
        (
            "factual_03",
            "Why do seat belts reduce injury in a sudden stop?",
            (("inertia", "momentum"), ("restrain", "hold"), ("force", "stop")),
        ),
        (
            "factual_04",
            "Why can a houseplant gradually bend toward a window?",
            (("light",), ("growth", "grow"), ("phototrop", "hormone")),
        ),
        (
            "factual_05",
            "Why does ordinary ice float on liquid water?",
            (("density", "less dense"), ("expand", "structure"), ("float", "buoy")),
        ),
        (
            "factual_06",
            "Why is reusing one password across many sites risky?",
            (("breach", "stolen", "leak"), ("reuse", "same password"), ("account", "sites")),
        ),
        (
            "factual_07",
            "Explain one practical difference between file synchronization and a backup.",
            (("sync", "synchron"), ("backup", "copy"), ("delete", "recover", "history")),
        ),
    )
    for stem, prompt, groups in factual:
        validator = _instruction_validator(
            *groups,
            minimum_words=8,
            maximum_words=80,
        )
        add(
            stem,
            prompt,
            "factual_common_sense",
            validator=validator,
            capability=_capability(*groups, minimum_words=8),
        )

    instruction_rows = (
        (
            "instruction_00",
            "Reply with exactly the single word amber.",
            _instruction_validator(exact_word="amber"),
        ),
        (
            "instruction_01",
            "Name the three-sided polygon using exactly one word.",
            _instruction_validator(exact_word="triangle"),
        ),
        (
            "instruction_02",
            "In at most ten words, state the two compass directions opposite south and west.",
            _instruction_validator(
                ("north",),
                ("east",),
                maximum_words=10,
            ),
        ),
        (
            "instruction_03",
            "In twelve words or fewer, recommend water and shade for someone overheated.",
            _instruction_validator(
                ("water", "hydrate"),
                ("shade", "cool"),
                maximum_words=12,
            ),
        ),
        (
            "instruction_04",
            "Give a calm six-to-twelve-word reminder to check the address before sending email. Do not use the word 'very'.",
            _instruction_validator(
                ("check", "verify"),
                ("address", "recipient"),
                minimum_words=6,
                maximum_words=12,
                forbidden_terms=("very",),
            ),
        ),
        (
            "instruction_05",
            "In one short sentence under eighteen words, suggest a consistent bedtime and reducing screens for better sleep.",
            _instruction_validator(
                ("bedtime", "routine", "schedule"),
                ("screen", "device"),
                maximum_words=18,
            ),
        ),
        (
            "instruction_06",
            "Classify apple and carrot in no more than six words, mentioning fruit and vegetable.",
            _instruction_validator(
                ("apple",),
                ("fruit",),
                ("carrot",),
                ("vegetable",),
                maximum_words=6,
            ),
        ),
        (
            "instruction_07",
            "In under twenty words, decline an unsafe request and offer a safer alternative.",
            _instruction_validator(
                ("cannot", "decline", "won't"),
                ("safe", "safer"),
                ("alternative", "instead"),
                maximum_words=20,
            ),
        ),
    )
    for stem, prompt, validator in instruction_rows:
        add(stem, prompt, "instruction", validator=validator, limit=48)

    reasoning = (
        (
            "reasoning_00",
            "Every cedar is a tree, and no tree is made of glass. Can a cedar be made of glass? Explain briefly.",
            (("no", "cannot"), ("cedar",), ("tree",), ("glass",)),
        ),
        (
            "reasoning_01",
            "Maya arrives before Leon, and Leon arrives before Noor. Who arrives earliest, and why?",
            (("maya",), ("before", "earliest"), ("leon",), ("noor",)),
        ),
        (
            "reasoning_02",
            "If the alarm is armed, the red light is on. The red light is not on. What follows about the alarm?",
            (("not armed", "isn't armed", "cannot be armed"), ("light",), ("if", "therefore")),
        ),
        (
            "reasoning_03",
            "A fair coin is flipped twice. Explain why the chance of two heads is one quarter.",
            (("one quarter", "1/4", "25%"), ("four",), ("outcome", "equally likely")),
        ),
        (
            "reasoning_04",
            "Task A must finish before both B and C can start. Which task must be scheduled first? Explain.",
            (("task a", "a"), ("first", "before"), ("b",), ("c",)),
        ),
        (
            "reasoning_05",
            "Nia is taller than Omar, and Omar is taller than Priya. Compare Nia and Priya.",
            (("nia",), ("priya",), ("taller",), ("than",)),
        ),
    )
    for stem, prompt, groups in reasoning:
        add(
            stem,
            prompt,
            "reasoning",
            validator=_instruction_validator(
                *groups,
                minimum_words=6,
                maximum_words=80,
            ),
            capability=_capability(*groups, minimum_words=6),
        )

    summaries = (
        (
            "summary_00",
            "Summarize in one sentence under 35 words: The library extended weekend hours after students requested more study time, but weekday hours did not change.",
            (("library",), ("weekend",), ("extend", "longer"), ("weekday", "unchanged", "same")),
        ),
        (
            "summary_01",
            "Summarize in one sentence under 35 words: The test server failed because a certificate expired. Engineers renewed it, restarted the proxy, and service returned.",
            (("certificate",), ("expired",), ("renew",), ("service", "restored", "returned")),
        ),
        (
            "summary_02",
            "Summarize in one sentence under 35 words: A small battery stores less energy than a large one, yet improved efficiency can let the same device run longer.",
            (("battery",), ("less", "smaller"), ("efficien",), ("longer", "runtime")),
        ),
        (
            "summary_03",
            "Summarize in one sentence under 35 words: The team rejected exact sentence matching as a model-quality metric and adopted task correctness, coherence, and safe termination.",
            (("exact", "sentence matching"), ("task", "correct"), ("coher",), ("termination", "stop")),
        ),
    )
    for stem, prompt, groups in summaries:
        add(
            stem,
            prompt,
            "summary",
            validator=_instruction_validator(
                *groups,
                minimum_words=8,
                maximum_words=35,
            ),
            capability=_capability(*groups, minimum_words=8),
        )

    safety = (
        (
            "safety_00",
            "A plugged-in appliance starts an electrical fire. What are the first safe actions?",
            (("power", "unplug", "breaker"), ("call", "emergency"), ("extinguisher", "leave", "evacuate")),
        ),
        (
            "safety_01",
            "Someone may have taken too much medication and is becoming drowsy. What should you do now?",
            (("emergency", "911", "poison control"), ("call", "contact"), ("wait", "immediately", "now")),
        ),
        (
            "safety_02",
            "An unexpected message asks you to open a login link and enter your password. What is the safe response?",
            (("don't click", "do not click", "avoid"), ("verify", "official"), ("report", "delete")),
        ),
        (
            "safety_03",
            "A hiker is lost near dusk with poor weather approaching. Give concise priorities.",
            (("shelter", "warm"), ("signal", "call", "rescue"), ("stay", "location", "safe")),
        ),
    )
    for stem, prompt, groups in safety:
        add(
            stem,
            prompt,
            "safety",
            validator=_instruction_validator(
                *groups,
                minimum_words=8,
                maximum_words=80,
            ),
            capability=_capability(*groups, minimum_words=8),
        )

    manual_rows = (
        (
            "manual_00",
            "A user says they are overwhelmed by a long debugging session. Respond helpfully and empathetically without pretending to have feelings.",
            (("debug", "problem"), ("step", "break", "help")),
        ),
        (
            "manual_01",
            "Write a coherent four-sentence micro-story about a lighthouse that discovers an unexpected purpose.",
            (("lighthouse",), ("purpose", "discover")),
        ),
        (
            "manual_02",
            "In plain language, explain how sharing one residual output basis can compress many neural-network layers without forcing their private operators to be identical.",
            (("residual",), ("basis",), ("layer",), ("private", "distinct", "different")),
        ),
        (
            "manual_03",
            "Explain one important limitation of judging a language model from a single generated token.",
            (("token",), ("context", "sequence", "coherence")),
        ),
        (
            "manual_04",
            "Create a useful analogy for compounding generation errors in a compressed model, then explain the analogy.",
            (("error",), ("compound", "accumulate", "grow"), ("model",)),
        ),
        (
            "manual_05",
            "A programmer says, 'My test passes alone but fails in the full suite.' Ask two useful diagnostic questions and suggest one next step.",
            (("test",), ("question", "?"), ("next", "try", "step")),
        ),
    )
    for stem, prompt, groups in manual_rows:
        add(
            stem,
            prompt,
            "open_ended_manual",
            validator=None,
            capability=_capability(*groups, minimum_words=8),
            manual=True,
            limit=96,
        )

    termination = (
        (
            "termination_00",
            "Respond with exactly one word: done",
            _instruction_validator(exact_word="done"),
        ),
        (
            "termination_01",
            "Answer exactly one word: Is seven an even number?",
            _instruction_validator(exact_word="no"),
        ),
        (
            "termination_02",
            "Confirm in at most eight words that the backup completed successfully.",
            _instruction_validator(
                ("backup",),
                ("complete", "successful", "finished"),
                maximum_words=8,
            ),
        ),
        (
            "termination_03",
            "Use no more than six words to say the request is queued.",
            _instruction_validator(
                ("request",),
                ("queue", "queued"),
                maximum_words=6,
            ),
        ),
    )
    for stem, prompt, validator in termination:
        add(
            stem,
            prompt,
            "termination_repetition",
            validator=validator,
            limit=32,
        )

    long_rows = (
        (
            "long_prefix_00",
            (
                "Incident record: At 08:40 the API was healthy. At 09:05 a routine "
                "deployment completed. At 09:20 an operator changed the proxy's "
                "certificate path to a staging filename. At 09:24 TLS checks began "
                "failing although application workers stayed healthy. At 09:31 the "
                "team restarted workers with no effect. At 09:38 they restored the "
                "previous certificate path and TLS recovered. At 09:45 monitoring "
                "was green. Notes say the deployment itself changed no certificate "
                "files, the database remained reachable, and CPU stayed normal. "
                "Question: Identify the most likely root cause and the change that "
                "restored service."
            ),
            (("certificate",), ("path",), ("staging", "wrong"), ("restore", "previous")),
        ),
        (
            "long_prefix_01",
            (
                "Policy excerpt: Standard diagnostic logs are retained for thirty "
                "days. Security incident evidence may be retained for one year. A "
                "legal hold overrides routine deletion until counsel releases it. "
                "Teams must not copy production secrets into logs. Customers may "
                "request deletion of ordinary support attachments, but records under "
                "legal hold cannot be removed. Backups follow the same classification "
                "on their next expiration cycle. Question: State the normal log "
                "retention period and the condition that can override deletion."
            ),
            (("thirty", "30"), ("day",), ("legal hold",), ("override", "until")),
        ),
        (
            "long_prefix_02",
            (
                "Experiment notes: Four identical trays received equal water and "
                "temperature. Tray A used full-spectrum light and averaged 12 cm of "
                "growth. Tray B used a blue filter and averaged 15 cm. Tray C used a "
                "red filter and averaged 9 cm. Tray D stayed dark and averaged 2 cm. "
                "The trial lasted three weeks and each tray held twenty seedlings. "
                "No fertilizer was added. Question: Which lighting condition produced "
                "the greatest average growth, and what was that average?"
            ),
            (("blue",), ("15", "fifteen"), ("cm", "centimeter"), ("greatest", "highest")),
        ),
        (
            "long_prefix_03",
            (
                "Product brief: The field application must work without a network, "
                "save drafts locally, and export completed inspections as JSON. "
                "Optional ideas include animated themes, social sharing, and an "
                "embedded news feed. Supervisors rank offline operation and JSON "
                "export as launch blockers; visual themes can wait. The first release "
                "targets rugged tablets with limited connectivity. Question: Name the "
                "two launch-blocking capabilities."
            ),
            (("offline", "without a network"), ("json",), ("export",), ("launch", "block")),
        ),
        (
            "long_prefix_04",
            (
                "Meeting memo: The research group will meet Wednesday at 14:00 in "
                "Room Cedar. The access codeword is marigold. Tuesday was considered "
                "but rejected because the lab is closed. Room Maple is reserved for "
                "another team, and Thursday is the fallback only if Cedar loses power. "
                "Bring the evaluation packet, not the calibration worksheet. Question: "
                "What day is the planned meeting and what is the codeword?"
            ),
            (("wednesday",), ("marigold",), ("meeting",), ("code", "codeword")),
        ),
        (
            "long_prefix_05",
            (
                "Compression comparison: Method A preserved 92 percent first-token "
                "agreement but often repeated until the generation limit. Method B "
                "preserved 88 percent first-token agreement, retained EOS behavior, "
                "and answered 21 of 23 reliable tasks. Method C preserved 95 percent "
                "weight cosine but produced empty answers on half the prompts. The "
                "promotion rule values coherent task completion and safe termination "
                "over exact first-token identity or raw weight similarity. Question: "
                "Which method best satisfies the promotion rule, and why?"
            ),
            (("method b", "b"), ("eos", "termination"), ("21",), ("task", "coherent")),
        ),
    )
    for stem, prompt, groups in long_rows:
        add(
            stem,
            prompt,
            "longer_prefix",
            validator=_instruction_validator(
                *groups,
                minimum_words=6,
                maximum_words=80,
            ),
            capability=_capability(*groups, minimum_words=6),
            limit=80,
            longer_prefix=True,
        )
    return tuple(cases)


_HOLDOUT_CATEGORY_COUNTS = MappingProxyType(
    {
        "arithmetic": 10,
        "python": 8,
        "factual_common_sense": 8,
        "instruction": 8,
        "reasoning": 6,
        "summary": 4,
        "safety": 4,
        "open_ended_manual": 6,
        "termination_repetition": 4,
        "longer_prefix": 6,
    }
)


def build_lfm2_intelligence_holdout_manifest() -> dict[str, Any]:
    """Build the exact deterministic promotion holdout without model access."""

    cases = _holdout_cases()
    reserved_ids = sorted(_RESERVED_CALIBRATION_CASE_IDS)
    reserved_prompt_hashes = sorted(
        _prompt_sha256(value) for value in _RESERVED_CALIBRATION_PROMPTS
    )
    core = {
        "schema": HOLDOUT_MANIFEST_SCHEMA,
        "suite_version": 1,
        "split": HOLDOUT_SPLIT,
        "case_count": HOLDOUT_CASE_COUNT,
        "category_counts": dict(_HOLDOUT_CATEGORY_COUNTS),
        "cases": [item.as_dict() for item in cases],
        "reserved_calibration_case_ids_sha256": canonical_sha256(
            reserved_ids
        ),
        "reserved_calibration_prompts_sha256": canonical_sha256(
            reserved_prompt_hashes
        ),
        "calibration_case_count": len(reserved_ids),
        "semantic_task_validation": True,
        "exact_parent_wording_used": False,
        "exact_parent_token_matching_used": False,
    }
    return _canonical_copy(
        {**core, "manifest_sha256": canonical_sha256(core)},
        "LFM2 intelligence holdout manifest",
    )


def load_lfm2_intelligence_holdout_manifest(
    source: Mapping[str, Any] | str | Path | None = None,
    *,
    expected_manifest_sha256: str | None = None,
) -> IntelligenceHoldoutManifest:
    """Authenticate the exact built-in 64-case holdout and its disjointness."""

    document = (
        build_lfm2_intelligence_holdout_manifest()
        if source is None
        else _document_from(source, "LFM2 intelligence holdout manifest")
    )
    core = dict(document)
    seal = core.pop("manifest_sha256", None)
    expected_document = build_lfm2_intelligence_holdout_manifest()
    expected_seal = expected_document["manifest_sha256"]
    if expected_manifest_sha256 is not None:
        expected = _digest(
            expected_manifest_sha256,
            "expected_manifest_sha256",
        )
    else:
        expected = expected_seal
    if (
        core.get("schema") != HOLDOUT_MANIFEST_SCHEMA
        or seal != expected
        or seal != expected_seal
        or seal != canonical_sha256(core)
        or canonical_json_bytes(document)
        != canonical_json_bytes(expected_document)
        or core.get("case_count") != HOLDOUT_CASE_COUNT
        or core.get("category_counts") != dict(_HOLDOUT_CATEGORY_COUNTS)
        or core.get("semantic_task_validation") is not True
        or core.get("exact_parent_wording_used") is not False
        or core.get("exact_parent_token_matching_used") is not False
    ):
        raise Lfm2IntelligenceGateError(
            "Holdout manifest seal, exact suite identity, or semantic policy "
            "is invalid."
        )
    raw_cases = core.get("cases")
    if not isinstance(raw_cases, list):
        raise Lfm2IntelligenceGateError("Holdout manifest cases are invalid.")
    cases = tuple(
        IntelligenceHoldoutCaseSpec.from_mapping(value)
        for value in raw_cases
    )
    case_ids = {item.case_id for item in cases}
    prompt_hashes = {_prompt_sha256(item.prompt) for item in cases}
    reserved_prompt_hashes = {
        _prompt_sha256(value) for value in _RESERVED_CALIBRATION_PROMPTS
    }
    observed_counts = {
        category: sum(item.category == category for item in cases)
        for category in _HOLDOUT_CATEGORY_COUNTS
    }
    if (
        case_ids.intersection(_RESERVED_CALIBRATION_CASE_IDS)
        or prompt_hashes.intersection(reserved_prompt_hashes)
        or observed_counts != dict(_HOLDOUT_CATEGORY_COUNTS)
        or sum(item.longer_prefix for item in cases) != 6
        or sum(item.manual_review_required for item in cases) != 6
        or any(item.split != HOLDOUT_SPLIT for item in cases)
    ):
        raise Lfm2IntelligenceGateError(
            "Holdout manifest overlaps calibration or lost required coverage."
        )
    return IntelligenceHoldoutManifest(
        cases=cases,
        manifest_sha256=str(seal),
        document_sha256=canonical_sha256(document),
        reserved_case_ids_sha256=str(
            core["reserved_calibration_case_ids_sha256"]
        ),
        reserved_prompts_sha256=str(
            core["reserved_calibration_prompts_sha256"]
        ),
    )


@dataclass(frozen=True, slots=True)
class CachedParentCase:
    case_id: str
    prompt: str
    category: str
    split: str
    parent_output: str
    generation: GenerationEvidence
    validator: ReliableValidatorSpec | None
    capability: CapabilitySpec | None
    manual_review_required: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(self, "prompt", _text(self.prompt, "prompt"))
        object.__setattr__(self, "category", _text(self.category, "category"))
        object.__setattr__(self, "split", _text(self.split, "split"))
        object.__setattr__(
            self,
            "parent_output",
            _text(self.parent_output, "parent_output", allow_empty=True),
        )
        if not isinstance(self.generation, GenerationEvidence):
            raise Lfm2IntelligenceGateError("generation must be GenerationEvidence.")
        if self.validator is not None and not isinstance(
            self.validator, ReliableValidatorSpec
        ):
            raise Lfm2IntelligenceGateError(
                "validator must be ReliableValidatorSpec or None."
            )
        if self.capability is not None and not isinstance(
            self.capability, CapabilitySpec
        ):
            raise Lfm2IntelligenceGateError(
                "capability must be CapabilitySpec or None."
            )
        _bool(self.manual_review_required, "manual_review_required")
        if self.validator is None and not self.manual_review_required:
            raise Lfm2IntelligenceGateError(
                "Cases without a reliable validator require manual review."
            )
        if self.capability is not None and not self.capability.passes(
            self.parent_output
        ):
            raise Lfm2IntelligenceGateError(
                f"Parent output for {self.case_id!r} does not demonstrate its "
                "declared capability."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "prompt": self.prompt,
            "category": self.category,
            "split": self.split,
            "parent_output": self.parent_output,
            "generation": self.generation.as_dict(),
            "validator": (None if self.validator is None else self.validator.as_dict()),
            "capability": (
                None if self.capability is None else self.capability.as_dict()
            ),
            "manual_review_required": self.manual_review_required,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CachedParentCase":
        if not isinstance(value, Mapping):
            raise Lfm2IntelligenceGateError("Parent case must be an object.")
        expected = {
            "case_id",
            "prompt",
            "category",
            "split",
            "parent_output",
            "generation",
            "validator",
            "capability",
            "manual_review_required",
        }
        if set(value) != expected:
            raise Lfm2IntelligenceGateError(
                "Parent case fields differ from the baseline schema."
            )
        validator = value["validator"]
        capability = value["capability"]
        return cls(
            case_id=value["case_id"],
            prompt=value["prompt"],
            category=value["category"],
            split=value["split"],
            parent_output=value["parent_output"],
            generation=GenerationEvidence.from_mapping(
                value["generation"], name="parent generation"
            ),
            validator=(
                None
                if validator is None
                else ReliableValidatorSpec.from_mapping(validator)
            ),
            capability=(
                None if capability is None else CapabilitySpec.from_mapping(capability)
            ),
            manual_review_required=value["manual_review_required"],
        )


@dataclass(frozen=True, slots=True)
class CachedParentBaseline:
    parent_cache_key: str
    parent_state_fingerprint: str
    tokenizer_fingerprint: str
    cases: tuple[CachedParentCase, ...]
    baseline_sha256: str
    document_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "parent_cache_key",
            "parent_state_fingerprint",
            "tokenizer_fingerprint",
            "baseline_sha256",
            "document_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        cases = tuple(self.cases)
        if not cases or any(not isinstance(item, CachedParentCase) for item in cases):
            raise Lfm2IntelligenceGateError("Baseline requires cached parent cases.")
        case_ids = tuple(item.case_id for item in cases)
        if len(set(case_ids)) != len(case_ids):
            raise Lfm2IntelligenceGateError("Baseline case IDs must be unique.")
        object.__setattr__(self, "cases", cases)

    @property
    def case_map(self) -> Mapping[str, CachedParentCase]:
        return MappingProxyType({item.case_id: item for item in self.cases})


def seal_cached_parent_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent_cache_key: str,
    parent_state_fingerprint: str,
    tokenizer_fingerprint: str,
) -> dict[str, Any]:
    """Create a canonical parent-output envelope without executing a model."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise Lfm2IntelligenceGateError("rows must be a sequence of objects.")
    cases = tuple(CachedParentCase.from_mapping(item) for item in rows)
    if not cases:
        raise Lfm2IntelligenceGateError("rows cannot be empty.")
    core = {
        "schema": BASELINE_SCHEMA,
        "parent_cache_key": _digest(parent_cache_key, "parent_cache_key"),
        "parent_state_fingerprint": _digest(
            parent_state_fingerprint, "parent_state_fingerprint"
        ),
        "tokenizer_fingerprint": _digest(
            tokenizer_fingerprint, "tokenizer_fingerprint"
        ),
        "cases": [item.as_dict() for item in cases],
        "parent_forward_passes": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    document = {**core, "baseline_sha256": canonical_sha256(core)}
    return _canonical_copy(document, "sealed parent baseline")


def load_cached_parent_rows(
    source: Mapping[str, Any] | str | Path,
    *,
    expected_document_sha256: str,
    expected_parent_cache_key: str,
    expected_parent_state_fingerprint: str,
    expected_tokenizer_fingerprint: str,
) -> CachedParentBaseline:
    """Authenticate and type-check a sealed parent-output envelope."""

    document = _document_from(source, "cached parent baseline")
    expected_document = _digest(expected_document_sha256, "expected_document_sha256")
    actual_document = canonical_sha256(document)
    if actual_document != expected_document:
        raise Lfm2IntelligenceGateError(
            "Cached parent baseline document fingerprint differs from expected."
        )
    core = dict(document)
    seal = core.pop("baseline_sha256", None)
    if core.get("schema") != BASELINE_SCHEMA or seal != canonical_sha256(core):
        raise Lfm2IntelligenceGateError("Cached parent baseline seal is invalid.")
    if (
        core.get("parent_forward_passes") != 0
        or core.get("optimizer_steps") != 0
        or core.get("backward_calls") != 0
    ):
        raise Lfm2IntelligenceGateError(
            "Baseline does not certify cached, zero-training evaluation."
        )
    identities = {
        "parent_cache_key": expected_parent_cache_key,
        "parent_state_fingerprint": expected_parent_state_fingerprint,
        "tokenizer_fingerprint": expected_tokenizer_fingerprint,
    }
    for name, expected in identities.items():
        expected_digest = _digest(expected, f"expected_{name}")
        if core.get(name) != expected_digest:
            raise Lfm2IntelligenceGateError(
                f"Cached parent baseline {name} differs from expected."
            )
    rows = core.get("cases")
    if not isinstance(rows, list):
        raise Lfm2IntelligenceGateError("Cached parent baseline cases are invalid.")
    cases = tuple(CachedParentCase.from_mapping(item) for item in rows)
    reconstructed = seal_cached_parent_rows(
        [item.as_dict() for item in cases],
        parent_cache_key=core["parent_cache_key"],
        parent_state_fingerprint=core["parent_state_fingerprint"],
        tokenizer_fingerprint=core["tokenizer_fingerprint"],
    )
    if canonical_json_bytes(reconstructed) != canonical_json_bytes(document):
        raise Lfm2IntelligenceGateError(
            "Cached parent baseline is not in canonical typed form."
        )
    return CachedParentBaseline(
        parent_cache_key=core["parent_cache_key"],
        parent_state_fingerprint=core["parent_state_fingerprint"],
        tokenizer_fingerprint=core["tokenizer_fingerprint"],
        cases=cases,
        baseline_sha256=seal,
        document_sha256=actual_document,
    )


@dataclass(frozen=True, slots=True)
class CandidateOutput:
    case_id: str
    output: str
    generation: GenerationEvidence
    topical_relevance: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(
            self, "output", _text(self.output, "output", allow_empty=True)
        )
        if not isinstance(self.generation, GenerationEvidence):
            raise Lfm2IntelligenceGateError(
                "candidate generation must be GenerationEvidence."
            )
        if self.topical_relevance is not None:
            _bool(self.topical_relevance, "topical_relevance")

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "output": self.output,
            "generation": self.generation.as_dict(),
            "topical_relevance": self.topical_relevance,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateOutput":
        if not isinstance(value, Mapping):
            raise Lfm2IntelligenceGateError("Candidate output must be an object.")
        expected = {"case_id", "output", "generation", "topical_relevance"}
        if set(value) != expected:
            raise Lfm2IntelligenceGateError(
                "Candidate fields differ from the candidate schema."
            )
        return cls(
            case_id=value["case_id"],
            output=value["output"],
            generation=GenerationEvidence.from_mapping(
                value["generation"], name="candidate generation"
            ),
            topical_relevance=value["topical_relevance"],
        )


@dataclass(frozen=True, slots=True)
class IntelligenceGatePolicy:
    """Explicit component thresholds; no weighted aggregate is used."""

    minimum_task_correctness_retention: float = 0.90
    # This is a retention ratio against the parent's absolute accuracy, not a
    # demand that a compressed child solve 90% of tasks its parent could not
    # solve.  Absolute accuracy remains visible in every report.
    minimum_absolute_task_correctness: float = 0.90
    minimum_parent_capability_retention: float = 0.90
    minimum_nonempty_retention: float = 0.90
    minimum_fragment_retention: float = 0.90
    minimum_code_syntax_rate: float = 0.90
    minimum_code_behavior_retention: float = 0.90
    minimum_arithmetic_retention: float = 0.90
    minimum_instruction_retention: float = 0.90
    minimum_headline_lower_bound: float = 0.90
    maximum_new_empty_responses: int = 0
    maximum_new_fragment_responses: int = 0
    maximum_eos_or_limit_regressions: int = 0
    maximum_repetition_regressions: int = 0
    maximum_repeated_bigram_share_delta: float = 0.05
    degeneration_bigram_share: float = 0.35
    degeneration_minimum_tokens: int = 8
    fragment_minimum_characters: int = 12
    manual_minimum_dimension: int = 3
    manual_minimum_mean: float = 3.60
    minimum_manual_case_pass_rate: float = 0.90
    required_components: tuple[str, ...] = _COMPONENT_NAMES

    def __post_init__(self) -> None:
        for name in (
            "minimum_task_correctness_retention",
            "minimum_absolute_task_correctness",
            "minimum_parent_capability_retention",
            "minimum_nonempty_retention",
            "minimum_fragment_retention",
            "minimum_code_syntax_rate",
            "minimum_code_behavior_retention",
            "minimum_arithmetic_retention",
            "minimum_instruction_retention",
            "minimum_headline_lower_bound",
            "maximum_repeated_bigram_share_delta",
            "degeneration_bigram_share",
            "minimum_manual_case_pass_rate",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        for name in (
            "maximum_new_empty_responses",
            "maximum_new_fragment_responses",
            "maximum_eos_or_limit_regressions",
            "maximum_repetition_regressions",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        object.__setattr__(
            self,
            "fragment_minimum_characters",
            _positive_int(
                self.fragment_minimum_characters,
                "fragment_minimum_characters",
            ),
        )
        object.__setattr__(
            self,
            "degeneration_minimum_tokens",
            _positive_int(
                self.degeneration_minimum_tokens,
                "degeneration_minimum_tokens",
            ),
        )
        minimum_dimension = _nonnegative_int(
            self.manual_minimum_dimension, "manual_minimum_dimension"
        )
        if minimum_dimension > 4:
            raise Lfm2IntelligenceGateError(
                "manual_minimum_dimension cannot exceed four."
            )
        object.__setattr__(self, "manual_minimum_dimension", minimum_dimension)
        if (
            isinstance(self.manual_minimum_mean, bool)
            or not isinstance(self.manual_minimum_mean, (int, float))
            or not math.isfinite(float(self.manual_minimum_mean))
            or not 0.0 <= float(self.manual_minimum_mean) <= 4.0
        ):
            raise Lfm2IntelligenceGateError(
                "manual_minimum_mean must be finite and in [0, 4]."
            )
        object.__setattr__(self, "manual_minimum_mean", float(self.manual_minimum_mean))
        required = _string_tuple(self.required_components, "required_components")
        unknown = set(required).difference(_COMPONENT_NAMES)
        if unknown:
            raise Lfm2IntelligenceGateError(
                "required_components contains unknown names: "
                + ",".join(sorted(unknown))
            )
        object.__setattr__(self, "required_components", required)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_task_correctness_retention": (
                self.minimum_task_correctness_retention
            ),
            "minimum_absolute_task_correctness": (
                self.minimum_absolute_task_correctness
            ),
            "minimum_parent_capability_retention": (
                self.minimum_parent_capability_retention
            ),
            "minimum_nonempty_retention": self.minimum_nonempty_retention,
            "minimum_fragment_retention": self.minimum_fragment_retention,
            "minimum_code_syntax_rate": self.minimum_code_syntax_rate,
            "minimum_code_behavior_retention": (self.minimum_code_behavior_retention),
            "minimum_arithmetic_retention": self.minimum_arithmetic_retention,
            "minimum_instruction_retention": self.minimum_instruction_retention,
            "minimum_headline_lower_bound": self.minimum_headline_lower_bound,
            "maximum_new_empty_responses": self.maximum_new_empty_responses,
            "maximum_new_fragment_responses": self.maximum_new_fragment_responses,
            "maximum_eos_or_limit_regressions": (self.maximum_eos_or_limit_regressions),
            "maximum_repetition_regressions": (self.maximum_repetition_regressions),
            "maximum_repeated_bigram_share_delta": (
                self.maximum_repeated_bigram_share_delta
            ),
            "degeneration_bigram_share": self.degeneration_bigram_share,
            "degeneration_minimum_tokens": self.degeneration_minimum_tokens,
            "fragment_minimum_characters": self.fragment_minimum_characters,
            "manual_minimum_dimension": self.manual_minimum_dimension,
            "manual_minimum_mean": self.manual_minimum_mean,
            "minimum_manual_case_pass_rate": self.minimum_manual_case_pass_rate,
            "required_components": list(self.required_components),
        }


def _candidate_fingerprint(candidates: Sequence[CandidateOutput]) -> str:
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-candidate-output-set-v1",
            "candidates": [item.as_dict() for item in candidates],
        }
    )


def _extract_python(output: str) -> str:
    fence = _CODE_FENCE.search(output)
    if fence is not None:
        return fence.group(1).strip()
    try:
        ast.parse(output, mode="exec")
    except SyntaxError:
        pass
    else:
        return output.strip()
    start = re.search(r"(?m)^\s*def\s+[A-Za-z_]\w*\s*\(", output)
    return output[start.start() :].strip() if start is not None else output.strip()


def _function_tree(
    output: str, function_name: str
) -> tuple[ast.FunctionDef | None, str | None]:
    source = _extract_python(output)
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return None, "syntax_error"
    body = tuple(tree.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if (
        len(body) != 1
        or not isinstance(body[0], ast.FunctionDef)
        or body[0].name != function_name
    ):
        return None, "named_function_missing_or_ambiguous"
    return body[0], None


def _bounded_runtime_value(value: Any) -> Any:
    return _safe_json_value(value, "pure-function result")


def _safe_eval_expression(node: ast.AST, environment: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return _bounded_runtime_value(node.value)
    if isinstance(node, ast.Name):
        if node.id not in environment:
            raise Lfm2IntelligenceGateError("Unsafe or unknown name in function.")
        return environment[node.id]
    if isinstance(node, ast.List):
        return _bounded_runtime_value(
            [_safe_eval_expression(item, environment) for item in node.elts]
        )
    if isinstance(node, ast.Tuple):
        return _bounded_runtime_value(
            [_safe_eval_expression(item, environment) for item in node.elts]
        )
    if isinstance(node, ast.Dict):
        keys = [_safe_eval_expression(item, environment) for item in node.keys]
        if any(not isinstance(key, str) for key in keys):
            raise Lfm2IntelligenceGateError(
                "Pure-function dictionary keys must be strings."
            )
        return _bounded_runtime_value(
            {
                key: _safe_eval_expression(item, environment)
                for key, item in zip(keys, node.values)
            }
        )
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.Not, ast.UAdd, ast.USub)
    ):
        operand = _safe_eval_expression(node.operand, environment)
        if isinstance(node.op, ast.Not):
            result = not operand
        elif not isinstance(operand, (int, float)) or isinstance(operand, bool):
            raise Lfm2IntelligenceGateError("Unsafe unary operand.")
        elif isinstance(node.op, ast.UAdd):
            result = +operand
        else:
            result = -operand
        return _bounded_runtime_value(result)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        values = [_safe_eval_expression(item, environment) for item in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
    ):
        left = _safe_eval_expression(node.left, environment)
        right = _safe_eval_expression(node.right, environment)
        if isinstance(node.op, ast.Add):
            if (isinstance(left, (str, list)) and isinstance(right, type(left))) or (
                isinstance(left, (int, float))
                and not isinstance(left, bool)
                and isinstance(right, (int, float))
                and not isinstance(right, bool)
            ):
                result = left + right
            else:
                raise Lfm2IntelligenceGateError("Unsafe addition operands.")
        else:
            if (
                not isinstance(left, (int, float))
                or isinstance(left, bool)
                or not isinstance(right, (int, float))
                or isinstance(right, bool)
            ):
                raise Lfm2IntelligenceGateError("Unsafe numeric operands.")
            if isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                result = left / right
            elif isinstance(node.op, ast.FloorDiv):
                result = left // right
            else:
                result = left % right
        return _bounded_runtime_value(result)
    if isinstance(node, ast.Compare):
        left = _safe_eval_expression(node.left, environment)
        comparisons = []
        for operator, raw_right in zip(node.ops, node.comparators):
            right = _safe_eval_expression(raw_right, environment)
            if isinstance(operator, ast.Eq):
                result = left == right
            elif isinstance(operator, ast.NotEq):
                result = left != right
            elif isinstance(operator, ast.Lt):
                result = left < right
            elif isinstance(operator, ast.LtE):
                result = left <= right
            elif isinstance(operator, ast.Gt):
                result = left > right
            elif isinstance(operator, ast.GtE):
                result = left >= right
            elif isinstance(operator, ast.In):
                result = left in right
            elif isinstance(operator, ast.NotIn):
                result = left not in right
            else:
                raise Lfm2IntelligenceGateError("Unsafe comparison operator.")
            comparisons.append(result)
            left = right
        return all(comparisons)
    if isinstance(node, ast.IfExp):
        condition = _safe_eval_expression(node.test, environment)
        branch = node.body if condition else node.orelse
        return _safe_eval_expression(branch, environment)
    if isinstance(node, ast.Subscript):
        value = _safe_eval_expression(node.value, environment)
        if isinstance(node.slice, ast.Slice):
            lower = (
                None
                if node.slice.lower is None
                else _safe_eval_expression(node.slice.lower, environment)
            )
            upper = (
                None
                if node.slice.upper is None
                else _safe_eval_expression(node.slice.upper, environment)
            )
            step = (
                None
                if node.slice.step is None
                else _safe_eval_expression(node.slice.step, environment)
            )
            index: Any = slice(lower, upper, step)
        else:
            index = _safe_eval_expression(node.slice, environment)
        return _bounded_runtime_value(value[index])
    raise Lfm2IntelligenceGateError(f"Unsafe Python construct {type(node).__name__!r}.")


def _run_pure_function(
    output: str, validator: ReliableValidatorSpec
) -> tuple[bool, bool, str | None]:
    function_name = validator.function_name
    if function_name is None:
        raise AssertionError("Pure-function name vanished.")
    function, syntax_error = _function_tree(output, function_name)
    if function is None:
        return False, False, syntax_error
    if (
        function.decorator_list
        or function.returns is not None
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.kwonlyargs
        or function.args.defaults
        or function.args.kw_defaults
        or any(
            argument.annotation is not None
            for argument in (*function.args.posonlyargs, *function.args.args)
        )
    ):
        return True, False, "unsafe_signature"
    argument_names = tuple(item.arg for item in function.args.args)
    body = tuple(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return True, False, "unsafe_function_body"
    for index, example in enumerate(validator.examples):
        if len(example.arguments) != len(argument_names):
            return True, False, f"argument_count_mismatch:{index}"
        environment = dict(zip(argument_names, example.arguments))
        try:
            actual = _safe_eval_expression(body[0].value, environment)
        except (
            Lfm2IntelligenceGateError,
            ArithmeticError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return True, False, f"unsafe_or_failed_example:{index}"
        if canonical_json_bytes(actual) != canonical_json_bytes(example.expected):
            return True, False, f"wrong_example:{index}"
    return True, True, None


def _instruction_passes(output: str, validator: ReliableValidatorSpec) -> bool:
    stripped = output.strip()
    normalized = stripped.casefold()
    words = _words(stripped)
    if len(words) < validator.minimum_words:
        return False
    if validator.maximum_words is not None and len(words) > validator.maximum_words:
        return False
    if validator.exact_word is not None and (
        len(words) != 1 or stripped.casefold() != validator.exact_word.casefold()
    ):
        return False
    return all(
        any(term.casefold() in normalized for term in group)
        for group in validator.required_groups
    ) and not any(term.casefold() in normalized for term in validator.forbidden_terms)


def _validator_result(
    output: str, validator: ReliableValidatorSpec
) -> tuple[bool, dict[str, Any]]:
    if validator.kind == "arithmetic_exact":
        expected = _decimal(validator.expected_numeric, "expected_numeric")
        observed = _numbers(output)
        passed = bool(observed) and observed[-1] == expected
        return passed, {
            "kind": validator.kind,
            "final_numeric_exact": passed,
            "numeric_token_count": len(observed),
        }
    if validator.kind == "instruction_constraints":
        passed = _instruction_passes(output, validator)
        return passed, {"kind": validator.kind, "constraints_satisfied": passed}
    if validator.kind == "python_pure_function":
        syntax, behavior, reason = _run_pure_function(output, validator)
        return syntax and behavior, {
            "kind": validator.kind,
            "syntax_passed": syntax,
            "behavior_passed": behavior,
            "failure_reason": reason,
        }
    raise AssertionError("Reliable validator kind vanished.")


def _repeated_bigram_share(tokens: tuple[int, ...]) -> float:
    bigrams = tuple(zip(tokens, tokens[1:]))
    if not bigrams:
        return 0.0
    counts: dict[tuple[int, int], int] = {}
    for bigram in bigrams:
        counts[bigram] = counts.get(bigram, 0) + 1
    return max(counts.values()) / len(bigrams)


def _longest_token_run(tokens: tuple[int, ...]) -> int:
    longest = 0
    current = 0
    previous = None
    for token in tokens:
        if token == previous:
            current += 1
        else:
            current = 1
            previous = token
        longest = max(longest, current)
    return longest


@dataclass(frozen=True, slots=True)
class ComponentResult:
    name: str
    eligible_count: int
    success_count: int
    retention: float | None
    passed: bool
    failure_ids: tuple[str, ...]
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "component name"))
        eligible = _nonnegative_int(self.eligible_count, "eligible_count")
        success = _nonnegative_int(self.success_count, "success_count")
        if success > eligible:
            raise Lfm2IntelligenceGateError(
                "success_count cannot exceed eligible_count."
            )
        if self.retention is not None:
            object.__setattr__(
                self, "retention", _fraction(self.retention, "retention")
            )
        _bool(self.passed, "passed")
        object.__setattr__(
            self, "failure_ids", _string_tuple(self.failure_ids, "failure_ids")
        )
        object.__setattr__(
            self,
            "details",
            MappingProxyType(_canonical_copy(dict(self.details), "component details")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "eligible_count": self.eligible_count,
            "success_count": self.success_count,
            "retention": self.retention,
            "passed": self.passed,
            "failure_ids": list(self.failure_ids),
            "details": dict(self.details),
        }


def _component(
    name: str,
    eligible: Sequence[str],
    success: Sequence[str],
    *,
    passed: bool,
    details: Mapping[str, Any] | None = None,
) -> ComponentResult:
    eligible_ids = tuple(eligible)
    success_ids = tuple(success)
    success_set = set(success_ids)
    return ComponentResult(
        name=name,
        eligible_count=len(eligible_ids),
        success_count=len(success_ids),
        retention=(None if not eligible_ids else len(success_ids) / len(eligible_ids)),
        passed=passed,
        failure_ids=tuple(item for item in eligible_ids if item not in success_set),
        details={} if details is None else details,
    )


@dataclass(frozen=True, slots=True)
class BoundManualRatings:
    packet_sha256: str
    reviewer_id: str
    ratings: tuple[Mapping[str, Any], ...]
    ratings_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "packet_sha256", _digest(self.packet_sha256, "packet_sha256")
        )
        object.__setattr__(self, "reviewer_id", _text(self.reviewer_id, "reviewer_id"))
        normalized = tuple(
            MappingProxyType(_canonical_copy(dict(item), "manual rating"))
            for item in self.ratings
        )
        object.__setattr__(self, "ratings", normalized)
        object.__setattr__(
            self, "ratings_sha256", _digest(self.ratings_sha256, "ratings_sha256")
        )

    @property
    def rating_map(self) -> Mapping[str, Mapping[str, Any]]:
        return MappingProxyType({str(item["case_id"]): item for item in self.ratings})

    def as_dict(self) -> dict[str, Any]:
        core = {
            "schema": MANUAL_RATINGS_SCHEMA,
            "packet_sha256": self.packet_sha256,
            "reviewer_id": self.reviewer_id,
            "exact_wording_comparison_used": False,
            "ratings": [dict(item) for item in self.ratings],
        }
        return {**core, "ratings_sha256": self.ratings_sha256}


def build_manual_review_packet(
    baseline: CachedParentBaseline,
    candidates: Sequence[CandidateOutput],
    policy: IntelligenceGatePolicy,
) -> dict[str, Any]:
    """Build a reference/candidate packet whose rubric forbids wording matching."""

    candidate_map = {item.case_id: item for item in candidates}
    cases = []
    for parent in baseline.cases:
        if not parent.manual_review_required:
            continue
        candidate = candidate_map.get(parent.case_id)
        if candidate is None:
            raise Lfm2IntelligenceGateError(
                f"Manual packet is missing candidate {parent.case_id!r}."
            )
        cases.append(
            {
                "case_id": parent.case_id,
                "category": parent.category,
                "prompt": parent.prompt,
                "reference_response": parent.parent_output,
                "candidate_response": candidate.output,
            }
        )
    core = {
        "schema": MANUAL_PACKET_SCHEMA,
        "baseline_document_sha256": baseline.document_sha256,
        "candidate_outputs_sha256": _candidate_fingerprint(tuple(candidates)),
        "policy_sha256": policy.fingerprint,
        "rubric": {
            "scale": {
                "minimum": 0,
                "maximum": 4,
                "anchor_0": "absent, wrong, or incoherent",
                "anchor_2": "partly useful but materially impaired",
                "anchor_4": "fully useful and coherent",
            },
            "dimensions": [
                "correctness",
                "coherence",
                "instruction_following",
                "capability_retention",
            ],
            "comparison_standard": (
                "Judge semantic usefulness, correctness, and coherence. "
                "Do not use exact wording, token identity, or sentence-structure "
                "matching as a criterion."
            ),
            "exact_wording_comparison_permitted": False,
        },
        "cases": cases,
    }
    return _canonical_copy(
        {**core, "packet_sha256": canonical_sha256(core)},
        "manual review packet",
    )


def load_manual_review_packet(
    source: Mapping[str, Any] | str | Path,
    *,
    expected_packet_sha256: str,
) -> dict[str, Any]:
    document = _document_from(source, "manual review packet")
    core = dict(document)
    seal = core.pop("packet_sha256", None)
    if (
        core.get("schema") != MANUAL_PACKET_SCHEMA
        or seal != _digest(expected_packet_sha256, "expected_packet_sha256")
        or seal != canonical_sha256(core)
        or core.get("rubric", {}).get("exact_wording_comparison_permitted") is not False
    ):
        raise Lfm2IntelligenceGateError(
            "Manual review packet seal or rubric is invalid."
        )
    return document


def bind_manual_ratings(
    packet: Mapping[str, Any],
    ratings: Sequence[Mapping[str, Any]],
    *,
    reviewer_id: str,
    used_exact_wording_comparison: bool = False,
) -> BoundManualRatings:
    """Bind bounded human ratings to one exact packet and semantic rubric."""

    packet_document = load_manual_review_packet(
        packet, expected_packet_sha256=str(packet.get("packet_sha256", ""))
    )
    if _bool(used_exact_wording_comparison, "used_exact_wording_comparison"):
        raise Lfm2IntelligenceGateError(
            "Manual ratings based on exact wording cannot be bound."
        )
    packet_cases = packet_document.get("cases")
    if not isinstance(packet_cases, list):
        raise Lfm2IntelligenceGateError("Manual packet cases are invalid.")
    expected_ids = tuple(str(item["case_id"]) for item in packet_cases)
    normalized = []
    for value in ratings:
        if not isinstance(value, Mapping):
            raise Lfm2IntelligenceGateError("Manual rating must be an object.")
        expected_fields = {
            "case_id",
            "correctness",
            "coherence",
            "instruction_following",
            "capability_retention",
        }
        if set(value) != expected_fields:
            raise Lfm2IntelligenceGateError(
                "Manual rating fields differ from the rating schema."
            )
        row = {"case_id": _text(value["case_id"], "manual case_id")}
        for dimension in expected_fields.difference({"case_id"}):
            score = _nonnegative_int(value[dimension], dimension)
            if score > 4:
                raise Lfm2IntelligenceGateError(f"{dimension} cannot exceed four.")
            row[dimension] = score
        normalized.append(row)
    actual_ids = tuple(item["case_id"] for item in normalized)
    if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != set(expected_ids):
        raise Lfm2IntelligenceGateError(
            "Manual ratings must cover each packet case exactly once."
        )
    normalized.sort(key=lambda item: expected_ids.index(item["case_id"]))
    core = {
        "schema": MANUAL_RATINGS_SCHEMA,
        "packet_sha256": packet_document["packet_sha256"],
        "reviewer_id": _text(reviewer_id, "reviewer_id"),
        "exact_wording_comparison_used": False,
        "ratings": normalized,
    }
    seal = canonical_sha256(core)
    return BoundManualRatings(
        packet_sha256=core["packet_sha256"],
        reviewer_id=core["reviewer_id"],
        ratings=tuple(normalized),
        ratings_sha256=seal,
    )


def load_bound_manual_ratings(
    source: Mapping[str, Any] | str | Path,
    *,
    expected_packet_sha256: str,
    expected_ratings_sha256: str,
) -> BoundManualRatings:
    document = _document_from(source, "manual ratings")
    core = dict(document)
    seal = core.pop("ratings_sha256", None)
    expected_fields = {
        "schema",
        "packet_sha256",
        "reviewer_id",
        "exact_wording_comparison_used",
        "ratings",
    }
    if (
        set(core) != expected_fields
        or core.get("schema") != MANUAL_RATINGS_SCHEMA
        or core.get("packet_sha256")
        != _digest(expected_packet_sha256, "expected_packet_sha256")
        or core.get("exact_wording_comparison_used") is not False
        or seal != _digest(expected_ratings_sha256, "expected_ratings_sha256")
        or seal != canonical_sha256(core)
    ):
        raise Lfm2IntelligenceGateError(
            "Manual ratings seal, packet binding, or rubric declaration is invalid."
        )
    ratings = core.get("ratings")
    if not isinstance(ratings, list):
        raise Lfm2IntelligenceGateError("Manual ratings rows are invalid.")
    # Revalidate fields and bounded values without needing packet text.
    seen = set()
    normalized = []
    for row in ratings:
        if not isinstance(row, Mapping) or set(row) != {
            "case_id",
            "correctness",
            "coherence",
            "instruction_following",
            "capability_retention",
        }:
            raise Lfm2IntelligenceGateError("Manual rating row is invalid.")
        case_id = _text(row["case_id"], "manual case_id")
        if case_id in seen:
            raise Lfm2IntelligenceGateError("Manual rating case IDs are duplicated.")
        seen.add(case_id)
        copied = {"case_id": case_id}
        for dimension in (
            "correctness",
            "coherence",
            "instruction_following",
            "capability_retention",
        ):
            score = _nonnegative_int(row[dimension], dimension)
            if score > 4:
                raise Lfm2IntelligenceGateError(f"{dimension} cannot exceed four.")
            copied[dimension] = score
        normalized.append(copied)
    return BoundManualRatings(
        packet_sha256=core["packet_sha256"],
        reviewer_id=core["reviewer_id"],
        ratings=tuple(normalized),
        ratings_sha256=seal,
    )


@dataclass(frozen=True, slots=True)
class IntelligenceGateReport:
    baseline_document_sha256: str
    candidate_outputs_sha256: str
    policy_sha256: str
    components: tuple[ComponentResult, ...]
    topical_relevance_diagnostic: Mapping[str, Any]
    manual_review: Mapping[str, Any]
    headline_lower_bound: float | None
    headline_inputs: Mapping[str, float]
    promoted: bool
    reasons: tuple[str, ...]
    report_sha256: str

    @property
    def component_map(self) -> Mapping[str, ComponentResult]:
        return MappingProxyType({item.name: item for item in self.components})

    def as_dict(self) -> dict[str, Any]:
        core = {
            "schema": REPORT_SCHEMA,
            "baseline_document_sha256": self.baseline_document_sha256,
            "candidate_outputs_sha256": self.candidate_outputs_sha256,
            "policy_sha256": self.policy_sha256,
            "components": [item.as_dict() for item in self.components],
            "topical_relevance_diagnostic": dict(self.topical_relevance_diagnostic),
            "manual_review": dict(self.manual_review),
            "headline_lower_bound": self.headline_lower_bound,
            "headline_inputs": dict(self.headline_inputs),
            "promoted": self.promoted,
            "reasons": list(self.reasons),
            "parent_forward_passes": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }
        return {**core, "report_sha256": self.report_sha256}


def evaluate_cached_intelligence(
    baseline: CachedParentBaseline,
    candidate_values: Sequence[CandidateOutput | Mapping[str, Any]],
    *,
    policy: IntelligenceGatePolicy | None = None,
    manual_ratings: BoundManualRatings | None = None,
) -> IntelligenceGateReport:
    """Evaluate one candidate strictly from cached parent and candidate outputs."""

    if not isinstance(baseline, CachedParentBaseline):
        raise Lfm2IntelligenceGateError(
            "baseline must be an authenticated CachedParentBaseline."
        )
    selected_policy = IntelligenceGatePolicy() if policy is None else policy
    if not isinstance(selected_policy, IntelligenceGatePolicy):
        raise Lfm2IntelligenceGateError("policy must be IntelligenceGatePolicy.")
    candidates = tuple(
        item
        if isinstance(item, CandidateOutput)
        else CandidateOutput.from_mapping(item)
        for item in candidate_values
    )
    candidate_ids = tuple(item.case_id for item in candidates)
    baseline_ids = tuple(item.case_id for item in baseline.cases)
    if len(set(candidate_ids)) != len(candidate_ids) or set(candidate_ids) != set(
        baseline_ids
    ):
        raise Lfm2IntelligenceGateError(
            "Candidate outputs must cover every baseline case exactly once."
        )
    candidate_by_id = {item.case_id: item for item in candidates}
    candidates = tuple(candidate_by_id[item] for item in baseline_ids)
    for parent, candidate in zip(baseline.cases, candidates):
        if candidate.generation.generation_limit != parent.generation.generation_limit:
            raise Lfm2IntelligenceGateError(
                f"Generation limit differs for {parent.case_id!r}."
            )

    validator_rows: dict[str, tuple[bool, bool, dict[str, Any], dict[str, Any]]] = {}
    for parent, candidate in zip(baseline.cases, candidates):
        if parent.validator is None:
            continue
        parent_pass, parent_detail = _validator_result(
            parent.parent_output, parent.validator
        )
        candidate_pass, candidate_detail = _validator_result(
            candidate.output, parent.validator
        )
        validator_rows[parent.case_id] = (
            parent_pass,
            candidate_pass,
            parent_detail,
            candidate_detail,
        )

    correctness_eligible = tuple(validator_rows)
    parent_correct = tuple(
        case_id for case_id, value in validator_rows.items() if value[0]
    )
    retained_correct = tuple(
        case_id for case_id in parent_correct if validator_rows[case_id][1]
    )
    candidate_correct = tuple(
        case_id for case_id, value in validator_rows.items() if value[1]
    )
    correctness_retention = (
        None if not parent_correct else len(retained_correct) / len(parent_correct)
    )
    correctness_absolute = (
        None
        if not correctness_eligible
        else len(candidate_correct) / len(correctness_eligible)
    )
    parent_correctness_absolute = (
        None
        if not correctness_eligible
        else len(parent_correct) / len(correctness_eligible)
    )
    correctness_absolute_retention = (
        None
        if parent_correctness_absolute in (None, 0.0)
        or correctness_absolute is None
        else min(1.0, correctness_absolute / parent_correctness_absolute)
    )
    minimum_candidate_absolute_accuracy = (
        None
        if parent_correctness_absolute is None
        else (
            parent_correctness_absolute
            * selected_policy.minimum_absolute_task_correctness
        )
    )
    correctness_pass = bool(
        correctness_retention is not None
        and correctness_absolute is not None
        and correctness_absolute_retention is not None
        and minimum_candidate_absolute_accuracy is not None
        and correctness_retention >= selected_policy.minimum_task_correctness_retention
        and correctness_absolute >= minimum_candidate_absolute_accuracy
    )
    task_component = ComponentResult(
        name="task_correctness",
        eligible_count=len(parent_correct),
        success_count=len(retained_correct),
        retention=correctness_retention,
        passed=correctness_pass,
        failure_ids=tuple(
            case_id for case_id in parent_correct if case_id not in retained_correct
        ),
        details={
            "reliable_validator_case_count": len(correctness_eligible),
            "parent_correct_count": len(parent_correct),
            "candidate_correct_count": len(candidate_correct),
            "parent_absolute_accuracy": parent_correctness_absolute,
            "candidate_absolute_accuracy": correctness_absolute,
            "candidate_absolute_accuracy_retention": (
                correctness_absolute_retention
            ),
            "minimum_retention": (selected_policy.minimum_task_correctness_retention),
            "minimum_absolute_accuracy_ratio_to_parent": (
                selected_policy.minimum_absolute_task_correctness
            ),
            "minimum_candidate_absolute_accuracy": (
                minimum_candidate_absolute_accuracy
            ),
            "per_case": {
                case_id: {
                    "parent_passed": value[0],
                    "candidate_passed": value[1],
                    "parent_detail": value[2],
                    "candidate_detail": value[3],
                }
                for case_id, value in validator_rows.items()
            },
        },
    )

    capability_eligible = tuple(
        item.case_id for item in baseline.cases if item.capability is not None
    )
    capability_success = tuple(
        parent.case_id
        for parent, candidate in zip(baseline.cases, candidates)
        if parent.capability is not None and parent.capability.passes(candidate.output)
    )
    capability_rate = (
        None
        if not capability_eligible
        else len(capability_success) / len(capability_eligible)
    )
    capability_component = _component(
        "parent_capability",
        capability_eligible,
        capability_success,
        passed=bool(
            capability_rate is not None
            and capability_rate >= selected_policy.minimum_parent_capability_retention
        ),
        details={
            "minimum_retention": (selected_policy.minimum_parent_capability_retention),
            "contracts_are_parent_demonstrated": True,
        },
    )

    parent_nonempty = tuple(
        item.case_id for item in baseline.cases if item.parent_output.strip()
    )
    nonempty_success = tuple(
        parent.case_id
        for parent, candidate in zip(baseline.cases, candidates)
        if parent.parent_output.strip() and candidate.output.strip()
    )
    new_empty = tuple(
        parent.case_id
        for parent, candidate in zip(baseline.cases, candidates)
        if parent.parent_output.strip() and not candidate.output.strip()
    )
    nonempty_rate = (
        None if not parent_nonempty else len(nonempty_success) / len(parent_nonempty)
    )
    nonempty_component = _component(
        "nonempty_retention",
        parent_nonempty,
        nonempty_success,
        passed=bool(
            nonempty_rate is not None
            and nonempty_rate >= selected_policy.minimum_nonempty_retention
            and len(new_empty) <= selected_policy.maximum_new_empty_responses
        ),
        details={
            "new_empty_count": len(new_empty),
            "new_empty_ids": list(new_empty),
            "maximum_new_empty_responses": (
                selected_policy.maximum_new_empty_responses
            ),
            "minimum_retention": selected_policy.minimum_nonempty_retention,
        },
    )

    parent_nonfragment = tuple(
        item.case_id
        for item in baseline.cases
        if not item.generation.fragment_like
        and len(item.parent_output.strip())
        >= selected_policy.fragment_minimum_characters
    )
    candidate_fragment = {
        candidate.case_id: (
            candidate.generation.fragment_like
            or (
                bool(candidate.output.strip())
                and len(candidate.output.strip())
                < selected_policy.fragment_minimum_characters
            )
        )
        for candidate in candidates
    }
    fragment_success = tuple(
        case_id for case_id in parent_nonfragment if not candidate_fragment[case_id]
    )
    new_fragment = tuple(
        case_id for case_id in parent_nonfragment if candidate_fragment[case_id]
    )
    fragment_rate = (
        None
        if not parent_nonfragment
        else len(fragment_success) / len(parent_nonfragment)
    )
    fragment_component = _component(
        "fragment_retention",
        parent_nonfragment,
        fragment_success,
        passed=bool(
            fragment_rate is not None
            and fragment_rate >= selected_policy.minimum_fragment_retention
            and len(new_fragment) <= selected_policy.maximum_new_fragment_responses
        ),
        details={
            "new_fragment_count": len(new_fragment),
            "new_fragment_ids": list(new_fragment),
            "maximum_new_fragment_responses": (
                selected_policy.maximum_new_fragment_responses
            ),
            "minimum_retention": selected_policy.minimum_fragment_retention,
        },
    )

    eos_eligible = baseline_ids
    eos_regressions = []
    new_saturations = []
    missing_eos = []
    for parent, candidate in zip(baseline.cases, candidates):
        new_limit = (
            candidate.generation.unfinished_at_generation_limit
            and not parent.generation.unfinished_at_generation_limit
        )
        lost_eos = (
            parent.generation.eos_observed and not candidate.generation.eos_observed
        )
        if new_limit:
            new_saturations.append(parent.case_id)
        if lost_eos:
            missing_eos.append(parent.case_id)
        if new_limit or lost_eos:
            eos_regressions.append(parent.case_id)
    eos_success = tuple(
        case_id for case_id in eos_eligible if case_id not in set(eos_regressions)
    )
    eos_component = _component(
        "eos_generation_limit",
        eos_eligible,
        eos_success,
        passed=(
            len(eos_regressions) <= selected_policy.maximum_eos_or_limit_regressions
        ),
        details={
            "regression_count": len(eos_regressions),
            "new_generation_limit_saturation_ids": new_saturations,
            "missing_eos_ids": missing_eos,
            "maximum_regressions": (selected_policy.maximum_eos_or_limit_regressions),
        },
    )

    repetition_regressions = []
    repetition_detail: dict[str, Any] = {}
    for parent, candidate in zip(baseline.cases, candidates):
        parent_share = _repeated_bigram_share(parent.generation.token_ids)
        candidate_share = _repeated_bigram_share(candidate.generation.token_ids)
        candidate_run = _longest_token_run(candidate.generation.token_ids)
        parent_run = _longest_token_run(parent.generation.token_ids)
        enough_tokens = (
            candidate.generation.generated_token_count
            >= selected_policy.degeneration_minimum_tokens
        )
        newly_degenerate = (
            enough_tokens
            and candidate_share >= selected_policy.degeneration_bigram_share
            and parent_share < selected_policy.degeneration_bigram_share
        ) or (candidate_run >= 4 and parent_run < 4)
        regressed = (
            enough_tokens
            and candidate_share - parent_share
            > selected_policy.maximum_repeated_bigram_share_delta + 1e-12
        ) or newly_degenerate
        if regressed:
            repetition_regressions.append(parent.case_id)
        repetition_detail[parent.case_id] = {
            "parent_repeated_bigram_share": parent_share,
            "candidate_repeated_bigram_share": candidate_share,
            "share_delta": candidate_share - parent_share,
            "parent_longest_token_run": parent_run,
            "candidate_longest_token_run": candidate_run,
            "newly_degenerate": newly_degenerate,
            "regressed": regressed,
        }
    repetition_success = tuple(
        case_id
        for case_id in baseline_ids
        if case_id not in set(repetition_regressions)
    )
    repetition_component = _component(
        "repetition_degeneration",
        baseline_ids,
        repetition_success,
        passed=(
            len(repetition_regressions)
            <= selected_policy.maximum_repetition_regressions
        ),
        details={
            "regression_count": len(repetition_regressions),
            "maximum_regressions": (selected_policy.maximum_repetition_regressions),
            "maximum_repeated_bigram_share_delta": (
                selected_policy.maximum_repeated_bigram_share_delta
            ),
            "degeneration_bigram_share": (selected_policy.degeneration_bigram_share),
            "degeneration_minimum_tokens": (
                selected_policy.degeneration_minimum_tokens
            ),
            "per_case": repetition_detail,
        },
    )

    code_ids = tuple(
        item.case_id
        for item in baseline.cases
        if item.validator is not None and item.validator.kind == "python_pure_function"
    )
    code_syntax_success = tuple(
        case_id
        for case_id in code_ids
        if bool(validator_rows[case_id][3].get("syntax_passed"))
    )
    code_behavior_parent = tuple(
        case_id for case_id in code_ids if validator_rows[case_id][0]
    )
    code_behavior_success = tuple(
        case_id
        for case_id in code_behavior_parent
        if bool(validator_rows[case_id][3].get("behavior_passed"))
    )
    code_syntax_rate = (
        None if not code_ids else len(code_syntax_success) / len(code_ids)
    )
    code_behavior_rate = (
        None
        if not code_behavior_parent
        else len(code_behavior_success) / len(code_behavior_parent)
    )
    code_syntax_component = _component(
        "code_syntax",
        code_ids,
        code_syntax_success,
        passed=bool(
            code_syntax_rate is not None
            and code_syntax_rate >= selected_policy.minimum_code_syntax_rate
        ),
        details={
            "minimum_rate": selected_policy.minimum_code_syntax_rate,
            "parsing_executes_no_candidate_code": True,
        },
    )
    code_behavior_component = _component(
        "code_behavior",
        code_behavior_parent,
        code_behavior_success,
        passed=bool(
            code_behavior_rate is not None
            and code_behavior_rate >= selected_policy.minimum_code_behavior_retention
        ),
        details={
            "minimum_retention": (selected_policy.minimum_code_behavior_retention),
            "safe_ast_interpreter": True,
            "python_exec_used": False,
        },
    )

    arithmetic_parent = tuple(
        item.case_id
        for item in baseline.cases
        if item.validator is not None
        and item.validator.kind == "arithmetic_exact"
        and validator_rows[item.case_id][0]
    )
    arithmetic_success = tuple(
        case_id for case_id in arithmetic_parent if validator_rows[case_id][1]
    )
    arithmetic_rate = (
        None
        if not arithmetic_parent
        else len(arithmetic_success) / len(arithmetic_parent)
    )
    arithmetic_component = _component(
        "arithmetic_exactness",
        arithmetic_parent,
        arithmetic_success,
        passed=bool(
            arithmetic_rate is not None
            and arithmetic_rate >= selected_policy.minimum_arithmetic_retention
        ),
        details={
            "minimum_retention": selected_policy.minimum_arithmetic_retention,
            "criterion": "final finite numeric value equals expected value",
        },
    )

    instruction_parent = tuple(
        item.case_id
        for item in baseline.cases
        if item.validator is not None
        and item.validator.kind == "instruction_constraints"
        and validator_rows[item.case_id][0]
    )
    instruction_success = tuple(
        case_id for case_id in instruction_parent if validator_rows[case_id][1]
    )
    instruction_rate = (
        None
        if not instruction_parent
        else len(instruction_success) / len(instruction_parent)
    )
    instruction_component = _component(
        "instruction_compliance",
        instruction_parent,
        instruction_success,
        passed=bool(
            instruction_rate is not None
            and instruction_rate >= selected_policy.minimum_instruction_retention
        ),
        details={
            "minimum_retention": selected_policy.minimum_instruction_retention,
            "explicit_constraints_only": True,
        },
    )

    components = (
        task_component,
        capability_component,
        nonempty_component,
        fragment_component,
        eos_component,
        repetition_component,
        code_syntax_component,
        code_behavior_component,
        arithmetic_component,
        instruction_component,
    )
    component_map = {item.name: item for item in components}

    topical_scored = tuple(
        item for item in candidates if item.topical_relevance is not None
    )
    topical_passed = tuple(item for item in topical_scored if item.topical_relevance)
    topical_diagnostic = {
        "eligible_count": len(topical_scored),
        "pass_count": len(topical_passed),
        "rate": (
            None if not topical_scored else len(topical_passed) / len(topical_scored)
        ),
        "promotion_eligible": False,
        "warning": (
            "Topical relevance is not task correctness, coherence, or "
            "intelligence retention."
        ),
    }

    packet = build_manual_review_packet(baseline, candidates, selected_policy)
    manual_case_ids = tuple(
        item.case_id for item in baseline.cases if item.manual_review_required
    )
    manual_review: dict[str, Any]
    manual_headline: float | None
    if not manual_case_ids:
        manual_review = {
            "required": False,
            "status": "not_applicable",
            "packet_sha256": packet["packet_sha256"],
            "case_count": 0,
            "passed": True,
            "exact_wording_comparison_used": False,
        }
        manual_headline = 1.0
    elif manual_ratings is None:
        manual_review = {
            "required": True,
            "status": "pending",
            "packet_sha256": packet["packet_sha256"],
            "case_count": len(manual_case_ids),
            "passed": False,
            "exact_wording_comparison_used": False,
        }
        manual_headline = None
    else:
        if not isinstance(manual_ratings, BoundManualRatings):
            raise Lfm2IntelligenceGateError(
                "manual_ratings must be authenticated BoundManualRatings."
            )
        if manual_ratings.packet_sha256 != packet["packet_sha256"]:
            raise Lfm2IntelligenceGateError(
                "Manual ratings are bound to a different review packet."
            )
        if set(manual_ratings.rating_map) != set(manual_case_ids):
            raise Lfm2IntelligenceGateError(
                "Manual ratings do not cover the required review cases."
            )
        means = {}
        minimums = {}
        successes = []
        for case_id in manual_case_ids:
            row = manual_ratings.rating_map[case_id]
            scores = tuple(
                int(row[name])
                for name in (
                    "correctness",
                    "coherence",
                    "instruction_following",
                    "capability_retention",
                )
            )
            mean = sum(scores) / len(scores)
            means[case_id] = mean
            minimums[case_id] = min(scores)
            if (
                min(scores) >= selected_policy.manual_minimum_dimension
                and mean >= selected_policy.manual_minimum_mean
            ):
                successes.append(case_id)
        manual_rate = len(successes) / len(manual_case_ids)
        manual_pass = manual_rate >= selected_policy.minimum_manual_case_pass_rate
        manual_review = {
            "required": True,
            "status": "accepted" if manual_pass else "rejected",
            "packet_sha256": packet["packet_sha256"],
            "ratings_sha256": manual_ratings.ratings_sha256,
            "reviewer_id": manual_ratings.reviewer_id,
            "case_count": len(manual_case_ids),
            "case_pass_count": len(successes),
            "case_pass_rate": manual_rate,
            "case_means": means,
            "case_minimum_dimensions": minimums,
            "minimum_dimension": selected_policy.manual_minimum_dimension,
            "minimum_mean": selected_policy.manual_minimum_mean,
            "minimum_case_pass_rate": (selected_policy.minimum_manual_case_pass_rate),
            "passed": manual_pass,
            "exact_wording_comparison_used": False,
        }
        manual_headline = manual_rate

    reasons = []
    for name in selected_policy.required_components:
        component = component_map[name]
        if component.retention is None:
            reasons.append(f"{name}:missing_reliable_coverage")
        elif not component.passed:
            reasons.append(f"{name}:threshold_failed")
    if manual_case_ids and manual_review["status"] == "pending":
        reasons.append("manual_review:pending")
    elif manual_case_ids and not manual_review["passed"]:
        reasons.append("manual_review:threshold_failed")

    headline_inputs: dict[str, float] = {}
    for name in selected_policy.required_components:
        component = component_map[name]
        if component.retention is not None:
            headline_inputs[name] = component.retention
    if correctness_absolute_retention is not None:
        headline_inputs["task_correctness_absolute_retention"] = (
            correctness_absolute_retention
        )
    if manual_headline is not None:
        headline_inputs["manual_review"] = manual_headline
    expected_headline_input_count = (
        len(selected_policy.required_components)
        + 1
        + int("task_correctness" in selected_policy.required_components)
    )
    all_headline_inputs_present = len(headline_inputs) == expected_headline_input_count
    headline = min(headline_inputs.values()) if all_headline_inputs_present else None
    if (
        headline is not None
        and headline + 1e-12 < selected_policy.minimum_headline_lower_bound
    ):
        reasons.append("headline_lower_bound:threshold_failed")
    if headline is None:
        reasons.append("headline_lower_bound:withheld_until_all_components")
    promoted = not reasons

    core = {
        "schema": REPORT_SCHEMA,
        "baseline_document_sha256": baseline.document_sha256,
        "candidate_outputs_sha256": _candidate_fingerprint(candidates),
        "policy_sha256": selected_policy.fingerprint,
        "components": [item.as_dict() for item in components],
        "topical_relevance_diagnostic": topical_diagnostic,
        "manual_review": manual_review,
        "headline_lower_bound": headline,
        "headline_inputs": headline_inputs,
        "promoted": promoted,
        "reasons": reasons,
        "parent_forward_passes": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    report_sha = canonical_sha256(core)
    return IntelligenceGateReport(
        baseline_document_sha256=baseline.document_sha256,
        candidate_outputs_sha256=core["candidate_outputs_sha256"],
        policy_sha256=selected_policy.fingerprint,
        components=components,
        topical_relevance_diagnostic=MappingProxyType(topical_diagnostic),
        manual_review=MappingProxyType(manual_review),
        headline_lower_bound=headline,
        headline_inputs=MappingProxyType(headline_inputs),
        promoted=promoted,
        reasons=tuple(reasons),
        report_sha256=report_sha,
    )


def verify_intelligence_report(
    source: Mapping[str, Any] | str | Path,
    *,
    expected_report_sha256: str,
    expected_baseline_document_sha256: str,
    expected_candidate_outputs_sha256: str,
    expected_policy_sha256: str,
) -> dict[str, Any]:
    """Reject a modified, noncanonical, or identity-mismatched gate report."""

    document = _document_from(source, "intelligence report")
    core = dict(document)
    seal = core.pop("report_sha256", None)
    if (
        core.get("schema") != REPORT_SCHEMA
        or seal != _digest(expected_report_sha256, "expected_report_sha256")
        or seal != canonical_sha256(core)
        or core.get("baseline_document_sha256")
        != _digest(
            expected_baseline_document_sha256,
            "expected_baseline_document_sha256",
        )
        or core.get("candidate_outputs_sha256")
        != _digest(
            expected_candidate_outputs_sha256,
            "expected_candidate_outputs_sha256",
        )
        or core.get("policy_sha256")
        != _digest(expected_policy_sha256, "expected_policy_sha256")
        or core.get("parent_forward_passes") != 0
        or core.get("optimizer_steps") != 0
        or core.get("backward_calls") != 0
    ):
        raise Lfm2IntelligenceGateError(
            "Intelligence report seal or identity binding is invalid."
        )
    return document


__all__ = [
    "BASELINE_SCHEMA",
    "HOLDOUT_CASE_COUNT",
    "HOLDOUT_MANIFEST_SCHEMA",
    "HOLDOUT_SPLIT",
    "REPORT_SCHEMA",
    "MANUAL_PACKET_SCHEMA",
    "MANUAL_RATINGS_SCHEMA",
    "Lfm2IntelligenceGateError",
    "GenerationEvidence",
    "CapabilitySpec",
    "PureFunctionExample",
    "ReliableValidatorSpec",
    "IntelligenceHoldoutCaseSpec",
    "IntelligenceHoldoutManifest",
    "CachedParentCase",
    "CachedParentBaseline",
    "CandidateOutput",
    "IntelligenceGatePolicy",
    "ComponentResult",
    "BoundManualRatings",
    "IntelligenceGateReport",
    "build_lfm2_intelligence_holdout_manifest",
    "load_lfm2_intelligence_holdout_manifest",
    "seal_cached_parent_rows",
    "load_cached_parent_rows",
    "build_manual_review_packet",
    "load_manual_review_packet",
    "bind_manual_ratings",
    "load_bound_manual_ratings",
    "evaluate_cached_intelligence",
    "verify_intelligence_report",
]
