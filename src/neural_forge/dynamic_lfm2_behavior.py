"""Parent-bound cached whole-output intelligence gate for production LFM2."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .dynamic_behavior_gate import (
    BehaviorCallableIdentity,
    CachedWholeOutputBehaviorGate,
    CandidateWholeOutput,
    DynamicBehaviorGateError,
    TaskScoreCoveragePolicy,
    WholeOutputCase,
    WholeOutputGateConfig,
    WholeOutputMetadata,
    expose_behavior_gate_binding,
    identify_behavior_callable,
    resolve_behavior_gate_binding,
)
from .dynamic_circuit_ir import canonical_sha256
from .dynamic_fold_executor import FoldDecision


SCHEMA = "neural-forge-lfm2-cached-whole-output-v2"
ENVELOPE_SCHEMA = "neural-forge-lfm2-whole-output-envelope-v1"
GENERATION_PROTOCOL = {
    "do_sample": False,
    "use_cache": True,
    "skip_special_tokens": True,
    "tokenizer_cleanup": "tokenizer_default",
}


class DynamicLfm2BehaviorError(RuntimeError):
    """Cached outputs or child generation violate the behavior contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicLfm2BehaviorError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicLfm2BehaviorError(f"{name} must be a positive integer.")
    return value


def _maximum_repeated_bigram_share(token_ids: list[int] | tuple[int, ...]) -> float:
    tokens = tuple(int(value) for value in token_ids)
    bigrams = tuple(zip(tokens, tokens[1:]))
    if not bigrams:
        return 0.0
    counts: dict[tuple[int, int], int] = {}
    for bigram in bigrams:
        counts[bigram] = counts.get(bigram, 0) + 1
    return max(counts.values()) / len(bigrams)


def _numbers(value: str) -> tuple[Decimal, ...]:
    result = []
    for token in re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?!\w)", value):
        try:
            result.append(Decimal(token))
        except InvalidOperation:
            continue
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Lfm2TaskContract:
    case_id: str
    kind: str
    required_groups: tuple[tuple[str, ...], ...] = ()
    expected_numeric: str | None = None
    exact_word: str | None = None
    function_name: str | None = None
    return_operator: str | None = None
    minimum_characters: int = 1
    minimum_words: int = 1

    def _keyword_relevance(self, output: str) -> bool:
        normalized = output.casefold()
        return all(
            any(value.casefold() in normalized for value in group)
            for group in self.required_groups
        )

    def relevance_score(self, output: str) -> bool:
        text = str(output)
        stripped = text.strip()
        if len(stripped) < self.minimum_characters:
            return False
        if len(re.findall(r"\b[\w'-]+\b", stripped)) < self.minimum_words:
            return False
        if self.kind == "numeric":
            return bool(_numbers(stripped))
        if self.kind == "python":
            if self.function_name is None:
                return False
            definition = re.search(
                rf"(?m)^\s*def\s+{re.escape(self.function_name)}\s*\(",
                text,
            )
            return (
                definition is not None
                and re.search(
                    r"(?m)^\s+return\b",
                    text,
                )
                is not None
            )
        if self.kind == "exact_word":
            return (
                self.exact_word is not None
                and re.fullmatch(r"[A-Za-z]+", stripped) is not None
                and stripped.casefold() == self.exact_word.casefold()
            )
        if self.kind in {"keywords", "creative"}:
            return self._keyword_relevance(text)
        raise DynamicLfm2BehaviorError(f"Unknown task-contract kind {self.kind!r}.")

    def correctness_score(self, output: str) -> bool | None:
        if self.kind == "numeric" and self.expected_numeric is not None:
            try:
                expected = Decimal(self.expected_numeric)
            except InvalidOperation as exc:
                raise DynamicLfm2BehaviorError(
                    f"Invalid expected number for {self.case_id!r}."
                ) from exc
            return expected in _numbers(output)
        if self.kind == "python" and self.return_operator is not None:
            if not self.relevance_score(output):
                return False
            operator = re.escape(self.return_operator)
            return (
                re.search(
                    rf"(?m)^\s+return\s+a\s*{operator}\s*b\s*(?:#.*)?$",
                    output,
                )
                is not None
            )
        if self.kind == "exact_word":
            return self.relevance_score(output)
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "kind": self.kind,
            "required_groups": [list(group) for group in self.required_groups],
            "expected_numeric": self.expected_numeric,
            "exact_word": self.exact_word,
            "function_name": self.function_name,
            "return_operator": self.return_operator,
            "minimum_characters": self.minimum_characters,
            "minimum_words": self.minimum_words,
        }


_MANUAL_KEYWORDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "manual_chat_00": (
        ("ai", "assistant", "model"),
        ("help", "assist"),
    ),
    "manual_fact_00": (
        ("rain", "water", "precipitation"),
        ("cloud", "condensation"),
        ("fall", "gravity", "droplet"),
    ),
    "manual_ai_00": (
        ("token",),
        ("context", "coherent", "sequence"),
    ),
    "manual_compare_00": (
        ("ram",),
        ("disk", "storage"),
        ("temporary", "volatile"),
        ("long-term", "permanent", "persistent", "durable"),
    ),
    "manual_fold_00": (
        ("residual",),
        ("attention",),
        ("fold", "merge", "compress"),
        ("neuron", "weight", "circuit"),
    ),
}


def _python_contract(
    case_id: str,
    prompt: str,
    expected: str | None,
) -> Lfm2TaskContract:
    function_name = None
    return_operator = None
    if expected:
        match = re.search(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(", expected)
        if match is not None:
            function_name = match.group(1)
        operator_match = re.search(
            r"(?m)^\s*return\s+a\s*([+\-*])\s*b\s*$",
            expected,
        )
        if operator_match is not None:
            return_operator = operator_match.group(1)
    if function_name is None:
        match = re.search(
            r"(?:named|function)\s+([A-Za-z_]\w*)",
            prompt,
        )
        if match is not None:
            function_name = match.group(1)
    if function_name is None:
        raise DynamicLfm2BehaviorError(
            f"Could not derive a Python function contract for {case_id!r}."
        )
    return Lfm2TaskContract(
        case_id=case_id,
        kind="python",
        function_name=function_name,
        return_operator=return_operator,
        minimum_characters=8,
        minimum_words=2,
    )


def _task_contract(
    *,
    case_id: str,
    category: str,
    prompt: str,
    expected: str | None,
    parent_output: str,
) -> Lfm2TaskContract | None:
    if category == "math":
        exact = expected
        if case_id == "manual_math_00":
            exact = "378"
        return Lfm2TaskContract(
            case_id=case_id,
            kind="numeric",
            expected_numeric=exact,
        )
    if category == "code":
        contract = _python_contract(case_id, prompt, expected)
        if case_id == "manual_code_00":
            return Lfm2TaskContract(
                case_id=case_id,
                kind="keywords",
                required_groups=(
                    ("def is_palindrome",),
                    ("return",),
                    ("[::-1]",),
                    ("lower", "casefold"),
                ),
                minimum_characters=20,
                minimum_words=3,
            )
        return contract
    if case_id == "manual_instruction_00":
        return Lfm2TaskContract(
            case_id=case_id,
            kind="exact_word",
            exact_word="blue",
        )
    if case_id == "manual_math_00":
        return Lfm2TaskContract(
            case_id=case_id,
            kind="numeric",
            expected_numeric="378",
        )
    if case_id == "manual_reason_00":
        return Lfm2TaskContract(
            case_id=case_id,
            kind="keywords",
            required_groups=(
                ("no", "cannot"),
                ("bloop",),
                ("razz",),
                ("green",),
            ),
            minimum_characters=12,
            minimum_words=5,
        )
    if case_id == "manual_safety_00":
        return Lfm2TaskContract(
            case_id=case_id,
            kind="keywords",
            required_groups=(
                ("leave", "evacuate", "outside"),
                ("call", "emergency", "gas company", "fire"),
            ),
            minimum_characters=12,
            minimum_words=4,
        )
    if case_id == "manual_summary_00":
        return Lfm2TaskContract(
            case_id=case_id,
            kind="keywords",
            required_groups=(
                ("activation",),
                ("error",),
                ("compound", "accumulate"),
                ("attention", "residual"),
            ),
            minimum_characters=20,
            minimum_words=6,
        )
    if case_id == "manual_creative_00":
        return Lfm2TaskContract(
            case_id=case_id,
            kind="creative",
            required_groups=(("robot",),),
            minimum_characters=50,
            minimum_words=12,
        )
    groups = _MANUAL_KEYWORDS.get(case_id)
    if groups is not None:
        return Lfm2TaskContract(
            case_id=case_id,
            kind="keywords",
            required_groups=groups,
            minimum_characters=12,
            minimum_words=4,
        )
    if parent_output.strip():
        raise DynamicLfm2BehaviorError(
            f"Nonempty parent case {case_id!r} has no semantic validator."
        )
    return None


def seal_lfm2_cached_whole_output_envelope(
    source_rows: str | Path,
    destination: str | Path,
    *,
    parent_cache_key: str,
    parent_state_fingerprint: str,
    tokenizer_fingerprint: str,
    generation_limit: int = 64,
) -> dict[str, Any]:
    """Bind existing one-time parent responses to their exact runtime identities."""

    source = Path(source_rows).expanduser().resolve(strict=True)
    target = Path(destination).expanduser().absolute()
    if not source.is_file() or source.is_symlink():
        raise DynamicLfm2BehaviorError(
            f"Parent response rows are missing or unsafe: {source}."
        )
    if target.exists():
        raise DynamicLfm2BehaviorError(
            f"Whole-output envelope already exists: {target}."
        )
    raw = source.read_bytes()
    try:
        rows = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DynamicLfm2BehaviorError(
            "Parent response rows are not valid UTF-8 JSON."
        ) from exc
    if not isinstance(rows, list) or not rows:
        raise DynamicLfm2BehaviorError(
            "Parent response rows must be a nonempty JSON list."
        )
    body = {
        "schema": ENVELOPE_SCHEMA,
        "parent_cache_key": _digest(parent_cache_key, "parent_cache_key"),
        "parent_state_fingerprint": _digest(
            parent_state_fingerprint,
            "parent_state_fingerprint",
        ),
        "tokenizer_fingerprint": _digest(
            tokenizer_fingerprint,
            "tokenizer_fingerprint",
        ),
        "generation": {
            **GENERATION_PROTOCOL,
            "max_new_tokens": _positive_int(
                generation_limit,
                "generation_limit",
            ),
        },
        "source_rows_sha256": _sha256_bytes(raw),
        "rows": rows,
        "parent_outputs_generated_once": True,
        "parent_forward_passes_after_seal": 0,
    }
    envelope = {
        **body,
        "envelope_sha256": canonical_sha256(body),
    }
    payload = _canonical_bytes(envelope) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(target),
        "file_sha256": _sha256_bytes(payload),
        "envelope_sha256": envelope["envelope_sha256"],
        "source_rows_sha256": body["source_rows_sha256"],
        "case_count": len(rows),
    }


@dataclass(frozen=True, slots=True)
class Lfm2CachedWholeOutputBaseline:
    source_path: str
    source_sha256: str
    envelope_sha256: str
    parent_cache_key: str
    parent_state_fingerprint: str
    tokenizer_fingerprint: str
    cases: tuple[WholeOutputCase, ...]
    contracts: tuple[tuple[str, Lfm2TaskContract | None], ...]
    generation_limit: int

    @property
    def contract_map(self) -> dict[str, Lfm2TaskContract | None]:
        return dict(self.contracts)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def task_score(self, case: WholeOutputCase, output: str) -> bool | None:
        contract = self.contract_map.get(case.case_id)
        if contract is None:
            return None
        return contract.relevance_score(output)

    def correctness_score(
        self,
        case: WholeOutputCase,
        output: str,
    ) -> bool | None:
        contract = self.contract_map.get(case.case_id)
        if contract is None:
            return None
        return contract.correctness_score(output)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "envelope_sha256": self.envelope_sha256,
            "parent_cache_key": self.parent_cache_key,
            "parent_state_fingerprint": self.parent_state_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "generation_limit": self.generation_limit,
            "generation_protocol": dict(GENERATION_PROTOCOL),
            "cases": [case.as_dict() for case in self.cases],
            "contracts": {
                case_id: (None if contract is None else contract.as_dict())
                for case_id, contract in self.contracts
            },
            "parent_forward_passes": 0,
        }


def load_lfm2_cached_whole_output_baseline(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_parent_cache_key: str,
    expected_parent_state_fingerprint: str,
    expected_tokenizer_fingerprint: str,
    expected_generation_limit: int = 64,
    fragment_minimum_characters: int = 12,
) -> Lfm2CachedWholeOutputBaseline:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise DynamicLfm2BehaviorError(
            f"Cached parent output envelope is missing or unsafe: {source}."
        )
    raw = source.read_bytes()
    source_sha256 = _sha256_bytes(raw)
    if source_sha256 != _digest(expected_sha256, "expected_sha256"):
        raise DynamicLfm2BehaviorError(
            "Cached parent output fingerprint differs from the supplied digest."
        )
    try:
        envelope = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DynamicLfm2BehaviorError(
            "Cached parent output envelope is not canonical JSON."
        ) from exc
    if not isinstance(envelope, Mapping):
        raise DynamicLfm2BehaviorError(
            "Cached parent output envelope must be an object."
        )
    body = dict(envelope)
    seal = body.pop("envelope_sha256", None)
    if (
        body.get("schema") != ENVELOPE_SCHEMA
        or seal != canonical_sha256(body)
        or _canonical_bytes(envelope) + b"\n" != raw
    ):
        raise DynamicLfm2BehaviorError(
            "Cached parent output envelope seal or encoding is invalid."
        )
    expected_identities = {
        "parent_cache_key": _digest(
            expected_parent_cache_key,
            "expected_parent_cache_key",
        ),
        "parent_state_fingerprint": _digest(
            expected_parent_state_fingerprint,
            "expected_parent_state_fingerprint",
        ),
        "tokenizer_fingerprint": _digest(
            expected_tokenizer_fingerprint,
            "expected_tokenizer_fingerprint",
        ),
    }
    for name, expected in expected_identities.items():
        if body.get(name) != expected:
            raise DynamicLfm2BehaviorError(
                f"Cached parent output envelope {name} differs from the live run."
            )
    generation_limit = _positive_int(
        expected_generation_limit,
        "expected_generation_limit",
    )
    expected_generation = {
        **GENERATION_PROTOCOL,
        "max_new_tokens": generation_limit,
    }
    if body.get("generation") != expected_generation:
        raise DynamicLfm2BehaviorError(
            "Cached parent generation protocol differs from the live gate."
        )
    fragment_characters = _positive_int(
        fragment_minimum_characters,
        "fragment_minimum_characters",
    )
    rows = body.get("rows")
    if not isinstance(rows, list) or not rows:
        raise DynamicLfm2BehaviorError("Cached parent output envelope has no rows.")
    cases = []
    contracts = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DynamicLfm2BehaviorError(
                f"Cached parent output row {index} is not an object."
            )
        case_id = row.get("id")
        prompt = row.get("prompt")
        output = row.get("parent_output_text")
        category = row.get("category")
        token_ids = row.get("parent_output_ids")
        expected = row.get("expected_response")
        if (
            not isinstance(case_id, str)
            or not case_id.strip()
            or case_id in seen
            or not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(output, str)
            or not isinstance(category, str)
            or not category.strip()
            or not isinstance(token_ids, list)
            or not token_ids
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in token_ids
            )
            or len(token_ids) > generation_limit
            or (expected is not None and not isinstance(expected, str))
        ):
            raise DynamicLfm2BehaviorError(
                f"Cached parent output row {index} is malformed."
            )
        seen.add(case_id)
        contract = _task_contract(
            case_id=case_id,
            category=category,
            prompt=prompt,
            expected=expected,
            parent_output=output,
        )
        parent_relevant = None if contract is None else contract.relevance_score(output)
        if output.strip() and parent_relevant is not True:
            raise DynamicLfm2BehaviorError(
                f"Semantic validator does not cover nonempty parent case {case_id!r}."
            )
        cases.append(
            WholeOutputCase(
                case_id=case_id,
                prompt=prompt,
                parent_output=output,
                category=category,
                parent_fragment_like=(
                    len(output.strip()) < fragment_characters
                    and parent_relevant is not True
                ),
                parent_metadata=WholeOutputMetadata(
                    generated_token_count=len(token_ids),
                    generation_limit=generation_limit,
                    maximum_repeated_ngram_share=(
                        _maximum_repeated_bigram_share(token_ids)
                    ),
                ),
                split="heldout",
            )
        )
        contracts.append((case_id, contract))
    nonempty_cases = sum(bool(case.parent_output.strip()) for case in cases)
    covered_nonempty_cases = sum(
        bool(case.parent_output.strip()) and contract is not None
        for case, (_case_id, contract) in zip(cases, contracts)
    )
    if nonempty_cases != covered_nonempty_cases:
        raise DynamicLfm2BehaviorError("Semantic-validator coverage is incomplete.")
    return Lfm2CachedWholeOutputBaseline(
        source_path=str(source),
        source_sha256=source_sha256,
        envelope_sha256=str(seal),
        parent_cache_key=expected_identities["parent_cache_key"],
        parent_state_fingerprint=expected_identities["parent_state_fingerprint"],
        tokenizer_fingerprint=expected_identities["tokenizer_fingerprint"],
        cases=tuple(cases),
        contracts=tuple(contracts),
        generation_limit=generation_limit,
    )


class Lfm2WholeOutputBehaviorEvaluator:
    """Generate independent child responses against a cached parent baseline."""

    def __init__(
        self,
        *,
        baseline: Lfm2CachedWholeOutputBaseline,
        tokenizer: Any,
        maximum_prompt_tokens: int = 512,
        fragment_minimum_characters: int = 12,
        maximum_new_empty_responses: int = 0,
        maximum_new_fragment_responses: int = 0,
        minimum_parent_nonempty_retention: float = 1.0,
        minimum_task_score_retention: float = 1.0,
        maximum_parent_task_score_regressions: int = 0,
        maximum_new_generation_limit_saturations: int = 0,
        maximum_repeated_ngram_share_regressions: int = 0,
        maximum_repeated_bigram_share_delta: float = 0.05,
        generator_identity: BehaviorCallableIdentity | None = None,
        task_scorer_identity: BehaviorCallableIdentity | None = None,
    ) -> None:
        if not isinstance(baseline, Lfm2CachedWholeOutputBaseline):
            raise DynamicLfm2BehaviorError(
                "baseline must be Lfm2CachedWholeOutputBaseline."
            )
        self.baseline = baseline
        self.tokenizer = tokenizer
        self.maximum_prompt_tokens = _positive_int(
            maximum_prompt_tokens,
            "maximum_prompt_tokens",
        )
        self.fragment_minimum_characters = _positive_int(
            fragment_minimum_characters,
            "fragment_minimum_characters",
        )
        if (
            isinstance(maximum_parent_task_score_regressions, bool)
            or not isinstance(maximum_parent_task_score_regressions, int)
            or maximum_parent_task_score_regressions < 0
        ):
            raise DynamicLfm2BehaviorError(
                "maximum_parent_task_score_regressions must be a "
                "non-negative integer."
            )
        self.maximum_parent_task_score_regressions = (
            maximum_parent_task_score_regressions
        )
        if (
            isinstance(maximum_repeated_bigram_share_delta, bool)
            or not isinstance(
                maximum_repeated_bigram_share_delta,
                (int, float),
            )
            or not 0.0 <= float(maximum_repeated_bigram_share_delta) <= 1.0
        ):
            raise DynamicLfm2BehaviorError(
                "maximum_repeated_bigram_share_delta must be in [0, 1]."
            )
        self.events: list[dict[str, Any]] = []
        self._generated_rows: list[dict[str, Any]] = []
        self._last_outputs: dict[str, CandidateWholeOutput] = {}
        required_categories = tuple(
            sorted(
                {
                    case.category
                    for case in baseline.cases
                    if baseline.task_score(case, case.parent_output) is True
                }
            )
        )
        if generator_identity is None:
            generator_identity = identify_behavior_callable(
                self._generate,
                implementation_id=(
                    "neural_forge.lfm2.greedy_cached_whole_output_generator"
                ),
                version=1,
            )
        if task_scorer_identity is None:
            task_scorer_identity = identify_behavior_callable(
                baseline.task_score,
                implementation_id=(
                    "neural_forge.lfm2.semantic_task_contract_scorer"
                ),
                version=1,
            )
        self.gate = CachedWholeOutputBehaviorGate(
            baseline.cases,
            generator=self._generate,
            config=WholeOutputGateConfig(
                maximum_new_empty_responses=maximum_new_empty_responses,
                maximum_new_fragment_responses=maximum_new_fragment_responses,
                minimum_parent_nonempty_retention=(
                    minimum_parent_nonempty_retention
                ),
                minimum_task_score_retention=minimum_task_score_retention,
                maximum_parent_task_score_regressions=(
                    maximum_parent_task_score_regressions
                ),
                maximum_new_generation_limit_saturations=(
                    maximum_new_generation_limit_saturations
                ),
                maximum_repeated_ngram_share_regressions=(
                    maximum_repeated_ngram_share_regressions
                ),
                maximum_repeated_ngram_share_delta=float(
                    maximum_repeated_bigram_share_delta
                ),
                task_score_coverage=TaskScoreCoveragePolicy.production(
                    required_splits=("heldout",),
                    required_categories=required_categories,
                    require_split_category_cross_product=bool(
                        required_categories
                    ),
                ),
            ),
            task_scorer=baseline.task_score,
            gate_id="neural_forge.lfm2.cached_whole_output_behavior",
            gate_version=2,
            generator_identity=generator_identity,
            task_scorer_identity=task_scorer_identity,
            baseline_binding={
                "baseline_fingerprint": baseline.fingerprint,
                "source_sha256": baseline.source_sha256,
                "envelope_sha256": baseline.envelope_sha256,
                "parent_cache_key": baseline.parent_cache_key,
                "parent_state_fingerprint": baseline.parent_state_fingerprint,
                "tokenizer_fingerprint": baseline.tokenizer_fingerprint,
            },
        )
        expose_behavior_gate_binding(self, resolve_behavior_gate_binding(self.gate))

    @staticmethod
    def _model_device(model: Any) -> Any:
        devices = {value.device for value in model.parameters()}
        if len(devices) != 1:
            raise DynamicLfm2BehaviorError(
                "Whole-output evaluation requires one materialized model device."
            )
        return next(iter(devices))

    def _generate(
        self,
        model: Any,
        cases: tuple[WholeOutputCase, ...],
    ) -> Mapping[str, CandidateWholeOutput]:
        import torch

        device = self._model_device(model)
        was_training = bool(model.training)
        model.eval()
        outputs = {}
        rows = []
        try:
            with torch.inference_mode():
                for case in cases:
                    encoded = self.tokenizer(
                        case.prompt,
                        return_tensors="pt",
                        add_special_tokens=True,
                    )
                    if not isinstance(encoded, Mapping):
                        raise DynamicLfm2BehaviorError(
                            "Tokenizer output is not a mapping."
                        )
                    input_ids = encoded.get("input_ids")
                    if (
                        not torch.is_tensor(input_ids)
                        or input_ids.ndim != 2
                        or input_ids.shape[0] != 1
                        or input_ids.shape[1] <= 0
                        or input_ids.shape[1] > self.maximum_prompt_tokens
                    ):
                        raise DynamicLfm2BehaviorError(
                            f"Prompt tokenization is invalid for {case.case_id!r}."
                        )
                    input_ids = input_ids.to(device=device, dtype=torch.long)
                    attention_mask = encoded.get("attention_mask")
                    kwargs: dict[str, Any] = {
                        "input_ids": input_ids,
                        "max_new_tokens": self.baseline.generation_limit,
                        "do_sample": False,
                        "use_cache": True,
                    }
                    if attention_mask is not None:
                        if not torch.is_tensor(attention_mask):
                            raise DynamicLfm2BehaviorError(
                                "Tokenizer attention_mask is not a tensor."
                            )
                        kwargs["attention_mask"] = attention_mask.to(
                            device=device,
                            dtype=torch.long,
                        )
                    pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
                    if pad_token_id is not None:
                        kwargs["pad_token_id"] = int(pad_token_id)
                    generated = model.generate(**kwargs)
                    if (
                        not torch.is_tensor(generated)
                        or generated.ndim != 2
                        or generated.shape[0] != 1
                        or generated.shape[1] < input_ids.shape[1]
                        or not torch.equal(
                            generated[:, : int(input_ids.shape[1])],
                            input_ids,
                        )
                    ):
                        raise DynamicLfm2BehaviorError(
                            f"Generation prefix is invalid for {case.case_id!r}."
                        )
                    continuation = generated[:, int(input_ids.shape[1]) :]
                    token_ids = [
                        int(value) for value in continuation[0].detach().cpu().tolist()
                    ]
                    text = self.tokenizer.decode(
                        token_ids,
                        skip_special_tokens=True,
                    )
                    repeated = _maximum_repeated_bigram_share(token_ids)
                    task_relevant = self.baseline.task_score(case, text)
                    outputs[case.case_id] = CandidateWholeOutput(
                        output=text,
                        fragment_like=(
                            len(text.strip()) < self.fragment_minimum_characters
                            and task_relevant is not True
                        ),
                        metadata=WholeOutputMetadata(
                            generated_token_count=len(token_ids),
                            generation_limit=self.baseline.generation_limit,
                            maximum_repeated_ngram_share=repeated,
                        ),
                    )
                    rows.append(
                        {
                            "case_id": case.case_id,
                            "category": case.category,
                            "output": text,
                            "output_token_ids": token_ids,
                            "task_relevant": task_relevant,
                            "task_correct": (
                                self.baseline.correctness_score(case, text)
                            ),
                            "maximum_repeated_bigram_share": repeated,
                        }
                    )
        finally:
            model.train(was_training)
        self._generated_rows = rows
        self._last_outputs = outputs
        return outputs

    def evaluate_model(self, model: Any, *, phase: str) -> FoldDecision:
        decision = self.gate.evaluate_model(model)
        per_case_regressions = []
        parent_correct = 0
        candidate_correct = 0
        correctness_scored = 0
        for case in self.baseline.cases:
            candidate = self._last_outputs[case.case_id]
            parent_relevance = self.baseline.task_score(
                case,
                case.parent_output,
            )
            candidate_relevance = self.baseline.task_score(
                case,
                candidate.output,
            )
            if parent_relevance is True and candidate_relevance is not True:
                per_case_regressions.append(case.case_id)
            parent_correctness = self.baseline.correctness_score(
                case,
                case.parent_output,
            )
            candidate_correctness = self.baseline.correctness_score(
                case,
                candidate.output,
            )
            if parent_correctness is not None and candidate_correctness is not None:
                correctness_scored += 1
                parent_correct += int(parent_correctness)
                candidate_correct += int(candidate_correctness)
        metrics = dict(decision.metrics)
        metrics.update(
            {
                "per_case_task_regression_count": len(per_case_regressions),
                "semantic_validator_coverage_nonempty": 1.0,
                "correctness_scored_case_count": correctness_scored,
                "parent_correct_case_count": parent_correct,
                "candidate_correct_case_count": candidate_correct,
            }
        )
        if (
            len(per_case_regressions)
            > self.maximum_parent_task_score_regressions
        ):
            decision = FoldDecision.reject(
                "Whole-output per-case semantic regression budget exceeded: "
                + ",".join(per_case_regressions),
                metrics,
            )
        elif decision.accepted:
            decision = FoldDecision.accept(decision.reason, metrics)
        else:
            decision = FoldDecision.reject(decision.reason, metrics)
        self.events.append(
            {
                "phase": phase,
                "decision": decision.as_dict(),
                "per_case_task_regression_ids": per_case_regressions,
                "outputs": list(self._generated_rows),
                "baseline_fingerprint": self.baseline.fingerprint,
                "behavior_gate_manifest_fingerprint": (
                    self.behavior_gate_manifest_fingerprint
                ),
                "parent_forward_passes": 0,
            }
        )
        return decision

    def evaluate(self, context: Any) -> FoldDecision:
        return self.evaluate_model(
            context.execution_context.child_model,
            phase="accepted_in_memory_child",
        )

    def evaluate_fresh_reload(self, model: Any) -> FoldDecision:
        return self.evaluate_model(model, phase="freshly_reloaded_child")

    def __call__(self, model: Any) -> FoldDecision:
        return self.evaluate_fresh_reload(model)


def _decision(value: FoldDecision | bool, name: str) -> FoldDecision:
    if isinstance(value, FoldDecision):
        return value
    if isinstance(value, bool):
        return FoldDecision(value, f"{name} returned {value}.")
    raise DynamicLfm2BehaviorError(f"{name} must return FoldDecision or bool.")


class CombinedLfm2FinalEvaluator:
    """Combine structural execution with decoded manual-review evidence.

    In diagnostic-only mode the child is still required to execute both
    evaluators successfully, but their heuristic numeric decisions cannot
    reject publication.  Exceptions and malformed execution remain failures;
    decoded responses and the counterfactual decisions are retained for
    manual intelligence review.
    """

    def __init__(
        self,
        *,
        trajectory_evaluator: Any,
        behavior_evaluator: Lfm2WholeOutputBehaviorEvaluator,
        performance_metrics_diagnostic_only: bool = False,
    ) -> None:
        if not isinstance(performance_metrics_diagnostic_only, bool):
            raise TypeError("performance_metrics_diagnostic_only must be boolean.")
        self.trajectory_evaluator = trajectory_evaluator
        self.behavior_evaluator = behavior_evaluator
        self.performance_metrics_diagnostic_only = (
            performance_metrics_diagnostic_only
        )
        self.events: list[dict[str, Any]] = []
        try:
            behavior_binding = resolve_behavior_gate_binding(behavior_evaluator)
        except DynamicBehaviorGateError:
            behavior_binding = None
        if behavior_binding is not None:
            expose_behavior_gate_binding(self, behavior_binding)

    @staticmethod
    def _combined(
        trajectory: FoldDecision,
        behavior: FoldDecision | None,
        *,
        performance_metrics_diagnostic_only: bool = False,
    ) -> FoldDecision:
        metrics = {
            f"trajectory_{key}": value
            for key, value in dict(trajectory.metrics).items()
        }
        if behavior is not None:
            metrics.update(
                {
                    f"whole_output_{key}": value
                    for key, value in dict(behavior.metrics).items()
                }
            )
        if performance_metrics_diagnostic_only:
            metrics.update(
                {
                    "counterfactual_trajectory_accepted": float(
                        trajectory.accepted
                    ),
                    "counterfactual_whole_output_accepted": float(
                        behavior is not None and behavior.accepted
                    ),
                    "numeric_decision_authority": 0.0,
                    "manual_intelligence_review_required": 1.0,
                }
            )
            return FoldDecision.accept(
                "Trajectory and whole-output metrics were recorded as "
                "counterfactual diagnostics; decoded manual intelligence review "
                "has promotion authority.",
                metrics,
            )
        if not trajectory.accepted:
            return FoldDecision.reject(
                "Trajectory gate rejected before whole-output generation: "
                + trajectory.reason,
                metrics,
            )
        if behavior is None or not behavior.accepted:
            return FoldDecision.reject(
                "Whole-output intelligence gate rejected: "
                + ("" if behavior is None else behavior.reason),
                metrics,
            )
        return FoldDecision.accept(
            "Trajectory and whole-output intelligence gates accepted.",
            metrics,
        )

    def _record(
        self,
        *,
        phase: str,
        trajectory: FoldDecision,
        behavior: FoldDecision | None,
        combined: FoldDecision,
    ) -> None:
        self.events.append(
            {
                "phase": phase,
                "summary": {
                    "passed": combined.accepted,
                    "performance_metrics_diagnostic_only": (
                        self.performance_metrics_diagnostic_only
                    ),
                    "numeric_decision_authority": (
                        not self.performance_metrics_diagnostic_only
                    ),
                    "manual_intelligence_review_required": (
                        self.performance_metrics_diagnostic_only
                    ),
                    "trajectory": trajectory.as_dict(),
                    "whole_output": (None if behavior is None else behavior.as_dict()),
                },
                "parent_forward_count": 0,
            }
        )

    def evaluate(self, context: Any) -> FoldDecision:
        trajectory = _decision(
            self.trajectory_evaluator.evaluate(context),
            "trajectory evaluator",
        )
        behavior = None
        if trajectory.accepted or self.performance_metrics_diagnostic_only:
            behavior = self.behavior_evaluator.evaluate_model(
                context.execution_context.child_model,
                phase="accepted_in_memory_child",
            )
        combined = self._combined(
            trajectory,
            behavior,
            performance_metrics_diagnostic_only=(
                self.performance_metrics_diagnostic_only
            ),
        )
        self._record(
            phase="accepted_in_memory_child",
            trajectory=trajectory,
            behavior=behavior,
            combined=combined,
        )
        return combined

    def evaluate_fresh_reload(self, model: Any) -> FoldDecision:
        trajectory = _decision(
            self.trajectory_evaluator.evaluate_fresh_reload(model),
            "fresh-reload trajectory evaluator",
        )
        behavior = None
        if trajectory.accepted or self.performance_metrics_diagnostic_only:
            behavior = self.behavior_evaluator.evaluate_model(
                model,
                phase="freshly_reloaded_child",
            )
        combined = self._combined(
            trajectory,
            behavior,
            performance_metrics_diagnostic_only=(
                self.performance_metrics_diagnostic_only
            ),
        )
        self._record(
            phase="freshly_reloaded_child",
            trajectory=trajectory,
            behavior=behavior,
            combined=combined,
        )
        return combined

    def __call__(self, model: Any) -> FoldDecision:
        return self.evaluate_fresh_reload(model)


__all__ = [
    "CombinedLfm2FinalEvaluator",
    "DynamicLfm2BehaviorError",
    "ENVELOPE_SCHEMA",
    "GENERATION_PROTOCOL",
    "Lfm2CachedWholeOutputBaseline",
    "Lfm2TaskContract",
    "Lfm2WholeOutputBehaviorEvaluator",
    "SCHEMA",
    "load_lfm2_cached_whole_output_baseline",
    "seal_lfm2_cached_whole_output_envelope",
]
