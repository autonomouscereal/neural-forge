#!/usr/bin/env python3
"""Offline, identity-bound intelligence evaluation for the 28-case LFM2 suite.

This runner is deliberately post-hoc.  It never loads or executes a parent or
candidate model and it never trains.  It authenticates the existing parent
response envelope, a previously recorded candidate whole-output event, and the
local tokenizer artifacts used to identify EOS.  It then adapts that evidence
to :mod:`neural_forge.hf_lfm2_intelligence_gate`.

The older whole-output gate's topical-relevance score remains a diagnostic.
Promotion is based on exact arithmetic, safe AST-interpreted code behavior,
instruction constraints, parent-demonstrated capability retention, termination,
degeneration, and packet-bound human ratings for genuinely open-ended cases.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.hf_lfm2_intelligence_gate import (
    CandidateOutput,
    CapabilitySpec,
    GenerationEvidence,
    IntelligenceGatePolicy,
    PureFunctionExample,
    ReliableValidatorSpec,
    build_manual_review_packet,
    evaluate_cached_intelligence,
    load_bound_manual_ratings,
    load_cached_parent_rows,
    seal_cached_parent_rows,
)


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

RUNNER_SCHEMA = "neural-forge-lfm2-posthoc-intelligence-evaluation-v1"
WHOLE_OUTPUT_ENVELOPE_SCHEMA = "neural-forge-lfm2-whole-output-envelope-v1"
EXPECTED_CASE_COUNT = 28
FRAGMENT_MINIMUM_CHARACTERS = 12
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CODE_FENCE = re.compile(
    r"```(?:python|py)?\s*(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
)
_MANUAL_CASE_IDS = frozenset(
    {
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
_OPEN_ENDED_MANUAL_IDS = frozenset(
    {
        "manual_chat_00",
        "manual_fact_00",
        "manual_code_00",
        "manual_ai_00",
        "manual_summary_00",
        "manual_creative_00",
        "manual_fold_00",
    }
)
_CORE_REQUIRED_COMPONENTS = (
    "task_correctness",
    "parent_capability",
    "nonempty_retention",
    "fragment_retention",
    "eos_generation_limit",
    "repetition_degeneration",
    "code_syntax",
    "arithmetic_exactness",
)
_PARENT_CAPABILITY_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
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
    "manual_creative_00": (("robot",),),
}


class Lfm2PosthocEvaluationError(RuntimeError):
    """Input evidence cannot support a conservative post-hoc evaluation."""


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise Lfm2PosthocEvaluationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2PosthocEvaluationError(f"{name} must be a positive integer.")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_input_file(value: str | os.PathLike[str], name: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2PosthocEvaluationError(f"{name} does not exist.") from exc
    if not path.is_file() or path.is_symlink():
        raise Lfm2PosthocEvaluationError(f"{name} is missing or unsafe: {path}.")
    return path


def _read_json_file(
    value: str | os.PathLike[str],
    *,
    name: str,
    expected_sha256: str,
) -> tuple[Path, bytes, Any]:
    path = _safe_input_file(value, name)
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != _digest(expected_sha256, f"expected {name} SHA-256"):
        raise Lfm2PosthocEvaluationError(
            f"{name} fingerprint differs from the expected SHA-256."
        )
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2PosthocEvaluationError(f"{name} is not valid UTF-8 JSON.") from exc
    return path, raw, document


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    target = path.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(value)) + b"\n"
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


def fingerprint_bound_local_tokenizer(
    tokenizer_directory: str | os.PathLike[str],
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    """Hash the same local tokenizer package used by the folding runner."""

    try:
        directory = Path(tokenizer_directory).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2PosthocEvaluationError(
            "Local tokenizer directory does not exist."
        ) from exc
    if not directory.is_dir() or directory.is_symlink():
        raise Lfm2PosthocEvaluationError(
            f"Local tokenizer directory is missing or unsafe: {directory}."
        )
    files = []
    for name in _TOKENIZER_ARTIFACTS:
        path = directory / name
        if path.is_symlink():
            raise Lfm2PosthocEvaluationError(
                f"Tokenizer artifact cannot be a symlink: {path}."
            )
        if path.is_file():
            files.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    if not files or files[0]["path"] != "tokenizer.json":
        raise Lfm2PosthocEvaluationError(
            "The bound local tokenizer package lacks tokenizer.json."
        )
    identity = canonical_sha256(
        {
            "schema": "neural-forge-local-tokenizer-artifacts-v1",
            "files": files,
        }
    )
    return identity, tuple(files)


def _read_bound_tokenizer_json(directory: Path, name: str) -> Mapping[str, Any]:
    path = directory / name
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Lfm2PosthocEvaluationError(
            f"Bound tokenizer artifact {name!r} is invalid."
        ) from exc
    if not isinstance(value, Mapping):
        raise Lfm2PosthocEvaluationError(
            f"Bound tokenizer artifact {name!r} must be an object."
        )
    return value


def _token_content(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping) and isinstance(value.get("content"), str):
        return (str(value["content"]),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = []
        for item in value:
            result.extend(_token_content(item))
        return tuple(result)
    return ()


def load_bound_special_token_ids(
    tokenizer_directory: str | os.PathLike[str],
    *,
    expected_tokenizer_fingerprint: str,
) -> dict[str, Any]:
    """Resolve special IDs solely from artifacts covered by the tokenizer hash."""

    directory = Path(tokenizer_directory).expanduser().resolve(strict=True)
    observed, files = fingerprint_bound_local_tokenizer(directory)
    expected = _digest(
        expected_tokenizer_fingerprint,
        "expected_tokenizer_fingerprint",
    )
    if observed != expected:
        raise Lfm2PosthocEvaluationError(
            "Local tokenizer fingerprint differs from the sealed identity."
        )
    tokenizer = _read_bound_tokenizer_json(directory, "tokenizer.json")
    tokenizer_config = _read_bound_tokenizer_json(
        directory, "tokenizer_config.json"
    )
    special_map = _read_bound_tokenizer_json(
        directory, "special_tokens_map.json"
    )
    content_to_id: dict[str, int] = {}
    added = tokenizer.get("added_tokens", ())
    if isinstance(added, Sequence) and not isinstance(added, (str, bytes)):
        for item in added:
            if (
                isinstance(item, Mapping)
                and isinstance(item.get("content"), str)
                and isinstance(item.get("id"), int)
                and not isinstance(item.get("id"), bool)
                and int(item["id"]) >= 0
            ):
                content_to_id[str(item["content"])] = int(item["id"])
    model = tokenizer.get("model")
    vocab = model.get("vocab") if isinstance(model, Mapping) else None
    if isinstance(vocab, Mapping):
        for content, token_id in vocab.items():
            if (
                isinstance(content, str)
                and isinstance(token_id, int)
                and not isinstance(token_id, bool)
                and token_id >= 0
            ):
                content_to_id.setdefault(content, token_id)
    elif isinstance(vocab, Sequence) and not isinstance(vocab, (str, bytes)):
        for token_id, item in enumerate(vocab):
            if (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                and item
                and isinstance(item[0], str)
            ):
                content_to_id.setdefault(str(item[0]), token_id)

    resolved: dict[str, list[int]] = {}
    for stem in ("eos", "bos", "pad", "unk", "sep", "cls", "mask"):
        direct_ids: set[int] = set()
        contents: list[str] = []
        for source in (tokenizer_config, special_map):
            direct = source.get(f"{stem}_token_id")
            direct_values = (
                direct
                if isinstance(direct, Sequence)
                and not isinstance(direct, (str, bytes))
                else (direct,)
            )
            for token_id in direct_values:
                if (
                    isinstance(token_id, int)
                    and not isinstance(token_id, bool)
                    and token_id >= 0
                ):
                    direct_ids.add(token_id)
            contents.extend(_token_content(source.get(f"{stem}_token")))
        content_ids = {
            content_to_id[content]
            for content in contents
            if content in content_to_id
        }
        unresolved = sorted(set(contents).difference(content_to_id))
        if unresolved:
            raise Lfm2PosthocEvaluationError(
                f"Bound {stem} token content is absent from tokenizer.json: "
                + ",".join(unresolved)
            )
        if direct_ids and content_ids and direct_ids != content_ids:
            raise Lfm2PosthocEvaluationError(
                f"Bound {stem} token ID and content disagree."
            )
        values = sorted(direct_ids or content_ids)
        if values:
            resolved[stem] = values
    if not resolved.get("eos"):
        raise Lfm2PosthocEvaluationError(
            "EOS IDs cannot be derived from the bound local tokenizer artifacts."
        )
    core = {
        "schema": "neural-forge-bound-local-special-token-ids-v1",
        "tokenizer_fingerprint": observed,
        "artifact_files": list(files),
        "special_token_ids": resolved,
        "network_access": False,
        "auto_tokenizer_used": False,
    }
    return {**core, "special_token_identity_sha256": canonical_sha256(core)}


def load_sealed_parent_envelope(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_parent_cache_key: str,
    expected_parent_state_fingerprint: str,
    expected_tokenizer_fingerprint: str,
    expected_case_count: int = EXPECTED_CASE_COUNT,
) -> tuple[Mapping[str, Any], str]:
    """Authenticate the original production whole-output envelope."""

    _, raw, document = _read_json_file(
        path,
        name="parent envelope",
        expected_sha256=expected_sha256,
    )
    if not isinstance(document, Mapping):
        raise Lfm2PosthocEvaluationError("Parent envelope must be an object.")
    if raw not in (canonical_json_bytes(document), canonical_json_bytes(document) + b"\n"):
        raise Lfm2PosthocEvaluationError(
            "Parent envelope encoding is not canonical."
        )
    core = dict(document)
    seal = core.pop("envelope_sha256", None)
    identities = {
        "parent_cache_key": _digest(
            expected_parent_cache_key, "expected_parent_cache_key"
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
    if (
        core.get("schema") != WHOLE_OUTPUT_ENVELOPE_SCHEMA
        or seal != canonical_sha256(core)
        or core.get("parent_outputs_generated_once") is not True
        or core.get("parent_forward_passes_after_seal") != 0
        or any(core.get(name) != value for name, value in identities.items())
    ):
        raise Lfm2PosthocEvaluationError(
            "Parent envelope seal, cache identity, tokenizer identity, or "
            "zero-reexecution declaration is invalid."
        )
    generation = core.get("generation")
    if (
        not isinstance(generation, Mapping)
        or generation.get("do_sample") is not False
        or generation.get("use_cache") is not True
    ):
        raise Lfm2PosthocEvaluationError(
            "Parent envelope generation protocol is not deterministic."
        )
    _positive_int(generation.get("max_new_tokens"), "generation max_new_tokens")
    rows = core.get("rows")
    count = _positive_int(expected_case_count, "expected_case_count")
    if not isinstance(rows, list) or len(rows) != count:
        raise Lfm2PosthocEvaluationError(
            f"Parent envelope must contain exactly {count} rows."
        )
    ids = [row.get("id") if isinstance(row, Mapping) else None for row in rows]
    if (
        any(not isinstance(case_id, str) or not case_id for case_id in ids)
        or len(set(ids)) != len(ids)
    ):
        raise Lfm2PosthocEvaluationError(
            "Parent envelope case IDs are missing or duplicated."
        )
    recovery_math = {item for item in ids if item.startswith("recovery_math_")}
    recovery_code = {item for item in ids if item.startswith("recovery_code_")}
    manual = set(ids).intersection(_MANUAL_CASE_IDS)
    if (
        len(recovery_math) != 8
        or len(recovery_code) != 8
        or manual != _MANUAL_CASE_IDS
        or len(recovery_math | recovery_code | manual) != count
    ):
        raise Lfm2PosthocEvaluationError(
            "Parent envelope differs from the sealed 8-math/8-code/12-manual "
            "evaluation contract."
        )
    return document, str(seal)


def _valid_output_rows(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(
            isinstance(item, Mapping)
            and {"case_id", "output", "output_token_ids"}.issubset(item)
            for item in value
        )
    )


def _find_candidate_events(
    value: Any,
    *,
    path: str = "$",
    inherited_phase: str | None = None,
    depth: int = 0,
) -> list[tuple[str, str | None, list[Mapping[str, Any]]]]:
    if depth > 12:
        raise Lfm2PosthocEvaluationError(
            "Candidate report is nested too deeply."
        )
    if _valid_output_rows(value):
        return [(path, inherited_phase, list(value))]
    found: list[tuple[str, str | None, list[Mapping[str, Any]]]] = []
    if isinstance(value, Mapping):
        phase_value = value.get("phase", inherited_phase)
        phase = phase_value if isinstance(phase_value, str) else inherited_phase
        outputs = value.get("outputs")
        if _valid_output_rows(outputs):
            found.append((f"{path}.outputs", phase, list(outputs)))
        for key, item in value.items():
            if key == "outputs" and _valid_output_rows(item):
                continue
            found.extend(
                _find_candidate_events(
                    item,
                    path=f"{path}.{key}",
                    inherited_phase=phase,
                    depth=depth + 1,
                )
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(
                _find_candidate_events(
                    item,
                    path=f"{path}[{index}]",
                    inherited_phase=inherited_phase,
                    depth=depth + 1,
                )
            )
    return found


def extract_candidate_output_rows(
    document: Any,
    *,
    candidate_phase: str | None = None,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Select one unambiguous whole-output event from a report or event."""

    events = _find_candidate_events(document)
    if candidate_phase is not None:
        events = [item for item in events if item[1] == candidate_phase]
    if len(events) != 1:
        phases = sorted({str(item[1]) for item in events})
        raise Lfm2PosthocEvaluationError(
            "Candidate report must resolve to exactly one whole-output event"
            + (
                f" for phase {candidate_phase!r}"
                if candidate_phase is not None
                else ""
            )
            + f"; found {len(events)} with phases {phases}."
        )
    path, phase, rows = events[0]
    return rows, {
        "selected_event_path": path,
        "selected_phase": phase,
        "selected_event_outputs_sha256": canonical_sha256(rows),
    }


def _generation_evidence(
    token_ids: Any,
    *,
    generation_limit: int,
    eos_token_ids: frozenset[int],
    fragment_like: bool,
    name: str,
) -> GenerationEvidence:
    if (
        not isinstance(token_ids, list)
        or len(token_ids) > generation_limit
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in token_ids
        )
    ):
        raise Lfm2PosthocEvaluationError(
            f"{name} token IDs are malformed or exceed the generation limit."
        )
    eos_indices = [
        index for index, token_id in enumerate(token_ids) if token_id in eos_token_ids
    ]
    if eos_indices and eos_indices != [len(token_ids) - 1]:
        raise Lfm2PosthocEvaluationError(
            f"{name} contains EOS before the final generated token."
        )
    return GenerationEvidence(
        generated_token_count=len(token_ids),
        generation_limit=generation_limit,
        eos_observed=bool(eos_indices),
        fragment_like=fragment_like,
        token_ids=tuple(token_ids),
    )


def _extract_python_source(output: str) -> str:
    match = _CODE_FENCE.search(output)
    if match is not None:
        return match.group(1).strip()
    stripped = output.strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].rstrip()
    if stripped.casefold().startswith("python\n"):
        stripped = stripped.split("\n", 1)[1]
    return stripped


def _trusted_expected_expression(
    node: ast.AST,
    environment: Mapping[str, int],
) -> int | float | bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return node.value
    if isinstance(node, ast.Name) and node.id in environment:
        return environment[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub)
    ):
        value = _trusted_expected_expression(node.operand, environment)
        if isinstance(value, bool):
            raise Lfm2PosthocEvaluationError("Expected code uses unsafe arithmetic.")
        return +value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult)
    ):
        left = _trusted_expected_expression(node.left, environment)
        right = _trusted_expected_expression(node.right, environment)
        if isinstance(left, bool) or isinstance(right, bool):
            raise Lfm2PosthocEvaluationError("Expected code uses unsafe operands.")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        left = _trusted_expected_expression(node.left, environment)
        right = _trusted_expected_expression(node.comparators[0], environment)
        operator = node.ops[0]
        if isinstance(operator, ast.GtE):
            return left >= right
        if isinstance(operator, ast.Gt):
            return left > right
        if isinstance(operator, ast.LtE):
            return left <= right
        if isinstance(operator, ast.Lt):
            return left < right
        if isinstance(operator, ast.Eq):
            return left == right
    if isinstance(node, ast.IfExp):
        condition = _trusted_expected_expression(node.test, environment)
        return _trusted_expected_expression(
            node.body if condition else node.orelse,
            environment,
        )
    raise Lfm2PosthocEvaluationError(
        "Expected recovery code is outside the safe two-argument expression subset."
    )


def _derive_recovery_code_validator(
    row: Mapping[str, Any],
) -> ReliableValidatorSpec:
    expected = row.get("expected_response")
    if not isinstance(expected, str) or not expected.strip():
        raise Lfm2PosthocEvaluationError(
            f"Recovery-code case {row.get('id')!r} lacks expected_response."
        )
    source = _extract_python_source(expected)
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise Lfm2PosthocEvaluationError(
            f"Expected recovery code for {row.get('id')!r} is invalid."
        ) from exc
    functions = [item for item in tree.body if isinstance(item, ast.FunctionDef)]
    if len(functions) != 1 or len(tree.body) != 1:
        raise Lfm2PosthocEvaluationError(
            "Expected recovery code must contain exactly one function."
        )
    function = functions[0]
    arguments = tuple(item.arg for item in function.args.args)
    if (
        len(arguments) != 2
        or function.args.posonlyargs
        or function.args.kwonlyargs
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.defaults
        or function.args.kw_defaults
        or len(function.body) != 1
        or not isinstance(function.body[0], ast.Return)
    ):
        raise Lfm2PosthocEvaluationError(
            "Expected recovery code is not a pure two-argument return function."
        )
    raw_examples = ((2, 3), (-4, 7), (11, -2))
    examples = []
    for values in raw_examples:
        environment = dict(zip(arguments, values))
        result = _trusted_expected_expression(function.body[0].value, environment)
        if not isinstance(result, (int, float, bool)) or not math.isfinite(
            float(result)
        ):
            raise Lfm2PosthocEvaluationError(
                "Expected recovery-code result is not bounded and numeric."
            )
        examples.append(PureFunctionExample(arguments=values, expected=result))
    return ReliableValidatorSpec(
        kind="python_pure_function",
        function_name=function.name,
        examples=tuple(examples),
    )


def _derive_validator(row: Mapping[str, Any]) -> ReliableValidatorSpec | None:
    case_id = str(row.get("id", ""))
    if case_id.startswith("recovery_math_"):
        expected = row.get("expected_response")
        if not isinstance(expected, str) or not expected.strip():
            raise Lfm2PosthocEvaluationError(
                f"Recovery-math case {case_id!r} lacks expected_response."
            )
        return ReliableValidatorSpec(
            kind="arithmetic_exact",
            expected_numeric=expected.strip(),
        )
    if case_id.startswith("recovery_code_"):
        return _derive_recovery_code_validator(row)
    if case_id == "manual_math_00":
        return ReliableValidatorSpec(
            kind="arithmetic_exact",
            expected_numeric="378",
        )
    if case_id == "manual_instruction_00":
        return ReliableValidatorSpec(
            kind="instruction_constraints",
            exact_word="blue",
            maximum_words=1,
        )
    if case_id == "manual_reason_00":
        return ReliableValidatorSpec(
            kind="instruction_constraints",
            required_groups=(
                ("no", "cannot"),
                ("bloop", "bloops"),
                ("razz", "razzies"),
                ("green",),
            ),
            minimum_words=5,
            maximum_words=60,
        )
    if case_id == "manual_safety_00":
        return ReliableValidatorSpec(
            kind="instruction_constraints",
            required_groups=(
                ("leave", "evacuate", "outside"),
                ("call", "emergency", "gas company", "fire department", "911"),
            ),
            minimum_words=5,
            maximum_words=60,
        )
    if case_id == "manual_summary_00":
        return ReliableValidatorSpec(
            kind="instruction_constraints",
            required_groups=(
                ("activation", "activations"),
                ("error", "errors", "mistake", "mistakes"),
                ("compound", "accumulate", "grow"),
                ("attention",),
                ("residual",),
            ),
            minimum_words=7,
            maximum_words=45,
        )
    if case_id == "manual_compare_00":
        return ReliableValidatorSpec(
            kind="instruction_constraints",
            required_groups=(
                ("ram",),
                ("disk", "storage"),
                ("temporary", "volatile"),
                ("long-term", "permanent", "persistent", "durable"),
            ),
            minimum_words=6,
            maximum_words=60,
        )
    return None


def _code_function_name(validator: ReliableValidatorSpec | None) -> str | None:
    if validator is not None and validator.kind == "python_pure_function":
        return validator.function_name
    return None


def _candidate_capability(
    row: Mapping[str, Any],
    *,
    validator: ReliableValidatorSpec | None,
    parent_output: str,
) -> CapabilitySpec | None:
    case_id = str(row["id"])
    groups: tuple[tuple[str, ...], ...] | None = None
    minimum_words = 1
    if validator is not None and validator.kind == "arithmetic_exact":
        groups = ((str(validator.expected_numeric),),)
    elif (function_name := _code_function_name(validator)) is not None:
        groups = ((f"def {function_name}",), ("return",))
        minimum_words = 2
    elif validator is not None and validator.kind == "instruction_constraints":
        if validator.exact_word is not None:
            groups = ((validator.exact_word,),)
        else:
            groups = validator.required_groups
        minimum_words = validator.minimum_words
    elif case_id == "manual_code_00":
        groups = (
            ("def is_palindrome",),
            ("return",),
            ("[::-1]",),
            ("lower", "casefold"),
        )
        minimum_words = 3
    elif case_id in _PARENT_CAPABILITY_GROUPS:
        groups = _PARENT_CAPABILITY_GROUPS[case_id]
        minimum_words = 4
    if not groups:
        return None
    capability = CapabilitySpec(
        required_concept_groups=groups,
        minimum_words=minimum_words,
    )
    return capability if capability.passes(parent_output) else None


def _prompt_fence_normalization(prompt: str, output: str) -> tuple[str, bool]:
    stripped_prompt = prompt.rstrip().casefold()
    stripped_output = output.lstrip()
    if (
        stripped_prompt.endswith("```python")
        and not stripped_output.startswith("```")
        and "```" in stripped_output
    ):
        return "```python\n" + output, True
    return output, False


def _fragment_like(
    output: str,
    validator: ReliableValidatorSpec | None,
) -> bool:
    stripped = output.strip()
    if not stripped:
        return True
    if validator is not None and (
        validator.kind == "arithmetic_exact" or validator.exact_word is not None
    ):
        return False
    return len(stripped) < FRAGMENT_MINIMUM_CHARACTERS


def build_gate_evidence(
    envelope: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    eos_token_ids: Sequence[int],
) -> tuple[list[Mapping[str, Any]], tuple[CandidateOutput, ...], dict[str, Any]]:
    """Derive typed gate rows without calling either model."""

    rows = envelope.get("rows")
    generation = envelope.get("generation")
    if not isinstance(rows, list) or not isinstance(generation, Mapping):
        raise Lfm2PosthocEvaluationError("Authenticated envelope lost its rows.")
    generation_limit = _positive_int(
        generation.get("max_new_tokens"),
        "generation max_new_tokens",
    )
    eos_ids = frozenset(int(value) for value in eos_token_ids)
    if not eos_ids:
        raise Lfm2PosthocEvaluationError("At least one bound EOS ID is required.")
    candidate_map: dict[str, Mapping[str, Any]] = {}
    for value in candidate_rows:
        case_id = value.get("case_id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in candidate_map
        ):
            raise Lfm2PosthocEvaluationError(
                "Candidate case IDs are missing or duplicated."
            )
        candidate_map[case_id] = value
    parent_ids = tuple(str(row["id"]) for row in rows)
    if set(candidate_map) != set(parent_ids):
        missing = sorted(set(parent_ids).difference(candidate_map))
        extra = sorted(set(candidate_map).difference(parent_ids))
        raise Lfm2PosthocEvaluationError(
            f"Candidate case coverage differs: missing={missing}, extra={extra}."
        )

    baseline_rows = []
    candidates = []
    validator_manifest = []
    normalizations = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise Lfm2PosthocEvaluationError("Parent envelope row is not an object.")
        case_id = str(row["id"])
        prompt = row.get("prompt")
        category = row.get("category")
        parent_raw = row.get("parent_output_text")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(category, str)
            or not category
            or not isinstance(parent_raw, str)
        ):
            raise Lfm2PosthocEvaluationError(
                f"Parent row {case_id!r} is malformed."
            )
        validator = _derive_validator(row)
        parent_output, parent_normalized = (
            _prompt_fence_normalization(prompt, parent_raw)
            if case_id.startswith("recovery_code_")
            else (parent_raw, False)
        )
        candidate_row = candidate_map[case_id]
        candidate_raw = candidate_row.get("output")
        if not isinstance(candidate_raw, str):
            raise Lfm2PosthocEvaluationError(
                f"Candidate output for {case_id!r} must be a string."
            )
        candidate_output, candidate_normalized = (
            _prompt_fence_normalization(prompt, candidate_raw)
            if case_id.startswith("recovery_code_")
            else (candidate_raw, False)
        )
        if parent_normalized or candidate_normalized:
            normalizations.append(
                {
                    "case_id": case_id,
                    "parent_prompt_fence_reconstituted": parent_normalized,
                    "candidate_prompt_fence_reconstituted": candidate_normalized,
                    "code_content_changed": False,
                    "rule": (
                        "The prompt supplied the opening Python fence; a synthetic "
                        "opening delimiter lets the safe AST parser pair the "
                        "verbatim continuation's closing delimiter."
                    ),
                }
            )
        manual_required = case_id in _OPEN_ENDED_MANUAL_IDS or validator is None
        capability = _candidate_capability(
            row,
            validator=validator,
            parent_output=parent_output,
        )
        parent_generation = _generation_evidence(
            row.get("parent_output_ids"),
            generation_limit=generation_limit,
            eos_token_ids=eos_ids,
            fragment_like=_fragment_like(parent_output, validator),
            name=f"parent {case_id}",
        )
        candidate_generation = _generation_evidence(
            candidate_row.get("output_token_ids"),
            generation_limit=generation_limit,
            eos_token_ids=eos_ids,
            fragment_like=_fragment_like(candidate_output, validator),
            name=f"candidate {case_id}",
        )
        topical = candidate_row.get(
            "task_relevant",
            candidate_row.get("topical_relevance"),
        )
        if topical is not None and not isinstance(topical, bool):
            raise Lfm2PosthocEvaluationError(
                f"Candidate topical relevance for {case_id!r} is not boolean/null."
            )
        baseline_rows.append(
            {
                "case_id": case_id,
                "prompt": prompt,
                "category": category,
                "split": (
                    "heldout_manual"
                    if case_id.startswith("manual_")
                    else "heldout"
                ),
                "parent_output": parent_output,
                "generation": parent_generation.as_dict(),
                "validator": (
                    None if validator is None else validator.as_dict()
                ),
                "capability": (
                    None if capability is None else capability.as_dict()
                ),
                "manual_review_required": manual_required,
            }
        )
        candidates.append(
            CandidateOutput(
                case_id=case_id,
                output=candidate_output,
                generation=candidate_generation,
                topical_relevance=topical,
            )
        )
        validator_manifest.append(
            {
                "case_id": case_id,
                "validator": None if validator is None else validator.as_dict(),
                "parent_capability": (
                    None if capability is None else capability.as_dict()
                ),
                "manual_review_required": manual_required,
                "parent_output_empty": not bool(parent_raw.strip()),
            }
        )
    derivation = {
        "schema": "neural-forge-lfm2-derived-validator-manifest-v1",
        "case_count": len(validator_manifest),
        "cases": validator_manifest,
        "prompt_fence_normalizations": normalizations,
        "normalization_changes_code_content": False,
        "exact_token_matching_used_for_intelligence": False,
        "topical_relevance_used_for_promotion": False,
    }
    derivation["manifest_sha256"] = canonical_sha256(derivation)
    return baseline_rows, tuple(candidates), derivation


def _safe_palindrome_expression(node: ast.AST, environment: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, bool, int)
    ):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in environment:
            raise Lfm2PosthocEvaluationError("Unknown name in palindrome function.")
        return environment[node.id]
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        left = _safe_palindrome_expression(node.left, environment)
        right = _safe_palindrome_expression(node.comparators[0], environment)
        if isinstance(node.ops[0], ast.Eq):
            return left == right
        if isinstance(node.ops[0], ast.NotEq):
            return left != right
        raise Lfm2PosthocEvaluationError(
            "Unsafe comparison in palindrome function."
        )
    if isinstance(node, ast.Subscript):
        value = _safe_palindrome_expression(node.value, environment)
        if not isinstance(value, (str, list, tuple)):
            raise Lfm2PosthocEvaluationError("Unsafe palindrome subscript.")
        if isinstance(node.slice, ast.Slice):
            lower = (
                None
                if node.slice.lower is None
                else _safe_palindrome_expression(node.slice.lower, environment)
            )
            upper = (
                None
                if node.slice.upper is None
                else _safe_palindrome_expression(node.slice.upper, environment)
            )
            step = (
                None
                if node.slice.step is None
                else _safe_palindrome_expression(node.slice.step, environment)
            )
            return value[slice(lower, upper, step)]
        index = _safe_palindrome_expression(node.slice, environment)
        if not isinstance(index, int) or isinstance(index, bool):
            raise Lfm2PosthocEvaluationError("Unsafe palindrome index.")
        return value[index]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _safe_palindrome_expression(node.operand, environment)
        if not isinstance(value, int) or isinstance(value, bool):
            raise Lfm2PosthocEvaluationError("Unsafe palindrome slice step.")
        return -value
    if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
        generators = node.generators
        if len(generators) != 1:
            raise Lfm2PosthocEvaluationError(
                "Palindrome comprehension must have one generator."
            )
        generator = generators[0]
        if (
            generator.is_async
            or not isinstance(generator.target, ast.Name)
            or generator.ifs is None
        ):
            raise Lfm2PosthocEvaluationError("Unsafe palindrome comprehension.")
        values = _safe_palindrome_expression(generator.iter, environment)
        if not isinstance(values, (str, list, tuple)) or len(values) > 4096:
            raise Lfm2PosthocEvaluationError("Palindrome input is not bounded.")
        result = []
        for value in values:
            nested = dict(environment)
            nested[generator.target.id] = value
            if all(
                bool(_safe_palindrome_expression(condition, nested))
                for condition in generator.ifs
            ):
                result.append(_safe_palindrome_expression(node.elt, nested))
        return result
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.keywords:
            raise Lfm2PosthocEvaluationError(
                "Palindrome methods cannot use keyword arguments."
            )
        receiver = _safe_palindrome_expression(node.func.value, environment)
        arguments = [
            _safe_palindrome_expression(item, environment) for item in node.args
        ]
        method = node.func.attr
        if isinstance(receiver, str) and method in {"lower", "casefold"} and not arguments:
            return receiver.lower() if method == "lower" else receiver.casefold()
        if isinstance(receiver, str) and method == "isalnum" and not arguments:
            return receiver.isalnum()
        if (
            isinstance(receiver, str)
            and method == "replace"
            and len(arguments) == 2
            and all(isinstance(item, str) for item in arguments)
        ):
            return receiver.replace(arguments[0], arguments[1])
        if (
            isinstance(receiver, str)
            and method == "join"
            and len(arguments) == 1
            and isinstance(arguments[0], (list, tuple))
            and all(isinstance(item, str) for item in arguments[0])
        ):
            return receiver.join(arguments[0])
        raise Lfm2PosthocEvaluationError(
            f"Unsafe palindrome method {method!r}."
        )
    raise Lfm2PosthocEvaluationError(
        f"Unsafe palindrome construct {type(node).__name__!r}."
    )


def _safe_palindrome_result(output: str) -> tuple[bool, dict[str, Any]]:
    source = _extract_python_source(output)
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return False, {"syntax_passed": False, "failure_reason": "syntax_error"}
    if (
        len(tree.body) != 1
        or not isinstance(tree.body[0], ast.FunctionDef)
        or tree.body[0].name != "is_palindrome"
    ):
        return False, {
            "syntax_passed": True,
            "failure_reason": "named_function_missing_or_ambiguous",
        }
    function = tree.body[0]
    if (
        function.decorator_list
        or function.args.posonlyargs
        or len(function.args.args) != 1
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.kwonlyargs
        or function.args.defaults
        or function.args.kw_defaults
    ):
        return False, {
            "syntax_passed": True,
            "failure_reason": "unsafe_signature",
        }
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body or not isinstance(body[-1], ast.Return):
        return False, {
            "syntax_passed": True,
            "failure_reason": "return_missing",
        }
    examples = (
        ("Never odd or even", True),
        ("Step on no pets", True),
        ("Race a car", False),
        ("OpenAI", False),
    )
    for index, (argument, expected) in enumerate(examples):
        environment: dict[str, Any] = {function.args.args[0].arg: argument}
        try:
            for statement in body[:-1]:
                if (
                    not isinstance(statement, ast.Assign)
                    or len(statement.targets) != 1
                    or not isinstance(statement.targets[0], ast.Name)
                ):
                    raise Lfm2PosthocEvaluationError(
                        "Unsafe palindrome statement."
                    )
                environment[statement.targets[0].id] = _safe_palindrome_expression(
                    statement.value, environment
                )
            actual = _safe_palindrome_expression(body[-1].value, environment)
        except (
            Lfm2PosthocEvaluationError,
            IndexError,
            TypeError,
            ValueError,
        ):
            return False, {
                "syntax_passed": True,
                "behavior_passed": False,
                "failure_reason": f"unsafe_or_failed_example:{index}",
            }
        if actual is not expected:
            return False, {
                "syntax_passed": True,
                "behavior_passed": False,
                "failure_reason": f"wrong_example:{index}",
            }
    return True, {
        "syntax_passed": True,
        "behavior_passed": True,
        "failure_reason": None,
        "safe_ast_interpreter": True,
        "python_exec_used": False,
        "examples": [
            {"input": argument, "expected": expected}
            for argument, expected in examples
        ],
    }


def _combined_reliable_correctness(
    gate_report: Mapping[str, Any],
    *,
    parent_manual_code: str,
    candidate_manual_code: str,
    minimum_retention: float,
    minimum_absolute_accuracy: float,
) -> dict[str, Any]:
    components = {
        str(item["name"]): item for item in gate_report.get("components", ())
    }
    task = components.get("task_correctness")
    if not isinstance(task, Mapping):
        raise Lfm2PosthocEvaluationError(
            "Core intelligence report lacks task correctness."
        )
    details = task.get("details")
    per_case = details.get("per_case") if isinstance(details, Mapping) else None
    if not isinstance(per_case, Mapping):
        raise Lfm2PosthocEvaluationError(
            "Core task-correctness report lacks per-case evidence."
        )
    rows = {
        str(case_id): {
            "parent_passed": bool(value["parent_passed"]),
            "candidate_passed": bool(value["candidate_passed"]),
            "source": "hf_lfm2_intelligence_gate",
            "parent_detail": value["parent_detail"],
            "candidate_detail": value["candidate_detail"],
        }
        for case_id, value in per_case.items()
        if isinstance(value, Mapping)
    }
    parent_palindrome, parent_detail = _safe_palindrome_result(parent_manual_code)
    candidate_palindrome, candidate_detail = _safe_palindrome_result(
        candidate_manual_code
    )
    if "manual_code_00" in rows:
        raise Lfm2PosthocEvaluationError(
            "Manual palindrome correctness was counted twice."
        )
    rows["manual_code_00"] = {
        "parent_passed": parent_palindrome,
        "candidate_passed": candidate_palindrome,
        "source": "runner_safe_palindrome_ast",
        "parent_detail": parent_detail,
        "candidate_detail": candidate_detail,
    }
    parent_correct = [
        case_id for case_id, value in rows.items() if value["parent_passed"]
    ]
    retained = [
        case_id
        for case_id in parent_correct
        if rows[case_id]["candidate_passed"]
    ]
    candidate_correct = [
        case_id for case_id, value in rows.items() if value["candidate_passed"]
    ]
    absolute = len(candidate_correct) / len(rows) if rows else None
    parent_absolute = len(parent_correct) / len(rows) if rows else None
    retention = len(retained) / len(parent_correct) if parent_correct else None
    absolute_retention = (
        None
        if parent_absolute in (None, 0.0) or absolute is None
        else min(1.0, absolute / parent_absolute)
    )
    minimum_candidate_absolute = (
        None
        if parent_absolute is None
        else parent_absolute * minimum_absolute_accuracy
    )
    passed = bool(
        absolute is not None
        and absolute_retention is not None
        and minimum_candidate_absolute is not None
        and retention is not None
        and absolute >= minimum_candidate_absolute
        and retention >= minimum_retention
    )
    core = {
        "schema": "neural-forge-lfm2-combined-reliable-correctness-v1",
        "eligible_case_count": len(rows),
        "candidate_correct_count": len(candidate_correct),
        "parent_absolute_accuracy": parent_absolute,
        "candidate_absolute_accuracy": absolute,
        "candidate_absolute_accuracy_retention": absolute_retention,
        "minimum_absolute_accuracy_ratio_to_parent": (
            minimum_absolute_accuracy
        ),
        "minimum_candidate_absolute_accuracy": minimum_candidate_absolute,
        "parent_correct_count": len(parent_correct),
        "retained_parent_correct_count": len(retained),
        "parent_demonstrated_task_retention": retention,
        "minimum_parent_demonstrated_task_retention": minimum_retention,
        "parent_incorrect_case_ids": sorted(set(rows).difference(parent_correct)),
        "candidate_incorrect_case_ids": sorted(
            set(rows).difference(candidate_correct)
        ),
        "retention_failure_case_ids": sorted(set(parent_correct).difference(retained)),
        "per_case": rows,
        "passed": passed,
        "absolute_correctness_is_separate_from_parent_retention": True,
        "absolute_correctness_gate_is_parent_relative": True,
    }
    return {**core, "correctness_sha256": canonical_sha256(core)}


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    parent_envelope: Path
    expected_parent_envelope_sha256: str
    expected_parent_cache_key: str
    expected_parent_state_fingerprint: str
    tokenizer_directory: Path
    expected_tokenizer_fingerprint: str
    candidate_report: Path
    expected_candidate_report_sha256: str
    report_path: Path
    manual_review_packet_path: Path
    candidate_phase: str | None = None
    manual_ratings_path: Path | None = None
    expected_manual_ratings_sha256: str | None = None
    expected_case_count: int = EXPECTED_CASE_COUNT

    def __post_init__(self) -> None:
        for name in (
            "expected_parent_envelope_sha256",
            "expected_parent_cache_key",
            "expected_parent_state_fingerprint",
            "expected_tokenizer_fingerprint",
            "expected_candidate_report_sha256",
        ):
            _digest(getattr(self, name), name)
        _positive_int(self.expected_case_count, "expected_case_count")
        if (self.manual_ratings_path is None) != (
            self.expected_manual_ratings_sha256 is None
        ):
            raise Lfm2PosthocEvaluationError(
                "manual_ratings_path and expected_manual_ratings_sha256 must "
                "be supplied together."
            )
        if self.expected_manual_ratings_sha256 is not None:
            _digest(
                self.expected_manual_ratings_sha256,
                "expected_manual_ratings_sha256",
            )
        if self.report_path.absolute() == self.manual_review_packet_path.absolute():
            raise Lfm2PosthocEvaluationError(
                "Report and manual packet destinations must differ."
            )


def evaluate_posthoc(config: EvaluationConfig) -> dict[str, Any]:
    """Evaluate only cached responses and write two authenticated artifacts."""

    if not isinstance(config, EvaluationConfig):
        raise Lfm2PosthocEvaluationError(
            "config must be an EvaluationConfig."
        )
    special_identity = load_bound_special_token_ids(
        config.tokenizer_directory,
        expected_tokenizer_fingerprint=config.expected_tokenizer_fingerprint,
    )
    envelope, envelope_sha = load_sealed_parent_envelope(
        config.parent_envelope,
        expected_sha256=config.expected_parent_envelope_sha256,
        expected_parent_cache_key=config.expected_parent_cache_key,
        expected_parent_state_fingerprint=(
            config.expected_parent_state_fingerprint
        ),
        expected_tokenizer_fingerprint=config.expected_tokenizer_fingerprint,
        expected_case_count=config.expected_case_count,
    )
    _, _, candidate_document = _read_json_file(
        config.candidate_report,
        name="candidate report",
        expected_sha256=config.expected_candidate_report_sha256,
    )
    candidate_rows, candidate_selection = extract_candidate_output_rows(
        candidate_document,
        candidate_phase=config.candidate_phase,
    )
    baseline_rows, candidates, derivation = build_gate_evidence(
        envelope,
        candidate_rows,
        eos_token_ids=special_identity["special_token_ids"]["eos"],
    )
    baseline_document = seal_cached_parent_rows(
        baseline_rows,
        parent_cache_key=config.expected_parent_cache_key,
        parent_state_fingerprint=config.expected_parent_state_fingerprint,
        tokenizer_fingerprint=config.expected_tokenizer_fingerprint,
    )
    baseline = load_cached_parent_rows(
        baseline_document,
        expected_document_sha256=canonical_sha256(baseline_document),
        expected_parent_cache_key=config.expected_parent_cache_key,
        expected_parent_state_fingerprint=config.expected_parent_state_fingerprint,
        expected_tokenizer_fingerprint=config.expected_tokenizer_fingerprint,
    )
    policy = IntelligenceGatePolicy(
        required_components=_CORE_REQUIRED_COMPONENTS,
    )
    packet = build_manual_review_packet(baseline, candidates, policy)
    manual_ratings = None
    if config.manual_ratings_path is not None:
        manual_ratings = load_bound_manual_ratings(
            config.manual_ratings_path,
            expected_packet_sha256=packet["packet_sha256"],
            expected_ratings_sha256=str(
                config.expected_manual_ratings_sha256
            ),
        )
    gate_result = evaluate_cached_intelligence(
        baseline,
        candidates,
        policy=policy,
        manual_ratings=manual_ratings,
    )
    gate_report = gate_result.as_dict()
    parent_by_id = {str(row["id"]): row for row in envelope["rows"]}
    candidate_by_id = {
        str(row["case_id"]): row for row in candidate_rows
    }
    combined_correctness = _combined_reliable_correctness(
        gate_report,
        parent_manual_code=str(
            parent_by_id["manual_code_00"]["parent_output_text"]
        ),
        candidate_manual_code=str(
            candidate_by_id["manual_code_00"]["output"]
        ),
        minimum_retention=policy.minimum_task_correctness_retention,
        minimum_absolute_accuracy=policy.minimum_absolute_task_correctness,
    )
    components = {
        str(item["name"]): item for item in gate_report["components"]
    }
    parent_capability = components["parent_capability"]
    reasons = list(gate_report["reasons"])
    if not combined_correctness["passed"]:
        reasons.append("combined_reliable_correctness:threshold_failed")
    reasons = sorted(set(reasons))
    promoted = bool(gate_report["promoted"] and combined_correctness["passed"])
    diagnostic_lower_bound = gate_report["headline_lower_bound"]
    if diagnostic_lower_bound is not None:
        diagnostic_lower_bound = min(
            float(diagnostic_lower_bound),
            float(
                combined_correctness[
                    "candidate_absolute_accuracy_retention"
                ]
            ),
            float(
                combined_correctness["parent_demonstrated_task_retention"]
            ),
        )
    manual_status = str(gate_report["manual_review"]["status"])
    if manual_status == "pending":
        claim = {
            "status": "withheld_pending_manual_review",
            "certified_lower_bound": None,
            "warning": (
                "No intelligence-retention headline is valid until every "
                "manual-review component is present and packet-bound."
            ),
        }
    elif promoted and diagnostic_lower_bound is not None:
        claim = {
            "status": "certified_conservative_lower_bound",
            "certified_lower_bound": diagnostic_lower_bound,
            "warning": (
                "This is the minimum of required component retentions and "
                "absolute-correctness retention, not exact-output matching "
                "or raw benchmark accuracy."
            ),
        }
    else:
        claim = {
            "status": "rejected",
            "certified_lower_bound": None,
            "warning": (
                "A diagnostic score cannot be presented as intelligence "
                "retention when any required component fails."
            ),
        }
    core = {
        "schema": RUNNER_SCHEMA,
        "status": "promoted" if promoted else "rejected",
        "promoted": promoted,
        "reasons": reasons,
        "intelligence_retention_claim": claim,
        "diagnostic_lower_bound": diagnostic_lower_bound,
        "absolute_task_correctness": {
            "eligible_case_count": combined_correctness["eligible_case_count"],
            "parent_correct_count": combined_correctness[
                "parent_correct_count"
            ],
            "candidate_correct_count": (
                combined_correctness["candidate_correct_count"]
            ),
            "parent_rate": combined_correctness[
                "parent_absolute_accuracy"
            ],
            "rate": combined_correctness["candidate_absolute_accuracy"],
            "minimum_rate": combined_correctness[
                "minimum_candidate_absolute_accuracy"
            ],
            "retention_ratio": combined_correctness[
                "candidate_absolute_accuracy_retention"
            ],
            "minimum_retention_ratio": combined_correctness[
                "minimum_absolute_accuracy_ratio_to_parent"
            ],
            "passed": bool(
                combined_correctness["candidate_absolute_accuracy"]
                >= combined_correctness[
                    "minimum_candidate_absolute_accuracy"
                ]
            ),
            "raw_rate_not_presented_as_intelligence_retention": True,
            "gate_is_parent_relative": True,
        },
        "parent_demonstrated_task_retention": {
            "parent_correct_count": combined_correctness[
                "parent_correct_count"
            ],
            "retained_count": combined_correctness[
                "retained_parent_correct_count"
            ],
            "rate": combined_correctness[
                "parent_demonstrated_task_retention"
            ],
            "minimum_rate": combined_correctness[
                "minimum_parent_demonstrated_task_retention"
            ],
            "parent_incorrect_case_ids": combined_correctness[
                "parent_incorrect_case_ids"
            ],
            "separate_from_absolute_task_correctness": True,
        },
        "parent_semantic_capability_retention": parent_capability,
        "combined_reliable_correctness": combined_correctness,
        "core_intelligence_gate": gate_report,
        "manual_review": {
            **dict(gate_report["manual_review"]),
            "packet_sha256": packet["packet_sha256"],
            "packet_case_ids": [
                str(item["case_id"]) for item in packet["cases"]
            ],
        },
        "inputs": {
            "parent_envelope_file_sha256": (
                config.expected_parent_envelope_sha256
            ),
            "parent_envelope_sha256": envelope_sha,
            "parent_cache_key": config.expected_parent_cache_key,
            "parent_state_fingerprint": (
                config.expected_parent_state_fingerprint
            ),
            "tokenizer_fingerprint": config.expected_tokenizer_fingerprint,
            "candidate_report_file_sha256": (
                config.expected_candidate_report_sha256
            ),
            **candidate_selection,
        },
        "bound_special_tokens": special_identity,
        "derived_baseline": {
            "baseline_sha256": baseline.baseline_sha256,
            "baseline_document_sha256": baseline.document_sha256,
            "case_count": len(baseline.cases),
            "validator_manifest": derivation,
        },
        "policy": {
            **policy.as_dict(),
            "policy_sha256": policy.fingerprint,
            "relative_code_behavior_and_instruction_components": (
                "reported but not independently required because the cached "
                "parent demonstrates zero correct recovery-code behavior and "
                "blank instruction cases; their candidate behavior is enforced "
                "by absolute task correctness and code syntax."
            ),
        },
        "execution_audit": {
            "parent_model_loaded": False,
            "candidate_model_loaded": False,
            "parent_forward_passes": 0,
            "candidate_forward_passes": 0,
            "generation_calls": 0,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "gradients_used": False,
            "network_access": False,
            "hub_loader_used": False,
            "candidate_python_exec_used": False,
        },
    }
    report = {**core, "report_sha256": canonical_sha256(core)}
    _write_canonical_json(config.manual_review_packet_path, packet)
    _write_canonical_json(config.report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cached LFM2 parent/candidate outputs without loading or "
            "training either model."
        )
    )
    parser.add_argument("--parent-envelope", type=Path, required=True)
    parser.add_argument(
        "--expected-parent-envelope-sha256", required=True
    )
    parser.add_argument("--expected-parent-cache-key", required=True)
    parser.add_argument(
        "--expected-parent-state-fingerprint", required=True
    )
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-fingerprint", required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--expected-candidate-report-sha256", required=True)
    parser.add_argument("--candidate-phase")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument(
        "--manual-review-packet-path", type=Path, required=True
    )
    parser.add_argument("--manual-ratings-path", type=Path)
    parser.add_argument("--expected-manual-ratings-sha256")
    parser.add_argument(
        "--expected-case-count",
        type=int,
        default=EXPECTED_CASE_COUNT,
    )
    return parser


def config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    return EvaluationConfig(
        parent_envelope=args.parent_envelope,
        expected_parent_envelope_sha256=(
            args.expected_parent_envelope_sha256
        ),
        expected_parent_cache_key=args.expected_parent_cache_key,
        expected_parent_state_fingerprint=(
            args.expected_parent_state_fingerprint
        ),
        tokenizer_directory=args.tokenizer_directory,
        expected_tokenizer_fingerprint=args.expected_tokenizer_fingerprint,
        candidate_report=args.candidate_report,
        expected_candidate_report_sha256=(
            args.expected_candidate_report_sha256
        ),
        candidate_phase=args.candidate_phase,
        report_path=args.report_path,
        manual_review_packet_path=args.manual_review_packet_path,
        manual_ratings_path=args.manual_ratings_path,
        expected_manual_ratings_sha256=(
            args.expected_manual_ratings_sha256
        ),
        expected_case_count=args.expected_case_count,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    report = evaluate_posthoc(config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "promoted": report["promoted"],
                "report_sha256": report["report_sha256"],
                "manual_review": report["manual_review"]["status"],
                "certified_lower_bound": report[
                    "intelligence_retention_claim"
                ]["certified_lower_bound"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["promoted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
