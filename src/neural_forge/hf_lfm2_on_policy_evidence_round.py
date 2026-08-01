"""Sealed on-policy evidence extension rounds for the LFM2 folding lane.

Every closed-form LFM2 solve currently fits parent evidence that was captured
on teacher-forced trajectories.  The folded child leaves that manifold as soon
as it generates autoregressively, and the A12 audit proved cached teacher-forced
fit does not predict autoregressive coherence.

This module closes that gap without a single gradient.  One bounded round:

1. freshly reloads the published standalone child and its tokenizer, recording
   the exact artifact inventory digest;
2. runs **only** the child to produce deterministic greedy rollouts and
   content-addressed deduplicated child-visited prefixes
   (:mod:`neural_forge.hf_on_policy_probe`);
3. re-tokenizes those exact decoded prefixes and packages them as
   ``build_lfm2_circuit_trace_cache`` capture batches;
4. drops every batch the sealed base cache already contains, then calls
   ``build_lfm2_circuit_trace_cache`` **once** with the parent over the
   remaining batches into a new sealed extension cache; and
5. seals a round manifest binding child artifact, parent identity, prompt
   corpus, manual-review disjointness proof, rollout contract, extension cache
   content identity, and counters proving zero gradients, zero optimizer steps,
   zero parent forwards during the rollout, and exactly one sanctioned parent
   capture pass per batch.

Step 3 has two sanctioned parent-input modes, and which one a round uses is the
lane's, not this module's, choice:

* ``parent-tokenizer-canonical`` — no bridge.  The prefill is the parent
  tokenization of the seed prompt and the forced chain is the parent
  tokenization of the child's own continuation.  Nothing about this path
  changed.
* ``target-tokenizer-canonical-then-one-row-mapped-to-donor`` — a strict
  tokenizer bridge was supplied, so the child speaks the target vocabulary.
  The prefill and forced chain are the **child's own** tokenization of those
  same decoded prefixes, mapped row by row through the sealed transport profile
  into donor IDs.  This is exactly the parent-input mode the bridged lane's
  sealed *base* cache was captured under, so base and extension evidence stay
  contract-identical, and the round seals both preimages: the parent-facing
  donor batch **and** the target-token preimage it is the bridge image of.
  Without that preimage no child replay exists for an extension batch, because
  a child replay has to be in target rows.

Downstream closed-form writers consume ``base_cache_path`` plus the round's
extension cache paths together; see
``docs/ON_POLICY_EVIDENCE_EXTENSION_2026-07-31.md``.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
import re
import stat
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceCaptureConfig,
    _prepare_batches as _prepare_lfm2_trace_batches,
    build_lfm2_circuit_trace_cache,
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_direct_local import (
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_model,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2StrictTokenizerBridgeTraceError,
    Lfm2TargetTokenizedRecord,
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from neural_forge.hf_on_policy_probe import (
    GreedyProbeSettings,
    OnPolicyProbeCorpus,
    build_on_policy_probe_corpus,
)
from neural_forge.hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    StrictTokenizerBridgeError,
    strict_tokenizer_bridge_target_package_fingerprint,
)


ROUND_MANIFEST_SCHEMA = "neural-forge-lfm2-on-policy-evidence-round-v1"
ROUND_REQUEST_SCHEMA = "neural-forge-lfm2-on-policy-evidence-round-request-v1"
MANUAL_PROMPT_MANIFEST_SCHEMA = (
    "neural-forge-lfm2-manual-review-prompt-manifest-v1"
)
# Additive on the *same* round schema.  A round captured without a bridge is
# byte-identical to the pre-bridge revision, so every already sealed round
# stays loadable and keeps its `manifest_sha256`; a bridged round carries one
# extra sealed block, and the consumption gate keys on that block's presence.
ROUND_TARGET_PREIMAGE_SCHEMA = (
    "neural-forge-lfm2-on-policy-round-target-preimage-v1"
)
ROUND_MANIFEST_FILENAME = "round-manifest.json"
GENERATION_CACHE_FILENAME = "generation-cache.json"
EXTENSION_CACHE_DIRNAME = "extension-cache"
# Written into a bridged publication beside the child checkpoint; the round
# uses it to prove the reloaded child belongs to the bridge it was handed.
CHILD_TOKENIZER_BRIDGE_REFERENCE_FILENAME = (
    "neural_forge_tokenizer_bridge_reference.json"
)
PARENT_CANONICAL_CAPTURE_INPUT_MODE = "parent-tokenizer-canonical"
BRIDGE_MAPPED_CAPTURE_INPUT_MODE = (
    "target-tokenizer-canonical-then-one-row-mapped-to-donor"
)
DEFAULT_MAX_ROUNDS = 2

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DTYPES = ("bfloat16", "float16", "float32", "float64")
# ``_prepare_batch`` rejects any prompt shorter than three tokens.
_MINIMUM_PROMPT_TOKENS = 3


class Lfm2OnPolicyEvidenceRoundError(RuntimeError):
    """Base failure for on-policy evidence extension rounds."""


class Lfm2OnPolicyEvidenceRoundIntegrityError(Lfm2OnPolicyEvidenceRoundError):
    """A sealed identity, artifact, or counter failed authentication."""


class Lfm2OnPolicyEvidenceRoundBudgetError(Lfm2OnPolicyEvidenceRoundError):
    """The bounded round budget was exceeded or is internally inconsistent."""


class Lfm2OnPolicyEvidenceRoundDisjointnessError(Lfm2OnPolicyEvidenceRoundError):
    """A fit prompt collides with the reserved manual-review prompt set."""


class Lfm2OnPolicyEvidenceRoundConflictError(Lfm2OnPolicyEvidenceRoundError):
    """An existing sealed round belongs to a different request."""


class Lfm2OnPolicyEvidenceRoundBridgeError(Lfm2OnPolicyEvidenceRoundError):
    """A round carries no usable strict-bridge target-token preimage."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2OnPolicyEvidenceRoundError(
            "On-policy evidence rounds require the 'hf' dependencies."
        ) from exc
    return torch


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Lfm2OnPolicyEvidenceRoundError(
            "Round evidence must contain only canonical JSON values."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text_sha256(value: str) -> str:
    return _bytes_sha256(value.encode("utf-8"))


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(result)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(result)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_deep_thaw(item) for item in value]
    return value


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _is_link(path: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", None)
    return path.is_symlink() or bool(callable(is_junction) and is_junction(path))


def _local_directory(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value).expanduser().absolute()
    try:
        result = path.lstat()
    except OSError as exc:
        raise Lfm2OnPolicyEvidenceRoundError(
            f"{name} is missing or unreadable: {path}."
        ) from exc
    if not stat.S_ISDIR(result.st_mode) or _is_link(path):
        raise Lfm2OnPolicyEvidenceRoundError(
            f"{name} must be one real local directory: {path}."
        )
    return path


def _regular_file(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value).expanduser().absolute()
    try:
        result = path.lstat()
    except OSError as exc:
        raise Lfm2OnPolicyEvidenceRoundError(
            f"{name} is missing or unreadable: {path}."
        ) from exc
    if not stat.S_ISREG(result.st_mode) or _is_link(path):
        raise Lfm2OnPolicyEvidenceRoundError(
            f"{name} must be one real local file: {path}."
        )
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sealed_json(path: Path, name: str) -> dict[str, Any]:
    """Parse one canonical JSON document and reject any noncanonical bytes."""

    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            f"Cannot parse {name}: {path}."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            f"{name} must contain exactly one JSON object."
        )
    if raw != _canonical_json(value) + b"\n":
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            f"{name} is noncanonical or was modified after sealing."
        )
    return value


def _write_sealed_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one canonical JSON document exclusively and atomically."""

    payload = _canonical_json(_deep_thaw(value)) + b"\n"
    if path.exists():
        raise Lfm2OnPolicyEvidenceRoundConflictError(
            f"Sealed document already exists: {path}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.stage-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.rename(temporary, path)
        except FileExistsError as exc:
            raise Lfm2OnPolicyEvidenceRoundConflictError(
                f"Sealed document already exists: {path}."
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _sealed_document(body: Mapping[str, Any], seal_name: str) -> dict[str, Any]:
    document = dict(body)
    if seal_name in document:
        raise Lfm2OnPolicyEvidenceRoundError(f"{seal_name} is reserved.")
    return {**document, seal_name: _json_sha256(document)}


def _verify_seal(value: Mapping[str, Any], seal_name: str, name: str) -> str:
    supplied = _sha256(value.get(seal_name), f"{name} {seal_name}")
    body = {key: item for key, item in value.items() if key != seal_name}
    if _json_sha256(body) != supplied:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            f"{name} {seal_name} does not match its contents."
        )
    return supplied


def _tensor_sha256(torch: Any, tensor: Any) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-on-policy-round-tensor-v1\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in value.shape]))
    digest.update(b"\0")
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _gradient_state(torch: Any, model: Any) -> tuple[tuple[Any, ...], ...]:
    """Snapshot per-parameter gradient presence, identity, and value."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return ()
    rows = []
    for name, parameter in named_parameters():
        grad = getattr(parameter, "grad", None)
        rows.append(
            (
                str(name),
                bool(getattr(parameter, "requires_grad", False)),
                None if grad is None else id(grad),
                None if grad is None else _tensor_sha256(torch, grad),
            )
        )
    return tuple(rows)


def _install_forward_counter(model: Any) -> tuple[dict[str, int], list[Any]]:
    """Count every module forward executed anywhere inside ``model``."""

    counter = {"calls": 0}

    def hook(_module: Any, _inputs: Any, _output: Any) -> None:
        counter["calls"] += 1

    handles = []
    modules = getattr(model, "modules", None)
    if callable(modules):
        for module in modules():
            register = getattr(module, "register_forward_hook", None)
            if callable(register):
                handles.append(register(hook))
    return counter, handles


def _torch_dtype(torch: Any, name: Any) -> Any:
    if name not in _DTYPES:
        raise ValueError(f"dtype must be one of {list(_DTYPES)}, got {name!r}.")
    return getattr(torch, str(name))


def _artifact_inventory(root: Path, name: str) -> tuple[dict[str, Any], ...]:
    """Enumerate one published artifact tree as content-addressed records."""

    try:
        entries = tuple(sorted(root.rglob("*"), key=lambda path: path.as_posix()))
    except OSError as exc:
        raise Lfm2OnPolicyEvidenceRoundError(
            f"{name} tree could not be enumerated: {root}."
        ) from exc
    records: list[dict[str, Any]] = []
    for path in entries:
        relative = path.relative_to(root).as_posix()
        if _is_link(path):
            raise Lfm2OnPolicyEvidenceRoundIntegrityError(
                f"{name} contains link-like path {relative!r}."
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise Lfm2OnPolicyEvidenceRoundIntegrityError(
                f"{name} entry {relative!r} is not a regular file."
            )
        records.append(
            {
                "relative_path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    if not records:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(f"{name} is empty: {root}.")
    return tuple(records)


def _artifact_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return _json_sha256(
        {
            "schema": "neural-forge-lfm2-on-policy-child-artifact-v1",
            "file_records": [dict(record) for record in records],
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2OnPolicyRolloutConfig:
    """Bounded, deterministic, child-only rollout contract for one round."""

    max_prompts: int = 64
    max_new_tokens: int = 16
    max_context_tokens: int = 512
    greedy: bool = True
    seed: int = 0
    strict_byte_identity: bool = True
    max_capture_batches: int = 256
    truncate_forced_decode_to_capture_budget: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_prompts",
            "max_new_tokens",
            "max_context_tokens",
            "max_capture_batches",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        object.__setattr__(self, "seed", _nonnegative_int(self.seed, "seed"))
        for name in (
            "greedy",
            "strict_byte_identity",
            "truncate_forced_decode_to_capture_budget",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")
        if not self.greedy:
            raise ValueError(
                "Only greedy child rollouts are sanctioned evidence; sampled "
                "trajectories are not reproducible from the sealed manifest."
            )
        if self.max_context_tokens <= self.max_new_tokens:
            raise ValueError("max_context_tokens must exceed max_new_tokens.")

    def probe_settings(self) -> GreedyProbeSettings:
        """Return the child-only greedy probe contract for this round."""

        return GreedyProbeSettings(
            max_new_tokens=self.max_new_tokens,
            max_context_tokens=self.max_context_tokens,
            trace_batch_size=1,
            seed=self.seed,
            strict_byte_identity=self.strict_byte_identity,
            # Step 0 is the seed prefix; the parent capture batch needs it as
            # the prefill boundary between prompt and child continuation.
            include_seed_prefix=True,
        )

    def as_dict(self) -> dict[str, Any]:
        body = {
            "algorithm": "greedy-argmax",
            "do_sample": False,
            "num_beams": 1,
            "max_prompts": self.max_prompts,
            "max_new_tokens": self.max_new_tokens,
            "max_context_tokens": self.max_context_tokens,
            "seed": self.seed,
            "strict_byte_identity": self.strict_byte_identity,
            "max_capture_batches": self.max_capture_batches,
            "truncate_forced_decode_to_capture_budget": (
                self.truncate_forced_decode_to_capture_budget
            ),
            "child_only_execution": True,
            "parent_executed_during_rollout": False,
        }
        return {**body, "rollout_sha256": _json_sha256(body)}


@dataclass(frozen=True, slots=True)
class Lfm2ManualReviewPromptManifest:
    """Reserved behavioral-review prompts that fit evidence must never touch."""

    path: Path
    manifest_sha256: str
    form: str
    prompt_count: int
    digests: frozenset[str]

    @property
    def digest_set_sha256(self) -> str:
        return _json_sha256(sorted(self.digests))

    def as_dict(self) -> dict[str, Any]:
        return {
            "manual_manifest_path": str(self.path),
            "manual_manifest_sha256": self.manifest_sha256,
            "manual_manifest_form": self.form,
            "manual_prompt_count": self.prompt_count,
            "manual_prompt_digest_count": len(self.digests),
            "manual_prompt_digest_set_sha256": self.digest_set_sha256,
        }


def _manual_digests_from_texts(texts: Sequence[str]) -> tuple[int, set[str]]:
    digests: set[str] = set()
    for text in texts:
        digests.add(_text_sha256(text))
        digests.add(_text_sha256(text.strip()))
    return len(texts), digests


def load_manual_review_prompt_manifest(
    path: str | os.PathLike[str],
) -> Lfm2ManualReviewPromptManifest:
    """Load and authenticate the reserved manual-review prompt manifest.

    Three fail-closed shapes are accepted: a sealed digest manifest, a bare
    JSON array of SHA-256 digests, and a review-prompt document whose rows each
    carry one nonempty ``prompt`` string.  Text shapes contribute both the exact
    and whitespace-stripped digest so a cosmetic edit cannot smuggle a reserved
    behavioral prompt into the fit corpus.
    """

    resolved = _regular_file(path, "manual review prompt manifest")
    manifest_sha256 = _file_sha256(resolved)
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            f"Manual review prompt manifest is not valid JSON: {resolved}."
        ) from exc

    rows: Any = document
    form = "digest-list"
    if isinstance(document, Mapping):
        if document.get("schema") == MANUAL_PROMPT_MANIFEST_SCHEMA:
            if "manifest_sha256" in document:
                _verify_seal(
                    document,
                    "manifest_sha256",
                    "manual review prompt manifest",
                )
            rows = document.get("prompt_sha256")
            form = "sealed-digest-manifest"
        elif "prompts" in document:
            rows = document.get("prompts")
            form = "review-prompt-rows"
        else:
            raise Lfm2OnPolicyEvidenceRoundIntegrityError(
                "Manual review prompt manifest object must use schema "
                f"{MANUAL_PROMPT_MANIFEST_SCHEMA!r} or carry a 'prompts' list."
            )
    if not isinstance(rows, list) or not rows:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "Manual review prompt manifest must contain a nonempty list."
        )

    digest_rows = all(isinstance(row, str) for row in rows)
    if form == "sealed-digest-manifest" and not digest_rows:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "A sealed manual review prompt manifest must list SHA-256 digests."
        )
    if form == "digest-list" and digest_rows:
        return Lfm2ManualReviewPromptManifest(
            path=resolved,
            manifest_sha256=manifest_sha256,
            form="digest-list",
            prompt_count=len(rows),
            digests=frozenset(
                _sha256(row, "manual review prompt digest") for row in rows
            ),
        )
    if form == "sealed-digest-manifest":
        return Lfm2ManualReviewPromptManifest(
            path=resolved,
            manifest_sha256=manifest_sha256,
            form=form,
            prompt_count=len(rows),
            digests=frozenset(
                _sha256(row, "manual review prompt digest") for row in rows
            ),
        )

    texts: list[str] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Lfm2OnPolicyEvidenceRoundIntegrityError(
                f"Manual review prompt row {position} must be one SHA-256 digest "
                "or one object carrying a 'prompt' string."
            )
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise Lfm2OnPolicyEvidenceRoundIntegrityError(
                f"Manual review prompt row {position} must carry one nonempty "
                "'prompt' string; supply a digest manifest for chat-message rows."
            )
        texts.append(prompt)
    count, digests = _manual_digests_from_texts(texts)
    return Lfm2ManualReviewPromptManifest(
        path=resolved,
        manifest_sha256=manifest_sha256,
        form="review-prompt-rows" if form == "review-prompt-rows" else (
            "prompt-object-list"
        ),
        prompt_count=count,
        digests=frozenset(digests),
    )


def _normalized_fit_prompts(
    fit_prompts: Sequence[str], max_prompts: int
) -> tuple[str, ...]:
    if isinstance(fit_prompts, (str, bytes)) or not isinstance(fit_prompts, Sequence):
        raise TypeError("fit_prompts must be a sequence of strings.")
    prompts = tuple(fit_prompts)
    if not prompts:
        raise ValueError("fit_prompts must not be empty.")
    seen: set[str] = set()
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"fit prompt {index} must be a nonempty string.")
        if prompt in seen:
            raise ValueError(f"fit prompt {index} is a duplicate.")
        seen.add(prompt)
    if len(prompts) > max_prompts:
        raise Lfm2OnPolicyEvidenceRoundBudgetError(
            f"{len(prompts)} fit prompts exceed the bounded rollout budget "
            f"max_prompts={max_prompts}."
        )
    return prompts


def _prompt_corpus_evidence(prompts: Sequence[str]) -> dict[str, Any]:
    digests = [_text_sha256(prompt) for prompt in prompts]
    body = {
        "prompt_count": len(prompts),
        "prompt_sha256": digests,
    }
    return {**body, "prompt_corpus_sha256": _json_sha256(body)}


def _verified_disjointness(
    prompts: Sequence[str],
    manual: Lfm2ManualReviewPromptManifest,
) -> dict[str, Any]:
    """Fail closed when any fit prompt hashes into the reserved manual set."""

    overlapping = []
    for index, prompt in enumerate(prompts):
        exact = _text_sha256(prompt)
        stripped = _text_sha256(prompt.strip())
        if exact in manual.digests or stripped in manual.digests:
            overlapping.append({"fit_prompt_index": index, "prompt_sha256": exact})
    if overlapping:
        indices = [entry["fit_prompt_index"] for entry in overlapping]
        raise Lfm2OnPolicyEvidenceRoundDisjointnessError(
            f"Fit prompts {indices} are reserved manual-review prompts. "
            "On-policy extension evidence must stay disjoint from the "
            "behavioral review set."
        )
    return {
        **manual.as_dict(),
        "comparison": "utf8-sha256-exact-and-stripped",
        "overlapping_prompt_count": 0,
        "verified_disjoint": True,
    }


@dataclass(frozen=True, slots=True)
class _CaptureBatch:
    """One child-visited trajectory packaged for the parent trace cache.

    ``prompt_token_ids``/``forced_decode_token_ids`` are always the IDs the
    parent is actually evaluated on.  On a bridged round they are the one-row
    donor image of ``target_prompt_token_ids``/``target_forced_decode_token_ids``,
    which are the child's own tokenization of exactly the same decoded text.
    """

    prompt_token_ids: tuple[int, ...]
    forced_decode_token_ids: tuple[int, ...]
    trajectory_sha256: str
    origin_prompt_indices: tuple[int, ...]
    visited_prefix_row_ids: tuple[str, ...]
    truncated: bool
    target_prompt_token_ids: tuple[int, ...] = ()
    target_forced_decode_token_ids: tuple[int, ...] = ()
    target_trajectory_sha256: str | None = None
    collapsed_target_trajectory_sha256: tuple[str, ...] = ()
    prompt_utf8_sha256: str | None = None
    trajectory_utf8_sha256: str | None = None

    def as_model_batch(self, torch: Any) -> dict[str, Any]:
        input_ids = torch.tensor([list(self.prompt_token_ids)], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "forced_decode_ids": torch.tensor(
                [list(self.forced_decode_token_ids)], dtype=torch.long
            ),
        }

    def as_target_record(self, record_id: str) -> Lfm2TargetTokenizedRecord:
        """Return the target-token preimage as one strict-bridge record."""

        torch = _torch()
        input_ids = torch.tensor(
            [list(self.target_prompt_token_ids)], dtype=torch.long
        )
        return Lfm2TargetTokenizedRecord(
            record_id=record_id,
            # A child-visited state is a fit state; nothing else is legal.
            split="fit",
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            forced_decode_ids=torch.tensor(
                [list(self.target_forced_decode_token_ids)], dtype=torch.long
            ),
        )


def _trajectory_sha256(
    prompt_ids: Sequence[int], forced_ids: Sequence[int]
) -> str:
    return _json_sha256(
        {
            "schema": "neural-forge-lfm2-on-policy-trajectory-v1",
            "input_ids": list(prompt_ids),
            "forced_decode_ids": list(forced_ids),
        }
    )


def _capture_batches_from_corpus(
    corpus: OnPolicyProbeCorpus,
    *,
    parent_tokenizer_name: str,
    max_decode_steps: int,
    truncate: bool,
    role: str = "parent",
    target_to_donor_row_indices: tuple[int, ...] | None = None,
) -> tuple[tuple[_CaptureBatch, ...], list[dict[str, Any]]]:
    """Convert deduplicated child-visited prefixes into parent capture batches.

    Each surviving batch is one exact child rollout expressed in one token
    space: the prefill is that tokenizer's encoding of the seed prompt and the
    forced decode chain is its encoding of the child's own greedy continuation.
    Capturing that batch evaluates the parent at every child-visited prefix of
    the trajectory in one pass.

    ``role`` names the tokenizer the rows come from and only shapes the
    rejection reasons.  With ``target_to_donor_row_indices`` the rows are the
    child's own target tokenization and every retained batch additionally
    carries its one-row donor image, which is what the parent is evaluated on.
    """

    parent_rows = corpus.tokenizations[parent_tokenizer_name]
    steps_by_prompt: dict[int, dict[int, int]] = {}
    for row_index, row in enumerate(corpus.rows):
        for origin in row.origins:
            steps_by_prompt.setdefault(origin.prompt_index, {})[origin.prefix_step] = (
                row_index
            )

    rejected: list[dict[str, Any]] = []
    ordered: list[_CaptureBatch] = []
    by_trajectory: dict[str, int] = {}
    for prompt_index in sorted(steps_by_prompt):
        steps = steps_by_prompt[prompt_index]
        ordered_steps = sorted(steps)
        if ordered_steps != list(range(len(ordered_steps))):
            rejected.append(
                {
                    "fit_prompt_index": prompt_index,
                    "reason": "child_prefix_steps_are_not_contiguous",
                }
            )
            continue
        last_step = ordered_steps[-1]
        if last_step == 0:
            rejected.append(
                {
                    "fit_prompt_index": prompt_index,
                    "reason": "child_produced_no_continuation",
                }
            )
            continue
        prompt_ids = tuple(parent_rows[steps[0]].token_ids)
        full_ids = tuple(parent_rows[steps[last_step]].token_ids)
        if len(prompt_ids) < _MINIMUM_PROMPT_TOKENS:
            rejected.append(
                {
                    "fit_prompt_index": prompt_index,
                    "reason": f"{role}_prompt_shorter_than_capture_minimum",
                }
            )
            continue
        if full_ids[: len(prompt_ids)] != prompt_ids:
            # The tokenizer merged across the prompt/continuation boundary, so
            # the forced chain would not be the child's states.
            rejected.append(
                {
                    "fit_prompt_index": prompt_index,
                    "reason": f"{role}_retokenization_shifted_the_prompt_boundary",
                }
            )
            continue
        forced_ids = full_ids[len(prompt_ids) :]
        if not forced_ids:
            rejected.append(
                {
                    "fit_prompt_index": prompt_index,
                    "reason": f"{role}_retokenized_continuation_is_empty",
                }
            )
            continue
        truncated = False
        if len(forced_ids) > max_decode_steps:
            if not truncate:
                raise Lfm2OnPolicyEvidenceRoundBudgetError(
                    f"Fit prompt {prompt_index} re-tokenizes into "
                    f"{len(forced_ids)} {role} forced-decode tokens, which "
                    f"exceeds max_decode_steps={max_decode_steps}. Lower "
                    "max_new_tokens, raise max_decode_steps, or enable "
                    "truncate_forced_decode_to_capture_budget."
                )
            forced_ids = forced_ids[:max_decode_steps]
            truncated = True
        target_prompt_ids: tuple[int, ...] = ()
        target_forced_ids: tuple[int, ...] = ()
        target_digest: str | None = None
        if target_to_donor_row_indices is not None:
            capacity = len(target_to_donor_row_indices)
            if max((*prompt_ids, *forced_ids)) >= capacity:
                rejected.append(
                    {
                        "fit_prompt_index": prompt_index,
                        "reason": f"{role}_token_outside_bridge_row_capacity",
                    }
                )
                continue
            target_prompt_ids = prompt_ids
            target_forced_ids = forced_ids
            target_digest = _trajectory_sha256(prompt_ids, forced_ids)
            prompt_ids = tuple(
                target_to_donor_row_indices[value] for value in prompt_ids
            )
            forced_ids = tuple(
                target_to_donor_row_indices[value] for value in forced_ids
            )
        row_ids = tuple(
            corpus.rows[steps[step]].row_id for step in ordered_steps
        )
        digest = _trajectory_sha256(prompt_ids, forced_ids)
        existing = by_trajectory.get(digest)
        if existing is not None:
            previous = ordered[existing]
            # The strict transport profile is not required to be injective, so
            # two distinct target chains can share one donor image.  The parent
            # cannot tell them apart; the first is kept and the collapse is
            # recorded rather than silently dropped.
            collapsed = previous.collapsed_target_trajectory_sha256
            if (
                target_digest is not None
                and target_digest != previous.target_trajectory_sha256
                and target_digest not in collapsed
            ):
                collapsed = collapsed + (target_digest,)
            ordered[existing] = _CaptureBatch(
                prompt_token_ids=previous.prompt_token_ids,
                forced_decode_token_ids=previous.forced_decode_token_ids,
                trajectory_sha256=previous.trajectory_sha256,
                origin_prompt_indices=(
                    previous.origin_prompt_indices + (prompt_index,)
                ),
                visited_prefix_row_ids=previous.visited_prefix_row_ids,
                truncated=previous.truncated or truncated,
                target_prompt_token_ids=previous.target_prompt_token_ids,
                target_forced_decode_token_ids=(
                    previous.target_forced_decode_token_ids
                ),
                target_trajectory_sha256=previous.target_trajectory_sha256,
                collapsed_target_trajectory_sha256=collapsed,
                prompt_utf8_sha256=previous.prompt_utf8_sha256,
                trajectory_utf8_sha256=previous.trajectory_utf8_sha256,
            )
            continue
        by_trajectory[digest] = len(ordered)
        ordered.append(
            _CaptureBatch(
                prompt_token_ids=prompt_ids,
                forced_decode_token_ids=forced_ids,
                trajectory_sha256=digest,
                origin_prompt_indices=(prompt_index,),
                visited_prefix_row_ids=row_ids,
                truncated=truncated,
                target_prompt_token_ids=target_prompt_ids,
                target_forced_decode_token_ids=target_forced_ids,
                target_trajectory_sha256=target_digest,
                prompt_utf8_sha256=corpus.rows[steps[0]].utf8_sha256,
                trajectory_utf8_sha256=(
                    corpus.rows[steps[last_step]].utf8_sha256
                ),
            )
        )
    return tuple(ordered), rejected


def _verified_round_bridge(
    value: Any,
    expected_artifact_sha256: str | None,
) -> StrictTokenizerBridgeArtifact:
    """Authenticate the strict bridge a bridged round is captured under."""

    if not isinstance(value, StrictTokenizerBridgeArtifact):
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            "tokenizer_bridge must be a StrictTokenizerBridgeArtifact or None."
        )
    try:
        value.verify_files()
    except StrictTokenizerBridgeError as exc:
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            "Strict tokenizer bridge file authentication failed."
        ) from exc
    if expected_artifact_sha256 is not None and value.artifact_sha256 != _sha256(
        expected_artifact_sha256, "expected_tokenizer_bridge_artifact_sha256"
    ):
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            "Strict tokenizer bridge differs from the expected artifact."
        )
    return value


def _bridge_reference_body(bridge: StrictTokenizerBridgeArtifact) -> dict[str, Any]:
    profile = bridge.transport_profile
    return {
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "transport_profile_sha256": bridge.profile_fingerprint,
        "target_to_donor_row_indices_sha256": _json_sha256(
            list(profile.target_to_donor_row_indices)
        ),
        "target_row_capacity": int(profile.target_row_capacity),
        "donor_row_capacity": int(profile.donor_row_capacity),
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "target_tokenizer_package_fingerprint": (
            strict_tokenizer_bridge_target_package_fingerprint(bridge)
        ),
    }


def _child_bridge_reference(
    child_root: Path,
    bridge: StrictTokenizerBridgeArtifact,
) -> dict[str, Any]:
    """Bind the published child's own bridge reference to this bridge.

    A bridged publication writes ``neural_forge_tokenizer_bridge_reference``
    beside the child checkpoint.  Reading it here is what proves the reloaded
    child is the child of *this* bridge rather than a same-shaped stranger.
    """

    path = child_root / CHILD_TOKENIZER_BRIDGE_REFERENCE_FILENAME
    try:
        resolved = _regular_file(path, "child tokenizer bridge reference")
    except Lfm2OnPolicyEvidenceRoundError as exc:
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            "The published child carries no strict tokenizer-bridge reference "
            f"at {path}; it was not published on a bridged lane."
        ) from exc
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            f"Child tokenizer bridge reference is not valid JSON: {resolved}."
        ) from exc
    if not isinstance(document, Mapping):
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            "Child tokenizer bridge reference must contain one JSON object."
        )
    expected = {
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
    }
    for name, value in expected.items():
        if document.get(name) != value:
            raise Lfm2OnPolicyEvidenceRoundBridgeError(
                f"The published child's bridge reference {name} differs from "
                "the strict tokenizer bridge this round was handed."
            )
    return {
        "path": str(resolved),
        "file_sha256": _file_sha256(resolved),
        **{str(name): document.get(name) for name in sorted(expected)},
        "target_tokenizer_package_fingerprint": document.get(
            "target_tokenizer_package_fingerprint"
        ),
    }


def _target_preimage_records(
    batches: Sequence[_CaptureBatch],
    *,
    round_index: int,
) -> tuple[Lfm2TargetTokenizedRecord, ...]:
    return tuple(
        batch.as_target_record(
            _target_preimage_record_id(round_index, position)
        )
        for position, batch in enumerate(batches)
    )


def _target_preimage_record_id(round_index: int, position: int) -> str:
    return f"on-policy-round-{round_index:03d}-batch-{position:06d}"


def _sealed_target_preimage(
    *,
    round_index: int,
    bridge: StrictTokenizerBridgeArtifact,
    batches: Sequence[_CaptureBatch],
    aligned_records: Sequence[Any],
    adapter_evidence_sha256: str,
    donor_input_sha256: Sequence[str],
    child_reference: Mapping[str, Any],
    parent_tokenizer_name: str,
    parent_preimages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal the child's own tokenization of every captured extension batch."""

    records = []
    for position, (batch, record) in enumerate(
        zip(batches, aligned_records, strict=True)
    ):
        if record.donor_cache_input_sha256 != donor_input_sha256[position]:
            raise Lfm2OnPolicyEvidenceRoundBridgeError(
                f"Target preimage {position} does not map onto the sealed "
                "donor batch the parent was captured on."
            )
        records.append(
            {
                "index": position,
                "record_id": record.record_id,
                "split": record.split,
                "target_input_ids": list(batch.target_prompt_token_ids),
                "target_forced_decode_ids": list(
                    batch.target_forced_decode_token_ids
                ),
                "target_prompt_token_count": len(
                    batch.target_prompt_token_ids
                ),
                "target_forced_decode_token_count": len(
                    batch.target_forced_decode_token_ids
                ),
                "target_token_ids_sha256": _target_token_ids_sha256(
                    batch.target_prompt_token_ids,
                    batch.target_forced_decode_token_ids,
                ),
                "target_trajectory_sha256": batch.target_trajectory_sha256,
                "donor_trajectory_sha256": batch.trajectory_sha256,
                "target_cache_input_sha256": record.target_cache_input_sha256,
                "donor_cache_input_sha256": record.donor_cache_input_sha256,
                "alignment_sha256": record.alignment_sha256,
                "numerically_different_token_count": int(
                    record.numerically_different_token_count
                ),
                "prompt_utf8_sha256": batch.prompt_utf8_sha256,
                "trajectory_utf8_sha256": batch.trajectory_utf8_sha256,
                "visited_prefix_row_ids": list(batch.visited_prefix_row_ids),
                "collapsed_target_trajectory_sha256": list(
                    batch.collapsed_target_trajectory_sha256
                ),
                "parent_tokenizer_preimage": dict(parent_preimages[position]),
            }
        )
    body = {
        "schema": ROUND_TARGET_PREIMAGE_SCHEMA,
        "round_index": round_index,
        **_bridge_reference_body(bridge),
        "child_bridge_reference": dict(child_reference),
        "child_tokenizer_is_the_sealed_bridge_target": True,
        "trace_adapter_evidence_sha256": adapter_evidence_sha256,
        "parent_cache_input_mode": BRIDGE_MAPPED_CAPTURE_INPUT_MODE,
        "parent_tokenizer_name": parent_tokenizer_name,
        "alignment": {
            "identity": "exact-decoded-utf8-sha256",
            "boundary_check": (
                "target-retokenized-prompt-is-an-exact-prefix-of-the-target-"
                "retokenized-trajectory"
            ),
            "target_to_donor_relation": "one-row-map",
            "numeric_token_id_equality_required": False,
            "child_replay_uses_original_target_ids": True,
            "parent_cache_uses_mapped_donor_ids": True,
        },
        "record_count": len(records),
        "records": records,
        "parent_model_forwards_outside_sealed_capture": 0,
    }
    return _sealed_document(body, "target_preimage_sha256")


def _parent_tokenizer_preimages(
    corpus: OnPolicyProbeCorpus,
    parent_tokenizer_name: str,
    batches: Sequence[_CaptureBatch],
) -> tuple[dict[str, Any], ...]:
    """Record the parent tokenizer's own reading of the same decoded rows.

    A bridged round does not evaluate the parent on these IDs — the row map is
    what keeps parent and child rows aligned — but sealing them keeps both
    preimages of every extension batch in the manifest, so a reviewer can see
    exactly what the parent tokenizer would have produced from the same text.
    """

    rows_by_id = {row.row_id: position for position, row in enumerate(corpus.rows)}
    parent_rows = corpus.tokenizations[parent_tokenizer_name]
    result: list[dict[str, Any]] = []
    for batch in batches:
        first = tuple(
            parent_rows[rows_by_id[batch.visited_prefix_row_ids[0]]].token_ids
        )
        full = tuple(
            parent_rows[rows_by_id[batch.visited_prefix_row_ids[-1]]].token_ids
        )
        preserved = full[: len(first)] == first
        forced = full[len(first) :] if preserved else ()
        result.append(
            {
                "tokenizer_name": parent_tokenizer_name,
                "prompt_token_count": len(first),
                "forced_decode_token_count": len(forced),
                "token_ids_sha256": _json_sha256(
                    {
                        "schema": (
                            "neural-forge-lfm2-on-policy-parent-token-ids-v1"
                        ),
                        "input_ids": list(first),
                        "forced_decode_ids": list(forced),
                    }
                ),
                "boundary_preserved": bool(preserved),
                "used_for_parent_capture": False,
            }
        )
    return tuple(result)


def _target_token_ids_sha256(
    prompt_ids: Sequence[int], forced_ids: Sequence[int]
) -> str:
    return _json_sha256(
        {
            "schema": "neural-forge-lfm2-on-policy-target-token-ids-v1",
            "target_input_ids": [int(value) for value in prompt_ids],
            "target_forced_decode_ids": [int(value) for value in forced_ids],
        }
    )


def _token_id_tuple(value: Any, name: str, capacity: int) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            f"{name} must be a nonempty list of token IDs."
        )
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise Lfm2OnPolicyEvidenceRoundBridgeError(
                f"{name} must contain only integer token IDs."
            )
        if item < 0 or item >= capacity:
            raise Lfm2OnPolicyEvidenceRoundBridgeError(
                f"{name} contains ID {item} outside target row capacity "
                f"{capacity}."
            )
        result.append(int(item))
    if not result:
        raise Lfm2OnPolicyEvidenceRoundBridgeError(f"{name} must be nonempty.")
    return tuple(result)


def _verify_sealed_target_preimage(
    block: Mapping[str, Any],
    *,
    cache: Lfm2CircuitTraceCache,
    name: str,
) -> None:
    """Authenticate one sealed target preimage against its extension cache."""

    if block.get("schema") != ROUND_TARGET_PREIMAGE_SCHEMA:
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            f"{name} target preimage uses an unsupported schema."
        )
    _verify_seal(block, "target_preimage_sha256", f"{name} target preimage")
    for field in (
        "bridge_artifact_sha256",
        "transport_profile_sha256",
        "target_to_donor_row_indices_sha256",
        "trace_adapter_evidence_sha256",
    ):
        _sha256(block.get(field), f"{name} target preimage {field}")
    if block.get("parent_cache_input_mode") != BRIDGE_MAPPED_CAPTURE_INPUT_MODE:
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            f"{name} target preimage does not declare the bridge-mapped "
            "parent input mode."
        )
    alignment = _require_mapping(
        block.get("alignment"), f"{name} target preimage alignment"
    )
    if (
        alignment.get("identity") != "exact-decoded-utf8-sha256"
        or alignment.get("numeric_token_id_equality_required") is not False
        or alignment.get("child_replay_uses_original_target_ids") is not True
    ):
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            f"{name} target preimage alignment contract is not the sealed one."
        )
    capacity = block.get("target_row_capacity")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            f"{name} target preimage target_row_capacity is invalid."
        )
    records = block.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            f"{name} target preimage must carry an ordered record list."
        )
    if len(records) != block.get("record_count") or len(records) != cache.batch_count:
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            f"{name} target preimage does not cover every sealed extension "
            "batch exactly once."
        )
    donor_inputs = [
        str(value)
        for value in cache.manifest["corpus"]["ordered_batch_input_sha256"]
    ]
    seen_ids: set[str] = set()
    for position, record in enumerate(records):
        entry = _require_mapping(record, f"{name} target preimage record")
        if entry.get("index") != position:
            raise Lfm2OnPolicyEvidenceRoundBridgeError(
                f"{name} target preimage records are not canonically ordered."
            )
        record_id = entry.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise Lfm2OnPolicyEvidenceRoundBridgeError(
                f"{name} target preimage record {position} has no record_id."
            )
        if record_id in seen_ids:
            raise Lfm2OnPolicyEvidenceRoundBridgeError(
                f"{name} target preimage repeats record_id {record_id!r}."
            )
        seen_ids.add(record_id)
        if entry.get("split") != "fit":
            raise Lfm2OnPolicyEvidenceRoundBridgeError(
                f"{name} target preimage record {position} is not a fit row; "
                "every child-visited state is a fit state."
            )
        prompt_ids = _token_id_tuple(
            entry.get("target_input_ids"),
            f"{name} target preimage record {position} target_input_ids",
            capacity,
        )
        forced_ids = _token_id_tuple(
            entry.get("target_forced_decode_ids"),
            f"{name} target preimage record {position} "
            "target_forced_decode_ids",
            capacity,
        )
        if (
            entry.get("target_prompt_token_count") != len(prompt_ids)
            or entry.get("target_forced_decode_token_count") != len(forced_ids)
            or entry.get("target_token_ids_sha256")
            != _target_token_ids_sha256(prompt_ids, forced_ids)
        ):
            raise Lfm2OnPolicyEvidenceRoundBridgeError(
                f"{name} target preimage record {position} token accounting "
                "does not match its own IDs."
            )
        for field in (
            "target_cache_input_sha256",
            "donor_cache_input_sha256",
            "alignment_sha256",
            "prompt_utf8_sha256",
            "trajectory_utf8_sha256",
        ):
            _sha256(
                entry.get(field),
                f"{name} target preimage record {position} {field}",
            )
        if entry.get("donor_cache_input_sha256") != donor_inputs[position]:
            raise Lfm2OnPolicyEvidenceRoundBridgeError(
                f"{name} target preimage record {position} is not the "
                "preimage of the sealed extension batch at that index."
            )


def _round_directory(output_root: Path, round_index: int) -> Path:
    return output_root / f"round-{round_index:03d}"


def _validate_round_budget(round_index: Any, max_rounds: Any) -> tuple[int, int]:
    bound = _positive_int(max_rounds, "max_rounds")
    index = _nonnegative_int(round_index, "round_index")
    if index >= bound:
        raise Lfm2OnPolicyEvidenceRoundBudgetError(
            f"round_index={index} exceeds the bounded on-policy evidence "
            f"budget max_rounds={bound}."
        )
    return index, bound


@dataclass(frozen=True, slots=True)
class Lfm2OnPolicyEvidenceReferences:
    """Base plus extension cache paths for one consolidated closed-form solve."""

    base_cache_path: Path
    extension_cache_paths: tuple[Path, ...]

    def as_tuple(self) -> tuple[Path, ...]:
        return (self.base_cache_path, *self.extension_cache_paths)

    def __iter__(self):
        return iter(self.as_tuple())

    def __len__(self) -> int:
        return 1 + len(self.extension_cache_paths)

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_cache_path": str(self.base_cache_path),
            "extension_cache_paths": [
                str(path) for path in self.extension_cache_paths
            ],
        }


@dataclass(frozen=True, slots=True)
class Lfm2OnPolicyEvidenceRound:
    """One sealed, verified on-policy evidence extension round."""

    manifest_path: Path
    round_directory: Path
    base_cache_path: Path
    extension_cache_path: Path
    manifest: Mapping[str, Any]

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest["manifest_sha256"])

    @property
    def request_sha256(self) -> str:
        return str(self.manifest["request_sha256"])

    @property
    def round_index(self) -> int:
        return int(self.manifest["round"]["index"])

    @property
    def max_rounds(self) -> int:
        return int(self.manifest["round"]["max_rounds"])

    @property
    def extension_cache_key(self) -> str:
        return str(self.manifest["extension_cache"]["cache_key"])

    @property
    def batch_count(self) -> int:
        return int(self.manifest["extension_cache"]["batch_count"])

    @property
    def parent_fingerprint(self) -> str:
        return str(self.manifest["parent"]["identity_sha256"])

    @property
    def target_preimage(self) -> Mapping[str, Any] | None:
        """The sealed target-token preimage block, or ``None`` if unbridged."""

        value = self.manifest.get("target_preimage")
        return None if value is None else value

    @property
    def carries_target_preimage(self) -> bool:
        return self.target_preimage is not None

    @property
    def target_preimage_bridge_artifact_sha256(self) -> str | None:
        block = self.target_preimage
        return None if block is None else str(block["bridge_artifact_sha256"])

    @property
    def parent_cache_input_mode(self) -> str:
        block = self.target_preimage
        if block is None:
            return PARENT_CANONICAL_CAPTURE_INPUT_MODE
        return str(block["parent_cache_input_mode"])

    def references(self) -> Lfm2OnPolicyEvidenceReferences:
        return Lfm2OnPolicyEvidenceReferences(
            base_cache_path=self.base_cache_path,
            extension_cache_paths=(self.extension_cache_path,),
        )


def _manifest_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser().absolute()
    if path.is_dir() and not _is_link(path):
        path = path / ROUND_MANIFEST_FILENAME
    return _regular_file(path, "on-policy evidence round manifest")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(f"{name} must be an object.")
    return value


def load_lfm2_on_policy_evidence_round(
    path: str | os.PathLike[str],
    *,
    verify_extension_cache: bool = False,
) -> Lfm2OnPolicyEvidenceRound:
    """Load one sealed round manifest and re-authenticate its extension cache.

    The loader is fail-closed: a noncanonical document, a broken self-seal, an
    unexpected schema, a missing extension cache, or an extension cache whose
    sealed manifest no longer matches the round is an error, never a warning.
    """

    resolved = _manifest_path(path)
    document = _read_sealed_json(resolved, "on-policy evidence round manifest")
    if document.get("schema") != ROUND_MANIFEST_SCHEMA:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            f"Unsupported on-policy evidence round schema: "
            f"{document.get('schema')!r}."
        )
    _verify_seal(document, "manifest_sha256", "on-policy evidence round manifest")
    _sha256(document.get("request_sha256"), "request_sha256")

    round_body = _require_mapping(document.get("round"), "round")
    _validate_round_budget(round_body.get("index"), round_body.get("max_rounds"))
    extension = _require_mapping(document.get("extension_cache"), "extension_cache")
    base = _require_mapping(document.get("base_cache"), "base_cache")
    parent = _require_mapping(document.get("parent"), "parent")
    _sha256(parent.get("identity_sha256"), "parent identity_sha256")

    round_directory = resolved.parent
    cache_path = Path(str(extension.get("path"))).expanduser().absolute()
    if not cache_path.is_dir() or _is_link(cache_path):
        # Round directories are relocatable; fall back to the sealed relative
        # location before failing.
        cache_path = (
            round_directory
            / EXTENSION_CACHE_DIRNAME
            / str(extension.get("cache_key"))
        )
    if not cache_path.is_dir() or _is_link(cache_path):
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            f"Sealed extension cache is missing: {cache_path}."
        )
    cache = open_lfm2_circuit_trace_cache(cache_path, verify=verify_extension_cache)
    if cache.cache_key != extension.get("cache_key"):
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "Sealed extension cache key differs from the round manifest."
        )
    if _json_sha256(cache.manifest) != extension.get("manifest_sha256"):
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "Sealed extension cache manifest differs from the round manifest."
        )
    if cache.parent_fingerprint != parent.get("identity_sha256"):
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "Sealed extension cache belongs to a different parent."
        )
    if cache.batch_count != extension.get("batch_count"):
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "Sealed extension cache batch count differs from the round manifest."
        )
    counters = _require_mapping(document.get("counters"), "counters")
    for name in (
        "optimizer_steps",
        "backward_calls",
        "training_steps",
        "parent_forward_calls_during_child_rollout",
        "network_calls",
    ):
        if counters.get(name) != 0:
            raise Lfm2OnPolicyEvidenceRoundIntegrityError(
                f"Sealed round counter {name} must be zero."
            )
    if counters.get("optimizer_constructed") is not False:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "Sealed round must prove no optimizer was constructed."
        )
    if counters.get("parent_capture_invocations") != 1:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "Sealed round must record exactly one parent capture invocation."
        )
    if counters.get("parent_capture_passes_per_batch") != 1:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "Sealed round must record exactly one parent pass per batch."
        )
    if counters.get("parent_capture_batches") != cache.batch_count:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "Sealed round parent batch accounting differs from the cache."
        )
    disjointness = _require_mapping(document.get("disjointness"), "disjointness")
    if disjointness.get("verified_disjoint") is not True or disjointness.get(
        "overlapping_prompt_count"
    ) != 0:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "Sealed round does not carry a manual-review disjointness proof."
        )
    # Additive and optional: a round captured without a bridge carries no
    # target preimage and loads exactly as it always has.  A bridged round's
    # preimage is authenticated against the extension cache it claims to be
    # the preimage of, here, before any consumer sees it.
    preimage = document.get("target_preimage")
    if preimage is not None:
        _verify_sealed_target_preimage(
            _require_mapping(preimage, "target_preimage"),
            cache=cache,
            name="on-policy evidence round",
        )
        request = _require_mapping(document.get("request"), "request")
        if request.get("tokenizer_bridge_artifact_sha256") != preimage.get(
            "bridge_artifact_sha256"
        ) or request.get("tokenizer_bridge_profile_sha256") != preimage.get(
            "transport_profile_sha256"
        ):
            raise Lfm2OnPolicyEvidenceRoundBridgeError(
                "Sealed round request and target preimage name different "
                "strict tokenizer bridges."
            )
    return Lfm2OnPolicyEvidenceRound(
        manifest_path=resolved,
        round_directory=round_directory,
        base_cache_path=Path(str(base.get("path"))).expanduser().absolute(),
        extension_cache_path=cache.path,
        manifest=_deep_freeze(document),
    )


def resolve_lfm2_on_policy_evidence_references(
    base_cache_directory: str | os.PathLike[str],
    round_manifest_paths: Sequence[str | os.PathLike[str]] = (),
    *,
    expected_parent_fingerprint: str | None = None,
    verify_extension_caches: bool = False,
) -> Lfm2OnPolicyEvidenceReferences:
    """Return ``(base_cache_path, *extension_cache_paths)`` for one solve.

    Every extension round is authenticated, bound to the same parent as the
    sealed base cache, ordered by contiguous round index, and required to sit on
    a distinct path so a downstream solver cannot double-count one cache.
    """

    base_path = _local_directory(base_cache_directory, "base_cache_directory")
    base_cache = open_lfm2_circuit_trace_cache(base_path, verify=False)
    parent_fingerprint = base_cache.parent_fingerprint
    if expected_parent_fingerprint is not None and parent_fingerprint != _sha256(
        expected_parent_fingerprint, "expected_parent_fingerprint"
    ):
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "Sealed base cache belongs to a different parent."
        )
    if isinstance(round_manifest_paths, (str, bytes)) or not isinstance(
        round_manifest_paths, Sequence
    ):
        raise TypeError("round_manifest_paths must be a sequence of paths.")

    rounds = [
        load_lfm2_on_policy_evidence_round(
            value, verify_extension_cache=verify_extension_caches
        )
        for value in round_manifest_paths
    ]
    seen: set[Path] = {base_path}
    extensions: list[Path] = []
    for position, value in enumerate(rounds):
        if value.round_index != position:
            raise Lfm2OnPolicyEvidenceRoundBudgetError(
                f"On-policy evidence rounds must be contiguous from 0; round "
                f"at position {position} declares index {value.round_index}."
            )
        if value.parent_fingerprint != parent_fingerprint:
            raise Lfm2OnPolicyEvidenceRoundIntegrityError(
                f"Round {value.round_index} extends a different parent than the "
                "sealed base cache."
            )
        if value.max_rounds < len(rounds):
            raise Lfm2OnPolicyEvidenceRoundBudgetError(
                f"Round {value.round_index} declares max_rounds="
                f"{value.max_rounds} but {len(rounds)} rounds were supplied."
            )
        if value.extension_cache_path in seen:
            raise Lfm2OnPolicyEvidenceRoundIntegrityError(
                "On-policy evidence caches must occupy distinct paths."
            )
        seen.add(value.extension_cache_path)
        extensions.append(value.extension_cache_path)
    return Lfm2OnPolicyEvidenceReferences(
        base_cache_path=base_path,
        extension_cache_paths=tuple(extensions),
    )


@dataclass(frozen=True, slots=True)
class _PriorEvidence:
    """Every batch and corpus this round must not capture a second time."""

    summaries: tuple[dict[str, Any], ...]
    batch_input_source: Mapping[str, str]
    corpus_fingerprints: Mapping[str, str]


def _previous_round_evidence(
    values: Sequence[str | os.PathLike[str]],
    *,
    round_index: int,
    max_rounds: int,
    parent_fingerprint: str,
    base_cache: Lfm2CircuitTraceCache,
    base_manifest: Mapping[str, Any],
) -> _PriorEvidence:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("previous_round_manifest_paths must be a sequence of paths.")
    if values and len(values) != round_index:
        raise Lfm2OnPolicyEvidenceRoundBudgetError(
            f"round_index={round_index} requires exactly {round_index} previous "
            f"round manifests; {len(values)} were supplied."
        )
    batch_input_source: dict[str, str] = {
        str(value): "sealed-base-cache"
        for value in base_manifest["corpus"]["ordered_batch_input_sha256"]
    }
    corpus_fingerprints: dict[str, str] = {
        base_cache.corpus_fingerprint: "sealed-base-cache"
    }
    summaries: list[dict[str, Any]] = []
    for position, value in enumerate(values):
        previous = load_lfm2_on_policy_evidence_round(value)
        if previous.round_index != position:
            raise Lfm2OnPolicyEvidenceRoundBudgetError(
                f"Previous round at position {position} declares index "
                f"{previous.round_index}."
            )
        if previous.max_rounds != max_rounds:
            raise Lfm2OnPolicyEvidenceRoundBudgetError(
                f"Previous round {position} declares max_rounds="
                f"{previous.max_rounds}; this round declares {max_rounds}."
            )
        if previous.parent_fingerprint != parent_fingerprint:
            raise Lfm2OnPolicyEvidenceRoundIntegrityError(
                f"Previous round {position} extends a different parent."
            )
        extension = previous.manifest["extension_cache"]
        label = f"sealed-round-{previous.round_index:03d}"
        for digest in extension["ordered_batch_input_sha256"]:
            batch_input_source.setdefault(str(digest), label)
        corpus_fingerprints.setdefault(str(extension["corpus_sha256"]), label)
        summaries.append(
            {
                "index": previous.round_index,
                "manifest_path": str(previous.manifest_path),
                "manifest_sha256": previous.manifest_sha256,
                "extension_cache_key": previous.extension_cache_key,
                "extension_corpus_sha256": str(extension["corpus_sha256"]),
                "extension_batch_count": int(extension["batch_count"]),
            }
        )
    return _PriorEvidence(
        summaries=tuple(summaries),
        batch_input_source=MappingProxyType(batch_input_source),
        corpus_fingerprints=MappingProxyType(corpus_fingerprints),
    )


def run_lfm2_on_policy_evidence_round(
    *,
    child_artifact_directory: str | os.PathLike[str],
    parent_model: Any,
    parent_tokenizer: Any,
    parent_fingerprint: str,
    fit_prompts: Sequence[str],
    manual_prompt_manifest_path: str | os.PathLike[str],
    base_cache_directory: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    capture_config: Lfm2CircuitTraceCaptureConfig,
    round_index: int = 0,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    rollout: Lfm2OnPolicyRolloutConfig | None = None,
    previous_round_manifest_paths: Sequence[str | os.PathLike[str]] = (),
    child_device: str = "cpu",
    child_dtype: str = "float32",
    input_device: Any | None = None,
    parent_tokenizer_name: str = "parent",
    require_matching_capture_selection: bool = True,
    verify_base_cache: bool = False,
    tokenizer_bridge: Any | None = None,
    expected_tokenizer_bridge_artifact_sha256: str | None = None,
) -> Lfm2OnPolicyEvidenceRound:
    """Run one bounded, sealed, zero-gradient on-policy evidence extension round.

    The child is freshly reloaded from ``child_artifact_directory`` and is the
    only model executed while rollouts are collected.  The parent is executed
    exactly once per surviving capture batch, inside the single sanctioned
    ``build_lfm2_circuit_trace_cache`` invocation that seals the round's
    extension cache.  No optimizer is constructed, no gradient is produced, and
    neither model is mutated.

    With ``tokenizer_bridge`` the round is captured for a bridged lane: the
    reloaded child's own tokenizer is proven to be the sealed bridge target,
    every capture batch is the one-row donor image of the child's own target
    tokenization, and the manifest seals that target preimage next to the
    parent-facing batch record.  Without it nothing about this function's
    behavior or sealed output changes.
    """

    torch = _torch()
    bridge = (
        None
        if tokenizer_bridge is None
        else _verified_round_bridge(
            tokenizer_bridge,
            expected_tokenizer_bridge_artifact_sha256,
        )
    )
    index, bound = _validate_round_budget(round_index, max_rounds)
    selected = Lfm2OnPolicyRolloutConfig() if rollout is None else rollout
    if not isinstance(selected, Lfm2OnPolicyRolloutConfig):
        raise TypeError("rollout must be Lfm2OnPolicyRolloutConfig or None.")
    if not isinstance(capture_config, Lfm2CircuitTraceCaptureConfig):
        raise TypeError("capture_config must be Lfm2CircuitTraceCaptureConfig.")
    if not isinstance(parent_tokenizer_name, str) or not parent_tokenizer_name.strip():
        raise ValueError("parent_tokenizer_name must be a nonempty string.")
    parent_tokenizer_name = parent_tokenizer_name.strip()
    if parent_tokenizer_name == "child":
        raise ValueError("parent_tokenizer_name cannot use the reserved name 'child'.")
    expected_parent = _sha256(parent_fingerprint, "parent_fingerprint")

    prompts = _normalized_fit_prompts(fit_prompts, selected.max_prompts)
    manual = load_manual_review_prompt_manifest(manual_prompt_manifest_path)
    disjointness = _verified_disjointness(prompts, manual)
    prompt_corpus = _prompt_corpus_evidence(prompts)

    child_root = _local_directory(child_artifact_directory, "child_artifact_directory")
    child_records = _artifact_inventory(child_root, "child artifact")
    child_artifact_sha256 = _artifact_sha256(child_records)

    base_path = _local_directory(base_cache_directory, "base_cache_directory")
    base_cache = open_lfm2_circuit_trace_cache(base_path, verify=verify_base_cache)
    if base_cache.parent_fingerprint != expected_parent:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "The sealed base cache belongs to a different parent than "
            "parent_fingerprint."
        )
    base_manifest = base_cache.manifest
    base_capture_selection = str(base_manifest["capture"]["selection_sha256"])
    capture_selection = str(capture_config.as_dict()["selection_sha256"])
    if require_matching_capture_selection and capture_selection != base_capture_selection:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "Extension evidence must be captured under the same selection "
            "policy as the sealed base cache so a closed-form writer can "
            "consume base and extension evidence together."
        )
    prior = _previous_round_evidence(
        previous_round_manifest_paths,
        round_index=index,
        max_rounds=bound,
        parent_fingerprint=expected_parent,
        base_cache=base_cache,
        base_manifest=base_manifest,
    )

    request = {
        "schema": ROUND_REQUEST_SCHEMA,
        "round_index": index,
        "max_rounds": bound,
        "child_artifact_sha256": child_artifact_sha256,
        "parent_identity_sha256": expected_parent,
        "prompt_corpus_sha256": prompt_corpus["prompt_corpus_sha256"],
        "manual_manifest_sha256": manual.manifest_sha256,
        "manual_prompt_digest_set_sha256": manual.digest_set_sha256,
        "rollout_sha256": selected.as_dict()["rollout_sha256"],
        "capture_selection_sha256": capture_selection,
        "base_cache_key": base_cache.cache_key,
        "parent_tokenizer_name": parent_tokenizer_name,
        "child_device": str(child_device),
        "child_dtype": str(child_dtype),
    }
    if bridge is not None:
        # Only a bridged request carries these keys, so an already sealed
        # unbridged round keeps its exact request digest and round directory.
        request = {
            **request,
            "tokenizer_bridge_artifact_sha256": bridge.artifact_sha256,
            "tokenizer_bridge_profile_sha256": bridge.profile_fingerprint,
            "target_tokenizer_semantics_sha256": (
                bridge.target_tokenizer_semantics_sha256
            ),
            "parent_cache_input_mode": BRIDGE_MAPPED_CAPTURE_INPUT_MODE,
        }
    request_sha256 = _json_sha256(request)

    output_path = Path(output_root).expanduser().absolute()
    round_directory = _round_directory(output_path, index)
    manifest_path = round_directory / ROUND_MANIFEST_FILENAME
    if manifest_path.exists():
        existing = load_lfm2_on_policy_evidence_round(manifest_path)
        if existing.request_sha256 != request_sha256:
            raise Lfm2OnPolicyEvidenceRoundConflictError(
                f"A different sealed on-policy round already occupies "
                f"{round_directory}."
            )
        return existing
    round_directory.mkdir(parents=True, exist_ok=True)

    child_model = load_lfm2_direct_local_model(
        child_root,
        device=child_device,
        dtype=_torch_dtype(torch, child_dtype),
    )
    if child_model is parent_model:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "The reloaded child must be a standalone model distinct from the parent."
        )
    child_tokenizer = load_lfm2_direct_local_tokenizer(child_root)
    child_bridge_reference: dict[str, Any] | None = None
    if bridge is not None:
        child_bridge_reference = _child_bridge_reference(child_root, bridge)
        try:
            # Proves in one call that the supplied parent tokenizer is the
            # sealed donor, that the reloaded child's own tokenizer is the
            # sealed target, and that the live pair still reproduces the
            # sealed one-row transport profile.
            bridge.verify_live_tokenizers(parent_tokenizer, child_tokenizer)
        except StrictTokenizerBridgeError as exc:
            raise Lfm2OnPolicyEvidenceRoundBridgeError(
                "The reloaded child tokenizer or the supplied parent "
                "tokenizer is not the sealed strict-bridge pair."
            ) from exc
    child_state_before = fingerprint_lfm2_direct_local_state(child_model)
    child_gradients_before = _gradient_state(torch, child_model)
    parent_before = fingerprint_lfm2_parent(parent_model)
    if parent_before.identity_sha256 != expected_parent:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "The supplied parent model does not match parent_fingerprint."
        )
    parent_gradients_before = _gradient_state(torch, parent_model)

    generation_cache_path = round_directory / GENERATION_CACHE_FILENAME
    parent_counter, parent_handles = _install_forward_counter(parent_model)
    with torch.no_grad():
        try:
            corpus = build_on_policy_probe_corpus(
                child_model,
                child_tokenizer,
                {parent_tokenizer_name: parent_tokenizer},
                prompts,
                settings=selected.probe_settings(),
                generation_cache_path=generation_cache_path,
                torch_module=torch,
            )
        finally:
            for handle in parent_handles:
                handle.remove()
        rollout_parent_forwards = int(parent_counter["calls"])
        if rollout_parent_forwards:
            raise Lfm2OnPolicyEvidenceRoundIntegrityError(
                f"The parent executed {rollout_parent_forwards} forwards during "
                "the child-only rollout phase."
            )
        probe_report = corpus.report
        if bool(probe_report["execution"]["parent_forward_calls"]):
            raise Lfm2OnPolicyEvidenceRoundIntegrityError(
                "The probe reported parent forwards during child generation."
            )

        candidates, rejected = _capture_batches_from_corpus(
            corpus,
            # A bridged round reads the child's own rows; the parent-facing
            # batch is their one-row donor image, which is the same parent
            # input mode the bridged base cache was captured under.
            parent_tokenizer_name=(
                parent_tokenizer_name if bridge is None else "child"
            ),
            max_decode_steps=capture_config.max_decode_steps,
            truncate=selected.truncate_forced_decode_to_capture_budget,
            role="parent" if bridge is None else "target",
            target_to_donor_row_indices=(
                None
                if bridge is None
                else tuple(
                    bridge.transport_profile.target_to_donor_row_indices
                )
            ),
        )
        if not candidates:
            raise Lfm2OnPolicyEvidenceRoundError(
                "The child rollout produced no usable capture batch; "
                f"rejections: {rejected}."
            )
        if len(candidates) > selected.max_capture_batches:
            raise Lfm2OnPolicyEvidenceRoundBudgetError(
                f"{len(candidates)} on-policy capture batches exceed the bounded "
                f"budget max_capture_batches={selected.max_capture_batches}."
            )

        # Evidence already sealed in the base cache or in a previous round is
        # never regenerated; that remains a closed path.
        model_batches = [batch.as_model_batch(torch) for batch in candidates]
        prepared = _prepare_lfm2_trace_batches(torch, model_batches, capture_config)
        selected_batches: list[_CaptureBatch] = []
        selected_models: list[dict[str, Any]] = []
        selected_inputs: list[str] = []
        already_sealed: list[dict[str, Any]] = []
        for batch, model_batch, prepared_batch in zip(
            candidates, model_batches, prepared, strict=True
        ):
            source = prior.batch_input_source.get(prepared_batch.input_sha256)
            if source is not None:
                already_sealed.append(
                    {
                        "trajectory_sha256": batch.trajectory_sha256,
                        "input_sha256": prepared_batch.input_sha256,
                        "sealed_in": source,
                    }
                )
                continue
            selected_batches.append(batch)
            selected_models.append(model_batch)
            selected_inputs.append(prepared_batch.input_sha256)
        if not selected_batches:
            raise Lfm2OnPolicyEvidenceRoundError(
                "Every child-visited trajectory is already sealed in the base "
                "cache or a previous round; regenerating that evidence is a "
                "closed path."
            )
        if len(set(selected_inputs)) != len(selected_inputs):
            raise Lfm2OnPolicyEvidenceRoundIntegrityError(
                "On-policy capture batches are not content-distinct."
            )

        aligned_records: tuple[Any, ...] = ()
        adapter_evidence_sha256: str | None = None
        if bridge is not None:
            try:
                adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
                    bridge,
                    _target_preimage_records(
                        selected_batches, round_index=index
                    ),
                    expected_artifact_sha256=bridge.artifact_sha256,
                )
            except Lfm2StrictTokenizerBridgeTraceError as exc:
                raise Lfm2OnPolicyEvidenceRoundBridgeError(
                    "The child's target tokenization does not form a strict "
                    "one-row bridge record for this round."
                ) from exc
            aligned_records = adapter.records
            adapter_evidence_sha256 = adapter.evidence_sha256
            # The batch the parent is about to see must be exactly the bridge
            # image of the sealed target preimage, tensor for tensor.
            for position, record in enumerate(aligned_records):
                expected = record.as_parent_cache_batch()
                observed = selected_models[position]
                if set(expected) != set(observed) or any(
                    not torch.equal(expected[name], observed[name])
                    for name in sorted(expected)
                ):
                    raise Lfm2OnPolicyEvidenceRoundBridgeError(
                        f"On-policy capture batch {position} is not the "
                        "one-row donor image of its target preimage."
                    )

        extension_root = round_directory / EXTENSION_CACHE_DIRNAME
        extension_cache = build_lfm2_circuit_trace_cache(
            extension_root,
            parent_model,
            selected_models,
            config=capture_config,
            input_device=input_device,
        )

    parent_after = fingerprint_lfm2_parent(parent_model)
    if parent_after != parent_before:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "The parent changed during the on-policy evidence round."
        )
    if _gradient_state(torch, parent_model) != parent_gradients_before:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "Parent gradients or requires_grad flags changed during the round."
        )
    child_state_after = fingerprint_lfm2_direct_local_state(child_model)
    if child_state_after != child_state_before:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "The child changed during the on-policy evidence round."
        )
    if _gradient_state(torch, child_model) != child_gradients_before:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "Child gradients or requires_grad flags changed during the round."
        )
    if _artifact_inventory(child_root, "child artifact") != child_records:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "The published child artifact changed during the round."
        )

    extension_manifest = extension_cache.manifest
    extension_inputs = [
        str(value)
        for value in extension_manifest["corpus"]["ordered_batch_input_sha256"]
    ]
    if extension_inputs != selected_inputs:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "The sealed extension cache did not capture exactly the requested "
            "child-visited batches."
        )
    if extension_cache.batch_count != len(selected_batches):
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "The sealed extension cache batch count differs from the request."
        )
    if extension_cache.parent_fingerprint != expected_parent:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            "The sealed extension cache belongs to a different parent."
        )
    duplicated = prior.corpus_fingerprints.get(extension_cache.corpus_fingerprint)
    if duplicated is not None:
        raise Lfm2OnPolicyEvidenceRoundIntegrityError(
            f"The extension corpus duplicates {duplicated}; regenerating "
            "already-sealed evidence is a closed path."
        )
    for name in ("implementation", "layer_topology", "execution"):
        key = f"{name}_sha256" if name != "layer_topology" else "layer_topology_sha256"
        if extension_manifest[name][key] != base_manifest[name][key]:
            raise Lfm2OnPolicyEvidenceRoundIntegrityError(
                f"Extension and base caches disagree on {name}; their evidence "
                "cannot be consumed by one closed-form solve."
            )

    body = {
        "schema": ROUND_MANIFEST_SCHEMA,
        "request": request,
        "request_sha256": request_sha256,
        "round": {
            "index": index,
            "max_rounds": bound,
            "previous_rounds": [dict(value) for value in prior.summaries],
        },
        "child": {
            "artifact_root": str(child_root),
            "artifact_sha256": child_artifact_sha256,
            "file_records": [dict(record) for record in child_records],
            "state_sha256": child_state_after,
            "device": str(child_device),
            "dtype": str(child_dtype),
            "freshly_reloaded_from_published_artifact": True,
            "standalone_model_distinct_from_parent": True,
            "unchanged_by_round": True,
        },
        "parent": {
            **parent_before.as_dict(),
            "unchanged_by_round": True,
            "executed_only_inside_sealed_capture": True,
        },
        "prompt_corpus": prompt_corpus,
        "disjointness": disjointness,
        "rollout": {
            **selected.as_dict(),
            "probe_report_sha256": str(probe_report["report_sha256"]),
            "probe_corpus_sha256": str(probe_report["corpus_sha256"]),
            "generation_cache_path": str(generation_cache_path),
            "generation_cache_sha256": str(
                probe_report["generation_cache"]["cache_sha256"]
            ),
            "generation_cache_hit": bool(probe_report["generation_cache"]["hit"]),
            "child_generate_calls": int(
                probe_report["generation_cache"]["generation_calls"]
            ),
            "visited_prefix_row_count": len(corpus.rows),
            "visited_prefix_origin_count": int(
                probe_report["alignment"]["origin_count"]
            ),
            "parent_tokenizer_name": parent_tokenizer_name,
            "rejected_prompts": [dict(value) for value in rejected],
        },
        "capture": {
            "config": capture_config.as_dict(),
            "candidate_batch_count": len(candidates),
            "prior_sealed_batch_input_count": len(prior.batch_input_source),
            "already_sealed_in_prior_evidence": already_sealed,
            "batch_count": len(selected_batches),
            "batches": [
                {
                    "index": position,
                    "input_sha256": selected_inputs[position],
                    "trajectory_sha256": batch.trajectory_sha256,
                    "prompt_token_count": len(batch.prompt_token_ids),
                    "forced_decode_token_count": len(batch.forced_decode_token_ids),
                    "origin_fit_prompt_indices": list(batch.origin_prompt_indices),
                    "visited_prefix_row_ids": list(batch.visited_prefix_row_ids),
                    "forced_decode_truncated": batch.truncated,
                }
                for position, batch in enumerate(selected_batches)
            ],
        },
        "base_cache": {
            "path": str(base_path),
            "cache_key": base_cache.cache_key,
            "corpus_sha256": base_cache.corpus_fingerprint,
            "parent_identity_sha256": base_cache.parent_fingerprint,
            "batch_count": base_cache.batch_count,
            "fully_verified_on_open": bool(verify_base_cache),
            "regenerated": False,
        },
        "extension_cache": {
            "root": str(extension_root),
            "path": str(extension_cache.path),
            "cache_key": extension_cache.cache_key,
            "manifest_sha256": _json_sha256(extension_manifest),
            "corpus_sha256": extension_cache.corpus_fingerprint,
            "parent_identity_sha256": extension_cache.parent_fingerprint,
            "implementation_sha256": extension_cache.implementation_fingerprint,
            "layer_topology_sha256": extension_cache.layer_topology_fingerprint,
            "batch_count": extension_cache.batch_count,
            "ordered_batch_input_sha256": list(extension_inputs),
            "disjoint_from_base_corpus": True,
        },
        "counters": {
            "parent_capture_invocations": 1,
            "parent_capture_batches": len(selected_batches),
            "parent_capture_passes_per_batch": 1,
            "parent_forward_calls_during_child_rollout": rollout_parent_forwards,
            "child_generate_calls": int(
                probe_report["generation_cache"]["generation_calls"]
            ),
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "training_steps": 0,
            "network_calls": 0,
            "grad_enabled_during_round": False,
            "parent_gradient_state_unchanged": True,
            "child_gradient_state_unchanged": True,
            "parent_state_unchanged": True,
            "child_state_unchanged": True,
            "child_artifact_unchanged": True,
        },
    }
    if bridge is not None:
        assert child_bridge_reference is not None
        assert adapter_evidence_sha256 is not None
        body = {
            **body,
            "target_preimage": _sealed_target_preimage(
                round_index=index,
                bridge=bridge,
                batches=selected_batches,
                aligned_records=aligned_records,
                adapter_evidence_sha256=adapter_evidence_sha256,
                donor_input_sha256=extension_inputs,
                child_reference=child_bridge_reference,
                parent_tokenizer_name=parent_tokenizer_name,
                parent_preimages=_parent_tokenizer_preimages(
                    corpus, parent_tokenizer_name, selected_batches
                ),
            ),
        }
    document = _sealed_document(body, "manifest_sha256")
    _write_sealed_json(manifest_path, document)
    return load_lfm2_on_policy_evidence_round(manifest_path)


def open_lfm2_on_policy_evidence_caches(
    references: Lfm2OnPolicyEvidenceReferences,
    *,
    verify: bool = False,
) -> tuple[Lfm2CircuitTraceCache, ...]:
    """Open the base plus every extension cache as one ordered evidence set."""

    return tuple(
        open_lfm2_circuit_trace_cache(path, verify=verify)
        for path in references.as_tuple()
    )


def lfm2_on_policy_round_missing_target_preimage_reason(
    round_value: Lfm2OnPolicyEvidenceRound,
) -> str | None:
    """Name why this round cannot be replayed by a bridged child, if it cannot."""

    if round_value.carries_target_preimage:
        return None
    return (
        f"on-policy round manifest {round_value.manifest_path} seals no "
        "target-token preimage: it was captured parent-canonically, so a "
        "bridged child (which speaks the target vocabulary) has no replay "
        "for its extension batches"
    )


def build_lfm2_on_policy_round_target_records(
    round_value: Lfm2OnPolicyEvidenceRound,
    *,
    bridge: Any,
    extension_cache: Lfm2CircuitTraceCache,
    expected_artifact_sha256: str | None = None,
) -> tuple[Lfm2TargetTokenizedRecord, ...]:
    """Rebuild one round's target-token replay records, fully authenticated.

    The proof is not "the manifest says so": every sealed target preimage is
    mapped through the **live** transport profile and the resulting donor batch
    digest is required to equal the digest the sealed extension cache actually
    holds at that index.  A tampered preimage therefore fails here even when
    its own seals were recomputed.
    """

    if not isinstance(round_value, Lfm2OnPolicyEvidenceRound):
        raise TypeError("round_value must be Lfm2OnPolicyEvidenceRound.")
    if not isinstance(extension_cache, Lfm2CircuitTraceCache):
        raise TypeError("extension_cache must be Lfm2CircuitTraceCache.")
    # Diagnosed before the bridge, because "this round has no preimage" is the
    # answer the consumption gate has to be able to state by name.
    if round_value.target_preimage is None:
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            lfm2_on_policy_round_missing_target_preimage_reason(round_value)
        )
    verified = _verified_round_bridge(bridge, expected_artifact_sha256)
    # The loaded manifest is deep-frozen; canonical JSON needs the plain
    # mapping/list shape the seal was computed over.
    block = _deep_thaw(round_value.target_preimage)
    if extension_cache.cache_key != round_value.extension_cache_key:
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            f"Sealed extension cache {extension_cache.cache_key} is not the "
            f"one round {round_value.manifest_path} sealed."
        )
    _verify_sealed_target_preimage(
        block,
        cache=extension_cache,
        name=f"on-policy round {round_value.manifest_path}",
    )
    live = _bridge_reference_body(verified)
    for name in (
        "bridge_artifact_sha256",
        "transport_profile_sha256",
        "target_to_donor_row_indices_sha256",
        "target_row_capacity",
        "donor_row_capacity",
        "donor_tokenizer_semantics_sha256",
        "target_tokenizer_semantics_sha256",
    ):
        if block.get(name) != live[name]:
            raise Lfm2OnPolicyEvidenceRoundBridgeError(
                f"Round {round_value.manifest_path} sealed its target preimage "
                f"under a different strict tokenizer bridge ({name} differs)."
            )
    torch = _torch()
    records = []
    for entry in block["records"]:
        input_ids = torch.tensor(
            [[int(value) for value in entry["target_input_ids"]]],
            dtype=torch.long,
        )
        records.append(
            Lfm2TargetTokenizedRecord(
                record_id=str(entry["record_id"]),
                split=str(entry["split"]),
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                forced_decode_ids=torch.tensor(
                    [
                        [
                            int(value)
                            for value in entry["target_forced_decode_ids"]
                        ]
                    ],
                    dtype=torch.long,
                ),
            )
        )
    try:
        adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
            verified,
            records,
            expected_artifact_sha256=verified.artifact_sha256,
        )
    except Lfm2StrictTokenizerBridgeTraceError as exc:
        raise Lfm2OnPolicyEvidenceRoundBridgeError(
            f"Round {round_value.manifest_path} target preimage does not form "
            "strict one-row bridge records under the live transport profile."
        ) from exc
    donor_inputs = [
        str(value)
        for value in extension_cache.manifest["corpus"][
            "ordered_batch_input_sha256"
        ]
    ]
    for position, (entry, record) in enumerate(
        zip(block["records"], adapter.records, strict=True)
    ):
        if (
            record.donor_cache_input_sha256 != donor_inputs[position]
            or record.donor_cache_input_sha256
            != entry["donor_cache_input_sha256"]
            or record.target_cache_input_sha256
            != entry["target_cache_input_sha256"]
            or record.alignment_sha256 != entry["alignment_sha256"]
        ):
            raise Lfm2OnPolicyEvidenceRoundBridgeError(
                f"Round {round_value.manifest_path} target preimage "
                f"{position} does not map onto sealed extension batch "
                f"{position}; the sealed parent evidence was captured on a "
                "different donor input."
            )
    return tuple(records)


__all__ = [
    "BRIDGE_MAPPED_CAPTURE_INPUT_MODE",
    "CHILD_TOKENIZER_BRIDGE_REFERENCE_FILENAME",
    "DEFAULT_MAX_ROUNDS",
    "EXTENSION_CACHE_DIRNAME",
    "GENERATION_CACHE_FILENAME",
    "MANUAL_PROMPT_MANIFEST_SCHEMA",
    "PARENT_CANONICAL_CAPTURE_INPUT_MODE",
    "ROUND_MANIFEST_FILENAME",
    "ROUND_MANIFEST_SCHEMA",
    "ROUND_REQUEST_SCHEMA",
    "ROUND_TARGET_PREIMAGE_SCHEMA",
    "Lfm2ManualReviewPromptManifest",
    "Lfm2OnPolicyEvidenceReferences",
    "Lfm2OnPolicyEvidenceRound",
    "Lfm2OnPolicyEvidenceRoundBridgeError",
    "Lfm2OnPolicyEvidenceRoundBudgetError",
    "Lfm2OnPolicyEvidenceRoundConflictError",
    "Lfm2OnPolicyEvidenceRoundDisjointnessError",
    "Lfm2OnPolicyEvidenceRoundError",
    "Lfm2OnPolicyEvidenceRoundIntegrityError",
    "Lfm2OnPolicyRolloutConfig",
    "build_lfm2_on_policy_round_target_records",
    "lfm2_on_policy_round_missing_target_preimage_reason",
    "load_lfm2_on_policy_evidence_round",
    "load_manual_review_prompt_manifest",
    "open_lfm2_on_policy_evidence_caches",
    "resolve_lfm2_on_policy_evidence_references",
    "run_lfm2_on_policy_evidence_round",
]
