#!/usr/bin/env python3
"""Compile a sealed LFM2 parent cache into a severe dense blank child.

This is the production CPCF orchestration path.  It accepts only explicit
local model, tokenizer, bridge, corpus, and sealed-cache paths.  The parent is
loaded to expose its immutable parameters, but it is never executed or asked
to generate.  Parent causal evidence is streamed from the already completed
cache exactly once into one resident phase collection, then reused for:

* a target-native tied vocabulary interface;
* independent predictive residual coordinates at every macrocell boundary;
* sequential operator and MLP compilation on actual folded-upstream states;
* a target-coordinate final RMSNorm; and
* standalone whole-response generation from a freshly reloaded child.

The runner has no optimizer, backward pass, training loop, quantization,
pruning, parent-backed fallback, hub loader, or cache-generation path.
Structural dry-plan mode is deliberately non-behavioral: it may prove a fresh
dense blank's exact parameter count, but cannot claim that a child was folded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


# Direct ``python experiments/<script>.py`` execution puts only the
# experiments directory on sys.path.  Add the canonical repository root so
# sibling experiment helpers remain importable without requiring PYTHONPATH
# or a module-style invocation.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


# Set the offline contract before importing any Transformers-backed module.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


RUN_SCHEMA = "neural-forge-lfm2-cpcf-blank-fold-run-v1"
DRY_PLAN_SCHEMA = "neural-forge-lfm2-cpcf-blank-dry-plan-v1"
TRAJECTORY_SCHEMA = "neural-forge-lfm2-bridge-trajectory-text-v1"
STANDALONE_EVALUATION_SCHEMA = (
    "neural-forge-lfm2-cpcf-standalone-whole-response-evaluation-v1"
)
MINIMUM_CLAIMABLE_COMPRESSION_RATIO = Decimal("100")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DTYPES = ("bfloat16", "float16", "float32")


class Lfm2CpcfBlankFoldRunnerError(RuntimeError):
    """The strict local CPCF run cannot be completed honestly."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Runner evidence must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_directory(value: Path, name: str) -> Path:
    path = value.expanduser().resolve(strict=True)
    if not path.is_dir() or path.is_symlink():
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must be an explicit nonsymlink local directory: {path}."
        )
    return path


def _local_file(value: Path, name: str) -> Path:
    path = value.expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must be an explicit nonsymlink local file: {path}."
        )
    return path


def _new_output_directory(value: Path) -> Path:
    path = value.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = path.parent.resolve(strict=True)
    result = resolved_parent / path.name
    if result.exists() or result.is_symlink():
        raise Lfm2CpcfBlankFoldRunnerError(
            f"Output directory already exists; refusing to overwrite it: {result}."
        )
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise Lfm2CpcfBlankFoldRunnerError(
            f"Receipt already exists; refusing to overwrite it: {destination}."
        )
    temporary = destination.with_name(
        f".{destination.name}.stage-{uuid.uuid4().hex}"
    )
    payload = _canonical_json(dict(value)) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must be a positive integer."
        )
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must be finite and nonnegative."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must be finite and nonnegative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must be finite and nonnegative."
        )
    return result


@dataclass(frozen=True, slots=True)
class CompressionRequest:
    """One unambiguous parameter-count target.

    Generic diagnostic controls may request any real compaction.  The runner's
    separate promotion gate decides whether a result may represent the current
    100x project objective.
    """

    compression_ratio: str | None
    target_unique_parameters: int | None

    def __post_init__(self) -> None:
        supplied = (
            self.compression_ratio is not None,
            self.target_unique_parameters is not None,
        )
        if sum(supplied) != 1:
            raise Lfm2CpcfBlankFoldRunnerError(
                "Supply exactly one compression ratio or target parameter count."
            )
        if self.compression_ratio is not None:
            try:
                ratio = Decimal(self.compression_ratio)
            except (InvalidOperation, ValueError) as exc:
                raise Lfm2CpcfBlankFoldRunnerError(
                    "compression_ratio must be a finite decimal."
                ) from exc
            if not ratio.is_finite() or ratio <= Decimal("1"):
                raise Lfm2CpcfBlankFoldRunnerError(
                    "compression_ratio must be finite and greater than one."
                )
            object.__setattr__(self, "compression_ratio", str(ratio))
        if self.target_unique_parameters is not None:
            _positive_int(
                self.target_unique_parameters,
                "target_unique_parameters",
            )

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "compression_ratio": self.compression_ratio,
            "target_unique_parameters": self.target_unique_parameters,
        }


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    parent_model_directory: Path
    donor_tokenizer_directory: Path
    target_tokenizer_directory: Path
    bridge_directory: Path
    trace_cache_directory: Path
    trajectory_corpus: Path
    output_directory: Path
    receipt: Path | None
    compression: CompressionRequest
    device: str
    dtype: str
    parent_name: str
    child_name: str
    blank_seed: int
    minimum_hidden_size: int
    maximum_candidates: int
    selected_candidate_rank: int | None
    ridge: float
    boundary_state_reconstruction_weight: float
    boundary_predictive_energy_floor: float
    boundary_validation_retention_floor: float
    evaluation_limit: int
    evaluation_max_new_tokens: int
    structural_dry_plan: bool
    diagnostic_control: bool

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise Lfm2CpcfBlankFoldRunnerError(
                f"dtype must be one of {_DTYPES!r}."
            )
        if not self.device.strip():
            raise Lfm2CpcfBlankFoldRunnerError("device must be nonempty.")
        if not self.parent_name.strip() or not self.child_name.strip():
            raise Lfm2CpcfBlankFoldRunnerError(
                "parent_name and child_name must be nonempty."
            )
        if isinstance(self.blank_seed, bool) or not isinstance(
            self.blank_seed, int
        ):
            raise Lfm2CpcfBlankFoldRunnerError(
                "blank_seed must be an integer."
            )
        for value, name in (
            (self.minimum_hidden_size, "minimum_hidden_size"),
            (self.maximum_candidates, "maximum_candidates"),
            (self.evaluation_limit, "evaluation_limit"),
            (
                self.evaluation_max_new_tokens,
                "evaluation_max_new_tokens",
            ),
        ):
            _positive_int(value, name)
        if self.selected_candidate_rank is not None and (
            isinstance(self.selected_candidate_rank, bool)
            or not isinstance(self.selected_candidate_rank, int)
            or self.selected_candidate_rank < 0
        ):
            raise Lfm2CpcfBlankFoldRunnerError(
                "selected_candidate_rank must be nonnegative."
            )
        _nonnegative_float(self.ridge, "ridge")
        _nonnegative_float(
            self.boundary_state_reconstruction_weight,
            "boundary_state_reconstruction_weight",
        )
        for value, name in (
            (
                self.boundary_predictive_energy_floor,
                "boundary_predictive_energy_floor",
            ),
            (
                self.boundary_validation_retention_floor,
                "boundary_validation_retention_floor",
            ),
        ):
            normalized = _nonnegative_float(value, name)
            if normalized > 1.0:
                raise Lfm2CpcfBlankFoldRunnerError(
                    f"{name} must be within [0, 1]."
                )
        if not isinstance(self.structural_dry_plan, bool):
            raise Lfm2CpcfBlankFoldRunnerError(
                "structural_dry_plan must be boolean."
            )
        if not isinstance(self.diagnostic_control, bool):
            raise Lfm2CpcfBlankFoldRunnerError(
                "diagnostic_control must be boolean."
            )


def _torch_dtype(torch: Any, name: str) -> Any:
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"Unsupported dtype {name!r}."
        ) from exc


def _read_trajectory_rows(path: Path) -> tuple[dict[str, str], ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Trajectory corpus must be valid UTF-8 JSON."
        ) from exc
    if not isinstance(value, list) or not value:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Trajectory corpus must be a nonempty JSON list."
        )
    result = []
    identifiers = set()
    splits = set()
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise Lfm2CpcfBlankFoldRunnerError(
                f"Trajectory row {index} is not an object."
            )
        record_id = row.get("id")
        split = row.get("split")
        input_text = row.get("input_text")
        forced_text = row.get("forced_decode_text")
        if (
            row.get("schema") != TRAJECTORY_SCHEMA
            or not isinstance(record_id, str)
            or not record_id
            or record_id in identifiers
            or split not in {"selection", "fit", "validation"}
            or not isinstance(input_text, str)
            or not input_text
            or not isinstance(forced_text, str)
            or not forced_text
        ):
            raise Lfm2CpcfBlankFoldRunnerError(
                f"Trajectory row {index} is malformed or duplicated."
            )
        identifiers.add(record_id)
        splits.add(str(split))
        result.append(
            {
                "id": record_id,
                "split": str(split),
                "input_text": input_text,
                "forced_decode_text": forced_text,
            }
        )
    if splits != {"selection", "fit", "validation"}:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Trajectory corpus must cover selection, fit, and validation."
        )
    return tuple(result)


def _token_ids(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool,
) -> tuple[int, ...]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
    )
    if not isinstance(encoded, Mapping):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Local tokenizer returned an invalid encoding."
        )
    values = encoded.get("input_ids")
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in values
        )
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Local tokenizer returned invalid token IDs."
        )
    return tuple(values)


def _trace_adapter(
    *,
    rows: Sequence[Mapping[str, str]],
    target_tokenizer: Any,
    bridge: Any,
    cache_manifest: Mapping[str, Any],
) -> Any:
    import torch

    from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (
        Lfm2TargetTokenizedRecord,
        build_lfm2_strict_tokenizer_bridge_trace_adapter,
    )

    capture = cache_manifest.get("capture")
    corpus = cache_manifest.get("corpus")
    if not isinstance(capture, Mapping) or not isinstance(corpus, Mapping):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Sealed cache lacks capture or corpus metadata."
        )
    max_decode_steps = _positive_int(
        capture.get("max_decode_steps"),
        "cache max_decode_steps",
    )
    records = []
    for row in rows:
        input_ids = _token_ids(
            target_tokenizer,
            row["input_text"],
            add_special_tokens=True,
        )
        forced_ids = _token_ids(
            target_tokenizer,
            row["forced_decode_text"],
            add_special_tokens=False,
        )
        if len(forced_ids) > max_decode_steps:
            raise Lfm2CpcfBlankFoldRunnerError(
                f"Trajectory {row['id']!r} exceeds the sealed decode count."
            )
        records.append(
            Lfm2TargetTokenizedRecord(
                record_id=row["id"],
                split=row["split"],
                input_ids=torch.tensor([input_ids], dtype=torch.long),
                attention_mask=torch.ones(
                    (1, len(input_ids)),
                    dtype=torch.long,
                ),
                forced_decode_ids=torch.tensor(
                    [forced_ids],
                    dtype=torch.long,
                ),
            )
        )
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        bridge,
        records,
        expected_artifact_sha256=bridge.artifact_sha256,
    )
    ordered = corpus.get("ordered_batch_input_sha256")
    observed = tuple(
        record.donor_cache_input_sha256 for record in adapter.records
    )
    if (
        not isinstance(ordered, Sequence)
        or isinstance(ordered, (str, bytes))
        or tuple(ordered) != observed
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Canonical target tokenization does not reproduce the sealed "
            "mapped-donor cache corpus."
        )
    return adapter


def _mapped_source_tied_table(
    parent_model: Any,
    bridge: Any,
) -> Any:
    import torch

    input_weight = getattr(parent_model.get_input_embeddings(), "weight", None)
    output_weight = getattr(parent_model.get_output_embeddings(), "weight", None)
    if (
        input_weight is None
        or output_weight is not input_weight
        or input_weight.ndim != 2
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "The current CPCF interface requires one native tied parent table."
        )
    mapping = bridge.transport_profile.target_to_donor_row_indices
    index = torch.tensor(
        mapping,
        device=input_weight.device,
        dtype=torch.long,
    )
    if int(index.max().item()) >= int(input_weight.shape[0]):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Strict bridge addresses a row outside the physical parent table."
        )
    return input_weight.detach().index_select(0, index).to(device="cpu")


def _checkpoint_inventory(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def _materialized_lfm2_ffn_width(model: Any, name: str) -> int:
    """Return the exact FFN width jointly proven by config and live weights."""

    config = getattr(model, "config", None)
    core = getattr(model, "model", None)
    layers = tuple(getattr(core, "layers", ()))
    if (
        config is None
        or getattr(config, "model_type", None) != "lfm2"
        or not layers
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} is not a materialized LFM2 model."
        )
    widths = set()
    for index, layer in enumerate(layers):
        feed_forward = getattr(layer, "feed_forward", None)
        w1 = getattr(getattr(feed_forward, "w1", None), "weight", None)
        w2 = getattr(getattr(feed_forward, "w2", None), "weight", None)
        w3 = getattr(getattr(feed_forward, "w3", None), "weight", None)
        if (
            w1 is None
            or w2 is None
            or w3 is None
            or w1.ndim != 2
            or w2.ndim != 2
            or w3.ndim != 2
            or int(w1.shape[0]) != int(w3.shape[0])
            or int(w1.shape[0]) != int(w2.shape[1])
        ):
            raise Lfm2CpcfBlankFoldRunnerError(
                f"{name} layer {index} has inconsistent materialized FFN weights."
            )
        widths.add(int(w1.shape[0]))
    if len(widths) != 1:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} layers do not share one materialized FFN width."
        )
    width = next(iter(widths))
    configured = getattr(config, "intermediate_size", None)
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured != width
        or getattr(config, "block_auto_adjust_ff_dim", None) is not False
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} runtime FFN width is not sealed by intermediate_size with "
            "block_auto_adjust_ff_dim=False."
        )
    return width


def _restore_lfm2_ffn_config_alias(model: Any, name: str) -> dict[str, Any]:
    """Restore the legacy alias consumed by Lfm2Config during local reload."""

    width = _materialized_lfm2_ffn_width(model, name)
    config = model.config
    previous = getattr(config, "block_ff_dim", None)
    if previous is not None and previous != width:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} block_ff_dim conflicts with its materialized FFN width."
        )
    config.block_ff_dim = width
    payload = config.to_dict()
    expected = {
        "intermediate_size": width,
        "block_ff_dim": width,
        "block_auto_adjust_ff_dim": False,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} cannot serialize shape-stable LFM2 FFN aliases: "
            f"{observed} != {expected}."
        )
    body = {
        "scope": name,
        "materialized_ffn_width": width,
        "previous_block_ff_dim": previous,
        "serialized_aliases": observed,
        "shape_stable": True,
    }
    return {**body, "report_sha256": _sha(body)}


def _verify_saved_lfm2_ffn_config(
    directory: Path,
    *,
    expected_width: int,
) -> dict[str, Any]:
    """Fail closed if checkpoint JSON cannot rebuild the exact FFN shape."""

    config_path = directory / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"Could not authenticate saved LFM2 config: {config_path}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Saved LFM2 config must contain a JSON object."
        )
    expected = {
        "intermediate_size": expected_width,
        "block_ff_dim": expected_width,
        "block_auto_adjust_ff_dim": False,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Saved config.json cannot reconstruct the materialized LFM2 FFN "
            f"shape: {observed} != {expected}."
        )
    body = {
        "scope": "saved_config_json",
        "config_json_sha256": _file_sha256(config_path),
        "serialized_aliases": observed,
        "shape_stable": True,
    }
    return {**body, "report_sha256": _sha(body)}


def _repetition_proxy(token_ids: Sequence[int]) -> dict[str, Any]:
    if not token_ids:
        return {
            "token_count": 0,
            "unique_token_fraction": 0.0,
            "maximum_identical_run": 0,
            "passed": False,
        }
    maximum_run = 1
    run = 1
    for previous, current in zip(token_ids, token_ids[1:]):
        if current == previous:
            run += 1
            maximum_run = max(maximum_run, run)
        else:
            run = 1
    unique_fraction = len(set(token_ids)) / len(token_ids)
    passed = (
        maximum_run <= max(4, int(math.ceil(len(token_ids) * 0.5)))
        and unique_fraction >= 0.10
    )
    return {
        "token_count": len(token_ids),
        "unique_token_fraction": unique_fraction,
        "maximum_identical_run": maximum_run,
        "passed": passed,
    }


def _standalone_whole_response_evaluation(
    model: Any,
    tokenizer: Any,
    *,
    rows: Sequence[Mapping[str, str]],
    device: Any,
    limit: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    validation = [row for row in rows if row["split"] == "validation"][:limit]
    if not validation:
        raise Lfm2CpcfBlankFoldRunnerError(
            "No validation prompts are available for standalone evaluation."
        )
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, Sequence) and not isinstance(eos, (str, bytes)):
        eos = eos[0] if eos else None
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = eos
    results = []
    model.eval()
    with torch.inference_mode():
        for row in validation:
            encoded = tokenizer(
                row["input_text"],
                add_special_tokens=True,
                return_tensors="pt",
            )
            if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                raise Lfm2CpcfBlankFoldRunnerError(
                    "Reloaded tokenizer returned no standalone input IDs."
                )
            input_ids = encoded["input_ids"].to(device=device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(device=device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad,
                eos_token_id=eos,
            )
            continuation = generated[0, int(input_ids.shape[1]) :].tolist()
            text = tokenizer.decode(
                continuation,
                skip_special_tokens=True,
            )
            repetition = _repetition_proxy(continuation)
            observed_token_count = len(continuation)
            token_count_matches_ids = (
                repetition["token_count"] == observed_token_count
            )
            within_generation_limit = observed_token_count <= max_new_tokens
            nonempty = bool(text.strip())
            control_count = sum(
                ord(character) < 32 and character not in "\n\r\t"
                for character in text
            )
            proxy_pass = nonempty and control_count == 0 and repetition["passed"]
            results.append(
                {
                    "record_id": row["id"],
                    "prompt": row["input_text"],
                    "cached_trajectory_reference": row[
                        "forced_decode_text"
                    ],
                    "child_output": text,
                    "child_output_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "continuation_token_ids": continuation,
                    "observed_continuation_token_count": observed_token_count,
                    "observed_token_count_matches_continuation_token_ids": (
                        token_count_matches_ids
                    ),
                    "observed_token_count_within_generation_limit": (
                        within_generation_limit
                    ),
                    "nonempty": nonempty,
                    "control_character_count": control_count,
                    "repetition_proxy": repetition,
                    "automatic_coherence_proxy_pass": proxy_pass,
                    "manual_intelligence_score": None,
                }
            )
    passed = sum(
        bool(result["automatic_coherence_proxy_pass"]) for result in results
    )
    body = {
        "schema": STANDALONE_EVALUATION_SCHEMA,
        "evaluation_mode": "fresh_reload_standalone_whole_response_generation",
        "requested_case_limit": limit,
        "requested_max_new_tokens": max_new_tokens,
        "generation_limit": max_new_tokens,
        "generation_limit_semantics": "maximum_new_tokens_per_case",
        "requested_max_new_tokens_matches_generation_limit": True,
        "case_count": len(results),
        "all_observed_token_counts_match_continuation_token_ids": all(
            bool(
                result[
                    "observed_token_count_matches_continuation_token_ids"
                ]
            )
            for result in results
        ),
        "all_observed_token_counts_within_generation_limit": all(
            bool(result["observed_token_count_within_generation_limit"])
            for result in results
        ),
        "automatic_coherence_proxy_pass_count": passed,
        "automatic_coherence_proxy_pass_fraction": passed / len(results),
        "cases": results,
        "exact_token_match_principal_objective": False,
        "cached_reference_is_not_refitted_or_reexecuted": True,
        "manual_semantic_intelligence_review_required": True,
        "manual_intelligence_retention_proven": False,
        "parent_model_calls": 0,
        "parent_generation_calls": 0,
        "target_generation_calls": len(results),
    }
    return {**body, "report_sha256": _sha(body)}


def _save_reload_evaluate(
    target_model: Any,
    target_tokenizer: Any,
    *,
    output: Path,
    parent_model: Any,
    planned_parameters: int,
    rows: Sequence[Mapping[str, str]],
    device: Any,
    dtype: Any,
    evaluation_limit: int,
    evaluation_max_new_tokens: int,
) -> tuple[
    tuple[
        Any,
        Mapping[str, Any],
        Mapping[str, Any],
        list[dict[str, Any]],
        Path,
        Mapping[str, Any],
    ],
    dict[str, Any],
]:
    from experiments.run_lfm2_residual_attention_fold import (
        load_local_lfm2_model,
        load_local_tokenizer,
    )
    from neural_forge.hf_lfm2_cpcf_model_fold import (
        audit_lfm2_cpcf_parameter_compaction,
        independent_parameter_inventory,
    )

    stage = output.parent / f".{output.name}.stage-{uuid.uuid4().hex}"
    if stage.exists() or stage.is_symlink():
        raise Lfm2CpcfBlankFoldRunnerError(
            f"Unexpected staging collision: {stage}."
        )
    stage.mkdir()
    try:
        target_aliases = _restore_lfm2_ffn_config_alias(
            target_model,
            "compiled_target_before_save",
        )
        target_model.save_pretrained(
            stage,
            safe_serialization=True,
        )
        target_tokenizer.save_pretrained(stage)
        saved_aliases = _verify_saved_lfm2_ffn_config(
            stage,
            expected_width=int(target_aliases["materialized_ffn_width"]),
        )
        saved_files = _checkpoint_inventory(stage)
        reloaded = load_local_lfm2_model(
            stage,
            device=device,
            dtype=dtype,
        )
        reloaded_aliases = _restore_lfm2_ffn_config_alias(
            reloaded,
            "freshly_reloaded_target",
        )
        alias_body = {
            "compiled_target": target_aliases,
            "saved_config": saved_aliases,
            "fresh_reload": reloaded_aliases,
            "materialized_width_preserved": (
                target_aliases["materialized_ffn_width"]
                == reloaded_aliases["materialized_ffn_width"]
            ),
            "block_ff_dim_restored_after_loader_consumption": True,
            "shape_stable": True,
        }
        alias_audit = {**alias_body, "report_sha256": _sha(alias_body)}
        reloaded_inventory = independent_parameter_inventory(reloaded)
        compression = audit_lfm2_cpcf_parameter_compaction(
            parent_model,
            reloaded,
            planned_target_unique_parameters=planned_parameters,
            promotion_compression_ratio=float(
                MINIMUM_CLAIMABLE_COMPRESSION_RATIO
            ),
        )
        reloaded_tokenizer = load_local_tokenizer(stage)
        behavior = _standalone_whole_response_evaluation(
            reloaded,
            reloaded_tokenizer,
            rows=rows,
            device=device,
            limit=evaluation_limit,
            max_new_tokens=evaluation_max_new_tokens,
        )
        return (
            reloaded,
            reloaded_inventory,
            compression,
            saved_files,
            stage,
            alias_audit,
        ), behavior
    except Exception:
        resolved = stage.resolve()
        if (
            resolved.parent == output.parent.resolve()
            and resolved.name.startswith(f".{output.name}.stage-")
        ):
            shutil.rmtree(resolved)
        raise


def _write_dry_plan_receipt(
    *,
    path: Path,
    parent_inventory: Mapping[str, Any],
    child_inventory: Mapping[str, Any],
    compression: Mapping[str, Any],
    blank: Any,
    cache: Any,
    bridge: Any,
) -> dict[str, Any]:
    body = {
        "schema": DRY_PLAN_SCHEMA,
        "status": "structural_dry_plan_only",
        "cache_key": cache.cache_key,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "blank_target": blank.as_dict(),
        "parent_parameter_inventory": dict(parent_inventory),
        "child_parameter_inventory": dict(child_inventory),
        "compression": dict(compression),
        "fresh_dense_blank_synthesized": True,
        "layer_compilation_performed": False,
        "folded_child_created": False,
        "standalone_behavior_evaluated": False,
        "intelligence_retention_claimed": False,
        "parameter_compaction_structurally_proven": True,
        "training_used": False,
        "quantization_used": False,
        "pruning_used": False,
        "fallback_used": False,
    }
    receipt = {**body, "receipt_sha256": _sha(body)}
    _atomic_json(path, receipt)
    return receipt


def run(config: RunnerConfig) -> dict[str, Any]:
    """Execute one strict local CPCF fold or an explicit structural dry plan."""

    import torch

    from neural_forge.hf_lfm2_circuit_trace_cache import (
        fingerprint_lfm2_parent,
        open_lfm2_circuit_trace_cache,
    )
    from neural_forge.hf_lfm2_cpcf_blank_target import (
        plan_and_synthesize_lfm2_cpcf_blank,
    )
    from neural_forge.hf_lfm2_cpcf_model_fold import (
        Lfm2CpcfModelFoldPolicy,
        audit_lfm2_cpcf_parameter_compaction,
        compile_lfm2_cpcf_model_fold,
        independent_parameter_inventory,
    )
    from neural_forge.hf_tokenizer_bridge import (
        fingerprint_hf_tokenizer_semantics,
        load_strict_tokenizer_bridge_artifact,
    )
    from experiments.run_lfm2_residual_attention_fold import (
        direct_local_io_audit,
        load_local_lfm2_model,
        load_local_tokenizer,
    )

    parent_directory = _local_directory(
        config.parent_model_directory,
        "parent_model_directory",
    )
    donor_tokenizer_directory = _local_directory(
        config.donor_tokenizer_directory,
        "donor_tokenizer_directory",
    )
    target_tokenizer_directory = _local_directory(
        config.target_tokenizer_directory,
        "target_tokenizer_directory",
    )
    bridge_directory = _local_directory(
        config.bridge_directory,
        "bridge_directory",
    )
    cache_directory = _local_directory(
        config.trace_cache_directory,
        "trace_cache_directory",
    )
    trajectory_path = _local_file(
        config.trajectory_corpus,
        "trajectory_corpus",
    )
    output = _new_output_directory(config.output_directory)
    if config.receipt is None:
        receipt_path = output.with_suffix(".receipt.json")
    else:
        receipt_path = config.receipt.expanduser().absolute()
        if receipt_path.exists() or receipt_path.is_symlink():
            raise Lfm2CpcfBlankFoldRunnerError(
                f"Receipt already exists: {receipt_path}."
            )
    dtype = _torch_dtype(torch, config.dtype)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Lfm2CpcfBlankFoldRunnerError(
            "CUDA was requested but is unavailable."
        )

    donor_tokenizer = load_local_tokenizer(donor_tokenizer_directory)
    target_tokenizer = load_local_tokenizer(target_tokenizer_directory)
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    bridge.verify_files()
    donor_fingerprint = fingerprint_hf_tokenizer_semantics(donor_tokenizer)
    target_fingerprint = fingerprint_hf_tokenizer_semantics(target_tokenizer)
    if (
        donor_fingerprint != bridge.donor_tokenizer_semantics_sha256
        or target_fingerprint != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Live local tokenizers differ from the sealed bridge."
        )

    # Metadata is opened first; the installability adapter performs the full
    # cache verification.  We intentionally do not run its separate audit()
    # pass, which would rematerialize every broad-cache tensor a second time.
    cache = open_lfm2_circuit_trace_cache(cache_directory, verify=False)
    parent = load_local_lfm2_model(
        parent_directory,
        device=device,
        dtype=dtype,
    )
    parent.requires_grad_(False)
    parent.eval()
    parent_fingerprint_before = fingerprint_lfm2_parent(parent)
    if parent_fingerprint_before.identity_sha256 != cache.parent_fingerprint:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Explicit local parent differs from the sealed trace cache."
        )
    parent_inventory = independent_parameter_inventory(parent)
    blank = plan_and_synthesize_lfm2_cpcf_blank(
        parent,
        parent_name=config.parent_name,
        parent_tokenizer_fingerprint=donor_fingerprint,
        evidence_fingerprint=cache.cache_key,
        child_name=config.child_name,
        target_vocabulary_size=(
            bridge.transport_profile.target_row_capacity
        ),
        child_tokenizer_fingerprint=target_fingerprint,
        tokenizer_bridge_artifact=bridge,
        minimum_hidden_size=config.minimum_hidden_size,
        maximum_candidates=config.maximum_candidates,
        selected_candidate_rank=config.selected_candidate_rank,
        blank_seed=config.blank_seed,
        device=device,
        dtype=dtype,
        **config.compression.as_kwargs(),
    )
    target = blank.model
    target.requires_grad_(False)
    target.eval()
    blank_inventory = independent_parameter_inventory(target)
    structural_compression = audit_lfm2_cpcf_parameter_compaction(
        parent,
        target,
        planned_target_unique_parameters=blank.unique_parameters,
        promotion_compression_ratio=float(
            MINIMUM_CLAIMABLE_COMPRESSION_RATIO
        ),
    )
    if (
        not structural_compression["promotion_ratio_met"]
        and not config.diagnostic_control
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "The independently measured target is a valid diagnostic fold "
            f"but is below the {MINIMUM_CLAIMABLE_COMPRESSION_RATIO}x "
            "project-promotion floor. Pass --diagnostic-control to run it "
            "without claiming project-goal success."
        )
    if config.structural_dry_plan:
        return _write_dry_plan_receipt(
            path=receipt_path,
            parent_inventory=parent_inventory,
            child_inventory=blank_inventory,
            compression=structural_compression,
            blank=blank,
            cache=cache,
            bridge=bridge,
        )

    from neural_forge.hf_lfm2_cpcf_installability import (
        Lfm2CpcfInstallabilityAdapter,
    )
    from neural_forge.hf_lfm2_cpcf_model_compiler import (
        build_lfm2_cpcf_target_phases,
    )

    rows = _read_trajectory_rows(trajectory_path)
    trace_adapter = _trace_adapter(
        rows=rows,
        target_tokenizer=target_tokenizer,
        bridge=bridge,
        cache_manifest=cache.manifest,
    )
    installability = Lfm2CpcfInstallabilityAdapter(cache)
    # Passing the generator directly avoids a second resident parent-evidence
    # tuple.  Target phases are the sole long-lived owner of the authenticated
    # broad-cache tensors used by all subsequent components.
    phases = build_lfm2_cpcf_target_phases(
        trace_adapter,
        installability.iter_compiler_inputs(),
    )
    if len({phase.batch_index for phase in phases}) != cache.batch_count:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Materialized compiler phases do not cover every cache batch."
        )
    source_tied_table = _mapped_source_tied_table(parent, bridge)
    compiled_fold = compile_lfm2_cpcf_model_fold(
        parent,
        target,
        phases=phases,
        source_tied_table=source_tied_table,
        blank_target_report=blank.as_dict(),
        planned_target_unique_parameters=blank.unique_parameters,
        policy=Lfm2CpcfModelFoldPolicy(
            ridge=config.ridge,
            state_reconstruction_weight=(
                config.boundary_state_reconstruction_weight
            ),
            minimum_predictive_energy_fraction=(
                config.boundary_predictive_energy_floor
            ),
            minimum_validation_block_retention=(
                config.boundary_validation_retention_floor
            ),
            enforce_validation_block_floor=True,
            promotion_compression_ratio=float(
                MINIMUM_CLAIMABLE_COMPRESSION_RATIO
            ),
            require_promotion_ratio=not config.diagnostic_control,
        ),
    )
    compiled_inventory = independent_parameter_inventory(target)
    if (
        int(compiled_inventory["unique_parameters"])
        != int(blank_inventory["unique_parameters"])
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Folding changed the child topology or parameter count."
        )
    parent_fingerprint_after = fingerprint_lfm2_parent(parent)
    if parent_fingerprint_after.identity_sha256 != (
        parent_fingerprint_before.identity_sha256
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Parent parameters or runtime buffers changed during folding."
        )

    saved, behavior = _save_reload_evaluate(
        target,
        target_tokenizer,
        output=output,
        parent_model=parent,
        planned_parameters=blank.unique_parameters,
        rows=rows,
        device=device,
        dtype=dtype,
        evaluation_limit=config.evaluation_limit,
        evaluation_max_new_tokens=config.evaluation_max_new_tokens,
    )
    (
        reloaded,
        reloaded_inventory,
        reload_compression,
        saved_files,
        stage,
        save_reload_config_aliases,
    ) = saved
    del reloaded
    if (
        reload_compression["target_unique_parameters"]
        != structural_compression["target_unique_parameters"]
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Fresh reload changed the independently measured parameter count."
        )
    operation_audit = {
        "local_parent_model_loads": 1,
        "parent_forward_calls": 0,
        "parent_generation_calls": 0,
        "parent_cache_generation_calls": 0,
        "parent_cache_regeneration_calls": 0,
        "sealed_cache_open_count": 1,
        "resident_parent_evidence_collections": 1,
        "separate_installability_audit_rematerialization": False,
        "target_generation_calls": behavior["target_generation_calls"],
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "quantized_parameters": 0,
        "pruned_parameters": 0,
        "fallback_model_calls": 0,
        "network_access": False,
    }
    body = {
        "schema": RUN_SCHEMA,
        "status": (
            "folded_child_requires_manual_intelligence_review"
            if reload_compression["promotion_ratio_met"]
            else "diagnostic_control_fold_requires_manual_review"
        ),
        "inputs": {
            "parent_model_directory": str(parent_directory),
            "donor_tokenizer_directory": str(donor_tokenizer_directory),
            "target_tokenizer_directory": str(target_tokenizer_directory),
            "bridge_directory": str(bridge_directory),
            "trace_cache_directory": str(cache_directory),
            "trajectory_corpus": str(trajectory_path),
            "trajectory_corpus_sha256": _file_sha256(trajectory_path),
            "cache_key": cache.cache_key,
            "parent_fingerprint": cache.parent_fingerprint,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "bridge_profile_sha256": bridge.profile_fingerprint,
            "trace_adapter_sha256": trace_adapter.evidence_sha256,
        },
        "local_io_audit": direct_local_io_audit(parent_directory),
        "blank_target": blank.as_dict(),
        "parent_parameter_inventory": dict(parent_inventory),
        "blank_parameter_inventory": dict(blank_inventory),
        "compiled_parameter_inventory": dict(compiled_inventory),
        "reloaded_parameter_inventory": dict(reloaded_inventory),
        "compression": dict(reload_compression),
        "compiled_model_fold": compiled_fold.as_dict(),
        "boundary_bundle": compiled_fold.boundary_bundle.as_dict(),
        "interface_install": dict(
            compiled_fold.interface_install_report
        ),
        "neutral_norm_install": dict(
            compiled_fold.neutral_norm_install_report
        ),
        "layer_results": [
            result.as_dict() for result in compiled_fold.layer_results
        ],
        "final_norm": dict(compiled_fold.final_norm_report),
        "standalone_whole_response_evaluation": behavior,
        "saved_checkpoint_files": saved_files,
        "save_reload_config_aliases": dict(save_reload_config_aliases),
        "output_directory": str(output),
        "parent_state_preserved": True,
        "all_target_layers_compiled_sequentially": True,
        "fresh_reload_verified": True,
        "structural_parameter_compaction_proven": True,
        "project_promotion_ratio_met": reload_compression[
            "promotion_ratio_met"
        ],
        "diagnostic_control": config.diagnostic_control,
        "behavioral_intelligence_retention_proven": False,
        "manual_intelligence_review_required": True,
        "operation_audit": operation_audit,
        "claims": {
            "folding_only": True,
            "blank_target_filled": True,
            "training_used": False,
            "quantization_used": False,
            "pruning_used": False,
            "parent_backed_fallback_used": False,
            "exact_token_match_used_as_principal_objective": False,
            "marginal_compaction_reported_as_success": False,
        },
    }
    receipt = {**body, "receipt_sha256": _sha(body)}
    internal_receipt = stage / "cpcf_run_receipt.json"
    _atomic_json(internal_receipt, receipt)
    if output.exists():
        raise Lfm2CpcfBlankFoldRunnerError(
            "Output appeared while the child was being compiled."
        )
    os.replace(stage, output)
    _atomic_json(receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model-directory", required=True, type=Path)
    parser.add_argument(
        "--donor-tokenizer-directory",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--target-tokenizer-directory",
        required=True,
        type=Path,
    )
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument(
        "--trace-cache-directory",
        required=True,
        type=Path,
    )
    parser.add_argument("--trajectory-corpus", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--compression-ratio")
    budget.add_argument("--target-unique-parameters", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=_DTYPES, default="bfloat16")
    parser.add_argument("--parent-name", default="lfm2-parent")
    parser.add_argument("--child-name", default="lfm2-cpcf-folded-child")
    parser.add_argument("--blank-seed", type=int, default=0)
    parser.add_argument("--minimum-hidden-size", type=int, default=32)
    parser.add_argument("--maximum-candidates", type=int, default=8)
    parser.add_argument("--selected-candidate-rank", type=int)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument(
        "--boundary-state-reconstruction-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--boundary-predictive-energy-floor",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--boundary-validation-retention-floor",
        type=float,
        default=0.0,
    )
    parser.add_argument("--evaluation-limit", type=int, default=12)
    parser.add_argument(
        "--evaluation-max-new-tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--structural-dry-plan",
        action="store_true",
        help=(
            "Synthesize and count the dense blank, then stop. This mode "
            "cannot claim a folded or behaviorally valid child."
        ),
    )
    parser.add_argument(
        "--diagnostic-control",
        action="store_true",
        help=(
            "Allow a real fold below the current 100x promotion floor while "
            "labeling it explicitly as a non-goal diagnostic control."
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RunnerConfig:
    return RunnerConfig(
        parent_model_directory=args.parent_model_directory,
        donor_tokenizer_directory=args.donor_tokenizer_directory,
        target_tokenizer_directory=args.target_tokenizer_directory,
        bridge_directory=args.bridge_directory,
        trace_cache_directory=args.trace_cache_directory,
        trajectory_corpus=args.trajectory_corpus,
        output_directory=args.output_directory,
        receipt=args.receipt,
        compression=CompressionRequest(
            compression_ratio=args.compression_ratio,
            target_unique_parameters=args.target_unique_parameters,
        ),
        device=args.device,
        dtype=args.dtype,
        parent_name=args.parent_name,
        child_name=args.child_name,
        blank_seed=args.blank_seed,
        minimum_hidden_size=args.minimum_hidden_size,
        maximum_candidates=args.maximum_candidates,
        selected_candidate_rank=args.selected_candidate_rank,
        ridge=args.ridge,
        boundary_state_reconstruction_weight=(
            args.boundary_state_reconstruction_weight
        ),
        boundary_predictive_energy_floor=(
            args.boundary_predictive_energy_floor
        ),
        boundary_validation_retention_floor=(
            args.boundary_validation_retention_floor
        ),
        evaluation_limit=args.evaluation_limit,
        evaluation_max_new_tokens=args.evaluation_max_new_tokens,
        structural_dry_plan=args.structural_dry_plan,
        diagnostic_control=args.diagnostic_control,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run(config_from_args(args))
    except (
        Lfm2CpcfBlankFoldRunnerError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
