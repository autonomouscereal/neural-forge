#!/usr/bin/env python3
"""Generate the independent LFM2 holdout parent responses exactly once.

The command accepts only explicit local model/tokenizer directories, disables
hub access, refuses to replace an output, calls ``generate`` once per manifest
case, and seals the responses with exact model/tokenizer/manifest identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

try:
    from experiments import evaluate_lfm2_intelligence_retention as posthoc
    from experiments import run_lfm2_residual_attention_fold as fold_runtime
except ImportError:  # Direct execution from the experiments directory.
    import evaluate_lfm2_intelligence_retention as posthoc
    import run_lfm2_residual_attention_fold as fold_runtime

from neural_forge.dynamic_circuit_ir import canonical_json_bytes
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_intelligence_gate import (
    IntelligenceHoldoutManifest,
    load_lfm2_intelligence_holdout_manifest,
)
from neural_forge.hf_lfm2_intelligence_holdout import (
    seal_lfm2_intelligence_holdout_parent_envelope,
)


class Lfm2HoldoutParentCacheError(RuntimeError):
    """Local identity, generation, or one-time publication failed closed."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2HoldoutParentCacheError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _local_directory(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2HoldoutParentCacheError(
            f"{name} must be an explicit local directory."
        )
    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2HoldoutParentCacheError(
            f"{name} does not exist."
        ) from exc
    if not path.is_dir() or path.is_symlink():
        raise Lfm2HoldoutParentCacheError(
            f"{name} is missing or unsafe: {path}."
        )
    return path


def _local_file(value: str | os.PathLike[str], name: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2HoldoutParentCacheError(
            f"{name} does not exist."
        ) from exc
    if not path.is_file() or path.is_symlink():
        raise Lfm2HoldoutParentCacheError(
            f"{name} is missing or unsafe: {path}."
        )
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_new_json(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise Lfm2HoldoutParentCacheError(
            f"Refusing to replace existing parent cache {destination}."
        )
    payload = canonical_json_bytes(dict(value)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise Lfm2HoldoutParentCacheError(
            "The local parent model has no Parameters."
        ) from exc


def generate_lfm2_holdout_rows_once(
    model: torch.nn.Module,
    tokenizer: Any,
    manifest: IntelligenceHoldoutManifest,
) -> tuple[dict[str, Any], ...]:
    """Call the local parent ``generate`` exactly once for every case."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(manifest, IntelligenceHoldoutManifest):
        raise TypeError("manifest must be IntelligenceHoldoutManifest.")
    device = _model_device(model)
    was_training = bool(model.training)
    model.eval()
    rows = []
    try:
        with torch.inference_mode():
            for spec in manifest.cases:
                encoded = tokenizer(
                    spec.prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                if (
                    not isinstance(encoded, Mapping)
                    or not isinstance(encoded.get("input_ids"), torch.Tensor)
                    or encoded["input_ids"].ndim != 2
                    or int(encoded["input_ids"].shape[0]) != 1
                ):
                    raise Lfm2HoldoutParentCacheError(
                        f"Tokenizer output is invalid for {spec.case_id}."
                    )
                input_ids = encoded["input_ids"].to(
                    device=device,
                    dtype=torch.long,
                )
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    if not isinstance(attention_mask, torch.Tensor):
                        raise Lfm2HoldoutParentCacheError(
                            f"Attention mask is invalid for {spec.case_id}."
                        )
                    attention_mask = attention_mask.to(
                        device=device,
                        dtype=torch.long,
                    )
                kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "max_new_tokens": spec.generation_limit,
                    "do_sample": False,
                    "use_cache": True,
                }
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
                if pad_token_id is not None:
                    kwargs["pad_token_id"] = int(pad_token_id)
                generated = model.generate(**kwargs)
                if (
                    not isinstance(generated, torch.Tensor)
                    or generated.ndim != 2
                    or int(generated.shape[0]) != 1
                    or int(generated.shape[1]) < int(input_ids.shape[1])
                ):
                    raise Lfm2HoldoutParentCacheError(
                        f"Parent generation is invalid for {spec.case_id}."
                    )
                continuation = generated[0, int(input_ids.shape[1]) :]
                output_ids = [
                    int(value)
                    for value in continuation.detach().to("cpu").tolist()
                ]
                if len(output_ids) > spec.generation_limit:
                    raise Lfm2HoldoutParentCacheError(
                        f"Parent exceeded the limit for {spec.case_id}."
                    )
                output = tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                )
                if not isinstance(output, str):
                    raise Lfm2HoldoutParentCacheError(
                        f"Decoded parent output is invalid for {spec.case_id}."
                    )
                rows.append(
                    {
                        "case_id": spec.case_id,
                        "input_token_count": int(input_ids.shape[1]),
                        "output": output,
                        "output_token_ids": output_ids,
                        # Parent sealing derives this value from the manifest,
                        # while candidate sealing requires the generation row
                        # to carry it explicitly.  The same one-shot generator
                        # is intentionally reused for both roles.
                        "generation_limit": spec.generation_limit,
                    }
                )
    finally:
        model.train(was_training)
    if len(rows) != len(manifest.cases):
        raise Lfm2HoldoutParentCacheError(
            "Parent generation did not cover the exact manifest."
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class ParentCacheConfig:
    parent_model_directory: Path
    tokenizer_directory: Path
    output_path: Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    manifest_path: Path | None = None
    expected_manifest_sha256: str | None = None
    expected_parent_state_fingerprint: str | None = None
    expected_parent_model_fingerprint: str | None = None
    expected_tokenizer_fingerprint: str | None = None

    def __post_init__(self) -> None:
        parent = _local_directory(
            self.parent_model_directory,
            "parent_model_directory",
        )
        tokenizer = _local_directory(
            self.tokenizer_directory,
            "tokenizer_directory",
        )
        output = Path(self.output_path).expanduser().absolute()
        if output.exists():
            raise Lfm2HoldoutParentCacheError(
                f"Refusing to replace existing parent cache {output}."
            )
        if output == parent or output == tokenizer:
            raise Lfm2HoldoutParentCacheError(
                "Parent cache path cannot replace an input directory."
            )
        manifest = (
            None
            if self.manifest_path is None
            else _local_file(self.manifest_path, "manifest_path")
        )
        if manifest is not None and self.expected_manifest_sha256 is None:
            raise Lfm2HoldoutParentCacheError(
                "An external manifest_path requires its expected SHA-256."
            )
        object.__setattr__(self, "parent_model_directory", parent)
        object.__setattr__(self, "tokenizer_directory", tokenizer)
        object.__setattr__(self, "output_path", output)
        object.__setattr__(self, "manifest_path", manifest)
        if not isinstance(self.device, str) or not self.device:
            raise Lfm2HoldoutParentCacheError("device must be nonempty.")
        fold_runtime._torch_dtype(self.dtype)
        for name in (
            "expected_manifest_sha256",
            "expected_parent_state_fingerprint",
            "expected_parent_model_fingerprint",
            "expected_tokenizer_fingerprint",
        ):
            value = getattr(self, name)
            if value is not None:
                _digest(value, name)


def build_parent_cache(config: ParentCacheConfig) -> dict[str, Any]:
    """Load one explicit local parent, generate once, and publish atomically."""

    if not isinstance(config, ParentCacheConfig):
        raise TypeError("config must be ParentCacheConfig.")
    if config.output_path.exists():
        raise Lfm2HoldoutParentCacheError(
            "Parent cache appeared before model loading; refusing overwrite."
        )
    manifest = load_lfm2_intelligence_holdout_manifest(
        config.manifest_path,
        expected_manifest_sha256=config.expected_manifest_sha256,
    )
    tokenizer_fingerprint, _tokenizer_files = (
        posthoc.fingerprint_bound_local_tokenizer(
            config.tokenizer_directory
        )
    )
    if (
        config.expected_tokenizer_fingerprint is not None
        and tokenizer_fingerprint
        != config.expected_tokenizer_fingerprint
    ):
        raise Lfm2HoldoutParentCacheError(
            "Local tokenizer differs from its expected fingerprint."
        )
    special = posthoc.load_bound_special_token_ids(
        config.tokenizer_directory,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
    )
    model = fold_runtime.load_local_lfm2_model(
        config.parent_model_directory,
        device=config.device,
        dtype=fold_runtime._torch_dtype(config.dtype),
    )
    model.eval()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise Lfm2HoldoutParentCacheError(
            "The local parent already contains gradients."
        )
    state_before = fold_runtime.model_state_sha256(model)
    model_fingerprint = fingerprint_lfm2_parent(model).identity_sha256
    if (
        config.expected_parent_state_fingerprint is not None
        and state_before != config.expected_parent_state_fingerprint
    ):
        raise Lfm2HoldoutParentCacheError(
            "Local parent state differs from its expected fingerprint."
        )
    if (
        config.expected_parent_model_fingerprint is not None
        and model_fingerprint
        != config.expected_parent_model_fingerprint
    ):
        raise Lfm2HoldoutParentCacheError(
            "Local parent model identity differs from its expectation."
        )
    generated_rows = generate_lfm2_holdout_rows_once(
        model,
        fold_runtime.load_local_tokenizer(config.tokenizer_directory),
        manifest,
    )
    state_after = fold_runtime.model_state_sha256(model)
    if (
        state_after != state_before
        or any(parameter.grad is not None for parameter in model.parameters())
    ):
        raise Lfm2HoldoutParentCacheError(
            "Parent state or gradients changed during inference."
        )
    # The reusable generator also serves candidate capture, whose typed rows
    # require an explicit generation_limit.  Parent-envelope normalization has
    # a deliberately narrower schema because limits are authenticated by the
    # manifest, so project to that exact schema before sealing the parent.
    parent_rows = tuple(
        {
            "case_id": row["case_id"],
            "input_token_count": row["input_token_count"],
            "output": row["output"],
            "output_token_ids": row["output_token_ids"],
        }
        for row in generated_rows
    )
    document = seal_lfm2_intelligence_holdout_parent_envelope(
        manifest,
        parent_rows,
        parent_state_fingerprint=state_before,
        parent_model_fingerprint=model_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
        eos_token_ids=special["special_token_ids"]["eos"],
    )
    _atomic_write_new_json(config.output_path, document)
    return {
        **document,
        "output_file_sha256": _file_sha256(config.output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the exact independent LFM2 parent holdout once from "
            "explicit local model and tokenizer directories."
        )
    )
    parser.add_argument(
        "--parent-model-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer-directory",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-parent-state-fingerprint")
    parser.add_argument("--expected-parent-model-fingerprint")
    parser.add_argument("--expected-tokenizer-fingerprint")
    return parser


def config_from_args(args: argparse.Namespace) -> ParentCacheConfig:
    return ParentCacheConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    document = build_parent_cache(config)
    print(
        json.dumps(
            {
                "manifest_sha256": document["manifest_sha256"],
                "parent_cache_key": document["parent_cache_key"],
                "parent_state_fingerprint": document[
                    "parent_state_fingerprint"
                ],
                "parent_model_fingerprint": document[
                    "parent_model_fingerprint"
                ],
                "tokenizer_fingerprint": document[
                    "tokenizer_fingerprint"
                ],
                "envelope_sha256": document["envelope_sha256"],
                "output_file_sha256": document["output_file_sha256"],
                "case_count": document["case_count"],
                "parent_generate_invocations": document[
                    "parent_generate_invocations"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
