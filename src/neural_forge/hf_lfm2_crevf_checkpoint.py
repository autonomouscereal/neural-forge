"""Standalone, local-only LFM2 checkpoints backed by a CREVF runtime.

The checkpoint boundary is intentionally narrower than the folding solver.  It
accepts an already solved and authenticated CREVF artifact, copies only a local
LFM2 configuration/generation/tokenizer package, seals every physical byte in a
canonical manifest, and reloads through the existing transactional CREVF
runtime installer.

No parent model, hub loader, network request, training step, Python pickle, or
serialized module participates in export or reload.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .hf_lfm2_causal_residual_entropy_value_fold import (
    CrevfBudgetError,
    CrevfError,
    CrevfPackedRuntime,
    CrevfReductionTarget,
    decode_crevf_artifact,
)
from .hf_lfm2_crevf_runtime_install import (
    Lfm2CrevfInstallReport,
    install_lfm2_crevf_runtime,
)


CHECKPOINT_SCHEMA = "neural-forge-lfm2-crevf-checkpoint-v1"
MANIFEST_FILENAME = "neural_forge_lfm2_crevf_manifest.json"
ARTIFACT_FILENAME = "model.crevf"
CONFIG_FILENAME = "config.json"
GENERATION_CONFIG_FILENAME = "generation_config.json"
TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
)
NONEXECUTING_AUXILIARY_KINDS = frozenset(
    {
        "metadata",
        "metadata.composite",
        "provenance",
        "diagnostic",
    }
)
_DIGEST_LENGTH = 64
_DEFAULT_MAX_AUXILIARY_PAYLOAD_BYTES = 64 * 1024 * 1024


class Lfm2CrevfCheckpointError(ValueError):
    """The local package or requested checkpoint is invalid."""


class Lfm2CrevfCheckpointBudgetError(Lfm2CrevfCheckpointError):
    """The checkpoint exceeds a physical, persistent, or decode ceiling."""


class Lfm2CrevfCheckpointFormatError(Lfm2CrevfCheckpointError):
    """The standalone checkpoint is incomplete, altered, or unsupported."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Lfm2CrevfCheckpointFormatError(
                    f"Duplicate JSON key {key!r} in {label}."
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise Lfm2CrevfCheckpointFormatError(
            f"Non-finite JSON constant {value!r} in {label}."
        )

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except Lfm2CrevfCheckpointFormatError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2CrevfCheckpointFormatError(
            f"Could not read strict JSON object {label!r}."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2CrevfCheckpointFormatError(
            f"JSON member {label!r} must contain an object."
        )
    return value


def _strict_keys(
    value: Any,
    expected: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Lfm2CrevfCheckpointFormatError(
            f"{label} fields differ from the checkpoint schema."
        )
    return value


def _positive_limit(value: int | None, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CrevfCheckpointError(f"{label} must be a positive integer.")
    return value


def _manifest_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Lfm2CrevfCheckpointFormatError(
            f"Manifest field {label!r} must be an integer."
        )
    return value


def _minimum_limit(*values: int | None) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _canonical_name(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "/" in value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise Lfm2CrevfCheckpointError(
            f"{label} must be a canonical non-path string."
        )
    return value


def _canonical_aliases(
    logical_to_canonical: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(logical_to_canonical, Mapping) or not logical_to_canonical:
        raise Lfm2CrevfCheckpointError(
            "logical_to_canonical must be a nonempty mapping."
        )
    aliases: dict[str, str] = {}
    for logical, canonical in logical_to_canonical.items():
        logical_name = _canonical_name(logical, "logical parameter name")
        canonical_name = _canonical_name(canonical, "canonical tensor name")
        if logical_name in aliases:
            raise Lfm2CrevfCheckpointError(
                f"Duplicate logical parameter name {logical_name!r}."
            )
        aliases[logical_name] = canonical_name
    aliases = dict(sorted(aliases.items()))
    embedding = aliases.get("model.embed_tokens.weight")
    head = aliases.get("lm_head.weight")
    if embedding is None or head is None or embedding != head:
        raise Lfm2CrevfCheckpointError(
            "The local LFM2 aliases must tie input embeddings and lm_head once."
        )
    if embedding != "model.embed_tokens.weight":
        raise Lfm2CrevfCheckpointError(
            "The tied vocabulary canonical name must be "
            "'model.embed_tokens.weight'."
        )
    return aliases


def _safe_source_directory(
    source: str | os.PathLike[str],
) -> Path:
    path = Path(source).expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise Lfm2CrevfCheckpointError(
            f"local_package_directory {path} must be a real local directory."
        )
    return path


def _read_local_package(
    source: Path,
) -> tuple[dict[str, bytes], dict[str, Any], dict[str, Any]]:
    required = (CONFIG_FILENAME, GENERATION_CONFIG_FILENAME, "tokenizer.json")
    members: dict[str, bytes] = {}
    for name in (CONFIG_FILENAME, GENERATION_CONFIG_FILENAME, *TOKENIZER_ARTIFACTS):
        path = source / name
        if path.is_symlink():
            raise Lfm2CrevfCheckpointError(
                f"Local package artifact cannot be a symlink: {path}."
            )
        if not path.is_file():
            if name in required:
                raise Lfm2CrevfCheckpointError(
                    f"Local package lacks required artifact {name!r}."
                )
            continue
        try:
            members[name] = path.read_bytes()
        except OSError as exc:
            raise Lfm2CrevfCheckpointError(
                f"Could not read local package artifact {name!r}."
            ) from exc
    config = _strict_json_bytes(members[CONFIG_FILENAME], CONFIG_FILENAME)
    generation = _strict_json_bytes(
        members[GENERATION_CONFIG_FILENAME],
        GENERATION_CONFIG_FILENAME,
    )
    _strict_json_bytes(members["tokenizer.json"], "tokenizer.json")
    for name in (
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        if name in members:
            _strict_json_bytes(members[name], name)
    if "chat_template.jinja" in members:
        try:
            members["chat_template.jinja"].decode("utf-8")
        except UnicodeError as exc:
            raise Lfm2CrevfCheckpointFormatError(
                "chat_template.jinja is not UTF-8."
            ) from exc
    _validate_config(config)
    return members, config, generation


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("model_type") != "lfm2"
        or config.get("tie_word_embeddings") is not True
        or config.get("_name_or_path") not in {None, ""}
        or "auto_map" in config
    ):
        raise Lfm2CrevfCheckpointFormatError(
            "The sealed config must be a local, tied LFM2 configuration."
        )
    for name in ("vocab_size", "hidden_size", "num_hidden_layers"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Lfm2CrevfCheckpointFormatError(
                f"The sealed LFM2 config has invalid {name!r}."
            )
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != config[
        "num_hidden_layers"
    ]:
        raise Lfm2CrevfCheckpointFormatError(
            "The sealed LFM2 layer inventory is inconsistent."
        )


def _artifact_entry(raw: bytes, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "size_bytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }


def _tokenizer_fingerprint(
    files: Mapping[str, Mapping[str, Any]],
) -> str:
    tokenizer = [
        {
            "file": name,
            "size_bytes": files[name]["size_bytes"],
            "sha256": files[name]["sha256"],
        }
        for name in TOKENIZER_ARTIFACTS
        if name in files
    ]
    return canonical_sha256(tokenizer)


def _validate_auxiliary_kinds(runtime: CrevfPackedRuntime) -> None:
    for name in runtime.auxiliary_names:
        kind = runtime.auxiliary(name).kind
        if kind not in NONEXECUTING_AUXILIARY_KINDS:
            raise Lfm2CrevfCheckpointFormatError(
                f"CREVF auxiliary {name!r} uses unsupported executable "
                f"kind {kind!r}; this checkpoint layer does not implement "
                "SERQ or writer execution semantics."
            )


def _runtime_inventory(runtime: CrevfPackedRuntime) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for name in runtime.tensor_names:
        store = runtime.tensor(name)
        tensors[name] = {
            "role": store.role,
            "shape": list(store.shape),
            "source_numel": store.source_numel,
            "group_size": store.group_size,
            "group_count": store.group_count,
            "exact_only": store.exact_only,
            "decoded_code_bytes": store.decoded_code_bytes,
            "persistent_encoded_bytes": store.persistent_encoded_bytes,
            "decoded_full_state_bytes": store.decoded_full_state_bytes,
        }
    auxiliaries: dict[str, Any] = {}
    for name in runtime.auxiliary_names:
        store = runtime.auxiliary(name)
        payload = store.payload_bytes()
        auxiliaries[name] = {
            "kind": store.kind,
            "payload_bytes": len(payload),
            "payload_sha256": _sha256_bytes(payload),
            "persistent_encoded_bytes": store.persistent_encoded_bytes,
            "decoded_full_state_bytes": store.decoded_full_state_bytes,
            "execution": "opaque_nonexecuting",
        }
    return {
        "tensors": tensors,
        "auxiliaries": auxiliaries,
        "tensor_names": list(runtime.tensor_names),
        "auxiliary_names": list(runtime.auxiliary_names),
        "persistent_encoded_bytes": runtime.persistent_encoded_bytes,
        "decoded_full_state_bytes": runtime.decoded_full_state_bytes,
        "maximum_tensor_decode_transient_bytes": (
            runtime.maximum_tensor_decode_transient_bytes
        ),
        "maximum_layer_decode_transient_bytes": (
            runtime.maximum_layer_decode_transient_bytes
        ),
        "layer_decode_transient_bytes": [
            [name, value] for name, value in runtime.layer_decode_transient_bytes
        ],
    }


def _write_new(path: Path, raw: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(raw)
    except OSError as exc:
        raise Lfm2CrevfCheckpointError(
            f"Could not write checkpoint artifact {path.name!r}."
        ) from exc


def _stable_manifest(
    *,
    other_bytes: int,
    factory: Any,
) -> tuple[dict[str, Any], bytes]:
    total = other_bytes
    for _ in range(32):
        manifest_bytes = max(0, total - other_bytes)
        core = factory(total, manifest_bytes)
        document = {**core, "manifest_sha256": canonical_sha256(core)}
        raw = canonical_json_bytes(document) + b"\n"
        actual = other_bytes + len(raw)
        if actual == total:
            return document, raw
        total = actual
    raise Lfm2CrevfCheckpointError(
        "CREVF checkpoint manifest byte accounting did not converge."
    )


def _decode_runtime(
    artifact: bytes,
    *,
    max_tensor_decode_bytes: int,
    max_auxiliary_payload_bytes: int,
    max_persistent_encoded_bytes: int,
    max_decoded_full_state_bytes: int,
    max_layer_transient_bytes: int,
) -> CrevfPackedRuntime:
    try:
        return decode_crevf_artifact(
            artifact,
            max_tensor_decode_bytes=max_tensor_decode_bytes,
            max_auxiliary_payload_bytes=max_auxiliary_payload_bytes,
            max_persistent_encoded_bytes=max_persistent_encoded_bytes,
            max_decoded_full_state_bytes=max_decoded_full_state_bytes,
            max_layer_transient_bytes=max_layer_transient_bytes,
        )
    except CrevfBudgetError as exc:
        raise Lfm2CrevfCheckpointBudgetError(str(exc)) from exc
    except CrevfError as exc:
        raise Lfm2CrevfCheckpointFormatError(str(exc)) from exc


def export_lfm2_crevf_checkpoint(
    destination: str | os.PathLike[str],
    *,
    artifact: bytes,
    local_package_directory: str | os.PathLike[str],
    logical_to_canonical: Mapping[str, str],
    target_reduction: float,
    dense_reference_bytes: int,
    max_tensor_decode_bytes: int | None = None,
    max_auxiliary_payload_bytes: int | None = None,
    max_persistent_encoded_bytes: int | None = None,
    max_decoded_full_state_bytes: int | None = None,
    max_layer_transient_bytes: int | None = None,
) -> dict[str, Any]:
    """Transactionally export one complete authenticated local checkpoint."""

    if not isinstance(artifact, bytes):
        raise Lfm2CrevfCheckpointError("artifact must be immutable bytes.")
    if (
        isinstance(dense_reference_bytes, bool)
        or not isinstance(dense_reference_bytes, int)
        or dense_reference_bytes <= 0
    ):
        raise Lfm2CrevfCheckpointError(
            "dense_reference_bytes must be a positive integer."
        )
    try:
        target = CrevfReductionTarget.from_reduction(target_reduction)
        target_ceiling = target.checkpoint_ceiling(dense_reference_bytes)
    except CrevfError as exc:
        raise Lfm2CrevfCheckpointError(str(exc)) from exc
    aliases = _canonical_aliases(logical_to_canonical)
    source = _safe_source_directory(local_package_directory)
    members, config, _generation = _read_local_package(source)
    limits = {
        "max_tensor_decode_bytes": (
            _positive_limit(max_tensor_decode_bytes, "max_tensor_decode_bytes")
            or dense_reference_bytes
        ),
        "max_auxiliary_payload_bytes": (
            _positive_limit(
                max_auxiliary_payload_bytes,
                "max_auxiliary_payload_bytes",
            )
            or min(dense_reference_bytes, _DEFAULT_MAX_AUXILIARY_PAYLOAD_BYTES)
        ),
        "max_persistent_encoded_bytes": _minimum_limit(
            target_ceiling,
            _positive_limit(
                max_persistent_encoded_bytes,
                "max_persistent_encoded_bytes",
            ),
        ),
        "max_decoded_full_state_bytes": (
            _positive_limit(
                max_decoded_full_state_bytes,
                "max_decoded_full_state_bytes",
            )
            or dense_reference_bytes
            + min(dense_reference_bytes, _DEFAULT_MAX_AUXILIARY_PAYLOAD_BYTES)
        ),
        "max_layer_transient_bytes": (
            _positive_limit(
                max_layer_transient_bytes,
                "max_layer_transient_bytes",
            )
            or dense_reference_bytes
        ),
    }
    runtime = _decode_runtime(artifact, **limits)
    _validate_auxiliary_kinds(runtime)
    if set(aliases.values()) != set(runtime.tensor_names):
        raise Lfm2CrevfCheckpointError(
            "Canonical logical aliases differ from the CREVF tensor inventory."
        )
    runtime_inventory = _runtime_inventory(runtime)
    source_tensor_bytes = sum(
        runtime.tensor(name).source_numel * 2 for name in runtime.tensor_names
    )
    source_nonweight_bytes = sum(len(raw) for raw in members.values())
    observed_reference = source_tensor_bytes + source_nonweight_bytes
    if observed_reference != dense_reference_bytes:
        raise Lfm2CrevfCheckpointError(
            "dense_reference_bytes must equal unique BF16 tensor bytes plus "
            "the exact local config/generation/tokenizer bytes."
        )
    if runtime.persistent_encoded_bytes > target_ceiling:
        raise Lfm2CrevfCheckpointBudgetError(
            "CREVF persistent encoded state exceeds the exact target ceiling."
        )

    output = Path(destination).expanduser().absolute()
    if output.exists():
        raise Lfm2CrevfCheckpointError(
            f"Refusing to replace existing destination {output}."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        file_entries: dict[str, dict[str, Any]] = {}
        for name in sorted(members):
            raw = members[name]
            _write_new(temporary / name, raw)
            kind = (
                "config"
                if name == CONFIG_FILENAME
                else "generation_config"
                if name == GENERATION_CONFIG_FILENAME
                else "tokenizer"
            )
            file_entries[name] = _artifact_entry(raw, kind)
        _write_new(temporary / ARTIFACT_FILENAME, artifact)
        file_entries[ARTIFACT_FILENAME] = _artifact_entry(
            artifact,
            "crevf_artifact",
        )
        file_entries = dict(sorted(file_entries.items()))
        other_bytes = sum(entry["size_bytes"] for entry in file_entries.values())
        tokenizer_files = [
            name for name in TOKENIZER_ARTIFACTS if name in file_entries
        ]
        tokenizer_fingerprint = _tokenizer_fingerprint(file_entries)

        def factory(total: int, manifest_bytes: int) -> dict[str, Any]:
            return {
                "schema": CHECKPOINT_SCHEMA,
                "files": file_entries,
                "target": {
                    "reduction_ppm": target.reduction_ppm,
                    "requested_reduction": target.reduction,
                    "dense_reference_bytes": dense_reference_bytes,
                    "target_checkpoint_ceiling_bytes": target_ceiling,
                    "physical_cap_bytes": target_ceiling,
                    "persistent_cap_bytes": limits[
                        "max_persistent_encoded_bytes"
                    ],
                },
                "topology": {
                    "model_type": "lfm2",
                    "architectures": config.get("architectures", []),
                    "vocab_size": config["vocab_size"],
                    "hidden_size": config["hidden_size"],
                    "num_hidden_layers": config["num_hidden_layers"],
                    "layer_types": config["layer_types"],
                    "tie_word_embeddings": True,
                    "logical_to_canonical": aliases,
                    "logical_parameter_count": len(aliases),
                    "canonical_tensor_count": len(runtime.tensor_names),
                    "tied_embedding_name": "model.embed_tokens.weight",
                    "tied_head_name": "lm_head.weight",
                    "tied_vocabulary_stored_once": True,
                },
                "runtime": {
                    "artifact_file": ARTIFACT_FILENAME,
                    "artifact_sha256": runtime.artifact_sha256,
                    "source_dtype": "torch.bfloat16",
                    "inventory": runtime_inventory,
                    "decode_limits": limits,
                    "installer": "install_lfm2_crevf_runtime",
                    "executable_auxiliary_kinds": [],
                },
                "tokenizer": {
                    "files": tokenizer_files,
                    "fingerprint": tokenizer_fingerprint,
                    "required_for_reload": True,
                    "local_backend_only": True,
                },
                "storage": {
                    "manifest_bytes": manifest_bytes,
                    "config_generation_tokenizer_bytes": source_nonweight_bytes,
                    "source_unique_bf16_tensor_bytes": source_tensor_bytes,
                    "crevf_artifact_bytes": len(artifact),
                    "auxiliary_payload_bytes": sum(
                        entry["payload_bytes"]
                        for entry in runtime_inventory["auxiliaries"].values()
                    ),
                    "checkpoint_physical_bytes": total,
                    "persistent_encoded_bytes": (
                        runtime.persistent_encoded_bytes
                    ),
                    "decoded_full_state_bytes": (
                        runtime.decoded_full_state_bytes
                    ),
                    "maximum_tensor_decode_transient_bytes": (
                        runtime.maximum_tensor_decode_transient_bytes
                    ),
                    "maximum_layer_decode_transient_bytes": (
                        runtime.maximum_layer_decode_transient_bytes
                    ),
                    "physical_headroom_bytes": target_ceiling - total,
                    "persistent_headroom_bytes": (
                        limits["max_persistent_encoded_bytes"]
                        - runtime.persistent_encoded_bytes
                    ),
                    "hidden_dense_checkpoint_copy": False,
                    "external_weight_sidecars": [],
                },
                "serialization": {
                    "canonical_manifest_json": True,
                    "python_pickle": False,
                    "serialized_modules": False,
                    "dense_safetensors": False,
                },
                "standalone": True,
                "parent_model_required_for_reload": False,
                "from_pretrained_used": False,
                "network_access": False,
                "training_used": False,
            }

        manifest, manifest_raw = _stable_manifest(
            other_bytes=other_bytes,
            factory=factory,
        )
        if manifest["storage"]["checkpoint_physical_bytes"] > target_ceiling:
            raise Lfm2CrevfCheckpointBudgetError(
                "Complete CREVF checkpoint directory exceeds the exact target "
                "physical ceiling."
            )
        _write_new(temporary / MANIFEST_FILENAME, manifest_raw)
        observed = sum(path.stat().st_size for path in temporary.iterdir())
        if observed != manifest["storage"]["checkpoint_physical_bytes"]:
            raise Lfm2CrevfCheckpointFormatError(
                "Staged directory bytes differ from the sealed manifest."
            )
        os.replace(temporary, output)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _verified_directory(
    checkpoint_directory: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    root = Path(checkpoint_directory).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise Lfm2CrevfCheckpointFormatError(
            f"Checkpoint directory {root} is missing or is a symlink."
        )
    manifest_path = root / MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise Lfm2CrevfCheckpointFormatError(f"Missing {MANIFEST_FILENAME}.")
    try:
        manifest_raw = manifest_path.read_bytes()
    except OSError as exc:
        raise Lfm2CrevfCheckpointFormatError(
            "Could not read the checkpoint manifest."
        ) from exc
    manifest = _strict_json_bytes(manifest_raw, MANIFEST_FILENAME)
    if manifest_raw != canonical_json_bytes(manifest) + b"\n":
        raise Lfm2CrevfCheckpointFormatError(
            "Checkpoint manifest is not canonical JSON."
        )
    _strict_keys(
        manifest,
        {
            "schema",
            "files",
            "target",
            "topology",
            "runtime",
            "tokenizer",
            "storage",
            "serialization",
            "standalone",
            "parent_model_required_for_reload",
            "from_pretrained_used",
            "network_access",
            "training_used",
            "manifest_sha256",
        },
        "manifest",
    )
    if manifest["schema"] != CHECKPOINT_SCHEMA:
        raise Lfm2CrevfCheckpointFormatError(
            "Unsupported CREVF checkpoint schema."
        )
    core = dict(manifest)
    claimed_hash = core.pop("manifest_sha256")
    if (
        not isinstance(claimed_hash, str)
        or len(claimed_hash) != _DIGEST_LENGTH
        or claimed_hash != canonical_sha256(core)
    ):
        raise Lfm2CrevfCheckpointFormatError("Manifest SHA-256 mismatch.")
    if (
        manifest["standalone"] is not True
        or manifest["parent_model_required_for_reload"] is not False
        or manifest["from_pretrained_used"] is not False
        or manifest["network_access"] is not False
        or manifest["training_used"] is not False
    ):
        raise Lfm2CrevfCheckpointFormatError(
            "Manifest does not describe a local-only standalone checkpoint."
        )
    serialization = _strict_keys(
        manifest["serialization"],
        {
            "canonical_manifest_json",
            "python_pickle",
            "serialized_modules",
            "dense_safetensors",
        },
        "serialization",
    )
    if serialization != {
        "canonical_manifest_json": True,
        "python_pickle": False,
        "serialized_modules": False,
        "dense_safetensors": False,
    }:
        raise Lfm2CrevfCheckpointFormatError(
            "Checkpoint serialization policy is unsafe or unsupported."
        )
    files = manifest["files"]
    if not isinstance(files, Mapping) or not files:
        raise Lfm2CrevfCheckpointFormatError(
            "Manifest file inventory is empty."
        )
    if any(
        not isinstance(name, str)
        or Path(name).name != name
        or name == MANIFEST_FILENAME
        for name in files
    ):
        raise Lfm2CrevfCheckpointFormatError(
            "Manifest contains an invalid artifact filename."
        )
    expected_names = {MANIFEST_FILENAME, *files}
    actual_children = tuple(root.iterdir())
    if (
        {path.name for path in actual_children} != expected_names
        or len(actual_children) != len(expected_names)
    ):
        raise Lfm2CrevfCheckpointFormatError(
            "Checkpoint directory contains missing or unclaimed files."
        )
    paths: dict[str, Path] = {}
    for name, entry_value in files.items():
        entry = _strict_keys(
            entry_value,
            {"kind", "size_bytes", "sha256"},
            f"files[{name!r}]",
        )
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise Lfm2CrevfCheckpointFormatError(
                f"Checkpoint artifact {name!r} is missing or is a symlink."
            )
        size = entry["size_bytes"]
        digest = entry["sha256"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or path.stat().st_size != size
            or not isinstance(digest, str)
            or len(digest) != _DIGEST_LENGTH
            or _sha256_file(path) != digest
        ):
            raise Lfm2CrevfCheckpointFormatError(
                f"Checkpoint artifact {name!r} failed size or SHA-256."
            )
        paths[name] = path
    required = {
        CONFIG_FILENAME: "config",
        GENERATION_CONFIG_FILENAME: "generation_config",
        "tokenizer.json": "tokenizer",
        ARTIFACT_FILENAME: "crevf_artifact",
    }
    for name, kind in required.items():
        if name not in files or files[name]["kind"] != kind:
            raise Lfm2CrevfCheckpointFormatError(
                f"Checkpoint lacks required {kind} artifact {name!r}."
            )
    allowed_names = {
        CONFIG_FILENAME,
        GENERATION_CONFIG_FILENAME,
        ARTIFACT_FILENAME,
        *TOKENIZER_ARTIFACTS,
    }
    if set(files).difference(allowed_names):
        raise Lfm2CrevfCheckpointFormatError(
            "Checkpoint claims an unsupported file kind."
        )
    if any(
        files[name]["kind"] != "tokenizer"
        for name in TOKENIZER_ARTIFACTS
        if name in files
    ):
        raise Lfm2CrevfCheckpointFormatError(
            "Tokenizer artifacts have inconsistent file kinds."
        )
    storage = _strict_keys(
        manifest["storage"],
        {
            "manifest_bytes",
            "config_generation_tokenizer_bytes",
            "source_unique_bf16_tensor_bytes",
            "crevf_artifact_bytes",
            "auxiliary_payload_bytes",
            "checkpoint_physical_bytes",
            "persistent_encoded_bytes",
            "decoded_full_state_bytes",
            "maximum_tensor_decode_transient_bytes",
            "maximum_layer_decode_transient_bytes",
            "physical_headroom_bytes",
            "persistent_headroom_bytes",
            "hidden_dense_checkpoint_copy",
            "external_weight_sidecars",
        },
        "storage",
    )
    for name in (
        "manifest_bytes",
        "config_generation_tokenizer_bytes",
        "source_unique_bf16_tensor_bytes",
        "crevf_artifact_bytes",
        "auxiliary_payload_bytes",
        "checkpoint_physical_bytes",
        "persistent_encoded_bytes",
        "decoded_full_state_bytes",
        "maximum_tensor_decode_transient_bytes",
        "maximum_layer_decode_transient_bytes",
        "physical_headroom_bytes",
        "persistent_headroom_bytes",
    ):
        _manifest_int(storage[name], f"storage.{name}")
    checkpoint_bytes = len(manifest_raw) + sum(
        path.stat().st_size for path in paths.values()
    )
    if (
        storage["manifest_bytes"] != len(manifest_raw)
        or storage["checkpoint_physical_bytes"] != checkpoint_bytes
        or storage["crevf_artifact_bytes"]
        != files[ARTIFACT_FILENAME]["size_bytes"]
        or storage["config_generation_tokenizer_bytes"]
        != sum(
            files[name]["size_bytes"]
            for name in files
            if name != ARTIFACT_FILENAME
        )
        or storage["hidden_dense_checkpoint_copy"] is not False
        or storage["external_weight_sidecars"] != []
    ):
        raise Lfm2CrevfCheckpointFormatError(
            "Complete checkpoint byte accounting is inconsistent."
        )
    target_data = _strict_keys(
        manifest["target"],
        {
            "reduction_ppm",
            "requested_reduction",
            "dense_reference_bytes",
            "target_checkpoint_ceiling_bytes",
            "physical_cap_bytes",
            "persistent_cap_bytes",
        },
        "target",
    )
    for name in (
        "reduction_ppm",
        "dense_reference_bytes",
        "target_checkpoint_ceiling_bytes",
        "physical_cap_bytes",
        "persistent_cap_bytes",
    ):
        _manifest_int(target_data[name], f"target.{name}")
    try:
        target = CrevfReductionTarget(target_data["reduction_ppm"])
        ceiling = target.checkpoint_ceiling(target_data["dense_reference_bytes"])
    except (CrevfError, TypeError) as exc:
        raise Lfm2CrevfCheckpointFormatError(
            "Checkpoint target arithmetic is invalid."
        ) from exc
    if (
        target_data["requested_reduction"] != target.reduction
        or target_data["target_checkpoint_ceiling_bytes"] != ceiling
        or target_data["physical_cap_bytes"] != ceiling
        or target_data["persistent_cap_bytes"] <= 0
        or target_data["persistent_cap_bytes"] > ceiling
        or storage["checkpoint_physical_bytes"] > ceiling
        or storage["persistent_encoded_bytes"]
        > target_data["persistent_cap_bytes"]
        or storage["physical_headroom_bytes"]
        != ceiling - storage["checkpoint_physical_bytes"]
        or storage["persistent_headroom_bytes"]
        != target_data["persistent_cap_bytes"]
        - storage["persistent_encoded_bytes"]
    ):
        raise Lfm2CrevfCheckpointBudgetError(
            "Checkpoint violates its exact physical or persistent target cap."
        )
    tokenizer = _strict_keys(
        manifest["tokenizer"],
        {
            "files",
            "fingerprint",
            "required_for_reload",
            "local_backend_only",
        },
        "tokenizer",
    )
    expected_tokenizer_files = [
        name for name in TOKENIZER_ARTIFACTS if name in files
    ]
    if (
        tokenizer["files"] != expected_tokenizer_files
        or tokenizer["fingerprint"] != _tokenizer_fingerprint(files)
        or tokenizer["required_for_reload"] is not True
        or tokenizer["local_backend_only"] is not True
    ):
        raise Lfm2CrevfCheckpointFormatError(
            "Tokenizer inventory or fingerprint is inconsistent."
        )
    return root, manifest, paths


def _verify_runtime(
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    max_persistent_encoded_bytes: int | None = None,
    max_decoded_full_state_bytes: int | None = None,
    max_layer_transient_bytes: int | None = None,
    max_tensor_decode_bytes: int | None = None,
    max_auxiliary_payload_bytes: int | None = None,
) -> CrevfPackedRuntime:
    runtime_meta = _strict_keys(
        manifest["runtime"],
        {
            "artifact_file",
            "artifact_sha256",
            "source_dtype",
            "inventory",
            "decode_limits",
            "installer",
            "executable_auxiliary_kinds",
        },
        "runtime",
    )
    if (
        runtime_meta["artifact_file"] != ARTIFACT_FILENAME
        or runtime_meta["source_dtype"] != "torch.bfloat16"
        or runtime_meta["installer"] != "install_lfm2_crevf_runtime"
        or runtime_meta["executable_auxiliary_kinds"] != []
    ):
        raise Lfm2CrevfCheckpointFormatError(
            "Runtime declaration is unsupported."
        )
    sealed_limits = _strict_keys(
        runtime_meta["decode_limits"],
        {
            "max_tensor_decode_bytes",
            "max_auxiliary_payload_bytes",
            "max_persistent_encoded_bytes",
            "max_decoded_full_state_bytes",
            "max_layer_transient_bytes",
        },
        "runtime.decode_limits",
    )
    supplied = {
        "max_tensor_decode_bytes": _positive_limit(
            max_tensor_decode_bytes,
            "max_tensor_decode_bytes",
        ),
        "max_auxiliary_payload_bytes": _positive_limit(
            max_auxiliary_payload_bytes,
            "max_auxiliary_payload_bytes",
        ),
        "max_persistent_encoded_bytes": _positive_limit(
            max_persistent_encoded_bytes,
            "max_persistent_encoded_bytes",
        ),
        "max_decoded_full_state_bytes": _positive_limit(
            max_decoded_full_state_bytes,
            "max_decoded_full_state_bytes",
        ),
        "max_layer_transient_bytes": _positive_limit(
            max_layer_transient_bytes,
            "max_layer_transient_bytes",
        ),
    }
    limits: dict[str, int] = {}
    for name, sealed in sealed_limits.items():
        if isinstance(sealed, bool) or not isinstance(sealed, int) or sealed <= 0:
            raise Lfm2CrevfCheckpointFormatError(
                f"Sealed decode limit {name!r} is invalid."
            )
        limits[name] = _minimum_limit(sealed, supplied[name])  # type: ignore[assignment]
    try:
        artifact = paths[ARTIFACT_FILENAME].read_bytes()
    except OSError as exc:
        raise Lfm2CrevfCheckpointFormatError(
            "Could not read the CREVF artifact."
        ) from exc
    runtime = _decode_runtime(artifact, **limits)
    _validate_auxiliary_kinds(runtime)
    inventory = _runtime_inventory(runtime)
    storage = manifest["storage"]
    if (
        runtime.artifact_sha256 != runtime_meta["artifact_sha256"]
        or inventory != runtime_meta["inventory"]
        or runtime.persistent_encoded_bytes
        != storage["persistent_encoded_bytes"]
        or runtime.decoded_full_state_bytes
        != storage["decoded_full_state_bytes"]
        or runtime.maximum_tensor_decode_transient_bytes
        != storage["maximum_tensor_decode_transient_bytes"]
        or runtime.maximum_layer_decode_transient_bytes
        != storage["maximum_layer_decode_transient_bytes"]
        or sum(
            value["payload_bytes"]
            for value in inventory["auxiliaries"].values()
        )
        != storage["auxiliary_payload_bytes"]
    ):
        raise Lfm2CrevfCheckpointFormatError(
            "Decoded CREVF runtime differs from the sealed inventory."
        )
    topology = manifest["topology"]
    aliases_value = topology.get("logical_to_canonical")
    try:
        aliases = _canonical_aliases(aliases_value)
    except Lfm2CrevfCheckpointError as exc:
        raise Lfm2CrevfCheckpointFormatError(
            "Topology logical aliases are invalid."
        ) from exc
    if set(aliases.values()) != set(runtime.tensor_names):
        raise Lfm2CrevfCheckpointFormatError(
            "Runtime tensor inventory differs from topology aliases."
        )
    source_tensor_bytes = sum(
        runtime.tensor(name).source_numel * 2 for name in runtime.tensor_names
    )
    if (
        source_tensor_bytes != storage["source_unique_bf16_tensor_bytes"]
        or source_tensor_bytes
        + storage["config_generation_tokenizer_bytes"]
        != manifest["target"]["dense_reference_bytes"]
    ):
        raise Lfm2CrevfCheckpointFormatError(
            "Dense reference accounting differs from the decoded tensors."
        )
    return runtime


def _logical_parameters_and_aliases(
    model: nn.Module,
) -> tuple[dict[str, nn.Parameter], dict[str, str]]:
    try:
        logical = dict(
            model.named_parameters(recurse=True, remove_duplicate=False)
        )
    except TypeError as exc:  # pragma: no cover
        raise Lfm2CrevfCheckpointError(
            "PyTorch must support remove_duplicate=False."
        ) from exc
    if not logical:
        raise Lfm2CrevfCheckpointFormatError(
            "Fresh meta LFM2 has no logical Parameters."
        )
    groups: dict[int, list[str]] = {}
    for name, parameter in logical.items():
        groups.setdefault(id(parameter), []).append(name)
    aliases: dict[str, str] = {}
    for names in groups.values():
        canonical = (
            "model.embed_tokens.weight"
            if "model.embed_tokens.weight" in names
            else sorted(names)[0]
        )
        aliases.update((name, canonical) for name in names)
    return logical, dict(sorted(aliases.items()))


def _load_tokenizer_verified(
    root: Path,
    manifest: Mapping[str, Any],
) -> Any:
    try:
        import tokenizers
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise Lfm2CrevfCheckpointError(
            "Local tokenizer reload requires tokenizers and Transformers."
        ) from exc
    try:
        backend = tokenizers.Tokenizer.from_str(
            (root / "tokenizer.json").read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise Lfm2CrevfCheckpointFormatError(
            "Authenticated tokenizer.json could not be loaded."
        ) from exc
    kwargs: dict[str, Any] = {}
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        path = root / name
        if path.is_file():
            kwargs.update(_strict_json_bytes(path.read_bytes(), name))
    for name in (
        "tokenizer_class",
        "backend",
        "is_local",
        "use_fast",
        "sp_model_kwargs",
    ):
        kwargs.pop(name, None)
    for name in (
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ):
        value = kwargs.get(name)
        if isinstance(value, Mapping) and "content" in value:
            kwargs[name] = str(value["content"])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, **kwargs)
    chat_template = root / "chat_template.jinja"
    if chat_template.is_file():
        tokenizer.chat_template = chat_template.read_text(encoding="utf-8")
    setattr(
        tokenizer,
        "neural_forge_tokenizer_fingerprint",
        manifest["tokenizer"]["fingerprint"],
    )
    return tokenizer


def load_lfm2_crevf_tokenizer(
    checkpoint_directory: str | os.PathLike[str],
) -> Any:
    """Authenticate and restore the tokenizer entirely from local files."""

    root, manifest, _paths = _verified_directory(checkpoint_directory)
    return _load_tokenizer_verified(root, manifest)


def _restore_rotary_bf16(
    model: nn.Module,
    config: Any,
    target_device: torch.device,
) -> int:
    try:
        rotary = model.model.rotary_emb
        restored = type(rotary)(config, device=target_device).to(
            dtype=torch.bfloat16
        )
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise Lfm2CrevfCheckpointFormatError(
            "Fresh LFM2 rotary buffers could not be restored locally."
        ) from exc
    model.model.rotary_emb = restored
    total = 0
    for _name, value in restored.named_buffers(
        recurse=True,
        remove_duplicate=True,
    ):
        if value.device.type == "meta" or (
            value.numel() and value.dtype != torch.bfloat16
        ):
            raise Lfm2CrevfCheckpointFormatError(
                "Fresh LFM2 rotary buffers are not materialized source BF16."
            )
        total += int(value.numel()) * int(value.element_size())
    return total


def _verify_installed_state(
    model: nn.Module,
    runtime: CrevfPackedRuntime,
) -> tuple[int, int]:
    parameters = tuple(model.named_parameters(recurse=True))
    if any(parameter.numel() for _name, parameter in parameters):
        raise Lfm2CrevfCheckpointFormatError(
            "Loaded CREVF model retained nonempty Parameters."
        )
    if tuple(name for name, _parameter in parameters) != (
        "neural_forge_device_sentinel",
    ):
        raise Lfm2CrevfCheckpointFormatError(
            "Loaded CREVF model has a hidden learned Parameter surface."
        )
    state = model.state_dict()
    if any(value.numel() for value in state.values()):
        raise Lfm2CrevfCheckpointFormatError(
            "Loaded CREVF model retained hidden dense persistent state."
        )
    rotary_bytes = 0
    packed_bytes = 0
    for name, value in model.named_buffers(
        recurse=True,
        remove_duplicate=True,
    ):
        if value.device.type == "meta":
            raise Lfm2CrevfCheckpointFormatError(
                f"Loaded runtime buffer {name!r} remains on meta."
            )
        nbytes = int(value.numel()) * int(value.element_size())
        if name.startswith("neural_forge_crevf_runtime."):
            packed_bytes += nbytes
        elif name.startswith("model.rotary_emb."):
            if value.numel() and value.dtype != torch.bfloat16:
                raise Lfm2CrevfCheckpointFormatError(
                    "Rotary state differs from source BF16."
                )
            rotary_bytes += nbytes
        elif value.numel():
            raise Lfm2CrevfCheckpointFormatError(
                f"Loaded runtime retained unclaimed buffer {name!r}."
            )
    if packed_bytes != runtime.persistent_encoded_bytes:
        raise Lfm2CrevfCheckpointFormatError(
            "Installed packed buffers differ from persistent accounting."
        )
    return packed_bytes, rotary_bytes


def load_lfm2_crevf_checkpoint(
    checkpoint_directory: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    max_checkpoint_bytes: int | None = None,
    max_persistent_encoded_bytes: int | None = None,
    max_decoded_full_state_bytes: int | None = None,
    max_layer_transient_bytes: int | None = None,
    max_tensor_decode_bytes: int | None = None,
    max_auxiliary_payload_bytes: int | None = None,
) -> nn.Module:
    """Fresh-load a sealed LFM2 shell and install its bounded CREVF runtime."""

    root, manifest, paths = _verified_directory(checkpoint_directory)
    checkpoint_limit = _positive_limit(
        max_checkpoint_bytes,
        "max_checkpoint_bytes",
    )
    checkpoint_bytes = manifest["storage"]["checkpoint_physical_bytes"]
    if checkpoint_limit is not None and checkpoint_bytes > checkpoint_limit:
        raise Lfm2CrevfCheckpointBudgetError(
            f"Checkpoint requires {checkpoint_bytes} physical bytes, exceeding "
            f"the caller's {checkpoint_limit}-byte limit."
        )
    runtime = _verify_runtime(
        manifest,
        paths,
        max_persistent_encoded_bytes=max_persistent_encoded_bytes,
        max_decoded_full_state_bytes=max_decoded_full_state_bytes,
        max_layer_transient_bytes=max_layer_transient_bytes,
        max_tensor_decode_bytes=max_tensor_decode_bytes,
        max_auxiliary_payload_bytes=max_auxiliary_payload_bytes,
    )
    config_data = _strict_json_bytes(
        paths[CONFIG_FILENAME].read_bytes(),
        CONFIG_FILENAME,
    )
    generation_data = _strict_json_bytes(
        paths[GENERATION_CONFIG_FILENAME].read_bytes(),
        GENERATION_CONFIG_FILENAME,
    )
    _validate_config(config_data)
    topology = _strict_keys(
        manifest["topology"],
        {
            "model_type",
            "architectures",
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "layer_types",
            "tie_word_embeddings",
            "logical_to_canonical",
            "logical_parameter_count",
            "canonical_tensor_count",
            "tied_embedding_name",
            "tied_head_name",
            "tied_vocabulary_stored_once",
        },
        "topology",
    )
    if (
        topology["model_type"] != "lfm2"
        or topology["architectures"] != config_data.get("architectures", [])
        or topology["vocab_size"] != config_data["vocab_size"]
        or topology["hidden_size"] != config_data["hidden_size"]
        or topology["num_hidden_layers"] != config_data["num_hidden_layers"]
        or topology["layer_types"] != config_data["layer_types"]
        or topology["tie_word_embeddings"] is not True
        or topology["tied_embedding_name"] != "model.embed_tokens.weight"
        or topology["tied_head_name"] != "lm_head.weight"
        or topology["tied_vocabulary_stored_once"] is not True
    ):
        raise Lfm2CrevfCheckpointFormatError(
            "Sealed LFM2 topology differs from config."
        )
    aliases = _canonical_aliases(topology["logical_to_canonical"])
    if (
        topology["logical_parameter_count"] != len(aliases)
        or topology["canonical_tensor_count"] != len(runtime.tensor_names)
    ):
        raise Lfm2CrevfCheckpointFormatError(
            "Topology parameter counts differ from the inventories."
        )
    try:
        from transformers import GenerationConfig, Lfm2Config, Lfm2ForCausalLM
    except ImportError as exc:
        raise Lfm2CrevfCheckpointError(
            "Loading requires Transformers with LFM2 support."
        ) from exc
    target_device = torch.device(device)
    if target_device.type == "meta":
        raise Lfm2CrevfCheckpointError(
            "device='meta' is not a materialized runtime target."
        )
    config = Lfm2Config.from_dict(config_data)
    with torch.device("meta"):
        model = Lfm2ForCausalLM(config)
    logical, observed_aliases = _logical_parameters_and_aliases(model)
    if observed_aliases != aliases:
        raise Lfm2CrevfCheckpointFormatError(
            "Fresh LFM2 logical aliases differ from the sealed topology."
        )
    for logical_name, parameter in logical.items():
        canonical = aliases[logical_name]
        if tuple(parameter.shape) != runtime.tensor(canonical).shape:
            raise Lfm2CrevfCheckpointFormatError(
                f"Fresh LFM2 shape for {logical_name!r} differs from CREVF."
            )
    rotary_bytes = _restore_rotary_bf16(model, config, target_device)
    runtime.to(device=target_device)
    report: Lfm2CrevfInstallReport = install_lfm2_crevf_runtime(
        model,
        runtime,
        aliases,
    )
    model.generation_config = GenerationConfig.from_dict(generation_data)
    tokenizer = _load_tokenizer_verified(root, manifest)
    packed_bytes, observed_rotary_bytes = _verify_installed_state(model, runtime)
    if observed_rotary_bytes != rotary_bytes:
        raise Lfm2CrevfCheckpointFormatError(
            "Rotary buffer accounting changed during CREVF installation."
        )
    setattr(model, "neural_forge_tokenizer", tokenizer)
    setattr(model, "neural_forge_checkpoint_manifest", manifest)
    setattr(
        model,
        "neural_forge_load_report",
        {
            "manifest_sha256": manifest["manifest_sha256"],
            "constructed_on_meta": True,
            "checkpoint_physical_bytes": checkpoint_bytes,
            "persistent_encoded_bytes": packed_bytes,
            "rotary_bf16_bytes": observed_rotary_bytes,
            "persistent_dense_parameter_elements": 0,
            "hidden_dense_state_bytes": 0,
            "zero_element_device_sentinel": True,
            "tied_vocabulary_stored_once": (
                report.tied_vocabulary_stored_once
            ),
            "tokenizer_restored_locally": True,
            "generation_config_restored_locally": True,
            "from_pretrained_used": False,
            "network_access": False,
            "parent_model_required": False,
            "training_used": False,
        },
    )
    return model


def inspect_lfm2_crevf_checkpoint(
    checkpoint_directory: str | os.PathLike[str],
) -> dict[str, Any]:
    """Authenticate the directory and report exact target/runtime accounting."""

    root, manifest, paths = _verified_directory(checkpoint_directory)
    runtime = _verify_runtime(manifest, paths)
    storage = manifest["storage"]
    target = manifest["target"]
    return {
        "checkpoint_directory": str(root),
        "manifest_sha256": manifest["manifest_sha256"],
        "requested_reduction": target["requested_reduction"],
        "realized_physical_reduction": (
            1.0
            - storage["checkpoint_physical_bytes"]
            / target["dense_reference_bytes"]
        ),
        "dense_reference_bytes": target["dense_reference_bytes"],
        "target_checkpoint_ceiling_bytes": target[
            "target_checkpoint_ceiling_bytes"
        ],
        "checkpoint_physical_bytes": storage["checkpoint_physical_bytes"],
        "persistent_encoded_bytes": runtime.persistent_encoded_bytes,
        "decoded_full_state_bytes": runtime.decoded_full_state_bytes,
        "maximum_layer_decode_transient_bytes": (
            runtime.maximum_layer_decode_transient_bytes
        ),
        "tensor_count": len(runtime.tensor_names),
        "auxiliary_count": len(runtime.auxiliary_names),
        "tied_vocabulary_stored_once": True,
        "standalone": True,
    }


__all__ = [
    "ARTIFACT_FILENAME",
    "CHECKPOINT_SCHEMA",
    "CONFIG_FILENAME",
    "GENERATION_CONFIG_FILENAME",
    "Lfm2CrevfCheckpointBudgetError",
    "Lfm2CrevfCheckpointError",
    "Lfm2CrevfCheckpointFormatError",
    "MANIFEST_FILENAME",
    "NONEXECUTING_AUXILIARY_KINDS",
    "TOKENIZER_ARTIFACTS",
    "export_lfm2_crevf_checkpoint",
    "inspect_lfm2_crevf_checkpoint",
    "load_lfm2_crevf_checkpoint",
    "load_lfm2_crevf_tokenizer",
]
