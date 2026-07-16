"""Strict content-addressed persistence for frozen teacher-token replay.

The cache deliberately knows nothing about candidate reservoirs, held-out
prompts, behavior targets, or recovery selection.  Its only payload is one
strictly authenticated teacher-token replay and the identities needed to
prove that replay is safe to reuse.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from neural_forge import hf_teacher_token_replay


REQUEST_SCHEMA = "neural-forge.teacher-replay-cache-request.v1"
ENVELOPE_SCHEMA = "neural-forge.teacher-replay-cache-envelope.v1"
PRODUCER_MANIFEST_SCHEMA = "neural-forge.teacher-replay-producers.v1"
CACHE_FILENAME = "cache_envelope.json"
_RESERVATION_SUFFIX = ".publish-reservation"
_RESERVATION_WAIT_SECONDS = 5.0
_RESERVATION_POLL_SECONDS = 0.01

_SHA256_HEX = frozenset("0123456789abcdef")
_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "source_artifact_tree",
        "calibration",
        "tokenizer_fingerprints",
        "teacher_parameters",
        "teacher_parameters_sha256",
        "local_files_only",
        "trust_remote_code",
        "producer_byte_hash_manifest",
        "runtime_identity",
        "request_sha256",
    }
)
_SOURCE_TREE_FIELDS = frozenset(
    {
        "expected_tree_sha256",
        "actual_tree_sha256",
        "file_count",
        "total_bytes",
    }
)
_CALIBRATION_FIELDS = frozenset(
    {
        "source_file_sha256",
        "ordered_text_count",
        "ordered_text_manifest_sha256",
    }
)
_TOKENIZER_FIELDS = frozenset({"source", "candidate"})
_PRODUCER_FIELDS = frozenset({"schema", "files", "sha256"})
_PRODUCER_FILE_FIELDS = frozenset({"role", "filename", "byte_count", "sha256"})
_RUNTIME_FIELDS = frozenset(
    {
        "python_implementation",
        "python_version",
        "torch_version",
        "transformers_version",
        "requested_device",
        "cuda_available",
        "cuda_runtime_version",
        "cuda_device_index",
        "cuda_device_name",
        "cuda_device_capability",
        "cuda_device_total_memory",
        "cudnn_version",
        "deterministic_algorithms_enabled",
        "deterministic_debug_mode",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cudnn_allow_tf32",
        "cuda_matmul_allow_tf32",
        "cublas_workspace_config",
        "sha256",
    }
)
_ENVELOPE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "contains_sealed_data",
        "sealed_data_accessed",
        "publication_authorized",
        "request",
        "request_sha256",
        "teacher_token_replay",
        "teacher_payload_sha256",
        "source_model_fingerprint",
        "source_model_state_dict_sha256",
        "envelope_sha256",
    }
)


class TeacherReplayCacheError(RuntimeError):
    """A replay cache request, entry, or publication failed authentication."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise TeacherReplayCacheError(f"{name} must be a lowercase SHA-256.")
    return value


def _strict_fields(value: Any, fields: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise TeacherReplayCacheError(
            f"{name} fields changed; expected={sorted(fields)!r}, actual={actual!r}."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TeacherReplayCacheError(f"{name} must be a positive integer.")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_file(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TeacherReplayCacheError(
                    f"Duplicate JSON key {key!r} in {path.name}."
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                TeacherReplayCacheError(
                    f"Non-finite JSON constant {constant!r} in {path.name}."
                )
            ),
        )
        _canonical_bytes(value)
    except TeacherReplayCacheError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise TeacherReplayCacheError(f"Cannot read strict JSON from {path}.") from error
    if not isinstance(value, dict):
        raise TeacherReplayCacheError(f"{path.name} must contain one JSON object.")
    return value


def producer_byte_hash_manifest(
    files: Sequence[tuple[str, Path | str]],
) -> dict[str, Any]:
    """Hash the exact producer bytes that define replay construction semantics."""

    expected_roles = {
        "prepare_recovery_inputs",
        "hf_teacher_token_replay",
        "hf_recovery",
        "hf_teacher_replay_cache",
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for role, raw_path in files:
        if role in seen or role not in expected_roles:
            raise TeacherReplayCacheError("Producer roles are missing, duplicated, or unknown.")
        seen.add(role)
        path = Path(raw_path).expanduser().resolve()
        if path.is_symlink() or not path.is_file():
            raise TeacherReplayCacheError(
                f"Producer {role!r} must be an existing real file."
            )
        rows.append(
            {
                "role": role,
                "filename": path.name,
                "byte_count": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if seen != expected_roles:
        raise TeacherReplayCacheError("The complete producer byte manifest is required.")
    rows.sort(key=lambda row: row["role"])
    body = {"schema": PRODUCER_MANIFEST_SCHEMA, "files": rows}
    return {**body, "sha256": _canonical_sha256(body)}


def _optional_call(owner: Any, name: str) -> Any:
    function = getattr(owner, name, None)
    if not callable(function):
        return None
    try:
        return function()
    except (RuntimeError, TypeError, ValueError):
        return None


def runtime_identity(torch: Any, transformers: Any, requested_device: str) -> dict[str, Any]:
    """Freeze runtime/device state that can alter deterministic replay bytes."""

    if not isinstance(requested_device, str) or not requested_device.strip():
        raise TeacherReplayCacheError("requested_device must be a nonblank string.")
    cuda = getattr(torch, "cuda", None)
    cuda_available = bool(_optional_call(cuda, "is_available"))
    wants_cuda = requested_device.casefold().startswith("cuda")
    if wants_cuda and not cuda_available:
        raise TeacherReplayCacheError(
            "A CUDA replay-cache request requires an available CUDA runtime."
        )
    device_index: int | None = None
    device_name: str | None = None
    capability: list[int] | None = None
    total_memory: int | None = None
    if wants_cuda:
        try:
            parsed = torch.device(requested_device)
            device_index = parsed.index
            if device_index is None:
                device_index = int(cuda.current_device())
            properties = cuda.get_device_properties(device_index)
            device_name = str(properties.name)
            capability = [int(properties.major), int(properties.minor)]
            total_memory = int(properties.total_memory)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise TeacherReplayCacheError(
                "CUDA replay-cache identity could not authenticate the requested device."
            ) from error
    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    cuda_backend = getattr(backends, "cuda", None)
    matmul = getattr(cuda_backend, "matmul", None)
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    cudnn_version = _optional_call(cudnn, "version")
    deterministic_enabled = _optional_call(torch, "are_deterministic_algorithms_enabled")
    deterministic_debug = _optional_call(torch, "get_deterministic_debug_mode")
    body = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "transformers_version": str(getattr(transformers, "__version__", "unknown")),
        "requested_device": requested_device,
        "cuda_available": cuda_available,
        "cuda_runtime_version": None if cuda_version is None else str(cuda_version),
        "cuda_device_index": device_index,
        "cuda_device_name": device_name,
        "cuda_device_capability": capability,
        "cuda_device_total_memory": total_memory,
        "cudnn_version": None if cudnn_version is None else int(cudnn_version),
        "deterministic_algorithms_enabled": (
            None if deterministic_enabled is None else bool(deterministic_enabled)
        ),
        "deterministic_debug_mode": (
            None if deterministic_debug is None else int(deterministic_debug)
        ),
        "cudnn_deterministic": getattr(cudnn, "deterministic", None),
        "cudnn_benchmark": getattr(cudnn, "benchmark", None),
        "cudnn_allow_tf32": getattr(cudnn, "allow_tf32", None),
        "cuda_matmul_allow_tf32": getattr(matmul, "allow_tf32", None),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    return {**body, "sha256": _canonical_sha256(body)}


def _verify_producer_manifest(payload: Any) -> dict[str, Any]:
    row = _strict_fields(payload, _PRODUCER_FIELDS, "producer_byte_hash_manifest")
    body = dict(row)
    claimed = _require_sha256(body.pop("sha256"), "producer manifest SHA")
    files = row["files"]
    if not isinstance(files, list) or len(files) != 4:
        raise TeacherReplayCacheError("Producer manifest must bind exactly four files.")
    roles: list[str] = []
    for index, value in enumerate(files):
        item = _strict_fields(value, _PRODUCER_FILE_FIELDS, f"producer files[{index}]")
        role = item["role"]
        if not isinstance(role, str) or not role:
            raise TeacherReplayCacheError("Producer role must be nonblank.")
        if not isinstance(item["filename"], str) or not item["filename"]:
            raise TeacherReplayCacheError("Producer filename must be nonblank.")
        _positive_int(item["byte_count"], "producer byte_count")
        _require_sha256(item["sha256"], "producer file SHA")
        roles.append(role)
    if roles != sorted(roles) or set(roles) != {
        "prepare_recovery_inputs",
        "hf_teacher_token_replay",
        "hf_recovery",
        "hf_teacher_replay_cache",
    }:
        raise TeacherReplayCacheError("Producer manifest roles changed.")
    if _canonical_sha256(body) != claimed:
        raise TeacherReplayCacheError("Producer manifest self-hash failed.")
    return dict(row)


def _verify_runtime_identity(payload: Any) -> dict[str, Any]:
    row = _strict_fields(payload, _RUNTIME_FIELDS, "runtime_identity")
    body = dict(row)
    claimed = _require_sha256(body.pop("sha256"), "runtime identity SHA")
    for name in ("python_implementation", "python_version", "torch_version", "transformers_version", "requested_device"):
        if not isinstance(row[name], str) or not row[name]:
            raise TeacherReplayCacheError(f"runtime_identity.{name} must be nonblank.")
    if not isinstance(row["cuda_available"], bool):
        raise TeacherReplayCacheError("runtime_identity.cuda_available must be boolean.")
    wants_cuda = row["requested_device"].casefold().startswith("cuda")
    cuda_values = (
        row["cuda_device_index"],
        row["cuda_device_name"],
        row["cuda_device_capability"],
        row["cuda_device_total_memory"],
    )
    if wants_cuda:
        if row["cuda_available"] is not True:
            raise TeacherReplayCacheError("CUDA request cannot bind an unavailable runtime.")
        if (
            isinstance(cuda_values[0], bool)
            or not isinstance(cuda_values[0], int)
            or cuda_values[0] < 0
            or not isinstance(cuda_values[1], str)
            or not cuda_values[1]
            or not isinstance(cuda_values[2], list)
            or len(cuda_values[2]) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in cuda_values[2])
            or isinstance(cuda_values[3], bool)
            or not isinstance(cuda_values[3], int)
            or cuda_values[3] <= 0
        ):
            raise TeacherReplayCacheError("CUDA device identity is incomplete.")
    elif any(value is not None for value in cuda_values):
        raise TeacherReplayCacheError("Non-CUDA runtime cannot bind CUDA device fields.")
    if _canonical_sha256(body) != claimed:
        raise TeacherReplayCacheError("Runtime identity self-hash failed.")
    return dict(row)


def build_request(
    *,
    expected_parent_artifact_tree_sha256: str,
    actual_parent_artifact_tree: Mapping[str, Any],
    calibration_source_file_sha256: str,
    ordered_calibration_texts: Sequence[str],
    source_tokenizer_fingerprint: Mapping[str, Any],
    candidate_tokenizer_fingerprint: Mapping[str, Any],
    teacher_parameters: Mapping[str, Any],
    producer_manifest: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the complete content-address identity without parent-derived data."""

    body = {
        "schema": REQUEST_SCHEMA,
        "source_artifact_tree": {
            "expected_tree_sha256": expected_parent_artifact_tree_sha256,
            "actual_tree_sha256": actual_parent_artifact_tree.get("tree_sha256"),
            "file_count": actual_parent_artifact_tree.get("file_count"),
            "total_bytes": actual_parent_artifact_tree.get("total_bytes"),
        },
        "calibration": {
            "source_file_sha256": calibration_source_file_sha256,
            "ordered_text_count": len(ordered_calibration_texts),
            "ordered_text_manifest_sha256": _canonical_sha256(
                list(ordered_calibration_texts)
            ),
        },
        "tokenizer_fingerprints": {
            "source": dict(source_tokenizer_fingerprint),
            "candidate": dict(candidate_tokenizer_fingerprint),
        },
        "teacher_parameters": dict(teacher_parameters),
        "teacher_parameters_sha256": _canonical_sha256(dict(teacher_parameters)),
        "local_files_only": True,
        "trust_remote_code": False,
        "producer_byte_hash_manifest": dict(producer_manifest),
        "runtime_identity": dict(runtime),
    }
    request = {**body, "request_sha256": _canonical_sha256(body)}
    return verify_request(request)


def verify_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = _strict_fields(payload, _REQUEST_FIELDS, "cache request")
    body = dict(row)
    claimed = _require_sha256(body.pop("request_sha256"), "cache request SHA")
    if row["schema"] != REQUEST_SCHEMA:
        raise TeacherReplayCacheError("Cache request schema changed.")
    tree = _strict_fields(row["source_artifact_tree"], _SOURCE_TREE_FIELDS, "source_artifact_tree")
    expected = _require_sha256(tree["expected_tree_sha256"], "expected source tree SHA")
    actual = _require_sha256(tree["actual_tree_sha256"], "actual source tree SHA")
    if expected != actual:
        raise TeacherReplayCacheError("Expected and actual source artifact trees differ.")
    _positive_int(tree["file_count"], "source artifact file_count")
    _positive_int(tree["total_bytes"], "source artifact total_bytes")
    calibration = _strict_fields(row["calibration"], _CALIBRATION_FIELDS, "calibration")
    _require_sha256(calibration["source_file_sha256"], "calibration file SHA")
    _positive_int(calibration["ordered_text_count"], "ordered calibration count")
    _require_sha256(calibration["ordered_text_manifest_sha256"], "calibration manifest SHA")
    tokenizers = _strict_fields(row["tokenizer_fingerprints"], _TOKENIZER_FIELDS, "tokenizer_fingerprints")
    for name in ("source", "candidate"):
        fingerprint = tokenizers[name]
        if not isinstance(fingerprint, Mapping):
            raise TeacherReplayCacheError(f"{name} tokenizer fingerprint must be an object.")
        _require_sha256(fingerprint.get("sha256"), f"{name} tokenizer fingerprint SHA")
    if not isinstance(row["teacher_parameters"], Mapping):
        raise TeacherReplayCacheError("teacher_parameters must be an object.")
    if _canonical_sha256(dict(row["teacher_parameters"])) != _require_sha256(
        row["teacher_parameters_sha256"], "teacher parameters SHA"
    ):
        raise TeacherReplayCacheError("Teacher parameters hash changed.")
    if row["local_files_only"] is not True or row["trust_remote_code"] is not False:
        raise TeacherReplayCacheError("Cache request must remain local-only and trust-disabled.")
    _verify_producer_manifest(row["producer_byte_hash_manifest"])
    _verify_runtime_identity(row["runtime_identity"])
    if _canonical_sha256(body) != claimed:
        raise TeacherReplayCacheError("Cache request self-hash failed.")
    return json.loads(json.dumps(dict(row), allow_nan=False))


def build_envelope(
    *,
    request: Mapping[str, Any],
    teacher_payload: Mapping[str, Any],
    live_source_model_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    verified_request = verify_request(request)
    try:
        verified_teacher = hf_teacher_token_replay.verify_teacher_token_replay_payload(
            teacher_payload
        )
    except (TypeError, ValueError) as error:
        raise TeacherReplayCacheError("Teacher replay failed strict verification.") from error
    teacher = json.loads(json.dumps(verified_teacher, allow_nan=False))
    live = json.loads(json.dumps(dict(live_source_model_fingerprint), allow_nan=False))
    if live != teacher["source_model_fingerprint"]:
        raise TeacherReplayCacheError(
            "Live source model fingerprint differs from generated teacher replay."
        )
    state_sha = _require_sha256(
        live["model_state_fingerprint"]["state_dict_sha256"],
        "source model state_dict SHA",
    )
    body = {
        "schema": ENVELOPE_SCHEMA,
        "status": "frozen-unaccepted",
        "contains_sealed_data": False,
        "sealed_data_accessed": False,
        "publication_authorized": False,
        "request": verified_request,
        "request_sha256": verified_request["request_sha256"],
        "teacher_token_replay": teacher,
        "teacher_payload_sha256": teacher["payload_sha256"],
        "source_model_fingerprint": live,
        "source_model_state_dict_sha256": state_sha,
    }
    envelope = {**body, "envelope_sha256": _canonical_sha256(body)}
    return verify_envelope(envelope, expected_request=verified_request)


def verify_envelope(
    payload: Mapping[str, Any],
    *,
    expected_request: Mapping[str, Any],
    live_source_model_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = _strict_fields(payload, _ENVELOPE_FIELDS, "cache envelope")
    body = dict(row)
    claimed = _require_sha256(body.pop("envelope_sha256"), "cache envelope SHA")
    request = verify_request(row["request"])
    expected = verify_request(expected_request)
    request_sha = _require_sha256(row["request_sha256"], "envelope request SHA")
    if request != expected or request_sha != request["request_sha256"]:
        raise TeacherReplayCacheError("Cache envelope is bound to a different request.")
    try:
        verified_teacher = hf_teacher_token_replay.verify_teacher_token_replay_payload(
            row["teacher_token_replay"]
        )
    except (TypeError, ValueError) as error:
        raise TeacherReplayCacheError("Cached teacher replay failed strict verification.") from error
    teacher = json.loads(json.dumps(verified_teacher, allow_nan=False))
    if teacher != row["teacher_token_replay"]:
        raise TeacherReplayCacheError("Cached teacher replay changed during verification.")
    teacher_sha = _require_sha256(row["teacher_payload_sha256"], "teacher payload SHA")
    if teacher_sha != teacher["payload_sha256"]:
        raise TeacherReplayCacheError("Cached teacher payload SHA is not the inner payload SHA.")
    calibration = request["calibration"]
    if (
        len(teacher["calibration_texts"]) != calibration["ordered_text_count"]
        or _canonical_sha256(teacher["calibration_texts"])
        != calibration["ordered_text_manifest_sha256"]
        or teacher["parameters"] != request["teacher_parameters"]
        or teacher["tokenizer_fingerprints"] != request["tokenizer_fingerprints"]
    ):
        raise TeacherReplayCacheError(
            "Cached replay calibration, parameters, or tokenizers differ from request."
        )
    source = row["source_model_fingerprint"]
    if not isinstance(source, Mapping) or dict(source) != teacher["source_model_fingerprint"]:
        raise TeacherReplayCacheError("Cached source model fingerprint binding changed.")
    state_sha = _require_sha256(
        row["source_model_state_dict_sha256"], "cached source state_dict SHA"
    )
    if state_sha != source["model_state_fingerprint"]["state_dict_sha256"]:
        raise TeacherReplayCacheError("Cached source state binding changed.")
    if live_source_model_fingerprint is not None and dict(live_source_model_fingerprint) != dict(source):
        raise TeacherReplayCacheError(
            "Live source model fingerprint differs from cached source identity."
        )
    if (
        row["schema"] != ENVELOPE_SCHEMA
        or row["status"] != "frozen-unaccepted"
        or row["contains_sealed_data"] is not False
        or row["sealed_data_accessed"] is not False
        or row["publication_authorized"] is not False
        or _canonical_sha256(body) != claimed
    ):
        raise TeacherReplayCacheError("Cache envelope self-hash failed.")
    return json.loads(json.dumps(dict(row), allow_nan=False))


def _absolute_path(value: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(value).expanduser())))


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise TeacherReplayCacheError(f"Cannot lstat cache path {path}.") from error


def _require_real_directory(path: Path, name: str) -> os.stat_result:
    info = _lstat_or_none(path)
    if info is None or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise TeacherReplayCacheError(f"{name} must be one real directory.")
    return info


def _directory_identity(info: os.stat_result) -> tuple[int, int]:
    return (int(info.st_dev), int(info.st_ino))


def _cache_root(value: Path | str, *, create: bool) -> Path:
    root = _absolute_path(value)
    if _lexists(root):
        _require_real_directory(root, "teacher replay cache directory")
        return root
    if not create:
        raise TeacherReplayCacheError(
            "teacher replay cache directory disappeared during authentication."
        )
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()
    except FileExistsError:
        _require_real_directory(root, "teacher replay cache directory")
    except OSError as error:
        raise TeacherReplayCacheError(
            "Cannot create teacher replay cache directory."
        ) from error
    else:
        _require_real_directory(root, "teacher replay cache directory")
        _fsync_directory(root.parent)
    return root


def cache_entry_path(cache_dir: Path | str, request: Mapping[str, Any]) -> Path:
    verified = verify_request(request)
    return _absolute_path(cache_dir) / verified["request_sha256"]


def _reservation_path(root: Path, request: Mapping[str, Any]) -> Path:
    return root / f".{request['request_sha256']}{_RESERVATION_SUFFIX}"


def _reservation_present(path: Path) -> bool:
    info = _lstat_or_none(path)
    if info is None:
        return False
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise TeacherReplayCacheError(
            "Cache publication reservation must be one real directory."
        )
    try:
        children = list(path.iterdir())
    except FileNotFoundError:
        return False
    except OSError as error:
        raise TeacherReplayCacheError(
            "Cannot inspect cache publication reservation."
        ) from error
    if children:
        raise TeacherReplayCacheError(
            "Cache publication reservation must remain an empty directory."
        )
    return True


def _entry_is_empty_real_directory(entry: Path) -> bool:
    info = _lstat_or_none(entry)
    if info is None or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return False
    try:
        return not any(entry.iterdir())
    except OSError:
        return False


def _read_entry_once(
    root: Path,
    verified_request: Mapping[str, Any],
) -> dict[str, Any] | None:
    entry = root / verified_request["request_sha256"]
    if not _lexists(entry):
        return None
    _require_real_directory(entry, "Cache request path")
    if entry.parent != root:
        raise TeacherReplayCacheError("Cache request path escaped the cache root.")
    try:
        children = list(entry.iterdir())
    except OSError as error:
        raise TeacherReplayCacheError("Cannot inspect cache request path.") from error
    if len(children) != 1 or children[0].name != CACHE_FILENAME:
        raise TeacherReplayCacheError(
            "Cache entry must contain exactly cache_envelope.json."
        )
    envelope_path = children[0]
    envelope_info = _lstat_or_none(envelope_path)
    if (
        envelope_info is None
        or stat.S_ISLNK(envelope_info.st_mode)
        or not stat.S_ISREG(envelope_info.st_mode)
    ):
        raise TeacherReplayCacheError("Cache envelope must be one real file.")
    envelope = _strict_json_file(envelope_path)
    verified = verify_envelope(envelope, expected_request=verified_request)
    if entry.name != verified["request_sha256"]:
        raise TeacherReplayCacheError("Cache request key and entry path differ.")
    return verified


def _read_reserved_entry(
    root: Path,
    verified_request: Mapping[str, Any],
    reservation: Path,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + _RESERVATION_WAIT_SECONDS
    entry = root / verified_request["request_sha256"]
    while True:
        reservation_exists = _reservation_present(reservation)
        try:
            cached = _read_entry_once(root, verified_request)
        except TeacherReplayCacheError:
            reservation_exists = _reservation_present(reservation)
            if not reservation_exists or (
                _lexists(entry) and not _entry_is_empty_real_directory(entry)
            ):
                raise
            cached = None
        if cached is not None:
            return cached
        if not reservation_exists:
            return _read_entry_once(root, verified_request)
        if time.monotonic() >= deadline:
            raise TeacherReplayCacheError(
                "Cache publication reservation did not complete before timeout."
            )
        time.sleep(_RESERVATION_POLL_SECONDS)


def read_entry(
    cache_dir: Path | str,
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return one statically verified entry; an invalid existing entry is fatal."""

    verified_request = verify_request(request)
    raw_root = _absolute_path(cache_dir)
    if not _lexists(raw_root):
        return None
    root = _cache_root(raw_root, create=False)
    reservation = _reservation_path(root, verified_request)
    if _reservation_present(reservation):
        return _read_reserved_entry(root, verified_request, reservation)
    try:
        cached = _read_entry_once(root, verified_request)
    except TeacherReplayCacheError:
        if _reservation_present(reservation):
            return _read_reserved_entry(root, verified_request, reservation)
        raise
    if cached is None and _reservation_present(reservation):
        return _read_reserved_entry(root, verified_request, reservation)
    return cached


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        dict(payload),
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes where the platform exposes directory fsync."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise TeacherReplayCacheError(
            f"Cannot open cache directory {path} for durability sync."
        ) from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise TeacherReplayCacheError(
            f"Cannot durability-sync cache directory {path}."
        ) from error
    finally:
        os.close(descriptor)


def _reserve_publication(path: Path) -> tuple[int, int] | None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        _reservation_present(path)
        return None
    except OSError as error:
        raise TeacherReplayCacheError(
            "Cannot create cache publication reservation."
        ) from error
    return _directory_identity(
        _require_real_directory(path, "Cache publication reservation")
    )


def _release_publication_reservation(
    path: Path,
    identity: tuple[int, int],
    root: Path,
) -> None:
    info = _require_real_directory(path, "Owned cache publication reservation")
    if _directory_identity(info) != identity:
        raise TeacherReplayCacheError(
            "Cache publication reservation identity changed before release."
        )
    if any(path.iterdir()):
        raise TeacherReplayCacheError(
            "Owned cache publication reservation unexpectedly contains files."
        )
    try:
        path.rmdir()
    except OSError as error:
        raise TeacherReplayCacheError(
            "Cannot release cache publication reservation."
        ) from error
    _fsync_directory(root)


def _create_entry_noreplace(entry: Path) -> tuple[int, int]:
    try:
        entry.mkdir(mode=0o700)
    except FileExistsError as error:
        if _lexists(entry):
            info = _lstat_or_none(entry)
            if info is not None and stat.S_ISLNK(info.st_mode):
                raise TeacherReplayCacheError(
                    "Cache entry race produced a symlink; refusing overwrite."
                ) from error
        raise TeacherReplayCacheError(
            "Cache entry appeared during exclusive reservation; refusing overwrite."
        ) from error
    except OSError as error:
        raise TeacherReplayCacheError(
            "Cannot reserve final cache entry directory."
        ) from error
    return _directory_identity(_require_real_directory(entry, "Reserved cache entry"))


def _link_envelope_noreplace(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise TeacherReplayCacheError(
            "Cache envelope appeared during publication; refusing overwrite."
        ) from error
    except OSError as error:
        raise TeacherReplayCacheError(
            "Cannot atomically link the cache envelope into its reserved entry."
        ) from error


def _remove_owned_empty_entry(
    entry: Path,
    identity: tuple[int, int],
    root: Path,
) -> None:
    info = _lstat_or_none(entry)
    if info is None:
        return
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or _directory_identity(info) != identity
        or any(entry.iterdir())
    ):
        return
    entry.rmdir()
    _fsync_directory(root)


def _remove_owned_staging(
    staging: Path,
    identity: tuple[int, int],
    root: Path,
) -> None:
    info = _lstat_or_none(staging)
    if info is None:
        return
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or _directory_identity(info) != identity
        or staging.parent != root
    ):
        raise TeacherReplayCacheError(
            "Owned cache staging directory identity changed before cleanup."
        )
    shutil.rmtree(staging)
    _fsync_directory(root)


def _verified_existing_result(
    existing: Mapping[str, Any],
    *,
    envelope: Mapping[str, Any],
    request: Mapping[str, Any],
    live_source_model_fingerprint: Mapping[str, Any],
    result: str,
) -> tuple[dict[str, Any], str]:
    verified_existing = verify_envelope(
        existing,
        expected_request=request,
        live_source_model_fingerprint=live_source_model_fingerprint,
    )
    if _canonical_bytes(verified_existing) != _canonical_bytes(envelope):
        raise TeacherReplayCacheError(
            "Existing cache envelope differs byte-for-byte; refusing overwrite."
        )
    return verified_existing, result


def publish_entry(
    *,
    cache_dir: Path | str,
    request: Mapping[str, Any],
    teacher_payload: Mapping[str, Any],
    live_source_model_fingerprint: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Atomically publish without overwrite; identical concurrent writers are safe."""

    verified_request = verify_request(request)
    envelope = build_envelope(
        request=verified_request,
        teacher_payload=teacher_payload,
        live_source_model_fingerprint=live_source_model_fingerprint,
    )
    root = _cache_root(cache_dir, create=True)
    entry = root / verified_request["request_sha256"]
    existing = read_entry(root, verified_request)
    if existing is not None:
        return _verified_existing_result(
            existing,
            envelope=envelope,
            request=verified_request,
            live_source_model_fingerprint=live_source_model_fingerprint,
            result="existing-byte-identical",
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{verified_request['request_sha256']}.",
            suffix=".staging",
            dir=root,
        )
    )
    staging_identity = _directory_identity(
        _require_real_directory(staging, "Cache staging directory")
    )
    reservation = _reservation_path(root, verified_request)
    try:
        _write_json_exclusive(staging / CACHE_FILENAME, envelope)
        _fsync_directory(staging)
        reread = _strict_json_file(staging / CACHE_FILENAME)
        verify_envelope(
            reread,
            expected_request=verified_request,
            live_source_model_fingerprint=live_source_model_fingerprint,
        )
        for _attempt in range(3):
            reservation_identity = _reserve_publication(reservation)
            if reservation_identity is None:
                concurrent = read_entry(root, verified_request)
                if concurrent is None:
                    continue
                return _verified_existing_result(
                    concurrent,
                    envelope=envelope,
                    request=verified_request,
                    live_source_model_fingerprint=live_source_model_fingerprint,
                    result="concurrent-byte-identical",
                )
            entry_identity: tuple[int, int] | None = None
            envelope_linked = False
            try:
                appeared = _read_entry_once(root, verified_request)
                if appeared is not None:
                    return _verified_existing_result(
                        appeared,
                        envelope=envelope,
                        request=verified_request,
                        live_source_model_fingerprint=live_source_model_fingerprint,
                        result="concurrent-byte-identical",
                    )
                entry_identity = _create_entry_noreplace(entry)
                _link_envelope_noreplace(
                    staging / CACHE_FILENAME,
                    entry / CACHE_FILENAME,
                )
                envelope_linked = True
                _fsync_directory(entry)
                _fsync_directory(root)
                published = _read_entry_once(root, verified_request)
                if published is None:
                    raise TeacherReplayCacheError("Published cache entry disappeared.")
                verified_published = verify_envelope(
                    published,
                    expected_request=verified_request,
                    live_source_model_fingerprint=live_source_model_fingerprint,
                )
                if verified_published != envelope:
                    raise TeacherReplayCacheError(
                        "Published cache envelope changed after no-replace link."
                    )
                return verified_published, "published"
            finally:
                if entry_identity is not None and not envelope_linked:
                    _remove_owned_empty_entry(entry, entry_identity, root)
                _release_publication_reservation(
                    reservation,
                    reservation_identity,
                    root,
                )
        raise TeacherReplayCacheError(
            "Concurrent cache publisher repeatedly exited without an entry."
        )
    finally:
        _remove_owned_staging(staging, staging_identity, root)


__all__ = [
    "CACHE_FILENAME",
    "ENVELOPE_SCHEMA",
    "REQUEST_SCHEMA",
    "TeacherReplayCacheError",
    "build_envelope",
    "build_request",
    "cache_entry_path",
    "producer_byte_hash_manifest",
    "publish_entry",
    "read_entry",
    "runtime_identity",
    "verify_envelope",
    "verify_request",
]
