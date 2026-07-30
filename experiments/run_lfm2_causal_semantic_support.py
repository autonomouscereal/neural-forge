"""Offline A12-to-causal-support adapter for the LFM2 semantic bridge.

The runner never constructs or executes a model.  It consumes:

* one completed authenticated A12 teacher-forced audit;
* one fit-only causal feature pack;
* one sealed prompt-token manifest whose fit IDs are disjoint from the future
  manual-review IDs;
* the authenticated A9 residual-OMP C_in codebook and A10/A11 C_out
  coefficients;
* direct local safetensor views of the donor and target token tables.

It publishes a content-addressed, reload-authenticated bundle containing full
independent C_in and C_out CSR routes.  All continuous coefficients are counted
as runtime child state.  Cached publication is keyed by mathematical inputs,
not by output paths or logging choices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_counted_child_state import (  # noqa: E402
    ContinuousChildStateAllocation,
)
from neural_forge.hf_lfm2_causal_semantic_support import (  # noqa: E402
    Lfm2CausalSupportFeaturePack,
    Lfm2CausalSupportSelection,
    authenticate_lfm2_causal_support_selection,
    lfm2_causal_support_request,
    select_lfm2_causal_semantic_supports,
)
from neural_forge.hf_lfm2_functional_output_refit import (  # noqa: E402
    authenticate_lfm2_functional_output_refit_result,
)
from neural_forge.hf_lfm2_sparse_semantic_codebook import (  # noqa: E402
    load_lfm2_sparse_semantic_codebook,
)


FIT_PROMPT_MANIFEST_SCHEMA = "neural-forge-lfm2-causal-support-fit-prompt-manifest-v1"
CAUSAL_SUPPORT_PUBLICATION_SCHEMA = "neural-forge-lfm2-causal-support-publication-v1"
CAUSAL_SUPPORT_RUN_SCHEMA = "neural-forge-lfm2-causal-support-offline-run-v1"
TRAJECTORY_COMPLETION_SCHEMA = (
    "neural-forge-lfm2-teacher-forced-trajectory-cache-completion-v1"
)
OUTPUT_TENSOR_KEYS = (
    "input_offsets",
    "input_target_ids",
    "input_coefficients",
    "output_offsets",
    "output_target_ids",
    "output_coefficients",
    "input_reselected_donor_ids",
    "output_reselected_donor_ids",
)
_ZERO_OPERATION_AUDIT = {
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "model_objects_constructed": 0,
    "model_forward_calls": 0,
    "model_generate_calls": 0,
    "parent_forward_calls": 0,
    "parent_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "global_map_solve_calls": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


class Lfm2CausalSupportRunnerError(RuntimeError):
    """Raised when an offline live input or publication fails closed."""


def _plain_json(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CausalSupportRunnerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CausalSupportRunnerError(f"{name} must be positive.")
    return int(value)


def _safe_file(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise Lfm2CausalSupportRunnerError(f"{name} must be an ordinary file.")
    return path


def _safe_directory(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_symlink() or not path.is_dir():
        raise Lfm2CausalSupportRunnerError(f"{name} must be an ordinary directory.")
    return path


def _output_directory(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise Lfm2CausalSupportRunnerError("Output root must be an ordinary directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CausalSupportRunnerError(f"Cannot read {name} JSON.") from exc
    if not isinstance(value, Mapping):
        raise Lfm2CausalSupportRunnerError(f"{name} must be a JSON object.")
    return _plain_json(value)


def authenticate_completed_trajectory_audit(
    *,
    audit_path: Path,
    completion_path: Path,
    expected_audit_file_sha256: str,
    expected_audit_sha256: str,
    expected_completion_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate A12's semantic receipt and atomic completion together."""

    audit_file_sha = _file_sha256(audit_path)
    if audit_file_sha != _digest(
        expected_audit_file_sha256, "expected trajectory audit file SHA-256"
    ):
        raise Lfm2CausalSupportRunnerError(
            "Trajectory audit file differs from its explicit file pin."
        )
    audit = _read_json(audit_path, "trajectory audit")
    semantic_sha = _digest(
        expected_audit_sha256, "expected trajectory semantic audit SHA-256"
    )
    if audit.get("audit_sha256") != semantic_sha:
        raise Lfm2CausalSupportRunnerError(
            "Trajectory audit differs from its explicit semantic seal."
        )
    completion = _read_json(completion_path, "trajectory completion")
    supplied = completion.pop("completion_sha256", None)
    if (
        supplied
        != _digest(expected_completion_sha256, "expected trajectory completion SHA-256")
        or supplied != canonical_sha256(completion)
        or completion.get("schema") != TRAJECTORY_COMPLETION_SCHEMA
        or completion.get("audit_file_sha256") != audit_file_sha
        or completion.get("audit_sha256") != semantic_sha
        or completion.get("request_cache_key") != audit.get("request_cache_key")
        or completion.get("reusable_identical_math") is not True
        or completion.get("atomic_batch_checkpoints") is not True
    ):
        raise Lfm2CausalSupportRunnerError(
            "Trajectory audit completion authentication failed."
        )
    return audit, {**completion, "completion_sha256": supplied}


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _plain_json(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_save_tensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".safetensors", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in tensors.items()
            },
            str(temporary_path),
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _sparse_table_sha256(value: torch.Tensor) -> str:
    """Match the sparse-codebook tensor hash domain."""

    tensor = value.detach()
    digest = hashlib.sha256(b"neural-forge-lfm2-sparse-semantic-tensor-v1\0")
    digest.update(str(tensor.dtype).removeprefix("torch.").encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    for start in range(0, int(tensor.shape[0]), 4096):
        block = tensor[start : min(start + 4096, int(tensor.shape[0]))]
        digest.update(
            block.to(device="cpu")
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def build_fit_prompt_manifest(
    *,
    trajectory_audit_sha256: str,
    feature_pack_artifact_sha256: str,
    fit_prompt_ids: Sequence[str],
    manual_review_prompt_ids: Sequence[str],
    prompt_token_ids_by_prompt: Mapping[str, Sequence[int]],
    source_trace_batch_receipt_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Seal exact fit prompt tokens obtained from authenticated trace batches."""

    fit_ids = tuple(str(value) for value in fit_prompt_ids)
    review_ids = tuple(str(value) for value in manual_review_prompt_ids)
    if (
        not fit_ids
        or not review_ids
        or any(not value for value in (*fit_ids, *review_ids))
        or len(set(fit_ids)) != len(fit_ids)
        or len(set(review_ids)) != len(review_ids)
        or set(fit_ids).intersection(review_ids)
    ):
        raise Lfm2CausalSupportRunnerError(
            "Fit and manual-review prompt IDs must be unique, nonempty, and disjoint."
        )
    tokens: dict[str, list[int]] = {}
    sources: dict[str, str] = {}
    for prompt_id in fit_ids:
        row = prompt_token_ids_by_prompt.get(prompt_id)
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or not row:
            raise Lfm2CausalSupportRunnerError(
                f"Fit prompt {prompt_id!r} has no token row."
            )
        token_ids = []
        for token_id in row:
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
            ):
                raise Lfm2CausalSupportRunnerError(
                    "Fit prompt manifest contains an invalid token ID."
                )
            token_ids.append(int(token_id))
        tokens[prompt_id] = token_ids
        sources[prompt_id] = _digest(
            source_trace_batch_receipt_sha256s.get(prompt_id),
            f"trace batch receipt for {prompt_id}",
        )
    if set(prompt_token_ids_by_prompt) != set(fit_ids) or set(
        source_trace_batch_receipt_sha256s
    ) != set(fit_ids):
        raise Lfm2CausalSupportRunnerError(
            "Fit manifest may contain exactly the selected fit prompt IDs only."
        )
    body = {
        "schema": FIT_PROMPT_MANIFEST_SCHEMA,
        "trajectory_audit_sha256": _digest(
            trajectory_audit_sha256, "trajectory_audit_sha256"
        ),
        "feature_pack_artifact_sha256": _digest(
            feature_pack_artifact_sha256, "feature_pack_artifact_sha256"
        ),
        "fit_prompt_ids": list(fit_ids),
        "fit_prompt_ids_sha256": canonical_sha256(list(fit_ids)),
        "manual_review_prompt_ids": list(review_ids),
        "manual_review_prompt_ids_sha256": canonical_sha256(list(review_ids)),
        "selection_and_manual_review_prompt_ids_disjoint": True,
        "prompt_token_ids_by_prompt": tokens,
        "prompt_token_ids_by_prompt_sha256": canonical_sha256(tokens),
        "source_trace_batch_receipt_sha256s": sources,
        "source_trace_batch_receipt_inventory_sha256": canonical_sha256(sources),
        "tokens_loaded_from_authenticated_cached_trace_batches": True,
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "training_steps": 0,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def authenticate_fit_prompt_manifest(
    value: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
    expected_audit_sha256: str,
    feature_pack: Lfm2CausalSupportFeaturePack,
) -> dict[str, Any]:
    manifest = _plain_json(value)
    supplied = manifest.pop("manifest_sha256", None)
    feature = feature_pack.as_dict()
    fit_ids = manifest.get("fit_prompt_ids")
    review_ids = manifest.get("manual_review_prompt_ids")
    tokens = manifest.get("prompt_token_ids_by_prompt")
    sources = manifest.get("source_trace_batch_receipt_sha256s")
    if (
        supplied != _digest(expected_manifest_sha256, "expected manifest SHA-256")
        or supplied != canonical_sha256(manifest)
        or manifest.get("schema") != FIT_PROMPT_MANIFEST_SCHEMA
        or manifest.get("trajectory_audit_sha256") != expected_audit_sha256
        or manifest.get("feature_pack_artifact_sha256") != feature_pack.artifact_sha256
        or fit_ids != feature["fit_prompt_ids"]
        or review_ids != feature["manual_review_prompt_ids"]
        or set(fit_ids).intersection(review_ids)
        or manifest.get("selection_and_manual_review_prompt_ids_disjoint") is not True
        or not isinstance(tokens, Mapping)
        or set(tokens) != set(fit_ids)
        or manifest.get("prompt_token_ids_by_prompt_sha256") != canonical_sha256(tokens)
        or not isinstance(sources, Mapping)
        or set(sources) != set(fit_ids)
        or manifest.get("source_trace_batch_receipt_inventory_sha256")
        != canonical_sha256(sources)
        or manifest.get("tokens_loaded_from_authenticated_cached_trace_batches")
        is not True
        or manifest.get("parent_forward_calls") != 0
        or manifest.get("parent_trace_recaptures") != 0
        or manifest.get("training_steps") != 0
    ):
        raise Lfm2CausalSupportRunnerError("Fit prompt manifest authentication failed.")
    return {**manifest, "manifest_sha256": supplied}


def _load_exact_tensor(
    path: Path,
    *,
    tensor_key: str,
    expected_file_sha256: str,
    name: str,
) -> torch.Tensor:
    if _file_sha256(path) != _digest(
        expected_file_sha256, f"expected {name} file SHA-256"
    ):
        raise Lfm2CausalSupportRunnerError(f"{name} source file hash differs.")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if tensor_key not in handle.keys():
            raise Lfm2CausalSupportRunnerError(
                f"{name} tensor key {tensor_key!r} is missing."
            )
        tensor = handle.get_tensor(tensor_key).detach().cpu().contiguous()
    if tensor.ndim != 2 or not torch.is_floating_point(tensor):
        raise Lfm2CausalSupportRunnerError(f"{name} must be a floating matrix.")
    return tensor


def _load_feature_pack(
    *,
    tensor_path: Path,
    receipt_path: Path,
    expected_tensor_file_sha256: str,
    expected_artifact_sha256: str,
) -> Lfm2CausalSupportFeaturePack:
    if _file_sha256(tensor_path) != _digest(
        expected_tensor_file_sha256, "expected feature tensor file SHA-256"
    ):
        raise Lfm2CausalSupportRunnerError("Feature tensor source file hash differs.")
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {
            "input_feature_transform",
            "output_feature_transform",
        }:
            raise Lfm2CausalSupportRunnerError(
                "Feature tensor inventory must contain exactly C_in and C_out transforms."
            )
        input_features = handle.get_tensor("input_feature_transform")
        output_features = handle.get_tensor("output_feature_transform")
    pack = Lfm2CausalSupportFeaturePack(
        input_features,
        output_features,
        _read_json(receipt_path, "causal feature receipt"),
    )
    if pack.artifact_sha256 != _digest(
        expected_artifact_sha256, "expected feature artifact SHA-256"
    ):
        raise Lfm2CausalSupportRunnerError(
            "Causal feature pack differs from its explicit pin."
        )
    return pack


def _selection_tensors(result: Lfm2CausalSupportSelection) -> dict[str, torch.Tensor]:
    return {name: getattr(result, name) for name in OUTPUT_TENSOR_KEYS}


def _load_cached_selection(
    directory: Path, *, expected_cache_key: str
) -> Lfm2CausalSupportSelection | None:
    tensor_path = directory / "routes.safetensors"
    receipt_path = directory / "receipt.json"
    completion_path = directory / "complete.json"
    observed = [path.exists() for path in (tensor_path, receipt_path, completion_path)]
    if not any(observed):
        return None
    if not all(observed) or any(
        path.is_symlink() or not path.is_file()
        for path in (tensor_path, receipt_path, completion_path)
    ):
        raise Lfm2CausalSupportRunnerError(
            "Causal support cache has an incomplete final publication."
        )
    completion = _read_json(completion_path, "causal support completion")
    supplied = completion.pop("publication_sha256", None)
    if (
        supplied != canonical_sha256(completion)
        or completion.get("schema") != CAUSAL_SUPPORT_PUBLICATION_SCHEMA
        or completion.get("cache_key") != expected_cache_key
        or completion.get("routes_file_sha256") != _file_sha256(tensor_path)
        or completion.get("receipt_file_sha256") != _file_sha256(receipt_path)
    ):
        raise Lfm2CausalSupportRunnerError(
            "Causal support cache publication authentication failed."
        )
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(OUTPUT_TENSOR_KEYS):
            raise Lfm2CausalSupportRunnerError(
                "Causal support output tensor inventory is not exact."
            )
        tensors = {name: handle.get_tensor(name) for name in OUTPUT_TENSOR_KEYS}
    result = authenticate_lfm2_causal_support_selection(
        **tensors,
        receipt=_read_json(receipt_path, "causal support result receipt"),
        expected_cache_key=expected_cache_key,
    )
    if completion.get("result_sha256") != result.result_sha256:
        raise Lfm2CausalSupportRunnerError(
            "Cached result seal differs from the publication."
        )
    return result


def _publish_selection(
    directory: Path, result: Lfm2CausalSupportSelection
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    tensor_path = directory / "routes.safetensors"
    receipt_path = directory / "receipt.json"
    _atomic_save_tensors(tensor_path, _selection_tensors(result))
    _atomic_write_json(receipt_path, result.as_dict())
    body = {
        "schema": CAUSAL_SUPPORT_PUBLICATION_SCHEMA,
        "cache_key": result.cache_key,
        "result_sha256": result.result_sha256,
        "routes_file": "routes.safetensors",
        "routes_file_sha256": _file_sha256(tensor_path),
        "receipt_file": "receipt.json",
        "receipt_file_sha256": _file_sha256(receipt_path),
        "content_addressed": True,
        "fresh_reload_required": True,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    publication = {**body, "publication_sha256": canonical_sha256(body)}
    _atomic_write_json(directory / "complete.json", publication)
    return publication


def causal_support_allocations_from_publication(
    directory: str | os.PathLike[str],
    *,
    expected_cache_key: str | None = None,
) -> tuple[ContinuousChildStateAllocation, ContinuousChildStateAllocation]:
    """Fresh-reload one route publication into two counted allocations.

    This is a package observer, not a planning echo: it authenticates the
    completion, reloads every route tensor, reconstructs the sealed selection,
    and derives exact C_in/C_out counts and hashes from that loaded artifact.
    """

    path = _safe_directory(directory, "causal support publication")
    completion = _read_json(path / "complete.json", "causal support completion")
    observed_cache_key = _digest(completion.get("cache_key"), "publication cache key")
    if expected_cache_key is not None and observed_cache_key != _digest(
        expected_cache_key, "expected publication cache key"
    ):
        raise Lfm2CausalSupportRunnerError(
            "Causal support publication belongs to different math."
        )
    selection = _load_cached_selection(path, expected_cache_key=observed_cache_key)
    if selection is None:  # pragma: no cover - completion was required above
        raise Lfm2CausalSupportRunnerError(
            "Causal support publication disappeared during fresh reload."
        )
    receipt = selection.as_dict()
    publication_dtype = receipt["request"]["dtype_contract"]["publication_dtype"]
    if (
        publication_dtype != "float32"
        or selection.input_coefficients.dtype != torch.float32
        or selection.output_coefficients.dtype != torch.float32
    ):
        raise Lfm2CausalSupportRunnerError(
            "Counted route publication is not the sealed FP32 representation."
        )
    artifact_sha256 = selection.result_sha256
    return (
        ContinuousChildStateAllocation(
            state_id="semantic-route.c_in",
            parameter_elements=int(selection.input_coefficients.numel()),
            tensor_sha256=str(receipt["input_coefficients_sha256"]),
            artifact_sha256=artifact_sha256,
            dtype=publication_dtype,
        ),
        ContinuousChildStateAllocation(
            state_id="semantic-route.c_out",
            parameter_elements=int(selection.output_coefficients.numel()),
            tensor_sha256=str(receipt["output_coefficients_sha256"]),
            artifact_sha256=artifact_sha256,
            dtype=publication_dtype,
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_grad_enabled(False)
    audit_path = _safe_file(args.trajectory_audit_json, "trajectory audit")
    completion_path = _safe_file(
        args.trajectory_completion_json, "trajectory completion"
    )
    feature_tensor_path = _safe_file(
        args.feature_tensors_safetensors, "feature tensors"
    )
    feature_receipt_path = _safe_file(args.feature_receipt_json, "feature receipt")
    prompt_manifest_path = _safe_file(
        args.fit_prompt_manifest_json, "fit prompt manifest"
    )
    donor_table_path = _safe_file(args.donor_table_safetensors, "donor table")
    target_table_path = _safe_file(args.target_table_safetensors, "target table")
    codebook_directory = _safe_directory(
        args.input_codebook_directory, "input codebook"
    )
    output_coefficients_path = _safe_file(
        args.output_coefficients_safetensors, "output coefficients"
    )
    output_refit_receipt_path = _safe_file(
        args.output_refit_receipt_json, "output refit receipt"
    )
    output_root = _output_directory(args.output_directory)

    audit, trajectory_completion = authenticate_completed_trajectory_audit(
        audit_path=audit_path,
        completion_path=completion_path,
        expected_audit_file_sha256=args.expected_trajectory_audit_file_sha256,
        expected_audit_sha256=args.expected_trajectory_audit_sha256,
        expected_completion_sha256=args.expected_trajectory_completion_sha256,
    )
    audit_sha = str(audit["audit_sha256"])
    feature_pack = _load_feature_pack(
        tensor_path=feature_tensor_path,
        receipt_path=feature_receipt_path,
        expected_tensor_file_sha256=args.expected_feature_tensor_file_sha256,
        expected_artifact_sha256=args.expected_feature_artifact_sha256,
    )
    manifest = authenticate_fit_prompt_manifest(
        _read_json(prompt_manifest_path, "fit prompt manifest"),
        expected_manifest_sha256=args.expected_fit_prompt_manifest_sha256,
        expected_audit_sha256=audit_sha,
        feature_pack=feature_pack,
    )
    codebook = load_lfm2_sparse_semantic_codebook(
        codebook_directory,
        expected_artifact_sha256=_digest(
            args.expected_input_codebook_artifact_sha256,
            "expected input codebook artifact SHA-256",
        ),
    )
    codebook_receipt = codebook.as_dict()
    donor_count = int(codebook_receipt["donor_row_count"])
    target_count = int(codebook_receipt["target_row_count"])
    hidden_size = int(codebook_receipt["hidden_size"])
    donor_table = _load_exact_tensor(
        donor_table_path,
        tensor_key=args.donor_table_key,
        expected_file_sha256=args.expected_donor_table_file_sha256,
        name="donor table",
    )
    # Parent checkpoints are normally BF16, while the authenticated sparse
    # codebook explicitly used the cached FP32 defined-row table.
    donor_table = donor_table[:donor_count, :].to(torch.float32).contiguous()
    target_table = _load_exact_tensor(
        target_table_path,
        tensor_key=args.target_table_key,
        expected_file_sha256=args.expected_target_table_file_sha256,
        name="target table",
    )[:target_count, :].contiguous()
    if (
        tuple(donor_table.shape) != (donor_count, hidden_size)
        or tuple(target_table.shape) != (target_count, hidden_size)
        or _sparse_table_sha256(donor_table) != codebook_receipt["donor_table_sha256"]
        or _sparse_table_sha256(target_table) != codebook_receipt["anchor_table_sha256"]
    ):
        raise Lfm2CausalSupportRunnerError(
            "Donor/target tables differ from the authenticated A9 codebook."
        )
    if _file_sha256(output_coefficients_path) != _digest(
        args.expected_output_coefficients_file_sha256,
        "expected output coefficient file SHA-256",
    ):
        raise Lfm2CausalSupportRunnerError(
            "Output coefficient source file hash differs."
        )
    with safe_open(
        str(output_coefficients_path), framework="pt", device="cpu"
    ) as handle:
        if set(handle.keys()) != {args.output_coefficients_key}:
            raise Lfm2CausalSupportRunnerError(
                "Output coefficient tensor inventory is not exact."
            )
        output_coefficients = handle.get_tensor(args.output_coefficients_key)
    output_result = authenticate_lfm2_functional_output_refit_result(
        output_semantic_code_coefficients=output_coefficients,
        receipt=_read_json(output_refit_receipt_path, "output refit receipt"),
        expected_cache_key=_digest(
            args.expected_output_refit_cache_key,
            "expected output refit cache key",
        ),
    )
    if (
        output_result.result_sha256
        != _digest(
            args.expected_output_refit_result_sha256,
            "expected output refit result SHA-256",
        )
        or output_result.receipt["input_codebook_artifact_sha256"]
        != codebook.artifact_sha256
    ):
        raise Lfm2CausalSupportRunnerError(
            "A10/A11 output route is not bound to the selected A9 codebook."
        )

    math_arguments = {
        "trajectory_audit": audit,
        "feature_pack": feature_pack,
        "prompt_token_ids_by_prompt": manifest["prompt_token_ids_by_prompt"],
        "donor_table": donor_table,
        "target_table": target_table,
        "target_to_donor": codebook.target_to_donor,
        "base_input_offsets": codebook.semantic_code_offsets,
        "base_input_target_ids": codebook.semantic_code_target_ids,
        "base_input_coefficients": codebook.semantic_code_coefficients,
        "base_output_offsets": codebook.semantic_code_offsets.clone(),
        "base_output_target_ids": codebook.semantic_code_target_ids.clone(),
        "base_output_coefficients": output_result.output_semantic_code_coefficients,
        "physical_child_parameter_count": _positive_int(
            args.physical_child_parameter_count,
            "physical_child_parameter_count",
        ),
        "maximum_combined_coefficient_count": _positive_int(
            args.maximum_combined_coefficient_count,
            "maximum_combined_coefficient_count",
        ),
        "input_support_width": _positive_int(
            args.input_support_width, "input_support_width"
        ),
        "output_support_width": _positive_int(
            args.output_support_width, "output_support_width"
        ),
        "maximum_input_donor_repairs": _positive_int(
            args.maximum_input_donor_repairs,
            "maximum_input_donor_repairs",
        ),
        "maximum_output_donor_repairs": _positive_int(
            args.maximum_output_donor_repairs,
            "maximum_output_donor_repairs",
        ),
        "position_bin_count": _positive_int(
            args.position_bin_count, "position_bin_count"
        ),
        "context_hash_bucket_count": _positive_int(
            args.context_hash_bucket_count,
            "context_hash_bucket_count",
        ),
        "context_weight": float(args.context_weight),
        "solve_ridge": float(args.solve_ridge),
        "solve_dtype": args.solve_dtype,
    }
    request = lfm2_causal_support_request(**math_arguments)
    cache_directory = output_root / request["cache_key"]
    if cache_directory.exists() and (
        cache_directory.is_symlink() or not cache_directory.is_dir()
    ):
        raise Lfm2CausalSupportRunnerError(
            "Content-addressed cache path is not an ordinary directory."
        )
    cached = (
        _load_cached_selection(cache_directory, expected_cache_key=request["cache_key"])
        if cache_directory.exists()
        else None
    )
    cache_hit = cached is not None
    if cached is None:
        result = select_lfm2_causal_semantic_supports(**math_arguments)
        publication = _publish_selection(cache_directory, result)
        cached = _load_cached_selection(
            cache_directory, expected_cache_key=request["cache_key"]
        )
        if cached is None:
            raise AssertionError("Fresh causal support publication did not reload.")
    else:
        publication = _read_json(
            cache_directory / "complete.json", "causal support publication"
        )
    body = {
        "schema": CAUSAL_SUPPORT_RUN_SCHEMA,
        "status": "complete-cached-offline-independent-support-selection",
        "cache_key": request["cache_key"],
        "cache_hit": cache_hit,
        "result_sha256": cached.result_sha256,
        "publication_sha256": publication["publication_sha256"],
        "output_directory": str(cache_directory),
        "trajectory_audit_sha256": audit_sha,
        "trajectory_completion_sha256": trajectory_completion["completion_sha256"],
        "feature_pack_artifact_sha256": feature_pack.artifact_sha256,
        "fit_prompt_manifest_sha256": manifest["manifest_sha256"],
        "input_codebook_artifact_sha256": codebook.artifact_sha256,
        "output_refit_result_sha256": output_result.result_sha256,
        "continuous_child_state_count": cached.receipt["continuous_child_state_count"],
        "effective_child_parameter_count": cached.receipt[
            "effective_child_parameter_count"
        ],
        "input_reselected_donor_count": cached.receipt["input_reselected_donor_count"],
        "output_reselected_donor_count": cached.receipt[
            "output_reselected_donor_count"
        ],
        "input_output_support_searches_independent": True,
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "training_steps": 0,
        "model_objects_constructed": 0,
    }
    return {**body, "run_sha256": canonical_sha256(body)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-audit-json", type=Path, required=True)
    parser.add_argument("--trajectory-completion-json", type=Path, required=True)
    parser.add_argument("--expected-trajectory-audit-file-sha256", required=True)
    parser.add_argument("--expected-trajectory-audit-sha256", required=True)
    parser.add_argument("--expected-trajectory-completion-sha256", required=True)
    parser.add_argument("--feature-tensors-safetensors", type=Path, required=True)
    parser.add_argument("--feature-receipt-json", type=Path, required=True)
    parser.add_argument("--expected-feature-tensor-file-sha256", required=True)
    parser.add_argument("--expected-feature-artifact-sha256", required=True)
    parser.add_argument("--fit-prompt-manifest-json", type=Path, required=True)
    parser.add_argument("--expected-fit-prompt-manifest-sha256", required=True)
    parser.add_argument("--donor-table-safetensors", type=Path, required=True)
    parser.add_argument("--donor-table-key", required=True)
    parser.add_argument("--expected-donor-table-file-sha256", required=True)
    parser.add_argument("--target-table-safetensors", type=Path, required=True)
    parser.add_argument("--target-table-key", required=True)
    parser.add_argument("--expected-target-table-file-sha256", required=True)
    parser.add_argument("--input-codebook-directory", type=Path, required=True)
    parser.add_argument("--expected-input-codebook-artifact-sha256", required=True)
    parser.add_argument("--output-coefficients-safetensors", type=Path, required=True)
    parser.add_argument(
        "--output-coefficients-key",
        default="output_semantic_code_coefficients",
    )
    parser.add_argument("--expected-output-coefficients-file-sha256", required=True)
    parser.add_argument("--output-refit-receipt-json", type=Path, required=True)
    parser.add_argument("--expected-output-refit-cache-key", required=True)
    parser.add_argument("--expected-output-refit-result-sha256", required=True)
    parser.add_argument("--physical-child-parameter-count", type=int, required=True)
    parser.add_argument("--maximum-combined-coefficient-count", type=int, required=True)
    parser.add_argument("--input-support-width", type=int, default=32)
    parser.add_argument("--output-support-width", type=int, default=32)
    parser.add_argument("--maximum-input-donor-repairs", type=int, required=True)
    parser.add_argument("--maximum-output-donor-repairs", type=int, required=True)
    parser.add_argument("--position-bin-count", type=int, default=4)
    parser.add_argument("--context-hash-bucket-count", type=int, default=256)
    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument("--solve-ridge", type=float, default=1e-8)
    parser.add_argument(
        "--solve-dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "schema": CAUSAL_SUPPORT_RUN_SCHEMA,
                    "status": "failed",
                    "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "message": str(exc),
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
        raise
    print(
        json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
