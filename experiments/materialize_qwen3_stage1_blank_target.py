#!/usr/bin/env python3
"""Materialize and publish the one canonical Qwen3 stage-1 blank target.

This is an artifact-construction command, not an experiment sweep.  Its seed,
shape, dtype, artifact name, and parameter budget are constants.  It creates
the checkpoint on the explicitly supplied staging filesystem, releases the
constructor model, reloads the saved checkpoint from disk, certifies its
config/count/state-key topology and state digest, and only then publishes it.

Publication reuses Neural Forge's fail-closed checkpoint publisher: an atomic
directory rename is preferred; EACCES/EXDEV filesystems use an incomplete
marker plus a complete hash-verified copy.  The staging tree is removed after
and only after successful verified-copy publication.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.replay_qwen_unified_v8_commit import (
    _artifact_tree,
    _publish_staged_checkpoint,
)
from neural_forge.hf_blank_checkpoint import (
    _parameter_counts,
    _state_dict_sha256,
)
from neural_forge.hf_qwen_target import (
    TARGET_REPORT_FILENAME,
    materialize_qwen3_target,
    plan_qwen3_target,
)


SCHEMA = "neural-forge-qwen3-stage1-blank-publication-v1"
CERTIFICATE_SCHEMA = "neural-forge-qwen3-stage1-blank-certificate-v1"
RECEIPT_SCHEMA = "neural-forge-qwen3-stage1-blank-publication-receipt-v1"

EXACT_PARAMETER_COUNT = 1_410_567_168
CANONICAL_SEED = 0
CANONICAL_DTYPE = "float16"
CANONICAL_ARTIFACT_NAME = "qwen3_stage1_blank_1410567168_fp16_seed0"
CANONICAL_STAGING_NAME = f".{CANONICAL_ARTIFACT_NAME}.staging"
KNOWN_EXISTING_STAGING_NAME = "qwen3_exact_1p410567168b_blank_seed0_fp16_20260712"
KNOWN_EXISTING_MODEL_SHA256 = (
    "4147674b29180d75e8f5bb518b5fc69349d475bb5c8bd49a8f5e7e0ecf364a45"
)
CERTIFICATE_FILENAME = "neural_forge_stage1_blank_certificate.json"
RECEIPT_SUFFIX = ".publication.json"

CANONICAL_CONFIG: dict[str, Any] = {
    "vocab_size": 151_936,
    "hidden_size": 1_536,
    "intermediate_size": 5_120,
    "num_hidden_layers": 30,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "tie_word_embeddings": False,
    "attention_bias": False,
}


class Stage1BlankArtifactError(RuntimeError):
    """The canonical stage-1 artifact cannot be proven safe to publish."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise Stage1BlankArtifactError(
            "Stage-1 provenance must be finite canonical JSON."
        ) from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> Any:
    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage1BlankArtifactError(
                    f"Duplicate JSON key {key!r} in {path.name}."
                )
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                Stage1BlankArtifactError(
                    f"Non-finite JSON constant {value!r} in {path.name}."
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Stage1BlankArtifactError(f"Cannot read strict JSON {path}.") from error


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def canonical_plan() -> dict[str, Any]:
    """Return the one-point exact shape plan; no alternative seed/shape exists."""

    plan = plan_qwen3_target(
        EXACT_PARAMETER_COUNT,
        vocab_size=CANONICAL_CONFIG["vocab_size"],
        constraints={
            "hidden_sizes": [CANONICAL_CONFIG["hidden_size"]],
            "intermediate_sizes": [CANONICAL_CONFIG["intermediate_size"]],
            "layer_counts": [CANONICAL_CONFIG["num_hidden_layers"]],
            "query_head_counts": [CANONICAL_CONFIG["num_attention_heads"]],
            "key_value_head_counts": [CANONICAL_CONFIG["num_key_value_heads"]],
            "head_dims": [CANONICAL_CONFIG["head_dim"]],
        },
        tie_word_embeddings=False,
        attention_bias=False,
        require_under_budget=True,
    )
    if plan["parameter_count"] != EXACT_PARAMETER_COUNT:
        raise Stage1BlankArtifactError(
            "Canonical stage-1 planner did not produce the exact parameter budget."
        )
    if dict(plan["config"]) != CANONICAL_CONFIG:
        raise Stage1BlankArtifactError(
            "Canonical stage-1 planner changed the fixed counted config."
        )
    return plan


def _saved_safetensor_keys(staged: Path) -> tuple[str, ...]:
    index_path = staged / "model.safetensors.index.json"
    if index_path.is_file():
        index = _strict_json(index_path)
        weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise Stage1BlankArtifactError(
                "Safetensors index has no non-empty weight_map."
            )
        keys = tuple(sorted(str(key) for key in weight_map))
    else:
        shards = tuple(sorted(staged.glob("*.safetensors")))
        if not shards:
            raise Stage1BlankArtifactError("Staged checkpoint has no safetensors.")
        try:
            from safetensors import safe_open
        except ImportError as error:  # pragma: no cover - dependency boundary
            raise RuntimeError(
                "Safetensors is required for stage-1 certification."
            ) from error
        found: list[str] = []
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                found.extend(str(key) for key in handle.keys())
        keys = tuple(sorted(found))
    if not keys or len(set(keys)) != len(keys):
        raise Stage1BlankArtifactError(
            "Saved safetensor state keys are empty or duplicated across shards."
        )
    return keys


def _checked_config(config: Any) -> dict[str, Any]:
    actual: dict[str, Any] = {}
    for key, expected in CANONICAL_CONFIG.items():
        value = getattr(config, key, None)
        if value != expected:
            raise Stage1BlankArtifactError(
                f"Reloaded config mismatch for {key}: {value!r} != {expected!r}."
            )
        actual[key] = value
    if getattr(config, "model_type", None) != "qwen3":
        raise Stage1BlankArtifactError("Reloaded checkpoint is not model_type qwen3.")
    return actual


def _fresh_reload_certificate(
    staged: Path,
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    """Fresh-load the saved bytes and certify count/config/key/state identity."""

    try:
        import torch
        import transformers
        from transformers import Qwen3ForCausalLM
    except (ImportError, AttributeError) as error:  # pragma: no cover
        raise RuntimeError(
            "Fresh stage-1 certification requires torch and Qwen3 Transformers."
        ) from error

    saved_keys = _saved_safetensor_keys(staged)
    model = Qwen3ForCausalLM.from_pretrained(
        staged,
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    try:
        model.eval()
        config = _checked_config(model.config)
        counts = _parameter_counts(model)
        if counts["unique_parameter_count"] != EXACT_PARAMETER_COUNT:
            raise Stage1BlankArtifactError(
                "Reloaded unique parameter count is not the exact stage-1 budget."
            )
        if counts["logical_parameter_count"] != EXACT_PARAMETER_COUNT:
            raise Stage1BlankArtifactError(
                "Reloaded untied target contains shared parameter references."
            )
        reloaded_keys = tuple(sorted(model.state_dict()))
        if reloaded_keys != saved_keys:
            raise Stage1BlankArtifactError(
                "Fresh-reloaded state keys differ from saved safetensor keys."
            )
        dtype_names = {
            str(parameter.dtype).replace("torch.", "")
            for parameter in model.parameters()
        }
        if dtype_names != {CANONICAL_DTYPE}:
            raise Stage1BlankArtifactError(
                f"Fresh-reloaded parameter dtypes are not only FP16: {dtype_names}."
            )
        populated_gradients = sum(
            int(parameter.grad is not None) for parameter in model.parameters()
        )
        if populated_gradients:
            raise Stage1BlankArtifactError(
                "Fresh-reloaded blank target unexpectedly has populated gradients."
            )
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        if input_weight is output_weight or (
            input_weight.untyped_storage().data_ptr()
            == output_weight.untyped_storage().data_ptr()
        ):
            raise Stage1BlankArtifactError(
                "Fresh-reloaded stage-1 embedding and LM head are tied."
            )
        state_sha256 = _state_dict_sha256(torch, model)
        expected_state_sha256 = materialization.get("random_initialization", {}).get(
            "state_dict_sha256"
        )
        if state_sha256 != expected_state_sha256:
            raise Stage1BlankArtifactError(
                "Fresh-reloaded state digest differs from constructor state digest."
            )
        saved_config = _strict_json(staged / "config.json")
        if not isinstance(saved_config, Mapping):
            raise Stage1BlankArtifactError("Saved config.json is not an object.")
        for key, expected in CANONICAL_CONFIG.items():
            if saved_config.get(key) != expected:
                raise Stage1BlankArtifactError(f"Saved config mismatch for {key}.")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            staged,
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer_length = len(tokenizer)
        expected_tokenizer_length = materialization.get("tokenizer", {}).get("length")
        if tokenizer_length != expected_tokenizer_length:
            raise Stage1BlankArtifactError(
                "Fresh-reloaded tokenizer length differs from materialization."
            )
        return {
            "performed": True,
            "checkpoint_source": str(staged),
            "local_files_only": True,
            "pretrained_parent_loaded": False,
            "config": config,
            "config_sha256": _sha256(dict(saved_config)),
            "parameter_counts": counts,
            "exact_parameter_count": EXACT_PARAMETER_COUNT,
            "exact_parameter_count_matches": True,
            "parameter_dtype": CANONICAL_DTYPE,
            "only_expected_parameter_dtype": True,
            "tie_word_embeddings": False,
            "embedding_and_head_storage_distinct": True,
            "populated_gradients": 0,
            "state_tensor_count": len(reloaded_keys),
            "saved_state_keys_sha256": _sha256(list(saved_keys)),
            "reloaded_state_keys_sha256": _sha256(list(reloaded_keys)),
            "saved_and_reloaded_state_keys_exact": True,
            "constructor_state_dict_sha256": expected_state_sha256,
            "reloaded_state_dict_sha256": state_sha256,
            "constructor_and_reload_state_exact": True,
            "tokenizer_length": tokenizer_length,
            "software": {
                "torch": str(torch.__version__),
                "transformers": str(transformers.__version__),
            },
        }
    finally:
        del model
        gc.collect()


def _validate_materialization(report: Mapping[str, Any], staged: Path) -> None:
    if report.get("seed") != CANONICAL_SEED:
        raise Stage1BlankArtifactError("Materializer did not use canonical seed zero.")
    if report.get("torch_dtype") != CANONICAL_DTYPE:
        raise Stage1BlankArtifactError("Materializer did not save an FP16 target.")
    if report.get("shape") != CANONICAL_CONFIG:
        raise Stage1BlankArtifactError("Materialized shape is not canonical stage-1.")
    count = report.get("parameter_count")
    if not isinstance(count, Mapping) or (
        count.get("formula") != EXACT_PARAMETER_COUNT
        or count.get("actual_unique_model_parameters") != EXACT_PARAMETER_COUNT
        or count.get("matches") is not True
    ):
        raise Stage1BlankArtifactError(
            "Materialized target did not certify the exact parameter budget."
        )
    random_initialization = report.get("random_initialization")
    if not isinstance(random_initialization, Mapping) or (
        random_initialization.get("pretrained_child_weights_loaded") is not False
        or random_initialization.get("pretrained_model_loader_called") is not False
    ):
        raise Stage1BlankArtifactError(
            "Materialization is not proven to be config-created random blank weights."
        )
    training = report.get("training")
    if not isinstance(training, Mapping) or (
        training.get("performed") is not False
        or training.get("optimizer_steps") != 0
        or training.get("backward_calls") != 0
    ):
        raise Stage1BlankArtifactError(
            "Blank artifact materialization unexpectedly performed training."
        )
    saved = _strict_json(staged / TARGET_REPORT_FILENAME)
    if saved != report:
        raise Stage1BlankArtifactError(
            "Saved Qwen target report differs from the in-memory report."
        )


def _remove_verified_staging(
    staged: Path,
    *,
    expected_tree: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> bool:
    if not staged.exists() and not staged.is_symlink():
        if publication.get("mode") != "atomic-replace":
            raise Stage1BlankArtifactError(
                "Staging vanished without atomic directory publication."
            )
        return False
    if publication.get("mode") != "verified-copy-fallback":
        raise Stage1BlankArtifactError(
            "Staging remained after a publication mode other than verified copy."
        )
    if staged.is_symlink() or not staged.is_dir() or staged.resolve() != staged:
        raise Stage1BlankArtifactError(
            "Refusing to remove an unexpected or linked staging path."
        )
    if _artifact_tree(staged) != dict(expected_tree):
        raise Stage1BlankArtifactError(
            "Refusing to remove staging because its final tree changed."
        )
    shutil.rmtree(staged)
    if staged.exists() or staged.is_symlink():
        raise Stage1BlankArtifactError("Verified staging cleanup did not complete.")
    return True


def materialize_stage1_blank(
    *,
    staging_root: Path,
    artifact_root: Path,
    tokenizer_model: str | Path,
    local_files_only: bool = True,
    existing_staged: Path | None = None,
) -> dict[str, Any]:
    """Materialize, fresh-certify, and publish exactly one canonical artifact."""

    staging_root = staging_root.expanduser().resolve(strict=True)
    artifact_root = artifact_root.expanduser().resolve(strict=True)
    if not staging_root.is_dir() or not artifact_root.is_dir():
        raise Stage1BlankArtifactError(
            "Staging and artifact roots must be directories."
        )
    staged = (
        staging_root / CANONICAL_STAGING_NAME
        if existing_staged is None
        else existing_staged.expanduser().resolve(strict=True)
    )
    destination = artifact_root / CANONICAL_ARTIFACT_NAME
    receipt_path = artifact_root / f"{CANONICAL_ARTIFACT_NAME}{RECEIPT_SUFFIX}"
    for label, path in (
        ("destination", destination),
        ("publication receipt", receipt_path),
    ):
        if path.exists() or path.is_symlink():
            raise Stage1BlankArtifactError(
                f"Canonical {label} path must be absent: {path}."
            )
    plan = canonical_plan()
    if existing_staged is None:
        if staged.exists() or staged.is_symlink():
            raise Stage1BlankArtifactError(
                f"Canonical staging path must be absent: {staged}."
            )
        try:
            import transformers
        except ImportError as error:  # pragma: no cover
            raise RuntimeError(
                "Transformers is required for tokenizer loading."
            ) from error
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(tokenizer_model),
            local_files_only=bool(local_files_only),
            trust_remote_code=False,
        )
        materialization = materialize_qwen3_target(
            plan,
            staged,
            tokenizer,
            seed=CANONICAL_SEED,
            torch_dtype=CANONICAL_DTYPE,
            safe_serialization=True,
        )
        del tokenizer
        gc.collect()
        source_mode = "materialized-single-fixed-seed"
        source_model_sha256 = None
    else:
        expected_staged = staging_root / KNOWN_EXISTING_STAGING_NAME
        if staged != expected_staged or staged.is_symlink() or not staged.is_dir():
            raise Stage1BlankArtifactError(
                "--existing-staged must be the known exact direct staging artifact: "
                f"{expected_staged}."
            )
        if (staged / CERTIFICATE_FILENAME).exists():
            raise Stage1BlankArtifactError(
                "Known staged artifact already contains a publication certificate."
            )
        model_path = staged / "model.safetensors"
        if not model_path.is_file():
            raise Stage1BlankArtifactError(
                "Known staged artifact has no single model.safetensors file."
            )
        source_model_sha256 = _file_sha256(model_path)
        if source_model_sha256 != KNOWN_EXISTING_MODEL_SHA256:
            raise Stage1BlankArtifactError(
                "Known staged model.safetensors differs from its recorded SHA-256."
            )
        loaded_report = _strict_json(staged / TARGET_REPORT_FILENAME)
        if not isinstance(loaded_report, Mapping):
            raise Stage1BlankArtifactError(
                "Known staged materialization report is not an object."
            )
        materialization = dict(loaded_report)
        source_mode = "verify-and-publish-known-existing-staging"
    _validate_materialization(materialization, staged)
    reload_certificate = _fresh_reload_certificate(staged, materialization)
    payload_tree = _artifact_tree(staged)
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        "status": "certified",
        "artifact_name": CANONICAL_ARTIFACT_NAME,
        "seed": CANONICAL_SEED,
        "dtype": CANONICAL_DTYPE,
        "shape": dict(CANONICAL_CONFIG),
        "parameter_count": EXACT_PARAMETER_COUNT,
        "shape_plan_sha256": plan["plan_sha256"],
        "materialization_report": TARGET_REPORT_FILENAME,
        "materialization_report_sha256": _sha256(materialization),
        "fresh_reload": reload_certificate,
        "payload_tree_before_certificate": payload_tree,
        "no_training": True,
        "no_parent_weights_loaded": True,
        "single_fixed_seed": True,
        "source_mode": source_mode,
        "known_existing_model_safetensors_sha256": source_model_sha256,
    }
    certificate["certificate_payload_sha256"] = _sha256(certificate)
    _write_json_exclusive(staged / CERTIFICATE_FILENAME, certificate)
    complete_tree = _artifact_tree(staged)
    publication = _publish_staged_checkpoint(
        staged,
        destination,
        expected_tree=complete_tree,
    )
    published_tree = _artifact_tree(destination)
    if published_tree != complete_tree:
        raise Stage1BlankArtifactError(
            "Published stage-1 tree changed after verified publication."
        )
    staging_removed = _remove_verified_staging(
        staged,
        expected_tree=complete_tree,
        publication=publication,
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "published",
        "operation_schema": SCHEMA,
        "destination": str(destination),
        "certificate": CERTIFICATE_FILENAME,
        "complete_artifact_tree": complete_tree,
        "publication": publication,
        "published_tree_reverified": True,
        "fallback_staging_removed_after_verification": staging_removed,
        "canonical_seed": CANONICAL_SEED,
        "single_artifact": True,
        "source_mode": source_mode,
        "known_existing_model_safetensors_sha256": source_model_sha256,
    }
    receipt["receipt_payload_sha256"] = _sha256(receipt)
    _write_json_exclusive(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-model",
        default="Qwen/Qwen3-4B-Base",
        help="Tokenizer source only; no model weights are loaded from this identifier.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require tokenizer files to already exist locally (default: true).",
    )
    parser.add_argument(
        "--existing-staged",
        type=Path,
        help=(
            "Verify and publish the one known stage-1 artifact instead of "
            "constructing weights. The path and model SHA-256 are fail-closed."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = materialize_stage1_blank(
        staging_root=args.staging_root,
        artifact_root=args.artifact_root,
        tokenizer_model=args.tokenizer_model,
        local_files_only=args.local_files_only,
        existing_staged=args.existing_staged,
    )
    print(json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
