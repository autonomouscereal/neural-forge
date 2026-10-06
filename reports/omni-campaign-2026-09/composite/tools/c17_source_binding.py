"""C17 metadata bindings for completed S3 and owner-recorded K3 artifacts.

This module is torch-free. It authorizes no install, capability or training claim.
Callers retain their actual tensor, native profile, paired-row and lineage checks.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

OWNER_DIAGNOSTIC_POLICY = "chimera_omni.owner_finite_diagnostics_only.2026-10-06.v1"
S3_BUNDLE_SCHEMA = "chimera_omni.s3_native_bundle.v1"
CHECKPOINT_SCHEMA = "chimera_omni.checkpoint.v1"
LAYER_GATES = frozenset(("G1_routed_write_heldout", "G1_y_lat_heldout",
                         "G2_held_over_fit", "G3_prefix_vs_zero_child"))
SHA = re.compile(r"[0-9a-f]{64}")
MAX_METADATA_BYTES = 4 << 20


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON metadata key")
        result[key] = value
    return result


def _json_float(value: str) -> float:
    result = float(value)
    require(math.isfinite(result), "nonfinite JSON metadata number")
    return result


def pinned_metadata(ref: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Read only an explicit, bounded JSON file; never follow a tensor filename."""
    require(isinstance(ref, dict) and _sha(ref.get("sha256")), "invalid metadata reference")
    require(isinstance(expected, dict) and _sha(expected.get("sha256")), "missing completion file pin")
    require(ref["sha256"] == expected["sha256"], "metadata reference differs from completion pin")
    size = expected.get("bytes")
    require(type(size) is int and 0 < size <= MAX_METADATA_BYTES, "invalid metadata byte bound")
    raw_path = ref.get("path")
    require(isinstance(raw_path, str) and raw_path, "missing metadata path")
    path = Path(raw_path)
    require(path.is_absolute() and path.suffix == ".json" and path.is_file(), "metadata path must be an existing absolute JSON file")
    before = path.stat()
    require(before.st_size == size, "metadata byte count differs from completion")
    data = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "metadata changed while read")
    require(hashlib.sha256(data).hexdigest() == expected["sha256"], "metadata SHA256 mismatch")
    value = json.loads(data, parse_constant=_reject_constant,
                       object_pairs_hook=_json_object, parse_float=_json_float)
    require(isinstance(value, dict), "metadata must be a JSON object")
    return value


def host_checkpoint_binding(receipt: dict[str, Any], *, source_binding_policy: str | None = None,
                            metadata_refs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve checkpoint identities without relabeling a completed text stage.

    Legacy standalone imports keep their original no-waiver contract. The new
    explicit path verifies saved config/index/runtime-contract bytes named by the
    completed S3 bundle; it preserves the observed-runtime qualifier verbatim.
    The caller separately pins the receipt itself and its capture reference.
    """
    require(isinstance(receipt, dict), "host receipt must be an object")
    if receipt.get("schema") == "chimera_omni.import.qwen35.v1":
        checkpoint = receipt.get("checkpoint")
        require(isinstance(checkpoint, dict) and
                all(_sha(checkpoint.get(k)) for k in ("config_sha256", "index_sha256")),
                "legacy host import lacks checkpoint identities")
        require(receipt.get("output_gate_waiver") is None and
                checkpoint.get("directory_shard_files_match_index") is True,
                "legacy host import has a waiver or unmatched shard set")
        return {"receipt_kind": "legacy_qwen35_import", "checkpoint": dict(checkpoint),
                "source_binding_policy": None}
    require(source_binding_policy == OWNER_DIAGNOSTIC_POLICY, "completed S3 requires explicit owner diagnostic policy")
    require(receipt.get("schema") == S3_BUNDLE_SCHEMA and receipt.get("complete") is True,
            "host source must be an actual completed S3 bundle")
    require(receipt.get("synthetic_fixture") is False and receipt.get("development_artifact") is True,
            "fixture or non-development bundle cannot use this path")
    require(receipt.get("numeric_fidelity_is_diagnostic") is True and receipt.get("ledger_band_logit_mask") is True,
            "S3 diagnostic/masking contract missing")
    require(_sha(receipt.get("native_state_sha256")), "S3 native state identity missing")
    files = receipt.get("files")
    require(isinstance(files, dict) and isinstance(metadata_refs, dict) and
            set(metadata_refs) == {"config", "index", "runtime_contract"}, "S3 exact saved metadata references missing")
    names = {"config": "native/config.json", "index": "native/model.safetensors.index.json",
             "runtime_contract": "runtime_contract.json"}
    objects = {name: pinned_metadata(metadata_refs.get(name), files.get(filename)) for name, filename in names.items()}
    config, index, contract = (objects[name] for name in names)
    imported = receipt.get("import_receipt")
    require(isinstance(imported, dict) and imported.get("schema") == "chimera_omni.import.qwen35.v1",
            "S3 nested native import receipt missing")
    require(config.get("schema") == CHECKPOINT_SCHEMA and config.get("config") == imported.get("config"),
            "saved native config differs from imported native profile")
    require(contract.get("schema") == "chimera_omni.s3_observed_runtime.v1" and
            contract.get("reference_kind") == "observed_hf_5_12_1" and contract.get("transformers_version") == "5.12.1",
            "unsupported or synthetic observed-runtime contract")
    require(contract.get("declared_output_gate") == "swish" and contract.get("effective_output_gate") == "sigmoid" and
            contract.get("gate_scope") == "full_attention_output_only" and
            contract.get("gdn_output_gate") == "silu" and contract.get("mlp_activation") == "silu" and
            contract.get("historical_training_gate") == "unknown", "S3 selected runtime/gate profile differs")
    native_config = config["config"]
    require(contract.get("config") == native_config and
            (native_config.get("metadata") or {}).get("attention_output_gate") == "sigmoid_assumed_declared_swish",
            "S3 observed-runtime qualifier differs from saved native profile")
    require(config.get("ledger") == receipt.get("ledger") == contract.get("ledger") and isinstance(receipt.get("ledger"), dict),
            "saved S3 ledger identities differ")
    require((config.get("source") or {}).get("contract_sha256") == files["runtime_contract.json"]["sha256"],
            "saved S3 config does not bind this runtime contract")
    require((index.get("metadata") or {}).get("schema") == CHECKPOINT_SCHEMA, "saved native index schema differs")
    weights = index.get("weight_map")
    hashes = (index.get("metadata") or {}).get("tensor_sha256")
    require(isinstance(weights, dict) and weights and isinstance(hashes, dict) and set(hashes) == set(weights) and
            all(_sha(hashes[name]) for name in weights), "saved native tensor inventory/digests incomplete")
    shards = set(weights.values())
    require(all(isinstance(name, str) and re.fullmatch(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors", name) for name in shards),
            "unsafe or unsupported saved native shard names")
    indexed = {"native/" + name for name in shards}
    declared = {name for name in files if name.startswith("native/") and name.endswith(".safetensors")}
    require(indexed == declared and all(_sha(files[name].get("sha256")) and
            type(files[name].get("bytes")) is int and files[name]["bytes"] > 0 for name in indexed),
            "completed native shard set differs from saved index")
    reload = receipt.get("reload_receipt")
    require(isinstance(reload, dict) and reload.get("state_sha256") == receipt["native_state_sha256"],
            "saved native reload state differs from completion")
    return {"receipt_kind": S3_BUNDLE_SCHEMA,
            "source_binding_policy": source_binding_policy,
            "checkpoint": {"config_sha256": files[names["config"]]["sha256"],
                           "index_sha256": files[names["index"]]["sha256"],
                           "completed_receipt_shard_set_matches_index": True},
            "verification_scope": "Pinned metadata and completed-receipt shard inventory; actual tensors remain caller-verified",
            "runtime_contract_sha256": files["runtime_contract.json"]["sha256"],
            "native_state_sha256": receipt["native_state_sha256"],
            "attention_output_gate": native_config["metadata"]["attention_output_gate"],
            "vision_runtime_installed": receipt.get("vision_runtime_installed"),
            "all_source_child_complete": receipt.get("all_source_child_complete"),
            "capability_evaluated": receipt.get("capability_evaluated")}


def k3_layer_source_admissible(value: dict[str, Any], *, version: str, constants: dict[str, Any],
                               source_binding_policy: str | None = None,
                               layer: int | None = None, rung: int | None = None) -> bool:
    """Legacy accepted ancestry or explicit coherent record-mode diagnostics.

    A numeric refused verdict is preserved, never converted into accepted. This
    predicate does not verify tensors/done markers/prefixes; callers still do.
    """
    if not isinstance(value, dict) or value.get("version") != version or value.get("constants") != constants:
        return False
    if source_binding_policy is None:
        return value.get("verdict") == "accepted"
    if source_binding_policy != OWNER_DIAGNOSTIC_POLICY:
        return False
    if value.get("mode") != "record":
        return (value.get("verdict") == "accepted" and value.get("mode") in (None, "gate") and
                (value.get("enforced") is None or value.get("enforced") is True))
    if value.get("enforced") is not False or value.get("verdict") not in ("accepted", "refused"):
        return False
    if layer is not None and (type(value.get("layer")) is not int or value["layer"] != layer):
        return False
    if rung is not None and (type(value.get("rung")) is not int or value["rung"] != rung):
        return False
    gates, failing = value.get("gates"), value.get("failing")
    if not isinstance(gates, dict) or set(gates) != LAYER_GATES or not isinstance(failing, list):
        return False
    if any(not isinstance(gate, dict) or type(gate.get("pass")) is not bool for gate in gates.values()):
        return False
    if any(not isinstance(name, str) for name in failing) or len(set(failing)) != len(failing):
        return False
    failed = {name for name, gate in gates.items() if not gate["pass"]}
    return set(failing) == failed and ((value["verdict"] == "refused") == bool(failed))


def k3_acceptance_core_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    # *_this_run additions do not rewrite immutable original ancestry.
    fields = ("version", "constants", "mode", "enforced", "verdict", "gates", "failing", "layer", "rung")
    return all(left.get(key) == right.get(key) for key in fields)
