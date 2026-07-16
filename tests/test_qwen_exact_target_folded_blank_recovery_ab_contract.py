from __future__ import annotations

import importlib
import hashlib
import json
import sys
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from neural_forge import hf_experimental_child_artifact as child_artifact
from neural_forge import hf_teacher_token_replay as teacher_replay
from neural_forge.hf_compiled_child_cache import model_state_fingerprint
from neural_forge.hf_progressive_target_bridge import _dense_model_state_sha256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

runner = importlib.import_module("run_qwen_exact_target_folded_blank_recovery_ab")
builder = importlib.import_module("prepare_qwen_exact_target_folded_blank_recovery_ab")

_TINY_STATE_KEY_SHA256 = runner.canonical_sha256(["scale", "weight"])
_TINY_CONFIG_PAYLOAD_SHA256 = runner.canonical_sha256({"width": 4})


def _digest(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return runner.stage._file_sha256(path)


def _write_canonical_json(path: Path, value: Any) -> str:
    path.write_bytes(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    return runner.stage._file_sha256(path)


def _with_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
    payload = dict(value)
    payload[field] = runner.canonical_sha256(payload)
    return payload


class _TinyFoldedModel(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.width = width
        self.weight = nn.Parameter(
            torch.arange(width * 3, dtype=torch.float32).reshape(3, width) / 13.0
        )
        self.register_buffer("scale", torch.linspace(0.25, 1.0, width))

    def save_pretrained(self, destination: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        save_file(self.state_dict(), destination / "model.safetensors")
        (destination / "config.json").write_text(
            json.dumps({"width": self.width}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, source: Path) -> _TinyFoldedModel:
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        model = cls(width=int(config["width"]))
        model.load_state_dict(
            load_file(source / "model.safetensors", device="cpu"),
            strict=True,
        )
        return model


class _TinyFoldedTokenizer:
    def save_pretrained(self, destination: Path) -> None:
        (destination / "tokenizer_config.json").write_text(
            json.dumps({"model_max_length": 128}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "tokenizer.json").write_text(
            json.dumps(
                {"model": {"type": "WordLevel", "vocab": {"<unk>": 0}}},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _contract_model_state_fingerprint(model: nn.Module) -> dict[str, Any]:
    result = dict(model_state_fingerprint(model))
    result.update(
        {
            "unique_parameter_count": runner.EXACT_UNIQUE_PARAMETERS,
            "logical_parameter_count": runner.EXACT_UNIQUE_PARAMETERS,
            "parameter_topology_sha256": _digest("parameter-topology"),
            "buffer_topology_sha256": _digest("buffer-topology"),
        }
    )
    return result


def _folded_native_certificate(_model: nn.Module) -> dict[str, Any]:
    return {
        "schema": "test-native-dense-certificate-v1",
        "exact_unique_parameters": runner.EXACT_UNIQUE_PARAMETERS,
        "logical_parameters": runner.EXACT_UNIQUE_PARAMETERS,
        "structure_sha256": _digest("folded-native-structure"),
        "state_key_sha256": _digest("folded-native-registered-keys"),
        "state_key_count": 2,
    }


def _folded_runtime_audit(_model: nn.Module, _path: Path) -> dict[str, int]:
    return {
        "adapter_or_registry_module_count": 0,
        "optimizer_parameter_count": 0,
        "source_sized_sidecar_parameter_count": 0,
        "parent_runtime_reference_count": 0,
    }


def _folded_artifact_hooks(
    *,
    model_loader: Any = None,
) -> Any:
    return runner.FoldedArtifactVerificationHooks(
        model_loader=(
            _TinyFoldedModel.from_pretrained if model_loader is None else model_loader
        ),
        native_certifier=_folded_native_certificate,
        runtime_auditor=_folded_runtime_audit,
    )


@contextmanager
def _folded_materialization_context(report_bytes: bytes) -> Iterator[bytes]:
    yield report_bytes


def _stepwise_token_event(step: int) -> dict[str, Any]:
    prompt = [100 + step, 200 + step]
    return {
        "step": step,
        "rows": [
            {
                "prompt_input_ids": prompt,
                "rollout_input_ids": [*prompt, 300 + step, 400 + step],
            }
        ],
    }


def _tokenizer_fingerprint(label: str) -> dict[str, Any]:
    body = {
        "class": f"tests.{label}Tokenizer",
        "tokenizer_length": 1024,
        "vocabulary_entry_count": 1024,
        "full_vocabulary_stream_sha256": _digest(f"{label}-vocabulary"),
        "backend_tokenizer_json_sha256": _digest(f"{label}-backend"),
        "special_token_ids": {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "unk_token_id": 3,
            "sep_token_id": None,
            "cls_token_id": None,
            "mask_token_id": None,
            "all_special_ids": [0, 1, 2, 3],
        },
        "padding_side": "left",
        "truncation_side": "right",
    }
    return {**body, "sha256": teacher_replay._canonical_sha256(body)}


def _source_model_fingerprint() -> dict[str, Any]:
    model_class = "tests.SourceModel"
    state_body = {
        "schema": teacher_replay.MODEL_STATE_FINGERPRINT_SCHEMA,
        "model_class": model_class,
        "state_dict_sha256": _digest("teacher-source-state"),
        "compiler_state_sha256": _digest("teacher-source-compiler-state"),
        "state_key_sha256": _digest("teacher-source-state-keys"),
        "state_manifest_sha256": _digest("teacher-source-manifest"),
        "state_key_count": 12,
        "logical_state_elements": 4096,
        "unique_parameter_count": 4000,
        "logical_parameter_count": 4000,
        "parameter_topology_sha256": _digest("teacher-source-parameter-topology"),
        "buffer_topology_sha256": _digest("teacher-source-buffer-topology"),
        "all_state_tensors_finite": True,
    }
    state = {**state_body, "sha256": teacher_replay._canonical_sha256(state_body)}
    body = {
        "class": model_class,
        "model_state_fingerprint": state,
        "config_sha256": _digest("teacher-source-config"),
        "generation_config_sha256": _digest("teacher-source-generation-config"),
    }
    return {**body, "sha256": teacher_replay._canonical_sha256(body)}


def _teacher_replay_payload() -> dict[str, Any]:
    calibration = [
        f"Frozen recovery calibration row {index:04d} remains training-only."
        for index in range(640)
    ]
    teacher_texts = [
        "alpha prompt teacher answer",
        "beta prompt teacher answer",
    ]
    on_policy = [
        ["event one exact text"],
        ["event two exact text"],
        ["event three exact text"],
    ]
    stepwise = [_stepwise_token_event(step) for step in range(1, 5)]
    parameters = {
        "steps": 4,
        "batch_size": 1,
        "max_length": 128,
        "response_loss_mode": "auto",
        "calibration_order": "original",
        "stepwise_donor_max_new_tokens": 16,
        "on_policy_max_new_tokens": 24,
        "teacher_generate_max_new_tokens": 48,
        "on_policy_interval": 2,
        "device": "cuda:1",
    }
    source_tokenizer = _tokenizer_fingerprint("Source")
    candidate_tokenizer = source_tokenizer
    source_model = _source_model_fingerprint()
    frozen_stepwise = runner.hf_recovery._freeze_stepwise_replay_bank(stepwise)
    frozen_on_policy = runner.hf_recovery._freeze_on_policy_replay_bank(on_policy)
    body = {
        "schema": teacher_replay.TEACHER_TOKEN_REPLAY_SCHEMA,
        "status": "frozen",
        "calibration_texts": calibration,
        "stepwise_scheduled_steps": [1, 2, 3, 4],
        "on_policy_scheduled_steps": [1, 2, 4],
        "stepwise_replay_bank": stepwise,
        "on_policy_replay_bank": on_policy,
        "precomputed_teacher_generated_texts": teacher_texts,
        "tokenizer_fingerprints": {
            "source": source_tokenizer,
            "candidate": candidate_tokenizer,
        },
        "source_model_fingerprint": source_model,
        "parameters": parameters,
        "provenance_sha256": {
            "calibration_texts": (
                runner.hf_recovery._ordered_recovery_texts_sha256(calibration)
            ),
            "parameters": teacher_replay._canonical_sha256(parameters),
            "source_tokenizer": source_tokenizer["sha256"],
            "candidate_tokenizer": candidate_tokenizer["sha256"],
            "source_model": source_model["sha256"],
            "source_model_state": source_model["model_state_fingerprint"][
                "state_dict_sha256"
            ],
            "stepwise_replay_bank": (
                runner.hf_recovery._stepwise_replay_bank_sha256(frozen_stepwise)
            ),
            "on_policy_replay_bank": (
                runner.hf_recovery._on_policy_replay_bank_sha256(frozen_on_policy)
            ),
            "precomputed_teacher_generated_texts": (
                runner.hf_recovery._ordered_recovery_texts_sha256(teacher_texts)
            ),
        },
    }
    return {**body, "payload_sha256": teacher_replay._canonical_sha256(body)}


def _fresh_behavior_binding(teacher_payload: dict[str, Any]) -> dict[str, Any]:
    inputs = runner.recovery_inputs
    calibration = teacher_payload["calibration_texts"]
    selection_rows = []
    reservoir_rows = []
    targets = []
    behavior_probes = []
    generation_prompts = []
    for index in range(inputs.FRESH_CANDIDATE_COUNT):
        token_ids = [500 + index * 16 + offset for offset in range(8)]
        token_sha = inputs.split_tools._token_sequence_sha256(token_ids)
        prompt_id = f"prompt-{token_sha}"
        prompt = f"Fresh evaluation candidate {index:02d}: solve the unseen task."
        first_prefix = f"Answer{index:02d}"
        lcp4_prefix = f"Answer{index:02d} continues correctly"
        decoded_continuation = f"{lcp4_prefix} with coherent reasoning"
        continuation = [900 + index * 4 + offset for offset in range(8)]
        first_id = f"parent-first-action-{prompt_id}"
        lcp4_id = f"parent-lcp4-{prompt_id}"
        generation_id = f"parent-generation-{prompt_id}"
        source_prompt_sha = f"{index + 1:064x}"
        selection_rows.append(
            {
                "reservoir_index": index,
                "prompt_id": prompt_id,
                "token_ids_sha256": token_sha,
                "token_count": len(token_ids),
                "source_line_number": index + 1,
                "source_prompt_sha256": source_prompt_sha,
                "decoded_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
            }
        )
        reservoir_rows.append(
            {
                "reservoir_index": index,
                "prompt_id": prompt_id,
                "token_ids": token_ids,
                "token_ids_sha256": token_sha,
                "decoded_prompt": prompt,
                "decoded_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "source_line_number": index + 1,
                "source_prompt_sha256": source_prompt_sha,
            }
        )
        behavior_probes.extend(
            [
                {
                    "id": first_id,
                    "category": "parent-first-action-decoded-prefix",
                    "prompt": prompt,
                    "kind": "literal_prefix",
                    "expected_prefix": first_prefix,
                },
                {
                    "id": lcp4_id,
                    "category": "parent-lcp4-decoded-prefix",
                    "prompt": prompt,
                    "kind": "literal_prefix",
                    "expected_prefix": lcp4_prefix,
                },
            ]
        )
        generation_prompts.append(
            {
                "id": generation_id,
                "category": "parent-lcp4-decoded-prefix",
                "prompt": prompt,
                "kind": "literal_prefix",
                "expected_prefix": lcp4_prefix,
                "expected_prefix_generated_token_count": 4,
                "relevance_rules": [],
                "require_generation_health": True,
            }
        )
        targets.append(
            {
                "index": index,
                "prompt_id": prompt_id,
                "prompt_token_ids": token_ids,
                "prompt_token_ids_sha256": token_sha,
                "prompt_token_count": len(token_ids),
                "decoded_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "generated_continuation_token_ids": continuation,
                "generated_continuation_token_ids_sha256": (
                    inputs.split_tools._token_sequence_sha256(continuation)
                ),
                "generated_continuation_token_count": len(continuation),
                "generated_continuation_decoded_text": decoded_continuation,
                "generated_continuation_decoded_text_sha256": hashlib.sha256(
                    decoded_continuation.encode("utf-8")
                ).hexdigest(),
                "pre_eos_token_count": len(continuation),
                "first_action_token_ids": continuation[:1],
                "first_action_token_ids_sha256": (
                    inputs.split_tools._token_sequence_sha256(continuation[:1])
                ),
                "first_action_decoded_prefix": first_prefix,
                "first_action_decoded_prefix_sha256": hashlib.sha256(
                    first_prefix.encode("utf-8")
                ).hexdigest(),
                "lcp4_token_ids": continuation[:4],
                "lcp4_token_ids_sha256": (
                    inputs.split_tools._token_sequence_sha256(continuation[:4])
                ),
                "lcp4_decoded_prefix": lcp4_prefix,
                "lcp4_decoded_prefix_sha256": hashlib.sha256(
                    lcp4_prefix.encode("utf-8")
                ).hexdigest(),
                "first_action_probe_id": first_id,
                "lcp4_probe_id": lcp4_id,
                "generation_prompt_id": generation_id,
            }
        )
    for index in range(inputs.FRESH_CANDIDATE_COUNT, 28):
        token_ids = [500 + index * 16 + offset for offset in range(8)]
        token_sha = inputs.split_tools._token_sequence_sha256(token_ids)
        prompt = f"Fresh reserve candidate {index:02d}: solve another unseen task."
        reservoir_rows.append(
            {
                "reservoir_index": index,
                "prompt_id": f"prompt-{token_sha}",
                "token_ids": token_ids,
                "token_ids_sha256": token_sha,
                "decoded_prompt": prompt,
                "decoded_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "source_line_number": index + 1,
                "source_prompt_sha256": f"{index + 1:064x}",
            }
        )
    suite = {
        "source": "fresh-token-domain-parent-greedy-v1",
        "text_evals": [],
        "choice_probes": [],
        "generation_prompts": generation_prompts,
        "behavior_probes": behavior_probes,
    }
    v4_preregistration = {
        "schema": inputs.PINNED_V4_PREREGISTRATION_SCHEMA,
        "source_filename": "preregistration.json",
        "file_sha256": _digest("v4-prereg"),
        "evaluation_contract_sha256": _digest("v4-evaluation"),
        "sealed_before_model_access": True,
    }
    corpus = {
        "file_sha256": inputs.PINNED_RECOVERY_CORPUS_SHA256,
        "nonblank_line_count": inputs.PINNED_RECOVERY_CORPUS_LINE_COUNT,
        "global_order": "source_prompt_sha256, then source_line_number",
    }
    recovery_calibration = {
        "text_count": len(calibration),
        "ordered_text_manifest_sha256": runner.canonical_sha256(calibration),
        "full_token_manifest_sha256": _digest("full-calibration-tokens"),
        "bounded_token_manifest_sha256": _digest("bounded-calibration-tokens"),
        "prefix_bookkeeping_max_tokens": 128,
        "prefix_bookkeeping_tokens_per_row": 8,
        "full_tokenization_bound_separately": True,
    }
    reservoir_manifest = [
        {
            key: value
            for key, value in row.items()
            if key not in {"token_ids", "decoded_prompt"}
        }
        for row in reservoir_rows
    ]
    reservoir_provenance_body = {
        "status": "candidate-reservoir-frozen-before-parent-load",
        "v4_preregistration": v4_preregistration,
        "v4_exclusion_splits": {},
        "corpus": corpus,
        "recovery_calibration": recovery_calibration,
        "reservoir_prompt_count": len(reservoir_rows),
        "reservoir_manifest_sha256": runner.canonical_sha256(reservoir_manifest),
        "reservoir_order": "pinned-corpus-global-order-after-static-exclusions",
        "reservoir_rows_hold_token_ids_and_decoded_text": True,
        "preselection_statistics": {
            "records_considered": (
                inputs.PINNED_RECOVERY_CORPUS_LINE_COUNT
                - inputs.PINNED_RECOVERY_CORPUS_DUPLICATE_SOURCE_ROW_COUNT
            ),
            "unique_source_prompt_count": (
                inputs.PINNED_RECOVERY_CORPUS_LINE_COUNT
                - inputs.PINNED_RECOVERY_CORPUS_DUPLICATE_SOURCE_ROW_COUNT
            ),
            "duplicate_source_row_count": (
                inputs.PINNED_RECOVERY_CORPUS_DUPLICATE_SOURCE_ROW_COUNT
            ),
            "skipped_candidate_text_present_in_calibration": (
                inputs.PINNED_RECOVERY_CORPUS_LINE_COUNT
                - inputs.PINNED_RECOVERY_CORPUS_DUPLICATE_SOURCE_ROW_COUNT
                - len(reservoir_rows)
            ),
            "skipped_under_minimum_prompt_tokens": 0,
            "skipped_over_token_limit": 0,
            "skipped_exact_prompt_collision": 0,
            "skipped_meaningful_prefix_collision": 0,
            "skipped_decoded_calibration_collision": 0,
            "skipped_duplicate_decoded_prompt": 0,
            "accepted_into_reservoir": len(reservoir_rows),
        },
        "sealed_exclusion_manifest_used_model_free": True,
        "sealed_prompts_supplied_to_model_or_trainer": False,
        "sealed_outputs_or_selection_scores_computed": False,
        "contains_sealed_data": False,
        "parent_model_loaded_during_reservoir_freeze": False,
        "parent_forward_or_generate_during_reservoir_freeze": False,
        "identity_tokenizer_used_during_reservoir_freeze": True,
    }
    reservoir_provenance = {
        **reservoir_provenance_body,
        "candidate_reservoir_payload_sha256": runner.canonical_sha256(
            reservoir_provenance_body
        ),
    }
    candidate_reservoir = {
        "provenance": reservoir_provenance,
        "rows": reservoir_rows,
        "rows_sha256": runner.canonical_sha256(reservoir_rows),
    }
    replay_derivation = inputs._derive_fresh_candidate_selection(
        reservoir_rows=reservoir_rows,
        teacher_payload=teacher_payload,
    )
    disjointness_audit = inputs._fresh_candidate_replay_disjointness_audit(
        decoded_prompts=[prompt["prompt"] for prompt in generation_prompts],
        prompt_token_ids=[target["prompt_token_ids"] for target in targets],
        teacher_payload=teacher_payload,
    )
    selection_body = {
        "schema": inputs.FRESH_SELECTION_SCHEMA,
        "status": "frozen-after-teacher-replay-exclusion",
        "mode": inputs.BEHAVIOR_MODE_FRESH,
        "causal_ab_eligible": True,
        "v4_preregistration": v4_preregistration,
        "v4_exclusion_splits": {},
        "corpus": corpus,
        "recovery_calibration": recovery_calibration,
        "candidate_reservoir": candidate_reservoir,
        "replay_exclusion_derivation": replay_derivation,
        "selected_candidate": {
            "prompt_count": len(selection_rows),
            "minimum_token_count": 8,
            "maximum_token_count": 8,
            "minimum_required_token_count": 8,
            "maximum_allowed_token_count": 128,
            "prompt_rows": selection_rows,
            "prompt_manifest_sha256": runner.canonical_sha256(selection_rows),
            "decoded_prompt_manifest_sha256": runner.canonical_sha256(
                [probe["prompt"] for probe in generation_prompts]
            ),
        },
        "selection_statistics": {
            "preselection": reservoir_provenance_body["preselection_statistics"],
            "replay_exclusion_reason_counts": replay_derivation["reason_counts"],
            "replay_disjoint_survivor_count": replay_derivation["survivor_count"],
            "replay_disjoint_survivor_margin": replay_derivation["survivor_margin"],
        },
        "disjointness": {
            "exact_token_prompt_overlap_count": 0,
            "meaningful_prefix_overlap_count": 0,
            "decoded_candidate_text_in_recovery_calibration_count": 0,
            "minimum_meaningful_prefix_tokens": 8,
            "recovery_calibration_prefixes_bounded_to_candidate_max_tokens": True,
        },
        "sealed_exclusion_manifest_used_model_free": True,
        "sealed_prompts_supplied_to_model_or_trainer": False,
        "sealed_outputs_or_selection_scores_computed": False,
        "contains_sealed_data": False,
        "candidate_reservoir_frozen_before_parent_load": True,
        "final_candidate_frozen_before_parent_load": False,
        "parent_model_loaded_before_final_selection": True,
        "teacher_replay_generated_before_final_selection": True,
        "teacher_replay_used_for_exclusion_only": True,
        "parent_forward_or_score_during_final_selection": False,
        "tokenizer_called_during_final_selection": False,
        "child_or_arm_outputs_accessed_during_final_selection": False,
        "final_replay_disjointness_audit_sha256": disjointness_audit[
            "payload_sha256"
        ],
    }
    selection = {
        **selection_body,
        "payload_sha256": runner.canonical_sha256(selection_body),
    }
    generation_by_id = {prompt["id"]: prompt for prompt in generation_prompts}
    health_receipts = []
    for target in targets:
        generation = generation_by_id[target["generation_prompt_id"]]
        decoded_continuation = target["generated_continuation_decoded_text"]
        health = inputs.hf_tools._generation_health(
            {
                **generation,
                "output": generation["prompt"] + decoded_continuation,
                "diagnostics": inputs.hf_tools._generation_diagnostics(
                    decoded_continuation,
                    target["generated_continuation_token_ids"][
                        : target["pre_eos_token_count"]
                    ],
                ),
            },
            min_continuation_chars=24,
            min_unique_token_ratio=0.20,
        )
        assert health["passed"] is True
        health_receipts.append(
            {"generation_prompt_id": generation["id"], "health": health}
        )
    adapter_body = {
        "schema": inputs.TOKEN_DOMAIN_ADAPTER_SCHEMA,
        "status": "frozen-parent-targets",
        "mode": inputs.BEHAVIOR_MODE_FRESH,
        "causal_ab_eligible": True,
        "selection": selection,
        "source_model_fingerprint_sha256": teacher_payload["source_model_fingerprint"][
            "sha256"
        ],
        "source_model_state_dict_sha256": teacher_payload["source_model_fingerprint"][
            "model_state_fingerprint"
        ]["state_dict_sha256"],
        "identity_tokenizer_fingerprint_sha256": teacher_payload[
            "tokenizer_fingerprints"
        ]["source"]["sha256"],
        "generation": {
            "do_sample": False,
            "max_new_tokens": 16,
            "minimum_real_pre_eos_tokens": 4,
            "batch_size": 1,
            "parent_model_loaded_once_for_replay_and_targets": True,
        },
        "recovery_replay_disjointness_audit": disjointness_audit,
        "recovery_replay_disjointness_audit_sha256": disjointness_audit[
            "payload_sha256"
        ],
        "targets": targets,
        "target_manifest_sha256": runner.canonical_sha256(targets),
        "suite_sha256": runner.canonical_sha256(suite),
        "behavior_probe_count": 48,
        "first_action_probe_count": 24,
        "lcp4_probe_count": 24,
        "generation_prompt_count": 24,
        "suite_prompt_exact_token_roundtrip_count": 24,
        "parent_full_decode_starts_canonical_prompt_count": 24,
        "parent_self_score_pass_count": 48,
        "parent_self_score_probe_count": 48,
        "parent_generation_health_pass_count": 24,
        "parent_generation_health_probe_count": 24,
        "parent_generation_health_receipts": health_receipts,
        "parent_generation_health_receipts_sha256": runner.canonical_sha256(
            health_receipts
        ),
        "sealed_exclusion_manifest_used_model_free": True,
        "sealed_prompt_decode_count": 0,
        "sealed_prompt_forward_count": 0,
        "sealed_prompt_output_count": 0,
        "sealed_prompts_supplied_to_model_or_trainer": False,
        "contains_sealed_data": False,
        "publication_authorized": False,
    }
    adapter = {
        **adapter_body,
        "payload_sha256": runner.canonical_sha256(adapter_body),
    }
    return inputs._binding_from_suite(
        suite=suite,
        mode=inputs.BEHAVIOR_MODE_FRESH,
        source_filename="preregistration.json",
        source_file_sha256=_digest("v4-prereg"),
        adapter_provenance=adapter,
    )


def _replay_payload() -> dict[str, Any]:
    frozen_teacher = _teacher_replay_payload()
    teacher_source_body = {
        "schema": runner.TEACHER_SOURCE_ARTIFACT_SCHEMA,
        "expected_parent_artifact_tree_sha256": _digest("parent-artifact-tree"),
        "actual_parent_artifact_tree_sha256": _digest("parent-artifact-tree"),
        "artifact_tree_file_count": 5,
        "artifact_tree_total_bytes": 8192,
        "tree_match_verified_before_model_load": True,
        "source_model_state_dict_sha256": frozen_teacher["source_model_fingerprint"][
            "model_state_fingerprint"
        ]["state_dict_sha256"],
    }
    return _with_hash(
        {
            "schema": runner.REPLAY_SCHEMA,
            "status": "frozen",
            "teacher_token_replay": frozen_teacher,
            "teacher_source_artifact": {
                **teacher_source_body,
                "sha256": runner.canonical_sha256(teacher_source_body),
            },
            "behavior_eval_suite": _fresh_behavior_binding(frozen_teacher),
            "frozen_before_preregistration": True,
            "contains_sealed_data": False,
            "publication_authorized": False,
        },
        "replay_payload_sha256",
    )


def _plan_payload() -> dict[str, Any]:
    return _with_hash(
        {
            "schema": runner.PLAN_SCHEMA,
            "status": "preregistered",
            "recovery_kwargs": {
                "steps": 4,
                "batch_size": 1,
                "max_length": 128,
                "response_loss_mode": "auto",
                "calibration_order": "original",
                "stepwise_donor_max_new_tokens": 16,
                "on_policy_max_new_tokens": 24,
                "teacher_generate_max_new_tokens": 48,
                "device": "cuda:6",
                "lr": 1e-5,
                "on_policy_interval": 2,
                "on_policy_anchor_kd_weight": 1.0,
                "stepwise_donor_kd_weight": 0.75,
                "behavior_eval_checkpoint_steps": [0, 2, 4],
                "behavior_max_probes": 48,
                "behavior_max_generation_prompts": 24,
                "behavior_max_new_tokens": 16,
                "adamw_foreach": False,
                "local_files_only": True,
                "trust_remote_code": False,
                "save_best_loss": False,
                "save_best_behavior": False,
            },
            "seeds": {
                "python": 17,
                "torch": 17,
                "cuda": 17,
                "data_order": 17,
            },
            "checkpoint_steps": [0, 2, 4],
            "retained_checkpoint_steps": [2, 4],
            "teacher_token_replay_payload_sha256": _teacher_replay_payload()[
                "payload_sha256"
            ],
            "zero_step_before_optimizer": True,
            "reset_rng_before_each_arm": True,
            "publication_authorized": False,
            "sealed_data_accessed": False,
        },
        "plan_payload_sha256",
    )


def _blank_binding() -> dict[str, Any]:
    return {
        "cache_path": "/authenticated/cache/identity",
        "compiler_identity_sha256": _digest("compiler-identity"),
        "parent_artifact_tree_sha256": _digest("parent-artifact-tree"),
        "receipt_sha256": _digest("cache-receipt"),
        "compiler_report_sha256": _digest("compiler-report"),
        "artifact_tree": {
            "file_count": 4,
            "total_bytes": 1234,
            "files": [],
            "tree_sha256": _digest("cache-tree"),
        },
        "artifact_tree_sha256": _digest("cache-tree"),
        "state": {
            "state_dict_sha256": _digest("blank-state"),
            "compiler_state_sha256": _digest("blank-compiler-state"),
            "state_key_sha256": _TINY_STATE_KEY_SHA256,
            "state_manifest_sha256": _digest("blank-manifest"),
            "parameter_topology_sha256": _digest("parameter-topology"),
            "buffer_topology_sha256": _digest("buffer-topology"),
            "unique_parameter_count": runner.EXACT_UNIQUE_PARAMETERS,
            "logical_parameter_count": runner.EXACT_UNIQUE_PARAMETERS,
        },
        "state_dict_sha256": _digest("blank-state"),
        "compiler_state_sha256": _digest("blank-compiler-state"),
        "state_key_sha256": _TINY_STATE_KEY_SHA256,
        "state_manifest_sha256": _digest("blank-manifest"),
        "parameter_topology_sha256": _digest("parameter-topology"),
        "buffer_topology_sha256": _digest("buffer-topology"),
        "config_file_sha256": _digest("config-file"),
        "config_payload_sha256": _TINY_CONFIG_PAYLOAD_SHA256,
        "config": dict(runner.blank.CANONICAL_CONFIG),
        "exact_unique_parameters": runner.EXACT_UNIQUE_PARAMETERS,
        "fresh_reload_verified": True,
        "model_loaded_for_authentication": False,
        "capability_proven": False,
        "publication_receipt": False,
    }


@pytest.fixture
def contract_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prereg_path = tmp_path / "v4_prereg.json"
    run_path = tmp_path / "v4_run.json"
    winner_path = tmp_path / "v4_winner.json"
    replay_path = tmp_path / "replay.json"
    plan_path = tmp_path / "plan.json"
    prereg = {
        "schema": "fake-authenticated-v4-prereg",
        "source_v3_sha256": _digest("v3-root"),
        "basis_sha256": _digest("basis-root"),
        "mask_sha256": _digest("mask-root"),
    }
    run = {
        "sealed_forward_count": 0,
        "sealed_evaluation_accessed": False,
        "publication_attempted": False,
        "frontier_sha256": _digest("frontier"),
    }
    winner_selection = {
        "candidate_only": True,
        "sealed_evaluation_accessed": False,
        "publication_attempted": False,
        "winner_sha256": _digest("winner"),
    }
    hashes = {
        "prereg": _write_json(prereg_path, prereg),
        "run": _write_json(run_path, run),
        "winner": _write_json(winner_path, winner_selection),
        "replay": _write_json(replay_path, _replay_payload()),
        "plan": _write_json(plan_path, _plan_payload()),
    }
    fake_contract = SimpleNamespace(
        evaluation_contract_sha256=_digest("v4-evaluation"),
    )
    winner = {
        "arm_id": "gud-winner",
        "arm_index": 2,
        "step": 24,
        "portable_factor_state_sha256": _digest("portable-v4"),
        "runtime_factor_state_sha256": _digest("runtime-v4"),
        "factor_pack": str(tmp_path / "winner.pt"),
        "factor_pack_sha256": _digest("factor-pack-file"),
        "candidate_frontier": {
            "path": str(tmp_path / "frontier.json"),
            "sha256": _digest("frontier"),
        },
    }

    def authenticate_with_history(_path: Path, prereg_sha: str) -> SimpleNamespace:
        receipt = {
            "schema": "test-v4-authentication-receipt-v1",
            "mode": "current-live",
            "preregistration_sha256": prereg_sha,
            "evaluation_contract_sha256": fake_contract.evaluation_contract_sha256,
            "implementation_contract_sha256": _digest("v4-implementation"),
            "live_implementation_contract_sha256": _digest("v4-live-implementation"),
            "historical_source": None,
        }
        return SimpleNamespace(
            contract=fake_contract,
            receipt=SimpleNamespace(as_dict=lambda: dict(receipt)),
        )

    monkeypatch.setattr(
        runner.tournament_history,
        "authenticate_guided_gud_candidate_preregistration_with_history",
        authenticate_with_history,
    )
    monkeypatch.setattr(
        runner.publication,
        "_winner_payload",
        lambda *_args, **_kwargs: (
            winner,
            {"schema": "neural-forge-target-native-factor-pack-v4"},
        ),
    )
    monkeypatch.setattr(runner, "_compiled_blank_binding", lambda _v4: _blank_binding())
    monkeypatch.setattr(
        child_artifact,
        "model_state_fingerprint",
        _contract_model_state_fingerprint,
    )

    return {
        "paths": {
            "tournament_preregistration": prereg_path,
            "tournament_run_report": run_path,
            "winner_selection": winner_path,
            "replay_bank": replay_path,
            "recovery_plan": plan_path,
        },
        "hashes": hashes,
        "winner": winner,
    }


def _build(inputs) -> tuple[dict[str, Any], Any]:
    paths = inputs["paths"]
    hashes = inputs["hashes"]
    return runner.build_recovery_ab_preregistration_payload(
        tournament_preregistration=paths["tournament_preregistration"],
        tournament_preregistration_sha256=hashes["prereg"],
        tournament_run_report=paths["tournament_run_report"],
        tournament_run_report_sha256=hashes["run"],
        winner_selection=paths["winner_selection"],
        winner_selection_sha256=hashes["winner"],
        replay_bank=paths["replay_bank"],
        replay_bank_sha256=hashes["replay"],
        recovery_plan=paths["recovery_plan"],
        recovery_plan_sha256=hashes["plan"],
        builder_path=Path(builder.__file__).resolve(),
    )


def _write_prereg(tmp_path: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    path = tmp_path / "ab_preregistration.json"
    return path, _write_json(path, payload)


def test_prereg_binds_candidate_blank_code_data_and_symmetric_schedule(
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)

    assert payload["status"] == "experimental-unaccepted"
    assert payload["v4_candidate"]["candidate_only"] is True
    assert payload["v4_candidate"]["sealed_evaluation_accessed"] is False
    assert payload["publication_authorized"] is False
    assert payload["sealed_data_accessed"] is False
    assert payload["accepted_artifact"] is False
    assert payload["exact_unique_parameters"] == 1_410_567_168
    assert payload["compiled_blank"]["fresh_reload_verified"] is True
    assert payload["compiled_blank"]["state_dict_sha256"] == _digest("blank-state")
    assert payload["compiled_blank"]["parameter_topology_sha256"] == _digest(
        "parameter-topology"
    )
    assert [arm["arm_id"] for arm in payload["arms"]] == ["folded", "blank"]
    assert (
        len({arm["shared_execution_contract_sha256"] for arm in payload["arms"]}) == 1
    )
    assert payload["recovery_plan"]["on_policy_event_steps"] == [1, 2, 4]
    assert payload["replay_bank"]["on_policy_event_count"] == 3
    assert payload["replay_bank"]["stepwise_event_count"] == 4
    assert payload["recovery_plan"]["stepwise_donor_kd_weight"] == 0.75
    assert payload["recovery_plan"]["stepwise_replay_bank_required"] is True
    assert payload["replay_bank"]["source_model_state_dict_sha256"] == _digest(
        "teacher-source-state"
    )
    assert payload["recovery_plan"]["adamw_foreach"] is False
    assert payload["recovery_plan"]["recovery_kwargs"]["device"] == "cuda:6"
    assert (
        payload["replay_bank"]["teacher_token_replay_parameters"]["device"] == "cuda:1"
    )
    assert payload["maximum_retained_checkpoints_per_arm"] == 3
    assert payload["maximum_total_retained_checkpoints"] == 6
    receipt = payload["v4_candidate"]["preregistration_authentication_receipt"]
    assert receipt["preregistration_sha256"] == contract_inputs["hashes"]["prereg"]
    assert receipt["evaluation_contract_sha256"] == _digest("v4-evaluation")
    assert payload["implementation"]["v4_historical_authenticator_sha256"] == (
        runner.stage._file_sha256(Path(runner.tournament_history.__file__).resolve())
    )
    roots = payload["transitive_sha256_roots"]
    assert _digest("v3-root") in roots["v4_preregistration"].values()
    assert _digest("basis-root") in roots["v4_preregistration"].values()
    assert _digest("mask-root") in roots["v4_preregistration"].values()
    assert (
        _digest("v4-implementation")
        in roots["v4_preregistration_authentication"].values()
    )
    assert payload["production_model_hook_implemented"] is True


def test_authentication_rebuilds_every_bound_source_and_rejects_drift(
    tmp_path: Path,
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)
    path, digest = _write_prereg(tmp_path, payload)
    contract = runner.authenticate_recovery_ab_preregistration(path, digest)
    assert contract.preflight_report()["status"] == "preflight-passed"
    assert contract.preflight_report()["model_weights_loaded"] is False

    replay_path = contract_inputs["paths"]["replay_bank"]
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["teacher_token_replay"]["calibration_texts"][0] = "tampered"
    _write_json(replay_path, replay)
    with pytest.raises(Exception, match="absent or changed"):
        runner.authenticate_recovery_ab_preregistration(path, digest)


def test_candidate_boundary_and_replay_schedule_fail_closed(
    contract_inputs,
) -> None:
    run_path = contract_inputs["paths"]["tournament_run_report"]
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["sealed_evaluation_accessed"] = True
    contract_inputs["hashes"]["run"] = _write_json(run_path, run)
    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="binding|fresh disjoint|sealed-free",
    ):
        _build(contract_inputs)

    run["sealed_evaluation_accessed"] = False
    contract_inputs["hashes"]["run"] = _write_json(run_path, run)
    replay_path = contract_inputs["paths"]["replay_bank"]
    replay = _replay_payload()
    replay["teacher_token_replay"]["on_policy_replay_bank"].pop()
    body = dict(replay)
    body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)
    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="Teacher-token replay",
    ):
        _build(contract_inputs)


@pytest.mark.parametrize(
    ("split", "sealed_flag_present", "sealed_flag"),
    [
        ("SEALED", True, False),
        ("sealed-holdout", True, False),
        (None, True, False),
        ("candidate", False, False),
        ("candidate", True, True),
    ],
)
def test_replay_suite_requires_explicit_candidate_sealed_free_metadata(
    contract_inputs,
    split,
    sealed_flag_present: bool,
    sealed_flag: bool,
) -> None:
    replay_path = contract_inputs["paths"]["replay_bank"]
    replay = _replay_payload()
    suite = replay["behavior_eval_suite"]
    if split is None:
        suite.pop("split")
    else:
        suite["split"] = split
    if sealed_flag_present:
        suite["sealed_data_accessed"] = sealed_flag
    else:
        suite.pop("sealed_data_accessed")
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="binding|fresh disjoint|sealed-free",
    ):
        _build(contract_inputs)


def test_explicit_json_diagnostic_suite_cannot_enter_production_causal_ab(
    contract_inputs,
) -> None:
    replay_path = contract_inputs["paths"]["replay_bank"]
    replay = _replay_payload()
    diagnostic_suite = {
        "source": "diagnostic.json",
        "text_evals": [],
        "choice_probes": [],
        "generation_prompts": [
            {"id": "diagnostic-generation", "prompt": "Diagnostic prompt."}
        ],
        "behavior_probes": [
            {
                "id": "diagnostic-probe",
                "prompt": "Diagnostic prompt.",
                "kind": "literal_prefix",
                "expected_prefix": "Only diagnostic",
            }
        ],
    }
    replay["behavior_eval_suite"] = runner.recovery_inputs._binding_from_suite(
        suite=diagnostic_suite,
        mode=runner.recovery_inputs.BEHAVIOR_MODE_EXPLICIT,
        source_filename="diagnostic.json",
        source_file_sha256=_digest("diagnostic-suite"),
        adapter_provenance=None,
    )
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)
    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="fresh disjoint causal adapter|diagnostic-only",
    ):
        _build(contract_inputs)


def test_replay_and_plan_reject_non_text_or_unknown_recovery_inputs(
    contract_inputs,
) -> None:
    replay_path = contract_inputs["paths"]["replay_bank"]
    replay = _replay_payload()
    replay["teacher_token_replay"]["calibration_texts"][0] = 7
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)
    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="Teacher-token replay",
    ):
        _build(contract_inputs)

    contract_inputs["hashes"]["replay"] = _write_json(
        replay_path,
        _replay_payload(),
    )
    plan_path = contract_inputs["paths"]["recovery_plan"]
    plan = _plan_payload()
    plan["recovery_kwargs"]["stepz"] = 4
    plan_body = dict(plan)
    plan_body.pop("plan_payload_sha256")
    plan["plan_payload_sha256"] = runner.canonical_sha256(plan_body)
    contract_inputs["hashes"]["plan"] = _write_json(plan_path, plan)
    with pytest.raises(runner.FoldedBlankRecoveryABError, match="unsupported names"):
        _build(contract_inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("local_files_only", False),
        ("local_files_only", "true"),
        ("trust_remote_code", True),
    ],
)
def test_recovery_plan_requires_local_only_non_remote_loading(
    contract_inputs,
    field: str,
    value: Any,
) -> None:
    plan_path = contract_inputs["paths"]["recovery_plan"]
    plan = _plan_payload()
    plan["recovery_kwargs"][field] = value
    plan_body = dict(plan)
    plan_body.pop("plan_payload_sha256")
    plan["plan_payload_sha256"] = runner.canonical_sha256(plan_body)
    contract_inputs["hashes"]["plan"] = _write_json(plan_path, plan)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="local-only trusted model loading",
    ):
        _build(contract_inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_length", 127),
        ("response_loss_mode", "none"),
        ("calibration_order", "domain-balanced"),
        ("stepwise_donor_max_new_tokens", 15),
        ("on_policy_max_new_tokens", 23),
        ("teacher_generate_max_new_tokens", 47),
    ],
)
def test_recovery_plan_must_exactly_match_frozen_teacher_replay_parameters(
    contract_inputs,
    field: str,
    value: Any,
) -> None:
    plan_path = contract_inputs["paths"]["recovery_plan"]
    plan = _plan_payload()
    plan["recovery_kwargs"][field] = value
    plan_body = dict(plan)
    plan_body.pop("plan_payload_sha256")
    plan["plan_payload_sha256"] = runner.canonical_sha256(plan_body)
    contract_inputs["hashes"]["plan"] = _write_json(plan_path, plan)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="Frozen teacher replay parameters differ",
    ):
        _build(contract_inputs)


def test_replay_rejects_tampered_teacher_tensor_provenance(
    contract_inputs,
) -> None:
    replay_path = contract_inputs["paths"]["replay_bank"]
    replay = _replay_payload()
    replay["teacher_token_replay"]["source_model_fingerprint"][
        "model_state_fingerprint"
    ]["state_dict_sha256"] = _digest("different-source-state")
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="Teacher-token replay",
    ):
        _build(contract_inputs)


def test_teacher_source_artifact_must_match_compiled_parent_lineage(
    contract_inputs,
) -> None:
    replay_path = contract_inputs["paths"]["replay_bank"]
    replay = _replay_payload()
    source = replay["teacher_source_artifact"]
    source["expected_parent_artifact_tree_sha256"] = _digest("wrong-parent-tree")
    source["actual_parent_artifact_tree_sha256"] = _digest("wrong-parent-tree")
    source_body = dict(source)
    source_body.pop("sha256")
    source["sha256"] = runner.canonical_sha256(source_body)
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="canonical parent",
    ):
        _build(contract_inputs)


def test_recovery_plan_must_bind_exact_teacher_replay_payload(
    contract_inputs,
) -> None:
    plan_path = contract_inputs["paths"]["recovery_plan"]
    plan = _plan_payload()
    plan["teacher_token_replay_payload_sha256"] = _digest("wrong-teacher-replay")
    plan_body = dict(plan)
    plan_body.pop("plan_payload_sha256")
    plan["plan_payload_sha256"] = runner.canonical_sha256(plan_body)
    contract_inputs["hashes"]["plan"] = _write_json(plan_path, plan)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="exact frozen teacher-token replay",
    ):
        _build(contract_inputs)


def test_stepwise_donor_requires_one_frozen_ordered_event_per_optimizer_step(
    contract_inputs,
) -> None:
    replay_path = contract_inputs["paths"]["replay_bank"]
    plan_path = contract_inputs["paths"]["recovery_plan"]
    replay = _replay_payload()
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    plan = _plan_payload()
    plan["recovery_kwargs"]["stepwise_donor_kd_weight"] = 0.75
    plan_body = dict(plan)
    plan_body.pop("plan_payload_sha256")
    plan["plan_payload_sha256"] = runner.canonical_sha256(plan_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)
    contract_inputs["hashes"]["plan"] = _write_json(plan_path, plan)

    payload, _v4 = _build(contract_inputs)
    assert payload["replay_bank"]["stepwise_event_count"] == 4
    assert payload["recovery_plan"]["stepwise_event_steps"] == [1, 2, 3, 4]
    assert payload["recovery_plan"]["stepwise_replay_bank_required"] is True
    shared = payload["shared_execution_contract"]
    assert shared["stepwise_event_count"] == 4
    assert shared["same_stepwise_replay_bank_required_for_both_arms"] is True
    assert shared["dynamic_stepwise_donor_generation_forbidden"] is True
    assert (
        shared["stepwise_replay_bank_schema"]
        == runner.hf_recovery._STEPWISE_REPLAY_SCHEMA
    )
    frozen = runner.hf_recovery._freeze_stepwise_replay_bank(
        replay["teacher_token_replay"]["stepwise_replay_bank"]
    )
    assert shared["stepwise_replay_bank_sha256"] == (
        runner.hf_recovery._stepwise_replay_bank_sha256(frozen)
    )
    assert (
        len({arm["shared_execution_contract_sha256"] for arm in payload["arms"]}) == 1
    )

    replay["teacher_token_replay"]["stepwise_replay_bank"].pop()
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)
    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="Teacher-token replay",
    ):
        _build(contract_inputs)

    replay["teacher_token_replay"].pop("stepwise_replay_bank")
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)
    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="Teacher-token replay",
    ):
        _build(contract_inputs)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda bank: bank.__setitem__(0, ["unsafe text replay"]),
        lambda bank: bank[0].__setitem__("step", 2),
        lambda bank: bank[0]["rows"][0].__setitem__("rollout_input_ids", [999, 301]),
        lambda bank: bank[0]["rows"][0].__setitem__("prompt_input_ids", [True, 201]),
    ],
)
def test_stepwise_donor_token_bank_rejects_unsafe_or_wrong_events(
    contract_inputs,
    mutate,
) -> None:
    replay_path = contract_inputs["paths"]["replay_bank"]
    plan_path = contract_inputs["paths"]["recovery_plan"]
    replay = _replay_payload()
    mutate(replay["teacher_token_replay"]["stepwise_replay_bank"])
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    plan = _plan_payload()
    plan["recovery_kwargs"]["stepwise_donor_kd_weight"] = 0.75
    plan_body = dict(plan)
    plan_body.pop("plan_payload_sha256")
    plan["plan_payload_sha256"] = runner.canonical_sha256(plan_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)
    contract_inputs["hashes"]["plan"] = _write_json(plan_path, plan)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="Teacher-token replay",
    ):
        _build(contract_inputs)


def test_resealed_stepwise_rollout_candidate_leak_is_rejected(
    contract_inputs,
) -> None:
    replay_path = contract_inputs["paths"]["replay_bank"]
    plan_path = contract_inputs["paths"]["recovery_plan"]
    replay = _replay_payload()
    teacher = replay["teacher_token_replay"]
    leaked_ids = replay["behavior_eval_suite"]["adapter_provenance"]["targets"][0][
        "prompt_token_ids"
    ]
    prompt_ids = teacher["stepwise_replay_bank"][0]["rows"][0]["prompt_input_ids"]
    teacher["stepwise_replay_bank"][0]["rows"][0]["rollout_input_ids"] = [
        *prompt_ids,
        *leaked_ids,
        777,
    ]
    frozen_stepwise = runner.hf_recovery._freeze_stepwise_replay_bank(
        teacher["stepwise_replay_bank"]
    )
    teacher["provenance_sha256"]["stepwise_replay_bank"] = (
        runner.hf_recovery._stepwise_replay_bank_sha256(frozen_stepwise)
    )
    teacher_body = dict(teacher)
    teacher_body.pop("payload_sha256")
    teacher["payload_sha256"] = teacher_replay._canonical_sha256(teacher_body)
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    plan = _plan_payload()
    plan["teacher_token_replay_payload_sha256"] = teacher["payload_sha256"]
    plan_body = dict(plan)
    plan_body.pop("plan_payload_sha256")
    plan["plan_payload_sha256"] = runner.canonical_sha256(plan_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)
    contract_inputs["hashes"]["plan"] = _write_json(plan_path, plan)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="complete frozen replay audit",
    ):
        _build(contract_inputs)


def test_fully_resealed_false_disjointness_receipt_is_rejected(
    contract_inputs,
) -> None:
    replay_path = contract_inputs["paths"]["replay_bank"]
    replay = _replay_payload()
    behavior = replay["behavior_eval_suite"]
    adapter = behavior["adapter_provenance"]
    audit = adapter["recovery_replay_disjointness_audit"]
    audit["training_text_counts"]["precomputed_teacher_generated_texts"] += 1
    audit["training_text_counts"]["total"] += 1
    audit["training_text_manifest_sha256"] = _digest("forged-training-text-manifest")
    audit_body = dict(audit)
    audit_body.pop("payload_sha256")
    audit["payload_sha256"] = runner.canonical_sha256(audit_body)
    adapter["recovery_replay_disjointness_audit_sha256"] = audit["payload_sha256"]
    adapter_body = dict(adapter)
    adapter_body.pop("payload_sha256")
    adapter["payload_sha256"] = runner.canonical_sha256(adapter_body)
    behavior_body = dict(behavior)
    behavior_body.pop("payload_sha256")
    behavior["payload_sha256"] = runner.canonical_sha256(behavior_body)
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="evidence changed after freezing",
    ):
        _build(contract_inputs)


def test_fully_resealed_false_reservoir_derivation_is_rejected(
    contract_inputs,
) -> None:
    replay_path = contract_inputs["paths"]["replay_bank"]
    replay = _replay_payload()
    behavior = replay["behavior_eval_suite"]
    adapter = behavior["adapter_provenance"]
    selection = adapter["selection"]
    derivation = selection["replay_exclusion_derivation"]
    last = derivation["classification_rows"][-1]
    assert last["classification"] == "replay_disjoint_survivor"
    last["classification"] = "bidirectional_recovery_text_collision"
    derivation["reason_counts"]["replay_disjoint_survivor"] -= 1
    derivation["reason_counts"]["bidirectional_recovery_text_collision"] += 1
    derivation["survivor_count"] -= 1
    derivation["survivor_margin"] -= 1
    derivation["classification_rows_sha256"] = runner.canonical_sha256(
        derivation["classification_rows"]
    )
    derivation_body = dict(derivation)
    derivation_body.pop("payload_sha256")
    derivation["payload_sha256"] = runner.canonical_sha256(derivation_body)
    selection_body = dict(selection)
    selection_body.pop("payload_sha256")
    selection["payload_sha256"] = runner.canonical_sha256(selection_body)
    adapter_body = dict(adapter)
    adapter_body.pop("payload_sha256")
    adapter["payload_sha256"] = runner.canonical_sha256(adapter_body)
    behavior_body = dict(behavior)
    behavior_body.pop("payload_sha256")
    behavior["payload_sha256"] = runner.canonical_sha256(behavior_body)
    replay_body = dict(replay)
    replay_body.pop("replay_payload_sha256")
    replay["replay_payload_sha256"] = runner.canonical_sha256(replay_body)
    contract_inputs["hashes"]["replay"] = _write_json(replay_path, replay)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="selection or replay-disjointness evidence changed",
    ):
        _build(contract_inputs)


def _receipt(body: dict[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["receipt_sha256"] = runner.canonical_sha256(result)
    return result


def _materialization_receipts(contract, tmp_path: Path) -> dict[str, dict[str, Any]]:
    blank = contract.raw["compiled_blank"]
    winner = contract.raw["v4_candidate"]["winner"]
    folded_model = _TinyFoldedModel()
    folded_initial_state_sha256 = _contract_model_state_fingerprint(folded_model)[
        "state_dict_sha256"
    ]
    bridge_native_child_state_sha256 = _digest("bridge-native-child-state")
    bridge_live_factor_state_sha256 = _digest("bridge-live-factor-state")
    materialized_bridge_state_sha256 = _dense_model_state_sha256(folded_model)
    experiment_contract = {
        "schema": runner.EXPERIMENTAL_RECOVERY_CONTRACT_SCHEMA,
        "recovery_ab_preregistration_sha256": contract.file_sha256,
        "recovery_ab_semantic_contract_sha256": contract.raw[
            "semantic_contract_sha256"
        ],
        "shared_execution_contract_sha256": contract.raw[
            "shared_execution_contract_sha256"
        ],
        "factor_state_sha256": winner["portable_factor_state_sha256"],
        "factor_payload_schema": contract.raw["v4_candidate"]["factor_pack_schema"],
        "native_child_state_sha256": bridge_native_child_state_sha256,
        "blank_control_state_sha256": bridge_native_child_state_sha256,
        "compiled_blank_cache_state_dict_sha256": blank["state_dict_sha256"],
        "restored_runtime_factor_state_sha256": winner["runtime_factor_state_sha256"],
        "restored_portable_factor_state_sha256": winner["portable_factor_state_sha256"],
        "bridge_native_child_state_sha256": bridge_native_child_state_sha256,
        "bridge_live_factor_state_sha256": bridge_live_factor_state_sha256,
        "exact_unique_parameters": runner.EXACT_UNIQUE_PARAMETERS,
        "publication_authorized": False,
        "sealed_data_accessed": False,
    }
    bridge_receipt = _receipt(
        {
            "schema": runner.EXPERIMENTAL_RECOVERY_MATERIALIZATION_RECEIPT_SCHEMA,
            "experiment_contract_sha256": runner.canonical_sha256(experiment_contract),
            "factor_state_sha256": winner["portable_factor_state_sha256"],
            "live_factor_state_sha256": bridge_live_factor_state_sha256,
            "native_child_state_sha256": bridge_native_child_state_sha256,
            "exact_unique_parameters": runner.EXACT_UNIQUE_PARAMETERS,
            "factor_payload_schema": contract.raw["v4_candidate"]["factor_pack_schema"],
            "status": "experimental-unaccepted",
            "publication_authorized": False,
            "sealed_data_accessed": False,
            "behavior_capability_proven": False,
            "recovery_training_required": True,
            "external_experiment_contract_revalidated_by_bridge": False,
            "external_factor_pack_revalidated_by_bridge": False,
            "live_factor_tensor_state_revalidated_by_bridge": True,
            "native_child_state_revalidated_by_bridge": True,
        }
    )
    bridge_report = {
        "schema": runner.EXACT_TARGET_EXPERIMENTAL_RECOVERY_MATERIALIZATION_SCHEMA,
        "status": "experimental-unaccepted",
        "exact_unique_parameters": runner.EXACT_UNIQUE_PARAMETERS,
        "exact_budget_match": True,
        "native_before": _folded_native_certificate(folded_model),
        "native_materialized": _folded_native_certificate(folded_model),
        "materialized_native_state_sha256": materialized_bridge_state_sha256,
        "native_child_state_sha256": bridge_native_child_state_sha256,
        "live_factor_state_sha256": bridge_live_factor_state_sha256,
        "materialized_registered_adapter_parameter_count": 0,
        "materialized_optimizer_parameter_count": 0,
        "materialized_registered_source_sized_sidecar_parameter_count": 0,
        "experimental_recovery_contract": experiment_contract,
        "experimental_recovery_receipt": bridge_receipt,
        "standalone_structure_materialized": True,
        "serialization_only_context": True,
        "serialized_state_must_match_materialized_state_sha256": True,
        "fresh_reload_and_recovery_required": True,
        "publication_authorized": False,
        "accepted_artifact": False,
        "sealed_data_accessed": False,
        "behavior_capability_proven": False,
        "external_factor_pack_revalidated_by_bridge": False,
        "external_factor_pack_receipt_bound": False,
        "contract_factor_state_sha256_is_external_label_only": True,
        "live_factor_state_sha256_is_bridge_authenticated": True,
    }
    bridge_report["materialization_transaction_sha256"] = runner.canonical_sha256(
        bridge_report
    )
    bridge_report_bytes = json.dumps(
        bridge_report,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    publication_index = len(list(tmp_path.glob("folded-artifact-publication-*")))
    artifact_publication_root = (
        tmp_path / f"folded-artifact-publication-{publication_index}"
    )
    artifact_receipt = child_artifact.publish_experimental_child_artifact(
        artifact_publication_root,
        model=folded_model,
        tokenizer=_TinyFoldedTokenizer(),
        materialization_context=_folded_materialization_context(bridge_report_bytes),
        expectations=child_artifact.ExperimentalChildArtifactExpectations(
            preregistration_sha256=contract.file_sha256,
            shared_execution_contract_sha256=contract.raw[
                "shared_execution_contract_sha256"
            ],
            config_payload_sha256=blank["config_payload_sha256"],
            state_key_sha256=blank["state_key_sha256"],
            parameter_topology_sha256=blank["parameter_topology_sha256"],
            buffer_topology_sha256=blank["buffer_topology_sha256"],
            exact_unique_parameters=runner.EXACT_UNIQUE_PARAMETERS,
            exact_logical_parameters=runner.EXACT_UNIQUE_PARAMETERS,
            arm_id="folded",
        ),
        model_loader=_TinyFoldedModel.from_pretrained,
        native_certifier=_folded_native_certificate,
        runtime_auditor=_folded_runtime_audit,
    )
    common = {
        "schema": runner.MATERIALIZATION_RECEIPT_SCHEMA,
        "status": "experimental-unaccepted",
        "exact_unique_parameters": runner.EXACT_UNIQUE_PARAMETERS,
        "config_payload_sha256": blank["config_payload_sha256"],
        "parameter_topology_sha256": blank["parameter_topology_sha256"],
        "buffer_topology_sha256": blank["buffer_topology_sha256"],
        "shared_execution_contract_sha256": contract.raw[
            "shared_execution_contract_sha256"
        ],
        "optimizer_steps": 0,
        "sealed_data_accessed": False,
        "publication_authorized": False,
        "accepted_artifact": False,
    }
    return {
        "folded": _receipt(
            {
                **common,
                "arm_id": "folded",
                "initial_state_sha256": folded_initial_state_sha256,
                "source_portable_factor_state_sha256": winner[
                    "portable_factor_state_sha256"
                ],
                "source_factor_pack_sha256": winner["factor_pack_sha256"],
                "factor_source_binding_sha256": contract.raw["v4_candidate"][
                    "factor_source_binding_sha256"
                ],
                "factor_pack_authenticated_separately": True,
                "materialization_evidence_sha256": bridge_receipt["receipt_sha256"],
                "materialization_transaction_sha256": bridge_report[
                    "materialization_transaction_sha256"
                ],
                "artifact_publication_root": str(artifact_publication_root),
                "artifact_receipt_sha256": artifact_receipt.receipt_sha256,
                "bridge_native_child_state_sha256": (bridge_native_child_state_sha256),
                "bridge_live_factor_state_sha256": bridge_live_factor_state_sha256,
                "materialized_bridge_state_sha256": materialized_bridge_state_sha256,
            }
        ),
        "blank": _receipt(
            {
                **common,
                "arm_id": "blank",
                "initial_state_sha256": blank["state_dict_sha256"],
                "source_portable_factor_state_sha256": None,
                "source_factor_pack_sha256": None,
                "factor_source_binding_sha256": None,
                "factor_pack_authenticated_separately": False,
                "materialization_evidence_sha256": blank["receipt_sha256"],
                "materialization_transaction_sha256": None,
                "artifact_publication_root": None,
                "artifact_receipt_sha256": None,
                "bridge_native_child_state_sha256": None,
                "bridge_live_factor_state_sha256": None,
                "materialized_bridge_state_sha256": None,
            }
        ),
    }


def _zero_step_receipts(contract, materialization) -> dict[str, dict[str, Any]]:
    common = {
        "schema": runner.ZERO_STEP_RECEIPT_SCHEMA,
        "status": "experimental-unaccepted",
        "shared_execution_contract_sha256": contract.raw[
            "shared_execution_contract_sha256"
        ],
        "replay_bank_payload_sha256": contract.raw["replay_bank"]["payload_sha256"],
        "checkpoint_step": 0,
        "optimizer_steps": 0,
        "sealed_data_accessed": False,
        "publication_authorized": False,
        "accepted_artifact": False,
    }
    values = {}
    for arm_id, correct in (("folded", 1), ("blank", 0)):
        behavior_breakdown = {
            "math": {
                "correct": correct,
                "total": 4,
                "pass_rate": correct / 4,
                "failed_ids": [f"probe-{index}" for index in range(correct + 1, 5)],
                "passed_ids": (["probe-1"] if correct else []),
            }
        }
        generation_breakdown = {
            "general": {
                "correct": correct,
                "total": 2,
                "pass_rate": correct / 2,
                "failed_ids": [
                    f"generation-{index}" for index in range(correct + 1, 3)
                ],
                "passed_ids": (["generation-1"] if correct else []),
            }
        }
        metrics = {
            "step": 0,
            "loss": None,
            "correct": correct,
            "total": 4,
            "pass_rate": correct / 4,
            "generation_health_correct": correct,
            "generation_health_total": 2,
            "generation_health_rate": correct / 2,
            "behavior_category_breakdown": behavior_breakdown,
            "generation_category_breakdown": generation_breakdown,
            "behavior_category_summary": (
                runner.hf_recovery._category_breakdown_summary(behavior_breakdown)
            ),
            "generation_category_summary": (
                runner.hf_recovery._category_breakdown_summary(generation_breakdown)
            ),
            "passed_ids": ["probe-1"] if correct else [],
            "generation_passed_prompts": (["generation prompt 1"] if correct else []),
        }
        values[arm_id] = _receipt(
            {
                **common,
                "arm_id": arm_id,
                "initial_state_sha256": materialization["receipts"][arm_id][
                    "initial_state_sha256"
                ],
                "metrics": metrics,
                "metrics_sha256": runner.canonical_sha256(metrics),
                "evaluator_implementation_sha256": contract.raw["implementation"][
                    "behavior_evaluator_implementation_sha256"
                ],
            }
        )
    return values


def test_materialization_and_zero_step_receipts_remain_unaccepted(
    tmp_path: Path,
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)
    path, digest = _write_prereg(tmp_path, payload)
    contract = runner.authenticate_recovery_ab_preregistration(path, digest)
    blank = contract.raw["compiled_blank"]
    artifact_hooks = _folded_artifact_hooks()
    materialization = runner.authenticate_materialization_receipts(
        contract,
        _materialization_receipts(contract, tmp_path),
        folded_artifact_hooks=artifact_hooks,
    )
    assert materialization["status"] == "materialization-authenticated"
    assert materialization["folded_and_blank_initial_states_distinct"] is True
    assert materialization["folded_artifact_independent_fresh_reload_verified"] is True
    assert materialization["model_loading_performed_by_this_authenticator"] is True
    assert materialization["optimizer_steps"] == 0
    folded_binding = materialization["authenticated_materialization_reports"]["folded"]
    assert (
        folded_binding["fresh_reload_state_dict_sha256"]
        == materialization["receipts"]["folded"]["initial_state_sha256"]
    )
    assert "materialized_cache_state_dict_sha256" not in folded_binding
    zero_step = runner.authenticate_zero_step_receipts(
        contract,
        materialization,
        _zero_step_receipts(contract, materialization),
        folded_artifact_hooks=artifact_hooks,
    )
    assert zero_step["status"] == "zero-step-authenticated"
    assert zero_step["training_launched"] is False
    assert zero_step["publication_authorized"] is False
    assert zero_step["sealed_data_accessed"] is False
    assert (
        zero_step["next_action"] == "authenticated-production-recovery-executor-ready"
    )

    for mutate in (
        lambda metrics: metrics.__setitem__("pass_rate", 0.5),
        lambda metrics: metrics["behavior_category_summary"].__setitem__(
            "hard_category_total", 99
        ),
        lambda metrics: metrics.__setitem__("unbound_metric", True),
    ):
        tampered_zero = _zero_step_receipts(contract, materialization)
        folded = tampered_zero["folded"]
        mutate(folded["metrics"])
        folded["metrics_sha256"] = runner.canonical_sha256(folded["metrics"])
        folded.pop("receipt_sha256")
        tampered_zero["folded"] = _receipt(folded)
        with pytest.raises(runner.FoldedBlankRecoveryABError):
            runner.authenticate_zero_step_receipts(
                contract,
                materialization,
                tampered_zero,
                folded_artifact_hooks=artifact_hooks,
            )

    for field, value in (
        ("preregistration_sha256", _digest("wrong-preregistration")),
        ("publication_authorized", True),
    ):
        tampered_materialization = json.loads(json.dumps(materialization))
        tampered_materialization[field] = value
        with pytest.raises(runner.FoldedBlankRecoveryABError):
            runner.authenticate_zero_step_receipts(
                contract,
                tampered_materialization,
                _zero_step_receipts(contract, materialization),
                folded_artifact_hooks=artifact_hooks,
            )

    extra_materialization = json.loads(json.dumps(materialization))
    extra_materialization["unbound"] = True
    with pytest.raises(runner.FoldedBlankRecoveryABError):
        runner.authenticate_zero_step_receipts(
            contract,
            extra_materialization,
            _zero_step_receipts(contract, materialization),
            folded_artifact_hooks=artifact_hooks,
        )

    bad = _materialization_receipts(contract, tmp_path)
    bad["blank"]["exact_unique_parameters"] -= 1
    with pytest.raises(runner.FoldedBlankRecoveryABError):
        runner.authenticate_materialization_receipts(
            contract,
            bad,
            folded_artifact_hooks=artifact_hooks,
        )

    forged_materialization = {
        "status": "materialization-authenticated",
        "receipts": {
            "folded": {"initial_state_sha256": _digest("forged-folded")},
            "blank": {"initial_state_sha256": _digest("forged-blank")},
        },
    }
    with pytest.raises(runner.FoldedBlankRecoveryABError):
        runner.authenticate_zero_step_receipts(
            contract,
            forged_materialization,
            _zero_step_receipts(contract, materialization),
            folded_artifact_hooks=artifact_hooks,
        )

    unrelated = _materialization_receipts(contract, tmp_path)
    unrelated["folded"]["materialization_evidence_sha256"] = blank["receipt_sha256"]
    unrelated["folded"] = _receipt(
        {
            key: value
            for key, value in unrelated["folded"].items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(runner.FoldedBlankRecoveryABError):
        runner.authenticate_materialization_receipts(
            contract,
            unrelated,
            folded_artifact_hooks=artifact_hooks,
        )

    sidecar = _materialization_receipts(contract, tmp_path)
    report_path = (
        Path(sidecar["folded"]["artifact_publication_root"])
        / child_artifact.BRIDGE_REPORT_FILENAME
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["materialized_registered_source_sized_sidecar_parameter_count"] = 1
    report.pop("materialization_transaction_sha256")
    report["materialization_transaction_sha256"] = runner.canonical_sha256(report)
    sidecar["folded"]["materialization_transaction_sha256"] = report[
        "materialization_transaction_sha256"
    ]
    _write_canonical_json(report_path, report)
    sidecar["folded"] = _receipt(
        {
            key: value
            for key, value in sidecar["folded"].items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(runner.FoldedBlankRecoveryABError):
        runner.authenticate_materialization_receipts(
            contract,
            sidecar,
            folded_artifact_hooks=artifact_hooks,
        )


def test_materialization_rejects_noncanonical_bridge_report_bytes(
    tmp_path: Path,
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)
    path, digest = _write_prereg(tmp_path, payload)
    contract = runner.authenticate_recovery_ab_preregistration(path, digest)
    receipts = _materialization_receipts(contract, tmp_path)
    report_path = (
        Path(receipts["folded"]["artifact_publication_root"])
        / child_artifact.BRIDGE_REPORT_FILENAME
    )
    report = json.loads(report_path.read_text(encoding="ascii"))
    _write_json(report_path, report)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="failed independent fresh reload",
    ):
        runner.authenticate_materialization_receipts(
            contract,
            receipts,
            folded_artifact_hooks=_folded_artifact_hooks(),
        )


def test_materialization_rejects_changed_folded_artifact_bytes(
    tmp_path: Path,
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)
    path, digest = _write_prereg(tmp_path, payload)
    contract = runner.authenticate_recovery_ab_preregistration(path, digest)
    receipts = _materialization_receipts(contract, tmp_path)
    artifact_path = (
        Path(receipts["folded"]["artifact_publication_root"])
        / child_artifact.ARTIFACT_DIRECTORY_NAME
        / "tokenizer.json"
    )
    artifact_path.write_bytes(artifact_path.read_bytes() + b"\n")

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="failed independent fresh reload",
    ):
        runner.authenticate_materialization_receipts(
            contract,
            receipts,
            folded_artifact_hooks=_folded_artifact_hooks(),
        )


def test_materialization_rejects_changed_external_artifact_receipt(
    tmp_path: Path,
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)
    path, digest = _write_prereg(tmp_path, payload)
    contract = runner.authenticate_recovery_ab_preregistration(path, digest)
    receipts = _materialization_receipts(contract, tmp_path)
    receipt_path = (
        Path(receipts["folded"]["artifact_publication_root"])
        / child_artifact.ARTIFACT_RECEIPT_FILENAME
    )
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="failed independent fresh reload",
    ):
        runner.authenticate_materialization_receipts(
            contract,
            receipts,
            folded_artifact_hooks=_folded_artifact_hooks(),
        )


def test_materialization_rejects_changed_independent_reload_evidence(
    tmp_path: Path,
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)
    path, digest = _write_prereg(tmp_path, payload)
    contract = runner.authenticate_recovery_ab_preregistration(path, digest)
    receipts = _materialization_receipts(contract, tmp_path)

    def changed_loader(source: Path) -> _TinyFoldedModel:
        loaded = _TinyFoldedModel.from_pretrained(source)
        with torch.no_grad():
            loaded.weight.add_(1.0)
        return loaded

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="failed independent fresh reload",
    ):
        runner.authenticate_materialization_receipts(
            contract,
            receipts,
            folded_artifact_hooks=_folded_artifact_hooks(model_loader=changed_loader),
        )


def test_materialization_rejects_legacy_bridge_only_folded_receipt(
    tmp_path: Path,
    contract_inputs,
) -> None:
    payload, _v4 = _build(contract_inputs)
    path, digest = _write_prereg(tmp_path, payload)
    contract = runner.authenticate_recovery_ab_preregistration(path, digest)
    receipts = _materialization_receipts(contract, tmp_path)
    folded_body = {
        key: value
        for key, value in receipts["folded"].items()
        if key
        not in {
            "receipt_sha256",
            "artifact_publication_root",
            "artifact_receipt_sha256",
        }
    }
    publication_root = Path(receipts["folded"]["artifact_publication_root"])
    report_path = publication_root / child_artifact.BRIDGE_REPORT_FILENAME
    folded_body.update(
        {
            "materialization_report": str(report_path),
            "materialization_report_file_sha256": runner.stage._file_sha256(
                report_path
            ),
        }
    )
    receipts["folded"] = _receipt(folded_body)

    with pytest.raises(
        runner.FoldedBlankRecoveryABError,
        match="fields changed",
    ):
        runner.authenticate_materialization_receipts(
            contract,
            receipts,
            folded_artifact_hooks=_folded_artifact_hooks(),
        )


def test_preparer_writes_only_atomic_model_free_contract(
    tmp_path: Path,
    contract_inputs,
) -> None:
    output = tmp_path / "prepared"
    report = builder.prepare_recovery_ab(
        output_dir=output,
        **contract_inputs["paths"],
    )
    assert report["status"] == "ready-experimental-unaccepted"
    assert report["model_weights_loaded"] is False
    assert report["gpu_work_launched"] is False
    assert report["publication_authorized"] is False
    assert report["sealed_data_accessed"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "build_report.json",
        "preregistration.json",
    ]
    prereg = output / "preregistration.json"
    runner.authenticate_recovery_ab_preregistration(
        prereg,
        runner.stage._file_sha256(prereg),
    )


def test_scaffold_source_has_no_model_or_gpu_execution_path() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "from_pretrained(" not in source
    assert "torch.load(" not in source
    assert ".cuda(" not in source
    assert '"production_model_hook_implemented": True' in source
    assert "verify_compiled_child_cache" in source
    assert "_winner_payload" in source
    assert "pydantic" not in source.lower()
    assert "sqlalchemy" not in source.lower()
