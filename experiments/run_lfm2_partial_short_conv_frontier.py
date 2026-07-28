"""Run the sealed layer-0 partial ShortConv folding frontier.

The producer for this experiment writes only folded component tensors.  This
runner reconstructs a fresh FP32 parent shell for every candidate, installs
exactly one layer-0 ShortConv replacement, and decodes the existing whole-
response prompt bank.  Parent text is read from the authenticated cache; the
parent model is never generated again.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from experiments import run_lfm2_fused_body_diagnostic as fused  # noqa: E402
from experiments import run_lfm2_progressive_component_ablation as b038  # noqa: E402
from experiments.review_lfm2_semantic_frontier import (  # noqa: E402
    _generate_rows,
    _load_parent_envelope,
    load_review_prompts,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_model,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_fused_body_diagnostic import (  # noqa: E402
    Lfm2BottleneckShortConv,
)
from neural_forge.hf_lfm2_progressive_component_ablation import (  # noqa: E402
    _replace_module,
    _unique_parameter_elements,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


RUN_SCHEMA = "neural-forge-lfm2-partial-short-conv-frontier-run-v1"
REVIEW_SCHEMA = "neural-forge-lfm2-partial-short-conv-manual-review-v1"
PROGRESS_SCHEMA = "neural-forge-lfm2-partial-short-conv-progress-v1"
EXPECTED_RUNTIME_KINDS = (
    "projected-parent-output",
    "projected-parent-output-factor",
    "qdeim-structural",
    "qdeim-local-writer",
    "qdeim-local-writer-output-factor",
)


class Lfm2PartialShortConvFrontierRunError(RuntimeError):
    """A publication, reconstruction, or decode invariant failed closed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required runtime dependency
        raise RuntimeError("Partial ShortConv runtime requires PyTorch.") from exc
    return torch


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_plain(value: Any) -> Any:
    """Recursively snapshot sealed mappings into JSON-native containers."""

    if isinstance(value, Mapping):
        return {str(key): _json_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_plain(item) for item in value]
    return copy.deepcopy(value)


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2PartialShortConvFrontierRunError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return value


def _ordinary_file(path: Path, *, label: str) -> Path:
    value = path.expanduser().absolute()
    if not value.is_file() or value.is_symlink():
        raise Lfm2PartialShortConvFrontierRunError(
            f"{label} must be one ordinary file: {value}."
        )
    return value


def _ordinary_directory(path: Path, *, label: str) -> Path:
    value = path.expanduser().absolute()
    if not value.is_dir() or value.is_symlink():
        raise Lfm2PartialShortConvFrontierRunError(
            f"{label} must be one ordinary directory: {value}."
        )
    return value


def _read_plan(
    path: Path,
    *,
    expected_plan_sha256: str,
    expected_file_sha256: str,
) -> Mapping[str, Any]:
    source = _ordinary_file(path, label="partial ShortConv plan")
    if _file_sha256(source) != _digest(
        expected_file_sha256,
        label="expected plan file",
    ):
        raise Lfm2PartialShortConvFrontierRunError("Plan file hash changed.")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2PartialShortConvFrontierRunError("Cannot read plan JSON.") from exc
    if not isinstance(value, Mapping):
        raise Lfm2PartialShortConvFrontierRunError("Plan must be one mapping.")
    body = copy.deepcopy(dict(value))
    supplied = body.pop("plan_sha256", None)
    if (
        body.get("schema")
        != "neural-forge-lfm2-partial-short-conv-frontier-plan-v1"
        or supplied != canonical_sha256(body)
        or supplied
        != _digest(expected_plan_sha256, label="expected partial ShortConv plan")
    ):
        raise Lfm2PartialShortConvFrontierRunError("Plan seal is invalid.")
    return MappingProxyType(dict(value))


def _load_archive(
    path: Path,
    *,
    expected_file_sha256: str,
    records: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    source = _ordinary_file(path, label="partial ShortConv tensor archive")
    if _file_sha256(source) != _digest(
        expected_file_sha256,
        label="expected tensor archive",
    ):
        raise Lfm2PartialShortConvFrontierRunError("Tensor archive hash changed.")
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - required runtime dependency
        raise RuntimeError("Partial ShortConv runtime requires safetensors.") from exc
    tensors: dict[str, Any] = {}
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        if tuple(sorted(handle.keys())) != tuple(sorted(records)):
            raise Lfm2PartialShortConvFrontierRunError(
                "Tensor archive namespace differs from the sealed plan."
            )
        for name in sorted(records):
            tensor = handle.get_tensor(name).detach().cpu().contiguous().clone()
            record = records[name]
            if (
                str(tensor.dtype).removeprefix("torch.") != record.get("dtype")
                or [int(size) for size in tensor.shape] != record.get("shape")
                or lfm2_authenticated_tensor_sha256(tensor) != record.get("sha256")
            ):
                raise Lfm2PartialShortConvFrontierRunError(
                    f"Archived tensor {name!r} differs from its seal."
                )
            tensors[name] = tensor
    return MappingProxyType(tensors)


def _candidate_state(
    candidate: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> Mapping[str, Any]:
    state_keys = candidate.get("state_tensor_keys")
    if not isinstance(state_keys, Mapping) or not state_keys:
        raise Lfm2PartialShortConvFrontierRunError(
            "Candidate state tensor mapping is missing."
        )
    state = {
        str(name): tensors[str(archive_name)].detach().cpu().contiguous().clone()
        for name, archive_name in state_keys.items()
    }
    expected = candidate.get("state_tensor_sha256")
    if not isinstance(expected, Mapping) or set(expected) != set(state):
        raise Lfm2PartialShortConvFrontierRunError(
            "Candidate state tensor seals are incomplete."
        )
    for name, tensor in state.items():
        if lfm2_authenticated_tensor_sha256(tensor) != expected[name]:
            raise Lfm2PartialShortConvFrontierRunError(
                f"Candidate state tensor {name!r} changed."
            )
    return MappingProxyType(state)


def _validated_candidates(plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Authenticate the complete four-rung mechanism-isolation frontier."""

    raw = plan.get("candidates")
    if not isinstance(raw, list) or len(raw) != len(EXPECTED_RUNTIME_KINDS):
        raise Lfm2PartialShortConvFrontierRunError(
            "Plan must contain the complete five-candidate frontier."
        )
    if any(not isinstance(value, Mapping) for value in raw):
        raise Lfm2PartialShortConvFrontierRunError(
            "Candidate rows must be mappings."
        )
    candidates = tuple(raw)
    kinds = tuple(value.get("runtime_kind") for value in candidates)
    identifiers = tuple(value.get("candidate_id") for value in candidates)
    if kinds != EXPECTED_RUNTIME_KINDS:
        raise Lfm2PartialShortConvFrontierRunError(
            "Plan candidate order or mechanism set differs from the sealed frontier."
        )
    if (
        any(not isinstance(value, str) or not value for value in identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise Lfm2PartialShortConvFrontierRunError(
            "Plan candidate identities must be nonempty and unique."
        )
    return candidates


def _selected_candidates(
    candidates: Sequence[Mapping[str, Any]],
    requested_candidate_ids: Sequence[str] | None,
) -> tuple[Mapping[str, Any], ...]:
    """Select a sealed subset while retaining canonical plan order."""

    complete = tuple(candidates)
    if requested_candidate_ids is None:
        return complete
    requested = tuple(requested_candidate_ids)
    if (
        not requested
        or any(not isinstance(value, str) or not value for value in requested)
        or len(set(requested)) != len(requested)
    ):
        raise Lfm2PartialShortConvFrontierRunError(
            "Requested candidate identifiers must be nonempty and unique."
        )
    available = {str(candidate["candidate_id"]) for candidate in complete}
    missing = tuple(value for value in requested if value not in available)
    if missing:
        raise Lfm2PartialShortConvFrontierRunError(
            f"Requested candidates are absent from the sealed plan: {missing!r}."
        )
    selected_ids = set(requested)
    return tuple(
        candidate
        for candidate in complete
        if str(candidate["candidate_id"]) in selected_ids
    )


def _install_candidate(
    model: Any,
    *,
    candidate: Mapping[str, Any],
    tensors: Mapping[str, Any],
    device: Any,
) -> Mapping[str, Any]:
    torch = _torch()
    state = _candidate_state(candidate, tensors)
    kind = candidate.get("runtime_kind")
    layer_index = int(candidate.get("layer_index", -1))
    module_path = str(candidate.get("module_path", ""))
    if layer_index != 0 or module_path != "model.layers.0.conv":
        raise Lfm2PartialShortConvFrontierRunError(
            "This frontier is intentionally restricted to layer-0 ShortConv."
        )
    external = int(candidate.get("external_hidden_size", -1))
    internal = int(candidate.get("internal_channels", -1))
    kernel = int(candidate.get("kernel_size", -1))
    if external <= 0 or internal <= 0 or kernel <= 0:
        raise Lfm2PartialShortConvFrontierRunError(
            "Candidate geometry is invalid."
        )

    if kind == "projected-parent-output":
        if set(state) != {"out_proj.weight"} or internal != external:
            raise Lfm2PartialShortConvFrontierRunError(
                "Projected parent output candidate state is malformed."
            )
        module = model.get_submodule(module_path)
        with torch.inference_mode():
            module.out_proj.weight.copy_(
                state["out_proj.weight"].to(device=device, dtype=torch.float32)
            )
        installed_module = module
    elif kind == "projected-parent-output-factor":
        expected_names = {"out_decoder.weight", "out_encoder.weight"}
        rank = int(candidate.get("output_action_rank", -1))
        if (
            set(state) != expected_names
            or internal != external
            or rank <= 0
        ):
            raise Lfm2PartialShortConvFrontierRunError(
                "Factored projected-parent output candidate state is malformed."
            )
        module = model.get_submodule(module_path)
        module.out_proj = torch.nn.Sequential(
            torch.nn.Linear(
                internal,
                rank,
                bias=False,
                device=device,
                dtype=torch.float32,
            ),
            torch.nn.Linear(
                rank,
                external,
                bias=False,
                device=device,
                dtype=torch.float32,
            ),
        )
        incompatible = module.out_proj.load_state_dict(
            {
                "0.weight": state["out_encoder.weight"].to(device=device),
                "1.weight": state["out_decoder.weight"].to(device=device),
            },
            strict=True,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise Lfm2PartialShortConvFrontierRunError(
                "Factored projected-parent output failed strict reconstruction."
            )
        installed_module = module
    elif kind in {"qdeim-structural", "qdeim-local-writer"}:
        if set(state) != {"conv.weight", "in_proj.weight", "out_proj.weight"}:
            raise Lfm2PartialShortConvFrontierRunError(
                "QDEIM candidate state is malformed."
            )
        installed_module = Lfm2BottleneckShortConv(
            parent_hidden_size=external,
            compact_hidden_size=internal,
            kernel_size=kernel,
            layer_index=layer_index,
            device=device,
        )
        incompatible = installed_module.load_state_dict(
            {name: tensor.to(device=device) for name, tensor in state.items()},
            strict=True,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise Lfm2PartialShortConvFrontierRunError(
                "QDEIM state failed strict reconstruction."
            )
        _replace_module(model, module_path, installed_module)
    elif kind == "qdeim-local-writer-output-factor":
        expected_names = {
            "conv.weight",
            "in_proj.weight",
            "out_decoder.weight",
            "out_encoder.weight",
        }
        rank = int(candidate.get("output_action_rank", -1))
        if set(state) != expected_names or rank <= 0:
            raise Lfm2PartialShortConvFrontierRunError(
                "Factored QDEIM candidate state is malformed."
            )
        installed_module = Lfm2BottleneckShortConv(
            parent_hidden_size=external,
            compact_hidden_size=internal,
            kernel_size=kernel,
            layer_index=layer_index,
            device=device,
        )
        installed_module.out_proj = torch.nn.Sequential(
            torch.nn.Linear(
                internal,
                rank,
                bias=False,
                device=device,
                dtype=torch.float32,
            ),
            torch.nn.Linear(
                rank,
                external,
                bias=False,
                device=device,
                dtype=torch.float32,
            ),
        )
        incompatible = installed_module.load_state_dict(
            {
                "conv.weight": state["conv.weight"].to(device=device),
                "in_proj.weight": state["in_proj.weight"].to(device=device),
                "out_proj.0.weight": state["out_encoder.weight"].to(device=device),
                "out_proj.1.weight": state["out_decoder.weight"].to(device=device),
            },
            strict=True,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise Lfm2PartialShortConvFrontierRunError(
                "Factored QDEIM state failed strict reconstruction."
            )
        _replace_module(model, module_path, installed_module)
    else:
        raise Lfm2PartialShortConvFrontierRunError(
            f"Unknown partial ShortConv runtime kind {kind!r}."
        )

    installed_module.requires_grad_(False)
    installed_module.eval()
    state_after = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in installed_module.state_dict().items()
    }
    installed_elements = sum(int(value.numel()) for value in state_after.values())
    expected_elements = int(candidate.get("candidate_parameter_elements", -1))
    if installed_elements != expected_elements:
        raise Lfm2PartialShortConvFrontierRunError(
            "Installed component parameter count differs from the plan."
        )
    body = {
        "candidate_id": candidate["candidate_id"],
        "runtime_kind": kind,
        "module_path": module_path,
        "installed_parameter_elements": installed_elements,
        "expected_parameter_elements": expected_elements,
        "state_tensor_sha256": {
            name: lfm2_authenticated_tensor_sha256(value)
            for name, value in sorted(state_after.items())
        },
        "frozen": all(not parameter.requires_grad for parameter in installed_module.parameters()),
        "eval_mode": installed_module.training is False,
        "training_steps": 0,
        "optimizer_steps": 0,
        "parent_forward_calls": 0,
    }
    return MappingProxyType({**body, "installation_sha256": canonical_sha256(body)})


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise Lfm2PartialShortConvFrontierRunError(
            f"Refusing to replace or misplace evidence file {path}."
        )
    payload = json.dumps(
        _json_plain(value),
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _review_document(
    *,
    candidate: Mapping[str, Any],
    installation: Mapping[str, Any] | None,
    responses: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    executed = installation is not None
    body = {
        "schema": REVIEW_SCHEMA,
        "status": "awaiting-manual-intelligence-review" if executed else "reference",
        "candidate_id": candidate["candidate_id"],
        "runtime_kind": candidate["runtime_kind"],
        "candidate": copy.deepcopy(dict(candidate)),
        "installation": None if installation is None else dict(installation),
        "responses": copy.deepcopy(list(responses)),
        "response_count": len(responses),
        "manual_review": (
            {
                "coherent_response_count": None,
                "useful_or_correct_response_count": None,
                "intelligence_retention_relative_to_parent": None,
                "notes": None,
            }
            if executed
            else {"reference_integrity_notes": None}
        ),
        "exact_token_match_required": False,
        "numeric_metrics_have_decision_authority": False,
        "automatic_acceptance_or_rejection_performed": False,
        "fresh_fp32_candidate_decode": executed,
        "cached_parent_reference_only": not executed,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    return MappingProxyType({**body, "review_sha256": canonical_sha256(body)})


def run(
    *,
    run_directory: Path,
    parent_checkpoint: Path,
    expected_parent_checkpoint_inventory_sha256: str,
    plan_file: Path,
    expected_plan_sha256: str,
    expected_plan_file_sha256: str,
    tensor_archive: Path,
    expected_tensor_archive_file_sha256: str,
    prompts_file: Path,
    expected_prompts_file_sha256: str,
    parent_envelope_file: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    device: str,
    candidate_ids: Sequence[str] | None = None,
    maximum_prompt_tokens: int = 1024,
    maximum_new_tokens: int = 256,
) -> Mapping[str, Any]:
    torch = _torch()
    target = run_directory.expanduser().absolute()
    if target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise Lfm2PartialShortConvFrontierRunError(
            "Run directory must be one new child of an existing directory."
        )
    parent_path = _ordinary_directory(parent_checkpoint, label="parent checkpoint")
    prompt_path = _ordinary_file(prompts_file, label="review prompts")
    envelope_path = _ordinary_file(parent_envelope_file, label="parent envelope")
    if _file_sha256(prompt_path) != _digest(
        expected_prompts_file_sha256,
        label="expected prompts file",
    ):
        raise Lfm2PartialShortConvFrontierRunError("Prompt bank hash changed.")
    plan = _read_plan(
        plan_file,
        expected_plan_sha256=expected_plan_sha256,
        expected_file_sha256=expected_plan_file_sha256,
    )
    tensors = _load_archive(
        tensor_archive,
        expected_file_sha256=expected_tensor_archive_file_sha256,
        records=plan["tensor_records"],
    )
    inventory_sha256 = canonical_sha256(b038._checkpoint_inventory(parent_path))
    if (
        inventory_sha256
        != _digest(
            expected_parent_checkpoint_inventory_sha256,
            label="expected parent checkpoint inventory",
        )
        or inventory_sha256 != plan.get("parent_checkpoint_inventory_sha256")
    ):
        raise Lfm2PartialShortConvFrontierRunError(
            "Parent checkpoint inventory differs from the producer."
        )

    prompt_rows = [dict(value) for value in load_review_prompts(prompt_path)]
    prompt_set_sha256 = canonical_sha256(prompt_rows)
    parent_cache = _load_parent_envelope(
        envelope_path,
        expected_envelope_sha256=_digest(
            expected_parent_envelope_sha256,
            label="expected parent envelope",
        ),
        expected_file_sha256=_digest(
            expected_parent_envelope_file_sha256,
            label="expected parent envelope file",
        ),
        expected_prompt_set_sha256=prompt_set_sha256,
        expected_prompt_ids=[str(row["id"]) for row in prompt_rows],
    )
    parent_cache_authentication = fused._authenticate_parent_cache_binding(
        parent_cache,
        plan=plan,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
    )
    parent_rows = {str(row["id"]): row for row in parent_cache["responses"]}
    tokenizer = load_lfm2_direct_local_tokenizer(parent_path)
    frontier_candidates = _validated_candidates(plan)
    candidates = _selected_candidates(frontier_candidates, candidate_ids)
    execution_scope = (
        "complete-frontier"
        if len(candidates) == len(frontier_candidates)
        else "authenticated-selected-subset"
    )

    target.mkdir()
    reviews_root = target / "manual-review"
    reviews_root.mkdir()
    completed: list[dict[str, Any]] = []
    resolved_device = torch.device(device)
    parent_parameter_elements = int(plan["parent_parameter_elements"])
    for candidate in candidates:
        model = load_lfm2_direct_local_model(
            parent_path,
            device=resolved_device,
            dtype=torch.float32,
        )
        if (
            fingerprint_lfm2_direct_local_state(model)
            != plan["parent_fp32_state_sha256"]
            or _unique_parameter_elements(model) != parent_parameter_elements
        ):
            raise Lfm2PartialShortConvFrontierRunError(
                "Fresh parent shell differs before candidate installation."
            )
        installation = _install_candidate(
            model,
            candidate=candidate,
            tensors=tensors,
            device=resolved_device,
        )
        expected_model_elements = (
            parent_parameter_elements
            - int(candidate["source_component_parameter_elements"])
            + int(candidate["candidate_parameter_elements"])
        )
        if _unique_parameter_elements(model) != expected_model_elements:
            raise Lfm2PartialShortConvFrontierRunError(
                "Installed model parameter count differs from physical accounting."
            )
        responses = _generate_rows(
            model,
            tokenizer,
            prompt_rows,
            maximum_prompt_tokens=maximum_prompt_tokens,
            maximum_new_tokens=maximum_new_tokens,
        )
        fused._attach_parent(responses, parent_rows=parent_rows)
        review = _review_document(
            candidate=candidate,
            installation=installation,
            responses=responses,
        )
        review_path = reviews_root / f"{candidate['candidate_id']}.json"
        _write_json(review_path, review)
        completed.append(
            {
                "candidate_id": candidate["candidate_id"],
                "runtime_kind": candidate["runtime_kind"],
                "review_file": str(review_path),
                "review_file_sha256": _file_sha256(review_path),
                "review_sha256": review["review_sha256"],
                "installation_sha256": installation["installation_sha256"],
            }
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress_body = {
            "schema": PROGRESS_SCHEMA,
            "status": "running",
            "plan_sha256": plan["plan_sha256"],
            "candidate_ids": [row["candidate_id"] for row in candidates],
            "completed": copy.deepcopy(completed),
            "automatic_acceptance_or_rejection_performed": False,
        }
        progress = {
            **progress_body,
            "progress_sha256": canonical_sha256(progress_body),
        }
        progress_path = target / "progress.json"
        if progress_path.exists():
            progress_path.unlink()
        _write_json(progress_path, progress)

    reference_candidate = {
        "candidate_id": "cached-parent-reference",
        "runtime_kind": "cached-parent-reference",
    }
    reference_rows = fused._cached_parent_reference_rows(
        prompt_rows,
        parent_rows=parent_rows,
    )
    reference = _review_document(
        candidate=reference_candidate,
        installation=None,
        responses=reference_rows,
    )
    reference_path = reviews_root / "cached-parent-reference.json"
    _write_json(reference_path, reference)

    body = {
        "schema": RUN_SCHEMA,
        "status": (
            "partial-short-conv-frontier-complete-awaiting-manual-review"
            if execution_scope == "complete-frontier"
            else "partial-short-conv-selected-subset-complete-awaiting-manual-review"
        ),
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": _file_sha256(_ordinary_file(plan_file, label="plan")),
        "tensor_archive_file_sha256": _file_sha256(
            _ordinary_file(tensor_archive, label="tensor archive")
        ),
        "parent_checkpoint_inventory_sha256": inventory_sha256,
        "parent_cache_authentication": dict(parent_cache_authentication),
        "execution_scope": execution_scope,
        "complete_plan_candidate_count": len(frontier_candidates),
        "selected_candidate_ids": [row["candidate_id"] for row in candidates],
        "candidate_count": len(candidates),
        "completed": completed,
        "cached_parent_reference_review": str(reference_path),
        "cached_parent_reference_review_file_sha256": _file_sha256(reference_path),
        "manual_whole_response_review_required": True,
        "numeric_metrics_have_decision_authority": False,
        "automatic_acceptance_or_rejection_performed": False,
        "operation_audit": {
            "parent_model_forward_calls": 0,
            "parent_model_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "candidate_generate_calls": len(candidates) * len(prompt_rows),
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
    }
    result = MappingProxyType({**body, "run_sha256": canonical_sha256(body)})
    _write_json(target / "run-receipt.json", result)
    progress_path = target / "progress.json"
    progress_path.unlink()
    final_progress_body = {
        "schema": PROGRESS_SCHEMA,
        "status": "complete",
        "plan_sha256": plan["plan_sha256"],
        "candidate_ids": [row["candidate_id"] for row in candidates],
        "completed": copy.deepcopy(completed),
        "automatic_acceptance_or_rejection_performed": False,
    }
    _write_json(
        progress_path,
        {
            **final_progress_body,
            "progress_sha256": canonical_sha256(final_progress_body),
        },
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-checkpoint-inventory-sha256", required=True)
    parser.add_argument("--plan-file", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-plan-file-sha256", required=True)
    parser.add_argument("--tensor-archive", type=Path, required=True)
    parser.add_argument("--expected-tensor-archive-file-sha256", required=True)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--parent-envelope-file", type=Path, required=True)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-parent-envelope-file-sha256", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--candidate-id",
        dest="candidate_ids",
        action="append",
        help="Run only this sealed plan candidate; repeat to select more than one.",
    )
    parser.add_argument("--maximum-prompt-tokens", type=int, default=1024)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run(
        run_directory=arguments.run_directory,
        parent_checkpoint=arguments.parent_checkpoint,
        expected_parent_checkpoint_inventory_sha256=(
            arguments.expected_parent_checkpoint_inventory_sha256
        ),
        plan_file=arguments.plan_file,
        expected_plan_sha256=arguments.expected_plan_sha256,
        expected_plan_file_sha256=arguments.expected_plan_file_sha256,
        tensor_archive=arguments.tensor_archive,
        expected_tensor_archive_file_sha256=(
            arguments.expected_tensor_archive_file_sha256
        ),
        prompts_file=arguments.prompts_file,
        expected_prompts_file_sha256=arguments.expected_prompts_file_sha256,
        parent_envelope_file=arguments.parent_envelope_file,
        expected_parent_envelope_sha256=arguments.expected_parent_envelope_sha256,
        expected_parent_envelope_file_sha256=(
            arguments.expected_parent_envelope_file_sha256
        ),
        device=arguments.device,
        candidate_ids=arguments.candidate_ids,
        maximum_prompt_tokens=arguments.maximum_prompt_tokens,
        maximum_new_tokens=arguments.maximum_new_tokens,
    )
    print(json.dumps(dict(result), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
