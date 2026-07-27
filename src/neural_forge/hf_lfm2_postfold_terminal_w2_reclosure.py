"""Authenticated terminal-W2 handoff for post-attention body reclosure.

The exact-4x body checkpoint carries the selected terminal MLP down projection
already installed.  Whole-stack output-writer reclosure, however, must solve
with that selected tensor as the terminal prior while the live final W2 is
restored to the original deferred seed.  This module authenticates that
rebase and performs the one final publication after source-local global
reclosure.  It executes no model, optimizer, autograd, pruning, quantization,
or parameter-topology operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import fingerprint_model_state_entries
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    Lfm2CpcfTerminalActionDownWeightPrior,
    lfm2_cpcf_terminal_action_tensor_sha256,
)
from neural_forge.hf_lfm2_global_residual_closure import (
    fingerprint_lfm2_global_residual_tensor,
)
from neural_forge.hf_lfm2_semantic_global_residual_integration import (
    LFM2_SEMANTIC_SOURCE_LOCAL_OUTPUT_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA,
    verify_lfm2_semantic_global_residual_report,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import (
    LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
)
from neural_forge.hf_lfm2_semantic_mlp_refold_adapter import (
    LFM2_SEMANTIC_MLP_INSTALLATION_SCHEMA,
)


LFM2_POSTFOLD_TERMINAL_W2_REBASE_SCHEMA = (
    "neural-forge-lfm2-postfold-terminal-w2-rebase-v1"
)
LFM2_POST_RECLOSURE_TERMINAL_W2_INSTALL_SCHEMA = (
    "neural-forge-lfm2-post-reclosure-terminal-w2-install-v1"
)
LFM2_DEFERRED_FINAL_W2_INSTALLATION_SCHEMA = (
    "neural-forge-lfm2-deferred-final-w2-installation-v1"
)


class Lfm2PostfoldTerminalW2ReclosureError(RuntimeError):
    """Terminal W2 ancestry, live state, or publication is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Terminal W2 reclosure requires PyTorch.") from exc
    return torch


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_deep_thaw(item) for item in value]
    return value


def _verified_seal(
    value: Any,
    *,
    schema: str,
    seal_field: str,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2PostfoldTerminalW2ReclosureError(
            f"{label} must be one sealed mapping."
        )
    body = _deep_thaw(value)
    supplied = body.pop(seal_field, None)
    if body.get("schema") != schema or supplied != canonical_sha256(body):
        raise Lfm2PostfoldTerminalW2ReclosureError(
            f"{label} is stale, malformed, or unauthenticated."
        )
    return MappingProxyType({**body, seal_field: supplied})


def _parameter_count(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _final_parameter(
    model: Any,
    *,
    parameter_name: str,
) -> Any:
    torch = _torch()
    named = dict(model.named_parameters())
    parameter = named.get(parameter_name)
    aliases = [name for name, value in named.items() if value is parameter]
    if (
        not isinstance(parameter_name, str)
        or not isinstance(parameter, torch.nn.Parameter)
        or aliases != [parameter_name]
        or parameter.layout != torch.strided
        or parameter.device.type == "meta"
        or parameter.ndim != 2
        or parameter.dtype not in {torch.bfloat16, torch.float32}
    ):
        raise Lfm2PostfoldTerminalW2ReclosureError(
            "The receipted final W2 is not one independently addressable dense "
            "BF16/FP32 Parameter."
        )
    return parameter


def _runtime_snapshot(model: Any) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        result[name] = {
            "shape": tuple(int(size) for size in parameter.shape),
            "dtype": str(parameter.dtype),
            "device": str(parameter.device),
            "parameter_id": id(parameter),
            "storage_data_ptr": int(parameter.data_ptr()),
            "requires_grad": bool(parameter.requires_grad),
            "gradient_id": None if gradient is None else id(gradient),
            "gradient_storage_data_ptr": (
                None if gradient is None else int(gradient.data_ptr())
            ),
            "gradient": (
                None
                if gradient is None
                else gradient.detach().to(device="cpu").contiguous().clone()
            ),
        }
    return MappingProxyType(result)


def _runtime_proof(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Mapping[str, Any]:
    torch = _torch()
    names = tuple(sorted(before))
    same_names = names == tuple(sorted(after))
    topology = bool(
        same_names
        and all(
            tuple(before[name][field] for field in ("shape", "dtype", "device"))
            == tuple(after[name][field] for field in ("shape", "dtype", "device"))
            for name in names
        )
    )
    identities = bool(
        topology
        and all(
            before[name]["parameter_id"] == after[name]["parameter_id"]
            for name in names
        )
    )
    storage = bool(
        topology
        and all(
            before[name]["storage_data_ptr"] == after[name]["storage_data_ptr"]
            for name in names
        )
    )

    def same_gradient(name: str) -> bool:
        left = before[name]
        right = after[name]
        if (
            left["requires_grad"] != right["requires_grad"]
            or left["gradient_id"] != right["gradient_id"]
            or left["gradient_storage_data_ptr"]
            != right["gradient_storage_data_ptr"]
        ):
            return False
        if left["gradient"] is None or right["gradient"] is None:
            return left["gradient"] is None and right["gradient"] is None
        return bool(torch.equal(left["gradient"], right["gradient"]))

    gradients = bool(topology and all(same_gradient(name) for name in names))
    before_topology = {
        name: {
            field: before[name][field] for field in ("shape", "dtype", "device")
        }
        for name in names
    }
    after_topology = {
        name: {
            field: after[name][field] for field in ("shape", "dtype", "device")
        }
        for name in sorted(after)
    }
    return MappingProxyType(
        {
            "parameter_topology_sha256_before": canonical_sha256(before_topology),
            "parameter_topology_sha256_after": canonical_sha256(after_topology),
            "parameter_topology_preserved": topology,
            "parameter_identities_preserved": identities,
            "parameter_storage_preserved": storage,
            "gradient_state_preserved": gradients,
        }
    )


def _changed_state_keys(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> tuple[str, ...]:
    if set(before) != set(after):
        raise Lfm2PostfoldTerminalW2ReclosureError(
            "Terminal W2 publication changed state topology."
        )
    return tuple(name for name in sorted(before) if before[name] != after[name])


def _terminal_frontier(stack: Mapping[str, Any]) -> Mapping[str, Any]:
    layer_index = stack.get("terminal_layer_index")
    layers = stack.get("layer_installations")
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or not isinstance(layers, list)
    ):
        raise Lfm2PostfoldTerminalW2ReclosureError(
            "Semantic stack has no authenticated terminal layer."
        )
    matches = [
        value
        for value in layers
        if isinstance(value, Mapping) and value.get("layer_index") == layer_index
    ]
    if len(matches) != 1:
        raise Lfm2PostfoldTerminalW2ReclosureError(
            "Semantic stack terminal layer is ambiguous."
        )
    return _verified_seal(
        matches[0].get("mlp_complete_candidate_frontier_receipt"),
        schema=LFM2_SEMANTIC_MLP_INSTALLATION_SCHEMA,
        seal_field="result_sha256",
        label="terminal MLP candidate-frontier receipt",
    )


@dataclass(frozen=True, slots=True)
class Lfm2PostfoldTerminalW2Rebase:
    """Selected installed W2 recovered as a prior; live W2 restored to seed."""

    terminal_prior: Lfm2CpcfTerminalActionDownWeightPrior
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    def verify(self) -> bool:
        try:
            verified = _verified_seal(
                self.receipt,
                schema=LFM2_POSTFOLD_TERMINAL_W2_REBASE_SCHEMA,
                seal_field="result_sha256",
                label="terminal W2 rebase receipt",
            )
        except Lfm2PostfoldTerminalW2ReclosureError:
            return False
        return bool(
            isinstance(
                self.terminal_prior,
                Lfm2CpcfTerminalActionDownWeightPrior,
            )
            and self.terminal_prior.verify_authentication()
            and verified.get("terminal_prior_authentication_sha256")
            == self.terminal_prior.authentication_sha256
            and verified.get("terminal_prior_tensor_sha256")
            == lfm2_cpcf_terminal_action_tensor_sha256(
                self.terminal_prior.down_weight
            )
            and verified.get("restored_live_w2_sha256")
            == self.terminal_prior.live_deferred_w2_sha256
        )


def rebase_lfm2_installed_terminal_w2_for_output_reclosure(
    target_model: Any,
    *,
    semantic_layer_stack_receipt: Mapping[str, Any],
    deferred_final_w2_installation_receipt: Mapping[str, Any],
    original_deferred_w2: Any,
) -> Lfm2PostfoldTerminalW2Rebase:
    """Recover the selected terminal W2 prior and restore its deferred seed."""

    torch = _torch()
    stack = _verified_seal(
        semantic_layer_stack_receipt,
        schema=LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
        seal_field="result_sha256",
        label="semantic layer-stack receipt",
    )
    installation = _verified_seal(
        deferred_final_w2_installation_receipt,
        schema=LFM2_DEFERRED_FINAL_W2_INSTALLATION_SCHEMA,
        seal_field="result_sha256",
        label="deferred final-W2 installation receipt",
    )
    frontier = _terminal_frontier(stack)
    final_name = installation.get("final_w2_parameter_name")
    final_w2 = _final_parameter(target_model, parameter_name=final_name)
    installed_sha = lfm2_cpcf_terminal_action_tensor_sha256(final_w2)
    seed_sha = lfm2_cpcf_terminal_action_tensor_sha256(original_deferred_w2)
    state_before = fingerprint_lfm2_parent(target_model).identity_sha256
    if (
        not isinstance(original_deferred_w2, torch.Tensor)
        or original_deferred_w2.layout != torch.strided
        or original_deferred_w2.device.type == "meta"
        or original_deferred_w2.ndim != 2
        or tuple(original_deferred_w2.shape) != tuple(final_w2.shape)
        or original_deferred_w2.dtype != final_w2.dtype
        or installation.get("semantic_layer_stack_result_sha256")
        != stack["result_sha256"]
        or installation.get("model_state_after_sha256") != state_before
        or installation.get("changed_state_keys") != [final_name]
        or installation.get("effectful_installation") is not True
        or installation.get("live_w2_sha256_after") != installed_sha
        or installation.get("prior_tensor_sha256") != installed_sha
        or installation.get("live_w2_sha256_before") != seed_sha
        or installation.get("model_state_before_sha256")
        != stack.get("model_state_after_sha256")
        or stack.get("deferred_final_w2_sha256") != seed_sha
        or stack.get("final_layer_mlp_w2_deferred_to_fused_terminal") is not True
        or stack.get("final_layer_mlp_w2_install_count") != 0
        or stack.get("final_layer_mlp_down_weight_prior_install_count") != 0
        or stack.get("final_layer_mlp_down_weight_prior_installed_before_terminal")
        is not False
        or stack.get("final_layer_mlp_down_weight_prior_tensor_sha256")
        != installed_sha
        or stack.get("final_layer_mlp_down_weight_prior_live_w2_sha256") != seed_sha
        or frontier.get("deferred_selected_w2_prior_terminal_action_sha256")
        != installed_sha
        or frontier.get("deferred_selected_w2_prior_selected_role")
        != "full-w1-w3-w2"
        or frontier.get("deferred_selected_w2_prior_installed_before_terminal")
        is not False
    ):
        raise Lfm2PostfoldTerminalW2ReclosureError(
            "Installed terminal W2 does not authenticate to the deferred semantic "
            "stack and exact live checkpoint."
        )
    prior = Lfm2CpcfTerminalActionDownWeightPrior(
        down_weight=final_w2,
        body_recipe_id=str(frontier.get("body_recipe_id")),
        body_recipe_sha256=str(frontier.get("body_recipe_sha256")),
        mlp_lane=str(frontier.get("recipe_component_lane")),
        selected_role=str(frontier.get("selected_role")),
        selected_candidate_sha256=str(frontier.get("selected_candidate_sha256")),
        preparation_sha256=str(frontier.get("preparation_sha256")),
        mlp_installation_core_sha256=str(
            frontier.get("deferred_selected_w2_prior_mlp_installation_core_sha256")
        ),
        live_deferred_w2_sha256=seed_sha,
    )
    if (
        prior.authentication_sha256
        != installation.get("prior_authentication_sha256")
        or prior.authentication_sha256
        != stack.get("final_layer_mlp_down_weight_prior_authentication_sha256")
        or prior.authentication_sha256
        != frontier.get("deferred_selected_w2_prior_authentication_sha256")
        or frontier.get("deferred_selected_w2_prior_candidate_sha256")
        != prior.selected_candidate_sha256
    ):
        raise Lfm2PostfoldTerminalW2ReclosureError(
            "Terminal prior metadata does not reproduce its authenticated prior."
        )
    entries_before = dict(fingerprint_model_state_entries(target_model))
    runtime_before = _runtime_snapshot(target_model)
    parameter_count = _parameter_count(target_model)
    installed_snapshot = final_w2.detach().to(device="cpu").contiguous().clone()
    try:
        with torch.no_grad():
            final_w2.copy_(
                original_deferred_w2.to(
                    device=final_w2.device,
                    dtype=final_w2.dtype,
                )
            )
        entries_after = dict(fingerprint_model_state_entries(target_model))
        runtime_after = _runtime_snapshot(target_model)
        runtime = _runtime_proof(runtime_before, runtime_after)
        changed = _changed_state_keys(entries_before, entries_after)
        restored_sha = lfm2_cpcf_terminal_action_tensor_sha256(final_w2)
        state_after = fingerprint_lfm2_parent(target_model).identity_sha256
        if (
            changed != (final_name,)
            or restored_sha != seed_sha
            or state_after != stack.get("model_state_after_sha256")
            or _parameter_count(target_model) != parameter_count
            or not all(
                runtime[name] is True
                for name in (
                    "parameter_topology_preserved",
                    "parameter_identities_preserved",
                    "parameter_storage_preserved",
                    "gradient_state_preserved",
                )
            )
        ):
            raise Lfm2PostfoldTerminalW2ReclosureError(
                "Terminal W2 rebase did not restore the exact deferred body."
            )
    except BaseException:
        with torch.no_grad():
            final_w2.copy_(
                installed_snapshot.to(device=final_w2.device, dtype=final_w2.dtype)
            )
        if fingerprint_lfm2_parent(target_model).identity_sha256 != state_before:
            raise Lfm2PostfoldTerminalW2ReclosureError(
                "Terminal W2 rebase failed and exact rollback was not proven."
            )
        raise
    body = {
        "schema": LFM2_POSTFOLD_TERMINAL_W2_REBASE_SCHEMA,
        "status": "installed-terminal-prior-recovered-and-live-seed-restored",
        "semantic_layer_stack_result_sha256": stack["result_sha256"],
        "source_deferred_w2_installation_sha256": installation["result_sha256"],
        "terminal_mlp_frontier_sha256": frontier["result_sha256"],
        "terminal_prior_authentication_sha256": prior.authentication_sha256,
        "terminal_prior_tensor_sha256": installed_sha,
        "final_w2_parameter_name": final_name,
        "installed_live_w2_sha256_before": installed_sha,
        "restored_live_w2_sha256": seed_sha,
        "model_state_before_sha256": state_before,
        "model_state_after_sha256": state_after,
        "changed_state_keys": list(changed),
        "parameter_element_count": parameter_count,
        **dict(runtime),
        "terminal_prior_installed_during_rebase": False,
        "live_final_w2_restored_to_original_deferred_seed": True,
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    receipt = MappingProxyType(
        {**body, "result_sha256": canonical_sha256(body)}
    )
    result = Lfm2PostfoldTerminalW2Rebase(
        terminal_prior=prior,
        receipt=receipt,
    )
    if not result.verify():  # pragma: no cover - construction seam
        raise Lfm2PostfoldTerminalW2ReclosureError(
            "Terminal W2 rebase result failed self-authentication."
        )
    return result


def install_lfm2_post_reclosure_terminal_w2(
    target_model: Any,
    *,
    rebase: Lfm2PostfoldTerminalW2Rebase,
    corrected_terminal_prior: Lfm2CpcfTerminalActionDownWeightPrior,
    source_local_global_reclosure_report: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Publish exactly the closure-corrected terminal W2 once."""

    torch = _torch()
    if not isinstance(rebase, Lfm2PostfoldTerminalW2Rebase) or not rebase.verify():
        raise Lfm2PostfoldTerminalW2ReclosureError(
            "Terminal W2 rebase is unauthenticated."
        )
    if (
        not isinstance(
            corrected_terminal_prior,
            Lfm2CpcfTerminalActionDownWeightPrior,
        )
        or not corrected_terminal_prior.verify_authentication()
    ):
        raise Lfm2PostfoldTerminalW2ReclosureError(
            "Corrected terminal W2 prior is unauthenticated."
        )
    report = _verified_seal(
        source_local_global_reclosure_report,
        schema=LFM2_SEMANTIC_SOURCE_LOCAL_OUTPUT_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA,
        seal_field="report_sha256",
        label="source-local global reclosure report",
    )
    if not verify_lfm2_semantic_global_residual_report(report):
        raise Lfm2PostfoldTerminalW2ReclosureError(
            "Source-local global reclosure report failed authentication."
        )
    final_name = rebase.receipt["final_w2_parameter_name"]
    final_w2 = _final_parameter(target_model, parameter_name=final_name)
    live_sha = lfm2_cpcf_terminal_action_tensor_sha256(final_w2)
    corrected_sha = lfm2_cpcf_terminal_action_tensor_sha256(
        corrected_terminal_prior.down_weight
    )
    if (
        corrected_terminal_prior.live_deferred_w2_sha256
        != rebase.receipt["restored_live_w2_sha256"]
        or live_sha != corrected_terminal_prior.live_deferred_w2_sha256
        or report.get("source_local_boundary_mode") is not True
        or report.get("absolute_endpoint_boundary_mode") is not False
        or report.get("final_w2_unchanged") is not True
        or report.get("final_w2_installation_count") != 0
        or report.get("final_w2_live_sha256_after") != live_sha
        or report.get("corrected_final_w2_prior_sha256")
        != corrected_terminal_prior.authentication_sha256
        or report.get("final_layer_mlp_down_weight_prior_authentication_sha256")
        != corrected_terminal_prior.authentication_sha256
        or report.get("final_layer_mlp_down_weight_prior_tensor_sha256")
        != corrected_sha
        or report.get("final_layer_mlp_down_weight_prior_live_w2_sha256")
        != live_sha
        or tuple(corrected_terminal_prior.down_weight.shape)
        != tuple(final_w2.shape)
        or corrected_terminal_prior.down_weight.dtype != final_w2.dtype
    ):
        raise Lfm2PostfoldTerminalW2ReclosureError(
            "Corrected terminal prior is stale for this source-local reclosure."
        )
    expected_writer_hashes = report.get("dense_live_writer_sha256_after")
    installed_site_ids = report.get("installed_writer_site_ids")
    named_parameters = dict(target_model.named_parameters())
    if (
        not isinstance(expected_writer_hashes, Mapping)
        or not isinstance(installed_site_ids, list)
        or set(expected_writer_hashes) != set(installed_site_ids)
        or any(
            site_id not in named_parameters
            or fingerprint_lfm2_global_residual_tensor(named_parameters[site_id])
            != expected_writer_hashes[site_id]
            for site_id in installed_site_ids
        )
    ):
        raise Lfm2PostfoldTerminalW2ReclosureError(
            "Live nonfinal writers differ from the authenticated reclosure result."
        )
    entries_before = dict(fingerprint_model_state_entries(target_model))
    runtime_before = _runtime_snapshot(target_model)
    state_before = fingerprint_lfm2_parent(target_model).identity_sha256
    parameter_count = _parameter_count(target_model)
    live_snapshot = final_w2.detach().to(device="cpu").contiguous().clone()
    try:
        with torch.no_grad():
            final_w2.copy_(
                corrected_terminal_prior.down_weight.to(
                    device=final_w2.device,
                    dtype=final_w2.dtype,
                )
            )
        entries_after = dict(fingerprint_model_state_entries(target_model))
        runtime_after = _runtime_snapshot(target_model)
        runtime = _runtime_proof(runtime_before, runtime_after)
        changed = _changed_state_keys(entries_before, entries_after)
        expected_changed = () if live_sha == corrected_sha else (final_name,)
        installed_sha = lfm2_cpcf_terminal_action_tensor_sha256(final_w2)
        state_after = fingerprint_lfm2_parent(target_model).identity_sha256
        if (
            changed != expected_changed
            or installed_sha != corrected_sha
            or _parameter_count(target_model) != parameter_count
            or not all(
                runtime[name] is True
                for name in (
                    "parameter_topology_preserved",
                    "parameter_identities_preserved",
                    "parameter_storage_preserved",
                    "gradient_state_preserved",
                )
            )
        ):
            raise Lfm2PostfoldTerminalW2ReclosureError(
                "Final terminal W2 publication changed unrelated state."
            )
    except BaseException:
        with torch.no_grad():
            final_w2.copy_(
                live_snapshot.to(device=final_w2.device, dtype=final_w2.dtype)
            )
        if fingerprint_lfm2_parent(target_model).identity_sha256 != state_before:
            raise Lfm2PostfoldTerminalW2ReclosureError(
                "Final W2 publication failed and exact rollback was not proven."
            )
        raise
    body = {
        "schema": LFM2_POST_RECLOSURE_TERMINAL_W2_INSTALL_SCHEMA,
        "status": "source-local-global-reclosure-terminal-w2-installed",
        "terminal_rebase_sha256": rebase.receipt["result_sha256"],
        "source_local_global_reclosure_sha256": report["report_sha256"],
        "corrected_terminal_prior_authentication_sha256": (
            corrected_terminal_prior.authentication_sha256
        ),
        "corrected_terminal_prior_tensor_sha256": corrected_sha,
        "final_w2_parameter_name": final_name,
        "live_w2_sha256_before": live_sha,
        "live_w2_sha256_after": installed_sha,
        "effectful_installation": live_sha != installed_sha,
        "installation_count": 1,
        "changed_state_keys": list(changed),
        "model_state_before_sha256": state_before,
        "model_state_after_sha256": state_after,
        "parameter_element_count": parameter_count,
        **dict(runtime),
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return MappingProxyType(
        {**body, "result_sha256": canonical_sha256(body)}
    )


__all__ = [
    "LFM2_DEFERRED_FINAL_W2_INSTALLATION_SCHEMA",
    "LFM2_POSTFOLD_TERMINAL_W2_REBASE_SCHEMA",
    "LFM2_POST_RECLOSURE_TERMINAL_W2_INSTALL_SCHEMA",
    "Lfm2PostfoldTerminalW2Rebase",
    "Lfm2PostfoldTerminalW2ReclosureError",
    "install_lfm2_post_reclosure_terminal_w2",
    "rebase_lfm2_installed_terminal_w2_for_output_reclosure",
]
