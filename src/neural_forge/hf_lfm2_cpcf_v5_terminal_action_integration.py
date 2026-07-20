"""Live V5 integration for the closed-form terminal-action residual fold.

The primitive in :mod:`hf_lfm2_cpcf_terminal_action_residual` is deliberately
model-free.  This module supplies the small LFM2-specific transaction around
that primitive: archive the actual final-macrocell inputs, derive four
trajectory-isolated evidence banks from the committed cache, publish a chosen
``w2`` into the existing parameter, and replay only the final installed layer.

No full-model forward, parent call, gradient, optimizer, pruning,
quantization, or parameter replacement is performed here.  The explicitly
counted target operation is a local final-layer replay.  A failed live replay
gate restores the incumbent writer bytes and proves the restoration with a
second local final-layer replay.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    Lfm2CpcfLayerReplayBank,
    Lfm2CpcfPhaseStateBank,
    Lfm2CpcfTargetPhase,
    replay_lfm2_cpcf_phase_bank,
    select_lfm2_cpcf_phase_state_bank,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    Lfm2CpcfTerminalActionEvidence,
    Lfm2CpcfTerminalActionPolicy,
    Lfm2CpcfTerminalActionSeed,
    compile_lfm2_cpcf_terminal_action_residual,
)
from neural_forge.hf_tied_causal_interface_fold import TiedCausalInterfaceFold


LFM2_CPCF_V5_TERMINAL_CAPTURE_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-terminal-action-capture-v1"
)
LFM2_CPCF_V5_TERMINAL_EVIDENCE_BUNDLE_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-terminal-action-evidence-bundle-v1"
)
LFM2_CPCF_V5_TERMINAL_INTEGRATION_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-terminal-action-integration-v1"
)
_FINAL_NORM_FIELD = "output.final_norm_output"
_FIT_CHECK_FRACTION_NUMERATOR = 1
_FIT_CHECK_FRACTION_DENOMINATOR = 8
_PARTITION_SEED = "lfm2-cpcf-v5-terminal-action-fit-check-v1"
_ZERO_OPERATION_AUDIT = MappingProxyType(
    {
        "parent_model_forward_calls": 0,
        "target_full_model_forward_calls": 0,
        "target_final_layer_replay_calls": 0,
        "generation_calls": 0,
        "parent_cache_generation_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_replaced": 0,
        "new_parameters_created": 0,
    }
)


class Lfm2CpcfV5TerminalActionIntegrationError(RuntimeError):
    """Terminal evidence, publication, live replay, or rollback failed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for terminal integration.") from exc
    return torch


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Only tensors have terminal integration tensor seals.")
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).removeprefix("torch.").encode("ascii"))
    digest.update(b"\0")
    digest.update(
        str(tuple(int(size) for size in tensor.shape)).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _owned_bfloat16(value: Any, name: str, *, dimensions: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or value.numel() == 0
        or value.dtype != torch.bfloat16
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            f"{name} must be detached dense exact-BF16 rank-{dimensions} evidence."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            f"{name} contains NaN or infinity."
        )
    return result


def _phase_lineage(phases: Sequence[Lfm2CpcfTargetPhase]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (phase.identity, phase.phase_sha256, phase.split, phase.record_id)
        for phase in phases
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5TerminalActionCapture:
    """Minimal actual-child archive retained across final evidence release."""

    layer_index: int
    input_bank: Lfm2CpcfPhaseStateBank
    mlp_input: Mapping[tuple[int, str], Any]
    pre_mlp_residual: Mapping[tuple[int, str], Any]
    incumbent_final_state_tensor_sha256: Mapping[tuple[int, str], str]
    incumbent_final_state_bank_sha256: str
    authentication_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
            or not isinstance(self.input_bank, Lfm2CpcfPhaseStateBank)
            or self.input_bank.layer_index != self.layer_index
        ):
            raise Lfm2CpcfV5TerminalActionIntegrationError(
                "Terminal capture has an invalid final-layer input bank."
            )
        identities = tuple(phase.identity for phase in self.input_bank.phases)
        if set(self.mlp_input) != set(identities) or set(
            self.pre_mlp_residual
        ) != set(identities) or set(self.incumbent_final_state_tensor_sha256) != set(
            identities
        ):
            raise Lfm2CpcfV5TerminalActionIntegrationError(
                "Terminal endpoint identities differ from the input bank."
            )
        mlp_inputs = {}
        pre_mlp = {}
        incumbent_hashes = {}
        for phase in self.input_bank.phases:
            identity = phase.identity
            mlp = _owned_bfloat16(
                self.mlp_input[identity],
                f"mlp_input[{identity!r}]",
                dimensions=3,
            )
            residual = _owned_bfloat16(
                self.pre_mlp_residual[identity],
                f"pre_mlp_residual[{identity!r}]",
                dimensions=3,
            )
            expected = (
                phase.batch_size,
                phase.sampled_suffix_tokens,
                self.input_bank.hidden_size,
            )
            if tuple(mlp.shape) != expected or tuple(residual.shape) != expected:
                raise Lfm2CpcfV5TerminalActionIntegrationError(
                    f"Terminal endpoints for {identity!r} differ from its prefix."
                )
            mlp_inputs[identity] = mlp
            pre_mlp[identity] = residual
            incumbent_hash = self.incumbent_final_state_tensor_sha256[identity]
            if (
                not isinstance(incumbent_hash, str)
                or len(incumbent_hash) != 64
                or any(value not in "0123456789abcdef" for value in incumbent_hash)
            ):
                raise Lfm2CpcfV5TerminalActionIntegrationError(
                    f"Incumbent terminal state seal for {identity!r} is invalid."
                )
            incumbent_hashes[identity] = incumbent_hash
        body = {
            "schema": LFM2_CPCF_V5_TERMINAL_CAPTURE_SCHEMA,
            "layer_index": self.layer_index,
            "input_bank_sha256": self.input_bank.state_bank_sha256,
            "incumbent_final_state_bank_sha256": (
                self.incumbent_final_state_bank_sha256
            ),
            "phase_lineage": [
                {
                    "identity": [phase.batch_index, phase.phase],
                    "record_id": phase.record_id,
                    "split": phase.split,
                    "phase_sha256": phase.phase_sha256,
                    "mlp_input_sha256": _tensor_sha256(mlp_inputs[phase.identity]),
                    "pre_mlp_residual_sha256": _tensor_sha256(
                        pre_mlp[phase.identity]
                    ),
                    "input_state_sha256": _tensor_sha256(
                        self.input_bank.states[phase.identity]
                    ),
                    "incumbent_final_state_sha256": incumbent_hashes[
                        phase.identity
                    ],
                }
                for phase in self.input_bank.phases
            ],
            "actual_child_final_layer_input": True,
            "actual_child_final_layer_endpoints": True,
            "tensor_free_parent_phase_shells": all(
                not phase.parent_evidence.final_norm_tensors
                and not any(
                    layer.tensors for layer in phase.parent_evidence.layers
                )
                for phase in self.input_bank.phases
            ),
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }
        expected_sha = canonical_sha256(body)
        if self.authentication_sha256 != expected_sha:
            raise Lfm2CpcfV5TerminalActionIntegrationError(
                "Terminal capture authentication is invalid."
            )
        object.__setattr__(self, "mlp_input", MappingProxyType(mlp_inputs))
        object.__setattr__(
            self,
            "pre_mlp_residual",
            MappingProxyType(pre_mlp),
        )
        object.__setattr__(
            self,
            "incumbent_final_state_tensor_sha256",
            MappingProxyType(incumbent_hashes),
        )

    def verify_authentication(self) -> bool:
        try:
            return self.authentication_sha256 == _capture_body_sha256(
                self.layer_index,
                self.input_bank,
                self.mlp_input,
                self.pre_mlp_residual,
                self.incumbent_final_state_tensor_sha256,
                self.incumbent_final_state_bank_sha256,
            )
        except (TypeError, ValueError, RuntimeError):
            return False


def _capture_body_sha256(
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    mlp_input: Mapping[tuple[int, str], Any],
    pre_mlp_residual: Mapping[tuple[int, str], Any],
    incumbent_final_state_tensor_sha256: Mapping[tuple[int, str], str],
    incumbent_final_state_bank_sha256: str,
) -> str:
    body = {
        "schema": LFM2_CPCF_V5_TERMINAL_CAPTURE_SCHEMA,
        "layer_index": layer_index,
        "input_bank_sha256": input_bank.state_bank_sha256,
        "incumbent_final_state_bank_sha256": incumbent_final_state_bank_sha256,
        "phase_lineage": [
            {
                "identity": [phase.batch_index, phase.phase],
                "record_id": phase.record_id,
                "split": phase.split,
                "phase_sha256": phase.phase_sha256,
                "mlp_input_sha256": _tensor_sha256(mlp_input[phase.identity]),
                "pre_mlp_residual_sha256": _tensor_sha256(
                    pre_mlp_residual[phase.identity]
                ),
                "input_state_sha256": _tensor_sha256(
                    input_bank.states[phase.identity]
                ),
                "incumbent_final_state_sha256": (
                    incumbent_final_state_tensor_sha256[phase.identity]
                ),
            }
            for phase in input_bank.phases
        ],
        "actual_child_final_layer_input": True,
        "actual_child_final_layer_endpoints": True,
        "tensor_free_parent_phase_shells": all(
            not phase.parent_evidence.final_norm_tensors
            and not any(layer.tensors for layer in phase.parent_evidence.layers)
            for phase in input_bank.phases
        ),
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    return canonical_sha256(body)


def capture_lfm2_cpcf_v5_terminal_action_inputs(
    *,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    final_replay: Lfm2CpcfLayerReplayBank,
) -> Lfm2CpcfV5TerminalActionCapture:
    """Archive only the final layer input and its two pre-writer endpoints."""

    if not isinstance(input_bank, Lfm2CpcfPhaseStateBank) or not isinstance(
        final_replay,
        Lfm2CpcfLayerReplayBank,
    ):
        raise TypeError("input_bank and final_replay must be typed LFM2 archives.")
    if (
        layer_index != input_bank.layer_index
        or final_replay.layer_index != layer_index
        or final_replay.input_bank_sha256 != input_bank.state_bank_sha256
        or _phase_lineage(final_replay.post_layer_bank.phases)
        != _phase_lineage(input_bank.phases)
    ):
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            "Final replay does not descend from the supplied actual-child bank."
        )
    splits = tuple(dict.fromkeys(phase.split for phase in input_bank.phases))
    owned_input_bank = select_lfm2_cpcf_phase_state_bank(
        input_bank,
        splits=splits,
    )
    if owned_input_bank.state_bank_sha256 != input_bank.state_bank_sha256:
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            "Owning the final input bank changed its authenticated state."
        )
    mlp_input = {
        phase.identity: phase.suffix(
            final_replay.replays[phase.identity].mlp_input
        )
        for phase in input_bank.phases
    }
    pre_mlp = {
        phase.identity: phase.suffix(
            final_replay.replays[phase.identity].post_operator_residual
        )
        for phase in input_bank.phases
    }
    incumbent_state_hashes = {
        phase.identity: _tensor_sha256(
            final_replay.post_layer_bank.states[phase.identity]
        )
        for phase in input_bank.phases
    }
    seal = _capture_body_sha256(
        layer_index,
        owned_input_bank,
        mlp_input,
        pre_mlp,
        incumbent_state_hashes,
        final_replay.post_layer_bank.state_bank_sha256,
    )
    return Lfm2CpcfV5TerminalActionCapture(
        layer_index=layer_index,
        input_bank=owned_input_bank,
        mlp_input=mlp_input,
        pre_mlp_residual=pre_mlp,
        incumbent_final_state_tensor_sha256=incumbent_state_hashes,
        incumbent_final_state_bank_sha256=(
            final_replay.post_layer_bank.state_bank_sha256
        ),
        authentication_sha256=seal,
    )


def _complete_trajectories(
    phases: Sequence[Lfm2CpcfTargetPhase],
) -> tuple[tuple[tuple[int, str], tuple[Lfm2CpcfTargetPhase, ...]], ...]:
    torch = _torch()
    ordered = tuple(phases)
    grouped: dict[tuple[int, str], list[Lfm2CpcfTargetPhase]] = {}
    order: list[tuple[int, str]] = []
    for phase in ordered:
        key = (phase.batch_index, phase.record_id)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(phase)
    result = []
    for key in order:
        trajectory = tuple(grouped[key])
        first = trajectory[0]
        expected_names = (first.phase,) + tuple(
            f"decode.{index:03d}" for index in range(len(trajectory) - 1)
        )
        if (
            first.phase not in {"prefill", "cached_prefill"}
            or tuple(phase.phase for phase in trajectory) != expected_names
            or len({phase.split for phase in trajectory}) != 1
        ):
            raise Lfm2CpcfV5TerminalActionIntegrationError(
                f"Trajectory {key!r} is not one complete ordered causal path."
            )
        previous = first
        for phase in trajectory[1:]:
            old_tokens = previous.full_prefix_tokens
            if (
                phase.full_prefix_tokens
                != old_tokens + phase.sampled_suffix_tokens
                or not torch.equal(phase.input_ids[:, :old_tokens], previous.input_ids)
                or not torch.equal(
                    phase.attention_mask[:, :old_tokens],
                    previous.attention_mask,
                )
                or not torch.equal(
                    phase.position_ids[:, :old_tokens],
                    previous.position_ids,
                )
            ):
                raise Lfm2CpcfV5TerminalActionIntegrationError(
                    f"Trajectory {key!r} changed an authenticated prefix."
                )
            previous = phase
        result.append((key, trajectory))
    return tuple(result)


def _trajectory_sha256(
    key: tuple[int, str],
    phases: Sequence[Lfm2CpcfTargetPhase],
    capture: Lfm2CpcfV5TerminalActionCapture,
) -> str:
    return canonical_sha256(
        {
            "batch_index": key[0],
            "record_id": key[1],
            "phases": [
                {
                    "phase_sha256": phase.phase_sha256,
                    "mlp_input_sha256": _tensor_sha256(
                        capture.mlp_input[phase.identity]
                    ),
                    "pre_mlp_residual_sha256": _tensor_sha256(
                        capture.pre_mlp_residual[phase.identity]
                    ),
                }
                for phase in phases
            ],
        }
    )


def _fit_check_trajectory_count(source_fit_trajectory_count: int) -> int:
    if (
        isinstance(source_fit_trajectory_count, bool)
        or not isinstance(source_fit_trajectory_count, int)
        or source_fit_trajectory_count < 2
    ):
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            "FIT-CHECK partitioning requires at least two FIT trajectories."
        )
    count = (
        source_fit_trajectory_count * _FIT_CHECK_FRACTION_NUMERATOR
        + _FIT_CHECK_FRACTION_DENOMINATOR
        - 1
    ) // _FIT_CHECK_FRACTION_DENOMINATOR
    return max(1, min(source_fit_trajectory_count - 1, count))


def _evidence_for_trajectories(
    split: str,
    trajectories: Sequence[
        tuple[tuple[int, str], tuple[Lfm2CpcfTargetPhase, ...]]
    ],
    capture: Lfm2CpcfV5TerminalActionCapture,
) -> Lfm2CpcfTerminalActionEvidence:
    torch = _torch()
    values = tuple(trajectories)
    if not values:
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            f"Terminal {split} contains no complete trajectories."
        )
    mlp_pieces = []
    pre_pieces = []
    parent_pieces = []
    weight_pieces = []
    row_ids = []
    trajectory_mass = 1.0 / len(values)
    for key, phases in values:
        rows_in_trajectory = sum(
            phase.batch_size * phase.sampled_suffix_tokens for phase in phases
        )
        for phase in phases:
            identity = phase.identity
            mlp = capture.mlp_input[identity].reshape(
                -1,
                capture.input_bank.hidden_size,
            )
            pre = capture.pre_mlp_residual[identity].reshape(
                -1,
                capture.input_bank.hidden_size,
            )
            try:
                parent = phase.parent_evidence.final_norm_tensors[_FINAL_NORM_FIELD]
            except KeyError as exc:
                raise Lfm2CpcfV5TerminalActionIntegrationError(
                    f"Phase {identity!r} has no live parent final output."
                ) from exc
            parent = _owned_bfloat16(
                parent,
                f"parent_final_norm_output[{identity!r}]",
                dimensions=3,
            ).reshape(-1, int(parent.shape[-1]))
            if int(parent.shape[0]) != int(mlp.shape[0]):
                raise Lfm2CpcfV5TerminalActionIntegrationError(
                    f"Phase {identity!r} parent/child row geometry differs."
                )
            mlp_pieces.append(mlp)
            pre_pieces.append(pre)
            parent_pieces.append(parent)
            weight_pieces.append(
                torch.full(
                    (int(mlp.shape[0]),),
                    trajectory_mass / rows_in_trajectory,
                    dtype=torch.float64,
                )
            )
            first_position = phase.full_prefix_tokens - phase.sampled_suffix_tokens
            for batch_row in range(phase.batch_size):
                for suffix_index in range(phase.sampled_suffix_tokens):
                    row_ids.append(
                        f"{split}|batch={key[0]}|record={key[1]}|"
                        f"phase={phase.phase}|row={batch_row}|"
                        f"position={first_position + suffix_index}"
                    )
    return Lfm2CpcfTerminalActionEvidence(
        split=split,
        row_ids=tuple(row_ids),
        mlp_input=torch.cat(mlp_pieces, dim=0),
        pre_mlp_residual=torch.cat(pre_pieces, dim=0),
        parent_final_norm_output=torch.cat(parent_pieces, dim=0),
        sample_weights=torch.cat(weight_pieces, dim=0),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5TerminalActionEvidenceBundle:
    """Four authenticated banks and their whole-trajectory partition receipt."""

    fit: Lfm2CpcfTerminalActionEvidence
    fit_check: Lfm2CpcfTerminalActionEvidence
    selection: Lfm2CpcfTerminalActionEvidence
    validation: Lfm2CpcfTerminalActionEvidence
    report: Mapping[str, Any]
    bundle_sha256: str

    def __post_init__(self) -> None:
        banks = (self.fit, self.fit_check, self.selection, self.validation)
        if tuple(bank.split for bank in banks) != (
            "fit",
            "fit_check",
            "selection",
            "validation",
        ) or not all(bank.verify_authentication() for bank in banks):
            raise Lfm2CpcfV5TerminalActionIntegrationError(
                "Terminal evidence bundle contains an invalid bank."
            )
        body = dict(self.report)
        supplied = body.pop("bundle_sha256", None)
        if supplied != self.bundle_sha256 or canonical_sha256(body) != supplied:
            raise Lfm2CpcfV5TerminalActionIntegrationError(
                "Terminal evidence bundle seal is invalid."
            )
        if body.get("split_evidence_sha256") != {
            bank.split: bank.authentication_sha256 for bank in banks
        }:
            raise Lfm2CpcfV5TerminalActionIntegrationError(
                "Terminal bundle report differs from its evidence banks."
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**body, "bundle_sha256": supplied}),
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def build_lfm2_cpcf_v5_terminal_action_evidence(
    capture: Lfm2CpcfV5TerminalActionCapture,
    final_phases: Sequence[Lfm2CpcfTargetPhase],
) -> Lfm2CpcfV5TerminalActionEvidenceBundle:
    """Subpartition FIT by whole trajectory and retain existing held-outs."""

    if not isinstance(capture, Lfm2CpcfV5TerminalActionCapture) or not (
        capture.verify_authentication()
    ):
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            "Terminal capture changed before evidence assembly."
        )
    phases = tuple(final_phases)
    if _phase_lineage(phases) != _phase_lineage(capture.input_bank.phases):
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            "Final-normalization phases differ from the captured child phases."
        )
    trajectories = _complete_trajectories(phases)
    by_split = {
        split: tuple(value for value in trajectories if value[1][0].split == split)
        for split in ("fit", "selection", "validation")
    }
    fit_values = by_split["fit"]
    if len(fit_values) < 2:
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            "Terminal FIT requires at least two complete trajectories so "
            "FIT-CHECK remains isolated."
        )
    fit_check_count = _fit_check_trajectory_count(len(fit_values))
    ranked = sorted(
        fit_values,
        key=lambda value: (
            canonical_sha256(
                {
                    "schema": LFM2_CPCF_V5_TERMINAL_EVIDENCE_BUNDLE_SCHEMA,
                    "partition_seed": _PARTITION_SEED,
                    "trajectory_sha256": _trajectory_sha256(
                        value[0],
                        value[1],
                        capture,
                    ),
                }
            ),
            value[0],
        ),
    )
    fit_check_keys = {value[0] for value in ranked[:fit_check_count]}
    retained_fit = tuple(value for value in fit_values if value[0] not in fit_check_keys)
    fit_check = tuple(value for value in fit_values if value[0] in fit_check_keys)
    bank_values = (
        _evidence_for_trajectories("fit", retained_fit, capture),
        _evidence_for_trajectories("fit_check", fit_check, capture),
        _evidence_for_trajectories("selection", by_split["selection"], capture),
        _evidence_for_trajectories("validation", by_split["validation"], capture),
    )
    trajectory_ids = {
        "fit": [[key[0], key[1]] for key, _phases in retained_fit],
        "fit_check": [[key[0], key[1]] for key, _phases in fit_check],
        "selection": [
            [key[0], key[1]] for key, _phases in by_split["selection"]
        ],
        "validation": [
            [key[0], key[1]] for key, _phases in by_split["validation"]
        ],
    }
    body = {
        "schema": LFM2_CPCF_V5_TERMINAL_EVIDENCE_BUNDLE_SCHEMA,
        "capture_sha256": capture.authentication_sha256,
        "partition_seed": _PARTITION_SEED,
        "fit_check_fraction": {
            "numerator": _FIT_CHECK_FRACTION_NUMERATOR,
            "denominator": _FIT_CHECK_FRACTION_DENOMINATOR,
        },
        "fit_check_count_rule": "ceil(source_fit_trajectory_count / 8)",
        "source_fit_trajectory_count": len(fit_values),
        "trajectory_ids": trajectory_ids,
        "trajectory_counts": {
            name: len(values) for name, values in trajectory_ids.items()
        },
        "split_rows": {bank.split: bank.rows for bank in bank_values},
        "split_evidence_sha256": {
            bank.split: bank.authentication_sha256 for bank in bank_values
        },
        "whole_trajectory_fit_fit_check_isolation": True,
        "existing_selection_retained": True,
        "existing_validation_retained": True,
        "new_causal_suffix_rows_only": True,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    seal = canonical_sha256(body)
    return Lfm2CpcfV5TerminalActionEvidenceBundle(
        fit=bank_values[0],
        fit_check=bank_values[1],
        selection=bank_values[2],
        validation=bank_values[3],
        report={**body, "bundle_sha256": seal},
        bundle_sha256=seal,
    )


def _target_terminal_seed(
    target_model: Any,
    interface: TiedCausalInterfaceFold,
    source_tied_table: Any,
    *,
    norm_epsilon: float,
) -> tuple[Lfm2CpcfTerminalActionSeed, Any]:
    torch = _torch()
    core = getattr(target_model, "model", None)
    layers = tuple(getattr(core, "layers", ()))
    if not layers or not isinstance(interface, TiedCausalInterfaceFold):
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            "Terminal integration requires a materialized LFM2 target/interface."
        )
    final_layer = layers[-1]
    feed_forward = getattr(final_layer, "feed_forward", None)
    final_norm = getattr(getattr(core, "embedding_norm", None), "weight", None)
    target_table = getattr(target_model.get_input_embeddings(), "weight", None)
    output_table = getattr(target_model.get_output_embeddings(), "weight", None)
    parameters = (
        getattr(getattr(feed_forward, "w1", None), "weight", None),
        getattr(getattr(feed_forward, "w3", None), "weight", None),
        getattr(getattr(feed_forward, "w2", None), "weight", None),
        final_norm,
        target_table,
    )
    if not all(isinstance(value, torch.nn.Parameter) for value in parameters):
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            "Target final SwiGLU, RMSNorm, or tied table is not materialized."
        )
    gate, up, down, norm, target_weight = parameters
    if (
        output_table is not target_weight
        or any(value.dtype != torch.bfloat16 for value in parameters)
        or not torch.equal(
            target_weight.detach().to(device="cpu"),
            interface.target_weight.to(torch.bfloat16),
        )
    ):
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            "Terminal integration requires the exact installed tied BF16 interface."
        )
    seed = Lfm2CpcfTerminalActionSeed(
        gate_weight=gate,
        up_weight=up,
        down_weight=down,
        final_norm_weight=norm,
        producer_encoder=interface.producer_encoder,
        target_tied_weight=target_weight,
        source_tied_table=source_tied_table,
        norm_epsilon=norm_epsilon,
    )
    return seed, down


def _rmsnorm(rows: Any, weight: Any, epsilon: float) -> Any:
    values = rows.to(_torch().float64)
    return (
        values
        * (values.square().mean(dim=1, keepdim=True) + epsilon).rsqrt()
        * weight.to(_torch().float64)
    )


@dataclass(slots=True)
class _LivePairAccumulator:
    error_square: float = 0.0
    reference_square: float = 0.0
    top1: float = 0.0
    margin: float = 0.0
    positive_margin: float = 0.0


def _accumulate_live_pair(
    accumulator: _LivePairAccumulator,
    candidate: Any,
    reference: Any,
    weights: Any,
) -> None:
    torch = _torch()
    candidate_centered = candidate - candidate.mean(dim=1, keepdim=True)
    reference_centered = reference - reference.mean(dim=1, keepdim=True)
    difference = candidate_centered - reference_centered
    accumulator.error_square += float(
        (weights[:, None] * difference.square()).sum().item()
    )
    accumulator.reference_square += float(
        (weights[:, None] * reference_centered.square()).sum().item()
    )
    winner = reference.argmax(dim=1, keepdim=True)
    candidate_winner = candidate.argmax(dim=1, keepdim=True)
    top_two = torch.topk(candidate, k=2, dim=1)
    best_is_reference = (top_two.indices[:, :1] == winner).squeeze(1)
    competitor = torch.where(best_is_reference, top_two.values[:, 1], top_two.values[:, 0])
    margin = candidate.gather(1, winner).squeeze(1) - competitor
    accumulator.top1 += float(
        (
            weights * (candidate_winner == winner).squeeze(1).to(torch.float64)
        )
        .sum()
        .item()
    )
    accumulator.margin += float((weights * margin).sum().item())
    accumulator.positive_margin += float(
        (weights * (margin > 0.0).to(torch.float64)).sum().item()
    )


def _finish_live_pair(accumulator: _LivePairAccumulator) -> dict[str, float]:
    torch = _torch()
    denominator = max(
        accumulator.reference_square,
        torch.finfo(torch.float64).tiny,
    )
    tolerance = 64.0 * torch.finfo(torch.float64).eps
    if (
        accumulator.top1 < -tolerance
        or accumulator.top1 > 1.0 + tolerance
        or accumulator.positive_margin < -tolerance
        or accumulator.positive_margin > 1.0 + tolerance
    ):
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            "Live weighted logit fractions escaped [0,1]."
        )

    def bounded_fraction(value: float) -> float:
        if abs(value) <= tolerance:
            return 0.0
        if abs(value - 1.0) <= tolerance:
            return 1.0
        return min(1.0, max(0.0, value))

    return {
        "centered_logit_relative_rmse": math.sqrt(
            max(0.0, accumulator.error_square) / denominator
        ),
        "top1_agreement_fraction": bounded_fraction(accumulator.top1),
        "reference_winner_mean_margin": accumulator.margin,
        "reference_winner_positive_margin_fraction": bounded_fraction(
            accumulator.positive_margin
        ),
    }


def _live_split_metrics(
    bank: Lfm2CpcfPhaseStateBank,
    phases: Sequence[Lfm2CpcfTargetPhase],
    evidence: Lfm2CpcfTerminalActionEvidence,
    seed: Lfm2CpcfTerminalActionSeed,
    *,
    metric_chunk_rows: int,
) -> dict[str, Any]:
    torch = _torch()
    child = torch.cat(
        [
            bank.suffix(phase).reshape(-1, bank.hidden_size)
            for phase in phases
        ],
        dim=0,
    ).to(torch.float64)
    if int(child.shape[0]) != evidence.rows:
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            f"Live {evidence.split} replay row count changed."
        )
    normalized = _rmsnorm(child, seed.final_norm_weight, seed.norm_epsilon)
    parent = evidence.parent_final_norm_output.to(torch.float64)
    oracle = parent @ seed.producer_encoder
    target_table = seed.target_tied_weight.to(torch.float64)
    source_table = seed.source_tied_table.to(torch.float64)
    weights = evidence.sample_weights
    oracle_accumulator = _LivePairAccumulator()
    parent_accumulator = _LivePairAccumulator()
    finite = bool(
        torch.isfinite(child).all().item()
        and torch.isfinite(normalized).all().item()
        and torch.isfinite(oracle).all().item()
        and torch.isfinite(parent).all().item()
    )
    for start in range(0, evidence.rows, metric_chunk_rows):
        stop = min(evidence.rows, start + metric_chunk_rows)
        candidate_logits = normalized[start:stop] @ target_table.T
        oracle_logits = oracle[start:stop] @ target_table.T
        parent_logits = parent[start:stop] @ source_table.T
        finite = bool(
            finite
            and torch.isfinite(candidate_logits).all().item()
            and torch.isfinite(oracle_logits).all().item()
            and torch.isfinite(parent_logits).all().item()
        )
        _accumulate_live_pair(
            oracle_accumulator,
            candidate_logits,
            oracle_logits,
            weights[start:stop],
        )
        _accumulate_live_pair(
            parent_accumulator,
            candidate_logits,
            parent_logits,
            weights[start:stop],
        )
    rms = child.square().mean(dim=1).sqrt()
    return {
        "oracle": _finish_live_pair(oracle_accumulator),
        "mapped_parent": _finish_live_pair(parent_accumulator),
        "row_rms": rms,
        "finite": finite,
        "metric_chunk_rows": metric_chunk_rows,
    }


def _live_metric_gates(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    policy: Lfm2CpcfTerminalActionPolicy,
    *,
    selection: bool,
) -> dict[str, bool]:
    torch = _torch()
    tolerance = policy.metric_regression_tolerance
    candidate_rms = candidate["row_rms"]
    incumbent_rms = incumbent["row_rms"]
    ratios = candidate_rms / incumbent_rms.clamp_min(
        torch.finfo(torch.float64).tiny
    )
    base_oracle = incumbent["oracle"]
    next_oracle = candidate["oracle"]
    base_parent = incumbent["mapped_parent"]
    next_parent = candidate["mapped_parent"]
    base_rmse = base_oracle["centered_logit_relative_rmse"]
    next_rmse = next_oracle["centered_logit_relative_rmse"]
    improvement = (base_rmse - next_rmse) / max(
        base_rmse,
        torch.finfo(torch.float64).tiny,
    )
    gates = {
        "finite": bool(candidate["finite"]),
        "post_layer_norm": bool(
            float(ratios.max().item()) <= policy.maximum_post_layer_rms_ratio
        ),
        "oracle_centered_rmse_nonregression": bool(
            next_rmse <= base_rmse + tolerance
        ),
        "oracle_top1_nonregression": bool(
            next_oracle["top1_agreement_fraction"] + tolerance
            >= base_oracle["top1_agreement_fraction"]
        ),
        "mapped_parent_centered_rmse_nonregression": bool(
            next_parent["centered_logit_relative_rmse"]
            <= base_parent["centered_logit_relative_rmse"] + tolerance
        ),
        "mapped_parent_top1_nonregression": bool(
            next_parent["top1_agreement_fraction"] + tolerance
            >= base_parent["top1_agreement_fraction"]
        ),
        "mapped_parent_positive_margin_nonregression": bool(
            next_parent["reference_winner_positive_margin_fraction"] + tolerance
            >= base_parent["reference_winner_positive_margin_fraction"]
        ),
    }
    if selection:
        gates.update(
            {
                "selection_centered_rmse_improvement": bool(
                    improvement + tolerance
                    >= policy.minimum_selection_centered_rmse_improvement_fraction
                ),
                "selection_oracle_top1_gain": bool(
                    next_oracle["top1_agreement_fraction"]
                    - base_oracle["top1_agreement_fraction"]
                    + tolerance
                    >= policy.minimum_selection_oracle_top1_gain
                ),
                "selection_parent_top1_gain": bool(
                    next_parent["top1_agreement_fraction"]
                    - base_parent["top1_agreement_fraction"]
                    + tolerance
                    >= policy.minimum_selection_parent_top1_gain
                ),
                "selection_parent_positive_margin_gain": bool(
                    next_parent["reference_winner_positive_margin_fraction"]
                    - base_parent["reference_winner_positive_margin_fraction"]
                    + tolerance
                    >= policy.minimum_selection_parent_positive_margin_gain
                ),
            }
        )
    return gates


def _public_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "oracle": dict(value["oracle"]),
        "mapped_parent": dict(value["mapped_parent"]),
        "maximum_row_rms": float(value["row_rms"].max().item()),
        "mean_row_rms": float(value["row_rms"].mean().item()),
        "finite": bool(value["finite"]),
    }


def _upstream_endpoint_gate(
    capture: Lfm2CpcfV5TerminalActionCapture,
    replay: Lfm2CpcfLayerReplayBank,
) -> tuple[bool, list[dict[str, Any]]]:
    records = []
    passed = True
    for phase in capture.input_bank.phases:
        archived = replay.replays[phase.identity]
        mlp_equal = bool(
            _torch().equal(
                phase.suffix(archived.mlp_input),
                capture.mlp_input[phase.identity],
            )
        )
        pre_equal = bool(
            _torch().equal(
                phase.suffix(archived.post_operator_residual),
                capture.pre_mlp_residual[phase.identity],
            )
        )
        passed = bool(passed and mlp_equal and pre_equal)
        records.append(
            {
                "identity": [phase.batch_index, phase.phase],
                "mlp_input_exact": mlp_equal,
                "pre_mlp_residual_exact": pre_equal,
                "replay_sha256": archived.replay_sha256,
            }
        )
    return passed, records


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5TerminalActionIntegrationResult:
    """Published or exactly restored final state plus authenticated receipt."""

    final_state_bank: Lfm2CpcfPhaseStateBank
    accepted: bool
    report: Mapping[str, Any]
    result_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.final_state_bank, Lfm2CpcfPhaseStateBank) or not isinstance(
            self.accepted,
            bool,
        ):
            raise TypeError("Terminal integration result has invalid typed state.")
        body = dict(self.report)
        supplied = body.pop("result_sha256", None)
        if (
            supplied != self.result_sha256
            or canonical_sha256(body) != supplied
            or body.get("published_final_state_bank_sha256")
            != self.final_state_bank.state_bank_sha256
            or body.get("accepted") is not self.accepted
        ):
            raise Lfm2CpcfV5TerminalActionIntegrationError(
                "Terminal integration result seal is invalid."
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**body, "result_sha256": supplied}),
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def compile_install_lfm2_cpcf_v5_terminal_action(
    *,
    target_model: Any,
    capture: Lfm2CpcfV5TerminalActionCapture,
    incumbent_final_state_bank: Lfm2CpcfPhaseStateBank,
    final_phases: Sequence[Lfm2CpcfTargetPhase],
    interface: TiedCausalInterfaceFold,
    source_tied_table: Any,
    norm_epsilon: float,
    policy: Lfm2CpcfTerminalActionPolicy,
) -> Lfm2CpcfV5TerminalActionIntegrationResult:
    """Compile, install, replay, live-gate, and locally roll back TARC."""

    torch = _torch()
    if (
        not isinstance(policy, Lfm2CpcfTerminalActionPolicy)
        or not policy.verify_authentication()
        or not isinstance(capture, Lfm2CpcfV5TerminalActionCapture)
        or not capture.verify_authentication()
        or not isinstance(incumbent_final_state_bank, Lfm2CpcfPhaseStateBank)
        or incumbent_final_state_bank.state_bank_sha256
        != capture.incumbent_final_state_bank_sha256
        or any(
            _tensor_sha256(incumbent_final_state_bank.states[phase.identity])
            != capture.incumbent_final_state_tensor_sha256[phase.identity]
            for phase in incumbent_final_state_bank.phases
        )
    ):
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            "Terminal integration inputs are unauthenticated or misbound."
        )
    phases = tuple(final_phases)
    evidence = build_lfm2_cpcf_v5_terminal_action_evidence(capture, phases)
    seed, down_parameter = _target_terminal_seed(
        target_model,
        interface,
        source_tied_table,
        norm_epsilon=norm_epsilon,
    )
    parameter_identity = {
        "python_id": id(down_parameter),
        "data_ptr": int(down_parameter.data_ptr()),
        "shape": [int(size) for size in down_parameter.shape],
        "dtype": str(down_parameter.dtype).removeprefix("torch."),
        "device": str(down_parameter.device),
    }
    incumbent_weight = down_parameter.detach().to(device="cpu").contiguous().clone()
    compiled = compile_lfm2_cpcf_terminal_action_residual(
        seed=seed,
        fit_evidence=evidence.fit,
        fit_check_evidence=evidence.fit_check,
        selection_evidence=evidence.selection,
        validation_evidence=evidence.validation,
        policy=policy,
    )
    if not compiled.verify_authentication():
        raise Lfm2CpcfV5TerminalActionIntegrationError(
            "The pure terminal-action result failed authentication."
        )
    split_trajectories = evidence.report["trajectory_ids"]
    phase_lookup = {(phase.batch_index, phase.record_id): [] for phase in phases}
    for phase in phases:
        phase_lookup[(phase.batch_index, phase.record_id)].append(phase)

    def phases_for(split: str) -> tuple[Lfm2CpcfTargetPhase, ...]:
        return tuple(
            phase
            for raw_key in split_trajectories[split]
            for phase in phase_lookup[(int(raw_key[0]), str(raw_key[1]))]
        )

    evidence_by_split = {
        "fit": evidence.fit,
        "fit_check": evidence.fit_check,
        "selection": evidence.selection,
        "validation": evidence.validation,
    }
    incumbent_live_metrics = {
        split: _live_split_metrics(
            incumbent_final_state_bank,
            phases_for(split),
            bank,
            seed,
            metric_chunk_rows=policy.metric_chunk_rows,
        )
        for split, bank in evidence_by_split.items()
    }
    installed = False
    target_final_layer_replay_calls = 0
    live_gate_passed = False
    live_gate_records: list[dict[str, Any]] = []
    candidate_live_metrics: dict[str, Mapping[str, Any]] = {}
    live_gates: dict[str, Mapping[str, bool]] = {}
    rollback_performed = False
    rollback_exact = True
    published_bank = incumbent_final_state_bank
    if compiled.accepted and compiled.changed:
        with torch.no_grad():
            down_parameter.copy_(
                compiled.published_down_weight.to(
                    device=down_parameter.device,
                    dtype=down_parameter.dtype,
                )
            )
        installed = True
        identity_after = {
            "python_id": id(down_parameter),
            "data_ptr": int(down_parameter.data_ptr()),
            "shape": [int(size) for size in down_parameter.shape],
            "dtype": str(down_parameter.dtype).removeprefix("torch."),
            "device": str(down_parameter.device),
        }
        if identity_after != parameter_identity or not torch.equal(
            down_parameter.detach().to(device="cpu"),
            compiled.published_down_weight,
        ):
            raise Lfm2CpcfV5TerminalActionIntegrationError(
                "Terminal writer installation changed Parameter identity or bytes."
            )
        candidate_replay = replay_lfm2_cpcf_phase_bank(
            target_model,
            layer_index=capture.layer_index,
            input_bank=capture.input_bank,
        )
        target_final_layer_replay_calls += 1
        upstream_passed, live_gate_records = _upstream_endpoint_gate(
            capture,
            candidate_replay,
        )
        candidate_live_metrics = {
            split: _live_split_metrics(
                candidate_replay.post_layer_bank,
                phases_for(split),
                bank,
                seed,
                metric_chunk_rows=policy.metric_chunk_rows,
            )
            for split, bank in evidence_by_split.items()
        }
        live_gates = {
            split: _live_metric_gates(
                candidate_live_metrics[split],
                incumbent_live_metrics[split],
                policy,
                selection=split == "selection",
            )
            for split in evidence_by_split
        }
        live_gate_passed = bool(
            upstream_passed
            and all(all(gates.values()) for gates in live_gates.values())
        )
        if live_gate_passed:
            published_bank = candidate_replay.post_layer_bank
        else:
            rollback_performed = True
            with torch.no_grad():
                down_parameter.copy_(
                    incumbent_weight.to(
                        device=down_parameter.device,
                        dtype=down_parameter.dtype,
                    )
                )
            restored_replay = replay_lfm2_cpcf_phase_bank(
                target_model,
                layer_index=capture.layer_index,
                input_bank=capture.input_bank,
            )
            target_final_layer_replay_calls += 1
            restored_upstream, _restored_records = _upstream_endpoint_gate(
                capture,
                restored_replay,
            )
            rollback_exact = bool(
                torch.equal(
                    down_parameter.detach().to(device="cpu"),
                    incumbent_weight,
                )
                and restored_upstream
                and restored_replay.post_layer_bank.state_bank_sha256
                == incumbent_final_state_bank.state_bank_sha256
            )
            if not rollback_exact:
                raise Lfm2CpcfV5TerminalActionIntegrationError(
                    "Terminal live-gate rollback did not exactly restore/replay."
                )
            published_bank = restored_replay.post_layer_bank
    accepted = bool(compiled.accepted and compiled.changed and live_gate_passed)
    body = {
        "schema": LFM2_CPCF_V5_TERMINAL_INTEGRATION_SCHEMA,
        "capture_sha256": capture.authentication_sha256,
        "evidence_bundle_sha256": evidence.bundle_sha256,
        "policy_sha256": policy.authentication_sha256,
        "pure_tarc_result_sha256": compiled.result_sha256,
        "pure_tarc_report": compiled.as_dict(),
        "pure_tarc_accepted": compiled.accepted,
        "pure_tarc_changed": compiled.changed,
        "writer_parameter_identity_before": parameter_identity,
        "writer_parameter_identity_preserved": True,
        "writer_installed_in_place": installed,
        "live_final_layer_replay_performed": installed,
        "live_upstream_endpoint_records": live_gate_records,
        "live_upstream_endpoints_exact": bool(
            not installed
            or all(
                record["mlp_input_exact"]
                and record["pre_mlp_residual_exact"]
                for record in live_gate_records
            )
        ),
        "live_incumbent_metrics": {
            split: _public_metrics(metrics)
            for split, metrics in incumbent_live_metrics.items()
        },
        "live_candidate_metrics": {
            split: _public_metrics(metrics)
            for split, metrics in candidate_live_metrics.items()
        },
        "live_gates": {
            split: dict(gates) for split, gates in live_gates.items()
        },
        "live_metric_chunk_rows": policy.metric_chunk_rows,
        "live_full_vocabulary_logit_materialization": False,
        "live_gate_passed": live_gate_passed,
        "rollback_performed": rollback_performed,
        "rollback_exact": rollback_exact,
        "accepted": accepted,
        "published_is_incumbent": not accepted,
        "incumbent_final_state_bank_sha256": (
            incumbent_final_state_bank.state_bank_sha256
        ),
        "published_final_state_bank_sha256": published_bank.state_bank_sha256,
        "parent_final_output_payload_live_during_compile": True,
        "final_layer_only_local_replay": True,
        "parameter_count_changed": False,
        "operation_audit": {
            **dict(_ZERO_OPERATION_AUDIT),
            "target_final_layer_replay_calls": target_final_layer_replay_calls,
        },
    }
    seal = canonical_sha256(body)
    return Lfm2CpcfV5TerminalActionIntegrationResult(
        final_state_bank=published_bank,
        accepted=accepted,
        report={**body, "result_sha256": seal},
        result_sha256=seal,
    )


__all__ = [
    "LFM2_CPCF_V5_TERMINAL_CAPTURE_SCHEMA",
    "LFM2_CPCF_V5_TERMINAL_EVIDENCE_BUNDLE_SCHEMA",
    "LFM2_CPCF_V5_TERMINAL_INTEGRATION_SCHEMA",
    "Lfm2CpcfV5TerminalActionCapture",
    "Lfm2CpcfV5TerminalActionEvidenceBundle",
    "Lfm2CpcfV5TerminalActionIntegrationError",
    "Lfm2CpcfV5TerminalActionIntegrationResult",
    "build_lfm2_cpcf_v5_terminal_action_evidence",
    "capture_lfm2_cpcf_v5_terminal_action_inputs",
    "compile_install_lfm2_cpcf_v5_terminal_action",
]
