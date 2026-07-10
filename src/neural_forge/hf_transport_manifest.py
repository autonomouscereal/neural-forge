"""Fail-closed target-role manifests for supported decoder layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neural_forge.hf_arch_layout import (
    DecoderLayout,
    build_decoder_layout,
    resolve_module,
)
from neural_forge.hf_transport import (
    Disposition,
    PlanValidationError,
    RequiredTargetRole,
    TransportPlan,
)


@dataclass(frozen=True, slots=True)
class DecoderTransportManifest:
    layout: DecoderLayout
    plan: TransportPlan
    parameter_tensors: tuple[str, ...]
    buffer_tensors: tuple[str, ...]
    claimed_tensors: tuple[str, ...]

    @property
    def unclaimed_parameter_tensors(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.parameter_tensors) - set(self.claimed_tensors)))

    @property
    def unclaimed_buffer_tensors(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.buffer_tensors) - set(self.claimed_tensors)))


def build_decoder_transport_manifest(
    model: Any,
    *,
    layout: DecoderLayout | None = None,
    plan_name: str | None = None,
    target_prefix: str = "model",
) -> DecoderTransportManifest:
    """Assign every supported decoder parameter to an explicit semantic role."""

    resolved_layout = layout or build_decoder_layout(model)
    prefix = _validate_prefix(target_prefix)
    roles: list[RequiredTargetRole] = []

    def add_tensor_role(
        name: str,
        module_path: str,
        *,
        description: str,
        allowed: tuple[Disposition, ...] = (
            Disposition.TRANSPORTED,
            Disposition.CLOSED_FORM_FITTED,
        ),
    ) -> None:
        tensors = _direct_module_tensors(model, module_path, prefix=prefix)
        if not tensors:
            raise PlanValidationError(
                f"Required weighted operator {module_path!r} exposes no direct tensors."
            )
        roles.append(
            RequiredTargetRole(
                name=name,
                target_tensors=tensors,
                allowed_dispositions=allowed,
                description=description,
            )
        )

    def add_native_role(
        name: str,
        *,
        description: str,
        module_paths: tuple[str, ...] = (),
    ) -> None:
        tensors = tuple(
            sorted(
                {
                    tensor
                    for module_path in module_paths
                    for tensor in _direct_module_tensors(
                        model,
                        module_path,
                        prefix=prefix,
                    )
                }
            )
        )
        roles.append(
            RequiredTargetRole(
                name=name,
                target_tensors=tensors,
                allowed_dispositions=(Disposition.TARGET_NATIVE,),
                description=description,
            )
        )

    add_native_role(
        "architecture.topology",
        description=(
            f"Validated {resolved_layout.family} serial pre-norm decoder topology."
        ),
    )
    add_tensor_role(
        "embeddings.token",
        resolved_layout.embeddings.token_path,
        description="Input token embedding rows.",
    )
    if resolved_layout.position.kind == "learned_absolute":
        assert resolved_layout.embeddings.position_path is not None
        add_tensor_role(
            "position.learned_absolute",
            resolved_layout.embeddings.position_path,
            description="Learned absolute position embedding rows.",
        )
    else:
        add_native_role(
            "position.rope",
            description="Deterministic RoPE operator and its derived buffers.",
            module_paths=resolved_layout.position.operator_paths,
        )
    if resolved_layout.embeddings.dropout_path is not None:
        add_native_role(
            "embeddings.dropout",
            description="Validated target embedding dropout policy.",
            module_paths=(resolved_layout.embeddings.dropout_path,),
        )

    for block in resolved_layout.blocks:
        base = f"block.{block.index}"
        add_native_role(
            f"{base}.residual",
            description="Validated serial pre-norm residual wiring.",
        )
        add_tensor_role(
            f"{base}.attention.norm",
            block.attention_input_norm.path,
            description="Pre-attention normalization parameters.",
        )
        attention = block.attention
        if attention.fused_qkv_path is not None:
            add_tensor_role(
                f"{base}.attention.qkv",
                attention.fused_qkv_path,
                description="Fused query/key/value projection.",
            )
        else:
            assert attention.query_path is not None
            assert attention.key_path is not None
            assert attention.value_path is not None
            for label, path in (
                ("query", attention.query_path),
                ("key", attention.key_path),
                ("value", attention.value_path),
            ):
                add_tensor_role(
                    f"{base}.attention.{label}",
                    path,
                    description=f"{label.title()} projection.",
                )
        add_tensor_role(
            f"{base}.attention.output",
            attention.output_path,
            description="Attention output projection.",
        )
        add_native_role(
            f"{base}.attention.policy",
            description="Validated head grouping, masking, and dropout policy.",
            module_paths=(attention.module_path,),
        )
        add_tensor_role(
            f"{base}.mlp.norm",
            block.mlp_input_norm.path,
            description="Pre-MLP normalization parameters.",
        )
        mlp = block.mlp
        if mlp.input_projection_path is not None:
            add_tensor_role(
                f"{base}.mlp.input",
                mlp.input_projection_path,
                description="Dense MLP input projection.",
            )
        if mlp.gate_projection_path is not None:
            add_tensor_role(
                f"{base}.mlp.gate",
                mlp.gate_projection_path,
                description="Gated MLP gate projection.",
            )
        if mlp.up_projection_path is not None:
            add_tensor_role(
                f"{base}.mlp.up",
                mlp.up_projection_path,
                description="Gated MLP value/up projection.",
            )
        add_tensor_role(
            f"{base}.mlp.down",
            mlp.down_projection_path,
            description="MLP output/down projection.",
        )
        add_native_role(
            f"{base}.mlp.activation",
            description=f"Validated {mlp.activation_name} activation operator.",
            module_paths=(mlp.activation_path,),
        )

    add_tensor_role(
        "decoder.final_norm",
        resolved_layout.final_norm.path,
        description="Final decoder normalization parameters.",
    )
    add_tensor_role(
        "lm_head",
        resolved_layout.lm_head_path,
        description="Output vocabulary head.",
    )
    add_native_role(
        "embeddings.tie_policy",
        description=(
            "Validated tied input/output weights."
            if resolved_layout.tie_word_embeddings
            else "Validated independent input/output weights."
        ),
    )

    parameter_tensors = _named_tensor_inventory(
        model,
        prefix=prefix,
        kind="parameter",
    )
    buffer_tensors = _named_tensor_inventory(model, prefix=prefix, kind="buffer")
    claimed = tuple(
        sorted({tensor for role in roles for tensor in role.target_tensors})
    )
    missing_parameters = tuple(sorted(set(parameter_tensors) - set(claimed)))
    if missing_parameters:
        raise PlanValidationError(
            "Supported decoder layout left trainable/state parameters unclaimed: "
            + ", ".join(missing_parameters)
        )

    unclaimed_buffers = tuple(sorted(set(buffer_tensors) - set(claimed)))
    if unclaimed_buffers:
        roles.append(
            RequiredTargetRole(
                name="runtime.derived_buffers",
                target_tensors=unclaimed_buffers,
                allowed_dispositions=(Disposition.TARGET_NATIVE,),
                description="Validated architecture-derived non-parameter buffers.",
            )
        )
        claimed = tuple(sorted(set(claimed) | set(unclaimed_buffers)))

    return DecoderTransportManifest(
        layout=resolved_layout,
        plan=TransportPlan(
            name=plan_name or f"{resolved_layout.family}-decoder-complete-transport",
            required_roles=tuple(roles),
        ),
        parameter_tensors=parameter_tensors,
        buffer_tensors=buffer_tensors,
        claimed_tensors=claimed,
    )


def _validate_prefix(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PlanValidationError("target_prefix must be a non-empty trimmed string.")
    return value.rstrip(".")


def _direct_module_tensors(
    model: Any,
    module_path: str,
    *,
    prefix: str,
) -> tuple[str, ...]:
    module = resolve_module(model, module_path)
    tensors = []
    for name, tensor in module.named_parameters(recurse=False):
        if tensor is not None:
            tensors.append(f"{prefix}.{module_path}.{name}")
    for name, tensor in module.named_buffers(recurse=False):
        if tensor is not None:
            tensors.append(f"{prefix}.{module_path}.{name}")
    return tuple(sorted(set(tensors)))


def _named_tensor_inventory(
    model: Any,
    *,
    prefix: str,
    kind: str,
) -> tuple[str, ...]:
    if kind == "parameter":
        iterator = model.named_parameters(remove_duplicate=False)
    elif kind == "buffer":
        iterator = model.named_buffers(remove_duplicate=False)
    else:  # pragma: no cover - private call contract
        raise AssertionError(f"Unknown tensor inventory kind: {kind}")
    return tuple(sorted(f"{prefix}.{name}" for name, _tensor in iterator))


__all__ = [
    "DecoderTransportManifest",
    "build_decoder_transport_manifest",
]
