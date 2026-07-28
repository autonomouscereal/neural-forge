import copy

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_tied_causal_interface_fold import (
    TiedCausalInterfaceFold,
    _tensor_sha256 as _interface_tensor_sha256,
)
from neural_forge.lfm2_circuit_balanced_coordinate_recipe import (
    Lfm2CircuitBalancedCoordinateEvidenceError,
    _causal_anchor_tensor_sha256,
    _sealed_interface_report,
)


def _interface(
    *,
    target_width: int,
    schema: str,
    binding_domain: str,
) -> TiedCausalInterfaceFold:
    source_width = 1024
    vocabulary_size = 7
    producer = torch.arange(
        source_width * target_width,
        dtype=torch.float64,
    ).reshape(source_width, target_width)
    producer.mul_(torch.finfo(torch.float64).eps)
    target = torch.arange(
        vocabulary_size * target_width,
        dtype=torch.float64,
    ).reshape(vocabulary_size, target_width)
    target.mul_(1e-6)
    if binding_domain == "base":
        hasher = _interface_tensor_sha256
    elif binding_domain == "anchor":
        hasher = _causal_anchor_tensor_sha256
    else:  # pragma: no cover - test helper contract
        raise AssertionError(binding_domain)
    body = {
        "schema": schema,
        "source_width": source_width,
        "target_width": target_width,
        "vocabulary_size": vocabulary_size,
        "producer_encoder_sha256": hasher(producer),
        "target_weight_sha256": hasher(target),
    }
    return TiedCausalInterfaceFold(
        target_weight=target,
        producer_encoder=producer,
        report={**body, "report_sha256": canonical_sha256(body)},
    )


@pytest.mark.parametrize("target_width", (776, 784))
def test_sealed_interface_accepts_causal_anchor_domain_at_dynamic_widths(
    target_width: int,
) -> None:
    interface = _interface(
        target_width=target_width,
        schema="neural-forge-tied-causal-anchor-residual-v1",
        binding_domain="anchor",
    )

    report = _sealed_interface_report(interface)

    assert report == dict(interface.report)
    assert report["target_width"] == target_width


def test_sealed_interface_preserves_legacy_base_domain_at_h776() -> None:
    interface = _interface(
        target_width=776,
        schema="neural-forge-tied-causal-interface-fold-v1",
        binding_domain="base",
    )

    assert _sealed_interface_report(interface) == dict(interface.report)


@pytest.mark.parametrize(
    ("schema", "binding_domain"),
    (
        ("neural-forge-tied-causal-anchor-residual-v1", "base"),
        ("neural-forge-tied-causal-interface-fold-v1", "anchor"),
    ),
)
def test_known_interface_schema_cannot_switch_tensor_hash_domain(
    schema: str,
    binding_domain: str,
) -> None:
    interface = _interface(
        target_width=784,
        schema=schema,
        binding_domain=binding_domain,
    )

    with pytest.raises(
        Lfm2CircuitBalancedCoordinateEvidenceError,
        match="does not bind its tensors",
    ):
        _sealed_interface_report(interface)


def test_sealed_anchor_interface_rejects_resealed_tensor_substitution() -> None:
    interface = _interface(
        target_width=784,
        schema="neural-forge-tied-causal-anchor-residual-v1",
        binding_domain="anchor",
    )
    forged_target = interface.target_weight.clone()
    forged_target[0, 0] += 1.0
    forged_body = copy.deepcopy(dict(interface.report))
    forged_body.pop("report_sha256")
    forged = TiedCausalInterfaceFold(
        target_weight=forged_target,
        producer_encoder=interface.producer_encoder,
        report={
            **forged_body,
            "report_sha256": canonical_sha256(forged_body),
        },
    )

    with pytest.raises(
        Lfm2CircuitBalancedCoordinateEvidenceError,
        match="does not bind its tensors",
    ):
        _sealed_interface_report(forged)


def test_unknown_interface_extension_must_bind_both_tensors_in_one_domain() -> None:
    base = _interface(
        target_width=784,
        schema="test-extension-interface-v1",
        binding_domain="base",
    )
    mixed_body = copy.deepcopy(dict(base.report))
    mixed_body.pop("report_sha256")
    mixed_body["target_weight_sha256"] = _causal_anchor_tensor_sha256(
        base.target_weight
    )
    mixed = TiedCausalInterfaceFold(
        target_weight=base.target_weight,
        producer_encoder=base.producer_encoder,
        report={
            **mixed_body,
            "report_sha256": canonical_sha256(mixed_body),
        },
    )

    with pytest.raises(
        Lfm2CircuitBalancedCoordinateEvidenceError,
        match="unknown tensor binding domain",
    ):
        _sealed_interface_report(mixed)
