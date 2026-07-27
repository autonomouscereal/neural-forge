from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (  # noqa: E402
    lfm2_cpcf_terminal_action_tensor_sha256,
)
from neural_forge.hf_lfm2_postfold_reclosure_seed_archive import (  # noqa: E402
    LFM2_DEFERRED_FINAL_W2_INSTALLATION_SCHEMA,
    Lfm2PostfoldReclosureSeedArchiveError,
    TENSOR_FILENAME,
    load_lfm2_postfold_reclosure_seed_archive,
    write_lfm2_postfold_reclosure_seed_archive,
)
from neural_forge import hf_lfm2_postfold_reclosure_seed_archive as archive  # noqa: E402
from neural_forge.hf_lfm2_semantic_foundation import (  # noqa: E402
    LFM2_SEMANTIC_FOUNDATION_INTERFACE_SCHEMA,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import (  # noqa: E402
    LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


def _sealed(
    body: dict[str, object],
    field: str = "result_sha256",
) -> MappingProxyType:
    return MappingProxyType({**body, field: canonical_sha256(body)})


def _evidence() -> dict[str, object]:
    generator = torch.Generator().manual_seed(901)
    mapped = torch.randn((7, 4), dtype=torch.float32, generator=generator)
    encoder = torch.randn((4, 3), dtype=torch.float64, generator=generator)
    seed = torch.randn((3, 5), dtype=torch.float32, generator=generator)

    active = _sealed(
        {
            "schema": "synthetic-tied-causal-interface-v1",
            "producer_encoder_sha256": "1" * 64,
        },
        "report_sha256",
    )
    foundation = _sealed(
        {
            "schema": LFM2_SEMANTIC_FOUNDATION_INTERFACE_SCHEMA,
            "active_interface_report": dict(active),
            "active_interface_report_sha256": active["report_sha256"],
            "source_tied_table_sha256": archive._foundation_tensor_sha256(mapped),
            "producer_encoder_sha256": archive._foundation_tensor_sha256(encoder),
        }
    )
    live_seed_terminal_sha256 = lfm2_cpcf_terminal_action_tensor_sha256(seed)
    installed_prior_sha256 = "2" * 64
    prior_authentication_sha256 = "3" * 64
    stack_model_state_sha256 = "4" * 64
    stack = _sealed(
        {
            "schema": LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
            "tied_interface_lineage": {
                "interface_report_sha256": active["report_sha256"],
                "producer_encoder_sha256": active["producer_encoder_sha256"],
            },
            "deferred_final_w2_sha256": lfm2_authenticated_tensor_sha256(seed),
            "final_layer_mlp_down_weight_prior_live_w2_sha256": (
                live_seed_terminal_sha256
            ),
            "final_layer_mlp_down_weight_prior_tensor_sha256": (
                installed_prior_sha256
            ),
            "final_layer_mlp_down_weight_prior_authentication_sha256": (
                prior_authentication_sha256
            ),
            "model_state_after_sha256": stack_model_state_sha256,
        }
    )
    installation = _sealed(
        {
            "schema": LFM2_DEFERRED_FINAL_W2_INSTALLATION_SCHEMA,
            "semantic_layer_stack_result_sha256": stack["result_sha256"],
            "prior_authentication_sha256": prior_authentication_sha256,
            "prior_tensor_sha256": installed_prior_sha256,
            "live_w2_sha256_before": live_seed_terminal_sha256,
            "live_w2_sha256_after": installed_prior_sha256,
            "model_state_before_sha256": stack_model_state_sha256,
            "model_state_after_sha256": "5" * 64,
        }
    )
    return {
        "mapped": mapped,
        "encoder": encoder,
        "seed": seed,
        "foundation": foundation,
        "stack": stack,
        "installation": installation,
    }


def _write(root: Path, values: dict[str, object]):
    return write_lfm2_postfold_reclosure_seed_archive(
        root,
        mapped_parent_tied_table=values["mapped"],
        producer_encoder=values["encoder"],
        deferred_final_w2_seed=values["seed"],
        foundation_interface_solve_receipt=values["foundation"],
        semantic_layer_stack_receipt=values["stack"],
        deferred_final_w2_installation_receipt=values["installation"],
    )


def _load(root: Path, values: dict[str, object], manifest_sha256: str):
    return load_lfm2_postfold_reclosure_seed_archive(
        root,
        foundation_interface_solve_receipt=values["foundation"],
        semantic_layer_stack_receipt=values["stack"],
        deferred_final_w2_installation_receipt=values["installation"],
        expected_manifest_sha256=manifest_sha256,
    )


def test_archive_roundtrip_freshly_loads_exact_reclosure_seeds(tmp_path) -> None:
    values = _evidence()
    root = tmp_path / "postfold-reclosure-seeds"

    manifest = _write(root, values)
    loaded, authenticated = _load(root, values, manifest["manifest_sha256"])

    assert authenticated["fresh_reload_verified"] is True
    assert authenticated["foundation_interface_solve_result_sha256"] == (
        values["foundation"]["result_sha256"]
    )
    assert authenticated["semantic_layer_stack_result_sha256"] == (
        values["stack"]["result_sha256"]
    )
    assert authenticated["deferred_final_w2_installation_result_sha256"] == (
        values["installation"]["result_sha256"]
    )
    assert torch.equal(loaded.mapped_parent_tied_table, values["mapped"])
    assert torch.equal(loaded.contextual_parent_rows, values["mapped"])
    assert torch.equal(loaded.producer_encoder, values["encoder"])
    assert torch.equal(loaded.deferred_final_w2_seed, values["seed"])


def test_archive_rejects_changed_tensor_payload(tmp_path) -> None:
    values = _evidence()
    root = tmp_path / "postfold-reclosure-seeds"
    manifest = _write(root, values)
    tensor_path = root / TENSOR_FILENAME
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)

    with pytest.raises(
        Lfm2PostfoldReclosureSeedArchiveError,
        match="manifest, ancestry, or tensor file changed",
    ):
        _load(root, values, manifest["manifest_sha256"])


def test_archive_requires_independent_manifest_digest_pin(tmp_path) -> None:
    values = _evidence()
    root = tmp_path / "postfold-reclosure-seeds"
    _write(root, values)

    with pytest.raises(
        Lfm2PostfoldReclosureSeedArchiveError,
        match="manifest, ancestry, or tensor file changed",
    ):
        _load(root, values, "0" * 64)


def test_archive_rejects_valid_but_different_foundation_ancestry(tmp_path) -> None:
    values = _evidence()
    root = tmp_path / "postfold-reclosure-seeds"
    manifest = _write(root, values)
    different = dict(values)
    changed_foundation = dict(values["foundation"])
    changed_foundation.pop("result_sha256")
    changed_foundation["source_tied_table_sha256"] = "f" * 64
    different["foundation"] = _sealed(changed_foundation)

    with pytest.raises(
        Lfm2PostfoldReclosureSeedArchiveError,
        match="do not match their foundation",
    ):
        _load(root, different, manifest["manifest_sha256"])


@pytest.mark.parametrize("receipt_name", ("foundation", "stack", "installation"))
def test_archive_is_bound_to_each_complete_ancestry_receipt(
    tmp_path,
    receipt_name: str,
) -> None:
    values = _evidence()
    root = tmp_path / "postfold-reclosure-seeds"
    manifest = _write(root, values)
    different = dict(values)
    changed_receipt = dict(values[receipt_name])
    changed_receipt.pop("result_sha256")
    changed_receipt["valid_but_different_receipt_nonce"] = receipt_name
    different[receipt_name] = _sealed(changed_receipt)

    with pytest.raises(Lfm2PostfoldReclosureSeedArchiveError):
        _load(root, different, manifest["manifest_sha256"])


def test_archive_refuses_to_replace_an_existing_publication(tmp_path) -> None:
    values = _evidence()
    root = tmp_path / "postfold-reclosure-seeds"
    _write(root, values)

    with pytest.raises(
        Lfm2PostfoldReclosureSeedArchiveError,
        match="destination already exists",
    ):
        _write(root, values)
