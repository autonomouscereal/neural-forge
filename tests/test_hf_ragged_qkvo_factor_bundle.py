from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from neural_forge.hf_functional_residual_adapter import RedirectedAttentionSite
from neural_forge.hf_qkvo_factor_pack import (
    QKVO_ROLE_SCALE_PROVENANCE_SCHEMA,
    QKVO_SOURCE_PROVENANCE_SCHEMA,
    adapter_state_sha256,
    canonical_sha256,
    tensor_sha256,
)
from neural_forge.hf_ragged_assignment_local_qkvo_residual_adapter import (
    RaggedAssignmentLocalQKVOResidualRegistry,
)
from neural_forge.hf_ragged_qkvo_factor_bundle import (
    RaggedQKVOFactorBundleError,
    install_ragged_qkvo_factor_bundle,
    load_ragged_qkvo_factor_bundle,
    write_ragged_qkvo_factor_bundle,
)


SITES = (
    RedirectedAttentionSite(15, 2, 122, 19, 5, 157),
    RedirectedAttentionSite(19, 4, 156, 21, 6, 174),
)


def _registry(seed: int = 0) -> RaggedAssignmentLocalQKVOResidualRegistry:
    return RaggedAssignmentLocalQKVOResidualRegistry(
        [site.site_pair for site in SITES],
        (4, 8),
        logical_layers=36,
        physical_groups=8,
        hidden_size=2560,
        query_group_width=512,
        head_dim=128,
        dtype=torch.float16,
        device="cpu",
        initialization_seed=seed,
    )


def _attention_registry():
    permutations = torch.arange(8, dtype=torch.int32).repeat(36, 1)
    role_scales = torch.empty(36, 8, 4, dtype=torch.float16)
    role_scales[..., 0] = 1.0
    role_scales[..., 1] = 1.0
    role_scales[..., 2] = -0.0012493133544921875
    role_scales[..., 3] = -0.0005393028259277344
    return SimpleNamespace(
        logical_layers=36,
        key_value_heads=8,
        output_width=2560,
        query_rows_per_group=512,
        head_dim=128,
        prototype_q_weight=torch.empty(1, dtype=torch.float16),
        group_permutations=permutations,
        role_scales=role_scales,
    )


def _role_provenance() -> tuple[dict[str, object], ...]:
    attention = _attention_registry()
    table_sha = tensor_sha256(attention.role_scales)
    result = []
    for site in SITES:
        canonical = int(
            attention.group_permutations[site.layer_index, site.physical_group].item()
        )
        row = attention.role_scales[site.layer_index, canonical].to(torch.float64)
        result.append(
            {
                "schema": QKVO_ROLE_SCALE_PROVENANCE_SCHEMA,
                "q": float(row[0]),
                "k": float(row[1]),
                "v": float(row[2]),
                "o": float(row[3]),
                "role_scales_trainable": False,
                "table_sha256": table_sha,
                "canonical_assignment_axis": canonical,
            }
        )
    return tuple(result)


def _source_provenance() -> dict[str, object]:
    return {
        "schema": QKVO_SOURCE_PROVENANCE_SCHEMA,
        "source_report_file_sha256": "b" * 64,
        "source_report_sha256": "c" * 64,
        "fixture_provenance_sha256": "d" * 64,
        "calibration_evaluation_sha256": "e" * 64,
        "heldout_evaluation_sha256": "f" * 64,
    }


def _write(tmp_path: Path):
    registry = _registry()
    with torch.no_grad():
        for role_index, role in enumerate(registry._ROLES, start=1):
            for site_index, value in enumerate(getattr(registry, f"{role}_left")):
                value.fill_((role_index + site_index) / 1024.0)
    selected_sha = adapter_state_sha256(registry.state_dict())
    directory = tmp_path / "ragged-bundle"
    manifest = write_ragged_qkvo_factor_bundle(
        directory,
        registry,
        selected_registry_state_sha256=selected_sha,
        sites=SITES,
        role_scale_provenance=_role_provenance(),
        source_provenance=_source_provenance(),
    )
    return registry, selected_sha, directory, manifest


def test_ragged_bundle_is_factor_only_and_reconstructs_full_state(
    tmp_path: Path,
) -> None:
    registry, selected_sha, directory, manifest = _write(tmp_path)
    loaded = load_ragged_qkvo_factor_bundle(directory)

    assert manifest["factor_tensor_count"] == 16
    assert manifest["factor_parameter_count"] == 138_240
    assert manifest["factor_tensor_bytes"] == 276_480
    assert manifest["selected_registry_state_sha256"] == selected_sha
    assert manifest["optimizer_tensor_count"] == 0
    assert manifest["parent_tensor_count"] == 0
    assert manifest["base_model_tensor_count"] == 0
    assert manifest["full_model_checkpoint"] is False
    assert len(loaded.site_packs) == 2
    assert sum(path.stat().st_size for path in directory.iterdir()) < 500_000

    fresh = _registry(seed=999)
    assert (
        install_ragged_qkvo_factor_bundle(
            fresh, loaded, attention_registry=_attention_registry()
        )
        == selected_sha
    )
    assert all(
        torch.equal(value, fresh.state_dict()[name])
        for name, value in registry.state_dict().items()
    )


def test_ragged_bundle_rejects_extra_files_and_wrong_fresh_rank_map(
    tmp_path: Path,
) -> None:
    _registry_value, _selected_sha, directory, _manifest = _write(tmp_path)
    extra = directory / "optimizer.pt"
    extra.write_bytes(b"forbidden")
    with pytest.raises(RaggedQKVOFactorBundleError, match="undeclared"):
        load_ragged_qkvo_factor_bundle(directory)
    extra.unlink()

    nested = directory / "optimizer"
    nested.mkdir()
    (nested / "state.pt").write_bytes(b"forbidden")
    with pytest.raises(RaggedQKVOFactorBundleError, match="nested"):
        load_ragged_qkvo_factor_bundle(directory)
    (nested / "state.pt").unlink()
    nested.rmdir()

    loaded = load_ragged_qkvo_factor_bundle(directory)
    wrong = RaggedAssignmentLocalQKVOResidualRegistry(
        [site.site_pair for site in SITES],
        (4, 6),
        logical_layers=36,
        physical_groups=8,
        hidden_size=2560,
        query_group_width=512,
        head_dim=128,
        dtype=torch.float16,
        device="cpu",
    )
    before = {name: value.clone() for name, value in wrong.state_dict().items()}
    with pytest.raises(RaggedQKVOFactorBundleError, match="sites/ranks"):
        install_ragged_qkvo_factor_bundle(
            wrong, loaded, attention_registry=_attention_registry()
        )
    assert all(
        torch.equal(value, wrong.state_dict()[name]) for name, value in before.items()
    )

    fresh = _registry(seed=4)
    wrong_attention = _attention_registry()
    wrong_attention.role_scales[0, 0, 0] = 0.5
    with pytest.raises(RaggedQKVOFactorBundleError, match="geometry"):
        install_ragged_qkvo_factor_bundle(
            fresh, loaded, attention_registry=wrong_attention
        )


def test_ragged_bundle_refuses_overwrite(tmp_path: Path) -> None:
    registry = _registry()
    directory = tmp_path / "existing"
    directory.mkdir()
    with pytest.raises(RaggedQKVOFactorBundleError, match="already exists"):
        write_ragged_qkvo_factor_bundle(
            directory,
            registry,
            selected_registry_state_sha256=adapter_state_sha256(registry.state_dict()),
            sites=SITES,
            role_scale_provenance=_role_provenance(),
            source_provenance=_source_provenance(),
        )


def test_ragged_bundle_rejects_self_consistent_top_manifest_rank_tamper(
    tmp_path: Path,
) -> None:
    _registry_value, _selected_sha, directory, _manifest = _write(tmp_path)
    path = directory / "bundle.manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["site_ranks"][0] = 5
    payload["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RaggedQKVOFactorBundleError, match="disagrees"):
        load_ragged_qkvo_factor_bundle(directory)
