from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import neural_forge.hf_qkvo_factor_pack as factor_pack_module

from neural_forge.hf_assignment_local_qkvo_residual_adapter import (
    SparseAssignmentLocalQKVOResidualRegistry,
)
from neural_forge.hf_qkvo_factor_pack import (
    MAX_QKVO_FACTOR_FILE_BYTES,
    QKVO_FACTOR_KEYS,
    QKVO_ROLE_SCALE_PROVENANCE_SCHEMA,
    QKVO_SOURCE_PROVENANCE_SCHEMA,
    QKVOFactorPackError,
    adapter_state_sha256,
    install_qkvo_factor_pack,
    load_qkvo_factor_pack,
    write_qkvo_factor_pack,
)


def _role_provenance() -> dict[str, object]:
    return {
        "schema": QKVO_ROLE_SCALE_PROVENANCE_SCHEMA,
        "q": 1.0,
        "k": 1.0,
        "v": -0.0012493133544921875,
        "o": -0.0005393028259277344,
        "role_scales_trainable": False,
        "table_sha256": "a" * 64,
    }


def _source_provenance() -> dict[str, object]:
    return {
        "schema": QKVO_SOURCE_PROVENANCE_SCHEMA,
        "source_report_file_sha256": "b" * 64,
        "source_report_sha256": "c" * 64,
        "fixture_provenance_sha256": "d" * 64,
        "calibration_evaluation_sha256": "e" * 64,
        "heldout_evaluation_sha256": "f" * 64,
    }


def _selected_state() -> dict[str, torch.Tensor]:
    registry = _registry([[19, 4]])
    with torch.no_grad():
        for index, role in enumerate(("q", "k", "v", "o"), start=1):
            getattr(registry, f"{role}_left").fill_(index / 1024.0)
    return {
        name: value.detach().clone() for name, value in registry.state_dict().items()
    }


def _registry(sites) -> SparseAssignmentLocalQKVOResidualRegistry:
    return SparseAssignmentLocalQKVOResidualRegistry(
        sites,
        logical_layers=36,
        physical_groups=8,
        hidden_size=2560,
        query_group_width=512,
        head_dim=128,
        rank=8,
        dtype=torch.float16,
        device="cpu",
        initialization_seed=0,
    )


def _write(tmp_path: Path):
    state = _selected_state()
    state_hash = adapter_state_sha256(state)
    factor_path = tmp_path / "candidate.safetensors"
    manifest_path = tmp_path / "candidate.manifest.json"
    manifest = write_qkvo_factor_pack(
        factor_path,
        manifest_path,
        state,
        selected_adapter_state_sha256=state_hash,
        site={"layer_index": 19, "physical_group": 4},
        rank=8,
        registry_geometry={"logical_layers": 36, "physical_groups": 8},
        role_scale_provenance=_role_provenance(),
        provenance=_source_provenance(),
    )
    return state, state_hash, factor_path, manifest_path, manifest


def test_factor_pack_contains_only_eight_factors_and_exact_manifest(
    tmp_path: Path,
) -> None:
    state, state_hash, factor_path, manifest_path, manifest = _write(tmp_path)
    loaded = load_qkvo_factor_pack(factor_path, manifest_path)

    assert tuple(loaded.factors) == QKVO_FACTOR_KEYS
    assert manifest["factor_parameter_count"] == 92_160
    assert manifest["factor_tensor_bytes"] == 184_320
    assert manifest["selected_adapter_state_sha256"] == state_hash
    assert manifest["optimizer_tensor_count"] == 0
    assert manifest["parent_tensor_count"] == 0
    assert manifest["base_model_tensor_count"] == 0
    assert manifest["full_model_checkpoint"] is False
    assert factor_path.stat().st_size < 200_000
    assert manifest_path.stat().st_size < 10_000
    for name in QKVO_FACTOR_KEYS:
        assert torch.equal(loaded.factors[name], state[name])

    fresh = _registry([[19, 4]])
    assert install_qkvo_factor_pack(fresh, loaded) == state_hash
    assert adapter_state_sha256(fresh.state_dict()) == state_hash

    wrong_site = _registry([[18, 4]])
    before = {
        name: value.detach().clone() for name, value in wrong_site.state_dict().items()
    }
    with pytest.raises(QKVOFactorPackError, match="geometry|immutable buffer"):
        install_qkvo_factor_pack(wrong_site, loaded)
    for name, value in wrong_site.state_dict().items():
        assert torch.equal(value, before[name])


def test_factor_pack_rejects_manifest_or_payload_tamper(tmp_path: Path) -> None:
    _state, _state_hash, factor_path, manifest_path, _manifest = _write(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["rank"] = 4
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(QKVOFactorPackError, match="Manifest SHA"):
        load_qkvo_factor_pack(factor_path, manifest_path)

    # Restore the valid manifest, then alter one byte in the safetensors file.
    manifest_path.unlink()
    factor_path.unlink()
    _state, _state_hash, factor_path, manifest_path, _manifest = _write(tmp_path)
    raw = bytearray(factor_path.read_bytes())
    raw[-1] ^= 1
    factor_path.write_bytes(raw)
    with pytest.raises(QKVOFactorPackError, match="content contract"):
        load_qkvo_factor_pack(factor_path, manifest_path)


def test_factor_pack_is_fail_closed_against_overwrite_or_wrong_state_hash(
    tmp_path: Path,
) -> None:
    state, state_hash, factor_path, manifest_path, _manifest = _write(tmp_path)
    with pytest.raises(QKVOFactorPackError, match="already exist"):
        write_qkvo_factor_pack(
            factor_path,
            manifest_path,
            state,
            selected_adapter_state_sha256=state_hash,
            site={"layer_index": 19, "physical_group": 4},
            rank=8,
            registry_geometry={"logical_layers": 36, "physical_groups": 8},
            role_scale_provenance=_role_provenance(),
            provenance=_source_provenance(),
        )

    other_factor = tmp_path / "other.safetensors"
    other_manifest = tmp_path / "other.json"
    with pytest.raises(QKVOFactorPackError, match="selected state"):
        write_qkvo_factor_pack(
            other_factor,
            other_manifest,
            state,
            selected_adapter_state_sha256="0" * 64,
            site={"layer_index": 19, "physical_group": 4},
            rank=8,
            registry_geometry={"logical_layers": 36, "physical_groups": 8},
            role_scale_provenance=_role_provenance(),
            provenance=_source_provenance(),
        )
    assert not other_factor.exists()
    assert not other_manifest.exists()


def test_writer_rejects_unstored_state_and_cleans_orphan_on_manifest_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _selected_state()
    with_extra = {**state, "unstored_parent_tensor": torch.zeros(1)}
    factor_path = tmp_path / "extra.safetensors"
    manifest_path = tmp_path / "extra.json"
    with pytest.raises(QKVOFactorPackError, match="exactly eight factors"):
        write_qkvo_factor_pack(
            factor_path,
            manifest_path,
            with_extra,
            selected_adapter_state_sha256=adapter_state_sha256(with_extra),
            site={"layer_index": 19, "physical_group": 4},
            rank=8,
            registry_geometry={"logical_layers": 36, "physical_groups": 8},
            role_scale_provenance=_role_provenance(),
            provenance=_source_provenance(),
        )

    def fail_manifest(*_args, **_kwargs):
        raise OSError("injected manifest failure")

    monkeypatch.setattr(factor_pack_module, "_atomic_write_json", fail_manifest)
    with pytest.raises(OSError, match="injected manifest failure"):
        write_qkvo_factor_pack(
            factor_path,
            manifest_path,
            state,
            selected_adapter_state_sha256=adapter_state_sha256(state),
            site={"layer_index": 19, "physical_group": 4},
            rank=8,
            registry_geometry={"logical_layers": 36, "physical_groups": 8},
            role_scale_provenance=_role_provenance(),
            provenance=_source_provenance(),
        )
    assert not factor_path.exists()
    assert not manifest_path.exists()


def test_writer_rejects_float_integer_nonfinite_and_cross_parent_contracts(
    tmp_path: Path,
) -> None:
    state = _selected_state()
    state_hash = adapter_state_sha256(state)
    common = {
        "adapter_state": state,
        "selected_adapter_state_sha256": state_hash,
        "site": {"layer_index": 19, "physical_group": 4},
        "registry_geometry": {"logical_layers": 36, "physical_groups": 8},
        "role_scale_provenance": _role_provenance(),
        "provenance": _source_provenance(),
    }
    with pytest.raises(QKVOFactorPackError, match="exact integer"):
        write_qkvo_factor_pack(
            tmp_path / "float.safetensors",
            tmp_path / "float.json",
            rank=8.0,
            **common,
        )
    nonfinite = {name: value.clone() for name, value in state.items()}
    nonfinite["q_left"].reshape(-1)[0] = float("nan")
    with pytest.raises(QKVOFactorPackError, match="finite"):
        write_qkvo_factor_pack(
            tmp_path / "nan.safetensors",
            tmp_path / "nan.json",
            nonfinite,
            selected_adapter_state_sha256=adapter_state_sha256(nonfinite),
            site=common["site"],
            rank=8,
            registry_geometry=common["registry_geometry"],
            role_scale_provenance=common["role_scale_provenance"],
            provenance=common["provenance"],
        )
    other = tmp_path / "other"
    with pytest.raises(QKVOFactorPackError, match="share one parent"):
        write_qkvo_factor_pack(
            tmp_path / "cross.safetensors",
            other / "cross.json",
            rank=8,
            **common,
        )
    string_roles = _role_provenance()
    string_roles["q"] = "1.0"
    with pytest.raises(QKVOFactorPackError, match="finite numbers"):
        write_qkvo_factor_pack(
            tmp_path / "string-role.safetensors",
            tmp_path / "string-role.json",
            state,
            selected_adapter_state_sha256=state_hash,
            site=common["site"],
            rank=8,
            registry_geometry=common["registry_geometry"],
            role_scale_provenance=string_roles,
            provenance=common["provenance"],
        )


def test_loader_rejects_hard_byte_cap_before_parse(tmp_path: Path) -> None:
    factor_path = tmp_path / "huge.safetensors"
    manifest_path = tmp_path / "huge.json"
    factor_path.write_bytes(b"0" * (MAX_QKVO_FACTOR_FILE_BYTES + 1))
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(QKVOFactorPackError, match="hard byte"):
        load_qkvo_factor_pack(factor_path, manifest_path)


def test_loader_rejects_nonfloating_or_malformed_factors_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _state, _state_hash, factor_path, manifest_path, _manifest = _write(tmp_path)
    valid = factor_pack_module.load_file(str(factor_path), device="cpu")
    forged_int = {name: value.to(torch.int16) for name, value in valid.items()}
    monkeypatch.setattr(
        factor_pack_module,
        "load_file",
        lambda *_args, **_kwargs: forged_int,
    )
    with pytest.raises(QKVOFactorPackError):
        load_qkvo_factor_pack(factor_path, manifest_path)

    malformed = dict(valid)
    malformed["q_left"] = malformed["q_left"].reshape(-1)
    monkeypatch.setattr(
        factor_pack_module,
        "load_file",
        lambda *_args, **_kwargs: malformed,
    )
    with pytest.raises(QKVOFactorPackError):
        load_qkvo_factor_pack(factor_path, manifest_path)

    state = _selected_state()
    with pytest.raises(QKVOFactorPackError, match="rank-three floating"):
        factor_pack_module._canonical_registry_contract(
            state,
            malformed,
            site={"layer_index": 19, "physical_group": 4},
            rank=8,
            registry_geometry={"logical_layers": 36, "physical_groups": 8},
        )
