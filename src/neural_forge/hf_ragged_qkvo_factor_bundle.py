"""Factor-only bundles for ragged multi-site Q/K/V/O residual registries.

Each site is stored as one already-verified eight-factor safetensors pack. A
small bundle manifest binds their order, ranks, immutable ragged buffers, and
the complete selected ragged registry state hash. No optimizer, parent, base
model, or full checkpoint tensor is accepted.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neural_forge.hf_functional_residual_adapter import RedirectedAttentionSite
from neural_forge.hf_qkvo_factor_pack import (
    LoadedQKVOFactorPack,
    QKVO_FACTOR_KEYS,
    adapter_state_sha256,
    canonical_sha256,
    file_sha256,
    load_qkvo_factor_pack,
    tensor_sha256,
    write_qkvo_factor_pack,
)
from neural_forge.hf_ragged_assignment_local_qkvo_residual_adapter import (
    RaggedAssignmentLocalQKVOResidualRegistry,
)


RAGGED_QKVO_FACTOR_BUNDLE_SCHEMA = "neural-forge-ragged-qkvo-factor-bundle-v1"
MAX_RAGGED_QKVO_BUNDLE_BYTES = 2 * 1024 * 1024
MAX_RAGGED_QKVO_BUNDLE_MANIFEST_BYTES = 256 * 1024


class RaggedQKVOFactorBundleError(RuntimeError):
    """A multi-site factor-only bundle failed its content contract."""


@dataclass(frozen=True, slots=True)
class LoadedRaggedQKVOFactorBundle:
    directory: Path
    manifest: Mapping[str, Any]
    site_packs: tuple[LoadedQKVOFactorPack, ...]


def _site_state(
    registry: RaggedAssignmentLocalQKVOResidualRegistry,
    site_index: int,
) -> dict[str, Tensor]:
    result = {
        name: getattr(registry, name.split("_")[0] + "_" + name.split("_")[1])[
            site_index
        ]
        .detach()
        .cpu()
        .unsqueeze(0)
        .contiguous()
        .clone()
        for name in QKVO_FACTOR_KEYS
    }
    result.update(
        {
            "site_indices": registry.site_indices[site_index : site_index + 1]
            .detach()
            .cpu()
            .clone(),
            "_registry_schema_version": torch.tensor(1, dtype=torch.int32),
            "_registry_layout": torch.tensor(1, dtype=torch.int32),
        }
    )
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RaggedQKVOFactorBundleError(
                f"Bundle JSON already exists: {path.name}."
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _synthesized_ragged_state(
    site_packs: Sequence[LoadedQKVOFactorPack],
    manifest: Mapping[str, Any],
) -> dict[str, Tensor]:
    try:
        site_indices = torch.tensor(manifest["site_indices"], dtype=torch.int32)
        site_ranks = torch.tensor(manifest["site_ranks"], dtype=torch.int32)
        schema = torch.tensor(manifest["registry_schema_version"], dtype=torch.int32)
        layout = torch.tensor(manifest["registry_layout"], dtype=torch.int32)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise RaggedQKVOFactorBundleError(
            "Bundle immutable registry state is malformed."
        ) from error
    state: dict[str, Tensor] = {
        "site_indices": site_indices,
        "site_ranks": site_ranks,
        "_registry_schema_version": schema,
        "_registry_layout": layout,
    }
    for site_index, pack in enumerate(site_packs):
        for name in QKVO_FACTOR_KEYS:
            value = pack.factors[name]
            if int(value.shape[0]) != 1:
                raise RaggedQKVOFactorBundleError(
                    "A site factor pack is not singleton-site."
                )
            state[f"{name}.{site_index}"] = value.squeeze(0).contiguous()
    return state


def write_ragged_qkvo_factor_bundle(
    directory: Path,
    registry: RaggedAssignmentLocalQKVOResidualRegistry,
    *,
    selected_registry_state_sha256: str,
    sites: Sequence[RedirectedAttentionSite],
    role_scale_provenance: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a new immutable directory containing site packs plus one manifest."""

    directory = directory.resolve()
    if directory.exists():
        raise RaggedQKVOFactorBundleError(
            "Ragged factor bundle directory already exists."
        )
    try:
        directory.mkdir(parents=False, exist_ok=False)
    except (FileExistsError, FileNotFoundError) as error:
        raise RaggedQKVOFactorBundleError(
            "Bundle needs one absent directory under an existing parent."
        ) from error
    success = False
    owned_file_hashes: dict[Path, str] = {}
    try:
        if any(
            name in source_provenance
            for name in (
                "ragged_selected_registry_state_sha256",
                "ragged_site_index",
            )
        ):
            raise RaggedQKVOFactorBundleError(
                "Source provenance uses bundle-reserved keys."
            )
        site_values = tuple(sites)
        if isinstance(role_scale_provenance, Mapping):
            role_scale_values = (dict(role_scale_provenance),) * registry.site_count
        else:
            role_scale_values = tuple(dict(value) for value in role_scale_provenance)
        if (
            len(site_values) != registry.site_count
            or len(role_scale_values) != registry.site_count
            or [list(value.site_pair) for value in site_values]
            != registry.site_indices.detach().cpu().tolist()
            or max(registry.ranks) > 8
        ):
            raise RaggedQKVOFactorBundleError(
                "Bundle sites/ranks differ from the live bounded registry."
            )
        actual_state_sha = adapter_state_sha256(registry.state_dict())
        if actual_state_sha != selected_registry_state_sha256:
            raise RaggedQKVOFactorBundleError(
                "Live ragged registry differs from the selected state hash."
            )
        site_rows: list[dict[str, Any]] = []
        loaded_packs: list[LoadedQKVOFactorPack] = []
        for index, (site, rank) in enumerate(
            zip(site_values, registry.ranks, strict=True)
        ):
            state = _site_state(registry, index)
            state_sha = adapter_state_sha256(state)
            factor_path = directory / f"site_{index:02d}.safetensors"
            manifest_path = directory / f"site_{index:02d}.manifest.json"
            source = {
                **dict(source_provenance),
                "ragged_selected_registry_state_sha256": actual_state_sha,
                "ragged_site_index": index,
            }
            site_manifest = write_qkvo_factor_pack(
                factor_path,
                manifest_path,
                state,
                selected_adapter_state_sha256=state_sha,
                site=site.as_dict(),
                rank=rank,
                registry_geometry={
                    "logical_layers": registry.logical_layers,
                    "physical_groups": registry.physical_groups,
                    "hidden_size": registry.hidden_size,
                    "query_group_width": registry.query_group_width,
                    "head_dim": registry.head_dim,
                    "rank": rank,
                    "site_count": 1,
                    "factor_dtype": str(registry.q_left[index].dtype),
                },
                role_scale_provenance=role_scale_values[index],
                provenance=source,
                expected_factor_parameter_count=11_520 * rank,
            )
            owned_file_hashes[factor_path] = site_manifest["factor_file_sha256"]
            owned_file_hashes[manifest_path] = file_sha256(manifest_path)
            loaded = load_qkvo_factor_pack(factor_path, manifest_path)
            loaded_packs.append(loaded)
            site_rows.append(
                {
                    "site_index": index,
                    "site": site.as_dict(),
                    "rank": rank,
                    "factor_file": factor_path.name,
                    "factor_file_sha256": site_manifest["factor_file_sha256"],
                    "manifest_file": manifest_path.name,
                    "manifest_file_sha256": file_sha256(manifest_path),
                    "site_state_sha256": state_sha,
                    "factor_parameter_count": site_manifest["factor_parameter_count"],
                }
            )
        manifest: dict[str, Any] = {
            "schema": RAGGED_QKVO_FACTOR_BUNDLE_SCHEMA,
            "selected_registry_state_sha256": actual_state_sha,
            "site_indices": registry.site_indices.detach().cpu().tolist(),
            "site_ranks": list(registry.ranks),
            "registry_schema_version": 1,
            "registry_layout": 2,
            "geometry": {
                "logical_layers": registry.logical_layers,
                "physical_groups": registry.physical_groups,
                "hidden_size": registry.hidden_size,
                "query_group_width": registry.query_group_width,
                "head_dim": registry.head_dim,
                "factor_dtype": str(registry.q_left[0].dtype),
            },
            "sites": site_rows,
            "site_count": registry.site_count,
            "factor_tensor_count": 8 * registry.site_count,
            "factor_parameter_count": registry.learned_parameter_count,
            "factor_tensor_bytes": sum(
                int(value.numel() * value.element_size())
                for value in registry.parameters()
            ),
            "role_scale_provenance_by_site": list(role_scale_values),
            "source_provenance": dict(source_provenance),
            "optimizer_tensor_count": 0,
            "parent_tensor_count": 0,
            "base_model_tensor_count": 0,
            "full_model_checkpoint": False,
            "factor_bundle_only": True,
        }
        synthesized = _synthesized_ragged_state(loaded_packs, manifest)
        if adapter_state_sha256(synthesized) != actual_state_sha:
            raise RaggedQKVOFactorBundleError(
                "Site packs do not reconstruct the selected ragged state."
            )
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        manifest_path = directory / "bundle.manifest.json"
        _atomic_json(manifest_path, manifest)
        owned_file_hashes[manifest_path] = file_sha256(manifest_path)
        total_bytes = sum(
            value.stat().st_size for value in directory.iterdir() if value.is_file()
        )
        if total_bytes > MAX_RAGGED_QKVO_BUNDLE_BYTES:
            raise RaggedQKVOFactorBundleError(
                "Ragged factor bundle exceeds the hard byte cap."
            )
        loaded_bundle = load_ragged_qkvo_factor_bundle(directory)
        if loaded_bundle.manifest["manifest_sha256"] != manifest["manifest_sha256"]:
            raise RaggedQKVOFactorBundleError("Written bundle failed exact reload.")
        success = True
        return manifest
    finally:
        if not success and directory.exists():
            for path, digest in owned_file_hashes.items():
                try:
                    if path.is_file() and file_sha256(path) == digest:
                        path.unlink()
                except OSError:
                    pass
            try:
                directory.rmdir()
            except OSError:
                pass


def load_ragged_qkvo_factor_bundle(
    directory: Path,
) -> LoadedRaggedQKVOFactorBundle:
    directory = directory.resolve()
    manifest_path = directory / "bundle.manifest.json"
    if (
        not directory.is_dir()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size > MAX_RAGGED_QKVO_BUNDLE_MANIFEST_BYTES
    ):
        raise RaggedQKVOFactorBundleError("Bundle directory/manifest is invalid.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RaggedQKVOFactorBundleError("Bundle manifest is unreadable.") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != RAGGED_QKVO_FACTOR_BUNDLE_SCHEMA
        or manifest.get("manifest_sha256")
        != canonical_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        or manifest.get("factor_bundle_only") is not True
        or manifest.get("full_model_checkpoint") is not False
        or manifest.get("optimizer_tensor_count") != 0
        or manifest.get("parent_tensor_count") != 0
        or manifest.get("base_model_tensor_count") != 0
    ):
        raise RaggedQKVOFactorBundleError("Bundle manifest contract failed.")
    site_rows = manifest.get("sites")
    site_count = manifest.get("site_count")
    site_indices = manifest.get("site_indices")
    site_ranks = manifest.get("site_ranks")
    geometry = manifest.get("geometry")
    role_scale_values = manifest.get("role_scale_provenance_by_site")
    source_provenance = manifest.get("source_provenance")
    if (
        not isinstance(site_rows, list)
        or not isinstance(site_count, int)
        or isinstance(site_count, bool)
        or site_count <= 1
        or len(site_rows) != site_count
        or manifest.get("factor_tensor_count") != 8 * site_count
        or not isinstance(site_indices, list)
        or len(site_indices) != site_count
        or any(
            not isinstance(value, list)
            or len(value) != 2
            or any(
                not isinstance(item, int) or isinstance(item, bool) for item in value
            )
            for value in site_indices
        )
        or site_indices != sorted(site_indices)
        or len({tuple(value) for value in site_indices}) != site_count
        or not isinstance(site_ranks, list)
        or len(site_ranks) != site_count
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value > 8
            for value in site_ranks
        )
        or not isinstance(geometry, Mapping)
        or set(geometry)
        != {
            "logical_layers",
            "physical_groups",
            "hidden_size",
            "query_group_width",
            "head_dim",
            "factor_dtype",
        }
        or not isinstance(role_scale_values, list)
        or len(role_scale_values) != site_count
        or not isinstance(source_provenance, Mapping)
        or manifest.get("registry_schema_version") != 1
        or manifest.get("registry_layout") != 2
    ):
        raise RaggedQKVOFactorBundleError("Bundle site inventory is invalid.")
    expected_names = {"bundle.manifest.json"}
    loaded: list[LoadedQKVOFactorPack] = []
    for index, row in enumerate(site_rows):
        if not isinstance(row, Mapping) or row.get("site_index") != index:
            raise RaggedQKVOFactorBundleError("Bundle site order is invalid.")
        factor_name = row.get("factor_file")
        site_manifest_name = row.get("manifest_file")
        if (
            not isinstance(factor_name, str)
            or Path(factor_name).name != factor_name
            or not isinstance(site_manifest_name, str)
            or Path(site_manifest_name).name != site_manifest_name
        ):
            raise RaggedQKVOFactorBundleError("Bundle contains an unsafe file name.")
        factor_path = directory / factor_name
        site_manifest_path = directory / site_manifest_name
        expected_names.update((factor_path.name, site_manifest_path.name))
        pack = load_qkvo_factor_pack(factor_path, site_manifest_path)
        pack_geometry = pack.manifest.get("registry_geometry")
        pack_provenance = pack.manifest.get("provenance")
        site = row.get("site")
        if (
            row.get("factor_file_sha256") != file_sha256(factor_path)
            or row.get("manifest_file_sha256") != file_sha256(site_manifest_path)
            or row.get("site_state_sha256")
            != pack.manifest["selected_adapter_state_sha256"]
            or row.get("rank") != pack.manifest["rank"]
            or row.get("site") != pack.manifest["site"]
            or row.get("rank") != site_ranks[index]
            or row.get("factor_parameter_count") != 11_520 * site_ranks[index]
            or not isinstance(site, Mapping)
            or [site.get("layer_index"), site.get("physical_group")]
            != site_indices[index]
            or not isinstance(pack_geometry, Mapping)
            or any(pack_geometry.get(name) != value for name, value in geometry.items())
            or pack_geometry.get("rank") != site_ranks[index]
            or pack_geometry.get("site_count") != 1
            or pack.manifest.get("role_scale_provenance") != role_scale_values[index]
            or not isinstance(pack_provenance, Mapping)
            or set(pack_provenance)
            != set(source_provenance)
            | {"ragged_selected_registry_state_sha256", "ragged_site_index"}
            or any(
                pack_provenance.get(name) != value
                for name, value in source_provenance.items()
            )
            or pack_provenance.get("ragged_selected_registry_state_sha256")
            != manifest.get("selected_registry_state_sha256")
            or pack_provenance.get("ragged_site_index") != index
        ):
            raise RaggedQKVOFactorBundleError(
                "A site pack disagrees with the bundle manifest."
            )
        loaded.append(pack)
    actual_entries = tuple(directory.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in actual_entries):
        raise RaggedQKVOFactorBundleError(
            "Bundle directory contains a nested, linked, or non-file entry."
        )
    actual_names = {path.name for path in actual_entries}
    if actual_names != expected_names:
        raise RaggedQKVOFactorBundleError(
            "Bundle directory contains missing or undeclared files."
        )
    if (
        sum(path.stat().st_size for path in actual_entries)
        > MAX_RAGGED_QKVO_BUNDLE_BYTES
    ):
        raise RaggedQKVOFactorBundleError("Bundle exceeds its hard byte cap.")
    synthesized = _synthesized_ragged_state(loaded, manifest)
    if adapter_state_sha256(synthesized) != manifest.get(
        "selected_registry_state_sha256"
    ):
        raise RaggedQKVOFactorBundleError(
            "Bundle does not reconstruct the selected ragged registry state."
        )
    if manifest.get("factor_parameter_count") != sum(
        int(value.numel()) for pack in loaded for value in pack.factors.values()
    ):
        raise RaggedQKVOFactorBundleError(
            "Bundle factor parameter accounting is invalid."
        )
    if manifest.get("factor_tensor_bytes") != sum(
        int(value.numel() * value.element_size())
        for pack in loaded
        for value in pack.factors.values()
    ):
        raise RaggedQKVOFactorBundleError("Bundle factor byte accounting is invalid.")
    return LoadedRaggedQKVOFactorBundle(
        directory=directory,
        manifest=manifest,
        site_packs=tuple(loaded),
    )


def _assert_live_child_contract(
    registry: RaggedAssignmentLocalQKVOResidualRegistry,
    loaded: LoadedRaggedQKVOFactorBundle,
    attention_registry: Any,
) -> None:
    geometry = loaded.manifest.get("geometry")
    role_scale_values = loaded.manifest.get("role_scale_provenance_by_site")
    if not isinstance(geometry, Mapping) or not isinstance(role_scale_values, list):
        raise RaggedQKVOFactorBundleError("Bundle live-child contract is missing.")
    registry_geometry = {
        "logical_layers": registry.logical_layers,
        "physical_groups": registry.physical_groups,
        "hidden_size": registry.hidden_size,
        "query_group_width": registry.query_group_width,
        "head_dim": registry.head_dim,
        "factor_dtype": str(registry.q_left[0].dtype),
    }
    attention_geometry = {
        "logical_layers": getattr(attention_registry, "logical_layers", None),
        "physical_groups": getattr(attention_registry, "key_value_heads", None),
        "hidden_size": getattr(attention_registry, "output_width", None),
        "query_group_width": getattr(attention_registry, "query_rows_per_group", None),
        "head_dim": getattr(attention_registry, "head_dim", None),
        "factor_dtype": str(
            getattr(
                getattr(attention_registry, "prototype_q_weight", None), "dtype", None
            )
        ),
    }
    role_scales = getattr(attention_registry, "role_scales", None)
    group_permutations = getattr(attention_registry, "group_permutations", None)
    if (
        registry_geometry != dict(geometry)
        or attention_geometry != dict(geometry)
        or not isinstance(role_scales, Tensor)
        or not isinstance(group_permutations, Tensor)
        or tensor_sha256(role_scales)
        != next(
            (
                value.get("table_sha256")
                for value in role_scale_values
                if isinstance(value, Mapping)
            ),
            None,
        )
    ):
        raise RaggedQKVOFactorBundleError(
            "Fresh ragged registry or live attention child differs from bundle geometry."
        )
    for site_index, ((layer, physical_group), provenance) in enumerate(
        zip(
            registry.site_indices.detach().cpu().tolist(),
            role_scale_values,
            strict=True,
        )
    ):
        if not isinstance(provenance, Mapping):
            raise RaggedQKVOFactorBundleError(
                "Bundle role-scale provenance is malformed."
            )
        try:
            canonical = int(group_permutations[layer, physical_group].item())
            row = tuple(
                float(value)
                for value in role_scales[layer, canonical]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .tolist()
            )
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            raise RaggedQKVOFactorBundleError(
                "Live role-scale assignment cannot be resolved."
            ) from error
        if (
            provenance.get("canonical_assignment_axis") != canonical
            or provenance.get("role_scales_trainable") is not False
            or provenance.get("table_sha256") != tensor_sha256(role_scales)
            or row
            != tuple(float(provenance.get(role)) for role in ("q", "k", "v", "o"))
            or loaded.site_packs[site_index].manifest.get("role_scale_provenance")
            != provenance
        ):
            raise RaggedQKVOFactorBundleError(
                "Live child role scales differ from bundle provenance."
            )


def install_ragged_qkvo_factor_bundle(
    registry: RaggedAssignmentLocalQKVOResidualRegistry,
    loaded: LoadedRaggedQKVOFactorBundle,
    *,
    attention_registry: Any,
) -> str:
    """Transactionally install site packs into one provenance-matched child."""

    if not isinstance(registry, RaggedAssignmentLocalQKVOResidualRegistry):
        raise RaggedQKVOFactorBundleError("A fresh ragged registry is required.")
    if (
        registry.site_indices.detach().cpu().tolist()
        != loaded.manifest.get("site_indices")
        or list(registry.ranks) != loaded.manifest.get("site_ranks")
        or registry.site_count != len(loaded.site_packs)
    ):
        raise RaggedQKVOFactorBundleError(
            "Fresh ragged registry sites/ranks differ from bundle."
        )
    _assert_live_child_contract(registry, loaded, attention_registry)
    original = {
        name: value.detach().clone() for name, value in registry.state_dict().items()
    }
    try:
        with torch.no_grad():
            for site_index, pack in enumerate(loaded.site_packs):
                for name in QKVO_FACTOR_KEYS:
                    destination = getattr(
                        registry, name.split("_")[0] + "_" + name.split("_")[1]
                    )[site_index]
                    source = pack.factors[name].squeeze(0)
                    if (
                        tuple(destination.shape) != tuple(source.shape)
                        or destination.dtype != source.dtype
                    ):
                        raise RaggedQKVOFactorBundleError(
                            "Fresh ragged factor shape/dtype differs from bundle."
                        )
                    destination.copy_(source.to(device=destination.device))
        installed_sha = adapter_state_sha256(registry.state_dict())
        if installed_sha != loaded.manifest.get("selected_registry_state_sha256"):
            raise RaggedQKVOFactorBundleError(
                "Installed bundle did not reconstruct selected ragged state."
            )
        registry._assert_finite()
        return installed_sha
    except Exception:
        with torch.no_grad():
            current = registry.state_dict()
            for name, value in original.items():
                current[name].copy_(value.to(device=current[name].device))
        raise


__all__ = [
    "MAX_RAGGED_QKVO_BUNDLE_BYTES",
    "MAX_RAGGED_QKVO_BUNDLE_MANIFEST_BYTES",
    "RAGGED_QKVO_FACTOR_BUNDLE_SCHEMA",
    "LoadedRaggedQKVOFactorBundle",
    "RaggedQKVOFactorBundleError",
    "install_ragged_qkvo_factor_bundle",
    "load_ragged_qkvo_factor_bundle",
    "write_ragged_qkvo_factor_bundle",
]
