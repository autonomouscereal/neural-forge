from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments import run_lfm2_exact4x_moving_gauge_v5 as v5
from experiments import run_lfm2_exact4x_moving_gauge_v5_r12_live as launch
from neural_forge.dynamic_circuit_ir import canonical_json_bytes
from neural_forge.dynamic_counted_child_state import (
    ContinuousChildStateAllocation,
    CountedChildStatePlan,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _exact_plan() -> CountedChildStatePlan:
    return CountedChildStatePlan(
        parent_unique_parameters=v5.CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS,
        target_total_parameter_elements=(
            v5.CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS
        ),
        dense_model_parameter_elements=(
            v5.CURRENT_LFM2_CARRIER_DENSE_PARAMETER_ELEMENTS
        ),
        continuous_allocations=(
            ContinuousChildStateAllocation(
                state_id="semantic-carrier-route.c_in",
                parameter_elements=v5.CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS,
                tensor_sha256=_digest("route-in-tensor"),
                artifact_sha256=_digest("route-in-artifact"),
            ),
            ContinuousChildStateAllocation(
                state_id="semantic-carrier-route.c_out",
                parameter_elements=v5.CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS,
                tensor_sha256=_digest("route-out-tensor"),
                artifact_sha256=_digest("route-out-artifact"),
            ),
        ),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _calibration_rows() -> list[dict[str, Any]]:
    rows = []
    for index in range(v5.EXPECTED_BATCH_COUNT):
        split = "selection" if index < 32 else "fit"
        rows.append(
            {
                "schema": launch.TARGET_CALIBRATION_SCHEMA,
                "id": f"source-{index:06d}",
                "split": split,
                "target_tokenization_mode": "tokenizer-call-add-special-tokens",
                "target_forced_decode_tokenization_mode": (
                    "tokenizer-call-no-special-tokens"
                ),
                "target_input_texts": [f"prompt {index}"],
                "target_forced_decode_texts": ["answer"],
                "input_ids": [1, 10 + index],
                "attention_mask": [1, 1],
                "forced_decode_ids": [2 + index],
            }
        )
    return rows


class _Authority:
    def __init__(self, *, key_id: str, secret: bytes) -> None:
        self.key_id = key_id
        self.secret_seen = bytes(secret)


class _Adapter:
    def __init__(self, records: tuple[Any, ...], evidence_sha256: str) -> None:
        self.records = records
        self.evidence_sha256 = evidence_sha256
        self.bridge_artifact_sha256 = _digest("bridge-artifact")
        self.transport_profile_fingerprint = _digest("bridge-profile")

    def verify(self) -> _Adapter:
        return self


class _Fixture:
    def __init__(self, tmp_path: Path) -> None:
        self.calls: list[str] = []
        self.authorities: list[_Authority] = []
        self.handler_calls: list[str] = []
        self.plan = _exact_plan()
        self.package = SimpleNamespace(
            package_sha256=_digest("child-package"),
            routes=SimpleNamespace(package_sha256=_digest("route-package")),
            receipt={"source_bridge_artifact_sha256": _digest("bridge-artifact")},
        )
        self.v3_request = {
            "request_sha256": _digest("v3-request"),
            "child_package_sha256": self.package.package_sha256,
            "child_state_sha256": _digest("child-identity"),
            "route_package_sha256": _digest("route-package"),
            "route_runtime_sha256": _digest("route-runtime"),
            "portable_runtime_receipt_sha256": _digest("portable-runtime"),
            "trace_cache_key": _digest("parent-cache-key"),
            "basis_phase_key_sha256": _digest("basis-key"),
            "basis_completion_sha256": _digest("basis-completion"),
            "basis_map_sha256": _digest("basis-map"),
            "runtime_device": "cuda:0",
            "runtime_dtype": "float32",
        }
        self.v3_ledger = {
            "ledger_sha256": _digest("v3-ledger"),
            "batch_count": v5.EXPECTED_BATCH_COUNT,
        }
        self.parent_input_sha256s = tuple(
            _digest(f"parent-input:{index}") for index in range(v5.EXPECTED_BATCH_COUNT)
        )
        horizons = [
            {
                "batch_index": index,
                "record_id": self.parent_input_sha256s[index],
                "split": "heldout" if index % 4 == 2 else "fit",
            }
            for index in range(v5.EXPECTED_BATCH_COUNT)
        ]
        self.basis = SimpleNamespace(
            phase_key=SimpleNamespace(phase_key_sha256=_digest("basis-key")),
            completion_sha256=_digest("basis-completion"),
            map_sha256=_digest("basis-map"),
            source_anchor={"hmac_sha256": _digest("basis-anchor")},
            report={"cache": {"record_horizons": horizons}},
        )
        self.bridge = SimpleNamespace(
            artifact_sha256=_digest("bridge-artifact"),
            profile_fingerprint=_digest("bridge-profile"),
        )
        local_paths = {}
        for name in v5.REQUIRED_LOCAL_PATH_KEYS:
            path = tmp_path / name
            path.mkdir()
            local_paths[name] = str(path.absolute())
        _write_json(
            Path(local_paths["v3_evidence"]) / "request.json",
            self.v3_request,
        )
        _write_json(
            Path(local_paths["v3_evidence"]) / "first-divergence-ledger.json",
            self.v3_ledger,
        )
        calibration_path = tmp_path / "target-calibration.json"
        _write_json(calibration_path, _calibration_rows())
        calibration_sha256 = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        bridge_root = tmp_path / "tokenizer-bridge"
        bridge_root.mkdir()
        provenance = {
            name: _digest(f"provenance:{name}") for name in v5_live_provenance_keys()
        }
        provenance.update(
            {
                "child_package": self.package.package_sha256,
                "child_identity": _digest("child-identity"),
                "child_state_dict": _digest("child-state"),
                "route_package": _digest("route-package"),
                "route_runtime": _digest("route-runtime"),
                "portable_runtime_receipt": _digest("portable-runtime"),
                "v3_request": self.v3_request["request_sha256"],
                "v3_ledger": self.v3_ledger["ledger_sha256"],
                "basis_phase_key": _digest("basis-key"),
                "basis_completion": _digest("basis-completion"),
                "basis_map": _digest("basis-map"),
                "bridge_artifact": self.bridge.artifact_sha256,
                "bridge_profile": self.bridge.profile_fingerprint,
                "bridge_trace_adapter": _digest("trace-adapter"),
            }
        )
        self.parent_cache = SimpleNamespace(
            cache_key=self.v3_request["trace_cache_key"],
            manifest={
                "manifest_sha256": provenance["parent_manifest"],
                "parent": {"identity_sha256": provenance["parent_identity"]},
                "layer_topology": {
                    "layer_topology_sha256": provenance["parent_topology"]
                },
                "corpus": {
                    "batch_count": v5.EXPECTED_BATCH_COUNT,
                    "ordered_batch_input_sha256": list(self.parent_input_sha256s),
                },
            },
            _completion={"completion_sha256": provenance["parent_completion"]},
        )
        lineage = {
            name: _digest(f"lineage:{name}") for name in v5.REQUIRED_LINEAGE_KEYS
        }
        lineage.update(
            {
                "r12_child_package": provenance["child_package"],
                "r12_child_identity": provenance["child_identity"],
                "r12_state_dict": provenance["child_state_dict"],
                "v3_request": provenance["v3_request"],
                "v3_ledger": provenance["v3_ledger"],
                "parent_trace_cache_completion": provenance["parent_completion"],
                "child_trace_cache_completion": provenance["child_completion"],
                "basis_phase_key": provenance["basis_phase_key"],
                "basis_phase_completion": provenance["basis_completion"],
                "basis_source_anchor": self.basis.source_anchor["hmac_sha256"],
                "tokenizer_bridge": provenance["bridge_artifact"],
                "counted_child_state_plan": self.plan.fingerprint,
            }
        )
        body = {
            "schema": launch.LAUNCH_SCHEMA,
            "local_paths": local_paths,
            "tokenizer_bridge_root": str(bridge_root.absolute()),
            "target_calibration": {
                "path": str(calibration_path.absolute()),
                "file_sha256": calibration_sha256,
            },
            "lineage_sha256s": lineage,
            "provenance_sha256s": provenance,
            "authority_key_ids": {
                "basis_phase": "basis-key-id",
                "boundary_phase": "boundary-key-id",
                "feature_phase": "feature-key-id",
                "conditioned_recapture": "conditioned-key-id",
            },
            "replay": {
                "devices": ["cuda:0"],
                "runtime_device": "cuda:0",
                "runtime_dtype": "float32",
                "recapture_checkpoint_batch_count": 16,
                "boundary_shard_batch_count": 32,
                "feature_shard_batch_count": 32,
            },
            "storage_dtypes": {
                "parent": "bfloat16",
                "child": "float32",
                "child_features": {
                    "short_conv.context": "float32",
                    "mlp.post_swiglu": "float32",
                },
            },
            "receipt_path": str(
                (Path(local_paths["output_root"]) / "r12-v5-receipt.json").absolute()
            ),
        }
        self.manifest = launch.seal_r12_v5_launch_manifest(body)
        self.manifest_path = tmp_path / "launch.json"
        _write_json(self.manifest_path, self.manifest)
        self.key_path = tmp_path / "authority.key"
        self.key_bytes = b"R12-v5-test-authority-key-material-000000"
        self.key_path.write_bytes(self.key_bytes)

    def authority(self, *, key_id: str, secret: bytes) -> _Authority:
        value = _Authority(key_id=key_id, secret=secret)
        self.authorities.append(value)
        return value

    def dependencies(self) -> launch.R12V5LaunchDependencies:
        def verify_runtime_surface() -> dict[str, str]:
            self.calls.append("runtime")
            return {"result_sha256": _digest("runtime-surface")}

        def preflight(request: Any) -> dict[str, str]:
            self.calls.append("preflight")
            assert request.counted_child_state_plan.fingerprint == self.plan.fingerprint
            return {"result_sha256": _digest("preflight")}

        def load_package(_path: Path, **kwargs: Any) -> Any:
            self.calls.append("package")
            assert kwargs["expected_package_sha256"] == self.package.package_sha256
            return self.package

        def reconstruct(package: Any, **kwargs: Any) -> Any:
            self.calls.append("count-plan")
            assert package is self.package
            assert kwargs["expected_package_sha256"] == self.package.package_sha256
            return self.plan

        def open_trace_cache(path: str, **kwargs: Any) -> Any:
            self.calls.append("parent-cache")
            assert path == self.manifest["local_paths"]["parent_trace_cache"]
            assert kwargs == {"verify": False}
            return self.parent_cache

        def authenticate_request(value: Any) -> Any:
            self.calls.append("v3-request")
            assert value == self.v3_request
            return dict(value)

        def authenticate_ledger(value: Any, **kwargs: Any) -> Any:
            self.calls.append("v3-ledger")
            assert value == self.v3_ledger
            assert kwargs["expected_request"] == self.v3_request
            return dict(value)

        def load_basis(_cache: Any, phase_key: str, **kwargs: Any) -> Any:
            self.calls.append("basis")
            assert phase_key == self.basis.phase_key.phase_key_sha256
            assert kwargs["anchor_authority"] in self.authorities
            return self.basis

        def load_bridge(_path: Path, **kwargs: Any) -> Any:
            self.calls.append("bridge")
            assert _path == Path(self.manifest["tokenizer_bridge_root"])
            assert kwargs == {
                "expected_artifact_sha256": self.bridge.artifact_sha256,
                "packaged": False,
            }
            return self.bridge

        def authenticate_packaged_bridge_reference(
            package: Any,
            bridge: Any,
        ) -> dict[str, str]:
            self.calls.append("packaged-bridge-reference")
            assert package is self.package
            assert bridge is self.bridge
            return {
                "bridge_artifact_sha256": bridge.artifact_sha256,
                "bridge_profile_sha256": bridge.profile_fingerprint,
            }

        def build_adapter(
            bridge: Any,
            records: tuple[Any, ...],
            **kwargs: Any,
        ) -> _Adapter:
            self.calls.append("adapter")
            assert bridge is self.bridge
            assert kwargs["expected_artifact_sha256"] == bridge.artifact_sha256
            assert tuple(value.record_id for value in records) == tuple(
                f"batch-{index:06d}" for index in range(v5.EXPECTED_BATCH_COUNT)
            )
            return _Adapter(records, _digest("trace-adapter"))

        def build_handlers(request: Any, **kwargs: Any) -> dict[str, Any]:
            self.calls.append("handlers")
            assert request.replay_devices == ("cuda:0",)
            split = kwargs["basis_split_by_batch"]
            assert tuple(split[index] for index in range(8)) == (
                "fit",
                "fit",
                "heldout",
                "fit",
                "fit",
                "fit",
                "heldout",
                "fit",
            )
            assert tuple(
                kwargs[name]
                for name in (
                    "basis_phase_authority",
                    "boundary_phase_authority",
                    "feature_phase_authority",
                    "conditioned_recapture_authority",
                )
            ) == tuple(self.authorities)

            def handler(name: str) -> Any:
                def execute(*_args: Any, **_kwargs: Any) -> Any:
                    self.handler_calls.append(name)
                    return SimpleNamespace(artifact_sha256=_digest(f"phase:{name}"))

                return execute

            return {name: handler(name) for name in v5.PHASE_ORDER}

        def run_machine(
            _request: Any,
            handlers: dict[str, Any],
            **_kwargs: Any,
        ) -> Any:
            self.calls.append("phase-machine")
            outcomes = {}
            for name in v5.PHASE_ORDER:
                outcomes[name] = handlers[name]({}, outcomes)
            return outcomes, {"result_sha256": _digest("phase-machine")}

        return launch.R12V5LaunchDependencies(
            verify_runtime_surface=verify_runtime_surface,
            authenticate_local_preflight=preflight,
            load_package=load_package,
            open_trace_cache=open_trace_cache,
            reconstruct_counted_plan=reconstruct,
            authenticate_v3_request=authenticate_request,
            authenticate_v3_ledger=authenticate_ledger,
            load_basis_phase=load_basis,
            load_bridge=load_bridge,
            authenticate_packaged_bridge_reference=(
                authenticate_packaged_bridge_reference
            ),
            build_trace_adapter=build_adapter,
            basis_authority_factory=self.authority,
            boundary_authority_factory=self.authority,
            feature_authority_factory=self.authority,
            conditioned_authority_factory=self.authority,
            build_live_handlers=build_handlers,
            run_phase_machine=run_machine,
        )


def v5_live_provenance_keys() -> frozenset[str]:
    # Keep the import surface under test explicit without copying the inventory.
    from experiments import run_lfm2_exact4x_moving_gauge_v5_live as live

    return live.REQUIRED_BOUNDARY_PROVENANCE_KEYS


def test_manifest_launch_rebuilds_exact_records_split_and_executes_nine(
    tmp_path: Path,
) -> None:
    fixture = _Fixture(tmp_path)
    manifest = launch.read_r12_v5_launch_manifest(fixture.manifest_path)

    receipt = launch.execute_r12_v5_launch(
        manifest,
        authority_key_path=fixture.key_path,
        dependencies=fixture.dependencies(),
        logger=lambda *_args, **_kwargs: None,
    )

    assert fixture.handler_calls == list(v5.PHASE_ORDER)
    assert len(fixture.authorities) == 4
    assert all(value.secret_seen == fixture.key_bytes for value in fixture.authorities)
    assert receipt["all_nine_handlers_executed_via_v5_phase_machine"] is True
    assert receipt["basis_split_counts"] == {"fit": 192, "heldout": 64}
    assert receipt["trace_record_ids_sha256"] == launch.canonical_sha256(
        [f"batch-{index:06d}" for index in range(256)]
    )
    receipt_path = Path(manifest["receipt_path"])
    assert receipt_path.read_bytes() == canonical_json_bytes(dict(receipt)) + b"\n"


def test_secret_authorities_and_key_path_are_never_serialized(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)

    receipt = launch.execute_r12_v5_launch(
        fixture.manifest,
        authority_key_path=fixture.key_path,
        dependencies=fixture.dependencies(),
    )
    payload = canonical_json_bytes(dict(receipt))
    manifest_payload = canonical_json_bytes(fixture.manifest)

    assert fixture.key_bytes not in payload
    assert str(fixture.key_path).encode() not in payload
    assert str(fixture.key_path).encode() not in manifest_payload
    assert b"basis_phase_authority" not in payload
    assert receipt["authority_objects_serialized"] is False
    assert receipt["authority_key_path_serialized"] is False
    assert receipt["authority_key_bytes_serialized"] is False


def test_runtime_api_failure_precedes_artifact_or_key_access(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    fixture.key_path.unlink()
    package_called = False

    def package(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal package_called
        package_called = True
        raise AssertionError("package must not be reached")

    dependencies = fixture.dependencies()
    dependencies = launch.R12V5LaunchDependencies(
        **{
            name: getattr(dependencies, name)
            for name in dependencies.__dataclass_fields__
            if name not in {"verify_runtime_surface", "load_package"}
        },
        verify_runtime_surface=lambda: (_ for _ in ()).throw(
            RuntimeError("conditioned API missing")
        ),
        load_package=package,
    )

    with pytest.raises(RuntimeError, match="conditioned API missing"):
        launch.execute_r12_v5_launch(
            fixture.manifest,
            authority_key_path=fixture.key_path,
            dependencies=dependencies,
        )
    assert package_called is False
    assert not Path(fixture.manifest["receipt_path"]).exists()


def test_missing_v3_artifact_fails_closed_without_running_handlers(
    tmp_path: Path,
) -> None:
    fixture = _Fixture(tmp_path)
    (Path(fixture.manifest["local_paths"]["v3_evidence"]) / "request.json").unlink()

    with pytest.raises(
        launch.Lfm2Exact4xMovingGaugeV5R12LaunchError,
        match="missing or unreadable",
    ):
        launch.execute_r12_v5_launch(
            fixture.manifest,
            authority_key_path=fixture.key_path,
            dependencies=fixture.dependencies(),
        )
    assert fixture.handler_calls == []
    assert not Path(fixture.manifest["receipt_path"]).exists()


def test_existing_receipt_rejects_before_any_runtime_work(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    receipt_path = Path(fixture.manifest["receipt_path"])
    receipt_path.write_text("occupied", encoding="ascii")
    dependencies = fixture.dependencies()

    with pytest.raises(
        launch.Lfm2Exact4xMovingGaugeV5R12LaunchError,
        match="existing R12 V5 launch receipt",
    ):
        launch.execute_r12_v5_launch(
            fixture.manifest,
            authority_key_path=fixture.key_path,
            dependencies=dependencies,
        )
    assert fixture.calls == []


def test_launch_surface_has_no_parent_hf_or_network_dependency() -> None:
    signature = inspect.signature(launch.execute_r12_v5_launch)
    source = inspect.getsource(launch)

    assert not {
        "parent_model",
        "parent_loader",
        "parent_forward",
        "hf_loader",
        "network_client",
    }.intersection(signature.parameters)
    assert "from_pretrained(" not in source
    assert "AutoModel" not in source
    assert "requests." not in source
    assert "socket." not in source
    assert launch.os.environ["HF_HUB_OFFLINE"] == "1"
    assert launch.os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_basis_split_rejects_noncanonical_record_identity(tmp_path: Path) -> None:
    fixture = _Fixture(tmp_path)
    fixture.basis.report["cache"]["record_horizons"][7]["record_id"] = "source-7"

    with pytest.raises(
        launch.Lfm2Exact4xMovingGaugeV5R12LaunchError,
        match="exact parent trace input identities",
    ):
        launch.derive_authoritative_basis_split(
            fixture.basis,
            fixture.parent_cache,
            provenance_sha256s=fixture.manifest["provenance_sha256s"],
            expected_cache_key=fixture.v3_request["trace_cache_key"],
        )


@pytest.mark.parametrize(
    ("reference_schema", "resource_fingerprint"),
    (
        (launch.CATALOG_BRIDGE_REFERENCE_SCHEMA, None),
        (launch.DIRECT_LOCAL_BRIDGE_REFERENCE_SCHEMA, _digest("resource")),
    ),
)
def test_packaged_strict_bridge_is_distinct_from_carrier_route_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_schema: str,
    resource_fingerprint: str | None,
) -> None:
    package_root = tmp_path / "published-child"
    package_root.mkdir()
    reference_path = package_root / launch.BRIDGE_REFERENCE_FILENAME
    bridge = SimpleNamespace(
        artifact_sha256=_digest("strict-bridge-artifact"),
        profile_fingerprint=_digest("strict-bridge-profile"),
        target_tokenizer_semantics_sha256=_digest("target-tokenizer-semantics"),
    )
    target_package = _digest("target-tokenizer-package")
    reference = {
        "schema": reference_schema,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "strict_bridge_manifest_is_external_provenance": True,
        "target_tokenizer_package_fingerprint": target_package,
        "target_tokenizer_semantics_sha256": (bridge.target_tokenizer_semantics_sha256),
    }
    if resource_fingerprint is not None:
        reference["resource_fingerprint"] = resource_fingerprint
    _write_json(reference_path, reference)
    raw = reference_path.read_bytes()
    package = SimpleNamespace(
        root=package_root,
        receipt={
            # This is the continuous carrier-route source and must remain a
            # distinct identity from the strict tokenizer bridge.
            "source_bridge_artifact_sha256": _digest("carrier-route-source"),
            "base_root_file_records": [
                {
                    "relative_path": launch.BRIDGE_REFERENCE_FILENAME,
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            ],
        },
    )
    monkeypatch.setattr(
        launch,
        "strict_tokenizer_bridge_target_package_fingerprint",
        lambda value: target_package if value is bridge else None,
    )

    authenticated = launch.authenticate_packaged_tokenizer_bridge_reference(
        package,
        bridge,
    )

    assert authenticated["bridge_artifact_sha256"] == bridge.artifact_sha256
    assert (
        package.receipt["source_bridge_artifact_sha256"]
        != (authenticated["bridge_artifact_sha256"])
    )

    mutated = {**reference, "bridge_artifact_sha256": _digest("foreign-bridge")}
    reference_path.write_bytes(canonical_json_bytes(mutated) + b"\n")
    with pytest.raises(
        launch.Lfm2Exact4xMovingGaugeV5R12LaunchError,
        match="Packaged tokenizer bridge reference differs",
    ):
        launch.authenticate_packaged_tokenizer_bridge_reference(package, bridge)
