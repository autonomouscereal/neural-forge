"""Read the existing C17 candidate schema without relabeling its acceptance.

This module is torch-free. It checks pinned metadata only; callers must retain
C10's real captured-boundary and chart tensor validation before using a chart.
No error, cosine, compression, or capability threshold is introduced here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

SCHEMA = "neural-forge.c17-native-chart-candidate.v1"
PLAN_SCHEMA = "neural-forge.c17-native-chart-plan.v1"
OWNER_POLICY = "chimera_omni.owner_finite_diagnostics_only.2026-10-06.v1"
CONVENTION = "y_hat = x_k3 @ C_in + C_in_bias; x_hat = y_trunk @ C_out + C_out_bias"
SHA = re.compile(r"[0-9a-f]{64}\Z")


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(value: Any, message: str) -> None:
    if not value:
        raise ValueError(message)


def digest(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(SHA.fullmatch(value)), f"invalid {label} SHA256")
    return value


def layer(value: Any, label: str) -> int:
    require(type(value) is int and value >= 0, f"invalid {label} layer")
    return value


def reference(value: Any, label: str) -> dict[str, str]:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, f"invalid {label} reference")
    require(isinstance(value["path"], str) and bool(value["path"]), f"missing {label} path")
    digest(value["sha256"], label)
    return value


def read_json(ref: dict[str, str]) -> tuple[dict[str, Any], Path]:
    reference(ref, "JSON")
    path = Path(ref["path"]).resolve(strict=True)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "pinned JSON SHA256 mismatch")
    value = json.loads(raw)
    require(isinstance(value, dict), "pinned JSON must be an object")
    return value, path


def captured_rows_pin(capture: dict[str, Any], receipt_path: Path, selected_layer: int) -> dict[str, str]:
    entries = [item for item in capture.get("layers", []) if isinstance(item, dict) and item.get("layer") == selected_layer]
    require(len(entries) == 1, "capture must declare selected layer exactly once")
    entry = entries[0]
    require(type(entry.get("rows")) is int and entry["rows"] >= 2, "capture row count must be an integer >=2")
    raw_path = entry.get("path")
    require(isinstance(raw_path, str) and bool(raw_path), "capture selected row path missing")
    path = (receipt_path.parent / raw_path).resolve(strict=True)
    require(path.is_relative_to(receipt_path.parent), "captured row path escapes capture directory")
    sha = digest(entry.get("sha256"), "captured rows")
    require(file_digest(path) == sha, "captured row SHA256 mismatch")
    return {"path": str(path), "sha256": sha}


def parse_bindings(values: list[str]) -> dict[int, dict[str, str]]:
    """CLI format: HOST_LAYER=ABSOLUTE_CANDIDATE_PATH@RAW_SHA256."""
    result: dict[int, dict[str, str]] = {}
    for value in values:
        require(isinstance(value, str) and "=" in value and "@" in value, "candidate binding needs layer=path@sha256")
        number, rest = value.split("=", 1)
        require(number.isdecimal(), "candidate host layer must be a nonnegative integer")
        k = int(number)
        path, sha = rest.rsplit("@", 1)
        require(k not in result, "duplicate candidate host layer")
        require(Path(path).is_absolute(), "candidate CLI path must be absolute")
        result[k] = reference({"path": path, "sha256": sha}, "candidate")
    return result


def validate_metadata(doc: dict[str, Any], expected: dict[str, Any], *, chart_path: Path) -> dict[str, Any]:
    require(doc.get("schema") == SCHEMA, "unsupported native chart candidate schema")
    require(doc.get("fixture") is False and doc.get("gradients_or_training") is False,
            "native candidate must be an actual forward-only non-fixture artifact")
    require(doc.get("status") == "candidate_only" and doc.get("production_install_eligible") is False,
            "candidate classification must remain candidate_only, production_install_eligible=false")
    require(doc.get("source_binding_policy") == OWNER_POLICY, "explicit original owner diagnostics policy required")
    k = layer(doc.get("host_layer"), "host")
    require(k == layer(expected.get("host_layer"), "expected host"), "candidate host layer mismatch")
    for key in ("host_index_sha256", "host_config_sha256", "evidence_sha256"):
        require(digest(doc.get(key), key) == digest(expected.get(key), f"expected {key}"), f"candidate {key} mismatch")
    digest(doc.get("boundary_sha256"), "boundary")
    sources = doc.get("k3_sources")
    require(isinstance(sources, list) and bool(sources), "candidate needs explicit source records")
    layers = [layer(item.get("layer"), "K3") for item in sources if isinstance(item, dict)]
    require(len(layers) == len(sources) and layers == sorted(set(layers)) and all(l > 0 for l in layers),
            "candidate K3 layers must be positive, sorted and unique")
    require(doc.get("k3_layers") == layers and expected.get("source_layers") == layers, "candidate source coverage mismatch")
    expected_ffns = expected.get("installed_ffn_sha256")
    require(isinstance(expected_ffns, dict) and set(expected_ffns) == {str(l) for l in layers}, "expected FFN coverage mismatch")
    for item in sources:
        number = str(item["layer"])
        require(digest(item.get("installed_ffn_sha256"), "source FFN") == digest(expected_ffns[number], "expected FFN"),
                "candidate installed source FFN mismatch")
        reference(item.get("capture"), "K3 capture")
        reference(item.get("accepted_receipt"), "K3 source receipt")
    reference(doc.get("host_import_receipt"), "host import")
    reference(doc.get("host_capture"), "host capture")
    reference(doc.get("plan"), "candidate plan")
    code = doc.get("code_sha256")
    require(isinstance(code, dict) and bool(code) and all(isinstance(p, str) and p for p in code), "missing producer code pins")
    for value in code.values():
        digest(value, "producer code")
    arms = doc.get("arms")
    require(isinstance(arms, dict) and set(arms) == {"ridge"}, "native candidate supports only its actual ridge arm")
    arm = arms["ridge"]
    require(isinstance(arm, dict) and arm.get("production_install_eligible") is False, "candidate arm classification mismatch")
    chart_ref = reference({"path": arm.get("path"), "sha256": arm.get("sha256")}, "chart")
    require(Path(chart_ref["path"]).resolve(strict=True) == chart_path.resolve(strict=True), "candidate chart path mismatch")
    constraints = arm.get("constraints")
    require(constraints == {"C_out_bias": "fixed zero during fit", "C_in_bias": "fixed zero during fit"},
            "candidate must carry the fitter's exact through-origin contract")
    provenance = arm.get("native_boundary_provenance")
    require(isinstance(provenance, dict) and provenance.get("fixture") is False
            and provenance.get("schema") == "neural-forge.c17-native-chart-boundaries.v1", "invalid candidate map provenance")
    maps = provenance.get("maps")
    require(isinstance(maps, dict) and set(maps) == {"C_out", "C_in"}, "candidate must bind both directed maps")
    for name, domain_family, domain_boundary, codomain_family, codomain_boundary in (
        ("C_out", "host", "ffn_input", "k3", "ffn_input"),
        ("C_in", "k3", "ffn_output", "host", "ffn_output"),
    ):
        entry = maps[name]
        require(isinstance(entry, dict) and set(entry) == {"domain", "codomain"}, "invalid directed map endpoints")
        for side, family, boundary in (("domain", domain_family, domain_boundary), ("codomain", codomain_family, codomain_boundary)):
            endpoints = entry[side]
            require(isinstance(endpoints, list) and len(endpoints) == len(layers), "directed map coverage mismatch")
            for endpoint, source in zip(endpoints, sources):
                require(isinstance(endpoint, dict) and endpoint.get("family") == family and endpoint.get("boundary") == boundary,
                        "wrong directed native map boundary")
                require(endpoint.get("layer") == (k if family == "host" else source["layer"]), "directed map layer mismatch")
                require(endpoint.get("capture_receipt") == (doc["host_capture"] if family == "host" else source["capture"]),
                        "directed map capture identity mismatch")
    # Preserve the actual candidate on disk. This compatibility view is passed
    # solely to the existing boundary verifier, never written as layer.json.
    compatibility = {"trunk_layer": k, "trunk_taps": {"rows_key": "ffn_input"},
                     "files": {"ridge": {"file_sha256": chart_ref["sha256"]}},
                     "arms": {"ridge": {"convention": CONVENTION, "native_boundary_provenance": provenance}}}
    return {"document": compatibility, "chart_sha256": chart_ref["sha256"], "expected": expected,
            "source_binding_policy": OWNER_POLICY, "status": "candidate_only", "production_install_eligible": False}


def read_binding(ref: dict[str, str], expected: dict[str, Any], *, chart_path: Path) -> dict[str, Any]:
    doc, path = read_json(ref)
    require(path == chart_path.resolve(strict=True).parent / "candidate_layer.json", "candidate must keep its canonical filename/location")
    result = validate_metadata(doc, expected, chart_path=chart_path)
    plan, _ = read_json(doc["plan"])
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("fixture") is False
            and plan.get("source_binding_policy") == OWNER_POLICY, "candidate plan schema/policy mismatch")
    for key in ("host_layer", "host_import_receipt", "host_capture"):
        require(plan.get(key) == doc.get(key), f"candidate plan {key} mismatch")
    require((plan.get("evidence") or {}).get("sha256") == doc["evidence_sha256"]
            and (plan.get("boundary_map") or {}).get("sha256") == doc["boundary_sha256"], "candidate plan evidence/boundary mismatch")
    require(plan.get("k3_captures") == [{"layer": item["layer"], "capture": item["capture"],
            "accepted_receipt": item["accepted_receipt"]} for item in doc["k3_sources"]], "candidate plan source joins mismatch")
    # Reuse the already adopted receipt reader: unrelated config/index/receipt
    # files cannot stand in for the completed native S3 host bundle.
    import c17_source_binding as source_binding
    imported, _ = read_json(doc["host_import_receipt"])
    host_binding = source_binding.host_checkpoint_binding(imported, source_binding_policy=OWNER_POLICY,
                                                          metadata_refs=plan.get("host_native_metadata"))
    require(doc.get("host_checkpoint_binding") == host_binding, "candidate host checkpoint binding mismatch")
    for name, key in (("config_sha256", "host_config_sha256"), ("index_sha256", "host_index_sha256")):
        require(host_binding["checkpoint"].get(name) == expected[key], "candidate imported host identity mismatch")
    host_capture, host_capture_path = read_json(doc["host_capture"])
    require(host_capture.get("schema") == "neural-forge.c17-host-boundaries.v1"
            and host_capture.get("gradients_or_training") is False and host_capture.get("fixture", False) is False,
            "candidate host capture schema mismatch")
    require((host_capture.get("host_receipt") or {}).get("sha256") == doc["host_import_receipt"]["sha256"]
            and host_capture.get("config_sha256") == expected["host_config_sha256"]
            and host_capture.get("index_sha256") == expected["host_index_sha256"]
            and host_capture.get("evidence_sha256") == expected["evidence_sha256"], "candidate host capture identity mismatch")
    row_pins = [captured_rows_pin(host_capture, host_capture_path, expected["host_layer"])]
    source_context = None
    for source in doc["k3_sources"]:
        capture, capture_path = read_json(source["capture"])
        receipt, _ = read_json(source["accepted_receipt"])
        require(capture.get("schema") == "neural-forge.c17-k3-boundaries.v1" and capture.get("fixture", False) is False
                and capture.get("gradients_or_training") is False and capture.get("stream") == "folded"
                and capture.get("accepted_prefix_replay_exact") is True, "candidate source capture contract mismatch")
        number = str(source["layer"])
        context = capture.get("accepted_resume_context")
        require(isinstance(context, dict) and bool(context) and isinstance(receipt.get("resume_context"), dict)
                and bool(receipt["resume_context"]), "candidate source resume context must be a nonempty object")
        require(source_context is None or context == source_context, "pooled candidate sources have different native contexts")
        source_context = context
        require((capture.get("accepted_receipt") or {}).get("sha256") == source["accepted_receipt"]["sha256"]
                and context == receipt["resume_context"]
                and (capture.get("installed_ffn_sha256") or {}).get(number) == source["installed_ffn_sha256"]
                and ((receipt.get("layers") or {}).get(number, {}).get("ffn") or {}).get("sha256") == source["installed_ffn_sha256"],
                "candidate source receipt/FFN/context mismatch")
        row_pins.append(captured_rows_pin(capture, capture_path, source["layer"]))
    refs = [ref, doc["plan"], doc["host_import_receipt"], doc["host_capture"],
                       *[ref for item in doc["k3_sources"] for ref in (item["capture"], item["accepted_receipt"])]]
    if host_binding["receipt_kind"] == source_binding.S3_BUNDLE_SCHEMA:
        refs.extend(plan["host_native_metadata"].values())
    for source_ref in refs:
        read_json(source_ref)
    require(file_digest(chart_path) == result["chart_sha256"], "candidate chart SHA256 mismatch")
    result["pins"] = [{"path": str(Path(item["path"]).resolve(strict=True)), "sha256": item["sha256"]} for item in refs]
    result["pins"].append({"path": str(chart_path.resolve(strict=True)), "sha256": result["chart_sha256"]})
    result["pins"].extend(row_pins)
    result["candidate_path"] = str(path)
    result["candidate_sha256"] = ref["sha256"]
    return result


def check_observed(binding: dict[str, Any], observed: dict[str, Any]) -> None:
    for key in ("host_index_sha256", "evidence_sha256", "source_layers", "installed_ffn_sha256"):
        require(observed.get(key) == binding["expected"].get(key), f"observed candidate {key} mismatch")


def verify_pins(pins: list[dict[str, str]]) -> None:
    for ref in pins:
        reference(ref, "stable input")
        require(file_digest(Path(ref["path"])) == ref["sha256"], "candidate input changed during consumption")
