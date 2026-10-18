"""CPU-only recovery-corpus normalization, evidence validation, and coverage audit.

This writes preparation artifacts only. It never loads a model, executes tool
calls, downloads media, invents schemas/results, or changes a campaign corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "recovery_corpus.v1"
ROUTES = (
    "text_instruction", "text_reasoning", "text_arithmetic", "text_code",
    "text_multiturn", "tool_schema", "tool_call", "tool_result", "tool_retry",
    "tool_parallel", "tool_multiturn", "vision_grounding", "image_generate",
    "image_edit", "video_understand", "audio_understand", "audio_generate",
    "music_generate", "joint_av", "volume_generate",
)
ALIASES = {
    "text": "text_instruction", "broad_domain": "text_instruction",
    "instruction": "text_instruction", "ops_instruction": "text_instruction",
    "knowledge_qa": "text_instruction", "science": "text_reasoning",
    "science_reasoning": "text_reasoning", "chemistry": "text_reasoning",
    "chemistry_science": "text_reasoning", "finance": "text_reasoning",
    "finance_economics": "text_reasoning", "healthcare": "text_instruction",
    "healthcare_workflow": "text_instruction", "math": "text_reasoning",
    "reasoning": "text_reasoning", "reasoning_algorithmic": "text_reasoning",
    "arc_grid_reasoning": "text_reasoning", "grid": "text_reasoning",
    "long_context": "text_reasoning", "synthesis": "text_reasoning",
    "table": "text_reasoning", "structured_plan": "text_reasoning",
    "web_concept_reasoning": "text_reasoning", "web_domain_extraction": "text_instruction",
    "web_knowledge": "text_instruction", "research_instruction": "text_instruction",
    "ops_runbook": "text_instruction", "defensive_runbook": "text_instruction",
    "data_pipeline_design": "text_code", "test_design": "text_code",
    "code": "text_code", "sql_text": "text_code", "code_and_metrics": "text_code",
    "tool": "tool_call", "tool_calling": "tool_call", "agentic_trace": "tool_multiturn",
    "agentic_task": "tool_multiturn", "terminal": "tool_call", "browser": "tool_call",
    "image": "image_generate", "image_generation": "image_generate",
    "image_editing": "image_edit", "ocr": "vision_grounding",
    "ocr_document": "vision_grounding", "vision": "vision_grounding",
    "image_grounding": "vision_grounding",
    "video": "video_understand", "video_media_flow": "joint_av",
    "media_flow": "joint_av", "multimodal": "joint_av",
    "audio": "audio_generate", "sound": "audio_generate", "tts": "audio_generate",
    "speech": "audio_understand", "music": "music_generate", "3d": "volume_generate",
    "volume": "volume_generate", "qimg": "image_generate", "hy3d": "volume_generate",
    "ace": "music_generate", "h3": "joint_av", "vit": "vision_grounding",
}
MEDIA_ROUTES = frozenset(ROUTES[11:])
GENERATIVE_MEDIA = frozenset({"image_generate", "image_edit", "audio_generate", "music_generate", "joint_av", "volume_generate"})
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", text.casefold()))


def strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            if key not in {"path", "sha256", "id", "source_ids", "group_id", "source"}:
                yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def task_text(record: dict) -> str:
    pieces = []
    for message in record.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            # A shared public tool catalog is not the task or target answer.
            if "Available tool schemas:" in content:
                content = content.split("Available tool schemas:", 1)[0]
            pieces.append(content)
        pieces.extend(strings(message.get("tool_calls") or []))
    return "\n".join(pieces)


class EvalIndex:
    """Exact tasks and >50% 13-gram overlap, plus source/media family isolation."""
    def __init__(self, records: Iterable[dict] = ()):
        self.exact: set[tuple[str, ...]] = set()
        self.ngrams: set[tuple[str, ...]] = set()
        self.identities: set[str] = set()
        for record in records:
            self.add(record)

    def add(self, record: dict) -> None:
        tokens = words(task_text(record))
        if len(tokens) >= 4:
            self.exact.add(tokens)
        self.ngrams.update(tokens[i:i + 13] for i in range(len(tokens) - 12))
        self.identities.update(identity_keys(record))

    def hits(self, record: dict) -> list[str]:
        hits = set()
        if self.identities.intersection(identity_keys(record)):
            hits.add("eval_source_or_media_identity")
        tokens = words(task_text(record))
        if tokens in self.exact and len(tokens) >= 4:
            hits.add("eval_exact_text")
        count = len(tokens) - 12
        if count > 0 and sum(tokens[i:i + 13] in self.ngrams for i in range(count)) / count > 0.5:
            hits.add("eval_13gram_near_duplicate")
        return sorted(hits)


class AssetResolver:
    def __init__(self, root: Path, path_maps: Iterable[tuple[str, str]] = (), max_asset_bytes: int = 1024 ** 3):
        self.root = root.resolve()
        self.path_maps = sorted(path_maps, key=lambda p: -len(p[0]))
        self.max_asset_bytes = max_asset_bytes
        self.cache: dict[tuple, dict] = {}

    def resolve(self, source: str) -> Path:
        value = source.replace("\\", "/")
        for old, new in self.path_maps:
            old = old.replace("\\", "/").rstrip("/")
            if value == old or value.startswith(old + "/"):
                return Path(new) / value[len(old):].lstrip("/")
        path = Path(source)
        return path if path.is_absolute() else self.root / path

    def inspect(self, asset: dict) -> dict:
        source = str(asset.get("path") or "")
        expected = str(asset.get("sha256") or "").lower()
        key = source, expected, asset.get("tensor"), tuple(asset.get("shape") or []), asset.get("representation")
        if key in self.cache:
            return self.cache[key]
        report: dict = {"source_path": source, "resolved_path": str(self.resolve(source)), "verified": False}
        errors = []
        path = self.resolve(source)
        if not source:
            errors.append("media_path_missing")
        elif not path.is_file():
            errors.append("media_file_missing")
        elif not SHA_RE.fullmatch(expected):
            errors.append("media_sha256_missing")
        elif path.stat().st_size > self.max_asset_bytes:
            errors.append("media_verification_byte_limit")
        else:
            actual = file_sha256(path)
            report.update(bytes=path.stat().st_size, actual_sha256=actual)
            if actual != expected:
                errors.append("media_sha256_mismatch")
            elif path.stat().st_size == 0:
                errors.append("media_empty_file")
            else:
                report["verified"] = True
                if path.suffix == ".safetensors":
                    try:
                        report["tensors"] = safetensors_header(path)
                        tensor = asset.get("tensor")
                        if tensor and tensor not in report["tensors"]:
                            errors.append("media_tensor_missing")
                        elif asset.get("representation") == "latent" and not tensor:
                            errors.append("media_latent_tensor_unspecified")
                        elif tensor and asset.get("shape") != report["tensors"][tensor]["shape"]:
                            errors.append("media_tensor_shape_mismatch")
                    except (ValueError, OSError, KeyError, TypeError) as exc:
                        errors.append("media_safetensors_header_invalid")
                        report["header_error"] = str(exc)
        report["errors"] = errors
        if errors:
            report["verified"] = False
        self.cache[key] = report
        return report


def safetensors_header(path: Path) -> dict:
    """Validate names, shapes, offsets and payload sizes without importing torch."""
    sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
             "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "U32": 4, "U64": 8}
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("short length prefix")
        length = struct.unpack("<Q", prefix)[0]
        if length > 16 * 1024 * 1024 or length < 2:
            raise ValueError("header length outside bound")
        header = json.loads(stream.read(length))
    payload = path.stat().st_size - 8 - length
    spans = []
    result = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        shape = spec["shape"]
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("invalid shape")
        begin, end = spec["data_offsets"]
        if type(begin) is not int or type(end) is not int or not 0 <= begin <= end <= payload:
            raise ValueError("invalid payload span")
        dtype = spec["dtype"]
        if dtype not in sizes or end - begin != math.prod(shape) * sizes[dtype]:
            raise ValueError("dtype/shape payload mismatch")
        result[name] = {"shape": shape, "dtype": dtype, "data_offsets": [begin, end]}
        spans.append((begin, end))
    spans.sort()
    if not spans or spans[0][0] != 0 or spans[-1][1] != payload or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("payload gaps, overlaps, or no tensors")
    return result


def identity_keys(record: dict) -> set[str]:
    provenance = record.get("provenance") or {}
    source = str(provenance.get("source") or "")
    result = {f"source:{source}:{sid}" for sid in provenance.get("source_ids", [])}
    for field in ("archive_payload_sha256s", "source_payload_sha256s"):
        result.update("payload:" + str(h) for h in provenance.get(field, []))
    if provenance.get("group_id"):
        result.add("group:" + str(provenance["group_id"]))
    if record.get("group_id"):
        result.add("group:" + str(record["group_id"]))
    for asset in record.get("media", []):
        if isinstance(asset, dict) and SHA_RE.fullmatch(str(asset.get("sha256") or "").lower()):
            result.add("media:" + asset["sha256"].lower())
    for key in ("evidence_key", "text_key"):
        if record.get(key):
            result.add("content:" + str(record[key]))
    return result


def split_for(group_id: str) -> str:
    bucket = int(hashlib.sha256(("recovery-corpus-v1:" + group_id).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def parsed_json_objects(text: Any) -> Iterable[Any]:
    """Only complete JSON or explicit fenced JSON; never infer schema from arguments."""
    if not isinstance(text, str):
        return
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.S))
    for candidate in candidates:
        try:
            yield json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass


def labeled_json(text: Any, label: str) -> Any:
    """Decode the explicitly labeled JSON value; do not search prose for schemas."""
    if not isinstance(text, str) or label not in text:
        return None
    try:
        return json.JSONDecoder().raw_decode(text.split(label, 1)[1].lstrip())[0]
    except (ValueError, TypeError):
        return None


def adapt_bfcl_schema(value: Any, *, annotations: list | None = None, path: str = "$") -> Any:
    """Adapt captured BFCL spellings; drop only provably redundant optional tags.

    JSON Schema's explicit object ``required`` list is the authority. A BFCL
    optional tag is removable only when that list proves the tagged property is
    optional. Contradictory or unknown annotations remain unsupported rather
    than being interpreted from descriptions, defaults or example arguments.
    """
    if isinstance(value, dict):
        result = {k: adapt_bfcl_schema(v, annotations=annotations, path=f"{path}.{k}") for k, v in value.items()}
        if isinstance(result.get("type"), str):
            result["type"] = {"dict": "object", "float": "number", "list": "array", "tuple": "array"}.get(result["type"], result["type"])
        properties, required = result.get("properties"), result.get("required")
        if (result.get("type") == "object" and isinstance(properties, dict)
                and isinstance(required, list) and all(isinstance(k, str) for k in required)
                and set(required).issubset(properties)):
            removed = []
            for name, parameter in properties.items():
                if not isinstance(parameter, dict) or name in required:
                    continue
                tag = parameter.get("optional")
                if tag is True or type(tag) is str and tag == "True":
                    removed.append({"path": f"{path}.properties.{name}.optional", "value": tag})
                    del parameter["optional"]
            tag = result.get("optional")
            if (isinstance(tag, list) and all(isinstance(k, str) for k in tag)
                    and set(tag).issubset(properties) and not set(tag).intersection(required)):
                removed.append({"path": f"{path}.optional", "value": tag})
                del result["optional"]
            if annotations is not None:
                annotations.extend({**item, "basis": "captured explicit required list excludes these declared properties",
                                    "required": list(required)} for item in removed)
        return result
    if isinstance(value, list):
        return [adapt_bfcl_schema(v, annotations=annotations, path=f"{path}[{index}]") for index, v in enumerate(value)]
    return value


def recover_bfcl(raw: dict) -> dict | None:
    """Select only schema-valid values already present in BFCL acceptable targets.

    Empty-string alternatives for optional parameters are represented by omission.
    Original alternatives and schemas are retained in the adaptation receipt.
    """
    captured = labeled_json(raw.get("instruction"), "Available tool schemas:")
    target = next((p for p in parsed_json_objects(raw.get("response")) if isinstance(p, dict) and p.get("answer_format") == "bfcl_ground_truth_json"), None)
    if not isinstance(captured, list) or not target:
        return None
    optional_annotations = []
    tools = [{"type": "function", "function": adapt_bfcl_schema(s, annotations=optional_annotations, path=f"$[{index}]")}
             for index, s in enumerate(captured) if isinstance(s, dict)]
    schemas = {d["function"].get("name"): d["function"].get("parameters", {}) for d in tools}
    calls = []
    selections = []
    for ci, expected in enumerate(target.get("expected_tool_calls", [])):
        if not isinstance(expected, dict) or len(expected) != 1:
            return None
        name, arguments = next(iter(expected.items()))
        if name not in schemas or not isinstance(arguments, dict):
            return None
        schema = schemas[name]
        result = {}
        for key, alternatives in arguments.items():
            if not isinstance(alternatives, list) or not alternatives:
                return None
            parameter = schema.get("properties", {}).get(key, {})
            if key not in schema.get("required", []) and "" in alternatives:
                selections.append({"call_index": ci, "parameter": key, "basis": "recorded optional empty alternative represented by omission"})
                continue
            valid = [v for v in alternatives if not check_arguments(v, parameter)]
            if not valid:
                return None
            result[key] = valid[0]
            selections.append({"call_index": ci, "parameter": key, "alternative_index": alternatives.index(valid[0])})
        if check_arguments(result, schema):
            return None
        calls.append({"type": "function", "function": {"name": name, "arguments": json.dumps(result, ensure_ascii=False, sort_keys=True)}})
    if not calls or target.get("call_count") != len(calls):
        return None
    return {"tools": tools, "messages": [{"role": "user", "content": raw["instruction"]}, {"role": "assistant", "content": None, "tool_calls": calls}],
            "receipt": {"basis": "captured labeled BFCL schemas and explicit acceptable ground-truth alternatives; no execution result asserted",
                        "original_schemas": captured, "original_acceptable_targets": target, "selections": selections,
                        "removed_optional_annotations": optional_annotations,
                        "type_aliases": {"dict": "object", "float": "number", "list": "array", "tuple": "array"}}}


def normalize_record(raw: dict, *, source_file: str = "", source_sha256: str = "",
                     source_verified: bool = False, collection_date: str = "") -> dict:
    """Losslessly adapt explicit legacy structures; absent IDs/schemas stay absent."""
    record = json.loads(json.dumps(raw))
    route_source = str(raw.get("route") or raw.get("raw_route") or "")
    record.update(schema=SCHEMA, source_route=route_source, route=ALIASES.get(route_source, route_source))
    record["id"] = str(raw.get("id") or raw.get("uid") or raw.get("rid") or raw.get("record_id") or digest(raw))
    provenance = dict(raw.get("provenance") or {})
    if raw.get("schema") != SCHEMA:
        provenance.setdefault("source", "curated_registry")
        provenance.setdefault("source_ids", [str(raw.get("rid") or raw.get("uid") or record["id"])])
        provenance.setdefault("source_verified", source_verified)
        provenance.setdefault("source_dates", [collection_date] if collection_date else [])
        provenance.setdefault("source_date_basis", "pinned_staging_collection_receipt; original author date not inferred")
        provenance.setdefault("group_id", raw.get("evidence_key") or raw.get("text_key") or record["id"])
    if source_file:
        provenance["normalized_from"] = {"path": source_file, "sha256": source_sha256, "record_id": record["id"]}
    record["provenance"] = provenance
    arithmetic_types = {"finance_calculation", "rate_problem", "weighted_score", "capacity_planning", "clinic_scheduling", "supply_planning", "compound_growth", "break_even", "working_capital", "inflation_adjustment", "weighted_average", "solution_calculation", "dilution"}
    if route_source == "math" and raw.get("task_type") in arithmetic_types:
        record["route"] = "text_arithmetic"
        provenance["route_assignment"] = {"basis": "explicit source arithmetic/calculation task_type", "task_type": raw["task_type"]}
    if raw.get("shard_mirror"):
        provenance["registry_origin"] = {key: raw.get(key) for key in ("rid", "shard_mirror", "shard_sha256", "line_number", "dataset_id", "source_dataset", "license", "license_tier", "task_type")}
    payloads = list(parsed_json_objects(raw.get("instruction"))) + list(parsed_json_objects(raw.get("response")))
    messages = raw.get("messages")
    tools = raw.get("tools") or []
    bfcl = recover_bfcl(raw)
    if bfcl and not messages:
        messages, tools = bfcl["messages"], bfcl["tools"]
        provenance["bfcl_schema_target_adaptation"] = bfcl["receipt"]
        record["structure_recovered_from"] = "explicit labeled BFCL schema JSON and acceptable ground-truth JSON"
    for payload in payloads:
        if isinstance(payload, dict):
            if not messages and isinstance(payload.get("messages"), list):
                messages = payload["messages"]
            if not tools and isinstance(payload.get("tools"), list):
                tools = payload["tools"]
            if not tools and isinstance(payload.get("functions"), list):
                tools = [{"type": "function", "function": definition} for definition in payload["functions"]]
    # Permit replay of earlier normalized BFCL rows through this same adapter.
    # Ordinary OpenAI/catalog schemas are not rewritten. Original captured
    # schemas and acceptable alternatives remain in the existing receipt.
    if raw.get("schema") == SCHEMA and isinstance(provenance.get("bfcl_schema_target_adaptation"), dict):
        annotations = []
        tools = adapt_bfcl_schema(tools, annotations=annotations, path="$.tools")
        if annotations:
            receipt = provenance["bfcl_schema_target_adaptation"]
            receipt["removed_optional_annotations"] = list(receipt.get("removed_optional_annotations", [])) + annotations
    if not messages:
        messages = []
        if isinstance(raw.get("instruction"), str) and raw["instruction"].strip():
            messages.append({"role": "user", "content": raw["instruction"]})
        structured = next((p for p in parsed_json_objects(raw.get("response")) if isinstance(p, dict) and p.get("tool_calls")), None)
        if structured:
            calls = []
            for original in structured["tool_calls"]:
                call = dict(original)
                if "function" not in call and "name" in call:
                    call = {k: v for k, v in call.items() if k not in {"name", "arguments"}}
                    call["function"] = {"name": original["name"], "arguments": original.get("arguments")}
                if isinstance(call.get("function", {}).get("arguments"), dict):
                    call["function"]["arguments"] = json.dumps(call["function"]["arguments"], ensure_ascii=False)
                call.setdefault("type", "function")
                calls.append(call)
            messages.append({"role": "assistant", "content": structured.get("content"), "tool_calls": calls})
            for result in structured.get("tool_results", []):
                message = {"role": "tool", "content": result.get("content"), "name": result.get("name") or result.get("tool")}
                if result.get("tool_call_id") or result.get("id"):
                    message["tool_call_id"] = result.get("tool_call_id") or result.get("id")
                if result.get("verifier"):
                    message["verifier"] = result["verifier"]
                messages.append(message)
            record["structure_recovered_from"] = "explicit response JSON; no identifiers or schemas invented"
        elif isinstance(raw.get("response"), str) and raw["response"].strip():
            messages.append({"role": "assistant", "content": raw["response"]})
    # IDs are transport identifiers, not supervision content. Preserve existing
    # IDs; derive stable IDs by explicit message/call positions. A result is
    # linked by name only when exactly one pending call has that name.
    # This never invents a schema, result, tool sequence, or parallel history.
    record["messages"] = messages
    canonicalize_tool_ids(record)
    record["tools"] = tools
    if bfcl and len(bfcl["messages"][-1]["tool_calls"]) > 1:
        record["route"] = "tool_parallel"
    record["media"] = raw.get("media") if isinstance(raw.get("media"), list) else [dict(a) for a in raw.get("sidecars", [])]
    supervision = dict(raw.get("supervision") or {})
    supervision.setdefault("message_mask", [m.get("role") == "assistant" for m in messages if isinstance(m, dict)])
    supplied_mask = supervision.get("message_mask")
    if isinstance(supplied_mask, list) and all(type(v) in (int, bool) and v in (0, 1) for v in supplied_mask):
        if any(type(v) is int for v in supplied_mask):
            provenance["supervision_normalization"] = "exact 0/1 upstream mask converted to canonical boolean assistant-only mask"
        supervision["message_mask"] = [bool(v) for v in supplied_mask]
    supervision.setdefault("policy", "assistant_only")
    supervision.setdefault("assistant_only", True)
    record["supervision"] = supervision
    record["split"] = raw.get("split") or provenance.get("split") or split_for(str(provenance.get("group_id") or record["id"]))
    return record


def canonicalize_tool_ids(record: dict) -> None:
    mappings = []
    pending = []
    messages = record.get("messages") or []
    if not isinstance(messages, list):
        return
    for mi, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                continue
            for ci, call in enumerate(calls):
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    continue
                function = call["function"]
                if isinstance(function.get("arguments"), dict):
                    function["arguments"] = json.dumps(function["arguments"], ensure_ascii=False, sort_keys=True)
                name = function.get("name")
                if not call.get("id") and name:
                    call["id"] = "derived_" + digest([record.get("id"), mi, ci, name])[:24]
                    mappings.append({"message_index": mi, "call_index": ci, "canonical_id": call["id"], "original_id": None, "basis": "explicit assistant message/call position; no original ID asserted"})
                pending.append(call)
        elif message.get("role") == "tool":
            identifier = message.get("tool_call_id")
            name = message.get("name")
            candidates = [c for c in pending if identifier and c.get("id") == identifier or not identifier and name and c.get("function", {}).get("name") == name]
            if len(candidates) == 1:
                call = candidates[0]
                if not identifier and call.get("id"):
                    message["tool_call_id"] = call["id"]
                    mappings.append({"message_index": mi, "canonical_id": call["id"], "original_id": None, "basis": "unique pending call with the explicitly recorded result tool name"})
                pending.remove(call)
        elif message.get("role") == "user":
            pending.clear()
    if mappings:
        record["provenance"]["derived_tool_identifier_mapping"] = mappings
        record["provenance"]["tool_identifier_authenticity"] = "canonical derived identifiers; original missing IDs are not claimed authentic"


def valid_date(value: Any) -> bool:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.year in (2025, 2026)
    except (ValueError, TypeError):
        return False


def check_arguments(value: Any, schema: dict, prefix: str = "arguments") -> list[str]:
    errors = []
    kinds = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool, "null": type(None)}
    kind = schema.get("type")
    if kind in kinds and (not isinstance(value, kinds[kind]) or kind in {"integer", "number"} and isinstance(value, bool)):
        return [prefix + ":type"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(prefix + ":enum")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(prefix + ":required:" + key)
        for key, child in value.items():
            if key in properties:
                errors.extend(check_arguments(child, properties[key], prefix + "." + key))
            elif schema.get("additionalProperties") is False:
                errors.append(prefix + ":additional:" + key)
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, child in enumerate(value):
            errors.extend(check_arguments(child, schema["items"], f"{prefix}[{index}]"))
    return errors


def supported_schema(schema: Any) -> bool:
    """Conservative executable subset; unsupported schemas remain quarantined."""
    if not isinstance(schema, dict):
        return False
    allowed = {"type", "properties", "required", "additionalProperties", "items", "enum", "description", "title", "default"}
    if set(schema) - allowed or "type" in schema and schema["type"] not in {"object", "array", "string", "integer", "number", "boolean", "null"}:
        return False
    if "properties" in schema and (not isinstance(schema["properties"], dict) or not all(supported_schema(v) for v in schema["properties"].values())):
        return False
    if "items" in schema and not supported_schema(schema["items"]):
        return False
    if "required" in schema and (not isinstance(schema["required"], list) or any(not isinstance(v, str) for v in schema["required"])):
        return False
    if isinstance(schema.get("additionalProperties"), dict):
        return False
    return True


def validate_record(record: dict, *, resolver: AssetResolver | None = None, eval_index: EvalIndex | None = None) -> dict:
    errors: set[str] = set()
    diagnostics: set[str] = set()
    route = record.get("route")
    capabilities = {route} if route in ROUTES else set()
    if route not in ROUTES:
        errors.add("unknown_route")
    provenance = record.get("provenance") or {}
    if provenance.get("source_verified") is not True:
        errors.add("source_not_verified")
    if not provenance.get("source") or not provenance.get("source_ids"):
        errors.add("source_identity_missing")
    dates = provenance.get("source_dates", [])
    if not dates or any(not valid_date(date) for date in dates):
        errors.add("source_dates_missing_or_outside_2025_2026")
    if record.get("split") not in {"train", "validation", "test"}:
        errors.add("invalid_split")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.add("messages_missing")
        messages = []
    mask = (record.get("supervision") or {}).get("message_mask")
    expected_mask = [int(isinstance(m, dict) and m.get("role") == "assistant") for m in messages]
    if not isinstance(mask, list) or len(mask) != len(messages) or any(type(v) not in (bool, int) or v not in (0, 1) for v in mask) or mask != expected_mask:
        errors.add("assistant_only_supervision_invalid")
    supervision = record.get("supervision") or {}
    if supervision.get("assistant_only") is not True and supervision.get("policy") != "assistant_only":
        errors.add("assistant_only_policy_missing")
    if not any(expected_mask):
        errors.add("assistant_target_missing")
    if route not in GENERATIVE_MEDIA and not any(isinstance(m, dict) and m.get("role") == "assistant" and (m.get("content") or m.get("tool_calls")) for m in messages):
        errors.add("assistant_target_content_missing")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        content = message["content"]
        if "CURRENT_TARGET:" in content and re.search(r"(?:teacher model producing|distillation target)", content, re.I):
            errors.add("teacher_rewrite_exposes_current_answer")
    schemas = {}
    for definition in record.get("tools", []):
        function = definition.get("function", {}) if isinstance(definition, dict) else {}
        if not isinstance(definition, dict) or not isinstance(function, dict) or definition.get("type") != "function" or not isinstance(function.get("parameters"), dict) or not function.get("name"):
            errors.add("tool_schema_incomplete")
        elif function["name"] in schemas:
            errors.add("tool_schema_duplicate_name")
        else:
            schemas[function["name"]] = function["parameters"]
            if not supported_schema(function["parameters"]):
                errors.add("tool_schema_unsupported_validation_subset")
    pending = {}
    call_ids = set()
    calls_count = results_count = 0
    failed_names = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            errors.add("message_role_invalid")
            continue
        if message["role"] == "assistant":
            if pending:
                errors.add("unresolved_call_before_assistant_turn")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                errors.add("tool_calls_not_array")
                continue
            if len(calls) > 1:
                capabilities.add("tool_parallel")
            for call in calls:
                calls_count += 1
                capabilities.add("tool_call")
                if not isinstance(call, dict):
                    errors.add("tool_call_invalid")
                    continue
                identifier = call.get("id")
                function = call.get("function") or {}
                if not isinstance(function, dict):
                    errors.add("tool_call_invalid")
                    continue
                name = function.get("name")
                if not identifier:
                    errors.add("tool_call_id_missing")
                elif identifier in call_ids:
                    errors.add("tool_call_id_duplicate")
                else:
                    call_ids.add(identifier)
                    pending[identifier] = name
                if call.get("type") != "function" or name not in schemas:
                    errors.add("authentic_tool_schema_missing")
                try:
                    arguments = json.loads(function.get("arguments"))
                except (TypeError, ValueError):
                    errors.add("tool_arguments_not_json_string")
                else:
                    if name in schemas and check_arguments(arguments, schemas[name]):
                        errors.add("tool_arguments_schema_mismatch")
                if name in failed_names:
                    capabilities.add("tool_retry")
        elif message["role"] == "tool":
            results_count += 1
            capabilities.add("tool_result")
            identifier = message.get("tool_call_id")
            if not identifier or identifier not in pending:
                errors.add("tool_result_call_link_missing")
            else:
                name = pending.pop(identifier)
                if message.get("name") and message["name"] != name:
                    errors.add("tool_result_name_mismatch")
                if message.get("is_error") is True or re.search(r"\b(error|failed|exception|timeout)\b", str(message.get("content") or ""), re.I):
                    failed_names.add(name)
        elif pending and message["role"] == "user":
            errors.add("unresolved_call_before_user_turn")
    if schemas:
        capabilities.add("tool_schema")
    if calls_count > 1 and sum(m.get("role") == "assistant" for m in messages if isinstance(m, dict)) > 1:
        capabilities.add("tool_multiturn")
    if sum(m.get("role") == "user" for m in messages if isinstance(m, dict)) > 1:
        capabilities.add("text_multiturn")
    if pending:
        diagnostics.add("terminal_open_tool_calls")
        if route not in {"tool_call", "tool_parallel", "tool_schema"}:
            errors.add("execution_history_incomplete")
    if route and route.startswith("tool_") and not calls_count and route != "tool_schema":
        errors.add("tool_route_has_no_structured_calls")
    if route == "tool_schema" and not schemas:
        errors.add("authentic_tool_schema_missing")
    if (record.get("quality") or {}).get("schema_complete") is False:
        errors.add("source_marks_schema_incomplete")
    assets = record.get("media") or []
    if route in MEDIA_ROUTES and not assets:
        errors.add("media_assets_missing")
    if route in GENERATIVE_MEDIA and not any(a.get("role") == "target" for a in assets if isinstance(a, dict)):
        errors.add("media_target_missing")
    if route in {"vision_grounding", "video_understand", "audio_understand", "image_edit"} and not any(a.get("role") in {"input", "conditioning"} for a in assets if isinstance(a, dict)):
        errors.add("media_input_missing")
    target_hashes = {a.get("sha256") for a in assets if isinstance(a, dict) and a.get("role") == "target" and a.get("sha256")}
    target_paths = {a.get("path") for a in assets if isinstance(a, dict) and a.get("role") == "target" and a.get("path")}
    asset_reports = []
    for asset in assets:
        if not isinstance(asset, dict):
            errors.add("media_asset_invalid")
            continue
        if asset.get("role") not in {"input", "conditioning", "target"} or not asset.get("type"):
            errors.add("media_semantic_role_missing")
        if asset.get("role") in {"input", "conditioning"}:
            if asset.get("sha256") in target_hashes or asset.get("path") in target_paths:
                errors.add("target_in_input_leakage")
            if route in GENERATIVE_MEDIA and asset.get("representation") == "latent":
                sigma = asset.get("noise_level")
                if type(sigma) not in (float, int) or not math.isfinite(sigma) or sigma <= 0:
                    errors.add("clean_latent_conditioning_leakage")
        if asset.get("role") == "target":
            teacher = asset.get("teacher") or {}
            if not teacher.get("model_id") or not SHA_RE.fullmatch(str(teacher.get("checkpoint_sha256") or "").lower()) or teacher.get("source_verified") is not True:
                errors.add("media_teacher_provenance_missing")
        if resolver is None:
            errors.add("media_assets_not_verified")
        else:
            inspected = resolver.inspect(asset)
            asset_reports.append(inspected)
            errors.update(inspected["errors"])
    if route == "joint_av":
        types = {a.get("type") for a in assets if isinstance(a, dict) and a.get("role") == "target"}
        if not {"audio", "video"}.issubset(types):
            errors.add("joint_av_targets_incomplete")
        av = record.get("av_alignment") or {}
        if not av.get("source_verified") or not av.get("duration_seconds") or not av.get("fps") or not av.get("sample_rate"):
            errors.add("joint_av_alignment_missing")
        origins = {a.get("source_media_sha256") for a in assets if isinstance(a, dict) and a.get("role") == "target"}
        offsets = {a.get("start_seconds") for a in assets if isinstance(a, dict) and a.get("role") == "target"}
        if len(origins) != 1 or None in origins or not SHA_RE.fullmatch(str(next(iter(origins), ""))) or len(offsets) != 1 or None in offsets or av.get("same_origin") is not True:
            errors.add("joint_av_same_origin_or_timestamp_missing")
        if av.get("pairing") == "unrelated" or any("not temporally related" in str(a.get("pairing") or "") for a in assets if isinstance(a, dict)):
            errors.add("joint_av_unrelated_pairing")
    if route == "volume_generate" and not any(a.get("type") in {"volume", "geometry", "mesh"} for a in assets if isinstance(a, dict) and a.get("role") == "target"):
        errors.add("genuine_3d_target_missing")
    if eval_index:
        hits = eval_index.hits(record)
        if hits:
            errors.update(hits)
    if record.get("source_route") != route:
        diagnostics.add("route_alias_preserved")
    return {"id": record.get("id"), "route": route, "split": record.get("split"), "train_ready": not errors,
            "errors": sorted(errors), "diagnostics": sorted(diagnostics), "capabilities": sorted(capabilities),
            "tool_calls": calls_count, "tool_results": results_count, "media_assets": asset_reports,
            "readiness_scope": "corpus structural/provenance readiness only; no semantic grading or runtime/trainer proof"}


def iter_jsonl(path: Path, max_rows: int = 100000, max_row_bytes: int = 4 * 1024 ** 2):
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number > max_rows:
                raise ValueError(f"row limit {max_rows} exceeded: {path}")
            if len(line) > max_row_bytes:
                raise ValueError(f"row {line_number} exceeds byte limit: {path}")
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"row {line_number} is not an object: {path}")
                yield row


def inventory_manifests(corpus_root: Path, resolver: AssetResolver) -> list[dict]:
    inventory = []
    for path in sorted(corpus_root.rglob("manifest.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        references = []
        def visit(value):
            if isinstance(value, dict):
                if isinstance(value.get("path"), str) and SHA_RE.fullmatch(str(value.get("sha256") or "").lower()):
                    references.append(resolver.inspect(value))
                else:
                    for child in value.values():
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        for key in ("shards", "files", "file", "items_file", "clip_files", "audio_files", "vision_in", "video_in"):
            visit(raw.get(key))
        counts = {key: raw[key] for key in ("rows", "documents", "items", "tokens", "real_tokens", "rows_by_route", "documents_by_route", "tokens_by_route", "summary") if key in raw}
        inventory.append({"path": str(path.resolve()), "sha256": file_sha256(path), "schema": raw.get("schema"),
                          "declared_counts": counts, "count_evidence": "manifest claims; payload validation reported separately",
                          "references": references, "verified_references": sum(r["verified"] for r in references)})
    return inventory


class GroupIndex:
    """Union source events, media hashes and content duplicates before assigning splits."""
    def __init__(self):
        self.parents: list[int] = []
        self.by_key: dict[str, int] = {}
        self.explicit: list[set[str]] = []

    def find(self, item):
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = self.parents[item]
        return item

    def add(self, keys: set[str], explicit_split: str | None) -> int:
        item = len(self.parents)
        self.parents.append(item)
        self.explicit.append({explicit_split} if explicit_split else set())
        for key in keys:
            if key in self.by_key:
                previous = self.find(self.by_key[key])
                current = self.find(item)
                if previous != current:
                    self.parents[current] = previous
                    self.explicit[previous].update(self.explicit[current])
            self.by_key[key] = item
        return item


def pinned_carve(path: Path, source_receipts: list[dict], allocation_body: dict | None = None) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in raw.items() if k != "carve_sha256"}
    actual = hashlib.sha256(json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()
    if actual != raw.get("carve_sha256"):
        raise ValueError("carve canonical seal mismatch")
    if not any(r["sha256"] == raw.get("rows_sha256") for r in source_receipts):
        raise ValueError("carve is not bound to an input staging file")
    if allocation_body and allocation_body.get("carve_sha256") != actual:
        raise ValueError("allocation and carve seals differ")
    if not isinstance(raw.get("assignment"), dict):
        raise ValueError("carve assignment missing")
    return raw


def prepare_av_pair(*, identifier: str, prompt: str, video: dict, audio: dict,
                    provenance: dict, duration_seconds: float, fps: float, sample_rate: int) -> dict:
    """Prepare an already-extracted real pair; validation checks lineage and files.

    Both target assets must carry the same source_media_sha256 and start_seconds.
    No unrelated audio is substituted and no synthetic synchronization is claimed.
    """
    assets = [{**video, "role": "target", "type": "video"}, {**audio, "role": "target", "type": "audio"}]
    origins = {a.get("source_media_sha256") for a in assets}
    offsets = {a.get("start_seconds") for a in assets}
    same = len(origins) == len(offsets) == 1 and None not in origins and None not in offsets
    row = {"schema": SCHEMA, "id": identifier, "route": "joint_av", "split": provenance.get("split", "train"),
           "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": None}], "tools": [],
           "media": assets, "supervision": {"assistant_only": True, "policy": "assistant_only", "message_mask": [False, True]},
           "provenance": provenance, "av_alignment": {"source_verified": provenance.get("source_verified") is True,
           "same_origin": same, "duration_seconds": duration_seconds, "fps": fps, "sample_rate": sample_rate,
           "pairing": "same-origin interval required; validated before use"}}
    return row


def normalize_media_item(kind: str, item: dict, receipt: dict, collection_date: str) -> dict:
    route = {"ace": "music_generate", "qimg": "image_generate", "h3": "joint_av", "hy3d": "volume_generate", "vit": "vision_grounding", "vision_in": "vision_grounding", "video_in": "video_understand"}[kind]
    prompt = item.get("prompt") or item.get("caption") or item.get("instruction")
    editing = kind == "vision_in" and item.get("task_type") == "local_image_editing_pair" and any(a.get("role") == "target" for a in item.get("media", []) if isinstance(a, dict))
    if editing:
        route = "image_edit"
    response = item.get("reference_response") if route in {"vision_grounding", "video_understand"} else None
    messages = ([{"role": "user", "content": prompt}] if prompt else []) + [{"role": "assistant", "content": response}]
    assets = []
    def asset(value, role, media_type):
        if not isinstance(value, dict):
            return
        assets.append({"role": role, "type": media_type, "representation": "raw", "path": value.get("stored_path") or value.get("path"),
                       "sha256": value.get("sha256"), "source_path": value.get("path"), "pairing": value.get("pairing"),
                       "source_media_sha256": (item.get("source") or {}).get("sha256") if media_type == "video" else value.get("source_media_sha256"),
                       "start_seconds": item.get("start") if media_type == "video" else value.get("start"), "probe": value.get("probe"),
                       "teacher": value.get("teacher") or {}, "shape": value.get("shape")})
    if kind in {"qimg", "hy3d", "vit"}:
        asset(item.get("image"), "input" if kind == "vit" else "target", "image")
    elif kind == "ace":
        asset(item.get("clip"), "target", "audio")
    elif kind == "h3":
        asset(item.get("video"), "target", "video")
        asset(item.get("audio"), "target", "audio")
    else:
        for value in item.get("media", []):
            asset(value, "target" if editing and value.get("role") == "target" else "input", value.get("media") or "image")
    provenance = {"source": "curated_registry", "source_ids": [str(item.get("rid") or item.get("uid"))],
                  "source_verified": receipt["source_verified"], "source_dates": [collection_date] if collection_date else [],
                  "source_date_basis": "collection receipt; original author date unknown", "group_id": item.get("rid") or item.get("uid"),
                  "normalized_from": {"path": receipt["path"], "sha256": receipt["sha256"], "record_id": item.get("rid")},
                  "media_metadata_only": True, "original_metadata": item}
    row = {"schema": SCHEMA, "id": f"{kind}:{item.get('rid') or item.get('uid')}", "route": route, "source_route": kind,
           "split": "train", "messages": messages, "tools": [], "media": assets,
           "supervision": {"assistant_only": True, "policy": "assistant_only", "message_mask": [False] * (len(messages) - 1) + [True]},
           "provenance": provenance}
    if kind == "h3":
        row["av_alignment"] = {"source_verified": False, "same_origin": False, "pairing": "unrelated"}
    return row

def build_corpus(*, corpus_root: Path, record_paths: list[Path], output: Path, path_maps=(),
                 expected_hashes: dict[str, str] | None = None, collection_date: str = "", allocation: Path | None = None,
                 eval_paths: list[Path] | None = None, carve: Path | None = None, media_metadata: dict[str, Path] | None = None,
                 max_rows: int = 100000, max_asset_bytes: int = 1024 ** 3) -> dict:
    """Two streaming passes: resolve groups then write normalized rows and diagnostics."""
    resolver = AssetResolver(corpus_root, path_maps, max_asset_bytes)
    expected_hashes = expected_hashes or {}
    source_receipts = []
    for path in record_paths:
        actual = file_sha256(path)
        expected = expected_hashes.get(str(path))
        if expected and actual != expected:
            raise ValueError(f"source hash mismatch: {path}")
        source_receipts.append({"path": str(path.resolve()), "sha256": actual, "source_verified": bool(expected)})
    allowlist = None
    allocation_sha = None
    a = None
    if allocation:
        a = json.loads(allocation.read_text(encoding="utf-8"))
        for receipt in source_receipts:
            if a.get("rows_sha256") == receipt["sha256"]:
                receipt["source_verified"] = True
        allowlist = set(a.get("text", {}).get("docs", []))
        if not allowlist or any(not isinstance(rid, str) for rid in allowlist):
            raise ValueError("allocation text.docs must be a nonempty source-row allowlist")
        allocation_sha = file_sha256(allocation)
    carved = pinned_carve(carve, source_receipts, a) if carve else None
    assignment = carved["assignment"] if carved else {}
    evaluation = EvalIndex()
    for path in eval_paths or []:
        if path.suffix == ".jsonl":
            for raw in iter_jsonl(path, max_rows):
                evaluation.add(normalize_record(raw))
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))
            evaluation.add(raw if isinstance(raw, dict) else {"records": raw})
    groups = GroupIndex()
    profiles = []
    def records():
        for source_path, receipt in zip(record_paths, source_receipts):
            for raw in iter_jsonl(source_path, max_rows):
                record = normalize_record(raw, source_file=receipt["path"], source_sha256=receipt["sha256"],
                                          source_verified=receipt["source_verified"], collection_date=collection_date)
                yield raw, record
    for raw, record in records():
        keys = identity_keys(record)
        content = digest({"messages": record.get("messages"), "tools": record.get("tools"), "media": record.get("media")})
        keys.add("normalized-content:" + content)
        explicit = raw.get("split") or (raw.get("provenance") or {}).get("split")
        group = groups.add(keys, explicit)
        eligible = raw.get("schema") == SCHEMA or allowlist is not None and raw.get("rid") in allowlist
        if carved and raw.get("schema") != SCHEMA:
            eligible = eligible and assignment.get(raw.get("rid")) == "train"
        profiles.append({"group": group, "content": content, "eligible": eligible})
    roots = defaultdict(list)
    for index, profile in enumerate(profiles):
        roots[groups.find(profile["group"])].append(index)
    root_split = {}
    root_identity = {}
    for root, indices in roots.items():
        root_identity[root] = min(profiles[i]["content"] for i in indices)
        explicit = groups.explicit[groups.find(root)]
        root_split[root] = next(iter(explicit)) if len(explicit) == 1 else split_for(root_identity[root])
    output.mkdir(parents=True, exist_ok=True)
    all_path, ready_path, diagnostic_path = output / "normalized.jsonl", output / "trainable.jsonl", output / "diagnostics.jsonl"
    counts = Counter({key: 0 for key in ("rows", "ready", "blocked", "ready_train", "ready_validation", "ready_test")})
    error_counts = Counter()
    coverage = {route: Counter(present=0, ready=0, blocked=0) for route in ROUTES}
    missing_structured_sources = Counter()
    representatives = {}
    for index, profile in enumerate(profiles):
        if profile["eligible"]:
            representatives.setdefault(profile["content"], index)
    split_paths = {split: output / (split + ".jsonl") for split in ("train", "validation", "test")}
    split_streams = {split: path.open("w", encoding="utf-8", newline="\n") for split, path in split_paths.items()}
    with all_path.open("w", encoding="utf-8", newline="\n") as all_stream, ready_path.open("w", encoding="utf-8", newline="\n") as ready_stream, diagnostic_path.open("w", encoding="utf-8", newline="\n") as diag_stream:
        for index, (raw, record) in enumerate(records()):
            profile = profiles[index]
            root = groups.find(profile["group"])
            record["split"] = root_split[root]
            record["provenance"]["recovery_group_id"] = root_identity[root]
            record["provenance"]["split"] = record["split"]
            assessment = validate_record(record, resolver=resolver, eval_index=evaluation)
            errors = set(assessment["errors"])
            if len(groups.explicit[root]) > 1:
                errors.add("source_group_crosses_existing_splits")
            if allowlist is not None and raw.get("schema") != SCHEMA and raw.get("rid") not in allowlist:
                errors.add("outside_pinned_screened_text_allocation")
            elif allowlist is None and raw.get("schema") != SCHEMA:
                errors.add("legacy_contamination_screen_not_bound")
            if carved and raw.get("schema") != SCHEMA and assignment.get(raw.get("rid")) != "train":
                errors.add("frozen_carve_" + str(assignment.get(raw.get("rid"), "assignment_missing")))
            if representatives.get(profile["content"], index) != index:
                errors.add("duplicate_content")
            assessment["errors"] = sorted(errors)
            assessment["train_ready"] = not errors
            record["quality"] = {**(record.get("quality") or {}), "train_ready": not errors,
                                 "schema_complete": not any("tool_" in error for error in errors), "blockers": sorted(errors)}
            record["coverage"] = assessment["capabilities"]
            # Preserve only normalized semantic/provenance fields, not flattened response copies.
            normalized = {key: record[key] for key in ("schema", "id", "route", "source_route", "split", "messages", "tools", "media", "supervision", "provenance", "quality", "coverage")}
            for key in ("av_alignment", "structure_recovered_from"):
                if key in record:
                    normalized[key] = record[key]
            line = canonical(normalized).decode() + "\n"
            all_stream.write(line)
            if not errors:
                ready_stream.write(line)
                split_streams[record["split"]].write(line)
            diag_stream.write(canonical(assessment).decode() + "\n")
            counts["rows"] += 1
            counts["ready" if not errors else "blocked"] += 1
            if not errors:
                counts["ready_" + record["split"]] += 1
            error_counts.update(errors)
            for capability in assessment["capabilities"]:
                coverage[capability]["present"] += 1
                coverage[capability]["ready" if not errors else "blocked"] += 1
            if errors.intersection({"authentic_tool_schema_missing", "tool_call_id_missing", "tool_route_has_no_structured_calls"}) and raw.get("shard_mirror"):
                missing_structured_sources[raw["shard_mirror"]] += 1
    for stream in split_streams.values():
        stream.close()
    media_report = []
    media_path = output / "blocked_media.jsonl"
    with media_path.open("w", encoding="utf-8", newline="\n") as media_stream:
        for kind, path in (media_metadata or {}).items():
            actual = file_sha256(path)
            expected = expected_hashes.get(str(path))
            if expected and actual != expected:
                raise ValueError(f"media metadata hash mismatch: {path}")
            receipt = {"path": str(path.resolve()), "sha256": actual, "source_verified": bool(expected)}
            errors_by_kind = Counter()
            count = 0
            for item in iter_jsonl(path, max_rows):
                record = normalize_media_item(kind, item, receipt, collection_date)
                assessment = validate_record(record, resolver=resolver, eval_index=evaluation)
                errors = set(assessment["errors"])
                if carved and assignment.get(item.get("rid")) != "train":
                    errors.add("frozen_carve_" + str(assignment.get(item.get("rid"), "assignment_missing")))
                # Metadata is not a training target/cache. Keep these separate
                # until a teacher materializes the representation and provenance.
                errors.add("media_teacher_cache_not_materialized")
                record["quality"] = {"train_ready": False, "blockers": sorted(errors)}
                media_stream.write(canonical(record).decode() + "\n")
                errors_by_kind.update(errors)
                count += 1
            media_report.append({"kind": kind, "rows": count, "receipt": receipt, "train_ready": 0, "blockers": dict(errors_by_kind)})
    manifest = {"schema": "recovery_corpus.manifest.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "normalizer": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__))},
                "corpus_schema": SCHEMA, "readiness_scope": "corpus structure/provenance only, not correctness or trainer readiness",
                "sources": source_receipts, "collection_date_basis": collection_date, "allocation_sha256": allocation_sha,
                "carve_sha256": carved.get("carve_sha256") if carved else None,
                "normalized_eval_screen": "exact complete task; >50% shared 13-grams excluding shared tool catalogs; source/media identity",
                "eval_sources": [{"path": str(p.resolve()), "sha256": file_sha256(p)} for p in eval_paths or []],
                "counts": dict(counts), "coverage": {k: dict(v) for k, v in coverage.items()}, "blockers": dict(error_counts),
                "missing_coverage": [r for r, c in coverage.items() if not c["ready"]],
                "group_split": "source-event/media/content connected components, existing split conflicts blocked, salted hash80/10/10 for unassigned groups",
                "tool_recovery_sources": [{"path": p, "rows": n} for p, n in missing_structured_sources.most_common()],
                "outputs": {"normalized": {"path": str(all_path.resolve()), "sha256": file_sha256(all_path)},
                            "trainable": {"path": str(ready_path.resolve()), "sha256": file_sha256(ready_path)},
                            "diagnostics": {"path": str(diagnostic_path.resolve()), "sha256": file_sha256(diagnostic_path)}},
                "split_outputs": {s: {"path": str(p.resolve()), "sha256": file_sha256(p)} for s, p in split_paths.items()},
                "blocked_media": {"path": str(media_path.resolve()), "sha256": file_sha256(media_path), "inventory": media_report},
                "manifest_inventory": inventory_manifests(corpus_root, resolver),
                "limitations": ["No model/GPU/tools executed; source responses are ungraded", "Media targets require verified real files and teacher provenance", "Original author dates are not inferred from collection dates", "Blocked rows remain in normalized/diagnostics and never enter trainable output"]}
    (output / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--records", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocation", type=Path)
    parser.add_argument("--carve", type=Path)
    parser.add_argument("--media-metadata", action="append", default=[], metavar="KIND=PATH")
    parser.add_argument("--eval-records", type=Path, action="append", default=[])
    parser.add_argument("--source-sha256", action="append", default=[], metavar="PATH=SHA256")
    parser.add_argument("--path-map", action="append", default=[], metavar="FROM=TO")
    parser.add_argument("--collection-date", default="")
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--max-asset-bytes", type=int, default=1024 ** 3)
    args = parser.parse_args(argv)
    if args.output.resolve() == args.corpus_root.resolve() or args.corpus_root.resolve() in args.output.resolve().parents:
        parser.error("output must be outside the original corpus tree")
    expected = dict(value.rsplit("=", 1) for value in args.source_sha256)
    maps = [tuple(value.split("=", 1)) for value in args.path_map]
    report = build_corpus(corpus_root=args.corpus_root, record_paths=args.records, output=args.output, path_maps=maps,
                          expected_hashes=expected, collection_date=args.collection_date, allocation=args.allocation,
                          eval_paths=args.eval_records, carve=args.carve, media_metadata={k: Path(p) for k, p in (v.split("=", 1) for v in args.media_metadata)},
                          max_rows=args.max_rows, max_asset_bytes=args.max_asset_bytes)
    print(json.dumps({"manifest": str((args.output / "manifest.json").resolve()), "counts": report["counts"],
                      "coverage": report["coverage"], "blockers": report["blockers"], "missing_coverage": report["missing_coverage"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
